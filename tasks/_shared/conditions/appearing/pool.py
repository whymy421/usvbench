"""Appearing-suite admission pools: admitted keys, explicit not-admitted keys, file codec (G3 rule 6).

An ``AppearingPool`` holds, per offered scenario key (eval seed, env, episode,
hazard group ``g3_appearing_<task>``), either the admitted placement with its
witnessed doses config.WITNESSED_DOSES (D1..D4; + the supplementary D5 under options M and E;
D0 = the same placement, never appearing) or an
explicit not-admitted record with every rejection reason (CONTEXT_R4 D16: the
key keeps its E5 layout and runs the appearing suite at D0; never a redraw).
A key with the pool's seed and group, env inside the grid and episode index >=
the grid's episodes is a not-admitted key with the single reason
"beyond_offered_grid" (CONTEXT_R4 D24); keys outside the env range or with
another seed or group raise KeyError (preflight errors).

Header counts per dose are [offered, admitted, not_admitted] rows, one per dose
D0..D_max (CONTEXT_R4 D22; joint admission, so every dose row is the same). The pool digest seals
the offered grid, every placement digest, every admitted (dose, candidate
digest, witness digest) and every not-admitted reason list.

G20d dose policy per (task, vessel) (config.offered_doses / required_doses / optional_doses):
  * a dose NOT APPLICABLE BY DESIGN (ladder.NOT_APPLICABLE_DOSES; BlueBoat station keeping D1) is never offered:
    its count row is [0, 0, 0], the header names it (dose_applicability, not_applicable_doses) and assignment()
    REFUSES it (ValueError: a preflight error at the env's first reset, never a record) -- distinct from a
    not-admitted key, which runs D0 and is counted;
  * an OPTIONAL dose (decision 5B: docking D5) is admitted per key: a key admitted for the required doses whose
    optional dose has no witness keeps that reason in optional_reasons and is NOT ADMITTED AT THAT DOSE ONLY (its
    count row then differs from the main rows);
  * every other pool is jointly admitted over all witnessed doses exactly as before.

Pool files: deterministic xz-compressed canonical JSON; witness trajectories
are packed little-endian float64 columns (base64). The env loads only files
pinned by SHA-256 in the family folder's MANIFEST.json (D19/D23); the loader
recomputes every digest and count and fails closed on any difference.

Where the files are read (G3c, controller option B, mirroring the D7 families):
a release's code tree keeps only .cfg/.json/.py/.toml/.yaml/.yml files, so the
.json.xz pools travel in the flat pool bundle beside the release. The loader
reads the pool FILE from the runtime pool directory when one is set --
cfg.journal_d7_pool_dir (the D19 runtime --set field of the HazardNav family),
else the environment variable USVBENCH_D7_POOL_DIR (the bundle folder) -- and
from tasks/<family>/appearing_pools only when no runtime directory is set (CPU
tests). The in-repo MANIFEST.json (a code-tree member, so it is in every
release) stays the authority: the file's size and SHA-256 must match its pin
wherever it is found; a missing or mismatching file fails closed (never a
silent fallback to another folder). The record names the source used.
"""
import base64
from dataclasses import dataclass
import hashlib
import json
import lzma
import os
from pathlib import Path
import struct

from ..dynamic import adapters
from ..dynamic.feasibility import WitnessResult, replay_witness
from ..dynamic.model import EpisodeKey, Pose, TimedState
from . import config
from .stamp import BEYOND_OFFERED_GRID, JOINT_ECHO, NOT_ADMITTED, ADMITTED, canonical, digest

POOL_SCHEMA = "usvbench.appearing_pool/1"
MANIFEST_SCHEMA = "usvbench.appearing_pool_manifest/1"
ASSIGNMENT_SCHEMA = "usvbench.appearing_assignment/1"
NOT_ADMITTED_SCHEMA = "usvbench.appearing_not_admitted/1"
POOL_DIRNAME = "appearing_pools"
MANIFEST_NAME = "MANIFEST.json"
POOL_DIR_ENV = "USVBENCH_D7_POOL_DIR"          # runtime pool directory (the pool bundle), as for the D7 families
POOL_DIR_FIELD = "journal_d7_pool_dir"        # cfg runtime field (cfg_override.JOURNAL_RUNTIME_FIELDS)
D16_RULE = ("CONTEXT_R4 D16 (appearing suite): a key the appearing admission does not admit keeps its E5 layout "
            "and runs the appearing suite at D0 (no cylinder); stamp exposure 'not_admitted' with the reasons, "
            "assigned false, itt_count 0 (D27); never a redraw, never a skipped scenario")
D24_RULE = ("CONTEXT_R4 D24 (appearing suite): a key with the pool's eval seed and hazard group, env inside the "
            "offered grid and episode index >= the grid's episodes is not admitted with the single reason "
            "'beyond_offered_grid': it keeps its E5 layout, runs the appearing suite at D0 and is never counted")
_STATE = struct.Struct("<7d")


def pool_file_name(task, vessel_id, seed=None):
    """Pool file name: the G3 legacy form without a seed, or the seed-qualified G20 form."""
    if seed is None:
        return "%s__%s.json.xz" % (task, vessel_id)
    return "%s__%s__s%d.json.xz" % (task, vessel_id, int(seed))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _key_tuple(key):
    return [int(key.eval_seed), int(key.env), int(key.episode), str(key.hazard_group)]


def _key_document(key):
    return {"eval_seed": int(key.eval_seed), "env": int(key.env), "episode": int(key.episode),
            "hazard_group": str(key.hazard_group)}


def witness_digest(result):
    """The E4 witness digest (adapters._witness_digest) of a WitnessResult."""
    return adapters._witness_digest(result)


def encode_trajectory(trajectory):
    raw = b"".join(_STATE.pack(s.time_s, s.pose.x, s.pose.y, s.pose.yaw, s.pose.speed, s.accel_mps2,
                               s.yaw_rate_rps) for s in trajectory)
    return base64.b64encode(raw).decode("ascii")


def decode_trajectory(text, count):
    raw = base64.b64decode(text.encode("ascii"), validate=True)
    if len(raw) != count * _STATE.size:
        raise ValueError("witness trajectory block has the wrong length")
    out = []
    for values in _STATE.iter_unpack(raw):
        t, x, y, yaw, speed, accel, rate = values
        out.append(TimedState(t, Pose(x, y, yaw, speed), accel, rate))
    return tuple(out)


# ------------------------------------------------------------------ assignments --
@dataclass(frozen=True)
class AppearingAssignment:
    """An ADMITTED key at one dose (D0..D4; D0 never appears)."""

    key: EpisodeKey
    task: str
    vessel_id: str
    geometry_id: str
    dose: int
    placement: dict
    candidate: object            # dose-specific candidate document (None at D0)
    witness_reason: object
    witness_min_clearance_m: object
    witness_digest: object
    position_digest: str
    candidate_digest: str
    pool_digest: str
    exposure = ADMITTED
    beyond_offered_grid = False

    @property
    def cylinder(self):
        c = self.placement["cylinder"]
        return float(c["x_m"]), float(c["y_m"]), float(c["radius_m"])

    @property
    def trigger_gap_m(self):
        return None if self.candidate is None else float(self.candidate["trigger_gap_m"])

    @property
    def appears(self):
        return self.dose > 0

    def summary(self, slot):
        """JSON-native stamp ``assignment`` block."""
        c = self.candidate or {}
        x, y, r = self.cylinder
        return {"exposure": ADMITTED, "key": _key_document(self.key), "task": self.task, "dose": self.dose,
                "vessel_id": self.vessel_id, "geometry_id": self.geometry_id, "slot": int(slot),
                "cylinder": {"x_m": x, "y_m": y, "radius_m": r},
                "site": self.placement["site"], "radius_rule": self.placement["radius_rule"],
                "gap_ratio": config.gap_ratio(self.dose), "trigger_gap_m": self.trigger_gap_m,
                "k_equivalent": (None if self.trigger_gap_m is None or c.get("t_avoid_s") is None
                                 else config.k_equivalent(self.trigger_gap_m, c["t_avoid_s"])),
                "ladder_option": config.LADDER_OPTION,
                "reference_speed_mps": config.REFERENCE_SPEED_MPS, "t_avoid_s": c.get("t_avoid_s"),
                "nominal_trigger_s": c.get("nominal_trigger_s"),
                "nominal_trigger_gap_m": c.get("nominal_trigger_gap_m"),
                "nominal_trigger_bearing_rad": c.get("nominal_trigger_bearing_rad"),
                # G20d (decision 7A): pool-time nominal visibility of this dose (None at D0 / trigger not reached)
                "nominal_visible_at_trigger": c.get("nominal_visible_at_trigger"),
                "nominal_visible_rays": c.get("nominal_visible_rays"),
                "witness_reason": self.witness_reason, "witness_min_clearance_m": self.witness_min_clearance_m,
                "witness_digest": self.witness_digest, "position_digest": self.position_digest,
                "candidate_digest": self.candidate_digest, "pool_digest": self.pool_digest}


def assignment_candidate_digest(position_digest, dose, candidate, witness_digest_value):
    return digest({"schema": ASSIGNMENT_SCHEMA, "position_digest": position_digest, "dose": int(dose),
                   "candidate": candidate, "witness_digest": witness_digest_value})


def not_admitted_digest(key, task, vessel_id, geometry_id, reasons):
    """Dose-independent identity of a not-admitted key (D20: same hash at every dose of the key)."""
    return digest({"schema": NOT_ADMITTED_SCHEMA, "key": _key_tuple(key), "task": task, "vessel_id": vessel_id,
                   "geometry_id": geometry_id, "reasons": [[int(d), str(r)] for d, r in reasons]})


@dataclass(frozen=True)
class NotAdmittedAppearing:
    """A key the appearing admission did not admit (D16) or a key beyond the grid (D24): runs D0."""

    key: EpisodeKey
    task: str
    vessel_id: str
    geometry_id: object          # layout geometry of an in-grid key; None beyond the grid
    reasons: tuple               # ((dose, reason), ...); dose 0 = dose-independent placement reason
    requested_dose: int
    pool_digest: str
    dose = 0
    exposure = NOT_ADMITTED
    candidate = None
    appears = False

    @property
    def beyond_offered_grid(self):
        return tuple(self.reasons) == ((0, BEYOND_OFFERED_GRID),)

    @property
    def position_digest(self):
        return not_admitted_digest(self.key, self.task, self.vessel_id, self.geometry_id, self.reasons)

    @property
    def candidate_digest(self):
        return self.position_digest

    def reason_strings(self):
        """D20 (c) normalised reasons: strings, the joint echo dropped, de-duplicated, first-seen order."""
        out = []
        for _, reason in self.reasons:
            if reason != JOINT_ECHO and reason not in out:
                out.append(reason)
        if not out:
            raise ValueError("a not-admitted appearing key needs a reason other than the joint echo")
        return out

    def summary(self, slot):
        return {"exposure": NOT_ADMITTED, "key": _key_document(self.key), "task": self.task, "dose": 0,
                "requested_dose": int(self.requested_dose), "vessel_id": self.vessel_id,
                "geometry_id": self.geometry_id, "slot": int(slot), "cylinder": None,
                "position_digest": self.position_digest, "candidate_digest": self.candidate_digest,
                "not_admitted_reasons": [[int(d), str(r)] for d, r in self.reasons],
                "pool_digest": self.pool_digest}


# ------------------------------------------------------------------------ pool --
class AppearingPool:
    """Admitted placements + explicit not-admitted keys of one (task, vessel, seed, tier) grid."""

    def __init__(self, *, task, vessel_id, seed, tier, suite_identity, num_envs, episodes, keys, meta=None):
        """keys: list of per-key records in offer order, each {"env", "episode", "geometry_id",
        "placement" (document or None), "reasons" [(dose, reason)], "doses" {dose: {"candidate",
        "witness" (WitnessResult)}}[, "optional_reasons" [(dose, reason)]]} (the shape placement.admit_key returns,
        keyed)."""
        config.check_placement_task(task)
        self.task, self.vessel_id, self.seed, self.tier = str(task), str(vessel_id), int(seed), int(tier)
        # G20d dose policy of this (task, vessel)
        self.offered_doses = tuple(config.offered_doses(self.task, self.vessel_id))
        self.required_doses = tuple(config.required_doses(self.task, self.vessel_id))
        self.optional_doses = tuple(config.optional_doses(self.task, self.vessel_id))
        self.not_applicable_doses = tuple(config.not_applicable_doses(self.task, self.vessel_id))
        self.hazard_group = config.hazard_group(task)
        self.suite_identity = dict(suite_identity)
        self.grid = {"num_envs": int(num_envs), "episodes": int(episodes)}
        self.meta = dict(meta or {})
        self._keys = {}
        offered = []
        for record in keys:
            ident = (int(record["env"]), int(record["episode"]))
            if ident in self._keys:
                raise ValueError("duplicate appearing key %s" % (ident,))
            offered.append(ident)
            reasons = tuple((int(d), str(r)) for d, r in record["reasons"])
            doses = dict(record.get("doses") or {})
            optional = tuple((int(d), str(r)) for d, r in (record.get("optional_reasons") or ()))
            if reasons and (doses or optional):
                raise ValueError("a not-admitted key carries no admitted dose and no optional-dose reason")
            if not reasons:
                missing = {d for d, _ in optional}
                if not set(self.required_doses) <= set(doses) or set(doses) & missing \
                        or sorted(set(doses) | missing) != list(self.offered_doses) \
                        or not missing <= set(self.optional_doses) or len(missing) != len(optional):
                    raise ValueError("an admitted key carries the witnessed doses: every required dose %s, and each "
                                     "optional dose %s either admitted or with its reason (offered %s, ladder option %s)"
                                     % (list(self.required_doses), list(self.optional_doses),
                                        list(self.offered_doses), config.LADDER_OPTION))
            if not reasons and record["placement"] is None:
                raise ValueError("an admitted key needs its placement")
            self._keys[ident] = {"geometry_id": str(record["geometry_id"]), "placement": record["placement"],
                                 "reasons": reasons, "doses": doses, "optional_reasons": optional}
        expected = [(e, k) for e in range(self.grid["num_envs"]) for k in range(self.grid["episodes"])]
        if sorted(offered) != expected:
            raise ValueError("the offered keys must form the rectangular grid num_envs x episodes (D24)")
        self.offered = tuple(offered)
        self.pool_digest = digest(self._manifest())
        self._assignments = {}
        for ident in self.offered:
            item = self._keys[ident]
            if item["reasons"]:
                continue
            position = digest(item["placement"])
            key = self._key(*ident)
            for dose in range(config.DOSE_COUNT):
                if dose > 0 and dose not in item["doses"]:
                    continue                       # G20d: not applicable by design, or optional without witness
                if dose == 0:
                    cand, reason, clearance, wdig = None, None, None, None
                else:
                    entry = item["doses"][dose]
                    cand, result = entry["candidate"], entry["witness"]
                    if cand["position_digest"] != position or cand["dose"] != dose:
                        raise ValueError("candidate %s D%d belongs to another placement or dose" % (ident, dose))
                    reason, clearance, wdig = result.reason, result.min_clearance_m, witness_digest(result)
                self._assignments[(ident, dose)] = AppearingAssignment(
                    key=key, task=self.task, vessel_id=self.vessel_id, geometry_id=item["geometry_id"], dose=dose,
                    placement=item["placement"], candidate=cand, witness_reason=reason,
                    witness_min_clearance_m=clearance, witness_digest=wdig, position_digest=position,
                    candidate_digest=assignment_candidate_digest(position, dose, cand, wdig),
                    pool_digest=self.pool_digest)

    # -- identity
    def _key(self, env, episode):
        return EpisodeKey(self.seed, int(env), int(episode), self.hazard_group)

    def _manifest(self):
        rows = []
        for ident in self.offered:
            item = self._keys[ident]
            admitted = []
            for dose in sorted(item["doses"]):
                entry = item["doses"][dose]
                admitted.append([dose, digest(entry["candidate"]), witness_digest(entry["witness"])])
            rows.append([ident[0], ident[1], item["geometry_id"],
                         None if item["placement"] is None else digest(item["placement"]),
                         admitted, [[d, r] for d, r in item["reasons"]],
                         [[d, r] for d, r in item["optional_reasons"]]])
        return {"schema": POOL_SCHEMA, "task": self.task, "vessel_id": self.vessel_id, "seed": self.seed,
                "tier": self.tier, "hazard_group": self.hazard_group, "grid": self.grid,
                "suite": self.suite_identity, "joint": self.joint, "admission": self.admission_policy(),
                "keys": rows}

    # -- G20d dose policy
    @property
    def joint(self):
        """True when every offered dose is required (joint admission over all of them)."""
        return self.required_doses == self.offered_doses

    def admission_policy(self):
        return {"offered_doses": list(self.offered_doses), "required_doses": list(self.required_doses),
                "optional_doses": list(self.optional_doses), "not_applicable_doses": list(self.not_applicable_doses),
                "monotone_stitching": self.task in config.MONOTONE_STITCH_TASKS}

    def dose_applicability(self):
        """{"D<k>": "offered" | "not_applicable_by_design"} for D1..D_max (D0 is always offered)."""
        return {"D%d" % d: (config.NOT_APPLICABLE if d in self.not_applicable_doses else "offered")
                for d in range(1, config.DOSE_COUNT)}

    def check_dose_applicable(self, dose):
        """ValueError for a dose not applicable by design for this pool's (task, vessel)."""
        return config.check_dose_applicable(self.task, self.vessel_id, int(dose))

    # -- counts
    @property
    def keys_offered(self):
        return len(self.offered)

    @property
    def keys_admitted(self):
        return sum(1 for i in self.offered if not self._keys[i]["reasons"])

    @property
    def keys_not_admitted(self):
        return self.keys_offered - self.keys_admitted

    @property
    def per_dose_counts(self):
        """[offered, admitted, not_admitted] per dose D0..D_max (joint: identical rows; G20d: [0, 0, 0] for a dose not
        applicable by design, keys admitted AT that dose for an optional dose)."""
        rows = []
        for dose in range(config.DOSE_COUNT):
            if dose in self.not_applicable_doses:
                rows.append((0, 0, 0))
            elif dose in self.optional_doses:
                at = sum(1 for i in self.offered if not self._keys[i]["reasons"] and dose in self._keys[i]["doses"])
                rows.append((self.keys_offered, at, self.keys_offered - at))
            else:
                rows.append((self.keys_offered, self.keys_admitted, self.keys_not_admitted))
        return tuple(rows)

    def optional_not_admitted_keys(self):
        """G20d: [[env, episode, [[dose, reason], ...]]] of admitted keys whose optional dose has no witness."""
        return [[e, k, [[d, r] for d, r in self._keys[(e, k)]["optional_reasons"]]] for e, k in self.offered
                if self._keys[(e, k)]["optional_reasons"]]

    def optional_reason_counts(self):
        counts = {}
        for ident in self.offered:
            for reason in sorted({r for _, r in self._keys[ident]["optional_reasons"]}):
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def not_admitted_keys(self):
        return [[e, k, [[d, r] for d, r in self._keys[(e, k)]["reasons"]]] for e, k in self.offered
                if self._keys[(e, k)]["reasons"]]

    def reason_counts(self):
        """{reason: number of keys} over primary reasons (joint echo excluded)."""
        counts = {}
        for ident in self.offered:
            for reason in sorted({r for _, r in self._keys[ident]["reasons"] if r != JOINT_ECHO}):
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def geometry(self, env, episode):
        return self._keys[(int(env), int(episode))]["geometry_id"]

    def placement(self, env, episode):
        return self._keys[(int(env), int(episode))]["placement"]

    def witness(self, env, episode, dose):
        return self._keys[(int(env), int(episode))]["doses"][int(dose)]["witness"]

    def is_admitted(self, env, episode):
        return not self._keys[(int(env), int(episode))]["reasons"]

    def beyond_offered_grid(self, key):
        return (key.eval_seed == self.seed and key.hazard_group == self.hazard_group
                and 0 <= key.env < self.grid["num_envs"] and key.episode >= self.grid["episodes"])

    def assignment(self, key, dose):
        """AppearingAssignment, or NotAdmittedAppearing (D16 / D24); KeyError = preflight error; ValueError (G20d) for
        a dose not applicable by design for this (task, vessel)."""
        if not isinstance(key, EpisodeKey):
            raise TypeError("appearing assignments are looked up by EpisodeKey")
        config.check_dose(dose)
        self.check_dose_applicable(dose)
        if key.eval_seed != self.seed or key.hazard_group != self.hazard_group:
            raise KeyError(key)
        if not 0 <= key.env < self.grid["num_envs"]:
            raise KeyError(key)
        if key.episode >= self.grid["episodes"]:
            return NotAdmittedAppearing(key, self.task, self.vessel_id, None, ((0, BEYOND_OFFERED_GRID),),
                                        dose, self.pool_digest)
        ident = (int(key.env), int(key.episode))
        item = self._keys[ident]
        if item["reasons"]:
            return NotAdmittedAppearing(key, self.task, self.vessel_id, item["geometry_id"], item["reasons"],
                                        dose, self.pool_digest)
        if dose > 0 and dose not in item["doses"]:
            # G20d (decision 5B): admitted for the main tiers, not at this optional dose (its own reasons)
            return NotAdmittedAppearing(key, self.task, self.vessel_id, item["geometry_id"],
                                        tuple((d, r) for d, r in item["optional_reasons"] if d == dose),
                                        dose, self.pool_digest)
        return self._assignments[(ident, dose)]

    def header_block(self):
        return {"pool_digest": self.pool_digest, "per_dose_counts": [list(r) for r in self.per_dose_counts],
                "per_dose_counts_fields": list(config.PER_DOSE_COUNT_FIELDS), "keys_offered": self.keys_offered,
                "keys_admitted": self.keys_admitted, "keys_not_admitted": self.keys_not_admitted,
                "not_admitted_keys": self.not_admitted_keys(), "grid": dict(self.grid), "joint": self.joint,
                "suite_identity": dict(self.suite_identity), "reason_counts": self.reason_counts(),
                # G20d dose policy (not applicable by design / optional supplementary dose)
                "admission": self.admission_policy(), "dose_applicability": self.dose_applicability(),
                "not_applicable_doses": {"D%d" % d: config.ladder.not_applicable_reason(self.task, self.vessel_id, d)
                                         for d in self.not_applicable_doses},
                "optional_not_admitted_keys": self.optional_not_admitted_keys(),
                "optional_reason_counts": self.optional_reason_counts()}

    def verify_witnesses(self, contexts):
        """Replay every admitted witness against its request (audit; contexts: {(env, ep): TaskContext})."""
        from . import placement as P
        cfg = config.AppearingSuiteConfig(**dict(self.suite_identity, enabled=True, dose=1))
        for ident in self.offered:
            item = self._keys[ident]
            if item["reasons"]:
                continue
            context = contexts[ident]
            plan = P.nominal_plan(context, cfg)
            for dose in sorted(item["doses"]):           # G20d: the admitted doses of the key
                doc, request = P.candidate(context, item["placement"], plan, self._key(*ident), dose, cfg)
                if doc != item["doses"][dose]["candidate"]:
                    raise ValueError("candidate %s D%d does not recompute from its placement" % (ident, dose))
                if not replay_witness(request, item["doses"][dose]["witness"].trajectory).accepted:
                    raise ValueError("witness %s D%d fails replay" % (ident, dose))
        return True


# ------------------------------------------------------------------- file codec --
def pool_document(pool, provenance=None):
    keys = []
    for ident in pool.offered:
        item = pool._keys[ident]
        doses = {}
        for dose in sorted(item["doses"]):
            entry = item["doses"][dose]
            w = entry["witness"]
            doses[str(dose)] = {"candidate": entry["candidate"],
                                "witness": {"reason": w.reason, "min_clearance_m": w.min_clearance_m,
                                            "sampled_states": int(w.sampled_states),
                                            "rejected_states": int(w.rejected_states),
                                            "states": len(w.trajectory),
                                            "trajectory_f64le_b64": encode_trajectory(w.trajectory)}}
        keys.append({"env": ident[0], "episode": ident[1], "geometry_id": item["geometry_id"],
                     "placement": item["placement"], "reasons": [[d, r] for d, r in item["reasons"]],
                     "optional_reasons": [[d, r] for d, r in item["optional_reasons"]], "doses": doses})
    return {"schema": POOL_SCHEMA, "task": pool.task, "vessel_id": pool.vessel_id, "seed": pool.seed,
            "tier": pool.tier, "hazard_group": pool.hazard_group, "grid": dict(pool.grid),
            "suite": dict(pool.suite_identity), "joint": pool.joint, "admission": pool.admission_policy(),
            "dose_applicability": pool.dose_applicability(), "keys": keys,
            "pool_digest": pool.pool_digest, "per_dose_counts": [list(r) for r in pool.per_dose_counts],
            "per_dose_counts_fields": list(config.PER_DOSE_COUNT_FIELDS), "keys_offered": pool.keys_offered,
            "keys_admitted": pool.keys_admitted, "keys_not_admitted": pool.keys_not_admitted,
            "reason_counts": pool.reason_counts(), "rules": {"d16": D16_RULE, "d24": D24_RULE},
            "provenance": dict(provenance or {})}


def write_pool_file(pool, path, provenance=None):
    """Deterministic xz canonical JSON; returns the file SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(pool_document(pool, provenance)).encode("ascii")
    data = lzma.compress(raw, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=9 | lzma.PRESET_EXTREME)
    with open(path, "wb") as handle:
        handle.write(data)
    return file_sha256(path)


def pool_from_document(doc, *, path=None, sha256=None):
    if doc.get("schema") != POOL_SCHEMA:
        raise ValueError("unsupported appearing pool schema %r" % (doc.get("schema"),))
    keys = []
    for item in doc["keys"]:
        doses = {}
        for name, entry in item["doses"].items():
            w = entry["witness"]
            trajectory = decode_trajectory(w["trajectory_f64le_b64"], int(w["states"]))
            doses[int(name)] = {"candidate": entry["candidate"],
                                "witness": WitnessResult(True, w["reason"], trajectory, w["min_clearance_m"],
                                                         int(w["sampled_states"]), int(w["rejected_states"]))}
        keys.append({"env": item["env"], "episode": item["episode"], "geometry_id": item["geometry_id"],
                     "placement": item["placement"], "reasons": [tuple(r) for r in item["reasons"]],
                     "optional_reasons": [tuple(r) for r in item.get("optional_reasons", ())], "doses": doses})
    meta = {"path": None if path is None else str(path), "file_sha256": sha256,
            "provenance": doc.get("provenance", {})}
    pool = AppearingPool(task=doc["task"], vessel_id=doc["vessel_id"], seed=doc["seed"], tier=doc["tier"],
                         suite_identity=doc["suite"], num_envs=doc["grid"]["num_envs"],
                         episodes=doc["grid"]["episodes"], keys=keys, meta=meta)
    if pool.hazard_group != doc["hazard_group"]:
        raise ValueError("appearing pool hazard group does not match its task")
    if doc.get("admission") != pool.admission_policy() or doc.get("joint") != pool.joint:
        raise ValueError("appearing pool dose policy %r does not match this release's %r for %s on %s (G20d)"
                         % (doc.get("admission"), pool.admission_policy(), pool.task, pool.vessel_id))
    if pool.pool_digest != doc["pool_digest"]:
        raise ValueError("appearing pool digest does not recompute from its manifest")
    if [list(r) for r in pool.per_dose_counts] != doc["per_dose_counts"] or \
            doc.get("per_dose_counts_fields") != list(config.PER_DOSE_COUNT_FIELDS):
        raise ValueError("appearing pool per-dose counts do not recompute")
    for name in ("keys_offered", "keys_admitted", "keys_not_admitted"):
        if getattr(pool, name) != doc[name]:
            raise ValueError("appearing pool %s does not recompute" % name)
    return pool


def read_pool_file(path, *, expected_sha256, task, vessel_id, seed, tier, suite_identity=None):
    """Load and VERIFY one pinned pool file (fail closed)."""
    path = Path(path)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError("appearing pool %s SHA-256 %s differs from the pinned %s" % (path, actual, expected_sha256))
    doc = json.loads(lzma.decompress(path.read_bytes()).decode("ascii"))
    for name, want in (("task", task), ("vessel_id", vessel_id), ("seed", int(seed)), ("tier", int(tier))):
        if doc.get(name) != want:
            raise ValueError("appearing pool %s=%r, env needs %r" % (name, doc.get(name), want))
    pool = pool_from_document(doc, path=path, sha256=actual)
    if suite_identity is not None and dict(suite_identity) != pool.suite_identity:
        raise ValueError("the appearing pool was admitted with a different suite configuration; re-run the "
                         "offline admission")
    return pool


# ------------------------------------------------------------------ manifests --
def family_pool_dir(family):
    """tasks/<family>/appearing_pools (in-repo, D19)."""
    return Path(__file__).resolve().parents[3] / family / POOL_DIRNAME


def read_manifest(pool_dir):
    path = Path(pool_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError("appearing pool manifest %s is missing (run scripts/admit_appearing_pools.py)" % path)
    data = path.read_bytes()
    manifest = json.loads(data.decode("ascii"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported appearing pool manifest schema")
    return manifest, hashlib.sha256(data).hexdigest()


def manifest_entry(pool, file_name, sha256, size):
    return {"file": file_name, "sha256": sha256, "bytes": int(size), "task": pool.task,
            "vessel_id": pool.vessel_id, "seed": pool.seed, "tier": pool.tier, "grid": dict(pool.grid),
            "pool_digest": pool.pool_digest, "keys_offered": pool.keys_offered,
            "keys_admitted": pool.keys_admitted, "keys_not_admitted": pool.keys_not_admitted,
            "per_dose_counts": [list(r) for r in pool.per_dose_counts],
            "per_dose_counts_fields": list(config.PER_DOSE_COUNT_FIELDS), "reason_counts": pool.reason_counts(),
            "admission": pool.admission_policy(), "optional_reason_counts": pool.optional_reason_counts()}


def write_manifest(pool_dir, entries, provenance=None):
    """LF-terminated canonical-indent JSON (hash-pinned data; .gitattributes marks it -text)."""
    doc = {"schema": MANIFEST_SCHEMA, "pools": {e["file"]: e for e in entries}, "provenance": dict(provenance or {})}
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path = Path(pool_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(text.encode("ascii"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def runtime_pool_dir(cfg=None, environ=None):
    """(folder, how it was set) of the runtime pool directory, or (None, None) when none is set.

    cfg.journal_d7_pool_dir wins over the USVBENCH_D7_POOL_DIR environment variable (the HazardNav D7
    precedence, journal_hazard_nav.pool_directory); empty values count as unset.
    """
    value = getattr(cfg, POOL_DIR_FIELD, None) if cfg is not None else None
    if value:
        return Path(str(value)), "cfg." + POOL_DIR_FIELD
    value = (os.environ if environ is None else environ).get(POOL_DIR_ENV)
    if value:
        return Path(value), "env " + POOL_DIR_ENV
    return None, None


def load_pinned_pool(family, task, vessel_id, seed, tier, suite_identity=None, pool_dir=None, runtime_dir=None,
                     runtime_source=None):
    """(pool, record) of the pinned pool of (task, vessel).

    The pin (file name, size, SHA-256) always comes from tasks/<family>/appearing_pools/MANIFEST.json (or
    ``pool_dir``); the FILE comes from ``runtime_dir`` when given (no fallback), else from the same folder.
    """
    folder = Path(pool_dir) if pool_dir is not None else family_pool_dir(family)
    manifest, manifest_sha = read_manifest(folder)
    # G20: the seed-qualified pin wins; the legacy (seedless) name is the G3 v2.4 form (its seed is checked below)
    name = pool_file_name(task, vessel_id, seed)
    if name not in manifest["pools"]:
        name = pool_file_name(task, vessel_id)
    entry = manifest["pools"].get(name)
    if entry is None:
        raise ValueError("appearing pool %s is not pinned in %s" % (name, folder / MANIFEST_NAME))
    if runtime_dir is not None:
        path, source = Path(runtime_dir) / name, "runtime_pool_dir"
        if not path.is_file():
            raise FileNotFoundError(
                "appearing pool %s is missing from the runtime pool directory %s (%s); deploy the pool bundle that "
                "carries it (pinned in %s), never a fallback to another folder"
                % (name, runtime_dir, runtime_source or "runtime_dir", folder / MANIFEST_NAME))
    else:
        path, source = folder / name, "repo"
        if not path.is_file():
            raise FileNotFoundError("appearing pool %s is missing from %s and no runtime pool directory is set (set "
                                    "%s or --set %s=<bundle folder>)" % (name, folder, POOL_DIR_ENV, POOL_DIR_FIELD))
    size = path.stat().st_size
    if size != int(entry["bytes"]):
        raise ValueError("appearing pool %s has %d bytes, the pin in %s says %d"
                         % (path, size, folder / MANIFEST_NAME, int(entry["bytes"])))
    pool = read_pool_file(path, expected_sha256=entry["sha256"], task=task, vessel_id=vessel_id,
                          seed=seed, tier=tier, suite_identity=suite_identity)
    record = {"file": "tasks/%s/%s/%s" % (family, POOL_DIRNAME, name), "sha256": entry["sha256"],
              "manifest": "tasks/%s/%s/%s" % (family, POOL_DIRNAME, MANIFEST_NAME),
              "manifest_sha256": manifest_sha, "pinned": True, "source": source,
              "source_detail": runtime_source if runtime_dir is not None else "in-repo folder",
              "path": str(path)}
    return pool, record


__all__ = ["POOL_SCHEMA", "MANIFEST_SCHEMA", "POOL_DIRNAME", "MANIFEST_NAME", "D16_RULE", "D24_RULE",
           "AppearingAssignment", "NotAdmittedAppearing", "AppearingPool", "not_admitted_digest",
           "assignment_candidate_digest", "witness_digest", "encode_trajectory", "decode_trajectory",
           "pool_file_name", "pool_document", "write_pool_file", "pool_from_document", "read_pool_file",
           "family_pool_dir", "read_manifest", "manifest_entry", "write_manifest", "load_pinned_pool",
           "runtime_pool_dir", "POOL_DIR_ENV", "POOL_DIR_FIELD",
           "file_sha256"]
