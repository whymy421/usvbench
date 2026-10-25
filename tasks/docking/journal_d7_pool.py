"""D7 assignment pool for the journal docking task: JSON file I/O (fail closed), CONTEXT_R4 D16.

D7 B02/B05 + D16: every journal episode runs the E5 scenario of its key
(seed, env, task episode k) - never a redraw - and E4 admits or rejects the
key OFFLINE (tools_f5/admit_d7_pool.py -> E4 admit_grid, joint across the
five doses, one call, attempt 0 only). The pool stores, per key, EITHER the
admitted assignment ladder (all five doses) OR an explicit ``not_admitted``
record carrying E4's rejection reasons. A not-admitted key runs the dynamic
suite at D0 (no intruder, empty assignment; journal_d7_exposure.py) with
``exposure = "not_admitted"`` stamped, at every requested dose.

The env loads the pool by PATH and refuses anything it cannot verify:

  * the file SHA-256 must equal the pinned digest (``expected_sha256``, else the
    committed ``<pool>.sha256`` side file);
  * every stored candidate / witness is rebuilt from typed JSON (a whitelist of
    E4 dataclass names; no pickle, no code execution) and re-digested with E4's
    canonical digest (stamp.digest); both digests must match the stored values
    AND a row of E4's admission manifest, and the pool digest is recomputed from
    that manifest exactly as E4 ``_assemble`` hashes it (offered keys in order,
    sorted accepted rows, rejections, joint);
  * every offered key is the plain task key EpisodeKey(seed, env, k,
    "d7_docking") of exactly one grid cell, admitted at all five doses or
    rejected with its reasons (recomputed from the manifest rejections);
  * the pool header must match the env (task, vessel, geometry tier, eval seed).

The loaded object duck-types E4 ``AdmissionPool`` (``assignment(key, dose)``,
``pool_digest``, ``per_dose_counts``, ``keys_offered``, ``keys_admitted``,
``joint``) for the TASK keys the mixin builds; for a not-admitted key
``assignment`` returns a ``NotAdmittedAssignment`` (no candidate) that the
family bridge wrapper (journal_d7_exposure.ExposureBridge) runs at D0.
``geometry_id(env, k)`` is checked by the env against the geometry it
rebuilds (journal_docking.build_scenario) for admitted and not-admitted keys.

CONTEXT_R4 D24 (episodes beyond the offered grid): the evaluators keep
stepping every env until the slowest one reaches its per-env quota, so a
fast env autoresets into task episodes k >= grid["episodes"]. Such a key -
the pool's eval seed and hazard group, env inside the grid
(0 <= env < grid["num_envs"]), episode index >= grid["episodes"] - is a
NOT-ADMITTED key with the single reason ``beyond_offered_grid``:
``assignment`` returns a ``NotAdmittedAssignment`` whose reasons are exactly
``BEYOND_GRID_REASONS`` = ((0, "beyond_offered_grid", 0),) (dose 0 = the dose
the episode runs at, fixed so the record digest is dose independent;
sampled_states 0: nothing was sampled), ``exposure`` / ``not_admitted_reasons``
answer accordingly, and its geometry id is the one of the E5 scenario of the
key (journal_docking.build_scenario, a pure function of the key; the pool
holds no geometry record for it, so ``geometry_id(env, k)`` still refuses and
the env skips the pool geometry check for that key). Keys outside the env
range, with another eval seed or hazard group, or with a negative episode
stay preflight errors (KeyError from ``assignment``; ValueError from the
cell accessors). Beyond-grid keys are not pool keys: the per-dose counts are
unchanged. Such episodes are never counted (F7 refuses a counted record with
this reason).

Storage: canonical ASCII JSON, xz-compressed (``*.json.xz``), carrying E4's
complete admission manifest (offered, accepted, rejections, joint; the pool
digest is recomputed exactly) with the typed admitted candidate/witness
objects and the explicit not-admitted list.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import lzma
import math
from pathlib import Path

POOL_SCHEMA = "usvbench.docking.d7_pool/2"      # D16: no attempts; explicit not_admitted list
EXPOSURE_ADMITTED = "admitted"
EXPOSURE_NOT_ADMITTED = "not_admitted"
JOINT_MEMBER = "joint_ladder_member_rejected"
BEYOND_GRID_REASON = "beyond_offered_grid"          # CONTEXT_R4 D24
BEYOND_GRID_REASONS = ((0, BEYOND_GRID_REASON, 0),)   # (dose, reason, sampled_states): exactly one entry
BEYOND_GRID_RULE = ("CONTEXT_R4 D24: a key with the pool's eval seed and hazard group, env inside the grid and "
                    "episode index >= the grid's episodes per env is not admitted with the single reason "
                    "beyond_offered_grid; it keeps its E5 scenario, runs the dynamic suite at D0 and is never "
                    "counted")
HAZARD_GROUP = "d7_docking"
_FLOAT_TAG = "__float__"
_TYPE_TAG = "__dc__"
_WHITELIST = ("Candidate", "WitnessRequest", "WitnessResult", "Pose", "VesselEnvelope", "Footprint",
              "Encounter", "EpisodeKey", "Intruder", "Circle", "TimedState", "CapacityPlan", "SensorSpec",
              "RevealEvidence", "NativeMotion", "StageSite", "Rejection")


def _dynamic():
    from .._shared.conditions import dynamic
    return dynamic


def _classes():
    from .._shared.conditions.dynamic import adapters, capacity, feasibility, model, native, visibility
    table = {}
    for module in (adapters, capacity, feasibility, model, native, visibility):
        for name in _WHITELIST:
            if hasattr(module, name):
                table[name] = getattr(module, name)
    missing = set(_WHITELIST) - set(table)
    if missing:
        raise ImportError("E4 dataclasses missing for the pool reader: %s" % sorted(missing))
    return table


def encode(value):
    """Typed, lossless JSON-native form (dataclasses tagged, tuples as lists, non-finite floats tagged)."""
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _WHITELIST:
            raise TypeError("pool writer refuses dataclass %s" % name)
        return {_TYPE_TAG: name, "f": {f.name: encode(getattr(value, f.name)) for f in fields(value) if f.init}}
    if isinstance(value, float) and not math.isfinite(value):
        return {_FLOAT_TAG: repr(value)}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        raise TypeError("pool writer supports dataclasses, tuples and JSON scalars only")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("pool writer cannot encode %r" % (type(value).__name__,))


def decode(value, classes):
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            text = value[_FLOAT_TAG]
            if text not in ("inf", "-inf", "nan"):
                raise ValueError("bad tagged float %r" % (text,))
            return float(text)
        if set(value) != {_TYPE_TAG, "f"} or value[_TYPE_TAG] not in classes:
            raise ValueError("pool reader refuses object %r" % (value.get(_TYPE_TAG),))
        cls = classes[value[_TYPE_TAG]]
        return cls(**{k: decode(v, classes) for k, v in value["f"].items()})
    if isinstance(value, list):
        return tuple(decode(v, classes) for v in value)
    return value


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def not_admitted_reasons(rejections, key):
    """E4 rejection reasons of one key: ((dose, reason, sampled_states), ...) of the FAILED doses, sorted.

    Joint-ladder bookkeeping rows ("joint_ladder_member_rejected": a dose that
    admitted but was dropped because another dose failed) are not reasons.
    """
    rows = [(int(r.dose), str(r.reason), int(r.sampled_states)) for r in rejections
            if r.key == key and r.reason != JOINT_MEMBER]
    return tuple(sorted(rows))


def _grid_index(value):
    """Integer index or None (bools and non-integral values are not indices)."""
    if isinstance(value, bool):
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index == value else None


def top_reason(reasons):
    """Reason class of the lowest failed dose (text before the first ':')."""
    return reasons[0][1].split(":")[0] if reasons else ""


def pool_to_document(pool, *, vessel, tier, seed, num_envs, episodes, offered, suite, geometry_ids,
                     generator=None, berth_walls=True):
    """JSON document of the E4 AdmissionPool of one vessel (D16: admitted + not admitted).

    offered: the task EpisodeKeys (seed, env, k, "d7_docking") in the order
    given to admit_grid (every grid cell exactly once, attempt 0 only);
    geometry_ids: {(env, k): geometry_id of the E5 scenario of that key}.
    """
    dynamic = _dynamic()
    from .._shared.conditions.dynamic.stamp import plain
    if len(offered) != pool.keys_offered:
        raise ValueError("the pool was admitted from a different key list")
    cells = [(int(k.env), int(k.episode)) for k in offered]
    if sorted(cells) != [(e, k) for e in range(int(num_envs)) for k in range(int(episodes))]:
        raise ValueError("the offered keys must be exactly the (env, episode) grid, once each")
    if any(k.eval_seed != int(seed) or k.hazard_group != HAZARD_GROUP for k in offered):
        raise ValueError("offered keys must be the plain task keys (seed, env, k, %r)" % HAZARD_GROUP)
    accepted = sorted([k.eval_seed, k.env, k.episode, k.hazard_group, d, a.candidate_digest, a.witness_digest]
                      for (k, d), a in pool.assignments.items())
    assignments = []
    ordered = sorted(pool.assignments.items(), key=lambda kv: (kv[0][0].env, kv[0][0].episode, kv[0][1]))
    for (key, dose), a in ordered:
        assignments.append({"key": encode(key), "dose": dose, "candidate": encode(a.candidate),
                            "witness": encode(a.witness), "candidate_digest": a.candidate_digest,
                            "witness_digest": a.witness_digest})
    admitted = {k for (k, _d) in pool.assignments}
    task_episodes, not_admitted = [], []
    for key in sorted(offered, key=lambda k: (k.env, k.episode)):
        cell = (int(key.env), int(key.episode))
        status = EXPOSURE_ADMITTED if key in admitted else EXPOSURE_NOT_ADMITTED
        task_episodes.append([cell[0], cell[1], geometry_ids[cell], status])
        if status == EXPOSURE_NOT_ADMITTED:
            reasons = not_admitted_reasons(pool.rejections, key)
            not_admitted.append({"env": cell[0], "episode": cell[1], "geometry_id": geometry_ids[cell],
                                 "reasons": [list(r) for r in reasons], "top_reason": top_reason(reasons)})
    return {
        "schema": POOL_SCHEMA, "task": "docking", "vessel": vessel, "geometry_tier": int(tier),
        "eval_seed": int(seed), "grid": {"num_envs": int(num_envs), "episodes": int(episodes)},
        "berth_walls": bool(berth_walls),
        "hazard_group": HAZARD_GROUP, "suite": plain(suite), "d7_suite_version": dynamic.D7_SUITE_VERSION,
        "joint": bool(pool.joint), "keys_offered": pool.keys_offered, "keys_admitted": pool.keys_admitted,
        "keys_not_admitted": len(not_admitted),
        "per_dose_counts": [list(x) for x in pool.per_dose_counts], "pool_digest": pool.pool_digest,
        "manifest_offered": [encode(k) for k in offered],
        "manifest_accepted": accepted,
        "manifest_rejections": [encode(r) for r in pool.rejections],
        "assignments": assignments,
        "task_episodes": task_episodes,
        "not_admitted": not_admitted,
        "generator": generator or {},
    }


def _record_digest(key, geometry_id, reasons, pool_digest):
    from .._shared.conditions.dynamic.stamp import digest
    return digest({"schema": POOL_SCHEMA, "exposure": EXPOSURE_NOT_ADMITTED,
                   "key": [key.eval_seed, key.env, key.episode, key.hazard_group],
                   "geometry_id": geometry_id, "reasons": [list(r) for r in reasons], "pool_digest": pool_digest})


class NotAdmittedAssignment:
    """Explicit not-admitted record of one key (D16): no candidate, no intruder.

    ``requested_dose`` is the dose the dynamic suite asked for; the episode
    runs at D0 whatever it is, so ``dose`` is always 0 and the D7 pairing
    primitive (candidate digest = ``record_digest``, task, dose 0, key) is the
    same at every requested dose and in every other suite's D0 run.
    """

    exposure = EXPOSURE_NOT_ADMITTED
    candidate = None
    witness = None
    dose = 0

    def __init__(self, key, geometry_id, reasons, pool_digest, vessel, requested_dose):
        self.key = key
        self.geometry_id = str(geometry_id)
        self.reasons = tuple(tuple(r) for r in reasons)
        self.pool_digest = pool_digest
        self.vessel = vessel
        self.requested_dose = int(requested_dose)
        self.record_digest = _record_digest(key, self.geometry_id, self.reasons, pool_digest)

    @property
    def top_reason(self):
        return top_reason(self.reasons)

    @property
    def beyond_grid(self):
        """True for a CONTEXT_R4 D24 key (episode index beyond the offered grid)."""
        return self.reasons == BEYOND_GRID_REASONS

    def __repr__(self):
        return "NotAdmittedAssignment(%r, D%d, %s)" % (self.key, self.requested_dose, self.top_reason)


class DockingD7Pool:
    """Verified, read-only pool (duck-types E4 AdmissionPool for JournalEnvMixin).

    ``assignment(key, dose)`` takes the TASK key the mixin builds,
    EpisodeKey(seed, env, k, "d7_docking"), and returns the admitted E4
    Assignment of that key at that dose, or a NotAdmittedAssignment when E4
    rejected the key (D16) or the episode index is beyond the offered grid
    (D24, reason "beyond_offered_grid"); a key outside the env range, with
    another eval seed or hazard group raises KeyError (the mixin's preflight
    error).
    """

    def __init__(self, document, by_task_key, task_episodes, not_admitted, sha256):
        self.document_header = {k: document[k] for k in ("schema", "task", "vessel", "geometry_tier", "eval_seed",
                                                          "grid", "berth_walls", "hazard_group",
                                                          "d7_suite_version")}
        self.berth_walls = bool(document["berth_walls"])
        self.vessel = document["vessel"]
        self.geometry_tier = int(document["geometry_tier"])
        self.eval_seed = int(document["eval_seed"])
        self.grid = dict(document["grid"])
        self.suite = dict(document["suite"])
        self.assignments = by_task_key
        self.pool_digest = document["pool_digest"]
        self.per_dose_counts = tuple(tuple(int(v) for v in x) for x in document["per_dose_counts"])
        self.keys_offered = int(document["keys_offered"])
        self.keys_admitted = int(document["keys_admitted"])
        self.keys_not_admitted = len(not_admitted)
        self.joint = bool(document["joint"])
        self.file_sha256 = sha256
        self._task_episodes = task_episodes          # (env, k) -> (geometry_id, exposure)
        self._not_admitted = not_admitted            # (env, k) -> reasons

    def assignment(self, key, dose):
        if type(dose) is not int or not 0 <= dose <= 4:
            raise KeyError((key, dose))
        if self.beyond_grid(key.env, key.episode, key.eval_seed, key.hazard_group):
            return NotAdmittedAssignment(key, self.beyond_grid_geometry_id(key.env, key.episode),
                                         BEYOND_GRID_REASONS, self.pool_digest, self.vessel, dose)
        cell = (int(key.env), int(key.episode))
        if (key.eval_seed == self.eval_seed and key.hazard_group == HAZARD_GROUP
                and cell in self._not_admitted):
            return NotAdmittedAssignment(key, self._task_episodes[cell][0], self._not_admitted[cell],
                                         self.pool_digest, self.vessel, dose)
        return self.assignments[(key, dose)]

    def covers(self, env, k):
        return (int(env), int(k)) in self._task_episodes

    def beyond_grid(self, env, k, seed=None, hazard_group=HAZARD_GROUP):
        """CONTEXT_R4 D24 (a): env inside the grid, episode index >= the grid's episodes per env.

        ``seed`` (when given) and ``hazard_group`` must be the pool's; any other
        key is not a beyond-grid key (it stays a preflight error).
        """
        if seed is not None and (_grid_index(seed) is None or int(seed) != self.eval_seed):
            return False
        if hazard_group != HAZARD_GROUP:
            return False
        env, k = _grid_index(env), _grid_index(k)
        if env is None or k is None:
            return False
        return 0 <= env < int(self.grid["num_envs"]) and k >= int(self.grid["episodes"])

    def beyond_grid_geometry_id(self, env, k):
        """Geometry id of a beyond-grid key: the E5 scenario of the key (pure function of the key).

        The pool holds no geometry record for it (D24); the id is rebuilt with
        journal_docking.build_scenario from the pool's vessel, tier, eval seed
        and berth_walls - the same call the env makes for that key.
        """
        if not self.beyond_grid(env, k):
            raise ValueError("env %r episode %r is not beyond the offered grid %s" % (env, k, self.grid))
        from .journal_docking import build_scenario
        return build_scenario(self.vessel, self.geometry_tier, self.eval_seed, int(env), int(k),
                              self.berth_walls).geometry_id

    def exposure(self, env, k):
        """"admitted" or "not_admitted" for (env, task episode k); ValueError outside the grid.

        D24: an episode index beyond the offered grid (env inside it) is "not_admitted".
        """
        if self.beyond_grid(env, k):
            return EXPOSURE_NOT_ADMITTED
        return self._entry(env, k)[1]

    def not_admitted_reasons(self, env, k):
        if self.beyond_grid(env, k):
            return BEYOND_GRID_REASONS
        self._entry(env, k)
        return self._not_admitted.get((int(env), int(k)), ())

    def geometry_id(self, env, k):
        return self._entry(env, k)[0]

    def _entry(self, env, k):
        try:
            return self._task_episodes[(int(env), int(k))]
        except KeyError:
            if self.beyond_grid(env, k):
                message = ("env %d episode %d is beyond the offered grid %s (CONTEXT_R4 D24): the pool holds no "
                           "record of it (a beyond-grid key of the pool's eval seed runs not admitted and the env "
                           "skips the pool geometry check for it)" % (int(env), int(k), self.grid))
                raise ValueError(message) from None
            raise ValueError("the D7 pool (seed %d, grid %s) holds no docking episode for env %d, episode %d; "
                             "admit a larger grid with tools_f5/admit_d7_pool.py (preflight error, never an "
                             "in-episode redraw; D7 B02)" % (self.eval_seed, self.grid, int(env), int(k)))

    def per_dose_exposure_counts(self):
        """Header per-dose counts incl. not_admitted (D16): [{dose, offered, admitted, not_admitted}]."""
        return [{"dose": d, "offered": int(o), "admitted": int(a), "not_admitted": int(o) - int(a)}
                for d, (o, a) in enumerate(self.per_dose_counts)]

    def header(self):
        return dict(self.document_header, pool_digest=self.pool_digest, file_sha256=self.file_sha256,
                    per_dose_counts=[list(x) for x in self.per_dose_counts],
                    per_dose_exposure_counts=self.per_dose_exposure_counts(),
                    task_episodes=len(self._task_episodes), keys_admitted=self.keys_admitted,
                    keys_not_admitted=self.keys_not_admitted,
                    not_admitted=[[e, k, top_reason(r)] for (e, k), r in sorted(self._not_admitted.items())],
                    exposure_rule=("CONTEXT_R4 D16: every key keeps its E5 scenario; a key E4 rejects runs the "
                                   "dynamic suite at D0 with exposure not_admitted"),
                    beyond_grid_rule=BEYOND_GRID_RULE)


def pool_from_document(document, sha256=None):
    """Rebuild and VERIFY a pool document (raises ValueError on any mismatch)."""
    dynamic = _dynamic()
    from .._shared.conditions.dynamic.stamp import digest
    if document.get("schema") != POOL_SCHEMA or document.get("task") != "docking":
        raise ValueError("not a docking D7 pool document (%r; expected %r)" % (document.get("schema"), POOL_SCHEMA))
    if document.get("d7_suite_version") != dynamic.D7_SUITE_VERSION:
        raise ValueError("pool admitted under D7 suite %r, runtime is %r"
                         % (document.get("d7_suite_version"), dynamic.D7_SUITE_VERSION))
    classes = _classes()
    offered = [decode(k, classes) for k in document["manifest_offered"]]
    rejections = [decode(r, classes) for r in document["manifest_rejections"]]
    accepted = [list(row) for row in document["manifest_accepted"]]
    manifest = {"offered": offered, "accepted": sorted(accepted), "rejections": rejections,
                "joint": bool(document["joint"])}
    if digest(manifest) != document["pool_digest"]:
        raise ValueError("pool digest does not match the recomputed E4 manifest digest")
    if len({(r[1], r[2]) for r in accepted}) != int(document["keys_admitted"]) \
            or len(offered) != int(document["keys_offered"]):
        raise ValueError("key counts disagree with the manifest")
    if not bool(document["joint"]):
        raise ValueError("docking pools are admitted jointly across D0..D4")
    seed, group = int(document["eval_seed"]), document["hazard_group"]
    grid = document["grid"]
    cells = sorted((int(k.env), int(k.episode)) for k in offered)
    if cells != [(e, k) for e in range(int(grid["num_envs"])) for k in range(int(grid["episodes"]))]:
        raise ValueError("the offered keys are not exactly the pool grid (one plain task key per cell)")
    if any(k.eval_seed != seed or k.hazard_group != group or group != HAZARD_GROUP for k in offered):
        raise ValueError("offered keys must be the plain task keys (seed, env, k, %r)" % HAZARD_GROUP)
    rows = {tuple(row) for row in accepted}
    stored = {}
    for item in document["assignments"]:
        key = decode(item["key"], classes)
        dose = int(item["dose"])
        candidate = decode(item["candidate"], classes)
        witness = decode(item["witness"], classes)
        candidate_digest = digest(candidate)
        witness_digest = digest({"reason": witness.reason, "trajectory": witness.trajectory,
                                 "min_clearance_m": witness.min_clearance_m})
        if candidate_digest != item["candidate_digest"] or witness_digest != item["witness_digest"]:
            raise ValueError("candidate/witness digest mismatch for %r dose %d" % (key, dose))
        row = (key.eval_seed, key.env, key.episode, key.hazard_group, dose, candidate_digest, witness_digest)
        if row not in rows:
            raise ValueError("stored assignment %r dose %d is not in the admitted manifest" % (key, dose))
        if candidate.request.encounter.key != key or candidate.request.encounter.dose != dose:
            raise ValueError("assignment key/dose disagree with the candidate encounter")
        if not witness.accepted:
            raise ValueError("stored witness was not accepted")
        candidate.validate()
        stored[(key, dose)] = dynamic.Assignment(candidate, witness, candidate_digest, witness_digest,
                                                 document["pool_digest"])
    if len(stored) != len(rows):
        raise ValueError("the pool must store every admitted assignment")
    admitted_keys = {k for (k, _d) in stored}
    listed = {(int(r["env"]), int(r["episode"])): r for r in document["not_admitted"]}
    task_episodes, not_admitted, by_task_key = {}, {}, {}
    for env, k, geometry_id, status in document["task_episodes"]:
        env, k, geometry_id = int(env), int(k), str(geometry_id)
        key = dynamic.EpisodeKey(seed, env, k, group)
        if (env, k) in task_episodes:
            raise ValueError("duplicate task episode env %d episode %d" % (env, k))
        if status == EXPOSURE_ADMITTED:
            if key not in admitted_keys or (env, k) in listed:
                raise ValueError("env %d episode %d is recorded admitted but the manifest disagrees" % (env, k))
            for dose in range(5):
                if (key, dose) not in stored:
                    raise ValueError("env %d task episode %d lacks an admitted dose D%d" % (env, k, dose))
                a = stored[(key, dose)]
                if a.candidate.geometry_id != geometry_id:
                    raise ValueError("assignment geometry differs from the recorded task episode geometry")
                by_task_key[(key, dose)] = a
        elif status == EXPOSURE_NOT_ADMITTED:
            if key in admitted_keys or (env, k) not in listed:
                raise ValueError("env %d episode %d is recorded not admitted but the manifest disagrees" % (env, k))
            reasons = not_admitted_reasons(rejections, key)
            record = listed[(env, k)]
            if (not reasons or [list(r) for r in reasons] != [list(r) for r in record["reasons"]]
                    or record["top_reason"] != top_reason(reasons) or str(record["geometry_id"]) != geometry_id):
                raise ValueError("not-admitted record of env %d episode %d disagrees with the E4 rejections"
                                 % (env, k))
            not_admitted[(env, k)] = reasons
        else:
            raise ValueError("unknown exposure status %r" % (status,))
        task_episodes[(env, k)] = (geometry_id, status)
    if sorted(task_episodes) != cells or len(listed) != len(not_admitted):
        raise ValueError("task episodes must cover the grid and every not-admitted record")
    if (int(document.get("keys_not_admitted", -1)) != len(not_admitted)
            or len(not_admitted) + int(document["keys_admitted"]) != len(offered)):
        raise ValueError("admitted + not admitted must equal the offered keys")
    return DockingD7Pool(document, by_task_key, task_episodes, not_admitted, sha256)


def write_pool(path, document):
    """Write canonical ASCII JSON (xz-compressed for ``*.xz``) and the ``.sha256`` side file.

    Returns the SHA-256 of the file bytes as written (the value to pin).
    """
    path = Path(path)
    data = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    if path.suffix == ".xz":
        data = lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6)
    path.write_bytes(data)
    sha = file_sha256(path)
    Path(str(path) + ".sha256").write_bytes(("%s  %s\n" % (sha, path.name)).encode("ascii"))
    return sha


def read_expected_sha256(path):
    side = Path(str(path) + ".sha256")
    if not side.is_file():
        raise FileNotFoundError("D7 pool digest file %s is missing (fail closed)" % side)
    text = side.read_text(encoding="ascii").split()
    if not text or len(text[0]) != 64:
        raise ValueError("malformed D7 pool digest file %s" % side)
    return text[0]


def load_pool(path, *, expected_sha256=None, vessel=None, tier=None, seed=None, berth_walls=None):
    """Load a pool file by path; every check fails closed (see module docstring)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("journal docking D7 pool %s is missing (fail closed; D7 B02/B05)" % path)
    expected = expected_sha256 or read_expected_sha256(path)
    sha = file_sha256(path)
    if sha != expected:
        raise ValueError("D7 pool %s SHA-256 %s differs from the pinned %s" % (path, sha, expected))
    data = path.read_bytes()
    if path.suffix == ".xz":
        data = lzma.decompress(data, format=lzma.FORMAT_XZ)
    document = json.loads(data.decode("ascii"))
    pool = pool_from_document(document, sha)
    for name, want, have in (("vessel", vessel, pool.vessel), ("geometry tier", tier, pool.geometry_tier),
                             ("eval seed", seed, pool.eval_seed), ("berth_walls", berth_walls, pool.berth_walls)):
        if want is not None and want != have:
            raise ValueError("D7 pool %s was admitted for %s %r, the env runs %r (re-run "
                             "tools_f5/admit_d7_pool.py)" % (path, name, have, want))
    return pool


def suite_matches(pool, suite):
    """The runtime DynamicSuiteConfig equals the admitted one except enabled/dose."""
    from .._shared.conditions.dynamic.stamp import plain
    have = dict(plain(suite))
    want = dict(pool.suite)
    for name in ("enabled", "dose"):
        have.pop(name, None)
        want.pop(name, None)
    return have == want


__all__ = ["POOL_SCHEMA", "HAZARD_GROUP", "EXPOSURE_ADMITTED", "EXPOSURE_NOT_ADMITTED", "BEYOND_GRID_REASON",
           "BEYOND_GRID_REASONS", "BEYOND_GRID_RULE", "DockingD7Pool",
           "NotAdmittedAssignment", "not_admitted_reasons", "top_reason", "encode", "decode", "pool_to_document",
           "pool_from_document", "write_pool", "load_pool", "read_expected_sha256", "file_sha256",
           "suite_matches"]
