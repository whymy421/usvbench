"""Station-keeping journal helpers (task contract v2, round F / F4).

CPU-importable: torch + numpy + the pure tasks/_shared modules; no Isaac Lab.
Used by the journal branch of station_keeping_env.py, by the offline D7
admission tool (work/tools_f4/admit_d7_pool.py) and by the CPU tests.

Contents
  * layout_rng / build_station_layout: the journal layout of one scenario
    (E5 journal_layouts_v2.build_layout) drawn from the scenario-keyed numpy
    stream (seed, env, task episode, "journal_layout"); the env and the
    offline D7 admission use the SAME function, so the admitted TaskContext is
    the one the env runs.
  * station_task_context: E4 TaskContext from the layout's d7_hook (env-local
    metres; the settled clock is the layout's FIXED time T - hold_s, never a
    runtime hold counter: D7 INTEGRATION_SPEC S4).
  * StationD7Pool: the admitted D7 assignment pool loaded from its JSON
    manifest by path (fail closed: file SHA-256, recomputed pool digest,
    geometry-version digest, seed/vessel/task). Each candidate is rebuilt from
    the bound REAL layout and must reproduce the admitted candidate digest;
    the admitted witness comes from the companion witness file (gzip+base64
    JSON per (key, dose), file SHA-256 pinned in the manifest) and must
    reproduce the admitted witness digest (without the file it is recomputed
    by E4's pure witness search, same digest check, slower).
    Round T (CONTEXT_R4 D16, D20, D22, D24): every key is classified as
    admitted, not_admitted (offered, E4 rejected it; reasons DERIVED from the
    stored E4 rejections, which the pool digest covers) or
    beyond_offered_grid (run's seed and hazard group, env inside the offered
    grid, episode index >= the grid's episodes per env); the last two get a
    NotAdmittedAssignment (dose 0, no intruder, dose-independent
    candidate_digest); any other key stays a preflight error.
  * StationExposureBridge: wrapper of the E4 HazardBridge (the interface the
    F0 mixin calls); admitted rows go to E4 unchanged, not-admitted rows run
    the dynamic suite at D0 (no D7 object, no contact) and stamp the D20 form.
  * admit_station_grid: the offline joint E4 admission of a key list on the
    keys' own E5 layouts (one admit_grid pass, no redraw); ``context_fn`` is a
    test hook that forces a REAL E4 rejection.
  * py_hypot: an exact torch replica of CPython's math.hypot (two arguments),
    so the tensor twin reproduces the scalar oracle's distances bit for bit.
  * StationKeepingTwin: vectorised torch twin of
    journal_contracts_v2.StationKeepingEpisode (the scalar oracle stays the
    reference; parity is test-asserted bit for bit).
  * point_return_rays: the 36-ray block (E4 visibility.py point-return proxy:
    the ray whose bearing is nearest to a VISIBLE object's centre returns the
    range to the object's nearest surface point).
  * body_rates / yaw_from_quat / quat_from_yaw: full 3-D hull transforms.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

from .._shared import journal_cfg_v2 as plans
from .._shared import journal_contracts_v2 as contracts
from .._shared import journal_layouts_v2 as layouts
from .._shared import vehicles_v2
from .._shared.scenario_rng import SCENARIO_PROTOCOL_VERSION, numpy_generator

TASK = "station_keeping"
HAZARD_GROUP = "d7_" + TASK
LAYOUT_GROUP = "journal_layout"
POOL_SCHEMA = "usvbench.f4.station_d7_pool/1"
# /2 (G7 item A): the companion file also stores every admitted candidate (typed JSON, gzip + base64).
WITNESS_SCHEMA = "usvbench.f4.station_d7_witnesses/2"
_FLOAT_TAG = "__float__"
_TYPE_TAG = "__dc__"
_E4_CLASSES = ("Candidate", "WitnessRequest", "WitnessResult", "Pose", "VesselEnvelope", "Footprint",
               "Encounter", "EpisodeKey", "Intruder", "Circle", "TimedState", "CapacityPlan", "SensorSpec",
               "RevealEvidence", "NativeMotion", "StageSite", "Rejection")
CONTROL_DT_S = plans.CONTROL_DT_S
TOP_DOSE_START_S = 0.0            # CONTEXT_R4 D6: constant current from t = 0 (no ramp)
# Round T: D7 exposure of a key (CONTEXT_R4 D16, D20, D24)
ADMITTED = "admitted"
NOT_ADMITTED = "not_admitted"                    # D16 exposure value (payload, payload.runtime, assignment)
BEYOND_OFFERED_GRID = "beyond_offered_grid"      # D24 (a): the single reason of a key past the offered grid
E4_REJECTED = "e4_rejected"                      # not-admitted kind of an offered key E4 rejected
JOINT_ECHO = "joint_ladder_member_rejected"      # E4 joint bookkeeping row, not a reason (D20 (c))
NOT_ADMITTED_SCHEMA = "usvbench.f4.station_d7_not_admitted/1"
PER_DOSE_COLUMNS = ("offered", ADMITTED, NOT_ADMITTED)
# G23e: a dose tier excluded by the approach design (sk_approach.py) is neither offered, admitted nor not_admitted
NOT_APPLICABLE = "not_applicable_by_design"
NOT_APPLICABLE_RULE = ("G23e (owner decision 2026-10-06): the station-keeping intruder meets the own ship on the "
                       "approach at 0.45 x start-to-centre distance from the hold centre; a tier whose nominal "
                       "reveal would precede the straight cruise-speed approach (BlueBoat D1: 1.08 m behind the "
                       "start) is not applicable by design: never offered, never run, distinct from not_admitted")
NOT_ADMITTED_RULE = ("CONTEXT_R4 D16: E4 rejected this offered key; the E5 layout is kept (no redraw) and the "
                     "dynamic suite runs at D0 (no intruder) at every requested dose")
BEYOND_GRID_RULE = ("CONTEXT_R4 D24: the episode index is past the pool's offered grid (an evaluator autoreset "
                    "after the env's quota); the E5 layout is built from the same keyed stream, the dynamic "
                    "suite runs at D0 (no intruder); such an episode is never counted")

TwinState = namedtuple("TwinState", ("com_xyz", "quat_wxyz", "lin_vel", "ang_vel", "water_level",
                                     "nominal_height"))


def _dynamic():
    from .._shared.conditions import dynamic
    return dynamic


# ----------------------------------------------------------------- layouts --
def layout_rng(seed, env_index, episode_index):
    """Scenario-keyed numpy stream of the journal layout (scenario_rng key algebra)."""
    return numpy_generator(SCENARIO_PROTOCOL_VERSION, int(seed), int(env_index), int(episode_index),
                           LAYOUT_GROUP)


def build_station_layout(tier, vessel_key, seed, env_index, episode_index):
    """Admitted journal layout of one (seed, env, task episode) scenario (E5)."""
    return layouts.build_layout(TASK, int(tier), vessel_key, layout_rng(seed, env_index, episode_index))


def station_task_context(layout, vessel_key):
    """E4 TaskContext of one layout (env-local metres), mirroring layout.d7_hook."""
    dyn = _dynamic()
    hook = layout.d7_hook
    if hook["task"] != TASK or layout.vessel != vessel_key:
        raise ValueError("layout belongs to another task or vessel")
    record = plans.vessel_record(vessel_key)
    envelope, _ = dyn.envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel_key),
                                            reference_speed_mps=record.cruise_speed_m_s,
                                            turning_radius_m=record.turning_radius_design_m)
    hooked = hook["envelope"]
    fp = envelope.footprint
    if envelope.vessel_id != hooked["vessel_id"] or not all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in
            ((fp.x_min_m, hooked["x_min_m"]), (fp.x_max_m, hooked["x_max_m"]),
             (fp.y_min_m, hooked["y_min_m"]), (fp.y_max_m, hooked["y_max_m"]))):
        raise ValueError("E4 envelope footprint differs from the E5 layout vessel geometry")
    sensor = hook["sensor"]
    if sensor is None:
        raise ValueError("journal station keeping carries the 36-ray sensor (D14)")
    x, y, yaw = (float(v) for v in hook["initial_pose"])
    return dyn.TaskContext(
        task=TASK, envelope=envelope, initial_pose=dyn.Pose(x, y, yaw, 0.0),
        bounds=tuple(float(v) for v in hook["bounds"]),
        goal=dyn.Circle(*(float(v) for v in hook["goal"])), deadline_s=float(hook["deadline_s"]),
        sensor=dyn.SensorSpec(kind="ray_fan", ray_count=int(sensor["ray_count"]),
                              max_range_m=float(sensor["max_range_m"]),
                              source="journal 36-ray fan, 30 m x length scale (journal_cfg_v2 plan; D14)"),
        hold_circle=dyn.Circle(*(float(v) for v in hook["hold_circle"])),
        settled_phase_start_s=float(hook["settled_phase_start_s"]),
        base_max=int(hook["base_max"]), base_count=int(hook["base_count"]),
        geometry_id=str(hook["geometry_id"]))


def appearing_admission_inputs(vessel_key, tier, seed, env_index, episode_index):
    """G20: (D7 TaskContext, length scale, None) of one key's REAL journal station layout (radius 0.8 s)."""
    layout = build_station_layout(int(tier), vessel_key, int(seed), int(env_index), int(episode_index))
    return station_task_context(layout, vessel_key), float(layout.params["length_scale"]), None


def episode_key(seed, env_index, episode_index):
    return _dynamic().EpisodeKey(int(seed), int(env_index), int(episode_index), HAZARD_GROUP)


def sha256_json(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def spawn_pose_hash(layout, com_z_above_water_m):
    """D11 family primitive ``spawn_pose``: exogenous spawn record (never trajectory values)."""
    return sha256_json({"schema": "usvbench.f4.spawn_pose/1", "task": TASK, "vessel": layout.vessel,
                        "com_xy_local_m": [repr(float(v)) for v in layout.start],
                        "heading_rad": repr(float(layout.start_heading_rad)),
                        "com_z_above_water_m": repr(float(com_z_above_water_m)),
                        "speed_mps": repr(0.0)})


def layout_hash(layout):
    """D11 family primitive ``layout``: the E5 geometry digest (sha256 hex)."""
    digest = str(layout.geometry_hash)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("E5 geometry hash is not a sha256 hex digest")
    return digest


# -------------------------------------------------------------------- pool --
def pool_manifest_digest(document):
    """Recompute E4's pool digest (adapters._assemble manifest) from the JSON document."""
    dyn = _dynamic()
    manifest = {"offered": [dict(k) for k in document["offered"]],
                "accepted": sorted([list(a) for a in document["accepted"]]),
                "rejections": [dict(r) for r in document["rejections"]],
                "joint": bool(document["joint"])}
    if document.get("not_applicable"):          # G23e: tiers not applicable by design (adapters._assemble)
        manifest["not_applicable"] = [dict(r) for r in document["not_applicable"]]
    return dyn.digest(manifest)


def _entry_key(key, dose):
    return "%d/%d/%d/%d" % (int(key.eval_seed), int(key.env), int(key.episode), int(dose))


def witness_blob(witness):
    """Deterministic gzip + base64 of the canonical JSON of an E4 WitnessResult."""
    text = _dynamic().stamp.canonical(witness).encode("ascii")
    return base64.b64encode(gzip.compress(text, compresslevel=9, mtime=0)).decode("ascii")


def witness_from_blob(blob):
    """Rebuild an E4 WitnessResult from witness_blob (exact float round trip)."""
    dyn = _dynamic()
    data = json.loads(gzip.decompress(base64.b64decode(blob.encode("ascii"))).decode("ascii"))
    trajectory = tuple(dyn.TimedState(s["time_s"], dyn.Pose(**s["pose"]), s["accel_mps2"], s["yaw_rate_rps"])
                       for s in data["trajectory"])
    return dyn.WitnessResult(data["accepted"], data["reason"], trajectory, data["min_clearance_m"],
                             data["sampled_states"], data["rejected_states"], data["search_exhausted"])


def _e4_classes():
    from .._shared.conditions.dynamic import adapters, capacity, feasibility, model, native, visibility
    table = {}
    for module in (adapters, capacity, feasibility, model, native, visibility):
        for name in _E4_CLASSES:
            if hasattr(module, name):
                table[name] = getattr(module, name)
    missing = set(_E4_CLASSES) - set(table)
    if missing:
        raise ImportError("E4 dataclasses missing for the candidate reader: %s" % sorted(missing))
    return table


def _encode_typed(value):
    """Typed, lossless JSON-native form (dataclasses tagged, tuples as lists, non-finite floats tagged)."""
    from dataclasses import fields as _fields, is_dataclass as _is_dataclass
    if _is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _E4_CLASSES:
            raise TypeError("candidate writer refuses dataclass %s" % name)
        return {_TYPE_TAG: name, "f": {f.name: _encode_typed(getattr(value, f.name)) for f in _fields(value) if f.init}}
    if isinstance(value, float) and not math.isfinite(value):
        return {_FLOAT_TAG: repr(value)}
    if isinstance(value, (list, tuple)):
        return [_encode_typed(v) for v in value]
    if isinstance(value, dict):
        raise TypeError("candidate writer supports dataclasses, tuples and JSON scalars only")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("candidate writer cannot encode %r" % (type(value).__name__,))


def _decode_typed(value, classes):
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            text = value[_FLOAT_TAG]
            if text not in ("inf", "-inf", "nan"):
                raise ValueError("bad tagged float %r" % (text,))
            return float(text)
        if set(value) != {_TYPE_TAG, "f"} or value[_TYPE_TAG] not in classes:
            raise ValueError("candidate reader refuses object %r" % (value.get(_TYPE_TAG),))
        cls = classes[value[_TYPE_TAG]]
        return cls(**{k: _decode_typed(v, classes) for k, v in value["f"].items()})
    if isinstance(value, list):
        return tuple(_decode_typed(v, classes) for v in value)
    return value


def candidate_blob(candidate):
    """Deterministic gzip + base64 of the typed canonical JSON of an E4 Candidate (exact float round trip)."""
    text = json.dumps(_encode_typed(candidate), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return base64.b64encode(gzip.compress(text.encode("ascii"), compresslevel=9, mtime=0)).decode("ascii")


def candidate_from_blob(blob):
    data = json.loads(gzip.decompress(base64.b64decode(blob.encode("ascii"))).decode("ascii"))
    return _decode_typed(data, _e4_classes())


def witness_document(pool, *, vessel_key, seed, tier):
    """Companion file: the admitted witness AND candidate of every (key, dose), keyed seed/env/episode/dose.

    G7 item A: the runtime rebuilds each candidate from the env's layout as a consistency check under the
    declared identity quantisation (stamp.identity_digest), then USES the stored candidate, whose exact
    digest is the admitted one; an exact digest of a candidate rebuilt on another machine would depend on
    the last ulp of libm atan2/sin/cos in the nominal plan.
    """
    items = sorted(pool.assignments.items(), key=lambda item: (item[0][0].env, item[0][0].episode, item[0][1]))
    return {"schema": WITNESS_SCHEMA, "task": TASK, "vessel": vessel_key, "seed": int(seed),
            "geometry_tier": int(tier), "pool_digest": pool.pool_digest,
            "witnesses": {_entry_key(k, d): witness_blob(a.witness) for (k, d), a in items},
            "candidates": {_entry_key(k, d): candidate_blob(a.candidate) for (k, d), a in items}}


def pool_document(pool, *, vessel_key, seed, tier, suite, grid, layouts_by_key, tool, witness_file=None):
    """JSON-native manifest of an E4 AdmissionPool (no witness trajectories: they are
    rebuilt deterministically at runtime and must reproduce the digests)."""
    dyn = _dynamic()
    accepted = sorted([[k.eval_seed, k.env, k.episode, k.hazard_group, int(d), a.candidate_digest,
                        a.witness_digest] for (k, d), a in pool.assignments.items()])
    offered = [dict(asdict(k)) for k in _offered_keys(pool, layouts_by_key)]
    rejections = [dyn.stamp.plain(r) for r in pool.rejections]
    document = {
        "schema": POOL_SCHEMA, "task": TASK, "vessel": vessel_key, "seed": int(seed),
        "geometry_tier": int(tier), "grid": dict(grid),
        "geometry_version_digest": plans.GEOMETRY_VERSION_DIGEST,
        "suite": asdict(suite), "joint": bool(pool.joint), "pool_digest": pool.pool_digest,
        "keys_offered": int(pool.keys_offered), "keys_admitted": int(pool.keys_admitted),
        "per_dose_counts": [list(x) for x in pool.per_dose_counts],
        "offered": offered, "accepted": accepted, "rejections": rejections,
        # G23e: tiers not applicable by design (station keeping BlueBoat D1), distinct from rejections
        "not_applicable": [dyn.stamp.plain(r) for r in getattr(pool, "not_applicable", ())],
        "layouts": {"%d/%d" % (k.env, k.episode): h for k, h in layouts_by_key.items()},
        "tool": dict(tool),
        "witness_file": None if witness_file is None else dict(witness_file),
    }
    if pool_manifest_digest(document) != pool.pool_digest:
        raise ValueError("pool manifest does not reproduce the E4 pool digest")
    return document


def _offered_keys(pool, layouts_by_key):
    keys = list(layouts_by_key)
    if len(keys) != pool.keys_offered:
        raise ValueError("every offered key needs its layout hash")
    return keys


# ------------------------------------------------- D16 / D24 not-admitted --
def not_admitted_reasons(rejections, eval_seed, env_index, episode_index):
    """E4 rejection entries of one key as [[dose, reason, sampled_states], ...] (manifest order).

    ``rejections`` are the stored JSON rejection records of a pool manifest
    (E4 ``Rejection`` dataclasses through ``stamp.plain``). The joint echo
    "joint_ladder_member_rejected" (a dose that admitted but was dropped
    because another dose failed) is bookkeeping, not a reason, and is left out.
    """
    key = (int(eval_seed), int(env_index), int(episode_index))
    out = []
    for record in rejections:
        k = record["key"]
        if (int(k["eval_seed"]), int(k["env"]), int(k["episode"])) == key and record["reason"] != JOINT_ECHO:
            out.append([int(record["dose"]), str(record["reason"]), int(record.get("sampled_states", 0))])
    return out


def _plain_key(key):
    return {"eval_seed": int(key.eval_seed), "env": int(key.env), "episode": int(key.episode),
            "hazard_group": str(key.hazard_group)}


class NotAdmittedAssignment:
    """D7 record of a key that runs the dynamic suite at D0 with no intruder (CONTEXT_R4 D16, D24).

    kind ``e4_rejected``: an offered key E4 rejected (reasons = its stored E4
    rejections); kind ``beyond_offered_grid``: a key past the pool's offered
    grid (single reason [0, "beyond_offered_grid", 0]). ``StationD7Pool``
    returns it at EVERY requested dose, so the F0 mixin's reset loads it like
    an admitted assignment; ``StationExposureBridge`` recognises it and never
    hands it to the E4 adapter. ``dose`` is always 0 and ``candidate_digest``
    (= ``record_digest``) depends only on the key, its kind and reasons, the
    stored layout hash and the pool digest, never on the requested dose: the
    D11 ``d7_assignment`` primitive (candidate digest, task, dose, key) is the
    same in every run of the key (D20).
    """

    exposure = NOT_ADMITTED
    candidate = None
    witness = None
    witness_digest = None
    dose = 0

    def __init__(self, key, reasons, *, kind, pool_digest, vessel, geometry_id, requested_dose):
        if kind not in (E4_REJECTED, BEYOND_OFFERED_GRID):
            raise ValueError("unknown not-admitted kind %r" % (kind,))
        reasons = tuple((int(d), str(r), int(n)) for d, r, n in reasons)
        if not reasons or any(r == JOINT_ECHO for _, r, _ in reasons):
            raise ValueError("a not-admitted key carries at least one real reason (no joint echo)")
        if kind == BEYOND_OFFERED_GRID and reasons != ((0, BEYOND_OFFERED_GRID, 0),):
            raise ValueError("a beyond-grid key carries exactly the reason [0, 'beyond_offered_grid', 0]")
        if kind == E4_REJECTED and any(r == BEYOND_OFFERED_GRID for _, r, _ in reasons):
            raise ValueError("'beyond_offered_grid' is not an E4 rejection reason")
        if key.hazard_group != HAZARD_GROUP:
            raise ValueError("not a station-keeping task key: %r" % (key,))
        self.key = key
        self.kind = kind
        self.reasons = reasons
        self.pool_digest = str(pool_digest)
        self.vessel = str(vessel)
        self.geometry_id = None if geometry_id is None else str(geometry_id)
        self.requested_dose = int(requested_dose)
        self.record_digest = _dynamic().digest({
            "schema": NOT_ADMITTED_SCHEMA, "exposure": NOT_ADMITTED, "kind": kind, "task": TASK,
            "key": [int(key.eval_seed), int(key.env), int(key.episode), str(key.hazard_group)],
            "geometry_id": self.geometry_id, "reasons": [list(r) for r in reasons],
            "pool_digest": self.pool_digest})
        self.candidate_digest = self.record_digest

    def __repr__(self):
        return "NotAdmittedAssignment(%r, %s, D%d requested)" % (self.key, self.kind, self.requested_dose)

    def reason_list(self):
        return [list(r) for r in self.reasons]

    @property
    def rule(self):
        return BEYOND_GRID_RULE if self.kind == BEYOND_OFFERED_GRID else NOT_ADMITTED_RULE

    def summary(self):
        """Stamp ``assignment`` block (D20 (b)): dose 0, no intruders, key/task/dose/candidate_digest."""
        return {"key": _plain_key(self.key), "task": TASK, "dose": 0, "exposure": NOT_ADMITTED,
                "not_admitted_kind": self.kind, "requested_dose": self.requested_dose,
                "candidate_digest": self.record_digest, "witness_digest": None, "intruders": [],
                "geometry_id": self.geometry_id, "vessel_id": self.vessel,
                "not_admitted_reasons": self.reason_list(), "pool_digest": self.pool_digest}


class StationD7Pool:
    """D7 pool for journal station keeping, loaded from its JSON manifest.

    Duck-types E4 AdmissionPool for journal_env_mixin (``assignment(key, dose)``,
    ``pool_digest``, ``per_dose_counts``, ``keys_offered``, ``keys_admitted``,
    ``joint``). Every offered key is admitted at all five doses (joint
    admission) or not admitted; ``key_status`` classifies a task key as
    ``admitted``, ``not_admitted`` (offered, E4 rejected it: the reasons are
    DERIVED from the stored E4 rejections, which the pool digest covers, so the
    pinned pool files need no change) or ``beyond_offered_grid`` (CONTEXT_R4
    D24 (a): the run's seed and hazard group, env inside the offered grid,
    episode index >= the offered grid's episodes per env) and raises the
    preflight ValueError for any other key (D24 (b)).

    ``assignment`` of an admitted key needs the REAL layout context of the key
    (``bind_context``, done by the env at reset before journal_reset); it
    rebuilds the candidate with the admission suite and fails closed unless the
    rebuilt candidate equals the stored one under the declared identity
    quantisation (G7 item A: stamp.identity_digest; the stored candidate, whose
    exact digest is the admitted one, is the one that runs) and the stored
    witness reproduces its admitted digest (without a witness file: exact digests
    of the rebuilt candidate and of a re-searched witness, same-machine only). A
    not-admitted key gets a ``NotAdmittedAssignment`` (kind e4_rejected; its
    layout context must be bound too, so the env's layout is checked against
    the stored one); a beyond-grid key gets a ``NotAdmittedAssignment`` (kind
    beyond_offered_grid) without any context (the pool has no record of it,
    so the geometry check is skipped). An invalid dose or a key outside the
    pool raises KeyError (the mixin turns it into the D7 B02 preflight error).

    ``offered_grid`` = {"num_envs": N, "episodes": E} is derived from the
    offered keys, which must be exactly envs 0..N-1 x episodes 0..E-1 (for the
    seed-42 pools E = 3: the two scored episodes plus the overflow episode of
    the round-4 admission, all admitted). ``per_dose_counts`` rows are
    [offered, admitted, not_admitted] (D22); ``e4_per_dose_counts`` keeps E4's
    two columns as stored.
    """

    def __init__(self, document, *, source=None, witnesses=None):
        dyn = _dynamic()
        if document.get("schema") != POOL_SCHEMA or document.get("task") != TASK:
            raise ValueError("not a journal station-keeping D7 pool manifest")
        if document["geometry_version_digest"] != plans.GEOMETRY_VERSION_DIGEST:
            raise ValueError("D7 pool was admitted on another journal geometry version (registry or "
                             "contract changed): re-run admit_d7_pool.py")
        if pool_manifest_digest(document) != document["pool_digest"]:
            raise ValueError("D7 pool manifest digest mismatch (tampered or truncated file)")
        self.document = document
        self.source = source
        self.vessel = str(document["vessel"])
        self.seed = int(document["seed"])
        self.tier = int(document["geometry_tier"])
        self.grid = dict(document["grid"])
        self.suite = dyn.DynamicSuiteConfig(**document["suite"])
        self.pool_digest = str(document["pool_digest"])
        self.joint = bool(document["joint"])
        self.keys_offered = int(document["keys_offered"])
        self.keys_admitted = int(document["keys_admitted"])
        self.e4_per_dose_counts = tuple(tuple(int(v) for v in row) for row in document["per_dose_counts"])
        self._accepted = {}
        for seed, env, episode, group, dose, cand, wit in document["accepted"]:
            self._accepted[(int(seed), int(env), int(episode), str(group), int(dose))] = (str(cand), str(wit))
        self._layouts = {str(k): str(v) for k, v in document["layouts"].items()}
        self._contexts = {}
        self._cache = {}
        self._witnesses = None
        self._candidates = None
        if witnesses is not None:
            if witnesses.get("schema") != WITNESS_SCHEMA or witnesses.get("pool_digest") != self.pool_digest:
                raise ValueError("D7 witness file belongs to another pool (or predates schema %s)" % WITNESS_SCHEMA)
            self._witnesses = dict(witnesses["witnesses"])
            self._candidates = dict(witnesses["candidates"])
        self._classify(document)

    def _classify(self, document):
        """Offered grid, admitted ladders and the not-admitted keys of the stored manifest (fail closed)."""
        offered = set()
        for item in document["offered"]:
            k3 = (int(item["eval_seed"]), int(item["env"]), int(item["episode"]))
            if k3[0] != self.seed or str(item["hazard_group"]) != HAZARD_GROUP or k3 in offered:
                raise ValueError("offered keys must be unique (%d, env, episode, %r) keys" % (self.seed, HAZARD_GROUP))
            offered.add(k3)
        if len(offered) != self.keys_offered:
            raise ValueError("D7 pool key counts disagree with the manifest")
        num_envs = len({k[1] for k in offered})
        episodes = len({k[2] for k in offered})
        if offered != {(self.seed, e, k) for e in range(num_envs) for k in range(episodes)}:
            raise ValueError("offered keys must be the full grid envs 0..N-1 x episodes 0..E-1")
        declared = self.grid.get("episodes_admitted", self.grid.get("episodes"))
        if int(self.grid["num_envs"]) != num_envs or declared is None or int(declared) != episodes:
            raise ValueError("manifest grid %s disagrees with the offered keys (%d envs x %d episodes)"
                             % (self.grid, num_envs, episodes))
        if set(self._layouts) != {"%d/%d" % (k[1], k[2]) for k in offered}:
            raise ValueError("every offered key needs its E5 layout hash")
        # G23e: tiers not applicable by design; every ADMITTED key must carry the same set (a design property); a
        # key rejected before the design could be evaluated may carry none
        na = {}
        for record in document.get("not_applicable") or ():
            k = record["key"]
            k3 = (int(k["eval_seed"]), int(k["env"]), int(k["episode"]))
            dose = int(record["dose"])
            if (str(k["hazard_group"]) != HAZARD_GROUP or k3 not in offered or record["reason"] != NOT_APPLICABLE
                    or not 1 <= dose <= 4 or dose in na.get(k3, set())):
                raise ValueError("a not-applicable record must name an offered key, a tier D1..D4 once, and the "
                                 "reason %r" % NOT_APPLICABLE)
            na.setdefault(k3, set()).add(dose)
        ladders = {}
        for seed, env, episode, group, dose in self._accepted:
            if group != HAZARD_GROUP or (seed, env, episode) not in offered:
                raise ValueError("an admitted assignment lies outside the offered keys")
            ladders.setdefault((seed, env, episode), set()).add(dose)
        na_sets = {frozenset(na.get(k3, ())) for k3 in ladders} if ladders else {frozenset(v) for v in na.values()}
        if len(na_sets) > 1:
            raise ValueError("not-applicable tiers differ between offered keys (they are a design property)")
        self.not_applicable_doses = tuple(sorted(next(iter(na_sets)))) if na_sets else ()
        if any(not set(doses) <= set(self.not_applicable_doses) for doses in na.values()):
            raise ValueError("not-applicable tiers differ between offered keys (they are a design property)")
        applicable = set(range(5)) - set(self.not_applicable_doses)
        if any(doses != applicable for doses in ladders.values()):
            raise ValueError("an admitted key must carry the whole D0..D4 ladder (joint admission, D16; G23e: "
                             "less the tiers not applicable by design)")
        if len(ladders) != self.keys_admitted:
            raise ValueError("D7 pool key counts disagree with the manifest")
        for record in document["rejections"]:
            k = record["key"]
            k3 = (int(k["eval_seed"]), int(k["env"]), int(k["episode"]))
            if str(k["hazard_group"]) != HAZARD_GROUP or k3 not in offered or k3 in ladders:
                raise ValueError("an E4 rejection must belong to an offered key that is not admitted")
        self._offered = frozenset(offered)
        self._not_admitted = {}
        for k3 in sorted(offered - set(ladders)):
            reasons = not_admitted_reasons(document["rejections"], *k3)
            if not reasons:
                raise ValueError("not-admitted key %s carries no E4 rejection reason" % (k3,))
            self._not_admitted[k3] = reasons
        self.keys_not_admitted = len(self._not_admitted)
        if self.keys_admitted + self.keys_not_admitted != self.keys_offered:
            raise ValueError("offered keys must be admitted or not admitted")
        excluded = {dose: sum(1 for doses in na.values() if dose in doses) for dose in self.not_applicable_doses}
        if len(self.e4_per_dose_counts) != 5 or any(
                row != ((self.keys_offered - excluded[dose], 0) if dose in excluded
                        else (self.keys_offered, self.keys_admitted))
                for dose, row in enumerate(self.e4_per_dose_counts)):
            raise ValueError("per-dose counts disagree with the joint admission")
        self.offered_grid = {"num_envs": num_envs, "episodes": episodes}
        # G23e: a tier not applicable by design publishes (0, 0, 0): it is never offered to a run
        self.per_dose_counts = tuple((0, 0, 0) if dose in excluded else (o, a, o - a)
                                     for dose, (o, a) in enumerate(self.e4_per_dose_counts))

    @classmethod
    def load(cls, path, sha256=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("D7 pool manifest %s is missing (run work/tools_f4/admit_d7_pool.py)" % path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if sha256 is not None and digest != sha256:
            raise ValueError("D7 pool manifest %s SHA-256 %s differs from the pinned %s" % (path, digest, sha256))
        document = json.loads(raw.decode("ascii"))
        witnesses = None
        entry = document.get("witness_file")
        if entry is not None:
            witness_path = path.parent / entry["name"]
            if not witness_path.is_file():
                raise FileNotFoundError("D7 witness file %s named by the manifest is missing" % witness_path)
            blob = witness_path.read_bytes()
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                raise ValueError("D7 witness file %s SHA-256 differs from the manifest" % witness_path)
            witnesses = json.loads(blob.decode("ascii"))
        return cls(document, source={"path": str(path), "sha256": digest}, witnesses=witnesses)

    def check_env(self, *, vessel, seed, tier, num_envs, suite=None):
        """Setup-time preflight: the pool covers this vessel, seed, tier and grid width."""
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("journal cfgs must carry an integer evaluation seed (cfg.seed; CONTEXT_R4 D7)")
        if vessel != self.vessel or int(seed) != self.seed or int(tier) != self.tier:
            raise ValueError("D7 pool (%s, seed %d, tier %d) does not cover the run (%s, seed %s, tier %s); "
                             "admit a pool for it offline" % (self.vessel, self.seed, self.tier, vessel, seed, tier))
        if int(num_envs) > int(self.offered_grid["num_envs"]):
            raise ValueError("D7 pool covers %d envs, the run has %d" % (self.offered_grid["num_envs"], num_envs))
        if suite is not None:
            want = {k: v for k, v in asdict(self.suite).items() if k not in ("enabled", "dose")}
            have = {k: v for k, v in asdict(suite).items() if k not in ("enabled", "dose")}
            if want != have:
                raise ValueError("runtime D7 suite parameters differ from the admitted pool's suite")
            if int(suite.dose) in self.not_applicable_doses:          # G23e: fail closed before the run
                raise ValueError("D%d is not applicable by design for station keeping %s (%s); run D%s"
                                 % (int(suite.dose), self.vessel, NOT_APPLICABLE_RULE,
                                    "/D".join(str(d) for d in range(1, 5) if d not in self.not_applicable_doses)))

    def key_status(self, key):
        """"admitted" | "not_admitted" | "beyond_offered_grid" of a task key (CONTEXT_R4 D16, D24).

        Any other key (another seed or hazard group, an env outside the offered
        grid, a negative index) is a preflight ValueError (D24 (b)).
        """
        seed, env, episode, group = int(key.eval_seed), int(key.env), int(key.episode), str(key.hazard_group)
        k3 = (seed, env, episode)
        if group == HAZARD_GROUP and k3 in self._offered:
            return NOT_ADMITTED if k3 in self._not_admitted else ADMITTED
        if (group == HAZARD_GROUP and seed == self.seed and 0 <= env < self.offered_grid["num_envs"]
                and episode >= self.offered_grid["episodes"]):
            return BEYOND_OFFERED_GRID
        raise ValueError("D7 pool (%s, seed %d, group %s, offered grid %d envs x %d episodes) holds no key "
                         "(seed %d, env %d, episode %d, group %s) and the key is not beyond the offered grid: "
                         "preflight error (CONTEXT_R4 D24 (b); admit a pool for this seed/grid offline)"
                         % (self.vessel, self.seed, HAZARD_GROUP, self.offered_grid["num_envs"],
                            self.offered_grid["episodes"], seed, env, episode, group))

    def not_admitted_reasons(self, env_index, episode_index):
        """Stored E4 reasons of an offered not-admitted key ([] for an admitted key)."""
        return [list(r) for r in self._not_admitted.get((self.seed, int(env_index), int(episode_index)), ())]

    def layout_hash_for(self, env_index, episode_index):
        return self._layouts.get("%d/%d" % (int(env_index), int(episode_index)))

    def bind_context(self, key, context):
        """Bind the env's REAL layout context of an OFFERED key (the stored layout hash must match).

        A key without a stored layout (beyond the offered grid, or outside the
        pool) raises KeyError: the env skips the bind for beyond-grid keys.
        """
        admitted = self.layout_hash_for(key.env, key.episode)
        if admitted is None:
            raise KeyError(key)
        if admitted != context.geometry_id:
            raise ValueError("env layout %s for %s differs from the admitted layout %s"
                             % (context.geometry_id, key, admitted))
        self._contexts[key] = context

    def assignment(self, key, dose):
        """Admitted E4 Assignment, or NotAdmittedAssignment (D16 / D24); KeyError outside the pool."""
        dyn = _dynamic()
        adapters = dyn.adapters
        dose = int(dose)
        if not 0 <= dose <= 4:
            raise KeyError((key, dose))
        try:
            status = self.key_status(key)
        except ValueError:
            raise KeyError((key, dose)) from None
        if status == BEYOND_OFFERED_GRID:
            return NotAdmittedAssignment(key, [[0, BEYOND_OFFERED_GRID, 0]], kind=BEYOND_OFFERED_GRID,
                                         pool_digest=self.pool_digest, vessel=self.vessel, geometry_id=None,
                                         requested_dose=dose)
        if status == NOT_ADMITTED:
            if key not in self._contexts:
                raise ValueError("bind the env's real layout context for %s before requesting its assignment"
                                 % (key,))
            return NotAdmittedAssignment(key, self._not_admitted[(key.eval_seed, key.env, key.episode)],
                                         kind=E4_REJECTED, pool_digest=self.pool_digest, vessel=self.vessel,
                                         geometry_id=self.layout_hash_for(key.env, key.episode),
                                         requested_dose=dose)
        if dose in self.not_applicable_doses:                         # G23e
            raise KeyError("D%d is not applicable by design for %s (%s)" % (dose, key, NOT_APPLICABLE))
        entry = self._accepted.get((key.eval_seed, key.env, key.episode, key.hazard_group, dose))
        if entry is None:
            raise KeyError((key, dose))
        cache_key = (key, dose)
        if cache_key in self._cache:
            return self._cache[cache_key]
        context = self._contexts.get(key)
        if context is None:
            raise ValueError("bind the env's real layout context for %s before requesting its assignment" % (key,))
        candidate = dyn.build_candidate(context, key, dose, self.suite.at_dose(dose))
        cand_digest, wit_digest = entry
        stored = None if self._candidates is None else self._candidates.get(_entry_key(key, dose))
        if stored is not None:
            # G7 item A: rebuilt-vs-admitted under the declared identity quantisation; the stored candidate
            # (exact admitted digest) is the one that runs and is stamped.
            admitted = candidate_from_blob(stored)
            if dyn.digest(admitted) != cand_digest:
                raise ValueError("stored D7 candidate for %s D%d does not reproduce its admitted digest" % (key, dose))
            if dyn.stamp.identity_digest(candidate) != dyn.stamp.identity_digest(admitted):
                raise ValueError("rebuilt D7 candidate for %s D%d differs from the admitted one beyond the identity "
                                 "quantisation" % (key, dose))
            candidate = admitted
        elif dyn.digest(candidate) != cand_digest:
            raise ValueError("rebuilt D7 candidate for %s D%d differs from the admitted one" % (key, dose))
        blob = None if self._witnesses is None else self._witnesses.get(_entry_key(key, dose))
        if blob is not None:
            witness, reason = witness_from_blob(blob), "stored witness"
        else:
            witness, reason, _ = adapters._witness_item(candidate)
        if witness is None or not witness.accepted or adapters._witness_digest(witness) != wit_digest:
            raise ValueError("rebuilt D7 witness for %s D%d differs from the admitted one (%s)"
                             % (key, dose, reason))
        result = dyn.Assignment(candidate, witness, cand_digest, wit_digest, self.pool_digest)
        self._cache = {k: v for k, v in self._cache.items() if k[0].env != key.env}
        self._cache[cache_key] = result
        return result

    def reason_counts(self):
        """{E4 reason class (text before ':'): dose entries} over the not-admitted keys."""
        counts = {}
        for reasons in self._not_admitted.values():
            for _, reason, _ in reasons:
                name = reason.split(":")[0]
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def header_fields(self):
        """D7 header additions: D22 count triples, the D24 offered grid and the D16 not-admitted keys."""
        return {"per_dose_counts": [list(r) for r in self.per_dose_counts],
                "per_dose_counts_columns": list(PER_DOSE_COLUMNS),
                "e4_per_dose_counts": [list(r) for r in self.e4_per_dose_counts],
                "keys_not_admitted": self.keys_not_admitted,
                "not_admitted": [[env, episode, reasons[0][1].split(":")[0]]
                                 for (_, env, episode), reasons in sorted(self._not_admitted.items())],
                "not_admitted_reason_counts": self.reason_counts(),
                "grid": dict(self.offered_grid),
                "pool_file_sha256": None if self.source is None else self.source["sha256"],
                "not_admitted_policy": NOT_ADMITTED_RULE,
                "beyond_offered_grid_policy": BEYOND_GRID_RULE,
                "not_applicable_doses": list(self.not_applicable_doses),   # G23e
                "not_applicable_policy": NOT_APPLICABLE_RULE}


class StationExposureBridge:
    """D16 / D24 wrapper of the E4 ``HazardBridge`` (the interface the F0 mixin calls).

    Admitted rows go to the E4 bridge unchanged (same samples, swept contact
    and E4 stamp). A row reset with a ``NotAdmittedAssignment`` runs the
    dynamic suite at D0: no D7 object (empty samples, the reserved slot stays
    unpublished), no contact, the tick clock checked as E4 does; ``finish``
    returns a D7 stamp envelope {"payload", "sha256"} (E4 schema / model
    version and canonical digest) in the D20 form:

        payload["exposure"] = payload["runtime"]["exposure"]
                            = payload["assignment"]["exposure"] = "not_admitted"
        payload["assignment"] = NotAdmittedAssignment.summary() (dose 0, no
                                intruders, key, task, dose-independent candidate_digest)
        payload["not_admitted_reasons"] = payload["not_admitted"]["reasons"]
                            = [[dose, reason, sampled_states], ...]  (E4 reasons, or
                              exactly [[0, "beyond_offered_grid", 0]])
    """

    def __init__(self, inner):
        self.inner = inner
        self.adapter = inner.adapter
        self.cfg = inner.cfg
        self.dt = inner.dt
        self.own_speed_bound = inner.own_speed_bound
        self._empty = {}          # env -> row of the CURRENT not-admitted episode
        self._mode = {}           # env -> "admitted" | "not_admitted" (current episode)

    @property
    def rows(self):
        out = {}
        for env, mode in self._mode.items():
            if mode == NOT_ADMITTED:
                out[env] = self._empty[env]
            elif env in self.inner.rows:
                out[env] = self.inner.rows[env]
        return out

    def exposure(self, env):
        return self._mode.get(int(env))

    def current_record(self, env):
        """The NotAdmittedAssignment of the env's current episode (None when admitted)."""
        return self._empty[int(env)]["record"] if self._mode.get(int(env)) == NOT_ADMITTED else None

    def reset_rows(self, env_ids, assignments):
        ids = tuple(int(i) for i in env_ids)
        current = self.rows
        for env in ids:
            if env in current and not current[env]["finished"]:
                raise ValueError("finish and stamp every assigned episode before reset")
        empty = {e: assignments[e] for e in ids if isinstance(assignments[e], NotAdmittedAssignment)}
        admitted = tuple(e for e in ids if e not in empty)
        for env, record in empty.items():
            if int(record.key.env) != env or record.key.hazard_group != HAZARD_GROUP:
                raise ValueError("wrong task or environment for a not-admitted record")
        if admitted:
            self.inner.reset_rows(admitted, {e: assignments[e] for e in admitted})
        for env in ids:
            if env in empty:
                self._empty[env] = {"record": empty[env], "last_tick": -1, "cache": None, "contact": False,
                                    "swept_intervals": 0, "observer_updates": 0, "finished": False,
                                    "stamp": None}
                self._mode[env] = NOT_ADMITTED
            else:
                self._mode[env] = ADMITTED

    def step(self, ticks, poses):
        Pose = _dynamic().Pose
        admitted = {e: t for e, t in ticks.items() if self._mode.get(e) != NOT_ADMITTED}
        out = self.inner.step(admitted, poses) if admitted else {}
        for env, tick in ticks.items():
            if env in admitted:
                continue
            row = self._empty[env]
            if row["finished"]:
                raise ValueError("row finished; reset it before stepping")
            if type(tick) is not int or tick < 0:
                raise ValueError("tick must be a nonnegative integer")
            if tick == row["last_tick"]:
                out[env] = row["cache"]
                continue
            if tick < row["last_tick"]:
                raise ValueError("physics tick went backwards without a reset")
            if not isinstance(poses[env], Pose):
                raise TypeError("pose must be a Pose")
            if row["last_tick"] >= 0:
                row["swept_intervals"] += 1
            row["observer_updates"] += 1
            row["last_tick"] = tick
            row["cache"] = ()
            out[env] = ()
        return out

    def contact(self, env):
        if self._mode.get(env) == NOT_ADMITTED:
            return False
        return self.inner.contact(env)

    def finish(self, env, termination_reason):
        if self._mode.get(env) != NOT_ADMITTED:
            return self.inner.finish(env, termination_reason)
        dyn = _dynamic()
        if termination_reason not in dyn.TERMINATION_REASONS:
            raise ValueError("explicit terminal reason required: " + repr(termination_reason))
        row = self._empty[env]
        if row["finished"]:
            return row["stamp"]
        row["stamp"] = not_admitted_stamp(row["record"], env, termination_reason, row, self.dt)
        row["finished"] = True
        return row["stamp"]


def not_admitted_stamp(record, env, termination_reason, row, physics_dt_s):
    """D7 stamp envelope of one finished not-admitted episode (E4 digest rule; D20 form)."""
    from .._shared.conditions.dynamic import stamp as e4_stamp
    reasons = record.reason_list()
    runtime = {
        "assigned": False, "exposure": NOT_ADMITTED, "itt_count": 0, "env": int(env),
        "termination_reason": termination_reason, "suite_dose": record.requested_dose, "run_dose": 0,
        "last_time_s": max(0.0, row["last_tick"] * physics_dt_s),
        "first_detection_s": None, "realized_ttc_at_first_detection_s": None,
        "detected_samples": 0, "observer_updates": row["observer_updates"],
        "min_sampled_surface_clearance_m": None, "sampled_contact": False,
        "non_exposure_reason": NOT_ADMITTED, "not_admitted_kind": record.kind,
        "bridge": {"swept_contact_latched": False, "first_swept_contact_interval_start_s": None,
                   "swept_intervals": row["swept_intervals"], "last_tick": row["last_tick"],
                   "physics_dt_s": physics_dt_s,
                   "d7_rows": "none: not admitted, run at D0 without intruder (CONTEXT_R4 D16/D24)"},
    }
    payload = {"schema_version": e4_stamp.SCHEMA_VERSION, "model_version": e4_stamp.MODEL_VERSION,
               "exposure": NOT_ADMITTED, "not_admitted_reasons": reasons,
               "not_admitted": {"kind": record.kind, "reasons": reasons, "record_digest": record.record_digest,
                                "pool_digest": record.pool_digest, "requested_dose": record.requested_dose,
                                "rule": record.rule},
               "assignment": e4_stamp.plain(record.summary()), "runtime": e4_stamp.plain(runtime),
               "semantics": dict(e4_stamp.SEMANTICS)}
    return json.loads(e4_stamp.canonical({"payload": payload, "sha256": e4_stamp.digest(payload)}))


def record_exposure(envelope):
    """"not_admitted" for a D16/D24 record, "admitted" for an E4 stamp (verifies the digest first)."""
    from .._shared.conditions.certificate import verify_dynamic_hazard_stamp
    return verify_dynamic_hazard_stamp(envelope)["payload"].get("exposure", ADMITTED)


# --------------------------------------------------------- offline admission --
def admit_station_grid(vessel_key, tier, seed, keys, *, suite=None, workers=1, context_fn=None):
    """Offline joint D7 admission of (env, episode) keys on their OWN E5 layouts (D7 B05, D16).

    ONE E4 ``admit_grid(joint=True)`` pass over the TaskContexts of the keys'
    journal layouts (the same function the env runs; no redraw, no retry):
    every key ends admitted (D0..D4) or not admitted with E4's rejection
    records. Returns (E4 AdmissionPool, {EpisodeKey: E5 layout hash} in key
    order, suite). ``context_fn(layout, vessel_key, key)`` replaces
    ``station_task_context`` (tests only: forcing a REAL E4 rejection; the
    context must keep the layout's geometry id).
    """
    dyn = _dynamic()
    suite = suite if suite is not None else dyn.DynamicSuiteConfig(enabled=True, vessel_id=vessel_key)
    jobs, layouts_by_key = [], {}
    for env, episode in keys:
        layout = build_station_layout(tier, vessel_key, seed, env, episode)
        key = episode_key(seed, env, episode)
        context = (station_task_context(layout, vessel_key) if context_fn is None
                   else context_fn(layout, vessel_key, key))
        if context.geometry_id != layout_hash(layout):
            raise ValueError("the admission context of %s must keep the E5 layout geometry id" % (key,))
        layouts_by_key[key] = layout_hash(layout)
        jobs.append((context, key, suite))
    pool = dyn.admit_grid(jobs, joint=True, workers=workers)
    return pool, layouts_by_key, suite


def pool_from_admission(pool, layouts_by_key, *, vessel_key, seed, tier, suite, grid, tool=None):
    """In-memory StationD7Pool (manifest + witnesses) of an admit_station_grid result."""
    witnesses = witness_document(pool, vessel_key=vessel_key, seed=seed, tier=tier)
    document = pool_document(pool, vessel_key=vessel_key, seed=seed, tier=tier, suite=suite, grid=grid,
                             layouts_by_key=layouts_by_key, tool=dict(tool or {"name": "admit_station_grid"}))
    return StationD7Pool(json.loads(json.dumps(document)), witnesses=witnesses)


# -------------------------------------------------------------- math utils --
_SPLIT = 134217729.0   # 2**27 + 1 (Veltkamp split)


def _two_prod(a, b):
    """Exact product error (Dekker): equals fma(a, b, -a*b) without overflow/underflow."""
    p = a * b
    ca = _SPLIT * a
    ah = ca - (ca - a)
    al = a - ah
    cb = _SPLIT * b
    bh = cb - (cb - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _fast_sum(a, b):
    x = a + b
    z = x - a
    return x, b - z


def py_hypot(x, y):
    """float64 torch replica of CPython's math.hypot(x, y) (Modules/mathmodule.c vector_norm).

    Bit-identical for finite normal inputs (test-asserted against math.hypot);
    0 for (0, 0), inf if either is infinite, nan otherwise if either is nan.
    """
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    mx = torch.maximum(x.abs(), y.abs())
    safe = torch.where(mx > 0, mx, torch.ones_like(mx))
    safe = torch.where(torch.isfinite(safe), safe, torch.ones_like(safe))
    _, exponent = torch.frexp(safe)
    scale = torch.ldexp(torch.ones_like(mx), -exponent)
    csum = torch.ones_like(mx)
    frac1 = torch.zeros_like(mx)
    frac2 = torch.zeros_like(mx)
    for value in (x, y):
        value = value * scale
        hi, lo = _two_prod(value, value)
        csum, low = _fast_sum(csum, hi)
        frac1 = frac1 + lo
        frac2 = frac2 + low
    h = torch.sqrt(csum - 1.0 + (frac1 + frac2))
    hi, lo = _two_prod(-h, h)
    csum, low = _fast_sum(csum, hi)
    frac1 = frac1 + lo
    frac2 = frac2 + low
    h = h + (csum - 1.0 + (frac1 + frac2)) / (2.0 * h)
    out = h / scale
    out = torch.where(mx == 0, torch.zeros_like(out), out)
    nan = torch.isnan(x) | torch.isnan(y)
    inf = torch.isinf(x) | torch.isinf(y)
    out = torch.where(nan, torch.full_like(out, math.nan), out)
    return torch.where(inf, torch.full_like(out, math.inf), out)


def rotation_matrix(quat_wxyz):
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)


def bow_axis_xy(quat_wxyz):
    """Unit bow direction (hull +x) projected on world XY, [N,2]."""
    r = rotation_matrix(quat_wxyz)
    fwd = torch.stack((r[:, 0, 0], r[:, 1, 0]), -1)
    return fwd / torch.norm(fwd, dim=-1, keepdim=True).clamp(min=1.0e-6)


def yaw_from_quat(quat_wxyz):
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw):
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), -1)


def body_rates(quat_wxyz, com_velocity_w):
    """Raw [surge m/s, sway m/s, yaw rate rad/s] in hull axes from the full 3-D transform (D3 s4)."""
    r = rotation_matrix(quat_wxyz.to(com_velocity_w.dtype))
    rt = r.transpose(-1, -2)
    v_b = (rt @ com_velocity_w[:, :3].unsqueeze(-1)).squeeze(-1)
    w_b = (rt @ com_velocity_w[:, 3:6].unsqueeze(-1)).squeeze(-1)
    return torch.stack((v_b[:, 0], v_b[:, 1], w_b[:, 2]), -1)


def ray_angles(ray_count, device=None, dtype=torch.float32):
    """Ray bearings relative to the bow, counter-clockwise (hazard_nav convention)."""
    return torch.arange(int(ray_count), device=device, dtype=dtype) * (2.0 * math.pi / int(ray_count))


def point_return_rays(com_xy_w, yaw, centers_w, radii, visible, ray_count, max_range_m):
    """[N, ray_count] ranges (m): E4 visibility.py point-return proxy.

    For every VISIBLE object the ray nearest to the bearing of its centre returns
    max(0, |centre - COM| - radius) (range to the nearest surface point,
    edge_range_and_occlusion); every other ray returns max_range_m. Objects
    beyond max range are not returned.
    """
    n = com_xy_w.shape[0]
    count = int(ray_count)
    out = torch.full((n, count), float(max_range_m), dtype=com_xy_w.dtype, device=com_xy_w.device)
    if centers_w.numel() == 0:
        return out
    rel = centers_w - com_xy_w.unsqueeze(1)
    distance = torch.norm(rel, dim=-1)
    edge = torch.clamp(distance - radii, min=0.0)
    bearing = torch.atan2(rel[..., 1], rel[..., 0]) - yaw.unsqueeze(1)
    bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
    index = torch.remainder(torch.round(bearing / (2.0 * math.pi / count)).to(torch.long), count)
    hit = visible & (edge <= float(max_range_m))
    for slot in range(centers_w.shape[1]):
        rows = hit[:, slot].nonzero().flatten()
        if rows.numel():
            cols = index[rows, slot]
            out[rows, cols] = torch.minimum(out[rows, cols], edge[rows, slot].to(out.dtype))
    return out


# -------------------------------------------------------------- tensor twin --
RUNNING, SUCCESS, COLLISION, TIMEOUT, OUT_OF_BOUNDS, FAILURE = 0, 1, 2, 3, 4, 5
CAPSIZE, SINKING, AIRBORNE, NONFINITE = 6, 7, 8, 9


class StationKeepingTwin:
    """Vectorised twin of journal_contracts_v2.StationKeepingEpisode (bit for bit).

    One row per env, float64 throughout; same operation order as the scalar
    oracle (CPython math.hypot replicated by py_hypot; the final-window
    metrics are accumulated interval by interval in the scalar's order).
    Lifecycle: reset(ids, ...) -> observe_physics(state) every physics substep
    -> step(time_s, state, collision, out_of_bounds) every control step; a row
    that completes latches its result (``completed_result(i)``, the
    EpisodeResult.evaluator_fields() equivalent) until its next completion.
    Rows already done are left untouched by observe_physics/step (the scalar
    oracle ignores observe_physics and refuses step; the env resets them in the
    same control step).
    """

    FW_FIELDS = ("observed", "squared", "inside", "run", "longest")

    def __init__(self, num_envs, device, *, capsize_angle_rad, min_heave_m, max_heave_m):
        limits = contracts.ValidityLimits(capsize_angle_rad=capsize_angle_rad, min_heave_m=min_heave_m,
                                          max_heave_m=max_heave_m)
        self.limits = limits
        self.n = int(num_envs)
        self.device = torch.device(device)
        self.last_any_finished = True       # G11: host copy of the last step's finished.any() (True = unknown)
        f = dict(dtype=torch.float64, device=self.device)
        b = dict(dtype=torch.bool, device=self.device)
        n = self.n
        self.capsize = float(limits.capsize_angle_rad)
        self.min_heave = float(limits.min_heave_m)
        self.max_heave = float(limits.max_heave_m)
        # frozen episode parameters
        self.station = torch.zeros((n, 2), **f)
        self.radius = torch.ones(n, **f)
        self.deadline = torch.ones(n, **f)
        self.hold_s = torch.ones(n, **f)
        self.final_window = torch.ones(n, **f)
        self.late_start = torch.zeros(n, **f)
        self.late_end = torch.ones(n, **f)
        self.top_dose_start = torch.zeros(n, **f)
        # base state
        self.reason = torch.zeros(n, dtype=torch.long, device=self.device)
        self.nonfinite_seen = torch.zeros(n, **b)
        self.done = torch.ones(n, **b)          # no episode until reset
        self.success = torch.zeros(n, **b)
        self.clean = torch.ones(n, **b)
        self.contact_since_step = torch.zeros(n, **b)
        self.prev_xy = torch.zeros((n, 2), **f)
        self.prev_time = torch.zeros(n, **f)
        self.full_path = torch.zeros(n, **f)
        self.prefix_path = torch.full((n,), math.nan, **f)
        self.tts = torch.full((n,), math.nan, **f)
        self.valid_prefix_only = torch.zeros(n, **b)
        self.min_clearance = torch.full((n,), math.inf, **f)
        # station state
        self.has_hold = torch.zeros(n, **b)
        self.hold_started = torch.zeros(n, **f)
        self.previous_error = torch.zeros(n, **f)
        self.previous_inside = torch.zeros(n, **b)
        self.longest_hold = torch.zeros(n, **f)
        self.fw = {name: torch.zeros(n, **f) for name in self.FW_FIELDS}
        # completed-episode latch (survives reset, like last_completed)
        self.has_completed = torch.zeros(n, **b)
        self.c_success = torch.zeros(n, **b)
        self.c_tts = torch.full((n,), math.nan, **f)
        self.c_prefix = torch.full((n,), math.nan, **f)
        self.c_full = torch.zeros(n, **f)
        self.c_reason = torch.zeros(n, dtype=torch.long, device=self.device)
        self.c_valid_prefix_only = torch.zeros(n, **b)
        self.c_nonfinite = torch.zeros(n, **b)
        self.c_min_clearance = torch.full((n,), math.inf, **f)
        self.c_longest_hold = torch.zeros(n, **f)
        self.c_fw = {name: torch.zeros(n, **f) for name in self.FW_FIELDS}
        self.c_params = [None] * n
        self._row_params = [None] * n

    # ------------------------------------------------------------ helpers --
    def _ids(self, env_ids):
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

    def _validity(self, state):
        """(codes 0/6/7/8/9, finite mask) = contracts.validity_reason per row."""
        com = state.com_xyz.to(torch.float64)
        q = state.quat_wxyz.to(torch.float64)
        lin = state.lin_vel.to(torch.float64)
        ang = state.ang_vel.to(torch.float64)
        water = torch.as_tensor(state.water_level, dtype=torch.float64, device=self.device).expand(self.n)
        nominal = torch.as_tensor(state.nominal_height, dtype=torch.float64, device=self.device).expand(self.n)
        q0, q1, q2, q3 = q.unbind(-1)
        norm = torch.sqrt(((q0 * q0 + q1 * q1) + q2 * q2) + q3 * q3)
        finite = (torch.isfinite(com).all(-1) & torch.isfinite(q).all(-1) & torch.isfinite(lin).all(-1)
                  & torch.isfinite(ang).all(-1) & torch.isfinite(water) & torch.isfinite(nominal)
                  & torch.isfinite(norm) & (norm > 1.0e-12))
        safe_norm = torch.where(finite, norm, torch.ones_like(norm))
        x = torch.where(finite, q1 / safe_norm, torch.zeros_like(q1))
        y = torch.where(finite, q2 / safe_norm, torch.zeros_like(q2))
        tilt = torch.acos(torch.clamp(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        heave = torch.where(finite, (com[:, 2] - water) - nominal, torch.zeros_like(norm))
        code = torch.zeros(self.n, dtype=torch.long, device=self.device)
        code = torch.where(heave > self.max_heave, torch.full_like(code, AIRBORNE), code)
        code = torch.where(heave < self.min_heave, torch.full_like(code, SINKING), code)
        code = torch.where(tilt > self.capsize, torch.full_like(code, CAPSIZE), code)
        code = torch.where(~finite, torch.full_like(code, NONFINITE), code)
        # a code already decided by a higher-priority cause must not be overwritten:
        # scalar order = nonfinite > capsize > sinking > airborne (the where chain above
        # applies the lowest priority first, so the last write is the highest priority)
        return code, finite, com

    def _error(self, com):
        return py_hypot(com[:, 0] - self.station[:, 0], com[:, 1] - self.station[:, 1])

    # -------------------------------------------------------------- reset --
    def reset(self, env_ids, com_xy, station_xy, radius_m, deadline_s, *, hold_s, late_interval_s,
              top_dose_start_s, final_window_s):
        """Start episodes; arguments are per reset row (sequences / tensors).

        Every row is validated by constructing the scalar oracle with the same
        arguments (identical ValueErrors).
        """
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        k = ids.numel()

        def rows(value, width=None):
            t = torch.as_tensor(value, dtype=torch.float64).to(self.device)
            if width is None:
                return t.reshape(-1).expand(k) if t.numel() == 1 else t.reshape(k)
            return t.reshape(k, width)

        com = rows(com_xy, 2)
        station = rows(station_xy, 2)
        radius = rows(radius_m)
        deadline = rows(deadline_s)
        hold = rows(hold_s)
        late = rows(late_interval_s, 2)
        top = rows(top_dose_start_s)
        window = rows(final_window_s)
        for j in range(k):
            params = dict(station_xy=tuple(station[j].tolist()), radius_m=float(radius[j]),
                          deadline_s=float(deadline[j]), hold_s=float(hold[j]),
                          late_interval_s=tuple(late[j].tolist()), top_dose_start_s=float(top[j]),
                          final_window_s=float(window[j]))
            contracts.StationKeepingEpisode(validity_limits=self.limits, **params)   # validation only
            self._row_params[int(ids[j])] = params
        self.station[ids] = station
        self.radius[ids] = radius
        self.deadline[ids] = deadline
        self.hold_s[ids] = hold
        self.late_start[ids] = late[:, 0]
        self.late_end[ids] = late[:, 1]
        self.top_dose_start[ids] = top
        self.final_window[ids] = window
        self.reason[ids] = RUNNING
        self.nonfinite_seen[ids] = False
        self.done[ids] = False
        self.success[ids] = False
        self.clean[ids] = True
        self.contact_since_step[ids] = False
        self.prev_xy[ids] = com
        self.prev_time[ids] = 0.0
        self.full_path[ids] = 0.0
        self.prefix_path[ids] = math.nan
        self.tts[ids] = math.nan
        self.valid_prefix_only[ids] = False
        self.min_clearance[ids] = math.inf
        error = py_hypot(com[:, 0] - station[:, 0], com[:, 1] - station[:, 1])
        inside = error <= radius
        self.has_hold[ids] = inside
        self.hold_started[ids] = 0.0
        self.previous_error[ids] = error
        self.previous_inside[ids] = inside
        self.longest_hold[ids] = 0.0
        for t in self.fw.values():
            t[ids] = 0.0

    # ---------------------------------------------------- observe_physics --
    def observe_physics(self, state, collision=None):
        """EpisodeBase.observe_physics + the station override, every live row."""
        act = ~self.done
        hit = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if collision is None
               else torch.as_tensor(collision, dtype=torch.bool, device=self.device).reshape(self.n))
        code, finite, com = self._validity(state)
        self._observe_rows(act, code, finite, com, hit)

    def _observe_rows(self, act, code, finite, com, hit):
        self.nonfinite_seen |= act & (code == NONFINITE)
        self.reason = torch.where(act & (self.reason == RUNNING), code, self.reason)
        self.contact_since_step |= act & hit
        error = self._error(torch.where(finite.unsqueeze(-1), com, torch.zeros_like(com)))
        inside = act & finite & (self.reason == RUNNING) & ~hit & (error <= self.radius)
        reset_hold = act & ~inside
        self.has_hold &= ~reset_hold
        self.previous_inside &= ~reset_hold

    # --------------------------------------------------------------- step --
    def step(self, time_s, state, collision=None, out_of_bounds=None):
        """One control sample; returns int64 codes [N] (0 = running or not live)."""
        act = ~self.done
        t = torch.as_tensor(time_s, dtype=torch.float64, device=self.device).reshape(-1).expand(self.n)
        hit = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if collision is None
               else torch.as_tensor(collision, dtype=torch.bool, device=self.device).reshape(self.n))
        oob = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if out_of_bounds is None
               else torch.as_tensor(out_of_bounds, dtype=torch.bool, device=self.device).reshape(self.n))
        # G11: both entry checks travel to the host in ONE transfer (same checks, same order of the errors)
        checks = torch.stack(((act & ~(torch.isfinite(t) & (t > self.prev_time))).any(),
                              (act & (t > self.deadline + 1.0e-9)).any())).tolist()
        if checks[0]:
            raise ValueError("sample clock must increase from the reset time zero")
        if checks[1]:
            raise ValueError("the adapter must include a sample exactly at the deadline")
        previous_time = self.prev_time.clone()
        previous_error = self.previous_error.clone()
        code, finite, com = self._validity(state)
        self._observe_rows(act, code, finite, com, hit)
        contact = act & self.contact_since_step
        self.contact_since_step = torch.where(act, torch.zeros_like(contact), self.contact_since_step)
        self.clean = torch.where(act, self.clean & ~contact, self.clean)
        valid = act & finite
        self.valid_prefix_only |= act & ~finite
        com_safe = torch.where(finite.unsqueeze(-1), com, torch.zeros_like(com))
        moved = py_hypot(com_safe[:, 0] - self.prev_xy[:, 0], com_safe[:, 1] - self.prev_xy[:, 1])
        self.full_path = torch.where(valid, self.full_path + moved, self.full_path)
        self.prev_xy = torch.where(valid.unsqueeze(-1), com_safe[:, :2], self.prev_xy)
        self.prev_time = torch.where(valid, t, self.prev_time)
        error = self._error(com_safe)
        inside = valid & (error <= self.radius) & (self.reason == RUNNING) & self.clean
        # interval (previous_time, t, previous_error, error, inside and previous_inside)
        self._accumulate(valid, previous_time, t, previous_error, error, inside & self.previous_inside)
        self.previous_error = torch.where(valid, error, self.previous_error)
        start = inside & ~self.has_hold
        self.hold_started = torch.where(start, t, self.hold_started)
        self.has_hold = self.has_hold | start
        self.longest_hold = torch.where(inside, torch.maximum(self.longest_hold, t - self.hold_started),
                                        self.longest_hold)
        self.has_hold = torch.where(act & ~inside, torch.zeros_like(inside), self.has_hold)
        self.previous_inside = torch.where(act, inside, self.previous_inside)
        candidate = (inside & self.has_hold & (self.hold_started <= self.late_start + 1.0e-9)
                     & (t >= self.late_end - 1.0e-9) & (t - self.hold_started >= self.hold_s - 1.0e-9))
        # _finish
        safe = (self.reason == RUNNING) & self.clean & ~oob
        newly = act & candidate & safe & ~self.success
        first = newly & torch.isnan(self.tts)
        self.success = self.success | newly
        self.prefix_path = torch.where(first, self.full_path, self.prefix_path)
        self.tts = torch.where(first, t, self.tts)
        timeout = t >= self.deadline - 1.0e-9
        out = torch.zeros(self.n, dtype=torch.long, device=self.device)
        out = torch.where(timeout, torch.full_like(out, TIMEOUT), out)
        out = torch.where(contact, torch.full_like(out, COLLISION), out)
        out = torch.where(oob, torch.full_like(out, OUT_OF_BOUNDS), out)
        out = torch.where(self.reason != RUNNING, self.reason, out)
        out = torch.where(act, out, torch.zeros_like(out))
        finished = out != RUNNING
        self.success = torch.where(finished & (out != TIMEOUT) & (out != SUCCESS),
                                   torch.zeros_like(self.success), self.success)
        any_finished = bool(finished.any())
        self.last_any_finished = any_finished          # G11: host copy for the env (codes != 0 == finished)
        if any_finished:
            self._latch(finished, out)
        self.done |= finished
        return out

    def _accumulate(self, valid, t0, t1, e0, e1, inside):
        fw = self.fw
        begin = self.deadline - self.final_window
        lo = torch.maximum(t0, begin)
        dt = t1 - lo
        use = valid & (dt > 0.0)
        span = torch.where(use, t1 - t0, torch.ones_like(t1))
        f = (lo - t0) / span
        ea = e0 + f * (e1 - e0)
        term = (dt * ((ea * ea + ea * e1) + e1 * e1)) / 3.0
        fw["observed"] = torch.where(use, fw["observed"] + dt, fw["observed"])
        fw["squared"] = torch.where(use, fw["squared"] + term, fw["squared"])
        grow = use & inside
        fw["inside"] = torch.where(grow, fw["inside"] + dt, fw["inside"])
        fw["run"] = torch.where(grow, fw["run"] + dt, torch.where(use, torch.zeros_like(dt), fw["run"]))
        fw["longest"] = torch.where(grow, torch.maximum(fw["longest"], fw["run"]), fw["longest"])

    def _latch(self, rows, codes):
        self.has_completed |= rows
        self.c_success = torch.where(rows, self.success, self.c_success)
        self.c_tts = torch.where(rows, torch.where(self.success, self.tts, torch.full_like(self.tts, math.nan)),
                                 self.c_tts)
        self.c_prefix = torch.where(rows, torch.where(self.success, self.prefix_path,
                                                      torch.full_like(self.prefix_path, math.nan)), self.c_prefix)
        self.c_full = torch.where(rows, self.full_path, self.c_full)
        self.c_reason = torch.where(rows, codes, self.c_reason)
        self.c_valid_prefix_only = torch.where(rows, self.valid_prefix_only, self.c_valid_prefix_only)
        self.c_nonfinite = torch.where(rows, self.nonfinite_seen, self.c_nonfinite)
        self.c_min_clearance = torch.where(rows, self.min_clearance, self.c_min_clearance)
        self.c_longest_hold = torch.where(rows, self.longest_hold, self.c_longest_hold)
        for name in self.FW_FIELDS:
            self.c_fw[name] = torch.where(rows, self.fw[name], self.c_fw[name])
        for i in rows.nonzero().flatten().tolist():
            self.c_params[i] = dict(self._row_params[i])

    # ------------------------------------------------------------ results --
    def completed_result(self, env_id):
        """EpisodeResult.evaluator_fields() of the row's last completed episode (Python floats)."""
        i = int(env_id)
        if not bool(self.has_completed[i]):
            raise RuntimeError("env %d has no completed station-keeping episode" % i)
        params = self.c_params[i]
        observed = float(self.c_fw["observed"][i])
        code = int(self.c_reason[i])
        metrics = dict(
            station_final_window_rms_error_m=(math.sqrt(float(self.c_fw["squared"][i]) / observed)
                                              if observed else None),
            station_final_window_in_region_fraction=(float(self.c_fw["inside"][i]) / observed
                                                     if observed else None),
            station_final_window_longest_hold_s=float(self.c_fw["longest"][i]) if observed else None,
            station_final_window_observed_s=observed,
            station_final_window_coverage_fraction=observed / params["final_window_s"],
            station_longest_hold_s=float(self.c_longest_hold[i]),
            station_late_interval_s=tuple(params["late_interval_s"]),
            station_top_dose_start_s=params["top_dose_start_s"],
            path_cross_track_error_integral_m_s=0.0,
            path_length_valid_prefix_only=bool(self.c_valid_prefix_only[i]),
            min_clearance_m=float(self.c_min_clearance[i]),
            nonfinite_seen=bool(self.c_nonfinite[i]),
            deadline_s=params["deadline_s"],
            integrated_squared_thrust_n2s=None)
        return dict(episode_success=bool(self.c_success[i]), time_to_success=float(self.c_tts[i]),
                    episode_path_length=float(self.c_full[i]),
                    success_prefix_path_length_m=float(self.c_prefix[i]),
                    full_episode_path_length_m=float(self.c_full[i]), termination_reason=code,
                    termination_reason_name=contracts.TERMINATION_REASON_NAMES[code], **metrics)


__all__ = ["TASK", "HAZARD_GROUP", "LAYOUT_GROUP", "POOL_SCHEMA", "CONTROL_DT_S", "TOP_DOSE_START_S",
           "TwinState", "layout_rng", "build_station_layout", "station_task_context", "episode_key",
           "sha256_json", "spawn_pose_hash", "layout_hash", "pool_manifest_digest", "pool_document",
           "WITNESS_SCHEMA", "witness_blob", "witness_from_blob", "witness_document",
           "StationD7Pool", "py_hypot", "rotation_matrix", "bow_axis_xy", "yaw_from_quat",
           "quat_from_yaw", "body_rates", "ray_angles", "point_return_rays", "StationKeepingTwin",
           "ADMITTED", "NOT_ADMITTED", "BEYOND_OFFERED_GRID", "E4_REJECTED", "JOINT_ECHO", "PER_DOSE_COLUMNS",
           "NOT_ADMITTED_SCHEMA", "not_admitted_reasons", "NotAdmittedAssignment", "StationExposureBridge",
           "not_admitted_stamp", "record_exposure", "admit_station_grid", "pool_from_admission"]
