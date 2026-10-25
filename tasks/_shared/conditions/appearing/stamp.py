"""Digest-sealed appearing-suite episode stamps (G3 rule 7). Standard library only.

Envelope = {"payload": {...}, "sha256": sha256(canonical(payload))}, the same
canonical JSON rule as the D7 stamp (dynamic/stamp.py: sort_keys, compact
separators, no NaN/Inf, ASCII). The digest is an integrity check of the
payload, not a signature.

payload:
  schema          usvbench.appearing_stamp/3 (G20b: realized closing speed and TTC in the trigger record;
                  G20d: visible_at_trigger / visible_rays in the trigger record, runtime.visible_at_trigger)
  suite_version   config.APPEARING_SUITE_VERSION (= ladder.SUITE_VERSION, names the ladder option)
  exposure        "admitted" | "not_admitted"   (D16 / D20 form)
  assignment      admitted: key, task, dose, cylinder, trigger gap d_k, digests ...
                  not admitted: dose 0, cylinder None, candidate_digest = the
                  dose-independent not-admitted digest, not_admitted_reasons
  runtime         assigned / itt_count (D27: not admitted -> False / 0),
                  exposure_realized, trigger record (tick, time, realized gap,
                  bearing, pose, closing_speed_mps, realized_ttc_s), swept-contact
                  latch, termination reason; G35 (J19, v2.6.1): trigger_speed_outside_envelope (true) and
                  trigger_speed_envelope {speed_mps, envelope_mps, rule} ONLY when the trigger pose speed lies
                  outside the vessel envelope (recorded, never refused; absent otherwise, so in-envelope stamps
                  are byte-identical to v2.6; the SEMANTICS text below is unchanged for the same reason)
  not_admitted    {"reasons": [...]}  (present only when not admitted)
  semantics       fixed text (below)
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

try:                                   # package import (tasks._shared.conditions.appearing.stamp)
    from . import ladder as _ladder
except ImportError:                    # loaded BY PATH (scripts/appearing_exposure.py): the sibling ladder.py
    _spec = importlib.util.spec_from_file_location("_g3_appearing_ladder", Path(__file__).with_name("ladder.py"))
    _ladder = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ladder)

SCHEMA = "usvbench.appearing_stamp/3"      # G20d: /2 + visible_at_trigger (owner decision 7A)
SUITE_VERSION = _ladder.SUITE_VERSION
LADDER_OPTION = _ladder.ACTIVE_LADDER
GAP_RATIOS = _ladder.GAP_RATIOS
MAX_DOSE = _ladder.MAX_DOSE
LADDER = _ladder                       # G20b/G20c: gap tables, pinned t_avoid, rung / start-gap / speed bounds
# G20c (REV-G20b m6): only the current schema is read (G20d: /3). Earlier appearing certificates (schema /1: v2.3 /
# v2.4 smoke; /2: v2.5 + G20b/G20c diagnostics) are validated with their own release's validator; an older reader here
# would re-admit stamps whose gaps and placements this release can no longer pin (other suite version), so none is added.
LEGACY_SCHEMA_NOTE = ("schema usvbench.appearing_stamp/1 and /2 certificates (releases up to v2.5 and the G20b / G20c "
                      "diagnostic trees) are validated with the validator of their own release")
# G20b (G20 DESIGN_ROLLOUT section 2, adjustment 1): realized time pressure at the trigger
CLOSING_SPEED_EPS_MPS = 1.0e-6
TRIGGER_PRESSURE_FIELDS = ("closing_speed_mps", "realized_ttc_s")
TRIGGER_VISIBILITY_FIELDS = ("visible_at_trigger", "visible_rays")       # G20d (decision 7A)
# G35 (controller decision J19, v2.6.1): a trigger pose speed outside the vessel's declared envelope
# (ladder.VESSEL_MAX_REVERSE_SPEED_MPS / VESSEL_TOP_SPEED_MPS) is physically possible (hard reverse transient,
# contact rebound) and is RECORDED, never refused: the writer adds both runtime fields below ONLY for such a
# trigger (a stamp whose trigger speed lies inside the envelope is byte-identical to v2.6), and the reader requires
# them exactly then. Integrity checks (digest, schema, pins, recomputed fields) are unchanged.
SPEED_ENVELOPE_FLAG = "trigger_speed_outside_envelope"
SPEED_ENVELOPE_RECORD = "trigger_speed_envelope"
SPEED_ENVELOPE_TOL_MPS = 1.0e-6
SPEED_ENVELOPE_RULE = ("J19 (v2.6.1): the trigger pose speed lies outside the vessel envelope [-max reverse, top] "
                       "m/s (ladder.VESSEL_MAX_REVERSE_SPEED_MPS / VESSEL_TOP_SPEED_MPS, tolerance 1e-6); "
                       "physically possible (hard reverse transient, contact rebound): recorded and counted, "
                       "never refused")
ADMITTED = "admitted"
NOT_ADMITTED = "not_admitted"
BEYOND_OFFERED_GRID = "beyond_offered_grid"
JOINT_ECHO = "joint_ladder_member_rejected"

SEMANTICS = {
    "meaning": "a static cylinder absent at reset appears on the nominal route when the own ship comes close",
    "dose": _ladder.dose_text(),
    "k_equivalent": "runtime.k_equivalent = trigger gap / (t_avoid u_ref), the D7-scale ratio of the gap in metres",
    "gap_override": ("runtime.gap_override_m present only with the diagnostic token appearing.gap_m (calibration "
                     "sweep; refused in official runs; bounded by the calibration rung D5 and the task's start gap): "
                     "the run's trigger gap; runtime.dose_gap_m = the dose's d_k"),
    "pins": ("the reader recomputes every gap from the ladder and the vessel's pinned t_avoid: assignment and runtime "
             "trigger gaps, both k_equivalent fields, gap_ratio; the trigger pose speed lies in the vessel envelope"),
    "gap": "D7 footprint rectangle (COM frame) to cylinder surface, signed (dynamic/geometry.point_clearance - r)",
    "trigger": ("checked once per control step after the physics substeps; at the first check with gap <= d_k "
                "the cylinder is physical and visible at once and stays so until the episode ends"),
    "position": "fixed before the episode per scenario key (own stream group); identical at every dose",
    "denominator": ("every ASSIGNED episode (admitted keys: assigned true, itt_count 1, whether or not the "
                    "trigger gap is reached); not-admitted keys are never assigned (assigned false, itt_count 0)"),
    "admission": ("over every offered dose (joint; G20d: docking main tiers D1..D4 without the supplementary D5, "
                  "which is admitted per key; BlueBoat station keeping D1 not applicable by design): placement on "
                  "the nominal route (station keeping: on the approach route at 0.45 x the start-to-centre "
                  "distance, the station and its hold circle clear), clearance >= footprint half-width to static "
                  "obstacles, start, goal and gate regions, and a replay-validated finite-search kinematic "
                  "witness with the cylinder appearing at the nominal trigger of every dose (frozen beam, then the "
                  "fallback beam; docking also monotone stitching from a smaller-gap witness)"),
    "feasibility": ("finite_search_kinematic_witness_not_simulator_proof; a key the finite search does not "
                    "certify is not admitted (reason no_witness_in_frozen_search: not certified by the finite "
                    "search, not proven infeasible)"),
    "visibility": ("trigger.visible_at_trigger = at least one ray of the fan (n rays from the COM at heading + 2 pi "
                   "k / n, true intersections, sensor range) has the cylinder as its FIRST hit among the key's static "
                   "occluder discs and the cylinder, at the realized trigger step; trigger.visible_rays = their number; "
                   "runtime.visible_at_trigger copies it (null when the exposure was not realized); assignment."
                   "nominal_visible_at_trigger = the same test at the dose's nominal trigger pose (pool time). "
                   "Occluded cylinders are kept (owner decision 7A); D7 intruders and native movers are not occluders"),
    "time_pressure": ("SURGE-BASED (REV-G20c m2: sway / crab under current or wind is not included): "
                      "trigger.closing_speed_mps = surge speed at the trigger * cos(relative bearing of the "
                      "cylinder centre) (the D7 pose carries the surge speed); trigger.realized_ttc_s = "
                      "max(realized gap, 0) / closing speed when the closing speed exceeds 1e-6 m/s, else null "
                      "(not closing); reported only, the trigger stays the gap rule"),
}


def closing_speed_mps(speed_mps, bearing_rad):
    """Realized closing speed at the trigger: the surge speed's component toward the cylinder centre."""
    return float(speed_mps) * math.cos(float(bearing_rad))


def realized_ttc_s(gap_m, closing_mps):
    """Realized time to contact at the trigger (None when not closing)."""
    if not closing_mps > CLOSING_SPEED_EPS_MPS:
        return None
    return max(float(gap_m), 0.0) / float(closing_mps)


def trigger_pressure(gap_m, bearing_rad, speed_mps):
    """{"closing_speed_mps", "realized_ttc_s"} of a trigger record (the runtime and the reader share it)."""
    closing = closing_speed_mps(speed_mps, bearing_rad)
    return {"closing_speed_mps": closing, "realized_ttc_s": realized_ttc_s(gap_m, closing)}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# G43 (v2.6.2, owner decision): the reader's recompute of the trigger pressure fields (closing speed = surge x
# math.cos(bearing) and the time to contact derived from it) allows a relative 1e-12, because libm cos differs
# between platforms in the last bit (glibc vs Windows, G42). The writer is unchanged (stamp bytes identical).
TRIGGER_RECOMPUTE_REL_TOL = 1.0e-12


def recomputes(have, want):
    """G43: a stored trigger field matches its recomputation: |have - want| <= 1e-12 x max(1, |want|) for two
    finite numbers; anything else (None, non-numbers) compares exactly as before (None only matches None)."""
    if _number(have) and _number(want):
        return abs(have - want) <= TRIGGER_RECOMPUTE_REL_TOL * max(1.0, abs(want))
    return have == want


def speed_envelope_mps(vessel):
    """[-max reverse, top] m/s of a pinned vessel (None when the vessel has no pinned envelope)."""
    if vessel not in _ladder.VESSEL_TOP_SPEED_MPS or vessel not in _ladder.VESSEL_MAX_REVERSE_SPEED_MPS:
        return None
    return [-_ladder.VESSEL_MAX_REVERSE_SPEED_MPS[vessel], _ladder.VESSEL_TOP_SPEED_MPS[vessel]]


def speed_outside_envelope(vessel, speed):
    """G35 (J19): True when a finite trigger speed lies outside the vessel envelope (the reader's v2.6 bounds)."""
    bounds = speed_envelope_mps(vessel)
    if bounds is None or not _number(speed):
        return False
    return not bounds[0] - SPEED_ENVELOPE_TOL_MPS <= speed <= bounds[1] + SPEED_ENVELOPE_TOL_MPS


def speed_envelope_record(vessel, speed):
    """G35 (J19): the runtime record of an out-of-envelope trigger speed (the reader recomputes it)."""
    return {"speed_mps": speed, "envelope_mps": speed_envelope_mps(vessel), "rule": SPEED_ENVELOPE_RULE}


def _trigger_speed(runtime):
    trigger = runtime.get("trigger") if isinstance(runtime, dict) else None
    if isinstance(trigger, dict) and isinstance(trigger.get("pose"), list) and len(trigger["pose"]) == 4:
        return True, trigger["pose"][3]
    return False, None


def speed_envelope_fields(exposure, assignment, runtime):
    """G35 (J19): {} for every stamp whose trigger speed is inside the envelope (or that has no trigger), else the
    two runtime fields {SPEED_ENVELOPE_FLAG: True, SPEED_ENVELOPE_RECORD: {...}} the writer adds."""
    has_pose, speed = _trigger_speed(runtime)
    vessel = assignment.get("vessel_id") if isinstance(assignment, dict) else None
    if exposure != ADMITTED or not has_pose or not speed_outside_envelope(vessel, speed):
        return {}
    return {SPEED_ENVELOPE_FLAG: True, SPEED_ENVELOPE_RECORD: speed_envelope_record(vessel, speed)}


def speed_envelope_errors(payload):
    """G35 (J19): the trigger-speed envelope fields are present exactly for an admitted stamp whose trigger speed
    lies outside the vessel envelope, and recompute; a non-finite speed stays an error."""
    runtime, assignment = payload["runtime"], payload["assignment"]
    has_pose, speed = _trigger_speed(runtime)
    if has_pose and payload["exposure"] == ADMITTED and not _number(speed):
        return ["appearing trigger speed %r is not a finite number" % (speed,)]
    want = speed_envelope_fields(payload["exposure"], assignment, runtime)
    have = {k: runtime[k] for k in (SPEED_ENVELOPE_FLAG, SPEED_ENVELOPE_RECORD) if k in runtime}
    if have == want:
        return []
    vessel = assignment.get("vessel_id")
    if want:
        return ["appearing trigger speed %r outside the %s envelope %r m/s is not recorded as %s / %s (J19)"
                % (speed, vessel, want[SPEED_ENVELOPE_RECORD]["envelope_mps"], SPEED_ENVELOPE_FLAG,
                   SPEED_ENVELOPE_RECORD)]
    return ["appearing %s / %s present but the trigger speed %r lies inside the %s envelope (or no admitted "
            "trigger)" % (SPEED_ENVELOPE_FLAG, SPEED_ENVELOPE_RECORD, speed, vessel)]


def trigger_errors(runtime):
    """Consistency of the runtime trigger block of one stamp (empty list = consistent). G20d (REV-G20c m1): the
    trigger is the FIRST check at or below d_k (previous_check_gap_m null or above the trigger gap) and its time is
    tick x physics_dt_s exactly."""
    realized = runtime.get("exposure_realized")
    trigger = runtime.get("trigger")
    if "visible_at_trigger" not in runtime:
        return ["appearing runtime lacks visible_at_trigger (schema %s, G20d)" % SCHEMA]
    if realized is False:
        errors = [] if trigger is None else ["an unrealized appearing exposure carries no trigger record"]
        if runtime["visible_at_trigger"] is not None:
            errors.append("an unrealized appearing exposure has visible_at_trigger null")
        return errors
    if realized is not True:
        return ["appearing runtime exposure_realized must be true or false"]
    if not isinstance(trigger, dict):
        return ["a realized appearing exposure needs its trigger record"]
    errors = ["appearing trigger record lacks " + f for f in ("gap_m", "bearing_rad", "pose") + TRIGGER_PRESSURE_FIELDS
              + TRIGGER_VISIBILITY_FIELDS if f not in trigger]
    if errors:
        return errors
    visible, rays = trigger["visible_at_trigger"], trigger["visible_rays"]
    if type(visible) is not bool or type(rays) is not int or rays < 0 or visible != (rays > 0):
        errors.append("appearing trigger visible_at_trigger %r / visible_rays %r are not a bool and a count with "
                      "visible == (rays > 0)" % (visible, rays))
    elif runtime["visible_at_trigger"] is not visible:
        errors.append("appearing runtime.visible_at_trigger %r differs from the trigger record's %r"
                      % (runtime["visible_at_trigger"], visible))
    if errors:
        return errors
    pose = trigger["pose"]
    if not (isinstance(pose, list) and len(pose) == 4 and all(_number(v) for v in pose)
            and _number(trigger["gap_m"]) and _number(trigger["bearing_rad"])):
        return ["appearing trigger gap, bearing and pose [x, y, yaw, speed] must be finite numbers"]
    want = trigger_pressure(trigger["gap_m"], trigger["bearing_rad"], pose[3])
    for field in TRIGGER_PRESSURE_FIELDS:
        if not recomputes(trigger[field], want[field]):
            errors.append("appearing trigger %s %r does not recompute from the trigger record (%r)"
                          % (field, trigger[field], want[field]))
    gap = runtime.get("trigger_gap_m")
    if _number(gap) and trigger["gap_m"] > gap:
        errors.append("appearing trigger fired at gap %r above d_k %r" % (trigger["gap_m"], gap))
    previous = trigger.get("previous_check_gap_m")
    if previous is not None and not (_number(previous) and _number(gap) and previous > gap):
        errors.append("appearing trigger is not the first check at or below d_k (previous check gap %r, d_k %r)"
                      % (previous, gap))
    dt, tick = runtime.get("physics_dt_s"), trigger.get("tick")
    if dt is not None or "time_s" in trigger:
        if not (type(tick) is int and _number(dt) and _number(trigger.get("time_s"))
                and trigger["time_s"] == tick * dt):
            errors.append("appearing trigger time %r is not tick %r x physics_dt_s %r"
                          % (trigger.get("time_s"), tick, dt))
    return errors


def exposure_kind_errors(payload):
    """G20d (REV-G20c m1, D16 / D27): a D0 or not-admitted stamp never claims an appearance."""
    runtime, assignment = payload["runtime"], payload["assignment"]
    if payload["exposure"] == NOT_ADMITTED or assignment.get("dose") == 0:
        want = "not_admitted" if payload["exposure"] == NOT_ADMITTED else "zero_dose"
        if runtime.get("exposure_realized") is not False or runtime.get("trigger") is not None \
                or runtime.get("non_exposure_reason") != want:
            return ["a %s appearing stamp has exposure_realized false, no trigger and non_exposure_reason %r"
                    % ("not-admitted" if want == "not_admitted" else "D0", want)]
    return []


def _same(a, b):
    return a is None and b is None or (_number(a) and _number(b) and a == b)


def pin_errors(payload):
    """G20c (REV-G20b M1): an ADMITTED stamp's gaps and k fields equal the ladder's, computed from the pinned
    t_avoid of the stamped vessel (ladder.T_AVOID_S), never from the stamp's own numbers."""
    assignment, runtime = payload["assignment"], payload["runtime"]
    if payload["exposure"] != ADMITTED:
        return []
    vessel, dose, task = assignment.get("vessel_id"), assignment["dose"], assignment.get("task")
    if vessel not in _ladder.T_AVOID_S:
        return ["appearing stamp vessel %r has no pinned ladder" % (vessel,)]
    t_avoid = _ladder.T_AVOID_S[vessel]
    errors = []
    if dose in _ladder.not_applicable_doses(task, vessel):        # G20d (decision 2B)
        errors.append("appearing D%d is not applicable by design for %s on %s (never offered)" % (dose, task, vessel))
    if dose == 0:
        for where, field in (("assignment", "trigger_gap_m"), ("runtime", "trigger_gap_m")):
            if payload[where].get(field) is not None:
                errors.append("appearing D0 stamp carries %s.%s" % (where, field))
        if runtime.get("gap_override_m") is not None:
            errors.append("appearing D0 stamp carries a gap override")
        return errors
    gap = _ladder.gap_m(dose, t_avoid, vessel)
    if not _same(assignment.get("trigger_gap_m"), gap):
        errors.append("appearing assignment.trigger_gap_m %r is not the ladder gap %r of D%d for %s"
                      % (assignment.get("trigger_gap_m"), gap, dose, vessel))
    if assignment.get("t_avoid_s") is not None and assignment["t_avoid_s"] != t_avoid:
        errors.append("appearing assignment.t_avoid_s %r is not the pinned %r" % (assignment["t_avoid_s"], t_avoid))
    want_ratio = None if _ladder.GAP_RATIOS is None else _ladder.GAP_RATIOS[dose]
    if assignment.get("gap_ratio") != want_ratio:
        errors.append("appearing assignment.gap_ratio %r is not %r" % (assignment.get("gap_ratio"), want_ratio))
    if not _same(assignment.get("k_equivalent"), _ladder.k_equivalent(gap, t_avoid)):
        errors.append("appearing assignment.k_equivalent does not recompute from the ladder gap")
    override = runtime.get("gap_override_m")
    if override is not None:
        low, high = _ladder.GAP_OVERRIDE_MIN_M[vessel], _ladder.start_gap_m(task, vessel)
        if not _number(override) or not low - 1e-12 <= override <= high + 1e-12:
            errors.append("appearing gap override %r outside [calibration rung %r, start gap %r] m (%s, %s)"
                          % (override, low, high, vessel, task))
        if not _same(runtime.get("dose_gap_m"), gap):
            errors.append("appearing runtime.dose_gap_m %r is not the ladder gap %r" % (runtime.get("dose_gap_m"), gap))
        want_trigger = override
    else:
        for field in ("dose_gap_m", "gap_override_position"):
            if field in runtime:
                errors.append("appearing runtime.%s without a gap override" % field)
        want_trigger = gap
    if not _same(runtime.get("trigger_gap_m"), want_trigger):
        errors.append("appearing runtime.trigger_gap_m %r is not %r (%s)" % (
            runtime.get("trigger_gap_m"), want_trigger, "the gap override" if override is not None else
            "the ladder gap of D%d" % dose))
    if not _same(runtime.get("k_equivalent"), _ladder.k_equivalent(want_trigger, t_avoid)):
        errors.append("appearing runtime.k_equivalent does not recompute from the trigger gap and the pinned t_avoid")
    # G35 (J19): the trigger-speed envelope is no longer a pin (it raised mid-training on a physically possible
    # hard-reverse trigger, G33 S3); speed_envelope_errors (read_appearing_stamp) requires the recorded flag instead
    return errors


def override_errors(runtime):
    """G20b: a diagnostic gap override is stamped consistently (the trigger gap IS the override)."""
    override = runtime.get("gap_override_m")
    if override is None:
        return []
    if not _number(override) or override <= 0:
        return ["appearing gap_override_m must be a positive number"]
    if runtime.get("trigger_gap_m") is not None and runtime["trigger_gap_m"] != override:
        return ["appearing trigger_gap_m %r differs from the gap override %r" % (runtime["trigger_gap_m"], override)]
    return []


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def make_appearing_stamp(exposure, assignment, runtime, reasons=None):
    """Seal one finished episode; ``assignment`` and ``runtime`` are JSON-native dicts."""
    if exposure not in (ADMITTED, NOT_ADMITTED):
        raise ValueError("appearing exposure must be 'admitted' or 'not_admitted'")
    payload = {"schema": SCHEMA, "suite_version": SUITE_VERSION, "exposure": exposure,
               "assignment": assignment, "runtime": runtime, "semantics": dict(SEMANTICS)}
    if exposure == NOT_ADMITTED:
        items = [str(r) for r in (reasons or ())]
        if not items:
            raise ValueError("a not-admitted appearing stamp needs at least one reason")
        payload["not_admitted"] = {"reasons": items}
    elif reasons:
        raise ValueError("an admitted appearing stamp carries no not-admitted reasons")
    extra = speed_envelope_fields(exposure, assignment, runtime)     # G35 (J19): {} inside the envelope
    if extra:
        payload["runtime"] = dict(runtime, **extra)
    return json.loads(canonical({"payload": payload, "sha256": digest(payload)}))


def read_appearing_stamp(value):
    """Verify and round-trip an envelope (JSON text or mapping); returns the mapping."""
    value = json.loads(value) if isinstance(value, str) else json.loads(canonical(value))
    if not isinstance(value, dict) or set(value) != {"payload", "sha256"}:
        raise ValueError("invalid appearing stamp envelope")
    if digest(value["payload"]) != value["sha256"]:
        raise ValueError("appearing stamp digest mismatch")
    payload = value["payload"]
    if payload.get("schema") != SCHEMA or payload.get("suite_version") != SUITE_VERSION:
        raise ValueError("unsupported appearing stamp version (%r, %r; this release reads %s / %s; %s)"
                         % (payload.get("schema"), payload.get("suite_version"), SCHEMA, SUITE_VERSION,
                            LEGACY_SCHEMA_NOTE))
    for field in ("exposure", "assignment", "runtime", "semantics"):
        if field not in payload:
            raise ValueError("incomplete appearing stamp")
    exposure = payload["exposure"]
    assignment, runtime = payload["assignment"], payload["runtime"]
    if exposure not in (ADMITTED, NOT_ADMITTED) or assignment.get("exposure") != exposure:
        raise ValueError("appearing stamp exposure claims disagree")
    for field in ("key", "task", "dose", "candidate_digest", "position_digest"):
        if field not in assignment:
            raise ValueError("appearing stamp assignment lacks " + field)
    if exposure == NOT_ADMITTED:
        if assignment["dose"] != 0 or assignment.get("cylinder") is not None:
            raise ValueError("a not-admitted appearing stamp runs D0 without a cylinder (D16)")
        if runtime.get("assigned") is not False or runtime.get("itt_count") != 0:
            raise ValueError("a not-admitted appearing stamp says assigned false, itt_count 0 (D27)")
        if not (payload.get("not_admitted") or {}).get("reasons"):
            raise ValueError("a not-admitted appearing stamp carries its reasons")
    else:
        if runtime.get("assigned") is not True or runtime.get("itt_count") != 1:
            raise ValueError("an admitted appearing stamp says assigned true, itt_count 1")
        if "not_admitted" in payload:
            raise ValueError("an admitted appearing stamp carries no not-admitted block")
    dose = assignment["dose"]
    if type(dose) is not int or not 0 <= dose <= MAX_DOSE:
        raise ValueError("appearing stamp dose %r outside D0..D%d (ladder option %s)" % (dose, MAX_DOSE, LADDER_OPTION))
    problems = exposure_kind_errors(payload) + trigger_errors(runtime) + override_errors(runtime) + pin_errors(payload)
    if not problems:
        problems = speed_envelope_errors(payload)                     # G35 (J19)
    if problems:
        raise ValueError("; ".join(problems))
    return value


def exposure_of(envelope):
    """("admitted" | "not_admitted", reasons) of a verified envelope (reasons without the joint echo)."""
    value = read_appearing_stamp(envelope)
    payload = value["payload"]
    if payload["exposure"] == ADMITTED:
        return ADMITTED, ()
    seen = []
    for item in payload["not_admitted"]["reasons"]:
        if item != JOINT_ECHO and item not in seen:
            seen.append(item)
    if not seen:
        raise ValueError("a not-admitted appearing stamp needs a reason other than the joint echo")
    return NOT_ADMITTED, tuple(seen)


__all__ = ["SCHEMA", "SUITE_VERSION", "ADMITTED", "NOT_ADMITTED", "BEYOND_OFFERED_GRID", "SEMANTICS",
           "LADDER_OPTION", "GAP_RATIOS", "MAX_DOSE", "CLOSING_SPEED_EPS_MPS", "TRIGGER_PRESSURE_FIELDS",
           "LADDER", "closing_speed_mps", "realized_ttc_s", "trigger_pressure", "trigger_errors", "override_errors",
           "TRIGGER_VISIBILITY_FIELDS", "exposure_kind_errors",
           "pin_errors", "LEGACY_SCHEMA_NOTE",
           "SPEED_ENVELOPE_FLAG", "SPEED_ENVELOPE_RECORD", "SPEED_ENVELOPE_TOL_MPS", "SPEED_ENVELOPE_RULE",
           "speed_envelope_mps", "speed_outside_envelope", "speed_envelope_record", "speed_envelope_fields",
           "speed_envelope_errors",
           "canonical", "digest", "make_appearing_stamp", "read_appearing_stamp", "exposure_of"]
