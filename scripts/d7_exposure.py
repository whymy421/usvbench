"""D16 exposure of journal D7 episode stamps and D7 header counts (F7; pure Python).

CONTEXT_R4 D16: a key that E4 does not admit keeps its E5 layout; the dynamic suite runs
that scenario at D0 (no intruder) and the finished-episode record says so. This module is
the ONE reader of that statement for the certificate writer (cert_v3), the validator
(validate_cert_v3) and the analysis inputs (analysis_input); it imports nothing outside
the standard library.

Wire form (CONTEXT_R4 D20: the form the five round-4b families emit; F7's earlier
empty-assignment form is withdrawn). A finished episode's D7 record is an E4 stamp
envelope

    record["dynamic_hazards"] = {"payload": P, "sha256": sha256(canonical(P))}

with exactly the two keys payload and sha256 (E3 verify_dynamic_hazard_stamp, the E4
read_stamp and the certificate schema accept nothing else; the digest is verified by the
callers).

(a) Exposure claims may sit at P["exposure"], P["runtime"]["exposure"] and
    P["assignment"]["exposure"], each optional; every present claim must agree and be
    "admitted" or "not_admitted"; no claim at all means admitted (E4 stamps carry none).
(b) A not_admitted stamp keeps a dose-0 assignment: P["assignment"] is a dict with
    dose == 0, intruders absent or [], and the keys key, task, dose, candidate_digest
    present (the fields F0 pairing.d7_primitive_hash reads).
(c) Its reasons come from the FIRST non-empty of REASON_SOURCES (below); every item is
    normalised (a string as is; a dict -> item["reason"]; a list or tuple -> its string
    elements), the joint-ladder echo "joint_ladder_member_rejected" is dropped, at least
    one reason must be left, and the reasons are returned de-duplicated in first-seen
    order. Where the five families put them: F1 hazard_nav P.not_admitted.reasons
    ([dose, reason, n] rows), F2 path_hazard P.not_admitted_reasons ([dose, reason, n]),
    F3 path_following P.assignment.rejection_reasons (strings), F5 docking
    P.not_admitted.reasons ({dose, reason, sampled_states}), F6 harbour
    P.assignment.not_admitted_reasons ({dose, reason, sampled_states}).
(d) The D11 d7_assignment primitive of any record is pairing.d7_primitive_hash(envelope):
    the dose-0 assignment of a not-admitted key hashes identically in every run of that
    key (every suite, every dose), so it is unexposed for the dynamic treatment and pairs
    for every other suite with no change to F0 pairing (D20, D21).
An admitted reading (explicit or implicit) carries no rejection reasons (round-4b rule,
kept: E4 admitted stamps have none).

CONTEXT_R4 D24: a counted record of a key beyond the offered admission grid (reason
BEYOND_OFFERED_GRID) is never valid; cert_v3 and validate_cert_v3 use
``is_beyond_offered_grid``.

Header per-dose counts (D16/D22): the D7 header ``per_dose_counts`` has one row per dose
D0..D4, each either
    [offered, admitted, not_admitted]              with offered == admitted + not_admitted, or
    {"offered": o, "admitted": a, "not_admitted": n}  (same identity),
or legacy [offered, admitted] rows PLUS ``per_dose_not_admitted`` = [n0..n4] (same identity).

G23b (owner decision 2026-10-03, NC 09: G23 option S plus option A's exposure metric). Admitted stamps
of the runtime rule ``d7_release_sync_v1`` (conditions/dynamic/release.py) carry
payload.runtime.runtime_rule, .release (hold point, release time and cause, state at release,
predicted closest approach), .finite_ttc_at_first_detection (G23c: metric A, perception timing),
.realized_ttc_at_first_detection_over_t_avoid, .exposure_record (G23c F4) and the non-exposure reason
"not_released". ``runtime_exposure`` reads and cross-checks one stamp;
``exposure_summary`` is the per-cell summary of exposure metric A that cert_v3 stamps as
dynamic_hazards_header.exposure_summary and validate_cert_v3 recomputes: over the ADMITTED
(assigned, D27) episodes of the cell's dose, the released and detected counts, metric A (G23c: a
perception-timing statistic, NOT conflict: episodes whose constant-velocity TTC at the first detection is
finite, and that TTC / t_avoid: n, median, inclusive quartiles, min, max) and the realized exposure of
runtime.exposure_record (contact after the intruder is visible, closest gap, ray-return lag, sync error).
Stamps without runtime_rule (v2.5 and earlier) read as rule None (pinned motion).
"""
from __future__ import annotations

import statistics

ADMITTED = "admitted"
NOT_ADMITTED = "not_admitted"
EXPOSURE_VALUES = (ADMITTED, NOT_ADMITTED)
JOINT_ECHO = "joint_ladder_member_rejected"
BEYOND_OFFERED_GRID = "beyond_offered_grid"
DOSES = 5
# (a): where an exposure claim may sit (path below the payload; () = the payload itself)
CLAIM_SOURCES = ((), ("runtime",), ("assignment",))
# (b): the dose-0 assignment fields of a not-admitted stamp
NOT_ADMITTED_ASSIGNMENT_KEYS = ("key", "task", "dose", "candidate_digest")
# (c): reason locations, first non-empty wins (paths below the payload)
REASON_SOURCES = (
    ("not_admitted", "reasons"),
    ("not_admitted_reasons",),
    ("assignment", "not_admitted_reasons"),
    ("assignment", "rejection_reasons"),
    ("runtime", "rejection_reasons"),
    ("runtime", "reasons"),
    ("rejection_reasons",),
    ("reasons",),
)


# G23b runtime rule S; G23c: rule table (L2 adds the reveal rule), metric A as a perception-timing statistic,
# the realized-exposure record (REV-G23b F4, F5)
RUNTIME_RULE_VERSION = "d7_release_sync_v1"
PINNED_RULE_VERSION = "pinned_v0"
REVEAL_RULE_VERSION = "d7_reveal_sync_v1"     # G23c L2: candidate rule S2 (diagnostic in layer 3)
HYBRID_RULE_VERSION = "d7_reveal_sync_v2"     # G23c L3: S2 + hybrid corridor predictor; G23e layer B: diagnostic
V3_RULE_VERSION = "d7_reveal_sync_v3"         # G23e layer B (REV-G23d M1-M4): the default
DIAGNOSTIC_RULES = (PINNED_RULE_VERSION, RUNTIME_RULE_VERSION, REVEAL_RULE_VERSION, HYBRID_RULE_VERSION)
DEFAULT_RULE = V3_RULE_VERSION
RULE_CAUSES = {RUNTIME_RULE_VERSION: ("sync", "accelerating_pace", "detection_guard"),
               REVEAL_RULE_VERSION: ("ttc_reveal", "clamped_start", "clamped_late", "late_contact_window"),
               HYBRID_RULE_VERSION: ("ttc_reveal", "clamped_start", "clamped_late", "late_contact_window",
                                     "cpa_reveal", "late_first_meeting", "clamped_accelerating"),   # G23d L4
               V3_RULE_VERSION: ("ttc_reveal", "clamped_start", "clamped_late", "late_contact_window",
                                 "cpa_reveal", "late_first_meeting", "clamped_accelerating", "late_reveal")}
UNRELEASED_REASONS = {RUNTIME_RULE_VERSION: "not_released", REVEAL_RULE_VERSION: "not_revealed",
                      HYBRID_RULE_VERSION: "not_revealed", V3_RULE_VERSION: "not_revealed"}
NOT_RELEASED = "not_released"
NOT_REVEALED = "not_revealed"
RELEASE_CAUSES = RULE_CAUSES[RUNTIME_RULE_VERSION]
NON_EXPOSURE_REASONS = (None, "native_motion_axis", "zero_dose", "no_existing_range_sensor",
                        "sensor_observations_missing", NOT_RELEASED, NOT_REVEALED, "terminated_before_nominal_reveal",
                        "not_detected_at_sampled_times")
EXPOSURE_RECORD_SCHEMA = "usvbench.d7_exposure_record/1"
EXPOSURE_METRIC = ("A, a perception-timing statistic and NOT a conflict measure: the share of admitted (assigned, "
                   "CONTEXT_R4 D27) episodes whose constant-velocity TTC at the first detection is finite, and that "
                   "TTC / t_avoid; the realized exposure (closest approach after the intruder is visible, contact, "
                   "sync error, ray-return lag) comes from runtime.exposure_record")
SUMMARY_SCHEMA = "usvbench.d7_exposure_summary/3"     # G23d L4 (REV-G23c H8): metric A moved to "diagnostic"
SUMMARY_HEADLINE = ("dose_achieved share: the independent constant-velocity TTC (ttc_linear; time to the closest point "
                    "of approach for a cpa_reveal) at the reveal tick within dose_achieved_tol of dose x t_avoid; "
                    "the rule's TTC at the first detection (route-aware prediction, aged to the detection); the "
                    "clamped share (dose not achievable for that own ship); metric A (constant-velocity finiteness at "
                    "the first detection) is structurally wrong under d7_reveal_sync_v2 on curved approaches and is "
                    "kept as a diagnostic field only")
DOSE_ACHIEVED_RULES = (HYBRID_RULE_VERSION, V3_RULE_VERSION)   # rules whose stamp must carry the H2 fields
V3_CLAMPED_CAUSES = ("clamped_start", "clamped_late", "late_contact_window", "late_first_meeting",
                     "clamped_accelerating", "late_reveal")
DOSE_TTC_RATIOS = (None, 4.0, 3.0, 2.0, 1.25)            # = conditions/dynamic/model.DOSE_TTC_RATIOS (test-checked)
DOSE_ACHIEVED_TOL = 0.05                                # = reveal.DOSE_ACHIEVED_TOL (test-checked)
_RATIO_TOLERANCE = 1e-9
# G23e (owner decision 2026-10-06): station keeping meets the intruder during the APPROACH to the hold circle
# (conditions/dynamic/sk_approach.py; the constants are repeated here, this module imports only the standard
# library, and a test asserts they agree). Its stamps under the reveal rules carry release.applies true,
# plan_kind "approach" and release.sk_approach; tiers excluded by the design are "not applicable by design"
# (header not_applicable_doses, per-dose rows all zero), distinct from not_admitted.
SK_TASK = "station_keeping"
SK_DESIGN = "sk_approach_crossing_v1"
SK_PLAN_KIND = "approach"
SK_MEETING_FRACTION = 0.45
SK_CROSSING_ANGLE_RAD = 1.5707963267948966
SK_RULES = (REVEAL_RULE_VERSION, HYBRID_RULE_VERSION, V3_RULE_VERSION)
NOT_APPLICABLE = "not_applicable_by_design"
_SK_TOLERANCE = 1e-9


class ExposureError(ValueError):
    """A D7 stamp or header makes an unreadable or contradictory exposure claim."""


def _where(path):
    return ".".join(("payload",) + tuple(path))


def _lookup(payload, path):
    value = payload
    for name in path:
        if not isinstance(value, dict) or name not in value:
            return None
        value = value[name]
    return value


def _claims(payload):
    claims = []
    for path in CLAIM_SOURCES:
        block = _lookup(payload, path) if path else payload
        if not isinstance(block, dict) or "exposure" not in block:
            continue
        value = block["exposure"]
        if value not in EXPOSURE_VALUES:
            raise ExposureError("%s.exposure must be one of %s, not %r" % (_where(path), EXPOSURE_VALUES, value))
        claims.append((_where(path), value))
    return claims


def _first_reasons(payload):
    """(source path, raw value) of the first non-empty reason location, or (None, None)."""
    for path in REASON_SOURCES:
        value = _lookup(payload, path)
        if value is None:
            continue
        if isinstance(value, (list, tuple, dict, str)) and len(value) == 0:
            continue
        return _where(path), value
    return None, None


def _normalise_reasons(where, value):
    if not isinstance(value, (list, tuple)):
        raise ExposureError("%s must be a list of reasons, not %s" % (where, type(value).__name__))
    out = []
    for item in value:
        if isinstance(item, str):
            names = [item]
        elif isinstance(item, dict):
            if "reason" not in item:
                raise ExposureError("%s: a reason object needs a 'reason' field: %r" % (where, item))
            names = [item["reason"]]
        elif isinstance(item, (list, tuple)):
            names = [x for x in item if isinstance(x, str)]
        else:
            raise ExposureError("%s: unreadable reason item %r" % (where, item))
        for name in names:
            if not isinstance(name, str) or not name:
                raise ExposureError("%s: every reason must be a non-empty string, not %r" % (where, name))
            if name != JOINT_ECHO and name not in out:
                out.append(name)
    return tuple(out)


def exposure_of(envelope):
    """("admitted", ()) or ("not_admitted", reasons) of a D7 stamp envelope (D20 a-c).

    Structural reading only; the digest is verified by the callers (E3
    verify_dynamic_hazard_stamp / E4 read_stamp) on the {payload, sha256} core.
    """
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise ExposureError("dynamic_hazards must be a D7 stamp envelope {payload, sha256}")
    if set(envelope) != {"payload", "sha256"}:
        raise ExposureError("dynamic_hazards must be exactly {payload, sha256}; every exposure claim "
                            "belongs inside the digest-covered payload")
    payload = envelope["payload"]
    claims = _claims(payload)
    if len({value for _, value in claims}) > 1:
        raise ExposureError("exposure claims disagree: %s" % ", ".join("%s=%s" % c for c in claims))
    where, raw = _first_reasons(payload)
    if not claims or claims[0][1] == ADMITTED:
        if where is not None:
            raise ExposureError("an admitted stamp carries no rejection reasons (%s)" % where)
        return ADMITTED, ()
    assignment = payload.get("assignment")
    if not isinstance(assignment, dict):
        raise ExposureError("a not_admitted stamp keeps a dose-0 assignment dict (payload.assignment)")
    missing = [k for k in NOT_ADMITTED_ASSIGNMENT_KEYS if k not in assignment]
    if missing:
        raise ExposureError("a not_admitted assignment needs the keys %s; missing %s"
                            % (list(NOT_ADMITTED_ASSIGNMENT_KEYS), missing))
    if type(assignment["dose"]) is not int or assignment["dose"] != 0:
        raise ExposureError("a not_admitted assignment runs at dose 0, not %r" % (assignment["dose"],))
    if "intruders" in assignment and assignment["intruders"] != []:
        raise ExposureError("a not_admitted assignment carries no intruders")
    if where is None:
        raise ExposureError("a not_admitted stamp must carry its E4 rejection reasons (one of %s)"
                            % ", ".join(_where(p) for p in REASON_SOURCES))
    reasons = _normalise_reasons(where, raw)
    if not reasons:
        raise ExposureError("%s holds no reason besides the joint-ladder echo %r" % (where, JOINT_ECHO))
    return NOT_ADMITTED, reasons


def is_beyond_offered_grid(envelope):
    """True when the stamp is a not-admitted record with the D24 reason "beyond_offered_grid"."""
    kind, reasons = exposure_of(envelope)
    return kind == NOT_ADMITTED and BEYOND_OFFERED_GRID in reasons


def d7_primitive(pairing, envelope):
    """D11 ``d7_assignment`` hash of one episode, admitted or not: pairing.d7_primitive_hash (D20 d)."""
    return pairing.d7_primitive_hash(envelope)


def per_dose_counts(header):
    """[{dose, offered, admitted, not_admitted}] x 5 from a D7 header, or raise ExposureError."""
    rows = header.get("per_dose_counts") if isinstance(header, dict) else None
    if not isinstance(rows, list) or len(rows) != DOSES:
        raise ExposureError("dynamic_hazards_header.per_dose_counts needs one row per dose D0..D4")
    extra = header.get("per_dose_not_admitted")
    out = []
    for dose, row in enumerate(rows):
        if isinstance(row, dict):
            values = [row.get("offered"), row.get("admitted"), row.get("not_admitted")]
        elif isinstance(row, list) and len(row) == 3:
            values = list(row)
        elif isinstance(row, list) and len(row) == 2 and isinstance(extra, list) and len(extra) == DOSES:
            values = list(row) + [extra[dose]]
        else:
            raise ExposureError("per_dose_counts row D%d has no not_admitted count (D16)" % dose)
        if any(type(v) is not int or v < 0 for v in values):
            raise ExposureError("per_dose_counts row D%d must hold nonnegative integers" % dose)
        offered, admitted, missing = values
        if offered != admitted + missing:
            raise ExposureError("per_dose_counts row D%d: offered %d != admitted %d + not_admitted %d"
                                % (dose, offered, admitted, missing))
        out.append({"dose": dose, "offered": offered, "admitted": admitted, "not_admitted": missing})
    return out


def not_applicable_doses(header):
    """G23e: the header's tiers not applicable by design (sorted tuple of D1..D4; () when absent), checked against
    the per-dose rows (all zero). Raises ExposureError on a malformed list."""
    if not isinstance(header, dict) or header.get("not_applicable_doses") is None:
        return ()
    doses = header["not_applicable_doses"]
    if (not isinstance(doses, list) or any(type(d) is not int or not 1 <= d < DOSES for d in doses)
            or doses != sorted(set(doses))):
        raise ExposureError("dynamic_hazards_header.not_applicable_doses must be a sorted list of distinct tiers "
                            "D1..D4")
    for row in per_dose_counts(header):
        if row["dose"] in doses and (row["offered"], row["admitted"], row["not_admitted"]) != (0, 0, 0):
            raise ExposureError("per_dose_counts row D%d of a tier not applicable by design must be all zero"
                                % row["dose"])
    return tuple(doses)


def _sk_approach_errors(release, assignment, dose):
    """G23e: the release block of an admitted station-keeping stamp under a reveal rule (raises ExposureError)."""
    if release.get("applies") is not True or release.get("plan_kind") != SK_PLAN_KIND:
        raise ExposureError("a station-keeping stamp of rule %s must time the approach encounter (release.applies "
                            "true, plan_kind %r): the hold-design pinned schedule is retired (G23e)"
                            % (release.get("rule"), SK_PLAN_KIND))
    block = release.get("sk_approach")
    if not isinstance(block, dict) or block.get("design") != SK_DESIGN:
        raise ExposureError("payload.runtime.release.sk_approach (%s) is missing" % SK_DESIGN)
    first = block.get("first_applicable_dose")
    if type(first) is not int or not 1 <= first <= dose:
        raise ExposureError("station keeping D%d is not applicable by design (first applicable tier %r)"
                            % (dose, first))
    if block.get("not_applicable_doses") != list(range(1, first)):
        raise ExposureError("release.sk_approach.not_applicable_doses must be the tiers below the first applicable")
    if block.get("side") not in (1, -1) or not isinstance(block.get("crossing_angle_rad"), (int, float)) or \
            abs(block["crossing_angle_rad"] - SK_CROSSING_ANGLE_RAD) > _SK_TOLERANCE:
        raise ExposureError("release.sk_approach: a 90 deg crossing from side +1 or -1 is required")
    if block.get("meeting_fraction") != SK_MEETING_FRACTION:
        raise ExposureError("release.sk_approach.meeting_fraction must be %r" % SK_MEETING_FRACTION)
    numbers = [block.get(name) for name in ("start_to_centre_m", "meeting_distance_to_centre_m",
                                            "hold_line_distance_m", "hold_required_m", "hold_clearance_m")]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in numbers):
        raise ExposureError("release.sk_approach lacks its geometry numbers")
    d_start, reach, line, required, clearance = numbers
    if abs(reach - SK_MEETING_FRACTION * d_start) > _SK_TOLERANCE * max(1.0, d_start):
        raise ExposureError("release.sk_approach: meeting distance is not 0.45 x start-to-centre distance")
    if clearance < 0.0 or abs(line - required - clearance) > _SK_TOLERANCE * max(1.0, line):
        raise ExposureError("release.sk_approach: the intruder path enters the hold-circle margin")
    meta = dict((k, v) for k, v in (assignment.get("encounter_metadata") or ()) if isinstance(k, str))
    if meta and (meta.get("sk_design") != SK_DESIGN or meta.get("first_applicable_dose") != first
                 or meta.get("side") != block.get("side")):
        raise ExposureError("release.sk_approach disagrees with the assignment's encounter metadata")


def _check_record(record, first):
    """G23c F4 / F10: the realized-exposure record of a runtime-rule stamp."""
    if not isinstance(record, dict) or record.get("schema") != EXPOSURE_RECORD_SCHEMA:
        raise ExposureError("a runtime-rule stamp at D1..D4 needs payload.runtime.exposure_record (%s)"
                            % EXPOSURE_RECORD_SCHEMA)
    for name in ("first_visible_s", "first_ray_return_s", "closest_after_moving", "closest_after_visible",
                 "designed_contact_s", "sync_error_s", "unsynchronizable_slack_s"):
        if name not in record:
            raise ExposureError("payload.runtime.exposure_record lacks %r" % name)
    if record["first_visible_s"] != first:
        raise ExposureError("exposure_record.first_visible_s must equal first_detection_s (one visibility flag)")
    ray = record["first_ray_return_s"]
    if ray is not None and (first is None or ray < first - 1e-12):
        raise ExposureError("exposure_record.first_ray_return_s precedes the first detection")
    if (record["closest_after_visible"] is None) != (first is None):
        raise ExposureError("exposure_record.closest_after_visible must exist exactly when the intruder was seen")


def runtime_exposure(envelope):
    """G23b/G23c: runtime reading of one D7 stamp envelope (structure only; digests are the callers').

    Returns None for a not-admitted stamp (D16/D24: never assigned, D27), else a dict with the keys
    rule, dose, released, release_cause, release_time_s, detected, finite_ttc, ttc_over_t_avoid,
    non_exposure_reason, contact_after_visible, closest_gap_after_visible_m, ray_return_lag_s,
    sync_error_s. Raises ExposureError when a runtime-rule stamp contradicts itself.
    """
    kind, _ = exposure_of(envelope)
    if kind == NOT_ADMITTED:
        return None
    payload = envelope["payload"]
    runtime = payload.get("runtime")
    assignment = payload.get("assignment")
    if not isinstance(runtime, dict) or not isinstance(assignment, dict):
        raise ExposureError("an admitted D7 stamp needs payload.runtime and payload.assignment dicts")
    dose = assignment.get("dose")
    if type(dose) is not int or not 0 <= dose < DOSES:
        raise ExposureError("payload.assignment.dose must be an integer D0..D4")
    rule_block = runtime.get("runtime_rule")
    rule = rule_block.get("version") if isinstance(rule_block, dict) else None
    first = runtime.get("first_detection_s")
    ttc = runtime.get("realized_ttc_at_first_detection_s")
    reason = runtime.get("non_exposure_reason")
    out = {"rule": rule, "dose": dose, "released": None, "release_cause": None, "release_time_s": None,
           "detected": first is not None, "finite_ttc": ttc is not None and first is not None,
           "ttc_over_t_avoid": None, "non_exposure_reason": reason, "contact_after_visible": None,
           "closest_gap_after_visible_m": None, "ray_return_lag_s": None, "sync_error_s": None,
           "dose_achieved": None, "clamped": None, "rule_ttc_at_first_detection_over_t_avoid": None,
           "dose_undefined": None}
    t_avoid = assignment.get("t_avoid_s")
    if ttc is not None and isinstance(t_avoid, (int, float)) and t_avoid > 0:
        out["ttc_over_t_avoid"] = ttc / t_avoid
    if rule is None:
        return out                       # v2.5 stamps, D0, moving basin (G23c F13: no rule block)
    if dose == 0 or assignment.get("task") == "moving_basin":
        raise ExposureError("D0 and moving-basin stamps carry no runtime_rule block (G23c F13)")
    if rule in DIAGNOSTIC_RULES:
        return out
    if rule not in RULE_CAUSES:
        raise ExposureError("unknown D7 runtime rule %r" % (rule,))
    if reason not in NON_EXPOSURE_REASONS:
        raise ExposureError("payload.runtime.non_exposure_reason %r is not a D7 reason" % (reason,))
    if runtime.get("finite_ttc_at_first_detection") is not (ttc is not None):
        raise ExposureError("payload.runtime.finite_ttc_at_first_detection must equal "
                            "(realized_ttc_at_first_detection_s is not None)")
    if ttc is not None and first is None:
        raise ExposureError("a realized TTC needs a first detection")
    ratio = runtime.get("realized_ttc_at_first_detection_over_t_avoid")
    if (ratio is None) != (ttc is None) or (ratio is not None and (
            out["ttc_over_t_avoid"] is None
            or abs(ratio - out["ttc_over_t_avoid"]) > _RATIO_TOLERANCE * max(1.0, abs(ratio)))):
        raise ExposureError("payload.runtime.realized_ttc_at_first_detection_over_t_avoid disagrees with "
                            "the TTC / t_avoid")
    out["ttc_over_t_avoid"] = ratio
    if (first is None) == (reason is None):
        raise ExposureError("non_exposure_reason must be None exactly when the intruder was detected")
    samples = runtime.get("detected_samples")
    if type(samples) is not int or (samples > 0) != (first is not None):
        raise ExposureError("detected_samples > 0 exactly when a first detection is stamped (G23c F10)")
    record = runtime.get("exposure_record")
    _check_record(record, first)
    release = runtime.get("release")
    if not isinstance(release, dict) or release.get("rule") != rule:
        raise ExposureError("a runtime-rule stamp at D1..D4 needs payload.runtime.release of rule %s" % rule)
    closest = record["closest_after_visible"]
    if closest is not None:
        out["closest_gap_after_visible_m"] = closest["surface_gap_m"]
        out["contact_after_visible"] = closest["surface_gap_m"] <= 0.0
    if record["first_ray_return_s"] is not None:
        out["ray_return_lag_s"] = record["first_ray_return_s"] - first
    out["sync_error_s"] = record["sync_error_s"]
    unreleased_reason = UNRELEASED_REASONS[rule]
    if assignment.get("task") == SK_TASK and rule in SK_RULES:   # G23e: approach design
        _sk_approach_errors(release, assignment, dose)
    if release.get("applies") is False:            # stationary own ship: pinned schedule (G23c F13)
        if release.get("plan_kind") != "hold" or "release_time_s" in release:
            raise ExposureError("only a hold task stamps release.applies false, without a release time")
        if reason == unreleased_reason:
            raise ExposureError("a pinned-schedule stamp is never %r" % unreleased_reason)
        return out
    released = release.get("released")
    if type(released) is not bool or released != (release.get("release_time_s") is not None):
        raise ExposureError("payload.runtime.release.released must equal (release_time_s is not None)")
    cause = release.get("cause")
    if (cause is None) == released or (cause is not None and cause not in RULE_CAUSES[rule]):
        raise ExposureError("payload.runtime.release.cause %r does not match the release state" % (cause,))
    if first is not None and (not released or first < release["release_time_s"] - 1e-9):
        raise ExposureError("the intruder was detected before its release / reveal (the runtime rule never "
                            "allows it)")
    unreleased_quiet = not released and first is None and reason not in (
        "no_existing_range_sensor", "sensor_observations_missing")
    if (reason == unreleased_reason) != unreleased_quiet:
        raise ExposureError("non_exposure_reason %r must be used exactly for an unreleased, undetected intruder"
                            % unreleased_reason)
    out.update(released=released, release_cause=cause, release_time_s=release.get("release_time_s"))
    if rule == V3_RULE_VERSION and released:          # G23e layer B (REV-G23d M3, m2): v3 invariants
        for name in ("ttc_independent_at_reveal_s", "dose_undefined", "independent_measure", "cpa"):
            if name not in release:
                raise ExposureError("payload.runtime.release lacks %r (rule %s, G23e)" % (name, rule))
        if release["dose_undefined"] is not (release["ttc_independent_at_reveal_s"] is None):
            raise ExposureError("release.dose_undefined must equal (ttc_independent_at_reveal_s is None)")
        if bool(release.get("clamped")) is not (release.get("cause") in V3_CLAMPED_CAUSES):
            raise ExposureError("release.clamped must equal (cause in the clamped causes)")
        if (release.get("cause") == "cpa_reveal") != (release["cpa"] is not None):
            raise ExposureError("release.cpa is stamped exactly for a cpa_reveal")
        if release.get("dose_achieved_tol") != DOSE_ACHIEVED_TOL:
            raise ExposureError("release.dose_achieved_tol must be %r" % DOSE_ACHIEVED_TOL)
        if (isinstance(t_avoid, (int, float)) and isinstance(release.get("dose_ttc_s"), (int, float))
                and abs(release["dose_ttc_s"] - DOSE_TTC_RATIOS[dose] * t_avoid) > 1e-9 * max(1.0, t_avoid)):
            raise ExposureError("release.dose_ttc_s must be the dose ratio x t_avoid")
    if rule in DOSE_ACHIEVED_RULES and released:       # G23d L4 (REV-G23c H2): independent dose check, self-consistent
        primary = "ttc_independent_at_reveal_s" if rule == V3_RULE_VERSION else "ttc_cv_at_reveal_s"
        for name in (primary, "dose_achieved", "dose_achieved_tol", "dose_ttc_s", "ttc_at_reveal_s",
                     "clamped"):
            if name not in release:
                raise ExposureError("payload.runtime.release lacks %r (rule %s, G23d)" % (name, rule))
        value, dose_ttc, tol = release[primary], release["dose_ttc_s"], release["dose_achieved_tol"]
        expected = (isinstance(value, (int, float)) and isinstance(dose_ttc, (int, float)) and dose_ttc > 0
                    and abs(value / dose_ttc - 1.0) <= tol)
        if release["dose_achieved"] is not expected:
            raise ExposureError("payload.runtime.release.dose_achieved disagrees with ttc_cv_at_reveal_s / dose_ttc_s")
        out["dose_achieved"] = expected
        out["dose_undefined"] = bool(release.get("dose_undefined")) if rule == V3_RULE_VERSION else None
        out["clamped"] = bool(release["clamped"])
        if first is not None and isinstance(release["ttc_at_reveal_s"], (int, float)) and t_avoid:
            out["rule_ttc_at_first_detection_over_t_avoid"] = (
                release["ttc_at_reveal_s"] - (first - release["release_time_s"])) / t_avoid
    return out


def _quartiles(values):
    values = sorted(values)
    if not values:
        return None
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) >= 2 else [values[0]] * 3
    return {"n": len(values), "median": statistics.median(values), "q25": q[0], "q75": q[2],
            "min": values[0], "max": values[-1]}


def exposure_summary(envelopes, dose):
    """G23b/G23c/G23d: per-cell summary of D7 stamp envelopes (one certificate cell = one dose). Headline (G23d,
    REV-G23c H8): released / detected shares, the dose_achieved and clamped shares, the rule's TTC at the first
    detection, the realized exposure of runtime.exposure_record; metric A under "diagnostic"."""
    if type(dose) is not int or not 0 <= dose < DOSES:
        raise ExposureError("the cell dose must be an integer D0..D4")
    readings, not_admitted, rules = [], 0, set()
    for envelope in envelopes:
        reading = runtime_exposure(envelope)
        if reading is None:
            not_admitted += 1
            continue
        if reading["dose"] != dose:
            raise ExposureError("a D%d stamp in a D%d cell" % (reading["dose"], dose))
        readings.append(reading)
        rules.add(reading["rule"])
    if len(rules) > 1:
        raise ExposureError("one cell mixes D7 runtime rules: %s" % sorted(str(r) for r in rules))
    n = len(readings)

    def share(count):
        return None if n == 0 or dose == 0 else count / n

    released = sum(1 for r in readings if r["released"])
    detected = sum(1 for r in readings if r["detected"])
    finite = [r["ttc_over_t_avoid"] for r in readings if r["finite_ttc"]]
    contact = sum(1 for r in readings if r["contact_after_visible"])
    causes, reasons = {}, {}
    for r in readings:
        if r["release_cause"] is not None:
            causes[r["release_cause"]] = causes.get(r["release_cause"], 0) + 1
        key = "detected" if r["non_exposure_reason"] is None else r["non_exposure_reason"]
        reasons[key] = reasons.get(key, 0) + 1
    checked = [r for r in readings if r["dose_achieved"] is not None]
    achieved = sum(1 for r in checked if r["dose_achieved"])
    clamped = sum(1 for r in readings if r["clamped"])
    extra = {}
    if V3_RULE_VERSION in rules:                     # G23e layer B (REV-G23d m5): undefined counted apart
        undefined = sum(1 for r in readings if r["dose_undefined"])
        extra = {"dose_undefined": undefined, "dose_undefined_share": (undefined / n) if n and dose else None}
    return dict(extra, **{"schema": SUMMARY_SCHEMA, "metric": SUMMARY_HEADLINE, "rule": next(iter(rules)) if rules else None,
            "dose": dose, "records": n + not_admitted, "not_admitted": not_admitted, "admitted": n,
            "released": released, "detected": detected,
            "released_share": share(released), "detected_share": share(detected),
            "dose_achieved": achieved if checked else None,
            "dose_achieved_share": (achieved / n) if checked and n and dose else None,
            "clamped": clamped if checked else None,
            "clamped_share": (clamped / n) if checked and n and dose else None,
            "rule_ttc_at_first_detection_over_t_avoid": _quartiles(
                [r["rule_ttc_at_first_detection_over_t_avoid"] for r in readings
                 if r["rule_ttc_at_first_detection_over_t_avoid"] is not None]),
            "diagnostic": {"metric_A": EXPOSURE_METRIC, "finite_ttc_at_detection": len(finite),
                           "finite_ttc_at_detection_share": share(len(finite)),
                           "ttc_over_t_avoid_at_detection": _quartiles([v for v in finite if v is not None])},
            "realized": {"contact_after_visible": contact, "contact_after_visible_share": share(contact),
                         "closest_gap_after_visible_m": _quartiles([r["closest_gap_after_visible_m"] for r in readings
                                                                    if r["closest_gap_after_visible_m"] is not None]),
                         "ray_return_lag_s": _quartiles([r["ray_return_lag_s"] for r in readings
                                                         if r["ray_return_lag_s"] is not None]),
                         "sync_error_s": _quartiles([r["sync_error_s"] for r in readings
                                                     if r["sync_error_s"] is not None])},
            "release_causes": dict(sorted(causes.items())), "outcomes": dict(sorted(reasons.items()))})


__all__ = ["ADMITTED", "NOT_ADMITTED", "JOINT_ECHO", "BEYOND_OFFERED_GRID", "REASON_SOURCES",
           "NOT_ADMITTED_ASSIGNMENT_KEYS", "ExposureError", "exposure_of", "is_beyond_offered_grid",
           "d7_primitive", "per_dose_counts", "RUNTIME_RULE_VERSION", "PINNED_RULE_VERSION", "DIAGNOSTIC_RULES",
           "RULE_CAUSES", "UNRELEASED_REASONS", "NOT_RELEASED", "RELEASE_CAUSES", "NON_EXPOSURE_REASONS",
           "EXPOSURE_RECORD_SCHEMA", "runtime_exposure", "exposure_summary", "not_applicable_doses",
           "NOT_APPLICABLE", "SK_DESIGN", "V3_RULE_VERSION"]
