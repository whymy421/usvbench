"""G3 appearing-suite stamps for the evaluators, the validator and the judges (pure Python).

Loads tasks/_shared/conditions/appearing/stamp.py BY PATH (standard library only; the
``tasks`` package initialisers never run) and exposes the helpers the certificate
writer (cert_v3), the validator (validate_cert_v3) and analysis tools need:

  exposure_of(envelope)      -> ("admitted" | "not_admitted", reasons) after the digest check
  read(envelope)             -> the verified envelope
  header_errors(block, records) -> list of error strings for header.appearing_hazards_header
  grid_check(block, num_envs, quota)  CONTEXT_R4 D24 (d) preflight for the appearing pool grid
  trigger_errors(envelope)   G20b: the trigger record's realized closing speed / TTC recompute
  realized_exposure_table(documents)  G20c: per (task, vessel, controller, dose) the ASSIGNED episodes, how
                             many realized the exposure (the trigger gap was reached), the share and its one-sided
                             90% Wilson lower bound (wilson_lower_bound); G20d: of the realized ones, how many were
                             visible / occluded at the trigger (stamp runtime.visible_at_trigger, decision 7A)
  applicability_check(block) G20d: the header's dose_applicability / not_applicable_doses equal the pinned
                             ladder.NOT_APPLICABLE_DOSES of its (task, vessel); a not-applicable dose has count row
                             [0, 0, 0] and is never the header dose (decision 2B: BlueBoat station keeping D1)
  trigger_speed_summary(document) / trigger_speed_table(documents)  G35 (J19, v2.6.1): how many appearing
                             episodes triggered with the pose speed outside the vessel envelope (recorded in the
                             stamp, never refused)
G20b: the dose range, the number of per-dose count rows and the header's ladder (suite version, gap
ratios) follow the ACTIVE ladder option (tasks/_shared/conditions/appearing/ladder.py, read through
the stamp module); a stamp's trigger record must carry closing_speed_mps and realized_ttc_s that
recompute from its gap, bearing and speed (stamp.trigger_errors, applied by read()).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ADMITTED = "admitted"
NOT_ADMITTED = "not_admitted"
BEYOND_OFFERED_GRID = "beyond_offered_grid"
PER_DOSE_COUNT_FIELDS = ["offered", "admitted", "not_admitted"]
_MODULE = {}


def stamp_module():
    if "stamp" not in _MODULE:
        path = Path(__file__).resolve().parents[1] / "tasks" / "_shared" / "conditions" / "appearing" / "stamp.py"
        if not path.is_file():
            raise ValueError("cannot verify appearing stamps: %s is missing" % path)
        spec = importlib.util.spec_from_file_location("_g3_appearing_stamp", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE["stamp"] = module
    return _MODULE["stamp"]


def read(envelope):
    return stamp_module().read_appearing_stamp(envelope)


def exposure_of(envelope):
    return stamp_module().exposure_of(envelope)


def dose_count():
    """Number of appearing doses D0..D_max of the active ladder (5, or 6 under option E)."""
    return int(stamp_module().MAX_DOSE) + 1


def per_dose_counts(block):
    """[{offered, admitted, not_admitted}] x dose_count() of an appearing header block (fail closed)."""
    rows = block.get("per_dose_counts")
    count = dose_count()
    if block.get("per_dose_counts_fields") != PER_DOSE_COUNT_FIELDS or not isinstance(rows, list) or \
            len(rows) != count:
        raise ValueError("appearing_hazards_header.per_dose_counts must be %d [offered, admitted, not_admitted] "
                         "rows, one per dose D0..D%d (CONTEXT_R4 D22)" % (count, count - 1))
    out = []
    for row in rows:
        if not (isinstance(row, list) and len(row) == 3 and all(type(v) is int and v >= 0 for v in row)
                and row[1] + row[2] == row[0]):
            raise ValueError("appearing per-dose count row %r is not [offered, admitted, not_admitted]" % (row,))
        out.append(dict(zip(PER_DOSE_COUNT_FIELDS, row)))
    return out


def grid_of(block):
    grid = block.get("grid") if isinstance(block, dict) else None
    if not isinstance(grid, dict) or any(type(grid.get(k)) is not int or grid.get(k) <= 0
                                         for k in ("num_envs", "episodes")):
        raise ValueError("appearing_hazards_header.grid must be {'num_envs': N, 'episodes': E} (CONTEXT_R4 D24 (c))")
    return {"num_envs": grid["num_envs"], "episodes": grid["episodes"]}


def grid_check(block, num_envs, quota):
    grid = grid_of(block)
    if int(quota) > grid["episodes"]:
        raise ValueError("journal run refused: per-env quota %d exceeds the appearing pool's offered grid episodes "
                         "%d (CONTEXT_R4 D24 (d))" % (int(quota), grid["episodes"]))
    if int(num_envs) > grid["num_envs"]:
        raise ValueError("journal run refused: num_envs %d exceeds the appearing pool's offered grid num_envs %d "
                         "(CONTEXT_R4 D24 (d))" % (int(num_envs), grid["num_envs"]))
    return grid


def pinned_check(block):
    if block.get("pool_pinned", False) is not True:
        raise ValueError("journal certificate refused: appearing_hazards_header.pool_pinned is not true; the "
                         "appearing pool is not in the family's pinned manifest (CONTEXT_R4 D23 applied to G3)")
    manifest = block.get("pool_manifest_sha256")
    if not (isinstance(manifest, str) and len(manifest) == 64 and all(c in "0123456789abcdef" for c in manifest)):
        raise ValueError("appearing_hazards_header.pool_manifest_sha256 must be the pinned manifest SHA-256 (D19)")


HEADER_LADDER_FIELDS = ("suite_version", "ladder_option", "trigger_gaps_m", "t_avoid_s")


def header_vessel(block):
    suite = block.get("suite")
    return suite.get("vessel_id") if isinstance(suite, dict) else None


def header_task(block):
    group = block.get("hazard_group")
    prefix = "g3_appearing_"
    return group[len(prefix):] if isinstance(group, str) and group.startswith(prefix) else None


def ladder_check(block):
    """G20b/G20c: the header names the active ladder; suite_version, ladder_option, trigger_gaps_m, t_avoid_s and
    suite.vessel_id are MANDATORY (REV-G20b M1), t_avoid_s is the vessel's pinned value and the gaps / k recompute
    from the pinned constants."""
    module = stamp_module()
    lad = module.LADDER
    missing = [f for f in HEADER_LADDER_FIELDS if block.get(f) is None]
    vessel = header_vessel(block)
    if vessel is None:
        missing.append("suite.vessel_id")
    if missing:
        raise ValueError("appearing_hazards_header lacks %s (mandatory since G20c)" % ", ".join(missing))
    if vessel not in lad.T_AVOID_S:
        raise ValueError("appearing_hazards_header.suite.vessel_id %r has no pinned ladder" % (vessel,))
    if block["t_avoid_s"] != lad.T_AVOID_S[vessel]:
        raise ValueError("appearing_hazards_header.t_avoid_s %r is not the pinned %r of %s"
                         % (block["t_avoid_s"], lad.T_AVOID_S[vessel], vessel))
    if block["suite_version"] != module.SUITE_VERSION:
        raise ValueError("appearing_hazards_header.suite_version %r is not the active ladder's %r (ladder option %s)"
                         % (block["suite_version"], module.SUITE_VERSION, module.LADDER_OPTION))
    if block["ladder_option"] != lad.ACTIVE_LADDER:
        raise ValueError("appearing_hazards_header.ladder_option %r is not the active %r"
                         % (block["ladder_option"], lad.ACTIVE_LADDER))
    if "gap_ratios" in block:
        want = None if lad.GAP_RATIOS is None else list(lad.GAP_RATIOS)
        if block["gap_ratios"] != want:
            raise ValueError("appearing_hazards_header.gap_ratios %r are not the active ladder's %r (ladder option %s)"
                             % (block["gap_ratios"], want, lad.ACTIVE_LADDER))
    t_avoid = lad.T_AVOID_S[vessel]
    want = [lad.gap_m(d, t_avoid, vessel) for d in range(lad.DOSE_COUNT)]
    if block["trigger_gaps_m"] != want:
        raise ValueError("appearing_hazards_header.trigger_gaps_m %r are not the active ladder's %r"
                         % (block["trigger_gaps_m"], want))
    k_eq = block.get("k_equivalent")
    if k_eq is not None and k_eq != [lad.k_equivalent(g, t_avoid) for g in want]:
        raise ValueError("appearing_hazards_header.k_equivalent does not recompute from the trigger gaps")
    override = block.get("gap_override_m")
    if override is not None:
        task = header_task(block)
        low = lad.GAP_OVERRIDE_MIN_M[vessel]
        high = lad.start_gap_m(task, vessel) if task else None
        if not isinstance(override, (int, float)) or isinstance(override, bool) or override < low - 1e-12 or \
                (high is not None and override > high + 1e-12):
            raise ValueError("appearing_hazards_header.gap_override_m %r outside [calibration rung %r, start gap %r] m"
                             % (override, low, high))


def applicability_check(block):
    """G20d (decision 2B): dose applicability of the header equals the pins; not-applicable rows are [0, 0, 0]."""
    lad = stamp_module().LADDER
    task, vessel = header_task(block), header_vessel(block)
    if task is None or vessel is None:
        return                                   # reported by ladder_check
    skip = lad.not_applicable_doses(task, vessel)
    want = {"D%d" % d: (lad.NOT_APPLICABLE if d in skip else "offered") for d in range(1, lad.DOSE_COUNT)}
    if block.get("dose_applicability") != want:
        raise ValueError("appearing_hazards_header.dose_applicability %r is not the pinned %r for %s on %s (G20d)"
                         % (block.get("dose_applicability"), want, task, vessel))
    reasons = {"D%d" % d: lad.not_applicable_reason(task, vessel, d) for d in skip}
    if block.get("not_applicable_doses") != reasons:
        raise ValueError("appearing_hazards_header.not_applicable_doses does not equal the pinned reasons (G20d)")
    if block.get("dose") in skip:
        raise ValueError("appearing_hazards_header.dose D%r is not applicable by design for %s on %s (never offered)"
                         % (block.get("dose"), task, vessel))
    rows = block.get("per_dose_counts")
    if isinstance(rows, list):
        for d in skip:
            if d < len(rows) and rows[d] != [0, 0, 0]:
                raise ValueError("appearing per-dose count row of the not-applicable D%d must be [0, 0, 0], got %r"
                                 % (d, rows[d]))


def _official(registration_purpose):
    """G20c (REV-G20b m4): a missing registration counts as OFFICIAL (as the G16 / T25 validators treat it)."""
    return registration_purpose != "diagnostic"


def gap_override_official_check(block, registration_purpose):
    """G20b/G20c: a diagnostic gap override (appearing.gap_m) never reaches an official certificate."""
    if isinstance(block, dict) and block.get("gap_override_m") is not None and _official(registration_purpose):
        raise ValueError("official run (seed purpose %r) with the diagnostic appearing.gap_m override %r"
                         % (registration_purpose, block["gap_override_m"]))


def supplementary_dose_official_check(block, registration_purpose):
    """G20c (REV-G20b m5): the supplementary admission rung (appearing=D5 under ladder M) is diagnostic only."""
    supplementary = stamp_module().LADDER.SUPPLEMENTARY_DOSES
    if isinstance(block, dict) and block.get("dose") in supplementary and _official(registration_purpose):
        raise ValueError("official run (seed purpose %r) at the supplementary appearing dose D%r (a calibration "
                         "admission rung, not a main-table dose)" % (registration_purpose, block["dose"]))


def trigger_errors(envelope):
    """G20b: trigger-record consistency of one verified stamp (realized closing speed and TTC recompute)."""
    payload = read(envelope)["payload"]
    return stamp_module().trigger_errors(payload["runtime"])


def header_errors(block, records):
    """Validator checks of header.appearing_hazards_header against the records."""
    errors = []
    if not isinstance(block, dict):
        return ["appearing_hazards_header must be an object"]
    for check in (pinned_check, grid_of, ladder_check, applicability_check):
        try:
            check(block)
        except ValueError as exc:
            errors.append(str(exc))
    try:
        counts = per_dose_counts(block)
    except ValueError as exc:
        return errors + [str(exc)]
    dose = block.get("dose")
    if type(dose) is not int or not 0 <= dose < dose_count():
        return errors + ["appearing_hazards_header.dose must be an integer D0..D%d" % (dose_count() - 1)]
    keys = set()
    for index, record in enumerate(records):
        envelope = record.get("appearing_hazards")
        if envelope is None:
            errors.append("records/%d: a certificate with an appearing header needs every record's appearing stamp"
                          % index)
            continue
        try:
            kind, _ = exposure_of(envelope)
            payload = read(envelope)["payload"]
        except (ValueError, TypeError, KeyError):
            continue      # reported by the record check
        if payload["exposure"] == ADMITTED and payload["assignment"]["dose"] != dose:
            errors.append("records/%d: appearing stamp dose %r differs from the header dose %d"
                          % (index, payload["assignment"]["dose"], dose))
        override = block.get("gap_override_m")
        if payload["exposure"] == ADMITTED and payload["runtime"].get("gap_override_m") != override:
            errors.append("records/%d: appearing stamp gap override %r differs from the header's %r"
                          % (index, payload["runtime"].get("gap_override_m"), override))
        gaps = block.get("trigger_gaps_m")
        if payload["exposure"] == ADMITTED and isinstance(gaps, list):
            # G20c (REV-G20b M1): the record's trigger gap is the header's gap of its dose (or the header override);
            # a header without trigger_gaps_m / suite.vessel_id is reported once by ladder_check
            stamp_dose = payload["assignment"]["dose"]
            want = override if (override is not None and stamp_dose > 0) else (
                gaps[stamp_dose] if 0 <= stamp_dose < len(gaps) else None)
            if payload["runtime"].get("trigger_gap_m") != want:
                errors.append("records/%d: appearing trigger_gap_m %r differs from the header's %r at D%d"
                              % (index, payload["runtime"].get("trigger_gap_m"), want, stamp_dose))
        if payload["exposure"] == ADMITTED and header_vessel(block) is not None:
            if payload["assignment"].get("vessel_id") != header_vessel(block):
                errors.append("records/%d: appearing stamp vessel %r differs from the header's %r"
                              % (index, payload["assignment"].get("vessel_id"), header_vessel(block)))
        if kind == NOT_ADMITTED:
            key = payload["assignment"]["key"]
            keys.add((key["env"], key["episode"]))
    if len(keys) > counts[dose]["not_admitted"]:
        errors.append("%d not-admitted appearing keys in the records but the header counts %d at D%d"
                      % (len(keys), counts[dose]["not_admitted"], dose))
    return errors


WILSON_Z_ONE_SIDED_90 = 1.2815515655446004      # standard normal 0.90 quantile


def wilson_lower_bound(successes, trials, z=WILSON_Z_ONE_SIDED_90):
    """One-sided Wilson score lower bound of a binomial share (None without trials)."""
    if trials <= 0:
        return None
    p = successes / trials
    z2 = z * z
    centre = p + z2 / (2.0 * trials)
    spread = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return max(0.0, (centre - spread) / (1.0 + z2 / trials))


def realized_exposure_table(documents):
    """G20c: exposure_realized per (task, vessel, controller, dose) over verified certificates.

    Denominator = ASSIGNED episodes only (admitted keys, assigned true, itt_count 1; D27): not-admitted keys are
    counted separately and never enter the share. Every stamp is read with the digest check and the pins. Rows:
    {"task", "vessel_id", "controller", "dose", "trigger_gap_m", "assigned", "realized", "not_admitted",
     "share", "wilson_lower_90", "realized_visible", "realized_occluded", "occluded_share"} sorted by key (G20d:
    occluded_share = realized episodes whose cylinder was not visible at the trigger / realized); a certificate
    without an appearing header is skipped.
    """
    table = {}
    for document in documents:
        header = document.get("header", {})
        block = header.get("appearing_hazards_header")
        if not isinstance(block, dict):
            continue
        controller = header.get("controller")
        controller = controller.get("name") if isinstance(controller, dict) else controller
        key = (header_task(block), header_vessel(block), controller, block.get("dose"))
        gaps = block.get("trigger_gaps_m")
        gap = block.get("gap_override_m") if block.get("gap_override_m") is not None else (
            gaps[block["dose"]] if isinstance(gaps, list) and isinstance(block.get("dose"), int)
            and 0 <= block["dose"] < len(gaps) else None)
        row = table.setdefault(key, {"task": key[0], "vessel_id": key[1], "controller": key[2], "dose": key[3],
                                     "trigger_gap_m": gap, "assigned": 0, "realized": 0, "not_admitted": 0,
                                     "realized_visible": 0, "realized_occluded": 0})
        for record in document.get("records", []):
            envelope = record.get("appearing_hazards")
            if envelope is None:
                continue
            payload = read(envelope)["payload"]
            if payload["exposure"] != ADMITTED:
                row["not_admitted"] += 1
                continue
            row["assigned"] += 1
            if payload["runtime"].get("exposure_realized") is True:
                row["realized"] += 1
                visible = payload["runtime"].get("visible_at_trigger")
                row["realized_visible" if visible is True else "realized_occluded"] += 1
    rows = []
    for key in sorted(table, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        row = table[key]
        row["share"] = row["realized"] / row["assigned"] if row["assigned"] else None
        row["wilson_lower_90"] = wilson_lower_bound(row["realized"], row["assigned"])
        row["occluded_share"] = row["realized_occluded"] / row["realized"] if row["realized"] else None
        rows.append(row)
    return rows


def record_errors(document):
    """Every per-record appearing stamp re-derives its digest; a counted beyond-grid record is an error."""
    errors = []
    for index, record in enumerate(document["records"]):
        envelope = record.get("appearing_hazards")
        if envelope is None:
            continue
        try:
            kind, reasons = exposure_of(envelope)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append("records/%d: appearing_hazards envelope rejected: %s" % (index, exc))
            continue
        if kind == NOT_ADMITTED and BEYOND_OFFERED_GRID in reasons:
            errors.append("records/%d: counted journal record of a key beyond the offered appearing grid "
                          "(CONTEXT_R4 D24)" % index)
    return errors


def trigger_speed_summary(document):
    """G35 (controller decision J19, v2.6.1): the certificate-level count of appearing episodes whose trigger pose
    speed lay outside the vessel envelope. The fact lives in each record's sealed stamp (runtime.
    trigger_speed_outside_envelope / trigger_speed_envelope, written only then; stamp.speed_envelope_errors checks
    it on every read); it is never refused. No header field carries the count: header.additionalProperties is
    false and analysis_input requires paired appearing headers to be identical, so a per-cell count there would
    break pairing. Returns {"records_with_stamp", "admitted", "realized", "outside_envelope", "episodes":
    [{"index", "env", "ep", "speed_mps", "envelope_mps"}]} (every stamp read with the digest check and the pins)."""
    flag = stamp_module().SPEED_ENVELOPE_FLAG
    record_key = stamp_module().SPEED_ENVELOPE_RECORD
    out = {"records_with_stamp": 0, "admitted": 0, "realized": 0, "outside_envelope": 0, "episodes": []}
    for index, record in enumerate(document.get("records", [])):
        envelope = record.get("appearing_hazards")
        if envelope is None:
            continue
        payload = read(envelope)["payload"]
        out["records_with_stamp"] += 1
        if payload["exposure"] != ADMITTED:
            continue
        out["admitted"] += 1
        runtime = payload["runtime"]
        out["realized"] += runtime.get("exposure_realized") is True
        if runtime.get(flag) is True:
            out["outside_envelope"] += 1
            out["episodes"].append({"index": index, "env": record.get("env"), "ep": record.get("ep"),
                                    "speed_mps": runtime[record_key]["speed_mps"],
                                    "envelope_mps": runtime[record_key]["envelope_mps"]})
    return out


def trigger_speed_table(documents):
    """G35 (J19): trigger_speed_summary per (task, vessel, controller, dose) over verified certificates, in the
    realized_exposure_table key order; rows {"task", "vessel_id", "controller", "dose", "admitted", "realized",
    "outside_envelope", "share_of_realized"} (a certificate without an appearing header is skipped)."""
    table = {}
    for document in documents:
        header = document.get("header", {})
        block = header.get("appearing_hazards_header")
        if not isinstance(block, dict):
            continue
        controller = header.get("controller")
        controller = controller.get("name") if isinstance(controller, dict) else controller
        key = (header_task(block), header_vessel(block), controller, block.get("dose"))
        row = table.setdefault(key, {"task": key[0], "vessel_id": key[1], "controller": key[2], "dose": key[3],
                                     "admitted": 0, "realized": 0, "outside_envelope": 0})
        summary = trigger_speed_summary(document)
        for name in ("admitted", "realized", "outside_envelope"):
            row[name] += summary[name]
    rows = []
    for key in sorted(table, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        row = table[key]
        row["share_of_realized"] = row["outside_envelope"] / row["realized"] if row["realized"] else None
        rows.append(row)
    return rows


__all__ = ["ADMITTED", "NOT_ADMITTED", "BEYOND_OFFERED_GRID", "stamp_module", "read", "exposure_of",
           "per_dose_counts", "grid_of", "grid_check", "pinned_check", "header_errors", "record_errors",
           "dose_count", "ladder_check", "trigger_errors", "gap_override_official_check",
           "supplementary_dose_official_check", "HEADER_LADDER_FIELDS", "wilson_lower_bound",
           "realized_exposure_table", "WILSON_Z_ONE_SIDED_90", "applicability_check",
           "trigger_speed_summary", "trigger_speed_table"]
