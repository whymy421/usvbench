"""The owner's green-water exclusion rule "A + B" (G10; pure Python, no torch/numpy).

Owner decision 2026-10-01 (NC 09_decision log; 34_full-dose smoke plan section 7):
a cell (task x vessel x wave level x controller) is EXCLUDED from the main
table and listed separately when EITHER
  (a) the formula probability exceeds 0.05: D9's per-cycle deck wetness under
      wave following (NORDFORSK criterion 0.05; green_water_formula.py), OR
  (b) the simulated green-water episode share exceeds 0.05 (green_water.py:
      a deck-edge point below the instantaneous water surface for at least
      N consecutive physics substeps).
Both limits were fixed before any data and are never relaxed afterwards
(``FORMULA_LIMIT`` / ``SHARE_LIMIT`` are constants; the validator rejects a
certificate that stamps any other value). The number of episodes with ANY
green water is always reported, whatever the verdict.

This module is imported by the certificate writer (scripts/cert_v3.py, cell
summary at build time) and by the validator (scripts/validate_cert_v3.py,
loaded by path), so the verdict has exactly one implementation.

T25, owner decision T1.35c (2026-10-02): the verdict USED FOR EXCLUSION is now
rule/2 (``RULE2_VERSION``, G18's candidate R2): a cell is excluded when the
formula per-cycle probability exceeds 0.05 OR the simulated PER-CYCLE
exceedance at the worst declared deck-edge point after the 3 s settling window
(G18 diagnostics: freeboard up-crossings / mean-level up-crossings, summed over
the counted episodes) exceeds 0.05 - the same unit on both sides. It is written
to header.green_water.exclusion (binding); rule/1 above is still computed and
reported unchanged in header.green_water.cell (not binding).

G28, owner decision 2026-10-03 17:0x (decision pack 02 item 3): the WAM-V
exclusion PROBE is the deck edge (six deck-plate edge points); the pontoon top
line is report-only (header.green_water.report_points / report, set
"wamv_pontoon_top"). BlueBoat is unchanged (its points are its deck edge).
header.green_water.exclusion.probe (PROBE_VERSION) names the probe set and its
labels; the validator checks that header.green_water.detection.points carry
exactly those labels. An exclusion block without "probe" is a v2.5 (T25)
certificate: still valid, with the v2.5 meaning (WAM-V pontoon top binding);
a WAM-V header whose binding points are the deck edge must carry the probe.
"""
from __future__ import annotations

import math

RULE_VERSION = "usvbench.green_water.rule/1"
HEADER_SCHEMA = "usvbench.green_water_header/1"
DETECTION_VERSION = "usvbench.green_water.detect/1"
FORMULA_VERSION = "usvbench.green_water.formula/1"     # = green_water_formula.FORMULA_VERSION
FORMULA_LIMIT = 0.05
SHARE_LIMIT = 0.05
# N consecutive physics substeps (see green_water.py N_CONSECUTIVE_SUBSTEPS for the rationale).
N_CONSECUTIVE_SUBSTEPS = 6
RULE_TEXT = (
    "A cell is EXCLUDED from the main table and listed separately when EITHER (a) the formula "
    "probability (D9 wave-following per-cycle deck wetness, NORDFORSK criterion) exceeds 0.05 OR (b) the "
    "share of episodes with simulated green water (a declared deck-edge point below the instantaneous "
    "water surface for at least N consecutive physics substeps) exceeds 0.05. Both 0.05 limits were "
    "fixed before any data and are never relaxed. The number of episodes with any green water is always "
    "reported, whatever the verdict.")
RECORD_FIELDS = ("green_water", "first_time_s", "first_point", "submerged_point_substeps",
                 "longest_run_substeps", "substeps")

# G18 (2026-10-02): read-only DIAGNOSTICS beside the rule/1 record. They never change the rule/1 record,
# the cell summary or the verdict; they add (per episode, records[].green_water_diagnostics) the detector
# re-run after a settling window that skips the reset transient (every family spawns the hull at the calm
# draft, level and at rest while the wave clock starts at 0), the number of qualifying events and the
# exposure after that window, and per point the up-crossings of the calm freeboard and of the calm mean
# level by the relative surface height (their ratio is the simulated PER-CYCLE exceedance, the formula's
# unit; Rice's formula). The header block header.green_water.diagnostics aggregates them over the counted
# episodes and lists candidate rules for the owner, explicitly non-binding.
DIAGNOSTICS_VERSION = "usvbench.green_water.diagnostics/1"
# Settling window, s after the reset (G18 REPORT section 1: CPU replay of the G15 cells; the hull deviation
# from its wave-following state reaches the steady value of a hull started ON the filtered surface after
# about 1 to 2 s on BlueBoat, and the heave transient of the WAM-V has decayed after about 3 s).
SETTLE_S = 3.0
REFERENCE_DURATION_S = 60.0
DIAGNOSTIC_FIELDS = ("version", "settle_s", "green_water_settled", "first_time_settled_s", "first_point_settled",
                     "events_settled", "exposure_settled_s", "freeboard_upcrossings", "mean_level_upcrossings")
# T25, owner decision T1.35c (2026-10-02): rule/2 = G18's R2 is the BINDING exclusion verdict (module docstring).
RULE2_VERSION = "usvbench.green_water.rule/2"
RULE2_DECISION = ("owner decision T1.35c, 2026-10-02: green-water rule/2 (G18 candidate R2) is the verdict used for "
                  "exclusion; rule/1 is still reported")
PER_CYCLE_LIMIT = 0.05
RULE2_TEXT = (
    "A cell is EXCLUDED from the main table and listed separately when EITHER (a) the formula probability (D9 "
    "wave-following per-cycle deck wetness, NORDFORSK criterion) exceeds 0.05 OR (b) the simulated per-cycle "
    "exceedance at the worst declared deck-edge point exceeds 0.05: per point, the up-crossings of the calm "
    "freeboard divided by the up-crossings of the calm mean level by the relative surface height, after the "
    "settling window, summed over the counted episodes (header.green_water.diagnostics). Both sides are per-cycle "
    "probabilities. Both 0.05 limits are fixed and never relaxed. Rule/1 (formula or simulated episode share, "
    "header.green_water.cell) is still reported and is not binding.")
# T25 (owner decision T1.35d, 2026-10-02): REPORT-ONLY point sets (WAM-V deck edge, green_water.report_point_sets)
# observed beside the declared set; per record records[].green_water_report = {set: {record, diagnostics}} (the
# rule/1 record and G18 diagnostics forms), summarised in header.green_water.report; never part of any verdict.
REPORT_VERSION = "usvbench.green_water.report_points/1"
REPORT_DECISION = ("owner decision T1.35d, 2026-10-02 (data first) and G28 owner decision 2026-10-03 17:0x item 3: the "
                   "WAM-V deck edge is the binding probe, the pontoon top line is the report-only set "
                   "wamv_pontoon_top; never used by the verdict")
# G28 (owner decision 2026-10-03 17:0x, decision pack 02 item 3): the exclusion probe per vessel. The labels are those
# of green_water.deck_edge_points (pure strings here: this module stays importable without torch / the registry).
PROBE_VERSION = "usvbench.green_water.probe/2"
PROBE_DECISION = ("owner decision 2026-10-03 17:0x (NC 09 decision record, reply to decision pack 02, item 3): the "
                  "WAM-V green-water exclusion probe is the deck edge; the pontoon top stays report-only; BlueBoat "
                  "unchanged")
_WAMV_DECK_EDGE_LABELS = tuple(f"{side}_deck_edge_{station}" for side in ("port", "starboard")
                               for station in ("aft", "mid", "fore"))
_WAMV_PONTOON_LABELS = tuple(f"{side}_pontoon_top_{station}" for side in ("port", "starboard")
                             for station in ("stern", "aft", "mid", "fore", "bow"))
_BLUEBOAT_DECK_EDGE_LABELS = tuple(f"{side}_{edge}_{station}" for side in ("port", "starboard")
                                   for edge in ("outer", "inner") for station in ("bow", "mid", "stern"))
PROBE_SETS = {
    "blueboat": {"set": "blueboat_deck_edge", "labels": _BLUEBOAT_DECK_EDGE_LABELS, "report_only_sets": (),
                 "geometry": "outer and inner deck edge (gunwale) of both hulls at the registry freeboard, bow / mid / "
                             "stern (12 points)"},
    "wamv": {"set": "wamv_deck_edge", "labels": _WAMV_DECK_EDGE_LABELS, "report_only_sets": ("wamv_pontoon_top",),
             "geometry": "perimeter of the top face of the VRX deck plate (top_base, base_link z 1.30 m), aft / mid / "
                         "fore of both long edges (6 points)"},
}


def rule_block():
    """Static rule stamp (header.green_water.rule)."""
    return {"version": RULE_VERSION, "text": RULE_TEXT, "formula_limit": FORMULA_LIMIT,
            "share_limit": SHARE_LIMIT, "comparison": "strictly greater than the limit excludes",
            "limits_fixed_before_data": True, "limits_relaxable": False,
            "always_reported": "episodes_with_green_water and episodes"}


def verdict(formula_probability, share):
    """(excluded, reasons): excluded = formula > 0.05 or share > 0.05 (share None = no episodes)."""
    reasons = []
    if formula_probability > FORMULA_LIMIT:
        reasons.append("formula_probability_exceeds_0.05")
    if share is not None and share > SHARE_LIMIT:
        reasons.append("simulated_green_water_share_exceeds_0.05")
    return bool(reasons), reasons


def cell_summary(formula_probability, green_records):
    """Header cell block from the counted episodes' green-water records."""
    n = len(green_records)
    k = sum(1 for r in green_records if r["green_water"] is True)
    share = (k / n) if n else None
    excluded, reasons = verdict(float(formula_probability), share)
    return {"rule_version": RULE_VERSION, "episodes": n, "episodes_with_green_water": k, "share": share,
            "formula_probability": float(formula_probability), "excluded_green_water": excluded,
            "reasons": reasons}


def _is_int(value):
    return type(value) is int


def record_errors(record, n_required, physics_dt_s, labels, prefix=""):
    """Internal consistency of one per-episode green-water record."""
    errors = []
    if not isinstance(record, dict):
        return [prefix + "green_water record must be an object"]
    missing = [k for k in RECORD_FIELDS if k not in record]
    if missing:
        return [prefix + "green_water record lacks " + ", ".join(missing)]
    green = record["green_water"]
    if not isinstance(green, bool):
        return [prefix + "green_water must be a bool"]
    for key in ("submerged_point_substeps", "longest_run_substeps", "substeps"):
        if not _is_int(record[key]) or record[key] < 0:
            errors.append(prefix + key + " must be a nonnegative integer")
    if errors:
        return errors
    longest, total, substeps = (record["longest_run_substeps"], record["submerged_point_substeps"],
                                record["substeps"])
    if longest > substeps or longest > total or total > substeps * max(1, len(labels)):
        errors.append(prefix + "green_water counters are inconsistent (longest <= substeps, longest <= total, "
                               "total <= substeps x points)")
    first, point = record["first_time_s"], record["first_point"]
    if green:
        if longest < n_required:
            errors.append(prefix + "green_water true with a longest run below N consecutive substeps")
        if isinstance(first, bool) or not isinstance(first, (int, float)) or not math.isfinite(first) or first < 0:
            errors.append(prefix + "green_water true needs a finite first_time_s >= 0")
        elif first > (substeps - n_required) * physics_dt_s + 1e-9:
            errors.append(prefix + "first_time_s is later than the last possible onset of an N-substep run")
        if point not in labels:
            errors.append(prefix + "first_point must name a declared deck-edge point")
    else:
        if longest >= n_required:
            errors.append(prefix + "a run of N consecutive submerged substeps must set green_water true")
        if first is not None or point is not None:
            errors.append(prefix + "green_water false needs first_time_s and first_point null")
    return errors


def header_errors(block, records):
    """header.green_water against the counted records (validator; the verdict recomputed here)."""
    errors = []
    if not isinstance(block, dict):
        return ["header.green_water must be an object"]
    if block.get("schema") != HEADER_SCHEMA:
        errors.append("header.green_water.schema must be %s" % HEADER_SCHEMA)
    rule = block.get("rule") or {}
    if rule.get("version") != RULE_VERSION:
        errors.append("header.green_water.rule.version must be %s" % RULE_VERSION)
    if rule.get("formula_limit") != FORMULA_LIMIT or rule.get("share_limit") != SHARE_LIMIT:
        errors.append("header.green_water rule limits must be exactly 0.05 and 0.05 (fixed before data, never "
                      "relaxed)")
    formula = block.get("formula") or {}
    probability = formula.get("probability")
    if formula.get("version") != FORMULA_VERSION:
        errors.append("header.green_water.formula.version must be %s" % FORMULA_VERSION)
    if formula.get("limit") != FORMULA_LIMIT:
        errors.append("header.green_water.formula.limit must be 0.05")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or \
            not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return errors + ["header.green_water.formula.probability must be a probability in [0, 1]"]
    detection = block.get("detection") or {}
    if detection.get("version") != DETECTION_VERSION:
        errors.append("header.green_water.detection.version must be %s" % DETECTION_VERSION)
    n_required = detection.get("n_consecutive_substeps")
    dt = detection.get("physics_dt_s")
    points = detection.get("points") or []
    labels = [p.get("label") for p in points if isinstance(p, dict)]
    if n_required != N_CONSECUTIVE_SUBSTEPS:
        errors.append("header.green_water.detection.n_consecutive_substeps must be %d" % N_CONSECUTIVE_SUBSTEPS)
        n_required = N_CONSECUTIVE_SUBSTEPS
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not dt > 0:
        errors.append("header.green_water.detection.physics_dt_s must be positive")
        dt = 0.0
    if not labels or len(set(labels)) != len(labels) or not all(isinstance(x, str) and x for x in labels):
        errors.append("header.green_water.detection.points must declare unique labelled points")
    green = []
    for index, record in enumerate(records):
        prefix = f"records/{index}: "
        if "green_water" not in record:
            errors.append(prefix + "journal record with waves lacks green_water")
            continue
        found = record_errors(record["green_water"], n_required, dt, labels, prefix)
        errors.extend(found)
        if not found:
            green.append(record["green_water"])
    if len(green) != len(records):
        return errors
    cell = block.get("cell")
    if not isinstance(cell, dict):
        return errors + ["header.green_water.cell (the cell summary) is missing"]
    expected = cell_summary(probability, green)
    for key in ("rule_version", "episodes", "episodes_with_green_water", "formula_probability",
                "excluded_green_water", "reasons"):
        if cell.get(key) != expected[key]:
            errors.append("header.green_water.cell.%s = %r but the records give %r"
                          % (key, cell.get(key), expected[key]))
    share = cell.get("share")
    if expected["share"] is None:
        if share is not None:
            errors.append("header.green_water.cell.share must be null without episodes")
    elif isinstance(share, bool) or not isinstance(share, (int, float)) or \
            not math.isclose(share, expected["share"], rel_tol=0.0, abs_tol=1e-15):
        errors.append("header.green_water.cell.share = %r but the records give %r" % (share, expected["share"]))
    # the verdict must be consistent with its two stamped numbers (whatever the records say)
    stamped_share = share if isinstance(share, (int, float)) and not isinstance(share, bool) else None
    stamped_p = cell.get("formula_probability")
    if isinstance(stamped_p, bool) or not isinstance(stamped_p, (int, float)):
        stamped_p = probability
    excluded, _ = verdict(float(stamped_p), stamped_share)
    if cell.get("excluded_green_water") is not excluded:
        errors.append("header.green_water.cell.excluded_green_water is inconsistent with its formula probability "
                      "and share (rule: formula > 0.05 or share > 0.05)")
    # G18: optional read-only diagnostics (never part of the verdict)
    errors.extend(diagnostics_errors_all(block, records, labels, dt, probability))
    # T25 (T1.35d): optional report-only point sets (never part of the verdict)
    errors.extend(report_errors_all(block, records))
    # T25 (T1.35c): the binding rule/2 exclusion verdict, recomputed from its stamped inputs
    errors.extend(exclusion_errors(block))
    return errors


# ------------------------------------------------------------ G18 diagnostics --
def _is_count_list(value, n):
    return isinstance(value, list) and len(value) == n and all(_is_int(v) and v >= 0 for v in value)


def diagnostics_errors(diag, green_record, labels, physics_dt_s, prefix=""):
    """Consistency of one records[].green_water_diagnostics against its rule/1 record (G18)."""
    if not isinstance(diag, dict):
        return [prefix + "green_water_diagnostics must be an object"]
    missing = [k for k in DIAGNOSTIC_FIELDS if k not in diag]
    if missing:
        return [prefix + "green_water_diagnostics lacks " + ", ".join(missing)]
    errors = []
    if diag["version"] != DIAGNOSTICS_VERSION:
        errors.append(prefix + "green_water_diagnostics.version must be %s" % DIAGNOSTICS_VERSION)
    if diag["settle_s"] != SETTLE_S:
        errors.append(prefix + "green_water_diagnostics.settle_s must be %r" % SETTLE_S)
    settled = diag["green_water_settled"]
    if not isinstance(settled, bool):
        return errors + [prefix + "green_water_settled must be a bool"]
    events = diag["events_settled"]
    if not _is_int(events) or events < 0:
        errors.append(prefix + "events_settled must be a nonnegative integer")
        events = 0
    n = len(labels)
    for key in ("freeboard_upcrossings", "mean_level_upcrossings"):
        if not _is_count_list(diag[key], n):
            errors.append(prefix + key + " must list one nonnegative integer per declared point")
    if errors:
        return errors
    substeps = green_record.get("substeps") if isinstance(green_record, dict) else None
    if _is_int(substeps):
        exposure = max(0.0, substeps * physics_dt_s - SETTLE_S)
        value = diag["exposure_settled_s"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isclose(value, exposure, rel_tol=0.0, abs_tol=1e-6):
            errors.append(prefix + "exposure_settled_s must be max(0, substeps x dt - settle_s)")
    first, point = diag["first_time_settled_s"], diag["first_point_settled"]
    if settled:
        if events < 1:
            errors.append(prefix + "green_water_settled true needs events_settled >= 1")
        if isinstance(first, bool) or not isinstance(first, (int, float)) or first < SETTLE_S - 1e-9:
            errors.append(prefix + "first_time_settled_s must be a time >= settle_s")
        if point not in labels:
            errors.append(prefix + "first_point_settled must name a declared deck-edge point")
        if isinstance(green_record, dict) and green_record.get("green_water") is not True:
            errors.append(prefix + "a settled green-water event needs green_water true in the rule/1 record")
    else:
        if events != 0 or first is not None or point is not None:
            errors.append(prefix + "green_water_settled false needs events_settled 0 and null first time / point")
    for j, (up_f, up_0) in enumerate(zip(diag["freeboard_upcrossings"], diag["mean_level_upcrossings"])):
        if up_f > up_0 + 1:
            errors.append(prefix + "freeboard up-crossings exceed mean-level up-crossings + 1 at point %d" % j)
            break
    return errors


def diagnostics_summary(formula_probability, green_records, diagnostics, labels):
    """header.green_water.diagnostics over the counted episodes (G18; non-binding, verdict unchanged)."""
    n = len(diagnostics)
    k = sum(1 for d in diagnostics if d["green_water_settled"] is True)
    share = (k / n) if n else None
    events = sum(int(d["events_settled"]) for d in diagnostics)
    exposure = sum(float(d["exposure_settled_s"]) for d in diagnostics)
    rate = (events / exposure) if exposure > 0 else None
    up_f = [sum(int(d["freeboard_upcrossings"][j]) for d in diagnostics) for j in range(len(labels))]
    up_0 = [sum(int(d["mean_level_upcrossings"][j]) for d in diagnostics) for j in range(len(labels))]
    per_cycle = [(f / z) if z > 0 else None for f, z in zip(up_f, up_0)]
    known = [(v, j) for j, v in enumerate(per_cycle) if v is not None]
    worst_value, worst_j = None, None
    if known:
        top = max(v for v, _ in known)
        worst_j = next(j for v, j in known if v >= top)
        worst_value = per_cycle[worst_j]
    p = float(formula_probability)
    share_any = (sum(1 for r in green_records if r["green_water"] is True) / n) if n else None
    rule1, _ = verdict(p, share_any)
    return {"version": DIAGNOSTICS_VERSION, "settle_s": SETTLE_S, "binding": False,
            "episodes": n, "episodes_with_green_water_settled": k, "share_settled": share,
            "events_settled": events, "exposure_settled_s": exposure,
            "events_per_minute_settled": (60.0 * rate) if rate is not None else None,
            "per_episode_probability_at_reference_s": ((1.0 - math.exp(-rate * REFERENCE_DURATION_S))
                                                       if rate is not None else None),
            "reference_duration_s": REFERENCE_DURATION_S,
            "freeboard_upcrossings": up_f, "mean_level_upcrossings": up_0,
            "simulated_per_cycle_probability": per_cycle,
            "simulated_per_cycle_worst": worst_value,
            "simulated_per_cycle_worst_point": labels[worst_j] if worst_j is not None else None,
            "candidate_verdicts": {
                "rule_1_formula_or_share": rule1,
                "formula_or_settled_share": bool(p > FORMULA_LIMIT or (share is not None and share > SHARE_LIMIT)),
                "per_cycle_formula_or_simulated": bool(p > FORMULA_LIMIT or (worst_value is not None
                                                                             and worst_value > FORMULA_LIMIT)),
                "note": "T25 (owner decision T1.35c): the binding verdict is header.green_water.exclusion "
                        "(usvbench.green_water.rule/2 = per_cycle_formula_or_simulated: the formula and the "
                        "simulated per-cycle exceedance after the settling window, the same unit); rule_1 "
                        "(usvbench.green_water.rule/1) is still reported in header.green_water.cell"}}


def diagnostics_errors_all(block, records, labels, physics_dt_s, probability):
    """All-or-none diagnostics on the records + the header summary recomputed (validator side, G18)."""
    carried = [i for i, r in enumerate(records) if "green_water_diagnostics" in r]
    summary = block.get("diagnostics")
    if not carried and summary is None:
        return []
    if len(carried) != len(records):
        return ["green_water_diagnostics must be on every record or on none (%d of %d)"
                % (len(carried), len(records))]
    errors = []
    for i, r in enumerate(records):
        errors.extend(diagnostics_errors(r["green_water_diagnostics"], r.get("green_water"), labels, physics_dt_s,
                                         f"records/{i}: "))
    if errors:
        return errors
    if not isinstance(summary, dict):
        return ["header.green_water.diagnostics is missing while the records carry diagnostics"]
    expected = diagnostics_summary(probability, [r["green_water"] for r in records],
                                   [r["green_water_diagnostics"] for r in records], labels)
    for key, value in expected.items():
        got = summary.get(key)
        if isinstance(value, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
            if not math.isclose(got, value, rel_tol=1e-12, abs_tol=1e-12):
                errors.append("header.green_water.diagnostics.%s = %r but the records give %r" % (key, got, value))
        elif got != value:
            errors.append("header.green_water.diagnostics.%s = %r but the records give %r" % (key, got, value))
    return errors


# ------------------------------------------------------------ T25 rule/2 (binding) --
def verdict_rule2(formula_probability, simulated_per_cycle_worst):
    """(excluded, reasons): formula > 0.05 or worst simulated per-cycle exceedance > 0.05 (None = no data)."""
    reasons = []
    if formula_probability > FORMULA_LIMIT:
        reasons.append("formula_probability_exceeds_0.05")
    if simulated_per_cycle_worst is not None and simulated_per_cycle_worst > PER_CYCLE_LIMIT:
        reasons.append("simulated_per_cycle_exceedance_exceeds_0.05")
    return bool(reasons), reasons


def _vessel_kind(vessel):
    if isinstance(vessel, str) and vessel.startswith("blueboat"):
        return "blueboat"
    if isinstance(vessel, str) and vessel.startswith("wamv_vrx_v2_"):
        return "wamv"
    return None


def probe_block(vessel):
    """header.green_water.exclusion.probe of a journal vessel (G28): the binding probe set and its labels."""
    kind = _vessel_kind(vessel)
    if kind is None:
        raise ValueError("no green-water exclusion probe declared for vessel %r" % (vessel,))
    entry = PROBE_SETS[kind]
    return {"version": PROBE_VERSION, "decision": PROBE_DECISION, "set": entry["set"],
            "geometry": entry["geometry"], "labels": list(entry["labels"]),
            "points_key": "header.green_water.detection.points",
            "report_only_sets": list(entry["report_only_sets"]),
            "applies_to": "formula points, rule/1 cell, G18 diagnostics and this rule/2 verdict"}


def exclusion_block(formula_probability, diagnostics, cell, vessel=None):
    """header.green_water.exclusion: the BINDING rule/2 verdict (T25, owner decision T1.35c).

    ``diagnostics`` = header.green_water.diagnostics (G18 summary over the counted episodes) or None when no
    counted episode carries diagnostics (then only the formula can exclude); ``cell`` = the rule/1 cell.
    ``vessel`` (G28): the header vessel; given, the block carries ``probe`` (probe_block); None reproduces the
    v2.5 (T25) block without a probe stamp."""
    has = isinstance(diagnostics, dict)
    worst = diagnostics.get("simulated_per_cycle_worst") if has else None
    point = diagnostics.get("simulated_per_cycle_worst_point") if has else None
    excluded, reasons = verdict_rule2(float(formula_probability), worst)
    return {"rule_version": RULE2_VERSION, "binding": True, "decision": RULE2_DECISION, "text": RULE2_TEXT,
            "formula_limit": FORMULA_LIMIT, "per_cycle_limit": PER_CYCLE_LIMIT, "settle_s": SETTLE_S,
            "comparison": "strictly greater than the limit excludes",
            "formula_probability": float(formula_probability),
            "simulated_per_cycle_worst": worst, "simulated_per_cycle_worst_point": point,
            "simulated_basis": ("header.green_water.diagnostics" if has else
                                "none (no counted episode carries diagnostics): only the formula can exclude"),
            "excluded_green_water": excluded, "reasons": reasons,
            "rule_1": {"rule_version": RULE_VERSION, "binding": False,
                       "excluded_green_water": cell["excluded_green_water"], "reasons": list(cell["reasons"])},
            **({} if vessel is None else {"probe": probe_block(vessel)})}


def exclusion_errors(block):
    """header.green_water.exclusion recomputed from the formula, the diagnostics summary and the rule/1 cell.

    Required whenever the header carries G18 diagnostics (every certificate written since T25); a G10-era
    certificate (no diagnostics, no exclusion block) stays valid under rule/1."""
    exclusion = block.get("exclusion")
    diagnostics = block.get("diagnostics")
    cell = block.get("cell")
    if exclusion is None:
        if diagnostics is not None:
            return ["header.green_water.exclusion (the binding rule/2 verdict, owner decision T1.35c) is missing"]
        return []
    if not isinstance(cell, dict) or not isinstance(exclusion, dict):
        return ["header.green_water.exclusion needs the rule/1 cell and must be an object"]
    probability = (block.get("formula") or {}).get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        return ["header.green_water.exclusion cannot be checked without the formula probability"]
    # G28: the probe stamp (absent = a v2.5 block) and the binding point labels
    vessel = block.get("vessel")
    stamped_probe = "probe" in exclusion
    labels = [p.get("label") for p in ((block.get("detection") or {}).get("points") or []) if isinstance(p, dict)]
    probe_errors = []
    if stamped_probe:
        try:
            expected_probe = probe_block(vessel)
        except ValueError as exc:
            return ["header.green_water.exclusion.probe: %s" % exc]
        if labels != expected_probe["labels"]:
            probe_errors.append("header.green_water.detection.points labels %r are not the declared exclusion probe "
                                "%s %r" % (labels, expected_probe["set"], expected_probe["labels"]))
    elif _vessel_kind(vessel) == "wamv" and labels == list(_WAMV_DECK_EDGE_LABELS):
        probe_errors.append("a WAM-V header whose binding points are the deck edge must stamp "
                            "header.green_water.exclusion.probe (G28)")
    try:
        expected = exclusion_block(probability, diagnostics, cell, vessel if stamped_probe else None)
    except (KeyError, TypeError) as exc:
        return probe_errors + ["header.green_water.exclusion cannot be recomputed: %s" % exc]
    errors = probe_errors
    for key in sorted(set(expected) | set(exclusion)):
        if exclusion.get(key, "<absent>") != expected.get(key, "<absent>"):
            errors.append("header.green_water.exclusion.%s = %r but rule/2 gives %r"
                          % (key, exclusion.get(key, "<absent>"), expected.get(key, "<absent>")))
    return errors


# ------------------------------------------------------------ T25 report-only point sets --
def report_summary(report_points, reports):
    """header.green_water.report over the counted episodes (T25, T1.35d; non-binding)."""
    sets = {}
    for name, block in report_points["sets"].items():
        labels = [p["label"] for p in block["points"]]
        recs = [r[name]["record"] for r in reports]
        diags = [r[name]["diagnostics"] for r in reports]
        n = len(recs)
        k = sum(1 for g in recs if g["green_water"] is True)
        k_settled = sum(1 for d in diags if d["green_water_settled"] is True)
        events = sum(int(d["events_settled"]) for d in diags)
        exposure = sum(float(d["exposure_settled_s"]) for d in diags)
        rate = (events / exposure) if exposure > 0 else None
        up_f = [sum(int(d["freeboard_upcrossings"][j]) for d in diags) for j in range(len(labels))]
        up_0 = [sum(int(d["mean_level_upcrossings"][j]) for d in diags) for j in range(len(labels))]
        per_cycle = [(f / z) if z > 0 else None for f, z in zip(up_f, up_0)]
        known = [(v, j) for j, v in enumerate(per_cycle) if v is not None]
        worst_j = None
        if known:
            top = max(v for v, _ in known)
            worst_j = next(j for v, j in known if v >= top)
        sets[name] = {"episodes": n, "episodes_with_green_water": k, "share": (k / n) if n else None,
                      "episodes_with_green_water_settled": k_settled, "share_settled": (k_settled / n) if n else None,
                      "events_settled": events, "exposure_settled_s": exposure,
                      "events_per_minute_settled": (60.0 * rate) if rate is not None else None,
                      "freeboard_upcrossings": up_f, "mean_level_upcrossings": up_0,
                      "simulated_per_cycle_probability": per_cycle,
                      "simulated_per_cycle_worst": per_cycle[worst_j] if worst_j is not None else None,
                      "simulated_per_cycle_worst_point": labels[worst_j] if worst_j is not None else None}
    return {"version": REPORT_VERSION, "binding": False, "sets": sets}


def report_errors_all(block, records):
    """All-or-none report-only sets on the records, their consistency and the header summary (T25, T1.35d)."""
    points = block.get("report_points")
    carried = [i for i, r in enumerate(records) if "green_water_report" in r]
    if points is None:
        if carried or block.get("report") is not None:
            return ["records[].green_water_report / header.green_water.report without header.green_water.report_points"]
        return []
    if not isinstance(points, dict) or points.get("version") != REPORT_VERSION or points.get("binding") is not False:
        return ["header.green_water.report_points must be the non-binding %s block" % REPORT_VERSION]
    sets = points.get("sets")
    if not isinstance(sets, dict) or not sets:
        return ["header.green_water.report_points.sets must name at least one set"]
    labels_of = {}
    for name, entry in sets.items():
        labels = [p.get("label") for p in (entry or {}).get("points") or [] if isinstance(p, dict)]
        if not labels or len(set(labels)) != len(labels) or not all(isinstance(x, str) and x for x in labels):
            return ["header.green_water.report_points.sets.%s must declare unique labelled points" % name]
        if entry.get("n_consecutive_substeps") != N_CONSECUTIVE_SUBSTEPS or entry.get("settle_s") != SETTLE_S:
            return ["header.green_water.report_points.sets.%s must use N = %d and settle_s = %r"
                    % (name, N_CONSECUTIVE_SUBSTEPS, SETTLE_S)]
        dt = entry.get("physics_dt_s")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not dt > 0:
            return ["header.green_water.report_points.sets.%s.physics_dt_s must be positive" % name]
        labels_of[name] = labels
    if len(carried) != len(records):
        return ["green_water_report must be on every record or on none (%d of %d)" % (len(carried), len(records))]
    errors = []
    for i, r in enumerate(records):
        report = r["green_water_report"]
        prefix = f"records/{i}: green_water_report "
        if not isinstance(report, dict) or set(report) != set(sets):
            errors.append(prefix + "must carry exactly the declared sets %s" % sorted(sets))
            continue
        for name, labels in labels_of.items():
            entry = report[name]
            if not isinstance(entry, dict) or set(entry) != {"record", "diagnostics"}:
                errors.append(prefix + "%s must be {record, diagnostics}" % name)
                continue
            dt = sets[name]["physics_dt_s"]
            errors.extend(record_errors(entry["record"], N_CONSECUTIVE_SUBSTEPS, dt, labels, prefix + name + ": "))
            errors.extend(diagnostics_errors(entry["diagnostics"], entry["record"], labels, dt, prefix + name + ": "))
            main = r.get("green_water")
            if isinstance(main, dict) and isinstance(entry["record"], dict) and \
                    entry["record"].get("substeps") != main.get("substeps"):
                errors.append(prefix + "%s observed %r substeps, the declared set %r" % (
                    name, entry["record"].get("substeps"), main.get("substeps")))
    if errors:
        return errors
    summary = block.get("report")
    if not isinstance(summary, dict):
        return ["header.green_water.report is missing while the records carry green_water_report"]
    expected = report_summary(points, [r["green_water_report"] for r in records])
    if summary.get("version") != expected["version"] or summary.get("binding") is not False or \
            set(summary.get("sets") or {}) != set(expected["sets"]):
        return ["header.green_water.report must be the non-binding %s summary of the declared sets" % REPORT_VERSION]
    for name, values in expected["sets"].items():
        got_set = summary["sets"][name]
        for key, value in values.items():
            got = got_set.get(key)
            if isinstance(value, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
                if not math.isclose(got, value, rel_tol=1e-12, abs_tol=1e-12):
                    errors.append("header.green_water.report.sets.%s.%s = %r but the records give %r"
                                  % (name, key, got, value))
            elif got != value:
                errors.append("header.green_water.report.sets.%s.%s = %r but the records give %r"
                              % (name, key, got, value))
    return errors


__all__ = ["RULE_VERSION", "HEADER_SCHEMA", "DETECTION_VERSION", "FORMULA_VERSION", "FORMULA_LIMIT",
           "SHARE_LIMIT", "N_CONSECUTIVE_SUBSTEPS", "RULE_TEXT", "RECORD_FIELDS", "rule_block", "verdict",
           "cell_summary", "record_errors", "header_errors", "DIAGNOSTICS_VERSION", "SETTLE_S",
           "REFERENCE_DURATION_S", "DIAGNOSTIC_FIELDS", "diagnostics_errors", "diagnostics_summary",
           "diagnostics_errors_all", "REPORT_VERSION", "REPORT_DECISION", "report_summary", "report_errors_all",
           "RULE2_VERSION", "RULE2_DECISION", "PER_CYCLE_LIMIT", "RULE2_TEXT", "verdict_rule2", "exclusion_block",
           "exclusion_errors", "PROBE_VERSION", "PROBE_DECISION", "PROBE_SETS", "probe_block"]
