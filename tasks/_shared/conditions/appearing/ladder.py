"""Appearing-suite dose ladder: the ONE switch (standard library only; loadable by path).

Trigger rule (unchanged, G3 rule 5): the cylinder appears at the first control step whose gap (D7 footprint
rectangle to cylinder surface) is <= the dose's trigger gap d_k.

OWNER DECISION (2026-10-03 17:3x, controller update to G20b): "D1: every controller can pass; D4: even the
fastest controller is at its limit." The default ladder "M" therefore gives the trigger gaps in METRES, one
table per vessel (initial values from the kinematics evidence, G20 work/evidence/kinematics_ladder.json:
plant = the D5 registry vessel through the D3 mixer, straight run at speed v, cylinder dead ahead, command
held for the latency, then the best single open-loop manoeuvre; "avoidable gap" = smallest gap that still
avoids contact):
  * D1 = 1.2 x the largest reaction-limited avoidable gap, 1.0 s re-plan latency at top speed:
    BlueBoat 3.973 m at 2.997 m/s, WAM-V 3.97 m at 2.26 m/s -> 1.2 x 3.97 = 4.8, set to 5.0 m for both;
  * D4 = the one-step-latency (1/60 s) avoidable gap at top speed: BlueBoat 1.025 m -> 1.0 m,
    WAM-V 1.76 m -> 1.8 m (the fastest controllers, MPPI and the G17 layer, re-plan every step);
  * D2, D3 geometric between D1 and D4: BlueBoat 5.0 x 0.2^(1/3) = 2.92 -> 2.9, x 0.2^(2/3) = 1.71 -> 1.7;
    WAM-V 5.0 x 0.36^(1/3) = 3.56 -> 3.6, x 0.36^(2/3) = 2.53 -> 2.5;
  * every rung stays above the owner's KINEMATIC FLOOR: the avoidable gap at u_ref = 1.0 m/s with the declared
    0.5 s reaction on the D3-mixer plant (kinematics_ladder.json rows latency 0.5, speed 1.0): BlueBoat 0.674 m,
    WAM-V 1.029 m (KINEMATIC_FLOOR_M: 0.67, 1.03). This is NOT a witness floor: the D7 placement witness model
    (unicycle, max_decel braking, 0.02 m clearance) certifies a dead-ahead cylinder only from 0.735 / 1.228 m
    (D7_WITNESS_FLOOR_M);
  * D5 (supplementary, outside the main table; refused outside diagnostic runs) = the witnessed calibration rung,
    the smallest gap the D7 placement witness certified on every probed key (BlueBoat 0.76 m, WAM-V 1.25 m,
    CALIBRATION_RUNG_M). Every pool is admitted jointly over D1..D5, so the same placements serve the ladder and the
    whole calibration sweep, whose floor IS D5 (G20c, REV-G20b M3; pairing across gaps). The diagnostic token
    appearing.gap_m accepts gaps from D5 (GAP_OVERRIDE_MIN_M) up to the task's start gap;
  * the equivalent k of a gap is k = gap / (t_avoid u_ref) (D7 scale), stamped per cell.
OWNER DECISION (2026-10-14, doses frozen; G47 / v2.6.4): the main table of ladder M is the result of the
pre-registered calibration-sweep selection on the c3 sweep (rule sha A488FA71; G43 c3sweep1, Linux-validated,
logs/sweep_select_c3sweep1_LINUX_20261013T160845Z.txt; target = this table, the sweep cells' 'ladder_in_force'):
  * D1 by the rule's 'order' clause (owner): BlueBoat 7.5 m (owner 2026-10-12: the sweep and start-gap cap, ~ the
    36-ray visibility guarantee 7.59 m), WAM-V 15 m (owner 2026-10-14: the sweep cap);
  * D4 = the rule's fallback (no sweep gap brought the fastest controller confidently to half its calm success):
    the witnessed calibration rung, BlueBoat 0.76 m, WAM-V 1.25 m (so D4 = D5 for both vessels);
  * D2, D3 geometric between D1 and D4 rounded to 0.1 m: BlueBoat 3.5, 1.6 m; WAM-V 6.6, 2.9 m.
The 2026-10-03 values above (5.0 / 2.9 / 1.7 / 1.0 and 5.0 / 3.6 / 2.5 / 1.8 m) are the initial ladder the sweep
replaced. Suite version g3_appearing_gap_v7m: every pool admitted under v6m (other gaps, other start gaps) is
refused at load; the pools are re-admitted and re-pinned with the new table.
The k ladders stay selectable for reference (not default): A = 4, 3, 2, 1.25 (the D7 TTC ratios, v2.3/v2.4),
B = 4, 2, 1, 0.5 (G20 recommendation), E = A + supplementary k = 0.5 (D5).

What a change of ACTIVE_LADDER changes (and nothing else): the trigger gaps, the witnessed doses, the start gap
of the placements (largest gap in use), the station-keeping approach bound (smallest main gap), the suite
version (so a pool admitted under another option is refused at load: "re-run the offline admission"), the
stamp / header text. Pools of every task must be re-admitted and re-pinned; DESIGN_FINAL.md section 3 lists the
tests that change.
"""

U_REF_MPS = 1.0                      # CONTEXT_R4 D1 reference speed of the D7 scale
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
# D7 vessel envelope t_avoid (dynamic/vessels.py envelope_from_spec_v2; kinematics_ladder.json "t_avoid_s"):
# identifies a vessel when a caller passes only t_avoid (pre-G20b call sites)
T_AVOID_S = {"blueboat_v2": 1.6450197479617348, "wamv_vrx_v2_opt1": 3.74273175}
# owner's kinematic floor (u_ref with the declared 0.5 s reaction on the D3-mixer plant; kinematics_ladder.json rows
# latency 0.5, speed 1.0: BlueBoat g_min 0.674 m, WAM-V 1.029 m; owner values 0.67 / 1.03 m). Informational only:
# no gap below the calibration rung is witnessed, so nothing in the code accepts a gap below CALIBRATION_RUNG_M.
KINEMATIC_FLOOR_M = {"blueboat_v2": 0.67, "wamv_vrx_v2_opt1": 1.03}
# calibration sweep (owner decision 2026-10-03, G20c revision after REV-G20b M3): floor = D5 (the witnessed rung),
# BlueBoat capped at 7.5 m (36 true-intersection rays at 10 deg return a disc of radius >= 0.8 m from any direction
# up to a trigger gap of 7.59 m, G20b FINAL_REPORT 4b); WAM-V ray guarantee 34.4 m
CALIBRATION_SWEEP_M = {"blueboat_v2": (0.76, 1.0, 1.4, 2.0, 2.9, 4.0, 5.0, 6.6, 7.5),
                       "wamv_vrx_v2_opt1": (1.25, 1.4, 1.8, 2.5, 3.6, 5.0, 7.5, 11.0, 15.0)}
# the calibration sweep runs on these tasks; their placements keep the start gap above the largest sweep gap
CALIBRATION_TASKS = ("scatter", "path_following")
# The D7 placement witness (feasibility.find_witness: unicycle at u_ref, nominal plan held for the declared 0.5 s
# reaction, braking at the envelope's max_decel, 0.02 m witness clearance) cannot certify a cylinder dead ahead
# closer than u t_r + u^2 / (2 a_dec) + clearance = BlueBoat 0.5 + 1 / (2 x 2.3254) + 0.02 = 0.735 m, WAM-V
# 0.5 + 1 / (2 x 0.7058) + 0.02 = 1.228 m (D7_WITNESS_FLOOR_M), above the owner's kinematic floors 0.67 / 1.03 m
# (D3-mixer plant). Measured (G20b evidence/witness_floor_{bb,wv}.json, scatter / path following / reach, 4 keys
# each): BlueBoat witnesses exist from 0.76 m (none at 0.70-0.74), WAM-V from 1.25 m (none at 1.05-1.20); with
# D5 at the owner floors or at G20b's smallest sweep gaps (0.70 / 1.05 m) no key of scatter, path following or
# docking admits (evidence/smoke_M_*.json, smoke_M2_*.json, 0/6 per task and vessel).
D7_WITNESS_FLOOR_M = {"blueboat_v2": 0.7350197479617346, "wamv_vrx_v2_opt1": 1.2284617500000001}
# the WITNESSED calibration rung (supplementary dose D5 of ladder M): the smallest gap measured witnessable on
# every probed key; also the floor of the calibration sweep and the lower bound of the diagnostic gap token.
CALIBRATION_RUNG_M = {"blueboat_v2": 0.76, "wamv_vrx_v2_opt1": 1.25}
GAP_OVERRIDE_MIN_M = dict(CALIBRATION_RUNG_M)      # lower bound of appearing.gap_m (G20c: was the kinematic floor)
# envelope speed bounds (dynamic/vessels.py envelope_from_spec_v2; pinned for the stamp reader's plausibility check
# of the trigger pose speed, REV-G20b T11): top speed and largest reverse speed, m/s
VESSEL_TOP_SPEED_MPS = {"blueboat_v2": 3.000326502719915, "wamv_vrx_v2_opt1": 2.2589523496083515}
VESSEL_MAX_REVERSE_SPEED_MPS = {"blueboat_v2": 0.9743162233660108, "wamv_vrx_v2_opt1": 0.6725570785847397}

LADDER_OPTIONS = {
    "M": {"kind": "metres",
          # owner decision 2026-10-14 (frozen; pre-registered sweep selection, G43 c3sweep1): D4 = the rung (fallback)
          "main_m": {"blueboat_v2": (7.5, 3.5, 1.6, 0.76), "wamv_vrx_v2_opt1": (15.0, 6.6, 2.9, 1.25)},
          "supplementary": "calibration_rung",
          # G20c: v5m (G20b's v4m pools and stamps are refused: station-keeping rule, sweep floor, reader pins)
          # G20d: v6m (owner placement decisions 2026-10-04: station keeping on the approach route, BlueBoat SK
          # without D1, docking mouth distance + 5B, fortress radius rule, 16 candidates, visible_at_trigger stamp)
          # G47 (v2.6.4): v7m (owner 2026-10-14 frozen main table; the v6m pools and stamps carry the old gaps)
          "suite_version": "g3_appearing_gap_v7m",
          "note": "owner decision 2026-10-03: per-vessel metres; D1 every controller can pass, D4 the fastest at "
                  "its limit; D5 = the witnessed calibration rung (supplementary, diagnostic runs only)"},
    "A": {"kind": "k", "main": (4.0, 3.0, 2.0, 1.25), "supplementary": (),
          # A keeps the v2.4 suite version, so the pinned v2.4 G3 pools (scatter, path following) stay valid
          "suite_version": "g3_appearing_gap_v2",
          "note": "reference: D7 TTC ratios (dynamic/model.py DOSE_TTC_RATIOS); v2.3/v2.4 ladder"},
    "B": {"kind": "k", "main": (4.0, 2.0, 1.0, 0.5), "supplementary": (),
          "suite_version": "g3_appearing_gap_v3b",
          "note": "reference: G20 recommendation, factor 2 per rung"},
    "E": {"kind": "k", "main": (4.0, 3.0, 2.0, 1.25), "supplementary": (0.5,),
          "suite_version": "g3_appearing_gap_v3e",
          "note": "reference: A in the main table + supplementary rung k = 0.5 (dose D5)"},
}
# THE SWITCH. Owner decision 2026-10-03: the per-vessel metre ladder M (A and B kept for reference).
ACTIVE_LADDER = "M"


def ladder_constants(option):
    """Derived constants of one ladder option (pure; also used by the tests for the inactive options)."""
    if option not in LADDER_OPTIONS:
        raise ValueError("unknown appearing ladder option %r (options: %s)" % (option, ", ".join(sorted(LADDER_OPTIONS))))
    spec = LADDER_OPTIONS[option]
    if spec["kind"] == "metres":
        tables = {}
        for vessel in VESSELS:
            main = tuple(float(v) for v in spec["main_m"][vessel])
            floor = float(D7_WITNESS_FLOOR_M[vessel])
            rung = float(CALIBRATION_RUNG_M[vessel])
            # G47 (v2.6.4): D4 may EQUAL the calibration rung (the pre-registered rule's D4 fallback, owner 2026-10-14)
            if len(main) != 4 or len(set(main)) != 4 or list(main) != sorted(main, reverse=True) \
                    or not main[-1] >= rung >= floor:
                raise ValueError("a metre ladder has four strictly decreasing gaps at or above the calibration rung, "
                                 "itself at or above the D7 witness-model floor (%s)" % vessel)
            tables[vessel] = (None,) + main + (rung,)
        n_main, n_supp = 4, 1
        ratios = None
    else:
        main = tuple(float(k) for k in spec["main"])
        supplementary = tuple(float(k) for k in spec["supplementary"])
        if len(main) != 4 or any(not k > 0.0 for k in main + supplementary) or list(main) != sorted(main, reverse=True):
            raise ValueError("a ladder has four decreasing positive main rungs")
        if main[0] != 4.0:
            raise ValueError("every k ladder keeps D1 = 4 (the start rule and the step-0 check use the largest gap)")
        if supplementary and not max(supplementary) < min(main):
            raise ValueError("a supplementary rung lies below the main table")
        ratios = (None,) + main + supplementary
        tables = None
        n_main, n_supp = len(main), len(supplementary)
    doses = tuple(range(1, n_main + n_supp + 1))
    return {"option": option, "kind": spec["kind"], "gap_ratios": ratios, "gaps_m": tables,
            "main_doses": doses[:n_main], "supplementary_doses": doses[n_main:], "witnessed_doses": doses,
            "max_dose": n_main + n_supp, "suite_version": spec["suite_version"], "note": spec["note"]}


ACTIVE = ladder_constants(ACTIVE_LADDER)
KIND = ACTIVE["kind"]                              # "metres" (M) or "k" (A, B, E)
GAP_RATIOS = ACTIVE["gap_ratios"]                  # k options: (None, k_1, ..., k_n); M: None (per-vessel metres)
GAPS_M = ACTIVE["gaps_m"]                          # M: {vessel: (None, d_1, ..., d_5)}; k options: None
MAIN_DOSES = ACTIVE["main_doses"]                  # (1, 2, 3, 4)
SUPPLEMENTARY_DOSES = ACTIVE["supplementary_doses"]  # M: (5,) calibration rung; E: (5,) k = 0.5; A, B: ()
WITNESSED_DOSES = ACTIVE["witnessed_doses"]        # every dose > 0; admission is joint over them
MAX_DOSE = ACTIVE["max_dose"]
DOSE_COUNT = MAX_DOSE + 1                          # rows of per_dose_counts (D0..D_max)
SUITE_VERSION = ACTIVE["suite_version"]


# G20d (owner decision 2B, 2026-10-04): doses that are NOT APPLICABLE BY DESIGN for a (task, vessel). Station keeping
# puts the cylinder on the approach route at 0.45 x the start-to-hold-centre distance (decision 1B/2B); the BlueBoat
# approach is only 10 m, so a cylinder 4.5 m from the hold centre is 3.9-4.2 m from the start footprint: the D1 gap
# (5.0 m) would be reached at step 0. D1 is therefore never offered there (pool header "not_applicable_by_design",
# per-dose counts [0, 0, 0]); it is NOT a not-admitted key. A run at a not-applicable dose is refused (pool and env).
NOT_APPLICABLE_DOSES = {
    ("station_keeping", "blueboat_v2"): {
        1: ("not applicable by design (owner decision 2B, 2026-10-04): the BlueBoat station-keeping approach is "
            "10 m; the cylinder on the route 4.5 m from the hold centre lies within the D1 gap (7.5 m since "
            "2026-10-14, 5.0 m before) of the start footprint, so D1 would appear at step 0"),
    },
    ("docking", "blueboat_v2"): {
        1: ("not applicable by design (owner decision 2026-10-14, as BlueBoat station keeping): with D1 = 7.5 m "
            "the BlueBoat docking route window after the start rule is empty (route_window_empty_after_start_rule), "
            "so no cylinder can be placed at D1; docking BlueBoat starts at D2 = 3.5 m"),
    },
}
NOT_APPLICABLE = "not_applicable_by_design"


def not_applicable_doses(task, vessel):
    """Sorted doses that are not applicable by design for (task, vessel) (G20d; empty for most pairs)."""
    return tuple(sorted(NOT_APPLICABLE_DOSES.get((task, vessel), {})))


def not_applicable_reason(task, vessel, dose):
    return NOT_APPLICABLE_DOSES.get((task, vessel), {}).get(dose)


def offered_doses(task, vessel):
    """Witnessed doses OFFERED for (task, vessel): every witnessed dose except the not-applicable ones."""
    skip = set(not_applicable_doses(task, vessel))
    return tuple(d for d in WITNESSED_DOSES if d not in skip)


def offered_main_doses(task, vessel):
    skip = set(not_applicable_doses(task, vessel))
    return tuple(d for d in MAIN_DOSES if d not in skip)


def vessel_of(t_avoid_s, vessel_id=None):
    """The ladder vessel: vessel_id when given, else the vessel whose D7 t_avoid equals t_avoid_s."""
    if vessel_id is not None:
        if vessel_id not in VESSELS:
            raise ValueError("no appearing ladder for vessel %r" % (vessel_id,))
        return vessel_id
    for vessel, value in T_AVOID_S.items():
        if abs(float(t_avoid_s) - value) <= 1e-9 * value:
            return vessel
    raise ValueError("cannot identify the vessel of t_avoid %r for the per-vessel appearing ladder" % (t_avoid_s,))


def gap_m(dose, t_avoid_s, vessel_id=None):
    """Trigger gap d_k (m) of a dose of the ACTIVE ladder; None at D0."""
    if dose == 0:
        return None
    if KIND == "metres":
        return GAPS_M[vessel_of(t_avoid_s, vessel_id)][dose]
    return GAP_RATIOS[dose] * float(t_avoid_s) * U_REF_MPS


def start_gap_m(task, vessel):
    """Smallest allowed gap between the start footprint and a placed cylinder (no trigger at step 0): the largest
    OFFERED witnessed gap (D1; G20d: BlueBoat station keeping, where D1 is not applicable by design, D2 = 2.9 m);
    under the metre ladder the calibration tasks use the largest calibration-sweep gap when it exceeds D1 (BlueBoat
    7.5 m, WAM-V 15.0 m). Also the upper bound of appearing.gap_m."""
    t_avoid = T_AVOID_S[vessel]
    gap = max(gap_m(d, t_avoid, vessel) for d in offered_doses(task, vessel))
    if KIND == "metres" and task in CALIBRATION_TASKS:
        gap = max(gap, max(CALIBRATION_SWEEP_M[vessel]))
    return gap


def k_equivalent(gap_m_value, t_avoid_s):
    """Equivalent D7 ratio of a gap: k = gap / (t_avoid u_ref)."""
    return None if gap_m_value is None else float(gap_m_value) / (float(t_avoid_s) * U_REF_MPS)


def dose_text():
    """Human-readable ladder line for stamps and headers."""
    if KIND == "metres":
        rows = "; ".join("%s %s m" % (v, " / ".join("%g" % g for g in GAPS_M[v][1:5])) for v in VESSELS)
        return ("trigger gap d_k in metres per vessel (D1..D4): %s; supplementary D5 (diagnostic runs only) = "
                "calibration rung %s m; "
                "equivalent k = d_k / (t_avoid u_ref), u_ref = 1.0 m/s; D0 = never appears; ladder option M "
                "(owner decision 2026-10-03); not applicable by design (G20d): %s"
                % (rows, " / ".join("%g" % CALIBRATION_RUNG_M[v] for v in VESSELS),
                   "; ".join("%s %s D%s" % (t, v, "/".join(str(d) for d in sorted(m)))
                             for (t, v), m in sorted(NOT_APPLICABLE_DOSES.items())) or "none"))
    main = ", ".join("%g" % k for k in GAP_RATIOS[1:1 + len(MAIN_DOSES)])
    text = "trigger gap d_k = k * t_avoid * u_ref, k = %s (D1..D%d" % (main, len(MAIN_DOSES))
    if SUPPLEMENTARY_DOSES:
        text += "; supplementary D%d k = %s outside the main table" % (
            SUPPLEMENTARY_DOSES[0], ", ".join("%g" % GAP_RATIOS[d] for d in SUPPLEMENTARY_DOSES))
    return text + "), u_ref = 1.0 m/s, t_avoid from the D7 vessel envelope; D0 = never appears; ladder option " \
        + ACTIVE_LADDER


__all__ = ["U_REF_MPS", "VESSELS", "T_AVOID_S", "KINEMATIC_FLOOR_M", "CALIBRATION_SWEEP_M", "CALIBRATION_TASKS",
           "CALIBRATION_RUNG_M", "D7_WITNESS_FLOOR_M", "GAP_OVERRIDE_MIN_M", "VESSEL_TOP_SPEED_MPS",
           "VESSEL_MAX_REVERSE_SPEED_MPS", "start_gap_m",
           "LADDER_OPTIONS", "ACTIVE_LADDER", "ladder_constants", "ACTIVE", "KIND", "GAP_RATIOS", "GAPS_M",
           "MAIN_DOSES", "SUPPLEMENTARY_DOSES", "WITNESSED_DOSES", "MAX_DOSE", "DOSE_COUNT", "SUITE_VERSION",
           "vessel_of", "gap_m", "k_equivalent", "dose_text", "NOT_APPLICABLE_DOSES", "NOT_APPLICABLE",
           "not_applicable_doses", "not_applicable_reason", "offered_doses", "offered_main_doses"]
