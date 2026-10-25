"""Appearing-suite configuration and dose rule (G3; controller rules 1-5; G20b ladder switch).

Dose: D0 = off. D1..D_max: the cylinder appears at the first control step whose
gap between the own-ship collision footprint (the D7 footprint rectangle) and the
cylinder surface is <= d_k, the trigger gap of the ACTIVE ladder option
(ladder.py ACTIVE_LADDER; owner decision 2026-10-03: option M, per-vessel metres,
BlueBoat 7.5 / 3.5 / 1.6 / 0.76 m, WAM-V 15 / 6.6 / 2.9 / 1.25 m since the owner's frozen doses of
2026-10-14 (v2.6.4; initially 5.0 / 2.9 / 1.7 / 1.0 and 5.0 / 3.6 / 2.5 / 1.8 m), supplementary D5
= the witnessed calibration rung 0.76 / 1.25 m, diagnostic runs only; the k
ladders A = 4, 3, 2, 1.25 and
B = 4, 2, 1, 0.5 (d_k = k * t_avoid * u_ref) stay selectable for reference). The
equivalent D7 ratio k = d_k / (t_avoid u_ref) is stamped per cell (u_ref = 1.0
m/s, CONTEXT_R4 D1; t_avoid from the D7 vessel envelope, BlueBoat 1.645 s,
WAM-V 3.743 s). Diagnostic only: the gap of a run may be overridden
(appearing.gap_m=<x>, calibration sweep), down to the vessel's calibration rung
D5 (GAP_OVERRIDE_MIN_M) and up to the task's start gap; refused in official runs,
like the supplementary dose appearing=D5 (SUPPLEMENTARY_DOSES).

Slot: the appearing cylinder owns the slot D7 already plans for it
(dynamic/capacity.py CapacityPlan.appearing_slots): index base_max + D7 reserve.
"""
from dataclasses import asdict, dataclass

from ..dynamic.capacity import CapacityPlan
from ..dynamic.model import check_number
from . import ladder

# G20b: the suite version names the active ladder option (A keeps "g3_appearing_gap_v2", the v2.4 pins)
APPEARING_SUITE_VERSION = ladder.SUITE_VERSION
SUITE_KEY = "appearing"
LADDER_OPTION = ladder.ACTIVE_LADDER         # owner decision 2026-10-03: "M" (per-vessel metres)
LADDER_KIND = ladder.KIND                    # "metres" or "k"
GAP_RATIOS = ladder.GAP_RATIOS               # k options: (None, k_1, ..., k_max); M: None (see trigger_gap_m)
MAIN_DOSES = ladder.MAIN_DOSES               # (1, 2, 3, 4): the main table
SUPPLEMENTARY_DOSES = ladder.SUPPLEMENTARY_DOSES   # M: (5,) = the calibration rung; E: (5,) k = 0.5 (diagnostic)
WITNESSED_DOSES = ladder.WITNESSED_DOSES     # every dose > 0; the admission is joint over them
MAX_DOSE = ladder.MAX_DOSE                   # 4 (A, B) or 5 (M, E)
GAP_OVERRIDE_MIN_M = ladder.GAP_OVERRIDE_MIN_M   # smallest appearing.gap_m per vessel = the calibration rung D5
GAP_OVERRIDE_FIELD = "gap_m"                 # diagnostic token appearing.gap_m=<x> (calibration sweep)
DOSE_COUNT = ladder.DOSE_COUNT               # D0..D_max
REFERENCE_SPEED_MPS = 1.0                    # u_ref, CONTEXT_R4 D1 (not configurable per run)
# G20 / G20b: the offline placement + admission (PLACEMENT_TASKS) and the ENV side (bridge, slot, contact,
# rays; WIRED_TASKS) cover all twelve journal tasks. G19b's training sampler reads WIRED_TASKS at run time
# (name and type, a tuple of journal task ids, are kept).
PLACEMENT_TASKS = ("scatter", "path_following", "reach", "ring", "fortress", "forced_crossing", "iceberg",
                   "moving_basin", "path_hazard", "harbour_mission", "station_keeping", "docking")
WIRED_TASKS = PLACEMENT_TASKS
# journal task id -> D7 TaskContext task name (adapters.TASKS); identity for every other task
CONTEXT_TASK = {"harbour_mission": "harbour"}
HAZARD_GROUP_PREFIX = "g3_appearing_"        # EpisodeKey.hazard_group of an appearing key
STREAM_GROUP_PREFIX = "g3_appearing."        # scenario_rng stream group of the placement draws
RADIUS_LENGTH_FACTOR = 0.8                   # radius rule fallback: 0.8 m x vessel length scale
APPEARING_RESERVE = 1                        # one appearing slot per env row
PER_DOSE_COUNT_FIELDS = ("offered", "admitted", "not_admitted")
# G20d admission policy (owner decision 5B, 2026-10-04): on these tasks the MAIN tiers D1..D4 no longer require the
# supplementary (diagnostic) D5 witness; D5 is admitted per key on its own (a key admitted for D1..D4 whose D5 has no
# witness is not admitted at D5 only, with that reason). Every other task stays jointly admitted over D1..D5.
SUPPLEMENTARY_OPTIONAL_TASKS = ("docking",)
# G20d (owner decision 5B, REV-G20b evidence_rev/monotone_recovery_counts.txt): "monotone stitching". A main dose d
# whose finite search failed is witnessed by the witness of a dose d' > d (smaller gap: the cylinder appears later,
# the vessel keeps to the nominal plan longer) spliced onto d's own nominal prefix (the prefix of d up to its release
# time, then the d' witness samples after that time), accepted only through d's own replay (feasibility.
# replay_witness): the same feasibility standard. Tried for every failing main dose, d' in decreasing order.
MONOTONE_STITCH_TASKS = ("docking",)
NOT_APPLICABLE = ladder.NOT_APPLICABLE
# G20d (owner decision 8B, 2026-10-04): when the frozen beam (witness_beam_width = 48) finds no witness, the same
# request is searched once more with witness_fallback_beam_width (4 x 48 = 192; tools_g20d/beam_probe.py: 96 and
# 192 recovered docking BlueBoat keys G20c had rejected); the replay standard is unchanged. 0 disables it.
WITNESS_FALLBACK_BEAM_WIDTH = 192
NOT_ADMITTED = "not_admitted"
ADMITTED = "admitted"
BEYOND_OFFERED_GRID = "beyond_offered_grid"


@dataclass(frozen=True)
class AppearingSuiteConfig:
    """Immutable appearing-suite configuration bound to one vessel.

    Only the dose is a run parameter (``appearing=D<k>``); the planning and
    witness fields shape the offline admission and must equal the pool's
    (checked at env construction, like the D7 suite).
    """

    enabled: bool = False
    dose: int = 0
    vessel_id: str = ""
    plan_dt_s: float = 0.25
    witness_dt_s: float = 0.25
    witness_beam_width: int = 48
    witness_clearance_m: float = 0.02
    witness_substeps: int = 4
    witness_fallback_beam_width: int = WITNESS_FALLBACK_BEAM_WIDTH   # G20d: second search after a failed frozen beam
    version: str = APPEARING_SUITE_VERSION
    gap_override_m: object = None    # G20b diagnostic token appearing.gap_m (never part of the identity)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if type(self.dose) is not int or not 0 <= self.dose <= MAX_DOSE:
            raise ValueError("appearing dose must be an integer D0..D%d (ladder option %s)" % (MAX_DOSE, LADDER_OPTION))
        if not self.enabled and self.dose != 0:
            raise ValueError("a disabled appearing suite must have dose 0")
        for name in ("plan_dt_s", "witness_dt_s"):
            check_number(name, getattr(self, name), positive=True)
        check_number("witness_clearance_m", self.witness_clearance_m, minimum=0.0)
        if type(self.witness_beam_width) is not int or not 1 <= self.witness_beam_width <= 4096:
            raise ValueError("witness_beam_width must be an integer in [1, 4096]")
        if type(self.witness_substeps) is not int or not 1 <= self.witness_substeps <= 256:
            raise ValueError("witness_substeps must be an integer in [1, 256]")
        if type(self.witness_fallback_beam_width) is not int or not (
                self.witness_fallback_beam_width == 0
                or self.witness_beam_width < self.witness_fallback_beam_width <= 4096):
            raise ValueError("witness_fallback_beam_width must be 0 or an integer in (witness_beam_width, 4096]")
        if self.plan_dt_s > self.witness_dt_s:
            raise ValueError("plan_dt_s must not exceed witness_dt_s (the witness replays the plan prefix)")
        if self.version != APPEARING_SUITE_VERSION:
            raise ValueError("unsupported appearing suite version")
        if self.enabled and not self.vessel_id:
            raise ValueError("an enabled appearing suite must name its vessel")
        if self.gap_override_m is not None:
            check_number("appearing.gap_m", self.gap_override_m, positive=True)
            if self.dose == 0:
                raise ValueError("appearing.gap_m needs the appearing suite at a dose >= 1 (D0 never appears)")
            low = GAP_OVERRIDE_MIN_M.get(self.vessel_id)
            if low is not None and self.gap_override_m < low - 1e-12:
                raise ValueError("appearing.gap_m %r is below the calibration rung %r m of %s (no smaller gap is "
                                 "witnessed)" % (self.gap_override_m, low, self.vessel_id))

    def at_dose(self, dose):
        values = asdict(self)
        values["dose"] = int(dose)
        values["enabled"] = values["enabled"] or int(dose) > 0
        return AppearingSuiteConfig(**values)

    def identity(self):
        """Fields that shape the admission (everything except the dose switch and the diagnostic gap override)."""
        values = asdict(self)
        for name in ("enabled", "dose", "gap_override_m"):
            values.pop(name)
        return values


def check_dose(dose):
    """An appearing dose of the active ladder (D0..D_max)."""
    if type(dose) is not int or not 0 <= dose <= MAX_DOSE:
        raise ValueError("appearing dose must be an integer D0..D%d (ladder option %s)" % (MAX_DOSE, LADDER_OPTION))
    return dose


def gap_ratio(dose):
    """k of a dose under a k ladder (A, B, E); None under the metre ladder M (use k_equivalent)."""
    check_dose(dose)
    return None if GAP_RATIOS is None else GAP_RATIOS[dose]


def trigger_gap_m(dose, t_avoid_s, vessel_id=None):
    """d_k in metres (None at D0, the cylinder never appears). Metre ladder: the vessel's table (vessel_id, or
    the vessel identified by its D7 t_avoid); k ladder: d_k = k * t_avoid * u_ref."""
    check_dose(dose)
    check_number("t_avoid_s", t_avoid_s, positive=True)
    if dose == 0:
        return None
    if LADDER_KIND == "metres":
        return ladder.gap_m(dose, t_avoid_s, vessel_id)
    return GAP_RATIOS[dose] * float(t_avoid_s) * REFERENCE_SPEED_MPS


def k_equivalent(gap_m_value, t_avoid_s):
    """Equivalent D7 ratio of a trigger gap, k = gap / (t_avoid u_ref) (stamped per cell)."""
    return ladder.k_equivalent(gap_m_value, t_avoid_s)


def trigger_gaps_m(t_avoid_s, vessel_id=None):
    return [trigger_gap_m(dose, t_avoid_s, vessel_id) for dose in range(DOSE_COUNT)]


def largest_trigger_gap_m(t_avoid_s, vessel_id=None):
    """max_k d_k over the witnessed doses (= D1 under every option)."""
    return max(trigger_gap_m(dose, t_avoid_s, vessel_id) for dose in WITNESSED_DOSES)


def smallest_trigger_gap_m(t_avoid_s, vessel_id=None):
    """min_k d_k over the witnessed doses (M: the calibration rung D5)."""
    return min(trigger_gap_m(dose, t_avoid_s, vessel_id) for dose in WITNESSED_DOSES)


def smallest_main_trigger_gap_m(t_avoid_s, vessel_id=None):
    """min_k d_k over the MAIN table D1..D4 (bounds the station-keeping approach gap: every main rung triggers
    on the final approach)."""
    return min(trigger_gap_m(dose, t_avoid_s, vessel_id) for dose in MAIN_DOSES)


def not_applicable_doses(task, vessel_id):
    """G20d: doses not applicable by design for (task, vessel) (ladder.NOT_APPLICABLE_DOSES)."""
    return ladder.not_applicable_doses(task, vessel_id)


def offered_doses(task, vessel_id):
    """G20d: witnessed doses offered for (task, vessel) (every witnessed dose minus the not-applicable ones)."""
    return ladder.offered_doses(task, vessel_id)


def required_doses(task, vessel_id):
    """G20d: doses a key must witness to be ADMITTED (D0 included in the admission): the offered doses, minus the
    supplementary rung on SUPPLEMENTARY_OPTIONAL_TASKS (decision 5B)."""
    offered = offered_doses(task, vessel_id)
    if task in SUPPLEMENTARY_OPTIONAL_TASKS:
        return tuple(d for d in offered if d not in SUPPLEMENTARY_DOSES)
    return offered


def optional_doses(task, vessel_id):
    """G20d: offered doses admitted per key on their own (decision 5B: docking D5)."""
    required = set(required_doses(task, vessel_id))
    return tuple(d for d in offered_doses(task, vessel_id) if d not in required)


def check_dose_applicable(task, vessel_id, dose):
    """G20d: refuse a run at a dose that is not applicable by design for (task, vessel)."""
    if dose in not_applicable_doses(task, vessel_id):
        raise ValueError("appearing=D%d is %s for %s on %s: %s" % (
            dose, NOT_APPLICABLE.replace("_", " "), task, vessel_id,
            ladder.not_applicable_reason(task, vessel_id, dose)))
    return dose


def offered_main_trigger_gaps_m(task, vessel_id):
    """G20d: {dose: d_k} of the offered MAIN doses."""
    t_avoid = ladder.T_AVOID_S[vessel_id]
    return {d: trigger_gap_m(d, t_avoid, vessel_id) for d in ladder.offered_main_doses(task, vessel_id)}


def start_gap_m(task, t_avoid_s, vessel_id=None):
    """Smallest allowed gap between the start footprint and a placed cylinder (no trigger at step 0).

    The largest OFFERED witnessed gap (D1; docking: D1 under M, controller decision; G20d: BlueBoat
    station keeping, D1 not applicable by design: D2 (3.5 m since v2.6.4)); under the metre ladder the
    calibration tasks (ladder.CALIBRATION_TASKS: scatter, path following) use the largest calibration-sweep gap
    when it exceeds D1, so the same placements serve every sweep gap (BlueBoat 7.5 m, WAM-V 15.0 m;
    ladder.start_gap_m, the stamp reader's copy of this rule).
    """
    return ladder.start_gap_m(task, ladder.vessel_of(t_avoid_s, vessel_id))


def check_gap_override(task, t_avoid_s, vessel_id, gap_m_value):
    """Diagnostic gap override (appearing.gap_m): between the vessel's calibration rung and the task's start gap."""
    check_number("appearing.gap_m", gap_m_value, positive=True)
    floor = GAP_OVERRIDE_MIN_M[ladder.vessel_of(t_avoid_s, vessel_id)]
    top = start_gap_m(task, t_avoid_s, vessel_id)
    if not floor - 1e-12 <= float(gap_m_value) <= top + 1e-12:
        raise ValueError("appearing.gap_m %r outside [calibration rung %r, start gap %r] m for %s on %s"
                         % (gap_m_value, floor, top, vessel_id, task))
    return float(gap_m_value)


def ladder_header(envelope):
    """Header fields of the active ladder for one vessel envelope (gap in metres and equivalent k per dose)."""
    gaps = trigger_gaps_m(envelope.t_avoid_s, envelope.vessel_id)
    return {"ladder_option": LADDER_OPTION, "ladder_kind": LADDER_KIND,
            "gap_ratios": None if GAP_RATIOS is None else list(GAP_RATIOS), "trigger_gaps_m": gaps,
            "k_equivalent": [k_equivalent(g, envelope.t_avoid_s) for g in gaps],
            "main_doses": list(MAIN_DOSES), "supplementary_doses": list(SUPPLEMENTARY_DOSES),
            "gap_override_min_m": GAP_OVERRIDE_MIN_M.get(envelope.vessel_id)}


def ladder_summary():
    """Vessel-free ladder description (the resolved condition stamp)."""
    return {"ladder_option": LADDER_OPTION, "ladder_kind": LADDER_KIND,
            "gap_ratios": None if GAP_RATIOS is None else list(GAP_RATIOS),
            "gaps_m": None if ladder.GAPS_M is None else {v: list(g) for v, g in ladder.GAPS_M.items()}}


def hazard_group(task):
    """EpisodeKey hazard group of an appearing key (never collides with 'd7_<task>')."""
    return HAZARD_GROUP_PREFIX + str(task)


def stream_group(task):
    """scenario_rng stream group of the placement draws (its own stream, G3 rule 3)."""
    return STREAM_GROUP_PREFIX + str(task)


def capacity_plan(base_max, dynamic_reserve):
    """D7 CapacityPlan with the appearing reserve (dynamic/capacity.py)."""
    need = int(base_max) + int(dynamic_reserve) + APPEARING_RESERVE
    return CapacityPlan(int(base_max), int(dynamic_reserve), APPEARING_RESERVE, need)


def appearing_slot(base_max, dynamic_reserve):
    """Storage index of the appearing cylinder: CapacityPlan.appearing_slots[0]."""
    slots = capacity_plan(base_max, dynamic_reserve).appearing_slots
    if len(slots) != 1:
        raise AssertionError("one appearing slot per row")
    return slots[0]


def check_wired(task):
    """Run-time check (env bridge, condition admission): the task's env carries the appearing slot."""
    if task not in WIRED_TASKS:
        raise ValueError("the appearing suite is wired on %s only; task %r is not"
                         % (", ".join(WIRED_TASKS), task))
    return task


def check_placement_task(task):
    """Offline check (placement, admission, pools): the task has a declared placement rule."""
    if task not in PLACEMENT_TASKS:
        raise ValueError("the appearing placement is declared for %s only; task %r is not"
                         % (", ".join(PLACEMENT_TASKS), task))
    return task


def context_task(task):
    """D7 TaskContext task name of a journal task (harbour_mission -> harbour)."""
    return CONTEXT_TASK.get(task, task)


def journal_task(context_task_name):
    """Journal task id of a D7 TaskContext task name (inverse of context_task)."""
    for journal, ctx in CONTEXT_TASK.items():
        if ctx == context_task_name:
            return journal
    return context_task_name


__all__ = ["APPEARING_SUITE_VERSION", "SUITE_KEY", "GAP_RATIOS", "REFERENCE_SPEED_MPS", "WIRED_TASKS",
           "LADDER_OPTION", "MAIN_DOSES", "SUPPLEMENTARY_DOSES", "WITNESSED_DOSES", "MAX_DOSE", "DOSE_COUNT",
           "check_dose", "largest_trigger_gap_m", "smallest_trigger_gap_m", "smallest_main_trigger_gap_m",
           "start_gap_m", "check_gap_override", "k_equivalent", "ladder_header", "ladder_summary", "LADDER_KIND",
           "GAP_OVERRIDE_MIN_M", "GAP_OVERRIDE_FIELD",
           "PLACEMENT_TASKS", "CONTEXT_TASK", "check_placement_task", "context_task", "journal_task",
           "HAZARD_GROUP_PREFIX", "STREAM_GROUP_PREFIX", "RADIUS_LENGTH_FACTOR", "APPEARING_RESERVE",
           "PER_DOSE_COUNT_FIELDS", "NOT_ADMITTED", "ADMITTED", "BEYOND_OFFERED_GRID",
           "SUPPLEMENTARY_OPTIONAL_TASKS", "MONOTONE_STITCH_TASKS", "NOT_APPLICABLE", "WITNESS_FALLBACK_BEAM_WIDTH",
           "not_applicable_doses", "offered_doses", "required_doses", "optional_doses", "check_dose_applicable",
           "offered_main_trigger_gaps_m",
           "AppearingSuiteConfig", "gap_ratio", "trigger_gap_m", "trigger_gaps_m", "hazard_group",
           "stream_group", "capacity_plan", "appearing_slot", "check_wired"]
