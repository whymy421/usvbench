"""Offline placement of the appearing cylinder and its per-dose witness requests (G3 rules 3, 4, 6).

Everything here is a pure function of the frozen task geometry of ONE scenario
key (the D7 TaskContext the family builds from its real journal layout) and of
the key's own scenario stream ``g3_appearing.<task>``; no controller state,
action or outcome enters. The offline admission tool calls ``admit_key``; the
runtime env only reads the admitted result from a pinned pool.

G20b note on d_1 below: it stands for the START GAP config.start_gap_m(task, vessel): the largest
witnessed gap D1 (under ladder A / B: 4 t_avoid u_ref, the numbers quoted for phase 1; under the default
metre ladder M: 5.0 m), and for the calibration tasks scatter / path following under M the largest
calibration-sweep gap (BlueBoat 7.5 m, WAM-V 15.0 m; G20c), so no dose and no sweep gap triggers at step 0.

Rule 3 (position, fixed before the episode, identical at every dose): the
nominal plan is D7's (adapters._nominal_plan: bounded-control unicycle through
the task's route circles at u_ref = 1.0 m/s). The site reuses the D7 per-task
site rule (adapters.PROFILES), with the owner's placement decisions of
2026-10-01 (G3 phase 1b, suite version g3_appearing_gap_v2):
  scatter         route_fraction band [0.4, 0.6] with RE-DRAW (owner option C):
                  candidates f_1, f_2, ... are drawn uniformly in the band from
                  the key's own stream, in that fixed order, up to
                  SCATTER_MAX_SITE_CANDIDATES = 8; the first candidate whose
                  cylinder passes every dose-independent clearance rule
                  (placement_reasons + start gap > d_1) is the site; if none
                  passes, the key is not admitted with the reason
                  "no_clear_site_in_route_band" plus every candidate's reasons.
                  The stretch is MEASURED per key (each candidate is tested on
                  the key's own route and field); the per-tier free share
                  along the route is reported by the admission report.
  path_following  leg_midpoint of the fixed route segment ``site_index`` (D7
                  adapters._conflict_time rule), moved forward along the same
                  segment when needed so that the travelled route distance from
                  the start is at least d_1 + footprint reach + cylinder radius
                  + PF_START_MARGIN_LENGTHS (1.0) x vessel length scale
                  (phase 1, ladder A: BlueBoat 6.580 + 0.794 + 0.8 + 1.0 =
                  9.174 m, WAM-V 14.971 + 3.092 + 3.582 + 4.478 = 26.123 m;
                  ladder M, d_1 = the largest sweep gap: BlueBoat 7.5 + 0.794
                  + 0.8 + 1.0 = 10.094 m, WAM-V 15.0 + 3.092 + 3.582 + 4.478
                  = 26.152 m); if the segment
                  ends before that distance the key is not admitted
                  ("site_cannot_clear_trigger_distance_on_leg").
G20 rollout, final rules of G20b (rules version PLACEMENT_RULES_VERSION,
recorded in every new placement document; the two phase-1 rules above and their
documents are unchanged byte for byte):
  route_window_redraw  reach, ring, path_hazard, harbour_mission (band
                  [0.40, 0.60]), fortress (annulus [0.30, 0.70]; ring and
                  fortress run on the gate-aligned appearing route of
                  journal_hazard_nav.appearing_gate_context), forced_crossing (post-gate exit
                  [0.65, 0.85]), iceberg (post-passage exit [0.75, 0.90]),
                  moving_basin (buffer between the rows and the orbiter
                  [0.40, 0.55]), docking (approach lane: the whole route
                  BEFORE the berth mouth). The windows outside the D29 band
                  (forced_crossing, iceberg, moving_basin, fortress) are
                  per-task extensions of D29 accepted by the controller
                  (G20b), each with its geometric reason in ROUTE_WINDOWS.
                  The window in travelled route distance is
                  [max(lo x L, s_min), min(hi x L, s_end)] with the
                  path-following start rule s_min = d_1 + footprint reach +
                  radius + start margin x length scale (margin
                  PF_START_MARGIN_LENGTHS = 1.0; docking 0.0,
                  START_MARGIN_LENGTHS_BY_TASK) and s_end = the nominal
                  arrival distance at the berth mouth (docking) or L; every
                  candidate must also keep its start gap above the LARGEST
                  trigger gap of the active ladder (no trigger at step 0);
                  docking candidates must clear the berth rectangle (walls
                  included) by the footprint half-width ("overlaps_berth",
                  never inside the berth geometry); candidates are drawn uniformly in it from
                  the key's stream, up to ROUTE_WINDOW_MAX_SITE_CANDIDATES = 8,
                  the first that passes every dose-independent rule is the
                  site; none -> "no_clear_site_in_route_window"; an empty
                  window -> "route_window_empty_after_start_rule".
  station_lane_flank station_keeping (G20c rule; replaces G20's front
                  sector and G20b's side sector, which put the cylinder inside
                  the hold circle - REV-G20b B1): the cylinder stands BESIDE
                  THE NOMINAL FINAL-APPROACH LANE, before the station. Lane
                  point P = the nominal pose a travelled distance s_back
                  before the nominal stop; the cylinder centre lies on the
                  normal of the plan heading at P (side from the key's
                  stream), at the lateral distance where the footprint at P
                  clears it by g* = hw + SK_LANE_GAP_INTERPOLATION (0.1) x
                  (D4 - hw), close to hw so that an approach deviating AWAY
                  from the cylinder by up to (D4 - g*) still triggers D4 (a
                  deviation towards it triggers earlier). Per side, s_lo = the smallest s_back (scanned in
                  steps of SK_LANE_SCAN_STEP_LENGTHS x length scale from the
                  stop) whose candidate keeps the STATION CLEARANCE below;
                  s_back is drawn uniformly in [s_lo, s_lo +
                  SK_LANE_WINDOW_LENGTHS x length scale]. A candidate is the
                  site iff, besides the generic rules (start gap > D1, no
                  trigger at step 0): (1) its surface keeps >= max(hold radius
                  + hw, SK_G17_EVADE_CLEARANCE_M[vessel]) from the station
                  centre, so the hold circle is clear and the station itself
                  is a valid evade target of the G17 avoidance layer
                  (scripts/journal_avoidance.py); (2) its surface keeps >=
                  footprint reach + hw from the centre (the centre is clear at
                  EVERY heading); (3) the nominal approach passes it at g_min
                  with hw <= g_min < D4 (every main rung triggers on the
                  nominal final approach, the nominal path stays clear); (4)
                  the nominal speed at the D4 trigger is >=
                  SK_MIN_TRIGGER_SPEED_FRACTION (0.25) x u_ref (candidate()
                  re-checks every main dose). Up to 8 candidates (reason
                  "no_clear_site_on_approach_lane"). The position depends on
                  the key, the ladder and the pinned G17 clearance constants
                  only (never on a controller's state; identical at every
                  dose).
  moving_basin additionally: the cylinder clears the swept envelope of every
                  native mover (rows: their full translation segment; the
                  orbiter: its orbit annulus) by the footprint half-width, a
                  dose-independent rule ("overlaps_native_sweep").
G20d (owner placement decisions of 2026-10-04, rules version PLACEMENT_RULES_VERSION = g20d_placement_v4):
  station_route_block station_keeping (decisions 1B / 2B; replaces G20c's station_lane_flank): the cylinder
                  centre stands ON the standard approach route (it blocks it). D_start = straight-line distance
                  from the start COM S to the hold-circle centre C (journal layout: BlueBoat 10.0 m, WAM-V
                  44.779 m); rho = SK_ROUTE_FRACTION (0.45) x D_start (4.5 m / 20.15 m). Controller decision J9
                  (shared with the D7 SK meeting point of G23e): the SK reference route turns in place to face
                  C, then runs straight, so the site is P = S + (1 - 0.45) (C - S), on the straight S-C line,
                  exactly rho from C (heading = bearing S -> C). The curved D7 nominal plan (which starts with
                  a turn at the spawn heading) passes within 0.21 m (BlueBoat) / 1.21 m (WAM-V) of P; its
                  offset is recorded (route_plan_offset_m) and the plan stays blocked (nominal_min_gap < 0).
                  No random draw: the position depends on the key's layout only. Admission rules (besides the
                  generic ones and
                  start gap > the largest OFFERED gap): surface >= station_clearance_m = max(hold radius +
                  hw, footprint reach + hw) from C (REV-G20c M6: no controller constant in the admission; whether
                  the station is a valid G17 evade target is recorded, report only); every offered main dose
                  triggers on
                  the nominal approach at >= SK_MIN_TRIGGER_SPEED_FRACTION x u_ref (candidate()). BlueBoat:
                  D1 is not applicable by design (ladder.NOT_APPLICABLE_DOSES); reason on failure
                  "no_clear_site_on_approach_route".
  docking mouth   (decision "new rule docking"): the cylinder centre keeps a berth-axis distance from the
                  berth-mouth line of at least DOCKING_MOUTH_MIN = approach_runup_m (= max(2 L, turning
                  radius, braking distance), journal_geometry_v2.VesselGeometry, the E5 run-up the layout
                  itself reserves before the mouth) + cylinder radius + hw: room to pass the cylinder and
                  then re-align with the berth axis before the mouth (BlueBoat 2.402 + 0.8 + 0.465 =
                  3.667 m, WAM-V 10.754 + 3.582 + 1.267 = 15.604 m). The route window ends DOCKING_MOUTH_MIN
                  of travelled distance before the mouth; "too_close_to_berth_mouth" checks the berth-axis
                  distance. The cylinder stays centred on the route (it blocks the approach).
  fortress radius (decision "new rule fortress"): per candidate site c with nominal heading h, the radius is
                  the largest r <= the rule-4 radius (0.8 s) with (a) a passage >= g on at least one side
                  (left / right of h): min over the static discs on that side of (|o - c| - r_o) - r >= g,
                  and (b) every compulsory gate opening (the segment between its two jamb surfaces) at a
                  distance >= r + g from c (never plugs a gate); g = the B17 gate gap of the journal tier
                  (journal_geometry_v2.physical_tier(2): beam + 2.0 s; BlueBoat 2.93 m, WAM-V 11.490 m).
                  A site whose r falls below the ray floor r_min = sin(pi/n) (start gap + reach) /
                  (1 - sin(pi/n)) (n = 36 rays: the smallest disc the fan is guaranteed to hit from any
                  bearing at the D1 trigger; BlueBoat 0.553 m, WAM-V 0.773 m) is rejected
                  ("fortress_radius_below_ray_floor"). Dynamic intruders are unchanged.
  candidates      every task that samples candidate positions draws up to 16 candidates (was 8: scatter,
                  every route_window_redraw task), same stream, same order, first feasible wins.
  visibility      every placement document carries the key's static occluder discs and the ray fan
                  (occlusion.py); every candidate (dose) carries nominal_visible_at_trigger / nominal_visible_
                  rays at its nominal trigger pose; occluded cylinders are kept (decision 7A).
  admission       per-task dose policy (config.required_doses / optional_doses; decision 5B: docking main
                  tiers do not need the D5 witness; monotone stitching on docking); a failed frozen-beam
                  search is retried once with config.WITNESS_FALLBACK_BEAM_WIDTH (decision 8B).
The radius and every candidate are drawn BEFORE the episode from the key's
stream; nothing depends on the dose. The cylinder centre is the nominal pose at
the site time (on the nominal path). D7's straight-window fit
(adapters._fit_conflict) is NOT applied.

Rule 4 (radius): the task's own static-obstacle radius rule when it has one
(scatter: uniform in journal_cfg_v2 radius_range_m = [0.8 s, 2.0 s], drawn ONCE
from the key's stream before the site candidates and kept for all of them),
else 0.8 m x the vessel length scale s (BlueBoat s = 1: 0.8 m; WAM-V
s = 4.47792: 3.582 m).

Rule 6 (admission, joint over every witnessed dose of the active ladder,
config.WITNESSED_DOSES): placement checks with clearance >= the
D7 footprint half-width hw (BlueBoat 0.465 m, WAM-V 1.267 m) to every static
obstacle surface, to the start footprint, to the goal region and to every gate
(mandatory route circle); the start gap must exceed every d_k; the nominal plan
must reach the trigger gap; and a feasibility witness (D7 feasibility.find_witness
+ replay_witness) with the cylinder physically present and the witness held to
the nominal plan until the nominal trigger time + the declared reaction time.
The witness treats the cylinder as present from t = 0, which is conservative:
before the trigger the nominal prefix keeps its gap > d_k > 0.
"""
from dataclasses import dataclass, replace as _replace
import math

from ..dynamic import adapters
from ..dynamic.feasibility import WitnessResult, _prefix as _witness_prefix, find_witness, replay_witness
from ..dynamic.geometry import point_clearance, to_body
from ..dynamic.model import Encounter, EpisodeKey, Pose, check_number
from ..dynamic.trajectory import PlacementError as D7PlacementError
from . import config, occlusion
from .stamp import digest

PLACEMENT_SCHEMA = "usvbench.appearing_placement/2"
CANDIDATE_SCHEMA = "usvbench.appearing_candidate/1"
SITE_RULES = {"scatter": "route_fraction_redraw", "path_following": "leg_midpoint_start_clear",
              # G20c: station keeping beside the final-approach lane (station_lane_flank, REV-G20b B1)
              # G20 rollout (final rules: G20b)
              "reach": "route_window_redraw", "ring": "route_window_redraw", "fortress": "route_window_redraw",
              "forced_crossing": "route_window_redraw", "iceberg": "route_window_redraw",
              "moving_basin": "route_window_redraw", "path_hazard": "route_window_redraw",
              "harbour_mission": "route_window_redraw", "docking": "route_window_redraw",
              "station_keeping": "station_route_block"}    # G20d: on the approach route (decisions 1B / 2B)
ROUTE_FRACTION_RANGE = (0.4, 0.6)
PLACEMENT_RULES_VERSION = "g20d_placement_v4"   # G20d: owner placement decisions 2026-10-04
# G20 / G20b: per-task route window (fractions of the nominal path length), its end anchor and the reason.
# D29 (owner, 2026-10-01) fixed the scatter band [0.40, 0.60]. The windows below that leave that band are
# per-task EXTENSIONS of D29 accepted by the controller (G20b), each for a geometric reason measured in G20
# (probe_geometry_n32.json, free share of the route fraction, 32 keys per task x vessel):
#   forced_crossing [0.65, 0.85]: the bulkhead gate lies at 0.46-0.56 of the route; inside [0.40, 0.60] the
#       cylinder would sit on the gate approach or in the gate region (free share 0.34-0.78 in the band);
#   iceberg [0.75, 0.90]: the nominal route hugs the blocker at 0.35-0.70 (free share 0.00-0.06 at 0.4-0.6),
#       so the only open water before the goal is the post-passage exit;
#   moving_basin [0.40, 0.55]: the buffer between the second mover row and the orbit annulus; beyond 0.55
#       the window enters the orbiter's swept annulus (overlaps_native_sweep on every candidate);
#   fortress [0.30, 0.70]: the annulus between the staggered gates is about 6 length scales wide, so the
#       D29 band alone left 3-6 keys of 32 without a clear site in 8 draws (gate-aligned route, G20 1.2).
ROUTE_WINDOWS = {
    "reach": ((0.40, 0.60), None, "open water: D29 band"),
    "ring": ((0.40, 0.60), None, "D29 band: after the ring gate on the leg to the goal (gate-aligned route)"),
    "fortress": ((0.30, 0.70), None, "D29 extension: the annulus between the staggered gates (gate-aligned "
                                     "nominal route)"),
    "path_hazard": ((0.40, 0.60), None, "D29 band: the middle legs between the blockers"),
    "harbour_mission": ((0.40, 0.60), None, "D29 band: inside the transit field (0.28-0.72 of the route)"),
    "forced_crossing": ((0.65, 0.85), None, "D29 extension: post-gate exit; the bulkhead gate lies at 0.46-0.56"),
    "iceberg": ((0.75, 0.90), None, "D29 extension: post-passage exit; the nominal route hugs the blocker at "
                                    "0.35-0.70"),
    "moving_basin": ((0.40, 0.55), None, "D29 extension: buffer between the second row and the orbit annulus"),
    "docking": ((0.0, 1.0), "berth_mouth", "approach lane: before the berth mouth, never in the berth"),
}
ROUTE_WINDOW_MAX_SITE_CANDIDATES = 16  # same named maximum as scatter (owner option C; G20d decision 8B: was 8)
# G20b (controller decision): docking drops the start-rule margin (under ladder A the D1 gap was 55 % of the
# BlueBoat approach; under M D1 = 5.0 m);
# the start-gap check against the LARGEST trigger gap and the berth exclusion stay
START_MARGIN_LENGTHS_BY_TASK = {"docking": 0.0}
# G20c station-keeping rule (station_lane_flank, REV-G20b B1)
SK_MIN_TRIGGER_SPEED_FRACTION = 0.25  # every main dose's nominal trigger at >= 0.25 u_ref (the vessel still moves)
SK_LANE_GAP_INTERPOLATION = 0.1       # lane gap g* = hw + 0.1 (D4 - hw): BlueBoat 0.519 m, WAM-V 1.320 m
SK_LANE_WINDOW_LENGTHS = 1.0          # along-lane window length, x vessel length scale
SK_LANE_SCAN_STEP_LENGTHS = 0.05      # scan step of the window start s_lo, x vessel length scale
# G17 avoidance layer evade clearance (scripts/classical_avoidance_v2.json vessels.<vessel>.evade_clearance_m,
# pinned here; tests/test_g20_rollout.py checks the json). G20c used it as an ADMISSION rule (the station is the
# layer's evade target, journal_avoidance._evade_target); G20d (REV-G20c M6): placement must not depend on a
# controller parameter, so it is REPORT ONLY (site field g17_evade_target_clear)
SK_G17_EVADE_CLEARANCE_M = {"blueboat_v2": 2.573050313783286, "wamv_vrx_v2_opt1": 6.317444760191088}
SK_BISECTIONS = 80
SCATTER_MAX_SITE_CANDIDATES = 16      # owner option C (2026-10-01); G20d (decision 8B, 2026-10-04): was 8
# G20d station keeping on the approach route (decisions 1B / 2B)
SK_ROUTE_FRACTION = 0.45              # site at straight-line distance 0.45 x D_start from the hold centre
# G20d docking: berth-axis distance of the cylinder centre from the mouth line >= approach run-up + radius + hw
DOCKING_MOUTH_RULE = ("berth-axis distance from the cylinder centre to the berth-mouth line >= approach_runup_m "
                      "(max(2 L, turning radius, braking distance)) + cylinder radius + footprint half-width")
# G20d fortress radius rule (passage >= the B17 gate gap on one side, never plugging a gate, ray floor)
FORTRESS_RADIUS_RULE = ("largest r <= the rule-4 radius with a passage >= g (B17 gate gap: beam + tier-2 margin) "
                        "on at least one side of the nominal heading and every gate opening >= r + g away; "
                        "rejected below the ray floor sin(pi/n)(start gap + reach)/(1 - sin(pi/n))")
PF_START_MARGIN_LENGTHS = 1.0         # owner decision 2 (2026-10-01): margin beyond d_1 + reach + radius, x length scale
DOSE_INDEPENDENT = 0           # dose tag of a placement-level (dose-independent) rejection reason
TRIGGER_SUBSAMPLES = 16        # plan samples per nominal-plan interval before bisection
TRIGGER_BISECTIONS = 60


class PlacementError(ValueError):
    """A counted admission rejection with a declared reason string (and optional further reasons)."""

    def __init__(self, reason, more=()):
        super().__init__(reason)
        self.reason = reason
        self.reasons = (reason,) + tuple(r for r in more if r != reason)


@dataclass(frozen=True)
class AppearingCylinder:
    """The static cylinder in the witness (duck-types a D7 intruder; never moves)."""

    x: float
    y: float
    radius_m: float
    reveal_s: float = 0.0

    def __post_init__(self):
        check_number("x", self.x)
        check_number("y", self.y)
        check_number("radius_m", self.radius_m, positive=True)
        check_number("reveal_s", self.reveal_s, minimum=0.0)

    def position(self, t):
        check_number("episode time", t, minimum=0.0)
        return self.x, self.y

    def velocity(self, t):
        return 0.0, 0.0

    @property
    def speed_bound_mps(self):
        return 0.0


def footprint_half_width(footprint):
    """Half the D7 footprint beam (rule 6 clearance)."""
    return 0.5 * footprint.beam_m


def radius_rule(task, length_scale, radius_range_m=None):
    """{"rule", "low_m", "high_m", "source"} of rule 4."""
    check_number("length_scale", length_scale, positive=True)
    if radius_range_m is not None:
        low, high = (float(v) for v in radius_range_m)
        if not 0.0 < low <= high:
            raise ValueError("radius range must satisfy 0 < low <= high")
        return {"rule": "task_static_radius_range", "low_m": low, "high_m": high,
                "source": "journal_cfg_v2 radius_range_m of %s (uniform, own stream)" % task}
    value = config.RADIUS_LENGTH_FACTOR * float(length_scale)
    return {"rule": "length_scale_factor", "low_m": value, "high_m": value,
            "source": "0.8 m x vessel length scale %.12g (G3 rule 4)" % float(length_scale)}


def gap_at(pose, x, y, radius_m, footprint):
    """Signed gap between the D7 footprint at ``pose`` and the cylinder surface."""
    return point_clearance(x, y, pose, footprint) - radius_m


def bearing_at(pose, x, y):
    """Relative bearing of the cylinder centre from the bow (rad, +port)."""
    bx, by = to_body(x, y, pose)
    return math.atan2(by, bx)


def nominal_plan(context, cfg):
    """D7's nominal plan of the context (the controller-independent reference)."""
    return adapters._nominal_plan(adapters.PROFILES[context.task], context, cfg)


def _stream(key, task):
    from ... import scenario_rng
    return scenario_rng.numpy_generator(scenario_rng.SCENARIO_PROTOCOL_VERSION, int(key.eval_seed),
                                        int(key.env), int(key.episode), config.stream_group(task))


def place(context, key, cfg, *, length_scale, radius_range_m=None, plan=None, regions=None):
    """(placement document, plan) for one key; raises PlacementError (dose-independent reason).

    The placement document is JSON-native; its digest is the dose-independent
    position digest of the pairing primitive. ``regions`` (G20b, docking only):
    {"berth": berth rectangle} supplied by the family; it is copied into the
    site block, where placement_reasons reads it.
    """
    task = config.journal_task(context.task)
    config.check_placement_task(task)
    if not isinstance(key, EpisodeKey) or key.hazard_group != config.hazard_group(task):
        raise ValueError("appearing keys use hazard_group %r" % config.hazard_group(task))
    plan = nominal_plan(context, cfg) if plan is None else plan
    if not plan.reached_end:
        raise PlacementError("nominal_plan_does_not_reach_goal")
    rng = _stream(key, task)
    rule = SITE_RULES[task]
    if regions is not None and rule != "route_window_redraw":
        raise ValueError("placement regions are declared for route-window tasks only")
    envelope = context.envelope
    rrule = radius_rule(task, length_scale, radius_range_m)
    if rrule["rule"] == "task_static_radius_range":
        radius = float(rng.uniform(rrule["low_m"], rrule["high_m"]))      # drawn once, before any site candidate
    else:
        radius = rrule["low_m"]
    # G20b: the start gap (no trigger at step 0): D1 of the ladder; under the metre ladder the calibration
    # tasks keep the largest sweep gap (config.start_gap_m). Under a k ladder it is d_1 = 4 t_avoid u_ref as in v2.4.
    d1 = config.start_gap_m(task, envelope.t_avoid_s, envelope.vessel_id)
    vis = visibility_block(context, length_scale)

    def document_at(t_site, site, pose=None, radius_m=None, radius_rule_doc=None):
        pose = plan.pose_at(t_site) if pose is None else pose
        return {
            "schema": PLACEMENT_SCHEMA, "suite_version": config.APPEARING_SUITE_VERSION, "task": task,
            "vessel_id": envelope.vessel_id,
            "key": [int(key.eval_seed), int(key.env), int(key.episode), str(key.hazard_group)],
            "stream_group": config.stream_group(task), "geometry_id": str(context.geometry_id),
            "cylinder": {"x_m": float(pose.x), "y_m": float(pose.y),
                         "radius_m": radius if radius_m is None else float(radius_m)},
            "site": site, "radius_rule": rrule if radius_rule_doc is None else radius_rule_doc,
            "nominal_heading_at_site_rad": float(pose.yaw),
            "plan": {"dt_s": float(cfg.plan_dt_s), "end_s": float(plan.end_s),
                     "path_length_m": float(plan.path_length_m), "reached_end": bool(plan.reached_end)},
            "visibility": vis,                                   # G20d: occluders + ray fan (occlusion.py)
        }

    if rule == "route_fraction_redraw":
        tried = []
        for index in range(SCATTER_MAX_SITE_CANDIDATES):
            fraction = float(rng.uniform(*ROUTE_FRACTION_RANGE))
            t_site = plan.time_at_fraction(fraction)
            site = {"rule": rule, "fraction": fraction, "fraction_range": list(ROUTE_FRACTION_RANGE),
                    "candidate_index": index, "max_candidates": SCATTER_MAX_SITE_CANDIDATES,
                    "rejected_candidates": [list(item) for item in tried], "t_site_s": t_site}
            document = document_at(t_site, site)
            reasons = site_reasons(context, document, d1)
            if not reasons:
                return document, plan
            tried.append((fraction, reasons))
        seen = []
        for _, reasons in tried:
            seen += [r for r in reasons if r not in seen]
        raise PlacementError("no_clear_site_in_route_band", seen)
    if rule == "route_window_redraw":
        return _place_route_window(context, task, plan, rng, radius, d1, length_scale, document_at, regions, rrule)
    if rule == "station_route_block":
        return _place_station_route(context, plan, radius, d1, document_at)
    if rule == "station_lane_flank":            # G20c rule, kept for reference (no task uses it since G20d)
        return _place_station_lane(context, plan, rng, radius, d1, length_scale, document_at)
    index = int(context.site_index)
    try:
        end = adapters._arrival_time(plan, index)
        start = 0.0 if index == 0 else adapters._arrival_time(plan, index - 1)
    except D7PlacementError as error:
        raise PlacementError(error.reason) from None
    t_mid = 0.5 * (start + end)
    minimum = d1 + envelope.footprint.reach_m + radius + PF_START_MARGIN_LENGTHS * float(length_scale)
    distance = plan.distance_at(t_mid)
    t_site, moved = t_mid, False
    if distance < minimum:
        t_site, moved = plan.time_at_fraction(minimum / plan.path_length_m), True
        if t_site >= end:
            raise PlacementError("site_cannot_clear_trigger_distance_on_leg")
    site = {"rule": rule, "leg_index": index, "leg_start_s": start, "leg_end_s": end, "leg_midpoint_s": t_mid,
            "midpoint_route_distance_m": distance, "min_route_distance_m": minimum,
            "start_rule": {"d1_m": d1, "footprint_reach_m": envelope.footprint.reach_m, "radius_m": radius,
                           "margin_lengths": PF_START_MARGIN_LENGTHS, "length_scale": float(length_scale)},
            "moved_forward": moved, "route_distance_m": plan.distance_at(t_site), "t_site_s": t_site}
    return document_at(t_site, site), plan


def _versioned(site):
    site = dict(site)
    site["rules_version"] = PLACEMENT_RULES_VERSION
    return site


def _redraw(rule_reason, draw, site_of, document_at, context, d1):
    """G20 redraw loop: up to ROUTE_WINDOW_MAX_SITE_CANDIDATES candidates in a fixed stream order."""
    tried = []
    for index in range(ROUTE_WINDOW_MAX_SITE_CANDIDATES):
        value = draw()
        t_site, pose, site, *rest = site_of(value)        # G20d: optional (radius, radius rule) per candidate
        site = _versioned(dict(site, candidate_index=index, max_candidates=ROUTE_WINDOW_MAX_SITE_CANDIDATES,
                               rejected_candidates=[list(item) for item in tried], t_site_s=t_site))
        document = document_at(t_site, site, pose, *rest)
        reasons = site_reasons(context, document, d1)
        if not reasons:
            return document
        tried.append((value, reasons))
    seen = []
    for _, reasons in tried:
        seen += [r for r in reasons if r not in seen]
    raise PlacementError(rule_reason, seen)


def _anchor_distance(context, plan, anchor):
    """Travelled nominal distance at the declared end anchor (docking: arrival at the berth-mouth gate)."""
    if anchor is None:
        return plan.path_length_m
    if anchor == "berth_mouth":
        flags = context.route_mandatory or (False,) * len(context.route)
        index = next((i for i, keep in enumerate(flags) if keep), None)
        if index is None:
            raise PlacementError("route_window_anchor_missing")
        try:
            return plan.distance_at(adapters._arrival_time(plan, index))
        except D7PlacementError as error:
            raise PlacementError(error.reason) from None
    raise ValueError("unknown route window anchor %r" % (anchor,))


def _place_route_window(context, task, plan, rng, radius, d1, length_scale, document_at, regions=None, rrule=None):
    (lo_f, hi_f), anchor, note = ROUTE_WINDOWS[task]
    length = plan.path_length_m
    reach = context.envelope.footprint.reach_m
    hw = footprint_half_width(context.envelope.footprint)
    margin = START_MARGIN_LENGTHS_BY_TASK.get(task, PF_START_MARGIN_LENGTHS)
    s_min = d1 + reach + radius + margin * float(length_scale)
    s_end = _anchor_distance(context, plan, anchor)
    mouth = None
    if task == "docking":
        # G20d: the window ends DOCKING_MOUTH_MIN of travelled distance before the berth mouth (the berth-axis
        # distance itself is checked by placement_reasons: too_close_to_berth_mouth)
        if not regions or "mouth" not in regions:
            raise ValueError("docking placement needs the family's berth-mouth block (regions['mouth'])")
        mouth = _mouth_rule(regions["mouth"], radius, hw)
        s_end = s_end - mouth["min_axis_distance_m"]
    lo, hi = max(lo_f * length, s_min), min(hi_f * length, s_end)
    if not lo < hi:
        raise PlacementError("route_window_empty_after_start_rule")
    base = {"rule": "route_window_redraw", "fraction_window": [lo_f, hi_f], "end_anchor": anchor,
            "window_note": note, "window_m": [lo, hi], "path_length_m": length, "anchor_distance_m": s_end,
            "start_rule": {"d1_m": d1, "footprint_reach_m": reach, "radius_m": radius,
                           "margin_lengths": margin, "length_scale": float(length_scale),
                           "min_route_distance_m": s_min, "start_gap_m": d1}}
    fortress = None
    if task == "docking":
        if not regions or "berth" not in regions:
            raise ValueError("docking placement needs the family's berth rectangle (regions['berth'])")
        base["berth_exclusion"] = _berth_region(regions["berth"])
        base["berth_mouth_rule"] = dict(mouth, anchor_route_distance_m=s_end + mouth["min_axis_distance_m"],
                                        window_end_m=s_end)
    elif task == "fortress":
        if not regions or "fortress" not in regions:
            raise ValueError("fortress placement needs the family's gate openings (regions['fortress'])")
        fortress = _fortress_inputs(regions["fortress"], context, d1, radius)
    elif regions:
        raise ValueError("placement regions are declared for docking and fortress only")

    def site_of(distance):
        t_site = plan.time_at_fraction(min(1.0, distance / length))
        site = dict(base, route_distance_m=distance, fraction=distance / length)
        if fortress is None:
            return t_site, None, site
        pose = plan.pose_at(t_site)
        value, detail = fortress_radius(fortress, context, pose)
        site["fortress_radius_rule"] = detail
        doc = dict(rrule, rule="fortress_passage_radius", base_rule=rrule["rule"], base_radius_m=radius,
                   low_m=value, high_m=value, radius_m=value, source=FORTRESS_RADIUS_RULE)
        return t_site, pose, site, value, doc

    document = _redraw("no_clear_site_in_route_window", lambda: float(rng.uniform(lo, hi)), site_of,
                       document_at, context, d1)
    return document, plan


def _berth_region(berth):
    """JSON-native berth rectangle (env-local centre, heading, berth-frame x range and half width)."""
    out = {"center_xy": [float(berth["center_xy"][0]), float(berth["center_xy"][1])],
           "heading_rad": float(berth["heading_rad"]), "x_min_m": float(berth["x_min_m"]),
           "x_max_m": float(berth["x_max_m"]), "half_width_m": float(berth["half_width_m"]),
           "source": str(berth.get("source", ""))}
    if not (out["x_min_m"] < out["x_max_m"] and out["half_width_m"] > 0.0):
        raise ValueError("berth rectangle must have x_min < x_max and a positive half width")
    return out


def _mouth_rule(mouth, radius, hw):
    """G20d docking: JSON-native berth-mouth block and the minimum berth-axis distance (run-up + r + hw)."""
    runup = float(mouth["approach_runup_m"])
    if not runup > 0.0:
        raise ValueError("approach_runup_m must be positive")
    return {"opening_x_m": float(mouth["opening_x_m"]), "approach_runup_m": runup, "radius_m": float(radius),
            "footprint_half_width_m": float(hw), "min_axis_distance_m": runup + float(radius) + float(hw),
            "rule": DOCKING_MOUTH_RULE, "source": str(mouth.get("source", ""))}


def berth_axis_x(region, x, y):
    """Berth-frame x of a point (the frame of _berth_region: centre, heading)."""
    dx, dy = x - region["center_xy"][0], y - region["center_xy"][1]
    return math.cos(region["heading_rad"]) * dx + math.sin(region["heading_rad"]) * dy


def ray_floor_radius_m(start_gap, reach, ray_count):
    """G20d: smallest disc radius a fan of ray_count rays from the COM is guaranteed to hit from any bearing when its
    surface is start_gap beyond the footprint (centre distance <= start_gap + reach + r; half-angle >= pi / n)."""
    s = math.sin(math.pi / int(ray_count))
    return s * (float(start_gap) + float(reach)) / (1.0 - s)


def _fortress_inputs(block, context, d1, radius):
    sensor = context.sensor
    rays = int(sensor.ray_count) if sensor is not None else 36
    reach = context.envelope.footprint.reach_m
    openings = []
    for gate in block["gate_openings"]:
        (ax, ay), (bx, by) = gate["segment"]
        openings.append({"name": str(gate["name"]), "segment": [[float(ax), float(ay)], [float(bx), float(by)]]})
    return {"passage_gap_m": float(block["passage_gap_m"]), "gap_source": str(block.get("source", "")),
            "gate_openings": openings, "base_radius_m": float(radius),
            "ray_floor_m": ray_floor_radius_m(d1, reach, rays), "ray_count": rays, "start_gap_m": float(d1),
            "footprint_reach_m": reach, "footprint_half_width_m": footprint_half_width(context.envelope.footprint)}


def _segment_distance(px, py, segment):
    (ax, ay), (bx, by) = segment
    sx, sy = bx - ax, by - ay
    n2 = sx * sx + sy * sy
    t = 0.0 if n2 <= 0.0 else max(0.0, min(1.0, ((px - ax) * sx + (py - ay) * sy) / n2))
    return math.hypot(px - ax - t * sx, py - ay - t * sy)


def fortress_radius(inputs, context, pose):
    """(radius, detail) of the G20d fortress rule at a site pose (cylinder centre = the pose position).

    r = min(base radius, max(left, right) - g, min over the gate openings of (distance - g), min over every static
    disc of (centre distance - r_o) - hw); left / right = min over the static discs on that side of the nominal
    heading of the centre-to-surface distance. The last term is the rule-6 placement clearance (>= hw to every
    static surface), so r is the largest radius satisfying every dose-independent rule the radius enters."""
    g, hw = inputs["passage_gap_m"], inputs["footprint_half_width_m"]
    hx, hy = math.cos(pose.yaw), math.sin(pose.yaw)
    left = right = nearest = math.inf
    for o in context.static_obstacles:
        dist = math.hypot(o.x - pose.x, o.y - pose.y) - o.radius
        nearest = min(nearest, dist)
        if hx * (o.y - pose.y) - hy * (o.x - pose.x) >= 0.0:
            left = min(left, dist)
        else:
            right = min(right, dist)
    passage = max(left, right) - g
    gate_bound = min((_segment_distance(pose.x, pose.y, gate["segment"]) - g for gate in inputs["gate_openings"]),
                     default=math.inf)
    clearance = nearest - hw
    terms = (("base_radius", inputs["base_radius_m"]), ("passage", passage), ("gate_opening", gate_bound),
             ("static_clearance", clearance))
    binding, value = min(terms, key=lambda kv: kv[1])

    def finite(v):
        return None if math.isinf(v) else float(v)

    detail = {"passage_gap_m": g, "gap_source": inputs["gap_source"],
              "left_surface_distance_m": finite(left), "right_surface_distance_m": finite(right),
              "passage_bound_m": finite(passage), "gate_bound_m": finite(gate_bound),
              "static_clearance_bound_m": finite(clearance), "base_radius_m": inputs["base_radius_m"],
              "radius_m": float(value), "binding": binding, "ray_floor_m": inputs["ray_floor_m"],
              "ray_count": inputs["ray_count"], "start_gap_m": inputs["start_gap_m"],
              "footprint_reach_m": inputs["footprint_reach_m"]}
    return float(value), detail


def visibility_block(context, length_scale):
    """G20d: the key's static occluder discs and the ray fan (pool-time and run-time visibility, occlusion.py)."""
    sensor = context.sensor
    rays = int(sensor.ray_count) if sensor is not None else 36
    reach = float(sensor.max_range_m) if sensor is not None else 30.0 * float(length_scale)
    return {"model": occlusion.RAY_MODEL, "ray_count": rays, "max_range_m": reach,
            "occluders": [[float(o.x), float(o.y), float(o.radius)] for o in context.static_obstacles]}


def berth_clearance(region, x, y, radius):
    """Clearance between a cylinder and the berth rectangle (walls included); <= 0 means overlap."""
    dx, dy = x - region["center_xy"][0], y - region["center_xy"][1]
    c, s = math.cos(region["heading_rad"]), math.sin(region["heading_rad"])
    bx, by = c * dx + s * dy, -s * dx + c * dy
    ex = max(region["x_min_m"] - bx, 0.0, bx - region["x_max_m"])
    ey = max(abs(by) - region["half_width_m"], 0.0)
    return math.hypot(ex, ey) - radius


def hold_pose(plan):
    """Nominal hold pose of a hold task: the plan's final pose (approach end, at rest)."""
    return plan.states[-1].pose


def station_centre_pose(context, plan):
    """Station centre with the nominal approach heading (the pose the side rule keeps clear by hw)."""
    hold = context.hold_circle if context.hold_circle is not None else context.goal
    return Pose(float(hold.x), float(hold.y), float(hold_pose(plan).yaw), 0.0)


def nominal_min_gap(plan, x, y, radius_m, footprint):
    """Smallest footprint-to-surface gap over the nominal plan (TRIGGER_SUBSAMPLES samples per interval)."""
    low = gap_at(plan.states[0].pose, x, y, radius_m, footprint)
    for before, after in zip(plan.states, plan.states[1:]):
        span = after.time_s - before.time_s
        for index in range(1, TRIGGER_SUBSAMPLES + 1):
            t = after.time_s if index == TRIGGER_SUBSAMPLES else before.time_s + span * index / TRIGGER_SUBSAMPLES
            low = min(low, gap_at(plan.pose_at(t), x, y, radius_m, footprint))
    return low


def _radial_distance_for_gap(pose, bearing, radius, footprint, gap):
    """Distance rho from the COM along ``bearing`` where the footprint-to-surface gap equals ``gap``.

    The footprint is a convex rectangle containing the COM, so the gap grows monotonically with rho.
    """
    def gap_of(rho):
        return gap_at(pose, pose.x + rho * math.cos(bearing), pose.y + rho * math.sin(bearing), radius, footprint)
    low, high = 0.0, footprint.length_m + radius + gap + 1.0
    while gap_of(high) < gap:
        high *= 2.0
    for _ in range(SK_BISECTIONS):
        mid = 0.5 * (low + high)
        if gap_of(mid) < gap:
            low = mid
        else:
            high = mid
    return high


def station_clearance_m(context):
    """Required clearance between the station centre and the cylinder surface: max(hold radius + hw, footprint
    reach + hw) (hold circle clear, centre clear at every heading). G20d (REV-G20c M6): the G17 evade clearance is no
    longer part of it (scene-only admission); see g17_evade_target_clear in the site document (report only)."""
    hold = context.hold_circle if context.hold_circle is not None else context.goal
    fp = context.envelope.footprint
    hw = footprint_half_width(fp)
    return max(float(hold.radius) + hw, fp.reach_m + hw)


def _lane_candidate(plan, s_back, side, radius, footprint, gap):
    """(lane pose, cylinder x, y) of a lane point s_back before the stop, cylinder on the given side (+1 port)."""
    length = plan.path_length_m
    if not 0.0 <= s_back < length:
        return None
    pose = plan.pose_at(plan.time_at_fraction((length - s_back) / length))
    bearing = pose.yaw + side * 0.5 * math.pi
    rho = _radial_distance_for_gap(pose, bearing, radius, footprint, gap)
    return pose, pose.x + rho * math.cos(bearing), pose.y + rho * math.sin(bearing)


def _place_station_lane(context, plan, rng, radius, d1, length_scale, document_at):
    stop = hold_pose(plan)
    station = station_centre_pose(context, plan)
    footprint = context.envelope.footprint
    hw = footprint_half_width(footprint)
    d4 = config.smallest_main_trigger_gap_m(context.envelope.t_avoid_s, context.envelope.vessel_id)
    gap = hw + SK_LANE_GAP_INTERPOLATION * (d4 - hw)
    need = station_clearance_m(context)
    step = SK_LANE_SCAN_STEP_LENGTHS * float(length_scale)
    window = SK_LANE_WINDOW_LENGTHS * float(length_scale)
    t_end = plan.end_s
    starts = {}
    for side in (1, -1):
        s, s_lo = 0.0, None
        while s < plan.path_length_m:
            cand = _lane_candidate(plan, s, side, radius, footprint, gap)
            if cand is not None and math.hypot(cand[1] - station.x, cand[2] - station.y) - radius >= need:
                s_lo = s
                break
            s += step
        starts[side] = s_lo

    def draw():
        u = float(rng.uniform(-1.0, 1.0))                  # one draw: sign = side, |u| = position in the window
        return u

    def site_of(u):
        side = 1 if u >= 0.0 else -1
        s_lo = starts[side]
        s_back = None if s_lo is None else s_lo + abs(u) * window
        cand = None if s_back is None else _lane_candidate(plan, s_back, side, radius, footprint, gap)
        if cand is None:                                   # no admissible lane window on this side
            lane, x, y = stop, station.x, station.y
        else:
            lane, x, y = cand
        g_min = nominal_min_gap(plan, x, y, radius, footprint)
        speed = None
        if hw <= g_min < d4 and gap_at(plan.states[0].pose, x, y, radius, footprint) > d4:
            t_min, _, _ = nominal_trigger(plan, x, y, radius, footprint, d4)
            speed = plan.pose_at(t_min).speed
        site = {"rule": "station_lane_flank", "side": side, "draw": u, "lane_window_start_m": s_lo,
                "lane_window_length_m": window, "s_back_m": s_back, "lane_pose": [lane.x, lane.y, lane.yaw],
                "lane_gap_target_m": gap, "station_centre": [station.x, station.y],
                "station_clearance_required_m": need,
                "station_surface_distance_m": math.hypot(x - station.x, y - station.y) - radius,
                "hold_radius_m": float((context.hold_circle or context.goal).radius),
                "g17_evade_clearance_m": SK_G17_EVADE_CLEARANCE_M.get(context.envelope.vessel_id),
                "footprint_reach_m": footprint.reach_m, "nominal_min_gap_m": g_min, "smallest_trigger_gap_m": d4,
                "footprint_half_width_m": hw, "ladder_option": config.LADDER_OPTION,
                "nominal_speed_at_smallest_trigger_mps": speed,
                "min_trigger_speed_fraction": SK_MIN_TRIGGER_SPEED_FRACTION,
                "hold_pose": [stop.x, stop.y, stop.yaw]}
        return t_end, Pose(x, y, lane.yaw, 0.0), site

    document = _redraw("no_clear_site_on_approach_lane", draw, site_of, document_at, context, d1)
    return document, plan


def station_line_point(context):
    """G20d / controller decision J9: (pose, D_start, rho) of the station-keeping site P = S + (1 - SK_ROUTE_FRACTION)
    (C - S) on the straight start-to-hold-centre line (heading = bearing S -> C): the point of the SK reference route
    (turn in place to face C, then straight) at straight-line distance rho = SK_ROUTE_FRACTION x D_start from C."""
    hold = context.hold_circle if context.hold_circle is not None else context.goal
    start = context.initial_pose
    dx, dy = float(hold.x) - float(start.x), float(hold.y) - float(start.y)
    d_start = math.hypot(dx, dy)
    if not d_start > 0.0:
        raise PlacementError("station_start_at_hold_centre")
    f = 1.0 - SK_ROUTE_FRACTION
    pose = Pose(float(start.x) + f * dx, float(start.y) + f * dy, math.atan2(dy, dx), 0.0)
    return pose, d_start, SK_ROUTE_FRACTION * d_start


def station_route_point(context, plan):
    """Reference only since J9 (not the site): (time, pose, D_start, rho) of the FIRST crossing of the curved D7 nominal
    plan with the circle of radius rho = SK_ROUTE_FRACTION x D_start about C (TRIGGER_SUBSAMPLES samples per plan
    interval, then TRIGGER_BISECTIONS bisections); used to report how far the plan passes from the J9 site."""
    hold = context.hold_circle if context.hold_circle is not None else context.goal
    start = context.initial_pose
    d_start = math.hypot(hold.x - start.x, hold.y - start.y)
    rho = SK_ROUTE_FRACTION * d_start

    def dist(t):
        pose = plan.pose_at(t)
        return math.hypot(pose.x - hold.x, pose.y - hold.y)

    if dist(plan.states[0].time_s) <= rho:
        raise PlacementError("station_route_point_at_start")
    previous = plan.states[0].time_s
    for before, after in zip(plan.states, plan.states[1:]):
        span = after.time_s - before.time_s
        for index in range(1, TRIGGER_SUBSAMPLES + 1):
            t = after.time_s if index == TRIGGER_SUBSAMPLES else before.time_s + span * index / TRIGGER_SUBSAMPLES
            if dist(t) <= rho:
                low, high = previous, t
                for _ in range(TRIGGER_BISECTIONS):
                    mid = 0.5 * (low + high)
                    if dist(mid) <= rho:
                        high = mid
                    else:
                        low = mid
                return high, plan.pose_at(high), d_start, rho
            previous = t
    raise PlacementError("nominal_plan_never_reaches_station_route_point")


def _plan_min_distance(plan, x, y):
    """Smallest COM distance of the nominal plan to (x, y) (TRIGGER_SUBSAMPLES samples per interval; report only)."""
    low = math.hypot(plan.states[0].pose.x - x, plan.states[0].pose.y - y)
    for before, after in zip(plan.states, plan.states[1:]):
        span = after.time_s - before.time_s
        for index in range(1, TRIGGER_SUBSAMPLES + 1):
            pose = plan.pose_at(before.time_s + span * index / TRIGGER_SUBSAMPLES)
            low = min(low, math.hypot(pose.x - x, pose.y - y))
    return low


def _place_station_route(context, plan, radius, d1, document_at):
    """G20d station_route_block: the cylinder centre on the straight start-to-centre line at 0.45 D_start from the hold
    centre (controller decision J9; deterministic, no stream draw); every dose-independent rule must pass, else the
    key is not admitted."""
    pose, d_start, rho = station_line_point(context)
    t_curve, curve, _, _ = station_route_point(context, plan)
    t_site = t_curve                                       # recorded only (the site is the straight-line point)
    hold = context.hold_circle if context.hold_circle is not None else context.goal
    footprint = context.envelope.footprint
    hw = footprint_half_width(footprint)
    surface = math.hypot(pose.x - hold.x, pose.y - hold.y) - radius
    task = config.journal_task(context.task)
    vessel = context.envelope.vessel_id
    offered_main = config.offered_main_trigger_gaps_m(task, vessel)
    d_min = min(offered_main.values())
    g_min = nominal_min_gap(plan, pose.x, pose.y, radius, footprint)
    speed = None
    if gap_at(plan.states[0].pose, pose.x, pose.y, radius, footprint) > d_min:
        try:
            t_min, _, _ = nominal_trigger(plan, pose.x, pose.y, radius, footprint, d_min)
            speed = plan.pose_at(t_min).speed
        except PlacementError:
            speed = None
    stop = hold_pose(plan)
    site = _versioned({
        "rule": "station_route_block", "route_fraction_of_start_distance": SK_ROUTE_FRACTION,
        "start_to_centre_distance_m": d_start, "site_centre_distance_m": rho,
        "definition": ("controller decision J9 (shared with the D7 SK meeting point): "
                       "P = start + 0.55 (centre - start), "
                       "on the straight start-to-hold-centre line at 0.45 x D_start from the centre; the SK reference "
                       "route turns in place to face the centre, then runs straight"),
        "route_plan_first_crossing_s": t_site, "route_plan_first_crossing_xy": [curve.x, curve.y],
        "route_plan_offset_m": math.hypot(curve.x - pose.x, curve.y - pose.y),
        "route_plan_min_distance_m": _plan_min_distance(plan, pose.x, pose.y), "path_length_m": plan.path_length_m,
        "station_centre": [float(hold.x), float(hold.y)], "hold_radius_m": float(hold.radius),
        "station_clearance_required_m": station_clearance_m(context), "station_surface_distance_m": surface,
        "g17_evade_clearance_m": SK_G17_EVADE_CLEARANCE_M.get(vessel),
        "g17_evade_target_clear": (None if vessel not in SK_G17_EVADE_CLEARANCE_M
                                   else surface >= SK_G17_EVADE_CLEARANCE_M[vessel]),        # report only (M6)
        "footprint_reach_m": footprint.reach_m,
        "footprint_half_width_m": hw, "nominal_min_gap_m": g_min, "smallest_offered_trigger_gap_m": d_min,
        "offered_main_doses": sorted(offered_main),
        "not_applicable_doses": list(config.not_applicable_doses(task, vessel)),
        "start_gap_m": d1, "nominal_speed_at_smallest_trigger_mps": speed,
        "min_trigger_speed_fraction": SK_MIN_TRIGGER_SPEED_FRACTION, "ladder_option": config.LADDER_OPTION,
        "hold_pose": [stop.x, stop.y, stop.yaw]})
    document = document_at(t_site, site, Pose(pose.x, pose.y, pose.yaw, 0.0))
    reasons = site_reasons(context, document, d1)
    if reasons:
        raise PlacementError("no_clear_site_on_approach_route", reasons)
    return document, plan


def native_sweep_clearance(native, x, y, radius):
    """Clearance between a cylinder and the dose-independent swept envelope of one native mover."""
    if native.kind == "translate":
        half = 0.5 * native.span_m
        ax, ay = native.cx - native.dx * half, native.cy - native.dy * half
        px, py = x - ax, y - ay
        t = min(max(px * native.dx + py * native.dy, 0.0), native.span_m)
        distance = math.hypot(px - native.dx * t, py - native.dy * t)
    elif native.kind == "orbit":
        distance = abs(math.hypot(x - native.cx, y - native.cy) - native.orbit_radius_m)
    else:
        distance = math.hypot(x - native.cx, y - native.cy)
    return distance - native.radius_m - radius


def site_reasons(context, document, d1):
    """Dose-independent clearance rules a site candidate must pass: rule-6 clearances + start gap > d_1.

    d_1 is the start gap (config.start_gap_m: the largest witnessed gap, and the largest calibration-sweep gap
    for the calibration tasks under the metre ladder), so no dose and no sweep gap can trigger at step 0.
    """
    reasons = list(placement_reasons(context, document))
    cyl = document["cylinder"]
    if gap_at(context.initial_pose, cyl["x_m"], cyl["y_m"], cyl["radius_m"], context.envelope.footprint) <= d1:
        reasons.append("trigger_gap_reached_at_start")
    return reasons


def placement_reasons(context, document):
    """Rule-6 placement checks (clearance >= footprint half-width); every failing reason, in order."""
    cyl = document["cylinder"]
    x, y, r = cyl["x_m"], cyl["y_m"], cyl["radius_m"]
    footprint = context.envelope.footprint
    hw = footprint_half_width(footprint)
    reasons = []
    if any(math.hypot(x - o.x, y - o.y) - r - o.radius < hw for o in context.static_obstacles):
        reasons.append("overlaps_static_obstacle")
    if gap_at(context.initial_pose, x, y, r, footprint) < hw:
        reasons.append("too_close_to_start")
    goal = context.goal
    site = document.get("site") or {}
    if site.get("rule") == "station_route_block":
        # G20d (decisions 1B / 2B): on the approach route; the station stays clear (hold circle + hw, every heading;
        # REV-G20c M6: no controller constant); every offered main dose triggers on the approach while moving
        cx, cy = site["station_centre"]
        surface = math.hypot(x - cx, y - cy) - r
        if surface < station_clearance_m(context) - 1e-9:
            reasons.append("too_close_to_station")
        if surface < footprint.reach_m + hw - 1e-9:
            reasons.append("station_centre_not_clear_at_every_heading")
        speed = site["nominal_speed_at_smallest_trigger_mps"]
        if speed is None or speed < SK_MIN_TRIGGER_SPEED_FRACTION * config.REFERENCE_SPEED_MPS:
            reasons.append("station_trigger_after_vessel_stops")
    elif site.get("rule") == "station_lane_flank":
        # G20c (REV-G20b B1): the station stays clear (hold circle + hw, G17 evade clearance, every heading), the
        # nominal approach passes at a gap in [hw, D4) and still moves at the D4 trigger
        cx, cy = site["station_centre"]
        surface = math.hypot(x - cx, y - cy) - r
        if site.get("s_back_m") is None:
            reasons.append("no_lane_window_on_this_side")
        if surface < station_clearance_m(context) - 1e-9:
            reasons.append("too_close_to_station")
        if surface < footprint.reach_m + hw - 1e-9:
            reasons.append("station_centre_not_clear_at_every_heading")
        d_min = config.smallest_main_trigger_gap_m(context.envelope.t_avoid_s, context.envelope.vessel_id)
        if not hw <= site["nominal_min_gap_m"] < d_min:
            reasons.append("nominal_approach_gap_outside_hw_to_d4")
        speed = site["nominal_speed_at_smallest_trigger_mps"]
        if speed is None or speed < SK_MIN_TRIGGER_SPEED_FRACTION * config.REFERENCE_SPEED_MPS:
            reasons.append("station_trigger_after_vessel_stops")
    elif math.hypot(x - goal.x, y - goal.y) - r - goal.radius < hw:
        reasons.append("overlaps_goal_region")
    flags = context.route_mandatory or (False,) * len(context.route)
    if any(keep and math.hypot(x - c.x, y - c.y) - r - c.radius < hw for c, keep in zip(context.route, flags)):
        reasons.append("overlaps_gate_region")
    if any(native_sweep_clearance(n, x, y, r) < hw for n in context.native_geometry):
        reasons.append("overlaps_native_sweep")          # G20: moving basin only (no other task has natives)
    berth = site.get("berth_exclusion")
    if berth is not None and berth_clearance(berth, x, y, r) < hw:
        reasons.append("overlaps_berth")                 # G20b: docking, never inside the berth geometry
    mouth = site.get("berth_mouth_rule")
    if mouth is not None and berth is not None and \
            mouth["opening_x_m"] - berth_axis_x(berth, x, y) < mouth["min_axis_distance_m"] - 1e-9:
        reasons.append("too_close_to_berth_mouth")       # G20d: room to pass and re-align before the mouth
    radius_rule = site.get("fortress_radius_rule")
    if radius_rule is not None and r < radius_rule["ray_floor_m"] - 1e-12:
        reasons.append("fortress_radius_below_ray_floor")  # G20d: the fan would not see it from every bearing
    return reasons


def nominal_trigger(plan, x, y, radius_m, footprint, gap_m):
    """(time, gap, bearing) of the first nominal-plan time with gap <= gap_m; PlacementError otherwise.

    The plan is sampled TRIGGER_SUBSAMPLES times per interval (exact unicycle
    poses, kinematics.integrate) and the first crossing is bisected.
    """
    check_number("gap_m", gap_m, positive=True)
    first = plan.states[0]
    if gap_at(first.pose, x, y, radius_m, footprint) <= gap_m:
        raise PlacementError("trigger_gap_reached_at_start")
    previous = first.time_s
    for before, after in zip(plan.states, plan.states[1:]):
        span = after.time_s - before.time_s
        for index in range(1, TRIGGER_SUBSAMPLES + 1):
            t = after.time_s if index == TRIGGER_SUBSAMPLES else before.time_s + span * index / TRIGGER_SUBSAMPLES
            if gap_at(plan.pose_at(t), x, y, radius_m, footprint) <= gap_m:
                low, high = previous, t
                for _ in range(TRIGGER_BISECTIONS):
                    mid = 0.5 * (low + high)
                    if gap_at(plan.pose_at(mid), x, y, radius_m, footprint) <= gap_m:
                        high = mid
                    else:
                        low = mid
                pose = plan.pose_at(high)
                return high, gap_at(pose, x, y, radius_m, footprint), bearing_at(pose, x, y)
            previous = t
    raise PlacementError("nominal_plan_never_reaches_trigger_gap")


def candidate(context, document, plan, key, dose, cfg):
    """(candidate document, WitnessRequest) of one witnessed dose of the active ladder; raises PlacementError."""
    if dose not in config.WITNESSED_DOSES:
        raise ValueError("witnessed appearing doses are %s (ladder option %s)"
                         % (list(config.WITNESSED_DOSES), config.LADDER_OPTION))
    config.check_dose_applicable(config.journal_task(context.task), context.envelope.vessel_id, dose)   # G20d
    envelope = context.envelope
    cyl = document["cylinder"]
    x, y, r = cyl["x_m"], cyl["y_m"], cyl["radius_m"]
    gap_m = config.trigger_gap_m(dose, envelope.t_avoid_s, envelope.vessel_id)
    floor_rung = config.LADDER_KIND == "metres" and dose in config.SUPPLEMENTARY_DOSES
    try:
        t_trigger, gap, bearing = nominal_trigger(plan, x, y, r, envelope.footprint, gap_m)
    except PlacementError as error:
        # G20b: the witness-floor rung (calibration) need not be reached by the nominal plan; the placement must
        # then stay feasible with the cylinder present from t = 0 (witness released after the declared reaction)
        if not (floor_rung and error.reason == "nominal_plan_never_reaches_trigger_gap"):
            raise
        t_trigger, gap, bearing = None, None, None
    station = (document.get("site") or {}).get("rule") in ("station_lane_flank", "station_route_block") \
        and dose in config.MAIN_DOSES
    if station:
        # G20b: every MAIN dose must reach the vessel on the final approach while it still moves
        speed = plan.pose_at(t_trigger).speed
        if speed < SK_MIN_TRIGGER_SPEED_FRACTION * config.REFERENCE_SPEED_MPS:
            raise PlacementError("station_trigger_after_vessel_stops")
    release = (0.0 if t_trigger is None else t_trigger) + envelope.reaction_s
    if release > plan.end_s + 1e-9:
        raise PlacementError("release_after_nominal_plan_end")
    ratio = config.gap_ratio(dose)
    cylinder = AppearingCylinder(x, y, r, 0.0 if t_trigger is None else t_trigger)
    # D7's Encounter carries doses D0..D4; the witness never reads the dose (feasibility.find_witness), so the
    # supplementary rung of ladder option E (D5) is witnessed as an encounter tagged with its real dose
    metadata = (("vessel_id", envelope.vessel_id), ("suite", config.SUITE_KEY))
    if dose > 4:
        metadata += (("appearing_dose", int(dose)),)
    encounter = Encounter(key=key, task=context.task, dose=min(dose, 4), t_avoid_s=envelope.t_avoid_s,
                          nominal_ttc_s=(ratio * envelope.t_avoid_s if ratio is not None
                                         else gap_m / config.REFERENCE_SPEED_MPS), intruders=(cylinder,),
                          duration_s=float(context.deadline_s), metadata=metadata)
    request = adapters._witness_request(context, cfg, encounter, plan.prefix_until(release),
                                        adapters.PROFILES[context.task])
    doc = {"schema": CANDIDATE_SCHEMA, "position_digest": digest(document), "dose": dose,
           "gap_ratio": ratio, "trigger_gap_m": gap_m, "reference_speed_mps": config.REFERENCE_SPEED_MPS,
           "t_avoid_s": envelope.t_avoid_s, "reaction_s": envelope.reaction_s,
           "nominal_trigger_s": t_trigger, "nominal_trigger_gap_m": gap, "nominal_trigger_bearing_rad": bearing,
           "witness_release_s": release, "witness_horizon_s": float(request.horizon_s),
           "witness_prefix_states": len(request.prefix)}
    if station:
        doc["nominal_trigger_speed_mps"] = speed
    if ratio is None:                    # G20b metre ladder: the equivalent D7 ratio of the gap
        doc["k_equivalent"] = config.k_equivalent(gap_m, envelope.t_avoid_s)
        doc["nominal_trigger_reached"] = t_trigger is not None
    vis = document.get("visibility")
    if vis is not None:
        # G20d (decision 7A): nominal visibility at this dose's nominal trigger pose (None: trigger not reached)
        rays = None
        if t_trigger is not None:
            pose = plan.pose_at(t_trigger)
            rays = occlusion.visible_rays(pose.x, pose.y, pose.yaw, (x, y, r), vis["occluders"], vis["max_range_m"],
                                          vis["ray_count"])
        doc["nominal_visible_rays"] = rays
        doc["nominal_visible_at_trigger"] = None if rays is None else rays > 0
    return doc, request


FALLBACK_REASON_SUFFIX = "_fallback_beam_%d"      # G20d: witness reason of a fallback-beam witness
FALLBACK_FAILED_REASON = "no_witness_in_frozen_search_or_fallback_beam_%d"   # G20d: both searches failed
STITCH_REASON = "monotone_stitched_from_D%d"       # G20d: witness reason of a monotone-stitched witness


def witness_outcome(request, fallback_beam_width=0):
    """(WitnessResult or None, reason, sampled states): D7 find_witness + independent replay.

    G20d (decision 8B): when the frozen beam finds no witness (reason no_witness_in_frozen_search) and
    ``fallback_beam_width`` > the request's beam width, the SAME request is searched once more with that beam; a
    witness found there carries the reason "<D7 reason>_fallback_beam_<w>" and is accepted only through the same
    replay against the original request (replay_witness never reads the beam width). When both fail the reason is
    "no_witness_in_frozen_search_or_fallback_beam_<w>"."""
    try:
        result = find_witness(request)
    except ValueError as error:
        return None, "invalid_witness_request: " + str(error), 0
    sampled = result.sampled_states
    if not result.accepted and result.reason == "no_witness_in_frozen_search" and \
            int(fallback_beam_width) > int(request.beam_width):
        wide = find_witness(_replace(request, beam_width=int(fallback_beam_width)))
        sampled += wide.sampled_states
        if wide.accepted:
            result = WitnessResult(True, wide.reason + FALLBACK_REASON_SUFFIX % int(fallback_beam_width),
                                   wide.trajectory, wide.min_clearance_m, sampled, wide.rejected_states)
        else:
            result = WitnessResult(False, FALLBACK_FAILED_REASON % int(fallback_beam_width), sampled_states=sampled)
    if not result.accepted:
        return None, result.reason, sampled
    checked = replay_witness(request, result.trajectory)
    if not checked.accepted:
        return None, "witness_replay_failed: " + checked.reason, 0
    return result, "", result.sampled_states


def stitch_witness(request, source):
    """G20d monotone stitching (REV-G20b monotone_probe.py): request's own nominal prefix (up to its release time)
    followed by the samples of ``source`` (the witness of a dose with a SMALLER gap) after that time; returned only
    if replay_witness(request, spliced) accepts (WitnessResult) else None."""
    prefix = _witness_prefix(request)
    t_b = prefix[-1].time_s
    spliced = tuple(prefix) + tuple(state for state in source.trajectory if state.time_s > t_b + 1e-12)
    checked = replay_witness(request, spliced)
    return checked if checked.accepted else None


def admit_key(context, key, cfg, *, length_scale, radius_range_m=None, regions=None):
    """Admission of one key over its offered doses (pure, deterministic; process-pool safe).

    Returns {"key", "geometry_id", "placement" (document or None), "doses":
    {dose: {"candidate", "witness" (WitnessResult)}}, "reasons": [(dose, reason)],
    "optional_reasons": [(dose, reason)]} with reasons empty iff admitted. Placement-level reasons carry dose 0
    (DOSE_INDEPENDENT). G20d: the offered doses are the witnessed doses minus the doses not applicable by design
    (config.offered_doses); the key is admitted iff every REQUIRED dose (config.required_doses: all offered doses,
    except the supplementary rung on the decision-5B tasks) has a witness; an OPTIONAL dose without a witness leaves
    the key admitted and is recorded in optional_reasons (the key is not admitted at that dose only). A rejected key
    lists every offered dose: its failure reason, or "joint_ladder_member_rejected" for a dose that passed (E4 form).
    Witness search: the frozen beam, then once more with cfg.witness_fallback_beam_width (decision 8B); on
    config.MONOTONE_STITCH_TASKS a REQUIRED dose whose search failed is retried by monotone stitching from every
    searched witness of a larger dose (smaller gap), largest first (stitch_witness). Budget rule (outside the
    stitching tasks): once a REQUIRED dose has failed, the key is rejected whatever the other doses give, so the
    remaining doses are searched with the frozen beam only (their reasons are still recorded)."""
    if cfg.vessel_id and cfg.vessel_id != context.envelope.vessel_id:
        raise ValueError("appearing suite vessel differs from the context envelope")
    task = config.journal_task(context.task)
    vessel = context.envelope.vessel_id
    out = {"key": key, "geometry_id": str(context.geometry_id), "placement": None, "doses": {}, "reasons": [],
           "optional_reasons": []}
    try:
        document, plan = place(context, key, cfg, length_scale=length_scale, radius_range_m=radius_range_m,
                               regions=regions)
    except PlacementError as error:
        out["reasons"].extend((DOSE_INDEPENDENT, reason) for reason in error.reasons)
        return out
    out["placement"] = document
    reasons = [(DOSE_INDEPENDENT, reason) for reason in placement_reasons(context, document)]
    if reasons:
        out["reasons"] = reasons
        return out
    offered = config.offered_doses(task, vessel)
    required = set(config.required_doses(task, vessel))
    failed, searched, passed = {}, {}, {}
    stitching = task in config.MONOTONE_STITCH_TASKS
    for dose in offered:
        try:
            doc, request = candidate(context, document, plan, key, dose, cfg)
        except PlacementError as error:
            failed[dose] = error.reason
            continue
        rejected = not stitching and any(d in required for d in failed)
        result, reason, _ = witness_outcome(request, 0 if rejected else cfg.witness_fallback_beam_width)
        if result is None:
            failed[dose] = reason
            searched[dose] = (doc, request)
            continue
        passed[dose] = {"candidate": doc, "witness": result}
    if stitching:
        direct = sorted(passed)
        for dose in sorted(failed):
            if dose not in required or dose not in searched:
                continue
            doc, request = searched[dose]
            for source in sorted((d for d in direct if d > dose), reverse=True):
                checked = stitch_witness(request, passed[source]["witness"])
                if checked is not None:
                    passed[dose] = {"candidate": doc, "witness": WitnessResult(
                        True, STITCH_REASON % source, checked.trajectory, checked.min_clearance_m,
                        checked.sampled_states, 0)}
                    del failed[dose]
                    break
    if any(dose in required for dose in failed):
        reasons = [(dose, failed[dose]) for dose in offered if dose in failed]
        reasons.extend((dose, "joint_ladder_member_rejected") for dose in offered if dose in passed)
        out["reasons"] = reasons
        return out
    out["doses"] = {dose: passed[dose] for dose in offered if dose in passed}
    out["optional_reasons"] = [(dose, failed[dose]) for dose in offered if dose in failed]
    return out


__all__ = ["PlacementError", "AppearingCylinder", "SITE_RULES", "ROUTE_FRACTION_RANGE", "DOSE_INDEPENDENT",
           "PLACEMENT_RULES_VERSION", "ROUTE_WINDOWS", "ROUTE_WINDOW_MAX_SITE_CANDIDATES", "SK_LANE_GAP_INTERPOLATION",
           "SK_LANE_WINDOW_LENGTHS", "SK_LANE_SCAN_STEP_LENGTHS", "SK_G17_EVADE_CLEARANCE_M", "station_clearance_m",
           "SK_MIN_TRIGGER_SPEED_FRACTION", "START_MARGIN_LENGTHS_BY_TASK", "station_centre_pose", "nominal_min_gap",
           "berth_clearance", "hold_pose", "native_sweep_clearance",
           "SCATTER_MAX_SITE_CANDIDATES", "PF_START_MARGIN_LENGTHS", "site_reasons",
           "SK_ROUTE_FRACTION", "DOCKING_MOUTH_RULE", "FORTRESS_RADIUS_RULE", "station_route_point", "berth_axis_x",
           "station_line_point",
           "ray_floor_radius_m", "fortress_radius", "visibility_block", "stitch_witness", "FALLBACK_REASON_SUFFIX",
           "STITCH_REASON",
           "footprint_half_width", "radius_rule", "gap_at", "bearing_at", "nominal_plan", "place",
           "placement_reasons", "nominal_trigger", "candidate", "witness_outcome", "admit_key"]
