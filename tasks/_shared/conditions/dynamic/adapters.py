"""Twelve task adapters: offline encounter builders + runtime C4-interface adapters.

Offline (before any controller is assigned):
  TaskContext (frozen task geometry for one episode key, env-local metres)
    -> build_ladder(context, key, cfg): one Candidate per dose D0..D4, built on
       the task's frozen NOMINAL plan (never the controller's state)
    -> admit_ladders(ladders): witness each candidate; with joint admission a
       key enters the pool only if ALL doses admit, so every dose is evaluated
       on the same geometry/key set (no dose-dependent selection); every
       rejection is counted with its reason.
Runtime (C4 hook names, conditions/dynamic_hazards.py:28-31):
  reset(env_ids, cfg, scenario)  scenario maps env -> admitted Assignment
  activate(env_ids, cfg)         arms at reset (the timing of activate never retimes the world;
                                 the runtime rule may: see below)
  update(frame, cfg)             frame = RuntimeFrame(episode seconds, observer
                                 poses); NOT C4's voyage fraction
  finish(env, reason)            detached JSON stamp, before the env resets.
All parameters come from the admitted assignment. Under the diagnostic rule
pinned_v0 the observer pose only feeds exposure accounting; under the runtime
rules it also times the object (G23c F11).
G23c L2: the default runtime rule is the candidate S2 = d7_reveal_sync_v1 (reveal.py: the intruder is hidden
until the own ship's constant-velocity first-contact TTC reaches the dose, then revealed moving on its line); rule S
(below) and pinned_v0 are diagnostic options.
G23b (owner decision 2026-10-03, NC 09, option S): under the default runtime
rule (release.RUNTIME_RULE_VERSION) the observer pose ALSO times the release of
the pinned intruder (release.py): it rests at its hold point until the own ship
is about to arrive, then moves on its pinned line. Every admitted parameter is
unchanged; DynamicRuntimeConfig.motion_rule = "pinned_v0" restores the v2.5
controller-independent motion (diagnostic only).

E4 additions: resolve_sensor (journal 36-ray sensor on every task, CONTEXT_R3
decision 14), LEGACY_RANGE_SENSOR (per-task table of the sensor each legacy
env already has), and deterministic process-level parallel admission
(admit_ladders / admit_grid with workers > 1; results identical to serial).
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import math
import multiprocessing
from pathlib import Path

from .capacity import CapacityPlan
from .exposure import ExposureTracker
from .feasibility import WitnessRequest, find_witness, replay_witness
from .geometry import point_clearance, ttc_linear
from .kinematics import integrate
from .model import (DOSE_TTC_RATIOS, UNFORCED_TTC_TOLERANCE_S, Circle, EpisodeKey, Pose, TimedState,
                    VesselEnvelope, is_unforced_command)
from .native import NativeMotion, native_encounter
from .reference import NominalPlan, build_transit_plan
from .release import (DEFAULT_RULE, HYBRID_RULE_VERSION, REVEAL_RULE_VERSION, RUNTIME_RULE_VERSION, RUNTIME_RULES,
                      V3_RULE_VERSION,
                      ReleasedIntruder,
                      ReleaseState, rule_block_for)
from .reveal import RevealState
from . import sk_approach
from .sk_approach import NotApplicableByDesign
from .stamp import digest, make_stamp, semantics_for
from .trajectory import PlacementError, generate_encounter, strata
from .visibility import (JOURNAL_RAY_SENSOR, RevealEvidence, SensorSpec, detectable,
                         edge_range_and_occlusion, nominal_first_detection_ok, validate_reveal)
from .config import DynamicSuiteConfig

TASKS = ("reach", "path_following", "station_keeping", "docking", "path_hazard", "scatter",
         "ring", "fortress", "forced_crossing", "iceberg", "moving_basin", "harbour")
# Range sensor each LEGACY env already has (J-base anchors, re-read by E4):
#   "none"             no ray block in the observation
#   "berth_walls_only" 36 rays / 30 m only when cfg.berth_walls (docking_env.py:781-784;
#                      cfg default False docking_env_cfg.py:214, rays :221-222)
#   "rays_36_30m"      36 rays / 30 m fan (path_hazard_env_cfg.py:185-186, hazard_nav_env_cfg.py:185-186,
#                      harbor_mission_env_cfg.py:179-180; ray casts path_hazard_env.py:680,
#                      hazard_nav_env.py:1196, harbor_mission_env.py:854)
# Reach (blueboat_calm_nav_env.py:1042), path following (path_following_env.py:561) and
# station keeping (station_keeping_env.py:657, "no rays" :146) observe no range sensor.
# With DynamicSuiteConfig.journal_ray_sensor_all_tasks=True (default, PENDING) every task
# is sensed with JOURNAL_RAY_SENSOR, so D1..D4 are distinct experiments on all 12 tasks.
LEGACY_RANGE_SENSOR = {
    "reach": "none", "path_following": "none", "station_keeping": "none",
    "docking": "berth_walls_only", "path_hazard": "rays_36_30m", "scatter": "rays_36_30m",
    "ring": "rays_36_30m", "fortress": "rays_36_30m", "forced_crossing": "rays_36_30m",
    "iceberg": "rays_36_30m", "moving_basin": "rays_36_30m", "harbour": "rays_36_30m",
}
SENSOR_SOURCES = ("task_existing_sensor", "journal_ray_sensor_all_tasks", "none")
# G23e: "approach_point" (station keeping, sk_approach.py) replaces the retired "settled_clock" (reveal at the
# start of the scoring window, encounter during the hold)
SITE_RULES = ("route_fraction", "leg_midpoint", "waypoint_arrival", sk_approach.SK_SITE_RULE, "stage", "native")
# C5 stop causes (termination_reason.py) + D4 validity causes (journal_contracts_v2.py:20-33).
TERMINATION_REASONS = ("success", "collision", "timeout", "out_of_bounds", "failure",
                       "capsize", "sinking", "airborne", "non_finite_state", "aborted")


@dataclass(frozen=True)
class TaskProfile:
    """Per-task encounter rule (C7 STRATEGY.md:146-161; NC04 section 3)."""

    task: str
    carrier: str
    objective: str
    real_world: str
    plan_kind: str               # transit | hold | native
    site_rule: str
    kinds: tuple = ("crossing",)
    sides: tuple = (1, -1)
    fractions: tuple = (0.5,)
    aim_offsets: tuple = (0.0,)
    leads_s: tuple = (0.0,)      # (retired settled_clock rule; one stratum, kept for the strata sizes)
    stop_at_end: bool = False

    def __post_init__(self):
        if self.task not in TASKS or self.site_rule not in SITE_RULES:
            raise ValueError("unknown task or site rule")
        if self.plan_kind not in ("transit", "hold", "native"):
            raise ValueError("unknown plan kind")


PROFILES = {p.task: p for p in (
    TaskProfile("reach", "transit_corridor", "single-target transit",
                "floating debris or small craft crossing the transit corridor",
                "transit", "route_fraction", kinds=("crossing", "drifting"), fractions=(0.4, 0.6)),
    TaskProfile("path_following", "fixed_route_segment", "ordered waypoints",
                "another vessel crossing the channel on a fixed leg",
                "transit", "leg_midpoint"),
    TaskProfile("station_keeping", "approach_route", "continuous hold after the approach",
                "small craft crossing the approach to the worksite (met during the approach, never during the hold)",
                "hold", sk_approach.SK_SITE_RULE, kinds=("crossing",), sides=(1, -1), stop_at_end=True),
    TaskProfile("docking", "approach_lane", "berth position, heading, speed and dwell",
                "harbour traffic crossing the approach lane",
                "transit", "leg_midpoint", stop_at_end=True),
    TaskProfile("path_hazard", "mandatory_route_segment", "ordered waypoints between blockers",
                "vessel crossing a mandatory channel segment", "transit", "leg_midpoint"),
    TaskProfile("scatter", "certified_corridor", "goal through a fixed field",
                "vessel or drifting object inside the corridor",
                "transit", "route_fraction", kinds=("crossing", "drifting"), fractions=(0.4, 0.6)),
    TaskProfile("ring", "escape_gap", "declared escape through the opening",
                "traffic transiting the narrow opening in the opposite direction",
                "transit", "waypoint_arrival", kinds=("head_on",), sides=(1,)),
    TaskProfile("fortress", "entry_gap", "declared ingress through the entry",
                "traffic leaving through the entry while the vessel enters",
                "transit", "waypoint_arrival", kinds=("head_on",), sides=(1,)),
    TaskProfile("forced_crossing", "fixed_gate", "mandatory gate crossing",
                "traffic transiting the lock/gate in the opposite direction",
                "transit", "waypoint_arrival", kinds=("head_on",), sides=(1,)),
    TaskProfile("iceberg", "side_passage", "planned side passage",
                "object approaching through the planned side passage",
                "transit", "waypoint_arrival", kinds=("head_on",), sides=(1,)),
    TaskProfile("moving_basin", "native_field", "timed passage of the native moving field",
                "native moving obstacles (the task's own motion axis)", "native", "native",
                kinds=(), sides=(1,)),
    TaskProfile("harbour", "declared_stage", "ordered harbour stages",
                "harbour traffic at one declared stage", "transit", "stage", stop_at_end=True),
)}


@dataclass(frozen=True)
class StageSite:
    """Harbour: one candidate encounter stage (rule applied to route index)."""

    name: str
    rule: str          # leg_midpoint | waypoint_arrival | route_fraction
    index: int = 0     # route index (fraction for route_fraction is in `fraction`)
    kind: str = "crossing"
    fraction: float = 0.5


@dataclass(frozen=True)
class TaskContext:
    """Frozen, controller-independent geometry of one episode, env-local SI units.

    route: ordered circles the NOMINAL plan visits before the goal.
    route_mandatory: per-route flag; mandatory circles are also witness gates.
    sensor: the task's EXISTING range sensor, or None (no such sensor); with
      journal_ray_sensor_all_tasks the suite then uses JOURNAL_RAY_SENSOR.
    base_max: base-geometry slot maximum (D4's resolved maximum, not the count).
    """

    task: str
    envelope: VesselEnvelope
    initial_pose: Pose
    bounds: tuple
    goal: Circle
    deadline_s: float
    route: tuple = ()
    route_mandatory: tuple = ()
    static_obstacles: tuple = ()
    occluders: tuple = ()
    sensor: object = None
    hold_circle: object = None
    settled_phase_start_s: float = 0.0
    final_speed_mps: object = None
    final_heading_rad: object = None
    heading_tolerance_rad: float = 0.15
    dwell_s: float = 0.0
    site_index: int = 0
    stages: tuple = ()
    native_geometry: tuple = ()
    base_max: int = 0
    base_count: int = 0
    base_slots: tuple = ()
    appearing_reserve: int = 0
    geometry_id: str = ""

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError("unknown task")
        if not isinstance(self.envelope, VesselEnvelope) or not isinstance(self.initial_pose, Pose):
            raise ValueError("context needs a VesselEnvelope and an initial Pose")
        for name in ("bounds", "route", "route_mandatory", "static_obstacles", "occluders",
                     "stages", "native_geometry", "base_slots"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(name + " must be a tuple")
        if self.route_mandatory and len(self.route_mandatory) != len(self.route):
            raise ValueError("route_mandatory must flag every route circle")
        if self.sensor is not None and not isinstance(self.sensor, SensorSpec):
            raise ValueError("sensor must be a SensorSpec or None")
        if any(c not in self.static_obstacles for c in self.occluders):
            raise ValueError("occluders must be physical static obstacles too")
        if not math.isfinite(self.deadline_s) or self.deadline_s <= 0:
            raise ValueError("deadline must be positive")
        if any(type(v) is not int or v < 0 for v in (self.base_max, self.base_count,
                                                        self.appearing_reserve, self.site_index)):
            raise ValueError("slot counts and site index must be nonnegative integers")
        if self.base_count > self.base_max:
            raise ValueError("base count exceeds the declared base maximum")
        if self.task == "moving_basin" and not all(isinstance(o, NativeMotion) for o in self.native_geometry):
            raise ValueError("moving basin needs NativeMotion templates")

    @property
    def mandatory_route(self):
        flags = self.route_mandatory or (False,) * len(self.route)
        return tuple(c for c, keep in zip(self.route, flags) if keep)


@dataclass(frozen=True)
class Candidate:
    task: str
    request: WitnessRequest
    capacity: CapacityPlan
    sensor: object = None
    reveal_evidence: object = None
    settled_phase_start_s: float = 0.0
    stage: str = ""
    site: str = ""
    perceivable: bool = True
    geometry_id: str = ""
    base_slots: tuple = ()
    base_count: int = 0
    sensor_source: str = "none"

    def validate(self):
        """Re-check every admission invariant (also called again at runtime reset)."""
        request, encounter = self.request, self.request.encounter
        if encounter.task != self.task:
            raise ValueError("encounter task mismatch")
        if encounter.duration_s + 1e-9 < request.horizon_s:
            raise ValueError("trajectory must cover the full feasibility deadline")
        if not math.isclose(encounter.t_avoid_s, request.envelope.t_avoid_s, rel_tol=1e-12):
            raise ValueError("encounter t_avoid does not match the frozen envelope")
        if encounter.meta("vessel_id", request.envelope.vessel_id) != request.envelope.vessel_id:
            raise ValueError("encounter built for another vessel")
        if self.sensor_source not in SENSOR_SOURCES or (self.sensor is None) != (self.sensor_source == "none"):
            raise ValueError("sensor source does not match the candidate sensor")
        if self.task == "moving_basin":
            if self.capacity.dynamic_reserve != 0 or len(self.base_slots) != len(encounter.intruders):
                raise ValueError("native basin must own only existing base slots")
            if len(set(self.base_slots)) != len(self.base_slots) or any(
                    type(s) is not int or not 0 <= s < self.capacity.base_max for s in self.base_slots):
                raise ValueError("native slots must be unique base slots")
            self.capacity.validate_counts(self.base_count, 0)
            return
        expected = 0 if encounter.dose == 0 else 1
        if len(encounter.intruders) != expected:
            raise ValueError("exactly one intruder at nonzero dose, none at D0")
        self.capacity.validate_counts(self.base_count, expected)
        if encounter.dose == 0:
            return
        intruder = encounter.intruders[0]
        target = DOSE_TTC_RATIOS[encounter.dose] * request.envelope.t_avoid_s
        if encounter.nominal_ttc_s is None or not math.isclose(encounter.nominal_ttc_s, target, rel_tol=1e-9):
            raise ValueError("nominal TTC dose mismatch")
        reference = Pose(encounter.meta("reference_x_m"), encounter.meta("reference_y_m"),
                         encounter.meta("reference_yaw_rad"), encounter.meta("reference_speed_mps"))
        # G7 item B: the candidate does not carry the plan's window, so the re-check accepts the declared
        # timing bound of a window with sub-tolerance commands (generation applied the exact check
        # whenever the window was exactly unforced).
        if not math.isclose(ttc_linear(reference, request.envelope, intruder, at_s=intruder.reveal_s),
                            target, rel_tol=1e-9, abs_tol=1e-9 + UNFORCED_TTC_TOLERANCE_S):
            raise ValueError("swept-footprint TTC does not deliver the assigned dose")
        if self.perceivable:
            validate_reveal(encounter, self.reveal_evidence, self.sensor)
            if any(c not in request.static_obstacles for c in self.reveal_evidence.occluders):
                raise ValueError("reveal occluders must also exist in physical witness geometry")
        elif self.sensor is not None:
            raise ValueError("an unperceivable candidate cannot carry a sensor")
        if self.task == "station_keeping":          # G23e: approach design (the settled-clock rule is retired)
            sk_approach.validate_candidate(self)
        if self.task == "harbour" and not self.stage:
            raise ValueError("harbour encounter stage must be declared before assignment")


# ------------------------------------------------------------------ builders
def _nominal_plan(profile, context, cfg):
    env = context.envelope
    if profile.plan_kind == "native":
        targets = context.route + (context.goal,)
        return build_transit_plan(context.initial_pose, env, targets, cfg.plan_dt_s, context.deadline_s)
    targets = context.route + (context.goal,)
    if profile.site_rule == sk_approach.SK_SITE_RULE:      # G23e: turn in place, then the straight approach
        plan = build_transit_plan(context.initial_pose, env, targets, cfg.plan_dt_s, context.deadline_s,
                                  stop_at_end=profile.stop_at_end, align_first=True)
    else:
        plan = build_transit_plan(context.initial_pose, env, targets, cfg.plan_dt_s, context.deadline_s,
                                  stop_at_end=profile.stop_at_end)
    if profile.plan_kind == "hold" or (profile.stop_at_end and plan.reached_end):
        plan = _extend_at_rest(plan, context.deadline_s, cfg.plan_dt_s)
    return plan


def _extend_at_rest(plan, until_s, dt_s):
    """Hold the final pose at rest until until_s in dt_s steps (exact rollout)."""
    last = plan.states[-1]
    if not plan.reached_end or abs(last.pose.speed) > 1e-9 or last.time_s >= until_s:
        return plan
    states = list(plan.states)
    pose, time_s = last.pose, last.time_s
    while time_s < until_s - 1e-9:
        step = min(dt_s, until_s - time_s)
        pose = integrate(pose, 0.0, 0.0, step)
        time_s = time_s + step
        states.append(TimedState(time_s, pose, 0.0, 0.0))
    return NominalPlan(tuple(states), plan.target_times, "transit", True)


def _fit_conflict(plan, t_nominal, window, t_max, t_min_reveal):
    """Earliest t >= t_nominal whose D1 window [t - window, t] is unforced.

    Slides the conflict forward along the nominal plan past any forced
    (turning/accelerating) interval; fails if that passes t_max. Pure
    function of the frozen plan: no controller input.
    """
    t = max(t_nominal, t_min_reveal + window)
    for _ in range(len(plan.states) + 1):
        if t > t_max + 1e-9:
            raise PlacementError("no_unforced_d1_window_before_site_end")
        low = t - window
        forced_end = None
        for before, after in zip(plan.states, plan.states[1:]):
            if after.time_s <= low + 1e-12 or before.time_s >= t - 1e-12:
                continue
            if not is_unforced_command(after.accel_mps2, after.yaw_rate_rps):   # G7 item B predicate
                forced_end = after.time_s
        if forced_end is None:
            # E4 fix: t_min_reveal + window - window can round one ulp below
            # t_min_reveal (seen for station keeping with the final registry:
            # D1 reveal 117.54392239748871 < settled 117.54392239748873), so
            # nudge t up by ulps until the D1 reveal is exactly not earlier.
            while t - window < t_min_reveal:
                t = math.nextafter(t, math.inf)
            return t
        t = forced_end + window
        # F6 proposal: forced_end + window - window can round one ulp BELOW forced_end,
        # and is_constant_velocity (no tolerance) then still sees the forced interval
        # (seen on 2 of 128 real harbour keys: "encounter_window_not_on_unforced_nominal_leg").
        while t - window < forced_end:
            t = math.nextafter(t, math.inf)
    raise PlacementError("no_unforced_d1_window_before_site_end")


def _arrival_time(plan, index):
    if index >= len(plan.target_times):
        raise PlacementError("nominal_plan_never_reaches_encounter_site")
    return plan.target_times[index]


def _conflict_time(profile, context, plan, cfg, choice, stage, min_reveal):
    """Frozen conflict time t_c on the nominal plan (then fitted to a straight window)."""
    rule = stage.rule if stage is not None else profile.site_rule
    window = DOSE_TTC_RATIOS[1] * context.envelope.t_avoid_s
    if rule == "route_fraction":
        fraction = stage.fraction if stage is not None else profile.fractions[choice["fraction"]]
        return _fit_conflict(plan, plan.time_at_fraction(fraction), window, plan.end_s, min_reveal)
    index = stage.index if stage is not None else context.site_index
    if rule == "waypoint_arrival":
        arrival = _arrival_time(plan, index)   # the gap/gate itself: no sliding allowed
        return _fit_conflict(plan, arrival, window, arrival, min_reveal)
    if rule == "leg_midpoint":
        end = _arrival_time(plan, index)
        start = 0.0 if index == 0 else _arrival_time(plan, index - 1)
        return _fit_conflict(plan, 0.5 * (start + end), window, end, min_reveal)
    raise ValueError("unsupported site rule " + rule)   # approach_point: sk_approach.meeting (build_candidate)


def _choices(profile, context, key):
    sites = context.stages if profile.site_rule == "stage" else (None,)
    kinds = profile.kinds if profile.site_rule != "stage" else (None,)
    sizes = (len(sites), len(kinds), len(profile.sides), len(profile.fractions),
             len(profile.aim_offsets), len(profile.leads_s))
    s, k, side, fraction, aim, lead = strata(key, sizes)
    stage = sites[s]
    kind = stage.kind if stage is not None else kinds[k]
    return {"stage": stage, "kind": kind, "side": profile.sides[side], "fraction": fraction,
            "aim": profile.aim_offsets[aim], "lead": lead}


def _capacity(context, dynamic):
    reserve = 1 if dynamic else 0
    need = context.base_max + reserve + context.appearing_reserve
    return CapacityPlan(context.base_max, reserve, context.appearing_reserve, need)


def _witness_request(context, cfg, encounter, prefix, profile):
    hold = context.hold_circle
    is_hold = profile.plan_kind == "hold"
    return WitnessRequest(
        initial_pose=context.initial_pose, envelope=context.envelope, encounter=encounter,
        bounds=context.bounds, goal=hold if (is_hold and hold is not None) else context.goal,
        static_obstacles=context.static_obstacles, waypoints=context.mandatory_route,
        dt_s=cfg.witness_dt_s, horizon_s=context.deadline_s, hold_circle=hold if is_hold else None,
        hold_from_s=context.settled_phase_start_s if is_hold else 0.0,
        dwell_s=context.dwell_s, final_speed_mps=context.final_speed_mps,
        final_heading_rad=context.final_heading_rad,
        heading_tolerance_rad=context.heading_tolerance_rad, clearance_m=cfg.witness_clearance_m,
        beam_width=cfg.witness_beam_width, collision_substeps=cfg.witness_substeps,
        minimum_completion_s=context.deadline_s if is_hold else 0.0, prefix=prefix)


def _prefix(plan, encounter, envelope):
    if not encounter.intruders:
        return ()
    release = max(obj.reveal_s for obj in encounter.intruders) + envelope.reaction_s
    if release > plan.end_s + 1e-9:
        raise PlacementError("release_after_nominal_plan_end")
    return plan.prefix_until(release)


def resolve_sensor(context, cfg):
    """(sensor, source) the suite uses for this context.

    The context's existing sensor wins; otherwise, with
    cfg.journal_ray_sensor_all_tasks (default True, CONTEXT_R3 decision 14,
    PENDING) the task is sensed with JOURNAL_RAY_SENSOR; otherwise there is no
    sensor and D7's fail-closed rule applies (allow_unsensed decides).
    """
    if context.sensor is not None:
        return context.sensor, "task_existing_sensor"
    if cfg.journal_ray_sensor_all_tasks:
        return JOURNAL_RAY_SENSOR, "journal_ray_sensor_all_tasks"
    return None, "none"


def build_candidate(context, key, dose, cfg):
    """One dose. Returns a Candidate or raises PlacementError (counted rejection)."""
    if not isinstance(cfg, DynamicSuiteConfig):
        raise ValueError("cfg must be a DynamicSuiteConfig")
    if cfg.vessel_id and cfg.vessel_id != context.envelope.vessel_id:
        raise ValueError("suite vessel differs from the context envelope")
    profile = PROFILES[context.task]
    if key.hazard_group != "d7_" + context.task:
        raise ValueError("hazard_group must be 'd7_<task>' so keys never collide with other streams")
    sensor, sensor_source = resolve_sensor(context, cfg)
    plan = _nominal_plan(profile, context, cfg)
    if profile.plan_kind == "native":
        encounter = native_encounter(key, dose, context.envelope, context.native_geometry,
                                     context.deadline_s, cfg.native_nu_policy, cfg.native_fixed_nu)
        prefix = _prefix(plan, encounter, context.envelope)
        request = _witness_request(context, cfg, encounter, prefix, profile)
        return Candidate(context.task, request, _capacity(context, False), sensor, None,
                         site=profile.carrier, geometry_id=context.geometry_id,
                         base_slots=context.base_slots, base_count=context.base_count,
                         sensor_source=sensor_source)
    choice = _choices(profile, context, key)
    stage = choice["stage"]
    min_reveal = 0.0          # G23e: the hold task's settled-clock floor is retired (approach encounter)
    first_dose, extra = 1, ()
    if profile.site_rule == sk_approach.SK_SITE_RULE:      # G23e: meeting point on the approach
        t_c, point, d_start, reach, offset = sk_approach.meeting(context, plan)
        first_dose = sk_approach.first_applicable_dose(context.envelope, plan, t_c)
        if 0 < dose < first_dose:
            raise NotApplicableByDesign()
        extra = sk_approach.encounter_metadata(t_c, point, d_start, reach, offset)
    else:
        t_c = _conflict_time(profile, context, plan, cfg, choice, stage, min_reveal)
    kind = choice["kind"]
    speed = cfg.drift_speed_mps if kind == "drifting" else cfg.intruder_speed_mps
    encounter = generate_encounter(
        key, context.task, dose, context.envelope, plan, t_c, context.deadline_s,
        kind=kind, side=choice["side"], intruder_speed_mps=speed,
        intruder_size_beams=cfg.intruder_size_beams, reveal_kind=cfg.reveal_kind,
        min_reveal_s=min_reveal, aim_offset_fraction=choice["aim"],
        static_obstacles=context.static_obstacles, static_margin_m=cfg.static_margin_m,
        first_dose=first_dose, extra_metadata=extra)
    if extra and dose > 0:
        encounter = sk_approach.check_hold_exclusion(context, encounter)
    perceivable = sensor is not None
    evidence = None
    if dose > 0:
        if not perceivable and not cfg.allow_unsensed:
            raise PlacementError("task_has_no_existing_range_sensor")
        if perceivable:
            evidence = _reveal_evidence(context, encounter, cfg, sensor)
            if not nominal_first_detection_ok(plan, encounter, evidence, sensor,
                                              cfg.visibility_sample_dt_s):
                raise PlacementError("nominal_observer_detects_before_reveal")
    prefix = _prefix(plan, encounter, context.envelope)
    request = _witness_request(context, cfg, encounter, prefix, profile)
    return Candidate(context.task, request, _capacity(context, True),
                     sensor, evidence,
                     settled_phase_start_s=context.settled_phase_start_s,
                     stage=stage.name if stage is not None else "", site=profile.carrier,
                     perceivable=perceivable, geometry_id=context.geometry_id,
                     base_slots=context.base_slots, base_count=context.base_count,
                     sensor_source=sensor_source)


def _reveal_evidence(context, encounter, cfg, sensor):
    obj = encounter.intruders[0]
    reference = Pose(encounter.meta("reference_x_m"), encounter.meta("reference_y_m"),
                     encounter.meta("reference_yaw_rad"), encounter.meta("reference_speed_mps"))
    if cfg.reveal_kind == "detection_horizon":
        edge, clearance = edge_range_and_occlusion(reference, obj, obj.reveal_s, context.occluders)
        if edge > sensor.max_range_m:
            raise PlacementError("detection_range_exceeds_existing_sensor_range")
        if clearance <= 0.0:
            raise PlacementError("nominal_detection_occluded_at_reveal")
        evidence = RevealEvidence("detection_horizon", reference, edge, context.occluders)
    else:
        evidence = RevealEvidence("occluder_exit", reference, sensor.max_range_m,
                                  context.occluders)
    try:
        validate_reveal(encounter, evidence, sensor)
    except ValueError as error:
        raise PlacementError("reveal_evidence_invalid: " + str(error))
    return evidence


def build_ladder(context, key, cfg):
    """Candidates (or PlacementError) for D0..D4 on the same frozen context and key."""
    out = []
    for dose in range(5):
        try:
            out.append(build_candidate(context, key, dose, cfg.at_dose(dose)))
        except PlacementError as error:
            out.append(error)
    return tuple(out)


# ----------------------------------------------------------------- admission
@dataclass(frozen=True)
class Rejection:
    key: EpisodeKey
    dose: int
    reason: str
    sampled_states: int = 0


@dataclass(frozen=True)
class Assignment:
    candidate: Candidate
    witness: object
    candidate_digest: str
    witness_digest: str
    pool_digest: str


@dataclass(frozen=True)
class AdmissionPool:
    assignments: dict
    rejections: tuple
    keys_offered: int
    keys_admitted: int
    joint: bool
    pool_digest: str
    per_dose_counts: tuple

    def assignment(self, key, dose):
        return self.assignments[(key, dose)]


@dataclass(frozen=True)
class AdmissionPoolNA(AdmissionPool):
    """G23e: an AdmissionPool whose ladders hold tiers that are not applicable by design (station keeping
    BlueBoat D1). not_applicable: Rejection-shaped records (key, dose, sk_approach.SK_NOT_APPLICABLE, 0) that are
    neither admissions nor rejections; the pool digest covers them. Pools without such tiers stay AdmissionPool
    (byte-identical manifests and digests)."""

    not_applicable: tuple = ()


_NA = object()          # G23e: ladder outcome marker of a not-applicable tier (never admitted, never rejected)


def _witness_digest(witness):
    return digest({"reason": witness.reason, "trajectory": witness.trajectory,
                   "min_clearance_m": witness.min_clearance_m})


def _witness_item(item):
    """Witness ONE ladder entry: (witness or None, reason, sampled states).

    Pure function of the frozen candidate (no controller input, no RNG), so
    it gives bit-identical results in any process. PlacementErrors are
    handled by the caller and never reach this function.
    """
    try:
        item.validate()
        witness = find_witness(item.request)
    except ValueError as error:
        return None, "invalid_candidate: " + str(error), 0
    if not witness.accepted:
        return None, witness.reason, witness.sampled_states
    checked = replay_witness(item.request, witness.trajectory)
    if not checked.accepted:
        return None, "witness_replay_failed: " + checked.reason, 0
    return witness, "", witness.sampled_states


def _checked_ladders(ladders):
    """Materialise (key, ladder) pairs in order; duplicate keys are errors."""
    out, seen = [], set()
    for key, ladder in ladders:
        if key in seen:
            raise ValueError("duplicate episode key")
        seen.add(key)
        ladder = tuple(ladder)
        if len(ladder) != 5:
            raise ValueError("a ladder has exactly five doses")
        out.append((key, ladder))
    return out


def _package_identity():
    """(package directory, package module name) of THIS loaded package."""
    name = __name__.rpartition(".")[0]
    return str(Path(__file__).resolve().parent), name


# Worker bootstrap, executed with the builtin exec: a spawned worker unpickles
# its initializer BEFORE it runs, so the initializer cannot live in this
# (path-loaded) package; builtins.exec plus this source string always can.
# It loads ONLY this package, by path, under the parent's module name; the
# parents of a dotted name are empty namespace stubs, so no task package
# __init__, gym registration or simulator import can run in a worker.
_BOOTSTRAP_SOURCE = """
import importlib.util, sys, types
sys.dont_write_bytecode = True
name, folder = %r, %r
if name not in sys.modules:
    parts = name.split(".")
    for index in range(1, len(parts)):
        parent = ".".join(parts[:index])
        if parent not in sys.modules:
            stub = types.ModuleType(parent)
            stub.__path__ = []
            sys.modules[parent] = stub
    spec = importlib.util.spec_from_file_location(name, folder + "/__init__.py",
                                                  submodule_search_locations=[folder])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
"""


def _bootstrap_worker(package_dir, package_name):
    """Process-pool initializer body (also callable directly, e.g. by tests)."""
    exec(_BOOTSTRAP_SOURCE % (package_name, str(package_dir)), {})


def _map_ordered(function, items, workers):
    """function over items, results in input order; workers > 1 -> spawn process pool.

    Results are returned in submission order (Executor.map), so the caller
    assembles exactly the serial sequence; every function used here is a
    deterministic pure function, which makes parallel output identical to
    the serial one (asserted by tests, not assumed).
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1 or len(items) <= 1:
        return [function(item) for item in items]
    package_dir, package_name = _package_identity()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=context,
                             initializer=exec,
                             initargs=(_BOOTSTRAP_SOURCE % (package_name, package_dir), {})) as pool:
        return list(pool.map(function, items, chunksize=1))


def admit_ladders(ladders, joint=True, workers=1):
    """ladders: iterable of (key, 5-tuple from build_ladder). Run once, offline.

    joint=True (default): a key is admitted only if every dose admits; any
    failure removes the key at all doses and every failed dose is recorded.
    Duplicate keys are configuration errors. No retries, no replacements.
    workers > 1 witnesses the (key, dose) entries in a spawn process pool;
    the pool, digests and rejection order are identical to workers == 1.
    """
    checked = _checked_ladders(ladders)
    items = [item for _, ladder in checked for item in ladder if not isinstance(item, PlacementError)]
    outcomes = iter(_map_ordered(_witness_item, items, workers))
    witnessed = []
    for key, ladder in checked:
        results = []
        for dose, item in enumerate(ladder):
            if isinstance(item, NotApplicableByDesign):          # G23e
                results.append((dose, _NA, None, item.reason, 0))
                continue
            if isinstance(item, PlacementError):
                results.append((dose, None, None, item.reason, 0))
                continue
            witness, reason, sampled = next(outcomes)
            results.append((dose, item if witness is not None else None, witness, reason, sampled))
        witnessed.append((key, results))
    return _assemble(witnessed, joint)


def _build_and_witness(job):
    """(context, key, cfg) -> (key, ladder, per-dose witness outcomes); pure."""
    context, key, cfg = job
    ladder = build_ladder(context, key, cfg)
    outcomes = tuple(None if isinstance(item, PlacementError) else _witness_item(item) for item in ladder)
    return key, ladder, outcomes


def admit_grid(jobs, joint=True, workers=1, group_by=None):
    """Build AND admit a grid offline: jobs = iterable of (TaskContext, EpisodeKey, cfg).

    Equivalent to admit_ladders([(key, build_ladder(context, key, cfg)), ...],
    joint), with ladder construction also distributed over `workers` spawn
    processes (one job per key). The returned pool is identical to the
    serial result (tests compare digests, rejections and assignments).
    group_by: optional function job -> hashable group label (e.g. (vessel,
    task)); then every job still runs in ONE process pool and a dict
    {label: AdmissionPool} is returned, each pool assembled from its own jobs
    in input order (keys need only be unique within a group).
    """
    jobs = list(jobs)
    labels = [None if group_by is None else group_by(job) for job in jobs]
    seen = set()
    for label, (_, key, _) in zip(labels, jobs):
        if (label, key) in seen:
            raise ValueError("duplicate episode key")
        seen.add((label, key))
    grouped = {}
    for label, (key, ladder, outcomes) in zip(labels, _map_ordered(_build_and_witness, jobs, workers)):
        if len(ladder) != 5:
            raise ValueError("a ladder has exactly five doses")
        results = []
        for dose, (item, outcome) in enumerate(zip(ladder, outcomes)):
            if isinstance(item, NotApplicableByDesign):          # G23e
                results.append((dose, _NA, None, item.reason, 0))
                continue
            if isinstance(item, PlacementError):
                results.append((dose, None, None, item.reason, 0))
                continue
            witness, reason, sampled = outcome
            results.append((dose, item if witness is not None else None, witness, reason, sampled))
        grouped.setdefault(label, []).append((key, results))
    if group_by is None:
        return _assemble(grouped.get(None, []), joint)
    return {label: _assemble(witnessed, joint) for label, witnessed in grouped.items()}


def _assemble(witnessed, joint):
    """Deterministic pool assembly from per-key witnessed results (input order)."""
    offered, accepted, rejections, not_applicable = [], {}, [], []
    per_dose = [[0, 0] for _ in range(5)]   # [offered, admitted] per dose
    for key, results in witnessed:
        offered.append(key)
        for dose in range(len(results)):
            if results[dose][1] is not _NA:          # G23e: a not-applicable tier is never offered
                per_dose[dose][0] += 1
        for dose, item, _, reason, sampled in results:
            if item is _NA:
                not_applicable.append(Rejection(key, dose, reason, sampled))
        failed = [r for r in results if r[1] is None]
        for dose, _, _, reason, sampled in failed:
            rejections.append(Rejection(key, dose, reason, sampled))
        if joint and failed:
            for dose, item, _, _, sampled in results:
                if item is not None and item is not _NA:
                    rejections.append(Rejection(key, dose, "joint_ladder_member_rejected", sampled))
            continue
        for dose, item, witness, _, _ in results:
            if item is not None and item is not _NA:
                accepted[(key, dose)] = (item, witness)
                per_dose[dose][1] += 1
    manifest = {"offered": offered, "accepted": sorted(
        [[k.eval_seed, k.env, k.episode, k.hazard_group, d, digest(c), _witness_digest(w)]
         for (k, d), (c, w) in accepted.items()]), "rejections": rejections, "joint": joint}
    if not_applicable:                               # G23e: only pools with such tiers carry the field
        manifest["not_applicable"] = not_applicable
    pool_id = digest(manifest)
    assignments = {kd: Assignment(c, w, digest(c), _witness_digest(w), pool_id)
                   for kd, (c, w) in accepted.items()}
    keys_admitted = len({k for k, _ in accepted})
    if not_applicable:
        return AdmissionPoolNA(assignments, tuple(rejections), len(offered), keys_admitted, joint,
                               pool_id, tuple(tuple(x) for x in per_dose), tuple(not_applicable))
    return AdmissionPool(assignments, tuple(rejections), len(offered), keys_admitted, joint,
                         pool_id, tuple(tuple(x) for x in per_dose))


# ------------------------------------------------------------------- runtime
@dataclass(frozen=True)
class DynamicRuntimeConfig:
    enabled: bool = True
    verify_witness_on_reset: bool = False
    # G23b: runtime motion rule of the D7 intruder (release.py); "pinned_v0" is diagnostic only
    motion_rule: str = DEFAULT_RULE

    def __post_init__(self):
        if self.motion_rule not in RUNTIME_RULES:
            raise ValueError("unknown D7 runtime motion rule " + repr(self.motion_rule))


@dataclass(frozen=True)
class RuntimeFrame:
    times_s: tuple
    observers: tuple = ()


@dataclass(frozen=True)
class ObstacleSample:
    slot: int
    x: float
    y: float
    radius_m: float
    physical: bool
    visible: bool


@dataclass
class _Episode:
    assignment: Assignment
    cfg: DynamicRuntimeConfig
    armed: bool = False
    last_time_s: float = -1.0
    first_detection_s: object = None
    realized_ttc_at_detection_s: object = None
    detected_samples: int = 0
    observer_updates: int = 0
    min_surface_clearance_m: float = math.inf
    sampled_contact: bool = False
    finished: bool = False
    stamp: object = None
    release: object = None          # G23b release.ReleaseState (None: D0, moving basin)
    objects: tuple = ()             # G23b: the intruders as moved at runtime
    exposure: object = None         # G23c F4: exposure.ExposureTracker (None: D0, moving basin)


def assignment_summary(assignment):
    """Compact, JSON-native identity of an admitted assignment (for stamps)."""
    c = assignment.candidate
    e = c.request.encounter
    env = c.request.envelope
    intruders = []
    for obj in e.intruders:
        record = {"x0": obj.position(0.0)[0], "y0": obj.position(0.0)[1],
                  "radius_m": obj.radius_m, "speed_bound_mps": obj.speed_bound_mps}
        if hasattr(obj, "vx"):
            record.update({"path_type": obj.path_type, "vx": obj.vx, "vy": obj.vy,
                           "reveal_s": obj.reveal_s, "reveal_kind": obj.reveal_kind,
                           "start_s": obj.start_s, "stop_s": obj.stop_s})
        else:
            record.update({"native_kind": obj.kind})
        intruders.append(record)
    return {
        "key": {"eval_seed": e.key.eval_seed, "env": e.key.env, "episode": e.key.episode,
                "hazard_group": e.key.hazard_group},
        "task": c.task, "dose": e.dose, "site": c.site, "stage": c.stage,
        "geometry_id": c.geometry_id,
        "vessel_id": env.vessel_id, "envelope_provenance": env.provenance,
        "t_avoid_s": env.t_avoid_s, "t_avoid_method": env.t_avoid_method,
        "t_avoid_components": [list(p) for p in env.t_avoid_components],
        "footprint": [env.footprint.x_min_m, env.footprint.x_max_m,
                      env.footprint.y_min_m, env.footprint.y_max_m],
        "nominal_ttc_s": e.nominal_ttc_s, "encounter_metadata": [list(p) for p in e.metadata],
        "intruders": intruders, "perceivable": c.perceivable,
        "sensor": None if c.sensor is None else {"ray_count": c.sensor.ray_count,
                                                  "max_range_m": c.sensor.max_range_m},
        "sensor_source": c.sensor_source,
        "detection_range_m": None if c.reveal_evidence is None else c.reveal_evidence.detection_range_m,
        "settled_phase_start_s": c.settled_phase_start_s, "deadline_s": c.request.horizon_s,
        "capacity": {"base_max": c.capacity.base_max, "dynamic_reserve": c.capacity.dynamic_reserve,
                     "appearing_reserve": c.capacity.appearing_reserve,
                     "capacity": c.capacity.capacity},
        "candidate_digest": assignment.candidate_digest, "witness_digest": assignment.witness_digest,
        "witness_reason": assignment.witness.reason,
        "witness_min_clearance_m": assignment.witness.min_clearance_m,
        "pool_digest": assignment.pool_digest,
    }


class TaskAdapter:
    """Runtime adapter; one instance serves many env rows of ONE task."""

    task = ""

    def __init__(self):
        self.profile = PROFILES[self.task]
        self.episodes = {}
        self.completed = []

    # Offline builders, bound to this task's profile.
    def build_ladder(self, context, key, cfg):
        if context.task != self.task:
            raise ValueError("context belongs to another task")
        return build_ladder(context, key, cfg)

    def reset(self, env_ids, cfg, scenario):
        ids = tuple(int(i) for i in env_ids)
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("reset ids must be unique nonnegative rows")
        staged = {}
        for env in ids:
            if env in self.episodes and not self.episodes[env].finished:
                raise ValueError("finish and stamp every assigned episode before reset")
            assignment = scenario[env]
            candidate = assignment.candidate
            if candidate.task != self.task or candidate.request.encounter.key.env != env:
                raise ValueError("wrong task or environment assignment")
            if not assignment.witness.accepted or assignment.candidate_digest != digest(candidate):
                raise ValueError("unadmitted or modified assignment")
            if assignment.witness_digest != _witness_digest(assignment.witness):
                raise ValueError("modified witness")
            candidate.validate()
            if cfg.verify_witness_on_reset and not replay_witness(
                    candidate.request, assignment.witness.trajectory).accepted:
                raise ValueError("assignment witness failed replay")
            if not cfg.enabled and candidate.request.encounter.dose != 0 and self.task != "moving_basin":
                raise ValueError("a disabled runtime requires a D0 assignment")
            episode = _Episode(assignment, cfg)
            encounter = candidate.request.encounter
            episode.objects = tuple(encounter.intruders)
            if self.task != "moving_basin" and encounter.dose > 0:
                if len(encounter.intruders) != 1:       # G23c F13: the runtime rules move ONE intruder
                    raise ValueError("a D7 runtime rule needs exactly one intruder, got %d"
                                     % len(encounter.intruders))
                # G23e: station keeping's approach encounter is timed like a transit route (plan kind "approach")
                timing = (sk_approach.SK_PLAN_KIND if self.profile.site_rule == sk_approach.SK_SITE_RULE
                          else self.profile.plan_kind)
                if cfg.motion_rule in (REVEAL_RULE_VERSION, HYBRID_RULE_VERSION, V3_RULE_VERSION):   # reveal.py
                    episode.release = RevealState(candidate, timing,
                                                  hybrid=cfg.motion_rule == HYBRID_RULE_VERSION,
                                                  v3=cfg.motion_rule == V3_RULE_VERSION)
                else:
                    episode.release = ReleaseState(candidate, timing, cfg.motion_rule)
                episode.objects = (ReleasedIntruder(encounter.intruders[0], episode.release),)
                episode.exposure = ExposureTracker(candidate, episode.release)
            staged[env] = episode
        self.episodes.update(staged)

    def activate(self, env_ids, cfg):
        """Arm at reset. Calling it earlier or later never retimes the world (the runtime rule's own
        timing, release.py, depends only on the update times and observer poses)."""
        for env in env_ids:
            episode = self.episodes[int(env)]
            if cfg != episode.cfg or episode.finished:
                raise ValueError("runtime config changed or episode already finished")
            episode.armed = True

    def _slots(self, candidate):
        if self.task == "moving_basin":
            return candidate.base_slots
        return candidate.capacity.dynamic_slots

    def update(self, progress, cfg):
        if not isinstance(progress, RuntimeFrame):
            raise TypeError("D7 update requires an episode-time RuntimeFrame, never route progress")
        times, observers = dict(progress.times_s), dict(progress.observers)
        if len(times) != len(progress.times_s) or len(observers) != len(progress.observers):
            raise ValueError("duplicate environment row in runtime frame")
        result = {}
        for env, time_s in times.items():
            episode = self.episodes[env]
            if cfg != episode.cfg or not episode.armed or episode.finished:
                raise ValueError("update needs an armed, unfinished episode with its own config")
            candidate = episode.assignment.candidate
            encounter = candidate.request.encounter
            if not math.isfinite(time_s) or time_s < 0 or time_s < episode.last_time_s:
                raise ValueError("episode clock must be finite and monotone")
            if time_s > encounter.duration_s + 1e-9:
                raise ValueError("runtime exceeded the frozen encounter deadline")
            observer = observers.get(env)
            if observer is not None and not isinstance(observer, Pose):
                raise TypeError("observer must be a Pose (exposure accounting and the runtime rule's timing)")
            if observer is not None:
                episode.observer_updates += 1
            samples, detected_now = [], False
            footprint = candidate.request.envelope.footprint
            if episode.release is not None:     # G23b: release decision BEFORE positions are sampled
                episode.release.observe(time_s, observer, candidate.reveal_evidence, candidate.sensor)
            for slot, obj in zip(self._slots(candidate), episode.objects):
                x, y = obj.position(time_s)
                visible = False
                if observer is not None and candidate.sensor is not None:
                    if self.task == "moving_basin":
                        edge = math.hypot(x - observer.x, y - observer.y) - obj.radius_m
                        visible = edge <= candidate.sensor.max_range_m
                    elif episode.release is not None:          # G23c L2: the rule decides (S2 hides)
                        visible = episode.release.detect(observer, obj, time_s, candidate.reveal_evidence,
                                                         candidate.sensor)
                    else:
                        visible = detectable(observer, obj, time_s, candidate.reveal_evidence,
                                             candidate.sensor)
                present = episode.release is None or episode.release.present(time_s)
                if observer is not None and present:   # G23d H6: a hidden (S2 / v2) object is no sample
                    gap = point_clearance(x, y, observer, footprint) - obj.radius_m
                    episode.min_surface_clearance_m = min(episode.min_surface_clearance_m, gap)
                    episode.sampled_contact |= gap <= 0.0
                if episode.exposure is not None:
                    episode.exposure.update(time_s, observer, x, y, visible,
                                            episode.release.released_at(time_s))
                samples.append(ObstacleSample(slot, x, y, obj.radius_m, present, visible))
                if visible and encounter.dose > 0 and self.task != "moving_basin":
                    detected_now = True
                    if episode.first_detection_s is None:
                        episode.first_detection_s = time_s
                        ttc = ttc_linear(observer, footprint, obj, at_s=time_s)
                        episode.realized_ttc_at_detection_s = None if math.isinf(ttc) else ttc
            if detected_now:
                episode.detected_samples += 1
            episode.last_time_s = time_s
            result[env] = tuple(samples)
        return result

    def finish(self, env, termination_reason, extra=None):
        """Stamp the finished episode (call BEFORE the env resets the row).

        extra: JSON-native bridge record (swept contact latch etc.), stamped
        under runtime["bridge"] inside the same digest.
        """
        if termination_reason not in TERMINATION_REASONS:
            raise ValueError("explicit terminal reason required: " + repr(termination_reason))
        episode = self.episodes[env]
        if episode.finished:
            return episode.stamp
        candidate = episode.assignment.candidate
        encounter = candidate.request.encounter
        if self.task == "moving_basin":
            reason = "native_motion_axis"
        elif encounter.dose == 0:
            reason = "zero_dose"
        elif not candidate.perceivable:
            reason = "no_existing_range_sensor"
        elif episode.first_detection_s is not None:
            reason = None
        elif not episode.observer_updates:
            reason = "sensor_observations_missing"
        elif episode.release is not None and episode.release.unreleased():
            reason = episode.release.unreleased_reason()   # G23b not_released (rule S)
        elif episode.last_time_s < encounter.intruders[0].reveal_s:
            reason = "terminated_before_nominal_reveal"
        else:
            reason = "not_detected_at_sampled_times"
        runtime = {
            "assigned": True, "itt_count": 1, "env": env, "termination_reason": termination_reason,
            "last_time_s": max(0.0, episode.last_time_s),
            "first_detection_s": episode.first_detection_s,
            "realized_ttc_at_first_detection_s": episode.realized_ttc_at_detection_s,
            "realized_ttc_definition": "constant-velocity extrapolation along observer heading",
            "detected_samples": episode.detected_samples,
            "observer_updates": episode.observer_updates,
            "min_sampled_surface_clearance_m": (None if math.isinf(episode.min_surface_clearance_m)
                                                else episode.min_surface_clearance_m),
            "sampled_contact": episode.sampled_contact,
            "contact_note": "endpoint samples only; the env bridge latches swept contact",
            "non_exposure_reason": reason,
            "bridge": extra,
            # G23b runtime rule (release.py); G23c: no rule block where no rule applies (D0, moving basin,
            # F13), the realized-exposure record (F4) and metric A as a perception-timing statistic (F5)
            "runtime_rule": (None if episode.release is None else rule_block_for(episode.cfg.motion_rule)),
            "release": (None if episode.release is None else
                        episode.release.stamp(candidate.request.envelope.footprint)),
            "exposure_record": None if episode.exposure is None else episode.exposure.stamp(),
            "finite_ttc_at_first_detection": episode.realized_ttc_at_detection_s is not None,
            "realized_ttc_at_first_detection_over_t_avoid": (
                None if episode.realized_ttc_at_detection_s is None
                else episode.realized_ttc_at_detection_s / candidate.request.envelope.t_avoid_s),
        }
        episode.stamp = make_stamp(assignment_summary(episode.assignment), runtime,
                                   None if episode.release is None else semantics_for(episode.cfg.motion_rule))
        episode.finished = True
        self.completed.append(episode.stamp)
        return episode.stamp


def _adapter_class(task_name):
    return type("".join(part.capitalize() for part in task_name.split("_")) + "Adapter",
                (TaskAdapter,), {"task": task_name,
                                 "__doc__": "D7 adapter for %s: %s" % (task_name,
                                                                       PROFILES[task_name].real_world)})


ADAPTERS = {name: _adapter_class(name) for name in TASKS}
# D4 names the composite task "harbour_mission" (D4 INTEGRATION_SPEC.md:265).
TASK_ALIASES = {"harbour_mission": "harbour"}


def make_adapter(task):
    return ADAPTERS[TASK_ALIASES.get(task, task)]()
