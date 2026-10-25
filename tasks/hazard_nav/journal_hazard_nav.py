"""Journal (task contract v2) glue of the HazardNav family (F1, CONTEXT_R4).

CPU importable: numpy + torch only, never Isaac Lab. The HazardNav env
(hazard_nav_env.py) calls this module ONLY from its journal branch
(``cfg.journal_contract_version == 2``); every legacy id is untouched.

Seven journal tasks ride this carrier (journal_cfg_v2.TASK_TABLE carrier
"hazard_nav"): reach (open water, single target), scatter, ring (sealed, D4
rule), fortress (two closed bands, staggered gates), forced_crossing,
iceberg and moving_basin (native motion through the D7 bridge, D7 N5/N8).

Contents
  * scenario: ``build_scenario(task, vessel, tier, seed, env, episode)`` -> an
    admitted E5 layout (journal_layouts_v2.build_layout) keyed by the scenario
    protocol, turned by a keyed QUARTER-TURN rotation (exact in floating point,
    so the axis-aligned layout bounds stay exact for the out-of-bounds rule and
    for the D7 witness), plus the moving-basin NativeMotion templates.
  * ``d7_context(scenario)``: the D7 TaskContext built from the REAL journal
    layout (shared by the env and the offline admission tool, so both see the
    same frozen geometry, D7 B05).
  * ``family_hashes(scenario)``: D11 family primitives (layout, rotation,
    spawn_pose, and basin_geometry for the moving basin).
  * D7 pool JSON codec + fail-closed loader (``load_d7_pool``); pools hold, per
    offered key, the admitted five-dose ladder OR an explicit not-admitted record
    with E4's rejection reasons (``HazardD7Pool``, CONTEXT_R4 D16). A key beyond
    the pool's offered grid (same seed and hazard group, env inside the grid,
    episode index >= the grid's episodes per env) is not admitted with the single
    reason "beyond_offered_grid" (CONTEXT_R4 D24); it is never a pool key.
  * ``NotAdmittedBridge``: the E4 HazardBridge plus D16 not-admitted rows (the key
    runs at D0, no intruder; stamp ``payload.exposure = "not_admitted"``).
  * ``ReachEpisodeTensor``: vectorised torch twin of the scalar E5 oracle
    ``journal_contracts_v2.ReachEpisode`` (every goal-disk task of this family
    uses it); bit-for-bit parity is a CPU test (scalar = reference).
  * torch geometry: exact CPython ``math.hypot`` twin, COM-polygon clearance
    (E5 vessel collision footprint, C1 finding 35) and its swept lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .._shared import journal_cfg_v2 as plans
from .._shared import journal_contracts_v2 as contracts
from .._shared import journal_layouts_v2 as layouts
from .._shared import scenario_rng
from .._shared import vehicles_v2

FAMILY = "hazard_nav"
CARRIED_TASKS = tuple(t for t in plans.TASKS if plans.TASK_TABLE[t]["carrier"] == FAMILY)
SCENARIO_SCHEMA = "usvbench.f1.hazard_scenario/1"
LAYOUT_GROUP = "journal_v2_layout."        # + task (scenario protocol group name)
ROTATION_GROUP = "journal_v2_rotation."    # + task
FAMILY_PRIMITIVES = ("layout", "rotation", "spawn_pose")
BASIN_PRIMITIVE = "basin_geometry"
POOL_SCHEMA = "usvbench.f1.d7_pool/2"      # /2: explicit not-admitted records (CONTEXT_R4 D16)
NOT_ADMITTED = "not_admitted"
PER_DOSE_COUNT_FIELDS = ("offered", "admitted", "not_admitted")
D16_RULE = ("CONTEXT_R4 D16: a key E4 does not admit keeps its E5 layout and runs the dynamic suite at D0 "
            "(no intruder; the moving basin frozen at its time-zero positions); never a redraw, never a "
            "skipped scenario")
BEYOND_OFFERED_GRID = "beyond_offered_grid"
BEYOND_GRID_REASONS = ((0, BEYOND_OFFERED_GRID, 0),)   # (dose, reason, sampled_states): one entry, dose 0
D24_RULE = ("CONTEXT_R4 D24: a key with the pool's eval seed and hazard group, env inside the offered grid and "
            "episode index >= the grid's episodes per env is not admitted with the single reason "
            "'beyond_offered_grid': it keeps its E5 layout, runs the dynamic suite at D0 (no intruder; the "
            "moving basin frozen at its time-zero positions) and is never counted (not a pool key)")
POOL_DIR = Path(__file__).resolve().parent / "d7_pools"
LAYOUT_MAX_ATTEMPTS = 200

_TWO_PI = 2.0 * math.pi


def _dynamic():
    from .._shared.conditions import dynamic
    return dynamic


def _pairing():
    from .._shared.conditions import pairing
    return pairing


# --------------------------------------------------------------------------
# quarter-turn rotation (exact)
# --------------------------------------------------------------------------

def quarter_turn_xy(xy, k):
    """Rotate XY points by k * 90 deg exactly (sign flips and swaps only)."""
    a = np.asarray(xy, dtype=np.float64)
    x, y = a[..., 0], a[..., 1]
    k = int(k) % 4
    if k == 0:
        out = np.stack((x, y), -1)
    elif k == 1:
        out = np.stack((-y, x), -1)
    elif k == 2:
        out = np.stack((-x, -y), -1)
    else:
        out = np.stack((y, -x), -1)
    return out + 0.0          # normalises -0.0 to +0.0 (no value change)


def quarter_turn_bounds(bounds, k):
    """(x0, x1, y0, y1) of the rotated axis-aligned box (exact)."""
    x0, x1, y0, y1 = (float(b) for b in bounds)
    k = int(k) % 4
    if k == 0:
        out = (x0, x1, y0, y1)
    elif k == 1:
        out = (-y1, -y0, x0, x1)
    elif k == 2:
        out = (-x1, -x0, -y1, -y0)
    else:
        out = (y0, y1, -x1, -x0)
    return tuple(v + 0.0 for v in out)


def wrap_angle(angle):
    """(-pi, pi] representative (same rule as the E4 kinematics wrap)."""
    a = math.atan2(math.sin(angle), math.cos(angle))
    return a if a != -math.pi else math.pi


def scenario_generator(seed, env, episode, group):
    """numpy Generator of the scenario protocol key (seed, env, episode, group)."""
    return scenario_rng.numpy_generator(scenario_rng.SCENARIO_PROTOCOL_VERSION, int(seed), int(env),
                                        int(episode), group)


def _sha(name, content):
    return _pairing().primitive_hash(name, content)


# --------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HazardScenario:
    """One admitted journal episode scenario, env-local metres (COM frame).

    ``centers/radii`` index = storage slot. ``static_slots`` are written into
    ``obstacle_active``; ``native_slots`` (moving basin only) are D7-owned
    base slots driven by the bridge (D7 B04/B07: D7 never writes
    ``obstacle_active``). The D7 intruder slot of the other tasks is
    ``d7_slot`` (= base maximum, D7 CapacityPlan.dynamic_slots).
    """

    task: str
    vessel: str
    tier: int
    seed: int
    env: int
    episode: int
    layout: object
    quarter_turns: int
    start_xy: tuple
    start_yaw: float
    goal_xy: tuple
    goal_radius_m: float
    centers: np.ndarray
    radii: np.ndarray
    static_slots: tuple
    native_slots: tuple
    native_templates: tuple
    bounds: tuple
    route: tuple                 # (x, y, radius, mandatory) D7 route circles
    deadline_s: float
    deadline_steps: int
    nominal_route_length_m: float
    reference_axis_rad: float
    geometry_id: str
    site_index: int
    d7_slot: int | None
    storage_width: int

    @property
    def obstacle_count(self):
        return int(self.radii.shape[0])

    def summary(self):
        """JSON-native scenario record (certificates / evaluator stamp)."""
        return {"schema": SCENARIO_SCHEMA, "task": self.task, "vessel": self.vessel, "tier": self.tier,
                "seed": self.seed, "env": self.env, "episode": self.episode,
                "layout_geometry_hash": self.layout.geometry_hash, "geometry_id": self.geometry_id,
                "quarter_turns": self.quarter_turns, "start_xy_m": list(self.start_xy),
                "start_yaw_rad": self.start_yaw, "goal_xy_m": list(self.goal_xy),
                "goal_radius_m": self.goal_radius_m, "obstacle_count": self.obstacle_count,
                "static_slots": len(self.static_slots), "native_slots": len(self.native_slots),
                "bounds_m": list(self.bounds), "deadline_s": self.deadline_s,
                "deadline_steps": self.deadline_steps,
                "nominal_route_length_m": self.nominal_route_length_m,
                "reference_axis_rad": self.reference_axis_rad,
                "layout_attempts": int(self.layout.attempts),
                "layout_rejections": dict(self.layout.rejections)}


def storage_width(task):
    """Obstacle storage slots of a journal task: base maximum + D7 reserve (D7 B04/N1).

    E5's template already folds the reserve into cfg.max_obstacles
    (journal_runtime_cfg_v2._journal_post_init), so the env allocates
    exactly cfg.max_obstacles and checks it equals this value.
    """
    return max(1, int(plans.STATIC_CAPACITY[task]) + int(plans.D7_RESERVED_SLOTS[task]))


def _native_templates(layout, k, radii):
    dyn = _dynamic()
    moving = layout.extras["moving"]
    kinds, params = moving["kind"], moving["params"]
    out = []
    for slot in moving["base_slots"]:
        kind, p = int(kinds[slot]), tuple(float(v) for v in params[slot])
        r = float(radii[slot])
        if kind == layouts.KIND_TRANSLATE:
            cx, cy = quarter_turn_xy((p[0], p[1]), k)
            dx, dy = quarter_turn_xy((p[2], p[3]), k)
            out.append(dyn.NativeMotion("translate", float(cx), float(cy), r, 0.0, dx=float(dx), dy=float(dy),
                                        span_m=p[6], phase_m=p[4]))
        elif kind == layouts.KIND_ORBIT:
            cx, cy = quarter_turn_xy((p[0], p[1]), k)
            out.append(dyn.NativeMotion("orbit", float(cx), float(cy), r, 0.0, orbit_radius_m=p[2],
                                        phase_rad=p[4] + k * (math.pi / 2.0)))
        else:
            raise ValueError("moving-basin layout carries a static base slot among its native slots")
    return tuple(out)


def build_scenario(task, vessel, tier, seed, env, episode, *, max_attempts=LAYOUT_MAX_ATTEMPTS):
    """Admitted, rotated journal scenario for one (seed, env, episode) key.

    The layout draws only from the scenario-protocol stream
    ``journal_v2_layout.<task>`` of this key and the quarter turn from
    ``journal_v2_rotation.<task>``, so a key names the same scenario for every
    controller, dose and process (controller-independent, D7 B03). The moving
    basin is generated at the default native dose (build_layout admits the
    geometry JOINTLY over the whole mu ladder, so the geometry is identical at
    every dose; the D7 dose only sets the speeds, D7 N8).
    Raises journal_layouts_v2.JournalAdmissionError when no admissible layout
    exists (a preflight error, never an episode outcome).
    """
    if task not in CARRIED_TASKS:
        raise ValueError("task %r is not carried by the HazardNav family" % (task,))
    for name, value in (("seed", seed), ("env", env), ("episode", episode), ("tier", tier)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    if int(env) < 0 or int(episode) < 0:
        raise ValueError("env and episode must be nonnegative")
    lay = layouts.build_layout(task, int(tier), vessel,
                               scenario_generator(seed, env, episode, LAYOUT_GROUP + task),
                               max_attempts=max_attempts)
    k = int(scenario_generator(seed, env, episode, ROTATION_GROUP + task).integers(4))
    width = storage_width(task)
    n = lay.obstacle_count
    if n > int(plans.STATIC_CAPACITY[task]) or (task != "moving_basin" and n >= width):
        raise RuntimeError("layout of %d obstacles exceeds the %s base capacity %d"
                           % (n, task, plans.STATIC_CAPACITY[task]))
    centers = quarter_turn_xy(lay.centers, k).reshape(-1, 2) if n else np.zeros((0, 2))
    radii = np.array(lay.radii, dtype=np.float64)
    start = quarter_turn_xy(lay.start, k)
    goal = quarter_turn_xy(lay.goal, k)
    bounds = quarter_turn_bounds(lay.bounds_m, k)
    start_yaw = wrap_angle(float(lay.start_heading_rad) + k * (math.pi / 2.0))
    native = tuple(int(s) for s in lay.extras["moving"]["base_slots"]) if task == "moving_basin" else ()
    static = tuple(i for i in range(n) if i not in set(native))
    templates = _native_templates(lay, k, radii) if task == "moving_basin" else ()
    route, site_index = d7_route(lay, k, task)
    axis_vec = goal - start
    axis = math.atan2(float(axis_vec[1]), float(axis_vec[0]))
    geometry_id = hashlib.sha256(json.dumps(
        {"schema": SCENARIO_SCHEMA, "layout": lay.geometry_hash, "quarter_turns": k},
        sort_keys=True).encode("ascii")).hexdigest()
    for arr in (centers, radii):
        arr.setflags(write=False)
    return HazardScenario(
        task=task, vessel=vessel, tier=int(tier), seed=int(seed), env=int(env), episode=int(episode),
        layout=lay, quarter_turns=k, start_xy=(float(start[0]), float(start[1])), start_yaw=start_yaw,
        goal_xy=(float(goal[0]), float(goal[1])), goal_radius_m=float(lay.params["goal_radius_m"]),
        centers=centers, radii=radii, static_slots=static, native_slots=native, native_templates=templates,
        bounds=bounds, route=tuple(route), deadline_s=float(lay.deadline_s),
        deadline_steps=int(lay.deadline_steps), nominal_route_length_m=float(lay.nominal_route_length_m),
        reference_axis_rad=axis, geometry_id=geometry_id, site_index=int(site_index),
        d7_slot=None if task == "moving_basin" else int(plans.STATIC_CAPACITY[task]),
        storage_width=width)


WITNESS_ROUTE_TASKS = ("scatter", "ring", "fortress", "forced_crossing", "iceberg")
# Witness vertices that are MANDATORY D7 waypoints: the fortress path runs
# around the annulus between the staggered gates, and E4's witness beam search
# orders candidates by straight-line distance to the next waypoint
# (feasibility._score), so without intermediate waypoints it pushes into the
# inner band (every fortress key was "no_witness_in_frozen_search" in the F1
# probe). Extra mandatory waypoints only make the certificate stricter.
GUIDED_WITNESS_TASKS = ("fortress",)
# The last APPROACH_WINDOW_FACTOR x (D1 TTC ratio x t_avoid x reference speed)
# metres before the encounter gate carry no guidance vertex, so the nominal
# plan's final approach can host the unforced D1 window (adapters._fit_conflict).
APPROACH_WINDOW_FACTOR = 1.25


def d7_route(layout, k, task):
    """(route circles, site_index) of the D7 nominal plan for one layout.

    Circles are (x, y, radius, mandatory), env-local after the quarter turn.
    Compulsory gates (ring, fortress, forced crossing) are MANDATORY circles
    of radius half the planned hull-centre width (D4 d7_hook). The nominal
    plan follows the layout's own certified witness polyline
    (journal_geometry_v2.route_witness, curvature-bounded, hull-swept): its
    interior vertices are NON-mandatory guidance circles merged with the gates
    in arc-length order (a vertex within one planning radius of a gate
    midpoint is the gate itself). Without them the nominal plan runs straight
    between gates / to the goal THROUGH the frozen geometry (fortress inner
    band, the iceberg blocker), so no candidate of those tasks could ever be
    admitted (measured in the F1 admission probe). Encounter site (E4
    TaskProfile rules): the first compulsory gate for ring / fortress / forced
    crossing (waypoint_arrival), the first witness vertex for the iceberg side
    passage; scatter uses route_fraction (site index unused); reach and the
    moving basin keep an empty route.
    """
    half = 0.5 * float(layout.params["planned_centre_width_m"])
    gates = [g for g in layout.gates if g.kind == "compulsory"]
    if task not in WITNESS_ROUTE_TASKS:
        route = [(g.midpoint_m, 0.5 * float(g.planned_centre_width_m), True) for g in gates]
        return _turned(route, k), 0
    vertices = np.asarray(layout.witness.vertices_m, dtype=np.float64)
    seg = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))

    def arc(point):
        p = np.asarray(point, dtype=np.float64)
        best = (math.inf, 0.0)
        for i in range(len(seg)):
            n2 = float(seg[i] @ seg[i])
            t = 0.0 if n2 == 0.0 else min(1.0, max(0.0, float((p - vertices[i]) @ seg[i]) / n2))
            d = float(np.linalg.norm(vertices[i] + t * seg[i] - p))
            if d < best[0]:
                best = (d, float(cumulative[i] + t * lengths[i]))
        return best[1]

    entries = [(arc(g.midpoint_m), 0, (g.midpoint_m, 0.5 * float(g.planned_centre_width_m), True)) for g in gates]
    guide_mandatory = task in GUIDED_WITNESS_TASKS
    approach_until = None
    if gates and task in ("ring", "fortress", "forced_crossing"):
        envelope = vessel_envelope(layout.vessel)
        window_m = _dynamic().DOSE_TTC_RATIOS[1] * envelope.t_avoid_s * envelope.reference_speed_mps
        first_gate = min(e[0] for e in entries)
        approach_until = (first_gate - APPROACH_WINDOW_FACTOR * window_m, first_gate)
    for i in range(1, len(vertices) - 1):
        v = vertices[i]
        if any(np.linalg.norm(v - np.asarray(g.midpoint_m)) <= 0.5 * float(g.planned_centre_width_m) + half
               for g in gates):
            continue
        s_v = float(cumulative[i])
        if approach_until is not None and approach_until[0] < s_v < approach_until[1]:
            continue          # keep the final approach to the encounter gate straight (D1 window)
        entries.append((s_v, 1, (tuple(v), half, guide_mandatory)))
    entries.sort(key=lambda e: (e[0], e[1]))
    route = [e[2] for e in entries]
    if task in ("ring", "fortress", "forced_crossing"):
        site = next((i for i, r in enumerate(route) if r[2]), 0)
    else:
        site = 0
    return _turned(route, k), site


def _turned(route, k):
    out = []
    for point, radius, mandatory in route:
        x, y = quarter_turn_xy(point, k)
        out.append((float(x), float(y), float(radius), bool(mandatory)))
    return tuple(out)


_ENVELOPES = {}


def vessel_envelope(vessel):
    """D7 VesselEnvelope of a journal vessel (same construction as journal_env_mixin)."""
    if vessel not in _ENVELOPES:
        dyn = _dynamic()
        rec = plans.vessel_record(vessel)
        _ENVELOPES[vessel] = dyn.envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel),
                                                       reference_speed_mps=rec.cruise_speed_m_s,
                                                       turning_radius_m=rec.turning_radius_design_m)[0]
    return _ENVELOPES[vessel]


def d7_context(scenario):
    """D7 TaskContext from the real journal scenario (env-local, rotated; D7 B05).

    Initial pose = the spawn pose AT REST (the env spawns with zero velocity).
    Sensor = the task plan's 36-ray fan (30 m x length scale), declared as the
    task's existing sensor. Moving basin: the walls are static obstacles, the
    rows and the orbiter are NativeMotion templates in their storage slots.
    """
    dyn = _dynamic()
    params = scenario.layout.params
    static = tuple(dyn.Circle(float(scenario.centers[i, 0]), float(scenario.centers[i, 1]),
                              float(scenario.radii[i])) for i in scenario.static_slots)
    route = tuple(dyn.Circle(x, y, r) for x, y, r, _ in scenario.route)
    mandatory = tuple(bool(m) for _, _, _, m in scenario.route)
    sensor = dyn.SensorSpec(kind="ray_fan", ray_count=int(params["ray_sensor_ray_count"]),
                            max_range_m=float(params["ray_sensor_max_range_m"]),
                            source="journal 36-ray fan, 30 m x length scale (journal_cfg_v2; F1 ray block)")
    basin = scenario.task == "moving_basin"
    return dyn.TaskContext(
        task=scenario.task, envelope=vessel_envelope(scenario.vessel),
        initial_pose=dyn.Pose(float(scenario.start_xy[0]), float(scenario.start_xy[1]),
                              float(scenario.start_yaw), 0.0),
        bounds=tuple(float(b) for b in scenario.bounds),
        goal=dyn.Circle(float(scenario.goal_xy[0]), float(scenario.goal_xy[1]), float(scenario.goal_radius_m)),
        deadline_s=float(scenario.deadline_s), route=route, route_mandatory=mandatory,
        static_obstacles=static, sensor=sensor, site_index=int(scenario.site_index),
        native_geometry=scenario.native_templates if basin else (),
        base_max=int(plans.STATIC_CAPACITY[scenario.task]), base_count=scenario.obstacle_count,
        base_slots=scenario.native_slots if basin else (), geometry_id=scenario.geometry_id)


def appearing_admission_inputs(task, vessel, tier, seed, env, episode):
    """G3: (D7 TaskContext, length scale, radius range or None) of one key's REAL journal scenario.

    The appearing admission places its cylinder on the same frozen geometry the
    env builds (D16). Rule 4: scatter has its own static-obstacle radius rule
    (journal_cfg_v2 radius_range_m), every other task uses 0.8 m x length scale.
    G20d: fortress returns a 4th item {"fortress": appearing_fortress_region(...)} (gate openings and the B17
    gate gap of the journal tier) for the fortress radius rule.
    """
    scenario = build_scenario(task, vessel, int(tier), int(seed), int(env), int(episode))
    params = scenario.layout.params
    radius_range = tuple(float(v) for v in params["radius_range_m"]) if task == "scatter" else None
    context = d7_context(scenario)
    if task == "moving_basin":
        context = appearing_basin_context(scenario, context)
    elif task in APPEARING_GATE_ALIGNED_TASKS:
        context = appearing_gate_context(scenario, context)
    if task == "fortress":
        return context, float(params["length_scale"]), radius_range, {"fortress": appearing_fortress_region(scenario)}
    return context, float(params["length_scale"]), radius_range


def appearing_fortress_region(scenario):
    """G20d (owner decision 2026-10-04, fortress radius rule): every compulsory gate opening as the segment between
    its two jamb SURFACES (env-local, after the quarter turn: scenario.centers / radii are the layout's storage slots
    in layout order, so GateRecord.jamb_indices index them), and the passage the rule keeps: the B17 gate gap of the
    journal tier, g = beam + c_k s (journal_geometry_v2.physical_tier; tier 2: c = 2.0 m; the gap every fortress
    gate itself has: GateRecord.physical_gap_m)."""
    from .._shared import journal_geometry_v2 as geo
    tier = geo.physical_tier(int(scenario.layout.params["tier"]), plans.vessel_record(scenario.vessel).geometry)
    openings = []
    for gate in scenario.layout.gates:
        if gate.kind != "compulsory":
            continue
        a, b = (int(i) for i in gate.jamb_indices)
        xa, ya, ra = float(scenario.centers[a, 0]), float(scenario.centers[a, 1]), float(scenario.radii[a])
        xb, yb, rb = float(scenario.centers[b, 0]), float(scenario.centers[b, 1]), float(scenario.radii[b])
        d = math.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / d, (yb - ya) / d
        openings.append({"name": str(gate.name), "segment": [[xa + ra * ux, ya + ra * uy], [xb - rb * ux, yb - rb * uy]],
                         "physical_gap_m": float(gate.physical_gap_m)})
    return {"passage_gap_m": float(tier.physical_surface_gap_m), "gate_openings": openings,
            "source": "journal_geometry_v2.physical_tier(%d): beam %.6g + c_k s %.6g (B17)"
                      % (tier.level, plans.vessel_record(scenario.vessel).geometry.beam_m, tier.hull_centre_width_m)}


# G20: tasks whose appearing view adds gate-alignment guidance circles (see appearing_gate_context)
APPEARING_GATE_ALIGNED_TASKS = ("ring", "fortress")   # ring: BlueBoat 29 -> 30/32, WAM-V 29 -> 32/32 admitted
APPEARING_GATE_ALIGN_LENGTHS = 2.0      # guidance circles two hull lengths before / after each compulsory gate


def appearing_gate_context(scenario, context):
    """G20 (ring, fortress): the appearing admission's nominal route crosses every compulsory gate square-on.

    Measured (G20 evidence fortress_prefix_bb.json): on 17 of 32 seed-42 BlueBoat keys the D7 nominal plan
    grazes the outer-gate jambs (footprint clearance <= the 0.02 m witness clearance at 0.18-0.26 of the
    route), so every appearing site after that gate was "nominal_prefix_unsafe" (BlueBoat 7/32 admitted;
    with 1.0 hull length 16/32; with 2.0 hull lengths no prefix rejection, 18/32 in [0.40, 0.60] and 25/32
    in [0.30, 0.70]; WAM-V 12/32 -> 23/32 and 29/32). The appearing view adds,
    for each compulsory gate on the route, a non-mandatory guidance circle APPEARING_GATE_ALIGN_LENGTHS (2.0) hull
    lengths before and after the gate midpoint along its passage normal (radius a quarter of the planned
    hull-centre width). Only the appearing admission's nominal plan changes; the D7 context, the layout and
    the gate certificates are untouched (D16).
    """
    dyn = _dynamic()
    k = scenario.quarter_turns
    offset = APPEARING_GATE_ALIGN_LENGTHS * context.envelope.footprint.length_m
    gates = []
    for g in scenario.layout.gates:
        if g.kind != "compulsory":
            continue
        mx, my = (float(v) for v in quarter_turn_xy(g.midpoint_m, k))
        nx, ny = (float(v) for v in quarter_turn_xy(g.through_m, k))
        gates.append((mx, my, nx, ny, 0.25 * float(g.planned_centre_width_m)))
    route, flags = [], []
    for circle, mandatory in zip(context.route, context.route_mandatory or (False,) * len(context.route)):
        match = next((g for g in gates if mandatory and abs(g[0] - circle.x) <= 1e-9 and abs(g[1] - circle.y) <= 1e-9),
                     None)
        if match is None:
            route.append(circle)
            flags.append(bool(mandatory))
            continue
        mx, my, nx, ny, radius = match
        route += [dyn.Circle(mx - offset * nx, my - offset * ny, radius), circle,
                  dyn.Circle(mx + offset * nx, my + offset * ny, radius)]
        flags += [False, True, False]
    site = next((i for i, keep in enumerate(flags) if keep), 0)
    return dyn.TaskContext(
        task=context.task, envelope=context.envelope, initial_pose=context.initial_pose, bounds=context.bounds,
        goal=context.goal, deadline_s=context.deadline_s, route=tuple(route), route_mandatory=tuple(flags),
        static_obstacles=context.static_obstacles, sensor=context.sensor, site_index=site,
        native_geometry=context.native_geometry, base_max=context.base_max, base_count=context.base_count,
        base_slots=context.base_slots, geometry_id=context.geometry_id)


def appearing_basin_context(scenario, context):
    """G20 (moving basin): the appearing admission's view of the basin WITHOUT the D7 suite.

    With the dynamic suite absent (or at D0) the E4 bridge runs the native movers at speed 0, i.e.
    frozen at their time-zero positions (adapters native_encounter, NATIVE_RATIOS[0] = 0; the D16
    rule says the same for not-admitted keys). The appearing witness therefore sees the frozen movers
    as static obstacles, and the nominal plan threads each frozen row through the surface gap nearest
    to the start-goal line (one non-mandatory guidance circle per row at the gap midpoint, radius a
    quarter of the hull-centre room of the gap): the D7 nominal plan of the basin is a straight line
    that runs through frozen row obstacles. The NativeMotion templates stay in native_geometry, so the
    placement keeps the cylinder clear of every mover's swept envelope (dose-independent). Stacked
    with dynamic=D1..D4 the movers move: the appearing certificate does not cover that stack.
    """
    dyn = _dynamic()
    frozen = tuple(dyn.Circle(*t.position(0.0), t.radius_m) for t in scenario.native_templates)
    sx, sy = scenario.start_xy
    gx, gy = scenario.goal_xy
    length = math.hypot(gx - sx, gy - sy)
    ux, uy = (gx - sx) / length, (gy - sy) / length
    beam = context.envelope.footprint.beam_m
    rows = {}
    for template, circle in zip(scenario.native_templates, frozen):
        if template.kind != "translate":
            continue
        along = (template.cx - sx) * ux + (template.cy - sy) * uy
        rows.setdefault(round(along, 6), []).append(circle)
    route = []
    for along in sorted(rows):
        members = rows[along]
        cross = sorted(((c.x - sx) * -uy + (c.y - sy) * ux, c.radius) for c in members)
        best = None
        for (a, ra), (b, rb) in zip(cross, cross[1:]):
            room = (b - rb) - (a + ra) - beam
            if room <= 0.0:
                continue
            mid = 0.5 * ((a + ra) + (b - rb))
            if best is None or abs(mid) < abs(best[0]):
                best = (mid, room)
        if best is None:
            continue
        mid, room = best
        route.append(dyn.Circle(sx + along * ux - mid * uy, sy + along * uy + mid * ux, 0.25 * room))
    return dyn.TaskContext(
        task=context.task, envelope=context.envelope, initial_pose=context.initial_pose, bounds=context.bounds,
        goal=context.goal, deadline_s=context.deadline_s, route=tuple(route),
        route_mandatory=tuple(False for _ in route), static_obstacles=context.static_obstacles + frozen,
        sensor=context.sensor, site_index=0, native_geometry=context.native_geometry,
        base_max=context.base_max, base_count=context.base_count, base_slots=context.base_slots,
        geometry_id=context.geometry_id)


def d7_key(scenario):
    return _dynamic().EpisodeKey(int(scenario.seed), int(scenario.env), int(scenario.episode),
                                 "d7_" + scenario.task)


def family_hashes(scenario, com_height_m):
    """D11 family primitives of one scenario (exogenous content only)."""
    lay = scenario.layout
    out = {
        "layout": _sha("layout", {"task": scenario.task, "vessel": scenario.vessel, "tier": scenario.tier,
                                  "geometry_hash": lay.geometry_hash, "deadline_steps": scenario.deadline_steps,
                                  "goal_radius_m": scenario.goal_radius_m}),
        "rotation": _sha("rotation", {"quarter_turns": scenario.quarter_turns}),
        "spawn_pose": _sha("spawn_pose", {"start_xy_m": list(scenario.start_xy),
                                          "start_yaw_rad": scenario.start_yaw,
                                          "com_height_above_surface_m": float(com_height_m),
                                          "speed_mps": 0.0}),
    }
    if scenario.task == "moving_basin":
        out[BASIN_PRIMITIVE] = _sha(BASIN_PRIMITIVE, {
            "native_slots": list(scenario.native_slots),
            "templates": [{"kind": t.kind, "cx": t.cx, "cy": t.cy, "radius_m": t.radius_m, "dx": t.dx, "dy": t.dy,
                           "span_m": t.span_m, "phase_m": t.phase_m, "orbit_radius_m": t.orbit_radius_m,
                           "phase_rad": t.phase_rad} for t in scenario.native_templates]})
    return out


# --------------------------------------------------------------------------
# D7 admission pool JSON codec (offline tool writes, env reads; fail closed)
# --------------------------------------------------------------------------

_POOL_TYPES = ("EpisodeKey", "Footprint", "VesselEnvelope", "Pose", "TimedState", "Circle", "Intruder",
               "Encounter", "WitnessRequest", "WitnessResult", "SensorSpec", "RevealEvidence", "NativeMotion",
               "CapacityPlan", "Candidate", "Rejection", "Assignment")


def _pool_classes():
    dyn = _dynamic()
    from .._shared.conditions.dynamic import feasibility
    table = {name: getattr(dyn, name) for name in _POOL_TYPES if hasattr(dyn, name)}
    table["WitnessRequest"] = feasibility.WitnessRequest
    table["WitnessResult"] = feasibility.WitnessResult
    missing = set(_POOL_TYPES) - set(table)
    if missing:
        raise RuntimeError("D7 pool codec cannot resolve types %s" % sorted(missing))
    return table


def pool_encode(value):
    """Tagged JSON-native encoding of D7 admission objects (exact floats)."""
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _POOL_TYPES:
            raise TypeError("type %s is not part of the D7 pool codec" % name)
        return {"__dc__": name,
                "f": {f.name: pool_encode(getattr(value, f.name)) for f in fields(value) if f.init},
                "x": {f.name: pool_encode(getattr(value, f.name)) for f in fields(value) if not f.init}}
    if isinstance(value, tuple):
        return [pool_encode(v) for v in value]
    if isinstance(value, list):
        return {"__list__": [pool_encode(v) for v in value]}
    if isinstance(value, dict):
        return {"__dict__": [[pool_encode(k), pool_encode(v)] for k, v in value.items()]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("cannot encode %r in a D7 pool" % (type(value).__name__,))


def pool_decode(value, classes=None):
    """Inverse of pool_encode; reconstructed dataclasses re-run their validation."""
    classes = _pool_classes() if classes is None else classes
    if isinstance(value, list):
        return tuple(pool_decode(v, classes) for v in value)
    if isinstance(value, dict):
        if "__dc__" in value:
            cls = classes[value["__dc__"]]
            obj = cls(**{k: pool_decode(v, classes) for k, v in value["f"].items()})
            for name, stored in value.get("x", {}).items():
                if pool_encode(getattr(obj, name)) != stored:
                    raise ValueError("D7 pool field %s.%s does not recompute" % (value["__dc__"], name))
            return obj
        if "__list__" in value:
            return [pool_decode(v, classes) for v in value["__list__"]]
        if "__dict__" in value:
            return {pool_decode(k, classes): pool_decode(v, classes) for k, v in value["__dict__"]}
        if "__float__" in value:
            return float(value["__float__"])
        raise ValueError("untagged object in a D7 pool file")
    return value


def pool_manifest_digest(offered, assignments, rejections, joint):
    """Recompute the E4 pool digest (adapters._assemble manifest) from decoded parts."""
    dyn = _dynamic()
    from .._shared.conditions.dynamic import adapters
    accepted = sorted([[k.eval_seed, k.env, k.episode, k.hazard_group, d, dyn.digest(a.candidate),
                        adapters._witness_digest(a.witness)] for (k, d), a in assignments.items()])
    return dyn.digest({"offered": list(offered), "accepted": accepted, "rejections": list(rejections),
                       "joint": joint})


def pool_document(pool, *, task, vessel, tier, seed, suite, offered, grid, provenance, geometry_ids=None):
    """JSON document of one admission pool (``pool_digest`` checked on read).

    Per offered key the document holds EITHER the admitted assignment ladder
    (all five doses, joint admission) OR an explicit ``not_admitted`` record
    with E4's rejection reasons and the geometry id of the E5 layout E4 saw
    (CONTEXT_R4 D16). ``geometry_ids``: {EpisodeKey: geometry_id} (optional).
    """
    records = not_admitted_records(pool, offered, geometry_ids)
    return {"schema": POOL_SCHEMA, "task": task, "vessel": vessel, "tier": int(tier), "seed": int(seed),
            "grid": grid, "suite": pool_encode_suite(suite), "joint": pool.joint,
            "pool_digest": pool.pool_digest, "keys_offered": pool.keys_offered,
            "keys_admitted": pool.keys_admitted, "keys_not_admitted": len(records),
            "per_dose_count_fields": list(PER_DOSE_COUNT_FIELDS),
            "per_dose_counts": [[int(o), int(a), int(o) - int(a)] for o, a in pool.per_dose_counts],
            "offered": pool_encode(tuple(offered)),
            "assignments": [[pool_encode(k), d, pool_encode(a)] for (k, d), a in sorted(
                pool.assignments.items(), key=lambda kv: (kv[0][0].env, kv[0][0].episode, kv[0][1]))],
            "not_admitted": [r.document() for r in records],
            "rejections": pool_encode(tuple(pool.rejections)), "provenance": provenance}


def pool_encode_suite(suite):
    from dataclasses import asdict
    return asdict(suite)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_pool_file(document, path):
    """gzip JSON (mtime 0, deterministic bytes); returns the file SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    with open(path, "wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    return file_sha256(path)


# G49 (J29, v2.6.5): one release pins the D7 pools of several evaluation seeds. The seed-42 pools keep their
# seedless v2.6.4 names and pins (MANIFEST.json "grid" / "pools"); the pool of any other seed is
# <task>__<vessel>__seed<seed>.json.gz, pinned in MANIFEST.json "seeds"["<seed>"] ({"grid", "pools"}).
LEGACY_POOL_SEED = 42


def pool_file_name(task, vessel, seed=None):
    """Pool file name of (task, vessel[, evaluation seed]); seed None or 42 -> the seedless v2.6.4 name (G49)."""
    if seed is None or int(seed) == LEGACY_POOL_SEED:
        return "%s__%s.json.gz" % (task, vessel)
    return "%s__%s__seed%d.json.gz" % (task, vessel, int(seed))


def pinned_pool_table(manifest, seed=None):
    """(pins {file name: entry}, grid) of one evaluation seed (G49): seed None or 42 -> the v2.6.4 top-level
    table, another seed -> MANIFEST.json "seeds"["<seed>"] (({}, None) when that seed has no pins)."""
    if seed is None or int(seed) == LEGACY_POOL_SEED:
        return manifest.get("pools", {}), manifest.get("grid")
    table = manifest.get("seeds", {}).get(str(int(seed)))
    if table is None:
        return {}, None
    return table.get("pools", {}), table.get("grid")


def pinned_pool_entry(manifest, name):
    """The pin of pool file ``name`` in any seed table (G49; names are unique across the tables), or None."""
    entry = manifest.get("pools", {}).get(name)
    if entry is not None:
        return entry
    for table in manifest.get("seeds", {}).values():
        entry = table.get("pools", {}).get(name)
        if entry is not None:
            return entry
    return None


def load_pool_manifest(pool_dir=POOL_DIR):
    path = Path(pool_dir) / "MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError("D7 pool manifest %s is missing (run the offline admission tool)" % path)
    return json.loads(path.read_text(encoding="ascii"))


def load_d7_pool(path, *, task, vessel, tier, seed, suite=None, expected_sha256=None, with_meta=False):
    """Load and VERIFY one pool file; returns a HazardD7Pool (E4 AdmissionPool + not-admitted keys).

    Fail closed on: file SHA-256 different from the expected digest (the
    manifest pin), schema/task/vessel/tier/seed mismatch, an admission suite
    different from the runtime suite (except enabled/dose), any candidate or
    witness digest that does not recompute, a pool digest that does not
    recompute from the decoded manifest, or not-admitted records / per-dose
    counts that do not recompute from the offered keys and E4's rejections.
    """
    dyn = _dynamic()
    from .._shared.conditions.dynamic import adapters
    path = Path(path)
    if expected_sha256 is None:
        manifest = load_pool_manifest(POOL_DIR)
        entry = pinned_pool_entry(manifest, path.name)                   # G49: any seed table
        if entry is None:
            raise ValueError("D7 pool %s is not pinned in %s/MANIFEST.json" % (path.name, POOL_DIR))
        expected_sha256 = entry["sha256"]
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError("D7 pool %s SHA-256 %s differs from the pinned %s" % (path, actual, expected_sha256))
    with gzip.open(path, "rb") as handle:
        doc = json.loads(handle.read().decode("ascii"))
    if doc.get("schema") != POOL_SCHEMA:
        raise ValueError("unsupported D7 pool schema %r" % (doc.get("schema"),))
    for name, want in (("task", task), ("vessel", vessel), ("tier", int(tier)), ("seed", int(seed))):
        if doc.get(name) != want:
            raise ValueError("D7 pool %s=%r, env needs %r" % (name, doc.get(name), want))
    if suite is not None:
        check_pool_suite(doc, suite)
    classes = _pool_classes()
    offered = pool_decode(doc["offered"], classes)
    rejections = pool_decode(doc["rejections"], classes)
    assignments = {}
    for key_raw, dose, assignment_raw in doc["assignments"]:
        key = pool_decode(key_raw, classes)
        assignment = pool_decode(assignment_raw, classes)
        if assignment.candidate_digest != dyn.digest(assignment.candidate):
            raise ValueError("D7 pool candidate digest does not recompute for %s D%d" % (key, dose))
        if assignment.witness_digest != adapters._witness_digest(assignment.witness):
            raise ValueError("D7 pool witness digest does not recompute for %s D%d" % (key, dose))
        if assignment.pool_digest != doc["pool_digest"]:
            raise ValueError("D7 pool assignment carries another pool digest")
        if assignment.candidate.request.encounter.key != key or assignment.candidate.request.encounter.dose != dose:
            raise ValueError("D7 pool assignment filed under the wrong key/dose")
        assignments[(key, int(dose))] = assignment
    digest = pool_manifest_digest(offered, assignments, rejections, doc["joint"])
    if digest != doc["pool_digest"]:
        raise ValueError("D7 pool digest does not recompute from its manifest")
    if doc.get("per_dose_count_fields") != list(PER_DOSE_COUNT_FIELDS):
        raise ValueError("D7 pool per-dose counts must be (offered, admitted, not_admitted) rows")
    per_dose = tuple((int(row[0]), int(row[1])) for row in doc["per_dose_counts"])
    admission = adapters.AdmissionPool(assignments, rejections, int(doc["keys_offered"]),
                                       int(doc["keys_admitted"]), bool(doc["joint"]), doc["pool_digest"],
                                       per_dose)
    if len(offered) != admission.keys_offered or len({k for k, _ in assignments}) != admission.keys_admitted:
        raise ValueError("D7 pool key counts do not match its manifest")
    stored = tuple(NotAdmitted.from_document(item, classes) for item in doc["not_admitted"])
    geometry = {r.key: r.geometry_id for r in stored}
    derived = tuple(not_admitted_records(admission, offered, geometry))
    if stored != derived:
        raise ValueError("D7 pool not-admitted records do not recompute from its offered keys and rejections")
    pool = HazardD7Pool(admission, offered, stored)
    if ([list(row) for row in pool.per_dose_counts] != doc["per_dose_counts"]
            or pool.keys_not_admitted != int(doc["keys_not_admitted"])):
        raise ValueError("D7 pool per-dose / not-admitted counts do not recompute")
    check_document_grid(doc.get("grid"), pool.grid)
    if not with_meta:
        return pool
    meta = {k: doc[k] for k in ("schema", "task", "vessel", "tier", "seed", "grid", "suite", "provenance")}
    meta["file_sha256"] = actual
    return pool, meta


def default_pool_path(task, vessel):
    return POOL_DIR / pool_file_name(task, vessel)


def pool_directory(cfg=None):
    """Directory holding the pool FILES: cfg.journal_d7_pool_dir, else the
    USVBENCH_D7_POOL_DIR environment variable, else tasks/hazard_nav/d7_pools.
    The SHA-256 pins always come from the in-repo tasks/hazard_nav/d7_pools/
    MANIFEST.json, so pointing elsewhere cannot swap an unpinned file in."""
    import os
    value = getattr(cfg, "journal_d7_pool_dir", None) if cfg is not None else None
    if not value:
        value = os.environ.get("USVBENCH_D7_POOL_DIR")
    return Path(value) if value else POOL_DIR


def resolve_d7_pool(cfg):
    """(pool, meta) for a journal cfg (fail closed).

    ``cfg.journal_d7_pool`` may be an admitted pool object (anything with
    ``assignment(key, dose)`` and ``pool_digest``, e.g. an in-process E4
    AdmissionPool; meta None) or a path; unset -> the pinned file of the
    (task, vessel) in pool_directory(cfg).
    """
    value = getattr(cfg, "journal_d7_pool", None)
    if value is not None and not isinstance(value, (str, Path)):
        return value, None
    # G49 (J29): the file name and the pin table of the run's evaluation seed (seed 42 = the v2.6.4 names/pins)
    seed = int(cfg.seed)
    path = (Path(value) if value is not None
            else pool_directory(cfg) / pool_file_name(cfg.journal_task, cfg.vehicle, seed))
    manifest = load_pool_manifest(POOL_DIR)
    pins, _grid = pinned_pool_table(manifest, seed)
    entry = pins.get(path.name)
    if entry is None:
        raise ValueError("D7 pool %s is not pinned in %s for evaluation seed %d"
                         % (path.name, POOL_DIR / "MANIFEST.json", seed))
    if not path.is_file():
        raise FileNotFoundError("D7 pool %s is missing: deploy the admitted pool files (pinned in %s) or set "
                                "USVBENCH_D7_POOL_DIR" % (path, POOL_DIR / "MANIFEST.json"))
    return load_d7_pool(path, task=cfg.journal_task, vessel=cfg.vehicle, tier=int(cfg.journal_geometry_tier),
                        seed=int(cfg.seed), expected_sha256=entry["sha256"], with_meta=True)


def check_pool_suite(meta, suite):
    """The runtime DynamicSuiteConfig must equal the admission suite (except enabled/dose)."""
    if meta is None:
        return
    from dataclasses import asdict
    have, want = dict(meta["suite"]), asdict(suite)
    for name in ("enabled", "dose"):
        have.pop(name, None)
        want.pop(name, None)
    if have != want:
        raise ValueError("the D7 pool was admitted with a different DynamicSuiteConfig (fields %s); re-run the "
                         "offline admission for this suite" % sorted(k for k in set(have) | set(want)
                                                                      if have.get(k) != want.get(k)))


# D16: admitted ladders + explicit not-admitted keys (never a redraw)

def _key_document(key):
    return [int(key.eval_seed), int(key.env), int(key.episode), str(key.hazard_group)]


@dataclass(frozen=True)
class NotAdmitted:
    """One offered key E4 did not admit (joint ladder) and its rejection reasons.

    ``reasons``: ((dose, reason, sampled_states), ...) in E4's rejection order
    (every failed dose plus the ``joint_ladder_member_rejected`` rows).
    ``geometry_id``: id of the E5 scenario E4 saw (None for in-process pools
    built without it); the env checks it against the scenario it builds.
    """

    key: object
    reasons: tuple
    geometry_id: str | None = None

    def primary_reason(self):
        """Reason of the lowest failed dose (the joint-ladder echo rows excluded)."""
        own = [r for r in self.reasons if r[1] != "joint_ladder_member_rejected"]
        return min(own or self.reasons, key=lambda r: r[0])[1]

    def reason_documents(self):
        return [[int(d), str(r), int(n)] for d, r, n in self.reasons]

    @property
    def beyond_offered_grid(self):
        """True for the D24 record of a key beyond the pool's offered grid (never an E4 rejection)."""
        return tuple(self.reasons) == BEYOND_GRID_REASONS

    def document(self):
        return {"key": _key_document(self.key), "geometry_id": self.geometry_id,
                "reasons": self.reason_documents()}

    @classmethod
    def from_document(cls, item, classes=None):
        dyn = _dynamic()
        seed, env, episode, group = item["key"]
        reasons = tuple((int(d), str(r), int(n)) for d, r, n in item["reasons"])
        return cls(dyn.EpisodeKey(int(seed), int(env), int(episode), str(group)), reasons, item["geometry_id"])


@dataclass(frozen=True)
class NotAdmittedAssignment:
    """What the family pool returns for a not-admitted key at any dose (D16).

    The dynamic suite runs the key at D0: no intruder (the moving basin keeps
    its native population frozen at the time-zero positions, E4's D0). ``dose``
    is the RUNTIME suite dose (stamped as ``suite_dose``); the stamped
    assignment dose is always 0, so the key's d7_assignment hash is identical
    at every dose and in every other suite's run (paired, unexposed).
    """

    record: NotAdmitted
    dose: int
    pool_digest: str
    candidate = None

    @property
    def key(self):
        return self.record.key

    @property
    def reasons(self):
        return self.record.reasons


def not_admitted_records(pool, offered, geometry_ids=None):
    """NotAdmitted records of every offered key without an admitted assignment (offered order)."""
    admitted = {k for k, _ in pool.assignments}
    by_key = {}
    for rejection in pool.rejections:
        by_key.setdefault(rejection.key, []).append((int(rejection.dose), str(rejection.reason),
                                                     int(rejection.sampled_states)))
    out = []
    for key in offered:
        if key in admitted:
            continue
        reasons = tuple(by_key.get(key, ()))
        if not reasons:
            raise ValueError("offered D7 key %s is neither admitted nor rejected" % (key,))
        out.append(NotAdmitted(key, reasons, None if geometry_ids is None else geometry_ids.get(key)))
    return out


def offered_grid(offered):
    """The pool's offered grid (CONTEXT_R4 D24), read from its offered keys.

    Returns {"eval_seed", "hazard_group", "num_envs": N, "episodes": E}. Fail
    closed unless the offered keys are exactly one seed, one hazard group and
    the full rectangle envs 0..N-1 x episodes 0..E-1 (what the offline
    admission tool offers): a pool without such a grid cannot say which keys
    lie beyond it.
    """
    keys = tuple(offered)
    if not keys:
        raise ValueError("D7 pool offers no key (no offered grid)")
    seeds = {k.eval_seed for k in keys}
    groups = {k.hazard_group for k in keys}
    if len(seeds) != 1 or len(groups) != 1:
        raise ValueError("D7 pool offered keys must share one eval seed and one hazard group (D24 grid); got "
                         "seeds %s, groups %s" % (sorted(seeds), sorted(groups)))
    num_envs = 1 + max(k.env for k in keys)
    episodes = 1 + max(k.episode for k in keys)
    cells = {(k.env, k.episode) for k in keys}
    if len(keys) != num_envs * episodes or cells != {(i, j) for i in range(num_envs) for j in range(episodes)}:
        raise ValueError("D7 pool offered keys are not the full grid envs 0..%d x episodes 0..%d (D24: the pool "
                         "grid must be one rectangle)" % (num_envs - 1, episodes - 1))
    return {"eval_seed": next(iter(seeds)), "hazard_group": next(iter(groups)), "num_envs": num_envs,
            "episodes": episodes}


def check_document_grid(grid, offered):
    """The pool file's declared grid must equal the grid of its offered keys (fail closed, D24)."""
    if not isinstance(grid, dict):
        raise ValueError("D7 pool file declares no grid (D24 needs num_envs and episodes)")
    episodes = grid.get("episodes")
    if isinstance(episodes, list):
        same_episodes = episodes == list(range(offered["episodes"]))
    else:
        same_episodes = (isinstance(episodes, int) and not isinstance(episodes, bool)
                         and episodes == offered["episodes"])
    same = (grid.get("num_envs") == offered["num_envs"] and same_episodes
            and grid.get("seed", offered["eval_seed"]) == offered["eval_seed"])
    if not same:
        raise ValueError("D7 pool file grid %r differs from the grid of its offered keys (num_envs %d, episodes "
                         "%d, seed %d)" % (grid, offered["num_envs"], offered["episodes"], offered["eval_seed"]))


def beyond_grid_record(key):
    """The D24 not-admitted record of a key beyond the offered grid (no geometry: the pool never saw it)."""
    return NotAdmitted(key, BEYOND_GRID_REASONS, None)


class HazardD7Pool:
    """F1 view of a joint E4 AdmissionPool plus its explicit not-admitted keys (D16).

    ``assignment(key, dose)`` returns the admitted E4 Assignment, or a
    NotAdmittedAssignment for an offered key E4 rejected (never an exception,
    never a redraw). D24: a key with the pool's seed and hazard group, env
    inside the offered grid and episode index >= the grid's episodes per env
    gets a NotAdmittedAssignment whose single reason is "beyond_offered_grid"
    (it is not a pool key: counts unchanged). Any other key outside the
    offered grid (env out of range, another seed or hazard group) stays a
    KeyError (preflight error).
    ``per_dose_counts`` rows are (offered, admitted, not_admitted);
    ``grid`` = offered_grid(offered).
    """

    def __init__(self, admission, offered, not_admitted):
        if not admission.joint:
            raise ValueError("journal D7 pools must be admitted jointly across doses (D11 pairing, D16)")
        self.admission = admission
        self.assignments = admission.assignments
        self.rejections = admission.rejections
        self.offered = tuple(offered)
        self.not_admitted = {r.key: r for r in not_admitted}
        self.pool_digest = admission.pool_digest
        self.joint = admission.joint
        self.keys_offered = int(admission.keys_offered)
        self.keys_admitted = int(admission.keys_admitted)
        self.keys_not_admitted = len(self.not_admitted)
        admitted = {}
        for key, dose in admission.assignments:
            admitted.setdefault(key, set()).add(int(dose))
        if any(doses != {0, 1, 2, 3, 4} for doses in admitted.values()):
            raise ValueError("an admitted D7 key must carry its whole five-dose ladder")
        if (len(set(self.offered)) != len(self.offered) or len(self.offered) != self.keys_offered
                or set(admitted) | set(self.not_admitted) != set(self.offered)
                or set(admitted) & set(self.not_admitted)):
            raise ValueError("D7 pool: every offered key is exactly one of admitted / not_admitted")
        self.per_dose_counts = tuple((int(o), int(a), int(o) - int(a)) for o, a in admission.per_dose_counts)
        if any(row[2] != self.keys_not_admitted for row in self.per_dose_counts):
            raise ValueError("D7 pool per-dose not-admitted counts disagree with the not-admitted keys")
        self.grid = offered_grid(self.offered)

    @classmethod
    def from_admission(cls, admission, offered=None, geometry_ids=None):
        """Wrap an in-process E4 AdmissionPool (offered keys derived when not given)."""
        if offered is None:
            seen = {}
            for key, _ in admission.assignments:
                seen.setdefault(key, None)
            for rejection in admission.rejections:
                seen.setdefault(rejection.key, None)
            offered = sorted(seen, key=lambda k: (k.eval_seed, k.hazard_group, k.env, k.episode))
        return cls(admission, offered, not_admitted_records(admission, offered, geometry_ids))

    def is_admitted(self, key):
        return (key, 0) in self.assignments

    def beyond_offered_grid(self, key):
        """D24: same eval seed and hazard group, env inside the grid, episode index >= episodes per env."""
        grid = self.grid
        return (key.eval_seed == grid["eval_seed"] and key.hazard_group == grid["hazard_group"]
                and 0 <= key.env < grid["num_envs"] and key.episode >= grid["episodes"])

    def assignment(self, key, dose):
        found = self.assignments.get((key, int(dose)))
        if found is not None:
            return found
        record = self.not_admitted.get(key)
        if record is None:
            if not self.beyond_offered_grid(key):
                raise KeyError(key)
            record = beyond_grid_record(key)
        return NotAdmittedAssignment(record, int(dose), self.pool_digest)


def family_pool(pool):
    """The env's pool object: HazardD7Pool as is, an E4 AdmissionPool wrapped, anything else refused."""
    if isinstance(pool, HazardD7Pool):
        return pool
    if hasattr(pool, "assignments") and hasattr(pool, "rejections") and hasattr(pool, "joint"):
        return HazardD7Pool.from_admission(pool)
    raise ValueError("HazardNav journal envs need an E4 AdmissionPool or a HazardD7Pool (the D16 "
                     "not-admitted semantics need the rejection records)")


def appearing_exposure(record_or_stamp):
    """G3: "admitted" or "not_admitted" of a record's (or a stamp's) appearing envelope."""
    from .._shared.conditions.appearing.stamp import exposure_of
    envelope = record_or_stamp.get("appearing_hazards", record_or_stamp)
    return exposure_of(envelope)[0]


def d7_exposure(record_or_stamp):
    """'not_admitted' for a D16 record/stamp, else 'admitted' (E4 stamps carry no exposure field)."""
    stamp = record_or_stamp.get("dynamic_hazards", record_or_stamp)
    return stamp["payload"].get("exposure", "admitted")


def _identity_equal(a, b):
    """Equal under the declared identity quantisation (G7 item A; same rule as geometry_digest)."""
    q = plans.geometry.canonical_quanta
    return len(a) == len(b) and all(q(x) == q(y) for x, y in zip(a, b))


def check_assignment_geometry(pool, scenario, dose):
    """The pool entry of this key must be for THIS scenario (fail closed); returns it.

    Admitted keys: the assignment's geometry id, spawn, goal, bounds and
    static count must match the scenario the env built. Not-admitted keys:
    the geometry id recorded by the admission tool must match (the E5 layout
    of the key, never a redraw). Keys beyond the offered grid (D24): no
    check, the pool has no record of them.
    """
    key = d7_key(scenario)
    try:
        assignment = pool.assignment(key, int(dose))
    except KeyError:
        grid = getattr(pool, "grid", None)
        where = ("" if grid is None else " (pool grid: seed %s, hazard group %s, envs 0..%d, episodes per env %d)"
                 % (grid["eval_seed"], grid["hazard_group"], grid["num_envs"] - 1, grid["episodes"]))
        raise ValueError("D7 key %s is outside this pool%s: env out of the pool's env range, or another eval "
                         "seed or hazard group; re-run the offline admission for this run's seed and env count "
                         "(preflight error, D7 B02; CONTEXT_R4 D24 b)" % (key, where)) from None
    if isinstance(assignment, NotAdmittedAssignment) and assignment.record.beyond_offered_grid:
        return assignment          # D24: the pool geometry check is skipped (the pool never saw this key)
    if isinstance(assignment, NotAdmittedAssignment):
        recorded = assignment.record.geometry_id
        if recorded is not None and recorded != scenario.geometry_id:
            raise ValueError("not-admitted D7 key %s was recorded on geometry %s, the env built %s (stale pool "
                             "or changed layout code)" % (key, recorded[:16], scenario.geometry_id[:16]))
        return assignment
    candidate = getattr(assignment, "candidate", None)
    if candidate is None:
        return assignment
    if candidate.geometry_id != scenario.geometry_id:
        raise ValueError("D7 assignment for %s was admitted on geometry %s, the env built %s (stale pool or "
                         "changed layout code)" % (key, candidate.geometry_id[:16], scenario.geometry_id[:16]))
    if scenario.task == "moving_basin" and tuple(candidate.base_slots) != tuple(scenario.native_slots):
        raise ValueError("D7 native slots of %s differ from the env's basin slots" % (key,))
    request = candidate.request
    pose = request.initial_pose
    # G7 item A: compared under the declared identity quantisation (journal_geometry_v2.IDENTITY_QUANTUM),
    # never as exact floats: the env's spawn yaw goes through wrap_angle (libm atan2/sin/cos), whose last
    # ulp may differ between the admission machine and the run machine.
    same = (_identity_equal((pose.x, pose.y, pose.yaw, pose.speed),
                            (scenario.start_xy[0], scenario.start_xy[1], scenario.start_yaw, 0.0))
            and _identity_equal(tuple(request.bounds), tuple(scenario.bounds))
            and _identity_equal((request.goal.x, request.goal.y, request.goal.radius),
                                (scenario.goal_xy[0], scenario.goal_xy[1], scenario.goal_radius_m))
            and len(request.static_obstacles) == len(scenario.static_slots))
    if not same:
        raise ValueError("D7 assignment for %s does not match the env's spawn/goal/bounds/obstacles" % (key,))
    return assignment


def _frozen_natives(scenario):
    """E4's D0 of the moving basin: the native templates at speed 0 (time-zero positions, forever)."""
    from dataclasses import replace
    return tuple((int(slot), replace(t, speed_mps=0.0)) for slot, t in zip(scenario.native_slots,
                                                                           scenario.native_templates))


class NotAdmittedBridge:
    """The E4 HazardBridge plus the D16 not-admitted rows, behind the same interface.

    The F0 mixin drives ``reset_rows / step / contact / finish / rows``.
    Admitted rows go to the E4 bridge unchanged. A not-admitted row (the pool
    returned a NotAdmittedAssignment) runs the key at D0 with no intruder:
    ``step`` returns no samples (moving basin: the native population frozen
    at its time-zero positions, published in its own slots and swept for
    contact exactly like the E4 bridge does, obstacle speed bound 0), and
    ``finish`` returns a D7-format stamp ``{"payload", "sha256"}`` with
    ``payload["exposure"] = "not_admitted"`` and E4's rejection reasons
    (``payload["not_admitted"]["reasons"]``). The env stages each row's
    scenario (``stage``) before the mixin resets the rows.
    """

    def __init__(self, inner, task):
        self.inner = inner
        self.task = task
        self.adapter = inner.adapter
        self.cfg = inner.cfg
        self.dt = inner.dt
        self.own_speed_bound = inner.own_speed_bound
        self._staged = {}
        self._na_rows = {}
        self._mode = {}

    # -- mixin interface ---------------------------------------------------
    @property
    def rows(self):
        return {env: (self.inner.rows[env] if mode == "admitted" else self._na_rows[env])
                for env, mode in self._mode.items()}

    def stage(self, env, scenario):
        self._staged[int(env)] = scenario

    def is_not_admitted(self, env):
        return self._mode.get(int(env)) == NOT_ADMITTED

    def reset_rows(self, env_ids, assignments):
        ids = tuple(int(i) for i in env_ids)
        na = tuple(i for i in ids if isinstance(assignments[i], NotAdmittedAssignment))
        admitted = tuple(i for i in ids if i not in na)
        staged = {}
        for env in na:
            assignment = assignments[env]
            scenario = self._staged.get(env)
            if scenario is None or d7_key(scenario) != assignment.key or assignment.key.env != env:
                raise ValueError("not-admitted D7 row %d needs its staged scenario of key %s" % (env, assignment.key))
            row = self.rows.get(env)
            if row is not None and not row["finished"]:
                raise ValueError("finish and stamp every assigned episode before reset")
            if assignment.record.geometry_id not in (None, scenario.geometry_id):
                raise ValueError("not-admitted D7 row %d: recorded geometry differs from the scenario" % env)
            staged[env] = self._new_row(assignment, scenario)
        for env in admitted:
            if self._mode.get(env) == NOT_ADMITTED and not self._na_rows[env]["finished"]:
                raise ValueError("finish and stamp every assigned episode before reset")
        if admitted:
            self.inner.reset_rows(admitted, {i: assignments[i] for i in admitted})
        for env in admitted:
            self._mode[env] = "admitted"
        for env, row in staged.items():
            self._na_rows[env] = row
            self._mode[env] = NOT_ADMITTED

    def step(self, ticks, poses):
        admitted = {env: t for env, t in ticks.items() if self._mode.get(env) == "admitted"}
        out = self.inner.step(admitted, poses) if admitted else {}
        for env, tick in ticks.items():
            if env in admitted:
                continue
            if self._mode.get(env) != NOT_ADMITTED:
                raise KeyError(env)
            out[env] = self._step_not_admitted(self._na_rows[env], env, tick, poses[env])
        return out

    def contact(self, env):
        if self._mode.get(env) == NOT_ADMITTED:
            return self._na_rows[env]["contact"]
        return self.inner.contact(env)

    def finish(self, env, termination_reason):
        if self._mode.get(env) != NOT_ADMITTED:
            return self.inner.finish(env, termination_reason)
        dyn = _dynamic()
        if termination_reason not in dyn.TERMINATION_REASONS:
            raise ValueError("explicit terminal reason required: " + repr(termination_reason))
        row = self._na_rows[env]
        if row["finished"]:
            return row["stamp"]
        row["stamp"] = not_admitted_stamp(row, env, termination_reason, self.dt)
        row["finished"] = True
        return row["stamp"]

    # -- not-admitted rows -------------------------------------------------
    def _new_row(self, assignment, scenario):
        objects = _frozen_natives(scenario) if scenario.task == "moving_basin" else ()
        return {"assignment": assignment, "scenario": scenario, "objects": objects,
                "footprint": vessel_envelope(scenario.vessel).footprint,
                "sensor_range_m": float(scenario.layout.params["ray_sensor_max_range_m"]),
                "last_tick": -1, "prev": None, "cache": None, "contact": False, "first_contact_s": None,
                "swept_intervals": 0, "observer_updates": 0, "min_surface_clearance_m": math.inf,
                "sampled_contact": False, "last_time_s": -1.0, "finished": False, "stamp": None}

    def _step_not_admitted(self, row, env, tick, pose):
        dyn = _dynamic()
        if row["finished"]:
            raise ValueError("row finished; reset it before stepping")
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        if tick == row["last_tick"]:
            return row["cache"]
        if tick < row["last_tick"]:
            raise ValueError("physics tick went backwards without a reset")
        if not isinstance(pose, dyn.Pose):
            raise TypeError("pose must be a Pose")
        time_s = tick * self.dt
        samples = []
        footprint = row["footprint"]
        row["observer_updates"] += 1
        for slot, obj in row["objects"]:
            x, y = obj.position(time_s)
            edge = math.hypot(x - pose.x, y - pose.y) - obj.radius_m
            gap = dyn.point_clearance(x, y, pose, footprint) - obj.radius_m
            row["min_surface_clearance_m"] = min(row["min_surface_clearance_m"], gap)
            row["sampled_contact"] |= gap <= 0.0
            samples.append(dyn.ObstacleSample(slot, x, y, obj.radius_m, True, edge <= row["sensor_range_m"]))
        samples = tuple(samples)
        prev = row["prev"]
        if prev is not None:
            prev_tick, prev_pose, prev_samples = prev
            interval = (tick - prev_tick) * self.dt
            deviation = (self.own_speed_bound + 0.0) * interval * 0.5
            start = [dyn.Circle(s.x, s.y, s.radius_m) for s in prev_samples if s.physical]
            end = [dyn.Circle(s.x, s.y, s.radius_m) for s in samples if s.physical]
            angular = abs(_e4_wrap(pose.yaw - prev_pose.yaw))
            row["swept_intervals"] += 1
            if start and not dyn.swept_interval_safe(prev_pose, pose, footprint, start, end, 0.0, angular,
                                                     deviation):
                if not row["contact"]:
                    row["first_contact_s"] = prev_tick * self.dt
                row["contact"] = True
        row["prev"] = (tick, pose, samples)
        row["last_tick"] = tick
        row["last_time_s"] = time_s
        row["cache"] = samples
        return samples


def not_admitted_stamp(row, env, termination_reason, physics_dt_s):
    """D7-format stamp of one finished not-admitted episode (D16; digest rule of E4 stamp.py).

    The ``assignment`` block keeps the fields the D11 d7_assignment hash
    reads (key, task, dose = 0, candidate_digest = None), so the hash is the
    same at every suite dose and in every other suite's run.
    """
    dyn = _dynamic()
    from .._shared.conditions.dynamic import adapters, stamp
    assignment, scenario = row["assignment"], row["scenario"]
    key, envelope = assignment.key, vessel_envelope(scenario.vessel)
    fp = envelope.footprint
    reasons = assignment.record.reason_documents()
    rule = D24_RULE if assignment.record.beyond_offered_grid else D16_RULE
    natives = [{"x0": obj.position(0.0)[0], "y0": obj.position(0.0)[1], "radius_m": obj.radius_m,
                "speed_bound_mps": 0.0, "native_kind": obj.kind, "slot": slot} for slot, obj in row["objects"]]
    summary = {
        "key": {"eval_seed": key.eval_seed, "env": key.env, "episode": key.episode,
                "hazard_group": key.hazard_group},
        "task": adapters.TASK_ALIASES.get(scenario.task, scenario.task), "dose": 0, "exposure": NOT_ADMITTED,
        "geometry_id": scenario.geometry_id, "vessel_id": envelope.vessel_id,
        "footprint": [fp.x_min_m, fp.x_max_m, fp.y_min_m, fp.y_max_m],
        "intruders": [], "native_objects_frozen": natives,
        "candidate_digest": None, "witness_digest": None, "pool_digest": assignment.pool_digest,
        "not_admitted_reasons": reasons, "primary_reason": assignment.record.primary_reason(),
    }
    runtime = {
        "assigned": False, "itt_count": 0, "exposure": NOT_ADMITTED, "env": int(env),
        "termination_reason": termination_reason, "suite_dose": int(assignment.dose),
        "last_time_s": max(0.0, row["last_time_s"]), "first_detection_s": None,
        "realized_ttc_at_first_detection_s": None, "detected_samples": 0,
        "observer_updates": row["observer_updates"],
        "min_sampled_surface_clearance_m": (None if math.isinf(row["min_surface_clearance_m"])
                                            else row["min_surface_clearance_m"]),
        "sampled_contact": row["sampled_contact"], "non_exposure_reason": NOT_ADMITTED,
        "bridge": {"swept_contact_latched": row["contact"],
                   "first_swept_contact_interval_start_s": row["first_contact_s"],
                   "swept_intervals": row["swept_intervals"], "last_tick": row["last_tick"],
                   "physics_dt_s": float(physics_dt_s),
                   "swept_contact_semantics": "conservative: may include uncertainty"},
    }
    payload = {"schema_version": stamp.SCHEMA_VERSION, "model_version": stamp.MODEL_VERSION,
               "exposure": NOT_ADMITTED,
               "not_admitted": {"reasons": reasons, "primary_reason": assignment.record.primary_reason(),
                                "rule": rule},
               "assignment": summary, "runtime": runtime,
               "semantics": dict(stamp.SEMANTICS, not_admitted=D16_RULE)}
    return json.loads(stamp.canonical({"payload": payload, "sha256": dyn.digest(payload)}))


def pool_header_extra(pool, meta=None):
    """Extra D7 certificate-header fields of the family pool (D16 counts, pins)."""
    manifest = POOL_DIR / "MANIFEST.json"
    out = {"per_dose_count_fields": list(PER_DOSE_COUNT_FIELDS),
           "keys_not_admitted": getattr(pool, "keys_not_admitted", None),
           "not_admitted_rule": D16_RULE,
           "pool_file_sha256": None if meta is None else meta.get("file_sha256"),
           "pool_manifest_sha256": (file_sha256(manifest) if meta is not None and manifest.is_file() else None)}
    grid = getattr(pool, "grid", None)
    if grid is None:
        raise ValueError("the family pool carries no offered grid (D24 header grid)")
    out["grid"] = {"num_envs": int(grid["num_envs"]), "episodes": int(grid["episodes"])}
    return out


def _e4_wrap(angle):
    from .._shared.conditions.dynamic.kinematics import wrap
    return wrap(angle)


@dataclass(frozen=True)
class JournalSceneSpec:
    """What HazardNavEnv._setup_scene reads from ``vehicle_spec`` for a v2 vessel."""

    name: str
    asset_kind: str = "rigid_object"
    bow_body_axis: str = "+x"


def journal_scene_spec(cfg):
    """Scene spec of a journal cfg's v2 vessel (legacy names rejected with the reason)."""
    from .._shared.journal_vehicle import v2_spec_or_raise
    spec = v2_spec_or_raise(cfg.vehicle)
    if spec.asset_kind != "rigid_object" or spec.bow_body_axis != "+x":
        raise ValueError("journal v2 vessels are rigid objects with the bow on +x")
    return JournalSceneSpec(spec.name, spec.asset_kind, spec.bow_body_axis)


# --------------------------------------------------------------------------
# exact CPython math.hypot twin (float64 tensors)
# --------------------------------------------------------------------------

_SPLIT = 134217729.0          # 2**27 + 1 (Veltkamp split for binary64)
_DBL_MIN = 2.2250738585072014e-308


def _two_product_error(a, b, p):
    """Exact error of p = fl(a*b) (Dekker; equals fma(a, b, -p) without over/underflow)."""
    c = _SPLIT * a
    a_hi = c - (c - a)
    a_lo = a - a_hi
    c = _SPLIT * b
    b_hi = c - (c - b)
    b_lo = b - b_hi
    return ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo


def py_hypot(x, y):
    """Bitwise twin of CPython >= 3.10 ``math.hypot(x, y)`` (mathmodule.c vector_norm).

    Lossless scaling by 2**-max_e, double-length squares and sums, one
    differential correction; inf wins over nan, zero returns zero; a
    subnormal maximum is rescaled by DBL_MIN first (as CPython does).
    Verified bitwise against math.hypot in the CPU tests.
    """
    ax, ay, mx, subnormal = _py_hypot_prepare(x, y)
    return _py_hypot_finish(ax, ay, mx, subnormal, bool(subnormal.any()))


def _py_hypot_prepare(x, y):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64, device=x.device)
    x, y = torch.broadcast_tensors(x, y)
    ax, ay = x.abs(), y.abs()
    mx = torch.maximum(ax, ay)
    subnormal = (mx > 0.0) & (mx < _DBL_MIN)
    return ax, ay, mx, subnormal


def py_hypot_many(pairs):
    """G11: ``[py_hypot(x, y) for x, y in pairs]`` with the per-pair subnormal checks travelling to the
    host in ONE transfer (each pair runs exactly py_hypot's operations; only the sync is shared)."""
    prepared = [_py_hypot_prepare(x, y) for x, y in pairs]
    flags = torch.stack([p[3].any() for p in prepared]).tolist()
    return [_py_hypot_finish(*p, flag) for p, flag in zip(prepared, flags)]


def _py_hypot_finish(ax, ay, mx, subnormal, any_subnormal):
    if any_subnormal:
        rescaled = _DBL_MIN * py_hypot(torch.where(subnormal, ax / _DBL_MIN, ax),
                                       torch.where(subnormal, ay / _DBL_MIN, ay))
        plain = py_hypot(torch.where(subnormal, torch.zeros_like(ax), ax),
                         torch.where(subnormal, torch.zeros_like(ay), ay))
        return torch.where(subnormal, rescaled, plain)
    nan = torch.isnan(ax) | torch.isnan(ay)
    inf = torch.isinf(ax) | torch.isinf(ay)
    safe = ~(nan | inf) & (mx > 0.0)
    m = torch.where(safe, mx, torch.ones_like(mx))
    _, exponent = torch.frexp(m)
    scale = torch.ldexp(torch.ones_like(m), -exponent)
    csum = torch.ones_like(m)
    frac1 = torch.zeros_like(m)
    frac2 = torch.zeros_like(m)
    for v in (torch.where(safe, ax, torch.zeros_like(ax)), torch.where(safe, ay, torch.zeros_like(ay))):
        s = v * scale
        hi = s * s
        lo = _two_product_error(s, s, hi)
        total = csum + hi
        low = hi - (total - csum)
        csum = total
        frac1 = frac1 + lo
        frac2 = frac2 + low
    h = torch.sqrt(csum - 1.0 + (frac1 + frac2))
    neg = -h
    hi = neg * h
    lo = _two_product_error(neg, h, hi)
    total = csum + hi
    low = hi - (total - csum)
    csum = total
    frac1 = frac1 + lo
    frac2 = frac2 + low
    correction = csum - 1.0 + (frac1 + frac2)
    h = h + correction / (2.0 * h)
    out = h / scale
    out = torch.where(mx == 0.0, torch.zeros_like(out), out)
    out = torch.where(nan, torch.full_like(out, math.nan), out)
    return torch.where(inf, torch.full_like(out, math.inf), out)


# --------------------------------------------------------------------------
# tensor twin of journal_contracts_v2.ReachEpisode
# --------------------------------------------------------------------------

class ReachEpisodeTensor:
    """Vectorised twin of the scalar E5 ``ReachEpisode`` (float64, one row per env).

    Same lifecycle: ``reset(ids, com_xy, goal_xy, goal_radius, deadline)``,
    ``observe_physics(...)`` after every physics substep, ``step(...)`` once per
    control step (returns terminated, truncated, reason codes) and the
    completed-episode fields (``completed_fields(i)``) that survive the reset.
    Every arithmetic step reproduces the scalar code's operation order, and
    ``py_hypot`` reproduces CPython's ``math.hypot``, so the decisions AND the
    float metrics are bit-identical to the scalar oracle (CPU parity test).
    The only libm function whose last bit can differ is ``acos`` in the tilt
    test, which only matters within one ulp of the declared capsize limit.
    The optional per-motor squared-thrust integral is not used by this family
    (the scalar oracle is driven without ``motor_thrust_n`` as well).
    """

    RUNNING = int(contracts.TerminationReason.RUNNING)

    def __init__(self, num_envs, device, validity_limits, nominal_com_height_m):
        n = int(num_envs)
        self.n, self.device, self.dtype = n, torch.device(device), torch.float64
        if not isinstance(validity_limits, contracts.ValidityLimits):
            validity_limits = contracts.ValidityLimits(**dict(validity_limits))
        self.limits = validity_limits
        self.nominal = float(nominal_com_height_m)
        f = lambda value: torch.full((n,), value, dtype=self.dtype, device=self.device)
        b = lambda value: torch.full((n,), value, dtype=torch.bool, device=self.device)
        self.goal = torch.zeros((n, 2), dtype=self.dtype, device=self.device)
        self.goal_radius = f(1.0)
        self.deadline = f(1.0)
        self.reason = torch.zeros(n, dtype=torch.long, device=self.device)
        self.nonfinite_seen = b(False)
        self.done = b(False)
        self.success = b(False)
        self.clean = b(True)
        self.contact_since_step = b(False)
        self.prev_xy = torch.zeros((n, 2), dtype=self.dtype, device=self.device)
        self.prev_time = f(0.0)
        self.full_path = f(0.0)
        self.prefix_path = f(math.nan)
        self.time_to_success = f(math.nan)
        self.xte_integral = f(0.0)
        self.valid_prefix_only = b(False)
        self.min_clearance = f(math.inf)
        # completed-episode record (survives reset; exported BEFORE reset)
        self.c_available = b(False)
        self.c_success = b(False)
        self.c_time_to_success = f(math.nan)
        self.c_prefix = f(math.nan)
        self.c_full = f(0.0)
        self.c_reason = torch.zeros(n, dtype=torch.long, device=self.device)
        self.c_goal_distance = f(math.nan)
        self.c_goal_distance_known = b(False)
        self.c_min_clearance = f(math.inf)
        self.c_nonfinite = b(False)
        self.c_valid_prefix_only = b(False)
        self.c_deadline = f(math.nan)
        self.c_xte = f(0.0)

    # ---- lifecycle
    def reset(self, env_ids, com_xy, goal_xy, goal_radius_m, deadline_s):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        com = torch.as_tensor(com_xy, dtype=self.dtype, device=self.device).reshape(-1, 2)
        goal = torch.as_tensor(goal_xy, dtype=self.dtype, device=self.device).reshape(-1, 2)
        radius = torch.as_tensor(goal_radius_m, dtype=self.dtype, device=self.device).reshape(-1)
        deadline = torch.as_tensor(deadline_s, dtype=self.dtype, device=self.device).reshape(-1)
        if not bool(torch.isfinite(goal).all()) or not bool((torch.isfinite(radius) & (radius > 0)).all()):
            raise ValueError("goal must be finite and the goal radius positive and finite")
        if not bool((torch.isfinite(deadline) & (deadline > 0)).all()):
            raise ValueError("deadline must be positive and finite")
        if not bool(torch.isfinite(com).all()):
            raise ValueError("malformed or non-finite vector")
        self.goal[ids] = goal
        self.goal_radius[ids] = radius
        self.deadline[ids] = deadline
        self.prev_xy[ids] = com
        self.prev_time[ids] = 0.0
        self.full_path[ids] = 0.0
        self.prefix_path[ids] = math.nan
        self.time_to_success[ids] = math.nan
        self.xte_integral[ids] = 0.0
        self.valid_prefix_only[ids] = False
        self.min_clearance[ids] = math.inf
        self.reason[ids] = self.RUNNING
        self.nonfinite_seen[ids] = False
        self.done[ids] = False
        self.success[ids] = False
        self.clean[ids] = True
        self.contact_since_step[ids] = False

    # ---- state (journal_contracts_v2._state_values / validity_reason)
    def _state(self, com, quat, lin, ang, water):
        com = torch.as_tensor(com, dtype=self.dtype, device=self.device).reshape(self.n, 3)
        quat = torch.as_tensor(quat, dtype=self.dtype, device=self.device).reshape(self.n, 4)
        lin = torch.as_tensor(lin, dtype=self.dtype, device=self.device).reshape(self.n, 3)
        ang = torch.as_tensor(ang, dtype=self.dtype, device=self.device).reshape(self.n, 3)
        water = torch.as_tensor(water, dtype=self.dtype, device=self.device).expand(self.n)
        w, x, y, z = quat.unbind(-1)
        norm = torch.sqrt(((w * w + x * x) + y * y) + z * z)
        valid = (torch.isfinite(com).all(-1) & torch.isfinite(quat).all(-1) & torch.isfinite(lin).all(-1)
                 & torch.isfinite(ang).all(-1) & torch.isfinite(water) & math.isfinite(self.nominal)
                 & torch.isfinite(norm) & (norm > 1.0e-12))
        safe_norm = torch.where(valid, norm, torch.ones_like(norm))
        qx = x / safe_norm
        qy = y / safe_norm
        return valid, com, qx, qy, water

    def validity_codes(self, com, quat, lin, ang, water):
        valid, com, qx, qy, water = self._state(com, quat, lin, ang, water)
        cosine = torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)
        tilt = torch.acos(torch.where(valid, cosine, torch.ones_like(cosine)))
        heave = (com[:, 2] - water) - self.nominal
        out = torch.zeros(self.n, dtype=torch.long, device=self.device)
        out = torch.where(valid & (heave > self.limits.max_heave_m), int(contracts.TerminationReason.AIRBORNE), out)
        out = torch.where(valid & (heave < self.limits.min_heave_m), int(contracts.TerminationReason.SINKING), out)
        out = torch.where(valid & (tilt > self.limits.capsize_angle_rad),
                          int(contracts.TerminationReason.CAPSIZE), out)
        out = torch.where(~valid, int(contracts.TerminationReason.NONFINITE), out)
        return out, valid, com

    def observe_physics(self, com, quat, lin, ang, water, *, collision=None, clearance_m=None, active=None,
                        clearance_present=None):
        self._observe(com, quat, lin, ang, water, collision=collision, clearance_m=clearance_m, active=active,
                      clearance_present=clearance_present)

    def _observe(self, com, quat, lin, ang, water, *, collision=None, clearance_m=None, active=None,
                 clearance_present=None):
        """EpisodeBase.observe_physics for every live row (done rows are skipped).

        ``clearance_present`` (bool per row, default all True when a clearance
        is given) marks rows whose scalar oracle receives ``clearance_m``;
        the other rows correspond to ``clearance_m=None`` (no obstacle to
        measure; a non-finite clearance that IS given counts as contact, as in
        the scalar code).
        """
        live = ~self.done if active is None else (~self.done & torch.as_tensor(active, device=self.device))
        observed, _, _ = self.validity_codes(com, quat, lin, ang, water)
        self.nonfinite_seen |= live & (observed == int(contracts.TerminationReason.NONFINITE))
        latch = live & (self.reason == self.RUNNING)
        self.reason = torch.where(latch, observed, self.reason)
        hit = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if collision is not None:
            hit = torch.as_tensor(collision, dtype=torch.bool, device=self.device).expand(self.n).clone()
        if clearance_m is not None:
            c = torch.as_tensor(clearance_m, dtype=self.dtype, device=self.device).expand(self.n)
            present = torch.ones(self.n, dtype=torch.bool, device=self.device) if clearance_present is None                 else torch.as_tensor(clearance_present, dtype=torch.bool, device=self.device).expand(self.n)
            finite = torch.isfinite(c)
            self.min_clearance = torch.where(live & present & finite, torch.minimum(self.min_clearance, c),
                                             self.min_clearance)
            hit = hit | (present & (~finite | (finite & (c <= 0.0))))
        self.contact_since_step |= live & hit

    def step(self, time_s, com, quat, lin, ang, water, *, collision=None, out_of_bounds=None, clearance_m=None,
             clearance_present=None):
        """ReachEpisode.step for every row; returns (terminated, truncated, reason).

        Rows already done must have been reset (the scalar oracle raises);
        the env resets every done row in the same control step.
        """
        # G11: the three entry checks travel to the host in ONE transfer (same checks, same order of the
        # errors; the clock checks have no side effect, so evaluating them before the first raise is inert)
        any_done = self.done.any()
        try:
            t = torch.as_tensor(time_s, dtype=self.dtype, device=self.device).expand(self.n)
        except Exception:
            if bool(any_done):
                raise RuntimeError("reset required after completion") from None
            raise
        checks = torch.stack((any_done, (torch.isfinite(t) & (t > self.prev_time)).all(),
                              (t > self.deadline + 1.0e-9).any())).tolist()
        if checks[0]:
            raise RuntimeError("reset required after completion")
        if not checks[1]:
            raise ValueError("sample clock must increase from the reset time zero")
        if checks[2]:
            raise ValueError("the adapter must include a sample exactly at the deadline")
        oob = torch.zeros(self.n, dtype=torch.bool, device=self.device) if out_of_bounds is None else \
            torch.as_tensor(out_of_bounds, dtype=torch.bool, device=self.device).expand(self.n)
        # _observe: observe_physics(state, collision, clearance) then consume the contact latch
        self._observe(com, quat, lin, ang, water, collision=collision, clearance_m=clearance_m,
                      clearance_present=clearance_present)
        contact = self.contact_since_step.clone()
        self.contact_since_step[:] = False
        self.clean &= ~contact
        _, valid, com3 = self.validity_codes(com, quat, lin, ang, water)
        self.valid_prefix_only |= ~valid
        px, py = com3[:, 0], com3[:, 1]
        previous = self.prev_xy.clone()
        # segment_hits_disk(previous, current, goal, radius) inputs (G11: computed before the three hypot
        # calls, which then share ONE host transfer; every input below is unchanged until its use)
        ax = previous[:, 0] - self.goal[:, 0]
        ay = previous[:, 1] - self.goal[:, 1]
        dx = px - previous[:, 0]
        dy = py - previous[:, 1]
        nrm = dx * dx + dy * dy
        frac = torch.clamp(-(ax * dx + ay * dy) / nrm, 0.0, 1.0)
        frac = torch.where(nrm == 0.0, torch.zeros_like(frac), frac)
        step_len, hit_distance, goal_distance = py_hypot_many((
            (px - previous[:, 0], py - previous[:, 1]),
            (ax + frac * dx, ay + frac * dy),
            (px - self.goal[:, 0], py - self.goal[:, 1])))
        # EpisodeMetrics.observe (valid rows only)
        self.full_path = torch.where(valid, self.full_path + step_len, self.full_path)
        self.prev_xy = torch.where(valid.unsqueeze(-1), torch.stack((px, py), -1), self.prev_xy)
        self.prev_time = torch.where(valid, t, self.prev_time)
        candidate = valid & (hit_distance <= self.goal_radius)
        # _finish
        running = self.reason == self.RUNNING
        safe = running & self.clean & ~oob
        newly = candidate & safe & ~self.success
        self.success |= newly
        latch = newly & torch.isnan(self.time_to_success)
        self.time_to_success = torch.where(latch, t, self.time_to_success)
        self.prefix_path = torch.where(latch, self.full_path, self.prefix_path)
        timeout = t >= self.deadline - 1.0e-9
        reason = torch.full((self.n,), self.RUNNING, dtype=torch.long, device=self.device)
        reason = torch.where(timeout, int(contracts.TerminationReason.TIMEOUT), reason)
        reason = torch.where(self.success, int(contracts.TerminationReason.SUCCESS), reason)
        reason = torch.where(contact, int(contracts.TerminationReason.COLLISION), reason)
        reason = torch.where(oob, int(contracts.TerminationReason.OUT_OF_BOUNDS), reason)
        reason = torch.where(~running, self.reason, reason)
        terminated = (reason != self.RUNNING) & (reason != int(contracts.TerminationReason.TIMEOUT))
        truncated = reason == int(contracts.TerminationReason.TIMEOUT)
        done = terminated | truncated
        keep_success = (reason == int(contracts.TerminationReason.SUCCESS)) | truncated
        self.success = torch.where(done & ~keep_success, torch.zeros_like(self.success), self.success)
        self.done |= done
        # completed record (EpisodeResult + details), atomically at the terminal step
        self.c_available = torch.where(done, torch.ones_like(done), self.c_available)
        self.c_success = torch.where(done, self.success, self.c_success)
        self.c_time_to_success = torch.where(done, torch.where(self.success, self.time_to_success,
                                                               torch.full_like(t, math.nan)),
                                             self.c_time_to_success)
        self.c_prefix = torch.where(done, torch.where(self.success, self.prefix_path, torch.full_like(t, math.nan)),
                                    self.c_prefix)
        self.c_full = torch.where(done, self.full_path, self.c_full)
        self.c_reason = torch.where(done, reason, self.c_reason)
        self.c_goal_distance = torch.where(done, torch.where(valid, goal_distance, torch.full_like(t, math.nan)),
                                           self.c_goal_distance)
        self.c_goal_distance_known = torch.where(done, valid, self.c_goal_distance_known)
        self.c_min_clearance = torch.where(done, self.min_clearance, self.c_min_clearance)
        self.c_nonfinite = torch.where(done, self.nonfinite_seen, self.c_nonfinite)
        self.c_valid_prefix_only = torch.where(done, self.valid_prefix_only, self.c_valid_prefix_only)
        self.c_deadline = torch.where(done, self.deadline, self.c_deadline)
        self.c_xte = torch.where(done, self.xte_integral, self.c_xte)
        return terminated, truncated, torch.where(done, reason, torch.zeros_like(reason))

    def completed_fields(self, env_id):
        """``EpisodeResult.evaluator_fields()`` of the last completed episode of one row."""
        i = int(env_id)
        if not bool(self.c_available[i]):
            raise RuntimeError("env %d has no completed contract episode" % i)
        code = int(self.c_reason[i])
        return dict(episode_success=bool(self.c_success[i]),
                    time_to_success=float(self.c_time_to_success[i]),
                    episode_path_length=float(self.c_full[i]),
                    success_prefix_path_length_m=float(self.c_prefix[i]),
                    full_episode_path_length_m=float(self.c_full[i]),
                    termination_reason=code,
                    termination_reason_name=contracts.TERMINATION_REASON_NAMES[code],
                    goal_distance_m=float(self.c_goal_distance[i]) if bool(self.c_goal_distance_known[i]) else None,
                    path_cross_track_error_integral_m_s=float(self.c_xte[i]),
                    path_length_valid_prefix_only=bool(self.c_valid_prefix_only[i]),
                    min_clearance_m=float(self.c_min_clearance[i]),
                    nonfinite_seen=bool(self.c_nonfinite[i]),
                    deadline_s=float(self.c_deadline[i]),
                    integrated_squared_thrust_n2s=None)


# --------------------------------------------------------------------------
# torch geometry: COM-polygon footprint clearance (E5 vessel record)
# --------------------------------------------------------------------------

def footprint_vertices(vessel, device="cpu"):
    """CCW COM-frame collision polygon of the E5 vessel record (bow +x), float64 [V,2]."""
    rec = plans.vessel_record(vessel)
    return torch.as_tensor(rec.footprint.vertices_m, dtype=torch.float64, device=device)


def polygon_clearance(com_xy, yaw, vertices, centers, radii, mask):
    """Exact signed separation between the posed convex polygon and every disk.

    com_xy [N,2], yaw [N], vertices [V,2] (CCW, COM frame), centers [N,M,2],
    radii [N,M], mask [N,M] -> [N] (min over masked disks, +inf when none).
    Same definition as journal_geometry_v2.pose_clearance (zero = contact).
    """
    com = com_xy.to(torch.float64)
    c, s = torch.cos(yaw.to(torch.float64)), torch.sin(yaw.to(torch.float64))
    rel = centers.to(torch.float64) - com.unsqueeze(1)
    bx = c.unsqueeze(1) * rel[..., 0] + s.unsqueeze(1) * rel[..., 1]
    by = -s.unsqueeze(1) * rel[..., 0] + c.unsqueeze(1) * rel[..., 1]
    p = torch.stack((bx, by), -1).unsqueeze(-2)                 # [N,M,1,2]
    a = vertices.unsqueeze(0).unsqueeze(0)                      # [1,1,V,2]
    b = torch.roll(vertices, -1, 0).unsqueeze(0).unsqueeze(0)
    e = b - a
    w = p - a
    den = (e * e).sum(-1).clamp_min(1.0e-30)
    t = ((w * e).sum(-1) / den).clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * e
    dist = torch.linalg.vector_norm(p - closest, dim=-1).amin(-1)  # [N,M]
    cross = e[..., 0] * w[..., 1] - e[..., 1] * w[..., 0]
    inside = (cross >= 0.0).all(-1)
    signed = torch.where(inside, -dist, dist) - radii.to(torch.float64)
    signed = torch.where(mask, signed, torch.full_like(signed, math.inf))
    if signed.shape[-1] == 0:
        return torch.full(com.shape[:-1], math.inf, dtype=torch.float64, device=com.device)
    return signed.amin(-1)


def swept_polygon_clearance(prev_pose, pose, vertices, centers, radii, mask, core_radius=None):
    """Conservative clearance over the straight motion prev_pose -> pose (static disks).

    Depth-0 form of journal_geometry_v2.swept_clearance: midpoint pose
    clearance minus the displacement bound |dxy|/2 + r_core |dyaw|/2 (the
    signed distance is 1-Lipschitz in rigid displacement), so a positive
    value certifies every intermediate pose. With no motion it equals the
    exact pose clearance. Poses are [N,3] (x, y, yaw), env-local.
    """
    prev_pose = prev_pose.to(torch.float64)
    pose = pose.to(torch.float64)
    dyaw = torch.atan2(torch.sin(pose[:, 2] - prev_pose[:, 2]), torch.cos(pose[:, 2] - prev_pose[:, 2]))
    mid_xy = 0.5 * (prev_pose[:, :2] + pose[:, :2])
    mid_yaw = prev_pose[:, 2] + 0.5 * dyaw
    core_r = footprint_core_radius(vertices) if core_radius is None else float(core_radius)
    motion = 0.5 * torch.linalg.vector_norm(pose[:, :2] - prev_pose[:, :2], dim=-1) + 0.5 * core_r * dyaw.abs()
    return polygon_clearance(mid_xy, mid_yaw, vertices, centers, radii, mask) - motion


def footprint_core_radius(vertices):
    """Largest COM-to-vertex distance of the footprint polygon (float; one host read)."""
    return float(torch.linalg.vector_norm(vertices, dim=-1).max())


def yaw_from_quat(quat):
    w, x, y, z = quat.to(torch.float64).unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


__all__ = ["HazardD7Pool", "NotAdmitted", "NotAdmittedAssignment", "NotAdmittedBridge", "JournalSceneSpec", "journal_scene_spec", "resolve_d7_pool", "check_pool_suite",
           "not_admitted_records", "family_pool", "d7_exposure", "not_admitted_stamp", "pool_header_extra", "check_assignment_geometry", "pool_directory", "FAMILY", "CARRIED_TASKS", "HazardScenario", "build_scenario", "d7_context", "d7_key",
           "family_hashes", "storage_width", "vessel_envelope", "quarter_turn_xy", "quarter_turn_bounds",
           "load_d7_pool", "pool_document", "write_pool_file", "pool_encode", "pool_decode", "py_hypot",
           "ReachEpisodeTensor", "polygon_clearance", "swept_polygon_clearance", "footprint_vertices",
           "yaw_from_quat", "default_pool_path", "load_pool_manifest", "BASIN_PRIMITIVE", "POOL_DIR",
           "offered_grid", "check_document_grid", "beyond_grid_record", "BEYOND_OFFERED_GRID", "D24_RULE"]
