"""Shared CPU test fixtures. Synthetic task-SHAPED geometry, not the real layouts.

Loads the package by path (no `tasks` import, no Isaac Lab, no bytecode).
Geometry is scaled per vessel by u = max(beam / 0.93, length / 1.2), the
D4-style "do not scale by beam alone" factor, so the same fixture is
meaningful for BlueBoat and WAM-V. Deadlines follow D4's rule
T = 2 * nominal route length / cruise speed + dwell (TASK_CONTRACTS_v2.md:153-161).

E4: vessel envelopes come from D5's FINAL registry (tasks/_shared/vehicles_v2.py
and vehicles_v2_data.py, loaded by path without executing any package
__init__), built with the journal defaults (reference 1.0 m/s, R = L,
speed-cap fraction 0.8); the D7 PROVISIONAL presets no longer exist.
"""
import importlib.util
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
SHARED = PACKAGE.parents[1]          # tasks/_shared (holds vehicles_v2.py)
SIMULATOR_ROOTS = ("torch", "isaaclab", "isaacsim", "omni", "pxr")
_BEFORE = set(sys.modules)
if "dynamic_cpu" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("dynamic_cpu", PACKAGE / "__init__.py",
                                                   submodule_search_locations=[str(PACKAGE)])
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["dynamic_cpu"] = _module
    _spec.loader.exec_module(_module)

import dynamic_cpu as d  # noqa: E402

REGISTRY = d.load_registry_by_path(SHARED)
# Modules that loading the package and the registry added to this process.
LOADED_BY_PACKAGE = frozenset(set(sys.modules) - _BEFORE)
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1", "wamv_vrx_v2_opt2")
RAYS = d.SensorSpec(ray_count=36, max_range_m=30.0, source="existing 36-ray fan, 30 m")
_ENVELOPES = {}


def spec(vessel_id):
    return REGISTRY.get_vehicle_v2(vessel_id)


def envelope(vessel_id):
    if vessel_id not in _ENVELOPES:
        _ENVELOPES[vessel_id] = d.envelope_from_spec_v2(spec(vessel_id))[0]
    return _ENVELOPES[vessel_id]


def synthetic_envelope(**overrides):
    values = dict(vessel_id="synthetic_fixture", footprint=d.Footprint.symmetric(2.0, 1.0),
                  reference_speed_mps=1.0, max_speed_mps=1.5, top_speed_mps=2.0,
                  max_accel_mps2=1.0, max_decel_mps2=1.0, max_yaw_rate_rps=1.0,
                  reaction_s=0.1, provenance="CPU synthetic bounds; NOT BlueBoat or WAM-V")
    values.update(overrides)
    return d.VesselEnvelope(**values)


def scale(env):
    return max(env.beam_m / 0.93, env.length_m / 1.2)


def _deadline(length_m, env, dwell=0.0):
    return 2.0 * length_m / env.reference_speed_mps + dwell


def _ring(cx, cy, radius, pillar_r, gap_angles, gap_half, count):
    out = []
    for i in range(count):
        a = 2 * math.pi * i / count
        if any(abs(math.remainder(a - g, 2 * math.pi)) < gap_half for g in gap_angles):
            continue
        out.append(d.Circle(cx + radius * math.cos(a), cy + radius * math.sin(a), pillar_r))
    return tuple(out)


def context(task, env, sensor=RAYS, **overrides):
    u = scale(env)
    v = env.reference_speed_mps
    moving = d.Pose(0.0, 0.0, 0.0, v)
    rest = d.Pose(0.0, 0.0, 0.0, 0.0)
    big = (-80 * u, 80 * u, -80 * u, 80 * u)
    common = dict(task=task, envelope=env, bounds=big, sensor=sensor)
    if task == "reach":
        c = dict(initial_pose=moving, goal=d.Circle(30 * u, 0, 2 * u), deadline_s=_deadline(30 * u, env))
    elif task in ("path_following", "path_hazard"):
        route = (d.Circle(12 * u, 0, 1.5 * u), d.Circle(24 * u, 6 * u, 1.5 * u), d.Circle(36 * u, 6 * u, 1.5 * u))
        obstacles = () if task == "path_following" else (
            d.Circle(18 * u, -4 * u, 1.0 * u), d.Circle(30 * u, 11 * u, 1.0 * u), d.Circle(6 * u, 4 * u, 0.8 * u))
        c = dict(initial_pose=moving, goal=d.Circle(48 * u, 6 * u, 2 * u), route=route,
                 route_mandatory=(True, True, True), static_obstacles=obstacles, site_index=2,
                 deadline_s=_deadline(52 * u, env), base_max=len(obstacles), base_count=len(obstacles))
    elif task == "station_keeping":
        settled = 2.0 * 12 * u / v + 10.0
        c = dict(initial_pose=d.Pose(-12 * u, 0.0, 0.0, 0.0), goal=d.Circle(0, 0, 0.5 * u),
                 hold_circle=d.Circle(0, 0, 3.0 * u), settled_phase_start_s=settled,
                 deadline_s=settled + 60.0)
    elif task == "docking":
        walls = (d.Circle(2.5 * u, 2.0 * u, 0.6 * u), d.Circle(2.5 * u, -2.0 * u, 0.6 * u),
                 d.Circle(4.0 * u, 0.0, 0.6 * u))
        c = dict(initial_pose=moving, goal=d.Circle(1.5 * u, 0, 0.5 * u), static_obstacles=walls,
                 final_speed_mps=0.1 * v, final_heading_rad=0.0, dwell_s=5.0,
                 initial_pose_override=None, deadline_s=_deadline(27 * u, env, 5.0),
                 base_max=3, base_count=3)
        c.pop("initial_pose_override")
        c["initial_pose"] = d.Pose(-25 * u, 0.0, 0.0, v)
    elif task == "scatter":
        field = tuple(d.Circle(x * u, y * u, 0.8 * u) for x, y in
                      ((8, 5), (14, -5), (20, 6), (26, -6), (32, 5), (11, -12), (23, 13)))
        c = dict(initial_pose=moving, goal=d.Circle(40 * u, 0, 2 * u), static_obstacles=field,
                 deadline_s=_deadline(40 * u, env), base_max=14, base_count=len(field))
    elif task == "ring":
        pillars = _ring(0, 0, 14 * u, 0.9 * u, (0.0,), 0.2, 36)
        c = dict(initial_pose=moving, goal=d.Circle(28 * u, 0, 2 * u),
                 route=(d.Circle(14 * u, 0, 1.2 * u),), route_mandatory=(True,), site_index=0,
                 static_obstacles=pillars, deadline_s=_deadline(28 * u, env),
                 base_max=36, base_count=len(pillars))
    elif task == "fortress":
        pillars = _ring(30 * u, 0, 12 * u, 0.9 * u, (math.pi,), 0.25, 32)
        c = dict(initial_pose=moving, goal=d.Circle(30 * u, 0, 2 * u),
                 route=(d.Circle(18 * u, 0, 1.2 * u),), route_mandatory=(True,), site_index=0,
                 static_obstacles=pillars, deadline_s=_deadline(30 * u, env),
                 base_max=32, base_count=len(pillars))
    elif task == "forced_crossing":
        wall = tuple(d.Circle(20 * u, y * u, 0.9 * u) for y in
                     (-20, -18, -16, -14, -12, -10, -8, -6, -4, 4, 6, 8, 10, 12, 14, 16, 18, 20))
        c = dict(initial_pose=moving, goal=d.Circle(40 * u, 0, 2 * u),
                 route=(d.Circle(20 * u, 0, 1.2 * u),), route_mandatory=(True,), site_index=0,
                 static_obstacles=wall, deadline_s=_deadline(40 * u, env),
                 base_max=len(wall), base_count=len(wall))
    elif task == "iceberg":
        bergs = (d.Circle(20 * u, 6 * u, 4 * u), d.Circle(20 * u, -6 * u, 2.5 * u), d.Circle(30 * u, 12 * u, 3 * u))
        c = dict(initial_pose=moving, goal=d.Circle(40 * u, 0, 2 * u),
                 route=(d.Circle(20 * u, 0.0, 1.0 * u),), route_mandatory=(False,), site_index=0,
                 static_obstacles=bergs, deadline_s=_deadline(40 * u, env), base_max=4, base_count=3)
    elif task == "moving_basin":
        span = 8 * u
        native = (d.NativeMotion("translate", 12 * u, 6 * u, 0.4 * u, 0.0, dx=0.0, dy=1.0, span_m=span),
                  d.NativeMotion("translate", 12 * u, -6 * u, 0.4 * u, 0.0, dx=0.0, dy=1.0, span_m=span,
                                 phase_m=span),
                  d.NativeMotion("orbit", 28 * u, 0.0, 0.4 * u, 0.0, orbit_radius_m=4 * u),
                  d.NativeMotion("static", 20 * u, 14 * u, 0.6 * u, 0.0))
        c = dict(initial_pose=moving, goal=d.Circle(40 * u, 0, 2 * u), native_geometry=native,
                 deadline_s=_deadline(40 * u, env), base_max=4, base_count=4, base_slots=(0, 1, 2, 3))
    elif task == "harbour":
        stages = (d.StageSite("exit_gate", "waypoint_arrival", 0, "head_on"),
                  d.StageSite("transit", "leg_midpoint", 1, "crossing"),
                  d.StageSite("berth_approach", "route_fraction", 0, "crossing", 0.9))
        c = dict(initial_pose=moving, goal=d.Circle(50 * u, 0, 1.0 * u),
                 route=(d.Circle(15 * u, 0, 1.2 * u), d.Circle(35 * u, 0, 1.5 * u)),
                 route_mandatory=(True, True), stages=stages, final_speed_mps=0.1 * v,
                 final_heading_rad=0.0, dwell_s=5.0, deadline_s=_deadline(50 * u, env, 5.0))
    else:
        raise KeyError(task)
    common.update(c)
    common.update(overrides)
    return d.TaskContext(**common)


def key(task, env=0, episode=0, seed=42):
    return d.EpisodeKey(seed, env, episode, "d7_" + task)


def suite(env, **overrides):
    return d.DynamicSuiteConfig(enabled=True, vessel_id=env.vessel_id, **overrides)
