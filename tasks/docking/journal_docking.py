"""Journal docking scenario (contract version 2): keyed primitives, D7 context, hashes.

Pure CPU module (numpy + the shared journal modules; no Isaac Lab import), used
by BOTH the environment (DockingEnv journal branch, tasks/docking/docking_env.py)
and the offline D7 admission tool, so the env and the admitted pool see the SAME
env-local geometry for every (eval seed, env, episode) key.

Primitives and their keyed streams (C1 finding 29: one named stream per
primitive, keyed by (protocol, eval seed, env, episode, group), reseeded per
episode, never a shared batch generator; tasks/_shared/scenario_rng.py
numpy_generator):
  * ``berth``      group "docking.berth_pose": the berth heading in the env frame
                   (uniform on [-pi, pi)); the berth centre is the env origin.
  * ``spawn_pose`` group "docking.spawn_pose": the numpy Generator handed to
                   journal_layouts_v2.build_layout("docking", ...), whose only
                   draws are the lateral start offset and the start heading
                   (journal_layouts_v2.py _cand_docking).
  * sea / wind / current / perception: the schema-3 condition layer's own keyed
    streams (conditions/keyed_rng.py, "cond3.*"), never touched here.
Enabling or dosing any suite therefore cannot move the berth or the spawn.

Geometry: E5's admitted layout is built in the berth frame (berth centre at the
origin, dock heading +x) and rotated rigidly by the berth heading about the env
origin (every E5 quantity is rotation covariant, journal_layouts_v2.py:1-16).
Positions are hull COM, env-local metres.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np

from .._shared import journal_cfg_v2 as plans
from .._shared import journal_layouts_v2 as layouts
from .._shared.scenario_rng import SCENARIO_PROTOCOL_VERSION, numpy_generator

TASK = "docking"
FAMILY = "docking"
BERTH_GROUP = "docking.berth_pose"
SPAWN_GROUP = "docking.spawn_pose"
SCENARIO_SCHEMA = "usvbench.docking.journal_scenario/1"


def _sha(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def rotate_xy(points, heading_rad):
    """Rotate (..., 2) berth-frame points about the origin by heading_rad."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    pts = np.asarray(points, dtype=np.float64)
    out = np.empty_like(pts)
    out[..., 0] = c * pts[..., 0] - s * pts[..., 1]
    out[..., 1] = s * pts[..., 0] + c * pts[..., 1]
    return out


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class DockingScenario:
    """One admitted journal docking episode in env-local coordinates (immutable)."""

    vessel: str
    tier: int
    seed: int
    env: int
    episode: int
    layout: object                   # journal_layouts_v2.JournalLayout (berth frame)
    berth_heading_rad: float         # berth-frame +x in the env frame
    start_xy: tuple                  # COM spawn, env-local
    start_heading_rad: float         # bow heading, env frame
    wall_centers: np.ndarray         # (W, 2) env-local
    wall_radii: np.ndarray           # (W,)
    berth: MappingProxyType          # Berth fields (center_xy, heading_rad, opening/back/half width) + tolerances
    bounds_berth_frame: tuple        # (x0, x1, y0, y1) in the berth frame (out-of-bounds test)
    deadline_s: float
    deadline_steps: int
    berth_walls: bool = True         # False: virtual berth (no wall geometry; rays see D7 objects only)

    @property
    def params(self):
        return self.layout.params

    @property
    def length_scale(self):
        return float(self.layout.params["length_scale"])

    @property
    def geometry_id(self):
        # G7 item A: the berth heading enters as its declared identity quantisation (journal_geometry_v2
        # canonical_identity_value, 1e-6 rad), never as the exact repr of its last ulp.
        payload = {"schema": SCENARIO_SCHEMA, "layout": self.layout.geometry_hash,
                   "identity_canon": plans.geometry.IDENTITY_CANON_VERSION,
                   "berth_heading_rad": plans.geometry.canonical_identity_value(float(self.berth_heading_rad))}
        if not self.berth_walls:
            payload["berth_walls"] = False
        return _sha(payload)

    # ---------------------------------------------------------- D11 hashes --
    def berth_content(self):
        """Exogenous berth primitive: pose, interior, tolerances, wall geometry."""
        b = self.berth
        return {"schema": SCENARIO_SCHEMA, "vessel": self.vessel, "tier": self.tier,
                "center_xy": [float(v) for v in b["center_xy"]], "heading_rad": b["heading_rad"],
                "opening_x_m": b["opening_x_m"], "back_x_m": b["back_x_m"], "half_width_m": b["half_width_m"],
                "position_tolerance_m": b["position_tolerance_m"],
                "heading_tolerance_rad": b["heading_tolerance_rad"],
                "speed_tolerance_m_s": b["speed_tolerance_m_s"], "hold_s": b["hold_s"],
                "berth_walls": self.berth_walls,
                "wall_centers": self.wall_centers.tolist(), "wall_radii": self.wall_radii.tolist(),
                "stream": {"group": BERTH_GROUP, "protocol": SCENARIO_PROTOCOL_VERSION,
                           "seed": self.seed, "env": self.env, "episode": self.episode}}

    def spawn_content(self):
        """Exogenous spawn primitive: COM start pose and the scenario deadline."""
        return {"schema": SCENARIO_SCHEMA, "vessel": self.vessel, "tier": self.tier,
                "start_xy": [float(v) for v in self.start_xy], "start_heading_rad": self.start_heading_rad,
                "start_speed_mps": 0.0, "deadline_s": self.deadline_s, "deadline_steps": self.deadline_steps,
                "nominal_route_length_m": float(self.layout.nominal_route_length_m),
                "stream": {"group": SPAWN_GROUP, "protocol": SCENARIO_PROTOCOL_VERSION,
                           "seed": self.seed, "env": self.env, "episode": self.episode}}

    def family_hashes(self):
        """{primitive: sha256} for pairing.FAMILY_PRIMITIVES['docking'] = (berth, spawn_pose)."""
        return {"berth": _sha({"primitive": "berth", "content": self.berth_content()}),
                "spawn_pose": _sha({"primitive": "spawn_pose", "content": self.spawn_content()})}

    def in_bounds(self, x, y):
        """COM (env-local) inside the admitted layout bounds (berth frame, exact)."""
        c, s = math.cos(self.berth_heading_rad), math.sin(self.berth_heading_rad)
        bx, by = c * x + s * y, -s * x + c * y
        x0, x1, y0, y1 = self.bounds_berth_frame
        return x0 <= bx <= x1 and y0 <= by <= y1


def berth_heading(seed, env, episode):
    """Keyed berth-pose draw (its own stream; C1 finding 29)."""
    rng = numpy_generator(SCENARIO_PROTOCOL_VERSION, int(seed), int(env), int(episode), BERTH_GROUP)
    return float(rng.uniform(-math.pi, math.pi))


def spawn_generator(seed, env, episode):
    """Keyed spawn-pose stream handed to build_layout (its only draws are the spawn)."""
    return numpy_generator(SCENARIO_PROTOCOL_VERSION, int(seed), int(env), int(episode), SPAWN_GROUP)


def build_scenario(vessel, tier, seed, env, episode, berth_walls=True):
    """(vessel, tier, eval seed, env, task episode k) -> DockingScenario (pure, deterministic).

    ``episode`` is the task episode index k of the env (0, 1, 2, ...; the
    condition layer and the D7 key use the same index, JournalEnvMixin).
    The scenario of a key is ALWAYS the E5 layout of that key (CONTEXT_R4
    D16): there is no redraw of any kind; a key that E4 does not admit keeps
    this geometry and runs the dynamic suite at D0 (``not_admitted``).
    ``berth_walls=False`` keeps the berth interior as a VIRTUAL region (the
    contract's containment is unchanged) but removes the wall geometry
    (no wall clearance, rays see the D7 slot only; D7 spec row D5 / R4 D14).
    """
    for name, value in (("seed", seed), ("env", env), ("episode", episode), ("tier", tier)):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError("%s must be a nonnegative integer" % name)
    layout = layouts.build_layout(TASK, int(tier), vessel, spawn_generator(seed, env, episode))
    theta = berth_heading(seed, env, episode)
    b = dict(layout.extras["berth"])
    if tuple(b["center_xy"]) != (0.0, 0.0) or b["heading_rad"] != 0.0:
        raise ValueError("E5 docking layouts are built in the berth frame (centre origin, heading 0)")
    start = rotate_xy(layout.start, theta)
    if berth_walls:
        centers = rotate_xy(layout.centers, theta)
        radii = np.array(layout.radii, dtype=np.float64)
    else:
        centers, radii = np.zeros((0, 2)), np.zeros(0)
    centers.setflags(write=False)
    radii.setflags(write=False)
    berth = dict(b)
    berth["center_xy"] = (0.0, 0.0)
    berth["heading_rad"] = theta
    return DockingScenario(
        vessel=vessel, tier=int(tier), seed=int(seed), env=int(env), episode=int(episode), layout=layout,
        berth_heading_rad=theta, start_xy=(float(start[0]), float(start[1])),
        start_heading_rad=wrap_angle(float(layout.start_heading_rad) + theta),
        wall_centers=centers, wall_radii=radii, berth=MappingProxyType(berth),
        bounds_berth_frame=tuple(float(v) for v in layout.bounds_m),
        deadline_s=float(layout.deadline_s), deadline_steps=int(layout.deadline_steps),
        berth_walls=bool(berth_walls))


def rotated_bounds_bbox(scenario):
    """Axis-aligned env-frame box enclosing the rotated berth-frame bounds (D7 witness box)."""
    x0, x1, y0, y1 = scenario.bounds_berth_frame
    corners = rotate_xy(np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1))), scenario.berth_heading_rad)
    return (float(corners[:, 0].min()), float(corners[:, 0].max()),
            float(corners[:, 1].min()), float(corners[:, 1].max()))


def d7_envelope(dynamic, vessel):
    """The vessel envelope exactly as JournalEnvMixin.journal_setup builds it."""
    from .._shared import vehicles_v2
    record = plans.vessel_record(vessel)
    envelope, _ = dynamic.envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel),
                                                reference_speed_mps=record.cruise_speed_m_s,
                                                turning_radius_m=record.turning_radius_design_m)
    return envelope


def d7_context(dynamic, scenario, envelope=None):
    """TaskContext (E4 adapters.TaskContext) of one scenario in env-local metres.

    Maps E5's plain-data ``d7_hook`` (journal_layouts_v2._d7_context) onto the
    D7 dataclasses after the berth rotation; the sensor is the journal 36-ray
    fan of the plan (30 m x s, CONTEXT_R3 decision 14 / R4 D14).
    """
    hook = scenario.layout.d7_hook
    theta = scenario.berth_heading_rad
    env = envelope if envelope is not None else d7_envelope(dynamic, scenario.vessel)

    def circle(item):
        x, y, r = item
        p = rotate_xy(np.array((x, y)), theta)
        return dynamic.Circle(float(p[0]), float(p[1]), float(r))

    sensor = hook["sensor"]
    heading = hook["final_heading_rad"]
    return dynamic.TaskContext(
        task=TASK, envelope=env,
        initial_pose=dynamic.Pose(float(scenario.start_xy[0]), float(scenario.start_xy[1]),
                                  float(scenario.start_heading_rad), 0.0),
        bounds=rotated_bounds_bbox(scenario), goal=circle(hook["goal"]), deadline_s=float(hook["deadline_s"]),
        route=tuple(circle(c) for c in hook["route"]),
        route_mandatory=tuple(bool(f) for f in hook["route_mandatory"]),
        static_obstacles=(tuple(circle(c) for c in hook["static_obstacles"]) if scenario.berth_walls else ()),
        sensor=dynamic.SensorSpec(kind=sensor["kind"], ray_count=int(sensor["ray_count"]),
                                  max_range_m=float(sensor["max_range_m"]),
                                  source="journal docking 36-ray fan (journal_cfg_v2 plan, 30 m x s)"),
        final_speed_mps=hook["final_speed_mps"],
        final_heading_rad=None if heading is None else wrap_angle(float(heading) + theta),
        heading_tolerance_rad=float(hook["heading_tolerance_rad"]), dwell_s=float(hook["dwell_s"]),
        site_index=int(hook["site_index"]), base_max=int(hook["base_max"]),
        base_count=int(hook["base_count"]) if scenario.berth_walls else 0,
        geometry_id=scenario.geometry_id)


def appearing_berth_region(scenario):
    """G20b: the berth rectangle the appearing cylinder must clear (env-local, walls included).

    Berth frame (E5 journal_layouts_v2._berth_structure): interior x in [opening, back face], |y| <= free
    half width; the wall cylinders (radius r_w) run from the opening to the back wall, their centres at
    |y| = half width + r_w and x = back face + r_w. The rectangle covers their full extent:
    x in [opening - r_w, back face + 2 r_w], |y| <= half width + 2 r_w.
    """
    b = scenario.berth
    rw = float(scenario.layout.params["wall_radius_m"])
    return {"center_xy": [float(b["center_xy"][0]), float(b["center_xy"][1])], "heading_rad": float(b["heading_rad"]),
            "x_min_m": float(b["opening_x_m"]) - rw, "x_max_m": float(b["back_x_m"]) + 2.0 * rw,
            "half_width_m": float(b["half_width_m"]) + 2.0 * rw,
            "source": "journal_docking berth (opening, back face, free half width) + wall radius"}


def appearing_admission_inputs(vessel, tier, seed, env, episode, berth_walls=True):
    """G20 / G20b: (D7 TaskContext, length scale, None, {"berth": rectangle, "mouth": block}) of one key's REAL
    journal docking scenario (radius 0.8 s); the berth rectangle feeds the "overlaps_berth" placement rule, the mouth
    block (G20d) the berth-mouth distance rule.

    berth_walls must equal the env's cfg.berth_walls (the journal templates set True).
    """
    from importlib import import_module
    dynamic = import_module(__name__.rsplit(".", 2)[0] + "._shared.conditions.dynamic")
    scenario = build_scenario(vessel, int(tier), int(seed), int(env), int(episode), berth_walls=berth_walls)
    return (d7_context(dynamic, scenario), float(scenario.layout.params["length_scale"]), None,
            {"berth": appearing_berth_region(scenario), "mouth": appearing_mouth_rule(scenario)})


def appearing_mouth_rule(scenario):
    """G20d (owner decision 2026-10-04, docking): the berth-mouth line in the berth frame of appearing_berth_region
    (x = the opening) and the vessel's approach run-up max(2 L, turning radius, braking distance)
    (journal_geometry_v2.VesselGeometry.approach_runup_m, the straight run-up E5 itself reserves before the mouth)."""
    from importlib import import_module
    plans = import_module(__name__.rsplit(".", 2)[0] + "._shared.journal_cfg_v2")
    geometry = plans.vessel_record(scenario.vessel).geometry
    return {"opening_x_m": float(scenario.berth["opening_x_m"]), "approach_runup_m": float(geometry.approach_runup_m),
            "source": "journal_docking berth opening; journal_geometry_v2.VesselGeometry.approach_runup_m"}


__all__ = ["BERTH_GROUP", "SPAWN_GROUP", "DockingScenario", "build_scenario", "berth_heading", "spawn_generator", "d7_context", "d7_envelope", "rotated_bounds_bbox", "rotate_xy", "wrap_angle"]
