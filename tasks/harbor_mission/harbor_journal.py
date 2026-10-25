"""Harbour-mission journal glue, contract version 2 (task F6). CPU-importable.

torch + numpy only (no Isaac Lab import), so the env, the offline D7
admission tool and the CPU tests share ONE implementation of:

* ``journal_layout``: the admitted E5 layout of (seed, env, episode), drawn
  from the SAME keyed stream the env uses (scenario_rng "layout" group).
* ``d7_task_context``: the E4 TaskContext of a REAL journal layout (offline
  admission input; see its docstring for the three documented deviations from
  E5's ``layout.d7_hook``).
* ``HarborMissionTensorOracle``: the vectorised torch twin of E5's scalar
  ``journal_contracts_v2.HarborMissionEpisode`` (ordered departure gates, the
  transit exit gate, directional berth entry + dwell; contract_scope
  ``harbour_full_mission`` for spawn phase 0, ``harbour_dock_phase`` for spawn
  phase 2). Bit-for-bit parity with the scalar oracle is a test obligation.
  Exactness note: every +, -, *, / and comparison is a separate float64 torch
  op (IEEE, no fusion), in the scalar's evaluation order; the transcendental
  kernels (sqrt, hypot, acos, atan2, sin, cos, and the builtin compensated
  sum() of the quaternion norm) are evaluated with CPython on the CPU float64
  values (``_exact``), because torch's CPU kernels are not correctly rounded
  on this machine (measured on 100,000 doubles, AVX512: sqrt 1,379, atan2
  2,541, acos 4,906 mismatches against math.*) and CPython's float sum() is
  compensated (Python >= 3.12).
  The logic (latches, gates, entry, dwell, termination precedence) is fully
  vectorised; only the scalar kernels are element loops.
* footprint clearance (COM polygon vs disks, pose and depth-0 swept bound, the
  torch form of journal_geometry_v2.pose_clearance / swept_clearance(max_depth=0)),
* ``phase_targets``: the F7 phase-target interface (PHASE_TARGETS_F6.md),
* ``family_scenario_hashes``: D11 family primitives (layout, route, berth, spawn_pose).

Frames: every position here is the hull COM in the ENV-LOCAL frame (world
axes, env origin subtracted), the frame of ``env.journal_estimate.position``.
The harbour journal layout is not rotated (gates along +x, berth heading 0).
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch

from .._shared import journal_cfg_v2 as plans
from .._shared import journal_contracts_v2 as contracts
from .._shared import journal_layouts_v2 as layouts
from .._shared.scenario_rng import SCENARIO_PROTOCOL_VERSION, numpy_generator

TASK = "harbour_mission"
D7_TASK = "harbour"
LAYOUT_GROUP = "layout"                # scenario_draws_hazard.GROUP_LAYOUT (same stream as the env)
GATE_NAMES = ("departure_1", "departure_2", "transit_exit", "berth_mouth")
VIRTUAL_GATES = GATE_NAMES[:3]
STAGE_NAMES = ("departure", "transit", "docking", "complete")
NO_GATE = 4                             # active gate index once the berth entry is latched
CONTRACT_SCOPES = {0: "harbour_full_mission", 2: "harbour_dock_phase"}
PHASE_TARGETS_SCHEMA = "usvbench.harbour_phase_targets/1"
F64 = torch.float64
_RUNNING = int(contracts.TerminationReason.RUNNING)


# ----------------------------------------------------------------- layouts --
def layout_rng(seed, env, episode, protocol_version=SCENARIO_PROTOCOL_VERSION):
    """numpy stream of one (eval seed, env, episode) harbour layout (== ScenarioRNG.numpy_rng("layout"))."""
    return numpy_generator(int(protocol_version), int(seed), int(env), int(episode), LAYOUT_GROUP)


def journal_layout(vessel, tier, seed, env, episode, protocol_version=SCENARIO_PROTOCOL_VERSION):
    """The admitted E5 journal layout (journal_layouts_v2.build_layout) of one scenario key."""
    return layouts.build_layout(TASK, int(tier), vessel, layout_rng(seed, env, episode, protocol_version))


def gate_table(layout):
    """Ordered gates (departure_1, departure_2, transit_exit, berth_mouth) of a layout.

    Returns dict of numpy arrays: midpoints [4,2], normals [4,2], half widths [4]
    (virtual gates: half the virtual gate width; berth mouth: the slip's inner
    half width), approach_m [3], min_normal_speed_mps [3].
    """
    by_name = {g.name: g for g in layout.gates}
    missing = [n for n in GATE_NAMES if n not in by_name]
    if missing:
        raise ValueError("harbour layout lacks gates %s" % (missing,))
    half = float(layout.extras["virtual_gate_half_width_m"])
    berth = layout.extras["berth"]
    mid = np.array([by_name[n].midpoint_m for n in GATE_NAMES], dtype=np.float64)
    nrm = np.array([by_name[n].through_m for n in GATE_NAMES], dtype=np.float64)
    widths = np.array([half, half, half, float(berth["half_width_m"])], dtype=np.float64)
    return dict(midpoints=mid, normals=nrm, half_widths=widths,
                approach_m=np.full(3, float(layout.extras["gate_approach_m"])),
                min_normal_speed_mps=np.full(3, float(layout.extras["gate_min_normal_speed_m_s"])))


def spawn_pose(layout, spawn_phase):
    """(x, y, heading) of the COM spawn in the layout frame.

    spawn phase 0: layout.start with the scenario start heading; spawn phase 2
    (isolated dock phase, contract scope harbour_dock_phase): the transit exit
    gate midpoint (the field exit, where a phase-2 hand-off starts), same heading.
    """
    if spawn_phase == 0:
        return float(layout.start[0]), float(layout.start[1]), float(layout.start_heading_rad)
    if spawn_phase == 2:
        exit_mid = gate_table(layout)["midpoints"][2]
        return float(exit_mid[0]), float(exit_mid[1]), float(layout.start_heading_rad)
    raise ValueError("harbour spawn phase must be 0 or 2")


def oracle_row(layout, spawn_phase=0, deadline_s=None):
    """Per-row ``HarborMissionTensorOracle.reset_rows`` arguments of one layout (numpy / floats)."""
    gates = gate_table(layout)
    rec = layout.extras["berth"]
    spawn = spawn_pose(layout, spawn_phase)
    return dict(com_xy=np.array(spawn[:2]), deadline_s=float(layout.deadline_s if deadline_s is None else deadline_s),
                berth=np.array((rec["center_xy"][0], rec["center_xy"][1], rec["heading_rad"], rec["opening_x_m"],
                                rec["back_x_m"], rec["half_width_m"]), dtype=np.float64),
                tolerances=np.array((rec["position_tolerance_m"], rec["heading_tolerance_rad"],
                                     rec["speed_tolerance_m_s"], rec["hold_s"]), dtype=np.float64),
                spawn_phase=int(spawn_phase), gate_midpoints=gates["midpoints"][:3], gate_normals=gates["normals"][:3],
                gate_half_widths=gates["half_widths"][:3], gate_approach_m=gates["approach_m"],
                gate_min_speed_mps=gates["min_normal_speed_mps"])


def stack_rows(rows):
    """Stack oracle_row dicts into reset_rows keyword arguments."""
    out = {}
    for name in rows[0]:
        values = [r[name] for r in rows]
        out[name] = torch.as_tensor(np.array(values), dtype=torch.long if name == "spawn_phase" else F64)
    return out


def scalar_oracle(layout, record, spawn_phase=0, limits=None, deadline_s=None):
    """E5's scalar HarborMissionEpisode for a journal layout (the parity reference)."""
    gates = gate_table(layout)
    berth = layout.extras["berth"]
    virtual = tuple(contracts.VirtualGate(GATE_NAMES[k], tuple(gates["midpoints"][k]), tuple(gates["normals"][k]),
                                          float(gates["half_widths"][k]), float(gates["approach_m"][k]),
                                          float(gates["min_normal_speed_mps"][k])) for k in range(3))
    if limits is None:
        lo, hi = record.heave_limits_m
        limits = contracts.ValidityLimits(record.tilt_limit_rad, lo, hi)
    kwargs = dict(hold_s=berth["hold_s"], position_tolerance_m=berth["position_tolerance_m"],
                  heading_tolerance_rad=berth["heading_tolerance_rad"],
                  speed_tolerance_mps=berth["speed_tolerance_m_s"], validity_limits=limits)
    deadline = float(layout.deadline_s if deadline_s is None else deadline_s)
    if spawn_phase == 0:
        return contracts.HarborMissionEpisode(contracts.Berth.from_layout(berth), record.polygon_com_xy_m,
                                              deadline, departure_gates=virtual[:2],
                                              transit_gate=virtual[2], spawn_phase=0, **kwargs)
    return contracts.HarborMissionEpisode(contracts.Berth.from_layout(berth), record.polygon_com_xy_m,
                                          deadline, spawn_phase=2, **kwargs)


# ------------------------------------------------------------- D7 context --
STAGE_SITES = {
    # stage -> (site rule, route anchor). ONE fixed rule per stage (round 4b, D16):
    # no fallback variant, no re-attempt. E4 fixes the stage per key by strata;
    # a key whose joint ladder E4 rejects is recorded as not_admitted (it keeps
    # its E5 layout and runs the dynamic suite at D0), never re-sited.
    "departure": ("waypoint_arrival", "departure_2"),
    "transit": ("waypoint_arrival", "field_start"),
    "docking": ("leg_midpoint", "berth_mouth"),
}


def d7_route(layout):
    """Route circles of the nominal D7 plan: the admitted witness polyline.

    Witness vertices between start and goal become circles (gate vertices:
    radius 0.5 x planned centre width of the gate for the berth mouth, the
    planning margin elsewhere); a non-gate vertex collinear with its
    neighbours is dropped. Returns (circles, mandatory flags, anchor index map).
    """
    witness = np.asarray(layout.witness.vertices_m, dtype=np.float64)
    p = layout.params
    anchors = {g.name: (np.asarray(g.midpoint_m, dtype=np.float64), g) for g in layout.gates}
    anchors["field_start"] = (np.array((p["field_start_x_m"], 0.0)), None)
    small = float(p["planning_margin_m"])
    route, mandatory, index = [], [], {}
    for i, v in enumerate(witness[1:-1]):
        tag = None
        for name, (point, _) in anchors.items():
            if abs(v[0] - point[0]) <= 1e-9 and abs(v[1] - point[1]) <= 1e-9:
                tag = name
        prev, nxt = witness[i], witness[i + 2]
        collinear = abs((v[0] - prev[0]) * (nxt[1] - v[1]) - (v[1] - prev[1]) * (nxt[0] - v[0])) < 1e-9
        if tag is None and collinear:
            continue
        radius = small
        if tag == "berth_mouth":
            radius = 0.5 * float(anchors[tag][1].planned_centre_width_m)
        route.append((float(v[0]), float(v[1]), float(radius)))
        mandatory.append(tag is not None and tag != "field_start")
        if tag is not None:
            index[tag] = len(route) - 1
    missing = [name for name in ("departure_2", "field_start", "berth_mouth") if name not in index]
    if missing:
        raise ValueError("witness does not visit the harbour anchors %s" % (missing,))
    return route, mandatory, index


def d7_task_context(layout, envelope, dynamic):
    """E4 TaskContext of one REAL journal harbour layout (env-local metres).

    Built from the E5 layout of the key itself (static obstacles = field +
    berth walls, bounds, goal, deadline, sensor, final speed/heading, dwell,
    base maximum, geometry id all from ``layout`` / ``layout.d7_hook``). Three
    documented deviations from E5's ``d7_hook`` (measured on the seed-42 grid,
    see FINAL_REPORT section D7):
      1. route = the admitted witness polyline (``d7_route``), not the gate
         midpoints: the straight gate-to-gate nominal plan runs through the
         field's designated centre-line blocker ("nominal_prefix_unsafe");
      2. initial pose = the start at rest heading along the route axis (0 rad),
         not the scenario start-heading jitter: E4's nominal plan re-aims with
         an absolute 1e-9 rad threshold, so any residual heading error leaves
         spurious forced intervals that void waypoint-arrival windows;
      3. stage sites (strata-chosen by E4, never rescheduled, ONE rule per
         stage, ``STAGE_SITES``): departure = arrival at departure gate 2;
         transit = arrival at the transit-field entry; docking = the
         approach-lane leg ending at the berth mouth (leg midpoint, slid
         forward by E4 past forced intervals). E5's transit-exit / berth-mouth
         arrivals have no D1-long unforced window on most real layouts.
    The layout is never redrawn or altered for admission (D16).
    """
    hook = layout.d7_hook
    route, mandatory, index = d7_route(layout)
    circles = tuple(dynamic.Circle(*c) for c in route)
    stages = tuple(dynamic.StageSite(stage, STAGE_SITES[stage][0], index[STAGE_SITES[stage][1]])
                   for stage in ("departure", "transit", "docking"))
    params = layout.params
    return dynamic.TaskContext(
        task=D7_TASK, envelope=envelope,
        initial_pose=dynamic.Pose(float(layout.start[0]), float(layout.start[1]), 0.0, 0.0),
        bounds=tuple(float(b) for b in hook["bounds"]), goal=dynamic.Circle(*hook["goal"]),
        deadline_s=float(hook["deadline_s"]), route=circles, route_mandatory=tuple(bool(m) for m in mandatory),
        static_obstacles=tuple(dynamic.Circle(*o) for o in hook["static_obstacles"]),
        sensor=dynamic.SensorSpec(ray_count=int(params["ray_sensor_ray_count"]),
                                  max_range_m=float(params["ray_sensor_max_range_m"]),
                                  source="journal 36-ray fan, 30 m x length scale (harbour env ray block)"),
        final_speed_mps=hook["final_speed_mps"], final_heading_rad=hook["final_heading_rad"],
        heading_tolerance_rad=hook["heading_tolerance_rad"], dwell_s=float(hook["dwell_s"]),
        site_index=0, stages=stages, base_max=int(hook["base_max"]), base_count=int(hook["base_count"]),
        geometry_id=str(hook["geometry_id"]))


def appearing_admission_inputs(vessel, tier, seed, env, episode):
    """G20: (D7 TaskContext, length scale, radius range) of one key's REAL journal harbour layout.

    The context is ``d7_task_context`` of the key's own E5 layout (task name "harbour", the D7
    alias of harbour_mission) with the envelope the runtime bridge uses. Radius rule 4: the task's
    own transit-field radius range (journal_cfg_v2 radius_range_m = [0.8 s, 1.8 s]).
    """
    from importlib import import_module
    from .._shared import vehicles_v2
    dynamic = import_module(__name__.rsplit(".", 2)[0] + "._shared.conditions.dynamic")
    record = plans.vessel_record(vessel)
    envelope, _ = dynamic.envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel),
                                                reference_speed_mps=record.cruise_speed_m_s,
                                                turning_radius_m=record.turning_radius_design_m)
    layout = journal_layout(vessel, int(tier), int(seed), int(env), int(episode))
    params = layout.params
    return (d7_task_context(layout, envelope, dynamic), float(params["length_scale"]),
            tuple(float(v) for v in params["radius_range_m"]))


def admit_harbour_grid(dynamic, vessel, tier, seed, keys, suite, envelope, *, workers=1, log=None,
                       context_fn=None):
    """Offline joint D7 admission of harbour keys on their E5 layouts (D7 B05, round 4b D16).

    keys: iterable of (env, episode). ONE E4 ``admit_grid`` pass over the
    contexts of the keys' own E5 layouts (attempt 0 only: no layout redraw,
    no site variant, no re-attempt). Every key ends either admitted (all five
    doses, joint) or ``not_admitted`` with E4's rejection records. Returns
    (pool, info) with info = {"keys", "layouts", "stage", "not_admitted"}
    where not_admitted = {key: ((dose, reason, sampled_states), ...)} in the
    order of ``pool.rejections``. ``context_fn(layout, envelope, dynamic, key)``
    replaces ``d7_task_context`` (tests only: forcing a real E4 rejection).
    """
    from importlib import import_module

    adapters = import_module(dynamic.__name__ + ".adapters")
    group = "d7_" + D7_TASK
    ordered = [dynamic.EpisodeKey(int(seed), int(e), int(k), group) for e, k in keys]
    layouts_by_key = {key: journal_layout(vessel, tier, seed, key.env, key.episode) for key in ordered}
    if context_fn is None:
        contexts = {key: d7_task_context(layouts_by_key[key], envelope, dynamic) for key in ordered}
    else:
        contexts = {key: context_fn(layouts_by_key[key], envelope, dynamic, key) for key in ordered}
    profile = adapters.PROFILES[D7_TASK]
    stage = {key: adapters._choices(profile, contexts[key], key)["stage"].name for key in ordered}
    pool = dynamic.admit_grid([(contexts[key], key, suite) for key in ordered], workers=workers)
    admitted = {key for (key, _dose) in pool.assignments}
    for (key, _dose), assignment in pool.assignments.items():
        if assignment.candidate.geometry_id != layouts_by_key[key].geometry_hash:
            raise RuntimeError("admitted candidate geometry differs from the layout (%r)" % (key,))
    not_admitted = {}
    for rejection in pool.rejections:
        if rejection.key not in admitted:
            not_admitted.setdefault(rejection.key, []).append(
                (int(rejection.dose), str(rejection.reason), int(rejection.sampled_states)))
    missing = [key for key in ordered if key not in admitted and key not in not_admitted]
    if missing:
        raise RuntimeError("keys neither admitted nor rejected by E4: %r" % (missing,))
    not_admitted = {key: tuple(not_admitted[key]) for key in ordered if key in not_admitted}
    if log is not None:
        log("%s: %d/%d keys admitted, %d not_admitted (single pass, attempt 0 only)"
            % (vessel, len(admitted), len(ordered), len(not_admitted)))
    return pool, {"keys": ordered, "layouts": layouts_by_key, "stage": stage, "not_admitted": not_admitted}


def pool_meta(info, vessel, tier, seed, suite, envelope, grid=None, tool=None):
    """Pool-file metadata of admit_harbour_grid output (harbor_d7_pool.write_pool).

    ``not_admitted`` is the EXPLICIT not-admitted list (D16): one entry per
    rejected key, [env, episode, E5 geometry id, strata stage, [[dose,
    reason, sampled_states], ...]] carrying every E4 rejection record of the
    key (verified against ``pool.rejections`` on load).
    """
    from dataclasses import asdict

    keys = info["keys"]
    return {
        "task": TASK, "d7_task": D7_TASK, "vessel": vessel, "seed": int(seed), "tier": int(tier),
        "grid": grid, "tool": tool, "suite": json.loads(json.dumps(asdict(suite))),
        "envelope": {"vessel_id": envelope.vessel_id, "provenance": envelope.provenance,
                     "t_avoid_s": envelope.t_avoid_s, "top_speed_mps": envelope.top_speed_mps,
                     "reference_speed_mps": envelope.reference_speed_mps},
        "stage_sites": {k: list(v) for k, v in STAGE_SITES.items()},
        "key_stages": [[k.env, k.episode, info["stage"][k]] for k in keys],
        "offered_keys": list(keys),
        "not_admitted": [[k.env, k.episode, info["layouts"][k].geometry_hash, info["stage"][k],
                          [list(r) for r in reasons]] for k, reasons in info["not_admitted"].items()],
    }


# ------------------------------------------------------ footprint clearance --
def _polygon_signed_distance(points, poly):
    """Signed distance [N,K] from points [N,K,2] to CCW convex polygons [N,V,2] (inside < 0)."""
    a = poly.unsqueeze(1)                                   # [N,1,V,2]
    b = torch.roll(poly, -1, dims=1).unsqueeze(1)
    q = points.unsqueeze(2)                                 # [N,K,1,2]
    edge = b - a
    rel = q - a
    norm = (edge * edge).sum(-1)
    frac = ((rel * edge).sum(-1) / norm.clamp_min(1e-300)).clamp(0.0, 1.0)
    diff = rel - frac.unsqueeze(-1) * edge
    dist = torch.sqrt((diff * diff).sum(-1)).amin(-1)       # [N,K]
    inside = ((edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]) >= 0.0).all(-1)
    return torch.where(inside, -dist, dist)


def footprint_pose_clearance(vertices, pose, centers, radii, mask):
    """Minimum signed hull-surface clearance [N] (torch form of journal_geometry_v2.pose_clearance).

    vertices [V,2] COM-relative CCW polygon; pose [N,3] (x, y, yaw); centers
    [N,K,2]; radii [N,K]; mask [N,K] (inactive slots ignored; all inactive -> +inf).
    """
    v = vertices.to(pose)
    c, s = torch.cos(pose[:, 2:3]), torch.sin(pose[:, 2:3])
    x = pose[:, 0:1] + c * v[:, 0] - s * v[:, 1]
    y = pose[:, 1:2] + s * v[:, 0] + c * v[:, 1]
    poly = torch.stack((x, y), -1)                          # [N,V,2]
    sd = _polygon_signed_distance(centers.to(pose), poly) - radii.to(pose)
    sd = torch.where(mask, sd, torch.full_like(sd, math.inf))
    return sd.amin(-1) if sd.shape[-1] else torch.full((pose.shape[0],), math.inf, dtype=pose.dtype)


def _wrap_pi(angle):
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def footprint_swept_clearance(vertices, start_pose, end_pose, centers, radii, mask):
    """Depth-0 swept lower bound [N] (== journal_geometry_v2.swept_clearance(..., max_depth=0), static disks).

    Midpoint pose clearance minus |dxy|/2 + r_core |dyaw|/2 (1-Lipschitz bound
    over every intermediate linear-XY / shortest-yaw pose): conservative.
    """
    a = start_pose.to(F64)
    b = end_pose.to(F64).clone()
    b[:, 2] = a[:, 2] + _wrap_pi(b[:, 2] - a[:, 2])
    mid = 0.5 * (a + b)
    core_r = float(torch.linalg.vector_norm(vertices.to(F64), dim=-1).max())
    motion = torch.linalg.vector_norm(b[:, :2] - a[:, :2], dim=-1) / 2.0 + core_r * (b[:, 2] - a[:, 2]).abs() / 2.0
    return footprint_pose_clearance(vertices.to(F64), mid, centers, radii, mask) - motion


# ------------------------------------------------------------- exact math --
def _exact(fn, *xs):
    """Elementwise CPython ``math`` kernel on CPU float64 (bit-exact with the scalar oracle)."""
    shape = xs[0].shape
    cols = [x.detach().to("cpu", F64).reshape(-1).tolist() for x in xs]
    return torch.tensor([fn(*vals) for vals in zip(*cols)], dtype=F64).reshape(shape)


def _quaternion_norm(w, x, y, z):
    return math.sqrt(sum(q * q for q in (w, x, y, z)))


def _safe(x, valid):
    """Replace invalid rows by 0 so math kernels never see non-finite inputs."""
    return torch.where(valid, x, torch.zeros_like(x))


# ---------------------------------------------------------- tensor oracle --
class HarborMissionTensorOracle:
    """Vectorised twin of journal_contracts_v2.HarborMissionEpisode for N envs (CPU float64).

    Lifecycle (identical to the scalar oracle, per row): ``reset_rows`` after
    installing the admitted layout; ``observe_physics`` after every physics
    substep (the mixin calls it on the PREVIOUS substep's state); ``step`` once
    per control step (before rewards) -> (terminated, truncated, reason);
    ``last_completed[i]`` (an E5 EpisodeResult) is replaced atomically at the
    terminal step and survives reset.

    State inputs are dicts of CPU tensors: com [N,3] (env-local COM), quat
    [N,4] (wxyz), lin [N,3], ang [N,3], water [N] (local surface z, D8),
    nominal [N] (calm COM height above the surface).
    """

    def __init__(self, num_envs, hull_vertices_m, limits):
        n = int(num_envs)
        self.num_envs = n
        self.hull = tuple(tuple(float(c) for c in v) for v in contracts._hull(hull_vertices_m))
        self.limits = limits if isinstance(limits, contracts.ValidityLimits) else contracts.ValidityLimits(**limits)
        z = lambda *s: torch.zeros(s if s else (n,), dtype=F64)
        b = lambda *s: torch.zeros(s if s else (n,), dtype=torch.bool)
        nan = lambda *s: torch.full(s if s else (n,), math.nan, dtype=F64)
        self.deadline_s = torch.ones(n, dtype=F64)
        self.spawn_phase = torch.zeros(n, dtype=torch.long)
        # berth: centre x, y, heading, cos, sin, opening, back, half width; tolerances
        self.berth = z(n, 8)
        self.tol = z(n, 4)                  # position, heading, speed, hold
        self.gate_mid = z(n, 3, 2)
        self.gate_nrm = z(n, 3, 2)
        self.gate_half = z(n, 3)
        self.gate_approach = z(n, 3)
        self.gate_vmin = z(n, 3)
        # EpisodeBase
        self.done = torch.ones(n, dtype=torch.bool)
        self.success = b()
        self.clean = torch.ones(n, dtype=torch.bool)
        self.contact_since_step = b()
        self.validity = torch.zeros(n, dtype=torch.long)
        self.nonfinite_seen = b()
        self.prev_xy = z(n, 2)
        self.prev_time = z()
        self.full_len = z()
        self.prefix_len = nan()
        self.tts = nan()
        self.valid_prefix_only = b()
        self.min_clearance = torch.full((n,), math.inf, dtype=F64)
        self.thrust_sq = None
        self.has_thrust = b()
        # DockingEpisode
        self.entered = b()
        self.hold_started = nan()
        self.prev_local = z(n, 2)
        self.prev_local_time = z()
        self.entry_time = nan()
        # HarborMissionEpisode
        self.gate_times = nan(n, 2)
        self.transit_time = nan()
        self.armed = b(n, 3)
        # last control-step view (phase targets / observation)
        self.dwell = z()
        self.success_now = b()
        self.last_completed = [None] * n

    # ------------------------------------------------------------ reset --
    def reset_rows(self, ids, *, com_xy, deadline_s, berth, tolerances, spawn_phase, gate_midpoints=None,
                   gate_normals=None, gate_half_widths=None, gate_approach_m=None, gate_min_speed_mps=None):
        """Per-row reset (EpisodeBase/DockingEpisode/HarborMissionEpisode.reset).

        berth [k,5] = (centre x, centre y, heading, opening_x, back_x, half_width) -> 6 columns;
        tolerances [k,4] = (position m, heading rad, speed m/s, hold s).
        """
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
        k = ids.numel()
        if k == 0:
            return
        com_xy = torch.as_tensor(com_xy, dtype=F64).reshape(k, 2)
        berth = torch.as_tensor(berth, dtype=F64).reshape(k, 6)
        phase = torch.as_tensor(spawn_phase, dtype=torch.long).reshape(-1).expand(k)
        if bool(((phase != 0) & (phase != 2)).any()):
            raise ValueError("supported spawn phases are 0 (full mission) and 2 (isolated dock)")
        deadline = torch.as_tensor(deadline_s, dtype=F64).reshape(-1).expand(k)
        if not bool((torch.isfinite(deadline) & (deadline > 0)).all()):
            raise ValueError("deadline must be positive and finite")
        heading = berth[:, 2]
        cos_h, sin_h = _exact(math.cos, heading), _exact(math.sin, heading)
        self.berth[ids] = torch.stack((berth[:, 0], berth[:, 1], heading, cos_h, sin_h,
                                       berth[:, 3], berth[:, 4], berth[:, 5]), -1)
        self.tol[ids] = torch.as_tensor(tolerances, dtype=F64).reshape(k, 4)
        self.deadline_s[ids] = deadline
        self.spawn_phase[ids] = phase
        full = phase == 0
        if bool(full.any()):
            if gate_midpoints is None:
                raise ValueError("full mission requires departure gates and a transit gate")
            self.gate_mid[ids] = torch.as_tensor(gate_midpoints, dtype=F64).reshape(k, 3, 2)
            self.gate_nrm[ids] = torch.as_tensor(gate_normals, dtype=F64).reshape(k, 3, 2)
            self.gate_half[ids] = torch.as_tensor(gate_half_widths, dtype=F64).reshape(k, 3)
            self.gate_approach[ids] = torch.as_tensor(gate_approach_m, dtype=F64).reshape(k, 3)
            self.gate_vmin[ids] = torch.as_tensor(gate_min_speed_mps, dtype=F64).reshape(k, 3)
        # EpisodeBase.reset
        self.prev_xy[ids] = com_xy
        self.prev_time[ids] = 0.0
        self.full_len[ids] = 0.0
        self.prefix_len[ids] = math.nan
        self.tts[ids] = math.nan
        self.valid_prefix_only[ids] = False
        self.min_clearance[ids] = math.inf
        self.validity[ids] = _RUNNING
        self.nonfinite_seen[ids] = False
        self.done[ids] = False
        self.success[ids] = False
        self.clean[ids] = True
        self.contact_since_step[ids] = False
        self.has_thrust[ids] = False
        if self.thrust_sq is not None:
            self.thrust_sq[ids] = 0.0
        # DockingEpisode.reset
        self.entered[ids] = False
        self.hold_started[ids] = math.nan
        self.prev_local[ids] = self._local(com_xy[:, 0], com_xy[:, 1], rows=ids)
        self.prev_local_time[ids] = 0.0
        self.entry_time[ids] = math.nan
        # HarborMissionEpisode.reset
        self.gate_times[ids] = math.nan
        self.transit_time[ids] = math.nan
        signed = self._signed_all(com_xy, rows=ids)                    # [k,3]
        armed = signed <= -self.gate_approach[ids]
        self.armed[ids] = torch.where(full.unsqueeze(-1), armed, torch.zeros_like(armed))
        self.dwell[ids] = 0.0
        self.success_now[ids] = False

    # --------------------------------------------------------- geometry --
    def _local(self, px, py, rows=None):
        berth = self.berth if rows is None else self.berth[rows]
        dx, dy = px - berth[:, 0], py - berth[:, 1]
        c, s = berth[:, 3], berth[:, 4]
        return torch.stack((c * dx + s * dy, (-s) * dx + c * dy), -1)

    def _signed_all(self, p, rows=None):
        mid = self.gate_mid if rows is None else self.gate_mid[rows]
        nrm = self.gate_nrm if rows is None else self.gate_nrm[rows]
        return (p[:, None, 0] - mid[..., 0]) * nrm[..., 0] + (p[:, None, 1] - mid[..., 1]) * nrm[..., 1]

    def _state_values(self, state):
        """(_safe_state) -> valid mask, com, normalized quat, lin, water, nominal."""
        com = torch.as_tensor(state["com"], dtype=F64)
        q = torch.as_tensor(state["quat"], dtype=F64)
        lin = torch.as_tensor(state["lin"], dtype=F64)
        ang = torch.as_tensor(state["ang"], dtype=F64)
        water = torch.as_tensor(state["water"], dtype=F64).reshape(-1).expand(self.num_envs)
        nominal = torch.as_tensor(state["nominal"], dtype=F64).reshape(-1).expand(self.num_envs)
        finite = (torch.isfinite(com).all(-1) & torch.isfinite(q).all(-1) & torch.isfinite(lin).all(-1)
                  & torch.isfinite(ang).all(-1) & torch.isfinite(water) & torch.isfinite(nominal))
        qs = _safe(q, finite.unsqueeze(-1))
        # CPython's builtin sum() of floats is compensated (Neumaier, Python >= 3.12); the scalar
        # oracle computes math.sqrt(sum(q * q for q in quaternion)), so the kernel calls exactly that.
        norm = _exact(_quaternion_norm, qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3])
        valid = finite & torch.isfinite(norm) & (norm > 1.0e-12)
        safe_norm = torch.where(valid, norm, torch.ones_like(norm))
        qn = _safe(qs, valid.unsqueeze(-1)) / safe_norm.unsqueeze(-1)
        return valid, _safe(com, valid.unsqueeze(-1)), qn, _safe(lin, valid.unsqueeze(-1)), \
            _safe(water, valid), _safe(nominal, valid)

    def _validity_reason(self, valid, com, qn, water, nominal):
        x, y = qn[:, 1], qn[:, 2]
        arg = 1.0 - 2.0 * (x * x + y * y)
        arg = torch.where(arg < 1.0, arg, torch.ones_like(arg))
        arg = torch.where(arg > -1.0, arg, -torch.ones_like(arg))
        tilt = _exact(math.acos, arg)
        heave = (com[:, 2] - water) - nominal
        lim = self.limits
        out = torch.zeros(self.num_envs, dtype=torch.long)
        out = torch.where(heave > lim.max_heave_m, int(contracts.TerminationReason.AIRBORNE), out)
        out = torch.where(heave < lim.min_heave_m, int(contracts.TerminationReason.SINKING), out)
        out = torch.where(tilt > lim.capsize_angle_rad, int(contracts.TerminationReason.CAPSIZE), out)
        return torch.where(valid, out, torch.full_like(out, int(contracts.TerminationReason.NONFINITE)))

    def _yaw(self, qn):
        w, x, y, z = qn.unbind(-1)
        return _exact(math.atan2, 2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrapped(angle):
        return _exact(math.atan2, _exact(math.sin, angle), _exact(math.cos, angle))

    def _contains(self, local, yaw):
        rel = yaw - self.berth[:, 2]
        c, s = _exact(math.cos, rel), _exact(math.sin, rel)
        x, y = local[:, 0], local[:, 1]
        opening, back, half = self.berth[:, 5], self.berth[:, 6], self.berth[:, 7]
        ok = torch.ones(self.num_envs, dtype=torch.bool)
        for vx, vy in self.hull:
            a = (x + c * vx) - s * vy
            b = (y + s * vx) + c * vy
            ok = ok & (opening <= a) & (a <= back) & (b.abs() <= half)
        return ok

    def _conjunction(self, valid, com, qn, lin):
        yaw = self._yaw(qn)
        local = self._local(com[:, 0], com[:, 1])
        contained = self._contains(local, yaw)
        dist = _exact(math.hypot, com[:, 0] - self.berth[:, 0], com[:, 1] - self.berth[:, 1])
        heading_err = self._wrapped(yaw - self.berth[:, 2]).abs()
        speed = _exact(math.hypot, lin[:, 0], lin[:, 1])
        conj = (self.entered & contained & (dist <= self.tol[:, 0]) & (heading_err <= self.tol[:, 1])
                & (speed <= self.tol[:, 2]))
        return conj & valid, dict(yaw=yaw, local=local, contained=contained, dist=dist,
                                  heading_err=heading_err, speed=speed)

    # --------------------------------------------------- observe_physics --
    def observe_physics(self, state, *, collision=None, clearance=None, has_clearance=None, dt_s=None,
                        motor_thrust=None, thrust_mask=None, rows=None):
        """EpisodeBase + DockingEpisode.observe_physics for the rows in ``rows`` (bool [N], default all)."""
        n = self.num_envs
        active = torch.ones(n, dtype=torch.bool) if rows is None else torch.as_tensor(rows, dtype=torch.bool)
        active = active & ~self.done
        valid, com, qn, lin, water, nominal = self._state_values(state)
        observed = self._validity_reason(valid, com, qn, water, nominal)
        self.nonfinite_seen |= active & (observed == int(contracts.TerminationReason.NONFINITE))
        latch = active & (self.validity == _RUNNING)
        self.validity = torch.where(latch, observed, self.validity)
        collision = torch.zeros(n, dtype=torch.bool) if collision is None else \
            torch.as_tensor(collision, dtype=torch.bool).reshape(-1).expand(n)
        hit = collision.clone()
        dock_hit = collision.clone()
        if clearance is not None:
            c = torch.as_tensor(clearance, dtype=F64).reshape(-1).expand(n)
            has = torch.ones(n, dtype=torch.bool) if has_clearance is None else \
                torch.as_tensor(has_clearance, dtype=torch.bool).reshape(-1).expand(n)
            finite = torch.isfinite(c)
            upd = active & has & finite & (c < self.min_clearance)
            self.min_clearance = torch.where(upd, c, self.min_clearance)
            hit = hit | (has & ~finite) | (has & finite & (c <= 0.0))
            dock_hit = dock_hit | (has & ~(c > 0.0))
        self.contact_since_step |= active & hit
        if motor_thrust is not None and dt_s is not None:
            f = torch.as_tensor(motor_thrust, dtype=F64)
            if self.thrust_sq is None:
                self.thrust_sq = torch.zeros((n, f.shape[-1]), dtype=F64)
            mask = torch.ones(n, dtype=torch.bool) if thrust_mask is None else \
                torch.as_tensor(thrust_mask, dtype=torch.bool)
            dt = float(dt_s)
            ok = active & mask & torch.isfinite(f).all(-1) & bool(math.isfinite(dt) and dt > 0.0)
            self.thrust_sq = torch.where(ok.unsqueeze(-1), self.thrust_sq + (f * f) * dt, self.thrust_sq)
            self.has_thrust |= ok
        # DockingEpisode.observe_physics (after super; done unchanged)
        conj, _ = self._conjunction(valid, com, qn, lin)
        drop = active & ((self.validity != _RUNNING) | dock_hit | ~valid | ~conj)
        self.hold_started = torch.where(drop, torch.full_like(self.hold_started, math.nan), self.hold_started)

    # ------------------------------------------------------------- gates --
    def _crossed(self, k, p0, p1, vel, call):
        """HarborMissionEpisode._crossed for gate k on rows ``call`` (side effect: arming)."""
        mid, nrm = self.gate_mid[:, k], self.gate_nrm[:, k]
        s0 = (p0[:, 0] - mid[:, 0]) * nrm[:, 0] + (p0[:, 1] - mid[:, 1]) * nrm[:, 1]
        s1 = (p1[:, 0] - mid[:, 0]) * nrm[:, 0] + (p1[:, 1] - mid[:, 1]) * nrm[:, 1]
        armed = self.armed[:, k]
        arming = call & ~armed
        self.armed[:, k] = torch.where(arming, s1 <= -self.gate_approach[:, k], armed)
        live = call & armed
        span = s1 - s0
        straddle = (s0 <= 0.0) & (0.0 <= s1) & (span > 1.0e-12)
        f = -s0 / torch.where(straddle, span, torch.ones_like(span))
        xi0 = p0[:, 0] + f * (p1[:, 0] - p0[:, 0])
        xi1 = p0[:, 1] + f * (p1[:, 1] - p0[:, 1])
        tx, ty = -nrm[:, 1], nrm[:, 0]
        lateral = ((xi0 - mid[:, 0]) * tx + (xi1 - mid[:, 1]) * ty).abs()
        vn = vel[:, 0] * nrm[:, 0] + vel[:, 1] * nrm[:, 1]
        return live & straddle & (lateral <= self.gate_half[:, k]) & (vn > self.gate_vmin[:, k])

    def _departure_time(self):
        complete = (self.spawn_phase == 0) & ~torch.isnan(self.gate_times).any(-1)
        return torch.where(complete, self.gate_times[:, -1], torch.full_like(self.transit_time, math.nan))

    # -------------------------------------------------------------- step --
    def step(self, time_s, state, *, collision=None, out_of_bounds=None, clearance=None, has_clearance=None,
             rows=None):
        """One control sample for ``rows`` (default all); returns (terminated, truncated, reason) [N]."""
        n = self.num_envs
        active = torch.ones(n, dtype=torch.bool) if rows is None else torch.as_tensor(rows, dtype=torch.bool)
        if bool((active & self.done).any()):
            raise RuntimeError("reset required after completion")
        t = torch.as_tensor(time_s, dtype=F64).reshape(-1).expand(n)
        bad_clock = active & ~(torch.isfinite(t) & (t > self.prev_time))
        if bool(bad_clock.any()):
            raise ValueError("sample clock must increase from the reset time zero")
        if bool((active & (t > self.deadline_s + 1.0e-9)).any()):
            raise ValueError("the adapter must include a sample exactly at the deadline")
        collision = torch.zeros(n, dtype=torch.bool) if collision is None else \
            torch.as_tensor(collision, dtype=torch.bool).reshape(-1).expand(n)
        oob = torch.zeros(n, dtype=torch.bool) if out_of_bounds is None else \
            torch.as_tensor(out_of_bounds, dtype=torch.bool).reshape(-1).expand(n)
        # ---- _observe
        self.observe_physics(state, collision=collision, clearance=clearance, has_clearance=has_clearance,
                             rows=active)
        contact = self.contact_since_step & active
        self.contact_since_step = self.contact_since_step & ~active
        self.clean = torch.where(active, self.clean & ~contact, self.clean)
        valid, com, qn, lin, _, _ = self._state_values(state)
        obs = active & valid
        self.valid_prefix_only |= active & ~valid
        p0 = self.prev_xy.clone()
        p1 = com[:, :2]
        step_len = _exact(math.hypot, p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1])
        self.full_len = torch.where(obs, self.full_len + step_len, self.full_len)
        self.prev_xy = torch.where(obs.unsqueeze(-1), p1, self.prev_xy)
        self.prev_time = torch.where(obs, t, self.prev_time)
        # ---- gates (spawn phase 0, clean, valid)
        gating = obs & (self.spawn_phase == 0) & self.clean & (self.validity == _RUNNING)
        for k in range(2):
            unset = torch.isnan(self.gate_times[:, k])
            crossed = self._crossed(k, p0, p1, lin, gating & unset)
            prior_ok = torch.ones(n, dtype=torch.bool) if k == 0 else ~torch.isnan(self.gate_times[:, k - 1])
            hit = gating & unset & crossed & prior_ok
            self.gate_times[:, k] = torch.where(hit, t, self.gate_times[:, k])
        unset_t = torch.isnan(self.transit_time)
        crossed_t = self._crossed(2, p0, p1, lin, gating & unset_t)
        dep = self._departure_time()
        transit_now = gating & unset_t & crossed_t & ~torch.isnan(dep) & (t > dep)
        self.transit_time = torch.where(transit_now, t, self.transit_time)
        # ---- entry readiness
        ready = (self.spawn_phase == 2) | ~torch.isnan(self.transit_time)
        reset_entry = active & ~ready
        self.entered = self.entered & ~reset_entry
        self.entry_time = torch.where(reset_entry, torch.full_like(self.entry_time, math.nan), self.entry_time)
        # ---- _docking_candidate (rows with obs)
        yaw = self._yaw(qn)
        local = self._local(com[:, 0], com[:, 1])
        prev_local = self.prev_local
        dx = local[:, 0] - prev_local[:, 0]
        opening, half = self.berth[:, 5], self.berth[:, 7]
        cross = obs & ~self.entered & (prev_local[:, 0] < opening) & (opening <= local[:, 0]) & (dx > 0.0)
        frac = (opening - prev_local[:, 0]) / torch.where(cross, dx, torch.ones_like(dx))
        y_cross = prev_local[:, 1] + frac * (local[:, 1] - prev_local[:, 1])
        rel = self._wrapped(yaw - self.berth[:, 2])
        sin_rel, cos_rel = _exact(math.sin, rel), _exact(math.cos, rel)
        half_ext = torch.zeros(n, dtype=F64)
        for vx, vy in self.hull:
            half_ext = torch.maximum(half_ext, (sin_rel * vx + cos_rel * vy).abs())
        inward = lin[:, 0] * self.berth[:, 3] + lin[:, 1] * self.berth[:, 4]
        enter = cross & self.clean & (inward > 0.0) & (cos_rel > 0.0) & (y_cross.abs() + half_ext <= half)
        self.entered = self.entered | enter
        entry_t = self.prev_local_time + frac * (t - self.prev_local_time)
        self.entry_time = torch.where(enter, entry_t, self.entry_time)
        self.prev_local = torch.where(obs.unsqueeze(-1), local, self.prev_local)
        self.prev_local_time = torch.where(obs, t, self.prev_local_time)
        conj_raw, geo = self._conjunction(valid, com, qn, lin)
        conj = obs & self.clean & conj_raw
        start_hold = conj & torch.isnan(self.hold_started)
        self.hold_started = torch.where(start_hold, t, self.hold_started)
        self.hold_started = torch.where(obs & ~conj, torch.full_like(self.hold_started, math.nan),
                                        self.hold_started)
        dwell = torch.where(conj, t - self.hold_started, torch.zeros_like(t))
        candidate = conj & (dwell >= self.tol[:, 3] - 1.0e-9)
        order_ok = (~torch.isnan(self.transit_time) & ~torch.isnan(self.entry_time)
                    & (self.transit_time <= self.entry_time))
        candidate = candidate & ((self.spawn_phase == 2) | order_ok)
        # ---- _finish
        safe = (self.validity == _RUNNING) & self.clean & ~oob
        latch = active & candidate & safe & ~self.success
        self.success = self.success | latch
        first = latch & torch.isnan(self.tts)
        self.tts = torch.where(first, t, self.tts)
        self.prefix_len = torch.where(first, self.full_len, self.prefix_len)
        timeout = t >= self.deadline_s - 1.0e-9
        reason = torch.zeros(n, dtype=torch.long)
        reason = torch.where(timeout, int(contracts.TerminationReason.TIMEOUT), reason)
        reason = torch.where(self.success, int(contracts.TerminationReason.SUCCESS), reason)
        reason = torch.where(contact, int(contracts.TerminationReason.COLLISION), reason)
        reason = torch.where(oob, int(contracts.TerminationReason.OUT_OF_BOUNDS), reason)
        reason = torch.where(self.validity != _RUNNING, self.validity, reason)
        reason = torch.where(active, reason, torch.zeros_like(reason))
        terminated = active & (reason != _RUNNING) & (reason != int(contracts.TerminationReason.TIMEOUT))
        truncated = active & (reason == int(contracts.TerminationReason.TIMEOUT))
        finished = terminated | truncated
        keep = (reason == int(contracts.TerminationReason.SUCCESS)) | (reason == int(contracts.TerminationReason.TIMEOUT))
        self.success = torch.where(finished & ~keep, torch.zeros_like(self.success), self.success)
        self.done = self.done | finished
        self.dwell = torch.where(active, dwell, self.dwell)
        self.success_now = active & latch
        for i in finished.nonzero().flatten().tolist():
            self.last_completed[i] = self._result(i, obs, geo, dwell, reason)
        return terminated, truncated, reason

    # ------------------------------------------------------------ export --
    @staticmethod
    def _opt(value):
        return None if math.isnan(value) else value

    def _result(self, i, obs, geo, dwell, reason):
        spawn0 = int(self.spawn_phase[i]) == 0
        entered = bool(self.entered[i])
        entry = self._opt(float(self.entry_time[i]))
        if bool(obs[i]):
            extras = dict(docking_position_error_m=float(geo["dist"][i]),
                          docking_heading_error_rad=float(geo["heading_err"][i]),
                          docking_planar_speed_mps=float(geo["speed"][i]),
                          docking_longitudinal_error_m=float(geo["local"][i, 0]),
                          docking_lateral_error_m=float(geo["local"][i, 1]),
                          berth_contained=bool(geo["contained"][i]), berth_entry_latched=entered,
                          berth_entry_time_s=entry, docking_valid_dwell_s=float(dwell[i]))
        else:
            extras = dict(docking_position_error_m=None, docking_heading_error_rad=None,
                          docking_planar_speed_mps=None, docking_longitudinal_error_m=None,
                          docking_lateral_error_m=None, berth_contained=False, berth_entry_latched=entered,
                          berth_entry_time_s=entry, docking_valid_dwell_s=0.0)
        gate_times = tuple(self._opt(float(v)) for v in self.gate_times[i])
        dep = self._opt(float(self._departure_time()[i]))
        transit = self._opt(float(self.transit_time[i]))
        extras.update(departure_gate_times_s=gate_times if spawn0 else None, departure_time_s=dep,
                      transit_time_s=transit, stage_departure_complete=dep is not None,
                      stage_transit_complete=transit is not None,
                      contract_scope=CONTRACT_SCOPES[int(self.spawn_phase[i])], isolated_phase=not spawn0)
        extras.update(path_cross_track_error_integral_m_s=0.0,
                      path_length_valid_prefix_only=bool(self.valid_prefix_only[i]),
                      min_clearance_m=float(self.min_clearance[i]), nonfinite_seen=bool(self.nonfinite_seen[i]),
                      deadline_s=float(self.deadline_s[i]),
                      integrated_squared_thrust_n2s=(tuple(float(v) for v in self.thrust_sq[i])
                                                     if bool(self.has_thrust[i]) else None))
        success = bool(self.success[i])
        return contracts.EpisodeResult(success, float(self.tts[i]) if success else math.nan,
                                       float(self.prefix_len[i]) if success else math.nan,
                                       float(self.full_len[i]), int(reason[i]), extras)

    # ------------------------------------------------------ task progress --
    def stage(self):
        """Stage id [N]: 0 departure, 1 transit, 2 docking, 3 complete (success latched)."""
        dep = ~torch.isnan(self._departure_time())
        transit = ~torch.isnan(self.transit_time)
        stage = torch.where(dep, torch.ones_like(self.spawn_phase), torch.zeros_like(self.spawn_phase))
        stage = torch.where(transit | (self.spawn_phase == 2), torch.full_like(stage, 2), stage)
        return torch.where(self.success, torch.full_like(stage, 3), stage)

    def gate_passed(self):
        """Bool [N,4]: departure_1, departure_2, transit_exit (latched times) and berth_mouth (entry latch)."""
        return torch.cat((~torch.isnan(self.gate_times), (~torch.isnan(self.transit_time)).unsqueeze(-1),
                          self.entered.unsqueeze(-1)), -1)


# ------------------------------------------------------------ phase targets --
def phase_targets(oracle, gate_midpoints, gate_normals, gate_half_widths, gate_approach_m, gate_min_speed_mps,
                  static_centers, static_radii, static_mask, hull_vertices, contract_scope, time_s, device):
    """The F7 phase-target interface (see PHASE_TARGETS_F6.md; env-local COM frame)."""
    n = oracle.num_envs
    stage = oracle.stage()
    passed = oracle.gate_passed()
    applicable = torch.ones((n, 4), dtype=torch.bool)
    applicable[:, :3] = (oracle.spawn_phase == 0).unsqueeze(-1)
    passed_or_na = passed | ~applicable
    remaining = ~passed_or_na
    first_open = torch.where(remaining.any(-1), remaining.float().argmax(-1),
                             torch.full((n,), NO_GATE, dtype=torch.long))
    berth = oracle.berth
    berth_pose = torch.stack((berth[:, 0], berth[:, 1], berth[:, 2]), -1)
    dock_dir = torch.stack((berth[:, 3], berth[:, 4]), -1)
    mid = torch.as_tensor(gate_midpoints, dtype=F64)
    nrm = torch.as_tensor(gate_normals, dtype=F64)
    widths = torch.as_tensor(gate_half_widths, dtype=F64)
    rows = torch.arange(n)
    idx = first_open.clamp(max=3)
    none = first_open == NO_GATE
    active_mid = torch.where(none.unsqueeze(-1), berth_pose[:, :2], mid[rows, idx])
    active_nrm = torch.where(none.unsqueeze(-1), dock_dir, nrm[rows, idx])
    active_half = torch.where(none, torch.zeros(n, dtype=F64), widths[rows, idx])
    to = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
    return {
        "schema": PHASE_TARGETS_SCHEMA, "frame": "env_local_com_xy", "stage_names": STAGE_NAMES,
        "gate_names": GATE_NAMES, "contract_scope": contract_scope,
        "stage": to(stage), "active_gate_index": to(first_open),
        "active_gate_midpoint": to(active_mid), "active_gate_normal": to(active_nrm),
        "active_gate_half_width_m": to(active_half),
        "gate_midpoints": to(mid), "gate_normals": to(nrm), "gate_half_widths_m": to(widths),
        "gate_approach_m": to(torch.as_tensor(gate_approach_m, dtype=F64)),
        "gate_min_normal_speed_mps": to(torch.as_tensor(gate_min_speed_mps, dtype=F64)),
        "gate_applicable": to(applicable), "gate_passed": to(passed & applicable),
        "remaining_gates": to(remaining),
        "berth_pose": to(berth_pose),
        "berth_interior": to(torch.stack((berth[:, 5], berth[:, 6], berth[:, 7]), -1)),
        "berth_tolerances": {"position_m": to(oracle.tol[:, 0].clone()), "heading_rad": to(oracle.tol[:, 1].clone()),
                             "speed_mps": to(oracle.tol[:, 2].clone()), "hold_s": to(oracle.tol[:, 3].clone())},
        "dwell_s": to(oracle.dwell.clone()), "deadline_s": to(oracle.deadline_s.clone()),
        "time_s": to(torch.as_tensor(time_s, dtype=F64).clone()),
        "static_obstacles": {"centers": to(static_centers), "radii": to(static_radii), "mask": to(static_mask)},
        "hull_polygon_m": to(torch.as_tensor(hull_vertices, dtype=F64)),
    }


# ------------------------------------------------------------ D11 hashes --
def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def family_scenario_hashes(layout, spawn, spawn_phase, datum_z_m):
    """Exogenous family primitives of one episode (pairing.FAMILY_PRIMITIVES['harbor_mission'])."""
    gates = gate_table(layout)
    berth = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(layout.extras["berth"]).items()}
    return {
        "layout": _sha({"geometry_hash": layout.geometry_hash, "tier": int(layout.tier), "vessel": layout.vessel,
                        "field_indices": list(layout.field_indices),
                        "structural_indices": list(layout.structural_indices)}),
        "route": _sha({"gates": list(GATE_NAMES), "midpoints": gates["midpoints"].tolist(),
                       "normals": gates["normals"].tolist(), "half_widths": gates["half_widths"].tolist(),
                       "approach_m": gates["approach_m"].tolist(),
                       "min_normal_speed_mps": gates["min_normal_speed_mps"].tolist(),
                       "witness": np.asarray(layout.witness.vertices_m, dtype=np.float64).tolist(),
                       "nominal_route_length_m": float(layout.nominal_route_length_m),
                       "deadline_steps": int(layout.deadline_steps)}),
        "berth": _sha(berth),
        "spawn_pose": _sha({"xy": [float(spawn[0]), float(spawn[1])], "heading_rad": float(spawn[2]),
                            "spawn_phase": int(spawn_phase), "com_datum_z_m": float(datum_z_m)}),
    }


__all__ = ["TASK", "D7_TASK", "LAYOUT_GROUP", "GATE_NAMES", "STAGE_NAMES", "NO_GATE", "CONTRACT_SCOPES",
           "PHASE_TARGETS_SCHEMA", "STAGE_SITES", "admit_harbour_grid", "pool_meta", "layout_rng", "journal_layout", "gate_table",
           "spawn_pose", "oracle_row", "stack_rows", "scalar_oracle", "d7_route", "d7_task_context", "footprint_pose_clearance",
           "footprint_swept_clearance", "HarborMissionTensorOracle", "phase_targets", "family_scenario_hashes"]
