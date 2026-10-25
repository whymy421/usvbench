"""Journal (task contract version 2) helpers of the path-hazard family (round F, task F2).

Pure torch/numpy; never imports Isaac Lab. Used by the journal branch of
``path_hazard_env.PathHazardEnv`` (only when ``cfg.journal_contract_version
== 2``), by the offline D7 admission tool and by the CPU tests.

Contents
--------
* ``journal_layout``: the E5 layout of one (vessel, tier, eval seed, env,
  episode), drawn from ``journal_layouts_v2.build_layout`` with a numpy
  Generator keyed by the scenario protocol (group ``journal_layout``). It is
  THE scenario of that key, always (D16: no layout redraw, no attempt
  streams; a key that D7 rejects keeps its layout and runs at D0). The
  legacy rejection sampler ``path_hazard_geometry.sample_layout`` (C1 finding
  32: uninitialised centres) is never used on this path.
* ``d7_context``: the D7 ``TaskContext`` of a journal layout (E4 dataclasses;
  envelope from the D5 registry exactly as ``journal_env_mixin`` builds it).
* ``exact_hypot``: a torch port of CPython's ``math.hypot`` (vector_norm with
  compensated summation and the differential correction) so that the tensor
  twin reproduces the scalar oracle bit for bit.
* ``OrderedPathTensorOracle``: the vectorised twin of E5's scalar
  ``OrderedPathEpisode`` (journal_contracts_v2.py:819-854) including the
  ``EpisodeBase`` validity/contact/metric/termination machinery
  (:354-448). The scalar oracle stays the reference in the tests.
* ``ThreadingTensorTwin``: the vectorised twin of E5's ``ThreadingPassage``
  (journal_contracts_v2.py:857-912): one portal per gate (entry/exit planes at
  -/+ gate radius along the incoming segment), paid only after the full hull
  crossed the exit plane cleanly (C1 finding 45); it drives the journal gate
  bonus.
* ``HullRectangle`` + ``pose_clearance`` / ``swept_clearance``: the D4
  collision model (COM-relative convex hull polygon, contact at clearance
  <= 0, swept every physics substep; TASK_CONTRACTS_v2 section 2.4) for the
  rectangular v2 hull polygons, vectorised.
* ``PathHazardD7Pool`` + ``save_pool_dose`` / ``load_pool_dose``: a
  tagged-JSON codec of the E4 ``AdmissionPool`` (only dataclasses of the D7
  package are decoded; every candidate/witness digest and the pool digest are
  recomputed on load, fail closed). Per key the pool holds either the
  admitted ladder or an explicit not-admitted record with E4's rejection
  reasons (D16); ``NotAdmittedBridge`` runs not-admitted rows at D0 (no
  intruder) and stamps them ``exposure = "not_admitted"``. The pool's offered
  keys form one full grid (``grid``: eval seed x num_envs x episodes); an
  episode index at or past the grid's episodes per env (same seed, group,
  env inside the grid) is a not-admitted key with the single reason
  ``beyond_offered_grid`` (CONTEXT_R4 D24): the evaluators' trailing
  autoreset episodes run at D0 and are never counted.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import gzip
import hashlib
import importlib
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
from .._shared.conditions import pairing

JOURNAL_TASK = "path_hazard"
FAMILY = "path_hazard"
LAYOUT_GROUP = "journal_layout"          # scenario-protocol group of the journal layout stream
LAYOUT_MAX_ATTEMPTS = 200                # journal_layouts_v2.build_layout default budget
SWEEP_SUBDIVISIONS = 4                   # fixed bisection depth 2 of the swept clearance bound
DISTANCE_NORM_FACTOR = 1.25              # legacy 20 m normaliser / 16 m segment (x length scale)
POOL_FORMAT = "usvbench.path_hazard.d7_pool/3"   # /3: D16 not-admitted records, no layout attempts
POOL_MANIFEST_FORMAT = "usvbench.path_hazard.d7_pool_manifest/2"
POOL_DIR = Path(__file__).resolve().parent / "journal_d7"
POOL_MANIFEST = POOL_DIR / "d7_pool_manifest.json"   # hash pins + not-admitted list (committed)
POOL_DIR_ENV = "USVBENCH_D7_POOL_DIR"    # RUNTIME variable (F7 allow-list): folder holding the pool files
NOT_ADMITTED = "not_admitted"            # D16 exposure value of a key E4 rejected
ADMITTED = "admitted"
BEYOND_OFFERED_GRID = "beyond_offered_grid"   # D24 reason: episode index >= the grid's episodes per env
# D24 reasons of a beyond-grid key in the family's (dose, reason, sampled_states) structure: one
# entry; dose 0 because the key is admitted at no dose and runs at D0 (the record is then the same
# at every suite dose); 0 sampled states because E4 never evaluated it.
BEYOND_OFFERED_GRID_REASONS = ((0, BEYOND_OFFERED_GRID, 0),)
ASSETS_V2_DIR = Path(__file__).resolve().parents[2] / "assets_v2"   # D5 v2 assets (hash-checked)


def _dynamic():
    from .._shared.conditions import dynamic
    return dynamic


# ------------------------------------------------------------------ layouts --
def journal_layout(vessel, tier, eval_seed, env_index, episode_index, *,
                   protocol_version=scenario_rng.SCENARIO_PROTOCOL_VERSION):
    """The E5 journal layout of one scenario key (raises JournalAdmissionError).

    One keyed stream (group ``journal_layout``), no attempt index: D7 admission
    never changes the layout of a key (D16).
    """
    if isinstance(eval_seed, bool) or not isinstance(eval_seed, int):
        raise ValueError("journal layouts need an integer evaluation seed")
    if int(episode_index) < 0:
        raise ValueError("episode index must be non-negative (reset the scenario counter first)")
    rng = scenario_rng.numpy_generator(int(protocol_version), int(eval_seed), int(env_index),
                                       int(episode_index), LAYOUT_GROUP)
    layout = layouts.build_layout(JOURNAL_TASK, int(tier), vessel, rng, max_attempts=LAYOUT_MAX_ATTEMPTS)
    check_layout(layout)
    return layout


def check_layout(layout):
    """Structural facts the env relies on (fail closed)."""
    p = layout.params
    if layout.task != JOURNAL_TASK:
        raise ValueError("not a path-hazard journal layout")
    if len(layout.waypoints) != int(p["segment_count"]):
        raise ValueError("journal path hazard has one gate per segment")
    if layout.obstacle_count != int(p["requested_count"]):
        raise ValueError("journal field count must be exact (count_mode exact)")
    if layout.obstacle_count > plans.STATIC_CAPACITY[JOURNAL_TASK]:
        raise ValueError("layout exceeds the journal static capacity")
    if not np.allclose(layout.start, 0.0):
        raise ValueError("journal path-hazard routes start at the env origin")
    return layout


def route_points(layout):
    """(start, gate 1..G) as a (G+1, 2) float64 array (the scalar oracle's route)."""
    return np.vstack((np.asarray(layout.start, dtype=np.float64)[None],
                      np.asarray(layout.waypoints, dtype=np.float64)))


def reference_axis_rad(layout):
    """Task reference axis = heading of the first route segment (env-local)."""
    pts = route_points(layout)
    d = pts[1] - pts[0]
    return float(math.atan2(float(d[1]), float(d[0])))


def family_scenario_hashes(layout, spawn_com_z_m):
    """Exogenous family primitives (pairing.FAMILY_PRIMITIVES['path_hazard']); no trajectory values."""
    p = layout.params
    layout_content = {"geometry_hash": layout.geometry_hash, "tier": int(layout.tier),
                      "vessel": layout.vessel,
                      "centers_m": [[float(v) for v in row] for row in np.asarray(layout.centers)],
                      "radii_m": [float(v) for v in np.asarray(layout.radii)],
                      "bounds_m": [float(v) for v in layout.bounds_m],
                      "deadline_s": float(layout.deadline_s)}
    route_content = {"waypoints_m": [[float(a), float(b)] for a, b in layout.waypoints],
                     "gate_radius_m": float(p["gate_radius_m"]),
                     "d7_site_index": int(layout.d7_hook["site_index"])}
    spawn_content = {"start_m": [float(v) for v in np.asarray(layout.start)],
                     "heading_rad": float(layout.start_heading_rad),
                     "com_z_m": float(spawn_com_z_m), "speed_mps": 0.0}
    hashes = {"layout": pairing.primitive_hash("layout", layout_content),
              "route": pairing.primitive_hash("route", route_content),
              "spawn_pose": pairing.primitive_hash("spawn_pose", spawn_content)}
    return pairing.validate_family_hashes(hashes, FAMILY)


# ----------------------------------------------------------------- D7 context --
def journal_envelope(vessel):
    """The D7 envelope exactly as journal_env_mixin.journal_setup builds it."""
    dynamic = _dynamic()
    record = plans.vessel_record(vessel)
    envelope, _ = dynamic.envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel),
                                                reference_speed_mps=record.cruise_speed_m_s,
                                                turning_radius_m=record.turning_radius_design_m)
    return envelope


NOMINAL_VERTEX_RADIUS_FACTOR = 0.25     # D7 nominal vertex circle radius / declared turning radius R


def _segment_clear(p, q, centers, radii, margin):
    """True if segment p->q keeps every disk surface at least ``margin`` away."""
    if not len(radii):
        return True
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    d = q - p
    n = float(d @ d)
    rel = centers - p
    t = np.clip((rel @ d) / n, 0.0, 1.0) if n > 0.0 else np.zeros(len(radii))
    gap = np.linalg.norm(rel - t[:, None] * d[None, :], axis=1) - radii
    return bool(np.all(gap >= margin))


def d7_nominal_route(layout):
    """D7 nominal route of a journal layout: (circles, mandatory flags, site_index).

    E5's d7_hook routes the D7 nominal plan straight from gate to gate, which
    runs through the designated line blockers of this task (every such key is
    rejected as ``nominal_prefix_unsafe``). The hook also exports the D4
    witness polyline (``witness_vertices``: collision-free, through every gate;
    TASK_CONTRACTS_v2 section 2.10 "witness waypoints"), so the nominal plan
    follows it. Inside each leg (gate to gate) the polyline is shortcut
    greedily: a vertex is skipped when the chord keeps every obstacle surface
    at least (hull circumscribed radius + planning margin) away, which leaves
    fewer, longer straight legs for D7's unforced TTC windows. Each kept
    interior vertex becomes a non-mandatory circle of radius R / 4 (R = the
    declared turning radius; a much smaller circle makes the bounded-yaw-rate
    nominal unicycle orbit it); each gate keeps its mandatory gate disk
    followed by an R / 4 circle at its centre (the plan continues to the gate
    centre instead of turning at the disk edge). The goal stays hook['goal'].
    The preassigned segment (hook ``site_index``, the leg between gates
    site_index and site_index + 1) maps to the longest straight sub-leg of
    that leg, so D7's ``leg_midpoint`` rule places the conflict on the
    mandatory segment where an unforced straight window can exist.
    """
    hook = layout.d7_hook
    p = layout.params
    scale = float(p["length_scale"])
    small = NOMINAL_VERTEX_RADIUS_FACTOR * float(p["turning_radius_m"])
    record = plans.vessel_record(layout.vessel)
    margin = float(record.circumscribed_radius_m) + float(p["planning_margin_m"])
    centers = np.asarray(layout.centers, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(layout.radii, dtype=np.float64)
    gates = [(float(c[0]), float(c[1]), float(c[2])) for c in hook["route"]]
    verts = [(float(v[0]), float(v[1])) for v in hook["witness_vertices"]]
    goal = hook["goal"]
    start_gap = math.hypot(verts[0][0] - hook["initial_pose"][0], verts[0][1] - hook["initial_pose"][1])
    goal_gap = math.hypot(verts[-1][0] - goal[0], verts[-1][1] - goal[1])
    if start_gap > 1e-6 * scale or goal_gap > 1e-6 * scale:
        raise ValueError("witness polyline must run from the start to the goal")
    anchors, gi = [0], 0
    for j, (x, y) in enumerate(verts[1:-1], start=1):
        if gi < len(gates) and math.hypot(x - gates[gi][0], y - gates[gi][1]) <= 1e-6 * scale:
            anchors.append(j)
            gi += 1
    if gi != len(gates):
        raise ValueError("every journal gate must be a vertex of the D4 witness polyline")
    anchors.append(len(verts) - 1)
    kept_legs = []
    for first, last in zip(anchors, anchors[1:]):
        keep, i = [first], first
        while i < last:
            j = last
            while j > i + 1 and not _segment_clear(verts[i], verts[j], centers, radii, margin):
                j -= 1
            keep.append(j)
            i = j
        kept_legs.append(keep)
    circles, flags, target_of_vertex = [], [], {}
    for leg_index, keep in enumerate(kept_legs):
        for j in keep[1:]:
            if j == len(verts) - 1:
                target_of_vertex[j] = None            # the goal (appended after the route by D7)
            elif leg_index < len(gates) and j == keep[-1]:
                target_of_vertex[j] = len(circles)
                circles.append(gates[leg_index])
                flags.append(True)
                circles.append((gates[leg_index][0], gates[leg_index][1], small))
                flags.append(False)
            else:
                target_of_vertex[j] = len(circles)
                circles.append((verts[j][0], verts[j][1], small))
                flags.append(False)
    target_of_vertex[len(verts) - 1] = len(circles)
    keep = kept_legs[int(hook["site_index"])]
    lengths = [math.hypot(verts[b][0] - verts[a][0], verts[b][1] - verts[a][1]) for a, b in zip(keep, keep[1:])]
    best = max(range(len(lengths)), key=lambda k: (lengths[k], -k))
    return tuple(circles), tuple(flags), int(target_of_vertex[keep[best + 1]])


def d7_context(layout, envelope=None):
    """TaskContext of an admitted journal layout (env-local SI units; D7 B05)."""
    dynamic = _dynamic()
    hook = layout.d7_hook
    envelope = journal_envelope(layout.vessel) if envelope is None else envelope
    declared = hook["envelope"]
    fp = envelope.footprint
    if declared["vessel_id"] != envelope.vessel_id or not math.isclose(
            declared["reference_speed_mps"], envelope.reference_speed_mps, rel_tol=1e-12):
        raise ValueError("layout d7_hook envelope disagrees with the registry envelope")
    for key, value in (("x_min_m", fp.x_min_m), ("x_max_m", fp.x_max_m), ("y_min_m", fp.y_min_m),
                       ("y_max_m", fp.y_max_m)):
        if not math.isclose(declared[key], value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("layout footprint disagrees with the D7 envelope footprint")
    sensor = hook["sensor"]
    x, y, yaw = hook["initial_pose"]
    circle = lambda c: dynamic.Circle(float(c[0]), float(c[1]), float(c[2]))
    route, flags, site = d7_nominal_route(layout)
    return dynamic.TaskContext(
        task=hook["task"], envelope=envelope, initial_pose=dynamic.Pose(float(x), float(y), float(yaw), 0.0),
        bounds=tuple(float(v) for v in hook["bounds"]), goal=circle(hook["goal"]),
        deadline_s=float(hook["deadline_s"]), route=tuple(circle(c) for c in route),
        route_mandatory=flags,
        static_obstacles=tuple(circle(c) for c in hook["static_obstacles"]),
        sensor=dynamic.SensorSpec(kind="ray_fan", ray_count=int(sensor["ray_count"]),
                                  max_range_m=float(sensor["max_range_m"]),
                                  source="journal 36-ray fan, 30 m x length scale (journal_cfg_v2 plan, D14)"),
        site_index=site, base_max=int(hook["base_max"]), base_count=int(hook["base_count"]),
        geometry_id=str(hook["geometry_id"]))


def appearing_admission_inputs(vessel, tier, seed, env_index, episode_index):
    """G20: (D7 TaskContext, length scale, radius range) of one key's REAL journal layout.

    The appearing admission places its cylinder on the same frozen geometry the env builds (D16).
    Radius rule 4: the task's own field-obstacle radius range (journal_cfg_v2 radius_range_m
    = [0.8 s, 1.5 s]), as scatter uses its own.
    """
    layout = journal_layout(vessel, int(tier), int(seed), int(env_index), int(episode_index))
    params = layout.params
    return d7_context(layout), float(params["length_scale"]), tuple(float(v) for v in params["radius_range_m"])


def episode_key(eval_seed, env_index, episode_index):
    return _dynamic().EpisodeKey(int(eval_seed), int(env_index), int(episode_index), "d7_" + JOURNAL_TASK)


# ------------------------------------------------------------------ geometry --
_SPLIT = 134217729.0        # 2**27 + 1 (Veltkamp split of a double)
_DBL_MIN = 2.2250738585072014e-308


def _two_product(a, b):
    """Error-free product (Dekker): a*b = p + e exactly (no over/underflow)."""
    p = a * b
    t = _SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = _SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _fast_two_sum(a, b):
    x = a + b
    z = x - a
    return x, b - z


def _vector_norm2(ax, ay, m):
    """CPython vector_norm for two finite nonnegative coordinates with max m > 0 normal."""
    _, exponent = torch.frexp(m)
    scale = torch.ldexp(torch.ones_like(m), -exponent)
    csum = torch.ones_like(m)
    frac1 = torch.zeros_like(m)
    frac2 = torch.zeros_like(m)
    for v in (ax, ay):
        x = v * scale
        hi, lo = _two_product(x, x)
        csum, s_lo = _fast_two_sum(csum, hi)
        frac1 = frac1 + lo
        frac2 = frac2 + s_lo
    h = torch.sqrt(csum - 1.0 + (frac1 + frac2))
    hi, lo = _two_product(-h, h)
    csum, s_lo = _fast_two_sum(csum, hi)
    frac1 = frac1 + lo
    frac2 = frac2 + s_lo
    x = csum - 1.0 + (frac1 + frac2)
    h = h + x / (2.0 * h)
    return h / scale


def exact_hypot(x, y):
    """Bitwise torch twin of CPython 3.13 ``math.hypot(x, y)`` (float64).

    Port of Modules/mathmodule.c vector_norm (scaled compensated sum of
    squares, one differential correction). Infinite -> inf, NaN -> nan,
    zero -> 0; subnormal maxima use the DBL_MIN rescale branch.
    """
    prepared = _exact_hypot_prepare(x, y)
    return _exact_hypot_finish(prepared, bool(prepared[-1].any()))


def _exact_hypot_prepare(x, y):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    x, y = torch.broadcast_tensors(x, y)
    ax, ay = x.abs(), y.abs()
    m = torch.maximum(ax, ay)
    finite = torch.isfinite(ax) & torch.isfinite(ay)
    positive = finite & (m > 0.0)
    safe_m = torch.where(positive, m, torch.ones_like(m))
    safe_x = torch.where(positive, ax, torch.zeros_like(ax))
    safe_y = torch.where(positive, ay, torch.zeros_like(ay))
    _, exponent = torch.frexp(safe_m)
    tiny = exponent < -1023
    return ax, ay, positive, safe_x, safe_y, safe_m, tiny


def exact_hypot_many(pairs):
    """G11: ``[exact_hypot(x, y) for x, y in pairs]`` with the per-pair subnormal checks travelling to
    the host in ONE transfer (each pair runs exactly exact_hypot's operations; only the sync is shared)."""
    prepared = [_exact_hypot_prepare(x, y) for x, y in pairs]
    flags = torch.stack([p[-1].any() for p in prepared]).tolist()
    return [_exact_hypot_finish(p, flag) for p, flag in zip(prepared, flags)]


def _exact_hypot_finish(prepared, any_tiny):
    ax, ay, positive, safe_x, safe_y, safe_m, tiny = prepared
    if any_tiny:
        normal = _vector_norm2(torch.where(tiny, safe_x / _DBL_MIN, safe_x),
                               torch.where(tiny, safe_y / _DBL_MIN, safe_y),
                               torch.where(tiny, safe_m / _DBL_MIN, safe_m))
        out = torch.where(tiny, normal * _DBL_MIN, normal)
    else:
        out = _vector_norm2(safe_x, safe_y, safe_m)
    out = torch.where(positive, out, torch.zeros_like(out))
    out = torch.where(torch.isnan(ax) | torch.isnan(ay), torch.full_like(out, math.nan), out)
    return torch.where(torch.isinf(ax) | torch.isinf(ay), torch.full_like(out, math.inf), out)


def _py_max0_min1(v):
    """max(0.0, min(1.0, v)) with Python's first-argument tie rule."""
    v = torch.where(v < 1.0, v, torch.ones_like(v))
    return torch.where(v > 0.0, v, torch.zeros_like(v))


def cross_track_error(px, py, ax, ay, bx, by):
    """Twin of journal_contracts_v2.cross_track_error_m for ONE segment a->b."""
    return exact_hypot(*_cross_track_args(px, py, ax, ay, bx, by))


def _cross_track_args(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    raw = ((px - ax) * dx + (py - ay) * dy) / torch.where(n != 0.0, n, torch.ones_like(n))
    t = torch.where(n != 0.0, _py_max0_min1(raw), torch.zeros_like(raw))
    return px - (ax + t * dx), py - (ay + t * dy)


def segment_hits_disk(p0x, p0y, p1x, p1y, cx, cy, radius):
    """Twin of journal_contracts_v2.segment_hits_disk."""
    return exact_hypot(*_segment_disk_args(p0x, p0y, p1x, p1y, cx, cy)) <= radius


def _segment_disk_args(p0x, p0y, p1x, p1y, cx, cy):
    ax, ay = p0x - cx, p0y - cy
    dx, dy = p1x - p0x, p1y - p0y
    n = dx * dx + dy * dy
    raw = -(ax * dx + ay * dy) / torch.where(n == 0.0, torch.ones_like(n), n)
    t = torch.where(n == 0.0, torch.zeros_like(raw), _py_max0_min1(raw))
    return ax + t * dx, ay + t * dy


def yaw_from_quat(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_rotate_inverse(quat, v):
    """Rotate world vectors into the body frame (unit quaternion wxyz)."""
    qw = quat[..., :1]
    qv = quat[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v - qw * t + torch.cross(qv, t, dim=-1)


def body_rates(quat, linear_velocity_w, angular_velocity_w):
    """[surge, sway, yaw rate] of the hull from the FULL 3-D transform (D3 section 4)."""
    v = quat_rotate_inverse(quat, linear_velocity_w)
    w = quat_rotate_inverse(quat, angular_velocity_w)
    return torch.stack((v[..., 0], v[..., 1], w[..., 2]), dim=-1)


class HullRectangle:
    """Axis-aligned COM-relative hull rectangle of a v2 vessel (D4 primary footprint)."""

    def __init__(self, vertices_m):
        v = np.asarray(vertices_m, dtype=np.float64)
        if v.shape != (4, 2) or not np.all(np.isfinite(v)):
            raise ValueError("journal path hazard supports the 4-vertex rectangular v2 hull polygon")
        xs, ys = sorted(set(v[:, 0].tolist())), sorted(set(v[:, 1].tolist()))
        if len(xs) != 2 or len(ys) != 2 or {(a, b) for a, b in v.tolist()} != {(a, b) for a in xs for b in ys}:
            raise ValueError("hull polygon is not an axis-aligned rectangle; extend pose_clearance first")
        self.x_min, self.x_max = xs
        self.y_min, self.y_max = ys
        if not (self.x_min < 0.0 < self.x_max and self.y_min < 0.0 < self.y_max):
            raise ValueError("the COM must lie inside the hull rectangle")
        self.core_radius_m = float(np.max(np.linalg.norm(v, axis=1)))

    @classmethod
    def for_vessel(cls, vessel):
        return cls(plans.vessel_record(vessel).geometry.footprint.vertices_m)


def pose_clearance(rect, x, y, yaw, centers, radii, mask):
    """Exact signed clearance hull rectangle vs disks; [N]; no disk -> +inf.

    x, y, yaw: [N] env-local COM pose; centers [N,K,2] env-local; radii [N,K];
    mask [N,K] bool. Non-finite poses fail closed (-inf).
    """
    x, y, yaw = (torch.as_tensor(t, dtype=torch.float64) for t in (x, y, yaw))
    centers = centers.to(torch.float64)
    radii = radii.to(torch.float64)
    if centers.shape[-2] == 0:
        return torch.full_like(x, math.inf)
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    dx = centers[..., 0] - x.unsqueeze(-1)
    dy = centers[..., 1] - y.unsqueeze(-1)
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    ox = torch.clamp(torch.maximum(rect.x_min - bx, bx - rect.x_max), min=0.0)
    oy = torch.clamp(torch.maximum(rect.y_min - by, by - rect.y_max), min=0.0)
    outside = torch.sqrt(ox * ox + oy * oy)
    depth = torch.minimum(torch.minimum(bx - rect.x_min, rect.x_max - bx),
                          torch.minimum(by - rect.y_min, rect.y_max - by))
    signed = torch.where((ox > 0.0) | (oy > 0.0), outside, -depth)
    clear = signed - radii
    clear = torch.where(mask, clear, torch.full_like(clear, math.inf))
    out = clear.min(dim=-1).values
    bad = ~(torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(yaw))
    return torch.where(bad, torch.full_like(out, -math.inf), out)


def _wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def swept_clearance(rect, start_pose, end_pose, centers, radii, mask, subdivisions=SWEEP_SUBDIVISIONS):
    """Certified lower bound of the clearance over one physics substep (static disks).

    Linear XY / shortest-angle yaw motion from start_pose to end_pose ([N,3]
    each, env-local x, y, yaw), split into ``subdivisions`` equal parts; on
    each part the midpoint clearance minus the rigid displacement bound
    (|dxy|/2 + core_radius |dyaw|/2) bounds every intermediate pose (the
    signed distance is 1-Lipschitz, journal_geometry_v2.swept_clearance). A
    stationary hull returns the exact pose clearance. Conservative: the
    bound may be up to one sub-part displacement (millimetres at 120 Hz)
    below the true clearance.
    """
    a = start_pose.to(torch.float64)
    b = end_pose.to(torch.float64)
    dyaw = _wrap(b[:, 2] - a[:, 2])
    dxy = b[:, :2] - a[:, :2]
    best = None
    m = int(subdivisions)
    for j in range(m):
        f = (j + 0.5) / m
        mid_x = a[:, 0] + f * dxy[:, 0]
        mid_y = a[:, 1] + f * dxy[:, 1]
        mid_yaw = a[:, 2] + f * dyaw
        motion = (torch.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2) / m) / 2.0 \
            + rect.core_radius_m * (dyaw.abs() / m) / 2.0
        lower = pose_clearance(rect, mid_x, mid_y, mid_yaw, centers, radii, mask) - motion
        best = lower if best is None else torch.minimum(best, lower)
    bad = ~(torch.isfinite(a).all(dim=-1) & torch.isfinite(b).all(dim=-1))
    return torch.where(bad, torch.full_like(best, -math.inf), best)


# ------------------------------------------------------------ tensor oracle --
RUNNING, SUCCESS, COLLISION, TIMEOUT, OUT_OF_BOUNDS = 0, 1, 2, 3, 4
CAPSIZE, SINKING, AIRBORNE, NONFINITE = 6, 7, 8, 9


def _rows_f64(value, like):
    """torch.as_tensor(value, float64, like.device).expand(N) without a host-to-device copy for a Python
    number (G11): torch.full writes the same float64 value on the device (a fill kernel, no sync)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.full((like.shape[0],), float(value), dtype=torch.float64, device=like.device)
    return torch.as_tensor(value, dtype=torch.float64, device=like.device).expand(like.shape[0])


def validity_codes(com, quat, lin, ang, water, nominal, limits):
    """Twin of journal_contracts_v2.validity_reason (codes 0, 6, 7, 8, 9); float64."""
    com, quat, lin, ang = (t.to(torch.float64) for t in (com, quat, lin, ang))
    water = _rows_f64(water, com)
    nominal = _rows_f64(nominal, com)
    norm = torch.sqrt(((quat[:, 0] * quat[:, 0] + quat[:, 1] * quat[:, 1]) + quat[:, 2] * quat[:, 2])
                      + quat[:, 3] * quat[:, 3])
    finite = (torch.isfinite(com).all(-1) & torch.isfinite(quat).all(-1) & torch.isfinite(lin).all(-1)
              & torch.isfinite(ang).all(-1) & torch.isfinite(water) & torch.isfinite(nominal)
              & torch.isfinite(norm) & (norm > 1.0e-12))
    safe_norm = torch.where(finite, norm, torch.ones_like(norm))
    qx = torch.where(finite, quat[:, 1] / safe_norm, torch.zeros_like(norm))
    qy = torch.where(finite, quat[:, 2] / safe_norm, torch.zeros_like(norm))
    tilt = torch.acos(torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0))
    heave = torch.where(finite, (com[:, 2] - water) - nominal, torch.zeros_like(norm))
    out = torch.zeros(com.shape[0], dtype=torch.long, device=com.device)
    out = torch.where(heave > limits.max_heave_m, torch.full_like(out, AIRBORNE), out)
    out = torch.where(heave < limits.min_heave_m, torch.full_like(out, SINKING), out)
    out = torch.where(tilt > limits.capsize_angle_rad, torch.full_like(out, CAPSIZE), out)
    out = torch.where(~finite, torch.full_like(out, NONFINITE), out)
    return out, finite


class OrderedPathTensorOracle:
    """Vectorised twin of E5 ``OrderedPathEpisode`` (one row per env).

    Same lifecycle as the scalar oracle: ``reset`` after installing a layout,
    ``observe_physics`` after every physics substep, ``step`` once per
    control step (returns the int64 termination code, 0 while running), and
    ``result(i)`` = the scalar ``last_completed`` EpisodeResult (immutable,
    survives reset). All arithmetic is float64 in the scalar's operation
    order (hypot via ``exact_hypot``), so codes and every metric match the
    scalar reference bit for bit (tests: recorded env traces and 200
    randomized traces). Only the tilt comparison uses torch.acos, whose last
    bit may differ from math.acos; a state within one ulp of the capsize
    threshold is the only possible divergence (not observed in the tests).
    """

    stop_on_success = True

    def __init__(self, num_envs, num_gates, device, limits, motors=2):
        self.n, self.g = int(num_envs), int(num_gates)
        self.device = torch.device(device)
        self.limits = limits if isinstance(limits, contracts.ValidityLimits) else contracts.ValidityLimits(**limits)
        self.motors = int(motors)
        n, g, dev, f64 = self.n, self.g, self.device, torch.float64
        self.route = torch.zeros((n, g + 1, 2), dtype=f64, device=dev)
        self.gate_radius = torch.ones(n, dtype=f64, device=dev)
        self.deadline = torch.ones(n, dtype=f64, device=dev)
        self.done = torch.ones(n, dtype=torch.bool, device=dev)       # never reset -> refuses steps
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.clean = torch.ones(n, dtype=torch.bool, device=dev)
        self.contact_since = torch.zeros(n, dtype=torch.bool, device=dev)
        self.validity = torch.zeros(n, dtype=torch.long, device=dev)
        self.nonfinite_seen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_xy = torch.zeros((n, 2), dtype=f64, device=dev)
        self.prev_time = torch.zeros(n, dtype=f64, device=dev)
        self.full_len = torch.zeros(n, dtype=f64, device=dev)
        self.prefix_len = torch.full((n,), math.nan, dtype=f64, device=dev)
        self.tts = torch.full((n,), math.nan, dtype=f64, device=dev)
        self.xte_int = torch.zeros(n, dtype=f64, device=dev)
        self.prev_xte = torch.zeros(n, dtype=f64, device=dev)
        self.valid_prefix_only = torch.zeros(n, dtype=torch.bool, device=dev)
        self.min_clear = torch.full((n,), math.inf, dtype=f64, device=dev)
        self.thrust_sq = torch.zeros((n, self.motors), dtype=f64, device=dev)
        self.thrust_seen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.next_gate = torch.ones(n, dtype=torch.long, device=dev)
        self.gate_times = torch.full((n, g), math.nan, dtype=f64, device=dev)
        self.last_completed = [None] * n
        self.last_code_list = [RUNNING] * n                                 # host copy of the last step's codes
        self.last_contact = torch.zeros(n, dtype=torch.bool, device=dev)   # contact of the last step

    # ------------------------------------------------------------ helpers --
    def _xte_active(self, px, py, gate_index):
        return exact_hypot(*self._xte_active_args(px, py, gate_index))

    def _xte_active_args(self, px, py, gate_index):
        idx = gate_index.clamp(1, self.g)
        rows = torch.arange(self.n, device=self.device)
        a = self.route[rows, idx - 1]
        b = self.route[rows, idx]
        return _cross_track_args(px, py, a[:, 0], a[:, 1], b[:, 0], b[:, 1])

    # -------------------------------------------------------------- reset --
    def reset(self, env_ids, route_xy, gate_radius_m, deadline_s, com_xy):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        route = torch.as_tensor(route_xy, dtype=torch.float64, device=self.device).reshape(-1, self.g + 1, 2)
        radius = torch.as_tensor(gate_radius_m, dtype=torch.float64, device=self.device).expand(ids.numel())
        deadline = torch.as_tensor(deadline_s, dtype=torch.float64, device=self.device).expand(ids.numel())
        start = torch.as_tensor(com_xy, dtype=torch.float64, device=self.device).reshape(-1, 2)
        if route.shape[0] != ids.numel() or start.shape[0] != ids.numel():
            raise ValueError("one route and one start per reset env")
        if not bool(torch.isfinite(route).all()) or not bool(torch.isfinite(start).all()):
            raise ValueError("malformed or non-finite vector")
        if not bool(((radius > 0.0) & torch.isfinite(radius)).all()):
            raise ValueError("path needs a start, gates, and a positive radius")
        if not bool(((deadline > 0.0) & torch.isfinite(deadline)).all()):
            raise ValueError("deadline must be positive and finite")
        self.route[ids] = route
        self.gate_radius[ids] = radius
        self.deadline[ids] = deadline
        self.prev_xy[ids] = start
        self.prev_time[ids] = 0.0
        self.full_len[ids] = 0.0
        self.prefix_len[ids] = math.nan
        self.tts[ids] = math.nan
        self.xte_int[ids] = 0.0
        self.valid_prefix_only[ids] = False
        self.min_clear[ids] = math.inf
        self.validity[ids] = RUNNING
        self.nonfinite_seen[ids] = False
        self.done[ids] = False
        self.success[ids] = False
        self.clean[ids] = True
        self.contact_since[ids] = False
        self.thrust_sq[ids] = 0.0
        self.thrust_seen[ids] = False
        self.next_gate[ids] = 1
        self.gate_times[ids] = math.nan
        a, b = route[:, 0], route[:, 1]
        self.prev_xte[ids] = cross_track_error(start[:, 0], start[:, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])

    # ------------------------------------------------------------ physics --
    def observe_physics(self, com, quat, lin, ang, water, nominal, *, collision=None, clearance=None,
                        clearance_mask=None, motor_thrust=None, dt_s=None, rows=None):
        """EpisodeBase.observe_physics for every live row (``rows`` bool mask narrows it).

        ``clearance_mask`` (bool [N]) marks the rows that supply a clearance; the others behave
        like the scalar call with ``clearance_m=None`` (e.g. a row without any obstacle, whose
        clearance +inf would otherwise fail closed as non-finite).
        """
        live = ~self.done if rows is None else (~self.done & rows)
        codes, _ = validity_codes(com, quat, lin, ang, water, nominal, self.limits)
        self.nonfinite_seen |= live & (codes == NONFINITE)
        latch = live & (self.validity == RUNNING)
        self.validity = torch.where(latch, codes, self.validity)
        hit = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if collision is not None:
            hit = torch.as_tensor(collision, dtype=torch.bool, device=self.device).expand(self.n).clone()
        if clearance is not None:
            c = torch.as_tensor(clearance, dtype=torch.float64, device=self.device).expand(self.n)
            given = (torch.ones(self.n, dtype=torch.bool, device=self.device) if clearance_mask is None else
                     torch.as_tensor(clearance_mask, dtype=torch.bool, device=self.device).expand(self.n))
            fin = torch.isfinite(c)
            lower = live & given & fin & (c < self.min_clear)
            self.min_clear = torch.where(lower, c, self.min_clear)
            hit = hit | (given & (~fin | (fin & (c <= 0.0))))
        self.contact_since |= live & hit
        if motor_thrust is not None and dt_s is not None:
            dt = float(dt_s)
            f = torch.as_tensor(motor_thrust, dtype=torch.float64, device=self.device).reshape(self.n, -1)
            if f.shape[1] != self.motors:
                raise ValueError("motor thrust width differs from the oracle's motor count")
            if math.isfinite(dt) and dt > 0.0:
                ok = live & torch.isfinite(f).all(dim=-1)
                self.thrust_sq = torch.where(ok.unsqueeze(-1), self.thrust_sq + (f * f) * dt, self.thrust_sq)
                self.thrust_seen |= ok

    # --------------------------------------------------------------- step --
    def step(self, time_s, com, quat, lin, ang, water, nominal, *, collision=None, out_of_bounds=None,
             clearance=None, clearance_mask=None):
        """One control sample for EVERY row (all rows must be live); returns codes [N] int64."""
        # G11: the three entry checks travel to the host in ONE transfer (same checks, same order of the
        # errors; the clock checks have no side effect, so evaluating them before the first raise is inert)
        any_done = self.done.any()
        try:
            t = torch.as_tensor(time_s, dtype=torch.float64, device=self.device).expand(self.n)
        except Exception:
            if bool(any_done):
                raise RuntimeError("reset required after completion") from None
            raise
        checks = torch.stack((any_done, (torch.isfinite(t) & (t > self.prev_time)).all(),
                              (t <= self.deadline + 1.0e-9).all())).tolist()
        if checks[0]:
            raise RuntimeError("reset required after completion")
        if not checks[1]:
            raise ValueError("sample clock must increase from the reset time zero")
        if not checks[2]:
            raise ValueError("the adapter must include a sample exactly at the deadline")
        com = com.to(torch.float64)
        _, safe = validity_codes(com, quat, lin, ang, water, nominal, self.limits)
        px, py = com[:, 0], com[:, 1]
        # G11: the three exact_hypot evaluations of this step (active cross-track error, step length,
        # swept gate test) share ONE host transfer of their subnormal checks; their inputs (position,
        # route, next gate, previous position, gate radius) are not changed before their original use
        prev_x, prev_y = self.prev_xy[:, 0].clone(), self.prev_xy[:, 1].clone()
        rows = torch.arange(self.n, device=self.device)
        target = self.route[rows, self.next_gate.clamp(max=self.g)]
        xte, step_len, gate_distance = exact_hypot_many((
            self._xte_active_args(px, py, self.next_gate),
            (px - prev_x, py - prev_y),
            _segment_disk_args(prev_x, prev_y, px, py, target[:, 0], target[:, 1])))
        # _observe
        self.observe_physics(com, quat, lin, ang, water, nominal, collision=collision, clearance=clearance,
                             clearance_mask=clearance_mask)
        contact = self.contact_since.clone()
        self.last_contact = contact
        self.contact_since = torch.zeros_like(self.contact_since)
        self.clean &= ~contact
        self.valid_prefix_only |= ~safe
        self.full_len = torch.where(safe, self.full_len + step_len, self.full_len)
        increment = (0.5 * (self.prev_xte + xte)) * (t - self.prev_time)
        self.xte_int = torch.where(safe, self.xte_int + increment, self.xte_int)
        self.prev_xte = torch.where(safe, xte, self.prev_xte)
        self.prev_xy = torch.where(safe.unsqueeze(-1), torch.stack((px, py), -1), self.prev_xy)
        self.prev_time = torch.where(safe, t, self.prev_time)
        # ordered gate (swept COM motion between control samples)
        pending = self.next_gate < self.g + 1
        hit = safe & pending & (gate_distance <= self.gate_radius)
        slot = (self.next_gate - 1).clamp(0, self.g - 1)
        times = self.gate_times.clone()
        times[rows, slot] = torch.where(hit, t, times[rows, slot])
        self.gate_times = times
        self.next_gate = self.next_gate + hit.long()
        rexte = hit & (self.next_gate < self.g + 1)
        if bool(rexte.any()):
            self.prev_xte = torch.where(rexte, self._xte_active(px, py, self.next_gate), self.prev_xte)
        # _finish
        oob = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if out_of_bounds is None
               else torch.as_tensor(out_of_bounds, dtype=torch.bool, device=self.device).expand(self.n))
        candidate = self.next_gate == self.g + 1
        safe_fin = (self.validity == RUNNING) & self.clean & ~oob
        newly = candidate & safe_fin & ~self.success
        first = newly & torch.isnan(self.tts)
        self.tts = torch.where(first, t, self.tts)
        self.prefix_len = torch.where(first, self.full_len, self.prefix_len)
        self.success |= newly
        timeout = t >= self.deadline - 1.0e-9
        code = torch.zeros(self.n, dtype=torch.long, device=self.device)
        code = torch.where(timeout, torch.full_like(code, TIMEOUT), code)
        code = torch.where(self.success & self.stop_on_success, torch.full_like(code, SUCCESS), code)
        code = torch.where(contact, torch.full_like(code, COLLISION), code)
        code = torch.where(oob, torch.full_like(code, OUT_OF_BOUNDS), code)
        code = torch.where(self.validity != RUNNING, self.validity, code)
        finished = code != RUNNING
        self.success &= ~(finished & (code != SUCCESS) & (code != TIMEOUT))
        self.done |= finished
        # G11: one transfer of the codes (finished == code != RUNNING); the env reuses the list for its export
        code_list = code.tolist()
        self.last_code_list = code_list
        for i, value in enumerate(code_list):
            if value != RUNNING:
                self.last_completed[i] = self._result(i, int(value))
        return code

    def _result(self, i, reason):
        success = bool(self.success[i])
        gates = int(self.next_gate[i]) - 1
        times = tuple(float(v) for v in self.gate_times[i, :gates].tolist())
        details = dict(ordered_gates_completed=gates, ordered_gates_total=self.g, gate_times_s=times)
        details.update(path_cross_track_error_integral_m_s=float(self.xte_int[i]),
                       path_length_valid_prefix_only=bool(self.valid_prefix_only[i]),
                       min_clearance_m=float(self.min_clear[i]),
                       nonfinite_seen=bool(self.nonfinite_seen[i]),
                       deadline_s=float(self.deadline[i]),
                       integrated_squared_thrust_n2s=(tuple(float(v) for v in self.thrust_sq[i].tolist())
                                                      if bool(self.thrust_seen[i]) else None))
        return contracts.EpisodeResult(success, float(self.tts[i]) if success else math.nan,
                                       float(self.prefix_len[i]) if success else math.nan,
                                       float(self.full_len[i]), int(reason), details)

    def result(self, env_id):
        result = self.last_completed[int(env_id)]
        if result is None:
            raise RuntimeError(f"env {int(env_id)} has no completed episode")
        return result


# ---------------------------------------------------------- threading twin --
def portal_frames(route_xy):
    """Gate portals of an ordered route: origins = gate centres, x axis = incoming segment direction.

    route_xy: [..., G+1, 2] (start, gate 1..G). Returns (origins [..., G, 2], headings [..., G]).
    """
    route = torch.as_tensor(route_xy, dtype=torch.float64)
    d = route[..., 1:, :] - route[..., :-1, :]
    return route[..., 1:, :], torch.atan2(d[..., 1], d[..., 0])


def portal_local(com_xy, origin, heading):
    """COM in a portal frame (x along the passage, y lateral); float64."""
    c, s = torch.cos(heading), torch.sin(heading)
    dx = com_xy[..., 0] - origin[..., 0]
    dy = com_xy[..., 1] - origin[..., 1]
    return c * dx + s * dy, -s * dx + c * dy


def hull_half_extents(rect, relative_yaw):
    """Conservative (longitudinal, lateral) half extents of the hull rectangle in a frame rotated by
    ``relative_yaw`` (max |coordinate| over the four COM-relative vertices; asymmetric COM covered)."""
    c, s = torch.cos(relative_yaw), torch.sin(relative_yaw)
    hx = torch.zeros_like(relative_yaw)
    hy = torch.zeros_like(relative_yaw)
    for vx in (rect.x_min, rect.x_max):
        for vy in (rect.y_min, rect.y_max):
            hx = torch.maximum(hx, (c * vx - s * vy).abs())
            hy = torch.maximum(hy, (s * vx + c * vy).abs())
    return hx, hy


class ThreadingTensorTwin:
    """Vectorised twin of E5 ``ThreadingPassage`` (journal_contracts_v2.py:857-912), one portal
    sequence per env (portal k = gate k of the ordered route, entry plane x = -r, exit plane x = +r,
    half width r = the gate radius). Only the ACTIVE portal (the first one neither paid nor failed)
    is updated, exactly as calling every scalar passage with ``portal_id`` = the active portal's id
    (a mismatching id leaves a scalar passage untouched). ``update`` returns the bonus paid this
    control sample per env. Bitwise parity with the scalar class is test-asserted."""

    def __init__(self, num_envs, num_portals, device, bonus):
        n, g, dev, f64 = int(num_envs), int(num_portals), torch.device(device), torch.float64
        self.n, self.g, self.device, self.bonus = n, g, dev, float(bonus)
        self.entry = torch.zeros((n, g), dtype=f64, device=dev)
        self.exit = torch.zeros((n, g), dtype=f64, device=dev)
        self.half_width = torch.ones((n, g), dtype=f64, device=dev)
        self.active = torch.zeros((n, g), dtype=torch.bool, device=dev)
        self.paid = torch.zeros((n, g), dtype=torch.bool, device=dev)
        self.failed = torch.zeros((n, g), dtype=torch.bool, device=dev)
        self.worst = torch.full((n, g), math.inf, dtype=f64, device=dev)
        self.prev_x = torch.full((n, g), math.nan, dtype=f64, device=dev)
        self.prev_y = torch.full((n, g), math.nan, dtype=f64, device=dev)
        self.prev_hx = torch.full((n, g), math.nan, dtype=f64, device=dev)
        self.prev_hy = torch.full((n, g), math.nan, dtype=f64, device=dev)
        self.current = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids, gate_radius_m):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        r = torch.as_tensor(gate_radius_m, dtype=torch.float64, device=self.device).reshape(-1, 1)
        r = r.expand(ids.numel(), self.g)
        self.entry[ids] = -r
        self.exit[ids] = r
        self.half_width[ids] = r
        for name in ("active", "paid", "failed"):
            getattr(self, name)[ids] = False
        self.worst[ids] = math.inf
        for name in ("prev_x", "prev_y", "prev_hx", "prev_hy"):
            getattr(self, name)[ids] = math.nan
        self.current[ids] = 0

    def update(self, x, y, hx, hy, collision, rows=None):
        """One sample of every env's ACTIVE portal (portal-local COM x, y; hull half extents).

        ``rows`` (bool [N]) excludes rows from this sample (e.g. a non-finite state, which ends
        the episode through the validity code anyway; the scalar class would raise on it).
        """
        live = self.current < self.g
        if rows is not None:
            live = live & torch.as_tensor(rows, dtype=torch.bool, device=self.device)
        rows = torch.arange(self.n, device=self.device)
        k = self.current.clamp(max=self.g - 1)
        x, y, hx, hy = (torch.as_tensor(v, dtype=torch.float64, device=self.device) for v in (x, y, hx, hy))
        if not bool(((torch.isfinite(hx) & (hx > 0) & torch.isfinite(hy) & (hy > 0)) | ~live).all()):
            raise ValueError("hull extents must be positive")
        collision = torch.as_tensor(collision, dtype=torch.bool, device=self.device).expand(self.n)
        g = lambda t: t[rows, k]  # noqa: E731
        active, failed, paid, worst = g(self.active), g(self.failed), g(self.paid), g(self.worst)
        active0, failed0, paid0, worst0 = active, failed, paid, worst      # gathered values (never mutated)
        prev, prev_y, prev_hx, prev_hy = g(self.prev_x), g(self.prev_y), g(self.prev_hx), g(self.prev_hy)
        entry, exit_, half = g(self.entry), g(self.exit), g(self.half_width)
        # `self.previous_hx_m or hx`: None (never updated) -> hx
        span_hx = torch.maximum(hx, torch.where(torch.isnan(prev_hx), hx, prev_hx))
        span_hy = torch.maximum(hy, torch.where(torch.isnan(prev_hy), hy, prev_hy))
        has_prev = ~torch.isnan(prev)
        enter = live & ~active & has_prev & (prev + span_hx < entry) & (entry <= x + span_hx)
        safe_den = torch.where(enter, x - prev, torch.ones_like(x))
        fraction = ((entry - prev) - span_hx) / safe_den
        entry_y = prev_y + fraction * (y - prev_y)
        entry_clear = (half - entry_y.abs()) - span_hy
        worst = torch.where(enter & (entry_clear < worst), entry_clear, worst)
        failed = torch.where(enter, entry_clear <= 0.0, failed)
        active = active | enter
        act = live & active
        clear = (half - y.abs()) - span_hy
        worst = torch.where(act & (clear < worst), clear, worst)
        failed = torch.where(act, failed | collision | (clear <= 0.0) | (has_prev & (x < prev)), failed)
        pay = act & ~failed & ((x - hx) >= exit_)
        active = torch.where(act & (failed | pay), torch.zeros_like(active), active)
        paid = paid | pay
        # G11: table[rows[live], k[live]] = value[live] wrote the live rows of the active slot; writing
        # where(live, value, gathered old value) into every row's active slot leaves the other rows' bits
        # unchanged (rows are unique) and needs no boolean-mask nonzero (3 host syncs per table before)
        for name, value, old in (("active", active, active0), ("failed", failed, failed0), ("paid", paid, paid0),
                                 ("worst", worst, worst0), ("prev_x", x, prev), ("prev_y", y, prev_y),
                                 ("prev_hx", hx, prev_hx), ("prev_hy", hy, prev_hy)):
            table = getattr(self, name)
            table[rows, k] = torch.where(live, value, old)
        self.current = torch.where(live & (paid | failed), self.current + 1, self.current)
        return torch.where(pay, torch.full_like(x, self.bonus), torch.zeros_like(x))

    def summary(self, env_id):
        i = int(env_id)
        return [{"portal_id": "gate_%d" % (k + 1), "paid": bool(self.paid[i, k]), "failed": bool(self.failed[i, k]),
                 "worst_clearance_m": float(self.worst[i, k])} for k in range(self.g)]


# --------------------------------------------------------------- pool codec --
_D7_MODULES = ("model", "feasibility", "capacity", "visibility", "native", "reference", "adapters", "config")


def _d7_class(tag):
    module_name, _, class_name = tag.partition(".")
    if module_name not in _D7_MODULES or not class_name.isidentifier():
        raise ValueError(f"pool file names a non-D7 type {tag!r}")
    module = importlib.import_module(_dynamic().__name__ + "." + module_name)
    cls = getattr(module, class_name, None)
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise ValueError(f"pool file names a non-dataclass {tag!r}")
    return cls


def _encode(value):
    if is_dataclass(value) and not isinstance(value, type):
        module = type(value).__module__.rpartition(".")[2]
        tag = module + "." + type(value).__name__
        _d7_class(tag)
        return {"__d7__": tag, "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value) if f.init}}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, list):
        return {"__list__": [_encode(v) for v in value]}
    if isinstance(value, dict):
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} in a D7 pool file")


def _decode(value):
    if isinstance(value, dict):
        if "__d7__" in value:
            cls = _d7_class(value["__d7__"])
            return cls(**{k: _decode(v) for k, v in value["fields"].items()})
        if "__tuple__" in value:
            return tuple(_decode(v) for v in value["__tuple__"])
        if "__list__" in value:
            return [_decode(v) for v in value["__list__"]]
        if "__dict__" in value:
            return {_decode(k): _decode(v) for k, v in value["__dict__"]}
        if "__float__" in value:
            text = value["__float__"]
            if text not in ("inf", "-inf", "nan"):
                raise ValueError("malformed non-finite float in pool file")
            return float(text)
        raise ValueError("untagged object in pool file")
    if isinstance(value, list):
        raise ValueError("untagged list in pool file")
    return value


def _adapters():
    return importlib.import_module(_dynamic().__name__ + ".adapters")


def _accepted_rows(assignments):
    dynamic, adapters = _dynamic(), _adapters()
    return sorted([[k.eval_seed, k.env, k.episode, k.hazard_group, d, dynamic.digest(a.candidate),
                    adapters._witness_digest(a.witness)] for (k, d), a in assignments.items()])


def manifest_digest(offered, accepted, rejections, joint):
    """E4's pool digest (adapters._assemble manifest) from its four inputs."""
    return _dynamic().digest({"offered": list(offered), "accepted": sorted(accepted), "rejections": rejections,
                              "joint": joint})


def suite_parameters(suite):
    """D7 suite fields that shape a candidate (everything except enabled/dose)."""
    from dataclasses import asdict
    values = asdict(suite)
    values.pop("enabled")
    values.pop("dose")
    return values


def _key3(value):
    return tuple(int(v) for v in value)


def not_admitted_reasons(rejections, eval_seed, env_index, episode_index):
    """E4 rejection entries of one key as [[dose, reason, sampled_states], ...] (manifest order)."""
    key = (int(eval_seed), int(env_index), int(episode_index))
    return [[int(r.dose), str(r.reason), int(r.sampled_states)] for r in rejections
            if (r.key.eval_seed, r.key.env, r.key.episode) == key]


def primary_reasons(reasons):
    """The real E4 failures of a key (the joint echo 'joint_ladder_member_rejected' left out)."""
    return [r for r in reasons if r[1] != "joint_ladder_member_rejected"]


def offered_grid(offered):
    """(eval_seed, hazard_group, num_envs, episodes) of a pool's offered keys (D24).

    The offered keys must be ONE full grid {eval_seed} x range(num_envs) x range(episodes) of the
    path-hazard D7 group (what the offline admission tool offers); anything else is refused
    (ValueError), because "beyond the offered grid" would then be undefined.
    """
    keys = {(k.eval_seed, k.env, k.episode, k.hazard_group) for k in offered}
    if not keys:
        raise ValueError("a D7 pool offers at least one key")
    seeds, groups = {k[0] for k in keys}, {k[3] for k in keys}
    if len(seeds) != 1 or groups != {"d7_" + JOURNAL_TASK}:
        raise ValueError("a path-hazard D7 pool offers the keys of ONE eval seed in the d7_path_hazard group, "
                         f"not seeds {sorted(seeds)} / groups {sorted(groups)}")
    num_envs, episodes = 1 + max(k[1] for k in keys), 1 + max(k[2] for k in keys)
    if len(keys) != num_envs * episodes:
        raise ValueError(f"the offered D7 keys are not one full grid of {num_envs} envs x {episodes} episodes")
    return seeds.pop(), "d7_" + JOURNAL_TASK, num_envs, episodes


def check_grid(grid):
    """{"num_envs": N, "episodes": E} with positive integers (the D24 header block), or ValueError."""
    if not isinstance(grid, dict) or set(grid) != {"num_envs", "episodes"} or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in grid.values()):
        raise ValueError("a journal D7 pool must expose its offered grid as {'num_envs': N, 'episodes': E} "
                         f"with positive integers (CONTEXT_R4 D24), not {grid!r}")
    return {"num_envs": int(grid["num_envs"]), "episodes": int(grid["episodes"])}


def pool_key_status(pool, eval_seed, env_index, episode_index):
    """"admitted" | "not_admitted" | "beyond_offered_grid" of one key; KeyError = preflight error.

    Uses ``pool.key_status`` (PathHazardD7Pool) and falls back to ``is_admitted`` for other
    PathHazardD7Pool-like objects.
    """
    status = getattr(pool, "key_status", None)
    if status is not None:
        return status(eval_seed, env_index, episode_index)
    return ADMITTED if pool.is_admitted(eval_seed, env_index, episode_index) else NOT_ADMITTED


class NotAdmittedAssignment:
    """D16 record of a key E4 rejected: the scenario runs at D0 (no intruder), exposure not_admitted.

    Returned by ``PathHazardD7Pool.assignment`` for such a key at EVERY dose, so the F0 mixin's
    reset loads it like an admitted assignment; ``NotAdmittedBridge`` recognises it and never
    hands it to the E4 adapter (which accepts only admitted, witnessed assignments).
    """

    __slots__ = ("key", "reasons", "pool_digest")

    def __init__(self, key, reasons, pool_digest):
        self.key = key
        self.reasons = tuple(tuple(r) for r in reasons)
        self.pool_digest = str(pool_digest)

    def __eq__(self, other):
        return (isinstance(other, NotAdmittedAssignment) and self.key == other.key
                and self.reasons == other.reasons and self.pool_digest == other.pool_digest)

    def __hash__(self):
        return hash((self.key, self.reasons, self.pool_digest))

    def __repr__(self):
        return "NotAdmittedAssignment(%r, %d reasons)" % (self.key, len(self.reasons))

    def summary(self):
        """The EMPTY assignment of a not-admitted key (JSON-native; pairing reads key/task/dose/digest)."""
        k = self.key
        return {"key": {"eval_seed": k.eval_seed, "env": k.env, "episode": k.episode,
                        "hazard_group": k.hazard_group},
                "task": JOURNAL_TASK, "dose": 0, "candidate_digest": None, "witness_digest": None,
                "intruders": [], "admitted": False, "pool_digest": self.pool_digest}


class PathHazardD7Pool:
    """Admitted D7 ladders + explicit not-admitted records of one vessel's grid (D16).

    Duck-types what journal_env_mixin reads (``assignment(key, dose)``,
    ``pool_digest``, ``per_dose_counts``, ``keys_offered``, ``keys_admitted``,
    ``joint``). Every offered key is either admitted at all five doses (joint
    admission) or not admitted; ``assignment`` returns a
    ``NotAdmittedAssignment`` for the latter (never an exception, never a
    redraw). The offered keys form one full grid (``eval_seed``,
    ``hazard_group``, ``grid`` = {"num_envs", "episodes"}; ``offered_grid``).
    D24: a key of that seed and group, env inside the grid and episode index
    >= the grid's episodes is NOT ADMITTED with the single reason
    ``beyond_offered_grid`` (``assignment`` returns its NotAdmittedAssignment;
    such keys are not pool keys: the per-dose counts, the not-admitted list and
    the geometry hashes never include them). KeyError only for a key outside
    the env range or with another seed or group (a configuration error:
    another seed/grid needs its own admission run).
    The not-admitted records are DERIVED from E4's rejections, which the pool
    digest covers. ``per_dose_counts`` rows are ``[offered, admitted,
    not_admitted]`` (``e4_per_dose_counts`` keeps E4's two columns).
    ``doses`` lists the doses whose assignments are loaded (a committed file
    holds one dose; an in-memory admission all five).
    """

    PER_DOSE_COLUMNS = ("offered", "admitted", NOT_ADMITTED)

    def __init__(self, *, vessel, tier, pool_digest, per_dose_counts, keys_offered, keys_admitted, joint,
                 offered, accepted, rejections, assignments, geometry_hashes, suite, doses):
        self.vessel, self.tier = vessel, int(tier)
        self.pool_digest = pool_digest
        self.e4_per_dose_counts = tuple(tuple(int(v) for v in row[:2]) for row in per_dose_counts)
        self.keys_offered, self.keys_admitted, self.joint = int(keys_offered), int(keys_admitted), bool(joint)
        self.offered, self.accepted, self.rejections = tuple(offered), [list(r) for r in accepted], tuple(rejections)
        self.assignments = dict(assignments)
        self.geometry_hashes = {_key3(k): str(h) for k, h in dict(geometry_hashes).items()}
        self.suite_parameters = dict(suite)
        self.doses = tuple(sorted(int(d) for d in doses))
        self.manifest_sha256 = None
        self.eval_seed, self.hazard_group, self.grid_num_envs, self.grid_episodes = offered_grid(self.offered)
        admitted = {(r[0], r[1], r[2]) for r in self.accepted}
        self.not_admitted = {}
        for k in self.offered:
            key = (k.eval_seed, k.env, k.episode)
            if key not in admitted:
                self.not_admitted[key] = not_admitted_reasons(self.rejections, *key)
        self.verify()

    @property
    def keys_not_admitted(self):
        return len(self.not_admitted)

    @property
    def per_dose_counts(self):
        return tuple((o, a, o - a) for o, a in self.e4_per_dose_counts)

    @property
    def grid(self):
        """The offered grid {"num_envs": N, "episodes": E} (D24; published as dynamic_hazard_header["grid"])."""
        return {"num_envs": self.grid_num_envs, "episodes": self.grid_episodes}

    @classmethod
    def from_admission(cls, pool, offered, geometry_hashes, *, vessel, tier, suite):
        return cls(vessel=vessel, tier=tier, pool_digest=pool.pool_digest, per_dose_counts=pool.per_dose_counts,
                   keys_offered=pool.keys_offered, keys_admitted=pool.keys_admitted, joint=pool.joint,
                   offered=offered, accepted=_accepted_rows(pool.assignments), rejections=pool.rejections,
                   assignments=pool.assignments, geometry_hashes=geometry_hashes,
                   suite=suite_parameters(suite), doses=range(5))

    def verify(self):
        """Recompute every loaded digest and the pool digest (fail closed)."""
        dynamic, adapters = _dynamic(), _adapters()
        if manifest_digest(self.offered, self.accepted, self.rejections, self.joint) != self.pool_digest:
            raise ValueError("D7 pool digest does not reproduce from its manifest")
        if not self.joint:
            raise ValueError("path-hazard D7 pools are admitted jointly across D0..D4 (D16 ladders)")
        offered = {(k.eval_seed, k.env, k.episode) for k in self.offered}
        if len(offered) != len(self.offered) or len(self.offered) != self.keys_offered or \
                len({tuple(r[:3]) for r in self.accepted}) != self.keys_admitted:
            raise ValueError("D7 pool key counts disagree with the manifest")
        if set(self.geometry_hashes) != offered:
            raise ValueError("every offered key needs its E5 layout geometry hash")
        if any(not primary_reasons(r) for r in self.not_admitted.values()):
            raise ValueError("a not-admitted key carries no E4 rejection reason")
        if self.keys_admitted + self.keys_not_admitted != self.keys_offered:
            raise ValueError("offered keys must be admitted or not admitted")
        per_key = {}
        for r in self.accepted:
            per_key.setdefault(tuple(r[:3]), set()).add(r[4])
        if any(doses != set(range(5)) for doses in per_key.values()):
            raise ValueError("an admitted key must carry the whole D0..D4 ladder")
        if len(self.e4_per_dose_counts) != 5 or any(
                row[1] != self.keys_admitted or row[0] != self.keys_offered for row in self.e4_per_dose_counts):
            raise ValueError("per-dose counts disagree with the joint admission")
        accepted = {tuple(r) for r in self.accepted}
        for (key, dose), assignment in self.assignments.items():
            if dose not in self.doses:
                raise ValueError("assignment outside the loaded doses")
            if assignment.pool_digest != self.pool_digest:
                raise ValueError("assignment pool digest differs from the pool")
            row = (key.eval_seed, key.env, key.episode, key.hazard_group, dose,
                   dynamic.digest(assignment.candidate), adapters._witness_digest(assignment.witness))
            if row not in accepted or assignment.candidate_digest != row[5] or assignment.witness_digest != row[6]:
                raise ValueError(f"assignment {key} D{dose} does not match the admitted manifest")
            enc = assignment.candidate.request.encounter
            if enc.key != key or enc.dose != dose:
                raise ValueError("assignment filed under the wrong key or dose")
        for row in accepted:
            if row[4] in self.doses:
                key = dynamic.EpisodeKey(row[0], row[1], row[2], row[3])
                if (key, row[4]) not in self.assignments:
                    raise ValueError("an admitted assignment of a loaded dose is missing")
        return self

    def is_offered(self, eval_seed, env_index, episode_index):
        return (int(eval_seed), int(env_index), int(episode_index)) in self.geometry_hashes

    def key_status(self, eval_seed, env_index, episode_index, hazard_group="d7_" + JOURNAL_TASK):
        """D24 classification of one task-episode key.

        Returns "admitted" or "not_admitted" for an offered key, "beyond_offered_grid" for a key of
        the pool's seed and group with the env inside the grid and episode index >= the grid's
        episodes per env (not admitted, runs at D0). Raises KeyError (a preflight error) for an env
        outside the grid, another eval seed, another hazard group or a negative index.
        """
        seed, env, episode = int(eval_seed), int(env_index), int(episode_index)
        if seed != self.eval_seed:
            raise KeyError(f"eval seed {seed} is not the pool's eval seed {self.eval_seed}")
        if hazard_group != self.hazard_group:
            raise KeyError(f"hazard group {hazard_group!r} is not the pool's group {self.hazard_group!r}")
        if not 0 <= env < self.grid_num_envs:
            raise KeyError(f"env {env} is outside the pool's env range 0..{self.grid_num_envs - 1}")
        if episode < 0:
            raise KeyError(f"episode {episode} is negative")
        if episode >= self.grid_episodes:
            return BEYOND_OFFERED_GRID
        return NOT_ADMITTED if (seed, env, episode) in self.not_admitted else ADMITTED

    def is_admitted(self, eval_seed, env_index, episode_index):
        """True for an admitted key; False for a not-admitted or beyond-grid key (D16/D24); KeyError otherwise."""
        return self.key_status(eval_seed, env_index, episode_index) == ADMITTED

    def assignment(self, key, dose):
        """Admitted Assignment, or NotAdmittedAssignment (D16, and D24 for a beyond-grid key).

        KeyError for a key outside the env range or with another seed or group (preflight error), and
        for an admitted key whose dose is not loaded.
        """
        if not 0 <= int(dose) <= 4:
            raise ValueError("dose must be 0..4")
        status = self.key_status(key.eval_seed, key.env, key.episode, key.hazard_group)
        if status == BEYOND_OFFERED_GRID:
            return NotAdmittedAssignment(key, BEYOND_OFFERED_GRID_REASONS, self.pool_digest)
        if status == NOT_ADMITTED:
            return NotAdmittedAssignment(key, self.not_admitted[(key.eval_seed, key.env, key.episode)],
                                         self.pool_digest)
        return self.assignments[(key, int(dose))]

    def check_suite(self, suite):
        if suite_parameters(suite) != self.suite_parameters:
            raise ValueError("D7 pool was admitted under different D7 suite parameters")
        return self

    def reason_counts(self):
        """{E4 reason (prefix before ':'): dose-entries} over the not-admitted keys (joint echo excluded)."""
        counts = {}
        for reasons in self.not_admitted.values():
            for _, reason, _ in primary_reasons(reasons):
                name = reason.split(":")[0]
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def header_fields(self):
        """D7 header additions (D16): per-dose counts with not_admitted and the reason histogram."""
        return {"per_dose_counts": [list(r) for r in self.per_dose_counts],
                "per_dose_counts_columns": list(self.PER_DOSE_COLUMNS),
                "grid": self.grid,
                "keys_not_admitted": self.keys_not_admitted,
                "not_admitted_reason_counts": self.reason_counts(),
                "pool_manifest_sha256": self.manifest_sha256,
                "not_admitted_policy": ("D16: a key E4 rejects keeps its E5 layout and runs the dynamic suite "
                                        "at D0 (no intruder); its record carries dynamic_hazards payload "
                                        "exposure='not_admitted' and the E4 reasons")}

    def document(self, dose):
        """JSON document of ONE dose (the committed file unit)."""
        if dose not in self.doses:
            raise ValueError("dose not loaded")
        chosen = {kd: a for kd, a in self.assignments.items() if kd[1] == dose}
        return {"format": POOL_FORMAT, "task": JOURNAL_TASK, "vessel": self.vessel, "tier": self.tier,
                "dose": int(dose), "pool_digest": self.pool_digest,
                "per_dose_counts": [list(r) for r in self.e4_per_dose_counts], "keys_offered": self.keys_offered,
                "keys_admitted": self.keys_admitted, "joint": self.joint, "offered": _encode(self.offered),
                "accepted": self.accepted, "rejections": _encode(self.rejections),
                "not_admitted": sorted([list(k) + [r] for k, r in self.not_admitted.items()]),
                "geometry_hashes": sorted([list(k) + [h] for k, h in self.geometry_hashes.items()]),
                "suite_parameters": self.suite_parameters, "assignments": _encode(chosen)}


def save_pool_dose(pool, dose, path):
    """Write one dose of a pool (tagged JSON, gzip mtime 0 -> deterministic bytes); returns SHA-256."""
    text = json.dumps(pool.document(dose), sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw = gzip.compress(text.encode("ascii"), compresslevel=9, mtime=0)
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load_pool_dose(path, *, vessel, tier, dose, file_sha256=None, pool_digest=None):
    """Load + verify one committed dose file (file hash, identity, every digest)."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if file_sha256 is not None and digest != file_sha256:
        raise ValueError(f"D7 pool file {path} SHA-256 {digest} differs from the pinned {file_sha256}")
    doc = json.loads(gzip.decompress(raw).decode("ascii"))
    if doc.get("format") != POOL_FORMAT or doc.get("task") != JOURNAL_TASK:
        raise ValueError("not a path-hazard D7 pool file of format " + POOL_FORMAT)
    if doc.get("vessel") != vessel or int(doc.get("tier", -1)) != int(tier) \
            or int(doc.get("dose", -1)) != int(dose):
        raise ValueError(f"D7 pool file is {doc.get('vessel')}/G{doc.get('tier')}/D{doc.get('dose')}, "
                         f"not {vessel}/G{tier}/D{dose}")
    if pool_digest is not None and doc["pool_digest"] != pool_digest:
        raise ValueError("D7 pool digest differs from the pinned digest")
    pool = PathHazardD7Pool(vessel=vessel, tier=tier, pool_digest=doc["pool_digest"],
                            per_dose_counts=doc["per_dose_counts"], keys_offered=doc["keys_offered"],
                            keys_admitted=doc["keys_admitted"], joint=doc["joint"],
                            offered=_decode(doc["offered"]), accepted=doc["accepted"],
                            rejections=_decode(doc["rejections"]), assignments=_decode(doc["assignments"]),
                            geometry_hashes={tuple(r[:3]): r[3] for r in doc["geometry_hashes"]},
                            suite=doc["suite_parameters"], doses=(int(dose),))
    if {tuple(r[:3]): r[3] for r in doc["not_admitted"]} != pool.not_admitted:
        raise ValueError("stored not-admitted records differ from the E4 rejections")
    return pool


def pinned_manifest(path=None):
    return json.loads(Path(POOL_MANIFEST if path is None else path).read_text(encoding="ascii"))


def pinned_manifest_sha256(path=None):
    """SHA-256 of the committed pinned manifest bytes (stamped as dynamic_hazards_header.pool_manifest_sha256)."""
    return hashlib.sha256(Path(POOL_MANIFEST if path is None else path).read_bytes()).hexdigest()


def pool_folder(source="", path_variable=""):
    """Folder of the pool files: ``path_variable`` (cfg.journal_d7_pool_path), else
    $USVBENCH_D7_POOL_DIR, else tasks/path_hazard/journal_d7 (a non-empty ``source`` path, see
    below, is checked before them).

    Point at the pool folder with cfg.journal_d7_pool_path or $USVBENCH_D7_POOL_DIR: these are the
    RUNTIME variables (D19, F7 allow-list); they only choose where the pinned files are read from,
    and every file must match the committed manifest's SHA-256 pin, so they cannot change a result.
    ``source`` is cfg.journal_d7_pool, a PROTECTED field (CONTEXT_R4 D12/D25) meant for an in-memory
    pool object; never use it to name the folder (a non-empty path string there is still read first,
    for backward compatibility with round-4 callers, but it cannot be set by a run override).
    """
    import os
    for value in (source, path_variable, os.environ.get(POOL_DIR_ENV, "")):
        if value is not None and str(value):
            return Path(value)
    return POOL_DIR


def pinned_seed_table(manifest, seed=None):
    """G49 (J29, v2.6.5): the pin table of one evaluation seed. The top level of the pinned manifest is the
    v2.6.4 table (its ``eval_seed``, 42); every other seed's table is ``seeds["<seed>"]`` with the same fields
    (eval_seed, envs, episodes, tier, vessels; files path_hazard_d7_pool_<vessel>_D<k>_seed<seed>.json.gz).
    seed None or the top-level eval_seed -> the top level; an unpinned seed -> {}."""
    if seed is None or int(seed) == int(manifest.get("eval_seed", -1)):
        return manifest
    return manifest.get("seeds", {}).get(str(int(seed)), {})


def resolve_pool(source, *, vessel, tier, suite, manifest_path=None, path_variable="", seed=None):
    """The D7 pool of a journal env: an in-memory pool object, or a directory with the pinned files.

    ``source`` is cfg.journal_d7_pool, a PROTECTED field (CONTEXT_R4 D12/D25): an in-memory
    PathHazardD7Pool-like object (``assignment`` + ``is_admitted`` + ``pool_digest`` + ``grid``;
    tests and programmatic callers), or "" (the default) to read the pinned files from the folder
    named by ``path_variable`` (cfg.journal_d7_pool_path) or $USVBENCH_D7_POOL_DIR, else
    tasks/path_hazard/journal_d7 (``pool_folder``). Users point at the pool folder with
    cfg.journal_d7_pool_path or $USVBENCH_D7_POOL_DIR, never with cfg.journal_d7_pool (a path
    string there is still accepted for backward compatibility). A folder is read through the
    committed manifest: the file of (vessel, tier, suite dose) must exist and match its pinned
    SHA-256 and pool digest (fail closed; D7 B02/B05); the pinned not-admitted list, geometry
    hashes and grid (eval seed, envs, episodes) must equal the file's.
    G49: ``seed`` (cfg.seed) selects the pin table of that evaluation seed (``pinned_seed_table``; None = the
    v2.6.4 top-level table of seed 42).
    """
    dose = int(suite.dose)
    if not isinstance(source, (str, Path)):
        if not (callable(getattr(source, "assignment", None)) and callable(getattr(source, "is_admitted", None))
                and hasattr(source, "pool_digest")):
            raise ValueError("cfg.journal_d7_pool holds an in-memory PathHazardD7Pool-like object (assignment, "
                             "is_admitted, pool_digest, grid); point at a pool folder with "
                             f"cfg.journal_d7_pool_path or ${POOL_DIR_ENV} instead")
        check_grid(getattr(source, "grid", None))
        check = getattr(source, "check_suite", None)
        if check is not None:
            check(suite)
        if getattr(source, "vessel", vessel) != vessel or int(getattr(source, "tier", tier)) != int(tier):
            raise ValueError("in-memory D7 pool belongs to another vessel or tier")
        return source
    folder = pool_folder(source, path_variable)
    manifest = pinned_manifest(manifest_path)
    if manifest.get("format") != POOL_MANIFEST_FORMAT:
        raise ValueError("pinned D7 manifest is not of format " + POOL_MANIFEST_FORMAT)
    table = pinned_seed_table(manifest, seed)                                 # G49
    entry = table.get("vessels", {}).get(vessel)
    if entry is None or int(table.get("tier", -1)) != int(tier):
        raise ValueError(f"no pinned D7 pool for {vessel}/G{tier}" + ("" if seed is None else
                                                                       f" (evaluation seed {seed})"))
    files = entry["files"]
    if str(dose) not in files:
        raise ValueError(f"no pinned D7 pool file for dose D{dose}")
    name, sha = files[str(dose)]["file"], files[str(dose)]["sha256"]
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"D7 pool file {path} is missing: copy work/evidence/d7_pool/{name} of task F2 "
                                f"into {folder} (or point cfg.journal_d7_pool_path / ${POOL_DIR_ENV} at the "
                                f"folder holding it)")
    pool = load_pool_dose(path, vessel=vessel, tier=tier, dose=dose, file_sha256=sha,
                          pool_digest=entry["pool_digest"])
    if {tuple(r[:3]): r[3] for r in entry["not_admitted"]} != pool.not_admitted:
        raise ValueError("pinned not-admitted list differs from the pool file")
    if {tuple(r[:3]): r[3] for r in entry["geometry_hashes"]} != pool.geometry_hashes:
        raise ValueError("pinned layout geometry hashes differ from the pool file")
    if (table.get("eval_seed"), table.get("envs"), table.get("episodes")) != \
            (pool.eval_seed, pool.grid_num_envs, pool.grid_episodes):
        raise ValueError("pinned grid (eval_seed, envs, episodes) differs from the pool file's offered grid")
    pool.manifest_sha256 = pinned_manifest_sha256(manifest_path)
    return pool.check_suite(suite)


# ------------------------------------------------------ D16 runtime bridge --
class NotAdmittedBridge:
    """D16 wrapper of the E4 ``HazardBridge`` (the interface the F0 mixin calls).

    Admitted rows go to the E4 bridge unchanged. A row reset with a
    ``NotAdmittedAssignment`` runs the dynamic suite at D0: no intruder, no D7
    sample, no swept contact; ``finish`` returns a D7 stamp envelope
    ``{"payload", "sha256"}`` (same digest rule, schema and model version as
    E4's) whose payload carries ``exposure = "not_admitted"``, the E4 rejection
    reasons and the EMPTY assignment (dose 0, candidate digest None, no
    intruders). Its pairing primitive therefore equals the same key's record at
    D0 of every other suite, and it is unexposed for the dynamic treatment.
    A beyond-grid key (D24) is the same kind of row; its reasons are
    ``[[0, "beyond_offered_grid", 0]]``.
    """

    def __init__(self, inner, suite_dose):
        self.inner = inner
        self.dt = inner.dt
        self.suite_dose = int(suite_dose)
        self._na = {}          # env -> row state, only for rows whose CURRENT episode is not admitted

    @property
    def rows(self):
        merged = {i: r for i, r in self.inner.rows.items() if i not in self._na}
        merged.update(self._na)
        return merged

    def reset_rows(self, env_ids, assignments):
        ids = tuple(int(i) for i in env_ids)
        current = self.rows
        for env in ids:
            if env in current and not current[env]["finished"]:
                raise ValueError("finish and stamp every assigned episode before reset")
        na = tuple(i for i in ids if isinstance(assignments[i], NotAdmittedAssignment))
        admitted = tuple(i for i in ids if i not in na)
        for env in na:
            record = assignments[env]
            if record.key.env != env or record.key.hazard_group != "d7_" + JOURNAL_TASK:
                raise ValueError("wrong task or environment not-admitted record")
        if admitted:
            self.inner.reset_rows(admitted, {i: assignments[i] for i in admitted})
        for env in admitted:
            self._na.pop(env, None)
        for env in na:
            self._na[env] = {"record": assignments[env], "last_tick": -1, "finished": False, "contact": False,
                             "stamp": None}

    def step(self, ticks, poses):
        own = {i: t for i, t in ticks.items() if i in self._na}
        out = {}
        if len(own) != len(ticks):
            out = self.inner.step({i: t for i, t in ticks.items() if i not in self._na},
                                  {i: p for i, p in poses.items() if i not in self._na})
        for env, tick in own.items():
            row = self._na[env]
            if row["finished"]:
                raise ValueError("row finished; reset it before stepping")
            if type(tick) is not int or tick < 0:
                raise ValueError("tick must be a nonnegative integer")
            if tick < row["last_tick"]:
                raise ValueError("physics tick went backwards without a reset")
            row["last_tick"] = tick
            out[env] = ()
        return out

    def contact(self, env):
        if env in self._na:
            return False
        return self.inner.contact(env)

    def is_not_admitted(self, env):
        return int(env) in self._na

    def finish(self, env, termination_reason):
        if env not in self._na:
            return self.inner.finish(env, termination_reason)
        dynamic = _dynamic()
        d7_stamp = importlib.import_module(dynamic.__name__ + ".stamp")
        if termination_reason not in dynamic.TERMINATION_REASONS:
            raise ValueError("explicit terminal reason required: " + repr(termination_reason))
        row = self._na[env]
        if row["finished"]:
            return row["stamp"]
        record = row["record"]
        # D27: a not-admitted key (D16 or D24) is never assigned, so it is outside E4's intention-to-treat
        # denominator (every ASSIGNED episode): assigned False, itt_count 0.
        runtime = {
            "assigned": False, "exposure": NOT_ADMITTED, "itt_count": 0, "env": int(env),
            "termination_reason": termination_reason, "suite_dose": self.suite_dose, "run_dose": 0,
            "last_time_s": max(0.0, row["last_tick"] * self.dt),
            "first_detection_s": None, "realized_ttc_at_first_detection_s": None,
            "detected_samples": 0, "observer_updates": 0, "min_sampled_surface_clearance_m": None,
            "sampled_contact": False, "non_exposure_reason": NOT_ADMITTED,
            "bridge": {"swept_contact_latched": False, "first_swept_contact_interval_start_s": None,
                       "swept_intervals": 0, "last_tick": row["last_tick"], "physics_dt_s": self.dt,
                       "d7_rows": "none: not admitted, run at D0 without intruder (D16)"},
        }
        payload = {"schema_version": d7_stamp.SCHEMA_VERSION, "model_version": d7_stamp.MODEL_VERSION,
                   "exposure": NOT_ADMITTED,
                   "not_admitted_reasons": [list(r) for r in record.reasons],
                   "assignment": record.summary(), "runtime": runtime,
                   "semantics": dict(d7_stamp.SEMANTICS)}
        row["stamp"] = json.loads(d7_stamp.canonical({"payload": payload, "sha256": d7_stamp.digest(payload)}))
        row["finished"] = True
        return row["stamp"]


def record_exposure(dynamic_hazards):
    """'not_admitted' for a D16 record, else 'admitted' (E4 stamps carry no exposure field)."""
    return dynamic_hazards["payload"].get("exposure", "admitted")


# ---------------------------------------------------------- offline admission --
def admit_pool(keys, *, seed, tier, workers=1, log=None):
    """Joint D7 admission of the E5 layouts of a grid (D16); returns ({vessel: pool}, summary).

    keys: iterable of (vessel, env, episode). Every key keeps its E5 layout (one keyed stream, no
    redraw); ONE E4 ``admit_grid`` call (joint across D0..D4, grouped by vessel) decides per key:
    admitted ladder, or not admitted with E4's reasons. Pure and deterministic.
    """
    dynamic = _dynamic()
    keys = list(keys)
    envelopes, suites, jobs, offered, hashes = {}, {}, [], {}, {}
    for vessel, env, episode in keys:
        if vessel not in envelopes:
            envelopes[vessel] = journal_envelope(vessel)
            suites[vessel] = dynamic.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
        layout = journal_layout(vessel, tier, seed, env, episode)
        key = episode_key(seed, env, episode)
        jobs.append((d7_context(layout, envelopes[vessel]), key, suites[vessel]))
        offered.setdefault(vessel, []).append(key)
        hashes.setdefault(vessel, {})[(seed, env, episode)] = layout.geometry_hash
    pools = dynamic.admit_grid(jobs, joint=True, workers=workers, group_by=lambda job: job[0].envelope.vessel_id)
    out, summary = {}, {}
    for vessel, pool in pools.items():
        fam = PathHazardD7Pool.from_admission(pool, offered[vessel], hashes[vessel], vessel=vessel, tier=tier,
                                              suite=suites[vessel])
        out[vessel] = fam
        summary[vessel] = {"offered": fam.keys_offered, "admitted": fam.keys_admitted,
                           "not_admitted": fam.keys_not_admitted, "reason_counts": fam.reason_counts()}
        if log is not None:
            log("%s: %d/%d admitted, %d not admitted %s" % (vessel, fam.keys_admitted, fam.keys_offered,
                                                            fam.keys_not_admitted, fam.reason_counts()))
    return out, summary


__all__ = ["JOURNAL_TASK", "LAYOUT_GROUP", "journal_layout", "check_layout", "route_points",
           "reference_axis_rad", "family_scenario_hashes", "journal_envelope", "d7_nominal_route", "d7_context",
           "episode_key", "exact_hypot", "cross_track_error", "segment_hits_disk", "yaw_from_quat", "body_rates",
           "HullRectangle", "pose_clearance", "swept_clearance", "validity_codes", "OrderedPathTensorOracle",
           "portal_frames", "portal_local", "hull_half_extents", "ThreadingTensorTwin",
           "PathHazardD7Pool", "NotAdmittedAssignment", "NotAdmittedBridge", "record_exposure",
           "offered_grid", "check_grid", "pool_key_status", "ADMITTED", "BEYOND_OFFERED_GRID",
           "BEYOND_OFFERED_GRID_REASONS",
           "not_admitted_reasons", "primary_reasons", "admit_pool", "save_pool_dose", "load_pool_dose",
           "resolve_pool", "pool_folder", "pinned_manifest", "pinned_manifest_sha256", "suite_parameters",
           "manifest_digest", "POOL_DIR", "POOL_DIR_ENV", "POOL_MANIFEST", "NOT_ADMITTED", "ASSETS_V2_DIR"]
