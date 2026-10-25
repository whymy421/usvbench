"""Journal layout generators for the 12 tasks, contract version 2 (CPU only).

build_layout(task, tier, vessel_key, rng) returns an ADMITTED, immutable
JournalLayout in the task-local frame (the env may apply its scenario-keyed
global rotation afterwards; every quantity here is rotation covariant). All
generators draw only from the supplied numpy Generator, so a scenario key
fully determines the layout. A candidate is admitted only on its FINAL
geometry (journal_geometry_v2.admit_final_layout); rejection reasons are
counted, and exhaustion raises JournalAdmissionError BEFORE any controller
sees the scenario (a rejection is never an episode outcome).

JournalLayout exposes the attribute names of the legacy HazardLayout
(start, goal, centers, radii, obstacle_count, geodesic_length, ...) so the
HazardNav reset dispatch can consume it without restructuring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
import math
import os
import sys
from types import MappingProxyType

import numpy as np


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cfg = _load("journal_cfg_v2_for_layouts", "journal_cfg_v2.py")
geo = cfg.geometry

KIND_STATIC, KIND_TRANSLATE, KIND_ORBIT = 0, 1, 2   # hazard_geometry.py:2836-2838 convention


class JournalAdmissionError(RuntimeError):
    """No admissible layout within the attempt budget; the scenario is
    rejected before controller assignment and the histogram is reported."""

    def __init__(self, task, vessel, tier, attempts, rejections):
        self.task, self.vessel, self.tier = task, vessel, tier
        self.attempts, self.rejections = attempts, dict(rejections)
        super().__init__("no admissible %s/%s/G%d layout in %d attempts: %s"
                         % (task, vessel, tier, attempts, dict(rejections)))


@dataclass(frozen=True)
class GateRecord:
    name: str
    kind: str                  # "compulsory" (certified), "virtual" (line), "waypoint" (disk)
    midpoint_m: tuple
    through_m: tuple           # unit normal, direction of required passage
    physical_gap_m: float
    hull_centre_width_m: float
    planned_centre_width_m: float
    jamb_indices: tuple = ()
    certified: bool = False
    walk: tuple = ()


@dataclass(frozen=True)
class JournalLayout:
    task: str
    vessel: str
    tier: int
    start: np.ndarray
    start_heading_rad: float
    goal: np.ndarray
    goal_heading_rad: float | None
    centers: np.ndarray
    radii: np.ndarray
    field_indices: tuple
    structural_indices: tuple
    requested_obstacle_count: int
    count_mode: str
    gates: tuple
    waypoints: tuple
    bounds_m: tuple
    witness: object
    nominal_route_length_m: float
    deadline_s: float
    deadline_steps: int
    admission: object
    attempts: int
    rejections: MappingProxyType
    params: MappingProxyType
    extras: MappingProxyType
    geometry_hash: str
    d7_hook: MappingProxyType

    # legacy HazardLayout duck-typing -------------------------------------
    @property
    def level(self):
        return self.tier

    @property
    def obstacle_count(self):
        return int(self.radii.shape[0])

    @property
    def realized_field_count(self):
        return len(self.field_indices)

    @property
    def structural_count(self):
        return len(self.structural_indices)

    @property
    def geodesic_length(self):
        """Nominal route (verified witness length), used for deadlines."""
        return self.nominal_route_length_m

    @property
    def direct_blocked(self):
        return not (self.admission.direct_physical_clearance_m > 0.0)

    @property
    def feasible(self):
        return True


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _unit(angle):
    return np.array((math.cos(angle), math.sin(angle)))


def wall_cylinders(p0, p1, radius_m, surface_overlap_m):
    """Tile a segment with cylinders whose PHYSICAL surfaces overlap by at
    least surface_overlap_m (count from a ceiling, so the overlap is a floor;
    same rule as hazard_geometry.wall_segment_cylinders:1933-1955)."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    stride = 2.0 * radius_m - surface_overlap_m
    if stride <= 0.0:
        raise ValueError("overlap must be smaller than the cylinder diameter")
    span = float(np.linalg.norm(p1 - p0))
    count = max(2, int(math.ceil(span / stride)) + 1)
    f = np.linspace(0.0, 1.0, count)[:, None]
    return p0[None, :] + f * (p1 - p0)[None, :]


def _dedupe(centers, radii, tol=1.0e-9):
    keep_c, keep_r = [], []
    for c, r in zip(centers, radii):
        if any(np.linalg.norm(c - k) <= tol and abs(r - q) <= tol for k, q in zip(keep_c, keep_r)):
            continue
        keep_c.append(np.asarray(c, dtype=np.float64))
        keep_r.append(float(r))
    return np.asarray(keep_c, dtype=np.float64).reshape(-1, 2), np.asarray(keep_r, dtype=np.float64)


def _place_field(rng, count, radius_range, sampler, placed_c, placed_r, min_gap, clear_fn, tries=3000):
    """Rejection-sample `count` disks; pairwise physical surface gap >= min_gap
    with every already placed field disk; clear_fn(center, radius) -> bool."""
    out_c, out_r = [], []
    for _ in range(count):
        ok = False
        for _ in range(tries):
            r = float(rng.uniform(*radius_range))
            c = np.asarray(sampler(), dtype=np.float64)
            if not clear_fn(c, r):
                continue
            allc = placed_c + out_c
            allr = placed_r + out_r
            if allc:
                gaps = np.linalg.norm(np.asarray(allc) - c, axis=1) - np.asarray(allr) - r
                if np.any(gaps + 1.0e-9 < min_gap):
                    continue
            out_c.append(c)
            out_r.append(r)
            ok = True
            break
        if not ok:
            return None
    return out_c, out_r


def moving_positions(base_centers, kind, params, t_s):
    """Centres at time t_s; same law as hazard_geometry.moving_basin_positions
    (:2958-2999): triangle-wave translation and circular orbit."""
    centers = np.array(base_centers, dtype=np.float64, copy=True)
    kind = np.asarray(kind)
    params = np.asarray(params, dtype=np.float64)
    rows = np.nonzero(kind == KIND_TRANSLATE)[0]
    if rows.size:
        base, direction = params[rows, 0:2], params[rows, 2:4]
        phase, speed, span = params[rows, 4], params[rows, 5], params[rows, 6]
        s = np.mod(speed * float(t_s) + phase, 2.0 * span)
        offset = np.where(s <= span, s, 2.0 * span - s) - 0.5 * span
        centers[rows] = base + direction * offset[:, None]
    orbits = np.nonzero(kind == KIND_ORBIT)[0]
    if orbits.size:
        theta = params[orbits, 3] * float(t_s) + params[orbits, 4]
        centers[orbits, 0] = params[orbits, 0] + params[orbits, 2] * np.cos(theta)
        centers[orbits, 1] = params[orbits, 1] + params[orbits, 2] * np.sin(theta)
    return centers


def _berth_structure(p, target_xy, heading):
    """U slip in the berth frame (target at origin, +x = dock heading).

    Interior x in [opening_x, back_face_x], |y| <= free_width/2 (INNER FACES).
    Side walls run from the opening to the back wall; the back wall's inner
    face is at back_face_x. Physical surfaces overlap (solid walls)."""
    w2 = 0.5 * p["slip_free_width_m"]
    rw = p["wall_radius_m"]
    ov = min(0.25 * p["length_scale"], 1.2 * rw)
    x_open, x_back = p["slip_opening_x_m"], p["slip_back_face_x_m"]
    y_wall = w2 + rw
    x_wall = x_back + rw
    segs = [((x_open, -y_wall), (x_wall, -y_wall)), ((x_open, y_wall), (x_wall, y_wall)),
            ((x_wall, -y_wall), (x_wall, y_wall))]
    cs = [wall_cylinders(a, b, rw, ov) for a, b in segs]
    local, radii = _dedupe(np.vstack(cs), np.full(sum(len(c) for c in cs), rw))
    rot = np.array(((math.cos(heading), -math.sin(heading)), (math.sin(heading), math.cos(heading))))
    world = local @ rot.T + np.asarray(target_xy, dtype=np.float64)
    # mouth jambs: the first cylinder of each side wall (at x_open)
    jl = int(np.argmin(np.linalg.norm(local - np.array((x_open, -y_wall)), axis=1)))
    jr = int(np.argmin(np.linalg.norm(local - np.array((x_open, y_wall)), axis=1)))
    berth = dict(center_xy=tuple(map(float, target_xy)), heading_rad=float(heading),
                 opening_x_m=float(x_open), back_x_m=float(x_back), half_width_m=float(w2),
                 position_tolerance_m=p["position_tolerance_m"],
                 heading_tolerance_rad=p["heading_tolerance_rad"],
                 speed_tolerance_m_s=p["speed_tolerance_m_s"], hold_s=p["hold_s"])
    return world, radii, (jl, jr), berth


def _turn_deviation_ok(witness, index_points, radius_limits):
    """Fillets cut corners by R(1/cos(theta/2)-1); require that to stay
    within each waypoint disk (or virtual gate half width)."""
    v = np.asarray(witness.vertices_m)
    R = witness.turn_radius_m
    for point, limit in zip(index_points, radius_limits):
        d = np.linalg.norm(v - np.asarray(point), axis=1)
        k = int(np.argmin(d))
        if d[k] > 1.0e-6:
            return False
        if 0 < k < len(v) - 1 and k not in witness.pivot_vertex_indices:
            a, b, c = v[k - 1], v[k], v[k + 1]
            h1 = math.atan2(*(b - a)[::-1])
            h2 = math.atan2(*(c - b)[::-1])
            theta = abs((h2 - h1 + math.pi) % (2.0 * math.pi) - math.pi)
            if R * (1.0 / math.cos(theta / 2.0) - 1.0) > limit + 1.0e-9:
                return False
    return True


# --------------------------------------------------------------------------
# candidate generators: each returns a dict or a rejection-reason string
# --------------------------------------------------------------------------

def _route_polyline(rng, p):
    heads = [0.0]
    turn = math.radians(p["heading_change_deg"])
    for _ in range(p["segment_count"] - 1):
        heads.append(heads[-1] + (turn if rng.random() < 0.5 else -turn))
    pts = [np.zeros(2)]
    for h in heads:
        pts.append(pts[-1] + p["segment_length_m"] * _unit(h))
    return pts


def _bbox(points, pad):
    a = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (float(a[:, 0].min() - pad), float(a[:, 0].max() + pad),
            float(a[:, 1].min() - pad), float(a[:, 1].max() + pad))


def _cand_reach(rng, p, vessel):
    d = p["straight_distance_m"]
    start, goal = np.zeros(2), np.array((d, 0.0))
    return dict(start=start, goal=goal, heading=float(rng.uniform(-math.pi, math.pi)),
                centers=np.zeros((0, 2)), radii=np.zeros(0), field=(), structural=(),
                bounds=_bbox((start, goal), p["region_padding_m"]), vias=(), gates=(),
                waypoints=(tuple(goal),))


def _cand_path(rng, p, vessel):
    pts = _route_polyline(rng, p)
    gates = tuple(GateRecord("waypoint_%d" % i, "waypoint", tuple(map(float, q)),
                             tuple(map(float, _unit(0.0))), 2.0 * p["gate_radius_m"],
                             2.0 * p["gate_radius_m"], 2.0 * p["gate_radius_m"])
                  for i, q in enumerate(pts[1:], start=1))
    return dict(start=pts[0], goal=pts[-1], heading=float(rng.uniform(-math.pi, math.pi)),
                centers=np.zeros((0, 2)), radii=np.zeros(0), field=(), structural=(),
                bounds=_bbox(pts, p["region_padding_m"]), vias=tuple(pts[1:-1]), gates=gates,
                waypoints=tuple(tuple(map(float, q)) for q in pts[1:]))


def _cand_station(rng, p, vessel):
    d = p["approach_distance_m"]
    start, station = np.zeros(2), np.array((d, 0.0))
    return dict(start=start, goal=station, heading=float(rng.uniform(-math.pi, math.pi)),
                centers=np.zeros((0, 2)), radii=np.zeros(0), field=(), structural=(),
                bounds=_bbox((start, station), p["region_padding_m"]), vias=(), gates=(),
                waypoints=(tuple(station),),
                station=dict(center_xy=tuple(station), radius_m=p["hold_radius_m"],
                             hold_s=p["hold_s"], final_window_s=p["final_window_s"]))


def _cand_docking(rng, p, vessel):
    centers, radii, jambs, berth = _berth_structure(p, (0.0, 0.0), 0.0)
    x_open = p["slip_opening_x_m"]
    runup = vessel.approach_runup_m
    start = np.array((x_open - p["approach_distance_m"],
                      float(rng.uniform(-p["lateral_offset_max_m"], p["lateral_offset_max_m"]))))
    mouth = np.array((x_open, 0.0))
    toward = math.atan2(-start[1], x_open - start[0])
    heading = toward + float(rng.uniform(-p["heading_offset_max_rad"], p["heading_offset_max_rad"]))
    run_pt = np.array((x_open - runup, 0.0))
    if run_pt[0] <= start[0] + 1.0e-6:
        return "approach_shorter_than_runup"
    gate = GateRecord("berth_mouth", "compulsory", tuple(mouth), (1.0, 0.0),
                      p["slip_free_width_m"], p["slip_free_width_m"] - vessel.beam_m,
                      p["slip_free_width_m"] - vessel.beam_m - 2.0 * p["planning_margin_m"], jambs)
    bounds = _bbox(np.vstack((start[None], centers)), p["region_padding_m"])
    return dict(start=start, goal=np.zeros(2), goal_heading=0.0, heading=heading, centers=centers,
                radii=radii, field=(), structural=tuple(range(len(radii))), bounds=bounds,
                vias=(run_pt, mouth), gates=(gate,), compulsory=(jambs,), berth=berth,
                waypoints=(tuple(run_pt), tuple(mouth), (0.0, 0.0)))


def _cand_path_hazard(rng, p, vessel):
    pts = _route_polyline(rng, p)
    k_blk = p["blocker_count"]
    n_req = p["requested_count"]
    segs = rng.choice(p["segment_count"], size=k_blk, replace=False)
    lo, hi = p["radius_range_m"]
    clear_r = p["gate_clear_radius_m"]
    gap = p["min_pair_surface_gap_m"]
    band = p["route_band_m"]

    def clear_fn(c, r):
        for q in pts:
            if np.linalg.norm(c - q) - r < clear_r:
                return False
        return True

    placed_c, placed_r, blockers = [], [], []
    for sidx in sorted(int(x) for x in segs):
        a, b = pts[sidx], pts[sidx + 1]
        u = (b - a) / np.linalg.norm(b - a)
        nrm = np.array((-u[1], u[0]))

        def sampler(a=a, b=b, nrm=nrm):
            f = rng.uniform(*p["blocker_fraction"])
            return a + f * (b - a) + nrm * rng.uniform(-p["blocker_offset_max_m"], p["blocker_offset_max_m"])

        got = _place_field(rng, 1, (lo, hi), sampler, placed_c, placed_r, gap, clear_fn)
        if got is None:
            return "blocker_placement_exhausted"
        placed_c += got[0]
        placed_r += got[1]
        blockers.append(sidx)
    box = _bbox(pts, band)

    def sampler_band():
        return np.array((rng.uniform(box[0], box[1]), rng.uniform(box[2], box[3])))

    def clear_band(c, r):
        d = min(float(geo._distances_to_segment(c[None], a, b)[0]) for a, b in zip(pts, pts[1:]))
        return d <= band and clear_fn(c, r)

    got = _place_field(rng, n_req - k_blk, (lo, hi), sampler_band, placed_c, placed_r, gap, clear_band)
    if got is None:
        return "field_placement_exhausted"
    centers = np.asarray(placed_c + got[0]).reshape(-1, 2)
    radii = np.asarray(placed_r + got[1])
    # recompute blocker identity on the final geometry
    realized_blocked = set()
    for sidx in range(len(pts) - 1):
        a, b = pts[sidx], pts[sidx + 1]
        L = np.linalg.norm(b - a)
        f = np.clip((centers - a) @ (b - a) / (L * L), 0.0, 1.0)
        d = geo._distances_to_segment(centers, a, b)
        hit = (d <= p["blocker_offset_max_m"] + 1.0e-9) & (f >= p["blocker_fraction"][0] - 1.0e-9) \
            & (f <= p["blocker_fraction"][1] + 1.0e-9)
        if np.any(hit):
            realized_blocked.add(sidx)
    if not set(blockers) <= realized_blocked:
        return "designated_blocker_lost"
    gates = tuple(GateRecord("waypoint_%d" % i, "waypoint", tuple(map(float, q)), (1.0, 0.0),
                             2.0 * p["gate_radius_m"], 2.0 * p["gate_radius_m"], 2.0 * p["gate_radius_m"])
                  for i, q in enumerate(pts[1:], start=1))
    return dict(start=pts[0], goal=pts[-1], heading=float(rng.uniform(-math.pi, math.pi)),
                centers=centers, radii=radii, field=tuple(range(len(radii))), structural=(),
                bounds=_bbox(pts, band + p["region_padding_m"]), vias=tuple(pts[1:-1]), gates=gates,
                waypoints=tuple(tuple(map(float, q)) for q in pts[1:]),
                extras=dict(designated_blocker_segments=tuple(sorted(blockers)),
                            realized_blocked_segments=tuple(sorted(realized_blocked))))


def _cand_scatter(rng, p, vessel):
    d = p["straight_distance_m"]
    start, goal = np.zeros(2), np.array((d, 0.0))
    lo, hi = p["radius_range_m"]
    e = p["endpoint_clear_m"]
    corr = p["corridor_half_width_m"]

    def clear_fn(c, r):
        return (np.linalg.norm(c - start) - r >= e) and (np.linalg.norm(c - goal) - r >= e)

    def blocker():
        f = rng.uniform(*p["blocker_fraction"])
        return np.array((f * d, 0.0))

    first = _place_field(rng, 1, (lo, hi), blocker, [], [], p["min_pair_surface_gap_m"], clear_fn)
    if first is None:
        return "blocker_placement_exhausted"
    rest = _place_field(rng, p["requested_count"] - 1, (lo, hi),
                        lambda: np.array((rng.uniform(0.0, d), rng.uniform(-corr, corr))),
                        first[0], first[1], p["min_pair_surface_gap_m"], clear_fn)
    if rest is None:
        return "field_placement_exhausted"
    centers = np.asarray(first[0] + rest[0]).reshape(-1, 2)
    radii = np.asarray(first[1] + rest[1])
    pad = p["region_padding_m"]
    return dict(start=start, goal=goal, heading=float(rng.uniform(-math.pi, math.pi)), centers=centers,
                radii=radii, field=tuple(range(len(radii))), structural=(),
                bounds=(-pad, d + pad, -(corr + pad), corr + pad), vias=(), gates=(),
                waypoints=(tuple(goal),))


def _cand_iceberg(rng, p, vessel):
    d = p["straight_distance_m"]
    start, goal = np.zeros(2), np.array((d, 0.0))
    lo, hi = p["radius_range_m"]
    e = p["endpoint_clear_m"]
    gap = p["min_edge_gap_m"]
    n = p["requested_count"]
    r0 = float(rng.uniform(lo, hi))
    x_lo, x_hi = max(r0 + e, 0.38 * d), min(d - r0 - e, 0.62 * d)
    if x_lo > x_hi:
        return "iceberg_blocker_does_not_fit"
    centers, radii = [np.array((rng.uniform(x_lo, x_hi), 0.0))], [r0]
    side0 = -1.0 if rng.random() < 0.5 else 1.0
    for idx in range(1, n):
        side = side0 if idx % 2 else -side0
        placed = False
        for _ in range(600):
            r = float(rng.uniform(lo, hi))
            x = float(rng.uniform(0.25 * d, 0.75 * d))
            need = radii[0] + r + gap
            dx = x - centers[0][0]
            y = side * (math.sqrt(max(0.0, need * need - dx * dx)) + float(rng.uniform(0.0, 4.0 * p["length_scale"])))
            c = np.array((x, y))
            if np.linalg.norm(c - start) - r < e or np.linalg.norm(c - goal) - r < e:
                continue
            gaps = np.linalg.norm(np.asarray(centers) - c, axis=1) - np.asarray(radii) - r
            if np.any(gaps + 1.0e-9 < gap):
                continue
            centers.append(c)
            radii.append(r)
            placed = True
            break
        if not placed:
            return "iceberg_placement_exhausted"
    centers, radii = np.asarray(centers), np.asarray(radii)
    ext = float(np.max(np.abs(centers[:, 1]) + radii))
    pad = p["region_padding_m"]
    return dict(start=start, goal=goal, heading=float(rng.uniform(-math.pi, math.pi)), centers=centers,
                radii=radii, field=tuple(range(n)), structural=(),
                bounds=(-pad, d + pad, -(ext + pad), ext + pad), vias=(), gates=(), waypoints=(tuple(goal),))


def _cand_ring(rng, p, vessel):
    tier = geo.physical_tier(p["tier"], vessel)
    bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    ring = geo.construct_sealed_ring(vessel, tier, ring_radius_m=p["ring_radius_m"],
                                     obstacle_radius_m=p["obstacle_radius_m"], gate_bearing_rad=bearing,
                                     seal_overlap_m=p["seal_overlap_m"])
    gb = bearing + float(rng.uniform(-p["goal_bearing_offset_max_rad"], p["goal_bearing_offset_max_rad"]))
    goal = p["goal_distance_m"] * _unit(gb)
    gate = GateRecord("ring_gate", "compulsory", tuple(map(float, ring.gate_midpoint_m)),
                      tuple(map(float, ring.gate_normal_m)), tier.physical_surface_gap_m,
                      tier.hull_centre_width_m, tier.planned_centre_width_m, ring.gate)
    ext = p["goal_distance_m"] + p["region_padding_m"]
    return dict(start=np.zeros(2), goal=goal, heading=float(rng.uniform(-math.pi, math.pi)),
                centers=np.array(ring.centers_m), radii=np.array(ring.radii_m), field=(),
                structural=tuple(range(len(ring.radii_m))), bounds=(-ext, ext, -ext, ext), vias=(),
                gates=(gate,), compulsory=(ring.gate,), waypoints=(tuple(goal),),
                extras=dict(ring_neighbour_overlap_m=ring.neighbour_overlap_m))


def _cand_fortress(rng, p, vessel):
    tier = geo.physical_tier(p["tier"], vessel)
    th_o = float(rng.uniform(0.0, 2.0 * math.pi))
    lo_st, hi_st = p["gate_stagger_rad"]
    th_i = th_o + (1.0 if rng.random() < 0.5 else -1.0) * float(rng.uniform(lo_st, hi_st))
    inner = geo.construct_sealed_ring(vessel, tier, ring_radius_m=p["inner_ring_radius_m"],
                                      obstacle_radius_m=p["obstacle_radius_m"], gate_bearing_rad=th_i,
                                      seal_overlap_m=p["seal_overlap_m"])
    outer = geo.construct_sealed_ring(vessel, tier, ring_radius_m=p["outer_ring_radius_m"],
                                      obstacle_radius_m=p["obstacle_radius_m"], gate_bearing_rad=th_o,
                                      seal_overlap_m=p["seal_overlap_m"])
    n_in = len(inner.radii_m)
    centers = np.vstack((inner.centers_m, outer.centers_m))
    radii = np.concatenate((inner.radii_m, outer.radii_m))
    g_in = inner.gate
    g_out = (outer.gate[0] + n_in, outer.gate[1] + n_in)
    sb = th_o + float(rng.uniform(-p["start_bearing_offset_max_rad"], p["start_bearing_offset_max_rad"]))
    start = p["start_distance_m"] * _unit(sb)
    gates = (GateRecord("outer_gate", "compulsory", tuple(map(float, outer.gate_midpoint_m)),
                        tuple(map(float, -outer.gate_normal_m)), tier.physical_surface_gap_m,
                        tier.hull_centre_width_m, tier.planned_centre_width_m, g_out),
             GateRecord("inner_gate", "compulsory", tuple(map(float, inner.gate_midpoint_m)),
                        tuple(map(float, -inner.gate_normal_m)), tier.physical_surface_gap_m,
                        tier.hull_centre_width_m, tier.planned_centre_width_m, g_in))
    ext = p["start_distance_m"] + p["region_padding_m"]
    stagger = abs((th_i - th_o + math.pi) % (2.0 * math.pi) - math.pi)
    if stagger + 1.0e-12 < lo_st:
        return "gate_stagger_below_90deg"
    return dict(start=start, goal=np.zeros(2), heading=float(rng.uniform(-math.pi, math.pi)),
                centers=centers, radii=radii, field=(), structural=tuple(range(len(radii))),
                bounds=(-ext, ext, -ext, ext), vias=(), gates=gates, compulsory=(g_out, g_in),
                forbid_direct=True, waypoints=((0.0, 0.0),),
                extras=dict(gate_stagger_rad=stagger, inner_count=n_in,
                            outer_count=len(outer.radii_m)))


def _cand_forced(rng, p, vessel):
    d = p["goal_distance_m"]
    hw = p["basin_half_width_m"]
    em = p["end_margin_m"]
    rp, rb = p["perimeter_radius_m"], p["bulkhead_radius_m"]
    ov = p["wall_surface_overlap_m"]
    g = p["physical_surface_gap_m"]
    approach = max(6.0 * p["length_scale"], vessel.approach_runup_m + rb)
    wall_x = 0.5 * d + float(rng.uniform(-p["wall_x_jitter_m"], p["wall_x_jitter_m"]))
    if wall_x < approach or wall_x > d - approach:
        return "bulkhead_too_close_to_endpoint"
    jamb = 0.5 * g + rb
    limit = hw - jamb - p["bulkhead_min_side_m"]
    if limit <= 0.0:
        return "gate_does_not_fit_basin"
    gate_y = float(rng.uniform(-limit, limit))
    x_back, x_front = -em, d + em
    perim = [((x_back, -hw), (x_front, -hw)), ((x_back, hw), (x_front, hw)),
             ((x_back, -hw), (x_back, hw)), ((x_front, -hw), (x_front, hw))]
    cs, rs = [], []
    for a, b in perim:
        c = wall_cylinders(a, b, rp, ov)
        cs.append(c)
        rs.append(np.full(len(c), rp))
    ov_b = min(ov, 1.2 * rb)
    for a, b in (((wall_x, -hw), (wall_x, gate_y - jamb)), ((wall_x, gate_y + jamb), (wall_x, hw))):
        c = wall_cylinders(a, b, rb, ov_b)
        cs.append(c)
        rs.append(np.full(len(c), rb))
    centers, radii = _dedupe(np.vstack(cs), np.concatenate(rs))
    ja = int(np.argmin(np.linalg.norm(centers - np.array((wall_x, gate_y - jamb)), axis=1)))
    jb = int(np.argmin(np.linalg.norm(centers - np.array((wall_x, gate_y + jamb)), axis=1)))
    measured = float(np.linalg.norm(centers[ja] - centers[jb]) - radii[ja] - radii[jb])
    if abs(measured - g) > 1.0e-6:
        return "gate_width_mismatch"
    start, goal = np.zeros(2), np.array((d, 0.0))
    for q in (start, goal):
        if np.any(np.linalg.norm(centers - q, axis=1) - radii < p["endpoint_clear_m"]):
            return "endpoint_not_clear"
    gate = GateRecord("bulkhead_gate", "compulsory", (wall_x, gate_y), (1.0, 0.0), measured,
                      measured - vessel.beam_m, measured - vessel.beam_m - 2.0 * p["planning_margin_m"],
                      (ja, jb))
    run = vessel.approach_runup_m
    via = (np.array((wall_x - run, gate_y)), np.array((wall_x, gate_y)), np.array((wall_x + run, gate_y)))
    return dict(start=start, goal=goal, heading=float(rng.uniform(-math.pi, math.pi)), centers=centers,
                radii=radii, field=(), structural=tuple(range(len(radii))),
                bounds=(x_back, x_front, -hw, hw), vias=via, gates=(gate,), compulsory=((ja, jb),),
                waypoints=(tuple(goal),), extras=dict(wall_x_m=wall_x, gate_y_m=gate_y))


def _moving_params(base_rows, orbit, mu, nu, top, period, orbit_radius):
    """Motion parameters for one dose; positions at t=0 are dose-independent."""
    speed, omega = mu * top, nu * top / orbit_radius
    kinds, params = [], []
    for (x, base_y, phase) in base_rows:
        if speed > 0.0:
            kinds.append(KIND_TRANSLATE)
            params.append((x, base_y, 0.0, 1.0, phase, speed, period))
        else:
            kinds.append(KIND_STATIC)
            params.append((0.0,) * 7)
    cx, oph = orbit
    if omega > 0.0:
        kinds.append(KIND_ORBIT)
        params.append((cx, 0.0, orbit_radius, omega, oph, 0.0, 0.0))
    else:
        kinds.append(KIND_STATIC)
        params.append((0.0,) * 7)
    return kinds, params


def _cand_moving(rng, p, vessel, mu, nu, nu_tied):
    """Admission is JOINT over the whole native-motion dose ladder (plus the
    requested dose), so the same scenario key yields the same geometry at
    every dose; only the motion parameters differ between doses."""
    hw, em, rp, ov = p["basin_half_width_m"], p["end_margin_m"], p["perimeter_radius_m"], p["wall_surface_overlap_m"]
    rr, gap = p["row_radius_m"], p["row_gap_m"]
    top = p["top_speed_m_s"]
    period = gap + 2.0 * rr
    x1 = p["approach_m"]
    x2 = x1 + p["row_separation_m"]
    cx = x2 + p["buffer_m"] + p["orbit_radius_m"]
    goal_x = p["goal_distance_m"]
    # constructive fairness bounds with the vessel's full length (legacy
    # hazard_geometry.py:2913-2955 uses 2*half_beam and a fixed top speed)
    depth = 2.0 * rr + vessel.length_m
    nu_bound = math.pi * p["orbit_radius_m"] / (2.0 * p["orbit_radius_m"] + vessel.length_m)
    doses = [(m, m if nu_tied else nu) for m in p["dose_mu_ladder"]]
    if (mu, nu) not in doses:
        doses.append((mu, nu))
    for m, n in doses:
        if not 0.0 <= m < 1.0 or n < 0.0:
            return "invalid_motion_dose"
        if m > 0.0 and not depth / (top * math.sqrt(1.0 - m * m)) < period / (m * top):
            return "row_tracking_infeasible"
        if n >= nu_bound:
            return "orbit_window_infeasible"
    x_back, x_front = -em, goal_x + em
    cs, rs = [], []
    for a, b in (((x_back, -hw), (x_front, -hw)), ((x_back, hw), (x_front, hw)),
                 ((x_back, -hw), (x_back, hw)), ((x_front, -hw), (x_front, hw))):
        for q in wall_cylinders(a, b, rp, ov):
            cs.append(q)
            rs.append(rp)
    n_static = len(rs)
    phase = float(rng.uniform(0.0, 2.0 * period))
    rows = []
    for x, centre_y in ((x1, 0.0), (x2, 0.5 * period)):
        reach = hw + 1.5 * period
        k_lo = int(math.floor((-reach - centre_y) / period))
        k_hi = int(math.ceil((reach - centre_y) / period))
        s_tri = math.fmod(phase, 2.0 * period)
        offset0 = (s_tri if s_tri <= period else 2.0 * period - s_tri) - 0.5 * period
        for k in range(k_lo, k_hi + 1):
            base_y = centre_y + k * period
            cs.append(np.array((x, base_y + offset0)))
            rs.append(rr)
            rows.append((x, base_y, phase))
    oph = float(rng.uniform(0.0, 2.0 * math.pi))
    cs.append(np.array((cx + p["orbit_radius_m"] * math.cos(oph), p["orbit_radius_m"] * math.sin(oph))))
    rs.append(rr)
    centers = np.asarray(cs, dtype=np.float64)
    radii = np.asarray(rs, dtype=np.float64)
    start, goal = np.zeros(2), np.array((goal_x, 0.0))
    bounds = (x_back, x_front, -hw, hw)
    margin = p["planning_margin_m"]
    n_s = p["admission_samples_per_period"]

    def kinds_params(m, n):
        k, q = _moving_params(rows, (cx, oph), m, n, top, period, p["orbit_radius_m"])
        kinds = np.asarray([KIND_STATIC] * n_static + k, dtype=np.int64)
        params = np.asarray([(0.0,) * 7] * n_static + q, dtype=np.float64)
        return kinds, params

    sampled = {}
    for m, n in doses:
        kinds, params = kinds_params(m, n)
        speed, omega = m * top, n * top / p["orbit_radius_m"]
        times = [0.0]
        if speed > 0.0:
            times += [2.0 * period / speed * i / n_s for i in range(1, n_s)]
        if omega > 0.0:
            times += [2.0 * math.pi / omega * i / n_s for i in range(1, n_s)]
        sampled[(m, n)] = tuple(times)
        for t in times:
            moved = moving_positions(centers, kinds, params, t)
            for q in (start, goal):
                if np.any(np.linalg.norm(moved - q, axis=1) - radii < p["endpoint_clear_m"]):
                    return "endpoint_not_clear_at_sampled_time"
            if t > 0.0:
                screen = geo.weighted_geodesic(start, goal, moved, radii,
                                               inflation_m=vessel.lateral_half_extent_m + margin,
                                               cell_m=p["grid_cell_m"], bounds_m=bounds)
                if not screen.feasible:
                    return "static_snapshot_sealed_at_sampled_time"
    kinds, params = kinds_params(mu, nu)
    moving_ids = tuple(range(n_static, len(radii)))
    motion_digest = geo.geometry_digest(kind=[int(k) for k in kinds], params=params, mu=mu, nu=nu)
    return dict(start=start, goal=goal, heading=float(rng.uniform(-math.pi, math.pi)), centers=centers,
                radii=radii, field=(), structural=tuple(range(len(radii))), bounds=bounds, vias=(),
                gates=(), waypoints=(tuple(goal),),
                moving=dict(kind=tuple(int(k) for k in kinds), params=tuple(map(tuple, params)), mu=mu,
                            nu=nu, nu_tied_to_mu=nu_tied, top_speed_m_s=top, row_period_m=period,
                            base_slots=moving_ids, admitted_doses=tuple(doses),
                            sampled_times_s=sampled[(mu, nu)], motion_digest=motion_digest),
                extras=dict(timed_feasibility="D7 responsibility (UNVERIFIED in D4)"))


def _cand_harbour(rng, p, vessel):
    s = p["length_scale"]
    fs, fl = p["field_start_x_m"], p["field_length_m"]
    fe = fs + fl
    x_open = fe + p["berth_offset_m"]
    target = np.array((x_open - p["slip_opening_x_m"], 0.0))
    b_c, b_r, jambs_local, berth = _berth_structure(p, target, 0.0)
    lo, hi = p["radius_range_m"]
    e = p["endpoint_clear_m"]
    half = 0.5 * p["virtual_gate_width_m"]
    gate_x = tuple(p["departure_gate_x_m"]) + (fe,)
    need = vessel.circumscribed_radius_m + p["planning_margin_m"]
    field_start, field_exit = np.array((fs, 0.0)), np.array((fe, 0.0))

    def clear_fn(c, r):
        if np.linalg.norm(c - field_start) - r < e or np.linalg.norm(c - field_exit) - r < e:
            return False
        for gx in gate_x:
            a, b = np.array((gx, -half)), np.array((gx, half))
            if float(geo._distances_to_segment(c[None], a, b)[0]) - r < need:
                return False
        return True

    def blocker():
        return np.array((fs + rng.uniform(0.30, 0.70) * fl, 0.0))

    first = _place_field(rng, 1, (lo, hi), blocker, [], [], p["min_pair_surface_gap_m"], clear_fn)
    if first is None:
        return "blocker_placement_exhausted"
    fh = p["field_half_width_m"]
    rest = _place_field(rng, p["requested_count"] - 1, (lo, hi),
                        lambda: np.array((rng.uniform(fs, fe), rng.uniform(-fh, fh))),
                        first[0], first[1], p["min_pair_surface_gap_m"], clear_fn)
    if rest is None:
        return "field_placement_exhausted"
    f_c = np.asarray(first[0] + rest[0]).reshape(-1, 2)
    f_r = np.asarray(first[1] + rest[1])
    n_f = len(f_r)
    centers = np.vstack((f_c, b_c))
    radii = np.concatenate((f_r, b_r))
    if np.any(np.linalg.norm(f_c[:, None, :] - b_c[None, :, :], axis=-1) - f_r[:, None] - b_r[None, :] < e):
        return "field_touches_berth"
    jambs = (jambs_local[0] + n_f, jambs_local[1] + n_f)
    heading = float(rng.uniform(-math.radians(45.0), math.radians(45.0)))
    run = vessel.approach_runup_m
    vias = (np.array((gate_x[0], 0.0)), np.array((gate_x[1], 0.0)), field_start, field_exit,
            np.array((x_open - run, 0.0)), np.array((x_open, 0.0)))
    gates = tuple(GateRecord(name, "virtual", (gx, 0.0), (1.0, 0.0), 2.0 * half, 2.0 * half, 2.0 * half)
                  for name, gx in zip(("departure_1", "departure_2", "transit_exit"), gate_x)) + (
        GateRecord("berth_mouth", "compulsory", (x_open, 0.0), (1.0, 0.0), p["slip_free_width_m"],
                   p["slip_free_width_m"] - vessel.beam_m,
                   p["slip_free_width_m"] - vessel.beam_m - 2.0 * p["planning_margin_m"], jambs),)
    ext_y = max(fh + hi, berth["half_width_m"] + 2.0 * p["wall_radius_m"])
    pad = p["region_padding_m"]
    bounds = (-pad, float(np.max(b_c[:, 0]) + pad), -(ext_y + pad), ext_y + pad)
    stage = ("departure", "transit", "docking")[int(rng.integers(0, 3))]
    return dict(start=np.zeros(2), goal=target, goal_heading=0.0, heading=heading, centers=centers,
                radii=radii, field=tuple(range(n_f)), structural=tuple(range(n_f, len(radii))),
                bounds=bounds, vias=vias, gates=gates, compulsory=(jambs,), berth=berth,
                waypoints=tuple(tuple(map(float, v)) for v in vias) + (tuple(map(float, target)),),
                extras=dict(d7_stage=stage, virtual_gate_half_width_m=half,
                            gate_approach_m=p["gate_approach_m"],
                            gate_min_normal_speed_m_s=p["gate_min_normal_speed_m_s"]))


_GENERATORS = {"reach": _cand_reach, "path_following": _cand_path, "station_keeping": _cand_station,
               "docking": _cand_docking, "path_hazard": _cand_path_hazard, "scatter": _cand_scatter,
               "ring": _cand_ring, "fortress": _cand_fortress, "forced_crossing": _cand_forced,
               "iceberg": _cand_iceberg, "harbour_mission": _cand_harbour}

_D7_SITE = {"reach": "transit_corridor", "path_following": "fixed_route_segment",
            "station_keeping": "holding_region_edge", "docking": "approach_lane",
            "path_hazard": "mandatory_route_segment", "scatter": "certified_corridor",
            "ring": "escape_gap", "fortress": "entry_gap", "forced_crossing": "fixed_gate",
            "iceberg": "side_passage", "moving_basin": "native_field", "harbour_mission": "declared_stage"}


def _d7_context(task, cand, p, record, vessel, gates, centers, radii, deadline, digest, adm, extras,
                segment_index=0):
    """Plain-data mirror of D7's TaskContext fields (D7 repo
    tasks/_shared/conditions/dynamic/adapters.py:120-153, state of
    2026-09-30 ~16:40; D7 was still changing). No D7 import: the integrator
    maps this mapping onto D7 dataclasses (Pose, Circle, VesselEnvelope,
    StageSite, NativeMotion). Circles are (x, y, radius) in metres."""
    circle = lambda xy, r: (float(xy[0]), float(xy[1]), float(r))
    route, flags, stages, site_index = [], [], [], 0
    if task in ("path_following", "path_hazard"):
        for gp in cand["waypoints"][:-1]:
            route.append(circle(gp, p["gate_radius_m"]))
            flags.append(True)
        goal = circle(cand["waypoints"][-1], p["gate_radius_m"])
        site_index = segment_index
    elif task == "station_keeping":
        goal = circle(cand["station"]["center_xy"], cand["station"]["radius_m"])
    elif task in ("docking", "harbour_mission"):
        # harbour stage -> the route circle whose arrival completes it
        stage_of_gate = {"departure_2": "departure", "transit_exit": "transit", "berth_mouth": "docking"}
        for gr in gates:
            route.append(circle(gr.midpoint_m, 0.5 * gr.planned_centre_width_m))
            flags.append(True)
            if task == "harbour_mission" and gr.name in stage_of_gate:
                stages.append((stage_of_gate[gr.name], "waypoint_arrival", len(route) - 1))
        goal = circle(cand["goal"], p["position_tolerance_m"])
        if task == "harbour_mission":
            site_index = [s[0] for s in stages].index(extras["d7_stage"])
    else:
        for gr in gates:
            if gr.kind == "compulsory":
                route.append(circle(gr.midpoint_m, 0.5 * gr.planned_centre_width_m))
                flags.append(True)
        goal = circle(cand["goal"], p["goal_radius_m"])
    x0, x1, y0, y1 = vessel.bounds
    hook = dict(
        task="harbour" if task == "harbour_mission" else task, site=_D7_SITE[task],
        envelope=dict(vessel_id=record.key, x_min_m=x0, x_max_m=x1, y_min_m=y0, y_max_m=y1,
                      reference_speed_mps=record.cruise_speed_m_s, top_speed_mps=record.top_speed_m_s),
        initial_pose=(float(cand["start"][0]), float(cand["start"][1]), float(cand["heading"])),
        bounds=tuple(map(float, cand["bounds"])), goal=goal, deadline_s=float(deadline),
        route=tuple(route), route_mandatory=tuple(flags),
        static_obstacles=tuple(circle(c, r) for c, r in zip(centers, radii)),
        hold_circle=goal if task == "station_keeping" else None,
        settled_phase_start_s=float(deadline - p["hold_s"]) if task == "station_keeping" else 0.0,
        final_speed_mps=p.get("speed_tolerance_m_s") if task in ("docking", "harbour_mission") else None,
        final_heading_rad=cand.get("goal_heading"),
        heading_tolerance_rad=p.get("heading_tolerance_rad", 0.15),
        dwell_s=p["hold_s"] if task in ("docking", "harbour_mission") else 0.0,
        site_index=int(site_index), stages=tuple(stages),
        base_max=int(cfg.STATIC_CAPACITY[task]), base_count=int(len(radii)),
        base_slots=tuple(cand["moving"]["base_slots"]) if task == "moving_basin" else (),
        appearing_reserve=0, geometry_id=digest,
        # TaskContext.sensor (D7 adapters.py:139; visibility.SensorSpec:38-51):
        # every journal task has the 36-ray fan (CONTEXT_R3 decision 14).
        sensor=(dict(kind="ray_fan", ray_count=int(p["ray_sensor_ray_count"]),
                     max_range_m=float(p["ray_sensor_max_range_m"]))
                if cfg.JOURNAL_RAY_SENSOR_ALL_TASKS else None),
        # D4 additions (not TaskContext fields)
        reserved_dynamic_slots=int(cfg.D7_RESERVED_SLOTS[task]),
        witness_vertices=tuple(tuple(map(float, v)) for v in adm.witness.vertices_m),
        declared_stage=extras.get("d7_stage"))
    return MappingProxyType(hook)


def build_layout(task, tier, vessel_key, rng, *, max_attempts=200, mu=None, nu=None, cruise_speed_m_s=None):
    """Generate and admit one journal layout (see module docstring).

    vessel_key: any registered v2 profile (journal_cfg_v2.REGISTERED_VESSELS).
    cruise_speed_m_s: ANALYSIS override of the contract cruise speed (the
    cruise trade-off survey); None uses the frozen contract value."""
    if task not in cfg.TASKS:
        raise ValueError("unknown task %r" % (task,))
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator from the scenario protocol")
    p = cfg.resolve_task_geometry(task, tier, vessel_key, cruise_speed_m_s=cruise_speed_m_s)
    record = cfg.vessel_record(vessel_key, cruise_speed_m_s)
    vessel = record.geometry
    tier_obj = geo.physical_tier(tier, vessel)
    nu_tied = nu is None
    if task == "moving_basin":
        mu = cfg.MOVING_DOSE_MU[2] if mu is None else float(mu)
        nu = mu if nu is None else float(nu)
    rejections = Counter()
    for attempt in range(1, int(max_attempts) + 1):
        if task == "moving_basin":
            cand = _cand_moving(rng, p, vessel, mu, nu, nu_tied)
        else:
            cand = _GENERATORS[task](rng, p, vessel)
        if isinstance(cand, str):
            rejections[cand] += 1
            continue
        # Total = exact requested field count + recorded structural count.
        # Structural walls/rings are deterministic constructions whose realized
        # count is part of the stamped dose; field obstacles must match exactly.
        requested_total = int(p.get("requested_count", 0)) + len(cand["structural"])
        adm = geo.admit_final_layout(
            vessel=vessel, tier=tier_obj, start=cand["start"], goal=cand["goal"],
            centers=cand["centers"], radii=cand["radii"], bounds_m=cand["bounds"],
            via_points=cand.get("vias", ()),
            requested_count=requested_total,
            count_mode="exact", compulsory_gates=cand.get("compulsory", ()),
            forbid_direct_shortcut=cand.get("forbid_direct", False),
            pair_gap_indices=cand["field"] if cand["field"] else None,
            min_pair_surface_gap_m=p.get("min_pair_surface_gap_m", p.get("min_edge_gap_m")),
            start_heading_rad=cand["heading"], goal_heading_rad=cand.get("goal_heading"),
            cell_m=p["grid_cell_m"])
        reasons = list(adm.reasons)
        # field count is exact; structural counts are recorded (dose identity)
        if cand["field"] and len(cand["field"]) != p.get("requested_count", len(cand["field"])):
            reasons.append("field_count_mismatch")
        if adm.admitted and adm.witness.time_equivalent_length_m > p["route_budget_m"] + 1.0e-9:
            reasons.append("nominal_route_exceeds_budget")
        if adm.admitted and task in ("path_following", "path_hazard"):
            if not _turn_deviation_ok(adm.witness, cand["waypoints"][:-1],
                                      [p["gate_radius_m"]] * (len(cand["waypoints"]) - 1)):
                reasons.append("witness_misses_waypoint_disk")
        if reasons:
            for r in reasons:
                rejections[r.split(":")[0]] += 1
            continue
        # Nominal route = time-equivalent witness length (pivots charged R*angle).
        L_nom = adm.witness.time_equivalent_length_m
        steps = cfg.deadline_steps(L_nom, record.cruise_speed_m_s, p["dwell_s"])
        dl = steps * cfg.CONTROL_DT_S
        gates = list(cand.get("gates", ()))
        certs = {c.gate: c for c in adm.certificates}
        for i, gte in enumerate(gates):
            if gte.kind == "compulsory":
                c = certs[tuple(gte.jamb_indices)]
                gates[i] = GateRecord(gte.name, gte.kind, gte.midpoint_m, gte.through_m, c.physical_gap_m,
                                      gte.hull_centre_width_m, gte.planned_centre_width_m,
                                      gte.jamb_indices, c.valid, c.walk)
        centers = np.array(cand["centers"], dtype=np.float64).reshape(-1, 2)
        radii = np.array(cand["radii"], dtype=np.float64)
        start = np.array(cand["start"], dtype=np.float64)
        goal = np.array(cand["goal"], dtype=np.float64)
        for arr in (centers, radii, start, goal):
            arr.setflags(write=False)
        # Dose-independent: motion parameters (moving basin) carry their own
        # motion_digest; the static geometry, endpoints and start heading are
        # identical at every environmental or motion dose. The vessel record
        # digest (every registry-derived number and contract vessel parameter)
        # is part of the identity: a registry change is a new geometry version.
        digest = geo.geometry_digest(task=task, vessel=vessel_key, tier=int(tier), start=start, goal=goal,
                                     start_heading=float(cand["heading"]), centers=centers, radii=radii,
                                     bounds=list(cand["bounds"]), contract_version=cfg.CONTRACT_VERSION,
                                     vessel_record=record.digest)
        extras = dict(cand.get("extras", {}))
        if "berth" in cand:
            extras["berth"] = cand["berth"]
        if "station" in cand:
            extras["station"] = cand["station"]
            extras["late_interval_s"] = (dl - p["hold_s"], dl)
        if "moving" in cand:
            extras["moving"] = cand["moving"]
        hook = _d7_context(task, cand, p, record, vessel, gates, centers, radii, dl, digest, adm, extras,
                           segment_index=(int(rng.integers(0, p["segment_count"]))
                                          if task in ("path_following", "path_hazard") else 0))
        n_field = len(cand["field"])
        return JournalLayout(
            task=task, vessel=vessel_key, tier=int(tier), start=start,
            start_heading_rad=float(cand["heading"]), goal=goal,
            goal_heading_rad=cand.get("goal_heading"), centers=centers, radii=radii,
            field_indices=tuple(cand["field"]), structural_indices=tuple(cand["structural"]),
            requested_obstacle_count=int(p.get("requested_count", 0)) if n_field else len(radii),
            count_mode="exact" if n_field or not cand["structural"] else "recorded",
            gates=tuple(gates), waypoints=tuple(cand.get("waypoints", ())), bounds_m=tuple(cand["bounds"]),
            witness=adm.witness, nominal_route_length_m=float(L_nom), deadline_s=float(dl),
            deadline_steps=int(steps), admission=adm, attempts=attempt,
            rejections=MappingProxyType(dict(rejections)), params=p,
            extras=MappingProxyType(extras), geometry_hash=digest, d7_hook=hook)
    raise JournalAdmissionError(task, vessel_key, int(tier), int(max_attempts), rejections)
