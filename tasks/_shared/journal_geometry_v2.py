"""CPU-only journal geometry, task contract version 2 (no simulator imports).

Units are metres and radians. All positions are the hull centre of mass (COM)
in the local layout frame. A clearance of exactly zero is contact (unsafe).

Contents
  * Footprints: COM-relative convex polygon (primary) and an enclosing capsule,
    both represented as a convex core point set plus a rounding radius.
  * VesselGeometry and the ONE physical tier definition shared by every
    obstacle family (surface gap, hull-centre width, planning margin).
  * Exact translation sweeps, a conservative general swept bound (rotation and
    moving obstacles), weighted A*/Dijkstra with continuously validated edges.
  * route_witness: a curvature-bounded, oriented-hull feasibility witness.
  * compulsory_gate_certificate: a parity (even-odd) proof, on the FINAL
    geometry, that every collision-free COM path must pass a declared gate.
  * Legacy band-fortress validation and generic final-layout admission.

Nothing here is a continuous shortest path: grid routes are approximations and
"no grid route" is never used as a proof that free space is disconnected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from typing import Sequence

import numpy as np

EPS = 1.0e-9

# Default reference length for stand-alone use (test fixtures): the BlueBoat
# v2 COM-polygon length of D5's registry (vehicles_v2_data.BLUEBOAT
# 'footprint_rectangle_com_xy_m': x from -0.594 to 0.606825846 m). Journal
# vessels do NOT rely on this literal: journal_cfg_v2 derives the reference
# length from the registry at import and passes it to every VesselGeometry
# (reference_length_m), so BlueBoat journal geometry has length scale exactly 1
# and a registry change propagates (E5; CONTEXT_R3 decision 1).
L_REF_M = 1.200825846195221
# Available aligned hull-centre width per tier at length scale 1 (metres).
TIER_CENTRE_WIDTH_REF_M = (4.0, 3.0, 2.0, 1.2)
# Per-side planning margin at length scale 1 (metres); legacy value 0.20 m.
PLANNING_MARGIN_REF_M = 0.20
TIER_LEVELS = (0, 1, 2, 3)


# --------------------------------------------------------------------------
# small validated helpers
# --------------------------------------------------------------------------

def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return value


def _nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be finite and nonnegative")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(name + " must be finite and nonnegative")
    return value


def _point(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("expected one finite XY point")
    return value


def _obstacles(centers, radii):
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if radii.shape != (len(centers),):
        raise ValueError("obstacle centre/radius counts differ")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        raise ValueError("obstacle geometry must be finite")
    if np.any(radii <= 0.0):
        raise ValueError("obstacle radii must be positive")
    return centers, radii


def _level(level):
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or int(level) not in TIER_LEVELS:
        raise ValueError("geometry tier must be the integer 0, 1, 2 or 3")
    return int(level)


def _orient(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _distances_to_segment(points, start, end):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    delta = end - start
    norm = float(delta @ delta)
    if norm == 0.0:
        return np.linalg.norm(points - start, axis=-1)
    fraction = np.clip((points - start) @ delta / norm, 0.0, 1.0)
    return np.linalg.norm(points - start - fraction[:, None] * delta, axis=-1)


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(((c, -s), (s, c)), dtype=np.float64)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def convex_hull(points):
    """Counter-clockwise convex hull (Andrew monotone chain), collinear dropped."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).reshape(-1, 2).tolist())))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _orient(lower[-2], lower[-1], p) <= EPS:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _orient(upper[-2], upper[-1], p) <= EPS:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _signed_distance_to_hull(points_query, hull):
    """Signed distance from query points to a convex CCW hull (inside < 0)."""
    q = np.asarray(points_query, dtype=np.float64).reshape(-1, 2)
    n = len(hull)
    if n == 1:
        return np.linalg.norm(q - hull[0], axis=1)
    if n == 2:
        return _distances_to_segment(q, hull[0], hull[1])
    minimum = np.full(len(q), np.inf)
    inside = np.ones(len(q), dtype=bool)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        minimum = np.minimum(minimum, _distances_to_segment(q, a, b))
        edge = b - a
        rel = q - a
        inside &= (edge[0] * rel[:, 1] - edge[1] * rel[:, 0]) >= 0.0
    return np.where(inside, -minimum, minimum)


# --------------------------------------------------------------------------
# footprints
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PolygonFootprint:
    """Strictly convex COM-relative body polygon (bow +x, port +y), metres."""

    vertices_m: tuple

    def __post_init__(self):
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("polygon requires at least three XY vertices")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("non-finite polygon")
        area2 = float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                             - np.roll(vertices[:, 0], -1) * vertices[:, 1]))
        if area2 < 0.0:
            vertices = vertices[::-1]
        edges = np.roll(vertices, -1, axis=0) - vertices
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if not np.all(cross > 0.0):
            raise ValueError("polygon must be strictly convex, simple and ordered")
        object.__setattr__(self, "vertices_m", tuple(tuple(map(float, p)) for p in vertices))

    @classmethod
    def from_bounds(cls, x_min, x_max, y_min, y_max):
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("invalid rectangle bounds")
        return cls(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

    @classmethod
    def from_dimensions(cls, length_m, beam_m, center_xy_m=(0.0, 0.0)):
        """Rectangle of given size centred at center_xy_m (COM frame)."""
        x = _positive(length_m, "length_m") / 2.0
        y = _positive(beam_m, "beam_m") / 2.0
        cx, cy = _point(center_xy_m)
        return cls.from_bounds(cx - x, cx + x, cy - y, cy + y)

    @classmethod
    def from_body_polygon(cls, body_vertices_m, com_xy_m):
        """Translate an asset/body-frame polygon into the COM frame."""
        com = _point(com_xy_m)
        return cls(tuple(tuple(np.asarray(v, dtype=np.float64) - com) for v in body_vertices_m))

    @property
    def core_points(self):
        return np.asarray(self.vertices_m, dtype=np.float64)

    @property
    def rounding_m(self):
        return 0.0

    @property
    def bounds(self):
        v = self.core_points
        return float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max())

    @property
    def radius_m(self):
        """Circumscribed radius about the COM."""
        return float(np.max(np.linalg.norm(self.core_points, axis=1)))

    @property
    def inscribed_radius_m(self):
        """Largest COM-centred disk inside the footprint (0 if COM outside)."""
        d = _signed_distance_to_hull(np.zeros((1, 2)), self.core_points)[0]
        return float(-d) if d < 0.0 else 0.0


@dataclass(frozen=True)
class CapsuleFootprint:
    """Stadium: spine segment along body x through center_xy_m, plus radius.

    ``enclosing(polygon)`` spans the polygon's full x extent with radius equal
    to the largest lateral offset, so it encloses the polygon including its
    corners (total length L+B for a rectangle). A capsule whose TOTAL length is
    L clips rectangle corners and is not a conservative substitute.
    """

    center_xy_m: tuple
    half_length_m: float
    radius_m_: float

    def __post_init__(self):
        c = _point(self.center_xy_m)
        object.__setattr__(self, "center_xy_m", (float(c[0]), float(c[1])))
        _nonnegative(self.half_length_m, "half_length_m")
        _positive(self.radius_m_, "capsule radius")

    @classmethod
    def enclosing(cls, polygon: PolygonFootprint):
        x_min, x_max, y_min, y_max = polygon.bounds
        cy = 0.5 * (y_min + y_max)
        return cls(((x_min + x_max) / 2.0, cy), (x_max - x_min) / 2.0,
                   max(y_max - cy, cy - y_min))

    @property
    def core_points(self):
        cx, cy = self.center_xy_m
        return np.array(((cx - self.half_length_m, cy), (cx + self.half_length_m, cy)))

    @property
    def rounding_m(self):
        return float(self.radius_m_)

    @property
    def radius_m(self):
        return float(np.max(np.linalg.norm(self.core_points, axis=1))) + self.rounding_m

    def contains_polygon(self, polygon: PolygonFootprint):
        a, b = self.core_points
        d = _distances_to_segment(polygon.core_points, a, b)
        return bool(np.all(d <= self.rounding_m + EPS))


def _core_radius(footprint):
    return float(np.max(np.linalg.norm(footprint.core_points, axis=1)))


def _placed_core(footprint, pose):
    pose = np.asarray(pose, dtype=np.float64)
    return footprint.core_points @ _rotation(float(pose[2])).T + pose[:2]


def pose_clearance(footprint, pose, centers, radii):
    """Exact minimum signed separation between the posed footprint and disks.

    Invalid pose or obstacle data fail closed (-inf).
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        return -math.inf
    try:
        centers, radii = _obstacles(centers, radii)
    except ValueError:
        return -math.inf
    if not len(radii):
        return math.inf
    hull = convex_hull(_placed_core(footprint, pose))
    return float(np.min(_signed_distance_to_hull(centers, hull) - footprint.rounding_m - radii))


def translation_sweep_clearance(footprint, start_xy, end_xy, heading_rad, centers, radii):
    """Clearance of a fixed-heading straight translation via its convex sweep.

    When positive it is EXACT (the minimum over every intermediate pose).
    When the motion collides it returns the penetration of the swept region,
    which can be deeper than any single pose: still <= 0, i.e. conservative.
    """
    a, b = _point(start_xy), _point(end_xy)
    if not math.isfinite(heading_rad):
        return -math.inf
    try:
        centers, radii = _obstacles(centers, radii)
    except ValueError:
        return -math.inf
    if not len(radii):
        return math.inf
    body = footprint.core_points @ _rotation(float(heading_rad)).T
    hull = convex_hull(np.vstack((body + a, body + b)))
    return float(np.min(_signed_distance_to_hull(centers, hull) - footprint.rounding_m - radii))


def segment_clearance(start, end, centers, radii, inflation_m=0.0):
    """Exact minimum clearance of a swept disk (radius inflation_m) on a segment."""
    start, end = _point(start), _point(end)
    centers, radii = _obstacles(centers, radii)
    inflation_m = _nonnegative(inflation_m, "inflation")
    if not len(radii):
        return math.inf
    return float(np.min(_distances_to_segment(centers, start, end) - radii - inflation_m))


def swept_clearance(footprint, start_pose, end_pose, centers, radii, *,
                    max_depth=14, resolution_m=1.0e-4, end_centers_m=None):
    """Conservative lower bound for linear XY / shortest-angle yaw motion.

    Midpoint clearance minus a displacement bound covers every intermediate
    pose (the signed distance is 1-Lipschitz in rigid displacement). Adaptive
    bisection tightens it; a colliding midpoint returns immediately; if the
    budget cannot certify a positive gap the returned bound is nonpositive.
    Optional end_centers_m moves the circles linearly over the same interval.
    Call once per physics substep; unobserved nonlinear motion between two
    samples is NOT reconstructed.
    """
    a, b = np.asarray(start_pose, dtype=np.float64), np.asarray(end_pose, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return -math.inf
    if max_depth < 0 or not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("invalid subdivision budget")
    try:
        first, radii = _obstacles(centers, radii)
        last, _ = _obstacles(first if end_centers_m is None else end_centers_m, radii)
    except ValueError:
        return -math.inf
    if not len(radii):
        return math.inf
    b = b.copy()
    b[2] = a[2] + _wrap(b[2] - a[2])
    core_r = _core_radius(footprint)
    best = math.inf
    stack = [(a, b, first, last, 0)]
    while stack:
        left, right, lc, rc, depth = stack.pop()
        mid = 0.5 * (left + right)
        mc = 0.5 * (lc + rc)
        motion = (float(np.linalg.norm(right[:2] - left[:2])) / 2.0
                  + core_r * abs(right[2] - left[2]) / 2.0
                  + float(np.max(np.linalg.norm(rc - lc, axis=1))) / 2.0)
        clearance = pose_clearance(footprint, mid, mc, radii)
        lower = clearance - motion
        if lower > 0.0 or motion <= resolution_m or depth >= max_depth:
            best = min(best, lower)
        elif clearance <= 0.0:
            return min(lower, 0.0)
        else:
            stack.append((left, mid, lc, mc, depth + 1))
            stack.append((mid, right, mc, rc, depth + 1))
    return best


# --------------------------------------------------------------------------
# vessel and the shared tier definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VesselGeometry:
    """Frozen geometric/manoeuvre inputs for admission. Positions are COM.

    turning_radius_m is the DECLARED design turning radius used by the
    curvature-bounded witness (UNVERIFIED until the D6 steady-turn test);
    stop_distance_m is the declared stopping distance from cruise.
    capsule_footprint is an optional recorded enclosing capsule (e.g. D5's
    registry capsule); it must contain the polygon (checked here). Without it
    ``capsule`` is the full-length enclosing construction.
    reference_length_m is the length at which the scale s = L / L_ref is 1
    (the BlueBoat registry polygon length for journal vessels).
    """

    name: str
    footprint: PolygonFootprint
    turning_radius_m: float
    cruise_speed_m_s: float
    stop_distance_m: float = 0.0
    capsule_footprint: CapsuleFootprint | None = None
    reference_length_m: float = L_REF_M

    def __post_init__(self):
        if not isinstance(self.footprint, PolygonFootprint):
            raise TypeError("footprint must be a PolygonFootprint")
        _positive(self.turning_radius_m, "turning_radius_m")
        _positive(self.cruise_speed_m_s, "cruise_speed_m_s")
        _nonnegative(self.stop_distance_m, "stop_distance_m")
        _positive(self.reference_length_m, "reference_length_m")
        if self.footprint.inscribed_radius_m <= 0.0:
            raise ValueError("the COM must lie strictly inside the footprint")
        if self.capsule_footprint is not None:
            if not isinstance(self.capsule_footprint, CapsuleFootprint):
                raise TypeError("capsule_footprint must be a CapsuleFootprint")
            if not self.capsule_footprint.contains_polygon(self.footprint):
                raise ValueError("the recorded capsule must contain the collision polygon")

    @classmethod
    def from_dimensions(cls, name, length_m, beam_m, turning_radius_m, cruise_speed_m_s=1.0,
                        stop_distance_m=0.0, com_offset_xy_m=(0.0, 0.0)):
        """Test fixture helper: rectangle whose centre sits at -com_offset."""
        off = _point(com_offset_xy_m)
        return cls(name, PolygonFootprint.from_dimensions(length_m, beam_m, -off),
                   turning_radius_m, cruise_speed_m_s, stop_distance_m)

    @property
    def bounds(self):
        return self.footprint.bounds

    @property
    def length_m(self):
        x0, x1, _, _ = self.bounds
        return x1 - x0

    @property
    def beam_m(self):
        _, _, y0, y1 = self.bounds
        return y1 - y0

    @property
    def forward_extent_m(self):
        return self.bounds[1]

    @property
    def aft_extent_m(self):
        return -self.bounds[0]

    @property
    def lateral_half_extent_m(self):
        _, _, y0, y1 = self.bounds
        return max(-y0, y1)

    @property
    def circumscribed_radius_m(self):
        return self.footprint.radius_m

    @property
    def inscribed_radius_m(self):
        return self.footprint.inscribed_radius_m

    @property
    def capsule(self):
        if self.capsule_footprint is not None:
            return self.capsule_footprint
        return CapsuleFootprint.enclosing(self.footprint)

    @property
    def length_scale(self):
        """s = L / L_ref. The ONE vessel scale for all journal lengths."""
        return self.length_m / self.reference_length_m

    @property
    def approach_runup_m(self):
        """Straight alignment run-up before a mandatory opening."""
        return max(2.0 * self.length_m, self.turning_radius_m, self.stop_distance_m)


@dataclass(frozen=True)
class PhysicalTier:
    """One shared definition for every obstacle family (all metres).

    physical_surface_gap_m   g = min surface-to-surface distance across the
                             named cross-section (never centre distance, arc
                             length or an already-inflated value);
    hull_centre_width_m      w = g - B (COM lateral room, hull aligned);
    planning_margin_m        m per side; planned_centre_width_m = w - 2m;
    centred_heading_tolerance_rad: largest |heading error| at which the hull
                             rectangle fits g - 2m when centred (pi/2 = any).
    """

    level: int
    physical_surface_gap_m: float
    hull_centre_width_m: float
    planning_margin_m: float
    planned_centre_width_m: float
    length_scale: float
    centred_heading_tolerance_rad: float

    def __post_init__(self):
        _level(self.level)
        for name in ("physical_surface_gap_m", "hull_centre_width_m", "planning_margin_m",
                     "planned_centre_width_m", "length_scale"):
            _positive(getattr(self, name), name)
        if abs(self.hull_centre_width_m - 2.0 * self.planning_margin_m
               - self.planned_centre_width_m) > 1.0e-9:
            raise ValueError("inconsistent planned centre width")


def centred_heading_tolerance(length_m, beam_m, width_m):
    """Largest heading error for which an L x B rectangle fits a width."""
    length_m, beam_m = _positive(length_m, "L"), _positive(beam_m, "B")
    diagonal = math.hypot(length_m, beam_m)
    if width_m >= diagonal:
        return math.pi / 2.0
    if width_m < beam_m:
        return 0.0
    return math.asin(width_m / diagonal) - math.atan2(beam_m, length_m)


def physical_tier(level, vessel: VesselGeometry) -> PhysicalTier:
    """Resolve G{level} for a vessel.

    w = c_k * s with c = (4, 3, 2, 1.2) m and s = L/L_REF; g = B + w;
    m = 0.20 * s. To first order the centred heading tolerance of an aligned
    passage is w/L, so scaling w by LENGTH keeps that tolerance equal across
    vessels, while the vessel's own beam enters g exactly. Turning radius and
    stopping distance are admission constraints (route_witness), not scale
    factors, so they cannot silently rescale a published tier.
    """
    level = _level(level)
    s = vessel.length_scale
    w = TIER_CENTRE_WIDTH_REF_M[level] * s
    m = PLANNING_MARGIN_REF_M * s
    g = vessel.beam_m + w
    return PhysicalTier(level, g, w, m, w - 2.0 * m, s,
                        centred_heading_tolerance(vessel.length_m, vessel.beam_m, g - 2.0 * m))


# --------------------------------------------------------------------------
# grid routes (approximations with continuously validated edges)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteResult:
    length_m: float
    points_m: tuple
    expanded_nodes: int
    approximation: str = "weighted_grid_segment_validated"

    @property
    def feasible(self):
        return math.isfinite(self.length_m)


def _grid_node_clearance(xs, ys, centers, radii, inflation_m):
    """Minimum inflated-disk clearance at every grid node (height, width)."""
    out = np.full((len(ys), len(xs)), np.inf)
    if not len(radii):
        return out
    gx, gy = np.meshgrid(xs, ys)
    flat = np.stack((gx.ravel(), gy.ravel()), axis=1)
    res = np.empty(len(flat))
    step = max(1, 4_000_000 // max(1, len(radii)))
    for k in range(0, len(flat), step):
        chunk = flat[k:k + step]
        d = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=-1) - radii[None, :] - inflation_m
        res[k:k + step] = np.min(d, axis=1)
    return res.reshape(len(ys), len(xs))


def weighted_geodesic(start, goal, centers, radii, *, inflation_m, cell_m=0.5, padding_m=6.0,
                      algorithm="astar", bounds_m=None, max_grid_nodes=2_000_000,
                      segment_allowed=None) -> RouteResult:
    """Metric A*/Dijkstra on an aligned 8-connected grid with exact edge checks.

    Orthogonal cost h, diagonal sqrt(2) h. Every graph edge and both
    endpoint-to-node connectors are validated against the continuous inflated
    disks (and the optional segment_allowed domain predicate). A failed
    connector rejects the query (no unsafe re-snapping). With bounds_m the
    grid is the frozen hull-centre rectangle; without, it is padded around the
    geometry (legacy comparisons only).
    """
    start, goal = _point(start), _point(goal)
    centers, radii = _obstacles(centers, radii)
    cell_m, padding_m = _positive(cell_m, "cell_m"), _positive(padding_m, "padding_m")
    inflation_m = _nonnegative(inflation_m, "inflation")
    if algorithm not in ("astar", "dijkstra"):
        raise ValueError("algorithm must be astar or dijkstra")
    unreachable = RouteResult(math.inf, (), 0)
    if bounds_m is None:
        pts = [start[None, :], goal[None, :]]
        if len(radii):
            pts += [centers - (radii + inflation_m)[:, None], centers + (radii + inflation_m)[:, None]]
        points = np.vstack(pts)
        low = np.floor((np.min(points, axis=0) - padding_m) / cell_m) * cell_m
        high = np.ceil((np.max(points, axis=0) + padding_m) / cell_m) * cell_m
    else:
        if len(bounds_m) != 4 or not np.all(np.isfinite(bounds_m)):
            raise ValueError("bounds must be xmin,xmax,ymin,ymax")
        low = np.ceil(np.array((bounds_m[0], bounds_m[2])) / cell_m) * cell_m
        high = np.floor(np.array((bounds_m[1], bounds_m[3])) / cell_m) * cell_m
        if np.any(start < low - cell_m) or np.any(start > high + cell_m) \
                or np.any(goal < low - cell_m) or np.any(goal > high + cell_m):
            return unreachable
    dims = np.rint((high - low) / cell_m).astype(int) + 1
    if np.any(dims <= 0) or int(np.prod(dims)) > max_grid_nodes:
        raise ValueError("invalid or excessive grid size")
    width, height = map(int, dims)

    def index(point):
        col, row = np.rint((point - low) / cell_m).astype(int)
        col = min(max(int(col), 0), width - 1)
        row = min(max(int(row), 0), height - 1)
        return row * width + col

    def position(node):
        row, col = divmod(node, width)
        return low + np.array((col, row), dtype=np.float64) * cell_m

    def clear(p, q):
        if segment_allowed is not None and not bool(segment_allowed(p, q)):
            return False
        return (not len(radii)) or segment_clearance(p, q, centers, radii, inflation_m) > 0.0

    node_clear = _grid_node_clearance(low[0] + np.arange(width) * cell_m,
                                      low[1] + np.arange(height) * cell_m, centers, radii, inflation_m)

    def edge_clear(a_node, b_node, length):
        # Exact fast accept: every edge point lies within length/2 of an
        # endpoint and clearance is 1-Lipschitz, so this implies a clear edge.
        ra, ca = divmod(a_node, width)
        rb, cb = divmod(b_node, width)
        if segment_allowed is None and min(node_clear[ra, ca], node_clear[rb, cb]) > 0.5 * length + 1.0e-12:
            return True
        return clear(position(a_node), position(b_node))

    first, last = index(start), index(goal)
    if not clear(start, position(first)) or not clear(position(last), goal):
        return unreachable
    distances = {first: 0.0}
    parents = {}

    def heuristic(node):
        return float(np.linalg.norm(position(node) - position(last))) if algorithm == "astar" else 0.0

    queue = [(heuristic(first), 0.0, first)]
    expanded = 0
    offsets = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost > distances[node]:
            continue
        expanded += 1
        if node == last:
            nodes = [last]
            while nodes[-1] != first:
                nodes.append(parents[nodes[-1]])
            points = [start] + [position(n) for n in reversed(nodes)] + [goal]
            length = sum(float(np.linalg.norm(q - p)) for p, q in zip(points, points[1:]))
            return RouteResult(length, tuple(tuple(map(float, p)) for p in points), expanded)
        row, col = divmod(node, width)
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            nxt = nr * width + nc
            step_len = cell_m * math.hypot(dr, dc)
            new_cost = cost + step_len
            if new_cost >= distances.get(nxt, math.inf):
                continue
            if not edge_clear(node, nxt, step_len):
                continue
            distances[nxt], parents[nxt] = new_cost, node
            heapq.heappush(queue, (new_cost + heuristic(nxt), new_cost, nxt))
    return RouteResult(math.inf, (), expanded)


@dataclass(frozen=True)
class DistanceField:
    xs_m: np.ndarray
    ys_m: np.ndarray
    distances_m: np.ndarray
    approximation: str = "weighted_grid_segment_validated"


def weighted_distance_field(goal, centers, radii, *, bounds_m, inflation_m, cell_m=0.5,
                            segment_allowed=None, max_grid_nodes=2_000_000) -> DistanceField:
    """Dijkstra guidance field with the same continuous edge validation.

    Values are at grid nodes including a validated goal connector. Inf means
    unreachable ON THIS GRID, not continuous disconnection. Queries between
    nodes must validate their own connector (no bilinear blending across
    occupied cells is provided).
    """
    goal = _point(goal)
    centers, radii = _obstacles(centers, radii)
    cell_m = _positive(cell_m, "cell_m")
    inflation_m = _nonnegative(inflation_m, "inflation")
    if len(bounds_m) != 4 or not np.all(np.isfinite(bounds_m)):
        raise ValueError("invalid bounds")
    low = np.ceil(np.array((bounds_m[0], bounds_m[2])) / cell_m) * cell_m
    high = np.floor(np.array((bounds_m[1], bounds_m[3])) / cell_m) * cell_m
    dims = np.rint((high - low) / cell_m).astype(int) + 1
    if np.any(dims <= 0) or int(np.prod(dims)) > max_grid_nodes:
        raise ValueError("invalid or excessive grid size")
    width, height = map(int, dims)
    xs = low[0] + np.arange(width) * cell_m
    ys = low[1] + np.arange(height) * cell_m
    dist = np.full((height, width), math.inf)
    if np.any(goal < low) or np.any(goal > high):
        return DistanceField(xs, ys, dist)
    col, row = map(int, np.rint((goal - low) / cell_m))
    snapped = np.array((xs[col], ys[row]))

    def clear(a, b):
        if segment_allowed is not None and not bool(segment_allowed(a, b)):
            return False
        return (not len(radii)) or segment_clearance(a, b, centers, radii, inflation_m) > 0.0

    if not clear(snapped, goal):
        return DistanceField(xs, ys, dist)
    node_clear = _grid_node_clearance(xs, ys, centers, radii, inflation_m)
    dist[row, col] = float(np.linalg.norm(snapped - goal))
    queue = [(dist[row, col], row, col)]
    moves = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while queue:
        cost, row, col = heapq.heappop(queue)
        if cost > dist[row, col]:
            continue
        here = np.array((xs[col], ys[row]))
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            step_len = cell_m * math.hypot(dr, dc)
            candidate = cost + step_len
            if candidate >= dist[nr, nc]:
                continue
            fast = (segment_allowed is None
                    and min(node_clear[row, col], node_clear[nr, nc]) > 0.5 * step_len + 1.0e-12)
            if fast or clear(here, np.array((xs[nc], ys[nr]))):
                dist[nr, nc] = candidate
                heapq.heappush(queue, (candidate, nr, nc))
    return DistanceField(xs, ys, dist)


# --------------------------------------------------------------------------
# curvature-bounded oriented-hull route witness
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteWitness:
    """Verified kinematic feasibility witness for the oriented hull.

    Straight segments at fixed heading are joined, at each interior vertex,
    by a circular fillet of the declared turning radius when it fits and is
    collision-free, otherwise by a pivot about the COM (both journal vessels
    are differential-thrust catamarans able to turn in place; a pivot is
    verified by a swept rotation). A start pivot aligns the initial heading.
    min_clearance_m is the PHYSICAL lower bound (>= margin_m when valid).
    time_equivalent_length_m = length_m + R * (total pivot angle): pivots
    are charged the arc length a cruise turn would need; used for deadlines.
    """

    feasible: bool
    length_m: float
    vertices_m: tuple
    turn_radius_m: float
    margin_m: float
    min_clearance_m: float
    reasons: tuple = ()
    n_fillets: int = 0
    n_pivots: int = 0
    pivot_angle_rad: float = 0.0
    pivot_vertex_indices: tuple = ()
    kind: str = "oriented_hull_fillet_or_pivot"

    @property
    def time_equivalent_length_m(self):
        return self.length_m + self.turn_radius_m * self.pivot_angle_rad

    @property
    def segment_headings_rad(self):
        v = np.asarray(self.vertices_m, dtype=np.float64)
        return tuple(float(math.atan2(*(q - p)[::-1])) for p, q in zip(v, v[1:]))


def _straight_ok(vessel, a, b, centers, radii, margin):
    """Exact oriented translation check with margin; returns clearance-margin."""
    if not len(radii):
        return math.inf
    d = b - a
    if float(np.hypot(*d)) <= EPS:
        # A degenerate segment is a single pose of unknown heading: require
        # the whole circumscribed disk to be clear (heading-independent).
        return segment_clearance(a, a, centers, radii + margin, vessel.circumscribed_radius_m)
    heading = math.atan2(d[1], d[0])
    # Cheap necessary test: COM segment vs inscribed disk.
    if segment_clearance(a, b, centers, radii + margin, vessel.inscribed_radius_m) <= 0.0:
        return -1.0
    return translation_sweep_clearance(vessel.footprint, a, b, heading, centers, radii + margin)


def _shortcut(vessel, points, centers, radii, margin):
    points = [np.asarray(p, dtype=np.float64) for p in points]
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and _straight_ok(vessel, points[i], points[j], centers, radii, margin) <= 0.0:
            j -= 1
        out.append(points[j])
        i = j
    return out


def route_witness(points, centers, radii, vessel: VesselGeometry, *, margin_m, bounds_m,
                  cell_m=None, turn_radius_m=None, start_heading_rad=None,
                  candidate_inflation_m=None, keep_points=None, max_arc_step_rad=math.radians(4.0),
                  max_depth=12) -> RouteWitness:
    """Build and VERIFY a curvature-bounded witness through ordered points.

    points: mandatory COM points (start, vias, goal). Each leg first tries the
    straight segment; otherwise a weighted grid route is string-pulled with
    EXACT oriented translation checks. Grid candidates are generated first at
    the orientation-free inflation (circumscribed radius + margin) and, if
    that fails, at the narrow inflation (lateral half extent + margin); the
    candidate inflation only proposes routes, it certifies nothing. At each
    interior vertex a fillet of radius R is used if its tangent length fits
    its share of both adjacent segments and its sub-arc swept bounds (minus
    the chord sagitta bound) are clear; otherwise a pivot about the COM,
    verified by a swept rotation, is used and recorded. A start pivot (from
    start_heading_rad) is verified the same way. Every check uses the full
    oriented footprint with obstacles inflated by margin_m; fails closed.
    """
    pts = [_point(p) for p in points]
    if len(pts) < 2:
        raise ValueError("a route needs at least a start and a goal")
    centers, radii = _obstacles(centers, radii)
    margin = _nonnegative(margin_m, "margin_m")
    R = vessel.turning_radius_m if turn_radius_m is None else _positive(turn_radius_m, "turn radius")
    cell = _positive(cell_m if cell_m is not None else 0.25 * vessel.length_scale, "cell_m")
    if candidate_inflation_m is None:
        candidates = (vessel.circumscribed_radius_m + margin, vessel.lateral_half_extent_m + margin)
    else:
        candidates = (_nonnegative(candidate_inflation_m, "candidate inflation"),)
    if len(bounds_m) != 4 or not np.all(np.isfinite(bounds_m)):
        raise ValueError("bounds must be xmin,xmax,ymin,ymax")
    last = None
    for cand in candidates:
        last = _witness_attempt(pts, centers, radii, vessel, margin, bounds_m, cell, R, cand,
                                start_heading_rad, keep_points, max_arc_step_rad, max_depth)
        if last.feasible:
            return last
    return last


def _witness_attempt(pts, centers, radii, vessel, margin, bounds_m, cell, R, cand,
                     start_heading_rad, keep_points, max_arc_step_rad, max_depth):
    x0, x1, y0, y1 = map(float, bounds_m)

    def fail(reason):
        return RouteWitness(False, math.inf, tuple(tuple(map(float, p)) for p in pts), R, margin,
                            -math.inf, (reason,))

    def inside(p):
        return x0 - EPS <= p[0] <= x1 + EPS and y0 - EPS <= p[1] <= y1 + EPS

    if not all(inside(p) for p in pts):
        return fail("mandatory_point_outside_domain")
    keep = set(range(len(pts))) if keep_points is None else set(keep_points)
    polyline = [pts[0]]
    mandatory = {0}
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        if _straight_ok(vessel, a, b, centers, radii, margin) > 0.0:
            leg = [a, b]
        else:
            route = weighted_geodesic(a, b, centers, radii, inflation_m=cand, cell_m=cell,
                                      bounds_m=bounds_m)
            if not route.feasible:
                return fail("no_candidate_route_leg_%d" % k)
            leg = _shortcut(vessel, route.points_m, centers, radii, margin)
        for p, q in zip(leg, leg[1:]):
            if _straight_ok(vessel, p, q, centers, radii, margin) <= 0.0:
                return fail("oriented_segment_collision_leg_%d" % k)
        polyline.extend(leg[1:])
        if (k + 1) in keep:
            mandatory.add(len(polyline) - 1)
    # drop duplicate/collinear vertices but never a mandatory via point
    cleaned = [polyline[0]]
    kept_mandatory = {0}
    for idx in range(1, len(polyline)):
        p = polyline[idx]
        if float(np.linalg.norm(p - cleaned[-1])) <= 1.0e-7:
            if idx in mandatory:
                kept_mandatory.add(len(cleaned) - 1)
            continue
        if len(cleaned) >= 2 and (len(cleaned) - 1) not in kept_mandatory:
            u, v = cleaned[-1] - cleaned[-2], p - cleaned[-1]
            cross = u[0] * v[1] - u[1] * v[0]
            if abs(cross) <= 1.0e-9 * max(1.0, float(np.linalg.norm(u) * np.linalg.norm(v))) \
                    and float(u @ v) > 0.0:
                cleaned[-1] = p
                if idx in mandatory:
                    kept_mandatory.add(len(cleaned) - 1)
                continue
        cleaned.append(p)
        if idx in mandatory:
            kept_mandatory.add(len(cleaned) - 1)
    v = cleaned
    if not all(inside(p) for p in v):
        return fail("witness_leaves_domain")
    n = len(v)
    seg = [v[i + 1] - v[i] for i in range(n - 1)]
    lens = [float(np.linalg.norm(d)) for d in seg]
    heads = [math.atan2(d[1], d[0]) for d in seg]
    min_clear = math.inf
    for i in range(n - 1):
        min_clear = min(min_clear, _straight_ok(vessel, v[i], v[i + 1], centers, radii, margin))
    turns = [0.0] * n
    tangents = [0.0] * n
    pivots = []

    def pivot_clear(i):
        return swept_clearance(vessel.footprint, (v[i][0], v[i][1], heads[i - 1]),
                               (v[i][0], v[i][1], heads[i]), centers, radii + margin,
                               max_depth=max_depth, resolution_m=1.0e-3) if len(radii) else math.inf

    def fillet_clear(i, t):
        theta = turns[i]
        sign = 1.0 if _wrap(heads[i] - heads[i - 1]) > 0.0 else -1.0
        u_in = seg[i - 1] / lens[i - 1]
        p_start = v[i] - t * u_in
        centre = p_start + R * np.array((-u_in[1], u_in[0])) * sign
        steps = max(1, int(math.ceil(theta / max_arc_step_rad)))
        delta = theta / steps
        sag = R * (1.0 - math.cos(delta / 2.0)) + R * delta ** 3 / 12.0
        poses = []
        for k in range(steps + 1):
            # rotate the arc start about the fillet centre by the swept angle
            pos = centre + (p_start - centre) @ _rotation(sign * delta * k).T
            if not inside(pos):
                return -math.inf
            poses.append((pos[0], pos[1], heads[i - 1] + sign * delta * k))
        if not len(radii):
            return math.inf
        worst = math.inf
        for p, q in zip(poses, poses[1:]):
            worst = min(worst, swept_clearance(vessel.footprint, p, q, centers, radii + margin,
                                               max_depth=max_depth, resolution_m=1.0e-3) - sag)
            if worst <= 0.0:
                return worst
        return worst

    for i in range(1, n - 1):
        turns[i] = abs(_wrap(heads[i] - heads[i - 1]))
    for i in range(1, n - 1):
        theta = turns[i]
        if theta <= 1.0e-9:
            continue
        # Each interior segment's length is shared by the fillets at its two
        # ends; the first/last segment has no fillet at the start/goal end.
        t = R * math.tan(theta / 2.0) if theta < math.pi - 1.0e-6 else math.inf
        left = lens[i - 1] if i - 1 == 0 else 0.5 * lens[i - 1]
        right = lens[i] if i + 1 == n - 1 else 0.5 * lens[i]
        c = fillet_clear(i, t) if t <= min(left, right) + 1.0e-12 else -math.inf
        if c > 0.0:
            tangents[i] = t
            min_clear = min(min_clear, c)
            continue
        c = pivot_clear(i)
        if c <= 0.0:
            return fail("no_clear_fillet_or_pivot_at_vertex_%d" % i)
        pivots.append(i)
        min_clear = min(min_clear, c)
    if start_heading_rad is not None and n >= 2 and len(radii):
        if not math.isfinite(start_heading_rad):
            return fail("invalid_start_heading")
        c = swept_clearance(vessel.footprint, (v[0][0], v[0][1], float(start_heading_rad)),
                            (v[0][0], v[0][1], heads[0]), centers, radii + margin,
                            max_depth=max_depth, resolution_m=1.0e-3)
        if c <= 0.0:
            return fail("start_alignment_collision")
        min_clear = min(min_clear, c)
    filleted = [i for i in range(1, n - 1) if tangents[i] > 0.0]
    length = sum(lens) - 2.0 * sum(tangents) + sum(R * turns[i] for i in filleted)
    pivot_angle = sum(turns[i] for i in pivots)
    if start_heading_rad is not None and n >= 2:
        pivot_angle += abs(_wrap(heads[0] - float(start_heading_rad)))
    return RouteWitness(True, float(length), tuple(tuple(map(float, p)) for p in v), R, margin,
                        float(min_clear + margin) if math.isfinite(min_clear) else math.inf, (),
                        len(filleted), len(pivots), float(pivot_angle), tuple(pivots))


# --------------------------------------------------------------------------
# compulsory-gate certificate (parity / even-odd proof)
# --------------------------------------------------------------------------

def _proper_cross(a, b, c, d, tol=1.0e-12):
    """1 proper crossing, 0 disjoint, -1 degenerate (touching/collinear)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    scale = max(1.0, float(np.max(np.abs(np.concatenate((a, b, c, d))))))
    t = tol * scale * scale
    if (abs(o1) <= t or abs(o2) <= t or abs(o3) <= t or abs(o4) <= t):
        # degenerate only if the segments actually touch
        if (min(a[0], b[0]) - t <= max(c[0], d[0]) and min(c[0], d[0]) - t <= max(a[0], b[0])
                and min(a[1], b[1]) - t <= max(c[1], d[1]) and min(c[1], d[1]) - t <= max(a[1], b[1])):
            if (o1 * o2 <= t) and (o3 * o4 <= t):
                return -1
        return 0
    return 1 if (o1 * o2 < 0.0 and o3 * o4 < 0.0) else 0


def _even_odd_inside(point, polyline):
    """Even-odd parity of a closed polyline (list of vertices, implicitly closed)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polyline)
    for k in range(n):
        a, b = polyline[k], polyline[(k + 1) % n]
        if (a[1] > y) != (b[1] > y):
            xc = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < xc:
                inside = not inside
    return inside


@dataclass(frozen=True)
class GateCertificate:
    valid: bool
    gate: tuple
    walk: tuple
    physical_gap_m: float
    com_opening_m: float
    reasons: tuple = ()


def compulsory_gate_certificate(start, goal, centers, radii, gate, inscribed_radius_m,
                                *, overlap_tolerance_m=1.0e-9) -> GateCertificate:
    """Prove every collision-free COM path start->goal crosses the gate.

    COM exclusion disks have radius r_k + r_in (r_in = COM-inscribed radius of
    the runtime footprint, NO planning margin): a collision-free hull keeps its
    COM outside them. Two disks overlapping strictly cover the segment between
    their centres. A closed walk through the overlap graph plus the gate edge
    whose segment crossings with [start, goal] are odd separates start from
    goal (even-odd parity is constant off the curve), and every point of the
    walk except the gate opening is covered. Hence every path crosses the
    opening. Edges touching or collinear with [start, goal] are excluded
    (conservative). The resulting walk is re-verified by an independent
    even-odd ray test. Failure is a rejection, never a proof of passability.
    """
    s, g = _point(start), _point(goal)
    centers, radii = _obstacles(centers, radii)
    r_in = _positive(inscribed_radius_m, "inscribed radius")
    i, j = (int(gate[0]), int(gate[1]))
    n = len(radii)
    if not (0 <= i < n and 0 <= j < n and i != j):
        return GateCertificate(False, (i, j), (), math.nan, math.nan, ("invalid_gate_indices",))
    excl = radii + r_in
    reasons = []
    for name, p in (("start", s), ("goal", g)):
        if np.any(np.linalg.norm(centers - p, axis=1) <= excl + EPS):
            reasons.append(name + "_inside_exclusion")
    d_gate = float(np.linalg.norm(centers[i] - centers[j]))
    physical_gap = d_gate - radii[i] - radii[j]
    com_opening = d_gate - excl[i] - excl[j]
    if com_opening <= 0.0:
        reasons.append("gate_closed_at_runtime_footprint")
    if reasons:
        return GateCertificate(False, (i, j), (), physical_gap, com_opening, tuple(reasons))
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    adj = dist < (excl[:, None] + excl[None, :] - overlap_tolerance_m)
    np.fill_diagonal(adj, False)
    adj[i, j] = adj[j, i] = False
    parity = {}
    for a in range(n):
        for b in np.nonzero(adj[a])[0]:
            b = int(b)
            if b < a:
                continue
            crossing = _proper_cross(centers[a], centers[b], s, g)
            if crossing < 0:
                adj[a, b] = adj[b, a] = False
            else:
                parity[(a, b)] = parity[(b, a)] = crossing
    gate_cross = _proper_cross(centers[i], centers[j], s, g)
    if gate_cross < 0:
        return GateCertificate(False, (i, j), (), physical_gap, com_opening,
                               ("gate_edge_degenerate_with_start_goal_segment",))
    target = (j, 1 ^ gate_cross)
    start_state = (i, 0)
    parent = {start_state: None}
    queue = [start_state]
    head = 0
    while head < len(queue):
        node, par = queue[head]
        head += 1
        if (node, par) == target:
            break
        for b in np.nonzero(adj[node])[0]:
            b = int(b)
            state = (b, par ^ parity[(node, b)])
            if state not in parent:
                parent[state] = (node, par)
                queue.append(state)
    if target not in parent:
        return GateCertificate(False, (i, j), (), physical_gap, com_opening,
                               ("no_separating_closed_barrier",))
    walk = [target]
    while walk[-1] != start_state:
        walk.append(parent[walk[-1]])
    walk = [w[0] for w in reversed(walk)]
    poly = [centers[k] for k in walk]
    if _even_odd_inside(s, poly) == _even_odd_inside(g, poly):
        return GateCertificate(False, (i, j), tuple(walk), physical_gap, com_opening,
                               ("parity_verification_failed",))
    return GateCertificate(True, (i, j), tuple(walk), float(physical_gap), float(com_opening), ())


@dataclass(frozen=True)
class Barrier:
    """Explicit ordered closed chain; gate_after indexes the gate edge
    (obstacle_indices[gate_after] -> obstacle_indices[gate_after + 1])."""

    obstacle_indices: tuple
    gate_after: int


def validate_barrier(barrier: Barrier, start, goal, centers, radii, runtime_inscribed_radius_m):
    """Check an explicit chain: all non-gate edges overlap at the runtime
    inscribed radius, the gate is open, and the chain separates the endpoints
    (even-odd; convexity NOT required). Returns rejection reasons."""
    centers, radii = _obstacles(centers, radii)
    ids = tuple(int(k) for k in barrier.obstacle_indices)
    if (len(ids) < 3 or len(set(ids)) != len(ids) or any(k < 0 or k >= len(radii) for k in ids)
            or not 0 <= int(barrier.gate_after) < len(ids)):
        return ("invalid_barrier_indices",)
    r_in = _positive(runtime_inscribed_radius_m, "runtime radius")
    vertices = centers[list(ids)]
    rr = radii[list(ids)]
    surface = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    free = surface - rr - np.roll(rr, -1) - 2.0 * r_in
    reasons = []
    if np.any(np.delete(free, int(barrier.gate_after)) >= -EPS):
        reasons.append("unintended_runtime_gap")
    if free[int(barrier.gate_after)] <= 0.0:
        reasons.append("declared_gate_closed")
    if _even_odd_inside(_point(start), vertices) == _even_odd_inside(_point(goal), vertices):
        reasons.append("barrier_does_not_separate_endpoints")
    return tuple(reasons)


# --------------------------------------------------------------------------
# counts, ring construction, fortress validation, generic admission
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    reasons: tuple
    realized_count: int
    dose_count: int | None = None


def admit_counts(requested, realized, *, mode="exact", recorded_realized_count=None) -> AdmissionResult:
    """exact: realized must equal requested. recorded: the realized count is
    itself the (declared, stamped) dose; never a silent fallback."""
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0
           for x in (requested, realized)):
        raise ValueError("counts must be nonnegative integers")
    if mode == "exact":
        ok = int(requested) == int(realized)
    elif mode == "recorded":
        ok = (recorded_realized_count is not None and not isinstance(recorded_realized_count, bool)
              and int(recorded_realized_count) == int(realized))
    else:
        raise ValueError("count mode must be exact or recorded")
    return AdmissionResult(ok, () if ok else ("obstacle_count_mismatch",), int(realized),
                           int(realized) if ok else None)


@dataclass(frozen=True)
class RingGeometry:
    centers_m: np.ndarray
    radii_m: np.ndarray
    gate: tuple
    gate_midpoint_m: np.ndarray
    gate_normal_m: np.ndarray
    tier: PhysicalTier
    neighbour_overlap_m: float


def construct_sealed_ring(vessel: VesselGeometry, tier: PhysicalTier, *, ring_radius_m,
                          obstacle_radius_m, gate_bearing_rad=0.0, seal_overlap_m=None,
                          center_xy_m=(0.0, 0.0)) -> RingGeometry:
    """One declared opening of physical gap g; every other neighbour pair
    overlaps at the RUNTIME inscribed radius (no planning margin) by at least
    seal_overlap_m (default 0.05 s). Obstacle count is a deterministic,
    recorded structural quantity."""
    if tier != physical_tier(tier.level, vessel):
        raise ValueError("tier must be resolved for this vessel before construction")
    R = _positive(ring_radius_m, "ring radius")
    r = _positive(obstacle_radius_m, "obstacle radius")
    if not math.isfinite(gate_bearing_rad):
        raise ValueError("invalid gate bearing")
    overlap = 0.05 * vessel.length_scale if seal_overlap_m is None else _positive(seal_overlap_m, "overlap")
    c0 = _point(center_xy_m)
    gate_chord = 2.0 * r + tier.physical_surface_gap_m
    if gate_chord >= 2.0 * R:
        raise ValueError("gate too wide for ring")
    gate_angle = 2.0 * math.asin(gate_chord / (2.0 * R))
    seal_chord = 2.0 * (r + vessel.inscribed_radius_m) - overlap
    if seal_chord <= 0.0:
        raise ValueError("seal overlap too large")
    max_step = 2.0 * math.asin(min(1.0, seal_chord / (2.0 * R)))
    segments = int(math.ceil((2.0 * math.pi - gate_angle) / max_step))
    angles = gate_bearing_rad + np.linspace(gate_angle / 2.0, 2.0 * math.pi - gate_angle / 2.0,
                                            segments + 1)
    centers = c0 + R * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    radii = np.full(len(angles), r)
    step = float(np.linalg.norm(centers[1] - centers[0]))
    realized_overlap = 2.0 * (r + vessel.inscribed_radius_m) - step
    mid = c0 + (R * math.cos(gate_angle / 2.0)) * np.array((math.cos(gate_bearing_rad),
                                                            math.sin(gate_bearing_rad)))
    normal = np.array((math.cos(gate_bearing_rad), math.sin(gate_bearing_rad)))
    for array in (centers, radii, mid, normal):
        array.setflags(write=False)
    return RingGeometry(centers, radii, (len(radii) - 1, 0), mid, normal, tier, float(realized_overlap))


def validate_band_fortress(start, goal, centers, radii, *, vessel: VesselGeometry, skipped_plugs,
                           requested_count=None, recorded_realized_count=None) -> AdmissionResult:
    """Legacy brick-wall band fortress, FINAL geometry (C1 finding 34).

    Rejects: missing/nonzero skipped-plug count (plug failure is admission
    failure, never silently tolerated); a straight start-goal traverse whose
    exact oriented-hull sweep (physical footprint, zero margin) is clear (a
    direct shortcut); count mismatch; non-finite data. Builder metadata such
    as direct_blocked is NOT trusted; the continuous query is recomputed.
    """
    try:
        s, g = _point(start), _point(goal)
        centers, radii = _obstacles(centers, radii)
    except ValueError:
        return AdmissionResult(False, ("nonfinite_or_invalid_geometry",), 0)
    reasons = []
    if skipped_plugs is None:
        reasons.append("plug_status_missing")
    elif isinstance(skipped_plugs, bool) or not isinstance(skipped_plugs, (int, np.integer)) \
            or int(skipped_plugs) != 0:
        reasons.append("plug_placement_failed")
    d = g - s
    if float(np.hypot(*d)) > EPS:
        heading = math.atan2(d[1], d[0])
        if translation_sweep_clearance(vessel.footprint, s, g, heading, centers, radii) > 0.0:
            reasons.append("direct_shortcut")
    count = admit_counts(len(radii) if requested_count is None else int(requested_count), len(radii),
                         mode="recorded" if requested_count is None else "exact",
                         recorded_realized_count=recorded_realized_count)
    reasons.extend(count.reasons)
    return AdmissionResult(not reasons, tuple(dict.fromkeys(reasons)), len(radii), count.dose_count)


@dataclass(frozen=True)
class LayoutAdmission:
    admitted: bool
    reasons: tuple
    realized_count: int
    dose_count: int | None
    witness: RouteWitness | None
    certificates: tuple
    min_pair_surface_gap_m: float
    start_clearance_m: float
    goal_clearance_m: float
    direct_physical_clearance_m: float


def admit_final_layout(*, vessel: VesselGeometry, tier: PhysicalTier, start, goal, centers, radii,
                       bounds_m, via_points=(), requested_count, count_mode="exact",
                       recorded_realized_count=None, compulsory_gates=(), forbid_direct_shortcut=False,
                       pair_gap_indices=None, min_pair_surface_gap_m=None, start_heading_rad=None,
                       goal_heading_rad=None, endpoint_clearance_m=None, cell_m=None,
                       keep_points=None) -> LayoutAdmission:
    """Admission on the FINAL geometry, after every insertion/removal.

    Requires finite arrays and positive radii, the exact (or recorded) count,
    full-footprint start clearance at the start heading and goal clearance
    (full footprint at goal_heading_rad if the task fixes the terminal
    heading, else the heading-free circumscribed disk), each by more than
    endpoint_clearance_m (default: the planning margin),
    optional minimum surface gap among pair_gap_indices, each declared
    compulsory gate certified at the runtime inscribed radius, optional
    absence of a physically clear straight shortcut, and a verified
    curvature-bounded witness with the tier planning margin.
    """
    empty = LayoutAdmission(False, ("nonfinite_or_invalid_geometry",), 0, None, None, (),
                            math.nan, math.nan, math.nan, math.nan)
    try:
        s, g = _point(start), _point(goal)
        centers, radii = _obstacles(centers, radii)
        vias = [_point(p) for p in via_points]
    except ValueError:
        return empty
    reasons = []
    margin = tier.planning_margin_m
    endpoint = margin if endpoint_clearance_m is None else _nonnegative(endpoint_clearance_m, "endpoint")
    count = admit_counts(int(requested_count), len(radii), mode=count_mode,
                         recorded_realized_count=recorded_realized_count)
    reasons.extend(count.reasons)
    if len(radii):
        heading0 = 0.0 if start_heading_rad is None else float(start_heading_rad)
        start_clear = pose_clearance(vessel.footprint, (s[0], s[1], heading0), centers, radii)
        if goal_heading_rad is None:
            goal_clear = segment_clearance(g, g, centers, radii, vessel.circumscribed_radius_m)
        else:
            goal_clear = pose_clearance(vessel.footprint, (g[0], g[1], float(goal_heading_rad)),
                                        centers, radii)
    else:
        start_clear = goal_clear = math.inf
    if start_clear <= endpoint:
        reasons.append("start_pose_not_clear")
    if goal_clear <= endpoint:
        reasons.append("goal_region_not_clear")
    pair_gap = math.inf
    if pair_gap_indices is not None and len(pair_gap_indices) >= 2:
        idx = np.asarray(sorted(set(int(k) for k in pair_gap_indices)))
        c, r = centers[idx], radii[idx]
        dd = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1) - r[:, None] - r[None, :]
        dd[np.tril_indices(len(idx))] = math.inf
        pair_gap = float(np.min(dd))
        if min_pair_surface_gap_m is not None and pair_gap + 1.0e-9 < float(min_pair_surface_gap_m):
            reasons.append("pair_surface_gap_below_tier")
    certs = []
    for gate in compulsory_gates:
        cert = compulsory_gate_certificate(s, g, centers, radii, gate, vessel.inscribed_radius_m)
        certs.append(cert)
        if not cert.valid:
            reasons.append("gate_%d_%d_not_certified:%s" % (gate[0], gate[1], ",".join(cert.reasons)))
    direct = math.inf
    d = g - s
    if len(radii) and float(np.hypot(*d)) > EPS:
        direct = translation_sweep_clearance(vessel.footprint, s, g, math.atan2(d[1], d[0]),
                                             centers, radii)
    if forbid_direct_shortcut and direct > 0.0:
        reasons.append("direct_shortcut")
    witness = None
    if not reasons:
        witness = route_witness([s] + vias + [g], centers, radii, vessel, margin_m=margin,
                                bounds_m=bounds_m, cell_m=cell_m, start_heading_rad=start_heading_rad,
                                keep_points=keep_points)
        if not witness.feasible:
            reasons.extend("witness:" + r for r in witness.reasons)
    return LayoutAdmission(not reasons, tuple(dict.fromkeys(reasons)), len(radii), count.dose_count,
                           witness, tuple(certs), pair_gap, float(start_clear), float(goal_clear),
                           float(direct))


# G7 item A (2026-10-01): platform-stable identities. A digest that decides
# whether a pool, layout or assignment is valid must not depend on last-ulp
# floating-point noise that differs between platforms or library versions
# (Python's sum(), libm sin/cos/exp, numpy SIMD or BLAS paths). Every float
# that enters geometry_digest is therefore quantised first: q = round(x / Q)
# with the declared quantum Q = 1e-6 in the value's SI unit (1 micrometre,
# 1 microradian, 1 microsecond, 1 milligram), round half to even; -0.0 maps
# to 0. Arrays hash as little-endian int64 quanta, scalars as JSON with each
# float written "f:<q>". The canonicalisation version enters every digest, so
# a digest of another rule can never compare equal by accident.
IDENTITY_CANON_VERSION = "usvbench.identity.fixed_quantum_1e-6/1"
IDENTITY_QUANTUM = 1.0e-6
_IDENTITY_SCALE = 1.0e6
_IDENTITY_INT_LIMIT = 2 ** 62


def canonical_quanta(value) -> int:
    """Declared quantisation of one finite float (IDENTITY_QUANTUM units, half to even)."""
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("identity values must be finite")
    q = round(x * _IDENTITY_SCALE)
    if abs(q) >= _IDENTITY_INT_LIMIT:
        raise ValueError("identity value out of the quantised range")
    return int(q)


def canonical_identity_value(value):
    """JSON-ready copy of ``value`` with every float replaced by its "f:<quanta>" text."""
    if isinstance(value, np.ndarray):
        return [canonical_identity_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return "f:%d" % canonical_quanta(value)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): canonical_identity_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_identity_value(v) for v in value]
    return "f:%d" % canonical_quanta(value)        # numbers of other types (json default=float before G7)


def _canonical_array_bytes(value) -> bytes:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("identity arrays must be finite")
    q = np.rint(arr * _IDENTITY_SCALE)
    if q.size and float(np.max(np.abs(q))) >= _IDENTITY_INT_LIMIT:
        raise ValueError("identity array out of the quantised range")
    q = np.ascontiguousarray(q.astype("<i8"))             # integers: -0.0 and 0.0 both give 0
    return repr(tuple(int(n) for n in arr.shape)).encode("ascii") + b"\0" + q.tobytes()


def _has_text(value) -> bool:
    """True for nested sequences holding already-canonical text (they hash as JSON, not as arrays)."""
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_text(v) for v in value)
    return False


def geometry_digest(**fields) -> str:
    """SHA-256 of canonical geometry: arrays as quantised int64 bytes, scalars as canonical JSON.

    Every float is quantised with IDENTITY_QUANTUM (see the note above); the
    canonicalisation version is the first item hashed.
    """
    h = hashlib.sha256()
    h.update(b"canon\0" + IDENTITY_CANON_VERSION.encode("ascii") + b"\0")
    for key in sorted(fields):
        value = fields[key]
        h.update(key.encode("ascii") + b"\0")
        if isinstance(value, np.ndarray) or (isinstance(value, (list, tuple)) and value
                                              and isinstance(value[0], (list, tuple, np.ndarray))
                                              and not _has_text(value)):
            h.update(_canonical_array_bytes(value) + b"\0")
        else:
            h.update(json.dumps(canonical_identity_value(value), sort_keys=True).encode("ascii") + b"\0")
    return h.hexdigest()
