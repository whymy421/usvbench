"""Pure planar footprint geometry and exact continuous linear-motion TTC.

TTC is exact for a fixed-yaw rectangle (the vessel footprint, possibly offset
from the COM reference point) translating at constant velocity against a
constant-velocity disk. Contact includes tangency and initial overlap. It is
not a rotating-ship collision solver: swept_interval_safe overbounds rotation.
"""

import math

from .model import Footprint


def footprint_of(obj):
    """Accept a Footprint or anything carrying one (e.g. VesselEnvelope)."""
    if isinstance(obj, Footprint):
        return obj
    footprint = getattr(obj, "footprint", None)
    if not isinstance(footprint, Footprint):
        raise ValueError("expected a Footprint or an object with .footprint")
    return footprint


def to_body(x, y, pose):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    dx, dy = x - pose.x, y - pose.y
    return c * dx + s * dy, -s * dx + c * dy


def to_world(bx, by, pose):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return pose.x + c * bx - s * by, pose.y + s * bx + c * by


def point_clearance(x, y, pose, footprint):
    """Signed point-to-footprint distance; negative inside, zero on the boundary."""
    footprint = footprint_of(footprint)
    bx, by = to_body(x, y, pose)
    cx, cy = footprint.center
    hx, hy = footprint.half_extents
    qx, qy = abs(bx - cx) - hx, abs(by - cy) - hy
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0)


def circle_clearance(pose, footprint, circle):
    return point_clearance(circle.x, circle.y, pose, footprint) - circle.radius


def world_aabb_half_extents(pose, footprint):
    """World-axis bounds of the rotated footprint relative to the reference point."""
    footprint = footprint_of(footprint)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    xs = [c * bx - s * by for bx, by in footprint.corners]
    ys = [s * bx + c * by for bx, by in footprint.corners]
    return min(xs), max(xs), min(ys), max(ys)


def _ray_box_ttc(px, py, vx, vy, hx, hy):
    lower, upper = 0.0, math.inf
    for p, v, half in ((px, vx, hx), (py, vy, hy)):
        if v == 0.0:
            if abs(p) > half:
                return math.inf
        else:
            a, b = (-half - p) / v, (half - p) / v
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
            if lower > upper:
                return math.inf
    return lower if upper >= 0.0 else math.inf


def _ray_circle_ttc(px, py, vx, vy, cx, cy, radius):
    dx, dy = px - cx, py - cy
    c = dx * dx + dy * dy - radius * radius
    if c <= 0.0:
        return 0.0
    a = vx * vx + vy * vy
    b = 2.0 * (dx * vx + dy * vy)
    if a == 0.0 or b >= 0.0:
        return math.inf
    discriminant = b * b - 4.0 * a * c
    # Preserve grazing contact against roundoff in an otherwise analytic test.
    tolerance = 1e-14 * max(b * b, abs(4.0 * a * c), 1.0)
    if discriminant < -tolerance:
        return math.inf
    root = math.sqrt(max(0.0, discriminant))
    q = -0.5 * (b - root)  # numerically stable smaller root
    return c / q if q > 0.0 else math.inf


def rounded_box_entry(px, py, vx, vy, hx, hy, radius):
    """First time a point at p moving with v enters box(h) Minkowski disk(radius)."""
    candidates = [_ray_box_ttc(px, py, vx, vy, hx, hy + radius),
                  _ray_box_ttc(px, py, vx, vy, hx + radius, hy)]
    for cx in (-hx, hx):
        for cy in (-hy, hy):
            candidates.append(_ray_circle_ttc(px, py, vx, vy, cx, cy, radius))
    return min(candidates)


def relative_body_motion(vessel, intruder, at_s):
    """Intruder position/velocity relative to the footprint centre, body axes.

    Uses the intruder velocity AT at_s (zero while it rests), extrapolated.
    """
    x, y = intruder.position(at_s)
    bx, by = to_body(x, y, vessel)
    ivx, ivy = intruder.velocity(at_s) if hasattr(intruder, "velocity") else (intruder.vx, intruder.vy)
    c, s = math.cos(vessel.yaw), math.sin(vessel.yaw)
    vx = c * ivx + s * ivy - vessel.speed
    vy = -s * ivx + c * ivy
    return bx, by, vx, vy


def ttc_linear(vessel, footprint, intruder, at_s=0.0, horizon_s=math.inf):
    """First contact seconds after at_s, or infinity when no contact occurs.

    vessel is the ownship state AT at_s, moving with constant speed along its
    yaw; the intruder keeps its velocity at at_s. Exact when both motions are
    constant-velocity until contact (generation guarantees this for the dose
    window); otherwise it is the usual constant-velocity extrapolation.
    Sensor visibility is deliberately irrelevant to physical TTC.
    """
    if not math.isfinite(at_s) or at_s < 0.0:
        raise ValueError("at_s must be finite and nonnegative")
    if math.isnan(horizon_s) or horizon_s < 0.0:
        raise ValueError("horizon_s must be nonnegative")
    footprint = footprint_of(footprint)
    bx, by, vx, vy = relative_body_motion(vessel, intruder, at_s)
    cx, cy = footprint.center
    hx, hy = footprint.half_extents
    result = rounded_box_entry(bx - cx, by - cy, vx, vy, hx, hy, intruder.radius_m)
    return result if result <= horizon_s else math.inf


def swept_interval_safe(start, end, footprint, obstacles_start, obstacles_end,
                        clearance_m=0.0, angular_travel_rad=None,
                        center_deviation_m=0.0):
    """Conservative whole-interval collision guard; False includes uncertainty.

    Endpoint centres are joined by line segments. angular_travel_rad bounds the
    full yaw excursion from `start`; the default assumes shortest-arc
    interpolation. center_deviation_m must bound the sum of ownship and
    obstacle deviations from those segments for curved/reflecting paths.
    """
    footprint = footprint_of(footprint)
    if len(obstacles_start) != len(obstacles_end):
        raise ValueError("obstacle identities/counts must match across interval")
    if angular_travel_rad is None:
        angular_travel_rad = abs(math.remainder(end.yaw - start.yaw, 2.0 * math.pi))
    for name, value in (("clearance_m", clearance_m),
                        ("angular_travel_rad", angular_travel_rad),
                        ("center_deviation_m", center_deviation_m)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(name + " must be finite and nonnegative")
    rotation_bound = 2.0 * footprint.reach_m * math.sin(min(angular_travel_rad, math.pi) / 2.0)
    inflate = rotation_bound + center_deviation_m + clearance_m
    c, s = math.cos(start.yaw), math.sin(start.yaw)
    cx, cy = footprint.center
    hx, hy = footprint.half_extents
    own_dx, own_dy = end.x - start.x, end.y - start.y
    for before, after in zip(obstacles_start, obstacles_end):
        px, py = to_body(before.x, before.y, start)
        dx, dy = after.x - before.x - own_dx, after.y - before.y - own_dy
        vx, vy = c * dx + s * dy, -s * dx + c * dy
        ttc = rounded_box_entry(px - cx, py - cy, vx, vy, hx, hy,
                                max(before.radius, after.radius) + inflate)
        if ttc <= 1.0:
            return False
    return True


def segment_circle_min_distance(x0, y0, x1, y1, circle):
    """Minimum centre distance from a point moving on a segment to a circle centre."""
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, ((circle.x - x0) * dx + (circle.y - y0) * dy) / length2))
    return math.hypot(x0 + t * dx - circle.x, y0 + t * dy - circle.y)
