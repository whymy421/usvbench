"""CPU reference sensing/contact operations over the SAME moving object state.

Owner rule (2026-09-30): hazards reach controllers only through each task's
existing sensors, never through a privileged channel. Therefore:
* ray_ranges includes an object only when it is sensor-visible;
* collides uses PHYSICAL presence (including before reveal) and truth pose;
* ray_detections turns the ray returns (static and dynamic alike, unlabeled)
  into point detections, the only hazard input permitted for planners such as
  MPPI or the classical planner, which otherwise read privileged static maps.
"""

import math

from .geometry import point_clearance
from .model import Circle, check_number


def ray_ranges(origin, world_angles, max_range_m, samples):
    """Analytic ray-disk ranges; only visible samples are sensed."""
    check_number("max_range_m", max_range_m, positive=True)
    out = []
    for angle in world_angles:
        ux, uy = math.cos(angle), math.sin(angle)
        closest = max_range_m
        for item in samples:
            if not item.visible:
                continue
            x, y = item.x - origin[0], item.y - origin[1]
            projection = x * ux + y * uy
            c = x * x + y * y - item.radius_m ** 2
            if c <= 0:
                closest = 0.0
                continue
            discriminant = projection ** 2 - c
            if discriminant >= 0:
                hit = projection - math.sqrt(discriminant)
                if 0 <= hit <= closest:
                    closest = hit
        out.append(closest)
    return tuple(out)


def combine_ranges(static_ranges, dynamic_ranges):
    """Elementwise minimum of physical ranges BEFORE any perception degradation."""
    if len(static_ranges) != len(dynamic_ranges):
        raise ValueError("ray blocks must have the same width")
    return tuple(min(a, b) for a, b in zip(static_ranges, dynamic_ranges))


def collides(pose, footprint, samples):
    """Instant truth contact, including physically present but unrevealed objects."""
    return any(point_clearance(s.x, s.y, pose, footprint) <= s.radius_m
               for s in samples if s.physical)


def ray_detections(origin, heading_rad, relative_angles, ranges, max_range_m, point_radius_m=0.0):
    """Unlabeled point detections from a ray block (hits strictly inside max range)."""
    check_number("max_range_m", max_range_m, positive=True)
    check_number("point_radius_m", point_radius_m, minimum=0.0)
    if len(relative_angles) != len(ranges):
        raise ValueError("one range per ray angle")
    out = []
    for angle, distance in zip(relative_angles, ranges):
        if distance < max_range_m:
            world = heading_rad + angle
            out.append(Circle(origin[0] + distance * math.cos(world),
                              origin[1] + distance * math.sin(world), point_radius_m))
    return tuple(out)
