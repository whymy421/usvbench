"""Vectorised journal docking kernels: hull-polygon clearance, swept bound, 36-ray fan (torch, CPU/GPU).

* ``polygon_circle_clearance``: exact signed separation between the vessel's
  COM-relative collision polygon (E5 VesselRecord.footprint, D5 registry,
  strictly convex, CCW) posed at (x, y, yaw) and circles (the berth walls):
  signed distance of every circle centre to the polygon minus the radius, min
  over circles. Planar yaw pose, as E5's admission (journal_geometry_v2.
  pose_clearance) and the contract's containment use it.
* ``swept_lower_bound``: clearance of every pose on the linear XY / shortest
  yaw interpolation between two samples is >= (c_a + c_b - M) / 2 with
  M = |dxy| + r_core |dyaw| (the signed distance is 1-Lipschitz in rigid
  displacement; r_core = circumscribed polygon radius about the COM). Used
  once per physics substep, so contact between samples cannot be missed
  (journal_geometry_v2.swept_clearance uses the same Lipschitz argument).
* ``ray_circle_ranges``: nearest forward ray/circle intersection per ray
  (0 inside a circle, max range when nothing is hit) -- the analytic ray fan
  of hazard_nav (hazard_geometry.ray_circle_ranges) and of E4
  dyn/sensors.ray_ranges; D7 objects enter only while ``visible``
  (point-return proxy of E4 visibility.py decides visibility), walls always.
  The wall block and the D7 block are combined by elementwise minimum BEFORE
  perception degradation (E4 sensors.combine_ranges; D7 spec row D5).
"""
from __future__ import annotations

import math

import torch


def yaw_from_quat(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def polygon_circle_clearance(xy, yaw, hull, centers, radii, active=None):
    """[N] min over circles of (signed distance centre->posed polygon) - radius; +inf if none."""
    if centers.shape[-2] == 0:
        return torch.full(xy.shape[:-1], math.inf, dtype=xy.dtype, device=xy.device)
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    d = centers - xy.unsqueeze(-2)
    px = c * d[..., 0] + s * d[..., 1]                 # circle centres in the hull frame
    py = -s * d[..., 0] + c * d[..., 1]
    a = hull
    b = torch.roll(hull, -1, dims=0)
    e = b - a                                          # [V, 2]
    rx = px.unsqueeze(-1) - a[:, 0]                    # [N, W, V]
    ry = py.unsqueeze(-1) - a[:, 1]
    length_sq = (e * e).sum(-1)
    t = ((rx * e[:, 0] + ry * e[:, 1]) / length_sq).clamp(0.0, 1.0)
    qx, qy = rx - t * e[:, 0], ry - t * e[:, 1]
    dist = torch.sqrt(qx * qx + qy * qy).amin(-1)
    inside = ((e[:, 0] * ry - e[:, 1] * rx) >= 0.0).all(-1)
    signed = torch.where(inside, -dist, dist) - radii
    if active is not None:
        signed = torch.where(active, signed, torch.full_like(signed, math.inf))
    return signed.amin(-1)


def swept_lower_bound(clear_a, clear_b, xy_a, yaw_a, xy_b, yaw_b, core_radius):
    """Lipschitz lower bound of the clearance over the interpolated motion a -> b."""
    motion = torch.linalg.vector_norm(xy_b - xy_a, dim=-1) + core_radius * wrap(yaw_b - yaw_a).abs()
    both = torch.isfinite(clear_a) & torch.isfinite(clear_b)
    bound = 0.5 * (clear_a + clear_b - motion)
    return torch.where(both, torch.minimum(bound, torch.minimum(clear_a, clear_b)),
                       torch.minimum(clear_a, clear_b))


def ray_circle_ranges(origins, world_angles, centers, radii, max_range_m, active=None):
    """[N, R] nearest forward hit of each ray on the circles (0 inside, max range if none)."""
    n, r = world_angles.shape
    if centers.shape[-2] == 0:
        return torch.full((n, r), float(max_range_m), dtype=origins.dtype, device=origins.device)
    ux, uy = torch.cos(world_angles), torch.sin(world_angles)                  # [N, R]
    to_x = (centers[..., 0] - origins[:, 0:1]).unsqueeze(1)                    # [N, 1, K]
    to_y = (centers[..., 1] - origins[:, 1:2]).unsqueeze(1)
    projection = ux.unsqueeze(-1) * to_x + uy.unsqueeze(-1) * to_y             # [N, R, K]
    center_sq = to_x * to_x + to_y * to_y
    radius_sq = (radii * radii).unsqueeze(1)
    discriminant = projection * projection - (center_sq - radius_sq)
    root = torch.sqrt(discriminant.clamp(min=0.0))
    inside = center_sq <= radius_sq
    entry = torch.where(inside, torch.zeros_like(projection), projection - root)
    valid = inside | ((discriminant >= 0.0) & (entry >= 0.0))
    if active is not None:
        valid = valid & active.unsqueeze(1)
    hits = torch.where(valid, entry, torch.full_like(entry, float(max_range_m)))
    return hits.amin(-1).clamp(max=float(max_range_m))


def fan_angles(yaw, ray_count):
    """[N, R] world angles of the journal fan: ray k at yaw + 2 pi k / R (bow = ray 0, CCW)."""
    k = torch.arange(int(ray_count), dtype=yaw.dtype, device=yaw.device) * (2.0 * math.pi / int(ray_count))
    return yaw.unsqueeze(-1) + k


__all__ = ["yaw_from_quat", "wrap", "polygon_circle_clearance", "swept_lower_bound", "ray_circle_ranges",
           "fan_angles"]
