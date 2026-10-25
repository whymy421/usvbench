"""G23c (REV-G23b F8): the E4 point-return ray proxy for D7 intruder slots, shared by the ray-circle families.

conditions/dynamic/visibility.py defines D7 detectability with a point-return proxy (the segment from the observer
to the nearest surface point on the centre ray) and requires the runtime ray block to use the same proxy. Path
following and station keeping already do (journal_path_following.point_return_rays, journal_station.point_return_rays);
hazard_nav, harbour mission, path hazard and docking intersected every ray with the full intruder disk, so the
stamped first detection and the controller's first ray return disagreed (REV-G23b probe7: a ray hit the disk at the
stamped detection in only 25-39 % of D1 episodes). With this module those four families return a D7 intruder slot
only through the proxy: for every VISIBLE intruder the fan ray nearest to the bearing of its centre (ray k at
yaw + 2 pi k / R, bow = ray 0, counter-clockwise; a tie goes to the higher index) returns max(0, |c - p| - r)
(clamped to the maximum range); every other ray ignores it. Static obstacles, moving-basin native slots and the
appearing slot keep the families' ray-circle intersection unchanged. Same arithmetic as
journal_path_following.point_return_rays (float64).
"""
from __future__ import annotations

import math

import torch

F64 = torch.float64


def point_return_rays(com_xy, yaw, centers, radii, visible, ray_count, max_range_m):
    """[N, ray_count] float64 ranges from the point-return proxy of the VISIBLE slots (others: max range).

    com_xy [N, 2], yaw [N], centers [N, S, 2] (same frame as com_xy), radii [N, S], visible [N, S] bool."""
    com = com_xy.to(F64)
    n = com.shape[0]
    count = int(ray_count)
    out = torch.full((n, count), float(max_range_m), dtype=F64, device=com.device)
    if centers.shape[1] == 0:
        return out
    delta = centers.to(F64) - com.unsqueeze(1)
    distance = torch.sqrt((delta * delta).sum(-1))
    edge = (distance - radii.to(F64)).clamp(min=0.0)
    bearing = torch.atan2(delta[..., 1], delta[..., 0]) - yaw.to(F64).unsqueeze(1)
    step = 2.0 * math.pi / count
    index = torch.remainder(torch.floor(torch.remainder(bearing, 2.0 * math.pi) / step + 0.5), count).long()
    ranges = torch.where(visible.to(torch.bool), edge.clamp(max=float(max_range_m)),
                         torch.full_like(edge, float(max_range_m)))
    for slot in range(centers.shape[1]):
        out.scatter_reduce_(1, index[:, slot:slot + 1], ranges[:, slot:slot + 1], reduce="amin")
    return out


def slot_mask(width, slots, device):
    """[width] bool mask of the D7 intruder slots (point-return slots)."""
    mask = torch.zeros(int(width), dtype=torch.bool, device=device)
    for slot in slots:
        mask[int(slot)] = True
    return mask


def merge(base_rays, com_xy, yaw, centers, radii, visible, ray_count, max_range_m):
    """min(base_rays, point-return rays of the visible D7 intruder slots), in base_rays' dtype."""
    if not bool(visible.any()):
        return base_rays
    points = point_return_rays(com_xy, yaw, centers, radii, visible, ray_count, max_range_m)
    return torch.minimum(base_rays.to(F64), points).to(base_rays.dtype)


__all__ = ["point_return_rays", "slot_mask", "merge"]
