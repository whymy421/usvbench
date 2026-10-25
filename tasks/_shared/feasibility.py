# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feasibility pooling: turn raw ranges into "how far can THIS hull travel?".

Raw per-ray ranges encode a passable gap as two nearby threats, and leave the
policy to infer from 36 numbers whether its own beam fits between them. Meyer
et al. (IEEE Access 2020, gym-auv) instead pool each sector into the maximum
distance the vessel could actually travel in that direction, so a gap wider
than the beam reads as open water. They also warn against the obvious
alternative -- per-sector min-pooling -- because it "yields an overly
restrictive observation vector, effectively telling the agent that a majority
of the travel directions are blocked".

Algorithm, per sector: walk candidate ranges from nearest to farthest. At a
candidate distance d, the rays that see beyond d + width/2 are the ones still
"open"; the arc they subtend at distance d has physical width d * dtheta per
ray. If no run of consecutive open rays spans more than the vessel width, the
hull cannot get past d, so d is the feasible travel distance.
"""

from __future__ import annotations

import torch


def feasibility_pool(
    ranges_m: torch.Tensor,
    *,
    n_sectors: int,
    vessel_width_m: float,
    ray_spacing_rad: float,
    width_multiplier: float = 1.0,
) -> torch.Tensor:
    """Pool ``(N, R)`` ray ranges into ``(N, n_sectors)`` feasible distances.

    ``width_multiplier`` inflates the required opening (gym-auv defaults to a
    generous 5.0); 1.0 asks only that the bare hull fits.
    """
    if ranges_m.dim() != 2:
        raise ValueError("ranges_m must be (num_envs, num_rays)")
    num_envs, num_rays = ranges_m.shape
    if num_rays % n_sectors:
        raise ValueError("num_rays must divide evenly into n_sectors")
    per_sector = num_rays // n_sectors
    width = float(vessel_width_m) * float(width_multiplier)

    # (N, S, K) -- rays grouped by sector
    grouped = ranges_m.view(num_envs, n_sectors, per_sector)
    # Candidate distances are the ray readings themselves, ascending.
    candidates, _ = torch.sort(grouped, dim=-1)

    # For every candidate c and ray r: is r open at c?  (N, S, K_c, K_r)
    open_mask = grouped.unsqueeze(-2) > (candidates.unsqueeze(-1) + width / 2.0)

    # Longest run of consecutive open rays, measured in metres of arc at the
    # candidate distance. Done with a cumulative-sum trick over the ray axis so
    # the whole thing stays vectorised.
    arc_per_ray = candidates.unsqueeze(-1) * float(ray_spacing_rad)
    contrib = open_mask.float() * arc_per_ray
    # run-length: reset the running sum wherever a ray is closed
    runs = torch.zeros_like(contrib)
    running = torch.zeros(contrib.shape[:-1], device=contrib.device)
    for k in range(per_sector):
        running = torch.where(
            open_mask[..., k], running + contrib[..., k], torch.zeros_like(running)
        )
        runs[..., k] = running
    widest_opening = runs.max(dim=-1).values  # (N, S, K_c)

    blocked = widest_opening <= width  # no run wide enough -> hull cannot pass
    # First blocking candidate gives the feasible distance; if none blocks, the
    # sector is open out to its farthest reading.
    big = torch.full_like(candidates, float("inf"))
    blocking_distance = torch.where(blocked, candidates, big).min(dim=-1).values
    farthest = grouped.max(dim=-1).values
    return torch.where(torch.isfinite(blocking_distance), blocking_distance, farthest)


__all__ = ["feasibility_pool"]
