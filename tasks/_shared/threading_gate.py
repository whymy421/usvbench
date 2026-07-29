# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot threading bonus: a half-sine arc across a gap, paid once.

Design (owner's, 2026-07-26): getting through without touching is the goal;
the middle is best, but off-centre-and-clean is fine too, so the payout should
bulge in the middle and fall to zero exactly where the hull would touch --
"half a sine arc". Formalised as

    e = (d_R - d_L) / 2                     signed offset from the gap centreline
    h = (d_L + d_R) / 2 - half_beam         how far the hull may slide before contact
    u = clamp(e / h, -1, 1)
    R = A * cos(pi * u / 2)                 == A * sin(pi * (u+1) / 2)

Dividing by ``h`` is what makes ONE formula cover every gap width: the arc is
automatically steep in a tight gap and gentle in a wide one (at a 5-beam gap,
0.6 m off-centre costs 13%; at a 2-beam gap, 0.3 m off-centre costs 50%). It
also self-defends against hugging a single pillar -- a lopsided pair drives
u toward 1 and the payout toward zero.

Payment is gated on a completed passage (entered, kept moving toward the goal,
came out the far side, never touched) and latched to once per episode, so the
bonus cannot be farmed by loitering, oscillating, or re-crossing.
"""

from __future__ import annotations

import math

import torch


def gap_geometry(
    ranges_m: torch.Tensor,
    *,
    ray_spacing_rad: float,
    half_beam_m: float,
    left_sector: tuple = (50.0, 130.0),
    right_sector: tuple = (-130.0, -50.0),
    front_sector: tuple = (-30.0, 30.0),
    rank: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (u, h, front_clearance) from a full ray sweep.

    ``rank`` picks the k-th smallest reading in each side sector instead of the
    minimum, so one stray ray cannot invent a gap. Ray 0 points along the bow
    and indices advance counter-clockwise, matching the sensor contract.
    """
    num_rays = ranges_m.shape[-1]
    idx = torch.arange(num_rays, device=ranges_m.device, dtype=torch.float32)
    angles = torch.rad2deg(idx * ray_spacing_rad)
    angles = torch.where(angles > 180.0, angles - 360.0, angles)

    def sector_rank(bounds, k):
        mask = (angles >= bounds[0]) & (angles <= bounds[1])
        picked = ranges_m[:, mask]
        k = min(k, picked.shape[-1] - 1)
        return torch.sort(picked, dim=-1).values[:, k]

    d_left = sector_rank(left_sector, rank)
    d_right = sector_rank(right_sector, rank)
    front = sector_rank(front_sector, 0)

    offset = (d_right - d_left) / 2.0
    half_width = (d_left + d_right) / 2.0 - half_beam_m
    u = torch.clamp(offset / half_width.clamp_min(1.0e-3), -1.0, 1.0)
    # A gap the hull cannot fit through at all has h <= 0; force the payout off.
    u = torch.where(half_width > 0.0, u, torch.ones_like(u))
    return u, half_width, front


def arc_payout(u: torch.Tensor, amplitude: float) -> torch.Tensor:
    """The half-sine arc: maximum on the centreline, zero at contact."""
    return amplitude * torch.cos(math.pi * u / 2.0)


class ThreadingLatch:
    """Tracks passage candidates and pays the arc once per episode."""

    def __init__(self, num_envs: int, device: torch.device, *,
                 amplitude: float = 5.0,
                 min_surge: float = 0.25,
                 min_goal_closing: float = 0.25,
                 min_goal_dot: float = 0.50,
                 max_sway: float = 0.35,
                 max_yaw_rate: float = 0.35,
                 front_clear_m: float = 1.50,
                 side_min_m: float = 0.45,
                 side_max_m: float = 2.25,
                 balance_m: float = 0.50,
                 hold_steps: int = 15):
        self.num_envs = num_envs
        self.device = device
        self.amplitude = amplitude
        self.min_surge = min_surge
        self.min_goal_closing = min_goal_closing
        self.min_goal_dot = min_goal_dot
        self.max_sway = max_sway
        self.max_yaw_rate = max_yaw_rate
        self.front_clear_m = front_clear_m
        self.side_min_m = side_min_m
        self.side_max_m = side_max_m
        self.balance_m = balance_m
        self.hold_steps = hold_steps

        self.paid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._in_gap = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._best_u = torch.ones(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.paid[env_ids] = False
        self._in_gap[env_ids] = 0
        self._best_u[env_ids] = 1.0

    def step(
        self,
        ranges_m: torch.Tensor,
        *,
        ray_spacing_rad: float,
        half_beam_m: float,
        surge_norm: torch.Tensor,
        sway_norm: torch.Tensor,
        yaw_rate_norm: torch.Tensor,
        goal_dot: torch.Tensor,
        goal_cross: torch.Tensor,
        contact: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one control step; return the reward to add (mostly zeros)."""
        u, half_width, front = gap_geometry(
            ranges_m, ray_spacing_rad=ray_spacing_rad, half_beam_m=half_beam_m
        )
        d_span = half_width + half_beam_m  # mean side reading

        # Geometry: a real, hull-sized portal on both sides, clear ahead.
        in_portal = (
            (d_span >= self.side_min_m)
            & (d_span <= self.side_max_m)
            & ((u.abs() * half_width * 2.0) <= self.balance_m)
            & (front >= self.front_clear_m)
            & (half_width > 0.0)
        )
        # Motion: actually driving through, toward the goal, not crabbing.
        closing = surge_norm * goal_dot + sway_norm * goal_cross
        moving = (
            (surge_norm >= self.min_surge)
            & (closing >= self.min_goal_closing)
            & (goal_dot >= self.min_goal_dot)
            & (sway_norm.abs() <= self.max_sway)
            & (yaw_rate_norm.abs() <= self.max_yaw_rate)
        )
        active = in_portal & moving & ~contact & ~self.paid

        # Track the tightest point of the passage while it is in progress.
        self._best_u = torch.where(
            active, torch.minimum(self._best_u, u.abs()), self._best_u
        )
        self._in_gap = torch.where(
            active, self._in_gap + 1, torch.zeros_like(self._in_gap)
        )
        self._best_u = torch.where(
            self._in_gap == 0, torch.ones_like(self._best_u), self._best_u
        )

        # A passage completes on the step the run reaches hold_steps: the hull
        # is then demonstrably inside a portal AND has been driving through it
        # for a quarter second, so paying here keeps the bonus a pure function
        # of the current step while still requiring a sustained transit.
        completed = active & (self._in_gap == self.hold_steps)
        reward = torch.where(
            completed, arc_payout(self._best_u, self.amplitude),
            torch.zeros_like(u)
        )
        self.paid |= completed
        return reward


__all__ = ["ThreadingLatch", "arc_payout", "gap_geometry"]
