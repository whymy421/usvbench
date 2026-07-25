# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-frame planar kinematics channels, shared by every task.

Rationale (from the HazardNav v11 finding): a scalar speed magnitude cannot
distinguish "moving forward", "sliding sideways", and "still rotating" -- three
states that demand different control. Exposing body-frame surge, sway and yaw
rate makes the hull's inertia observable, which is what lets a policy brake and
line up instead of orbiting its target.

The encoding is fixed here so every task that adopts it agrees channel for
channel:

    [surge_norm, sway_norm, yaw_rate_norm]

with surge/sway divided by ``SPEED_SCALE_MPS`` (the docking scale already used
by the frozen observation superset) and yaw rate by ``yaw_rate_scale_rad_s``,
all clamped to [-1, 1].
"""

from __future__ import annotations

import torch

try:
    from .obs_superset import SPEED_SCALE_MPS
except ImportError:  # direct execution of the module or its test
    from obs_superset import SPEED_SCALE_MPS


def body_planar_kinematics(
    forward_2d: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    yaw_rate_w: torch.Tensor,
    *,
    speed_scale_mps: float = SPEED_SCALE_MPS,
    yaw_rate_scale_rad_s: float = 1.0,
) -> torch.Tensor:
    """Return the (N, 3) [surge, sway, yaw_rate] normalised channel block.

    ``forward_2d`` must be the unit bow direction in world XY, matching the
    convention each task already uses for its goal-direction channels.
    """
    if speed_scale_mps <= 0.0 or yaw_rate_scale_rad_s <= 0.0:
        raise ValueError("kinematic scales must be positive")

    velocity_2d = linear_velocity_w[:, :2]
    left = torch.stack((-forward_2d[:, 1], forward_2d[:, 0]), dim=-1)
    surge = torch.sum(velocity_2d * forward_2d, dim=-1, keepdim=True)
    sway = torch.sum(velocity_2d * left, dim=-1, keepdim=True)
    yaw_rate = yaw_rate_w.reshape(-1, 1)

    return torch.hstack(
        (
            torch.clamp(surge / speed_scale_mps, -1.0, 1.0),
            torch.clamp(sway / speed_scale_mps, -1.0, 1.0),
            torch.clamp(yaw_rate / yaw_rate_scale_rad_s, -1.0, 1.0),
        )
    )


__all__ = ["body_planar_kinematics"]
