# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Within-episode linear current-speed ramp for StationKeeping variants.

Deliberately Isaac-free (torch only), like ``.._shared.sea_state`` and
``..path_hazard.path_hazard_geometry``, so the modulation law can be
unit-tested on a CPU-only machine with no simulator install.
"""

from __future__ import annotations

import torch


def ramped_current_vec(
    base_current_vec: torch.Tensor,
    episode_step: torch.Tensor,
    max_episode_steps: int,
    ramp_start_mps: float,
    ramp_end_mps: float,
) -> torch.Tensor:
    """Rescale per-episode current vectors onto a linear within-episode ramp.

    ``base_current_vec`` (``(..., 2)``) carries the direction the task sampled
    once per episode; only its direction is used, so the fixed-per-episode
    direction survives even a zero-speed ramp endpoint. The returned vector
    has magnitude ``ramp_start_mps`` at ``episode_step == 0`` and
    ``ramp_end_mps`` at ``episode_step == max_episode_steps``, linear in
    between and clamped to the endpoint speeds outside that window. A zero
    base vector has no direction and maps to a zero output.
    """
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if ramp_start_mps < 0.0 or ramp_end_mps < 0.0:
        raise ValueError("ramp speeds must be non-negative")
    direction = base_current_vec / torch.linalg.vector_norm(
        base_current_vec, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    fraction = (
        episode_step.to(base_current_vec.dtype) / float(max_episode_steps)
    ).clamp(0.0, 1.0)
    speed = ramp_start_mps + (ramp_end_mps - ramp_start_mps) * fraction
    return direction * speed.unsqueeze(-1)


__all__ = ["ramped_current_vec"]
