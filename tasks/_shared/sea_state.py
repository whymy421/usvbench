# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Irregular wave field (sea state) as a shared, task-agnostic modifier.

Design constraints inherited from the benchmark constitution:

* **Deterministic given the episode seed.** Phases are drawn once per reset
  from the environment RNG, so a scenario replays byte for byte.
* **Closed-form in time.** The field is a finite sum of Airy components; the
  force at step ``t`` depends only on ``(position, t)``, never on history, so
  the memoryless-reward rule and fixed-horizon scoring are unaffected.
* **Same interface as the certified current.** Waves produce a world-frame
  force/torque added exactly where ``_compute_current_forces`` is added; a
  task enables them with a config flag and nothing else changes.

Model. A JONSWAP-shaped set of ``n_components`` regular waves with directional
spread around a mean heading. For deep water, omega^2 = g k. Each component
contributes an orbital-velocity field; the hull feels

    F = c_d * (v_water - v_hull) * |v_water - v_hull|        (quadratic drag)
    M = k_slope * slope_perpendicular                        (pitch/roll moment)

plus a heave force from the local surface elevation. The wave *elevation* is
also exposed so tasks that want a "sea state" observation can add it without
re-deriving the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


GRAVITY = 9.81


@dataclass
class SeaStateCfg:
    """One sea state. ``significant_height_m`` = H_s, the usual scale."""

    enable: bool = False
    significant_height_m: float = 0.25
    peak_period_s: float = 3.0
    mean_direction_rad: float = 0.0
    direction_spread_rad: float = 0.35
    n_components: int = 8
    # Hull coupling. Orbital drag reuses the current drag coefficient so a
    # wave field and a current push the hull on the same physical scale.
    orbital_drag_coeff: float = 8.0
    heave_force_scale: float = 60.0
    slope_moment_scale: float = 40.0


class SeaState:
    """Vectorised irregular wave field, one independent realisation per env."""

    def __init__(self, cfg: SeaStateCfg, num_envs: int, device: torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        n = int(cfg.n_components)
        if n < 1:
            raise ValueError("n_components must be >= 1")

        # JONSWAP-ish discretisation: components spread around the peak.
        peak_omega = 2.0 * math.pi / float(cfg.peak_period_s)
        ratios = torch.linspace(0.6, 1.6, n, device=device)
        self.omega = peak_omega * ratios  # (n,)
        self.k = self.omega**2 / GRAVITY  # deep-water dispersion

        # Equal-energy split of H_s across components. With m0 = sum(a_i^2/2)
        # = n a^2 / 2 and H_s = 4 sqrt(m0), the amplitude that reproduces the
        # requested significant height is a = H_s / (2 sqrt(2n)).
        self.amplitude = torch.full(
            (n,),
            float(cfg.significant_height_m) / (2.0 * math.sqrt(2.0 * n)),
            device=device,
        )

        self.phase = torch.zeros((num_envs, n), device=device)
        self.direction = torch.zeros((num_envs, n), device=device)
        self.resample(torch.arange(num_envs, device=device))

    def resample(self, env_ids: torch.Tensor) -> None:
        """Draw fresh phases and directional spread for the given envs."""
        n = self.omega.numel()
        count = len(env_ids)
        self.phase[env_ids] = torch.rand((count, n), device=self.device) * (
            2.0 * math.pi
        )
        spread = (torch.rand((count, n), device=self.device) - 0.5) * (
            2.0 * self.cfg.direction_spread_rad
        )
        self.direction[env_ids] = self.cfg.mean_direction_rad + spread

    def _phase_at(self, xy: torch.Tensor, t: float) -> torch.Tensor:
        """Per-component phase argument at planar positions ``xy`` and time t."""
        kx = self.k.view(1, -1) * torch.cos(self.direction)
        ky = self.k.view(1, -1) * torch.sin(self.direction)
        return (
            kx * xy[:, 0:1]
            + ky * xy[:, 1:2]
            - self.omega.view(1, -1) * t
            + self.phase
        )

    def elevation(self, xy: torch.Tensor, t: float) -> torch.Tensor:
        """Free-surface elevation (m) at each environment's hull position."""
        theta = self._phase_at(xy, t)
        return (self.amplitude.view(1, -1) * torch.cos(theta)).sum(dim=-1)

    def orbital_velocity(self, xy: torch.Tensor, t: float) -> torch.Tensor:
        """Horizontal orbital velocity (m/s) at the surface, per environment."""
        theta = self._phase_at(xy, t)
        # Surface horizontal orbital velocity of component i: a_i * omega_i,
        # aligned with the component's propagation direction.
        speed = self.amplitude.view(1, -1) * self.omega.view(1, -1) * torch.cos(theta)
        vx = (speed * torch.cos(self.direction)).sum(dim=-1)
        vy = (speed * torch.sin(self.direction)).sum(dim=-1)
        return torch.stack((vx, vy), dim=-1)

    def surface_slope(self, xy: torch.Tensor, t: float) -> torch.Tensor:
        """Surface gradient (d(eta)/dx, d(eta)/dy), drives the trim moment."""
        theta = self._phase_at(xy, t)
        common = -self.amplitude.view(1, -1) * self.k.view(1, -1) * torch.sin(theta)
        sx = (common * torch.cos(self.direction)).sum(dim=-1)
        sy = (common * torch.sin(self.direction)).sum(dim=-1)
        return torch.stack((sx, sy), dim=-1)

    def forces(
        self, xy: torch.Tensor, hull_velocity_xy: torch.Tensor, t: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame (force_xyz, torque_xyz) exerted on the hull.

        Drop-in companion to the certified current force: same quadratic
        relative-velocity law, so enabling both simply adds their water
        velocities.
        """
        v_water = self.orbital_velocity(xy, t)
        v_rel = v_water - hull_velocity_xy
        v_mag = torch.norm(v_rel, dim=-1, keepdim=True)
        drag_xy = self.cfg.orbital_drag_coeff * v_rel * v_mag

        eta = self.elevation(xy, t)
        slope = self.surface_slope(xy, t)

        force = torch.zeros((xy.shape[0], 3), device=xy.device)
        force[:, :2] = drag_xy
        force[:, 2] = self.cfg.heave_force_scale * eta

        torque = torch.zeros((xy.shape[0], 3), device=xy.device)
        # A surface tilted along +x pitches the hull about -y, and vice versa.
        torque[:, 0] = self.cfg.slope_moment_scale * slope[:, 1]
        torque[:, 1] = -self.cfg.slope_moment_scale * slope[:, 0]
        return force, torque


__all__ = ["GRAVITY", "SeaState", "SeaStateCfg"]
