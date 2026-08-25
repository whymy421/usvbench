# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Irregular wave field (sea state) as a shared, task-agnostic modifier.

Merged implementation:

* **Spectrum from Yutong's** ``jonswap_wave.py`` (repo ``main``): a proper
  JONSWAP spectrum -- Phillips constant normalised to H_s, Pierson-Moskowitz
  shape, peak-enhancement factor gamma with the standard sigma = 0.07/0.09
  split -- with component amplitudes drawn from the spectrum as
  ``a_i = sqrt(2 S(f_i) df)``, 30 components over 0.04-0.5 Hz, and per-episode
  randomised (H_s, T_p, gamma, mean heading, +/-30 deg directional spread).
* **Force interface and acceptance tests from this codebase**: the hull sees
  the same quadratic relative-velocity law as the certified current, so a
  current and a sea state simply add their water velocities, and the module
  ships with physics tests (dispersion, H_s recovery, zero mean, determinism,
  bounded force).

Constitution compliance: the field is closed-form in ``(position, time)`` with
phases fixed at reset, so rewards stay memoryless and episodes stay replayable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


GRAVITY = 9.81


@dataclass
class SeaStateCfg:
    """One sea state band. Per-episode draws are uniform inside each range."""

    enable: bool = False
    # Significant wave height H_s (m). The default band is DELIBERATELY mild;
    # a sea state must pass the admission protocol (measurable behaviour change
    # vs calm) before it may be used as a benchmark modifier.
    hs_range: tuple = (0.3, 1.0)
    tp_range: tuple = (4.0, 7.0)          # peak period (s)
    gamma_range: tuple = (1.0, 5.0)       # JONSWAP peak enhancement
    direction_spread_rad: float = math.pi / 6.0   # +/- 30 deg, per component
    n_components: int = 30
    f_min: float = 0.04
    f_max: float = 0.5
    # Hull coupling. Orbital drag reuses the current drag coefficient so a wave
    # field and a current push the hull on the same physical scale.
    orbital_drag_coeff: float = 8.0
    heave_force_scale: float = 100.0
    slope_moment_scale: float = 200.0


class SeaState:
    """Vectorised JONSWAP wave field, one independent realisation per env."""

    def __init__(self, cfg: SeaStateCfg, num_envs: int, device: torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        n = int(cfg.n_components)
        if n < 1:
            raise ValueError("n_components must be >= 1")

        self.df = (cfg.f_max - cfg.f_min) / n
        self.freqs = torch.linspace(
            cfg.f_min + self.df / 2.0, cfg.f_max - self.df / 2.0, n, device=device
        )
        self.omega = 2.0 * math.pi * self.freqs
        self.k = self.omega**2 / GRAVITY  # deep-water dispersion

        self.hs = torch.zeros(num_envs, device=device)
        self.tp = torch.zeros(num_envs, device=device)
        self.gamma = torch.zeros(num_envs, device=device)
        self.mean_direction = torch.zeros(num_envs, device=device)
        self.amplitude = torch.zeros((num_envs, n), device=device)
        self.phase = torch.zeros((num_envs, n), device=device)
        self.direction = torch.zeros((num_envs, n), device=device)
        self.resample(torch.arange(num_envs, device=device))

    def _jonswap(
        self, hs: torch.Tensor, tp: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """S(f) = alpha f^-5 exp(-1.25 (fp/f)^4) gamma^r, per env."""
        fp = (1.0 / tp).unsqueeze(1)
        hs_e = hs.unsqueeze(1)
        gamma_e = gamma.unsqueeze(1)
        f = self.freqs.unsqueeze(0)
        alpha = 5.0 / 16.0 * hs_e**2 * fp**4
        pm = alpha * f.pow(-5) * torch.exp(-1.25 * (fp / f).pow(4))
        sigma = torch.where(f <= fp, 0.07, 0.09)
        r = torch.exp(-0.5 * ((f - fp) / (sigma * fp)).pow(2))
        return torch.clamp(pm * gamma_e.pow(r), min=0.0)

    def resample(self, env_ids: torch.Tensor, scenario=None) -> None:
        """Draw a fresh sea state (H_s, T_p, gamma, heading, phases) per env.

        ``scenario`` is an optional ``.._shared.scenario_rng.ScenarioRNG``.
        With ``None`` -- the default, and what ``__init__`` below passes --
        every draw comes from the GLOBAL torch RNG exactly as it always did.
        That is the historical evaluation defect: skrl's ``Runner.__init__``
        reseeds the global RNG to the constant in the agent YAML after the env
        has been built, so two ``--eval-seed`` values drew the SAME wave field.
        With a ScenarioRNG the same six draws, in the same order and with the
        same arithmetic, come off the per-(env, episode) ``wave`` stream.

        ``__init__`` deliberately passes no scenario: at construction time no
        episode has begun, and every value written here is overwritten by the
        first ``_reset_idx``, which resets all envs.
        """
        # Same try/except idiom test_sea_state.py uses at its own import: this
        # module is loaded both as a package member (by the envs) and as a bare
        # top-level module (by the standalone CPU tests).
        try:
            from .scenario_draws import GROUP_WAVE, unit_uniform
        except ImportError:  # direct execution
            from scenario_draws import GROUP_WAVE, unit_uniform

        n = self.freqs.numel()

        def unit(size: tuple = ()) -> torch.Tensor:
            """U[0, 1) of exactly the shape the replaced torch.rand produced."""
            return unit_uniform(scenario, GROUP_WAVE, env_ids, self.device, size)

        def uniform(rng: tuple) -> torch.Tensor:
            return unit() * (rng[1] - rng[0]) + rng[0]

        self.hs[env_ids] = uniform(self.cfg.hs_range)
        self.tp[env_ids] = uniform(self.cfg.tp_range)
        self.gamma[env_ids] = uniform(self.cfg.gamma_range)
        self.mean_direction[env_ids] = unit() * 2.0 * math.pi
        self.phase[env_ids] = unit((n,)) * (2.0 * math.pi)
        spread = (unit((n,)) - 0.5) * (
            2.0 * self.cfg.direction_spread_rad
        )
        self.direction[env_ids] = self.mean_direction[env_ids].unsqueeze(1) + spread
        spectrum = self._jonswap(
            self.hs[env_ids], self.tp[env_ids], self.gamma[env_ids]
        )
        amplitude = torch.sqrt(2.0 * spectrum * self.df)
        # Energy normalisation. The Phillips constant alpha = 5/16 H_s^2 fp^4
        # is exact only for gamma = 1; with peak enhancement the discretised
        # spectrum carries ~20% too much energy, so the realised sea would be
        # taller than requested. Rescale each realisation so the identity
        # H_s = 4 sqrt(m0), m0 = sum(a_i^2 / 2), holds exactly.
        m0 = (amplitude**2 / 2.0).sum(dim=1, keepdim=True)
        target_m0 = (self.hs[env_ids] / 4.0).pow(2).unsqueeze(1)
        scale = torch.sqrt(target_m0 / m0.clamp_min(1.0e-12))
        self.amplitude[env_ids] = amplitude * scale

    def _phase_at(self, xy: torch.Tensor, t: float) -> torch.Tensor:
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
        return (self.amplitude * torch.cos(self._phase_at(xy, t))).sum(dim=-1)

    def orbital_velocity(self, xy: torch.Tensor, t: float) -> torch.Tensor:
        """Surface horizontal orbital velocity (m/s), per environment."""
        theta = self._phase_at(xy, t)
        speed = self.amplitude * self.omega.view(1, -1) * torch.cos(theta)
        vx = (speed * torch.cos(self.direction)).sum(dim=-1)
        vy = (speed * torch.sin(self.direction)).sum(dim=-1)
        return torch.stack((vx, vy), dim=-1)

    def surface_slope(self, xy: torch.Tensor, t: float) -> torch.Tensor:
        """Surface gradient (d eta/dx, d eta/dy); drives the trim moment."""
        theta = self._phase_at(xy, t)
        common = -self.amplitude * self.k.view(1, -1) * torch.sin(theta)
        sx = (common * torch.cos(self.direction)).sum(dim=-1)
        sy = (common * torch.sin(self.direction)).sum(dim=-1)
        return torch.stack((sx, sy), dim=-1)

    def forces(
        self, xy: torch.Tensor, hull_velocity_xy: torch.Tensor, t: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame (force_xyz, torque_xyz) on the hull."""
        v_rel = self.orbital_velocity(xy, t) - hull_velocity_xy
        v_mag = torch.norm(v_rel, dim=-1, keepdim=True)
        drag_xy = self.cfg.orbital_drag_coeff * v_rel * v_mag

        eta = self.elevation(xy, t)
        slope = self.surface_slope(xy, t)

        force = torch.zeros((xy.shape[0], 3), device=xy.device)
        force[:, :2] = drag_xy
        force[:, 2] = self.cfg.heave_force_scale * eta

        torque = torch.zeros((xy.shape[0], 3), device=xy.device)
        torque[:, 0] = self.cfg.slope_moment_scale * slope[:, 1]
        torque[:, 1] = -self.cfg.slope_moment_scale * slope[:, 0]
        return force, torque

    def observation(self, forward_2d: torch.Tensor) -> torch.Tensor:
        """Sea-state channels for tasks that expose it: (cos, sin, H_s norm).

        Encoding mirrors the goal-direction convention already used across the
        benchmark, so a policy reads "where the sea comes from" the same way it
        reads "where the goal is".
        """
        wave_x = torch.cos(self.mean_direction)
        wave_y = torch.sin(self.mean_direction)
        dot = forward_2d[:, 0] * wave_x + forward_2d[:, 1] * wave_y
        cross = forward_2d[:, 0] * wave_y - forward_2d[:, 1] * wave_x
        hs_norm = self.hs / max(self.cfg.hs_range[1], 1.0e-6)
        return torch.stack((dot, cross, hs_norm), dim=-1)


__all__ = ["GRAVITY", "SeaState", "SeaStateCfg"]
