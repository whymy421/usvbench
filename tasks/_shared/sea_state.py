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
  ``a_i = sqrt(2 S(f_i) df)``, 104 cell-centred components over 0.04-1.60 Hz
  (see BAND below; 30 @ 0.04-0.5 Hz before 2026-08-26), and per-episode
  randomised (H_s, T_p, gamma, mean heading, +/-30 deg directional spread).
* **Force interface and acceptance tests from this codebase**: the hull sees
  the same quadratic relative-velocity law as the certified current, so a
  current and a sea state simply add their water velocities, and the module
  ships with physics tests (dispersion, H_s recovery, zero mean, determinism,
  bounded force).

Constitution compliance: the field is closed-form in ``(position, time)`` with
phases fixed at reset, so rewards stay memoryless and episodes stay replayable.

BAND (frozen 2026-08-26)
------------------------
The component band is 104 cell-centred components over [0.04, 1.60] Hz
(df = 0.0150 Hz, the same resolution as the historical 30 @ [0.04, 0.50]).
The old 0.50 Hz ceiling truncated short-Tp spectra catastrophically: the
analytic-spectrum capture at the certified worst corner (Tp = 1.5 s,
gamma = 1) was 1.9%; on the new band it is 96.30%.  The Hs renormalisation
in ``resample`` hid this completely (see ``analytic_spectrum_capture``),
which is why the acceptance gate in ``scripts/wave_acceptance.py`` checks
capture against the ANALYTIC spectrum, never against the renormalised
amplitudes.

STEEPNESS ADMISSION (DNV-RP-C205)
---------------------------------
``validate_steepness_admission`` refuses, at construction time, any
``(hs_range, tp_range)`` box whose steepest corner exceeds the deep-water
breaking limit Sp = Hs / (g Tp^2 / (2 pi)) = 1/7.  Draws are uniform inside
the box, so validating the (max Hs, min Tp) corner bounds every draw.  The
certified StationKeep-BlueBoat-Wave box (Hs 0.30-0.60 m, Tp 1.5-3.0 s)
contains Hs = 0.60 m at Tp = 1.5 s, Sp ~ 1/5.9 -- beyond the limit.  It is
GRANDFATHERED by value (see ``GRANDFATHERED_STEEPNESS_BOXES``) with a
runtime warning, never silently: the id is certified history and stays
frozen, but every NEW box must pass.

DUAL CALM REFERENCE (zero-amplitude force floor)
------------------------------------------------
``forces`` applies the quadratic hull drag ``orbital_drag_coeff * v_rel *
|v_rel|`` even at zero wave amplitude: with Hs = 0 the orbital velocity
vanishes but a moving hull still sees ``orbital_drag_coeff * speed^2`` of
drag (0.08 N at 0.1 m/s, 8 N at 1 m/s, 72 N at 3 m/s -- quantified by
``zero_amplitude_drag_force``).  ``enable=True`` with Hs -> 0 is therefore
NOT the same environment as ``enable=False``.  This is deliberate physics
and is NOT changed by the 2026-08-26 fix package; any dose ladder over a
Wave variant must carry BOTH a true-calm reference (enable=False) and an
intercept rung (enable=True, Hs -> 0) and must never present the two as
interchangeable.

SLOPE-CHANNEL DISCLOSURE (band extension, 2026-08-26)
-----------------------------------------------------
The roll/pitch moment forcing integrand grows like f^4 * S(f) and does NOT
converge as f_max grows.  Extending the band from 0.50 Hz to 1.60 Hz
roughly DOUBLES the surface-slope RMS at unchanged (Hs, Tp) labels
(measured 0.088 -> 0.178 at Hs = 0.45 m, Tp = 2.25 s, gamma = 3.3).  This
is a physics decision, not a bug: pre-band-change and post-band-change
roll/pitch moment channels are NOT comparable, and
``scripts/wave_acceptance.py`` prints this disclosure in every report.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch


GRAVITY = 9.81

# Deep-water breaking limit on significant steepness (DNV-RP-C205): a box
# whose steepest admissible draw exceeds Hs / (g Tp^2 / (2 pi)) = 1/7 is
# physically breaking and is refused at construction time.
STEEPNESS_LIMIT = 1.0 / 7.0

# ---------------------------------------------------------------------------
# GRANDFATHERED STEEPNESS BOXES -- read this before you touch it.
# ---------------------------------------------------------------------------
# The certified Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1 band
# (tasks/station_keeping/station_keeping_env_cfg.py -- Hs 0.30-0.60 m,
# Tp 1.5-3.0 s) predates the steepness admission rule and contains the corner
# Hs = 0.60 m / Tp = 1.5 s with Sp ~ 1/5.9 > 1/7.  That id is certified
# history: its numbers are frozen, so its box is admitted BY VALUE here --
# LOUDLY, with a RuntimeWarning at every construction, never silently.
# NOTHING may ever be added to this tuple without a new certification
# decision; every new box must pass the 1/7 limit instead.
GRANDFATHERED_STEEPNESS_BOXES = (
    ((0.30, 0.60), (1.5, 3.0)),
)


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
    # Band frozen 2026-08-26: 104 cell-centred components over [0.04, 1.60]
    # Hz, df = 0.0150 Hz (the historical 30 @ [0.04, 0.50] resolution).
    # Worst-corner (Tp = 1.5 s, gamma = 1) analytic capture is 96.30%; the
    # old 0.50 Hz ceiling captured 1.9% there.  Read the module docstring
    # (BAND and SLOPE-CHANNEL DISCLOSURE) before changing either number.
    n_components: int = 104
    f_min: float = 0.04
    f_max: float = 1.60
    # Hull coupling. Orbital drag reuses the current drag coefficient so a wave
    # field and a current push the hull on the same physical scale.
    orbital_drag_coeff: float = 8.0
    heave_force_scale: float = 100.0
    slope_moment_scale: float = 200.0
    # FROZEN sea-intensity reference (m) for the hs_norm observation channel:
    # hs_norm = hs / hs_norm_ref.  0.60 m is the certified evaluation ceiling
    # (the StationKeep-BlueBoat-Wave hs_range[1]), so the certified id reads
    # bit-identically what it always read (proven by
    # test_wave_fix_package.py).  Dividing by the per-config hs_range[1] --
    # the pre-2026-08-26 behaviour -- made a pinned rung hs_range=(x, x)
    # read exactly 1.0 at EVERY rung, erasing the intensity signal the
    # channel exists to carry.  Do not tie this back to hs_range.
    hs_norm_ref: float = 0.60


def significant_steepness(hs: float, tp: float) -> float:
    """Significant steepness Sp = Hs / (g Tp^2 / (2 pi)) (DNV-RP-C205)."""
    return float(hs) / (GRAVITY * float(tp) ** 2 / (2.0 * math.pi))


def validate_steepness_admission(cfg: SeaStateCfg) -> None:
    """Refuse any (Hs, Tp) box whose steepest corner exceeds Sp = 1/7.

    Draws are uniform inside ``hs_range x tp_range``, so the steepest
    possible draw is the (max Hs, min Tp) corner; admitting the corner
    admits every draw.  Called by ``SeaState.__init__`` -- construction time
    is cfg validation time for this module.  The certified station-keeping
    box is grandfathered BY VALUE (``GRANDFATHERED_STEEPNESS_BOXES``): it
    keeps constructing, but never silently -- a RuntimeWarning fires on
    every construction.
    """
    hs_hi = float(max(cfg.hs_range))
    tp_lo = float(min(cfg.tp_range))
    if hs_hi <= 0.0:
        return  # a calm box has zero steepness whatever Tp says
    if tp_lo <= 0.0:
        raise ValueError(f"sea state tp_range {cfg.tp_range} must be positive")
    sp = significant_steepness(hs_hi, tp_lo)
    if sp <= STEEPNESS_LIMIT:
        return
    box = (
        tuple(float(v) for v in cfg.hs_range),
        tuple(float(v) for v in cfg.tp_range),
    )
    if box in GRANDFATHERED_STEEPNESS_BOXES:
        warnings.warn(
            "sea state box hs_range={}, tp_range={} has significant "
            "steepness Sp = 1/{:.1f}, beyond the DNV-RP-C205 deep-water "
            "breaking limit 1/7; it is admitted ONLY because it is the "
            "grandfathered certified StationKeep-BlueBoat-Wave band -- "
            "every new config must stay inside the limit".format(
                box[0], box[1], 1.0 / sp
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return
    raise ValueError(
        "sea state box hs_range={}, tp_range={} is REFUSED: worst-corner "
        "significant steepness Sp = Hs/(g*Tp^2/(2*pi)) = 1/{:.1f} exceeds "
        "the DNV-RP-C205 deep-water breaking limit 1/7 (Hs={:.3f} m at "
        "Tp={:.3f} s). Raise Tp, lower Hs, or split the box.".format(
            cfg.hs_range, cfg.tp_range, 1.0 / sp, hs_hi, tp_lo
        )
    )


def zero_amplitude_drag_force(
    hull_speed_mps: float, drag_coeff: float = 8.0
) -> float:
    """Hull drag magnitude (N) that ``forces`` applies at ZERO wave amplitude.

    With Hs = 0 the orbital velocity vanishes, so ``v_rel = -hull_velocity``
    and the quadratic drag reduces to ``drag_coeff * speed^2`` opposing the
    hull's motion: 0.08 N at 0.1 m/s (station keeping), 8 N at 1 m/s, 72 N
    at 3 m/s.  ``enable=True`` with Hs -> 0 is therefore NOT calm water.
    This helper exists so ladder designs can CITE the intercept instead of
    assuming it away; see DUAL CALM REFERENCE in the module docstring.  The
    physics itself is intentionally unchanged.
    """
    return float(drag_coeff) * float(hull_speed_mps) ** 2


def jonswap_spectrum(
    f: torch.Tensor, hs: float, tp: float, gamma: float
) -> torch.Tensor:
    """Scalar-parameter JONSWAP S(f) on a 1-D frequency grid ``f`` (Hz).

    Mirrors ``SeaState._jonswap`` line for line (Phillips alpha normalised to
    Hs, P-M shape, peak enhancement with the 0.07/0.09 sigma split) for a
    single (hs, tp, gamma); the coupling is pinned by
    ``test_wave_fix_package.py``.  Keep the two in sync.
    """
    fp = 1.0 / float(tp)
    alpha = 5.0 / 16.0 * float(hs) ** 2 * fp**4
    pm = alpha * f.pow(-5) * torch.exp(-1.25 * (fp / f).pow(4))
    sigma = torch.where(f <= fp, 0.07, 0.09)
    r = torch.exp(-0.5 * ((f - fp) / (sigma * fp)).pow(2))
    return torch.clamp(pm * float(gamma) ** r, min=0.0)


def analytic_spectrum_capture(
    n_components: int,
    f_min: float,
    f_max: float,
    hs: float,
    tp: float,
    gamma: float,
    quad_points: int = 400_000,
    quad_f_max_hz: float = 50.0,
) -> float:
    """Fraction of the ANALYTIC JONSWAP variance the discretised band holds.

    ``captured = sum_i S_analytic(f_i) * df`` over the cell-centred grid,
    divided by the analytic ``m0 = integral_0^inf S(f) df`` (high-resolution
    trapezoidal quadrature in float64; for gamma = 1 this matches the closed
    form m0 = Hs^2 / 16 to ~1e-12 relative).

    THIS IS THE ONLY CHECK THAT CAN SEE TRUNCATION.  The Hs renormalisation
    in ``SeaState.resample`` rescales every realisation so H_s = 4 sqrt(m0)
    holds EXACTLY, so any energy the band fails to cover is silently pumped
    back into the covered components: at Tp = 2.25 s / gamma = 3.3 on the
    old 0.50 Hz band the renorm scale factor was 1.024 -- within 3% of
    unity -- while the band was missing 37% of the analytic variance.  An
    amplitude-based check can therefore never detect truncation; always
    measure against the analytic spectrum, PRE-normalisation, as
    ``scripts/wave_acceptance.py`` does (FAIL below 95%).
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    df = (float(f_max) - float(f_min)) / int(n_components)
    grid = torch.linspace(
        float(f_min) + df / 2.0,
        float(f_max) - df / 2.0,
        int(n_components),
        dtype=torch.float64,
    )
    band = float(jonswap_spectrum(grid, hs, tp, gamma).sum()) * df
    quad = torch.linspace(
        1.0e-6, float(quad_f_max_hz), int(quad_points), dtype=torch.float64
    )
    m0 = float(torch.trapezoid(jonswap_spectrum(quad, hs, tp, gamma), quad))
    return band / m0


class SeaState:
    """Vectorised JONSWAP wave field, one independent realisation per env."""

    def __init__(self, cfg: SeaStateCfg, num_envs: int, device: torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        n = int(cfg.n_components)
        if n < 1:
            raise ValueError("n_components must be >= 1")
        # F2 steepness admission: construction time is cfg validation time.
        validate_steepness_admission(cfg)

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
        reads "where the goal is".  The intensity channel is
        ``hs / cfg.hs_norm_ref`` against the FROZEN 0.60 m reference, so the
        channel keeps carrying the actual sea intensity even when a ladder
        pins ``hs_range`` to a single rung.
        """
        wave_x = torch.cos(self.mean_direction)
        wave_y = torch.sin(self.mean_direction)
        dot = forward_2d[:, 0] * wave_x + forward_2d[:, 1] * wave_y
        cross = forward_2d[:, 0] * wave_y - forward_2d[:, 1] * wave_x
        # FROZEN denominator: hs_norm_ref (0.60 m), never the per-config
        # hs_range[1] -- a pinned rung hs_range=(x, x) must not read 1.0 at
        # every rung.  Certified-path bit-identity (hs_range[1] == 0.60 ==
        # hs_norm_ref) is proven by test_wave_fix_package.py.
        hs_norm = self.hs / max(self.cfg.hs_norm_ref, 1.0e-6)
        return torch.stack((dot, cross, hs_norm), dim=-1)


__all__ = [
    "GRANDFATHERED_STEEPNESS_BOXES",
    "GRAVITY",
    "STEEPNESS_LIMIT",
    "SeaState",
    "SeaStateCfg",
    "analytic_spectrum_capture",
    "jonswap_spectrum",
    "significant_steepness",
    "validate_steepness_admission",
    "zero_amplitude_drag_force",
]
