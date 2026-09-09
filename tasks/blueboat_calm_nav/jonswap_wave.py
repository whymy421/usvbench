"""
JONSWAP Irregular Wave Module for Isaac Lab
============================================
This module replaces a single sinusoid with an irregular JONSWAP spectrum.

Usage:
    Replace `_init_wave_field()` and `_compute_wave_forces()` in MultiUSVEnv.

    from .jonswap_wave import JONSWAPWaveField

    # In _setup_scene() or __init__:
    self.wave_field = JONSWAPWaveField(cfg=cfg.wave_cfg, num_envs=self.num_envs, device=self.device)

    # In _compute_wave_forces():
    eta, vel_x, vel_y = self.wave_field.compute(t, positions_x, positions_y)

Author: Raina (Multi-USV MARL Project)
"""

import torch
import math


class JONSWAPWaveField:
    """Irregular wave field generated from a JONSWAP spectrum.

    Core model:
        η(x,y,t) = Σ aₙ · cos(kₙ·(dx·x + dy·y) - ωₙ·t + φₙ)

        where:
        - aₙ = √(2·S(fₙ)·Δf) is the JONSWAP-derived harmonic amplitude
        - ωₙ = 2π·fₙ is angular frequency
        - kₙ = ωₙ²/g is the deep-water wavenumber
        - φₙ is a random phase in [0, 2π)
        - (dx, dy) is the unit propagation direction
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        # JONSWAP spectrum parameters.
        hs_range: tuple[float, float] = (0.3, 1.0),     # Significant wave height (m)
        tp_range: tuple[float, float] = (4.0, 7.0),     # Peak period (s)
        gamma_range: tuple[float, float] = (1.0, 5.0),  # Peak enhancement factor
        # Discretization parameters.
        n_components: int = 30,                          # Harmonic component count
        f_min: float = 0.04,                              # Minimum frequency (Hz)
        f_max: float = 0.5,                               # Maximum frequency (Hz)
        # Physical constants.
        gravity: float = 9.81,
    ):
        self.num_envs = num_envs
        self.device = device
        self.n_components = n_components
        self.gravity = gravity
        self.f_min = f_min
        self.f_max = f_max
        self.hs_range = hs_range
        self.tp_range = tp_range
        self.gamma_range = gamma_range

        # All environments share the same frequency grid.
        self.df = (f_max - f_min) / n_components
        # (n_components,)
        self.freqs = torch.linspace(
            f_min + self.df / 2,
            f_max - self.df / 2,
            n_components,
            device=device
        )
        self.omegas = 2 * math.pi * self.freqs  # (n_components,)
        self.wave_numbers = self.omegas ** 2 / gravity  #   (n_components,)

        # Per-environment wave parameters, randomized on reset.
        self.hs = torch.zeros(num_envs, device=device)
        self.tp = torch.zeros(num_envs, device=device)
        self.gamma = torch.zeros(num_envs, device=device)
        self.wave_dir = torch.zeros(num_envs, 2, device=device)  # (dx, dy)

        # Per-environment component amplitudes and phases.
        # (num_envs, n_components)
        self.amplitudes = torch.zeros(num_envs, n_components, device=device)
        self.phases = torch.zeros(num_envs, n_components, device=device)
        self.comp_dir_x = torch.zeros(num_envs, n_components, device=device)
        self.comp_dir_y = torch.zeros(num_envs, n_components, device=device)
        # Initialize all environments.
        all_ids = torch.arange(num_envs, device=device)
        self.randomize(all_ids)

    def _compute_jonswap_spectrum(self, hs: torch.Tensor, tp: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Compute the JONSWAP spectral density.

        S(f) = α · f^(-5) · exp(-5/4 · (fp/f)^4) · γ^r

        Parameters:
            α = 5/16 · Hs² · fp⁴  (Phillips normalization to Hs)
            r = exp(-0.5 · ((f - fp) / (σ · fp))²)
            σ = 0.07 (f ≤ fp), 0.09 (f > fp)

        Args:
            hs: (num_envs,) significant wave heights
            tp: (num_envs,) peak periods
            gamma: (num_envs,) peak enhancement factors

        Returns:
            S: (num_envs, n_components) spectral density in m²/Hz
        """
        fp = 1.0 / tp  # (num_envs,)

        # Broadcast to (num_envs, 1) and (1, n_components).
        fp_e = fp.unsqueeze(1)        # (num_envs, 1)
        hs_e = hs.unsqueeze(1)        # (num_envs, 1)
        gamma_e = gamma.unsqueeze(1)  # (num_envs, 1)
        f_e = self.freqs.unsqueeze(0) # (1, n_components)

        # Phillips normalization.
        alpha = 5.0 / 16.0 * hs_e ** 2 * fp_e ** 4

        # Pierson-Moskowitz base spectrum.
        pm = alpha * f_e.pow(-5) * torch.exp(-1.25 * (fp_e / f_e).pow(4))

        # sigma: 0.07 for f <= fp, 0.09 for f > fp
        sigma = torch.where(f_e <= fp_e, 0.07, 0.09)

        # Peak-enhancement exponent.
        r = torch.exp(-0.5 * ((f_e - fp_e) / (sigma * fp_e)).pow(2))

        # JONSWAP = PM × γ^r
        S = pm * gamma_e.pow(r)

        # Numerical safety.
        S = torch.clamp(S, min=0.0)

        return S

    def randomize(self, env_ids):
        num = len(env_ids)

        self.hs[env_ids] = torch.rand(num, device=self.device) * (self.hs_range[1] - self.hs_range[0]) + self.hs_range[0]
        self.tp[env_ids] = torch.rand(num, device=self.device) * (self.tp_range[1] - self.tp_range[0]) + self.tp_range[0]
        self.gamma[env_ids] = torch.rand(num, device=self.device) * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]

        angles = torch.rand(num, device=self.device) * 2 * math.pi
        self.wave_dir[env_ids, 0] = torch.cos(angles)
        self.wave_dir[env_ids, 1] = torch.sin(angles)

        self.phases[env_ids] = torch.rand(num, self.n_components, device=self.device) * 2 * math.pi

        # Spread each component within +/-30 degrees of the mean direction.
        spread_angles = (torch.rand(num, self.n_components, device=self.device) - 0.5) * (math.pi / 3)
        base_angle = torch.atan2(self.wave_dir[env_ids, 1], self.wave_dir[env_ids, 0])
        comp_angles = base_angle.unsqueeze(1) + spread_angles
        self.comp_dir_x[env_ids] = torch.cos(comp_angles)
        self.comp_dir_y[env_ids] = torch.sin(comp_angles)

        S = self._compute_jonswap_spectrum(self.hs[env_ids], self.tp[env_ids], self.gamma[env_ids])
        self.amplitudes[env_ids] = torch.sqrt(2.0 * S * self.df)

    def compute_elevation(self, t, pos_x, pos_y):
        spatial_phase = (self.comp_dir_x * pos_x.unsqueeze(1) + self.comp_dir_y * pos_y.unsqueeze(1)) * self.wave_numbers.unsqueeze(0)
        temporal_phase = self.omegas.unsqueeze(0) * t
        total_phase = spatial_phase - temporal_phase + self.phases
        eta = (self.amplitudes * torch.cos(total_phase)).sum(dim=1)
        return eta

    def compute_forces(self, t, pos_x, pos_y, forward_2d):
        eta = self.compute_elevation(t, pos_x, pos_y)

        spatial_phase = (self.comp_dir_x * pos_x.unsqueeze(1) + self.comp_dir_y * pos_y.unsqueeze(1)) * self.wave_numbers.unsqueeze(0)
        temporal_phase = self.omegas.unsqueeze(0) * t
        total_phase = spatial_phase - temporal_phase + self.phases
        deta = -(self.amplitudes * self.wave_numbers.unsqueeze(0) * torch.sin(total_phase)).sum(dim=1)

        sideways = torch.stack([-forward_2d[:, 1], forward_2d[:, 0]], dim=-1)
        lateral_exposure = torch.abs(self.wave_dir[:, 0] * sideways[:, 0] + self.wave_dir[:, 1] * sideways[:, 1])
        frontal_exposure = self.wave_dir[:, 0] * forward_2d[:, 0] + self.wave_dir[:, 1] * forward_2d[:, 1]

        heave_force = eta * 100.0
        roll_torque = lateral_exposure * deta * 200.0
        wave_drag = torch.clamp(-frontal_exposure, min=0) * torch.abs(eta) * 15.0

        return {
            "eta": eta,
            "heave_force": heave_force,
            "roll_torque": roll_torque,
            "lateral_exposure": lateral_exposure,
            "wave_drag": wave_drag,
        }

    def get_obs(self, forward_2d: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return wave features for the RL policy.

        Args:
            forward_2d: (num_envs, 2) vehicle forward directions

        Returns:
            dict with:
            - wave_dot: cosine of the bow-to-wave angle
            - wave_cross: sine of the bow-to-wave angle
            - wave_height_norm: normalized significant wave height
            - hs: current significant wave height for a centralized critic
            - tp: current peak period
        """
        wave_dot = (forward_2d[:, 0] * self.wave_dir[:, 0]
                    + forward_2d[:, 1] * self.wave_dir[:, 1])
        wave_cross = (forward_2d[:, 0] * self.wave_dir[:, 1]
                      - forward_2d[:, 1] * self.wave_dir[:, 0])
        wave_height_norm = self.hs / self.hs_range[1]  #  [0,1]

        return {
            "wave_dot": wave_dot.unsqueeze(-1),
            "wave_cross": wave_cross.unsqueeze(-1),
            "wave_height_norm": wave_height_norm.unsqueeze(-1),
            "hs": self.hs,
            "tp": self.tp,
        }


# ============================================
# Configuration class following the Isaac Lab configclass style.
# ============================================

class JONSWAPWaveCfg:
    """JONSWAP wave configuration replacing WavePhysicsCfg."""
    enable_wave: bool = True
    # JONSWAP spectrum ranges randomized during training.
    hs_min: float = 0.3        # Minimum significant wave height (m)
    hs_max: float = 1.0        # Maximum significant wave height (m)
    tp_min: float = 4.0        # Minimum peak period (s)
    tp_max: float = 7.0        # Maximum peak period (s)
    gamma_min: float = 1.0     # Minimum peak enhancement factor
    gamma_max: float = 5.0     # Maximum peak enhancement factor
    # Frequency discretization.
    n_components: int = 30     # Harmonic component count
    f_min: float = 0.04        # Minimum frequency (Hz)
    f_max: float = 0.5         # Maximum frequency (Hz)


# ============================================
# Validation helper for debugging and regression checks.
# ============================================

def validate_spectrum(hs: float = 1.0, tp: float = 5.0, gamma: float = 3.3, n_components: int = 30):
    """Validate the JONSWAP spectrum implementation.

    Checks:
    1. 4√m₀ ≈ Hs (zeroth-moment normalization)
    2. The spectral peak is near fp = 1/Tp
    3. The wave-surface time-series statistics

    Usage:
        python -c "from jonswap_wave import validate_spectrum; validate_spectrum()"
    """
    device = torch.device("cpu")

    wave = JONSWAPWaveField(
        num_envs=1,
        device=device,
        hs_range=(hs, hs),
        tp_range=(tp, tp),
        gamma_range=(gamma, gamma),
        n_components=n_components,
    )

    # Evaluate the spectrum.
    S = wave._compute_jonswap_spectrum(
        torch.tensor([hs]),
        torch.tensor([tp]),
        torch.tensor([gamma])
    )

    m0 = (S * wave.df).sum().item()
    hs_check = 4 * math.sqrt(m0)

    print(f"JONSWAP Spectrum Validation")
    print(f"{'='*40}")
    print(f"Input:  Hs={hs}m, Tp={tp}s, gamma={gamma}")
    print(f"N components: {n_components}")
    print(f"Freq range: {wave.f_min}-{wave.f_max} Hz")
    print(f"{'='*40}")
    print(f"m0 = {m0:.4f} m²")
    print(f"4√m0 = {hs_check:.3f} m (should ≈ {hs} m)")
    print(f"Error: {abs(hs_check - hs)/hs*100:.1f}%")

    # Locate the spectral peak.
    peak_idx = S[0].argmax().item()
    peak_freq = wave.freqs[peak_idx].item()
    print(f"Peak freq: {peak_freq:.3f} Hz (should ≈ {1/tp:.3f} Hz)")

    # Check the time series.
    dt = 0.1
    t_max = 600.0
    etas = []
    for t in range(int(t_max / dt)):
        eta = wave.compute_elevation(
            t * dt,
            torch.zeros(1),
            torch.zeros(1)
        )
        etas.append(eta.item())

    etas_t = torch.tensor(etas)
    hs_ts = 4 * etas_t.std().item()
    print(f"\nTime series (600s):")
    print(f"  Hs from time series: {hs_ts:.3f} m (should ≈ {hs} m)")
    print(f"  Max elevation: {etas_t.max().item():.3f} m")
    print(f"  Min elevation: {etas_t.min().item():.3f} m")
    print(f"  Mean (should ≈ 0): {etas_t.mean().item():.4f} m")

    print(f"\n✅ Validation complete!")
    return m0, hs_check


if __name__ == "__main__":
    validate_spectrum()
