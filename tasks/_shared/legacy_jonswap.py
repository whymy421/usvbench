"""Legacy calm-task API backed by the single shared JONSWAP spectrum."""

from __future__ import annotations

import math

import torch

from .waves import JONSWAPWaveField as _SharedJONSWAPWaveField


class LegacyJONSWAPWaveField(_SharedJONSWAPWaveField):
    """Expose old method names while using the normalized shared spectrum."""

    @property
    def wave_dir(self) -> torch.Tensor:
        return self.direction

    def _compute_jonswap_spectrum(
        self, hs: torch.Tensor, tp: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        return self._spectrum(hs, tp, gamma)

    def compute_elevation(
        self, t: float, pos_x: torch.Tensor, pos_y: torch.Tensor
    ) -> torch.Tensor:
        return self.elevation(t, pos_x, pos_y)

    def compute_forces(
        self,
        t: float,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        forward_2d: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the old plant's force channels from the shared wave field."""
        eta = self.elevation(t, pos_x, pos_y)
        deta = -(
            self.amplitudes
            * self.wave_numbers.unsqueeze(0)
            * torch.sin(self._total_phase(t, pos_x, pos_y))
        ).sum(dim=1)

        sideways = torch.stack((-forward_2d[:, 1], forward_2d[:, 0]), dim=-1)
        lateral_exposure = torch.abs((self.direction * sideways).sum(dim=1))
        frontal_exposure = (self.direction * forward_2d).sum(dim=1)

        return {
            "eta": eta,
            "heave_force": eta * 100.0,
            "roll_torque": lateral_exposure * deta * 200.0,
            "lateral_exposure": lateral_exposure,
            "wave_drag": torch.clamp(-frontal_exposure, min=0.0)
            * torch.abs(eta)
            * 15.0,
        }

    def get_obs(self, forward_2d: torch.Tensor) -> dict[str, torch.Tensor]:
        wave_dot = (forward_2d * self.direction).sum(dim=1)
        wave_cross = (
            forward_2d[:, 0] * self.direction[:, 1]
            - forward_2d[:, 1] * self.direction[:, 0]
        )
        return {
            "wave_dot": wave_dot.unsqueeze(-1),
            "wave_cross": wave_cross.unsqueeze(-1),
            "wave_height_norm": (self.hs / self.hs_range[1]).unsqueeze(-1),
            "hs": self.hs,
            "tp": self.tp,
        }


class JONSWAPWaveCfg:
    """Legacy field names for configs that have not moved to WaveCfg yet."""

    enable_wave: bool = True
    hs_min: float = 0.3
    hs_max: float = 1.0
    tp_min: float = 4.0
    tp_max: float = 7.0
    gamma_min: float = 1.0
    gamma_max: float = 5.0
    n_components: int = 30
    f_min: float = 0.04
    f_max: float = 0.5


def validate_spectrum(
    hs: float = 1.0,
    tp: float = 5.0,
    gamma: float = 3.3,
    n_components: int = 30,
) -> tuple[float, float]:
    """Return the discrete m0 and recovered Hs for a fixed sea state."""
    field = LegacyJONSWAPWaveField(
        num_envs=1,
        device=torch.device("cpu"),
        hs_range=(hs, hs),
        tp_range=(tp, tp),
        gamma_range=(gamma, gamma),
        n_components=n_components,
    )
    m0 = (field.amplitudes[0].square() / 2.0).sum().item()
    return m0, 4.0 * math.sqrt(m0)
