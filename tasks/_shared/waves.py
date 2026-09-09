"""Pure-Torch wave fields shared by environments and tests.

A wave field is the *only* thing waves change about an environment: the water
surface stops being the constant plane ``water_surface_z`` and becomes
``water_surface_z + elevation(t, x, y)``. Buoyancy, and therefore heave, then
follows from the existing hydrostatic code with no extra force terms. The
surface slope is exposed separately so an environment can tilt its restoring
equilibrium toward the wave normal (Froude-Krylov approximation) instead of
world-up.

This module deliberately depends on neither Isaac Lab nor Gymnasium, matching
``restoring.py`` and ``obs_superset.py``, so smoke tests can import it in a
plain Python process.

``CalmWater`` is the default and returns exact zeros, so a task that does not
opt into waves is numerically identical to one built before this module
existed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

GRAVITY_M_S2 = 9.81


class WaveField:
    """Interface shared by every wave model.

    Subclasses own their randomization; ``elevation`` and ``slope`` are pure
    functions of ``(t, x, y)`` given that state.
    """

    def randomize(
        self,
        env_ids: torch.Tensor,
        episode_indices: torch.Tensor | None = None,
    ) -> None:
        """Draw a fresh sea state / phase set for ``env_ids``.

        ``episode_indices`` is part of the reproducibility contract.  When it
        is supplied, every random quantity is a pure function of
        ``(seed, env_id, episode_index)`` and therefore does not depend on
        reset order or the process-wide Torch RNG.  Omitting it is retained as
        a convenience for older callers; those calls consume a per-environment
        deterministic counter rather than global randomness.
        """
        raise NotImplementedError

    def set_time_origin(self, env_ids: torch.Tensor, time_s: float) -> None:
        """Anchor the sea-state clock at reset for episode-local replay.

        The public wave equations take a scalar simulation time for efficient
        vectorised evaluation.  Implementations store this per-environment
        origin so a reset at a different wall-clock instant still starts the
        same seeded episode at local time zero.
        """

        return

    def elevation(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Surface elevation in metres, shape ``(num_envs,)``."""
        raise NotImplementedError

    def slope(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Surface gradient ``(deta/dx, deta/dy)``, shape ``(num_envs, 2)``."""
        raise NotImplementedError

    def lateral_slope(
        self, t: float, x: torch.Tensor, y: torch.Tensor, sideways: torch.Tensor
    ) -> torch.Tensor:
        """Surface slope along the hull's beam direction, shape ``(num_envs,)``.

        This is what tilts a hull into roll, and it is deliberately not the
        along-propagation slope: each spectral component's own direction is
        projected onto ``sideways`` before summing, so components arriving from
        different bearings contribute with the right sign.
        """
        raise NotImplementedError

    def vertical_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Surface vertical velocity ``deta/dt``, shape ``(num_envs,)``."""
        raise NotImplementedError

    @property
    def mean_direction(self) -> torch.Tensor:
        """Per-env mean propagation direction as a unit ``(num_envs, 2)``."""
        raise NotImplementedError

    def elevation_field(
        self, t: float, xs: torch.Tensor, ys: torch.Tensor, env_index: int = 0
    ) -> torch.Tensor:
        """Elevation over an arbitrary point cloud for one env's sea state.

        ``elevation`` evaluates one point per env, which is what the physics
        needs. Rendering needs the opposite: many points sharing a single env's
        phases and directions, so the water surface drawn on screen is the same
        surface the boat is feeling.
        """
        raise NotImplementedError

    def elevation_at(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Elevation at ``(num_envs, points)`` offsets, one sea state per row.

        Used to sample the surface under several stations on the same hull.
        """
        raise NotImplementedError

    def orbital_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Horizontal water particle velocity, shape ``(num_envs, 2)``.

        Deep-water Airy kinematics at the surface: each component contributes
        ``a * omega * cos(phase)`` along its own travel direction. Drag taken
        against ``v_boat - u_orbital`` instead of ``v_boat`` is how waves push
        a hull around without an authored wave-drag gain.
        """
        raise NotImplementedError


    @property
    def significant_height(self) -> torch.Tensor:
        """Per-env Hs in metres, for logging and wave-robustness metrics."""
        raise NotImplementedError


class CalmWater(WaveField):
    """Flat water. Zero-cost no-op that keeps calm baselines bit-identical."""

    def __init__(self, num_envs: int, device: torch.device):
        self._zeros = torch.zeros(num_envs, device=device)
        self._zeros2 = torch.zeros(num_envs, 2, device=device)

    def randomize(
        self,
        env_ids: torch.Tensor,
        episode_indices: torch.Tensor | None = None,
    ) -> None:
        return

    def set_time_origin(self, env_ids: torch.Tensor, time_s: float) -> None:
        return

    def elevation(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._zeros

    def slope(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._zeros2

    def lateral_slope(
        self, t: float, x: torch.Tensor, y: torch.Tensor, sideways: torch.Tensor
    ) -> torch.Tensor:
        return self._zeros

    def vertical_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self._zeros

    @property
    def mean_direction(self) -> torch.Tensor:
        return self._zeros2

    def elevation_field(
        self, t: float, xs: torch.Tensor, ys: torch.Tensor, env_index: int = 0
    ) -> torch.Tensor:
        return torch.zeros_like(xs)

    def elevation_at(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(x)

    def orbital_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self._zeros2

    @property
    def significant_height(self) -> torch.Tensor:
        return self._zeros


def _seed_for(seed: int, env_id: int, episode_index: int, stream: int) -> int:
    """Stable 63-bit seed mixer used by the stateless wave draws.

    This is intentionally a small integer mixer rather than Python's hash or
    Torch's global RNG.  Python hash randomisation, reset ordering, and other
    workers can therefore never change a component's phase or heading.
    """

    mask = (1 << 63) - 1
    value = int(seed) & mask
    value = (value + (int(env_id) + 1) * 6364136223846793005) & mask
    value = (value + (int(episode_index) + 1) * 1442695040888963407) & mask
    value = (value + (int(stream) + 1) * 3202034522624059733) & mask
    value ^= value >> 30
    value = (value * 3202034522624059733) & mask
    value ^= value >> 27
    return int(value & mask)


def _cpu_uniform(
    shape: Sequence[int], seed: int, env_id: int, episode_index: int, stream: int
) -> torch.Tensor:
    """Stateless CPU draw, returned as float32 for device-independent replay."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed_for(seed, env_id, episode_index, stream))
    return torch.rand(tuple(int(dim) for dim in shape), generator=generator)


def _resolve_randomize_indices(
    counter: torch.Tensor,
    env_ids: torch.Tensor,
    episode_indices: torch.Tensor | None,
) -> tuple[list[int], list[int]]:
    """Resolve env ids and episode indices without depending on tensor order."""

    ids = [int(value) for value in env_ids.detach().cpu().reshape(-1).tolist()]
    if episode_indices is None:
        episodes = [int(value) for value in counter[ids].tolist()]
        if ids:
            counter[ids] += 1
    else:
        episodes = [
            int(value)
            for value in episode_indices.detach().cpu().reshape(-1).tolist()
        ]
        if len(episodes) != len(ids):
            raise ValueError("episode_indices must have one value per env_id")
        if ids:
            counter[ids] = torch.as_tensor(episodes, dtype=counter.dtype) + 1
    return ids, episodes


class AiryWaveField(WaveField):
    """Single-frequency regular wave, deep-water dispersion.

        eta = a * cos(k * (dx * x + dy * y) - omega * t + phi)

    One frequency and one direction, so the sea state is fully described by
    (height, period, direction). This is the controlled comparison point for
    JONSWAP: it isolates "is there a wave at all" from "is the wave irregular".
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        height_m: float = 0.5,
        period_s: float = 5.0,
        direction_deg: float | None = None,
        gravity: float = GRAVITY_M_S2,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.height_m = float(height_m)
        self.period_s = float(period_s)
        self.direction_deg = direction_deg
        self.seed = int(seed)
        self._randomize_counter = torch.zeros(num_envs, dtype=torch.long)

        self.amplitude = torch.full(
            (num_envs,), self.height_m / 2.0, device=device
        )
        self.omega = 2.0 * math.pi / self.period_s
        self.wave_number = self.omega ** 2 / gravity

        self.direction = torch.zeros(num_envs, 2, device=device)
        self.phase = torch.zeros(num_envs, device=device)
        self.time_origin = torch.zeros(num_envs, device=device)
        self.randomize(
            torch.arange(num_envs, device=device),
            episode_indices=torch.zeros(num_envs, dtype=torch.long, device=device),
        )

    def randomize(
        self,
        env_ids: torch.Tensor,
        episode_indices: torch.Tensor | None = None,
    ) -> None:
        ids, episodes = _resolve_randomize_indices(
            self._randomize_counter, env_ids, episode_indices
        )
        if not ids:
            return
        angles = []
        phases = []
        for env_id, episode_index in zip(ids, episodes):
            if self.direction_deg is None:
                angle = _cpu_uniform((1,), self.seed, env_id, episode_index, 1)[0]
                angles.append(float(angle * (2.0 * math.pi)))
            else:
                angles.append(math.radians(self.direction_deg))
            phase = _cpu_uniform((1,), self.seed, env_id, episode_index, 2)[0]
            phases.append(float(phase * (2.0 * math.pi)))
        index = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        angle_tensor = torch.as_tensor(angles, device=self.device)
        self.direction[index, 0] = torch.cos(angle_tensor)
        self.direction[index, 1] = torch.sin(angle_tensor)
        self.phase[index] = torch.as_tensor(phases, device=self.device)

    def set_time_origin(self, env_ids: torch.Tensor, time_s: float) -> None:
        self.time_origin[env_ids] = float(time_s)

    def _total_phase(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        projected = self.direction[:, 0] * x + self.direction[:, 1] * y
        local_time = t - self.time_origin
        return self.wave_number * projected - self.omega * local_time + self.phase

    def elevation(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.amplitude * torch.cos(self._total_phase(t, x, y))

    def slope(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        common = -self.amplitude * self.wave_number * torch.sin(
            self._total_phase(t, x, y)
        )
        return torch.stack(
            (common * self.direction[:, 0], common * self.direction[:, 1]), dim=-1
        )

    def lateral_slope(
        self, t: float, x: torch.Tensor, y: torch.Tensor, sideways: torch.Tensor
    ) -> torch.Tensor:
        along = -self.amplitude * self.wave_number * torch.sin(
            self._total_phase(t, x, y)
        )
        # Single component, so its direction is the wave direction itself.
        projection = (
            self.direction[:, 0] * sideways[:, 0]
            + self.direction[:, 1] * sideways[:, 1]
        )
        return along * projection

    def vertical_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.amplitude * self.omega * torch.sin(self._total_phase(t, x, y))

    @property
    def mean_direction(self) -> torch.Tensor:
        return self.direction

    def elevation_field(
        self, t: float, xs: torch.Tensor, ys: torch.Tensor, env_index: int = 0
    ) -> torch.Tensor:
        projected = (
            self.direction[env_index, 0] * xs + self.direction[env_index, 1] * ys
        )
        phase = (
            self.wave_number * projected
            - self.omega * (t - self.time_origin[env_index])
            + self.phase[env_index]
        )
        return self.amplitude[env_index] * torch.cos(phase)

    def elevation_at(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        projected = (
            self.direction[:, 0:1] * x + self.direction[:, 1:2] * y
        )
        phase = (
            self.wave_number * projected
            - self.omega * (t - self.time_origin).unsqueeze(1)
            + self.phase.unsqueeze(1)
        )
        return self.amplitude.unsqueeze(1) * torch.cos(phase)

    def orbital_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        speed = self.amplitude * self.omega * torch.cos(self._total_phase(t, x, y))
        return torch.stack(
            (speed * self.direction[:, 0], speed * self.direction[:, 1]), dim=-1
        )

    @property
    def significant_height(self) -> torch.Tensor:
        # A regular wave has no spectrum; Hs is defined as its wave height so
        # that sea-state logging stays comparable across models.
        return torch.full(
            (self.num_envs,), self.height_m, device=self.device
        )


class JONSWAPWaveField(WaveField):
    """Irregular sea built from a JONSWAP spectrum with directional spreading.

        eta(x, y, t) = sum_n a_n * cos(k_n * (dx_n * x + dy_n * y)
                                       - omega_n * t + phi_n)

    with ``a_n = sqrt(2 * S(f_n) * df)`` from the spectrum, ``k_n`` from the
    deep-water dispersion relation, random phases, and each component's
    direction scattered around the per-env mean direction.

    Spectrum implementation follows ``tasks/rov_calm_nav/jonswap_wave.py``
    (author: Raina), which the calm reference tasks already carry; the moving
    part here is the added ``slope`` and the shared interface.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        hs_range: tuple[float, float] = (0.3, 1.0),
        tp_range: tuple[float, float] = (4.0, 7.0),
        gamma_range: tuple[float, float] = (1.0, 5.0),
        n_components: int = 30,
        f_min: float = 0.04,
        f_max: float = 0.5,
        spread_deg: float = 30.0,
        direction_deg: float | None = None,
        gravity: float = GRAVITY_M_S2,
        seed: int = 0,
        sampling_mode: str = "uniform",
        hs_levels: tuple[float, ...] | None = None,
        tp_levels: tuple[float, ...] | None = None,
        gamma_levels: tuple[float, ...] | None = None,
        max_steepness: float | None = None,
        frequency_jitter: float = 0.22,
    ):
        self.num_envs = num_envs
        self.device = device
        self.n_components = int(n_components)
        self.hs_range = hs_range
        self.tp_range = tp_range
        self.gamma_range = gamma_range
        self.sampling_mode = str(sampling_mode).lower()
        self.hs_levels = tuple(float(value) for value in (hs_levels or ()))
        self.tp_levels = tuple(float(value) for value in (tp_levels or ()))
        self.gamma_levels = tuple(float(value) for value in (gamma_levels or ()))
        self.max_steepness = max_steepness
        self.seed = int(seed)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.spread_deg = float(spread_deg)
        self.spread_rad = math.radians(self.spread_deg)
        self.direction_deg = direction_deg

        if self.sampling_mode not in {"uniform", "levels"}:
            raise ValueError("sampling_mode must be 'uniform' or 'levels'")
        if self.n_components < 1 or not self.f_min < self.f_max:
            raise ValueError("JONSWAP requires n_components >= 1 and f_min < f_max")
        if (
            self.f_min <= 0.0
            or self.hs_range[0] < 0.0
            or self.hs_range[0] > self.hs_range[1]
            or self.tp_range[0] <= 0.0
            or self.tp_range[0] > self.tp_range[1]
            or self.gamma_range[0] > self.gamma_range[1]
        ):
            raise ValueError("JONSWAP frequencies, Tp, and Hs bounds must be valid")
        if self.gamma_range[0] <= 0.0:
            raise ValueError("JONSWAP gamma bounds must be positive")
        if self.spread_deg < 0.0:
            raise ValueError("spread_deg must be non-negative")
        if not 0.0 <= float(frequency_jitter) < 0.5:
            raise ValueError("frequency_jitter must be in [0, 0.5)")
        if self.sampling_mode == "levels":
            for name, values in (
                ("hs_levels", self.hs_levels),
                ("tp_levels", self.tp_levels),
                ("gamma_levels", self.gamma_levels),
            ):
                if not values:
                    raise ValueError(f"{name} must be non-empty when sampling_mode='levels'")
        for name, values, bounds in (
            ("hs_levels", self.hs_levels, self.hs_range),
            ("tp_levels", self.tp_levels, self.tp_range),
            ("gamma_levels", self.gamma_levels, self.gamma_range),
        ):
            if values and any(value < bounds[0] or value > bounds[1] for value in values):
                raise ValueError(f"{name} must lie inside its corresponding range")
        if self.max_steepness is not None:
            hs_domain = self.hs_levels if self.sampling_mode == "levels" else (self.hs_range[1],)
            tp_domain = self.tp_levels if self.sampling_mode == "levels" else (self.tp_range[0],)
            worst_steepness = max(
                float(hs) / (gravity * float(tp) ** 2 / (2.0 * math.pi))
                for hs in hs_domain
                for tp in tp_domain
            )
            if worst_steepness > float(self.max_steepness) + 1.0e-12:
                raise ValueError(
                    "Hs/Tp domain exceeds max_steepness: "
                    f"worst Hs/lambda_p={worst_steepness:.4f} > "
                    f"{float(self.max_steepness):.4f}"
                )

        self.df = (self.f_max - self.f_min) / self.n_components
        # A fixed, nonuniform frequency grid avoids the exact 1/df repetition
        # of a uniformly binned finite Fourier sum.  The jitter is deliberately
        # small enough to keep the centres ordered and inside the configured
        # band; the individual bin widths below preserve m0 exactly.
        base_freqs = torch.linspace(
            self.f_min + self.df / 2.0,
            self.f_max - self.df / 2.0,
            self.n_components,
            device=device,
        )
        offsets = torch.as_tensor(
            [
                frequency_jitter * self.df * math.sin((index + 1) * math.sqrt(2.0) + 0.37)
                for index in range(self.n_components)
            ],
            device=device,
            dtype=base_freqs.dtype,
        )
        self.freqs = base_freqs + offsets
        if self.n_components > 1 and bool(torch.any(self.freqs[1:] <= self.freqs[:-1])):
            raise ValueError("frequency_jitter produced a non-increasing frequency grid")
        edges = torch.empty(self.n_components + 1, device=device, dtype=base_freqs.dtype)
        edges[0] = self.f_min
        edges[-1] = self.f_max
        if self.n_components > 1:
            edges[1:-1] = 0.5 * (self.freqs[:-1] + self.freqs[1:])
        self.bin_widths = edges[1:] - edges[:-1]
        self.omegas = 2.0 * math.pi * self.freqs
        self.wave_numbers = self.omegas ** 2 / gravity

        self.hs = torch.zeros(num_envs, device=device)
        self.tp = torch.zeros(num_envs, device=device)
        self.gamma = torch.zeros(num_envs, device=device)
        self.direction = torch.zeros(num_envs, 2, device=device)
        self.time_origin = torch.zeros(num_envs, device=device)

        self.amplitudes = torch.zeros(num_envs, self.n_components, device=device)
        self.phases = torch.zeros(num_envs, self.n_components, device=device)
        self.comp_dir_x = torch.zeros(num_envs, self.n_components, device=device)
        self.comp_dir_y = torch.zeros(num_envs, self.n_components, device=device)
        self._randomize_counter = torch.zeros(num_envs, dtype=torch.long)

        self.randomize(
            torch.arange(num_envs, device=device),
            episode_indices=torch.zeros(num_envs, dtype=torch.long, device=device),
        )

    def _spectrum(
        self, hs: torch.Tensor, tp: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """JONSWAP S(f) = alpha * f^-5 * exp(-1.25 (fp/f)^4) * gamma^r.

        Normalised numerically against the *discrete, truncated* band rather
        than by the usual ``(1 - 0.287 ln gamma)`` closed form. That factor is
        derived for a continuous spectrum over all frequencies, so once the
        band is cut at f_min/f_max and binned it no longer holds: at Tp = 4 s
        with gamma = 5 it leaves the realised Hs about 8% low. Rescaling by the
        band's own m0 makes ``4 sqrt(m0) == Hs`` exact for any band, Tp, or
        gamma, which is what lets a sea state be specified by Hs at all.
        """
        fp = 1.0 / tp
        fp_e = fp.unsqueeze(1)
        hs_e = hs.unsqueeze(1)
        gamma_e = gamma.unsqueeze(1)
        f_e = self.freqs.unsqueeze(0)

        shape = f_e.pow(-5) * torch.exp(-1.25 * (fp_e / f_e).pow(4))
        sigma = torch.where(f_e <= fp_e, 0.07, 0.09)
        r = torch.exp(-0.5 * ((f_e - fp_e) / (sigma * fp_e)).pow(2))
        spectrum = torch.clamp(shape * gamma_e.pow(r), min=0.0)

        m0 = (spectrum * self.bin_widths.unsqueeze(0)).sum(dim=1, keepdim=True).clamp(min=1e-12)
        return spectrum * (hs_e / (4.0 * torch.sqrt(m0))) ** 2

    def _sample_parameter(
        self,
        bounds: tuple[float, float],
        levels: tuple[float, ...],
        env_id: int,
        episode_index: int,
        stream: int,
    ) -> float:
        unit = float(_cpu_uniform((1,), self.seed, env_id, episode_index, stream)[0])
        if self.sampling_mode == "levels":
            values = levels
            return values[min(int(unit * len(values)), len(values) - 1)]
        low, high = bounds
        return low + unit * (high - low)

    def randomize(
        self,
        env_ids: torch.Tensor,
        episode_indices: torch.Tensor | None = None,
    ) -> None:
        ids, episodes = _resolve_randomize_indices(
            self._randomize_counter, env_ids, episode_indices
        )
        if not ids:
            return
        hs_values, tp_values, gamma_values, base_angles = [], [], [], []
        phase_values, spread_values = [], []
        for env_id, episode_index in zip(ids, episodes):
            hs_values.append(
                self._sample_parameter(self.hs_range, self.hs_levels, env_id, episode_index, 1)
            )
            tp_values.append(
                self._sample_parameter(self.tp_range, self.tp_levels, env_id, episode_index, 2)
            )
            gamma_values.append(
                self._sample_parameter(self.gamma_range, self.gamma_levels, env_id, episode_index, 3)
            )
            if self.direction_deg is None:
                base = float(_cpu_uniform((1,), self.seed, env_id, episode_index, 4)[0])
                base_angles.append(base * 2.0 * math.pi)
            else:
                base_angles.append(math.radians(self.direction_deg))
            phase_values.append(
                _cpu_uniform((self.n_components,), self.seed, env_id, episode_index, 5)
                * (2.0 * math.pi)
            )
            spread_values.append(
                (_cpu_uniform((self.n_components,), self.seed, env_id, episode_index, 6) - 0.5)
                * self.spread_rad
            )

        index = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        base_angle = torch.as_tensor(base_angles, device=self.device)
        self.hs[index] = torch.as_tensor(hs_values, device=self.device)
        self.tp[index] = torch.as_tensor(tp_values, device=self.device)
        self.gamma[index] = torch.as_tensor(gamma_values, device=self.device)
        self.direction[index, 0] = torch.cos(base_angle)
        self.direction[index, 1] = torch.sin(base_angle)
        self.phases[index] = torch.stack(phase_values).to(self.device)
        spread = torch.stack(spread_values).to(self.device)
        comp_angles = base_angle.unsqueeze(1) + spread
        self.comp_dir_x[index] = torch.cos(comp_angles)
        self.comp_dir_y[index] = torch.sin(comp_angles)

        spectrum = self._spectrum(self.hs[index], self.tp[index], self.gamma[index])
        self.amplitudes[index] = torch.sqrt(
            2.0 * spectrum * self.bin_widths.unsqueeze(0)
        )

    def set_time_origin(self, env_ids: torch.Tensor, time_s: float) -> None:
        self.time_origin[env_ids] = float(time_s)

    @property
    def nominal_repeat_period_s(self) -> float:
        """The old uniform-grid period, retained as a diagnostic only."""

        return 1.0 / self.df

    def _total_phase(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        projected = (
            self.comp_dir_x * x.unsqueeze(1) + self.comp_dir_y * y.unsqueeze(1)
        )
        spatial = projected * self.wave_numbers.unsqueeze(0)
        local_time = t - self.time_origin.unsqueeze(1)
        return spatial - self.omegas.unsqueeze(0) * local_time + self.phases

    def elevation(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (self.amplitudes * torch.cos(self._total_phase(t, x, y))).sum(dim=1)

    def slope(self, t: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        common = -(
            self.amplitudes
            * self.wave_numbers.unsqueeze(0)
            * torch.sin(self._total_phase(t, x, y))
        )
        return torch.stack(
            ((common * self.comp_dir_x).sum(dim=1),
             (common * self.comp_dir_y).sum(dim=1)),
            dim=-1,
        )

    def lateral_slope(
        self, t: float, x: torch.Tensor, y: torch.Tensor, sideways: torch.Tensor
    ) -> torch.Tensor:
        # Each component is projected onto the beam direction before summing --
        # with directional spreading the components arrive from different
        # bearings, so a single mean-direction projection would lose their signs.
        component_lateral = (
            self.comp_dir_x * sideways[:, 0:1] + self.comp_dir_y * sideways[:, 1:2]
        )
        return -(
            self.amplitudes
            * self.wave_numbers.unsqueeze(0)
            * torch.sin(self._total_phase(t, x, y))
            * component_lateral
        ).sum(dim=1)

    def vertical_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.amplitudes
            * self.omegas.unsqueeze(0)
            * torch.sin(self._total_phase(t, x, y))
        ).sum(dim=1)

    @property
    def mean_direction(self) -> torch.Tensor:
        return self.direction

    def elevation_field(
        self, t: float, xs: torch.Tensor, ys: torch.Tensor, env_index: int = 0
    ) -> torch.Tensor:
        # (points, components) -- one env's spectrum evaluated everywhere.
        projected = (
            self.comp_dir_x[env_index].unsqueeze(0) * xs.unsqueeze(1)
            + self.comp_dir_y[env_index].unsqueeze(0) * ys.unsqueeze(1)
        )
        phase = (
            projected * self.wave_numbers.unsqueeze(0)
            - self.omegas.unsqueeze(0) * (t - self.time_origin[env_index])
            + self.phases[env_index].unsqueeze(0)
        )
        return (self.amplitudes[env_index].unsqueeze(0) * torch.cos(phase)).sum(dim=1)

    def elevation_at(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # (envs, points, components): every station on the hull sees the same
        # sea state, evaluated at its own position.
        projected = (
            self.comp_dir_x.unsqueeze(1) * x.unsqueeze(2)
            + self.comp_dir_y.unsqueeze(1) * y.unsqueeze(2)
        )
        phase = (
            projected * self.wave_numbers.view(1, 1, -1)
            - self.omegas.view(1, 1, -1)
            * (t - self.time_origin).view(-1, 1, 1)
            + self.phases.unsqueeze(1)
        )
        return (self.amplitudes.unsqueeze(1) * torch.cos(phase)).sum(dim=2)

    def orbital_velocity(
        self, t: float, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        speed = (
            self.amplitudes
            * self.omegas.unsqueeze(0)
            * torch.cos(self._total_phase(t, x, y))
        )
        return torch.stack(
            ((speed * self.comp_dir_x).sum(dim=1),
             (speed * self.comp_dir_y).sum(dim=1)),
            dim=-1,
        )

    @property
    def significant_height(self) -> torch.Tensor:
        return self.hs


def make_wave_field(
    cfg, num_envs: int, device: torch.device, seed: int = 0
) -> WaveField:
    """Build the wave field named by ``cfg.mode`` ("calm" / "airy" / "jonswap").

    ``cfg`` is any object carrying the attributes below -- an Isaac Lab
    ``configclass`` in the environments, a plain namespace in tests.
    """
    mode = str(getattr(cfg, "mode", "calm")).lower()
    if mode == "calm":
        return CalmWater(num_envs, device)
    if mode == "airy":
        if float(cfg.airy_height_m) <= 0.0:
            return CalmWater(num_envs, device)
        return AiryWaveField(
            num_envs=num_envs,
            device=device,
            height_m=cfg.airy_height_m,
            period_s=cfg.airy_period_s,
            direction_deg=cfg.direction_deg,
            seed=seed,
        )
    if mode == "jonswap":
        if float(cfg.hs_max_m) <= 0.0:
            return CalmWater(num_envs, device)
        return JONSWAPWaveField(
            num_envs=num_envs,
            device=device,
            hs_range=(cfg.hs_min_m, cfg.hs_max_m),
            tp_range=(cfg.tp_min_s, cfg.tp_max_s),
            gamma_range=(cfg.gamma_min, cfg.gamma_max),
            n_components=cfg.n_components,
            f_min=cfg.f_min_hz,
            f_max=cfg.f_max_hz,
            spread_deg=cfg.spread_deg,
            direction_deg=cfg.direction_deg,
            seed=seed,
            sampling_mode=getattr(cfg, "sampling_mode", "uniform"),
            hs_levels=getattr(cfg, "hs_levels_m", None),
            tp_levels=getattr(cfg, "tp_levels_s", None),
            gamma_levels=getattr(cfg, "gamma_levels", None),
            max_steepness=getattr(cfg, "max_steepness", None),
        )
    raise ValueError(
        f"unknown wave mode {mode!r}; expected 'calm', 'airy' or 'jonswap'"
    )
