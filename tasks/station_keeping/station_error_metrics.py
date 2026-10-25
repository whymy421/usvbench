"""Exact per-episode horizontal station-error statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class StationErrorAccumulator:
    """Retain control-step errors and latch exact RMS/p95 episode summaries."""

    def __init__(
        self,
        *,
        num_envs: int,
        max_episode_samples: int,
        control_step_s: float,
        device: torch.device | str,
        final_window_s: float = 60.0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if max_episode_samples <= 0:
            raise ValueError("max_episode_samples must be positive")
        if control_step_s <= 0.0 or final_window_s <= 0.0:
            raise ValueError("control_step_s and final_window_s must be positive")

        window_samples = final_window_s / control_step_s
        self.final_window_samples = int(round(window_samples))
        if not math.isclose(
            window_samples, self.final_window_samples, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise ValueError("final_window_s must be an integer number of control steps")

        self.num_envs = int(num_envs)
        self.max_episode_samples = int(max_episode_samples)
        self._env_indices = torch.arange(self.num_envs, device=device)
        self._samples = torch.empty(
            (self.num_envs, self.max_episode_samples), device=device
        )
        self._sample_counts = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self._overflow = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self.episode_rms_m = torch.zeros(self.num_envs, device=device)
        self.episode_p95_m = torch.zeros(self.num_envs, device=device)
        self.episode_final_60s_rms_m = torch.zeros(self.num_envs, device=device)
        self.episode_final_60s_p95_m = torch.zeros(self.num_envs, device=device)

    def record(self, horizontal_error_m: torch.Tensor) -> None:
        """Record one horizontal-error sample for every environment."""

        if tuple(horizontal_error_m.shape) != (self.num_envs,):
            raise ValueError(
                "horizontal_error_m must have shape "
                f"({self.num_envs},), got {tuple(horizontal_error_m.shape)}"
            )
        writable = self._sample_counts < self.max_episode_samples
        rows = self._env_indices[writable]
        self._samples[rows, self._sample_counts[writable]] = horizontal_error_m[
            writable
        ]
        self._sample_counts.add_(writable)
        self._overflow.logical_or_(~writable)

    def latch(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Publish summaries for completed ``env_ids`` without clearing them."""

        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self._sample_counts.device
        )
        for env_id in ids.tolist():
            if bool(self._overflow[env_id]):
                raise RuntimeError(
                    f"environment {env_id} exceeded max_episode_samples="
                    f"{self.max_episode_samples}"
                )
            count = int(self._sample_counts[env_id])
            if count == 0:
                nan = torch.tensor(
                    torch.nan,
                    dtype=self._samples.dtype,
                    device=self._samples.device,
                )
                self.episode_rms_m[env_id] = nan
                self.episode_p95_m[env_id] = nan
                self.episode_final_60s_rms_m[env_id] = nan
                self.episode_final_60s_p95_m[env_id] = nan
                continue

            samples = self._samples[env_id, :count]
            final_samples = samples[-self.final_window_samples :]
            self.episode_rms_m[env_id] = samples.square().mean().sqrt()
            self.episode_p95_m[env_id] = torch.quantile(samples, 0.95)
            self.episode_final_60s_rms_m[env_id] = (
                final_samples.square().mean().sqrt()
            )
            self.episode_final_60s_p95_m[env_id] = torch.quantile(
                final_samples, 0.95
            )

    def reset(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Start fresh running statistics for only the selected environments."""

        ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self._sample_counts.device
        )
        self._sample_counts[ids] = 0
        self._overflow[ids] = False
