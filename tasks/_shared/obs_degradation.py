# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval-time observation-channel degradation (OOD stress hooks).

Injected inside an env's ``_get_observations`` on the IDEAL MEASUREMENT
quantities -- goal vectors in meters, body-frame velocities in m/s and rad/s,
ray returns in meters -- BEFORE any feature construction or normalization.
Reward, termination, and success metrics keep reading the clean simulator
state, so a degraded run is judged by the same ground-truth ledger as a clean
one; only the policy's senses are stressed.

Four axes, applied in sensor-pipeline order:

1. bias    -- per-episode constant offset, drawn once per (env, episode,
              channel) as ``N(0, bias_sigma^2)``, fixed within the episode.
2. noise   -- zero-mean white Gaussian per step, physical units, per-channel
              sigma.
3. delay   -- ring buffer of ``delay_steps`` control steps, cleared on reset.
              Until the pipeline has produced its first delayed sample the
              output is the CLEAN initial measurement captured at reset.
4. dropout -- per-step Bernoulli whole-frame loss with sample-and-hold of the
              last valid (post-delay) frame. First-frame rule: the hold is
              seeded with the CLEAN initial measurement, so a dropout on the
              very first step replays the clean initial frame, never zeros.

An optional physical clamp (e.g. ray returns to [0, max_range]) is applied to
the final output only.

RNG: one independent ``torch.Generator`` per (env, group), reseeded from
``mix(base_seed, env_index, episode_index, group_id)`` on the first call after
that env resets. It never touches the env's layout RNG, numpy, or torch's
global generator, so degradation composes with the paired layout streams of
``scripts/eval_v6_frozen.py`` -- two doses at the same ``--eval-seed`` see the
identical episode/layout sequence.

Zero-config contract: ``build_obs_degrader`` returns ``None`` when every knob
is zero (the env then skips its hooks entirely -- byte-identical frozen
behavior), and ``apply`` on a group with nothing to do returns the input
tensor OBJECT unchanged (identity, not a copy).

Draw-order contract per (env, group) stream: ``bias`` (dim values, only if any
bias sigma is nonzero) at episode init, then per step ``noise`` (dim values,
only if any noise sigma is nonzero) followed by ``dropout`` (one value, only
if dropout_p > 0). Group ids are the sorted group-name order, so streams are
independent of the call order within a step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# SplitMix64-flavoured odd constants; the mask keeps the seed in the positive
# int64 range Generator.manual_seed accepts.
_MIX_A = 0x9E3779B97F4A7C15
_MIX_B = 0xBF58476D1CE4E5B9
_MIX_C = 0x94D049BB133111EB
_MIX_D = 0x2545F4914F6CDD1D
_MASK63 = 0x7FFFFFFFFFFFFFFF


def mix_seed(base_seed: int, env_index: int, episode_index: int, group_id: int) -> int:
    """Deterministic per-(env, episode, group) stream seed."""
    return (
        base_seed * _MIX_A
        + env_index * _MIX_B
        + episode_index * _MIX_C
        + (group_id + 1) * _MIX_D
    ) & _MASK63


@dataclass(frozen=True)
class ObsChannelGroup:
    """One measurement group: per-channel sigmas in PHYSICAL units."""

    dim: int
    noise_sigma: tuple[float, ...]
    bias_sigma: tuple[float, ...]
    clamp_min: float | None = None
    clamp_max: float | None = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("ObsChannelGroup.dim must be positive")
        for name, sigmas in (("noise_sigma", self.noise_sigma),
                             ("bias_sigma", self.bias_sigma)):
            if len(sigmas) != self.dim:
                raise ValueError(f"{name} must have exactly dim={self.dim} entries")
            if any(s < 0.0 for s in sigmas):
                raise ValueError(f"{name} entries must be non-negative")


class _GroupState:
    def __init__(
        self,
        spec: ObsChannelGroup,
        group_id: int,
        num_envs: int,
        device: torch.device,
        delay_steps: int,
    ) -> None:
        self.spec = spec
        self.group_id = group_id
        self.noise_sigma = torch.tensor(spec.noise_sigma, device=device)
        self.bias_sigma = torch.tensor(spec.bias_sigma, device=device)
        self.any_noise = any(s > 0.0 for s in spec.noise_sigma)
        self.any_bias = any(s > 0.0 for s in spec.bias_sigma)
        self.generators = []
        for _ in range(num_envs):
            generator = torch.Generator(device=device)
            generator.manual_seed(0)
            self.generators.append(generator)
        self.bias = torch.zeros((num_envs, spec.dim), device=device)
        self.init_frame = torch.zeros((num_envs, spec.dim), device=device)
        self.hold = torch.zeros((num_envs, spec.dim), device=device)
        # Control steps since this env's episode start (== index of the frame
        # being produced right now, 0-based).
        self.frames = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.needs_init = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.pending_init = True
        self.ring = (
            torch.zeros((delay_steps + 1, num_envs, spec.dim), device=device)
            if delay_steps > 0
            else None
        )


class ObsDegrader:
    """Stateful per-group measurement corrupter. See the module docstring."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        base_seed: int,
        groups: dict[str, ObsChannelGroup],
        delay_steps: int = 0,
        dropout_p: float = 0.0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not groups:
            raise ValueError("at least one channel group is required")
        if int(delay_steps) != delay_steps or delay_steps < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        if not 0.0 <= dropout_p <= 1.0:
            raise ValueError("dropout_p must lie in [0, 1]")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.base_seed = int(base_seed)
        self.delay_steps = int(delay_steps)
        self.dropout_p = float(dropout_p)
        # Episode counters are shared by every group and live on the CPU: they
        # are only read one int at a time while reseeding generators.
        self._episode = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._env_range = torch.arange(self.num_envs, device=self.device)
        self._groups: dict[str, _GroupState] = {}
        for group_id, name in enumerate(sorted(groups)):
            self._groups[name] = _GroupState(
                groups[name], group_id, self.num_envs, self.device,
                self.delay_steps,
            )

    # ------------------------------------------------------------------ api --
    def wants(self, name: str) -> bool:
        """True when applying this group would change anything at all."""
        state = self._groups[name]
        return bool(
            state.any_noise
            or state.any_bias
            or self.delay_steps > 0
            or self.dropout_p > 0.0
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        """Mark envs as freshly reset: new episode index, buffers re-seeded.

        The actual re-seeding, bias re-draw, and clean-initial-frame capture
        happen lazily on the next ``apply`` call, which is the first time the
        new episode's measurement exists.
        """
        if torch.is_tensor(env_ids):
            ids = env_ids.detach().cpu().tolist()
        else:
            ids = [int(i) for i in env_ids]
        if not ids:
            return
        self._episode[ids] += 1
        device_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        for state in self._groups.values():
            state.needs_init[device_ids] = True
            state.pending_init = True

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Degrade one (num_envs, dim) measurement tensor for this step.

        Must be called exactly ONCE per control step per group: delay and
        dropout are stateful. The input is never mutated in place.
        """
        state = self._groups[name]
        if not self.wants(name):
            return x
        if x.shape != (self.num_envs, state.spec.dim):
            raise ValueError(
                f"group {name!r} expects shape "
                f"{(self.num_envs, state.spec.dim)}, got {tuple(x.shape)}"
            )

        if state.pending_init:
            init_ids = torch.nonzero(state.needs_init).flatten()
            for i in init_ids.tolist():
                seed = mix_seed(
                    self.base_seed, i, int(self._episode[i]), state.group_id
                )
                state.generators[i].manual_seed(seed)
                if state.any_bias:
                    state.bias[i] = (
                        torch.randn(
                            state.spec.dim,
                            generator=state.generators[i],
                            device=self.device,
                        )
                        * state.bias_sigma
                    )
            # First-frame rule: the CLEAN measurement seeds the delay ring's
            # warm-up output and the dropout hold.
            state.init_frame[init_ids] = x[init_ids]
            state.hold[init_ids] = x[init_ids]
            state.frames[init_ids] = 0
            state.needs_init[init_ids] = False
            state.pending_init = False

        corrupted = x
        if state.any_bias:
            corrupted = corrupted + state.bias
        if state.any_noise:
            noise = torch.stack(
                [
                    torch.randn(
                        state.spec.dim, generator=generator, device=self.device
                    )
                    for generator in state.generators
                ]
            )
            corrupted = corrupted + noise * state.noise_sigma

        if self.delay_steps > 0:
            capacity = self.delay_steps + 1
            write_slot = state.frames % capacity
            state.ring[write_slot, self._env_range] = corrupted
            read_slot = (state.frames - self.delay_steps) % capacity
            warmed_up = (state.frames >= self.delay_steps).unsqueeze(-1)
            delayed = torch.where(
                warmed_up,
                state.ring[read_slot, self._env_range],
                state.init_frame,
            )
        else:
            delayed = corrupted

        if self.dropout_p > 0.0:
            uniforms = torch.stack(
                [
                    torch.rand(1, generator=generator, device=self.device)
                    for generator in state.generators
                ]
            ).reshape(self.num_envs, 1)
            dropped = uniforms < self.dropout_p
            output = torch.where(dropped, state.hold, delayed)
            state.hold = output
        else:
            output = delayed

        state.frames = state.frames + 1
        if state.spec.clamp_min is not None or state.spec.clamp_max is not None:
            output = output.clamp(
                min=state.spec.clamp_min, max=state.spec.clamp_max
            )
        return output


def build_obs_degrader(
    num_envs: int,
    device: torch.device | str,
    base_seed: int,
    groups: dict[str, ObsChannelGroup],
    delay_steps: int = 0,
    dropout_p: float = 0.0,
) -> ObsDegrader | None:
    """Return an ``ObsDegrader``, or ``None`` when every knob is zero.

    ``None`` is the frozen-behavior contract: the env keeps no degrader object
    and its guarded hooks never fire, so the certified observation path is
    byte-identical.
    """
    active = delay_steps > 0 or dropout_p > 0.0 or any(
        any(s > 0.0 for s in spec.noise_sigma)
        or any(s > 0.0 for s in spec.bias_sigma)
        for spec in groups.values()
    )
    if not active:
        return None
    return ObsDegrader(
        num_envs=num_envs,
        device=device,
        base_seed=base_seed,
        groups=groups,
        delay_steps=delay_steps,
        dropout_p=dropout_p,
    )


__all__ = [
    "ObsChannelGroup",
    "ObsDegrader",
    "build_obs_degrader",
    "mix_seed",
]
