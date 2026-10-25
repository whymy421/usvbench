"""Condition adapter for the unchanged physical-unit observation degrader.

The env declares channel groups from the resolved perception config, then
calls ``apply`` exactly once per control step per group, BEFORE normalization
and feature construction. Reward, termination and privileged metrics continue
to use ground truth. This adapter intentionally retains legacy hold-last
dropout and clean-first-frame semantics; changing them requires a new model
version and new certificates, not silently changing a preset's meaning.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


try:
    from ..obs_degradation import ObsChannelGroup, build_obs_degrader
except ImportError:
    # CPU tests import by path and must not execute tasks/__init__ (Isaac Lab).
    _name = "_usvbench_conditions_obs_degradation"
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            _name, Path(__file__).resolve().parents[1] / "obs_degradation.py"
        )
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _module
        _spec.loader.exec_module(_module)
    ObsChannelGroup = sys.modules[_name].ObsChannelGroup
    build_obs_degrader = sys.modules[_name].build_obs_degrader


class PerceptionCondition:
    """Per-env, per-episode, per-group deterministic sensor degradation.

    ``groups`` maps measurement names to ObsChannelGroup, with sigma tuples
    assembled by the env from cfg.noise_{pos_m,vel_mps,range_m} and
    cfg.bias_{pos_m,vel_mps,range_m}. This preserves each family's physical
    channel semantics. cfg.delay_steps and cfg.dropout_p are common temporal
    axes; cfg.enabled is the strict identity gate.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        device: str | torch.device,
        seed: int,
        groups: dict[str, ObsChannelGroup],
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg
        self._groups = groups
        self._degrader = None
        if cfg.enabled:
            self._degrader = build_obs_degrader(
                num_envs=num_envs,
                device=device,
                base_seed=seed,
                groups=groups,
                delay_steps=cfg.delay_steps,
                dropout_p=cfg.dropout_p,
            )

    def wants(self, name: str) -> bool:
        """Whether the named group actually changes policy measurements."""
        if name not in self._groups:
            raise KeyError(name)
        return self._degrader is not None and self._degrader.wants(name)

    def apply(self, name: str, measurement: torch.Tensor) -> torch.Tensor:
        """Return an unchanged object when off; never mutate the input."""
        if name not in self._groups:
            raise KeyError(name)
        if self._degrader is None:
            return measurement
        return self._degrader.apply(name, measurement)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Advance only reset environments' independent episode streams."""
        if self._degrader is not None:
            self._degrader.reset(env_ids)

    def episode_parameters(self, env_id: int) -> dict:
        """Snapshot realized bias; noise/dropout replay uses seed + frame count."""
        if self._degrader is None:
            return {}
        result = {}
        for name, state in self._degrader._groups.items():
            pending = bool(state.needs_init[env_id])
            result[name] = {
                "measurement_initialized": not pending,
                "bias": None if pending else state.bias[env_id].detach().cpu().tolist(),
                "frames": 0 if pending else int(state.frames[env_id]),
                "group_id": state.group_id,
            }
        return result


__all__ = ["PerceptionCondition", "ObsChannelGroup"]
