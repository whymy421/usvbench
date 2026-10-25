"""Condition-layer bridge to the shared SeaState (C2 physics-version aware).

Exactly ONE calm-water increment subtraction (D1, J-base):
* physics v2: ``SeaState(..., physics_version=2)`` with
  ``calm_water_increment=True`` subtracts the zero-orbital drag itself
  (C2, sea_state.py ``forces``); this adapter adds nothing.
* physics v1 (C4 prototype path, legacy cfgs only): the frozen v1 SeaState
  keeps its drag floor, so this adapter subtracts it, as C4 did.

Time comes from the layer's single per-env integer physics clock (C2
``WaveClock``), evaluated before it advances once per physics substep and
reset only for reset environments. Enabled Hs=0 bypasses the sea object and
returns exact zeros. The separate wave redesign (task D9) is outstanding.
"""
from dataclasses import asdict
import torch
from ..sea_state import SeaState, SeaStateCfg


class WaveCondition:
    def __init__(self, cfg, num_envs, device, physics_version=1):
        self.cfg, self.sea = cfg, None
        self.physics_version = int(physics_version)
        if self.physics_version not in (1, 2):
            raise ValueError("physics_version must be 1 or 2")
        if cfg.enabled and max(cfg.hs_range) > 0:
            kwargs = asdict(cfg)
            kwargs["enable"] = kwargs.pop("enabled")
            # v2: SeaState owns the only calm-increment subtraction. v1
            # SeaState ignores this flag and keeps its legacy floor.
            kwargs["calm_water_increment"] = True
            # Legacy constructor samples once from global RNG; isolate that
            # throwaway CPU initialization, then reset from the layer stream.
            with torch.random.fork_rng(devices=[]):
                self.sea = SeaState(SeaStateCfg(**kwargs), num_envs, torch.device("cpu"),
                                    physics_version=self.physics_version)
            self.sea.device = torch.device(device)
            for name, value in vars(self.sea).copy().items():
                if isinstance(value, torch.Tensor):
                    setattr(self.sea, name, value.to(device))

    def reset(self, env_ids, scenario):
        if self.sea is not None:
            self.sea.resample(env_ids, scenario)

    def wrench(self, position_xy, velocity_xy, t):
        """World (force, torque) at per-env time ``t`` ([N] or [N,1])."""
        if self.sea is None:
            force = position_xy.new_zeros((len(position_xy), 3))
            return force, torch.zeros_like(force)
        force, torque = self.sea.forces(position_xy, velocity_xy, t)
        if self.sea.physics_version == 1:
            calm = -self.cfg.orbital_drag_coeff * velocity_xy * torch.norm(velocity_xy, dim=-1, keepdim=True)
            force[:, :2] -= calm
        return force, torque

    def episode_parameters(self, env_id):
        if self.sea is None:
            return {}
        parameters = {name: getattr(self.sea, name)[env_id].detach().cpu().tolist()
                      for name in ("hs", "tp", "gamma", "mean_direction", "amplitude", "phase", "direction")}
        grid = self.sea.freqs if self.physics_version == 1 else self.sea.freqs[env_id]
        parameters.update(
            physics_version=self.physics_version,
            applied_n_components=int(grid.numel()),
            frequencies_hz=grid.detach().cpu().tolist(),
            calm_increment_owner="SeaState" if self.physics_version == 2 else "WaveCondition",
        )
        return parameters
