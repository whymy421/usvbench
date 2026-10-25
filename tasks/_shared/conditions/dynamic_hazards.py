"""Progress-trigger interface. Geometry/collision adapters are task owned.

Definitions transcribed from 04 section 3; proposals, not approved new tasks.
Adapters must reserve capacity and update sensing AND collision state before
claiming support. Station keeping has no voyage: its exposure/progress metric
needs an explicit scientific decision, not an invented elapsed-time proxy.
"""
from typing import Protocol
import torch


TASK_DEFINITIONS = {
    "reach": "appearing obstacle along the route: floating debris",
    "path_following": "moving obstacle crossing the route: crossing vessel",
    "station_keeping": "drifting obstacle approaching hold circle: nearby debris or boat",
    "docking": "moving obstacle near berth: harbour traffic",
    "path_hazard": "appearing pillar on route: sudden blockage",
    "scatter": "appearing pillar: sudden blockage",
    "iceberg": "appearing pillar: sudden blockage",
    "ring": "appearing or moving pillar near gap: temporary narrow-channel occupancy",
    "fortress": "appearing or moving pillar near gap: temporary narrow-channel occupancy",
    "forced_crossing": "appearing pillar near gate: temporary lock occupancy",
    "moving_basin": "intrinsic moving obstacles; do not add a duplicate hazard population",
    "harbour": "moving obstacle near exit gate or berth: harbour traffic",
}


class HazardAdapter(Protocol):
    def reset(self, env_ids, cfg, scenario): ...
    def activate(self, env_ids, cfg): ...
    def update(self, progress, cfg): ...


class DynamicHazardCondition:
    def __init__(self, cfg, num_envs, device, adapter=None):
        self.cfg, self.adapter = cfg, adapter
        self.triggered = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if cfg.enabled and adapter is None:
            raise NotImplementedError("dynamic hazards require a reviewed task geometry/collision adapter")

    def reset(self, env_ids, scenario):
        if self.cfg.enabled:
            self.triggered[env_ids] = False
            self.adapter.reset(env_ids, self.cfg, scenario)

    def runtime(self, progress=None):
        if not self.cfg.enabled:
            return
        if progress is None or progress.shape != self.triggered.shape:
            raise ValueError("enabled hazards need one task-defined voyage fraction per environment")
        if not torch.isfinite(progress).all() or ((progress < 0) | (progress > 1)).any():
            raise ValueError("progress must be finite and in [0, 1]")
        activate = (~self.triggered) & (progress >= self.cfg.trigger_progress)
        ids = activate.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.adapter.activate(ids, self.cfg)
            self.triggered[ids] = True
        self.adapter.update(progress, self.cfg)
