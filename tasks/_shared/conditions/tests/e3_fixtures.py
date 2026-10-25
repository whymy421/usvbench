"""Shared CPU fixtures for the E3 hydro_v2 condition-layer tests.

Loads tasks/_shared as a bare namespace package (never tasks/__init__.py,
never Isaac Lab). Provides a recording FakeRuntime with the HydroV2Runtime
interface, vessel profiles built only from the D5 registry adapters, and a
real HydroV2Runtime factory.

Real runtime and pre-E2 J-base: D3's hydro_v2.hydro_wrench calls
``implicit_damping_wrench(..., water_velocity)``; J-base's physics_v2 does
not have that argument yet (E2 adds it with D3's arithmetic). When the
in-tree function lacks it, ``real_runtime`` binds, INSIDE the loaded
hydro_v2 module only, a test-local copy of D3's reference arithmetic
(D3 repo tasks/_shared/physics_v2.py:23-55). On the merged tree the in-tree
function is used unchanged. ``RUNTIME_KERNEL`` records which one ran.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

sys.dont_write_bytecode = True
import torch

SHARED = Path(__file__).resolve().parents[2]
REPO = SHARED.parents[1]
PACKAGE = "_e3_shared"
if PACKAGE not in sys.modules:
    _parent = ModuleType(PACKAGE)
    _parent.__path__ = [str(SHARED)]
    sys.modules[PACKAGE] = _parent
    _spec = importlib.util.spec_from_file_location(
        PACKAGE + ".conditions", SHARED / "conditions" / "__init__.py",
        submodule_search_locations=[str(SHARED / "conditions")])
    _api = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _api
    _spec.loader.exec_module(_api)

api = sys.modules[PACKAGE + ".conditions"]


def module(name):
    return importlib.import_module(PACKAGE + "." + name)


config_v3 = module("conditions.config_v3")
presets_v3 = module("conditions.presets_v3")
strata = module("conditions.strata")
keyed_rng = module("conditions.keyed_rng")
perception_v3 = module("conditions.perception_v3")
observation_groups = module("conditions.observation_groups")
hydro_v2_layer = module("conditions.hydro_v2_layer")
wind_backend = module("wind")
sea_state = module("sea_state")
scenario_rng = module("scenario_rng")
physics_v2 = module("physics_v2")
vehicles_v2 = module("vehicles_v2")
hydro_v2 = module("hydro_v2")
hydro_v2_runtime = module("hydro_v2_runtime")
thruster_mixer = module("thruster_mixer")

DT = 1.0 / 120.0
DECIMATION = 2
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def profile(name="blueboat_v2"):
    spec = vehicles_v2.get_vehicle_v2(name)
    return SimpleNamespace(
        name=name + "_e3_test_profile", status="test (D5 registry adapters only)",
        hydro=hydro_v2.HydroSpec(**vehicles_v2.to_d3_hydro_kwargs(spec)),
        thrusters=thruster_mixer.ThrusterSpec(**vehicles_v2.to_d3_thruster_kwargs(spec)),
        com_offset_actor_m=(0.0, 0.0, 0.0), actor_to_hull=IDENTITY)


class FakeRuntime:
    """HydroV2Runtime interface that records every call (no physics)."""

    def __init__(self, prof, num_envs, dt=DT, dtype=torch.float32, saturation="radial"):
        self.profile, self.num_envs, self.dt, self.dtype = prof, int(num_envs), float(dt), dtype
        self.device = torch.device("cpu")
        self.saturation = saturation
        self.ticks = torch.zeros(self.num_envs, dtype=torch.int64)
        self.calls = []
        self.degradation = thruster_mixer.ThrusterDegradation()
        self.resistance_scale = 1.0

    def time(self):
        return self.ticks.to(self.dtype) * self.dt

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long)
        self.ticks[ids] = 0
        self.calls.append(("reset", ids.tolist()))

    def configure(self, degradation=None, resistance_scale=None):
        if degradation is not None:
            self.degradation = degradation
        if resistance_scale is not None:
            self.resistance_scale = resistance_scale
        self.calls.append(("configure",
                           None if degradation is None else tuple(
                               torch.as_tensor(v).clone() for v in (degradation.thrust_cap,
                                                                    degradation.motor_gain,
                                                                    degradation.additional_lag_s)),
                           None if resistance_scale is None else torch.as_tensor(resistance_scale).clone()))

    def step(self, actions, com_position_w, com_velocity_w, actor_quat_wxyz, env_origins_w,
             current_velocity_w=None, waves=None, mean_water_level_local=0.0,
             wind_force_w=None, wind_torque_w=None):
        t = self.time().clone()
        sample = None
        if waves is not None:
            yaw = torch.zeros(self.num_envs)
            sample = waves.sample(com_position_w[:, :2].to(self.dtype), env_origins_w[:, :2].to(self.dtype),
                                  t, yaw, self.profile.hydro, footprint=True)
        self.calls.append(("step", t, None if current_velocity_w is None else current_velocity_w.clone(),
                           sample, None if wind_force_w is None else wind_force_w.clone(),
                           None if wind_torque_w is None else wind_torque_w.clone()))
        self.ticks += 1


def _reference_implicit_damping_wrench(velocity, mass_matrix, damping_matrix, dt,
                                       other_wrench=None, water_velocity=None):
    """Test-local copy of D3 physics_v2.implicit_damping_wrench (D3 :23-55)."""
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("physics dt must be positive and finite")
    rhs = (mass_matrix @ velocity.unsqueeze(-1)).squeeze(-1)
    if other_wrench is not None:
        rhs = rhs + dt * other_wrench
    if water_velocity is not None:
        rhs = rhs + dt * (damping_matrix @ water_velocity.unsqueeze(-1)).squeeze(-1)
    next_velocity = torch.linalg.solve(mass_matrix + dt * damping_matrix, rhs.unsqueeze(-1)).squeeze(-1)
    if water_velocity is not None:
        next_velocity = next_velocity - water_velocity
    return -(damping_matrix @ next_velocity.unsqueeze(-1)).squeeze(-1)


def _runtime_kernel():
    parameters = inspect.signature(physics_v2.implicit_damping_wrench).parameters
    if "water_velocity" in parameters:
        return "in_tree_physics_v2"
    hydro_v2.implicit_damping_wrench = _reference_implicit_damping_wrench
    return "test_local_D3_reference (pre-E2 J-base physics_v2 lacks water_velocity)"


RUNTIME_KERNEL = _runtime_kernel()


def real_runtime(prof, num_envs, dt=DT, dtype=torch.float32):
    return hydro_v2_runtime.HydroV2Runtime(prof, num_envs, "cpu", dt, dtype=dtype,
                                           engine_gravity=(0.0, 0.0, -prof.hydro.gravity))


def env_cfg(family="station_keeping", **kwargs):
    physics = SimpleNamespace(physics_version=2, hydrodynamics_model="hydro_v2", enable_current=False)
    values = dict(conditions=None, condition_presets=(), seed=42, vehicle="blueboat_v2",
                  sim=SimpleNamespace(dt=DT), decimation=DECIMATION, journal_mode=True,
                  underwater_physics_cfg=physics)
    if family == "station_keeping":
        values.update(obs_kinematic=True)
    elif family == "hazard_nav":
        values.update(obs_v3=True, ray_count=36, ray_max_range_m=30.0)
    values.update(kwargs)
    return SimpleNamespace(**values)


def build(presets=(), overrides=None, *, family="station_keeping", n=4, runtime="fake", seed=42,
          vessel="blueboat_v2", **cfg_kwargs):
    cfg = env_cfg(family, condition_presets=tuple(presets), conditions=overrides, seed=seed,
                  vehicle=vessel, **cfg_kwargs)
    layer = api.build_conditions(cfg, n, "cpu", family=family)
    prof = profile(vessel)
    rt = FakeRuntime(prof, n) if runtime == "fake" else real_runtime(prof, n)
    layer.bind_runtime(rt)
    return layer, rt


def task_episode_ids(layer, ids):
    """F0 / D7: the task scenario counter of the reset rows (mandatory at reset)."""
    return layer.scenario.episode_indices()[torch.as_tensor(ids, dtype=torch.long)] + 1


def reset(layer, rt, ids=None, axis=None, **kwargs):
    ids = torch.arange(layer.num_envs) if ids is None else torch.as_tensor(ids, dtype=torch.long)
    rt.reset(ids)
    axis = torch.zeros(ids.numel(), dtype=torch.float64) if axis is None else axis
    kwargs.setdefault("task_episode_ids", task_episode_ids(layer, ids))   # F0 / D7
    layer.reset_runtime(ids, reference_axis_rad=axis, **kwargs)
    layer.configure_runtime(rt, ids)
    return ids


def quaternion_yaw(yaw):
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    return torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw),
                        torch.sin(yaw / 2)), -1)
