"""Wind suite backend: the C3 ``Wind`` module behind the condition layer (D1).

Replaces C4's interface stub. Contract (all verified by CPU tests in
scripts/test_jbase_integration.py):

* World-frame wrench, returned through ``ConditionLayer.environmental_wrench``
  and added to the env's world accumulator BEFORE its single world-to-body
  rotation and BEFORE the condition ``dynamics`` mass/payload scaling.
* Advanced exactly once per physics substep (``step`` is called by the layer
  after the wrench is evaluated, like C3's reference adapter).
* Gust streams keyed by (eval seed, env index, scenario episode index,
  physics step) through C3's BLAKE2b protocol ``usvbench.wind.ou.v1``.
* Heading is the env's PHYSICAL bow axis (``quat_apply(quat, forward_vec)``);
  air-relative velocity uses the hull COM velocity (``root_com_vel_w``).
* Per-episode mean direction: ``fixed`` (world flow-to azimuth), ``uniform``
  (keyed draw on [0, 2*pi) relative to the task reference axis) or
  ``balanced`` (a discrete set of offsets relative to the task reference
  axis, assigned deterministically so every direction appears equally often
  over a complete grid; see ``balanced_slot``).
* Dose axis: ``reference="u10"`` converts the requested 10 m mean wind to the
  vessel reference height with a neutral logarithmic marine profile,
  U(z) = U10 * ln(z / z0) / ln(10 / z0), roughness ``roughness_length_m``
  (DECLARED ASSUMPTION, owner decision pending); ``reference="vessel_height"``
  applies the requested speed directly. Both the requested and the applied
  vessel-height speed are stamped.

Disabled wind is an exact zero wrench and consumes no random draws. Wind is
never exposed to the controller (no observation channel is added).
Physical calibration (coefficients, superstructure, WAM-V) is UNVERIFIED.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch

try:
    from ..wind import VesselWindage, Wind, WindCfg
except ImportError:
    # CPU tests may load this package by path without a parent package.
    _name = "_usvbench_conditions_wind_backend"
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            _name, Path(__file__).resolve().parents[1] / "wind.py"
        )
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _module
        _spec.loader.exec_module(_module)
    VesselWindage = sys.modules[_name].VesselWindage
    Wind = sys.modules[_name].Wind
    WindCfg = sys.modules[_name].WindCfg

from .windage import WINDAGE_PROFILE_VERSION, get_profile, require_usable

WIND_IMPLEMENTATION = "c3_fossen_ou_v1"
WIND_RNG_PROTOCOL = "usvbench.wind.ou.v1"
U10_HEIGHT_M = 10.0
DIRECTION_GROUP = "condition_wind_direction"
REFERENCES = ("u10", "vessel_height")
DIRECTION_MODES = ("fixed", "uniform", "balanced")


def vessel_height_speed(requested_mps, reference, reference_height_m, roughness_length_m):
    """Mean wind at the vessel reference height for the declared dose axis."""
    requested = float(requested_mps)
    if reference == "vessel_height":
        return requested
    if reference != "u10":
        raise ValueError(f"wind reference must be one of {REFERENCES}")
    z = float(reference_height_m)
    z0 = float(roughness_length_m)
    if not (0.0 < z0 < z) or not z0 < U10_HEIGHT_M:
        raise ValueError("log profile needs 0 < roughness length < reference height and < 10 m")
    return requested * math.log(z / z0) / math.log(U10_HEIGHT_M / z0)


def balanced_slot(env_index, episode_index, eval_seed, count):
    """Deterministic balanced assignment of one of ``count`` directions.

    slot = (env + episode + seed) mod count. For every episode index the
    slots are a cyclic shift of the env indices, so when num_envs is a
    multiple of ``count`` each direction is used by exactly num_envs/count
    envs in every episode, hence also over any complete (env, ep) grid.
    Controllers evaluated with the same seed see identical assignments.
    """
    return (int(env_index) + int(episode_index) + int(eval_seed)) % int(count)


class WindCondition:
    def __init__(self, cfg, num_envs, device, *, eval_seed=0, vehicle="blueboat"):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.eval_seed = int(eval_seed)
        self.vehicle = vehicle
        self.backend = None
        self.profile = None
        self.applied_speed_mps = None
        self.reference_height_m = None
        # Per-episode bookkeeping for stamps (CPU, float64).
        self.reference_axis_rad = torch.zeros(self.num_envs, dtype=torch.float64)
        self.direction_world_rad = torch.zeros(self.num_envs, dtype=torch.float64)
        self.direction_relative_rad = torch.zeros(self.num_envs, dtype=torch.float64)
        self.reference_axis_given = torch.zeros(self.num_envs, dtype=torch.bool)
        self.episode_index = torch.full((self.num_envs,), -1, dtype=torch.long)
        if not cfg.enabled:
            return
        if cfg.implementation != WIND_IMPLEMENTATION:
            raise ValueError(f"wind implementation must be {WIND_IMPLEMENTATION!r}")
        if self.eval_seed < 0:
            raise ValueError("enabled wind needs a nonnegative evaluation seed")
        profile = require_usable(get_profile(cfg.windage_profile), vehicle)
        self.profile = profile
        self.reference_height_m = float(profile.centroid_height_m)
        self.applied_speed_mps = vessel_height_speed(
            cfg.speed_mps, cfg.reference, self.reference_height_m, cfg.roughness_length_m
        )
        backend_cfg = WindCfg(
            enable=True,
            mean_speed_mps=self.applied_speed_mps,
            reference_height_m=self.reference_height_m,
            direction_rad=float(cfg.direction_rad),
            direction_drift_rad_s=float(cfg.direction_drift_rad_s),
            gust_intensity=float(cfg.gust_intensity),
            gust_correlation_time_s=float(cfg.gust_correlation_time_s),
            air_density_kg_m3=float(cfg.air_density_kg_m3),
            windage=VesselWindage(
                frontal_area_m2=float(profile.frontal_area_m2),
                lateral_area_m2=float(profile.lateral_area_m2),
                length_m=float(profile.length_m),
                centroid_height_m=float(profile.centroid_height_m),
                longitudinal_cp_m=float(profile.longitudinal_cp_m),
                c_x0=float(profile.c_x0),
                c_y0=float(profile.c_y0),
                c_n0=float(profile.c_n0),
            ),
        )
        self.backend = Wind(backend_cfg, self.num_envs, self.device,
                            eval_seed=self.eval_seed, dtype=torch.float32)

    @property
    def enabled(self):
        return self.backend is not None

    def reset(self, env_ids, scenario, reference_axis_rad=None):
        """New episode for ``env_ids``; call AFTER the scenario counter advanced."""
        if self.backend is None:
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long).detach().cpu().flatten()
        if ids.numel() == 0:
            return
        episodes = scenario.episode_indices()[ids]
        if bool((episodes < 0).any()):
            raise RuntimeError("scenario must be reset before the wind episode starts")
        mode = self.cfg.direction_mode
        if reference_axis_rad is None:
            if mode != "fixed":
                raise ValueError(f"wind direction_mode {mode!r} needs the task reference axis")
            reference = torch.zeros(ids.numel(), dtype=torch.float64)
        else:
            reference = torch.as_tensor(reference_axis_rad).detach().cpu().to(torch.float64).flatten()
            if reference.numel() != ids.numel() or not bool(torch.isfinite(reference).all()):
                raise ValueError("reference_axis_rad must be finite, one value per reset env")
        if mode == "fixed":
            world = torch.full((ids.numel(),), float(self.cfg.direction_rad), dtype=torch.float64)
            relative = world - reference
        elif mode == "uniform":
            relative = scenario.uniform(DIRECTION_GROUP, ids, 0.0, 2.0 * math.pi)
            relative = relative.detach().cpu().to(torch.float64).flatten()
            world = reference + relative
        else:
            choices = torch.as_tensor(tuple(self.cfg.balanced_directions_rad), dtype=torch.float64)
            slots = torch.tensor([
                balanced_slot(i, e, self.eval_seed, choices.numel())
                for i, e in zip(ids.tolist(), episodes.tolist())
            ], dtype=torch.long)
            relative = choices[slots]
            world = reference + relative
        self.episode_index[ids] = episodes
        self.reference_axis_rad[ids] = reference
        self.reference_axis_given[ids] = reference_axis_rad is not None
        self.direction_world_rad[ids] = world
        self.direction_relative_rad[ids] = relative
        self.backend.reset(ids, episodes, direction_rad=world)

    def wrench(self, position_xy, velocity_xy, elapsed_s=None, *, heading_rad=None,
               orientation_wxyz=None, angular_velocity_w=None):
        """World [N,3] force and torque at the current wind state (read-only).

        ``elapsed_s`` is accepted for interface compatibility and ignored: the
        wind stream owns its keyed physics-step counter and drift clock.
        """
        force = velocity_xy.new_zeros((self.num_envs, 3))
        torque = torch.zeros_like(force)
        if self.backend is None:
            return force, torque
        if heading_rad is None:
            raise ValueError("enabled wind needs the physical bow heading (forward_vec)")
        dtype = self.backend.dtype
        velocity = velocity_xy[:, :2].to(device=self.device, dtype=dtype)
        heading = heading_rad.reshape(self.num_envs).to(device=self.device, dtype=dtype)
        wind_force, wind_torque = self.backend.forces(velocity, heading)
        return wind_force.to(force.dtype), wind_torque.to(force.dtype)

    def step(self, dt):
        """Advance every stream by one PHYSICS step (never per control step)."""
        if self.backend is not None:
            self.backend.step(float(dt))

    def header_parameters(self):
        """Static model description for the certificate context."""
        result = {
            "implementation": WIND_IMPLEMENTATION,
            "enabled": self.backend is not None,
            "windage_profile_version": WINDAGE_PROFILE_VERSION,
            "frames": ("direction = air flow-to azimuth, CCW from world +X; "
                       "relative offsets are measured from the task reference axis"),
            "exposed_to_controller": False,
        }
        if self.backend is None:
            return result
        result.update(
            rng_protocol=WIND_RNG_PROTOCOL,
            windage_profile=self.profile.as_dict(),
            requested_speed_mps=float(self.cfg.speed_mps),
            requested_reference=self.cfg.reference,
            applied_vessel_height_speed_mps=float(self.applied_speed_mps),
            vessel_reference_height_m=float(self.reference_height_m),
            profile_law=("neutral_log: U(z)=U10*ln(z/z0)/ln(10/z0)"
                         if self.cfg.reference == "u10" else "none (vessel-height dose)"),
            roughness_length_m=float(self.cfg.roughness_length_m),
            assumption=("u10 conversion uses a declared neutral log profile down to the "
                        "vessel reference height; validity this close to the surface is UNVERIFIED")
            if self.cfg.reference == "u10" else "dose applied directly at vessel height",
        )
        return result

    def episode_parameters(self, env_id):
        if self.backend is None:
            return {}
        i = int(env_id)
        return {
            "stream_key": {"eval_seed": self.eval_seed, "env_key": int(self.backend.env_keys[i]),
                           "episode_index": int(self.episode_index[i])},
            "physics_steps": int(self.backend._physics_steps[i]),
            "direction_mode": self.cfg.direction_mode,
            "direction_world_rad": float(self.direction_world_rad[i]),
            "direction_relative_rad": float(self.direction_relative_rad[i]),
            "reference_axis_rad": float(self.reference_axis_rad[i]),
            "reference_axis_supplied": bool(self.reference_axis_given[i]),
            "requested_speed_mps": float(self.cfg.speed_mps),
            "applied_vessel_height_speed_mps": float(self.applied_speed_mps),
        }


__all__ = ["WindCondition", "vessel_height_speed", "balanced_slot",
           "WIND_IMPLEMENTATION", "WIND_RNG_PROTOCOL", "DIRECTION_GROUP"]
