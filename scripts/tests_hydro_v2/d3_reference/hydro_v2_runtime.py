"""Per-environment hydro_v2 runtime: clock, motor states and one physics tick.

Pure Python/torch (no Isaac import). An environment owns one HydroV2Runtime
and calls ``step`` once per PHYSICS substep (inside ``_apply_action``) and
``reset(env_ids)`` for the environments it resets. The runtime implements
INTEGRATION_SPEC.md section 2:

  * one integer physics-tick clock per env (evaluate sources at tick t, then
    advance once), shared by waves, current schedules and motor lag;
  * the per-motor mixer (thruster_mixer.step_thrusters) with healthy or
    degraded motors -- the motor model is active even with every suite off;
  * wave kinematics sampled at env-LOCAL COM coordinates;
  * one water-relative hull wrench (hydro_v2.hydro_wrench) with the motor
    and wind wrenches as known forcing in the implicit predictor;
  * conversion from the canonical hull frame to the actor frame.

Environmental inputs (current velocity, waves, wind) are never exposed as
observations by this module. Gravity stays engine-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:
    from .hydro_v2 import WaveSample, hydro_wrench, required_high_frequency_hz
    from .physics_v2 import rotation_matrix
    from .thruster_mixer import ThrusterDegradation, ThrusterState, step_thrusters
except ImportError:  # Path-loaded CPU tests; never import the tasks package.
    from hydro_v2 import WaveSample, hydro_wrench, required_high_frequency_hz
    from physics_v2 import rotation_matrix
    from thruster_mixer import ThrusterDegradation, ThrusterState, step_thrusters


HYDRODYNAMICS_MODELS = ("legacy", "hydro_v2")


def hydrodynamics_model(physics_cfg) -> str:
    """Validated selector; an absent field means the legacy/C2 code path."""
    value = getattr(physics_cfg, "hydrodynamics_model", "legacy")
    if value not in HYDRODYNAMICS_MODELS:
        raise ValueError("hydrodynamics_model must be one of %r" % (HYDRODYNAMICS_MODELS,))
    return value


def check_wave_band(profile, sea_cfg) -> float:
    """Fail closed if a C2 SeaState v2 band is too narrow for this hull.

    hydro_v2 filters waves over the hull footprint; the band must reach
    hydro_v2.required_high_frequency_hz (>= 4 fp and the filter edge) at the
    LONGEST peak period of the dose box (the ratio grows with Tp). Returns the
    required v2_f_max_over_fp. See SPECTRAL_REPORT.txt / DESIGN_NOTES.md 4.
    """
    tp_max = float(max(sea_cfg.tp_range))
    if not math.isfinite(tp_max) or tp_max <= 0:
        raise ValueError("sea state tp_range must be positive and finite")
    needed = required_high_frequency_hz(profile.hydro, 1.0 / tp_max) * tp_max
    have = float(getattr(sea_cfg, "v2_f_max_over_fp", 0.0))
    if have < needed - 1e-9:
        raise ValueError("SeaState v2_f_max_over_fp=%g is below the hydro_v2 requirement %.4g for %s "
                         "at Tp=%g s" % (have, needed, profile.name, tp_max))
    return needed


@dataclass(frozen=True)
class RuntimeStep:
    force_actor: torch.Tensor
    torque_actor: torch.Tensor
    force_hull: torch.Tensor
    torque_hull: torch.Tensor
    hydro: object
    motors: object
    waves: WaveSample
    time_s: torch.Tensor


class HydroV2Runtime:
    """State owner for one vessel profile and one batch of environments."""

    def __init__(self, profile, num_envs, device, dt, saturation="radial",
                 dtype=torch.float32, gyroscopic_predictor=False, engine_gravity=None):
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("physics dt must be positive and finite")
        if saturation not in ("radial", "clip"):
            raise ValueError("saturation must be 'radial' or 'clip'")
        if engine_gravity is not None:
            # The implicit predictor and hydrostatics use profile.hydro.gravity;
            # the engine integrates its own gravity. They must agree (world -z).
            g = [float(x) for x in engine_gravity]
            if (len(g) != 3 or abs(g[0]) > 1e-9 or abs(g[1]) > 1e-9
                    or not math.isclose(-g[2], profile.hydro.gravity, rel_tol=1e-6)):
                raise ValueError("engine gravity %r does not match profile gravity -%r z"
                                 % (tuple(g), profile.hydro.gravity))
        basis = torch.as_tensor(profile.actor_to_hull, dtype=torch.float64)
        if (basis.shape != (3, 3)
                or not torch.allclose(basis.T @ basis, torch.eye(3, dtype=torch.float64), atol=1e-12)
                or not math.isclose(float(torch.linalg.det(basis)), 1.0, abs_tol=1e-12)):
            raise ValueError("actor_to_hull must be a proper orthonormal 3x3 basis")
        self.profile = profile
        self.num_envs = int(num_envs)
        self.device = device
        self.dt = float(dt)
        self.dtype = dtype
        self.saturation = saturation
        self.gyroscopic_predictor = bool(gyroscopic_predictor)
        self.basis = basis.to(device=device, dtype=dtype)
        self.ticks = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        zeros = torch.zeros((self.num_envs, 2), dtype=dtype, device=device)
        self.thruster_state = ThrusterState(zeros, zeros.clone())
        self.degradation = ThrusterDegradation()
        self.resistance_scale = 1.0

    def time(self) -> torch.Tensor:
        return self.ticks.to(self.dtype) * self.dt

    def configure(self, degradation=None, resistance_scale=None) -> None:
        """Latch actuator-suite faults and the resistance multiplier (per episode).

        degradation: ThrusterDegradation (per-env tensors [N,1] or [N,2]);
        resistance_scale: scalar or [N] tensor, finite and nonnegative. Values
        are validated here so that a bad dose fails at reset, not mid-step.
        """
        if degradation is not None:
            if not isinstance(degradation, ThrusterDegradation):
                raise ValueError("degradation must be a ThrusterDegradation")
            probe = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
            step_thrusters(probe, ThrusterState(probe, probe.clone()),
                           self.profile.thrusters, self.dt, degradation)
            self.degradation = degradation
        if resistance_scale is not None:
            scale = torch.as_tensor(resistance_scale, dtype=self.dtype, device=self.device)
            if (scale.numel() not in (1, self.num_envs) or not bool(torch.isfinite(scale).all())
                    or bool((scale < 0).any())):
                raise ValueError("resistance_scale must be finite, nonnegative, scalar or [N]")
            self.resistance_scale = scale.reshape(-1) if scale.numel() > 1 else scale.reshape(())

    def reset(self, env_ids=None) -> None:
        """Zero the clock and BOTH motor-lag stages of the selected envs only.

        env_ids=None resets every environment (Isaac Lab's all-envs convention).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.ticks[ids] = 0
        nominal = self.thruster_state.nominal.clone()
        extra = self.thruster_state.extra.clone()
        nominal[ids] = 0.0
        extra[ids] = 0.0
        self.thruster_state = ThrusterState(nominal, extra)

    def step(self, actions, com_position_w, com_velocity_w, actor_quat_wxyz, env_origins_w,
             current_velocity_w=None, waves=None, mean_water_level_local=0.0,
             wind_force_w=None, wind_torque_w=None) -> RuntimeStep:
        """Evaluate one physics tick, then advance the clock exactly once.

        com_velocity_w is [N,6] (linear COM velocity, world angular velocity).
        waves is a SpectralWaves (or None for calm water). Returns actor-frame
        body force/torque to submit through the engine's body-frame wrench API.
        """
        n = self.num_envs
        spec = self.profile.hydro
        if com_velocity_w.shape != (n, 6):
            raise ValueError("com_velocity_w must be [N,6] (linear, angular) in world axes")
        rotation_actor = rotation_matrix(actor_quat_wxyz.to(self.dtype))
        rotation = rotation_actor @ self.basis
        position = com_position_w.to(self.dtype)
        linear = com_velocity_w[:, :3].to(self.dtype)
        angular = com_velocity_w[:, 3:].to(self.dtype)
        origins = env_origins_w.to(self.dtype)
        t = self.time()
        if waves is None:
            sample = WaveSample.calm(position)
        else:
            yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
            sample = waves.sample(position[:, :2], origins[:, :2], t, yaw, spec, footprint=True)
            sample = WaveSample(*(None if value is None else value.to(self.dtype)
                                  for value in (sample.elevation, sample.slope,
                                                sample.orbital_velocity_world,
                                                sample.surface_angular_velocity_world)))
        current = (torch.zeros_like(linear) if current_velocity_w is None
                   else current_velocity_w.to(self.dtype))
        motors = step_thrusters(actions.to(self.dtype), self.thruster_state,
                                self.profile.thrusters, self.dt, self.degradation,
                                self.saturation)
        self.thruster_state = motors.state
        inverse = rotation.transpose(-1, -2)
        known_force = motors.body_wrench[:, :3]
        known_torque = motors.body_wrench[:, 3:]
        if wind_force_w is not None:
            known_force = known_force + (inverse @ wind_force_w.to(self.dtype).unsqueeze(-1)).squeeze(-1)
        if wind_torque_w is not None:
            known_torque = known_torque + (inverse @ wind_torque_w.to(self.dtype).unsqueeze(-1)).squeeze(-1)
        predictor_torque = known_torque
        if self.gyroscopic_predictor:
            inertia = torch.as_tensor(spec.inertia_kgm2, dtype=self.dtype, device=position.device)
            omega_hull = (inverse @ angular.unsqueeze(-1)).squeeze(-1)
            predictor_torque = known_torque - torch.cross(
                omega_hull, (inertia @ omega_hull.unsqueeze(-1)).squeeze(-1), dim=-1)
        hydro = hydro_wrench(spec, rotation, position, linear, angular, current, sample,
                             self.dt, known_force, predictor_torque, self.resistance_scale,
                             water_level_world=origins[:, 2] + mean_water_level_local)
        force_hull = known_force + hydro.force_body
        torque_hull = known_torque + hydro.torque_body
        force_actor = (self.basis @ force_hull.unsqueeze(-1)).squeeze(-1)
        torque_actor = (self.basis @ torque_hull.unsqueeze(-1)).squeeze(-1)
        self.ticks += 1
        return RuntimeStep(force_actor, torque_actor, force_hull, torque_hull,
                           hydro, motors, sample, t)
