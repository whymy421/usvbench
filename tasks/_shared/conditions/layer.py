"""Four lifecycle hooks, with small helpers inside the physics hook.

No simulator imports. Environments own frame transforms, clean observations,
task geometry and truth-based reward/termination. Disabled factories return
None so existing arithmetic and random draw order are bypassed unchanged
(legacy cfgs). Journal cfgs (``journal_mode=True``) always get a layer, which
is an exact no-op while every suite is off.

D1 (J-base) reconciliation:
* ONE per-env physics clock: a C2 ``WaveClock`` (integer ticks), evaluated
  before it advances once per ``environmental_wrench`` call (= once per
  physics substep) and reset only for the reset envs. Waves read
  ``clock.time()`` exactly like the C2 env path; the current ramp reads the
  same ticks in float64 seconds. Journal cfgs reject the legacy sea state,
  so no second (env-owned) wave clock can exist.
* ONE calm-increment subtraction: see waves.py.
* ONE drag/mass source for v2 damping: ``hull_damping_scales``.
* Wind: C3 backend (wind.py), stepped once per physics substep.
"""
from dataclasses import asdict, replace
import math
import torch

from ..scenario_rng import ScenarioRNG, SCENARIO_PROTOCOL_VERSION
from ..obs_superset import SPEED_SCALE_MPS
from ..physics_v2 import WaveClock
from .config import resolve, validate
from .current import CurrentCondition
from .actuator import ActuatorCondition
from .perception import PerceptionCondition, ObsChannelGroup
from .waves import WaveCondition
from .wind import WindCondition
from .windage import profile_name_for_vehicle
from .dynamic_hazards import DynamicHazardCondition
from .stamp import stamp_conditions

WIRED_FAMILIES = ("station_keeping", "hazard_nav")


class ConditionLayer:
    def __init__(self, cfg, num_envs, device, *, groups=None, context=None, hazard_adapter=None,
                 physics_version=1, physics_dt=None):
        cfg = resolve(overrides=cfg)
        self.context = dict(context or {})
        vehicle = self.context.get("vehicle", "blueboat")
        if cfg.wind.enabled and cfg.wind.windage_profile == "auto":
            # Resolve BEFORE hashing so the digest names the concrete profile.
            cfg = validate(replace(cfg, wind=replace(
                cfg.wind, windage_profile=profile_name_for_vehicle(vehicle))))
        self.cfg = cfg
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.physics_version = int(physics_version)
        if self.physics_version not in (1, 2):
            raise ValueError("physics_version must be 1 or 2")
        self.context.setdefault("physics_version", self.physics_version)
        if cfg.perception.enabled and not groups:
            raise ValueError("enabled perception requires explicit physical measurement groups")
        self.clock = None if physics_dt is None else WaveClock(self.num_envs, self.device, float(physics_dt))
        self.episode_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.scenario = ScenarioRNG(num_envs=num_envs, device=device, eval_seed=cfg.seed)
        # Fail unavailable backends before creating any other runtime state.
        self.wind = WindCondition(cfg.wind, num_envs, device, eval_seed=cfg.seed, vehicle=vehicle)
        self.hazards = DynamicHazardCondition(cfg.dynamic_hazards, num_envs, device, hazard_adapter)
        self.current = CurrentCondition(cfg.current, num_envs, device)
        self.actuator = ActuatorCondition(cfg.actuator, num_envs, device)
        self.waves = WaveCondition(cfg.waves, num_envs, device, physics_version=self.physics_version)
        self.perception = PerceptionCondition(cfg.perception, num_envs, device, cfg.seed, groups or {})
        self.context["wind_model"] = self.wind.header_parameters()
        self.context["clock"] = ("one C2 WaveClock per layer: integer physics ticks, evaluated before "
                                 "one advance per physics substep, reset only for reset envs")

    # ------------------------------------------------------------- clock --
    def _clock_for(self, dt):
        if self.clock is None:
            self.clock = WaveClock(self.num_envs, self.device, float(dt))
        elif float(dt) != self.clock.dt:
            raise ValueError("physics dt changed after the condition clock was created")
        return self.clock

    @property
    def elapsed_s(self):
        """Episode-local physics time in float64 seconds (ticks * dt)."""
        if self.clock is None:
            return torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        return self.clock.ticks.to(torch.float64) * self.clock.dt

    # ------------------------------------------------------------- hooks --
    def action_in(self, actions):
        return self.actuator.action_in(actions)

    def command(self, thrust, yaw, physics_dt, half_beam=None):
        lever = self.cfg.actuator.half_beam_m if half_beam is None else half_beam
        return self.actuator.command(thrust, yaw, physics_dt, lever)

    def scale_drag(self, value):
        """Physics v1 only: v2 takes the multiplier via hull_damping_scales."""
        return self.actuator.scale_drag(value)

    def hull_damping_scales(self):
        """Single drag/mass source for C2's implicit v2 hull damping."""
        return self.actuator.hull_damping_scales(self.cfg.actuator.reference_mass_kg,
                                                 self.cfg.actuator.reference_izz_kgm2)

    def environmental_wrench(self, position_xy, velocity_xy, dt, *,
                             orientation_wxyz=None, angular_velocity_w=None, heading_rad=None):
        """WORLD [N,3] increments at episode-local physics time t, then t += dt.

        ``velocity_xy`` is the hull COM velocity (root_com_vel_w[:, :2]);
        ``heading_rad`` is the physical bow azimuth from forward_vec.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("physics dt must be finite and positive")
        clock = self._clock_for(dt)
        force = position_xy.new_zeros((self.num_envs, 3))
        torque = torch.zeros_like(force)
        if self.cfg.current.enabled:
            f, t = self.current.wrench(velocity_xy, self.elapsed_s)
            force += f; torque += t
        if self.cfg.waves.enabled:
            f, t = self.waves.wrench(position_xy, velocity_xy, clock.time())
            force += f; torque += t
        if self.cfg.wind.enabled:
            f, t = self.wind.wrench(position_xy, velocity_xy, heading_rad=heading_rad,
                                    orientation_wxyz=orientation_wxyz,
                                    angular_velocity_w=angular_velocity_w)
            force += f; torque += t
            self.wind.step(dt)
        if self.cfg.any_enabled:
            clock.advance()
        return force, torque

    def dynamics(self, forces_body, torques_body):
        return self.actuator.dynamics(forces_body, torques_body,
                                     self.cfg.actuator.reference_mass_kg,
                                     self.cfg.actuator.reference_izz_kgm2)

    def wants(self, name):
        return self.perception.wants(name)

    def observation_out(self, name, measurement):
        """Physical measurement boundary, once per group/control observation."""
        return self.perception.apply(name, measurement)

    apply = observation_out

    def reset_runtime(self, env_ids=None, progress=None, *, reference_axis_rad=None,
                      task_episode_ids=None):
        """Reset selected rows OR update task-defined progress at runtime.

        ``reference_axis_rad`` is the task axis (one value per reset env) used
        by relative wind directions. ``task_episode_ids`` (optional) are the
        task ScenarioRNG episode indices; they must equal this layer's.
        """
        if env_ids is None:
            if self.cfg.any_enabled:
                self.hazards.runtime(progress)
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.unique().numel() != ids.numel() or ((ids < 0) | (ids >= self.num_envs)).any():
            raise ValueError("reset ids must be unique and in bounds")
        if not self.cfg.any_enabled:
            # All suites off (journal calm): bookkeeping only, no draws.
            self.episode_index[ids] += 1
            if self.clock is not None:
                self.clock.reset(ids)
            return
        self.scenario.reset_idx(ids)
        self.episode_index[ids] += 1
        if task_episode_ids is not None:
            expected = self.scenario.episode_indices()[ids.detach().cpu()]
            supplied = torch.as_tensor(task_episode_ids).detach().cpu().to(torch.long).flatten()
            if not torch.equal(expected, supplied):
                raise ValueError("condition episode counter diverged from the task scenario counter")
        if self.clock is not None:
            self.clock.reset(ids)
        self.current.reset(ids, self.scenario)
        self.waves.reset(ids, self.scenario)
        self.wind.reset(ids, self.scenario, reference_axis_rad=reference_axis_rad)
        self.actuator.reset(ids)
        self.perception.reset(ids)
        self.hazards.reset(ids, self.scenario)

    # ------------------------------------------------------------ stamps --
    def certificate_stamp(self):
        return stamp_conditions({}, self.cfg, context=self.context)["conditions"]

    def episode_stamp(self, env_id):
        """Call BEFORE reset; env latches this for the completed episode."""
        i = int(env_id)
        runtime = {
            "layer_active": True,
            "env_index": i, "episode_index": int(self.episode_index[i]),
            "elapsed_physics_s": float(self.elapsed_s[i]),
            "clock_ticks": 0 if self.clock is None else int(self.clock.ticks[i]),
            "scenario_protocol_version": SCENARIO_PROTOCOL_VERSION,
            "current_direction_rad": float(self.current.direction[i]),
            "wave_realization": self.waves.episode_parameters(i),
            "wind_realization": self.wind.episode_parameters(i),
            "perception_realization": self.perception.episode_parameters(i),
            "tensor_device_type": self.device.type,
            "hazard_triggered": bool(self.hazards.triggered[i]),
        }
        return stamp_conditions({}, self.cfg, runtime=runtime, context=self.context)["conditions"]


HYDRODYNAMICS_MODELS = ("legacy", "hydro_v2")


def hydrodynamics_model_of(env_cfg):
    """``underwater_physics_cfg.hydrodynamics_model``; absent means legacy (D3 section 1)."""
    physics = getattr(env_cfg, "underwater_physics_cfg", None)
    value = getattr(physics, "hydrodynamics_model", "legacy")
    if value not in HYDRODYNAMICS_MODELS:
        raise ValueError(f"hydrodynamics_model must be one of {HYDRODYNAMICS_MODELS}")
    return value


def validate_no_legacy_overlap(env_cfg, physics):
    """Legacy disturbance controls that must be off beside a hydro_v2 layer.

    Same statements as the legacy factory below (kept inline there so the
    legacy path is textually unchanged).
    """
    if (getattr(physics, "enable_current", False)
            or getattr(env_cfg, "current_speed_override_mps", 0)):
        raise ValueError("condition layer requires legacy current to be off")
    if (getattr(env_cfg, "current_ramp_start_mps", None) is not None
            or getattr(env_cfg, "current_ramp_end_mps", None) is not None):
        raise ValueError("condition layer requires the legacy current ramp to be undeclared")
    if getattr(getattr(env_cfg, "sea_state", None), "enable", False):
        raise ValueError("condition layer requires legacy sea_state to be off")
    for name, neutral in (("mass_scale", 1), ("drag_scale", 1),
                          ("thrust_cap_scale", 1), ("motor_tau_s", 0),
                          ("thrust_imbalance", 0), ("payload_mass_kg", 0)):
        if getattr(env_cfg, name, neutral) != neutral or getattr(env_cfg, name + "_choices", ()):
            raise ValueError("condition layer requires legacy actuator axes to be neutral")
    legacy_obs = ("obs_noise_sigma_pos", "obs_noise_sigma_vel", "obs_noise_sigma_ray",
                  "obs_bias_sigma_pos", "obs_bias_sigma_vel", "obs_bias_sigma_ray",
                  "obs_delay_steps", "obs_dropout_p")
    if any(getattr(env_cfg, key, 0) for key in legacy_obs):
        raise ValueError("condition layer requires legacy observation degradation to be off")
    if getattr(env_cfg, "appear_count", 0):
        raise ValueError("condition layer requires legacy timed obstacle appearance to be off")


def build_conditions(env_cfg, num_envs, device, *, family):
    """Return a ConditionLayer, or None for a legacy cfg with every suite off.

    Journal cfgs (``journal_mode=True``) always get a layer and are validated
    even when every suite is off: physics v2 is mandatory and every legacy
    wave/current/actuator/perception/appearance control must be off, so no
    legacy disturbance path can run beside (or instead of) the layer.

    E3: ``underwater_physics_cfg.hydrodynamics_model == "hydro_v2"`` returns
    the schema-3 HydroV2ConditionLayer (always, even with every suite off);
    the legacy composition below is unchanged apart from two added
    rejections of inconsistent configurations (condition waves with an
    observed sea state, D2 item 18; C4 dynamic hazards in journal mode, D7
    B01).
    """
    if hydrodynamics_model_of(env_cfg) == "hydro_v2":
        from .hydro_v2_layer import build_hydro_v2_conditions
        return build_hydro_v2_conditions(env_cfg, num_envs, device, family=family)
    raw = getattr(env_cfg, "conditions", None)
    presets = getattr(env_cfg, "condition_presets", ())
    seed = getattr(env_cfg, "seed", None)
    cfg = resolve(presets, raw, seed=0 if seed is None else seed)
    journal = bool(getattr(env_cfg, "journal_mode", False))
    if not cfg.any_enabled and not journal:
        return None
    if family not in WIRED_FAMILIES:
        raise ValueError(f"family {family!r} has not been wired")

    physics = getattr(env_cfg, "underwater_physics_cfg", None)
    version = int(getattr(physics, "physics_version", 1))
    if journal and version != 2:
        raise ValueError("journal configuration requires physics_version=2")
    # Preserve legacy ids; make ambiguous double-application an error.
    if (getattr(physics, "enable_current", False)
                                or getattr(env_cfg, "current_speed_override_mps", 0)):
        raise ValueError("condition layer requires legacy current to be off")
    if (getattr(env_cfg, "current_ramp_start_mps", None) is not None
            or getattr(env_cfg, "current_ramp_end_mps", None) is not None):
        raise ValueError("condition layer requires the legacy current ramp to be undeclared")
    legacy_sea = getattr(env_cfg, "sea_state", None)
    if getattr(legacy_sea, "enable", False):
        raise ValueError("condition layer requires legacy sea_state to be off")
    if journal and getattr(env_cfg, "obs_sea_state", False):
        raise ValueError("journal configurations must not observe the sea state")
    if cfg.waves.enabled and getattr(env_cfg, "obs_sea_state", False):
        # D2 item 18 (REVIEW.md F18): the observation would read the legacy
        # sea only and silently change width; reject the combination.
        raise ValueError("condition waves cannot be combined with an observed sea state (obs_sea_state)")
    if journal and cfg.dynamic_hazards.enabled:
        # D7 INTEGRATION_SPEC.md B01: in journal mode the C4 endogenous
        # progress trigger stays inert; dynamic hazards are the D7 package.
        raise ValueError("journal configurations must keep the C4 dynamic_hazards trigger disabled "
                         "(D7 package conditions/dynamic owns dynamic hazards)")
    for name, neutral in (("mass_scale", 1), ("drag_scale", 1),
                              ("thrust_cap_scale", 1), ("motor_tau_s", 0),
                              ("thrust_imbalance", 0), ("payload_mass_kg", 0)):
        if getattr(env_cfg, name, neutral) != neutral or getattr(env_cfg, name + "_choices", ()):
            raise ValueError("condition layer requires legacy actuator axes to be neutral")
    if journal and cfg.current.enabled and cfg.current.force_mode != "incremental_from_calm":
        raise ValueError("journal current must be incremental_from_calm until the D3 water-relative redesign")
    if cfg.actuator.enabled and cfg.actuator.reference_vehicle != getattr(env_cfg, "vehicle", "blueboat"):
        raise ValueError("actuator reference_vehicle must match task vehicle; set measured mass/inertia/lever too")
    p = cfg.perception
    legacy_obs = ("obs_noise_sigma_pos", "obs_noise_sigma_vel", "obs_noise_sigma_ray",
                  "obs_bias_sigma_pos", "obs_bias_sigma_vel", "obs_bias_sigma_ray",
                  "obs_delay_steps", "obs_dropout_p")
    if any(getattr(env_cfg, key, 0) for key in legacy_obs):
        raise ValueError("condition layer requires legacy observation degradation to be off")
    if getattr(env_cfg, "appear_count", 0):
        raise ValueError("condition layer requires legacy timed obstacle appearance to be off")
    if family == "station_keeping" and p.enabled:
        if p.noise_range_m or p.bias_range_m:
            raise ValueError("station keeping has no range sensor group")
        if (p.noise_vel_mps or p.bias_vel_mps) and not getattr(env_cfg, "obs_kinematic", False):
            raise ValueError("velocity perception requires a station cfg with obs_kinematic=True")
    yaw_scale = float(getattr(env_cfg, "yaw_rate_obs_scale_rad_s", 1.0))
    ratio = yaw_scale / SPEED_SCALE_MPS
    groups = {
        "pos": ObsChannelGroup(2, (p.noise_pos_m,) * 2, (p.bias_pos_m,) * 2),
        "vel": ObsChannelGroup(3, (p.noise_vel_mps, p.noise_vel_mps, p.noise_vel_mps * ratio),
                               (p.bias_vel_mps, p.bias_vel_mps, p.bias_vel_mps * ratio)),
    }
    if family == "hazard_nav":
        n = int(env_cfg.ray_count)
        groups["ray"] = ObsChannelGroup(n, (p.noise_range_m,) * n, (p.bias_range_m,) * n,
                                       0.0, float(env_cfg.ray_max_range_m))
    context = {
        "family": family, "vehicle": getattr(env_cfg, "vehicle", "blueboat"),
        "journal_mode": journal,
        "physics_version": version,
        "physics_dt_s": float(env_cfg.sim.dt),
        "control_decimation": int(env_cfg.decimation),
        "observation_groups": {name: asdict(group) for name, group in groups.items()},
        "current_observed": False,
        "waves_observed": False,
        "wind_observed": False,
        "legacy_current_enabled": bool(getattr(physics, "enable_current", False)),
        "legacy_waves_enabled": bool(getattr(legacy_sea, "enable", False)),
        "legacy_observation_degradation": {key: getattr(env_cfg, key, 0) for key in legacy_obs},
    }
    return ConditionLayer(cfg, num_envs, device, groups=groups, context=context,
                          physics_version=version, physics_dt=float(env_cfg.sim.dt))
