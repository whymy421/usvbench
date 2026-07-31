# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the static Hazard Navigation direct task."""

from __future__ import annotations

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.obs_superset import SUPERSET_DIM
from .._shared.vehicles import VehicleSpec, get_vehicle


_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)
_DEFAULT_VEHICLE = get_vehicle("blueboat")


def _build_robot_cfg(spec: VehicleSpec) -> ArticulationCfg | RigidObjectCfg:
    """Build a registry-selected vehicle without overriding authored mass."""
    spawn_kwargs = {
        "usd_path": _os.path.join(_ASSET_DIR, spec.usd_relpath),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0 if spec.asset_kind == "articulation" else 8.0,
            max_angular_velocity=573.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        "activate_contact_sensors": False,
    }
    if spec.mass_kg is not None:
        spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass_kg)

    if spec.asset_kind == "articulation":
        spawn_kwargs["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, -0.15),
                rot=(1.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
            collision_group=0,
        )

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


@configclass
class UnderwaterPhysicsCfg:
    """Calm-water hull dynamics populated from the vehicle registry."""

    water_density: float = 1000.0
    gravity: float = 9.8
    water_surface_z: float = 0.0
    rov_volume: float = _DEFAULT_VEHICLE.displaced_volume_m3
    rov_height: float = _DEFAULT_VEHICLE.hull_height_m
    buoyancy_center_offset: float = _DEFAULT_VEHICLE.buoyancy_center_offset_m
    surge_lin_damping: float = _DEFAULT_VEHICLE.surge_lin
    surge_quad_damping: float = _DEFAULT_VEHICLE.surge_quad
    sway_lin_damping: float | None = _DEFAULT_VEHICLE.sway_lin
    sway_quad_damping: float | None = _DEFAULT_VEHICLE.sway_quad
    heave_damping: float = _DEFAULT_VEHICLE.heave_damping
    yaw_lin_damping: float = _DEFAULT_VEHICLE.yaw_lin
    yaw_quad_damping: float = _DEFAULT_VEHICLE.yaw_quad
    restoring_stiffness_roll: float = _DEFAULT_VEHICLE.restoring_stiffness_roll
    restoring_stiffness_pitch: float = _DEFAULT_VEHICLE.restoring_stiffness_pitch
    rollpitch_rate_damping: float = _DEFAULT_VEHICLE.rollpitch_rate_damping


def _build_underwater_physics_cfg(spec: VehicleSpec) -> UnderwaterPhysicsCfg:
    return UnderwaterPhysicsCfg(
        rov_volume=spec.displaced_volume_m3,
        rov_height=spec.hull_height_m,
        buoyancy_center_offset=spec.buoyancy_center_offset_m,
        surge_lin_damping=spec.surge_lin,
        surge_quad_damping=spec.surge_quad,
        sway_lin_damping=spec.sway_lin,
        sway_quad_damping=spec.sway_quad,
        heave_damping=spec.heave_damping,
        yaw_lin_damping=spec.yaw_lin,
        yaw_quad_damping=spec.yaw_quad,
        restoring_stiffness_roll=spec.restoring_stiffness_roll,
        restoring_stiffness_pitch=spec.restoring_stiffness_pitch,
        rollpitch_rate_damping=spec.rollpitch_rate_damping,
    )


@configclass
class WaveCfg:
    """Sea state. ``mode="calm"`` is the frozen calm-water baseline.

    Waves enter the physics through one channel only: the water surface stops
    being the plane ``water_surface_z`` and becomes ``water_surface_z + eta``,
    so buoyancy and heave follow from the existing hydrostatics. ``slope_*``
    additionally tilts the restoring equilibrium toward the wave normal, which
    is what produces roll; setting ``slope_torque_scale = 0`` leaves heave as
    the only wave effect.
    """

    mode: str = "calm"  # calm | airy | jonswap

    # Shared. None randomizes per env; a number pins every env to that heading
    # (degrees, 0 = +x), which is what a controlled beam/following-sea sweep
    # needs.
    direction_deg: float | None = None

    # Sea states are scaled to the hull, not copied from the calm reference
    # tasks. BlueBoat displaces 0.0346 m^3 against the ROV's 20 m^3, and its
    # hull is 0.376 m tall, so the surface only has to move +-0.188 m for the
    # boat to leave the water or submerge completely. The rov/boat defaults
    # (Airy 0.5 m, Hs 0.3-1.0 m) saturate buoyancy 25-46% of the time, which
    # replaces wave response with free-fall and makes every wave model look
    # alike. These values keep the peak well inside the hull.
    airy_height_m: float = 0.12
    airy_period_s: float = 3.0

    # JONSWAP irregular sea.
    hs_min_m: float = 0.06
    hs_max_m: float = 0.18
    tp_min_s: float = 2.0
    tp_max_s: float = 4.0
    gamma_min: float = 1.0
    gamma_max: float = 5.0
    n_components: int = 30
    # Band follows Tp: at Tp = 2-4 s the peak sits at 0.25-0.5 Hz, so the
    # ocean-scale 0.04-0.5 Hz band used by the calm tasks would clip the whole
    # high-frequency side of the spectrum and under-deliver Hs.
    f_min_hz: float = 0.10
    f_max_hz: float = 1.60
    spread_deg: float = 30.0

    # Wave loads, matching tasks/boat_calm_nav so the ROV/boat baselines and
    # the E7 line stay comparable. heave lifts the hull, roll is driven by how
    # beam-on the sea is, and drag acts against the bow so heading into the
    # waves costs speed -- that last one is what makes waves felt at all.
    heave_force_gain: float = 100.0
    roll_torque_gain: float = 200.0
    wave_drag_gain: float = 15.0

    # Optional second roll channel: redirects the restoring equilibrium toward
    # the wave normal instead of world-up. Off by default -- the gains above
    # already carry roll, and stacking both double-counts it.
    slope_torque_scale: float = 0.0
    # Guard against the linear-wave model being pushed past its validity: a
    # steepness this high is already outside deep-water linear theory.
    max_slope: float = 0.30


@configclass
class VisualCfg:
    """Render-only layer; no field is read by task physics or scoring."""

    enable_water: bool = True
    water_size_m: float = 1000.0
    water_res: int = 80
    water_color: tuple = (0.1, 0.3, 0.8)
    enable_goal_ring: bool = True
    goal_ring_segments: int = 64
    goal_ring_line_width_m: float = 0.30
    goal_ring_color: tuple = (1.0, 0.45, 0.0)
    enable_obstacles: bool = True
    obstacle_color: tuple = (0.35, 0.37, 0.40)
    obstacle_height_m: float = 2.0
    # Deforming surface used instead of the flat plane when waves are on. It
    # rides with env 0's boat, so it only has to cover what the camera sees.
    wave_mesh_size_m: float = 120.0
    wave_mesh_res: int = 96
    # Height colouring for the wave surface, matching the E7 line's rendering.
    wave_colormap: str = "turbo"


@configclass
class HazardNavEnvCfg(DirectRLEnvCfg):
    """Adversarially-converged Task A: Static Hazard Navigation."""

    vehicle: str = "blueboat"

    decimation = 2
    episode_length_s = 120.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 39
    state_space = 0
    emit_superset_obs: bool = False
    # B1 (v2 id): native obs gains the reached latch + speed_norm (41-D), the
    # superset stage slot carries the latch, and the swiftness term activates.
    obs_v2: bool = False
    # v3 exposes the minimal planar dynamic state: body-frame surge, sway, and
    # yaw rate. It is mutually exclusive with the append-only v2 layout.
    obs_v3: bool = False
    yaw_rate_obs_scale_rad_s: float = 1.0

    # Render-only official Blue Robotics model. The gray physics mesh remains
    # authoritative for mass, inertia, collision, buoyancy, and drag.
    use_official_blueboat_visual: bool = True
    official_blueboat_visual_usd_path: str = _os.path.join(
        _ASSET_DIR, "BB120_official_visual_only.usd"
    )
    official_blueboat_visual_translation: tuple = (-0.60396, -0.59322, -0.22357)
    official_blueboat_visual_orientation: tuple = (0.5, 0.5, 0.5, 0.5)

    goal_radius: float = 2.0
    min_goal_distance_m: float = 20.0
    max_goal_distance_m: float = 40.0
    half_beam_m: float = 0.45
    collision_margin_m: float = 0.20
    safe_clearance_m: float = 0.90

    ray_count: int = 36  # v3: 10 deg spacing
    ray_max_range_m: float = 30.0
    # Level 3 spawns K=14; 12-slot buffers crashed the first level-3 reset,
    # so no pre-v6 run ever contained level-3 experience.
    max_obstacles: int = 14
    layout_max_attempts: int = 20
    layout_seed: int = 0

    curriculum_start_level: float = 0.0
    curriculum_level_increment: float = 1.0
    curriculum_max_level: float = 3.0  # v4: ultimate tier = 2x-beam gaps
    curriculum_ema_decay: float = 0.99
    curriculum_success_threshold: float = 0.60
    # Eval-only: the global curriculum EMA updates on every completed episode,
    # so a good policy can cross the 0.60 threshold MID-EVAL and silently
    # shift the tier under a harvest run. Freeze pins the level for the whole
    # process; every reported number must state its tier.
    curriculum_frozen: bool = False
    eval_level: int = 0

    reward_progress_scale: float = 20.0
    # v6a: quadratic graze tax retired (0.0) but KEPT for the shape-ablation
    # arm -- reward_clearance_scale=1.0 + reward_prox_scale=0.0 restores the
    # exact v5 reward path.
    reward_clearance_scale: float = 0.0
    # v6a: per-ray log proximity field over the 36 rays, band
    # [prox_ray_floor_m, safe_clearance_m + half_beam_m] = [0.45, 1.35] m.
    # Zero beyond ~2x hull-beam clearance; w=4.1 matches v5's at-touch cost
    # averaged over ray phase at the r=1.4 m reference cylinder.
    reward_prox_scale: float = 4.1
    prox_ray_floor_m: float = 0.45
    reward_contact_entry_penalty: float = 25.0
    reward_contact_dwell_penalty: float = 1.0
    # Paid once on the first collision-free goal entry. The bounded 50..100
    # range rewards decisive entry without letting lucky exploratory successes
    # dominate PPO's value targets.
    reward_goal_entry_bonus: float = 50.0
    reward_goal_time_bonus: float = 50.0
    # Integrated squared negative surge command. A full 120 s reverse episode
    # costs 6 points; a one-second escape manoeuvre costs at most 0.05.
    reward_reverse_action_scale: float = 0.05
    # Integrated squared negative body-frame surge speed. This catches
    # stern-first coasting after the policy releases a negative thrust command.
    reward_reverse_velocity_scale: float = 0.0
    # Swiftness applies only when velocity is observable (v2/v3).
    # Full-idle episode cost = scale * 120 s = 30.0, making a timeout worse
    # than active progress while remaining comparable to one contact penalty.
    reward_swift_scale: float = 0.25

    thrust_max_fwd: float = _DEFAULT_VEHICLE.thrust_fwd_n
    thrust_max_rev: float = _DEFAULT_VEHICLE.thrust_rev_n
    yaw_torque_max: float = _DEFAULT_VEHICLE.yaw_torque_nm

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )
    robot_cfg: ArticulationCfg | RigidObjectCfg = _build_robot_cfg(_DEFAULT_VEHICLE)
    underwater_physics_cfg: UnderwaterPhysicsCfg = _build_underwater_physics_cfg(
        _DEFAULT_VEHICLE
    )
    # 96 m spacing contains a 40 m route and its sampled lateral corridor.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=96.0,
        replicate_physics=True,
    )
    visual: VisualCfg = VisualCfg()
    wave: WaveCfg = WaveCfg()
    dof_names = []

    def __post_init__(self) -> None:
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm

        if self.obs_v2 and self.obs_v3:
            raise ValueError("obs_v2 and obs_v3 are mutually exclusive")
        if self.obs_v3:
            native_dim = 6 + self.ray_count
        elif self.obs_v2:
            native_dim = 5 + self.ray_count
        else:
            native_dim = 3 + self.ray_count
        expected_observation_space = (
            SUPERSET_DIM if self.emit_superset_obs else native_dim
        )
        if self.ray_count != 36 or self.observation_space != expected_observation_space:
            raise ValueError(
                "HazardNav requires 36 rays and observation_space="
                f"{expected_observation_space} when "
                f"emit_superset_obs={self.emit_superset_obs}, "
                f"obs_v2={self.obs_v2}, obs_v3={self.obs_v3}"
            )
        if self.yaw_rate_obs_scale_rad_s <= 0.0:
            raise ValueError("yaw_rate_obs_scale_rad_s must be positive")
        if self.max_obstacles < 14:
            raise ValueError("max_obstacles must accommodate curriculum level 3 (K=14)")
        if self.min_goal_distance_m != 20.0 or self.max_goal_distance_m != 40.0:
            raise ValueError("HazardNav v1 fixes D0 sampling to U[20, 40] m")
        if self.reward_goal_entry_bonus < 0.0 or self.reward_goal_time_bonus < 0.0:
            raise ValueError("goal-entry reward scales must be non-negative")
        if self.reward_reverse_action_scale < 0.0:
            raise ValueError("reward_reverse_action_scale must be non-negative")
        if self.reward_reverse_velocity_scale < 0.0:
            raise ValueError("reward_reverse_velocity_scale must be non-negative")


@configclass
class HazardNavV2EnvCfg(HazardNavEnvCfg):
    """B1 wave: velocity observability + reached latch + swiftness term.

    Registered under the append-only gym id Isaac-USV-HazardNav-Direct-v2;
    the v1 id, layout, champions, and certificates stay untouched.
    """

    obs_v2: bool = True
    observation_space = 41


@configclass
class HazardNavV3EnvCfg(HazardNavEnvCfg):
    """Minimal Markov planar observation for inertia-aware navigation."""

    obs_v3: bool = True
    observation_space = 42
    # Fixed-angle chase view for v11 playback and video capture. Isaac Lab's
    # viewport controller translates this camera with env 0's boat at every
    # render update, but does not rotate it with the hull.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, -6.0, 11.0),
        lookat=(0.0, 0.0, 0.30),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        resolution=(1280, 720),
    )


@configclass
class HazardNavV4EnvCfg(HazardNavV3EnvCfg):
    """V12: preserve v11 navigation while making sustained reverse costly."""

    reward_reverse_action_scale: float = 0.75
    reward_reverse_velocity_scale: float = 0.75


@configclass
class HazardNavV3AiryEnvCfg(HazardNavV3EnvCfg):
    """v3 under a regular wave. Observation and action layouts are unchanged.

    The policy contract stays byte-identical to v3, which is the point: a v3
    checkpoint runs here with no adaptation, so the score difference measures
    what waves cost a calm-trained policy and nothing else.
    """

    wave: WaveCfg = WaveCfg(mode="airy")


@configclass
class HazardNavV3JonswapEnvCfg(HazardNavV3EnvCfg):
    """v3 under an irregular JONSWAP sea. Layouts unchanged, as for Airy."""

    wave: WaveCfg = WaveCfg(mode="jonswap")
