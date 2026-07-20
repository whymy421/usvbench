# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.vehicles import VehicleSpec, get_vehicle


# USVBench asset directory. This locates the USD; it is not a physics knob.
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)


_DEFAULT_VEHICLE = get_vehicle("wamv")


def _build_robot_cfg(spec: VehicleSpec) -> RigidObjectCfg:
    """Build a docking robot spawn config without overriding authored mass."""
    spawn_kwargs = {
        "usd_path": _os.path.join(_ASSET_DIR, spec.usd_relpath),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Non-binding safety bounds, not vehicle-physics parameters.
            max_linear_velocity=8.0,
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


# Kept as a public compatibility alias; its values now come from the registry.
BOAT_CONFIG = _build_robot_cfg(_DEFAULT_VEHICLE)


@configclass
class UnderwaterPhysicsCfg:
    """Calm-water hull dynamics populated from a :class:`VehicleSpec`."""

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

    enable_current: bool = False
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


def _build_underwater_physics_cfg(spec: VehicleSpec) -> UnderwaterPhysicsCfg:
    """Translate registry names to the existing docking physics interface."""
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
class WavePhysicsCfg:
    """Dormant calm-water settings copied from the realistic boat baseline."""

    enable_wave: bool = False
    wave_height: float = 0.5
    wave_period: float = 5.0
    wave_dir_x: float = 1.0
    wave_dir_y: float = 0.0


@configclass
class VisualCfg:
    """Render-only settings; never read by physics, observations, or reward."""

    enable_water: bool = True
    water_size_m: float = 300.0
    water_res: int = 60
    water_color: tuple = (0.1, 0.3, 0.8)
    enable_tolerance_ring: bool = True
    tolerance_ring_segments: int = 64
    tolerance_ring_color: tuple = (1.0, 0.15, 0.0)
    tolerance_ring_line_width_m: float = 0.35
    enable_berth_marker: bool = True
    berth_arrow_length_m: float = 3.0
    berth_arrow_width_m: float = 0.5
    berth_arrow_color: tuple = (1.0, 0.85, 0.0)
    enable_berth_box: bool = True
    berth_box_length_m: float = 6.5
    berth_box_width_m: float = 3.2
    berth_box_line_width_m: float = 0.25
    berth_box_color: tuple = (1.0, 1.0, 1.0)


@configclass
class DockingEnvCfg(DirectRLEnvCfg):
    vehicle: str = "wamv"

    decimation = 2
    episode_length_s = 120.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 6
    state_space = 0

    success_position_tolerance_m: float = 2.5
    # Mission-evaluation alias of the position tolerance, used to compute SPL.
    goal_radius: float = 2.5
    success_heading_tolerance_deg: float = 15.0
    success_speed_tolerance_mps: float = 0.3
    required_hold_time_s: float = 5.0

    # V3 begins inside the position tolerance so align-and-stop is learnable
    # before the approach distance grows.
    curriculum_start_distance_m: float = 2.0
    curriculum_distance_increment_m: float = 1.0
    curriculum_max_distance_m: float = 25.0
    curriculum_ema_decay: float = 0.99
    curriculum_success_threshold: float = 0.6

    # V5 spawns on the approach lane behind the berth, with positions and bow
    # headings sampled relative to each environment's dock_heading rather than
    # in fixed world coordinates. Bow-ward drift and forward thrust therefore
    # carry the boat toward the berth while alignment supports the approach.
    spawn_bearing_limit_deg: float = 60.0
    spawn_heading_offset_limit_deg: float = 45.0

    observation_distance_scale_m: float = 25.0
    observation_speed_scale_mps: float = 2.0
    reference_reward_distance_scale_m: float = 25.0
    reference_reward_alignment_scale: float = 0.5
    reference_reward_alignment_decay_m: float = 2.5
    reference_reward_braking_scale: float = 0.4
    reference_reward_braking_decay_m: float = 2.5
    reference_reward_braking_speed_scale_mps: float = 1.0
    reference_reward_success_bonus: float = 3.0

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

    robot_cfg: RigidObjectCfg = BOAT_CONFIG
    underwater_physics_cfg: UnderwaterPhysicsCfg = _build_underwater_physics_cfg(
        _DEFAULT_VEHICLE
    )
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()
    # Rendering only: this block must not affect physics, observations, reward,
    # termination, or success semantics.
    visual: VisualCfg = VisualCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=56.0,
        replicate_physics=True,
    )

    dof_names = []

    def __post_init__(self) -> None:
        """Resolve all hull-dependent config from the selected registry entry."""
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm


@configclass
class DockingBlueBoatEnvCfg(DockingEnvCfg):
    vehicle: str = "blueboat"
