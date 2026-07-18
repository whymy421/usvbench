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


# USVBench asset directory. This locates the USD; it is not a physics knob.
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)


# Vehicle and dynamics values below are copied exactly from the realistic boat
# baseline via tasks/station_keeping_boat. Hydrodynamic and actuator sources are
# cited alongside the relevant constants.
BOAT_CONFIG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "boat_physics.usdc"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Non-binding safety bounds (>=2x the terminal values that emerge
            # from thrust/drag balance: 2.0 m/s surge, 0.8 rad/s yaw).
            # Isaac Lab units: linear is m/s and angular is deg/s; 573 deg/s is
            # 10 rad/s and is never reached in practice.
            max_linear_velocity=8.0,
            max_angular_velocity=573.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        activate_contact_sensors=False,
    ),
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
    """Realistic calm-water boat dynamics copied from the boat baseline.

    Drag is per-DOF linear plus quadratic and Froude-scaled (lambda=0.8,
    matching the 100 kg hull) from the VRX WAM-V coefficients in
    ``wamv_gazebo_dynamics_plugin.xacro``: xU=100, xUU=150, yV=100,
    yVV=100, zW=500, nR=800, nRR=800. Linear drag scales by lambda^2.5,
    quadratic by lambda^2, yaw linear by lambda^4.5, and yaw quadratic by
    lambda^5. The resulting surge and yaw terminal values are 2.0 m/s at
    500 N and 0.8 rad/s at 400 N*m.
    """

    water_density: float = 1000.0
    gravity: float = 9.8
    # A 100 kg boat displaces 0.1 m^3; volume 0.2 m^3 balances at half submersion.
    rov_volume: float = 0.2
    rov_height: float = 1.0
    water_surface_z: float = 0.0
    buoyancy_center_offset: float = 0.0

    # Hull drag is anisotropic and applied in the BODY frame.
    surge_lin_damping: float = 50.0
    surge_quad_damping: float = 100.0
    sway_lin_damping: float = 50.0
    sway_quad_damping: float = 65.0
    heave_damping: float = 300.0
    yaw_lin_damping: float = 300.0
    yaw_quad_damping: float = 250.0
    # Roll/pitch are not task DOFs. These baseline values overdamp wobble modes.
    attitude_spring: float = 5000.0
    # V9b: sustained yaw spin (~0.27 rad/s) at rest pumps roll/pitch through
    # the asset's inertia products (COM sits 1.03 m off the USD origin) and
    # the boat tumbles within ~3 s (ghost_force_probe 2026-07-16). Explicit
    # overdamping is numerically unstable past ~2*I*f, so the real fix is
    # balancing the boat USD inertia (authorized, ROV-campaign tooling);
    # until then the reference recipe cannot hold yaw spins near the berth.
    rollpitch_rate_damping: float = 2000.0

    enable_current: bool = False
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


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

    # VRX classic thrusters provide 250 N forward and 100 N reverse each.
    # The two aft thrusters therefore give the asymmetric 500 N / 200 N limits
    # used by the realistic boat baseline. Actions are hard-clipped to [-1, 1].
    thrust_max_fwd: float = 500.0
    thrust_max_rev: float = 200.0
    yaw_torque_max: float = 400.0

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
    underwater_physics_cfg: UnderwaterPhysicsCfg = UnderwaterPhysicsCfg()
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
