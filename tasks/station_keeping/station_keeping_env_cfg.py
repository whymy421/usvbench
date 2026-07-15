# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# USVBench asset directory. This locates the USD; it is not a physics knob.
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)


# Vehicle and dynamics values below are copied from the realistic ROV baseline
# on jinshi-brady/realistic-dynamics. Hydrodynamic and actuator sources are cited
# alongside the relevant constants.
ROV_CONFIG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "ROV_rigged.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Velocity limits are NON-BINDING safety bounds only (>=2x the terminal
            # values that emerge from thrust/drag balance: ~1.54 m/s, ~1.0 rad/s).
            # NOTE Isaac Lab units: max_linear_velocity is m/s, max_angular_velocity
            # is DEG/s (schemas_cfg.py) -- 573 deg/s = 10 rad/s.
            max_linear_velocity=10.0,
            max_angular_velocity=573.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            # The USD-authored 4.0/4.0 damping is disabled so UnderwaterPhysicsCfg
            # is the single source of fluid damping, as in the realistic ROV baseline.
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        activate_contact_sensors=False,
    ),
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


@configclass
class UnderwaterPhysicsCfg:
    """Realistic calm-water ROV dynamics.

    The per-DOF linear-plus-quadratic damping structure follows the experimentally
    validated BlueROV2 model of von Benzon et al. 2022 (JMSE 10(12):1898).
    Magnitudes are re-derived for this 100 kg / Iz~141 kg*m^2 craft:

    - surge: F_d = -(30 + 150|v|)v -> terminal 1.54 m/s at 400 N
    - yaw: T_d = -(20 + 180|r|)r -> terminal 1.00 rad/s at 200 N*m

    The quadratic surge term uses Cd~=1.2 and frontal area~=0.25 m^2
    (0.5*rho*Cd*A~=150). Values match the realistic ROV baseline exactly.
    """

    water_density: float = 1000.0
    gravity: float = 9.8
    rov_volume: float = 0.5
    rov_height: float = 0.6
    water_surface_z: float = 0.0
    buoyancy_center_offset: float = -0.1

    surge_lin_damping: float = 30.0
    surge_quad_damping: float = 150.0
    heave_damping: float = 400.0
    yaw_lin_damping: float = 20.0
    yaw_quad_damping: float = 180.0
    # The realistic ROV baseline makes roll/pitch strongly overdamped (zeta~=1.2)
    # to suppress the asymmetric-rotor whirl instability without bleeding yaw energy.
    attitude_spring: float = 5000.0
    rollpitch_rate_damping: float = 2000.0

    enable_current: bool = False
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


@configclass
class WavePhysicsCfg:
    """Dormant calm-water settings copied from the realistic ROV baseline."""

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
    enable_hold_zone_marker: bool = True
    hold_zone_segments: int = 64
    # High-contrast against the blue water in rendered videos: a thin orange ring
    # washed out to pale yellow at 0.12 m, so the ring is wider and near-red.
    hold_zone_color: tuple = (1.0, 0.15, 0.0)
    hold_zone_line_width_m: float = 0.35


@configclass
class StationKeepingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 120.0

    action_space = 2
    observation_space = 3
    state_space = 0

    hold_radius: float = 2.0
    required_hold_time_s: float = 60.0
    min_spawn_distance: float = 5.0
    max_spawn_distance: float = 15.0
    reference_reward_distance_scale: float = 30.0

    # Blue Robotics T200 datasheet: 51.5 N at 16 V. The baseline represents
    # roughly eight thrusters; 200 N*m is two 200 N thrusters at +/-0.5 m.
    # Actions are hard-clipped to [-1, 1] before these limits are applied.
    thrust_max: float = 400.0
    yaw_torque_max: float = 200.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot_cfg: ArticulationCfg = ROV_CONFIG
    underwater_physics_cfg: UnderwaterPhysicsCfg = UnderwaterPhysicsCfg()
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()
    # Rendering only: this block must not affect physics, observations, reward,
    # termination, or success semantics.
    visual: VisualCfg = VisualCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=36.0,
        replicate_physics=True,
    )

    dof_names = []
