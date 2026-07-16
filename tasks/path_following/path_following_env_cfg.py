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


# Vehicle and dynamics values are copied from tasks/rov_calm_nav. Do not tune
# them per mission: all ROV capability tasks share this realistic calm-water hull.
ROV_CONFIG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "ROV_rigged.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Non-binding safety limits; terminal speeds emerge from drag balance.
            # Isaac Lab expresses max angular velocity in degrees per second.
            max_linear_velocity=10.0,
            max_angular_velocity=573.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
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
    """Realistic calm-water ROV dynamics copied from ``rov_calm_nav``.

    The BlueROV2-style per-DOF linear-plus-quadratic damping values are
    re-derived for this 100 kg hull. They give approximately 1.54 m/s terminal
    surge speed at 400 N and 1.00 rad/s terminal yaw rate at 200 N m.
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
    attitude_spring: float = 5000.0
    rollpitch_rate_damping: float = 2000.0

    enable_current: bool = False
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


@configclass
class WavePhysicsCfg:
    """Dormant calm-water settings copied from ``rov_calm_nav``."""

    enable_wave: bool = False
    wave_height: float = 0.5
    wave_period: float = 5.0
    wave_dir_x: float = 1.0
    wave_dir_y: float = 0.0


@configclass
class VisualCfg:
    """Render-only settings; never read by task science or vehicle physics."""

    enable_water: bool = True
    water_size_m: float = 300.0
    water_res: int = 60
    water_color: tuple = (0.1, 0.3, 0.8)
    enable_waypoint_markers: bool = True
    waypoint_segments: int = 64
    waypoint_color: tuple = (1.0, 0.15, 0.0)
    waypoint_line_width_m: float = 0.35


@configclass
class PathFollowingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 120.0

    action_space = 2
    observation_space = 3
    state_space = 0

    num_waypoints: int = 4
    segment_length_min: float = 12.0
    segment_length_max: float = 20.0
    heading_change_max_deg: float = 60.0
    goal_radius: float = 2.0
    reference_reward_gate_bonus: float = 10.0

    # Blue Robotics T200 baseline: body +Y thrust and body +Z yaw torque.
    # Actions are hard-clipped to [-1, 1] before applying these limits.
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
    visual: VisualCfg = VisualCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        # Four 20 m segments may extend 80 m from an origin.
        env_spacing=180.0,
        replicate_physics=True,
    )

    dof_names = []
