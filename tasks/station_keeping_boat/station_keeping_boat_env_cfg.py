# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os as _os

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
# baseline in tasks/boat_calm_nav. Hydrodynamic and actuator sources are cited
# alongside the relevant constants.
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
class StationKeepingBoatEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 120.0

    action_space = 2
    observation_space = 3
    state_space = 0

    hold_radius: float = 3.0
    required_hold_time_s: float = 60.0
    min_spawn_distance: float = 5.0
    max_spawn_distance: float = 15.0
    reference_reward_distance_scale: float = 30.0

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=36.0,
        replicate_physics=True,
    )

    dof_names = []
