# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import os as _os

# USVBench asset directory.
# Set the USVBENCH_ASSETS env var to your <repo>/assets folder.
# Defaults to ~/usvbench/assets (works if you cloned the repo into your home dir).
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)

# ============================================
# ROV配置 (原始模型)
# ============================================
ROV_CONFIG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "ROV_rigged.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Velocity limits are NON-BINDING safety bounds only (>=2x the terminal
            # values that emerge from thrust/drag balance: ~1.54 m/s, ~1.0 rad/s).
            # NOTE Isaac Lab units: max_linear_velocity is m/s, max_angular_velocity
            # is DEG/s (schemas_cfg.py) — the old value 5.0 was an effective 5 deg/s
            # cap that dominated yaw dynamics.
            max_linear_velocity=10.0,
            max_angular_velocity=573.0,  # = 10 rad/s, never reached in practice
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            # Explicitly zero PhysX body damping (ROV_rigged.usd authors 4.0/4.0,
            # which silently dominated all hydrodynamic drag). All fluid damping now
            # lives in UnderwaterPhysicsCfg below — single source of truth.
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=100.0,
        ),
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


# ============================================
# 水下物理参数配置
# ============================================
@configclass
class UnderwaterPhysicsCfg:
    """ROV水下物理参数配置

    Damping structure (per-DOF linear + quadratic) follows the experimentally
    validated BlueROV2 model of von Benzon et al. 2022 (JMSE 10(12):1898).
    Magnitudes are re-derived for this 100 kg / Iz~141 kg·m² craft (by inertia it
    is a ~4 m surface vehicle, so ROV coefficients are NOT copied directly):
      surge:  F_d = -(30 + 150·|v|)·v   →  terminal 1.54 m/s at 400 N thrust
      yaw:    T_d = -(20 + 180·|r|)·r   →  terminal 1.00 rad/s at 200 N·m
    Quadratic surge term from Cd≈1.2, frontal area≈0.25 m² (0.5·ρ·Cd·A≈150).
    Terminal speeds EMERGE from force balance; engine velocity caps never bind.
    """
    water_density: float = 1000.0
    gravity: float = 9.8
    rov_volume: float = 0.5  # m^3 (增大排水体积，确保波浪下不沉)
    rov_height: float = 0.6
    water_surface_z: float = 0.0
    buoyancy_center_offset: float = -0.1

    surge_lin_damping: float = 30.0     # N·s/m   (skin friction, ~10% at v_max)
    surge_quad_damping: float = 150.0   # N·s²/m² (form drag)
    heave_damping: float = 400.0        # N·s/m   (settles bobbing, ζ≈0.22)
    yaw_lin_damping: float = 20.0       # N·m·s/rad
    yaw_quad_damping: float = 180.0     # N·m·s²/rad²
    # Roll/pitch passive stability. These DOFs are not task-relevant; they are
    # made strongly overdamped (ζ≈1.2) so the asymmetric-rotor whirl instability
    # (transverse inertia 140 vs 65 + world-frame spring; growth ~2.6/s at
    # sustained 1 rad/s yaw, ~10 s incubation from float noise) can never build.
    # With the hull inertia balanced (products zeroed in ROV_rigged.usd), steady
    # roll/pitch rates are zero, so this damping does no work in normal sailing
    # and does NOT bleed yaw energy.
    attitude_spring: float = 5000.0         # N·m/rad
    rollpitch_rate_damping: float = 2000.0  # N·m·s/rad (overdamped on purpose)

    enable_current: bool = False   # CALM: 关掉洋流
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


# ============================================
# 波浪物理参数配置
# ============================================
@configclass
class WavePhysicsCfg:
    enable_wave: bool = False   # CALM: 关掉波浪物理
    wave_height: float = 0.5
    wave_period: float = 5.0
    wave_dir_x: float = 1.0
    wave_dir_y: float = 0.0


# ============================================
# 环境配置
# ============================================
@configclass
class MyFirstTaskEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 120.0

    action_space = 2
    # 🔧 默认值原来是 7,和本任务 STARTER_TASK.md 要求的 OBS_DIM=3、以及仓库里 ship 的
    #    checkpoint 都对不上 —— 不显式设 OBS_DIM 直接跑,加载 checkpoint 会崩。
    observation_space = int(_os.environ.get('OBS_DIM', '3'))
    state_space = 0

    goal_radius: float = 2.0
    max_spawn_distance: float = 30.0
    min_spawn_distance: float = 10.0
    use_learned_reward: bool = False

    # Actuator limits. Actions are HARD-CLIPPED to [-1,1] in _pre_physics_step
    # (previously unbounded: trained policies emitted |action|~170, i.e. ~17 kN).
    # 400 N ≈ 8x BlueRobotics T200 @16V (8×51.5 N, datasheet); thrust-to-weight
    # 0.41. Body-axis forward is +Y. 200 N·m ≈ two 200 N thrusters at ±0.5 m.
    thrust_max: float = 400.0      # N,   surge force at action[0] = ±1
    yaw_torque_max: float = 200.0  # N·m, yaw torque at action[1] = ±1

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=12.0,
        replicate_physics=True
    )

    dof_names = []
