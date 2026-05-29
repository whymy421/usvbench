# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

# ============================================
# ROV配置
# ============================================
ROV_CONFIG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/Yutong/NavRL/NavRL2026/isaac_underwater/boat_physics.usdc",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=5.0,
            max_angular_velocity=5.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=1.0,   # terminal v = thrust/(m*damp) = 200/(100*1) = 2.0 m/s,够导航
            angular_damping=3.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=100.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, float(__import__('os').environ.get('INIT_Z', '0.0'))),  # 默认0=平衡点出生(训练用);设INIT_Z=1.0看高空掉下演示
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    collision_group=0,
)


# ============================================
# 水下物理参数配置
# ============================================
@configclass
class UnderwaterPhysicsCfg:
    """ROV水下物理参数配置"""
    # ========== 浮力参数 ==========
    water_density: float = 1000.0  # kg/m^3 (淡水密度)
    gravity: float = 9.8  # m/s^2
    rov_volume: float = 0.2  # m^3 (boat: 100kg船在z=0时50%浸没→浮力平衡)
    rov_height: float = 1.0  # m (船体高度)
    water_surface_z: float = 0.0  # m (水面高度，世界坐标系)
    buoyancy_center_offset: float = 0.0  # 关掉浮力扭矩,防止USD质心偏移导致飘走

    # ========== 阻尼参数 ==========
    max_linear_damping: float = 30.0  # N/(m/s) 降低,让浮力作用更明显(弹跳)
    max_angular_damping: float = 200.0  # Nm/(rad/s) 大幅提高,压制旋转飘移
    air_linear_damping: float = 0.5
    air_angular_damping: float = 0.05

    # ========== 洋流参数 ==========
    enable_current: bool = False  # 静水baseline:关掉洋流,船应原地漂浮
    current_speed_min: float = 0.2  # m/s
    current_speed_max: float = 0.3  # m/s
    current_drag_coeff: float = 8.0  # N/(m/s)²


# ============================================
# 🆕 波浪物理参数配置
# ============================================
@configclass
class WavePhysicsCfg:
    """波浪参数配置"""
    enable_wave: bool = True
    wave_height: float = 0.5
    wave_period: float = 5.0  # 波周期 (秒)
    wave_dir_x: float = 1.0  # 波浪来向 X
    wave_dir_y: float = 0.0  # 波浪来向 Y


# ============================================
# 环境配置
# ============================================
@configclass
class MyFirstTaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 120.0

    # sp
    #    Position: (38.9, -45.6) | Z=-0.30m
    #    Target:   (28.2, -48.1) | Distance: 11.0m
    #    Attitude: Roll=-3.8° Pitch=-1.8°
    #    Heading:  Yaw= -56.9° | ToTarget=-103.1° | Error= 46.2°
    #    Speed:    2.33 m/s
    #    Current:  Dir= -62.7° | Speed=0.30 m/s
    #    Buoyancy: 1004N
    #    Current Force: [41.0, -35.7] N
    #  45%|█████████████████████████████████████████████████▏       aces definition
    action_space = 2  # [forward_thrust, yaw_torque]
    observation_space = 3
    state_space = 0

    # 任务参数
    goal_radius: float = 2.0  # 到达判定半径 (米)
    max_spawn_distance: float = 30.0  # 目标最大生成距离
    min_spawn_distance: float = 10.0  # 目标最小生成距离
    use_learned_reward: bool = False


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    # robot - 使用RigidObject (boat USD没有ArticulationRootAPI,用RigidObject加载)
    robot_cfg: RigidObjectCfg = ROV_CONFIG

    # 水下物理配置
    underwater_physics_cfg: UnderwaterPhysicsCfg = UnderwaterPhysicsCfg()

    # 🆕 波浪配置
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg(enable_wave=False)  # 关闭波浪

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=12.0,
        replicate_physics=True
    )

    # ROV没有关节
    dof_names = []