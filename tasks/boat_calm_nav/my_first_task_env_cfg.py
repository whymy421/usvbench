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
import os as _os

# USVBench asset directory.
# Set the USVBENCH_ASSETS env var to your <repo>/assets folder.
# Defaults to ~/usvbench/assets (works if you cloned the repo into your home dir).
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)

# ============================================
# Boat 配置(boat_physics.usdc;RigidObject 因无 ArticulationRootAPI)
# ============================================
ROV_CONFIG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "boat_physics.usdc"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=5.0,
            max_angular_velocity=5.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,   # PhysX 关掉,用 Python 自定义阻尼
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=100.0,
        ),
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


# ============================================
# 水下物理参数配置
# ============================================
@configclass
class UnderwaterPhysicsCfg:
    """ROV水下物理参数配置"""
    water_density: float = 1000.0
    gravity: float = 9.8
    rov_volume: float = 0.2  # boat 100kg / 1000kg/m³ = 0.1m³ 全浸没,0.2 半浸没(平衡)
    rov_height: float = 1.0   # boat 船体高度
    water_surface_z: float = 0.0
    buoyancy_center_offset: float = 0.0

    max_linear_damping: float = 50.0   # 与 calm_E7frame 同步
    max_angular_damping: float = 50.0
    air_linear_damping: float = 0.5
    air_angular_damping: float = 0.05

    enable_current: bool = False   # CALM
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


# ============================================
# 波浪物理参数配置
# ============================================
@configclass
class WavePhysicsCfg:
    enable_wave: bool = False   # CALM
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
    observation_space = int(__import__('os').environ.get('OBS_DIM', '3'))
    state_space = 0

    goal_radius: float = 3.0   # boat 5.5m 长,2m 过紧
    max_spawn_distance: float = 30.0
    min_spawn_distance: float = 10.0
    use_learned_reward: bool = False

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot_cfg: RigidObjectCfg = ROV_CONFIG
    underwater_physics_cfg: UnderwaterPhysicsCfg = UnderwaterPhysicsCfg()
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=12.0,
        replicate_physics=True
    )

    dof_names = []
