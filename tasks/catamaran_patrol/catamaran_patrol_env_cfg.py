# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
import os as _os
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)

# ── Catamaran rigid body (uses our converted catamaran.usd) ──────────────────
CATAMARAN_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "catamaran.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=8.0,   # m/s
            # BUGFIX: this field is in DEGREES per second, not rad/s. At 5.0 the hull was
            # capped at 0.087 rad/s, i.e. a 49 m turning circle at cruise speed, while the
            # patrol circuit has a 12 m radius and a 3 m goal — the task was not physically
            # solvable. MAX_TORQUE=100 N.m against angular damping 40 implies a design turn
            # rate of ~2.5 rad/s, so the cap simply must not be the binding constraint.
            max_angular_velocity=90.0,  # deg/s (~1.57 rad/s)
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,   # applied manually in env
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=120.0),
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

# ── Water physics ─────────────────────────────────────────────────────────────
@configclass
class UnderwaterPhysicsCfg:
    water_density: float = 1000.0
    gravity: float = 9.8
    rov_volume: float = 0.3        # 120 kg / 1000 = 0.12 m³ minimum; 0.3 gives positive buoyancy
    rov_height: float = 1.0        # effective height for submersion calc
    water_surface_z: float = 0.0
    max_linear_damping: float = 40.0   # catamaran is wider → more drag
    max_angular_damping: float = 40.0
    air_linear_damping: float = 0.5
    air_angular_damping: float = 0.05
    enable_current: bool = False
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0

@configclass
class WavePhysicsCfg:
    enable_wave: bool = False
    wave_height: float = 0.5
    wave_period: float = 5.0
    wave_dir_x: float = 1.0
    wave_dir_y: float = 0.0

# ── Task config ───────────────────────────────────────────────────────────────
@configclass
class CatamaranPatrolEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 120.0          # 2 min — shorter episodes → more resets → faster learning
    action_space: int = 2
    observation_space: int = int(_os.environ.get("OBS_DIM", "12"))
    state_space: int = 0

    # Patrol geometry — smaller circuit for faster early learning
    goal_radius: float = 3.0
    patrol_radius: float = 12.0              # tightened from 25m; easier early waypoint reaching
    max_spawn_distance: float = 15.0         # spawn close to circuit
    min_spawn_distance: float = 3.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot_cfg: RigidObjectCfg = CATAMARAN_CFG
    underwater_physics_cfg: UnderwaterPhysicsCfg = UnderwaterPhysicsCfg()
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=120.0,    # wider spacing — patrol routes are large
        replicate_physics=False,  # explicit USD cloning — works with referenced USDs
    )

    dof_names = []
