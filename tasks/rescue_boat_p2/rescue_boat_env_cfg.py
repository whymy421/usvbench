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

# ── Rescue boat rigid body ────────────────────────────────────────────────────
# ARENA RIB 8.50m, 300 kg, twin outboards (~23 knots top speed)
RESCUE_BOAT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "rescue_boat.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=12.0,   # ~23 knots
            # NOTE: this is DEGREES per second (PhysX schema), not rad/s.
            # Confirmed empirically: 120 capped yaw at 2.09 rad/s (= 120 deg/s) and
            # 3.0 capped it at 0.05 rad/s (= 3 deg/s), which made the boat unable to
            # steer at all. 60 deg/s ~= 1.05 rad/s: agile for an 8.5 m RIB without
            # being unphysical.
            max_angular_velocity=60.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,    # applied manually in env (water physics)
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=300.0),
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
    # 300 kg needs 0.30 m³ merely to break even. The previous 0.35 left only 17%
    # reserve buoyancy, so the hull floated 86% submerged and could not recover
    # once heeled. 0.75 gives ~150% reserve, matching the catamaran (0.30 m³ for
    # 120 kg), which rides at 40% submerged.
    rov_volume: float = 0.75
    rov_height: float = 1.5         # effective height for submersion calc
    water_surface_z: float = 0.0
    max_linear_damping: float = 30.0   # RIB has less beam drag than catamaran
    max_angular_damping: float = 60.0  # high yaw damping for realistic steering
    air_linear_damping: float = 0.5
    air_angular_damping: float = 0.05
    enable_current: bool = False       # calm water for baseline training
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0

# ── Task config ───────────────────────────────────────────────────────────────
@configclass
class RescueBoatEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 120.0          # 2-minute episode
    action_space: int = 2
    observation_space: int = 14              # see rescue_boat_env.py for breakdown
    state_space: int = 0

    # ── Casualty parameters (V3, calibrated 20 Aug 2026) ─────────────────────
    # These were re-derived against a scripted pure-pursuit oracle rather than
    # chosen a priori. The original values gave that oracle only 0.32, i.e. the
    # 0.70 bar was unreachable by ANY controller and no training run could have
    # passed. Under the values below the oracle reaches 0.887, so 0.70 represents
    # a good-but-imperfect policy rather than a superhuman one.
    #
    # NOTE: this changes the benchmark definition and must be stated in the
    # methods and agreed with the supervisor. See PROGRESS.md.
    n_casualties: int = 4

    # 5.0 m was smaller than the hull itself (8.5 m) AND smaller than the minimum
    # turning radius (v/omega ~= 7.6 m at cruise), so an overshoot could not be
    # corrected and the vessel orbited outside the capture zone indefinitely.
    # 15 m exceeds both, and reads as an assistance range rather than contact.
    rescue_radius: float = 15.0

    casualty_timer_min: float = 40.0
    casualty_timer_max: float = 90.0

    # CONSTRAINT: spawn_min must exceed rescue_radius, or casualties start already
    # within capture distance and are rescued at t=0 with the vessel stationary.
    # A previous configuration (capture 20 m, spawn from 10 m) put 40% of
    # casualties inside the capture radius at spawn and produced a meaningless
    # rescue_rate of 1.0000.
    casualty_spawn_r_min: float = 80.0
    # SCALE: the vessel covers 1440 m in a 120 s episode. With casualties inside
    # 60 m every one is reachable ~18x over, nothing has to be given up, and the
    # task does not test prioritisation at all. At 80-200 m a four-point tour runs
    # comparable to the deadlines, so triage is forced.
    #
    # V1 used 30-120 m, which was closer to correct. It was cut to 10-50 m to
    # address "reward sparsity" that was in fact a symptom of a hull limited to
    # 0.8 m/s by the capsize defect. The workaround outlived its cause.
    casualty_spawn_r_max: float = 200.0

    # Boat spawn
    spawn_at_origin: bool = True             # boat always spawns near env origin

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot_cfg: RigidObjectCfg = RESCUE_BOAT_CFG
    underwater_physics_cfg: UnderwaterPhysicsCfg = UnderwaterPhysicsCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        # Must exceed roughly twice the furthest a vessel can roam, or boats from
        # neighbouring environments share space and can collide. With casualties
        # spawning out to 350 m under the rescaled task, 200 m is far too tight.
        # Rule of thumb: env_spacing >= 3 x casualty_spawn_r_max.
        env_spacing=1200.0,
        replicate_physics=False,
    )

    dof_names = []
