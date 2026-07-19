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


# USVBench asset directory. The training scripts normally set this explicitly.
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)


BLUEBOAT_CONFIG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BlueBoat",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_os.path.join(_ASSET_DIR, "blueboat_physics.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Non-binding safety bounds, comfortably above the calibrated
            # ~3.0 m/s surge and ~1.6 rad/s yaw terminal speeds.
            max_linear_velocity=8.0,
            max_angular_velocity=573.0,  # Isaac Lab expects deg/s; 573 ~= 10 rad/s.
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            # Disable PhysX damping: the calibrated hydrodynamic model below is
            # the sole damping source, avoiding hidden double counting.
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        # Blue Robotics BlueBoat CAD/USD source: the 17.3 kg two-battery mass,
        # center of mass, and inertia are already authored on /World/BlueBoat
        # through MassAPI. Do not pass a spawn-time mass-properties override.
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
class BlueBoatUnderwaterPhysicsCfg:
    """Calibrated calm-water physics for the displacement catamaran."""

    water_density: float = 1000.0
    gravity: float = 9.8

    # Blue Robotics BlueBoat Technical Details/CAD: 17.3 kg in the two-battery
    # condition. The source task's linear submersion model reaches equilibrium
    # at 50% submerged, hence V_model = 2*m/rho = 0.0346 m^3.
    rov_volume: float = 0.0346
    # Blue Robotics BlueBoat CAD mapped bounds: overall hull height is 0.376 m.
    rov_height: float = 0.376
    water_surface_z: float = 0.0
    # Estimated below-COM buoyancy-center distance, scaled from the CAD hull
    # geometry. This remains an uncertainty pending an immersion/sysid test.
    buoyancy_center_offset: float = -0.05

    # Two-anchor surge calibration:
    #   (9.1 + 5.9*3)*3 = 80.4 N at the datasheet 3.0 m/s terminal speed;
    #   (9.1 + 5.9*1)*1 = 15 N at 1 m/s, from 30 W electrical cruise and an
    #   assumed ~50% propulsive efficiency (explicit uncertainty).
    surge_lin_damping: float = 9.1
    surge_quad_damping: float = 5.9
    # BlueBoat CAD: displacement catamaran with 0.7214 m hull spacing. Lateral
    # bluffness is approximated as ~3x surge pending sway system identification.
    sway_lin_damping: float = 27.0
    sway_quad_damping: float = 18.0
    heave_damping: float = 200.0
    # Plausible-order yaw estimate: (4 + 6|r|)r balances ~23 N*m near 1.6 rad/s;
    # replace with measured coefficients when yaw system identification exists.
    yaw_lin_damping: float = 4.0
    yaw_quad_damping: float = 6.0

    # BlueBoat CAD hydrostatics at equilibrium displaced volume 0.0173 m^3:
    # rho*g*V_disp*GM gives ~280 N*m/rad for GM_T=1.64 m and ~141 for
    # GM_L=0.83 m. Applied through tasks/docking/restoring.py.
    restoring_stiffness_roll: float = 280.0
    restoring_stiffness_pitch: float = 141.0
    # Source boat used c=2000 at k=5000. Scaling c with sqrt(k/5000) keeps
    # damping ratio zeta approximately constant: 2000*sqrt(280/5000) ~= 473;
    # 470 is the working estimate pending roll/pitch system identification.
    rollpitch_rate_damping: float = 470.0

    enable_current: bool = False  # Calm-water benchmark.
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


@configclass
class WavePhysicsCfg:
    enable_wave: bool = False  # Calm-water benchmark.
    wave_height: float = 0.5
    wave_period: float = 5.0
    wave_dir_x: float = 1.0
    wave_dir_y: float = 0.0


@configclass
class BlueBoatCalmNavEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 120.0

    action_space = 2
    observation_space = int(_os.environ.get("OBS_DIM", "3"))
    state_space = 0

    # Blue Robotics BlueBoat Technical Details: two-battery operating mass.
    # Informational only; physics reads the authored MassAPI value from the USD.
    vehicle_mass_kg: float = 17.3

    goal_radius: float = 3.0
    # Keep the WAM-V task's 10--30 m target range: although BlueBoat's official
    # top speed is 3.0 m/s versus the source task's calibrated 2.0 m/s, these
    # distances remain appropriate for a controlled hull-swap comparison.
    max_spawn_distance: float = 30.0
    min_spawn_distance: float = 10.0
    use_learned_reward: bool = False

    # Blue Robotics M200 datasheet, two motors. Positive policy thrust is total
    # bow-first thrust; reverse is ~60% due to propeller asymmetry.
    thrust_max_fwd: float = 80.0
    thrust_max_rev: float = 48.0
    # BlueBoat CAD hull/thruster spacing is 0.7214 m. At a 0.3607 m lever arm,
    # one motor forward (40.2 N) and the other reverse (24 N) produce
    # 0.3607*(40.2+24) = 23.16 N*m, represented by the rounded 23 N*m limit.
    hull_spacing: float = 0.7214
    yaw_torque_max: float = 23.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot_cfg: RigidObjectCfg = BLUEBOAT_CONFIG
    underwater_physics_cfg: BlueBoatUnderwaterPhysicsCfg = (
        BlueBoatUnderwaterPhysicsCfg()
    )
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=12.0,
        replicate_physics=True,
    )

    dof_names = []
