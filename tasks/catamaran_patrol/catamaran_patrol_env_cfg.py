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
            # Non-binding safety bounds only: >=2x the terminal values that come out of
            # the thrust/drag balance below (2.5 m/s surge, 1.0 rad/s yaw).
            # NOTE Isaac Lab units: max_linear_velocity is m/s, max_angular_velocity is
            # DEG/s. This task originally shipped 5.0 here meaning rad/s, which is an
            # effective 5 deg/s yaw cap — a 49 m turning circle against a 12 m patrol
            # circuit, so the task was not solvable at all.
            max_linear_velocity=8.0,    # m/s
            max_angular_velocity=573.0,  # = 10 rad/s, never reached in practice
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

    # Hydrodynamic damping: per-DOF linear+quadratic, derived exactly the way
    # boat_calm_nav derives its own — Froude-scaled from the VRX WAM-V shipped
    # coefficients (xU=100, xUU=150, yV=100, yVV=100, zW=500, nR=800, nRR=800), with
    # linear x lambda^2.5, quadratic x lambda^2, yaw linear x lambda^4.5, yaw quadratic
    # x lambda^5. The boat uses lambda=0.8 for its 100 kg hull, which implies a 195 kg
    # WAM-V reference; this 120 kg hull therefore gives lambda=(120/195)^(1/3)=0.850.
    #   surge terminal: (65 + 110v)v  = 850 N   -> 2.50 m/s
    #   yaw terminal:   (385 + 355r)r = 740 N.m -> 1.00 rad/s
    # Hull drag is applied in BODY frame (surge and sway are separate).
    #
    # CAVEAT, deliberately written down rather than hidden: this scaling is anchored on
    # mass, following the boat. By mass the catamaran is the *larger* vessel
    # (lambda 0.850 vs 0.800), but its hull is 3.0 m against the boat's 5 m — it is
    # heavier and shorter. Mass-anchored Froude scaling therefore overstates its length
    # scale, and no catamaran hull-form correction is applied at all: twin slender demi-
    # hulls have lower wave-making but more wetted area, and their beam separation should
    # raise sway and yaw damping well above a monohull's. Treat these as a documented
    # first approximation, not as measured hydrodynamics.
    surge_lin_damping: float = 65.0    # N·s/m    (body-X, along the hulls)
    surge_quad_damping: float = 110.0  # N·s²/m²
    sway_lin_damping: float = 65.0     # N·s/m    (body-Y)
    # NOT the Froude-scaled value (70). The VRX coefficients give this hull less
    # resistance sideways than forwards, which is unphysical for any displacement hull:
    # the lateral underwater area is several times the frontal area and it meets the flow
    # bluff-on. Left at 70 the vessel skated through its turns — measured drift angle
    # between velocity and bow averaged 17.4 deg with a p95 of 35 deg, and sway speed
    # reached 1.53 m/s against a 2.37 m/s surge. A hull in a steady turn sits near 5-10.
    #
    # Replaced with the standard crossflow-drag estimate, 0.5*rho*Cd*A_lateral with
    # Cd = 1.0 and A_lateral = L*T = 3.0 m x 0.400 m draft (the measured equilibrium
    # draft) = 1.20 m^2, giving 600. That puts sway/surge quadratic at 5.5, inside the
    # 3-10 band real hulls sit in.
    #
    # The same method applied to the other two axes is what says the problem is specific
    # to sway rather than the scaling as a whole: it gives 48 for surge against the 110
    # in use (same order, and lower, so the method is not simply inflating everything)
    # and 506 for yaw against 355 (a factor of 1.4, inside the model's uncertainty).
    # Yaw is therefore left alone — raising it would move the terminal yaw rate and hence
    # the turning radius, which is the open sizing decision documented in FIXES.md.
    sway_quad_damping: float = 600.0   # N·s²/m²  crossflow, not Froude-scaled
    heave_damping: float = 330.0       # N·s/m    (zeta = 0.28 against the buoyancy spring)
    yaw_lin_damping: float = 385.0     # N·m·s/rad
    yaw_quad_damping: float = 355.0    # N·m·s²/rad²
    # Roll/pitch are not task DOFs: stiff spring + overdamping, same treatment and same
    # values as both reference tasks.
    attitude_spring: float = 5000.0         # N·m/rad
    rollpitch_rate_damping: float = 2000.0  # N·m·s/rad

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

    # Actuator limits. Actions are hard-clipped to [-1,1] in _pre_physics_step; these
    # constants are the single source of truth (they used to be module-level MAX_THRUST /
    # MAX_TORQUE in the env file). Sized against the damping above:
    #   850 N  -> 2.50 m/s terminal surge. Thrust-to-weight 0.72, deliberately above the
    #            boat reference's 0.51, because this task is a *fast* patrol.
    #   740 N.m -> 1.00 rad/s terminal yaw, i.e. a 2.5 m turning radius at cruise — the
    #            same turning radius the boat reference achieves, and comfortably inside
    #            the 12 m patrol circuit.
    # Reverse thrust keeps the boat's 0.4 forward/reverse ratio (VRX classic thruster).
    thrust_max_fwd: float = 850.0   # N,   at action[0] = +1 (bow-first)
    thrust_max_rev: float = 340.0   # N,   at action[0] = -1 (astern)
    yaw_torque_max: float = 740.0   # N·m, at action[1] = ±1

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
