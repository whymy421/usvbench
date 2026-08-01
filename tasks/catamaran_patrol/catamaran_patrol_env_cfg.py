# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
import os as _os
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .._shared.vehicles import VehicleSpec, get_vehicle

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
def _build_robot_cfg(spec: VehicleSpec) -> RigidObjectCfg:
    """Build the rigid-body cfg from a registry entry."""
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_os.path.join(_ASSET_DIR, spec.usd_relpath),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                # Non-binding safety bounds only. NOTE Isaac Lab units:
                # max_linear_velocity is m/s but max_angular_velocity is DEG/s. This task
                # originally shipped 5.0 here meaning rad/s, an effective 5 deg/s yaw cap
                # -- a 49 m turning circle against a 12 m patrol circuit.
                max_linear_velocity=8.0,     # m/s
                max_angular_velocity=573.0,  # = 10 rad/s, never reached in practice
                max_depenetration_velocity=1.0,
                disable_gravity=False,
                linear_damping=0.0,   # applied manually in env
                angular_damping=0.0,
            ),
            mass_props=(
                None if spec.mass_kg is None
                else sim_utils.MassPropertiesCfg(mass=spec.mass_kg)
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


def _build_underwater_physics_cfg(spec: VehicleSpec) -> "UnderwaterPhysicsCfg":
    """Translate registry names to this task's physics interface."""
    return UnderwaterPhysicsCfg(
        rov_volume=spec.displaced_volume_m3,
        rov_height=spec.hull_height_m,
        buoyancy_center_offset=spec.buoyancy_center_offset_m,
        surge_lin_damping=spec.surge_lin,
        surge_quad_damping=spec.surge_quad,
        sway_lin_damping=spec.sway_lin,
        sway_quad_damping=spec.sway_quad,
        heave_damping=spec.heave_damping,
        yaw_lin_damping=spec.yaw_lin,
        yaw_quad_damping=spec.yaw_quad,
        restoring_stiffness_roll=spec.restoring_stiffness_roll,
        restoring_stiffness_pitch=spec.restoring_stiffness_pitch,
        rollpitch_rate_damping=spec.rollpitch_rate_damping,
    )


@configclass
class UnderwaterPhysicsCfg:
    """Hull physics. Every value is resolved from the vehicle registry in
    ``CatamaranPatrolEnvCfg.__post_init__`` -- see ``tasks/_shared/vehicles.py`` for the
    catamaran entry and the provenance of each number. The defaults here are only what a
    bare instantiation gets."""

    water_density: float = 1000.0
    gravity: float = 9.8
    water_surface_z: float = 0.0

    rov_volume: float = 0.3
    rov_height: float = 1.0
    buoyancy_center_offset: float = 0.0

    surge_lin_damping: float = 65.0
    surge_quad_damping: float = 110.0
    sway_lin_damping: float | None = 195.0
    sway_quad_damping: float | None = 330.0
    heave_damping: float = 330.0
    yaw_lin_damping: float = 385.0
    yaw_quad_damping: float = 355.0

    # Anisotropic, as the shared restoring module supports: a slender catamaran is far
    # stiffer in pitch than in roll. The single isotropic attitude_spring this task used
    # before came from the boat/ROV tasks, where it is a stabilisation device rather than
    # a hydrostatic quantity.
    restoring_stiffness_roll: float = 265.0
    restoring_stiffness_pitch: float = 2934.0
    rollpitch_rate_damping: float = 460.0

    # Thruster response time. A real propeller cannot reverse its thrust instantly: the
    # motor, the shaft inertia and the water column all take time, and a T200/M200-class
    # unit sits around 0.1-0.2 s. Without it the actuator is a perfect zero-order hold and
    # the policy is free to slam the command from one rail to the other every few steps,
    # which is exactly what it learned to do -- measured 1.72 command sign flips per second
    # and a mean |yaw rate| of 0.585 rad/s on the straight legs, against the 0.205 rad/s
    # that a steady turn round the 12 m circuit actually needs.
    #
    # This is an addition to the model, not a bug fix: the task never had actuator
    # dynamics. It is applied at sim dt so the time constant is independent of decimation.
    # 0.15 s was tried first and is too slow: it fixed the weaving (command sign flips
    # 1.72 -> 0.19 per second, matching the boat reference's 0.17) but cost 27% of the
    # score, 19.762 -> 14.344 targets/episode, with 7 of 64 envs no longer scoring at all
    # and worst-case approach drifting from 3.02 m to 5.85 m. The hull could no longer
    # correct tightly enough near the goal. 0.06 s keeps the mechanism and returns most
    # of the steering bandwidth; it is at the fast end of a real T200-class unit rather
    # than the slow end.
    thruster_tau: float = 0.06  # s, first-order lag on both action channels

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

    # Which registry entry supplies the hull. Everything vehicle-specific -- asset, mass,
    # actuator limits, damping, restoring stiffness -- is resolved from it in
    # __post_init__, the same way docking/station_keeping/path_following do it. Swapping
    # hull is then a one-line subclass, and the sea-state variants those tasks define can
    # be applied here unchanged.
    vehicle: str = "catamaran"

    # Filled from the registry; the values here are only what a bare instantiation gets.
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

    def __post_init__(self) -> None:
        """Resolve all hull-dependent config from the selected registry entry."""
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm
