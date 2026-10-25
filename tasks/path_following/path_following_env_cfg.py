# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .._shared.vehicles import PhysicsVersionCfg, physics_version

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.sea_state import SeaStateCfg
from .._shared.vehicles import VehicleSpec, get_vehicle


# USVBench asset directory. This locates the USD; it is not a physics knob.
_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)


_DEFAULT_VEHICLE = get_vehicle("rov")


def _build_robot_cfg(spec: VehicleSpec) -> ArticulationCfg | RigidObjectCfg:
    """Build the selected asset while preserving the legacy ROV spawn config."""
    spawn_kwargs = {
        "usd_path": _os.path.join(_ASSET_DIR, spec.usd_relpath),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # Non-binding safety bounds, not vehicle-physics parameters.
            max_linear_velocity=10.0 if spec.asset_kind == "articulation" else 8.0,
            max_angular_velocity=573.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        "activate_contact_sensors": False,
    }
    if spec.mass_kg is not None:
        spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass_kg)

    if spec.asset_kind == "articulation":
        spawn_kwargs["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
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

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(**spawn_kwargs),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


# Public compatibility alias; values now resolve through the registry.
ROV_CONFIG = _build_robot_cfg(_DEFAULT_VEHICLE)


@configclass
class UnderwaterPhysicsCfg(PhysicsVersionCfg):
    """Calm-water hull dynamics populated from a :class:`VehicleSpec`."""

    water_density: float = 1000.0
    gravity: float = 9.8
    rov_volume: float = _DEFAULT_VEHICLE.displaced_volume_m3
    rov_height: float = _DEFAULT_VEHICLE.hull_height_m
    water_surface_z: float = 0.0
    buoyancy_center_offset: float = _DEFAULT_VEHICLE.buoyancy_center_offset_m

    surge_lin_damping: float = _DEFAULT_VEHICLE.surge_lin
    surge_quad_damping: float = _DEFAULT_VEHICLE.surge_quad
    sway_lin_damping: float | None = _DEFAULT_VEHICLE.sway_lin
    sway_quad_damping: float | None = _DEFAULT_VEHICLE.sway_quad
    heave_damping: float = _DEFAULT_VEHICLE.heave_damping
    yaw_lin_damping: float = _DEFAULT_VEHICLE.yaw_lin
    yaw_quad_damping: float = _DEFAULT_VEHICLE.yaw_quad
    restoring_stiffness_roll: float = _DEFAULT_VEHICLE.restoring_stiffness_roll
    restoring_stiffness_pitch: float = _DEFAULT_VEHICLE.restoring_stiffness_pitch
    rollpitch_rate_damping: float = _DEFAULT_VEHICLE.rollpitch_rate_damping

    enable_current: bool = False
    current_speed_min: float = 0.2
    current_speed_max: float = 0.3
    current_drag_coeff: float = 8.0


def _build_underwater_physics_cfg(spec: VehicleSpec) -> UnderwaterPhysicsCfg:
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
    waypoint_ordered_colors: tuple = (
        (1.0, 0.9, 0.1),
        (1.0, 0.55, 0.0),
        (1.0, 0.15, 0.0),
        (0.7, 0.0, 0.8),
    )
    waypoint_line_width_m: float = 0.35
    enable_path_line: bool = True
    path_line_width_m: float = 0.2
    path_line_color: tuple = (1.0, 1.0, 1.0)


def _build_journal_robot_cfg(usd_path: str, com_height_m: float) -> RigidObjectCfg:
    """Journal v2 hull (F3): the hashed assets_v2 USD, origin = COM, bow +x.

    Mass and inertia are the asset's authored values (D5 registry, SHA-256
    checked by journal_vehicle.verify_asset); they are not overridden here.
    The initial COM height is the calm equilibrium above the water level.
    """
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                # Non-binding safety bounds, as in the legacy rigid-object spawn above.
                max_linear_velocity=8.0,
                max_angular_velocity=573.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(com_height_m)),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


def _journal_parent_post_init(cfg) -> None:
    """Parent-cfg half of a journal path-following cfg (ANCHORS_E5 C6/C8; CONTEXT_R4 D13/D14; F3).

    Runs inside E5's ``_journal_post_init`` (cfg.journal_plan is already set,
    physics_version already 2). Resolves the v2 vessel through
    ``journal_vehicle.resolve_journal_vehicle`` BEFORE the env builds the
    asset, verifies the asset hash (fail closed), selects hydro_v2, declares
    the vessel profile for the certificate, the plan's 36-ray sensor, the
    journal observation contract (native 7 + 36 rays, no velocity channel)
    and flips the family hooks flag. Legacy fields of the physics cfg are
    unused under hydro_v2; the legacy thrust limits are set to NaN so that no
    legacy consumer can silently use them in a journal run.
    """
    from pathlib import Path as _Path

    from .._shared import journal_vehicle
    from .._shared.conditions.observation_groups import HOOKS_FLAG

    plan = getattr(cfg, "journal_plan", None)
    if plan is None or plan.task != "path_following" or plan.vessel != cfg.vehicle:
        raise ValueError("journal path-following cfgs are built by the E5 templates "
                         "(journal_runtime_cfg_v2): journal_plan must be the path_following plan of cfg.vehicle")
    options = dict(getattr(cfg, "journal_vehicle_options", None) or {})
    vehicle = journal_vehicle.resolve_journal_vehicle(cfg.vehicle, **options)
    assets_dir = _Path(__file__).resolve().parents[2] / journal_vehicle.ASSETS_DIRNAME
    usd_path = vehicle.verify_asset(assets_dir)
    cfg.robot_cfg = _build_journal_robot_cfg(str(usd_path), vehicle.calm_com_height_above_water_m)
    physics = cfg.underwater_physics_cfg
    physics.physics_version = physics_version(physics)
    physics.hydrodynamics_model = "hydro_v2"
    physics.vessel_profile = vehicle.name
    physics.vessel_profile_status = vehicle.profile_status
    physics.enable_current = False
    params = plan.params
    cfg.ray_count = int(params["ray_sensor_ray_count"])
    cfg.ray_max_range_m = float(params["ray_sensor_max_range_m"])
    cfg.observation_space = 7 + cfg.ray_count
    cfg.observation_contract = "journal_nav7_ray36"
    setattr(cfg, HOOKS_FLAG, True)
    cfg.thrust_max_fwd = float("nan")
    cfg.thrust_max_rev = float("nan")
    cfg.yaw_torque_max = float("nan")


@configclass
class PathFollowingEnvCfg(DirectRLEnvCfg):
    vehicle: str = "rov"

    decimation = 2
    episode_length_s = 120.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 7
    state_space = 0

    num_waypoints: int = 4
    segment_length_min: float = 12.0
    segment_length_max: float = 20.0
    heading_change_max_deg: float = 60.0
    goal_radius: float = 2.0
    reference_reward_progress_scale: float = 20.0
    reference_reward_gate_bonus: float = 10.0
    reference_reward_terminal_success: float = 100.0

    thrust_max_fwd: float = _DEFAULT_VEHICLE.thrust_fwd_n
    thrust_max_rev: float = _DEFAULT_VEHICLE.thrust_rev_n
    yaw_torque_max: float = _DEFAULT_VEHICLE.yaw_torque_nm

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    robot_cfg: ArticulationCfg | RigidObjectCfg = ROV_CONFIG
    underwater_physics_cfg: UnderwaterPhysicsCfg = _build_underwater_physics_cfg(
        _DEFAULT_VEHICLE
    )
    wave_cfg: WavePhysicsCfg = WavePhysicsCfg()
    visual: VisualCfg = VisualCfg()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        # Four 20 m segments may extend 80 m from an origin.
        env_spacing=180.0,
        replicate_physics=True,
    )

    dof_names = []

    def __post_init__(self) -> None:
        if getattr(self, "journal_contract_version", None) == 2:
            # Journal cfgs (E5 journal_runtime_cfg_v2 templates; F3): the vessel is a
            # v2 registry name resolved through journal_vehicle, never get_vehicle.
            base_post_init = getattr(super(), "__post_init__", None)
            if base_post_init is not None:
                base_post_init()
            _journal_parent_post_init(self)
            return
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        version = physics_version(self.underwater_physics_cfg)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
        self.underwater_physics_cfg.physics_version = version
        self.thrust_max_fwd = spec.thrust_fwd_n
        self.thrust_max_rev = spec.thrust_rev_n
        self.yaw_torque_max = spec.yaw_torque_nm


@configclass
class PathFollowingBlueBoatEnvCfg(PathFollowingEnvCfg):
    vehicle: str = "blueboat"


@configclass
class PathFollowingBlueBoatWaveEnvCfg(PathFollowingBlueBoatEnvCfg):
    """Path following x waves: the certified route exam in an irregular sea.

    BACKGROUND DISTURBANCE ONLY. Unlike the station-keeping Wave id (which
    appends 3 sea-state observation channels), this variant changes NOTHING
    about the observation, reward, termination, or route sampling of its
    parent (Isaac-USV-PathFollow-BlueBoat-Direct-v1): the JONSWAP field
    enters through the same world-frame force path as the buoyancy/drag
    wrench, and the policy is never told the sea exists. A certified
    checkpoint therefore loads zero-shot and the comparison is "same policy,
    same observation, world now has waves".

    Band: the frozen evaluation box (2026-08-26): H_s 0.30-0.60 m (the
    station-keeping wave band for this hull), T_p 2.0-2.5 s. Every corner
    passes the DNV-RP-C205 steepness admission
    (sea_state.validate_steepness_admission; worst corner Hs=0.60 m at
    Tp=2.0 s gives Sp ~ 1/10.4 < 1/7) and captures >= 95% of the analytic
    JONSWAP variance on the default 104-component 0.04-1.60 Hz band (worst
    corner 98.8% at gamma=1). The rung ladder pins Hs per-rung later via
    --set; hs_range stays the full certified band here. The certified
    station-keeping id's 1.5 s floor stays excluded: it is grandfathered
    history, not a template.

    DUAL CALM REFERENCE: enable=True with Hs -> 0 is NOT calm water -- the
    quadratic hull drag in sea_state.forces() still opposes the hull with
    orbital_drag_coeff * speed^2 (0.08 N at 0.1 m/s, 8 N at 1 m/s, 72 N at
    3 m/s; see sea_state.zero_amplitude_drag_force). Any dose ladder over
    this variant must carry BOTH a true-calm reference (enable=False) and an
    intercept rung (enable=True, Hs -> 0).
    """

    sea_state: SeaStateCfg = SeaStateCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sea_state.enable = True
        self.sea_state.hs_range = (0.30, 0.60)
        self.sea_state.tp_range = (2.0, 2.5)
