# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the C4 x C5 Path Hazard direct task."""

from __future__ import annotations

from .._shared.vehicles import PhysicsVersionCfg, physics_version

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.obs_superset import SUPERSET_DIM
from .._shared.sea_state import SeaStateCfg
from .._shared.vehicles import VehicleSpec, get_vehicle


_ASSET_DIR = _os.environ.get(
    "USVBENCH_ASSETS",
    _os.path.join(_os.path.expanduser("~"), "usvbench", "assets"),
)
_DEFAULT_VEHICLE = get_vehicle("blueboat")


def _build_robot_cfg(spec: VehicleSpec) -> ArticulationCfg | RigidObjectCfg:
    """Build a registry-selected vehicle without overriding authored mass."""
    spawn_kwargs = {
        "usd_path": _os.path.join(_ASSET_DIR, spec.usd_relpath),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
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


def _journal_vessel_names():
    """v2 registry names (journal vessels); legacy names are never journal vessels."""
    from .._shared.vehicles_v2 import VEHICLES_V2

    return VEHICLES_V2


def _build_journal_robot_cfg(vehicle) -> RigidObjectCfg:
    """v2 asset (origin = COM, bow +x, authored mass) spawned with the COM at the calm datum."""
    from .path_hazard_journal import ASSETS_V2_DIR

    spawn = sim_utils.UsdFileCfg(
        usd_path=str(ASSETS_V2_DIR / vehicle.asset_relpath),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=8.0,
            max_angular_velocity=573.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        activate_contact_sensors=False,
    )
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(vehicle.calm_com_height_above_water_m)),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        collision_group=0,
    )


@configclass
class UnderwaterPhysicsCfg(PhysicsVersionCfg):
    """Calm-water hull dynamics populated from the vehicle registry."""

    water_density: float = 1000.0
    gravity: float = 9.8
    water_surface_z: float = 0.0
    rov_volume: float = _DEFAULT_VEHICLE.displaced_volume_m3
    rov_height: float = _DEFAULT_VEHICLE.hull_height_m
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


@configclass
class JournalUnderwaterPhysicsCfg(UnderwaterPhysicsCfg):
    """Journal (contract v2) physics declarations, read by the condition layer and the E1
    certificate (cert_v3.declared_hydrodynamics_model / declared_vessel_profile). A separate
    subclass so the legacy UnderwaterPhysicsCfg field set stays exactly the C2 one. The legacy
    hull fields it inherits are not used under hydro_v2 (the D5 profile is)."""

    hydrodynamics_model: str = "hydro_v2"
    vessel_profile: str | None = None
    vessel_profile_status: str | None = None


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
class VisualCfg:
    """Render-only settings; task state never reads these fields."""

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
    enable_obstacles: bool = True
    obstacle_color: tuple = (0.35, 0.37, 0.40)
    obstacle_height_m: float = 2.0


@configclass
class PathHazardEnvCfg(DirectRLEnvCfg):
    """C4 x C5: ordered path following with forced on-line hazards."""

    vehicle: str = "blueboat"
    # Task contract version: 1 = every legacy id (byte-identical paths); the E5
    # journal templates (journal_runtime_cfg_v2.PathHazard*JournalEnvCfg) set 2.
    journal_contract_version: int = 1
    # Journal-only fields (inert at contract 1): the D7 pool, the observation
    # contract and the F0 hooks flag. Point at the D7 pool FOLDER with
    # journal_d7_pool_path or $USVBENCH_D7_POOL_DIR (RUNTIME variables, F7
    # allow-list; default tasks/path_hazard/journal_d7; the files are SHA-256
    # pinned, so the folder choice cannot change a result). journal_d7_pool is
    # a PROTECTED field (CONTEXT_R4 D12/D25) that may hold an in-memory pool
    # object (tests, programmatic callers); "" = read the pinned files from the
    # folder above. Never use journal_d7_pool to name the pool folder.
    journal_d7_pool: object = ""
    journal_d7_pool_path: str = ""
    observation_contract: str = ""
    journal_observation_hooks_wired: bool = False

    decimation = 2
    episode_length_s = 150.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 43
    state_space = 0
    emit_superset_obs: bool = False

    num_waypoints: int = 4
    segment_length_min: float = 12.0
    segment_length_max: float = 20.0
    heading_change_max_deg: float = 60.0
    goal_radius: float = 2.0

    obstacle_count: int = 6
    required_on_line_blockers: int = 3
    # Tier-regrade probe: pins 1..4 on-line blockers at evaluation time
    # (0 = the frozen v1 value of three, byte-identical). A separate knob so
    # the v1 validator below keeps protecting the certified field itself.
    on_line_blockers_override: int = 0
    obstacle_radius_min_m: float = 0.8
    obstacle_radius_max_m: float = 1.5
    half_beam_m: float = 0.45
    collision_margin_m: float = 0.20
    gate_clear_radius_m: float = 3.0
    layout_max_attempts: int = 20
    layout_seed: int = 0

    ray_count: int = 36
    ray_max_range_m: float = 30.0
    safe_clearance_m: float = 0.6  # v2: no distant barrier tax while threading 2 m gates

    reference_reward_progress_scale: float = 20.0
    reference_reward_gate_bonus: float = 25.0  # v2: threading past on-line blockers must outweigh one contact entry
    # v6a: quadratic graze tax retired (0.0) but KEPT for the shape-ablation
    # arm -- reward_clearance_scale=1.0 + reward_prox_scale=0.0 restores the
    # exact v5 reward path.
    reward_clearance_scale: float = 0.0
    # v6a: per-ray log proximity field, band [0.45, 0.6 + 0.45 = 1.05] m.
    # w=5.7 matches v5's at-touch cost averaged over ray phase at r=1.4 m.
    # Two-flank summation inside scatter pairs is deliberate (gap centering);
    # watched by kill criterion K3-P.
    reward_prox_scale: float = 5.7
    prox_ray_floor_m: float = 0.45
    reward_contact_entry_penalty: float = 25.0
    reward_contact_dwell_penalty: float = 1.0

    # --- v11 recipe (co-advisor's HazardNav v3), OFF by default ---------------
    # Kinematic observability + a terminal anchor at the last gate + outcome
    # termination. These only take effect on the dedicated v2 gym id.
    obs_kinematic: bool = False
    yaw_rate_obs_scale_rad_s: float = 1.0
    reward_goal_entry_bonus: float = 0.0
    reward_goal_time_bonus: float = 0.0
    reward_reverse_action_scale: float = 0.0
    reward_swift_scale: float = 0.0
    terminate_on_outcome: bool = False

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
    robot_cfg: ArticulationCfg | RigidObjectCfg = _build_robot_cfg(_DEFAULT_VEHICLE)
    underwater_physics_cfg: UnderwaterPhysicsCfg = _build_underwater_physics_cfg(
        _DEFAULT_VEHICLE
    )
    visual: VisualCfg = VisualCfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=180.0,
        replicate_physics=True,
    )
    dof_names = []

    def __post_init__(self) -> None:
        if getattr(self, "journal_contract_version", 1) == 2 and self.vehicle in _journal_vessel_names():
            # ANCHORS_E5 C6/C8: a v2 registry vessel on a journal cfg; the
            # legacy BlueBoat-only asserts below do not apply (they stay in
            # force for every legacy cfg, unchanged).
            self._journal_post_init()
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

        if self.vehicle != "blueboat":
            raise ValueError("PathHazard v1 fixes vehicle='blueboat'")
        if self.episode_length_s != 150.0 or self.sim.dt * self.decimation != 1 / 60:
            raise ValueError("PathHazard v1 fixes 150 s at a 60 Hz control rate")
        if self.num_waypoints != 4:
            raise ValueError("PathHazard v1 requires four ordered gates")
        if (
            self.segment_length_min != 12.0
            or self.segment_length_max != 20.0
            or self.heading_change_max_deg != 60.0
            or self.goal_radius != 2.0
        ):
            raise ValueError(
                "PathHazard v1 fixes 12-20 m segments, +/-60 deg turns, "
                "and 2.0 m gates"
            )
        if self.obstacle_count != 6 or self.required_on_line_blockers != 3:
            raise ValueError("PathHazard v1 fixes K=6 with three on-line blockers")
        if (
            self.obstacle_radius_min_m != 0.8
            or self.obstacle_radius_max_m != 1.5
            or self.half_beam_m != 0.45
            or self.collision_margin_m != 0.20
            or self.gate_clear_radius_m != 3.0
            or self.layout_max_attempts != 20
        ):
            raise ValueError(
                "PathHazard v1 fixes radii, hull inflation, gate disks, and "
                "the 20-attempt layout budget"
            )
        expected_observation_space = (
            SUPERSET_DIM if self.emit_superset_obs else 7 + self.ray_count
        )
        if self.obs_kinematic:
            expected_observation_space += 3
        if self.ray_count != 36 or self.observation_space != expected_observation_space:
            raise ValueError(
                "PathHazard v1 requires 36 rays and observation_space="
                f"{expected_observation_space} when "
                f"emit_superset_obs={self.emit_superset_obs}"
            )


    def _journal_post_init(self) -> None:
        """Parent-cfg edits of a journal cfg (CONTEXT_R4 F-common item 6; ANCHORS_E5 C6/C8).

        Runs inside the E5 template's __post_init__ (journal_runtime_cfg_v2
        _journal_post_init), which has already bound ``journal_plan`` and
        selected physics v2; the template re-validates afterwards
        (journal_validate). Every task number comes from the plan (E5), every
        vessel number from the D5 registry through journal_vehicle (E2).
        """
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()
        from .._shared import journal_cfg_v2 as plans
        from .._shared.journal_vehicle import resolve_journal_vehicle
        from .path_hazard_journal import DISTANCE_NORM_FACTOR, JOURNAL_TASK, LAYOUT_MAX_ATTEMPTS

        plan = getattr(self, "journal_plan", None)
        if getattr(self, "journal_task", None) != JOURNAL_TASK or plan is None or plan.task != JOURNAL_TASK:
            raise ValueError("a PathHazard journal cfg needs journal_task='path_hazard' and its E5 journal_plan")
        plan.assert_vehicle_override(self.vehicle)
        vehicle = resolve_journal_vehicle(self.vehicle, **dict(getattr(self, "journal_vehicle_options", None) or {}))
        params = plan.params
        scale = float(params["length_scale"])
        record = plans.vessel_record(self.vehicle)
        self.robot_cfg = _build_journal_robot_cfg(vehicle)
        physics = JournalUnderwaterPhysicsCfg()
        physics.physics_version = plans.PHYSICS_VERSION
        physics.hydrodynamics_model = "hydro_v2"
        physics.vessel_profile = vehicle.name
        physics.vessel_profile_status = vehicle.profile_status
        self.underwater_physics_cfg = physics
        self.num_waypoints = int(params["segment_count"])
        self.segment_length_min = float(params["segment_length_m"])
        self.segment_length_max = DISTANCE_NORM_FACTOR * float(params["segment_length_m"])
        self.heading_change_max_deg = float(params["heading_change_deg"])
        self.goal_radius = float(params["gate_radius_m"])
        self.obstacle_count = int(plans.STATIC_CAPACITY[JOURNAL_TASK])
        self.required_on_line_blockers = int(params["blocker_count"])
        self.on_line_blockers_override = 0
        self.obstacle_radius_min_m = float(params["radius_range_m"][0])
        self.obstacle_radius_max_m = float(params["radius_range_m"][1])
        self.half_beam_m = 0.5 * float(record.beam_m)
        self.gate_clear_radius_m = float(params["gate_clear_radius_m"])
        self.layout_max_attempts = LAYOUT_MAX_ATTEMPTS
        self.ray_count = int(params["ray_sensor_ray_count"])
        self.ray_max_range_m = float(params["ray_sensor_max_range_m"])
        self.safe_clearance_m = 0.6 * scale
        self.prox_ray_floor_m = self.half_beam_m
        self.episode_length_s = float(plan.max_deadline_s)
        self.terminate_on_outcome = True
        self.observation_contract = "journal_kin46_ray36"
        self.journal_observation_hooks_wired = True
        self.scene.env_spacing = 180.0 * scale
        self.visual.water_size_m = 300.0 * scale
        if self.emit_superset_obs or not self.obs_kinematic:
            raise ValueError("the journal path-hazard observation is the kinematic46 layout (no superset)")
        self.observation_space = 7 + self.ray_count + 3
        if self.ray_count != 36 or self.decimation != 2 or abs(self.sim.dt - 1.0 / 120.0) > 1e-15:
            raise ValueError("journal path hazard keeps 36 rays, dt 1/120 s and decimation 2 (CONTEXT_R4 D2/D14)")


@configclass
class PathHazardV2EnvCfg(PathHazardEnvCfg):
    """Line x hazard on the v11 recipe (the recipe that fixed dock x current).

    Adds, relative to v1: body-frame surge/sway/yaw-rate channels, a terminal
    bonus at the last gate scaled by remaining time, reverse and idling costs,
    and termination on outcome. Registered under its own gym id so the
    certified v1 champion and its numbers stay untouched.
    """

    obs_kinematic: bool = True
    observation_space = 46
    reward_goal_entry_bonus: float = 50.0
    reward_goal_time_bonus: float = 50.0
    reward_reverse_action_scale: float = 0.05
    reward_swift_scale: float = 0.25
    terminate_on_outcome: bool = True


@configclass
class PathHazardWaveEnvCfg(PathHazardEnvCfg):
    """Path hazard x waves: the certified line-avoid exam in an irregular sea.

    BACKGROUND DISTURBANCE ONLY. Unlike the station-keeping Wave id (which
    appends 3 sea-state observation channels), this variant changes NOTHING
    about the observation, reward, termination, or layout sampling of its
    parent (Isaac-USV-PathHazard-Direct-v1): the JONSWAP field enters through
    the same world-frame force path as the buoyancy/drag wrench, and the
    policy is never told the sea exists. A certified checkpoint therefore
    loads zero-shot and the comparison is "same policy, same observation,
    world now has waves".

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
