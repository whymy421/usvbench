# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the C4 x C5 Path Hazard direct task."""

from __future__ import annotations

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .._shared.obs_superset import SUPERSET_DIM
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


@configclass
class UnderwaterPhysicsCfg:
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
        base_post_init = getattr(super(), "__post_init__", None)
        if base_post_init is not None:
            base_post_init()

        spec = get_vehicle(self.vehicle)
        self.robot_cfg = _build_robot_cfg(spec)
        self.underwater_physics_cfg = _build_underwater_physics_cfg(spec)
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
