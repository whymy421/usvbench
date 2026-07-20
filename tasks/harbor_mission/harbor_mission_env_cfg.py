# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Task B: Ordered Harbor Mission."""

from __future__ import annotations

import os as _os

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

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
    """Render-only layer; no field is read by task physics or scoring."""

    enable_water: bool = True
    water_size_m: float = 1000.0
    water_res: int = 80
    water_color: tuple = (0.1, 0.3, 0.8)
    enable_gates: bool = True
    gate_post_radius_m: float = 0.20
    gate_post_height_m: float = 2.0
    gate_crossbar_thickness_m: float = 0.16
    exit_gate_color: tuple = (0.1, 0.85, 1.0)
    field_exit_gate_color: tuple = (0.2, 1.0, 0.35)
    enable_obstacles: bool = True
    obstacle_color: tuple = (0.35, 0.37, 0.40)
    obstacle_height_m: float = 2.0
    enable_berth: bool = True
    berth_ring_segments: int = 64
    berth_ring_line_width_m: float = 0.35
    berth_ring_color: tuple = (1.0, 0.15, 0.0)
    berth_arrow_length_m: float = 3.0
    berth_arrow_width_m: float = 0.5
    berth_arrow_color: tuple = (1.0, 0.85, 0.0)
    berth_box_length_m: float = 6.5
    berth_box_width_m: float = 3.2
    berth_box_line_width_m: float = 0.25
    berth_box_color: tuple = (1.0, 1.0, 1.0)


@configclass
class HarborMissionEnvCfg(DirectRLEnvCfg):
    """Adversarially-converged Task B: Ordered Harbor Mission."""

    vehicle: str = "blueboat"

    decimation = 2
    episode_length_s = 240.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = 46
    state_space = 0

    goal_radius: float = 2.5
    success_position_tolerance_m: float = 2.5
    success_heading_tolerance_deg: float = 15.0
    success_speed_tolerance_mps: float = 0.3
    required_hold_time_s: float = 5.0

    gate_half_width_m: float = 3.0
    gate_approach_distance_m: float = 1.0
    gate_min_normal_speed_mps: float = 0.2

    half_beam_m: float = 0.45
    safe_clearance_m: float = 0.90
    ray_count: int = 36
    ray_max_range_m: float = 30.0
    max_obstacles: int = 10
    layout_max_attempts: int = 50
    hazard_layout_max_attempts: int = 20
    layout_seed: int = 0

    reward_progress_scale: float = 20.0
    reward_clearance_scale: float = 1.0
    reward_contact_entry_penalty: float = 25.0
    reward_contact_dwell_penalty: float = 1.0
    reward_dock_conjunction: float = 3.0

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=112.0,
        replicate_physics=True,
    )
    visual: VisualCfg = VisualCfg()
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

        if self.observation_space != 3 + 36 + 4 + 2 + 1:
            raise ValueError("HarborMission requires observation_space=46")
        if self.ray_count != 36:
            raise ValueError("HarborMission v1 requires exactly 36 rays")
        if self.max_obstacles < 10:
            raise ValueError("max_obstacles must accommodate the specified 6--10 range")
        if self.goal_radius != self.success_position_tolerance_m:
            raise ValueError("goal_radius must alias the docking position tolerance")
