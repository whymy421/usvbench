# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for C4 x C5 ordered path following with hazards."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.obs_superset import SPEED_SCALE_MPS
from .._shared.restoring import restoring_torque_body
from .._shared.vehicles import get_vehicle
from .path_hazard_env_cfg import PathHazardEnvCfg
from .path_hazard_geometry import (
    analytic_min_clearance,
    ray_circle_ranges,
    sample_layout,
)


def _space_dim(space: object) -> int:
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class PathHazardEnv(DirectRLEnv):
    """Fixed-horizon ordered route task with forced centerline blockers."""

    cfg: PathHazardEnvCfg

    def __init__(self, cfg: PathHazardEnvCfg, render_mode: str | None = None, **kwargs):
        self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        isaac_seed = getattr(cfg, "seed", None)
        layout_seed = cfg.layout_seed if isaac_seed is None else int(isaac_seed)
        self._layout_rng = np.random.default_rng(layout_seed)
        super().__init__(cfg, render_mode, **kwargs)

        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        self.goal_radius = float(self.cfg.goal_radius)
        route_shape = (self.num_envs, self.cfg.num_waypoints, 2)
        self.waypoints = torch.zeros(route_shape, device=self.device)
        self.segment_lengths = torch.zeros(
            (self.num_envs, self.cfg.num_waypoints), device=self.device
        )
        self.route_length = torch.zeros(self.num_envs, device=self.device)

        self.obstacle_centers = torch.zeros(
            (self.num_envs, self.cfg.obstacle_count, 2), device=self.device
        )
        self.obstacle_radii = torch.zeros(
            (self.num_envs, self.cfg.obstacle_count), device=self.device
        )
        self.obstacle_active = torch.zeros(
            (self.num_envs, self.cfg.obstacle_count),
            dtype=torch.bool,
            device=self.device,
        )
        self.obstacle_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.gates_passed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self.xte_rms = torch.zeros(self.num_envs, device=self.device)
        self._min_clearance = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )

        self._success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_gate_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_prev = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_before_last_gate = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gates_passed_this_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._first_success_time_s = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self._episode_finished = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._previous_xy = self.robot.data.root_com_pos_w[:, :2].clone()
        self._prev_target_distance = torch.zeros(self.num_envs, device=self.device)
        self._pre_transition_target_distance = torch.zeros(
            self.num_envs, device=self.device
        )
        self._xte_squared_sum = torch.zeros(self.num_envs, device=self.device)
        self._xte_sample_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Last completed-episode values survive DirectRLEnv automatic reset.
        self.episode_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_to_success = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.episode_gates_passed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_path_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_min_clearance = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.episode_xte_rms = torch.zeros(self.num_envs, device=self.device)
        self.episode_route_length = torch.zeros(self.num_envs, device=self.device)
        self.actions = torch.zeros(
            (self.num_envs, _space_dim(self.cfg.action_space)), device=self.device
        )

    def _setup_scene(self) -> None:
        if self.vehicle_spec.asset_kind == "articulation":
            self.robot = Articulation(self.cfg.robot_cfg)
        else:
            self.robot = RigidObject(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.vehicle_spec.asset_kind == "articulation":
            self.scene.articulations["robot"] = self.robot
        else:
            self.scene.rigid_objects["robot"] = self.robot

        self._obstacle_translate_ops = []
        self._obstacle_radius_attrs = []
        self._obstacle_prims = []
        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")
        if self.cfg.visual.enable_obstacles:
            try:
                self._create_render_only_obstacle_markers()
            except Exception as exc:
                self._obstacle_translate_ops = []
                self._obstacle_radius_attrs = []
                self._obstacle_prims = []
                print(f"[WARN] Obstacle visualization could not be created: {exc}")

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        axis_vectors = {
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
        }
        body_yaw_offsets = {
            "+x": 0.0,
            "-x": math.pi,
            "+y": -math.pi / 2.0,
            "-y": math.pi / 2.0,
        }
        bow_axis = self.vehicle_spec.bow_body_axis
        forward_components = axis_vectors[bow_axis]
        self._fwd_x = forward_components[0]
        self._fwd_y = forward_components[1]
        self._surge_axis_idx = 0 if self._fwd_x else 1
        self._sway_axis_idx = 1 - self._surge_axis_idx
        self._body_yaw_from_bow_offset = body_yaw_offsets[bow_axis]
        self.forward_vec = torch.tensor(
            forward_components, device=self.device
        ).repeat(self.num_envs, 1)
        self._ray_angles = torch.arange(
            self.cfg.ray_count, device=self.device, dtype=torch.float32
        ) * (2.0 * torch.pi / self.cfg.ray_count)

    def _create_static_water_mesh(self) -> None:
        """Create path_following's flat render-only water mesh."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        water_path = "/World/StaticWater"
        if stage.GetPrimAtPath(water_path).IsValid():
            stage.RemovePrim(water_path)

        res = int(self.cfg.visual.water_res)
        size = float(self.cfg.visual.water_size_m)
        if res < 2 or size <= 0.0:
            raise ValueError("visual water_res must be >=2 and water_size_m positive")
        z = float(self.physics_cfg.water_surface_z)
        spacing = size / (res - 1)
        start = -0.5 * size
        points = Vt.Vec3fArray(
            [
                Gf.Vec3f(start + i * spacing, start + j * spacing, z)
                for j in range(res)
                for i in range(res)
            ]
        )
        counts = []
        indices = []
        for j in range(res - 1):
            for i in range(res - 1):
                v0 = j * res + i
                counts.append(4)
                indices.extend((v0, v0 + 1, v0 + res + 1, v0 + res))
        mesh = UsdGeom.Mesh.Define(stage, water_path)
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
        color = Gf.Vec3f(*self.cfg.visual.water_color)
        mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color] * len(points)))
        mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
        mesh.GetDoubleSidedAttr().Set(True)

    def _create_env0_waypoint_markers(self) -> None:
        """Recreate path_following's ordered-color gate annuli for env 0."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        segments = int(self.cfg.visual.waypoint_segments)
        radius = float(self.cfg.goal_radius)
        line_width = float(self.cfg.visual.waypoint_line_width_m)
        if segments < 3 or radius <= 0.0 or line_width <= 0.0:
            raise ValueError("invalid waypoint visual geometry")
        stage = omni.usd.get_context().get_stage()
        marker_root = "/World/PathHazardWaypoints"
        if stage.GetPrimAtPath(marker_root).IsValid():
            stage.RemovePrim(marker_root)

        marker_z = float(self.physics_cfg.water_surface_z) + 0.02
        outer_radius = radius + line_width
        counts = []
        indices = []
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            inner = 2 * segment
            outer = inner + 1
            next_inner = 2 * next_segment
            next_outer = next_inner + 1
            counts.extend((3, 3))
            indices.extend((inner, outer, next_outer, inner, next_outer, next_inner))
        count_array = Vt.IntArray(counts)
        index_array = Vt.IntArray(indices)
        opacities = Vt.FloatArray([1.0] * (2 * segments))
        world_waypoints = (
            self.waypoints[0] + self.scene.env_origins[0, :2]
        ).detach().cpu().tolist()

        for waypoint_index, (center_x, center_y) in enumerate(world_waypoints):
            ordered_colors = self.cfg.visual.waypoint_ordered_colors
            color_value = (
                ordered_colors[waypoint_index]
                if waypoint_index < len(ordered_colors)
                else self.cfg.visual.waypoint_color
            )
            color = Gf.Vec3f(*color_value)
            points = []
            for segment in range(segments):
                angle = 2.0 * math.pi * segment / segments
                cosine, sine = math.cos(angle), math.sin(angle)
                points.extend(
                    (
                        Gf.Vec3f(
                            center_x + radius * cosine,
                            center_y + radius * sine,
                            marker_z,
                        ),
                        Gf.Vec3f(
                            center_x + outer_radius * cosine,
                            center_y + outer_radius * sine,
                            marker_z,
                        ),
                    )
                )
            mesh = UsdGeom.Mesh.Define(stage, f"{marker_root}/Gate_{waypoint_index}")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
            mesh.GetFaceVertexCountsAttr().Set(count_array)
            mesh.GetFaceVertexIndicesAttr().Set(index_array)
            mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color] * len(points)))
            mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
            mesh.GetDisplayOpacityAttr().Set(opacities)
            mesh.GetDoubleSidedAttr().Set(True)

    def _create_env0_path_line(self) -> None:
        """Recreate path_following's render-only line for env 0."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        width = float(self.cfg.visual.path_line_width_m)
        if width <= 0.0:
            raise ValueError("visual.path_line_width_m must be positive")
        stage = omni.usd.get_context().get_stage()
        path_root = "/World/PathHazardPathLine"
        if stage.GetPrimAtPath(path_root).IsValid():
            stage.RemovePrim(path_root)
        origin = self.scene.env_origins[0, :2].detach().cpu().tolist()
        waypoints = (
            self.waypoints[0] + self.scene.env_origins[0, :2]
        ).detach().cpu().tolist()
        route_points = [origin, *waypoints]
        z = float(self.physics_cfg.water_surface_z) + 0.015
        half_width = 0.5 * width
        color = Gf.Vec3f(*self.cfg.visual.path_line_color)
        for index, (start, end) in enumerate(zip(route_points, route_points[1:])):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 0.0:
                raise ValueError("path line segments must have positive length")
            offset_x, offset_y = -dy / length * half_width, dx / length * half_width
            points = [
                Gf.Vec3f(start[0] + offset_x, start[1] + offset_y, z),
                Gf.Vec3f(end[0] + offset_x, end[1] + offset_y, z),
                Gf.Vec3f(end[0] - offset_x, end[1] - offset_y, z),
                Gf.Vec3f(start[0] - offset_x, start[1] - offset_y, z),
            ]
            mesh = UsdGeom.Mesh.Define(stage, f"{path_root}/Segment_{index}")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
            mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color] * 4))
            mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
            mesh.GetDisplayOpacityAttr().Set(Vt.FloatArray([1.0] * 4))
            mesh.GetDoubleSidedAttr().Set(True)

    @staticmethod
    def _get_or_add_translate_op(prim):
        """Reuse a cloned translate xform op or add one through generic API."""
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate)

    def _create_render_only_obstacle_markers(self) -> None:
        """Author gray cylinders without collision schemas."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        colors = Vt.Vec3fArray([Gf.Vec3f(*self.cfg.visual.obstacle_color)])
        for env_index in range(self.num_envs):
            env_ops = []
            env_radius_attrs = []
            env_prims = []
            for obstacle_index in range(self.cfg.obstacle_count):
                path = (
                    f"/World/envs/env_{env_index}/"
                    f"PathHazardObstacle_{obstacle_index}"
                )
                cylinder = UsdGeom.Cylinder.Define(stage, path)
                cylinder.GetAxisAttr().Set("Z")
                cylinder.GetHeightAttr().Set(float(self.cfg.visual.obstacle_height_m))
                cylinder.GetRadiusAttr().Set(0.5)
                cylinder.GetDisplayColorAttr().Set(colors)
                prim = cylinder.GetPrim()
                UsdGeom.Imageable(prim).MakeInvisible()
                env_ops.append(self._get_or_add_translate_op(prim))
                env_radius_attrs.append(cylinder.GetRadiusAttr())
                env_prims.append(prim)
            self._obstacle_translate_ops.append(env_ops)
            self._obstacle_radius_attrs.append(env_radius_attrs)
            self._obstacle_prims.append(env_prims)

    def _update_render_only_obstacle_markers(
        self,
        env_ids: torch.Tensor,
        local_centers: torch.Tensor,
        radii: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        if not self._obstacle_translate_ops:
            return
        try:
            from pxr import Gf, UsdGeom

            env_list = env_ids.detach().cpu().tolist()
            centers = local_centers.detach().cpu().tolist()
            radii_list = radii.detach().cpu().tolist()
            active_list = active.detach().cpu().tolist()
            for row, env_index in enumerate(env_list):
                for obstacle_index in range(self.cfg.obstacle_count):
                    prim = self._obstacle_prims[env_index][obstacle_index]
                    imageable = UsdGeom.Imageable(prim)
                    if active_list[row][obstacle_index]:
                        center = centers[row][obstacle_index]
                        self._obstacle_translate_ops[env_index][obstacle_index].Set(
                            Gf.Vec3d(center[0], center[1], 0.0)
                        )
                        self._obstacle_radius_attrs[env_index][obstacle_index].Set(
                            float(radii_list[row][obstacle_index])
                        )
                        imageable.MakeVisible()
                    else:
                        imageable.MakeInvisible()
        except Exception as exc:
            print(f"[WARN] Obstacle marker update skipped: {exc}")

    def _root_quat(self) -> torch.Tensor:
        return self.robot.data.root_link_quat_w

    def _compute_buoyancy_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.robot.data.root_pos_w
        orientations = self._root_quat()
        center_of_h = self.physics_cfg.rov_height / 2.0
        depth = self.physics_cfg.water_surface_z - positions[:, 2]
        submerged_ratio = torch.clamp(
            (depth + center_of_h) / self.physics_cfg.rov_height,
            min=0.0,
            max=1.0,
        )
        submerged_volume = self.physics_cfg.rov_volume * submerged_ratio
        magnitude = (
            self.physics_cfg.water_density
            * submerged_volume
            * self.physics_cfg.gravity
        )
        force_world = torch.zeros((self.num_envs, 3), device=self.device)
        force_world[:, 2] = magnitude
        offset_body = torch.zeros((self.num_envs, 3), device=self.device)
        offset_body[:, 2] = self.physics_cfg.buoyancy_center_offset
        offset_world = math_utils.quat_apply(orientations, offset_body)
        return force_world, torch.cross(offset_world, force_world, dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        num_bodies = self.robot.num_bodies
        forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        torques = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        thrust_action = self.actions[:, 0]
        thrust = torch.where(
            thrust_action >= 0.0,
            thrust_action * self.cfg.thrust_max_fwd,
            thrust_action * self.cfg.thrust_max_rev,
        )
        forces[:, 0, 0] = thrust * self._fwd_x
        forces[:, 0, 1] = thrust * self._fwd_y
        torques[:, 0, 2] = self.actions[:, 1] * self.cfg.yaw_torque_max

        force_world = torch.zeros_like(forces[:, 0, :])
        torque_world = torch.zeros_like(torques[:, 0, :])
        buoyancy_force, buoyancy_torque = self._compute_buoyancy_forces()
        force_world += buoyancy_force
        torque_world += buoyancy_torque
        velocity_world = self.robot.data.root_com_vel_w
        if velocity_world.shape[-1] == 6:
            linear_velocity = velocity_world[:, :3]
            angular_velocity = velocity_world[:, 3:]
        else:
            linear_velocity = velocity_world
            angular_velocity = self.robot.data.root_ang_vel_w

        quat = self._root_quat()
        if (
            self.physics_cfg.sway_lin_damping is None
            or self.physics_cfg.sway_quad_damping is None
        ):
            planar_velocity = linear_velocity[:, :2]
            planar_speed = torch.norm(planar_velocity, dim=-1, keepdim=True)
            force_world[:, :2] += -(
                self.physics_cfg.surge_lin_damping
                + self.physics_cfg.surge_quad_damping * planar_speed
            ) * planar_velocity
        else:
            velocity_body = math_utils.quat_apply_inverse(quat, linear_velocity)
            drag_body = torch.zeros_like(velocity_body)
            surge = velocity_body[:, self._surge_axis_idx]
            drag_body[:, self._surge_axis_idx] = -(
                self.physics_cfg.surge_lin_damping
                + self.physics_cfg.surge_quad_damping * torch.abs(surge)
            ) * surge
            sway = velocity_body[:, self._sway_axis_idx]
            drag_body[:, self._sway_axis_idx] = -(
                self.physics_cfg.sway_lin_damping
                + self.physics_cfg.sway_quad_damping * torch.abs(sway)
            ) * sway
            forces[:, 0, :2] += drag_body[:, :2]
        force_world[:, 2] += -self.physics_cfg.heave_damping * linear_velocity[:, 2]

        yaw_rate = angular_velocity[:, 2]
        torque_world[:, 2] += -(
            self.physics_cfg.yaw_lin_damping
            + self.physics_cfg.yaw_quad_damping * torch.abs(yaw_rate)
        ) * yaw_rate
        torque_world[:, :2] += (
            -self.physics_cfg.rollpitch_rate_damping * angular_velocity[:, :2]
        )
        if (
            self.physics_cfg.restoring_stiffness_roll
            == self.physics_cfg.restoring_stiffness_pitch
        ):
            up_body_world = math_utils.quat_apply(
                quat, self.up_dir.expand(self.num_envs, 3)
            )
            tilt_axis = torch.stack(
                (up_body_world[:, 1], -up_body_world[:, 0]), dim=-1
            )
            torque_world[:, :2] += (
                self.physics_cfg.restoring_stiffness_roll * tilt_axis
            )
        else:
            torques[:, 0, :] += restoring_torque_body(
                quat,
                self.physics_cfg.restoring_stiffness_roll,
                self.physics_cfg.restoring_stiffness_pitch,
            )

        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_world)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_world)
        self.robot.set_external_force_and_torque(forces, torques)

    def _com_xy(self) -> torch.Tensor:
        return self.robot.data.root_com_pos_w[:, :2]

    def _waypoint_world(self, waypoint_indices: torch.Tensor) -> torch.Tensor:
        env_indices = torch.arange(self.num_envs, device=self.device)
        return (
            self.waypoints[env_indices, waypoint_indices]
            + self.scene.env_origins[:, :2]
        )

    def _current_waypoint_world(self) -> torch.Tensor:
        indices = self.gates_passed.clamp(max=self.cfg.num_waypoints - 1)
        return self._waypoint_world(indices)

    def _active_segment_relative(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_indices = torch.arange(self.num_envs, device=self.device)
        indices = self.gates_passed.clamp(max=self.cfg.num_waypoints - 1)
        segment_end = self.waypoints[env_indices, indices]
        previous_indices = (indices - 1).clamp(min=0)
        previous_waypoint = self.waypoints[env_indices, previous_indices]
        segment_start = torch.where(
            (indices == 0).unsqueeze(-1),
            torch.zeros_like(previous_waypoint),
            previous_waypoint,
        )
        return segment_start, segment_end

    def _heading_and_direction(
        self, target_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative = target_xy - self._com_xy()
        distance = torch.norm(relative, dim=-1, keepdim=True).clamp_min(1.0e-6)
        direction = relative / distance
        forwards = math_utils.quat_apply(self._root_quat(), self.forward_vec)[:, :2]
        forwards = forwards / torch.norm(
            forwards, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        return forwards, direction

    def _clearance(self) -> torch.Tensor:
        return analytic_min_clearance(
            self._com_xy(),
            self.obstacle_centers,
            self.obstacle_radii,
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self.obstacle_active,
        )

    def _get_observations(self) -> dict:
        # First seven values intentionally match path_following v4b exactly.
        target_xy = self._current_waypoint_world()
        forwards, direction = self._heading_and_direction(target_xy)
        distance = torch.norm(target_xy - self._com_xy(), dim=-1, keepdim=True)
        dot = torch.sum(forwards * direction, dim=-1, keepdim=True)
        cross = (
            forwards[:, 0:1] * direction[:, 1:2]
            - forwards[:, 1:2] * direction[:, 0:1]
        )
        distance_norm = distance / self.cfg.segment_length_max
        stage_norm = self.gates_passed.float().unsqueeze(-1) / self.cfg.num_waypoints

        # Clamp fallback: at the final gate next == current. Never pad a novel
        # (1, 0, 0) triplet that can flip the learned thrust out of distribution.
        next_indices = (self.gates_passed + 1).clamp(
            max=self.cfg.num_waypoints - 1
        )
        next_target_xy = self._waypoint_world(next_indices)
        next_forwards, next_direction = self._heading_and_direction(next_target_xy)
        next_distance = torch.norm(
            next_target_xy - self._com_xy(), dim=-1, keepdim=True
        )
        next_dot = torch.sum(
            next_forwards * next_direction, dim=-1, keepdim=True
        )
        next_cross = (
            next_forwards[:, 0:1] * next_direction[:, 1:2]
            - next_forwards[:, 1:2] * next_direction[:, 0:1]
        )
        next_distance_norm = next_distance / self.cfg.segment_length_max

        left = torch.stack((-forwards[:, 1], forwards[:, 0]), dim=-1)
        cosine = torch.cos(self._ray_angles).view(1, -1, 1)
        sine = torch.sin(self._ray_angles).view(1, -1, 1)
        ray_directions = cosine * forwards.unsqueeze(1) + sine * left.unsqueeze(1)
        ranges = ray_circle_ranges(
            self._com_xy(),
            ray_directions,
            self.obstacle_centers,
            self.obstacle_radii,
            max_range_m=self.cfg.ray_max_range_m,
            active_mask=self.obstacle_active,
        )
        ranges_norm = ranges / self.cfg.ray_max_range_m
        native_observation = torch.hstack(
            (
                dot,
                cross,
                distance_norm,
                stage_norm,
                next_dot,
                next_cross,
                next_distance_norm,
                ranges_norm,
            )
        )
        if not self.cfg.emit_superset_obs:
            return {"policy": native_observation}

        zero = torch.zeros_like(distance_norm)
        speed_norm = (
            torch.norm(
                self.robot.data.root_com_vel_w[:, :2], dim=-1, keepdim=True
            )
            / SPEED_SCALE_MPS
        )
        phase_one_hot = torch.hstack((torch.ones_like(zero), zero, zero, zero))
        superset_observation = torch.hstack(
            (
                dot,
                cross,
                distance_norm,
                stage_norm,
                next_dot,
                next_cross,
                next_distance_norm,
                zero,
                zero,  # no berth alignment
                speed_norm,
                ranges_norm,
                phase_one_hot,
                zero,  # no dwell state
            )
        )
        return {"policy": superset_observation}

    def _get_rewards(self) -> torch.Tensor:
        # Exact path_following potential: raw distance delta against the gate
        # active at step start, with no per-env D0 normalization.
        progress = self._prev_target_distance - self._pre_transition_target_distance
        clearance = self._clearance()
        proximity = torch.clamp(
            (self.cfg.safe_clearance_m - clearance) / self.cfg.safe_clearance_m,
            min=0.0,
            max=1.0,
        )
        contact_now = clearance < 0.0
        contact_entry = contact_now & ~self._contact_prev
        self._contact_prev.copy_(contact_now)
        # Incentive alignment (v3): first contact kills the predicate, so it
        # must kill the income too -- otherwise grinding along a cylinder
        # still pays progress and gate bonuses on a dead episode.
        alive = (~self._contact_before_last_gate).float()
        reward = (
            self.cfg.reference_reward_progress_scale * progress * alive
            + self._gates_passed_this_step.float()
            * self.cfg.reference_reward_gate_bonus * alive
            - self.cfg.reward_clearance_scale
            * self.control_step_s
            * proximity.square()
            - self.cfg.reward_contact_entry_penalty * contact_entry.float()
            - self.cfg.reward_contact_dwell_penalty * contact_now.float()
        )

        # Re-anchor only after rewarding the step-start gate, exactly as in v4b.
        target_xy = self._current_waypoint_world()
        self._prev_target_distance.copy_(
            torch.norm(target_xy - self._com_xy(), dim=-1)
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_xy = self._com_xy()
        prefix_active = ~self._last_gate_reached
        travelled = torch.norm(current_xy - self._previous_xy, dim=-1)
        self.path_length += torch.where(
            prefix_active, travelled, torch.zeros_like(travelled)
        )
        self._previous_xy.copy_(current_xy)

        # Sample XTE against the segment active at step start and only on the
        # pre-last-gate prefix used by the trajectory certificate.
        segment_start, segment_end = self._active_segment_relative()
        segment_vector = segment_end - segment_start
        segment_direction = segment_vector / torch.norm(
            segment_vector, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        position_relative = current_xy - self.scene.env_origins[:, :2]
        from_start = position_relative - segment_start
        xte = torch.abs(
            segment_direction[:, 0] * from_start[:, 1]
            - segment_direction[:, 1] * from_start[:, 0]
        )
        self._xte_squared_sum += torch.where(
            prefix_active, xte.square(), torch.zeros_like(xte)
        )
        self._xte_sample_count += prefix_active.long()
        self.xte_rms.copy_(
            torch.sqrt(
                self._xte_squared_sum
                / self._xte_sample_count.float().clamp_min(1.0)
            )
        )

        clearance = self._clearance()
        self._min_clearance.copy_(
            torch.where(
                prefix_active,
                torch.minimum(self._min_clearance, clearance),
                self._min_clearance,
            )
        )
        contact_now = clearance < 0.0
        self._contact_before_last_gate |= prefix_active & contact_now

        target_xy = self._current_waypoint_world()
        distance = torch.norm(target_xy - current_xy, dim=-1)
        self._pre_transition_target_distance.copy_(distance)
        reached = prefix_active & (distance <= self.goal_radius)
        self._gates_passed_this_step.copy_(reached)
        self.gates_passed += reached.long()
        reached_last_now = reached & (
            self.gates_passed >= self.cfg.num_waypoints
        )
        success_now = reached_last_now & ~self._contact_before_last_gate
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(success_now, elapsed_s, self._first_success_time_s)
        )
        self._success |= success_now
        self._last_gate_reached |= reached_last_now

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._episode_finished.copy_(time_out)
        # Fixed-horizon trajectory scoring: success and contact never terminate.
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        completed_ids = env_ids[self._episode_finished[env_ids]]
        if len(completed_ids) > 0:
            success = self._success[completed_ids]
            success_time = self._first_success_time_s[completed_ids]
            self.episode_success[completed_ids] = success
            self.time_to_success[completed_ids] = success_time
            self.episode_gates_passed[completed_ids] = self.gates_passed[completed_ids]
            self.episode_path_length[completed_ids] = self.path_length[completed_ids]
            self.episode_min_clearance[completed_ids] = self._min_clearance[
                completed_ids
            ]
            self.episode_xte_rms[completed_ids] = self.xte_rms[completed_ids]
            self.episode_route_length[completed_ids] = self.route_length[completed_ids]

            self.extras.setdefault("log", {})
            self.extras["log"]["Episode/success"] = success.float().mean()
            if success.any():
                self.extras["log"]["Episode/time_to_success_s"] = success_time[
                    success
                ].mean()
            else:
                self.extras["log"]["Episode/time_to_success_s"] = torch.tensor(
                    torch.nan, device=self.device
                )
            self.extras["log"]["Episode/gates_passed"] = self.gates_passed[
                completed_ids
            ].float().mean()
            self.extras["log"]["Episode/path_length_m"] = self.path_length[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/min_clearance_m"] = self._min_clearance[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/xte_rms_m"] = self.xte_rms[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/route_length_m"] = self.route_length[
                completed_ids
            ].mean()

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        max_obstacles = self.cfg.obstacle_count
        waypoints_np = np.zeros(
            (num_resets, self.cfg.num_waypoints, 2), dtype=np.float32
        )
        segment_lengths_np = np.zeros(
            (num_resets, self.cfg.num_waypoints), dtype=np.float32
        )
        route_lengths_np = np.zeros(num_resets, dtype=np.float32)
        centers_np = np.zeros((num_resets, max_obstacles, 2), dtype=np.float32)
        radii_np = np.zeros((num_resets, max_obstacles), dtype=np.float32)
        active_np = np.zeros((num_resets, max_obstacles), dtype=np.bool_)
        counts_np = np.zeros(num_resets, dtype=np.int64)
        for row in range(num_resets):
            layout = sample_layout(
                rng=self._layout_rng,
                max_attempts=self.cfg.layout_max_attempts,
                obstacle_count=self.cfg.obstacle_count,
            )
            count = layout.obstacle_count
            waypoints_np[row] = layout.waypoints
            segment_lengths_np[row] = layout.segment_lengths
            route_lengths_np[row] = layout.route_length
            centers_np[row, :count] = layout.centers
            radii_np[row, :count] = layout.radii
            active_np[row, :count] = True
            counts_np[row] = count

        waypoints = torch.as_tensor(
            waypoints_np, device=self.device, dtype=torch.float32
        )
        segment_lengths = torch.as_tensor(
            segment_lengths_np, device=self.device, dtype=torch.float32
        )
        local_centers = torch.as_tensor(
            centers_np, device=self.device, dtype=torch.float32
        )
        radii = torch.as_tensor(radii_np, device=self.device, dtype=torch.float32)
        active = torch.as_tensor(active_np, device=self.device, dtype=torch.bool)
        origins_xy = self.scene.env_origins[env_ids, :2]
        self.waypoints[env_ids] = waypoints
        self.segment_lengths[env_ids] = segment_lengths
        self.route_length[env_ids] = torch.as_tensor(
            route_lengths_np, device=self.device
        )
        self.obstacle_centers[env_ids] = origins_xy.unsqueeze(1) + local_centers
        self.obstacle_radii[env_ids] = radii
        self.obstacle_active[env_ids] = active
        self.obstacle_count[env_ids] = torch.as_tensor(
            counts_np, device=self.device
        )

        spawn_headings = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        body_yaws = spawn_headings + self._body_yaw_from_bow_offset
        spawn_quats = math_utils.quat_from_angle_axis(
            body_yaws.unsqueeze(-1), self.up_dir
        ).reshape(num_resets, 4)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        com_offset_body = self.robot.data.com_pos_b[env_ids].reshape(num_resets, 3)
        com_offset_world = math_utils.quat_apply(spawn_quats, com_offset_body)
        root_state[:, :2] = origins_xy - com_offset_world[:, :2]
        root_state[:, 3:7] = spawn_quats
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self.gates_passed[env_ids] = 0
        self.path_length[env_ids] = 0.0
        self.xte_rms[env_ids] = 0.0
        self._xte_squared_sum[env_ids] = 0.0
        self._xte_sample_count[env_ids] = 0
        self._success[env_ids] = False
        self._last_gate_reached[env_ids] = False
        self._contact_prev[env_ids] = False
        self._contact_before_last_gate[env_ids] = False
        self._gates_passed_this_step[env_ids] = False
        self._first_success_time_s[env_ids] = torch.nan
        self._episode_finished[env_ids] = False
        self._previous_xy[env_ids] = origins_xy
        first_targets = waypoints[:, 0] + origins_xy
        first_distance = torch.norm(first_targets - origins_xy, dim=-1)
        self._prev_target_distance[env_ids] = first_distance
        self._pre_transition_target_distance[env_ids] = first_distance
        initial_clearance = analytic_min_clearance(
            origins_xy,
            self.obstacle_centers[env_ids],
            self.obstacle_radii[env_ids],
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self.obstacle_active[env_ids],
        )
        self._min_clearance[env_ids] = initial_clearance
        self._contact_before_last_gate[env_ids] = initial_clearance < 0.0
        self.actions[env_ids] = 0.0

        self._update_render_only_obstacle_markers(
            env_ids, local_centers, radii, active
        )
        if self.cfg.visual.enable_waypoint_markers and bool((env_ids == 0).any().item()):
            try:
                self._create_env0_waypoint_markers()
            except Exception as exc:
                print(f"[WARN] Waypoint visualization could not be created: {exc}")
        if self.cfg.visual.enable_path_line and bool((env_ids == 0).any().item()):
            try:
                self._create_env0_path_line()
            except Exception as exc:
                print(f"[WARN] Path-line visualization could not be created: {exc}")
