# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .path_following_env_cfg import PathFollowingEnvCfg


def _space_dim(space: object) -> int:
    """Return the leading dimension of a Gym space or a legacy integer size."""
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class PathFollowingEnv(DirectRLEnv):
    """Calm-water ordered waypoint following with reward-independent success."""

    cfg: PathFollowingEnvCfg

    def __init__(self, cfg: PathFollowingEnvCfg, render_mode: str | None = None, **kwargs):
        self.physics_cfg = cfg.underwater_physics_cfg
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.num_waypoints != 4:
            raise ValueError("PathFollow v1 requires exactly four waypoint segments")
        if self.cfg.segment_length_min != 12.0 or self.cfg.segment_length_max != 20.0:
            raise ValueError("PathFollow v1 segment lengths are fixed to [12, 20] m")
        if self.cfg.heading_change_max_deg != 60.0:
            raise ValueError("PathFollow v1 heading changes are fixed to +/-60 degrees")

        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        shape = (self.num_envs, self.cfg.num_waypoints, 2)

        # Waypoints remain explicitly env-origin-relative. World-space targets are
        # formed only at lookup time, which also makes procedural paths inspectable.
        self.waypoints = torch.zeros(shape, device=self.device)
        self.segment_lengths = torch.zeros(
            (self.num_envs, self.cfg.num_waypoints), device=self.device
        )
        self.route_length = torch.zeros(self.num_envs, device=self.device)

        self.gates_passed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self.xte_rms = torch.zeros(self.num_envs, device=self.device)

        # Last completed-episode values survive DirectRLEnv's automatic reset.
        self.episode_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_to_success = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.episode_path_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_gates_passed = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_xte_rms = torch.zeros(self.num_envs, device=self.device)
        self.episode_route_length = torch.zeros(self.num_envs, device=self.device)

        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gates_passed_this_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._xte_squared_sum = torch.zeros(self.num_envs, device=self.device)
        self._xte_sample_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._previous_xy = self.robot.data.root_pos_w[:, :2].clone()
        self._prev_target_distance = torch.zeros(self.num_envs, device=self.device)
        self._pre_transition_target_distance = torch.zeros(
            self.num_envs, device=self.device
        )
        self.actions = torch.zeros(
            (self.num_envs, _space_dim(self.cfg.action_space)), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.forward_vec = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

    def _create_static_water_mesh(self) -> None:
        """Create the station-keeping-style flat render-only water mesh."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        water_path = "/World/StaticWater"
        if stage.GetPrimAtPath(water_path).IsValid():
            stage.RemovePrim(water_path)

        res = int(self.cfg.visual.water_res)
        size = float(self.cfg.visual.water_size_m)
        if res < 2:
            raise ValueError("visual.water_res must be at least 2")
        if size <= 0.0:
            raise ValueError("visual.water_size_m must be positive")

        surface_z = float(self.physics_cfg.water_surface_z)
        spacing = size / (res - 1)
        start = -size / 2.0
        points = Vt.Vec3fArray(
            [
                Gf.Vec3f(start + i * spacing, start + j * spacing, surface_z)
                for j in range(res)
                for i in range(res)
            ]
        )

        face_counts = []
        face_indices = []
        for j in range(res - 1):
            for i in range(res - 1):
                v0 = j * res + i
                v1 = v0 + 1
                v2 = (j + 1) * res + i + 1
                v3 = (j + 1) * res + i
                face_counts.append(4)
                face_indices.extend((v0, v1, v2, v3))

        mesh = UsdGeom.Mesh.Define(stage, water_path)
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))

        color = Gf.Vec3f(*self.cfg.visual.water_color)
        vertex_count = res * res
        mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color] * vertex_count))
        mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
        mesh.GetDisplayOpacityAttr().Set(Vt.FloatArray([1.0] * vertex_count))
        mesh.GetDoubleSidedAttr().Set(True)

    def _create_env0_waypoint_markers(self) -> None:
        """Recreate four render-only annuli for env 0's current path."""
        import math

        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        segments = int(self.cfg.visual.waypoint_segments)
        radius = float(self.cfg.goal_radius)
        line_width = float(self.cfg.visual.waypoint_line_width_m)
        if segments < 3:
            raise ValueError("visual.waypoint_segments must be at least 3")
        if radius <= 0.0:
            raise ValueError("goal_radius must be positive to render waypoint markers")
        if line_width <= 0.0:
            raise ValueError("visual.waypoint_line_width_m must be positive")

        stage = omni.usd.get_context().get_stage()
        marker_root = "/World/PathFollowingWaypoints"
        if stage.GetPrimAtPath(marker_root).IsValid():
            stage.RemovePrim(marker_root)

        marker_z = float(self.physics_cfg.water_surface_z) + 0.02
        outer_radius = radius + line_width
        face_counts = []
        face_indices = []
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            inner = 2 * segment
            outer = inner + 1
            next_inner = 2 * next_segment
            next_outer = next_inner + 1
            face_counts.extend((3, 3))
            face_indices.extend(
                (inner, outer, next_outer, inner, next_outer, next_inner)
            )

        count_array = Vt.IntArray(face_counts)
        index_array = Vt.IntArray(face_indices)
        opacities = Vt.FloatArray([1.0] * (2 * segments))
        ordered_colors = self.cfg.visual.waypoint_ordered_colors
        world_waypoints = (
            self.waypoints[0] + self.scene.env_origins[0, :2]
        ).detach().cpu().tolist()

        for waypoint_index, (center_x, center_y) in enumerate(world_waypoints):
            color_value = (
                ordered_colors[waypoint_index]
                if waypoint_index < len(ordered_colors)
                else self.cfg.visual.waypoint_color
            )
            color = Gf.Vec3f(*color_value)
            colors = Vt.Vec3fArray([color] * (2 * segments))
            points = []
            for segment in range(segments):
                angle = 2.0 * math.pi * segment / segments
                cos_angle = math.cos(angle)
                sin_angle = math.sin(angle)
                points.extend(
                    (
                        Gf.Vec3f(
                            center_x + radius * cos_angle,
                            center_y + radius * sin_angle,
                            marker_z,
                        ),
                        Gf.Vec3f(
                            center_x + outer_radius * cos_angle,
                            center_y + outer_radius * sin_angle,
                            marker_z,
                        ),
                    )
                )

            mesh = UsdGeom.Mesh.Define(stage, f"{marker_root}/Gate_{waypoint_index}")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
            mesh.GetFaceVertexCountsAttr().Set(count_array)
            mesh.GetFaceVertexIndicesAttr().Set(index_array)
            mesh.GetDisplayColorAttr().Set(colors)
            mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
            mesh.GetDisplayOpacityAttr().Set(opacities)
            mesh.GetDoubleSidedAttr().Set(True)

    def _create_env0_path_line(self) -> None:
        """Recreate the render-only route from env 0's spawn origin to its gates."""
        import math

        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        line_width = float(self.cfg.visual.path_line_width_m)
        if line_width <= 0.0:
            raise ValueError("visual.path_line_width_m must be positive")

        stage = omni.usd.get_context().get_stage()
        path_root = "/World/PathFollowingPathLine"
        if stage.GetPrimAtPath(path_root).IsValid():
            stage.RemovePrim(path_root)

        origin_xy = self.scene.env_origins[0, :2].detach().cpu().tolist()
        waypoint_xy = (
            self.waypoints[0] + self.scene.env_origins[0, :2]
        ).detach().cpu().tolist()
        route_points = [origin_xy, *waypoint_xy]
        marker_z = float(self.physics_cfg.water_surface_z) + 0.015
        half_width = line_width / 2.0
        color = Gf.Vec3f(*self.cfg.visual.path_line_color)

        for segment_index, (start, end) in enumerate(
            zip(route_points, route_points[1:])
        ):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length <= 0.0:
                raise ValueError("path line segments must have positive length")
            offset_x = -dy / segment_length * half_width
            offset_y = dx / segment_length * half_width
            points = [
                Gf.Vec3f(start[0] + offset_x, start[1] + offset_y, marker_z),
                Gf.Vec3f(end[0] + offset_x, end[1] + offset_y, marker_z),
                Gf.Vec3f(end[0] - offset_x, end[1] - offset_y, marker_z),
                Gf.Vec3f(start[0] - offset_x, start[1] - offset_y, marker_z),
            ]
            mesh = UsdGeom.Mesh.Define(stage, f"{path_root}/Segment_{segment_index}")
            mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
            mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color] * 4))
            mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
            mesh.GetDisplayOpacityAttr().Set(Vt.FloatArray([1.0] * 4))
            mesh.GetDoubleSidedAttr().Set(True)

    # The following dynamics block deliberately matches tasks/rov_calm_nav and
    # tasks/station_keeping: body +Y thrust, calm-water buoyancy and drag, and the
    # world-frame-to-body-frame fix at the point of force application.
    def _compute_buoyancy_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.robot.data.root_pos_w
        orientations = self.robot.data.root_link_quat_w

        z_positions = positions[:, 2]
        center_of_h = self.physics_cfg.rov_height / 2
        water_surface = self.physics_cfg.water_surface_z
        depth_below_surface = water_surface - z_positions

        submerged_ratio = torch.clamp(
            (depth_below_surface + center_of_h) / self.physics_cfg.rov_height,
            min=0.0,
            max=1.0,
        )

        submerged_volume = self.physics_cfg.rov_volume * submerged_ratio
        buoyancy_magnitude = (
            self.physics_cfg.water_density
            * submerged_volume
            * self.physics_cfg.gravity
        )

        buoyancy_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        buoyancy_force_world[:, 2] = buoyancy_magnitude

        buoyancy_center_offset_body = torch.zeros((self.num_envs, 3), device=self.device)
        buoyancy_center_offset_body[:, 2] = self.physics_cfg.buoyancy_center_offset

        buoyancy_center_offset_world = math_utils.quat_apply(
            orientations,
            buoyancy_center_offset_body,
        )

        buoyancy_torque = torch.cross(
            buoyancy_center_offset_world,
            buoyancy_force_world,
            dim=-1,
        )

        return buoyancy_force_world, buoyancy_torque

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        num_bodies = self.robot.num_bodies
        forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        torques = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)

        forces[:, 0, 1] = self.actions[:, 0] * self.cfg.thrust_max
        torques[:, 0, 2] = self.actions[:, 1] * self.cfg.yaw_torque_max

        force_w = torch.zeros_like(forces[:, 0, :])
        torque_w = torch.zeros_like(torques[:, 0, :])

        buoyancy_force, buoyancy_torque = self._compute_buoyancy_forces()
        force_w += buoyancy_force
        torque_w += buoyancy_torque

        vel_w = self.robot.data.root_com_vel_w
        if vel_w.shape[-1] == 6:
            linear_velocity = vel_w[:, :3]
            angular_velocity = vel_w[:, 3:]
        else:
            linear_velocity = vel_w
            angular_velocity = self.robot.data.root_ang_vel_w

        v_xy = linear_velocity[:, :2]
        speed_xy = torch.norm(v_xy, dim=-1, keepdim=True)
        force_w[:, :2] += -(
            self.physics_cfg.surge_lin_damping
            + self.physics_cfg.surge_quad_damping * speed_xy
        ) * v_xy
        force_w[:, 2] += -self.physics_cfg.heave_damping * linear_velocity[:, 2]

        wz = angular_velocity[:, 2]
        torque_w[:, 2] += -(
            self.physics_cfg.yaw_lin_damping
            + self.physics_cfg.yaw_quad_damping * torch.abs(wz)
        ) * wz
        torque_w[:, :2] += -self.physics_cfg.rollpitch_rate_damping * angular_velocity[:, :2]

        quat = self.robot.data.root_link_quat_w
        # Yaw-invariant attitude spring: restoring torque k*(up_body_in_world x
        # world_up). The previous Euler-angle form applied BODY tilt angles as
        # FIXED world-axis torques, which is restoring only near the spawn yaw
        # and becomes precessing/anti-restoring past ~90 deg (capsize-by-turning).
        up_body_w = math_utils.quat_apply(quat, self.up_dir.expand(self.num_envs, 3))
        tilt_axis = torch.stack((up_body_w[:, 1], -up_body_w[:, 0]), dim=-1)
        torque_w[:, :2] += self.physics_cfg.attitude_spring * tilt_axis

        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_w)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_w)
        self.robot.set_external_force_and_torque(forces, torques)

    def _waypoint_world(self, waypoint_indices: torch.Tensor) -> torch.Tensor:
        env_indices = torch.arange(self.num_envs, device=self.device)
        return (
            self.waypoints[env_indices, waypoint_indices]
            + self.scene.env_origins[:, :2]
        )

    def _current_waypoint_world(self) -> torch.Tensor:
        waypoint_indices = self.gates_passed.clamp(max=self.cfg.num_waypoints - 1)
        return self._waypoint_world(waypoint_indices)

    def _active_segment_relative(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_indices = torch.arange(self.num_envs, device=self.device)
        waypoint_indices = self.gates_passed.clamp(max=self.cfg.num_waypoints - 1)
        segment_end = self.waypoints[env_indices, waypoint_indices]
        previous_indices = (waypoint_indices - 1).clamp(min=0)
        previous_waypoint = self.waypoints[env_indices, previous_indices]
        segment_start = torch.where(
            (waypoint_indices == 0).unsqueeze(-1),
            torch.zeros_like(previous_waypoint),
            previous_waypoint,
        )
        return segment_start, segment_end

    def _heading_and_direction(self, target_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rpos = target_xy - self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(rpos, dim=-1, keepdim=True).clamp(min=1.0e-6)
        direction = rpos / distance

        forwards = math_utils.quat_apply(
            self.robot.data.root_link_quat_w,
            self.forward_vec,
        )
        forwards_2d = forwards[:, :2]
        forwards_2d = forwards_2d / torch.norm(
            forwards_2d, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)
        return forwards_2d, direction

    def _get_observations(self) -> dict:
        target_xy = self._current_waypoint_world()
        forwards_2d, direction = self._heading_and_direction(target_xy)
        distance = torch.norm(
            target_xy - self.robot.data.root_pos_w[:, :2], dim=-1, keepdim=True
        )

        dot = torch.sum(forwards_2d * direction, dim=-1, keepdim=True)
        cross = (
            forwards_2d[:, 0:1] * direction[:, 1:2]
            - forwards_2d[:, 1:2] * direction[:, 0:1]
        )
        distance_norm = distance / self.cfg.segment_length_max

        stage_norm = self.gates_passed.float().unsqueeze(-1) / self.cfg.num_waypoints

        current_is_last = self.gates_passed >= self.cfg.num_waypoints - 1
        next_indices = (self.gates_passed + 1).clamp(max=self.cfg.num_waypoints - 1)
        next_target_xy = self._waypoint_world(next_indices)
        next_forwards_2d, next_direction = self._heading_and_direction(next_target_xy)
        next_distance = torch.norm(
            next_target_xy - self.robot.data.root_pos_w[:, :2], dim=-1, keepdim=True
        )
        next_dot = torch.sum(next_forwards_2d * next_direction, dim=-1, keepdim=True)
        next_cross = (
            next_forwards_2d[:, 0:1] * next_direction[:, 1:2]
            - next_forwards_2d[:, 1:2] * next_direction[:, 0:1]
        )
        next_distance_norm = next_distance / self.cfg.segment_length_max

        # The last waypoint has no successor. Pad its look-ahead triplet with the
        # neutral "straight ahead, arrived" geometry (dot=1, cross=0, distance=0).
        last_mask = current_is_last.unsqueeze(-1)
        next_dot = torch.where(last_mask, torch.ones_like(next_dot), next_dot)
        next_cross = torch.where(last_mask, torch.zeros_like(next_cross), next_cross)
        next_distance_norm = torch.where(
            last_mask, torch.zeros_like(next_distance_norm), next_distance_norm
        )
        return {
            "policy": torch.hstack(
                [
                    dot,
                    cross,
                    distance_norm,
                    stage_norm,
                    next_dot,
                    next_cross,
                    next_distance_norm,
                ]
            )
        }

    def _get_rewards(self) -> torch.Tensor:
        """Reference-only potential-progress reward; success never reads this value."""
        progress = (
            self._prev_target_distance - self._pre_transition_target_distance
        )
        reward = (
            self.cfg.reference_reward_progress_scale * progress
            + self._gates_passed_this_step.float() * self.cfg.reference_reward_gate_bonus
            + self.cfg.reference_reward_terminal_success * self._success.float()
        )

        # Re-anchor after rewarding progress against the step-start waypoint.
        # On a gate transition this measures the new active waypoint, avoiding a
        # mixed old-target/new-target delta on the next control step.
        target_xy = self._current_waypoint_world()
        self._prev_target_distance.copy_(
            torch.norm(target_xy - self.robot.data.root_pos_w[:, :2], dim=-1)
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_xy = self.robot.data.root_pos_w[:, :2]
        self.path_length += torch.norm(current_xy - self._previous_xy, dim=-1)
        self._previous_xy.copy_(current_xy)

        # XTE is sampled against the segment that is active at the start of this
        # control step, before a possible gate transition.
        segment_start, segment_end = self._active_segment_relative()
        segment_vector = segment_end - segment_start
        segment_direction = segment_vector / torch.norm(
            segment_vector, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)
        position_relative = current_xy - self.scene.env_origins[:, :2]
        from_segment_start = position_relative - segment_start
        xte = torch.abs(
            segment_direction[:, 0] * from_segment_start[:, 1]
            - segment_direction[:, 1] * from_segment_start[:, 0]
        )
        self._xte_squared_sum += xte.square()
        self._xte_sample_count += 1
        self.xte_rms.copy_(
            torch.sqrt(self._xte_squared_sum / self._xte_sample_count.float().clamp(min=1.0))
        )

        target_xy = self._current_waypoint_world()
        rpos = target_xy - current_xy
        distance = torch.norm(rpos, dim=-1)
        self._pre_transition_target_distance.copy_(distance)

        reached = (distance <= self.cfg.goal_radius) & ~self._success
        self._gates_passed_this_step.copy_(reached)
        self.gates_passed += reached.long()
        self._success.copy_(self.gates_passed >= self.cfg.num_waypoints)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        completed_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(completed_ids) > 0:
            success = self._success[completed_ids]
            elapsed_s = self.episode_length_buf[completed_ids].float() * self.control_step_s
            time_to_success = torch.where(
                success,
                elapsed_s,
                torch.full_like(elapsed_s, torch.nan),
            )

            self.episode_success[completed_ids] = success
            self.time_to_success[completed_ids] = time_to_success
            self.episode_path_length[completed_ids] = self.path_length[completed_ids]
            self.episode_gates_passed[completed_ids] = self.gates_passed[completed_ids]
            self.episode_xte_rms[completed_ids] = self.xte_rms[completed_ids]
            self.episode_route_length[completed_ids] = self.route_length[completed_ids]

            self.extras.setdefault("log", {})
            self.extras["log"]["Episode/success"] = success.float().mean()
            if success.any():
                self.extras["log"]["Episode/time_to_success_s"] = elapsed_s[success].mean()
            else:
                self.extras["log"]["Episode/time_to_success_s"] = torch.tensor(
                    torch.nan, device=self.device
                )
            self.extras["log"]["Episode/path_length_m"] = self.path_length[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/gates_passed"] = self.gates_passed[
                completed_ids
            ].float().mean()
            self.extras["log"]["Episode/xte_rms_m"] = self.xte_rms[completed_ids].mean()

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        self.segment_lengths[env_ids] = self.cfg.segment_length_min + torch.rand(
            (num_resets, self.cfg.num_waypoints), device=self.device
        ) * (self.cfg.segment_length_max - self.cfg.segment_length_min)

        headings = torch.empty(
            (num_resets, self.cfg.num_waypoints), device=self.device
        )
        headings[:, 0] = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        max_change = torch.deg2rad(
            torch.tensor(self.cfg.heading_change_max_deg, device=self.device)
        )
        heading_changes = (
            torch.rand(
                (num_resets, self.cfg.num_waypoints - 1), device=self.device
            )
            * 2.0
            - 1.0
        ) * max_change
        headings[:, 1:] = headings[:, 0:1] + torch.cumsum(heading_changes, dim=1)

        segment_vectors = torch.stack(
            (
                self.segment_lengths[env_ids] * torch.cos(headings),
                self.segment_lengths[env_ids] * torch.sin(headings),
            ),
            dim=-1,
        )
        self.waypoints[env_ids] = torch.cumsum(segment_vectors, dim=1)
        self.route_length[env_ids] = self.segment_lengths[env_ids].sum(dim=1)

        spawn_headings = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 3:7] = math_utils.quat_from_angle_axis(
            spawn_headings.unsqueeze(-1), self.up_dir
        ).reshape(num_resets, 4)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self.gates_passed[env_ids] = 0
        self.path_length[env_ids] = 0.0
        self.xte_rms[env_ids] = 0.0
        self._xte_squared_sum[env_ids] = 0.0
        self._xte_sample_count[env_ids] = 0
        self._success[env_ids] = False
        self._gates_passed_this_step[env_ids] = False
        self._previous_xy[env_ids] = root_state[:, :2]

        first_targets = self.waypoints[env_ids, 0] + self.scene.env_origins[env_ids, :2]
        first_rpos = first_targets - root_state[:, :2]
        first_target_distance = torch.norm(first_rpos, dim=-1)
        self._prev_target_distance[env_ids] = first_target_distance
        self._pre_transition_target_distance[env_ids] = first_target_distance

        # This is reset-only USD authoring and has no disabled-mode or per-step cost.
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
