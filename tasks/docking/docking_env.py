# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.vehicles import get_vehicle
from .curriculum import DockingCurriculum
from .docking_env_cfg import DockingEnvCfg
from .restoring import restoring_torque_body


def _space_dim(space: object) -> int:
    """Return the leading dimension of a Gym space or a legacy integer size."""
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class DockingEnv(DirectRLEnv):
    """Calm-water boat docking with reward-independent success."""

    cfg: DockingEnvCfg

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kwargs):
        self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        self.curriculum = DockingCurriculum(
            start_distance=cfg.curriculum_start_distance_m,
            distance_increment=cfg.curriculum_distance_increment_m,
            max_distance=cfg.curriculum_max_distance_m,
            ema_decay=cfg.curriculum_ema_decay,
            success_threshold=cfg.curriculum_success_threshold,
        )
        super().__init__(cfg, render_mode, **kwargs)

        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        required_steps = self.cfg.required_hold_time_s / self.control_step_s
        self.required_hold_steps = int(round(required_steps))
        if abs(required_steps - self.required_hold_steps) > 1.0e-6:
            raise ValueError("required_hold_time_s must be an integer number of control steps")

        self._heading_dot_threshold = math.cos(
            math.radians(self.cfg.success_heading_tolerance_deg)
        )
        self.dock_point = self.scene.env_origins[:, :2].clone()
        self.dock_heading = torch.tensor([1.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

        self.hold_timer = torch.zeros(self.num_envs, device=self.device)
        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_spawn_distance = torch.tensor(
            self.curriculum.spawn_distance, device=self.device
        )

        # Last completed-episode values remain available per environment as well
        # as through extras["log"]. Failed episodes retain NaN time-to-success.
        self.episode_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_to_success = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.episode_path_length = torch.zeros(self.num_envs, device=self.device)
        self.final_hold_timer = torch.zeros(self.num_envs, device=self.device)

        self._hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._max_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._first_success_time_s = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_finished = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_xy = self.robot.data.root_pos_w[:, :2].clone()
        self.actions = torch.zeros(
            (self.num_envs, _space_dim(self.cfg.action_space)), device=self.device
        )

    @property
    def current_spawn_distance(self) -> float:
        """Current curriculum spawn radius in metres."""
        return self.curriculum.spawn_distance

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.rigid_objects["robot"] = self.robot

        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")

        if self.cfg.visual.enable_tolerance_ring:
            try:
                self._create_tolerance_ring_markers()
            except Exception as exc:
                print(f"[WARN] Tolerance-ring visualization could not be created: {exc}")

        if self.cfg.visual.enable_berth_marker:
            try:
                self._create_berth_markers()
            except Exception as exc:
                print(f"[WARN] Berth visualization could not be created: {exc}")

        if self.cfg.visual.enable_berth_box:
            try:
                self._create_berth_box_markers()
            except Exception as exc:
                print(f"[WARN] Berth-box visualization could not be created: {exc}")

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
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

    def _create_static_water_mesh(self) -> None:
        """Create a flat, render-only water mesh with no per-step updates."""
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

    def _create_tolerance_ring_markers(self) -> None:
        """Create one static, render-only tolerance annulus at each dock point."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        segments = int(self.cfg.visual.tolerance_ring_segments)
        radius = float(self.cfg.success_position_tolerance_m)
        line_width = float(self.cfg.visual.tolerance_ring_line_width_m)
        if segments < 3:
            raise ValueError("visual.tolerance_ring_segments must be at least 3")
        if radius <= 0.0:
            raise ValueError(
                "success_position_tolerance_m must be positive to render its marker"
            )
        if line_width <= 0.0:
            raise ValueError("visual.tolerance_ring_line_width_m must be positive")

        marker_z = float(self.physics_cfg.water_surface_z) + 0.02
        outer_radius = radius + line_width
        points = []
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            points.extend(
                (
                    Gf.Vec3f(radius * cos_angle, radius * sin_angle, marker_z),
                    Gf.Vec3f(outer_radius * cos_angle, outer_radius * sin_angle, marker_z),
                )
            )

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

        self._author_marker_mesh(
            "ToleranceRing",
            points,
            face_counts,
            face_indices,
            self.cfg.visual.tolerance_ring_color,
            Gf,
            UsdGeom,
            Vt,
        )

    def _create_berth_markers(self) -> None:
        """Create a flat, render-only +X chevron at each dock point."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        length = float(self.cfg.visual.berth_arrow_length_m)
        width = float(self.cfg.visual.berth_arrow_width_m)
        if length <= 0.0:
            raise ValueError("visual.berth_arrow_length_m must be positive")
        if width <= 0.0:
            raise ValueError("visual.berth_arrow_width_m must be positive")

        marker_z = float(self.physics_cfg.water_surface_z) + 0.03
        half_length = length / 2.0
        half_width = width / 2.0
        head_base = half_length - min(length * 0.4, width * 2.0)
        points = [
            Gf.Vec3f(-half_length, -half_width, marker_z),
            Gf.Vec3f(head_base, -half_width, marker_z),
            Gf.Vec3f(head_base, -width, marker_z),
            Gf.Vec3f(half_length, 0.0, marker_z),
            Gf.Vec3f(head_base, width, marker_z),
            Gf.Vec3f(head_base, half_width, marker_z),
            Gf.Vec3f(-half_length, half_width, marker_z),
        ]
        self._author_marker_mesh(
            "BerthHeading",
            points,
            [4, 3],
            [0, 1, 5, 6, 2, 3, 4],
            self.cfg.visual.berth_arrow_color,
            Gf,
            UsdGeom,
            Vt,
        )

    def _create_berth_box_markers(self) -> None:
        """Create a render-only parking-slot outline aligned with dock +X."""
        from pxr import Gf, UsdGeom, Vt

        length = float(self.cfg.visual.berth_box_length_m)
        width = float(self.cfg.visual.berth_box_width_m)
        line_width = float(self.cfg.visual.berth_box_line_width_m)
        if length <= 0.0:
            raise ValueError("visual.berth_box_length_m must be positive")
        if width <= 0.0:
            raise ValueError("visual.berth_box_width_m must be positive")
        if line_width <= 0.0:
            raise ValueError("visual.berth_box_line_width_m must be positive")
        if line_width >= min(length, width):
            raise ValueError("visual.berth_box_line_width_m must fit inside the box")

        marker_z = float(self.physics_cfg.water_surface_z) + 0.03
        half_length = length / 2.0
        half_width = width / 2.0
        half_line = line_width / 2.0
        points = [
            # Long sides, parallel to dock heading (+X).
            Gf.Vec3f(-half_length, -half_width - half_line, marker_z),
            Gf.Vec3f(half_length, -half_width - half_line, marker_z),
            Gf.Vec3f(half_length, -half_width + half_line, marker_z),
            Gf.Vec3f(-half_length, -half_width + half_line, marker_z),
            Gf.Vec3f(-half_length, half_width - half_line, marker_z),
            Gf.Vec3f(half_length, half_width - half_line, marker_z),
            Gf.Vec3f(half_length, half_width + half_line, marker_z),
            Gf.Vec3f(-half_length, half_width + half_line, marker_z),
            # End sides, perpendicular to dock heading.
            Gf.Vec3f(-half_length - half_line, -half_width, marker_z),
            Gf.Vec3f(-half_length + half_line, -half_width, marker_z),
            Gf.Vec3f(-half_length + half_line, half_width, marker_z),
            Gf.Vec3f(-half_length - half_line, half_width, marker_z),
            Gf.Vec3f(half_length - half_line, -half_width, marker_z),
            Gf.Vec3f(half_length + half_line, -half_width, marker_z),
            Gf.Vec3f(half_length + half_line, half_width, marker_z),
            Gf.Vec3f(half_length - half_line, half_width, marker_z),
        ]
        self._author_marker_mesh(
            "BerthBox",
            points,
            [4, 4, 4, 4],
            list(range(16)),
            self.cfg.visual.berth_box_color,
            Gf,
            UsdGeom,
            Vt,
        )

    def _author_marker_mesh(
        self, name, points, face_counts, face_indices, color_value, Gf, UsdGeom, Vt
    ) -> None:
        """Author identical display-only mesh data under every environment."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        point_array = Vt.Vec3fArray(points)
        count_array = Vt.IntArray(face_counts)
        index_array = Vt.IntArray(face_indices)
        color = Gf.Vec3f(*color_value)
        colors = Vt.Vec3fArray([color] * len(points))
        opacities = Vt.FloatArray([1.0] * len(points))

        for env_index in range(self.num_envs):
            marker_path = f"/World/envs/env_{env_index}/{name}"
            if stage.GetPrimAtPath(marker_path).IsValid():
                stage.RemovePrim(marker_path)
            mesh = UsdGeom.Mesh.Define(stage, marker_path)
            mesh.GetPointsAttr().Set(point_array)
            mesh.GetFaceVertexCountsAttr().Set(count_array)
            mesh.GetFaceVertexIndicesAttr().Set(index_array)
            mesh.GetDisplayColorAttr().Set(colors)
            mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
            mesh.GetDisplayOpacityAttr().Set(opacities)
            mesh.GetDoubleSidedAttr().Set(True)

    def _compute_buoyancy_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.robot.data.root_pos_w
        orientations = self.robot.data.root_quat_w

        center_of_h = self.physics_cfg.rov_height / 2
        depth_below_surface = self.physics_cfg.water_surface_z - positions[:, 2]
        submerged_ratio = torch.clamp(
            (depth_below_surface + center_of_h) / self.physics_cfg.rov_height,
            min=0.0,
            max=1.0,
        )

        submerged_volume = self.physics_cfg.rov_volume * submerged_ratio
        buoyancy_magnitude = (
            self.physics_cfg.water_density * submerged_volume * self.physics_cfg.gravity
        )

        buoyancy_force_world = torch.zeros((self.num_envs, 3), device=self.device)
        buoyancy_force_world[:, 2] = buoyancy_magnitude

        buoyancy_center_offset_body = torch.zeros((self.num_envs, 3), device=self.device)
        buoyancy_center_offset_body[:, 2] = self.physics_cfg.buoyancy_center_offset
        buoyancy_center_offset_world = math_utils.quat_apply(
            orientations, buoyancy_center_offset_body
        )
        buoyancy_torque = torch.cross(
            buoyancy_center_offset_world, buoyancy_force_world, dim=-1
        )
        return buoyancy_force_world, buoyancy_torque

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # HARD action clip: the declared [-1, 1] range is enforced here.
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        num_bodies = self.robot.num_bodies
        forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        torques = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)

        # Baseline asymmetric actuation in the body frame. Positive policy
        # thrust moves bow-first along body -X; negative thrust is astern.
        a0 = self.actions[:, 0]
        thrust_magnitude = torch.where(
            a0 >= 0.0,
            a0 * self.cfg.thrust_max_fwd,
            a0 * self.cfg.thrust_max_rev,
        )
        forces[:, 0, 0] = thrust_magnitude * self._fwd_x
        forces[:, 0, 1] = thrust_magnitude * self._fwd_y
        torques[:, 0, 2] = self.actions[:, 1] * self.cfg.yaw_torque_max

        # World-frame buoyancy, heave, and torques are inverse-rotated before
        # application because set_external_force_and_torque consumes body-frame
        # components by default. Hull surge/sway drag is computed in body frame.
        quat = self.robot.data.root_quat_w
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

        vel_b = math_utils.quat_apply_inverse(quat, linear_velocity)
        drag_b = torch.zeros_like(vel_b)
        surge_velocity = vel_b[:, self._surge_axis_idx]
        drag_b[:, self._surge_axis_idx] = -(
            self.physics_cfg.surge_lin_damping
            + self.physics_cfg.surge_quad_damping * torch.abs(surge_velocity)
        ) * surge_velocity
        if (
            self.physics_cfg.sway_lin_damping is not None
            and self.physics_cfg.sway_quad_damping is not None
        ):
            sway_velocity = vel_b[:, self._sway_axis_idx]
            drag_b[:, self._sway_axis_idx] = -(
                self.physics_cfg.sway_lin_damping
                + self.physics_cfg.sway_quad_damping * torch.abs(sway_velocity)
            ) * sway_velocity
        forces[:, 0, :2] += drag_b[:, :2]
        force_w[:, 2] += -self.physics_cfg.heave_damping * linear_velocity[:, 2]

        wz = angular_velocity[:, 2]
        torque_w[:, 2] += -(
            self.physics_cfg.yaw_lin_damping
            + self.physics_cfg.yaw_quad_damping * torch.abs(wz)
        ) * wz
        torque_w[:, :2] += (
            -self.physics_cfg.rollpitch_rate_damping * angular_velocity[:, :2]
        )

        # Native body-frame anisotropic hydrostatic restoring torque. Expressing
        # world-up in body coordinates makes it invariant to the boat's yaw.
        restoring_torque = restoring_torque_body(
            quat,
            self.physics_cfg.restoring_stiffness_roll,
            self.physics_cfg.restoring_stiffness_pitch,
        )
        torques[:, 0, :2] += restoring_torque[:, :2]

        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_w)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_w)
        self.robot.set_external_force_and_torque(forces, torques)

    def _linear_velocity_world(self) -> torch.Tensor:
        vel_w = self.robot.data.root_com_vel_w
        return vel_w[:, :3] if vel_w.shape[-1] == 6 else vel_w

    def _horizontal_distance(self) -> torch.Tensor:
        # Measure the CENTER OF MASS, not the USD body origin: the boat asset's
        # origin sits ~1.0 m from its COM, so a yaw reorientation sweeps the
        # origin on a 1 m arc and can leave the 2.5 m tolerance while the hull
        # itself never moves (probed 2026-07-16; broke both PID and RL).
        return torch.norm(
            self.dock_point - self.robot.data.root_com_pos_w[:, :2], dim=-1
        )

    def _forward_2d(self) -> torch.Tensor:
        forwards = math_utils.quat_apply(self.robot.data.root_quat_w, self.forward_vec)
        forwards_2d = forwards[:, :2]
        return forwards_2d / torch.norm(
            forwards_2d, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)

    def _task_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distance = self._horizontal_distance()
        forwards_2d = self._forward_2d()
        dock_dot = torch.sum(forwards_2d * self.dock_heading, dim=-1)
        planar_speed = torch.norm(self._linear_velocity_world()[:, :2], dim=-1)
        instantaneous_success = (
            (distance <= self.cfg.success_position_tolerance_m)
            & (dock_dot >= self._heading_dot_threshold)
            & (planar_speed <= self.cfg.success_speed_tolerance_mps)
        )
        return distance, dock_dot, planar_speed, instantaneous_success

    def _get_observations(self) -> dict:
        forwards_2d = self._forward_2d()
        # COM-referenced, consistent with the success predicate.
        rpos = self.dock_point - self.robot.data.root_com_pos_w[:, :2]
        distance = torch.norm(rpos, dim=-1, keepdim=True)
        direction = rpos / distance.clamp(min=1.0e-6)

        dot = torch.sum(forwards_2d * direction, dim=-1, keepdim=True)
        cross = (
            forwards_2d[:, 0:1] * direction[:, 1:2]
            - forwards_2d[:, 1:2] * direction[:, 0:1]
        )
        distance_norm = distance / self.cfg.observation_distance_scale_m

        dock_dot = torch.sum(
            forwards_2d * self.dock_heading, dim=-1, keepdim=True
        )
        dock_cross = (
            forwards_2d[:, 0:1] * self.dock_heading[:, 1:2]
            - forwards_2d[:, 1:2] * self.dock_heading[:, 0:1]
        )
        speed_norm = (
            torch.norm(self._linear_velocity_world()[:, :2], dim=-1, keepdim=True)
            / self.cfg.observation_speed_scale_mps
        )
        return {
            "policy": torch.hstack(
                [dot, cross, distance_norm, dock_dot, dock_cross, speed_norm]
            )
        }

    def _get_rewards(self) -> torch.Tensor:
        """Reference baseline only; benchmark methods may use any reward shaping."""
        distance, dock_dot, planar_speed, instantaneous_success = self._task_state()
        braking_credit = (
            self.cfg.reference_reward_braking_scale
            * torch.exp(-distance / self.cfg.reference_reward_braking_decay_m)
            * torch.clamp(
                1.0
                - planar_speed / self.cfg.reference_reward_braking_speed_scale_mps,
                min=0.0,
                max=1.0,
            )
        )
        return (
            -distance / self.cfg.reference_reward_distance_scale_m
            + self.cfg.reference_reward_alignment_scale
            * dock_dot
            * torch.exp(-distance / self.cfg.reference_reward_alignment_decay_m)
            + braking_credit
            + self.cfg.reference_reward_success_bonus * instantaneous_success.float()
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # COM travel: the origin arcs around the COM during yaw, which would
        # count pure reorientation as path length.
        current_xy = self.robot.data.root_com_pos_w[:, :2]
        self.path_length += torch.norm(current_xy - self._previous_xy, dim=-1)
        self._previous_xy.copy_(current_xy)

        _, _, _, instantaneous_success = self._task_state()
        self._hold_steps = torch.where(
            instantaneous_success,
            self._hold_steps + 1,
            torch.zeros_like(self._hold_steps),
        )
        self._max_hold_steps.copy_(
            torch.maximum(self._max_hold_steps, self._hold_steps)
        )
        self.hold_timer.copy_(self._hold_steps * self.control_step_s)

        achieved_now = (
            ~self._success & (self._max_hold_steps >= self.required_hold_steps)
        )
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(achieved_now, elapsed_s, self._first_success_time_s)
        )
        self._success.copy_(self._max_hold_steps >= self.required_hold_steps)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._episode_finished.copy_(time_out)
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        completed_ids = env_ids[self._episode_finished[env_ids]]
        if len(completed_ids) > 0:
            success = self._success[completed_ids]
            time_to_success = self._first_success_time_s[completed_ids]

            self.episode_success[completed_ids] = success
            self.time_to_success[completed_ids] = time_to_success
            self.episode_path_length[completed_ids] = self.path_length[completed_ids]
            self.final_hold_timer[completed_ids] = self.hold_timer[completed_ids]

            self.extras.setdefault("log", {})
            self.extras["log"]["Episode/success"] = success.float().mean()
            if success.any():
                self.extras["log"]["Episode/time_to_success_s"] = time_to_success[
                    success
                ].mean()
            else:
                self.extras["log"]["Episode/time_to_success_s"] = torch.tensor(
                    torch.nan, device=self.device
                )
            self.extras["log"]["Episode/path_length_m"] = self.path_length[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/final_hold_timer_s"] = self.hold_timer[
                completed_ids
            ].mean()

            # Each completed environment contributes one episodic EMA sample.
            for episode_success in success.detach().cpu().tolist():
                self.curriculum.update(episode_success)

        self.extras.setdefault("log", {})
        self.curriculum_spawn_distance.fill_(self.current_spawn_distance)
        self.extras["log"][
            "Curriculum/spawn_distance_m"
        ] = self.curriculum_spawn_distance
        self.extras["log"]["Curriculum/success_rate_ema"] = torch.tensor(
            self.curriculum.success_rate_ema, device=self.device
        )

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        distances = torch.full(
            (num_resets,), self.current_spawn_distance, device=self.device
        )
        dock_headings = self.dock_heading[env_ids]
        dock_headings = dock_headings / torch.norm(
            dock_headings, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)

        bearing_limit = math.radians(self.cfg.spawn_bearing_limit_deg)
        bearing_offsets = (
            2.0 * torch.rand(num_resets, device=self.device) - 1.0
        ) * bearing_limit
        bearing_cos = torch.cos(bearing_offsets)
        bearing_sin = torch.sin(bearing_offsets)
        spawn_directions = torch.stack(
            (
                dock_headings[:, 0] * bearing_cos
                - dock_headings[:, 1] * bearing_sin,
                dock_headings[:, 0] * bearing_sin
                + dock_headings[:, 1] * bearing_cos,
            ),
            dim=-1,
        )

        heading_limit = math.radians(self.cfg.spawn_heading_offset_limit_deg)
        heading_offsets = (
            2.0 * torch.rand(num_resets, device=self.device) - 1.0
        ) * heading_limit
        forward_headings = torch.atan2(dock_headings[:, 1], dock_headings[:, 0])
        forward_headings += heading_offsets
        # Convert the desired world bow heading to the asset's body yaw using
        # the selected hull's registry-authored bow axis.
        body_yaws = forward_headings + self._body_yaw_from_bow_offset

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        # Spawn on the approach lane: bow-ward drift now moves toward the berth,
        # forward thrust serves position control, and alignment cooperates with
        # the approach instead of fighting a return leg.
        spawn_quats = math_utils.quat_from_angle_axis(
            body_yaws.unsqueeze(-1), self.up_dir
        ).reshape(num_resets, 4)
        # Place the CENTER OF MASS (the predicate's reference point) at the
        # curriculum distance; the USD origin sits ~1.0 m from the COM, so the
        # written root pose is offset by the rotated body-frame COM offset.
        com_target_xy = (
            self.dock_point[env_ids] - distances.unsqueeze(-1) * spawn_directions
        )
        com_offset_b = self.robot.data.com_pos_b[env_ids].reshape(num_resets, 3)
        com_offset_w = math_utils.quat_apply(spawn_quats, com_offset_b)
        root_state[:, :2] = com_target_xy - com_offset_w[:, :2]
        root_state[:, 3:7] = spawn_quats
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self._hold_steps[env_ids] = 0
        self._max_hold_steps[env_ids] = 0
        self._first_success_time_s[env_ids] = torch.nan
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._success[env_ids] = False
        self._episode_finished[env_ids] = False
        self._previous_xy[env_ids] = com_target_xy
