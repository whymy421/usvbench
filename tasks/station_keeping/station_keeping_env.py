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

from .station_keeping_env_cfg import StationKeepingEnvCfg


class StationKeepingEnv(DirectRLEnv):
    """Calm-water station keeping with reward-independent success."""

    cfg: StationKeepingEnvCfg

    def __init__(self, cfg: StationKeepingEnvCfg, render_mode: str | None = None, **kwargs):
        self.physics_cfg = cfg.underwater_physics_cfg
        super().__init__(cfg, render_mode, **kwargs)

        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        required_steps = self.cfg.required_hold_time_s / self.control_step_s
        self.required_hold_steps = int(round(required_steps))
        if abs(required_steps - self.required_hold_steps) > 1.0e-6:
            raise ValueError("required_hold_time_s must be an integer number of control steps")

        self.hold_point = self.scene.env_origins[:, :2].clone()
        self.hold_timer = torch.zeros(self.num_envs, device=self.device)
        self.path_length = torch.zeros(self.num_envs, device=self.device)

        # Last completed-episode values are kept per environment as well as sent
        # through extras["log"]. Failed episodes carry NaN time-to-success.
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_to_success = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.episode_path_length = torch.zeros(self.num_envs, device=self.device)
        self.final_hold_timer = torch.zeros(self.num_envs, device=self.device)

        self._hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")

        if self.cfg.visual.enable_hold_zone_marker:
            try:
                self._create_hold_zone_markers()
            except Exception as exc:
                print(f"[WARN] Hold-zone visualization could not be created: {exc}")

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.forward_vec = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)

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

    def _create_hold_zone_markers(self) -> None:
        """Create one static, render-only annulus at each environment origin."""
        import math

        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        segments = int(self.cfg.visual.hold_zone_segments)
        radius = float(self.cfg.hold_radius)
        line_width = float(self.cfg.visual.hold_zone_line_width_m)
        if segments < 3:
            raise ValueError("visual.hold_zone_segments must be at least 3")
        if radius <= 0.0:
            raise ValueError("hold_radius must be positive to render its marker")
        if line_width <= 0.0:
            raise ValueError("visual.hold_zone_line_width_m must be positive")

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
                (
                    inner,
                    outer,
                    next_outer,
                    inner,
                    next_outer,
                    next_inner,
                )
            )

        stage = omni.usd.get_context().get_stage()
        point_array = Vt.Vec3fArray(points)
        count_array = Vt.IntArray(face_counts)
        index_array = Vt.IntArray(face_indices)
        color = Gf.Vec3f(*self.cfg.visual.hold_zone_color)
        colors = Vt.Vec3fArray([color] * len(points))
        opacities = Vt.FloatArray([1.0] * len(points))

        for env_index in range(self.num_envs):
            marker_path = f"/World/envs/env_{env_index}/HoldZone"
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
        # HARD action clip: the declared [-1,1] range is enforced here.
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        num_bodies = self.robot.num_bodies
        forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        torques = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)

        # Thruster control in the body frame: clipped action times cfg limits.
        forces[:, 0, 1] = self.actions[:, 0] * self.cfg.thrust_max
        torques[:, 0, 2] = self.actions[:, 1] * self.cfg.yaw_torque_max

        # Hydrodynamics are calculated in the world frame, then inverse-rotated
        # before set_external_force_and_torque applies body-frame components.
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

    def _horizontal_distance(self) -> torch.Tensor:
        return torch.norm(self.hold_point - self.robot.data.root_pos_w[:, :2], dim=-1)

    def _get_observations(self) -> dict:
        forwards = math_utils.quat_apply(
            self.robot.data.root_link_quat_w,
            self.forward_vec,
        )

        rpos = self.hold_point - self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(rpos, dim=-1, keepdim=True).clamp(min=1.0e-6)
        direction = rpos / distance

        forwards_2d = forwards[:, :2]
        forwards_2d = forwards_2d / torch.norm(
            forwards_2d, dim=-1, keepdim=True
        ).clamp(min=1.0e-6)

        dot = torch.sum(forwards_2d * direction, dim=-1, keepdim=True)
        cross = (
            forwards_2d[:, 0:1] * direction[:, 1:2]
            - forwards_2d[:, 1:2] * direction[:, 0:1]
        )
        distance_norm = distance / self.cfg.max_spawn_distance

        return {"policy": torch.hstack([dot, cross, distance_norm])}

    def _get_rewards(self) -> torch.Tensor:
        """Reference baseline only; methods may use any reward shaping."""
        distance = self._horizontal_distance()
        inside_hold_zone = distance <= self.cfg.hold_radius
        return (
            -distance / self.cfg.reference_reward_distance_scale
            + inside_hold_zone.float()
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_xy = self.robot.data.root_pos_w[:, :2]
        self.path_length += torch.norm(current_xy - self._previous_xy, dim=-1)
        self._previous_xy.copy_(current_xy)

        inside_hold_zone = self._horizontal_distance() <= self.cfg.hold_radius
        self._hold_steps = torch.where(
            inside_hold_zone,
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
            self.extras["log"]["Episode/path_length_m"] = self.path_length[completed_ids].mean()
            self.extras["log"]["Episode/final_hold_timer_s"] = self.hold_timer[completed_ids].mean()

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        distances = self.cfg.min_spawn_distance + torch.rand(
            num_resets, device=self.device
        ) * (self.cfg.max_spawn_distance - self.cfg.min_spawn_distance)
        spawn_angles = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        headings = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 0] += distances * torch.cos(spawn_angles)
        root_state[:, 1] += distances * torch.sin(spawn_angles)
        root_state[:, 3:7] = math_utils.quat_from_angle_axis(
            headings.unsqueeze(-1), self.up_dir
        ).reshape(num_resets, 4)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self._hold_steps[env_ids] = 0
        self._max_hold_steps[env_ids] = 0
        self._first_success_time_s[env_ids] = torch.nan
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._success[env_ids] = False
        self._episode_finished[env_ids] = False
        self._previous_xy[env_ids] = root_state[:, :2]
