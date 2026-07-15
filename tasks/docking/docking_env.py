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

from .curriculum import DockingCurriculum
from .docking_env_cfg import DockingEnvCfg


def _space_dim(space: object) -> int:
    """Return the leading dimension of a Gym space or a legacy integer size."""
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class DockingEnv(DirectRLEnv):
    """Calm-water boat docking with reward-independent success."""

    cfg: DockingEnvCfg

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kwargs):
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

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        # The boat asset's bow/forward axis is body -X.
        self._fwd_x = -1.0
        self.forward_vec = torch.tensor([-1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

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
        drag_b[:, 0] = -(
            self.physics_cfg.surge_lin_damping
            + self.physics_cfg.surge_quad_damping * torch.abs(vel_b[:, 0])
        ) * vel_b[:, 0]
        drag_b[:, 1] = -(
            self.physics_cfg.sway_lin_damping
            + self.physics_cfg.sway_quad_damping * torch.abs(vel_b[:, 1])
        ) * vel_b[:, 1]
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

        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        pitch_angle = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        roll_angle = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        torque_w[:, 0] += -pitch_angle * self.physics_cfg.attitude_spring
        torque_w[:, 1] += -roll_angle * self.physics_cfg.attitude_spring

        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_w)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_w)
        self.robot.set_external_force_and_torque(forces, torques)

    def _linear_velocity_world(self) -> torch.Tensor:
        vel_w = self.robot.data.root_com_vel_w
        return vel_w[:, :3] if vel_w.shape[-1] == 6 else vel_w

    def _horizontal_distance(self) -> torch.Tensor:
        return torch.norm(self.dock_point - self.robot.data.root_pos_w[:, :2], dim=-1)

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
        rpos = self.dock_point - self.robot.data.root_pos_w[:, :2]
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
        hold_progress_credit = (
            self.cfg.reference_reward_hold_progress_scale
            * self.hold_timer
            / self.cfg.required_hold_time_s
        )
        return (
            -distance / self.cfg.reference_reward_distance_scale_m
            + self.cfg.reference_reward_alignment_scale
            * dock_dot
            * torch.exp(-distance / self.cfg.reference_reward_alignment_decay_m)
            + braking_credit
            + hold_progress_credit
            + self.cfg.reference_reward_success_bonus * instantaneous_success.float()
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_xy = self.robot.data.root_pos_w[:, :2]
        self.path_length += torch.norm(current_xy - self._previous_xy, dim=-1)
        self._previous_xy.copy_(current_xy)

        _, _, _, instantaneous_success = self._task_state()
        self._hold_steps = torch.where(
            instantaneous_success,
            self._hold_steps + 1,
            torch.zeros_like(self._hold_steps),
        )
        self.hold_timer.copy_(self._hold_steps * self.control_step_s)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # A timeout is explicitly a failure, including the final control step.
        self._success = (self._hold_steps >= self.required_hold_steps) & ~time_out
        self._episode_finished.copy_(self._success | time_out)
        return self._success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        completed_ids = env_ids[self._episode_finished[env_ids]]
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
            self.final_hold_timer[completed_ids] = self.hold_timer[completed_ids]

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
        # The asset's bow is body -X, so its body yaw is pi beyond the desired
        # world-frame bow heading.
        body_yaws = forward_headings + torch.pi

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, :2] = (
            self.dock_point[env_ids] + distances.unsqueeze(-1) * spawn_directions
        )
        root_state[:, 3:7] = math_utils.quat_from_angle_axis(
            body_yaws.unsqueeze(-1), self.up_dir
        ).reshape(num_resets, 4)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self._hold_steps[env_ids] = 0
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._success[env_ids] = False
        self._episode_finished[env_ids] = False
        self._previous_xy[env_ids] = root_state[:, :2]
