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
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_xy = self.robot.data.root_pos_w[:, :2].clone()
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.forward_vec = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)

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
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        pitch_angle = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        roll_angle = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        torque_w[:, 0] += -pitch_angle * self.physics_cfg.attitude_spring
        torque_w[:, 1] += -roll_angle * self.physics_cfg.attitude_spring

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
        self.hold_timer.copy_(self._hold_steps * self.control_step_s)

        self._success = self._hold_steps >= self.required_hold_steps
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
            self.final_hold_timer[completed_ids] = self.hold_timer[completed_ids]

            self.extras.setdefault("log", {})
            self.extras["log"]["Episode/success"] = success.float().mean()
            if success.any():
                self.extras["log"]["Episode/time_to_success_s"] = elapsed_s[success].mean()
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
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._success[env_ids] = False
        self._previous_xy[env_ids] = root_state[:, :2]
