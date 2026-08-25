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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.restoring import restoring_torque_body
from .._shared.scenario_draws import (
    make_scenario_rng,
    path_following_route,
    spawn_heading,
    stamp_scenario,
)
from .._shared.vehicles import get_vehicle
from .path_following_env_cfg import PathFollowingEnvCfg


def _space_dim(space: object) -> int:
    """Return the leading dimension of a Gym space or a legacy integer size."""
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class PathFollowingEnv(DirectRLEnv):
    """Calm-water ordered waypoint following with reward-independent success."""

    cfg: PathFollowingEnvCfg

    def __init__(self, cfg: PathFollowingEnvCfg, render_mode: str | None = None, **kwargs):
        self.vehicle_spec = get_vehicle(cfg.vehicle)
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

        # --- Controller-independent episode scenarios -----------------------
        # The waypoint chain and the spawn heading used to come off the GLOBAL
        # torch RNG, which skrl's Runner reseeds to a constant AFTER the env is
        # built, so two --eval-seed values drew the same routes. This family is
        # the worst-affected of the three: _get_dones returns self._success as
        # `terminated`, so episodes end at DIFFERENT steps under different
        # policies and the global stream is consumed in a controller-dependent
        # order. The keyed per-(env, episode) streams remove both problems.
        # cfg.seed None -> None -> the historical global-RNG lines, unchanged.
        self._scenario = make_scenario_rng(self.cfg, self.num_envs, self.device)
        # _scenario_* is the RUNNING episode; episode_scenario_* is latched at
        # reset for the episode that just ENDED (same discipline as
        # episode_xte_rms), because the evaluator reads them after the reset.
        self._scenario_params = [{} for _ in range(self.num_envs)]
        self._scenario_hashes = [{} for _ in range(self.num_envs)]
        self.episode_scenario = [{} for _ in range(self.num_envs)]
        self.episode_scenario_hashes = [{} for _ in range(self.num_envs)]

    def _setup_scene(self):
        if self.vehicle_spec.asset_kind == "articulation":
            self.robot = Articulation(self.cfg.robot_cfg)
        else:
            self.robot = RigidObject(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        if self.vehicle_spec.asset_kind == "articulation":
            self.scene.articulations["robot"] = self.robot
        else:
            self.scene.rigid_objects["robot"] = self.robot

        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")

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

        a0 = self.actions[:, 0]
        thrust_magnitude = torch.where(
            a0 >= 0.0,
            a0 * self.cfg.thrust_max_fwd,
            a0 * self.cfg.thrust_max_rev,
        )
        forces[:, 0, 0] = thrust_magnitude * self._fwd_x
        forces[:, 0, 1] = thrust_magnitude * self._fwd_y
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

        quat = self.robot.data.root_link_quat_w
        if (
            self.physics_cfg.sway_lin_damping is None
            or self.physics_cfg.sway_quad_damping is None
        ):
            # Preserve the legacy ROV's isotropic horizontal damping exactly.
            v_xy = linear_velocity[:, :2]
            speed_xy = torch.norm(v_xy, dim=-1, keepdim=True)
            force_w[:, :2] += -(
                self.physics_cfg.surge_lin_damping
                + self.physics_cfg.surge_quad_damping * speed_xy
            ) * v_xy
        else:
            vel_b = math_utils.quat_apply_inverse(quat, linear_velocity)
            drag_b = torch.zeros_like(vel_b)
            surge_velocity = vel_b[:, self._surge_axis_idx]
            drag_b[:, self._surge_axis_idx] = -(
                self.physics_cfg.surge_lin_damping
                + self.physics_cfg.surge_quad_damping * torch.abs(surge_velocity)
            ) * surge_velocity
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
        torque_w[:, :2] += -self.physics_cfg.rollpitch_rate_damping * angular_velocity[:, :2]

        # Yaw-invariant attitude spring: restoring torque k*(up_body_in_world x
        # world_up). The previous Euler-angle form applied BODY tilt angles as
        # FIXED world-axis torques, which is restoring only near the spawn yaw
        # and becomes precessing/anti-restoring past ~90 deg (capsize-by-turning).
        if (
            self.physics_cfg.restoring_stiffness_roll
            == self.physics_cfg.restoring_stiffness_pitch
        ):
            up_body_w = math_utils.quat_apply(
                quat, self.up_dir.expand(self.num_envs, 3)
            )
            tilt_axis = torch.stack((up_body_w[:, 1], -up_body_w[:, 0]), dim=-1)
            torque_w[:, :2] += (
                self.physics_cfg.restoring_stiffness_roll * tilt_axis
            )
        else:
            torques[:, 0, :] += restoring_torque_body(
                quat,
                self.physics_cfg.restoring_stiffness_roll,
                self.physics_cfg.restoring_stiffness_pitch,
            )

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

        # At the final gate the clamp makes next == current: the look-ahead
        # smoothly degenerates to the current target ("the route ends here").
        # v4 padded it with (dot=1, cross=0, dist=0) instead -- an
        # out-of-distribution input the policy had never seen; probing showed
        # the thrust command SIGN-FLIPPED the moment the padding appeared,
        # deterministically stranding every episode at 3/4 gates.
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
            for completed in completed_ids.tolist():
                self.episode_scenario[completed] = self._scenario_params[completed]
                self.episode_scenario_hashes[completed] = self._scenario_hashes[
                    completed
                ]

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

        # New episode for these envs: advance their per-env episode counter and
        # reseed every primitive stream. MUST precede the first draw below.
        if self._scenario is not None:
            self._scenario.reset_idx(env_ids)

        num_resets = len(env_ids)
        max_change = torch.deg2rad(
            torch.tensor(self.cfg.heading_change_max_deg, device=self.device)
        )
        # Identical ranges to before -- segment length uniform on
        # [segment_length_min, segment_length_max), first heading uniform on
        # [0, 2*pi), each heading change uniform on [-max_change, +max_change).
        # Only the stream the three uniforms come from changed.
        segment_lengths, first_headings, heading_changes = path_following_route(
            self._scenario,
            env_ids,
            self.device,
            self.cfg.num_waypoints,
            self.cfg.segment_length_min,
            self.cfg.segment_length_max,
            max_change,
        )
        self.segment_lengths[env_ids] = segment_lengths

        headings = torch.empty(
            (num_resets, self.cfg.num_waypoints), device=self.device
        )
        headings[:, 0] = first_headings
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

        # Uniform on [0, 2*pi) exactly as before; only the stream changed.
        spawn_headings = spawn_heading(self._scenario, env_ids, self.device)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        body_yaws = spawn_headings + self._body_yaw_from_bow_offset
        root_state[:, 3:7] = math_utils.quat_from_angle_axis(
            body_yaws.unsqueeze(-1), self.up_dir
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

        # Resolved scenario for the episode that starts now, plus a hash per
        # primitive group for the certificate. Values only -- never the key --
        # so two eval seeds that happened to draw the same route still hash the
        # same and stay visible to scripts/check_scenario_independence.py.
        #
        # Every entry is a RAW DRAW. The route used to stamp headings_rad, the
        # per-waypoint ABSOLUTE heading, which is the running sum of the draws
        # (:696-697) and not one of them. In exact arithmetic that sum is
        # invertible, but it is computed in float32 and the accumulation is
        # lossy: a heading change small against the ulp of the running total
        # vanishes into it. Two checks, both reproducible:
        #   * the hand case -- first heading 1.234 rad with changes
        #     (0.7, 1e-9) and (0.7, 4e-9), values inside
        #     [-heading_change_max, +heading_change_max) -- stamps the
        #     byte-identical headings [1.234, 1.934, 1.934]. Honest caveat:
        #     those two tails are inside the RANGE but below what this chain
        #     can actually draw. heading_changes are (u*2 - 1) * max_change
        #     with u a float32 unit draw, so the smallest nonzero magnitude
        #     available is 6.2e-8 rad, not 1e-9.
        #   * the achievable case, which is the one that decides it: perturbing
        #     one real unit draw by a single ulp over 4e6 four-waypoint routes
        #     changed the stamped heading_change in 3.0e6 of them, and of THOSE
        #     25.5% still stamped byte-identical absolute headings.
        # Two different exam papers, one hash -- at a quarter of the
        # neighbouring draws, not as a curiosity.
        #
        # first_heading_rad plus heading_change_rad are what the stream actually
        # produced, and they still reconstruct the absolute headings exactly
        # (headings[:, 0] = first_heading; headings[:, 1:] = first_heading +
        # cumsum(heading_change)), so recording the draw costs the certificate
        # nothing and closes the collision.
        for env_index, resolved, hashes in stamp_scenario(
            env_ids,
            {
                "route": {
                    "segment_lengths_m": self.segment_lengths[env_ids],
                    "first_heading_rad": first_headings,
                    "heading_change_rad": heading_changes,
                },
                "spawn_pose": {"heading_rad": spawn_headings},
            },
        ):
            self._scenario_params[env_index] = resolved
            self._scenario_hashes[env_index] = hashes

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
