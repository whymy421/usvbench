# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for Task A: Static Hazard Navigation."""

from __future__ import annotations

from collections.abc import Sequence
import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.obs_superset import SPEED_SCALE_MPS
from .._shared.restoring import restoring_torque_body
from .._shared.vehicles import get_vehicle
from ..docking.curriculum import DockingCurriculum
from .hazard_geometry import (
    analytic_min_clearance,
    ray_circle_ranges,
    sample_layout,
    sample_ring_layout,
)
from .hazard_nav_env_cfg import HazardNavEnvCfg


def _space_dim(space: object) -> int:
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class HazardNavEnv(DirectRLEnv):
    """Fixed-horizon navigation through static analytic cylinder hazards."""

    cfg: HazardNavEnvCfg

    def __init__(self, cfg: HazardNavEnvCfg, render_mode: str | None = None, **kwargs):
        self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        # Reuse docking's advance cooldown and low-EMA retreat state machine;
        # its scalar "spawn distance" is the integer HazardNav level index.
        self.curriculum = DockingCurriculum(
            start_distance=cfg.curriculum_start_level,
            distance_increment=cfg.curriculum_level_increment,
            max_distance=cfg.curriculum_max_level,
            ema_decay=cfg.curriculum_ema_decay,
            success_threshold=cfg.curriculum_success_threshold,
        )
        isaac_seed = getattr(cfg, "seed", None)
        layout_seed = cfg.layout_seed if isaac_seed is None else int(isaac_seed)
        self._layout_rng = np.random.default_rng(layout_seed)
        super().__init__(cfg, render_mode, **kwargs)

        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        self.goal_radius = float(self.cfg.goal_radius)
        self.target_pos = self.scene.env_origins[:, :2].clone()
        self.d0_per_env = torch.ones(self.num_envs, device=self.device)

        max_obstacles = int(self.cfg.max_obstacles)
        self.obstacle_centers = torch.zeros(
            (self.num_envs, max_obstacles, 2), device=self.device
        )
        self.obstacle_radii = torch.zeros(
            (self.num_envs, max_obstacles), device=self.device
        )
        self.obstacle_active = torch.zeros(
            (self.num_envs, max_obstacles), dtype=torch.bool, device=self.device
        )
        self.obstacle_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.curriculum_level = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._route_geodesic_length = torch.zeros(self.num_envs, device=self.device)

        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self._min_clearance = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._previous_xy = self.robot.data.root_com_pos_w[:, :2].clone()
        self._previous_potential = torch.full(
            (self.num_envs,), -1.0, device=self.device
        )
        self._reached_goal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_prev = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_before_goal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._clean_goal_entry_this_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._first_success_time_s = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self._episode_finished = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Last completed-episode snapshots survive DirectRLEnv auto-reset and
        # mirror the evaluator's station-keeping/docking discovery names.
        self.episode_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_to_success = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.episode_path_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_min_clearance = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.route_geodesic_length = torch.zeros(self.num_envs, device=self.device)
        self.actions = torch.zeros(
            (self.num_envs, _space_dim(self.cfg.action_space)), device=self.device
        )

        # Diagnostics only -- none of these feed back into the reward.
        # Counterfactual ledger for the CUT v6 time fee (0.01 per step with
        # distance_norm > 0.10): measures for free what the fee would have
        # charged, instead of spending a training arm on a predicted null.
        self._fee_steps = torch.zeros(self.num_envs, device=self.device)
        self._ep_prox_cost = torch.zeros(self.num_envs, device=self.device)
        self._ep_contact_entries = torch.zeros(self.num_envs, device=self.device)
        self._ep_goal_bonus = torch.zeros(self.num_envs, device=self.device)
        self._ep_reverse_cost = torch.zeros(self.num_envs, device=self.device)

        # Owner's one-shot half-sine threading bonus. Off unless the amplitude
        # is positive, so every certified id keeps its exact reward.
        self._threading = None
        if self.cfg.reward_threading_amplitude > 0.0:
            from .._shared.threading_gate import ThreadingLatch

            self._threading = ThreadingLatch(
                self.num_envs, self.device,
                amplitude=self.cfg.reward_threading_amplitude,
            )

    @property
    def current_level(self) -> int:
        """Current integer difficulty index encoded by DockingCurriculum."""
        if self.cfg.curriculum_frozen:
            return int(self.cfg.eval_level)
        return int(round(self.curriculum.spawn_distance))

    def _setup_scene(self) -> None:
        if self.vehicle_spec.asset_kind == "articulation":
            self.robot = Articulation(self.cfg.robot_cfg)
        else:
            self.robot = RigidObject(self.cfg.robot_cfg)
        if (
            self.vehicle_spec.name == "blueboat"
            and self.cfg.use_official_blueboat_visual
        ):
            try:
                self._attach_official_blueboat_visual()
            except Exception as exc:
                print(f"[WARN] Official BlueBoat visual could not be attached: {exc}")
        self.scene.clone_environments(copy_from_source=False)
        if self.vehicle_spec.asset_kind == "articulation":
            self.scene.articulations["robot"] = self.robot
        else:
            self.scene.rigid_objects["robot"] = self.robot

        # Visual authoring is guarded and has no return path into task state.
        self._goal_translate_ops = []
        self._obstacle_translate_ops = []
        self._obstacle_radius_attrs = []
        self._obstacle_prims = []
        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")
        if self.cfg.visual.enable_goal_ring or self.cfg.visual.enable_obstacles:
            try:
                self._create_render_only_hazard_markers()
            except Exception as exc:
                self._goal_translate_ops = []
                self._obstacle_translate_ops = []
                self._obstacle_radius_attrs = []
                self._obstacle_prims = []
                print(f"[WARN] Hazard visualization could not be created: {exc}")

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

    def _attach_official_blueboat_visual(self) -> None:
        """Replace only the gray render mesh with the collision-free blue model."""
        import omni.usd
        from pxr import UsdGeom

        visual_path = self.cfg.official_blueboat_visual_usd_path
        if not os.path.isfile(visual_path):
            raise FileNotFoundError(visual_path)

        stage = omni.usd.get_context().get_stage()
        body_path = "/World/envs/env_0/Robot/BlueBoat"
        visual_cfg = sim_utils.UsdFileCfg(usd_path=visual_path)
        visual_cfg.func(
            f"{body_path}/OfficialVisual",
            visual_cfg,
            translation=self.cfg.official_blueboat_visual_translation,
            orientation=self.cfg.official_blueboat_visual_orientation,
        )
        gray_visuals = stage.GetPrimAtPath(f"{body_path}/Visuals")
        if gray_visuals.IsValid():
            UsdGeom.Imageable(gray_visuals).MakeInvisible()

    def _create_static_water_mesh(self) -> None:
        """Create one flat display-only water plane."""
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

    def _create_render_only_hazard_markers(self) -> None:
        """Author goal annuli and cylinders without any collision schemas."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        marker_z = float(self.physics_cfg.water_surface_z) + 0.03

        if self.cfg.visual.enable_goal_ring:
            segments = int(self.cfg.visual.goal_ring_segments)
            radius = float(self.cfg.goal_radius)
            outer = radius + float(self.cfg.visual.goal_ring_line_width_m)
            if segments < 3 or radius <= 0.0 or outer <= radius:
                raise ValueError("invalid goal-ring visual geometry")
            points = []
            for segment in range(segments):
                angle = 2.0 * math.pi * segment / segments
                cosine, sine = math.cos(angle), math.sin(angle)
                points.extend(
                    (
                        Gf.Vec3f(radius * cosine, radius * sine, marker_z),
                        Gf.Vec3f(outer * cosine, outer * sine, marker_z),
                    )
                )
            counts = []
            indices = []
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                inner = 2 * segment
                outer_index = inner + 1
                next_inner = 2 * next_segment
                next_outer = next_inner + 1
                counts.extend((3, 3))
                indices.extend(
                    (inner, outer_index, next_outer, inner, next_outer, next_inner)
                )
            point_array = Vt.Vec3fArray(points)
            count_array = Vt.IntArray(counts)
            index_array = Vt.IntArray(indices)
            color = Gf.Vec3f(*self.cfg.visual.goal_ring_color)
            colors = Vt.Vec3fArray([color] * len(points))
            for env_index in range(self.num_envs):
                path = f"/World/envs/env_{env_index}/HazardGoalRing"
                mesh = UsdGeom.Mesh.Define(stage, path)
                mesh.GetPointsAttr().Set(point_array)
                mesh.GetFaceVertexCountsAttr().Set(count_array)
                mesh.GetFaceVertexIndicesAttr().Set(index_array)
                mesh.GetDisplayColorAttr().Set(colors)
                mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
                mesh.GetDoubleSidedAttr().Set(True)
                self._goal_translate_ops.append(
                    self._get_or_add_translate_op(mesh.GetPrim())
                )

        if self.cfg.visual.enable_obstacles:
            color = Gf.Vec3f(*self.cfg.visual.obstacle_color)
            colors = Vt.Vec3fArray([color])
            for env_index in range(self.num_envs):
                env_ops = []
                env_radius_attrs = []
                env_prims = []
                for obstacle_index in range(self.cfg.max_obstacles):
                    path = (
                        f"/World/envs/env_{env_index}/"
                        f"HazardObstacle_{obstacle_index}"
                    )
                    cylinder = UsdGeom.Cylinder.Define(stage, path)
                    cylinder.GetAxisAttr().Set("Z")
                    cylinder.GetHeightAttr().Set(
                        float(self.cfg.visual.obstacle_height_m)
                    )
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

    def _update_render_only_hazard_markers(
        self,
        env_ids: torch.Tensor,
        local_goals: torch.Tensor,
        local_centers: torch.Tensor,
        radii: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Move display prims after reset; analytic tensors remain authoritative."""
        if not self._goal_translate_ops and not self._obstacle_translate_ops:
            return
        try:
            from pxr import Gf, UsdGeom

            env_list = env_ids.detach().cpu().tolist()
            goals = local_goals.detach().cpu().tolist()
            centers = local_centers.detach().cpu().tolist()
            radii_list = radii.detach().cpu().tolist()
            active_list = active.detach().cpu().tolist()
            for row, env_index in enumerate(env_list):
                if self._goal_translate_ops:
                    self._goal_translate_ops[env_index].Set(
                        Gf.Vec3d(goals[row][0], goals[row][1], 0.0)
                    )
                if self._obstacle_translate_ops:
                    for obstacle_index in range(self.cfg.max_obstacles):
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
            print(f"[WARN] Hazard marker update skipped: {exc}")

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
        buoyancy_magnitude = (
            self.physics_cfg.water_density
            * submerged_volume
            * self.physics_cfg.gravity
        )
        force_world = torch.zeros((self.num_envs, 3), device=self.device)
        force_world[:, 2] = buoyancy_magnitude
        offset_body = torch.zeros((self.num_envs, 3), device=self.device)
        offset_body[:, 2] = self.physics_cfg.buoyancy_center_offset
        offset_world = math_utils.quat_apply(orientations, offset_body)
        torque_world = torch.cross(offset_world, force_world, dim=-1)
        return force_world, torque_world

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # The declared bounded Box is a real actuator contract, not metadata.
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

    @staticmethod
    def _get_or_add_translate_op(prim):
        """Reuse an existing translate xformOp if the prim already has one.

        Cloned env prim trees can carry a translate op; blindly calling
        AddTranslateOp then raises 'xformOp:translate already exists' and the
        guarded visual block silently disabled all hazard markers.
        """
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xformable.AddTranslateOp()

    def _com_xy(self) -> torch.Tensor:
        return self.robot.data.root_com_pos_w[:, :2]

    def _horizontal_distance(self) -> torch.Tensor:
        return torch.norm(self.target_pos - self._com_xy(), dim=-1)

    def _forward_2d(self) -> torch.Tensor:
        forward_world = math_utils.quat_apply(self._root_quat(), self.forward_vec)
        forward_2d = forward_world[:, :2]
        return forward_2d / torch.norm(
            forward_2d, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)

    def _body_motion_observation(self) -> tuple[torch.Tensor, ...]:
        velocity_world = self.robot.data.root_com_vel_w
        if velocity_world.shape[-1] == 6:
            linear_velocity = velocity_world[:, :3]
            angular_velocity = velocity_world[:, 3:]
        else:
            linear_velocity = velocity_world
            angular_velocity = self.robot.data.root_ang_vel_w

        quat = self._root_quat()
        linear_body = math_utils.quat_apply_inverse(quat, linear_velocity)
        angular_body = math_utils.quat_apply_inverse(quat, angular_velocity)
        surge = (
            linear_body[:, 0:1] * self._fwd_x
            + linear_body[:, 1:2] * self._fwd_y
        )
        sway = (
            linear_body[:, 0:1] * -self._fwd_y
            + linear_body[:, 1:2] * self._fwd_x
        )
        surge_norm = torch.clamp(surge / SPEED_SCALE_MPS, min=-1.0, max=1.0)
        sway_norm = torch.clamp(sway / SPEED_SCALE_MPS, min=-1.0, max=1.0)
        yaw_rate_norm = torch.clamp(
            angular_body[:, 2:3] / self.cfg.yaw_rate_obs_scale_rad_s,
            min=-1.0,
            max=1.0,
        )
        return surge_norm, sway_norm, yaw_rate_norm

    def _clearance(self) -> torch.Tensor:
        return analytic_min_clearance(
            self._com_xy(),
            self.obstacle_centers,
            self.obstacle_radii,
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self.obstacle_active,
        )

    def _ray_ranges_m(self) -> torch.Tensor:
        """Analytic 36-ray ranges in METERS, shared by obs and reward.

        Recomputed at every call site -- never cached across callbacks. The
        DirectRLEnv order is dones -> rewards -> resets -> obs, so a buffer
        written here would be stale for freshly reset envs and reusing last
        step's obs rays would smuggle hidden history into the reward.
        """
        forward = self._forward_2d()
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        cosine = torch.cos(self._ray_angles).view(1, -1, 1)
        sine = torch.sin(self._ray_angles).view(1, -1, 1)
        ray_directions = cosine * forward.unsqueeze(1) + sine * left.unsqueeze(1)
        return ray_circle_ranges(
            self._com_xy(),
            ray_directions,
            self.obstacle_centers,
            self.obstacle_radii,
            max_range_m=self.cfg.ray_max_range_m,
            active_mask=self.obstacle_active,
        )

    def _get_observations(self) -> dict:
        position_to_goal = self.target_pos - self._com_xy()
        distance = torch.norm(position_to_goal, dim=-1, keepdim=True)
        direction_to_goal = position_to_goal / distance.clamp_min(1.0e-6)
        forward = self._forward_2d()
        dot = torch.sum(forward * direction_to_goal, dim=-1, keepdim=True)
        cross = (
            forward[:, 0:1] * direction_to_goal[:, 1:2]
            - forward[:, 1:2] * direction_to_goal[:, 0:1]
        )
        distance_norm = torch.clamp(
            distance / self.d0_per_env.unsqueeze(-1).clamp_min(1.0e-6),
            min=0.0,
            max=1.0,
        )

        ranges = self._ray_ranges_m()
        ranges_norm = ranges / self.cfg.ray_max_range_m

        speed_norm = (
            torch.norm(
                self.robot.data.root_com_vel_w[:, :2], dim=-1, keepdim=True
            )
            / SPEED_SCALE_MPS
        )
        # B1 (v2): the reached latch enters the obs (cross-task "stage
        # complete" slot), making latch-gated reward terms augmented-Markov.
        # v1 keeps its historical layouts bit-stable.
        reached = self._reached_goal.float().unsqueeze(-1)

        if self.cfg.obs_feasibility:
            # Replace the raw ray block with per-sector feasible travel
            # distance. A gap the hull fits through then reads as open water
            # instead of two threats.
            from .._shared.feasibility import feasibility_pool

            pooled = feasibility_pool(
                self._ray_ranges_m(),
                n_sectors=self.cfg.feasibility_sectors,
                vessel_width_m=2.0 * self.cfg.half_beam_m,
                ray_spacing_rad=2.0 * torch.pi / self.cfg.ray_count,
                width_multiplier=self.cfg.feasibility_width_multiplier,
            )
            ranges_norm = pooled / self.cfg.ray_max_range_m

        if self.cfg.obs_v3:
            surge_norm, sway_norm, yaw_rate_norm = self._body_motion_observation()
            native_observation = torch.hstack(
                (
                    dot,
                    cross,
                    distance_norm,
                    surge_norm,
                    sway_norm,
                    yaw_rate_norm,
                    ranges_norm,
                )
            )
        elif self.cfg.obs_v2:
            native_observation = torch.hstack(
                (dot, cross, distance_norm, reached, speed_norm, ranges_norm)
            )
        else:
            native_observation = torch.hstack(
                (dot, cross, distance_norm, ranges_norm)
            )
        if not self.cfg.emit_superset_obs:
            return {"policy": native_observation}

        zero = torch.zeros_like(distance_norm)
        stage_slot = reached if self.cfg.obs_v2 else zero
        phase_one_hot = torch.hstack((torch.ones_like(zero), zero, zero, zero))
        superset_observation = torch.hstack(
            (
                dot,
                cross,
                distance_norm,
                stage_slot,  # v2: reached latch; v1: no ordered gates
                dot,
                cross,
                distance_norm,  # no look-ahead: next == current
                zero,
                zero,  # no berth alignment
                speed_norm,
                ranges_norm,
                phase_one_hot,
                zero,  # no dwell state
            )
        )
        return {"policy": superset_observation}

    def _potential(self, distance: torch.Tensor) -> torch.Tensor:
        # v4: the potential pulls all the way to the goal CENTER. v1-v3 floored
        # it at goal_radius, which erased the reward gradient over the final
        # 2 m -- demos showed the policy orbiting at the rim with no incentive
        # for a decisive entry. Pulling to center stays memoryless and keeps
        # the <=1 total positive-progress ledger cap (denominator unchanged).
        fraction = torch.clamp(
            distance / self.d0_per_env.clamp_min(1.0e-6),
            min=0.0,
            max=1.0,
        )
        if getattr(self.cfg, "pbrs_shift_potential", False):
            # Route fraction COVERED, in [0, 1]. The discounted shaping drift
            # (gamma-1)*Phi is then always a cost, never a standing bonus.
            return 1.0 - fraction
        return -fraction

    def _get_rewards(self) -> torch.Tensor:
        distance = self._horizontal_distance()
        potential = self._potential(distance)
        if self.cfg.pbrs_correct:
            # gamma*Phi(s') - Phi(s): the form that provably leaves the optimal
            # policy unchanged (Ng, Harada & Russell 1999). IsaacLab fills
            # reset_terminated/reset_time_outs from _get_dones BEFORE calling
            # this, so the terminal state is already known here -- and Grzes
            # (AAMAS 2017) requires Phi = 0 there, for EVERY way an episode can
            # end (success, contact, timeout), or the leftover gamma^N Phi(s_N)
            # is action-dependent and changes the optimal policy.
            if self.cfg.pbrs_zero_at_terminal:
                # TRUE terminals only. A time-limit truncation is not an
                # absorbing state -- the critic bootstraps through it -- so
                # zeroing Phi there would pay 20*(d/D0) for running out of
                # time far from the goal (measured: 75.8% -> 0.0%).
                next_potential = torch.where(
                    self.reset_terminated, torch.zeros_like(potential), potential
                )
            else:
                next_potential = potential
            progress = self.cfg.pbrs_gamma * next_potential - self._previous_potential
        else:
            progress = potential - self._previous_potential
        self._previous_potential.copy_(potential)

        clearance = self._clearance()
        proximity = torch.clamp(
            (self.cfg.safe_clearance_m - clearance) / self.cfg.safe_clearance_m,
            min=0.0,
            max=1.0,
        )
        contact_now = clearance < 0.0
        safety_cost = (
            self.cfg.reward_clearance_scale
            * self.control_step_s
            * proximity.square()
        )
        # v3 contact ledger: analytic obstacles have no physical walls, so
        # "contact" is a REGION the hull can dwell in, not an instantaneous
        # event. v2's -50/step turned a single blind transit into a -2e4
        # return; that return variance blew up the value function and PPO
        # degenerated into full-throttle wandering (path length 142 m).
        # Penalize the ENTRY event (-25 < 0 once per crossing) plus a small
        # bounded dwell cost (-1/step); progress total <= 20 stays below one
        # entry, so the anti-farming ledger survives with sane variance.
        # v6a: per-ray log proximity field replacing the quadratic graze tax
        # (kept above behind reward_clearance_scale=0.0 for the ablation arm).
        # Pure closed-form function of the 36 ray obs slots: zero beyond the
        # band cap (~2x hull-beam clearance), smooth 1/rho gradient inside,
        # floored at half-beam so interior cost stays dwell's job. NavRL-style
        # (RA-L 2025) log barrier, band-limited per path_hazard v2's
        # no-distant-tax lesson.
        rays_m = self._ray_ranges_m()
        prox_cap_m = self.cfg.safe_clearance_m + self.cfg.half_beam_m
        prox_cost = (
            -self.cfg.reward_prox_scale
            * self.control_step_s
            * torch.log(
                rays_m.clamp(self.cfg.prox_ray_floor_m, prox_cap_m) / prox_cap_m
            ).mean(dim=-1)
        )

        # Diagnostics (no reward effect): counterfactual cut-fee ledger uses
        # the exact obs[2] formula so the logged number is what the fee gate
        # would have seen.
        distance_norm = torch.clamp(
            distance / self.d0_per_env.clamp_min(1.0e-6), min=0.0, max=1.0
        )
        self._fee_steps += (distance_norm > 0.10).float()
        self._ep_prox_cost += prox_cost

        contact_entry = contact_now & ~self._contact_prev
        self._contact_prev.copy_(contact_now)
        self._ep_contact_entries += contact_entry.float()

        swift_cost = torch.zeros_like(prox_cost)
        if self.cfg.obs_v2 or self.cfg.obs_v3:
            speed_norm = (
                torch.norm(self.robot.data.root_com_vel_w[:, :2], dim=-1)
                / SPEED_SCALE_MPS
            )
            swift_cost = (
                self.cfg.reward_swift_scale
                * self.control_step_s
                * (1.0 - speed_norm.clamp(max=1.0))
                * (~self._reached_goal).float()
            )

        threading_bonus = torch.zeros(self.num_envs, device=self.device)
        if self._threading is not None:
            surge_n, sway_n, yaw_n = self._body_motion_observation()
            to_goal = self.target_pos - self._com_xy()
            direction = to_goal / torch.norm(
                to_goal, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            forward = self._forward_2d()
            g_dot = torch.sum(forward * direction, dim=-1)
            g_cross = forward[:, 0] * direction[:, 1] - forward[:, 1] * direction[:, 0]
            threading_bonus = self._threading.step(
                self._ray_ranges_m(),
                ray_spacing_rad=2.0 * torch.pi / self.cfg.ray_count,
                half_beam_m=self.cfg.half_beam_m,
                surge_norm=surge_n.squeeze(-1),
                sway_norm=sway_n.squeeze(-1),
                yaw_rate_norm=yaw_n.squeeze(-1),
                goal_dot=g_dot,
                goal_cross=g_cross,
                contact=contact_now,
            )

        reverse_action = torch.relu(-self.actions[:, 0])
        reverse_cost = (
            self.cfg.reward_reverse_action_scale
            * self.control_step_s
            * reverse_action.square()
        )
        remaining_fraction = torch.clamp(
            (self.max_episode_length - self.episode_length_buf.float())
            / max(float(self.max_episode_length), 1.0),
            min=0.0,
            max=1.0,
        )
        goal_bonus = self._clean_goal_entry_this_step.float() * (
            self.cfg.reward_goal_entry_bonus
            + self.cfg.reward_goal_time_bonus * remaining_fraction
        )
        self._ep_goal_bonus += goal_bonus
        self._ep_reverse_cost += reverse_cost

        return (
            self.cfg.reward_progress_scale * progress
            + goal_bonus
            + threading_bonus
            - safety_cost
            - prox_cost
            - swift_cost
            - reverse_cost
            - self.cfg.reward_contact_entry_penalty * contact_entry.float()
            - self.cfg.reward_contact_dwell_penalty * contact_now.float()
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_xy = self._com_xy()
        prefix_active = ~self._reached_goal
        travelled = torch.norm(current_xy - self._previous_xy, dim=-1)
        self.path_length += torch.where(
            prefix_active, travelled, torch.zeros_like(travelled)
        )
        self._previous_xy.copy_(current_xy)

        clearance = self._clearance()
        self._min_clearance.copy_(
            torch.where(
                prefix_active,
                torch.minimum(self._min_clearance, clearance),
                self._min_clearance,
            )
        )
        contact_now = clearance < 0.0
        self._contact_before_goal |= prefix_active & contact_now

        reached_now = prefix_active & (
            self._horizontal_distance() <= self.goal_radius
        )
        success_now = reached_now & ~self._contact_before_goal
        self._clean_goal_entry_this_step.copy_(success_now)
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(success_now, elapsed_s, self._first_success_time_s)
        )
        self._success |= success_now
        self._reached_goal |= reached_now

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Single-goal NavRL episodes terminate on either outcome: reaching
        # the goal or entering an obstacle's contact region. The contact
        # ledger is updated before ``success_now``, so simultaneous contact
        # and goal entry is correctly scored as a failed episode.
        terminated = reached_now | contact_now
        self._episode_finished.copy_(terminated | time_out)
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
            self.episode_path_length[completed_ids] = self.path_length[completed_ids]
            self.episode_min_clearance[completed_ids] = self._min_clearance[
                completed_ids
            ]
            self.route_geodesic_length[completed_ids] = (
                self._route_geodesic_length[completed_ids]
            )

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
            self.extras["log"]["Episode/path_length_m"] = self.path_length[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/min_clearance_m"] = self._min_clearance[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/route_geodesic_length_m"] = (
                self._route_geodesic_length[completed_ids].mean()
            )
            self.extras["log"]["Episode/counterfactual_fee"] = (
                0.01 * self._fee_steps[completed_ids].mean()
            )
            self.extras["log"]["Episode/reward_prox_cost_sum"] = (
                self._ep_prox_cost[completed_ids].mean()
            )
            self.extras["log"]["Episode/contact_entries"] = (
                self._ep_contact_entries[completed_ids].mean()
            )
            self.extras["log"]["Episode/reward_goal_bonus_sum"] = (
                self._ep_goal_bonus[completed_ids].mean()
            )
            self.extras["log"]["Episode/reward_reverse_cost_sum"] = (
                self._ep_reverse_cost[completed_ids].mean()
            )

            if not self.cfg.curriculum_frozen:
                for episode_success in success.detach().cpu().tolist():
                    self.curriculum.update(episode_success)

        self.extras.setdefault("log", {})
        self.extras["log"]["Curriculum/level"] = torch.tensor(
            float(self.current_level), device=self.device
        )
        self.extras["log"]["Curriculum/success_rate_ema"] = torch.tensor(
            self.curriculum.success_rate_ema, device=self.device
        )

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        max_obstacles = self.cfg.max_obstacles
        local_goals_np = np.zeros((num_resets, 2), dtype=np.float32)
        local_centers_np = np.zeros(
            (num_resets, max_obstacles, 2), dtype=np.float32
        )
        radii_np = np.zeros((num_resets, max_obstacles), dtype=np.float32)
        active_np = np.zeros((num_resets, max_obstacles), dtype=np.bool_)
        geodesic_np = np.zeros(num_resets, dtype=np.float32)
        d0_np = np.zeros(num_resets, dtype=np.float32)
        counts_np = np.zeros(num_resets, dtype=np.int64)
        level = self.current_level
        for row in range(num_resets):
            if self.cfg.layout_mode == "ring":
                # Ring siege: spawn encircled, exactly one tier-width gap.
                # Avoidance stops being optional -- escape requires threading.
                layout = sample_ring_layout(
                    level,
                    rng=self._layout_rng,
                    max_attempts=max(40, self.cfg.layout_max_attempts),
                )
            else:
                layout = sample_layout(
                    level,
                    rng=self._layout_rng,
                    max_attempts=self.cfg.layout_max_attempts,
                )
            goal_angle = float(self._layout_rng.uniform(0.0, 2.0 * math.pi))
            cosine, sine = math.cos(goal_angle), math.sin(goal_angle)
            rotation = np.array(((cosine, -sine), (sine, cosine)))
            rotated_goal = layout.goal @ rotation.T
            rotated_centers = layout.centers @ rotation.T
            count = layout.obstacle_count
            local_goals_np[row] = rotated_goal
            local_centers_np[row, :count] = rotated_centers
            radii_np[row, :count] = layout.radii
            active_np[row, :count] = True
            geodesic_np[row] = layout.geodesic_length
            d0_np[row] = float(np.linalg.norm(layout.goal - layout.start))
            counts_np[row] = count

        local_goals = torch.as_tensor(
            local_goals_np, device=self.device, dtype=torch.float32
        )
        local_centers = torch.as_tensor(
            local_centers_np, device=self.device, dtype=torch.float32
        )
        radii = torch.as_tensor(radii_np, device=self.device, dtype=torch.float32)
        active = torch.as_tensor(active_np, device=self.device, dtype=torch.bool)
        env_origins_xy = self.scene.env_origins[env_ids, :2]
        self.target_pos[env_ids] = env_origins_xy + local_goals
        self.obstacle_centers[env_ids] = env_origins_xy.unsqueeze(1) + local_centers
        self.obstacle_radii[env_ids] = radii
        self.obstacle_active[env_ids] = active
        self.d0_per_env[env_ids] = torch.as_tensor(d0_np, device=self.device)
        self._route_geodesic_length[env_ids] = torch.as_tensor(
            geodesic_np, device=self.device
        )
        self.obstacle_count[env_ids] = torch.as_tensor(counts_np, device=self.device)
        self.curriculum_level[env_ids] = level

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        body_yaws = torch.full(
            (num_resets,), self._body_yaw_from_bow_offset, device=self.device
        )
        spawn_quats = math_utils.quat_from_angle_axis(
            body_yaws.unsqueeze(-1), self.up_dir
        ).reshape(num_resets, 4)
        com_offset_body = self.robot.data.com_pos_b[env_ids].reshape(num_resets, 3)
        com_offset_world = math_utils.quat_apply(spawn_quats, com_offset_body)
        com_target = self.scene.env_origins[env_ids, :3].clone()
        # "Start at the environment origin" is a planar COM statement. Keep
        # the USD-authored/default vertical spawn used by the calm-water task.
        root_state[:, :2] = com_target[:, :2] - com_offset_world[:, :2]
        root_state[:, 3:7] = spawn_quats
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self.path_length[env_ids] = 0.0
        self._previous_xy[env_ids] = env_origins_xy
        self._previous_potential[env_ids] = -1.0
        self._reached_goal[env_ids] = False
        self._contact_prev[env_ids] = False
        self._contact_before_goal[env_ids] = False
        self._success[env_ids] = False
        self._clean_goal_entry_this_step[env_ids] = False
        if self._threading is not None:
            # Clear the once-per-episode latch, or the bonus would pay on the
            # first episode of the run and never again.
            self._threading.reset(env_ids)
        self._first_success_time_s[env_ids] = torch.nan
        self._episode_finished[env_ids] = False
        initial_clearance = analytic_min_clearance(
            env_origins_xy,
            self.obstacle_centers[env_ids],
            self.obstacle_radii[env_ids],
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self.obstacle_active[env_ids],
        )
        self._min_clearance[env_ids] = initial_clearance
        self._contact_before_goal[env_ids] = initial_clearance < 0.0
        self.actions[env_ids] = 0.0
        self._fee_steps[env_ids] = 0.0
        self._ep_prox_cost[env_ids] = 0.0
        self._ep_contact_entries[env_ids] = 0.0
        self._ep_goal_bonus[env_ids] = 0.0
        self._ep_reverse_cost[env_ids] = 0.0

        self._update_render_only_hazard_markers(
            env_ids, local_goals, local_centers, radii, active
        )
