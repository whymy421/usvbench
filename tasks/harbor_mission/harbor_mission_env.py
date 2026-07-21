# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for Task B: Ordered Harbor Mission."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
import torch.nn.functional as functional

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.restoring import restoring_torque_body
from .._shared.vehicles import get_vehicle
from ..hazard_nav.hazard_geometry import analytic_min_clearance, ray_circle_ranges
from .harbor_geometry import (
    EXIT_GATE_X_M,
    arm_gate_approach,
    gate_crossing_mask,
    sample_harbor_route,
)
from .harbor_mission_env_cfg import HarborMissionEnvCfg


def _space_dim(space: object) -> int:
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class HarborMissionEnv(DirectRLEnv):
    """Fixed-horizon ordered exit, hazard-transit, and docking mission."""

    cfg: HarborMissionEnvCfg

    def __init__(
        self, cfg: HarborMissionEnvCfg, render_mode: str | None = None, **kwargs
    ):
        self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        isaac_seed = getattr(cfg, "seed", None)
        layout_seed = cfg.layout_seed if isaac_seed is None else int(isaac_seed)
        self._layout_rng = np.random.default_rng(layout_seed)
        super().__init__(cfg, render_mode, **kwargs)

        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        required_steps = self.cfg.required_hold_time_s / self.control_step_s
        self.required_hold_steps = int(round(required_steps))
        if abs(required_steps - self.required_hold_steps) > 1.0e-6:
            raise ValueError("required_hold_time_s must be an integer control-step count")
        self._heading_dot_threshold = math.cos(
            math.radians(self.cfg.success_heading_tolerance_deg)
        )
        self.goal_radius = float(self.cfg.goal_radius)

        max_obstacles = int(self.cfg.max_obstacles)
        self.gate_midpoints = torch.zeros((self.num_envs, 3, 2), device=self.device)
        self.gate_normals = torch.zeros((self.num_envs, 3, 2), device=self.device)
        self.berth_point = self.scene.env_origins[:, :2].clone()
        self.dock_heading = torch.tensor((1.0, 0.0), device=self.device).repeat(
            self.num_envs, 1
        )
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
        self.stage_start_x = torch.zeros((self.num_envs, 3), device=self.device)
        self.stage_end_x = torch.ones((self.num_envs, 3), device=self.device)
        self.field_geodesic_length = torch.zeros(self.num_envs, device=self.device)

        # Explicit monotone automaton state. Exit gate progress is the q=0
        # substate needed to enforce gate 1 before gate 2.
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._exit_gate_progress = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._gate_armed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._m1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._m2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._m3 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._m1_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._m2_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._m3_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._phase_at_step_start = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._exit_progress_at_step_start = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._potential_at_step_start = torch.zeros(
            self.num_envs, device=self.device
        )
        self._hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.hold_timer = torch.zeros(self.num_envs, device=self.device)

        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self._min_clearance = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._previous_xy = self.robot.data.root_com_pos_w[:, :2].clone()
        self._contact_prev = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_before_dock = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._first_success_time_s = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self._episode_finished = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Completed-episode snapshots survive DirectRLEnv auto-reset.
        self.episode_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_to_success = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.stage_reached = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_path_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_min_clearance = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.m1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.m2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.m3 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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

        self._gate_post_ops = []
        self._gate_crossbar_ops = []
        self._obstacle_translate_ops = []
        self._obstacle_radius_attrs = []
        self._obstacle_prims = []
        self._berth_translate_ops = []
        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")
        if (
            self.cfg.visual.enable_gates
            or self.cfg.visual.enable_obstacles
            or self.cfg.visual.enable_berth
        ):
            try:
                self._create_render_only_markers()
            except Exception as exc:
                self._gate_post_ops = []
                self._gate_crossbar_ops = []
                self._obstacle_translate_ops = []
                self._obstacle_radius_attrs = []
                self._obstacle_prims = []
                self._berth_translate_ops = []
                print(f"[WARN] Harbor visualization could not be created: {exc}")

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

        self.up_dir = torch.tensor((0.0, 0.0, 1.0), device=self.device)
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

    @staticmethod
    def _get_or_add_translate_op(prim):
        """Return an existing translate transform op, or author one if absent."""
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xformable.AddTranslateOp()

    def _define_marker_mesh(
        self,
        stage,
        path,
        points,
        counts,
        indices,
        color_value,
        Gf,
        UsdGeom,
        Vt,
    ):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
        color = Gf.Vec3f(*color_value)
        mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color] * len(points)))
        mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
        mesh.GetDoubleSidedAttr().Set(True)
        return self._get_or_add_translate_op(mesh.GetPrim())

    def _create_render_only_markers(self) -> None:
        """Author display-only gates, cylinders, ring, arrow, and berth box."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        marker_z = float(self.physics_cfg.water_surface_z) + 0.03

        ring_points = []
        ring_counts = []
        ring_indices = []
        segments = int(self.cfg.visual.berth_ring_segments)
        radius = float(self.cfg.success_position_tolerance_m)
        outer = radius + float(self.cfg.visual.berth_ring_line_width_m)
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            cosine, sine = math.cos(angle), math.sin(angle)
            ring_points.extend(
                (
                    Gf.Vec3f(radius * cosine, radius * sine, marker_z),
                    Gf.Vec3f(outer * cosine, outer * sine, marker_z),
                )
            )
            next_segment = (segment + 1) % segments
            inner = 2 * segment
            ring_counts.extend((3, 3))
            ring_indices.extend(
                (inner, inner + 1, 2 * next_segment + 1, inner, 2 * next_segment + 1, 2 * next_segment)
            )

        arrow_length = float(self.cfg.visual.berth_arrow_length_m)
        arrow_width = float(self.cfg.visual.berth_arrow_width_m)
        arrow_half_length = arrow_length / 2.0
        arrow_half_width = arrow_width / 2.0
        head_base = arrow_half_length - min(arrow_length * 0.4, arrow_width * 2.0)
        arrow_points = [
            Gf.Vec3f(-arrow_half_length, -arrow_half_width, marker_z),
            Gf.Vec3f(head_base, -arrow_half_width, marker_z),
            Gf.Vec3f(head_base, -arrow_width, marker_z),
            Gf.Vec3f(arrow_half_length, 0.0, marker_z),
            Gf.Vec3f(head_base, arrow_width, marker_z),
            Gf.Vec3f(head_base, arrow_half_width, marker_z),
            Gf.Vec3f(-arrow_half_length, arrow_half_width, marker_z),
        ]

        box_length = float(self.cfg.visual.berth_box_length_m)
        box_width = float(self.cfg.visual.berth_box_width_m)
        half_length = box_length / 2.0
        half_width = box_width / 2.0
        half_line = float(self.cfg.visual.berth_box_line_width_m) / 2.0
        box_points = [
            Gf.Vec3f(-half_length, -half_width - half_line, marker_z),
            Gf.Vec3f(half_length, -half_width - half_line, marker_z),
            Gf.Vec3f(half_length, -half_width + half_line, marker_z),
            Gf.Vec3f(-half_length, -half_width + half_line, marker_z),
            Gf.Vec3f(-half_length, half_width - half_line, marker_z),
            Gf.Vec3f(half_length, half_width - half_line, marker_z),
            Gf.Vec3f(half_length, half_width + half_line, marker_z),
            Gf.Vec3f(-half_length, half_width + half_line, marker_z),
            Gf.Vec3f(-half_length - half_line, -half_width, marker_z),
            Gf.Vec3f(-half_length + half_line, -half_width, marker_z),
            Gf.Vec3f(-half_length + half_line, half_width, marker_z),
            Gf.Vec3f(-half_length - half_line, half_width, marker_z),
            Gf.Vec3f(half_length - half_line, -half_width, marker_z),
            Gf.Vec3f(half_length + half_line, -half_width, marker_z),
            Gf.Vec3f(half_length + half_line, half_width, marker_z),
            Gf.Vec3f(half_length - half_line, half_width, marker_z),
        ]

        crossbar_half_x = float(self.cfg.visual.gate_crossbar_thickness_m) / 2.0
        crossbar_half_y = float(self.cfg.gate_half_width_m)
        crossbar_points = [
            Gf.Vec3f(-crossbar_half_x, -crossbar_half_y, marker_z),
            Gf.Vec3f(crossbar_half_x, -crossbar_half_y, marker_z),
            Gf.Vec3f(crossbar_half_x, crossbar_half_y, marker_z),
            Gf.Vec3f(-crossbar_half_x, crossbar_half_y, marker_z),
        ]

        for env_index in range(self.num_envs):
            if self.cfg.visual.enable_gates:
                env_posts = []
                env_crossbars = []
                for gate_index in range(3):
                    color = (
                        self.cfg.visual.exit_gate_color
                        if gate_index < 2
                        else self.cfg.visual.field_exit_gate_color
                    )
                    gate_posts = []
                    for side in range(2):
                        path = (
                            f"/World/envs/env_{env_index}/HarborGate_{gate_index}_Post_{side}"
                        )
                        cylinder = UsdGeom.Cylinder.Define(stage, path)
                        cylinder.GetAxisAttr().Set("Z")
                        cylinder.GetHeightAttr().Set(
                            float(self.cfg.visual.gate_post_height_m)
                        )
                        cylinder.GetRadiusAttr().Set(
                            float(self.cfg.visual.gate_post_radius_m)
                        )
                        cylinder.GetDisplayColorAttr().Set(
                            Vt.Vec3fArray([Gf.Vec3f(*color)])
                        )
                        gate_posts.append(
                            self._get_or_add_translate_op(cylinder.GetPrim())
                        )
                    env_posts.append(gate_posts)
                    env_crossbars.append(
                        self._define_marker_mesh(
                            stage,
                            f"/World/envs/env_{env_index}/HarborGate_{gate_index}_Crossbar",
                            crossbar_points,
                            [4],
                            [0, 1, 2, 3],
                            color,
                            Gf,
                            UsdGeom,
                            Vt,
                        )
                    )
                self._gate_post_ops.append(env_posts)
                self._gate_crossbar_ops.append(env_crossbars)

            if self.cfg.visual.enable_obstacles:
                env_ops = []
                env_radius_attrs = []
                env_prims = []
                color = Gf.Vec3f(*self.cfg.visual.obstacle_color)
                for obstacle_index in range(self.cfg.max_obstacles):
                    path = f"/World/envs/env_{env_index}/HarborObstacle_{obstacle_index}"
                    cylinder = UsdGeom.Cylinder.Define(stage, path)
                    cylinder.GetAxisAttr().Set("Z")
                    cylinder.GetHeightAttr().Set(
                        float(self.cfg.visual.obstacle_height_m)
                    )
                    cylinder.GetRadiusAttr().Set(0.5)
                    cylinder.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))
                    prim = cylinder.GetPrim()
                    UsdGeom.Imageable(prim).MakeInvisible()
                    env_ops.append(self._get_or_add_translate_op(prim))
                    env_radius_attrs.append(cylinder.GetRadiusAttr())
                    env_prims.append(prim)
                self._obstacle_translate_ops.append(env_ops)
                self._obstacle_radius_attrs.append(env_radius_attrs)
                self._obstacle_prims.append(env_prims)

            if self.cfg.visual.enable_berth:
                berth_ops = [
                    self._define_marker_mesh(
                        stage,
                        f"/World/envs/env_{env_index}/HarborBerthRing",
                        ring_points,
                        ring_counts,
                        ring_indices,
                        self.cfg.visual.berth_ring_color,
                        Gf,
                        UsdGeom,
                        Vt,
                    ),
                    self._define_marker_mesh(
                        stage,
                        f"/World/envs/env_{env_index}/HarborBerthHeading",
                        arrow_points,
                        [4, 3],
                        [0, 1, 5, 6, 2, 3, 4],
                        self.cfg.visual.berth_arrow_color,
                        Gf,
                        UsdGeom,
                        Vt,
                    ),
                    self._define_marker_mesh(
                        stage,
                        f"/World/envs/env_{env_index}/HarborBerthBox",
                        box_points,
                        [4, 4, 4, 4],
                        list(range(16)),
                        self.cfg.visual.berth_box_color,
                        Gf,
                        UsdGeom,
                        Vt,
                    ),
                ]
                self._berth_translate_ops.append(berth_ops)

    def _update_render_only_markers(
        self,
        env_ids: torch.Tensor,
        local_gates: torch.Tensor,
        local_centers: torch.Tensor,
        radii: torch.Tensor,
        active: torch.Tensor,
        local_berths: torch.Tensor,
    ) -> None:
        """Move display prims after reset; analytic tensors remain authoritative."""
        if not (
            self._gate_post_ops
            or self._obstacle_translate_ops
            or self._berth_translate_ops
        ):
            return
        try:
            from pxr import Gf, UsdGeom

            env_list = env_ids.detach().cpu().tolist()
            gates = local_gates.detach().cpu().tolist()
            centers = local_centers.detach().cpu().tolist()
            radii_list = radii.detach().cpu().tolist()
            active_list = active.detach().cpu().tolist()
            berths = local_berths.detach().cpu().tolist()
            post_z = 0.5 * float(self.cfg.visual.gate_post_height_m)
            for row, env_index in enumerate(env_list):
                if self._gate_post_ops:
                    for gate_index in range(3):
                        midpoint = gates[row][gate_index]
                        for side, y_sign in enumerate((-1.0, 1.0)):
                            self._gate_post_ops[env_index][gate_index][side].Set(
                                Gf.Vec3d(
                                    midpoint[0],
                                    midpoint[1] + y_sign * self.cfg.gate_half_width_m,
                                    post_z,
                                )
                            )
                        self._gate_crossbar_ops[env_index][gate_index].Set(
                            Gf.Vec3d(midpoint[0], midpoint[1], 0.0)
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
                if self._berth_translate_ops:
                    berth = berths[row]
                    for translate_op in self._berth_translate_ops[env_index]:
                        translate_op.Set(Gf.Vec3d(berth[0], berth[1], 0.0))
        except Exception as exc:
            print(f"[WARN] Harbor marker update skipped: {exc}")

    def _root_quat(self) -> torch.Tensor:
        return self.robot.data.root_link_quat_w

    def _compute_buoyancy_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.robot.data.root_pos_w
        orientations = self._root_quat()
        half_height = self.physics_cfg.rov_height / 2.0
        depth = self.physics_cfg.water_surface_z - positions[:, 2]
        submerged_ratio = torch.clamp(
            (depth + half_height) / self.physics_cfg.rov_height,
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
        torque_world = torch.cross(offset_world, force_world, dim=-1)
        return force_world, torque_world

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)
        # Capture the automaton state and potential before physics. Isaac Lab
        # versions differ in whether dones or rewards are queried first after
        # simulation; this snapshot makes the old-phase delta order-invariant.
        self._phase_at_step_start.copy_(self.phase)
        self._exit_progress_at_step_start.copy_(self._exit_gate_progress)
        self._potential_at_step_start.copy_(
            self._phase_potential(self._com_xy(), self.phase)
        )

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

    def _linear_velocity_world(self) -> torch.Tensor:
        velocity = self.robot.data.root_com_vel_w
        return velocity[:, :3] if velocity.shape[-1] == 6 else velocity

    def _forward_2d(self) -> torch.Tensor:
        forward_world = math_utils.quat_apply(self._root_quat(), self.forward_vec)
        forward = forward_world[:, :2]
        return forward / torch.norm(forward, dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _clearance(self) -> torch.Tensor:
        return analytic_min_clearance(
            self._com_xy(),
            self.obstacle_centers,
            self.obstacle_radii,
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self.obstacle_active,
        )

    def _active_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        gate_index = torch.where(
            self.phase == 0,
            self._exit_gate_progress.clamp(max=1),
            torch.full_like(self.phase, 2),
        )
        rows = torch.arange(self.num_envs, device=self.device)
        return self.gate_midpoints[rows, gate_index], self.gate_normals[rows, gate_index]

    def _phase_target(self) -> torch.Tensor:
        midpoint, _ = self._active_gate()
        return torch.where(
            (self.phase >= 2).unsqueeze(-1), self.berth_point, midpoint
        )

    def _dock_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distance = torch.norm(self.berth_point - self._com_xy(), dim=-1)
        dock_dot = torch.sum(self._forward_2d() * self.dock_heading, dim=-1)
        planar_speed = torch.norm(self._linear_velocity_world()[:, :2], dim=-1)
        conjunction = (
            (distance <= self.cfg.success_position_tolerance_m)
            & (dock_dot >= self._heading_dot_threshold)
            & (planar_speed <= self.cfg.success_speed_tolerance_mps)
        )
        return distance, dock_dot, planar_speed, conjunction

    def _phase_potential(
        self, positions: torch.Tensor, phases: torch.Tensor
    ) -> torch.Tensor:
        phase_index = phases.clamp(min=0, max=2)
        starts = torch.gather(self.stage_start_x, 1, phase_index.unsqueeze(-1)).squeeze(-1)
        ends = torch.gather(self.stage_end_x, 1, phase_index.unsqueeze(-1)).squeeze(-1)
        normalized = torch.clamp(
            (positions[:, 0] - starts) / (ends - starts).clamp_min(1.0e-6),
            min=0.0,
            max=1.0,
        )
        return normalized / 3.0

    def _get_observations(self) -> dict:
        current_xy = self._com_xy()
        target = self._phase_target()
        to_target = target - current_xy
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        direction = to_target / distance.clamp_min(1.0e-6)
        forward = self._forward_2d()
        dot = torch.sum(forward * direction, dim=-1, keepdim=True)
        cross = (
            forward[:, 0:1] * direction[:, 1:2]
            - forward[:, 1:2] * direction[:, 0:1]
        )
        phase_index = self.phase.clamp(max=2)
        span = torch.gather(
            self.stage_end_x - self.stage_start_x,
            1,
            phase_index.unsqueeze(-1),
        )
        distance_norm = torch.clamp(distance / span.clamp_min(1.0e-6), 0.0, 1.0)
        base = torch.hstack((dot, cross, distance_norm))
        base = torch.where((self.phase < 3).unsqueeze(-1), base, torch.zeros_like(base))

        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        cosine = torch.cos(self._ray_angles).view(1, -1, 1)
        sine = torch.sin(self._ray_angles).view(1, -1, 1)
        ray_directions = cosine * forward.unsqueeze(1) + sine * left.unsqueeze(1)
        ranges = ray_circle_ranges(
            current_xy,
            ray_directions,
            self.obstacle_centers,
            self.obstacle_radii,
            max_range_m=self.cfg.ray_max_range_m,
            active_mask=self.obstacle_active,
        )
        ranges_norm = ranges / self.cfg.ray_max_range_m

        phase_one_hot = functional.one_hot(self.phase, num_classes=4).float()
        dock_dot = torch.sum(forward * self.dock_heading, dim=-1, keepdim=True)
        dock_cross = (
            forward[:, 0:1] * self.dock_heading[:, 1:2]
            - forward[:, 1:2] * self.dock_heading[:, 0:1]
        )
        dock_alignment = torch.hstack((dock_dot, dock_cross))
        in_dock_phase = (self.phase == 2).unsqueeze(-1)
        dock_alignment = torch.where(
            in_dock_phase, dock_alignment, torch.zeros_like(dock_alignment)
        )
        dwell_fraction = (
            self._hold_steps.float().unsqueeze(-1) / float(self.required_hold_steps)
        ).clamp(0.0, 1.0)
        dwell_fraction = torch.where(
            in_dock_phase, dwell_fraction, torch.zeros_like(dwell_fraction)
        )
        observation = torch.hstack(
            (base, ranges_norm, phase_one_hot, dock_alignment, dwell_fraction)
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        """Ledger-capped progress under the phase active at step start."""
        current_xy = self._com_xy()
        reward_phase = self._phase_at_step_start
        active = reward_phase < 3
        potential = self._phase_potential(current_xy, reward_phase)
        progress = torch.where(
            active,
            potential - self._potential_at_step_start,
            torch.zeros_like(potential),
        )

        clearance = self._clearance()
        proximity = torch.clamp(
            (self.cfg.safe_clearance_m - clearance) / self.cfg.safe_clearance_m,
            min=0.0,
            max=1.0,
        )
        barrier = (
            self.cfg.reward_clearance_scale
            * self.control_step_s
            * proximity.square()
        )
        contact_now = clearance < 0.0
        contact_entry = active & contact_now & ~self._contact_prev
        self._contact_prev.copy_(
            torch.where(active, contact_now, self._contact_prev)
        )
        _, _, _, dock_conjunction = self._dock_state()
        dock_pay = (active & (reward_phase == 2) & dock_conjunction).float()
        return (
            self.cfg.reward_progress_scale * progress
            - barrier
            - self.cfg.reward_contact_entry_penalty * contact_entry.float()
            - self.cfg.reward_contact_dwell_penalty * (active & contact_now).float()
            + self.cfg.reward_dock_conjunction * dock_pay
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_xy = self._com_xy()
        phase_at_step_start = self._phase_at_step_start
        exit_progress_at_step_start = self._exit_progress_at_step_start
        prefix_active = ~self._m3

        travelled = torch.norm(current_xy - self._previous_xy, dim=-1)
        self.path_length += torch.where(
            prefix_active, travelled, torch.zeros_like(travelled)
        )
        clearance = self._clearance()
        self._min_clearance.copy_(
            torch.where(
                prefix_active,
                torch.minimum(self._min_clearance, clearance),
                self._min_clearance,
            )
        )
        self._contact_before_dock |= prefix_active & (clearance < 0.0)

        active_gate_midpoint, active_gate_normal = self._active_gate()
        gate_phase = phase_at_step_start <= 1
        self._gate_armed.copy_(
            torch.where(
                gate_phase,
                arm_gate_approach(
                    self._gate_armed,
                    self._previous_xy,
                    active_gate_midpoint,
                    active_gate_normal,
                    approach_distance_m=self.cfg.gate_approach_distance_m,
                ),
                self._gate_armed,
            )
        )
        crossed = gate_crossing_mask(
            self._previous_xy,
            current_xy,
            self._linear_velocity_world()[:, :2],
            active_gate_midpoint,
            active_gate_normal,
            self._gate_armed,
            half_width_m=self.cfg.gate_half_width_m,
            min_normal_speed_mps=self.cfg.gate_min_normal_speed_mps,
        ) & gate_phase

        gate1_crossed = (
            crossed & (phase_at_step_start == 0) & (exit_progress_at_step_start == 0)
        )
        gate2_crossed = (
            crossed & (phase_at_step_start == 0) & (exit_progress_at_step_start == 1)
        )
        field_exit_crossed = crossed & (phase_at_step_start == 1)
        self._exit_gate_progress.copy_(
            torch.where(gate1_crossed, torch.ones_like(self._exit_gate_progress), self._exit_gate_progress)
        )

        step_index = self.episode_length_buf.long()
        self._m1 |= gate2_crossed
        self._m1_step.copy_(
            torch.where(gate2_crossed, step_index, self._m1_step)
        )
        self.phase.copy_(torch.where(gate2_crossed, torch.ones_like(self.phase), self.phase))
        self._m2 |= field_exit_crossed
        self._m2_step.copy_(
            torch.where(field_exit_crossed, step_index, self._m2_step)
        )
        self.phase.copy_(
            torch.where(field_exit_crossed, torch.full_like(self.phase, 2), self.phase)
        )

        # A newly active gate gets a fresh latch, immediately armed when the
        # crossing sample is already at least 1 m before that next plane.
        gate_advanced = gate1_crossed | gate2_crossed | field_exit_crossed
        next_midpoint, next_normal = self._active_gate()
        reset_armed = torch.zeros_like(self._gate_armed)
        reset_armed = arm_gate_approach(
            reset_armed,
            current_xy,
            next_midpoint,
            next_normal,
            approach_distance_m=self.cfg.gate_approach_distance_m,
        )
        self._gate_armed.copy_(
            torch.where(
                gate_advanced & (self.phase <= 1),
                reset_armed,
                torch.where(gate_advanced, torch.zeros_like(self._gate_armed), self._gate_armed),
            )
        )

        _, _, _, dock_conjunction = self._dock_state()
        dock_active = (phase_at_step_start == 2) & ~self._m3
        self._hold_steps.copy_(
            torch.where(
                dock_active,
                torch.where(
                    dock_conjunction,
                    self._hold_steps + 1,
                    torch.zeros_like(self._hold_steps),
                ),
                self._hold_steps,
            )
        )
        self.hold_timer.copy_(self._hold_steps.float() * self.control_step_s)
        dock_completed = dock_active & (self._hold_steps >= self.required_hold_steps)
        self._m3 |= dock_completed
        self._m3_step.copy_(
            torch.where(dock_completed, step_index, self._m3_step)
        )
        self.phase.copy_(
            torch.where(dock_completed, torch.full_like(self.phase, 3), self.phase)
        )

        ordered_trace = (
            self._m1
            & self._m2
            & self._m3
            & (self._m1_step < self._m2_step)
            & (self._m2_step < self._m3_step)
        )
        success_now = dock_completed & ordered_trace & ~self._contact_before_dock
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(success_now, elapsed_s, self._first_success_time_s)
        )
        self._success |= success_now

        self._previous_xy.copy_(current_xy)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._episode_finished.copy_(time_out)
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
            completed_m1 = self._m1[completed_ids]
            completed_m2 = self._m2[completed_ids]
            completed_m3 = self._m3[completed_ids]
            self.episode_success[completed_ids] = success
            self.time_to_success[completed_ids] = success_time
            self.stage_reached[completed_ids] = self.phase[completed_ids]
            self.episode_path_length[completed_ids] = self.path_length[completed_ids]
            self.episode_min_clearance[completed_ids] = self._min_clearance[completed_ids]
            self.m1[completed_ids] = completed_m1
            self.m2[completed_ids] = completed_m2
            self.m3[completed_ids] = completed_m3

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
            self.extras["log"]["Episode/stage_reached"] = self.phase[
                completed_ids
            ].float().mean()
            self.extras["log"]["Episode/path_length_m"] = self.path_length[
                completed_ids
            ].mean()
            self.extras["log"]["Episode/min_clearance_m"] = self._min_clearance[
                completed_ids
            ].mean()
            self.extras["log"]["Stages/P(M1)"] = completed_m1.float().mean()
            self.extras["log"]["Stages/P(M2|M1)"] = (
                completed_m2[completed_m1].float().mean()
                if completed_m1.any()
                else torch.tensor(torch.nan, device=self.device)
            )
            self.extras["log"]["Stages/P(M3|M2)"] = (
                completed_m3[completed_m2].float().mean()
                if completed_m2.any()
                else torch.tensor(torch.nan, device=self.device)
            )

        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        max_obstacles = self.cfg.max_obstacles
        local_gates_np = np.zeros((num_resets, 3, 2), dtype=np.float32)
        gate_normals_np = np.zeros((num_resets, 3, 2), dtype=np.float32)
        local_berths_np = np.zeros((num_resets, 2), dtype=np.float32)
        local_centers_np = np.zeros(
            (num_resets, max_obstacles, 2), dtype=np.float32
        )
        radii_np = np.zeros((num_resets, max_obstacles), dtype=np.float32)
        active_np = np.zeros((num_resets, max_obstacles), dtype=np.bool_)
        count_np = np.zeros(num_resets, dtype=np.int64)
        geodesic_np = np.zeros(num_resets, dtype=np.float32)
        stage_starts_np = np.zeros((num_resets, 3), dtype=np.float32)
        stage_ends_np = np.zeros((num_resets, 3), dtype=np.float32)
        for row in range(num_resets):
            route = sample_harbor_route(
                self._layout_rng,
                max_attempts=self.cfg.layout_max_attempts,
                hazard_max_attempts=self.cfg.hazard_layout_max_attempts,
            )
            count = route.obstacle_count
            local_gates_np[row] = route.gate_midpoints
            gate_normals_np[row] = route.gate_normals
            local_berths_np[row] = route.berth_point
            local_centers_np[row, :count] = route.obstacle_centers
            radii_np[row, :count] = route.obstacle_radii
            active_np[row, :count] = True
            count_np[row] = count
            geodesic_np[row] = route.field_geodesic_length
            stage_starts_np[row] = (0.0, EXIT_GATE_X_M[1], route.field_exit[0])
            stage_ends_np[row] = (
                EXIT_GATE_X_M[1],
                route.field_exit[0],
                route.berth_point[0],
            )

        local_gates = torch.as_tensor(local_gates_np, device=self.device)
        gate_normals = torch.as_tensor(gate_normals_np, device=self.device)
        local_berths = torch.as_tensor(local_berths_np, device=self.device)
        local_centers = torch.as_tensor(local_centers_np, device=self.device)
        radii = torch.as_tensor(radii_np, device=self.device)
        active = torch.as_tensor(active_np, device=self.device)
        origins_xy = self.scene.env_origins[env_ids, :2]
        self.gate_midpoints[env_ids] = origins_xy.unsqueeze(1) + local_gates
        self.gate_normals[env_ids] = gate_normals
        self.berth_point[env_ids] = origins_xy + local_berths
        self.dock_heading[env_ids] = torch.tensor((1.0, 0.0), device=self.device)
        self.obstacle_centers[env_ids] = origins_xy.unsqueeze(1) + local_centers
        self.obstacle_radii[env_ids] = radii
        self.obstacle_active[env_ids] = active
        self.obstacle_count[env_ids] = torch.as_tensor(count_np, device=self.device)
        self.field_geodesic_length[env_ids] = torch.as_tensor(
            geodesic_np, device=self.device
        )
        self.stage_start_x[env_ids] = origins_xy[:, 0:1] + torch.as_tensor(
            stage_starts_np, device=self.device
        )
        self.stage_end_x[env_ids] = origins_xy[:, 0:1] + torch.as_tensor(
            stage_ends_np, device=self.device
        )

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
        root_state[:, :2] = origins_xy - com_offset_world[:, :2]
        root_state[:, 3:7] = spawn_quats
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self.phase[env_ids] = 0
        self._exit_gate_progress[env_ids] = 0
        self._m1[env_ids] = False
        self._m2[env_ids] = False
        self._m3[env_ids] = False
        self._m1_step[env_ids] = -1
        self._m2_step[env_ids] = -1
        self._m3_step[env_ids] = -1
        self._hold_steps[env_ids] = 0
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._previous_xy[env_ids] = origins_xy
        self._phase_at_step_start[env_ids] = 0
        self._exit_progress_at_step_start[env_ids] = 0
        self._potential_at_step_start[env_ids] = 0.0
        self._contact_prev[env_ids] = False
        self._contact_before_dock[env_ids] = False
        self._success[env_ids] = False
        self._first_success_time_s[env_ids] = torch.nan
        self._episode_finished[env_ids] = False
        self.actions[env_ids] = 0.0

        gate1 = self.gate_midpoints[env_ids, 0]
        normal1 = self.gate_normals[env_ids, 0]
        self._gate_armed[env_ids] = arm_gate_approach(
            torch.zeros(num_resets, dtype=torch.bool, device=self.device),
            origins_xy,
            gate1,
            normal1,
            approach_distance_m=self.cfg.gate_approach_distance_m,
        )
        initial_clearance = analytic_min_clearance(
            origins_xy,
            self.obstacle_centers[env_ids],
            self.obstacle_radii[env_ids],
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self.obstacle_active[env_ids],
        )
        self._min_clearance[env_ids] = initial_clearance
        self._contact_before_dock[env_ids] = initial_clearance < 0.0

        self._update_render_only_markers(
            env_ids, local_gates, local_centers, radii, active, local_berths
        )
