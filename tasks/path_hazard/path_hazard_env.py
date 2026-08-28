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
from .._shared.scenario_draws import (
    make_scenario_rng,
    spawn_heading,
    stamp_scenario,
)
from .._shared.vehicles import get_vehicle
from .path_hazard_env_cfg import PathHazardEnvCfg
from .path_hazard_geometry import (
    analytic_min_clearance,
    ray_circle_ranges,
    sample_layout,
)


# Primitive group for the numpy layout stream. Kept distinct from
# scenario_draws.GROUP_ROUTE ("route", the torch stream path_following draws its
# waypoint chain from) because this family samples its route AND its obstacle
# field together, by rejection, from one numpy Generator -- and because the
# certificate already records the two together under "layout" (see the
# stamp_scenario call in _reset_idx). Group names are hashed as TEXT, not as a
# positional index (tasks/_shared/scenario_rng.py:56-63), so introducing this
# name cannot disturb any stream that already exists.
GROUP_LAYOUT = "layout"


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

        # --- Controller-independent episode scenarios -----------------------
        # Two separate defects, both fixed by the same stream.
        #
        # SPAWN HEADING was a bare torch.rand on the GLOBAL torch RNG, which
        # skrl's Runner reseeds to the constant in the agent YAML after the env
        # is built -- so it did not follow --eval-seed at all.
        #
        # THE LAYOUT (route + obstacle field) did follow cfg.seed, because
        # self._layout_rng is seeded from it (:61-63), but it was ONE generator
        # advanced once per reset for the whole batch. That is seed-derived and
        # still not reproducible: this task terminates early on success or on a
        # prefix contact (see _get_dones), so two controllers reach reset k at
        # different times and in different groupings, consume a different number
        # of rejection-sampler draws, and are handed DIFFERENT layouts for the
        # same (eval seed, env, episode). Both primitives now ride a stream
        # keyed by (protocol version, cfg.seed, env index, per-env episode
        # index, group), which is keyed rather than sequential and therefore
        # cannot drift with consumption order.
        #
        # cfg.seed None -> None -> the historical lines, unchanged: the global
        # torch RNG for the heading and the single self._layout_rng sequence for
        # the layout.
        self._scenario = make_scenario_rng(self.cfg, self.num_envs, self.device)
        # _scenario_* is the RUNNING episode; episode_scenario_* is latched at
        # reset for the episode that just ENDED, matching episode_min_clearance.
        self._scenario_params = [{} for _ in range(self.num_envs)]
        self._scenario_hashes = [{} for _ in range(self.num_envs)]
        self.episode_scenario = [{} for _ in range(self.num_envs)]
        self.episode_scenario_hashes = [{} for _ in range(self.num_envs)]

        # Sea state (Wave variants only). Certified ids have no sea_state cfg
        # field, so getattr returns None and nothing here runs for them. The
        # field is a pure background force: it never touches the observation,
        # reward, or termination paths of this env.
        self._sea = None
        if getattr(self.cfg, "sea_state", None) is not None and self.cfg.sea_state.enable:
            from .._shared.sea_state import SeaState

            self._sea = SeaState(self.cfg.sea_state, self.num_envs, self.device)

        # Diagnostics only -- none of these feed back into the reward.
        self._fee_steps = torch.zeros(self.num_envs, device=self.device)
        self._ep_prox_cost = torch.zeros(self.num_envs, device=self.device)
        self._clean_last_gate_this_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._ep_contact_entries = torch.zeros(self.num_envs, device=self.device)

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

        if self._sea is not None:
            # Same world-frame entry point as the buoyancy/drag wrench (and as
            # the certified current path in the families that carry one), so
            # the wave field and a current simply add their water velocities.
            # Mirrors station_keeping_env.py:541-551.
            t = float(self.episode_length_buf[0]) * self.control_step_s
            wave_f, wave_t = self._sea.forces(
                self.robot.data.root_com_pos_w[:, :2],
                self.robot.data.root_com_vel_w[:, :2],
                t,
            )
            force_world += wave_f
            torque_world += wave_t

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

    def _ray_ranges_m(self) -> torch.Tensor:
        """Analytic 36-ray ranges in METERS, shared by obs and reward.

        Heading math replicates _heading_and_direction exactly so obs stays
        bit-identical. Recomputed at every call site -- never cached across
        callbacks (dones -> rewards -> resets -> obs ordering would serve
        stale rays to freshly reset envs).
        """
        forwards = math_utils.quat_apply(self._root_quat(), self.forward_vec)[:, :2]
        forwards = forwards / torch.norm(
            forwards, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        left = torch.stack((-forwards[:, 1], forwards[:, 0]), dim=-1)
        cosine = torch.cos(self._ray_angles).view(1, -1, 1)
        sine = torch.sin(self._ray_angles).view(1, -1, 1)
        ray_directions = cosine * forwards.unsqueeze(1) + sine * left.unsqueeze(1)
        return ray_circle_ranges(
            self._com_xy(),
            ray_directions,
            self.obstacle_centers,
            self.obstacle_radii,
            max_range_m=self.cfg.ray_max_range_m,
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

        ranges = self._ray_ranges_m()
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
        if getattr(self.cfg, "obs_kinematic", False):
            # v11 block: a scalar speed cannot separate surging, sliding and
            # spinning -- exactly the states that threading a gap needs.
            from .._shared.kinematics import body_planar_kinematics

            native_observation = torch.hstack(
                (
                    native_observation,
                    body_planar_kinematics(
                        forwards,
                        self.robot.data.root_com_vel_w[:, :3],
                        self.robot.data.root_ang_vel_w[:, 2],
                        yaw_rate_scale_rad_s=self.cfg.yaw_rate_obs_scale_rad_s,
                    ),
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

        # v6a: per-ray log proximity field replacing the quadratic graze tax
        # (kept behind reward_clearance_scale=0.0 for the ablation arm). Same
        # structure as hazard_nav v6a; band cap 1.05 m keeps path v2's
        # no-distant-tax narrowing verbatim.
        rays_m = self._ray_ranges_m()
        prox_cap_m = self.cfg.safe_clearance_m + self.cfg.half_beam_m
        prox_cost = (
            -self.cfg.reward_prox_scale
            * self.control_step_s
            * torch.log(
                rays_m.clamp(self.cfg.prox_ray_floor_m, prox_cap_m) / prox_cap_m
            ).mean(dim=-1)
        )

        # Diagnostics (no reward effect): counterfactual ledger for the CUT
        # v6 time fee (0.01 per pre-completion step, gate = the stage latch).
        self._fee_steps += (
            self.gates_passed < self.cfg.num_waypoints
        ).float()
        self._ep_prox_cost += prox_cost
        self._ep_contact_entries += contact_entry.float()

        reward = (
            self.cfg.reference_reward_progress_scale * progress
            + self._gates_passed_this_step.float()
            * self.cfg.reference_reward_gate_bonus
            - self.cfg.reward_clearance_scale
            * self.control_step_s
            * proximity.square()
            - prox_cost
            - self.cfg.reward_contact_entry_penalty * contact_entry.float()
            - self.cfg.reward_contact_dwell_penalty * contact_now.float()
        )

        # Re-anchor only after rewarding the step-start gate, exactly as in v4b.
        target_xy = self._current_waypoint_world()
        self._prev_target_distance.copy_(
            torch.norm(target_xy - self._com_xy(), dim=-1)
        )
        if self.cfg.terminate_on_outcome:
            # Terminal anchor. Without it, terminating on the last gate would
            # just end the reward stream early and teach the boat to stall.
            remaining_fraction = torch.clamp(
                1.0
                - self.episode_length_buf.float() / float(self.max_episode_length),
                min=0.0,
                max=1.0,
            )
            reward = reward + self._clean_last_gate_this_step.float() * (
                self.cfg.reward_goal_entry_bonus
                + self.cfg.reward_goal_time_bonus * remaining_fraction
            )
        if self.cfg.reward_reverse_action_scale > 0.0:
            reverse_action = torch.relu(-self.actions[:, 0])
            reward = reward - (
                self.cfg.reward_reverse_action_scale
                * self.control_step_s
                * reverse_action.square()
            )
        if self.cfg.reward_swift_scale > 0.0:
            speed_norm = (
                torch.norm(self.robot.data.root_com_vel_w[:, :2], dim=-1)
                / SPEED_SCALE_MPS
            )
            reward = reward - (
                self.cfg.reward_swift_scale
                * self.control_step_s
                * (1.0 - speed_norm.clamp(max=1.0))
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

        self._clean_last_gate_this_step.copy_(success_now)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.terminate_on_outcome:
            # v11 semantics: the episode ends on the outcome, and the terminal
            # bonus above is what keeps that from being a reward for stalling.
            terminated = success_now | (prefix_active & contact_now)
        else:
            # Fixed-horizon trajectory scoring: nothing terminates early.
            terminated = torch.zeros_like(time_out)
        # "Finished" has to mean "Isaac Lab is about to reset this env", which
        # is terminated OR timed out. The latch read time_out ALONE, so on the
        # v2 gym id -- terminate_on_outcome=True at
        # path_hazard_env_cfg.py:308 -- an episode that ended on its outcome
        # never entered the per-episode stats block of _reset_idx (:962). Two
        # consequences, both silent:
        #   * every latch that block writes (episode_success, time_to_success,
        #     episode_gates_passed, episode_path_length, episode_min_clearance,
        #     episode_xte_rms, episode_route_length) kept the value of whatever
        #     episode this env last TIMED OUT on, and under this cfg a success
        #     terminates by construction, so a successful episode could never
        #     latch its own success;
        #   * episode_scenario_hashes[env], latched in the same block (:977),
        #     therefore named a DIFFERENT episode than the record the evaluator
        #     wrote it into (scripts/eval_v6_frozen.py:213-215) -- a certificate
        #     row for episode k carrying episode j's digests, which is exactly
        #     the confusion the digests exist to rule out.
        # Nothing about the OUTCOME moves here: success_now, _success and the
        # (terminated, time_out) pair returned below are untouched, so which
        # episodes count as successes and how every statistic is computed are
        # unchanged; only which episodes reach the recording block changes. On
        # every cfg with terminate_on_outcome=False (v1 and the certified
        # champion) `terminated` is all-False, so `time_out | terminated` is
        # bit-identical to the line it replaces.
        self._episode_finished.copy_(time_out | terminated)
        return terminated, time_out

    def _layout_generator(self, env_index: int) -> np.random.Generator:
        """The numpy stream this env's CURRENT episode layout is drawn from.

        Seeded env: a fresh ``np.random.Generator`` keyed by (protocol version,
        cfg.seed, env index, this env's episode counter, "layout"), built by
        ``ScenarioRNG.numpy_rng`` off the same blake2b key algebra as the torch
        streams. ``_reset_idx`` calls ``reset_idx`` before this, so the counter
        already names the episode about to start. Because the key -- not the
        number of draws taken so far -- decides the stream, the layout for a
        given (eval seed, env, episode) is the same whether the previous
        episode ran to the horizon or ended on step three, and the same whether
        this env reset alone or with sixty-three others.

        Unseeded env (cfg.seed None, so make_scenario_rng returned None): the
        historical single advancing generator, byte for byte.
        """
        if self._scenario is None:
            return self._layout_rng
        return self._scenario.numpy_rng(GROUP_LAYOUT, int(env_index))

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
            for completed in completed_ids.tolist():
                self.episode_scenario[completed] = self._scenario_params[completed]
                self.episode_scenario_hashes[completed] = self._scenario_hashes[
                    completed
                ]

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
            self.extras["log"]["Episode/counterfactual_fee"] = (
                0.01 * self._fee_steps[completed_ids].mean()
            )
            self.extras["log"]["Episode/reward_prox_cost_sum"] = (
                self._ep_prox_cost[completed_ids].mean()
            )
            self.extras["log"]["Episode/contact_entries"] = (
                self._ep_contact_entries[completed_ids].mean()
            )

        super()._reset_idx(env_ids)

        # New episode for these envs: advance their per-env episode counter and
        # reseed their primitive streams. MUST precede the spawn-heading draw.
        if self._scenario is not None:
            self._scenario.reset_idx(env_ids)

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
        _ob = getattr(self.cfg, "on_line_blockers_override", 0)
        # <0 = force ZERO on-line blockers (line-avoid tier-0 probe);
        # 0 = frozen v1 default of three; 1..4 = pinned count. Non-negative
        # values behave bit-identically to the pre-sentinel plumbing.
        _blockers_override = 0 if _ob < 0 else (int(_ob) or None)
        # One generator per (env, episode) instead of one shared generator
        # advanced per reset. Everything below -- the rejection sampler, its
        # attempt budget, its K-reduction ladder, every range it samples -- is
        # untouched; only the Generator handed to it changes, and only when the
        # env is seeded. See _layout_generator.
        reset_env_indices = env_ids.tolist()
        for row in range(num_resets):
            layout = sample_layout(
                rng=self._layout_generator(reset_env_indices[row]),
                max_attempts=self.cfg.layout_max_attempts,
                obstacle_count=self.cfg.obstacle_count,
                blockers_override=_blockers_override,
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

        # Uniform on [0, 2*pi) exactly as before; only the stream changed.
        spawn_headings = spawn_heading(self._scenario, env_ids, self.device)
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
        self._fee_steps[env_ids] = 0.0
        self._ep_prox_cost[env_ids] = 0.0
        self._ep_contact_entries[env_ids] = 0.0
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

        # Fresh sea for the episode that starts now (Wave variants only), off
        # the protocol's per-(env, episode) "wave" stream so eval seeds control
        # the sea; unseeded envs fall back to the historical global-RNG line
        # inside SeaState.resample. Stamp mirrors station_keeping_env.py:829-844
        # (every entry is a raw draw or, for direction_rad, recoverable beside
        # mean_direction_rad by subtraction).
        wave_scenario: dict[str, dict[str, object]] = {}
        if self._sea is not None:
            self._sea.resample(env_ids, scenario=self._scenario)
            wave_scenario["wave"] = {
                "hs_m": self._sea.hs[env_ids],
                "tp_s": self._sea.tp[env_ids],
                "gamma": self._sea.gamma[env_ids],
                "mean_direction_rad": self._sea.mean_direction[env_ids],
                "phase_rad": self._sea.phase[env_ids],
                "direction_rad": self._sea.direction[env_ids],
            }

        # Resolved scenario for the episode that starts now. Both primitives
        # are on the protocol: "spawn_pose" off the torch stream, "layout" off
        # the per-(env, episode) numpy stream of _layout_generator. Values
        # only, never the key.
        #
        # The layout entries are the ACCEPTED sample, not the raw uniforms the
        # rejection sampler burned to reach it -- unlike the current or the
        # heading chain there is no shorter raw form to record, because
        # sample_layout draws an unbounded number of candidates and returns the
        # first feasible one. The accepted geometry IS the scenario, and it is
        # what the certificate has to be able to compare episode by episode.
        for env_index, resolved, hashes in stamp_scenario(
            env_ids,
            {
                "spawn_pose": {"heading_rad": spawn_headings},
                "layout": {
                    "waypoints_m": waypoints,
                    "obstacle_centers_m": local_centers,
                    "obstacle_radii_m": radii,
                    "obstacle_count": self.obstacle_count[env_ids],
                },
                **wave_scenario,
            },
        ):
            self._scenario_params[env_index] = resolved
            self._scenario_hashes[env_index] = hashes

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
