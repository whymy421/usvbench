# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for C4 x C5 ordered path following with hazards."""

from __future__ import annotations

from .._shared.termination_reason import publish_termination_reason

from collections.abc import Sequence
import math

import numpy as np
from .._shared.vehicles import physics_version
from .._shared.physics_v2 import add_implicit_hull_damping, WaveClock

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
from .._shared.journal_env_mixin import JournalEnvMixin, device_copies
from . import path_hazard_journal as journal_ph


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


class PathHazardEnv(JournalEnvMixin, DirectRLEnv):
    """Fixed-horizon ordered route task with forced centerline blockers."""

    cfg: PathHazardEnvCfg
    journal_family = "path_hazard"   # JournalEnvMixin carrier check (journal cfgs only)
    # G20b appearing obstacle: the journal task of this family wired for it (slot after the D7 slot)
    journal_appearing_tasks = ("path_hazard",)

    def __init__(self, cfg: PathHazardEnvCfg, render_mode: str | None = None, **kwargs):
        if getattr(cfg, "journal_contract_version", 1) == 2:
            self.vehicle_spec = self._journal_preflight(cfg)
        else:
            self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        physics_version(self.physics_cfg)  # Reject unsupported versions before simulation.
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

            self._sea = SeaState(
                self.cfg.sea_state, self.num_envs, self.device,
                physics_version=getattr(self.physics_cfg, "physics_version", 1),
            )
            if getattr(self.physics_cfg, "physics_version", 1) == 2:
                self._wave_clock = WaveClock(self.num_envs, self.device, self.cfg.sim.dt)

        # Diagnostics only -- none of these feed back into the reward.
        self._fee_steps = torch.zeros(self.num_envs, device=self.device)
        self._ep_prox_cost = torch.zeros(self.num_envs, device=self.device)
        self._clean_last_gate_this_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._ep_contact_entries = torch.zeros(self.num_envs, device=self.device)
        if getattr(self.cfg, "journal_contract_version", 1) == 2:
            self._journal_init()

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
        if getattr(self.physics_cfg, "physics_version", 1) == 1:
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
            if getattr(self.physics_cfg, "physics_version", 1) == 1:
                t = float(self.episode_length_buf[0]) * self.control_step_s
            else:
                t = self._wave_clock.time()
            wave_f, wave_t = self._sea.forces(
                self.robot.data.root_com_pos_w[:, :2],
                self.robot.data.root_com_vel_w[:, :2],
                t,
            )
            if getattr(self.physics_cfg, "physics_version", 1) == 2:
                self._wave_clock.advance()
            force_world += wave_f
            torque_world += wave_t

        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_world)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_world)
        if getattr(self.physics_cfg, "physics_version", 1) == 2:
            add_implicit_hull_damping(self, forces, torques, quat)
        self.robot.set_external_force_and_torque(forces, torques)

    def _com_xy(self) -> torch.Tensor:
        if getattr(self.cfg, "journal_contract_version", 1) == 2:
            # E5 C5 / D3 section 4: root-link pose = COM of a v2 asset; never
            # root_com_pos_w (lazily refreshed after a reset).
            return self.robot.data.root_link_pose_w[:, :2]
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
        if getattr(self.cfg, "journal_contract_version", 1) == 2:
            # B07 static mask | DETECTABLE D7 (G23c F12: no privileged undetected D7 signal in the reward;
            # D7 collisions reach the contract through the bridge's swept-contact latch) on the D4 hull polygon.
            return self._journal_pose_clearance(self.obstacle_active | self._d7_visible).to(torch.float32)
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
        if getattr(self.cfg, "journal_contract_version", 1) == 2:
            # B07 ray mask (static | visible D7), rays from the COM.
            return self._journal_ray_ranges()
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
        publish_termination_reason(
            self, terminated, time_out, success=self._success,
            collision=prefix_active & contact_now,
        )
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
                physics_version=self.physics_cfg.physics_version,
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
            if getattr(self.physics_cfg, "physics_version", 1) == 2:
                self._wave_clock.reset(env_ids)
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

    # ------------------------------------------------------------------
    # Journal branch (task contract version 2; CONTEXT_R4 round F, task F2).
    # Everything below runs only when cfg.journal_contract_version == 2;
    # __init__ binds five DirectRLEnv hooks to these methods, so the
    # legacy _pre_physics_step/_apply_action/_get_dones/_reset_idx/
    # _get_observations sources above stay byte-identical (D15) and
    # _get_rewards is shared (its _com_xy/_clearance/_ray_ranges_m calls
    # take the journal branches: root-link COM, hull polygon, B07 masks).
    # Call order = tasks/_shared/journal_env_mixin.py docstring (F0).
    # ------------------------------------------------------------------
    def _journal_suite(self, cfg):
        """The D7 suite exactly as journal_env_mixin.journal_setup resolves it."""
        resolved = getattr(cfg, "journal_conditions", None)
        if resolved is not None:
            return resolved.dynamic_config(cfg.vehicle)
        return journal_ph._dynamic().DynamicSuiteConfig(vessel_id=cfg.vehicle)

    def _journal_preflight(self, cfg):
        """D7 B02 + E2 asset preflight BEFORE super().__init__ builds the asset (fail closed)."""
        from pathlib import Path
        from types import SimpleNamespace

        from .._shared.conditions.journal_config import check_journal_cfg
        from .._shared.journal_vehicle import resolve_journal_vehicle

        if check_journal_cfg(cfg) != journal_ph.FAMILY or cfg.journal_task != journal_ph.JOURNAL_TASK:
            raise ValueError("PathHazardEnv carries only the path_hazard journal task")
        seed = getattr(cfg, "seed", None)
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("journal cfgs must carry an integer evaluation seed (CONTEXT_R4 D7)")
        vehicle = resolve_journal_vehicle(cfg.vehicle, **dict(getattr(cfg, "journal_vehicle_options", None) or {}))
        verified = vehicle.verify_asset(journal_ph.ASSETS_V2_DIR)
        if Path(cfg.robot_cfg.spawn.usd_path).resolve() != Path(verified).resolve():
            raise ValueError(f"robot usd_path {cfg.robot_cfg.spawn.usd_path} is not the hash-checked v2 asset "
                             f"{verified}")
        if int(getattr(cfg, "appear_count", 0) or 0) != 0:
            raise ValueError("legacy appearance is disabled in journal cfgs; D7 owns dynamic hazards (B02)")
        if cfg.obstacle_count != journal_ph.plans.STATIC_CAPACITY[journal_ph.JOURNAL_TASK]:
            raise ValueError("journal obstacle_count must be the journal static capacity (base sampler bound)")
        self._jph_pool = journal_ph.resolve_pool(cfg.journal_d7_pool, vessel=cfg.vehicle,
                                                 tier=int(cfg.journal_geometry_tier), suite=self._journal_suite(cfg),
                                                 path_variable=getattr(cfg, "journal_d7_pool_path", ""),
                                                 seed=getattr(cfg, "seed", None))       # G49: seed's pin table
        return SimpleNamespace(name=cfg.vehicle, asset_kind="rigid", bow_body_axis=vehicle.bow_axis)

    def _journal_init(self):
        """journal_setup + journal buffers (D7 H1: storage width = base count + 1 D7 slot)."""
        cfg = self.cfg
        self.journal_setup(d7_pool=self._jph_pool)
        # D16: keys E4 rejected keep their E5 layout and run the dynamic suite at D0 (no intruder);
        # the wrapper hands admitted rows to the E4 bridge unchanged and stamps the others
        # exposure = "not_admitted" with E4's reasons. The header counts them per dose.
        self._d7_bridge = journal_ph.NotAdmittedBridge(self._d7_bridge, int(self._d7_suite.dose))
        header_fields = getattr(self._jph_pool, "header_fields", None)
        if header_fields is not None:
            self.dynamic_hazard_header.update(header_fields())
        # D24: the pool's offered grid; an episode index >= its episodes per env runs at D0, not admitted
        self.dynamic_hazard_header["grid"] = journal_ph.check_grid(getattr(self._jph_pool, "grid", None))
        plan = cfg.journal_plan
        params = plan.params
        if plan.task != journal_ph.JOURNAL_TASK or plan.vessel != cfg.vehicle:
            raise ValueError("journal plan does not belong to this task/vessel")
        if int(params["segment_count"]) != int(cfg.num_waypoints):
            raise ValueError("journal gate count differs from cfg.num_waypoints")
        if abs(float(params["gate_radius_m"]) - float(cfg.goal_radius)) > 1e-12:
            raise ValueError("cfg.goal_radius must be the plan's gate radius")
        if int(cfg.ray_count) != int(params["ray_sensor_ray_count"]) or \
                abs(float(cfg.ray_max_range_m) - float(params["ray_sensor_max_range_m"])) > 1e-9:
            raise ValueError("journal ray sensor must come from cfg.journal_plan.params (D14)")
        com_offset = getattr(self.robot.data, "com_pos_b", None)
        if com_offset is not None and float(torch.as_tensor(com_offset).abs().max()) > 1.0e-4:
            raise ValueError("v2 assets have the actor origin at the COM; com_pos_b is not zero (E5 C5)")
        n, dev = self.num_envs, self.device
        record = journal_ph.plans.vessel_record(cfg.vehicle)
        self._jph_scale = float(params["length_scale"])
        self._jph_rect = journal_ph.HullRectangle.for_vessel(cfg.vehicle)
        self._jph_nominal_height = float(plan.nominal_com_height_m)
        self._jph_equilibrium_z = float(self.journal_vehicle.hydro_profile.hydro.equilibrium_com_z_m)
        if abs(self._jph_equilibrium_z - self._jph_nominal_height) > 1.0e-9:
            raise ValueError("profile equilibrium COM height differs from the contract's nominal height")
        self._jph_water_z = float(self.physics_cfg.water_surface_z)
        self._jph_limits = journal_ph.contracts.ValidityLimits(**plan.validity_limits)
        self._jph_gate_radius = float(params["gate_radius_m"])
        self._jph_oracle = journal_ph.OrderedPathTensorOracle(n, int(cfg.num_waypoints), dev, self._jph_limits,
                                                              motors=2)
        # E5 ThreadingPassage per gate portal: the journal gate bonus is paid only after the full hull
        # crossed the portal's exit plane cleanly (C1 finding 45), not at COM entry into the gate disk.
        self._jph_threading = journal_ph.ThreadingTensorTwin(n, int(cfg.num_waypoints), dev,
                                                             bonus=float(cfg.reference_reward_gate_bonus))
        self._jph_portal_origin = torch.zeros((n, cfg.num_waypoints, 2), dtype=torch.float64, device=dev)
        self._jph_portal_heading = torch.zeros((n, cfg.num_waypoints), dtype=torch.float64, device=dev)
        base = int(cfg.obstacle_count)
        width = base + int(journal_ph.plans.D7_RESERVED_SLOTS[journal_ph.JOURNAL_TASK])
        self._jph_d7_slots = tuple(range(base, width))
        if self._appearing_bridge is not None:
            # G20b: the appearing cylinder owns one more storage column after the D7 slot
            # (CapacityPlan.appearing_slots[0] = base_max + D7 reserve); inactive until its trigger
            if self._appearing_slot != width:
                raise ValueError(f"appearing slot {self._appearing_slot} is not the column after the D7 slots {width}")
            width += 1
            self.appearing_hazard_header["collision_model"] = (
                "path hazard static obstacle channel: once triggered the cylinder is in obstacle_active (its own "
                "storage column after the D7 slot), so the contract's swept hull clearance, the observation rays "
                "and the reward clearance see it exactly like a layout obstacle")
        self._jph_storage_width = width
        self.obstacle_centers = torch.zeros((n, width, 2), device=dev)
        self.obstacle_radii = torch.zeros((n, width), device=dev)
        self.obstacle_active = torch.zeros((n, width), dtype=torch.bool, device=dev)
        self._d7_physical = torch.zeros((n, width), dtype=torch.bool, device=dev)
        self._d7_visible = torch.zeros((n, width), dtype=torch.bool, device=dev)
        # G23c F8: the D7 slots answer the rays through the point-return proxy only
        from .._shared import d7_point_return as _d7pr
        self._jd7pr = _d7pr
        self._jd7_point_slots = _d7pr.slot_mask(width, self._jph_d7_slots, dev)
        self._jph_centers = torch.zeros((n, width, 2), dtype=torch.float64, device=dev)   # env-local, exact
        self._jph_radii = torch.zeros((n, width), dtype=torch.float64, device=dev)
        self._jph_waypoints = torch.zeros((n, cfg.num_waypoints, 2), dtype=torch.float64, device=dev)
        self._jph_bounds = torch.zeros((n, 4), dtype=torch.float64, device=dev)
        self._jph_deadline = torch.ones(n, dtype=torch.float64, device=dev)
        self._jph_prev_pose = torch.zeros((n, 3), dtype=torch.float64, device=dev)
        self._jph_layouts = [None] * n
        self._jph_exposure = [None] * n
        self._jph_family_hashes = [None] * n
        self.episode_contract_result = [None] * n
        self.journal_completed = [None] * n
        self._jph_distance_norm = float(cfg.segment_length_max)
        self._jph_yaw_rate_scale = float(cfg.yaw_rate_obs_scale_rad_s) / self._jph_scale
        # D15: bind the DirectRLEnv hooks for this instance only (legacy sources untouched).
        self._pre_physics_step = self._journal_pre_physics_step
        self._apply_action = self._journal_apply_action
        self._get_dones = self._journal_get_dones
        self._reset_idx = self._journal_reset_idx
        self._get_observations = self._journal_get_observations

    # -------------------------------------------------------------- state --
    def _jph_state(self, robot_data=None, env_origins=None):
        """Truth COM state, env-local, float64 (root-link pose = COM for v2 assets; E5 C5)."""
        data = self.robot.data if robot_data is None else robot_data
        origins = self.scene.env_origins if env_origins is None else env_origins
        pose = data.root_link_pose_w.to(torch.float64)
        vel = data.root_com_vel_w.to(torch.float64)
        com = pose[:, :3] - origins.to(torch.float64)
        quat = pose[:, 3:7]
        planar = torch.stack((com[:, 0], com[:, 1], journal_ph.yaw_from_quat(quat)), dim=-1)
        return com, quat, vel[:, :3], vel[:, 3:6], planar

    def _journal_pose_clearance(self, mask):
        _, _, _, _, planar = self._jph_state()
        return journal_ph.pose_clearance(self._jph_rect, planar[:, 0], planar[:, 1], planar[:, 2],
                                         self._jph_centers, self._jph_radii, mask)

    def _journal_swept_static_clearance(self, planar):
        clearance = journal_ph.swept_clearance(self._jph_rect, self._jph_prev_pose, planar, self._jph_centers,
                                               self._jph_radii, self.obstacle_active)
        self._jph_prev_pose = planar.clone()
        return clearance

    def _journal_ray_ranges(self):
        """36 rays from the COM over static | visible D7 obstacles (B07), metres, float32."""
        _, _, _, _, planar = self._jph_state()
        angles = planar[:, 2:3] + self._ray_angles.to(torch.float64).unsqueeze(0)
        directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        mask = self.obstacle_active | self._d7_visible
        ranges = ray_circle_ranges(planar[:, :2], directions, self._jph_centers, self._jph_radii,
                                   max_range_m=self.cfg.ray_max_range_m,
                                   active_mask=mask & ~self._jd7_point_slots)
        # G23c F8: the visible D7 intruder through the point-return proxy
        ranges = self._jd7pr.merge(ranges, planar[:, :2], planar[:, 2], self._jph_centers, self._jph_radii,
                                   mask & self._jd7_point_slots, int(self.cfg.ray_count),
                                   float(self.cfg.ray_max_range_m))
        return ranges.to(torch.float32)

    # ------------------------------------------------------ mixin hooks --
    def _journal_observe_physics(self, robot_data, env_origins, water_level_local):
        """Contract observe_physics on the PREVIOUS substep's state (D8 water level, swept hull clearance)."""
        com, quat, lin, ang, planar = self._jph_state(robot_data, env_origins)
        clearance = self._journal_swept_static_clearance(planar)
        thrust = None
        if self._journal_substeps and self.journal_last_step is not None:
            # thrust of the substep that produced this state (the first substep of a control
            # step re-observes the state already observed at the previous _get_dones)
            thrust = self.journal_last_step.motors.delivered_thrust
        self._jph_oracle.observe_physics(com, quat, lin, ang, water_level_local, self._jph_nominal_height,
                                         clearance=clearance, clearance_mask=self.obstacle_active.any(dim=-1),
                                         motor_thrust=thrust,
                                         dt_s=None if thrust is None else self.cfg.sim.dt)

    def _journal_family_scenario_hashes(self, env_id):
        hashes = self._jph_family_hashes[int(env_id)]
        if hashes is None:
            raise RuntimeError(f"env {int(env_id)} has no journal layout yet")
        return dict(hashes)

    def _journal_publish_appearing(self, samples) -> None:
        """G20b mixin hook: the appearing cylinder in its column; physical = visible = obstacle_active."""
        if not samples:
            return
        rows, xs, ys, rs, on = [], [], [], [], []
        for env_index, items in samples.items():
            for item in items:
                if int(item.slot) != self._appearing_slot or bool(item.physical) != bool(item.visible):
                    raise ValueError(f"appearing sample slot {item.slot} / masks do not match the appearing slot")
                rows.append(int(env_index))
                xs.append(float(item.x))
                ys.append(float(item.y))
                rs.append(float(item.radius_m))
                on.append(bool(item.physical))
        if not rows:
            return
        row_t = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        slot = self._appearing_slot
        local = torch.stack((torch.as_tensor(xs, dtype=torch.float64), torch.as_tensor(ys, dtype=torch.float64)),
                            -1).to(self.device)
        radius = torch.as_tensor(rs, dtype=torch.float64).to(self.device)
        self._jph_centers[row_t, slot] = local
        self._jph_radii[row_t, slot] = radius
        self.obstacle_centers[row_t, slot] = (local + self.scene.env_origins[row_t, :2].to(torch.float64)).to(
            self.obstacle_centers.dtype)
        self.obstacle_radii[row_t, slot] = radius.to(self.obstacle_radii.dtype)
        self.obstacle_active[row_t, slot] = torch.as_tensor(on, dtype=torch.bool).to(self.device)

    def _journal_appearing_geometry(self, env_id) -> str:
        """G20b mixin hook: geometry id of the env's current E5 layout (the appearing pool records it per key)."""
        layout = self._jph_layouts[int(env_id)]
        if layout is None:
            raise RuntimeError(f"env {int(env_id)} has no journal layout")
        return str(layout.d7_hook["geometry_id"])

    def _journal_publish_d7(self, samples):
        """B07: D7 samples into the reserved slot; physical/visible masks split; obstacle_active untouched."""
        if not samples:
            return
        keys = sorted(samples)
        d7_slots = self._jph_d7_slots
        if (d7_slots and keys == list(range(int(self.num_envs)))
                and d7_slots == tuple(range(d7_slots[0], d7_slots[-1] + 1))):
            # G11: every row, contiguous reserved slots -> the same elements cleared by basic slicing
            # (a fill kernel; the index tensors and the scalar index_put each cost a host sync before)
            self._d7_physical[:, d7_slots[0]:d7_slots[-1] + 1] = False
            self._d7_visible[:, d7_slots[0]:d7_slots[-1] + 1] = False
        else:
            rows = torch.as_tensor(keys, dtype=torch.long, device=self.device)
            slots = torch.as_tensor(d7_slots, dtype=torch.long, device=self.device)
            self._d7_physical[rows.unsqueeze(-1), slots.unsqueeze(0)] = False
            self._d7_visible[rows.unsqueeze(-1), slots.unsqueeze(0)] = False
        env, slot, xy, radius, physical, visible = [], [], [], [], [], []
        for i in sorted(samples):
            for s in samples[i]:
                if int(s.slot) not in self._jph_d7_slots:
                    raise ValueError(f"D7 sample slot {s.slot} outside the reserved D7 slots {self._jph_d7_slots}")
                env.append(int(i))
                slot.append(int(s.slot))
                xy.append((float(s.x), float(s.y)))
                radius.append(float(s.radius_m))
                physical.append(bool(s.physical))
                visible.append(bool(s.visible))
        if not env:
            return
        # G11: the six host tensors of the per-field as_tensor(..., device=) calls, built on the host
        # exactly as those calls build them, moved in ONE host-to-device copy (bytes verbatim)
        e, k, local, r, physical_t, visible_t = device_copies(
            [torch.as_tensor(env, dtype=torch.long), torch.as_tensor(slot, dtype=torch.long),
             torch.as_tensor(xy, dtype=torch.float64), torch.as_tensor(radius, dtype=torch.float64),
             torch.as_tensor(physical, dtype=torch.bool), torch.as_tensor(visible, dtype=torch.bool)], self.device)
        self._jph_centers[e, k] = local
        self._jph_radii[e, k] = r
        self.obstacle_centers[e, k] = (local + self.scene.env_origins[e, :2].to(torch.float64)).to(
            self.obstacle_centers.dtype)
        self.obstacle_radii[e, k] = r.to(self.obstacle_radii.dtype)
        self._d7_physical[e, k] = physical_t
        self._d7_visible[e, k] = visible_t

    # ------------------------------------------------------- env hooks --
    def _journal_pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.journal_pre_physics(actions)

    def _journal_apply_action(self) -> None:
        from .._shared.body_wrench import apply_body_wrench   # G6: no per-call deprecation warning
        forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins, self._jph_water_z)
        apply_body_wrench(self.robot, forces, torques)

    def _journal_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        view = self.journal_pre_dones(self.robot.data, self.scene.env_origins, self._jph_water_z)  # B06
        com, quat, lin, ang, planar = self._jph_state()
        static_clearance = self._journal_swept_static_clearance(planar)
        water = view.water_level_local
        oracle = self._jph_oracle
        oracle.observe_physics(com, quat, lin, ang, water, self._jph_nominal_height,
                               motor_thrust=self.journal_last_step.motors.delivered_thrust, dt_s=self.cfg.sim.dt)
        time_s = self.episode_length_buf.to(torch.float64) * self.control_step_s
        b = self._jph_bounds
        out_of_bounds = (com[:, 0] < b[:, 0]) | (com[:, 0] > b[:, 1]) | (com[:, 1] < b[:, 2]) | (com[:, 1] > b[:, 3])
        d7_contact = view.d7_contact.to(self.device)
        prefix_active = ~self._last_gate_reached
        old_gates = self.gates_passed.clone()
        # legacy-style ledgers (logging and the shared reward; the certificate uses the oracle)
        current_xy = self._com_xy()
        travelled = torch.norm(current_xy - self._previous_xy, dim=-1)
        self.path_length += torch.where(prefix_active, travelled, torch.zeros_like(travelled))
        self._previous_xy.copy_(current_xy)
        target_xy = self._current_waypoint_world()
        self._pre_transition_target_distance.copy_(torch.norm(target_xy - current_xy, dim=-1))
        contact_now = (static_clearance <= 0.0) | d7_contact
        self._min_clearance.copy_(torch.where(prefix_active, torch.minimum(
            self._min_clearance, static_clearance.to(self._min_clearance.dtype)), self._min_clearance))
        self._contact_before_last_gate |= prefix_active & contact_now
        codes = oracle.step(time_s, com, quat, lin, ang, water, self._jph_nominal_height, collision=d7_contact,
                            out_of_bounds=out_of_bounds, clearance=static_clearance,
                            clearance_mask=self.obstacle_active.any(dim=-1))
        self.gates_passed.copy_((oracle.next_gate - 1).to(self.gates_passed.dtype))
        # gate bonus of the shared reward = ThreadingPassage payment of the active gate portal
        rows = torch.arange(self.num_envs, device=self.device)
        portal = self._jph_threading.current.clamp(max=self.cfg.num_waypoints - 1)
        heading = self._jph_portal_heading[rows, portal]
        local_x, local_y = journal_ph.portal_local(com[:, :2], self._jph_portal_origin[rows, portal], heading)
        half_x, half_y = journal_ph.hull_half_extents(self._jph_rect, planar[:, 2] - heading)
        finite = torch.isfinite(local_x) & torch.isfinite(local_y) & torch.isfinite(half_x) & torch.isfinite(half_y)
        paid = self._jph_threading.update(local_x, local_y, half_x, half_y, oracle.last_contact, rows=finite)
        self._gates_passed_this_step.copy_(paid > 0.0)
        success_now = codes == journal_ph.SUCCESS
        self._first_success_time_s.copy_(torch.where(success_now, time_s.to(self._first_success_time_s.dtype),
                                                     self._first_success_time_s))
        self._success.copy_(oracle.success)
        self._last_gate_reached |= self.gates_passed >= self.cfg.num_waypoints
        self._clean_last_gate_this_step.copy_(success_now)
        terminated, truncated = self.journal_dones(codes)      # D9 publisher + B08 latch (before reset)
        self._episode_finished.copy_(terminated | truncated)
        # G11: the rows with a nonzero code, from the host code list the oracle step already transferred
        self._journal_export([i for i, c in enumerate(oracle.last_code_list) if c != 0])
        return terminated, truncated

    def _journal_export(self, rows):
        """Completed-episode export BEFORE any reset: contract result + journal_episode_record(i)."""
        if not rows:
            return
        for i in rows:
            result = self._jph_oracle.result(i)
            record = self.journal_episode_record(i)
            self.episode_contract_result[i] = result
            self.episode_success[i] = bool(result.episode_success)
            self.time_to_success[i] = float(result.time_to_success)
            self.episode_gates_passed[i] = int(result.metrics["ordered_gates_completed"])
            self.episode_path_length[i] = float(result.episode_path_length)
            self.episode_min_clearance[i] = float(result.metrics["min_clearance_m"])
            self.episode_xte_rms[i] = self.xte_rms[i]
            self.episode_route_length[i] = self.route_length[i]
            self.episode_scenario[i] = self._scenario_params[i]
            layout = self._jph_layouts[i]
            self.journal_completed[i] = {"contract": result.evaluator_fields(), "record": record,
                                         "d7_exposure": journal_ph.record_exposure(record["dynamic_hazards"]),
                                         "geometry_hash": layout.geometry_hash,
                                         "layout_rejections": dict(layout.rejections),
                                         "portals": self._jph_threading.summary(i)}
        ids = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        self.extras.setdefault("log", {})
        self.extras["log"]["Episode/success"] = self.episode_success[ids].float().mean()
        self.extras["log"]["Episode/gates_passed"] = self.episode_gates_passed[ids].float().mean()
        self.extras["log"]["Episode/path_length_m"] = self.episode_path_length[ids].mean()
        self.extras["log"]["Episode/min_clearance_m"] = self.episode_min_clearance[ids].mean()

    def _journal_reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        # task scenario counter (the condition layer's must follow it; D7 B03)
        self._scenario.reset_idx(env_ids)
        ids = env_ids.detach().cpu().tolist()
        episodes = self._scenario.episode_indices()[torch.as_tensor(ids, dtype=torch.long)]
        cfg, seed, tier = self.cfg, int(self.cfg.seed), int(self.cfg.journal_geometry_tier)
        n_reset, width, base = len(ids), self._jph_storage_width, int(cfg.obstacle_count)
        routes = torch.zeros((n_reset, cfg.num_waypoints + 1, 2), dtype=torch.float64)
        centers = torch.zeros((n_reset, width, 2), dtype=torch.float64)
        radii = torch.zeros((n_reset, width), dtype=torch.float64)
        active = torch.zeros((n_reset, width), dtype=torch.bool)
        counts = torch.zeros(n_reset, dtype=torch.long)
        bounds = torch.zeros((n_reset, 4), dtype=torch.float64)
        deadlines = torch.zeros(n_reset, dtype=torch.float64)
        route_len = torch.zeros(n_reset, dtype=torch.float64)       # G28: nominal route length (outage rule)
        headings = torch.zeros(n_reset, dtype=torch.float64)
        axes = torch.zeros(n_reset, dtype=torch.float64)
        for row, (i, episode) in enumerate(zip(ids, episodes.tolist())):
            try:
                status = journal_ph.pool_key_status(self._jph_pool, seed, i, episode)
            except KeyError as exc:
                detail = exc.args[0] if exc.args else ""
                raise ValueError(f"(seed {seed}, env {i}, episode {episode}) is outside the offered D7 grid "
                                 f"{self.dynamic_hazard_header.get('grid')}: {detail}; admit this seed/grid "
                                 "offline first (configuration error; a key inside the grid that E4 rejects "
                                 "runs at D0, D16; an episode index beyond the grid's episodes runs at D0 as "
                                 "not admitted, D24)") from None
            # D16: the E5 layout of the key, always (one keyed stream, no redraw)
            layout = journal_ph.journal_layout(cfg.vehicle, tier, seed, i, episode,
                                               protocol_version=self._scenario.protocol_version)
            # D24: a beyond-grid key is no pool key (the pool has no geometry record of it): no check
            pinned = None if status == journal_ph.BEYOND_OFFERED_GRID else (
                getattr(self._jph_pool, "geometry_hashes", {}).get((seed, i, episode)))
            if pinned is not None and pinned != layout.geometry_hash:
                raise ValueError(f"layout of (seed {seed}, env {i}, episode {episode}) differs from the one the "
                                 "D7 pool was admitted on")
            if layout.deadline_s > float(cfg.episode_length_s) + 1.0e-9:
                raise ValueError("scenario deadline exceeds the journal episode cap")
            count = layout.obstacle_count
            if count > base:
                raise ValueError("layout exceeds the static obstacle capacity")
            routes[row] = torch.as_tensor(journal_ph.route_points(layout), dtype=torch.float64)
            centers[row, :count] = torch.as_tensor(np.array(layout.centers, dtype=np.float64))
            radii[row, :count] = torch.as_tensor(np.array(layout.radii, dtype=np.float64))
            active[row, :count] = True
            counts[row] = count
            bounds[row] = torch.as_tensor(layout.bounds_m, dtype=torch.float64)
            deadlines[row] = float(layout.deadline_s)
            route_len[row] = float(layout.nominal_route_length_m)
            headings[row] = float(layout.start_heading_rad)
            axes[row] = journal_ph.reference_axis_rad(layout)
            self._jph_layouts[i] = layout
            self._jph_exposure[i] = journal_ph.ADMITTED if status == journal_ph.ADMITTED else journal_ph.NOT_ADMITTED
            spawn_z = self._jph_water_z + self._jph_equilibrium_z
            self._jph_family_hashes[i] = journal_ph.family_scenario_hashes(layout, spawn_z)
            self._scenario_params[i] = {
                "layout": {"geometry_hash": layout.geometry_hash, "d7_exposure": self._jph_exposure[i],
                           "obstacle_count": int(count), "deadline_s": float(layout.deadline_s),
                           "designated_blocker_segments": list(layout.extras["designated_blocker_segments"])},
                "spawn_pose": {"heading_rad": float(layout.start_heading_rad), "com_z_m": float(spawn_z)}}
        dev = self.device
        env_idx = env_ids
        origins = self.scene.env_origins[env_idx]
        origins_xy64 = origins[:, :2].to(torch.float64)
        waypoints = routes[:, 1:].to(dev)
        segments = torch.linalg.vector_norm(routes[:, 1:] - routes[:, :-1], dim=-1)
        self._jph_waypoints[env_idx] = waypoints
        self.waypoints[env_idx] = waypoints.to(self.waypoints.dtype)
        self.segment_lengths[env_idx] = segments.to(dev, self.segment_lengths.dtype)
        self.route_length[env_idx] = segments.sum(dim=-1).to(dev, self.route_length.dtype)
        self._jph_centers[env_idx] = centers.to(dev)
        self._jph_radii[env_idx] = radii.to(dev)
        self.obstacle_centers[env_idx] = (centers.to(dev) + origins_xy64.unsqueeze(1)).to(self.obstacle_centers.dtype)
        self.obstacle_radii[env_idx] = radii.to(dev, self.obstacle_radii.dtype)
        self.obstacle_active[env_idx] = active.to(dev)
        self.obstacle_count[env_idx] = counts.to(dev)
        self._d7_physical[env_idx] = False
        self._d7_visible[env_idx] = False
        self._jph_bounds[env_idx] = bounds.to(dev)
        self._jph_deadline[env_idx] = deadlines.to(dev)
        # spawn: COM at the start point, water level + equilibrium datum, bow along the start heading
        starts = routes[:, 0].to(dev)
        half = 0.5 * headings.to(dev)
        quats = torch.stack((torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)), -1)
        root_state = self.robot.data.default_root_state[env_idx].clone()
        root_state[:, 0:2] = (origins_xy64 + starts).to(root_state.dtype)
        root_state[:, 2] = (origins[:, 2].to(torch.float64) + self._jph_water_z + self._jph_equilibrium_z).to(
            root_state.dtype)
        root_state[:, 3:7] = quats.to(root_state.dtype)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_idx)
        # task buffers
        self.gates_passed[env_idx] = 0
        self.path_length[env_idx] = 0.0
        self.xte_rms[env_idx] = 0.0
        self._xte_squared_sum[env_idx] = 0.0
        self._xte_sample_count[env_idx] = 0
        self._success[env_idx] = False
        self._last_gate_reached[env_idx] = False
        self._contact_prev[env_idx] = False
        self._contact_before_last_gate[env_idx] = False
        self._gates_passed_this_step[env_idx] = False
        self._first_success_time_s[env_idx] = torch.nan
        self._episode_finished[env_idx] = False
        self._fee_steps[env_idx] = 0.0
        self._ep_prox_cost[env_idx] = 0.0
        self._ep_contact_entries[env_idx] = 0.0
        self.actions[env_idx] = 0.0
        start_world = (origins_xy64 + starts).to(self._previous_xy.dtype)
        self._previous_xy[env_idx] = start_world
        first_distance = torch.linalg.vector_norm(waypoints[:, 0] - starts, dim=-1).to(self._prev_target_distance.dtype)
        self._prev_target_distance[env_idx] = first_distance
        self._pre_transition_target_distance[env_idx] = first_distance
        start_pose = torch.cat((starts, headings.to(dev).unsqueeze(-1)), dim=-1)
        self._jph_prev_pose[env_idx] = start_pose
        initial = journal_ph.pose_clearance(self._jph_rect, start_pose[:, 0], start_pose[:, 1], start_pose[:, 2],
                                            self._jph_centers[env_idx], self._jph_radii[env_idx],
                                            self.obstacle_active[env_idx])
        self._min_clearance[env_idx] = initial.to(self._min_clearance.dtype)
        self._contact_before_last_gate[env_idx] = initial <= 0.0
        self._jph_oracle.reset(env_idx, routes.to(dev), self._jph_gate_radius, deadlines.to(dev), starts)
        origins_portal, headings_portal = journal_ph.portal_frames(routes)
        self._jph_portal_origin[env_idx] = origins_portal.to(dev)
        self._jph_portal_heading[env_idx] = headings_portal.to(dev)
        self._jph_threading.reset(env_idx, self._jph_gate_radius)
        # conditions, runtime and D7 rows (after the ROOT WRITE; F0 call order)
        self.journal_reset(env_idx, axes, episodes.clone(), outage_start_s=None, reference_route_length_m=route_len)
        self._update_render_only_obstacle_markers(env_idx, centers[:, :base].to(torch.float32), radii[:, :base],
                                                  active[:, :base])
        if self.cfg.visual.enable_waypoint_markers and bool((env_idx == 0).any().item()):
            try:
                self._create_env0_waypoint_markers()
            except Exception as exc:
                print(f"[WARN] Waypoint visualization could not be created: {exc}")
        if self.cfg.visual.enable_path_line and bool((env_idx == 0).any().item()):
            try:
                self._create_env0_path_line()
            except Exception as exc:
                print(f"[WARN] Path-line visualization could not be created: {exc}")

    def _journal_get_observations(self) -> dict:
        """kinematic46 journal layout from the D5 estimates (position, velocity, rays); no sea channel."""
        com, quat, lin, ang, planar = self._jph_state()
        rates = journal_ph.body_rates(quat, lin, ang)
        rays = self._journal_ray_ranges()
        est = self.journal_estimates(planar[:, :2], rates, rays)
        position = torch.as_tensor(est.position, dtype=torch.float64).to(self.device)
        velocity = torch.as_tensor(est.velocity, dtype=torch.float64).to(self.device)
        ranges = torch.as_tensor(est.rays, dtype=torch.float64).to(self.device)
        heading = planar[:, 2]
        forwards = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
        rows = torch.arange(self.num_envs, device=self.device)
        last = self.cfg.num_waypoints - 1
        current = self._jph_waypoints[rows, self.gates_passed.clamp(max=last)]
        following = self._jph_waypoints[rows, (self.gates_passed + 1).clamp(max=last)]

        def channels(target):
            relative = target - position
            distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
            direction = relative / distance.clamp_min(1.0e-6)
            dot = torch.sum(forwards * direction, dim=-1, keepdim=True)
            cross = forwards[:, 0:1] * direction[:, 1:2] - forwards[:, 1:2] * direction[:, 0:1]
            return dot, cross, distance / self._jph_distance_norm

        dot, cross, distance_norm = channels(current)
        next_dot, next_cross, next_distance_norm = channels(following)
        stage_norm = self.gates_passed.to(torch.float64).unsqueeze(-1) / self.cfg.num_waypoints
        kinematics = torch.stack((
            torch.clamp(velocity[:, 0] / SPEED_SCALE_MPS, -1.0, 1.0),
            torch.clamp(velocity[:, 1] / SPEED_SCALE_MPS, -1.0, 1.0),
            torch.clamp(velocity[:, 2] / self._jph_yaw_rate_scale, -1.0, 1.0)), dim=-1)
        observation = torch.cat((dot, cross, distance_norm, stage_norm, next_dot, next_cross, next_distance_norm,
                                 ranges / float(self.cfg.ray_max_range_m), kinematics), dim=-1)
        return {"policy": observation.to(torch.float32)}
