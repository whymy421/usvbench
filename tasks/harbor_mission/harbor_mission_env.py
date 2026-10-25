# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for Task B: Ordered Harbor Mission."""

from __future__ import annotations

from .._shared.termination_reason import publish_termination_reason

from collections.abc import Sequence
import math

import numpy as np
from .._shared.vehicles import physics_version
from .._shared.physics_v2 import add_implicit_hull_damping, WaveClock

import torch
import torch.nn.functional as functional

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.restoring import restoring_torque_body
from .._shared.scenario_draws import make_scenario_rng, stamp_scenario
from .._shared.scenario_draws_hazard import (
    GROUP_LAYOUT,
    GROUP_SPAWN,
    episode_layout_rng,
    spawn_jitter_offsets,
)
from .._shared.vehicles import get_vehicle
from .._shared.journal_env_mixin import JournalEnvMixin, device_copies, host_copies
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


class HarborMissionEnv(JournalEnvMixin, DirectRLEnv):
    """Fixed-horizon ordered exit, hazard-transit, and docking mission."""

    cfg: HarborMissionEnvCfg
    journal_family = "harbor_mission"
    # G20b appearing obstacle: the journal task of this family wired for it (column after the D7 slot)
    journal_appearing_tasks = ("harbour_mission",)

    def __init__(
        self, cfg: HarborMissionEnvCfg, render_mode: str | None = None, **kwargs
    ):
        if getattr(cfg, "journal_contract_version", 1) == 2:
            # Journal v2 vessel (ANCHORS_E5 C6): resolved BEFORE super().__init__
            # builds the asset; the legacy registry never sees the v2 name.
            self.vehicle_spec = _journal_vehicle_spec(cfg)
        else:
            self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        physics_version(self.physics_cfg)  # Reject unsupported versions before simulation.
        # UNSEEDED BEHAVIOUR, UNCHANGED AND DOCUMENTED RATHER THAN FIXED: with
        # cfg.seed None this falls back to cfg.layout_seed, which defaults to
        # the CONSTANT 0 (harbor_mission_env_cfg.py:181), so an unseeded run
        # replays one fixed route sequence instead of drawing fresh entropy.
        # This generator remains the whole scenario source on that path; the
        # migration below changes only the SEEDED path, which is the one every
        # evaluator takes (scripts/eval_v6_frozen.py:104 sets env_cfg.seed).
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
        self._m1_at_step_start = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._m2_at_step_start = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._m3_at_step_start = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
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
        self._max_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._phase_time_s = torch.zeros((self.num_envs, 4), device=self.device)
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
        self.episode_max_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_phase_time_s = torch.zeros(
            (self.num_envs, 4), device=self.device
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

        # v11 recipe: set in _get_dones on the stage-terminal clean milestone,
        # consumed by _get_rewards the same step (dones -> rewards ordering).
        self._stage_goal_entry_this_step = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # --- Controller-independent episode scenarios -----------------------
        # The route and the staged-spawn jitter both came off self._layout_rng
        # (:65): ONE generator, advanced once per reset for whatever batch
        # happened to be resetting. That is seed-derived and still not
        # reproducible across controllers, because this family terminates EARLY
        # -- on a milestone and on contact (_get_dones at :1133 and :1140; the
        # certified HarborStage/HarborDockPhase cfgs set both,
        # harbor_mission_env_cfg.py:294-295 and :338-339). Two controllers therefore arrive at reset k at
        # different times and in different groupings, consume a different number
        # of rejection-sampler draws inside sample_harbor_route, and are handed
        # DIFFERENT routes for the same (eval seed, env, episode). The sibling
        # family measured this directly: three controllers agreed on only
        # 53.8% / 56.9% / 55.6% of shared (env, episode) slots on fields that
        # depend on the layout alone. Both primitives now ride a stream keyed by
        # (protocol version, cfg.seed, env index, per-env episode index, group).
        #
        # cfg.seed None -> None -> the historical single self._layout_rng
        # sequence, unchanged (see the comment at :56-62 for what it is).
        self._scenario = make_scenario_rng(self.cfg, self.num_envs, self.device)
        # _scenario_* is the RUNNING episode; episode_scenario_* is latched at
        # reset for the episode that just ENDED, matching episode_min_clearance,
        # which is where scripts/eval_v6_frozen.py:213-215 reads it.
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
        # v6: milestone latches at step start, so the reward can pay each
        # rising edge exactly once (one-transition form; the phase one-hot in
        # obs carries the same information, keeping the bonus augmented-Markov).
        self._m1_at_step_start.copy_(self._m1)
        self._m2_at_step_start.copy_(self._m2)
        self._m3_at_step_start.copy_(self._m3)

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

    def _ray_ranges_m(self) -> torch.Tensor:
        """Analytic 36-ray ranges in METERS, shared by obs and reward.

        Recomputed at every call site -- never cached across callbacks
        (dones -> rewards -> resets -> obs ordering would serve stale rays
        to freshly reset envs).
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
        if self.cfg.obs_kinematic:
            # v11 kinematic channels: body-frame surge, sway, yaw rate.
            velocity_2d = self._linear_velocity_world()[:, :2]
            left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
            surge = torch.sum(velocity_2d * forward, dim=-1, keepdim=True)
            sway = torch.sum(velocity_2d * left, dim=-1, keepdim=True)
            from .._shared.obs_superset import SPEED_SCALE_MPS as _SPD
            surge_norm = torch.clamp(surge / _SPD, min=-1.0, max=1.0)
            sway_norm = torch.clamp(sway / _SPD, min=-1.0, max=1.0)
            yaw_rate = self.robot.data.root_ang_vel_w[:, 2:3]
            yaw_rate_norm = torch.clamp(
                yaw_rate / self.cfg.yaw_rate_obs_scale_rad_s, min=-1.0, max=1.0
            )
            base = torch.hstack((base, surge_norm, sway_norm, yaw_rate_norm))

        ranges = self._ray_ranges_m()
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
        # v6: per-ray log proximity field replacing the quadratic graze tax
        # (kept above behind reward_clearance_scale=0.0 for the ablation arm).
        # Same band/coefficient as the certified hazard_nav v6a (cap = 0.90 +
        # 0.45 = 1.35 m, w = 4.1). Zeroed once the mission is complete.
        rays_m = self._ray_ranges_m()
        prox_cap_m = self.cfg.safe_clearance_m + self.cfg.half_beam_m
        prox_cost = (
            -self.cfg.reward_prox_scale
            * self.control_step_s
            * torch.log(
                rays_m.clamp(self.cfg.prox_ray_floor_m, prox_cap_m) / prox_cap_m
            ).mean(dim=-1)
        ) * active.float()

        # v6 milestone bonuses (the Task-A "decisive entry" lesson, applied at
        # the stage gates): one-time payout on each latch rising edge. The
        # value cliff at the boundary is what cures boundary hesitation; the
        # phase one-hot in obs carries the latch, so this is one-transition
        # legal. +25 = one contact entry = proven-safe event scale.
        milestone_edges = (
            (self._m1 & ~self._m1_at_step_start).float()
            + (self._m2 & ~self._m2_at_step_start).float()
            + (self._m3 & ~self._m3_at_step_start).float()
        )
        milestone_bonus = self.cfg.reward_milestone_bonus * milestone_edges

        contact_now = clearance < 0.0
        contact_entry = active & contact_now & ~self._contact_prev
        self._contact_prev.copy_(
            torch.where(active, contact_now, self._contact_prev)
        )
        _, _, _, dock_conjunction = self._dock_state()
        dock_pay = (active & (reward_phase == 2) & dock_conjunction).float()

        # v11 stage-terminal bonus: entry + time-remaining scaled, paid once
        # on the clean stage milestone (flag set by _get_dones this step).
        remaining_fraction = torch.clamp(
            (self.max_episode_length - self.episode_length_buf.float())
            / max(float(self.max_episode_length), 1.0),
            min=0.0,
            max=1.0,
        )
        stage_bonus = self._stage_goal_entry_this_step.float() * (
            self.cfg.reward_stage_goal_bonus
            + self.cfg.reward_stage_time_bonus * remaining_fraction
        )

        return (
            self.cfg.reward_progress_scale * progress
            - barrier
            - prox_cost
            + milestone_bonus
            + stage_bonus
            - self.cfg.reward_contact_entry_penalty * contact_entry.float()
            - self.cfg.reward_contact_dwell_penalty * (active & contact_now).float()
            + self.cfg.reward_dock_conjunction * dock_pay
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_xy = self._com_xy()
        phase_at_step_start = self._phase_at_step_start
        exit_progress_at_step_start = self._exit_progress_at_step_start
        prefix_active = ~self._m3
        self._phase_time_s += (
            functional.one_hot(phase_at_step_start, num_classes=4).float()
            * self.control_step_s
        )

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
        self._max_phase.copy_(torch.maximum(self._max_phase, self.phase))

        ordered_trace = (
            self._m1
            & self._m2
            & self._m3
            & (self._m1_step < self._m2_step)
            & (self._m2_step < self._m3_step)
        )
        success_full = dock_completed & ordered_trace & ~self._contact_before_dock

        # Staged curriculum: the stage-terminal milestone (M1/M2/M3 by
        # mission_depth) scores the episode; a clean terminal step also pays
        # the v11 entry + time bonus and (if configured) ends the episode.
        depth = int(self.cfg.mission_depth)
        if depth == 1:
            stage_done_now = gate2_crossed
            stage_success_now = gate2_crossed & ~self._contact_before_dock
        elif depth == 2:
            stage_done_now = field_exit_crossed
            stage_success_now = (
                field_exit_crossed
                & self._m1
                & (self._m1_step < self._m2_step)
                & ~self._contact_before_dock
            )
        else:
            stage_done_now = dock_completed
            stage_success_now = success_full
        self._stage_goal_entry_this_step.copy_(stage_success_now)

        success_now = stage_success_now
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(success_now, elapsed_s, self._first_success_time_s)
        )
        self._success |= success_now

        self._previous_xy.copy_(current_xy)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        if self.cfg.terminate_on_milestone:
            terminal_milestone = stage_done_now
            if self.cfg.stage2_full_mission_rewards and depth == 2:
                # Keep Stage 2 success pinned to M2, but continue through M3 so
                # phase-2 berth-approach and docking rewards are not gated off.
                terminal_milestone = dock_completed
            terminated = terminated | terminal_milestone
        if self.cfg.terminate_on_contact:
            terminated = terminated | (prefix_active & (clearance < 0.0))
        self._episode_finished.copy_(terminated | time_out)
        publish_termination_reason(
            self, terminated, time_out, success=self._success,
            collision=(prefix_active & (clearance < 0.0))
            if self.cfg.terminate_on_contact else None,
        )
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
            self.episode_max_phase[completed_ids] = self._max_phase[completed_ids]
            self.episode_phase_time_s[completed_ids] = self._phase_time_s[completed_ids]
            self.time_to_success[completed_ids] = success_time
            self.stage_reached[completed_ids] = self.phase[completed_ids]
            self.episode_path_length[completed_ids] = self.path_length[completed_ids]
            self.episode_min_clearance[completed_ids] = self._min_clearance[completed_ids]
            self.m1[completed_ids] = completed_m1
            self.m2[completed_ids] = completed_m2
            self.m3[completed_ids] = completed_m3
            # Same latch, same moment, as episode_min_clearance above: the
            # evaluator reads the FINISHED episode's scenario alongside the
            # finished episode's metrics.
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

        # New episode for these envs: advance their per-env episode counter and
        # reseed their primitive streams. MUST precede every draw below.
        if self._scenario is not None:
            self._scenario.reset_idx(env_ids)

        num_resets = len(env_ids)
        reset_env_indices = env_ids.tolist()
        max_obstacles = self.cfg.max_obstacles
        local_gates_np = np.zeros((num_resets, 3, 2), dtype=np.float32)
        gate_normals_np = np.zeros((num_resets, 3, 2), dtype=np.float32)
        local_berths_np = np.zeros((num_resets, 2), dtype=np.float32)
        local_exits_np = np.zeros((num_resets, 2), dtype=np.float32)
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
            # One generator per (env, episode) instead of one shared generator
            # advanced per reset. sample_harbor_route itself -- its rejection
            # loop, both attempt budgets, every range it samples, the BFS
            # feasibility oracle -- is untouched; only the Generator handed to it
            # changes, and only when the env is seeded. See episode_layout_rng.
            route = sample_harbor_route(
                episode_layout_rng(
                    self._scenario, self._layout_rng, reset_env_indices[row]
                ),
                max_attempts=self.cfg.layout_max_attempts,
                hazard_max_attempts=self.cfg.hazard_layout_max_attempts,
            )
            count = route.obstacle_count
            local_gates_np[row] = route.gate_midpoints
            gate_normals_np[row] = route.gate_normals
            local_berths_np[row] = route.berth_point
            local_exits_np[row] = route.field_exit
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

        # spawn_phase > 0 drops the boat straight into a later stage. The dock
        # specialist trained from the harbor mouth never reached the dock phase
        # in 128 certification episodes, so it was handed the dock with a policy
        # that had never been in that state; spawning at the hand-off point puts
        # training on the state distribution the composite actually produces.
        start_phase = int(getattr(self.cfg, "spawn_phase", 0))
        # Resolved spawn jitter for the certificate stamp, present only when a
        # draw actually happened: the default cfg has spawn_phase 0 and
        # spawn_phase_jitter_m 0.0 (harbor_mission_env_cfg.py:187-188), draws no
        # randomness here, and so contributes no spawn_pose hash.
        spawn_scenario: dict[str, dict[str, object]] = {}
        if start_phase >= 2:
            jitter_m = float(getattr(self.cfg, "spawn_phase_jitter_m", 0.0))
            # Uniform over the disc of radius jitter_m exactly as before -- the
            # angle, then the sqrt-scaled radius that keeps it area-uniform, and
            # no draw at all when the jitter is zero. Only the stream changed.
            offsets = spawn_jitter_offsets(
                self._scenario, self._layout_rng, reset_env_indices, jitter_m
            )
            if jitter_m > 0.0:
                spawn_scenario[GROUP_SPAWN] = {"jitter_offset_m": offsets.tolist()}
            spawn_xy = origins_xy + torch.as_tensor(
                local_exits_np + offsets, device=self.device
            )
        else:
            spawn_xy = origins_xy

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
        root_state[:, :2] = spawn_xy - com_offset_world[:, :2]
        root_state[:, 3:7] = spawn_quats
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self.phase[env_ids] = start_phase
        # Both exit gates count as crossed, otherwise the gate machinery would
        # re-arm behind a boat that starts past them.
        self._exit_gate_progress[env_ids] = 2 if start_phase >= 1 else 0
        self._m1[env_ids] = start_phase >= 1
        self._m2[env_ids] = start_phase >= 2
        self._m3[env_ids] = False
        self._m1_at_step_start[env_ids] = False
        self._m2_at_step_start[env_ids] = False
        self._m3_at_step_start[env_ids] = False
        self._stage_goal_entry_this_step[env_ids] = False
        self._m1_step[env_ids] = -1
        self._m2_step[env_ids] = -1
        self._m3_step[env_ids] = -1
        self._hold_steps[env_ids] = 0
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._max_phase[env_ids] = start_phase
        self._phase_time_s[env_ids] = 0.0
        self._previous_xy[env_ids] = spawn_xy
        self._phase_at_step_start[env_ids] = start_phase
        self._exit_progress_at_step_start[env_ids] = 2 if start_phase >= 1 else 0
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
            spawn_xy,
            gate1,
            normal1,
            approach_distance_m=self.cfg.gate_approach_distance_m,
        )
        initial_clearance = analytic_min_clearance(
            spawn_xy,
            self.obstacle_centers[env_ids],
            self.obstacle_radii[env_ids],
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self.obstacle_active[env_ids],
        )
        self._min_clearance[env_ids] = initial_clearance
        self._contact_before_dock[env_ids] = initial_clearance < 0.0

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

        # Resolved scenario for the episode that starts now. Values only, never
        # the key: a key-salted hash would make two eval seeds differ by
        # construction and could never expose two runs that genuinely drew the
        # same paper (tasks/_shared/scenario_draws.py:49-57).
        #
        # The layout entries are the ACCEPTED route, not the raw uniforms the
        # rejection sampler burned to reach it -- sample_harbor_route draws an
        # unbounded number of candidates and returns the first feasible one, so
        # the accepted geometry IS the scenario. field_geodesic_m is carried
        # because it is the harbor analogue of the layout-only fields a
        # cross-controller audit compares
        # (scripts/check_scenario_independence.py:175).
        for env_index, resolved, hashes in stamp_scenario(
            env_ids,
            {
                GROUP_LAYOUT: {
                    "gate_midpoints_m": local_gates,
                    "gate_normals": gate_normals,
                    "berth_point_m": local_berths,
                    "field_exit_m": local_exits_np.tolist(),
                    "obstacle_centers_m": local_centers,
                    "obstacle_radii_m": radii,
                    "obstacle_count": count_np.tolist(),
                    "field_geodesic_m": geodesic_np.tolist(),
                },
                **spawn_scenario,
                **wave_scenario,
            },
        ):
            self._scenario_params[env_index] = resolved
            self._scenario_hashes[env_index] = hashes

        self._update_render_only_markers(
            env_ids, local_gates, local_centers, radii, active, local_berths
        )

    # ------------------------------------------------------------------------
    # Journal contract v2 (task F6). NEW methods only: _journal_init binds them
    # over _pre_physics_step / _apply_action / _get_dones / _reset_idx /
    # _get_observations (and the _com_xy / _clearance / _ray_ranges_m helpers
    # that the unchanged _get_rewards reads), so every legacy method source
    # stays byte-identical (CONTEXT_R4 D15). Call order = journal_env_mixin
    # docstring (EXACT CALL ORDER). Rewards, termination and metrics use truth
    # (root-link pose = COM for v2 assets, CONTEXT_R3 decision 12); the
    # observation uses the D5 estimates only.
    # ------------------------------------------------------------------------
    def _journal_init(self) -> None:
        from . import harbor_d7_pool as hpool
        from . import harbor_journal as hj
        from .._shared import journal_cfg_v2 as plans
        from .._shared import journal_contracts_v2 as contracts
        from .._shared.conditions import dynamic

        cfg = self.cfg
        n = self.num_envs
        plan = cfg.journal_plan
        if plan.task != hj.TASK or plan.vessel != cfg.vehicle:
            raise ValueError("journal harbour env needs the harbour_mission plan of its own vessel")
        self._hj = hj
        self._jplans = plans
        record = plan.record
        base_max = int(plans.STATIC_CAPACITY[hj.TASK])
        storage = base_max + int(plans.D7_RESERVED_SLOTS[hj.TASK])
        if int(cfg.max_obstacles) != storage or self.obstacle_centers.shape[1] != storage:
            raise ValueError(f"journal harbour storage must be STATIC_CAPACITY + D7 slot = {storage} "
                             f"(D7 M1/B04), got max_obstacles={cfg.max_obstacles}")
        self._jbase_max, self._jstorage = base_max, storage
        # ---- D7 admission pool: loaded BY PATH and verified (fail closed), B02/B05; D16 records
        pool = getattr(cfg, "journal_d7_pool", None)
        pool_source = "cfg.journal_d7_pool (object)"
        if pool is None:
            path, pool_source = hpool.resolve_pool_source(cfg)
            if path is None:
                raise ValueError("journal harbour envs need the admitted D7 pool: set cfg.journal_d7_pool_path, "
                                 f"${hpool.POOL_PATH_ENV} (file or directory) or ${hpool.POOL_DIR_ENV}, or deploy "
                                 "the pinned pool into tasks/harbor_mission/d7_pools (produced offline by "
                                 "tools_harbour_mission/admit_d7_pool.py)")
            raw, meta = hpool.load_pool(path, dynamic, vessel=cfg.vehicle)
            if int(meta.get("seed", -1)) != int(cfg.seed) or int(meta.get("tier", -1)) != int(plan.geometry_tier):
                raise ValueError(f"D7 pool {path} was admitted for seed {meta.get('seed')} tier {meta.get('tier')}, "
                                 f"the env runs seed {cfg.seed} tier {plan.geometry_tier}")
            pool = hpool.HarbourPoolView(raw, meta)
        elif isinstance(pool, dynamic.AdmissionPool):
            pool = hpool.HarbourPoolView(pool)
        self.journal_setup(d7_pool=pool)
        # D16: rejected keys keep their E5 layout and run at D0 (NotAdmitted rows, exposure "not_admitted")
        self._d7_bridge = hpool.HarbourD7Bridge(self._d7_bridge, dynamic)
        self._jwidth = storage
        if self._appearing_bridge is not None:
            # G20b: the appearing cylinder owns one more storage column after the D7 slot
            # (CapacityPlan.appearing_slots[0] = STATIC_CAPACITY + D7 reserve); it uses the D7 masks
            # (physical -> reward clearance, visible -> rays) and its swept-contact latch joins the
            # contract collision input like the D7 contact; the static-disk contract storage is untouched
            if self._appearing_slot != storage:
                raise ValueError(f"appearing slot {self._appearing_slot} is not the column after the D7 slot {storage}")
            self._jwidth = storage + 1
            self.obstacle_centers = torch.zeros((n, self._jwidth, 2), device=self.device)
            self.obstacle_radii = torch.zeros((n, self._jwidth), device=self.device)
            self.obstacle_active = torch.zeros((n, self._jwidth), dtype=torch.bool, device=self.device)
            self.appearing_hazard_header["collision_model"] = (
                "harbour mission: the cylinder has its own storage column after the D7 slot with the D7 masks "
                "(physical: reward clearance; visible: rays); its swept-contact latch (D7 swept guard) is OR'ed "
                "into the contract collision input like the D7 contact")
        header_fields = getattr(pool, "header_fields", None)
        if header_fields is not None:
            self.dynamic_hazard_header.update(header_fields())
        if not isinstance(self.dynamic_hazard_header.get("grid"), dict):
            # D24 (c): the header publishes the pool's offered grid; HarbourPoolView derives it (fail closed)
            raise ValueError("the harbour D7 pool must publish its offered grid {num_envs, episodes} (CONTEXT_R4 "
                             "D24); pass a HarbourPoolView, an E4 AdmissionPool or a pool file")
        self.dynamic_hazard_header["pool_source"] = pool_source
        self.dynamic_hazard_header["requested_dose"] = int(self._d7_suite.dose)
        # the admitted candidates were built with the pool's suite parameters (dose aside)
        admitted_suite = dict(getattr(pool, "meta", {}).get("suite") or {})
        if admitted_suite:
            from dataclasses import asdict

            running = asdict(self._d7_suite)
            ignored = ("dose", "enabled")
            if {k: v for k, v in admitted_suite.items() if k not in ignored} !=                     {k: v for k, v in running.items() if k not in ignored}:
                raise ValueError("the D7 suite parameters of this run differ from those the pool was admitted "
                                 "with (only the dose may change; re-admit the pool)")
        # ---- COM reference (decision 12 / E5 C5): origin = COM, verified on the engine
        com_b = self.robot.data.com_pos_b.detach().to("cpu", torch.float64).reshape(n, -1, 3)[:, 0]
        if not bool(torch.isfinite(com_b).all()) or float(com_b.abs().max()) > 1.0e-6:
            raise ValueError("journal harbour needs a v2 asset whose root link is the COM (com_pos_b = 0)")
        # ---- contract, geometry and D7 storage
        self._jhull = torch.as_tensor(record.footprint.vertices_m, dtype=torch.float64)
        self._jhull_dev = self._jhull.to(self.device, torch.float32)
        self._jlimits = contracts.ValidityLimits(**plan.validity_limits)
        self._oracle = hj.HarborMissionTensorOracle(n, record.footprint.vertices_m, self._jlimits)
        self._jnominal = float(plan.nominal_com_height_m)
        self._jdatum = float(self.journal_vehicle.hydro_profile.hydro.equilibrium_com_z_m)
        self._jspawn_phase = int(cfg.spawn_phase)
        self._jscope = hj.CONTRACT_SCOPES[self._jspawn_phase]
        self._jcontrol_dt = float(plans.CONTROL_DT_S)
        self._d7_physical = torch.zeros((n, self._jwidth), dtype=torch.bool, device=self.device)
        self._d7_visible = torch.zeros((n, self._jwidth), dtype=torch.bool, device=self.device)
        # G23c F8: the D7 slots (after the base maximum) answer the rays through the point-return proxy only
        from .._shared import d7_point_return as _d7pr
        self._jd7pr = _d7pr
        # G23d (REV-G23c H11): the D7 slots only (base_max .. base_max + D7_RESERVED_SLOTS), not "to the end of the
        # storage": an appearing slot added later (G20b) must keep the ray-circle model
        # (merge x G20b/G20c, R2: the mask spans _jwidth = storage + the appearing column)
        d7_end = int(self._jbase_max) + int(plans.D7_RESERVED_SLOTS[hj.TASK])
        self._jd7_point_slots = _d7pr.slot_mask(self._jwidth, range(int(self._jbase_max), d7_end), self.device)
        self._jlayouts = [None] * n
        self._jgate_mid = torch.zeros((n, 4, 2), dtype=torch.float64)
        self._jgate_nrm = torch.zeros((n, 4, 2), dtype=torch.float64)
        self._jgate_half = torch.zeros((n, 4), dtype=torch.float64)
        self._jgate_approach = torch.zeros((n, 3), dtype=torch.float64)
        self._jgate_vmin = torch.zeros((n, 3), dtype=torch.float64)
        self._jbounds = torch.zeros((n, 4), dtype=torch.float64)
        self._jstatic_centers = torch.zeros((n, base_max, 2), dtype=torch.float64)
        self._jstatic_radii = torch.zeros((n, base_max), dtype=torch.float64)
        self._jstatic_mask = torch.zeros((n, base_max), dtype=torch.bool)
        self._jprev_pose = torch.zeros((n, 3), dtype=torch.float64)
        self._jthrust_valid = torch.zeros(n, dtype=torch.bool)
        self._jfamily_hashes = [None] * n
        self._jd7_stage = [None] * n
        self._jd7_identity = [None] * n
        self.episode_contract_result = [None] * n
        self.episode_termination_reason = torch.zeros(n, dtype=torch.long, device=self.device)
        # ---- bind the journal methods (legacy sources untouched, D15)
        self._pre_physics_step = self._journal_pre_physics_step
        self._apply_action = self._journal_apply_action
        self._get_dones = self._journal_get_dones
        self._reset_idx = self._journal_reset_idx
        self._get_observations = self._journal_get_observations
        self._com_xy = self._journal_com_xy
        self._clearance = self._journal_reward_clearance
        self._ray_ranges_m = self._journal_true_rays

    # ---- truth helpers -------------------------------------------------------
    def _journal_com_xy(self) -> torch.Tensor:
        """World COM xy = root-link xy (v2 origin = COM; never root_com_pos_w after a reset)."""
        return self.robot.data.root_link_pose_w[:, :2]

    def _journal_yaw(self) -> torch.Tensor:
        forward = self._forward_2d()
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _journal_reward_clearance(self) -> torch.Tensor:
        """Hull-footprint clearance to static AND DETECTABLE D7 disks, for rewards (G23c F12: a physical but
        undetected D7 disk is no privileged signal; D7 collisions still reach the contract via the bridge latch)."""
        pose = torch.cat((self._journal_com_xy(), self._journal_yaw().unsqueeze(-1)), -1)
        mask = self.obstacle_active | self._d7_visible
        return self._hj.footprint_pose_clearance(self._jhull_dev, pose, self.obstacle_centers,
                                                 self.obstacle_radii, mask).to(pose.dtype)

    def _journal_true_rays(self) -> torch.Tensor:
        """Physical 36-ray ranges (m) from the COM: static slots + VISIBLE D7 slots (B07 ray mask, M3)."""
        forward = self._forward_2d()
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        cosine = torch.cos(self._ray_angles).view(1, -1, 1)
        sine = torch.sin(self._ray_angles).view(1, -1, 1)
        directions = cosine * forward.unsqueeze(1) + sine * left.unsqueeze(1)
        mask = self.obstacle_active | self._d7_visible
        rays = ray_circle_ranges(
            self._journal_com_xy(), directions, self.obstacle_centers, self.obstacle_radii,
            max_range_m=self.cfg.ray_max_range_m, active_mask=mask & ~self._jd7_point_slots)
        # G23c F8: the visible D7 intruder through the point-return proxy
        return self._jd7pr.merge(rays, self._journal_com_xy(), torch.atan2(forward[:, 1], forward[:, 0]),
                                 self.obstacle_centers, self.obstacle_radii, mask & self._jd7_point_slots,
                                 int(self.cfg.ray_count), float(self.cfg.ray_max_range_m))

    def _journal_state(self, robot_data, env_origins, water_level_local, extra=()):
        """Contract state (CPU float64, env-local COM) and the planar pose (x, y, yaw).

        G11: pose, velocity, origins, water level (and the ``extra`` tensors, returned as host copies
        when given) reach the host in ONE device-to-host copy (bytes verbatim); the float64 widening
        runs on the host copies, where each blocking .to("cpu", torch.float64) ran it."""
        n = self.num_envs
        hosts = host_copies([robot_data.root_link_pose_w, robot_data.root_com_vel_w, env_origins,
                             torch.as_tensor(water_level_local), *extra])
        pose = hosts[0].to("cpu", torch.float64)
        vel = hosts[1].to("cpu", torch.float64)
        origins = hosts[2].to("cpu", torch.float64)
        com = pose[:, :3] - origins[:, :3]
        quat = pose[:, 3:7]
        water = hosts[3].to("cpu", torch.float64).reshape(-1).expand(n).clone()
        state = {"com": com, "quat": quat, "lin": vel[:, :3], "ang": vel[:, 3:6], "water": water,
                 "nominal": torch.full((n,), self._jnominal, dtype=torch.float64)}
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        if extra:
            return state, torch.stack((com[:, 0], com[:, 1], yaw), -1), hosts[4:]
        return state, torch.stack((com[:, 0], com[:, 1], yaw), -1)

    def _journal_static_clearance(self, start_pose, end_pose):
        """Swept hull clearance to the STATIC layout disks (D7 contact comes from the bridge)."""
        has = self._jstatic_mask.any(-1)
        clearance = self._hj.footprint_swept_clearance(self._jhull, start_pose, end_pose, self._jstatic_centers,
                                                       self._jstatic_radii, self._jstatic_mask)
        return torch.where(has, clearance, torch.zeros_like(clearance)), has

    # ---- mixin family hooks --------------------------------------------------
    def _journal_observe_physics(self, robot_data, env_origins, water_level_local) -> None:
        """Contract observe_physics on the PREVIOUS substep's state (D8 water level, swept static clearance)."""
        thrust, mask = None, None
        last = self.journal_last_step
        if last is not None:
            # G11: the delivered thrust travels with the state (one device-to-host copy)
            state, pose, (thrust_host,) = self._journal_state(robot_data, env_origins, water_level_local,
                                                              extra=(last.motors.delivered_thrust,))
        else:
            state, pose = self._journal_state(robot_data, env_origins, water_level_local)
        clearance, has = self._journal_static_clearance(self._jprev_pose, pose)
        if last is not None:
            thrust = thrust_host.to("cpu", torch.float64)
            mask = self._jthrust_valid.clone()
        self._oracle.observe_physics(state, clearance=clearance, has_clearance=has, dt_s=float(self.cfg.sim.dt),
                                     motor_thrust=thrust, thrust_mask=mask)
        self._jprev_pose = pose
        self._jthrust_valid[:] = True

    def _journal_family_scenario_hashes(self, env_id) -> dict:
        return dict(self._jfamily_hashes[int(env_id)])

    def _journal_publish_d7(self, samples) -> None:
        """B07: D7 samples into the reserved slot; physical / visible masks split; obstacle_active untouched."""
        if not samples:
            return
        base = self._jbase_max
        rows = [int(i) for i in samples]
        if sorted(rows) == list(range(int(self.num_envs))):
            # G11: every row -> the same elements cleared by basic slicing (fill kernels; the index
            # tensor and the scalar index_put each cost a host sync before)
            self._d7_physical[:, base:self._jstorage] = False     # G20b: D7 slots only (never the appearing one)
            self._d7_visible[:, base:self._jstorage] = False
        else:
            index = torch.as_tensor(rows, dtype=torch.long, device=self.device)
            self._d7_physical[index, base:self._jstorage] = False
            self._d7_visible[index, base:self._jstorage] = False
        env_rows, slots, values, flags = [], [], [], []
        for i in rows:
            for sample in samples[i]:
                if not base <= int(sample.slot) < self._jstorage:
                    raise ValueError(f"D7 sample slot {sample.slot} outside the reserved slots "
                                     f"[{base}, {self._jstorage})")
                env_rows.append(i)
                slots.append(int(sample.slot))
                values.append((float(sample.x), float(sample.y), float(sample.radius_m)))
                flags.append((bool(sample.physical), bool(sample.visible)))
        if not env_rows:
            return
        # G11: the four host tensors of the per-field as_tensor(..., device=) calls, built on the host
        # exactly as those calls build them, moved in ONE host-to-device copy (bytes verbatim)
        er, sl, val, flag = device_copies(
            [torch.as_tensor(env_rows, dtype=torch.long), torch.as_tensor(slots, dtype=torch.long),
             torch.as_tensor(values, dtype=self.obstacle_centers.dtype), torch.as_tensor(flags, dtype=torch.bool)],
            self.device)
        self.obstacle_centers[er, sl] = val[:, :2] + self.scene.env_origins[er, :2].to(val.dtype)
        self.obstacle_radii[er, sl] = val[:, 2]
        self._d7_physical[er, sl] = flag[:, 0]
        self._d7_visible[er, sl] = flag[:, 1]

    def _journal_publish_appearing(self, samples) -> None:
        """G20b mixin hook: the appearing cylinder in its own column (physical = visible masks)."""
        if not samples:
            return
        rows, values, on = [], [], []
        for env_index, items in samples.items():
            for item in items:
                if int(item.slot) != self._appearing_slot or bool(item.physical) != bool(item.visible):
                    raise ValueError(f"appearing sample slot {item.slot} / masks do not match the appearing slot")
                rows.append(int(env_index))
                values.append((float(item.x), float(item.y), float(item.radius_m)))
                on.append(bool(item.physical))
        if not rows:
            return
        row_t = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        slot = self._appearing_slot
        val = torch.as_tensor(values, dtype=self.obstacle_centers.dtype).to(self.device)
        flag = torch.as_tensor(on, dtype=torch.bool).to(self.device)
        self.obstacle_centers[row_t, slot] = val[:, :2] + self.scene.env_origins[row_t, :2].to(val.dtype)
        self.obstacle_radii[row_t, slot] = val[:, 2]
        self._d7_physical[row_t, slot] = flag
        self._d7_visible[row_t, slot] = flag

    def _journal_appearing_geometry(self, env_id) -> str:
        """G20b mixin hook: geometry id of the env's current E5 layout (the appearing pool records it per key)."""
        layout = self._jlayouts[int(env_id)]
        if layout is None:
            raise RuntimeError(f"env {int(env_id)} has no journal layout")
        return str(layout.d7_hook["geometry_id"])

    # ---- DirectRLEnv hooks (journal) ------------------------------------------
    def _journal_pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.journal_pre_physics(actions)
        self._phase_at_step_start.copy_(self.phase)
        self._exit_progress_at_step_start.copy_(self._exit_gate_progress)
        self._potential_at_step_start.copy_(self._phase_potential(self._journal_com_xy(), self.phase))
        self._m1_at_step_start.copy_(self._m1)
        self._m2_at_step_start.copy_(self._m2)
        self._m3_at_step_start.copy_(self._m3)

    def _journal_apply_action(self) -> None:
        from .._shared.body_wrench import apply_body_wrench   # G6: no per-call deprecation warning
        forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins,
                                               float(self.physics_cfg.water_surface_z))
        apply_body_wrench(self.robot, forces.to(device=self.device, dtype=torch.float32),
                          torques.to(device=self.device, dtype=torch.float32))

    def _journal_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        view = self.journal_pre_dones(self.robot.data, self.scene.env_origins,
                                      float(self.physics_cfg.water_surface_z))
        # G11: the episode counter and the D7 contact latch travel with the state (one copy)
        state, pose, (steps_host, contact_host) = self._journal_state(
            self.robot.data, self.scene.env_origins, view.water_level_local,
            extra=(self.episode_length_buf, torch.as_tensor(view.d7_contact)))
        clearance, has = self._journal_static_clearance(self._jprev_pose, pose)
        self._jprev_pose = pose
        time_s = steps_host.to("cpu", torch.float64) * self._jcontrol_dt
        x, y, b = state["com"][:, 0], state["com"][:, 1], self._jbounds
        out_of_bounds = ~((b[:, 0] <= x) & (x <= b[:, 1]) & (b[:, 2] <= y) & (y <= b[:, 3]))
        d7_contact = contact_host.to("cpu", torch.bool)
        if getattr(self, "_appearing_bridge", None) is not None:
            d7_contact = d7_contact | self.journal_appearing_contact().to("cpu")      # G20b swept latch
        terminated_o, truncated_o, codes = self._oracle.step(
            time_s, state, collision=d7_contact, out_of_bounds=out_of_bounds, clearance=clearance,
            has_clearance=has)
        codes_device = self._journal_mirror_task_state(codes)
        finished = (terminated_o | truncated_o).nonzero().flatten().tolist()
        for i in finished:
            self._journal_export_completed(i)
        if finished:
            self._journal_log_completed(finished)
        terminated, truncated = self.journal_dones(codes_device)
        self._episode_finished.copy_(terminated | truncated)
        return terminated, truncated

    def _journal_mirror_task_state(self, codes=None):
        """Legacy automaton fields (read by the unchanged _get_rewards) mirrored from the tensor oracle.

        G11: the fourteen host tensors (each converted on the host to the dtype its blocking
        .to(dev, dtype) copy carried, as that copy converts on the host) and ``codes`` (returned as
        codes.to(dev)) travel in ONE host-to-device copy, bytes verbatim; each copy_ then receives the
        same device tensor as before."""
        o, dev = self._oracle, self.device
        spawn2 = o.spawn_phase == 2
        departed = spawn2 | ~torch.isnan(o._departure_time())
        transited = spawn2 | ~torch.isnan(o.transit_time)
        progress = (~torch.isnan(o.gate_times)).sum(-1)
        host = [o.stage(), torch.where(spawn2, torch.full_like(progress, 2), progress), departed, transited,
                o.success, o.success_now, o.success, o.tts.to(self._first_success_time_s.dtype),
                o.dwell.to(self.hold_timer.dtype), torch.round(o.dwell / self._jcontrol_dt).to(torch.long),
                o.full_len.to(self.path_length.dtype), o.min_clearance.to(self._min_clearance.dtype), ~o.clean]
        if codes is not None:
            host.append(codes)
        moved = device_copies(host, dev)
        self.phase.copy_(moved[0])
        self._exit_gate_progress.copy_(moved[1])
        self._m1.copy_(moved[2])
        self._m2.copy_(moved[3])
        self._m3.copy_(moved[4])
        self._stage_goal_entry_this_step.copy_(moved[5])
        self._success.copy_(moved[6])
        self._first_success_time_s.copy_(moved[7])
        self.hold_timer.copy_(moved[8])
        self._hold_steps.copy_(moved[9])
        self.path_length.copy_(moved[10])
        self._min_clearance.copy_(moved[11])
        self._contact_before_dock.copy_(moved[12])
        self._max_phase.copy_(torch.maximum(self._max_phase, self.phase))
        self._phase_time_s += (functional.one_hot(self._phase_at_step_start, num_classes=4).float()
                               * self.control_step_s)
        self._previous_xy.copy_(self._journal_com_xy())
        return moved[13] if codes is not None else None

    def journal_contract_record(self, env_id) -> dict | None:
        """Latched finished-episode contract result (E5 EpisodeResult fields + harbour identity), or None."""
        record = self.episode_contract_result[int(env_id)]
        return None if record is None else dict(record)

    def _journal_export_completed(self, i: int) -> None:
        """Completed-episode export BEFORE the reset (contract result -> legacy buffers + record)."""
        result = self._oracle.last_completed[i]
        layout = self._jlayouts[i]
        self.episode_success[i] = bool(result.episode_success)
        self.time_to_success[i] = float(result.time_to_success)
        self.episode_path_length[i] = float(result.full_episode_path_length_m)
        self.episode_min_clearance[i] = float(result.metrics["min_clearance_m"])
        self.episode_max_phase[i] = self._max_phase[i]
        self.episode_phase_time_s[i] = self._phase_time_s[i]
        self.stage_reached[i] = self.phase[i]
        self.m1[i] = self._m1[i]
        self.m2[i] = self._m2[i]
        self.m3[i] = self._m3[i]
        self.episode_termination_reason[i] = int(result.termination_reason)
        self.episode_scenario[i] = self._scenario_params[i]
        stage = self._jd7_stage[i]
        identity = self._jd7_identity[i] or {}
        not_admitted = identity.get("exposure") == "not_admitted"
        skipped = self._jspawn_phase == 2 and stage in ("departure", "transit")
        if not_admitted and identity.get("beyond_offered_grid"):
            exposure = ("not_admitted (D24: beyond_offered_grid; not a pool key, E5 layout of the key, dynamic "
                        "suite at D0, never counted)")
        elif not_admitted:
            exposure = "not_admitted (D16: E4 rejected the key; E5 layout kept, dynamic suite at D0)"
        elif skipped:
            exposure = "skipped_by_spawn_phase (recorded non-exposure, never rescheduled)"
        else:
            exposure = "in_scope"
        fields = dict(result.evaluator_fields())
        fields.update(task=self._hj.TASK, vessel=self.cfg.vehicle, geometry_tier=int(layout.tier),
                      geometry_hash=layout.geometry_hash,
                      nominal_route_length_m=float(layout.nominal_route_length_m),
                      scenario_deadline_steps=int(layout.deadline_steps), layout_attempts=int(layout.attempts),
                      layout_rejections=dict(layout.rejections), spawn_phase=self._jspawn_phase,
                      d7_declared_stage=stage, d7_stage_exposure=exposure,
                      d7_exposure=identity.get("exposure"), d7_requested_dose=identity.get("requested_dose"),
                      d7_not_admitted_reasons=[list(r) for r in identity.get("reasons", ())])
        self.episode_contract_result[i] = fields

    def _journal_log_completed(self, rows) -> None:
        ids = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        self.extras.setdefault("log", {})
        self.extras["log"]["Episode/success"] = self.episode_success[ids].float().mean()
        self.extras["log"]["Episode/stage_reached"] = self.stage_reached[ids].float().mean()
        self.extras["log"]["Episode/path_length_m"] = self.episode_path_length[ids].mean()

    def _journal_reset_idx(self, env_ids: Sequence[int] | None) -> None:
        hj = self._hj
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        self._scenario.reset_idx(env_ids)
        ids = env_ids.detach().to("cpu").tolist()
        episodes = self._scenario.episode_indices()[ids].clone()
        k, base = len(ids), self._jbase_max
        cfg = self.cfg
        centers = torch.zeros((k, self._jwidth, 2), dtype=torch.float64)       # G20b: + the appearing column
        radii = torch.zeros((k, self._jwidth), dtype=torch.float64)
        active = torch.zeros((k, self._jwidth), dtype=torch.bool)
        spawn = torch.zeros((k, 3), dtype=torch.float64)
        berth = torch.zeros((k, 6), dtype=torch.float64)
        tol = torch.zeros((k, 4), dtype=torch.float64)
        deadline = torch.zeros(k, dtype=torch.float64)
        stage_x = torch.zeros((k, 4), dtype=torch.float64)
        for row, i in enumerate(ids):
            layout = hj.journal_layout(cfg.vehicle, int(cfg.journal_geometry_tier), int(cfg.seed), i,
                                       int(episodes[row]), protocol_version=self._scenario.protocol_version)
            count = int(layout.obstacle_count)
            if count > base:
                raise ValueError(f"harbour layout has {count} static disks > STATIC_CAPACITY {base}")
            if layout.deadline_s > float(cfg.episode_length_s) + 1.0e-9:
                raise ValueError("scenario deadline exceeds the plan's episode cap")
            self._jlayouts[i] = layout
            gates = hj.gate_table(layout)
            rec = layout.extras["berth"]
            centers[row, :count] = torch.from_numpy(np.array(layout.centers, dtype=np.float64))
            radii[row, :count] = torch.from_numpy(np.array(layout.radii, dtype=np.float64))
            active[row, :count] = True
            spawn[row] = torch.as_tensor(hj.spawn_pose(layout, self._jspawn_phase), dtype=torch.float64)
            berth[row] = torch.tensor((rec["center_xy"][0], rec["center_xy"][1], rec["heading_rad"],
                                       rec["opening_x_m"], rec["back_x_m"], rec["half_width_m"]),
                                      dtype=torch.float64)
            tol[row] = torch.tensor((rec["position_tolerance_m"], rec["heading_tolerance_rad"],
                                     rec["speed_tolerance_m_s"], rec["hold_s"]), dtype=torch.float64)
            deadline[row] = float(layout.deadline_s)
            self._jgate_mid[i] = torch.as_tensor(gates["midpoints"])
            self._jgate_nrm[i] = torch.as_tensor(gates["normals"])
            self._jgate_half[i] = torch.as_tensor(gates["half_widths"])
            self._jgate_approach[i] = torch.as_tensor(gates["approach_m"])
            self._jgate_vmin[i] = torch.as_tensor(gates["min_normal_speed_mps"])
            self._jbounds[i] = torch.as_tensor(layout.bounds_m, dtype=torch.float64)
            self._jstatic_centers[i] = centers[row, :base]
            self._jstatic_radii[i] = radii[row, :base]
            self._jstatic_mask[i] = active[row, :base]
            stage_x[row] = torch.tensor((float(spawn[row, 0]), gates["midpoints"][1][0],
                                         gates["midpoints"][2][0], rec["center_xy"][0]), dtype=torch.float64)
            self._scenario_params[i] = {"layout": {"geometry_hash": layout.geometry_hash, "obstacle_count": count,
                                                   "deadline_s": float(layout.deadline_s),
                                                   "nominal_route_length_m": float(layout.nominal_route_length_m)}}
            self._jfamily_hashes[i] = hj.family_scenario_hashes(layout, tuple(spawn[row].tolist()),
                                                                self._jspawn_phase, self._jdatum)
        dev, dtype = self.device, self.obstacle_centers.dtype
        origins = self.scene.env_origins[env_ids]
        origins_xy = origins[:, :2]
        self.obstacle_centers[env_ids] = origins_xy.unsqueeze(1) + centers.to(dev, dtype)
        self.obstacle_radii[env_ids] = radii.to(dev, dtype)
        self.obstacle_active[env_ids] = active.to(dev)
        self.obstacle_count[env_ids] = active.sum(-1).to(dev)
        self._d7_physical[env_ids] = False
        self._d7_visible[env_ids] = False
        self.gate_midpoints[env_ids] = origins_xy.unsqueeze(1) + self._jgate_mid[ids, :3].to(dev, dtype)
        self.gate_normals[env_ids] = self._jgate_nrm[ids, :3].to(dev, dtype)
        self.berth_point[env_ids] = origins_xy + berth[:, :2].to(dev, dtype)
        headings = berth[:, 2].tolist()
        self.dock_heading[env_ids] = torch.tensor([[math.cos(h), math.sin(h)] for h in headings],
                                                  dtype=dtype, device=dev)
        self.field_geodesic_length[env_ids] = torch.tensor(
            [float(self._jlayouts[i].nominal_route_length_m) for i in ids], dtype=dtype, device=dev)
        self.stage_start_x[env_ids] = origins_xy[:, 0:1] + stage_x[:, 0:3].to(dev, dtype)
        self.stage_end_x[env_ids] = origins_xy[:, 0:1] + stage_x[:, 1:4].to(dev, dtype)
        # ---- spawn: the COM at water level + the profile datum (D3 section 4); ROOT WRITE
        root_state = self.robot.data.default_root_state[env_ids].clone()
        yaw = (spawn[:, 2] + self._body_yaw_from_bow_offset).to(dev, root_state.dtype)
        root_state[:, 0] = origins[:, 0] + spawn[:, 0].to(dev, root_state.dtype)
        root_state[:, 1] = origins[:, 1] + spawn[:, 1].to(dev, root_state.dtype)
        root_state[:, 2] = origins[:, 2] + float(self.physics_cfg.water_surface_z) + self._jdatum
        root_state[:, 3:7] = torch.stack((torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw),
                                          torch.sin(0.5 * yaw)), -1)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)
        # ---- legacy automaton fields (read by the unchanged _get_rewards)
        start_phase = 2 if self._jspawn_phase == 2 else 0
        self.phase[env_ids] = start_phase
        self._exit_gate_progress[env_ids] = 2 if start_phase >= 1 else 0
        self._m1[env_ids] = start_phase >= 1
        self._m2[env_ids] = start_phase >= 2
        self._m3[env_ids] = False
        self._m1_at_step_start[env_ids] = start_phase >= 1
        self._m2_at_step_start[env_ids] = start_phase >= 2
        self._m3_at_step_start[env_ids] = False
        self._stage_goal_entry_this_step[env_ids] = False
        self._m1_step[env_ids] = -1
        self._m2_step[env_ids] = -1
        self._m3_step[env_ids] = -1
        self._hold_steps[env_ids] = 0
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._max_phase[env_ids] = start_phase
        self._phase_time_s[env_ids] = 0.0
        self._previous_xy[env_ids] = origins_xy + spawn[:, :2].to(dev, dtype)
        self._phase_at_step_start[env_ids] = start_phase
        self._exit_progress_at_step_start[env_ids] = 2 if start_phase >= 1 else 0
        self._potential_at_step_start[env_ids] = 0.0
        self._contact_prev[env_ids] = False
        self._contact_before_dock[env_ids] = False
        self._success[env_ids] = False
        self._first_success_time_s[env_ids] = torch.nan
        self._episode_finished[env_ids] = False
        self._min_clearance[env_ids] = torch.inf
        self.actions[env_ids] = 0.0
        # ---- contract twin + sweep origin
        self._oracle.reset_rows(ids, com_xy=spawn[:, :2], deadline_s=deadline, berth=berth, tolerances=tol,
                                spawn_phase=self._jspawn_phase, gate_midpoints=self._jgate_mid[ids, :3],
                                gate_normals=self._jgate_nrm[ids, :3], gate_half_widths=self._jgate_half[ids, :3],
                                gate_approach_m=self._jgate_approach[ids], gate_min_speed_mps=self._jgate_vmin[ids])
        self._jprev_pose[ids] = spawn
        self._jthrust_valid[ids] = False
        # ---- journal glue (after the root write): runtime, conditions, D7 rows + tick 0
        self.journal_reset(env_ids, torch.zeros(k, dtype=torch.float64), episodes,
                           reference_route_length_m=torch.tensor(                                  # G28
                               [float(self._jlayouts[i].nominal_route_length_m) for i in ids], dtype=torch.float64))
        for i in ids:
            identity = self._d7_bridge.row_identity(i)
            geometry_id = identity["geometry_id"]
            if identity.get("beyond_offered_grid"):
                # D24: a key beyond the offered grid is not a pool key (no recorded geometry, stage None);
                # the layout above is the key's E5 layout from its keyed stream, so nothing to compare
                self._jd7_stage[i] = identity["stage"]
                self._jd7_identity[i] = identity
                continue
            if geometry_id is None:
                raise ValueError(f"D7 record of env {i} ({identity['exposure']}) carries no geometry id "
                                 "(load the pool from a file / pool_meta)")
            if geometry_id != self._jlayouts[i].geometry_hash:
                raise ValueError(f"D7 {identity['exposure']} record of env {i} was admitted on geometry "
                                 f"{geometry_id[:12]}, the env built {self._jlayouts[i].geometry_hash[:12]} "
                                 "(pool/layout mismatch)")
            self._jd7_stage[i] = identity["stage"]
            self._jd7_identity[i] = identity
        self._update_render_only_markers(env_ids, self.gate_midpoints[env_ids] - origins_xy.unsqueeze(1),
                                         centers.to(dev, dtype), radii.to(dev, dtype), active.to(dev),
                                         berth[:, :2].to(dev, dtype))

    def _journal_get_observations(self) -> dict:
        """Journal 49-D observation (journal_kin48_ray36): every position-derived channel from est.position."""
        data = self.robot.data
        origins = self.scene.env_origins
        com_local = data.root_link_pose_w[:, :2] - origins[:, :2]
        forward = self._forward_2d()
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        velocity = data.root_com_vel_w
        planar = velocity[:, :2]
        yaw_rate = math_utils.quat_apply_inverse(data.root_link_pose_w[:, 3:7], velocity[:, 3:6])[:, 2]
        body = torch.stack((torch.sum(planar * forward, dim=-1), torch.sum(planar * left, dim=-1), yaw_rate), -1)
        est = self.journal_estimates(com_local, body, self._journal_true_rays())
        dtype = forward.dtype
        position = est.position.to(dtype)
        stage = self.phase.clamp(max=3)
        # G11: the oracle's gate index, the gate midpoints and the dwell (host tensors, converted on the
        # host to the dtype their blocking .to(device, dtype) copies carried) in ONE host-to-device copy
        gate_index, gate_mid_all, dwell_dev = device_copies(
            [self._oracle.gate_passed()[:, :3].sum(-1).clamp(max=2), self._jgate_mid.to(dtype),
             self._oracle.dwell.to(dtype)], self.device)
        rows = torch.arange(self.num_envs, device=self.device)
        gate_mid = gate_mid_all[rows, gate_index]
        berth_local = self.berth_point - origins[:, :2]
        target = torch.where((stage >= 2).unsqueeze(-1), berth_local, gate_mid)
        to_target = target - position
        distance = torch.norm(to_target, dim=-1, keepdim=True)
        direction = to_target / distance.clamp_min(1.0e-6)
        dot = torch.sum(forward * direction, dim=-1, keepdim=True)
        cross = forward[:, 0:1] * direction[:, 1:2] - forward[:, 1:2] * direction[:, 0:1]
        span = torch.gather(self.stage_end_x - self.stage_start_x, 1, stage.clamp(max=2).unsqueeze(-1))
        distance_norm = torch.clamp(distance / span.clamp_min(1.0e-6), 0.0, 1.0)
        base = torch.hstack((dot, cross, distance_norm))
        base = torch.where((stage < 3).unsqueeze(-1), base, torch.zeros_like(base))
        from .._shared.obs_superset import SPEED_SCALE_MPS as _SPD

        vel = est.velocity.to(dtype)
        kinematic = torch.stack((torch.clamp(vel[:, 0] / _SPD, -1.0, 1.0), torch.clamp(vel[:, 1] / _SPD, -1.0, 1.0),
                                 torch.clamp(vel[:, 2] / self.cfg.yaw_rate_obs_scale_rad_s, -1.0, 1.0)), -1)
        ranges_norm = est.rays.to(dtype) / self.cfg.ray_max_range_m
        phase_one_hot = functional.one_hot(stage, num_classes=4).float()
        dock_dot = torch.sum(forward * self.dock_heading, dim=-1, keepdim=True)
        dock_cross = forward[:, 0:1] * self.dock_heading[:, 1:2] - forward[:, 1:2] * self.dock_heading[:, 0:1]
        in_dock = (stage == 2).unsqueeze(-1)
        dock_alignment = torch.where(in_dock, torch.hstack((dock_dot, dock_cross)),
                                     torch.zeros((self.num_envs, 2), dtype=dtype, device=self.device))
        dwell = (dwell_dev.unsqueeze(-1) / float(self.cfg.required_hold_time_s)).clamp(0.0, 1.0)
        dwell = torch.where(in_dock, dwell, torch.zeros_like(dwell))
        observation = torch.hstack((base, kinematic, ranges_norm, phase_one_hot, dock_alignment, dwell))
        return {"policy": observation}

    def journal_phase_targets(self) -> dict:
        """Phase targets for a classical staged controller (PHASE_TARGETS_F6.md; env-local COM frame).

        Task geometry (gates, berth, static chart without the D7 slot) and the
        contract's stage latches only; no D7 object, sea, current or wind
        state and no truth pose. Pair with ``self.journal_estimate`` (D5).
        """
        time_s = self.episode_length_buf.detach().to("cpu", torch.float64) * self._jcontrol_dt
        return self._hj.phase_targets(self._oracle, self._jgate_mid, self._jgate_nrm, self._jgate_half,
                                      self._jgate_approach, self._jgate_vmin, self._jstatic_centers,
                                      self._jstatic_radii, self._jstatic_mask, self._jhull, self._jscope,
                                      time_s, self.device)


def _journal_vehicle_spec(cfg):
    """Minimal spawn descriptor of a v2 vessel for _setup_scene (rigid asset, bow +x)."""
    from types import SimpleNamespace

    from .._shared.journal_vehicle import v2_spec_or_raise

    spec = v2_spec_or_raise(cfg.vehicle)
    if spec.bow_body_axis != "+x":
        raise ValueError("journal v2 assets have their bow at +x")
    return SimpleNamespace(name=spec.name, asset_kind="rigid", bow_body_axis="+x")
