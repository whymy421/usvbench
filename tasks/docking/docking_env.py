# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .._shared.termination_reason import publish_termination_reason

import math
import os
from collections.abc import Sequence

from .._shared.vehicles import physics_version
from .._shared.physics_v2 import add_implicit_hull_damping, WaveClock

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.vehicles import get_vehicle
from .._shared.journal_env_mixin import JournalEnvMixin, device_copies
from .curriculum import DockingCurriculum
from .docking_env_cfg import DockingEnvCfg
from .restoring import restoring_torque_body


def _space_dim(space: object) -> int:
    """Return the leading dimension of a Gym space or a legacy integer size."""
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


class DockingEnv(JournalEnvMixin, DirectRLEnv):
    """Calm-water boat docking with reward-independent success.

    Journal cfgs (journal_contract_version == 2, E5 templates) run the journal
    branch (``_journal_init`` and the ``_journal_*`` methods at the end of this
    class); every legacy method is unchanged (CONTEXT_R4 D15).
    """

    cfg: DockingEnvCfg
    journal_family = "docking"
    # G20b appearing obstacle: the journal task of this family wired for it (own one-slot storage)
    journal_appearing_tasks = ("docking",)

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kwargs):
        if getattr(cfg, "journal_contract_version", None) == 2:
            self._journal_init(cfg, render_mode, **kwargs)
            return
        self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        physics_version(self.physics_cfg)  # Reject unsupported versions before simulation.
        # Evaluation reproducibility. hazard_nav has always drawn its layouts
        # from a generator seeded by cfg.seed, which is why two --eval-seed
        # values give two genuinely different exam papers there. Docking drew
        # from the GLOBAL torch RNG (seeded by the trainer, not by cfg), so
        # cert_dock_wall_e42 and _e123 came out identical episode for episode --
        # two "independent" seeds that were in fact one paper.
        self._spawn_gen = None
        _seed = getattr(cfg, "seed", None)
        if _seed is not None:
            self._spawn_gen = torch.Generator(device="cpu")
            self._spawn_gen.manual_seed(int(_seed))
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
        if getattr(self.cfg, "curriculum_frozen", False):
            # Evaluation contract: pin the spawn distance to the requested
            # level and never advance it, so a 128-episode score measures one
            # difficulty instead of drifting upward as successes accumulate.
            levels = self.cfg.eval_level_distances
            if self.physics_cfg.physics_version == 2:
                levels = self.cfg.eval_level_distances_v2
            index = min(max(int(self.cfg.eval_level), 0), len(levels) - 1)
            self.curriculum.spawn_distance = float(levels[index])
            override = float(getattr(self.cfg, "spawn_distance_override_m", 0.0))
            if override > 0.0:
                # Tier-regrade probe distance; 0.0 keeps the level table.
                self.curriculum.spawn_distance = override
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
        # Longest CONTINUOUS stretch of the finished episode that satisfied the
        # berthing predicate -- position, heading AND speed together, as
        # _task_state assembles it -- in seconds. ``held`` thresholds exactly
        # this quantity (:842), so recording it exposes the graded metric the
        # binary criterion hides: an approach that never once arrived and one
        # that arrived and could not hold still are both "failure", and only
        # this number separates them. It is defined for EVERY episode, success
        # or failure, so a dose response reads off it with no conditioning on
        # the outcome.
        #
        # Read it with the berth-wall caveat. On the wall ids success is the
        # threshold AND a clean prefix (:843-844), so max_hold_s past the
        # requirement is NECESSARY but not SUFFICIENT there: an episode that
        # scraped a wall on the approach and then berthed perfectly records the
        # full hold and success=False. scripts/check_hold_consistency.py
        # asserts the bare equivalence, so on a wall certificate it will list
        # those episodes -- that is the contact veto showing through, not a
        # latch bug. On the open-water ids berth_walls stays False
        # (docking_env_cfg.py:209), the veto is vacuous, and the threshold is
        # exactly success as in the station-keeping families.
        self.episode_max_hold_s = torch.zeros(self.num_envs, device=self.device)

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

        # Sea state (Wave variants only). Certified ids have no sea_state cfg
        # field, so getattr returns None and nothing here runs for them. The
        # field is a pure background force: it never touches the observation,
        # reward, or termination paths of this env.
        #
        # Docking is the one family without the scenario protocol -- its own
        # spawn draws ride the seed-derived per-env _spawn_gen above, which is
        # immune to the skrl Runner reseed (tasks/_shared/scenario_rng.py:34-38)
        # -- so the wave draws get their own keyed per-(env, episode) "wave"
        # stream via make_scenario_rng: two --eval-seed values sail two
        # genuinely different seas. cfg.seed None -> None -> the historical
        # global-RNG fallback inside SeaState.resample.
        self._sea = None
        self._sea_scenario = None
        if getattr(self.cfg, "sea_state", None) is not None and self.cfg.sea_state.enable:
            from .._shared.scenario_draws import make_scenario_rng
            from .._shared.sea_state import SeaState

            self._sea_scenario = make_scenario_rng(
                self.cfg, self.num_envs, self.device
            )
            self._sea = SeaState(
                self.cfg.sea_state, self.num_envs, self.device,
                physics_version=getattr(self.physics_cfg, "physics_version", 1),
            )
            if getattr(self.physics_cfg, "physics_version", 1) == 2:
                self._wave_clock = WaveClock(self.num_envs, self.device, self.cfg.sim.dt)

        # C3 x C5: solid berth walls. Cylinders reuse hazard_nav's certified
        # clearance/ray kernels so the contact predicate and the 36-ray sensor
        # keep identical semantics across tasks.
        self._wall_centers = torch.zeros((self.num_envs, 1, 2), device=self.device)
        self._wall_radii = torch.zeros((self.num_envs, 1), device=self.device)
        self._wall_active = torch.zeros(
            (self.num_envs, 1), dtype=torch.bool, device=self.device
        )
        self._contact_prev = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact_before_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._min_clearance = torch.full((self.num_envs,), torch.inf, device=self.device)
        self.episode_min_clearance = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        if self.cfg.berth_walls:
            self._build_berth_walls()
            self._ray_angles = torch.arange(
                self.cfg.ray_count, device=self.device, dtype=torch.float32
            ) * (2.0 * torch.pi / self.cfg.ray_count)

    def _rand(self, n: int) -> torch.Tensor:
        """Uniform draw from the eval-reproducible generator when seeded."""
        if self._spawn_gen is None:
            return torch.rand(n, device=self.device)
        return torch.rand(n, generator=self._spawn_gen).to(self.device)

    @property
    def current_spawn_distance(self) -> float:
        """Current curriculum spawn radius in metres."""
        return self.curriculum.spawn_distance

    def _build_berth_walls(self) -> None:
        """Author the U-shaped slip once: geometry is fixed per environment."""
        import numpy as np

        from .berth_geometry import berth_wall_cylinders, slip_width_for_beams, to_world

        width = slip_width_for_beams(self.cfg.slip_width_beams)
        local, radii = berth_wall_cylinders(
            width, slip_length_m=self.cfg.slip_length_m
        )
        dock_np = self.dock_point.detach().cpu().numpy()
        heading_np = self.dock_heading.detach().cpu().numpy()
        stacked = np.stack(
            [to_world(local, dock_np[i], heading_np[i]) for i in range(self.num_envs)]
        )
        self._wall_centers = torch.as_tensor(
            stacked, device=self.device, dtype=torch.float32
        )
        self._wall_radii = torch.as_tensor(
            np.tile(radii, (self.num_envs, 1)), device=self.device, dtype=torch.float32
        )
        self._wall_active = torch.ones(
            self._wall_radii.shape, dtype=torch.bool, device=self.device
        )

    def _wall_clearance(self) -> torch.Tensor:
        from ..hazard_nav.hazard_geometry import analytic_min_clearance

        if not self.cfg.berth_walls:
            return torch.full((self.num_envs,), torch.inf, device=self.device)
        return analytic_min_clearance(
            self.robot.data.root_com_pos_w[:, :2],
            self._wall_centers,
            self._wall_radii,
            half_beam_m=self.cfg.half_beam_m,
            active_mask=self._wall_active,
        )

    def _wall_ray_ranges_m(self) -> torch.Tensor:
        from ..hazard_nav.hazard_geometry import ray_circle_ranges

        forward = self._forward_2d()
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        cosine = torch.cos(self._ray_angles).view(1, -1, 1)
        sine = torch.sin(self._ray_angles).view(1, -1, 1)
        directions = cosine * forward.unsqueeze(1) + sine * left.unsqueeze(1)
        return ray_circle_ranges(
            self.robot.data.root_com_pos_w[:, :2],
            directions,
            self._wall_centers,
            self._wall_radii,
            max_range_m=self.cfg.ray_max_range_m,
            active_mask=self._wall_active,
        )

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
        # Reportable disturbance state. It remains zero for calm variants and
        # is deliberately excluded from the policy observation.
        self.current_vec = torch.zeros((self.num_envs, 2), device=self.device)

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

    def _resample_current(self, env_ids: torch.Tensor) -> None:
        """Sample one constant world-frame current vector per reset environment."""
        speeds = self.physics_cfg.current_speed_min + self._rand(
            len(env_ids)
        ) * (
            self.physics_cfg.current_speed_max
            - self.physics_cfg.current_speed_min
        )
        directions = self._rand(len(env_ids)) * 2.0 * torch.pi
        self.current_vec[env_ids, 0] = speeds * torch.cos(directions)
        self.current_vec[env_ids, 1] = speeds * torch.sin(directions)

    def _compute_current_forces(self) -> torch.Tensor:
        """Compute quadratic relative-velocity drag in the world frame."""
        vel_w = self.robot.data.root_com_vel_w
        boat_vel_xy = vel_w[:, :2]
        v_rel = boat_vel_xy - self.current_vec
        v_rel_mag = torch.norm(v_rel, dim=1, keepdim=True)
        drag_force_xy = -self.physics_cfg.current_drag_coeff * v_rel * v_rel_mag

        drag_force = torch.zeros((self.num_envs, 3), device=self.device)
        drag_force[:, :2] = drag_force_xy
        return drag_force

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

        if getattr(self.physics_cfg, "physics_version", 1) == 1:
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

        # Match blueboat_calm_nav: current drag is a world-frame force and is
        # added before the shared world-to-body rotation. Calm tasks do not
        # enter the current-force path at all.
        if self.physics_cfg.enable_current:
            force_w += self._compute_current_forces()
        if self._sea is not None:
            # Same world-frame entry point as the current, so a task may carry
            # both and their water velocities simply add.
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
            force_w += wave_f
            torque_w += wave_t

        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_w)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_w)
        if getattr(self.physics_cfg, "physics_version", 1) == 2:
            add_implicit_hull_damping(self, forces, torques, quat)
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
        base = torch.hstack(
            [dot, cross, distance_norm, dock_dot, dock_cross, speed_norm]
        )
        if getattr(self.cfg, "obs_kinematic", False):
            # v11 block: scalar speed_norm above cannot separate surge, sway
            # and residual yaw -- exactly the states a final approach needs.
            from .._shared.kinematics import body_planar_kinematics

            base = torch.hstack(
                (
                    base,
                    body_planar_kinematics(
                        forwards_2d,
                        self._linear_velocity_world()[:, :3],
                        self.robot.data.root_ang_vel_w[:, 2],
                        yaw_rate_scale_rad_s=self.cfg.yaw_rate_obs_scale_rad_s,
                    ),
                )
            )
        if self.cfg.berth_walls:
            # Same 36-ray contract as hazard_nav so the sensor stays frozen.
            rays_norm = self._wall_ray_ranges_m() / self.cfg.ray_max_range_m
            base = torch.hstack([base, rays_norm])
        return {"policy": base}

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
        reward = (
            -distance / self.cfg.reference_reward_distance_scale_m
            + self.cfg.reference_reward_alignment_scale
            * dock_dot
            * torch.exp(-distance / self.cfg.reference_reward_alignment_decay_m)
            + braking_credit
            + self.cfg.reference_reward_success_bonus * instantaneous_success.float()
        )
        if self.cfg.berth_walls:
            # Contact economics carried over verbatim from the certified
            # hazard ledger: one entry event dominates, dwell stays bounded.
            clearance = self._wall_clearance()
            contact_now = clearance < 0.0
            contact_entry = contact_now & ~self._contact_prev
            self._contact_prev.copy_(contact_now)
            reward = (
                reward
                - self.cfg.reward_contact_entry_penalty * contact_entry.float()
                - self.cfg.reward_contact_dwell_penalty * contact_now.float()
            )
        return reward

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

        if self.cfg.berth_walls:
            # Prefix semantics: only contact BEFORE the hold completes voids
            # the episode, mirroring hazard_nav's "clean until achieved" rule.
            clearance = self._wall_clearance()
            pre_success = ~self._success
            self._min_clearance.copy_(
                torch.where(
                    pre_success,
                    torch.minimum(self._min_clearance, clearance),
                    self._min_clearance,
                )
            )
            self._contact_before_success |= pre_success & (clearance < 0.0)

        achieved_now = (
            ~self._success & (self._max_hold_steps >= self.required_hold_steps)
        )
        if self.cfg.berth_walls:
            achieved_now = achieved_now & ~self._contact_before_success
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(achieved_now, elapsed_s, self._first_success_time_s)
        )
        held = self._max_hold_steps >= self.required_hold_steps
        if self.cfg.berth_walls:
            held = held & ~self._contact_before_success
        self._success.copy_(held)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._episode_finished.copy_(time_out)
        terminated = torch.zeros_like(time_out)
        publish_termination_reason(self, terminated, time_out)
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
            # Latched here, alongside episode_path_length and under the same
            # discipline: _max_hold_steps is zeroed further down in this reset,
            # so reading it any later would report the NEXT episode's value.
            self.episode_max_hold_s[completed_ids] = (
                self._max_hold_steps[completed_ids].float() * self.control_step_s
            )

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
            if self.physics_cfg.enable_current:
                self.extras["log"]["Episode/current_speed"] = torch.norm(
                    self.current_vec[completed_ids], dim=1
                ).mean()

            # Each completed environment contributes one episodic EMA sample.
            # Frozen evaluation skips this entirely -- otherwise the difficulty
            # rises inside the eval run and the 128 episodes are not comparable.
            if not getattr(self.cfg, "curriculum_frozen", False):
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

        if self.physics_cfg.enable_current:
            self._resample_current(env_ids)

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
            2.0 * self._rand(num_resets) - 1.0
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
            2.0 * self._rand(num_resets) - 1.0
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
        self._contact_prev[env_ids] = False
        self._contact_before_success[env_ids] = False
        self._min_clearance[env_ids] = torch.inf
        self._episode_finished[env_ids] = False
        self._previous_xy[env_ids] = com_target_xy
        if self._sea is not None:
            # Fresh sea for the episode that starts now, off the per-(env,
            # episode) "wave" stream when seeded. reset_idx MUST advance the
            # episode counter before the draw (tasks/_shared/scenario_rng.py);
            # with no scenario the draw falls back to the global torch RNG,
            # the historical training path.
            if self._sea_scenario is not None:
                self._sea_scenario.reset_idx(env_ids)
            self._sea.resample(env_ids, scenario=self._sea_scenario)
            if getattr(self.physics_cfg, "physics_version", 1) == 2:
                self._wave_clock.reset(env_ids)

    # ======================================================================
    # Journal branch (task contract v2, CONTEXT_R4 part F / task F5).
    #
    # Selected in __init__ when cfg.journal_contract_version == 2; the legacy
    # methods above are never modified (D15): the journal versions are NEW
    # methods bound over _pre_physics_step / _apply_action / _get_dones /
    # _get_rewards / _reset_idx / _get_observations on the instance. The call
    # order is JournalEnvMixin's (tasks/_shared/journal_env_mixin.py docstring).
    # Truth (COM = root-link pose of the origin = COM v2 asset; CONTEXT_R3
    # decision 12) feeds contract, rewards, clearances and metrics; the
    # observation reads the D5 estimates (position, velocity, rays).
    # ======================================================================

    def _journal_init(self, cfg, render_mode, **kwargs):
        from types import SimpleNamespace

        from .journal_d7_pool import load_pool

        self._journal = True
        self.physics_cfg = cfg.underwater_physics_cfg
        # _setup_scene only reads bow_body_axis; every v2 asset has its bow at +x.
        self.vehicle_spec = SimpleNamespace(name=cfg.vehicle, bow_body_axis="+x")
        # D7 B02 preflight BEFORE the scene is built: no legacy appearance, and the
        # admitted pool loaded by path and verified (fail closed).
        if int(getattr(cfg, "appear_count", 0) or 0) != 0:
            raise ValueError("legacy appearance is disabled in journal cfgs; D7 owns dynamic hazards")
        seed = getattr(cfg, "seed", None)
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("journal cfgs must carry an integer evaluation seed (CONTEXT_R4 D7)")
        pool = getattr(cfg, "journal_d7_pool", None)
        if pool is None:
            from .docking_env_cfg import JOURNAL_D7_POOLS, journal_d7_pool_name

            explicit = str(getattr(cfg, "d7_pool_file", "") or "")
            if explicit:
                path = explicit
            else:
                name = journal_d7_pool_name(cfg.vehicle, cfg.journal_geometry_tier, seed, 64, 2,
                                            bool(cfg.berth_walls))
                path = os.path.join(str(cfg.d7_pool_dir), name)
            pinned = JOURNAL_D7_POOLS.get(os.path.basename(path))
            if pinned is None:
                raise ValueError("D7 pool %s has no pinned SHA-256 in docking_env_cfg.JOURNAL_D7_POOLS "
                                 "(admit it with tools_f5/admit_d7_pool.py and pin the digest)" % path)
            pool = load_pool(path, expected_sha256=pinned, vessel=cfg.vehicle, tier=int(cfg.journal_geometry_tier),
                             seed=seed, berth_walls=bool(cfg.berth_walls))
        if bool(getattr(pool, "berth_walls", cfg.berth_walls)) != bool(cfg.berth_walls):
            raise ValueError("the D7 pool was admitted with berth_walls=%r, the cfg has %r"
                             % (getattr(pool, "berth_walls", None), cfg.berth_walls))
        if getattr(pool, "vessel", cfg.vehicle) != cfg.vehicle or getattr(pool, "eval_seed", seed) != seed:
            raise ValueError("the D7 pool was admitted for another vessel or evaluation seed")
        if not all(callable(getattr(pool, name, None)) for name in ("exposure", "geometry_id", "beyond_grid")):
            raise ValueError("journal docking needs the docking D7 pool (exposure(env, k), geometry_id(env, k), "
                             "beyond_grid(env, k, seed); CONTEXT_R4 D16/D24)")
        super().__init__(cfg, render_mode, **kwargs)
        self.journal_setup(d7_pool=pool)
        self._journal_post_setup(pool)
        self._pre_physics_step = self._journal_pre_physics_step
        self._apply_action = self._journal_apply_action
        self._get_dones = self._journal_get_dones
        self._get_rewards = self._journal_get_rewards
        self._reset_idx = self._journal_reset_idx
        self._get_observations = self._journal_get_observations

    def _journal_post_setup(self, pool):
        from .._shared import journal_cfg_v2 as plans
        from .docking_env_cfg import JOURNAL_D7_POOL_MANIFEST_SHA256
        from .journal_contract_torch import DockingEpisodeBatch
        from .journal_d7_exposure import ExposureBridge
        from .journal_d7_pool import suite_matches
        from .journal_docking import build_scenario
        from .level_table import level_table_stamp

        n, device, cfg = self.num_envs, self.device, self.cfg
        if not suite_matches(pool, self._d7_suite):
            raise ValueError("the runtime D7 suite parameters differ from the admitted pool's suite "
                             "(re-admit with tools_f5/admit_d7_pool.py)")
        if int(pool.grid.get("num_envs", 0)) < n:
            raise ValueError("the D7 pool covers %s envs; the env has %d" % (pool.grid.get("num_envs"), n))
        # D16: rows of keys E4 rejected run an empty D0 row (no intruder) and stamp exposure
        # "not_admitted"; admitted rows go to the E4 HazardBridge unchanged.
        self._d7_bridge = ExposureBridge(self._d7_bridge)
        data = self.robot.data
        com_b = getattr(data, "body_com_pos_b", None)
        if com_b is None:
            com_b = getattr(data, "com_pos_b", None)
        if com_b is not None and float(torch.as_tensor(com_b).abs().max()) > 1.0e-5:
            raise ValueError("journal v2 assets must have the actor origin at the COM (CONTEXT_R3 decision 12)")
        self.control_step_s = float(cfg.sim.dt) * int(cfg.decimation)
        if self.control_step_s != plans.CONTROL_DT_S:
            raise ValueError("journal control step must be 1/60 s (sim.dt 1/120 x decimation 2)")
        record = plans.vessel_record(cfg.vehicle)
        plan = cfg.journal_plan
        self._jd_plan, self._jd_record, self._jd_pool = plan, record, pool
        self._jd_params = plan.params
        self._jd_scale = float(self._jd_params["length_scale"])
        self._jd_water_z = float(self.physics_cfg.water_surface_z)
        self._jd_nominal = float(plan.nominal_com_height_m)
        self._jd_limits = plan.validity_limits
        hull = record.footprint.vertices_m
        self._jd_hull = torch.tensor(hull, dtype=torch.float64, device=device)
        self._jd_core_radius = float(record.footprint.radius_m)
        self._jd_oracle = DockingEpisodeBatch(n, device, hull)
        self._jd_berth_walls = bool(cfg.berth_walls)
        probe = build_scenario(cfg.vehicle, plan.geometry_tier, int(cfg.seed), 0, 0, self._jd_berth_walls)
        walls = int(probe.wall_radii.shape[0])
        f64 = torch.float64
        self._jd_walls = walls
        self._jd_wall_centers = torch.zeros((n, walls, 2), dtype=f64, device=device)
        self._jd_wall_radii = torch.zeros((n, walls), dtype=f64, device=device)
        # One D7 slot, separate from the wall storage (D7 spec D1 / B04): capacity + 1.
        self._jd_d7_slot = int(plans.STATIC_CAPACITY["docking"])     # E4 CapacityPlan.dynamic_slots[0] = base_max
        self._d7_centers = torch.zeros((n, 1, 2), dtype=f64, device=device)
        self._d7_radii = torch.zeros((n, 1), dtype=f64, device=device)
        self._d7_physical = torch.zeros((n, 1), dtype=torch.bool, device=device)
        self._d7_visible = torch.zeros((n, 1), dtype=torch.bool, device=device)
        if self._appearing_bridge is not None:
            # G20b: the appearing cylinder's own storage (slot STATIC_CAPACITY + 1 D7 = CapacityPlan.appearing_slots),
            # env-local float64 like the D7 slot; no D7 write can overwrite it
            self._appearing_centers = torch.zeros((n, 1, 2), dtype=f64, device=device)
            self._appearing_radii = torch.zeros((n, 1), dtype=f64, device=device)
            self._appearing_physical = torch.zeros((n, 1), dtype=torch.bool, device=device)
            self._appearing_visible = torch.zeros((n, 1), dtype=torch.bool, device=device)
            self.appearing_hazard_header["collision_model"] = (
                "docking: the cylinder has its own one-slot storage; its swept-contact latch (D7 swept guard) is "
                "OR'ed into the contract collision input (observe_physics and the control-step check) and the "
                "reward contact like the D7 contact; the 36 rays see it once visible (minimum with the walls and "
                "the D7 slot)")
        self._jd_heading = torch.zeros(n, dtype=f64, device=device)
        # G6: the berth target published under the legacy names (docking_env.py:85-88), read by
        # classical_baseline / journal_controllers (DockingLaw), eval_mppi (mode "dock") and
        # _horizontal_distance (cert_v3 d0). WORLD frame, float32 as on the legacy env: the journal
        # berth centre is the env origin (journal_docking.build_scenario: berth frame centre (0, 0))
        # and success needs the bow along the berth heading (journal_contracts_v2
        # DockingEpisode._conjunction); dock_heading is set per episode in _journal_reset_idx.
        self.dock_point = self.scene.env_origins[:, :2].clone()
        self.dock_heading = torch.tensor([1.0, 0.0], device=self.device).repeat(n, 1)
        self._jd_bounds = torch.zeros((n, 4), dtype=f64, device=device)
        self._jd_deadline = torch.zeros(n, dtype=f64, device=device)
        self._jd_prev_pose = torch.zeros((n, 3), dtype=f64, device=device)
        self._jd_prev_clear = torch.full((n,), math.inf, dtype=f64, device=device)
        self._jd_last_clear = torch.full((n,), math.inf, dtype=f64, device=device)
        self._jd_contact_prev = torch.zeros(n, dtype=torch.bool, device=device)
        self._jd_d7_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._jd_max_dwell = torch.zeros(n, dtype=f64, device=device)
        self._jd_task_episode = torch.full((n,), -1, dtype=torch.long)
        self._jd_scenarios = [None] * n
        # Completed-episode latches (written in _get_dones, BEFORE any reset).
        self.journal_episode_results = [None] * n
        self.episode_success = torch.zeros(n, dtype=torch.bool, device=device)
        self.time_to_success = torch.full((n,), torch.nan, device=device)
        self.episode_path_length = torch.zeros(n, device=device)
        self.final_hold_timer = torch.zeros(n, device=device)
        self.episode_max_hold_s = torch.zeros(n, device=device)
        self.episode_min_clearance = torch.full((n,), torch.nan, device=device)
        self.docking_level_table = level_table_stamp(cfg)
        self.dynamic_hazard_header["docking_pool"] = pool.header() if callable(getattr(pool, "header", None)) else {}
        # D16 header counts (per dose, incl. not_admitted) and the pinned pool manifest (D19).
        counts = getattr(pool, "per_dose_exposure_counts", None)
        self.dynamic_hazard_header["per_dose_exposure_counts"] = counts() if callable(counts) else [
            {"dose": d, "offered": int(o), "admitted": int(a), "not_admitted": int(o) - int(a)}
            for d, (o, a) in enumerate(pool.per_dose_counts)]
        self.dynamic_hazard_header["keys_not_admitted"] = int(getattr(pool, "keys_not_admitted", 0))
        self.dynamic_hazard_header["pool_manifest_sha256"] = JOURNAL_D7_POOL_MANIFEST_SHA256
        # D22: header per-dose counts are [offered, admitted, not_admitted] rows in every family (the mixin
        # copied E4's two columns); per_dose_exposure_counts above stays as an extra field.
        self.dynamic_hazard_header["per_dose_counts"] = [[int(row[0]), int(row[1]), int(row[0]) - int(row[1])]
                                                         for row in pool.per_dose_counts]
        self.dynamic_hazard_header["per_dose_counts_fields"] = ["offered", "admitted", "not_admitted"]
        # D24 (c): the pool's offered grid; task episodes k >= episodes run at D0, not admitted, never counted.
        self.dynamic_hazard_header["grid"] = {"num_envs": int(pool.grid["num_envs"]),
                                              "episodes": int(pool.grid["episodes"])}
        self.stop_on_success = True

    # ------------------------------------------------------------ truth views --
    def _journal_truth(self, robot_data, env_origins, water_level_local):
        """Contract RigidState batch (float64, env-local COM = root-link pose)."""
        pose = robot_data.root_link_pose_w.to(torch.float64)
        vel = robot_data.root_com_vel_w.to(torch.float64)
        origins = env_origins.to(torch.float64)
        water = torch.as_tensor(water_level_local, dtype=torch.float64, device=pose.device).expand(self.num_envs)
        return {"com": pose[:, :3] - origins, "quat": pose[:, 3:7], "lin": vel[:, :3], "ang": vel[:, 3:],
                "water": water.clone(), "nominal": torch.full_like(water, self._jd_nominal)}

    def _journal_clearance(self, state, *, sweep=True):
        """Swept hull-polygon clearance to the berth walls since the previous sample (float64)."""
        from .journal_kernels import polygon_circle_clearance, swept_lower_bound, yaw_from_quat

        xy = state["com"][:, :2]
        yaw = yaw_from_quat(state["quat"])
        now = polygon_circle_clearance(xy, yaw, self._jd_hull, self._jd_wall_centers, self._jd_wall_radii)
        now = torch.where(torch.isfinite(xy).all(-1) & torch.isfinite(yaw), now, torch.full_like(now, -math.inf))
        if sweep:
            bound = swept_lower_bound(self._jd_prev_clear, now, self._jd_prev_pose[:, :2], self._jd_prev_pose[:, 2],
                                      xy, yaw, self._jd_core_radius)
        else:
            bound = now
        self._jd_prev_pose = torch.cat((xy, yaw.unsqueeze(-1)), -1)
        self._jd_prev_clear = now
        return bound

    def _journal_d7_contact_now(self):
        rows = self._d7_bridge.rows
        values = [bool(rows[i]["contact"]) if i in rows else False for i in range(self.num_envs)]
        # G11: the host list rarely changes (the latch flips once per contact); while it equals the last
        # uploaded list, a device clone of that upload replaces the host-to-device copy (same bits, a
        # fresh tensor, so the in-place reset write cannot reach the cache)
        cache = getattr(self, "_jd_contact_cache", None)
        if cache is None or cache[0] != values or cache[1].device != torch.device(self.device):
            cache = (values, torch.tensor(values, dtype=torch.bool, device=self.device))
            self._jd_contact_cache = cache
        return cache[1].clone()

    def _journal_out_of_bounds(self, state):
        """COM outside the admitted layout bounds (berth frame, exact)."""
        x, y = state["com"][:, 0], state["com"][:, 1]
        c, s = torch.cos(self._jd_heading), torch.sin(self._jd_heading)
        bx, by = c * x + s * y, -s * x + c * y
        b = self._jd_bounds
        inside = (b[:, 0] <= bx) & (bx <= b[:, 1]) & (b[:, 2] <= by) & (by <= b[:, 3])
        return ~inside & torch.isfinite(x) & torch.isfinite(y)

    # ------------------------------------------------------------ mixin hooks --
    def _journal_observe_physics(self, robot_data, env_origins, water_level_local):
        state = self._journal_truth(robot_data, env_origins, water_level_local)
        clearance = self._journal_clearance(state)
        self._jd_d7_contact = self._journal_d7_contact_now()
        if getattr(self, "_appearing_bridge", None) is not None:
            self._jd_d7_contact = self._jd_d7_contact | self.journal_appearing_contact().to(self.device)   # G20b
        self._jd_oracle.observe_physics(state, collision=self._jd_d7_contact,
                                        clearance_m=clearance if self._jd_walls else None)

    def _journal_publish_appearing(self, samples):
        """G20b mixin hook: write the appearing sample (one slot) into its own storage; physical = visible."""
        if not samples:
            return
        rows = sorted(int(env) for env in samples if samples[env])
        if not rows:
            return
        xy, radius, physical, visible = [], [], [], []
        for env in rows:
            (item,) = samples[env]
            if int(item.slot) != self._appearing_slot or bool(item.physical) != bool(item.visible):
                raise ValueError("appearing sample slot %d / masks do not match the appearing slot" % int(item.slot))
            xy.append((float(item.x), float(item.y)))
            radius.append(float(item.radius_m))
            physical.append(bool(item.physical))
            visible.append(bool(item.visible))
        ids = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        self._appearing_centers[ids, 0] = torch.as_tensor(xy, dtype=torch.float64).to(self.device)
        self._appearing_radii[ids, 0] = torch.as_tensor(radius, dtype=torch.float64).to(self.device)
        self._appearing_physical[ids, 0] = torch.as_tensor(physical, dtype=torch.bool).to(self.device)
        self._appearing_visible[ids, 0] = torch.as_tensor(visible, dtype=torch.bool).to(self.device)

    def _journal_appearing_geometry(self, env_id):
        """G20b mixin hook: geometry id of the env's current docking scenario (the appearing pool records it)."""
        scenario = self._jd_scenarios[int(env_id)]
        if scenario is None:
            raise RuntimeError("env %d has no journal docking scenario" % int(env_id))
        return scenario.geometry_id

    def _journal_family_scenario_hashes(self, env_id):
        scenario = self._jd_scenarios[int(env_id)]
        if scenario is None:
            raise RuntimeError("env %d has no journal docking scenario" % int(env_id))
        return scenario.family_hashes()

    def _journal_publish_d7(self, samples):
        """B07: write D7 samples into the reserved slot (physical / visible masks split)."""
        if not samples:
            return
        rows, xs, ys, rs, phys, vis = [], [], [], [], [], []
        for env, items in samples.items():
            rows.append(int(env))
            if items:
                if len(items) != 1 or int(items[0].slot) != self._jd_d7_slot:
                    raise ValueError("docking owns exactly one D7 slot (%d)" % self._jd_d7_slot)
                item = items[0]
                xs.append(float(item.x))
                ys.append(float(item.y))
                rs.append(float(item.radius_m))
                phys.append(bool(item.physical))
                vis.append(bool(item.visible))
            else:
                xs.append(0.0)
                ys.append(0.0)
                rs.append(0.0)
                phys.append(False)
                vis.append(False)
        if sorted(rows) == list(range(int(self.num_envs))) and not any(samples[k] for k in samples):
            # G11: every row present and empty -> the values written are 0.0 / False everywhere: the same
            # elements written by basic slicing (fill kernels instead of six host-to-device copies)
            self._d7_centers[:, 0, 0] = 0.0
            self._d7_centers[:, 0, 1] = 0.0
            self._d7_radii[:, 0] = 0.0
            self._d7_physical[:, 0] = False
            self._d7_visible[:, 0] = False
            return
        # G11: the six host tensors of the per-field torch.tensor(..., device=) calls, built on the host
        # exactly as those calls build them, moved in ONE host-to-device copy (bytes verbatim)
        ids, xs_t, ys_t, rs_t, phys_t, vis_t = device_copies(
            [torch.tensor(rows, dtype=torch.long), torch.tensor(xs, dtype=torch.float64),
             torch.tensor(ys, dtype=torch.float64), torch.tensor(rs, dtype=torch.float64),
             torch.tensor(phys, dtype=torch.bool), torch.tensor(vis, dtype=torch.bool)], self.device)
        self._d7_centers[ids, 0, 0] = xs_t
        self._d7_centers[ids, 0, 1] = ys_t
        self._d7_radii[ids, 0] = rs_t
        self._d7_physical[ids, 0] = phys_t
        self._d7_visible[ids, 0] = vis_t

    # ------------------------------------------------------------- DirectRLEnv --
    def _journal_pre_physics_step(self, actions):
        self.actions = self.journal_pre_physics(actions)

    def _journal_apply_action(self):
        from .._shared.body_wrench import apply_body_wrench   # G6: no per-call deprecation warning
        forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins, self._jd_water_z)
        apply_body_wrench(self.robot, forces, torques)

    def _journal_get_dones(self):
        view = self.journal_pre_dones(self.robot.data, self.scene.env_origins, self._jd_water_z)
        state = self._journal_truth(self.robot.data, self.scene.env_origins, view.water_level_local)
        clearance = self._journal_clearance(state)
        self._jd_last_clear = clearance
        d7_contact = torch.as_tensor(view.d7_contact, dtype=torch.bool, device=self.device)
        if getattr(self, "_appearing_bridge", None) is not None:
            d7_contact = d7_contact | self.journal_appearing_contact().to(self.device)      # G20b swept latch
        self._jd_d7_contact = d7_contact
        oob = self._journal_out_of_bounds(state)
        time_s = self.episode_length_buf.to(torch.float64) * self.control_step_s
        # No wall geometry (virtual berth): no clearance sample, as the scalar contract's clearance_m=None.
        codes = self._jd_oracle.step(time_s, state, collision=d7_contact, out_of_bounds=oob,
                                     clearance_m=clearance if self._jd_walls else None)
        self._jd_max_dwell = torch.maximum(self._jd_max_dwell, self._jd_oracle.dwell_s)
        # G11: rows with a nonzero code from the host code list the oracle step already transferred
        # (None: no row ended, the step's own ended.any() said so)
        code_list = self._jd_oracle.last_code_list
        finished = [] if code_list is None else [i for i, c in enumerate(code_list) if c != 0]
        for i in finished:
            result = dict(self._jd_oracle.last_completed[i])
            scenario = self._jd_scenarios[i]
            exposure = self._jd_pool.exposure(i, scenario.episode)
            reasons = self._jd_pool.not_admitted_reasons(i, scenario.episode)
            result.update(task_episode=int(scenario.episode), d7_exposure=exposure,
                          d7_not_admitted_reasons=[list(r) for r in reasons],
                          d7_requested_dose=int(self._d7_suite.dose),
                          d7_run_dose=0 if exposure == "not_admitted" else int(self._d7_suite.dose),
                          berth_heading_rad=float(scenario.berth_heading_rad),
                          scenario_deadline_s=float(scenario.deadline_s),
                          docking_max_valid_dwell_s=float(self._jd_max_dwell[i]),
                          docking_level_table_in_force=self.docking_level_table["docking_level_table_in_force"])
            self.journal_episode_results[i] = result
            self.episode_success[i] = bool(result["episode_success"])
            self.time_to_success[i] = float(result["time_to_success"])
            self.episode_path_length[i] = float(result["full_episode_path_length_m"])
            self.final_hold_timer[i] = float(result["docking_valid_dwell_s"])
            self.episode_max_hold_s[i] = float(self._jd_max_dwell[i])
            self.episode_min_clearance[i] = float(result["min_clearance_m"])
        return self.journal_dones(codes.to(self.device))

    def _journal_get_rewards(self):
        """Reference baseline only (truth); benchmark methods may use any shaping."""
        from .journal_kernels import yaw_from_quat

        cfg = self.cfg
        pose = self.robot.data.root_link_pose_w.to(torch.float64)
        xy = pose[:, :2] - self.scene.env_origins[:, :2].to(torch.float64)
        distance = torch.linalg.vector_norm(xy, dim=-1)          # berth centre = env origin
        dock_dot = torch.cos(yaw_from_quat(pose[:, 3:7]) - self._jd_heading)
        speed = torch.linalg.vector_norm(self.robot.data.root_com_vel_w[:, :2].to(torch.float64), dim=-1)
        braking = (cfg.reference_reward_braking_scale * torch.exp(-distance / cfg.reference_reward_braking_decay_m)
                   * torch.clamp(1.0 - speed / cfg.reference_reward_braking_speed_scale_mps, 0.0, 1.0))
        reward = (-distance / cfg.reference_reward_distance_scale_m
                  + cfg.reference_reward_alignment_scale * dock_dot
                  * torch.exp(-distance / cfg.reference_reward_alignment_decay_m)
                  + braking + cfg.reference_reward_success_bonus * self._jd_oracle.conjunction.to(torch.float64))
        contact_now = (self._jd_last_clear <= 0.0) | self._jd_d7_contact
        entry = contact_now & ~self._jd_contact_prev
        self._jd_contact_prev = contact_now.clone()
        reward = (reward - cfg.reward_contact_entry_penalty * entry.to(torch.float64)
                  - cfg.reward_contact_dwell_penalty * contact_now.to(torch.float64))
        return reward.to(torch.float32)

    def _journal_reset_idx(self, env_ids):
        from .journal_docking import build_scenario
        from .journal_kernels import polygon_circle_clearance

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if not ids.numel():
            return
        super()._reset_idx(ids)
        cfg, pool = self.cfg, self._jd_pool
        host = ids.detach().cpu().tolist()
        scenarios = []
        for i in host:
            self._jd_task_episode[i] += 1
            k = int(self._jd_task_episode[i])
            scenario = build_scenario(cfg.vehicle, self._jd_plan.geometry_tier, int(cfg.seed), i, k,
                                      self._jd_berth_walls)
            # D24: a task episode beyond the pool's offered grid has no pool geometry record: its E5
            # scenario (a pure function of the key) is built as for every key and the check is skipped.
            if not pool.beyond_grid(i, k, int(cfg.seed)) and scenario.geometry_id != pool.geometry_id(i, k):
                raise ValueError("rebuilt docking geometry for env %d episode %d differs from the admitted pool"
                                 % (i, k))
            if int(scenario.wall_radii.shape[0]) != self._jd_walls:
                raise ValueError("berth wall count changed between scenarios")
            self._jd_scenarios[i] = scenario
            scenarios.append(scenario)
        f64 = torch.float64
        dev = self.device
        if self._jd_walls:
            self._jd_wall_centers[ids] = torch.tensor(np.stack([s.wall_centers for s in scenarios]), dtype=f64,
                                                      device=dev)
            self._jd_wall_radii[ids] = torch.tensor(np.stack([s.wall_radii for s in scenarios]), dtype=f64,
                                                    device=dev)
        heading = torch.tensor([s.berth_heading_rad for s in scenarios], dtype=f64, device=dev)
        self._jd_heading[ids] = heading
        self.dock_heading[ids] = torch.stack((torch.cos(heading), torch.sin(heading)), -1).to(self.dock_heading.dtype)
        self._jd_bounds[ids] = torch.tensor([s.bounds_berth_frame for s in scenarios], dtype=f64, device=dev)
        self._jd_deadline[ids] = torch.tensor([s.deadline_s for s in scenarios], dtype=f64, device=dev)
        # Spawn: the COM (= actor origin for v2 assets) at the admitted start, at the calm
        # equilibrium height above the water (D3 section 4: full-XYZ COM write).
        start = torch.tensor([s.start_xy for s in scenarios], dtype=f64, device=dev)
        yaw = torch.tensor([s.start_heading_rad for s in scenarios], dtype=f64, device=dev)
        origins = self.scene.env_origins[ids].to(f64)
        com_w = torch.stack((origins[:, 0] + start[:, 0], origins[:, 1] + start[:, 1],
                             origins[:, 2] + self._jd_water_z + self._jd_nominal), -1)
        quat = torch.stack((torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw),
                            torch.sin(0.5 * yaw)), -1)
        from .._shared.reference_point import actor_position_from_com
        from .._shared.physics_v2 import rotation_matrix

        offset = torch.zeros_like(com_w)          # origin = COM (checked at construction)
        actor = actor_position_from_com(com_w, rotation_matrix(quat), offset)
        root_state = self.robot.data.default_root_state[ids].clone()
        root_state[:, :3] = actor.to(root_state.dtype)
        root_state[:, 3:7] = quat.to(root_state.dtype)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, ids)
        # contract oracle (tensor twin) for the new episode
        b = [s.berth for s in scenarios]
        self._jd_oracle.reset(
            ids, start,
            berth=dict(center_x=[x["center_xy"][0] for x in b], center_y=[x["center_xy"][1] for x in b],
                       heading_rad=heading, opening_x_m=[x["opening_x_m"] for x in b],
                       back_x_m=[x["back_x_m"] for x in b], half_width_m=[x["half_width_m"] for x in b]),
            deadline_s=self._jd_deadline[ids], hold_s=[x["hold_s"] for x in b],
            position_tolerance_m=[x["position_tolerance_m"] for x in b],
            heading_tolerance_rad=[x["heading_tolerance_rad"] for x in b],
            speed_tolerance_mps=[x["speed_tolerance_m_s"] for x in b],
            capsize_angle_rad=self._jd_limits["capsize_angle_rad"], min_heave_m=self._jd_limits["min_heave_m"],
            max_heave_m=self._jd_limits["max_heave_m"])
        # swept-clearance reference = the spawn pose
        spawn_clear = polygon_circle_clearance(start, yaw, self._jd_hull, self._jd_wall_centers[ids],
                                               self._jd_wall_radii[ids])
        self._jd_prev_pose[ids] = torch.cat((start, yaw.unsqueeze(-1)), -1)
        self._jd_prev_clear[ids] = spawn_clear
        self._jd_last_clear[ids] = spawn_clear
        self._jd_contact_prev[ids] = False
        self._jd_d7_contact[ids] = False
        self._jd_max_dwell[ids] = 0.0
        self._d7_physical[ids] = False
        self._d7_visible[ids] = False
        if getattr(self, "_appearing_bridge", None) is not None:
            self._appearing_physical[ids] = False
            self._appearing_visible[ids] = False
        # B06 after the root write: runtime, conditions (reference axis = berth heading,
        # consecutive task episode ids), D7 rows (admitted assignment, or the D16 empty
        # not-admitted row), tick-0 step.
        self.journal_reset(ids, heading, self._jd_task_episode[ids.cpu()].clone(), outage_start_s=None,
                           reference_route_length_m=torch.tensor(
                               [float(s.layout.nominal_route_length_m) for s in scenarios], dtype=f64))   # G28

    def _journal_get_observations(self):
        """journal_dock6_ray36: native six (berth pose from the POSITION ESTIMATE) + 36 ranges."""
        from .journal_kernels import fan_angles, ray_circle_ranges, yaw_from_quat

        data = self.robot.data
        pose = data.root_link_pose_w
        origins = self.scene.env_origins
        com_xy = pose[:, :2] - origins[:, :2]
        quat = pose[:, 3:7]
        w, x, y, z = quat.unbind(-1)
        rot = torch.stack((
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)
        vel = data.root_com_vel_w
        v_body = (rot.transpose(-1, -2) @ vel[:, :3].unsqueeze(-1)).squeeze(-1)
        w_body = (rot.transpose(-1, -2) @ vel[:, 3:6].unsqueeze(-1)).squeeze(-1)
        body_rates = torch.stack((v_body[:, 0], v_body[:, 1], w_body[:, 2]), -1)
        yaw = yaw_from_quat(quat.to(torch.float64))
        angles = fan_angles(yaw, self.cfg.ray_count)
        origin64 = com_xy.to(torch.float64)
        walls = ray_circle_ranges(origin64, angles, self._jd_wall_centers, self._jd_wall_radii,
                                  self.cfg.ray_max_range_m)
        if getattr(self, "_appearing_bridge", None) is not None:
            # G20b: the appearing slot joins the ray block once visible (ray-circle, float64 before the cast;
            # merge G23c L1 x G20b: it is not a D7 intruder slot, so no point-return proxy)
            walls = torch.minimum(walls, ray_circle_ranges(origin64, angles, self._appearing_centers,
                                                           self._appearing_radii, self.cfg.ray_max_range_m,
                                                           active=self._appearing_visible))
        # G23c F8: the visible D7 intruder through the point-return proxy (visibility.py), walls unchanged
        from .._shared import d7_point_return as _d7pr
        rays = _d7pr.merge(walls, origin64, yaw, self._d7_centers, self._d7_radii, self._d7_visible,
                           int(self.cfg.ray_count), float(self.cfg.ray_max_range_m)).to(com_xy.dtype)
        est = self.journal_estimates(com_xy, body_rates, rays)
        heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1).to(com_xy.dtype)
        dock = torch.stack((torch.cos(self._jd_heading), torch.sin(self._jd_heading)), -1).to(com_xy.dtype)
        rpos = -est.position                          # berth centre = env origin (env-local)
        distance = torch.linalg.vector_norm(rpos, dim=-1, keepdim=True)
        direction = rpos / distance.clamp(min=1.0e-6)
        dot = torch.sum(heading * direction, dim=-1, keepdim=True)
        cross = heading[:, 0:1] * direction[:, 1:2] - heading[:, 1:2] * direction[:, 0:1]
        dock_dot = torch.sum(heading * dock, dim=-1, keepdim=True)
        dock_cross = heading[:, 0:1] * dock[:, 1:2] - heading[:, 1:2] * dock[:, 0:1]
        speed_norm = (torch.linalg.vector_norm(est.velocity[:, :2], dim=-1, keepdim=True)
                      / self.cfg.observation_speed_scale_mps)
        base = torch.hstack((dot, cross, distance / self.cfg.observation_distance_scale_m, dock_dot, dock_cross,
                             speed_norm, est.rays / self.cfg.ray_max_range_m))
        return {"policy": base.to(torch.float32)}
