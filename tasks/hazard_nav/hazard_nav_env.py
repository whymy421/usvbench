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

from .._shared.obs_superset import (
    SLICES,
    SPEED_SCALE_MPS,
    SUPERSET_DIM,
    SUPERSET_DIM_V2,
)
from .._shared.restoring import restoring_torque_body
from .._shared.scenario_draws import make_scenario_rng, stamp_scenario
from .._shared.scenario_draws_hazard import (
    GROUP_DRAG_SCALE,
    GROUP_LAYOUT,
    GROUP_MASS_SCALE,
    GROUP_MOTOR_TAU_S,
    GROUP_ROTATION,
    GROUP_THRUST_CAP_SCALE,
    GROUP_THRUST_IMBALANCE,
    actuator_choice_indices,
    actuator_scenario_entry,
    episode_layout_rng,
    layout_rotation_angle,
)
from .._shared.vehicles import get_vehicle
from ..docking.curriculum import DockingCurriculum
from .hazard_geometry import (
    GRID_CELL_M,
    HALF_BEAM_M,
    analytic_min_clearance,
    geodesic_descent_directions,
    geodesic_distance_field,
    ray_circle_ranges,
    sample_band_fortress_layout,
    sample_double_ring_fortress_layout,
    sample_double_ring_layout,
    sample_forced_crossing_layout,
    sample_iceberg_layout,
    sample_layout,
    sample_open_basin_layout,
    sample_ring_fortress_layout,
    sample_ring_layout,
)
from .hazard_nav_env_cfg import HazardNavEnvCfg


def _space_dim(space: object) -> int:
    shape = getattr(space, "shape", None)
    return int(shape[0]) if shape else int(space)


_GEODESIC_FIELD_EXTENT_M = 45.0
_GEODESIC_FIELD_PADDING_M = 1.0
_GEODESIC_FIELD_CELLS = int(2.0 * _GEODESIC_FIELD_EXTENT_M / GRID_CELL_M) + 1


def _bilinear_numpy(field: np.ndarray, xy: np.ndarray, origin: np.ndarray) -> float:
    """Sample one in-bounds CPU field using the runtime interpolation rule."""
    col_f, row_f = (np.asarray(xy) - origin) / GRID_CELL_M
    col0, row0 = int(math.floor(col_f)), int(math.floor(row_f))
    col1, row1 = min(col0 + 1, field.shape[1] - 1), min(
        row0 + 1, field.shape[0] - 1
    )
    col_t, row_t = col_f - col0, row_f - row0
    return float(
        (1.0 - row_t) * (1.0 - col_t) * field[row0, col0]
        + (1.0 - row_t) * col_t * field[row0, col1]
        + row_t * (1.0 - col_t) * field[row1, col0]
        + row_t * col_t * field[row1, col1]
    )


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
        # UNSEEDED BEHAVIOUR, UNCHANGED AND DOCUMENTED RATHER THAN FIXED: with
        # cfg.seed None this falls back to cfg.layout_seed, which defaults to
        # the CONSTANT 0 (hazard_nav_env_cfg.py:187), so an unseeded run replays
        # one fixed layout sequence instead of drawing fresh entropy. This
        # generator remains the whole scenario source on that path (see
        # episode_layout_rng and the scenario block below); the migration changes
        # only the SEEDED path, which is the one every evaluator takes
        # (scripts/eval_v6_frozen.py:104 sets env_cfg.seed).
        isaac_seed = getattr(cfg, "seed", None)
        layout_seed = cfg.layout_seed if isaac_seed is None else int(isaac_seed)
        self._layout_rng = np.random.default_rng(layout_seed)
        super().__init__(cfg, render_mode, **kwargs)

        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        self.goal_radius = float(self.cfg.goal_radius)
        self.target_pos = self.scene.env_origins[:, :2].clone()
        self.d0_per_env = torch.ones(self.num_envs, device=self.device)
        self._thrust_imbalance_per_env = torch.zeros(
            self.num_envs, device=self.device
        )
        self._mass_scale_per_env = torch.ones(self.num_envs, device=self.device)
        self._drag_scale_per_env = torch.ones(self.num_envs, device=self.device)
        self._thrust_cap_scale_per_env = torch.ones(
            self.num_envs, device=self.device
        )
        self._motor_tau_s_per_env = torch.zeros(
            self.num_envs, device=self.device
        )
        self._applied_thrust_per_env = torch.zeros(
            self.num_envs, device=self.device
        )
        self._applied_yaw_per_env = torch.zeros(
            self.num_envs, device=self.device
        )

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
        if self.cfg.reward_progress_geodesic:
            self._geodesic_fields = torch.full(
                (
                    self.num_envs,
                    _GEODESIC_FIELD_CELLS,
                    _GEODESIC_FIELD_CELLS,
                ),
                torch.inf,
                device=self.device,
            )
            self._geodesic_field_origin = torch.zeros(
                (self.num_envs, 2), device=self.device
            )
            self._geodesic_field_cell = torch.full(
                (self.num_envs,), GRID_CELL_M, device=self.device
            )
            self._geodesic_field_shape = torch.zeros(
                (self.num_envs, 2), dtype=torch.long, device=self.device
            )
            self._geodesic_field_max = torch.zeros(
                self.num_envs, device=self.device
            )
            if self.cfg.nav_targets_waypoint:
                self._geodesic_descent_directions = torch.zeros(
                    (
                        self.num_envs,
                        _GEODESIC_FIELD_CELLS,
                        _GEODESIC_FIELD_CELLS,
                        2,
                    ),
                    dtype=torch.int8,
                    device=self.device,
                )

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
        self.episode_contact_steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_contact_longest_steps = torch.zeros(
            self.num_envs, device=self.device
        )
        self.episode_contact_depth_sum = torch.zeros(
            self.num_envs, device=self.device
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
        self._ep_open_water_cost = torch.zeros(self.num_envs, device=self.device)
        self._ep_contact_entries = torch.zeros(self.num_envs, device=self.device)
        self._ep_contact_steps = torch.zeros(self.num_envs, device=self.device)
        self._ep_contact_run_steps = torch.zeros(self.num_envs, device=self.device)
        self._ep_contact_longest_steps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._ep_contact_depth_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_goal_bonus = torch.zeros(self.num_envs, device=self.device)
        self._ep_reverse_cost = torch.zeros(self.num_envs, device=self.device)

        # --- Controller-independent episode scenarios -----------------------
        # The layout, the global rotation and the five actuator/dynamics knobs
        # all came off self._layout_rng (:118): ONE generator, advanced once per
        # reset for whatever batch happened to be resetting. That is
        # seed-derived, and it is still not reproducible across controllers,
        # because this family terminates EARLY -- on goal or on contact
        # (_get_dones at :1453-1454, and contact_terminates defaults True at
        # hazard_nav_env_cfg.py:232). Two controllers therefore arrive at reset
        # k at different times and in different groupings, consume a different
        # number of rejection-sampler draws, and are handed DIFFERENT layouts
        # for the same (eval seed, env, episode).
        #
        # Measured, at one eval seed, on the crossing task: three controllers
        # certified over 128 episodes each agreed on only 53.8% / 56.9% / 55.6%
        # of shared (env, episode) slots when compared on route_geodesic_m and
        # d0_m, two fields that depend on the layout ALONE. Every primitive now
        # rides a stream keyed by (protocol version, cfg.seed, env index, per-env
        # episode index, group), which cannot drift with consumption order.
        #
        # cfg.seed None -> None -> the historical single self._layout_rng
        # sequence, unchanged (see the comment at :108-115 for what it is).
        self._scenario = make_scenario_rng(self.cfg, self.num_envs, self.device)
        # _scenario_* is the RUNNING episode; episode_scenario_* is latched at
        # reset for the episode that just ENDED, matching episode_min_clearance,
        # which is where scripts/eval_v6_frozen.py:213-215 reads it.
        self._scenario_params = [{} for _ in range(self.num_envs)]
        self._scenario_hashes = [{} for _ in range(self.num_envs)]
        self.episode_scenario = [{} for _ in range(self.num_envs)]
        self.episode_scenario_hashes = [{} for _ in range(self.num_envs)]

        # Owner's one-shot half-sine threading bonus. Off unless the amplitude
        # is positive, so every certified id keeps its exact reward.
        self._threading = None
        if self.cfg.reward_threading_amplitude > 0.0:
            from .._shared.threading_gate import ThreadingLatch

            self._threading = ThreadingLatch(
                self.num_envs, self.device,
                amplitude=self.cfg.reward_threading_amplitude,
            )

        # --- Eval-time observation-channel degradation (OOD hooks) ----------
        # Every knob defaults to zero, in which case NO degrader object is
        # built and the guarded hooks in _get_observations never fire: the
        # frozen observation path stays byte-identical. Doses are meant to be
        # set per run via scripts/eval_v6_frozen.py --set, never on a
        # registered training id. Groups carry PHYSICAL units and are applied
        # to ideal measurements before normalization; reward/termination keep
        # reading the clean state.
        noise_pos = float(getattr(self.cfg, "obs_noise_sigma_pos", 0.0) or 0.0)
        noise_vel = float(getattr(self.cfg, "obs_noise_sigma_vel", 0.0) or 0.0)
        noise_ray = float(getattr(self.cfg, "obs_noise_sigma_ray", 0.0) or 0.0)
        bias_pos = float(getattr(self.cfg, "obs_bias_sigma_pos", 0.0) or 0.0)
        bias_vel = float(getattr(self.cfg, "obs_bias_sigma_vel", 0.0) or 0.0)
        bias_ray = float(getattr(self.cfg, "obs_bias_sigma_ray", 0.0) or 0.0)
        delay_steps = int(getattr(self.cfg, "obs_delay_steps", 0) or 0)
        dropout_p = float(getattr(self.cfg, "obs_dropout_p", 0.0) or 0.0)
        self._obs_degrader = None
        if any((noise_pos, noise_vel, noise_ray, bias_pos, bias_vel,
                bias_ray, delay_steps, dropout_p)):
            from .._shared.obs_degradation import (
                ObsChannelGroup,
                build_obs_degrader,
            )

            # The yaw-rate channel rides the vel knob at the same fraction of
            # its full observation scale as the linear channels: sigma_yaw =
            # sigma_vel * (yaw_rate_obs_scale_rad_s / SPEED_SCALE_MPS).
            yaw_ratio = (
                float(self.cfg.yaw_rate_obs_scale_rad_s) / SPEED_SCALE_MPS
            )
            ray_count = int(self.cfg.ray_count)
            self._obs_degrader = build_obs_degrader(
                num_envs=self.num_envs,
                device=self.device,
                base_seed=layout_seed,
                groups={
                    "pos": ObsChannelGroup(
                        dim=2,
                        noise_sigma=(noise_pos,) * 2,
                        bias_sigma=(bias_pos,) * 2,
                    ),
                    "vel": ObsChannelGroup(
                        dim=3,
                        noise_sigma=(
                            noise_vel, noise_vel, noise_vel * yaw_ratio
                        ),
                        bias_sigma=(
                            bias_vel, bias_vel, bias_vel * yaw_ratio
                        ),
                    ),
                    "ray": ObsChannelGroup(
                        dim=ray_count,
                        noise_sigma=(noise_ray,) * ray_count,
                        bias_sigma=(bias_ray,) * ray_count,
                        clamp_min=0.0,
                        clamp_max=float(self.cfg.ray_max_range_m),
                    ),
                },
                delay_steps=delay_steps,
                dropout_p=dropout_p,
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
        if self.cfg.thrust_cap_scale_choices:
            thrust_cap_scale = self._thrust_cap_scale_per_env
        else:
            thrust_cap_scale = self.cfg.thrust_cap_scale
        if (
            self.cfg.thrust_cap_scale_choices
            or self.cfg.thrust_cap_scale != 1.0
        ):
            thrust = thrust * thrust_cap_scale
        yaw_command = self.actions[:, 1] * self.cfg.yaw_torque_max
        if self.cfg.thrust_imbalance_choices:
            imbalance = self._thrust_imbalance_per_env
        else:
            imbalance = self.cfg.thrust_imbalance
        if self.cfg.thrust_imbalance_choices or self.cfg.thrust_imbalance != 0.0:
            # Suite D axis: one motor pulls harder than the other. The action
            # space is lumped (surge, yaw), so scaling the net force would only
            # make the boat slower -- it would not veer, which is the whole
            # point of an imbalance. Split the command back into the port and
            # starboard thruster shares a differential hull would actually
            # produce, scale each side, then recombine:
            #     T_port + T_stbd            = surge force
            #     (T_stbd - T_port) * lever  = yaw torque
            # At imbalance 0 this is an algebraic identity, so every existing
            # id stays bit-identical (asserted in test_thrust_imbalance.py).
            lever = max(float(self.cfg.half_beam_m), 1.0e-6)
            half_yaw = yaw_command / lever
            t_stbd = 0.5 * (thrust + half_yaw)
            t_port = 0.5 * (thrust - half_yaw)
            t_port = t_port * (1.0 - imbalance)
            t_stbd = t_stbd * (1.0 + imbalance)
            thrust = t_port + t_stbd
            yaw_command = (t_stbd - t_port) * lever

        if self.cfg.motor_tau_s_choices:
            motor_tau_s = self._motor_tau_s_per_env
        else:
            motor_tau_s = self.cfg.motor_tau_s
        if self.cfg.motor_tau_s_choices or self.cfg.motor_tau_s != 0.0:
            # Backward-Euler first-order lag, independently for surge and yaw.
            # torch.where is essential for a mixed pack containing tau=0: the
            # selected command is the original tensor exactly, not the result
            # of a nominally equivalent subtract/add round trip.
            physics_step_s = self.cfg.sim.dt
            alpha = physics_step_s / (motor_tau_s + physics_step_s)
            filtered_thrust = self._applied_thrust_per_env + (
                thrust - self._applied_thrust_per_env
            ) * alpha
            filtered_yaw = self._applied_yaw_per_env + (
                yaw_command - self._applied_yaw_per_env
            ) * alpha
            if self.cfg.motor_tau_s_choices:
                thrust = torch.where(motor_tau_s == 0.0, thrust, filtered_thrust)
                yaw_command = torch.where(
                    motor_tau_s == 0.0, yaw_command, filtered_yaw
                )
            else:
                thrust = filtered_thrust
                yaw_command = filtered_yaw
            self._applied_thrust_per_env.copy_(thrust)
            self._applied_yaw_per_env.copy_(yaw_command)

        forces[:, 0, 0] = thrust * self._fwd_x
        forces[:, 0, 1] = thrust * self._fwd_y
        torques[:, 0, 2] = yaw_command

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
            planar_drag = -(
                self.physics_cfg.surge_lin_damping
                + self.physics_cfg.surge_quad_damping * planar_speed
            ) * planar_velocity
            if self.cfg.drag_scale_choices:
                planar_drag = planar_drag * self._drag_scale_per_env.unsqueeze(-1)
            elif self.cfg.drag_scale != 1.0:
                planar_drag = planar_drag * self.cfg.drag_scale
            force_world[:, :2] += planar_drag
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
            if self.cfg.drag_scale_choices:
                drag_body[:, :2] *= self._drag_scale_per_env.unsqueeze(-1)
            elif self.cfg.drag_scale != 1.0:
                drag_body[:, :2] *= self.cfg.drag_scale
            forces[:, 0, :2] += drag_body[:, :2]
        force_world[:, 2] += -self.physics_cfg.heave_damping * linear_velocity[:, 2]

        yaw_rate = angular_velocity[:, 2]
        yaw_drag = -(
            self.physics_cfg.yaw_lin_damping
            + self.physics_cfg.yaw_quad_damping * torch.abs(yaw_rate)
        ) * yaw_rate
        if self.cfg.drag_scale_choices:
            yaw_drag = yaw_drag * self._drag_scale_per_env
        elif self.cfg.drag_scale != 1.0:
            yaw_drag = yaw_drag * self.cfg.drag_scale
        torque_world[:, 2] += yaw_drag
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
        if self.cfg.mass_scale_choices:
            # The rigid-body mass/inertia live in the USD. Dividing the full
            # external wrench gives the same free-body accelerations as a
            # common mass/inertia multiplier without mutating the asset.
            mass_scale = self._mass_scale_per_env.unsqueeze(-1).unsqueeze(-1)
            scaled_forces = forces / mass_scale
            scaled_torques = torques / mass_scale
            identity = mass_scale == 1.0
            forces = torch.where(identity, forces, scaled_forces)
            torques = torch.where(identity, torques, scaled_torques)
        elif self.cfg.mass_scale != 1.0:
            forces = forces / self.cfg.mass_scale
            torques = torques / self.cfg.mass_scale
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

    def _body_motion_measurement(self) -> torch.Tensor:
        """Raw body-frame (N, 3) [surge m/s, sway m/s, yaw rate rad/s].

        The ideal kinematic measurement BEFORE normalization -- the injection
        point for the eval-time observation degradation hooks.
        """
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
        return torch.hstack((surge, sway, angular_body[:, 2:3]))

    def _body_motion_observation(
        self, measurement: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        # Split into measurement + normalization (bit-identical ops) so the
        # degradation hook can corrupt the physical-unit triple in between.
        # Callers that pass nothing -- the reward path and external
        # diagnostics -- always read the CLEAN measurement.
        if measurement is None:
            measurement = self._body_motion_measurement()
        surge_norm = torch.clamp(
            measurement[:, 0:1] / SPEED_SCALE_MPS, min=-1.0, max=1.0
        )
        sway_norm = torch.clamp(
            measurement[:, 1:2] / SPEED_SCALE_MPS, min=-1.0, max=1.0
        )
        yaw_rate_norm = torch.clamp(
            measurement[:, 2:3] / self.cfg.yaw_rate_obs_scale_rad_s,
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
        degrader = self._obs_degrader
        position_to_goal = self.target_pos - self._com_xy()
        distance = torch.norm(position_to_goal, dim=-1, keepdim=True)
        if self.cfg.nav_targets_waypoint:
            geodesic_distance = self._geodesic_progress_distance(
                distance.squeeze(-1)
            )
            waypoint = self._geodesic_waypoint(geodesic_distance)
            position_to_goal = waypoint - self._com_xy()
            distance = geodesic_distance.unsqueeze(-1)
            bearing_distance = torch.norm(position_to_goal, dim=-1, keepdim=True)
        else:
            bearing_distance = distance
        if degrader is not None and degrader.wants("pos"):
            # OOD hook: corrupt the assembled goal vector (ideal relative-
            # position measurement, meters) and re-derive the norms from it.
            # In waypoint mode the geodesic distance channel above stays
            # clean (the carrot machinery reads true state); only the bearing
            # vector is a sensor here.
            position_to_goal = degrader.apply("pos", position_to_goal)
            bearing_distance = torch.norm(
                position_to_goal, dim=-1, keepdim=True
            )
            if not self.cfg.nav_targets_waypoint:
                distance = bearing_distance
        degraded_kin = None
        if degrader is not None and degrader.wants("vel"):
            # OOD hook: body-frame velocity measurement (surge m/s, sway m/s,
            # yaw rad/s), degraded exactly once per step -- delay/dropout are
            # stateful -- and reused by every kinematic feature below.
            degraded_kin = degrader.apply(
                "vel", self._body_motion_measurement()
            )
        direction_to_goal = position_to_goal / bearing_distance.clamp_min(1.0e-6)
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
        if degrader is not None and degrader.wants("ray"):
            # OOD hook: rangefinder returns in METERS, before normalization
            # and before feasibility pooling; the group spec clamps back to
            # the physical [0, ray_max_range_m] envelope. The reward path
            # recomputes its own clean rays.
            ranges = degrader.apply("ray", ranges)
        ranges_norm = ranges / self.cfg.ray_max_range_m

        speed_norm = (
            torch.norm(
                self.robot.data.root_com_vel_w[:, :2], dim=-1, keepdim=True
            )
            / SPEED_SCALE_MPS
        )
        if degraded_kin is not None:
            # Speed is the same velocity sensor: derive it from the degraded
            # planar measurement instead of the clean world-frame state.
            speed_norm = (
                torch.norm(degraded_kin[:, :2], dim=-1, keepdim=True)
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

            # Same tensor as the ranges block above: at defaults this reuses
            # the deterministic recompute bit for bit; under ray degradation
            # the pooled sectors must see the SAME degraded returns (a second
            # _ray_ranges_m() call here would silently bypass the hook).
            pooled = feasibility_pool(
                ranges,
                n_sectors=self.cfg.feasibility_sectors,
                vessel_width_m=2.0 * self.cfg.half_beam_m,
                ray_spacing_rad=2.0 * torch.pi / self.cfg.ray_count,
                width_multiplier=self.cfg.feasibility_width_multiplier,
            )
            ranges_norm = pooled / self.cfg.ray_max_range_m

        if self.cfg.obs_v3:
            surge_norm, sway_norm, yaw_rate_norm = self._body_motion_observation(
                degraded_kin
            )
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
        if getattr(self.cfg, "superset_version", 1) == 2:
            contract_observation = superset_observation.new_zeros(
                (*superset_observation.shape[:-1], SUPERSET_DIM_V2)
            )
            contract_observation[..., :SUPERSET_DIM] = superset_observation
            contract_observation[..., SLICES["kinematics"]] = torch.hstack(
                self._body_motion_observation(degraded_kin)
            )
            # Hazard navigation has no task-specific v2 channels.
            contract_observation[..., SLICES["task_specific"]] = 0.0
            contract_observation[..., SLICES["reserved"]] = 0.0
            return {"policy": contract_observation}
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

    def _geodesic_progress_distance(
        self, euclidean_distance: torch.Tensor
    ) -> torch.Tensor:
        """Bilinearly sample each environment's local geodesic field."""
        local_xy = self._com_xy() - self.scene.env_origins[:, :2]
        grid_xy = (local_xy - self._geodesic_field_origin) / (
            self._geodesic_field_cell.unsqueeze(-1)
        )
        col_f, row_f = grid_xy.unbind(dim=-1)
        height = self._geodesic_field_shape[:, 0]
        width = self._geodesic_field_shape[:, 1]
        outside = (
            (col_f < 0.0)
            | (row_f < 0.0)
            | (col_f > (width - 1).float())
            | (row_f > (height - 1).float())
        )

        col_f = torch.minimum(
            torch.clamp_min(col_f, 0.0), (width - 1).float()
        )
        row_f = torch.minimum(
            torch.clamp_min(row_f, 0.0), (height - 1).float()
        )
        col0 = torch.floor(col_f).long()
        row0 = torch.floor(row_f).long()
        col1 = torch.minimum(col0 + 1, width - 1)
        row1 = torch.minimum(row0 + 1, height - 1)
        col_t = col_f - col0.float()
        row_t = row_f - row0.float()
        env_index = torch.arange(self.num_envs, device=self.device)

        def weighted(row: torch.Tensor, col: torch.Tensor, weight: torch.Tensor):
            values = self._geodesic_fields[env_index, row, col]
            return torch.where(weight > 0.0, values * weight, torch.zeros_like(values))

        sampled = (
            weighted(row0, col0, (1.0 - row_t) * (1.0 - col_t))
            + weighted(row0, col1, (1.0 - row_t) * col_t)
            + weighted(row1, col0, row_t * (1.0 - col_t))
            + weighted(row1, col1, row_t * col_t)
        )
        # Leaving the solved arena (or overlapping an inflated blocked cell)
        # cannot manufacture progress: use field_max + straight-line distance.
        fallback = self._geodesic_field_max + euclidean_distance
        return torch.where(outside | ~torch.isfinite(sampled), fallback, sampled)

    def _geodesic_waypoint(
        self, remaining_geodesic: torch.Tensor
    ) -> torch.Tensor:
        """Follow each prebuilt descent map to its metric-lookahead cell."""
        local_xy = self._com_xy() - self.scene.env_origins[:, :2]
        cell_m = self._geodesic_field_cell
        grid_xy = (local_xy - self._geodesic_field_origin) / cell_m.unsqueeze(-1)
        col_f, row_f = grid_xy.unbind(dim=-1)
        height = self._geodesic_field_shape[:, 0]
        width = self._geodesic_field_shape[:, 1]
        col = torch.minimum(
            torch.clamp_min(torch.round(col_f).long(), 0), width - 1
        )
        row = torch.minimum(
            torch.clamp_min(torch.round(row_f).long(), 0), height - 1
        )
        waypoint_local = self._geodesic_field_origin + cell_m.unsqueeze(-1) * (
            torch.stack((col, row), dim=-1).float()
        )
        lookahead_m = float(self.cfg.waypoint_lookahead_m)
        budget = torch.clamp_min(
            lookahead_m - torch.norm(waypoint_local - local_xy, dim=-1), 0.0
        )
        route_target = remaining_geodesic >= lookahead_m
        active = route_target & (budget > 1.0e-9)
        fallback_to_goal = torch.zeros_like(active)
        env_index = torch.arange(self.num_envs, device=self.device)

        # Cardinal hops are the shortest, so this bound also covers diagonal
        # routes and the sub-cell attachment from the continuous boat pose.
        max_hops = int(math.ceil(lookahead_m / GRID_CELL_M)) + 1
        for _ in range(max_hops):
            direction = self._geodesic_descent_directions[
                env_index, row, col
            ].long()
            hop_m = cell_m * torch.norm(direction.float(), dim=-1)
            stuck = active & (hop_m <= 0.0)
            fallback_to_goal |= stuck
            moving = active & ~stuck
            waypoint_local += (
                cell_m.unsqueeze(-1)
                * torch.stack((direction[:, 1], direction[:, 0]), dim=-1)
                * moving.unsqueeze(-1)
            )
            budget = torch.where(moving, budget - hop_m, budget)
            row += direction[:, 0] * moving
            col += direction[:, 1] * moving
            active = moving & (budget > 1.0e-9)

        waypoint_world = waypoint_local + self.scene.env_origins[:, :2]
        use_goal = ~route_target | fallback_to_goal
        return torch.where(use_goal.unsqueeze(-1), self.target_pos, waypoint_world)

    def _get_rewards(self) -> torch.Tensor:
        distance = self._horizontal_distance()
        progress_distance = distance
        if self.cfg.reward_progress_geodesic:
            progress_distance = self._geodesic_progress_distance(distance)
        potential = self._potential(progress_distance)
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
        self._ep_contact_steps += contact_now.float()
        self._ep_contact_run_steps.copy_(
            torch.where(
                contact_now,
                self._ep_contact_run_steps + 1.0,
                torch.zeros_like(self._ep_contact_run_steps),
            )
        )
        self._ep_contact_longest_steps.copy_(
            torch.maximum(
                self._ep_contact_longest_steps, self._ep_contact_run_steps
            )
        )
        self._ep_contact_depth_sum += torch.clamp(-clearance, min=0.0)

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

        # Open-water tax: charged only out in the empty water AROUND the
        # obstacle field, so the detour stops being the cheap option. Exempt
        # near the goal -- its own 5 m clear disk is open water by
        # construction, and taxing it would penalise arriving.
        open_water_cost = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.reward_open_water_scale > 0.0:
            goal_distance = torch.norm(self.target_pos - self._com_xy(), dim=-1)
            in_open = (
                (clearance > self.cfg.open_water_radius_m)
                & (goal_distance > self.cfg.open_water_goal_exempt_m)
                & ~self._reached_goal
            )
            open_water_cost = (
                self.cfg.reward_open_water_scale
                * self.control_step_s
                * in_open.float()
            )
            self._ep_open_water_cost += open_water_cost

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
            - open_water_cost
            - reverse_cost
            - self.cfg.reward_contact_entry_penalty * contact_entry.float()
            - self.cfg.reward_contact_dwell_penalty
            * (
                (
                    self._ep_contact_run_steps
                    * self.control_step_s
                    / self.cfg.contact_dwell_tau_s
                ).square()
                * contact_now.float()
                if self.cfg.reward_contact_dwell_quadratic
                else contact_now.float()
            )
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
        # Certification success remains collision-free for every variant.
        # ``clean_goal_gate`` changes only whether a contacted arrival earns
        # the reward; it must never change episode_success or success timing.
        success_now = reached_now & ~self._contact_before_goal
        if self.cfg.clean_goal_gate:
            rewarded_goal_entry_now = success_now
        else:
            rewarded_goal_entry_now = reached_now
        self._clean_goal_entry_this_step.copy_(rewarded_goal_entry_now)
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(success_now, elapsed_s, self._first_success_time_s)
        )
        self._success |= success_now
        self._reached_goal |= reached_now

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Reaching the single goal always terminates. Contact termination is a
        # variant switch; the contact ledger and clean-success metric above
        # are recorded identically even when the episode is allowed to carry
        # on after contact.
        terminated = (
            reached_now | contact_now
            if self.cfg.contact_terminates
            else reached_now
        )
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
            self.episode_contact_steps[completed_ids] = self._ep_contact_steps[
                completed_ids
            ]
            self.episode_contact_longest_steps[completed_ids] = (
                self._ep_contact_longest_steps[completed_ids]
            )
            self.episode_contact_depth_sum[completed_ids] = (
                self._ep_contact_depth_sum[completed_ids]
            )
            self.route_geodesic_length[completed_ids] = (
                self._route_geodesic_length[completed_ids]
            )
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
            self.extras["log"]["Episode/contact_steps"] = (
                self._ep_contact_steps[completed_ids].mean()
            )
            self.extras["log"]["Episode/contact_longest_s"] = (
                self._ep_contact_longest_steps[completed_ids].mean()
                * self.control_step_s
            )
            contact_steps = self._ep_contact_steps[completed_ids]
            contact_depth_mean = torch.where(
                contact_steps > 0.0,
                self._ep_contact_depth_sum[completed_ids]
                / contact_steps.clamp_min(1.0),
                torch.zeros_like(contact_steps),
            )
            self.extras["log"]["Episode/contact_depth_mean_m"] = (
                contact_depth_mean.mean()
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

        # New episode for these envs: advance their per-env episode counter and
        # reseed their primitive streams. MUST precede every draw below.
        if self._scenario is not None:
            self._scenario.reset_idx(env_ids)

        num_resets = len(env_ids)
        reset_env_indices = env_ids.tolist()
        # Resolved values of the knobs that were actually drawn, for the
        # certificate stamp at the end of this method. A knob whose *_choices
        # tuple is empty draws nothing and contributes nothing.
        actuator_scenario: dict[str, dict[str, object]] = {}
        # Each knob owns its OWN stream, so enabling one knob cannot re-draw
        # another -- scripts/eval_imbalance.py sweeps these one at a time and a
        # shared stream would confound exactly the attribution it exists to
        # make. The five blocks below keep their original order, their original
        # ranges and their original uniform-over-choices distribution; only the
        # generator changed, and only when the env is seeded.
        if self.cfg.thrust_imbalance_choices:
            choice_indices = actuator_choice_indices(
                self._scenario,
                self._layout_rng,
                GROUP_THRUST_IMBALANCE,
                reset_env_indices,
                len(self.cfg.thrust_imbalance_choices),
            )
            choices = np.asarray(
                self.cfg.thrust_imbalance_choices, dtype=np.float32
            )
            drawn = choices[choice_indices]
            self._thrust_imbalance_per_env[env_ids] = torch.as_tensor(
                drawn, device=self.device
            )
            actuator_scenario.update(
                actuator_scenario_entry(GROUP_THRUST_IMBALANCE, drawn.tolist())
            )
        if self.cfg.mass_scale_choices:
            choice_indices = actuator_choice_indices(
                self._scenario,
                self._layout_rng,
                GROUP_MASS_SCALE,
                reset_env_indices,
                len(self.cfg.mass_scale_choices),
            )
            choices = np.asarray(self.cfg.mass_scale_choices, dtype=np.float32)
            drawn = choices[choice_indices]
            self._mass_scale_per_env[env_ids] = torch.as_tensor(
                drawn, device=self.device
            )
            actuator_scenario.update(
                actuator_scenario_entry(GROUP_MASS_SCALE, drawn.tolist())
            )
        if self.cfg.drag_scale_choices:
            choice_indices = actuator_choice_indices(
                self._scenario,
                self._layout_rng,
                GROUP_DRAG_SCALE,
                reset_env_indices,
                len(self.cfg.drag_scale_choices),
            )
            choices = np.asarray(self.cfg.drag_scale_choices, dtype=np.float32)
            drawn = choices[choice_indices]
            self._drag_scale_per_env[env_ids] = torch.as_tensor(
                drawn, device=self.device
            )
            actuator_scenario.update(
                actuator_scenario_entry(GROUP_DRAG_SCALE, drawn.tolist())
            )
        if self.cfg.thrust_cap_scale_choices:
            choice_indices = actuator_choice_indices(
                self._scenario,
                self._layout_rng,
                GROUP_THRUST_CAP_SCALE,
                reset_env_indices,
                len(self.cfg.thrust_cap_scale_choices),
            )
            choices = np.asarray(
                self.cfg.thrust_cap_scale_choices, dtype=np.float32
            )
            drawn = choices[choice_indices]
            self._thrust_cap_scale_per_env[env_ids] = torch.as_tensor(
                drawn, device=self.device
            )
            actuator_scenario.update(
                actuator_scenario_entry(GROUP_THRUST_CAP_SCALE, drawn.tolist())
            )
        if self.cfg.motor_tau_s_choices:
            choice_indices = actuator_choice_indices(
                self._scenario,
                self._layout_rng,
                GROUP_MOTOR_TAU_S,
                reset_env_indices,
                len(self.cfg.motor_tau_s_choices),
            )
            choices = np.asarray(self.cfg.motor_tau_s_choices, dtype=np.float32)
            drawn = choices[choice_indices]
            self._motor_tau_s_per_env[env_ids] = torch.as_tensor(
                drawn, device=self.device
            )
            actuator_scenario.update(
                actuator_scenario_entry(GROUP_MOTOR_TAU_S, drawn.tolist())
            )
        # Actuator memory is episode-local for both scalar and choice modes.
        self._applied_thrust_per_env[env_ids] = 0.0
        self._applied_yaw_per_env[env_ids] = 0.0
        max_obstacles = self.cfg.max_obstacles
        local_starts_np = np.zeros((num_resets, 2), dtype=np.float32)
        local_goals_np = np.zeros((num_resets, 2), dtype=np.float32)
        local_centers_np = np.zeros(
            (num_resets, max_obstacles, 2), dtype=np.float32
        )
        radii_np = np.zeros((num_resets, max_obstacles), dtype=np.float32)
        active_np = np.zeros((num_resets, max_obstacles), dtype=np.bool_)
        geodesic_np = np.zeros(num_resets, dtype=np.float32)
        d0_np = np.zeros(num_resets, dtype=np.float32)
        counts_np = np.zeros(num_resets, dtype=np.int64)
        if self.cfg.reward_progress_geodesic:
            geodesic_fields_np = np.full(
                (
                    num_resets,
                    _GEODESIC_FIELD_CELLS,
                    _GEODESIC_FIELD_CELLS,
                ),
                np.inf,
                dtype=np.float32,
            )
            geodesic_origins_np = np.zeros((num_resets, 2), dtype=np.float32)
            geodesic_shapes_np = np.zeros((num_resets, 2), dtype=np.int64)
            geodesic_max_np = np.zeros(num_resets, dtype=np.float32)
            if self.cfg.nav_targets_waypoint:
                geodesic_descent_np = np.zeros(
                    (
                        num_resets,
                        _GEODESIC_FIELD_CELLS,
                        _GEODESIC_FIELD_CELLS,
                        2,
                    ),
                    dtype=np.int8,
                )
        level = self.current_level
        # Global rotations, one per resetting env, kept for the scenario stamp.
        rotation_angles: list[float] = []
        for row in range(num_resets):
            # One generator per (env, episode) instead of one shared generator
            # advanced per reset. Every sampler below -- its rejection loop, its
            # attempt budget, its tier ladder, every range it samples -- is
            # untouched; only the Generator handed to it changes, and only when
            # the env is seeded. See episode_layout_rng.
            layout_rng = episode_layout_rng(
                self._scenario, self._layout_rng, reset_env_indices[row]
            )
            if self.cfg.layout_mode == "ring":
                # Ring siege: spawn encircled, exactly one tier-width gap.
                # Avoidance stops being optional -- escape requires threading.
                layout = sample_ring_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(40, self.cfg.layout_max_attempts),
                    neighbor_overlap_m=getattr(
                        self.cfg, "ring_neighbor_overlap_m", None
                    ),
                )
            elif self.cfg.layout_mode == "ring2":
                # Harder siege: two sealed, tier-width exits separated by at
                # least 90 degrees. Both have to be threaded in sequence.
                layout, _gaps = sample_double_ring_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(60, self.cfg.layout_max_attempts),
                )
            elif self.cfg.layout_mode == "fortress":
                # Ring fortress inverts the siege: the goal is at the ring's
                # center and the sampled spawn is outside its only opening.
                layout, _gap = sample_ring_fortress_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(60, self.cfg.layout_max_attempts),
                )
            elif self.cfg.layout_mode == "fortress2":
                # The sampled spawn is outside both misaligned sealed rings;
                # success therefore requires threading both openings inward.
                layout, _gaps = sample_double_ring_fortress_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(80, self.cfg.layout_max_attempts),
                )
            elif self.cfg.layout_mode == "bandfort":
                # Constructive double bands keep every cylinder edge gap at
                # the tier aperture; the sampler supplies the outside spawn.
                layout = sample_band_fortress_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(80, self.cfg.layout_max_attempts),
                    aperture_override_m=(
                        getattr(self.cfg, "fortress_aperture_override_m", 0.0)
                        or None
                    ),
                )
            elif self.cfg.layout_mode == "basin":
                # Control for the crossing task: same walls, no bulkhead. The
                # crossing basin changes two things at once and the champion
                # dies on the walls before reaching the gate, so this isolates
                # which of the two the failure belongs to.
                layout = sample_open_basin_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(60, self.cfg.layout_max_attempts),
                )
            elif self.cfg.layout_mode == "iceberg":
                # Open-water long-range avoidance: the sampler supplies the
                # spawn because its direct route is constructively blocked.
                layout = sample_iceberg_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(60, self.cfg.layout_max_attempts),
                )
            elif self.cfg.layout_mode == "forced":
                # Closed basin split by a gated bulkhead. Admission proves the
                # goal is unreachable at the true half-beam once the gate is
                # sealed, so unlike scatter there is no detour to prefer.
                layout, _gate = sample_forced_crossing_layout(
                    level,
                    rng=layout_rng,
                    max_attempts=max(60, self.cfg.layout_max_attempts),
                    gate_width_override_m=(
                        getattr(self.cfg, "gate_width_override_m", 0.0) or None
                    ),
                )
            else:
                if getattr(self.cfg, "suite_s_class", ""):
                    # Suite S: a frozen, checksum-verified structural layout
                    # replaces the scatter draw. Every pre-existing cfg lacks
                    # the field, so this getattr guard keeps their reset path
                    # byte-identical (see _suite_s_layout_for below).
                    layout = self._suite_s_layout_for(int(env_ids[row]))
                else:
                    layout = sample_layout(
                        level,
                        rng=layout_rng,
                        max_attempts=self.cfg.layout_max_attempts,
                    )
            # Uniform on [0, 2*pi) exactly as before; only the stream changed.
            # It is its OWN group, not a further draw off the layout stream, so
            # the angle cannot shift with how many candidates the rejection
            # sampler burned -- which also means the Suite S branch above, which
            # consumes no layout draws at all, gets its rotation from the same
            # key algebra as every other layout_mode.
            goal_angle = layout_rotation_angle(
                self._scenario, self._layout_rng, reset_env_indices[row]
            )
            rotation_angles.append(goal_angle)
            cosine, sine = math.cos(goal_angle), math.sin(goal_angle)
            rotation = np.array(((cosine, -sine), (sine, cosine)))
            rotated_start = layout.start @ rotation.T
            rotated_goal = layout.goal @ rotation.T
            rotated_centers = layout.centers @ rotation.T
            count = layout.obstacle_count
            local_starts_np[row] = rotated_start
            local_goals_np[row] = rotated_goal
            local_centers_np[row, :count] = rotated_centers
            radii_np[row, :count] = layout.radii
            active_np[row, :count] = True
            if self.cfg.reward_progress_geodesic:
                inflated = layout.radii + HALF_BEAM_M
                bounds = (
                    min(
                        rotated_start[0],
                        rotated_goal[0],
                        np.min(rotated_centers[:, 0] - inflated),
                    )
                    - _GEODESIC_FIELD_PADDING_M,
                    max(
                        rotated_start[0],
                        rotated_goal[0],
                        np.max(rotated_centers[:, 0] + inflated),
                    )
                    + _GEODESIC_FIELD_PADDING_M,
                    min(
                        rotated_start[1],
                        rotated_goal[1],
                        np.min(rotated_centers[:, 1] - inflated),
                    )
                    - _GEODESIC_FIELD_PADDING_M,
                    max(
                        rotated_start[1],
                        rotated_goal[1],
                        np.max(rotated_centers[:, 1] + inflated),
                    )
                    + _GEODESIC_FIELD_PADDING_M,
                )
                field = geodesic_distance_field(
                    rotated_centers,
                    layout.radii,
                    rotated_goal,
                    bounds,
                    cell_m=GRID_CELL_M,
                    half_beam_m=HALF_BEAM_M,
                )
                height, width = field.shape
                if (
                    height > _GEODESIC_FIELD_CELLS
                    or width > _GEODESIC_FIELD_CELLS
                ):
                    raise RuntimeError(
                        f"geodesic field {field.shape} exceeds packed "
                        f"{_GEODESIC_FIELD_CELLS}x{_GEODESIC_FIELD_CELLS} arena"
                    )
                origin = GRID_CELL_M * np.floor(
                    np.asarray((bounds[0], bounds[2])) / GRID_CELL_M
                )
                finite = np.isfinite(field)
                if not np.any(finite):
                    raise RuntimeError("geodesic field has no goal-reachable cells")
                spawn_geodesic = _bilinear_numpy(field, rotated_start, origin)
                if not math.isfinite(spawn_geodesic):
                    raise RuntimeError("spawn is not finite in geodesic field")
                geodesic_fields_np[row, :height, :width] = field
                geodesic_origins_np[row] = origin
                geodesic_shapes_np[row] = (height, width)
                geodesic_max_np[row] = float(np.max(field[finite]))
                if self.cfg.nav_targets_waypoint:
                    geodesic_descent_np[row, :height, :width] = (
                        geodesic_descent_directions(field)
                    )
                # The progress denominator and the public route diagnostic are
                # exactly the same bilinear spawn-to-goal field distance.
                geodesic_np[row] = spawn_geodesic
                d0_np[row] = spawn_geodesic
            else:
                geodesic_np[row] = layout.geodesic_length
                d0_np[row] = float(np.linalg.norm(layout.goal - layout.start))
            counts_np[row] = count

        local_starts = torch.as_tensor(
            local_starts_np, device=self.device, dtype=torch.float32
        )
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
        if self.cfg.reward_progress_geodesic:
            self._geodesic_fields[env_ids] = torch.as_tensor(
                geodesic_fields_np, device=self.device
            )
            self._geodesic_field_origin[env_ids] = torch.as_tensor(
                geodesic_origins_np, device=self.device
            )
            self._geodesic_field_shape[env_ids] = torch.as_tensor(
                geodesic_shapes_np, device=self.device
            )
            self._geodesic_field_max[env_ids] = torch.as_tensor(
                geodesic_max_np, device=self.device
            )
            if self.cfg.nav_targets_waypoint:
                self._geodesic_descent_directions[env_ids] = torch.as_tensor(
                    geodesic_descent_np, device=self.device
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
        com_target_xy = env_origins_xy + local_starts
        # The sampler's start is a planar COM statement. Keep the USD-authored
        # default vertical spawn used by the calm-water task.
        root_state[:, :2] = com_target_xy - com_offset_world[:, :2]
        root_state[:, 3:7] = spawn_quats
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self.path_length[env_ids] = 0.0
        self._previous_xy[env_ids] = com_target_xy
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
        if self._obs_degrader is not None:
            # New episode: bump the (env, episode) RNG stream, resample the
            # per-episode bias, and clear delay/dropout buffers. The clean
            # initial frame is captured on the next _get_observations call.
            self._obs_degrader.reset(env_ids)
        self._first_success_time_s[env_ids] = torch.nan
        self._episode_finished[env_ids] = False
        initial_clearance = analytic_min_clearance(
            com_target_xy,
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
        self._ep_open_water_cost[env_ids] = 0.0
        self._ep_contact_entries[env_ids] = 0.0
        self._ep_contact_steps[env_ids] = 0.0
        self._ep_contact_run_steps[env_ids] = 0.0
        self._ep_contact_longest_steps[env_ids] = 0.0
        self._ep_contact_depth_sum[env_ids] = 0.0
        self._ep_goal_bonus[env_ids] = 0.0
        self._ep_reverse_cost[env_ids] = 0.0

        # Resolved scenario for the episode that starts now. Values only, never
        # the key: a key-salted hash would make two eval seeds differ by
        # construction and could never expose two runs that genuinely drew the
        # same paper (tasks/_shared/scenario_draws.py:49-57).
        #
        # The layout entries are the ACCEPTED, already-rotated sample rather
        # than the raw uniforms the rejection sampler burned to reach it: like
        # path_hazard (path_hazard_env.py:1114-1119) there is no shorter raw
        # form, because the sampler draws an unbounded number of candidates and
        # returns the first feasible one. route_geodesic_m and d0_m are carried
        # explicitly because they are the two per-episode fields the certificate
        # already writes that depend on the layout ALONE, and therefore the two
        # a cross-controller audit compares
        # (scripts/check_scenario_independence.py:175).
        for env_index, resolved, hashes in stamp_scenario(
            env_ids,
            {
                GROUP_LAYOUT: {
                    "start_m": local_starts,
                    "goal_m": local_goals,
                    "obstacle_centers_m": local_centers,
                    "obstacle_radii_m": radii,
                    "obstacle_count": self.obstacle_count[env_ids],
                    "route_geodesic_m": geodesic_np.tolist(),
                    "d0_m": d0_np.tolist(),
                },
                GROUP_ROTATION: {"angle_rad": rotation_angles},
                **actuator_scenario,
            },
        ):
            self._scenario_params[env_index] = resolved
            self._scenario_hashes[env_index] = hashes

        self._update_render_only_hazard_markers(
            env_ids, local_goals, local_centers, radii, active
        )

    # --- Suite S: frozen structural-generalization layout injection ---------
    def _suite_s_layout_for(self, env_index: int):
        """Frozen Suite S layout for one resetting env.

        Reached only when the cfg carries a non-empty ``suite_s_class`` (the
        HazardSuiteSEnvCfg family). The ten public layouts of that class are
        loaded once per process from the committed JSON assets --
        ``load_public_layouts`` re-verifies each file's SHA-256 checksum, so
        a corrupted or edited asset fails loudly here rather than silently
        certifying on the wrong geometry. The returned ``SuiteSLayout`` then
        flows through the standard scatter placement code (random global
        rotation, obstacle buffers, ``geodesic_np`` latch), which is how
        ``route_geodesic_length`` picks up the asset's audited
        ``geodesic_length_m`` and USV-10K scoring gets basis=geo for free.

        Layout choice is ``rotation_index``: a pure function of the env index
        and that env's own completed-reset count, deliberately independent of
        the shared layout RNG and of other envs' reset timing.  That is already
        the controller-independence property the scenario protocol gives every
        other primitive, by its own mechanism (``_suite_s_episode_counter``
        counts this env's resets exactly as ``ScenarioRNG``'s per-env counter
        does), so this branch is left on it rather than re-keyed -- re-keying
        would change which frozen asset each slot draws and retire every Suite S
        number for no gain.  The global rotation applied to the returned layout
        does move onto the protocol, at the ``rotation`` group.
        """
        from .suite_s_layouts import load_public_layouts, rotation_index

        if not hasattr(self, "_suite_s_layouts"):
            layouts = load_public_layouts(
                str(self.cfg.suite_s_class),
                level=int(getattr(self.cfg, "suite_s_level", 2)),
                asset_dir=str(getattr(self.cfg, "suite_s_asset_dir", "")),
            )
            worst = max(layout.obstacle_count for layout in layouts)
            if worst > int(self.cfg.max_obstacles):
                raise ValueError(
                    f"Suite S class {self.cfg.suite_s_class!r} needs {worst} "
                    f"obstacle slots but max_obstacles={self.cfg.max_obstacles};"
                    " raise the cap in the Suite S cfg, never in a base cfg"
                )
            self._suite_s_layouts = layouts
            self._suite_s_episode_counter = np.zeros(
                self.num_envs, dtype=np.int64
            )
        index = rotation_index(
            env_index,
            int(self._suite_s_episode_counter[env_index]),
            rotation=bool(getattr(self.cfg, "suite_s_index_rotation", True)),
            layout_count=len(self._suite_s_layouts),
        )
        self._suite_s_episode_counter[env_index] += 1
        return self._suite_s_layouts[index]
