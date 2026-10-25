# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for Task A: Static Hazard Navigation."""

from __future__ import annotations

from .._shared.termination_reason import publish_termination_reason

from collections.abc import Sequence
import math
import os

import numpy as np
from .._shared.vehicles import physics_version
from .._shared.physics_v2 import add_implicit_hull_damping, WaveClock

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.conditions import build_conditions
from .._shared.journal_env_mixin import JournalEnvMixin, device_copies, host_copies
from .._shared import d7_point_return as _d7pr
from .._shared.obs_superset import (
    SLICES,
    SPEED_SCALE_MPS,
    SUPERSET_DIM,
    SUPERSET_DIM_V2,
)
from .._shared.restoring import restoring_torque_body
from .._shared.scenario_draws import make_scenario_rng, stamp_scenario
from .._shared.scenario_draws_hazard import (
    GROUP_APPEARANCE,
    GROUP_DRAG_SCALE,
    GROUP_LAYOUT,
    GROUP_MASS_SCALE,
    GROUP_MOTOR_TAU_S,
    GROUP_PAYLOAD_MASS_KG,
    GROUP_ROTATION,
    GROUP_THRUST_CAP_SCALE,
    GROUP_THRUST_IMBALANCE,
    actuator_choice_indices,
    actuator_scenario_entry,
    appearance_stream,
    episode_layout_rng,
    layout_rotation_angle,
)
from .._shared.vehicles import get_vehicle
from ..docking.curriculum import DockingCurriculum
from .hazard_geometry import (
    GRID_CELL_M,
    HALF_BEAM_M,
    COLLISION_MARGIN_M,
    KIND_ORBIT,
    KIND_TRANSLATE,
    analytic_min_clearance,
    geodesic_descent_directions,
    geodesic_distance_field,
    moving_basin_positions,
    plan_obstacle_appearance,
    rotate_moving_params,
    ray_circle_ranges,
    route_direction_xy,
    sample_band_fortress_layout,
    sample_double_ring_fortress_layout,
    sample_double_ring_layout,
    sample_forced_crossing_layout,
    sample_iceberg_layout,
    sample_layout,
    sample_moving_basin_layout,
    sample_open_basin_layout,
    sample_ring_fortress_layout,
    sample_ring_layout,
)
from .hazard_nav_env_cfg import HazardNavEnvCfg
from . import journal_hazard_nav as _journal


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


class HazardNavEnv(JournalEnvMixin, DirectRLEnv):
    """Fixed-horizon navigation through static analytic cylinder hazards.

    Journal cfgs (``journal_contract_version == 2``, E5 templates) run the
    seven HazardNav journal tasks through the F0 JournalEnvMixin; the journal
    methods (``_journal_*`` below) are bound in ``_journal_init`` so every
    legacy method source stays byte-identical (CONTEXT_R4 D15).
    """

    cfg: HazardNavEnvCfg
    journal_family = "hazard_nav"
    # G3 appearing obstacle: the journal tasks of this family wired for it (G20 rollout: all seven;
    # the slot, publish, contact and ray paths below are task-independent)
    journal_appearing_tasks = ("reach", "scatter", "ring", "fortress", "forced_crossing", "iceberg", "moving_basin")

    def __init__(self, cfg: HazardNavEnvCfg, render_mode: str | None = None, **kwargs):
        if getattr(cfg, "journal_contract_version", 1) == 2:
            # v2 registry vessel (ANCHORS_E5 C6): resolved BEFORE super().__init__
            # builds the asset; the legacy get_vehicle registry has no v2 names.
            self.vehicle_spec = _journal.journal_scene_spec(cfg)
        else:
            self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        physics_version(self.physics_cfg)  # Reject unsupported versions before simulation.
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
        self._payload_mass_kg_per_env = torch.zeros(
            self.num_envs, device=self.device
        )
        self._applied_thrust_per_env = torch.zeros(
            self.num_envs, device=self.device
        )
        self._applied_yaw_per_env = torch.zeros(
            self.num_envs, device=self.device
        )

        max_obstacles = int(getattr(self.cfg, "obstacle_slot_count", self.cfg.max_obstacles))
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

        # P5 moving basin. The motion law is mirrored per env so the per-step
        # update is one vectorised tensor write with no Python loop. Allocated
        # ONLY for the moving id, so every certified cfg keeps exactly the
        # buffer set it had and its step path is untouched.
        self._moving_enabled = (
            getattr(self.cfg, "layout_mode", "") == "moving_basin"
        )
        if self._moving_enabled:
            self._moving_kind = torch.zeros(
                (self.num_envs, max_obstacles),
                dtype=torch.long,
                device=self.device,
            )
            self._moving_params = torch.zeros(
                (self.num_envs, max_obstacles, 7), device=self.device
            )
            # The ROTATED local centres as sampled. Static cylinders keep these
            # forever; moving ones are recomputed from the params each step, so
            # this array must not be the same storage the update writes into.
            self._moving_base = torch.zeros(
                (self.num_envs, max_obstacles, 2), device=self.device
            )

        # --- Mid-episode obstacle appearance (sudden terrain change) --------
        # The buffers exist on every variant so the record fields are always
        # discoverable, but with appear_count == 0 (every certified id)
        # nothing below is read or written after this constructor:
        # _reset_appearance returns an empty stamp before touching a buffer
        # or a stream, and _maybe_appear_obstacles returns on its first line.
        # Both inert paths are tripwire-asserted in
        # tasks/hazard_nav/test_mid_episode_appearance.py.
        self._appear_count = int(getattr(self.cfg, "appear_count", 0) or 0)
        self._appear_rngs: list = [None] * self.num_envs
        self._appear_time_s = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._appear_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._appear_happened = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._appear_skipped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._appear_spawned = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._appear_azimuth_rad = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        # How much open water the appeared cylinder landed in: the free gap
        # from its surface to the nearest PRE-EXISTING active cylinder's
        # surface. Added 2026-09-01 to settle a question three separate
        # readings failed to answer from outcomes alone, namely why the travel
        # aligned deficit persists at long range on the crossing family. If
        # those placements land against the bulkhead this number will be small
        # there and large on an open scatter field; if it does not differ, the
        # structure explanation is dead and something else is going on. It is
        # rotation invariant, which raw coordinates are not, because every
        # layout is turned by a per-episode angle.
        self._appear_clearance_m = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self._appear_distance_m = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        # Last COMPLETED episode's appearance record, latched beside
        # episode_min_clearance so the evaluator reads the finished episode's
        # appearance next to the finished episode's metrics.
        self.episode_appear_time_s = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.episode_appear_happened = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_appear_skipped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.episode_appear_spawned = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.episode_appear_azimuth_rad = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.episode_appear_clearance_m = torch.full(
            (self.num_envs,), torch.nan, device=self.device
        )
        self.episode_appear_distance_m = torch.full(
            (self.num_envs,), torch.nan, device=self.device
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

        # Opt-in journal suites. None bypasses every hook for legacy ids.
        if getattr(self.cfg, "journal_contract_version", 1) == 2:
            # Journal contract v2: the F0 mixin builds the hydro_v2 runtime, the
            # schema-3 condition layer and the D7 bridge (journal_setup).
            self._journal_init()
        else:
            self._conditions = build_conditions(
                self.cfg, self.num_envs, self.device, family="hazard_nav"
            )
            self.episode_conditions = [{} for _ in range(self.num_envs)]

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
                for obstacle_index in range(
                    getattr(self.cfg, "obstacle_slot_count", self.cfg.max_obstacles)
                ):
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
                    for obstacle_index in range(
                        getattr(self.cfg, "obstacle_slot_count", self.cfg.max_obstacles)
                    ):
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
        if self._conditions is not None:
            self.actions = self._conditions.action_in(self.actions)
            self._conditions.reset_runtime(progress=None)

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

        if self._conditions is not None:
            thrust, yaw_command = self._conditions.command(
                thrust, yaw_command, self.cfg.sim.dt, None
            )
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
        # D1 merge (C2 x C4): v1 keeps the explicit legacy drag, including the
        # C4 condition drag multiplier. v2 applies NO explicit hull drag here;
        # add_implicit_hull_damping below owns every hull damping term and
        # takes the condition drag multiplier from hull_damping_scales().
        if getattr(self.physics_cfg, "physics_version", 1) == 1:
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
                if self._conditions is not None:
                    planar_drag = self._conditions.scale_drag(planar_drag)
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
                if self._conditions is not None:
                    drag_body[:, :2] = self._conditions.scale_drag(drag_body[:, :2])
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
            if self._conditions is not None:
                yaw_drag = self._conditions.scale_drag(yaw_drag)
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

        if self._sea is not None:
            # Same world-frame entry point as the buoyancy/drag wrench, before
            # the shared world-to-body rotation and before the mass-scale
            # division (a heavier hull accelerates less under the same sea,
            # exactly as it does under the same drag).
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

        if self._conditions is not None:
            # Physical bow azimuth (registry bow axis via forward_vec) for the
            # wind suite; the hull COM velocity is the air-relative input.
            # Added before payload/mass scaling like every other world load.
            bow_world = math_utils.quat_apply(quat, self.forward_vec)
            condition_force, condition_torque = self._conditions.environmental_wrench(
                self.robot.data.root_com_pos_w[:, :2],
                self.robot.data.root_com_vel_w[:, :2],
                self.cfg.sim.dt,
                orientation_wxyz=quat,
                angular_velocity_w=angular_velocity,
                heading_rad=torch.atan2(bow_world[:, 1], bow_world[:, 0]),
            )
            force_world += condition_force
            torque_world += condition_torque
        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_world)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_world)
        if self.cfg.payload_mass_kg_choices or self.cfg.payload_mass_kg != 0.0:
            # Suite D axis: deck payload -- a point mass mp at body offset
            # (dx, dy) from the authored COM. Model and non-redundancy
            # argument live at the payload_* fields in hazard_nav_env_cfg.py.
            # Three exact planar effects, in this order: the COM-shift arm
            # moment on the TOTAL planar force (thrust + drag + rotated world
            # wrench -- all applied at the hull reference point), then the
            # linear-mass ratio on every force component, then the yaw-inertia
            # ratio on the z torque alone (planar point mass: roll/pitch
            # torques stay). Ratios are taken against the mass_scale-d hull so
            # the two axes compose exactly: after the mass_scale block below
            # also divides by s, the total linear divisor is (s*m + mp)/m and
            # the total yaw divisor is (s*Iz + mu_s*|d|^2)/Iz with
            # mu_s = s*m*mp/(s*m + mp). Drawn-zero envs keep the untouched
            # tensors through torch.where; the all-scalar-zero default never
            # enters this block at all.
            if self.cfg.payload_mass_kg_choices:
                payload_mass = self._payload_mass_kg_per_env
            else:
                payload_mass = torch.full(
                    (self.num_envs,),
                    float(self.cfg.payload_mass_kg),
                    device=self.device,
                )
            if self.cfg.mass_scale_choices:
                hull_scale = self._mass_scale_per_env
            else:
                hull_scale = torch.full(
                    (self.num_envs,),
                    float(self.cfg.mass_scale),
                    device=self.device,
                )
            hull_mass = hull_scale * float(self.cfg.payload_ref_mass_kg)
            hull_izz = hull_scale * float(self.cfg.payload_ref_izz_kgm2)
            total_mass = hull_mass + payload_mass
            offset_x = float(self.cfg.payload_offset_x_m)
            offset_y = float(self.cfg.payload_offset_y_m)
            delta_x = payload_mass * offset_x / total_mass
            delta_y = payload_mass * offset_y / total_mass
            reduced_mass = hull_mass * payload_mass / total_mass
            offset_sq = offset_x * offset_x + offset_y * offset_y
            mass_ratio = total_mass / hull_mass
            izz_ratio = (hull_izz + reduced_mass * offset_sq) / hull_izz
            arm_torque = delta_y * forces[:, 0, 0] - delta_x * forces[:, 0, 1]
            loaded_torque_z = (torques[:, 0, 2] + arm_torque) / izz_ratio
            loaded_forces = forces / mass_ratio.unsqueeze(-1).unsqueeze(-1)
            identity = payload_mass == 0.0
            torques[:, 0, 2] = torch.where(
                identity, torques[:, 0, 2], loaded_torque_z
            )
            forces = torch.where(
                identity.unsqueeze(-1).unsqueeze(-1), forces, loaded_forces
            )
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
        # D1 merge order: condition mass/payload scaling first (C4), then the
        # v2 implicit hull damping (C2) on that final scaled external wrench.
        # The condition layer is the single drag/mass source in v2.
        if self._conditions is not None:
            forces, torques = self._conditions.dynamics(forces, torques)
        if getattr(self.physics_cfg, "physics_version", 1) == 2:
            add_implicit_hull_damping(
                self, forces, torques, quat,
                scales=(None if self._conditions is None
                        else self._conditions.hull_damping_scales()),
            )
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

    def _planar_velocity_2d(self) -> torch.Tensor:
        """World-frame planar velocity, the DIRECTION OF TRAVEL.

        Not the heading: under thrust imbalance, sideslip or a turn the hull
        points somewhere other than where it is going, and the mid-episode
        appearance has to be placed where the boat is actually heading.  Same
        six-or-three shape guard the physics step uses on root_com_vel_w.
        """
        velocity_world = self.robot.data.root_com_vel_w
        if velocity_world.shape[-1] == 6:
            return velocity_world[:, :2]
        return velocity_world[:, :2]

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
        if self._conditions is not None and any(
            self._conditions.wants(group) for group in ("pos", "vel", "ray")
        ):
            degrader = self._conditions
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

    def _reset_appearance(
        self, env_ids: torch.Tensor, reset_env_indices: list
    ) -> dict:
        """Draw each resetting env's appearance time off its own stream.

        Returns the ``stamp_scenario`` block for the ``appearance`` group, or
        an empty dict when the axis is off (``appear_count == 0``), in which
        case NOTHING is read or written -- no buffer, no RNG draw -- so every
        certified id keeps reset-path bit-identity and its scenario hash keys
        stay exactly the set it certified with.

        The extra obstacle slots need no explicit writes here: the layout
        block above already zeroed every slot past ``obstacle_count`` for the
        resetting envs, which IS the reservation -- present in the buffers the
        rays and the contact predicate read every step, but inactive.

        The drawn generator is KEPT (``self._appear_rngs``) because the
        azimuth-candidate permutation is drawn from the same per-(env,
        episode) stream at the moment of appearance; see appearance_stream.
        """
        if self._appear_count <= 0:
            return {}
        # One helper call per env: calling appearance_stream twice for the
        # same (env, episode) would restart the keyed stream, so the kept
        # generator must be the SAME object the time was drawn from.
        rngs = []
        times = []
        for env_index in reset_env_indices:
            rng = appearance_stream(self._scenario, self._layout_rng, env_index)
            times.append(
                float(
                    rng.uniform(
                        float(self.cfg.appear_time_min_s),
                        float(self.cfg.appear_time_max_s),
                    )
                )
            )
            rngs.append(rng)
        for row, env_index in enumerate(reset_env_indices):
            self._appear_rngs[int(env_index)] = rngs[row]
        self._appear_time_s[env_ids] = torch.as_tensor(
            times, dtype=torch.float32, device=self.device
        )
        self._appear_pending[env_ids] = True
        self._appear_happened[env_ids] = False
        self._appear_skipped[env_ids] = False
        self._appear_spawned[env_ids] = 0
        self._appear_azimuth_rad[env_ids] = torch.nan
        self._appear_clearance_m[env_ids] = torch.nan
        self._appear_distance_m[env_ids] = torch.nan
        return {GROUP_APPEARANCE: {"time_s": times}}

    def _update_moving_obstacles(self) -> None:
        """Rewrite ``obstacle_centers`` for the P5 moving basin, once per step.

        Both consumers of obstacle geometry read the buffers analytically every
        step -- ``ray_circle_ranges`` for the 36-ray sensor and
        ``analytic_min_clearance`` for contact -- and neither queries the
        physics engine.  So "moving" is one vectorised tensor write: no new
        collision body, no new asset, no PhysX change.  The only thing left out
        is the visual prim, and certification evaluation is headless.

        Static cylinders keep the centres they were sampled with; only the
        rows and the orbiter are recomputed, from a pure function of episode
        time, which is what makes an episode reproducible from its scenario
        draw alone.
        """
        elapsed = self.episode_length_buf.float() * self.control_step_s
        kind = self._moving_kind
        params = self._moving_params
        local = self._moving_base.clone()

        translate = kind == KIND_TRANSLATE
        if bool(translate.any()):
            base = params[..., 0:2]
            direction = params[..., 2:4]
            phase = params[..., 4]
            speed = params[..., 5]
            span = params[..., 6].clamp_min(1.0e-9)
            # Triangle wave: fold a sawtooth of period 2*span back on itself so
            # the cylinder reverses at each end. A wrapping saw would teleport
            # it and break contact and clearance continuity mid episode.
            raw = torch.remainder(speed * elapsed[:, None] + phase, 2.0 * span)
            offset = torch.where(raw <= span, raw, 2.0 * span - raw) - 0.5 * span
            local = torch.where(
                translate.unsqueeze(-1), base + direction * offset.unsqueeze(-1), local
            )

        orbit = kind == KIND_ORBIT
        if bool(orbit.any()):
            radius = params[..., 2]
            theta = params[..., 3] * elapsed[:, None] + params[..., 4]
            circle = torch.stack(
                (
                    params[..., 0] + radius * torch.cos(theta),
                    params[..., 1] + radius * torch.sin(theta),
                ),
                dim=-1,
            )
            local = torch.where(orbit.unsqueeze(-1), circle, local)

        self.obstacle_centers.copy_(
            self.scene.env_origins[:, None, :2] + local
        )
        self._refresh_moving_markers(local)

    def _refresh_moving_markers(self, local: torch.Tensor) -> None:
        """Walk the DISPLAY prims of the moving cylinders to where they now are.

        The analytic tensors are authoritative and are what every measurement
        reads, so a certification run does not need this at all.  A recording
        does: ``_update_render_only_hazard_markers`` runs once per RESET, so
        without this the visual cylinders would sit frozen at their t = 0 poses
        while the boat dodges obstacles that, on screen, are not there.  That
        would make every demo of this task quietly wrong.

        Cost when nothing is being drawn is one truthiness test: headless runs
        never build the translate ops, so the guard returns immediately and the
        certified step path is unchanged.  When prims DO exist only the moving
        cylinders are touched, never the static walls, which is most of them.
        """
        if not self._obstacle_translate_ops:
            return
        moving = (self._moving_kind != 0).cpu().numpy()
        if not moving.any():
            return
        # Imported here, not at module scope, for the same reason every other
        # USD touch in this file does it: pxr is only importable once Isaac has
        # booted, and a headless certification run must never reach this line.
        from pxr import Gf

        centers = local.detach().cpu().numpy()
        for env_index, ops in enumerate(self._obstacle_translate_ops):
            if env_index >= moving.shape[0]:
                break
            for slot in np.nonzero(moving[env_index])[0]:
                if slot >= len(ops):
                    break
                ops[slot].Set(
                    Gf.Vec3d(
                        float(centers[env_index, slot, 0]),
                        float(centers[env_index, slot, 1]),
                        0.0,
                    )
                )

    def _maybe_appear_obstacles(self) -> None:
        """Flip the reserved extra cylinders active once the clock passes t0.

        Runs first in ``_get_dones`` -- the head of the post-physics
        callback chain (dones -> rewards -> obs all read the same pose) --
        so the appeared cylinder is part of THIS step's contact predicate,
        reward field and observation rays alike, and the ray the azimuth is
        aligned to is exactly one of the rays the policy sees this step.

        The decision itself (fairness clamp to [9.2, 25] m from the CURRENT
        boat position, ray-aligned azimuth, sampler-grade BFS admission over
        the post-appearance field, bounded azimuth resampling) lives in
        ``hazard_geometry.plan_obstacle_appearance``; this method only
        executes an accepted plan or records the skip.  With
        ``appear_count == 0`` it returns before reading anything else.
        """
        if self._appear_count <= 0:
            return
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        due = self._appear_pending & (elapsed_s >= self._appear_time_s)
        if not bool(due.any()):
            return
        boat_xy = self._com_xy()
        forward = self._forward_2d()
        velocity_2d = (
            self._planar_velocity_2d()
            if bool(getattr(self.cfg, "appear_velocity_aligned", False))
            else None
        )
        max_obstacles = int(getattr(self.cfg, "obstacle_slot_count", self.cfg.max_obstacles))
        for env_index in torch.nonzero(due, as_tuple=False).squeeze(-1).tolist():
            self._appear_pending[env_index] = False
            origin = self.scene.env_origins[env_index, :2]
            boat_local = (
                (boat_xy[env_index] - origin)
                .detach().cpu().numpy().astype(np.float64)
            )
            forward_local = (
                forward[env_index].detach().cpu().numpy().astype(np.float64)
            )
            goal_local = (
                (self.target_pos[env_index] - origin)
                .detach().cpu().numpy().astype(np.float64)
            )
            active = self.obstacle_active[env_index]
            field_centers = (
                (self.obstacle_centers[env_index][active] - origin)
                .detach().cpu().numpy().astype(np.float64)
            )
            field_radii = (
                self.obstacle_radii[env_index][active]
                .detach().cpu().numpy().astype(np.float64)
            )
            pre_centers = field_centers.copy()
            pre_radii = field_radii.copy()
            rng = self._appear_rngs[env_index]
            base_slot = int(self.obstacle_count[env_index])
            spawned = 0
            skipped = False
            # Computed once per env, not once per cylinder: it is a Dijkstra
            # field over the whole basin and the boat does not move between
            # the cylinders of a single appearance.
            route_local = (
                route_direction_xy(
                    boat_local,
                    goal_local,
                    field_centers,
                    field_radii,
                    cell_m=float(self.cfg.appear_bfs_cell_m),
                    inflation_m=float(self.cfg.half_beam_m) + COLLISION_MARGIN_M,
                    lookahead_m=float(
                        getattr(self.cfg, "appear_route_lookahead_m", 8.0)
                    ),
                )
                if bool(getattr(self.cfg, "appear_route_aligned", False))
                else None
            )
            velocity_local = (
                None
                if velocity_2d is None
                else velocity_2d[env_index]
                .detach().cpu().numpy().astype(np.float64)
            )
            if (
                bool(getattr(self.cfg, "appear_route_aligned", False))
                and route_local is None
                and velocity_local is None
            ):
                # Route alignment was asked for and could not be computed for
                # THIS episode, which is a per episode geometry outcome, not a
                # configuration mistake. Skip the appearance and record it, the
                # same way an inadmissible azimuth is handled. Passing None on
                # to the planner would trip its near rung guard and kill the
                # whole 128 episode cell over one env in sixty-four: that guard
                # exists to catch an operator who never enabled alignment at
                # all, and must not fire here.
                self._appear_spawned[env_index] = 0
                self._appear_happened[env_index] = False
                self._appear_skipped[env_index] = True
                self._appear_azimuth_rad[env_index] = math.nan
                self._appear_clearance_m[env_index] = math.nan
                self._appear_distance_m[env_index] = math.nan
                continue
            first_azimuth = math.nan
            first_clearance = math.nan
            first_distance = math.nan
            for _ in range(self._appear_count):
                slot = base_slot + spawned
                if slot >= max_obstacles:
                    raise RuntimeError(
                        f"appear_count={self._appear_count} needs obstacle "
                        f"slot {slot} but max_obstacles={max_obstacles}; "
                        "raise max_obstacles on the appearance cfg, never on "
                        "a certified base cfg"
                    )
                candidate_order = rng.permutation(int(self.cfg.ray_count))[
                    : int(self.cfg.appear_max_attempts)
                ]
                plan = plan_obstacle_appearance(
                    boat_local,
                    forward_local,
                    goal_local,
                    field_centers,
                    field_radii,
                    distance_m=float(self.cfg.appear_distance_m),
                    radius_m=float(self.cfg.appear_radius_m),
                    candidate_ray_indices=candidate_order,
                    ray_count=int(self.cfg.ray_count),
                    goal_radius_m=float(self.goal_radius),
                    half_beam_m=float(self.cfg.half_beam_m),
                    bfs_cell_m=float(self.cfg.appear_bfs_cell_m),
                    stop_distance_m=float(
                        getattr(self.cfg, "appear_stop_distance_m", 0.868)
                    ),
                    velocity_xy=velocity_local,
                    route_xy=route_local,
                    strict_ray=bool(
                        getattr(self.cfg, "appear_strict_ray", False)
                    ),
                    ray_max_range_m=float(self.cfg.ray_max_range_m),
                )
                if plan is None:
                    # Every candidate azimuth was inadmissible: skip the
                    # appearance for this env and record it, rather than
                    # forcing an unfair or task-breaking spawn.
                    skipped = True
                    break
                self.obstacle_centers[env_index, slot] = origin + torch.as_tensor(
                    plan.center, dtype=torch.float32, device=self.device
                )
                self.obstacle_radii[env_index, slot] = float(plan.radius_m)
                self.obstacle_active[env_index, slot] = True
                # Later cylinders of the same appearance are admitted against
                # the field INCLUDING the ones just placed.
                field_centers = (
                    np.vstack((field_centers, plan.center[None, :]))
                    if field_centers.size
                    else plan.center[None, :].copy()
                )
                field_radii = np.concatenate(
                    (field_radii, (float(plan.radius_m),))
                )
                if spawned == 0:
                    first_azimuth = float(plan.azimuth_rad)
                    first_distance = float(plan.distance_m)
                    if pre_radii.size:
                        gaps = (
                            np.linalg.norm(pre_centers - plan.center, axis=1)
                            - pre_radii
                            - float(plan.radius_m)
                        )
                        first_clearance = float(gaps.min())
                    else:
                        first_clearance = math.inf
                spawned += 1
            self._appear_spawned[env_index] = spawned
            self._appear_happened[env_index] = spawned > 0
            self._appear_skipped[env_index] = skipped
            self._appear_azimuth_rad[env_index] = first_azimuth
            self._appear_clearance_m[env_index] = first_clearance
            self._appear_distance_m[env_index] = first_distance
            if (
                spawned > 0
                and getattr(
                    getattr(self.cfg, "underwater_physics_cfg", None),
                    "physics_version", 1,
                ) == 2
                and getattr(self, "_obstacle_translate_ops", None)
            ):
                # V2 appearance slots have display prims too. Refresh only the
                # affected env after admission, using the authoritative buffers.
                ids = torch.tensor([env_index], device=self.device, dtype=torch.long)
                origins = self.scene.env_origins[ids, :2]
                self._update_render_only_hazard_markers(
                    ids,
                    self.target_pos[ids] - origins,
                    self.obstacle_centers[ids] - origins.unsqueeze(1),
                    self.obstacle_radii[ids],
                    self.obstacle_active[ids],
                )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Guarded at the call site as well as inside the method, so the
        # certified inert path (appear_count == 0) makes no call at all. The
        # guard reads the CFG field (like contact_terminates below), not the
        # buffer mirror, so pre-existing cfg pickles without the field stay
        # inert via the getattr default.
        if getattr(self, "_moving_enabled", False):
            # Before the appearance hook, so that if the two axes are ever
            # combined the new cylinder is admitted against the field as it
            # actually stands this step rather than against last step's.
            self._update_moving_obstacles()
        if getattr(self.cfg, "appear_count", 0) > 0:
            self._maybe_appear_obstacles()
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
        publish_termination_reason(
            self, terminated, time_out, success=self._success,
            collision=contact_now if self.cfg.contact_terminates else None,
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
                if self._conditions is not None:
                    self.episode_conditions[completed] = (
                        self._conditions.episode_stamp(completed)
                    )
                self.episode_scenario[completed] = self._scenario_params[completed]
                self.episode_scenario_hashes[completed] = self._scenario_hashes[
                    completed
                ]

            if getattr(self.cfg, "appear_count", 0) > 0:
                # Same latch, same moment, as episode_min_clearance above:
                # the FINISHED episode's appearance record (did it happen,
                # was it skipped, where and when) survives the auto-reset so
                # the evaluator reads it beside that episode's metrics.
                self.episode_appear_time_s[completed_ids] = self._appear_time_s[
                    completed_ids
                ]
                self.episode_appear_happened[completed_ids] = (
                    self._appear_happened[completed_ids]
                )
                self.episode_appear_skipped[completed_ids] = (
                    self._appear_skipped[completed_ids]
                )
                self.episode_appear_spawned[completed_ids] = (
                    self._appear_spawned[completed_ids]
                )
                self.episode_appear_azimuth_rad[completed_ids] = (
                    self._appear_azimuth_rad[completed_ids]
                )
                self.episode_appear_distance_m[completed_ids] = (
                    self._appear_distance_m[completed_ids]
                )
                self.episode_appear_clearance_m[completed_ids] = (
                    self._appear_clearance_m[completed_ids]
                )
                self.extras.setdefault("log", {})
                self.extras["log"]["Episode/appearance_happened"] = (
                    self._appear_happened[completed_ids].float().mean()
                )
                self.extras["log"]["Episode/appearance_skipped"] = (
                    self._appear_skipped[completed_ids].float().mean()
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
        if self.cfg.payload_mass_kg_choices:
            choice_indices = actuator_choice_indices(
                self._scenario,
                self._layout_rng,
                GROUP_PAYLOAD_MASS_KG,
                reset_env_indices,
                len(self.cfg.payload_mass_kg_choices),
            )
            choices = np.asarray(
                self.cfg.payload_mass_kg_choices, dtype=np.float32
            )
            drawn = choices[choice_indices]
            self._payload_mass_kg_per_env[env_ids] = torch.as_tensor(
                drawn, device=self.device
            )
            actuator_scenario.update(
                actuator_scenario_entry(GROUP_PAYLOAD_MASS_KG, drawn.tolist())
            )
        # Actuator memory is episode-local for both scalar and choice modes.
        self._applied_thrust_per_env[env_ids] = 0.0
        self._applied_yaw_per_env[env_ids] = 0.0
        max_obstacles = getattr(self.cfg, "obstacle_slot_count", self.cfg.max_obstacles)
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
        moving_kind_np = np.zeros((num_resets, max_obstacles), dtype=np.int64)
        moving_params_np = np.zeros(
            (num_resets, max_obstacles, 7), dtype=np.float32
        )
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
            moving_spec = None
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
            elif self.cfg.layout_mode == "moving_basin":
                # P5 layer three: two staggered rows sliding sideways, a
                # buffer, then one orbiting cylinder. Admission checks
                # feasibility at sampled times ACROSS a period rather than at
                # t = 0 alone, because a moving field can be open when sampled
                # and sealed a second later -- a failure mode no static
                # sampler has.
                layout, moving_spec = sample_moving_basin_layout(
                    level,
                    rng=layout_rng,
                    mu=float(getattr(self.cfg, "moving_mu", 0.0)),
                    nu=float(getattr(self.cfg, "moving_nu", 0.0)),
                    max_attempts=max(60, self.cfg.layout_max_attempts),
                    gap_width_override_m=(
                        getattr(self.cfg, "moving_gap_width_override_m", 0.0)
                        or None
                    ),
                    anti_phase=bool(
                        getattr(self.cfg, "moving_anti_phase", False)
                    ),
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
            if (
                getattr(self.cfg.underwater_physics_cfg, "physics_version", 1) == 2
                and count > int(self.cfg.max_obstacles)
            ):
                raise RuntimeError(
                    f"static layout needs {count} obstacle slots but "
                    f"max_obstacles={self.cfg.max_obstacles}; appearance slots "
                    "are reserved separately"
                )
            local_starts_np[row] = rotated_start
            local_goals_np[row] = rotated_goal
            local_centers_np[row, :count] = rotated_centers
            radii_np[row, :count] = layout.radii
            active_np[row, :count] = True
            if moving_spec is not None:
                # The whole scene is turned by goal_angle, so the MOTION has to
                # be turned with it or the cylinders keep sliding along the old
                # world axis while the basin faces a new way.
                moving_kind_np[row, :count] = moving_spec.kind
                moving_params_np[row, :count] = rotate_moving_params(
                    moving_spec.params, moving_spec.kind, goal_angle
                )
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
        if self._moving_enabled:
            self._moving_kind[env_ids] = torch.as_tensor(
                moving_kind_np, device=self.device
            )
            self._moving_params[env_ids] = torch.as_tensor(
                moving_params_np, device=self.device
            )
            self._moving_base[env_ids] = local_centers
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

        # Mid-episode appearance (appearance variants only): the per-episode
        # time t0 off the protocol's own "appearance" stream, stamped as the
        # raw draw. appear_count == 0 contributes {} without touching
        # anything -- no call, no buffer, no draw -- so certified ids keep
        # their exact scenario-hash key set.
        appear_scenario = (
            self._reset_appearance(env_ids, reset_env_indices)
            if getattr(self.cfg, "appear_count", 0) > 0
            else {}
        )

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
                **appear_scenario,
                **wave_scenario,
            },
        ):
            self._scenario_params[env_index] = resolved
            self._scenario_hashes[env_index] = hashes

        self._update_render_only_hazard_markers(
            env_ids, local_goals, local_centers, radii, active
        )
        if self._conditions is not None:
            # Task reference axis for relative wind directions: the bearing
            # of this episode's start-to-goal route (env-local frame, which
            # has the world orientation).
            route_axis = local_goals - local_starts
            self._conditions.reset_runtime(
                env_ids=env_ids,
                reference_axis_rad=torch.atan2(route_axis[:, 1], route_axis[:, 0]),
                task_episode_ids=(
                    None if self._scenario is None
                    else self._scenario.episode_indices()[env_ids.cpu()]
                ),
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

    # ======================================================================
    # Journal task contract v2 (F1; CONTEXT_R4 part F). Everything below runs
    # ONLY for cfgs with journal_contract_version == 2 (E5 templates). The
    # journal versions of the DirectRLEnv hooks are bound as INSTANCE
    # attributes by _journal_init, so the legacy class methods
    # (_pre_physics_step, _apply_action, _get_dones, _reset_idx,
    # _get_observations, ...) keep their exact source (D15). Call order: the
    # F0 JournalEnvMixin docstring (tasks/_shared/journal_env_mixin.py).
    # ======================================================================
    _JOURNAL_BOUND = (
        "_pre_physics_step",
        "_apply_action",
        "_get_dones",
        "_reset_idx",
        "_get_observations",
        "_com_xy",
        "_clearance",
        "_ray_ranges_m",
    )

    def _journal_init(self) -> None:
        """Validate the journal cfg, run journal_setup, allocate, bind the hooks."""
        from .._shared.conditions.observation_groups import HOOKS_FLAG

        cfg = self.cfg
        task = getattr(cfg, "journal_task", None)
        if task not in _journal.CARRIED_TASKS:
            raise ValueError(
                f"journal task {task!r} is not carried by HazardNav "
                f"(carried: {_journal.CARRIED_TASKS})"
            )
        if cfg.layout_mode != "journal":
            raise ValueError("journal cfgs use layout_mode='journal' (E5 build_layout)")
        width = int(self.obstacle_centers.shape[1])
        appearing = bool(getattr(getattr(cfg, "journal_conditions", None), "appearing_enabled", False))
        if width != _journal.storage_width(task) + int(appearing):
            raise ValueError(
                f"journal obstacle storage {width} != base maximum + D7 reserve "
                f"{_journal.storage_width(task)} (+1 appearing slot: {appearing}) for {task!r} (D7 B04/N1)"
            )
        if getattr(cfg, HOOKS_FLAG, False) is not True:
            raise ValueError(f"journal cfgs set {HOOKS_FLAG}=True (estimate API wired)")
        com_offset = getattr(self.robot.data, "com_pos_b", None)
        if com_offset is not None and com_offset.numel() > 0:
            if float(com_offset.detach().abs().max()) > 1.0e-5:
                raise ValueError(
                    "v2 assets carry the actor origin at the COM (ANCHORS_E5 C5); "
                    f"body com_pos_b = {com_offset.detach().abs().max().item():.3e} m"
                )
        raw_pool, pool_meta = _journal.resolve_d7_pool(cfg)
        # D16: per offered key the admitted ladder OR an explicit not-admitted record
        pool = _journal.family_pool(raw_pool)
        self.journal_setup(d7_pool=pool)
        _journal.check_pool_suite(pool_meta, self._d7_suite)
        self._journal_pool_meta = pool_meta
        # D16: not-admitted keys run at D0 (no intruder) through the family wrapper of
        # the E4 bridge; admitted rows reach the E4 bridge unchanged
        self._d7_bridge = _journal.NotAdmittedBridge(self._d7_bridge, task)
        self.dynamic_hazard_header.update(_journal.pool_header_extra(pool, pool_meta))
        if self._appearing_bridge is not None:
            # G3: the appearing cylinder owns the last storage slot (CapacityPlan.appearing_slots)
            if self._appearing_slot != width - 1:
                raise ValueError(f"appearing slot {self._appearing_slot} is not the last storage slot {width - 1}")
            self.appearing_hazard_header["collision_model"] = (
                "HazardNav static obstacle channel: once triggered the cylinder is in obstacle_active, so the "
                "contract's swept E5 polygon clearance, the observation rays and the reward clearance see it "
                "exactly like a layout obstacle"
            )
        n, device = self.num_envs, self.device
        record = _journal.plans.vessel_record(cfg.vehicle)
        self._journal_record = record
        self._journal_water_z = float(self.physics_cfg.water_surface_z)
        self._journal_eq_z = float(self._hydro_v2.profile.hydro.equilibrium_com_z_m)
        if abs(self._journal_eq_z - float(record.com_height_above_waterline_m)) > 1.0e-12:
            raise ValueError("hydro_v2 equilibrium COM height differs from the contract record")
        self._journal_twin = _journal.ReachEpisodeTensor(
            n, device, cfg.journal_plan.validity_limits, record.com_height_above_waterline_m
        )
        self._journal_vertices = _journal.footprint_vertices(cfg.vehicle, device)
        self._journal_ordinal = [0] * n
        self._journal_episode_ordinal = torch.full((n,), -1, dtype=torch.long)
        self._d7_physical = torch.zeros((n, width), dtype=torch.bool, device=device)
        self._d7_visible = torch.zeros_like(self._d7_physical)
        # G23c F8: the D7 intruder (reserved) slots answer the rays through the point-return proxy only
        # (moving-basin native slots and the appearing slot keep the ray-circle intersection)
        base = int(_journal.plans.STATIC_CAPACITY[task])
        self._jd7_point_slots = _d7pr.slot_mask(width, range(base, _journal.storage_width(task)), device)
        self._journal_prev_pose = torch.zeros((n, 3), dtype=torch.float64, device=device)
        self._journal_bounds = torch.zeros((n, 4), dtype=torch.float64, device=device)
        self._journal_scenarios = [None] * n
        self._journal_family_hash = [None] * n
        self.journal_completed_records = [None] * n
        self.journal_contract_trace = None
        self.episode_termination_reason = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self.episode_success_prefix_path_length_m = torch.full(
            (n,), torch.nan, device=device
        )
        self.episode_full_path_length_m = torch.zeros(n, device=device)
        if task == "moving_basin":
            self.journal_pairing_inventory = tuple(
                sorted(set(self.journal_pairing_inventory) | {_journal.BASIN_PRIMITIVE})
            )
        for name in self._JOURNAL_BOUND:
            setattr(self, name, getattr(self, "_journal" + name))

    # ---- state helpers (root-link pose = COM for v2 assets; never root_com_pos_w)
    def _journal_pose(self, robot_data, env_origins) -> torch.Tensor:
        pose = robot_data.root_link_pose_w
        xy = (pose[:, :2] - env_origins[:, :2]).to(torch.float64)
        yaw = _journal.yaw_from_quat(pose[:, 3:7])
        return torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)

    def _journal_rigid(self, robot_data, env_origins):
        pose = robot_data.root_link_pose_w
        velocity = robot_data.root_com_vel_w
        com = pose[:, :3] - env_origins[:, :3]
        return com, pose[:, 3:7], velocity[:, :3], velocity[:, 3:6]

    def _journal_centers_local(self) -> torch.Tensor:
        return self.obstacle_centers - self.scene.env_origins[:, None, :2]

    def _journal_static_clearance(self, pose: torch.Tensor) -> torch.Tensor:
        """Swept static clearance (E5 footprint) from the last observed pose."""
        return _journal.swept_polygon_clearance(
            self._journal_prev_pose,
            pose,
            self._journal_vertices,
            self._journal_centers_local(),
            self.obstacle_radii,
            self.obstacle_active,
            core_radius=self._journal_core_radius(),
        )

    def _journal_core_radius(self) -> float:
        """G11: footprint_core_radius of the footprint polygon, read from the device once (it was read
        on every swept-clearance call); recomputed if the tensor is replaced or written in place."""
        vertices = self._journal_vertices
        cache = getattr(self, "_journal_core_radius_cache", None)
        if cache is None or cache[0] is not vertices or cache[1] != vertices._version:
            cache = (vertices, vertices._version, _journal.footprint_core_radius(vertices))
            self._journal_core_radius_cache = cache
        return cache[2]

    def _journal_rays(self, mask: torch.Tensor) -> torch.Tensor:
        """36 analytic rays from the COM over the slots in ``mask`` (metres)."""
        forward = self._forward_2d()
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        cosine = torch.cos(self._ray_angles).view(1, -1, 1)
        sine = torch.sin(self._ray_angles).view(1, -1, 1)
        directions = cosine * forward.unsqueeze(1) + sine * left.unsqueeze(1)
        point = mask & self._jd7_point_slots              # G23c F8: D7 intruder slots -> point-return proxy
        rays = ray_circle_ranges(
            self.robot.data.root_link_pose_w[:, :2],
            directions,
            self.obstacle_centers,
            self.obstacle_radii,
            max_range_m=float(self.cfg.ray_max_range_m),
            active_mask=mask & ~self._jd7_point_slots,
        )
        return _d7pr.merge(rays, self.robot.data.root_link_pose_w[:, :2], torch.atan2(forward[:, 1], forward[:, 0]),
                           self.obstacle_centers, self.obstacle_radii, point, int(self.cfg.ray_count),
                           float(self.cfg.ray_max_range_m))

    def _journal_record_trace(self, kind, **values) -> None:
        if self.journal_contract_trace is not None:
            entry = {"kind": kind}
            for key, value in values.items():
                entry[key] = (
                    value.detach().to("cpu", torch.float64).clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
            self.journal_contract_trace.append(entry)

    # ---- bound DirectRLEnv hooks
    def _journal_pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.journal_pre_physics(actions)

    def _journal_apply_action(self) -> None:
        forces, torques = self.journal_substep(
            self.robot.data, self.scene.env_origins, self._journal_water_z
        )
        num_bodies = int(getattr(self.robot, "num_bodies", 1))
        if num_bodies != 1:
            full_f = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
            full_t = torch.zeros_like(full_f)
            full_f[:, :1] = forces.to(full_f.dtype)
            full_t[:, :1] = torques.to(full_t.dtype)
            forces, torques = full_f, full_t
        from .._shared.body_wrench import apply_body_wrench   # G6: no per-call deprecation warning
        apply_body_wrench(self.robot, forces, torques)

    def _journal_observe_physics(self, robot_data, env_origins, water_level_local) -> None:
        """Contract observe_physics on the PREVIOUS substep's state (mixin step 1, D8 level)."""
        pose = self._journal_pose(robot_data, env_origins)
        clearance = self._journal_static_clearance(pose)
        com, quat, lin, ang = self._journal_rigid(robot_data, env_origins)
        water = torch.as_tensor(water_level_local, device=self.device).expand(self.num_envs)
        present = self.obstacle_active.any(dim=-1)
        self._journal_record_trace(
            "observe", com=com, quat=quat, lin=lin, ang=ang, water=water, clearance=clearance,
            present=present,
        )
        self._journal_twin.observe_physics(
            com, quat, lin, ang, water, clearance_m=clearance, clearance_present=present
        )
        self._journal_prev_pose = pose

    def _journal_publish_d7(self, samples) -> None:
        """B07: D7 samples into the reserved slot (native basin slots), split masks."""
        if not samples:
            return
        rows, slots, xs, ys, rs, physical, visible = [], [], [], [], [], [], []
        width = int(self.obstacle_centers.shape[1])
        for env_index, items in samples.items():
            for item in items:
                if not 0 <= int(item.slot) < width:
                    raise ValueError(f"D7 slot {item.slot} outside the {width}-slot storage")
                rows.append(int(env_index))
                slots.append(int(item.slot))
                xs.append(float(item.x))
                ys.append(float(item.y))
                rs.append(float(item.radius_m))
                physical.append(bool(item.physical))
                visible.append(bool(item.visible))
        if not rows:
            return
        dtype = self.obstacle_centers.dtype
        # G11: the seven host tensors of the per-field as_tensor(..., device=) calls, built on the host
        # exactly as those calls build them, moved in ONE host-to-device copy (bytes verbatim)
        row_t, slot_t, xs_t, ys_t, rs_t, physical_t, visible_t = device_copies(
            [torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(slots, dtype=torch.long),
             torch.as_tensor(xs, dtype=dtype), torch.as_tensor(ys, dtype=dtype),
             torch.as_tensor(rs, dtype=self.obstacle_radii.dtype), torch.as_tensor(physical, dtype=torch.bool),
             torch.as_tensor(visible, dtype=torch.bool)], self.device)
        origin = self.scene.env_origins[row_t, :2]
        self.obstacle_centers[row_t, slot_t, 0] = xs_t + origin[:, 0]
        self.obstacle_centers[row_t, slot_t, 1] = ys_t + origin[:, 1]
        self.obstacle_radii[row_t, slot_t] = rs_t
        self._d7_physical[row_t, slot_t] = physical_t
        self._d7_visible[row_t, slot_t] = visible_t

    def _journal_publish_appearing(self, samples) -> None:
        """G3: the appearing cylinder in its slot; physical = visible = obstacle_active (static channel)."""
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
        origin = self.scene.env_origins[row_t, :2]
        dtype = self.obstacle_centers.dtype
        self.obstacle_centers[row_t, slot, 0] = torch.as_tensor(xs, dtype=dtype, device=self.device) + origin[:, 0]
        self.obstacle_centers[row_t, slot, 1] = torch.as_tensor(ys, dtype=dtype, device=self.device) + origin[:, 1]
        self.obstacle_radii[row_t, slot] = torch.as_tensor(rs, dtype=self.obstacle_radii.dtype, device=self.device)
        active = torch.as_tensor(on, dtype=torch.bool, device=self.device)
        newly = active & ~self.obstacle_active[row_t, slot]
        self.obstacle_active[row_t, slot] = active
        if bool(newly.any()):
            self._journal_show_appearing_marker(row_t[newly].tolist())

    def _journal_show_appearing_marker(self, env_indices) -> None:
        """Render-only: make the appearing slot's display prim visible (analytic tensors stay authoritative)."""
        if not self._obstacle_translate_ops:
            return
        try:
            from pxr import Gf, UsdGeom

            slot = self._appearing_slot
            for env_index in env_indices:
                center = (self.obstacle_centers[env_index, slot] - self.scene.env_origins[env_index, :2]).tolist()
                self._obstacle_translate_ops[env_index][slot].Set(Gf.Vec3d(center[0], center[1], 0.0))
                self._obstacle_radius_attrs[env_index][slot].Set(float(self.obstacle_radii[env_index, slot]))
                UsdGeom.Imageable(self._obstacle_prims[env_index][slot]).MakeVisible()
        except Exception as exc:
            print(f"[WARN] Appearing marker update skipped: {exc}")

    def _journal_appearing_geometry(self, env_id) -> str:
        """Geometry id of the env's current E5 scenario (the appearing pool records it per key)."""
        scenario = self._journal_scenarios[int(env_id)]
        if scenario is None:
            raise RuntimeError(f"env {int(env_id)} has no journal scenario")
        return scenario.geometry_id

    def _journal_family_scenario_hashes(self, env_id) -> dict:
        hashes = self._journal_family_hash[int(env_id)]
        if hashes is None:
            raise RuntimeError(f"env {int(env_id)} has no journal scenario")
        return dict(hashes)

    def _journal_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        view = self.journal_pre_dones(
            self.robot.data, self.scene.env_origins, self._journal_water_z
        )
        pose = self._journal_pose(self.robot.data, self.scene.env_origins)
        clearance = self._journal_static_clearance(pose)
        com, quat, lin, ang = self._journal_rigid(self.robot.data, self.scene.env_origins)
        bounds = self._journal_bounds
        com64 = com.to(torch.float64)
        out_of_bounds = (
            (com64[:, 0] < bounds[:, 0])
            | (com64[:, 0] > bounds[:, 1])
            | (com64[:, 1] < bounds[:, 2])
            | (com64[:, 1] > bounds[:, 3])
        )
        time_s = self.episode_length_buf.to(torch.float64) * _journal.plans.CONTROL_DT_S
        water = torch.as_tensor(view.water_level_local, device=self.device).expand(self.num_envs)
        present = self.obstacle_active.any(dim=-1)
        self._journal_record_trace(
            "step", time=time_s, com=com, quat=quat, lin=lin, ang=ang, water=water,
            clearance=clearance, present=present, collision=view.d7_contact,
            out_of_bounds=out_of_bounds,
        )
        terminated_t, truncated_t, codes = self._journal_twin.step(
            time_s, com, quat, lin, ang, water,
            collision=view.d7_contact, out_of_bounds=out_of_bounds, clearance_m=clearance,
            clearance_present=present,
        )
        self._journal_prev_pose = pose
        terminated, truncated = self.journal_dones(codes)
        # G11: the four host copies of the consistency check in ONE device-to-host copy
        term_h, term_t_h, trunc_h, trunc_t_h = host_copies([terminated, terminated_t, truncated, truncated_t])
        if not (torch.equal(term_h, term_t_h) and torch.equal(trunc_h, trunc_t_h)):
            raise RuntimeError("contract done masks disagree with the E5 termination table")
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)
        done = terminated | truncated
        done_any = bool((term_h | trunc_h).any())          # host copies of the same masks: no device sync
        success_now = codes == int(_journal.contracts.TerminationReason.SUCCESS)
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._clean_goal_entry_this_step.copy_(success_now)
        self._first_success_time_s.copy_(
            torch.where(success_now, elapsed_s, self._first_success_time_s)
        )
        self._success |= success_now
        self._reached_goal |= success_now
        self._episode_finished.copy_(done)
        self.path_length.copy_(self._journal_twin.full_path.to(self.path_length.dtype))
        self._min_clearance.copy_(self._journal_twin.min_clearance.to(self._min_clearance.dtype))
        if done_any:
            self._journal_export_completed(done, codes)
        return terminated, truncated

    def _journal_export_completed(self, done: torch.Tensor, codes: torch.Tensor) -> None:
        """Completed-episode export (contract fields + D11 record) BEFORE the reset."""
        from .._shared.termination_reason import termination_code_name

        twin = self._journal_twin
        self.episode_success[done] = twin.c_success[done]
        self.time_to_success[done] = twin.c_time_to_success[done].to(self.time_to_success.dtype)
        self.episode_path_length[done] = twin.c_full[done].to(self.episode_path_length.dtype)
        self.episode_full_path_length_m[done] = twin.c_full[done].to(torch.float32)
        self.episode_success_prefix_path_length_m[done] = twin.c_prefix[done].to(torch.float32)
        self.episode_min_clearance[done] = twin.c_min_clearance[done].to(
            self.episode_min_clearance.dtype
        )
        self.route_geodesic_length[done] = self._route_geodesic_length[done]
        self.episode_termination_reason[done] = codes[done]
        for env_index in done.nonzero().flatten().tolist():
            code = int(codes[env_index])
            scenario = self._journal_scenarios[env_index]
            self.journal_completed_records[env_index] = {
                "schema": "usvbench.f1.completed_episode/2",   # /2: d7_exposure (D16)
                "task": self.cfg.journal_task,
                "vessel": self.cfg.vehicle,
                "env": int(env_index),
                "episode_ordinal": int(self._journal_episode_ordinal[env_index]),
                "scenario_episode": int(scenario.episode),
                "termination_code": code,
                "termination_name": termination_code_name(code),
                "contract": twin.completed_fields(env_index),
                "scenario": scenario.summary(),
                **self.journal_episode_record(env_index),
            }
            record = self.journal_completed_records[env_index]
            record["d7_exposure"] = _journal.d7_exposure(record)
            if "appearing_hazards" in record:
                record["appearing_exposure"] = _journal.appearing_exposure(record)
            self.episode_scenario[env_index] = scenario.summary()
        self.extras.setdefault("log", {})
        self.extras["log"]["Episode/success"] = twin.c_success[done].float().mean()
        self.extras["log"]["Episode/path_length_m"] = twin.c_full[done].float().mean()

    def _journal_next_episode(self, env_index: int) -> tuple[int, int]:
        """(task episode ordinal, scenario episode) of this env's next episode.

        D16: the scenario of key (seed, env, episode) is the E5 layout of that
        key, always; the consecutive ordinal IS the scenario episode (no
        admitted-only schedule, no skipped scenario, no redraw).
        """
        ordinal = self._journal_ordinal[env_index]
        self._journal_ordinal[env_index] = ordinal + 1
        return ordinal, ordinal

    def _journal_reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        super()._reset_idx(env_ids)
        cfg = self.cfg
        ids = env_ids.detach().cpu().tolist()
        # 1. task scenario counter: consecutive ordinals = scenario episodes (condition
        #    layer, D7 key, E5 layout key; D16)
        pairs = [self._journal_next_episode(i) for i in ids]
        ordinals = [p[0] for p in pairs]
        episodes = [p[1] for p in pairs]
        self._scenario.reset_idx(ids, episode_index=episodes)
        self._journal_episode_ordinal[env_ids.cpu()] = torch.as_tensor(ordinals, dtype=torch.long)
        # 2. layout from E5 build_layout (quarter-turn rotation, env-local)
        scenarios = [
            _journal.build_scenario(
                cfg.journal_task, cfg.vehicle, int(cfg.journal_geometry_tier),
                int(cfg.seed), i, episode,
            )
            for i, episode in zip(ids, episodes)
        ]
        dose = int(self._d7_suite.dose)
        for i, scenario in zip(ids, scenarios):
            # admitted -> the E4 assignment of this geometry; not admitted -> D0 row (D16);
            # beyond the offered grid -> D0 row, single reason beyond_offered_grid, no pool check (D24)
            _journal.check_assignment_geometry(self._d7_pool, scenario, dose)
            self._d7_bridge.stage(i, scenario)
        count = len(ids)
        width = int(self.obstacle_centers.shape[1])
        centers = np.zeros((count, width, 2), dtype=np.float64)
        radii = np.zeros((count, width), dtype=np.float64)
        static = np.zeros((count, width), dtype=np.bool_)
        layout_slots = np.zeros((count, width), dtype=np.bool_)
        starts = np.zeros((count, 2), dtype=np.float64)
        goals = np.zeros((count, 2), dtype=np.float64)
        yaws = np.zeros(count, dtype=np.float64)
        bounds = np.zeros((count, 4), dtype=np.float64)
        axes = np.zeros(count, dtype=np.float64)
        counts = np.zeros(count, dtype=np.int64)
        route_len = np.zeros(count, dtype=np.float64)
        for row, scenario in enumerate(scenarios):
            n_obs = scenario.obstacle_count
            centers[row, :n_obs] = scenario.centers
            radii[row, :n_obs] = scenario.radii
            static[row, list(scenario.static_slots)] = True
            layout_slots[row, :n_obs] = True
            starts[row] = scenario.start_xy
            goals[row] = scenario.goal_xy
            yaws[row] = scenario.start_yaw
            bounds[row] = scenario.bounds
            axes[row] = scenario.reference_axis_rad
            counts[row] = n_obs
            route_len[row] = scenario.nominal_route_length_m
        device = self.device
        origins = self.scene.env_origins[env_ids]
        centers_t = torch.as_tensor(centers, dtype=torch.float32, device=device)
        radii_t = torch.as_tensor(radii, dtype=torch.float32, device=device)
        static_t = torch.as_tensor(static, device=device)
        starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)
        goals_t = torch.as_tensor(goals, dtype=torch.float32, device=device)
        self.target_pos[env_ids] = origins[:, :2] + goals_t
        self.obstacle_centers[env_ids] = origins[:, None, :2] + centers_t
        self.obstacle_radii[env_ids] = radii_t
        self.obstacle_active[env_ids] = static_t
        self._d7_physical[env_ids] = False
        self._d7_visible[env_ids] = False
        self.obstacle_count[env_ids] = torch.as_tensor(counts, device=device)
        self._journal_bounds[env_ids] = torch.as_tensor(bounds, dtype=torch.float64, device=device)
        d0 = np.linalg.norm(goals - starts, axis=1)
        if self.cfg.reward_progress_geodesic:
            d0 = self._journal_write_geodesic(env_ids, scenarios)
        self.d0_per_env[env_ids] = torch.as_tensor(d0, dtype=torch.float32, device=device)
        self._route_geodesic_length[env_ids] = torch.as_tensor(
            route_len, dtype=torch.float32, device=device
        )
        # 3. spawn: COM at water level + equilibrium COM height, at rest (D3 s4)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        yaw_t = torch.as_tensor(yaws, dtype=torch.float64, device=device)
        root_state[:, 0:2] = origins[:, :2] + starts_t
        root_state[:, 2] = origins[:, 2] + (self._journal_water_z + self._journal_eq_z)
        root_state[:, 3] = torch.cos(0.5 * yaw_t).to(root_state.dtype)
        root_state[:, 4:6] = 0.0
        root_state[:, 6] = torch.sin(0.5 * yaw_t).to(root_state.dtype)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)
        # 4. per-episode task buffers (reward bookkeeping reads truth)
        self.path_length[env_ids] = 0.0
        self._previous_xy[env_ids] = origins[:, :2] + starts_t
        self._previous_potential[env_ids] = -1.0
        self._reached_goal[env_ids] = False
        self._contact_prev[env_ids] = False
        self._contact_before_goal[env_ids] = False
        self._success[env_ids] = False
        self._clean_goal_entry_this_step[env_ids] = False
        self._first_success_time_s[env_ids] = torch.nan
        self._episode_finished[env_ids] = False
        self.actions[env_ids] = 0.0
        for buffer in (
            self._fee_steps, self._ep_prox_cost, self._ep_open_water_cost,
            self._ep_contact_entries, self._ep_contact_steps, self._ep_contact_run_steps,
            self._ep_contact_longest_steps, self._ep_contact_depth_sum,
            self._ep_goal_bonus, self._ep_reverse_cost,
        ):
            buffer[env_ids] = 0.0
        spawn_pose = torch.cat(
            (torch.as_tensor(starts, dtype=torch.float64, device=device), yaw_t.unsqueeze(-1)), dim=-1
        )
        initial = _journal.polygon_clearance(
            spawn_pose[:, :2], spawn_pose[:, 2], self._journal_vertices,
            centers_t.to(torch.float64), radii_t, static_t,
        )
        if bool((initial <= 0.0).any()):
            raise RuntimeError("journal spawn footprint touches an obstacle (admission violated)")
        self._min_clearance[env_ids] = initial.to(self._min_clearance.dtype)
        self._journal_prev_pose[env_ids] = spawn_pose
        # 5. contract twin (per-scenario goal and deadline)
        self._journal_twin.reset(
            env_ids, starts, goals,
            [s.goal_radius_m for s in scenarios], [s.deadline_s for s in scenarios],
        )
        for i, scenario in zip(ids, scenarios):
            self._journal_scenarios[i] = scenario
            self._journal_family_hash[i] = _journal.family_hashes(scenario, self._journal_eq_z)
            self._scenario_params[i] = scenario.summary()
        # 6. shared glue: runtime reset, conditions, D7 rows and the tick-0 step
        self.journal_reset(
            env_ids,
            torch.as_tensor(axes, dtype=torch.float64),
            torch.as_tensor(ordinals, dtype=torch.long),
            reference_route_length_m=torch.as_tensor(route_len, dtype=torch.float64),   # G28 outage rule
        )
        self._update_render_only_hazard_markers(
            env_ids, goals_t, centers_t, radii_t,
            torch.as_tensor(layout_slots, device=device),
        )

    def _journal_write_geodesic(self, env_ids: torch.Tensor, scenarios) -> np.ndarray:
        """Geodesic progress field with the E5 footprint inflation (not HALF_BEAM_M).

        Inflation = the vessel's lateral half extent + the tier planning margin
        (D4 INTEGRATION_SPEC 3.1), cell = the plan's grid cell; static slots
        only. Returns the spawn geodesic distance per reset env.
        """
        params = scenarios[0].layout.params
        cell = float(params["grid_cell_m"])
        inflation = float(self._journal_record.geometry.lateral_half_extent_m) + float(
            params["planning_margin_m"]
        )
        fields, origins, spawn = [], [], []
        for scenario in scenarios:
            slots = list(scenario.static_slots)
            b = scenario.bounds
            box = (b[0] - 2.0 * cell, b[1] + 2.0 * cell, b[2] - 2.0 * cell, b[3] + 2.0 * cell)
            field = geodesic_distance_field(
                scenario.centers[slots], scenario.radii[slots], np.asarray(scenario.goal_xy),
                box, cell_m=cell, half_beam_m=inflation,
            )
            origin = cell * np.floor(np.asarray((box[0], box[2])) / cell)
            finite = np.isfinite(field)
            if not np.any(finite):
                raise RuntimeError("journal geodesic field has no goal-reachable cells")
            col_f, row_f = (np.asarray(scenario.start_xy) - origin) / cell
            col0, row0 = int(math.floor(col_f)), int(math.floor(row_f))
            col1 = min(col0 + 1, field.shape[1] - 1)
            row1 = min(row0 + 1, field.shape[0] - 1)
            col_t, row_t = col_f - col0, row_f - row0
            value = float(
                (1.0 - row_t) * (1.0 - col_t) * field[row0, col0]
                + (1.0 - row_t) * col_t * field[row0, col1]
                + row_t * (1.0 - col_t) * field[row1, col0]
                + row_t * col_t * field[row1, col1]
            )
            if not math.isfinite(value):
                raise RuntimeError("journal spawn is not finite in the geodesic field")
            fields.append(field)
            origins.append(origin)
            spawn.append(value)
        height = max(f.shape[0] for f in fields)
        width = max(f.shape[1] for f in fields)
        if height > self._geodesic_fields.shape[1] or width > self._geodesic_fields.shape[2]:
            grown = torch.full(
                (self.num_envs, max(height, self._geodesic_fields.shape[1]),
                 max(width, self._geodesic_fields.shape[2])),
                torch.inf, device=self.device,
            )
            grown[:, : self._geodesic_fields.shape[1], : self._geodesic_fields.shape[2]] = (
                self._geodesic_fields
            )
            self._geodesic_fields = grown
        for row, env_index in enumerate(env_ids.tolist()):
            field = fields[row]
            self._geodesic_fields[env_index] = torch.inf
            self._geodesic_fields[env_index, : field.shape[0], : field.shape[1]] = torch.as_tensor(
                field, device=self.device
            )
            self._geodesic_field_origin[env_index] = torch.as_tensor(
                origins[row], dtype=torch.float32, device=self.device
            )
            self._geodesic_field_cell[env_index] = cell
            self._geodesic_field_shape[env_index, 0] = field.shape[0]
            self._geodesic_field_shape[env_index, 1] = field.shape[1]
            self._geodesic_field_max[env_index] = float(np.max(field[np.isfinite(field)]))
        return np.asarray(spawn, dtype=np.float64)

    def _journal_get_observations(self) -> dict:
        """HazardNav v3 layout from the own-state estimates (D5) + the 36-ray block (D14)."""
        data = self.robot.data
        origins = self.scene.env_origins
        com_local = data.root_link_pose_w[:, :2] - origins[:, :2]
        body = self._body_motion_measurement()
        rays = self._journal_rays(self.obstacle_active | self._d7_visible)
        est = self.journal_estimates(com_local, body, rays)
        position = est.position.to(torch.float32)
        to_goal = (self.target_pos - origins[:, :2]) - position
        distance = torch.norm(to_goal, dim=-1, keepdim=True)
        direction = to_goal / distance.clamp_min(1.0e-6)
        forward = self._forward_2d()
        dot = torch.sum(forward * direction, dim=-1, keepdim=True)
        cross = forward[:, 0:1] * direction[:, 1:2] - forward[:, 1:2] * direction[:, 0:1]
        distance_norm = torch.clamp(
            distance / self.d0_per_env.unsqueeze(-1).clamp_min(1.0e-6), min=0.0, max=1.0
        )
        surge_norm, sway_norm, yaw_rate_norm = self._body_motion_observation(
            est.velocity.to(torch.float32)
        )
        ranges_norm = est.rays.to(torch.float32) / float(self.cfg.ray_max_range_m)
        observation = torch.hstack(
            (dot, cross, distance_norm, surge_norm, sway_norm, yaw_rate_norm, ranges_norm)
        )
        return {"policy": observation}

    # ---- helpers read by the (unchanged) legacy _get_rewards
    def _journal_com_xy(self) -> torch.Tensor:
        return self.robot.data.root_link_pose_w[:, :2]

    def _journal_clearance(self) -> torch.Tensor:
        """Current-pose footprint clearance to the static objects and the DETECTABLE D7 objects (reward shaping;
        G23c F12: a physical but undetected D7 disk is no privileged training signal; D7 collisions still reach
        the contract through the bridge's swept-contact latch)."""
        pose = self._journal_pose(self.robot.data, self.scene.env_origins)
        return _journal.polygon_clearance(
            pose[:, :2], pose[:, 2], self._journal_vertices, self._journal_centers_local(),
            self.obstacle_radii, self.obstacle_active | self._d7_visible,
        ).to(torch.float32)

    def _journal_ray_ranges_m(self) -> torch.Tensor:
        """Reward-path rays over the static and the DETECTABLE D7 objects (G23c F12, as the observation)."""
        return self._journal_rays(self.obstacle_active | self._d7_visible)
