# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .._shared.termination_reason import publish_termination_reason

import math
from collections.abc import Sequence

from .._shared.vehicles import physics_version
from .._shared.physics_v2 import add_implicit_hull_damping, WaveClock

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .._shared.conditions import build_conditions
from .._shared.journal_env_mixin import JournalEnvMixin, device_copies
from .._shared.restoring import restoring_torque_body
from .._shared.scenario_draws import (
    make_scenario_rng,
    stamp_scenario,
    station_keeping_current,
    station_keeping_spawn,
)
from .._shared.vehicles import get_vehicle
from .current_ramp import ramped_current_vec
from .station_error_metrics import StationErrorAccumulator
from .station_keeping_env_cfg import StationKeepingEnvCfg, journal_d7_pool_for_seed, journal_pinned_pool_fields


class StationKeepingEnv(JournalEnvMixin, DirectRLEnv):
    """Calm-water station keeping with reward-independent success.

    Journal cfgs (``journal_contract_version == 2``, round F / F4) take the
    branch at the top of ``__init__``: the journal methods (``_journal_*``,
    below the legacy methods) are bound on the instance, so every legacy
    method source stays byte-identical (CONTEXT_R4 D15) and legacy ids never
    reach journal code.
    """

    cfg: StationKeepingEnvCfg
    # G20 / G20b appearing obstacle: the journal task of this family wired for it
    journal_appearing_tasks = ("station_keeping",)
    journal_family = "station_keeping"

    def __init__(self, cfg: StationKeepingEnvCfg, render_mode: str | None = None, **kwargs):
        if getattr(cfg, "journal_contract_version", None) == 2:
            self._journal_pre_init(cfg)
            super().__init__(cfg, render_mode, **kwargs)
            self._journal_post_init()
            return
        self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
        physics_version(self.physics_cfg)  # Reject unsupported versions before simulation.
        override = float(getattr(cfg, "current_speed_override_mps", 0.0))
        if override > 0.0:
            # Tier-regrade probe: one pinned current speed, force-enabled.
            self.physics_cfg.enable_current = True
            self.physics_cfg.current_speed_min = override
            self.physics_cfg.current_speed_max = override
        # Within-episode current-speed ramp (RampCurrent variant). Every cfg
        # that does not declare BOTH fields resolves to None here, and the
        # certified current path then runs byte-identical.
        ramp_start = getattr(cfg, "current_ramp_start_mps", None)
        ramp_end = getattr(cfg, "current_ramp_end_mps", None)
        if (ramp_start is None) != (ramp_end is None):
            raise ValueError(
                "current_ramp_start_mps and current_ramp_end_mps must be "
                "declared together"
            )
        self._current_ramp: tuple[float, float] | None = None
        if ramp_start is not None:
            if not self.physics_cfg.enable_current:
                raise ValueError(
                    "a current ramp requires enable_current=True (the ramp "
                    "modulates the sampled current, it does not create one)"
                )
            self._current_ramp = (float(ramp_start), float(ramp_end))
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
        # Longest CONTINUOUS hold achieved in the finished episode, in seconds.
        # Success is literally a threshold on this quantity (_get_dones compares
        # _max_hold_steps against required_hold_steps), so recording it exposes
        # the graded metric the binary criterion hides. Unlike path length it is
        # defined for EVERY episode, success or failure, so reading a dose
        # response off it needs no conditioning on the outcome -- which matters,
        # because conditioning on "both cells succeeded" was shown to flip the
        # SIGN of a p~1e-19 result on the deadband-PID wave ladder.
        self.episode_max_hold_s = torch.zeros(self.num_envs, device=self.device)

        self._hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
        # --- Controller-independent episode scenarios -----------------------
        # Spawn pose, current and wave used to come off the GLOBAL torch RNG,
        # which skrl's Runner reseeds to the agent YAML's constant AFTER the
        # env is built -- so --eval-seed 42 and --eval-seed 123 drew the same
        # episodes. They now come off a stream keyed by (protocol version,
        # cfg.seed, env index, per-env episode index, primitive group). With
        # cfg.seed None (an unseeded smoke run) make_scenario_rng returns None
        # and every draw falls back to the historical global-RNG line.
        self._scenario = make_scenario_rng(self.cfg, self.num_envs, self.device)
        # _scenario_* describe the RUNNING episode; episode_scenario_* are
        # latched from them at reset for the episode that just ENDED, the same
        # discipline as episode_path_length -- the evaluator reads these after
        # _reset_idx has already started the next episode.
        self._scenario_params = [{} for _ in range(self.num_envs)]
        self._scenario_hashes = [{} for _ in range(self.num_envs)]
        self.episode_scenario = [{} for _ in range(self.num_envs)]
        self.episode_scenario_hashes = [{} for _ in range(self.num_envs)]

        self._sea = None
        if getattr(self.cfg, "sea_state", None) is not None and self.cfg.sea_state.enable:
            from .._shared.sea_state import SeaState

            self._sea = SeaState(
                self.cfg.sea_state, self.num_envs, self.device,
                physics_version=getattr(self.physics_cfg, "physics_version", 1),
            )
            if getattr(self.physics_cfg, "physics_version", 1) == 2:
                self._wave_clock = WaveClock(self.num_envs, self.device, self.cfg.sim.dt)
        self._previous_xy = self.robot.data.root_pos_w[:, :2].clone()
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # --- Eval-time observation-channel degradation (OOD hooks) ----------
        # Every knob defaults to zero, in which case NO degrader object is
        # built and the guarded hooks in _get_observations never fire: the
        # frozen observation path stays byte-identical. Doses are meant to be
        # set per run via scripts/eval_v6_frozen.py --set, never on a
        # registered training id. Station keeping carries the goal-vector and
        # kinematics groups (no rays); the sea-state block stays clean.
        noise_pos = float(getattr(self.cfg, "obs_noise_sigma_pos", 0.0) or 0.0)
        noise_vel = float(getattr(self.cfg, "obs_noise_sigma_vel", 0.0) or 0.0)
        bias_pos = float(getattr(self.cfg, "obs_bias_sigma_pos", 0.0) or 0.0)
        bias_vel = float(getattr(self.cfg, "obs_bias_sigma_vel", 0.0) or 0.0)
        delay_steps = int(getattr(self.cfg, "obs_delay_steps", 0) or 0)
        dropout_p = float(getattr(self.cfg, "obs_dropout_p", 0.0) or 0.0)
        self._obs_degrader = None
        if any((noise_pos, noise_vel, bias_pos, bias_vel, delay_steps,
                dropout_p)):
            from .._shared.obs_degradation import (
                ObsChannelGroup,
                build_obs_degrader,
            )
            from .._shared.obs_superset import SPEED_SCALE_MPS

            # The yaw-rate channel rides the vel knob at the same fraction of
            # its full observation scale as the linear channels: sigma_yaw =
            # sigma_vel * (yaw_rate_obs_scale_rad_s / SPEED_SCALE_MPS).
            yaw_ratio = (
                float(getattr(self.cfg, "yaw_rate_obs_scale_rad_s", 1.0))
                / SPEED_SCALE_MPS
            )
            seed = getattr(self.cfg, "seed", None)
            self._obs_degrader = build_obs_degrader(
                num_envs=self.num_envs,
                device=self.device,
                base_seed=0 if seed is None else int(seed),
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
                },
                delay_steps=delay_steps,
                dropout_p=dropout_p,
            )

        # Additive measurement only: allocation happens after every existing
        # runtime component is built, and record() reads position only after the
        # existing reward/success/timeout calculations have finished.
        self._station_error = StationErrorAccumulator(
            num_envs=self.num_envs,
            max_episode_samples=int(self.max_episode_length),
            control_step_s=self.control_step_s,
            device=self.device,
        )
        self.episode_station_error_rms_m = self._station_error.episode_rms_m
        self.episode_station_error_p95_m = self._station_error.episode_p95_m
        self.episode_station_error_final_60s_rms_m = (
            self._station_error.episode_final_60s_rms_m
        )
        self.episode_station_error_final_60s_p95_m = (
            self._station_error.episode_final_60s_p95_m
        )

        # Opt-in journal suites. None bypasses every hook for legacy ids.
        self._conditions = build_conditions(
            self.cfg, self.num_envs, self.device, family="station_keeping"
        )
        self.episode_conditions = [{} for _ in range(self.num_envs)]

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

        if self.cfg.visual.enable_hold_zone_marker:
            try:
                self._create_hold_zone_markers()
            except Exception as exc:
                print(f"[WARN] Hold-zone visualization could not be created: {exc}")

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

    def _create_hold_zone_markers(self) -> None:
        """Create one static, render-only annulus at each environment origin."""
        import math

        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        segments = int(self.cfg.visual.hold_zone_segments)
        radius = float(self.cfg.hold_radius)
        line_width = float(self.cfg.visual.hold_zone_line_width_m)
        if segments < 3:
            raise ValueError("visual.hold_zone_segments must be at least 3")
        if radius <= 0.0:
            raise ValueError("hold_radius must be positive to render its marker")
        if line_width <= 0.0:
            raise ValueError("visual.hold_zone_line_width_m must be positive")

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
                (
                    inner,
                    outer,
                    next_outer,
                    inner,
                    next_outer,
                    next_inner,
                )
            )

        stage = omni.usd.get_context().get_stage()
        point_array = Vt.Vec3fArray(points)
        count_array = Vt.IntArray(face_counts)
        index_array = Vt.IntArray(face_indices)
        color = Gf.Vec3f(*self.cfg.visual.hold_zone_color)
        colors = Vt.Vec3fArray([color] * len(points))
        opacities = Vt.FloatArray([1.0] * len(points))

        for env_index in range(self.num_envs):
            marker_path = f"/World/envs/env_{env_index}/HoldZone"
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

    def _resample_current(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one constant world-frame current vector per reset environment.

        Same uniform speed on [current_speed_min, current_speed_max) and same
        uniform direction on [0, 2*pi) as before; only the stream the two
        uniforms come from changed (see station_keeping_current).

        Returns the RAW ``(speeds, directions)`` it drew, before the sine and
        cosine, so the caller can stamp the draw itself rather than the vector
        derived from it. current_vec is written exactly as before, so the
        physics path is untouched.
        """
        speeds, directions = station_keeping_current(
            self._scenario,
            env_ids,
            self.device,
            self.physics_cfg.current_speed_min,
            self.physics_cfg.current_speed_max,
        )
        self.current_vec[env_ids, 0] = speeds * torch.cos(directions)
        self.current_vec[env_ids, 1] = speeds * torch.sin(directions)
        return speeds, directions

    def _compute_current_forces(self) -> torch.Tensor:
        """Compute quadratic relative-velocity drag in the world frame."""
        vel_w = self.robot.data.root_com_vel_w
        boat_vel_xy = vel_w[:, :2]
        current_vec = self.current_vec
        if self._current_ramp is not None:
            # RampCurrent variant: keep the per-episode direction, override
            # the magnitude with a linear within-episode ramp that reaches
            # ramp_end on the final control step. self.current_vec stays the
            # sampled base vector (the direction carrier), so the reset-time
            # Episode/current_speed log reports the SAMPLED base magnitude,
            # not the instantaneous ramped speed.
            ramp_start_mps, ramp_end_mps = self._current_ramp
            current_vec = ramped_current_vec(
                current_vec,
                self.episode_length_buf,
                max(int(self.max_episode_length) - 1, 1),
                ramp_start_mps,
                ramp_end_mps,
            )
        v_rel = boat_vel_xy - current_vec
        v_rel_mag = torch.norm(v_rel, dim=1, keepdim=True)
        drag_force_xy = -self.physics_cfg.current_drag_coeff * v_rel * v_rel_mag

        drag_force = torch.zeros((self.num_envs, 3), device=self.device)
        drag_force[:, :2] = drag_force_xy
        return drag_force

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # HARD action clip: the declared [-1,1] range is enforced here.
        self.actions = torch.clamp(actions, -1.0, 1.0)
        if self._conditions is not None:
            self.actions = self._conditions.action_in(self.actions)
            self._conditions.reset_runtime(progress=None)

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
        yaw_command = self.actions[:, 1] * self.cfg.yaw_torque_max
        if self._conditions is not None:
            thrust_magnitude, yaw_command = self._conditions.command(
                thrust_magnitude, yaw_command, self.cfg.sim.dt, None
            )
        forces[:, 0, 0] = thrust_magnitude * self._fwd_x
        forces[:, 0, 1] = thrust_magnitude * self._fwd_y
        torques[:, 0, 2] = yaw_command

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

        quat = self.robot.data.root_link_quat_w
        # D1 merge (C2 x C4): v1 keeps the explicit legacy drag, including the
        # C4 condition drag multiplier. v2 applies NO explicit hull drag here;
        # add_implicit_hull_damping below owns every hull damping term and
        # takes the condition drag multiplier from hull_damping_scales().
        if getattr(self.physics_cfg, "physics_version", 1) == 1:
            if (
                self.physics_cfg.sway_lin_damping is None
                or self.physics_cfg.sway_quad_damping is None
            ):
                # Preserve the legacy ROV's isotropic horizontal damping exactly.
                v_xy = linear_velocity[:, :2]
                speed_xy = torch.norm(v_xy, dim=-1, keepdim=True)
                planar_drag = -(
                    self.physics_cfg.surge_lin_damping
                    + self.physics_cfg.surge_quad_damping * speed_xy
                ) * v_xy
                if self._conditions is not None:
                    planar_drag = self._conditions.scale_drag(planar_drag)
                force_w[:, :2] += planar_drag
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
                if self._conditions is not None:
                    drag_b[:, :2] = self._conditions.scale_drag(drag_b[:, :2])
                forces[:, 0, :2] += drag_b[:, :2]
            force_w[:, 2] += -self.physics_cfg.heave_damping * linear_velocity[:, 2]

            wz = angular_velocity[:, 2]
            yaw_drag = -(
                self.physics_cfg.yaw_lin_damping
                + self.physics_cfg.yaw_quad_damping * torch.abs(wz)
            ) * wz
            if self._conditions is not None:
                yaw_drag = self._conditions.scale_drag(yaw_drag)
            torque_w[:, 2] += yaw_drag
            torque_w[:, :2] += -self.physics_cfg.rollpitch_rate_damping * angular_velocity[:, :2]

        # Yaw-invariant attitude spring: restoring torque k*(up_body_in_world x
        # world_up). The previous Euler-angle form applied BODY tilt angles as
        # FIXED world-axis torques, which is restoring only near the spawn yaw
        # and becomes precessing/anti-restoring past ~90 deg (capsize-by-turning).
        if (
            self.physics_cfg.restoring_stiffness_roll
            == self.physics_cfg.restoring_stiffness_pitch
        ):
            # Preserve the legacy equal-stiffness ROV restoring path exactly.
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

        # Match blueboat_calm_nav: current drag is a world-frame force and is
        # added before the shared world-to-body rotation. Calm tasks do not
        # enter the current-force path at all.
        if self.physics_cfg.enable_current:
            force_w += self._compute_current_forces()
        if self._sea is not None:
            # Same world-frame entry point as the current, so a task may carry
            # both and their water velocities simply add.
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

        if self._conditions is not None:
            # Physical bow azimuth (registry bow axis via forward_vec) for the
            # wind suite; the hull COM velocity is the air-relative input.
            bow_w = math_utils.quat_apply(quat, self.forward_vec)
            condition_force, condition_torque = self._conditions.environmental_wrench(
                self.robot.data.root_com_pos_w[:, :2],
                self.robot.data.root_com_vel_w[:, :2],
                self.cfg.sim.dt,
                orientation_wxyz=quat,
                angular_velocity_w=angular_velocity,
                heading_rad=torch.atan2(bow_w[:, 1], bow_w[:, 0]),
            )
            force_w += condition_force
            torque_w += condition_torque
        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_w)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_w)
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

    def _horizontal_distance(self) -> torch.Tensor:
        return torch.norm(self.hold_point - self.robot.data.root_pos_w[:, :2], dim=-1)

    def _get_observations(self) -> dict:
        degrader = self._obs_degrader
        if self._conditions is not None and any(
            self._conditions.wants(group) for group in ("pos", "vel")
        ):
            degrader = self._conditions
        forwards = math_utils.quat_apply(
            self.robot.data.root_link_quat_w,
            self.forward_vec,
        )

        rpos = self.hold_point - self.robot.data.root_pos_w[:, :2]
        if degrader is not None and degrader.wants("pos"):
            # OOD hook: corrupt the assembled hold-point vector (ideal
            # relative-position measurement, meters) before distance and
            # bearing features are derived from it.
            rpos = degrader.apply("pos", rpos)
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

        observation = torch.hstack([dot, cross, distance_norm])
        if getattr(self.cfg, "obs_sea_state", False) and self._sea is not None:
            # Bow-relative sea direction plus normalised significant height,
            # encoded the same way the goal direction is, so the policy reads
            # "where the sea comes from" with the convention it already knows.
            observation = torch.hstack(
                (observation, self._sea.observation(forwards_2d))
            )
        if getattr(self.cfg, "obs_kinematic", False):
            # Shared v11-style block: body-frame surge, sway, yaw rate. A
            # scalar speed cannot tell "driving forward" from "sliding
            # sideways" from "still spinning"; station keeping needs all three.
            from .._shared.kinematics import body_planar_kinematics

            if degrader is not None and degrader.wants(
                "vel"
            ):
                # OOD hook: corrupt the raw body-frame rates (m/s, rad/s)
                # between measurement and normalization; applied exactly once
                # per step because delay/dropout are stateful.
                from .._shared.kinematics import (
                    body_planar_rates,
                    normalize_planar_rates,
                )

                observation = torch.hstack(
                    (
                        observation,
                        normalize_planar_rates(
                            degrader.apply(
                                "vel",
                                body_planar_rates(
                                    forwards_2d,
                                    self.robot.data.root_com_vel_w[:, :3],
                                    self.robot.data.root_ang_vel_w[:, 2],
                                ),
                            ),
                            yaw_rate_scale_rad_s=(
                                self.cfg.yaw_rate_obs_scale_rad_s
                            ),
                        ),
                    )
                )
            else:
                observation = torch.hstack(
                    (
                        observation,
                        body_planar_kinematics(
                            forwards_2d,
                            self.robot.data.root_com_vel_w[:, :3],
                            self.robot.data.root_ang_vel_w[:, 2],
                            yaw_rate_scale_rad_s=self.cfg.yaw_rate_obs_scale_rad_s,
                        ),
                    )
                )
        return {"policy": observation}

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
        self._max_hold_steps.copy_(
            torch.maximum(self._max_hold_steps, self._hold_steps)
        )
        self.hold_timer.copy_(self._hold_steps * self.control_step_s)

        achieved_now = (
            ~self._success & (self._max_hold_steps >= self.required_hold_steps)
        )
        elapsed_s = self.episode_length_buf.float() * self.control_step_s
        self._first_success_time_s.copy_(
            torch.where(achieved_now, elapsed_s, self._first_success_time_s)
        )
        self._success.copy_(self._max_hold_steps >= self.required_hold_steps)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._episode_finished.copy_(time_out)
        self._station_error.record(self._horizontal_distance())
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
            for completed in completed_ids.tolist():
                if self._conditions is not None:
                    self.episode_conditions[completed] = (
                        self._conditions.episode_stamp(completed)
                    )
                self.episode_scenario[completed] = self._scenario_params[completed]
                self.episode_scenario_hashes[completed] = self._scenario_hashes[
                    completed
                ]

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
            self.extras["log"]["Episode/path_length_m"] = self.path_length[completed_ids].mean()
            self.extras["log"]["Episode/final_hold_timer_s"] = self.hold_timer[completed_ids].mean()
            if self.physics_cfg.enable_current:
                self.extras["log"]["Episode/current_speed"] = torch.norm(
                    self.current_vec[completed_ids], dim=1
                ).mean()
            self._station_error.latch(completed_ids)

        super()._reset_idx(env_ids)

        # A new episode for these envs: advance their per-env episode counter
        # and reseed every primitive stream from the new key. This MUST run
        # before the first draw below.
        if self._scenario is not None:
            self._scenario.reset_idx(env_ids)

        # Keep the RAW (speed, direction) pair, not just the vector it becomes:
        # the certificate stamps the DRAW (see the scenario_groups block below).
        current_speeds = None
        current_directions = None
        if self.physics_cfg.enable_current:
            current_speeds, current_directions = self._resample_current(env_ids)

        num_resets = len(env_ids)
        # Same uniform distance on [min_spawn_distance, max_spawn_distance) and
        # same uniform angle/heading on [0, 2*pi); only the stream changed.
        distances, spawn_angles, headings = station_keeping_spawn(
            self._scenario,
            env_ids,
            self.device,
            self.cfg.min_spawn_distance,
            self.cfg.max_spawn_distance,
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 0] += distances * torch.cos(spawn_angles)
        root_state[:, 1] += distances * torch.sin(spawn_angles)
        body_yaws = headings + self._body_yaw_from_bow_offset
        root_state[:, 3:7] = math_utils.quat_from_angle_axis(
            body_yaws.unsqueeze(-1), self.up_dir
        ).reshape(num_resets, 4)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids)

        self._hold_steps[env_ids] = 0
        self._max_hold_steps[env_ids] = 0
        self._first_success_time_s[env_ids] = torch.nan
        self.hold_timer[env_ids] = 0.0
        self.path_length[env_ids] = 0.0
        self._station_error.reset(env_ids)
        self._success[env_ids] = False
        self._episode_finished[env_ids] = False
        self._previous_xy[env_ids] = root_state[:, :2]
        if self._sea is not None:
            self._sea.resample(env_ids, scenario=self._scenario)
            if getattr(self.physics_cfg, "physics_version", 1) == 2:
                self._wave_clock.reset(env_ids)
        if self._obs_degrader is not None:
            # New episode: bump the (env, episode) RNG stream, resample the
            # per-episode bias, and clear delay/dropout buffers. The clean
            # initial frame is captured on the next _get_observations call.
            self._obs_degrader.reset(env_ids)

        # Resolved scenario for the episode that starts now, plus a hash per
        # primitive group for the certificate. Values only -- never the key --
        # so two eval seeds that drew the same numbers still hash the same and
        # stay detectable by scripts/check_scenario_independence.py.
        #
        # Every entry is a RAW DRAW, never a kernel-derived quantity. The
        # current used to be stamped as current_vec[env_ids], which is the draw
        # AFTER the trigonometry (:410-411). Two reasons that is the wrong
        # thing to hash:
        #
        #   * It is not injective, and not only in a corner case. At speed zero
        #     -- nothing in the cfg schema forbids current_speed_min = 0.0 --
        #     EVERY direction collapses onto the same (0, 0) vector. The
        #     collision is LIVE inside the shipped range as well: at speed
        #     0.2 m/s the directions 0.5000017285346985 and 0.5000017881393433
        #     (adjacent float32 values, each reachable from a float32 unit draw
        #     through the *2*pi of station_keeping_current) give a
        #     byte-identical float32 current_vec. Sweeping the real chain --
        #     unit speed and unit direction drawn as here, scaled to [0.2, 0.3)
        #     and [0, 2*pi) -- 8.4e6 neighbouring draw pairs whose RECORDED
        #     direction differs produced the same float32 vector in 1.09% of
        #     cases, at speeds spread across the whole range.
        #     An earlier revision of this comment reported that 4e6 randomly
        #     sampled (speed, direction) pairs gave 4e6 distinct vectors and
        #     concluded "a robustness argument rather than a live collision".
        #     That measurement was sound but answered the wrong question: far
        #     apart draws almost never collide (a birthday bound), which says
        #     nothing about whether the map is injective. It is not, and the
        #     old conclusion was false.
        #     What the 1.09% is NOT: the chance that two episodes of one
        #     certificate share a hash. Two independent episodes draw
        #     directions nowhere near each other, so that stays vanishingly
        #     rare. The point is only that injectivity is not available as an
        #     argument, so the certificate must not rest on it.
        #   * It welds the certificate to the kernel. current_vec is also the
        #     direction carrier the RampCurrent variant modulates (:419-427);
        #     any future change to how the vector is formed from the draw would
        #     move every episode hash while the drawn numbers stayed identical,
        #     which reads as "these two runs saw different scenarios" when they
        #     did not.
        #
        # Speed and direction are what the stream produced, and current_vec is
        # recoverable from them, so nothing is lost.
        scenario_groups = {
            "spawn_pose": {
                "distance_m": distances,
                "angle_rad": spawn_angles,
                "heading_rad": headings,
            }
        }
        if self.physics_cfg.enable_current:
            scenario_groups["current"] = {
                "speed_mps": current_speeds,
                "direction_rad": current_directions,
            }
        if self._sea is not None:
            # H_s, T_p, gamma, the mean direction and the phases are each an
            # affine image of one unit draw (tasks/_shared/sea_state.py:133-137)
            # and so ARE the draw. direction_rad is mean_direction + spread
            # (:138-141), but mean_direction_rad is stamped beside it, so the
            # per-component spread draw is recoverable by subtraction and the
            # pair still identifies the draw uniquely. amplitude is omitted on
            # purpose: it is a pure function of (hs, tp, gamma), which are
            # already here, so stamping it would add no information.
            scenario_groups["wave"] = {
                "hs_m": self._sea.hs[env_ids],
                "tp_s": self._sea.tp[env_ids],
                "gamma": self._sea.gamma[env_ids],
                "mean_direction_rad": self._sea.mean_direction[env_ids],
                "phase_rad": self._sea.phase[env_ids],
                "direction_rad": self._sea.direction[env_ids],
            }
        for env_index, resolved, hashes in stamp_scenario(env_ids, scenario_groups):
            self._scenario_params[env_index] = resolved
            self._scenario_hashes[env_index] = hashes
        if self._conditions is not None:
            # Task reference axis for relative wind directions: the bearing
            # from this episode's spawn point to the hold point.
            self._conditions.reset_runtime(
                env_ids=env_ids,
                reference_axis_rad=spawn_angles + math.pi,
                task_episode_ids=(
                    None if self._scenario is None
                    else self._scenario.episode_indices()[env_ids.cpu()]
                ),
            )

    # ======================================================================
    # Journal (task contract v2) methods, round F / F4. Bound on the instance
    # by the journal branch of __init__ (the D3 station-proof pattern); legacy
    # cfgs never reach them. Call order = tasks/_shared/journal_env_mixin.py.
    # Truth (root-link pose = COM for v2 assets, CONTEXT_R3 decision 12) for
    # rewards, termination and metrics; the own-state estimates (D5) for the
    # observation. Waves, current and wind are never observed.
    # ======================================================================
    _JOURNAL_BOUND = ("_setup_scene", "_pre_physics_step", "_apply_action", "_get_observations",
                      "_get_rewards", "_get_dones", "_reset_idx")

    def _journal_pre_init(self, cfg) -> None:
        """Checks + vessel resolution BEFORE DirectRLEnv.__init__ builds the asset."""
        import os
        from pathlib import Path

        from .._shared.journal_vehicle import ASSETS_DIRNAME, resolve_journal_vehicle
        from .station_keeping_env_cfg import _REPO_ROOT

        if getattr(cfg, "journal_task", None) != "station_keeping":
            raise ValueError("StationKeepingEnv carries only the journal task 'station_keeping'")
        for name in ("current_ramp_start_mps", "current_ramp_end_mps"):
            if getattr(cfg, name, None) is not None:
                raise ValueError("journal station keeping never declares the legacy current ramp; the "
                                 "_current_ramp path is legacy-only (CONTEXT_R4 D6, ANCHORS_E5 C3): " + name)
        physics = cfg.underwater_physics_cfg
        if getattr(physics, "hydrodynamics_model", "legacy") != "hydro_v2" or \
                int(getattr(physics, "physics_version", 1)) != 2:
            raise ValueError("journal station keeping runs hydrodynamics_model='hydro_v2' on physics_version 2")
        if getattr(physics, "enable_current", False) or float(getattr(cfg, "current_speed_override_mps", 0.0) or 0.0):
            raise ValueError("journal current is condition-layer water velocity; legacy current must be off")
        if getattr(getattr(cfg, "sea_state", None), "enable", False) or getattr(cfg, "obs_sea_state", False):
            raise ValueError("journal station keeping has no legacy sea state and observes no sea")
        if not getattr(cfg, "obs_kinematic", False) or int(cfg.observation_space) != 6 + int(cfg.ray_count):
            raise ValueError("journal station-keeping observation = kinematic 6 + the 36-ray block")
        options = dict(getattr(cfg, "journal_vehicle_options", None) or {})
        vehicle = resolve_journal_vehicle(cfg.vehicle, **options)
        asset = vehicle.verify_asset(os.path.join(_REPO_ROOT, ASSETS_DIRNAME))
        if Path(str(cfg.robot_cfg.spawn.usd_path)).resolve() != Path(asset).resolve():
            raise ValueError("journal robot asset %s is not the verified v2 asset %s"
                             % (cfg.robot_cfg.spawn.usd_path, asset))
        self.vehicle_spec = None
        self.physics_cfg = physics
        self._journal_vehicle_pre = vehicle
        for name in self._JOURNAL_BOUND:
            setattr(self, name, getattr(self, "_journal" + name))

    def _journal_post_init(self) -> None:
        """After the scene exists: D7 pool, mixin setup, task buffers (fail closed)."""
        from .._shared.scenario_rng import ScenarioRNG
        from . import journal_station as js

        cfg = self.cfg
        n, device = int(self.num_envs), self.device
        self.control_step_s = cfg.sim.dt * cfg.decimation
        self._jplan = cfg.journal_plan
        self._jparams = self._jplan.params
        self._jtier = int(cfg.journal_geometry_tier)
        # G49 (J29): the INDEX.json pool of the run's evaluation seed (the evaluator sets cfg.seed after
        # __post_init__ resolved the v2.6.4 entry); the cfg is updated so it names the pool actually loaded
        pool, pool_sha256 = journal_d7_pool_for_seed(cfg, self._jtier)
        if isinstance(pool, str) and (pool, pool_sha256) != (cfg.journal_d7_pool, cfg.journal_d7_pool_sha256):
            cfg.journal_d7_pool, cfg.journal_d7_pool_sha256 = pool, pool_sha256
        if pool is None:
            raise ValueError("no admitted D7 pool for %s tier %d (cfg.journal_d7_pool); run "
                             "work/tools_f4/admit_d7_pool.py (D7 B02/B05)" % (cfg.vehicle, self._jtier))
        if not callable(getattr(pool, "assignment", None)):
            pool = js.StationD7Pool.load(pool, getattr(cfg, "journal_d7_pool_sha256", None))
        pool.check_env(vessel=cfg.vehicle, seed=getattr(cfg, "seed", None), tier=self._jtier, num_envs=n)
        self.journal_setup(d7_pool=pool)
        # Round T (CONTEXT_R4 D16, D20, D22, D24): not-admitted and beyond-grid rows run the
        # dynamic suite at D0 through the family wrapper of the E4 bridge; header counts are
        # [offered, admitted, not_admitted] and the header publishes the pool's offered grid.
        self._d7_bridge = js.StationExposureBridge(self._d7_bridge)
        self.dynamic_hazard_header.update(pool.header_fields())
        self.dynamic_hazard_header.update(journal_pinned_pool_fields(pool, cfg.vehicle, self._jtier,
                                                                     getattr(cfg, "seed", None)))
        if self.journal_vehicle.name != self._journal_vehicle_pre.name:
            raise ValueError("journal vessel changed between asset build and runtime setup")
        pool.check_env(vessel=cfg.vehicle, seed=cfg.seed, tier=self._jtier, num_envs=n, suite=self._d7_suite)
        if self._conditions.cfg.current.speed_table:
            raise ValueError("journal station-keeping current is CONSTANT from t = 0 (CONTEXT_R4 D6, "
                             "StationKeepingEpisode.top_dose_start_s = 0): speed_table rejected")
        self._journal_check_com_offset()
        self._jscenario = ScenarioRNG(num_envs=n, device="cpu", eval_seed=int(cfg.seed))
        self._jnominal = float(self._jplan.nominal_com_height_m)
        self._jcom_z = float(self.physics_cfg.water_surface_z) + float(
            self.journal_vehicle.hydro_profile.hydro.equilibrium_com_z_m)
        self._jtwin = js.StationKeepingTwin(n, device, **self._jplan.validity_limits)
        f64 = dict(dtype=torch.float64, device=device)
        self._jlayouts = [None] * n
        self._jspawn_hash = [None] * n
        self._jstation = torch.zeros((n, 2), **f64)
        self._jradius = torch.ones(n, **f64)
        # G6: the station centre published under the legacy name (station_keeping_env.py legacy
        # __init__: hold_point = env origin, WORLD frame, float32), read by classical_baseline /
        # journal_controllers, eval_mppi (mode "hold") and _horizontal_distance (cert_v3 d0);
        # set per episode in the journal reset from the env-local _jstation.
        self.hold_point = self.scene.env_origins[:, :2].clone()
        self._jbounds = torch.zeros((n, 4), **f64)
        self._jdeadline_steps = torch.zeros(n, dtype=torch.long, device=device)
        self._jsettle = torch.zeros(n, **f64)
        # D7 storage (B04/B07): station keeping has no static obstacles, so the
        # storage is exactly the one reserved D7 slot; physical/visible masks split.
        self._d7_centers_w = torch.zeros((n, 1, 2), device=device)
        self._d7_radii = torch.zeros((n, 1), device=device)
        self._d7_physical = torch.zeros((n, 1), dtype=torch.bool, device=device)
        self._d7_visible = torch.zeros((n, 1), dtype=torch.bool, device=device)
        if self._appearing_bridge is not None:
            # G20: the appearing cylinder's own storage (slot 0 + D7 1 = 1, CapacityPlan.appearing_slots),
            # world xy like the D7 slot; the D7 storage above is untouched (no D7 write can overwrite it)
            self._appearing_centers_w = torch.zeros((n, 1, 2), device=device)
            self._appearing_radii = torch.zeros((n, 1), device=device)
            self._appearing_physical = torch.zeros((n, 1), dtype=torch.bool, device=device)
            self._appearing_visible = torch.zeros((n, 1), dtype=torch.bool, device=device)
            self.appearing_hazard_header["collision_model"] = (
                "station keeping has no static obstacles: the appearing swept-contact latch (D7 swept guard) is "
                "OR'ed into the contract collision input like the D7 contact; rays use the point-return proxy over "
                "the visible D7 and appearing slots")
        # completed-episode latches (written in _get_dones BEFORE any reset)
        self.episode_success = torch.zeros(n, dtype=torch.bool, device=device)
        self.time_to_success = torch.full((n,), torch.nan, device=device)
        self.episode_path_length = torch.zeros(n, device=device)
        self.episode_max_hold_s = torch.zeros(n, device=device)
        self.episode_station_error_final_60s_rms_m = torch.full((n,), torch.nan, device=device)
        self.episode_termination_code = torch.zeros(n, dtype=torch.long, device=device)
        self.journal_episode_results = [None] * n
        self.journal_contract_trace = None       # set to a list to record the oracle inputs (tests)
        self.actions = torch.zeros((n, self.cfg.action_space), device=device)

    def _journal_check_com_offset(self) -> None:
        """CONTEXT_R3 decision 12: v2 assets have a zero COM offset (checked when readable)."""
        data = self.robot.data
        com = getattr(data, "body_com_pos_b", None)
        if com is None:
            pose = getattr(data, "body_com_pose_b", None)
            com = None if pose is None else pose[..., :3]
        if com is None:
            self.journal_com_offset_check = "UNVERIFIED: no body COM buffer on robot.data"
            return
        if float(torch.as_tensor(com).abs().max()) > 1.0e-4:
            raise ValueError("v2 asset COM offset must be zero: root-link pose = COM (CONTEXT_R3 decision 12)")
        self.journal_com_offset_check = "zero COM offset verified"

    def _journal_setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.rigid_objects["robot"] = self.robot
        if self.cfg.visual.enable_water:
            try:
                self._create_static_water_mesh()
            except Exception as exc:
                print(f"[WARN] Static water visualization could not be created: {exc}")
        if self.cfg.visual.enable_hold_zone_marker:
            try:
                self._journal_create_hold_zone_markers()
            except Exception as exc:
                print(f"[WARN] Hold-zone visualization could not be created: {exc}")
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        # v2 assets: bow +x, origin at the COM (vehicles_v2 validation)
        self.forward_vec = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.num_envs, 1)
        self.current_vec = torch.zeros((self.num_envs, 2), device=self.device)

    def _journal_create_hold_zone_markers(self) -> None:
        """Render-only annulus at the plan's station point (fixed per plan: (approach, 0) env-local)."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        params = self.cfg.journal_plan.params
        centre_x = float(params["approach_distance_m"])
        radius = float(params["hold_radius_m"])
        segments = int(self.cfg.visual.hold_zone_segments)
        width = float(self.cfg.visual.hold_zone_line_width_m) * float(params["length_scale"])
        marker_z = float(self.physics_cfg.water_surface_z) + 0.02
        points, counts, indices = [], [], []
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            c, s = math.cos(angle), math.sin(angle)
            points.extend((Gf.Vec3f(centre_x + radius * c, radius * s, marker_z),
                           Gf.Vec3f(centre_x + (radius + width) * c, (radius + width) * s, marker_z)))
            nxt = (segment + 1) % segments
            counts.extend((3, 3))
            indices.extend((2 * segment, 2 * segment + 1, 2 * nxt + 1, 2 * segment, 2 * nxt + 1, 2 * nxt))
        stage = omni.usd.get_context().get_stage()
        color = Gf.Vec3f(*self.cfg.visual.hold_zone_color)
        for env_index in range(self.num_envs):
            path = f"/World/envs/env_{env_index}/HoldZone"
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
            mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color] * len(points)))
            mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
            mesh.GetDisplayOpacityAttr().Set(Vt.FloatArray([1.0] * len(points)))
            mesh.GetDoubleSidedAttr().Set(True)

    # ------------------------------------------------------------ physics --
    def _journal_pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.journal_pre_physics(actions)      # clamp to [-1, 1] once (D13)

    def _journal_apply_action(self) -> None:
        """hydro_v2 force block through the mixin (D3 section 2; DESIGN_E3 section 3)."""
        forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins,
                                               float(self.physics_cfg.water_surface_z))
        dtype = self.robot.data.root_link_pose_w.dtype
        from .._shared.body_wrench import apply_body_wrench   # G6: no per-call deprecation warning
        apply_body_wrench(self.robot, forces.to(dtype), torques.to(dtype))

    def _journal_state(self, robot_data, env_origins, water_level):
        from .journal_station import TwinState

        pose = robot_data.root_link_pose_w
        vel = robot_data.root_com_vel_w
        n = int(self.num_envs)
        com = (pose[:, :3] - env_origins[:, :3]).to(torch.float64)
        water = torch.as_tensor(water_level, dtype=torch.float64, device=com.device).expand(n).clone()
        return TwinState(com, pose[:, 3:7].to(torch.float64), vel[:, :3].to(torch.float64),
                         vel[:, 3:6].to(torch.float64), water,
                         torch.full((n,), self._jnominal, dtype=torch.float64, device=com.device))

    def _journal_observe_physics(self, robot_data, env_origins, water_level_local) -> None:
        """Mixin hook: contract observe_physics on the PREVIOUS substep's state (D8 water level)."""
        state = self._journal_state(robot_data, env_origins, water_level_local)
        if self.journal_contract_trace is not None:
            self.journal_contract_trace.append(("observe", state))
        self._jtwin.observe_physics(state)

    def _journal_publish_d7(self, samples) -> None:
        """Mixin hook (D7 B07): write the D7 samples into the reserved slot (world xy)."""
        if not samples:
            return
        rows, xy, radius, physical, visible, empty = [], [], [], [], [], []
        origins = self.scene.env_origins
        for env, items in samples.items():
            if not items:
                empty.append(int(env))
                continue
            if len(items) != 1 or int(items[0].slot) != 0:
                raise ValueError("station keeping stores exactly one D7 object in slot 0 (capacity 0 + 1)")
            item = items[0]
            rows.append(int(env))
            xy.append((float(item.x), float(item.y)))
            radius.append(float(item.radius_m))
            physical.append(bool(item.physical))
            visible.append(bool(item.visible))
        if empty and sorted(empty) == list(range(int(self.num_envs))):
            # G11: every row empty -> the same elements written by basic slicing (fill kernels; the index
            # tensor and the three scalar index_puts each cost a host sync before)
            self._d7_physical[:, 0] = False
            self._d7_visible[:, 0] = False
            self._d7_radii[:, 0] = 0.0
        elif empty:
            ids = torch.as_tensor(empty, dtype=torch.long, device=self.device)
            self._d7_physical[ids, 0] = False
            self._d7_visible[ids, 0] = False
            self._d7_radii[ids, 0] = 0.0
        if rows:
            # G11: the five host tensors of the per-field as_tensor(..., device=) calls, built on the host
            # exactly as those calls build them, moved in ONE host-to-device copy (bytes verbatim)
            ids, local, radius_t, physical_t, visible_t = device_copies(
                [torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(xy, dtype=self._d7_centers_w.dtype),
                 torch.as_tensor(radius, dtype=self._d7_radii.dtype), torch.as_tensor(physical, dtype=torch.bool),
                 torch.as_tensor(visible, dtype=torch.bool)], self.device)
            self._d7_centers_w[ids, 0] = local + origins[ids, :2].to(local.dtype)
            self._d7_radii[ids, 0] = radius_t
            self._d7_physical[ids, 0] = physical_t
            self._d7_visible[ids, 0] = visible_t

    def _journal_publish_appearing(self, samples) -> None:
        """G20 mixin hook: write the appearing sample (one slot) into its own storage; physical = visible."""
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
        local = torch.as_tensor(xy, dtype=self._appearing_centers_w.dtype, device=self.device)
        self._appearing_centers_w[ids, 0] = local + self.scene.env_origins[ids, :2].to(local.dtype)
        self._appearing_radii[ids, 0] = torch.as_tensor(radius, dtype=self._appearing_radii.dtype, device=self.device)
        self._appearing_physical[ids, 0] = torch.as_tensor(physical, dtype=torch.bool, device=self.device)
        self._appearing_visible[ids, 0] = torch.as_tensor(visible, dtype=torch.bool, device=self.device)

    def _journal_appearing_geometry(self, env_id) -> str:
        """G20 mixin hook: geometry id of the env's current E5 layout (the appearing pool records it per key)."""
        layout = self._jlayouts[int(env_id)]
        if layout is None:
            raise RuntimeError("env %d has no journal layout" % int(env_id))
        return str(layout.d7_hook["geometry_id"])

    def _journal_family_scenario_hashes(self, env_id):
        """Mixin hook (D11): family primitives layout + spawn_pose (exogenous only)."""
        from .journal_station import layout_hash

        i = int(env_id)
        return {"layout": layout_hash(self._jlayouts[i]), "spawn_pose": self._jspawn_hash[i]}

    # -------------------------------------------------------- observations --
    def _journal_ray_ranges(self) -> torch.Tensor:
        """Truth 36-ray block (m) from the COM: D7 VISIBLE objects only (B07), point-return proxy."""
        from .journal_station import point_return_rays, yaw_from_quat

        pose = self.robot.data.root_link_pose_w
        if getattr(self, "_appearing_bridge", None) is None:
            return point_return_rays(pose[:, :2], yaw_from_quat(pose[:, 3:7]), self._d7_centers_w,
                                     self._d7_radii, self._d7_visible, int(self.cfg.ray_count),
                                     float(self.cfg.ray_max_range_m))
        # G20: the appearing slot joins the point-return proxy (invisible until triggered)
        return point_return_rays(pose[:, :2], yaw_from_quat(pose[:, 3:7]),
                                 torch.cat((self._d7_centers_w, self._appearing_centers_w), 1),
                                 torch.cat((self._d7_radii, self._appearing_radii), 1),
                                 torch.cat((self._d7_visible, self._appearing_visible), 1),
                                 int(self.cfg.ray_count), float(self.cfg.ray_max_range_m))

    def _journal_get_observations(self) -> dict:
        """kinematic 6 + 36 rays; position-derived channels from est.position (D5)."""
        from .._shared.kinematics import normalize_planar_rates
        from .journal_station import body_rates, bow_axis_xy

        data = self.robot.data
        pose = data.root_link_pose_w
        com_local = pose[:, :2] - self.scene.env_origins[:, :2]
        est = self.journal_estimates(com_local, body_rates(pose[:, 3:7], data.root_com_vel_w),
                                     self._journal_ray_ranges())
        position = est.position.to(torch.float32)
        rpos = self._jstation.to(torch.float32) - position
        distance = torch.norm(rpos, dim=-1, keepdim=True).clamp(min=1.0e-6)
        direction = rpos / distance
        forwards_2d = bow_axis_xy(pose[:, 3:7].to(torch.float32))
        dot = torch.sum(forwards_2d * direction, dim=-1, keepdim=True)
        cross = forwards_2d[:, 0:1] * direction[:, 1:2] - forwards_2d[:, 1:2] * direction[:, 0:1]
        distance_norm = distance / float(self.cfg.max_spawn_distance)
        kinematics = normalize_planar_rates(est.velocity.to(torch.float32),
                                            yaw_rate_scale_rad_s=self.cfg.yaw_rate_obs_scale_rad_s)
        rays = est.rays.to(torch.float32) / float(self.cfg.ray_max_range_m)
        return {"policy": torch.cat((dot, cross, distance_norm, kinematics, rays), dim=-1)}

    # ------------------------------------------------------- reward / done --
    def _journal_get_rewards(self) -> torch.Tensor:
        """Reference baseline only (truth COM); methods may use any reward shaping."""
        com = self.robot.data.root_link_pose_w[:, :2] - self.scene.env_origins[:, :2]
        distance = torch.norm(self._jstation.to(com.dtype) - com, dim=-1)
        inside = distance <= self._jradius.to(distance.dtype)
        return -distance / float(self.cfg.reference_reward_distance_scale) + inside.float()

    def _journal_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """D7 final step, tensor-twin contract step, D9 publisher + latch before reset."""
        from .journal_station import CONTROL_DT_S

        view = self.journal_pre_dones(self.robot.data, self.scene.env_origins,
                                      float(self.physics_cfg.water_surface_z))
        state = self._journal_state(self.robot.data, self.scene.env_origins, view.water_level_local)
        steps = self.episode_length_buf.to(device=self.device, dtype=torch.long)
        time_s = steps.to(torch.float64) * CONTROL_DT_S
        com, bounds = state.com_xyz, self._jbounds
        out_of_bounds = ((com[:, 0] < bounds[:, 0]) | (com[:, 0] > bounds[:, 1])
                         | (com[:, 1] < bounds[:, 2]) | (com[:, 1] > bounds[:, 3]))
        collision = torch.as_tensor(view.d7_contact, dtype=torch.bool).to(self.device)
        if getattr(self, "_appearing_bridge", None) is not None:
            collision = collision | self.journal_appearing_contact().to(self.device)   # G20 swept latch
        if self.journal_contract_trace is not None:
            self.journal_contract_trace.append(("step", time_s.clone(), state, collision.clone(),
                                                out_of_bounds.clone()))
        codes = self._jtwin.step(time_s, state, collision, out_of_bounds)
        if self.journal_contract_trace is not None:
            self.journal_contract_trace.append(("codes", codes.clone()))
        if bool(((codes == 0) & (steps >= self._jdeadline_steps)).any()):
            raise RuntimeError("station-keeping episode ran past its layout deadline without a timeout code")
        # G11: the twin step already moved "any finished row" (codes != 0) to the host; the nonzero
        # transfer runs only when a row finished
        if self._jtwin.last_any_finished:
            done = (codes != 0).nonzero().flatten()
            if done.numel():
                self._journal_latch(done, codes)
        return self.journal_dones(codes)

    def _journal_latch(self, done, codes) -> None:
        twin = self._jtwin
        self.episode_success[done] = twin.c_success[done]
        self.time_to_success[done] = twin.c_tts[done].to(self.time_to_success.dtype)
        self.episode_path_length[done] = twin.c_full[done].to(self.episode_path_length.dtype)
        self.episode_max_hold_s[done] = twin.c_longest_hold[done].to(self.episode_max_hold_s.dtype)
        self.episode_termination_code[done] = codes[done]
        for i in done.tolist():
            result = twin.completed_result(i)
            self.journal_episode_results[i] = result
            if self.journal_contract_trace is not None:
                self.journal_contract_trace.append(("completed", i, dict(result)))
            rms = result["station_final_window_rms_error_m"]
            self.episode_station_error_final_60s_rms_m[i] = math.nan if rms is None else rms
        self.extras.setdefault("log", {})
        self.extras["log"]["Episode/success"] = twin.c_success[done].float().mean()

    def journal_episode_result(self, env_id) -> dict:
        """Evaluator fields of the env's last completed episode (EpisodeResult.evaluator_fields)."""
        result = self.journal_episode_results[int(env_id)]
        if result is None:
            raise RuntimeError(f"env {int(env_id)} has no completed journal episode")
        return dict(result)

    # -------------------------------------------------------------- reset --
    def _journal_reset_idx(self, env_ids: Sequence[int] | None):
        """Scenario counter, E5 layout, COM spawn + ROOT WRITE, twin, then journal_reset."""
        from . import journal_station as js

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        super()._reset_idx(ids)
        cpu = ids.detach().cpu()
        self._jscenario.reset_idx(cpu)
        episodes = self._jscenario.episode_indices()[cpu]
        seed, vessel = int(self.cfg.seed), self.cfg.vehicle
        starts, headings, stations, radii, deadlines, steps, lates, bounds, windows = ([] for _ in range(9))
        com_z_above_water = self._jcom_z - float(self.physics_cfg.water_surface_z)
        for i, episode in zip(cpu.tolist(), episodes.tolist()):
            layout = js.build_station_layout(self._jtier, vessel, seed, i, episode)
            station = layout.extras["station"]
            key = js.episode_key(seed, i, episode)
            # admitted / not_admitted (offered): the env layout is checked against the stored one;
            # beyond_offered_grid (D24 (a)): the pool has no record, the geometry check is skipped;
            # any other key raises the preflight ValueError (D24 (b)).
            if self._d7_pool.key_status(key) != js.BEYOND_OFFERED_GRID:
                self._d7_pool.bind_context(key, js.station_task_context(layout, vessel))
            self._jlayouts[i] = layout
            self._jspawn_hash[i] = js.spawn_pose_hash(layout, com_z_above_water)
            starts.append([float(v) for v in layout.start])
            headings.append(float(layout.start_heading_rad))
            stations.append([float(v) for v in station["center_xy"]])
            radii.append(float(station["radius_m"]))
            deadlines.append(float(layout.deadline_s))
            steps.append(int(layout.deadline_steps))
            lates.append([float(v) for v in layout.extras["late_interval_s"]])
            bounds.append([float(v) for v in layout.bounds_m])
            windows.append(float(station["final_window_s"]))
        f64 = dict(dtype=torch.float64, device=self.device)
        start = torch.tensor(starts, **f64)
        heading = torch.tensor(headings, **f64)
        station_xy = torch.tensor(stations, **f64)
        radius = torch.tensor(radii, **f64)
        deadline = torch.tensor(deadlines, **f64)
        late = torch.tensor(lates, **f64)
        window = torch.tensor(windows, **f64)
        self._jstation[ids] = station_xy
        self.hold_point[ids] = (self.scene.env_origins[ids, :2].to(torch.float64) + station_xy).to(
            self.hold_point.dtype)
        self._jradius[ids] = radius
        self._jbounds[ids] = torch.tensor(bounds, **f64)
        self._jdeadline_steps[ids] = torch.tensor(steps, dtype=torch.long, device=self.device)
        self._jsettle[ids] = late[:, 0]
        root_state = self.robot.data.default_root_state[ids].clone()
        origins = self.scene.env_origins[ids]
        root_state[:, 0] = origins[:, 0] + start[:, 0].to(root_state.dtype)
        root_state[:, 1] = origins[:, 1] + start[:, 1].to(root_state.dtype)
        root_state[:, 2] = origins[:, 2] + self._jcom_z
        root_state[:, 3:7] = js.quat_from_yaw(heading).to(root_state.dtype)
        root_state[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root_state, ids)
        hold_s = float(self._jparams["hold_s"])
        self._jtwin.reset(ids, start, station_xy, radius, deadline, hold_s=hold_s, late_interval_s=late,
                          top_dose_start_s=js.TOP_DOSE_START_S, final_window_s=window)
        if self.journal_contract_trace is not None:
            self.journal_contract_trace.append(("reset", ids.clone(), start.clone(), station_xy.clone(),
                                                radius.clone(), deadline.clone(), hold_s, late.clone(),
                                                window.clone()))
        self._d7_physical[ids] = False
        self._d7_visible[ids] = False
        self._d7_radii[ids] = 0.0
        if getattr(self, "_appearing_bridge", None) is not None:
            self._appearing_physical[ids] = False
            self._appearing_visible[ids] = False
        axis = torch.atan2(station_xy[:, 1] - start[:, 1], station_xy[:, 0] - start[:, 0])
        # Perception outage: the task's critical phase = the scored late-interval start (fixed clock).
        self.journal_reset(ids, axis.detach().cpu(), episodes.clone(), outage_start_s=late[:, 0].detach().cpu())
