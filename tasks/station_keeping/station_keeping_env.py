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
    stamp_scenario,
    station_keeping_current,
    station_keeping_spawn,
)
from .._shared.vehicles import get_vehicle
from .current_ramp import ramped_current_vec
from .station_keeping_env_cfg import StationKeepingEnvCfg


class StationKeepingEnv(DirectRLEnv):
    """Calm-water station keeping with reward-independent success."""

    cfg: StationKeepingEnvCfg

    def __init__(self, cfg: StationKeepingEnvCfg, render_mode: str | None = None, **kwargs):
        self.vehicle_spec = get_vehicle(cfg.vehicle)
        self.physics_cfg = cfg.underwater_physics_cfg
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

            self._sea = SeaState(self.cfg.sea_state, self.num_envs, self.device)
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
            t = float(self.episode_length_buf[0]) * self.control_step_s
            wave_f, wave_t = self._sea.forces(
                self.robot.data.root_com_pos_w[:, :2],
                self.robot.data.root_com_vel_w[:, :2],
                t,
            )
            force_w += wave_f
            torque_w += wave_t

        forces[:, 0, :] += math_utils.quat_apply_inverse(quat, force_w)
        torques[:, 0, :] += math_utils.quat_apply_inverse(quat, torque_w)
        self.robot.set_external_force_and_torque(forces, torques)

    def _horizontal_distance(self) -> torch.Tensor:
        return torch.norm(self.hold_point - self.robot.data.root_pos_w[:, :2], dim=-1)

    def _get_observations(self) -> dict:
        forwards = math_utils.quat_apply(
            self.robot.data.root_link_quat_w,
            self.forward_vec,
        )

        rpos = self.hold_point - self.robot.data.root_pos_w[:, :2]
        if self._obs_degrader is not None and self._obs_degrader.wants("pos"):
            # OOD hook: corrupt the assembled hold-point vector (ideal
            # relative-position measurement, meters) before distance and
            # bearing features are derived from it.
            rpos = self._obs_degrader.apply("pos", rpos)
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

            if self._obs_degrader is not None and self._obs_degrader.wants(
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
                            self._obs_degrader.apply(
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
        terminated = torch.zeros_like(time_out)
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
            for completed in completed_ids.tolist():
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
        self._success[env_ids] = False
        self._episode_finished[env_ids] = False
        self._previous_xy[env_ids] = root_state[:, :2]
        if self._sea is not None:
            self._sea.resample(env_ids, scenario=self._scenario)
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
