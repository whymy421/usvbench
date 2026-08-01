# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Catamaran Patrol Task — Week 3 USVBench.

The agent controls a catamaran that must visit N waypoints arranged as a
circuit, repeating the loop until the episode times out.

Observations (OBS_DIM=12 default):
  [0-1]  unit vector to current waypoint in body frame (xy)
  [2]    distance to current waypoint (normalised)
  [3-4]  body-frame xy velocity (normalised)
  [5]    body-frame yaw rate (normalised)
  [6-7]  unit vector to NEXT waypoint in body frame (look-ahead for turns)
  [8]    position in the circuit = wp_idx / N_WAYPOINTS, bounded to [0, 1)
  [9]    distance to next waypoint (normalised)
  [10-11] padding zeros (or extended obs if OBS_DIM > 12)

Actions [N, 2]:
  [0]  forward thrust   ∈ [-1, 1]  → cfg.thrust_max_fwd / thrust_max_rev along body +X
  [1]  yaw torque       ∈ [-1, 1]  → ±cfg.yaw_torque_max N·m around body +Z
"""

from __future__ import annotations
import math, os
import torch
try:
    import wandb as _wandb
    _WANDB = True
except ImportError:
    _WANDB = False
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate_inverse
from .._shared.restoring import restoring_torque_body
from .catamaran_patrol_env_cfg import CatamaranPatrolEnvCfg

# ── tunable via environment variables ────────────────────────────────────────
PROGRESS_COEF = float(os.environ.get("PROGRESS_COEF", "3.0"))   # 3× stronger navigation signal
REACH_BONUS   = float(os.environ.get("REACH_BONUS",   "150.0")) # 3× bigger waypoint bonus
HEADING_W     = float(os.environ.get("HEADING_W",     "0.5"))   # max heading reward per step
SPEED_COUPLE  = int(  os.environ.get("SPEED_COUPLE",  "1"))
# Terminal surge speed that comes out of the thrust/drag balance in the cfg. If you
# change thrust_max_fwd or the surge damping, recompute this — the speed-coupled heading
# reward is normalised by it, so a stale value silently rescales the whole reward.
SPEED_REF     = float(os.environ.get("SPEED_REF",     "2.5"))   # m/s
OBS_DIM       = int(  os.environ.get("OBS_DIM",       "12"))
N_WAYPOINTS   = int(  os.environ.get("N_WAYPOINTS",   "4"))
WANDB_EVERY   = int(  os.environ.get("WANDB_EVERY",   "60"))    # log every N steps, not every step

FWD_X      = 1       # body +X = bow


class CatamaranPatrolEnv(DirectRLEnv):
    cfg: CatamaranPatrolEnvCfg

    def __init__(self, cfg: CatamaranPatrolEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N = self.num_envs
        D = self.device

        # Patrol state
        self.waypoint_pos  = torch.zeros(N, N_WAYPOINTS, 3, device=D)
        self.wp_idx        = torch.zeros(N, dtype=torch.long, device=D)
        self.wps_done      = torch.zeros(N, device=D)
        self.prev_dist     = torch.zeros(N, device=D)

        # For eval script compatibility (same attr as boat/ROV tasks)
        self.reached_count = 0
        self._last_reached_mask = torch.zeros(N, dtype=torch.bool, device=D)

        # Env origins — scene.env_origins is None without a terrain system,
        # so compute a grid manually from env_spacing.
        if self.scene.env_origins is not None:
            self._env_origins = self.scene.env_origins.clone()
        else:
            spacing = self.cfg.scene.env_spacing
            num_cols = max(1, int(N ** 0.5))
            origins = torch.zeros(N, 3, device=D)
            for i in range(N):
                origins[i, 0] = (i % num_cols) * spacing
                origins[i, 1] = (i // num_cols) * spacing
            self._env_origins = origins

        print(f"[CatamaranPatrolEnv] envs={N}  N_WAYPOINTS={N_WAYPOINTS}  OBS_DIM={OBS_DIM}")
        print(f"[CatamaranPatrolEnv] PROGRESS_COEF={PROGRESS_COEF}  REACH_BONUS={REACH_BONUS}  SPEED_COUPLE={SPEED_COUPLE}")

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        import isaaclab.sim as sim_utils

        self.robot = self.cfg.robot_cfg.class_type(self.cfg.robot_cfg)

        # Ground plane placed 50 m below water surface so it never collides
        # with the vessel (which floats at z ≈ -0.4 m due to buoyancy).
        # If placed at z=0 the hull clips in, friction pins the vessel, speed→0.
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg, translation=(0.0, 0.0, -50.0))

        # Register with scene so it gets cloned across envs
        self.scene.rigid_objects["robot"] = self.robot

        # Without a light the scene renders black, so --video and the GUI viewport
        # produce unusable frames. Same dome light as the boat reference task.
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Waypoint helpers ──────────────────────────────────────────────────────
    def _cur_wp(self) -> torch.Tensor:
        """[N, 3] current target waypoint positions."""
        idx = self.wp_idx.view(-1, 1, 1).expand(-1, 1, 3)
        return self.waypoint_pos.gather(1, idx).squeeze(1)

    def _nxt_wp(self) -> torch.Tensor:
        """[N, 3] next waypoint (circular look-ahead)."""
        nxt = (self.wp_idx + 1) % N_WAYPOINTS
        idx = nxt.view(-1, 1, 1).expand(-1, 1, 3)
        return self.waypoint_pos.gather(1, idx).squeeze(1)

    # ── Isaac Lab callbacks ───────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        """Called once per physics sub-step (decimation times per policy step).

        Frame discipline, matching boat_calm_nav: hull drag is anisotropic and therefore
        lives in the BODY frame, alongside the thrusters; buoyancy, heave damping and the
        attitude terms are natural in the WORLD frame. The two are kept in separate
        accumulators and the world-frame one is rotated into the body frame exactly once,
        at the point of application.

        set_external_force_and_torque() defaults to is_global=False, i.e. it treats
        whatever it is given as body-local and rotates it by the hull attitude. Handing it
        a world-frame vector applies that rotation a second time. The error is exactly zero
        at yaw = 0 — which is where every episode starts — and grows with heading, so it
        survives any quick test. scripts/check_wrench_frame.py is the check that catches it.
        """
        phys = self.cfg.underwater_physics_cfg
        cfg  = self.cfg
        quat = self.robot.data.root_quat_w
        pos  = self.robot.data.root_pos_w
        vel_w = self.robot.data.root_lin_vel_w
        ang_w = self.robot.data.root_ang_vel_w

        forces  = torch.zeros(self.num_envs, 3, device=self.device)
        torques = torch.zeros(self.num_envs, 3, device=self.device)
        force_w  = torch.zeros_like(forces)
        torque_w = torch.zeros_like(torques)

        # ── Thrusters (body frame), asymmetric forward/reverse ────────────────
        a0 = self.actions[:, 0]
        thrust = torch.where(a0 >= 0, a0 * cfg.thrust_max_fwd, a0 * cfg.thrust_max_rev)
        forces[:, 0]  = FWD_X * thrust
        torques[:, 2] = self.actions[:, 1] * cfg.yaw_torque_max

        # ── Buoyancy (world frame), proportional to submersion ────────────────
        z   = pos[:, 2]
        sub = torch.clamp(-z / max(phys.rov_height, 1e-3), 0.0, 1.0)
        force_w[:, 2] += sub * phys.water_density * phys.gravity * phys.rov_volume

        in_water = (z < phys.water_surface_z).float()

        # ── Hull drag (body frame), per-DOF linear + quadratic ────────────────
        vel_b = quat_rotate_inverse(quat, vel_w)
        forces[:, 0] += -(phys.surge_lin_damping
                          + phys.surge_quad_damping * vel_b[:, 0].abs()) * vel_b[:, 0] * in_water
        forces[:, 1] += -(phys.sway_lin_damping
                          + phys.sway_quad_damping * vel_b[:, 1].abs()) * vel_b[:, 1] * in_water

        # Out of the water the hull only sees air drag, on every axis.
        air = (1.0 - in_water).unsqueeze(-1) * phys.air_linear_damping
        forces -= air * vel_b

        # ── Heave damping (world frame) ───────────────────────────────────────
        force_w[:, 2] += -(in_water * phys.heave_damping) * vel_w[:, 2]

        # ── Yaw damping, linear + quadratic; roll/pitch rate damping ──────────
        wz = ang_w[:, 2]
        torque_w[:, 2] += -(phys.yaw_lin_damping
                            + phys.yaw_quad_damping * wz.abs()) * wz * in_water
        torque_w[:, :2] += -(in_water * phys.rollpitch_rate_damping).unsqueeze(-1) * ang_w[:, :2]
        torque_w -= ((1.0 - in_water) * phys.air_angular_damping).unsqueeze(-1) * ang_w

        # ── Hydrostatic restoring moment (body frame) ─────────────────────────
        # Shared implementation, tasks/_shared/restoring.py — the same one the newer
        # tasks use, rather than a private copy. It is yaw-invariant by construction and
        # takes separate roll and pitch stiffnesses, which this hull needs: measured off
        # the mesh it is an order of magnitude stiffer in pitch than in roll.
        #
        # An earlier private version extracted roll/pitch from the quaternion and applied
        # them about fixed WORLD axes. Those angles are body quantities; the two frames
        # coincide only at yaw = 0, so it was restoring near the spawn heading and drove
        # the hull over once it turned. check_catamaran_physics.py phase 3 guards this.
        torques += restoring_torque_body(
            quat, phys.restoring_stiffness_roll, phys.restoring_stiffness_pitch
        )

        # ── World frame → body frame, then apply once ─────────────────────────
        forces  += quat_rotate_inverse(quat, force_w)
        torques += quat_rotate_inverse(quat, torque_w)
        self.robot.set_external_force_and_torque(forces.unsqueeze(1), torques.unsqueeze(1))

    def _get_observations(self) -> dict:
        pos   = self.robot.data.root_pos_w
        rot   = self.robot.data.root_quat_w
        vel_w = self.robot.data.root_lin_vel_w
        ang_w = self.robot.data.root_ang_vel_w

        cur_wp  = self._cur_wp()
        nxt_wp  = self._nxt_wp()

        # World-frame offsets
        to_cur_w  = cur_wp  - pos
        to_nxt_w  = nxt_wp  - pos

        # Body-frame
        to_cur_b  = quat_rotate_inverse(rot, to_cur_w)
        to_nxt_b  = quat_rotate_inverse(rot, to_nxt_w)
        vel_b     = quat_rotate_inverse(rot, vel_w)
        ang_b     = quat_rotate_inverse(rot, ang_w)

        dist_cur  = torch.norm(to_cur_b[:, :2], dim=-1, keepdim=True)
        dist_nxt  = torch.norm(to_nxt_b[:, :2], dim=-1, keepdim=True)

        dir_cur   = to_cur_b[:, :2] / (dist_cur  + 1e-6)
        dir_nxt   = to_nxt_b[:, :2] / (dist_nxt  + 1e-6)

        # Normalisers track the terminal values from the cfg force balance (2.5 m/s,
        # 1.0 rad/s) with headroom. Leaving these at the old 8.0 / 3.0 would squash the
        # velocity channels into a fraction of their range.
        max_dist  = max(self.cfg.patrol_radius * 2.0, 1.0)
        max_speed = 4.0
        max_yaw   = 2.0

        # BUGFIX: the circuit repeats, so wps_done / N_WAYPOINTS grew past 1 and kept
        # growing for the whole episode — an unbounded, drifting observation. What the
        # policy actually needs is where it is *inside* the current lap, which is
        # wp_idx / N_WAYPOINTS and stays in [0, 1).
        lap_progress = self.wp_idx.float().unsqueeze(-1) / max(N_WAYPOINTS, 1)

        # Core 10-element obs
        obs_core = torch.cat([
            dir_cur,                                          # 2
            dist_cur / max_dist,                             # 1
            vel_b[:, :2] / max_speed,                       # 2
            ang_b[:, 2:3] / max_yaw,                        # 1
            dir_nxt,                                         # 2
            lap_progress,                                    # 1
            dist_nxt / max_dist,                             # 1
        ], dim=-1)  # shape [N, 10]

        # Pad or trim to OBS_DIM
        if OBS_DIM > 10:
            pad = torch.zeros(self.num_envs, OBS_DIM - 10, device=self.device)
            obs = torch.cat([obs_core, pad], dim=-1)
        else:
            obs = obs_core[:, :OBS_DIM]

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos   = self.robot.data.root_pos_w
        rot   = self.robot.data.root_quat_w
        vel_w = self.robot.data.root_lin_vel_w

        cur_wp = self._cur_wp()
        dist   = torch.norm(pos[:, :2] - cur_wp[:, :2], dim=-1)

        # ── Progress reward ───────────────────────────────────────────────
        progress = (self.prev_dist - dist) * PROGRESS_COEF
        self.prev_dist = dist.clone()

        # ── Waypoint reach bonus ──────────────────────────────────────────
        reached = dist < self.cfg.goal_radius
        bonus   = reached.float() * REACH_BONUS
        self._last_reached_mask = reached

        # Advance waypoint index for envs that reached
        if reached.any():
            env_ids = reached.nonzero(as_tuple=False).squeeze(-1)
            self.wp_idx[env_ids]    = (self.wp_idx[env_ids] + 1) % N_WAYPOINTS
            self.wps_done[env_ids] += 1.0
            self.reached_count     += int(reached.sum().item())
            # Reset distance estimate to new waypoint
            new_wp = self._cur_wp()
            new_dist = torch.norm(pos[:, :2] - new_wp[:, :2], dim=-1)
            self.prev_dist[env_ids] = new_dist[env_ids]

        # ── Heading alignment reward, coupled to forward speed ─────────────
        # cos(angle between body +X and vector to waypoint), clamped to [0, 1].
        to_wp_w  = cur_wp - pos
        to_wp_b  = quat_rotate_inverse(rot, to_wp_w)
        dist_xy  = torch.norm(to_wp_b[:, :2], dim=-1)
        heading_cos = to_wp_b[:, 0] / (dist_xy + 1e-6)     # [-1, 1]
        align       = torch.clamp(heading_cos, 0.0, 1.0)

        vel_b     = quat_rotate_inverse(rot, vel_w)
        fwd_speed = vel_b[:, 0] * FWD_X

        # BUGFIX: previously this term paid up to 0.5 every step for merely pointing at
        # the waypoint, and SPEED_COUPLE only *added* a separate speed bonus instead of
        # coupling the two. Over a 7200-step episode that is up to 3600 reward for
        # sitting still and aiming — far more than any waypoint bonus, so the optimal
        # policy was to stop outside the goal radius and stare at it. Multiplying by
        # normalised forward speed pays the alignment only while actually closing in,
        # which is the same recipe as the boat reference task (V23).
        speed_frac = torch.clamp(fwd_speed / SPEED_REF, 0.0, 1.0)
        if SPEED_COUPLE:
            heading_rew = align * speed_frac * HEADING_W
        else:
            heading_rew = align * HEADING_W

        return progress + bonus + heading_rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated  = self.episode_length_buf >= self.max_episode_length

        # ── Log metrics directly to wandb (same pattern as boat_calm_nav) ──────
        if _WANDB and _wandb.run is not None:
            # Nav metrics were logged on every one of the 60 control steps per second,
            # which dominated the step time and flooded the run. Once per second of
            # simulated time is enough to read the curves.
            if self.common_step_counter % max(WANDB_EVERY, 1) == 0:
                pos    = self.robot.data.root_pos_w
                vel_w  = self.robot.data.root_lin_vel_w
                rot    = self.robot.data.root_quat_w
                cur_wp = self._cur_wp()
                dist   = torch.norm(pos[:, :2] - cur_wp[:, :2], dim=-1)
                vel_b  = quat_rotate_inverse(rot, vel_w)
                fwd_speed = vel_b[:, 0] * FWD_X

                try:
                    _wandb.log({
                        "Nav/speed":              fwd_speed.mean().item(),
                        "Nav/distance_to_target": dist.mean().item(),
                        "Nav/waypoint_index":     self.wp_idx.float().mean().item(),
                    })
                except Exception:
                    pass

            # Episode-level metrics (when envs finish)
            done = terminated | truncated
            if done.any():
                try:
                    _wandb.log({
                        "Metrics/targets_per_episode": self.wps_done[done].float().mean().item(),
                    })
                except Exception:
                    pass

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        D = self.device
        cfg = self.cfg

        # Random heading
        yaw = torch.rand(n, device=D) * 2.0 * math.pi
        qw  = torch.cos(yaw * 0.5)
        qz  = torch.sin(yaw * 0.5)
        rot = torch.stack([qw, torch.zeros(n, device=D), torch.zeros(n, device=D), qz], dim=-1)

        # Random spawn near first waypoint (placed after waypoints are generated)
        spawn_r   = torch.rand(n, device=D) * (cfg.max_spawn_distance - cfg.min_spawn_distance) + cfg.min_spawn_distance
        spawn_ang = torch.rand(n, device=D) * 2.0 * math.pi
        origin    = self._env_origins[env_ids]
        pos       = origin.clone()
        pos[:, 0] += spawn_r * torch.cos(spawn_ang)
        pos[:, 1] += spawn_r * torch.sin(spawn_ang)
        pos[:, 2]  = 0.0     # water surface

        self.robot.write_root_pose_to_sim(torch.cat([pos, rot], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=D), env_ids=env_ids)

        # ── Generate patrol circuit ───────────────────────────────────────
        # N_WAYPOINTS equally spaced around a circle, rotated randomly per env
        base_angles   = torch.arange(N_WAYPOINTS, device=D).float() / N_WAYPOINTS * 2.0 * math.pi
        circuit_rot   = torch.rand(n, device=D) * 2.0 * math.pi                  # [n]
        angles        = base_angles.unsqueeze(0) + circuit_rot.unsqueeze(1)       # [n, WP]
        # Vary radius slightly so circuit isn't a perfect circle (+/-20% jitter)
        r             = cfg.patrol_radius * (0.8 + 0.4 * torch.rand(n, N_WAYPOINTS, device=D))
        wp_x          = r * torch.cos(angles) + origin[:, 0:1]
        wp_y          = r * torch.sin(angles) + origin[:, 1:2]
        wp_z          = torch.zeros(n, N_WAYPOINTS, device=D)
        self.waypoint_pos[env_ids] = torch.stack([wp_x, wp_y, wp_z], dim=-1)

        # Reset patrol counters
        self.wp_idx[env_ids]   = 0
        self.wps_done[env_ids] = 0.0

        # Initial distance to first waypoint
        wp0 = self.waypoint_pos[env_ids, 0, :]
        self.prev_dist[env_ids] = torch.norm(pos[:, :2] - wp0[:, :2], dim=-1)
