# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Rescue Boat Task — USVBench P2.

Deadline-aware multi-casualty prioritisation:
  - N casualties spawned at episode start (random positions, 30-120m radius)
  - Each casualty has a countdown timer (30-70 s); expire → casualty "lost"
  - Agent must reach casualties (within rescue_radius) before timers expire
  - Core challenge: trade off nearest vs. most urgent

Observations (14-dim):
  [0-1]   unit vec to priority-0 casualty in body frame (xy)
  [2]     dist to priority-0 (normalised)
  [3]     urgency of priority-0  (1 - t_remaining / t_max)  ← deadline signal
  [4-5]   unit vec to priority-1 casualty in body frame (xy)
  [6]     dist to priority-1 (normalised)
  [7]     urgency of priority-1
  [8-9]   body-frame xy velocity (normalised)
  [10]    yaw rate (normalised)
  [11]    rescued_count / n_casualties
  [12]    alive_count / n_casualties
  [13]    episode time fraction remaining

Actions [N, 2]:
  [0]  forward thrust  ∈ [-1, 1] → ±MAX_THRUST N along body +X
  [1]  yaw torque      ∈ [-1, 1] → ±MAX_TORQUE N·m around body +Z

Primary metric: rescue_rate = rescued / n_casualties  (P2 bar ≥ 0.70)
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
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse
from .rescue_boat_env_cfg import RescueBoatEnvCfg

# ── Thruster constants (from open-loop physics check) ────────────────────────
MAX_THRUST = 500.0   # N — full forward thrust (body +X)
MAX_TORQUE = 800.0   # N·m — full yaw torque (body +Z)
FWD_X      = 1       # body +X is bow

# ── Priority weighting ────────────────────────────────────────────────────────
# Priority score: (urgency^URGENCY_POW) / (dist + DIST_EPS)
# Higher urgency + closer = higher priority.
# Adjust URGENCY_POW to control urgency vs. distance trade-off.
URGENCY_POW = float(os.environ.get("URGENCY_POW", "2.0"))

# ── Reward coefficients ───────────────────────────────────────────────────────
PROGRESS_COEF = float(os.environ.get("PROGRESS_COEF", "3.0"))   # v2b: raised 2→3
RESCUE_BONUS  = float(os.environ.get("RESCUE_BONUS",  "500.0"))  # v2b: raised 300→500
LOSE_PENALTY  = float(os.environ.get("LOSE_PENALTY",  "50.0"))
# Proximity reward disabled — caused reward hacking (agent hovering near casualties
# without rescuing). Progress reward provides sufficient dense gradient at 10-50m spawn.
PROX_COEF     = float(os.environ.get("PROX_COEF",    "0.0"))
PROX_RADIUS   = float(os.environ.get("PROX_RADIUS",  "25.0"))   # m (unused when PROX_COEF=0)

# ── Locomotion shaping (v4) ───────────────────────────────────────────────────
# The v3 reward was defined purely on distance to the current target, so nothing
# constrained HOW the vessel travelled. Measured behaviour: 19 revolutions per
# episode with the hull pointing 87 degrees away from its direction of travel —
# the policy span rapidly and drifted on accumulated momentum, which reaches the
# target but is not something a real vessel can do.
#
# This is the third specification failure of the same kind in the project. V2's
# proximity term rewarded being near a casualty, so the agent hovered. V3's
# progress term rewarded closing distance, so the agent spiralled. In both cases
# the reward named a correlate of the goal rather than the goal.
#
#   HEADING_COEF   rewards pointing at the target: cos(bearing error), so +1 when
#                  aimed at it, -1 when aimed away. Gives a reason to hold a
#                  heading that closing distance alone does not.
#   YAWRATE_COEF   penalises sustained rotation, quadratic so that ordinary
#                  course corrections are cheap and spinning is not.
#
# Both default to 0.0 so v3 results remain reproducible.
HEADING_COEF  = float(os.environ.get("HEADING_COEF",  "0.0"))
YAWRATE_COEF  = float(os.environ.get("YAWRATE_COEF",  "0.0"))

# ── Environmental disturbance (sea state) ─────────────────────────────────────
# All default to 0.0, so the environment is calm unless explicitly configured.
# Existing calm-water results are therefore unaffected by this addition.
#
# Used for zero-shot robustness evaluation: policies trained in calm water are
# evaluated under increasing disturbance to quantify the sim2real transfer gap.
#
#   CURRENT_SPEED   steady drift force, as an equivalent free-stream speed (m/s)
#   CURRENT_DIR     current heading in world frame (degrees, 0 = +X)
#   WAVE_AMP        peak oscillatory force amplitude (N)
#   WAVE_PERIOD     wave period (s)
#   WAVE_YAW_AMP    peak oscillatory yaw torque amplitude (N·m)
#   GUST_STD        std. dev. of per-step random force (N), models turbulence
#
# Suggested sea states (roughly Douglas scale 0/2/4/5):
#   calm      CURRENT_SPEED=0.0  WAVE_AMP=0     WAVE_YAW_AMP=0    GUST_STD=0
#   slight    CURRENT_SPEED=0.5  WAVE_AMP=150   WAVE_YAW_AMP=200  GUST_STD=25
#   moderate  CURRENT_SPEED=1.0  WAVE_AMP=300   WAVE_YAW_AMP=400  GUST_STD=50
#   rough     CURRENT_SPEED=1.5  WAVE_AMP=500   WAVE_YAW_AMP=650  GUST_STD=90
# ── Hull stability ────────────────────────────────────────────────────────────
# A surface vessel needs a righting moment. Buoyancy here is applied at the centre
# of mass, so nothing opposes roll: any heel tilts body +Z away from vertical, at
# which point the body-frame yaw torque acquires a horizontal component and rolls
# the hull further. That feedback loop capsized the boat within ~10 s of any turn,
# after which it sank to the ground plane and the episode was unrecoverable.
#
# Two corrections, both enabled by default:
#   HORIZONTAL_ACTUATION  apply yaw torque about WORLD +Z and keep thrust in the
#                         horizontal plane, so steering cannot induce roll
#   RIGHTING_K / _C       spring-damper restoring roll and pitch toward level,
#                         standing in for the metacentric stability of a real hull
# Render a water surface and casualty markers. Off by default (headless training
# gains nothing from it); set VISUALISE=1 when recording video.
VISUALISE = os.environ.get("VISUALISE", "0") == "1"

# Default OFF from 21 Aug: full 6-DOF actuation, as Yutong prefers for benchmark
# consistency. It was only ever a workaround for the capsize, and with the
# restoring torque now yaw-invariant it should not be needed. Kept as a switch so
# the two can be compared directly.
HORIZONTAL_ACTUATION = os.environ.get("HORIZONTAL_ACTUATION", "0") == "1"

# set_external_force_and_torque() expects the wrench in the body's LOCAL frame.
# The wrench here is assembled in world coordinates, so it must be rotated back
# before it is applied. Set WRENCH_WORLD=1 to restore the previous (incorrect)
# behaviour of passing world vectors straight through, for comparison.
WRENCH_WORLD = os.environ.get("WRENCH_WORLD", "0") == "1"

# Gains deliberately modest. With HORIZONTAL_ACTUATION on, the yaw command can no
# longer induce roll, so this only has to correct residual heel rather than fight a
# runaway — and a stiff spring-damper against an unknown rotational inertia can
# overshoot within a single 1/120 s step and diverge. RIGHTING_MAX bounds the
# torque so a bad gain degrades into sluggish righting rather than an explosion.
RIGHTING_K   = float(os.environ.get("RIGHTING_K",   "1500.0"))  # N·m per rad of heel
RIGHTING_C   = float(os.environ.get("RIGHTING_C",   "400.0"))   # N·m per rad/s
RIGHTING_MAX = float(os.environ.get("RIGHTING_MAX", "2500.0"))  # N·m hard cap

CURRENT_SPEED = float(os.environ.get("CURRENT_SPEED", "0.0"))
CURRENT_DIR   = float(os.environ.get("CURRENT_DIR",   "0.0"))
WAVE_AMP      = float(os.environ.get("WAVE_AMP",      "0.0"))
WAVE_PERIOD   = float(os.environ.get("WAVE_PERIOD",   "7.0"))
WAVE_YAW_AMP  = float(os.environ.get("WAVE_YAW_AMP",  "0.0"))
GUST_STD      = float(os.environ.get("GUST_STD",      "0.0"))

DISTURBED = (CURRENT_SPEED != 0.0 or WAVE_AMP != 0.0
             or WAVE_YAW_AMP != 0.0 or GUST_STD != 0.0)


class RescueBoatEnv(DirectRLEnv):
    cfg: RescueBoatEnvCfg

    def __init__(self, cfg: RescueBoatEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N = self.num_envs
        D = self.device
        C = cfg.n_casualties

        # Casualty state — [N, C]
        self.casualty_pos   = torch.zeros(N, C, 3, device=D)   # world positions
        self.casualty_timer = torch.zeros(N, C, device=D)       # time remaining (s)
        self.casualty_init_timer = torch.zeros(N, C, device=D)  # initial timer (for urgency calc)
        self.casualty_alive = torch.zeros(N, C, dtype=torch.bool, device=D)

        # Per-episode stats (per-env tensors)
        self.rescued_count  = torch.zeros(N, device=D)
        self.lost_count     = torch.zeros(N, device=D)

        # Global cumulative counter — train_with_eval.py reads this as reached_count
        # to compute targets_per_episode during the eval sweep
        self.reached_count  = 0

        # Water force accumulators
        self._water_F = torch.zeros(N, 3, device=D)
        self._water_T = torch.zeros(N, 3, device=D)

        # ── Disturbance state ────────────────────────────────────────────────
        # Per-env wave phase offset so environments are not synchronised; a
        # shared phase would let the policy exploit a single global oscillation.
        self._wave_phase = torch.rand(N, device=D) * (2.0 * math.pi)
        self._disturb_t  = torch.zeros(N, device=D)   # elapsed time per env (s)
        _cdir = math.radians(CURRENT_DIR)
        self._current_dir_xy = torch.tensor([math.cos(_cdir), math.sin(_cdir)], device=D)

        # Prev dist to priority casualty (for progress reward)
        self.prev_dist_priority = torch.full((N,), float("inf"), device=D)

        # Env origins
        if self.scene.env_origins is not None:
            self._env_origins = self.scene.env_origins.clone()
        else:
            spacing  = self.cfg.scene.env_spacing
            num_cols = max(1, int(N ** 0.5))
            origins  = torch.zeros(N, 3, device=D)
            for i in range(N):
                origins[i, 0] = (i % num_cols) * spacing
                origins[i, 1] = (i // num_cols) * spacing
            self._env_origins = origins

        print(f"[RescueBoatEnv] envs={N}  n_casualties={C}  episode={cfg.episode_length_s}s")
        print(f"[RescueBoatEnv] rescue_radius={cfg.rescue_radius}m  "
              f"timer=[{cfg.casualty_timer_min},{cfg.casualty_timer_max}]s")

        # ── Task validity: no casualty may start inside the capture radius ────
        # If spawn_r_min <= rescue_radius, some fraction of casualties are within
        # capture distance at t=0 and are rescued with the vessel stationary. The
        # task is then partly solved before the episode begins, and rescue_rate
        # measures the spawn distribution rather than the policy. This was how a
        # reported 1.0000 (768/768) arose: 40% of casualties spawned inside a 20 m
        # capture radius drawn from a 10-35 m spawn annulus.
        if cfg.casualty_spawn_r_min <= cfg.rescue_radius:
            _frac = (min(cfg.rescue_radius, cfg.casualty_spawn_r_max)
                     - cfg.casualty_spawn_r_min) / max(
                         cfg.casualty_spawn_r_max - cfg.casualty_spawn_r_min, 1e-9)
            print("[RescueBoatEnv] " + "!" * 60)
            print(f"[RescueBoatEnv] INVALID TASK: spawn_r_min "
                  f"({cfg.casualty_spawn_r_min}m) <= rescue_radius "
                  f"({cfg.rescue_radius}m)")
            print(f"[RescueBoatEnv] {_frac * 100:.0f}% of casualties start already "
                  f"within capture distance and are rescued at t=0.")
            print("[RescueBoatEnv] rescue_rate from this configuration is not a "
                  "measure of policy quality.")
            print("[RescueBoatEnv] " + "!" * 60)
        print(f"[RescueBoatEnv] URGENCY_POW={URGENCY_POW}  "
              f"RESCUE_BONUS={RESCUE_BONUS}  LOSE_PENALTY={LOSE_PENALTY}  "
              f"PROX_COEF={PROX_COEF}  PROX_RADIUS={PROX_RADIUS}")
        if DISTURBED:
            print(f"[RescueBoatEnv] SEA STATE: current={CURRENT_SPEED}m/s @{CURRENT_DIR}deg  "
                  f"wave_amp={WAVE_AMP}N period={WAVE_PERIOD}s  "
                  f"wave_yaw={WAVE_YAW_AMP}Nm  gust_std={GUST_STD}N")
        else:
            print("[RescueBoatEnv] SEA STATE: calm (no disturbance)")
        # Provenance: a checkpoint trained under one actuation model and scored
        # under another is a transfer result, not a re-score. Printing both here
        # means every log records which physics produced its numbers.
        print(f"[RescueBoatEnv] ACTUATION: "
              f"{'horizontal (yaw about world Z, thrust planar)' if HORIZONTAL_ACTUATION else 'full 6-DOF (body-frame wrench)'}")
        print(f"[RescueBoatEnv] RIGHTING: yaw-invariant K*(hull_up x world_up)  "
              f"K={RIGHTING_K} C={RIGHTING_C} max={RIGHTING_MAX}")
        print(f"[RescueBoatEnv] WRENCH: "
              f"{'world frame (UNCORRECTED - double rotation)' if WRENCH_WORLD else 'body frame (corrected)'}")
        print(f"[RescueBoatEnv] TASK: rescue_radius={cfg.rescue_radius}m  "
              f"spawn=[{cfg.casualty_spawn_r_min},{cfg.casualty_spawn_r_max}]m")

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        import isaaclab.sim as sim_utils

        self.robot = self.cfg.robot_cfg.class_type(self.cfg.robot_cfg)

        # Ground plane 50m below to avoid friction with hull
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg, translation=(0.0, 0.0, -50.0))

        # ── Visualisation ────────────────────────────────────────────────────
        # Off by default so headless training is unaffected; enable with
        # VISUALISE=1 when recording video for the poster or oral exam.
        # Without this the renders show only a wireframe ground plane 50 m below
        # and a distant speck, because the water is a force model with no surface
        # and the casualties have no geometry.
        if VISUALISE:
            # Lighting. Isaac Lab adds no default light in headless rendering, so
            # without this the whole scene renders near-black and neither the water
            # sheet nor the casualty markers are visible — which is why the earlier
            # videos showed only a dark silhouette.
            sky = sim_utils.DomeLightCfg(
                intensity=380.0, color=(0.62, 0.74, 0.90), texture_file=None
            )
            sky.func("/World/SkyLight", sky)

            sun = sim_utils.DistantLightCfg(
                intensity=2600.0, color=(1.0, 0.97, 0.90), angle=1.2
            )
            sun.func("/World/SunLight", sun,
                     orientation=(0.88, 0.33, 0.30, 0.10))   # low sun, long shadows

            # Semi-transparent water surface at z = 0 for visual reference only:
            # a visual prim carries no collider, so the physics is untouched.
            water = sim_utils.CuboidCfg(
                size=(1000.0, 1000.0, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.04, 0.20, 0.38), roughness=0.22, metallic=0.05
                ),
            )
            water.func("/World/WaterSurface", water, translation=(0.0, 0.0, 0.0))

            # Casualty markers, recoloured each step by urgency in _update_markers.
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._casualty_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Casualties",
                    markers={
                        "alive": sim_utils.SphereCfg(
                            radius=7.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.10, 0.85, 0.25), emissive_color=(0.03, 0.30, 0.08)
                            ),
                        ),
                        "urgent": sim_utils.SphereCfg(
                            radius=7.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.98, 0.65, 0.05), emissive_color=(0.40, 0.22, 0.0)
                            ),
                        ),
                        "critical": sim_utils.SphereCfg(
                            radius=7.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.95, 0.10, 0.10), emissive_color=(0.45, 0.02, 0.02)
                            ),
                        ),
                        "lost": sim_utils.SphereCfg(
                            radius=3.5,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.25, 0.25, 0.28), opacity=0.4
                            ),
                        ),
                    },
                )
            )
        else:
            self._casualty_markers = None

        self.scene.rigid_objects["robot"] = self.robot

    def _update_markers(self):
        """
        Colour casualty markers by urgency so the deadline pressure is legible on
        screen: green -> amber past 50% of the timer -> red past 80% -> grey once
        lost. This is what makes the prioritisation behaviour visible in a video;
        without it the task looks like undifferentiated waypoint chasing.
        """
        if self._casualty_markers is None:
            return

        N, C = self.num_envs, self.cfg.n_casualties
        pos = self.casualty_pos.reshape(-1, 3).clone()
        pos[:, 2] = 5.0   # float clear of the surface for an elevated camera

        urgency = 1.0 - (self.casualty_timer / (self.casualty_init_timer + 1e-6))
        urgency = urgency.clamp(0.0, 1.0).reshape(-1)
        alive = self.casualty_alive.reshape(-1)

        # Marker index maps to the prototype order declared in _setup_scene.
        idx = torch.zeros_like(urgency, dtype=torch.long)
        idx = torch.where(urgency > 0.5, torch.ones_like(idx), idx)
        idx = torch.where(urgency > 0.8, torch.full_like(idx, 2), idx)
        idx = torch.where(~alive, torch.full_like(idx, 3), idx)

        self._casualty_markers.visualize(translations=pos, marker_indices=idx)

    # ── Casualty helpers ──────────────────────────────────────────────────────
    def _priority_order(self) -> torch.Tensor:
        """[N, n_casualties] indices sorted by descending priority score.

        Priority = urgency^URGENCY_POW / (dist_xy + 1e-3)
        Dead casualties get priority = -inf so they sink to bottom.
        """
        pos  = self.robot.data.root_pos_w           # [N, 3]
        dp   = self.casualty_pos - pos.unsqueeze(1)  # [N, C, 3]
        dist = torch.norm(dp[:, :, :2], dim=-1)     # [N, C]

        urgency = 1.0 - (self.casualty_timer / (self.casualty_init_timer + 1e-6))
        urgency = urgency.clamp(0.0, 1.0)

        score = urgency.pow(URGENCY_POW) / (dist + 1e-3)   # [N, C]
        # Dead casualties: force to -inf
        score = score.masked_fill(~self.casualty_alive, float("-inf"))

        # Sort descending: highest score = highest priority
        return score.argsort(dim=-1, descending=True)   # [N, C]

    def _get_casualty_obs(self, priority_idx: torch.Tensor, rank: int):
        """Obs slice for the rank-th priority casualty: [dir_x, dir_y, dist_norm, urgency]."""
        N = self.num_envs
        D = self.device
        pos = self.robot.data.root_pos_w
        rot = self.robot.data.root_quat_w
        C   = self.cfg.n_casualties

        c_idx = priority_idx[:, rank]                        # [N]
        alive = self.casualty_alive.gather(1, c_idx.unsqueeze(1)).squeeze(1)  # [N]

        # World offset to casualty
        c_pos = self.casualty_pos[torch.arange(N, device=D), c_idx]  # [N, 3]
        to_c_w = c_pos - pos                                           # [N, 3]
        to_c_b = quat_rotate_inverse(rot, to_c_w)                      # [N, 3] body frame

        dist_xy = torch.norm(to_c_b[:, :2], dim=-1, keepdim=True)     # [N, 1]
        dir_xy  = to_c_b[:, :2] / (dist_xy + 1e-6)                    # [N, 2]

        max_dist = max(self.cfg.casualty_spawn_r_max * 1.5, 1.0)
        dist_norm = dist_xy / max_dist                                  # [N, 1]

        t_left = self.casualty_timer[torch.arange(N, device=D), c_idx]         # [N]
        t_init = self.casualty_init_timer[torch.arange(N, device=D), c_idx]    # [N]
        urgency = (1.0 - t_left / (t_init + 1e-6)).clamp(0.0, 1.0).unsqueeze(-1)  # [N,1]

        # Mask dead casualties with zeros
        mask = alive.float().unsqueeze(-1)
        dir_xy    = dir_xy    * mask
        dist_norm = dist_norm * mask
        urgency   = urgency   * mask

        return dir_xy, dist_norm, urgency

    # ── Water physics ─────────────────────────────────────────────────────────
    def _compute_water_forces(self):
        phys  = self.cfg.underwater_physics_cfg
        pos   = self.robot.data.root_pos_w
        vel_w = self.robot.data.root_lin_vel_w
        ang_w = self.robot.data.root_ang_vel_w

        z   = pos[:, 2]
        sub = torch.clamp(-z / max(phys.rov_height, 1e-3), 0.0, 1.0)
        F_buo_z = sub * phys.water_density * phys.gravity * phys.rov_volume

        in_water = (z < phys.water_surface_z).float()
        d_lin    = in_water * phys.max_linear_damping  + (1 - in_water) * phys.air_linear_damping
        d_ang    = in_water * phys.max_angular_damping + (1 - in_water) * phys.air_angular_damping

        F = -d_lin.unsqueeze(-1) * vel_w
        F[:, 2] += F_buo_z
        T = -d_ang.unsqueeze(-1) * ang_w

        # ── Environmental disturbance ────────────────────────────────────────
        # Skipped entirely when all coefficients are zero, so calm-water runs
        # are bit-identical to the pre-disturbance implementation.
        if DISTURBED:
            in_w = in_water.unsqueeze(-1)

            # Steady current: drag acts on velocity *relative to the moving water*,
            # so a current appears as a force proportional to the relative velocity
            # rather than as a constant push. This is what makes a current harder to
            # counter when heading into it than across it.
            if CURRENT_SPEED != 0.0:
                v_water = torch.zeros_like(vel_w)
                v_water[:, 0] = CURRENT_SPEED * self._current_dir_xy[0]
                v_water[:, 1] = CURRENT_SPEED * self._current_dir_xy[1]
                F = F + in_w * d_lin.unsqueeze(-1) * v_water

            # Wave forcing: sinusoidal surge/sway plus a yaw moment, each env
            # carrying its own phase offset.
            if WAVE_AMP != 0.0 or WAVE_YAW_AMP != 0.0:
                omega = 2.0 * math.pi / max(WAVE_PERIOD, 1e-3)
                ph    = omega * self._disturb_t + self._wave_phase
                s, c  = torch.sin(ph), torch.cos(ph)
                if WAVE_AMP != 0.0:
                    F[:, 0] = F[:, 0] + in_water * WAVE_AMP * s
                    F[:, 1] = F[:, 1] + in_water * WAVE_AMP * c * 0.6
                if WAVE_YAW_AMP != 0.0:
                    # Quarter-period lag: the yaw moment peaks as the boat rolls
                    # through the wave face, not at the force peak.
                    T[:, 2] = T[:, 2] + in_water * WAVE_YAW_AMP * torch.sin(ph - 0.5 * math.pi)

            # Gusts / turbulence: zero-mean random forcing.
            if GUST_STD != 0.0:
                F[:, :2] = F[:, :2] + in_w * torch.randn_like(F[:, :2]) * GUST_STD

        # ── Righting moment (metacentric stability) ──────────────────────────
        # Buoyancy above is a point force at the centre of mass and therefore
        # generates no restoring torque. Without this block the hull has zero
        # roll stability and capsizes under any sustained yaw command.
        if RIGHTING_K > 0.0:
            q = self.robot.data.root_quat_w

            # YAW-INVARIANT restoring torque:  K * (hull_up x world_up)
            #
            # An earlier version took roll and pitch as Euler angles from the
            # quaternion and applied the torque about world X and Y. Those angles
            # are body quantities and those axes are world axes, so they agree
            # only at yaw = 0. Once the hull turns, the roll correction acts partly
            # about the pitch axis and the two pump each other until it tumbles.
            # The error is in direction, not magnitude, which is why it is silent:
            # at 90 deg of heading the correction sat on an axis perpendicular to
            # the tilt it was supposed to remove.
            #
            # The cross product form has no such dependence. Its magnitude is
            # K*sin(tilt) about the axis that carries hull_up back to vertical,
            # whatever the heading.
            #
            # (Yutong: to be replaced by tasks/_shared/restoring.py once I can
            #  branch off yutong/catamaran-registry, so all vessels share one
            #  implementation.)
            up_w = torch.zeros_like(ang_w)
            up_w[:, 2] = 1.0
            hull_up = quat_rotate(q, up_w)                 # body +Z expressed in world
            tilt_axis = torch.cross(hull_up, up_w, dim=-1)  # |.| = sin(tilt)

            # Damp only the tilting motion. Removing the component along world up
            # leaves yaw untouched, so steering authority is unaffected.
            w_along_up = (ang_w * up_w).sum(dim=-1, keepdim=True) * up_w
            w_tilt = (ang_w - w_along_up).clamp(-5.0, 5.0)

            t_restore = RIGHTING_K * tilt_axis - RIGHTING_C * w_tilt

            # Bound the total torque so a poor gain degrades into sluggish
            # righting rather than a divergence inside one 1/120 s step.
            t_norm = t_restore.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = (RIGHTING_MAX / t_norm).clamp(max=1.0)
            t_restore = t_restore * scale

            T += in_water.unsqueeze(-1) * t_restore

        self._water_F = F
        self._water_T = T

    # ── Isaac Lab callbacks ───────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        self._compute_water_forces()
        rot = self.robot.data.root_quat_w

        F_body = torch.zeros_like(self._water_F)
        T_body = torch.zeros_like(self._water_T)
        F_body[:, 0] = FWD_X * self.actions[:, 0] * MAX_THRUST
        T_body[:, 2] =         self.actions[:, 1] * MAX_TORQUE

        F_world = quat_rotate(rot, F_body)

        if HORIZONTAL_ACTUATION:
            # Steering acts about world +Z, so a heeled hull cannot convert a yaw
            # command into a roll torque. Thrust is likewise held in the horizontal
            # plane rather than being allowed to drive the hull under or airborne.
            # Both are standard simplifications for a surface vessel and remove the
            # capsize feedback loop at source.
            F_world[:, 2] = 0.0
            T_world = torch.zeros_like(self._water_T)
            T_world[:, 2] = self.actions[:, 1] * MAX_TORQUE
            T_world = T_world + self._water_T
        else:
            T_world = quat_rotate(rot, T_body) + self._water_T

        F_world = F_world + self._water_F

        # ── Frame of the applied wrench ──────────────────────────────────────
        # set_external_force_and_torque() takes the wrench in the body's LOCAL
        # frame. Everything above is assembled in world coordinates (drag from
        # world velocity, buoyancy along world +Z, the restoring torque from
        # hull_up x world_up), so it has to be rotated back before being applied.
        #
        # The previous code passed world vectors straight in, so PhysX rotated
        # them a second time. Like the restoring-torque bug, this is exactly
        # correct at yaw = 0 -- the rotation is identity -- which is why it
        # survived every single-axis open-loop test and only appears once the
        # hull turns. Worse, the double-rotated drag stops opposing motion and
        # starts injecting energy, which is what drove the capsize.
        #
        # WRENCH_WORLD=1 restores the old behaviour for A/B comparison.
        if not WRENCH_WORLD:
            F_apply = quat_rotate_inverse(rot, F_world)
            T_apply = quat_rotate_inverse(rot, T_world)
        else:
            F_apply, T_apply = F_world, T_world

        self.robot.set_external_force_and_torque(
            F_apply.unsqueeze(1), T_apply.unsqueeze(1)
        )

        # ── Tick casualty timers ────────────────────────────────────────────
        # dt per apply_action call = sim.dt (not decimated)
        dt = self.cfg.sim.dt
        self.casualty_timer -= dt
        self.casualty_timer.clamp_(min=0.0)

        # Advance the disturbance clock so wave phase evolves with sim time
        if DISTURBED:
            self._disturb_t += dt

        if VISUALISE:
            self._update_markers()

        # Check newly expired casualties (alive + timer reached 0)
        just_expired = self.casualty_alive & (self.casualty_timer <= 0.0)
        if just_expired.any():
            self.casualty_alive[just_expired] = False
            self.lost_count += just_expired.float().sum(dim=-1)

    def _get_observations(self) -> dict:
        N    = self.num_envs
        D    = self.device
        vel_w = self.robot.data.root_lin_vel_w
        ang_w = self.robot.data.root_ang_vel_w
        rot   = self.robot.data.root_quat_w

        priority_idx = self._priority_order()  # [N, C]

        # Top-2 priority casualties
        dir0, dist0, urg0 = self._get_casualty_obs(priority_idx, 0)
        dir1, dist1, urg1 = self._get_casualty_obs(priority_idx, 1)

        # Body-frame velocity
        vel_b    = quat_rotate_inverse(rot, vel_w)
        max_spd  = 12.0
        max_yaw  = 3.0

        alive_count   = self.casualty_alive.float().sum(dim=-1, keepdim=True)
        rescued_frac  = (self.rescued_count / max(self.cfg.n_casualties, 1)).unsqueeze(-1)
        alive_frac    = (alive_count / max(self.cfg.n_casualties, 1))
        ep_remaining  = (1.0 - self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1)

        obs = torch.cat([
            dir0,                         # 2
            dist0,                        # 1
            urg0,                         # 1
            dir1,                         # 2
            dist1,                        # 1
            urg1,                         # 1
            vel_b[:, :2] / max_spd,       # 2
            ang_w[:, 2:3] / max_yaw,      # 1
            rescued_frac,                 # 1
            alive_frac,                   # 1
            ep_remaining,                 # 1
        ], dim=-1)   # [N, 14]

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        N   = self.num_envs
        D   = self.device
        pos = self.robot.data.root_pos_w

        priority_idx = self._priority_order()

        # Distance to priority-0 casualty
        c0_idx = priority_idx[:, 0]
        c0_pos = self.casualty_pos[torch.arange(N, device=D), c0_idx]
        dist0  = torch.norm(pos[:, :2] - c0_pos[:, :2], dim=-1)  # [N]

        # Only compute progress if casualty-0 is alive
        c0_alive = self.casualty_alive[torch.arange(N, device=D), c0_idx]
        dist0_masked = torch.where(c0_alive, dist0, self.prev_dist_priority)

        # Progress reward
        progress = (self.prev_dist_priority - dist0_masked) * PROGRESS_COEF
        progress = torch.where(c0_alive, progress, torch.zeros_like(progress))
        self.prev_dist_priority = dist0_masked.clone()

        # ── Locomotion shaping ────────────────────────────────────────────
        # Without these the reward says only "reduce distance", which a hull can
        # satisfy by spinning and drifting. Measured on v3: 19 revolutions per
        # episode, hull 87 degrees off its direction of travel.
        heading_rew = torch.zeros(N, device=D)
        yaw_pen = torch.zeros(N, device=D)

        if HEADING_COEF != 0.0 or YAWRATE_COEF != 0.0:
            rot = self.robot.data.root_quat_w
            w_, x_, y_, z_ = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
            yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))

            if HEADING_COEF != 0.0:
                # cos of the bearing error to the current target: +1 aimed at it,
                # -1 aimed away. Rewards holding a heading, which closing distance
                # alone does not.
                to_c = c0_pos[:, :2] - pos[:, :2]
                bearing = torch.atan2(to_c[:, 1], to_c[:, 0]) - yaw
                heading_rew = torch.cos(bearing) * HEADING_COEF
                heading_rew = torch.where(c0_alive, heading_rew,
                                          torch.zeros_like(heading_rew))

            if YAWRATE_COEF != 0.0:
                # Quadratic, so ordinary course corrections stay cheap while
                # sustained rotation becomes expensive.
                wz = self.robot.data.root_ang_vel_w[:, 2]
                yaw_pen = -(wz ** 2) * YAWRATE_COEF

        # ── Rescue detection ──────────────────────────────────────────────
        pos_exp = pos[:, :2].unsqueeze(1).expand(-1, self.cfg.n_casualties, -1)
        cas_exp = self.casualty_pos[:, :, :2]
        dists   = torch.norm(pos_exp - cas_exp, dim=-1)   # [N, C]

        just_rescued = self.casualty_alive & (dists < self.cfg.rescue_radius)
        rescue_bonus = torch.zeros(N, device=D)

        if just_rescued.any():
            self.casualty_alive[just_rescued] = False
            bonus_per_env = just_rescued.float().sum(dim=-1) * RESCUE_BONUS
            self.rescued_count += just_rescued.float().sum(dim=-1)
            self.reached_count += int(just_rescued.sum().item())  # global counter for train_with_eval.py
            rescue_bonus = bonus_per_env
            # Reset prev_dist to NEW top-priority casualty distance — NOT inf.
            # Setting to inf causes a massive reward spike on the next step → NaN gradients.
            envs_with_rescue = just_rescued.any(dim=-1)
            new_prio = self._priority_order()
            new_c0_idx = new_prio[:, 0]
            new_c0_alive = self.casualty_alive[torch.arange(N, device=D), new_c0_idx]
            new_c0_pos = self.casualty_pos[torch.arange(N, device=D), new_c0_idx]
            new_dist = torch.norm(pos[:, :2] - new_c0_pos[:, :2], dim=-1)
            self.prev_dist_priority[envs_with_rescue & new_c0_alive] = \
                new_dist[envs_with_rescue & new_c0_alive]
            self.prev_dist_priority[envs_with_rescue & ~new_c0_alive] = 0.0

        # ── Proximity shaping ─────────────────────────────────────────────
        # Dense gradient: reward proportional to how close we are to any alive
        # casualty within PROX_RADIUS. Sum over all alive casualties.
        if PROX_COEF > 0.0:
            pos_exp2  = pos[:, :2].unsqueeze(1).expand(-1, self.cfg.n_casualties, -1)
            cas_exp2  = self.casualty_pos[:, :, :2]
            dists_all = torch.norm(pos_exp2 - cas_exp2, dim=-1)   # [N, C]
            prox      = torch.clamp(1.0 - dists_all / max(PROX_RADIUS, 1e-3), min=0.0)
            prox      = (prox * self.casualty_alive.float()).sum(dim=-1) * PROX_COEF
        else:
            prox = torch.zeros(N, device=D)

        return progress + rescue_bonus + prox + heading_rew + yaw_pen

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated  = self.episode_length_buf >= self.max_episode_length

        # ── Wandb metrics ─────────────────────────────────────────────────
        if _WANDB and _wandb.run is not None:
            pos   = self.robot.data.root_pos_w
            vel_w = self.robot.data.root_lin_vel_w
            rot   = self.robot.data.root_quat_w
            vel_b = quat_rotate_inverse(rot, vel_w)
            fwd_speed = vel_b[:, 0] * FWD_X

            try:
                _wandb.log({
                    "Nav/speed":          fwd_speed.mean().item(),
                    "Nav/rescued_so_far": self.rescued_count.mean().item(),
                    "Nav/alive_count":    self.casualty_alive.float().sum(dim=-1).mean().item(),
                })
            except Exception:
                pass

            done = terminated | truncated
            if done.any():
                try:
                    rr = (self.rescued_count[done] / self.cfg.n_casualties).mean().item()
                    _wandb.log({
                        "Metrics/rescue_rate":             rr,
                        "Metrics/casualties_rescued":      self.rescued_count[done].mean().item(),
                        "Metrics/casualties_lost":         self.lost_count[done].mean().item(),
                    })
                except Exception:
                    pass

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        n = len(env_ids)
        D = self.device
        C = self.cfg.n_casualties
        cfg = self.cfg

        # ── Spawn boat ─────────────────────────────────────────────────────
        yaw = torch.rand(n, device=D) * 2.0 * math.pi
        qw  = torch.cos(yaw * 0.5)
        qz  = torch.sin(yaw * 0.5)
        rot = torch.stack([qw, torch.zeros(n, device=D), torch.zeros(n, device=D), qz], dim=-1)

        origin = self._env_origins[env_ids]
        pos    = origin.clone()
        pos[:, 2] = 0.0

        self.robot.write_root_pose_to_sim(torch.cat([pos, rot], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=D), env_ids=env_ids)

        # ── Reset disturbance state ────────────────────────────────────────
        # Fresh wave phase per episode so the policy cannot memorise a fixed
        # phase relationship between episode start and the wave cycle.
        if DISTURBED:
            self._disturb_t[env_ids]  = 0.0
            self._wave_phase[env_ids] = torch.rand(n, device=D) * (2.0 * math.pi)

        # ── Spawn casualties ───────────────────────────────────────────────
        r_range = cfg.casualty_spawn_r_max - cfg.casualty_spawn_r_min
        r   = torch.rand(n, C, device=D) * r_range + cfg.casualty_spawn_r_min
        ang = torch.rand(n, C, device=D) * 2.0 * math.pi

        cas_x = r * torch.cos(ang) + origin[:, 0:1]
        cas_y = r * torch.sin(ang) + origin[:, 1:2]
        cas_z = torch.zeros(n, C, device=D)
        self.casualty_pos[env_ids] = torch.stack([cas_x, cas_y, cas_z], dim=-1)

        # Random timers
        t_range = cfg.casualty_timer_max - cfg.casualty_timer_min
        timers  = torch.rand(n, C, device=D) * t_range + cfg.casualty_timer_min
        self.casualty_timer[env_ids]      = timers
        self.casualty_init_timer[env_ids] = timers.clone()
        self.casualty_alive[env_ids]      = True

        # Reset stats
        self.rescued_count[env_ids] = 0.0
        self.lost_count[env_ids]    = 0.0

        # Init priority distance
        p_idx   = self._priority_order()
        c0_idx  = p_idx[env_ids, 0]
        c0_pos  = self.casualty_pos[env_ids, c0_idx]
        self.prev_dist_priority[env_ids] = torch.norm(
            pos[:, :2] - c0_pos[:, :2], dim=-1
        )

    @property
    def rescue_rate(self) -> float:
        """Fraction of casualties rescued this episode (eval_benchmark.py metric)."""
        return float((self.rescued_count / self.cfg.n_casualties).mean().item())
