"""GPU-batched MPPI (Model Predictive Path Integral) baseline for USVBench.

INFORMATION-DIET DISCLOSURE (same diet as classical_planner_pid.py, published
beside every number): the controller consumes PRIVILEGED ground truth -- the
boat pose/velocities, the goal (or the env's ordered gate chain), and the
per-episode obstacle centers/radii list. It is a "control with a perfect map
and a nominal model" floor, NOT an observation-only baseline.

The internal dynamics model is NOMINAL and FROZEN: it reproduces the deployed
calm-water BlueBoat default physics and deliberately does NOT track any Suite D
parameter shift (mass_scale, drag_scale, thrust_cap_scale, motor_tau_s,
thrust_imbalance are all ignored even when the evaluated env applies them).
That mismatch is the point of the baseline.

Nominal model: 3-DOF planar rigid body, state [x, y, yaw, u, v, r] with world
position (x, y), yaw, body-frame surge/sway velocity (u, v) and yaw rate r.
Integration mirrors the deployed env exactly: forces recomputed and
semi-implicit-Euler stepped at the 1/120 s physics dt, 2 substeps per control
step (60 Hz control), because DirectRLEnv calls _apply_action() inside the
decimation loop once per physics step.

Every parameter is sourced from the repo (no invented constants):

  mass 17.26 kg          assets/blueboat_physics.usd /World/BlueBoat MassAPI
                         (registry keeps mass_kg=None to preserve the USD
                         author: tasks/_shared/vehicles.py:103; the datasheet
                         number 17.3 kg is documented in
                         tasks/blueboat_calm_nav/STARTER_TASK.md:33 and
                         blueboat_calm_nav_env_cfg.py:125)
  Izz  2.376 kg m^2      same USD MassAPI diagonalInertia=(1.789,1.094,2.376);
                         CAD-derived, intentionally not overridden per
                         tasks/blueboat_calm_nav/STARTER_TASK.md:34
  thrust fwd 80.0 N      tasks/_shared/vehicles.py:104 (two M200 thrusters,
                         ~40 N each: STARTER_TASK.md:38)
  thrust rev 48.0 N      tasks/_shared/vehicles.py:105 (60% of forward,
                         propeller asymmetry: STARTER_TASK.md:39)
  yaw torque 23.0 N m    tasks/_shared/vehicles.py:106; differential-thrust
                         geometry, CAD half-spacing lever 0.3607 m *
                         (40.2 + 24) N = 23.16: STARTER_TASK.md:40
  surge drag -(9.1+5.9|u|)u   vehicles.py:107-108; two-anchor calibration
                         (3.0 m/s terminal @ 80.4 N; 15 N @ 1 m/s):
                         STARTER_TASK.md:56-69
  sway drag -(27+18|v|)v      vehicles.py:109-110; ~3x surge catamaran
                         estimate, labeled "pending system identification"
                         in STARTER_TASK.md:42 (the env deploys it as-is,
                         so nominal-vs-env is still exact)
  yaw drag -(4+6|r|)r         vehicles.py:112-113; "plausible-order estimate"
                         per STARTER_TASK.md:44, deployed as-is
  physics dt 1/120 s     tasks/path_following/path_following_env_cfg.py:191
  decimation 2           tasks/path_following/path_following_env_cfg.py:170
  action map             path_following_env.py:369-377 (a0>=0 scales 80 N,
                         a0<0 scales 48 N along the bow axis; a1 scales the
                         23 N m body-z torque; identical block in
                         hazard_nav_env.py _apply_action with the Suite D
                         overrides inert at defaults)
  drag decomposition     path_following_env.py:407-419 (body surge/sway) and
                         :422-426 (yaw), recomputed per physics step
  half beam 0.45 m       tasks/hazard_nav/hazard_nav_env_cfg.py:173 and
                         hazard_geometry.py:19 (clearance convention
                         ||p-c|| - radius - 0.45, hazard_geometry.py:1828)

Closed-form cross-check baked into the unit tests: terminal surge speed
solves 5.9 u^2 + 9.1 u = 80 -> u = 2.991 m/s, which is exactly V_MAX in
scripts/usv10k_score.py:28 ("drag-limited top speed from the hydro
calibration"). Terminal yaw rate solves 6 r^2 + 4 r = 23 -> r = 1.653 rad/s.

Deliberately NOT modeled (planar nominal): buoyancy/heave, roll/pitch
restoring and damping, the 4.7 cm COM offset, PhysX contact forces, and the
non-binding velocity safety bounds (8 m/s, 10 rad/s) that the boat never
reaches. None of these couple into calm-water planar motion at nominal.

MPPI (Williams et al., 2017, "Information-Theoretic MPC"): K perturbed action
sequences ~ N(U, diag(sigma)^2) clipped to [-1,1]^2, rolled out H control
steps under the nominal model, scored, softmax-weighted with temperature
lambda, and the weighted-average sequence's FIRST action is executed;
the averaged sequence (shifted one step, last step repeated) warm-starts the
next control step. Everything is batched [N_env, K, ...]; the only python
loop is over the horizon H (and the 2 physics substeps inside the model).

Cost (per rollout), with the route potential
phi_t = dist(pos_t, active target) + remaining route length after it
(the remainder telescopes away as gates pass; zero for goal/hold/dock):

    w_goal * sum_t phi_t
  + w_obstacle * sum_t relu(margin - clearance_t)^2
  + collision_cost * sum_t 1[clearance_t < 0]
  + w_effort * sum_t |a_t|^2
  + w_terminal * phi_H
  - w_gate * (gates/goals newly reached in the rollout)
  + (hold mode only) w_speed_hold * sum_t speed^2 inside the hold radius
  + (dock mode, and the berth leg of mission mode) the TERMINAL POSE cost
      w_dock_heading * sum_t (1 - cos psi_t)                    [UNGATED]
    + w_dock_speed   * sum_t speed_t^2 * 1[dist_t <= goal_radius]
    where psi is the angle between the bow and the berth's dock_heading.

Why docking needs a terminal POSE cost and not a goal-distance cost: the
docking success predicate is a CONJUNCTION of position, heading and speed
held for five continuous seconds (tasks/docking/docking_env.py:646-658 and
:776-789), so a controller that merely arrives inside the 2.5 m tolerance
scores zero. Arriving pointed the wrong way, or arriving fast, is not a
berthing.

THE TWO TERMS ARE GATED DIFFERENTLY, and that asymmetry is the whole
design. Heading is a state the boat can fix IN PLACE; approach speed is the
means of entry. So:

  * the heading term is UNGATED -- it depends on yaw only, never on
    position, so its gradient with respect to distance is identically zero
    and it can never oppose closing on the berth;
  * the speed term is gated by 1[dist <= goal_radius], the same hard
    indicator hold mode already uses for w_speed_hold, so the long approach
    runs at full speed and only the last 2.5 m are braked. That gate IS a
    step at the boundary, but its height w_dock_speed * v^2 goes to zero as
    the boat slows -- and slowing down before entering a berth is the
    behaviour we want, so the barrier is one the boat removes for free.

WEIGHT DERIVATION (both anchored at the success speed tolerance and the
horizon; neither ever fitted to an evaluation pack):

  w_dock_heading = w_goal * v_tol * dt * (H - 1) / 2 = 0.15
      Read it as: even at the SLOWEST speed the boat ever needs to close at
      (v_tol = 0.3 m/s), turning fully abeam to chase a purely lateral berth
      offset must still pay for itself. Over H steps, closing at v buys
      w_goal * v * dt * H * (H-1) / 2 of position cost while a 90 degree bow
      offset costs w_dock_heading * (1 - cos 90 deg) * H; equating them at
      v_tol gives 0.3 * (1/60) * 59 / 2 = 0.1475, shipped as 0.15. Tiny
      against the approach, dominant at the berth -- where the position term
      has vanished and this is the only term left -- and a per-step cost
      always beats the one-off effort burst of a turn (w_effort * |a|^2 is
      at most 0.04 for a fraction of a second).

  w_dock_speed = w_goal * (H - 1) * dt / (2 * v_tol) = 59 / 60 / 0.6 = 1.64
      The same balance solved for the speed term instead: equating the two
      at v_tol makes 0.300 m/s the BREAK-EVEN approach speed inside the
      berth. Above it, slowing pays; below it, closing pays. It lands next
      to hold mode's certified w_speed_hold = 0.6 -- the same
      speed-squared-inside-a-radius term, already solved once for this hull
      at this horizon.

Both defaults are quoted at the shipped H = 60 and dt = 1/60 s and at the
deployed tolerances, which are identical in the two families:
tasks/docking/docking_env_cfg.py:158-163 and
tasks/harbor_mission/harbor_mission_env_cfg.py:164-168.

REJECTED ATTEMPT 1 -- a smooth proximity gate exp(-dist / R), R =
goal_radius, borrowed from the env's own reference alignment/braking credit
(docking_env.py:710-730), with equal-tolerance weights (w_dock_heading =
2.5 / (1 - cos 15 deg) = 73.4, w_dock_speed = 2.5 / 0.3^2 = 27.8). A gate
that GROWS as the boat closes is a position barrier: the stationary point
of dist + W*(1 - cos psi)*exp(-d/R) sits at R*ln(W*(1 - cos psi)/R). The
CPU toy parked at 10.02 m -- perfectly aligned, stopped, 10 m short -- on
both +/-60 degree spawn bearings, and at 4.15 m on a 30 degree one.

REJECTED ATTEMPT 2 -- the same equal-tolerance weights on the hard
indicator, both terms gated. Scaling each term to bite at its own tolerance
is the obvious construction and it walls the boundary: at the rim the
position benefit of entering is ZERO while the heading penalty is already
full, so a misaligned boat may only enter if it is ALREADY within 15
degrees, and with the gate on, nothing outside the disk rewards aligning.
The toy parked at exactly d = 2.50 m on four of five spawns, bow at
dot = -0.82 on one. Dropping the pair to w_dock_heading = 1.25 (a reversed
bow costing what the position tolerance costs) and w_dock_speed = 1.64
recovered four of the five, and still parked the 30 degree spawn at exactly
d = 2.50 m with the bow 89 degrees off: ANY position-gated heading term has
this rim, at any weight, because the rim is where the term's benefit is
zero and its penalty is already full. That is why the shipped heading term
is ungated instead -- and why the shipped speed term may stay gated, its
rim being one the boat erases by slowing down.

All three toys are 2026-08-24, CPU, nominal model as the world, dev spawns
only; no weight here has ever been run against a Suite D or any other
evaluation pack.

dock -- and mission's berth leg -- deliberately do NOT use the reached/done
freeze that gates and goal use: success needs FIVE seconds of continuous
conjunction, far beyond the 1.0 s horizon, so the rollout must keep paying
for a bad pose at every step, exactly as hold mode does. A perfect berth
(on the point, on heading, stopped) has stage cost zero anyway, so
finishing early is still rewarded.

mission mode chains the two: an ORDERED gate list scored by the env's own
crossing rule (tasks/harbor_mission/harbor_geometry.py:175-223 -- adjacent
samples must straddle the plane in the +normal direction, the interpolated
intersection must lie inside the half-width window, and the normal speed
must exceed the minimum), followed by the dock leg. A gate crossed from the
WRONG SIDE is never credited, because the straddle test is signed.
The one part of the env rule the rollout does not re-derive is the 1 m
approach latch (arm_gate_approach): the env sets it at reset and re-sets it
at every gate advance, and on the shipped harbor routes consecutive gates
are >= 10 m apart (harbor_geometry.py:42-45, :286-299), so the latch is
never the binding condition -- modelling it from a horizon that cannot see
the past would instead UN-credit a gate the boat is already 0.5 m short of
and pull it back across the plane.

Why a route POTENTIAL and not plain distance-to-active-gate: with plain
distance, a sample that passes gate g mid-rollout pays gate g+1's distance
for every remaining step, so "touch the radius exactly at the horizon end"
strictly dominates "pass it now" -- the closed loop then brakes to a
permanent park at 2.02 m outside the first gate (observed in the CPU toy,
kept as a regression test). The potential is continuous across the switch,
so earlier passing is always at least as good; w_gate (50) is only a
tie-breaker that makes actually entering the 2.0 m radius strictly better
than skimming past it, and stage costs freeze once a rollout finishes the
last target, which rewards finishing early.

Default calibration note (CPU toys, nominal dev seeds only): at 60 Hz a
0.5 s horizon separates "turn toward an abeam goal" from "do nothing" by
only ~1-3 cost units, so lambda must sit well below that spread and the
terminal weight must amplify it; and a horizon that cannot see around an
obstacle's 2.05 m keep-out bubble parks the boat at the margin boundary in a
textbook local minimum (observed at H=30/40). The shipped defaults
K=256, H=60 (1.0 s), sigma=(0.5, 0.8), lambda=0.3, w_terminal=20 reach
open-water goals at 0/90/180 degree bearings in 3.1-3.7 s, finish a two-gate
chain in 6.7 s, hold station to 0.01 m, and round a 1 m obstacle with 0.5 m
executed clearance in the CPU toys. lambda=3.0 with H=30 demonstrably stalls
on a goal 90 degrees abeam -- kept as a regression note.

The same defaults, unchanged, plus the two berthing weights below, hold the
full docking conjunction on five approach-lane spawns out of five (25 m out
at 0 and +/-60 degree bearings with up to 45 degrees of bow offset, and two
close-in ones), first entering the success set 3.2-15.3 s in and holding it
45-57 s with the bow inside 5 degrees and the hull under 0.03 m/s; and they
run a fabricated 60 m harbor mission -- three ordered gates then a berth,
with the gates credited by the DEPLOYED env's own gate_crossing_mask rather
than by this file's opinion -- crossing M1 at 5.9 s and M2 at 18.7 s (22.3 s
through a three-cylinder field, 0.71 m minimum clearance) and berthing to
0.00 m. All CPU, all 2026-08-24, nominal model as the world.

TUNING DISCIPLINE: cost weights and sigma/lambda defaults were tuned only on
CPU toy scenarios driven by this same nominal model (open water, no Isaac)
plus nominal dev seeds; they must never be tuned against shifted (Suite D)
evaluation packs. w_dock_heading and w_dock_speed were never fitted at all:
they are closed-form consequences of the horizon and the deployed success
tolerances (see the derivation above), and the CPU toys were used only to
falsify two earlier GATING choices, never to search a weight.

This module imports torch at module scope and must therefore only be imported
AFTER AppLauncher on the Isaac boxes (scripts/eval_mppi.py does exactly that);
CPU tests import it directly with no Isaac anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# Nominal BlueBoat parameters. Sources: module docstring table (file:line).
NOMINAL_BLUEBOAT = {
    "mass_kg": 17.26,
    "izz_kgm2": 2.376,
    "thrust_fwd_n": 80.0,
    "thrust_rev_n": 48.0,
    "yaw_torque_nm": 23.0,
    "surge_lin": 9.1,
    "surge_quad": 5.9,
    "sway_lin": 27.0,
    "sway_quad": 18.0,
    "yaw_lin": 4.0,
    "yaw_quad": 6.0,
    "physics_dt_s": 1.0 / 120.0,
    "decimation": 2,
    "half_beam_m": 0.45,
}

STATE_DIM = 6  # [x, y, yaw, u, v, r]
ACTION_DIM = 2  # [surge command, yaw command], each in [-1, 1]

# Berthing success tolerances, identical in both docking families:
# tasks/docking/docking_env_cfg.py:158-163 (Dock) and
# tasks/harbor_mission/harbor_mission_env_cfg.py:164-168 (HarborMission).
DOCK_POSITION_TOLERANCE_M = 2.5
DOCK_HEADING_TOLERANCE_DEG = 15.0
DOCK_SPEED_TOLERANCE_MPS = 0.3
DOCK_HOLD_TIME_S = 5.0

# Harbor gate geometry, from tasks/harbor_mission/harbor_geometry.py:46-49
# (and mirrored in harbor_mission_env_cfg.py:170-172). Passed per call by
# scripts/eval_mppi.py from the live cfg; these are the documented defaults.
GATE_HALF_WIDTH_M = 3.0
GATE_MIN_NORMAL_SPEED_MPS = 0.2


def terminal_speed(thrust_n: float, lin: float, quad: float) -> float:
    """Positive root of quad*v^2 + lin*v = thrust (drag-limited speed)."""
    if thrust_n <= 0.0:
        return 0.0
    return (-lin + math.sqrt(lin * lin + 4.0 * quad * thrust_n)) / (2.0 * quad)


class NominalBlueBoatDynamics:
    """Frozen nominal 3-DOF planar BlueBoat model (torch, any batch shape).

    step() advances one CONTROL step = ``decimation`` semi-implicit Euler
    substeps at ``physics_dt_s`` with the action held and drag recomputed per
    substep, mirroring the deployed env's per-physics-step _apply_action().
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        params: dict | None = None,
    ) -> None:
        p = dict(NOMINAL_BLUEBOAT)
        if params:
            unknown = set(params) - set(p)
            if unknown:
                raise ValueError(f"unknown dynamics params: {sorted(unknown)}")
            p.update(params)
        if p["mass_kg"] <= 0.0 or p["izz_kgm2"] <= 0.0:
            raise ValueError("mass and yaw inertia must be positive")
        if p["physics_dt_s"] <= 0.0 or int(p["decimation"]) < 1:
            raise ValueError("physics_dt_s must be positive, decimation >= 1")
        self.params = p
        self.device = torch.device(device)
        self.dtype = dtype
        self.control_dt_s = p["physics_dt_s"] * int(p["decimation"])

    def terminal_speed_forward(self) -> float:
        p = self.params
        return terminal_speed(p["thrust_fwd_n"], p["surge_lin"], p["surge_quad"])

    def terminal_speed_reverse(self) -> float:
        p = self.params
        return terminal_speed(p["thrust_rev_n"], p["surge_lin"], p["surge_quad"])

    def terminal_yaw_rate(self) -> float:
        p = self.params
        return terminal_speed(p["yaw_torque_nm"], p["yaw_lin"], p["yaw_quad"])

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """state [..., 6], action [..., 2] -> next state [..., 6]."""
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"state last dim must be {STATE_DIM}")
        if action.shape[-1] != ACTION_DIM:
            raise ValueError(f"action last dim must be {ACTION_DIM}")
        p = self.params
        dt = p["physics_dt_s"]
        inv_m = 1.0 / p["mass_kg"]
        inv_izz = 1.0 / p["izz_kgm2"]

        a0 = action[..., 0].clamp(-1.0, 1.0)
        a1 = action[..., 1].clamp(-1.0, 1.0)
        # Asymmetric thrust map: path_following_env.py:369-374.
        thrust = torch.where(
            a0 >= 0.0, a0 * p["thrust_fwd_n"], a0 * p["thrust_rev_n"]
        )
        tau_cmd = a1 * p["yaw_torque_nm"]

        x = state[..., 0]
        y = state[..., 1]
        yaw = state[..., 2]
        u = state[..., 3]
        v = state[..., 4]
        r = state[..., 5]

        for _ in range(int(p["decimation"])):
            # Body-frame forces from CURRENT body velocities (drag recomputed
            # each physics step, as the env does).
            f_surge = thrust - (p["surge_lin"] + p["surge_quad"] * u.abs()) * u
            f_sway = -(p["sway_lin"] + p["sway_quad"] * v.abs()) * v
            tau = tau_cmd - (p["yaw_lin"] + p["yaw_quad"] * r.abs()) * r

            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            # World-frame velocity and force (PhysX integrates in world frame).
            vwx = u * cos_y - v * sin_y + (f_surge * cos_y - f_sway * sin_y) * (inv_m * dt)
            vwy = u * sin_y + v * cos_y + (f_surge * sin_y + f_sway * cos_y) * (inv_m * dt)
            # Semi-implicit Euler: positions advance with the NEW velocity.
            x = x + vwx * dt
            y = y + vwy * dt
            r = r + tau * (inv_izz * dt)
            yaw = yaw + r * dt
            # Re-project the (unchanged) world velocity onto the NEW heading.
            cos_n = torch.cos(yaw)
            sin_n = torch.sin(yaw)
            u = vwx * cos_n + vwy * sin_n
            v = -vwx * sin_n + vwy * cos_n

        return torch.stack((x, y, yaw, u, v, r), dim=-1)


@dataclass
class MPPIParams:
    """MPPI hyper-parameters. Tuned ONLY on nominal CPU toys / dev seeds."""

    K: int = 256
    H: int = 60
    sigma: tuple[float, float] = (0.50, 0.80)
    lambda_: float = 0.3
    w_goal: float = 1.0
    w_gate: float = 50.0
    w_obstacle: float = 40.0
    clearance_margin_m: float = 0.60
    collision_cost: float = 2000.0
    w_effort: float = 0.02
    w_terminal: float = 20.0
    w_speed_hold: float = 0.6
    # Terminal-pose weights for dock / mission's berth leg, derived in the
    # module docstring at the deployed 0.3 m/s speed tolerance and the
    # shipped H=60 / dt=1/60 horizon; never tuned against an evaluation pack.
    #   heading (UNGATED):        v_tol * dt * (H-1) / 2 = 0.15
    #   speed   (inside the disk): dt * (H-1) / (2 * v_tol) = 1.64
    w_dock_heading: float = 0.15
    w_dock_speed: float = 1.64
    seed: int = 0

    def validate(self) -> None:
        if self.K < 2 or self.H < 1:
            raise ValueError("K must be >= 2 and H >= 1")
        if min(self.sigma) <= 0.0 or self.lambda_ <= 0.0:
            raise ValueError("sigma components and lambda must be positive")
        if self.clearance_margin_m < 0.0:
            raise ValueError("clearance_margin_m must be non-negative")
        if self.w_dock_heading < 0.0 or self.w_dock_speed < 0.0:
            raise ValueError("dock pose weights must be non-negative")

    def as_dict(self) -> dict:
        return {
            "K": self.K,
            "H": self.H,
            "sigma": list(self.sigma),
            "lambda": self.lambda_,
            "w_goal": self.w_goal,
            "w_gate": self.w_gate,
            "w_obstacle": self.w_obstacle,
            "clearance_margin_m": self.clearance_margin_m,
            "collision_cost": self.collision_cost,
            "w_effort": self.w_effort,
            "w_terminal": self.w_terminal,
            "w_speed_hold": self.w_speed_hold,
            "w_dock_heading": self.w_dock_heading,
            "w_dock_speed": self.w_dock_speed,
            "seed": self.seed,
        }


class MPPIController:
    """Batched MPPI over the nominal model. Pure torch; no env dependency.

    Task inputs (targets, obstacles) are passed per call so the same object
    drives PathFollow (mode="gates"), HazardCross/HazardNav (mode="goal"),
    StationKeep (mode="hold"), Dock (mode="dock") and HarborMission /
    HarborStage1-3 (mode="mission"). No python loops over K or env anywhere.
    """

    MODES = ("gates", "goal", "hold", "dock", "mission")

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cpu",
        params: MPPIParams | None = None,
        dynamics: NominalBlueBoatDynamics | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.params = params or MPPIParams()
        self.params.validate()
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.device = torch.device(device)
        self.dtype = dtype
        self.dynamics = dynamics or NominalBlueBoatDynamics(self.device, dtype)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(self.params.seed))
        self.U = torch.zeros(
            (self.num_envs, self.params.H, ACTION_DIM),
            device=self.device,
            dtype=dtype,
        )

    def reset(self, done_mask: torch.Tensor) -> None:
        """Zero the warm-start sequence of freshly reset envs."""
        mask = done_mask.to(self.device).reshape(-1).bool()
        if mask.shape[0] != self.num_envs:
            raise ValueError("done_mask must have num_envs entries")
        if bool(mask.any()):
            self.U[mask] = 0.0

    def rollout(
        self,
        state: torch.Tensor,
        K: int | None = None,
        H: int | None = None,
        sigma: tuple[float, float] | None = None,
        lambda_: float | None = None,
        *,
        targets: torch.Tensor,
        active_index: torch.Tensor | None = None,
        goal_radius: float,
        mode: str = "goal",
        obstacle_centers: torch.Tensor | None = None,
        obstacle_radii: torch.Tensor | None = None,
        obstacle_active: torch.Tensor | None = None,
        dock_heading: torch.Tensor | None = None,
        gate_normals: torch.Tensor | None = None,
        gate_half_width_m: float = GATE_HALF_WIDTH_M,
        gate_min_normal_speed_mps: float = GATE_MIN_NORMAL_SPEED_MPS,
    ) -> tuple[torch.Tensor, dict]:
        """Sample, roll out, weight; return (first action [N,2], info dict).

        state         [N, 6]  [x, y, yaw, u, v, r] (world pose, body vel)
        targets       [N, G, 2] world-frame ordered targets (G=1 for
                      goal/hold/dock; the env's waypoint chain for gates;
                      the ordered gate midpoints followed by the berth
                      point for mission)
        active_index  [N] long, the env's own progress counter (None ->
                      zeros): gates_passed for gates, the resolved
                      gate/berth leg index for mission
        obstacle_*    [N, M, 2] / [N, M] / [N, M] privileged obstacle list;
                      clearance = ||p - c|| - radius - half_beam (0.45 m),
                      matching hazard_geometry.analytic_min_clearance.
        dock_heading  [N, 2] unit bow direction the berth requires; REQUIRED
                      by dock and mission, ignored by the other modes.
        gate_normals  [N, G-1, 2] signed outward normals of the ordered
                      gates; REQUIRED by mission, ignored elsewhere. A
                      crossing counts only in the +normal direction.
        gate_half_width_m / gate_min_normal_speed_mps
                      the env's gate window and minimum normal speed
                      (mission only), so the rollout credits exactly the
                      crossings tasks/harbor_mission/harbor_geometry.py
                      gate_crossing_mask would credit.
        """
        p = self.params
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if state.dim() != 2 or state.shape != (self.num_envs, STATE_DIM):
            raise ValueError(f"state must be [{self.num_envs}, {STATE_DIM}]")
        if targets.dim() != 3 or targets.shape[0] != self.num_envs or targets.shape[-1] != 2:
            raise ValueError("targets must be [num_envs, G, 2]")
        if goal_radius <= 0.0:
            raise ValueError("goal_radius must be positive")
        berth_leg = mode in ("dock", "mission")
        if berth_leg:
            if dock_heading is None:
                raise ValueError(f"mode {mode!r} requires dock_heading [N, 2]")
            if dock_heading.shape != (self.num_envs, 2):
                raise ValueError(f"dock_heading must be [{self.num_envs}, 2]")
        if mode == "dock" and targets.shape[1] != 1:
            raise ValueError("dock mode takes exactly one target (the berth)")
        if mode == "mission":
            if targets.shape[1] < 2:
                raise ValueError(
                    "mission mode needs >= 1 gate plus the berth in targets"
                )
            if gate_normals is None:
                raise ValueError("mission mode requires gate_normals [N, G-1, 2]")
            if gate_normals.shape != (self.num_envs, targets.shape[1] - 1, 2):
                raise ValueError(
                    "gate_normals must be [num_envs, targets_G - 1, 2]"
                )
            if gate_half_width_m <= 0.0:
                raise ValueError("gate_half_width_m must be positive")
        K = int(K if K is not None else p.K)
        H = int(H if H is not None else p.H)
        sig = sigma if sigma is not None else p.sigma
        lam = float(lambda_ if lambda_ is not None else p.lambda_)
        if K < 2 or H < 1 or min(sig) <= 0.0 or lam <= 0.0:
            raise ValueError("invalid K/H/sigma/lambda override")

        N = self.num_envs
        device, dtype = self.device, self.dtype
        state = state.to(device=device, dtype=dtype)
        targets = targets.to(device=device, dtype=dtype)
        G = targets.shape[1]
        if active_index is None:
            idx0 = torch.zeros(N, dtype=torch.long, device=device)
        else:
            idx0 = active_index.to(device=device, dtype=torch.long).clamp(0, G - 1)

        have_obstacles = (
            obstacle_centers is not None
            and obstacle_radii is not None
            and obstacle_centers.shape[1] > 0
        )
        if have_obstacles:
            centers = obstacle_centers.to(device=device, dtype=dtype)
            eff_radii = obstacle_radii.to(device=device, dtype=dtype) + float(
                self.dynamics.params["half_beam_m"]
            )
            if obstacle_active is not None:
                inactive = ~obstacle_active.to(device=device, dtype=torch.bool)
                eff_radii = eff_radii.masked_fill(inactive, -1.0e6)

        # Route potential: stage cost charges distance-to-active-target PLUS
        # the remaining route length after it, so passing a gate telescopes
        # the potential smoothly instead of JUMPING to the next gate's
        # distance. Without this, "touch the radius exactly at the horizon
        # end" dominates "pass it now" (every post-switch step pays the next
        # gate's distance), and the closed loop brakes to a permanent park at
        # 2.02 m -- observed, not hypothetical. G=1 (goal/hold) reduces to a
        # zero remainder.
        if G > 1:
            seg_len = torch.linalg.vector_norm(
                targets[:, 1:, :] - targets[:, :-1, :], dim=-1
            )
            route_rem = torch.zeros((N, G), device=device, dtype=dtype)
            route_rem[:, :-1] = seg_len.flip(1).cumsum(1).flip(1)
        else:
            route_rem = torch.zeros((N, G), device=device, dtype=dtype)

        if self.U.shape[1] != H:  # H override: rebuild the warm start
            self.U = torch.zeros((N, H, ACTION_DIM), device=device, dtype=dtype)

        sigma_t = torch.as_tensor(sig, device=device, dtype=dtype)
        noise = (
            torch.randn((N, K, H, ACTION_DIM), generator=self.generator, device=device, dtype=dtype)
            * sigma_t
        )
        seq = (self.U[:, None, :, :] + noise).clamp(-1.0, 1.0)

        s = state[:, None, :].expand(N, K, STATE_DIM).reshape(N * K, STATE_DIM).clone()
        idx = idx0[:, None].expand(N, K).clone()
        done = torch.zeros((N, K), dtype=torch.bool, device=device)
        cost = torch.zeros((N, K), device=device, dtype=dtype)
        min_clearance = torch.full((N, K), math.inf, device=device, dtype=dtype)

        if berth_leg:
            # Bow direction the berth requires, broadcast over the K samples.
            dock_h = dock_heading.to(device=device, dtype=dtype)[:, None, :]
        if mode == "mission":
            # One dummy normal row keeps gather() legal on the berth index;
            # every value read from it is discarded by ~at_last below.
            normals = torch.zeros((N, G, 2), device=device, dtype=dtype)
            normals[:, : G - 1, :] = gate_normals.to(device=device, dtype=dtype)
            pos_prev = state[:, None, :2].expand(N, K, 2).clone()
            half_width = float(gate_half_width_m)
            min_normal_speed = float(gate_min_normal_speed_mps)

        for t in range(H):
            a = seq[:, :, t, :]
            s = self.dynamics.step(s, a.reshape(N * K, ACTION_DIM))
            pos = s.reshape(N, K, STATE_DIM)[..., :2]
            tgt = torch.gather(targets, 1, idx.unsqueeze(-1).expand(N, K, 2))
            dist = torch.linalg.vector_norm(pos - tgt, dim=-1)
            potential = dist + torch.gather(route_rem, 1, idx)

            stage = p.w_goal * potential + p.w_effort * (a * a).sum(dim=-1)
            if have_obstacles:
                sep = torch.linalg.vector_norm(
                    pos.unsqueeze(2) - centers.unsqueeze(1), dim=-1
                )
                clearance = (sep - eff_radii.unsqueeze(1)).amin(dim=-1)
                min_clearance = torch.minimum(min_clearance, clearance)
                stage = stage + p.w_obstacle * torch.relu(
                    p.clearance_margin_m - clearance
                ).square() + p.collision_cost * (clearance < 0.0).to(dtype)
            if mode == "hold":
                body_uv = s.reshape(N, K, STATE_DIM)[..., 3:5]
                inside = (dist <= goal_radius).to(dtype)
                stage = stage + p.w_speed_hold * (body_uv * body_uv).sum(dim=-1) * inside
            elif berth_leg:
                # TERMINAL POSE. Heading is UNGATED -- it depends on yaw
                # only, so it has no distance gradient and can never oppose
                # closing on the berth; approach speed is gated by the same
                # hard indicator hold mode uses, so only the last
                # goal_radius metres are braked. Any position gate on the
                # heading term walls the rim at every weight; see the module
                # docstring's two rejected attempts. The bow unit vector is
                # (cos yaw, sin yaw) by construction of the nominal state, so
                # this dot product is the env's own dock_dot
                # (tasks/docking/docking_env.py:651).
                full = s.reshape(N, K, STATE_DIM)
                yaw_now = full[..., 2]
                body_uv = full[..., 3:5]
                align = (
                    torch.cos(yaw_now) * dock_h[..., 0]
                    + torch.sin(yaw_now) * dock_h[..., 1]
                )
                heading_cost = p.w_dock_heading * (1.0 - align)
                speed_cost = (
                    p.w_dock_speed
                    * (body_uv * body_uv).sum(dim=-1)
                    * (dist <= goal_radius).to(dtype)
                )
                pose = heading_cost + speed_cost
                if mode == "mission":
                    # Gate legs pay nothing: there dist is a gate's, not the
                    # berth's, and heading through a gate is unconstrained.
                    pose = pose * (idx >= (G - 1)).to(dtype)
                stage = stage + pose

            cost = cost + torch.where(done, torch.zeros_like(stage), stage)

            if mode == "mission":
                # The env's own crossing rule, sample for sample:
                # harbor_geometry.gate_crossing_mask:175-223. Signed straddle
                # in the +normal direction only, interpolated intersection
                # inside the half-width window, normal speed above the floor.
                # The berth leg (at_last) never advances and never freezes:
                # five seconds of held pose are what finishes the mission.
                gate_n = torch.gather(normals, 1, idx.unsqueeze(-1).expand(N, K, 2))
                prev_signed = ((pos_prev - tgt) * gate_n).sum(dim=-1)
                cur_signed = ((pos - tgt) * gate_n).sum(dim=-1)
                denominator = cur_signed - prev_signed
                straddles = (
                    (prev_signed <= 0.0)
                    & (cur_signed >= 0.0)
                    & (denominator > 1.0e-9)
                )
                fraction = torch.where(
                    straddles,
                    -prev_signed / denominator.clamp_min(1.0e-9),
                    torch.zeros_like(denominator),
                )
                intersection = pos_prev + fraction.unsqueeze(-1) * (pos - pos_prev)
                tangent = torch.stack((-gate_n[..., 1], gate_n[..., 0]), dim=-1)
                lateral = ((intersection - tgt) * tangent).sum(dim=-1).abs()
                full = s.reshape(N, K, STATE_DIM)
                cos_now = torch.cos(full[..., 2])
                sin_now = torch.sin(full[..., 2])
                world_vx = full[..., 3] * cos_now - full[..., 4] * sin_now
                world_vy = full[..., 3] * sin_now + full[..., 4] * cos_now
                normal_speed = world_vx * gate_n[..., 0] + world_vy * gate_n[..., 1]
                at_last = idx >= (G - 1)
                crossed = (
                    straddles
                    & (lateral <= half_width)
                    & (normal_speed > min_normal_speed)
                    & ~at_last
                )
                cost = cost - p.w_gate * crossed.to(dtype)
                idx = torch.where(crossed, idx + 1, idx)
                pos_prev = pos
            elif mode not in ("hold", "dock"):
                reached = (dist <= goal_radius) & ~done
                cost = cost - p.w_gate * reached.to(dtype)
                at_last = idx >= (G - 1)
                done = done | (reached & at_last)
                idx = torch.where(reached & ~at_last, idx + 1, idx)

        pos = s.reshape(N, K, STATE_DIM)[..., :2]
        tgt = torch.gather(targets, 1, idx.unsqueeze(-1).expand(N, K, 2))
        final_dist = torch.linalg.vector_norm(pos - tgt, dim=-1)
        final_potential = final_dist + torch.gather(route_rem, 1, idx)
        cost = cost + torch.where(
            done,
            torch.zeros_like(final_potential),
            p.w_terminal * final_potential,
        )

        beta = cost.amin(dim=1, keepdim=True)
        weights = torch.softmax(-(cost - beta) / lam, dim=1)
        u_new = torch.einsum("nk,nkha->nha", weights, seq).clamp(-1.0, 1.0)
        action = u_new[:, 0, :].clone()
        # Receding horizon: shift left, repeat the last step.
        self.U = torch.cat((u_new[:, 1:, :], u_new[:, -1:, :]), dim=1)

        ess = 1.0 / (weights.square().sum(dim=1))
        info = {
            "cost_min": cost.amin(dim=1),
            "cost_mean": cost.mean(dim=1),
            "ess": ess,
            "rollout_done_frac": done.to(dtype).mean(dim=1),
            "rollout_min_clearance": min_clearance,
        }
        return action, info
