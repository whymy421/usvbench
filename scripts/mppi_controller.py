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

JOURNAL IDS (F7, CONTEXT_R4): the legacy description above applies to the
legacy task ids only. For journal cfgs the internal model is VesselDynamicsV2
(the D5 registry vessel through the D3 mixer, calm nominal, one model per
vessel; dynamics_for_cfg), the parameters are journal_mppi_params: the declared
length/time scaling of the BlueBoat reference values (mppi_params_for_vessel,
identity for the BlueBoat) plus, since round 4b, the per-vessel temperature
lambda frozen by the pre-declared MPPI_TUNING_PROCEDURE_V2.md in
mppi_params_v2.json (H stays the declared rule), the obstacle list is the ray block of the D5 estimate (capsule
clearance, VesselDynamicsV2.hull_clearance) and the state is the D5 estimate
(scripts/eval_mppi.py JournalEnvBridge): no privileged map, no sea, current or
wind.

This module imports torch at module scope and must therefore only be imported
AFTER AppLauncher on the Isaac boxes (scripts/eval_mppi.py does exactly that);
CPU tests import it directly with no Isaac anywhere.

CUDA GRAPH IMPLEMENTATION (G9): CudaGraphMPPI (end of this module) replays the
SAME MPPIController.rollout as one captured CUDA graph (stamp "cuda_graph_v1");
the noise keys, the sigma constant, the input staging and the counters stay on
the host, outside the graph, in the eager order. Same keyed noise, same
arithmetic, same actions; eval_mppi selects it with --mppi-impl (default: the
graph for journal ids on a CUDA device). Tested in scripts/tests_g9/.
"""

from __future__ import annotations

import copy
import json
import math
import time
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


# ---------------------------------------------------------------------------
# F7 (CONTEXT_R4, journal ids): the D5 registry vessel + the D3 mixer.
#
# The legacy NominalBlueBoatDynamics above is kept, byte for byte, for the
# LEGACY task ids (their env still runs the legacy independent surge/yaw
# channels of tasks/_shared/vehicles.py). Journal ids (cfg.journal_contract_
# version == 2, hydrodynamics_model "hydro_v2") run the D5 v2 registry vessel
# through the D3 per-motor mixer, so their MPPI model is VesselDynamicsV2,
# one instance per vessel, selected from the journal cfg's ``vehicle``
# (dynamics_for_cfg). Every number comes from the registry at construction:
#
#   mass, I_zz            vehicles_v2.get_vehicle_v2(name).mass_kg /
#                         inertia_com_kg_m2[2][2] (planar model: only I_zz)
#   motors                vehicles_v2.to_d3_thruster_kwargs(spec) -> D3
#                         thruster_mixer.ThrusterSpec (per-motor forward /
#                         reverse limits, centreline arm, yaw request scale;
#                         pitch coupling OFF, CONTEXT_R3 decision 6)
#   allocation            thruster_mixer.step_thrusters_prepared, RADIAL
#                         saturation (CONTEXT_R3 decision 4), nominal lag 0 s
#                         (both profiles; a nonzero nominal lag is refused
#                         because the 6-D rollout state has no motor stage)
#   damping               linear/quadratic surge, sway, yaw of the env's OWN
#                         hydro_v2 profile, hydro_profiles.profile_for_vehicle(
#                         name, swayyaw_damping=...) (registry values plus the
#                         G4-A5 sway/yaw rule: journal default = Sarda 2016
#                         cross-flow sway + strip-theory yaw added to the
#                         VRX-classic linear terms for the WAM-V option 1,
#                         registry values for every other vessel; the rule is
#                         read from cfg.journal_vehicle_options exactly as the
#                         env's resolve_journal_vehicle reads it), on
#                         the WATER-RELATIVE body velocity with CALM nominal
#                         water (u_w = 0; MPPI never observes waves, current
#                         or wind), frozen quadratic coefficient and one
#                         backward-Euler step per physics substep as
#                         hydro_v2.hydro_wrench / physics_v2.implicit_damping_
#                         wrench compute it; resistance scale 1 (nominal)
#   hull footprint        spec.collision capsule (spine along body x through
#                         the capsule centre, radius), COM frame
#   dt / decimation       1/120 s x 2 (CONTEXT_R4 D2)
#
# Terminal speed solves X = (lin + quad |u|) u with X = 2 * forward limit
# (= spec.speeds.top_model_m_s); terminal spin solves N = (lin_r + quad_r |r|)
# r with N = arm * 2 * reverse limit (radial saturation of the (0, 1)
# request) -- the E2 CPU_SCENARIOS anchors surge_60s_mps / spin_60s_rad_s.
# Deliberately NOT modelled (planar calm nominal): heave/roll/pitch, the
# inertia products of the WAM-V tensor, current, waves, wind, actuator doses.
# Tested in scripts/test_mppi_vessel_v2.py.
# ---------------------------------------------------------------------------
V2_PHYSICS_DT_S = 1.0 / 120.0
V2_DECIMATION = 2
V2_MODEL_NAME = "d5_registry_d3_mixer_calm_nominal_v1"
# G39 fix F2: side (cells) of the geodesic potential raster (scripts/mppi_g39_fix.GRID_CELLS).
GEODESIC_GRID_CELLS = 160


def _shared_module(name):
    """tasks/_shared/<name> via the package, the deployed tree, or by path."""
    import importlib
    for package in ("tasks._shared", "isaaclab_tasks.direct._shared"):
        try:
            return importlib.import_module(package + "." + name)
        except ImportError:
            continue
    import sys
    import types
    from pathlib import Path
    private = "_f7_mppi_shared"
    if private not in sys.modules:
        module = types.ModuleType(private)
        module.__path__ = [str(Path(__file__).resolve().parents[1] / "tasks" / "_shared")]
        sys.modules[private] = module
    return importlib.import_module(private + "." + name)


PLANAR_DAMPING_KEYS = ("surge_lin", "surge_quad", "sway_lin", "sway_quad", "yaw_lin", "yaw_quad")


def profile_planar_damping(profile) -> dict:
    """Planar damping of a hydro_profiles.VesselProfileV2, keyed as in vessel_params_v2.

    HydroSpec order: linear/quadratic_damping = (surge, sway, heave),
    angular_linear/quadratic_damping = (roll, pitch, yaw).
    """
    h = profile.hydro
    return {"surge_lin": float(h.linear_damping[0]), "surge_quad": float(h.quadratic_damping[0]),
            "sway_lin": float(h.linear_damping[1]), "sway_quad": float(h.quadratic_damping[1]),
            "yaw_lin": float(h.angular_linear_damping[2]), "yaw_quad": float(h.angular_quadratic_damping[2])}


def vessel_params_v2(vessel: str, swayyaw_damping: str = "journal_default") -> dict:
    """Registry-derived planar model parameters of one v2 vessel (JSON-native).

    Damping comes from the env's own hydro_v2 profile
    (hydro_profiles.profile_for_vehicle with the G4-A5 ``swayyaw_damping`` rule),
    so the MPPI model and the env physics carry identical surge/sway/yaw
    coefficients for the same rule.
    """
    registry = _shared_module("vehicles_v2")
    spec = registry.get_vehicle_v2(vessel)
    thrusters = registry.to_d3_thruster_kwargs(spec, pitch_coupling=False)
    if any(float(x) != 0.0 for x in thrusters["nominal_lag_s"]):
        raise ValueError(f"vessel {vessel!r} declares a nominal motor lag; the planar MPPI rollout "
                         "has no motor state (extend the model before using this profile)")
    profile = _shared_module("hydro_profiles").profile_for_vehicle(vessel, swayyaw_damping=swayyaw_damping)
    capsule = spec.collision
    return {
        "vessel": spec.name,
        "model": V2_MODEL_NAME,
        "mass_kg": float(spec.mass_kg),
        "izz_kgm2": float(spec.inertia_com_kg_m2[2][2]),
        **profile_planar_damping(profile),
        "swayyaw_damping": str(profile.sources["options"]["swayyaw_damping"]),
        "forward_limit_n": [float(x) for x in thrusters["forward_limit_n"]],
        "reverse_limit_n": [float(x) for x in thrusters["reverse_limit_n"]],
        "arm_m": float(thrusters["arm_m"]),
        "yaw_command_limit_nm": float(thrusters["yaw_command_limit_nm"]),
        "mount_z_m": float(thrusters["mount_z_m"]),
        "saturation": "radial",
        "water_velocity_mps": [0.0, 0.0],
        "resistance_scale": 1.0,
        "physics_dt_s": V2_PHYSICS_DT_S,
        "decimation": V2_DECIMATION,
        "capsule_center_com_xy_m": [float(x) for x in capsule.capsule_center_com_xy_m],
        "capsule_segment_half_length_m": float(capsule.capsule_segment_half_length_m),
        "capsule_radius_m": float(capsule.capsule_radius_m),
        "top_model_m_s": float(spec.speeds.top_model_m_s),
        "source": ("tasks/_shared/vehicles_v2.py get_vehicle_v2(%r) + to_d3_thruster_kwargs; damping "
                   "tasks/_shared/hydro_profiles.py profile_for_vehicle(swayyaw_damping=%r); "
                   "tasks/_shared/thruster_mixer.py step_thrusters_prepared (radial)"
                   % (spec.name, swayyaw_damping)),
    }


class VesselDynamicsV2:
    """Calm nominal planar model of a D5 registry vessel with the D3 mixer.

    Same state/action/step contract as NominalBlueBoatDynamics (state
    [..., 6] = [x, y, yaw, u, v, r], action [..., 2] in [-1, 1]); step()
    advances one CONTROL step = ``decimation`` physics substeps. Per substep:
    body velocity after one backward-Euler damping step with the quadratic
    coefficient frozen at the current water-relative speed (calm water), the
    world velocity integrated by the engine at the pre-step heading, the
    position advanced with the NEW velocity (semi-implicit), then the body
    velocity re-projected on the new heading. The mixer wrench (surge force
    X, yaw moment N about the COM) is held over the control step (nominal
    motor lag is zero, so every substep would compute the same wrench).
    """

    def __init__(self, vessel: str, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32, swayyaw_damping: str = "journal_default") -> None:
        self.params = vessel_params_v2(vessel, swayyaw_damping)
        self.vessel = self.params["vessel"]
        self.device = torch.device(device)
        self.dtype = dtype
        self.control_dt_s = self.params["physics_dt_s"] * int(self.params["decimation"])
        mixer = _shared_module("thruster_mixer")
        self._mixer = mixer
        self.thruster_spec = mixer.ThrusterSpec(
            forward_limit_n=tuple(self.params["forward_limit_n"]),
            reverse_limit_n=tuple(self.params["reverse_limit_n"]),
            arm_m=self.params["arm_m"], yaw_command_limit_nm=self.params["yaw_command_limit_nm"],
            nominal_lag_s=(0.0, 0.0), mount_z_m=self.params["mount_z_m"])
        self._prepared = {}
        p = self.params
        self.capsule = (float(p["capsule_center_com_xy_m"][0]), float(p["capsule_center_com_xy_m"][1]),
                        float(p["capsule_segment_half_length_m"]), float(p["capsule_radius_m"]))

    # -- mixer ------------------------------------------------------------
    def wrench(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(surge force X [...], yaw moment N [...]) of an action via the D3 mixer."""
        a = action.clamp(-1.0, 1.0)
        key = (tuple(a.shape), str(a.device), a.dtype)
        parameters = self._prepared.get(key)
        if parameters is None:
            parameters = self._mixer.prepare_thruster_parameters(
                self.thruster_spec, self._mixer.ThrusterDegradation(), a)
            self._prepared = {key: parameters}
        state = self._mixer.ThrusterState(torch.zeros_like(a), torch.zeros_like(a))
        out = self._mixer.step_thrusters_prepared(a, state, self.thruster_spec, self.params["physics_dt_s"],
                                                  parameters, "radial")
        return out.body_wrench[..., 0], out.body_wrench[..., 5]

    # -- closed forms (tests) -----------------------------------------------
    def terminal_speed_forward(self) -> float:
        p = self.params
        return terminal_speed(sum(p["forward_limit_n"]), p["surge_lin"], p["surge_quad"])

    def terminal_yaw_rate(self) -> float:
        p = self.params
        torque = p["arm_m"] * 2.0 * min(p["reverse_limit_n"])
        if p["yaw_quad"] == 0.0:
            return torque / p["yaw_lin"]
        return terminal_speed(torque, p["yaw_lin"], p["yaw_quad"])

    # -- dynamics -----------------------------------------------------------
    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """state [..., 6], action [..., 2] -> next state [..., 6]."""
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"state last dim must be {STATE_DIM}")
        if action.shape[-1] != ACTION_DIM:
            raise ValueError(f"action last dim must be {ACTION_DIM}")
        p = self.params
        dt = p["physics_dt_s"]
        m, izz = p["mass_kg"], p["izz_kgm2"]
        force_x, moment_n = self.wrench(action.to(state.dtype))
        x, y, yaw, u, v, r = state.unbind(-1)
        for _ in range(int(p["decimation"])):
            # Backward Euler with the quadratic coefficient frozen at this
            # water-relative state (calm: u_w = 0): (M + dt D) v' = M v + dt f.
            u_new = (m * u + dt * force_x) / (m + dt * (p["surge_lin"] + p["surge_quad"] * u.abs()))
            v_new = (m * v) / (m + dt * (p["sway_lin"] + p["sway_quad"] * v.abs()))
            r_new = (izz * r + dt * moment_n) / (izz + dt * (p["yaw_lin"] + p["yaw_quad"] * r.abs()))
            cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
            vwx = u_new * cos_y - v_new * sin_y
            vwy = u_new * sin_y + v_new * cos_y
            x = x + vwx * dt
            y = y + vwy * dt
            r = r_new
            yaw = yaw + r * dt
            cos_n, sin_n = torch.cos(yaw), torch.sin(yaw)
            u = vwx * cos_n + vwy * sin_n
            v = -vwx * sin_n + vwy * cos_n
        return torch.stack((x, y, yaw, u, v, r), dim=-1)

    def hull_clearance(self, pos: torch.Tensor, yaw: torch.Tensor, centers: torch.Tensor,
                       radii: torch.Tensor) -> torch.Tensor:
        """Capsule-to-disc clearance: pos [N,K,2], yaw [N,K], centers [N,M,2], radii [N,M] -> [N,K,M]."""
        cx, cy, half, radius = self.capsule
        cos_y, sin_y = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        rel = centers.unsqueeze(1) - pos.unsqueeze(2)                   # world, [N,K,M,2]
        bx = rel[..., 0] * cos_y + rel[..., 1] * sin_y - cx              # body frame, spine centre
        by = -rel[..., 0] * sin_y + rel[..., 1] * cos_y - cy
        along = bx.clamp(-half, half)
        distance = torch.sqrt((bx - along) ** 2 + by ** 2)
        return distance - radius - radii.unsqueeze(1)


def vessel_length_scale(vessel: str) -> float:
    """E5 length ratio s = L / L_ref (journal_cfg_v2.vessel_record; BlueBoat 1)."""
    return float(_shared_module("journal_cfg_v2").vessel_record(vessel).length_scale)


def mppi_params_for_vessel(vessel: str, params: "MPPIParams | None" = None) -> "MPPIParams":
    """Declared per-vessel scaling of the BlueBoat-reference MPPI parameters (F7).

    NOT a tuning step. The shipped defaults were calibrated for the BlueBoat
    (module docstring): a 1.0 s horizon lets it change heading inside the
    horizon (yaw time constant I_zz / d_r ~ 0.2 s). The journal task
    contracts scale every length by s = L / L_ref at the SAME cruise speed
    (E5 rule, CONTEXT_R4 D1/D3), so every task time scales by s as well; the
    WAM-V (s = 4.48, yaw time constant ~1.7 s) cannot change heading within
    1 s and a 1 s horizon leaves all samples with the same cost (observed:
    effective sample size ~K, no progress toward an abeam goal). Rule, for
    s > 1 only (the BlueBoat keeps the shipped values bit for bit):
      H          -> round(H * s)            (horizon seconds scale with s)
      clearance_margin_m -> margin * s      (a length)
      w_dock_heading, w_dock_speed -> the module-docstring closed forms
        w_goal * v_tol * dt * (H - 1) / 2 and w_goal * (H - 1) * dt / (2 v_tol)
        at the scaled H, v_tol = 0.3 m/s (speed tolerances scale with the
        cruise-speed ratio, 1 for both vessels).
    Everything else (K, sigma, lambda, w_goal, w_gate, w_obstacle,
    collision_cost, w_effort, w_terminal, w_speed_hold, seed) is unchanged.
    """
    from dataclasses import replace

    base = params or MPPIParams()
    s = vessel_length_scale(vessel)
    if s <= 1.0 + 1e-12:
        return base
    horizon = int(round(base.H * s))
    dt = V2_PHYSICS_DT_S * V2_DECIMATION
    v_tol = DOCK_SPEED_TOLERANCE_MPS
    scaled = replace(base, H=horizon, clearance_margin_m=base.clearance_margin_m * s,
                     w_dock_heading=base.w_goal * v_tol * dt * (horizon - 1) / 2.0,
                     w_dock_speed=base.w_goal * (horizon - 1) * dt / (2.0 * v_tol))
    scaled.validate()
    return scaled


MPPI_FROZEN_FORMAT = "usvbench-mppi-params-v1"
MPPI_FROZEN_FILE = "mppi_params_v2.json"


def model_damping_record(vessel: str) -> dict:
    """Planar damping of the tuning model (journal-default sway/yaw rule), frozen beside H / lambda.

    G4-A5: scripts/mppi_params_v2.json and scripts/classical_gains_v2.json record it under
    ``model_damping`` so a later change of the calm nominal model without a re-freeze
    fails closed (journal_mppi_params, journal_controllers.load_gains).
    """
    params = vessel_params_v2(vessel)
    record = {"swayyaw_damping": params["swayyaw_damping"]}
    record.update({key: params[key] for key in PLANAR_DAMPING_KEYS})
    return record


def load_mppi_frozen(path=None):
    """(data, file sha256) of the frozen per-vessel MPPI parameters (F7 round 4b).

    scripts/mppi_params_v2.json is written by tune_mppi_v2.py from the pre-declared
    MPPI_TUNING_PROCEDURE_V2.md; fail closed when missing or not frozen.
    """
    import hashlib
    import json
    from pathlib import Path

    file = Path(path) if path is not None else Path(__file__).resolve().with_name(MPPI_FROZEN_FILE)
    if not file.is_file():
        raise ValueError(f"frozen MPPI parameters missing: {file} (run tune_mppi_v2.py)")
    raw = file.read_bytes()
    data = json.loads(raw)
    if data.get("format") != MPPI_FROZEN_FORMAT or data.get("status") != "frozen":
        raise ValueError(f"{file} is not a frozen {MPPI_FROZEN_FORMAT} file")
    return data, hashlib.sha256(raw).hexdigest()


def journal_mppi_params(vessel: str, params: "MPPIParams | None" = None, path=None):
    """Journal MPPI parameters of a vessel: declared scaling + the frozen H / lambda.

    Returns (MPPIParams, stamp). H comes from the declared rule
    (mppi_params_for_vessel) and must equal the frozen H (a changed rule without a
    new freeze fails closed); lambda is the value chosen by the pre-declared
    procedure. The stamp names the file digest, the procedure digest, the rule and
    whether the choice met the procedure's acceptance.
    """
    from dataclasses import replace

    scaled = mppi_params_for_vessel(vessel, params)
    data, sha = load_mppi_frozen(path)
    canonical = str(vessel_params_v2(vessel)["vessel"])
    block = (data.get("vessels") or {}).get(canonical)
    if block is None:
        raise ValueError(f"no frozen MPPI parameters for vessel {canonical}")
    if int(block["H"]) != scaled.H:
        raise ValueError(f"frozen MPPI horizon {block['H']} differs from the declared rule {scaled.H} "
                         f"for {canonical}; re-run the procedure")
    tuned_on = (data.get("model_damping") or {}).get(canonical)
    current = model_damping_record(canonical)
    if tuned_on != current:
        raise ValueError(f"frozen MPPI parameters for {canonical} were tuned on model damping {tuned_on}; the "
                         f"current calm nominal model has {current} (G4-A5); re-run the procedure")
    frozen = replace(scaled, lambda_=float(block["lambda"]))
    frozen.validate()
    # G39: the release candidate's declared fix switches (scripts/mppi_g39_fix.py RELEASE_VARIANT);
    # H / lambda stay the frozen per-vessel values.
    g39 = g39_fix_module()
    frozen = g39.params_for_variant(frozen, g39.RELEASE_VARIANT)
    stamp = {"file": "scripts/" + MPPI_FROZEN_FILE, "sha256": sha,
             "procedure": data.get("procedure"), "procedure_sha256": data.get("procedure_sha256"),
             "vessel": canonical, "H": frozen.H, "lambda": frozen.lambda_, "rule": block.get("rule"),
             "accepted": block.get("accepted"), "horizon_rule": block.get("horizon_rule"),
             "model_damping": tuned_on,
             "g39_variant": g39.RELEASE_VARIANT, "g39_format": g39.FIX_FORMAT}
    return frozen, stamp


def g39_fix_module():
    """scripts/mppi_g39_fix.py (sibling module; by path when scripts/ is not on sys.path)."""
    import importlib
    try:
        return importlib.import_module("mppi_g39_fix")
    except ImportError:
        import importlib.util
        import sys
        from pathlib import Path
        path = Path(__file__).resolve().with_name("mppi_g39_fix.py")
        spec = importlib.util.spec_from_file_location("mppi_g39_fix", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["mppi_g39_fix"] = module
        spec.loader.exec_module(module)
        return module


def dynamics_for_cfg(env_cfg, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
    """One model per vessel: VesselDynamicsV2 for journal cfgs, the legacy model otherwise."""
    if int(getattr(env_cfg, "journal_contract_version", 0) or 0) == 2:
        physics = getattr(env_cfg, "underwater_physics_cfg", None)
        if getattr(physics, "hydrodynamics_model", None) != "hydro_v2":
            raise ValueError("journal cfg without hydrodynamics_model='hydro_v2' has no v2 MPPI model")
        dt = float(env_cfg.sim.dt)
        if abs(dt - V2_PHYSICS_DT_S) > 1.0e-12 or int(env_cfg.decimation) != V2_DECIMATION:
            raise ValueError("journal cfg dt/decimation differ from the model (1/120 s x 2, CONTEXT_R4 D2)")
        # G4-A5: the same sway/yaw rule the env resolves (journal_env_mixin.journal_setup ->
        # resolve_journal_vehicle(cfg.vehicle, **cfg.journal_vehicle_options)).
        options = dict(getattr(env_cfg, "journal_vehicle_options", None) or {})
        return VesselDynamicsV2(str(env_cfg.vehicle), device, dtype,
                                swayyaw_damping=options.get("swayyaw_damping", "journal_default"))
    return NominalBlueBoatDynamics(device, dtype)


# ---------------------------------------------------------------------------
# Keyed counter-based sampling noise (C1 finding 52).
#
# The legacy controller drew from ONE torch.Generator shared by all envs, so an
# env's noise depended on the batch size and on how many draws other envs had
# consumed (different episode lengths shift the stream). Each row is now an
# independent SplitMix64 stream keyed by (controller seed, env, episode,
# control step); element j reads outputs 2j+1 and 2j+2 and is mapped to a
# standard normal by the Box-Muller cosine branch. SplitMix64 (Steele, Lea &
# Flood 2014; the seeding generator of xoshiro) is random access by
# construction: output i of the stream keyed k is mix64(k + i * GAMMA).
# ---------------------------------------------------------------------------
NOISE_BACKEND = "splitmix64-boxmuller-f64-v2"
# The pre-D8 stream (one torch.Generator shared by every env, drawn in batch
# order). Kept ONLY to reproduce historical certified MPPI numbers bit for bit;
# it is batch-size and episode-order dependent and never the default.
LEGACY_NOISE_BACKEND = "legacy-shared-generator-v1"
NOISE_BACKENDS = (NOISE_BACKEND, LEGACY_NOISE_BACKEND)
NOISE_KEY_PREFIX = "mppi-noise-v2"
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_TWO_PI = 2.0 * math.pi
_INV_2_53 = 1.0 / float(1 << 53)


def _signed64(value: int) -> int:
    value &= _MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def noise_key(controller_seed: int, env: int, episode: int, control_step: int) -> int:
    """Signed int64 key of one (controller seed, env, episode, step) row."""
    import hashlib
    text = "%s/%d/%d/%d/%d" % (NOISE_KEY_PREFIX, controller_seed, env, episode, control_step)
    return _signed64(int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "little"))


def splitmix64_reference(key: int, index: int) -> int:
    """Pure-Python unsigned output number ``index`` (1-based) of the keyed stream."""
    z = (key + index * _GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
    z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
    return z ^ (z >> 31)


def normal_reference(key: int, element: int) -> float:
    """Pure-Python reference for element ``element`` of a keyed normal row."""
    u1 = ((splitmix64_reference(key, 2 * element + 1) >> 11) + 1) * _INV_2_53
    u2 = (splitmix64_reference(key, 2 * element + 2) >> 11) * _INV_2_53
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(_TWO_PI * u2)


def _srl(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical right shift of int64 bit patterns (torch >> is arithmetic)."""
    return (value >> shift) & ((1 << (64 - shift)) - 1)


def _mix64(z: torch.Tensor) -> torch.Tensor:
    z = (z ^ _srl(z, 30)) * _signed64(_MIX1)
    z = (z ^ _srl(z, 27)) * _signed64(_MIX2)
    return z ^ _srl(z, 31)


def counter_normals(keys: torch.Tensor, count: int) -> torch.Tensor:
    """[N] int64 keys -> [N, count] float64 standard normals on keys.device."""
    if keys.dtype != torch.int64 or keys.dim() != 1:
        raise ValueError("keys must be a 1-D int64 tensor")
    index = torch.arange(1, 2 * count + 1, dtype=torch.int64, device=keys.device)
    z = _mix64(keys[:, None] + index[None, :] * _signed64(_GAMMA))
    bits = _srl(z, 11).to(torch.float64)
    u1 = (bits[:, 0::2] + 1.0) * _INV_2_53
    u2 = bits[:, 1::2] * _INV_2_53
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(_TWO_PI * u2)


def noise_self_check(device) -> str:
    """Compare the device stream with the pure-Python reference, or raise."""
    keys = [noise_key(0, 0, 0, 0), noise_key(123456789, 63, 7, 7199), _signed64(_MASK64), 0]
    count = 5
    got_bits = _mix64(torch.tensor(keys, dtype=torch.int64, device=device)[:, None]
                      + torch.arange(1, 2 * count + 1, dtype=torch.int64, device=device)[None, :]
                      * _signed64(_GAMMA)).cpu().tolist()
    for row, key in zip(got_bits, keys):
        want = [_signed64(splitmix64_reference(key, i)) for i in range(1, 2 * count + 1)]
        if row != want:
            raise RuntimeError("MPPI noise backend self-check failed on %s: int64 stream "
                               "differs from the reference" % (device,))
    got = counter_normals(torch.tensor(keys, dtype=torch.int64, device=device), count).cpu().tolist()
    for row, key in zip(got, keys):
        for j, value in enumerate(row):
            want = normal_reference(key, j)
            if not math.isclose(value, want, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("MPPI noise backend self-check failed on %s: normal "
                                   "transform differs from the reference" % (device,))
    return "passed:" + NOISE_BACKEND


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
    noise_backend: str = NOISE_BACKEND
    # G39 (scripts/mppi_g39_fix.py): berth-leg position dead zone as a fraction of the success
    # position tolerance (0 = off, the v2.6.1 cost), and the geodesic cost-to-go potential of goal /
    # dock modes read from the set_potential buffers (False = straight-line distance, v2.6.1).
    dock_deadzone_fraction: float = 0.0
    geodesic_potential: bool = False

    def validate(self) -> None:
        if self.K < 2 or self.H < 1:
            raise ValueError("K must be >= 2 and H >= 1")
        if not (0.0 <= float(self.dock_deadzone_fraction) < 1.0):
            raise ValueError("dock_deadzone_fraction must lie in [0, 1)")
        if not isinstance(self.geodesic_potential, bool):
            raise ValueError("geodesic_potential must be a bool")
        if self.noise_backend not in NOISE_BACKENDS:
            raise ValueError("noise_backend must be one of " + ", ".join(NOISE_BACKENDS))
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
            "noise_backend": self.noise_backend,
            "dock_deadzone_fraction": self.dock_deadzone_fraction,
            "geodesic_potential": self.geodesic_potential,
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
        env_ids=None,
    ) -> None:
        self.params = params or MPPIParams()
        self.params.validate()
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.device = torch.device(device)
        self.dtype = dtype
        self.dynamics = dynamics or NominalBlueBoatDynamics(self.device, dtype)
        self.env_ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if (len(self.env_ids) != self.num_envs or len(set(self.env_ids)) != self.num_envs
                or any(type(i) is not int or i < 0 for i in self.env_ids)):
            raise ValueError("env_ids must be unique nonnegative integer scenario identities")
        self.episode_ids = [0] * self.num_envs
        self.control_steps = [0] * self.num_envs
        if self.params.noise_backend == LEGACY_NOISE_BACKEND:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(self.params.seed))
            self._noise_self_check = "not applicable (legacy shared generator)"
        else:
            # Refuse to run if this device's integer arithmetic does not
            # reproduce the pure-Python reference stream (wrapping int64
            # multiply, logical shifts).
            self._noise_self_check = noise_self_check(self.device)
        self.U = torch.zeros(
            (self.num_envs, self.params.H, ACTION_DIM),
            device=self.device,
            dtype=dtype,
        )
        # G39 fix F2: geodesic potential buffers (filled by set_potential; rollout reads them by
        # address, so the CUDA-graph replay sees every in-place update). valid False -> straight line.
        self.potential_field = None
        if self.params.geodesic_potential:
            cells = GEODESIC_GRID_CELLS
            self.potential_field = torch.zeros((self.num_envs, cells * cells), device=self.device, dtype=dtype)
            self.potential_origin = torch.zeros((self.num_envs, 2), device=self.device, dtype=dtype)
            self.potential_cell = torch.ones((self.num_envs,), device=self.device, dtype=dtype)
            self.potential_valid = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def set_potential(self, field, origin, cell, valid) -> None:
        """G39 F2: copy one raster per env into the potential buffers (in place).

        field [N, n*n] cost-to-go in metres (row-major, row = y index), origin [N, 2] world xy of
        cell (0, 0), cell [N] cell size (m), valid [N] bool (False: the straight-line potential).
        """
        if self.potential_field is None:
            raise ValueError("set_potential needs MPPIParams.geodesic_potential=True")
        with torch.inference_mode():
            self.potential_field.copy_(field.to(self.potential_field.device, self.potential_field.dtype))
            self.potential_origin.copy_(origin.to(self.potential_origin.device, self.potential_origin.dtype))
            self.potential_cell.copy_(cell.to(self.potential_cell.device, self.potential_cell.dtype))
            self.potential_valid.copy_(valid.to(self.potential_valid.device, torch.bool))

    def _geodesic(self, pos: torch.Tensor, straight: torch.Tensor) -> torch.Tensor:
        """Bilinear cost-to-go at pos [N, K, 2]; outside the raster: edge value + distance to it."""
        n = GEODESIC_GRID_CELLS
        cell = self.potential_cell[:, None]
        gx = (pos[..., 0] - self.potential_origin[:, None, 0]) / cell
        gy = (pos[..., 1] - self.potential_origin[:, None, 1]) / cell
        cx = gx.clamp(0.0, n - 1.0)
        cy = gy.clamp(0.0, n - 1.0)
        x0 = cx.floor().clamp(max=n - 2.0)
        y0 = cy.floor().clamp(max=n - 2.0)
        fx, fy = cx - x0, cy - y0
        i00 = (y0 * n + x0).long()
        field = self.potential_field
        v00 = torch.gather(field, 1, i00)
        v01 = torch.gather(field, 1, i00 + 1)
        v10 = torch.gather(field, 1, i00 + n)
        v11 = torch.gather(field, 1, i00 + n + 1)
        value = (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy)
        value = value + cell * torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
        return torch.where(self.potential_valid[:, None], value, straight)

    def reset(self, done_mask: torch.Tensor, episode_ids=None) -> None:
        """Zero the warm-start sequence of freshly reset envs."""
        mask = done_mask.to(self.device).reshape(-1).bool()
        if mask.shape[0] != self.num_envs:
            raise ValueError("done_mask must have num_envs entries")
        if bool(mask.any()):
            self.U[mask] = 0.0
        if episode_ids is not None and len(episode_ids) != self.num_envs:
            raise ValueError("episode_ids must have num_envs entries")
        for i, done in enumerate(mask.tolist()):
            if done:
                episode = self.episode_ids[i] + 1 if episode_ids is None else episode_ids[i]
                if type(episode) is not int or episode < 0:
                    raise ValueError("episode identities must be nonnegative integers")
                self.episode_ids[i] = episode
                self.control_steps[i] = 0

    def noise_metadata(self):
        if self.params.noise_backend == LEGACY_NOISE_BACKEND:
            return {"backend": LEGACY_NOISE_BACKEND, "keyed": False,
                    "controller_seed": int(self.params.seed), "device": str(self.device),
                    "torch_version": str(torch.__version__),
                    "self_check": self._noise_self_check,
                    "warning": ("one generator shared by all envs: an env's noise depends on "
                                "batch size and episode timing (C1 finding 52); reproduction "
                                "of historical certificates only")}
        return {"backend": NOISE_BACKEND, "keyed": True,
                "controller_seed": int(self.params.seed),
                "key": "sha256('%s/<controller_seed>/<env>/<episode>/<control_step>')"
                       "[:8] little-endian uint64" % NOISE_KEY_PREFIX,
                "key_fields": ["controller_seed", "env", "episode", "control_step"],
                "counter": ("element j (row-major over candidate, horizon_index, "
                            "action_index) uses SplitMix64 outputs 2j+1 and 2j+2 of "
                            "the keyed stream"),
                "element_order": ["candidate", "horizon_index", "action_index"],
                "transform": "Box-Muller cosine branch in float64, then cast to controller dtype",
                "device": str(self.device), "torch_version": str(torch.__version__),
                "self_check": self._noise_self_check,
                "cross_device_note": ("the 64-bit integer stream is exact on every device; "
                                      "float64 log/cos may differ by an ulp between devices")}

    def noise_keys(self) -> list:
        """Signed int64 keys of this control step, one per env row (host side, finding 52).

        noise_key(controller seed, env, episode, control step) in env order. G9: the
        eager sample_noise and the CUDA-graph implementation (CudaGraphMPPI) both take
        their keys from here, so the graph replays exactly the evaluator's keyed stream.
        """
        return [noise_key(int(self.params.seed), env, episode, step)
                for env, episode, step in zip(self.env_ids, self.episode_ids, self.control_steps)]

    def keyed_normals(self, keys: torch.Tensor, K: int, H: int) -> torch.Tensor:
        """[N] int64 device keys -> this step's [N, K, H, 2] standard normals (controller dtype)."""
        return counter_normals(keys, K * H * ACTION_DIM).reshape(
            self.num_envs, K, H, ACTION_DIM).to(self.dtype)

    def sigma_tensor(self, sig, device, dtype) -> torch.Tensor:
        """The per-channel sampling sigma as a device constant (a host-to-device copy).

        G9: a CUDA graph cannot capture this copy; CudaGraphMPPI calls this method once,
        outside the capture, and its capture shadow returns that same tensor.
        """
        return torch.as_tensor(sig, device=device, dtype=dtype)

    def sample_noise(self, K, H):
        """Random-access per-(env, episode, control step) noise (finding 52).

        Batch size, env scheduling and episode order cannot change a stream:
        each row depends only on its own key. The integer stream is computed on
        the controller's device, so it costs no host round trip inside the
        timed controller call.
        """
        if self.params.noise_backend == LEGACY_NOISE_BACKEND:
            return torch.randn((self.num_envs, K, H, ACTION_DIM), generator=self.generator,
                               device=self.device, dtype=self.dtype)
        keys = torch.tensor(self.noise_keys(), dtype=torch.int64, device=self.device)
        return self.keyed_normals(keys, K, H)

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
        # F7: a registry vessel model carries its own capsule footprint
        # (VesselDynamicsV2.hull_clearance); the legacy model keeps the
        # certified circle arithmetic below unchanged.
        capsule = getattr(self.dynamics, "capsule", None) is not None
        if have_obstacles:
            centers = obstacle_centers.to(device=device, dtype=dtype)
            eff_radii = obstacle_radii.to(device=device, dtype=dtype) + (
                0.0 if capsule else float(self.dynamics.params["half_beam_m"])
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

        sigma_t = self.sigma_tensor(sig, device, dtype)
        noise = (
            self.sample_noise(K, H)
            * sigma_t
        )
        self.control_steps = [step + 1 for step in self.control_steps]
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

        # G39 (scripts/mppi_g39_fix.py): F2 geodesic potential (goal / dock, one target) and F1
        # berth-leg dead zone relu(P - f * goal_radius); both off by default (v2.6.1 cost).
        geodesic = (bool(p.geodesic_potential) and self.potential_field is not None
                    and mode in ("goal", "dock") and G == 1)
        deadzone = (float(p.dock_deadzone_fraction) * float(goal_radius)
                    if berth_leg and p.dock_deadzone_fraction > 0.0 else 0.0)

        def shaped(position, distance, index):
            base = self._geodesic(position, distance) if geodesic else distance
            value = base + torch.gather(route_rem, 1, index)
            if deadzone > 0.0:
                shaved = torch.relu(value - deadzone)
                value = torch.where(index >= (G - 1), shaved, value) if mode == "mission" else shaved
            return value

        for t in range(H):
            a = seq[:, :, t, :]
            s = self.dynamics.step(s, a.reshape(N * K, ACTION_DIM))
            pos = s.reshape(N, K, STATE_DIM)[..., :2]
            tgt = torch.gather(targets, 1, idx.unsqueeze(-1).expand(N, K, 2))
            dist = torch.linalg.vector_norm(pos - tgt, dim=-1)
            potential = shaped(pos, dist, idx)

            stage = p.w_goal * potential + p.w_effort * (a * a).sum(dim=-1)
            if have_obstacles:
                if capsule:
                    clearance = self.dynamics.hull_clearance(
                        pos, s.reshape(N, K, STATE_DIM)[..., 2], centers, eff_radii
                    ).amin(dim=-1)
                else:
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
        final_potential = shaped(pos, final_dist, idx)
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


# ---------------------------------------------------------------------------
# G9: CUDA-graph execution of one MPPI control step (implementation "cuda_graph_v1").
#
# The eager MPPIController.rollout launches ~13,000 small kernels per control
# step (G8 probe, RTX 5070, release v2.2: 0.36 s per call, GPU busy 3-6 %).
# CudaGraphMPPI replays THE SAME rollout() from one captured CUDA graph:
#
#   inside the graph  MPPIController.rollout itself, executed under
#       torch.cuda.graph on a shallow copy ("shadow") of the evaluator's
#       controller whose sample_noise reads keyed_normals(static key buffer)
#       and whose sigma_tensor returns the pre-computed constant: input dtype
#       conversions, active index clamp, obstacle radii / masks, route
#       potential, sigma scaling, warm start + noise clipping, the H x
#       decimation rollout of the vessel model, every cost term of the mode
#       (goal, gates, hold, dock, mission), softmax weights, weighted update,
#       the warm-start shift and the info reductions.
#   outside the graph (host, per step, in the eager order)
#       the warm-start rebuild when U does not have params.H steps (rollout's
#       own rule); the noise keys (MPPIController.noise_keys: sha256 on the
#       host) copied into the static int64 key buffer; every tensor input
#       copied into its static buffer; replay; the new warm start copied into
#       controller.U in place; the control-step counters advanced exactly as
#       rollout() advances them; output clones. Once per capture: the sigma
#       constant (MPPIController.sigma_tensor, a host-to-device copy, illegal
#       inside a capture). reset() is MPPIController.reset on the same
#       controller (episode ids, counters, zeroed rows of the static U).
#   signature  shapes, dtypes and presence of the 8 tensor inputs; the scalar
#       arguments (goal_radius, mode, gate window, gate normal speed; a graph
#       bakes them); every MPPIParams value; the dynamics object, vessel,
#       parameters and footprint; the controller's num_envs, device, dtype and
#       warm-start shape. A step whose signature differs from the live graph's
#       RECAPTURES (the old graph and its private memory pool are released
#       first, so at most one graph is alive); a graph is never replayed on
#       another signature and every replay restages all inputs and keys.
#   fail closed  a warmup or capture failure raises GraphCaptureError; there
#       is no eager fallback (eval_mppi exits nonzero, no certificate).
# ---------------------------------------------------------------------------
MPPI_IMPL_EAGER = "eager"
MPPI_IMPL_GRAPH = "cuda_graph_v1"
MPPI_IMPLEMENTATIONS = (MPPI_IMPL_EAGER, MPPI_IMPL_GRAPH)
# eval_mppi --mppi-impl values (eval_mppi keeps a literal copy; a CPU test keeps them equal).
MPPI_IMPL_CHOICES = ("auto", "eager", "graph")
GRAPH_TENSOR_INPUTS = ("state", "targets", "active_index", "obstacle_centers", "obstacle_radii",
                       "obstacle_active", "dock_heading", "gate_normals")
GRAPH_SCALAR_INPUTS = ("goal_radius", "mode", "gate_half_width_m", "gate_min_normal_speed_mps")
GRAPH_INFO_KEYS = ("cost_min", "cost_mean", "ess", "rollout_done_frac", "rollout_min_clearance")
GRAPH_WARMUP_ITERATIONS = 3


class GraphCaptureError(RuntimeError):
    """Graph mode cannot capture this control step.

    Fail closed: nothing falls back to the eager rollout; eval_mppi exits nonzero and
    writes no certificate (scripts/evaluator_exit.py, G6b failure path).
    """


def eager_implementation_stamp() -> dict:
    """controller.configuration.mppi_implementation of an eager run."""
    return {"mode": MPPI_IMPL_EAGER, "captured_graphs": 0, "recaptures": 0}


class CudaGraphBackend:
    """torch.cuda.CUDAGraph capture and replay on one CUDA device.

    warmup: ``iterations`` eager calls on a side stream (cuBLAS handles, allocator
    blocks and the mixer's prepared-parameter cache are created there, outside the
    capture; PyTorch's whole-network capture recipe). capture: torch.cuda.graph with
    capture_error_mode "thread_local", so CUDA calls of other (Kit) threads cannot
    invalidate the capture while a capture-illegal call of this thread raises. Each
    capture owns a private memory pool; release() drops the graph.
    """

    name = "torch.cuda.CUDAGraph"
    capture_error_mode = "thread_local"

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise GraphCaptureError("the CUDA-graph MPPI implementation needs a CUDA device, got %s" % self.device)
        if not torch.cuda.is_available():
            raise GraphCaptureError("the CUDA-graph MPPI implementation needs CUDA (torch.cuda.is_available() "
                                    "is False)")

    def warmup(self, fn, iterations):
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                for _ in range(int(iterations)):
                    fn()
            current.wait_stream(side)
            torch.cuda.synchronize(self.device)

    def capture(self, fn):
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode=self.capture_error_mode):
                outputs = fn()
            torch.cuda.synchronize(self.device)
        return graph, outputs

    def replay(self, handle):
        handle.replay()

    def release(self, handle):
        handle.reset()


def _graph_scalars(goal_radius, mode, gate_half_width_m, gate_min_normal_speed_mps) -> dict:
    """The scalar rollout arguments a graph bakes at capture (Python values only)."""
    values = dict(zip(GRAPH_SCALAR_INPUTS, (goal_radius, mode, gate_half_width_m, gate_min_normal_speed_mps)))
    for name, value in values.items():
        if name == "mode":
            ok = isinstance(value, str)
        else:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not ok:
            raise GraphCaptureError("rollout argument %s must be a Python %s in graph mode (got %s): a "
                                    "graph bakes scalar arguments at capture"
                                    % (name, "str" if name == "mode" else "number", type(value).__name__))
    return values


def _signature_changes(old, new) -> list:
    """Names of the signature parts that differ (the recapture reason)."""
    changes = []
    for old_part, new_part in zip(old[0], new[0]):
        if old_part != new_part:
            changes.append("%s %s -> %s" % (new_part[0], old_part[1:], new_part[1:]))
    for old_part, new_part in zip(old[1], new[1]):
        if old_part != new_part:
            changes.append("%s %r -> %r" % (new_part[0], old_part[2], new_part[2]))
    for old_part, new_part in zip(old[2:], new[2:]):
        if old_part != new_part:
            changes.append(str(new_part[0]))
    return changes or ["signature"]


class _LiveGraph:
    """The one captured graph and everything it reads or writes by address."""

    __slots__ = ("signature", "static", "keys", "sigma", "u_static", "shadow", "handle", "outputs", "pins")


class CudaGraphMPPI:
    """MPPIController.rollout replayed from a CUDA graph; the same actions as the eager rollout.

    CudaGraphMPPI(controller) wraps the evaluator's own MPPIController. rollout() takes the
    eager rollout's keyword arguments (no K/H/sigma/lambda overrides) and returns the same
    (first action [N, 2], info dict); reset() is MPPIController.reset. The controller keeps
    its bookkeeping (env_ids, episode_ids, control_steps) and its U tensor IS the graph's
    static warm-start buffer, so the evaluator sees the eager controller state after every
    step. What runs inside and outside the graph: the section comment above.
    """

    implementation = MPPI_IMPL_GRAPH

    def __init__(self, controller, backend=None, warmup_iterations=GRAPH_WARMUP_ITERATIONS, log=None):
        for name in ("noise_keys", "keyed_normals", "sigma_tensor", "rollout", "reset"):
            if not callable(getattr(controller, name, None)):
                raise GraphCaptureError("controller %s has no %s(); the graph implementation needs the G9 "
                                        "MPPIController" % (type(controller).__name__, name))
        if controller.params.noise_backend != NOISE_BACKEND:
            raise GraphCaptureError("the CUDA-graph implementation needs the keyed noise backend %s (got %s)"
                                    % (NOISE_BACKEND, controller.params.noise_backend))
        if int(warmup_iterations) < 1:
            raise ValueError("warmup_iterations must be >= 1")
        self.controller = controller
        self.backend = CudaGraphBackend(controller.device) if backend is None else backend
        self.warmup_iterations = int(warmup_iterations)
        self._log = log
        self._live = None
        self.captured_graphs = 0
        self.recaptures = 0
        self.captures = []
        self.last_keys = None

    # -- the eager controller's view ------------------------------------
    @property
    def params(self):
        return self.controller.params

    @property
    def num_envs(self):
        return self.controller.num_envs

    @property
    def device(self):
        return self.controller.device

    @property
    def dtype(self):
        return self.controller.dtype

    @property
    def dynamics(self):
        return self.controller.dynamics

    @property
    def U(self):
        return self.controller.U

    def noise_metadata(self):
        return self.controller.noise_metadata()

    def implementation_stamp(self) -> dict:
        """controller.configuration.mppi_implementation of this run."""
        return {"mode": self.implementation, "captured_graphs": int(self.captured_graphs),
                "recaptures": int(self.recaptures)}

    def reset(self, done_mask, episode_ids=None) -> None:
        """MPPIController.reset (zeroes rows of the static warm start in place)."""
        with torch.inference_mode():
            self.controller.reset(done_mask, episode_ids)

    # -- one control step ----------------------------------------------------
    def rollout(self, state, *, targets, active_index=None, goal_radius, mode="goal",
                obstacle_centers=None, obstacle_radii=None, obstacle_active=None, dock_heading=None,
                gate_normals=None, gate_half_width_m=GATE_HALF_WIDTH_M,
                gate_min_normal_speed_mps=GATE_MIN_NORMAL_SPEED_MPS):
        """Graph replay of MPPIController.rollout (same keyword arguments, same results)."""
        tensors = dict(zip(GRAPH_TENSOR_INPUTS, (state, targets, active_index, obstacle_centers, obstacle_radii,
                                                 obstacle_active, dock_heading, gate_normals)))
        scalars = _graph_scalars(goal_radius, mode, gate_half_width_m, gate_min_normal_speed_mps)
        c = self.controller
        with torch.inference_mode():
            H = int(c.params.H)
            if c.U.shape[1] != H:
                # MPPIController.rollout's own warm-start rebuild (U without H steps), on the host
                c.U = torch.zeros((c.num_envs, H, ACTION_DIM), device=c.device, dtype=c.dtype)
            signature = self._signature(tensors, scalars)
            keys = c.noise_keys()
            live = self._live
            if live is None or live.signature != signature:
                live = self._capture(tensors, scalars, signature, keys)
            else:
                self._stage(live, tensors, keys)
            if c.U is not live.u_static:
                # the warm start was rebound outside this wrapper: copy its values in
                live.u_static.copy_(c.U)
                c.U = live.u_static
            self.backend.replay(live.handle)
            outputs = live.outputs
            c.U.copy_(outputs[1])
            c.control_steps = [step + 1 for step in c.control_steps]
            self.last_keys = keys
            info = {name: outputs[2 + index].clone() for index, name in enumerate(GRAPH_INFO_KEYS)}
            return outputs[0].clone(), info

    # -- internals ------------------------------------------------------------
    def _signature(self, tensors, scalars):
        c = self.controller
        parts = []
        for name, value in tensors.items():
            if value is None:
                parts.append((name, None))
            elif isinstance(value, torch.Tensor):
                parts.append((name, tuple(value.shape), str(value.dtype)))
            else:
                raise GraphCaptureError("rollout input %s must be a tensor or None in graph mode (got %s)"
                                        % (name, type(value).__name__))
        dynamics = c.dynamics
        return (tuple(parts),
                tuple((name, type(value).__name__, value) for name, value in scalars.items()),
                ("params", json.dumps(c.params.as_dict(), sort_keys=True)),
                ("dynamics", id(dynamics), type(dynamics).__name__, str(getattr(dynamics, "vessel", "")),
                 json.dumps(getattr(dynamics, "params", None), sort_keys=True, default=repr),
                 repr(getattr(dynamics, "capsule", None))),
                ("controller", int(c.num_envs), str(c.device), str(c.dtype), tuple(c.U.shape)))

    @staticmethod
    def _stage(live, tensors, keys):
        for name, value in tensors.items():
            if value is not None:
                live.static[name].copy_(value)
        live.keys.copy_(torch.tensor(keys, dtype=torch.int64))

    def _fail(self, stage, exc):
        message = ("CUDA-graph MPPI %s failed (%s: %s); graph mode never falls back to the eager rollout"
                   % (stage, type(exc).__name__, exc))
        if self._log is not None:
            self._log("[MPPI-GRAPH] " + message)
        raise GraphCaptureError(message) from exc

    def _capture(self, tensors, scalars, signature, keys):
        c = self.controller
        previous = self._live
        reason = ["first capture"] if previous is None else _signature_changes(previous.signature, signature)
        if previous is not None:
            # at most one graph (and one private pool) alive; the stale graph is never replayed again
            self._live = None
            self.backend.release(previous.handle)
            previous = None
        live = _LiveGraph()
        live.signature = signature
        live.static = {name: (None if value is None else torch.empty_like(value, device=c.device))
                       for name, value in tensors.items()}
        live.keys = torch.zeros(c.num_envs, dtype=torch.int64, device=c.device)
        live.sigma = c.sigma_tensor(c.params.sigma, c.device, c.dtype)
        self._stage(live, tensors, keys)
        live.u_static = c.U
        shadow = copy.copy(c)
        sigma_values = tuple(c.params.sigma)

        def static_noise(K, H):
            return shadow.keyed_normals(live.keys, K, H)

        def static_sigma(sig, device, dtype):
            if tuple(sig) != sigma_values:
                raise GraphCaptureError("sigma %r differs from the captured constant %r" % (tuple(sig), sigma_values))
            return live.sigma

        shadow.sample_noise = static_noise
        shadow.sigma_tensor = static_sigma
        live.shadow = shadow
        arguments = {name: live.static[name] for name in GRAPH_TENSOR_INPUTS[1:]}
        arguments.update(scalars)

        def step():
            shadow.U = live.u_static
            action, info = shadow.rollout(live.static["state"], **arguments)
            new_u = shadow.U
            shadow.U = live.u_static
            return (action, new_u) + tuple(info[name] for name in GRAPH_INFO_KEYS)

        begin = time.perf_counter()
        try:
            self.backend.warmup(step, self.warmup_iterations)
        except ValueError:
            raise          # the eager rollout's own input validation, unchanged
        except Exception as exc:  # noqa: BLE001 - every other warmup failure fails closed
            self._fail("warmup", exc)
        # Every tensor the graph reads by address stays alive with it: the static buffers,
        # the keys, sigma, the warm start (above) and the dynamics state at capture (the
        # mixer caches its prepared parameters for ONE shape on the shared dynamics object).
        live.pins = (dict(vars(c.dynamics)), dict(getattr(c.dynamics, "_prepared", None) or {}))
        try:
            live.handle, live.outputs = self.backend.capture(step)
        except Exception as exc:  # noqa: BLE001 - a failed capture fails closed
            self._fail("capture", exc)
        seconds = time.perf_counter() - begin
        self._live = live
        self.captured_graphs += 1
        if reason != ["first capture"]:
            self.recaptures += 1
        self.captures.append({"index": self.captured_graphs, "reason": list(reason), "seconds": round(seconds, 6)})
        if self._log is not None:
            self._log("[MPPI-GRAPH] captured graph %d (%s) in %.3f s"
                      % (self.captured_graphs, "; ".join(reason), seconds))
        return live

    def describe(self) -> dict:
        """What the graph holds (diagnostics; scripts/diag_mppi_graph.py)."""
        live = self._live
        return {"implementation": self.implementation,
                "backend": getattr(self.backend, "name", type(self.backend).__name__),
                "capture_error_mode": getattr(self.backend, "capture_error_mode", None),
                "warmup_iterations": self.warmup_iterations,
                "stamp": self.implementation_stamp(),
                "captures": [dict(record) for record in self.captures],
                "static_inputs": None if live is None else {
                    name: (None if buffer is None else list(buffer.shape)) for name, buffer in live.static.items()},
                "scalar_inputs": None if live is None else {name: value for name, _, value in live.signature[1]},
                "inside_graph": ["MPPIController.rollout on the shadow controller: keyed_normals(static keys), "
                                 "sigma scaling, warm start + noise clip, H x decimation rollout, costs, softmax "
                                 "weights, weighted update, warm-start shift, info reductions"],
                "outside_graph": ["noise keys (MPPIController.noise_keys, sha256 on the host) -> static int64 "
                                  "buffer before each replay",
                                  "sigma constant (MPPIController.sigma_tensor, host-to-device copy) once per capture",
                                  "input staging copies; new warm start copied into controller.U; control-step "
                                  "counters; output clones; reset (MPPIController.reset)"]}
