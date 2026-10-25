"""Journal-id controller inputs and classical control laws (F7, CONTEXT_R4 D5/D14).

INFORMATION DIET of every journal controller (MPPI, planner + LOS/PID,
observation LOS/PID, staged harbour controller), published beside every
number: the SAME degraded own-state estimate PPO sees, read from
``env.journal_estimate`` (tasks/_shared/journal_env_mixin.py, set once per
control step by the family's _get_observations through the condition
layer's estimate API):

  position   est.position [N, 2], env-local COM xy (truth + perception error)
  velocity   est.velocity [N, 3], body surge, sway (m/s) and yaw rate (rad/s)
  rays       est.rays [N, 36], ranges (m) from the COM, clamped to
             [0, max range]; ray i points at i * 2 pi / 36 from the bow,
             counter-clockwise (hazard_nav_env.py ray fan convention); a range
             at the maximum is "no return"
  heading    the root-link attitude (compass). No journal suite degrades the
             heading (perception degrades position, velocity and rays; D5), so
             it is the one truth channel a controller reads.
  task       the task's own targets as the env publishes them (goal, gate
             chain, hold point, berth pose, harbour phase targets); these are
             task definitions, not state.

Obstacles come ONLY from the ray block (``ray_obstacles``): each returning
ray is a surface point. Controllers never read D7 objects (the reserved
obstacle slot, the bridge, samples), the static obstacle list, the sea
state, current or wind (CONTEXT_R4 D14, R3 decision 8). The heading-
estimate caveat above is the only non-estimate state channel.

Gains of the journal classical laws are FROZEN DATA
(scripts/classical_gains_v2.json), chosen per vessel by the pre-declared
procedure (F7 work/TUNING_PROCEDURE.md) on the calm nominal CPU model
(mppi_controller.VesselDynamicsV2). The journal controllers refuse to run
without a frozen, accepted gain block for their vessel, or when the file's
recorded tuning-model damping (``model_damping``, G4-A5) differs from the
current calm nominal model (mppi_controller.model_damping_record).

Imports torch at module scope: import it only after AppLauncher on the
Isaac boxes (the evaluators import it lazily); CPU tests import it directly.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import torch

SCRIPTS = Path(__file__).resolve().parent
GAINS_PATH = SCRIPTS / "classical_gains_v2.json"
GAINS_FORMAT = "usvbench-classical-gains-v1"
INFORMATION_DIET = ("journal own-state estimate (env.journal_estimate: position, velocity, 36 rays; D5) "
                    "+ root-link heading + task targets; obstacles only from the ray block; never D7 "
                    "objects, the static obstacle list, sea, current or wind")
# Env members a journal controller must NEVER read (tests trap every access).
FORBIDDEN_ENV_MEMBERS = ("obstacle_centers", "obstacle_radii", "obstacle_active", "_d7_bridge",
                         "_journal_d7_samples", "_d7_pool", "journal_last_step", "_conditions",
                         "_hydro_v2", "sea_state", "wave_field", "current_velocity", "wind")
LEGACY_REFERENCE_LENGTH_NOTE = "lengths scale with journal_cfg_v2.vessel_record(v).length_scale (E5 rule)"


def shared_module(name):
    """tasks/_shared/<name> (package, deployed tree, or by path; see mppi_controller)."""
    from mppi_controller import _shared_module
    return _shared_module(name)


def length_scale(vessel):
    """E5 length ratio s = L / L_ref of a registered v2 vessel (BlueBoat 1)."""
    return float(shared_module("journal_cfg_v2").vessel_record(vessel).length_scale)


# --------------------------------------------------------------------- gains --
def _lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_gains(vessel, path=None):
    """Frozen gain block of ``vessel`` + provenance; fail closed if absent or not accepted."""
    path = Path(path or GAINS_PATH)
    if not path.is_file():
        raise RuntimeError(f"journal classical gains file missing: {path} (run scripts/tune_classical_v2.py)")
    data = json.loads(path.read_text(encoding="ascii"))
    if data.get("format") != GAINS_FORMAT:
        raise RuntimeError(f"{path}: not a {GAINS_FORMAT} file")
    block = (data.get("vessels") or {}).get(vessel)
    if not isinstance(block, dict):
        raise RuntimeError(f"no frozen journal gains for vessel {vessel!r} in {path}")
    for name in ("los_heading_pid", "dock_yaw_pd"):
        entry = block.get(name)
        if not isinstance(entry, dict) or entry.get("status") != "frozen":
            raise RuntimeError(f"journal gains {vessel}/{name} are not frozen (status "
                               f"{None if not isinstance(entry, dict) else entry.get('status')!r}); the "
                               "procedure failed for this vessel and the controller refuses to run")
    from mppi_controller import model_damping_record
    tuned_on = (data.get("model_damping") or {}).get(vessel)
    current = model_damping_record(vessel)
    if tuned_on != current:
        raise RuntimeError(f"journal gains {vessel} were tuned on model damping {tuned_on}; the current calm "
                           f"nominal model has {current} (G4-A5); re-run scripts/tune_classical_v2.py")
    return {"vessel": vessel, "los_heading_pid": dict(block["los_heading_pid"]["gains"]),
            "dock_yaw_pd": dict(block["dock_yaw_pd"]["gains"]),
            "rules": {k: block[k]["rule"] for k in ("los_heading_pid", "dock_yaw_pd")},
            "procedure_sha256": data.get("procedure_sha256"), "file": str(path),
            "file_sha256": _lf_sha256(path), "model_damping": tuned_on}


# -------------------------------------------------------------------- inputs --
class JournalInputs(NamedTuple):
    position_world: torch.Tensor    # [N, 2] estimated COM xy, world (est.position + env origin)
    position_local: torch.Tensor    # [N, 2] est.position (env-local)
    yaw: torch.Tensor               # [N] root-link heading
    velocity: torch.Tensor          # [N, 3] est.velocity (surge, sway, yaw rate)
    rays: torch.Tensor              # [N, R] est.rays (m)


def yaw_from_quat(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_journal_inputs(env):
    """The ONE state read of every journal controller (see the module docstring)."""
    est = getattr(env, "journal_estimate", None)
    if est is None:
        raise RuntimeError("env.journal_estimate is None: the journal env has not produced an "
                           "observation yet (reset the env first); journal controllers read only "
                           "the D5 estimate")
    position = est.position
    origins = env.scene.env_origins[:, :2].to(position.device, position.dtype)
    yaw = yaw_from_quat(env.robot.data.root_link_pose_w[:, 3:7]).to(position.dtype)
    return JournalInputs(position + origins, position, yaw, est.velocity.to(position.dtype),
                         est.rays.to(position.dtype))


def ray_max_range(env_cfg):
    """Maximum ray range (m) of a journal cfg: the plan's sensor range (30 m x s)."""
    plan = getattr(env_cfg, "journal_plan", None)
    params = getattr(plan, "params", None) if plan is not None else None
    if isinstance(params, dict) and params.get("ray_sensor_max_range_m"):
        return float(params["ray_sensor_max_range_m"])
    value = getattr(env_cfg, "ray_max_range_m", None)
    if value:
        return float(value)
    raise ValueError("journal cfg declares no ray range (journal_plan.params['ray_sensor_max_range_m'])")


def ray_obstacles(position_world, yaw, rays, max_range_m, *, fill_gaps=True):
    """Ray returns as obstacle discs: (centers [N,R,2], radii [N,R], active [N,R]).

    Ray i leaves the COM at heading yaw + i * 2 pi / R. A return (range below
    the maximum) is the surface point range * direction; with ``fill_gaps``
    the disc radius is range * tan(pi / R), half the arc between neighbouring
    rays, so a wall seen by adjacent rays is closed. No return = inactive.
    """
    count = rays.shape[-1]
    angles = torch.arange(count, device=rays.device, dtype=rays.dtype) * (2.0 * math.pi / count)
    heading = yaw.unsqueeze(-1) + angles
    direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    ranges = rays.clamp(0.0, float(max_range_m))
    centers = position_world.unsqueeze(1) + ranges.unsqueeze(-1) * direction
    active = ranges < float(max_range_m) * (1.0 - 1.0e-6)
    radii = ranges * math.tan(math.pi / count) if fill_gaps else torch.zeros_like(ranges)
    return centers, torch.where(active, radii, torch.zeros_like(radii)), active


# ---------------------------------------------------------------- LOS / PID --
def wrap(angle):
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def bearing_error(position, yaw, target):
    to_target = target - position
    return wrap(torch.atan2(to_target[..., 1], to_target[..., 0]) - yaw)


class HeadingPID:
    """Vectorised heading PID on a wrapped error (derivative 0 after a reset)."""

    def __init__(self, num_envs, device, dtype, dt, kp, ki, kd):
        if dt <= 0.0:
            raise ValueError("control timestep must be positive")
        # Scalars, or per-env [N] tensors (the tuning grid runs every gain pair at once).
        self.dt = float(dt)
        self.kp, self.ki, self.kd = (g if torch.is_tensor(g) else float(g) for g in (kp, ki, kd))
        self.integral = torch.zeros(num_envs, device=device, dtype=dtype)
        self.previous = torch.zeros_like(self.integral)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def __call__(self, error, active=None):
        active = torch.ones_like(self.initialized) if active is None else active
        self.integral.add_(torch.where(active, error * self.dt, torch.zeros_like(error)))
        derivative = torch.where(self.initialized & active, wrap(error - self.previous) / self.dt,
                                 torch.zeros_like(error))
        yaw = (self.kp * error + self.ki * self.integral + self.kd * derivative).clamp(-1.0, 1.0)
        self.previous.copy_(error)
        self.initialized.fill_(True)
        return yaw

    def reset(self, mask):
        mask = torch.as_tensor(mask, device=self.integral.device).reshape(-1).bool()
        self.integral[mask] = 0.0
        self.previous[mask] = 0.0
        self.initialized[mask] = False


def pass_through_thrust(error):
    """Legacy pass-through throttle: full on the LOS, 0.2 creep while turning."""
    return 0.2 + 0.8 * torch.cos(error).clamp(0.0, 1.0)


class LOSController:
    """Journal LOS/PID toward one target per env (pass-through or hold zone).

    ``act(inputs, targets, hold_zone=None)``: targets [N, 2] world. With
    ``hold_zone`` = (zone_radius_m, slow_radius_m) the legacy deadband hold
    law of classical_baseline.LOSPIDController applies (zero thrust and PID
    memory wiped inside the zone; speed profile outside); otherwise the
    pass-through throttle. ``hold_mode`` "continuous" = the legacy continuous
    branch; "continuous_v2" (G24b, the journal station-keeping law since the
    owner decision of 2026-10-04) = that branch with its defects D1 / D2
    removed (_continuous_v2).
    """

    def __init__(self, num_envs, device, dtype, dt, gains):
        g = gains["los_heading_pid"]
        self.pid = HeadingPID(num_envs, device, dtype, dt, g["kp"], g["ki"], g["kd"])

    HOLD_MODES = ("deadband", "continuous", "continuous_v2")
    CONTINUOUS_HOLD_THRUST_CAP = 0.6     # legacy continuous branch (classical_baseline continuous_hold_thrust_cap)

    def act(self, inputs, targets, hold_zone=None, hold_mode="deadband"):
        error = bearing_error(inputs.position_world, inputs.yaw, targets)
        if hold_zone is None:
            yaw = self.pid(error)
            return torch.stack((pass_through_thrust(error), yaw), dim=-1)
        zone, slow = (float(x) for x in hold_zone)
        if slow <= zone:
            raise ValueError("slow radius must exceed the hold zone radius")
        if hold_mode not in self.HOLD_MODES:
            raise ValueError("hold_mode must be one of %s" % (self.HOLD_MODES,))
        if hold_mode == "continuous_v2":
            return self._continuous_v2(error, targets, inputs, zone, slow)
        distance = torch.linalg.vector_norm(targets - inputs.position_world, dim=-1)
        inside = distance <= zone
        profile = ((distance - zone) / (slow - zone)).clamp(0.0, 1.0)
        thrust = torch.where(error.abs() < math.pi / 4.0, profile, torch.full_like(profile, 0.2))
        if hold_mode == "deadband":
            # classical_baseline.LOSPIDController "deadband": zero thrust and
            # PID memory wiped inside the zone.
            yaw = self.pid(error, active=~inside)
            yaw = torch.where(inside, torch.zeros_like(yaw), yaw)
            self.pid.reset(inside)
            thrust = torch.where(inside, torch.zeros_like(thrust), thrust)
        else:
            # "continuous" (textbook DP): keep regulating inside the zone, thrust
            # proportional to the remaining offset (cap 0.6), integral kept with
            # its yaw authority clamped (legacy anti-windup).
            yaw = self.pid(error)
            ki = self.pid.ki
            if not torch.is_tensor(ki) and ki > 0.0:
                self.pid.integral.clamp_(-1.0 / ki, 1.0 / ki)
            thrust = torch.where(inside, (distance / max(zone, 1e-6)).clamp(0.0, 0.6), thrust)
        return torch.stack((thrust, yaw), dim=-1)

    def _continuous_v2(self, error, targets, inputs, zone, slow):
        """G24b journal station-keeping hold law "continuous_v2" (owner decision: continuous mode, not deadband).

        The legacy "continuous" branch above with its two law defects removed (both visible in the law itself;
        G24b work/evidence/sk_continuous_mechanism.txt shows what they do on the journal SK env):
          D1  inside the zone it commanded forward thrust min(d / zone, 0.6) whatever the heading, so with the
              station behind the bow it drove the hull AWAY from the station (calm: the hull circles through the
              zone at about cruise speed). Fix: times the alignment factor clamp(cos e, 0, 1) of the
              pass-through throttle (no forward thrust with the station abeam or astern).
          D2  the restoring thrust collapsed from 0.6 just inside the zone edge to the profile (d - zone) /
              (slow - zone) ~ 0 just outside it, so a steady push (wind, current) balanced OUTSIDE the zone
              (the deadband failure mode: WAM-V wind D2, error 7.1 m > hold radius 6.72 m). Fix: outside the
              zone an aligned hull gets max(profile, 0.6), monotone and continuous at the edge.
        Unchanged: the heading PID on the bearing error (frozen gains, integral kept, legacy anti-windup), the
        0.2 creep while |e| >= pi / 4 outside the zone, the 0.6 cap, the zone / slow radii. No new constant.
        """
        cap = self.CONTINUOUS_HOLD_THRUST_CAP
        yaw = self.pid(error)
        ki = self.pid.ki
        if not torch.is_tensor(ki) and ki > 0.0:
            self.pid.integral.clamp_(-1.0 / ki, 1.0 / ki)
        distance = torch.linalg.vector_norm(targets - inputs.position_world, dim=-1)
        inside = distance <= zone
        profile = ((distance - zone) / (slow - zone)).clamp(0.0, 1.0)
        outside = torch.where(error.abs() < math.pi / 4.0, profile.clamp(min=cap), torch.full_like(profile, 0.2))
        held = (distance / max(zone, 1e-6)).clamp(0.0, cap) * torch.cos(error).clamp(0.0, 1.0)
        thrust = torch.where(inside, held, outside)
        return torch.stack((thrust, yaw), dim=-1)

    def reset(self, done):
        self.pid.reset(done)


class DockingLaw:
    """Three-phase docking (APPROACH / ALIGN+BRAKE / HOLD) on the journal inputs.

    The legacy classical_baseline.DockingController, re-expressed on the D5
    estimate: position from est.position, the yaw-rate damping term from the
    estimated yaw rate est.velocity[:, 2], the planar speed from the estimated
    surge/sway. Radii derive from the berth's own success position tolerance
    (legacy 2.2 / 3.0 / 2.0 m at a 2.5 m tolerance -> 0.88 / 1.2 / 0.8 x tol);
    the brake radius is 6 m x s; the HOLD speed gate 0.2 m/s is scaled by the
    cruise-speed ratio (1 for both journal vessels, CONTEXT_R4 D1).
    """

    APPROACH, ALIGN, HOLD = 0, 1, 2

    def __init__(self, num_envs, device, dtype, dt, gains, *, tolerance_m, scale):
        g = gains["dock_yaw_pd"]
        self.kp, self.k_rate = (x if torch.is_tensor(x) else float(x) for x in (g["kp"], g["k_rate"]))
        self.align_radius = 0.88 * float(tolerance_m)
        self.exit_radius = 1.2 * float(tolerance_m)
        self.stop_radius = 0.8 * float(tolerance_m)
        self.brake_radius = max(6.0 * float(scale), self.exit_radius + 1.0e-3)
        self.phase = torch.full((num_envs,), self.APPROACH, device=device, dtype=torch.long)

    def act(self, inputs, berth_point, berth_heading):
        heading = berth_heading / torch.linalg.vector_norm(berth_heading, dim=-1, keepdim=True).clamp_min(1e-6)
        to_berth = berth_point - inputs.position_world
        distance = torch.linalg.vector_norm(to_berth, dim=-1)
        los_error = bearing_error(inputs.position_world, inputs.yaw, berth_point)
        dock_error = wrap(torch.atan2(heading[..., 1], heading[..., 0]) - inputs.yaw)
        speed = torch.linalg.vector_norm(inputs.velocity[..., :2], dim=-1)
        previous = self.phase.clone()
        self.phase[(previous == self.APPROACH) & (distance <= self.align_radius)] = self.ALIGN
        ready = (distance <= self.align_radius) & (dock_error.abs() <= math.radians(10.0)) & (speed < 0.2)
        self.phase[(previous == self.ALIGN) & ready] = self.HOLD
        self.phase[(previous == self.HOLD) & ~ready] = self.ALIGN
        self.phase[(previous != self.APPROACH) & (distance > self.exit_radius)] = self.APPROACH
        error = torch.where(self.phase == self.APPROACH, los_error, dock_error)
        yaw = (self.kp * error - self.k_rate * inputs.velocity[..., 2]).clamp(-1.0, 1.0)
        deadband = (self.phase == self.HOLD) & (dock_error.abs() < math.radians(8.0))
        yaw = torch.where(deadband, torch.zeros_like(yaw), yaw)
        approach = ((distance - self.stop_radius) / (self.brake_radius - self.stop_radius)).clamp(0.0, 1.0)
        approach = approach * torch.cos(los_error).clamp(0.0, 1.0)
        thrust = torch.where(self.phase == self.APPROACH, approach, torch.zeros_like(approach))
        return torch.stack((thrust, yaw), dim=-1)

    def reset(self, done):
        mask = torch.as_tensor(done, device=self.phase.device).reshape(-1).bool()
        self.phase[mask] = self.APPROACH


class RunupDockingLaw(DockingLaw):
    """G24b: DockingLaw whose APPROACH first steers to the berth-axis run-up point (owner decision 2026-10-04).

    G24 (20261003_g24_sk_dk_avoidance work/evidence/dk_trace_*.txt): the three-phase law steers its APPROACH
    straight at the berth centre at full thrust, so an episode that starts laterally offset reaches the berth
    mouth on a slant and hits a jamb (v2.3 / v2.4 / v2.5 calm docking: 7/32 BlueBoat and 8/32 WAM-V collisions,
    identical per episode). This law keeps every phase, radius, gain and the ALIGN / HOLD actions of DockingLaw;
    only the APPROACH steering target changes: first the run-up point on the berth axis, ``runup_x_m`` along the
    berth heading from the berth centre (journal_layouts_v2._cand_docking's own via: slip opening x minus the
    registry approach_runup_m), until the vessel is within ``accept_m`` of it or past it along the axis
    (x_b >= runup_x_m); from then on (latched per env, cleared by reset) the berth centre, DockingLaw's target.
    APPROACH thrust = DockingLaw's profile on the distance to the berth centre x cos(LOS error to the CURRENT
    target); yaw = DockingLaw's PD (kp e - k_rate r) on that LOS error.
    """

    def __init__(self, num_envs, device, dtype, dt, gains, *, tolerance_m, scale, runup_x_m, accept_m):
        super().__init__(num_envs, device, dtype, dt, gains, tolerance_m=tolerance_m, scale=scale)
        self.runup_x = float(runup_x_m)
        self.runup_accept = float(accept_m)
        if not (self.runup_x < 0.0 and self.runup_accept > 0.0):
            raise ValueError("the run-up point must lie before the berth centre (runup_x_m < 0) and accept_m > 0")
        self.past_runup = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def act(self, inputs, berth_point, berth_heading):
        action = super().act(inputs, berth_point, berth_heading)      # phases + the ALIGN / HOLD actions
        heading = berth_heading / torch.linalg.vector_norm(berth_heading, dim=-1, keepdim=True).clamp_min(1e-6)
        x_b = ((inputs.position_world - berth_point) * heading).sum(-1)
        runup = berth_point + self.runup_x * heading
        near = torch.linalg.vector_norm(runup - inputs.position_world, dim=-1) <= self.runup_accept
        self.past_runup |= near | (x_b >= self.runup_x)
        target = torch.where(self.past_runup.unsqueeze(-1), berth_point, runup)
        distance = torch.linalg.vector_norm(berth_point - inputs.position_world, dim=-1)
        los_error = bearing_error(inputs.position_world, inputs.yaw, target)
        yaw = (self.kp * los_error - self.k_rate * inputs.velocity[..., 2]).clamp(-1.0, 1.0)
        thrust = ((distance - self.stop_radius) / (self.brake_radius - self.stop_radius)).clamp(0.0, 1.0)
        thrust = thrust * torch.cos(los_error).clamp(0.0, 1.0)
        approach = (self.phase == self.APPROACH).unsqueeze(-1)
        return torch.where(approach, torch.stack((thrust, yaw), dim=-1).to(action.dtype), action)

    def reset(self, done):
        super().reset(done)
        mask = torch.as_tensor(done, device=self.past_runup.device).reshape(-1).bool()
        self.past_runup[mask] = False


class WaypointTracker:
    """Per-env waypoint list follower (acceptance radius) feeding a LOSController."""

    def __init__(self, num_envs, accept_radius_m):
        self.accept = float(accept_radius_m)
        self.plans = [None] * int(num_envs)
        self.index = [0] * int(num_envs)

    def set_plan(self, env_index, waypoints):
        self.plans[env_index] = torch.as_tensor(waypoints, dtype=torch.float64).reshape(-1, 2)
        self.index[env_index] = 0

    def targets(self, position_world):
        rows = []
        pos = position_world.detach().to("cpu", torch.float64)
        for i, plan in enumerate(self.plans):
            if plan is None:
                raise RuntimeError(f"env {i} has no plan")
            k = self.index[i]
            while k < len(plan) - 1 and float(torch.linalg.vector_norm(plan[k] - pos[i])) < self.accept:
                k += 1
            self.index[i] = k
            rows.append(plan[k])
        return torch.stack(rows).to(position_world.device, position_world.dtype)

    def reset(self, done):
        for i, flag in enumerate(torch.as_tensor(done).reshape(-1).tolist()):
            if flag:
                self.plans[i] = None
                self.index[i] = 0


class JournalBaselineController:
    """classical_baseline.py on journal ids: LOS/PID or the docking law on the estimate.

    Modes (classical_baseline._task_mode): ``path_follow`` (pass-through LOS to
    the env's current waypoint), ``station_keep`` (LOS with the hold zone),
    ``docking`` (DockingLaw on the berth pose), ``point_nav`` (LOS with the
    goal zone). Targets are the env's task definitions in world frame
    (_current_waypoint_world() / waypoints + origin, hold_point, dock_point +
    dock_heading, target_pos); the own state is read_journal_inputs(env).
    ``act(obs)`` ignores the policy observation (journal obs columns are not a
    controller interface).
    """

    MODES = ("path_follow", "station_keep", "docking", "point_nav")

    def __init__(self, base, mode, gains, *, dt, zone_radius_m, slow_radius_m, tolerance_m, scale,
                 hold_mode="deadband", dtype=torch.float32, runup=None):
        if mode not in self.MODES:
            raise ValueError("mode must be one of %s" % (self.MODES,))
        self.base, self.mode, self.hold_mode = base, mode, hold_mode
        n, device = int(base.num_envs), base.device
        self.zone, self.slow = float(zone_radius_m), float(slow_radius_m)
        self.los = LOSController(n, device, dtype, dt, gains)
        # G24b: runup = {"runup_x_m", "accept_m"} selects RunupDockingLaw (journal docking); None keeps DockingLaw.
        if runup is None:
            self.dock = DockingLaw(n, device, dtype, dt, gains, tolerance_m=tolerance_m, scale=scale)
        else:
            self.dock = RunupDockingLaw(n, device, dtype, dt, gains, tolerance_m=tolerance_m, scale=scale,
                                        runup_x_m=runup["runup_x_m"], accept_m=runup["accept_m"])

    def _waypoint(self):
        helper = getattr(self.base, "_current_waypoint_world", None)
        if callable(helper):
            return helper()
        waypoints = self.base.waypoints
        index = self.base.gates_passed.clamp(max=waypoints.shape[1] - 1).long()
        local = waypoints[torch.arange(waypoints.shape[0], device=waypoints.device), index]
        return local + self.base.scene.env_origins[:, :2]

    def act(self, obs=None):
        del obs
        inputs = read_journal_inputs(self.base)
        dtype = inputs.position_world.dtype
        if self.mode == "docking":
            return self.dock.act(inputs, self.base.dock_point.to(dtype), self.base.dock_heading.to(dtype))
        if self.mode == "path_follow":
            action = self.los.act(inputs, self._waypoint().to(dtype))
            passed = getattr(self.base, "gates_passed", None)
            waypoints = getattr(self.base, "waypoints", None)
            if passed is not None and waypoints is not None:
                parked = (passed >= waypoints.shape[1]).unsqueeze(-1)
                action = torch.where(parked, torch.zeros_like(action), action)
            return action
        target = self.base.hold_point if self.mode == "station_keep" else self.base.target_pos
        return self.los.act(inputs, target.to(dtype), hold_zone=(self.zone, self.slow), hold_mode=self.hold_mode)

    def reset(self, done):
        self.los.reset(done)
        self.dock.reset(done)


def controller_stamp(env_cfg, gains, extra=None):
    """Certificate block of a journal classical controller (gains + diet)."""
    block = {"journal": True, "vessel": str(env_cfg.vehicle), "information": INFORMATION_DIET,
             "gains": gains, "length_scale": length_scale(str(env_cfg.vehicle)),
             "ray_max_range_m": ray_max_range(env_cfg)}
    block.update(extra or {})
    return block


__all__ = ["GAINS_PATH", "INFORMATION_DIET", "FORBIDDEN_ENV_MEMBERS", "JournalInputs", "read_journal_inputs",
           "ray_max_range", "ray_obstacles", "load_gains", "length_scale", "HeadingPID", "LOSController",
           "DockingLaw", "RunupDockingLaw", "WaypointTracker", "JournalBaselineController", "bearing_error", "wrap",
           "pass_through_thrust",
           "controller_stamp", "yaw_from_quat"]
