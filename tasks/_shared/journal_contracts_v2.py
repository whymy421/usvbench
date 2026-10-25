"""CPU-only journal episode contracts (task contract version 2).

Scalar reference oracles; a production adapter keeps one oracle (or an
equivalent tensor implementation that reproduces the same traces) per env.

Lifecycle per env:
  reset(com_xy)                      after installing an admitted layout;
  observe_physics(state, ...)        after EVERY physics substep (1/120 s);
  step(time_s, state, ...)           once per control step (1/60 s), before
                                     rewards; returns a Decision;
  last_completed                     immutable EpisodeResult; survives reset
                                     and must be exported before autoreset.
All positions are the hull COM in the env-local frame. No legacy task module
is changed by importing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


class TerminationReason(IntEnum):
    """C5 codes 0-5 unchanged (C5 repo tasks/_shared/termination_reason.py:9-14);
    D4 appends the validity reasons 6-9."""

    RUNNING = 0
    SUCCESS = 1
    COLLISION = 2
    TIMEOUT = 3
    OUT_OF_BOUNDS = 4
    FAILURE = 5
    CAPSIZE = 6
    SINKING = 7
    AIRBORNE = 8
    NONFINITE = 9


TERMINATION_REASON_NAMES = MappingProxyType({
    0: "running", 1: "success", 2: "collision", 3: "timeout", 4: "out_of_bounds", 5: "failure",
    6: "capsize", 7: "sinking", 8: "airborne", 9: "non_finite_state"})
VALIDITY_REASONS = (TerminationReason.CAPSIZE, TerminationReason.SINKING,
                    TerminationReason.AIRBORNE, TerminationReason.NONFINITE)


@dataclass(frozen=True)
class TerminationRow:
    """One row of the exported termination table (see TERMINATION_TABLE)."""

    code: int
    name: str
    origin: str            # "C5" (termination_reason.py:9-23) or "D4" (appended validity codes)
    category: str          # running | validity | bounds | contact | success | failure | timeout
    precedence: int        # 1 = wins over every simultaneous cause; 0 = not a stop cause
    ends_as: str           # "none" | "terminated" | "truncated" (gymnasium done flag)
    outcome: str | None    # certificate outcome label when the code alone decides it
    outcome_rule: str


# ONE exported table for evaluators / certificates (CONTEXT_R3 decision 17;
# E1 certificate ``outcome`` mapping imports it). Codes 0-5 are C5's, 6-9 are
# D4's, unchanged. Precedence (TASK_CONTRACTS_v2 section 2.8, implemented by
# validity_reason and termination_decision below): validity > out_of_bounds >
# collision > success > failure > timeout; inside validity the scalar order is
# non_finite_state > capsize > sinking > airborne. Validity reasons are
# additionally LATCHED first-in-time at physics substeps (ValidityLatch): the
# precedence ranks only causes observed at the same sample.
TERMINATION_TABLE = (
    TerminationRow(0, "running", "C5", "running", 0, "none", None,
                   "not a stop cause (active row)"),
    TerminationRow(9, "non_finite_state", "D4", "validity", 1, "terminated", "non_finite_state",
                   "model-validity exclusion; stays in the denominator"),
    TerminationRow(6, "capsize", "D4", "validity", 2, "terminated", "capsized",
                   "tilt above the declared limit (60 deg); stays in the denominator"),
    TerminationRow(7, "sinking", "D4", "validity", 3, "terminated", "sinking",
                   "heave below -(freeboard + hull depth); stays in the denominator"),
    TerminationRow(8, "airborne", "D4", "validity", 4, "terminated", "airborne",
                   "heave above draft + hull depth; stays in the denominator"),
    TerminationRow(4, "out_of_bounds", "C5", "bounds", 5, "terminated", "out_of_bounds",
                   "COM left the layout bounds"),
    TerminationRow(2, "collision", "C5", "contact", 6, "terminated", None,
                   "contact (swept analytic clearance <= 0 or engine contact) ends the episode; the "
                   "certificate label depends on the task latches (evaluator-owned mapping)"),
    TerminationRow(1, "success", "C5", "success", 7, "terminated", "success",
                   "first valid clean success of a task that stops on success"),
    TerminationRow(5, "failure", "C5", "failure", 8, "terminated", None,
                   "task-declared terminal failure; label depends on task latches"),
    TerminationRow(3, "timeout", "C5", "timeout", 9, "truncated", None,
                   "deadline reached; episode_success=true for a fixed-horizon task that succeeded "
                   "(station keeping), otherwise the deadline failure"),
)
TERMINATION_PRECEDENCE = tuple(row.code for row in sorted(TERMINATION_TABLE, key=lambda r: r.precedence)
                               if row.precedence > 0)


@dataclass(frozen=True)
class ValidityLimits:
    """Declared model-validity exclusions (not calibrated capsize physics).

    tilt = angle between body up and world up; heave = COM z - local water
    surface z - calm equilibrium COM height above the surface. Defaults are
    the BlueBoat values; use journal_cfg_v2 JournalTaskPlan.validity_limits.
    """

    capsize_angle_rad: float = math.pi / 3.0
    min_heave_m: float = -0.529
    max_heave_m: float = 0.599

    def __post_init__(self):
        values = (self.capsize_angle_rad, self.min_heave_m, self.max_heave_m)
        if any(isinstance(v, bool) for v in values) or not all(math.isfinite(v) for v in values):
            raise ValueError("validity limits must be finite numbers")
        if not 0.0 < self.capsize_angle_rad < math.pi:
            raise ValueError("capsize angle must lie in (0, pi)")
        if not self.min_heave_m < 0.0 < self.max_heave_m:
            raise ValueError("heave band must contain the equilibrium")


@dataclass(frozen=True)
class RigidState:
    com_position_xyz: Sequence[float]
    quaternion_wxyz: Sequence[float]
    linear_velocity_xyz: Sequence[float]
    angular_velocity_xyz: Sequence[float]
    water_level_m: float = 0.0
    nominal_com_height_m: float = 0.0


def _field(state, name):
    return state[name] if isinstance(state, Mapping) else getattr(state, name)


def _vector(values, size):
    result = tuple(float(v) for v in values)
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError("malformed or non-finite vector")
    return result


def _state_values(state):
    position = _vector(_field(state, "com_position_xyz"), 3)
    quaternion = _vector(_field(state, "quaternion_wxyz"), 4)
    linear = _vector(_field(state, "linear_velocity_xyz"), 3)
    angular = _vector(_field(state, "angular_velocity_xyz"), 3)
    water = float(_field(state, "water_level_m"))
    nominal = float(_field(state, "nominal_com_height_m"))
    norm = math.sqrt(sum(q * q for q in quaternion))
    if not all(math.isfinite(v) for v in (water, nominal, norm)) or norm <= 1.0e-12:
        raise ValueError("invalid quaternion or water reference")
    return position, tuple(q / norm for q in quaternion), linear, angular, water, nominal


def _safe_state(state):
    try:
        return _state_values(state)
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, IndexError):
        return None


def validity_reason(state, limits: ValidityLimits = ValidityLimits()):
    """Priority: non-finite/malformed, capsize, sinking, airborne."""
    values = _safe_state(state)
    if values is None:
        return TerminationReason.NONFINITE
    position, quat, _, _, water, nominal = values
    _, x, y, _ = quat
    tilt = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))
    if tilt > limits.capsize_angle_rad:
        return TerminationReason.CAPSIZE
    heave = position[2] - water - nominal
    if heave < limits.min_heave_m:
        return TerminationReason.SINKING
    if heave > limits.max_heave_m:
        return TerminationReason.AIRBORNE
    return TerminationReason.RUNNING


def validity_reasons_batch(com_xyz, quat_wxyz, lin_vel, ang_vel, water_level, nominal_height,
                           capsize_angle_rad, min_heave_m, max_heave_m):
    """Vectorised twin of validity_reason (numpy; same ops port to torch).

    Arrays: (N,3), (N,4), (N,3), (N,3), (N,), (N,); limits scalars or (N,).
    Returns int64 codes (0, 6, 7, 8, 9) with the scalar priority order.
    """
    com = np.asarray(com_xyz, dtype=np.float64)
    q = np.asarray(quat_wxyz, dtype=np.float64)
    lin = np.asarray(lin_vel, dtype=np.float64)
    ang = np.asarray(ang_vel, dtype=np.float64)
    water = np.asarray(water_level, dtype=np.float64)
    nominal = np.asarray(nominal_height, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        norm = np.sqrt(np.sum(q * q, axis=1))
        finite = (np.all(np.isfinite(com), axis=1) & np.all(np.isfinite(q), axis=1)
                  & np.all(np.isfinite(lin), axis=1) & np.all(np.isfinite(ang), axis=1)
                  & np.isfinite(water) & np.isfinite(nominal) & np.isfinite(norm) & (norm > 1.0e-12))
        safe_norm = np.where(finite, norm, 1.0)
        x = np.where(finite, q[:, 1] / safe_norm, 0.0)
        y = np.where(finite, q[:, 2] / safe_norm, 0.0)
        tilt = np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        heave = np.where(finite, com[:, 2] - water - nominal, 0.0)
    out = np.zeros(len(com), dtype=np.int64)
    out = np.where(heave > max_heave_m, int(TerminationReason.AIRBORNE), out)
    out = np.where(heave < min_heave_m, int(TerminationReason.SINKING), out)
    out = np.where(tilt > capsize_angle_rad, int(TerminationReason.CAPSIZE), out)
    out = np.where(~finite, int(TerminationReason.NONFINITE), out)
    return out


@dataclass
class ValidityLatch:
    """Latches the FIRST fatal validity event; a later non-finite state is
    additionally flagged (nonfinite_seen) without rewriting the cause."""

    limits: ValidityLimits = field(default_factory=ValidityLimits)
    reason: TerminationReason = TerminationReason.RUNNING
    nonfinite_seen: bool = False

    def update(self, state):
        observed = validity_reason(state, self.limits)
        if observed == TerminationReason.NONFINITE:
            self.nonfinite_seen = True
        if self.reason == TerminationReason.RUNNING:
            self.reason = observed
        return self.reason

    def reset(self):
        self.reason = TerminationReason.RUNNING
        self.nonfinite_seen = False


@dataclass(frozen=True)
class Decision:
    terminated: bool
    truncated: bool
    reason: TerminationReason

    @property
    def done(self):
        return self.terminated or self.truncated


def termination_decision(*, validity=TerminationReason.RUNNING, collision=False, out_of_bounds=False,
                         success=False, failure=False, timeout=False):
    """validity > out_of_bounds > collision > success > failure > timeout.

    The ordinary order is C5's (termination_reason.py:31-45: terminal causes
    beat a simultaneous time limit; out_of_bounds, collision, success,
    failure). ``success`` means success is an enabled stopping rule; a fixed
    horizon task passes False and reports TIMEOUT with episode_success true.
    """
    validity = TerminationReason(validity)
    if validity not in (TerminationReason.RUNNING,) + VALIDITY_REASONS:
        raise ValueError("validity must be RUNNING or a validity reason")
    for enabled, reason in ((validity != TerminationReason.RUNNING, validity),
                            (out_of_bounds, TerminationReason.OUT_OF_BOUNDS),
                            (collision, TerminationReason.COLLISION),
                            (success, TerminationReason.SUCCESS),
                            (failure, TerminationReason.FAILURE)):
        if enabled:
            return Decision(True, False, reason)
    return Decision(False, bool(timeout), TerminationReason.TIMEOUT if timeout else TerminationReason.RUNNING)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _yaw(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def segment_hits_disk(p0, p1, center, radius):
    """True if the straight COM motion p0->p1 enters the closed disk."""
    ax, ay = p0[0] - center[0], p0[1] - center[1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = dx * dx + dy * dy
    t = 0.0 if n == 0.0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / n))
    return math.hypot(ax + t * dx, ay + t * dy) <= radius


@dataclass(frozen=True)
class EpisodeResult:
    episode_success: bool
    time_to_success: float
    success_prefix_path_length_m: float
    full_episode_path_length_m: float
    termination_reason: int
    metrics: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def termination_reason_name(self):
        return TERMINATION_REASON_NAMES[self.termination_reason]

    @property
    def episode_path_length(self):
        """Frozen-evaluator compatibility alias = FULL-episode COM travel.
        For tasks that terminate on success it equals the success prefix."""
        return self.full_episode_path_length_m

    def evaluator_fields(self):
        return dict(episode_success=self.episode_success, time_to_success=self.time_to_success,
                    episode_path_length=self.episode_path_length,
                    success_prefix_path_length_m=self.success_prefix_path_length_m,
                    full_episode_path_length_m=self.full_episode_path_length_m,
                    termination_reason=self.termination_reason,
                    termination_reason_name=self.termination_reason_name, **self.metrics)


@dataclass
class EpisodeMetrics:
    previous_xy: tuple
    previous_time_s: float = 0.0
    full_episode_path_length_m: float = 0.0
    success_prefix_path_length_m: float = math.nan
    time_to_success: float = math.nan
    path_cross_track_error_integral_m_s: float = 0.0
    previous_cross_track_error_m: float | None = None
    valid_path_prefix_only: bool = False
    min_clearance_m: float = math.inf

    def observe(self, time_s, com_xy, cross_track_error_m=None):
        if not math.isfinite(time_s) or time_s <= self.previous_time_s:
            raise ValueError("sample clock must be finite and strictly increasing")
        point = _vector(com_xy, 2)
        self.full_episode_path_length_m += _distance(point, self.previous_xy)
        if cross_track_error_m is not None:
            error = float(cross_track_error_m)
            if not math.isfinite(error) or error < 0.0:
                raise ValueError("cross-track error must be finite and nonnegative")
            prior = error if self.previous_cross_track_error_m is None else self.previous_cross_track_error_m
            self.path_cross_track_error_integral_m_s += 0.5 * (prior + error) * (time_s - self.previous_time_s)
            self.previous_cross_track_error_m = error
        self.previous_xy, self.previous_time_s = point, float(time_s)

    def latch_success(self, time_s):
        if math.isnan(self.time_to_success):
            self.time_to_success = float(time_s)
            self.success_prefix_path_length_m = self.full_episode_path_length_m


class EpisodeBase:
    """Common validity/contact/metric/termination machinery.

    Collision (analytic swept contact OR engine contact, ORed by the adapter)
    and validity failures terminate every task. A completed EpisodeResult
    replaces last_completed atomically at the terminal step.
    """

    stop_on_success = True

    def __init__(self, deadline_s, validity_limits=ValidityLimits()):
        if isinstance(deadline_s, bool) or not math.isfinite(deadline_s) or deadline_s <= 0.0:
            raise ValueError("deadline must be positive and finite")
        self.deadline_s = float(deadline_s)
        self.validity = ValidityLatch(validity_limits)
        self.last_completed = None
        self.reset((0.0, 0.0))

    def reset(self, com_xy):
        self.metrics = EpisodeMetrics(_vector(com_xy, 2))
        self.validity.reset()
        self.done = False
        self.success = False
        self.clean = True
        self.contact_since_step = False
        self.thrust_sq_integral_n2s = None

    def observe_physics(self, state, *, collision=False, clearance_m=None, dt_s=None, motor_thrust_n=None):
        """Latch substep validity, contact (clearance <= 0 counts as contact),
        minimum clearance and optional per-motor squared-thrust integral."""
        if self.done:
            return
        self.validity.update(state)
        hit = bool(collision)
        if clearance_m is not None:
            c = float(clearance_m)
            if not math.isfinite(c):
                hit = True          # non-finite clearance fails closed
            else:
                self.metrics.min_clearance_m = min(self.metrics.min_clearance_m, c)
                hit = hit or c <= 0.0
        self.contact_since_step |= hit
        if motor_thrust_n is not None and dt_s is not None:
            f = tuple(float(v) for v in motor_thrust_n)
            if all(math.isfinite(v) for v in f) and math.isfinite(dt_s) and dt_s > 0.0:
                if self.thrust_sq_integral_n2s is None:
                    self.thrust_sq_integral_n2s = [0.0] * len(f)
                for i, v in enumerate(f):
                    self.thrust_sq_integral_n2s[i] += v * v * float(dt_s)

    def _observe(self, time_s, state, collision, cross_track_error_m=None, clearance_m=None):
        if self.done:
            raise RuntimeError("reset required after completion")
        if isinstance(time_s, bool) or not math.isfinite(time_s) or time_s <= self.metrics.previous_time_s:
            raise ValueError("sample clock must increase from the reset time zero")
        if time_s > self.deadline_s + 1.0e-9:
            raise ValueError("the adapter must include a sample exactly at the deadline")
        self.observe_physics(state, collision=collision, clearance_m=clearance_m)
        contact = self.contact_since_step
        self.contact_since_step = False
        self.clean &= not contact
        values = _safe_state(state)
        if values is None:
            self.metrics.valid_path_prefix_only = True
            return None, contact
        previous = self.metrics.previous_xy
        self.metrics.observe(time_s, values[0][:2], cross_track_error_m)
        return (values, previous), contact

    def _finish(self, time_s, *, candidate=False, collision=False, out_of_bounds=False, extras=None):
        safe = self.validity.reason == TerminationReason.RUNNING and self.clean and not out_of_bounds
        if candidate and safe and not self.success:
            self.success = True
            self.metrics.latch_success(time_s)
        decision = termination_decision(validity=self.validity.reason, collision=collision,
                                        out_of_bounds=out_of_bounds,
                                        success=self.success and self.stop_on_success,
                                        timeout=time_s >= self.deadline_s - 1.0e-9)
        if decision.done:
            self.done = True
            if decision.reason not in (TerminationReason.SUCCESS, TerminationReason.TIMEOUT):
                self.success = False
            details = dict(extras or {})
            details.update(path_cross_track_error_integral_m_s=self.metrics.path_cross_track_error_integral_m_s,
                           path_length_valid_prefix_only=self.metrics.valid_path_prefix_only,
                           min_clearance_m=self.metrics.min_clearance_m,
                           nonfinite_seen=self.validity.nonfinite_seen,
                           deadline_s=self.deadline_s,
                           integrated_squared_thrust_n2s=(None if self.thrust_sq_integral_n2s is None
                                                          else tuple(self.thrust_sq_integral_n2s)))
            self.last_completed = EpisodeResult(
                self.success, self.metrics.time_to_success if self.success else math.nan,
                self.metrics.success_prefix_path_length_m if self.success else math.nan,
                self.metrics.full_episode_path_length_m, int(decision.reason), details)
        return decision


class ReachEpisode(EpisodeBase):
    """One immutable goal disk; success on the first clean valid entry of the
    COM motion (swept between control samples), then terminate. Used by
    reach and every goal-disk obstacle task (scatter, ring, fortress,
    crossing, iceberg, moving basin)."""

    def __init__(self, goal_xy, goal_radius_m, deadline_s, validity_limits=ValidityLimits()):
        self.goal_xy = _vector(goal_xy, 2)
        if isinstance(goal_radius_m, bool) or not math.isfinite(goal_radius_m) or goal_radius_m <= 0.0:
            raise ValueError("goal radius must be positive and finite")
        self.goal_radius_m = float(goal_radius_m)
        super().__init__(deadline_s, validity_limits)

    def step(self, time_s, state, *, collision=False, out_of_bounds=False, clearance_m=None):
        obs, contact = self._observe(time_s, state, collision, clearance_m=clearance_m)
        candidate = obs is not None and segment_hits_disk(obs[1], obs[0][0], self.goal_xy, self.goal_radius_m)
        return self._finish(time_s, candidate=candidate, collision=contact, out_of_bounds=out_of_bounds,
                            extras=dict(goal_distance_m=None if obs is None else _distance(obs[0][0], self.goal_xy)))


@dataclass(frozen=True)
class Berth:
    """Slip interior (INNER wall faces) in the berth frame: x along the dock
    heading from opening_x_m to back_x_m, |y| <= half_width_m."""

    center_xy: tuple
    heading_rad: float
    opening_x_m: float
    back_x_m: float
    half_width_m: float

    def __post_init__(self):
        _vector(self.center_xy, 2)
        vals = (self.heading_rad, self.opening_x_m, self.back_x_m, self.half_width_m)
        if not all(math.isfinite(v) for v in vals) or self.opening_x_m >= self.back_x_m or self.half_width_m <= 0:
            raise ValueError("invalid berth interior")

    @classmethod
    def from_layout(cls, berth):
        return cls(tuple(berth["center_xy"]), berth["heading_rad"], berth["opening_x_m"],
                   berth["back_x_m"], berth["half_width_m"])

    def local(self, point):
        dx, dy = point[0] - self.center_xy[0], point[1] - self.center_xy[1]
        c, s = math.cos(self.heading_rad), math.sin(self.heading_rad)
        return c * dx + s * dy, -s * dx + c * dy

    def contains_polygon(self, point, yaw_rad, hull_vertices_m):
        """Every COM-relative hull vertex inside the interior (asymmetric COM ok)."""
        vertices = tuple(_vector(v, 2) for v in hull_vertices_m)
        if len(vertices) < 3:
            raise ValueError("hull polygon requires at least three vertices")
        x, y = self.local(point)
        rel = yaw_rad - self.heading_rad
        c, s = math.cos(rel), math.sin(rel)
        return all(self.opening_x_m <= x + c * vx - s * vy <= self.back_x_m
                   and abs(y + s * vx + c * vy) <= self.half_width_m for vx, vy in vertices)


def _hull(hull_vertices_m):
    vertices = tuple(_vector(v, 2) for v in hull_vertices_m)
    if len(vertices) < 3:
        raise ValueError("hull polygon requires at least three vertices")
    area2 = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(vertices, vertices[1:] + vertices[:1])))
    if area2 <= 1.0e-12:
        raise ValueError("hull polygon must have nonzero area")
    return vertices


class DockingEpisode(EpisodeBase):
    """Directional entry through the mouth, then a continuous clean hold of
    hold_s with the WHOLE hull inside the interior and COM position, heading
    and planar-speed tolerances (C1 findings 27 and 60)."""

    def __init__(self, berth, hull_vertices_m, deadline_s, *, hold_s=5.0, position_tolerance_m=2.5,
                 heading_tolerance_rad=math.radians(15.0), speed_tolerance_mps=0.3,
                 validity_limits=ValidityLimits()):
        self.berth = berth
        self.hull_vertices_m = _hull(hull_vertices_m)
        vals = (hold_s, position_tolerance_m, heading_tolerance_rad, speed_tolerance_mps)
        if any(isinstance(v, bool) for v in vals) or not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError("docking tolerances must be positive and finite")
        self.hold_s, self.position_tolerance_m = float(hold_s), float(position_tolerance_m)
        self.heading_tolerance_rad, self.speed_tolerance_mps = float(heading_tolerance_rad), float(speed_tolerance_mps)
        super().__init__(deadline_s, validity_limits)

    def reset(self, com_xy):
        super().reset(com_xy)
        self.entered = False
        self.hold_started_s = None
        self.previous_local = self.berth.local(com_xy)
        self.previous_local_time_s = 0.0
        self.berth_entry_time_s = None

    def _conjunction(self, position, quat, velocity):
        yaw = _yaw(quat)
        return (self.entered and self.berth.contains_polygon(position, yaw, self.hull_vertices_m)
                and _distance(position, self.berth.center_xy) <= self.position_tolerance_m
                and abs(_wrapped(yaw - self.berth.heading_rad)) <= self.heading_tolerance_rad
                and math.hypot(velocity[0], velocity[1]) <= self.speed_tolerance_mps)

    def observe_physics(self, state, **kwargs):
        super().observe_physics(state, **kwargs)
        if self.done:
            return
        values = _safe_state(state)
        hit = bool(kwargs.get("collision", False)) or (
            kwargs.get("clearance_m") is not None and not float(kwargs["clearance_m"]) > 0.0)
        if self.validity.reason != TerminationReason.RUNNING or hit or values is None \
                or not self._conjunction(values[0], values[1], values[2]):
            self.hold_started_s = None

    def _entry_update(self, time_s, position, quat, velocity):
        local = self.berth.local(position)
        previous = self.previous_local
        dx = local[0] - previous[0]
        if not self.entered and previous[0] < self.berth.opening_x_m <= local[0] and dx > 0.0:
            fraction = (self.berth.opening_x_m - previous[0]) / dx
            y_cross = previous[1] + fraction * (local[1] - previous[1])
            rel = _wrapped(_yaw(quat) - self.berth.heading_rad)
            half = max(abs(math.sin(rel) * vx + math.cos(rel) * vy) for vx, vy in self.hull_vertices_m)
            inward = velocity[0] * math.cos(self.berth.heading_rad) + velocity[1] * math.sin(self.berth.heading_rad)
            if self.clean and inward > 0.0 and math.cos(rel) > 0.0 and abs(y_cross) + half <= self.berth.half_width_m:
                self.entered = True
                self.berth_entry_time_s = self.previous_local_time_s + fraction * (time_s - self.previous_local_time_s)
        self.previous_local = local
        self.previous_local_time_s = time_s
        return local

    def _docking_candidate(self, time_s, obs):
        empty = dict(docking_position_error_m=None, docking_heading_error_rad=None,
                     docking_planar_speed_mps=None, docking_longitudinal_error_m=None,
                     docking_lateral_error_m=None, berth_contained=False, berth_entry_latched=self.entered,
                     berth_entry_time_s=self.berth_entry_time_s, docking_valid_dwell_s=0.0)
        if obs is None:
            return False, empty
        position, quat, velocity = obs[0][0], obs[0][1], obs[0][2]
        local = self._entry_update(time_s, position, quat, velocity)
        yaw = _yaw(quat)
        conj = self.clean and self._conjunction(position, quat, velocity)
        if conj:
            if self.hold_started_s is None:
                self.hold_started_s = time_s
        else:
            self.hold_started_s = None
        dwell = time_s - self.hold_started_s if conj else 0.0
        extras = dict(docking_position_error_m=_distance(position, self.berth.center_xy),
                      docking_heading_error_rad=abs(_wrapped(yaw - self.berth.heading_rad)),
                      docking_planar_speed_mps=math.hypot(velocity[0], velocity[1]),
                      docking_longitudinal_error_m=local[0], docking_lateral_error_m=local[1],
                      berth_contained=self.berth.contains_polygon(position, yaw, self.hull_vertices_m),
                      berth_entry_latched=self.entered, berth_entry_time_s=self.berth_entry_time_s,
                      docking_valid_dwell_s=dwell)
        return conj and dwell >= self.hold_s - 1.0e-9, extras

    def step(self, time_s, state, *, collision=False, out_of_bounds=False, clearance_m=None):
        obs, contact = self._observe(time_s, state, collision, clearance_m=clearance_m)
        candidate, extras = self._docking_candidate(time_s, obs)
        return self._finish(time_s, candidate=candidate, collision=contact, out_of_bounds=out_of_bounds,
                            extras=extras)


@dataclass(frozen=True)
class VirtualGate:
    """Finite gate line through midpoint, required passage along +normal.
    Crossing = COM segment crosses the line inside +-half_width, after an
    earlier sample at least approach_m before it, with normal speed above
    min_normal_speed (legacy harbor_geometry.py:162-223 semantics, COM based)."""

    name: str
    midpoint_xy: tuple
    normal_xy: tuple
    half_width_m: float
    approach_m: float = 1.0
    min_normal_speed_mps: float = 0.2


class HarborMissionEpisode(DockingEpisode):
    """Full mission (spawn_phase 0): ordered departure gates, then the transit
    exit gate, then the task-4 dock, in STRICT event order with a clean valid
    prefix. Isolated DockPhase (spawn_phase 2, C1 finding 28): its own
    contract scope, success = valid entry + dwell; departure/transit are
    reported as None (unobserved), never fabricated."""

    def __init__(self, berth, hull_vertices_m, deadline_s, *, departure_gates=(), transit_gate=None,
                 spawn_phase=0, **kwargs):
        if spawn_phase not in (0, 2):
            raise ValueError("supported spawn phases are 0 (full mission) and 2 (isolated dock)")
        if spawn_phase == 0 and (not departure_gates or transit_gate is None):
            raise ValueError("full mission requires departure gates and a transit gate")
        self.spawn_phase = spawn_phase
        self.departure_gates = tuple(departure_gates)
        self.transit_gate = transit_gate
        super().__init__(berth, hull_vertices_m, deadline_s, **kwargs)

    def reset(self, com_xy):
        super().reset(com_xy)
        self.gate_times_s = [None] * len(self.departure_gates)
        self.transit_time_s = None
        gates = self.departure_gates + ((self.transit_gate,) if self.transit_gate is not None else ())
        self._armed = [self._signed(g, com_xy) <= -g.approach_m for g in gates]

    @staticmethod
    def _signed(gate, p):
        return (p[0] - gate.midpoint_xy[0]) * gate.normal_xy[0] + (p[1] - gate.midpoint_xy[1]) * gate.normal_xy[1]

    def _crossed(self, k, gate, p0, p1, velocity):
        s0, s1 = self._signed(gate, p0), self._signed(gate, p1)
        if not self._armed[k]:
            self._armed[k] = s1 <= -gate.approach_m
            return False
        if not (s0 <= 0.0 <= s1 and s1 - s0 > 1.0e-12):
            return False
        f = -s0 / (s1 - s0)
        xi = (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]))
        tx, ty = -gate.normal_xy[1], gate.normal_xy[0]
        lateral = abs((xi[0] - gate.midpoint_xy[0]) * tx + (xi[1] - gate.midpoint_xy[1]) * ty)
        vn = velocity[0] * gate.normal_xy[0] + velocity[1] * gate.normal_xy[1]
        return lateral <= gate.half_width_m and vn > gate.min_normal_speed_mps

    @property
    def departure_time_s(self):
        return self.gate_times_s[-1] if self.spawn_phase == 0 and all(t is not None for t in self.gate_times_s) else None

    def step(self, time_s, state, *, collision=False, out_of_bounds=False, clearance_m=None):
        obs, contact = self._observe(time_s, state, collision, clearance_m=clearance_m)
        if self.spawn_phase == 0 and obs is not None and self.clean and self.validity.reason == 0:
            p1, velocity = obs[0][0], obs[0][2]
            p0 = obs[1]
            for k, g in enumerate(self.departure_gates):
                prior_ok = k == 0 or self.gate_times_s[k - 1] is not None
                if self.gate_times_s[k] is None and self._crossed(k, g, p0, p1, velocity) and prior_ok:
                    self.gate_times_s[k] = time_s
            kt = len(self.departure_gates)
            if self.transit_time_s is None and self._crossed(kt, self.transit_gate, p0, p1, velocity) \
                    and self.departure_time_s is not None and time_s > self.departure_time_s:
                self.transit_time_s = time_s
        ready_for_entry = self.spawn_phase == 2 or self.transit_time_s is not None
        if not ready_for_entry:
            # entry before the transit milestone does not count; keep the
            # previous local sample current so a later genuine entry is seen
            self.entered = False
            self.berth_entry_time_s = None
        candidate, extras = self._docking_candidate(time_s, obs)
        if self.spawn_phase == 0 and not (self.transit_time_s is not None and self.berth_entry_time_s is not None
                                          and self.transit_time_s <= self.berth_entry_time_s):
            candidate = False
        extras.update(departure_gate_times_s=tuple(self.gate_times_s) if self.spawn_phase == 0 else None,
                      departure_time_s=self.departure_time_s, transit_time_s=self.transit_time_s,
                      stage_departure_complete=self.departure_time_s is not None,
                      stage_transit_complete=self.transit_time_s is not None,
                      contract_scope="harbour_dock_phase" if self.spawn_phase == 2 else "harbour_full_mission",
                      isolated_phase=self.spawn_phase == 2)
        return self._finish(time_s, candidate=candidate, collision=contact, out_of_bounds=out_of_bounds,
                            extras=extras)


class StationKeepingEpisode(EpisodeBase):
    """Continuous in-region hold spanning the declared late interval
    [T - hold_s, T] (C1 finding 26). Success is decided only at T; the
    episode ends by TIMEOUT with episode_success true on success."""

    stop_on_success = False

    def __init__(self, station_xy, radius_m, deadline_s, *, hold_s=60.0, late_interval_s=None,
                 top_dose_start_s=None, final_window_s=60.0, validity_limits=ValidityLimits()):
        self.station_xy = _vector(station_xy, 2)
        self.radius_m, self.hold_s = float(radius_m), float(hold_s)
        self.final_window_s = float(final_window_s)
        self.late_interval_s = tuple(late_interval_s or (deadline_s - hold_s, deadline_s))
        self.top_dose_start_s = self.late_interval_s[0] if top_dose_start_s is None else float(top_dose_start_s)
        start, end = self.late_interval_s
        if not all(math.isfinite(v) for v in (radius_m, hold_s, final_window_s, start, end, self.top_dose_start_s)):
            raise ValueError("station timing and thresholds must be finite")
        if radius_m <= 0 or hold_s <= 0 or not 0 < final_window_s <= deadline_s:
            raise ValueError("invalid station radius/hold/window")
        if not 0.0 <= start < end or abs(end - deadline_s) > 1.0e-9 or end - start < hold_s - 1.0e-9:
            raise ValueError("late interval must end at the deadline and contain the required hold")
        if not 0.0 <= self.top_dose_start_s <= start + 1.0e-9:
            raise ValueError("the ramp must deliver its top dose no later than the scored interval start")
        super().__init__(deadline_s, validity_limits)

    def reset(self, com_xy):
        super().reset(com_xy)
        self.hold_started_s = 0.0 if _distance(com_xy, self.station_xy) <= self.radius_m else None
        self.intervals = []
        self.previous_error = _distance(com_xy, self.station_xy)
        self.previous_inside = self.hold_started_s is not None
        self.longest_hold_s = 0.0

    def observe_physics(self, state, **kwargs):
        super().observe_physics(state, **kwargs)
        if self.done:
            return
        values = _safe_state(state)
        hit = bool(kwargs.get("collision", False)) or (
            kwargs.get("clearance_m") is not None and not float(kwargs["clearance_m"]) > 0.0)
        inside = (values is not None and self.validity.reason == TerminationReason.RUNNING and not hit
                  and _distance(values[0], self.station_xy) <= self.radius_m)
        if not inside:
            self.hold_started_s = None
            self.previous_inside = False

    def _station_metrics(self):
        begin = self.deadline_s - self.final_window_s
        observed = squared = inside_duration = longest = run = 0.0
        for t0, t1, e0, e1, inside in self.intervals:
            lo = max(t0, begin)
            dt = t1 - lo
            if dt <= 0.0:
                continue
            f = (lo - t0) / (t1 - t0)
            ea = e0 + f * (e1 - e0)
            observed += dt
            squared += dt * (ea * ea + ea * e1 + e1 * e1) / 3.0   # exact for linear e(t)
            if inside:
                inside_duration += dt
                run += dt
                longest = max(longest, run)
            else:
                run = 0.0
        return dict(station_final_window_rms_error_m=math.sqrt(squared / observed) if observed else None,
                    station_final_window_in_region_fraction=inside_duration / observed if observed else None,
                    station_final_window_longest_hold_s=longest if observed else None,
                    station_final_window_observed_s=observed,
                    station_final_window_coverage_fraction=observed / self.final_window_s,
                    station_longest_hold_s=self.longest_hold_s,
                    station_late_interval_s=self.late_interval_s,
                    station_top_dose_start_s=self.top_dose_start_s)

    def step(self, time_s, state, *, collision=False, out_of_bounds=False, clearance_m=None):
        previous_time = self.metrics.previous_time_s
        obs, contact = self._observe(time_s, state, collision, clearance_m=clearance_m)
        inside = False
        if obs is not None:
            error = _distance(obs[0][0], self.station_xy)
            inside = error <= self.radius_m and self.validity.reason == 0 and self.clean
            self.intervals.append((previous_time, time_s, self.previous_error, error, inside and self.previous_inside))
            self.previous_error = error
        if inside:
            if self.hold_started_s is None:
                self.hold_started_s = time_s
            self.longest_hold_s = max(self.longest_hold_s, time_s - self.hold_started_s)
        else:
            self.hold_started_s = None
        self.previous_inside = inside
        begin, end = self.late_interval_s
        candidate = (inside and self.hold_started_s is not None and self.hold_started_s <= begin + 1.0e-9
                     and time_s >= end - 1.0e-9 and time_s - self.hold_started_s >= self.hold_s - 1.0e-9)
        return self._finish(time_s, candidate=candidate, collision=contact, out_of_bounds=out_of_bounds,
                            extras=self._station_metrics())


def cross_track_error_m(point, route_xy):
    """Shortest COM distance to a (sub)polyline, metres."""
    point = _vector(point, 2)
    route = tuple(_vector(p, 2) for p in route_xy)
    if len(route) < 2:
        raise ValueError("route requires at least two points")
    best = math.inf
    for a, b in zip(route, route[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / n)) if n else 0.0
        best = min(best, _distance(point, (a[0] + t * dx, a[1] + t * dy)))
    return best


class OrderedPathEpisode(EpisodeBase):
    """Visit the gate disks strictly in order (the COM motion between control
    samples is swept, so a fast pass cannot tunnel); success when the last
    disk is entered cleanly. XTE is integrated to the ACTIVE segment only."""

    def __init__(self, route_xy, gate_radius_m, deadline_s, validity_limits=ValidityLimits()):
        self.route_xy = tuple(_vector(p, 2) for p in route_xy)
        if len(self.route_xy) < 2 or isinstance(gate_radius_m, bool) or not math.isfinite(gate_radius_m) \
                or gate_radius_m <= 0:
            raise ValueError("path needs a start, gates, and a positive radius")
        self.gate_radius_m = float(gate_radius_m)
        super().__init__(deadline_s, validity_limits)

    def reset(self, com_xy):
        super().reset(com_xy)
        self.next_gate = 1
        self.gate_times_s = []
        self.metrics.previous_cross_track_error_m = cross_track_error_m(com_xy, self.route_xy[:2])

    def step(self, time_s, state, *, collision=False, out_of_bounds=False, clearance_m=None):
        values = _safe_state(state)
        active = self.route_xy[self.next_gate - 1:self.next_gate + 1]
        xte = None if values is None else cross_track_error_m(values[0][:2], active)
        obs, contact = self._observe(time_s, state, collision, xte, clearance_m=clearance_m)
        if obs is not None and self.next_gate < len(self.route_xy) \
                and segment_hits_disk(obs[1], obs[0][0], self.route_xy[self.next_gate], self.gate_radius_m):
            self.gate_times_s.append(time_s)
            self.next_gate += 1
            if self.next_gate < len(self.route_xy):
                self.metrics.previous_cross_track_error_m = cross_track_error_m(
                    obs[0][0][:2], self.route_xy[self.next_gate - 1:self.next_gate + 1])
        return self._finish(time_s, candidate=self.next_gate == len(self.route_xy), collision=contact,
                            out_of_bounds=out_of_bounds,
                            extras=dict(ordered_gates_completed=self.next_gate - 1,
                                        ordered_gates_total=len(self.route_xy) - 1,
                                        gate_times_s=tuple(self.gate_times_s)))


@dataclass
class ThreadingPassage:
    """One identified portal (C1 finding 45): pay only after the FULL hull
    has crossed the exit plane, cleanly and inside the aperture throughout.
    Input is the COM in portal-local coordinates (x along the passage) and
    conservative longitudinal/lateral hull half extents at every sample."""

    portal_id: str
    entry_x_m: float
    exit_x_m: float
    half_width_m: float
    bonus: float = 5.0
    active: bool = False
    paid: bool = False
    failed: bool = False
    worst_clearance_m: float = math.inf
    previous_x_m: float | None = None
    previous_y_m: float | None = None
    previous_hx_m: float | None = None
    previous_hy_m: float | None = None

    def __post_init__(self):
        if not self.portal_id or not all(math.isfinite(x) for x in (self.entry_x_m, self.exit_x_m,
                                                                     self.half_width_m, self.bonus)) \
                or self.entry_x_m >= self.exit_x_m or self.half_width_m <= 0 or self.bonus < 0:
            raise ValueError("invalid portal")

    def update(self, portal_id, com_local_xy, longitudinal_half_extent_m, lateral_half_extent_m, *, collision=False):
        x, y = _vector(com_local_xy, 2)
        hx, hy = float(longitudinal_half_extent_m), float(lateral_half_extent_m)
        if not all(math.isfinite(v) and v > 0 for v in (hx, hy)):
            raise ValueError("hull extents must be positive")
        if portal_id != self.portal_id or self.paid or self.failed:
            return 0.0
        previous, previous_y = self.previous_x_m, self.previous_y_m
        span_hx = max(hx, self.previous_hx_m or hx)
        span_hy = max(hy, self.previous_hy_m or hy)
        self.previous_x_m, self.previous_y_m, self.previous_hx_m, self.previous_hy_m = x, y, hx, hy
        if not self.active and previous is not None and previous + span_hx < self.entry_x_m <= x + span_hx:
            self.active = True
            fraction = (self.entry_x_m - previous - span_hx) / (x - previous)
            entry_y = previous_y + fraction * (y - previous_y)
            entry_clearance = self.half_width_m - abs(entry_y) - span_hy
            self.worst_clearance_m = min(self.worst_clearance_m, entry_clearance)
            self.failed = entry_clearance <= 0
        if self.active:
            clearance = self.half_width_m - abs(y) - span_hy
            self.worst_clearance_m = min(self.worst_clearance_m, clearance)
            self.failed |= bool(collision) or clearance <= 0 or (previous is not None and x < previous)
            if self.failed:
                self.active = False
            elif x - hx >= self.exit_x_m:
                self.paid = True
                self.active = False
                return self.bonus
        return 0.0
