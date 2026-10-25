"""Immutable, simulator-free encounter inputs in SI units (pure Python).

Frames: planar world/env-local frame (x, y) in metres, yaw in radians about +z.
The vessel reference point is its centre of mass (COM); the footprint is a
body-frame rectangle about that point (+x bow, +y port), so an asymmetric hull
outline is represented exactly instead of by a COM-centred disk or box.

The vessel envelope is supplied by the vessel task (D5); this module contains
no vessel-specific numbers. t_avoid is a frozen, declared kinematic time scale
computed from that envelope. It is NOT a measured controller response and NOT a
proof that a particular scene admits avoidance (the feasibility witness is).
"""

from dataclasses import dataclass, field
import math


DOSE_TTC_RATIOS = (None, 4.0, 3.0, 2.0, 1.25)
# min_brake_steer_v1 (default): reaction + min(full-brake, max-rate turn) time to
# displace the hull by clearance_beams * beam from its unmanoeuvred course. These
# are the two classical evasive manoeuvres ("stop or turn away"). The surge
# variant also admits "accelerate ahead", which only helps crossing geometry.
T_AVOID_METHODS = ("min_brake_steer_v1", "min_brake_surge_steer_v1",
                   "serial_stop_turn_translate_v1")
DEFAULT_T_AVOID_METHOD = "min_brake_steer_v1"
PATH_TYPES = ("crossing", "head_on", "overtaking", "drifting")
REVEAL_KINDS = ("detection_horizon", "occluder_exit")

# G7 item B (owner-approved 2026-10-01 12:5x): the unforced-window predicate.
# A nominal-plan command counts as ZERO when |yaw_rate| < 1e-6 rad/s and
# |accel| < 1e-6 m/s^2. The plan's own zeroing thresholds (reference.py,
# 1e-9 rad heading error and 1e-12 m/s speed gap) are unchanged; they leave
# pursuit corrections of about 4e-9 rad/s that an exact "!= 0.0" test read as
# manoeuvres. Error bound: over a window of length W whose commands obey the
# predicate, starting at speed v, the plan's position deviates from straight
# constant-velocity motion by at most
#     |dp| <= 0.5 * (v * e_w + e_a) * W**2 + (1/3) * e_a * e_w * W**3
# (from |v(s) e^{i psi(s)} - v e^{i psi0}| <= e_a s + (v + e_a s) e_w s); its
# heading by at most e_w * W; the along-track timing error is |dp| / v.
# Longest window: WAM-V D1 = 4 x t_avoid = 4 x 3.743 s = 14.97 s at v = 1 m/s:
# |dp| <= 2.24e-4 m (<= 0.25 mm), timing <= 2.24e-4 s (<= 0.25 ms), far below
# the 1/60 s control step. The dose definitions (TTC / t_avoid) are unchanged.
UNFORCED_YAW_RATE_TOL_RPS = 1.0e-6
UNFORCED_ACCEL_TOL_MPS2 = 1.0e-6
UNFORCED_WINDOW_RULE = "abs_yaw_rate_lt_1e-6_rad_s_and_abs_accel_lt_1e-6_m_s2/1"
UNFORCED_WINDOW_BOUND = ("|dp| <= 0.5*(v*e_w + e_a)*W^2 + (1/3)*e_a*e_w*W^3, |dpsi| <= e_w*W, "
                         "timing <= |dp|/v; e_w = 1e-6 rad/s, e_a = 1e-6 m/s^2")


# TTC invariant under the predicate (declared, owner-approved timing bound "<= 0.25 ms"): when the D1
# window holds a nonzero sub-tolerance command, the constant-velocity TTC from the reveal pose may
# differ from the constructed contact time by the bounded deviation above; the invariant then accepts
# UNFORCED_TTC_TOLERANCE_S on top of the exact 1e-9 check. 0.25 ms >= |dp|/v of the longest window
# (2.24e-4 s, WAM-V D1 at 1 m/s); far below the 1/60 s control step; dose definitions unchanged.
UNFORCED_TTC_TOLERANCE_S = 2.5e-4


def is_unforced_command(accel_mps2, yaw_rate_rps):
    """The ONE unforced predicate (reference.NominalPlan.is_constant_velocity and adapters._fit_conflict)."""
    return abs(accel_mps2) < UNFORCED_ACCEL_TOL_MPS2 and abs(yaw_rate_rps) < UNFORCED_YAW_RATE_TOL_RPS


def unforced_window_position_bound_m(window_s, speed_mps, yaw_rate_tol_rps=UNFORCED_YAW_RATE_TOL_RPS,
                                     accel_tol_mps2=UNFORCED_ACCEL_TOL_MPS2):
    """Declared bound on the deviation from straight constant-velocity motion over an unforced window."""
    w, v = float(window_s), float(speed_mps)
    return (0.5 * (v * yaw_rate_tol_rps + accel_tol_mps2) * w * w
            + accel_tol_mps2 * yaw_rate_tol_rps * w ** 3 / 3.0)


def unforced_window_timing_bound_s(window_s, speed_mps):
    """Along-track timing error bound |dp| / v of an unforced window."""
    return unforced_window_position_bound_m(window_s, speed_mps) / float(speed_mps)


def check_number(name, value, minimum=None, positive=False):
    # E4 hot-loop fast path: an exact float is numeric and never a bool, so the
    # two isinstance checks are skipped; every other type takes D7's path.
    if type(value) is not float and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(name + " must be numeric")
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(name + " is below its minimum")
    if positive and value <= 0.0:
        raise ValueError(name + " must be positive")


@dataclass(frozen=True)
class EpisodeKey:
    """Same identity as ScenarioRNG keys: (eval_seed, env, episode, group)."""

    eval_seed: int
    env: int
    episode: int
    hazard_group: str

    def __post_init__(self):
        for name in ("eval_seed", "env", "episode"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(name + " must be an integer")
        if self.env < 0 or self.episode < 0:
            raise ValueError("env and episode must be nonnegative")
        if not isinstance(self.hazard_group, str) or not self.hazard_group:
            raise ValueError("hazard_group must be a nonempty string")


@dataclass(frozen=True)
class Footprint:
    """Body-frame collision rectangle about the vessel reference point (COM)."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self):
        for name in ("x_min_m", "x_max_m", "y_min_m", "y_max_m"):
            check_number(name, getattr(self, name))
        if not (self.x_min_m < self.x_max_m and self.y_min_m < self.y_max_m):
            raise ValueError("footprint needs x_min<x_max and y_min<y_max")
        # E4 hot-loop cache of derived geometry: the SAME expressions D7's
        # properties evaluated on every call, computed once. They are plain
        # instance attributes, not dataclass fields, so equality, hashing,
        # asdict() and therefore every candidate/witness digest are unchanged.
        length, beam = self.x_max_m - self.x_min_m, self.y_max_m - self.y_min_m
        corners = ((self.x_min_m, self.y_min_m), (self.x_max_m, self.y_min_m),
                   (self.x_max_m, self.y_max_m), (self.x_min_m, self.y_max_m))
        object.__setattr__(self, "_derived", (
            length, beam,
            (0.5 * (self.x_min_m + self.x_max_m), 0.5 * (self.y_min_m + self.y_max_m)),
            (0.5 * length, 0.5 * beam), corners, max(math.hypot(x, y) for x, y in corners)))

    @classmethod
    def symmetric(cls, length_m, beam_m):
        check_number("length_m", length_m, positive=True)
        check_number("beam_m", beam_m, positive=True)
        return cls(-length_m / 2.0, length_m / 2.0, -beam_m / 2.0, beam_m / 2.0)

    @property
    def length_m(self):
        return self._derived[0]

    @property
    def beam_m(self):
        return self._derived[1]

    @property
    def center(self):
        return self._derived[2]

    @property
    def half_extents(self):
        return self._derived[3]

    @property
    def corners(self):
        return self._derived[4]

    @property
    def reach_m(self):
        """Largest distance from the reference point to the outline."""
        return self._derived[5]


# ---------------------------------------------------------------- t_avoid ---
def brake_displacement_time(speed, decel, distance):
    """Seconds for full braking to open `distance` against the unmanoeuvred course."""
    if speed <= 0.0 or decel <= 0.0:
        return math.inf
    stop_distance = speed * speed / (2.0 * decel)
    if distance <= stop_distance:
        return math.sqrt(2.0 * distance / decel)
    return (distance + stop_distance) / speed


def surge_displacement_time(speed, max_speed, accel, distance):
    """Seconds for full acceleration (capped at max_speed) to open `distance`."""
    if accel <= 0.0 or max_speed <= speed:
        return math.inf
    ramp_s = (max_speed - speed) / accel
    ramp_distance = 0.5 * accel * ramp_s * ramp_s
    if distance <= ramp_distance:
        return math.sqrt(2.0 * distance / accel)
    return ramp_s + (distance - ramp_distance) / (max_speed - speed)


def _steer_shape(theta):
    return math.hypot(theta - math.sin(theta), 1.0 - math.cos(theta))


def steer_displacement_time(speed, yaw_rate, distance):
    """Seconds for a constant-speed maximum-rate turn to open `distance`.

    Displacement from the straight course is (u/r)*f(r t) with
    f(q) = |(q - sin q, 1 - cos q)|; d(f^2)/dq = 2 q (1 - cos q) >= 0, so f is
    monotone and bisection on q is exact up to the tolerance.
    """
    if speed <= 0.0 or yaw_rate <= 0.0:
        return math.inf
    target = distance * yaw_rate / speed
    if target <= 0.0:
        return 0.0
    high = 1.0
    while _steer_shape(high) < target:
        high *= 2.0
        if high > 1e9:
            return math.inf
    low = 0.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _steer_shape(mid) < target:
            low = mid
        else:
            high = mid
    return high / yaw_rate


def serial_stop_turn_translate_time(speed, decel, yaw_rate, max_speed, accel, distance):
    """First-attempt definition, retained for comparison: stop, turn 90 deg, translate."""
    ramp_distance = max_speed * max_speed / (2.0 * accel)
    if distance <= ramp_distance:
        traverse = math.sqrt(2.0 * distance / accel)
    else:
        traverse = max_speed / accel + (distance - ramp_distance) / max_speed
    return speed / decel + math.pi / (2.0 * yaw_rate) + traverse


@dataclass(frozen=True)
class VesselEnvelope:
    """Planar kinematic manoeuvring envelope (bounded-control unicycle).

    reference_speed_mps: prescribed approach speed used for nominal plans and
      for t_avoid (C7 STRATEGY.md section 5: "at a prescribed approach speed").
    max_speed_mps: speed cap the witness may use (<= top_speed_mps).
    top_speed_mps: vessel top speed; the moving-basin native axis is a ratio of it.
    max_accel_mps2 / max_decel_mps2: bounds valid on [0, max_speed_mps].
    max_yaw_rate_rps: yaw-rate bound, independent of speed in this model.
    reaction_s: declared latency between visibility onset and manoeuvre start.
    clearance_beams: the displacement defining t_avoid, in vessel beams.

    With a zero reference speed only braking-free options exist: brake and
    steer both return infinity, so a stationary reference is rejected here.
    A single t_avoid per vessel, frozen at the cruise reference speed, is used
    for every task of that vessel (including station keeping).
    """

    vessel_id: str
    footprint: Footprint
    reference_speed_mps: float
    max_speed_mps: float
    top_speed_mps: float
    max_accel_mps2: float
    max_decel_mps2: float
    max_yaw_rate_rps: float
    reaction_s: float
    provenance: str
    t_avoid_method: str = DEFAULT_T_AVOID_METHOD
    clearance_beams: float = 1.0
    # Astern motion for the WITNESS only (t_avoid uses forward braking). Speed
    # may lie in [-max_reverse_speed_mps, max_speed_mps]; while moving astern
    # the negative-acceleration bound is min(max_decel, max_astern_accel).
    max_reverse_speed_mps: float = 0.0
    max_astern_accel_mps2: float = 0.0
    t_avoid_s: float = field(init=False)
    t_avoid_components: tuple = field(init=False)

    def __post_init__(self):
        for name in ("vessel_id", "provenance"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(name + " must be a nonempty string")
        if not isinstance(self.footprint, Footprint):
            raise ValueError("footprint must be a Footprint")
        for name in ("max_speed_mps", "top_speed_mps", "max_accel_mps2",
                     "max_decel_mps2", "max_yaw_rate_rps", "clearance_beams"):
            check_number(name, getattr(self, name), positive=True)
        check_number("reference_speed_mps", self.reference_speed_mps, minimum=0.0)
        check_number("reaction_s", self.reaction_s, minimum=0.0)
        check_number("max_reverse_speed_mps", self.max_reverse_speed_mps, minimum=0.0)
        check_number("max_astern_accel_mps2", self.max_astern_accel_mps2, minimum=0.0)
        if self.max_reverse_speed_mps > 0.0 and self.max_astern_accel_mps2 <= 0.0:
            raise ValueError("astern motion needs a positive astern acceleration bound")
        if self.reference_speed_mps > self.max_speed_mps:
            raise ValueError("reference speed exceeds the witness speed cap")
        if self.max_speed_mps > self.top_speed_mps:
            raise ValueError("witness speed cap exceeds top speed")
        if self.t_avoid_method not in T_AVOID_METHODS:
            raise ValueError("unknown t_avoid method")
        distance = self.clearance_beams * self.footprint.beam_m
        u = self.reference_speed_mps
        if self.t_avoid_method == "min_brake_steer_v1":
            parts = (("brake_s", brake_displacement_time(u, self.max_decel_mps2, distance)),
                     ("steer_s", steer_displacement_time(u, self.max_yaw_rate_rps, distance)))
            best = min(value for _, value in parts)
        elif self.t_avoid_method == "min_brake_surge_steer_v1":
            parts = (("brake_s", brake_displacement_time(u, self.max_decel_mps2, distance)),
                     ("surge_s", surge_displacement_time(u, self.max_speed_mps,
                                                         self.max_accel_mps2, distance)),
                     ("steer_s", steer_displacement_time(u, self.max_yaw_rate_rps, distance)))
            best = min(value for _, value in parts)
        else:
            best = serial_stop_turn_translate_time(u, self.max_decel_mps2, self.max_yaw_rate_rps,
                                                   self.max_speed_mps, self.max_accel_mps2,
                                                   distance)
            parts = (("serial_s", best),)
        value = self.reaction_s + best
        check_number("t_avoid_s", value, positive=True)
        # Infeasible strategies (infinite time) are recorded as None: stamps are strict JSON.
        parts = tuple((name, part if math.isfinite(part) else None) for name, part in parts)
        object.__setattr__(self, "t_avoid_s", value)
        object.__setattr__(self, "t_avoid_components",
                           (("reaction_s", self.reaction_s), ("clearance_m", distance)) + parts)

    @property
    def length_m(self):
        return self.footprint.length_m

    @property
    def beam_m(self):
        return self.footprint.beam_m

    @property
    def nominal_speed_mps(self):
        return self.reference_speed_mps

    def accel_lower_bound(self, end_speed):
        """Most negative admissible acceleration on an interval ending at end_speed."""
        if end_speed >= -1e-12:
            return -self.max_decel_mps2
        return -min(self.max_decel_mps2, self.max_astern_accel_mps2)


@dataclass(frozen=True)
class Pose:
    """Planar pose; speed is SIGNED surge speed (negative = astern)."""

    x: float
    y: float
    yaw: float
    speed: float = 0.0

    def __post_init__(self):
        # E4 hot-loop fast path: four exact finite floats (v - v == 0.0 is False
        # for inf and nan) always pass check_number; anything else takes D7's path.
        x, y, yaw, speed = self.x, self.y, self.yaw, self.speed
        if (type(x) is float and type(y) is float and type(yaw) is float and type(speed) is float
                and x - x == 0.0 and y - y == 0.0 and yaw - yaw == 0.0 and speed - speed == 0.0):
            return
        for name in ("x", "y", "yaw", "speed"):
            check_number(name, getattr(self, name))


@dataclass(frozen=True)
class TimedState:
    """Controls (accel, yaw rate) belong to the interval ENDING at this state."""

    time_s: float
    pose: Pose
    accel_mps2: float = 0.0
    yaw_rate_rps: float = 0.0


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float

    def __post_init__(self):
        check_number("x", self.x)
        check_number("y", self.y)
        check_number("radius", self.radius, minimum=0.0)


@dataclass(frozen=True)
class Intruder:
    """Disk physically present for the whole episode (never created/destroyed).

    Motion is rest-move-rest: x(t) = x0 + v * clamp(t, start_s, stop_s), i.e.
    at rest at x(start_s) before start_s, constant velocity v on
    [start_s, stop_s], then at rest at x(stop_s). stop_s None means it never
    stops. (x0, y0) is the extrapolated position at t = 0 of the moving line,
    so the linear formula is exact on the moving interval. Generation keeps
    the whole D1 window [t_c - 4 t_avoid, t_c] inside the moving interval.
    """

    path_type: str
    x0: float
    y0: float
    vx: float
    vy: float
    radius_m: float
    reveal_s: float
    reveal_kind: str
    start_s: float = 0.0
    stop_s: object = None

    def __post_init__(self):
        if self.path_type not in PATH_TYPES:
            raise ValueError("unsupported intruder path type")
        if self.reveal_kind not in REVEAL_KINDS:
            raise ValueError("reveal must be a detection-horizon onset or an occluder exit")
        for name in ("x0", "y0", "vx", "vy"):
            check_number(name, getattr(self, name))
        check_number("radius_m", self.radius_m, positive=True)
        check_number("reveal_s", self.reveal_s, minimum=0.0)
        check_number("start_s", self.start_s, minimum=0.0)
        if self.stop_s is not None:
            check_number("stop_s", self.stop_s, minimum=0.0)
            if self.stop_s < self.start_s:
                raise ValueError("stop_s precedes start_s")

    def _clock(self, t):
        check_number("episode time", t, minimum=0.0)
        t = max(t, self.start_s)
        return t if self.stop_s is None else min(t, self.stop_s)

    def position(self, t):
        s = self._clock(t)
        return self.x0 + self.vx * s, self.y0 + self.vy * s

    def velocity(self, t):
        """Velocity at t (right-continuous); zero while at rest."""
        check_number("episode time", t, minimum=0.0)
        moving = t >= self.start_s and (self.stop_s is None or t < self.stop_s)
        return (self.vx, self.vy) if moving else (0.0, 0.0)

    @property
    def speed_bound_mps(self):
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class Encounter:
    """Preassigned physical encounter; metadata is immutable (key, scalar) pairs.

    Intruders may also be native-motion objects providing position(t),
    radius_m, reveal_s and speed_bound_mps (the moving-basin contract).
    """

    key: EpisodeKey
    task: str
    dose: int
    t_avoid_s: float
    nominal_ttc_s: object
    intruders: tuple
    duration_s: float
    metadata: tuple = ()

    def __post_init__(self):
        if not isinstance(self.key, EpisodeKey):
            raise ValueError("encounter requires an EpisodeKey")
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("task must be a nonempty string")
        if isinstance(self.dose, bool) or not isinstance(self.dose, int) or not 0 <= self.dose <= 4:
            raise ValueError("dose must be an integer in [0, 4]")
        check_number("t_avoid_s", self.t_avoid_s, positive=True)
        check_number("duration_s", self.duration_s, positive=True)
        if self.nominal_ttc_s is not None:
            check_number("nominal_ttc_s", self.nominal_ttc_s, minimum=0.0)
        if not isinstance(self.intruders, tuple) or not isinstance(self.metadata, tuple):
            raise ValueError("intruders and metadata must be tuples")
        names = set()
        for pair in self.metadata:
            if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str):
                raise ValueError("metadata must contain (string, scalar) tuples")
            if pair[0] in names:
                raise ValueError("duplicate metadata key")
            names.add(pair[0])
            if not isinstance(pair[1], (str, int, float, bool, type(None))):
                raise ValueError("metadata values must be immutable JSON scalars")
            if isinstance(pair[1], float) and not math.isfinite(pair[1]):
                raise ValueError("metadata floats must be finite")

    def meta(self, name, default=None):
        for key, value in self.metadata:
            if key == name:
                return value
        return default
