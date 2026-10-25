"""Deterministic nominal ownship plans: the controller-independent reference.

A nominal plan is a bounded-control unicycle rollout that follows the task's
frozen route (or holds a frozen station) at the envelope's reference speed. It
fixes (a) where and when the encounter conflict happens, (b) the TTC dose
geometry, and (c) the witness prefix: the witness vessel behaves nominally until
the intruder is revealed plus the reaction time. No controller state, action or
outcome enters a plan.
"""

from dataclasses import dataclass
import math

from .kinematics import integrate, pose_on_states, wrap
from .model import Circle, Pose, TimedState, check_number, is_unforced_command


@dataclass(frozen=True)
class NominalPlan:
    states: tuple
    target_times: tuple = ()
    kind: str = "transit"
    reached_end: bool = True

    def __post_init__(self):
        if not isinstance(self.states, tuple) or not self.states:
            raise ValueError("plan needs a nonempty tuple of TimedState")
        if self.states[0].time_s != 0.0:
            raise ValueError("plan starts at episode time zero")
        for before, after in zip(self.states, self.states[1:]):
            if not after.time_s > before.time_s:
                raise ValueError("plan times must strictly increase")
        if self.kind not in ("transit", "stationary"):
            raise ValueError("unknown plan kind")

    @property
    def end_s(self):
        return self.states[-1].time_s

    def pose_at(self, time_s):
        for state in self.states:
            if state.time_s == time_s:
                return state.pose
        return pose_on_states(self.states, time_s)

    def is_constant_velocity(self, start_s, end_s):
        """True when every plan interval overlapping [start_s, end_s] is unforced.

        Unforced = model.is_unforced_command (G7 item B: |yaw rate| < 1e-6 rad/s and
        |accel| < 1e-6 m/s^2, with the declared deviation bound in model.py).
        """
        if start_s < 0.0 or end_s > self.end_s + 1e-9 or end_s < start_s:
            return False
        for before, after in zip(self.states, self.states[1:]):
            if after.time_s <= start_s or before.time_s >= end_s:
                continue
            if not is_unforced_command(after.accel_mps2, after.yaw_rate_rps):
                return False
        return True

    def prefix_until(self, time_s):
        """Exact state prefix ending at time_s (the last interval is split)."""
        check_number("prefix time", time_s, minimum=0.0)
        if time_s > self.end_s + 1e-9:
            raise ValueError("prefix extends beyond the nominal plan")
        out = [self.states[0]]
        for before, after in zip(self.states, self.states[1:]):
            if after.time_s <= time_s + 1e-12:
                out.append(after)
                if abs(after.time_s - time_s) <= 1e-12:
                    break
                continue
            if before.time_s < time_s:
                pose = integrate(before.pose, after.accel_mps2, after.yaw_rate_rps,
                                 time_s - before.time_s)
                out.append(TimedState(time_s, pose, after.accel_mps2, after.yaw_rate_rps))
            break
        return tuple(out)

    def distance_at(self, time_s):
        total = 0.0
        for before, after in zip(self.states, self.states[1:]):
            if before.time_s >= time_s:
                break
            dt = min(after.time_s, time_s) - before.time_s
            total += before.pose.speed * dt + 0.5 * after.accel_mps2 * dt * dt
        return total

    @property
    def path_length_m(self):
        return self.distance_at(self.end_s)

    def time_at_fraction(self, fraction):
        """First time the nominal travelled distance reaches fraction * length."""
        check_number("route fraction", fraction, minimum=0.0)
        if fraction > 1.0:
            raise ValueError("route fraction must be in [0, 1]")
        goal = fraction * self.path_length_m
        travelled = 0.0
        for before, after in zip(self.states, self.states[1:]):
            dt = after.time_s - before.time_s
            step = before.pose.speed * dt + 0.5 * after.accel_mps2 * dt * dt
            if travelled + step >= goal - 1e-12 and step > 0.0:
                need = goal - travelled
                a, v = after.accel_mps2, before.pose.speed
                if abs(a) < 1e-15:
                    local = need / v
                else:
                    disc = max(0.0, v * v + 2.0 * a * need)
                    local = (-v + math.sqrt(disc)) / a
                return before.time_s + max(0.0, min(dt, local))
            travelled += step
        return self.end_s


def build_stationary_plan(pose, horizon_s):
    """A vessel holding a frozen station pose (station keeping reference)."""
    check_number("horizon_s", horizon_s, positive=True)
    if pose.speed != 0.0:
        raise ValueError("a stationary reference has zero speed")
    return NominalPlan((TimedState(0.0, pose), TimedState(float(horizon_s), pose)),
                       (), "stationary", True)


def build_transit_plan(initial_pose, envelope, targets, dt_s=0.25, horizon_s=600.0,
                       stop_at_end=False, speed_mps=None, align_first=False):
    """Deterministic route follower through ordered target circles.

    Straight legs are exactly unforced (accel == 0 and yaw rate == 0), which is
    what makes linear TTC exact on them. Heading is corrected at the bounded
    yaw rate; speed is reduced by cos(heading error) so tight turns are
    feasible for a unicycle. The plan ends at arrival in the last target (and
    at rest when stop_at_end), or at horizon_s with reached_end=False.

    G23e: align_first=True (station keeping's approach route only) holds the
    desired speed at zero until the heading first points at the first target
    (|error| < 1e-9 rad, the same zeroing threshold as the yaw rate): the
    vessel turns in place, then runs the straight start -> target line. With
    the default False every arithmetic step is the one of the earlier plan.
    """
    check_number("dt_s", dt_s, positive=True)
    check_number("horizon_s", horizon_s, positive=True)
    targets = tuple(targets)
    if not targets or any(not isinstance(t, Circle) or t.radius <= 0.0 for t in targets):
        raise ValueError("transit plan needs positive-radius Circle targets")
    speed = envelope.reference_speed_mps if speed_mps is None else speed_mps
    check_number("plan speed", speed, positive=True)
    if speed > envelope.max_speed_mps:
        raise ValueError("plan speed exceeds the envelope")
    if initial_pose.speed > envelope.max_speed_mps:
        raise ValueError("initial speed exceeds the envelope")
    acc, dec, rate = envelope.max_accel_mps2, envelope.max_decel_mps2, envelope.max_yaw_rate_rps
    states = [TimedState(0.0, initial_pose)]
    target_times = []
    index = 0
    step = 0
    pose = initial_pose
    aligned = not align_first

    def within(target, p):
        return math.hypot(p.x - target.x, p.y - target.y) <= target.radius

    while True:
        while index < len(targets) and within(targets[index], pose):
            target_times.append(step * dt_s)
            index += 1
        if index == len(targets) and (not stop_at_end or pose.speed <= 1e-12):
            return NominalPlan(tuple(states), tuple(target_times), "transit", True)
        if (step + 1) * dt_s > horizon_s + 1e-9:
            return NominalPlan(tuple(states), tuple(target_times), "transit", False)
        if index == len(targets):
            accel = max(-dec, -pose.speed / dt_s)
            yaw_rate = 0.0
        else:
            target = targets[index]
            bearing = math.atan2(target.y - pose.y, target.x - pose.x)
            error = wrap(bearing - pose.yaw)
            yaw_rate = 0.0 if abs(error) < 1e-9 else max(-rate, min(rate, error / dt_s))
            desired = speed * max(0.0, math.cos(error))
            if not aligned:
                if abs(error) < 1e-9:
                    aligned = True
                else:
                    desired = 0.0
            if stop_at_end and index == len(targets) - 1:
                distance = math.hypot(target.x - pose.x, target.y - pose.y)
                desired = min(desired, math.sqrt(2.0 * dec * max(0.0, distance - 0.5 * target.radius)))
            gap = desired - pose.speed
            accel = 0.0 if abs(gap) < 1e-12 else max(-dec, min(acc, gap / dt_s))
            accel = max(accel, -pose.speed / dt_s)
        step += 1
        pose = integrate(pose, accel, yaw_rate, dt_s)
        states.append(TimedState(step * dt_s, pose, accel, yaw_rate))
