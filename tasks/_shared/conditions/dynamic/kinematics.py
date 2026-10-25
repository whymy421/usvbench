"""Exact bounded-control unicycle integration (shared by plans and witnesses)."""

import math

from .model import Pose


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def integrate(pose, accel_mps2, yaw_rate_rps, dt_s):
    """Exact forward unicycle integration with constant acceleration/yaw rate.

    Closed form of int_0^T (v0 + a t) exp(i(psi0 + w t)) dt written about the
    interval midpoint; the series branch avoids cancellation for tiny w*T.
    Controls and speed are NOT silently saturated here.
    """
    q = yaw_rate_rps * dt_s * 0.5
    if abs(q) < 1e-3:
        sinc = 1.0 - q * q / 6.0 + q ** 4 / 120.0
        cross = q / 3.0 - q ** 3 / 30.0 + q ** 5 / 840.0
    else:
        sinc = math.sin(q) / q
        cross = (math.sin(q) - q * math.cos(q)) / (q * q)
    mid_yaw = pose.yaw + q
    along = (pose.speed + 0.5 * accel_mps2 * dt_s) * dt_s * sinc
    across = accel_mps2 * dt_s * dt_s * 0.5 * cross
    end_speed = pose.speed + accel_mps2 * dt_s
    if -1e-12 < end_speed < 0.0:
        end_speed = 0.0  # floating-point roundoff at an exact bounded stop
    return Pose(
        pose.x + along * math.cos(mid_yaw) - across * math.sin(mid_yaw),
        pose.y + along * math.sin(mid_yaw) + across * math.cos(mid_yaw),
        wrap(pose.yaw + yaw_rate_rps * dt_s),
        end_speed,
    )


def pose_on_states(states, time_s):
    """Pose at time_s on a TimedState sequence (controls belong to the ending state)."""
    if not states:
        raise ValueError("empty state sequence")
    if time_s < states[0].time_s - 1e-12 or time_s > states[-1].time_s + 1e-9:
        raise ValueError("time outside the state sequence")
    if time_s <= states[0].time_s:
        return states[0].pose
    for before, after in zip(states, states[1:]):
        if time_s <= after.time_s:
            return integrate(before.pose, after.accel_mps2, after.yaw_rate_rps,
                             time_s - before.time_s)
    return states[-1].pose
