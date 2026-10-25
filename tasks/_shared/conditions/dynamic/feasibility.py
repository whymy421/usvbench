"""Time-dependent feasibility witness: a bounded-control unicycle that behaves
nominally until it could see the hazard, then avoids it and completes the task.

This is a controller-independent ADMISSION model, not a claim about high-
fidelity vessel dynamics. A failed finite search means only that this frozen
search found no witness; the candidate is then rejected and counted. Neither
search nor replay reads observations, controller actions, or outcomes.

Model: signed surge speed v in [-max_reverse, max_speed], acceleration in
[accel_lower_bound(v_end), max_accel] (forward braking bound while v >= 0,
the smaller of braking/astern bounds once astern), |yaw rate| <= max_yaw_rate.
Both vessels have astern thrust, so forbidding reverse would falsely reject
feasible head-on encounters (and bias the admitted set toward easy scenes).

Clearance certification is continuous in time: on every subinterval the
distance to each obstacle is Lipschitz with constant
    |vessel speed| bound + footprint reach * |yaw rate| + obstacle speed bound,
so (midpoint clearance - Lipschitz * half-subinterval) lower-bounds clearance
over the whole subinterval, including curved motion and reflecting/orbiting
native obstacles (whose speed bound is their true maximum speed). |v| is
linear-in-time's absolute value, so its maximum is at a subinterval endpoint.
"""

from dataclasses import dataclass
import math

from .geometry import circle_clearance, point_clearance, segment_circle_min_distance, world_aabb_half_extents
from .kinematics import integrate, wrap
from .model import Circle, Encounter, Pose, TimedState, VesselEnvelope

WITNESS_VERSION = "bounded_unicycle_lipschitz_prefix_v3_astern_tangent"


@dataclass(frozen=True)
class WitnessRequest:
    initial_pose: Pose
    envelope: VesselEnvelope
    encounter: Encounter
    bounds: tuple
    goal: Circle
    static_obstacles: tuple = ()
    waypoints: tuple = ()
    dt_s: float = 0.25
    horizon_s: float = 60.0
    hold_circle: object = None
    hold_from_s: float = 0.0
    dwell_s: float = 0.0
    final_speed_mps: object = None
    final_heading_rad: object = None
    heading_tolerance_rad: float = 0.15
    clearance_m: float = 0.02
    beam_width: int = 64
    collision_substeps: int = 4
    minimum_completion_s: float = 0.0
    prefix: tuple = ()


@dataclass(frozen=True)
class WitnessResult:
    accepted: bool
    reason: str
    trajectory: tuple = ()
    min_clearance_m: object = None
    sampled_states: int = 0
    rejected_states: int = 0
    search_exhausted: bool = False


@dataclass(frozen=True)
class _Node:
    trajectory: tuple
    waypoint: int
    dwell: float
    clearance: float


def release_time(request):
    """Earliest time the witness may deviate from the nominal prefix."""
    if not request.encounter.intruders:
        return 0.0
    return max(obj.reveal_s for obj in request.encounter.intruders) + request.envelope.reaction_s


def completion_time(request):
    """The certificate must cover the encounter, not end in its prefix."""
    nominal_ttc = request.encounter.nominal_ttc_s or 0.0
    encounter_time = max((obj.reveal_s for obj in request.encounter.intruders), default=0.0)
    return max(request.minimum_completion_s, encounter_time + nominal_ttc)


def _prefix(request):
    return request.prefix if request.prefix else (TimedState(0.0, request.initial_pose),)


def _speed_ok(env, speed):
    return -env.max_reverse_speed_mps - 1e-9 <= speed <= env.max_speed_mps + 1e-9


def _control_ok(env, start_speed, accel, yaw_rate, duration):
    end_speed = start_speed + accel * duration
    return (accel <= env.max_accel_mps2 + 1e-9
            and accel >= env.accel_lower_bound(end_speed) - 1e-9
            and abs(yaw_rate) <= env.max_yaw_rate_rps + 1e-9
            and _speed_ok(env, end_speed))


def _validate_request(request):
    env = request.envelope
    if not all(isinstance(value, tuple) for value in
               (request.bounds, request.static_obstacles, request.waypoints, request.prefix)):
        raise ValueError("bounds, static obstacles, waypoints and prefix must be immutable tuples")
    if len(request.bounds) != 4:
        raise ValueError("bounds must be xmin,xmax,ymin,ymax")
    numeric = (request.dt_s, request.horizon_s, request.dwell_s,
               request.heading_tolerance_rad, request.clearance_m,
               request.minimum_completion_s, request.hold_from_s, *request.bounds)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in numeric):
        raise ValueError("witness parameters must be finite")
    if request.bounds[0] >= request.bounds[1] or request.bounds[2] >= request.bounds[3]:
        raise ValueError("bounds must be xmin,xmax,ymin,ymax")
    if request.dt_s <= 0 or request.horizon_s <= 0 or request.dwell_s < 0:
        raise ValueError("positive time step/deadline and nonnegative dwell required")
    if request.clearance_m < 0 or request.minimum_completion_s < 0 or request.hold_from_s < 0:
        raise ValueError("clearance, minimum completion and hold start must be nonnegative")
    if not isinstance(request.beam_width, int) or not 1 <= request.beam_width <= 4096:
        raise ValueError("beam_width must be an integer in [1,4096]")
    if not isinstance(request.collision_substeps, int) or not 1 <= request.collision_substeps <= 256:
        raise ValueError("collision_substeps must be an integer in [1,256]")
    if not _speed_ok(env, request.initial_pose.speed):
        raise ValueError("initial speed is outside the envelope")
    for region in (*request.static_obstacles, *request.waypoints, request.goal, request.hold_circle):
        if region is not None and not isinstance(region, Circle):
            raise ValueError("geometry must be Circle objects")
    if request.final_speed_mps is not None and not (
            math.isfinite(request.final_speed_mps) and 0 <= request.final_speed_mps <= env.max_speed_mps):
        raise ValueError("final speed is outside the envelope")
    if request.final_heading_rad is not None and not math.isfinite(request.final_heading_rad):
        raise ValueError("final heading must be finite")
    if not 0 < request.heading_tolerance_rad <= math.pi:
        raise ValueError("heading tolerance must be in (0,pi]")
    for obj in request.encounter.intruders:
        if obj.radius_m <= 0 or obj.speed_bound_mps < 0 or not all(
                math.isfinite(v) for v in (obj.radius_m, obj.speed_bound_mps, obj.reveal_s)):
            raise ValueError("invalid physical intruder")
    if request.encounter.duration_s + 1e-9 < request.horizon_s:
        raise ValueError("physical intruder trajectories must cover the witness deadline")
    prefix = _prefix(request)
    if prefix[0] != TimedState(0.0, request.initial_pose):
        raise ValueError("prefix must start at the initial pose at time zero")
    for before, after in zip(prefix, prefix[1:]):
        duration = after.time_s - before.time_s
        if not 0 < duration <= request.dt_s + 1e-9:
            raise ValueError("prefix intervals must be positive and at most dt_s")
        if not _control_ok(env, before.pose.speed, after.accel_mps2, after.yaw_rate_rps, duration):
            raise ValueError("prefix controls or speed outside the envelope")
        expected = integrate(before.pose, after.accel_mps2, after.yaw_rate_rps, duration)
        if max(abs(expected.x - after.pose.x), abs(expected.y - after.pose.y),
               abs(wrap(expected.yaw - after.pose.yaw)),
               abs(expected.speed - after.pose.speed)) > 1e-7:
            raise ValueError("prefix is not a kinematic rollout")
    if request.encounter.intruders and abs(prefix[-1].time_s - release_time(request)) > 1e-9:
        raise ValueError("nominal prefix must end exactly at reveal plus reaction")


def _point_clearances(request, pose, at_s, span_end_s=None):
    """Yield (clearance, obstacle speed bound, uses_hull) for every constraint.

    span_end_s is the end of the certified sub-interval around at_s; the hold
    constraint is applied to every sub-interval that reaches past hold_from_s
    (conservative: it may also bind slightly before the settled clock).
    """
    footprint = request.envelope.footprint
    xmin, xmax, ymin, ymax = request.bounds
    lo_x, hi_x, lo_y, hi_y = world_aabb_half_extents(pose, footprint)
    yield min(pose.x + lo_x - xmin, xmax - pose.x - hi_x,
              pose.y + lo_y - ymin, ymax - pose.y - hi_y), 0.0, True
    for obj in request.static_obstacles:
        yield circle_clearance(pose, footprint, obj), 0.0, True
    for obj in request.encounter.intruders:
        x, y = obj.position(at_s)
        # Collision geometry exists before a sensor reveal as well as after it.
        yield point_clearance(x, y, pose, footprint) - obj.radius_m, obj.speed_bound_mps, True
    reaches = at_s if span_end_s is None else span_end_s
    if request.hold_circle is not None and reaches > request.hold_from_s - 1e-12:
        hold = request.hold_circle
        # The holding metric constrains the reference point, not the hull.
        yield hold.radius - math.hypot(pose.x - hold.x, pose.y - hold.y), 0.0, False


def _interval_clearance_reference(request, start, accel, yaw_rate, start_s, duration):
    """D7's original formulation (kept verbatim as the reference for tests)."""
    if duration == 0:
        return min(c for c, _, _ in _point_clearances(request, start, start_s))
    sub_dt = duration / request.collision_substeps
    reach = request.envelope.footprint.reach_m
    min_clearance = math.inf
    for index in range(request.collision_substeps):
        local_s = (index + 0.5) * sub_dt
        pose = integrate(start, accel, yaw_rate, local_s)
        v_bound = max(abs(start.speed + accel * index * sub_dt),
                      abs(start.speed + accel * (index + 1) * sub_dt))
        span_end = start_s + (index + 1) * sub_dt
        for clearance, obstacle_speed, uses_hull in _point_clearances(request, pose, start_s + local_s,
                                                                      span_end):
            rate = v_bound + obstacle_speed + (reach * abs(yaw_rate) if uses_hull else 0.0)
            min_clearance = min(min_clearance, clearance - rate * sub_dt * 0.5)
    return min_clearance


def _interval_clearance(request, start, accel, yaw_rate, start_s, duration):
    """E4 hot-loop version of _interval_clearance_reference, bit-identical.

    The same floating-point expressions in the same association order as
    _point_clearances / geometry.point_clearance / world_aabb_half_extents /
    to_body, inlined so each sub-step evaluates cos/sin of the pose once and
    no generator or helper call is made per constraint. min() over the
    constraint terms is order-independent for non-NaN floats, and every term
    is finite or inf (never NaN) because poses and obstacles are validated
    finite. Equality with the reference is asserted by tests/test_e4.py and
    by the full-grid admission digests (E4 TEST_REPORT).
    """
    if duration == 0:
        return min(c for c, _, _ in _point_clearances(request, start, start_s))
    sub_dt = duration / request.collision_substeps
    footprint = request.envelope.footprint
    reach = footprint.reach_m
    corners = footprint.corners
    fcx, fcy = footprint.center
    fhx, fhy = footprint.half_extents
    xmin, xmax, ymin, ymax = request.bounds
    statics = request.static_obstacles
    intruders = request.encounter.intruders
    hold = request.hold_circle
    hold_from = request.hold_from_s - 1e-12
    hull_turn = reach * abs(yaw_rate)
    hypot, cos, sin = math.hypot, math.cos, math.sin
    min_clearance = math.inf
    for index in range(request.collision_substeps):
        local_s = (index + 0.5) * sub_dt
        pose = integrate(start, accel, yaw_rate, local_s)
        v_bound = max(abs(start.speed + accel * index * sub_dt),
                      abs(start.speed + accel * (index + 1) * sub_dt))
        span_end = start_s + (index + 1) * sub_dt
        at_s = start_s + local_s
        px, py = pose.x, pose.y
        c, s = cos(pose.yaw), sin(pose.yaw)
        # bounds (world_aabb_half_extents + the min of four margins)
        xs = [c * bx - s * by for bx, by in corners]
        ys = [s * bx + c * by for bx, by in corners]
        clearance = min(px + min(xs) - xmin, xmax - px - max(xs),
                        py + min(ys) - ymin, ymax - py - max(ys))
        rate = v_bound + 0.0 + hull_turn
        min_clearance = min(min_clearance, clearance - rate * sub_dt * 0.5)
        # static disks (circle_clearance = point_clearance - radius)
        for obj in statics:
            dx, dy = obj.x - px, obj.y - py
            qx, qy = abs(c * dx + s * dy - fcx) - fhx, abs(-s * dx + c * dy - fcy) - fhy
            clearance = hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - obj.radius
            rate = v_bound + 0.0 + hull_turn
            min_clearance = min(min_clearance, clearance - rate * sub_dt * 0.5)
        # moving objects, physically present before and after any reveal
        for obj in intruders:
            x, y = obj.position(at_s)
            dx, dy = x - px, y - py
            qx, qy = abs(c * dx + s * dy - fcx) - fhx, abs(-s * dx + c * dy - fcy) - fhy
            clearance = hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - obj.radius_m
            rate = v_bound + obj.speed_bound_mps + hull_turn
            min_clearance = min(min_clearance, clearance - rate * sub_dt * 0.5)
        if hold is not None and span_end > hold_from:
            clearance = hold.radius - hypot(px - hold.x, py - hold.y)
            rate = v_bound + 0.0 + 0.0
            min_clearance = min(min_clearance, clearance - rate * sub_dt * 0.5)
    return min_clearance


def _within(region, pose):
    return math.hypot(pose.x - region.x, pose.y - region.y) <= region.radius


def _waypoint_index(request, pose, index):
    while index < len(request.waypoints) and _within(request.waypoints[index], pose):
        index += 1
    return index


def _eligible(request, pose, waypoint):
    if waypoint != len(request.waypoints) or not _within(request.goal, pose):
        return False
    if request.final_speed_mps is not None and abs(pose.speed) > request.final_speed_mps + 1e-9:
        return False
    if request.final_heading_rad is not None and abs(wrap(pose.yaw - request.final_heading_rad)) > request.heading_tolerance_rad:
        return False
    return True


def _dwell_interval(request, pose, accel, yaw_rate, duration, waypoint):
    if waypoint != len(request.waypoints):
        return False
    mid = integrate(pose, accel, yaw_rate, duration * 0.5)
    max_speed = max(abs(pose.speed), abs(pose.speed + accel * duration))
    if math.hypot(mid.x - request.goal.x, mid.y - request.goal.y) + max_speed * duration * 0.5 > request.goal.radius:
        return False
    if request.final_speed_mps is not None and max_speed > request.final_speed_mps + 1e-9:
        return False
    if request.final_heading_rad is not None and abs(wrap(mid.yaw - request.final_heading_rad)) + abs(yaw_rate) * duration * 0.5 > request.heading_tolerance_rad:
        return False
    return True


def _done(request, node):
    state = node.trajectory[-1]
    return (state.time_s + 1e-9 >= completion_time(request)
            and _eligible(request, state.pose, node.waypoint)
            and node.dwell + 1e-9 >= request.dwell_s)


def _bounded_accel(request, pose, desired, duration):
    """Clamp a desired acceleration into the signed-speed envelope for one interval."""
    env = request.envelope
    v = pose.speed
    accel = min(desired, env.max_accel_mps2, (env.max_speed_mps - v) / duration)
    accel = max(accel, (-env.max_reverse_speed_mps - v) / duration)
    lower = env.accel_lower_bound(v + accel * duration)
    return max(accel, lower)


def _advance(request, node, accel, yaw_rate, duration):
    old = node.trajectory[-1]
    clearance = _interval_clearance(request, old.pose, accel, yaw_rate, old.time_s, duration)
    if clearance <= request.clearance_m:
        return None
    pose = integrate(old.pose, accel, yaw_rate, duration)
    waypoint = _waypoint_index(request, pose, node.waypoint)
    dwell = node.dwell + duration if _dwell_interval(request, old.pose, accel, yaw_rate, duration, node.waypoint) else 0.0
    state = TimedState(old.time_s + duration, pose, accel, yaw_rate)
    return _Node(node.trajectory + (state,), waypoint, dwell, min(node.clearance, clearance))


def _prefix_node(request):
    """Advance through the frozen nominal prefix; None if the prefix is unsafe."""
    prefix = _prefix(request)
    clearance = _interval_clearance(request, prefix[0].pose, 0.0, 0.0, 0.0, 0.0)
    node = _Node((prefix[0],), _waypoint_index(request, prefix[0].pose, 0), 0.0, clearance)
    if clearance <= request.clearance_m:
        return None, node
    for state in prefix[1:]:
        old = node.trajectory[-1]
        advanced = _advance(request, node, state.accel_mps2, state.yaw_rate_rps,
                            state.time_s - old.time_s)
        if advanced is None:
            return None, node
        # Keep the exact supplied prefix object for bitwise replay equality.
        node = _Node(node.trajectory + (state,), advanced.waypoint, advanced.dwell,
                     advanced.clearance)
    return node, node


def _step_duration(request, time_s):
    return min(request.dt_s, request.horizon_s - time_s)


def _target(request, node):
    return request.waypoints[node.waypoint] if node.waypoint < len(request.waypoints) else request.goal


def _snapshot(request, at_s):
    circles = list(request.static_obstacles)
    for obj in request.encounter.intruders:
        x, y = obj.position(at_s)
        circles.append(Circle(x, y, obj.radius_m))
    return circles


def _steer_heading(request, pose, target, at_s, side):
    """Bearing to target, or to a tangent of the nearest disk blocking that line.

    side: +1 passes blocking disks on the left (counter-clockwise), -1 on the
    right, 0 picks the tangent closer to the direct bearing. Pure geometry
    of the frozen scene at at_s; used only to order/propose witness controls.
    """
    direct = math.atan2(target.y - pose.y, target.x - pose.x)
    inflate = request.envelope.footprint.reach_m + request.clearance_m + 0.1
    best, best_along = None, math.inf
    ux, uy = math.cos(direct), math.sin(direct)
    tx, ty = target.x, target.y
    for circle in _snapshot(request, at_s):
        reach = circle.radius + inflate
        if segment_circle_min_distance(pose.x, pose.y, tx, ty, circle) >= reach:
            continue
        along = (circle.x - pose.x) * ux + (circle.y - pose.y) * uy
        if along < best_along:
            best, best_along = circle, along
    if best is None:
        return direct
    dx, dy = best.x - pose.x, best.y - pose.y
    distance = math.hypot(dx, dy)
    reach = best.radius + inflate
    centre = math.atan2(dy, dx)
    offset = math.pi / 2.0 if distance <= reach else math.asin(reach / distance)
    left, right = centre + offset, centre - offset
    if side > 0:
        return left
    if side < 0:
        return right
    return left if abs(wrap(left - direct)) <= abs(wrap(right - direct)) else right


def _guided_control(request, node, duration, side=0):
    state = node.trajectory[-1]
    pose = state.pose
    env = request.envelope
    target = _target(request, node)
    distance = math.hypot(pose.x - target.x, pose.y - target.y)
    at_terminal = node.waypoint == len(request.waypoints)
    if at_terminal and distance <= target.radius and request.final_heading_rad is not None:
        heading = request.final_heading_rad
    else:
        heading = _steer_heading(request, pose, target, state.time_s, side)
    error = wrap(heading - pose.yaw)
    yaw_rate = min(max(error / duration, -env.max_yaw_rate_rps), env.max_yaw_rate_rps)
    desired_speed = env.max_speed_mps * max(0.0, math.cos(error))
    if at_terminal and distance <= target.radius * 0.5:
        desired_speed = 0.0
    if at_terminal and (request.final_speed_mps is not None or request.dwell_s > 0):
        final_speed = request.final_speed_mps or 0.0
        stopping_distance = max(0.0, distance - target.radius * 0.5)
        desired_speed = min(desired_speed, math.sqrt(final_speed ** 2 + 2.0 * env.max_decel_mps2 * stopping_distance))
    accel = (desired_speed - pose.speed) / duration
    return _bounded_accel(request, pose, accel, duration), yaw_rate


def _controls(request, node, duration):
    pose = node.trajectory[-1].pose
    env = request.envelope
    emitted = set()
    for side in (0, 1, -1):
        guided = _guided_control(request, node, duration, side)
        if guided not in emitted:
            emitted.add(guided)
            yield guided
    for desired in (env.max_accel_mps2, 0.0, -pose.speed / duration, -env.max_decel_mps2):
        accel = _bounded_accel(request, pose, desired, duration)
        for yaw_rate in (0.0, env.max_yaw_rate_rps, -env.max_yaw_rate_rps,
                         0.5 * env.max_yaw_rate_rps, -0.5 * env.max_yaw_rate_rps):
            control = (accel, yaw_rate)
            if control not in emitted:
                emitted.add(control)
                yield control


def _score(request, node):
    pose = node.trajectory[-1].pose
    target = _target(request, node)
    distance = max(0.0, math.hypot(target.x - pose.x, target.y - pose.y) - target.radius * 0.5)
    targets = (*request.waypoints[node.waypoint:], request.goal)
    for start, end in zip(targets, targets[1:]):
        distance += math.hypot(end.x - start.x, end.y - start.y)
    heading_error = abs(wrap(math.atan2(target.y - pose.y, target.x - pose.x) - pose.yaw))
    if _within(target, pose) and node.waypoint == len(request.waypoints):
        heading_error = 0.0 if request.final_heading_rad is None else abs(wrap(request.final_heading_rad - pose.yaw))
    risk_penalty = 0.0
    # The score only orders the search; it never enters replay acceptance.
    for obj in request.encounter.intruders:
        x, y = obj.position(node.trajectory[-1].time_s)
        gap = max(0.01, math.hypot(pose.x - x, pose.y - y) - obj.radius_m)
        risk_penalty += 0.05 / gap
    return distance + 0.2 * heading_error - 0.2 * node.dwell + risk_penalty


def _bucket(request, node):
    pose = node.trajectory[-1].pose
    scale = max(0.1, min(request.envelope.length_m, request.envelope.beam_m) * 0.25)
    return (round(pose.x / scale), round(pose.y / scale), round(pose.yaw / 0.2),
            round(pose.speed / 0.15), node.waypoint, round(node.dwell / request.dt_s))


# Deterministic manoeuvre library tried first, in this order. Each entry:
# (turn_s, turn_side, hold_mode, hold_s, steer_side). Phases run from the
# release time: optional max-rate turn at constant speed, then optional
# "stop" (brake to rest and hold) or "back" (full astern) phase, then guided
# tangent steering to the remaining gates/goal.
LIBRARY = tuple(
    [(0.0, 0, "none", 0.0, side) for side in (0, 1, -1)]
    + [(0.0, 0, mode, hold, side) for mode in ("stop", "back") for hold in (1.0, 2.0, 4.0, 8.0, 16.0)
       for side in (0, 1, -1)]
    + [(turn, tside, mode, hold, tside) for turn in (1.0, 2.0, 4.0) for tside in (1, -1)
       for mode, hold in (("none", 0.0), ("stop", 2.0), ("stop", 6.0), ("back", 3.0))]
)


def _library_control(request, node, origin, entry, duration):
    turn_s, turn_side, mode, hold_s, steer_side = entry
    old = node.trajectory[-1]
    env = request.envelope
    elapsed = old.time_s - origin
    if elapsed < turn_s - 1e-9:
        return _bounded_accel(request, old.pose, 0.0, duration), turn_side * env.max_yaw_rate_rps
    if elapsed < turn_s + hold_s - 1e-9:
        if mode == "stop":
            return _bounded_accel(request, old.pose, -old.pose.speed / duration, duration), 0.0
        if mode == "back":
            return _bounded_accel(request, old.pose, -env.max_decel_mps2, duration), 0.0
    return _guided_control(request, node, duration, steer_side)


def find_witness(request):
    """Return a replay-checked witness or a counted finite-search rejection.

    After the frozen prefix the fixed manoeuvre LIBRARY is tried in order,
    then a deterministic beam of bounded control samples. No random seed,
    controller identity, observed vessel state, or achieved progress enters.
    """
    _validate_request(request)
    start, last = _prefix_node(request)
    if start is None:
        reason = "initial_geometry_unsafe" if len(last.trajectory) == 1 else "nominal_prefix_unsafe"
        return WitnessResult(False, reason, min_clearance_m=last.clearance)
    if completion_time(request) > request.horizon_s + 1e-9:
        return WitnessResult(False, "deadline_precedes_encounter")
    sampled = rejected = 0
    origin = start.trajectory[-1].time_s
    for entry in LIBRARY:
        node = start
        while node.trajectory[-1].time_s < request.horizon_s - 1e-9:
            if _done(request, node):
                break
            duration = _step_duration(request, node.trajectory[-1].time_s)
            accel, yaw_rate = _library_control(request, node, origin, entry, duration)
            sampled += 1
            next_node = _advance(request, node, accel, yaw_rate, duration)
            if next_node is None:
                rejected += 1
                break
            node = next_node
        if _done(request, node):
            checked = replay_witness(request, node.trajectory)
            if checked.accepted:
                return WitnessResult(True, "finite_library_witness", checked.trajectory,
                                     checked.min_clearance_m, sampled, rejected)
    beam = [start]
    while beam and beam[0].trajectory[-1].time_s < request.horizon_s - 1e-9:
        next_nodes = []
        for node in beam:
            if _done(request, node):
                checked = replay_witness(request, node.trajectory)
                if checked.accepted:
                    return WitnessResult(True, "finite_beam_witness", checked.trajectory,
                                         checked.min_clearance_m, sampled, rejected)
            duration = _step_duration(request, node.trajectory[-1].time_s)
            for accel, yaw_rate in _controls(request, node, duration):
                sampled += 1
                candidate = _advance(request, node, accel, yaw_rate, duration)
                if candidate is None:
                    rejected += 1
                else:
                    next_nodes.append(candidate)
        next_nodes.sort(key=lambda item: _score(request, item))
        buckets = set()
        beam = []
        for candidate in next_nodes:
            bucket = _bucket(request, candidate)
            if bucket not in buckets:
                buckets.add(bucket)
                beam.append(candidate)
                if len(beam) == request.beam_width:
                    break
    for node in beam:
        if _done(request, node):
            checked = replay_witness(request, node.trajectory)
            if checked.accepted:
                return WitnessResult(True, "finite_beam_witness", checked.trajectory,
                                     checked.min_clearance_m, sampled, rejected)
    return WitnessResult(False, "no_witness_in_frozen_search", sampled_states=sampled,
                         rejected_states=rejected, search_exhausted=True)


def replay_witness(request, trajectory):
    """Independently validate prefix, controls, swept geometry, gates, dwell, deadline.

    Externally proposed candidates are accepted only through this validator.
    """
    _validate_request(request)
    trajectory = tuple(trajectory)
    prefix = _prefix(request)
    if len(trajectory) < len(prefix) or trajectory[:len(prefix)] != prefix:
        return WitnessResult(False, "nominal_prefix_changed")
    start, last = _prefix_node(request)
    if start is None:
        return WitnessResult(False, "nominal_prefix_unsafe", min_clearance_m=last.clearance)
    node = start
    env = request.envelope
    for supplied in trajectory[len(prefix):]:
        old = node.trajectory[-1]
        duration = supplied.time_s - old.time_s
        values = (supplied.time_s, supplied.accel_mps2, supplied.yaw_rate_rps,
                  supplied.pose.x, supplied.pose.y, supplied.pose.yaw, supplied.pose.speed)
        if not all(math.isfinite(value) for value in values) or duration <= 0 or duration > request.dt_s + 1e-9:
            return WitnessResult(False, "invalid_time_or_nonfinite_state")
        if supplied.time_s > request.horizon_s + 1e-9:
            return WitnessResult(False, "deadline_exceeded")
        if not _speed_ok(env, supplied.pose.speed) or not _speed_ok(env, old.pose.speed + supplied.accel_mps2 * duration):
            return WitnessResult(False, "speed_outside_envelope")
        if not _control_ok(env, old.pose.speed, supplied.accel_mps2, supplied.yaw_rate_rps, duration):
            return WitnessResult(False, "control_outside_envelope")
        expected = integrate(old.pose, supplied.accel_mps2, supplied.yaw_rate_rps, duration)
        if max(abs(expected.x - supplied.pose.x), abs(expected.y - supplied.pose.y),
               abs(wrap(expected.yaw - supplied.pose.yaw)), abs(expected.speed - supplied.pose.speed)) > 1e-7:
            return WitnessResult(False, "kinematic_rollout_mismatch")
        updated = _advance(request, node, supplied.accel_mps2, supplied.yaw_rate_rps, duration)
        if updated is None:
            return WitnessResult(False, "swept_clearance_not_certified")
        node = updated
    if not _done(request, node):
        return WitnessResult(False, "task_completion_not_certified", min_clearance_m=node.clearance)
    return WitnessResult(True, "replay_validated", trajectory, node.clearance,
                         sampled_states=len(trajectory) - 1)
