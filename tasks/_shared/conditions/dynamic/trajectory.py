"""Controller-independent encounter generation from immutable episode keys.

Construction (C7 STRATEGY.md section 5, "vary reveal distance only"):
1. A frozen nominal plan and a frozen conflict time t_c fix where the ownship
   nominally is when first contact would occur.
2. One constant-velocity intruder is placed so that its disk first touches the
   swept ownship footprint exactly at t_c, aimed at the footprint centre.
   Its world trajectory depends on the key, task spec and vessel, NOT on dose.
3. Dose only sets the reveal (visibility onset) time:
   reveal_s = t_c - ratio(dose) * t_avoid, so TTC at reveal equals the dose.
The whole D1 window [t_c - 4 t_avoid, t_c] must lie on one unforced (straight,
constant-speed) leg of the plan, which makes linear TTC exact for every dose.
No random draws occur here; the only key-dependent choices are balanced strata.
"""

from dataclasses import replace
import hashlib
import json
import math

from .geometry import rounded_box_entry, segment_circle_min_distance, to_world, ttc_linear
from .model import (DOSE_TTC_RATIOS, PATH_TYPES, REVEAL_KINDS, UNFORCED_TTC_TOLERANCE_S, Encounter, Intruder,
                    check_number)

GENERATOR_VERSION = "fixed_path_reveal_dose_v2"


class PlacementError(ValueError):
    """A candidate cannot be constructed; admission counts it as a rejection."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def key_digest(key):
    payload = json.dumps(["d7_episode_key_v1", key.eval_seed, key.env, key.episode,
                          key.hazard_group], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def strata(key, sizes):
    """Exactly balanced deterministic strata for a (env, episode) grid.

    s = (env + episode) mod prod(sizes), decomposed mixed-radix. In a batch of
    E envs x K episodes with E a multiple of prod(sizes), every joint stratum
    receives the same count. No hashing: balance is exact, not statistical.
    """
    total = 1
    for size in sizes:
        if type(size) is not int or size < 1:
            raise ValueError("stratum sizes must be positive integers")
        total *= size
    value = (key.env + key.episode) % total
    out = []
    for size in sizes:
        out.append(value % size)
        value //= size
    return tuple(out)


def relative_heading(kind, side, approach_angle_rad):
    if kind == "head_on":
        return math.pi
    if kind == "overtaking":
        return 0.0
    return side * approach_angle_rad


def generate_encounter(key, task, dose, envelope, plan, conflict_time_s, duration_s, *,
                       kind="crossing", side=1, approach_angle_rad=math.pi / 2.0,
                       intruder_speed_mps=0.5, intruder_size_beams=0.5,
                       reveal_kind="detection_horizon", min_reveal_s=0.0,
                       aim_offset_fraction=0.0, static_obstacles=(), static_margin_m=0.0,
                       first_dose=1, extra_metadata=()):
    """Return the Encounter for one key and dose; raise PlacementError if unplaceable.

    intruder_size_beams is the DIAMETER in vessel beams. side is +1 (intruder
    velocity to port of the ownship heading) or -1. aim_offset_fraction in
    (-1, 1) shifts the relative approach line sideways by that fraction of the
    swept footprint's half-width across the line, so the object grazes rather
    than bisects the hull while first contact stays exactly at t_c.

    G23e: first_dose > 1 (station keeping's approach, sk_approach.py) makes the
    unforced window that of the lowest APPLICABLE tier, [t_c - ratio(first_dose)
    t_avoid, t_c]; d1_window_start_s stays t_c - 4 t_avoid (informative), and
    first_applicable_dose / dose_window_start_s plus extra_metadata are appended
    to the metadata. With the defaults every value and metadata entry is the
    earlier one.
    """
    if task == "moving_basin":
        raise ValueError("moving_basin uses its native motion-ratio generator")
    if isinstance(dose, bool) or not isinstance(dose, int) or not 0 <= dose <= 4:
        raise ValueError("dose must be an integer in [0, 4]")
    if kind not in PATH_TYPES or reveal_kind not in REVEAL_KINDS:
        raise ValueError("unsupported trajectory type or reveal kind")
    if side not in (-1, 1):
        raise ValueError("side must be +1 or -1")
    for name, value in (("conflict_time_s", conflict_time_s), ("duration_s", duration_s),
                        ("approach_angle_rad", approach_angle_rad),
                        ("intruder_speed_mps", intruder_speed_mps),
                        ("intruder_size_beams", intruder_size_beams),
                        ("min_reveal_s", min_reveal_s)):
        check_number(name, value)
    check_number("aim_offset_fraction", aim_offset_fraction)
    if intruder_speed_mps < 0.0 or intruder_size_beams <= 0.0 or min_reveal_s < 0.0:
        raise ValueError("intruder speed/size and minimum reveal must be valid")
    if not -1.0 < aim_offset_fraction < 1.0:
        raise ValueError("aim_offset_fraction must lie strictly inside (-1, 1)")
    if first_dose not in (1, 2, 3, 4) or isinstance(first_dose, bool) or (0 < dose < first_dose):
        raise ValueError("first_dose must be a tier 1..4 not above a nonzero dose")
    t_avoid = envelope.t_avoid_s
    widest = DOSE_TTC_RATIOS[1] * t_avoid
    first_reveal = conflict_time_s - widest
    d1_start = first_reveal
    if first_dose != 1:                                  # G23e: window of the lowest applicable tier
        first_reveal = conflict_time_s - DOSE_TTC_RATIOS[first_dose] * t_avoid
    base = (("generator_version", GENERATOR_VERSION),
            ("key_sha256", key_digest(key)),
            ("vessel_id", envelope.vessel_id),
            ("envelope_provenance", envelope.provenance),
            ("t_avoid_method", envelope.t_avoid_method),
            ("conflict_time_s", float(conflict_time_s)),
            ("d1_window_start_s", float(d1_start)),
            ("path_type", kind), ("side", int(side)),
            ("intruder_speed_mps", float(intruder_speed_mps)),
            ("intruder_size_beams", float(intruder_size_beams)),
            ("size_definition", "diameter_in_vessel_beams"),
            ("aim_offset_fraction", float(aim_offset_fraction)),
            ("reveal_kind", reveal_kind))
    if first_dose != 1 or extra_metadata:                # G23e (station keeping only)
        base = base + (("first_applicable_dose", int(first_dose)),
                       ("dose_window_start_s", float(first_reveal))) + tuple(extra_metadata)
    if conflict_time_s > duration_s or conflict_time_s > plan.end_s + 1e-9:
        raise PlacementError("conflict_after_deadline_or_plan_end")
    if first_reveal < min_reveal_s - 1e-9:
        raise PlacementError("d1_reveal_precedes_allowed_window")
    if not plan.is_constant_velocity(first_reveal, conflict_time_s):
        raise PlacementError("encounter_window_not_on_unforced_nominal_leg")
    if dose == 0:
        return Encounter(key, task, 0, t_avoid, None, (), float(duration_s),
                         base + (("dose_axis", "none"),))
    own = plan.pose_at(conflict_time_s)
    heading = relative_heading(kind, side, approach_angle_rad)
    local_vx = intruder_speed_mps * math.cos(heading)
    local_vy = intruder_speed_mps * math.sin(heading)
    rel_vx, rel_vy = local_vx - own.speed, local_vy
    rel_speed = math.hypot(rel_vx, rel_vy)
    if rel_speed < 1e-9:
        raise PlacementError("no_relative_closing_motion")
    radius = 0.5 * intruder_size_beams * envelope.beam_m
    footprint = envelope.footprint
    cx, cy = footprint.center
    hx, hy = footprint.half_extents
    nx, ny = rel_vx / rel_speed, rel_vy / rel_speed
    # Support of the rounded box across the approach line, then a declared offset.
    half_width = hx * abs(ny) + hy * abs(nx) + radius
    lateral = aim_offset_fraction * half_width
    px, py = -ny * lateral, nx * lateral
    back = footprint.reach_m + math.hypot(cx, cy) + radius + abs(lateral) + 1.0
    entry = rounded_box_entry(px - nx * back, py - ny * back, nx, ny, hx, hy, radius)
    if not math.isfinite(entry):
        raise PlacementError("aim_offset_misses_footprint")
    qx = cx + px - nx * back + nx * entry
    qy = cy + py - ny * back + ny * entry
    contact_x, contact_y = to_world(qx, qy, own)
    c, s = math.cos(own.yaw), math.sin(own.yaw)
    world_vx, world_vy = c * local_vx - s * local_vy, s * local_vx + c * local_vy
    ratio = DOSE_TTC_RATIOS[dose]
    target_ttc = ratio * t_avoid
    reveal_s = conflict_time_s - target_ttc
    intruder = Intruder(kind, contact_x - world_vx * conflict_time_s,
                        contact_y - world_vy * conflict_time_s, world_vx, world_vy,
                        radius, reveal_s, reveal_kind)
    if static_obstacles:
        # Dose-invariant: the core is the D1 window, the same at every dose.
        intruder = fit_motion_window(intruder, static_obstacles, first_reveal,
                                     conflict_time_s, duration_s, static_margin_m)
    reference = plan.pose_at(reveal_s)
    actual = ttc_linear(reference, footprint, intruder, at_s=reveal_s)
    # G7 item B: an exactly unforced window keeps the exact check; a window with sub-tolerance
    # commands (model.is_unforced_command) is allowed the declared timing bound on top.
    exact_window = all(s.accel_mps2 == 0.0 and s.yaw_rate_rps == 0.0
                       for b, s in zip(plan.states, plan.states[1:])
                       if s.time_s > first_reveal + 1e-12 and b.time_s < conflict_time_s - 1e-12)
    slack = 0.0 if exact_window else UNFORCED_TTC_TOLERANCE_S
    if not math.isclose(actual, target_ttc, rel_tol=1e-9, abs_tol=1e-9 + slack):
        raise ArithmeticError("first-contact construction failed its TTC invariant")
    return Encounter(key, task, dose, t_avoid, target_ttc, (intruder,), float(duration_s),
                     base + (("dose_axis", "TTC_over_t_avoid"), ("ttc_ratio", ratio),
                             ("relative_heading_rad", heading),
                             ("closing_speed_mps", rel_speed),
                             ("reference_x_m", reference.x), ("reference_y_m", reference.y),
                             ("reference_yaw_rad", reference.yaw),
                             ("reference_speed_mps", reference.speed)))


def _forbidden_interval(intruder, circle, margin_m):
    """Times t at which the MOVING-LINE disk overlaps an inflated static disk."""
    reach = circle.radius + intruder.radius_m + margin_m
    dx, dy = intruder.x0 - circle.x, intruder.y0 - circle.y
    a = intruder.vx * intruder.vx + intruder.vy * intruder.vy
    b = 2.0 * (dx * intruder.vx + dy * intruder.vy)
    c = dx * dx + dy * dy - reach * reach
    if a == 0.0:
        return (-math.inf, math.inf) if c <= 0.0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    return ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))


def fit_motion_window(intruder, static_obstacles, core_start_s, core_end_s, duration_s,
                      margin_m=0.0):
    """Rest-move-rest intruder that never overlaps static geometry.

    The moving interval is the MAXIMAL interval inside [0, duration_s] that
    contains the core [core_start_s, core_end_s] and keeps the disk at least
    margin_m from every static disk; outside it the object rests at the
    interval endpoint. Raises PlacementError if the core itself collides.
    Exact (closed-form quadratic per static disk), no sampling.
    """
    start, stop = 0.0, math.inf
    for circle in static_obstacles:
        window = _forbidden_interval(intruder, circle, margin_m)
        if window is None:
            continue
        low, high = window
        if high < core_start_s:
            start = max(start, high)
        elif low > core_end_s:
            stop = min(stop, low)
        else:
            raise PlacementError("intruder_path_intersects_static_geometry")
    if start > core_start_s or stop < core_end_s:  # unreachable by construction; kept as a guard
        raise PlacementError("intruder_motion_window_too_short")
    stop_s = None if stop >= duration_s else stop
    return replace(intruder, start_s=start, stop_s=stop_s)


def intruder_path_signature(encounter):
    """Dose-invariant physical identity of the intruder (for pairing tests/stamps)."""
    if not encounter.intruders:
        return None
    obj = encounter.intruders[0]
    return (obj.path_type, obj.x0, obj.y0, obj.vx, obj.vy, obj.radius_m, obj.start_s, obj.stop_s)


def intruder_static_clearance(intruder, static_obstacles, start_s, end_s):
    """Minimum surface gap between the intruder's swept path on [start_s, end_s] and static disks."""
    x0, y0 = intruder.position(start_s)
    x1, y1 = intruder.position(end_s)
    return min((segment_circle_min_distance(x0, y0, x1, y1, circle) - circle.radius
                - intruder.radius_m for circle in static_obstacles), default=math.inf)
