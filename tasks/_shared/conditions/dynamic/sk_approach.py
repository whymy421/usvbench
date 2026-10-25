"""G23e: station-keeping D7 encounter DURING THE APPROACH to the hold circle (owner decision 2026-10-06,
CONTEXT_R6 section 3; replaces the retired design that revealed the intruder at the start of the scoring window
and met the own ship while it held station).

Design (sk_approach_crossing_v1)
  * Reference route: the frozen nominal plan of station keeping turns in place until it faces the hold centre,
    then runs the straight start -> centre line at the reference speed and stops inside the circle
    (reference.build_transit_plan(align_first=True); adapters._nominal_plan). D_start = |centre - start|.
  * Meeting point M: the point of the reference route whose straight-line distance to the hold centre equals
    SK_MEETING_FRACTION x D_start on the final (inbound) approach. On the straight route M = start +
    (1 - SK_MEETING_FRACTION) (centre - start) exactly (BlueBoat 4.5 m, WAM-V 20.15 m from the centre). The
    designed first contact t_c is the nominal time at which the own ship's pose origin (the COM) is at M; the
    intruder is placed by trajectory.generate_encounter so that its disk first touches the own footprint at t_c.
  * Crossing at 90 deg only (path_type crossing, relative heading +-pi/2), aim at the footprint centre; side
    +1 (intruder velocity to port) or -1 by the scene key: strata((env + episode) mod 2), exact 50/50 on every
    grid with an even env count. Radius and speed: the suite defaults (intruder_size_beams 0.5 = a
    diameter of half a beam, i.e. radius beam / 4; intruder_speed_mps 0.5).
  * Reveal: the default runtime rule (d7_reveal_sync_v2 in layer A, d7_reveal_sync_v3 after G23e layer B; TTC =
    dose x t_avoid; the route corridor predictor sees the approach as a transit route, plan kind "approach").
  * Tier applicability: a tier is applicable when its nominal reveal t_c - ratio x t_avoid lies on the straight
    cruise-speed part of the approach (the plan is unforced on [t_c - ratio t_avoid, t_c] and the reveal is not
    before t = 0); the lowest applicable tier is first_applicable_dose. BlueBoat: D1 would reveal 4 x 1.645 =
    6.58 m before M, i.e. 1.08 m behind the start: NOT APPLICABLE BY DESIGN (D2..D4 remain); WAM-V: D1..D4.
    A not-applicable tier is neither admitted nor rejected (NotApplicableByDesign, recorded separately from
    not_admitted in the pool and in the certificate header).
  * Hold-circle exclusion: the intruder's fixed path (the whole straight line, conservative) must keep at least
    hold radius + intruder radius + half beam from the hold centre; otherwise the key is rejected with
    SK_HOLD_REJECTION (no re-orientation: only 90 deg is allowed and the side flip does not move the line).
  * The SK scoring contract (last 60 s inside the circle) is unchanged; t_c lies before the scoring window.
"""

import math

from .model import DOSE_TTC_RATIOS, is_unforced_command
from .trajectory import PlacementError

SK_DESIGN = "sk_approach_crossing_v1"
SK_SITE_RULE = "approach_point"
SK_PLAN_KIND = "approach"            # timing kind of the runtime rule (release.TIMED_PLAN_KINDS)
SK_MEETING_FRACTION = 0.45           # owner decision 2026-10-06
SK_CROSSING_ANGLE_RAD = math.pi / 2.0
SK_SIDE_RULE = "strata((env + episode) mod 2): +1 port / -1 starboard"
SK_NOT_APPLICABLE = "not_applicable_by_design"
SK_HOLD_REJECTION = "sk_intruder_path_enters_hold_circle"
SK_MEETING_TOL_M = 1.0e-6            # own pose at t_c vs the geometric meeting point (straight route)
SK_TEXT = ("station keeping (G23e, sk_approach_crossing_v1): the intruder meets the own ship during the approach "
           "to the hold circle, never during the hold: meeting point on the straight start-to-centre approach at "
           "0.45 x start-to-centre distance from the hold centre (own ship there at the designed first contact), "
           "90 deg crossing from port or starboard by scene key, revealed by the runtime rule (TTC = dose x "
           "t_avoid); tiers whose nominal reveal would precede the straight cruise-speed approach are not "
           "applicable by design; the intruder line keeps hold radius + intruder radius + half beam from the "
           "hold centre. Supersedes the 'hold tasks: pinned schedule' clause of the runtime_rule text for "
           "station keeping (rule constants unchanged).")


class NotApplicableByDesign(PlacementError):
    """A dose tier the encounter design excludes: neither admitted nor rejected (never a not_admitted reason)."""

    def __init__(self, reason=SK_NOT_APPLICABLE):
        super().__init__(reason)


def meeting(context, plan):
    """(t_c, (mx, my), d_start, reach, offset) of the approach meeting point (module docstring).

    t_c: first time the nominal pose is at straight-line distance reach = SK_MEETING_FRACTION * d_start from the
    hold centre (inbound), solved exactly on its constant-velocity plan interval; offset: distance between the
    own pose at t_c and the geometric point M (0 up to rounding on the straight route)."""
    hold, start = context.hold_circle, context.initial_pose
    if hold is None:
        raise ValueError("station keeping needs its hold circle")
    d_start = math.hypot(hold.x - start.x, hold.y - start.y)
    if not d_start > hold.radius:
        raise PlacementError("sk_start_inside_hold_circle")
    reach = SK_MEETING_FRACTION * d_start
    mx = hold.x + SK_MEETING_FRACTION * (start.x - hold.x)
    my = hold.y + SK_MEETING_FRACTION * (start.y - hold.y)
    for before, after in zip(plan.states, plan.states[1:]):
        d0 = math.hypot(before.pose.x - hold.x, before.pose.y - hold.y)
        d1 = math.hypot(after.pose.x - hold.x, after.pose.y - hold.y)
        if not d0 > reach >= d1:
            continue
        if not is_unforced_command(after.accel_mps2, after.yaw_rate_rps) or before.pose.speed <= 0.0:
            raise PlacementError("sk_meeting_point_not_on_unforced_approach_leg")
        p = before.pose
        ux, uy = math.cos(p.yaw), math.sin(p.yaw)
        ex, ey = p.x - hold.x, p.y - hold.y
        b = ex * ux + ey * uy
        disc = b * b - (ex * ex + ey * ey - reach * reach)
        s = -b - math.sqrt(max(0.0, disc))            # inbound root: distance along the heading
        t_c = before.time_s + max(0.0, s) / p.speed
        own = plan.pose_at(t_c)
        offset = math.hypot(own.x - mx, own.y - my)
        if offset > SK_MEETING_TOL_M:
            raise PlacementError("sk_reference_route_misses_meeting_point")
        return t_c, (mx, my), d_start, reach, offset
    raise PlacementError("sk_nominal_never_reaches_meeting_distance")


def first_applicable_dose(envelope, plan, t_c):
    """Lowest tier whose nominal reveal lies on the straight cruise-speed approach (module docstring)."""
    for dose in (1, 2, 3, 4):
        start = t_c - DOSE_TTC_RATIOS[dose] * envelope.t_avoid_s
        if start >= 0.0 and plan.is_constant_velocity(start, t_c):
            return dose
    raise PlacementError("sk_no_tier_fits_the_approach")


def encounter_metadata(t_c, point, d_start, reach, offset):
    """SK metadata appended to the encounter (generate_encounter extra_metadata)."""
    return (("sk_design", SK_DESIGN), ("sk_meeting_fraction", SK_MEETING_FRACTION),
            ("sk_start_to_centre_m", float(d_start)), ("sk_meeting_x_m", float(point[0])),
            ("sk_meeting_y_m", float(point[1])), ("sk_meeting_distance_to_centre_m", float(reach)),
            ("sk_meeting_offset_m", float(offset)), ("sk_crossing_angle_rad", SK_CROSSING_ANGLE_RAD),
            ("sk_side_rule", SK_SIDE_RULE))


def hold_geometry(hold, envelope, intruder):
    """(line distance, required distance) of the intruder's fixed straight path against the hold circle."""
    vx, vy = intruder.vx, intruder.vy
    speed = math.hypot(vx, vy)
    if speed <= 0.0:
        distance = math.hypot(intruder.x0 - hold.x, intruder.y0 - hold.y)
    else:
        distance = abs((hold.x - intruder.x0) * vy - (hold.y - intruder.y0) * vx) / speed
    return distance, hold.radius + intruder.radius_m + 0.5 * envelope.beam_m


def check_hold_exclusion(context, encounter):
    """The encounter with the hold-clearance metadata appended, or PlacementError(SK_HOLD_REJECTION)."""
    from dataclasses import replace
    distance, required = hold_geometry(context.hold_circle, context.envelope, encounter.intruders[0])
    if distance < required:
        raise PlacementError(SK_HOLD_REJECTION)
    extra = (("sk_hold_line_distance_m", float(distance)), ("sk_hold_required_m", float(required)),
             ("sk_hold_clearance_m", float(distance - required)))
    return replace(encounter, metadata=encounter.metadata + extra)


def validate_candidate(candidate):
    """Admission invariants of an SK candidate at a nonzero dose (Candidate.validate; module docstring)."""
    request, encounter = candidate.request, candidate.request.encounter
    if encounter.meta("sk_design") != SK_DESIGN:
        raise ValueError("station keeping candidate predates the approach design %s (re-admit the pool)" % SK_DESIGN)
    if encounter.meta("path_type") != "crossing" or not math.isclose(
            abs(encounter.meta("relative_heading_rad", 0.0)), SK_CROSSING_ANGLE_RAD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("station keeping intruder must cross at 90 deg")
    first = encounter.meta("first_applicable_dose")
    if type(first) is not int or not 1 <= first <= encounter.dose:
        raise ValueError("station keeping dose D%d is not applicable by design" % encounter.dose)
    if not encounter.meta("conflict_time_s") < candidate.settled_phase_start_s:
        raise ValueError("station keeping encounter must happen before the scoring window (approach only)")
    if request.hold_circle is None:
        raise ValueError("station keeping candidate lost its hold circle")
    distance, required = hold_geometry(request.hold_circle, request.envelope, encounter.intruders[0])
    if distance < required:
        raise ValueError("station keeping intruder path enters the hold circle margin")


def stamp_block(encounter):
    """JSON-native SK block of the runtime release stamp (None for every other task)."""
    if encounter.meta("sk_design") is None:
        return None
    first = encounter.meta("first_applicable_dose")
    return {"design": SK_DESIGN, "text": SK_TEXT,
            "meeting_point_m": [encounter.meta("sk_meeting_x_m"), encounter.meta("sk_meeting_y_m")],
            "meeting_fraction": encounter.meta("sk_meeting_fraction"),
            "start_to_centre_m": encounter.meta("sk_start_to_centre_m"),
            "meeting_distance_to_centre_m": encounter.meta("sk_meeting_distance_to_centre_m"),
            "crossing_angle_rad": encounter.meta("sk_crossing_angle_rad"), "side": encounter.meta("side"),
            "first_applicable_dose": first, "not_applicable_doses": list(range(1, first)),
            "hold_line_distance_m": encounter.meta("sk_hold_line_distance_m"),
            "hold_required_m": encounter.meta("sk_hold_required_m"),
            "hold_clearance_m": encounter.meta("sk_hold_clearance_m")}
