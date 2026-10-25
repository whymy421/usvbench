"""Per-vessel D7 dose table (pure Python): t_avoid, TTC per dose, detection ranges.

Package version of D7's tools/d7_dose_table.py, so that the published numbers
(E4 DOSE_TABLE_E4.md) come from tested code. Construction (identical to D7's
tool): a straight nominal leg along +x at the envelope's reference speed,
conflict time t_c = 200 s, horizon 400 s, default intruder (0.5 m/s; drifting
0.1 m/s; diameter 0.5 own beam). The detection range of a dose is the surface
range from the ownship COM to the intruder at the nominal reveal time, i.e.
the range a detection_horizon encounter requires from the task's sensor.

Because generation requires the whole D1 window to lie on one unforced
(straight, constant-speed) leg (trajectory.py), these ranges depend only on the
vessel, the speeds, the dose, the path type and the (side, aim offset)
choice, not on the task layout. Station keeping is the stationary-ownship
case. REQUIRED_PATHS lists the (ownship motion, path type, aim offsets) that
the twelve task profiles can generate (adapters.PROFILES and the harbour
stage kinds); the sensor-fit check is made over those, over both sides.
"""

import math

from .model import DOSE_TTC_RATIOS, Circle, EpisodeKey, Pose
from .native import NATIVE_RATIOS
from .reference import build_stationary_plan, build_transit_plan
from .trajectory import generate_encounter
from .visibility import edge_range_and_occlusion

DETECTION_PATH_TYPES = ("crossing", "head_on", "overtaking", "drifting")
T_CONFLICT_S = 200.0
HORIZON_S = 400.0
# (ownship, path type, aim offsets, tasks that use it); from adapters.PROFILES:
# reach/scatter crossing+drifting, path_following/path_hazard/docking crossing,
# ring/fortress/forced_crossing/iceberg head_on, harbour stages (exit gate
# head_on, transit and berth approach crossing), station keeping stationary
# crossing/head_on/overtaking with aim offsets +-0.5.
REQUIRED_PATHS = (
    ("transit", "crossing", (0.0,), ("reach", "path_following", "docking", "path_hazard", "scatter",
                                    "harbour")),
    ("transit", "drifting", (0.0,), ("reach", "scatter")),
    ("transit", "head_on", (0.0,), ("ring", "fortress", "forced_crossing", "iceberg", "harbour")),
    ("stationary", "crossing", (0.5, -0.5), ("station_keeping",)),
    ("stationary", "head_on", (0.5, -0.5), ("station_keeping",)),
    ("stationary", "overtaking", (0.5, -0.5), ("station_keeping",)),
)


def _plan(envelope, stationary):
    if stationary:
        return build_stationary_plan(Pose(0.0, 0.0, 0.0, 0.0), HORIZON_S)
    return build_transit_plan(Pose(0.0, 0.0, 0.0, envelope.reference_speed_mps), envelope,
                              (Circle(2000.0, 0.0, 1.0),), 0.25, HORIZON_S)


def detection_ranges(envelope, kind, *, stationary=False, side=1, aim_offset=0.0,
                     intruder_speed_mps=0.5, drift_speed_mps=0.1, intruder_size_beams=0.5, plan=None):
    """Surface range (m) at the nominal reveal for D1..D4 (tuple of four floats)."""
    if kind not in DETECTION_PATH_TYPES:
        raise ValueError("unknown path type")
    plan = _plan(envelope, stationary) if plan is None else plan
    task = "station_keeping" if stationary else "reach"
    speed = drift_speed_mps if kind == "drifting" else intruder_speed_mps
    out = []
    for dose in range(1, 5):
        encounter = generate_encounter(EpisodeKey(0, 0, 0, "d7_" + task), task, dose, envelope, plan,
                                       T_CONFLICT_S, HORIZON_S, kind=kind, side=side,
                                       intruder_speed_mps=speed, intruder_size_beams=intruder_size_beams,
                                       aim_offset_fraction=aim_offset)
        obj = encounter.intruders[0]
        edge, _ = edge_range_and_occlusion(plan.pose_at(obj.reveal_s), obj, obj.reveal_s, ())
        out.append(edge)
    return tuple(out)


def vessel_dose_table(envelope, sensor_range_m=30.0, **intruder):
    """Every published number for one vessel envelope, as a plain dict.

    detection["transit"|"stationary"][kind] = {"nominal": D1..D4 at side +1,
    aim 0 (D7's convention), "worst": elementwise max over both sides and the
    task aim offsets of REQUIRED_PATHS (aim 0 when the path is not required)}.
    fits_sensor is True iff every REQUIRED path's worst D1 range (the largest
    range of a ladder) is within sensor_range_m; violations lists the others.
    """
    components = dict(envelope.t_avoid_components)
    plans = {"transit": _plan(envelope, False), "stationary": _plan(envelope, True)}
    required = {(motion, kind): (aims, tasks) for motion, kind, aims, tasks in REQUIRED_PATHS}
    detection, violations = {}, []
    for motion in ("transit", "stationary"):
        detection[motion] = {}
        kinds = DETECTION_PATH_TYPES if motion == "transit" else ("crossing", "head_on", "overtaking")
        for kind in kinds:
            plan = plans[motion]
            nominal = detection_ranges(envelope, kind, stationary=motion == "stationary", plan=plan, **intruder)
            aims, tasks = required.get((motion, kind), ((0.0,), ()))
            rows = [detection_ranges(envelope, kind, stationary=motion == "stationary", side=side,
                                     aim_offset=aim, plan=plan, **intruder)
                    for side in (1, -1) for aim in aims]
            worst = tuple(max(row[i] for row in rows) for i in range(4))
            detection[motion][kind] = {"nominal": nominal, "worst": worst, "required_by": tasks}
            if tasks and worst[0] > sensor_range_m:
                violations.append((motion, kind, worst[0], tasks))
    top = envelope.top_speed_mps
    return {
        "vessel_id": envelope.vessel_id,
        "reference_speed_mps": envelope.reference_speed_mps,
        "t_avoid_s": envelope.t_avoid_s,
        "t_avoid_method": envelope.t_avoid_method,
        "t_avoid_components": components,
        "ttc_s": tuple(ratio * envelope.t_avoid_s for ratio in DOSE_TTC_RATIOS[1:]),
        "intruder_diameter_m": intruder.get("intruder_size_beams", 0.5) * envelope.beam_m,
        "detection_m": detection,
        "sensor_range_m": sensor_range_m,
        "max_required_d1_m": max(detection[m][k]["worst"][0] for m, k, _, _ in REQUIRED_PATHS),
        "fits_sensor": not violations,
        "violations": tuple(violations),
        "native_speeds_mps": tuple(mu * top for mu in NATIVE_RATIOS),
        "top_speed_mps": top,
    }


def head_on_range_closed_form(envelope, dose, intruder_speed_mps=0.5, intruder_size_beams=0.5):
    """Independent closed form for the transit head-on case (used by tests).

    Head-on, aim 0: the intruder centre moves along the line through the
    footprint centre (y_c) at closing speed u + v_i, first touching the bow
    face at x_max + r at t_c. At reveal it is closing * TTC further ahead, so
    the COM-to-surface range is hypot(x_max + r + closing*TTC, y_c) - r.
    """
    radius = 0.5 * intruder_size_beams * envelope.beam_m
    closing = envelope.reference_speed_mps + intruder_speed_mps
    ttc = DOSE_TTC_RATIOS[dose] * envelope.t_avoid_s
    y_centre = envelope.footprint.center[1]
    return math.hypot(envelope.footprint.x_max_m + radius + closing * ttc, y_centre) - radius
