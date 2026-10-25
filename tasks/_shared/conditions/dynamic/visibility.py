"""Reveal = physical detectability through the task's EXISTING range sensor.

Owner rule (2026-09-30): hazards reach controllers only through each task's
existing sensors, never through a privileged channel. Under the v2.5 motion
(pinned_v0) and rule S physical existence is never gated, only detectability
(G23c F11: the runtime rule may still TIME the object, release.py); under the
default rule S2 (d7_reveal_sync_v1, reveal.py) the object is absent (neither
detectable nor collidable) before its reveal. Two declared reveal mechanisms,
both fixed before assignment and both independent of the tested controller:

* detection_horizon: the intruder is detectable by the task's existing ray
  sensor only within detection_range_m of its surface (an object property,
  e.g. small or low-contrast debris), with detection_range_m <= the sensor's
  maximum range. The dose sets detection_range_m so that the FROZEN NOMINAL
  observer first detects it exactly at reveal_s, when its TTC equals
  ratio * t_avoid.
* occluder_exit: a physical static occluder (also present in collision and
  witness geometry) blocks the nominal line of sight until reveal_s.

At runtime visibility is purely physical: range to the object's surface
<= min(sensor max range, detection range) and a clear line of sight. There is
NO time gate: an observer that is ahead of, behind or off the nominal track
detects the object earlier, later or never. That is an exposure OUTCOME,
recorded per episode; visibility itself never moves the object or changes its
parameters (the runtime rule may time the object: release.py, G23c F11).
(The first takeover draft added a "not before reveal_s" gate, which hides an
in-range object and is itself a privileged manipulation; it was removed.)

E4 (CONTEXT_R3 decision 14, PENDING): in the journal every task has the
36-ray, 30 m fan (JOURNAL_RAY_SENSOR below), so reach, path following,
station keeping and docking without berth walls are sensed too and D1..D4
are distinct experiments on all 12 tasks. The rule "no privileged channel"
is unchanged: the hazard still reaches controllers only through that fan.

Line of sight uses the segment from the observer to the nearest surface
point on the centre ray (a point-return proxy for a ray sensor), not full
disk-silhouette visibility; the runtime ray block must use the same proxy or
the encounter must be re-admitted under a silhouette model.
"""

from dataclasses import dataclass
import math

from .model import Circle, Pose, check_number


@dataclass(frozen=True)
class SensorSpec:
    """A task's range sensor (None in a context means: no existing sensor)."""

    kind: str = "ray_fan"
    ray_count: int = 36
    max_range_m: float = 30.0
    source: str = ""

    def __post_init__(self):
        if self.kind != "ray_fan":
            raise ValueError("only the existing analytic ray fan is modelled")
        if type(self.ray_count) is not int or self.ray_count < 1:
            raise ValueError("ray_count must be a positive integer")
        check_number("max_range_m", self.max_range_m, positive=True)


# E4 / CONTEXT_R3 decision 14 (PENDING): every journal task carries the
# standard 36-ray, 30 m range fan -- the same contract as the existing fans of
# hazard_nav (J-base hazard_nav_env_cfg.py:185-186), path_hazard
# (path_hazard_env_cfg.py:185-186), harbour (harbor_mission_env_cfg.py:179-180)
# and docking with berth walls (docking_env_cfg.py:221-222). With
# DynamicSuiteConfig.journal_ray_sensor_all_tasks=True (default) a TaskContext
# without an existing sensor is treated as sensed with this spec.
JOURNAL_RAY_SENSOR = SensorSpec(
    kind="ray_fan", ray_count=36, max_range_m=30.0,
    source="journal 36-ray 30 m range sensor on every task (CONTEXT_R3 decision 14, PENDING)")


@dataclass(frozen=True)
class RevealEvidence:
    mode: str
    reference_at_reveal: Pose
    detection_range_m: float
    occluders: tuple = ()
    tolerance_s: float = 0.001

    def __post_init__(self):
        if self.mode not in ("detection_horizon", "occluder_exit"):
            raise ValueError("reveal mode must be detection_horizon or occluder_exit")
        if not isinstance(self.reference_at_reveal, Pose):
            raise ValueError("reveal evidence requires a frozen nominal Pose")
        check_number("detection_range_m", self.detection_range_m, positive=True)
        check_number("tolerance_s", self.tolerance_s, positive=True)
        if not isinstance(self.occluders, tuple) or any(
                not isinstance(circle, Circle) or circle.radius <= 0.0 for circle in self.occluders):
            raise ValueError("occluders must be an immutable tuple of positive-radius Circles")
        if self.mode == "occluder_exit" and not self.occluders:
            raise ValueError("occluder_exit requires physical occluder geometry")


def _observer_at(reference, offset_s):
    return Pose(reference.x + reference.speed * math.cos(reference.yaw) * offset_s,
                reference.y + reference.speed * math.sin(reference.yaw) * offset_s,
                reference.yaw, reference.speed)


def _segment_circle_clearance(ax, ay, bx, by, circle):
    dx, dy = bx - ax, by - ay
    squared = dx * dx + dy * dy
    fraction = 0.0 if squared == 0.0 else max(0.0, min(1.0, ((circle.x - ax) * dx + (circle.y - ay) * dy) / squared))
    return math.hypot(ax + fraction * dx - circle.x, ay + fraction * dy - circle.y) - circle.radius


def edge_range_and_occlusion(observer, intruder, time_s, occluders):
    """(surface range, line-of-sight clearance to the nearest occluder)."""
    x, y = intruder.position(time_s)
    dx, dy = x - observer.x, y - observer.y
    distance = math.hypot(dx, dy)
    edge = max(0.0, distance - intruder.radius_m)
    fraction = edge / distance if distance > 0.0 else 0.0
    near_x, near_y = observer.x + fraction * dx, observer.y + fraction * dy
    clearance = min((_segment_circle_clearance(observer.x, observer.y, near_x, near_y, c)
                     for c in occluders), default=math.inf)
    return edge, clearance


def detectable(observer, intruder, time_s, evidence, sensor):
    """Physical detectability for an observer pose (None sensor -> never)."""
    if sensor is None or evidence is None:
        return False
    edge, clearance = edge_range_and_occlusion(observer, intruder, time_s, evidence.occluders)
    return edge <= min(sensor.max_range_m, evidence.detection_range_m) and clearance > 0.0


def validate_reveal(encounter, evidence, sensor):
    """Raise ValueError unless the named nominal reveal event is geometrically true."""
    if encounter.task == "moving_basin":
        return
    if encounter.dose == 0:
        if encounter.intruders:
            raise ValueError("zero generic dose must have no intruder")
        return
    if sensor is None:
        raise ValueError("task_has_no_existing_range_sensor")
    if not isinstance(sensor, SensorSpec):
        raise ValueError("sensor must be a SensorSpec")
    if not isinstance(evidence, RevealEvidence) or not encounter.intruders:
        raise ValueError("nonzero encounter requires an intruder and RevealEvidence")
    reveal = encounter.intruders[0].reveal_s
    if any(obj.reveal_s != reveal for obj in encounter.intruders):
        raise ValueError("one nominal reveal reference requires matching reveal times")
    step = evidence.tolerance_s
    if reveal < step or reveal + step > encounter.duration_s:
        raise ValueError("reveal needs a complete before/after bracket within the episode")
    reference = evidence.reference_at_reveal
    for key, value in (("reference_x_m", reference.x), ("reference_y_m", reference.y),
                       ("reference_yaw_rad", reference.yaw),
                       ("reference_speed_mps", reference.speed)):
        stored = encounter.meta(key)
        if stored is None or not math.isclose(stored, value, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("reveal evidence disagrees with the generated nominal reference")
    if evidence.detection_range_m > sensor.max_range_m + 1e-9:
        raise ValueError("detection_range_exceeds_existing_sensor_range")
    before, after = _observer_at(reference, -step), _observer_at(reference, step)
    horizon = evidence.detection_range_m
    for obj in encounter.intruders:
        if obj.reveal_kind != evidence.mode:
            raise ValueError("intruder reveal kind disagrees with geometric evidence")
        range_before, block_before = edge_range_and_occlusion(before, obj, reveal - step, evidence.occluders)
        range_at, block_at = edge_range_and_occlusion(reference, obj, reveal, evidence.occluders)
        range_after, block_after = edge_range_and_occlusion(after, obj, reveal + step, evidence.occluders)
        if evidence.mode == "detection_horizon":
            if abs(range_at - horizon) > 1e-9 * max(1.0, horizon):
                raise ValueError("scheduled reveal is not on the nominal detection horizon")
            if not range_before > horizon or not range_after < horizon:
                raise ValueError("detection-horizon reveal must enter from outside to inside")
            if block_at <= 0.0 or block_after <= 0.0:
                raise ValueError("nominal detection is still occluded")
        else:
            if not block_before <= 0.0 or not block_after > 0.0:
                raise ValueError("occluder reveal must transition from blocked to clear")
            if range_at > min(horizon, sensor.max_range_m) or range_after > min(horizon, sensor.max_range_m):
                raise ValueError("occluder exit must lie inside the detection range")


def nominal_first_detection_ok(plan, encounter, evidence, sensor, sample_dt_s):
    """Sampled check that the nominal observer does NOT detect before reveal_s.

    Samples [0, reveal_s - tolerance] every sample_dt_s (plus the endpoint).
    This is a sampled check, not a proof between samples; sample_dt_s should
    be small against (detection range margin / closing speed).
    """
    check_number("sample_dt_s", sample_dt_s, positive=True)
    if not encounter.intruders:
        return True
    reveal = encounter.intruders[0].reveal_s
    end = reveal - evidence.tolerance_s
    count = max(1, int(math.ceil(end / sample_dt_s)))
    for index in range(count + 1):
        t = min(end, index * sample_dt_s)
        observer = plan.pose_at(t)
        for obj in encounter.intruders:
            if detectable(observer, obj, t, evidence, sensor):
                return False
    return True
