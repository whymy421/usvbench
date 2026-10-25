"""G23c (REV-G23b F4, F8): realized-exposure record of one D7 episode (pure Python).

No certificate stores trajectories, so the adapter keeps, per episode and from the
update samples it already makes (observer pose, intruder position, visibility):

* first_visible_s: first update at which the intruder is detectable (visible flag);
* first_ray_return_s: first update at which the 36-ray fan actually returns it under
  the point-return proxy (visibility.py: the ray nearest to the bearing of the
  intruder centre returns the nearest surface point) and no static disk of the
  candidate geometry is nearer along that ray (static shadow); with G23c F8 every
  family uses this proxy for its D7 slots, so this is the time the controller's
  observation first contains the intruder;
* closest approach of the own ship to the intruder after the intruder MOVES and after
  it is VISIBLE: time, footprint-to-disk surface gap (m, <= 0 is contact), and the
  intruder centre in the own body frame relative to the footprint centre (along-track,
  cross-track, m);
* the first contact after the intruder moves (first update with gap <= 0), the contact
  time the runtime rule designed and the sync error (first contact time, or the time of
  the closest approach when there is no contact, minus that designed time);
* the own ship's passage of the nominal conflict pose (rule S) and the
  unsynchronizable slack max(0, T_rel - t_passage): > 0 means even a release at t = 0
  could not have synchronized the encounter (REV-G23b F6).
All values are sampled at the update times (no interpolation); None when not reached.
"""

import math

from .geometry import point_clearance, to_body

EXPOSURE_SCHEMA = "usvbench.d7_exposure_record/1"


def _ray_circle(ox, oy, ux, uy, circle):
    """Distance along the unit ray (ox, oy) + s (ux, uy), s >= 0, to the circle (0 inside), or inf."""
    dx, dy = ox - circle.x, oy - circle.y
    c = dx * dx + dy * dy - circle.radius * circle.radius
    if c <= 0.0:
        return 0.0
    b = dx * ux + dy * uy
    disc = b * b - c
    if disc < 0.0 or b > 0.0:
        return math.inf
    return -b - math.sqrt(disc)


class ExposureTracker:
    def __init__(self, candidate, state):
        envelope = candidate.request.envelope
        self.footprint = envelope.footprint
        self.radius = candidate.request.encounter.intruders[0].radius_m
        self.statics = tuple(candidate.request.static_obstacles)
        self.sensor = candidate.sensor
        self.state = state
        self.first_visible_s = None
        self.first_ray_return_s = None
        self.after_moving = None        # (gap, time, along, cross)
        self.after_visible = None
        self.passage = None             # (distance to the nominal conflict pose, time)
        self.first_contact_s = None     # first update with gap <= 0 after the intruder moves

    def ray_returns(self, observer, x, y):
        """Point-return proxy on the sensor's fan with static shadowing (see module doc)."""
        if self.sensor is None:
            return False
        dx, dy = x - observer.x, y - observer.y
        distance = math.hypot(dx, dy)
        edge = max(0.0, distance - self.radius)
        if edge > self.sensor.max_range_m:
            return False
        count = int(self.sensor.ray_count)
        step = 2.0 * math.pi / count
        bearing = math.atan2(dy, dx) - observer.yaw
        index = int(math.floor((bearing % (2.0 * math.pi)) / step + 0.5)) % count
        angle = observer.yaw + index * step
        ux, uy = math.cos(angle), math.sin(angle)
        shadow = min((_ray_circle(observer.x, observer.y, ux, uy, c) for c in self.statics), default=math.inf)
        return edge <= shadow

    def update(self, time_s, observer, x, y, visible, moving):
        if observer is None:
            return
        gap = point_clearance(x, y, observer, self.footprint) - self.radius
        bx, by = to_body(x, y, observer)
        cx, cy = self.footprint.center
        record = (gap, time_s, bx - cx, by - cy)
        if moving and (self.after_moving is None or gap < self.after_moving[0]):
            self.after_moving = record
        if moving and gap <= 0.0 and self.first_contact_s is None:
            self.first_contact_s = time_s
        if visible:
            if self.first_visible_s is None:
                self.first_visible_s = time_s
            if self.after_visible is None or gap < self.after_visible[0]:
                self.after_visible = record
            if self.first_ray_return_s is None and self.ray_returns(observer, x, y):
                self.first_ray_return_s = time_s
        pose = self.state.conflict_pose if self.state is not None else None
        if pose is not None:
            d = math.hypot(observer.x - pose.x, observer.y - pose.y)
            if self.passage is None or d < self.passage[0]:
                self.passage = (d, time_s)

    @staticmethod
    def _closest(record):
        if record is None:
            return None
        gap, t, along, cross = record
        return {"time_s": t, "surface_gap_m": gap, "along_track_m": along, "cross_track_m": cross}

    def stamp(self):
        designed = None if self.state is None else self.state.designed_contact_s()
        sync = None
        if designed is not None and self.after_moving is not None:
            sync = (self.first_contact_s if self.first_contact_s is not None else self.after_moving[1]) - designed
        slack = None
        lead = getattr(getattr(self.state, "route", None), "lead_s", None)
        if self.passage is not None and lead is not None:
            slack = max(0.0, lead - self.passage[1])
        return {"schema": EXPOSURE_SCHEMA, "first_visible_s": self.first_visible_s,
                "first_ray_return_s": self.first_ray_return_s,
                "ray_model": "point-return proxy on the %s-ray fan, static shadowing from the candidate geometry"
                             % (None if self.sensor is None else self.sensor.ray_count),
                "closest_after_moving": self._closest(self.after_moving),
                "closest_after_visible": self._closest(self.after_visible),
                "first_contact_after_moving_s": self.first_contact_s,
                "designed_contact_s": designed, "sync_error_s": sync,
                "conflict_pose_passage": (None if self.passage is None else
                                          {"distance_m": self.passage[0], "time_s": self.passage[1]}),
                "unsynchronizable_slack_s": slack,
                "sampling": "update samples (no interpolation)"}
