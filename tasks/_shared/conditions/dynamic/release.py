"""G23b: arrival-synchronized release of the pinned D7 intruder (runtime rule S).

Owner decision 2026-10-03 (NC 09, G23 option S): the D7 intruder rests at a
point of its pinned line and is released when the own ship is about to
arrive, so that a real conflict exists at every dose. This amends the rule
"object motion is independent of the controller" (C7 STRATEGY.md:407 P0-6,
D7 DYNAMIC_HAZARDS.md:143-146) for this purpose only. Every pinned quantity is
unchanged: key, path type, intruder line, speed and radius, start/stop of the
admitted motion window, t_c, t_avoid, detection range per dose, witness and
layout; the admitted pools are not re-admitted (D7_SUITE_VERSION unchanged).

Transit tasks (plan kind "transit", rule RUNTIME_RULE_VERSION):
  t_1    = t_c - 4 t_avoid (the D1 reveal time; [t_1, t_c] is the unforced D1 leg)
  m      = release_margin(task)  (RELEASE_MARGIN_S = 2 s, G23; docking 0 s)
  t_hold = the earliest t in [max(start_s, t_1 - m), t_1] from which the
           nominal speed is constant up to t_c (|accel| < 1e-6 m/s^2 on every
           plan interval of [t, t_c]); so t_hold = max(start_s, t_1 - m) (G23)
           unless the nominal still accelerates or turns there (G23b)
  T_rel  = t_c - t_hold                                          (release lead)
  H      = intruder.position(t_hold)                             (hold point)
  Before the release the intruder rests at H (physically present, sensed like
  any object). It is released at the first update at which the own ship's
  predicted time to the conflict along the frozen nominal route is <= T_rel,
  or at the first update at which the resting object is detectable (the
  detection guard: nothing is ever detected before it moves). From the
  release time t_r on it moves on its pinned line with phase
  t_hold + (t - t_r) (it still stops at the admitted stop point).
  Predicted time to conflict = G23's constant-velocity rule, distance to go
  along the nominal route over the own ship's speed: the observer is
  projected on the nominal route (the candidate's own witness prefix up to
  t_1, then the straight D1 leg to the nominal pose at t_c; identical bytes
  at every dose; monotone progress, a segment is passed once the observer is
  beyond its end normal); d_go = nominal distance from the foot point to the
  conflict point; speed V = k * v_nom(t*), the nominal speed at the foot point
  scaled by the pace ratio k = max(Vw, F) / max(Vw_nom, F) of the observer's
  surge and the nominal foot-point speed, both time-weighted means over the
  last PACE_WINDOW_S of updates (same trapezoid on both; F =
  PACE_SPEED_FLOOR_MPS); tau = d_go / max(V, F). G23c (REV-G23b F9): where the
  window-mean nominal speed is below NOMINAL_REST_FRACTION * u_ref (the nominal
  is at or near rest, e.g. spawned at rest while the own ship spawns moving)
  the pace ratio is meaningless (up to 40x) and V is the window-mean surge
  itself. V is a noise-robust surge
  estimate that equals the nominal speed exactly for the nominal own ship
  (k = 1), so for it tau = t_c - t on [t_hold, t_c] (constant speed there) and
  tau > T_rel before t_hold (the nominal never exceeds its cruise speed):
  t_r = t_hold and the motion is the pinned one from t_hold on (u_ref
  identity: for the nominal own ship at u_ref the D1 TTC scale and the dose
  ratios are those of v2.5). For any other own ship the dose stays a
  detection DISTANCE (the reveal is visibility.detectable with the pinned
  detection range), so the realized TTC / t_avoid at the first detection
  scales with the closing speed (REV-G23b F1: WAM-V at 2.0 m/s D1 2.0-2.4,
  D4 0.63-0.75); exposure records that per episode. t_r is interpolated
  inside the update interval (t_r = t - (T_rel - tau), clamped to the previous
  update time), so the release depends on the update rate by at most one
  interval; |t_r - t_hold| <= RELEASE_SNAP_S snaps to t_hold.

Docking refinement (G23: docking passes 0.5-1.9 m clear with the naive rule;
the own ship starts at rest a short way from the conflict, accelerates, and
enters the release distance while it is still accelerating):
  * margin 0 s (TASK_RELEASE_MARGIN_S): the shortest lead that keeps the u_ref
    identity (t_hold <= t_c - 4 t_avoid = the D1 reveal time);
  * accelerating pace (ACCELERATING_PACE_TASKS): while the observer is
    faster than the nominal (window means, by FASTER_MARGIN_MPS) and its surge
    speed rises (slope over the pace window >= ACCEL_GUARD_MPS2), tau is the
    time to cover the remaining nominal distance accelerating at that rate up
    to the vessel top speed (envelope.top_speed_mps), when that is earlier.
  Both only ever release EARLIER, never for an observer on the nominal plan.
  G23b CPU sweep (validation/sweep): applied to every transit task the
  accelerating pace costs conflicts at 0.7-1.5 m/s (it assumes acceleration
  to top speed) and helps only near top speed, so it is docking-only; a
  "range guard" (release once the distance to go <= the nominal distance in
  T_rel) never fired in 44228 validation episodes (the sync condition always
  came first) and was removed. A "nominal-paced" estimator (tau = remaining
  nominal time / k) was also exact at u_ref but assumes the own ship slows in
  every nominal turn; it lost most conflicts of slow BlueBoat runs (path
  following at 0.7 m/s: 0 of 32, against 32 of 32 with the rule above).

Hold tasks (station keeping): the own ship is stationary at the conflict point
by construction; its time to conflict is the intruder's own travel time, so the
pinned schedule is already arrival-synchronized. S is the identity there:
phase(t) = t for every t (G23c F13: no hold or release time is stamped; release.applies = false).
G23e: retired for station keeping, whose encounter now lies on the approach to the hold circle
(sk_approach.py, plan kind "approach" in TIMED_PLAN_KINDS: timed like a transit route).

Moving basin: excluded (native motion axis; no pinned line, no t_c; the
objects are visible from reset and the dose is the native speed ratio).

Everything here is a deterministic pure function of the admitted candidate and
the sequence of (time, observer pose) updates of one env row and episode.
"""

from collections import deque
import hashlib
import json
import math

from .geometry import point_clearance
from .kinematics import integrate
from .model import UNFORCED_TTC_TOLERANCE_S, Pose, TimedState
from .reference import NominalPlan
from .visibility import detectable

RUNTIME_RULE_VERSION = "d7_release_sync_v1"
PINNED_RULE_VERSION = "pinned_v0"           # diagnostic only: v2.5 behaviour (controller-independent)
REVEAL_RULE_VERSION = "d7_reveal_sync_v1"   # G23c L2 candidate rule S2 (reveal.py); diagnostic in layer 3
HYBRID_RULE_VERSION = "d7_reveal_sync_v2"   # G23c L3: S2 + hybrid corridor predictor (reveal.py); G23e B: diagnostic
V3_RULE_VERSION = "d7_reveal_sync_v3"       # G23e layer B (REV-G23d M1-M4, reveal.py), the default
RUNTIME_RULES = (V3_RULE_VERSION, HYBRID_RULE_VERSION, REVEAL_RULE_VERSION, RUNTIME_RULE_VERSION, PINNED_RULE_VERSION)
DEFAULT_RULE = V3_RULE_VERSION
DIAGNOSTIC_RULES = (HYBRID_RULE_VERSION, REVEAL_RULE_VERSION, RUNTIME_RULE_VERSION, PINNED_RULE_VERSION)
RELEASE_MARGIN_S = 2.0                      # G23 predict_sync.py m = 2 s
TASK_RELEASE_MARGIN_S = {"docking": 0.0}    # docking refinement
ACCELERATING_PACE_TASKS = frozenset(("docking",))
PACE_SPEED_FLOOR_MPS = 0.05                 # floor of both speeds in the pace ratio
PACE_WINDOW_S = 0.5                         # pace ratio and acceleration from the last 0.5 s of updates
# |t_r - t_hold| <= this -> t_r = t_hold: the owner-approved timing bound of a sub-tolerance unforced window
# (model.UNFORCED_TTC_TOLERANCE_S = 0.25 ms; the nominal speed may still change by < 1e-6 m/s^2 there)
RELEASE_SNAP_S = UNFORCED_TTC_TOLERANCE_S
ACCEL_GUARD_MPS2 = 0.05                     # "accelerating" threshold of the docking pace refinement
FASTER_MARGIN_MPS = 0.05                    # "faster than the nominal": mean v > mean v_nom + this margin
CPA_HORIZON_S = 120.0                       # predicted closest approach horizon after the release
CAUSES = ("sync", "accelerating_pace", "detection_guard")
RULE_TEXT = ("transit: rest at H = position(t_hold), t_hold = earliest t in [max(start_s, t_c - 4 t_avoid - m), "
             "t_c - 4 t_avoid] with constant nominal speed on [t, t_c], m = 2 s (docking 0 s); release at the first "
             "update with route distance to go / (k * nominal foot-point speed) <= t_c - t_hold (k = window-mean "
             "surge / window-mean nominal foot-point speed), docking also when accelerating faster than nominal and "
             "the accelerated arrival is within the lead, or when the resting object is detectable; then phase "
             "t_hold + (t - t_r) on the pinned line. hold: pinned schedule. moving_basin: native axis, excluded")
CONSTANT_SPEED_TOL_MPS2 = 1.0e-6            # = model.UNFORCED_ACCEL_TOL_MPS2 (the hold stretch test)
# G23e: plan kinds timed by a runtime rule; "approach" = station keeping's approach encounter (sk_approach.py),
# which replaced the hold-task pinned schedule
TIMED_PLAN_KINDS = ("transit", "approach")
NOMINAL_REST_FRACTION = 0.5                 # G23c F9: below this x u_ref the speed estimate is the window mean


def release_margin(task):
    return TASK_RELEASE_MARGIN_S.get(task, RELEASE_MARGIN_S)


def rule_constants():
    return {"release_margin_s": RELEASE_MARGIN_S, "task_release_margin_s": dict(TASK_RELEASE_MARGIN_S),
            "accelerating_pace_tasks": sorted(ACCELERATING_PACE_TASKS),
            "pace_speed_floor_mps": PACE_SPEED_FLOOR_MPS, "pace_window_s": PACE_WINDOW_S,
            "release_snap_s": RELEASE_SNAP_S, "accel_guard_mps2": ACCEL_GUARD_MPS2,
            "faster_margin_mps": FASTER_MARGIN_MPS, "cpa_horizon_s": CPA_HORIZON_S,
            "constant_speed_tol_mps2": CONSTANT_SPEED_TOL_MPS2, "nominal_rest_fraction": NOMINAL_REST_FRACTION}


def rule_block():
    """JSON-native declaration of the runtime rule (stamped in every admitted D7 stamp, runtime.runtime_rule)."""
    return {"version": RUNTIME_RULE_VERSION, "constants": rule_constants(), "causes": list(CAUSES),
            "text": RULE_TEXT}


def rule_block_for(rule):
    """The runtime_rule block a stamp carries for a DynamicRuntimeConfig.motion_rule (G23c)."""
    if rule == V3_RULE_VERSION:                                  # G23e layer B default
        from .reveal import rule_block as reveal_block          # lazy: reveal.py imports this module
        return reveal_block(v3=True)
    if rule == HYBRID_RULE_VERSION:                              # G23c L3; G23e layer B: diagnostic
        from .reveal import rule_block as reveal_block
        return dict(reveal_block(hybrid=True), diagnostic_only=True)
    if rule == REVEAL_RULE_VERSION:                              # G23c L3: S2 v1 is diagnostic
        from .reveal import rule_block as reveal_block
        return dict(reveal_block(), diagnostic_only=True)
    if rule == RUNTIME_RULE_VERSION:                             # G23c L2: rule S is diagnostic
        return dict(rule_block(), diagnostic_only=True)
    if rule not in RUNTIME_RULES:
        raise ValueError("unknown D7 runtime rule " + repr(rule))
    return {"version": rule, "diagnostic_only": True}


def _digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class ReleaseRoute:
    """Dose-invariant nominal route of one candidate up to the conflict (transit tasks)."""

    def __init__(self, candidate):
        encounter = candidate.request.encounter
        obj = encounter.intruders[0]
        self.task = candidate.task
        self.t_c = float(encounter.meta("conflict_time_s"))
        self.t_avoid = float(encounter.t_avoid_s)
        # G23e: the leg starts at the lowest APPLICABLE tier's window (station keeping BlueBoat: D2); every other
        # encounter carries no dose_window_start_s and keeps the D1 window start
        self.t_1 = float(encounter.meta("dose_window_start_s", encounter.meta("d1_window_start_s")))
        prefix = tuple(candidate.request.prefix)
        if not prefix or prefix[-1].time_s < self.t_1 - 1e-9:
            raise ValueError("release route needs the witness prefix through the D1 window start")
        nodes = [s for s in prefix if s.time_s < self.t_1]
        after = next(s for s in prefix if s.time_s >= self.t_1)
        pose_1 = NominalPlan(prefix).pose_at(self.t_1)
        nodes.append(TimedState(self.t_1, pose_1, after.accel_mps2, after.yaw_rate_rps))
        pose_c = integrate(pose_1, 0.0, 0.0, self.t_c - self.t_1)
        nodes.append(TimedState(self.t_c, pose_c, 0.0, 0.0))
        self.nodes = tuple(nodes)
        self.conflict_pose = pose_c
        low = max(float(obj.start_s), self.t_1 - release_margin(self.task))
        start_of_constant = self.t_1          # earliest node from which every interval up to t_c keeps its speed
        for before, after_ in zip(reversed(self.nodes[:-1]), reversed(self.nodes[1:])):
            if abs(after_.accel_mps2) >= CONSTANT_SPEED_TOL_MPS2:
                break
            start_of_constant = min(start_of_constant, before.time_s)
        self.t_hold = min(self.t_1, max(low, start_of_constant))
        self.margin_s = self.t_1 - self.t_hold
        self.lead_s = self.t_c - self.t_hold
        self.hold_point = obj.position(self.t_hold)
        self._dist = [0.0]               # nominal travelled distance at each node
        for before, after_ in zip(self.nodes, self.nodes[1:]):
            dt = after_.time_s - before.time_s
            self._dist.append(self._dist[-1] + before.pose.speed * dt + 0.5 * after_.accel_mps2 * dt * dt)
        self.length_m = self._dist[-1]
        self.digest = _digest([[s.time_s, s.pose.x, s.pose.y, s.pose.yaw, s.pose.speed, s.accel_mps2,
                                s.yaw_rate_rps] for s in self.nodes])

    def distance_in_segment(self, index, tau):
        before, after = self.nodes[index], self.nodes[index + 1]
        return self._dist[index] + before.pose.speed * tau + 0.5 * after.accel_mps2 * tau * tau

    def segment_pose(self, index, tau):
        before, after = self.nodes[index], self.nodes[index + 1]
        return integrate(before.pose, after.accel_mps2, after.yaw_rate_rps, tau)

    def foot(self, index, q):
        """Foot point tau in segment index (Newton on (p - q) . T, from the chord projection)."""
        before, after = self.nodes[index], self.nodes[index + 1]
        dt = after.time_s - before.time_s
        ax, ay = before.pose.x, before.pose.y
        cx, cy = after.pose.x - ax, after.pose.y - ay
        chord2 = cx * cx + cy * cy
        if chord2 <= 1e-24:
            return dt
        tau = max(0.0, min(1.0, ((q.x - ax) * cx + (q.y - ay) * cy) / chord2)) * dt
        a, w = after.accel_mps2, after.yaw_rate_rps
        for _ in range(8):
            p = integrate(before.pose, a, w, tau)
            c, s = math.cos(p.yaw), math.sin(p.yaw)
            ex, ey = p.x - q.x, p.y - q.y
            dg = (before.pose.speed + a * tau) + w * (-ex * s + ey * c)
            if dg <= 1e-9:
                break
            new = max(0.0, min(dt, tau - (ex * c + ey * s) / dg))
            if abs(new - tau) <= 1e-13:
                tau = new
                break
            tau = new
        return tau


class ReleaseState:
    """Per env row and episode: hold, release decision, phase."""

    def __init__(self, candidate, plan_kind, rule=RUNTIME_RULE_VERSION):
        if rule not in RUNTIME_RULES:
            raise ValueError("unknown D7 runtime rule " + repr(rule))
        self.rule = rule
        self.plan_kind = plan_kind
        encounter = candidate.request.encounter
        self.obj = encounter.intruders[0]
        self.top_speed = float(candidate.request.envelope.top_speed_mps)
        self.u_ref = float(candidate.request.envelope.reference_speed_mps)
        self.accelerating_pace = candidate.task in ACCELERATING_PACE_TASKS
        self.route = None
        self.segment = 0
        self.release_s = None
        self.release_tick_s = None
        self.cause = None
        self.snapshot = None
        self.last_time = None
        self.window = deque()           # (time, observer surge, nominal speed at the foot point)
        self.window_sums = [0.0, 0.0]   # running trapezoid integrals of the two speeds over the window
        self.updates_held = 0
        self.t_c = float(encounter.meta("conflict_time_s"))
        self.t_hold = None
        if rule == PINNED_RULE_VERSION or plan_kind not in TIMED_PLAN_KINDS:
            # pinned schedule: the v2.5 motion (diagnostic rule) and every hold task (station keeping: the
            # own ship is stationary at the conflict point, so the pinned schedule is already synchronized;
            # G23c F13: no hold time and no release time are stamped for it)
            return
        self.route = ReleaseRoute(candidate)
        self.t_hold = self.route.t_hold

    # ----------------------------------------------------------- motion law --
    def phase(self, time_s):
        if self.route is None:          # pinned rule and hold tasks: the pinned schedule itself
            return time_s
        if self.release_s is None or time_s < self.release_s:
            return self.t_hold
        if self.release_s == self.t_hold:
            return time_s
        return self.t_hold + (time_s - self.release_s)

    def released_at(self, time_s):
        if self.route is None:          # pinned schedule: moving from the admitted start
            return time_s >= self.obj.start_s
        return self.release_s is not None and time_s >= self.release_s

    def unreleased(self):
        """G23c: True when the rule holds the object and has not released it (non-exposure reason)."""
        return self.route is not None and self.release_s is None

    def unreleased_reason(self):
        return "not_released"

    def detect(self, observer, obj, time_s, evidence, sensor):
        """Detectability of the runtime object (rule S / pinned: visibility.detectable, never gated)."""
        return detectable(observer, obj, time_s, evidence, sensor)

    def present(self, time_s):
        """Physically present (collidable) at time_s: always under rules S and pinned_v0."""
        return True

    def moving_since(self):
        return self.obj.start_s if self.route is None else self.release_s

    def designed_contact_s(self):
        """Contact time the rule designed: t_c (pinned schedule), t_r + T_rel (rule S)."""
        if self.route is None:
            return self.t_c
        return None if self.release_s is None else self.release_s + self.route.lead_s

    @property
    def conflict_pose(self):
        return None if self.route is None else self.route.conflict_pose

    # -------------------------------------------------------- release logic --
    def _progress(self, q):
        """(segment, tau): monotone route progress; a segment is passed once q is beyond its end normal."""
        nodes = self.route.nodes
        last = len(nodes) - 2
        while self.segment < last:
            end = nodes[self.segment + 1].pose
            if (q.x - end.x) * math.cos(end.yaw) + (q.y - end.y) * math.sin(end.yaw) < 0.0:
                break
            self.segment += 1
        return self.segment, self.route.foot(self.segment, q)

    def _push(self, time_s, speed, v_nom):
        """Append one update to the pace window and drop what is older than PACE_WINDOW_S (keeping one
        sample at or before the window start); running trapezoid sums, the same arithmetic on both."""
        w, sums = self.window, self.window_sums
        if w:
            t0, a0, b0 = w[-1]
            sums[0] += 0.5 * (a0 + speed) * (time_s - t0)
            sums[1] += 0.5 * (b0 + v_nom) * (time_s - t0)
        w.append((time_s, speed, v_nom))
        while len(w) > 2 and w[1][0] <= time_s - PACE_WINDOW_S:
            (t0, a0, b0), (t1, a1, b1) = w[0], w[1]
            sums[0] -= 0.5 * (a0 + a1) * (t1 - t0)
            sums[1] -= 0.5 * (b0 + b1) * (t1 - t0)
            w.popleft()
        span = w[-1][0] - w[0][0]
        if len(w) == 1 or span <= 0.0:
            return speed, v_nom, None
        accel = (speed - w[0][1]) / span if span >= 0.5 * PACE_WINDOW_S else None
        return sums[0] / span, sums[1] / span, accel

    def _accelerating_time(self, distance, speed, accel):
        """Time to cover distance accelerating at accel from speed up to the vessel top speed."""
        top = max(self.top_speed, speed)
        if accel <= 0.0 or speed >= top:
            return distance / max(speed, PACE_SPEED_FLOOR_MPS)
        ramp = (top - speed) / accel
        ramp_distance = speed * ramp + 0.5 * accel * ramp * ramp
        if distance <= ramp_distance:
            return (-speed + math.sqrt(speed * speed + 2.0 * accel * distance)) / accel
        return ramp + (distance - ramp_distance) / top

    def observe(self, time_s, observer, evidence=None, sensor=None):
        """Advance the release decision with one update; returns True when released at this update."""
        if self.release_s is not None or self.route is None:
            return False
        previous_time = self.last_time
        self.last_time = time_s
        if observer is None:
            return False
        route = self.route
        index, tau_seg = self._progress(observer)
        before, after = route.nodes[index], route.nodes[index + 1]
        t_star = before.time_s + tau_seg
        v_nom = before.pose.speed + after.accel_mps2 * tau_seg
        v_obs, v_ref, accel = self._push(time_s, observer.speed, v_nom)
        remaining = route.t_c - t_star
        k = max(v_obs, PACE_SPEED_FLOOR_MPS) / max(v_ref, PACE_SPEED_FLOOR_MPS)
        distance_to_go = route.length_m - route.distance_in_segment(index, tau_seg)
        if v_ref >= NOMINAL_REST_FRACTION * self.u_ref:
            speed = max(k * v_nom, PACE_SPEED_FLOOR_MPS)
        else:                   # G23c F9: nominal near rest at the foot point; the window-mean surge
            speed = max(v_obs, PACE_SPEED_FLOOR_MPS)
        tau = distance_to_go / speed
        # G23c F13: a sync condition met within the snap tolerance is a sync (ties at t_hold were labelled
        # detection_guard when the held object reached the detection horizon on the same update)
        cause = "sync" if tau <= route.lead_s + RELEASE_SNAP_S else None
        if (cause is None and self.accelerating_pace and accel is not None and accel >= ACCEL_GUARD_MPS2
                and v_obs > v_ref + FASTER_MARGIN_MPS and v_obs > PACE_SPEED_FLOOR_MPS):
            predicted = self._accelerating_time(distance_to_go, observer.speed, accel)
            if predicted <= route.lead_s:
                cause, tau = "accelerating_pace", predicted
        if cause is None and evidence is not None and sensor is not None and \
                detectable(observer, _Held(self.obj, route.hold_point), time_s, evidence, sensor):
            cause, tau = "detection_guard", route.lead_s
        if cause is None:
            self.updates_held += 1
            return False
        lower = time_s if previous_time is None else previous_time
        release = time_s if cause == "detection_guard" else max(lower, min(time_s, time_s - (route.lead_s - tau)))
        if abs(release - self.t_hold) <= RELEASE_SNAP_S:
            release = self.t_hold
        self.release_s = release
        self.release_tick_s = time_s
        self.cause = cause
        self.snapshot = {"observer": {"x": observer.x, "y": observer.y, "yaw": observer.yaw,
                                      "speed": observer.speed},
                         "route_segment": index, "nominal_time_at_foot_s": t_star,
                         "nominal_speed_at_foot_mps": v_nom, "remaining_nominal_s": remaining,
                         "distance_to_go_m": distance_to_go, "pace_ratio": k, "speed_estimate_mps": speed,
                         "mean_observer_speed_mps": v_obs, "mean_nominal_speed_mps": v_ref,
                         "predicted_time_to_conflict_s": tau, "observer_accel_mps2": accel}
        return True

    # ---------------------------------------------------------------- stamp --
    def stamp(self, footprint):
        if self.rule == PINNED_RULE_VERSION:
            return {"rule": PINNED_RULE_VERSION, "plan_kind": self.plan_kind, "applies": False,
                    "reason": "diagnostic pinned schedule (v2.5 motion)"}
        if self.route is None:
            return {"rule": RUNTIME_RULE_VERSION, "plan_kind": self.plan_kind, "applies": False,
                    "reason": "stationary own ship: pinned schedule (no hold, no release)"}
        out = {"rule": RUNTIME_RULE_VERSION, "plan_kind": self.plan_kind, "applies": True,
               "t_hold_s": self.t_hold, "release_time_s": self.release_s,
               "release_tick_time_s": self.release_tick_s, "cause": self.cause,
               "released": self.release_s is not None}
        out.update(hold_point=list(self.route.hold_point), margin_s=self.route.margin_s,
                   release_lead_s=self.route.lead_s, conflict_time_s=self.route.t_c,
                   route_digest=self.route.digest, route_length_m=self.route.length_m,
                   state_at_release=self.snapshot, updates_held=self.updates_held,
                   predicted_closest_approach=(None if self.snapshot is None else
                                               self.closest_approach(footprint)))
        return out

    def closest_approach(self, footprint):
        """Constant-velocity prediction at the release update: own ship along its heading at its surge
        speed, intruder on its line from its released position (stopping at its stop point). Returns the
        minimum footprint-to-disk surface gap and when it occurs (seconds after the release update)."""
        own = self.snapshot["observer"]
        start = Pose(own["x"], own["y"], own["yaw"], own["speed"])
        phase0 = self.phase(self.release_tick_s)
        horizon = CPA_HORIZON_S
        c, s = math.cos(start.yaw), math.sin(start.yaw)

        def gap(dt):
            ox = start.x + own["speed"] * c * dt
            oy = start.y + own["speed"] * s * dt
            x, y = self.obj.position(phase0 + dt)        # Intruder.position stops at stop_s
            return point_clearance(x, y, Pose(ox, oy, start.yaw, start.speed), footprint) - self.obj.radius_m

        # scan (0.2 s grid) then a golden-section refinement around the best grid point
        steps = 600
        best_t, best = 0.0, gap(0.0)
        for i in range(1, steps + 1):
            t = horizon * i / steps
            g = gap(t)
            if g < best:
                best_t, best = t, g
        low, high = max(0.0, best_t - horizon / steps), min(horizon, best_t + horizon / steps)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        for _ in range(60):
            a = high - ratio * (high - low)
            b = low + ratio * (high - low)
            if gap(a) <= gap(b):
                high = b
            else:
                low = a
        t = 0.5 * (low + high)
        g = gap(t)
        if g < best:
            best_t, best = t, g
        return {"min_surface_gap_m": best, "time_after_release_s": best_t, "predicted_contact": best <= 0.0,
                "note": "constant-velocity prediction at the release update, NOT a conflict guarantee "
                        "(REV-G23b F4: disagrees with unmanoeuvred contact in 35% of probe runs)",
                "model": "constant velocity from the release update (own: heading and surge speed; "
                         "intruder: pinned line, stops at its stop point)", "horizon_s": horizon}


class _Held:
    """The resting intruder at its hold point (duck-typed for visibility.detectable)."""

    def __init__(self, obj, point):
        self.radius_m = obj.radius_m
        self._point = point

    def position(self, time_s):
        return self._point


class ReleasedIntruder:
    """The pinned intruder under the runtime rule (duck-typed Intruder for geometry/visibility)."""

    def __init__(self, obj, state):
        self.base = obj
        self.state = state
        self.path_type = obj.path_type
        self.vx, self.vy = obj.vx, obj.vy
        self.radius_m = obj.radius_m
        self.reveal_s = obj.reveal_s
        self.reveal_kind = obj.reveal_kind

    @property
    def speed_bound_mps(self):
        return self.base.speed_bound_mps

    def position(self, time_s):
        line = getattr(self.state, "line_position", None)      # G23c L2: S2 (extended certified line)
        if line is not None:
            return line(time_s)
        return self.base.position(self.state.phase(time_s))

    def velocity(self, time_s):
        line = getattr(self.state, "line_velocity", None)
        if line is not None:
            return line(time_s)
        if self.state.route is not None and not self.state.released_at(time_s):
            return 0.0, 0.0
        return self.base.velocity(self.state.phase(time_s))
