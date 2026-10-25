"""Env-side appearing-suite bridge (G3 rule 5): CPU reference, one instance per env, many rows.

Clock: the same integer physics tick per row as the D7 bridge (time = tick *
physics_dt_s, reset to zero only for reset rows).

Call pattern (JournalEnvMixin, right after the D7 bridge in each place):
  * journal_reset: reset_rows(...) then step({env: 0}, poses) (tick 0, no check);
  * journal_substep (every physics substep): step(ticks, poses) -- swept contact
    only, never a trigger check;
  * journal_pre_dones (once per control step, after the substeps): step(ticks,
    poses, check_trigger=True) -- the CONTROL-RATE trigger check.
A tick that was already consumed is not swept twice (the D7 tick discipline).

Trigger (rule 5): an admitted row at D1..D_max holds an inactive cylinder; at the
first trigger check whose gap (D7 footprint rectangle at the COM pose to the
cylinder surface, dynamic/geometry.point_clearance - r) is <= d_k, the cylinder
becomes PHYSICAL and VISIBLE at once and stays so until the episode ends. The
realized gap, the relative bearing, the time, the tick and the pose are
recorded, and (G20b) the realized closing speed (surge x cos bearing) and the
realized time to contact (gap / closing speed, None when not closing;
stamp.trigger_pressure, recomputed by the stamp reader). G20d (owner decision 7A):
the trigger record also says whether the cylinder was VISIBLE at the trigger step
(visible_at_trigger: at least one of the ray fan's rays has the cylinder as its
first hit, occlusion.py, with the key's static occluders from the placement
document; visible_rays = how many), copied to runtime.visible_at_trigger (None
when the exposure was not realized). Occluded cylinders are kept. A row that
never reaches d_k never sees the cylinder: its stamp says exposure_realized false
(still assigned, itt_count 1). D0 rows and not-admitted rows (D16/D24) never
publish a sample.

Contact: after the trigger, every interval between consumed ticks is checked
with the D7 swept guard (geometry.swept_interval_safe, rotation bound plus a
centre-deviation bound own_speed_bound * interval / 2; the cylinder is static),
latched like the D7 bridge. Families whose static obstacles are analytic
(HazardNav) additionally see the cylinder in their own clearance path once it
is published physical; families without static obstacles (path following) use
this latch as the collision signal of the cylinder (journal_appearing_contact).
"""
import math

from ..dynamic.adapters import ObstacleSample
from ..dynamic.geometry import swept_interval_safe
from ..dynamic.kinematics import wrap
from ..dynamic.model import Circle, Pose, check_number
from . import config, occlusion
from .placement import bearing_at, gap_at
from .stamp import ADMITTED, NOT_ADMITTED, make_appearing_stamp, trigger_pressure

TERMINATION_REASONS = ("success", "collision", "timeout", "out_of_bounds", "failure", "capsize", "sinking",
                       "airborne", "non_finite_state", "aborted")


class AppearingBridge:
    def __init__(self, task, slot, physics_dt_s, own_speed_bound_mps, footprint, t_avoid_s, gap_override_m=None):
        """``gap_override_m`` (G20b, diagnostic token appearing.gap_m): every active row triggers at this gap
        instead of its dose's d_k; bounded by the vessel's calibration rung D5 (ladder.GAP_OVERRIDE_MIN_M) and the
        task's start gap."""
        config.check_wired(task)
        check_number("physics_dt_s", physics_dt_s, positive=True)
        check_number("own_speed_bound_mps", own_speed_bound_mps, positive=True)
        check_number("t_avoid_s", t_avoid_s, positive=True)
        self.task = task
        self.slot = int(slot)
        self.dt = float(physics_dt_s)
        self.own_speed_bound = float(own_speed_bound_mps)
        self.footprint = footprint
        self.t_avoid_s = float(t_avoid_s)
        self.gap_override_m = (None if gap_override_m is None
                               else config.check_gap_override(task, t_avoid_s, None, gap_override_m))
        self.rows = {}
        self.completed = []

    # ----------------------------------------------------------------- rows --
    def reset_rows(self, env_ids, assignments):
        ids = tuple(int(i) for i in env_ids)
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("reset ids must be unique nonnegative rows")
        staged = {}
        for env in ids:
            if env in self.rows and not self.rows[env]["finished"]:
                raise ValueError("finish and stamp every appearing episode before reset")
            a = assignments[env]
            if a.key.env != env or a.key.hazard_group != config.hazard_group(self.task) or a.task != self.task:
                raise ValueError("wrong task or environment appearing assignment")
            if a.exposure not in (ADMITTED, NOT_ADMITTED):
                raise ValueError("appearing assignment without an exposure kind")
            active = a.exposure == ADMITTED and a.dose > 0
            row = {"assignment": a, "active": active, "last_tick": -1, "prev": None, "finished": False,
                   "stamp": None, "triggered": False, "trigger": None, "checks": 0, "last_check_gap_m": None,
                   "min_gap_before_trigger_m": None, "min_sampled_gap_after_trigger_m": None,
                   "sampled_contact_after_trigger": False, "contact": False, "first_contact_s": None,
                   "swept_intervals": 0, "observer_updates": 0}
            if active:
                x, y, r = a.cylinder
                gap = a.trigger_gap_m
                expected = config.trigger_gap_m(a.dose, self.t_avoid_s, a.vessel_id)
                if gap is None or not math.isclose(gap, expected, rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError("appearing assignment trigger gap does not match d_k of its dose")
                row.update(cylinder=(x, y, r), dose_gap_m=gap,
                           trigger_gap_m=gap if self.gap_override_m is None else self.gap_override_m,
                           visibility=a.placement.get("visibility"))      # G20d: occluders + ray fan
            staged[env] = row
        self.rows.update(staged)

    def _samples(self, row):
        if not row["active"]:
            return ()
        x, y, r = row["cylinder"]
        on = row["triggered"]
        return (ObstacleSample(self.slot, x, y, r, on, on),)

    # ----------------------------------------------------------------- step --
    def step(self, ticks, poses, check_trigger=False):
        """ticks: {env: int}, poses: {env: Pose (COM, env-local)} -> {env: samples}."""
        out = {}
        for env, tick in ticks.items():
            row = self.rows[env]
            if row["finished"]:
                raise ValueError("appearing row finished; reset it before stepping")
            if type(tick) is not int or tick < 0:
                raise ValueError("tick must be a nonnegative integer")
            if tick < row["last_tick"]:
                raise ValueError("physics tick went backwards without a reset")
            pose = poses[env]
            if not isinstance(pose, Pose):
                raise TypeError("pose must be a Pose")
            if tick != row["last_tick"]:
                row["observer_updates"] += 1
                if row["active"]:
                    self._sweep(row, tick, pose)
                row["prev"] = (tick, pose)
                row["last_tick"] = tick
            if check_trigger and row["active"] and not row["triggered"]:
                self._check(row, tick, row["prev"][1])
            out[env] = self._samples(row)
        return out

    def _sweep(self, row, tick, pose):
        x, y, r = row["cylinder"]
        prev = row["prev"]
        if row["triggered"]:
            gap = gap_at(pose, x, y, r, self.footprint)
            low = row["min_sampled_gap_after_trigger_m"]
            row["min_sampled_gap_after_trigger_m"] = gap if low is None else min(low, gap)
            row["sampled_contact_after_trigger"] |= gap <= 0.0
        if prev is None or not row["triggered"] or row["trigger"]["tick"] > prev[0]:
            return
        prev_tick, prev_pose = prev
        interval = (tick - prev_tick) * self.dt
        deviation = self.own_speed_bound * interval * 0.5
        circle = Circle(x, y, r)
        angular = abs(wrap(pose.yaw - prev_pose.yaw))
        row["swept_intervals"] += 1
        if not swept_interval_safe(prev_pose, pose, self.footprint, [circle], [circle], 0.0, angular, deviation):
            if not row["contact"]:
                row["first_contact_s"] = prev_tick * self.dt
            row["contact"] = True

    def _check(self, row, tick, pose):
        x, y, r = row["cylinder"]
        gap = gap_at(pose, x, y, r, self.footprint)
        row["checks"] += 1
        if gap <= row["trigger_gap_m"]:
            row["triggered"] = True
            bearing = bearing_at(pose, x, y)
            vis = row.get("visibility")
            if vis is None:
                raise ValueError("appearing placement without its visibility block (G20d pools carry it)")
            rays = occlusion.visible_rays(pose.x, pose.y, pose.yaw, (x, y, r), vis["occluders"], vis["max_range_m"],
                                          vis["ray_count"])
            row["trigger"] = {"tick": int(tick), "time_s": tick * self.dt, "gap_m": gap,
                              "bearing_rad": bearing,
                              "pose": [pose.x, pose.y, pose.yaw, pose.speed],
                              "previous_check_gap_m": row["last_check_gap_m"],
                              "check_index": row["checks"],
                              "visible_at_trigger": rays > 0, "visible_rays": int(rays),
                              **trigger_pressure(gap, bearing, pose.speed)}
            row["min_sampled_gap_after_trigger_m"] = gap
            row["sampled_contact_after_trigger"] = gap <= 0.0
            return
        low = row["min_gap_before_trigger_m"]
        row["min_gap_before_trigger_m"] = gap if low is None else min(low, gap)
        row["last_check_gap_m"] = gap

    # --------------------------------------------------------------- outputs --
    def contact(self, env):
        """Swept-contact latch of the appearing cylinder (always False before the trigger)."""
        return bool(self.rows[env]["contact"])

    def triggered(self, env):
        return bool(self.rows[env]["triggered"])

    def finish(self, env, termination_reason):
        """Stamp the finished episode (call BEFORE the env resets the row); idempotent."""
        if termination_reason not in TERMINATION_REASONS:
            raise ValueError("explicit terminal reason required: " + repr(termination_reason))
        row = self.rows[env]
        if row["finished"]:
            return row["stamp"]
        a = row["assignment"]
        admitted = a.exposure == ADMITTED
        if not admitted:
            reason = NOT_ADMITTED
        elif a.dose == 0:
            reason = "zero_dose"
        elif row["triggered"]:
            reason = None
        elif not row["checks"]:
            reason = "no_trigger_check_before_end"
        else:
            reason = "trigger_gap_never_reached"
        runtime = {
            "assigned": admitted, "itt_count": 1 if admitted else 0, "env": int(env),
            "termination_reason": termination_reason, "dose": int(a.dose),
            "exposure_realized": bool(row["triggered"]), "non_exposure_reason": reason,
            "trigger_gap_m": row.get("trigger_gap_m"), "trigger": row["trigger"], "trigger_checks": row["checks"],
            "trigger_rule": "control-rate check after the physics substeps; first gap <= d_k",
            "min_gap_before_trigger_m": row["min_gap_before_trigger_m"],
            "min_sampled_gap_after_trigger_m": row["min_sampled_gap_after_trigger_m"],
            "sampled_contact_after_trigger": bool(row["sampled_contact_after_trigger"]),
            "swept_contact_latched": bool(row["contact"]),
            "first_swept_contact_interval_start_s": row["first_contact_s"],
            "swept_intervals": row["swept_intervals"], "observer_updates": row["observer_updates"],
            "last_tick": row["last_tick"], "last_time_s": max(0, row["last_tick"]) * self.dt,
            "physics_dt_s": self.dt, "slot": self.slot,
            "swept_contact_semantics": "conservative: may include uncertainty (D7 swept guard)",
            "k_equivalent": config.k_equivalent(row.get("trigger_gap_m"), self.t_avoid_s),
            # G20d (decision 7A): visible at the realized trigger step (None when not realized)
            "visible_at_trigger": None if row["trigger"] is None else bool(row["trigger"]["visible_at_trigger"]),
        }
        if self.gap_override_m is not None:
            # G20b diagnostic gap override: the trigger gap of this run replaces the dose's witnessed d_k
            low = config.smallest_trigger_gap_m(self.t_avoid_s, a.vessel_id)
            high = config.largest_trigger_gap_m(self.t_avoid_s, a.vessel_id)
            where = ("below_smallest_witnessed_gap" if self.gap_override_m < low - 1e-12 else
                     "above_largest_witnessed_gap" if self.gap_override_m > high + 1e-12 else
                     "between_witnessed_gaps")
            runtime.update(gap_override_m=self.gap_override_m, dose_gap_m=row.get("dose_gap_m"),
                           gap_override_position=where,
                           gap_override_rule=("diagnostic only (appearing.gap_m, calibration sweep): the placement "
                                              "is witnessed at the ladder's gaps (D1..D4 and the calibration rung "
                                              "D5), not at this gap"))
        reasons = None if admitted else a.reason_strings()
        stamp = make_appearing_stamp(ADMITTED if admitted else NOT_ADMITTED, a.summary(self.slot), runtime,
                                     reasons=reasons)
        row["stamp"], row["finished"] = stamp, True
        self.completed.append(stamp)
        return stamp


__all__ = ["AppearingBridge", "TERMINATION_REASONS"]
