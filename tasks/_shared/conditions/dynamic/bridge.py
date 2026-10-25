"""CPU reference of the env-side D7 glue (one bridge per env instance, many rows).

This is the executable specification of INTEGRATION_SPEC.md B04-B09. A later
vectorised torch version must match it on recorded traces (parity test);
it is not meant for 64 envs x 120 Hz in production Python.

Clock: an integer physics tick per row, time = tick * physics_dt_s, reset to
zero only for reset rows. Never env 0's clock, never route progress.

Call pattern per physics substep (INTEGRATION_SPEC.md B06):
  * start of every _apply_action: step({env: tick}, {env: COM pose now});
  * _get_dones: step() once more with the post-substep tick so the final
    substep of the control step is swept before rewards/termination.
A tick that was already consumed returns the cached samples and is NOT swept
or counted twice, so the next _apply_action cannot double count.

Swept contact: between consecutive consumed ticks the hull rectangle and each
D7-owned obstacle are checked over the whole interval with
geometry.swept_interval_safe, inflated by the rotation bound and a declared
centre-deviation bound (own top speed + obstacle speed bound) * interval / 2,
so a fast crossing between samples cannot tunnel. The result is conservative
(False may include uncertainty) and is recorded, not silently converted.
"""

import math

from .adapters import RuntimeFrame
from .geometry import swept_interval_safe
from .kinematics import wrap
from .model import Circle, Pose, check_number


class HazardBridge:
    def __init__(self, adapter, runtime_cfg, physics_dt_s, own_speed_bound_mps):
        check_number("physics_dt_s", physics_dt_s, positive=True)
        check_number("own_speed_bound_mps", own_speed_bound_mps, positive=True)
        self.adapter = adapter
        self.cfg = runtime_cfg
        self.dt = float(physics_dt_s)
        self.own_speed_bound = float(own_speed_bound_mps)
        self.rows = {}

    def reset_rows(self, env_ids, assignments):
        """Load admitted assignments; previous episodes must already be finished."""
        ids = tuple(int(i) for i in env_ids)
        self.adapter.reset(ids, self.cfg, assignments)
        self.adapter.activate(ids, self.cfg)
        for env in ids:
            candidate = assignments[env].candidate
            self.rows[env] = {"last_tick": -1, "prev": None, "cache": None,
                              "footprint": candidate.request.envelope.footprint,
                              "speed_bounds": tuple(o.speed_bound_mps for o in
                                                    candidate.request.encounter.intruders),
                              "contact": False, "first_contact_s": None,
                              "swept_intervals": 0, "finished": False}

    def step(self, ticks, poses):
        """ticks: {env: int}, poses: {env: Pose (COM, env-local)} -> {env: samples}."""
        out = {}
        for env, tick in ticks.items():
            row = self.rows[env]
            if row["finished"]:
                raise ValueError("row finished; reset it before stepping")
            if type(tick) is not int or tick < 0:
                raise ValueError("tick must be a nonnegative integer")
            if tick == row["last_tick"]:
                out[env] = row["cache"]
                continue
            if tick < row["last_tick"]:
                raise ValueError("physics tick went backwards without a reset")
            pose = poses[env]
            if not isinstance(pose, Pose):
                raise TypeError("pose must be a Pose")
            time_s = tick * self.dt
            samples = self.adapter.update(RuntimeFrame(((env, time_s),), ((env, pose),)), self.cfg)[env]
            prev = row["prev"]
            if prev is not None:
                prev_tick, prev_pose, prev_samples = prev
                interval = (tick - prev_tick) * self.dt
                obstacle_bound = max(row["speed_bounds"], default=0.0)
                deviation = (self.own_speed_bound + obstacle_bound) * interval * 0.5
                # G23c L2: an obstacle is swept only over intervals where it is physical at BOTH ends; one that
                # appears (S2 reveal) is checked at the new endpoint (zero-length interval at the current pose)
                before = {s.slot: s for s in prev_samples if s.physical}
                after = {s.slot: s for s in samples if s.physical}
                both = [slot for slot in after if slot in before]
                start = [Circle(before[k].x, before[k].y, before[k].radius_m) for k in both]
                end = [Circle(after[k].x, after[k].y, after[k].radius_m) for k in both]
                fresh = [Circle(after[k].x, after[k].y, after[k].radius_m) for k in after if k not in before]
                angular = abs(wrap(pose.yaw - prev_pose.yaw))
                row["swept_intervals"] += 1
                unsafe = bool(start) and not swept_interval_safe(prev_pose, pose, row["footprint"], start, end,
                                                                 0.0, angular, deviation)
                if fresh and not swept_interval_safe(pose, pose, row["footprint"], fresh, fresh, 0.0, 0.0, 0.0):
                    unsafe = True
                if unsafe:
                    if not row["contact"]:
                        row["first_contact_s"] = prev_tick * self.dt
                    row["contact"] = True
            row["prev"] = (tick, pose, samples)
            row["last_tick"] = tick
            row["cache"] = samples
            out[env] = samples
        return out

    def contact(self, env):
        """Swept-contact latch for the env's collision/safety logic."""
        return self.rows[env]["contact"]

    def finish(self, env, termination_reason):
        row = self.rows[env]
        extra = {"swept_contact_latched": row["contact"],
                 "first_swept_contact_interval_start_s": row["first_contact_s"],
                 "swept_intervals": row["swept_intervals"],
                 "last_tick": row["last_tick"], "physics_dt_s": self.dt,
                 "swept_contact_semantics": "conservative: may include uncertainty"}
        stamp = self.adapter.finish(env, termination_reason, extra=extra)
        row["finished"] = True
        return stamp
