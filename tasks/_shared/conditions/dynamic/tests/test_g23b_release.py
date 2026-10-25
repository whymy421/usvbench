"""G23b: runtime rule S (arrival-synchronized release of the pinned D7 intruder), CPU only.

Owner decision 2026-10-03 (NC 09, G23 option S). Checks, on the synthetic task-shaped fixtures of
fixtures.py (both journal vessels):
* u_ref identity: an own ship ON the nominal plan releases at exactly t_hold and every object sample
  from t_hold on equals the pinned trajectory bit for bit (first detection and realized TTC too);
* determinism (same observer sequence -> identical stamps) and dose invariance of the release route;
* no detection before the release, for nominal, fast, slow, parked and hold-point-hunting observers
  (the detection guard releases a resting object at the update it becomes detectable);
* parked own ship -> never released, non-exposure reason "not_released", object stays at H;
* station keeping (G23e): the approach encounter is released like a transit route (plan kind "approach");
* docking from rest: the accelerating-pace refinement releases earlier than the base rule;
* moving basin and D0 carry no release; motion_rule "pinned_v0" restores v2.5 controller independence;
* the stamp round trip (read_stamp) and the release block fields.
"""
import json
import math
import unittest
from dataclasses import replace

import fixtures as f

d = f.d
BB, WAMV = f.VESSELS[0], f.VESSELS[1]
REL = d.release
DT = 1.0 / 120.0


def admitted(task, vessel, env_row=0, **overrides):
    env = f.envelope(vessel)
    ctx = f.context(task, env, **overrides)
    key = f.key(task, env=env_row)
    pool = d.admit_ladders([(key, d.build_ladder(ctx, key, f.suite(env)))])
    assert pool.keys_admitted == 1, [(r.dose, r.reason) for r in pool.rejections]
    return ctx, key, pool


def nominal_plan(ctx):
    profile = d.PROFILES[ctx.task]
    return d.adapters._nominal_plan(profile, ctx, f.suite(ctx.envelope))


def run(task, assignment, pose_fn, dt=DT, rule=None, until=None):
    # G23c: these are rule-S tests; S is the default in layer 1 and a diagnostic option in layer 2
    cfg = d.DynamicRuntimeConfig(motion_rule=REL.RUNTIME_RULE_VERSION if rule is None else rule)
    adapter = d.make_adapter(task)
    row = assignment.candidate.request.encounter.key.env
    adapter.reset((row,), cfg, {row: assignment})
    adapter.activate((row,), cfg)
    end = assignment.candidate.request.encounter.duration_s if until is None else until
    samples = []
    i = 0
    while i * dt <= end + 1e-12:
        t = i * dt
        out = adapter.update(d.RuntimeFrame(((row, t),), ((row, pose_fn(t)),)), cfg)[row]
        samples.append((t, tuple((s.x, s.y, s.visible) for s in out)))
        i += 1
    return adapter.finish(row, "timeout"), samples, adapter


def retimed(plan, factor):
    """The nominal path traversed `factor` times faster (speeds scaled by factor)."""
    def pose(t):
        p = plan.pose_at(min(plan.end_s, factor * t))
        return d.Pose(p.x, p.y, p.yaw, p.speed * factor if factor * t <= plan.end_s else 0.0)
    return pose


class TestUrefIdentity(unittest.TestCase):
    TASKS = ("reach", "path_following", "path_hazard", "scatter", "ring", "fortress", "forced_crossing",
             "iceberg", "harbour", "docking")

    def test_nominal_own_ship_reproduces_the_pinned_motion_bit_for_bit(self):
        for vessel in (BB, WAMV):
            for task in self.TASKS:
                ctx, key, pool = admitted(task, vessel)
                plan = nominal_plan(ctx)
                pose = (lambda t, p=plan: p.pose_at(min(t, p.end_s)))
                for dose in (1, 4):
                    with self.subTest(vessel=vessel, task=task, dose=dose):
                        a = pool.assignment(key, dose)
                        until = a.candidate.request.encounter.meta("conflict_time_s") + 5.0
                        s_stamp, s_samples, _ = run(task, a, pose, until=until)
                        p_stamp, p_samples, _ = run(task, a, pose, rule=REL.PINNED_RULE_VERSION, until=until)
                        rel = s_stamp["payload"]["runtime"]["release"]
                        self.assertEqual(rel["release_time_s"], rel["t_hold_s"])
                        self.assertEqual(rel["cause"], "sync")
                        t_hold = rel["t_hold_s"]
                        after = [(x, y) for x, y in zip(s_samples, p_samples) if x[0] >= t_hold]
                        self.assertTrue(after)
                        self.assertTrue(all(x == y for x, y in after))
                        srt, prt = s_stamp["payload"]["runtime"], p_stamp["payload"]["runtime"]
                        self.assertEqual(srt["first_detection_s"], prt["first_detection_s"])
                        self.assertEqual(srt["realized_ttc_at_first_detection_s"],
                                         prt["realized_ttc_at_first_detection_s"])
                        # the dose: TTC / t_avoid at the first detection within one update interval
                        ratio = d.DOSE_TTC_RATIOS[dose]
                        self.assertTrue(srt["finite_ttc_at_first_detection"])
                        self.assertAlmostEqual(srt["realized_ttc_at_first_detection_over_t_avoid"], ratio,
                                               delta=2 * DT / ctx.envelope.t_avoid_s)
                        # before t_hold the object rests at the hold point H = pinned position(t_hold)
                        obj = a.candidate.request.encounter.intruders[0]
                        hx, hy = obj.position(t_hold)
                        for t, sample in s_samples:
                            if t < t_hold:
                                self.assertEqual(sample[0][:2], (hx, hy))

    def test_route_is_dose_invariant_and_constants_declared(self):
        for vessel in (BB, WAMV):
            for task in ("reach", "harbour", "docking"):
                ctx, key, pool = admitted(task, vessel)
                digests = {REL.ReleaseRoute(pool.assignment(key, k).candidate).digest for k in (1, 2, 3, 4)}
                self.assertEqual(len(digests), 1, (vessel, task))
        block = REL.rule_block()
        self.assertEqual(block["version"], "d7_release_sync_v1")
        self.assertEqual(block["constants"]["release_margin_s"], 2.0)
        self.assertEqual(block["constants"]["task_release_margin_s"], {"docking": 0.0})
        self.assertEqual(block["constants"]["accelerating_pace_tasks"], ["docking"])


class TestReleaseBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.key, cls.pool = admitted("reach", WAMV)
        cls.plan = nominal_plan(cls.ctx)

    def a(self, dose):
        return self.pool.assignment(self.key, dose)

    def test_determinism(self):
        pose = retimed(self.plan, 1.7)
        one = run("reach", self.a(2), pose, dt=0.05)[0]
        two = run("reach", self.a(2), pose, dt=0.05)[0]
        self.assertEqual(json.dumps(one, sort_keys=True), json.dumps(two, sort_keys=True))

    def test_fast_own_ship_is_released_early_and_still_hit(self):
        for factor in (1.5, 2.0):
            stamp, samples, _ = run("reach", self.a(3), retimed(self.plan, factor), dt=0.05)
            rt = stamp["payload"]["runtime"]
            rel = rt["release"]
            self.assertLess(rel["release_time_s"], rel["t_hold_s"])
            self.assertTrue(rt["finite_ttc_at_first_detection"], factor)
            self.assertTrue(rt["sampled_contact"], factor)          # the unmanoeuvred ship is hit
            self.assertLess(rt["realized_ttc_at_first_detection_over_t_avoid"], d.DOSE_TTC_RATIOS[3])
            self.assertTrue(rel["predicted_closest_approach"]["predicted_contact"])

    def test_no_detection_before_release_for_any_observer(self):
        obj = self.a(1).candidate.request.encounter.intruders[0]
        route = REL.ReleaseRoute(self.a(1).candidate)
        hx, hy = route.hold_point
        # creeps at H while its surge reading is zero (pace ratio at the floor: never a sync release)
        hunter = (lambda t: d.Pose(hx - 40.0 + 1.5 * t, hy, 0.0, 0.0))
        observers = {"nominal": retimed(self.plan, 1.0), "fast": retimed(self.plan, 2.0),
                     "slow": retimed(self.plan, 0.5), "hunter": hunter,
                     "parked": (lambda t: d.Pose(-200.0, -200.0, 0.0, 0.0))}
        for dose in (1, 2, 3, 4):
            for name, pose in observers.items():
                with self.subTest(dose=dose, observer=name):
                    stamp, samples, _ = run("reach", self.a(dose), pose, dt=0.05)
                    rt = stamp["payload"]["runtime"]
                    rel = rt["release"]
                    if rt["first_detection_s"] is not None:
                        self.assertIsNotNone(rel["release_time_s"])
                        self.assertGreaterEqual(rt["first_detection_s"], rel["release_time_s"])
                    if name == "hunter":
                        self.assertEqual(rel["cause"], "detection_guard")
                        self.assertEqual(rt["first_detection_s"], rel["release_time_s"])
                    for t, sample in samples:
                        if sample[0][2]:
                            self.assertGreaterEqual(t, rel["release_time_s"])
        self.assertGreater(obj.speed_bound_mps, 0.0)

    def test_parked_own_ship_never_releases(self):
        stamp, samples, adapter = run("reach", self.a(2), lambda t: d.Pose(-200.0, -200.0, 0.0, 0.0), dt=0.25)
        rt = stamp["payload"]["runtime"]
        self.assertEqual(rt["non_exposure_reason"], "not_released")
        self.assertFalse(rt["release"]["released"])
        self.assertIsNone(rt["release"]["cause"])
        self.assertFalse(rt["finite_ttc_at_first_detection"])
        hx, hy = rt["release"]["hold_point"]
        self.assertTrue(all(s[0][:2] == (hx, hy) for _, s in samples))

    def test_release_time_is_update_rate_robust(self):
        pose = retimed(self.plan, 2.0)
        fine = run("reach", self.a(1), pose, dt=DT)[0]["payload"]["runtime"]["release"]
        coarse = run("reach", self.a(1), pose, dt=0.1)[0]["payload"]["runtime"]["release"]
        self.assertEqual(fine["cause"], coarse["cause"])
        self.assertLessEqual(abs(fine["release_time_s"] - coarse["release_time_s"]), 0.1 + 1e-9)

    def test_stamp_round_trip_and_fields(self):
        stamp = run("reach", self.a(3), retimed(self.plan, 1.5), dt=0.05)[0]
        self.assertEqual(d.read_stamp(json.dumps(stamp)), stamp)
        rt = stamp["payload"]["runtime"]
        self.assertEqual(rt["runtime_rule"]["version"], REL.RUNTIME_RULE_VERSION)
        for field in ("t_hold_s", "release_time_s", "release_tick_time_s", "cause", "released", "hold_point",
                      "release_lead_s", "conflict_time_s", "route_digest", "state_at_release",
                      "predicted_closest_approach"):
            self.assertIn(field, rt["release"])
        self.assertIn("pace_ratio", rt["release"]["state_at_release"])
        self.assertEqual(rt["finite_ttc_at_first_detection"], rt["realized_ttc_at_first_detection_s"] is not None)
        self.assertAlmostEqual(rt["realized_ttc_at_first_detection_over_t_avoid"],
                               rt["realized_ttc_at_first_detection_s"] / stamp["payload"]["assignment"]["t_avoid_s"])
        self.assertIn("d7_release_sync_v1", stamp["payload"]["semantics"]["schedule"])
        tampered = json.loads(json.dumps(stamp))
        tampered["payload"]["runtime"]["release"]["release_time_s"] = 0.0
        with self.assertRaises(ValueError):
            d.read_stamp(tampered)

    def test_pinned_rule_is_controller_independent(self):
        a = self.a(2)
        one = run("reach", a, retimed(self.plan, 1.0), dt=0.25, rule=REL.PINNED_RULE_VERSION)
        two = run("reach", a, lambda t: d.Pose(-50.0, -50.0, 2.0, 0.0), dt=0.25, rule=REL.PINNED_RULE_VERSION)
        obj = a.candidate.request.encounter.intruders[0]
        self.assertEqual([s[1][0][:2] for s in one[1]], [s[1][0][:2] for s in two[1]])
        self.assertEqual([s[1][0][:2] for s in one[1]], [obj.position(s[0]) for s in one[1]])
        self.assertEqual(one[0]["payload"]["runtime"]["runtime_rule"],
                         {"version": REL.PINNED_RULE_VERSION, "diagnostic_only": True})
        with self.assertRaises(ValueError):
            d.DynamicRuntimeConfig(motion_rule="free_running")


class TestStationDockingBasin(unittest.TestCase):
    def test_station_keeping_approach_is_released_like_a_transit_route(self):
        # G23e: the retired hold design kept the pinned schedule (release.applies false); the approach encounter
        # is released by rule S like any transit encounter: the nominal own ship releases at exactly t_hold
        for vessel in (BB, WAMV):
            ctx, key, pool = admitted("station_keeping", vessel, env_row=1)
            plan = nominal_plan(ctx)
            for dose in ((2, 4) if vessel == BB else (1, 4)):
                a = pool.assignment(key, dose)
                obj = a.candidate.request.encounter.intruders[0]
                stamp, samples, _ = run("station_keeping", a, lambda t, p=plan: p.pose_at(min(t, p.end_s)), dt=0.25)
                rel = stamp["payload"]["runtime"]["release"]
                self.assertTrue(rel["applies"])
                self.assertEqual(rel["plan_kind"], "approach")
                self.assertEqual(rel["release_time_s"], rel["t_hold_s"])
                after = [(t, s) for t, s in samples if t >= rel["t_hold_s"]]
                self.assertEqual([s[0][:2] for _, s in after], [obj.position(t) for t, _ in after])
                parked = run("station_keeping", a, lambda t: d.Pose(-300.0, 0.0, 0.0, 0.0), dt=0.25)[0]
                self.assertEqual(parked["payload"]["runtime"]["non_exposure_reason"], "not_released")

    def test_docking_from_rest_accelerating_refinement_releases_earlier(self):
        env = f.envelope(WAMV)
        u = f.scale(env)
        # a short approach from rest (t_c ~ 29.6 s, t_hold ~ 12.7 s): the 2 u_ref ship enters the release
        # distance while it is still accelerating, the case G23 found 0.5-1.9 m clear on the real pools
        ctx, key, pool = admitted("docking", WAMV, initial_pose=d.Pose(-12 * u, 0.0, 0.0, 0.0))
        a = pool.assignment(key, 1)
        route = REL.ReleaseRoute(a.candidate)
        path = route.nodes
        accel, cruise = env.max_accel_mps2, 2.0 * env.reference_speed_mps

        def pose(t):          # accelerate from rest at the envelope bound to 2 u_ref along the (straight) route
            ramp = cruise / accel
            s = 0.5 * accel * t * t if t <= ramp else 0.5 * accel * ramp * ramp + cruise * (t - ramp)
            v = min(cruise, accel * t)
            start = path[0].pose
            return d.Pose(start.x + s * math.cos(start.yaw), start.y + s * math.sin(start.yaw), start.yaw, v)
        refined_rt = run("docking", a, pose, dt=0.05)[0]["payload"]["runtime"]
        refined = refined_rt["release"]
        saved = REL.TASK_RELEASE_MARGIN_S, REL.ACCELERATING_PACE_TASKS
        try:            # the base rule of the other transit tasks: margin 2 s, no accelerating pace
            REL.TASK_RELEASE_MARGIN_S, REL.ACCELERATING_PACE_TASKS = {}, frozenset()
            base_rt = run("docking", a, pose, dt=0.05)[0]["payload"]["runtime"]
            base = base_rt["release"]
        finally:
            REL.TASK_RELEASE_MARGIN_S, REL.ACCELERATING_PACE_TASKS = saved
        self.assertEqual(base["cause"], "sync")
        self.assertEqual(refined["cause"], "accelerating_pace")
        self.assertEqual((refined["margin_s"], base["margin_s"]), (0.0, 2.0))
        self.assertLess(refined["release_time_s"], base["release_time_s"])
        # ... and turns the near miss of the base rule into a conflict of the unmanoeuvred ship
        self.assertGreater(base_rt["min_sampled_surface_clearance_m"], 0.0)
        self.assertTrue(refined_rt["sampled_contact"])
        self.assertTrue(refined_rt["finite_ttc_at_first_detection"])
        # the nominal docking ship (accelerating from rest at the same bound) still releases exactly at t_hold
        plan = nominal_plan(ctx)
        nominal = run("docking", a, lambda t: plan.pose_at(min(t, plan.end_s)), dt=DT,
                      until=route.t_c + 1.0)[0]["payload"]["runtime"]["release"]
        self.assertEqual(nominal["release_time_s"], nominal["t_hold_s"])
        self.assertEqual(nominal["cause"], "sync")

    def test_moving_basin_and_d0_carry_no_release(self):
        ctx, key, pool = admitted("moving_basin", BB)
        stamp = run("moving_basin", pool.assignment(key, 3), lambda t: ctx.initial_pose, dt=1.0)[0]
        self.assertIsNone(stamp["payload"]["runtime"]["release"])
        self.assertEqual(stamp["payload"]["runtime"]["non_exposure_reason"], "native_motion_axis")
        self.assertIsNone(stamp["payload"]["runtime"]["runtime_rule"])      # G23c F13: no rule applies
        ctx, key, pool = admitted("reach", BB)
        stamp = run("reach", pool.assignment(key, 0), lambda t: ctx.initial_pose, dt=1.0)[0]
        self.assertIsNone(stamp["payload"]["runtime"]["release"])
        self.assertFalse(stamp["payload"]["runtime"]["finite_ttc_at_first_detection"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
