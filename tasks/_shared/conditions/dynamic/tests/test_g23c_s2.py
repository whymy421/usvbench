"""G23c layer 2: candidate runtime rule S2 = d7_reveal_sync_v1 (reveal.py), CPU only, synthetic fixtures.

* u_ref identity: for the nominal own ship the reveal time equals reveal_s(dose), the phase snaps to the pinned
  one and every sample from the reveal update on (position and visible flag) is the v2.5 (pinned_v0) sample; the
  first detection and its TTC are identical (10 transit tasks x 2 vessels x D1/D4, 1/120 s);
* hidden before the reveal: never physical, never visible; physical from the reveal update on;
* the dose at the reveal: the own ship's constant-velocity TTC at the reveal is dose x t_avoid at 0.5 x and 1.5 x
  the nominal pace (crossing and head-on) and for a 3 m lateral offset (crossing);
* a parked own ship never meets the line: "not_revealed"; station keeping keeps the pinned schedule;
* the bridge sweeps an obstacle only while it is physical at both ends and checks the newly revealed one;
* geometry: a ship crossing a head-on candidate's line at 90 deg is handled by the crossing branch (the collinear
  test is geometric) and meets the dose; a straight track that meets the line 50 m from the conflict point does
  not reveal (contact window), one that meets it 5 m away does.
"""
import math
import unittest

import fixtures as f
import test_g23b_release as T

d = f.d
REL = d.release
S2 = REL.REVEAL_RULE_VERSION
DT = 1.0 / 120.0


def offset(plan, lateral):
    def pose(t):
        p = plan.pose_at(min(t, plan.end_s))
        return d.Pose(p.x - lateral * math.sin(p.yaw), p.y + lateral * math.cos(p.yaw), p.yaw,
                      p.speed if t <= plan.end_s else 0.0)
    return pose


class TestS2Identity(unittest.TestCase):
    def test_nominal_reveal_equals_v25(self):
        for vessel in f.VESSELS[:2]:
            for task in T.TestUrefIdentity.TASKS:
                ctx, key, pool = T.admitted(task, vessel)
                plan = T.nominal_plan(ctx)
                pose = (lambda t, p=plan: p.pose_at(min(t, p.end_s)))
                for dose in (1, 4):
                    with self.subTest(vessel=vessel, task=task, dose=dose):
                        a = pool.assignment(key, dose)
                        obj = a.candidate.request.encounter.intruders[0]
                        until = a.candidate.request.encounter.meta("conflict_time_s") + 5.0
                        s, ss, _ = T.run(task, a, pose, until=until, rule=S2)
                        p, ps, _ = T.run(task, a, pose, until=until, rule=REL.PINNED_RULE_VERSION)
                        rel = s["payload"]["runtime"]["release"]
                        self.assertEqual(rel["cause"], "ttc_reveal")
                        self.assertTrue(rel["snapped_to_pinned"])
                        self.assertAlmostEqual(rel["release_time_s"], obj.reveal_s, delta=2.5e-4)
                        tick = rel["release_tick_time_s"]
                        self.assertLess(tick - obj.reveal_s, DT + 1e-9)
                        after = [(x, y) for x, y in zip(ss, ps) if x[0] >= tick]
                        self.assertTrue(after and all(x == y for x, y in after))
                        before = [x for x in ss if x[0] < tick]
                        self.assertTrue(all(not v for _, sample in before for (_, _, v) in sample))
                        srt, prt = s["payload"]["runtime"], p["payload"]["runtime"]
                        self.assertEqual(srt["first_detection_s"], prt["first_detection_s"])
                        self.assertEqual(srt["realized_ttc_at_first_detection_s"],
                                         prt["realized_ttc_at_first_detection_s"])
                        self.assertAlmostEqual(rel["ttc_at_reveal_over_t_avoid"], d.DOSE_TTC_RATIOS[dose], places=6)


class TestS2Behaviour(unittest.TestCase):
    def test_hidden_before_and_physical_after(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[1])
        plan = T.nominal_plan(ctx)
        a = pool.assignment(key, 2)
        adapter = d.make_adapter("reach")
        self.assertEqual(d.DynamicRuntimeConfig().motion_rule, REL.DEFAULT_RULE)   # G23c L3: v2 is the default
        cfg = d.DynamicRuntimeConfig(motion_rule=S2)
        adapter.reset((0,), cfg, {0: a})
        adapter.activate((0,), cfg)
        seen = []
        t = 0.0
        while t <= a.candidate.request.encounter.meta("conflict_time_s"):
            sample = adapter.update(d.RuntimeFrame(((0, t),), ((0, plan.pose_at(min(t, plan.end_s))),)), cfg)[0][0]
            seen.append((t, sample.physical, sample.visible))
            t += 0.05
        tick = adapter.episodes[0].release.reveal_tick_s
        self.assertTrue(all(not ph and not vis for tt, ph, vis in seen if tt < tick))
        self.assertTrue(all(ph for tt, ph, vis in seen if tt >= tick))
        stamp = adapter.finish(0, "collision")
        self.assertIsNone(stamp["payload"]["runtime"]["non_exposure_reason"])

    def test_dose_at_reveal_for_other_paces_and_offsets(self):
        for vessel in f.VESSELS[:2]:
            for task in ("reach", "ring"):
                ctx, key, pool = T.admitted(task, vessel)
                plan = T.nominal_plan(ctx)
                cases = [("0.5x", T.retimed(plan, 0.5)), ("1.5x", T.retimed(plan, 1.5))]
                if task == "reach":
                    cases += [("left3", offset(plan, 3.0)), ("right3", offset(plan, -3.0))]
                for name, pose in cases:
                    for dose in (1, 4):
                        with self.subTest(vessel=vessel, task=task, case=name, dose=dose):
                            stamp = T.run(task, pool.assignment(key, dose), pose, dt=DT, rule=S2)[0]
                            rel = stamp["payload"]["runtime"]["release"]
                            self.assertEqual(rel["cause"], "ttc_reveal")
                            self.assertAlmostEqual(rel["ttc_at_reveal_over_t_avoid"], d.DOSE_TTC_RATIOS[dose],
                                                   delta=DT / f.envelope(vessel).t_avoid_s + 1e-9)
                            self.assertTrue(stamp["payload"]["runtime"]["sampled_contact"])

    def test_parked_and_station_keeping(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[0])
        stamp = T.run("reach", pool.assignment(key, 2), lambda t: d.Pose(-300.0, -300.0, 0.0, 0.0), dt=0.5, rule=S2)[0]
        rt = stamp["payload"]["runtime"]
        self.assertEqual(rt["non_exposure_reason"], "not_revealed")
        self.assertFalse(rt["release"]["released"])
        # G23e: station keeping meets the intruder on the approach; an own ship parked at the start never meets it
        ctx, key, pool = T.admitted("station_keeping", f.VESSELS[0], env_row=1)
        a = pool.assignment(key, 2)
        stamp, samples, _ = T.run("station_keeping", a, lambda t: ctx.initial_pose, dt=0.5, rule=S2)
        rel = stamp["payload"]["runtime"]["release"]
        self.assertTrue(rel["applies"])
        self.assertEqual(rel["plan_kind"], "approach")
        self.assertFalse(rel["released"])
        self.assertEqual(stamp["payload"]["runtime"]["non_exposure_reason"], "not_revealed")
        self.assertTrue(all(not s[0][2] for _, s in samples))

    def test_bridge_sweeps_only_physical_obstacles(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[0])
        plan = T.nominal_plan(ctx)
        bridge = d.HazardBridge(d.make_adapter("reach"), d.DynamicRuntimeConfig(), DT, ctx.envelope.top_speed_mps)
        a = pool.assignment(key, 2)
        bridge.reset_rows((0,), {0: a})
        tick = 0
        while tick * DT <= a.candidate.request.encounter.meta("conflict_time_s") + 2.0:
            bridge.step({0: tick}, {0: plan.pose_at(min(tick * DT, plan.end_s))})
            tick += 1
        self.assertTrue(bridge.contact(0))
        first = bridge.finish(0, "collision")["payload"]["runtime"]["bridge"]["first_swept_contact_interval_start_s"]
        self.assertGreaterEqual(first, a.candidate.request.encounter.intruders[0].reveal_s - DT)

    def test_certified_segment(self):
        ctx, key, pool = T.admitted("path_hazard", f.VESSELS[0])
        cand = pool.assignment(key, 2).candidate
        obj = cand.request.encounter.intruders[0]
        lo, hi = d.certified_segment(obj, cand.request.static_obstacles, cand.request.encounter.duration_s)
        self.assertLessEqual(lo, obj.start_s)
        self.assertGreaterEqual(lo, obj.start_s - d.reveal.EXTENSION_MAX_S)
        if obj.stop_s is not None:
            self.assertEqual(hi, obj.stop_s)


def straight(x0, y0, yaw, speed):
    return lambda t: d.Pose(x0 + speed * t * math.cos(yaw), y0 + speed * t * math.sin(yaw), yaw, speed)


class TestS2Geometry(unittest.TestCase):
    """G23c: geometric collinear test (side approach to a head-on line) and the contact window."""

    def test_side_approach_to_head_on_line_uses_crossing_branch(self):
        for vessel in f.VESSELS[:2]:
            ctx, key, pool = T.admitted("ring", vessel)
            for dose in (1, 4):
                with self.subTest(vessel=vessel, dose=dose):
                    a = pool.assignment(key, dose)
                    enc = a.candidate.request.encounter
                    self.assertEqual(enc.meta("path_type"), "head_on")
                    cx, cy = enc.intruders[0].position(enc.meta("conflict_time_s"))
                    pose = straight(cx, cy - 40.0, math.pi / 2, 1.0)
                    stamp = T.run("ring", a, pose, dt=DT, rule=S2, until=min(80.0, enc.duration_s))[0]
                    rel = stamp["payload"]["runtime"]["release"]
                    self.assertEqual(rel["cause"], "ttc_reveal")
                    self.assertEqual(rel["state_at_reveal"]["branch"], "crossing")
                    self.assertAlmostEqual(rel["ttc_at_reveal_over_t_avoid"], d.DOSE_TTC_RATIOS[dose],
                                           delta=DT / f.envelope(vessel).t_avoid_s + 1e-9)
                    self.assertTrue(stamp["payload"]["runtime"]["sampled_contact"])

    def test_contact_far_from_the_conflict_point_does_not_reveal(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[0])
        a = pool.assignment(key, 2)
        enc = a.candidate.request.encounter
        obj = enc.intruders[0]
        t_c = enc.meta("conflict_time_s")
        for shift_m, revealed in ((50.0, False), (5.0, True)):
            with self.subTest(shift_m=shift_m):
                xx, yy = obj.position(t_c + shift_m / 0.5)            # the line point shift_m beyond the conflict
                pose = straight(xx - 40.0, yy, 0.0, 1.0)              # crosses the line there at 90 deg
                stamp = T.run("reach", a, pose, dt=0.05, rule=S2, until=min(70.0, enc.duration_s))[0]
                rel = stamp["payload"]["runtime"]["release"]
                self.assertEqual(rel["released"], revealed)
                if revealed:
                    self.assertEqual(rel["outside_window_updates"], 0)
                    self.assertLessEqual(abs(rel["state_at_reveal"]["contact_phase_s"] - t_c) * 0.5,
                                         d.reveal.CONTACT_WINDOW_M)
                else:
                    self.assertGreater(rel["outside_window_updates"], 0)
                    self.assertEqual(stamp["payload"]["runtime"]["non_exposure_reason"], "not_revealed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
