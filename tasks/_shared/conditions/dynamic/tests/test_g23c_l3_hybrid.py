"""G23c layer 3: rule d7_reveal_sync_v2 = S2 + hybrid corridor predictor (reveal.py), CPU only, synthetic fixtures.

* the default runtime rule is v2; S2 v1, rule S and pinned_v0 are diagnostic (rule blocks);
* vessel-scaled constants: w_c = r + 3 hy, sin theta_c = w_c / (u_ref 4 t_avoid), window = |v| 4 t_avoid + w_c;
* u_ref identity: for the nominal own ship the reveal equals reveal_s(dose), snapped, and every sample from the
  reveal update on is the v2.5 (pinned_v0) sample (10 transit tasks x 2 vessels x D1/D4, 1/120 s);
* in the corridor (0.5 x / 1.5 x pace, WAM-V 3 m offset) the reveal is predicted by the hybrid branch, meets the
  dose and the unmanoeuvred ship makes contact; a BlueBoat 3 m offset is outside its corridor (constant velocity);
* a side approach to a head-on line is outside the corridor and handled by S2's crossing branch.
"""
import math
import unittest

import fixtures as f
import test_g23b_release as T
import test_g23c_s2 as S

d = f.d
REL = d.release
V2 = REL.HYBRID_RULE_VERSION
DT = 1.0 / 120.0


class TestRules(unittest.TestCase):
    def test_default_and_diagnostic_rules(self):
        # G23e layer B: d7_reveal_sync_v3 is the default; v2 (this file's rule) is diagnostic, behaviour unchanged
        self.assertEqual(REL.DEFAULT_RULE, REL.V3_RULE_VERSION)
        self.assertEqual(d.DynamicRuntimeConfig().motion_rule, REL.V3_RULE_VERSION)
        self.assertIn(REL.REVEAL_RULE_VERSION, REL.DIAGNOSTIC_RULES)
        self.assertIn(V2, REL.DIAGNOSTIC_RULES)
        self.assertTrue(REL.rule_block_for(REL.REVEAL_RULE_VERSION)["diagnostic_only"])
        block = REL.rule_block_for(V2)
        self.assertEqual(block["version"], V2)
        self.assertTrue(block["diagnostic_only"])
        self.assertNotIn("diagnostic_only", REL.rule_block_for(REL.V3_RULE_VERSION))
        self.assertEqual(block["constants"]["corridor_clearance_beams"], 1.0)
        self.assertEqual(REL.rule_block_for(REL.REVEAL_RULE_VERSION)["constants"]["contact_window_m"], 10.0)

    def test_vessel_scaled_constants(self):
        for vessel in f.VESSELS[:2]:
            ctx, key, pool = T.admitted("reach", vessel)
            a = pool.assignment(key, 2)
            state = d.RevealState(a.candidate, "transit", hybrid=True)
            env = a.candidate.request.envelope
            hy, r = env.footprint.half_extents[1], a.candidate.request.encounter.intruders[0].radius_m
            w_c = r + 3.0 * hy
            with self.subTest(vessel=vessel):
                self.assertAlmostEqual(state.w_c, w_c, places=12)
                self.assertAlmostEqual(math.sin(state.theta_c),
                                       min(1.0, w_c / (env.reference_speed_mps * 4.0 * env.t_avoid_s)), places=12)
                v = a.candidate.request.encounter.intruders[0]
                self.assertAlmostEqual(state.window_m, math.hypot(v.vx, v.vy) * 4.0 * env.t_avoid_s + w_c, places=12)
                self.assertEqual(d.RevealState(a.candidate, "transit").window_m, 10.0)


class TestV2Identity(unittest.TestCase):
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
                        s, ss, _ = T.run(task, a, pose, until=until, rule=V2)
                        p, ps, _ = T.run(task, a, pose, until=until, rule=REL.PINNED_RULE_VERSION)
                        rel = s["payload"]["runtime"]["release"]
                        self.assertEqual(rel["rule"], V2)
                        self.assertEqual(rel["cause"], "ttc_reveal")
                        self.assertTrue(rel["snapped_to_pinned"])
                        self.assertTrue(rel["state_at_reveal"]["branch"].startswith("hybrid_"))
                        self.assertAlmostEqual(rel["release_time_s"], obj.reveal_s, delta=2.5e-4)
                        tick = rel["release_tick_time_s"]
                        after = [(x, y) for x, y in zip(ss, ps) if x[0] >= tick]
                        self.assertTrue(after and all(x == y for x, y in after))
                        before = [x for x in ss if x[0] < tick]
                        self.assertTrue(all(not v for _, sample in before for (_, _, v) in sample))
                        self.assertEqual(s["payload"]["runtime"]["first_detection_s"],
                                         p["payload"]["runtime"]["first_detection_s"])


class TestV2Corridor(unittest.TestCase):
    def test_paces_and_offsets(self):
        for vessel in f.VESSELS[:2]:
            ctx, key, pool = T.admitted("reach", vessel)
            plan = T.nominal_plan(ctx)
            w_c = d.RevealState(pool.assignment(key, 2).candidate, "transit", hybrid=True).w_c
            cases = [("0.5x", T.retimed(plan, 0.5), True), ("1.5x", T.retimed(plan, 1.5), True),
                     ("left3", S.offset(plan, 3.0), 3.0 <= w_c), ("right3", S.offset(plan, -3.0), 3.0 <= w_c)]
            for name, pose, inside in cases:
                for dose in (1, 4):
                    with self.subTest(vessel=vessel, case=name, dose=dose):
                        stamp = T.run("reach", pool.assignment(key, dose), pose, dt=DT, rule=V2)[0]
                        rel = stamp["payload"]["runtime"]["release"]
                        self.assertEqual(rel["cause"], "ttc_reveal")
                        self.assertEqual(rel["state_at_reveal"]["branch"].startswith("hybrid_"), inside)
                        self.assertAlmostEqual(rel["ttc_at_reveal_over_t_avoid"], d.DOSE_TTC_RATIOS[dose],
                                               delta=DT / f.envelope(vessel).t_avoid_s + 1e-9)
                        self.assertTrue(stamp["payload"]["runtime"]["sampled_contact"])

    def test_side_approach_to_head_on_line_is_outside_the_corridor(self):
        for vessel in f.VESSELS[:2]:
            ctx, key, pool = T.admitted("ring", vessel)
            a = pool.assignment(key, 4)
            enc = a.candidate.request.encounter
            cx, cy = enc.intruders[0].position(enc.meta("conflict_time_s"))
            stamp = T.run("ring", a, S.straight(cx, cy - 40.0, math.pi / 2, 1.0), dt=DT, rule=V2,
                          until=min(80.0, enc.duration_s))[0]
            rel = stamp["payload"]["runtime"]["release"]
            with self.subTest(vessel=vessel):
                self.assertEqual(rel["state_at_reveal"]["branch"], "crossing")
                self.assertIsNone(rel["state_at_reveal"]["corridor"])
                self.assertTrue(stamp["payload"]["runtime"]["sampled_contact"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
