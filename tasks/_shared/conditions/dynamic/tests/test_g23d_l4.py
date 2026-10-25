"""G23d layer 4 (REV-G23c fixes) of rule d7_reveal_sync_v2, CPU only, synthetic fixtures.

* H2: the stamp carries the independent constant-velocity TTC at the reveal tick and dose_achieved; for the nominal
  own ship at u_ref it equals the dose (identity) and the reveal is unchanged;
* H1: a head-on parallel offset beyond the touch lane but inside the corridor is revealed at time-to-CPA = dose
  (cause cpa_reveal, stamped lateral offset, independent time-to-CPA within 5 %); beyond the corridor it is not
  revealed;
* H3: a first meeting with the line already inside the dose is late_first_meeting (clamped); every
  clamped_accelerating reveal misses the dose on the independent measure;
* H6: a hidden (never revealed) intruder overlapping a parked own ship is no stamp sample;
* H9: the nominal-speed property is asserted at construction and stamped;
* the v2 speed estimate is the calibrated finite-window bias correction (rule constants).
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


def head_on(vessel):
    ctx, key, pool = T.admitted("ring", vessel)
    return ctx, key, pool, T.nominal_plan(ctx)


class TestIndependentDose(unittest.TestCase):
    def test_nominal_dose_achieved_and_constants(self):
        block = REL.rule_block_for(V2)
        self.assertEqual(block["constants"]["speed_estimate"], "bias_corrected_calibrated_v1")
        self.assertEqual(block["constants"]["dose_achieved_tol"], 0.05)
        for vessel in f.VESSELS[:2]:
            for task in ("reach", "ring", "path_following"):
                ctx, key, pool = T.admitted(task, vessel)
                plan = T.nominal_plan(ctx)
                for dose in (1, 4):
                    with self.subTest(vessel=vessel, task=task, dose=dose):
                        a = pool.assignment(key, dose)
                        stamp = T.run(task, a, lambda t, p=plan: p.pose_at(min(t, p.end_s)), rule=V2,
                                      until=a.candidate.request.encounter.meta("conflict_time_s") + 3.0)[0]
                        rel = stamp["payload"]["runtime"]["release"]
                        self.assertEqual(rel["cause"], "ttc_reveal")
                        self.assertEqual(rel["independent_measure"], "ttc_linear")
                        self.assertTrue(rel["dose_achieved"])
                        self.assertAlmostEqual(rel["ttc_cv_at_reveal_over_t_avoid"], d.DOSE_TTC_RATIOS[dose],
                                               delta=DT / f.envelope(vessel).t_avoid_s + 1e-6)
                        self.assertLessEqual(rel["nominal_max_excess_over_leg_speed_mps"],
                                             d.reveal.NOMINAL_LEG_SPEED_TOL_MPS)


class TestHeadOnOffset(unittest.TestCase):
    def test_cpa_reveal_inside_the_corridor_only(self):
        for vessel in f.VESSELS[:2]:
            ctx, key, pool, plan = head_on(vessel)
            a = pool.assignment(key, 2)
            state = d.RevealState(a.candidate, "transit", hybrid=True)
            env = f.envelope(vessel)
            touch = env.footprint.half_extents[1] + a.candidate.request.encounter.intruders[0].radius_m
            inside = 0.5 * (touch + state.w_c)                 # beyond the touch lane, inside the corridor
            for lateral, revealed in ((inside, True), (-inside, True), (state.w_c + 0.5, False)):
                with self.subTest(vessel=vessel, lateral=lateral):
                    stamp = T.run("ring", a, S.offset(plan, lateral), dt=DT, rule=V2)[0]
                    rel = stamp["payload"]["runtime"]["release"]
                    self.assertEqual(rel["released"], revealed)
                    if not revealed:
                        self.assertEqual(stamp["payload"]["runtime"]["non_exposure_reason"], "not_revealed")
                        continue
                    self.assertEqual(rel["cause"], "cpa_reveal")
                    self.assertFalse(rel["clamped"])
                    self.assertEqual(rel["independent_measure"], "time_to_cpa_constant_velocity")
                    self.assertAlmostEqual(abs(rel["cpa"]["lateral_offset_m"]), abs(lateral), delta=0.05)
                    self.assertAlmostEqual(rel["ttc_at_reveal_over_t_avoid"], d.DOSE_TTC_RATIOS[2],
                                           delta=DT / env.t_avoid_s + 1e-9)
                    self.assertTrue(rel["dose_achieved"])
                    self.assertFalse(stamp["payload"]["runtime"]["sampled_contact"])   # a near miss abeam


class TestLabels(unittest.TestCase):
    def test_late_first_meeting(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[1])          # WAM-V: T_d(D1) = 15 s
        a = pool.assignment(key, 1)
        enc = a.candidate.request.encounter
        obj = enc.intruders[0]
        xc, yc = obj.position(enc.meta("conflict_time_s"))
        t_turn = 20.0

        def pose(t):                       # parallel to the line 4 m off, then straight at it (first meeting ~4 s)
            if t < t_turn:
                return d.Pose(xc - 4.0, yc - (t_turn - t) * 1.0, math.pi / 2, 1.0)
            return d.Pose(xc - 4.0 + (t - t_turn), yc, 0.0, 1.0)
        stamp = T.run("reach", a, pose, dt=0.05, rule=V2, until=min(60.0, enc.duration_s))[0]
        rel = stamp["payload"]["runtime"]["release"]
        self.assertEqual(rel["cause"], "late_first_meeting")
        self.assertTrue(rel["clamped"])
        self.assertLess(rel["ttc_cv_at_reveal_s"], d.reveal.LATE_MEETING_FRACTION * rel["dose_ttc_s"])

    def test_accelerating_label_consistency(self):
        for vessel in f.VESSELS[:2]:
            ctx, key, pool = T.admitted("reach", vessel)
            plan = T.nominal_plan(ctx)
            accel = 0.8 * f.envelope(vessel).max_accel_mps2

            def pose(t, p=plan):           # hard acceleration along the nominal path (speed a t)
                s = 0.5 * accel * t * t
                q = p.pose_at(min(p.end_s, s / max(1e-9, f.envelope(vessel).reference_speed_mps)))
                return d.Pose(q.x, q.y, q.yaw, accel * t)
            for dose in (1, 4):
                with self.subTest(vessel=vessel, dose=dose):
                    rel = T.run("reach", pool.assignment(key, dose), pose, dt=DT, rule=V2)[0]["payload"]["runtime"][
                        "release"]
                    if rel["cause"] == "clamped_accelerating":
                        self.assertFalse(rel["dose_achieved"])
                        self.assertTrue(rel["clamped"])
                    if rel["cause"] == "ttc_reveal":
                        self.assertTrue(rel["dose_achieved"] or rel["ttc_cv_at_reveal_s"] is None
                                        or abs(rel["state_at_reveal"]["observer_accel_mps2"] or 0.0) <= 0.05)


class TestHiddenSamplesAndProperty(unittest.TestCase):
    def test_hidden_overlap_is_no_sample(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[1])
        a = pool.assignment(key, 2)
        x0, y0 = a.candidate.request.encounter.intruders[0].position(0.0)
        stamp = T.run("reach", a, lambda t: d.Pose(x0, y0, 0.0, 0.0), dt=0.5, rule=V2, until=20.0)[0]
        rt = stamp["payload"]["runtime"]
        self.assertFalse(rt["release"]["released"])
        self.assertFalse(rt["sampled_contact"])                   # L3 counted the hidden overlap
        self.assertIsNone(rt["min_sampled_surface_clearance_m"])

    def test_nominal_speed_property_is_asserted(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[0])
        cand = pool.assignment(key, 2).candidate
        saved = d.reveal.NOMINAL_LEG_SPEED_TOL_MPS
        try:
            d.reveal.NOMINAL_LEG_SPEED_TOL_MPS = -1.0             # any candidate now violates it
            with self.assertRaises(ValueError):
                d.RevealState(cand, "transit", hybrid=True)
        finally:
            d.reveal.NOMINAL_LEG_SPEED_TOL_MPS = saved
        self.assertLessEqual(d.RevealState(cand, "transit", hybrid=True).nominal_excess, saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
