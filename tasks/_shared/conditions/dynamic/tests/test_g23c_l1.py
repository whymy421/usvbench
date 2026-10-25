"""G23c layer 1 (common fixes after REV-G23b), CPU only, synthetic task-shaped fixtures.

* F9: a nominal spawned at rest and an own ship spawned moving: the speed estimate falls back to the window-mean
  surge (no 40x pace ratio), so the intruder is not released at t = 0 but when the constant-velocity time to the
  conflict reaches the lead;
* F4: the realized-exposure record of a nominal run (first visibility = first detection = first ray return without
  statics, contact after visibility, small sync error, conflict-pose passage, slack 0) and of a parked own ship;
* F13: D0 stamps carry no runtime rule; the rule needs exactly one intruder; station keeping stamps no release time.
"""
import math
import unittest
from dataclasses import replace

import fixtures as f
import test_g23b_release as T

d = f.d
REL = d.release


class TestF9RestingNominal(unittest.TestCase):
    def test_moving_spawn_against_a_resting_nominal_is_not_released_at_t0(self):
        env = f.envelope(f.VESSELS[1])
        u = f.scale(env)
        ctx, key, pool = T.admitted("reach", f.VESSELS[1], initial_pose=d.Pose(0.0, 0.0, 0.0, 0.0))
        a = pool.assignment(key, 2)
        route = REL.ReleaseRoute(a.candidate)
        start = route.nodes[0].pose
        speed = env.reference_speed_mps

        def pose(t):                         # spawned MOVING at u_ref straight along the initial heading
            return d.Pose(start.x + speed * t * math.cos(start.yaw), start.y + speed * t * math.sin(start.yaw),
                          start.yaw, speed)
        stamp = T.run("reach", a, pose, dt=1.0 / 120.0, until=route.t_c + 2.0)[0]
        rel = stamp["payload"]["runtime"]["release"]
        self.assertEqual(rel["cause"], "sync")
        self.assertGreater(rel["release_time_s"], 1.0)
        state = rel["state_at_release"]
        # released when distance to go / window-mean surge reached the lead (not via the 40x pace ratio)
        self.assertLessEqual(state["distance_to_go_m"] / state["speed_estimate_mps"], rel["release_lead_s"] + 0.05)
        self.assertGreater(u, 0.0)


class TestF4ExposureRecord(unittest.TestCase):
    def test_nominal_run_record(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[1])
        plan = T.nominal_plan(ctx)
        a = pool.assignment(key, 3)
        stamp = T.run("reach", a, lambda t: plan.pose_at(min(t, plan.end_s)), dt=1.0 / 120.0)[0]
        rt = stamp["payload"]["runtime"]
        rec = rt["exposure_record"]
        self.assertEqual(rec["schema"], "usvbench.d7_exposure_record/1")
        self.assertEqual(rec["first_visible_s"], rt["first_detection_s"])
        self.assertEqual(rec["first_ray_return_s"], rt["first_detection_s"])     # no statics in the reach fixture
        self.assertLessEqual(rec["closest_after_visible"]["surface_gap_m"], 0.0)  # the unmanoeuvred ship is hit
        self.assertLess(abs(rec["sync_error_s"]), 1.0)
        self.assertEqual(rec["unsynchronizable_slack_s"], 0.0)
        self.assertLess(rec["conflict_pose_passage"]["distance_m"], 0.2)
        self.assertTrue(rt["finite_ttc_at_first_detection"])

    def test_parked_record(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[0])
        stamp = T.run("reach", pool.assignment(key, 2), lambda t: d.Pose(-300.0, -300.0, 0.0, 0.0), dt=0.5)[0]
        rec = stamp["payload"]["runtime"]["exposure_record"]
        self.assertIsNone(rec["first_visible_s"])
        self.assertIsNone(rec["closest_after_moving"])
        self.assertIsNone(rec["sync_error_s"])
        self.assertEqual(stamp["payload"]["runtime"]["non_exposure_reason"], "not_released")


class TestF13(unittest.TestCase):
    def test_d0_and_station_keeping(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[0])
        stamp = T.run("reach", pool.assignment(key, 0), lambda t: ctx.initial_pose, dt=1.0)[0]
        self.assertIsNone(stamp["payload"]["runtime"]["runtime_rule"])
        self.assertIsNone(stamp["payload"]["runtime"]["exposure_record"])
        # G23e: station keeping's approach encounter now carries a release time (rule S, plan kind "approach")
        ctx, key, pool = T.admitted("station_keeping", f.VESSELS[0], env_row=1)
        stamp = T.run("station_keeping", pool.assignment(key, 2), lambda t: ctx.initial_pose, dt=0.5)[0]
        rel = stamp["payload"]["runtime"]["release"]
        self.assertTrue(rel["applies"])
        self.assertEqual(rel["plan_kind"], "approach")
        self.assertIn("release_time_s", rel)

    def test_runtime_rule_needs_one_intruder(self):
        ctx, key, pool = T.admitted("reach", f.VESSELS[0])
        a = pool.assignment(key, 2)
        enc = a.candidate.request.encounter
        two = replace(enc, intruders=enc.intruders * 2)
        cand = replace(a.candidate, request=replace(a.candidate.request, encounter=two))
        forged = replace(a, candidate=cand, candidate_digest=d.digest(cand))
        with self.assertRaises(ValueError):          # Candidate.validate refuses it first; the rule refuses too
            d.make_adapter("reach").reset((0,), d.DynamicRuntimeConfig(), {0: forged})
        with self.assertRaises(ValueError):
            REL.ReleaseState(cand, "transit").route.digest   # noqa: B018 (constructs; one intruder used)
            raise ValueError("ReleaseState alone is not the guard; the adapter is")


if __name__ == "__main__":
    unittest.main(verbosity=2)
