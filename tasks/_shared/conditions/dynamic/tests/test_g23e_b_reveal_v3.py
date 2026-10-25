"""G23e layer B: rule d7_reveal_sync_v3 (REV-G23d M1-M4, m1, m2, m5), CPU only, synthetic fixtures.

* u_ref identity: the nominal own ship reveals at reveal_s(dose) and then shows the pinned motion bit for bit
  (10 transit tasks x 2 vessels x D1 / D4, 1/120 s), cause ttc_reveal, dose_achieved;
* M1: a ship holding 1.0 m/s along a route whose nominal slows in its bends gets V within 1 % of its surge;
* M2: under white surge noise (0.05 m/s at 120 Hz) the v3 estimate is close to the window mean's noise level and
  far below the v2 endpoint correction's; the named switch selects the alternatives;
* M3: an accelerating ship is stamped accelerating, its independent measure differs from the constant-velocity
  value, and it is clamped_accelerating whenever that measure misses the dose; the measure for the nominal
  equals the dose;
* M4: a reveal whose independent TTC is a fraction of the dose after an earlier meeting is late_reveal (clamped);
  no unclamped ttc_reveal / cpa_reveal below 0.95 x dose for crabbing ships on head-on lines;
* m1: release.cpa exactly for a cpa_reveal; m2 / m5: d7_exposure invariants and the dose_undefined count.
"""
import importlib.util
import json
import math
import random
import sys
import unittest

import fixtures as f
import test_g23b_release as T

d = f.d
REL = d.release
RV = sys.modules[d.RevealState.__module__]
V3 = REL.V3_RULE_VERSION
DT = 1.0 / 120.0
BB, WAMV = f.VESSELS[0], f.VESSELS[1]
TRANSIT = ("reach", "path_following", "path_hazard", "scatter", "ring", "fortress", "forced_crossing",
           "iceberg", "harbour", "docking")


def d7x():
    if "g23e_b_d7x" not in sys.modules:
        spec = importlib.util.spec_from_file_location("g23e_b_d7x", f.PACKAGE.parents[3] / "scripts" / "d7_exposure.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["g23e_b_d7x"] = module
        spec.loader.exec_module(module)
    return sys.modules["g23e_b_d7x"]


def along(plan, speed_fn, yaw_offset=0.0, surge_scale=1.0):
    """Observer on the nominal PATH with its own speed profile speed_fn(t) (distance integrated at 1/120 s)."""
    length = plan.path_length_m
    state = {"t": 0.0, "s": 0.0}

    def pose(t):
        while state["t"] + 1e-12 < t:
            step = min(DT, t - state["t"])
            state["s"] += speed_fn(state["t"]) * step
            state["t"] += step
        frac = min(1.0, state["s"] / length)
        p = plan.pose_at(plan.time_at_fraction(frac))
        return d.Pose(p.x, p.y, p.yaw + yaw_offset, speed_fn(t) * surge_scale)
    return pose


class TestIdentity(unittest.TestCase):
    def test_nominal_own_ship_reproduces_the_pinned_motion(self):
        for vessel in (BB, WAMV):
            for task in TRANSIT:
                ctx, key, pool = T.admitted(task, vessel)
                plan = T.nominal_plan(ctx)
                for dose in (1, 4):
                    a = pool.assignment(key, dose)
                    obj = a.candidate.request.encounter.intruders[0]
                    t_c = a.candidate.request.encounter.meta("conflict_time_s")
                    stamp, samples, _ = T.run(task, a, lambda t, p=plan: p.pose_at(min(t, p.end_s)), rule=V3,
                                              until=t_c + 0.5)
                    rel = stamp["payload"]["runtime"]["release"]
                    with self.subTest(vessel=vessel, task=task, dose=dose):
                        self.assertEqual(stamp["payload"]["runtime"]["runtime_rule"]["version"], V3)
                        self.assertEqual(rel["release_time_s"], obj.reveal_s)
                        self.assertEqual(rel["cause"], "ttc_reveal")
                        self.assertTrue(rel["dose_achieved"])
                        self.assertFalse(rel["dose_undefined"])
                        self.assertFalse(rel["accelerating_at_reveal"])
                        for t, s in samples:
                            if t >= rel["release_tick_time_s"]:
                                self.assertEqual(s[0][:2], obj.position(t))
                            else:
                                self.assertFalse(s[0][2])


class TestSpeedEstimate(unittest.TestCase):
    def state(self, task="path_following", vessel=BB, dose=1):
        ctx, key, pool = T.admitted(task, vessel)
        a = pool.assignment(key, dose)
        return ctx, d.RevealState(a.candidate, "transit", v3=True), T.nominal_plan(ctx)

    def test_m1_constant_speed_through_nominal_speed_dips(self):
        ctx, st, plan = self.state()
        speeds = [s.pose.speed for s in plan.states]
        self.assertLess(min(v for v in speeds[len(speeds) // 4:] if v > 0), 0.99)   # the nominal does slow down
        pose = along(plan, lambda t: 1.0)
        worst = 0.0
        for i in range(1, int(0.8 * plan.end_s / DT)):
            v, _ = st._speed_v3(i * DT, pose(i * DT))
            worst = max(worst, abs(v - 1.0))
        self.assertLess(worst, 0.01)

    def test_m2_noise_and_switch(self):
        ctx, st, plan = self.state("reach")
        rng = random.Random(7)
        pose0 = along(plan, lambda t: 1.0)
        values, raw = [], []
        for i in range(1, 2400):
            p = pose0(i * DT)
            noisy = d.Pose(p.x, p.y, p.yaw, 1.0 + rng.gauss(0.0, 0.05))
            v, _ = st._speed_v3(i * DT, noisy)
            if i > 120:
                values.append(v - 1.0)
                raw.append(noisy.speed - 1.0)
        sd = math.sqrt(sum(x * x for x in values) / len(values))
        sd_raw = math.sqrt(sum(x * x for x in raw) / len(raw))
        self.assertLess(sd, 0.25 * sd_raw)              # window-mean level (about 0.13 sigma), not single samples
        self.assertEqual(RV.V3_SPEED_ESTIMATOR, "mean_unless_trend_v1")
        self.assertEqual(set(RV.V3_SPEED_ESTIMATORS), {"mean_unless_trend_v1", "window_mean", "instantaneous",
                                                        "ls_endpoint"})
        saved = RV.V3_SPEED_ESTIMATOR
        try:
            RV.V3_SPEED_ESTIMATOR = "instantaneous"
            ctx, st2, plan = self.state("reach")
            p = along(plan, lambda t: 1.0)(1.0)
            self.assertEqual(st2._speed_v3(1.0, d.Pose(p.x, p.y, p.yaw, 1.234))[0], 1.234)
        finally:
            RV.V3_SPEED_ESTIMATOR = saved

    def test_trend_uses_the_latest_sample(self):
        ctx, st, plan = self.state("reach")
        pose = along(plan, lambda t: 0.2 + 0.8 * t)
        for i in range(1, 121):
            v, accel = st._speed_v3(i * DT, pose(i * DT))
        self.assertAlmostEqual(v, 0.2 + 0.8 * 1.0, places=9)
        self.assertAlmostEqual(accel, 0.8, places=6)


class TestIndependentAndLabels(unittest.TestCase):
    def test_m3_accelerating_ship_is_clamped_or_achieved_by_the_independent_measure(self):
        seen = 0
        for vessel in (BB, WAMV):
            env = f.envelope(vessel)
            for task in ("reach", "path_following", "scatter"):
                ctx, key, pool = T.admitted(task, vessel)
                plan = T.nominal_plan(ctx)
                for dose in (1, 2):
                    a = pool.assignment(key, dose)
                    t_c = a.candidate.request.encounter.meta("conflict_time_s")
                    top = env.top_speed_mps
                    pose = along(plan, lambda t, top=top: min(0.9 * top, 0.3 + 0.12 * t))
                    stamp, _, _ = T.run(task, a, pose, rule=V3, until=t_c + 5.0)
                    rel = stamp["payload"]["runtime"]["release"]
                    if not rel["released"] or rel["cause"] not in ("ttc_reveal", "clamped_accelerating"):
                        continue
                    capped = rel["state_at_reveal"]["observer"]["speed"] >= 0.9 * top - 1e-9
                    with self.subTest(vessel=vessel, task=task, dose=dose, capped=capped):
                        if capped:                        # speed levelled off at the cap: not accelerating
                            continue
                        self.assertTrue(rel["accelerating_at_reveal"])
                        self.assertAlmostEqual(rel["accel_at_reveal_mps2"], 0.12, places=6)
                        if rel["ttc_independent_at_reveal_s"] is not None and rel["ttc_cv_at_reveal_s"] is not None:
                            self.assertLess(rel["ttc_independent_at_reveal_s"], rel["ttc_cv_at_reveal_s"])
                        self.assertEqual(rel["cause"] == "clamped_accelerating", not rel["dose_achieved"])
                        seen += 1
        self.assertGreater(seen, 2)

    def test_m4_late_reveal_after_an_earlier_meeting(self):
        ctx, key, pool = T.admitted("reach", BB)
        plan = T.nominal_plan(ctx)
        a = pool.assignment(key, 1)
        enc = a.candidate.request.encounter
        t_c, dose_s = enc.meta("conflict_time_s"), 4.0 * enc.t_avoid_s
        st = d.RevealState(a.candidate, "transit", v3=True)
        t0 = t_c - dose_s - 3.0
        for t in (t0 - DT, t0):
            self.assertFalse(st.observe(t, plan.pose_at(t)))      # met the line, TTC above the dose
        t1 = t_c - 0.5 * dose_s                                   # next update: half the dose left
        self.assertTrue(st.observe(t1, plan.pose_at(t1)))
        rel = st.stamp(a.candidate.request.envelope.footprint)
        self.assertEqual(rel["cause"], "late_reveal")
        self.assertTrue(rel["clamped"])
        self.assertLess(rel["ttc_independent_at_reveal_s"], 0.95 * dose_s)

    def test_m4_crab_has_no_unclamped_late_reveal_and_m1_cpa_only_for_cpa_reveal(self):
        runs = 0
        for vessel in (BB, WAMV):
            for task in ("ring", "forced_crossing", "iceberg"):
                ctx, key, pool = T.admitted(task, vessel)
                plan = T.nominal_plan(ctx)
                for beta in (math.radians(10.0), math.radians(15.0), 0.0):
                    for lateral in (0.0, 1.0, 3.0):
                        a = pool.assignment(key, 1)
                        t_c = a.candidate.request.encounter.meta("conflict_time_s")

                        def pose(t, p=plan, b=beta, e=lateral):
                            q = p.pose_at(min(t, p.end_s))
                            return d.Pose(q.x - e * math.sin(q.yaw), q.y + e * math.cos(q.yaw), q.yaw + b,
                                          q.speed * math.cos(b))
                        stamp, _, _ = T.run(task, a, pose, rule=V3, dt=1.0 / 60.0, until=t_c + 2.0)
                        rel = stamp["payload"]["runtime"]["release"]
                        runs += 1
                        with self.subTest(vessel=vessel, task=task, beta=beta, lateral=lateral):
                            self.assertEqual(rel["cause"] == "cpa_reveal", rel["cpa"] is not None)
                            if rel["released"] and rel["cause"] in ("ttc_reveal", "cpa_reveal"):
                                self.assertGreaterEqual(rel["ttc_at_reveal_s"], 0.95 * rel["dose_ttc_s"] - 1e-9)
                                v = rel["ttc_independent_at_reveal_s"]
                                self.assertTrue(v is None or v >= 0.95 * rel["dose_ttc_s"] - 1e-9)
                            d7x().runtime_exposure(json.loads(json.dumps(stamp)))
        self.assertEqual(runs, 54)


class TestStampJson(unittest.TestCase):
    def test_exact_trend_stamps_null_t(self):
        # regression: a noise-free exact speed ramp has zero residual, |t| = inf, which JSON cannot hold
        ctx, key, pool = T.admitted("reach", BB)
        plan = T.nominal_plan(ctx)
        a = pool.assignment(key, 2)
        st = d.RevealState(a.candidate, "transit", v3=True)
        t_c = a.candidate.request.encounter.meta("conflict_time_s")
        t, revealed = 0.0, False
        while not revealed and t < t_c:
            revealed = st.observe(t, plan.pose_at(t))
            t += DT
        self.assertTrue(revealed)
        st.trend = (0.1, math.inf, 0.1, False)
        st.independent = st._independent_v3(st.reveal_tick_s, plan.pose_at(st.reveal_tick_s), False, 0.0)
        rel = st.stamp(a.candidate.request.envelope.footprint)
        self.assertIsNone(rel["speed_trend_t"])
        json.dumps(rel, allow_nan=False)


class TestValidator(unittest.TestCase):
    def test_m2_invariants_and_m5_undefined_count(self):
        e = d7x()
        self.assertEqual(e.V3_RULE_VERSION, V3)
        self.assertEqual(e.DOSE_TTC_RATIOS, tuple(d.DOSE_TTC_RATIOS))
        self.assertEqual(e.DOSE_ACHIEVED_TOL, RV.DOSE_ACHIEVED_TOL)
        self.assertEqual(set(e.V3_CLAMPED_CAUSES), set(RV.V3_CLAMPED_CAUSES))
        self.assertEqual(tuple(e.RULE_CAUSES[V3]), tuple(RV.V3_CAUSES))
        ctx, key, pool = T.admitted("reach", BB)
        plan = T.nominal_plan(ctx)
        a = pool.assignment(key, 2)
        stamp, _, _ = T.run("reach", a, lambda t: plan.pose_at(min(t, plan.end_s)), rule=V3, dt=1.0 / 60.0)
        good = json.loads(json.dumps(stamp))
        self.assertTrue(e.runtime_exposure(good)["dose_achieved"])
        for mutate in (lambda r: r.__setitem__("dose_ttc_s", r["dose_ttc_s"] * 1.1),
                       lambda r: r.__setitem__("clamped", True),
                       lambda r: r.__setitem__("cpa", {"lateral_offset_m": 1.0}),
                       lambda r: r.__setitem__("dose_achieved_tol", 0.5),
                       lambda r: r.__setitem__("dose_undefined", True),
                       lambda r: r.__setitem__("ttc_independent_at_reveal_s", 0.5 * r["dose_ttc_s"])):
            bad = json.loads(json.dumps(stamp))
            mutate(bad["payload"]["runtime"]["release"])
            with self.assertRaises(e.ExposureError):
                e.runtime_exposure(bad)
        parked = T.run("reach", a, lambda t: d.Pose(-300.0, -300.0, 0.0, 0.0), rule=V3, dt=0.5)[0]
        summary = e.exposure_summary([good, json.loads(json.dumps(parked))], 2)
        self.assertEqual(summary["rule"], V3)
        self.assertEqual(summary["dose_undefined"], 0)
        self.assertIn("dose_undefined_share", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
