"""G7 item B (CPU): the E4 unforced-window tolerance (owner-approved 2026-10-01 12:5x).

A nominal-plan command counts as zero when |yaw rate| < 1e-6 rad/s and |accel| < 1e-6 m/s^2, in BOTH
places (NominalPlan.is_constant_velocity and adapters._fit_conflict). Declared bound over a window W at
speed v: |dp| <= 0.5 (v e_w + e_a) W^2 + (1/3) e_a e_w W^3 (<= 0.25 mm for WAM-V D1, 4 x 3.743 s, 1 m/s),
timing |dp| / v <= 0.25 ms. The plan's own zeroing thresholds (1e-9 rad, 1e-12 m/s) are unchanged.
"""
import math
import sys
import unittest

import fixtures as f

d = f.d
model = sys.modules[d.__name__ + ".model"]
reference = sys.modules[d.__name__ + ".reference"]
adapters = sys.modules[d.__name__ + ".adapters"]
kinematics = sys.modules[d.__name__ + ".kinematics"]
config = sys.modules[d.__name__ + ".config"]
WAMV = f.VESSELS[1]


def plan_with(commands, dt=0.25, speed=1.0):
    """NominalPlan from (accel, yaw_rate) per dt step, starting at the origin heading +x at ``speed``."""
    pose = d.Pose(0.0, 0.0, 0.0, speed)
    states = [d.TimedState(0.0, pose)]
    for i, (a, w) in enumerate(commands):
        pose = kinematics.integrate(pose, a, w, dt)
        states.append(d.TimedState((i + 1) * dt, pose, a, w))
    return d.NominalPlan(tuple(states), (), "transit", True)


class TestPredicate(unittest.TestCase):
    def test_declared_rule(self):
        self.assertEqual(model.UNFORCED_YAW_RATE_TOL_RPS, 1.0e-6)
        self.assertEqual(model.UNFORCED_ACCEL_TOL_MPS2, 1.0e-6)
        self.assertTrue(model.is_unforced_command(0.0, 0.0))
        self.assertTrue(model.is_unforced_command(9.99e-7, -4.2e-9))
        self.assertFalse(model.is_unforced_command(0.0, 1.0e-6))
        self.assertFalse(model.is_unforced_command(-1.0e-6, 0.0))
        self.assertFalse(model.is_unforced_command(0.0, 0.05))

    def test_plan_thresholds_are_unchanged(self):
        import inspect
        src = inspect.getsource(reference.build_transit_plan)
        self.assertIn("yaw_rate = 0.0 if abs(error) < 1e-9 else", src)
        self.assertIn("accel = 0.0 if abs(gap) < 1e-12 else", src)

    def test_suite_identity_names_the_rule(self):
        suite = config.DynamicSuiteConfig(enabled=True, vessel_id=WAMV)
        self.assertEqual(config.D7_SUITE_VERSION, "d7_ttc_dose_v3")
        self.assertEqual(suite.unforced_window_rule, model.UNFORCED_WINDOW_RULE)
        self.assertIn("window=" + model.UNFORCED_WINDOW_RULE, suite.to_c4_override()["definition"])
        with self.assertRaises(ValueError):
            config.DynamicSuiteConfig(enabled=True, vessel_id=WAMV, unforced_window_rule="exact_zero")


class TestWindowTest(unittest.TestCase):
    def test_tiny_corrections_pass_and_real_manoeuvres_fail_in_both_places(self):
        tiny = [(0.0, 0.0)] * 30 + [(0.0, -4.2e-9)] + [(0.0, 0.0)] * 29         # G5's ring/BlueBoat case
        plan = plan_with(tiny)
        self.assertTrue(plan.is_constant_velocity(1.0, 15.0))
        self.assertEqual(adapters._fit_conflict(plan, 12.0, 6.58, 12.0, 0.0), 12.0)
        for real in ((0.0, 0.05), (0.2, 0.0), (0.0, 1.0e-6), (1.0e-6, 0.0)):
            plan = plan_with([(0.0, 0.0)] * 30 + [real] + [(0.0, 0.0)] * 29)
            self.assertFalse(plan.is_constant_velocity(1.0, 15.0), real)
            with self.assertRaises(d.PlacementError) as ctx:
                adapters._fit_conflict(plan, 12.0, 6.58, 12.0, 0.0)
            self.assertEqual(ctx.exception.reason, "no_unforced_d1_window_before_site_end")
            # with room to slide, the window moves past the real manoeuvre (unchanged E4 behaviour)
            t = adapters._fit_conflict(plan, 12.0, 6.58, 15.0, 0.0)
            self.assertGreaterEqual(t - 6.58, 7.75)


class TestDeclaredBound(unittest.TestCase):
    def test_longest_window_bound_is_below_a_quarter_millimetre(self):
        t_avoid = f.envelope(WAMV).t_avoid_s
        window = model.DOSE_TTC_RATIOS[1] * t_avoid
        self.assertAlmostEqual(t_avoid, 3.743, delta=0.001)
        bound = model.unforced_window_position_bound_m(window, 1.0)
        self.assertLessEqual(bound, 2.5e-4)
        self.assertLessEqual(model.unforced_window_timing_bound_s(window, 1.0), 2.5e-4)
        self.assertLessEqual(model.unforced_window_timing_bound_s(window, 1.0), model.UNFORCED_TTC_TOLERANCE_S)
        self.assertLess(model.UNFORCED_TTC_TOLERANCE_S, 1.0 / 60.0)

    def test_bound_holds_numerically_on_the_worst_window(self):
        """Commands just below the tolerance on EVERY step of the longest window, all signs."""
        t_avoid = f.envelope(WAMV).t_avoid_s
        window = model.DOSE_TTC_RATIOS[1] * t_avoid
        steps = int(math.ceil(window / 0.01))
        dt = window / steps
        e = 0.999999e-6
        bound = model.unforced_window_position_bound_m(window, 1.0)
        worst = 0.0
        for sa in (-1.0, 1.0):
            for sw in (-1.0, 1.0):
                pose = d.Pose(0.0, 0.0, 0.0, 1.0)
                for _ in range(steps):
                    pose = kinematics.integrate(pose, sa * e, sw * e, dt)
                dev = math.hypot(pose.x - window * 1.0, pose.y)
                worst = max(worst, dev)
                self.assertLessEqual(dev, bound * (1.0 + 1e-9))
                self.assertLessEqual(abs(pose.yaw), e * window * (1.0 + 1e-12))
        # not vacuous: along- and cross-track terms each reach their half of the bound, so the worst
        # deviation is their hypot, 1/sqrt(2) of the declared (summed) bound
        self.assertGreater(worst, 0.7 * bound)
        self.assertLessEqual(worst, 2.5e-4)


class TestFixtureLadders(unittest.TestCase):
    def test_every_task_still_admits_its_ladder_and_validates(self):
        for vessel in f.VESSELS[:2]:
            env = f.envelope(vessel)
            for task in d.TASKS:
                with self.subTest(vessel=vessel, task=task):
                    ctx = f.context(task, env)
                    key = f.key(task, env=1 if task == "station_keeping" else 0)
                    ladder = d.build_ladder(ctx, key, f.suite(env))
                    for dose, cand in enumerate(ladder):
                        if (vessel, task, dose) == (f.VESSELS[0], "station_keeping", 1):   # G23e: by design
                            self.assertIsInstance(cand, d.adapters.NotApplicableByDesign)
                            continue
                        self.assertIsInstance(cand, d.Candidate, cand)
                        cand.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
