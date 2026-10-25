"""CPU unit tests: TTC geometry, generation, capacity, native basin, sensors, stamps, vessels.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s <this dir> -v
"""
import dataclasses
import json
import math
import os
import random
import subprocess
import sys
import unittest

import fixtures as f

d = f.d
from dynamic_cpu.sensors import collides, ray_ranges, ray_detections  # noqa: E402
from dynamic_cpu.trajectory import intruder_path_signature  # noqa: E402


def obj(x, y, vx, vy, radius=0.5, **kw):
    return d.Intruder("crossing", x, y, vx, vy, radius, 0.0, "detection_horizon", **kw)


class TestTTCAnalytic(unittest.TestCase):
    """Rectangle hull (2 m x 1 m, COM-centred) against a disk; closed-form answers."""

    def setUp(self):
        self.env = f.synthetic_envelope()
        self.own = d.Pose(0, 0, 0, 1)

    def test_head_on(self):
        # gap 10 - 1 (half length) - 0.5 (radius) = 8.5 m at 2 m/s closing
        self.assertAlmostEqual(d.ttc_linear(self.own, self.env, obj(10, 0, -1, 0)), 4.25)

    def test_crossing_from_port(self):
        # stationary hull, disk from +y at 1 m/s: 5 - 0.5 - 0.5 = 4
        self.assertAlmostEqual(d.ttc_linear(d.Pose(0, 0, 0), self.env, obj(0, 5, 0, -1)), 4.0)

    def test_corner_contact_is_rounded(self):
        # disk centre moving on the diagonal toward the corner (1, 0.5)
        self.assertAlmostEqual(d.ttc_linear(d.Pose(0, 0, 0), self.env, obj(3, 2.5, -1, -1)),
                               2 - 0.5 / math.sqrt(2))

    def test_tangent_grazing_counts(self):
        self.assertAlmostEqual(d.ttc_linear(self.own, self.env, obj(10, 1, 0, 0)), 9.0)

    def test_overlap_separating_and_horizon(self):
        self.assertEqual(d.ttc_linear(self.own, self.env, obj(0, 0, 1, 0)), 0.0)
        self.assertTrue(math.isinf(d.ttc_linear(self.own, self.env, obj(10, 0, 2, 0))))
        self.assertTrue(math.isinf(d.ttc_linear(self.own, self.env, obj(10, 0, -1, 0), horizon_s=4)))

    def test_rotated_hull_and_nonzero_clock(self):
        # heading +y, at t=2 the disk is at y=10: gap 10 - 1 - 0.5 = 8.5 at 2 m/s
        self.assertAlmostEqual(d.ttc_linear(d.Pose(0, 0, math.pi / 2, 1), self.env,
                                            obj(0, 12, 0, -1), at_s=2), 4.25)

    def test_offset_footprint_about_com(self):
        # bow extends 1.5 m ahead of the COM, stern 0.5 m: head-on gap 10 - 1.5 - 0.5 = 8
        env = f.synthetic_envelope(footprint=d.Footprint(-0.5, 1.5, -0.5, 0.5))
        self.assertAlmostEqual(d.ttc_linear(self.own, env, obj(10, 0, -1, 0)), 4.0)

    def test_resting_intruder_uses_its_current_velocity(self):
        parked = obj(10, 0, -1, 0, stop_s=0.0)
        self.assertEqual(parked.position(5.0), (10.0, 0.0))
        self.assertAlmostEqual(d.ttc_linear(self.own, self.env, parked), 8.5)

    def test_swept_tunnel_between_samples_is_caught(self):
        self.assertFalse(d.swept_interval_safe(d.Pose(-3, 0, 0), d.Pose(3, 0, 0), self.env,
                                               (d.Circle(0, 0, 0.1),), (d.Circle(0, 0, 0.1),)))
        self.assertTrue(d.swept_interval_safe(d.Pose(-3, 0, 0), d.Pose(3, 0, 0), self.env,
                                              (d.Circle(0, 3, 0.1),), (d.Circle(0, 3, 0.1),)))


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.env = f.synthetic_envelope()
        self.plan = d.build_transit_plan(d.Pose(0, 0, 0, 1), self.env, (d.Circle(60, 0, 1),), 0.25, 80)
        self.key = f.key("reach")

    def test_every_dose_delivers_its_ttc_ratio_for_every_path_type(self):
        for kind in ("crossing", "head_on", "overtaking", "drifting"):
            for dose in range(1, 5):
                with self.subTest(kind=kind, dose=dose):
                    speed = 0.1 if kind == "drifting" else 0.5
                    enc = d.generate_encounter(self.key, "reach", dose, self.env, self.plan, 30.0, 80.0,
                                               kind=kind, intruder_speed_mps=speed)
                    ref = self.plan.pose_at(enc.intruders[0].reveal_s)
                    self.assertAlmostEqual(d.ttc_linear(ref, self.env, enc.intruders[0],
                                                        at_s=enc.intruders[0].reveal_s),
                                           d.DOSE_TTC_RATIOS[dose] * self.env.t_avoid_s, places=9)

    def test_nominal_ownship_collides_exactly_at_conflict(self):
        enc = d.generate_encounter(self.key, "reach", 3, self.env, self.plan, 30.0, 80.0)
        intruder = enc.intruders[0]
        for t, expect_contact in ((29.9, False), (30.0, True)):
            gap = d.point_clearance(*intruder.position(t), self.plan.pose_at(t), self.env) - intruder.radius_m
            self.assertEqual(gap <= 1e-9, expect_contact, (t, gap))

    def test_physical_path_is_dose_invariant(self):
        sigs = {intruder_path_signature(d.generate_encounter(self.key, "reach", dose, self.env,
                                                             self.plan, 30.0, 80.0))
                for dose in range(1, 5)}
        self.assertEqual(len(sigs), 1)

    def test_zero_dose_has_no_intruder(self):
        enc = d.generate_encounter(self.key, "reach", 0, self.env, self.plan, 30.0, 80.0)
        self.assertEqual(enc.intruders, ())
        self.assertIsNone(enc.nominal_ttc_s)
        self.assertEqual(enc.meta("dose_axis"), "none")

    def test_global_rng_cannot_change_generation(self):
        args = (self.key, "reach", 2, self.env, self.plan, 30.0, 80.0)
        first = d.generate_encounter(*args)
        random.seed(9)
        [random.random() for _ in range(100)]
        self.assertEqual(first, d.generate_encounter(*args))

    def test_aim_offset_grazes_but_still_hits_at_tc(self):
        enc = d.generate_encounter(self.key, "reach", 2, self.env, self.plan, 30.0, 80.0,
                                   aim_offset_fraction=0.8)
        ref = self.plan.pose_at(enc.intruders[0].reveal_s)
        self.assertAlmostEqual(d.ttc_linear(ref, self.env, enc.intruders[0], at_s=enc.intruders[0].reveal_s),
                               d.DOSE_TTC_RATIOS[2] * self.env.t_avoid_s, places=9)
        with self.assertRaises(ValueError):
            d.generate_encounter(self.key, "reach", 2, self.env, self.plan, 30.0, 80.0, aim_offset_fraction=1.0)

    def test_window_must_be_unforced(self):
        turning = d.build_transit_plan(d.Pose(0, 0, 0, 1), self.env,
                                       (d.Circle(10, 0, 1), d.Circle(10, 20, 1)), 0.25, 80)
        t_turn = turning.target_times[0] + 0.5
        with self.assertRaises(d.PlacementError):
            d.generate_encounter(self.key, "reach", 1, self.env, turning, t_turn, 80.0)

    def test_static_geometry_is_never_crossed(self):
        # a wall disk sits on the intruder's line after the conflict: it must stop before it
        enc = d.generate_encounter(self.key, "reach", 1, self.env, self.plan, 30.0, 80.0, side=1)
        line = enc.intruders[0]
        wall = d.Circle(*line.position(45.0), 1.0)
        fitted = d.generate_encounter(self.key, "reach", 1, self.env, self.plan, 30.0, 80.0, side=1,
                                      static_obstacles=(wall,))
        obj_ = fitted.intruders[0]
        self.assertIsNotNone(obj_.stop_s)
        for t in [i * 0.5 for i in range(161)]:
            x, y = obj_.position(t)
            self.assertGreaterEqual(math.hypot(x - wall.x, y - wall.y) - wall.radius - obj_.radius_m, -1e-9)
        blocking = d.Circle(*line.position(29.0), 0.3)
        with self.assertRaises(d.PlacementError):
            d.generate_encounter(self.key, "reach", 1, self.env, self.plan, 30.0, 80.0, side=1,
                                 static_obstacles=(blocking,))


class TestCapacityNative(unittest.TestCase):
    def test_reserve_covers_base_plus_both_reserves(self):
        with self.assertRaises(ValueError):
            d.CapacityPlan(14, 1, 1, 15)
        plan = d.CapacityPlan(14, 1, 1, 16)
        self.assertEqual(plan.dynamic_slots, (14,))
        self.assertEqual(plan.appearing_slots, (15,))
        with self.assertRaises(ValueError):
            plan.validate_counts(15, 1, 1)
        with self.assertRaises(ValueError):
            plan.validate_counts(14, 2, 0)
        plan.validate_counts(14, 1, 1)

    def test_c1_finding_43_static_insertion_persists_under_native_motion(self):
        table = d.SlotState(d.CapacityPlan(1, 0, 1, 2))
        table.insert_motion(0, d.NativeMotion("translate", 3, 4, .3, 1))
        inserted = d.Circle(8, 9, .5)
        table.insert_static(1, inserted)
        for t in (0, 1, 7, 25):
            self.assertEqual(table.update(t)[1], inserted)
            self.assertEqual(table.base[1], inserted)

    def test_native_ratio_ladder_and_static_d0(self):
        env = f.synthetic_envelope()
        geometry = (d.NativeMotion("translate", 5, 0, .2, 90, span_m=6),
                    d.NativeMotion("orbit", 8, 0, .2, 90),
                    d.NativeMotion("static", 12, 0, .2, 0))
        for dose, mu in enumerate(d.NATIVE_RATIOS):
            enc = d.native_encounter(f.key("moving_basin"), dose, env, geometry, 20)
            self.assertEqual(len(enc.intruders), 3)
            self.assertAlmostEqual(enc.intruders[0].speed_mps, mu * env.top_speed_mps)
            self.assertAlmostEqual(enc.intruders[1].speed_mps, mu * env.top_speed_mps)  # tied nu
            self.assertEqual(enc.intruders[2].speed_mps, 0.0)
            if dose == 0:
                for item in enc.intruders:
                    self.assertEqual(item.position(0), item.position(19))
        fixed = d.native_encounter(f.key("moving_basin"), 3, env, geometry, 20, nu_policy="fixed")
        self.assertEqual(fixed.intruders[1].speed_mps, 0.0)

    def test_native_speed_follows_the_vessel(self):
        geometry = (d.NativeMotion("translate", 5, 0, .2, 0, span_m=6),)
        bb = d.native_encounter(f.key("moving_basin"), 2, f.envelope(f.VESSELS[0]), geometry, 20)
        wv = d.native_encounter(f.key("moving_basin"), 2, f.envelope(f.VESSELS[1]), geometry, 20)
        self.assertAlmostEqual(bb.intruders[0].speed_mps / f.envelope(f.VESSELS[0]).top_speed_mps, 0.08)
        self.assertAlmostEqual(wv.intruders[0].speed_mps / f.envelope(f.VESSELS[1]).top_speed_mps, 0.08)
        self.assertNotAlmostEqual(bb.intruders[0].speed_mps, wv.intruders[0].speed_mps)

    def test_triangle_wave_matches_env_law_and_is_continuous(self):
        # hazard_nav_env.py:1724-1725 raw = (v t + phase) mod 2 span; fold; minus span/2
        m = d.NativeMotion("translate", 0, 0, .2, 1, span_m=2)
        for t, expect in ((0.0, -1.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0), (4.0, -1.0)):
            self.assertAlmostEqual(m.position(t)[0], expect)
        self.assertLess(math.dist(m.position(2 - 1e-6), m.position(2 + 1e-6)), 1e-5)


class TestSensorsAndReveal(unittest.TestCase):
    def test_rays_see_only_visible_objects_and_collision_ignores_visibility(self):
        visible = d.ObstacleSample(0, 5, 0, 1, True, True)
        hidden = d.ObstacleSample(0, 0, 0, 1, True, False)
        self.assertEqual(ray_ranges((0, 0), (0, math.pi), 20, (visible,)), (4, 20))
        self.assertEqual(ray_ranges((0, 0), (0,), 20, (hidden,)), (20,))
        self.assertTrue(collides(d.Pose(0, 0, 0), f.synthetic_envelope(), (hidden,)))
        dets = ray_detections((0, 0), 0.0, (0.0, math.pi), (4.0, 20.0), 20.0)
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0].x, 4.0)

    def test_detection_is_physical_with_no_time_gate(self):
        sensor = d.SensorSpec(max_range_m=30.0)
        intruder = d.Intruder("crossing", 10, 0, 0, 0, 0.5, 50.0, "detection_horizon")
        evidence = d.RevealEvidence("detection_horizon", d.Pose(0, 0, 0), 5.0)
        # nominal reveal at 50 s, but an observer already within 5 m at t=1 detects it
        self.assertTrue(d.detectable(d.Pose(6, 0, 0), intruder, 1.0, evidence, sensor))
        self.assertFalse(d.detectable(d.Pose(0, 0, 0), intruder, 60.0, evidence, sensor))
        self.assertFalse(d.detectable(d.Pose(6, 0, 0), intruder, 1.0, evidence, None))
        blocked = d.RevealEvidence("detection_horizon", d.Pose(0, 0, 0), 5.0, (d.Circle(7.5, 0, 0.3),))
        self.assertFalse(d.detectable(d.Pose(6, 0, 0), intruder, 1.0, blocked, sensor))


class TestStamp(unittest.TestCase):
    def test_roundtrip_and_tamper(self):
        stamp = d.make_stamp({"a": 1, "b": [1.5, None]}, {"c": "x"})
        self.assertEqual(d.read_stamp(json.dumps(stamp)), stamp)
        changed = json.loads(json.dumps(stamp))
        changed["payload"]["runtime"]["c"] = "y"
        with self.assertRaises(ValueError):
            d.read_stamp(changed)
        with self.assertRaises(ValueError):
            d.make_stamp({"bad": math.inf}, {})


class TestVessels(unittest.TestCase):
    def test_t_avoid_is_frozen_vessel_specific_and_derived(self):
        values = {vid: f.envelope(vid).t_avoid_s for vid in f.VESSELS}
        self.assertEqual(len(set(values.values())), 3)
        env = f.envelope(f.VESSELS[0])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            env.t_avoid_s = 1.0
        parts = dict(env.t_avoid_components)
        self.assertAlmostEqual(env.t_avoid_s, env.reaction_s + min(parts["brake_s"], parts["steer_s"]))

    def test_brake_time_closed_form(self):
        from dynamic_cpu.model import brake_displacement_time, steer_displacement_time
        self.assertAlmostEqual(brake_displacement_time(1.0, 2.0, 0.2), math.sqrt(0.2))   # before stop
        self.assertAlmostEqual(brake_displacement_time(1.0, 2.0, 1.0), 1.25)             # (d + v^2/2a)/v
        # steer: small-angle lateral offset u*r*t^2/2 = d  ->  t ~ sqrt(2d/(u r))
        self.assertAlmostEqual(steer_displacement_time(1.0, 0.01, 1e-4), math.sqrt(2e-4 / 0.01), places=4)

    def test_zero_yaw_capacity_with_asymmetric_com(self):
        fwd, rev, torque = d.two_motor_capacities(40, 20, 40, 20, 0.4, 0.3)
        # T_port = k*0.3, T_stbd = k*0.4 (no yaw); starboard binds: k = 40/0.4; total = k*0.7
        self.assertAlmostEqual(fwd, 40 * 0.7 / 0.4)
        self.assertAlmostEqual(rev, 20 * 0.7 / 0.4)
        self.assertAlmostEqual(torque, 20 * 0.7)
        # E4 (CONTEXT_R3 decision 1): journal envelopes use the CENTRELINE arm, so the
        # BlueBoat COM offset no longer reduces straight capacity (D7 had 71.144 N /
        # 2.78593 m/s from its PROVISIONAL preset; D5 final: 80.4 N / 3.000 m/s,
        # D5 ANALYSIS.md:22, PARAMS.md:181,194). Zero-surge torque is unchanged.
        bb, _ = d.physics_from_spec_v2(f.spec("blueboat_v2"))
        self.assertAlmostEqual(bb.thrust_fwd_total_n, 80.41453, places=4)
        self.assertAlmostEqual(bb.terminal_speed_mps, 3.0003, places=4)
        self.assertAlmostEqual(bb.yaw_torque_zero_surge_nm, 14.477, places=2)  # D5 ANALYSIS.md:22

    def test_wamv_numbers_match_their_sources(self):
        # E4: the PROVISIONAL "current"/"classic" presets are replaced by the final D5
        # profiles opt2 (VRX J damping + thrust) and opt1 (classic), 283.3847 kg.
        cur, _ = d.physics_from_spec_v2(f.spec("wamv_vrx_v2_opt2"))
        cla, _ = d.physics_from_spec_v2(f.spec("wamv_vrx_v2_opt1"))
        self.assertAlmostEqual(cur.terminal_speed_mps, 5.2784, places=3)    # D5 ANALYSIS.md:25
        self.assertAlmostEqual(cur.steady_spin_rate_rps, 1.17864, places=4) # D5 ANALYSIS.md:25
        self.assertAlmostEqual(cla.terminal_speed_mps, 2.26, places=2)      # D5 ANALYSIS.md:23
        self.assertAlmostEqual(cla.steady_spin_rate_rps, 0.5135, places=3)  # 205.4 / 400

    def test_envelope_from_duck_typed_d5_spec(self):
        # E4: the stub now carries the FINAL VesselSpecV2 attribute names (D5
        # vehicles_v2.py:121-209); D7's draft names (com_body_m, position_body_m,
        # collision.polygon_xy_m, speeds.top_m_s) do not exist in the final registry.
        ns = type("NS", (), {})
        spec = ns()
        spec.name, spec.schema_version, spec.bow_body_axis = "dummy", 2, "+x"
        spec.mass_kg, spec.com_actor_m = 17.26, (0.0, 0.0, 0.0)
        spec.damping = ns(); spec.damping.linear = (9.1, 0, 0, 0, 0, 4.0); spec.damping.quadratic = (5.9, 0, 0, 0, 0, 6.0)
        m1, m2 = ns(), ns()
        m1.name, m1.direction_body = "port", (1.0, 0.0, 0.0)
        m2.name, m2.direction_body = "starboard", (1.0, 0.0, 0.0)
        m1.forward_limit_n, m1.reverse_limit_n, m1.position_com_m = 40.207265, 20.067924, (-0.514, 0.4077, -0.17)
        m2.forward_limit_n, m2.reverse_limit_n, m2.position_com_m = 40.207265, 20.067924, (-0.514, -0.3137, -0.17)
        spec.propulsion = ns(); spec.propulsion.motors = (m1, m2); spec.propulsion.planar_arm_m = 0.3607
        spec.propulsion.pure_yaw_limit_nm = 2 * 0.3607 * 20.067924
        spec.collision = ns()
        spec.collision.polygon_com_xy_m = ((-0.594, -0.418), (0.6068, -0.418), (0.6068, 0.512), (-0.594, 0.512))
        spec.speeds = ns(); spec.speeds.reference_m_s = 1.0
        spec.speeds.top_model_m_s = (-9.1 + math.sqrt(9.1 ** 2 + 4 * 5.9 * 2 * 40.207265)) / (2 * 5.9)
        env, derivation = d.envelope_from_spec_v2(spec, turning_radius_m=2.4)
        self.assertAlmostEqual(env.footprint.y_min_m, -0.418)
        self.assertAlmostEqual(dict(derivation)["terminal_speed_mps"], 3.0003, places=3)
        self.assertAlmostEqual(dict(derivation)["straight_thrust_fwd_n"], 2 * 40.207265)

    def test_infinite_strategy_is_recorded_as_none_and_digestable(self):
        env = f.synthetic_envelope(reference_speed_mps=1.5, max_speed_mps=1.5,
                                   t_avoid_method="min_brake_surge_steer_v1")
        self.assertIsNone(dict(env.t_avoid_components)["surge_s"])
        self.assertEqual(len(d.digest(env)), 64)

    def test_inconsistent_physics_rejected(self):
        bb, _ = d.physics_from_spec_v2(f.spec("blueboat_v2"))   # E4: final registry, not a preset
        with self.assertRaises(ValueError):
            d.envelope_from_physics(dataclasses.replace(bb, reference_speed_mps=2.5))
        with self.assertRaises(ValueError):
            d.envelope_from_physics(dataclasses.replace(bb, turning_radius_m=0.1))

    def test_no_simulator_import_and_no_bytecode(self):
        # E4: checked on the modules the PACKAGE (and the registry loader) added
        # and in a fresh interpreter, because D1's CPU harness imports torch
        # itself before running a test file (tools_d1/run_cpu_tests.py:93), which
        # made D7's whole-process check fail there without any package import.
        roots = {n.split(".")[0] for n in f.LOADED_BY_PACKAGE}
        self.assertFalse(roots & set(f.SIMULATOR_ROOTS), sorted(roots & set(f.SIMULATOR_ROOTS)))
        probe = ("import importlib.util, sys; sys.dont_write_bytecode = True; p = sys.argv[1]; "
                 "s = importlib.util.spec_from_file_location('dyn_probe', p + '/__init__.py', "
                 "submodule_search_locations=[p]); m = importlib.util.module_from_spec(s); "
                 "sys.modules['dyn_probe'] = m; s.loader.exec_module(m); "
                 "m.load_registry_by_path(sys.argv[2]); "
                 "bad = sorted(n for n in sys.modules if n.split('.')[0] in %r); "
                 "print('SIMULATOR_MODULES', bad); sys.exit(1 if bad else 0)" % (f.SIMULATOR_ROOTS,))
        result = subprocess.run([sys.executable, "-B", "-c", probe, str(f.PACKAGE), str(f.SHARED)],
                                capture_output=True, text=True, timeout=120, cwd=str(f.PACKAGE),
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SIMULATOR_MODULES []", result.stdout)
        self.assertTrue(sys.dont_write_bytecode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
