"""E4 CPU tests: final-registry adapter, centreline capacities, journal ray sensor
flag, dose table, deterministic parallel admission and hot-loop identity.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s <this dir> -v
Vessel numbers are read from D5's final registry (tasks/_shared/vehicles_v2.py);
pinned values cite D5 ANALYSIS.md / PARAMS.md and E4 DOSE_TABLE_E4.md.
"""
import dataclasses
import math
import os
import subprocess
import sys
import unittest

import fixtures as f

d = f.d
from dynamic_cpu import feasibility as feas  # noqa: E402
from dynamic_cpu.adapters import _fit_conflict  # noqa: E402
from dynamic_cpu.dose_table import REQUIRED_PATHS, head_on_range_closed_form  # noqa: E402

BB, WAMV, WAMV_OPT2 = f.VESSELS
UNSENSED_TASKS = ("reach", "path_following", "station_keeping", "docking")


class TestFinalRegistryAdapter(unittest.TestCase):
    """envelope_from_spec_v2 against the REAL final D5 registry (every profile)."""

    def test_every_registry_profile_builds_with_journal_defaults(self):
        self.assertEqual(set(f.VESSELS) - set(f.REGISTRY.VEHICLES_V2), set())
        self.assertEqual(f.REGISTRY.DEFAULT_WAMV_PROFILE, d.DEFAULT_WAMV_PROFILE)
        for name, spec in f.REGISTRY.VEHICLES_V2.items():
            with self.subTest(profile=name):
                env, derivation = d.envelope_from_spec_v2(spec)
                values = dict(derivation)
                self.assertEqual(env.vessel_id, name)
                self.assertEqual(env.reference_speed_mps, 1.0)               # decision 14 (PENDING)
                self.assertEqual(values["turning_radius_m"], env.footprint.length_m)   # R = L (D4)
                self.assertEqual(values["speed_cap_fraction"], 0.8)          # no 0.95 special case
                self.assertAlmostEqual(env.max_speed_mps, 0.8 * values["terminal_speed_mps"], places=12)
                self.assertEqual(env.top_speed_mps, spec.speeds.top_model_m_s)
                self.assertAlmostEqual(env.max_yaw_rate_rps, 1.0 / env.footprint.length_m, places=12)
                self.assertIn("CONTEXT_R3 decision 14, PENDING", env.provenance)
                self.assertIn("D5 final registry", env.provenance)

    def test_centreline_capacities_are_motor_sums_and_pure_yaw(self):
        for name, spec in f.REGISTRY.VEHICLES_V2.items():
            with self.subTest(profile=name):
                physics, inputs = d.physics_from_spec_v2(spec)
                port, stbd = spec.propulsion.motors
                self.assertEqual(physics.thrust_fwd_total_n, port.forward_limit_n + stbd.forward_limit_n)
                self.assertEqual(physics.thrust_rev_total_n, port.reverse_limit_n + stbd.reverse_limit_n)
                arm = spec.propulsion.planar_arm_m
                self.assertAlmostEqual(physics.yaw_torque_zero_surge_nm,
                                       2 * arm * min(port.forward_limit_n, port.reverse_limit_n), places=9)
                self.assertAlmostEqual(physics.yaw_torque_zero_surge_nm, spec.propulsion.pure_yaw_limit_nm,
                                       places=9)
                self.assertAlmostEqual(physics.terminal_speed_mps, spec.speeds.top_model_m_s, places=12)
        bb, _ = d.physics_from_spec_v2(f.spec(BB))
        # D5 ANALYSIS.md:22 (80.4 / 40.1 N, 3.00 m/s, pure yaw 14.5 N m); PARAMS.md:181,194
        self.assertAlmostEqual(bb.thrust_fwd_total_n, 80.41453, places=5)
        self.assertAlmostEqual(bb.thrust_rev_total_n, 40.13585, places=5)
        self.assertAlmostEqual(bb.terminal_speed_mps, 3.00033, places=5)
        self.assertAlmostEqual(bb.yaw_torque_zero_surge_nm, 14.47700, places=5)
        # D7's COM-lever convention (arms 0.4077 / 0.3137 m about the COM, D5 PARAMS.md:184)
        # would have given 71.144 N; the centreline decision removes that 11.5 % loss.
        com_fwd, _, com_torque = d.two_motor_capacities(40.20726499999999, 20.067924449378328,
                                                        40.20726499999999, 20.067924449378328,
                                                        0.3607 + 0.04699999839067459,
                                                        0.3607 - 0.04699999839067459)
        self.assertAlmostEqual(com_fwd, 71.144, places=3)
        self.assertAlmostEqual(com_torque, bb.yaw_torque_zero_surge_nm, places=9)
        o1, _ = d.physics_from_spec_v2(f.spec(WAMV))
        o2, _ = d.physics_from_spec_v2(f.spec(WAMV_OPT2))
        self.assertAlmostEqual(o1.thrust_fwd_total_n, 485.3, places=1)     # D5 ANALYSIS.md:23
        self.assertAlmostEqual(o1.thrust_rev_total_n, 200.0, places=9)
        self.assertAlmostEqual(o1.yaw_torque_zero_surge_nm, 205.43, places=2)   # D5 PARAMS.md:141
        self.assertAlmostEqual(o2.thrust_fwd_total_n, 4707.07, places=2)    # D5 ANALYSIS.md:25
        self.assertAlmostEqual(o2.yaw_torque_zero_surge_nm, 2054.27, places=2)

    def test_footprint_is_the_registry_polygon_about_the_com(self):
        for name in f.VESSELS:
            spec = f.spec(name)
            env = f.envelope(name)
            xs = [p[0] for p in spec.collision.polygon_com_xy_m]
            ys = [p[1] for p in spec.collision.polygon_com_xy_m]
            self.assertEqual((env.footprint.x_min_m, env.footprint.x_max_m, env.footprint.y_min_m,
                              env.footprint.y_max_m), (min(xs), max(xs), min(ys), max(ys)))
        bb = f.envelope(BB).footprint
        self.assertAlmostEqual(bb.beam_m, 0.93, places=9)                  # D5 PARAMS.md:192
        self.assertAlmostEqual(bb.length_m, 1.200825846, places=8)         # R = L = 1.201 m (D4 :73)
        self.assertAlmostEqual(f.envelope(WAMV).footprint.length_m, 5.37719612, places=7)  # 5.377 m

    def test_final_schema_is_required_and_checked(self):
        spec = f.spec(BB)
        for old in ("com_body_m",):
            self.assertFalse(hasattr(spec, old))
        self.assertFalse(hasattr(spec.motors[0], "position_body_m"))
        self.assertFalse(hasattr(spec.collision, "polygon_xy_m"))
        self.assertFalse(hasattr(spec.speeds, "top_m_s"))
        with self.assertRaisesRegex(ValueError, "COM"):
            d.envelope_from_spec_v2(dataclasses.replace(spec, com_actor_m=(0.0, -0.047, 0.0)))
        port, stbd = spec.propulsion.motors
        lopsided = dataclasses.replace(spec.propulsion, motors=(
            port, dataclasses.replace(stbd, position_com_m=(stbd.position_com_m[0], stbd.position_com_m[1] - 0.05,
                                                            stbd.position_com_m[2]))))
        with self.assertRaisesRegex(ValueError, "mirror"):
            d.envelope_from_spec_v2(dataclasses.replace(spec, propulsion=lopsided))
        with self.assertRaisesRegex(ValueError, "pure_yaw"):
            d.envelope_from_spec_v2(dataclasses.replace(
                spec, propulsion=dataclasses.replace(spec.propulsion, pure_yaw_limit_nm=20.0)))
        with self.assertRaisesRegex(ValueError, "top_model"):
            d.envelope_from_spec_v2(dataclasses.replace(
                spec, speeds=dataclasses.replace(spec.speeds, top_model_m_s=2.9)))
        with self.assertRaisesRegex(ValueError, "schema-2"):
            d.envelope_from_spec_v2(dataclasses.replace(spec, schema_version=1))

    def test_parameters_and_the_retired_wamv_special_case(self):
        opt1 = f.spec(WAMV)
        env, derivation = d.envelope_from_spec_v2(opt1, turning_radius_m=10.0, reference_speed_mps=0.8)
        self.assertEqual(dict(derivation)["turning_radius_m"], 10.0)
        self.assertAlmostEqual(env.max_yaw_rate_rps, 0.08, places=12)
        self.assertIn("supplied by caller", env.provenance)
        # D7's 0.95 cap existed only for the old 2.0 m/s WAM-V reference: at 0.8 of the
        # 2.259 m/s terminal speed (cap 1.807) that reference is now rejected, not patched.
        with self.assertRaisesRegex(ValueError, "exceeds witness cap"):
            d.envelope_from_spec_v2(opt1, reference_speed_mps=2.0)

    def test_t_avoid_values_and_components(self):
        expected = {BB: (1.64502, 1.14502, 1.52868), WAMV: (3.74273, 3.24273, 5.29147),
                    WAMV_OPT2: (3.10512, 2.60512, 5.29147)}
        for name, (t_avoid, brake, steer) in expected.items():
            with self.subTest(vessel=name):
                env = f.envelope(name)
                parts = dict(env.t_avoid_components)
                self.assertAlmostEqual(env.t_avoid_s, t_avoid, places=5)
                self.assertAlmostEqual(parts["brake_s"], brake, places=5)
                self.assertAlmostEqual(parts["steer_s"], steer, places=5)
                self.assertEqual(parts["reaction_s"], 0.5)
                self.assertAlmostEqual(parts["clearance_m"], env.beam_m, places=12)
                self.assertEqual(env.t_avoid_s, 0.5 + min(brake_and_steer(parts)))


def brake_and_steer(parts):
    return (parts["brake_s"], parts["steer_s"])


class TestJournalRaySensorFlag(unittest.TestCase):
    """CONTEXT_R3 decision 14 (PENDING): journal_ray_sensor_all_tasks, default True."""

    def test_flag_default_and_stamping(self):
        self.assertTrue(d.DynamicSuiteConfig().journal_ray_sensor_all_tasks)
        self.assertIn("journal_rays=1", f.suite(f.envelope(BB), dose=2).to_c4_override()["definition"])
        self.assertIn("journal_rays=0", f.suite(f.envelope(BB), dose=2, journal_ray_sensor_all_tasks=False)
                      .to_c4_override()["definition"])
        with self.assertRaises(ValueError):
            d.DynamicSuiteConfig(journal_ray_sensor_all_tasks=1)
        self.assertEqual((d.JOURNAL_RAY_SENSOR.ray_count, d.JOURNAL_RAY_SENSOR.max_range_m), (36, 30.0))

    def test_legacy_sensor_table(self):
        self.assertEqual(set(d.LEGACY_RANGE_SENSOR), set(d.TASKS))
        self.assertEqual({t for t, s in d.LEGACY_RANGE_SENSOR.items() if s != "rays_36_30m"},
                         set(UNSENSED_TASKS))
        self.assertEqual(d.LEGACY_RANGE_SENSOR["docking"], "berth_walls_only")

    def test_unsensed_tasks_become_distinct_sensed_experiments(self):
        for vessel in (BB, WAMV):
            env = f.envelope(vessel)
            for task in UNSENSED_TASKS:
                with self.subTest(vessel=vessel, task=task):
                    ctx = f.context(task, env, sensor=None)
                    ladder = d.build_ladder(ctx, f.key(task, env=1), f.suite(env))
                    # G23e: station keeping BlueBoat D1 is not applicable by design
                    ladder = tuple(c for c in ladder if not isinstance(c, d.adapters.NotApplicableByDesign))
                    self.assertTrue(all(isinstance(c, d.Candidate) for c in ladder), ladder)
                    for cand in ladder:
                        self.assertIs(cand.sensor, d.JOURNAL_RAY_SENSOR)
                        self.assertEqual(cand.sensor_source, "journal_ray_sensor_all_tasks")
                        cand.validate()
                    ranges = [c.reveal_evidence.detection_range_m for c in ladder[1:]]
                    self.assertTrue(all(a > b for a, b in zip(ranges, ranges[1:])), ranges)
                    self.assertTrue(all(c.perceivable for c in ladder[1:]))
                    self.assertLessEqual(ranges[0], 30.0)
                    closed = d.build_ladder(ctx, f.key(task, env=1),
                                            f.suite(env, journal_ray_sensor_all_tasks=False))
                    self.assertIsInstance(closed[0], d.Candidate)
                    na = ["not_applicable_by_design"] if (vessel, task) == (BB, "station_keeping") else []   # G23e
                    self.assertEqual([c.reason for c in closed[1:]], na + ["task_has_no_existing_range_sensor"] * (4 - len(na)))

    def test_existing_sensor_wins_and_admission_runtime_stamp(self):
        env = f.envelope(BB)
        ctx = f.context("reach", env)                    # RAYS = the task's existing fan
        self.assertEqual(d.resolve_sensor(ctx, f.suite(env)), (f.RAYS, "task_existing_sensor"))
        unsensed = f.context("reach", env, sensor=None)
        key = f.key("reach")
        pool = d.admit_ladders([(key, d.build_ladder(unsensed, key, f.suite(env)))])
        self.assertEqual(pool.keys_admitted, 1, [(r.dose, r.reason) for r in pool.rejections])
        assignment = pool.assignment(key, 2)
        plan = d.build_transit_plan(unsensed.initial_pose, env, (unsensed.goal,), 0.25, unsensed.deadline_s)
        adapter = d.make_adapter("reach")
        cfg = d.DynamicRuntimeConfig()
        adapter.reset((0,), cfg, {0: assignment})
        adapter.activate((0,), cfg)
        for step in range(int(plan.end_s * 4)):
            adapter.update(d.RuntimeFrame(((0, step * 0.25),), ((0, plan.pose_at(step * 0.25)),)), cfg)
        stamp = adapter.finish(0, "success")
        self.assertEqual(stamp["payload"]["assignment"]["sensor_source"], "journal_ray_sensor_all_tasks")
        self.assertEqual(stamp["payload"]["assignment"]["sensor"], {"ray_count": 36, "max_range_m": 30.0})
        self.assertIsNotNone(stamp["payload"]["runtime"]["first_detection_s"])
        self.assertIsNone(stamp["payload"]["runtime"]["non_exposure_reason"])
        self.assertEqual(d.read_stamp(stamp), stamp)
        with self.assertRaisesRegex(ValueError, "sensor source"):
            dataclasses.replace(assignment.candidate, sensor_source="none").validate()


class TestDoseTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = {name: d.vessel_dose_table(f.envelope(name)) for name in f.VESSELS}

    def test_ttc_and_components(self):
        for name, table in self.tables.items():
            env = f.envelope(name)
            self.assertEqual(table["ttc_s"], tuple(r * env.t_avoid_s for r in d.DOSE_TTC_RATIOS[1:]))
            self.assertEqual(table["t_avoid_components"], dict(env.t_avoid_components))
            self.assertEqual(table["native_speeds_mps"], tuple(mu * env.top_speed_mps for mu in d.NATIVE_RATIOS))

    def test_detection_ranges_are_graded_and_fit_the_30m_fan(self):
        for name, table in self.tables.items():
            with self.subTest(vessel=name):
                for motion, rows in table["detection_m"].items():
                    for kind, row in rows.items():
                        for series in (row["nominal"], row["worst"]):
                            self.assertTrue(all(a > b for a, b in zip(series, series[1:])), (motion, kind, series))
                        self.assertTrue(all(w >= n for w, n in zip(row["worst"], row["nominal"])))
                self.assertTrue(table["fits_sensor"], table["violations"])
                self.assertLess(table["max_required_d1_m"], 30.0)
        # largest required range: WAM-V opt1 head-on D1 (DOSE_TABLE_E4.md)
        self.assertAlmostEqual(self.tables[WAMV]["max_required_d1_m"], 25.276278, places=5)
        self.assertAlmostEqual(self.tables[BB]["max_required_d1_m"], 10.477047, places=5)
        self.assertAlmostEqual(self.tables[WAMV_OPT2]["max_required_d1_m"], 21.450585, places=5)
        self.assertEqual({(m, k) for m, k, _, _ in REQUIRED_PATHS},
                         {("transit", "crossing"), ("transit", "drifting"), ("transit", "head_on"),
                          ("stationary", "crossing"), ("stationary", "head_on"), ("stationary", "overtaking")})

    def test_head_on_matches_closed_form(self):
        for name, table in self.tables.items():
            env = f.envelope(name)
            for dose in range(1, 5):
                self.assertAlmostEqual(table["detection_m"]["transit"]["head_on"]["nominal"][dose - 1],
                                       head_on_range_closed_form(env, dose), places=9)

    def test_table_equals_the_range_admission_actually_uses(self):
        # ring: head_on, side +1, aim 0 on a straight run-in -> the table's nominal row
        for name in (BB, WAMV):
            env = f.envelope(name)
            ctx = f.context("ring", env)
            ladder = d.build_ladder(ctx, f.key("ring"), f.suite(env))
            used = [c.reveal_evidence.detection_range_m for c in ladder[1:]]
            table = self.tables[name]["detection_m"]["transit"]["head_on"]["nominal"]
            for a, b in zip(used, table):
                self.assertAlmostEqual(a, b, places=9)

    def test_violation_is_reported(self):
        table = d.vessel_dose_table(f.envelope(WAMV), sensor_range_m=22.0)
        self.assertFalse(table["fits_sensor"])
        self.assertIn(("transit", "head_on"), {(m, k) for m, k, _, _ in table["violations"]})


def _pool_view(pool):
    return (pool.pool_digest, pool.per_dose_counts, pool.keys_offered, pool.keys_admitted, pool.rejections,
            sorted((k.env, dose, a.candidate_digest, a.witness_digest) for (k, dose), a in pool.assignments.items()))


class TestParallelAdmission(unittest.TestCase):
    """workers > 1 must give exactly the serial pool (digests, rejections, witnesses)."""

    def test_admit_grid_parallel_equals_serial(self):
        jobs = []
        for vessel in (BB, WAMV):
            env = f.envelope(vessel)
            for task in ("reach", "ring", "moving_basin"):
                ctx = f.context(task, env)
                jobs += [(ctx, f.key(task, env=e), f.suite(env)) for e in range(3)]
        # one key whose D1 fails (short sensor), so rejections and joint removal are compared too
        short = d.SensorSpec(ray_count=36, max_range_m=22.0, source="short")
        jobs.append((f.context("ring", f.envelope(WAMV), sensor=short), f.key("ring", env=7), f.suite(f.envelope(WAMV))))
        group = lambda job: (job[0].envelope.vessel_id, job[0].task)
        serial = d.admit_grid(jobs, workers=1, group_by=group)
        parallel = d.admit_grid(jobs, workers=3, group_by=group)
        self.assertEqual(list(serial), list(parallel))
        for label in serial:
            self.assertEqual(_pool_view(serial[label]), _pool_view(parallel[label]), label)
            for kd, a in serial[label].assignments.items():
                self.assertEqual(a.witness, parallel[label].assignments[kd].witness)
        self.assertTrue(any(r.reason == "detection_range_exceeds_existing_sensor_range"
                            for r in serial[(WAMV, "ring")].rejections))

    def test_admit_ladders_parallel_equals_serial_and_matches_admit_grid(self):
        env = f.envelope(BB)
        ctx = f.context("docking", env)
        keys = [f.key("docking", env=e) for e in range(3)]
        ladders = [(k, d.build_ladder(ctx, k, f.suite(env))) for k in keys]
        serial = d.admit_ladders(ladders)
        parallel = d.admit_ladders(ladders, workers=2)
        grid = d.admit_grid([(ctx, k, f.suite(env)) for k in keys])
        self.assertEqual(_pool_view(serial), _pool_view(parallel))
        self.assertEqual(_pool_view(serial), _pool_view(grid))
        with self.assertRaises(ValueError):
            d.admit_ladders(ladders, workers=0)
        with self.assertRaises(ValueError):
            d.admit_grid([(ctx, keys[0], f.suite(env))] * 2)

    def test_worker_bootstrap_never_imports_a_task_package(self):
        # A pool worker re-loads the package BY PATH under the parent's module name;
        # for a dotted (in-simulator) name the parents must be empty stubs, so the
        # real tasks/ packages (gym registration, Isaac Lab) are never imported.
        probe = ("import importlib.util, sys; sys.dont_write_bytecode = True; p = sys.argv[1]; "
                 "s = importlib.util.spec_from_file_location('dyn_boot', p + '/__init__.py', "
                 "submodule_search_locations=[p]); m = importlib.util.module_from_spec(s); "
                 "sys.modules['dyn_boot'] = m; s.loader.exec_module(m); "
                 "m.adapters._bootstrap_worker(p, 'tasks._shared.conditions.dynamic'); "
                 "import tasks._shared.conditions.dynamic.adapters as a; "
                 "bad = sorted(n for n in sys.modules if n.split('.')[0] in "
                 "('torch', 'isaaclab', 'isaacsim', 'omni', 'pxr')); "
                 "print('OK', a.__name__, sys.modules['tasks'].__path__, "
                 "sys.modules['tasks._shared'].__path__, bad); sys.exit(1 if bad else 0)")
        result = subprocess.run([sys.executable, "-B", "-c", probe, str(f.PACKAGE)],
                                capture_output=True, text=True, timeout=120, cwd=str(f.PACKAGE),
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK tasks._shared.conditions.dynamic.adapters [] [] []", result.stdout)


class TestHotLoopIdentity(unittest.TestCase):
    """E4 optimisations must be bit-identical to D7's formulation."""

    def test_interval_clearance_equals_reference_bitwise(self):
        checked = 0
        cases = ((BB, "station_keeping", 1), (WAMV, "ring", 0), (BB, "moving_basin", 0), (WAMV, "docking", 0),
                 (BB, "harbour", 2))
        for vessel, task, row in cases:
            env = f.envelope(vessel)
            ctx = f.context(task, env)
            ladder = d.build_ladder(ctx, f.key(task, env=row), f.suite(env))
            for cand in ladder:
                if isinstance(cand, d.adapters.NotApplicableByDesign):   # G23e: BlueBoat station keeping D1
                    continue
                request = cand.request
                witness = d.find_witness(request)
                self.assertTrue(witness.accepted, (vessel, task, witness.reason))
                states = witness.trajectory
                for before, after in list(zip(states, states[1:]))[::3]:
                    duration = after.time_s - before.time_s
                    for accel, rate in ((after.accel_mps2, after.yaw_rate_rps), (env.max_accel_mps2, env.max_yaw_rate_rps),
                                        (-env.max_decel_mps2, -0.5 * env.max_yaw_rate_rps)):
                        args = (request, before.pose, accel, rate, before.time_s, duration)
                        self.assertEqual(feas._interval_clearance(*args), feas._interval_clearance_reference(*args))
                        checked += 1
        self.assertGreater(checked, 1000)

    def test_footprint_cache_is_not_a_field_and_matches_formulas(self):
        fp = d.Footprint(-0.594, 0.6068, -0.418, 0.512)
        self.assertEqual([x.name for x in dataclasses.fields(fp)], ["x_min_m", "x_max_m", "y_min_m", "y_max_m"])
        self.assertEqual(dataclasses.asdict(fp), {"x_min_m": -0.594, "x_max_m": 0.6068, "y_min_m": -0.418,
                                                  "y_max_m": 0.512})
        self.assertEqual(fp.length_m, 0.6068 - -0.594)
        self.assertEqual(fp.center, (0.5 * (-0.594 + 0.6068), 0.5 * (-0.418 + 0.512)))
        self.assertEqual(fp.half_extents, (0.5 * (0.6068 - -0.594), 0.5 * (0.512 - -0.418)))
        self.assertEqual(fp.reach_m, max(math.hypot(x, y) for x, y in fp.corners))
        self.assertEqual(fp, dataclasses.replace(fp))
        self.assertEqual(d.digest(fp), d.digest({"x_min_m": -0.594, "x_max_m": 0.6068, "y_min_m": -0.418,
                                                 "y_max_m": 0.512}))

    def test_pose_fast_path_keeps_validation(self):
        for bad in ((math.inf, 0.0, 0.0, 0.0), (0.0, math.nan, 0.0, 0.0), (0.0, 0.0, True, 0.0), (0.0, 0.0, 0.0, "1")):
            with self.assertRaises(ValueError):
                d.Pose(*bad)
        self.assertEqual(d.Pose(1, 2, 3), d.Pose(1.0, 2.0, 3.0, 0.0))

    def test_station_d1_reveal_never_precedes_the_settled_clock(self):
        # G23e: the station-keeping encounter now lies on the approach (sk_approach.py): every applicable reveal
        # is after t = 0 and the conflict precedes the settled clock. E4 fix in _fit_conflict kept (other rules):
        # t_min + window - window may round one ulp low.
        for vessel in f.VESSELS:
            env = f.envelope(vessel)
            ctx = f.context("station_keeping", env)
            for row in range(8):
                ladder = d.build_ladder(ctx, f.key("station_keeping", env=row), f.suite(env))
                for cand in ladder[1:]:
                    if isinstance(cand, d.adapters.NotApplicableByDesign):
                        continue
                    enc = cand.request.encounter
                    self.assertGreaterEqual(enc.intruders[0].reveal_s, 0.0)
                    self.assertLess(enc.meta("conflict_time_s"), ctx.settled_phase_start_s)
        # the exact D7 case that rounded low with the final registry
        window = 4.0 * f.envelope(WAMV).t_avoid_s
        plan = d.build_stationary_plan(d.Pose(0.0, 0.0, 0.0, 0.0), 400.0)
        settled = 117.54392239748873
        t = _fit_conflict(plan, settled + window, window, settled + window, settled)
        self.assertGreaterEqual(t - window, settled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
