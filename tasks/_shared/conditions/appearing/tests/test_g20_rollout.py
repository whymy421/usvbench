"""G20 rollout of the appearing suite, final rules of G20b / G20c: CPU tests of the shared module.

Covers: the per-task placement table (every journal task has a site rule, a family and the env side,
config.WIRED_TASKS = all twelve, a tuple G19b's sampler reads); the ladder switch (the active option's
constants, every option's derived constants, distinct suite versions, D1 = 4 everywhere); placement
FEASIBILITY on the REAL journal layouts of the ten new tasks for both vessels (the site passes every
dose-independent rule, lies in its declared window or sector, is the same at every dose, every dose's
nominal trigger lies on the nominal plan at the declared gap); the per-task D29 extensions; the docking
start rule (margin 0, start gap above the largest trigger gap, berth exclusion; G20d: berth-axis distance from
the mouth >= run-up + r + hw); the G20d station-keeping rule (cylinder ON the approach route at 0.45 x the
start-to-centre distance; station surface distance >= max(R_hold + hw, reach + hw) (REV-G20c M6: the G17 evade
clearance is report only);
the nominal approach is blocked; moving trigger of every offered main dose; BlueBoat D1 not applicable by design);
the G20d fortress radius rule (r <= 0.8 s, >= the ray floor, passage >= g on one side, gates >= r + g);
the moving-basin native-sweep clearance; the two v2.4 G3 pools untouched (files, pins, placements);
and the DIAG42 bundle pins of all 24 task-vessels (in-repo manifests, no pool file in the repo).
Run: python -B test_g20_rollout.py
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_fixtures as F  # noqa: E402

C, P, POOL, LAD = F.config, F.placement, F.pool_module, F.ladder
FAM = F.families
JC = F.load("_shared.conditions.journal_config")
NEW_TASKS = ("reach", "ring", "fortress", "forced_crossing", "iceberg", "moving_basin", "path_hazard",
             "harbour_mission", "station_keeping", "docking")
D29_EXTENSIONS = ("fortress", "forced_crossing", "iceberg", "moving_basin")
# T-base v2.4 pins of the two G3 pools (ADMISSION_REPORT_1b / g3cpool bundle); never rewritten
V24_PINS = {("hazard_nav", "scatter__blueboat_v2.json.xz"):
            "a7426370306bbc6fb0223e6fa1882eb26820e6b2bda96d8acb9e3d7b0d49a4df",
            ("hazard_nav", "scatter__wamv_vrx_v2_opt1.json.xz"):
            "5a6f2ffb373a18e0e880e581d731814a923562fbed43d358af11763ff6c159bd",
            ("path_following", "path_following__blueboat_v2.json.xz"):
            "3a4d3f98605a1038e53d78ba353561f26a1698396b6a7ed5748218bcb66f19d8",
            ("path_following", "path_following__wamv_vrx_v2_opt1.json.xz"):
            "6cda52ff3e9e8a50bcdf1d1c4082ba05fbc483375b204864b2f5779be50545e9"}
KEYS = ((0, 0), (1, 1))


def inputs(task, vessel, env, episode):
    return tuple(FAM.admission_inputs(task, vessel, F.TIER, F.SEED, env, episode))


class Table(unittest.TestCase):
    def test_every_journal_task_has_a_rule_a_family_and_the_env_side(self):
        tasks = tuple(F.plans.TASKS)
        self.assertEqual(sorted(C.PLACEMENT_TASKS), sorted(tasks))
        self.assertIsInstance(C.WIRED_TASKS, tuple)                         # G19b's sampler reads it
        self.assertEqual(sorted(C.WIRED_TASKS), sorted(tasks))
        for task in tasks:
            self.assertIn(task, P.SITE_RULES)
            self.assertIn(task, FAM.FAMILY_OF_TASK)
            self.assertEqual(C.check_wired(task), task)
            if P.SITE_RULES[task] == "route_window_redraw":
                (lo, hi), anchor, note = P.ROUTE_WINDOWS[task]
                self.assertTrue(0.0 <= lo < hi <= 1.0 and note)
        self.assertEqual(P.SITE_RULES["station_keeping"], "station_route_block")      # G20d
        self.assertEqual(P.PLACEMENT_RULES_VERSION, "g20d_placement_v4")                # G20d
        self.assertEqual(C.context_task("harbour_mission"), "harbour")
        self.assertEqual(C.journal_task("harbour"), "harbour_mission")
        with self.assertRaises(ValueError):
            C.check_placement_task("no_such_task")

    def test_d29_band_and_its_declared_extensions(self):
        for task, ((lo, hi), anchor, note) in P.ROUTE_WINDOWS.items():
            if task in D29_EXTENSIONS:
                self.assertNotEqual((lo, hi), P.ROUTE_FRACTION_RANGE)
                self.assertIn("D29 extension", note)
            elif task != "docking":
                self.assertEqual((lo, hi), P.ROUTE_FRACTION_RANGE, task)      # the D29 band [0.40, 0.60]
        self.assertEqual(P.ROUTE_FRACTION_RANGE, (0.4, 0.6))
        self.assertEqual(P.START_MARGIN_LENGTHS_BY_TASK, {"docking": 0.0})

    def test_slots_unchanged(self):
        for task in C.PLACEMENT_TASKS:
            base, reserve = F.plans.STATIC_CAPACITY[task], F.plans.D7_RESERVED_SLOTS[task]
            self.assertEqual(C.appearing_slot(base, reserve), base + reserve)
        self.assertEqual(C.appearing_slot(128, 0), 128)                 # moving basin: no D7 reserve

    def test_pool_file_names(self):
        self.assertEqual(POOL.pool_file_name("scatter", F.BB), "scatter__blueboat_v2.json.xz")
        self.assertEqual(POOL.pool_file_name("reach", F.BB, 42), "reach__blueboat_v2__s42.json.xz")
        self.assertEqual(POOL.pool_file_name("reach", F.WAMV, 41041), "reach__wamv_vrx_v2_opt1__s41041.json.xz")


class Ladder(unittest.TestCase):
    def test_the_active_option_drives_every_constant(self):
        active = LAD.ladder_constants(LAD.ACTIVE_LADDER)
        self.assertEqual(C.LADDER_OPTION, LAD.ACTIVE_LADDER)
        self.assertEqual(C.GAP_RATIOS, active["gap_ratios"])
        self.assertEqual(C.WITNESSED_DOSES, active["witnessed_doses"])
        self.assertEqual(C.MAX_DOSE, active["max_dose"])
        self.assertEqual(C.DOSE_COUNT, active["max_dose"] + 1)
        self.assertEqual(C.APPEARING_SUITE_VERSION, active["suite_version"])
        self.assertEqual(F.stamp.SUITE_VERSION, active["suite_version"])
        self.assertEqual(F.stamp.GAP_RATIOS, active["gap_ratios"])
        self.assertIn("ladder option " + LAD.ACTIVE_LADDER, F.stamp.SEMANTICS["dose"])

    def test_default_is_the_owner_metre_ladder(self):
        # owner decision 2026-10-03: per-vessel metres, D1 every controller can pass, D4 the fastest at its limit
        self.assertEqual(LAD.ACTIVE_LADDER, "M")
        self.assertIsNone(C.GAP_RATIOS)
        # G47 (v2.6.4): the owner's frozen doses of 2026-10-14 (test_g47_frozen_doses.py has the derivation)
        self.assertEqual(LAD.GAPS_M[F.BB], (None, 7.5, 3.5, 1.6, 0.76, 0.76))
        self.assertEqual(LAD.GAPS_M[F.WAMV], (None, 15.0, 6.6, 2.9, 1.25, 1.25))
        self.assertEqual((C.MAIN_DOSES, C.SUPPLEMENTARY_DOSES), ((1, 2, 3, 4), (5,)))
        self.assertEqual(C.APPEARING_SUITE_VERSION, "g3_appearing_gap_v7m")          # G47 (v2.6.4; G20d v6m)
        self.assertEqual(LAD.KINEMATIC_FLOOR_M, {F.BB: 0.67, F.WAMV: 1.03})           # informational only
        self.assertFalse(hasattr(LAD, "WITNESS_FLOOR_M"))                              # G20c: no mislabel
        self.assertEqual(LAD.CALIBRATION_RUNG_M, {F.BB: 0.76, F.WAMV: 1.25})
        self.assertEqual(LAD.GAP_OVERRIDE_MIN_M, LAD.CALIBRATION_RUNG_M)
        self.assertEqual(LAD.CALIBRATION_SWEEP_M[F.BB], (0.76, 1.0, 1.4, 2.0, 2.9, 4.0, 5.0, 6.6, 7.5))
        self.assertEqual(LAD.CALIBRATION_SWEEP_M[F.WAMV], (1.25, 1.4, 1.8, 2.5, 3.6, 5.0, 7.5, 11.0, 15.0))
        for vessel in (F.BB, F.WAMV):
            self.assertEqual(min(LAD.CALIBRATION_SWEEP_M[vessel]), LAD.GAPS_M[vessel][5])   # sweep floor = D5

    def test_metre_ladder_rationale_numbers(self):
        # G47 (v2.6.4, owner 2026-10-14): D1 by the owner (sweep caps 7.5 / 15 m), D4 = the pre-registered rule's
        # fallback (the calibration rung), D2, D3 geometric (rounded to 0.1 m); every rung at or above the D7
        # witness-model floor; D5 = the measured calibration rung (= D4 now)
        for vessel, d1, d4 in ((F.BB, 7.5, 0.76), (F.WAMV, 15.0, 1.25)):
            gaps = LAD.GAPS_M[vessel]
            self.assertEqual(gaps[1], d1)
            self.assertEqual(gaps[4], d4)
            ratio = (d4 / d1) ** (1.0 / 3.0)
            self.assertAlmostEqual(gaps[2], round(d1 * ratio, 1), places=9)
            self.assertAlmostEqual(gaps[3], round(d1 * ratio * ratio, 1), places=9)
            env = F.envelope(vessel)
            floor = env.reaction_s * 1.0 + 1.0 / (2.0 * env.max_decel_mps2) + 0.02
            self.assertAlmostEqual(LAD.D7_WITNESS_FLOOR_M[vessel], floor, places=12)
            self.assertGreater(gaps[5], floor)
            self.assertEqual(gaps[4], gaps[5])                     # G47: D4 = the calibration rung (fallback)
            self.assertGreaterEqual(floor, LAD.KINEMATIC_FLOOR_M[vessel])

    def test_every_option_derives_its_constants(self):
        a, b, e, m = (LAD.ladder_constants(o) for o in ("A", "B", "E", "M"))
        self.assertEqual((m["kind"], m["max_dose"], m["supplementary_doses"]), ("metres", 5, (5,)))
        self.assertIsNone(m["gap_ratios"])
        self.assertEqual(a["gap_ratios"], (None, 4.0, 3.0, 2.0, 1.25))
        self.assertEqual((a["witnessed_doses"], a["supplementary_doses"], a["max_dose"]), ((1, 2, 3, 4), (), 4))
        self.assertEqual(a["suite_version"], "g3_appearing_gap_v2")              # the v2.4 pins stay valid under A
        self.assertEqual(b["gap_ratios"], (None, 4.0, 2.0, 1.0, 0.5))
        self.assertEqual(e["gap_ratios"], (None, 4.0, 3.0, 2.0, 1.25, 0.5))
        self.assertEqual((e["main_doses"], e["supplementary_doses"], e["max_dose"]), ((1, 2, 3, 4), (5,), 5))
        self.assertEqual(len({a["suite_version"], b["suite_version"], e["suite_version"], m["suite_version"]}), 4)
        for c in (a, b, e):
            self.assertEqual(c["gap_ratios"][1], 4.0)                            # k ladders: D1 = 4
        with self.assertRaises(ValueError):
            LAD.ladder_constants("C")

    def test_levels_follow_the_ladder(self):
        for k in range(C.MAX_DOSE + 1):
            self.assertEqual(JC.resolve_journal_conditions(["appearing=D%d" % k], seed=1).appearing_dose, k)
        with self.assertRaises(JC.SuiteConflict) as caught:
            JC.resolve_journal_conditions(["appearing=D%d" % (C.MAX_DOSE + 1)], seed=1)
        self.assertEqual(caught.exception.code, "C6")
        with self.assertRaises(JC.SuiteConflict):
            JC.resolve_journal_conditions(["waves=D5"], seed=1)              # other suites keep D0..D4

    def test_trigger_gap_bounds(self):
        for vessel in (F.BB, F.WAMV):
            t_avoid = F.envelope(vessel).t_avoid_s
            gaps = C.trigger_gaps_m(t_avoid, vessel)[1:]
            self.assertEqual(C.largest_trigger_gap_m(t_avoid, vessel), max(gaps))
            self.assertEqual(C.smallest_trigger_gap_m(t_avoid, vessel), min(gaps))
            self.assertEqual(C.smallest_main_trigger_gap_m(t_avoid, vessel), min(gaps[:4]))
            self.assertEqual(C.start_gap_m("reach", t_avoid, vessel), max(gaps))
            if C.LADDER_KIND == "metres":                 # calibration tasks: the largest sweep gap
                self.assertEqual(C.start_gap_m("scatter", t_avoid, vessel), max(LAD.CALIBRATION_SWEEP_M[vessel]))
            for task in C.PLACEMENT_TASKS:                # the stamp reader's stdlib copy of the rule agrees
                self.assertEqual(LAD.start_gap_m(task, vessel), C.start_gap_m(task, t_avoid, vessel), task)
            self.assertEqual(LAD.T_AVOID_S[vessel], t_avoid)
            env = F.envelope(vessel)
            self.assertEqual(LAD.VESSEL_TOP_SPEED_MPS[vessel], env.top_speed_mps)


class PlacementFeasibility(unittest.TestCase):
    """The final rule places a cylinder that passes every dose-independent rule (real layouts)."""

    def check_site(self, task, vessel, env, episode):
        item = inputs(task, vessel, env, episode)
        context, scale, rng = item[:3]
        regions = item[3] if len(item) > 3 else None
        self.assertEqual(context.task, C.context_task(task))
        cfg = F.admission.admission_suite(vessel)
        key = F.dynamic.EpisodeKey(F.SEED, env, episode, C.hazard_group(task))
        try:
            doc, plan = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng, regions=regions)
        except P.PlacementError as error:
            return None, error.reasons
        again, _ = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng, regions=regions)
        self.assertEqual(doc, again)
        site = doc["site"]
        self.assertEqual(site["rule"], P.SITE_RULES[task])
        self.assertEqual(site["rules_version"], P.PLACEMENT_RULES_VERSION)
        d1 = C.start_gap_m(task, context.envelope.t_avoid_s, vessel)
        self.assertEqual(P.site_reasons(context, doc, d1), [])
        self.assertEqual(P.placement_reasons(context, doc), [])
        cyl = doc["cylinder"]
        fp = context.envelope.footprint
        hw = 0.5 * fp.beam_m
        start_gap = P.gap_at(context.initial_pose, cyl["x_m"], cyl["y_m"], cyl["radius_m"], fp)
        self.assertGreater(start_gap, C.start_gap_m(task, context.envelope.t_avoid_s, vessel))  # no step-0 trigger
        if site["rule"] == "route_window_redraw":
            lo, hi = site["window_m"]
            self.assertTrue(lo - 1e-9 <= site["route_distance_m"] <= hi + 1e-9)
            self.assertGreaterEqual(lo, site["start_rule"]["min_route_distance_m"] - 1e-9)
            pose = plan.pose_at(site["t_site_s"])
            self.assertAlmostEqual(cyl["x_m"], pose.x, places=12)
            self.assertAlmostEqual(cyl["y_m"], pose.y, places=12)
            if task == "docking":
                self.assertLessEqual(site["route_distance_m"], site["anchor_distance_m"])
                self.assertEqual(site["start_rule"]["margin_lengths"], 0.0)
                self.assertGreaterEqual(P.berth_clearance(site["berth_exclusion"], cyl["x_m"], cyl["y_m"],
                                                          cyl["radius_m"]), hw - 1e-9)
                # G20d: berth-axis distance from the mouth line >= approach run-up + r + hw
                mouth = site["berth_mouth_rule"]
                runup = F.plans.vessel_record(vessel).geometry.approach_runup_m
                self.assertAlmostEqual(mouth["min_axis_distance_m"], runup + cyl["radius_m"] + hw, places=12)
                self.assertGreaterEqual(mouth["opening_x_m"] - P.berth_axis_x(site["berth_exclusion"], cyl["x_m"],
                                                                              cyl["y_m"]),
                                        mouth["min_axis_distance_m"] - 1e-9)
                self.assertLessEqual(site["route_distance_m"], mouth["window_end_m"] + 1e-9)
            else:
                self.assertEqual(site["start_rule"]["margin_lengths"], P.PF_START_MARGIN_LENGTHS)
                self.assertNotIn("berth_exclusion", site)
            if task == "moving_basin":
                for native in context.native_geometry:
                    self.assertGreaterEqual(P.native_sweep_clearance(native, cyl["x_m"], cyl["y_m"],
                                                                     cyl["radius_m"]), hw - 1e-9)
        else:
            # G20d (decisions 1B / 2B): ON the approach route at 0.45 x D_start from the hold centre (blocking)
            self.assertEqual(site["rule"], "station_route_block")
            cx, cy = site["station_centre"]
            hold = context.hold_circle or context.goal
            self.assertEqual((cx, cy), (hold.x, hold.y))
            surface = math.hypot(cyl["x_m"] - cx, cyl["y_m"] - cy) - cyl["radius_m"]
            self.assertAlmostEqual(site["station_surface_distance_m"], surface, places=12)
            need = max(float(hold.radius) + hw, fp.reach_m + hw)               # REV-G20c M6: no G17 constant
            self.assertEqual(P.station_clearance_m(context), need)
            self.assertGreaterEqual(surface, need - 1e-9)                        # hold circle + hw, every heading
            self.assertEqual(site["g17_evade_target_clear"], surface >= P.SK_G17_EVADE_CLEARANCE_M[vessel])
            for k in range(16):                                                  # centre clear at EVERY heading
                pose = F.dynamic.Pose(cx, cy, 2.0 * math.pi * k / 16.0, 0.0)
                self.assertGreaterEqual(P.gap_at(pose, cyl["x_m"], cyl["y_m"], cyl["radius_m"], fp), hw - 1e-9)
            start = context.initial_pose
            d_start = math.hypot(cx - start.x, cy - start.y)
            self.assertAlmostEqual(site["start_to_centre_distance_m"], d_start, places=12)
            self.assertAlmostEqual(math.hypot(cyl["x_m"] - cx, cyl["y_m"] - cy), P.SK_ROUTE_FRACTION * d_start,
                                   places=9)                                     # 0.45 D_start from the centre
            # controller decision J9: on the straight start-to-centre line (the SK reference route)
            self.assertAlmostEqual(cyl["x_m"], start.x + (1.0 - P.SK_ROUTE_FRACTION) * (cx - start.x), places=12)
            self.assertAlmostEqual(cyl["y_m"], start.y + (1.0 - P.SK_ROUTE_FRACTION) * (cy - start.y), places=12)
            self.assertLess(site["route_plan_min_distance_m"], cyl["radius_m"])   # the curved plan is blocked too
            self.assertLess(site["nominal_min_gap_m"], 0.0)                      # it blocks the nominal approach
            self.assertEqual(site["offered_main_doses"], list(LAD.offered_main_doses(task, vessel)))
            self.assertEqual(site["start_gap_m"], C.start_gap_m(task, context.envelope.t_avoid_s, vessel))
        if rng is None and task != "fortress":
            self.assertAlmostEqual(cyl["radius_m"], C.RADIUS_LENGTH_FACTOR * scale, places=12)
        elif task == "fortress":
            rule = site["fortress_radius_rule"]                                  # G20d radius rule
            self.assertLessEqual(cyl["radius_m"], C.RADIUS_LENGTH_FACTOR * scale + 1e-12)
            self.assertGreaterEqual(cyl["radius_m"], rule["ray_floor_m"] - 1e-12)
            self.assertEqual(cyl["radius_m"], rule["radius_m"])
            self.assertEqual(doc["radius_rule"]["rule"], "fortress_passage_radius")
            g = rule["passage_gap_m"]
            sides = [d for d in (rule["left_surface_distance_m"], rule["right_surface_distance_m"]) if d is not None]
            self.assertGreaterEqual(max(sides) - cyl["radius_m"], g - 1e-9)       # a passage >= g on one side
            self.assertGreaterEqual(rule["gate_bound_m"], cyl["radius_m"] - 1e-9)  # every gate opening >= r + g
        else:
            self.assertTrue(rng[0] <= cyl["radius_m"] <= rng[1])
        triggers = []
        for dose in C.offered_doses(task, vessel):
            try:
                cand, request = P.candidate(context, doc, plan, key, dose, cfg)
            except P.PlacementError as error:
                return doc, [error.reason]
            self.assertEqual(cand["position_digest"], F.stamp.digest(doc))
            if cand["nominal_trigger_s"] is None:              # calibration rung D5, not reached nominally
                self.assertIn(dose, C.SUPPLEMENTARY_DOSES)
                continue
            self.assertLessEqual(cand["nominal_trigger_gap_m"], cand["trigger_gap_m"])
            self.assertGreater(cand["nominal_trigger_gap_m"], cand["trigger_gap_m"] - 1e-6)
            if task == "station_keeping" and dose in C.MAIN_DOSES:     # final approach, still moving
                self.assertGreaterEqual(cand["nominal_trigger_speed_mps"],
                                        P.SK_MIN_TRIGGER_SPEED_FRACTION * C.REFERENCE_SPEED_MPS)
                self.assertLess(cand["nominal_trigger_s"], plan.end_s)
            triggers.append(cand["nominal_trigger_s"])
        self.assertEqual(triggers, sorted(triggers))                       # shorter gap -> later trigger
        return doc, []

    def run_task(self, task):
        placed = 0
        for vessel in (F.BB, F.WAMV):
            for env, episode in KEYS:
                with self.subTest(task=task, vessel=vessel, key=(env, episode)):
                    doc, reasons = self.check_site(task, vessel, env, episode)
                    if doc is not None and not reasons:
                        placed += 1
                    elif doc is None:                  # a counted rejection with the rule's declared reason
                        self.assertIn(reasons[0], ("no_clear_site_in_route_window",
                                                   "route_window_empty_after_start_rule",
                                                   "no_clear_site_on_approach_route"))
        return placed

    def test_open_and_gate_tasks(self):
        for task in ("reach", "ring", "forced_crossing", "iceberg"):
            self.assertGreaterEqual(self.run_task(task), 3, task)

    def test_field_tasks(self):
        # path hazard: key (0, 0) BlueBoat has no clear site in 8 candidates (G20 probe), so 1 of 4 here
        self.assertGreaterEqual(self.run_task("path_hazard"), 1)
        self.assertGreaterEqual(self.run_task("harbour_mission"), 3)

    def test_fortress_places_or_rejects_with_declared_reasons(self):
        for vessel in (F.BB, F.WAMV):
            for env, episode in KEYS:
                doc, reasons = self.check_site("fortress", vessel, env, episode)
                if doc is None:
                    self.assertEqual(reasons[0], "no_clear_site_in_route_window")

    def test_station_keeping_route_block(self):
        # G20d: deterministic site on the approach route; a key failing a dose-independent rule would be a counted
        # rejection (no_clear_site_on_approach_route, checked in run_task)
        self.assertEqual(self.run_task("station_keeping"), 4)

    def test_docking_approach_lane(self):
        self.assertGreaterEqual(self.run_task("docking"), 2)

    def test_moving_basin_buffer_clears_every_native_sweep(self):
        context, scale, rng = inputs("moving_basin", F.BB, 0, 0)
        plain = F.hazard.d7_context(F.hazard.build_scenario("moving_basin", F.BB, F.TIER, F.SEED, 0, 0))
        self.assertGreater(len(context.native_geometry), 0)
        # the frozen (D0) movers are static obstacles of the appearing view, in addition to the walls
        self.assertEqual(len(context.static_obstacles), len(plain.static_obstacles) + len(context.native_geometry))
        self.assertEqual(context.static_obstacles[:len(plain.static_obstacles)], plain.static_obstacles)
        self.assertGreaterEqual(len(context.route), 2)                    # one guidance circle per frozen row
        self.assertTrue(all(not m for m in context.route_mandatory))
        self.assertEqual(context.geometry_id, plain.geometry_id)
        doc, reasons = self.check_site("moving_basin", F.BB, 0, 0)
        self.assertIsNotNone(doc, reasons)


class DockingAndStationRules(unittest.TestCase):
    def test_docking_berth_rectangle_and_exclusion(self):
        context, scale, rng, regions = inputs("docking", F.BB, 1, 0)
        berth = regions["berth"]
        scenario = F.load("docking.journal_docking").build_scenario(F.BB, 2, F.SEED, 1, 0)
        rw = float(scenario.layout.params["wall_radius_m"])
        self.assertAlmostEqual(berth["x_min_m"], scenario.berth["opening_x_m"] - rw)
        self.assertAlmostEqual(berth["x_max_m"], scenario.berth["back_x_m"] + 2.0 * rw)
        self.assertAlmostEqual(berth["half_width_m"], scenario.berth["half_width_m"] + 2.0 * rw)
        # every wall cylinder lies inside the rectangle; the berth centre (target) is inside it
        region = P._berth_region(berth)
        for (x, y), r in zip(scenario.wall_centers.tolist(), scenario.wall_radii.tolist()):
            self.assertLessEqual(P.berth_clearance(region, x, y, r), 1e-9)
        self.assertLess(P.berth_clearance(region, 0.0, 0.0, 0.1), 0.0)
        # a cylinder placed at the berth centre is "overlaps_berth"
        cfg = F.admission.admission_suite(F.BB)
        key = F.dynamic.EpisodeKey(F.SEED, 1, 0, C.hazard_group("docking"))
        doc, _ = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng, regions=regions)
        moved = dict(doc, cylinder=dict(doc["cylinder"], x_m=0.0, y_m=0.0))
        self.assertIn("overlaps_berth", P.placement_reasons(context, moved))
        with self.assertRaises(ValueError):                              # docking needs its berth rectangle
            P.place(context, key, cfg, length_scale=scale, radius_range_m=rng)
        rcontext, rscale, rrng = inputs("reach", F.BB, 0, 0)[:3]
        with self.assertRaises(ValueError):                              # regions are a docking-only input
            P.place(rcontext, F.dynamic.EpisodeKey(F.SEED, 0, 0, C.hazard_group("reach")), cfg,
                    length_scale=rscale, radius_range_m=rrng, regions=regions)

    def test_docking_start_rule_drops_only_the_margin(self):
        context, scale, rng, regions = inputs("docking", F.BB, 1, 0)
        cfg = F.admission.admission_suite(F.BB)
        key = F.dynamic.EpisodeKey(F.SEED, 1, 0, C.hazard_group("docking"))
        doc, _ = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng, regions=regions)
        rule = doc["site"]["start_rule"]
        d1 = C.start_gap_m("docking", context.envelope.t_avoid_s, F.BB)
        # G48/G49 (owner 2026-10-14): BlueBoat docking D1 is not applicable by design, so its start gap is D2
        # (3.5 m of the frozen table) -- the rule itself (start gap + reach + radius, no margin) is unchanged
        self.assertEqual(d1, C.trigger_gap_m(2, context.envelope.t_avoid_s, F.BB))
        self.assertAlmostEqual(rule["min_route_distance_m"],
                               d1 + context.envelope.footprint.reach_m + doc["cylinder"]["radius_m"], places=12)
        self.assertEqual(rule["start_gap_m"], d1)

    def test_station_rule_reasons(self):
        context, scale, rng = inputs("station_keeping", F.BB, 0, 0)[:3]
        cfg = F.admission.admission_suite(F.BB)
        key = F.dynamic.EpisodeKey(F.SEED, 0, 0, C.hazard_group("station_keeping"))
        doc, plan = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng)
        site = doc["site"]
        self.assertEqual(site["ladder_option"], C.LADDER_OPTION)
        # G20d: BlueBoat offers D2..D4 (+ D5); the smallest offered main gap is D4, the start gap is D2
        # (G47 / v2.6.4 frozen doses: D4 = 0.76 m, D2 = 3.5 m; before 1.0 / 2.9 m)
        self.assertEqual(site["offered_main_doses"], [2, 3, 4])
        self.assertEqual(site["not_applicable_doses"], [1])
        self.assertEqual(site["smallest_offered_trigger_gap_m"], 0.76)
        self.assertEqual(site["start_gap_m"], 3.5)
        self.assertEqual(P.placement_reasons(context, doc), [])
        for patch, reason in ((dict(nominal_speed_at_smallest_trigger_mps=None), "station_trigger_after_vessel_stops"),
                              (dict(nominal_speed_at_smallest_trigger_mps=0.1), "station_trigger_after_vessel_stops")):
            self.assertIn(reason, P.placement_reasons(context, dict(doc, site=dict(site, **patch))))
        with self.assertRaises(ValueError):                              # D1 is never offered on BlueBoat SK
            P.candidate(context, doc, plan, key, 1, cfg)
        cx, cy = site["station_centre"]
        moved = dict(doc, cylinder=dict(doc["cylinder"], x_m=cx, y_m=cy))
        reasons = P.placement_reasons(context, moved)
        self.assertIn("too_close_to_station", reasons)
        self.assertIn("station_centre_not_clear_at_every_heading", reasons)
        # REV-G20c M6: the G17 evade clearance is NOT an admission rule any more; just inside R_hold + hw
        # (BlueBoat 1.965 m > reach + hw = 1.259 m) is refused, just outside it is not (even inside 2.573 m)
        r = doc["cylinder"]["radius_m"]
        for rho, refused in ((r + 1.5 + 0.465 - 0.01, True), (r + 1.5 + 0.465 + 0.01, False)):
            inside = dict(doc, cylinder=dict(doc["cylinder"], x_m=cx, y_m=cy + rho))
            self.assertEqual("too_close_to_station" in P.placement_reasons(context, inside), refused)

    def test_g17_evade_clearance_pins_follow_the_avoidance_layer(self):
        import json
        layer = json.loads((F.REPO / "scripts" / "classical_avoidance_v2.json").read_text(encoding="utf-8"))
        for vessel in (F.BB, F.WAMV):
            self.assertEqual(P.SK_G17_EVADE_CLEARANCE_M[vessel], layer["vessels"][vessel]["evade_clearance_m"])


class LegacyPools(unittest.TestCase):
    def test_v24_g3_pools_are_untouched(self):
        for (family, name), sha in V24_PINS.items():
            folder = POOL.family_pool_dir(family)
            manifest, _ = POOL.read_manifest(folder)
            self.assertEqual(manifest["pools"][name]["sha256"], sha)
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), sha)

    def test_v24_placements_recompute_with_the_final_code(self):
        for task, vessel in (("scatter", F.BB), ("path_following", F.WAMV)):
            placements = F.legacy_placements(task, vessel)
            for env, episode in sorted(placements)[:6]:
                context, scale, rng = inputs(task, vessel, env, episode)[:3]
                key = F.dynamic.EpisodeKey(F.SEED, env, episode, C.hazard_group(task))
                try:
                    doc, _ = P.place(context, key, F.admission.admission_suite(vessel), length_scale=scale,
                                     radius_range_m=rng)
                except P.PlacementError:
                    doc = None
                stored = placements[(env, episode)]
                if C.LADDER_KIND == "metres":
                    # the metre ladder moves the start gap of the calibration tasks (largest sweep gap), so the
                    # v2.4 sites differ by design; the legacy document keeps the A start gap d_1 = 4 t_avoid
                    if stored is not None and task == "path_following":
                        self.assertAlmostEqual(stored["site"]["start_rule"]["d1_m"],
                                               4.0 * F.envelope(vessel).t_avoid_s, places=9)
                    continue
                if C.LADDER_OPTION != "A" and doc is not None and stored is not None:
                    # the document names the suite version (the ladder option); the rest recomputes byte for byte
                    self.assertEqual(stored["suite_version"], "g3_appearing_gap_v2")
                    doc, stored = dict(doc, suite_version=None), dict(stored, suite_version=None)
                self.assertEqual(doc, stored)


class BundlePins(unittest.TestCase):
    def test_every_task_vessel_has_a_pinned_diag42_pool_in_the_bundle(self):
        pinned = 0
        for task in C.PLACEMENT_TASKS:
            family = FAM.FAMILY_OF_TASK[task]
            folder = POOL.family_pool_dir(family)
            manifest, _ = POOL.read_manifest(folder)
            for vessel in (F.BB, F.WAMV):
                name = POOL.pool_file_name(task, vessel, 42)
                with self.subTest(pool=name):
                    entry = manifest["pools"][name]
                    self.assertEqual((entry["task"], entry["vessel_id"], entry["seed"], entry["tier"]),
                                     (task, vessel, 42, 2))
                    self.assertEqual(entry["grid"], {"num_envs": 64, "episodes": 2})
                    self.assertEqual(entry["keys_offered"], 128)
                    self.assertEqual(entry["keys_admitted"] + entry["keys_not_admitted"], 128)
                    self.assertEqual(len(entry["per_dose_counts"]), C.DOSE_COUNT)
                    self.assertFalse((folder / name).exists())           # the file travels in the bundle
                    pinned += 1
            prov = manifest["provenance"]
            if any(e["file"].endswith("__s42.json.xz") for e in manifest["pools"].values()):
                self.assertEqual(prov["ladder"]["option"], C.LADDER_OPTION)
                self.assertEqual(prov["ladder"]["suite_version"], C.APPEARING_SUITE_VERSION)
        self.assertEqual(pinned, 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
