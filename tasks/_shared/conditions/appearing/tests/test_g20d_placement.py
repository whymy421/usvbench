"""G20d: the owner's appearing-obstacle placement decisions of 2026-10-04 (CPU tests, real journal layouts).

One test class per rule:
  * StationRouteBlock   station keeping: cylinder ON the nominal approach route at 0.45 x the straight-line
                        start-to-hold-centre distance (both vessels; BlueBoat 4.5 m, WAM-V 20.15 m), station
                        clearance, no step-0 trigger above the largest OFFERED gap, every offered main dose triggers
                        on the approach while moving;
  * NotApplicableByDesign BlueBoat station keeping D1: never offered, count row [0, 0, 0], refused by the pool and the
                        candidate, rejected by the stamp reader and the validator helper, distinct from not admitted;
  * DockingMouth        the berth-axis distance from the mouth >= approach run-up + r + hw (numbers), the window end,
                        the "too_close_to_berth_mouth" reason;
  * OptionalSupplementary decision 5B: docking requires D1..D4 only; a key without a D5 witness stays admitted for the
                        main tiers and is not admitted at D5 (own reasons, own count row);
  * MonotoneStitching   the smaller-gap witness spliced onto the larger gap's prefix replays on the larger gap's
                        request; a spliced trajectory that leaves the prefix is refused;
  * FortressRadius      the ray floor, the B17 gap, the radius bound terms and the floor reason;
  * Candidates          16 candidates, same stream and order: a key whose site lies within the first 8 draws gets the
                        same site as with 8;
  * WitnessFallback     the fallback beam (48 -> 192) only after a frozen-beam miss, the reasons, the replay standard,
                        the budget rule after a required-dose failure;
  * Visibility          the ray test, the nominal flag per dose, the runtime stamp and its reader checks, the
                        occluded share of the realized-exposure table.
Run: python -B test_g20d_placement.py
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_fixtures as F  # noqa: E402

C, P, POOL, LAD, S, RT = F.config, F.placement, F.pool_module, F.ladder, F.stamp, F.runtime
FAM = F.families
OCC = F.load("_shared.conditions.appearing.occlusion")
FEAS = F.load("_shared.conditions.dynamic.feasibility")


def inputs(task, vessel, env, episode):
    return tuple(FAM.admission_inputs(task, vessel, F.TIER, F.SEED, env, episode))


def placed(task, vessel, env, episode):
    item = inputs(task, vessel, env, episode)
    context, scale, rng = item[:3]
    regions = item[3] if len(item) > 3 else None
    cfg = F.admission.admission_suite(vessel)
    key = F.key(task, env, episode)
    doc, plan = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng, regions=regions)
    return context, cfg, key, doc, plan, item


def exposure_module():
    path = F.REPO / "scripts" / "appearing_exposure.py"
    spec = importlib.util.spec_from_file_location("_g20d_appearing_exposure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StationRouteBlock(unittest.TestCase):
    def test_site_on_the_route_at_045_of_the_start_distance(self):
        for vessel, d_start, rho in ((F.BB, 10.0, 4.5), (F.WAMV, 44.7791504231185, 20.150617690403326)):
            for env, episode in ((0, 0), (5, 1), (31, 1)):
                with self.subTest(vessel=vessel, key=(env, episode)):
                    context, cfg, key, doc, plan, _ = placed("station_keeping", vessel, env, episode)
                    site, cyl = doc["site"], doc["cylinder"]
                    hold = context.hold_circle
                    self.assertEqual(site["rule"], "station_route_block")
                    self.assertAlmostEqual(site["start_to_centre_distance_m"], d_start, places=9)
                    self.assertAlmostEqual(site["site_centre_distance_m"], rho, places=9)
                    self.assertAlmostEqual(math.hypot(cyl["x_m"] - hold.x, cyl["y_m"] - hold.y), rho, places=9)
                    start = context.initial_pose                                       # J9: straight S-C line
                    self.assertAlmostEqual(cyl["x_m"], start.x + 0.55 * (hold.x - start.x), places=12)
                    self.assertAlmostEqual(cyl["y_m"], start.y + 0.55 * (hold.y - start.y), places=12)
                    self.assertAlmostEqual(doc["nominal_heading_at_site_rad"],
                                           math.atan2(hold.y - start.y, hold.x - start.x), places=12)
                    self.assertLessEqual(site["route_plan_offset_m"], 0.21 if vessel == F.BB else 1.22)
                    self.assertLess(site["nominal_min_gap_m"], 0.0)                    # it blocks the route
                    fp = context.envelope.footprint
                    hw = 0.5 * fp.beam_m
                    surface = rho - cyl["radius_m"]
                    need = max(hold.radius + hw, fp.reach_m + hw)          # REV-G20c M6: scene-only rule
                    self.assertEqual(P.station_clearance_m(context), need)
                    self.assertGreaterEqual(surface, need)
                    self.assertTrue(site["g17_evade_target_clear"])          # report only; true under 0.45 D_start
                    start_gap = P.gap_at(context.initial_pose, cyl["x_m"], cyl["y_m"], cyl["radius_m"], fp)
                    largest_offered = max(C.trigger_gap_m(d, context.envelope.t_avoid_s, vessel)
                                          for d in C.offered_doses("station_keeping", vessel))
                    self.assertGreater(start_gap, largest_offered)                     # never at step 0
                    if vessel == F.BB:      # D1 would trigger at step 0 here: that is why it is not offered
                        self.assertLessEqual(start_gap, C.trigger_gap_m(1, context.envelope.t_avoid_s, vessel))
                    for dose in LAD.offered_main_doses("station_keeping", vessel):
                        cand, _ = P.candidate(context, doc, plan, key, dose, cfg)
                        self.assertGreaterEqual(cand["nominal_trigger_speed_mps"], 0.25)
                        self.assertGreater(cand["nominal_trigger_s"], 0.0)

    def test_curved_plan_first_crossing_is_recorded(self):
        # J9: the site is the straight-line point; the curved D7 plan's first crossing is reported only
        context, cfg, key, doc, plan, _ = placed("station_keeping", F.BB, 3, 0)
        t, pose, d_start, rho = P.station_route_point(context, plan)
        self.assertEqual(doc["site"]["route_plan_first_crossing_xy"], [pose.x, pose.y])
        hold = context.hold_circle
        self.assertAlmostEqual(math.hypot(pose.x - hold.x, pose.y - hold.y), rho, places=9)
        step = t / 400.0
        for k in range(400):
            p = plan.pose_at(k * step)
            self.assertGreater(math.hypot(p.x - hold.x, p.y - hold.y), rho - 1e-9)


class NotApplicableByDesign(unittest.TestCase):
    def test_ladder_pins(self):
        self.assertEqual(LAD.not_applicable_doses("station_keeping", F.BB), (1,))
        self.assertEqual(LAD.not_applicable_doses("station_keeping", F.WAMV), ())
        self.assertEqual(LAD.offered_doses("station_keeping", F.BB), (2, 3, 4, 5))
        self.assertEqual(LAD.offered_main_doses("station_keeping", F.BB), (2, 3, 4))
        self.assertEqual(LAD.start_gap_m("station_keeping", F.BB), 3.5)      # G47 (v2.6.4): D2 of the frozen table
        self.assertEqual(LAD.start_gap_m("station_keeping", F.WAMV), 15.0)   # G47: D1
        # G48/G49 (owner 2026-10-14): BlueBoat docking D1 is not applicable by design too -> start gap D2
        self.assertEqual(LAD.not_applicable_doses("docking", F.BB), (1,))
        self.assertEqual(LAD.not_applicable_doses("docking", F.WAMV), ())
        self.assertEqual(LAD.start_gap_m("docking", F.BB), 3.5)
        for task in C.PLACEMENT_TASKS:
            if task not in ("station_keeping", "docking"):
                for vessel in (F.BB, F.WAMV):
                    self.assertEqual(LAD.not_applicable_doses(task, vessel), ())

    def test_pool_counts_header_and_refusal(self):
        pool = F.small_pool("station_keeping", F.BB, num_envs=2, episodes=1)
        rows = pool.per_dose_counts
        self.assertEqual(rows[1], (0, 0, 0))
        for d in (0, 2, 3, 4, 5):
            self.assertEqual(rows[d][0], 2)
        block = pool.header_block()
        self.assertEqual(block["dose_applicability"]["D1"], LAD.NOT_APPLICABLE)
        self.assertIn("D1", block["not_applicable_doses"])
        self.assertEqual(block["admission"]["offered_doses"], [2, 3, 4, 5])
        with self.assertRaises(ValueError):
            pool.assignment(F.key("station_keeping", 0, 0), 1)
        with self.assertRaises(ValueError):
            pool.check_dose_applicable(1)
        self.assertEqual(pool.assignment(F.key("station_keeping", 0, 0), 2).exposure, "admitted")
        # not applicable is not "not admitted": no key carries a D1 reason
        for env, episode in pool.offered:
            self.assertFalse(any(d == 1 for d, _ in pool._keys[(env, episode)]["reasons"]))

    def test_stamp_and_validator_reject_the_not_applicable_dose(self):
        pool = F.small_pool("station_keeping", F.BB, num_envs=2, episodes=1)
        a = pool.assignment(F.key("station_keeping", 0, 0), 2)
        summary = a.summary(1)
        runtime = {"assigned": True, "itt_count": 1, "exposure_realized": False, "trigger": None,
                   "trigger_gap_m": a.trigger_gap_m, "k_equivalent": C.k_equivalent(a.trigger_gap_m, a.candidate[
                       "t_avoid_s"]), "visible_at_trigger": None}
        good = S.make_appearing_stamp("admitted", summary, runtime)
        S.read_appearing_stamp(good)
        bad_summary = dict(summary, dose=1, trigger_gap_m=5.0, k_equivalent=C.k_equivalent(5.0, a.candidate["t_avoid_s"]))
        bad = S.make_appearing_stamp("admitted", bad_summary, dict(runtime, trigger_gap_m=5.0,
                                                                   k_equivalent=bad_summary["k_equivalent"]))
        with self.assertRaisesRegex(ValueError, "not applicable by design"):
            S.read_appearing_stamp(bad)
        EX = exposure_module()
        header = {"hazard_group": C.hazard_group("station_keeping"), "suite": {"vessel_id": F.BB}, "dose": 2,
                  **pool.header_block()}
        EX.applicability_check(header)
        for patch in ({"dose": 1}, {"dose_applicability": dict(header["dose_applicability"], D1="offered")},
                      {"per_dose_counts": [[2, 2, 0], [2, 2, 0]] + header["per_dose_counts"][2:]}):
            with self.assertRaises(ValueError):
                EX.applicability_check(dict(header, **patch))


class DockingMouth(unittest.TestCase):
    def test_numbers_and_rule(self):
        for vessel, want in ((F.BB, 2.401651692390442 + 0.8 + 0.465), (F.WAMV, 10.754392239748872 +
                                                                         3.5823320338494806 + 1.267135)):
            for env, episode in ((1, 0), (0, 0), (2, 1), (3, 1)):
                try:
                    context, cfg, key, doc, plan, item = placed("docking", vessel, env, episode)
                    break
                except P.PlacementError:
                    continue
            mouth = doc["site"]["berth_mouth_rule"]
            self.assertAlmostEqual(mouth["min_axis_distance_m"], want, places=9)
            self.assertEqual(mouth["approach_runup_m"], F.plans.vessel_record(vessel).geometry.approach_runup_m)
            berth = doc["site"]["berth_exclusion"]
            cyl = doc["cylinder"]
            self.assertGreaterEqual(mouth["opening_x_m"] - P.berth_axis_x(berth, cyl["x_m"], cyl["y_m"]),
                                    want - 1e-9)
            self.assertLessEqual(doc["site"]["window_m"][1], mouth["window_end_m"] + 1e-12)
            # move the cylinder to the mouth line: refused
            ux, uy = math.cos(berth["heading_rad"]), math.sin(berth["heading_rad"])
            bx = P.berth_axis_x(berth, cyl["x_m"], cyl["y_m"])
            shift = mouth["opening_x_m"] - bx - 0.5 * want
            moved = dict(doc, cylinder=dict(cyl, x_m=cyl["x_m"] + shift * ux, y_m=cyl["y_m"] + shift * uy))
            self.assertIn("too_close_to_berth_mouth", P.placement_reasons(context, moved))
            with self.assertRaises(ValueError):                    # the docking family must supply the mouth block
                regions = dict(item[3])
                regions.pop("mouth")
                P.place(context, key, cfg, length_scale=item[1], radius_range_m=item[2], regions=regions)


class OptionalSupplementary(unittest.TestCase):
    def test_policy(self):
        self.assertEqual(C.required_doses("docking", F.BB), (2, 3, 4))           # G48/G49: D1 n/a by design
        self.assertEqual(C.required_doses("docking", F.WAMV), (1, 2, 3, 4))
        self.assertEqual(C.optional_doses("docking", F.BB), (5,))
        self.assertEqual(C.required_doses("scatter", F.BB), (1, 2, 3, 4, 5))
        self.assertEqual(C.optional_doses("scatter", F.BB), ())
        self.assertEqual(C.required_doses("station_keeping", F.BB), (2, 3, 4, 5))

    def test_admitted_main_tiers_without_d5(self):
        pool = F.small_pool("docking", F.BB, num_envs=2, episodes=1)
        env, episode = F.first_admitted(pool)
        item = pool._keys[(env, episode)]
        doses = {d: v for d, v in item["doses"].items() if d != 5}
        record = {"env": env, "episode": episode, "geometry_id": item["geometry_id"], "placement": item["placement"],
                  "reasons": [], "doses": doses, "optional_reasons": [(5, "no_witness_in_frozen_search")]}
        others = []
        for ident in pool.offered:
            if ident != (env, episode):
                it = pool._keys[ident]
                others.append({"env": ident[0], "episode": ident[1], "geometry_id": it["geometry_id"],
                               "placement": it["placement"], "reasons": list(it["reasons"]), "doses": dict(it["doses"]),
                               "optional_reasons": list(it["optional_reasons"])})
        made = POOL.AppearingPool(task="docking", vessel_id=F.BB, seed=F.SEED, tier=F.TIER,
                                  suite_identity=pool.suite_identity, num_envs=2, episodes=1,
                                  keys=sorted([record] + others, key=lambda r: (r["env"], r["episode"])))
        self.assertFalse(made.joint)
        self.assertEqual(made.per_dose_counts[4], (2, made.keys_admitted, 2 - made.keys_admitted))
        with5 = sum(1 for i in made.offered if not made._keys[i]["reasons"] and 5 in made._keys[i]["doses"])
        self.assertEqual(made.per_dose_counts[5], (2, with5, 2 - with5))
        self.assertLess(with5, made.keys_admitted)
        k = F.key("docking", env, episode)
        self.assertEqual(made.assignment(k, 4).exposure, "admitted")
        at5 = made.assignment(k, 5)
        self.assertEqual(at5.exposure, "not_admitted")
        self.assertEqual(at5.reason_strings(), ["no_witness_in_frozen_search"])
        self.assertIn([env, episode, [[5, "no_witness_in_frozen_search"]]], made.optional_not_admitted_keys())
        # the codec round-trips the optional reasons and seals them
        doc = POOL.pool_document(made)
        again = POOL.pool_from_document(doc)
        self.assertEqual(again.pool_digest, made.pool_digest)
        # a required dose may not be missing; a joint task may not carry optional reasons
        with self.assertRaises(ValueError):
            POOL.AppearingPool(task="docking", vessel_id=F.BB, seed=F.SEED, tier=F.TIER,
                               suite_identity=pool.suite_identity, num_envs=1, episodes=1,
                               keys=[dict(record, env=0, episode=0, doses={d: v for d, v in doses.items() if d != 4},
                                          optional_reasons=[(5, "x")])])
        scat = F.small_pool("scatter", F.BB, num_envs=1, episodes=1)
        it = scat._keys[(0, 0)]
        if not it["reasons"]:
            with self.assertRaises(ValueError):
                POOL.AppearingPool(task="scatter", vessel_id=F.BB, seed=F.SEED, tier=F.TIER,
                                   suite_identity=scat.suite_identity, num_envs=1, episodes=1,
                                   keys=[{"env": 0, "episode": 0, "geometry_id": it["geometry_id"],
                                          "placement": it["placement"], "reasons": [],
                                          "doses": {d: v for d, v in it["doses"].items() if d != 5},
                                          "optional_reasons": [(5, "x")]}])
        tampered = dict(doc, admission=dict(doc["admission"], required_doses=[1, 2, 3, 4, 5]))
        with self.assertRaises(ValueError):
            POOL.pool_from_document(tampered)


class MonotoneStitching(unittest.TestCase):
    def test_smaller_gap_witness_spliced_onto_the_larger_gap_prefix(self):
        context, cfg, key, doc, plan, _ = placed("docking", F.BB, 3, 1)
        _, req4 = P.candidate(context, doc, plan, key, 4, cfg)
        _, req2 = P.candidate(context, doc, plan, key, 2, cfg)
        w4, reason, _ = P.witness_outcome(req4)
        self.assertIsNotNone(w4, reason)
        self.assertFalse(FEAS.replay_witness(req2, w4.trajectory).accepted)      # raw: nominal_prefix_changed
        stitched = P.stitch_witness(req2, w4)
        self.assertIsNotNone(stitched)
        self.assertTrue(FEAS.replay_witness(req2, stitched.trajectory).accepted)
        prefix = FEAS._prefix(req2)
        self.assertEqual(stitched.trajectory[:len(prefix)], tuple(prefix))
        # a source whose samples after the prefix do not continue it kinematically is refused by the replay
        k = len(prefix) + 3
        state = w4.trajectory[k]
        bent = state.__class__(state.time_s, dataclasses.replace(state.pose, x=state.pose.x + 0.5), state.accel_mps2,
                               state.yaw_rate_rps)
        broken = dataclasses.replace(w4, trajectory=w4.trajectory[:k] + (bent,) + w4.trajectory[k + 1:])
        self.assertIsNone(P.stitch_witness(req2, broken))

    def test_admission_uses_stitching_only_on_docking(self):
        self.assertEqual(C.MONOTONE_STITCH_TASKS, ("docking",))
        self.assertEqual(P.STITCH_REASON % 4, "monotone_stitched_from_D4")


class FortressRadius(unittest.TestCase):
    def test_floor_and_gap_numbers(self):
        self.assertAlmostEqual(P.ray_floor_radius_m(5.0, 0.7939657461288199, 36), 0.5531911758730093, places=12)
        self.assertAlmostEqual(P.ray_floor_radius_m(5.0, 3.091559871619274, 36), 0.7725588510803049, places=12)
        for vessel, g in ((F.BB, 2.93), (F.WAMV, 11.4901000846237)):
            region = inputs("fortress", vessel, 0, 1)[3]["fortress"]
            self.assertAlmostEqual(region["passage_gap_m"], g, places=9)
            self.assertEqual(len(region["gate_openings"]), 2)
            for gate in region["gate_openings"]:
                (ax, ay), (bx, by) = gate["segment"]
                self.assertAlmostEqual(math.hypot(bx - ax, by - ay), gate["physical_gap_m"], places=6)

    def test_radius_rule_terms_and_reason(self):
        found = 0
        for vessel in (F.BB, F.WAMV):
            for env, episode in ((1, 0), (2, 0), (3, 1), (4, 0)):
                try:
                    context, cfg, key, doc, plan, item = placed("fortress", vessel, env, episode)
                except P.PlacementError as error:
                    self.assertEqual(error.reason, "no_clear_site_in_route_window")
                    continue
                found += 1
                rule, cyl = doc["site"]["fortress_radius_rule"], doc["cylinder"]
                r0 = C.RADIUS_LENGTH_FACTOR * item[1]
                bounds = [r0] + [rule[k] for k in ("passage_bound_m", "gate_bound_m", "static_clearance_bound_m")
                                 if rule[k] is not None]
                self.assertEqual(cyl["radius_m"], min(bounds))
                self.assertGreaterEqual(cyl["radius_m"], rule["ray_floor_m"])
                pose = plan.pose_at(doc["site"]["t_site_s"])
                value, detail = P.fortress_radius(P._fortress_inputs(item[3]["fortress"], context,
                                                                     rule["start_gap_m"], r0), context, pose)
                self.assertEqual((value, detail), (cyl["radius_m"], rule))
                small = dict(doc, cylinder=dict(cyl, radius_m=0.5 * rule["ray_floor_m"]))
                self.assertIn("fortress_radius_below_ray_floor", P.placement_reasons(context, small))
        self.assertGreaterEqual(found, 4)


class Candidates(unittest.TestCase):
    def test_sixteen_candidates_same_stream_prefix(self):
        self.assertEqual((P.SCATTER_MAX_SITE_CANDIDATES, P.ROUTE_WINDOW_MAX_SITE_CANDIDATES), (16, 16))
        for task, vessel in (("scatter", F.WAMV), ("reach", F.BB), ("path_hazard", F.BB)):
            for env, episode in ((0, 0), (2, 1)):
                with self.subTest(task=task, vessel=vessel, key=(env, episode)):
                    try:
                        _, _, _, doc16, _, _ = placed(task, vessel, env, episode)
                    except P.PlacementError:
                        doc16 = None
                    saved = (P.SCATTER_MAX_SITE_CANDIDATES, P.ROUTE_WINDOW_MAX_SITE_CANDIDATES)
                    P.SCATTER_MAX_SITE_CANDIDATES = P.ROUTE_WINDOW_MAX_SITE_CANDIDATES = 8
                    try:
                        try:
                            _, _, _, doc8, _, _ = placed(task, vessel, env, episode)
                        except P.PlacementError:
                            doc8 = None
                    finally:
                        P.SCATTER_MAX_SITE_CANDIDATES, P.ROUTE_WINDOW_MAX_SITE_CANDIDATES = saved
                    if doc8 is not None:          # found within 8: the same site with 16
                        self.assertIsNotNone(doc16)
                        self.assertEqual(doc8["cylinder"], doc16["cylinder"])
                        self.assertEqual(doc8["site"]["candidate_index"], doc16["site"]["candidate_index"])
                        self.assertEqual(doc16["site"]["max_candidates"], 16)
                    elif doc16 is not None:
                        self.assertGreaterEqual(doc16["site"]["candidate_index"], 8)


class WitnessFallback(unittest.TestCase):
    def setUp(self):
        context, cfg, key, doc, plan, _ = placed("reach", F.BB, 0, 0)
        _, self.request = P.candidate(context, doc, plan, key, 2, cfg)
        self.real = P.find_witness
        self.calls = []

    def tearDown(self):
        P.find_witness = self.real

    def fake(self, frozen_ok, wide_ok):
        def find(request):
            self.calls.append(request.beam_width)
            result = self.real(request)
            ok = frozen_ok if request.beam_width == 48 else wide_ok
            if ok:
                return result
            return FEAS.WitnessResult(False, "no_witness_in_frozen_search", sampled_states=7, search_exhausted=True)
        P.find_witness = find

    def test_constants(self):
        self.assertEqual(C.AppearingSuiteConfig().witness_beam_width, 48)
        self.assertEqual(C.AppearingSuiteConfig().witness_fallback_beam_width, 192)
        self.assertIn("witness_fallback_beam_width", C.AppearingSuiteConfig().identity())
        with self.assertRaises(ValueError):
            C.AppearingSuiteConfig(witness_fallback_beam_width=48)

    def test_fallback_only_after_a_frozen_miss(self):
        self.fake(True, True)
        result, reason, _ = P.witness_outcome(self.request, 192)
        self.assertEqual(self.calls, [48])
        self.assertFalse(result.reason.endswith("_fallback_beam_192"))
        self.calls.clear()
        self.fake(False, True)
        result, reason, _ = P.witness_outcome(self.request, 192)
        self.assertEqual(self.calls, [48, 192])
        self.assertTrue(result.reason.endswith("_fallback_beam_192"))
        self.assertTrue(FEAS.replay_witness(self.request, result.trajectory).accepted)
        self.calls.clear()
        self.fake(False, False)
        result, reason, _ = P.witness_outcome(self.request, 192)
        self.assertIsNone(result)
        self.assertEqual(reason, "no_witness_in_frozen_search_or_fallback_beam_192")
        self.calls.clear()
        result, reason, _ = P.witness_outcome(self.request, 0)                     # budget rule: frozen only
        self.assertEqual((self.calls, reason), ([48], "no_witness_in_frozen_search"))

    def test_budget_rule_after_a_required_failure(self):
        context, cfg, key, doc, plan, item = placed("reach", F.BB, 0, 0)
        widths = []

        def outcome(request, fallback=0):
            widths.append(fallback)
            return None, "no_witness_in_frozen_search", 0
        saved = P.witness_outcome
        P.witness_outcome = outcome
        try:
            out = P.admit_key(context, key, cfg, length_scale=item[1], radius_range_m=item[2])
        finally:
            P.witness_outcome = saved
        self.assertEqual(widths, [192, 0, 0, 0, 0])
        self.assertEqual([d for d, _ in out["reasons"]], [1, 2, 3, 4, 5])


class Visibility(unittest.TestCase):
    def test_ray_test(self):
        cyl = (5.0, 0.0, 0.8)
        self.assertGreater(OCC.visible_rays(0.0, 0.0, 0.0, cyl, [], 30.0), 0)
        wall = [(2.5, 0.0, 1.5)]                                    # a disc between the COM and the cylinder
        self.assertEqual(OCC.visible_rays(0.0, 0.0, 0.0, cyl, wall, 30.0), 0)
        self.assertEqual(OCC.visible_rays(0.0, 0.0, 0.0, cyl, [], 3.0), 0)    # beyond the sensor range
        self.assertEqual(OCC.ray_entry(0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0), 4.0)
        self.assertIsNone(OCC.ray_entry(0.0, 0.0, -1.0, 0.0, 5.0, 0.0, 1.0))

    def test_same_answer_as_the_hazard_nav_ray_kernel(self):
        # REV-G20c's env-side recipe: visible iff some ray of the env's own kernel (hazard_geometry.ray_circle_ranges,
        # 36 rays from the COM at heading + 2 pi k / 36) is strictly shorter with the appearing slot than without it
        import random
        import torch
        geo = F.load("hazard_nav.hazard_geometry")
        rng = random.Random(7)
        checked = 0
        for task, vessel, env, episode in (("ring", F.BB, 2, 1), ("fortress", F.BB, 1, 0), ("fortress", F.WAMV, 3, 0)):
            context = inputs(task, vessel, env, episode)[0]
            occ = [(o.x, o.y, o.radius) for o in context.static_obstacles]
            rmax = context.sensor.max_range_m
            xs = [o[0] for o in occ]
            ys = [o[1] for o in occ]
            for _ in range(60):
                px, py = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
                cx, cy = px + rng.uniform(-12.0, 12.0), py + rng.uniform(-12.0, 12.0)
                yaw, r = rng.uniform(-math.pi, math.pi), rng.uniform(0.5, 1.5)
                if any(math.hypot(px - ox, py - oy) <= orad for ox, oy, orad in occ + [(cx, cy, r)]):
                    continue
                angles = torch.arange(36, dtype=torch.float64) * (2.0 * math.pi / 36) + yaw
                dirs = torch.stack((torch.cos(angles), torch.sin(angles)), -1).unsqueeze(0)
                origin = torch.tensor([[px, py]], dtype=torch.float64)
                cen = torch.tensor([[[o[0], o[1]] for o in occ]], dtype=torch.float64)
                rad = torch.tensor([[o[2] for o in occ]], dtype=torch.float64)
                without = geo.ray_circle_ranges(origin, dirs, cen, rad, max_range_m=rmax)
                cen2 = torch.cat((cen, torch.tensor([[[cx, cy]]], dtype=torch.float64)), 1)
                rad2 = torch.cat((rad, torch.tensor([[r]], dtype=torch.float64)), 1)
                with_slot = geo.ray_circle_ranges(origin, dirs, cen2, rad2, max_range_m=rmax)
                kernel = int((with_slot < without).sum())
                self.assertEqual(OCC.visible_rays(px, py, yaw, (cx, cy, r), occ, rmax, 36), kernel)
                checked += 1
        self.assertGreater(checked, 100)

    def test_nominal_flag_per_dose_and_runtime_stamp(self):
        pool = F.small_pool("ring", F.BB, num_envs=2, episodes=1)
        env, episode = F.first_admitted(pool)
        doc = pool.placement(env, episode)
        self.assertEqual(doc["visibility"]["ray_count"], 36)
        self.assertEqual(len(doc["visibility"]["occluders"]), len(inputs("ring", F.BB, env, episode)[0].static_obstacles))
        for dose in C.WITNESSED_DOSES:
            a = pool.assignment(F.key("ring", env, episode), dose)
            cand = a.candidate
            if cand["nominal_trigger_s"] is not None:
                self.assertIsInstance(cand["nominal_visible_at_trigger"], bool)
                self.assertEqual(cand["nominal_visible_at_trigger"], cand["nominal_visible_rays"] > 0)
            self.assertEqual(a.summary(5)["nominal_visible_at_trigger"], cand["nominal_visible_at_trigger"])
        # runtime: drive straight at the cylinder; the trigger record carries the visibility and the reader checks it
        a = pool.assignment(F.key("ring", env, episode), 1)
        x, y, r = a.cylinder
        env_fp = F.envelope(F.BB)
        bridge = RT.AppearingBridge("ring", 9, 1.0 / 120.0, env_fp.top_speed_mps, env_fp.footprint, env_fp.t_avoid_s)
        bridge.reset_rows((env,), {env: a})
        start = (x - 12.0, y)
        F.straight_run(bridge, env, start, 0.0, 1.0, 1.0 / 120.0, 2, 900)
        stamp = bridge.finish(env, "timeout")
        rt = stamp["payload"]["runtime"]
        self.assertTrue(rt["exposure_realized"])
        trig = rt["trigger"]
        pose = trig["pose"]
        want = OCC.visible_rays(pose[0], pose[1], pose[2], (x, y, r), doc["visibility"]["occluders"],
                                doc["visibility"]["max_range_m"], 36)
        self.assertEqual(trig["visible_rays"], want)
        self.assertEqual(rt["visible_at_trigger"], want > 0)
        S.read_appearing_stamp(stamp)
        payload = stamp["payload"]
        for runtime in (dict(payload["runtime"], visible_at_trigger=not rt["visible_at_trigger"]),
                        {k: v for k, v in payload["runtime"].items() if k != "visible_at_trigger"},
                        dict(payload["runtime"], trigger=dict(trig, visible_rays=-1))):
            forged = {"payload": dict(payload, runtime=runtime)}
            forged["sha256"] = S.digest(forged["payload"])
            with self.assertRaises(ValueError):
                S.read_appearing_stamp(forged)

    def test_reader_rejects_impossible_stamps(self):
        # REV-G20c m1: D0 / not-admitted stamps never claim an appearance; the trigger is the first check at or
        # below d_k; its time is tick x physics_dt_s
        pool = F.small_pool("ring", F.BB, num_envs=2, episodes=1)
        env, episode = F.first_admitted(pool)
        env_fp = F.envelope(F.BB)
        stamps = {}
        for dose in (0, 1):
            a = pool.assignment(F.key("ring", env, episode), dose)
            bridge = RT.AppearingBridge("ring", 9, 1.0 / 120.0, env_fp.top_speed_mps, env_fp.footprint,
                                        env_fp.t_avoid_s)
            bridge.reset_rows((env,), {env: a})
            x, y, _ = pool.assignment(F.key("ring", env, episode), 1).cylinder
            F.straight_run(bridge, env, (x - 12.0, y), 0.0, 1.0, 1.0 / 120.0, 2, 900)
            stamps[dose] = bridge.finish(env, "timeout")
            S.read_appearing_stamp(stamps[dose])
        real = stamps[1]["payload"]

        def forged(payload):
            out = {"payload": payload}
            out["sha256"] = S.digest(payload)
            return out
        d0 = stamps[0]["payload"]
        cases = [dict(d0, runtime=dict(d0["runtime"], exposure_realized=True, trigger=real["runtime"]["trigger"],
                                       non_exposure_reason=None)),
                 dict(real, runtime=dict(real["runtime"], trigger=dict(real["runtime"]["trigger"],
                                                                       previous_check_gap_m=0.5))),
                 dict(real, runtime=dict(real["runtime"], trigger=dict(real["runtime"]["trigger"],
                                                                       time_s=real["runtime"]["trigger"]["time_s"] + 0.01)))]
        for payload in cases:
            with self.assertRaises(ValueError):
                S.read_appearing_stamp(forged(payload))

    def test_occluded_share_in_the_realized_table(self):
        EX = exposure_module()
        pool = F.small_pool("ring", F.BB, num_envs=2, episodes=1)
        env, episode = F.first_admitted(pool)
        a = pool.assignment(F.key("ring", env, episode), 2)
        records = []
        for visible in (True, False, True):
            trig = {"tick": 10, "time_s": 1.0, "gap_m": 1.0, "bearing_rad": 0.0, "pose": [0.0, 0.0, 0.0, 1.0],
                    "visible_at_trigger": visible, "visible_rays": 2 if visible else 0,
                    **S.trigger_pressure(1.0, 0.0, 1.0)}
            runtime = {"assigned": True, "itt_count": 1, "exposure_realized": True, "trigger": trig,
                       "physics_dt_s": 0.1,
                       "trigger_gap_m": a.trigger_gap_m, "visible_at_trigger": visible,
                       "k_equivalent": C.k_equivalent(a.trigger_gap_m, a.candidate["t_avoid_s"])}
            records.append({"appearing_hazards": S.make_appearing_stamp("admitted", a.summary(1), runtime)})
        header = {"hazard_group": C.hazard_group("ring"), "suite": {"vessel_id": F.BB}, "dose": 2,
                  "trigger_gaps_m": [None, 5.0, 2.9, 1.7, 1.0, 0.76], "controller": "x"}
        rows = EX.realized_exposure_table([{"header": {"appearing_hazards_header": header, "controller": "c"},
                                            "records": records}])
        self.assertEqual((rows[0]["realized_visible"], rows[0]["realized_occluded"]), (2, 1))
        self.assertAlmostEqual(rows[0]["occluded_share"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
