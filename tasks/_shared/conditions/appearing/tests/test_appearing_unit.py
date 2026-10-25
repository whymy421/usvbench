"""G3 appearing suite: unit tests of the shared module (CPU only; run: python -B test_appearing_unit.py).

Covers the controller rules as implemented: dose gaps d_k = k * t_avoid * u_ref
for both vessels (rule 2), the radius rule numbers (rule 4), the slot rule
(rule 5, D7 CapacityPlan.appearing_slots), placement on the nominal route with
the same position at every dose (rule 3), the admission reasons and the D16 /
D24 lookups with [offered, admitted, not_admitted] counts (rule 6), the
digest-sealed stamps and D27 invariants (rule 7), the bridge trigger at the
declared gap within one control step, absent before and physical + visible
after, never-reached and D0 rows, the swept-contact latch (rule 5), and the
pool file codec with its fail-closed loader. G20b: the dose rule follows the
ladder switch (ladder.ACTIVE_LADDER, default B); pools come from in-memory
reference grids (F.reference_pool), the production pools being in the bundle.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_fixtures as F  # noqa: E402

C, ST, P, POOL, RT = F.config, F.stamp, F.placement, F.pool_module, F.runtime
PHYSICS_DT, DECIMATION = 1.0 / 120.0, 2


# declared trigger gaps (m, rounded to 0.01) per ladder option: A = brief rule 2 (G3), B = G20 recommendation
# G47 (v2.6.4): M = the owner's frozen doses of 2026-10-14 (D4 = D5 = the calibration rung)
DECLARED_GAPS = {"M": {F.BB: (7.5, 3.5, 1.6, 0.76, 0.76), F.WAMV: (15.0, 6.6, 2.9, 1.25, 1.25)},
                 "A": {F.BB: (6.58, 4.94, 3.29, 2.06), F.WAMV: (14.97, 11.23, 7.49, 4.68)},
                 "B": {F.BB: (6.58, 3.29, 1.65, 0.82), F.WAMV: (14.97, 7.49, 3.74, 1.87)},
                 "E": {F.BB: (6.58, 4.94, 3.29, 2.06, 0.82), F.WAMV: (14.97, 11.23, 7.49, 4.68, 1.87)}}


class DoseRule(unittest.TestCase):
    def test_trigger_gaps_match_the_declared_numbers(self):
        expected = {F.BB: 1.645, F.WAMV: 3.743}
        for vessel, t_avoid in expected.items():
            gaps = DECLARED_GAPS[C.LADDER_OPTION][vessel]
            env = F.envelope(vessel)
            self.assertAlmostEqual(env.t_avoid_s, t_avoid, places=3)
            values = C.trigger_gaps_m(env.t_avoid_s)
            self.assertIsNone(values[0])
            for k, (value, want) in enumerate(zip(values[1:], gaps), start=1):
                if C.LADDER_KIND == "k":
                    self.assertAlmostEqual(value, C.GAP_RATIOS[k] * env.t_avoid_s * 1.0, places=12)
                else:                                    # metre ladder: the vessel's table, k only stamped
                    self.assertEqual(value, F.ladder.GAPS_M[vessel][k])
                    self.assertAlmostEqual(C.k_equivalent(value, env.t_avoid_s), value / env.t_avoid_s, places=12)
                self.assertEqual(value, C.trigger_gap_m(k, env.t_avoid_s, vessel))
                self.assertAlmostEqual(round(value, 2), want, places=6, msg=(vessel, k))
            self.assertEqual(len(values), C.DOSE_COUNT)
        self.assertEqual(C.GAP_RATIOS, F.ladder.ladder_constants(C.LADDER_OPTION)["gap_ratios"])
        self.assertEqual(C.REFERENCE_SPEED_MPS, 1.0)

    def test_slot_is_the_d7_appearing_reserve(self):
        for task in C.WIRED_TASKS:
            base, reserve = F.plans.STATIC_CAPACITY[task], F.plans.D7_RESERVED_SLOTS[task]
            plan = C.capacity_plan(base, reserve)
            self.assertEqual(plan.appearing_slots, (base + reserve,))
            self.assertEqual(C.appearing_slot(base, reserve), base + reserve)
        self.assertEqual(C.appearing_slot(14, 1), 15)        # scatter: after the D7 slot 14
        self.assertEqual(C.appearing_slot(0, 1), 1)          # path following: after the D7 slot 0

    def test_radius_rule_numbers(self):
        s_wamv = F.plans.vessel_record(F.WAMV).geometry.length_scale
        self.assertAlmostEqual(s_wamv, 4.47791504231185, places=10)
        bb = P.radius_rule("scatter", 1.0, (0.8, 2.0))
        self.assertEqual((bb["rule"], bb["low_m"], bb["high_m"]), ("task_static_radius_range", 0.8, 2.0))
        _, scale, rng = F.hazard.appearing_admission_inputs("scatter", F.WAMV, 2, 42, 0, 0)
        self.assertAlmostEqual(rng[0], 0.8 * s_wamv)
        self.assertAlmostEqual(rng[1], 2.0 * s_wamv)
        pf_bb = P.radius_rule("path_following", 1.0)
        pf_wamv = P.radius_rule("path_following", s_wamv)
        self.assertEqual(pf_bb["low_m"], 0.8)
        self.assertAlmostEqual(pf_wamv["low_m"], 3.582332033849481, places=9)

    def test_suite_config_validation(self):
        with self.assertRaises(ValueError):
            C.AppearingSuiteConfig(enabled=False, dose=2)
        with self.assertRaises(ValueError):
            C.AppearingSuiteConfig(enabled=True, dose=C.MAX_DOSE + 1, vessel_id=F.BB)
        with self.assertRaises(ValueError):
            C.AppearingSuiteConfig(enabled=True, dose=1)          # an enabled suite names its vessel
        cfg = C.AppearingSuiteConfig(enabled=True, dose=0, vessel_id=F.BB)
        self.assertEqual(cfg.at_dose(3).dose, 3)
        self.assertNotIn("dose", cfg.identity())
        for task in C.PLACEMENT_TASKS:            # G20b: every journal task is wired
            self.assertEqual(C.check_wired(task), task)
        with self.assertRaises(ValueError):
            C.check_wired("no_such_task")


class Placement(unittest.TestCase):
    def test_position_is_on_the_nominal_route_and_dose_independent(self):
        for task, vessel in (("scatter", F.BB), ("path_following", F.WAMV)):
            context, scale, rng = F.inputs(task, vessel)(0, 1)
            cfg = F.admission.admission_suite(vessel)
            key = F.key(task, 0, 1)
            doc, plan = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng)
            again, _ = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng)
            self.assertEqual(doc, again)                                  # deterministic, keyed
            t_site = doc["site"]["t_site_s"]
            pose = plan.pose_at(t_site)
            self.assertEqual((doc["cylinder"]["x_m"], doc["cylinder"]["y_m"]), (pose.x, pose.y))
            if task == "scatter":
                self.assertTrue(0.4 <= doc["site"]["fraction"] <= 0.6)
                self.assertTrue(rng[0] <= doc["cylinder"]["radius_m"] <= rng[1])
            else:
                self.assertEqual(doc["site"]["rule"], "leg_midpoint_start_clear")
                self.assertEqual(doc["site"]["leg_index"], context.site_index)
                self.assertGreaterEqual(doc["site"]["route_distance_m"], doc["site"]["min_route_distance_m"] - 1e-9)
            digests = set()
            for dose in C.WITNESSED_DOSES:
                cand, request = P.candidate(context, doc, plan, key, dose, cfg)
                digests.add(cand["position_digest"])
                self.assertEqual(request.encounter.intruders[0].x, doc["cylinder"]["x_m"])
                self.assertAlmostEqual(cand["trigger_gap_m"], C.trigger_gap_m(dose, context.envelope.t_avoid_s))
                self.assertLessEqual(cand["nominal_trigger_gap_m"], cand["trigger_gap_m"])
                self.assertGreater(cand["nominal_trigger_gap_m"], cand["trigger_gap_m"] - 1e-9)
                self.assertAlmostEqual(request.prefix[-1].time_s,
                                       cand["nominal_trigger_s"] + context.envelope.reaction_s, places=9)
            self.assertEqual(len(digests), 1)

    def test_trigger_times_are_ordered_by_dose(self):
        context, scale, rng = F.inputs("path_following", F.WAMV)(1, 0)
        cfg = F.admission.admission_suite(F.WAMV)
        key = F.key("path_following", 1, 0)
        doc, plan = P.place(context, key, cfg, length_scale=scale, radius_range_m=rng)
        times = [P.candidate(context, doc, plan, key, dose, cfg)[0]["nominal_trigger_s"] for dose in C.WITNESSED_DOSES]
        self.assertEqual(times, sorted(times))

    def test_admission_reasons_are_recorded_jointly(self):
        pool = F.small_pool("scatter", F.BB, num_envs=64, episodes=2)
        rejected = [(e, k) for e, k in pool.offered if not pool.is_admitted(e, k)]
        self.assertTrue(rejected)
        for env, episode in rejected:
            item = pool._keys[(env, episode)]
            reasons = [r for _, r in item["reasons"]]
            self.assertTrue(reasons)
            self.assertEqual(item["doses"], {})
            if "no_clear_site_in_route_band" in reasons:          # placement failed: every candidate's reasons follow
                self.assertIsNone(item["placement"])
                self.assertEqual(reasons[0], "no_clear_site_in_route_band")
                self.assertGreater(len(reasons), 1)
        self.assertIn("no_clear_site_in_route_band", pool.reason_counts())

    def test_scatter_site_is_redrawn_inside_the_band_up_to_the_named_maximum(self):
        self.assertEqual(P.SCATTER_MAX_SITE_CANDIDATES, 16)                 # G20d (owner decision 8B): was 8
        for vessel in (F.BB, F.WAMV):
            pool = F.reference_pool("scatter", vessel)
            redrawn = 0
            for env, episode in pool.offered:
                doc = pool.placement(env, episode)
                if doc is None:
                    continue
                site = doc["site"]
                self.assertEqual(site["rule"], "route_fraction_redraw")
                self.assertTrue(0.4 <= site["fraction"] <= 0.6)
                self.assertLess(site["candidate_index"], P.SCATTER_MAX_SITE_CANDIDATES)
                self.assertEqual(len(site["rejected_candidates"]), site["candidate_index"])
                for fraction, reasons in site["rejected_candidates"]:
                    self.assertTrue(0.4 <= fraction <= 0.6 and reasons)
                redrawn += site["candidate_index"] > 0
                context, _, _ = F.inputs("scatter", vessel)(env, episode)
                d1 = C.start_gap_m("scatter", context.envelope.t_avoid_s, vessel)
                self.assertEqual(P.site_reasons(context, doc, d1), [])      # the chosen site passes every rule
            self.assertGreater(redrawn, 0)

    def test_path_following_site_clears_the_trigger_distance_from_the_start(self):
        for vessel in (F.BB, F.WAMV):
            pool = F.reference_pool("path_following", vessel)
            env0 = F.envelope(vessel)
            scale = F.plans.vessel_record(vessel).geometry.length_scale
            # start gap (A: d_1 = 4 t_avoid; M: the largest calibration-sweep gap) + reach + radius + 1.0 x scale
            minimum = (C.start_gap_m("path_following", env0.t_avoid_s, vessel) + env0.footprint.reach_m
                       + C.RADIUS_LENGTH_FACTOR * scale + P.PF_START_MARGIN_LENGTHS * scale)
            moved = 0
            for env, episode in pool.offered:
                doc = pool.placement(env, episode)
                if doc is None:
                    continue
                site = doc["site"]
                self.assertAlmostEqual(site["min_route_distance_m"], minimum, places=2)
                self.assertGreaterEqual(site["route_distance_m"], site["min_route_distance_m"] - 1e-9)
                self.assertEqual(site["start_rule"]["margin_lengths"], P.PF_START_MARGIN_LENGTHS)
                moved += site["moved_forward"]
                if site["moved_forward"]:
                    self.assertEqual(site["leg_index"], 0)
                    self.assertLess(site["t_site_s"], site["leg_end_s"])
                if pool.is_admitted(env, episode):
                    context, _, _ = F.inputs("path_following", vessel)(env, episode)
                    cyl = doc["cylinder"]
                    gap = P.gap_at(context.initial_pose, cyl["x_m"], cyl["y_m"], cyl["radius_m"],
                                   context.envelope.footprint)
                    self.assertGreater(gap, C.start_gap_m("path_following", context.envelope.t_avoid_s, vessel))
            if vessel == F.BB:
                self.assertGreater(moved, 0)                         # the first-leg keys of phase 1 move forward


class Pools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = F.small_pool("path_following", F.BB, num_envs=4, episodes=2)

    def test_counts_and_lookups(self):
        pool = self.pool
        self.assertEqual(pool.grid, {"num_envs": 4, "episodes": 2})
        self.assertEqual(len(pool.per_dose_counts), C.DOSE_COUNT)
        for row in pool.per_dose_counts:
            self.assertEqual(row, (pool.keys_offered, pool.keys_admitted, pool.keys_not_admitted))
        env, episode = F.first_admitted(pool)
        for dose in range(C.DOSE_COUNT):
            a = pool.assignment(F.key("path_following", env, episode), dose)
            self.assertEqual(a.exposure, "admitted")
            self.assertEqual(a.dose, dose)
            self.assertEqual(a.position_digest, pool.assignment(F.key("path_following", env, episode), 0)
                             .position_digest)
            self.assertEqual(a.appears, dose > 0)
        beyond = pool.assignment(F.key("path_following", 0, 2), 3)
        self.assertTrue(beyond.beyond_offered_grid)
        self.assertEqual(beyond.reason_strings(), ["beyond_offered_grid"])
        self.assertEqual(beyond.dose, 0)
        self.assertEqual(beyond.position_digest, pool.assignment(F.key("path_following", 0, 2), 1).position_digest)
        for bad in (F.key("path_following", 4, 0), F.key("path_following", 0, 0, seed=7),
                    F.dynamic.EpisodeKey(42, 0, 0, "d7_path_following")):
            with self.assertRaises(KeyError):
                pool.assignment(bad, 1)

    def test_not_admitted_lookup_is_dose_independent(self):
        pool = F.reference_pool("scatter", F.BB)
        env, episode = F.first_not_admitted(pool)
        items = [pool.assignment(F.key("scatter", env, episode), dose) for dose in range(C.DOSE_COUNT)]
        self.assertEqual({a.position_digest for a in items}, {items[0].position_digest})
        self.assertTrue(all(a.dose == 0 and a.exposure == "not_admitted" for a in items))
        self.assertEqual(items[0].geometry_id, pool.geometry(env, episode))
        self.assertTrue(set(items[0].reason_strings()) <= set(pool.reason_counts()))
        self.assertNotIn("joint_ladder_member_rejected", items[0].reason_strings())

    def test_file_round_trip_and_tamper(self):
        pool = self.pool
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / POOL.pool_file_name("path_following", F.BB)
            sha = POOL.write_pool_file(pool, path, provenance={"test": True})
            sha2 = POOL.write_pool_file(pool, Path(folder) / "again.json.xz", provenance={"test": True})
            self.assertEqual(sha, sha2)                                   # deterministic bytes
            loaded = POOL.read_pool_file(path, expected_sha256=sha, task="path_following", vessel_id=F.BB,
                                         seed=F.SEED, tier=F.TIER)
            self.assertEqual(loaded.pool_digest, pool.pool_digest)
            env, episode = F.first_admitted(pool)
            k = F.key("path_following", env, episode)
            self.assertEqual(loaded.assignment(k, 2), pool.assignment(k, 2))
            with self.assertRaises(ValueError):
                POOL.read_pool_file(path, expected_sha256="0" * 64, task="path_following", vessel_id=F.BB,
                                    seed=F.SEED, tier=F.TIER)
            with self.assertRaises(ValueError):
                POOL.read_pool_file(path, expected_sha256=sha, task="path_following", vessel_id=F.BB,
                                    seed=43, tier=F.TIER)
            doc = POOL.pool_document(pool)
            bad = copy.deepcopy(doc)
            admitted = next(item for item in bad["keys"] if item["doses"])
            admitted["doses"]["2"]["candidate"]["trigger_gap_m"] += 1e-6
            with self.assertRaises(ValueError):
                POOL.pool_from_document(bad)
            bad = copy.deepcopy(doc)
            bad["per_dose_counts"][0][1] += 1
            with self.assertRaises(ValueError):
                POOL.pool_from_document(bad)

    def test_legacy_v24_pools_keep_their_pins_and_follow_the_ladder_switch(self):
        # G20b: the v2.4 G3 files (ladder A, legacy names) stay pinned and untouched; under any other ladder the
        # loader refuses them (suite identity), so the bundle carries seed-qualified pools of the active ladder
        for task, family in (("scatter", "hazard_nav"), ("path_following", "path_following")):
            manifest, _ = POOL.read_manifest(POOL.family_pool_dir(family))
            for vessel in (F.BB, F.WAMV):
                if C.WITNESSED_DOSES != (1, 2, 3, 4):        # option E: the four-dose files do not even parse
                    with self.assertRaisesRegex(ValueError, "witnessed doses"):
                        F.legacy_pool(task, vessel)
                    continue
                pool, sha = F.legacy_pool(task, vessel)
                entry = manifest["pools"][POOL.pool_file_name(task, vessel)]
                self.assertEqual(entry["sha256"], sha)
                self.assertEqual(entry["pool_digest"], pool.pool_digest)
                self.assertEqual(pool.grid, {"num_envs": 64, "episodes": 2})
                self.assertEqual((pool.seed, pool.tier), (42, 2))
                self.assertEqual(pool.suite_identity["version"], "g3_appearing_gap_v2")
                identity = F.admission.admission_suite(vessel).identity()
                folder = POOL.family_pool_dir(family)
                if C.LADDER_OPTION == "A":
                    POOL.read_pool_file(folder / entry["file"], expected_sha256=sha, task=task, vessel_id=vessel,
                                        seed=42, tier=2, suite_identity=identity)
                else:
                    with self.assertRaisesRegex(ValueError, "different suite configuration"):
                        POOL.read_pool_file(folder / entry["file"], expected_sha256=sha, task=task,
                                            vessel_id=vessel, seed=42, tier=2, suite_identity=identity)

    def test_legacy_v24_placements_still_derive_from_the_code(self):
        # drift guard: a few keys of every v2.4 G3 pool are re-admitted now; the placement (ladder-independent apart
        # from its suite_version field: the start rule uses d_1, k = 4 under every option) must equal the pinned
        # one; under ladder A the whole document, the candidates and witnesses too
        for task, vessel in (("scatter", F.BB), ("scatter", F.WAMV), ("path_following", F.BB),
                             ("path_following", F.WAMV)):
            placements = F.legacy_placements(task, vessel)
            cfg = F.admission.admission_suite(vessel)
            for env, episode in ((0, 0), (5, 1), (37, 0)):
                context, scale, rng = F.inputs(task, vessel)(env, episode)
                record = F.admission.admit_job((context, F.key(task, env, episode), cfg, scale, rng))
                if C.LADDER_KIND == "metres":
                    # the metre ladder moves the start gap (calibration tasks: the largest sweep gap), so the v2.4
                    # site differs by design; the file still holds the same key's v2.4 document (A start rule)
                    stored_doc = placements[(env, episode)]
                    if stored_doc is not None and task == "path_following":
                        self.assertAlmostEqual(stored_doc["site"]["start_rule"]["d1_m"],
                                               4.0 * F.envelope(vessel).t_avoid_s, places=9)
                    continue
                if C.LADDER_OPTION != "A":
                    # the placement document names the suite version (= the ladder option); everything else is
                    # ladder-independent and must recompute
                    self.assertEqual(dict(placements[(env, episode)] or {}, suite_version=None),
                                     dict(record["placement"] or {}, suite_version=None), (task, vessel, env, episode))
                    continue
                pool, _ = F.legacy_pool(task, vessel)
                stored = pool._keys[(env, episode)]
                self.assertEqual(record["placement"], stored["placement"], (task, vessel, env, episode))
                self.assertEqual(tuple(tuple(r) for r in record["reasons"]), stored["reasons"])
                self.assertEqual(sorted(record["doses"]), sorted(stored["doses"]))
                for dose, entry in record["doses"].items():
                    self.assertEqual(entry["candidate"], stored["doses"][dose]["candidate"])
                    self.assertEqual(POOL.witness_digest(entry["witness"]),
                                     POOL.witness_digest(stored["doses"][dose]["witness"]))


class Stamps(unittest.TestCase):
    def test_round_trip_tamper_and_invariants(self):
        pool = F.reference_pool("path_following", F.BB)
        env, episode = F.first_admitted(pool)
        bridge = RT.AppearingBridge("path_following", 1, PHYSICS_DT, 3.0, F.envelope(F.BB).footprint,
                                    F.envelope(F.BB).t_avoid_s)
        bridge.reset_rows((env,), {env: pool.assignment(F.key("path_following", env, episode), 2)})
        bridge.step({env: 0}, {env: F.dynamic.Pose(0.0, 0.0, 0.0, 0.0)})
        envelope = bridge.finish(env, "timeout")
        self.assertEqual(ST.read_appearing_stamp(envelope), envelope)
        self.assertEqual(ST.exposure_of(envelope), ("admitted", ()))
        runtime = envelope["payload"]["runtime"]
        self.assertEqual((runtime["assigned"], runtime["itt_count"], runtime["exposure_realized"]), (True, 1, False))
        self.assertEqual(runtime["non_exposure_reason"], "no_trigger_check_before_end")
        self.assertIs(bridge.finish(env, "timeout"), envelope)                    # idempotent
        tampered = copy.deepcopy(envelope)
        tampered["payload"]["assignment"]["dose"] = 3
        with self.assertRaises(ValueError):
            ST.read_appearing_stamp(tampered)
        spool = F.reference_pool("scatter", F.BB)
        row, episode_na = F.first_not_admitted(spool)
        sbridge = RT.AppearingBridge("scatter", 15, PHYSICS_DT, 3.0, F.envelope(F.BB).footprint,
                                     F.envelope(F.BB).t_avoid_s)
        sbridge.reset_rows((row,), {row: spool.assignment(F.key("scatter", row, episode_na), 4)})
        sbridge.step({row: 0}, {row: F.dynamic.Pose(0.0, 0.0, 0.0, 0.0)})
        na = sbridge.finish(row, "success")
        kind, reasons = ST.exposure_of(na)
        self.assertEqual(kind, "not_admitted")
        self.assertTrue(reasons and set(reasons) <= set(spool.reason_counts()))
        a = na["payload"]["assignment"]
        self.assertEqual((a["dose"], a["requested_dose"], a["cylinder"]), (0, 4, None))
        self.assertTrue({"key", "task", "dose", "candidate_digest"} <= set(a))       # D20 (b) form
        r = na["payload"]["runtime"]
        self.assertEqual((r["assigned"], r["itt_count"], r["exposure_realized"]), (False, 0, False))   # D27
        with self.assertRaises(ValueError):
            ST.make_appearing_stamp("not_admitted", a, r, reasons=())


class Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = F.reference_pool("path_following", F.WAMV)

    def _bridge(self, vessel=F.WAMV):
        env = F.envelope(vessel)
        return RT.AppearingBridge("path_following", 1, PHYSICS_DT, env.top_speed_mps, env.footprint, env.t_avoid_s)

    def test_every_dose_triggers_at_the_declared_gap_within_one_control_step(self):
        env_row, episode = F.first_admitted(self.pool)
        envelope = F.envelope(F.WAMV)
        speed = 1.0
        step_travel = speed * PHYSICS_DT * DECIMATION
        for dose in C.WITNESSED_DOSES:
            a = self.pool.assignment(F.key("path_following", env_row, episode), dose)
            x, y, r = a.cylinder
            heading = 0.3
            start = (x - 60.0 * math.cos(heading), y - 60.0 * math.sin(heading))
            bridge = self._bridge()
            bridge.reset_rows((env_row,), {env_row: a})
            trace = F.straight_run(bridge, env_row, start, heading, speed, PHYSICS_DT, DECIMATION, 4000)
            first = next(i for i, (_, _, on) in enumerate(trace) if on)
            gap = P.gap_at(trace[first][1], x, y, r, envelope.footprint)
            previous = P.gap_at(trace[first - 1][1], x, y, r, envelope.footprint)
            d_k = C.trigger_gap_m(dose, envelope.t_avoid_s)
            self.assertLessEqual(gap, d_k)
            self.assertGreater(previous, d_k)
            self.assertGreater(gap, d_k - step_travel - 1e-9)
            trig = bridge.rows[env_row]["trigger"]
            self.assertAlmostEqual(trig["gap_m"], gap, places=12)
            self.assertAlmostEqual(trig["bearing_rad"], 0.0, places=9)        # dead ahead on the line
            self.assertEqual(trig["tick"], trace[first][0])
            # absent before (no sample physical/visible), physical + visible after and forever
            samples_before = trace[:first]
            self.assertTrue(all(not on for _, _, on in samples_before))
            self.assertTrue(all(on for _, _, on in trace[first:]))
            sample = bridge.step({env_row: trace[-1][0]}, {env_row: trace[-1][1]})[env_row]
            self.assertEqual(len(sample), 1)
            self.assertTrue(sample[0].physical and sample[0].visible)
            self.assertEqual(sample[0].slot, 1)
            stamp = bridge.finish(env_row, "collision")
            rt = stamp["payload"]["runtime"]
            self.assertTrue(rt["exposure_realized"])
            self.assertTrue(rt["swept_contact_latched"])                       # drove straight through it
            self.assertAlmostEqual(rt["trigger"]["gap_m"], gap, places=12)

    def test_trigger_is_checked_only_at_the_control_rate(self):
        env_row, episode = F.first_admitted(self.pool)
        a = self.pool.assignment(F.key("path_following", env_row, episode), 4)
        x, y, r = a.cylinder
        bridge = self._bridge()
        bridge.reset_rows((env_row,), {env_row: a})
        inside = F.dynamic.Pose(x - r - 3.0, y, 0.0, 0.0)     # well within d_4
        bridge.step({env_row: 0}, {env_row: inside})
        bridge.step({env_row: 1}, {env_row: inside})          # substep: never a trigger check
        self.assertFalse(bridge.triggered(env_row))
        bridge.step({env_row: 2}, {env_row: inside}, check_trigger=True)
        self.assertTrue(bridge.triggered(env_row))

    def test_never_reached_and_zero_dose_rows(self):
        env_row, episode = F.first_admitted(self.pool)
        far = F.dynamic.Pose(-1.0e4, -1.0e4, 0.0, 0.0)
        for dose in (0, 3):
            bridge = self._bridge()
            bridge.reset_rows((env_row,), {env_row: self.pool.assignment(F.key("path_following", env_row, episode),
                                                                         dose)})
            out = bridge.step({env_row: 0}, {env_row: far})
            for tick in range(1, 50):
                out = bridge.step({env_row: tick}, {env_row: far}, check_trigger=(tick % 2 == 0))
            self.assertEqual(len(out[env_row]), 0 if dose == 0 else 1)
            if dose:
                self.assertFalse(out[env_row][0].physical or out[env_row][0].visible)
            rt = bridge.finish(env_row, "timeout")["payload"]["runtime"]
            self.assertFalse(rt["exposure_realized"])
            self.assertEqual((rt["assigned"], rt["itt_count"]), (True, 1))
            self.assertEqual(rt["non_exposure_reason"], "zero_dose" if dose == 0 else "trigger_gap_never_reached")

    def test_reset_requires_finish_and_rejects_foreign_rows(self):
        env_row, episode = F.first_admitted(self.pool)
        a = self.pool.assignment(F.key("path_following", env_row, episode), 1)
        bridge = self._bridge()
        bridge.reset_rows((env_row,), {env_row: a})
        with self.assertRaises(ValueError):
            bridge.reset_rows((env_row,), {env_row: a})
        with self.assertRaises(ValueError):
            bridge.reset_rows((env_row + 1,), {env_row + 1: a})                 # key of another env


if __name__ == "__main__":
    unittest.main(verbosity=2)
