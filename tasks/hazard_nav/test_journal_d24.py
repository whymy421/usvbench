"""F1 round T: CONTEXT_R4 D24 (episodes beyond the offered grid run at D0, never counted), CPU only.

* pool: the offered grid {num_envs N, episodes E} is read from the offered keys
  (one seed, one hazard group, full rectangle; anything else refused); a key
  with the pool's seed and hazard group, env < N and episode >= E is a
  not-admitted key with the single reason "beyond_offered_grid" at every dose
  (never a KeyError, never a pool key: counts unchanged); env >= N, another
  seed or another hazard group stay KeyError / preflight ValueError; a pool
  file whose declared grid differs from its offered keys fails closed;
* env: a row whose next task episode index equals E (and E+1, E+2, ...) resets
  and runs a FULL episode at EVERY dynamic level D0..D4; the record stamps
  exposure "not_admitted" with the single reason "beyond_offered_grid" (D20
  wire form: dose-0 assignment with key, task, dose 0, candidate_digest None);
  its layout is the E5 layout of that key; F0 pairing.d7_primitive_hash
  accepts the stamp and gives one value at every level; no intruder is ever
  published; the moving basin keeps its natives frozen at time zero (E4 D0);
* header: dynamic_hazard_header["grid"] == {"num_envs": N, "episodes": E}
  and the per-dose counts are the pool's (beyond-grid keys never counted).

Run: python -B tasks/hazard_nav/test_journal_d24.py
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import journal_test_support as S  # noqa: E402

RESTORE = S.install_isaaclab_stubs()
FAM = S.load_family()
J, D, P = FAM.journal, FAM.dynamic, FAM.pairing
LAYOUTS = __import__(S.ROOT + "._shared.journal_layouts_v2", fromlist=["build_layout"])
SCRATCH = S.REPO.parent / "test_scratch_f1_d24"
VESSEL = "blueboat_v2"
NUM_ENVS, EPISODES = 2, (0,)                    # offered grid 2 x 1: episode 1 is one past the grid
BEYOND = [[0, "beyond_offered_grid", 0]]


def grid_pool(task="reach", num_envs=NUM_ENVS, episodes=EPISODES, seed=S.SEED):
    return S.admitted_pool(FAM, task, VESSEL, num_envs=num_envs, episodes=episodes, seed=seed)


def key(env, episode, task="reach", seed=S.SEED):
    return D.EpisodeKey(seed, env, episode, "d7_" + task)


def finish_in(env, steps):
    env._journal_twin.deadline[:] = steps * FAM.plans.CONTROL_DT_S


def check_beyond_record(case, rec, env_index, episode, dose, task="reach"):
    case.assertIsNotNone(rec)
    case.assertEqual(rec["scenario_episode"], episode)
    case.assertEqual(rec["episode_ordinal"], episode)
    case.assertEqual(rec["termination_name"], "timeout")
    case.assertEqual(rec["d7_exposure"], "not_admitted")
    payload = D.read_stamp(rec["dynamic_hazards"])["payload"]       # E4 envelope + digest
    case.assertEqual(payload["exposure"], "not_admitted")
    case.assertEqual(payload["not_admitted"]["reasons"], BEYOND)
    case.assertEqual(payload["not_admitted"]["primary_reason"], "beyond_offered_grid")
    case.assertEqual(payload["not_admitted"]["rule"], J.D24_RULE)
    case.assertEqual(payload["assignment"]["not_admitted_reasons"], BEYOND)
    assignment = payload["assignment"]
    for field in ("key", "task", "dose", "candidate_digest"):
        case.assertIn(field, assignment)
    case.assertEqual(assignment["dose"], 0)
    case.assertEqual(assignment["intruders"], [])
    case.assertIsNone(assignment["candidate_digest"])
    case.assertEqual(assignment["task"], task)
    case.assertEqual(assignment["key"], {"eval_seed": S.SEED, "env": env_index, "episode": episode,
                                         "hazard_group": "d7_" + task})
    case.assertEqual(payload["runtime"]["exposure"], "not_admitted")
    case.assertFalse(payload["runtime"]["assigned"])
    case.assertEqual(payload["runtime"]["itt_count"], 0)
    case.assertEqual(payload["runtime"]["suite_dose"], dose)
    case.assertEqual(P.d7_primitive_hash(rec["dynamic_hazards"]), rec["scenario_hashes"]["d7_assignment"])
    return payload


class PoolGrid(unittest.TestCase):
    def test_grid_is_the_offered_rectangle(self):
        pool = J.family_pool(grid_pool())
        self.assertEqual(pool.grid, {"eval_seed": S.SEED, "hazard_group": "d7_reach", "num_envs": NUM_ENVS,
                                     "episodes": len(EPISODES)})
        self.assertEqual(J.pool_header_extra(pool)["grid"], {"num_envs": NUM_ENVS, "episodes": len(EPISODES)})

    def test_beyond_grid_keys_are_not_admitted_with_one_reason_at_every_dose(self):
        pool = J.family_pool(grid_pool())
        counts = (pool.per_dose_counts, pool.keys_offered, pool.keys_admitted, pool.keys_not_admitted,
                  dict(pool.not_admitted))
        for env in range(NUM_ENVS):
            for episode in (len(EPISODES), len(EPISODES) + 1, len(EPISODES) + 57):
                k = key(env, episode)
                self.assertTrue(pool.beyond_offered_grid(k))
                self.assertFalse(pool.is_admitted(k))
                for dose in range(5):
                    got = pool.assignment(k, dose)
                    self.assertIsInstance(got, J.NotAdmittedAssignment)
                    self.assertEqual((got.key, got.dose), (k, dose))
                    self.assertEqual(got.reasons, ((0, "beyond_offered_grid", 0),))
                    self.assertTrue(got.record.beyond_offered_grid)
                    self.assertIsNone(got.record.geometry_id)
                    self.assertEqual(got.record.primary_reason(), "beyond_offered_grid")
                    self.assertEqual(got.pool_digest, pool.pool_digest)
        for k in pool.offered:                     # offered keys keep their D16 / admitted entries
            self.assertFalse(pool.beyond_offered_grid(k))
            got = pool.assignment(k, 2)
            if isinstance(got, J.NotAdmittedAssignment):
                self.assertFalse(got.record.beyond_offered_grid)
        self.assertEqual((pool.per_dose_counts, pool.keys_offered, pool.keys_admitted, pool.keys_not_admitted,
                          dict(pool.not_admitted)), counts)          # never a pool key

    def test_outside_env_range_or_other_seed_or_group_stay_key_errors(self):
        pool = J.family_pool(grid_pool())
        for bad in (key(NUM_ENVS, 0), key(NUM_ENVS, len(EPISODES)), key(NUM_ENVS + 5, 9),
                    key(0, len(EPISODES), seed=7), key(0, 0, seed=7), key(0, len(EPISODES), task="ring"),
                    key(0, 0, task="ring")):
            self.assertFalse(pool.beyond_offered_grid(bad))
            for dose in (0, 3):
                with self.assertRaises(KeyError):
                    pool.assignment(bad, dose)
        scenario = J.build_scenario("reach", VESSEL, 2, S.SEED, NUM_ENVS, 0)          # env out of range
        with self.assertRaisesRegex(ValueError, "outside this pool.*env range"):
            J.check_assignment_geometry(pool, scenario, 0)
        other_seed = J.build_scenario("reach", VESSEL, 2, 7, 0, len(EPISODES))
        with self.assertRaisesRegex(ValueError, "outside this pool"):
            J.check_assignment_geometry(pool, other_seed, 0)

    def test_pool_without_a_rectangular_grid_is_refused(self):
        holed = grid_pool(num_envs=1, episodes=(0, 2))
        with self.assertRaisesRegex(ValueError, "full grid"):
            J.family_pool(holed)
        with self.assertRaises(ValueError):
            J.offered_grid([key(0, 0), key(0, 0, seed=7)])
        with self.assertRaises(ValueError):
            J.offered_grid([])

    def test_pool_file_grid_must_equal_the_offered_grid(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, SCRATCH, True)
        admission = grid_pool()
        suite = D.DynamicSuiteConfig(enabled=True, vessel_id=VESSEL)
        offered = [key(env, episode) for env in range(NUM_ENVS) for episode in EPISODES]
        good = {"num_envs": NUM_ENVS, "episodes": list(EPISODES), "seed": S.SEED, "tier": 2}
        doc = J.pool_document(admission, task="reach", vessel=VESSEL, tier=2, seed=S.SEED, suite=suite,
                              offered=offered, grid=good, provenance={})
        path = SCRATCH / J.pool_file_name("reach", VESSEL)
        digest = J.write_pool_file(doc, path)
        pool = J.load_d7_pool(path, task="reach", vessel=VESSEL, tier=2, seed=S.SEED, expected_sha256=digest)
        self.assertEqual((pool.grid["num_envs"], pool.grid["episodes"]), (NUM_ENVS, len(EPISODES)))
        for mutate in (lambda d: d["grid"].__setitem__("num_envs", NUM_ENVS + 1),
                       lambda d: d["grid"].__setitem__("episodes", list(EPISODES) + [len(EPISODES)]),
                       lambda d: d["grid"].__setitem__("seed", 7),
                       lambda d: d.pop("grid")):
            bad = json.loads(json.dumps(doc))
            mutate(bad)
            tampered = SCRATCH / "tampered.json.gz"
            with self.assertRaisesRegex(ValueError, "grid"):
                J.load_d7_pool(tampered, task="reach", vessel=VESSEL, tier=2, seed=S.SEED,
                               expected_sha256=J.write_pool_file(bad, tampered))


class BeyondGridEpisodes(unittest.TestCase):
    """Reach rows with the offered grid 2 x 1: episode 1 (= E) is one past the grid."""

    @classmethod
    def setUpClass(cls):
        cls.records = {}
        for dose in range(5):
            cfg = S.journal_cfg(FAM, "reach", VESSEL, tokens=("dynamic=D%d" % dose,), num_envs=NUM_ENVS,
                                pool=grid_pool())
            env = S.make_env(FAM, cfg)
            env.reset()
            finish_in(env, 1)
            env.step(S.zero_action(env, (0.4, 0.0)))           # episode 0 (offered) completes
            first = [env.journal_completed_records[i] for i in range(NUM_ENVS)]
            beyond_rows = [env._d7_bridge.is_not_admitted(i) for i in range(NUM_ENVS)]
            scenarios = list(env._journal_scenarios)
            finish_in(env, 2)
            published = []
            for _ in range(2):
                published.append(bool(env._d7_physical.any() or env._d7_visible.any()))
                env.step(S.zero_action(env, (0.4, 0.05)))
            cls.records[dose] = {"env": env, "first": first, "beyond_rows": beyond_rows, "scenarios": scenarios,
                                 "published": published,
                                 "beyond": [env.journal_completed_records[i] for i in range(NUM_ENVS)]}

    def test_full_episode_at_every_dynamic_level_stamps_beyond_offered_grid(self):
        for dose, run in self.records.items():
            env = run["env"]
            for i in range(NUM_ENVS):
                self.assertEqual(run["first"][i]["scenario_episode"], 0)
                self.assertTrue(run["beyond_rows"][i], (dose, i))
                payload = check_beyond_record(self, run["beyond"][i], i, len(EPISODES), dose)
                self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 2 * env.cfg.decimation)
                self.assertEqual(payload["assignment"]["native_objects_frozen"], [])

    def test_d7_primitive_hash_is_the_same_at_every_level(self):
        per_env = {i: set() for i in range(NUM_ENVS)}
        for run in self.records.values():
            for i in range(NUM_ENVS):
                envelope = run["beyond"][i]["dynamic_hazards"]
                per_env[i].add(P.d7_primitive_hash(envelope))
        for i in range(NUM_ENVS):
            self.assertEqual(len(per_env[i]), 1, i)
        self.assertNotEqual(per_env[0], per_env[1])               # key-dependent, not one shared value

    def test_layout_is_the_e5_layout_of_the_key(self):
        for run in self.records.values():
            for i in range(NUM_ENVS):
                sc = run["scenarios"][i]
                self.assertEqual((sc.env, sc.episode), (i, len(EPISODES)))
                lay = LAYOUTS.build_layout("reach", 2, VESSEL,
                                           J.scenario_generator(S.SEED, i, len(EPISODES), J.LAYOUT_GROUP + "reach"),
                                           max_attempts=J.LAYOUT_MAX_ATTEMPTS)
                self.assertEqual(sc.layout.geometry_hash, lay.geometry_hash)
                rebuilt = J.build_scenario("reach", VESSEL, 2, S.SEED, i, len(EPISODES))
                self.assertEqual(sc.geometry_id, rebuilt.geometry_id)
                rec = run["beyond"][i]
                self.assertEqual(rec["scenario"]["layout_geometry_hash"], lay.geometry_hash)
                self.assertEqual(rec["dynamic_hazards"]["payload"]["assignment"]["geometry_id"], sc.geometry_id)
                self.assertEqual(rec["scenario_hashes"]["layout"],
                                 J.family_hashes(rebuilt, run["env"]._journal_eq_z)["layout"])

    def test_no_intruder_is_published_for_beyond_grid_rows(self):
        for dose, run in self.records.items():
            self.assertEqual(run["published"], [False, False], dose)

    def test_header_grid_and_counts_are_the_pool_grid(self):
        pool = J.family_pool(grid_pool())
        for run in self.records.values():
            header = run["env"].journal_certificate_header(treatment="dynamic")["dynamic_hazards_header"]
            self.assertEqual(header["grid"], {"num_envs": NUM_ENVS, "episodes": len(EPISODES)})
            self.assertEqual(header["per_dose_counts"], [list(row) for row in pool.per_dose_counts])
            self.assertEqual(header["keys_offered"], NUM_ENVS * len(EPISODES))
            self.assertEqual(header["keys_not_admitted"], pool.keys_not_admitted)

    def test_pairs_with_the_same_key_in_another_suite(self):
        cfg = S.journal_cfg(FAM, "reach", VESSEL, tokens=("waves=D2", "dynamic=D0"), num_envs=NUM_ENVS,
                            pool=grid_pool())
        env = S.make_env(FAM, cfg)
        env.reset()
        finish_in(env, 1)
        env.step(S.zero_action(env, (0.4, 0.0)))
        finish_in(env, 2)
        for _ in range(2):
            env.step(S.zero_action(env, (0.4, 0.05)))
        for i in range(NUM_ENVS):
            other = env.journal_completed_records[i]
            check_beyond_record(self, other, i, len(EPISODES), 0)
            for dose in range(5):
                summary = P.validate_pair(self.records[dose]["beyond"][i], other, "waves")
                self.assertNotIn("d7_assignment", summary["changed_primitives"])
                self.assertIn("layout", summary["unchanged_primitives"])
            with self.assertRaises(P.PairingError):            # unexposed for the dynamic treatment
                P.validate_pair(self.records[0]["beyond"][i], self.records[3]["beyond"][i], "dynamic")

    def test_trailing_episodes_keep_running(self):
        """Unbounded trailing episodes: E, E+1, E+2 ... each reset and complete at D4."""
        cfg = S.journal_cfg(FAM, "reach", VESSEL, tokens=("dynamic=D4",), num_envs=NUM_ENVS, pool=grid_pool())
        env = S.make_env(FAM, cfg)
        env.reset()
        finish_in(env, 1)
        env.step(S.zero_action(env))
        for episode in range(len(EPISODES), len(EPISODES) + 4):
            finish_in(env, 1)
            env.step(S.zero_action(env, (0.3, 0.0)))
            for i in range(NUM_ENVS):
                rec = env.journal_completed_records[i]
                payload = check_beyond_record(self, rec, i, episode, 4)
                self.assertEqual(payload["runtime"]["bridge"]["last_tick"], env.cfg.decimation)


class ObstacleTaskBeyondGrid(unittest.TestCase):
    def test_ring_row_beyond_the_grid_runs_with_static_geometry_and_no_intruder(self):
        cfg = S.journal_cfg(FAM, "ring", VESSEL, tokens=("dynamic=D3",), num_envs=NUM_ENVS,
                            pool=grid_pool("ring"))
        env = S.make_env(FAM, cfg)
        env.reset()
        finish_in(env, 1)
        env.step(S.zero_action(env))
        slot = FAM.plans.STATIC_CAPACITY["ring"]
        finish_in(env, 2)
        for _ in range(2):
            for i in range(NUM_ENVS):
                self.assertTrue(env._d7_bridge.is_not_admitted(i))
                self.assertFalse(bool(env._d7_physical[i, slot]))
                self.assertFalse(bool(env._d7_visible[i].any()))
                self.assertTrue(bool(env.obstacle_active[i].any()))          # the E5 ring is there
            env.step(S.zero_action(env, (0.4, 0.0)))
        for i in range(NUM_ENVS):
            check_beyond_record(self, env.journal_completed_records[i], i, len(EPISODES), 3, task="ring")
            sc = env.journal_completed_records[i]["scenario"]
            self.assertEqual(sc["layout_geometry_hash"],
                             J.build_scenario("ring", VESSEL, 2, S.SEED, i, len(EPISODES)).layout.geometry_hash)


class MovingBasinBeyondGrid(unittest.TestCase):
    def test_basin_row_beyond_the_grid_keeps_natives_frozen_at_time_zero(self):
        pool = S.admitted_pool(FAM, "moving_basin", VESSEL, num_envs=1, episodes=(0,))
        cfg = S.journal_cfg(FAM, "moving_basin", VESSEL, tokens=("dynamic=D4",), num_envs=1, pool=pool)
        env = S.make_env(FAM, cfg)

        def forbidden(*args, **kwargs):
            raise AssertionError("legacy mover called in the journal basin")
        env._update_moving_obstacles = forbidden
        env._maybe_appear_obstacles = forbidden
        env.reset()
        finish_in(env, 1)
        env.step(S.zero_action(env, (0.5, 0.0)))                     # episode 0 (offered) completes
        self.assertEqual(env.journal_completed_records[0]["scenario_episode"], 0)
        self.assertTrue(env._d7_bridge.is_not_admitted(0))
        sc = env._journal_scenarios[0]
        self.assertEqual(sc.episode, 1)
        native = list(sc.native_slots)
        self.assertTrue(native)
        self.assertTrue(bool(env._d7_physical[0, native].all()))
        self.assertFalse(bool(env.obstacle_active[0, native].any()))
        start = env.obstacle_centers[0, native].clone()
        local = start - env.scene.env_origins[0, :2]
        for row, slot in enumerate(native):
            self.assertAlmostEqual(float(local[row, 0]), float(sc.centers[slot, 0]), places=4)
            self.assertAlmostEqual(float(local[row, 1]), float(sc.centers[slot, 1]), places=4)
        finish_in(env, 4)
        for _ in range(3):
            env.step(S.zero_action(env, (0.5, 0.0)))
            self.assertTrue(torch.equal(env.obstacle_centers[0, native], start))     # frozen (E4 D0)
            self.assertTrue(bool(env._d7_physical[0, native].all()))
        env.step(S.zero_action(env, (0.5, 0.0)))
        rec = env.journal_completed_records[0]
        payload = check_beyond_record(self, rec, 0, 1, 4, task="moving_basin")
        frozen = payload["assignment"]["native_objects_frozen"]
        self.assertEqual(len(frozen), len(native))
        self.assertTrue(all(o["speed_bound_mps"] == 0.0 for o in frozen))
        self.assertEqual(sorted(o["slot"] for o in frozen), sorted(native))
        for obj in frozen:
            self.assertAlmostEqual(obj["x0"], float(sc.centers[obj["slot"], 0]), places=6)
            self.assertAlmostEqual(obj["y0"], float(sc.centers[obj["slot"], 1]), places=6)


class PreflightErrors(unittest.TestCase):
    def test_env_outside_the_pool_env_range_fails_closed(self):
        cfg = S.journal_cfg(FAM, "reach", VESSEL, tokens=("dynamic=D2",), num_envs=NUM_ENVS + 1, pool=grid_pool())
        env = S.make_env(FAM, cfg)
        with self.assertRaisesRegex(ValueError, "outside this pool"):
            env.reset()

    def test_pool_of_another_seed_fails_closed(self):
        cfg = S.journal_cfg(FAM, "reach", VESSEL, tokens=("dynamic=D2",), num_envs=NUM_ENVS,
                            pool=grid_pool(seed=7))
        env = S.make_env(FAM, cfg)
        with self.assertRaisesRegex(ValueError, "outside this pool"):
            env.reset()

    def test_pinned_pool_header_grid(self):
        evidence = S.REPO.parent / "evidence" / "d7_pools"
        if not evidence.is_dir():
            self.skipTest("evidence/d7_pools not beside this repository copy")
        cfg = S.journal_cfg(FAM, "reach", VESSEL, tokens=("dynamic=D1",), num_envs=2)
        cfg.journal_d7_pool_dir = str(evidence)
        env = S.make_env(FAM, cfg)
        manifest = J.load_pool_manifest()["grid"]
        self.assertEqual(env.dynamic_hazard_header["grid"],
                         {"num_envs": manifest["num_envs"], "episodes": len(manifest["episodes"])})
        self.assertEqual(env._d7_pool.grid["eval_seed"], manifest["seed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
