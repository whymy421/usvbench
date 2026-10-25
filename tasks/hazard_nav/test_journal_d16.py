"""F1 round 4b: CONTEXT_R4 D16 (D7-rejected keys never change the task layout), CPU only.

* pool semantics: per offered key either the admitted five-dose ladder or an
  explicit not_admitted record with E4's rejection reasons; per-dose counts
  (offered, admitted, not_admitted); keys outside the grid stay a preflight
  error; the pool JSON round trip keeps the records and fails closed when
  they do not recompute;
* a not-admitted key runs a FULL episode at EVERY dynamic level D0..D4 with
  ``payload.exposure = "not_admitted"`` (no intruder, empty assignment,
  reasons stamped), in the same env as admitted rows, and pairs
  (pairing.validate_pair) with the same key at D0 of every other suite; for
  the dynamic treatment it is unexposed (no treatment difference);
* the layout of a not-admitted key IS the E5 layout of that key (no redraw)
  and the env's episode counter never skips a scenario;
* moving basin: a not-admitted key runs E4's D0 (native population frozen at
  its time-zero positions in its own slots, legacy movers never called);
* header: per-dose counts include not_admitted, keys_not_admitted, pins.

Run: python -B tasks/hazard_nav/test_journal_d16.py
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

def _d7_shown(env, row):
    """G23c L2 (rule S2): a row's D7 intruder is physical exactly when its runtime rule has revealed it
    (hidden before the reveal; rows without a runtime rule: physical when they carry an intruder)."""
    episode = env._d7_bridge.adapter.episodes.get(int(row))
    if episode is None:
        return False
    if episode.release is None:
        return bool(episode.objects)
    return bool(episode.release.released_at(episode.last_time_s))


import journal_test_support as S  # noqa: E402

RESTORE = S.install_isaaclab_stubs()
FAM = S.load_family()
J, D, P = FAM.journal, FAM.dynamic, FAM.pairing
LAYOUTS = __import__(S.ROOT + "._shared.journal_layouts_v2", fromlist=["build_layout"])
SCRATCH = S.REPO.parent / "test_scratch_f1_d16"
# probe grid below holds admitted AND not-admitted keys (G7: fortress; under the G7 window rule ring /
# BlueBoat admits all 128 seed-42 keys, fortress env 1 stays a real E4 rejection)
TASK, VESSEL = "fortress", "blueboat_v2"
NUM_ENVS, EPISODES = 3, (0, 1)
OTHER_SUITES = (("waves=D2", "waves"), ("wind=D2", "wind"), ("current=D2", "current"),
                ("actuator=D2", "actuator"), ("perception=D2", "perception"),
                ("perception.bundle=D2", "perception.bundle"))


def probe_pool():
    return S.admitted_pool(FAM, TASK, VESSEL, num_envs=NUM_ENVS, episodes=EPISODES)


def family(pool):
    return J.family_pool(pool)


def na_rows(pool, episode=0):
    fp = family(pool)
    return [i for i in range(NUM_ENVS) if not fp.is_admitted(D.EpisodeKey(S.SEED, i, episode, "d7_" + TASK))]


def run_episode(tokens, steps=2, pool=None, task=TASK, vessel=VESSEL, num_envs=NUM_ENVS):
    pool = probe_pool() if pool is None else pool
    cfg = S.journal_cfg(FAM, task, vessel, tokens=tokens, num_envs=num_envs, pool=pool)
    env = S.make_env(FAM, cfg)
    env.reset()
    env._journal_twin.deadline[:] = steps * FAM.plans.CONTROL_DT_S
    for _ in range(steps):
        env.step(S.zero_action(env, (0.4, 0.05)))
    return env


class PoolSemantics(unittest.TestCase):
    def test_probe_grid_has_both_kinds(self):
        pool = family(probe_pool())
        self.assertGreater(pool.keys_admitted, 0)
        self.assertGreater(pool.keys_not_admitted, 0)
        self.assertEqual(pool.keys_admitted + pool.keys_not_admitted, pool.keys_offered)
        self.assertEqual(pool.per_dose_counts, tuple((pool.keys_offered, pool.keys_admitted,
                                                      pool.keys_not_admitted) for _ in range(5)))

    def test_assignment_is_a_ladder_or_a_not_admitted_record_at_every_dose(self):
        pool = family(probe_pool())
        for key in pool.offered:
            for dose in range(5):
                got = pool.assignment(key, dose)
                if pool.is_admitted(key):
                    self.assertIsInstance(got, D.Assignment)
                    self.assertEqual(got.candidate.request.encounter.dose, dose)
                else:
                    self.assertIsInstance(got, J.NotAdmittedAssignment)
                    self.assertEqual((got.key, got.dose), (key, dose))
                    self.assertTrue(got.reasons)
                    want = [(r.dose, r.reason, r.sampled_states) for r in pool.rejections if r.key == key]
                    self.assertEqual(list(got.reasons), want)
        # CONTEXT_R4 D24 (round T): episode >= the grid's episodes per env (env inside the grid, same seed and
        # group) is a not-admitted key with the single reason beyond_offered_grid (test_journal_d24.py);
        # an env outside the grid stays a preflight error
        beyond = pool.assignment(D.EpisodeKey(S.SEED, 0, 99, "d7_" + TASK), 0)
        self.assertEqual(beyond.reasons, ((0, "beyond_offered_grid", 0),))
        with self.assertRaises(KeyError):
            pool.assignment(D.EpisodeKey(S.SEED, NUM_ENVS, 0, "d7_" + TASK), 0)
        with self.assertRaises(ValueError):
            J.family_pool(replace(probe_pool(), joint=False))

    def test_pool_file_round_trip_keeps_not_admitted_records(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, SCRATCH, True)
        admission = probe_pool()
        suite = D.DynamicSuiteConfig(enabled=True, vessel_id=VESSEL)
        offered, geometry = [], {}
        for env in range(NUM_ENVS):
            for episode in EPISODES:
                sc = J.build_scenario(TASK, VESSEL, 2, S.SEED, env, episode)
                offered.append(J.d7_key(sc))
                geometry[J.d7_key(sc)] = sc.geometry_id
        doc = J.pool_document(admission, task=TASK, vessel=VESSEL, tier=2, seed=S.SEED, suite=suite,
                              offered=offered, grid={"num_envs": NUM_ENVS, "episodes": list(EPISODES)},
                              provenance={}, geometry_ids=geometry)
        self.assertEqual(doc["schema"], J.POOL_SCHEMA)
        self.assertEqual(doc["per_dose_count_fields"], ["offered", "admitted", "not_admitted"])
        path = SCRATCH / J.pool_file_name(TASK, VESSEL)
        digest = J.write_pool_file(doc, path)
        pool = J.load_d7_pool(path, task=TASK, vessel=VESSEL, tier=2, seed=S.SEED, suite=suite,
                              expected_sha256=digest)
        self.assertIsInstance(pool, J.HazardD7Pool)
        self.assertEqual(pool.keys_not_admitted, len(doc["not_admitted"]))
        for key, record in pool.not_admitted.items():
            self.assertEqual(record.geometry_id, geometry[key])
        for mutate in (lambda d: d["not_admitted"].pop(),
                       lambda d: d["not_admitted"][0]["reasons"][0].__setitem__(1, "edited"),
                       lambda d: d["per_dose_counts"][2].__setitem__(2, 0)):
            bad = json.loads(json.dumps(doc))
            mutate(bad)
            tampered = SCRATCH / "tampered.json.gz"
            with self.assertRaises(ValueError):
                J.load_d7_pool(tampered, task=TASK, vessel=VESSEL, tier=2, seed=S.SEED,
                               expected_sha256=J.write_pool_file(bad, tampered))


class NotAdmittedEpisodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.na = na_rows(probe_pool())
        cls.admitted = [i for i in range(NUM_ENVS) if i not in cls.na]
        assert cls.na and cls.admitted, "probe grid must mix admitted and not-admitted keys"
        cls.records = {}
        for dose in range(5):
            env = run_episode(("dynamic=D%d" % dose,))
            cls.records[dose] = (env, [env.journal_completed_records[i] for i in range(NUM_ENVS)])

    def test_full_episode_at_every_dynamic_level_with_exposure_not_admitted(self):
        pool = family(probe_pool())
        hashes = set()
        for dose, (env, records) in self.records.items():
            for i in self.na:
                rec = records[i]
                self.assertIsNotNone(rec, (dose, i))
                self.assertEqual(rec["termination_name"], "timeout")
                self.assertEqual(rec["d7_exposure"], "not_admitted")
                stamp = D.read_stamp(rec["dynamic_hazards"])            # E4 envelope format and digest
                payload = stamp["payload"]
                self.assertEqual(payload["exposure"], "not_admitted")
                self.assertEqual(payload["assignment"]["dose"], 0)
                self.assertEqual(payload["assignment"]["intruders"], [])
                self.assertIsNone(payload["assignment"]["candidate_digest"])
                self.assertEqual(payload["runtime"]["suite_dose"], dose)
                self.assertFalse(payload["runtime"]["assigned"])
                key = D.EpisodeKey(S.SEED, i, 0, "d7_" + TASK)
                self.assertEqual(payload["not_admitted"]["reasons"], pool.not_admitted[key].reason_documents())
                self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 2 * env.cfg.decimation)
                hashes.add(rec["scenario_hashes"]["d7_assignment"])
            for i in self.admitted:
                self.assertEqual(records[i]["d7_exposure"], "admitted")
                self.assertEqual(records[i]["dynamic_hazards"]["payload"]["assignment"]["dose"], dose)
            header = env.journal_certificate_header(treatment="dynamic")["dynamic_hazards_header"]
            self.assertEqual(header["per_dose_count_fields"], ["offered", "admitted", "not_admitted"])
            self.assertEqual(header["per_dose_counts"], [[pool.keys_offered, pool.keys_admitted,
                                                          pool.keys_not_admitted]] * 5)
            self.assertEqual(header["keys_not_admitted"], pool.keys_not_admitted)
        self.assertEqual(len(hashes), len(self.na))       # one d7 hash per key, identical at every dose

    def test_no_intruder_is_ever_published_for_a_not_admitted_row(self):
        env = S.make_env(FAM, S.journal_cfg(FAM, TASK, VESSEL, tokens=("dynamic=D3",), num_envs=NUM_ENVS,
                                            pool=probe_pool()))
        env.reset()
        slot = FAM.plans.STATIC_CAPACITY[TASK]
        for _ in range(3):
            for i in self.na:
                self.assertFalse(bool(env._d7_physical[i].any()))
                self.assertFalse(bool(env._d7_visible[i].any()))
                self.assertTrue(env._d7_bridge.is_not_admitted(i))
                self.assertFalse(env._d7_bridge.contact(i))
            for i in self.admitted:                 # G23c L2 (rule S2): physical once revealed
                self.assertEqual(bool(env._d7_physical[i, slot]), _d7_shown(env, i))
            env.step(S.zero_action(env, (0.5, 0.0)))

    def test_pairs_with_the_same_key_at_d0_of_every_other_suite(self):
        for tokens, treatment in OTHER_SUITES:
            env = run_episode((tokens, "dynamic=D0"))
            for i in self.na:
                other = env.journal_completed_records[i]
                self.assertEqual(other["d7_exposure"], "not_admitted")
                for dose in range(5):
                    mine = self.records[dose][1][i]
                    summary = P.validate_pair(mine, other, treatment)
                    self.assertNotIn("d7_assignment", summary["changed_primitives"])
                    self.assertIn("layout", summary["unchanged_primitives"])
        # suite absent (no dynamic token) pairs as well
        env = run_episode(("waves=D2",))
        for i in self.na:
            P.validate_pair(self.records[3][1][i], env.journal_completed_records[i], "waves")

    def test_unexposed_for_the_dynamic_treatment(self):
        for i in self.na:
            for dose in range(1, 5):
                with self.assertRaises(P.PairingError):            # no treatment difference
                    P.validate_pair(self.records[0][1][i], self.records[dose][1][i], "dynamic")
        for i in self.admitted:
            P.validate_pair(self.records[0][1][i], self.records[2][1][i], "dynamic")

    def test_layout_is_the_e5_layout_of_the_key(self):
        env = S.make_env(FAM, S.journal_cfg(FAM, TASK, VESSEL, tokens=("dynamic=D2",), num_envs=NUM_ENVS,
                                            pool=probe_pool()))
        env.reset()
        pool = family(probe_pool())
        for i in self.na:
            sc = env._journal_scenarios[i]
            self.assertEqual((sc.env, sc.episode), (i, 0))
            lay = LAYOUTS.build_layout(TASK, 2, VESSEL, J.scenario_generator(S.SEED, i, 0, J.LAYOUT_GROUP + TASK),
                                       max_attempts=J.LAYOUT_MAX_ATTEMPTS)
            self.assertEqual(sc.layout.geometry_hash, lay.geometry_hash)
            self.assertEqual(sc.geometry_id, J.build_scenario(TASK, VESSEL, 2, S.SEED, i, 0).geometry_id)
            rec = self.records[2][1][i]
            self.assertEqual(rec["scenario"]["layout_geometry_hash"], lay.geometry_hash)
            self.assertEqual(rec["dynamic_hazards"]["payload"]["assignment"]["geometry_id"], sc.geometry_id)
            self.assertIn(D.EpisodeKey(S.SEED, i, 0, "d7_" + TASK), pool.not_admitted)

    def test_stale_not_admitted_geometry_fails_closed(self):
        admission = probe_pool()
        key = D.EpisodeKey(S.SEED, self.na[0], 0, "d7_" + TASK)
        pool = J.HazardD7Pool.from_admission(admission, geometry_ids={key: "0" * 64})
        env = S.make_env(FAM, S.journal_cfg(FAM, TASK, VESSEL, tokens=("dynamic=D2",), num_envs=NUM_ENVS,
                                            pool=pool))
        with self.assertRaises(ValueError):
            env.reset()

    def test_episode_counter_never_skips_a_scenario(self):
        env = S.make_env(FAM, S.journal_cfg(FAM, TASK, VESSEL, tokens=("dynamic=D2",), num_envs=NUM_ENVS,
                                            pool=probe_pool()))
        env.reset()
        for episode in EPISODES:
            for i in range(NUM_ENVS):
                self.assertEqual(env._journal_scenarios[i].episode, episode)
                self.assertEqual(int(env._journal_episode_ordinal[i]), episode)
            env._journal_twin.deadline[:] = FAM.plans.CONTROL_DT_S
            env.step(S.zero_action(env))
        # episode 2 is beyond the probe grid: since CONTEXT_R4 D24 it runs at D0 (beyond_offered_grid)
        for i in range(NUM_ENVS):
            self.assertEqual(env._journal_scenarios[i].episode, len(EPISODES))
            self.assertEqual(int(env._journal_episode_ordinal[i]), len(EPISODES))
            self.assertTrue(env._d7_bridge.is_not_admitted(i))


class MovingBasinNotAdmitted(unittest.TestCase):
    def synthetic_pool(self):
        """A real basin pool with key (seed, 0, 0) turned into a not-admitted key (test construction)."""
        admission = S.admitted_pool(FAM, "moving_basin", VESSEL, num_envs=1, episodes=(0, 1))
        key = D.EpisodeKey(S.SEED, 0, 0, "d7_moving_basin")
        if (key, 0) not in admission.assignments:
            return admission
        kept = {kd: a for kd, a in admission.assignments.items() if kd[0] != key}
        rejections = admission.rejections + (D.Rejection(key, 2, "synthetic_test_rejection", 0),)
        counts = tuple((o, a - 1) for o, a in admission.per_dose_counts)
        return replace(admission, assignments=kept, rejections=rejections, keys_admitted=admission.keys_admitted - 1,
                       per_dose_counts=counts)

    def test_basin_runs_frozen_native_population(self):
        pool = self.synthetic_pool()
        cfg = S.journal_cfg(FAM, "moving_basin", VESSEL, tokens=("dynamic=D4",), num_envs=1, pool=pool)
        env = S.make_env(FAM, cfg)

        def forbidden(*args, **kwargs):
            raise AssertionError("legacy mover called in the journal basin")
        env._update_moving_obstacles = forbidden
        env._maybe_appear_obstacles = forbidden
        env.reset()
        self.assertTrue(env._d7_bridge.is_not_admitted(0))
        sc = env._journal_scenarios[0]
        native = list(sc.native_slots)
        self.assertTrue(bool(env._d7_physical[0, native].all()))
        self.assertFalse(bool(env.obstacle_active[0, native].any()))
        start = env.obstacle_centers[0, native].clone()
        local = start - env.scene.env_origins[0, :2]
        for row, slot in enumerate(native):
            self.assertAlmostEqual(float(local[row, 0]), float(sc.centers[slot, 0]), places=4)
            self.assertAlmostEqual(float(local[row, 1]), float(sc.centers[slot, 1]), places=4)
        env._journal_twin.deadline[:] = 4 * FAM.plans.CONTROL_DT_S
        for _ in range(3):
            env.step(S.zero_action(env, (0.5, 0.0)))
            self.assertTrue(torch.equal(env.obstacle_centers[0, native], start))     # frozen (E4 D0)
            self.assertTrue(bool(env._d7_physical[0, native].all()))
        env.step(S.zero_action(env, (0.5, 0.0)))
        rec = env.journal_completed_records[0]
        self.assertEqual(rec["termination_name"], "timeout")
        self.assertFalse(env._d7_bridge.is_not_admitted(0))        # next key (seed, 0, 1) is admitted
        payload = D.read_stamp(rec["dynamic_hazards"])["payload"]
        self.assertEqual(payload["exposure"], "not_admitted")
        self.assertEqual(payload["runtime"]["suite_dose"], 4)
        self.assertEqual(len(payload["assignment"]["native_objects_frozen"]), len(native))
        self.assertTrue(all(o["speed_bound_mps"] == 0.0 for o in payload["assignment"]["native_objects_frozen"]))
        self.assertEqual(payload["not_admitted"]["reasons"][-1][1], "synthetic_test_rejection")


class PinnedPoolsD16(unittest.TestCase):
    def test_pinned_manifest_counts_admitted_and_not_admitted(self):
        manifest = J.load_pool_manifest()
        self.assertEqual(manifest["pool_schema"], J.POOL_SCHEMA)
        for name, entry in manifest["pools"].items():
            self.assertEqual(entry["keys_admitted"] + entry["keys_not_admitted"], entry["keys_offered"], name)
            self.assertEqual(entry["per_dose_count_fields"], ["offered", "admitted", "not_admitted"])
            for row in entry["per_dose_counts"]:
                self.assertEqual(row, [entry["keys_offered"], entry["keys_admitted"], entry["keys_not_admitted"]])

    def test_evidence_not_admitted_geometry_is_the_e5_layout(self):
        evidence = S.REPO.parent / "evidence" / "d7_pools"
        if not evidence.is_dir():
            self.skipTest("evidence/d7_pools not beside this repository copy")
        checked = 0
        for name in sorted(J.load_pool_manifest()["pools"]):
            task, vessel = name[:-len(".json.gz")].split("__")
            if task == "moving_basin":
                continue                                  # 6-13 s per basin layout; covered by the tool
            pool = J.load_d7_pool(evidence / name, task=task, vessel=vessel, tier=2, seed=S.SEED)
            for key, record in list(pool.not_admitted.items())[:2]:
                sc = J.build_scenario(task, vessel, 2, S.SEED, key.env, key.episode)
                self.assertEqual(record.geometry_id, sc.geometry_id, (name, key))
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
