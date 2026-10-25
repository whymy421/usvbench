"""F3 round T: CONTEXT_R4 D24 episodes beyond the offered grid on the REAL PathFollowingEnv (stubbed Isaac Lab, CPU).

The per-env quota evaluators keep stepping every env until the slowest one has
its quota, so a fast env autoresets into task episodes past the pool's offered
grid (num_envs x episodes per env). D24 for F3 (cfg.journal_d7_online_admission
False, the default and the journal setting):

* (a) a key with the run's seed and hazard group, env inside the grid and
  episode index >= the grid's episodes is a NOT-ADMITTED key with the single
  reason "beyond_offered_grid": the pool answers it (never KeyError), the env
  builds the key's E5 layout (no pool geometry record exists, the comparison
  is skipped), the episode runs D0 at every requested dynamic level and the
  stamp is the D16 not_admitted stamp (dose 0, dose-independent candidate
  digest, rejections [[0, "beyond_offered_grid"]]); F0 pairing accepts it with
  one d7_assignment hash at every level;
* (b) keys outside the env range, of another seed or of another hazard group
  stay preflight errors;
* (c) dynamic_hazard_header["grid"] = {"num_envs": N, "episodes": E} from the
  pool's offered grid; per-dose counts unchanged;
* the opt-in online admission (True) keeps its behaviour (online admission of
  such keys, unchanged).
Run: python -B test_f3_d24_beyond_grid.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import f3_fixtures as F  # noqa: E402

J = F.jpf
NA = J.EXPOSURE_NOT_ADMITTED
REASON = "beyond_offered_grid"
GRID_ENVS, GRID_EPISODES = 1, 2           # test pool: env 0 x episodes 0, 1 at seed 42


def force_end(env, row=0):
    """End the running episode of ``row`` out of bounds at the next control step (code 4)."""
    env.robot.data.root_link_pose_w[row, 0] += 5000.0
    env.step(F.seek_actions(env))
    assert int(env.termination_reason[row]) == 4


def beyond_env(vessel, tokens, extra_episodes=0, online=False):
    """An env of the 1 x 2 test pool whose env 0 runs task episode E + extra_episodes (beyond the grid)."""
    cfg = F.journal_cfg(vessel, num_envs=GRID_ENVS, episodes=GRID_EPISODES, tokens=tokens)
    if online:
        cfg.journal_d7_online_admission = True
    env = F.make_env(cfg)
    env.reset()
    for _ in range(GRID_EPISODES + extra_episodes):
        force_end(env)
    assert int(env._scenario.episode_indices()[0]) == GRID_EPISODES + extra_episodes
    return env


def key_of(episode, env=0, seed=F.SEED, group=J.HAZARD_GROUP):
    return F.dynamic.EpisodeKey(seed, env, episode, group)


class BeyondGridEpisodeTests(unittest.TestCase):
    def check_stamp(self, test_env, record, vessel, episode, requested_dose):
        stamp = record["dynamic_hazards"]
        F.dynamic.read_stamp(stamp)                                       # E4 envelope + digest
        payload = stamp["payload"]
        self.assertEqual(payload["exposure"], NA)
        self.assertEqual(J.stamp_exposure(stamp), NA)
        self.assertEqual(record["dynamic_hazards_exposure"], NA)
        assignment, runtime = payload["assignment"], payload["runtime"]
        self.assertEqual((assignment["dose"], assignment["requested_dose"]), (0, requested_dose))
        self.assertEqual(assignment["intruders"], [])
        self.assertEqual(assignment["rejections"], [[0, REASON]])         # exactly one entry
        self.assertEqual(assignment["rejection_reasons"], [REASON])
        self.assertEqual(assignment["key"], {"eval_seed": F.SEED, "env": 0, "episode": episode,
                                             "hazard_group": J.HAZARD_GROUP})
        layout = J.journal_layout(vessel, 2, F.SEED, 0, episode)
        self.assertEqual(assignment["geometry_id"], layout.geometry_hash)
        self.assertEqual(assignment["candidate_digest"],
                         J.not_admitted_digest(key_of(episode), layout.geometry_hash, vessel, ((0, REASON),)))
        self.assertEqual(assignment["pool_digest"], test_env._d7_pool.pool_digest)
        self.assertEqual(payload["semantics"][REASON], J.D24_RULE)
        self.assertFalse(runtime["assigned"])
        self.assertEqual(runtime["exposure"], NA)
        self.assertEqual(runtime["non_exposure_reason"], NA)
        d7_hash = F.pairing.d7_primitive_hash(stamp)                      # F0 accepts the stamp
        self.assertEqual(record["scenario_hashes"]["d7_assignment"], d7_hash)
        return d7_hash

    def test_full_episode_beyond_the_grid_at_every_dynamic_level(self):
        vessel = "blueboat_v2"
        runs = []
        for dose in range(5):
            env = beyond_env(vessel, ("dynamic=D%d" % dose,))
            pool = env._d7_pool
            key = key_of(GRID_EPISODES)
            self.assertFalse(pool.covers(0, GRID_EPISODES))
            self.assertTrue(pool.beyond_grid(key))
            assignment = pool.assignment(key, dose)                       # never KeyError
            self.assertIsInstance(assignment, J.NotAdmittedAssignment)
            self.assertTrue(assignment.beyond_offered_grid)
            self.assertEqual(env._journal_d7_exposure[0], NA)
            self.assertEqual(env._d7_bridge.exposure(0), NA)
            layout = env._journal_layouts[0]
            self.assertEqual(layout.geometry_hash,
                             J.journal_layout(vessel, 2, F.SEED, 0, GRID_EPISODES).geometry_hash)   # E5 layout
            with self.assertRaises(ValueError):                           # the pool has no geometry record
                pool.geometry(0, GRID_EPISODES)
            forces, code = [], 0
            for _ in range(int(layout.deadline_steps) + 5):
                env.step(F.seek_actions(env))
                forces.append(torch.cat(env.robot.applied[-1], -1)[0].clone())
                code = int(env.termination_reason[0])
                if code:
                    break
                self.assertFalse(bool(env._d7_physical.any()) or bool(env._d7_visible.any()))   # no D7 object
                self.assertEqual(env._journal_d7_samples.get(0), ())
                self.assertFalse(env._d7_bridge.contact(0))
            self.assertIn(code, (1, 3), "the episode must end naturally (success or timeout)")
            record = env.journal_path_record(0)
            self.assertEqual(record["contract"]["termination_reason"], code)
            self.assertEqual(record["dynamic_hazards"]["payload"]["runtime"]["termination_reason"],
                             {1: "success", 3: "timeout"}[code])
            d7_hash = self.check_stamp(env, record, vessel, GRID_EPISODES, dose)
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["grid"], {"num_envs": GRID_ENVS, "episodes": GRID_EPISODES})
            self.assertEqual(header["grid"], pool.grid_record())
            self.assertEqual(header["beyond_grid_rule"], J.D24_RULE)
            self.assertFalse(header["online_admission"])
            self.assertEqual(header["online_admitted_keys"], [])
            self.assertEqual(header["per_dose_counts"], [list(x) for x in pool.per_dose_counts])
            self.assertTrue(all(row[0] == GRID_ENVS * GRID_EPISODES for row in header["per_dose_counts"]))
            self.assertEqual(header["keys_offered"], GRID_ENVS * GRID_EPISODES)    # not pool keys
            self.assertNotIn(REASON, str(header["not_admitted_keys"]))
            self.assertEqual(header["finished_episodes_by_exposure"][NA],
                             1 + sum(not pool.is_admitted(0, k) for k in range(GRID_EPISODES)))
            runs.append((forces, record, d7_hash))
        reference_forces, reference, reference_hash = runs[0]
        for dose, (forces, record, d7_hash) in enumerate(runs):
            self.assertEqual(d7_hash, reference_hash, dose)              # one d7_assignment hash at every level
            self.assertEqual(len(forces), len(reference_forces), dose)
            for a, b in zip(forces, reference_forces):
                self.assertTrue(torch.equal(a, b), dose)                  # every level runs D0
            self.assertEqual(record["scenario_hashes"], reference["scenario_hashes"], dose)
            self.assertEqual(record["contract"], reference["contract"], dose)
        print("beyond-grid BlueBoat key (42, 0, %d): %d control steps, code %d at every dynamic level"
              % (GRID_EPISODES, len(reference_forces), reference["contract"]["termination_reason"]))

    def test_trailing_episodes_are_unbounded_and_wamv(self):
        env = beyond_env("wamv_vrx_v2_opt1", ("dynamic=D4", "waves=D1"), extra_episodes=2)
        self.assertEqual(env._journal_d7_exposure[0], NA)
        record = env.journal_path_record(0)                               # episode E + 1, forced end
        self.check_stamp(env, record, "wamv_vrx_v2_opt1", GRID_EPISODES + 1, 4)
        self.assertEqual(env._journal_layouts[0].geometry_hash,
                         J.journal_layout("wamv_vrx_v2_opt1", 2, F.SEED, 0, GRID_EPISODES + 2).geometry_hash)
        force_end(env)
        self.check_stamp(env, env.journal_path_record(0), "wamv_vrx_v2_opt1", GRID_EPISODES + 2, 4)

    def test_beyond_grid_record_pairs_for_other_suites_and_is_unexposed(self):
        def record(tokens):
            env = beyond_env("blueboat_v2", tokens)
            force_end(env)
            return env.journal_path_record(0)
        base = record(("dynamic=D0",))
        dynamic = record(("dynamic=D3",))
        self.assertEqual(dynamic["scenario_hashes"], base["scenario_hashes"])
        with self.assertRaises(F.pairing.PairingError):                   # unexposed for the dynamic treatment
            F.pairing.validate_pair(dynamic, base, "dynamic")
        waves = record(("dynamic=D3", "waves=D2"))
        summary = F.pairing.validate_pair(waves, base, "waves")
        self.assertIn("d7_assignment", summary["unchanged_primitives"])
        self.assertIn("route", summary["unchanged_primitives"])


class BeyondGridPoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = J.read_pool_file(F.pool_file("blueboat_v2", GRID_ENVS, GRID_EPISODES))

    def test_pool_answers_beyond_grid_keys_dose_independently(self):
        pool = self.pool
        self.assertEqual(pool.grid, (GRID_ENVS, GRID_EPISODES))
        self.assertEqual(pool.grid_record(), {"num_envs": GRID_ENVS, "episodes": GRID_EPISODES})
        digests = {}
        for episode in (GRID_EPISODES, GRID_EPISODES + 1, GRID_EPISODES + 40):
            seen = set()
            for dose in range(5):
                item = pool.assignment(key_of(episode), dose)
                self.assertIsInstance(item, J.NotAdmittedAssignment)
                self.assertTrue(item.beyond_offered_grid)
                self.assertEqual(item.rejections, ((0, REASON),))
                self.assertEqual(J.primary_reasons(item.rejections), (REASON,))
                self.assertEqual(item.requested_dose, dose)
                self.assertEqual(item.geometry_id,
                                 J.journal_layout("blueboat_v2", 2, F.SEED, 0, episode).geometry_hash)
                seen.add(item.candidate_digest)
            self.assertEqual(len(seen), 1)                                # dose-independent
            digests[episode] = seen.pop()
        self.assertEqual(len(set(digests.values())), len(digests))        # key-dependent
        for episode in range(GRID_EPISODES):                              # inside the grid: unchanged
            self.assertFalse(pool.beyond_grid(key_of(episode)))
            item = pool.assignment(key_of(episode), 0)
            self.assertFalse(getattr(item, "beyond_offered_grid", False))
        self.assertEqual([list(x) for x in pool.per_dose_counts],
                         [[GRID_ENVS * GRID_EPISODES, pool.keys_admitted, pool.keys_not_admitted]] * 5)

    def test_other_keys_stay_preflight_errors(self):
        pool = self.pool
        bad = [key_of(GRID_EPISODES, env=GRID_ENVS), key_of(0, env=GRID_ENVS), key_of(GRID_EPISODES + 7, env=5),
               key_of(GRID_EPISODES, seed=F.SEED + 1), key_of(0, seed=F.SEED + 1),
               key_of(GRID_EPISODES, group="d7_station_keeping")]
        for key in bad:
            self.assertFalse(pool.beyond_grid(key), key)
            with self.assertRaises(KeyError):
                pool.assignment(key, 0)
        with self.assertRaises(KeyError):
            pool.assignment(key_of(GRID_EPISODES), 5)                     # no dose outside 0..4
        plain = J.D16Pool(pool.pool, pool.offered, pool._geometry, pool._not_admitted, pool.vessel)
        self.assertIsNone(plain.geometry_tier)                            # no tier: the D24 path is off
        with self.assertRaises(KeyError):
            plain.assignment(key_of(GRID_EPISODES), 0)

    def test_env_outside_the_env_range_or_another_seed_fails_closed(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=GRID_ENVS + 1, episodes=GRID_EPISODES)
        cfg.journal_d7_pool_path = str(F.pool_file("blueboat_v2", GRID_ENVS, GRID_EPISODES))
        env = F.make_env(cfg)                                             # pool and seed check out
        with self.assertRaises(ValueError) as caught:                     # env 1 is outside the grid's envs
            env.reset()
        self.assertIn("no admitted D7 pool record", str(caught.exception))
        self.assertIn("env %d episode 0" % GRID_ENVS, str(caught.exception))
        cfg = F.journal_cfg("blueboat_v2", num_envs=GRID_ENVS, episodes=GRID_EPISODES, seed=F.SEED + 1)
        cfg.journal_d7_pool_path = str(F.pool_file("blueboat_v2", GRID_ENVS, GRID_EPISODES))
        with self.assertRaises(ValueError) as caught:
            F.make_env(cfg)
        self.assertIn("seed", str(caught.exception))

    def test_meta_grid_must_match_the_offered_keys(self):
        pool = self.pool
        meta = dict(pool.meta)
        meta["episodes"] = GRID_EPISODES + 1
        path = Path(F._POOL_DIR) / "d24_meta_grid" / Path(pool.path).name
        J.write_pool_file(path, pool.pool, pool.offered, meta)
        with self.assertRaises(ValueError) as caught:
            J.read_pool_file(path)
        self.assertIn("grid", str(caught.exception))

    def test_committed_pools_publish_their_grid(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            committed = J.read_pool_file(J.POOL_DIR / J.pool_file_name(vessel, F.SEED))
            self.assertEqual(committed.grid_record(), {"num_envs": 64, "episodes": 2})
            self.assertEqual(committed.grid, (committed.meta["num_envs"], committed.meta["episodes"]))
            self.assertTrue(committed.beyond_grid(key_of(2, env=63)))
            self.assertFalse(committed.beyond_grid(key_of(2, env=64)))


class OnlineAdmissionUnchangedTests(unittest.TestCase):
    def test_opt_in_online_admission_keeps_its_behaviour(self):
        env = beyond_env("blueboat_v2", ("dynamic=D2",), online=True)
        header = env.dynamic_hazard_header
        self.assertTrue(header["online_admission"])
        self.assertEqual(header["beyond_grid_rule"], J.ONLINE_BEYOND_GRID_RULE)
        self.assertEqual(header["grid"], {"num_envs": GRID_ENVS, "episodes": GRID_EPISODES})   # committed grid
        self.assertIsInstance(env._d7_pool, J.OnlineExtendedPool)
        self.assertFalse(env._d7_pool.beyond_grid(key_of(GRID_EPISODES)))
        keys = header["online_admitted_keys"]
        self.assertEqual([(k["env"], k["episode"]) for k in keys], [(0, GRID_EPISODES)])
        self.assertNotEqual(keys[0]["pool_digest"], header["pool_digest"])
        assignment = env._d7_pool.assignment(key_of(GRID_EPISODES), 2)
        self.assertFalse(getattr(assignment, "beyond_offered_grid", False))
        self.assertEqual(env._journal_d7_exposure[0], keys[0]["exposure"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
