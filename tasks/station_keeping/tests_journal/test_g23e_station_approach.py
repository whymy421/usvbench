"""G23e: station-keeping D7 pools under the approach design (sk_approach_crossing_v1), CPU only.

Run: python -B tasks/station_keeping/tests_journal/test_g23e_station_approach.py

Checks the shipped pools (tasks/station_keeping/d7_pools_approach, seed 42 = development keys) and the env path:
* the INDEX pins load; BlueBoat lists D1 as not applicable by design (every offered key, never a rejection),
  WAM-V none; per-dose E4 rows (0, 0) for the excluded tier; side balance 96 / 96;
* StationD7Pool refuses the excluded tier (KeyError at assignment, ValueError at check_env with that dose) and
  publishes not_applicable_doses / not_applicable_policy in the header; a pool whose keys disagree on the excluded
  tiers, or with a foreign reason, is refused;
* an in-memory admit_station_grid pool reproduces the shipped records of its keys;
* the REAL env (stubbed Isaac Lab) refuses BlueBoat dynamic=D1 at setup and runs BlueBoat dynamic=D2 with
  approach-design stamps (release.applies true, plan kind approach, sk_approach block);
* the env reads the new folder (station_keeping_env_cfg.JOURNAL_D7_POOL_DIR).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import station_journal_fixtures as F

RESTORE = None
FOLDER = F.TASKS / "station_keeping" / "d7_pools_approach"


def setUpModule():
    global RESTORE, js, dyn, envcfg
    RESTORE = F.install_isaaclab_stubs()
    js = F.ns("station_keeping.journal_station")
    dyn = F.ns("_shared.conditions.dynamic")
    envcfg = F.ns("station_keeping.station_keeping_env_cfg")


def tearDownModule():
    if RESTORE is not None:
        RESTORE()


def shipped(vessel):
    index = json.loads((FOLDER / "INDEX.json").read_text(encoding="ascii"))
    entry = index["pools"]["%s/tier2" % vessel]
    return js.StationD7Pool.load(FOLDER / entry["file"], entry["sha256"]), entry


class ShippedPools(unittest.TestCase):
    def test_env_reads_the_approach_folder(self):
        self.assertEqual(Path(envcfg.JOURNAL_D7_POOL_DIR).name, "d7_pools_approach")
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            path, sha = envcfg.journal_d7_pool_entry(vessel, 2)
            self.assertEqual(Path(path).parent, FOLDER)

    def test_not_applicable_tiers_and_counts(self):
        for vessel, na in (("blueboat_v2", (1,)), ("wamv_vrx_v2_opt1", ())):
            pool, _ = shipped(vessel)
            self.assertEqual(pool.not_applicable_doses, na)
            self.assertEqual(pool.keys_offered, 192)
            self.assertEqual(pool.keys_admitted, 192)
            self.assertEqual(pool.per_dose_counts,
                             tuple((0, 0, 0) if d in na else (192, 192, 0) for d in range(5)))
            records = pool.document["not_applicable"]
            self.assertEqual(len(records), 192 * len(na))
            self.assertEqual({r["reason"] for r in records}, {"not_applicable_by_design"} if na else set())
            self.assertEqual(pool.document["rejections"], [])
            header = pool.header_fields()
            self.assertEqual(header["not_applicable_doses"], list(na))
            self.assertIn("not applicable by design", header["not_applicable_policy"])

    def test_excluded_tier_is_refused(self):
        pool, _ = shipped("blueboat_v2")
        key = js.episode_key(42, 0, 0)
        layout = js.build_station_layout(2, "blueboat_v2", 42, 0, 0)
        pool.bind_context(key, js.station_task_context(layout, "blueboat_v2"))
        with self.assertRaises(KeyError):
            pool.assignment(key, 1)
        a = pool.assignment(key, 2)
        self.assertEqual(a.candidate.request.encounter.meta("sk_design"), "sk_approach_crossing_v1")
        suite1 = dyn.DynamicSuiteConfig(enabled=True, dose=1, vessel_id="blueboat_v2")
        with self.assertRaisesRegex(ValueError, "not applicable by design"):
            pool.check_env(vessel="blueboat_v2", seed=42, tier=2, num_envs=64, suite=suite1)
        pool.check_env(vessel="blueboat_v2", seed=42, tier=2, num_envs=64,
                       suite=dyn.DynamicSuiteConfig(enabled=True, dose=2, vessel_id="blueboat_v2"))

    def test_side_balance(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            pool, _ = shipped(vessel)
            sides = {1: 0, -1: 0}
            for env in range(64):
                for episode in range(3):
                    key = js.episode_key(42, env, episode)
                    blob = pool._candidates[js._entry_key(key, 2)]
                    side = js.candidate_from_blob(blob).request.encounter.meta("side")
                    self.assertEqual(side, (1, -1)[(env + episode) % 2])
                    sides[side] += 1
            self.assertEqual(sides, {1: 96, -1: 96})

    def test_forged_not_applicable_records_are_refused(self):
        pool, _ = shipped("blueboat_v2")
        def forged(change):
            doc = copy.deepcopy(pool.document)
            change(doc)
            doc["pool_digest"] = js.pool_manifest_digest(doc)
            return doc
        with self.assertRaisesRegex(ValueError, "differ between offered keys"):
            js.StationD7Pool(forged(lambda d: d["not_applicable"].pop()))
        def foreign(doc):
            doc["not_applicable"][0]["reason"] = "no_witness_in_frozen_search"
        with self.assertRaisesRegex(ValueError, "not-applicable record"):
            js.StationD7Pool(forged(foreign))
        def stripped(doc):
            doc["not_applicable"] = []
        with self.assertRaises(ValueError):          # the D1 rows / ladders no longer match the joint admission
            js.StationD7Pool(forged(stripped))

    def test_in_memory_admission_reproduces_the_shipped_records(self):
        pool, _ = shipped("blueboat_v2")
        raw, layouts, suite = js.admit_station_grid("blueboat_v2", 2, 42, [(0, 0), (0, 1), (1, 0), (1, 1)])
        for (key, dose), a in raw.assignments.items():
            stored = pool._accepted[(42, key.env, key.episode, js.HAZARD_GROUP, dose)]
            self.assertEqual((a.candidate_digest, a.witness_digest), stored)
        self.assertEqual(sorted((r.key.env, r.key.episode, r.dose) for r in raw.not_applicable),
                         [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
        mem = js.pool_from_admission(raw, layouts, vessel_key="blueboat_v2", seed=42, tier=2, suite=suite,
                                     grid={"num_envs": 2, "episodes": 2})
        self.assertEqual(mem.not_applicable_doses, (1,))


class EnvPath(unittest.TestCase):
    def test_blueboat_d1_is_refused_at_setup(self):
        with self.assertRaisesRegex(ValueError, "not applicable by design"):
            F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("dynamic=D1",)))

    def test_blueboat_d2_runs_the_approach_design(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("dynamic=D2",)))
        header = env.journal_certificate_header()["dynamic_hazards_header"]
        self.assertEqual(header["not_applicable_doses"], [1])
        self.assertEqual(header["per_dose_counts"][1], [0, 0, 0])
        env.reset()
        for _ in range(3):
            env.step(F.hold_controller(env))
        env.robot.data.root_link_pose_w[1, 0] += 1.0e3       # end row 1 early (out of bounds)
        env.step(F.hold_controller(env))
        stamp = dyn.read_stamp(env.journal_episode_record(1)["dynamic_hazards"])
        rel = stamp["payload"]["runtime"]["release"]
        self.assertTrue(rel["applies"])
        self.assertEqual(rel["plan_kind"], "approach")
        self.assertEqual(rel["sk_approach"]["first_applicable_dose"], 2)
        self.assertEqual(stamp["payload"]["assignment"]["dose"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
