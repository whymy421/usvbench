"""F1: journal scenarios, D7 contexts from the REAL layouts, the pool codec and loader (CPU only).

* build_scenario: admitted E5 layout, exact quarter-turn rotation, keyed and
  deterministic, storage width = base maximum + D7 reserve;
* d7_context: TaskContext of the real layout (spawn at rest, the plan's ray
  sensor, gates mandatory, fortress witness guidance, basin NativeMotion);
* pool JSON: round trip through write_pool_file / load_d7_pool, fail closed on
  a changed file (SHA-256 pin) and on a changed candidate (digest);
* round 4b (D16): the family pool maps EVERY offered key (admitted ladder or
  not_admitted record; no admitted-only schedule), non-joint pools refused
  (the D16 env behaviour itself: test_journal_d16.py);
* the in-repo pin file tasks/hazard_nav/d7_pools/MANIFEST.json (14 pools) and,
  when the evidence folder is present beside the repository, every pinned file.

Run: python -B tasks/hazard_nav/test_journal_d7_pool.py
"""
from __future__ import annotations

import gzip
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
from types import ModuleType
import unittest

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NS = "_f1_pool"
if NS not in sys.modules:
    _root = ModuleType(NS)
    _root.__path__ = [str(REPO / "tasks")]
    sys.modules[NS] = _root
    for _folder in ("_shared", "hazard_nav"):
        _sub = ModuleType(NS + "." + _folder)
        _sub.__path__ = [str(REPO / "tasks" / _folder)]
        sys.modules[NS + "." + _folder] = _sub
J = importlib.import_module(NS + ".hazard_nav.journal_hazard_nav")
D = importlib.import_module(NS + "._shared.conditions.dynamic")
P = importlib.import_module(NS + "._shared.journal_cfg_v2")
SCRATCH = REPO.parent / "test_scratch_f1"
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def small_pool(task, vessel, envs=2, episodes=(0,), seed=42):
    suite = D.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
    ladders, offered = [], []
    for env in range(envs):
        for episode in episodes:
            sc = J.build_scenario(task, vessel, 2, seed, env, episode)
            key = J.d7_key(sc)
            offered.append(key)
            ladders.append((key, D.build_ladder(J.d7_context(sc), key, suite)))
    return D.admit_ladders(ladders), suite, offered


class Scenarios(unittest.TestCase):
    def test_quarter_turns_are_exact(self):
        rng = random.Random(3)
        pts = np.array([[rng.uniform(-200, 200), rng.uniform(-200, 200)] for _ in range(500)])
        for k in range(4):
            back = J.quarter_turn_xy(J.quarter_turn_xy(pts, k), (4 - k) % 4)
            self.assertTrue(np.array_equal(back, pts + 0.0))
            b = (-3.5, 7.25, -1.0, 2.0)
            corners = np.array([[x, y] for x in b[:2] for y in b[2:]])
            turned = J.quarter_turn_xy(corners, k)
            box = J.quarter_turn_bounds(b, k)
            self.assertEqual(box, (turned[:, 0].min(), turned[:, 0].max(), turned[:, 1].min(), turned[:, 1].max()))

    def test_every_task_and_vessel_builds_an_admitted_keyed_scenario(self):
        for task in J.CARRIED_TASKS:
            for vessel in VESSELS:
                with self.subTest(task=task, vessel=vessel):
                    a = J.build_scenario(task, vessel, 2, 42, 3, 1)
                    b = J.build_scenario(task, vessel, 2, 42, 3, 1)
                    self.assertEqual(a.geometry_id, b.geometry_id)
                    self.assertEqual(a.quarter_turns, b.quarter_turns)
                    self.assertLessEqual(a.obstacle_count, P.STATIC_CAPACITY[task])
                    self.assertEqual(a.storage_width, max(1, P.STATIC_CAPACITY[task] + P.D7_RESERVED_SLOTS[task]))
                    self.assertEqual(a.d7_slot, None if task == "moving_basin" else P.STATIC_CAPACITY[task])
                    lay = a.layout
                    self.assertTrue(np.array_equal(a.centers, J.quarter_turn_xy(lay.centers, a.quarter_turns)
                                                   .reshape(-1, 2)) or a.obstacle_count == 0)
                    self.assertEqual(a.bounds, J.quarter_turn_bounds(lay.bounds_m, a.quarter_turns))
                    self.assertAlmostEqual(math.dist(a.start_xy, a.goal_xy), math.dist(lay.start, lay.goal), places=9)
                    self.assertEqual(a.deadline_s, lay.deadline_s)
                    hashes = J.family_hashes(a, 0.1)
                    want = {"layout", "rotation", "spawn_pose"} | ({J.BASIN_PRIMITIVE} if task == "moving_basin"
                                                                   else set())
                    self.assertEqual(set(hashes), want)
        c = J.build_scenario("scatter", "blueboat_v2", 2, 42, 3, 2)
        self.assertNotEqual(c.geometry_id, J.build_scenario("scatter", "blueboat_v2", 2, 42, 3, 1).geometry_id)

    def test_d7_context_mirrors_the_real_layout(self):
        for task in J.CARRIED_TASKS:
            for vessel in VESSELS:
                with self.subTest(task=task, vessel=vessel):
                    sc = J.build_scenario(task, vessel, 2, 42, 0, 0)
                    ctx = J.d7_context(sc)
                    params = sc.layout.params
                    self.assertEqual(ctx.initial_pose.speed, 0.0)
                    self.assertEqual((ctx.initial_pose.x, ctx.initial_pose.y), sc.start_xy)
                    self.assertEqual(ctx.sensor.ray_count, 36)
                    self.assertEqual(ctx.sensor.max_range_m, params["ray_sensor_max_range_m"])
                    self.assertEqual(ctx.bounds, sc.bounds)
                    self.assertEqual(ctx.base_max, P.STATIC_CAPACITY[task])
                    self.assertEqual(ctx.geometry_id, sc.geometry_id)
                    self.assertEqual(len(ctx.static_obstacles), len(sc.static_slots))
                    self.assertEqual(ctx.envelope.vessel_id, vessel)
                    mandatory = [c for c, m in zip(ctx.route, ctx.route_mandatory) if m]
                    gates = [g for g in sc.layout.gates if g.kind == "compulsory"]
                    if task in ("ring", "forced_crossing"):
                        self.assertEqual(len(mandatory), len(gates))
                    if task == "fortress":
                        self.assertGreaterEqual(len(mandatory), len(gates))
                        self.assertTrue(ctx.route_mandatory[ctx.site_index])
                    if task == "iceberg":
                        self.assertTrue(ctx.route and not any(ctx.route_mandatory))
                    if task == "moving_basin":
                        self.assertEqual(ctx.base_slots, sc.native_slots)
                        self.assertEqual(len(ctx.native_geometry), len(sc.native_slots))
                        for slot, motion in zip(sc.native_slots, ctx.native_geometry):
                            x, y = motion.position(0.0)
                            self.assertAlmostEqual(x, sc.centers[slot, 0], places=6)
                            self.assertAlmostEqual(y, sc.centers[slot, 1], places=6)


class PoolCodec(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, SCRATCH, True)

    def test_round_trip_and_fail_closed(self):
        pool, suite, offered = small_pool("reach", "blueboat_v2")
        doc = J.pool_document(pool, task="reach", vessel="blueboat_v2", tier=2, seed=42, suite=suite,
                              offered=offered, grid={"num_envs": 2, "episodes": [0]}, provenance={})
        path = SCRATCH / J.pool_file_name("reach", "blueboat_v2")
        digest = J.write_pool_file(doc, path)
        self.assertEqual(digest, J.write_pool_file(doc, path))          # deterministic bytes
        loaded, meta = J.load_d7_pool(path, task="reach", vessel="blueboat_v2", tier=2, seed=42, suite=suite,
                                      expected_sha256=digest, with_meta=True)
        self.assertEqual(loaded.pool_digest, pool.pool_digest)
        self.assertEqual(set(loaded.assignments), set(pool.assignments))
        for key in pool.assignments:
            self.assertEqual(D.digest(loaded.assignments[key].candidate), pool.assignments[key].candidate_digest)
        self.assertEqual(meta["file_sha256"], digest)
        with self.assertRaises(ValueError):                                        # SHA-256 pin
            J.load_d7_pool(path, task="reach", vessel="blueboat_v2", tier=2, seed=42, expected_sha256="0" * 64)
        with self.assertRaises(ValueError):                                        # wrong task
            J.load_d7_pool(path, task="scatter", vessel="blueboat_v2", tier=2, seed=42, expected_sha256=digest)
        other = D.DynamicSuiteConfig(enabled=True, vessel_id="blueboat_v2", intruder_speed_mps=0.7)
        with self.assertRaises(ValueError):                                        # admission suite differs
            J.load_d7_pool(path, task="reach", vessel="blueboat_v2", tier=2, seed=42, suite=other,
                           expected_sha256=digest)
        doc2 = json.loads(json.dumps(doc))
        entry = doc2["assignments"][0][2]["f"]["candidate"]["f"]["request"]["f"]["initial_pose"]["f"]
        entry["x"] = entry["x"] + 1.0e-9
        tampered = SCRATCH / "tampered.json.gz"
        digest2 = J.write_pool_file(doc2, tampered)
        with self.assertRaises(ValueError):                                        # candidate digest
            J.load_d7_pool(tampered, task="reach", vessel="blueboat_v2", tier=2, seed=42, expected_sha256=digest2)

    def test_family_pool_maps_every_offered_key_without_a_schedule(self):
        pool, _, offered = small_pool("ring", "blueboat_v2", envs=1, episodes=(0, 1, 2, 3, 4, 5))
        fam = J.HazardD7Pool.from_admission(pool, offered)
        self.assertEqual(fam.offered, tuple(offered))
        for key in offered:                        # key (seed, 0, e) -> ITS entry, never another episode's
            got = fam.assignment(key, 2)
            if fam.is_admitted(key):
                self.assertEqual(got.candidate.request.encounter.key, key)
            else:
                self.assertEqual(got.key, key)
                self.assertIsInstance(got, J.NotAdmittedAssignment)
        self.assertFalse(hasattr(J, "ScheduledPool"))
        self.assertFalse(hasattr(J, "admitted_schedule"))

    def test_non_joint_pools_are_refused(self):
        pool, _, _ = small_pool("reach", "blueboat_v2", envs=1)
        from dataclasses import replace
        with self.assertRaises(ValueError):
            J.family_pool(replace(pool, joint=False))


class PinnedPools(unittest.TestCase):
    def test_in_repo_manifest_pins_fourteen_pools(self):
        manifest = J.load_pool_manifest()
        self.assertEqual(manifest["grid"], {"num_envs": 64, "episodes": [0, 1], "seed": 42, "tier": 2})
        names = {J.pool_file_name(t, v) for t in J.CARRIED_TASKS for v in VESSELS}
        self.assertEqual(set(manifest["pools"]), names)
        for name, entry in manifest["pools"].items():
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertEqual(entry["keys_offered"], 128)
            self.assertGreater(entry["keys_admitted"], 0, name)
            self.assertEqual(entry["keys_admitted"] + entry["keys_not_admitted"], 128, name)

    def test_evidence_pools_match_their_pins_when_present(self):
        evidence = REPO.parent / "evidence" / "d7_pools"
        if not evidence.is_dir():
            self.skipTest("evidence/d7_pools not beside this repository copy")
        manifest = J.load_pool_manifest()
        for name, entry in sorted(manifest["pools"].items()):
            path = evidence / name
            self.assertEqual(J.file_sha256(path), entry["sha256"], name)
            task, vessel = name[:-len(".json.gz")].split("__")
            pool = J.load_d7_pool(path, task=task, vessel=vessel, tier=2, seed=42)
            self.assertEqual(pool.pool_digest, entry["pool_digest"])
            self.assertEqual(pool.keys_admitted, entry["keys_admitted"])
            self.assertEqual(pool.keys_not_admitted, entry["keys_not_admitted"])

    def test_missing_pool_file_fails_closed(self):
        class Cfg:
            journal_task, vehicle, journal_geometry_tier, seed = "reach", "blueboat_v2", 2, 42
            journal_d7_pool = None
            journal_d7_pool_dir = str(SCRATCH / "nowhere")
        with self.assertRaises(FileNotFoundError):
            J.resolve_d7_pool(Cfg())


if __name__ == "__main__":
    unittest.main(verbosity=2)
