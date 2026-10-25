"""G53 (v2.6.7): the D7 pools of the competence-gate seed 11001 (selection-gate-01) are pinned (CPU only).

Run: python -B scripts/test_g53_gate_pools.py
     USVBENCH_D7_POOL_DIR=<the G53 pool bundle folder> python -B scripts/test_g53_gate_pools.py   (adds the resolve
     tests: 11001 x every D7 task x both vessels through the family's OWN loader against the bundle files)

The PPO competence gate (scripts/run_cell.py --run-purpose selection --seed-namespace selection --eval-seed-block
selection-gate-01 --level 0) builds every journal env at seed 11001; each family's env loads its D7 pool at
construction even without a dynamic dose (hazard_nav_env.py resolve_d7_pool, path_hazard_env.py resolve_pool,
journal_path_following.py load_journal_d7_pool, station_keeping_env.py journal_d7_pool_for_seed, docking_env.py
load_pool, harbor_mission_env.py resolve_pool_source). Appearing pools are loaded only when the condition stack names
the appearing suite (journal_env_mixin.py), never at level 0, so 11001 has no appearing pins.
The checks reuse scripts/test_g49_seed_pools.py (its pin helpers and its Resolve tests, run for seed 11001 only).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_g49_seed_pools as G  # noqa: E402

GATE_SEED = 11001


def setUpModule():
    G.setUpModule()


def tearDownModule():
    G.tearDownModule()


class GatePins(unittest.TestCase):
    def test_gate_seed_is_the_registry_gate_block(self):
        registry = G.load_json("scripts/seed_registry.json")
        blocks = {b["id"]: b for b in registry["namespaces"]["selection"]["blocks"]}
        self.assertEqual(blocks["selection-gate-01"]["seed"], GATE_SEED)

    def test_every_d7_task_vessel_is_pinned(self):
        for vessel in G.VESSELS:
            for task in G.HAZARD_TASKS:
                name, pin, grid = G.f1_pin(GATE_SEED, task, vessel)
                self.assertIsNotNone(pin, (task, vessel))
                self.assertEqual((pin["file"], grid["seed"]), (name, GATE_SEED))
                self.assertTrue(name.endswith("__seed11001.json.gz"), name)
            table = G.f2_table(GATE_SEED)
            self.assertEqual((table["eval_seed"], table["tier"]), (GATE_SEED, 2))
            self.assertEqual(sorted(table["vessels"][vessel]["files"]), ["0", "1", "2", "3", "4"])
            name, pin = G.f3_pin(GATE_SEED, vessel)
            self.assertEqual((pin["seed"], pin["vessel"]), (GATE_SEED, vessel), name)
            path, sha = G.M.sk.journal_d7_pool_entry(vessel, 2, GATE_SEED)
            self.assertIsNotNone(path, vessel)
            self.assertEqual(Path(path).parent.name, "d7_pools_approach")
            self.assertEqual(G.sha256_file(path), sha)                              # ships in the release
            self.assertEqual(G.M.js.StationD7Pool.load(path, sha).seed, GATE_SEED)
            name, pin = G.f5_pin(GATE_SEED, vessel)
            self.assertIsNotNone(pin, name)
            self.assertIsNotNone(G.f6_pin(GATE_SEED, vessel), vessel)

    def test_no_appearing_pins_for_the_gate_seed(self):
        for vessel in G.VESSELS:
            for task in G.M.aconf.PLACEMENT_TASKS:
                family, name, entry = G.appearing_pin(task, vessel, GATE_SEED)
                self.assertIsNone(entry, name)

    def test_station_keeping_env_switches_to_the_gate_entry(self):
        from types import SimpleNamespace
        for vessel in G.VESSELS:
            legacy = G.M.sk.journal_d7_pool_entry(vessel, 2)
            cfg = SimpleNamespace(vehicle=vessel, seed=GATE_SEED, journal_d7_pool=legacy[0],
                                  journal_d7_pool_sha256=legacy[1])
            self.assertEqual(G.M.sk.journal_d7_pool_for_seed(cfg, 2), G.M.sk.journal_d7_pool_entry(vessel, 2, GATE_SEED))
            self.assertNotEqual(G.M.sk.journal_d7_pool_for_seed(cfg, 2), legacy)


@unittest.skipUnless(os.environ.get("USVBENCH_D7_POOL_DIR"), "set USVBENCH_D7_POOL_DIR=<G53 bundle> for the resolve tests")
class GateResolve(G.Resolve):
    """test_g49_seed_pools.Resolve (D7 families) for seed 11001 only; its appearing test does not apply."""
    test_appearing = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._seeds = G.SEEDS
        G.SEEDS = (GATE_SEED,)

    @classmethod
    def tearDownClass(cls):
        G.SEEDS = cls._seeds
        super().tearDownClass()

    def test_bundle_is_the_g49_bundle(self):
        self.assertEqual(G.sha256_file(self.bundle / "BUNDLE_MANIFEST.txt"), G.G53_BUNDLE_MANIFEST_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
