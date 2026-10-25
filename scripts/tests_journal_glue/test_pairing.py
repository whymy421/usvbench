"""F0 / CONTEXT_R4 D11: scenario_hashes per primitive and pairing.validate_pair.

Records come from REAL episodes of glue_fixtures.FakeJournalEnv (real runtime,
condition layer and E4 bridge; the episodes end at a 3-step deadline so the
completed-episode latch runs). Two records that differ only in one suite level
pair cleanly; any other difference is rejected. CPU only; run: python -B test_pairing.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

P = G.pairing
BASE = ("waves=D1", "wind=D2", "current=D1", "actuator.cap=D1", "actuator.lag=D2", "perception.noise=D1",
        "perception.latency=D1", "perception.bundle=D1", "dynamic=D1")
CACHE = {}


def records(tokens, *, seed=G.SEED, actions=None, vessel="blueboat_v2"):
    key = (tuple(tokens), seed, None if actions is None else "alt", vessel)
    if key not in CACHE:
        cfg = G.journal_cfg(vessel, tokens=tokens, seed=seed)
        env = G.run(cfg, 3, deadline_s=3 / 60, actions=actions, pool=G.d7_pool(vessel, 2, seed=seed))
        CACHE[key] = [env.journal_episode_record(i) for i in range(2)]
    return CACHE[key]


def mid_episode_poses(tokens, *, seed=G.SEED, actions=None, vessel="blueboat_v2"):
    """G23d: own-ship positions after 2 of the 3 steps (before the deadline reset): the trajectory witness."""
    cfg = G.journal_cfg(vessel, tokens=tokens, seed=seed)
    env = G.run(cfg, 2, deadline_s=3 / 60, actions=actions, pool=G.d7_pool(vessel, 2, seed=seed))
    return env.robot.data.root_link_pose_w[:, :2].clone()


def swap(tokens, old, new):
    return tuple(new if t == old else t for t in tokens)


class Pairing(unittest.TestCase):
    def test_inventory_is_fixed_and_complete(self):
        for tokens in ((), BASE):
            hashes = records(tokens)[0]["scenario_hashes"]
            self.assertEqual(set(hashes), set(P.CONDITION_PRIMITIVES) | {P.D7_PRIMITIVE, "layout", "rotation",
                                                                         "spawn_pose"})
            self.assertTrue(all(P.is_hash(v) for v in hashes.values()))

    def test_one_suite_level_pairs_cleanly(self):
        cases = (("waves", swap(BASE, "waves=D1", "waves=D3"), ["waves"]),
                 ("wind", swap(BASE, "wind=D2", "wind=D4"), ["wind"]),
                 ("current", swap(BASE, "current=D1", "current=D3"), ["current"]),
                 ("actuator.cap", swap(BASE, "actuator.cap=D1", "actuator.cap=D3"), ["actuator.cap"]),
                 ("actuator.lag", swap(BASE, "actuator.lag=D2", "actuator.lag=D4"), ["actuator.lag"]),
                 ("perception.noise", swap(BASE, "perception.noise=D1", "perception.noise=D4"),
                  ["perception.noise"]),
                 ("perception.latency", swap(BASE, "perception.latency=D1", "perception.latency=D2"),
                  ["perception.latency"]),
                 ("perception.bundle", swap(BASE, "perception.bundle=D1", "perception.bundle=D3"),
                  ["perception.bundle"]),
                 ("dynamic", swap(BASE, "dynamic=D1", "dynamic=D3"), ["d7_assignment"]),
                 ("actuator", swap(BASE, "actuator.cap=D1", "actuator.cap=D2"), ["actuator.cap"]),
                 ("waves", BASE + ("waves.offset_deg=45",), ["waves"]))
        base = records(BASE)
        for treatment, tokens, changed in cases:
            with self.subTest(treatment=treatment, tokens=tokens):
                other = records(tokens)
                for a, b in zip(base, other):
                    result = P.validate_pair(a, b, treatment)
                    self.assertEqual(result["changed_primitives"], changed)
                    self.assertEqual(result["identity"]["env_index"], a["conditions"]["runtime"]["env_index"])

    def test_any_other_difference_is_rejected(self):
        base = records(BASE)
        two = records(swap(swap(BASE, "waves=D1", "waves=D3"), "wind=D2", "wind=D3"))
        with self.assertRaisesRegex(P.PairingError, r"outside the treatment 'waves'.*wind"):
            P.validate_pair(base[0], two[0], "waves")
        waves = records(swap(BASE, "waves=D1", "waves=D3"))
        with self.assertRaisesRegex(P.PairingError, "outside the treatment 'wind'"):
            P.validate_pair(base[0], waves[0], "wind")                  # wrong treatment declared
        offset = records(BASE + ("wind.offset_deg=30",))
        with self.assertRaisesRegex(P.PairingError, "wind"):
            P.validate_pair(base[0], offset[0], "waves")                # an undeclared offset is a difference
        other_seed = records(BASE, seed=G.SEED + 1)
        with self.assertRaisesRegex(P.PairingError, "different scenarios"):
            P.validate_pair(base[0], other_seed[0], "waves")
        with self.assertRaisesRegex(P.PairingError, "different scenarios"):
            P.validate_pair(base[0], waves[1], "waves")                 # another env row
        with self.assertRaisesRegex(P.PairingError, "no treatment difference"):
            P.validate_pair(base[0], base[0], "waves")
        with self.assertRaisesRegex(P.PairingError, "unknown treatment"):
            P.validate_pair(base[0], waves[0], "tide")
        wamv = records(BASE, vessel="wamv_vrx_v2_opt1")
        with self.assertRaises(P.PairingError):
            P.validate_pair(base[0], wamv[0], "waves")

    def test_tampering_is_detected(self):
        base, waves = records(BASE), records(swap(BASE, "waves=D1", "waves=D3"))
        forged = copy.deepcopy(waves[0])
        forged["scenario_hashes"]["wind"] = "0" * 64
        with self.assertRaisesRegex(P.PairingError, "does not match its condition stamp"):
            P.validate_pair(base[0], forged, "waves")
        forged = copy.deepcopy(waves[0])
        forged["conditions"]["parameters"]["wind"]["u10_mps"] = 12.0
        with self.assertRaisesRegex(P.PairingError, "digest"):
            P.validate_pair(base[0], forged, "waves")
        forged = copy.deepcopy(waves[0])
        forged["scenario_hashes"][P.D7_PRIMITIVE] = base[0]["scenario_hashes"]["layout"]
        with self.assertRaisesRegex(P.PairingError, "D7 stamp"):
            P.validate_pair(base[0], forged, "waves")

    def test_zero_dose_and_absent_suite_hash_identically(self):
        absent = records(())
        zero = records(("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0",
                        "perception.bundle=D0", "dynamic=D0"))
        for a, b in zip(absent, zero):
            self.assertEqual(a["scenario_hashes"], b["scenario_hashes"])
        waves = records(("waves=D2",))
        self.assertEqual(P.validate_pair(absent[0], waves[0], "waves")["changed_primitives"], ["waves"])

    def test_hashes_are_controller_independent(self):
        same = records(BASE)
        alt = lambda k: torch.tensor([[-0.7, 0.9], [1.0, 1.0]])
        other = records(BASE, actions=alt)
        # the trajectories really differ (G23d H6: the D7 clearance is no longer sampled while the intruder is
        # hidden under the reveal rules, so the own-ship positions mid-episode are the trajectory witness)
        self.assertFalse(torch.equal(mid_episode_poses(BASE), mid_episode_poses(BASE, actions=alt)))
        for a, b in zip(same, other):
            self.assertEqual(a["scenario_hashes"], b["scenario_hashes"])

    def test_header_declaration_matches_the_analysis_input_format(self):
        env = G.run(G.journal_cfg(tokens=BASE), 1)
        declaration = env.journal_certificate_header(treatment="waves")["pairing"]
        self.assertEqual(declaration["changed_primitives"], ["waves"])
        unchanged, changed = declaration["unchanged_primitives"], declaration["changed_primitives"]
        for values in (unchanged, changed):
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), len(set(values)))
            self.assertTrue(all(isinstance(v, str) and v for v in values))
        self.assertFalse(set(unchanged) & set(changed))
        self.assertEqual(sorted(unchanged + changed), sorted(env.journal_pairing_inventory))
        self.assertEqual(P.pairing_declaration(env.journal_pairing_inventory, ("waves", "wind"))["changed_primitives"],
                         ["waves", "wind"])

    def test_family_primitives_are_validated(self):
        good = {"layout": "a" * 64, "rotation": "b" * 64, "spawn_pose": "c" * 64}
        P.validate_family_hashes(good, "hazard_nav")
        for bad in ({"layout": "a" * 64}, {**good, "waves": "d" * 64}, {**good, "route": "xyz"}):
            with self.subTest(bad=sorted(bad)):
                with self.assertRaises(P.PairingError):
                    P.validate_family_hashes(bad, "hazard_nav")


if __name__ == "__main__":
    unittest.main(verbosity=2)
