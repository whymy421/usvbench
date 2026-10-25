"""G31 (v2.6 assembly): PPO training light doses are explicit (owner decision J10). CPU only.

Run: python -B tasks/_shared/conditions/tests/test_g31_training_levels.py
Owner decision J10 (2026-10-06): the training dose of every suite is its CURRENT D1; the appearing D1 is
5.0 m for both boats; BlueBoat station keeping has no D1 by design for the dynamic (G23e) and the appearing
(G20d) suite and trains them at D2. Covers: the level table over every journal task and both vessels, the
stack tokens / resolution / stack checks, the refusal of a D1 stack for BlueBoat station keeping, hazard
doses, the stamp (levels and numeric values enter the digest) and the numeric training values (appearing
trigger gap 5.0 m both boats except BlueBoat SK 2.9 m = D2; D7 TTC ratio 4 x t_avoid except BlueBoat SK 3).
"""
from __future__ import annotations

import importlib
import sys
import unittest

sys.dont_write_bytecode = True

import e3_fixtures as F

tm = importlib.import_module(F.PACKAGE + ".conditions.training_mixture")
jc = importlib.import_module(F.PACKAGE + ".conditions.journal_config")
ac = importlib.import_module(F.PACKAGE + ".conditions.appearing.config")
cfgv2 = importlib.import_module(F.PACKAGE + ".journal_cfg_v2")

SEED = 1001
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
EXCEPTION = ("station_keeping", "blueboat_v2")


def mixture(task, vessel, seed=SEED):
    return tm.TrainingMixture(seed=seed, task=task, vessel=vessel, appearing_wired=task in ac.WIRED_TASKS)


class Table(unittest.TestCase):
    def test_d1_everywhere_except_blueboat_station_keeping(self):
        self.assertEqual(tm.TRAINING_LEVEL, 1)
        self.assertEqual(dict(tm.TRAINING_LEVEL_EXCEPTIONS),
                         {("station_keeping", "blueboat_v2", "dynamic"): 2,
                          ("station_keeping", "blueboat_v2", "appearing"): 2})
        for task in cfgv2.TASKS:
            for vessel in VESSELS:
                m = mixture(task, vessel)
                for suite in tm.SUITE_ORDER:
                    want = 2 if (task, vessel) == EXCEPTION and suite in tm.HAZARD_SUITES else 1
                    self.assertEqual(tm.training_level(task, vessel, suite), want, (task, vessel, suite))
                self.assertEqual(m.training_levels(), {s: m.level(s) for s in m.suites})
        with self.assertRaises(tm.MixtureError):
            tm.training_level("reach", "blueboat_v2", "sensor_bundle")

    def test_station_keeping_is_wired_for_both_hazard_suites(self):
        # the exception matters only because both hazard suites run on station keeping in v2.6
        self.assertIn("station_keeping", ac.WIRED_TASKS)


class Stack(unittest.TestCase):
    def test_blueboat_station_keeping_trains_hazards_at_d2(self):
        m = mixture(*EXCEPTION)
        tokens = m.required_tokens()
        self.assertIn("dynamic=D2", tokens)
        self.assertIn("appearing=D2", tokens)
        self.assertNotIn("dynamic=D1", tokens)
        for t in ("waves=D1", "wind=D1", "current=D1", "actuator=D1", "perception=D1"):
            self.assertIn(t, tokens)
        resolved = jc.resolve_journal_conditions(tokens, seed=SEED)
        self.assertTrue(tm.check_stack(m, resolved))
        self.assertTrue(tm.check_layer_config(m, resolved.conditions))
        self.assertEqual((resolved.dynamic_dose, resolved.appearing_dose), (2, 2))

    def test_every_other_task_vessel_trains_at_d1(self):
        for task in cfgv2.TASKS:
            for vessel in VESSELS:
                if (task, vessel) == EXCEPTION:
                    continue
                m = mixture(task, vessel)
                resolved = jc.resolve_journal_conditions(m.required_tokens(), seed=SEED)
                self.assertTrue(tm.check_stack(m, resolved))
                self.assertEqual(resolved.dynamic_dose, 1, (task, vessel))
                if m.appearing_wired:
                    self.assertEqual(resolved.appearing_dose, 1, (task, vessel))

    def test_a_d1_stack_is_refused_for_blueboat_station_keeping(self):
        m = mixture(*EXCEPTION)
        for swap in (("dynamic=D2", "dynamic=D1"), ("appearing=D2", "appearing=D1")):
            tokens = [swap[1] if t == swap[0] else t for t in m.required_tokens()]
            resolved = jc.resolve_journal_conditions(tokens, seed=SEED)
            with self.assertRaises(tm.MixtureError):
                tm.check_stack(m, resolved)
        # and the WAM-V station keeping mixture refuses a D2 stack
        w = mixture("station_keeping", "wamv_vrx_v2_opt1")
        tokens = ["dynamic=D2" if t == "dynamic=D1" else t for t in w.required_tokens()]
        with self.assertRaises(tm.MixtureError):
            tm.check_stack(w, jc.resolve_journal_conditions(tokens, seed=SEED))


class DosesAndStamp(unittest.TestCase):
    def test_hazard_doses_follow_the_level(self):
        m = mixture(*EXCEPTION)
        seen = set()
        for env in range(4):
            for episode in range(2 * m.block_length):
                comp = m.component(env, episode)
                for suite in tm.HAZARD_SUITES:
                    dose = m.hazard_dose(suite, env, episode)
                    self.assertEqual(dose, 2 if comp == suite else 0)
                    seen.add(dose)
        self.assertEqual(seen, {0, 2})

    def test_stamp_carries_the_levels_and_values(self):
        b, w = mixture(*EXCEPTION), mixture("station_keeping", "wamv_vrx_v2_opt1")
        sb, sw = b.spec(), w.spec()
        self.assertEqual(sb["training_levels"]["dynamic"], 2)
        self.assertEqual(sb["training_levels"]["appearing"], 2)
        self.assertEqual(sw["training_levels"]["dynamic"], 1)
        self.assertIn("J10", sb["training_level_decision"])
        self.assertIn("dynamic=D2", sb["preset_names"])
        self.assertEqual(sb["presets"]["dynamic"]["level"], 2)
        self.assertEqual(sb["presets"]["dynamic"]["values"]["ttc_over_t_avoid"], 3.0)
        self.assertEqual(sw["presets"]["dynamic"]["values"]["ttc_over_t_avoid"], 4.0)
        self.assertNotEqual(b.digest(), w.digest())

    def test_appearing_training_gap_is_5m_both_boats(self):
        for task in ac.WIRED_TASKS:
            for vessel in VESSELS:
                m = mixture(task, vessel)
                gap = m.preset_records()["appearing"]["values"]["trigger_gap_m"]
                want = 2.9 if (task, vessel) == EXCEPTION else 5.0
                self.assertAlmostEqual(gap, want, places=12, msg=(task, vessel))


if __name__ == "__main__":
    unittest.main(verbosity=2)
