"""G19 (prototype, rebased on v2.5 by G19b): the training mixture's D1 stack on all 24 journal cfgs.

For each of E5's 24 journal cfg templates (constructed by glue_fixtures under isaaclab stubs):
* the mixture's required stack (every wired suite at D1; appearing only where wired) resolves and ADMITS
  (journal_config.admit_journal_conditions, the same admission the evaluators use);
* the condition-layer factory attaches the mixture from ``cfg.journal_training_mixture`` on the REAL journal
  observation contract (estimate groups pos / vel / ray), and refuses a stack that is not the mixture's;
* a layer with a mixture refuses the certificate stamp.
CPU only; run: python -B test_training_mixture_journal.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glue_fixtures as G  # noqa: E402

JC = G.journal_config
TM = G.shared("conditions.training_mixture")
APPEARING = G.shared("conditions.appearing.config")
SEED = 1001
TEMPLATES = None


def templates():
    global TEMPLATES
    if TEMPLATES is None:
        TEMPLATES = G.build_templates()
    return TEMPLATES


def mixture_for(cfg):
    return TM.TrainingMixture(seed=SEED, task=cfg.journal_task, vessel=cfg.vehicle,
                              appearing_wired=cfg.journal_task in APPEARING.WIRED_TASKS)


class TrainingStackOnJournalCfgs(unittest.TestCase):
    def test_required_stack_admits_and_attaches_on_all_24(self):
        cfgs = templates()
        self.assertEqual(len(cfgs), 24)
        wired = 0
        for name, cfg in sorted(cfgs.items()):
            m = mixture_for(cfg)
            wired += m.appearing_wired
            resolved = JC.resolve_journal_conditions(m.required_tokens(), seed=SEED)
            report = JC.admit_journal_conditions(cfg, resolved)
            self.assertEqual(report["task"], cfg.journal_task, name)
            self.assertEqual("appearing" in report, m.appearing_wired, name)
            self.assertTrue(report["dynamic"]["enabled"], name)
            # G31 / owner decision J10: D1, except BlueBoat station keeping = D2 (no D1 by design)
            self.assertEqual(report["dynamic"]["dose"], m.level("dynamic"), name)
            self.assertEqual(m.level("dynamic"),
                             2 if (cfg.journal_task, cfg.vehicle) == ("station_keeping", "blueboat_v2") else 1, name)
            built = copy.copy(cfg)
            JC.apply_journal_conditions(built, resolved)
            built.journal_training_mixture = m
            layer = G.conditions.build_conditions(built, 4, "cpu", family=JC.journal_family(built))
            self.assertIs(layer.training_mixture, m, name)
            self.assertTrue(getattr(layer.contract, "journal", False), name)
            self.assertFalse(layer.context["certificate_eligible"], name)
        # v2.5: scatter and path following, both vessels (4); G19b: read from WIRED_TASKS so the appearing
        # rollout (G20b) changes the expectation, never the test
        self.assertEqual(wired, 2 * len(set(APPEARING.WIRED_TASKS) & {c.journal_task for c in cfgs.values()}))
        self.assertGreaterEqual(wired, 4)
        print(f"G19 TRAINING STACK: 24 journal cfgs admitted and attached ({wired} with appearing)", flush=True)

    def test_a_foreign_stack_is_refused_by_the_factory(self):
        cfg = templates()["ScatterBlueBoatJournalEnvCfg"]
        m = mixture_for(cfg)
        resolved = JC.resolve_journal_conditions(["waves=D2"] + m.required_tokens()[1:], seed=SEED)
        built = copy.copy(cfg)
        JC.apply_journal_conditions(built, resolved)
        built.journal_training_mixture = m
        with self.assertRaises(TM.MixtureError):
            G.conditions.build_conditions(built, 4, "cpu", family=JC.journal_family(built))


if __name__ == "__main__":
    unittest.main(verbosity=2)
