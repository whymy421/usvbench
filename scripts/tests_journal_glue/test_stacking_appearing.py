"""G3 extension of the F0 / CONTEXT_R4 D4 stacking proof: the appearing suite stacks with every suite.

* All 2^7 = 128 subsets of {waves, wind, current, actuator (4 sub-axes),
  perception (4 primary axes), dynamic, appearing} at representative levels
  (D2) resolve AND admit on the journal cfgs the appearing suite is wired on
  (G20b: all twelve tasks of appearing/config.WIRED_TASKS, BlueBoat and
  WAM-V = 24 cfgs) = 3072 admissions; the
  appearing level never changes the schema-3 parameter digest, the report
  carries the appearing slot (D7 CapacityPlan.appearing_slots).
* Every appearing level D0..D_max of the active ladder resolves; D0 means "present at dose 0".
* No cfg is unwired any more (C7 never fires on a journal template);
  appearing parameters are C6; the resolved stamp of a stack WITHOUT the
  appearing suite is byte-identical to the F0 stamp (no appearing block).
* The CLI / file syntax carries appearing=D<k> like every other level token.
CPU only; run: python -B test_stacking_appearing.py
"""
from __future__ import annotations

from itertools import product
import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glue_fixtures as G  # noqa: E402

JC = G.journal_config
SUITES = ("waves", "wind", "current", "actuator", "perception", "dynamic", "appearing")
REPRESENTATIVE = dict(G.REPRESENTATIVE, appearing="appearing=D2")
TEMPLATES = None
WIRED = None


def wired():
    """Template names whose journal task is env-wired for the appearing suite (G20: appearing/config.WIRED_TASKS)."""
    global WIRED
    if WIRED is None:
        import importlib
        acfg = importlib.import_module(JC.__name__.rsplit(".", 1)[0] + ".appearing.config")
        WIRED = tuple(sorted(n for n, c in templates().items() if c.journal_task in acfg.WIRED_TASKS))
    return WIRED


def templates():
    global TEMPLATES
    if TEMPLATES is None:
        TEMPLATES = G.build_templates()
    return TEMPLATES


class AppearingStacking(unittest.TestCase):
    def test_all_128_subsets_admit_on_the_wired_cfgs(self):
        cfgs = templates()
        rows, digests = 0, {}
        for mask in product((False, True), repeat=len(SUITES)):
            subset = tuple(s for s, on in zip(SUITES, mask) if on)
            resolved = JC.resolve_journal_conditions(presets=[REPRESENTATIVE[s] for s in subset], seed=1234)
            self.assertEqual(resolved.appearing_enabled, "appearing" in subset)
            self.assertEqual(resolved.appearing_dose, 2 if "appearing" in subset else 0)
            self.assertEqual(resolved.dynamic_enabled, "dynamic" in subset)
            for name in wired():
                cfg = cfgs[name]
                report = JC.admit_journal_conditions(cfg, resolved)
                self.assertEqual(report["task"], cfg.journal_task)
                if "appearing" in subset:
                    self.assertEqual(report["appearing"]["dose"], 2)
                    self.assertEqual(report["appearing"]["vessel_id"], cfg.vehicle)
                    base = G.plans.STATIC_CAPACITY[cfg.journal_task] + G.plans.D7_RESERVED_SLOTS[cfg.journal_task]
                    self.assertEqual(report["appearing"]["slot"], base)
                else:
                    self.assertNotIn("appearing", report)
                key = tuple(s for s in subset if s not in ("appearing", "dynamic"))
                digests.setdefault(key, set()).add(report["parameters_sha256"])
                rows += 1
        self.assertEqual(len(wired()), 24)
        self.assertEqual(rows, 128 * len(wired()))
        # appearing (like dynamic) is not a schema-3 suite: it never changes the parameter digest
        self.assertTrue(all(len(v) == 1 for v in digests.values()))
        self.assertEqual(len(digests), 32)
        print(f"APPEARING STACKING PROOF: {rows} admissions (128 subsets x {len(wired())} wired cfgs), 0 rejected",
              flush=True)

    def test_every_level_resolves_and_d0_is_present(self):
        import importlib
        acfg = importlib.import_module(JC.__name__.rsplit(".", 1)[0] + ".appearing.config")
        for level in range(acfg.DOSE_COUNT):
            resolved = JC.resolve_journal_conditions([f"appearing=D{level}", "dynamic=D1"], seed=3)
            self.assertTrue(resolved.appearing_enabled)
            self.assertEqual(resolved.appearing_dose, level)
            config = resolved.appearing_config("blueboat_v2")
            self.assertEqual((config.enabled, config.dose), (True, level))
            self.assertIn(f"appearing=D{level}", resolved.tokens())
            self.assertEqual(resolved.stamp()["appearing"]["dose"], level)
            self.assertEqual(JC.resolve_journal_conditions(resolved.tokens(), seed=3).stamp(), resolved.stamp())

    def test_unwired_tasks_reject_with_c7_and_parameters_with_c6(self):
        resolved = JC.resolve_journal_conditions(["appearing=D1"], seed=1)
        rejected = []
        for name, cfg in sorted(templates().items()):
            if name in wired():
                continue
            with self.assertRaises(JC.SuiteConflict) as caught:
                JC.admit_journal_conditions(cfg, resolved)
            self.assertEqual(caught.exception.code, "C7", name)
            rejected.append(name)
        self.assertEqual(len(rejected), 0)
        import importlib
        acfg = importlib.import_module(JC.__name__.rsplit(".", 1)[0] + ".appearing.config")
        beyond = "appearing=D%d" % (acfg.MAX_DOSE + 1)
        for tokens in (["appearing=D1", "appearing.vessel_id=\"x\""], ["appearing.dose=2"], [beyond]):
            with self.assertRaises(JC.SuiteConflict) as caught:
                JC.resolve_journal_conditions(tokens, seed=1)
            self.assertEqual(caught.exception.code, "C6")
        self.assertIn("C7", JC.DECLARED_CONFLICTS)

    def test_absent_suite_leaves_the_f0_stamp_unchanged(self):
        tokens = ["waves=D2", "wind=D3", "actuator.cap=D1", "dynamic=D2", "wind.offset_deg=30"]
        resolved = JC.resolve_journal_conditions(tokens, seed=42)
        stamp = resolved.stamp()
        self.assertNotIn("appearing", stamp)
        self.assertNotIn("appearing", stamp["levels"])
        self.assertEqual(set(stamp), {"schema", "seed", "tokens", "levels", "presets",
                                      "conditions_parameters_sha256", "dynamic", "direction_rule"})
        report = JC.admit_journal_conditions(templates()["ScatterBlueBoatJournalEnvCfg"], resolved)
        self.assertNotIn("appearing", report)
        with_a = JC.resolve_journal_conditions(tokens + ["appearing=D3"], seed=42)
        self.assertEqual(with_a.conditions, resolved.conditions)
        self.assertEqual(with_a.stamp()["conditions_parameters_sha256"], stamp["conditions_parameters_sha256"])

    def test_cli_and_file_syntax(self):
        tokens = ["waves=D2", "appearing=D4", "dynamic=D1"]
        python = JC.resolve_journal_conditions(presets=tokens, seed=42)
        cli = JC.main(["--suite", *tokens, "--seed", "42"])
        with tempfile.TemporaryDirectory() as folder:
            mapped = Path(folder) / "suite_map.json"
            mapped.write_text(json.dumps({"suites": {"waves": "D2", "appearing": "D4", "dynamic": "D1"},
                                          "seed": 42}), encoding="ascii")
            from_file = JC.main(["--suite-file", str(mapped)])
        for other in (cli, from_file):
            self.assertEqual(other.stamp(), python.stamp())
            self.assertEqual((other.appearing_enabled, other.appearing_dose), (True, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
