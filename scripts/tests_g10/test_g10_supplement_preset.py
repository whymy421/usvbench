"""G10 item 4: the BlueBoat 0.75 m supplementary wave preset journal.waves.supp075 (CPU).

* values: Hs 0.75 m, Tp 4 s and EVERY other wave field equal to ladder A;
* DNV-RP-C205 3.5.4 admission at import (Sp 0.0300, Ss 0.0496);
* provenance marks it a SUPPLEMENT; role / main_table stamp fields; the
  ladder presets are unchanged (no role field, same provenance record keys);
* selectable like the other presets: the F0 suite parser (token by name, C1
  against waves=D<k>, canonical tokens round-trip, stamp ``supplement``),
  eval_protocol.apply_journal_suites, run_cell --suite pass-through;
* stacks without the supplement keep a byte-identical JournalConditions stamp;
* the condition layer builds and steps it for both journal vessels.

Run: python -B scripts/tests_g10/test_g10_supplement_preset.py   (or pytest)
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

P = G.shared("conditions.presets_v3")
C = G.shared("conditions.config_v3")
JC = G.journal_config
NAME = "journal.waves.supp075"


class Values(unittest.TestCase):
    def test_values_equal_ladder_a_except_hs(self):
        supp = P.JOURNAL_PRESETS[NAME]["values"]["waves"]
        a4 = P.JOURNAL_PRESETS["journal.waves.D4"]["values"]["waves"]
        self.assertEqual(supp["hs_m"], 0.75)
        self.assertEqual({k: v for k, v in supp.items() if k != "hs_m"}, {k: v for k, v in a4.items() if k != "hs_m"})
        self.assertEqual((supp["tp_s"], supp["gamma"], supp["spreading_rad"], supp["band_min_over_fp"],
                          supp["band_max_over_fp"], supp["n_components"], supp["recurrence_min_s"]),
                         (4.0, 3.3, math.pi / 4, 0.2, 4.0, 1024, 480.0))

    def test_dnv_admission_at_import(self):
        entry = P.ADMISSION[NAME]
        self.assertTrue(entry["dnv_admitted"] and entry["generator_admitted"])
        self.assertAlmostEqual(entry["sp"], 2 * math.pi * 0.75 / (9.81 * 16.0), places=12)
        self.assertAlmostEqual(entry["sp"], 0.0300, places=4)
        self.assertAlmostEqual(entry["ss"], 0.0496, places=4)
        self.assertAlmostEqual(entry["tz_s"], 3.1107, places=4)
        self.assertLessEqual(entry["sp"], 1 / 15)
        self.assertLessEqual(entry["ss"], 1 / 10)

    def test_provenance_marks_a_supplement(self):
        record = P.provenance(NAME)
        self.assertIn("SUPPLEMENT", record["provenance"])
        self.assertIn("never a main-table level", record["provenance"])
        self.assertEqual((record["role"], record["main_table"], record["level"]), ("SUPPLEMENT", False, None))
        self.assertEqual(P.SUPPLEMENT_PRESETS, (NAME,))
        for name in P.JOURNAL_PRESETS:
            if name != NAME:
                self.assertNotIn("role", P.JOURNAL_PRESETS[name], name)
                self.assertEqual(set(P.provenance(name)), {"preset", "suite", "axis", "level", "values",
                                                           "provenance", "status", "registry"})


class Selection(unittest.TestCase):
    def test_suite_tokens(self):
        resolved = JC.resolve_journal_conditions([NAME, "wind=D1"], seed=7)
        self.assertEqual(resolved.conditions.waves.hs_m, 0.75)
        self.assertEqual(resolved.presets, ("journal.wind.D1", NAME))
        self.assertEqual(resolved.tokens(), ["wind=D1", NAME])
        again = JC.resolve_journal_conditions(resolved.tokens(), seed=7)
        self.assertEqual(again.stamp(), resolved.stamp())
        self.assertEqual(resolved.stamp()["supplement"], {"presets": [NAME], "role": "SUPPLEMENT",
                                                          "main_table": False})
        with self.assertRaises(JC.SuiteConflict) as caught:
            JC.resolve_journal_conditions([NAME, "waves=D2"], seed=7)
        self.assertEqual(caught.exception.code, "C1")

    def test_stacks_without_the_supplement_keep_their_stamp(self):
        resolved = JC.resolve_journal_conditions(["waves=D2", "wind=D3"], seed=3)
        self.assertNotIn("supplement", resolved.stamp())
        self.assertEqual(resolved.tokens(), ["waves=D2", "wind=D3"])

    def test_eval_protocol_and_run_cell(self):
        import eval_protocol
        import test_journal_evaluators as T
        cfg = G.journal_cfg("blueboat_v2")
        args = SimpleNamespace(suite=[NAME], suite_file=None)
        stamp = eval_protocol.apply_journal_suites(cfg, args)
        self.assertEqual(cfg.conditions.waves.hs_m, 0.75)
        self.assertEqual(stamp["supplement"]["presets"], [NAME])
        self.assertEqual(args.journal_suites["tokens"], [NAME])
        with patch.dict("os.environ", {}, clear=True):
            plan = T.run_cell.resolve_cell(T._cell("--suite", NAME, controller="mppi"))
        command = plan["command"]
        self.assertEqual(command[command.index("--suite") + 1], NAME)

    def test_layer_builds_and_steps_for_both_vessels(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            cfg = G.journal_cfg(vessel, tokens=[NAME])
            env = G.FakeJournalEnv(cfg, 2)
            env.reset()
            env.step(torch.tensor([[0.5, 0.5], [0.2, -0.2]]))
            resolved = env._conditions.context["resolved"]["waves"]
            self.assertEqual((resolved["hs_m"], resolved["tp_s"]), (0.75, 4.0))
            stamp = env._conditions.certificate_stamp()
            self.assertIn(NAME, stamp["parameters"]["presets"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
