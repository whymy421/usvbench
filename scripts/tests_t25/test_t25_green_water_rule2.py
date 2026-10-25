"""T25 (owner decision T1.35c): green-water rule/2 (G18 candidate R2) is the binding exclusion verdict (CPU).

* verdict_rule2: formula > 0.05 or worst simulated per-cycle exceedance > 0.05 (strictly greater; None = no
  simulated data, only the formula decides);
* exclusion_block: built from the formula, the G18 diagnostics summary and the rule/1 cell, which it reports;
  rule/1 and rule/2 can disagree in both directions (settling window and unit);
* real writer + validator: every waves certificate carries header.green_water.exclusion (binding), equal to
  its recomputation; the rule/1 cell is unchanged; a tampered or missing verdict is rejected; wave-free
  certificates carry nothing; a G10-era block (no diagnostics, no exclusion) stays valid, a block with
  diagnostics but without the verdict does not.

Run: python -B scripts/tests_t25/test_t25_green_water_rule2.py   (or pytest)
"""
from __future__ import annotations

import copy
import os
import sys
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

RULE = G.shared("conditions.green_water_rule")
DT = 1.0 / 120.0
LABELS = ["a", "b"]


def green(flag, first=0.5):
    return {"green_water": flag, "first_time_s": first if flag else None, "first_point": "a" if flag else None,
            "submerged_point_substeps": 12 if flag else 0, "longest_run_substeps": 6 if flag else 0,
            "substeps": 600}


def diag(up_f, up_0, settled=False, events=0, first=None, point=None):
    return {"version": RULE.DIAGNOSTICS_VERSION, "settle_s": RULE.SETTLE_S, "green_water_settled": settled,
            "first_time_settled_s": first, "first_point_settled": point, "events_settled": events,
            "exposure_settled_s": 600 * DT - RULE.SETTLE_S, "freeboard_upcrossings": list(up_f),
            "mean_level_upcrossings": list(up_0)}


def block_for(probability, records, diagnostics):
    cell = RULE.cell_summary(probability, records)
    summary = RULE.diagnostics_summary(probability, records, diagnostics, LABELS)
    return cell, summary, RULE.exclusion_block(probability, summary, cell)


class Verdict(unittest.TestCase):
    def test_truth_table_and_boundary(self):
        self.assertEqual(RULE.verdict_rule2(0.01, 0.01), (False, []))
        self.assertEqual(RULE.verdict_rule2(0.05, 0.05), (False, []))          # strictly greater excludes
        self.assertEqual(RULE.verdict_rule2(0.0501, None), (True, ["formula_probability_exceeds_0.05"]))
        self.assertEqual(RULE.verdict_rule2(0.0, 0.06), (True, ["simulated_per_cycle_exceedance_exceeds_0.05"]))
        self.assertEqual(RULE.verdict_rule2(0.2, 0.3)[1], ["formula_probability_exceeds_0.05",
                                                           "simulated_per_cycle_exceedance_exceeds_0.05"])
        self.assertEqual(RULE.verdict_rule2(0.0, None), (False, []))
        self.assertEqual((RULE.PER_CYCLE_LIMIT, RULE.FORMULA_LIMIT, RULE.RULE2_VERSION),
                         (0.05, 0.05, "usvbench.green_water.rule/2"))
        self.assertIn("T1.35c", RULE.RULE2_DECISION)

    def test_rule1_and_rule2_can_disagree(self):
        # a t = 0 hit in one of two episodes (share 0.5 > 0.05: rule/1 excludes) with no settled crossing
        cell, summary, excl = block_for(0.001, [green(True), green(False)],
                                        [diag([0, 0], [8, 8]), diag([0, 0], [7, 9])])
        self.assertTrue(cell["excluded_green_water"])
        self.assertEqual(summary["simulated_per_cycle_worst"], 0.0)
        self.assertFalse(excl["excluded_green_water"])
        self.assertEqual(excl["rule_1"], {"rule_version": RULE.RULE_VERSION, "binding": False,
                                          "excluded_green_water": True,
                                          "reasons": ["simulated_green_water_share_exceeds_0.05"]})
        # per-cycle exceedance 3/20 = 0.15 > 0.05 on point b without any qualifying N-substep event:
        # rule/1 keeps the cell, rule/2 excludes it
        cell, summary, excl = block_for(0.001, [green(False), green(False)],
                                        [diag([0, 2], [10, 10]), diag([0, 1], [10, 10])])
        self.assertFalse(cell["excluded_green_water"])
        self.assertEqual((summary["simulated_per_cycle_worst"], summary["simulated_per_cycle_worst_point"]),
                         (0.15, "b"))
        self.assertTrue(excl["excluded_green_water"])
        self.assertEqual(excl["reasons"], ["simulated_per_cycle_exceedance_exceeds_0.05"])
        self.assertEqual((excl["simulated_per_cycle_worst"], excl["simulated_per_cycle_worst_point"]), (0.15, "b"))
        self.assertIs(excl["binding"], True)
        # the formula alone still excludes; without diagnostics only the formula decides
        no_data = RULE.exclusion_block(0.3, None, RULE.cell_summary(0.3, []))
        self.assertTrue(no_data["excluded_green_water"])
        self.assertIsNone(no_data["simulated_per_cycle_worst"])
        self.assertTrue(no_data["simulated_basis"].startswith("none"))


class Certificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_journal_evaluators as T
        cls.T = T
        cls.docs = {tokens: T.journal_document(tokens) for tokens in (("waves=D4",), ("waves=D0",),
                                                                       ("journal.waves.supp075",))}

    def test_every_waves_certificate_carries_the_binding_verdict(self):
        for tokens, doc in self.docs.items():
            doc = copy.deepcopy(doc)
            self.assertEqual(self.T._check(doc), [], tokens)
            block = doc["header"]["green_water"]
            excl = block["exclusion"]
            self.assertEqual(excl, RULE.exclusion_block(block["formula"]["probability"], block["diagnostics"],
                                                        block["cell"], block["vessel"]), tokens)   # G28: + probe
            self.assertEqual(excl["rule_version"], RULE.RULE2_VERSION)
            self.assertEqual(excl["simulated_per_cycle_worst"], block["diagnostics"]["simulated_per_cycle_worst"])
            # the rule/1 cell keeps its G10 keys and arithmetic (reported)
            self.assertEqual(set(block["cell"]), {"rule_version", "episodes", "episodes_with_green_water", "share",
                                                  "formula_probability", "excluded_green_water", "reasons"})
            self.assertEqual(excl["rule_1"]["excluded_green_water"], block["cell"]["excluded_green_water"])
            self.assertIn("rule/2", block["diagnostics"]["candidate_verdicts"]["note"])

    def test_tampered_or_missing_verdict_is_rejected(self):
        doc = copy.deepcopy(self.docs[("waves=D4",)])
        doc["header"]["green_water"]["exclusion"]["excluded_green_water"] ^= True
        self.assertTrue(any("exclusion.excluded_green_water" in e for e in self.T._check(doc)))
        doc = copy.deepcopy(self.docs[("waves=D4",)])
        doc["header"]["green_water"]["exclusion"]["simulated_per_cycle_worst"] = 0.5
        self.assertTrue(self.T._check(doc))
        doc = copy.deepcopy(self.docs[("waves=D4",)])
        del doc["header"]["green_water"]["exclusion"]
        self.assertTrue(any("binding rule/2 verdict" in e for e in self.T._check(doc)))
        doc = copy.deepcopy(self.docs[("waves=D4",)])
        doc["header"]["green_water"]["exclusion"]["rule_1"]["excluded_green_water"] ^= True
        self.assertTrue(any("exclusion.rule_1" in e for e in self.T._check(doc)))

    def test_g10_era_blocks_and_wave_free_certificates(self):
        doc = copy.deepcopy(self.docs[("waves=D4",)])
        block = doc["header"]["green_water"]
        del block["exclusion"], block["diagnostics"], block["detection"]["diagnostics"]
        for r in doc["records"]:
            del r["green_water_diagnostics"]
        self.assertEqual(RULE.header_errors(block, doc["records"]), [])
        calm = self.T.journal_document(("wind=D2",))
        self.assertEqual(self.T._check(calm), [])
        self.assertNotIn("green_water", calm["header"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
