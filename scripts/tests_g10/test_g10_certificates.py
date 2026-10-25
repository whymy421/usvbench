"""G10 item 3: green-water certificate fields, schema and validator rules (CPU).

Certificates come from the REAL cert_v3 writer over the REAL mixin
(scripts/test_journal_evaluators.journal_document, F0 FakeJournalEnv):
* a journal certificate with waves carries header.green_water (rule, formula,
  detection, cell summary) and one green_water record per episode; it validates;
* wave-free journal certificates carry neither and stay valid;
* the validator rejects: a missing header block or record, a cell that
  disagrees with the records, a verdict inconsistent with its two numbers,
  relaxed limits, an inconsistent record, formula inputs that differ from the
  condition stamp, a green-water block on a wave-free certificate;
* the verdict rule: formula > 0.05 OR share > 0.05 (strictly greater);
* the writer refuses a waves journal env that publishes no green-water header.

Run: python -B scripts/tests_g10/test_g10_certificates.py   (or pytest)
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
from jsonschema import Draft202012Validator  # noqa: E402

import test_journal_evaluators as T  # noqa: E402
import validate_cert_v3  # noqa: E402

RULE = validate_cert_v3.green_water_rule_module()


def check(doc):
    return T._check(doc)


def has(errors, text):
    return any(text in e for e in errors)


class Rule(unittest.TestCase):
    def test_verdict_is_strictly_greater_on_either_number(self):
        self.assertEqual(RULE.verdict(0.05, 0.05), (False, []))
        self.assertEqual(RULE.verdict(0.0500001, 0.0)[0], True)
        self.assertEqual(RULE.verdict(0.0, 4 / 64)[1], ["simulated_green_water_share_exceeds_0.05"])
        self.assertEqual(RULE.verdict(0.06, 0.1)[1], ["formula_probability_exceeds_0.05",
                                                      "simulated_green_water_share_exceeds_0.05"])
        self.assertEqual(RULE.verdict(0.01, None), (False, []))
        self.assertEqual((RULE.FORMULA_LIMIT, RULE.SHARE_LIMIT), (0.05, 0.05))

    def test_cell_summary_counts_any_green_water(self):
        records = [{"green_water": g} for g in (True, False, False, True)]
        cell = RULE.cell_summary(0.001, records)
        self.assertEqual((cell["episodes"], cell["episodes_with_green_water"], cell["share"]), (4, 2, 0.5))
        self.assertTrue(cell["excluded_green_water"])
        empty = RULE.cell_summary(0.2, [])
        self.assertIsNone(empty["share"])
        self.assertTrue(empty["excluded_green_water"])


class Certificates(unittest.TestCase):
    def doc(self, tokens=("waves=D4",)):
        return T.journal_document(tokens)

    def test_waves_certificate_carries_and_validates_the_fields(self):
        for tokens in (("waves=D4",), ("waves=D0",), ("journal.waves.supp075",), ("waves.B=D3", "wind=D1")):
            doc = self.doc(tokens)
            self.assertEqual(check(doc), [], tokens)
            block = doc["header"]["green_water"]
            self.assertEqual(block["schema"], "usvbench.green_water_header/1")
            self.assertEqual(block["rule"]["formula_limit"], 0.05)
            self.assertEqual(block["formula"]["criterion"], "NORDFORSK per-cycle wetness, limit 0.05")
            cell = block["cell"]
            self.assertEqual(cell["episodes"], len(doc["records"]))
            self.assertEqual(cell["episodes_with_green_water"],
                             sum(r["green_water"]["green_water"] for r in doc["records"]))
            self.assertEqual(cell["excluded_green_water"],
                             cell["formula_probability"] > 0.05 or (cell["share"] or 0) > 0.05)
            for record in doc["records"]:
                self.assertEqual(set(record["green_water"]), set(RULE.RECORD_FIELDS))

    def test_wave_free_certificates_carry_nothing_and_stay_valid(self):
        for tokens in ((), ("wind=D2",), ("current=D1", "actuator=D1")):
            doc = self.doc(tokens)
            self.assertEqual(check(doc), [], tokens)
            self.assertNotIn("green_water", doc["header"])
            self.assertTrue(all("green_water" not in r for r in doc["records"]))

    def test_validator_rejections(self):
        doc = self.doc()
        bad = copy.deepcopy(doc)
        del bad["header"]["green_water"]
        self.assertTrue(has(check(bad), "green_water"))
        bad = copy.deepcopy(doc)
        del bad["records"][0]["green_water"]
        self.assertTrue(has(check(bad), "green_water"))
        # T25 (T1.35a, spawn on waves by default): forged values relative to the actual cell (the fixture's
        # green-episode count is no longer 0)
        cell = doc["header"]["green_water"]["cell"]
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["cell"]["episodes_with_green_water"] = cell["episodes_with_green_water"] + 1
        self.assertTrue(has(check(bad), "cell.episodes_with_green_water"))
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["cell"]["excluded_green_water"] = not cell["excluded_green_water"]
        self.assertTrue(has(check(bad), "excluded_green_water"))
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["cell"]["share"] = 0.25 if cell["share"] != 0.25 else 0.75
        self.assertTrue(has(check(bad), "cell.share"))
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["rule"]["share_limit"] = 0.1
        self.assertTrue(check(bad))                       # schema const (limits never relaxed)
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["formula"]["limit"] = 0.1
        self.assertTrue(check(bad))
        # T25 (T1.35a): tamper a DRY record (record 1 of the fixture is no longer dry)
        dry = next(i for i, r in enumerate(doc["records"]) if r["green_water"]["green_water"] is False)
        bad = copy.deepcopy(doc)
        bad["records"][dry]["green_water"].update(green_water=True)
        self.assertTrue(has(check(bad), "green_water true"))
        bad = copy.deepcopy(doc)
        bad["records"][dry]["green_water"].update(longest_run_substeps=6)
        self.assertTrue(has(check(bad), "must set green_water true"))
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["formula"]["inputs"]["hs_m"] = 0.5
        self.assertTrue(has(check(bad), "formula.inputs.hs_m"))
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["formula"]["vessel"] = "wamv_vrx_v2_opt1"
        self.assertTrue(has(check(bad), "formula.vessel"))
        bad = copy.deepcopy(doc)
        bad["header"]["green_water"]["detection"]["n_consecutive_substeps"] = 3
        self.assertTrue(check(bad))

    def test_verdict_must_follow_its_numbers(self):
        doc = self.doc()
        bad = copy.deepcopy(doc)
        # a forged formula probability above 0.05 with an unchanged "not excluded" verdict
        bad["header"]["green_water"]["formula"]["probability"] = 0.2
        bad["header"]["green_water"]["cell"]["formula_probability"] = 0.2
        # T25 (T1.35a): the fixture's cell may already be excluded by its share; force the "not excluded" verdict
        bad["header"]["green_water"]["cell"]["excluded_green_water"] = False
        bad["header"]["green_water"]["cell"]["reasons"] = []
        errors = check(bad)
        self.assertTrue(has(errors, "excluded_green_water"), errors)

    def test_green_water_on_a_wave_free_certificate_is_rejected(self):
        waves = self.doc()
        calm = self.doc(("wind=D2",))
        bad = copy.deepcopy(calm)
        bad["header"]["green_water"] = copy.deepcopy(waves["header"]["green_water"])
        self.assertTrue(has(check(bad), "has no waves"))
        bad = copy.deepcopy(calm)
        bad["records"][0]["green_water"] = copy.deepcopy(waves["records"][0]["green_water"])
        self.assertTrue(has(check(bad), "without header.green_water"))

    def test_schema_definitions(self):
        schema = validate_cert_v3.load_json(os.path.join(_SCRIPTS, "cert_v3.schema.json"))
        record = dict(schema["$defs"]["green_water_record"], **{"$defs": schema["$defs"]})
        good = {"green_water": True, "first_time_s": 0.25, "first_point": "port_outer_bow",
                "submerged_point_substeps": 30, "longest_run_substeps": 7, "substeps": 600}
        self.assertFalse(list(Draft202012Validator(record).iter_errors(good)))
        for change in ({"extra": 1}, {"substeps": -1}, {"green_water": "yes"}):
            self.assertTrue(list(Draft202012Validator(record).iter_errors(dict(good, **change))), change)
        header = dict(schema["$defs"]["green_water_header"], **{"$defs": schema["$defs"]})
        block = self.doc()["header"]["green_water"]
        self.assertFalse(list(Draft202012Validator(header).iter_errors(block)))
        self.assertTrue(list(Draft202012Validator(header).iter_errors({k: v for k, v in block.items() if k != "cell"})))

    def test_writer_refuses_a_waves_env_without_the_header(self):
        env, cfg = T.journal_env(("waves=D2",))
        original = env.journal_certificate_header
        env.journal_certificate_header = lambda treatment=None: {k: v for k, v in original(treatment).items()
                                                                 if k != "green_water_header"}
        with self.assertRaises(ValueError) as caught:
            T.journal_run(env, cfg, ("waves=D2",))
        self.assertIn("green_water_header", str(caught.exception))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
