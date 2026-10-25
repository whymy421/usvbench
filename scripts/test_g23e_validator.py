"""G23e: validate_cert_v3 support of the station-keeping approach design (tiers not applicable by design).

Run: python -B scripts/test_g23e_validator.py

* dynamic_header_errors: a certificate of a tier listed in dynamic_hazards_header.not_applicable_doses is an
  error; malformed lists and non-zero rows of an excluded tier are errors; a header without the field and an
  applicable tier pass;
* the schema declares the two header fields.
"""
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validate_cert_v3 as V  # noqa: E402

ROWS = [[192, 192, 0], [0, 0, 0], [192, 192, 0], [192, 192, 0], [192, 192, 0]]


def document(dose, **header):
    block = {"pool_manifest_sha256": "a" * 64, "per_dose_counts": [list(r) for r in ROWS],
             "suite": {"dose": dose}, "not_applicable_doses": [1],
             "not_applicable_policy": "G23e: not applicable by design"}
    block.update(header)
    return {"header": {"dynamic_hazards_header": block}, "records": []}


class HeaderTests(unittest.TestCase):
    def test_applicable_tier_passes(self):
        for dose in (0, 2, 3, 4):
            self.assertEqual(V.dynamic_header_errors(document(dose)), [])

    def test_certificate_of_an_excluded_tier_is_an_error(self):
        errors = V.dynamic_header_errors(document(1))
        self.assertTrue(any("not applicable by design" in e for e in errors), errors)

    def test_malformed_lists_and_rows(self):
        for bad in ([0], [2, 1], [1, 1], "1"):
            errors = V.dynamic_header_errors(document(2, not_applicable_doses=bad))
            self.assertTrue(errors, bad)
        rows = [list(r) for r in ROWS]
        rows[1] = [192, 192, 0]
        self.assertTrue(V.dynamic_header_errors(document(2, per_dose_counts=rows)))

    def test_header_without_the_field(self):
        doc = document(1)
        del doc["header"]["dynamic_hazards_header"]["not_applicable_doses"]
        doc["header"]["dynamic_hazards_header"]["per_dose_counts"] = [[192, 192, 0]] * 5
        self.assertEqual(V.dynamic_header_errors(doc), [])

    def test_schema_declares_the_fields(self):
        schema = json.loads((HERE / "cert_v3.schema.json").read_text(encoding="utf-8"))
        props = schema["properties"]["header"]["properties"]["dynamic_hazards_header"]["properties"]
        self.assertEqual(props["not_applicable_doses"]["items"], {"type": "integer", "minimum": 1, "maximum": 4})
        self.assertIn("not_applicable_policy", props)


if __name__ == "__main__":
    unittest.main(verbosity=2)
