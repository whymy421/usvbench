"""G43 (owner decision, v2.6.2): the reader recomputes the appearing trigger pressure fields (closing_speed_mps =
surge x math.cos(bearing), realized_ttc_s) with a relative tolerance |stored - recomputed| <= 1e-12 x max(1,
|recomputed|); None compares exactly as before (CPU only; run: python -B test_g43_recompute_tolerance.py, or pytest).

Regression of G42: a certificate written on a Linux rental (glibc cos) failed validate_cert_v3 on Windows because the
recomputed closing speed differed in the last bit (Linux 2.3860822268248216 vs Windows 2.386082226824822). Covers:
  * the G42 pair itself and a one-ulp change of a REAL bridge stamp's closing speed / time to contact pass;
  * a real mismatch (relative 1e-6) and a None / number swap are still rejected;
  * the writer is unchanged: a fresh stamp stores exactly trigger_pressure(...) (bit-identical, no tolerance used).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_fixtures as F  # noqa: E402
from test_g35_speed_envelope import reseal, run_stamp  # noqa: E402

ST = F.stamp
G42_LINUX, G42_WINDOWS = 2.3860822268248216, 2.386082226824822   # G42 logs/cos_linux_vs_windows.txt


class RecomputeTolerance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stamp = run_stamp(1.0)
        cls.trigger = cls.stamp["payload"]["runtime"]["trigger"]

    def edited(self, field, value):
        bad = copy.deepcopy(self.stamp)
        bad["payload"]["runtime"]["trigger"][field] = value
        return reseal(bad)

    def test_writer_unchanged(self):
        t = self.trigger
        want = ST.trigger_pressure(t["gap_m"], t["bearing_rad"], t["pose"][3])
        self.assertEqual(t["closing_speed_mps"], want["closing_speed_mps"])
        self.assertEqual(t["realized_ttc_s"], want["realized_ttc_s"])
        self.assertIsNotNone(t["realized_ttc_s"])
        self.assertGreater(t["closing_speed_mps"], ST.CLOSING_SPEED_EPS_MPS)
        self.assertEqual(ST.read_appearing_stamp(self.stamp), self.stamp)

    def test_g42_pair_passes(self):
        self.assertNotEqual(G42_LINUX, G42_WINDOWS)                     # the exact v2.6.1 comparison failed
        self.assertEqual(math.nextafter(G42_LINUX, math.inf), G42_WINDOWS)  # they are one ulp apart
        self.assertTrue(ST.recomputes(G42_WINDOWS, G42_LINUX))
        self.assertTrue(ST.recomputes(G42_LINUX, G42_WINDOWS))

    def test_last_ulp_mismatch_passes(self):
        for field in ("closing_speed_mps", "realized_ttc_s"):
            for direction in (math.inf, -math.inf):
                value = math.nextafter(self.trigger[field], direction)
                self.assertNotEqual(value, self.trigger[field])
                envelope = self.edited(field, value)
                self.assertEqual(ST.read_appearing_stamp(envelope), envelope, (field, direction))
                self.assertEqual(ST.trigger_errors(envelope["payload"]["runtime"]), [], (field, direction))

    def test_real_mismatch_still_fails(self):
        for field in ("closing_speed_mps", "realized_ttc_s"):
            envelope = self.edited(field, self.trigger[field] * (1.0 + 1e-6))
            with self.assertRaises(ValueError) as caught:
                ST.read_appearing_stamp(envelope)
            self.assertIn("does not recompute", str(caught.exception))
            self.assertTrue(ST.trigger_errors(envelope["payload"]["runtime"]))

    def test_none_handled_as_before(self):
        self.assertTrue(ST.recomputes(None, None))
        self.assertFalse(ST.recomputes(None, 0.0))
        self.assertFalse(ST.recomputes(1e-13, None))
        self.assertFalse(ST.recomputes(float("nan"), float("nan")))     # non-finite: exact (nan != nan), as before
        envelope = self.edited("realized_ttc_s", None)                   # closing, but the stamp claims "not closing"
        with self.assertRaises(ValueError):
            ST.read_appearing_stamp(envelope)

    def test_tolerance_scale(self):
        self.assertEqual(ST.TRIGGER_RECOMPUTE_REL_TOL, 1e-12)
        self.assertTrue(ST.recomputes(0.0, 0.9e-12))                    # absolute 1e-12 below |b| = 1
        self.assertFalse(ST.recomputes(0.0, 2e-12))
        self.assertTrue(ST.recomputes(1000.0 * (1 + 0.9e-12), 1000.0))   # relative above |b| = 1
        self.assertFalse(ST.recomputes(1000.0 * (1 + 2e-12), 1000.0))


if __name__ == "__main__":
    unittest.main()
