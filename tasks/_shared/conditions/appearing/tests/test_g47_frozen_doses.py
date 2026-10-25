"""G47 (v2.6.4, owner decision 2026-10-14): the appearing-distance doses are frozen in ladder M (CPU only; run:
python -B test_g47_frozen_doses.py, or pytest).

Owner record (NC 09_decision log 10-12 00:xx, 10-14 00:xx, 10-14 06:xx): BlueBoat D1..D4 = 7.5 / 3.5 / 1.6 / 0.76 m,
WAM-V = 15 / 6.6 / 2.9 / 1.25 m, from the pre-registered calibration-sweep selection on the c3 sweep (G43 c3sweep1,
logs/sweep_select_c3sweep1_LINUX_20261013T160845Z.txt: D4 fallback = the calibration rung for both vessels, D1 by the
owner under the rule's 'order' clause, D2 / D3 geometric rounded to 0.1 m). Covers:
  * the table, the dose text, the k equivalents and the supplementary D5 (= D4 now);
  * the derivation: D2 = D1 (D4/D1)^(1/3), D3 = D1 (D4/D1)^(2/3), rounded to 0.1 m (the rule's D2_D3 clause);
  * the validity rule accepts D4 = the rung and still refuses a ladder below the rung or with a repeated main gap;
  * start gaps (largest offered gap; calibration tasks keep the sweep cap) and the station-keeping BlueBoat D1 stays
    not applicable by design (7.5 m > the 3.9-4.2 m start distance; D2 = 3.5 m is below it);
  * the suite version is v7m (pools admitted under v6m are refused at load).
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_fixtures as F  # noqa: E402

LAD, C = F.ladder, F.config
FROZEN = {F.BB: (7.5, 3.5, 1.6, 0.76), F.WAMV: (15.0, 6.6, 2.9, 1.25)}
RUNG = {F.BB: 0.76, F.WAMV: 1.25}


class FrozenDoses(unittest.TestCase):
    def test_table_is_the_owner_frozen_doses(self):
        self.assertEqual(LAD.ACTIVE_LADDER, "M")
        for vessel, main in FROZEN.items():
            self.assertEqual(LAD.LADDER_OPTIONS["M"]["main_m"][vessel], main)
            self.assertEqual(LAD.GAPS_M[vessel], (None,) + main + (RUNG[vessel],))
            t_avoid = LAD.T_AVOID_S[vessel]
            for dose, gap in enumerate(main, start=1):
                self.assertEqual(C.trigger_gap_m(dose, t_avoid, vessel), gap)
                self.assertAlmostEqual(C.k_equivalent(gap, t_avoid), gap / t_avoid, places=12)
            self.assertEqual(C.trigger_gap_m(5, t_avoid, vessel), RUNG[vessel])
            self.assertEqual(C.smallest_main_trigger_gap_m(t_avoid, vessel), RUNG[vessel])
            self.assertEqual(C.largest_trigger_gap_m(t_avoid, vessel), main[0])
        self.assertEqual((C.MAIN_DOSES, C.SUPPLEMENTARY_DOSES, C.MAX_DOSE), ((1, 2, 3, 4), (5,), 5))
        text = LAD.dose_text()
        self.assertIn("blueboat_v2 7.5 / 3.5 / 1.6 / 0.76 m", text)
        self.assertIn("wamv_vrx_v2_opt1 15 / 6.6 / 2.9 / 1.25 m", text)

    def test_geometric_middle_rungs_follow_the_preregistered_rule(self):
        for vessel, (d1, d2, d3, d4) in FROZEN.items():
            self.assertEqual(d4, LAD.CALIBRATION_RUNG_M[vessel])            # the rule's D4 fallback
            self.assertEqual(round(d1 * (d4 / d1) ** (1.0 / 3.0), 1), d2)
            self.assertEqual(round(d1 * (d4 / d1) ** (2.0 / 3.0), 1), d3)
            self.assertIn(d1, LAD.CALIBRATION_SWEEP_M[vessel])              # D1 = the sweep cap (owner)
            self.assertEqual(d1, max(LAD.CALIBRATION_SWEEP_M[vessel]))

    def test_validity_rule_accepts_d4_at_the_rung_only(self):
        spec = LAD.LADDER_OPTIONS["M"]
        good = LAD.ladder_constants("M")
        self.assertEqual(good["gaps_m"][F.BB][4], good["gaps_m"][F.BB][5])
        saved = spec["main_m"]
        try:
            for bad in ({F.BB: (7.5, 3.5, 1.6, 0.75), F.WAMV: FROZEN[F.WAMV]},       # below the rung
                        {F.BB: (7.5, 3.5, 0.76, 0.76), F.WAMV: FROZEN[F.WAMV]},      # repeated main gap
                        {F.BB: (3.5, 7.5, 1.6, 0.76), F.WAMV: FROZEN[F.WAMV]},       # not decreasing
                        {F.BB: FROZEN[F.BB], F.WAMV: (15.0, 6.6, 2.9, 1.2)}):         # WAM-V below its rung
                spec["main_m"] = bad
                with self.assertRaises(ValueError):
                    LAD.ladder_constants("M")
        finally:
            spec["main_m"] = saved
        self.assertEqual(LAD.ladder_constants("M"), good)

    def test_start_gaps_and_not_applicable_doses(self):
        self.assertEqual(LAD.not_applicable_doses("station_keeping", F.BB), (1,))
        self.assertEqual(LAD.not_applicable_doses("station_keeping", F.WAMV), ())
        self.assertIn("7.5 m", LAD.not_applicable_reason("station_keeping", F.BB, 1))
        self.assertEqual(LAD.start_gap_m("station_keeping", F.BB), 3.5)
        # G48/G49 (owner 2026-10-14): BlueBoat docking D1 is not applicable by design as well (D1 = 7.5 m leaves
        # no route window after the start rule); the start gap becomes D2 = 3.5 m. WAM-V docking keeps D1.
        self.assertEqual(LAD.not_applicable_doses("docking", F.BB), (1,))
        self.assertEqual(LAD.not_applicable_doses("docking", F.WAMV), ())
        self.assertIn("route_window_empty_after_start_rule", LAD.not_applicable_reason("docking", F.BB, 1))
        self.assertEqual(LAD.start_gap_m("docking", F.BB), 3.5)
        self.assertEqual(LAD.start_gap_m("docking", F.WAMV), 15.0)
        self.assertEqual(sorted(LAD.NOT_APPLICABLE_DOSES), [("docking", F.BB), ("station_keeping", F.BB)])
        for task in C.PLACEMENT_TASKS:
            for vessel, main in FROZEN.items():
                offered = LAD.offered_doses(task, vessel)
                want = max(main[d - 1] if d <= 4 else RUNG[vessel] for d in offered)
                if task in LAD.CALIBRATION_TASKS:
                    want = max(want, max(LAD.CALIBRATION_SWEEP_M[vessel]))
                self.assertEqual(LAD.start_gap_m(task, vessel), want, (task, vessel))

    def test_suite_version_names_the_frozen_table(self):
        self.assertEqual(LAD.SUITE_VERSION, "g3_appearing_gap_v7m")
        self.assertEqual(C.APPEARING_SUITE_VERSION, "g3_appearing_gap_v7m")
        self.assertEqual(LAD.LADDER_OPTIONS["A"]["suite_version"], "g3_appearing_gap_v2")    # reference options kept


if __name__ == "__main__":
    unittest.main(verbosity=2)
