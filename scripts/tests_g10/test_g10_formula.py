"""G10 item 1: the D9 wave-following wetness formula port (CPU, numpy only for the formula).

* the port reproduces D9's PUBLISHED per-level values when given D9's own
  provisional vessel inputs (fixture values copied from D9
  work/results/ladder_levels.json, "p_wet_following"; DOSE_DESIGN.md section 5);
* the closed-form waterplane integrals equal D9's Gauss-Legendre quadrature;
* the compass-8 headings equal D9's 0..180 deg headings for both registry
  point sets (port/starboard symmetry);
* pinned values: every wave preset of presets_v3 (ladder A, ladder B, the
  combinations' wave components, the 0.75 m supplement) for both journal
  vessels with the FINAL registry inputs;
* resolved_band equals the band the condition layer realises;
* calm doses give exactly 0; stamp fields (version, criterion, inputs).

Run: python -B scripts/tests_g10/test_g10_formula.py   (or pytest)
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tests_journal_glue"))

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

F = G.shared("conditions.green_water_formula")
GW = G.shared("conditions.green_water")
P = G.shared("conditions.presets_v3")
C = G.shared("conditions.config_v3")
HP = G.shared("hydro_profiles")
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")

# D9 results/ladder_levels.json "p_wet_following" (seed-42 C2 generator realisation, float32 amplitudes,
# band [0.2, 4.0] fp, 1024 components, headings 0..180 deg); D9 DOSE_DESIGN.md section 5 tables.
D9_PUBLISHED = {
    "wave.A.D1": (0.125, 4.0, 3.1853139461589186e-298, 4.5979296413918865e-114),
    "wave.A.D2": (0.25, 4.0, 4.224623988712564e-75, 4.6306353233581337e-29),
    "wave.A.D3": (0.5, 4.0, 2.549453030379751e-19, 8.24916793695515e-08),
    "wave.A.D4": (1.0, 4.0, 2.247044917110109e-05, 0.016947378782668086),
    "wave.B.D1": (0.21712025004971644, 1.9542790629547644, 2.7726416768037824e-05, 1.9788647720213393e-05),
    "wave.B.D2": (0.4527645863384214, 2.4967628052770894, 0.002746066513627552, 0.013389852472787272),
    "wave.B.D3": (0.7058400251154971, 2.8950444064492844, 0.013343261652219999, 0.06311338312397764),
    "wave.B.D4": (0.9754131187827947, 3.2246506152866483, 0.028071419232442517, 0.12942447048864938),
}

# G10 pinned values with the FINAL registry inputs: (probability, worst heading, worst point, band/fp).
PINNED = {
    ("journal.waves.D0", "blueboat_v2"): (0.0, None, None, None),
    ("journal.waves.D1", "blueboat_v2"): (1.50035775347e-204, 45.0, "port_outer_bow", [0.2, 8.29]),
    ("journal.waves.D2", "blueboat_v2"): (1.10674790035e-51, 45.0, "port_outer_bow", [0.2, 8.29]),
    ("journal.waves.D3", "blueboat_v2"): (1.82394684115e-13, 45.0, "port_outer_bow", [0.2, 8.29]),
    ("journal.waves.D4", "blueboat_v2"): (0.000653511213797, 45.0, "port_outer_bow", [0.2, 8.29]),
    ("journal.waves.supp075", "blueboat_v2"): (2.17885151125e-06, 45.0, "port_outer_bow", [0.2, 8.29]),
    ("journal.waves.B.D0", "blueboat_v2"): (0.0, None, None, None),
    ("journal.waves.B.D1", "blueboat_v2"): (2.16849732898e-05, 45.0, "port_outer_bow", [0.2, 4.05]),
    ("journal.waves.B.D2", "blueboat_v2"): (0.00268927600916, 45.0, "port_outer_bow", [0.2, 5.18]),
    ("journal.waves.B.D3", "blueboat_v2"): (0.0145298115738, 45.0, "port_outer_bow", [0.2, 6.0]),
    ("journal.waves.B.D4", "blueboat_v2"): (0.0353751422828, 45.0, "port_outer_bow", [0.2, 6.68]),
    # G28: the WAM-V binding probe is the deck edge (owner decision 2026-10-03 17:0x item 3); the deck plate sits
    # about 0.9 m above the pontoon tops, so the wave-following formula is negligible at every preset (D1 / D2
    # underflow to exactly 0.0; D3 is a subnormal and is checked as < 1e-300 only)
    ("journal.waves.D0", "wamv_vrx_v2_opt1"): (0.0, None, None, None),
    ("journal.waves.D1", "wamv_vrx_v2_opt1"): (0.0, 0.0, "port_deck_edge_aft", [0.2, 4.88]),
    ("journal.waves.D2", "wamv_vrx_v2_opt1"): (0.0, 0.0, "port_deck_edge_aft", [0.2, 4.88]),
    ("journal.waves.D3", "wamv_vrx_v2_opt1"): (3.677345305e-315, 0.0, "port_deck_edge_mid", [0.2, 4.88]),
    ("journal.waves.D4", "wamv_vrx_v2_opt1"): (2.46254175467e-79, 0.0, "port_deck_edge_mid", [0.2, 4.88]),
    ("journal.waves.supp075", "wamv_vrx_v2_opt1"): (1.78381013754e-140, 0.0, "port_deck_edge_mid", [0.2, 4.88]),
    ("journal.waves.B.D0", "wamv_vrx_v2_opt1"): (0.0, None, None, None),
    ("journal.waves.B.D1", "wamv_vrx_v2_opt1"): (1.08374235689e-156, 0.0, "port_deck_edge_mid", [0.2, 4.0]),
    ("journal.waves.B.D2", "wamv_vrx_v2_opt1"): (8.01468469336e-75, 0.0, "port_deck_edge_mid", [0.2, 4.0]),
    ("journal.waves.B.D3", "wamv_vrx_v2_opt1"): (1.45046054327e-50, 0.0, "port_deck_edge_mid", [0.2, 4.0]),
    ("journal.waves.B.D4", "wamv_vrx_v2_opt1"): (8.89964672815e-39, 0.0, "port_deck_edge_mid", [0.2, 4.0]),
}

# The v2.4 / v2.5 WAM-V pinned values (pontoon top line, report-only since G28): the same formula code at the
# pontoon_top_points set must still give them (regression of the formula itself).
PINNED_WAMV_PONTOON_TOP = {
    "journal.waves.D1": (1.50929755102e-112, 45.0, "port_pontoon_top_bow", [0.2, 4.88]),
    "journal.waves.D2": (1.10839285322e-28, 45.0, "port_pontoon_top_bow", [0.2, 4.88]),
    "journal.waves.D3": (1.02606158728e-07, 45.0, "port_pontoon_top_bow", [0.2, 4.88]),
    "journal.waves.D4": (0.0178975406424, 45.0, "port_pontoon_top_bow", [0.2, 4.88]),
    "journal.waves.supp075": (0.000783167861877, 45.0, "port_pontoon_top_bow", [0.2, 4.88]),
    "journal.waves.B.D1": (2.07746153529e-05, 45.0, "port_pontoon_top_bow", [0.2, 4.0]),
    "journal.waves.B.D2": (0.0136505848501, 45.0, "port_pontoon_top_bow", [0.2, 4.0]),
    "journal.waves.B.D3": (0.0638970551635, 45.0, "port_pontoon_top_bow", [0.2, 4.0]),
    "journal.waves.B.D4": (0.130610514881, 45.0, "port_pontoon_top_bow", [0.2, 4.0]),
}


def d9_geometries():
    """D9's provisional VesselScreenSpec inputs (vessel_envelope.py:95-162), as WetnessGeometry."""
    ring = [(sx * 0.6, sy * 0.45) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if (sx, sy) != (0, 0)]
    blueboat = F.WetnessGeometry("d9_blueboat", (F.Strip(0.0, 0.0, 1.2, 0.9),),
                                 tuple(("p%d" % i, x, y, 0.15) for i, (x, y) in enumerate(ring)))
    chord = 2.0 * math.sqrt(2.0 * 0.213 * 0.113 - 0.113 ** 2)
    pts = [(sx * 2.45, sy * 1.03) for sx in (-1, 0, 1) for sy in (-1, 1)]
    wamv = F.WetnessGeometry("d9_wamv", (F.Strip(0.0, 1.03, 4.9, chord), F.Strip(0.0, -1.03, 4.9, chord)),
                             tuple(("p%d" % i, x, y, 0.29) for i, (x, y) in enumerate(pts)))
    return blueboat, wamv


def log_rel(a, b):
    return abs(math.log(a) - math.log(b)) / abs(math.log(b))


class FormulaPort(unittest.TestCase):
    def test_reproduces_d9_published_values_with_d9_inputs(self):
        blueboat, wamv = d9_geometries()
        for name, (hs, tp, p_bb, p_wamv) in D9_PUBLISHED.items():
            f, e = F.midpoint_spectrum(hs, tp, 3.3, (0.2, 4.0), 1024)
            for geometry, published in ((blueboat, p_bb), (wamv, p_wamv)):
                value = F.wetness(geometry, f, e, math.pi / 4, headings_deg=F.D9_HEADINGS_DEG)["probability"]
                # D9 used float32 generator amplitudes; the port uses the float64 mirror: log-relative 1e-6
                self.assertLess(log_rel(value, published), 1e-6, (name, geometry.vessel, value, published))

    def test_closed_form_equals_d9_quadrature(self):
        geometries = list(d9_geometries()) + [GW.wetness_geometry(v) for v in VESSELS]
        for geometry in geometries:
            for hs, tp, band in ((1.0, 4.0, (0.2, 8.29)),):
                f, e = F.midpoint_spectrum(hs, tp, 3.3, band, 1024)
                for heading in (0.0, 90.0):
                    a = F.relative_motion_m0(geometry, f, e, heading, math.pi / 4)
                    # D9's 32 x 8 nodes are exact to rounding only up to about kL/2 = 8; the raised BlueBoat
                    # band (8.29 fp, kL/2 = 10.4) needs more nodes for the reference (D9's own band: 4 fp)
                    b = F.relative_motion_quadrature(geometry, f, e, heading, math.pi / 4, nx=64, ny=32)
                    for x, y in zip(a, b):
                        self.assertTrue(math.isclose(x, y, rel_tol=1e-9), (geometry.vessel, heading, x, y))

    def test_d9_quadrature_nodes_at_d9_band(self):
        geometries = list(d9_geometries()) + [GW.wetness_geometry(v) for v in VESSELS]
        f, e = F.midpoint_spectrum(1.0, 4.0, 3.3, (0.2, 4.0), 1024)
        for geometry in geometries:
            a = F.relative_motion_m0(geometry, f, e, 45.0, math.pi / 4)
            b = F.relative_motion_quadrature(geometry, f, e, 45.0, math.pi / 4)      # D9's 32 x 8 nodes
            for x, y in zip(a, b):
                self.assertTrue(math.isclose(x, y, rel_tol=1e-9), (geometry.vessel, x, y))

    def test_compass8_equals_d9_headings_for_symmetric_point_sets(self):
        for vessel in VESSELS:
            geometry = GW.wetness_geometry(vessel)
            f, e = F.midpoint_spectrum(1.0, 4.0, 3.3, (0.2, 6.0), 1024)
            eight = F.wetness(geometry, f, e, math.pi / 4)["probability"]
            five = F.wetness(geometry, f, e, math.pi / 4, headings_deg=F.D9_HEADINGS_DEG)["probability"]
            self.assertTrue(math.isclose(eight, five, rel_tol=1e-12), (vessel, eight, five))

    def test_pinned_values_every_wave_preset_and_vessel(self):
        names = [n for n, e in P.JOURNAL_PRESETS.items() if e["suite"] == "waves"]
        self.assertEqual(len(names), 11)          # A D0-D4, B D0-D4, supp075
        self.assertEqual({k[0] for k in PINNED}, set(names))
        for vessel in VESSELS:
            hydro = HP.profile_for_vehicle(vessel).hydro
            for name in names:
                w = C.resolve_v3((name,)).waves
                record = GW.formula_for(w, vessel, hydro=hydro, presets=(name,))
                prob, heading, point, band = PINNED[(name, vessel)]
                self.assertEqual(record["version"], F.FORMULA_VERSION)
                self.assertEqual(record["criterion"], "NORDFORSK per-cycle wetness, limit 0.05")
                self.assertEqual(record["limit"], 0.05)
                self.assertEqual(record["vessel"], vessel)
                self.assertEqual(record["preset"], name)
                self.assertEqual(record["preset_role"], "SUPPLEMENT" if name.endswith("supp075") else "MAIN")
                if prob == 0.0 and heading is not None:          # G28: WAM-V deck edge, underflow (not calm)
                    self.assertEqual(record["probability"], 0.0)
                    self.assertFalse(record["inputs"]["calm"])
                    self.assertEqual((record["worst_heading_deg"], record["worst_point"]), (heading, point))
                    continue
                if 0.0 < prob < 1e-300:                          # G28: subnormal, platform-sensitive digits
                    self.assertTrue(0.0 <= record["probability"] < 1e-300, (name, vessel, record["probability"]))
                    continue
                if prob == 0.0:
                    self.assertEqual(record["probability"], 0.0)
                    self.assertTrue(record["inputs"]["calm"])
                    continue
                self.assertTrue(math.isclose(record["probability"], prob, rel_tol=1e-9), (name, vessel, record))
                self.assertEqual((record["worst_heading_deg"], record["worst_point"]), (heading, point))
                self.assertEqual(record["inputs"]["band_over_fp"], band)
                self.assertEqual(record["inputs"]["hs_m"], w.hs_m)
                self.assertEqual(record["inputs"]["tp_s"], w.tp_s)
                self.assertEqual(record["inputs"]["geometry"]["vessel"], vessel)

    def test_wamv_pontoon_top_values_unchanged_as_report_only_set(self):
        # G28: the formula at the v2.5 binding set (pontoon tops) reproduces the v2.5 pins exactly
        hydro = HP.profile_for_vehicle("wamv_vrx_v2_opt1").hydro
        pontoon = GW.pontoon_top_points("wamv_vrx_v2_opt1")
        for name, (prob, heading, point, band) in PINNED_WAMV_PONTOON_TOP.items():
            w = C.resolve_v3((name,)).waves
            record = GW.formula_for(w, "wamv_vrx_v2_opt1", hydro=hydro, presets=(name,), points=pontoon)
            self.assertTrue(math.isclose(record["probability"], prob, rel_tol=1e-9), (name, record["probability"]))
            self.assertEqual((record["worst_heading_deg"], record["worst_point"]), (heading, point))
            self.assertEqual(record["inputs"]["band_over_fp"], band)

    def test_combination_wave_components_share_their_wave_preset_value(self):
        for combo, components in P.COMBINATIONS.items():
            wave = [c for c in components if P.JOURNAL_PRESETS[c]["suite"] == "waves"]
            self.assertEqual(len(wave), 1, combo)
            for vessel in VESSELS:
                hydro = HP.profile_for_vehicle(vessel).hydro
                cfg = C.resolve_v3((combo,))
                record = GW.formula_for(cfg.waves, vessel, hydro=hydro, presets=cfg.presets)
                self.assertEqual(record["preset"], wave[0])
                self.assertTrue(math.isclose(record["probability"], PINNED[(wave[0], vessel)][0], rel_tol=1e-9))

    def test_formula_verdict_part_a(self):
        # G28: with the WAM-V deck-edge probe no preset exceeds 0.05 by the formula (v2.5: WAM-V ladder B3/B4 did at
        # the pontoon tops, 0.064 / 0.131, still visible in the report-only pins)
        over = sorted(k for k, v in PINNED.items() if v[0] > 0.05)
        self.assertEqual(over, [])
        self.assertEqual(sorted(k for k, v in PINNED_WAMV_PONTOON_TOP.items() if v[0] > 0.05),
                         ["journal.waves.B.D3", "journal.waves.B.D4"])

    def test_resolved_band_matches_the_condition_layer(self):
        for vessel in VESSELS:
            hydro = HP.profile_for_vehicle(vessel).hydro
            for name in ("journal.waves.D2", "journal.waves.D4", "journal.waves.supp075", "journal.waves.B.D1",
                         "journal.waves.B.D4"):
                cfg = G.journal_cfg(vessel, tokens=[name])
                env = G.FakeJournalEnv(cfg, 2)
                resolved = env._conditions.context["resolved"]["waves"]
                lo, hi, n = F.resolved_band(cfg.conditions.waves, hydro)
                self.assertEqual((lo, hi, n), (resolved["band_min_over_fp"], resolved["band_max_over_fp"],
                                               resolved["n_components"]), (vessel, name))
                header = env._green_water.header()
                self.assertEqual(header["formula"]["inputs"]["band_over_fp"], [lo, hi])
                self.assertEqual(header["formula"]["inputs"]["n_components"], n)
                self.assertTrue(math.isclose(header["formula"]["probability"], PINNED[(name, vessel)][0],
                                             rel_tol=1e-9))

    def test_calm_doses_are_exactly_zero(self):
        for vessel in VESSELS:
            geometry = GW.wetness_geometry(vessel)
            for waves in (C.WavesV3(), C.WavesV3(enabled=True, hs_m=0.0)):
                record = F.formula_record(waves, geometry)
                self.assertEqual(record["probability"], 0.0)
                self.assertIsNone(record["worst_point"])

    def test_probability_is_monotone_in_hs_on_ladder_a(self):
        for vessel in VESSELS:
            values = [PINNED[("journal.waves.%s" % lvl, vessel)][0] for lvl in ("D1", "D2", "D3", "supp075", "D4")]
            self.assertEqual(values, sorted(values), vessel)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
