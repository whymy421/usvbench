"""G28 item 2: the WAM-V green-water exclusion probe is the deck edge; the pontoon top is report-only (CPU).

Owner decision 2026-10-03 17:0x (NC 09 decision record, reply to decision pack 02, item 3).
* geometry: deck_edge_points(WAM-V) = the six deck-plate edge points (T25's report set), labels = the probe
  labels of green_water_rule.PROBE_SETS; report_point_sets(WAM-V) = the ten pontoon-top points
  ("wamv_pontoon_top"); BlueBoat unchanged (12 points, no report set);
* the formula is evaluated at the binding set (WAM-V deck edge: negligible at every preset);
* exclusion_block: with a vessel it stamps the probe; without one it is the v2.5 block byte for byte;
* real writer + validator: a WAM-V waves certificate binds the deck edge, stamps the probe, reports the pontoon
  set and validates; tampered probe / labels, a deck-edge header without probe, and wrong report sets are
  rejected; a v2.5-form block (no probe, pontoon-top binding points) stays valid; BlueBoat certificates
  validate with the blueboat_deck_edge probe;
* the schema declares the probe block.

Run: python -B scripts/tests_g28/test_g28_green_water_probe.py   (or pytest)
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue"), os.path.join(_SCRIPTS, "tests_t25")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

GW = G.shared("conditions.green_water")
RULE = G.shared("conditions.green_water_rule")
DATA = G.shared("vehicles_v2_data")
C = G.shared("conditions.config_v3")
P = G.shared("conditions.presets_v3")
HP = G.shared("hydro_profiles")
WAMV = "wamv_vrx_v2_opt1"
WAMVS = ("wamv_vrx_v2_opt1", "wamv_vrx_v2_opt1_bare")


def check(doc):
    import validate_cert_v3
    return validate_cert_v3.validate_certificate(doc, num_envs=2, episodes_per_env=1)


class Geometry(unittest.TestCase):
    def test_wamv_binding_points_are_the_deck_edge(self):
        for vessel in WAMVS:
            points = GW.deck_edge_points(vessel)
            self.assertEqual(points, GW.wamv_deck_edge_points(vessel))
            self.assertEqual([p.label for p in points], list(RULE.PROBE_SETS["wamv"]["labels"]))
            config = "bare" if vessel.endswith("_bare") else "competition"
            com_b = DATA.WAMV[config]["com_base_link_m"]
            for p in points:
                self.assertEqual(p.z_m, 1.25 + 0.05 - com_b[2])
            self.assertGreater(min(p.calm_freeboard_m for p in points), 1.18)
            sets = GW.report_point_sets(vessel)
            self.assertEqual([name for name, _ in sets], ["wamv_pontoon_top"])
            self.assertEqual(sets[0][1], GW.pontoon_top_points(vessel))
            self.assertEqual(len(sets[0][1]), 10)
            self.assertTrue(all(p.label.split("_pontoon_top_")[0] in ("port", "starboard") for p in sets[0][1]))

    def test_blueboat_is_unchanged(self):
        points = GW.deck_edge_points("blueboat_v2")
        self.assertEqual(len(points), 12)
        self.assertEqual([p.label for p in points], list(RULE.PROBE_SETS["blueboat"]["labels"]))
        self.assertEqual(GW.report_point_sets("blueboat_v2"), ())
        with self.assertRaises(ValueError):
            GW.pontoon_top_points("blueboat_v2")
        with self.assertRaises(ValueError):
            GW.wamv_deck_edge_points("blueboat_v2")

    def test_formula_uses_the_binding_set(self):
        geometry = GW.wetness_geometry(WAMV)
        self.assertEqual([p[0] for p in geometry.points], list(RULE.PROBE_SETS["wamv"]["labels"]))
        hydro = HP.profile_for_vehicle(WAMV).hydro
        for name in [n for n, e in P.JOURNAL_PRESETS.items() if e["suite"] == "waves"]:
            record = GW.formula_for(C.resolve_v3((name,)).waves, WAMV, hydro=hydro, presets=(name,))
            self.assertLess(record["probability"], 1e-30, name)
            if record["worst_point"] is not None:
                self.assertIn("_deck_edge_", record["worst_point"])


class ProbeBlock(unittest.TestCase):
    def test_probe_block_per_vessel(self):
        wamv = RULE.probe_block(WAMV)
        self.assertEqual((wamv["version"], wamv["set"]), ("usvbench.green_water.probe/2", "wamv_deck_edge"))
        self.assertEqual(wamv["report_only_sets"], ["wamv_pontoon_top"])
        self.assertEqual(wamv["points_key"], "header.green_water.detection.points")
        blue = RULE.probe_block("blueboat_v2")
        self.assertEqual((blue["set"], blue["report_only_sets"]), ("blueboat_deck_edge", []))
        with self.assertRaises(ValueError):
            RULE.probe_block("rov")

    def test_exclusion_block_without_vessel_is_the_v25_block(self):
        cell = {"excluded_green_water": False, "reasons": []}
        diagnostics = {"simulated_per_cycle_worst": 0.01, "simulated_per_cycle_worst_point": "p"}
        old = RULE.exclusion_block(0.001, diagnostics, cell)
        new = RULE.exclusion_block(0.001, diagnostics, cell, WAMV)
        self.assertNotIn("probe", old)
        self.assertEqual(new.pop("probe"), RULE.probe_block(WAMV))
        self.assertEqual(new, old)


class Certificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_t25_green_water_report_points import wave_document
        cls.wamv = wave_document(WAMV, ("waves=D4",), deadline_s=4.0)
        cls.blue = wave_document("blueboat_v2", ("waves=D3",), deadline_s=1.0)

    def test_wamv_certificate_binds_the_deck_edge_and_validates(self):
        doc = copy.deepcopy(self.wamv)
        self.assertEqual(check(doc), [])
        block = doc["header"]["green_water"]
        labels = [p["label"] for p in block["detection"]["points"]]
        self.assertEqual(labels, list(RULE.PROBE_SETS["wamv"]["labels"]))
        self.assertEqual(block["exclusion"]["probe"], RULE.probe_block(WAMV))
        self.assertEqual(set(block["report_points"]["sets"]), {"wamv_pontoon_top"})
        self.assertEqual(len(block["report_points"]["sets"]["wamv_pontoon_top"]["points"]), 10)
        self.assertIn("pontoon", block["report_points"]["sets"]["wamv_pontoon_top"]["source"])
        self.assertEqual(set(block["report"]["sets"]), {"wamv_pontoon_top"})
        for r in doc["records"]:
            self.assertIn(r["green_water"]["first_point"] or "_deck_edge_", ("_deck_edge_",) + tuple(labels))
            self.assertEqual(len(r["green_water_diagnostics"]["freeboard_upcrossings"]), 6)
            self.assertEqual(len(r["green_water_report"]["wamv_pontoon_top"]["diagnostics"]["freeboard_upcrossings"]),
                             10)
        point = block["exclusion"]["simulated_per_cycle_worst_point"]
        self.assertTrue(point is None or point in labels)

    def test_blueboat_certificate_carries_the_blueboat_probe(self):
        doc = copy.deepcopy(self.blue)
        self.assertEqual(check(doc), [])
        block = doc["header"]["green_water"]
        self.assertEqual(block["exclusion"]["probe"]["set"], "blueboat_deck_edge")
        self.assertNotIn("report_points", block)

    def test_validator_rejects_probe_tampering(self):
        doc = copy.deepcopy(self.wamv)
        doc["header"]["green_water"]["exclusion"]["probe"]["set"] = "wamv_pontoon_top"      # schema enum
        self.assertTrue(any("exclusion/probe/set" in e for e in check(doc)))
        doc = copy.deepcopy(self.wamv)
        doc["header"]["green_water"]["exclusion"]["probe"]["set"] = "blueboat_deck_edge"    # semantic check
        self.assertTrue(any("exclusion.probe" in e for e in check(doc)))
        doc = copy.deepcopy(self.wamv)
        del doc["header"]["green_water"]["exclusion"]["probe"]
        self.assertTrue(any("must stamp header.green_water.exclusion.probe" in e for e in check(doc)))
        doc = copy.deepcopy(self.wamv)
        doc["header"]["green_water"]["detection"]["points"][0]["label"] = "port_pontoon_top_bow"
        self.assertTrue(any("not the declared exclusion probe" in e for e in check(doc)))
        doc = copy.deepcopy(self.wamv)
        doc["header"]["green_water"]["exclusion"]["probe"]["labels"] = list(RULE._WAMV_PONTOON_LABELS)
        self.assertTrue(any("exclusion.probe" in e for e in check(doc)))

    def test_v25_form_block_stays_valid(self):
        # a v2.5 block: pontoon-top binding labels and no probe (rule/2 recomputed without the probe stamp)
        block = {"vessel": WAMV, "formula": {"probability": 0.0179},
                 "detection": {"points": [{"label": x} for x in RULE._WAMV_PONTOON_LABELS]},
                 "cell": {"excluded_green_water": True, "reasons": ["simulated_green_water_share_exceeds_0.05"]},
                 "diagnostics": {"simulated_per_cycle_worst": 0.06, "simulated_per_cycle_worst_point":
                                 "port_pontoon_top_bow"}}
        block["exclusion"] = RULE.exclusion_block(0.0179, block["diagnostics"], block["cell"])
        self.assertEqual(RULE.exclusion_errors(block), [])
        block["exclusion"]["excluded_green_water"] = False
        self.assertTrue(RULE.exclusion_errors(block))

    def test_schema_declares_the_probe(self):
        with open(os.path.join(_SCRIPTS, "cert_v3.schema.json"), encoding="utf-8") as handle:
            schema = json.load(handle)
        text = json.dumps(schema)
        self.assertIn("usvbench.green_water.probe/2", text)
        found = []

        def walk(node):
            if isinstance(node, dict):
                if "probe" in node.get("properties", {}) and "rule_1" in node.get("properties", {}):
                    found.append(node["properties"]["probe"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(schema)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["properties"]["set"]["enum"], ["blueboat_deck_edge", "wamv_deck_edge"])


if __name__ == "__main__":
    unittest.main()
