"""T25 (owner decision T1.35d): report-only green-water point set at the WAM-V deck edge (CPU).

* geometry: the WAM-V deck-edge set comes from the VRX deck plate (top_base box, top face at base_link
  z 1.30 m) and the registry COM / calm COM height; BlueBoat has no second set because its declared points
  already sit at its registry deck freeboard;
* the monitor: default points unchanged; a report monitor counts its own runs, events and crossings;
* read-only: a WAM-V waves run with the report set and the same run without it give bit-identical applied
  wrenches, observations, dones, RNG states, rule/1 records, G18 diagnostics and headers (only the new keys
  differ); BlueBoat runs carry nothing new;
* real writer + validator: a WAM-V waves certificate carries records[].green_water_report and the
  non-binding header.green_water.report and validates; the rule/1 cell and the G18 diagnostics are those of
  the run without the report set; tampered, partial or orphaned report data are rejected.

Run: python -B scripts/tests_t25/test_t25_green_water_report_points.py   (or pytest)
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import itertools
import json
import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

GW = G.shared("conditions.green_water")
RULE = G.shared("conditions.green_water_rule")
DATA = G.shared("vehicles_v2_data")
DT = 1.0 / 120.0
WAMV = "wamv_vrx_v2_opt1"
ACTIONS = torch.tensor([[0.6, 0.4], [0.3, -0.2]])
REPORT_SET = "wamv_pontoon_top"      # G28: the WAM-V report-only set (T25: "wamv_deck_edge", now the binding set)


def drive(tokens, vessel=WAMV, steps=90, deadline_s=0.5, report=True):
    torch.manual_seed(77)
    np.random.seed(77)
    cfg = G.journal_cfg(vessel, tokens=tokens)
    context = nullcontext() if report else patch.object(GW, "report_monitors_for", lambda main: {})
    with context:
        env = G.FakeJournalEnv(cfg, 2, deadline_s=deadline_s)
    trace = {"obs": [env.reset()], "dones": [], "records": [], "headers": []}
    for _ in range(steps):
        obs, terminated, truncated = env.step(ACTIONS)
        trace["obs"].append(obs)
        trace["dones"].append((terminated.clone(), truncated.clone()))
        for i in (terminated | truncated).nonzero().flatten().tolist():
            trace["records"].append(json.loads(json.dumps(env.journal_episode_record(i), sort_keys=True)))
    trace["headers"].append(json.loads(json.dumps(env.journal_certificate_header(), sort_keys=True)))
    trace["applied"] = env.robot.applied
    trace["rng"] = (torch.get_rng_state(), np.random.get_state()[1].copy())
    return env, trace


def wave_document(vessel, tokens, deadline_s=0.5, report=True):
    """2 x 1 journal certificate from the REAL writer over the REAL mixin (as test_g10b_combo_tokens builds it)."""
    import cert_v3
    import test_journal_evaluators as T
    import validate_cert_v3  # noqa: F401  (path set up by T)
    cfg = G.journal_cfg(vessel, tokens=list(tokens), seed=42)
    pool = T.PoolView(G.d7_pool(vessel, 2, seed=42), 0)
    context = nullcontext() if report else patch.object(GW, "report_monitors_for", lambda main: {})
    with context:
        env = G.FakeJournalEnv(cfg, 2, deadline_s=deadline_s, pool=pool)
    env.dynamic_hazard_header["grid"] = dict(T.GRID)
    env.journal_d7_pool_manifest_sha256 = T.MANIFEST_SHA
    provenance = {k: "a" * 64 for k in ("tasks_sha256", "scripts_sha256", "evaluator_sha256")}
    provenance.update(tasks_root="t/tasks", scripts_root="t/scripts", environment_source="t/tasks/x/env.py",
                      evaluator_path="t/scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
    provenance["tree_sha256"] = cert_v3.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {k: "test" for k in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                   "isaac_lab_version", "physx_version")}
    ticks = itertools.count()
    task = T.TASK_ID if vessel == "blueboat_v2" else T.TASK_ID.replace("BlueBoat", "WAMVVRX")
    run = cert_v3.CertificateRun(base=env, env_cfg=cfg, evaluator=T.__file__, task=task, controller="test",
                                 controller_config={}, checkpoint=None, eval_seed=42, level=0, episodes_requested=2,
                                 overrides={}, argv=["eval.py", "--suite", *tokens], cli_args={"suite": list(tokens)},
                                 provenance=provenance, runtime=runtime, synchronize=lambda: None,
                                 clock=lambda: next(ticks) / 1000.0)
    env.reset()
    run.after_reset()
    records, done = [], set()
    while len(done) < 2:
        with run.time_controller():
            action = torch.tensor([[0.6, 0.1], [0.5, -0.1]])
        with run.time_environment():
            _, terminated, truncated = env.step(action)
        run.after_step()
        for i in (terminated | truncated).nonzero().flatten().tolist():
            if i in done:
                run.discard_episode(i)
                continue
            records.append(run.finish_episode(i, dict(env=i, ep=0, success=False, terminated=bool(terminated[i]),
                                                      truncated=bool(truncated[i]), tts_s=None, path_length_m=1.0)))
            done.add(i)
    return run.build({"records": records})


def check(doc):
    import validate_cert_v3
    return validate_cert_v3.validate_certificate(doc, num_envs=2, episodes_per_env=1)


class Geometry(unittest.TestCase):
    def test_wamv_deck_points_from_the_vrx_plate_and_the_registry(self):
        wamv = sorted(n for n in G.vehicles_v2.VEHICLES_V2 if n.startswith("wamv_vrx_v2_"))
        self.assertIn(WAMV, wamv)
        for vessel in wamv:
            spec = G.vehicles_v2.get_vehicle_v2(vessel)
            config = "bare" if vessel.endswith("_bare") else "competition"
            com_b = DATA.WAMV[config]["com_base_link_m"]
            sets = GW.report_point_sets(vessel)
            self.assertEqual([name for name, _ in sets], [REPORT_SET])
            points = GW.deck_edge_points(vessel)             # G28: the deck edge is the binding set
            self.assertEqual([p.label for p in points],
                             ["port_deck_edge_aft", "port_deck_edge_mid", "port_deck_edge_fore",
                              "starboard_deck_edge_aft", "starboard_deck_edge_mid", "starboard_deck_edge_fore"])
            z = 1.25 + 0.05 - com_b[2]
            for p in points:
                self.assertEqual(p.z_m, z)
                self.assertEqual(p.calm_freeboard_m, z + spec.hydrostatics.com_height_above_water_m)
                self.assertAlmostEqual(abs(p.y_m + com_b[1]), 0.5, places=12)
                self.assertIn(round(p.x_m + com_b[0], 12), (-0.925, 0.0, 0.925))
            pontoon = sets[0][1]
            self.assertEqual(pontoon, GW.pontoon_top_points(vessel))
            self.assertGreater(min(p.calm_freeboard_m for p in points) - max(q.calm_freeboard_m for q in pontoon), 0.8)
        competition = GW.deck_edge_points(WAMV)[0].calm_freeboard_m
        self.assertAlmostEqual(competition, 1.30 - DATA.WAMV["competition"]["com_base_link_m"][2]
                               + DATA.WAMV["competition"]["com_height_above_water_m"], places=12)
        self.assertAlmostEqual(competition, 1.18937, places=4)

    def test_blueboat_has_no_report_set_because_its_points_are_its_deck_edge(self):
        self.assertEqual(GW.report_point_sets("blueboat_v2"), ())
        freeboard = G.vehicles_v2.get_vehicle_v2("blueboat_v2").hydrostatics.freeboard_m
        for p in GW.deck_edge_points("blueboat_v2"):
            self.assertTrue(math.isclose(p.calm_freeboard_m, freeboard, rel_tol=0.0, abs_tol=1e-12), p)


class Monitor(unittest.TestCase):
    def test_default_points_are_unchanged(self):
        main = GW.GreenWaterMonitor(WAMV, 2, "cpu", torch.float32, DT, {"probability": 0.0})
        self.assertEqual(main.points, GW.deck_edge_points(WAMV))
        reports = GW.report_monitors_for(main)
        self.assertEqual(list(reports), [REPORT_SET])
        self.assertEqual(reports[REPORT_SET].points, GW.report_point_sets(WAMV)[0][1])
        self.assertIsNone(reports[REPORT_SET].formula)
        self.assertEqual(GW.report_monitors_for(None), {})
        blue = GW.GreenWaterMonitor("blueboat_v2", 2, "cpu", torch.float32, DT, {"probability": 0.0})
        self.assertEqual(GW.report_monitors_for(blue), {})

    def test_pontoon_wash_is_not_deck_wetness(self):
        # G28: main = the binding deck-edge set, report = the pontoon top line
        main = GW.GreenWaterMonitor(WAMV, 2, "cpu", torch.float32, DT, {"probability": 0.0})
        pontoon = GW.report_monitors_for(main)[REPORT_SET]
        com_h = G.vehicles_v2.get_vehicle_v2(WAMV).hydrostatics.com_height_above_water_m
        origins = torch.zeros((2, 3))

        def pose(sink):
            p = torch.zeros((2, 7))
            p[:, 3] = 1.0
            p[:, 2] = com_h
            p[0, 2] = com_h - sink
            return p
        for sink, deck_wet in ((0.5, False), (1.5, True)):
            main.reset(torch.arange(2))
            pontoon.reset(torch.arange(2))
            for _ in range(10):
                main.observe(pose(sink), origins, 0.0, torch.zeros(2), None)
                pontoon.observe(pose(sink), origins, 0.0, torch.zeros(2), None)
            self.assertTrue(pontoon.episode_record(0)["green_water"])       # pontoon tops (0.29 m) under
            self.assertEqual(main.episode_record(0)["green_water"], deck_wet)
            self.assertFalse(main.episode_record(1)["green_water"] or pontoon.episode_record(1)["green_water"])


class EnvReadOnly(unittest.TestCase):
    def test_report_set_is_read_only(self):
        for tokens in (["waves=D4"], ["waves=D2", "wind=D2", "current=D1"]):
            env_on, on = drive(tokens)
            env_off, off = drive(tokens, report=False)
            self.assertEqual(list(env_on._green_water_report), [REPORT_SET])
            self.assertEqual(env_off._green_water_report, {})
            self.assertTrue(on["records"])
            for (fa, ta), (fb, tb) in zip(on["applied"], off["applied"]):
                self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
            self.assertEqual(len(on["applied"]), len(off["applied"]))
            for oa, ob in zip(on["obs"], off["obs"]):
                self.assertTrue(torch.equal(oa, ob))
            for (ta, ua), (tb, ub) in zip(on["dones"], off["dones"]):
                self.assertTrue(torch.equal(ta, tb) and torch.equal(ua, ub))
            self.assertTrue(torch.equal(on["rng"][0], off["rng"][0]))
            self.assertTrue(np.array_equal(on["rng"][1], off["rng"][1]))
            stripped = copy.deepcopy(on["records"])
            for r in stripped:
                self.assertIn("green_water_report", r)
                self.assertEqual(r["green_water_report"][REPORT_SET]["record"]["substeps"],
                                 r["green_water"]["substeps"])
                del r["green_water_report"]
            self.assertEqual(stripped, off["records"])
            header = copy.deepcopy(on["headers"][0])
            block = header["green_water_header"].pop("report_points")
            self.assertEqual(header, off["headers"][0])
            self.assertIs(block["binding"], False)
            self.assertEqual(len(block["sets"][REPORT_SET]["points"]), 10)

    def test_blueboat_and_calm_runs_carry_nothing_new(self):
        env, trace = drive(["waves=D4"], vessel="blueboat_v2")
        self.assertEqual(env._green_water_report, {})
        self.assertTrue(all("green_water_report" not in r for r in trace["records"]))
        self.assertNotIn("report_points", trace["headers"][0]["green_water_header"])
        env, trace = drive([], vessel=WAMV, steps=40)
        self.assertEqual(env._green_water_report, {})
        self.assertTrue(all("green_water_report" not in r for r in trace["records"]))


class Certificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = wave_document(WAMV, ("waves=D4",), deadline_s=4.0)
        cls.plain = wave_document(WAMV, ("waves=D4",), deadline_s=4.0, report=False)

    def test_certificate_carries_and_validates_the_report_set(self):
        doc = copy.deepcopy(self.doc)
        self.assertEqual(check(doc), [])
        block = doc["header"]["green_water"]
        self.assertEqual(block["report_points"]["version"], RULE.REPORT_VERSION)
        summary = block["report"]["sets"][REPORT_SET]
        self.assertEqual(summary["episodes"], len(doc["records"]))
        self.assertGreater(sum(summary["mean_level_upcrossings"]), 0)       # 4 s episodes: past the 3 s window
        for r in doc["records"]:
            self.assertEqual(set(r["green_water_report"]), {REPORT_SET})
        # the binding / reported blocks are those of the run without the report set
        plain = self.plain["header"]["green_water"]
        for key in ("cell", "diagnostics", "formula", "detection", "rule", "exclusion"):
            self.assertEqual(block[key], plain[key], key)
        self.assertEqual(check(copy.deepcopy(self.plain)), [])

    def test_validator_rejects_tampered_partial_or_orphaned_report_data(self):
        doc = copy.deepcopy(self.doc)
        doc["header"]["green_water"]["report"]["sets"][REPORT_SET]["episodes_with_green_water"] += 1
        self.assertTrue(any("report.sets.%s.episodes_with_green_water" % REPORT_SET in e for e in check(doc)))
        doc = copy.deepcopy(self.doc)
        del doc["records"][0]["green_water_report"]
        self.assertTrue(any("every record or on none" in e for e in check(doc)))
        doc = copy.deepcopy(self.doc)
        doc["records"][1]["green_water_report"][REPORT_SET]["record"]["substeps"] += 1
        self.assertTrue(check(doc))
        doc = copy.deepcopy(self.doc)
        del doc["header"]["green_water"]["report_points"]
        self.assertTrue(any("without header.green_water.report_points" in e for e in check(doc)))
        doc = copy.deepcopy(self.doc)
        del doc["header"]["green_water"]["report"]
        self.assertTrue(any("report is missing" in e for e in check(doc)))
        doc = copy.deepcopy(self.doc)
        doc["header"]["green_water"]["report_points"]["binding"] = True
        self.assertTrue(any("non-binding" in e for e in check(doc)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
