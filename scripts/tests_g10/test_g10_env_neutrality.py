"""G10: green-water detection through the REAL journal mixin (F0 FakeJournalEnv, CPU).

* read-only: with waves on (D4, D0 and every suite at once) the run with the
  detector and the run with the detector removed give bit-identical applied
  wrenches, observations, termination codes, global RNG states and every
  EXISTING record/header field (the only difference is the added green_water
  keys);
* calm and wave-free stacks build no detector (no attribute read, no extra
  call, no new field): records and headers carry exactly the pre-G10 keys;
* the per-episode record is latched before the reset (reset-row isolation:
  a reset row restarts at zero, the other rows keep counting) and its substep
  count equals the runtime ticks of the episode;
* a deterministic immersion (water surface raised 1 m) gives green water with
  onset 0 s; waves D0 (enabled, Hs 0) uses the calm surface and stamps formula 0;
* the 0.75 m supplement stamps preset role SUPPLEMENT.

Run: python -B scripts/tests_g10/test_g10_env_neutrality.py   (or pytest)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tests_journal_glue"))

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

GW = G.shared("conditions.green_water")
RULE = G.shared("conditions.green_water_rule")
ACTIONS = torch.tensor([[0.6, 0.4], [0.3, -0.2]])
ALL_SUITES = ["waves=D4", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "dynamic=D2"]


def drive(tokens, vessel="blueboat_v2", steps=90, deadline_s=0.5, detector=True):
    """Run a FakeJournalEnv; returns (env, trace) with every observable output collected."""
    torch.manual_seed(1234)
    np.random.seed(1234)
    cfg = G.journal_cfg(vessel, tokens=tokens)
    if detector:
        env = G.FakeJournalEnv(cfg, 2, deadline_s=deadline_s)
    else:
        with patch.object(GW, "monitor_for", lambda *args, **kwargs: None):
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


def strip(items, *keys):
    out = copy.deepcopy(items)
    for item in out:
        for key in keys:
            item.pop(key, None)
    return out


class Neutrality(unittest.TestCase):
    def assert_identical_except_green(self, a, b):
        self.assertEqual(len(a["applied"]), len(b["applied"]))
        for (fa, ta), (fb, tb) in zip(a["applied"], b["applied"]):
            self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
        self.assertEqual(len(a["obs"]), len(b["obs"]))
        for oa, ob in zip(a["obs"], b["obs"]):
            self.assertTrue(torch.equal(oa, ob))
        for (ta, ua), (tb, ub) in zip(a["dones"], b["dones"]):
            self.assertTrue(torch.equal(ta, tb) and torch.equal(ua, ub))
        self.assertTrue(torch.equal(a["rng"][0], b["rng"][0]))
        self.assertTrue(np.array_equal(a["rng"][1], b["rng"][1]))
        # G18: the read-only diagnostics ride beside the rule/1 record (records[].green_water_diagnostics);
        # T25 (T1.35d): so do the report-only point sets (records[].green_water_report, WAM-V)
        self.assertEqual(strip(a["records"], "green_water", "green_water_diagnostics", "green_water_report"),
                         b["records"])
        self.assertEqual(strip(a["headers"], "green_water_header"), b["headers"])

    def test_waves_d4_detector_is_read_only(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            env_on, on = drive(["waves=D4"], vessel)
            env_off, off = drive(["waves=D4"], vessel, detector=False)
            self.assertIsNotNone(env_on._green_water)
            self.assertIsNone(env_off._green_water)
            self.assertTrue(on["records"], "episodes must finish inside the test")
            self.assertTrue(all("green_water" in r for r in on["records"]))
            self.assertIn("green_water_header", on["headers"][0])
            self.assert_identical_except_green(on, off)

    def test_every_suite_at_once_detector_is_read_only(self):
        _, on = drive(ALL_SUITES)
        _, off = drive(ALL_SUITES, detector=False)
        self.assert_identical_except_green(on, off)

    def test_waves_d0_uses_the_calm_surface(self):
        env, on = drive(["waves=D0"])
        _, off = drive(["waves=D0"], detector=False)
        self.assert_identical_except_green(on, off)
        header = on["headers"][0]["green_water_header"]
        self.assertEqual(header["formula"]["probability"], 0.0)
        self.assertTrue(header["formula"]["inputs"]["calm"])
        self.assertTrue(all(r["green_water"]["green_water"] is False for r in on["records"]))

    def test_calm_and_wave_free_stacks_build_no_detector(self):
        for tokens in ([], ["wind=D3", "current=D2"], ["actuator=D1", "dynamic=D1"]):
            calls = []
            original = GW.monitor_for
            with patch.object(GW, "monitor_for", lambda *a, **k: calls.append(1) or original(*a, **k)):
                env, trace = drive(tokens, steps=60)
            self.assertIsNone(env._green_water)
            self.assertEqual(calls, [1])                 # consulted once at setup, returned None
            for record in trace["records"]:
                self.assertEqual(set(record), {"conditions", "dynamic_hazards", "scenario_hashes"})
            self.assertEqual(set(trace["headers"][0]), {"conditions", "dynamic_hazards_header", "pairing",
                                                        "journal_mixin"})


class Latching(unittest.TestCase):
    def test_records_are_latched_before_reset_with_runtime_ticks(self):
        env, trace = drive(["waves=D4"], steps=90, deadline_s=0.5)
        labels = env._green_water.labels
        self.assertGreaterEqual(len(trace["records"]), 4)
        for record in trace["records"]:
            green = record["green_water"]
            self.assertEqual(RULE.record_errors(green, 6, 1.0 / 120.0, labels), [])
            self.assertEqual(green["substeps"], record["conditions"]["runtime"]["runtime_ticks"])

    def test_raised_surface_gives_green_water_from_the_first_substep(self):
        class Flooded(G.FakeJournalEnv):
            def step(self, actions):
                self.actions = self.journal_pre_physics(actions)
                for _ in range(self.cfg.decimation):
                    level = torch.tensor([1.0, 0.0])          # env 0: surface 1 m above every deck point
                    forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins, level)
                    self.robot.set_external_force_and_torque(forces, torques)
                    self.robot.integrate(self.cfg.sim.dt)
                self.episode_length_buf += 1
                terminated, truncated = self._get_dones()
                done = (terminated | truncated).nonzero().flatten()
                if done.numel():
                    self._reset_idx(done)
                return self._get_observations(), terminated, truncated

            def _get_dones(self):
                # the flooded hull would trip the contract's validity codes; this test only needs the
                # mixin's call order (pre_dones + dones with "running" codes)
                self.journal_pre_dones(self.robot.data, self.scene.env_origins, 0.0)
                return self.journal_dones(torch.zeros(self.num_envs, dtype=torch.long))

        cfg = G.journal_cfg("blueboat_v2", tokens=["waves=D2"])
        env = Flooded(cfg, 2, deadline_s=30.0)
        env.reset()
        for _ in range(4):
            env.step(ACTIONS)
        gw = env._green_water
        self.assertEqual(gw.green.tolist(), [True, False])
        record = gw.episode_record(0, runtime_ticks=int(env._hydro_v2.ticks[0]))
        self.assertEqual(record["first_time_s"], 0.0)
        self.assertEqual(record["longest_run_substeps"], 8)
        self.assertEqual(record["submerged_point_substeps"], 8 * 12)
        # reset-row isolation through the mixin: reset env 0 only
        counters = {k: getattr(gw, k)[1].clone() for k in ("run", "longest", "total", "green", "ticks")}
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(int(gw.ticks[0]), 0)
        self.assertFalse(bool(gw.green[0]))
        for key, value in counters.items():
            self.assertTrue(torch.equal(getattr(gw, key)[1], value), key)

    def test_supplement_preset_stamps_its_role(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            cfg = G.journal_cfg(vessel, tokens=["journal.waves.supp075"])
            env = G.FakeJournalEnv(cfg, 2)
            header = env.journal_certificate_header()["green_water_header"]
            self.assertEqual(header["formula"]["preset"], "journal.waves.supp075")
            self.assertEqual(header["formula"]["preset_role"], "SUPPLEMENT")
            self.assertEqual(header["presets"], ["journal.waves.supp075"])
            self.assertEqual(cfg.conditions.waves.hs_m, 0.75)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
