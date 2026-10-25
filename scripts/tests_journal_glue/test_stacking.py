"""F0 / CONTEXT_R4 D4 CPU proof: arbitrary suite stacking on all 24 journal cfgs.

* All 2^6 = 64 subsets of {waves, wind, current, actuator (4 sub-axes),
  perception (4 primary axes), dynamic} at representative levels (D2) resolve
  AND admit on each of E5's 24 journal cfg templates (constructed from the real
  cfg modules under stubbed isaaclab; the family obligations the harness
  applies are listed in glue_fixtures.FAMILY_OBLIGATIONS) = 1536 admissions.
* Offsets, the sensor bundle and independent sub-axis levels stack too.
* Only the declared conflicts C1-C6 reject, each with its code and reason.
* The CLI and JSON/YAML files take the same syntax and resolve identically.
Set F0_STACKING_EVIDENCE=<path> to write the 1536-row admission table as JSON.
CPU only; run: python -B test_stacking.py
"""
from __future__ import annotations

from itertools import product
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glue_fixtures as G  # noqa: E402

JC = G.journal_config
SUITES = ("waves", "wind", "current", "actuator", "perception", "dynamic")
TEMPLATES = None


def templates():
    global TEMPLATES
    if TEMPLATES is None:
        TEMPLATES = G.build_templates()
    return TEMPLATES


def subsets():
    for mask in product((False, True), repeat=len(SUITES)):
        yield tuple(s for s, on in zip(SUITES, mask) if on)


class StackingProof(unittest.TestCase):
    def test_all_64_subsets_admit_on_all_24_journal_cfgs(self):
        cfgs = templates()
        self.assertEqual(len(cfgs), 24)
        self.assertEqual({(c.journal_task, c.vehicle) for c in cfgs.values()}, set(G.plans.JOURNAL_ENV_IDS))
        rows, digests = [], {}
        for subset in subsets():
            tokens = [G.REPRESENTATIVE[s] for s in subset]
            resolved = JC.resolve_journal_conditions(presets=tokens, seed=1234)
            for suite in ("waves", "wind", "current", "actuator", "perception"):
                self.assertEqual(getattr(resolved.conditions, suite).enabled, suite in subset, (subset, suite))
            self.assertEqual(resolved.dynamic_enabled, "dynamic" in subset)
            for name, cfg in sorted(cfgs.items()):
                report = JC.admit_journal_conditions(cfg, resolved)
                self.assertEqual(report["task"], cfg.journal_task)
                waves, wind = report["resolved"]["waves"], report["resolved"]["wind"]
                if "waves" in subset:          # band raised to the hull requirement at bind time
                    self.assertGreaterEqual(waves["band_max_over_fp"], waves["band_required_max_over_fp"])
                    self.assertGreaterEqual(waves["group_repeat_s"], 480.0 - 1e-9)
                if "wind" in subset:           # U10 converted with the vessel's own windage height
                    self.assertGreater(wind["applied_vessel_height_speed_mps"], 0.0)
                    self.assertLess(wind["applied_vessel_height_speed_mps"], wind["u10_mps"])
                self.assertEqual(report["dynamic"]["vessel_id"], cfg.vehicle)
                rows.append({"cfg": name, "task": cfg.journal_task, "vessel": cfg.vehicle,
                             "suites": list(subset), "contract": report["contract"],
                             "parameters_sha256": report["parameters_sha256"],
                             "dynamic_dose": report["dynamic"]["dose"]})
                digests.setdefault(subset, set()).add(report["parameters_sha256"])
        self.assertEqual(len(rows), 64 * 24)
        # one parameter digest per stack (the stamp does not depend on the task), 64 distinct stacks
        self.assertTrue(all(len(v) == 1 for v in digests.values()))
        self.assertEqual(len({next(iter(v)) for v in digests.values()}), 64 // 2)   # dynamic is not schema 3
        target = os.environ.get("F0_STACKING_EVIDENCE")
        if target:
            Path(target).write_text(json.dumps({"obligations": list(G.FAMILY_OBLIGATIONS), "rows": rows},
                                               indent=1), encoding="ascii")
        print(f"STACKING PROOF: {len(rows)} admissions (64 subsets x 24 journal cfgs), 0 rejected", flush=True)

    def test_independent_levels_offsets_and_bundle_stack(self):
        tokens = ["waves.B=D3", "wind=D4", "current=D1", "actuator.cap=D1", "actuator.gain=D3",
                  "actuator.lag=D4", "actuator.resistance=D2", "perception.noise=D1", "perception.bias=D4",
                  "perception.latency=D3", "perception.dropout=D2", "perception.bundle=D2", "dynamic=D4",
                  "wind.offset_deg=30", "waves.offset_deg=-45", "current.offset_deg=180",
                  "perception.offset_deg=90"]
        resolved = JC.resolve_journal_conditions(presets=tokens, seed=5)
        c = resolved.conditions
        self.assertEqual((c.actuator.thrust_cap, c.actuator.motor_gain_one_sided, c.actuator.additional_lag_s,
                          c.actuator.resistance_scale), (0.9, 0.70, 0.5, 1.25))
        self.assertEqual((c.perception.pos_noise_sigma_m, c.perception.pos_bias_m, c.perception.latency_s,
                          c.perception.outage_s, c.perception.vel_noise_sigma_mps, c.perception.ray_noise_sigma_m),
                         (0.02, 1.0, 0.25, 0.3, 0.05, 0.03))
        self.assertAlmostEqual(c.wind.direction_offset_rad, 0.5235987755982988)
        self.assertAlmostEqual(c.current.direction_offset_rad, 3.141592653589793)
        admitted, rejected = [], {}
        for name, cfg in sorted(templates().items()):
            try:
                JC.admit_journal_conditions(cfg, resolved)
                admitted.append(name)
            except JC.SuiteConflict as error:
                rejected[name] = error
        # the bundle's velocity noise is a declared conflict (C4) on path following only
        self.assertEqual(sorted(rejected), ["PathFollowingBlueBoatJournalEnvCfg", "PathFollowingWAMVJournalEnvCfg"])
        self.assertTrue(all(e.code == "C4" for e in rejected.values()))
        self.assertEqual(len(admitted), 22)
        without_bundle = JC.resolve_journal_conditions(presets=[t for t in tokens if "bundle" not in t], seed=5)
        for name in rejected:
            JC.admit_journal_conditions(templates()[name], without_bundle)
        ray_only = JC.resolve_journal_conditions(["perception.ray_noise_sigma_m=0.1", "perception.enabled=true"],
                                                 seed=5)
        JC.admit_journal_conditions(templates()["StationKeepingBlueBoatJournalEnvCfg"], ray_only)   # D14

    def test_offsets_are_stamped_and_change_the_digest(self):
        cfg = templates()["ReachBlueBoatJournalEnvCfg"]
        base = JC.admit_journal_conditions(cfg, JC.resolve_journal_conditions(["waves=D2", "wind=D2"], seed=3))
        off = JC.admit_journal_conditions(cfg, JC.resolve_journal_conditions(["waves=D2", "wind=D2",
                                                                              "wind.offset_deg=30"], seed=3))
        self.assertNotEqual(base["parameters_sha256"], off["parameters_sha256"])


class DeclaredConflicts(unittest.TestCase):
    CASES = (
        ("C1", ["actuator=D2", "actuator.cap=D1"]),
        ("C1", ["waves=D2", "waves.B=D2"]),
        ("C1", ["current=D2", "current.speed_mps=0.3"]),
        ("C1", ["wind=D1", "wind=D2"]),
        ("C2", ["wind.offset_deg=30"]),
        ("C2", ["current.speed_mps=0.3"]),
        ("C2", ["perception.noise=D2", "perception.offset_deg=45"]),
        ("C3", ["current.enabled=true", "current.speed_mps=0.3", "current.direction_mode=\"fixed\"",
                "current.offset_deg=30"]),
        ("C3", ["waves=D2", "waves.offset_deg=200"]),
        ("C3", ["waves=D2", "waves.offset_deg=-180"]),
        ("C5", ["waves.enabled=true", "waves.hs_m=1.0", "waves.tp_s=2.0"]),
        ("C6", ["foo=D1"]),
        ("C6", ["waves=D7"]),
        ("C6", ["actuator.mass=D1"]),
        ("C6", ["wind.speed_mps=3"]),
        ("C6", ["dynamic.dose=3"]),
    )

    def test_every_declared_conflict_rejects_with_its_code(self):
        for code, tokens in self.CASES:
            with self.subTest(tokens=tokens):
                with self.assertRaises(JC.SuiteConflict) as caught:
                    JC.resolve_journal_conditions(presets=tokens, seed=1)
                self.assertEqual(caught.exception.code, code, str(caught.exception))
                self.assertIn(JC.DECLARED_CONFLICTS[code], str(caught.exception))

    def test_admission_time_conflicts(self):
        cfgs = templates()
        cases = (("C4", "PathFollowingBlueBoatJournalEnvCfg", ["perception.bundle=D1"]),
                 ("C4", "DockingWAMVJournalEnvCfg", ["perception.enabled=true",
                                                     "perception.yaw_rate_noise_sigma_rad_s=0.05"]),
                 ("C5", "ScatterBlueBoatJournalEnvCfg", ["perception.enabled=true", "perception.latency_s=0.01"]),
                 ("C5", "ScatterBlueBoatJournalEnvCfg", ["waves.enabled=true", "waves.hs_m=0.5",
                                                         "waves.band_policy=\"fixed\""]))
        for code, name, tokens in cases:
            with self.subTest(cfg=name, tokens=tokens):
                resolved = JC.resolve_journal_conditions(tokens, seed=1)
                with self.assertRaises(JC.SuiteConflict) as caught:
                    JC.admit_journal_conditions(cfgs[name], resolved)
                self.assertEqual(caught.exception.code, code, str(caught.exception))

    def test_same_value_twice_is_not_a_conflict(self):
        resolved = JC.resolve_journal_conditions(["actuator=D2", "actuator.cap=D2", "current=D2",
                                                  "current.speed_mps=0.5"], seed=1)
        self.assertEqual(resolved.conditions.actuator.thrust_cap, 0.75)

    def test_missing_seed_is_an_error(self):
        with self.assertRaises(ValueError):
            JC.resolve_journal_conditions(["waves=D1"], seed=None)

    def test_family_hooks_flag_gates_perception(self):
        cfg = templates()["ReachBlueBoatJournalEnvCfg"]
        unwired = type(cfg).__new__(type(cfg))
        unwired.__dict__.update(cfg.__dict__)
        setattr(unwired, G.observation_groups.HOOKS_FLAG, False)
        resolved = JC.resolve_journal_conditions(["perception=D1"], seed=1)
        with self.assertRaises(JC.SuiteConflict) as caught:
            JC.admit_journal_conditions(unwired, resolved)
        self.assertEqual(caught.exception.code, "C4")
        JC.admit_journal_conditions(unwired, JC.resolve_journal_conditions(["waves=D1"], seed=1))


class SameSyntaxEverywhere(unittest.TestCase):
    TOKENS = ["waves=D2", "wind=D3", "actuator.cap=D1", "dynamic=D2", "wind.offset_deg=30"]

    def test_cli_json_yaml_and_python_resolve_identically(self):
        python = JC.resolve_journal_conditions(presets=self.TOKENS, seed=42)
        cli = JC.main(["--suite", *self.TOKENS, "--seed", "42"])
        split = JC.main(["--suite", "waves=D2", "wind=D3", "--suite", "actuator.cap=D1", "dynamic=D2",
                         "wind.offset_deg=30", "--seed", "42"])
        with tempfile.TemporaryDirectory() as folder:
            listed = Path(folder) / "suite.json"
            listed.write_text(json.dumps({"suite": self.TOKENS, "seed": 42}), encoding="ascii")
            mapped = Path(folder) / "suite_map.json"
            mapped.write_text(json.dumps({"suites": {"waves": "D2", "wind": "D3", "actuator.cap": "D1",
                                                     "dynamic": "D2"},
                                          "overrides": {"wind": {"offset_deg": 30}}, "seed": 42}),
                              encoding="ascii")
            yaml_file = Path(folder) / "suite.yaml"
            yaml_file.write_text("suite:\n" + "".join(f"  - {t}\n" for t in self.TOKENS) + "seed: 42\n",
                                 encoding="ascii")
            from_files = [JC.main(["--suite-file", str(p)]) for p in (listed, mapped, yaml_file)]
        mapping = JC.resolve_journal_conditions(presets={"waves": "D2", "wind": "D3", "actuator.cap": "D1",
                                                         "dynamic": "D2"},
                                                overrides={"wind": {"offset_deg": 30}}, seed=42)
        for other in (cli, split, mapping, *from_files):
            self.assertEqual(other.stamp(), python.stamp())
            self.assertEqual(other.conditions, python.conditions)
        self.assertEqual(JC.resolve_journal_conditions(python.tokens(), seed=42).stamp(), python.stamp())


if __name__ == "__main__":
    unittest.main(verbosity=2)
