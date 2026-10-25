"""F0 changes inside tasks/_shared/conditions (layer level, real HydroV2Runtime).

D14 journal contracts, D5 estimate semantics, D7 seed / episode ids / band
rename, D4 per-suite offsets, journal ray sensor from the task plan.
CPU only; run: python -B test_conditions_f0.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

OG = G.observation_groups
layer_module = G.shared("conditions.hydro_v2_layer")
strata = G.shared("conditions.strata")
perception_v3 = G.shared("conditions.perception_v3")
config_v3 = G.shared("conditions.config_v3")
runtime_module = G.shared("hydro_v2_runtime")
journal_vehicle = G.shared("journal_vehicle")


def built(tokens=(), *, task="reach", vessel="blueboat_v2", n=4, seed=G.SEED, **extra):
    cfg = G.journal_cfg(vessel, task=task, tokens=tokens, seed=seed, **extra)
    family = OG.JOURNAL_TASK_CONTRACTS[task][0]
    layer = G.conditions.build_conditions(cfg, n, "cpu", family=family)
    runtime = runtime_module.HydroV2Runtime(journal_vehicle.resolve_journal_vehicle(vessel).hydro_profile, n,
                                            torch.device("cpu"), G.DT, engine_gravity=(0.0, 0.0, -9.81))
    layer.bind_runtime(runtime)
    return layer, runtime


def reset(layer, runtime, axis=0.25):
    ids = torch.arange(layer.num_envs)
    runtime.reset(ids)
    layer.reset_runtime(ids, reference_axis_rad=torch.full((ids.numel(),), axis, dtype=torch.float64),
                        task_episode_ids=layer.scenario.episode_indices()[ids] + 1)
    layer.configure_runtime(runtime, ids)
    return ids


class JournalContracts(unittest.TestCase):
    def test_every_task_has_a_journal_contract_with_the_ray_block(self):
        self.assertEqual(set(OG.JOURNAL_TASK_CONTRACTS), set(G.plans.TASKS))
        for task, (family, key) in OG.JOURNAL_TASK_CONTRACTS.items():
            self.assertEqual(family, G.plans.TASK_TABLE[task]["carrier"])
            contract = OG.lookup(family, key)
            self.assertTrue(contract.journal and key.startswith("journal_") and key.endswith("_ray36"))
            self.assertIn("ray", contract.groups)
            self.assertEqual(contract.estimate_groups, ("pos", "vel", "ray"))
            self.assertFalse(contract.hooks_wired)                    # families flip it (their patch)
            self.assertFalse(contract.sea_state_observed)
        self.assertEqual(OG.lookup("station_keeping", "journal_kin6_ray36").vel_channels, OG.FULL)
        self.assertEqual(OG.lookup("docking", "journal_dock6_ray36").vel_channels, OG.PLANAR)
        self.assertNotIn("vel", OG.lookup("path_following", "journal_nav7_ray36").groups)

    def test_range_noise_is_admitted_for_station_keeping(self):
        layer, _ = built(("perception.enabled=true", "perception.ray_noise_sigma_m=0.1"), task="station_keeping")
        self.assertTrue(layer.wants("ray"))

    def test_hooks_flag_unlocks_only_journal_rows(self):
        p = config_v3.resolve_v3((), {"perception": {"enabled": True, "pos_noise_sigma_m": 0.1}}).perception
        flagged = SimpleNamespace(**{OG.HOOKS_FLAG: True})
        journal = OG.effective_contract(OG.lookup("path_following", "journal_nav7_ray36"), flagged)
        OG.admit_perception(journal, p)
        legacy = OG.effective_contract(OG.lookup("path_following", "native7"), flagged)
        with self.assertRaisesRegex(ValueError, "not wired"):
            OG.admit_perception(legacy, p)
        with self.assertRaisesRegex(ValueError, "not wired"):
            OG.admit_perception(OG.lookup("path_following", "journal_nav7_ray36"), p)
        with self.assertRaises(ValueError):
            OG.effective_contract(OG.lookup("docking", "journal_dock6_ray36"),
                                  SimpleNamespace(**{OG.HOOKS_FLAG: "yes"}))

    def test_journal_derivation_and_carrier_check(self):
        cfg = SimpleNamespace(journal_task="scatter", journal_contract_version=2)
        self.assertEqual(OG.derive_contract("hazard_nav", cfg), "journal_v3_ray36")
        with self.assertRaisesRegex(ValueError, "carried by family"):
            OG.derive_contract("docking", cfg)
        cfg.observation_contract = "journal_v3_ray36"
        self.assertEqual(OG.derive_contract("hazard_nav", cfg), "journal_v3_ray36")


class Estimates(unittest.TestCase):
    def test_own_position_estimate_is_truth_plus_error(self):
        contract = OG.effective_contract(OG.lookup("hazard_nav", "journal_v3_ray36"),
                                         SimpleNamespace(**{OG.HOOKS_FLAG: True}))
        p = config_v3.resolve_v3(("journal.perception.noise.D3", "journal.perception.bias.D2"), seed=9).perception
        make = lambda: perception_v3.PerceptionV3Degrader(p, 4, "cpu", 9, contract, 1 / 60, ray_count=36,  # noqa
                                                          ray_max_range_m=30.0)
        own, rel = make(), make()
        for model in (own, rel):
            model.reset(torch.arange(4), torch.zeros(4, dtype=torch.long), torch.arange(4),
                        torch.zeros(4, dtype=torch.float64))
        truth = torch.tensor([[1.0, 2.0], [3.0, -4.0], [0.0, 0.0], [10.0, 5.0]])
        for _ in range(5):
            estimate = own.estimate("pos", truth)
            relative = rel.apply("pos", -truth)                         # goal at the origin: x = -truth
            torch.testing.assert_close(estimate - truth, -(relative + truth))
            self.assertFalse(torch.equal(estimate, truth))

    def test_temporal_axes_build_every_estimate_group(self):
        layer, runtime = built(("perception.latency=D2",), task="path_following")
        self.assertEqual(sorted(layer.perception.groups), ["pos", "ray", "vel"])  # vel although nav7 has none
        reset(layer, runtime)
        rates = torch.ones(4, 3)
        self.assertTrue(torch.equal(layer.estimate_velocity(rates), rates))      # first frame held

    def test_estimate_counters_and_journal_refusal_of_observation_out(self):
        layer, runtime = built(("perception=D1",))
        with self.assertRaises(RuntimeError):
            layer.estimate_position(torch.zeros(4, 2))                           # never reset
        reset(layer, runtime)
        layer.estimate_position(torch.zeros(4, 2))
        with self.assertRaisesRegex(RuntimeError, "exactly once per control step"):
            layer.estimate_position(torch.zeros(4, 2))
        with self.assertRaisesRegex(RuntimeError, "own-state estimate"):
            layer.observation_out("pos", torch.zeros(4, 2))
        self.assertEqual(layer.estimate_counts(0), {"pos": 1, "vel": 0, "ray": 0})


class D7(unittest.TestCase):
    def test_seed_is_mandatory_and_must_match(self):
        cfg = G.journal_cfg()
        family = "hazard_nav"
        for bad in (None, True, 3.0):
            cfg.seed = bad
            with self.subTest(seed=bad):
                with self.assertRaisesRegex(ValueError, "seed"):
                    G.conditions.build_conditions(cfg, 2, "cpu", family=family)
        cfg.seed = G.SEED + 1                                                  # cfg.conditions carries 42
        with self.assertRaisesRegex(ValueError, "differs from the env evaluation seed"):
            G.conditions.build_conditions(cfg, 2, "cpu", family=family)

    def test_task_episode_ids_are_mandatory(self):
        layer, runtime = built(())
        runtime.reset()
        with self.assertRaisesRegex(ValueError, "task_episode_ids is mandatory"):
            layer.reset_runtime(torch.arange(4), reference_axis_rad=torch.zeros(4, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "diverged"):
            layer.reset_runtime(torch.arange(4), reference_axis_rad=torch.zeros(4, dtype=torch.float64),
                                task_episode_ids=torch.tensor([0, 0, 1, 0]))

    def test_band_rename_and_true_band_edges(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            layer, runtime = built(("waves=D2",), vessel=vessel)
            reset(layer, runtime)
            record = layer.episode_stamp(1)["runtime"]["waves"]           # 0 ticks, 0 estimates
            self.assertNotIn("band_hz", record)
            lo, hi = record["band_edges_over_fp"]
            fp = 1.0 / record["tp_s"]
            self.assertAlmostEqual(record["band_edges_hz"][0], lo * fp)
            self.assertAlmostEqual(record["band_edges_hz"][1], hi * fp)
            first, last = record["component_frequency_range_hz"]
            self.assertTrue(lo * fp < first < last < hi * fp)             # midpoint grid inside the edges
            resolved = layer.context["resolved"]["waves"]
            self.assertEqual([lo, hi], [resolved["band_min_over_fp"], resolved["band_max_over_fp"]])


class Offsets(unittest.TestCase):
    def test_offsets_shift_only_their_suite(self):
        tokens = ("waves=D2", "wind=D2", "current=D2", "perception.bias=D2")
        base, rt0 = built(tokens)
        shifted, rt1 = built(tokens + ("wind.offset_deg=30", "current.offset_deg=-90", "perception.offset_deg=180"))
        for layer, rt in ((base, rt0), (shifted, rt1)):
            reset(layer, rt)
        world = base.stratum_world
        self.assertTrue(torch.equal(world, shifted.stratum_world))           # the shared stratum is unchanged
        self.assertTrue(torch.equal(base.sea.mean_direction, shifted.sea.mean_direction))
        torch.testing.assert_close(shifted.wind.direction_rad.to(torch.float64).cpu(),
                                   (world + math.radians(30)).to(shifted.wind.direction_rad.dtype).double())
        torch.testing.assert_close(shifted._current_direction_world(), world - math.pi / 2)
        torch.testing.assert_close(shifted.perception.bias_direction, base.perception.bias_direction + math.pi)
        self.assertTrue(torch.equal(base._current_direction_world(), world))     # zero offset: exact identity

    def test_offset_validation(self):
        with self.assertRaisesRegex(ValueError, "disabled"):
            config_v3.resolve_v3((), {"wind": {"direction_offset_rad": 0.5}})
        with self.assertRaisesRegex(ValueError, "direction_mode='strata'"):
            config_v3.resolve_v3((), {"current": {"enabled": True, "speed_mps": 0.2, "direction_mode": "fixed",
                                                  "direction_offset_rad": 0.5}})
        with self.assertRaisesRegex(ValueError, r"\(-pi, pi\]"):
            config_v3.resolve_v3(("journal.waves.D1",), {"waves": {"direction_offset_rad": -math.pi}})
        ok = config_v3.resolve_v3(("journal.waves.D1",), {"waves": {"direction_offset_rad": math.pi}})
        self.assertEqual(ok.waves.direction_offset_rad, math.pi)


class JournalRaySensor(unittest.TestCase):
    def test_ray_sensor_comes_from_the_plan(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            layer, _ = built(("perception.bundle=D2",), vessel=vessel)
            plan = G.plans.plan_class("reach", vessel)(2)
            self.assertEqual(layer.perception.groups["ray"].dim, 36)
            self.assertAlmostEqual(layer.perception.ray_max, plan.params["ray_sensor_max_range_m"])
        with self.assertRaisesRegex(ValueError, "differs from the task plan"):
            built(("perception.bundle=D2",), vessel="wamv_vrx_v2_opt1", ray_count=36, ray_max_range_m=30.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
