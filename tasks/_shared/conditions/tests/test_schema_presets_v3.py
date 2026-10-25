"""E3: schema 3, journal dose presets, certificates and the D7 glue (CPU only).

Run: python -B tasks/_shared/conditions/tests/test_schema_presets_v3.py
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch

import e3_fixtures as F

api = F.api
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_cfg(family="station_keeping", **kwargs):
    return F.env_cfg(family, **kwargs)


class SchemaRejections(unittest.TestCase):
    def assert_rejected(self, overrides=None, presets=(), family="station_keeping", **cfg):
        with self.assertRaises(ValueError):
            api.build_conditions(build_cfg(family, conditions=overrides, condition_presets=presets, **cfg),
                                 2, "cpu", family=family)

    def test_legacy_keys_are_rejected_with_reasons(self):
        cases = {
            ("waves", "heave_force_scale"): 100.0, ("waves", "slope_moment_scale"): 1.0,
            ("waves", "orbital_drag_coeff"): 8.0, ("waves", "hs_range"): (0.1, 0.1),
            ("current", "force_mode"): "incremental_from_calm", ("current", "drag_coeff"): 8.0,
            ("actuator", "half_beam_m"): 0.45, ("actuator", "thrust_imbalance"): 0.1,
            ("actuator", "mass_scale"): 1.2, ("actuator", "payload_mass_kg"): 4.0,
            ("actuator", "payload_offset_x_m"): 0.2, ("actuator", "motor_tau_s"): 0.1,
            ("perception", "dropout_p"): 0.1, ("wind", "speed_mps"): 3.0,
        }
        for (suite, key), value in cases.items():
            with self.subTest(key=suite + "." + key):
                with self.assertRaisesRegex(ValueError, "legacy key"):
                    api.resolve_v3((), {suite: {"enabled": True, key: value}})
                self.assert_rejected({suite: {key: value}})

    def test_unknown_keys_schema2_objects_and_zero_wind_are_rejected(self):
        with self.assertRaises(ValueError):
            api.resolve_v3((), {"waves": {"nonsense": 1}})
        with self.assertRaises(ValueError):
            api.resolve_v3((), {"nonsense": {}})
        with self.assertRaises(ValueError):
            api.resolve_v3((), api.resolve(["wave.medium"]))          # schema-2 object
        with self.assertRaisesRegex(ValueError, "module is OFF"):
            api.resolve_v3((), {"wind": {"enabled": True, "u10_mps": 0.0}})
        with self.assertRaises(ValueError):
            api.resolve_v3((), {"schema_version": 2})
        with self.assertRaises(ValueError):
            api.resolve_v3(("journal.waves.D9",))

    def test_dynamic_hazards_c4_trigger_rejected_b01(self):
        with self.assertRaisesRegex(ValueError, "D7"):
            api.resolve_v3((), {"dynamic_hazards": {"enabled": True}})
        cfg = api.resolve_v3((), {"dynamic_hazards": {"enabled": False}})
        self.assertFalse(cfg.any_enabled)
        # legacy J-base journal mode (C2 physics v2 without hydro_v2): also rejected
        legacy = SimpleNamespace(conditions={"dynamic_hazards": {"enabled": True}}, condition_presets=(),
                                 seed=1, vehicle="blueboat", sim=SimpleNamespace(dt=F.DT), decimation=2,
                                 journal_mode=True, obs_kinematic=True,
                                 underwater_physics_cfg=SimpleNamespace(physics_version=2))
        with self.assertRaisesRegex(ValueError, "dynamic_hazards"):
            api.build_conditions(legacy, 2, "cpu", family="station_keeping")

    def test_hydro_v2_requires_physics_v2_and_a_v2_vessel(self):
        cfg = build_cfg()
        cfg.underwater_physics_cfg.physics_version = 1
        with self.assertRaisesRegex(ValueError, "physics_version=2"):
            api.build_conditions(cfg, 2, "cpu", family="station_keeping")
        self.assert_rejected(vehicle="blueboat")       # legacy registry name
        cfg = build_cfg()
        cfg.underwater_physics_cfg.hydrodynamics_model = "hydro_v3"
        with self.assertRaises(ValueError):
            api.build_conditions(cfg, 2, "cpu", family="station_keeping")

    def test_legacy_overlap_rejected(self):
        self.assert_rejected(current_speed_override_mps=0.3)
        self.assert_rejected(sea_state=SimpleNamespace(enable=True))
        self.assert_rejected(mass_scale=2.0)
        self.assert_rejected(obs_noise_sigma_pos=0.1)
        self.assert_rejected(appear_count=1, family="hazard_nav")

    def test_waves_observed_plus_condition_waves_rejected_item18(self):
        self.assert_rejected(presets=("journal.waves.D2",), obs_sea_state=True)
        # legacy composition: condition waves + observed sea state is rejected too
        legacy = SimpleNamespace(conditions=None, condition_presets=("wave.medium",), seed=1,
                                 vehicle="blueboat", sim=SimpleNamespace(dt=F.DT), decimation=2,
                                 obs_sea_state=True, obs_kinematic=True)
        with self.assertRaisesRegex(ValueError, "observed sea state"):
            api.build_conditions(legacy, 2, "cpu", family="station_keeping")

    def test_perception_dose_on_missing_group_rejected_item7(self):
        # native HazardNav v1 has no velocity channel (D2 F7)
        self.assert_rejected({"perception": {"enabled": True, "vel_noise_sigma_mps": 0.05}},
                             family="hazard_nav", obs_v3=False)
        # station keeping without kinematics: no vel group; station has no ray group
        self.assert_rejected({"perception": {"enabled": True, "vel_noise_sigma_mps": 0.05}},
                             obs_kinematic=False)
        self.assert_rejected({"perception": {"enabled": True, "ray_noise_sigma_m": 0.1}})
        # a family whose observation hooks are not wired yet
        with self.assertRaisesRegex(ValueError, "not wired"):
            api.build_conditions(build_cfg("path_following", observation_contract="native7",
                                           condition_presets=("journal.perception.noise.D2",)),
                                 2, "cpu", family="path_following")
        # but the position axes are admitted where the group exists
        for family, kwargs in (("hazard_nav", {"obs_v3": False}), ("station_keeping", {"obs_kinematic": False})):
            layer = api.build_conditions(build_cfg(family, condition_presets=("journal.perception.noise.D2",),
                                                   **kwargs), 2, "cpu", family=family)
            self.assertTrue(layer.wants("pos"))

    def test_latency_must_be_a_multiple_of_the_control_period(self):
        self.assert_rejected({"perception": {"enabled": True, "latency_s": 0.04}})
        layer = api.build_conditions(build_cfg(condition_presets=("journal.perception.latency.D1",)),
                                     2, "cpu", family="station_keeping")
        self.assertEqual(layer.perception.latency, 3)

    def test_steepness_admission(self):
        with self.assertRaisesRegex(ValueError, "1/7"):
            api.resolve_v3((), {"waves": {"enabled": True, "hs_m": 1.0, "tp_s": 2.0}})
        with self.assertRaisesRegex(ValueError, "DNV"):
            api.resolve_v3((), {"waves": {"enabled": True, "hs_m": 0.6, "tp_s": 2.5}})
        ok = api.resolve_v3((), {"waves": {"enabled": True, "hs_m": 0.6, "tp_s": 2.5,
                                           "steepness_rule": "generator_1_7"}})
        self.assertEqual(ok.waves.hs_m, 0.6)

    def test_preset_field_conflicts(self):
        with self.assertRaisesRegex(ValueError, "both set"):
            api.resolve_v3(("journal.waves.D2", "journal.waves.D3"))
        with self.assertRaisesRegex(ValueError, "both set"):
            api.resolve_v3(("journal.combo.wave_wind.D2", "journal.wind.D3"))
        combined = api.resolve_v3(("journal.actuator.cap.D2", "journal.actuator.lag.D3"))
        self.assertEqual((combined.actuator.thrust_cap, combined.actuator.additional_lag_s), (0.75, 0.25))


class Presets(unittest.TestCase):
    LADDERS = {
        "journal.waves.D{}": [("waves", "hs_m", v) for v in (0.0, 0.125, 0.25, 0.5, 1.0)],
        "journal.current.D{}": [("current", "speed_mps", v) for v in (0.0, 0.25, 0.5, 1.0, 1.5)],
        "journal.actuator.cap.D{}": [("actuator", "thrust_cap", v) for v in (1.0, 0.9, 0.75, 0.5, 0.25)],
        "journal.actuator.gain.D{}": [("actuator", "motor_gain_one_sided", v) for v in (1.0, 0.95, 0.85, 0.7, 0.5)],
        "journal.actuator.lag.D{}": [("actuator", "additional_lag_s", v) for v in (0.0, 0.05, 0.1, 0.25, 0.5)],
        "journal.actuator.resistance.D{}": [("actuator", "resistance_scale", v) for v in (1.0, 1.1, 1.25, 1.5, 2.0)],
        "journal.perception.noise.D{}": [("perception", "pos_noise_sigma_m", v) for v in (0.0, 0.02, 0.1, 0.4, 1.0)],
        "journal.perception.bias.D{}": [("perception", "pos_bias_m", v) for v in (0.0, 0.05, 0.2, 0.5, 1.0)],
        "journal.perception.latency.D{}": [("perception", "latency_s", v) for v in (0.0, 0.05, 0.1, 0.25, 0.5)],
        "journal.perception.dropout.D{}": [("perception", "outage_s", v) for v in (0.0, 0.1, 0.3, 1.0, 3.0)],
    }

    def test_ladder_numbers(self):
        for pattern, rows in self.LADDERS.items():
            for level, (suite, key, value) in enumerate(rows):
                cfg = api.resolve_v3((pattern.format(level),))
                self.assertEqual(getattr(getattr(cfg, suite), key), value, pattern.format(level))
                self.assertTrue(getattr(cfg, suite).enabled)
        wind = [api.resolve_v3(("journal.wind.D%d" % k,)).wind for k in range(5)]
        self.assertFalse(wind[0].enabled)
        self.assertEqual([w.u10_mps for w in wind[1:]], [3.0, 6.0, 9.0, 12.0])
        self.assertTrue(all(w.gust_intensity == 0.15 and w.gust_correlation_time_s == 2.0 for w in wind[1:]))
        noise = api.resolve_v3(("journal.perception.noise.D3",)).perception
        self.assertEqual(noise.pos_noise_correlation_s, 0.2)
        a = api.resolve_v3(("journal.waves.D4",)).waves
        self.assertEqual((a.tp_s, a.gamma, a.spreading_rad, a.band_min_over_fp, a.band_max_over_fp,
                          a.n_components), (4.0, 3.3, math.pi / 4, 0.2, 4.0, 1024))

    def test_ladder_b_matches_d9_cem_values(self):
        expected = [(0.217, 1.954), (0.453, 2.497), (0.706, 2.895), (0.975, 3.225)]  # D9 DOSE_DESIGN.md:116-121
        for level, (hs, tp) in enumerate(expected, start=1):
            w = api.resolve_v3(("journal.waves.B.D%d" % level,)).waves
            self.assertAlmostEqual(w.hs_m, hs, places=3)
            self.assertAlmostEqual(w.tp_s, tp, places=3)

    def test_every_preset_resolves_builds_and_passes_admission(self):
        names = list(api.JOURNAL_PRESETS) + list(api.COMBINATIONS)
        # F0 (CONTEXT_R4 D4): + the five labelled sensor-bundle presets journal.perception.bundle.D0-D4
        # G10: + the supplementary BlueBoat boundary level journal.waves.supp075 (SUPPLEMENT, not a main level)
        self.assertEqual(len(api.JOURNAL_PRESETS), 66)
        self.assertEqual(len(api.COMBINATIONS), 6)
        for name in names:
            record = F.presets_v3.provenance(name)
            self.assertTrue(record["provenance"])
            cfg = api.resolve_v3((name,), seed=3)
            if cfg.waves.enabled and cfg.waves.hs_m > 0:
                s = F.config_v3.steepness(cfg.waves.hs_m, cfg.waves.tp_s, cfg.waves.gamma)
                self.assertTrue(s["dnv_admitted"], name)
                self.assertLessEqual(s["sp"], 1 / 15); self.assertLessEqual(s["ss"], 1 / 10)
            for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
                family = "hazard_nav"
                layer, rt = F.build((name,), n=2, family=family, vessel=vessel)
                F.reset(layer, rt)
                inputs = layer.physics_inputs(torch.zeros(2, 6), F.quaternion_yaw(torch.zeros(2)))
                rt.step(torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(2, 6), F.quaternion_yaw(torch.zeros(2)),
                        torch.zeros(2, 3), **inputs.as_kwargs())
                layer.episode_stamp(0)
        for name, entry in api.ADMISSION.items():
            if entry["hs_m"] > 0:
                self.assertTrue(entry["dnv_admitted"] and entry["generator_admitted"], name)

    def test_combinations_match_d9(self):
        expected = {
            "journal.combo.wave_wind.D2": ((0.25, 4.0), 6.0, None),
            "journal.combo.wave_wind.D3": ((0.5, 4.0), 9.0, None),
            "journal.combo.wave_current.D2": ((0.25, 4.0), None, 0.5),
            "journal.combo.wave_current.D3": ((0.5, 4.0), None, 1.0),
            "journal.combo.wave_wind_local.D2": ((0.453, 2.497), 6.0, None),
            "journal.combo.wave_wind_local.D3": ((0.706, 2.895), 9.0, None),
        }
        for name, ((hs, tp), u10, current) in expected.items():
            cfg = api.resolve_v3((name,))
            self.assertAlmostEqual(cfg.waves.hs_m, hs, places=3)
            self.assertAlmostEqual(cfg.waves.tp_s, tp, places=3)
            self.assertEqual(cfg.wind.enabled, u10 is not None)
            if u10 is not None:
                self.assertEqual(cfg.wind.u10_mps, u10)
            self.assertEqual(cfg.current.enabled, current is not None)
            if current is not None:
                self.assertEqual(cfg.current.speed_mps, current)
            self.assertEqual(cfg.presets, (name,))

    def test_combination_partner_is_co_directed_with_the_waves(self):
        layer, rt = F.build(("journal.combo.wave_current.D3",), n=16)
        axis = torch.linspace(0, 3, 16, dtype=torch.float64)
        F.reset(layer, rt, axis=axis)
        u = layer.current_velocity_world()
        flow = torch.atan2(u[:, 1], u[:, 0])
        diff = torch.remainder(flow - layer.sea.mean_direction + math.pi, 2 * math.pi) - math.pi
        self.assertLess(float(diff.abs().max()), 1e-5)


class StampsAndCertificates(unittest.TestCase):
    def legacy_env(self):
        cfg = SimpleNamespace(conditions=None, condition_presets=("current.medium",), seed=42,
                              vehicle="blueboat", sim=SimpleNamespace(dt=F.DT), decimation=2,
                              obs_kinematic=True, ray_count=36, ray_max_range_m=12.0)
        layer = api.build_conditions(cfg, 3, "cpu", family="station_keeping")
        layer.reset_runtime(env_ids=torch.arange(3), reference_axis_rad=torch.zeros(3))
        return SimpleNamespace(cfg=cfg, _conditions=layer)

    def test_schema2_stamps_stay_readable_and_schema3_round_trips(self):
        env = self.legacy_env()
        header = api.certificate_fields(env)
        self.assertEqual(header["conditions"]["schema"], "usvbench.conditions/2")
        self.assertEqual(api.config_from_stamp(header), env._conditions.cfg)
        layer, rt = F.build(("journal.combo.wave_wind.D2",), n=2)
        F.reset(layer, rt)
        stamp = layer.certificate_stamp()
        self.assertEqual(api.config_from_stamp({"conditions": stamp}), layer.cfg)
        bad = json.loads(json.dumps(stamp)); bad["schema"] = "usvbench.conditions/9"
        with self.assertRaises(ValueError):
            api.config_from_stamp({"conditions": bad})
        tampered = json.loads(json.dumps(stamp)); tampered["parameters"]["wind"]["u10_mps"] = 7.0
        with self.assertRaises(ValueError):
            api.config_from_stamp({"conditions": tampered})

    def test_evaluator_boundary_hydro_v2(self):
        layer, rt = F.build(("journal.current.D2",), n=2)
        F.reset(layer, rt)
        env = SimpleNamespace(cfg=F.env_cfg(condition_presets=("journal.current.D2",)), _conditions=layer)
        header = api.certificate_fields(env, include_off=True)["conditions"]
        self.assertEqual(header["schema"], "usvbench.conditions/3")
        env.episode_conditions = [layer.episode_stamp(0), layer.episode_stamp(1)]
        record = api.episode_certificate_fields(env, 1, include_off=True)
        self.assertEqual(record["conditions"]["parameters_sha256"], header["parameters_sha256"])
        self.assertNotIn(api.DYNAMIC_HAZARD_FIELD, record)
        no_layer = SimpleNamespace(cfg=F.env_cfg(), _conditions=None)
        with self.assertRaisesRegex(ValueError, "hydro_v2"):
            api.certificate_fields(no_layer, include_off=True)

    def test_d7_stamp_is_embedded_unchanged_b09(self):
        stamp = json.loads((FIXTURES / "d7_episode_stamp_reach_blueboat_d2.json").read_text(encoding="ascii"))
        self.assertEqual(api.dynamic_hazard_digest(stamp["payload"]), stamp["sha256"])
        env = self.legacy_env()
        env.episode_conditions = [env._conditions.episode_stamp(i) for i in range(3)]
        env.episode_dynamic_hazards = [None, stamp, None]
        env.dynamic_hazard_header = {"pool_digest": "ab" * 32, "per_dose_counts": [[1, 1]] * 5}
        record = api.episode_certificate_fields(env, 1)
        self.assertEqual(record[api.DYNAMIC_HAZARD_FIELD], stamp)            # embedded unchanged
        self.assertEqual(json.dumps(record[api.DYNAMIC_HAZARD_FIELD], sort_keys=True),
                         json.dumps(stamp, sort_keys=True))
        self.assertNotIn(api.DYNAMIC_HAZARD_FIELD, api.episode_certificate_fields(env, 0))
        header = api.certificate_fields(env)
        self.assertEqual(header[api.DYNAMIC_HAZARD_HEADER_FIELD]["pool_digest"], "ab" * 32)
        tampered = json.loads(json.dumps(stamp)); tampered["payload"]["assignment"]["dose"] = 4
        env.episode_dynamic_hazards[1] = tampered
        with self.assertRaisesRegex(ValueError, "digest"):
            api.episode_certificate_fields(env, 1)
        env.episode_dynamic_hazards[1] = {"payload": {}, "sha256": "0", "extra": 1}
        with self.assertRaises(ValueError):
            api.episode_certificate_fields(env, 1)

    def test_legacy_payload_axis_unchanged_item19(self):
        cfg = api.resolve(["actuator.payload"])
        self.assertEqual(cfg.actuator.payload_mass_kg, 4.0)
        layer = api.ConditionLayer(cfg, 2, "cpu")
        forces = torch.ones(2, 1, 3)
        out, _ = layer.dynamics(forces, torch.zeros(2, 1, 3))
        self.assertFalse(torch.equal(out, forces))
        with self.assertRaisesRegex(ValueError, "payload"):
            api.resolve_v3((), {"actuator": {"enabled": True, "payload_mass_kg": 4.0}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
