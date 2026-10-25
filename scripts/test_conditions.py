"""CPU-only condition contract tests; load by path, never import Isaac Lab."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "tasks" / "_shared"
PACKAGE = "_c4_test_shared"
parent = ModuleType(PACKAGE)
parent.__path__ = [str(SHARED)]
sys.modules[PACKAGE] = parent
spec = importlib.util.spec_from_file_location(PACKAGE + ".conditions", SHARED / "conditions" / "__init__.py",
                                             submodule_search_locations=[str(SHARED / "conditions")])
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)
current = sys.modules[PACKAGE + ".conditions.current"]
dynamic = sys.modules[PACKAGE + ".conditions.dynamic_hazards"]
perception = sys.modules[PACKAGE + ".conditions.perception"]


def env_config(**kwargs):
    values = dict(conditions=None, condition_presets=(), seed=42,
                  vehicle="blueboat", sim=SimpleNamespace(dt=1/120), decimation=2,
                  obs_kinematic=True, ray_count=36, ray_max_range_m=12.0)
    values.update(kwargs)
    return SimpleNamespace(**values)


class ConditionsTests(unittest.TestCase):
    def layer(self, presets=(), overrides=None, n=3, seed=42, **kw):
        cfg = api.resolve(presets, overrides, seed=seed)
        result = api.ConditionLayer(cfg, n, "cpu", **kw)
        result.reset_runtime(env_ids=torch.arange(n))
        return result

    def test_all_off_zero_identity_no_rng(self):
        before = torch.random.get_rng_state().clone()
        group = {"pos": perception.ObsChannelGroup(2, (0.0, 0.0), (0.0, 0.0))}
        layer = self.layer(groups=group)
        actions = torch.tensor([[0.1, -0.2], [0.0, -0.0], [-0.8, 0.9]])
        self.assertIs(layer.action_in(actions), actions)
        self.assertIs(layer.observation_out("pos", actions), actions)
        f, t = layer.environmental_wrench(actions, actions, 1/120)
        self.assertTrue(torch.equal(f, torch.zeros(3, 3)))
        self.assertTrue(torch.equal(t, torch.zeros(3, 3)))
        forces, torques = torch.ones(3, 2, 3), torch.ones(3, 2, 3)
        rf, rt = layer.dynamics(forces, torques)
        self.assertIs(rf, forces); self.assertIs(rt, torques)
        self.assertIs(layer.scale_drag(forces), forces)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertIsNone(api.build_conditions(env_config(), 3, "cpu", family="station_keeping"))
        self.assertIsNone(api.build_conditions(env_config(conditions=asdict(api.ConditionConfig())),
                                              3, "cpu", family="hazard_nav"))

    def test_current_matches_extracted_legacy_law(self):
        velocity = torch.tensor([[2.0, -3.0], [0.0, 0.0], [-0.2, 0.1]])
        flow = torch.tensor([[0.5, 0.0], [-0.5, 1.0], [0.1, 0.2]])
        relative = velocity - flow
        expected = -8.0 * relative * torch.norm(relative, dim=1, keepdim=True)
        self.assertTrue(torch.equal(current.relative_drag(velocity, flow, 8.0), expected))
        layer = self.layer(overrides={"current": {"enabled": True, "speed_mps": 0.5,
                                                 "direction_rad": 0.0, "force_mode": "legacy_relative_drag"}})
        force, torque = layer.environmental_wrench(torch.zeros(3, 2), velocity, 1/120)
        wanted = current.relative_drag(velocity, torch.tensor([[0.5, 0.0]]).expand(3, 2), 8.0)
        self.assertTrue(torch.equal(force[:, :2], wanted))
        self.assertEqual(torch.count_nonzero(torque).item(), 0)

    def test_current_increment_and_rotation(self):
        layer = self.layer(overrides={"current": {"enabled": True, "speed_mps": 0.5, "direction_rad": 0.0}})
        zero = torch.zeros(3, 2)
        f, _ = layer.environmental_wrench(zero, zero, 0.1)
        self.assertTrue(torch.equal(f[:, 0], torch.full((3,), 2.0)))
        rotated = self.layer(overrides={"current": {"enabled": True, "speed_mps": 0.5,
                                                   "direction_rad": torch.pi / 2}})
        fy, _ = rotated.environmental_wrench(zero, zero, 0.1)
        torch.testing.assert_close(fy[:, 1], f[:, 0])
        no_flow = self.layer(overrides={"current": {"enabled": True}})
        force, _ = no_flow.environmental_wrench(zero, torch.ones(3, 2), 0.1)
        self.assertEqual(torch.count_nonzero(force).item(), 0)

    def test_current_ramp_table_and_asynchronous_reset(self):
        layer = self.layer(overrides={"current": {"enabled": True, "speed_mps": 0.5,
                                                 "ramp_end_mps": 1.5, "ramp_duration_s": 2.0}})
        torch.testing.assert_close(layer.current.speed(torch.tensor([0., 1., 9.])), torch.tensor([0.5, 1., 1.5]))
        for _ in range(3):
            layer.environmental_wrench(torch.zeros(3, 2), torch.zeros(3, 2), 0.5)
        layer.reset_runtime(env_ids=[1])
        torch.testing.assert_close(layer.elapsed_s, torch.tensor([1.5, 0.0, 1.5], dtype=torch.float64))
        table = self.layer(overrides={"current": {"enabled": True,
                         "speed_table": ((0., 0.1), (2., 0.5), (4., 0.2))}})
        torch.testing.assert_close(table.current.speed(torch.tensor([-1., 1., 3., 8.])),
                                   torch.tensor([0.1, 0.3, 0.35, 0.2]))

    def test_scenario_seed_batch_and_group_independence(self):
        a = self.layer(["current.medium"])
        b = api.ConditionLayer(a.cfg, 3, "cpu")
        b.reset_runtime(env_ids=[2]); b.reset_runtime(env_ids=[0, 1])
        self.assertTrue(torch.equal(a.current.direction, b.current.direction))
        c = self.layer(["current.medium", "wave.medium"])
        self.assertTrue(torch.equal(a.current.direction, c.current.direction))
        d = self.layer(["current.medium"], seed=43)
        self.assertFalse(torch.equal(a.current.direction, d.current.direction))
        a.reset_runtime(env_ids=[1]); b.reset_runtime(env_ids=[1])
        self.assertTrue(torch.equal(a.current.direction, b.current.direction))

    def test_waves_zero_floor_and_partial_reset(self):
        zero_wave = self.layer(overrides={"waves": {"enabled": True, "hs_range": (0., 0.)}})
        f, t = zero_wave.environmental_wrench(torch.zeros(3, 2), torch.ones(3, 2), 0.1)
        self.assertEqual(torch.count_nonzero(f).item(), 0)
        self.assertEqual(torch.count_nonzero(t).item(), 0)
        before = torch.random.get_rng_state().clone()
        wave = self.layer(["wave.medium"])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        other = self.layer(["wave.medium"])
        xy, v = torch.zeros(3, 2), torch.ones(3, 2)
        wave.environmental_wrench(xy, v, 0.1)
        other.environmental_wrench(xy, v, 0.1)
        wave.reset_runtime(env_ids=[0])
        wf, wt = wave.environmental_wrench(xy, v, 0.1)
        of, ot = other.environmental_wrench(xy, v, 0.1)
        self.assertTrue(torch.equal(wf[1:], of[1:]))
        self.assertTrue(torch.equal(wt[1:], ot[1:]))
        self.assertFalse(torch.equal(wf[0], of[0]))
        self.assertTrue(torch.isfinite(wf).all())

    def test_combined_incremental_force_is_sum(self):
        both = self.layer(["current.medium", "wave.medium"])
        curr = self.layer(["current.medium"])
        wave = self.layer(["wave.medium"])
        xy, v = torch.ones(3, 2), torch.ones(3, 2) * 0.2
        bf, bt = both.environmental_wrench(xy, v, 0.1)
        cf, ct = curr.environmental_wrench(xy, v, 0.1)
        wf, wt = wave.environmental_wrench(xy, v, 0.1)
        torch.testing.assert_close(bf, cf + wf)
        torch.testing.assert_close(bt, ct + wt)

    def test_preset_full_resolution_and_round_trip(self):
        cfg = api.resolve(("wave.medium", "current.medium", "actuator.payload"),
                          {"actuator": {"payload_offset_y_m": 0.1}}, seed=9)
        original = {"task": "prototype", "records": []}
        certificate = api.stamp_conditions(original, cfg, runtime={"episode": 2})
        self.assertNotIn("conditions", original)
        self.assertEqual(api.config_from_stamp(json.loads(json.dumps(certificate))), cfg)
        self.assertEqual(set(certificate["conditions"]["parameters"]), set(asdict(cfg)))
        tampered = copy.deepcopy(certificate)
        tampered["conditions"]["parameters"]["current"]["speed_mps"] = 99.0
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            api.config_from_stamp(tampered)
        layer = self.layer(["current.medium", "wave.medium"])
        stamp = layer.episode_stamp(0)
        self.assertEqual(len(stamp["runtime"]["wave_realization"]["phase"]), 104)
        old = copy.deepcopy(stamp)
        layer.reset_runtime(env_ids=[0])
        self.assertEqual(stamp, old)
        self.assertEqual(layer.episode_stamp(0)["runtime"]["episode_index"], 1)

    def test_preset_and_numeric_rejections(self):
        for presets, overrides in [
            (["missing"], None), (["wave.mild", "wave.heavy"], None),
            ([], {"curent": {}}), ([], {"current": {"speed": 1}}),
            ([], {"current": {"speed_mps": float("nan")}}),
            ([], {"current": {"direction_rad": "north"}}),
            ([], {"current": {"speed_mps": True}}),
            ([], {"current": {"drag_coeff": None}}),
            ([], {"current": {"speed_table": [[2, 1], [1, 2]]}}),
            ([], {"current": {"speed_table": [[0, 1]], "ramp_end_mps": 1}}),
            ([], {"actuator": {"mass_scale": 0}}),
            ([], {"actuator": {"thrust_imbalance": 2}}),
            ([], {"perception": {"dropout_p": 2}}),
            ([], {"perception": {"delay_steps": 1.2}}),
            ([], {"waves": {"n_components": 0}}),
            ([], {"seed": True}),
            ([], {"schema_version": True}),
        ]:
            with self.subTest(presets=presets, overrides=overrides):
                with self.assertRaises(ValueError):
                    api.resolve(presets, overrides)

    def test_stubs_fail_closed_and_dynamic_progress_contract(self):
        # D1: wind is no longer a stub (C3 backend). Dynamic hazards still
        # fail closed; wind fails closed for vessels without a usable
        # windage profile (legacy "wamv" has none; "wamv_vrx" is a D5
        # placeholder) and works for the provisional BlueBoat profile.
        with self.assertRaises(NotImplementedError):
            self.layer(overrides={"dynamic_hazards": {"enabled": True}})
        for vehicle in ("wamv", "wamv_vrx", "rov"):
            with self.assertRaises(ValueError):
                self.layer(overrides={"wind": {"enabled": True, "speed_mps": 5.0}},
                           context={"vehicle": vehicle})
        wind = self.layer(overrides={"wind": {"enabled": True, "speed_mps": 5.0}})
        self.assertEqual(wind.cfg.wind.windage_profile, "blueboat_hull_only_c3")
        class Adapter:
            def __init__(self): self.events = []
            def reset(self, ids, cfg, scenario): self.events.append(("reset", ids.tolist()))
            def activate(self, ids, cfg): self.events.append(("activate", ids.tolist()))
            def update(self, progress, cfg): self.events.append(("update", progress.tolist()))
        adapter = Adapter()
        layer = self.layer(overrides={"dynamic_hazards": {"enabled": True}}, hazard_adapter=adapter)
        layer.reset_runtime(progress=torch.tensor([0.2, 0.5, 0.9]))
        layer.reset_runtime(progress=torch.tensor([0.8, 0.6, 1.0]))
        self.assertEqual([event for event in adapter.events if event[0] == "activate"],
                         [("activate", [1, 2]), ("activate", [0])])
        layer.reset_runtime(env_ids=[2])
        self.assertFalse(layer.hazards.triggered[2])
        self.assertTrue(layer.hazards.triggered[0])
        with self.assertRaises(ValueError): layer.reset_runtime(progress=None)
        with self.assertRaises(ValueError): layer.reset_runtime(progress=torch.ones(3) * 2)
        self.assertEqual(len(dynamic.TASK_DEFINITIONS), 12)

    def test_factory_groups_and_legacy_overlap_validation(self):
        for family in ("station_keeping", "hazard_nav"):
            layer = api.build_conditions(env_config(condition_presets=("perception.noise",)), 3, "cpu", family=family)
            self.assertTrue(layer.wants("pos")); self.assertTrue(layer.wants("vel"))
            layer.reset_runtime(env_ids=[0, 1, 2])
            x = torch.zeros(3, 2)
            layer.observation_out("pos", x)
            self.assertFalse(torch.equal(layer.observation_out("pos", x), x))
        cases = [
            env_config(condition_presets=("current.medium",), underwater_physics_cfg=SimpleNamespace(enable_current=True)),
            env_config(condition_presets=("wave.medium",), sea_state=SimpleNamespace(enable=True)),
            env_config(condition_presets=("actuator.mass",), mass_scale=2),
            env_config(condition_presets=("actuator.mass",), vehicle="wamv"),
            env_config(condition_presets=("perception.noise",), obs_noise_sigma_pos=0.1),
            env_config(condition_presets=("perception.noise",), obs_kinematic=False),
        ]
        for cfg in cases:
            with self.assertRaises(ValueError):
                api.build_conditions(cfg, 3, "cpu", family="station_keeping")

    def test_reset_validation_and_clock_cadence(self):
        layer = self.layer(["current.ramp"])
        for _ in range(2):
            layer.action_in(torch.zeros(3, 2))
            for _ in range(2):
                layer.environmental_wrench(torch.zeros(3, 2), torch.zeros(3, 2), 1/120)
        torch.testing.assert_close(layer.elapsed_s, torch.full((3,), 4/120, dtype=torch.float64))
        for ids in ([0, 0], [-1], [3]):
            with self.assertRaises(ValueError): layer.reset_runtime(env_ids=ids)
        with self.assertRaises(ValueError):
            layer.environmental_wrench(torch.zeros(3, 2), torch.zeros(3, 2), 0)

    def test_direct_mapping_and_missing_perception_groups(self):
        layer = api.ConditionLayer({"current": {"enabled": True}}, 3, "cpu")
        self.assertTrue(layer.cfg.current.enabled)
        with self.assertRaises(ValueError):
            api.ConditionLayer(api.resolve(["perception.noise"]), 3, "cpu")
        with self.assertRaises(ValueError):
            api.build_conditions(env_config(conditions=api.resolve(["actuator.mass"]), vehicle="wamv"),
                                 3, "cpu", family="hazard_nav")
        with self.assertRaises(ValueError):
            api.build_conditions(env_config(condition_presets=("current.medium",), motor_tau_s=1.0),
                                 3, "cpu", family="hazard_nav")

    def test_certificate_evaluator_fields(self):
        env = SimpleNamespace(cfg=env_config(), _conditions=None)
        self.assertEqual(api.certificate_fields(env), {})
        env.cfg.conditions = {}
        off = api.certificate_fields(env)
        self.assertFalse(api.config_from_stamp(off).any_enabled)
        env._conditions = self.layer(["current.medium", "wave.medium"])
        header = api.certificate_fields(env)
        env.episode_conditions = [env._conditions.episode_stamp(i) for i in range(3)]
        env._conditions.reset_runtime(env_ids=[0])
        record = api.episode_certificate_fields(env, 0)
        self.assertEqual(record["conditions"]["runtime"]["episode_index"], 0)
        self.assertEqual(api.config_from_stamp(header), api.config_from_stamp(record))
        sensor = api.build_conditions(env_config(condition_presets=("perception.bias",)),
                                      3, "cpu", family="station_keeping")
        sensor.reset_runtime(env_ids=[0, 1, 2])
        self.assertIsNone(sensor.episode_stamp(0)["runtime"]["perception_realization"]["pos"]["bias"])
        sensor.observation_out("pos", torch.zeros(3, 2))
        captured = sensor.episode_stamp(0)["runtime"]["perception_realization"]
        self.assertEqual(len(captured["pos"]["bias"]), 2)
        self.assertEqual(captured["pos"]["frames"], 1)

    def test_each_environment_module_off(self):
        layer = self.layer()
        xy, velocity = torch.ones(3, 2), torch.ones(3, 2) * 7
        for module in (layer.waves, layer.wind):
            f, t = module.wrench(xy, velocity, layer.elapsed_s)
            self.assertEqual(torch.count_nonzero(f).item(), 0)
            self.assertEqual(torch.count_nonzero(t).item(), 0)
        f, t = layer.current.wrench(velocity, layer.elapsed_s)
        self.assertEqual(torch.count_nonzero(f).item(), 0)
        self.assertEqual(torch.count_nonzero(t).item(), 0)
        layer.hazards.runtime(None)

    def test_cli_wiring_and_ascii_sources(self):
        import ast
        import tempfile
        spec = importlib.util.spec_from_file_location("_c4_condition_args", ROOT / "scripts/condition_args.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = env_config()
        mod.apply_condition_args(cfg)
        self.assertIsNone(cfg.conditions)
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            path = Path(directory) / "conditions.json"
            path.write_text('{"current":{"direction_rad":0.0}}', encoding="ascii")
            mod.apply_condition_args(cfg, ["current.medium"], path)
        self.assertEqual(api.resolve(cfg.condition_presets, cfg.conditions).current.direction_rad, 0.0)
        with self.assertRaises(ValueError):
            mod.apply_condition_args(SimpleNamespace(), ["current.medium"])
        evaluator = ast.parse((ROOT / "scripts/eval_v6_frozen.py").read_text(encoding="utf-8"))
        calls = [node.func.id for node in ast.walk(evaluator)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        for name in ("apply_condition_args", "certificate_fields", "episode_certificate_fields"):
            self.assertIn(name, calls)
        for path in (SHARED / "conditions").rglob("*.py"):
            path.read_bytes().decode("ascii")
        self.assertFalse(any(name.startswith("isaaclab") for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
