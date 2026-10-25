"""D1 item d: certificate schema 3.1 condition stamps and termination reason.

CPU only; never imports Isaac Lab. Run: python -B scripts/test_cert_v31_conditions.py
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
for path in (str(SCRIPTS), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cert = load("cert_v3")
validator = load("validate_cert_v3")
import tasks._shared.conditions as conditions  # noqa: E402  (pure python + torch)
from tasks._shared.conditions import stamp as stamp_module  # noqa: E402

WRITERS = ("eval_v6_frozen.py", "eval_mppi.py", "classical_planner_pid.py",
           "classical_baseline.py", "eval_obs_bridge.py", "eval_imbalance.py")


def env_cfg(**kwargs):
    values = dict(seed=42, eval_level=0, curriculum_frozen=True, vehicle="blueboat",
                  scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(dt=1 / 120), decimation=2,
                  underwater_physics_cfg=SimpleNamespace(physics_version=2, enable_current=False),
                  journal_mode=True, conditions=None, condition_presets=(), obs_kinematic=True)
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_run(base, config, requested=2):
    provenance = {key: "a" * 64 for key in ("tasks_sha256", "scripts_sha256", "tree_sha256", "evaluator_sha256")}
    provenance.update(tasks_root="t/tasks", scripts_root="t/scripts", environment_source="t/tasks/f/env.py",
                      evaluator_path="t/scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
    # E1: synthetic legacy digests must be self-consistent (validator recomputes tree_sha256, D2 item 10).
    provenance["tree_sha256"] = cert.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {key: "test" for key in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                      "isaac_lab_version", "physx_version")}
    counter = iter(i / 1000 for i in range(10000))
    return cert.CertificateRun(
        base=base, env_cfg=config, evaluator=__file__, task="Test-Journal", controller="test",
        controller_config={}, checkpoint=None, eval_seed=42, level=0, episodes_requested=requested,
        overrides={}, argv=["eval.py"], cli_args={}, provenance=provenance, runtime=runtime,
        synchronize=lambda: None, clock=lambda: next(counter))


def record(env):
    return dict(env=env, ep=0, success=False, terminated=False, truncated=True, tts_s=None,
                path_length_m=1.0, min_clearance_m=None, d0_m=2.0, route_geodesic_m=None)


def step(run):
    with run.time_controller():
        pass
    with run.time_environment():
        pass
    run.after_step()


def layer_document():
    config = env_cfg(conditions={"current": {"enabled": True, "speed_mps": 0.4, "direction_rad": 0.1},
                                 "wind": {"enabled": True, "speed_mps": 6.0}})
    layer = conditions.build_conditions(config, 2, "cpu", family="station_keeping")
    layer.reset_runtime(env_ids=torch.arange(2), reference_axis_rad=torch.zeros(2))
    base = SimpleNamespace(num_envs=2, device="cpu", termination_reason=[3, 3], _conditions=layer,
                           episode_conditions=[{}, {}], cfg=config)
    run = make_run(base, config)
    for _ in range(3):
        layer.environmental_wrench(torch.zeros(2, 2), torch.ones(2, 2), 1 / 120, heading_rad=torch.zeros(2))
    # The env latches the finished episode before its autoreset (C4).
    base.episode_conditions = [layer.episode_stamp(i) for i in range(2)]
    layer.reset_runtime(env_ids=torch.arange(2), reference_axis_rad=torch.zeros(2))
    step(run)
    records = [run.finish_episode(i, record(i)) for i in range(2)]
    return run.build({"records": records, **conditions.certificate_fields(base)}), base, layer


def off_document():
    config = SimpleNamespace(seed=7, eval_level=0, vehicle="blueboat", scene=SimpleNamespace(num_envs=2),
                             underwater_physics_cfg=SimpleNamespace(physics_version=1))
    base = SimpleNamespace(num_envs=2, device="cpu", termination_reason=[3, 3])
    run = make_run(base, config)
    step(run)
    return run.build({"records": [run.finish_episode(i, record(i)) for i in range(2)]})


def errors(document):
    return validator.validate_certificate(document, num_envs=2, episodes_per_env=1)


class ConditionStampTests(unittest.TestCase):
    def test_layer_header_and_finished_episode_stamps(self):
        document, base, layer = layer_document()
        self.assertEqual(errors(document), [])
        self.assertEqual(document["schema_version"], 3)
        # E1: the writer emits schema 3.2 (3.1 condition stamps + E1 fields).
        self.assertEqual(document["schema_minor_version"], 2)
        header = document["header"]["conditions"]
        self.assertEqual(header["schema"], "usvbench.conditions/2")
        self.assertEqual(header, document["conditions"])  # C4 legacy alias
        self.assertEqual(document["header"]["physics_version"], "2")
        self.assertFalse(any("physics_version" in note for note in document["header"]["unverified"]))
        for i, rec in enumerate(document["records"]):
            runtime = rec["conditions"]["runtime"]
            self.assertEqual(runtime["env_index"], i)
            self.assertEqual(runtime["episode_index"], 0)  # finished episode, not the new one
            self.assertEqual(runtime["clock_ticks"], 3)
            self.assertEqual(runtime["wind_realization"]["stream_key"]["episode_index"], 0)
            self.assertEqual(rec["conditions"]["parameters_sha256"], header["parameters_sha256"])
        self.assertEqual(int(layer.episode_index[0]), 1)  # the env already runs episode 1
        wind = header["context"]["wind_model"]
        self.assertIn("requested_speed_mps", wind)
        self.assertIn("applied_vessel_height_speed_mps", wind)
        json.dumps(document, allow_nan=False)

    def test_legacy_env_gets_explicit_off_stamps(self):
        document = off_document()
        self.assertEqual(errors(document), [])
        header = document["header"]["conditions"]
        self.assertFalse(header["context"]["layer_active"])
        self.assertEqual(header["context"]["physics_version"], 1)
        self.assertEqual(document["header"]["physics_version"], "1")
        self.assertFalse(conditions.config_from_stamp({"conditions": header}).any_enabled)
        for i, rec in enumerate(document["records"]):
            self.assertEqual(rec["conditions"]["runtime"], {"layer_active": False, "env_index": i})

    def test_validator_rejects_tampering_and_inconsistency(self):
        base_document, _, _ = layer_document()

        def restamp(block):
            block["parameters_sha256"] = validator.condition_digest(block["parameters"])

        cases = [
            (lambda d: d["header"]["conditions"]["parameters"]["wind"].update(speed_mps=9.0),
             "header condition digest"),
            (lambda d: d["records"][0]["conditions"]["parameters"]["current"].update(speed_mps=1.0),
             "records/0: condition digest"),
            (lambda d: (d["records"][1]["conditions"]["parameters"]["current"].update(speed_mps=1.0),
                        restamp(d["records"][1]["conditions"])), "differ from the header"),
            (lambda d: d["records"][0]["conditions"]["runtime"].update(env_index=1), "env_index"),
            (lambda d: d["conditions"].update(dose_status="calibrated"), "legacy alias"),
            (lambda d: d["header"].update(physics_version="1"), "physics_version"),
            (lambda d: d["records"][0].pop("conditions"), "schema"),
            (lambda d: d["header"].pop("conditions"), "schema"),
            (lambda d: d["header"]["conditions"].update(schema="usvbench.conditions/1"), "schema"),
            (lambda d: d.update(schema_minor_version=3), "schema"),  # E1: 2 is now a known minor
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                document = copy.deepcopy(base_document)
                mutate(document)
                found = errors(document)
                self.assertTrue(any(message in error for error in found), found)

    def test_schema_3_0_documents_remain_valid(self):
        document = off_document()
        document.pop("schema_minor_version")
        document["header"].pop("conditions")
        for rec in document["records"]:
            rec.pop("conditions")
        self.assertEqual(errors(document), [])

    def test_validator_digest_equals_stamp_digest(self):
        cfg = conditions.resolve(["current.medium"], {"wind": {"enabled": True, "speed_mps": 3.0}}, seed=3)
        block = conditions.stamp_conditions({}, cfg)["conditions"]
        self.assertEqual(validator.condition_digest(block["parameters"]), block["parameters_sha256"])
        self.assertEqual(stamp_module.parameters_digest(block["parameters"]), block["parameters_sha256"])

    def test_writer_supplied_stamp_must_equal_latch(self):
        _, base, _ = layer_document()
        run = make_run(base, base.cfg)
        step(run)
        rec = record(0)
        rec.update(conditions.episode_certificate_fields(base, 0))  # eval_v6_frozen (C4) path
        self.assertEqual(run.finish_episode(0, rec)["conditions"], base.episode_conditions[0])
        step(run)
        rec = record(1)
        rec["conditions"] = copy.deepcopy(base.episode_conditions[0])
        with self.assertRaises(ValueError):
            run.finish_episode(1, rec)

    def test_active_layer_without_latch_is_an_error(self):
        _, base, _ = layer_document()
        base.episode_conditions = [{}, {}]
        run = make_run(base, base.cfg)
        step(run)
        with self.assertRaises(ValueError):
            run.finish_episode(0, record(0))


def _function(tree, name):
    return next(n for c in tree.body if isinstance(c, ast.ClassDef)
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)


class TerminationAndWriterTests(unittest.TestCase):
    def test_termination_reason_is_published_last_in_every_get_dones(self):
        paths = sorted((REPO / "tasks").glob("*/*env.py"))
        checked = 0
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            try:
                dones = _function(tree, "_get_dones")
            except StopIteration:
                continue
            with self.subTest(env=path.parent.name):
                publish = [n for n in ast.walk(dones) if isinstance(n, ast.Call)
                           and ast.unparse(n.func) == "publish_termination_reason"]
                returns = [n for n in ast.walk(dones) if isinstance(n, ast.Return)]
                self.assertEqual(len(publish), 1)
                self.assertEqual(len(returns), 1)
                last, final = dones.body[-2], dones.body[-1]
                self.assertIsInstance(final, ast.Return)
                self.assertTrue(isinstance(last, ast.Expr) and last.value is publish[0])
                returned = [ast.unparse(e) for e in final.value.elts]
                self.assertEqual([ast.unparse(a) for a in publish[0].args[1:3]], returned)  # args[0] is self
                checked += 1
        self.assertEqual(checked, 10)

    def test_every_writer_certifies_through_certificate_run(self):
        for name in WRITERS:
            with self.subTest(writer=name):
                tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
                calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
                self.assertIn("CertificateRun", calls)
                self.assertTrue(any(call.endswith("finish_episode") for call in calls))
                self.assertTrue(any(call.endswith(".build") for call in calls))

    def test_ppo_evaluator_keeps_c4_and_c5_paths(self):
        tree = ast.parse((SCRIPTS / "eval_v6_frozen.py").read_text(encoding="utf-8"))
        source = ast.unparse(tree)
        for fragment in ("apply_condition_args(env_cfg, args_cli.condition_preset, args_cli.condition_config)",
                         "rec.update(episode_certificate_fields(base, i))",
                         "records.append(cert.finish_episode(i, rec))",
                         "**certificate_fields(base)", "allow_nan=False"):
            self.assertIn(fragment, source)
        self.assertLess(source.index("rec.update(episode_certificate_fields(base, i))"),
                        source.index("records.append(cert.finish_episode(i, rec))"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
