"""CPU regressions for the classical certificate loop; never import Isaac."""
import ast
import importlib.util
import itertools
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cert_classical_test_helper", ROOT / "cert_v3.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def _mission_loop(episodes=4, max_episode_length=3, completion_periods=(1, 3)):
    source = ROOT / "classical_baseline.py"
    tree = ast.parse(source.read_text(encoding="ascii"), filename=str(source))
    wanted = {"_run_mission", "_initial_distances", "_values_at", "_completed_path_lengths"}
    definitions = [node for node in tree.body if getattr(node, "name", None) in wanted]
    assert {node.name for node in definitions} == wanted
    namespace = dict(
        torch=torch, math=math, statistics=statistics,
        args_cli=SimpleNamespace(episodes=episodes, num_envs=2),
        episode_scenario_hashes_for=lambda base, env: None,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    base = SimpleNamespace(
        num_envs=2, device="cpu", max_episode_length=max_episode_length,
        episode_success=torch.tensor([True, False]),
        episode_path_length=torch.tensor([4.0, 9.0]),
        time_to_success=torch.tensor([1.0, math.nan]),
        d0_per_env=torch.tensor([10.0, 20.0]),
        termination_reason=torch.tensor([1, 3]),
    )
    base._horizontal_distance = lambda: base.d0_per_env.clone()

    class FakeEnv:
        def __init__(self):
            self.steps = 0

        def step(self, actions):
            self.steps += 1
            terminated = torch.tensor([self.steps % completion_periods[0] == 0, False])
            truncated = torch.tensor([False, self.steps % completion_periods[1] == 0])
            # Autoreset replaces the initial distance before returning.
            base.d0_per_env[terminated | truncated] += 100.0
            return None, None, terminated, truncated, {}

    env = FakeEnv()
    controller = SimpleNamespace(act=lambda obs: None, reset=lambda mask: None)
    ticks = itertools.count()
    cert = helper.CertificateRun(
        base=base, env_cfg=SimpleNamespace(), evaluator=str(source), task="fake",
        controller="fake", controller_config={}, checkpoint=None,
        eval_seed=42, level=0, episodes_requested=episodes, overrides={},
        argv=[], cli_args={}, provenance={}, runtime={},
        synchronize=lambda: None, clock=lambda: next(ticks) / 1000.0,
    )
    metrics, records = namespace["_run_mission"](
        env, base, controller, "station_keep", 2.0, "cpu", None, cert
    )
    return metrics, records, env, cert


def test_async_mission_uses_equal_per_env_quota_and_latched_distances():
    metrics, records, env, cert = _mission_loop()
    assert env.steps == 6
    assert metrics["episodes"] == 4
    assert metrics["episodes_requested"] == 4
    assert metrics["episodes_per_env"] == 2
    assert metrics["complete"] is True
    assert [(r["env"], r["ep"]) for r in records] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [r["d0_m"] for r in records] == [10.0, 110.0, 20.0, 120.0]
    assert [r["termination_reason"] for r in records] == ["success", "success", "timeout", "timeout"]
    assert [(r["terminated"], r["truncated"]) for r in records] == [
        (True, False), (True, False), (False, True), (False, True)
    ]
    assert [r["controller_compute_time_s"]["n"] for r in records] == [1, 1, 3, 3]
    assert [r["environment_step_time_s"]["n"] for r in records] == [1, 1, 3, 3]
    assert cert.build({"records": records})["header"]["complete"] is True


def test_step_budget_preserves_incomplete_grid_instead_of_pooled_success():
    metrics, records, env, cert = _mission_loop(episodes=4, max_episode_length=1)
    assert env.steps == 5
    assert metrics["episodes"] == 3
    assert metrics["complete"] is False
    assert [(r["env"], r["ep"]) for r in records] == [(0, 0), (0, 1), (1, 0)]
    assert cert.build({"records": records})["header"]["complete"] is False


def test_classical_throughput_loop_keeps_fixed_step_protocol():
    tree = ast.parse((ROOT / "classical_baseline.py").read_text(encoding="ascii"))
    loop = next(node for node in tree.body if getattr(node, "name", None) == "_run_throughput")
    body = ast.unparse(loop)
    assert "range(args_cli.eval_steps)" in body
    assert "cert." not in body


def test_zero_completions_still_produces_an_incomplete_certificate():
    metrics, records, env, cert = _mission_loop(
        episodes=4, max_episode_length=1, completion_periods=(100, 100)
    )
    assert env.steps == 5
    assert records == []
    assert metrics["episodes"] == 0
    assert math.isnan(metrics["sr"])
    assert cert.build({"records": records})["header"]["complete"] is False


def test_planner_nested_overrides_use_shared_coercion():
    spec = importlib.util.spec_from_file_location("planner_cfg_test", ROOT / "classical_planner_pid.py")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    args = planner._build_parser().parse_args([
        "--set", "sea_state.hs_range=0.6,0.6", "--set", "enabled=false"
    ])
    cfg = SimpleNamespace(sea_state=SimpleNamespace(hs_range=(0.1, 0.2)), enabled=True)
    applied = dict(planner.apply_overrides(cfg, args.extra_sets, echo=False))
    assert cfg.sea_state.hs_range == (0.6, 0.6)
    assert cfg.enabled is False
    assert applied == {"sea_state.hs_range": (0.6, 0.6), "enabled": False}


def test_writer_sources_are_ascii_and_parse_without_simulator_import():
    for name in ("eval_mppi.py", "classical_planner_pid.py", "classical_baseline.py"):
        source = ROOT / name
        assert source.read_bytes().isascii()
        ast.parse(source.read_text(encoding="ascii"), filename=str(source))
