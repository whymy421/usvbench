"""D8 takeover (attempt 4) regression tests: CPU only, no Isaac import.

Run: python -B -m pytest -p no:cacheprovider repo/scripts/test_d8_takeover.py -q
"""
import ast
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cert = load("cert_v3")
protocol = load("eval_protocol")
intervals = load("paired_intervals")
seeds = load("seed_registry")
inputs = load("analysis_input")
fixtures = load("test_cert_v3")

MODULE_LEVEL_WRITERS = ("eval_v6_frozen.py", "eval_obs_bridge.py", "eval_imbalance.py",
                        "eval_composite.py", "convergence_curve.py", "eval_hazard_nav.py",
                        "eval_cross_champion.py", "classical_baseline.py")
FUNCTION_LEVEL_WRITERS = ("eval_mppi.py", "classical_planner_pid.py")
NEW_OR_CHANGED = ("analysis_input.py", "cert_v3.py", "crossmachine_sample.py", "episode_path_observer.py",
                  "eval_protocol.py", "fixed_grid.py", "paired_intervals.py", "run_cell.py",
                  "seed_registry.py", "paired_dose_test.py", "usv10k_score.py", "deployment_metrics.py",
                  "cpi_verdict.py", "mppi_controller.py", "test_d8_takeover.py",
                  "test_run_cell_v3plus.py") + MODULE_LEVEL_WRITERS + FUNCTION_LEVEL_WRITERS


# ------------------------------------------------------------------ hygiene

@pytest.mark.parametrize("name", NEW_OR_CHANGED + ("seed_registry.json", "cert_v3.schema.json"))
def test_ascii_and_parseable(name):
    data = (SCRIPTS / name).read_bytes()
    assert all(b in (9, 10, 13) or 32 <= b <= 126 for b in data), name
    if name.endswith(".py"):
        ast.parse(data.decode("ascii"), filename=name)


# ------------------------------------------------------------------ every writer's preflight

def _module_top(target):
    source = (SCRIPTS / target).read_text(encoding="utf-8")
    marker = next(m for m in ("app = AppLauncher(args_cli).app", "app_launcher = AppLauncher(args_cli)")
                  if m in source)
    return source[:source.index(marker)]


def run_top(target, argv, environ=None, capture=None):
    """Execute a writer up to its AppLauncher() with a stubbed isaaclab; exit code or -1."""
    if capture is not None:
        import contextlib
        import io
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            code = run_top(target, argv, environ)
        capture.append(stream.getvalue())
        return code
    stub_app = ModuleType("isaaclab.app")

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--device", type=str, default=None)

    stub_app.AppLauncher = AppLauncher
    stub_root = ModuleType("isaaclab")
    stub_root.app = stub_app
    saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.app")}
    saved_argv = sys.argv
    sys.modules["isaaclab"], sys.modules["isaaclab.app"] = stub_root, stub_app
    sys.argv = [str(SCRIPTS / target)] + argv
    try:
        with patch.dict("os.environ", environ or {}, clear=True):
            exec(compile(_module_top(target), target, "exec"),
                 {"__name__": "__smoke__", "__file__": str(SCRIPTS / target)})
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved_argv
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return -1


def _required_args(target):
    here = str(SCRIPTS / "fixed_grid.py")  # an existing file for checkpoint-like flags
    return {
        "eval_v6_frozen.py": ["--checkpoint", here],
        "eval_obs_bridge.py": ["--checkpoint", here, "--train-task",
                               "Isaac-USV-StationKeep-BlueBoat-Kin-Direct-v1", "--eval-task",
                               "Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1"],
        "eval_imbalance.py": ["--checkpoint", here, "--imbalance", "0.1"],
        "eval_composite.py": ["--nav-ckpt", here, "--nav-id", "a", "--dock-ckpt", here, "--dock-id", "b"],
        "convergence_curve.py": ["--run-dir", str(SCRIPTS)],
        "eval_hazard_nav.py": ["--checkpoint", here],
        "eval_cross_champion.py": ["--champion-checkpoint", here,
                                   "--target-task", "Isaac-USV-HazardNav-Direct-v1",
                                   "--champion-task", "Isaac-USV-HazardNav-Direct-v1", "--level", "0"],
        "classical_baseline.py": ["--task", "Isaac-USV-StationKeep-BlueBoat-Direct-v1",
                                  "--protocol", "mission"],
    }[target]


def _valid_argv(target):
    argv = _required_args(target) + ["--num_envs", "64", "--episodes", "128"]
    if target == "convergence_curve.py":
        argv += ["--out", "curve.json"]
    return argv


@pytest.mark.parametrize("target", MODULE_LEVEL_WRITERS)
def test_every_module_writer_valid_call_reaches_launch(target):
    """Positive control: without the bad argument the same call parses, passes
    preflight and stops only at the stubbed AppLauncher (exit code -1)."""
    stderr = []
    assert run_top(target, _valid_argv(target), capture=stderr) == -1, stderr


@pytest.mark.parametrize("target", MODULE_LEVEL_WRITERS)
@pytest.mark.parametrize("bad,environ,reason", [
    (["--episodes", "65"], None, "divisible"),                        # finding 57
    (["--set", "vehicle=wamv"], None, "select the vessel by task id"),  # finding 46
    (["--set", "seed=7"], None, "conflicts with protocol value"),      # finding 53
    ([], {"GOAL_RADIUS_OVERRIDE": "3"}, "require explicit --allow-env"),  # finding 61
    (["--allow-env", "NOT_A_READ_NAME"], None, "unknown --allow-env"),
    (["--seed-namespace", "final_test", "--run-purpose", "final_test",
      "--seed-block", "final-01", "<SEED>", "41041"], None, "must be planned by run_cell.py"),
    (["--seed-namespace", "final_test", "--run-purpose", "final_test",
      "--seed-block", "final-01"], None, "seed does not match the declared seed block"),
])
def test_every_module_writer_refuses_before_launch(target, bad, environ, reason):
    argv = _valid_argv(target) + [_seed_flag(target) if item == "<SEED>" else item for item in bad]
    stderr = []
    assert run_top(target, argv, environ, capture=stderr) == 2
    assert reason in stderr[0], stderr[0][-400:]


def _seed_flag(target):
    return "--seed" if target in ("eval_hazard_nav.py", "eval_cross_champion.py",
                                  "classical_baseline.py") else "--eval-seed"


@pytest.mark.parametrize("target", MODULE_LEVEL_WRITERS)
def test_registered_eval_seed_needs_its_namespace(target):
    stderr = []
    assert run_top(target, _valid_argv(target) + [_seed_flag(target), "41041"], capture=stderr) == 2
    assert "belongs to the final_test namespace" in stderr[0], stderr[0][-400:]


@pytest.mark.parametrize("target", FUNCTION_LEVEL_WRITERS)
def test_function_writers_preflight_with_parser_before_launch(target):
    tree = ast.parse((SCRIPTS / target).read_text(encoding="utf-8"))
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_isaac_eval")
    calls = [n for n in ast.walk(body) if isinstance(n, ast.Call)]
    pre = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "preflight_protocol"]
    launch = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "AppLauncher"]
    assert len(pre) == 1 and len(pre[0].args) == 3 and pre[0].args[2].id == "parser"
    assert launch and pre[0].lineno < min(n.lineno for n in launch)


# ------------------------------------------------------------------ override precedence

def cfg():
    return SimpleNamespace(seed=0, eval_level=0, curriculum_frozen=False, drag_scale=1.0,
                           gate_width_override_m=1.5, vehicle="blueboat",
                           scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device="cpu"))


def args(**extra):
    values = dict(eval_seed=5, level=0, num_envs=2, episodes=4, extra_sets=[], device="cpu")
    values.update(extra)
    return SimpleNamespace(**values)


def test_dedicated_flag_and_set_conflict_rejected_equal_accepted():
    with pytest.raises(ValueError, match="dedicated flag"):
        protocol.resolve_protocol(cfg(), args(gate_width_override=1.2,
                                              extra_sets=["gate_width_override_m=1.3"]))
    applied = protocol.resolve_protocol(cfg(), args(gate_width_override=1.2,
                                                    extra_sets=["gate_width_override_m=1.2"]))
    assert applied["gate_width_override_m"] == 1.2 and applied["curriculum_frozen"] is True
    with pytest.raises(ValueError, match="vessel"):
        protocol.resolve_protocol(cfg(), args(extra_sets=["vehicle=wamv"]))
    with pytest.raises(ValueError, match="evaluation level"):
        protocol.resolve_protocol(SimpleNamespace(seed=0, drag_scale=1.0, scene=SimpleNamespace(num_envs=2),
                                                  sim=SimpleNamespace(device="cpu")), args(level=2))


def test_runtime_selectors_stamped_not_refused_and_hydra_refused():
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4"}, clear=True):
        a = args(allow_env=[], seed_namespace=None)
        protocol.preflight_protocol(a, ["eval.py"])
        assert a.runtime_environment == {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4"}
        assert a.scientific_environment == {} and a.seed_registration["purpose"] == "diagnostic"
        with pytest.raises(ValueError, match="Hydra"):
            protocol.preflight_protocol(args(allow_env=[]), ["eval.py", "env.drag_scale=2"])
    names = protocol.scientific_environment_names()
    assert set(protocol.KNOWN_SCIENTIFIC_ENV) <= names and "OBS_DIM" in names


# ------------------------------------------------------------------ certificate additions

def run_with(base_extra=None, cfg_extra=None, **kwargs):
    base = SimpleNamespace(num_envs=2, device="cpu", termination_reason=[3, 3],
                           control_step_s=1 / 60, max_episode_length=7200)
    for key, value in (base_extra or {}).items():
        setattr(base, key, value)
    config = SimpleNamespace(seed=42, eval_level=0, curriculum_frozen=True, vehicle="blueboat",
                             scene=SimpleNamespace(num_envs=2))
    for key, value in (cfg_extra or {}).items():
        setattr(config, key, value)
    return fixtures.make_run(2, 4, base=base, env_cfg=config, overrides={}, argv=["eval.py"], **kwargs)


def test_shared_grid_collector_keeps_slow_episode_and_discards_fast_extras():
    run = run_with()
    order = [0, 0, 0, 1, 0, 1]  # env 0 finishes fast, env 1 slowly
    kept = []
    for env in order:
        fixtures.step(run)
        episode = run.next_episode(env)
        if episode is not None:
            kept.append(run.finish_episode(env, fixtures.sample_record(env, episode)))
    assert sorted((r["env"], r["ep"]) for r in kept) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert run.grid.complete
    with pytest.raises(ValueError, match="next assigned grid key"):
        run.finish_episode(0, fixtures.sample_record(0, 2))
    with pytest.raises(ValueError):
        run.next_episode(5)


def test_speed_reference_from_vessel_spec():
    vehicles = load_vehicles()
    blueboat = vehicles.get_vehicle("blueboat")
    nominal = cert.nominal_speed_reference(blueboat)
    assert nominal["value_m_s"] == pytest.approx(2.991, abs=5e-4)  # legacy V_MAX
    run = run_with(base_extra={"vehicle_spec": blueboat}, cli_args={})
    reference = run.header["speed_reference"]
    assert reference["basis"] == "nominal" and reference["vessel"] == "blueboat"
    assert reference["value_m_s"] == pytest.approx(nominal["value_m_s"])
    with pytest.raises(ValueError, match="condition_adjusted"):
        run_with(base_extra={"vehicle_spec": blueboat},
                 cli_args={"speed_reference_m_s": 1.5, "speed_reference_basis": "nominal"})
    adjusted = run_with(base_extra={"vehicle_spec": blueboat},
                        cli_args={"speed_reference_m_s": 1.5,
                                  "speed_reference_basis": "condition_adjusted"})
    assert adjusted.header["speed_reference"]["value_m_s"] == 1.5
    missing = run_with(cli_args={})
    assert missing.header["speed_reference"] is None
    assert any("speed_reference" in item for item in missing.unverified)


def load_vehicles():
    path = SCRIPTS.parent / "tasks" / "_shared" / "vehicles.py"
    spec = importlib.util.spec_from_file_location("d8_vehicles", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["d8_vehicles"] = module
    spec.loader.exec_module(module)
    return module


def test_vessel_family_and_mismatch():
    assert cert.vessel_family("wamv_vrx_nominal") == "wamv" and cert.vessel_family("blueboat") == "blueboat"
    with pytest.raises(ValueError, match="runtime"):
        run_with(cli_args={"vessel": "wamv"})
    assert run_with(cli_args={"vessel": "blueboat"}).header["vessel"] == "blueboat"


def test_header_d8_stamps_and_schema_validity():
    run = run_with(cli_args={"cell_config_record": {"path": "c.json", "sha256": "e" * 64,
                                                    "configuration": {"format": "usvbench-cell-v1"}},
                             "runtime_environment": {"CUDA_VISIBLE_DEVICES": "0"}})
    records = []
    for ep in range(2):
        fixtures.step(run)
        for env in range(2):
            records.append(run.finish_episode(env, fixtures.sample_record(env, ep)))
    document = run.build({"records": records})
    header = document["header"]
    assert header["resolved_deadline_s"] == pytest.approx(120.0)
    assert header["resolved_deadline_source"] == "base.max_episode_length * control_step_s"
    assert header["controller_latency"]["control_step_s"] == pytest.approx(1 / 60)
    assert header["seed_namespace"]["purpose"] == "diagnostic"
    assert header["cell_config"]["sha256"] == "e" * 64
    validator = load("validate_cert_v3")
    assert not validator.validate_certificate(document, num_envs=2, episodes_per_env=2)


def test_frozen_level_checked_on_env_cfg_too():
    run = run_with()
    run.base.cfg = SimpleNamespace(curriculum_frozen=True, eval_level=0)
    fixtures.step(run)
    run.base.cfg.curriculum_frozen = False
    with pytest.raises(RuntimeError, match="curriculum"):
        fixtures.step(run)


def test_terminal_success_prefix_equals_full_episode_and_observer_does_not_stack():
    dones = ([False, False], [False, False])
    base = SimpleNamespace(num_envs=2, device="cpu", termination_reason=[1, 3],
                           control_step_s=1 / 60, max_episode_length=7200,
                           robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=[[0., 0., 0.], [0., 0., 0.]])))
    calls = []
    base._get_dones = lambda: calls.append(1) or dones
    run = run_with(base_extra=vars(base))
    second = run_with(base_extra={"_get_dones": run.base._get_dones,
                                  "_usvbench_original_get_dones": run.base._usvbench_original_get_dones,
                                  "robot": run.base.robot})
    assert second.base._usvbench_original_get_dones is run.base._usvbench_original_get_dones
    run.after_reset()
    run.base.robot.data.root_pos_w = [[3., 4., 0.], [0., 1., 0.]]
    with run.time_controller():
        pass
    with run.time_environment():
        run.base._get_dones = run.base._get_dones  # env.step calls the wrapper
        result = run.base._get_dones()
        run._paths.observe(([True, False], [False, True]))
    run.after_step()
    assert result is dones and len(calls) == 1
    rec0 = run.finish_episode(0, dict(fixtures.sample_record(0, 0), success=True, terminated=True,
                                      truncated=False, tts_s=0.1))
    assert rec0["termination_reason"] == "success"
    assert rec0["path_length_full_episode_m"] == 5.0 == rec0["path_length_success_prefix_m"]
    rec1 = run.finish_episode(1, fixtures.sample_record(1, 0))
    assert rec1["path_length_full_episode_m"] == 1.0 and rec1["path_length_success_prefix_m"] is None


# ------------------------------------------------------------------ paired intervals

def test_tango_closed_form_mcnemar_and_symmetry():
    z = 1.959963984540054
    for n in (16, 128, 512):
        lo, hi = intervals.tango_interval(0, 0, n)
        assert hi == pytest.approx(z * z / (n + z * z), rel=1e-9) and lo == pytest.approx(-hi)
    a = intervals.paired_rate_interval([(True, False)] * 20 + [(False, True)] * 5 + [(True, True)] * 50)
    b = intervals.paired_rate_interval([(False, True)] * 20 + [(True, False)] * 5 + [(True, True)] * 50)
    assert a["difference"] == pytest.approx(0.2) and a["interval"] == pytest.approx([-x for x in b["interval"]][::-1])
    assert a["interval"][0] > 0 and a["mcnemar_exact_p"] == pytest.approx(
        2 * sum(math.comb(25, i) for i in range(6)) / 2 ** 25)
    assert intervals.mcnemar_exact_p(0, 0) == 1.0
    with pytest.raises(ValueError):
        intervals.paired_rate_interval([(1, 0)])


def test_tango_coverage_small_simulation():
    rng = random.Random(7)
    p10, p01, n, reps = 0.1, 0.05, 128, 600
    hits = 0
    for _ in range(reps):
        b = c = 0
        for _ in range(n):
            u = rng.random()
            b += u < p10
            c += p10 <= u < p10 + p01
        lo, hi = intervals.tango_interval(b, c, n)
        hits += lo <= p10 - p01 <= hi
    assert 0.92 <= hits / reps <= 0.98


# ------------------------------------------------------------------ MPPI noise

def test_counter_noise_matches_reference_and_is_keyed():
    torch = pytest.importorskip("torch")
    mppi = load("mppi_controller")
    key = mppi.noise_key(3, 5, 7, 11)
    got = mppi.counter_normals(torch.tensor([key], dtype=torch.int64), 16)[0].tolist()
    want = [mppi.normal_reference(key, j) for j in range(16)]
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
    assert mppi.noise_self_check("cpu").startswith("passed")
    keys = {mppi.noise_key(s, e, ep, t) for s in (0, 1) for e in (0, 1) for ep in (0, 1) for t in (0, 1)}
    assert len(keys) == 16
    sample = mppi.counter_normals(torch.tensor([mppi.noise_key(0, e, 0, 0) for e in range(8)],
                                               dtype=torch.int64), 4000).reshape(-1)
    assert abs(float(sample.mean())) < 0.03 and abs(float(sample.var()) - 1.0) < 0.05


# ------------------------------------------------------------------ gate record and analysis policy

def selection_certificate(training_seed=42, successes=30):
    registry = seeds.load_registry()
    stamp = seeds.resolve_seed_block("selection-gate-01", "selection", "selection", registry)
    n, quota = 64, 2
    config = SimpleNamespace(seed=stamp["seed"], eval_level=0, curriculum_frozen=True, vehicle="blueboat",
                             scene=SimpleNamespace(num_envs=n))
    base = SimpleNamespace(num_envs=n, device="cpu", termination_reason=[3] * n,
                           control_step_s=1 / 60, max_episode_length=7200)
    cell = {"path": "gate.cell.json", "sha256": "f" * 64,
            "configuration": {"format": "usvbench-cell-v1", "training_seed": training_seed,
                              "preset": None, "effective_override_literals": {}}}
    run = fixtures.make_run(n, n * quota, base=base, env_cfg=config, overrides={}, argv=["eval.py"],
                            task="Isaac-USV-HazardCross-Direct-v1", controller="learned",
                            eval_seed=stamp["seed"],
                            cli_args={"seed_registration": stamp, "cell_config_record": cell})
    records, count = [], 0
    for ep in range(quota):
        fixtures.step(run)
        for env in range(n):
            ok = count < successes
            count += 1
            records.append(run.finish_episode(env, dict(fixtures.sample_record(env, ep), success=ok,
                                                        tts_s=5.0 if ok else None)))
    return run.build({"records": records})


def test_gate_record_built_from_validated_certificate():
    registry = seeds.load_registry()
    document = selection_certificate(successes=30)
    record = seeds.gate_record_from_certificate(
        document, registry=registry, task_id="Isaac-USV-HazardCross-Direct-v1", vessel="blueboat",
        training_seed=42, training_status="completed", budget_timesteps=10, completed_timesteps=10,
        checkpoint_sha256=None, certificate_path="gate.json", certificate_sha256="a" * 64)
    assert record["verdict"] == "invalid"  # certificate carries no checkpoint -> no admission
    document["header"]["checkpoint"] = {"path": "p.pt", "sha256": "b" * 64}
    record = seeds.gate_record_from_certificate(
        document, registry=registry, task_id="Isaac-USV-HazardCross-Direct-v1", vessel="blueboat",
        training_seed=42, training_status="completed", budget_timesteps=10, completed_timesteps=10,
        certificate_path="gate.json", certificate_sha256="a" * 64)
    assert record["verdict"] == "pass" and record["gate_certificate"]["successes"] == 30
    with pytest.raises(ValueError, match="training seed"):
        seeds.gate_record_from_certificate(
            document, registry=registry, task_id="Isaac-USV-HazardCross-Direct-v1", vessel="blueboat",
            training_seed=1001, training_status="completed", budget_timesteps=10, completed_timesteps=10)
    with pytest.raises(ValueError, match="purpose"):
        seeds.gate_record_from_certificate(
            fixtures.document(64, 2), registry=registry, task_id="Test-BlueBoat", vessel="blueboat",
            training_seed=42, training_status="completed", budget_timesteps=10, completed_timesteps=10)


def test_analysis_purpose_filter_and_untracked_treatment_primitive():
    document = selection_certificate()
    inputs.validate_analysis_input(document, allowed_purposes=("selection",))
    with pytest.raises(ValueError, match="purpose"):
        inputs.validate_analysis_input(document, allowed_purposes=("final_test",))
    base, treat = selection_certificate(), selection_certificate()
    for doc in (base, treat):
        doc["scenario_protocol"] = {"version": 2}
        for row in doc["records"]:
            row["scenario_hashes"] = {"layout": "L%d%d" % (row["env"], row["ep"]),
                                      "rotation": "R", "scenario": "S%d%d" % (row["env"], row["ep"])}
        doc["header"]["resolved_config"]["obs_noise_sigma_pos"] = 0.0
    treat["header"]["resolved_config"]["obs_noise_sigma_pos"] = 0.4
    # Perception is the declared treatment: it has no hash and is not compared.
    assert len(inputs.validate_pair(base, treat, ["perception"])) == 128
    # Undeclared, the nonzero perception dose must be hash-verified: refused.
    with pytest.raises(ValueError, match="missing"):
        inputs.validate_pair(base, treat)


def test_unpaired_tables_may_mix_scenario_protocols():
    a, b = selection_certificate(), selection_certificate()
    a["scenario_protocol"], b["scenario_protocol"] = {"version": 2}, "off-protocol"
    inputs.validate_compatible([a, b])
    with pytest.raises(ValueError, match="scenario_protocol"):
        inputs.validate_compatible([a, b], same_scenario_protocol=True)
