"""CPU tests for seed_registry, run_cell and crossmachine_sample (no Isaac).

Run: python -B -m pytest -p no:cacheprovider repo/scripts/test_run_cell_v3plus.py -q
"""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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


seeds = load("seed_registry")
protocol = load("eval_protocol")
runner = load("run_cell")
cross = load("crossmachine_sample")
cert = load("cert_v3")

CROSS = "Isaac-USV-HazardCross-Direct-v1"          # hazard_nav family
STATION = "Isaac-USV-StationKeep-BlueBoat-Direct-v1"  # station_keeping family
HARBOR = "Isaac-USV-HarborMission-Direct-v1"        # harbor_mission family


@pytest.fixture
def scratch():
    root = SCRIPTS.parents[1] / "test_scratch"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runcell-", dir=root) as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def clean_environment():
    with patch.dict("os.environ", {}, clear=True):
        yield


def frozen_registry(directory, selection_mode=None):
    registry = seeds.load_registry()
    registry["status"] = "frozen"
    registry["competence_gate"]["status"] = "frozen"
    if selection_mode is not None:          # G19b follow-up m3: run_cell applies the registry's declared rule
        registry["ppo_training"]["selection_mode"] = selection_mode
        if selection_mode != registry["ppo_training"].get("selection_mode_confirmation", {}).get("confirmed_mode"):
            # G28: a hypothetical mode carries no owner confirmation of that mode
            registry["ppo_training"].pop("selection_mode_confirmation", None)
    path = directory / "registry_frozen.json"
    path.write_text(json.dumps(registry, indent=2), encoding="ascii")
    return path, seeds.load_registry(path)


def checkpoint(directory, name="policy.pt", content=b"weights"):
    path = directory / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def gate_record(registry, task, vessel, seed, successes=None, status="completed",
                checkpoint_sha="c" * 64):
    gate = registry["competence_gate"]
    evidence = None
    if successes is not None:
        evidence = {"path": "gate.json", "sha256": "d" * 64, "seed_namespace": gate["seed_namespace"],
                    "seed_block": gate["seed_block"], "episodes": gate["episodes"],
                    "num_envs": gate["num_envs"], "successes": successes, "complete": True}
    record = {"task_id": task, "vessel": vessel, "training_seed": seed,
              "candidate_index": registry["ppo_training"]["candidate_order"].index(seed),
              "training": {"status": status, "budget_timesteps": 1000,
                           "completed_timesteps": 1000, "note": ""},
              "checkpoint": {"path": "ckpt.pt", "sha256": checkpoint_sha},
              "gate_certificate": evidence, "verdict": None, "reason": None}
    record["verdict"], record["reason"] = seeds.gate_verdict(gate, record)
    return record


def ledger_with(registry, records):
    ledger = seeds.empty_ledger(registry)
    ledger["records"] = records
    return ledger


# ------------------------------------------------------------------ registry

def test_registry_disjoint_and_encodes_owner_decisions():
    registry = seeds.load_registry()
    blocks = [b for entry in registry["namespaces"].values() for b in entry["blocks"]]
    assert len(blocks) == len({b["seed"] for b in blocks})
    order = registry["ppo_training"]["candidate_order"]
    assert order[0] == 42 and registry["ppo_training"]["admit_first_passing"] == 3
    assert [b["seed"] for b in registry["namespaces"]["training"]["blocks"]] == order
    assert registry["owner_decisions_encoded"]["trained_controllers"] == ["ppo"]
    final = registry["namespaces"]["final_test"]["blocks"]
    assert [b["seed"] for b in final if b["cohort"] == "main"] == [41041]          # G52: owner 2026-10-14
    assert [b["seed"] for b in final if b["id"] in ("final-02", "final-03", "final-04")] == [41047, 41051, 41057]
    assert registry["namespaces"]["final_test"]["planned_main_blocks"] == 1
    assert any(b["cohort"] == "reserve" for b in final)
    for legacy in (7, 123):
        assert seeds.namespace_for_seed(legacy, registry) is None
    gate = registry["competence_gate"]
    assert gate["status"] == "frozen" and seeds.gate_min_successes(gate) == 26       # G52: frozen, 0.2 kept
    assert registry["status"] == "frozen"


@pytest.mark.parametrize("block,namespace,purpose", [
    ("final-01", "selection", "selection"),       # test block used for selection
    ("final-01", "final_test", "selection"),      # wrong purpose for the namespace
    ("selection-gate-01", "final_test", "final_test"),
    ("pilot-01", "crossmachine_calibration", "calibration"),
    ("train-0042", "training", "training"),       # training seeds are not evaluation seeds
    ("42", "final_test", "final_test"),
])
def test_seed_outside_its_namespace_refused(block, namespace, purpose):
    with pytest.raises(ValueError):
        seeds.resolve_seed_block(block, namespace, purpose)


def test_registry_reuse_and_unfrozen_gate_rejected():
    registry = seeds.load_registry()
    broken = copy.deepcopy(registry)
    broken["namespaces"]["dose_pilots"]["blocks"][0]["seed"] = 41041
    with pytest.raises(ValueError, match="reused"):
        seeds.validate_registry(broken)
    broken = copy.deepcopy(registry)
    broken["namespaces"]["dose_pilots"]["blocks"][0]["seed"] = 42
    with pytest.raises(ValueError, match="reused"):
        seeds.validate_registry(broken)
    broken = copy.deepcopy(registry)
    broken["competence_gate"]["status"] = "OWNER_DECISION_PENDING"   # G52
    with pytest.raises(ValueError, match="frozen competence gate"):
        seeds.validate_registry(broken)
    broken = copy.deepcopy(registry)
    broken["ppo_training"]["candidate_order"] = [1001, 42] + registry["ppo_training"]["candidate_order"][2:]
    with pytest.raises(ValueError, match="candidate_order"):
        seeds.validate_registry(broken)


def test_direct_diagnostic_seeds():
    registry = seeds.load_registry()
    stamp = seeds.diagnostic_registration(42, registry)
    assert stamp["purpose"] == "diagnostic" and stamp["also_registered_as"] == "training"
    assert seeds.diagnostic_registration(7, registry)["legacy_excluded"] is True
    for seed in (41041, 11001, 21001, 31001):
        with pytest.raises(ValueError, match="namespace"):
            seeds.diagnostic_registration(seed, registry)


# ------------------------------------------------------------------ gate / ledger

def test_gate_verdicts():
    registry = seeds.load_registry()
    assert gate_record(registry, CROSS, "blueboat", 42, successes=26)["verdict"] == "pass"
    assert gate_record(registry, CROSS, "blueboat", 42, successes=25)["verdict"] == "fail"
    assert gate_record(registry, CROSS, "blueboat", 42, status="diverged")["verdict"] == "fail"
    assert gate_record(registry, CROSS, "blueboat", 42,
                       status="infrastructure_failure")["verdict"] == "invalid"
    assert gate_record(registry, CROSS, "blueboat", 42)["verdict"] == "invalid"  # no evidence
    record = gate_record(registry, CROSS, "blueboat", 42, successes=90)
    record["gate_certificate"]["seed_block"] = "selection-02"
    assert seeds.gate_verdict(registry["competence_gate"], record)[0] == "invalid"


def test_first_three_passing_admitted_and_dead_seeds_reported():
    registry = seeds.load_registry()
    records = [gate_record(registry, CROSS, "blueboat", 42, successes=80),
               gate_record(registry, CROSS, "blueboat", 1001, successes=3),
               gate_record(registry, CROSS, "blueboat", 2003, status="infrastructure_failure"),
               gate_record(registry, CROSS, "blueboat", 2003, successes=60),
               gate_record(registry, CROSS, "blueboat", 3001, status="diverged"),
               gate_record(registry, CROSS, "blueboat", 4001, successes=26),
               gate_record(registry, CROSS, "blueboat", 5003, successes=127),
               gate_record(registry, STATION, "blueboat", 42, successes=1)]
    result = seeds.admission(ledger_with(registry, records), registry, CROSS, "blueboat")
    assert result["status"] == "complete"
    assert [a["training_seed"] for a in result["admitted"]] == [42, 2003, 4001]
    assert [d["training_seed"] for d in result["dead"]] == [1001, 3001]
    # 5003 was trained anyway; it is reported, never used.
    assert result["records_beyond_rule"] == [5003]
    station = seeds.admission(ledger_with(registry, records), registry, STATION, "blueboat")
    assert station["status"] == "pending" and station["waiting_on_candidate"] == 1001
    assert [d["training_seed"] for d in station["dead"]] == [42]
    exhausted = [gate_record(registry, CROSS, "wamv", s, successes=0)
                 for s in registry["ppo_training"]["candidate_order"]]
    assert seeds.admission(ledger_with(registry, exhausted), registry, CROSS, "wamv")["status"] == "exhausted"


def test_ledger_tampering_and_retries_rejected():
    registry = seeds.load_registry()
    failed = gate_record(registry, CROSS, "blueboat", 42, successes=3)
    edited = copy.deepcopy(failed)
    edited["verdict"] = "pass"
    with pytest.raises(ValueError, match="recomputed"):
        seeds.validate_ledger(ledger_with(registry, [edited]), registry)
    retry = gate_record(registry, CROSS, "blueboat", 42, successes=90)
    with pytest.raises(ValueError, match="retries"):
        seeds.validate_ledger(ledger_with(registry, [failed, retry]), registry)
    wrong_index = copy.deepcopy(failed)
    wrong_index["candidate_index"] = 5
    with pytest.raises(ValueError, match="candidate_index"):
        seeds.validate_ledger(ledger_with(registry, [wrong_index]), registry)
    other = ledger_with(registry, [failed])
    other["gate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="gate definition"):
        seeds.validate_ledger(other, registry)


def test_reserve_replacement_is_declared_order():
    registry = seeds.load_registry()
    first = seeds.next_reserve_block(registry, set())
    assert first["id"] == "final-02"            # G52: final-02..04 are reserves (owner 2026-10-14), declared first
    assert seeds.next_reserve_block(registry, {first["id"]})["id"] == "final-03"
    used = {"final-02", "final-03", "final-04"}
    assert seeds.next_reserve_block(registry, used)["id"] == "final-reserve-01"


# ------------------------------------------------------------------ run_cell

def cell_args(*extra, controller="mppi", task=CROSS, block="pilot-01", namespace="dose_pilots",
              purpose="pilot"):
    values = ["--task", task, "--vessel", "blueboat", "--controller", controller,
              "--eval-seed-block", block, "--seed-namespace", namespace,
              "--run-purpose", purpose, "--level", "0", "--episodes", "128",
              "--out", str(SCRIPTS.parents[1] / "test_scratch" / "cell.json"), "--dry-run"]
    return runner.parser_for_cell().parse_args(values + list(extra))


def test_registered_task_table_is_static_and_complete():
    tasks = runner.registered_tasks()
    assert tasks[CROSS] == "hazard_nav" and tasks[STATION] == "station_keeping"
    assert tasks[HARBOR] == "harbor_mission"
    assert "Isaac-USV-HazardMovingBasin-Direct-v1" in tasks


def test_dispatch_per_controller_and_family():
    result = runner.resolve_cell(cell_args())
    assert Path(result["command"][1]).name == "eval_mppi.py"
    assert result["command"][result["command"].index("--mppi-seed") + 1] == "0"
    planner = runner.resolve_cell(cell_args(controller="classical"))
    assert Path(planner["command"][1]).name == "classical_planner_pid.py"
    baseline = runner.resolve_cell(cell_args(controller="classical", task=STATION))
    command = baseline["command"]
    assert Path(command[1]).name == "classical_baseline.py"
    assert command[2:4] == ["--protocol", "mission"] and "--seed" in command
    assert baseline["configuration"]["controller_information"] == "observation-only LOS/PID"
    with pytest.raises(ValueError, match="no declared classical"):
        runner.resolve_cell(cell_args(controller="classical", task=HARBOR))
    with pytest.raises(ValueError, match="unknown task"):
        runner.resolve_cell(cell_args(task="Isaac-USV-Nope-v1"))


@pytest.mark.parametrize("extra", [
    ("--episodes", "65"), ("--set", "seed=41047"), ("--set", "eval_level=2"),
    ("--set", "curriculum_frozen=false"), ("--set", "scene.num_envs=32"),
    ("--set", "drag_scale=nan"), ("--set", "vehicle=wamv"),
    ("--set", "drag_scale=2", "--set", "drag_scale=3"), ("--controller-seed", "5"),
])
def test_protocol_conflicts_rejected(extra):
    with pytest.raises(ValueError):
        runner.resolve_cell(cell_args(*extra))


def test_final_test_requires_frozen_registry(scratch):
    unfrozen = seeds.load_registry()                       # G52: the in-tree registry is frozen since v2.6.6
    unfrozen["status"], unfrozen["competence_gate"]["status"] = "proposal", "OWNER_DECISION_PENDING"
    proposal_path = scratch / "registry_proposal.json"
    proposal_path.write_text(json.dumps(unfrozen, indent=2), encoding="ascii")
    args = cell_args("--seed-registry", str(proposal_path), block="final-01", namespace="final_test",
                     purpose="final_test")
    with pytest.raises(ValueError, match="frozen"):
        runner.resolve_cell(args)
    result = runner.resolve_cell(cell_args(block="final-01", namespace="final_test", purpose="final_test"))
    assert result["configuration"]["seed_registration"]["registry_status"] == "frozen"
    assert result["configuration"]["seed_registration"]["cohort"] == "main"
    path, _ = frozen_registry(scratch)
    args = cell_args("--seed-registry", str(path), block="final-01", namespace="final_test",
                     purpose="final_test")
    result = runner.resolve_cell(args)
    assert result["configuration"]["seed_registration"]["registry_status"] == "frozen"
    assert "--seed-registry" in result["command"]


def test_ppo_refusal_logic_and_admission(scratch):
    path, registry = frozen_registry(scratch, selection_mode="first_passing")
    ckpt, sha = checkpoint(scratch)
    base = ("--seed-registry", str(path), "--checkpoint", str(ckpt))
    final = dict(controller="ppo", block="final-01", namespace="final_test", purpose="final_test")
    with pytest.raises(ValueError, match="training-seed"):
        runner.resolve_cell(cell_args(*base, **final))
    with pytest.raises(ValueError, match="gate-ledger"):
        runner.resolve_cell(cell_args(*base, "--training-seed", "42", **final))
    records = [gate_record(registry, CROSS, "blueboat", 42, successes=3, checkpoint_sha=sha),
               gate_record(registry, CROSS, "blueboat", 1001, successes=99, checkpoint_sha=sha)]
    ledger = scratch / "ledger.json"
    ledger.write_text(json.dumps(ledger_with(registry, records)), encoding="ascii")
    with pytest.raises(ValueError, match="not admitted"):
        runner.resolve_cell(cell_args(*base, "--training-seed", "42", "--gate-ledger", str(ledger), **final))
    ok = runner.resolve_cell(cell_args(*base, "--training-seed", "1001", "--gate-ledger", str(ledger), **final))
    assert ok["configuration"]["admission"]["status"] == "admitted"
    assert ok["configuration"]["checkpoint"]["sha256"] == sha
    other, _ = checkpoint(scratch, "other.pt", b"different")
    with pytest.raises(ValueError, match="admitted checkpoint"):
        runner.resolve_cell(cell_args("--seed-registry", str(path), "--checkpoint", str(other),
                                      "--training-seed", "1001", "--gate-ledger", str(ledger), **final))
    with pytest.raises(ValueError, match="only to PPO"):
        runner.resolve_cell(cell_args("--seed-registry", str(path), "--training-seed", "42",
                                      block="final-01", namespace="final_test", purpose="final_test"))


def test_selection_is_the_gate_only(scratch):
    ckpt, _ = checkpoint(scratch)
    gate = dict(controller="ppo", block="selection-gate-01", namespace="selection", purpose="selection")
    ok = runner.resolve_cell(cell_args("--checkpoint", str(ckpt), "--training-seed", "42", **gate))
    assert ok["configuration"]["admission"]["status"] == "gate_evaluation"
    with pytest.raises(ValueError, match="nominal"):
        runner.resolve_cell(cell_args("--checkpoint", str(ckpt), "--training-seed", "42",
                                      "--set", "drag_scale=2", **gate))
    with pytest.raises(ValueError, match="competence gate"):
        runner.resolve_cell(cell_args("--checkpoint", str(ckpt), "--training-seed", "42",
                                      controller="ppo", block="selection-02", namespace="selection",
                                      purpose="selection"))
    with pytest.raises(ValueError, match="episodes"):
        runner.resolve_cell(cell_args("--checkpoint", str(ckpt), "--training-seed", "42",
                                      "--episodes", "256", **gate))
    with pytest.raises(ValueError, match="selection namespace belongs"):
        runner.resolve_cell(cell_args(block="selection-gate-01", namespace="selection",
                                      purpose="selection"))


def test_presets_precede_cli_and_are_digested(scratch):
    presets = scratch / "presets.json"
    presets.write_text(json.dumps({"format": "usvbench-dose-presets-v1", "presets": {
        "low": {"level": 1, "overrides": {"sea_state.hs_range": [0.05, 0.05], "drag_scale": 1.0}}}}),
        encoding="ascii")
    args = cell_args("--set", "drag_scale=2.0")
    args.level, args.dose_preset, args.presets_file = None, "low", presets
    result = runner.resolve_cell(args)
    literals = result["configuration"]["effective_override_literals"]
    assert literals == {"sea_state.hs_range": "0.05,0.05", "drag_scale": "2.0"}
    assert result["configuration"]["level"] == 1
    assert result["command"].count("drag_scale=2.0") == 1 and "drag_scale=1.0" not in result["command"]
    assert "--preset-sha256" in result["command"]


def test_ambient_environment_policy():
    with patch.dict("os.environ", {"GOAL_RADIUS_OVERRIDE": "3"}):
        with pytest.raises(ValueError, match="allow-env"):
            runner.resolve_cell(cell_args())
        result = runner.resolve_cell(cell_args("--allow-env", "GOAL_RADIUS_OVERRIDE"))
        assert result["configuration"]["allowed_environment"] == {"GOAL_RADIUS_OVERRIDE": "3"}
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
        result = runner.resolve_cell(cell_args())
        assert result["configuration"]["runtime_environment"] == {"CUDA_VISIBLE_DEVICES": "1"}


def test_dry_run_prints_json_without_isaac_or_subprocess():
    stream = io.StringIO()
    argv = ["--task", CROSS, "--vessel", "blueboat", "--controller", "mppi",
            "--eval-seed-block", "pilot-01", "--seed-namespace", "dose_pilots",
            "--run-purpose", "pilot", "--level", "0", "--out", "x.json", "--dry-run"]
    with patch.object(sys, "stdout", stream), patch.object(runner.subprocess, "run") as run:
        assert runner.main(argv) == 0
    run.assert_not_called()
    printed = json.loads(stream.getvalue())
    assert printed["configuration"]["format"] == "usvbench-cell-v1"
    assert not any(name == "isaaclab" or name.startswith("isaaclab.") for name in sys.modules)


def test_launch_writes_cell_config_and_evaluator_verifies_it(scratch):
    out = scratch / "cells" / "cross_mppi.json"
    argv = ["--task", CROSS, "--vessel", "blueboat", "--controller", "mppi",
            "--eval-seed-block", "pilot-01", "--seed-namespace", "dose_pilots",
            "--run-purpose", "pilot", "--level", "0", "--out", str(out),
            "--set", "drag_scale=2.0"]
    with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        assert runner.main(argv) == 0
    command = run.call_args[0][0]
    cell = Path(command[command.index("--cell-config") + 1])
    sha = command[command.index("--cell-config-sha256") + 1]
    assert cell.is_file() and hashlib.sha256(cell.read_bytes()).hexdigest() == sha
    with patch.object(runner.subprocess, "run") as run, pytest.raises(SystemExit):
        runner.main(argv)  # cell config exists: never overwritten
    run.assert_not_called()
    # Evaluator side: the launched argv must be exactly the planned cell.
    evaluator_args = SimpleNamespace(
        task=CROSS, episodes=128, num_envs=64, level=0, eval_seed=21001,
        seed_namespace="dose_pilots", seed_block="pilot-01", run_purpose="pilot",
        extra_sets=["drag_scale=2.0"], allow_env=[], device="cuda:0", seed_registry=None,
        cell_config=str(cell), cell_config_sha256=sha, speed_reference_m_s=None)
    argv0 = [str(SCRIPTS / "eval_mppi.py")]
    # G6b / CONTEXT_R4 D28: an official (pilot) run needs RE-1; inject the reference versions
    renv = protocol.reference_environment
    with patch.object(renv, "observed_versions", lambda: dict(renv.load_required()[1])):
        protocol.preflight_protocol(evaluator_args, argv0)
    assert evaluator_args.cell_config_record["sha256"] == sha
    assert evaluator_args.cell_config_record["configuration"]["controller"] == "mppi"
    tampered = copy.copy(evaluator_args)
    tampered.extra_sets = ["drag_scale=3.0"]
    with pytest.raises(ValueError, match="--set literals"):
        protocol.preflight_protocol(tampered, argv0)
    wrong_script = copy.copy(evaluator_args)
    with pytest.raises(ValueError, match="evaluator"):
        protocol.preflight_protocol(wrong_script, [str(SCRIPTS / "eval_v6_frozen.py")])
    bad_sha = copy.copy(evaluator_args)
    bad_sha.cell_config_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        protocol.preflight_protocol(bad_sha, argv0)


def test_ppo_cells_follow_the_declared_fixed_first_n_rule(scratch):
    """G19b follow-up m3: under fixed_first_n (the registry's owner-confirmed mode) candidates 1-3 are delivered with
    their flags (a below-gate candidate too); candidate 4 never is."""
    path, registry = frozen_registry(scratch)
    assert registry["ppo_training"]["selection_mode"] == "fixed_first_n"
    ckpt, sha = checkpoint(scratch)
    final = dict(controller="ppo", block="final-01", namespace="final_test", purpose="final_test")
    records = [gate_record(registry, CROSS, "blueboat", 42, successes=3, checkpoint_sha=sha),
               gate_record(registry, CROSS, "blueboat", 3001, successes=99, checkpoint_sha=sha)]
    ledger = scratch / "ledger_b.json"
    ledger.write_text(json.dumps(ledger_with(registry, records)), encoding="ascii")
    base = ("--seed-registry", str(path), "--checkpoint", str(ckpt), "--gate-ledger", str(ledger))
    ok = runner.resolve_cell(cell_args(*base, "--training-seed", "42", **final))
    admission = ok["configuration"]["admission"]
    assert admission["status"] == "admitted" and admission["selection_mode"] == "fixed_first_n"
    assert admission["flag"] == "below_gate"
    with pytest.raises(ValueError, match="not admitted"):
        runner.resolve_cell(cell_args(*base, "--training-seed", "3001", **final))   # beyond the first three
    with pytest.raises(ValueError, match="not admitted"):
        runner.resolve_cell(cell_args(*base, "--training-seed", "1001", **final))   # no gate verdict yet


def test_direct_final_test_without_run_cell_refused(scratch):
    path, _ = frozen_registry(scratch)
    args = SimpleNamespace(task=CROSS, episodes=128, num_envs=64, level=0, eval_seed=41041,
                           seed_namespace="final_test", seed_block="final-01",
                           run_purpose="final_test", extra_sets=[], allow_env=[], device=None,
                           seed_registry=str(path), cell_config=None, speed_reference_m_s=None)
    with pytest.raises(ValueError, match="run_cell"):
        protocol.preflight_protocol(args, ["eval_mppi.py"])


# ------------------------------------------------------------------ cross-machine

def test_crossmachine_plan_is_24_stratified_cells(scratch):
    registry = seeds.load_registry()
    ckpt, sha = checkpoint(scratch)
    strata = {("blueboat", "short"): CROSS, ("blueboat", "long"): STATION,
              ("wamv", "short"): "Isaac-USV-PathHazard-Direct-v1",
              ("wamv", "long"): "Isaac-USV-Dock-BlueBoat-Direct-v1"}
    records = [gate_record(registry, task, vessel, 42, successes=64, checkpoint_sha=sha)
               for (vessel, _), task in strata.items()]
    ledger = scratch / "ledger.json"
    ledger.write_text(json.dumps(ledger_with(registry, records)), encoding="ascii")
    tasks = [{"vessel": vessel, "duration": duration, "task": task,
              "deadline_s": 120 if duration == "short" else 240, "level": 0,
              "checkpoint": str(ckpt), "training_seed": 42,
              "doses": {"low": {"drag_scale": 1.0}, "high": {"drag_scale": 2.0}}}
             for (vessel, duration), task in strata.items()]
    plan = cross.build_sample({"gate_ledger": str(ledger), "tasks": tasks}, ["boxA", "boxB"])
    assert plan["batches_per_machine"] == 24 and len({c["cell_id"] for c in plan["cells"]}) == 24
    assert {c["controller"] for c in plan["cells"]} == {"ppo", "mppi", "classical"}
    for cell in plan["cells"]:
        a, b = cell["commands"]["boxA"], cell["commands"]["boxB"]
        assert "calibration" in a["command"]
        replay = runner.resolve_cell(runner.parser_for_cell().parse_args(a["run_cell_argv"]))
        assert replay["command"] == a["command"]
        # Same cell on both machines except the output location.
        strip = lambda cmd: [x for x in cmd if "box" not in x and len(x) != 64]
        assert strip(a["command"]) == strip(b["command"])
    tasks[-1]["duration"] = "short"
    with pytest.raises(ValueError):
        cross.build_sample({"gate_ledger": str(ledger), "tasks": tasks}, ["boxA", "boxB"])


def calibration_certificate(host, outcomes=(True, False, True, False)):
    n, quota = 2, len(outcomes) // 2
    config = SimpleNamespace(seed=31001, eval_level=0, curriculum_frozen=True, vehicle="blueboat",
                             episode_length_s=120.0, scene=SimpleNamespace(num_envs=n))
    base = SimpleNamespace(num_envs=n, device="cpu", termination_reason=[3] * n,
                           control_step_s=1 / 60, max_episode_length=7200)
    provenance = {key: "a" * 64 for key in ("tasks_sha256", "scripts_sha256", "tree_sha256",
                                            "evaluator_sha256")}
    provenance.update(tasks_root="tasks", scripts_root="scripts", environment_source="tasks/env.py",
                      evaluator_path="scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
    # E1: synthetic legacy digests must be self-consistent (validator recomputes tree_sha256, D2 item 10).
    provenance["tree_sha256"] = cert.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {key: "test" for key in ("gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                      "isaac_lab_version", "physx_version")}
    runtime["hostname"] = host
    # G6b / CONTEXT_R4 D28: official certificate fixtures come from an (injected) RE-1 environment.
    runtime["reference_environment"] = cert.reference_environment.reference_block(
        cert.reference_environment.load_required()[1])
    clock = iter(index / 1000 for index in range(10000))
    run = cert.CertificateRun(
        base=base, env_cfg=config, evaluator=__file__, task=CROSS, controller="mppi",
        controller_config={"seed": 0}, checkpoint=None, eval_seed=31001, level=0,
        episodes_requested=len(outcomes), overrides={}, argv=["eval.py"],
        cli_args={"seed_registration": seeds.resolve_seed_block(
            "calibration-01", "crossmachine_calibration", "calibration")},
        provenance=provenance, runtime=runtime, synchronize=lambda: None, clock=lambda: next(clock))
    records = []
    for ep in range(quota):
        with run.time_controller():
            pass
        with run.time_environment():
            pass
        run.after_step()
        for env in range(n):
            success = outcomes[ep * n + env]
            row = dict(env=env, ep=ep, success=success, terminated=False, truncated=True,
                       tts_s=1.0 if success else None, path_length_m=1.0, min_clearance_m=2.0,
                       d0_m=1.0, route_geodesic_m=1.0,
                       scenario_hashes={"layout": "l-%s-%s" % (env, ep),
                                        "rotation": "r-%s-%s" % (env, ep),
                                        "scenario": "s-%s-%s" % (env, ep)})
            records.append(run.finish_episode(env, row))
    return run.build({"records": records, "scenario_protocol": {"version": "fake-v1"}})


def test_crossmachine_comparison():
    a, b = calibration_certificate("boxA"), calibration_certificate("boxB")
    report = cross.compare_certificates(a, b)
    assert report["agreement_count"] == 4 and report["mcnemar_exact_p"] == 1.0
    low, high = report["success_difference_interval_95"]
    assert low < 0 < high  # agreement is not zero uncertainty
    b = calibration_certificate("boxB", (True, True, True, False))
    report = cross.compare_certificates(a, b)
    assert (report["successes_a"], report["successes_b"]) == (2, 3)
    assert report["success_difference_b_minus_a"] == 0.25 and report["agreement_count"] == 3
    pooled = cross.pooled_summary([report, cross.compare_certificates(a, calibration_certificate("boxB"))])
    assert pooled["episodes"] == 8 and pooled["b_only"] == 1


def test_crossmachine_rejects_mismatches():
    a = calibration_certificate("boxA")
    for mutate, message in (
        (lambda d: d["records"][0]["scenario_hashes"].update(layout="x"), "primitive"),
        (lambda d: d["header"]["controller"]["configuration"].update(seed=3), "controller"),
        (lambda d: d["header"]["resolved_config"].update(episode_length_s=99.0), "resolved_config"),
        (lambda d: d["records"].pop(), "grid"),
    ):
        b = calibration_certificate("boxB")
        mutate(b)
        with pytest.raises(ValueError, match=message):
            cross.compare_certificates(a, b)
    with pytest.raises(ValueError, match="two different machines"):
        cross.compare_certificates(a, calibration_certificate("boxA"))
