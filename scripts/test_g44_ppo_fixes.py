"""G44 (v2.6.3) CPU tests: failure terminal penalty bound per family (R2), normaliser freeze, learning-rate floor
and best checkpoint by evaluation (R1), and the trainer integration (fake skrl backend of test_g19b). The family
tests (tasks/*/test_g19b_training_pool.py, code-9 real-env cases) additionally derive the penalty from every
family's REAL journal cfg and bounds through g19b_training_test_support.code9_run."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ppo_training_fixes as FX  # noqa: E402
import ppo_checkpoint_eval as CE  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _load("_g44_g19b_tests", SCRIPTS / "test_g19b_train_journal_ppo.py")
T, R = G.T, G.R


def scratch_dir():
    root = SCRIPTS.parents[1] / "test_scratch"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="g44-", dir=root))      # kept (no cleanup)


def cfg(**fields):
    base = dict(sim=types.SimpleNamespace(dt=1.0 / 120.0), decimation=2, episode_length_s=60.0)
    base.update(fields)
    return types.SimpleNamespace(**base)


HAZ = dict(reward_progress_scale=20.0, reward_clearance_scale=0.0, reward_prox_scale=4.1, safe_clearance_m=0.9,
           half_beam_m=0.45, prox_ray_floor_m=0.45, reward_swift_scale=0.25, reward_open_water_scale=0.0,
           reward_reverse_action_scale=0.05, reward_goal_entry_bonus=50.0, reward_goal_time_bonus=50.0,
           reward_threading_amplitude=0.0, pbrs_correct=False)
PF = dict(reference_reward_progress_scale=20.0, reference_reward_gate_bonus=10.0, reference_reward_terminal_success=100.0)
PH = dict(reference_reward_progress_scale=20.0, reference_reward_gate_bonus=25.0, reward_clearance_scale=0.0,
          reward_prox_scale=4.1, safe_clearance_m=0.9, half_beam_m=0.45, prox_ray_floor_m=0.45,
          reward_reverse_action_scale=0.05, reward_swift_scale=0.25, reward_goal_entry_bonus=50.0,
          reward_goal_time_bonus=50.0)
SK = dict(reference_reward_distance_scale=30.0)
DK = dict(reference_reward_distance_scale_m=25.0, reference_reward_alignment_scale=0.5,
          reference_reward_braking_scale=0.4, reference_reward_success_bonus=3.0)
HM = dict(reward_progress_scale=20.0, reward_clearance_scale=0.0, reward_prox_scale=4.1, safe_clearance_m=0.9,
          half_beam_m=0.45, prox_ray_floor_m=0.45, reward_milestone_bonus=25.0, reward_stage_goal_bonus=50.0,
          reward_stage_time_bonus=50.0, reward_dock_conjunction=1.0)
PLAN = {"length_scale": 1.0, "gate_radius_m": 2.0, "segment_count": 4}


def test_horizon_and_hazard_terms():
    c = cfg(**HAZ)
    assert FX.control_dt(c) == pytest.approx(1.0 / 60.0)
    assert FX.horizon_steps(c) == 3601
    t = FX.penalty_terms("hazard_nav", c, PLAN)
    dense = (1.0 / 60.0) * (4.1 * math.log(1.35 / 0.45) + 0.25 + 0.05)
    assert t["B"] == 0.0 and t["A"] == pytest.approx(20.0 + 3601 * dense + 100.0)


@pytest.mark.parametrize("carrier,fields", [("hazard_nav", HAZ), ("path_following", PF), ("path_hazard", PH),
                                            ("station_keeping", SK), ("docking", DK), ("harbor_mission", HM)])
def test_every_family_derives_a_negative_penalty(carrier, fields):
    t = FX.penalty_terms(carrier, cfg(**fields), PLAN)
    assert t["A"] > 0.0 and t["B"] >= 0.0 and t["N"] == 3601
    assert FX.failure_penalty(t, 50.0) <= FX.failure_penalty(t, 0.0) <= -t["A"] + 1e-12 < 0.0


def test_refusals():
    with pytest.raises(FX.FixError):
        FX.penalty_terms("hazard_nav", cfg(**dict(HAZ, reward_threading_amplitude=5.0)), PLAN)
    with pytest.raises(FX.FixError):
        FX.penalty_terms("hazard_nav", cfg(**{k: v for k, v in HAZ.items() if k != "reward_prox_scale"}), PLAN)
    with pytest.raises(FX.FixError):
        FX.penalty_terms("unknown", cfg(), PLAN)
    with pytest.raises(FX.FixError):
        FX.penalty_terms("station_keeping", cfg(reference_reward_distance_scale=0.0), PLAN)


def _worst_continuation_pf(t, d_prev, diag, n):
    """Lowest return of a continuation that does not fail (path following): the current segment drifts to the far
    corner (progress 20 x (d_prev - D)), no gate bonus."""
    return 20.0 * (d_prev - diag)


def test_path_following_guarantee_on_stylised_paths():
    c = cfg(**PF)
    t = FX.penalty_terms("path_following", c, PLAN)
    diag = math.hypot(50.0, 44.0)
    p = FX.failure_penalty(t, diag)
    for d_prev in (0.5, 5.0, 16.0, diag):
        fail = 20.0 * min(d_prev, 10.0 / 60.0) + 10.0 + p       # best possible failing step (+ a gate event)
        assert fail <= _worst_continuation_pf(t, d_prev, diag, t["N"]) + 1e-9
        assert fail <= 0.0                                      # waiting in place earns 0 in calm water


def test_station_keeping_guarantee():
    c = cfg(**SK)
    t = FX.penalty_terms("station_keeping", c, PLAN)
    diag = 30.0
    worst_wait = t["N"] * (-diag / 30.0)                       # the farthest in-bounds point for the whole horizon
    assert 1.0 + FX.failure_penalty(t, diag) <= worst_wait + 1e-9


class FakeBase:
    def __init__(self, rows):
        self._journal_bounds = torch.tensor(rows, dtype=torch.float64)


def test_failure_penalty_uses_the_ended_episode_and_failure_codes_only():
    t = {"A": 10.0, "B": 1.0, "N": 1, "control_dt_s": 1.0, "carrier": "hazard_nav", "parts": {}}
    base = FakeBase([[0, 3, 0, 4], [0, 6, 0, 8], [0, 3, 0, 4]])
    fp = FX.FailurePenalty(t, base, "hazard_nav", 3)
    fp.refresh()
    assert fp.values == [-15.0, -20.0, -15.0]
    base._journal_bounds = torch.tensor([[0, 30, 0, 40]] * 3, dtype=torch.float64)    # auto-reset happened
    got = fp.penalties([True, True, True], [4, 1, 3])          # out of bounds, success, timeout
    assert got == {0: -15.0}                                    # the value of the episode that ended
    fp.refresh([0, 1, 2])
    assert fp.values == [-60.0] * 3
    assert fp.penalties([True, True, True, False][:3], [2, 9, 5]) == {0: -60.0, 1: -60.0, 2: -60.0}
    assert fp.summary()["per_code"] == {"2": 1, "4": 1, "5": 1, "9": 1}
    with pytest.raises(FX.FixError):
        fp.penalties([True, False, False], None)


class Scaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(2, dtype=torch.float64))

    def forward(self, x, train=False, inverse=False, no_grad=True):
        if train:
            self.running_mean += 1.0
        return x


def test_normaliser_freeze_blocks_train_updates_only_after_warmup():
    agent = types.SimpleNamespace(_state_preprocessor=Scaler())
    fr = FX.NormaliserFreeze(10)
    fr.attach(agent)
    s = agent._state_preprocessor
    s(torch.zeros(1, 2), train=True)
    s(torch.zeros(1, 2), True)
    assert float(s.running_mean[0]) == 2.0 and not fr.maybe_freeze(9)
    assert fr.maybe_freeze(10)
    s(torch.zeros(1, 2), train=True)
    s(torch.zeros(1, 2), True, False)
    s(torch.zeros(1, 2))
    assert float(s.running_mean[0]) == 2.0
    summary = fr.summary()
    assert summary["unchanged_after_freeze"] and summary["train_calls_blocked_after_freeze"] == 2
    assert summary["frozen_at_timestep"] == 10 and summary["train_calls_before_freeze"] == 2
    with pytest.raises(FX.FixError):
        FX.NormaliserFreeze(1).attach(types.SimpleNamespace(_state_preprocessor=None))


def test_learning_rate_floor_raises_min_lr():
    model = torch.nn.Linear(1, 1)
    agent = types.SimpleNamespace(optimizer=torch.optim.SGD(model.parameters(), lr=1.5e-4),
                                  scheduler=types.SimpleNamespace(min_lr=1.0e-6))
    fl = FX.LearningRateFloor(1.5e-4)
    fl.attach(agent)
    assert agent.scheduler.min_lr == pytest.approx(1.5e-5)
    for _ in range(20):                                          # KLAdaptiveLR-like decay with the floor
        group = agent.optimizer.param_groups[0]
        group["lr"] = max(group["lr"] / 1.5, agent.scheduler.min_lr)
        fl.observe()
    s = fl.summary()
    assert s["floor_respected"] and s["updates_at_floor"] > 0 and s["lowest_lr_seen"] == pytest.approx(1.5e-5)
    with pytest.raises(FX.FixError):
        FX.LearningRateFloor(1e-4).attach(types.SimpleNamespace(scheduler=None))


def test_select_best_tie_goes_to_the_later_checkpoint():
    ev = [{"timestep": 9600, "successes": 64, "episodes": 64, "sha256": "a"},
          {"timestep": 19200, "successes": 64, "episodes": 64, "sha256": "b"},
          {"timestep": 96000, "successes": 0, "episodes": 64, "sha256": "c"}]
    assert FX.select_best(ev)["timestep"] == 19200
    with pytest.raises(FX.FixError):
        FX.select_best([dict(ev[0]), dict(ev[1], episodes=32)])
    with pytest.raises(FX.FixError):
        FX.select_best([])


def _cert(path, sha, k, n=64):
    doc = {"header": {"complete": True, "checkpoint": {"sha256": sha}, "eval_seed": 42,
                      "seed_namespace": {"namespace": "diagnostic", "purpose": "diagnostic", "seed": 42,
                                         "block_id": None}},   # v2.6.4 (G47): the selection reads it
           "records": [{"success": i < k} for i in range(n)]}
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


def test_completed_run_carries_the_fixes_and_best_checkpoint_selection():
    root = scratch_dir()
    out = root / "run"
    plan, args = G.plan_for(out, task="journal:path_following")
    assert plan["g44"]["warmup_timesteps"] == 32 and plan["g44"]["plan_params"]["segment_count"] == 4
    backend = G.fake_backend(4)
    assert T.run_training(plan, args, backend) == T.EXIT_OK
    record = R.load_record(out / "training_record.json", run_dir=out)
    assert "ppo_fix.json" in record["log_digests"]
    fix = json.loads((out / "ppo_fix.json").read_text(encoding="ascii"))
    assert fix["normaliser_freeze"]["unchanged_after_freeze"] and fix["normaliser_freeze"]["frozen_at_timestep"] == 32
    # G46: the LR floor is removed (KLAdaptiveLR as in the yaml); the learning rate of every update is logged
    assert fix["learning_rate_floor"]["disabled"] and fix["learning_rate_log"]["updates_seen"] > 0
    assert fix["learning_rate_log"]["lowest_lr_seen"] < fix["learning_rate_log"]["first_lr"]
    fp = fix["failure_terminal_penalty"]
    assert fp["rows_penalised"] > 0 and set(fp["per_code"]) == {"5"}          # fake env: odd rows end with code 5
    expected = -(fp["terms"]["A"] + fp["terms"]["B"] * math.hypot(50.0, 44.0))
    assert fp["penalty_min"] == pytest.approx(expected) and fp["penalty_max"] == pytest.approx(expected)
    # best checkpoint by evaluation (fake certificates naming the record's checkpoint hashes)
    certs = {}
    for i, c in enumerate(record["checkpoints"]):
        path = root / ("cert_%d.json" % c["timestep"])
        _cert(path, c["sha256"], k=(64 if i in (0, 3) else 10))
        certs[c["timestep"]] = str(path)
    doc = CE.select(out, certs, root / "sel")
    assert doc["best"]["timestep"] == record["checkpoints"][3]["timestep"] and not doc["best_is_final"]
    assert doc["best_copy_sha256"] == record["checkpoints"][3]["sha256"]
    with pytest.raises(FX.FixError):                               # never overwrite
        CE.select(out, certs, root / "sel")
    bad = root / "bad.json"
    _cert(bad, "0" * 64, k=1)
    with pytest.raises(FX.FixError):
        CE.select(out, {record["checkpoints"][0]["timestep"]: str(bad)}, root / "sel2")


def test_install_refuses_without_a_scheduler():
    root = scratch_dir()
    plan, args = G.plan_for(root / "run", task="journal:scatter")
    backend = G.fake_backend(4)
    original = backend.make_runner

    def runner_without_scheduler(wrapped, c):
        r = original(wrapped, c)
        r.agent.scheduler = None
        return r
    backend.make_runner = runner_without_scheduler
    assert T.run_training(plan, args, backend) == T.EXIT_INFRA
