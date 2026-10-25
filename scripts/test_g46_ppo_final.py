"""v2.6.4 PPO final CPU tests (G47; G46 plan 46 config P1 tests kept): discounted failure penalty per family
(bound, horizon, top speed); v2.6.4: the G46 learning-rate smoke switch is removed (refused when set),
KLAdaptiveLR only, checkpoints every 2 % by default (a campaign run refuses another interval), an EMPTY
USVBENCH_ASSETS_V2 is refused, the trainer integration (fake skrl backend of test_g19b: no LR floor, LR log,
init checkpoint, record agent_config.ppo_final) and the validation-seed rule of the best-checkpoint selection
(scripts/ppo_checkpoint_eval.py: campaign runs selection-02 / 11002 only and every checkpoint; smoke runs
selection-02 or diagnostic 42; final_test / pilot / calibration seeds always refused)."""
from __future__ import annotations

import importlib.util
import json
import math
import os
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
import ppo_final_g46 as G46  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _load("_g46_g19b_tests", SCRIPTS / "test_g19b_train_journal_ppo.py")
T, R = G.T, G.R


def scratch_dir():
    root = SCRIPTS.parents[1] / "test_scratch"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="g46-", dir=root))      # kept (no cleanup)


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
PLAN = {"length_scale": 2.0, "gate_radius_m": 2.0, "segment_count": 4}
BB = 3.000326502719915            # vehicles_v2_data PINNED_DERIVED blueboat_v2 speeds.top_model_m_s
FAMILIES = [("hazard_nav", HAZ), ("path_following", PF), ("path_hazard", PH), ("station_keeping", SK),
            ("docking", DK), ("harbor_mission", HM)]


def test_horizon_is_ceil_of_one_over_one_minus_gamma_capped_by_n():
    assert G46.discount_horizon(0.99, 3601) == (100, 100)          # 1/(1-0.99) = 100.00000000000009 -> 100
    assert G46.discount_horizon(0.999, 3601) == (1000, 1000)
    assert G46.discount_horizon(0.997, 3601) == (334, 334)
    assert G46.discount_horizon(0.999, 500) == (500, 1000)          # the continuation cannot outlast the episode
    with pytest.raises(G46.G46Error):
        G46.discount_horizon(1.0, 10)


def test_top_speed_from_the_pinned_vehicle_data():
    assert G46.top_speed_mps("blueboat_v2") == pytest.approx(BB)
    assert G46.top_speed_mps("wamv_vrx_v2_opt1") == pytest.approx(2.2589523496083515)
    with pytest.raises(G46.G46Error):
        G46.top_speed_mps("rov")


@pytest.mark.parametrize("carrier,fields", FAMILIES)
def test_every_family_derives_a_negative_discounted_penalty(carrier, fields):
    t = G46.discounted_terms(carrier, cfg(**fields), PLAN, 0.99, BB)
    assert t["H"] == 100 and t["N"] == 3601 and t["step_m"] == pytest.approx(BB / 60.0)
    assert t["A"] > 0.0 and t["B"] >= 0.0
    assert t["A"] + t["B"] * 50.0 == pytest.approx(t["H"] * (t["L_per_step"] + t["L_per_step_per_metre_D"] * 50.0)
                                                   + t["E"])
    g44 = FX.penalty_terms(carrier, cfg(**fields), PLAN)
    # the discounted bound is never larger in magnitude than the G44 undiscounted bound (H << N here)
    assert FX.failure_penalty(t, 50.0) >= FX.failure_penalty(g44, 50.0)
    assert FX.failure_penalty(t, 50.0) <= FX.failure_penalty(t, 0.0) < 0.0
    print("G46 SYNTHETIC %s A=%.4g B=%.4g P(D=50)=%.4g (G44 %.4g)" % (carrier, t["A"], t["B"],
                                                                      FX.failure_penalty(t, 50.0),
                                                                      FX.failure_penalty(g44, 50.0)))


def test_path_following_value_and_discounted_guarantee():
    t = G46.discounted_terms("path_following", cfg(**PF), PLAN, 0.99, BB)
    lp = 20.0 * (BB / 60.0) / 2.0
    assert t["L_per_step"] == pytest.approx(lp) and t["E"] == pytest.approx(110.0 + lp)
    assert t["A"] == pytest.approx(100 * lp + 110.0 + lp) and t["B"] == 0.0
    p = FX.failure_penalty(t, 80.0)
    gamma = 0.99
    # worst continuation that never fails: moves away from the gate at top speed for the whole episode
    worst = sum(gamma ** k * (-lp) for k in range(3601))
    best_fail = lp + 10.0 + 100.0 + p                           # one step of gain + gate + success events
    assert best_fail <= worst + 1e-9


def test_hazard_nav_value_and_discounted_telescoping_guarantee():
    c = cfg(**HAZ)
    t = G46.discounted_terms("hazard_nav", c, PLAN, 0.999, BB)
    dense = (1.0 / 60.0) * (4.1 * math.log(1.35 / 0.45) + 0.25 + 0.05)
    assert t["H"] == 1000 and t["A"] == pytest.approx(20.0 + 1000 * dense + 100.0) and t["B"] == 0.0
    assert t["parts"]["speed_used"] is False
    gamma = 0.999
    for phi0 in (0.0, -0.3, -1.0):          # potential Phi in [-1, 0]; worst continuation drifts to Phi = -1
        cont = sum(gamma ** k * 20.0 * ((-1.0 if k == 0 else -1.0) - (phi0 if k == 0 else -1.0)) for k in range(3601))
        cont -= sum(gamma ** k * dense for k in range(3601))
        fail = 20.0 * (0.0 - phi0) + 100.0 + FX.failure_penalty(t, 0.0)    # best failing step reaches Phi = 0
        assert fail <= cont + 1e-6


def test_station_keeping_and_docking_scale_with_the_bounds():
    t = G46.discounted_terms("station_keeping", cfg(**SK), PLAN, 0.99, BB)
    assert t["A"] == pytest.approx(1.0) and t["B"] == pytest.approx(100.0 / 30.0)
    d = G46.discounted_terms("docking", cfg(**DK), PLAN, 0.997, BB)
    assert d["H"] == 334 and d["B"] == pytest.approx(334 / 25.0) and d["A"] == pytest.approx(334 * 0.5 + 3.9)


def test_refusals():
    with pytest.raises(FX.FixError):
        G46.discounted_terms("hazard_nav", cfg(**dict(HAZ, reward_threading_amplitude=5.0)), PLAN, 0.99, BB)
    with pytest.raises(FX.FixError):
        G46.discounted_terms("unknown", cfg(), PLAN, 0.99, BB)
    with pytest.raises(FX.FixError):
        G46.discounted_terms("path_following", cfg(**PF), PLAN, 0.99, 0.0)


YAML_CFG = {"agent": {"discount_factor": 0.99, "learning_rate": 1.5e-4, "learning_rate_scheduler": "KLAdaptiveLR",
                     "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008}}}


def test_block_is_kl_only_and_refuses_the_removed_switch():
    b = G46.block("training_smoke", YAML_CFG, "blueboat_v2", 96000, environ={})
    assert b["lr_mode"] == {"mode": "kl", "lr": None} and b["lr_floor"] is None and b["gamma"] == 0.99
    assert b["checkpoint_parts"] == 50 and b["recommended_checkpoint_every"] == 1920
    assert b["validation_blocks"] == [{"namespace": "selection", "block_id": "selection-02", "seed": 11002},
                                      {"namespace": "diagnostic", "block_id": None, "seed": 42}]
    c = G46.block("training", YAML_CFG, "wamv_vrx_v2_opt1", 192000, environ={})
    assert c["recommended_checkpoint_every"] == 3840
    assert c["validation_blocks"] == [{"namespace": "selection", "block_id": "selection-02", "seed": 11002}]
    for value in ("kl", "fixed:1.5e-4", "fixed:3e-5"):
        with pytest.raises(G46.G46Error):
            G46.block("training_smoke", YAML_CFG, "blueboat_v2", 96000, environ={G46.REMOVED_ENV_VAR: value})
    assert G46.removed_switch_error({G46.REMOVED_ENV_VAR: "  "}) is None
    bad = json.loads(json.dumps(YAML_CFG))
    bad["agent"]["learning_rate_scheduler"] = None
    with pytest.raises(G46.G46Error):
        G46.block("training_smoke", bad, "blueboat_v2", 96000, environ={})
    for name in ("parse_lr_mode", "apply_lr_mode", "selected", "FIXED_ALLOWED", "ENV_VAR"):
        assert not hasattr(G46, name), name


def test_checkpoint_default_is_two_percent_and_campaign_refuses_another_interval():
    assert T.CHECKPOINT_PARTS == G46.CHECKPOINT_PARTS == 50
    assert T.DEFAULT_BUDGET == {"blueboat": 96000, "wamv": 192000}
    for budget in T.DEFAULT_BUDGET.values():
        assert budget % 50 == 0 and G46.checkpoint_every_error("training", budget, budget // 50) is None
        assert G46.checkpoint_every_error("training", budget, budget // 10) is not None
        assert G46.checkpoint_every_error("training_smoke", budget, budget // 10) is None
    root = scratch_dir()
    args = G.args_for(root / "dry", "--dry-run", task="journal:path_following")      # no --checkpoint-every
    plan = T.preflight(args, root=T.namespace_tasks(), reference_block=G.NO_MATCH, code_tree_sha256=G.tree_sha())
    assert plan["checkpoint_every"] == 1920 and plan["budget_timesteps"] == 96000
    assert plan["g46"]["version"] == G46.G46_VERSION and plan["g46"]["lr_mode"]["mode"] == "kl"


def test_empty_assets_env_is_refused_unset_and_set_are_fine():
    assert G46.assets_env_error({}) is None
    assert G46.assets_env_error({"USVBENCH_ASSETS_V2": "/x/assets_v2"}) is None
    assert G46.assets_env_error({"USVBENCH_ASSETS_V2": ""}) is not None
    assert G46.assets_env_error({"USVBENCH_ASSETS_V2": "  "}) is not None
    old = os.environ.get("USVBENCH_ASSETS_V2")
    try:
        os.environ["USVBENCH_ASSETS_V2"] = ""
        root = scratch_dir()
        args = G.args_for(root / "dry", "--dry-run", task="journal:path_following")
        with pytest.raises(T.TrainingRefusal):
            T.preflight(args, root=T.namespace_tasks(), reference_block=G.NO_MATCH, code_tree_sha256=G.tree_sha())
    finally:
        if old is None:
            os.environ.pop("USVBENCH_ASSETS_V2", None)
        else:
            os.environ["USVBENCH_ASSETS_V2"] = old


def _with_env(value):
    old = os.environ.get(G46.REMOVED_ENV_VAR)
    if value is None:
        os.environ.pop(G46.REMOVED_ENV_VAR, None)
    else:
        os.environ[G46.REMOVED_ENV_VAR] = value
    return old


def _restore(old):
    if old is None:
        os.environ.pop(G46.REMOVED_ENV_VAR, None)
    else:
        os.environ[G46.REMOVED_ENV_VAR] = old


def test_kl_run_has_no_floor_logs_lr_saves_init_and_records_ppo_final():
    old = _with_env(None)
    try:
        root = scratch_dir()
        out = root / "run"
        plan, args = G.plan_for(out, task="journal:path_following")
        assert plan["g46"]["lr_mode"]["mode"] == "kl" and plan["g46"]["gamma"] == 0.99
        assert T.run_training(plan, args, G.fake_backend(4)) == T.EXIT_OK
        record = R.load_record(out / "training_record.json", run_dir=out)
        fix = json.loads((out / "ppo_fix.json").read_text(encoding="ascii"))
        assert fix["learning_rate_floor"]["disabled"] and fix["learning_rate_log"]["scheduler"] == "SimpleNamespace"
        lo = fix["learning_rate_log"]
        assert lo["updates_seen"] > 0 and lo["lowest_lr_seen"] == pytest.approx(1.0e-6)   # fake min_lr, no floor
        terms = fix["failure_terminal_penalty"]["terms"]
        assert terms["H"] == 100 and terms["top_speed_mps"] == pytest.approx(BB)
        assert fix["failure_terminal_penalty"]["rule"] == G46.PENALTY_RULE
        assert fix["validation_rule"] == G46.VALIDATION_RULE
        assert (out / G46.INIT_CHECKPOINT).is_file() and fix["g46"]["init_checkpoint"]["path"] == G46.INIT_CHECKPOINT
        final = record["agent_config"]["ppo_final"]
        assert final["lr_mode"]["mode"] == "kl" and final["lr_floor"] is None and final["version"] == G46.G46_VERSION
        assert final["checkpoint_every"] == plan["checkpoint_every"] and "g46_lr" not in record["agent_config"]
        assert record["agent_config"]["config"]["agent"]["learning_rate_scheduler"] == "KLAdaptiveLR"
    finally:
        _restore(old)


def test_the_removed_switch_is_refused_at_preflight_and_at_run():
    old = _with_env("fixed:3e-5")
    try:
        root = scratch_dir()
        with pytest.raises(T.TrainingRefusal):
            G.plan_for(root / "run", task="journal:path_following")
        os.environ.pop(G46.REMOVED_ENV_VAR, None)
        plan, args = G.plan_for(root / "run2", task="journal:path_following")
        os.environ[G46.REMOVED_ENV_VAR] = "kl"
        assert T.run_training(plan, args, G.fake_backend(4)) == T.EXIT_INFRA      # set after the preflight
    finally:
        _restore(old)


def test_a_scheduler_less_agent_is_refused():
    old = _with_env(None)
    try:
        root = scratch_dir()
        plan, args = G.plan_for(root / "run", task="journal:path_following")
        backend = G.fake_backend(4)
        original = backend.make_runner

        def no_scheduler(wrapped, c):
            r = original(wrapped, c)
            r.agent.scheduler = None
            return r
        backend.make_runner = no_scheduler
        assert T.run_training(plan, args, backend) == T.EXIT_INFRA
    finally:
        _restore(old)


# ------------------------------------------------------------------ validation-seed rule of the selection
CE = _load("_g47_checkpoint_eval", SCRIPTS / "ppo_checkpoint_eval.py")
SEL02 = {"namespace": "selection", "purpose": "selection", "block_id": "selection-02", "seed": 11002,
         "role": "reserve"}
DIAG42 = {"namespace": "diagnostic", "purpose": "diagnostic", "seed": 42, "block_id": None,
          "also_registered_as": "training"}
GATE = {"namespace": "selection", "purpose": "selection", "block_id": "selection-gate-01", "seed": 11001}
FINAL = {"namespace": "final_test", "purpose": "final_test", "block_id": "test-main-01", "seed": 41001}
PILOT = {"namespace": "dose_pilots", "purpose": "pilot", "block_id": "pilot-01", "seed": 21001}


def _fake_run(root, purpose, steps=(1920, 3840, 5760)):
    run = root / ("run_%s" % purpose)
    (run / "ck").mkdir(parents=True)
    entries = []
    for s in steps:
        p = run / "ck" / ("agent_%d.pt" % s)
        p.write_bytes(b"checkpoint %d" % s)
        entries.append({"timestep": s, "path": "ck/agent_%d.pt" % s, "sha256": CE.sha256(p)})
    record = {"purpose": purpose, "checkpoints": entries,
              "final_checkpoint": {"path": "final.pt", "sha256": entries[-1]["sha256"],
                                   "source_path": entries[-1]["path"], "source_sha256": entries[-1]["sha256"]}}
    (run / "training_record.json").write_text(json.dumps(record), encoding="ascii")
    return run, entries


def _certs(root, entries, stamp, ks, tag):
    certs = {}
    for e, k in zip(entries, ks):
        path = root / ("%s_%d.json" % (tag, e["timestep"]))
        doc = {"header": {"complete": True, "checkpoint": {"sha256": e["sha256"]}, "eval_seed": stamp["seed"],
                          "seed_namespace": stamp},
               "records": [{"success": i < k} for i in range(64)]}
        path.write_text(json.dumps(doc), encoding="utf-8")
        certs[e["timestep"]] = str(path)
    return certs


def test_campaign_selection_uses_selection_02_only_and_every_checkpoint():
    root = scratch_dir()
    run, entries = _fake_run(root, "training")
    doc = CE.select(run, _certs(root, entries, SEL02, (64, 64, 3), "s"), root / "sel")
    assert doc["best"]["timestep"] == 3840 and not doc["best_is_final"]            # tie 64/64 -> the later one
    assert doc["final_evaluation"]["timestep"] == 5760 and doc["final_evaluation"]["successes"] == 3
    assert doc["purpose"] == "training" and doc["validation_rule"] == G46.VALIDATION_RULE
    for bad, tag in ((DIAG42, "d"), (GATE, "g"), (FINAL, "f"), (PILOT, "p")):
        with pytest.raises(FX.FixError):
            CE.select(run, _certs(root, entries, bad, (64, 64, 64), tag), root / ("sel_" + tag))
    partial = _certs(root, entries[:2], SEL02, (64, 64), "part")
    with pytest.raises(FX.FixError):                                             # every checkpoint is evaluated
        CE.select(run, partial, root / "sel_part")
    no_stamp = root / "nostamp.json"
    no_stamp.write_text(json.dumps({"header": {"complete": True, "checkpoint": {"sha256": entries[0]["sha256"]},
                                               "eval_seed": 11002}, "records": [{"success": True}] * 64}),
                        encoding="utf-8")
    with pytest.raises(FX.FixError):
        CE.select(run, {**_certs(root, entries, SEL02, (1, 1, 1), "ns"), entries[0]["timestep"]: str(no_stamp)},
                  root / "sel_ns")


def test_smoke_selection_allows_selection_02_or_diagnostic_42_never_test_seeds():
    root = scratch_dir()
    run, entries = _fake_run(root, "training_smoke")
    assert CE.select(run, _certs(root, entries, DIAG42, (1, 9, 9), "d"), root / "sd")["best"]["timestep"] == 5760
    assert CE.select(run, _certs(root, entries[:2], SEL02, (9, 1), "s"), root / "ss")["best"]["timestep"] == 1920
    for bad, tag in ((FINAL, "f"), (PILOT, "p"), (GATE, "g")):
        with pytest.raises(FX.FixError):
            CE.select(run, _certs(root, entries, bad, (64, 64, 64), tag), root / ("x" + tag))
    mixed = {**_certs(root, entries[:1], DIAG42, (1,), "m1"), **_certs(root, entries[1:], SEL02, (1, 1), "m2")}
    with pytest.raises(FX.FixError):                                             # one seed per selection
        CE.select(run, mixed, root / "xm")
