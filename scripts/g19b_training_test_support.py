"""G19b follow-up C1: CPU test support (not a test module) - drive a REAL family journal env (its CPU fixture,
stubbed Isaac Lab) through train_journal_ppo.StepMonitor and inject a non-finite pose into one row, exactly as the
REV-G19b probe does (probes/probe_nan_reward.py: env 0 root-link pose and COM velocity set to NaN). The env then
ends that row with the journal code 9 (non_finite_state) and a non-finite reward; the training wrapper must replace
that reward by the family's terminal failure penalty, read from the env's OWN cfg by
train_journal_ppo.code9_replacement (verify2 C1), and count it, without raising Divergence."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

SCRIPTS = Path(__file__).resolve().parent
_TRAINER = None


def trainer():
    global _TRAINER
    if _TRAINER is None:
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        spec = importlib.util.spec_from_file_location("_g19b_trainer_for_tests", SCRIPTS / "train_journal_ppo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TRAINER = module
    return _TRAINER


_G46_YAML = {"hazard_nav": "hazard_nav/agents/skrl_ppo_cfg.yaml", "path_following": "path_following/agents/skrl_ppo_cfg.yaml",
             "path_hazard": "path_hazard/agents/skrl_ppo_cfg.yaml", "station_keeping": "station_keeping/agents/skrl_ppo_cfg.yaml",
             "harbor_mission": "harbor_mission/agents/skrl_ppo_cfg.yaml", "docking": "docking/agents/skrl_ppo_cfg.yaml"}


def _g46_profile(cfg):
    name = str(getattr(cfg, "vehicle", ""))
    return "blueboat_v2" if name.startswith("blueboat") else (name if name.startswith("wamv_vrx_v2") else "wamv_vrx_v2_opt1")


def _g46_gamma(carrier, cfg):
    rel = _G46_YAML[carrier]
    if carrier == "docking" and _g46_profile(cfg) == "blueboat_v2":
        rel = "docking/agents/skrl_ppo_blueboat_cfg.yaml"
    text = (SCRIPTS.parent / "tasks" / rel).read_text(encoding="utf-8")
    return float(re.search(r"^\s*discount_factor:\s*([0-9.eE+-]+)", text, re.M).group(1))


def code9_run(env, carrier, *, inject_at=3, steps=6, width=2):
    """Returns (monitor, steps) with steps = [(rewards seen by the agent, termination codes)] per env step; the
    replacement is the family's penalty read from env.cfg (train_journal_ppo.code9_replacement)."""
    T = trainer()
    n = int(env.num_envs)
    wrapped = SimpleNamespace(step=env.step, num_envs=n)
    value, source = T.code9_replacement(env.cfg, carrier)
    plan = {"budget_timesteps": 100000, "rollouts": 32, "num_envs": n, "code9_reward": value,
            "code9_source": source}
    scratch = Path(tempfile.mkdtemp(prefix="g19b_code9_", dir=str(SCRIPTS.parents[1] / "test_scratch")
                                    if (SCRIPTS.parents[1] / "test_scratch").is_dir() else None))
    monitor = T.StepMonitor(wrapped, env, plan, scratch, lambda: None)
    # G44 (v2.6.3): the failure terminal penalty derived from the REAL env cfg and bounds (R2); the code-9 row
    # gets the family's code-9 replacement PLUS that penalty
    # G46 (P1-b): the DISCOUNTED penalty the trainer installs (family yaml gamma, the vessel's top speed); printed
    gamma, profile = _g46_gamma(carrier, env.cfg), _g46_profile(env.cfg)
    terms = T.G46.discounted_terms(carrier, env.cfg, env.cfg.journal_plan.params, gamma, T.G46.top_speed_mps(profile))
    print("G46 PENALTY %s %s gamma=%s H=%d N=%d L=%.6g E=%.6g A=%.6g B=%.6g" % (carrier, profile, gamma, terms["H"],
          terms["N"], terms["L_per_step"], terms["E"], terms["A"], terms["B"]), flush=True)
    monitor.g44 = {"failure": T.FX.FailurePenalty(terms, env, carrier, n)}
    monitor.g44_applied = []
    out = []
    env.reset()
    try:
        for k in range(steps):
            if k == inject_at:
                data = env.robot.data
                pose = data.root_link_pose_w.clone()
                pose[0, :] = float("nan")
                data.root_link_pose_w = pose
                vel = data.root_com_vel_w.clone()
                vel[0, :] = float("nan")
                data.root_com_vel_w = vel
            result = wrapped.step(torch.zeros((n, width)))
            monitor.g44_applied.append(dict(monitor.g44["failure"].last))
            codes = getattr(env, "termination_reason", None)
            out.append((result[1].detach().clone().reshape(n, -1)[:, 0],
                        None if codes is None else [int(c) for c in codes.reshape(-1).tolist()]))
    finally:
        monitor.close()
    return monitor, out


def check_code9(case, monitor, steps, expected, inject_at=3):
    """The assertions shared by the family tests (expected: the family's penalty, -25.0 or 0.0). G44: the row
    reward is that replacement plus the failure terminal penalty of the episode (derived from the real cfg)."""
    rewards, codes = steps[inject_at]
    case.assertEqual(codes[0], 9, "the env must end the NaN row with code 9 (non_finite_state)")
    case.assertEqual(monitor.code9_reward, expected)
    penalty = monitor.g44_applied[inject_at][0]
    case.assertLess(penalty, 0.0)
    case.assertAlmostEqual(float(rewards[0]), expected + penalty, places=3)
    for k, (r, c) in enumerate(steps):                    # only failure codes are penalised
        for i in monitor.g44_applied[k]:
            case.assertIn(c[i], (2, 4, 5, 6, 7, 8, 9), (k, i))
    case.assertEqual(monitor.nonfinite_terminal()["replacement_source"]["value_rule"],
                     "-env_cfg.reward_contact_entry_penalty" if expected else
                     trainer().R.CODE9_NO_PENALTY_RULE)
    for r, _ in steps:
        case.assertTrue(bool(torch.isfinite(r).all()), "the agent must never see a non-finite reward")
    case.assertEqual(monitor.code9_rows, 1)
    case.assertEqual(sum(monitor.nonfinite_terminal()["per_component"].values()), 1)
