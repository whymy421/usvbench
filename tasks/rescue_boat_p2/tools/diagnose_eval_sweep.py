"""
diagnose_eval_sweep.py — Diagnose why the P2 eval sweep returns 0.000 for most
checkpoints and non-zero at exactly agent_240000 and agent_540000 across seeds.

Four independent seeds scoring non-zero at identical checkpoints is not something
policy learning does, so this script tests the evaluation harness rather than the
policy. It runs four phases:

  PHASE 1  Determinism      Evaluate one checkpoint twice in a row.
                            Differing scores => the eval itself is not repeatable
                            (env state carries over between checkpoint evals).

  PHASE 2  Contamination    Evaluate A, then B, then A again.
                            A != A' => loading B leaves state that corrupts the
                            next eval (the prime suspect: observation preprocessor).

  PHASE 3  Preprocessor     Fingerprint the observation normaliser after every
                            load. If the fingerprint does not change between
                            checkpoints, agent.load() is silently not restoring
                            it and every policy is being fed wrongly-scaled
                            observations. Then force-restore it from the raw
                            checkpoint and re-evaluate.

  PHASE 4  Horizon bias     The sweep uses 1500 steps = 25 s of a 120 s episode,
                            but divides by a full-episode equivalent. It therefore
                            only ever measures the opening of an episode, when the
                            boat is furthest from every casualty. Re-evaluate over
                            a full episode and compare.

Usage (from the project root, with env_isaaclab active):

  python diagnose_eval_sweep.py --task=Isaac-RescueBoat-Direct-v1 --num_envs=64 \
      --headless --log_dir="logs/skrl/usvbench/<run folder>"

If --log_dir is omitted the most recently modified run under logs/skrl/usvbench
that contains a checkpoints/ folder is used.
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose the checkpoint eval sweep.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--log_dir", type=str, default=None,
                    help="Run folder containing checkpoints/. Defaults to most recent.")
parser.add_argument("--ckpt_a", type=str, default="agent_300000.pt",
                    help="Checkpoint that scores 0.000 in the sweep.")
parser.add_argument("--ckpt_b", type=str, default="agent_540000.pt",
                    help="Checkpoint that scores non-zero in the sweep.")
parser.add_argument("--eval_steps", type=int, default=1500,
                    help="Steps per eval, matching train_with_eval.py's default sweep.")
parser.add_argument("--full_episode_steps", type=int, default=7200,
                    help="Steps for the phase 4 full-episode eval.")
parser.add_argument("--skip_phase4", action="store_true", default=False,
                    help="Skip the full-episode eval (phase 4 is the slow one).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs import multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = "skrl_cfg_entry_point"

SEP = "=" * 78
SUB = "-" * 78


# ── helpers (mirroring train_with_eval.py so behaviour matches the real sweep) ──

def _base_env(env):
    e = env
    for _ in range(10):
        if hasattr(e, "reached_count"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _get_obs_prep(agent):
    for attr in ("_obs_preprocessor", "_observation_preprocessor", "observation_preprocessor"):
        cand = getattr(agent, attr, None)
        if cand is not None and callable(cand):
            return cand
    return None


def _prep_fingerprint(agent):
    """Summarise the observation normaliser so we can see whether load() changed it."""
    prep = _get_obs_prep(agent)
    if prep is None:
        return "no-preprocessor"
    if not hasattr(prep, "running_mean"):
        return "preprocessor-without-running-mean"
    rm = prep.running_mean.detach().float()
    rv = getattr(prep, "running_variance", None)
    rv_s = f" var_sum={rv.detach().float().sum().item():+.6f}" if rv is not None else ""
    return (f"mean_sum={rm.sum().item():+.6f} mean_absmax={rm.abs().max().item():.6f}"
            f"{rv_s} all_zero={bool(torch.all(rm == 0))}")


def _force_restore_preprocessor(agent, ckpt_path):
    """
    Unconditionally reload the observation preprocessor from the raw checkpoint.

    train_with_eval.py only does this when running_mean is exactly all-zeros, so a
    checkpoint that leaves stale values from the previously-loaded checkpoint slips
    through silently. This is the suspected root cause.
    """
    prep = _get_obs_prep(agent)
    if prep is None or not hasattr(prep, "load_state_dict"):
        return "no-preprocessor-to-restore"
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"could-not-read-checkpoint: {e}"
    for k in ("observation_preprocessor", "_observation_preprocessor",
              "state_preprocessor", "_state_preprocessor"):
        if k in raw:
            try:
                prep.load_state_dict(raw[k])
                return f"restored-from '{k}'"
            except Exception as e:
                return f"found '{k}' but load failed: {e}"
    return f"no preprocessor key in checkpoint (keys: {sorted(raw.keys())[:8]})"


def _mini_eval(agent, env, n_steps, num_envs):
    """Identical scoring to train_with_eval.py's sweep."""
    obs_prep = _get_obs_prep(agent)
    base = _base_env(env)
    max_ep = float(getattr(base, "max_episode_length", n_steps))

    obs, _ = env.reset()
    reached_total = 0
    reached_prev = int(getattr(base, "reached_count", 0))

    for _ in range(n_steps):
        with torch.inference_mode():
            obs_raw = obs["policy"] if isinstance(obs, dict) else obs
            obs_in = obs_prep(obs_raw, train=False) if obs_prep is not None else obs_raw
            _acts, _info = agent.policy.act({"observations": obs_in}, role="policy")
            actions = _info.get("mean_actions", _acts)
            obs, _, _, _, _ = env.step(actions)

        cur = int(getattr(base, "reached_count", 0))
        d = cur - reached_prev
        if d > 0:
            reached_total += d
        reached_prev = cur

    ep_equiv = (num_envs * n_steps) / max(max_ep, 1.0)
    return reached_total / max(ep_equiv, 1e-9), reached_total, ep_equiv


def _load_and_eval(agent, env, ckpt_path, n_steps, num_envs, force_prep=False, label=""):
    agent.load(ckpt_path)
    note = ""
    if force_prep:
        note = _force_restore_preprocessor(agent, ckpt_path)
    fp = _prep_fingerprint(agent)
    score, raw_count, ep_equiv = _mini_eval(agent, env, n_steps, num_envs)
    print(f"  {label:<34s} {score:6.3f} tgt/ep   ({raw_count} rescues / {ep_equiv:.2f} ep-equiv)")
    print(f"      preprocessor: {fp}")
    if note:
        print(f"      force-restore: {note}")
    return score


def _resolve_log_dir(explicit):
    if explicit:
        return os.path.abspath(explicit)
    root = os.path.abspath(os.path.join("logs", "skrl", "usvbench"))
    cands = [d for d in glob.glob(os.path.join(root, "*"))
             if os.path.isdir(os.path.join(d, "checkpoints"))]
    if not cands:
        raise SystemExit(f"No run folders with checkpoints/ found under {root}")
    return max(cands, key=os.path.getmtime)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    log_dir = _resolve_log_dir(args_cli.log_dir)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    path_a = os.path.join(ckpt_dir, args_cli.ckpt_a)
    path_b = os.path.join(ckpt_dir, args_cli.ckpt_b)

    print()
    print(SEP)
    print("  EVAL SWEEP DIAGNOSTIC")
    print(SEP)
    print(f"  run folder : {log_dir}")
    print(f"  ckpt A     : {args_cli.ckpt_a}  (scores 0.000 in the recorded sweep)")
    print(f"  ckpt B     : {args_cli.ckpt_b}  (scores non-zero in the recorded sweep)")
    print(f"  eval steps : {args_cli.eval_steps}")
    print(SEP)

    for p in (path_a, path_b):
        if not os.path.exists(p):
            print(f"\n  MISSING: {p}")
            print("  Available checkpoints:")
            for c in sorted(glob.glob(os.path.join(ckpt_dir, "agent_*.pt"))):
                print("   ", os.path.basename(c))
            raise SystemExit(1)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    runner = Runner(env, agent_cfg)
    agent = runner.agent
    for m in agent.models.values():
        if hasattr(m, "eval"):
            m.eval()

    n = args_cli.eval_steps
    ne = args_cli.num_envs
    base = _base_env(env)
    max_ep = float(getattr(base, "max_episode_length", 0))

    print(f"\n  max_episode_length = {max_ep:.0f} steps")
    print(f"  eval covers {n} steps = {100.0 * n / max(max_ep, 1):.1f}% of one episode\n")

    results = {}

    # ── PHASE 1 ───────────────────────────────────────────────────────────────
    print(SUB)
    print("  PHASE 1 — Is a single evaluation repeatable?")
    print(SUB)
    b1 = _load_and_eval(agent, env, path_b, n, ne, label=f"{args_cli.ckpt_b} (1st)")
    b2 = _load_and_eval(agent, env, path_b, n, ne, label=f"{args_cli.ckpt_b} (2nd)")
    results["phase1"] = (b1, b2)
    print()
    if abs(b1 - b2) < 1e-9:
        print("  -> Repeatable. Eval noise is not the explanation.")
    else:
        print(f"  -> NOT repeatable: {b1:.3f} vs {b2:.3f} (delta {abs(b1 - b2):.3f}).")
        print("     The same weights score differently, so the sweep is measuring")
        print("     leftover environment state, not policy quality.")

    # ── PHASE 2 ───────────────────────────────────────────────────────────────
    print()
    print(SUB)
    print("  PHASE 2 — Does loading one checkpoint corrupt the next eval?")
    print(SUB)
    a1 = _load_and_eval(agent, env, path_a, n, ne, label=f"{args_cli.ckpt_a} (fresh)")
    _ = _load_and_eval(agent, env, path_b, n, ne, label=f"{args_cli.ckpt_b} (interposed)")
    a2 = _load_and_eval(agent, env, path_a, n, ne, label=f"{args_cli.ckpt_a} (after B)")
    results["phase2"] = (a1, a2)
    print()
    if abs(a1 - a2) < 1e-9:
        print("  -> No contamination from load order.")
    else:
        print(f"  -> CONTAMINATION: {args_cli.ckpt_a} scored {a1:.3f} fresh but {a2:.3f}")
        print("     after loading B. Sweep results depend on evaluation order, so the")
        print("     recorded per-checkpoint numbers are not comparable.")

    # ── PHASE 3 ───────────────────────────────────────────────────────────────
    print()
    print(SUB)
    print("  PHASE 3 — Is the observation preprocessor actually being restored?")
    print(SUB)
    print("  Fingerprints above should differ between A and B. If they are identical,")
    print("  agent.load() is not restoring the normaliser and every policy after the")
    print("  first is being fed wrongly-scaled observations.\n")
    a3 = _load_and_eval(agent, env, path_a, n, ne, force_prep=True,
                        label=f"{args_cli.ckpt_a} (forced prep)")
    b3 = _load_and_eval(agent, env, path_b, n, ne, force_prep=True,
                        label=f"{args_cli.ckpt_b} (forced prep)")
    results["phase3"] = (a3, b3)
    print()
    if a3 > a1 + 1e-9:
        print(f"  -> ROOT CAUSE CONFIRMED: {args_cli.ckpt_a} rose from {a1:.3f} to {a3:.3f}")
        print("     once the preprocessor was force-restored. The 0.000 scores are a")
        print("     checkpoint-loading bug, NOT late-training policy degradation.")
        print("     Every number in the sweep needs regenerating before it is reported.")
    else:
        print(f"  -> Forcing the preprocessor did not raise {args_cli.ckpt_a}")
        print(f"     ({a1:.3f} -> {a3:.3f}). The zeros are more likely genuine.")

    # ── PHASE 4 ───────────────────────────────────────────────────────────────
    if not args_cli.skip_phase4:
        print()
        print(SUB)
        print("  PHASE 4 — Does the 1500-step window understate performance?")
        print(SUB)
        print(f"  The sweep measures only the first {n} steps ({100.0 * n / max(max_ep, 1):.1f}%) of an")
        print("  episode, when the boat is furthest from every casualty, then divides by a")
        print("  full-episode equivalent. Re-evaluating over a whole episode:\n")
        full = int(args_cli.full_episode_steps)
        b_full = _load_and_eval(agent, env, path_b, full, ne, force_prep=True,
                               label=f"{args_cli.ckpt_b} ({full} steps)")
        results["phase4"] = (b3, b_full)
        print()
        if b_full > b3 + 1e-9:
            print(f"  -> The short window understates performance: {b3:.3f} -> {b_full:.3f}")
            print("     over a full episode. Reported rescue rates are biased low.")
        else:
            print(f"  -> Full-episode eval gives {b_full:.3f} vs {b3:.3f}; window length")
            print("     is not the dominant issue.")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    p1a, p1b = results["phase1"]
    p2a, p2b = results["phase2"]
    p3a, p3b = results["phase3"]
    print(f"  repeatable eval        : {'YES' if abs(p1a - p1b) < 1e-9 else 'NO'}  ({p1a:.3f} / {p1b:.3f})")
    print(f"  order-dependent        : {'NO' if abs(p2a - p2b) < 1e-9 else 'YES'}  ({p2a:.3f} fresh / {p2b:.3f} after B)")
    print(f"  preprocessor was fault : {'YES' if p3a > p2a + 1e-9 else 'NO'}  ({p2a:.3f} -> {p3a:.3f})")
    if "phase4" in results:
        s, f = results["phase4"]
        print(f"  short-window bias      : {'YES' if f > s + 1e-9 else 'NO'}  ({s:.3f} -> {f:.3f} full episode)")
    print(SEP)
    print("  If any row reads YES apart from 'repeatable eval', the recorded")
    print("  eval_sweep.csv values are unsafe to report and the sweep must be rerun")
    print("  with the fix before these numbers go into the thesis.")
    print(SEP)
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
