"""
eval_fixed.py — Corrected checkpoint evaluation for the rescue boat task.

Replaces the sweep in train_with_eval.py, which the diagnostic showed to be
unusable. Three faults are fixed here:

  1. NORMALISER NOT RESTORED
     agent.load() did not reliably restore the observation preprocessor, and the
     fallback in train_with_eval.py only fired when running_mean was exactly zero
     and only looked for the legacy 'state_preprocessor' key, which these
     checkpoints do not use. A checkpoint could therefore be scored while being
     fed observations normalised for a different checkpoint. Here the
     preprocessor is restored explicitly, every time, and the script aborts if it
     cannot be.

  2. NO TRUE RESET BETWEEN CHECKPOINTS
     Scores depended on evaluation order (A scored 0.000 fresh and 0.600 after
     loading B). Here every checkpoint starts from a hard reset of all envs, and
     a burn-in period is discarded so no rescue from a previous evaluation can be
     counted.

  3. INFLATED DENOMINATOR
     The old sweep ran 1500 steps but divided by (num_envs * 1500) / 7200 =
     13.33 "episode-equivalents". 1500 steps is 25 s of a 120 s episode, and
     almost every rescue that will ever occur has already occurred by then, so
     the numerator was near-complete while the denominator was scaled down by
     4.6x. Here each env runs whole episodes and the metric is simply

         rescue_rate = rescues / (num_envs * n_casualties * episodes)

     which is the definition the P2 bar of 0.70 refers to.

Usage — score every checkpoint in a run folder:

  python eval_fixed.py --task=Isaac-RescueBoat-Direct-v1 --num_envs=64 --headless \
      --log_dir="logs/skrl/usvbench/<run folder>" --episodes=1

Writes eval_fixed.csv into the run folder and prints a ranked table.
"""

import argparse
import csv
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Corrected checkpoint evaluation.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--seed", type=int, default=12345,
                    help="Eval seed. Fixed across checkpoints so all see the same scenarios.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--log_dir", type=str, default=None,
                    help="Run folder containing checkpoints/. Defaults to most recent.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Score a single checkpoint instead of the whole folder.")
parser.add_argument("--episodes", type=int, default=1,
                    help="Complete episodes per env per checkpoint (64 envs x 1 = 64 episodes).")
parser.add_argument("--burn_in", type=int, default=60,
                    help="Steps discarded after reset before counting begins.")
parser.add_argument("--out", type=str, default=None, help="CSV path. Defaults to <log_dir>/eval_fixed.csv")
parser.add_argument("--update_best", action="store_true", default=False,
                    help="Copy the winning checkpoint over best_agent.pt.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import shutil

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = "skrl_cfg_entry_point"


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


def _restore_preprocessor(agent, ckpt_path):
    """Restore the observation normaliser explicitly. Returns (ok, message)."""
    prep = _get_obs_prep(agent)
    if prep is None:
        return True, "no preprocessor in use"
    if not hasattr(prep, "load_state_dict"):
        return True, "preprocessor not stateful"
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"checkpoint unreadable: {e}"
    for k in ("observation_preprocessor", "_observation_preprocessor",
              "state_preprocessor", "_state_preprocessor"):
        if k in raw:
            try:
                prep.load_state_dict(raw[k])
                return True, f"restored '{k}'"
            except Exception as e:
                return False, f"'{k}' present but load failed: {e}"
    return False, f"no preprocessor key (have: {sorted(raw.keys())[:6]})"


def _hard_reset(env, base, burn_in, agent, obs_prep):
    """
    Reset every env and discard a short burn-in.

    A plain reset() proved insufficient: scores depended on which checkpoint was
    evaluated previously. Stepping through a burn-in with the loaded policy and
    ignoring the result guarantees that nothing counted below originated in a
    previous evaluation.
    """
    obs, _ = env.reset()
    for _ in range(burn_in):
        with torch.inference_mode():
            obs_raw = obs["policy"] if isinstance(obs, dict) else obs
            obs_in = obs_prep(obs_raw, train=False) if obs_prep is not None else obs_raw
            _a, _i = agent.policy.act({"observations": obs_in}, role="policy")
            obs, _, _, _, _ = env.step(_i.get("mean_actions", _a))
    obs, _ = env.reset()
    return obs


def _evaluate(agent, env, base, num_envs, n_casualties, episodes, max_ep, burn_in):
    """
    Score over completed episodes and return (rescue_rate, targets_per_episode,
    rescues, denominator).

    Rescues are harvested from the environment's own per-episode `rescued_count`
    at the moment each episode ends, rather than by differencing a global counter
    over a fixed step window.

    The window approach is fragile: if the reset before counting does not fully
    take effect, the window is offset from the episode boundary, each environment
    crosses into a second episode, fresh casualties spawn, and the count exceeds
    the number of casualties that existed. That produced rescue_rate = 1.0156 —
    260 rescues against a denominator of 256 — which is how the fault was found.

    Sampling at episode end is immune to window alignment: each environment
    contributes exactly one observation per completed episode, and each
    observation is bounded by n_casualties by construction.
    """
    obs_prep = _get_obs_prep(agent)
    obs = _hard_reset(env, base, burn_in, agent, obs_prep)

    target_eps = int(episodes)
    # Per-episode rescue counts, collected per environment.
    collected = [[] for _ in range(num_envs)]
    prev_len = base.episode_length_buf.clone()

    # Generous step budget: episodes may not be aligned after the reset, so run
    # until every environment has contributed the required number, with a cap.
    max_steps = int(max_ep) * (target_eps + 2)

    for i in range(max_steps):
        with torch.inference_mode():
            obs_raw = obs["policy"] if isinstance(obs, dict) else obs
            obs_in = obs_prep(obs_raw, train=False) if obs_prep is not None else obs_raw
            _a, _inf = agent.policy.act({"observations": obs_in}, role="policy")
            rescued_before = base.rescued_count.clone()
            obs, _, _, _, _ = env.step(_inf.get("mean_actions", _a))

        # An environment that has just reset has a lower episode_length_buf than
        # on the previous step; its rescued_count from before the step is that
        # episode's final tally.
        cur_len = base.episode_length_buf
        just_done = (cur_len < prev_len).nonzero(as_tuple=False).flatten()
        for e in just_done.tolist():
            if len(collected[e]) < target_eps:
                collected[e].append(float(rescued_before[e]))
        prev_len = cur_len.clone()

        if all(len(c) >= target_eps for c in collected):
            break

        if (i + 1) % 2400 == 0:
            done_n = sum(len(c) for c in collected)
            print(f"      step {i + 1}/{max_steps}  episodes complete "
                  f"{done_n}/{num_envs * target_eps}")

    flat = [v for c in collected for v in c[:target_eps]]
    n_eps = len(flat)
    if n_eps == 0:
        print("      WARNING: no episode completed within the step budget")
        return 0.0, 0.0, 0, 0

    if n_eps < num_envs * target_eps:
        print(f"      NOTE: {n_eps} episodes completed of "
              f"{num_envs * target_eps} requested; scoring on those.")

    rescues = int(round(sum(flat)))
    denom = n_eps * n_casualties
    rescue_rate = rescues / max(denom, 1)
    tgt_per_ep = rescues / max(n_eps, 1)

    # rescue_rate above 1.0 is impossible; if it occurs the estimator is wrong.
    assert rescue_rate <= 1.0 + 1e-9, (
        f"rescue_rate {rescue_rate:.4f} exceeds 1.0 "
        f"({rescues} rescues / {denom}) — estimator is counting across episodes")

    return rescue_rate, tgt_per_ep, rescues, denom


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

    if args_cli.checkpoint:
        ckpts = [os.path.abspath(args_cli.checkpoint)]
    else:
        ckpts = sorted(
            glob.glob(os.path.join(ckpt_dir, "agent_*.pt")),
            key=lambda p: int(os.path.basename(p).replace("agent_", "").replace(".pt", "")),
        )
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {ckpt_dir}")

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

    base = _base_env(env)
    max_ep = float(getattr(base, "max_episode_length", 7200))
    n_cas = int(getattr(base.cfg, "n_casualties", 4))
    ne = args_cli.num_envs
    eps = args_cli.episodes

    print()
    print("=" * 78)
    print("  CORRECTED CHECKPOINT EVALUATION")
    print("=" * 78)
    print(f"  run          : {log_dir}")
    print(f"  checkpoints  : {len(ckpts)}")
    print(f"  episodes/env : {eps}   ({ne * eps} complete episodes per checkpoint)")
    print(f"  casualties   : {n_cas}  -> denominator {ne * n_cas * eps} per checkpoint")
    print(f"  episode len  : {max_ep:.0f} steps")
    print(f"  metric       : rescue_rate = rescues / (envs x casualties x episodes)")
    print(f"  P2 bar       : 0.70")
    print("=" * 78)
    print()

    rows = []
    for ck in ckpts:
        name = os.path.basename(ck)
        print(f"  {name}")
        agent.load(ck)
        ok, msg = _restore_preprocessor(agent, ck)
        print(f"      preprocessor: {msg}")
        if not ok:
            print("      SKIPPED - cannot trust a score without the normaliser")
            rows.append((name, None, None, 0, 0, msg))
            continue

        rr, tpe, resc, denom = _evaluate(agent, env, base, ne, n_cas, eps, max_ep, args_cli.burn_in)
        print(f"      rescue_rate {rr:.4f}   ({resc}/{denom})   {tpe:.3f} tgt/ep")
        print()
        rows.append((name, rr, tpe, resc, denom, msg))

    scored = [r for r in rows if r[1] is not None]
    scored.sort(key=lambda r: r[1], reverse=True)

    print("=" * 78)
    print("  RANKED RESULTS")
    print("=" * 78)
    print(f"  {'checkpoint':<24s} {'rescue_rate':>12s} {'tgt/ep':>9s} {'rescues':>10s}")
    print("  " + "-" * 60)
    for name, rr, tpe, resc, denom, _m in scored:
        flag = "  <- best" if (scored and name == scored[0][0]) else ""
        print(f"  {name:<24s} {rr:>12.4f} {tpe:>9.3f} {resc:>6d}/{denom:<4d}{flag}")
    print()

    if scored:
        best_name, best_rr = scored[0][0], scored[0][1]
        print(f"  Best: {best_name} at rescue_rate {best_rr:.4f}")
        print(f"  P2 bar 0.70 -> {'MET' if best_rr >= 0.70 else f'{best_rr / 0.70 * 100:.0f}% of bar'}")
        if args_cli.update_best:
            dst = os.path.join(ckpt_dir, "best_agent.pt")
            shutil.copy2(os.path.join(ckpt_dir, best_name), dst)
            print(f"  Copied to {dst}")
    print("=" * 78)

    out = args_cli.out or os.path.join(log_dir, "eval_fixed.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "rescue_rate", "targets_per_episode",
                    "rescues", "denominator", "preprocessor"])
        for name, rr, tpe, resc, denom, m in rows:
            w.writerow([name,
                        f"{rr:.4f}" if rr is not None else "",
                        f"{tpe:.4f}" if tpe is not None else "",
                        resc, denom, m])
    print(f"  Written: {out}")
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
