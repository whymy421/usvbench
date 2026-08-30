"""
eval_robustness.py — Zero-shot robustness evaluation across sea states.

Takes policies trained in calm water and evaluates them, without any retraining,
under progressively harsher environmental disturbance. The output is a curve of
rescue rate against sea state: the sim2real transfer gap this project exists to
quantify.

Because no training happens, each sea state costs one evaluation pass rather than
a full training run, so the whole sweep finishes in minutes.

The disturbance itself lives in rescue_boat_env.py and is driven by environment
variables (CURRENT_SPEED, WAVE_AMP, WAVE_YAW_AMP, GUST_STD). Those are read at
module import, so a separate process is required per sea state -- this script
launches them and collects the results. Run it through run_robustness.ps1 rather
than directly if you want the whole sweep.

Single sea state (called by the runner):

  python eval_robustness.py --task=Isaac-RescueBoat-Direct-v1 --num_envs=64 \
      --headless --checkpoint="<path to .pt>" --sea_state=moderate \
      --eval_steps=7200 --out=robustness_results.csv

Note on eval length: the default is a full episode (7200 steps), not the 1500
used by the training sweep. A short window only ever samples the opening of an
episode, when the boat is furthest from every casualty, which biases rescue rate
downward. Robustness numbers should not inherit that bias.
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

# Sea states, roughly aligned to the Douglas scale. Each entry sets the env vars
# consumed by rescue_boat_env.py at import time.
SEA_STATES = {
    "calm":     dict(CURRENT_SPEED="0.0", WAVE_AMP="0.0",   WAVE_YAW_AMP="0.0",   GUST_STD="0.0"),
    "slight":   dict(CURRENT_SPEED="0.5", WAVE_AMP="150.0", WAVE_YAW_AMP="200.0", GUST_STD="25.0"),
    "moderate": dict(CURRENT_SPEED="1.0", WAVE_AMP="300.0", WAVE_YAW_AMP="400.0", GUST_STD="50.0"),
    "rough":    dict(CURRENT_SPEED="1.5", WAVE_AMP="500.0", WAVE_YAW_AMP="650.0", GUST_STD="90.0"),
}

parser = argparse.ArgumentParser(description="Zero-shot robustness evaluation across sea states.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint.")
parser.add_argument("--sea_state", type=str, default="calm", choices=sorted(SEA_STATES.keys()))
parser.add_argument("--eval_steps", type=int, default=7200,
                    help="Steps per evaluation. Default 7200 = one full episode.")
parser.add_argument("--label", type=str, default="", help="Run label recorded in the CSV (e.g. seed).")
parser.add_argument("--out", type=str, default="robustness_results.csv")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Sea state must be applied to os.environ BEFORE the env module is imported,
# since rescue_boat_env.py reads these at module scope.
for k, v in SEA_STATES[args_cli.sea_state].items():
    os.environ[k] = v

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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


def _force_restore_preprocessor(agent, ckpt_path):
    """
    Restore the observation normaliser explicitly rather than relying on load().

    The training sweep's fallback only fires when running_mean is exactly zero and
    only looks for the legacy 'state_preprocessor' key, which these checkpoints do
    not use. Restoring unconditionally here removes that failure mode from the
    robustness numbers regardless of how the sweep issue is resolved.
    """
    prep = _get_obs_prep(agent)
    if prep is None or not hasattr(prep, "load_state_dict"):
        return "none"
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"unreadable: {e}"
    for k in ("observation_preprocessor", "_observation_preprocessor",
              "state_preprocessor", "_state_preprocessor"):
        if k in raw:
            try:
                prep.load_state_dict(raw[k])
                return f"restored '{k}'"
            except Exception as e:
                return f"'{k}' load failed: {e}"
    return "no preprocessor key found"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    state = SEA_STATES[args_cli.sea_state]
    print()
    print("=" * 74)
    print(f"  ROBUSTNESS EVAL — sea state: {args_cli.sea_state.upper()}")
    print("=" * 74)
    print(f"  checkpoint : {os.path.basename(ckpt)}")
    print(f"  label      : {args_cli.label or '(none)'}")
    print(f"  current    : {state['CURRENT_SPEED']} m/s")
    print(f"  wave amp   : {state['WAVE_AMP']} N   yaw {state['WAVE_YAW_AMP']} N-m")
    print(f"  gust std   : {state['GUST_STD']} N")
    print(f"  eval steps : {args_cli.eval_steps}")
    print("=" * 74)

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
    agent.load(ckpt)
    note = _force_restore_preprocessor(agent, ckpt)
    print(f"  preprocessor: {note}")
    for m in agent.models.values():
        if hasattr(m, "eval"):
            m.eval()

    obs_prep = _get_obs_prep(agent)
    base = _base_env(env)
    max_ep = float(getattr(base, "max_episode_length", args_cli.eval_steps))
    n_casualties = int(getattr(base.cfg, "n_casualties", 4))

    obs, _ = env.reset()
    reached_total = 0
    reached_prev = int(getattr(base, "reached_count", 0))

    n = args_cli.eval_steps
    for i in range(n):
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

        if (i + 1) % 1200 == 0:
            print(f"    step {i + 1}/{n}  rescues so far: {reached_total}")

    ep_equiv = (args_cli.num_envs * n) / max(max_ep, 1.0)
    tgt_per_ep = reached_total / max(ep_equiv, 1e-9)
    rescue_rate = tgt_per_ep / max(n_casualties, 1)

    print()
    print("-" * 74)
    print(f"  RESULT  {args_cli.sea_state:<9s}  {tgt_per_ep:.3f} tgt/ep   "
          f"rescue_rate {rescue_rate:.3f}")
    print(f"          ({reached_total} rescues over {ep_equiv:.2f} episode-equivalents)")
    print("-" * 74)
    print()

    out = os.path.abspath(args_cli.out)
    new = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "checkpoint", "sea_state", "current_speed", "wave_amp",
                        "wave_yaw_amp", "gust_std", "eval_steps", "rescues",
                        "episode_equiv", "targets_per_episode", "rescue_rate"])
        w.writerow([args_cli.label, os.path.basename(ckpt), args_cli.sea_state,
                    state["CURRENT_SPEED"], state["WAVE_AMP"], state["WAVE_YAW_AMP"],
                    state["GUST_STD"], n, reached_total, f"{ep_equiv:.4f}",
                    f"{tgt_per_ep:.4f}", f"{rescue_rate:.4f}"])
    print(f"  appended to {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
