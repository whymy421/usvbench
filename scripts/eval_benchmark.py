"""
USVBench — standardized evaluation protocol.

Runs a trained checkpoint with a DETERMINISTIC policy (mean actions) for a fixed
number of steps and reports the benchmark metrics. Built on Isaac Lab's play.py
loading logic so checkpoint loading matches exactly.

Protocol (report like this in the paper):
  - Run this on 3 seed-trained checkpoints (42 / 123 / 456).
  - Report mean ± std ACROSS the 3 seeds for each metric.

Metrics:
  - targets_per_episode : primary score (total targets reached / episode-equivalent)
  - mean_speed          : avg planar speed over the rollout (m/s)
  - oob_per_episode     : out-of-bounds events per episode-equivalent. Control-quality
                          signal — in calm water this is overshoot, not a safety failure
                          (safety only becomes meaningful under waves). Can exceed 1.0.
  - mean_episode_len    : avg steps per completed episode

Example (ROV, calm):
  $env:OBS_DIM="3"
  python eval_benchmark.py --task=Isaac-My-First-Task-Calm-Direct-v1 \
    --num_envs=64 --eval_steps=6000 --headless \
    --checkpoint=<path>/best_agent.pt

Example (boat V26):
  $env:OBS_DIM="9"; $env:OBS_EXTENDED="1"; $env:REWARD_VARIANT="V23"
  $env:SPEED_COUPLE="1"; $env:REACH_BONUS="50.0"
  python eval_benchmark.py --task=Isaac-My-First-Task-Calm-Boat-Direct-v1 \
    --num_envs=64 --eval_steps=6000 --headless \
    --checkpoint=<path>/best_agent.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="USVBench standardized evaluation.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
parser.add_argument("--eval_steps", type=int, default=6000, help="Total env steps to evaluate over.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--csv", type=str, default=None, help="If set, append one result row to this CSV.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym
import skrl
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _base_env(env):
    """Dig through wrappers to the underlying Isaac env that has reached_count."""
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


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    resume_path = os.path.abspath(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    base = _base_env(env)

    # episode length in steps (continuous-retargeting tasks rarely terminate inside
    # the window, so we normalise by "episode-equivalents" instead of completed episodes)
    max_ep = float(getattr(base, "max_episode_length", 0)) or float(args_cli.eval_steps)

    obs, _ = env.reset()
    reached_total = 0
    reached_prev = int(getattr(base, "reached_count", 0))
    n_oob = 0           # out-of-bounds terminations (the env's only non-timeout done)
    n_timeout = 0
    speed_sum, speed_n = 0.0, 0

    print(f"[INFO] Evaluating {args_cli.eval_steps} steps x {args_cli.num_envs} envs (deterministic)...")
    for t in range(args_cli.eval_steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])  # deterministic
            obs, _, terminated, truncated, _ = env.step(actions)

        # accumulate targets via positive deltas (robust to the env's internal counter reset)
        cur = int(getattr(base, "reached_count", 0))
        d = cur - reached_prev
        if d > 0:
            reached_total += d
        reached_prev = cur

        try:
            vel = base.robot.data.root_com_vel_w
            speed_sum += torch.norm(vel[:, :2], dim=-1).mean().item()
            speed_n += 1
        except Exception:
            pass

        n_oob += int(torch.as_tensor(terminated).sum().item())
        n_timeout += int(torch.as_tensor(truncated).sum().item())

    # episode-equivalents = total env-steps / steps-per-episode
    ep_equiv = (args_cli.num_envs * args_cli.eval_steps) / max(max_ep, 1.0)
    tgt_per_ep = reached_total / max(ep_equiv, 1e-9)
    oob_per_episode = n_oob / max(ep_equiv, 1e-9)   # events per episode-equivalent (can be >1)
    mean_speed = speed_sum / max(speed_n, 1)

    print("\n" + "=" * 56)
    print(f"  USVBench eval — {args_cli.task}")
    print(f"  checkpoint: {os.path.basename(resume_path)}  seed={args_cli.seed}")
    print("-" * 56)
    print(f"  targets_per_episode : {tgt_per_ep:7.2f}   (primary score)")
    print(f"  mean_speed (m/s)    : {mean_speed:7.2f}")
    print(f"  oob_per_episode     : {oob_per_episode:7.2f}   (out-of-bounds events / episode)")
    print(f"  total_targets       : {reached_total}")
    print(f"  episode_equivalents : {ep_equiv:7.1f}   (ep_len={max_ep:.0f} steps)")
    print("=" * 56 + "\n")

    if args_cli.csv:
        new = not os.path.exists(args_cli.csv)
        with open(args_cli.csv, "a", encoding="utf-8") as f:
            if new:
                f.write("task,seed,targets_per_episode,mean_speed,oob_per_episode,total_targets\n")
            f.write(f"{args_cli.task},{args_cli.seed},{tgt_per_ep:.3f},{mean_speed:.3f},{oob_per_episode:.4f},{reached_total}\n")
        print(f"[INFO] appended result to {args_cli.csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
