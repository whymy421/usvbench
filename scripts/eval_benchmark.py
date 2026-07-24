"""
USVBench standardized deterministic evaluation protocol.

The primary throughput metric is normalized by episode-equivalents so partial
rollouts remain comparable. ``oob_per_episode`` is an event-rate signal and can
exceed 1.0. The script separately reports ``oob_fraction_completed``, the true
fraction of completed episodes that ended out of bounds.

Use the same ``--seed`` for every training-seed checkpoint. ``--train_seed`` is
metadata only and records which training run produced the checkpoint.
"""

# ruff: noqa: E402  # Isaac Sim must launch before importing simulation modules

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="USVBench standardized deterministic evaluation.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--eval_steps", type=int, default=6000, help="Vector-environment steps to evaluate.")
parser.add_argument("--seed", type=int, default=2026, help="Fixed evaluation-scenario seed.")
parser.add_argument("--train_seed", type=int, default=None, help="Training seed metadata for reports/CSV.")
parser.add_argument("--csv", type=str, default=None, help="Append one result row to this CSV.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import math
import os

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


def _base_env(env):
    e = env
    for _ in range(12):
        if hasattr(e, "reached_count"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "_env"):
            e = e._env
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return e


def _set_eval_mode(agent):
    if hasattr(agent, "set_running_mode"):
        agent.set_running_mode("eval")
    elif hasattr(agent, "set_training_mode"):
        agent.set_training_mode(False)
    else:
        for model in agent.models.values():
            if hasattr(model, "eval"):
                model.eval()


def _observation_preprocessor(agent):
    for attr in (
        "_observation_preprocessor",
        "_state_preprocessor",
        "observation_preprocessor",
        "state_preprocessor",
    ):
        preprocessor = getattr(agent, attr, None)
        if preprocessor is not None and hasattr(preprocessor, "load_state_dict"):
            return attr, preprocessor
    return None, None


def _load_checkpoint_compat(agent, checkpoint_path):
    agent.load(checkpoint_path)
    attr, preprocessor = _observation_preprocessor(agent)
    used_compat = False

    if preprocessor is not None:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "observation_preprocessor" not in raw:
            for old_key in ("state_preprocessor", "_state_preprocessor"):
                if old_key in raw:
                    preprocessor.load_state_dict(raw[old_key])
                    used_compat = True
                    break

    print(f"[INFO] Observation preprocessor: {attr or 'none'}")
    if used_compat:
        print("[INFO] Loaded old-format state_preprocessor into the observation preprocessor")


def _seeded_reset(env, seed):
    base = _base_env(env)
    if hasattr(base, "reached_count"):
        base.reached_count = 0
    if hasattr(base, "episode_count"):
        base.episode_count = 0
    if hasattr(base, "_prev_distance"):
        base._prev_distance = None
    if hasattr(base, "_buf_idx"):
        base._buf_idx = 0
    if hasattr(base, "_action_buffer"):
        base._action_buffer = torch.zeros_like(base._action_buffer)
    if hasattr(base, "_prev_action_obs"):
        base._prev_action_obs = torch.zeros_like(base._prev_action_obs)
    if hasattr(base, "seed"):
        base.seed(seed)

    if hasattr(env, "_reset_once"):
        env._reset_once = True
        env._seed = seed
    else:
        raw_env = getattr(env, "_env", None)
        if raw_env is not None and hasattr(raw_env, "seed"):
            raw_env.seed(seed)
    return env.reset()


def _deterministic_action(agent, observations):
    policy_observations = observations.get("policy", observations) if isinstance(observations, dict) else observations
    try:
        outputs = agent.act(policy_observations, None, timestep=0, timesteps=0)
    except TypeError:
        outputs = agent.act(policy_observations, timestep=0, timesteps=0)
    actions = outputs[0]
    info = outputs[-1] if isinstance(outputs[-1], dict) else {}
    return info.get("mean_actions", actions)


def _step_reaches(base, previous_count):
    reached_mask = getattr(base, "_last_reached_mask", None)
    current_count = int(getattr(base, "reached_count", previous_count))
    if isinstance(reached_mask, torch.Tensor):
        return int(reached_mask.sum().item()), current_count
    return max(current_count - previous_count, 0), current_count


def _fmt_optional(value, digits=3):
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, experiment_cfg):
    if args_cli.eval_steps <= 0:
        raise ValueError("--eval_steps must be > 0")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    resume_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    _load_checkpoint_compat(runner.agent, resume_path)
    _set_eval_mode(runner.agent)

    base = _base_env(env)
    max_episode_length = float(getattr(base, "max_episode_length", 0)) or float(args_cli.eval_steps)
    observations, _ = _seeded_reset(env, args_cli.seed)
    previous_count = int(getattr(base, "reached_count", 0))

    total_targets = 0
    oob_episodes = 0
    timeout_episodes = 0
    speed_sum = 0.0
    speed_samples = 0
    completed_episode_steps = 0
    episode_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.device)

    print(
        f"[INFO] Evaluating {args_cli.eval_steps} steps x {args_cli.num_envs} envs "
        f"with fixed seed {args_cli.seed}"
    )
    for _ in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = _deterministic_action(runner.agent, observations)
            observations, _, terminated, truncated, _ = env.step(actions)

        reached_now, previous_count = _step_reaches(base, previous_count)
        total_targets += reached_now

        terminated = torch.as_tensor(terminated).view(-1).bool()
        truncated = torch.as_tensor(truncated).view(-1).bool()
        done = terminated | truncated
        episode_steps += 1
        if done.any():
            completed_episode_steps += int(episode_steps[done].sum().item())
            episode_steps[done] = 0
        oob_episodes += int(terminated.sum().item())
        timeout_episodes += int((truncated & ~terminated).sum().item())

        try:
            velocity = base.robot.data.root_com_vel_w
            speed_sum += torch.norm(velocity[:, :2], dim=-1).mean().item()
            speed_samples += 1
        except Exception:
            pass

    episode_equivalents = (args_cli.num_envs * args_cli.eval_steps) / max(max_episode_length, 1.0)
    targets_per_episode = total_targets / max(episode_equivalents, 1e-9)
    oob_per_episode = oob_episodes / max(episode_equivalents, 1e-9)
    completed_episodes = oob_episodes + timeout_episodes
    oob_fraction_completed = oob_episodes / completed_episodes if completed_episodes else float("nan")
    mean_episode_len = completed_episode_steps / completed_episodes if completed_episodes else float("nan")
    mean_speed = speed_sum / max(speed_samples, 1)

    print("\n" + "=" * 68)
    print(f"  USVBench eval: {args_cli.task}")
    print(f"  checkpoint : {os.path.basename(resume_path)}")
    print(f"  train seed : {args_cli.train_seed if args_cli.train_seed is not None else 'n/a'}")
    print(f"  eval seed  : {args_cli.seed}")
    print("-" * 68)
    print(f"  targets_per_episode     : {targets_per_episode:8.3f}  (primary)")
    print(f"  mean_speed (m/s)        : {mean_speed:8.3f}")
    print(f"  oob_per_episode         : {oob_per_episode:8.3f}  (can exceed 1.0)")
    print(f"  oob_fraction_completed  : {_fmt_optional(oob_fraction_completed)}")
    print(f"  mean_episode_len        : {_fmt_optional(mean_episode_len, 1)} steps")
    print(f"  completed episodes      : {completed_episodes} ({oob_episodes} OOB, {timeout_episodes} timeout)")
    print(f"  total targets           : {total_targets}")
    print(f"  episode-equivalents     : {episode_equivalents:.1f}")
    print("=" * 68 + "\n")

    if args_cli.csv:
        csv_path = os.path.abspath(args_cli.csv)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        new_file = not os.path.exists(csv_path)
        row = {
            "task": args_cli.task,
            "train_seed": "" if args_cli.train_seed is None else args_cli.train_seed,
            "eval_seed": args_cli.seed,
            "targets_per_episode": f"{targets_per_episode:.6f}",
            "mean_speed": f"{mean_speed:.6f}",
            "oob_per_episode": f"{oob_per_episode:.6f}",
            "oob_fraction_completed": (
                "" if not math.isfinite(oob_fraction_completed) else f"{oob_fraction_completed:.6f}"
            ),
            "mean_episode_len": "" if not math.isfinite(mean_episode_len) else f"{mean_episode_len:.3f}",
            "completed_episodes": completed_episodes,
            "oob_episodes": oob_episodes,
            "timeout_episodes": timeout_episodes,
            "total_targets": total_targets,
            "episode_equivalents": f"{episode_equivalents:.6f}",
            "checkpoint": resume_path,
        }
        with open(csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"[INFO] Appended result to {csv_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
