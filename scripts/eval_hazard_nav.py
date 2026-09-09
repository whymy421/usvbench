"""Deterministic evaluation for HazardNav checkpoints.

SPL uses the environment's obstacle-aware route_geodesic_length rather than
the obstructed straight-line start/goal distance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a HazardNav checkpoint.")
parser.add_argument("--task", default="Isaac-USV-HazardNav-Direct-v3")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_level", type=int, default=1)
parser.add_argument("--output_json", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
    parser.error("--episodes and --num_envs must be positive")
checkpoint_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(checkpoint_path):
    parser.error(f"checkpoint does not exist: {checkpoint_path}")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import skrl  # noqa: E402,F401
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _base_env(env):
    current = env
    for _ in range(10):
        if hasattr(current, "episode_success"):
            return current
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
        elif hasattr(current, "env"):
            current = current.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _values_at(base, name: str, env_ids: torch.Tensor, default) -> list:
    value = getattr(base, name, None)
    if value is None:
        return [default] * int(env_ids.numel())
    tensor = torch.as_tensor(value, device=env_ids.device).reshape(-1)
    return tensor[env_ids].detach().cpu().tolist()


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.eval_level
    env_cfg.log_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["agent"]["experiment"]["wandb"] = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, experiment_cfg)
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")
    base = _base_env(env)
    device = torch.device(getattr(base, "device", env_cfg.sim.device))

    observation, _ = env.reset()
    completed = 0
    successes = 0
    spl_values: list[float] = []
    success_times: list[float] = []
    path_lengths: list[float] = []
    min_clearances: list[float] = []
    reverse_action_steps = 0
    reverse_velocity_steps = 0
    active_steps = 0

    print(
        f"[INFO] deterministic eval: {args_cli.episodes} episodes, "
        f"level={args_cli.eval_level}, seed={args_cli.seed}"
    )
    while completed < args_cli.episodes:
        active = ~base._reached_goal
        with torch.inference_mode():
            outputs = runner.agent.act(observation, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            active_steps += int(active.sum().item())
            reverse_action_steps += int(((actions[:, 0] < 0.0) & active).sum().item())
            surge_norm, _, _ = base._body_motion_observation()
            reverse_velocity_steps += int(
                ((surge_norm.squeeze(-1) < 0.0) & active).sum().item()
            )
            observation, _, terminated, truncated, _ = env.step(actions)

        done = torch.as_tensor(terminated, device=device).reshape(-1).bool()
        done |= torch.as_tensor(truncated, device=device).reshape(-1).bool()
        done_ids = torch.nonzero(done, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        counted_ids = done_ids[: args_cli.episodes - completed]
        episode_success = [
            bool(value)
            for value in _values_at(base, "episode_success", counted_ids, False)
        ]
        episode_paths = [
            float(value)
            for value in _values_at(base, "episode_path_length", counted_ids, math.nan)
        ]
        episode_routes = [
            float(value)
            for value in _values_at(base, "route_geodesic_length", counted_ids, math.nan)
        ]
        episode_times = [
            float(value)
            for value in _values_at(base, "time_to_success", counted_ids, math.nan)
        ]
        episode_clearances = [
            float(value)
            for value in _values_at(base, "episode_min_clearance", counted_ids, math.nan)
        ]

        for success, path_length, route_length, time_s, clearance in zip(
            episode_success,
            episode_paths,
            episode_routes,
            episode_times,
            episode_clearances,
        ):
            successes += int(success)
            path_lengths.append(path_length)
            min_clearances.append(clearance)
            if success and math.isfinite(time_s):
                success_times.append(time_s)

            shortest_to_goal_region = max(route_length - float(env_cfg.goal_radius), 0.0)
            denominator = max(path_length, shortest_to_goal_region)
            spl_values.append(
                shortest_to_goal_region / denominator
                if success and denominator > 0.0
                else 0.0
            )

        completed += int(counted_ids.numel())

    finite_paths = [value for value in path_lengths if math.isfinite(value)]
    finite_clearances = [value for value in min_clearances if math.isfinite(value)]
    result = {
        "task": args_cli.task,
        "checkpoint": checkpoint_path,
        "checkpoint_name": os.path.basename(checkpoint_path),
        "seed": args_cli.seed,
        "eval_level": args_cli.eval_level,
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / completed,
        "spl": sum(spl_values) / completed,
        "mean_time_to_success_s": (
            sum(success_times) / len(success_times) if success_times else math.nan
        ),
        "median_time_to_success_s": (
            statistics.median(success_times) if success_times else math.nan
        ),
        "mean_path_length_m": (
            sum(finite_paths) / len(finite_paths) if finite_paths else math.nan
        ),
        "mean_min_clearance_m": (
            sum(finite_clearances) / len(finite_clearances)
            if finite_clearances
            else math.nan
        ),
        "reverse_action_ratio_pre_goal": reverse_action_steps / max(active_steps, 1),
        "reverse_velocity_ratio_pre_goal": reverse_velocity_steps / max(active_steps, 1),
    }

    print("\n" + "=" * 64)
    print(f"checkpoint                  : {result['checkpoint_name']}")
    print(f"SR                          : {result['success_rate']:.4f}")
    print(f"obstacle-aware SPL          : {result['spl']:.4f}")
    print(f"mean time to success (s)    : {result['mean_time_to_success_s']:.3f}")
    print(f"mean path length (m)        : {result['mean_path_length_m']:.3f}")
    print(f"mean min clearance (m)      : {result['mean_min_clearance_m']:.3f}")
    print(f"reverse action ratio        : {result['reverse_action_ratio_pre_goal']:.4f}")
    print(f"reverse velocity ratio      : {result['reverse_velocity_ratio_pre_goal']:.4f}")
    print("=" * 64)

    if args_cli.output_json:
        output_path = os.path.abspath(args_cli.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=True)
        print(f"[INFO] wrote {output_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
