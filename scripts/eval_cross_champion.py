"""Evaluate an unchanged component champion on a superset-emitting target.

The target environment emits the frozen 51-D observation.  Before every
policy call this runner selects the champion task's original observation order,
so the loaded checkpoint keeps exactly the input dimensionality it was trained
with.

Example:
  python scripts/eval_cross_champion.py \
    --target-task=Isaac-USV-PathHazard-Direct-v1 \
    --champion-task=Isaac-USV-PathFollow-Direct-v1 \
    --champion-checkpoint=<path>/best_agent.pt \
    --num_envs=64 --episodes=128 --headless
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import numpy as np


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tasks._shared.obs_superset import (  # noqa: E402
    NATIVE_LAYOUTS,
    SUPERSET_DIM,
    extract_native,
)

from isaaclab.app import AppLauncher


CERTIFICATE_TARGETS = {
    "Isaac-USV-HazardNav-Direct-v1",
    "Isaac-USV-HazardNav-Direct-v2",
    "Isaac-USV-PathHazard-Direct-v1",
}

parser = argparse.ArgumentParser(
    description="USVBench zero-shot cross-champion certificate evaluation."
)
parser.add_argument("--target-task", type=str, required=True)
parser.add_argument("--champion-checkpoint", type=str, required=True)
parser.add_argument("--champion-task", type=str, required=True)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")
if args_cli.target_task not in CERTIFICATE_TARGETS:
    parser.error(
        "--target-task must currently be one of: "
        + ", ".join(sorted(CERTIFICATE_TARGETS))
    )
if args_cli.champion_task not in NATIVE_LAYOUTS:
    parser.error(
        "--champion-task has no frozen native layout: "
        f"{args_cli.champion_task!r}"
    )

checkpoint_path = os.path.abspath(args_cli.champion_checkpoint)
if not os.path.isfile(checkpoint_path):
    parser.error(f"--champion-checkpoint does not exist: {checkpoint_path}")

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
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


AGENT_CFG_ENTRY_POINT = "skrl_cfg_entry_point"


def _base_env(env):
    """Dig through wrappers to the underlying mission environment."""
    current = env
    for _ in range(10):
        if hasattr(current, "episode_success") or hasattr(current, "path_length"):
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
    try:
        tensor = torch.as_tensor(value, device=env_ids.device).reshape(-1)
        return tensor[env_ids].detach().cpu().tolist()
    except Exception:
        return [default] * int(env_ids.numel())


def _completed_path_lengths(base, env_ids: torch.Tensor) -> list[float]:
    name = "episode_path_length" if hasattr(base, "episode_path_length") else "path_length"
    return [float(value) for value in _values_at(base, name, env_ids, math.nan)]


def _radius_from_cfg(env_cfg) -> float:
    for name in ("hold_radius", "goal_radius"):
        value = getattr(env_cfg, name, None)
        if value is not None:
            return float(value)
    return math.nan


def _initial_distances(base, num_envs: int, device) -> torch.Tensor:
    helper = getattr(base, "_horizontal_distance", None)
    if helper is None:
        return torch.full((num_envs,), torch.nan, device=device)
    try:
        distances = torch.as_tensor(
            helper(), device=device, dtype=torch.float32
        ).reshape(-1)
        if distances.numel() != num_envs:
            raise ValueError(f"expected {num_envs} distances, got {distances.numel()}")
        return distances.clone()
    except Exception as exc:
        print(f"[WARN] Could not read initial horizontal distance ({exc}); SPL will be nan.")
        return torch.full((num_envs,), torch.nan, device=device)


def _set_native_runner_space(env, native_dim: int) -> None:
    """Give Runner the checkpoint's input size without changing target output."""
    native_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(native_dim,),
        dtype=np.float32,
    )
    if hasattr(env, "_observation_space"):
        env._observation_space = native_space
    else:
        env.observation_space = native_space


def _champion_observation(superset_observation):
    return extract_native(superset_observation, args_cli.champion_task)


@hydra_task_config(args_cli.target_task, AGENT_CFG_ENTRY_POINT)
def main(env_cfg, _target_experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.seed = args_cli.seed
    env_cfg.emit_superset_obs = True
    env_cfg.observation_space = SUPERSET_DIM
    env_cfg.log_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    experiment_cfg = load_cfg_from_registry(
        args_cli.champion_task, AGENT_CFG_ENTRY_POINT
    )
    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.target_task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    native_dim = len(NATIVE_LAYOUTS[args_cli.champion_task])
    _set_native_runner_space(env, native_dim)
    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading {args_cli.champion_task} checkpoint: {checkpoint_path}")
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")

    base = _base_env(env)
    device = torch.device(getattr(base, "device", env_cfg.sim.device))
    goal_radius = _radius_from_cfg(env_cfg)
    if not hasattr(base, "_horizontal_distance"):
        print("[WARN] Target env has no _horizontal_distance(); SPL will be nan.")

    superset_observation, _ = env.reset()
    if superset_observation.shape[-1] != SUPERSET_DIM:
        raise RuntimeError(
            f"Target emitted {superset_observation.shape[-1]} observations; "
            f"expected frozen superset dimension {SUPERSET_DIM}"
        )
    observation = _champion_observation(superset_observation)
    initial_distance = _initial_distances(base, args_cli.num_envs, device)

    completed = 0
    successes = 0
    failures_timeout = 0
    failures_other = 0
    spl_values: list[float] = []
    success_times: list[float] = []

    print(
        f"[INFO] Evaluating {args_cli.episodes} completed {args_cli.target_task} "
        f"episodes x {args_cli.num_envs} envs with a deterministic "
        f"{args_cli.champion_task} champion ({native_dim}D from {SUPERSET_DIM}D)..."
    )
    while completed < args_cli.episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(observation, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            superset_observation, _, terminated, truncated, _ = env.step(actions)
            observation = _champion_observation(superset_observation)

        terminated = torch.as_tensor(terminated, device=device).reshape(-1).bool()
        truncated = torch.as_tensor(truncated, device=device).reshape(-1).bool()
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        remaining = args_cli.episodes - completed
        counted_ids = done_ids[:remaining]
        counted_terminated = terminated[counted_ids].detach().cpu().tolist()
        counted_truncated = truncated[counted_ids].detach().cpu().tolist()
        episode_success = [
            bool(value)
            for value in _values_at(base, "episode_success", counted_ids, False)
        ]
        episode_paths = _completed_path_lengths(base, counted_ids)
        episode_times = [
            float(value)
            for value in _values_at(base, "time_to_success", counted_ids, math.nan)
        ]
        episode_d0 = initial_distance[counted_ids].detach().cpu().tolist()

        for success, timed_out, terminated_flag, d0, path_length, time_s in zip(
            episode_success,
            counted_truncated,
            counted_terminated,
            episode_d0,
            episode_paths,
            episode_times,
        ):
            if success:
                successes += 1
                if math.isfinite(time_s):
                    success_times.append(time_s)
            elif timed_out:
                failures_timeout += 1
            elif terminated_flag:
                failures_other += 1
            else:
                failures_other += 1

            if (
                math.isfinite(goal_radius)
                and math.isfinite(d0)
                and math.isfinite(path_length)
            ):
                shortest_path = max(float(d0) - goal_radius, 0.0)
                denominator = max(float(path_length), shortest_path)
                spl = (
                    shortest_path / denominator
                    if success and denominator > 0.0
                    else 0.0
                )
                spl_values.append(spl)
            else:
                spl_values.append(math.nan)

        completed += int(counted_ids.numel())
        if completed >= args_cli.episodes:
            break

        next_distance = _initial_distances(base, args_cli.num_envs, device)
        initial_distance[done_ids] = next_distance[done_ids]

    sr = successes / completed
    spl = (
        sum(spl_values) / completed
        if all(math.isfinite(value) for value in spl_values)
        else math.nan
    )
    mean_time = (
        sum(success_times) / len(success_times) if success_times else math.nan
    )
    median_time = statistics.median(success_times) if success_times else math.nan

    print("\n" + "=" * 64)
    print(f"  USVBench cross-champion eval - {args_cli.target_task}")
    print(f"  champion: {args_cli.champion_task}")
    print(
        f"  checkpoint: {os.path.basename(checkpoint_path)}  "
        f"seed={args_cli.seed}  episodes={completed}"
    )
    print("-" * 64)
    print(f"  SR                         : {sr:8.4f}  ({successes}/{completed})")
    print(f"  SPL                        : {spl:8.4f}")
    print(f"  mean time to success (s)   : {mean_time:8.3f}")
    print(f"  median time to success (s) : {median_time:8.3f}")
    print(f"  failures - timeout         : {failures_timeout:8d}")
    print(f"  failures - other           : {failures_other:8d}")
    print("=" * 64 + "\n")
    sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
