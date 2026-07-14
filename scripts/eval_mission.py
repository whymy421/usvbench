"""Standardized evaluation protocol for terminating USVBench missions.

MISSION-style tasks finish each episode with success or failure. This evaluator
counts a fixed number of completed episodes, uses deterministic policy actions,
and reports success rate (SR), success-weighted path length (SPL), success time,
and failure types.

Examples:
  python scripts/eval_mission.py --task=Isaac-USV-StationKeep-Direct-v1 \
    --checkpoint=<path>/best_agent.pt --num_envs=64 --episodes=128 --headless

  python scripts/eval_mission.py --task=Isaac-USV-StationKeep-Direct-v1 \
    --policy=zero --num_envs=2 --episodes=2 --headless
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="USVBench standardized mission evaluation.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, required=True)
policy_group = parser.add_mutually_exclusive_group(required=True)
policy_group.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.pt).")
policy_group.add_argument(
    "--policy",
    type=str,
    choices=["zero"],
    help="Use zero actions without loading a checkpoint (dry-run mode).",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=128,
    help="Number of completed episodes to include.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--csv", type=str, default=None, help="If set, append one result row to this CSV.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import skrl  # noqa: F401
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: F401, E501
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _base_env(env):
    """Dig through wrappers to the underlying mission environment."""
    e = env
    for _ in range(10):
        if hasattr(e, "episode_success") or hasattr(e, "path_length"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            # resolve against THIS repo, not the caller's process cwd
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cfg_fingerprint(env_cfg) -> tuple[str, dict[str, object]]:
    """Hash protocol-relevant scalar config fields in a stable order."""
    names = {"episode_length_s", "hold_radius", "goal_radius"}
    names.update(
        name
        for name in dir(env_cfg)
        if "spawn" in name.lower() and "distance" in name.lower()
    )

    scalars = {}
    for name in sorted(names):
        if name.startswith("_") or not hasattr(env_cfg, name):
            continue
        value = getattr(env_cfg, name)
        if isinstance(value, (bool, int, float, str)):
            scalars[name] = value

    encoded = json.dumps(scalars, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10], scalars


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
        distances = torch.as_tensor(helper(), device=device, dtype=torch.float32).reshape(-1)
        if distances.numel() != num_envs:
            raise ValueError(f"expected {num_envs} distances, got {distances.numel()}")
        return distances.clone()
    except Exception as exc:
        print(f"[WARN] Could not read initial horizontal distance ({exc}); SPL will be nan.")
        return torch.full((num_envs,), torch.nan, device=device)


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
    # DirectRLEnv auto-resets done envs inside step(). StationKeep therefore
    # snapshots path_length before reset as episode_path_length. Other mission
    # envs may leave path_length readable, so retain it as a graceful fallback.
    name = "episode_path_length" if hasattr(base, "episode_path_length") else "path_length"
    return [float(value) for value in _values_at(base, name, env_ids, math.nan)]


def _zero_actions(base, num_envs: int, device) -> torch.Tensor:
    if hasattr(base, "actions"):
        return torch.zeros_like(base.actions)

    action_space = getattr(base.cfg, "action_space", None)
    if isinstance(action_space, int):
        shape = (num_envs, action_space)
    elif hasattr(action_space, "shape"):
        shape = (num_envs, *action_space.shape)
    else:
        raise RuntimeError("Cannot infer zero-action shape from the mission environment")
    return torch.zeros(shape, device=device)


def _append_csv(path: str, repro: dict[str, object], metrics: dict[str, object]) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [f"# {key}={value}" if index == 0 else f"{key}={value}"
             for index, (key, value) in enumerate(repro.items())]
        )
        if new_file:
            writer.writerow(
                [
                    "task",
                    "checkpoint",
                    "seed",
                    "episodes",
                    "sr",
                    "spl",
                    "mean_time_to_success_s",
                    "median_time_to_success_s",
                    "successes",
                    "failures_timeout",
                    "failures_other",
                    "cfg_sha1",
                    "git_commit",
                ]
            )
        writer.writerow(
            [
                repro["task"],
                repro["checkpoint"],
                repro["seed"],
                repro["episodes"],
                f"{metrics['sr']:.6f}",
                f"{metrics['spl']:.6f}",
                f"{metrics['mean_time_to_success_s']:.6f}",
                f"{metrics['median_time_to_success_s']:.6f}",
                metrics["successes"],
                metrics["failures_timeout"],
                metrics["failures_other"],
                repro["cfg_sha1"],
                repro["git_commit"],
            ]
        )


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    resume_path = None
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
        env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    cfg_sha1, cfg_scalars = _cfg_fingerprint(env_cfg)
    checkpoint_name = os.path.basename(resume_path) if resume_path else "zero"
    repro = {
        "git_commit": _git_commit(),
        "task": args_cli.task,
        "checkpoint": checkpoint_name,
        "episodes": args_cli.episodes,
        "seed": args_cli.seed,
        "cfg_sha1": cfg_sha1,
    }
    print("[REPRO] " + "  ".join(f"{key}={value}" for key, value in repro.items()))
    print("[REPRO] cfg_scalars=" + json.dumps(cfg_scalars, sort_keys=True, separators=(",", ":")))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    runner = None
    if resume_path:
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.set_running_mode("eval")

    base = _base_env(env)
    device = torch.device(getattr(base, "device", env_cfg.sim.device))
    goal_radius = _radius_from_cfg(env_cfg)
    if not math.isfinite(goal_radius):
        print("[WARN] cfg has neither hold_radius nor goal_radius; SPL will be nan.")
    if not hasattr(base, "_horizontal_distance"):
        print("[WARN] Mission env has no _horizontal_distance(); SPL will be nan.")

    obs, _ = env.reset()
    initial_distance = _initial_distances(base, args_cli.num_envs, device)
    zero_actions = _zero_actions(base, args_cli.num_envs, device) if runner is None else None

    completed = 0
    successes = 0
    failures_timeout = 0
    failures_other = 0
    spl_values = []
    success_times = []

    policy_label = "zero actions" if runner is None else "deterministic"
    print(f"[INFO] Evaluating {args_cli.episodes} completed episodes x {args_cli.num_envs} envs ({policy_label})...")
    while completed < args_cli.episodes:
        with torch.inference_mode():
            if runner is None:
                actions = zero_actions
            else:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])  # deterministic
            obs, _, terminated, truncated, _ = env.step(actions)

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
                # A done without either flag should not occur, but keep the
                # failure partition exhaustive if a wrapper supplies one.
                failures_other += 1

            if math.isfinite(goal_radius) and math.isfinite(d0) and math.isfinite(path_length):
                shortest_path = max(float(d0) - goal_radius, 0.0)
                denominator = max(float(path_length), shortest_path)
                spl = shortest_path / denominator if success and denominator > 0.0 else 0.0
                spl_values.append(spl)
            else:
                spl_values.append(math.nan)

        completed += int(counted_ids.numel())
        if completed >= args_cli.episodes:
            break

        # DirectRLEnv has already reset every done env. Capture the next d0 now;
        # completed episodes beyond the requested count are deliberately ignored.
        next_distance = _initial_distances(base, args_cli.num_envs, device)
        initial_distance[done_ids] = next_distance[done_ids]

    sr = successes / completed
    spl = sum(spl_values) / completed if all(math.isfinite(value) for value in spl_values) else math.nan
    mean_time = sum(success_times) / len(success_times) if success_times else math.nan
    median_time = statistics.median(success_times) if success_times else math.nan
    metrics = {
        "sr": sr,
        "spl": spl,
        "mean_time_to_success_s": mean_time,
        "median_time_to_success_s": median_time,
        "successes": successes,
        "failures_timeout": failures_timeout,
        "failures_other": failures_other,
    }

    print("\n" + "=" * 64)
    print(f"  USVBench mission eval - {args_cli.task}")
    print(f"  checkpoint: {checkpoint_name}  seed={args_cli.seed}  episodes={completed}")
    print("-" * 64)
    print(f"  SR                         : {sr:8.4f}  ({successes}/{completed})")
    print(f"  SPL                        : {spl:8.4f}")
    print(f"  mean time to success (s)   : {mean_time:8.3f}")
    print(f"  median time to success (s) : {median_time:8.3f}")
    print(f"  failures - timeout         : {failures_timeout:8d}")
    print(f"  failures - other           : {failures_other:8d}")
    print("=" * 64 + "\n")
    sys.stdout.flush()

    if args_cli.csv:
        _append_csv(args_cli.csv, repro, metrics)
        print(f"[INFO] appended result to {args_cli.csv}")
        sys.stdout.flush()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
