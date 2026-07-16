"""Zero-learning LOS + PID baseline for USVBench.

The controller consumes only the policy observation: target dot/cross products
and normalized distance in observation columns 0:3. It supports the fixed-step
throughput protocol from eval_benchmark.py and the completed-episode mission
protocol from eval_mission.py.

The default gains are hand-set method-side choices, not tuned benchmark
parameters. Authors may tune their classical controller, but must report the
values used.

Examples:
  python scripts/classical_baseline.py \
    --task Isaac-My-First-Task-Calm-Direct-v1 --protocol throughput \
    --num_envs 64 --eval_steps 6000 --headless

  python scripts/classical_baseline.py \
    --task Isaac-USV-StationKeep-Direct-v1 --protocol mission \
    --num_envs 64 --episodes 128 --headless
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
from collections.abc import Mapping

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="USVBench classical LOS + PID baseline.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument(
    "--protocol",
    type=str,
    required=True,
    choices=["throughput", "mission"],
    help="Fixed-step targets/episode or completed-episode SR/SPL evaluation.",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128, help="Completed episodes for mission protocol.")
parser.add_argument("--eval_steps", type=int, default=6000, help="Steps for throughput protocol.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--csv", type=str, default=None, help="If set, append one result row to this CSV.")
parser.add_argument(
    "--dist-scale",
    type=float,
    default=None,
    help="Observation distance scale in metres (default: 30 point-nav, 15 station-keep).",
)
parser.add_argument("--slow-radius", type=float, default=5.0, help="Distance in metres at which thrust ramps down.")
parser.add_argument("--kp", type=float, default=2.0, help="Heading PID proportional gain (hand-set default).")
parser.add_argument("--ki", type=float, default=0.0, help="Heading PID integral gain (hand-set default).")
parser.add_argument("--kd", type=float, default=0.5, help="Heading PID derivative gain (hand-set default).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")
if args_cli.eval_steps <= 0:
    parser.error("--eval_steps must be positive")
if args_cli.dist_scale is not None and args_cli.dist_scale <= 0.0:
    parser.error("--dist-scale must be positive")
if args_cli.slow_radius <= 0.0:
    parser.error("--slow-radius must be positive")
if not all(math.isfinite(value) for value in (args_cli.kp, args_cli.ki, args_cli.kd)):
    parser.error("PID gains must be finite")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import (  # noqa: F401
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _base_env(env):
    """Dig through evaluation wrappers to the underlying Isaac environment."""
    e = env
    for _ in range(10):
        if any(hasattr(e, name) for name in ("reached_count", "episode_success", "path_length")):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _git_commit() -> str:
    """Resolve HEAD against the repository containing this script."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cfg_fingerprint(env_cfg) -> tuple[str, dict[str, object]]:
    """Hash protocol-relevant scalar config fields in a stable order."""
    names = {"episode_length_s", "hold_radius", "goal_radius"}
    names.update(name for name in dir(env_cfg) if "spawn" in name.lower() and "distance" in name.lower())

    scalars = {}
    for name in sorted(names):
        if name.startswith("_") or not hasattr(env_cfg, name):
            continue
        value = getattr(env_cfg, name)
        if isinstance(value, (bool, int, float, str)):
            scalars[name] = value

    encoded = json.dumps(scalars, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10], scalars


def _radius_from_env(base, env_cfg) -> float:
    """Read the fixed task-zone radius, preferring an instance override."""
    for owner in (base, getattr(base, "cfg", None), env_cfg):
        if owner is None:
            continue
        for name in ("hold_radius", "goal_radius"):
            value = getattr(owner, name, None)
            if value is not None:
                return float(value)
    raise RuntimeError("Task has neither hold_radius nor goal_radius; cannot define the speed controller zone")


def _control_step_s(env_cfg) -> float:
    return float(env_cfg.sim.dt) * int(getattr(env_cfg, "decimation", 1))


def _policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, Mapping):
        obs = obs.get("policy", obs)
    if isinstance(obs, Mapping):
        if len(obs) != 1:
            raise RuntimeError("Cannot select a single policy observation from the environment mapping")
        obs = next(iter(obs.values()))
    tensor = torch.as_tensor(obs)
    if tensor.ndim != 2 or tensor.shape[1] < 3:
        raise RuntimeError(f"Expected policy observations shaped (num_envs, >=3), got {tuple(tensor.shape)}")
    return tensor


class LOSPIDController:
    """Vectorized line-of-sight guidance and heading PID controller."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        dt: float,
        dist_scale: float,
        zone_radius: float,
        slow_radius: float,
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("Control timestep must be positive")
        if slow_radius <= zone_radius:
            raise ValueError(
                f"--slow-radius ({slow_radius:g}) must exceed the task zone radius ({zone_radius:g})"
            )
        self.dt = dt
        self.dist_scale = dist_scale
        self.zone_radius = zone_radius
        self.slow_radius = slow_radius
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = torch.zeros(num_envs, device=device, dtype=dtype)
        self.previous_error = torch.zeros_like(self.integral)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def act(self, obs) -> torch.Tensor:
        policy_obs = _policy_obs(obs)
        dot = policy_obs[:, 0]
        cross = policy_obs[:, 1]
        distance = policy_obs[:, 2].clamp_min(0.0) * self.dist_scale

        # atan2 returns the LOS heading error already wrapped to [-pi, pi].
        error = torch.atan2(cross, dot)
        inside_zone = distance <= self.zone_radius
        active = ~inside_zone
        self.integral.add_(torch.where(active, error * self.dt, torch.zeros_like(error)))
        derivative = torch.where(
            self.initialized & active,
            (error - self.previous_error) / self.dt,
            torch.zeros_like(error),
        )
        yaw = (self.kp * error + self.ki * self.integral + self.kd * derivative).clamp(-1.0, 1.0)
        self.previous_error.copy_(error)
        self.initialized.fill_(True)

        # Station keeping has no heading objective inside the hold zone. Clear
        # PID memory there so a later drift-out re-engages without derivative kick.
        yaw = torch.where(inside_zone, torch.zeros_like(yaw), yaw)
        self.integral[inside_zone] = 0.0
        self.previous_error[inside_zone] = 0.0
        self.initialized[inside_zone] = False

        speed_profile = ((distance - self.zone_radius) / (self.slow_radius - self.zone_radius)).clamp(0.0, 1.0)
        aligned = error.abs() < (math.pi / 4.0)
        thrust = torch.where(aligned, speed_profile, torch.full_like(speed_profile, 0.2))
        thrust = torch.where(inside_zone, torch.zeros_like(thrust), thrust)
        return torch.stack((thrust, yaw), dim=-1)

    def reset(self, done) -> None:
        mask = torch.as_tensor(done, device=self.integral.device).reshape(-1).bool()
        if mask.any():
            self.integral[mask] = 0.0
            self.previous_error[mask] = 0.0
            self.initialized[mask] = False


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
    name = "episode_path_length" if hasattr(base, "episode_path_length") else "path_length"
    return [float(value) for value in _values_at(base, name, env_ids, math.nan)]


def _append_csv(path: str, repro: dict[str, object], metrics: dict[str, object]) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    common_fields = [
        "task",
        "controller",
        "seed",
        "num_envs",
        "dist_scale",
        "slow_radius",
        "kp",
        "ki",
        "kd",
    ]
    if repro["protocol"] == "throughput":
        metric_fields = [
            "eval_steps",
            "targets_per_episode",
            "mean_speed",
            "oob_per_episode",
            "total_targets",
            "episode_equivalents",
        ]
    else:
        metric_fields = [
            "episodes",
            "sr",
            "spl",
            "mean_time_to_success_s",
            "median_time_to_success_s",
            "successes",
            "failures_timeout",
            "failures_other",
        ]
    fields = common_fields + metric_fields + ["cfg_sha1", "git_commit"]

    row = {**repro, **metrics}
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [f"# {key}={value}" if index == 0 else f"{key}={value}" for index, (key, value) in enumerate(repro.items())]
        )
        if new_file:
            writer.writerow(fields)
        writer.writerow([row[field] for field in fields])


def _run_throughput(env, base, controller: LOSPIDController, obs) -> dict[str, object]:
    max_ep = float(getattr(base, "max_episode_length", 0)) or float(args_cli.eval_steps)
    reached_total = 0
    reached_prev = int(getattr(base, "reached_count", 0))
    n_oob = 0
    speed_sum, speed_n = 0.0, 0

    print(f"[INFO] Evaluating {args_cli.eval_steps} steps x {args_cli.num_envs} envs (classical deterministic)...")
    for _ in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = controller.act(obs)
            obs, _, terminated, truncated, _ = env.step(actions)

        cur = int(getattr(base, "reached_count", 0))
        delta = cur - reached_prev
        if delta > 0:
            reached_total += delta
        reached_prev = cur

        try:
            vel = base.robot.data.root_com_vel_w
            speed_sum += torch.norm(vel[:, :2], dim=-1).mean().item()
            speed_n += 1
        except Exception:
            pass

        terminated = torch.as_tensor(terminated).reshape(-1)
        truncated = torch.as_tensor(truncated).reshape(-1)
        n_oob += int(terminated.sum().item())
        controller.reset(terminated.bool() | truncated.bool())

    ep_equiv = (args_cli.num_envs * args_cli.eval_steps) / max(max_ep, 1.0)
    return {
        "eval_steps": args_cli.eval_steps,
        "targets_per_episode": reached_total / max(ep_equiv, 1.0e-9),
        "mean_speed": speed_sum / max(speed_n, 1),
        "oob_per_episode": n_oob / max(ep_equiv, 1.0e-9),
        "total_targets": reached_total,
        "episode_equivalents": ep_equiv,
        "max_episode_length": max_ep,
    }


def _run_mission(env, base, controller: LOSPIDController, zone_radius: float, device, obs) -> dict[str, object]:
    if not hasattr(base, "_horizontal_distance"):
        print("[WARN] Mission env has no _horizontal_distance(); SPL will be nan.")

    initial_distance = _initial_distances(base, args_cli.num_envs, device)
    completed = 0
    successes = 0
    failures_timeout = 0
    failures_other = 0
    spl_values = []
    success_times = []

    print(
        f"[INFO] Evaluating {args_cli.episodes} completed episodes x "
        f"{args_cli.num_envs} envs (classical deterministic)..."
    )
    while completed < args_cli.episodes:
        with torch.inference_mode():
            actions = controller.act(obs)
            obs, _, terminated, truncated, _ = env.step(actions)

        terminated = torch.as_tensor(terminated, device=device).reshape(-1).bool()
        truncated = torch.as_tensor(truncated, device=device).reshape(-1).bool()
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        controller.reset(terminated | truncated)
        if done_ids.numel() == 0:
            continue

        remaining = args_cli.episodes - completed
        counted_ids = done_ids[:remaining]
        counted_terminated = terminated[counted_ids].detach().cpu().tolist()
        counted_truncated = truncated[counted_ids].detach().cpu().tolist()
        episode_success = [bool(value) for value in _values_at(base, "episode_success", counted_ids, False)]
        episode_paths = _completed_path_lengths(base, counted_ids)
        episode_times = [float(value) for value in _values_at(base, "time_to_success", counted_ids, math.nan)]
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

            if math.isfinite(d0) and math.isfinite(path_length):
                shortest_path = max(float(d0) - zone_radius, 0.0)
                denominator = max(float(path_length), shortest_path)
                spl = shortest_path / denominator if success and denominator > 0.0 else 0.0
                spl_values.append(spl)
            else:
                spl_values.append(math.nan)

        completed += int(counted_ids.numel())
        if completed >= args_cli.episodes:
            break

        next_distance = _initial_distances(base, args_cli.num_envs, device)
        initial_distance[done_ids] = next_distance[done_ids]

    return {
        "episodes": completed,
        "sr": successes / completed,
        "spl": sum(spl_values) / completed if all(math.isfinite(value) for value in spl_values) else math.nan,
        "mean_time_to_success_s": sum(success_times) / len(success_times) if success_times else math.nan,
        "median_time_to_success_s": statistics.median(success_times) if success_times else math.nan,
        "successes": successes,
        "failures_timeout": failures_timeout,
        "failures_other": failures_other,
    }


def _print_summary(metrics: dict[str, object]) -> None:
    print("\n" + "=" * 64)
    print(f"  USVBench classical {args_cli.protocol} eval - {args_cli.task}")
    print(f"  controller: LOS+PID  seed={args_cli.seed}")
    print("-" * 64)
    if args_cli.protocol == "throughput":
        print(f"  targets_per_episode : {metrics['targets_per_episode']:7.2f}   (primary score)")
        print(f"  mean_speed (m/s)    : {metrics['mean_speed']:7.2f}")
        print(f"  oob_per_episode     : {metrics['oob_per_episode']:7.2f}   (out-of-bounds events / episode)")
        print(f"  total_targets       : {metrics['total_targets']}")
        print(
            f"  episode_equivalents : {metrics['episode_equivalents']:7.1f}"
            f"   (ep_len={metrics['max_episode_length']:.0f} steps)"
        )
    else:
        completed = metrics["episodes"]
        print(f"  SR                         : {metrics['sr']:8.4f}  ({metrics['successes']}/{completed})")
        print(f"  SPL                        : {metrics['spl']:8.4f}")
        print(f"  mean time to success (s)   : {metrics['mean_time_to_success_s']:8.3f}")
        print(f"  median time to success (s) : {metrics['median_time_to_success_s']:8.3f}")
        print(f"  failures - timeout         : {metrics['failures_timeout']:8d}")
        print(f"  failures - other           : {metrics['failures_other']:8d}")
    print("=" * 64 + "\n")
    # Isaac/Kit shutdown can swallow buffered output on some machines.
    sys.stdout.flush()


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    cfg_sha1, cfg_scalars = _cfg_fingerprint(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    base = _base_env(env)
    device = torch.device(getattr(base, "device", env_cfg.sim.device))
    obs, _ = env.reset()
    obs_tensor = _policy_obs(obs)
    zone_radius = _radius_from_env(base, env_cfg)
    is_station_keep = hasattr(env_cfg, "hold_radius") or hasattr(getattr(base, "cfg", None), "hold_radius")
    dist_scale = args_cli.dist_scale if args_cli.dist_scale is not None else (15.0 if is_station_keep else 30.0)

    controller = LOSPIDController(
        num_envs=args_cli.num_envs,
        device=obs_tensor.device,
        dtype=obs_tensor.dtype,
        dt=_control_step_s(env_cfg),
        dist_scale=dist_scale,
        zone_radius=zone_radius,
        slow_radius=args_cli.slow_radius,
        kp=args_cli.kp,
        ki=args_cli.ki,
        kd=args_cli.kd,
    )

    repro = {
        "git_commit": _git_commit(),
        "task": args_cli.task,
        "controller": "los_pid",
        "protocol": args_cli.protocol,
        "num_envs": args_cli.num_envs,
        "episodes" if args_cli.protocol == "mission" else "eval_steps": (
            args_cli.episodes if args_cli.protocol == "mission" else args_cli.eval_steps
        ),
        "seed": args_cli.seed,
        "dist_scale": dist_scale,
        "slow_radius": args_cli.slow_radius,
        "kp": args_cli.kp,
        "ki": args_cli.ki,
        "kd": args_cli.kd,
        "cfg_sha1": cfg_sha1,
    }
    print("[REPRO] " + "  ".join(f"{key}={value}" for key, value in repro.items()))
    print("[REPRO] cfg_scalars=" + json.dumps(cfg_scalars, sort_keys=True, separators=(",", ":")))

    if args_cli.protocol == "throughput":
        metrics = _run_throughput(env, base, controller, obs)
    else:
        metrics = _run_mission(env, base, controller, zone_radius, device, obs)

    if args_cli.csv:
        _append_csv(args_cli.csv, repro, metrics)
        print(f"[INFO] appended result to {args_cli.csv}")

    _print_summary(metrics)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
