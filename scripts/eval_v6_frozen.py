"""Frozen-level eval with per-episode records for paired statistics.

Layouts come from the env's layout RNG (seeded by cfg seed) and reset timing
is fixed-horizon, so two policies evaluated with the same --eval-seed see the
IDENTICAL episode/layout stream -> per-episode records support McNemar /
paired-bootstrap comparisons offline.
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-USV-HazardNav-Direct-v1")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument("--out", default=None, help="JSON per-episode records")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env_cfg.seed = args_cli.eval_seed
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
experiment_cfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False
runner = Runner(wrapped, experiment_cfg)
runner.agent.load(os.path.abspath(args_cli.checkpoint))
runner.agent.set_running_mode("eval")

base = env.unwrapped
obs, _ = wrapped.reset()
records = []
ep_counter = torch.zeros(base.num_envs, dtype=torch.long)
max_steps = (args_cli.episodes // base.num_envs + 3) * base.max_episode_length
step = 0
while len(records) < args_cli.episodes and step < max_steps:
    with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    # d0_per_env is rewritten inside _reset_idx, which runs during step(), so
    # reading it alongside the other episode stats would report the NEXT
    # episode's straight-line distance. Snapshot it while it still belongs to
    # the episode that is about to end.
    d0_prev = (base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None)
    obs, _, term, trunc, _ = wrapped.step(actions)
    step += 1
    done = term | trunc
    done = done.squeeze(-1) if done.dim() > 1 else done
    ids = torch.nonzero(done).flatten()
    for i in ids.tolist():
        episode_contact_steps = getattr(base, "episode_contact_steps", None)
        episode_contact_longest_steps = getattr(
            base, "episode_contact_longest_steps", None
        )
        episode_contact_depth_sum = getattr(
            base, "episode_contact_depth_sum", None
        )
        episode_max_phase = getattr(base, "episode_max_phase", None)
        rec = {
            "env": i,
            "ep": int(ep_counter[i]),
            "success": bool(base.episode_success[i]),
            "tts_s": (None if math.isnan(float(base.time_to_success[i]))
                      else float(base.time_to_success[i])),
            "min_clearance_m": float(base.episode_min_clearance[i]),
            "path_length_m": float(base.episode_path_length[i]),
        }
        if (
            episode_contact_steps is not None
            and episode_contact_longest_steps is not None
            and episode_contact_depth_sum is not None
        ):
            contact_steps = float(episode_contact_steps[i])
            rec["contact_steps"] = int(contact_steps)
            rec["contact_seconds"] = contact_steps * base.control_step_s
            rec["contact_longest_seconds"] = (
                float(episode_contact_longest_steps[i]) * base.control_step_s
            )
            rec["contact_depth_mean_m"] = (
                float(episode_contact_depth_sum[i]) / contact_steps
                if contact_steps > 0.0
                else 0.0
            )
        if episode_max_phase is not None:
            rec["max_phase"] = int(episode_max_phase[i])
        if hasattr(base, "episode_gates_passed"):
            rec["gates"] = int(base.episode_gates_passed[i])
        if d0_prev is not None:
            rec["d0_m"] = float(d0_prev[i])
        records.append(rec)
        ep_counter[i] += 1

records = records[: args_cli.episodes]
n = len(records)
succ = [r for r in records if r["success"]]
tts = sorted(r["tts_s"] for r in succ if r["tts_s"] is not None)
collided = sum(1 for r in records if r["min_clearance_m"] < 0.0)
clr = sorted(r["min_clearance_m"] for r in records)


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[k]


sr = len(succ) / max(n, 1)
print(f"EVAL task={TASK} level={args_cli.level} seed={args_cli.eval_seed} "
      f"ckpt={os.path.basename(args_cli.checkpoint)}")
print(f"  episodes={n} SR={sr:.4f} ({len(succ)}/{n})")
print(f"  tts median={pct(tts, 0.5):.1f}s p90={pct(tts, 0.9):.1f}s" if tts
      else "  tts: no successes")
print(f"  collision_episodes={collided}/{n} ({collided / max(n, 1):.3f}) "
      f"min_clearance p10={pct(clr, 0.10):.2f}m")
if getattr(base, "episode_max_phase", None) is not None:
    phase_distribution = {
        phase: sum(r.get("max_phase") == phase for r in records)
        for phase in range(4)
    }
    print(f"  stages: reached_phase distribution {phase_distribution}")
if records and all("contact_steps" in r for r in records):
    contact_records = [r for r in records if r["contact_steps"] > 0]
    contact_seconds = sorted(r["contact_seconds"] for r in contact_records)
    contact_longest_seconds = sorted(
        r["contact_longest_seconds"] for r in contact_records
    )
    total_contact_steps = sum(r["contact_steps"] for r in contact_records)
    depth_mean = (
        sum(
            r["contact_depth_mean_m"] * r["contact_steps"]
            for r in contact_records
        ) / total_contact_steps
        if total_contact_steps > 0
        else float("nan")
    )
    print(
        f"  contact: episodes_with_contact={len(contact_records)}/{n} "
        f"total_s median={pct(contact_seconds, 0.5):.2f} "
        f"p90={pct(contact_seconds, 0.9):.2f}  "
        f"longest_s median={pct(contact_longest_seconds, 0.5):.2f}  "
        f"depth_mean={depth_mean:.3f} m"
    )
# Success rate alone has no discriminative power on a task where detouring
# works: the shifted-potential run certified 100%/100% on Task A while walking
# a 68.6 m median path against a 29.7 m straight line -- a BIGGER detour than
# the baseline policy the validity audit had already flagged as going around.
# Only the path length made that visible, so certification prints it.
path = sorted(r["path_length_m"] for r in succ) or sorted(
    r["path_length_m"] for r in records)
print(f"  path_len median={pct(path, 0.5):.1f}m p90={pct(path, 0.9):.1f}m", end="")
ratios = sorted(r["path_length_m"] / r["d0_m"] for r in succ
                if r.get("d0_m", 0.0) > 0.0)
print(f"  detour median={pct(ratios, 0.5):.2f}x straight" if ratios else "")
if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "level": args_cli.level,
                   "seed": args_cli.eval_seed,
                   "checkpoint": os.path.abspath(args_cli.checkpoint),
                   "records": records}, f, indent=1)
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
