"""Zero-training failure autopsy: WHY do episodes fail, not just how often.

Task A certifies at 39% with only 4-6% collision episodes, so ~55% of episodes
fail without ever touching anything. That single number cannot distinguish
"the policy is deterred and never commits to a gap" from "the policy commits
and cannot control the hull through it" -- and the two call for opposite fixes.
This script runs an existing champion unchanged and classifies every episode
from per-step telemetry, so the next reward change is chosen from evidence.

Taxonomy (mutually exclusive, evaluated in order):
  collision          contacted anything
  stalled            barely moved (path < 3 hull lengths)
  never_engaged      never came within the proximity band of any obstacle
  retreated          entered the band, then backed out and stayed out
  stuck_in_field     spent the episode inside the field without getting through
  past_field_no_goal got clear of obstacles but never reached the goal ring
  at_rim_no_entry    came within (goal_radius + 1 hull length) but never latched

Usage:
    python scripts/diagnose_failures.py --checkpoint <ckpt> --episodes 128
"""

import argparse
import json
import sys

# The taxonomy labels are Chinese. Redirected to a file on Windows, Python
# picks cp1252 and the final print raises UnicodeEncodeError -- which threw
# away 447 s of completed GPU work, because the JSON was written after the
# printing. Fix both halves: make stdout encoding-proof here, and dump the
# records BEFORE printing them (see the end of this file).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-USV-HazardNav-Direct-v1")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

BAND_M = 1.35          # proximity band cap, the env's "engaged with obstacles" line
HULL_LEN_M = 1.2
STALL_PATH_M = 3.0 * HULL_LEN_M

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
n_envs = base.num_envs
dev = base.device
goal_radius = float(getattr(base.cfg, "goal_radius", 2.0))

# Per-episode accumulators, reset when an episode ends.
acc_min_goal = torch.full((n_envs,), 1e9, device=dev)   # closest approach to goal
acc_engaged = torch.zeros(n_envs, device=dev)           # steps inside the band
acc_engaged_last = torch.full((n_envs,), -1.0, device=dev)  # last step engaged
acc_min_clear = torch.full((n_envs,), 1e9, device=dev)
acc_fwd = torch.zeros(n_envs, device=dev)               # steps with positive surge cmd
acc_steps = torch.zeros(n_envs, device=dev)
acc_speed = torch.zeros(n_envs, device=dev)


def reset_acc(ids):
    acc_min_goal[ids] = 1e9
    acc_engaged[ids] = 0.0
    acc_engaged_last[ids] = -1.0
    acc_min_clear[ids] = 1e9
    acc_fwd[ids] = 0.0
    acc_steps[ids] = 0.0
    acc_speed[ids] = 0.0


obs, _ = wrapped.reset()
reset_acc(torch.arange(n_envs, device=dev))
records = []
max_steps = (args_cli.episodes // n_envs + 3) * base.max_episode_length
step = 0

while len(records) < args_cli.episodes and step < max_steps:
    with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    obs, _, term, trunc, _ = wrapped.step(actions)
    step += 1

    with torch.inference_mode():
        # distance to goal, clearance, speed, commanded surge -- all from the env
        goal_d = base._horizontal_distance()
        clear = base._clearance()
        speed = torch.norm(base.robot.data.root_com_vel_w[:, :2], dim=-1)
        surge_cmd = actions[:, 0] if actions.dim() > 1 else actions

        acc_min_goal.copy_(torch.minimum(acc_min_goal, goal_d))
        acc_min_clear.copy_(torch.minimum(acc_min_clear, clear))
        engaged = clear < (BAND_M - 0.45)  # clearance is surface-to-surface
        acc_engaged += engaged.float()
        acc_engaged_last.copy_(
            torch.where(engaged, acc_steps.clone(), acc_engaged_last)
        )
        acc_fwd += (surge_cmd > 0.0).float()
        acc_speed += speed
        acc_steps += 1.0

    done = term | trunc
    done = done.squeeze(-1) if done.dim() > 1 else done
    ids = torch.nonzero(done).flatten()
    for i in ids.tolist():
        steps_i = max(float(acc_steps[i]), 1.0)
        rec = {
            "success": bool(base.episode_success[i]),
            "min_clearance_m": float(base.episode_min_clearance[i]),
            "path_length_m": float(base.episode_path_length[i]),
            "closest_to_goal_m": float(acc_min_goal[i]),
            "engaged_steps": int(acc_engaged[i]),
            "engaged_fraction": float(acc_engaged[i]) / steps_i,
            "last_engaged_frac": (
                float(acc_engaged_last[i]) / steps_i
                if float(acc_engaged_last[i]) >= 0 else None
            ),
            "fwd_thrust_fraction": float(acc_fwd[i]) / steps_i,
            "mean_speed_mps": float(acc_speed[i]) / steps_i,
            "episode_steps": int(steps_i),
        }
        records.append(rec)
    if len(ids):
        reset_acc(ids)

records = records[: args_cli.episodes]


def classify(r):
    if r["success"]:
        return "success"
    if r["min_clearance_m"] < 0.0:
        return "collision"
    if r["path_length_m"] < STALL_PATH_M:
        return "stalled"
    if r["closest_to_goal_m"] <= goal_radius + HULL_LEN_M:
        return "at_rim_no_entry"
    if r["engaged_steps"] == 0:
        return "never_engaged"
    # engaged at some point; did it stay engaged to the end, or back out?
    if r["last_engaged_frac"] is not None and r["last_engaged_frac"] < 0.5:
        return "retreated"
    return "stuck_in_field"


for r in records:
    r["verdict"] = classify(r)

n = len(records)
order = ["success", "collision", "at_rim_no_entry", "never_engaged", "retreated",
         "stuck_in_field", "stalled"]
counts = {k: sum(1 for r in records if r["verdict"] == k) for k in order}

# Persist BEFORE reporting. The rollout is the expensive part; nothing about
# formatting it should be able to destroy it.
if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "level": args_cli.level,
                   "seed": args_cli.eval_seed,
                   "checkpoint": os.path.abspath(args_cli.checkpoint),
                   "counts": counts, "records": records}, f, indent=1)

print(f"DIAGNOSE task={TASK} level={args_cli.level} seed={args_cli.eval_seed} "
      f"ckpt={os.path.basename(args_cli.checkpoint)}")
print(f"  episodes={n}")
label = {
    "success": "成功",
    "collision": "碰撞",
    "at_rim_no_entry": "到了圈边不进去",
    "never_engaged": "从未靠近障碍(不敢进/绕远)",
    "retreated": "接近过障碍后退出并再没回去",
    "stuck_in_field": "困在障碍带里出不来",
    "stalled": "几乎没动",
}
for k in order:
    c = counts[k]
    if c:
        print(f"  {label[k]:<28} {c:>4}/{n} = {c / n:>6.1%}")

fails = [r for r in records if not r["success"]]
if fails:
    eng = [r["engaged_fraction"] for r in fails]
    spd = [r["mean_speed_mps"] for r in fails]
    fwd = [r["fwd_thrust_fraction"] for r in fails]
    succ = [r for r in records if r["success"]]
    print()
    print("  失败局 vs 成功局的行为对比(判断'不敢进'还是'进去控不住'):")
    print(f"    在障碍带内的时间占比: 失败 {sum(eng)/len(eng):.3f}"
          + (f" | 成功 {sum(r['engaged_fraction'] for r in succ)/len(succ):.3f}"
             if succ else ""))
    print(f"    平均速度 m/s:         失败 {sum(spd)/len(spd):.2f}"
          + (f" | 成功 {sum(r['mean_speed_mps'] for r in succ)/len(succ):.2f}"
             if succ else ""))
    print(f"    正推力时间占比:       失败 {sum(fwd)/len(fwd):.3f}"
          + (f" | 成功 {sum(r['fwd_thrust_fraction'] for r in succ)/len(succ):.3f}"
             if succ else ""))

if args_cli.out:
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
