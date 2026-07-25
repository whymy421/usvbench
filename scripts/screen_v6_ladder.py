"""Single-process checkpoint ladder screening: one Isaac startup, N checkpoints.

Each checkpoint gets one fixed-horizon round (64 envs = 64 episodes) at the
frozen eval level. NOTE (audit finding): the layout RNG stream advances across
checkpoints within one process, so successive checkpoints see DIFFERENT layout
batches -- this is noisy SCREENING, not a paired comparison. Champions must
get fresh-process 128-episode certification evals (same eval seed = same
layout stream) before any paired claim.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--task", default="Isaac-USV-HazardNav-Direct-v1")
parser.add_argument("--every", type=int, default=8, help="screen every Nth checkpoint")
parser.add_argument("--only", default=None,
                    help="comma-separated step numbers; overrides --every")
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import os
import re

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

ckpt_dir = os.path.join(args_cli.run_dir, "checkpoints")
cks = sorted(
    (f for f in os.listdir(ckpt_dir) if re.fullmatch(r"agent_\d+\.pt", f)),
    key=lambda f: int(re.findall(r"\d+", f)[0]),
)
if args_cli.only:
    wanted = {int(s) for s in args_cli.only.split(",")}
    ladder = [c for c in cks if int(re.findall(r"\d+", c)[0]) in wanted]
else:
    ladder = cks[args_cli.every - 1 :: args_cli.every]
    if cks and cks[-1] not in ladder:
        ladder.append(cks[-1])
print(f"screening {len(ladder)}/{len(cks)} checkpoints: {ladder}")

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
base = env.unwrapped

results = []
for ck in ladder:
    runner.agent.load(os.path.join(ckpt_dir, ck))
    runner.agent.set_running_mode("eval")
    obs, _ = wrapped.reset()
    horizon = int(base.max_episode_length)
    for _ in range(horizon):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, term, trunc, _ = wrapped.step(actions)
    succ = base.episode_success.clone()
    tts = base.time_to_success.clone()
    ok = tts[~torch.isnan(tts)]
    med = float(ok.median()) if len(ok) else float("nan")
    sr = float(succ.float().mean())
    results.append((ck, sr, med))
    print(f"LADDER {ck}: SR={sr:.4f} ({int(succ.sum())}/64) median_tts={med:.1f}s",
          flush=True)

results.sort(key=lambda r: (-r[1], r[2] if r[2] == r[2] else 1e9))
print("TOP3: " + " | ".join(f"{c} SR={s:.3f} tts={m:.1f}" for c, s, m in results[:3]))
sys.stdout.flush()
env.close()
app.close()
