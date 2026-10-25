"""Harbor stage-probability eval: aligned rounds, collects M1/M2/M3 latches.

Reads the milestone latches at the LAST pre-reset step of each aligned round
(fixed horizon -> all envs complete together), giving per-episode stage
outcomes without relying on extras batch means.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--eval-seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

TASK = "Isaac-USV-HarborMission-Direct-v1"
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env_cfg.seed = args_cli.eval_seed
experiment_cfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
env = gym.make(TASK, cfg=env_cfg)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False
runner = Runner(wrapped, experiment_cfg)
runner.agent.load(os.path.abspath(args_cli.checkpoint))
runner.agent.set_running_mode("eval")

base = env.unwrapped
horizon = int(base.max_episode_length)
rounds = max(1, (args_cli.episodes + 63) // 64)
m1_total = m2_total = m3_total = succ_total = n_total = 0

obs, _ = wrapped.reset()
for r in range(rounds):
    # Running OR over the whole round: the latches are monotone within an
    # episode and the auto-reset wipes them one step earlier than a single
    # end-sample expects (that off-by-one zeroed a whole first attempt).
    m1 = torch.zeros(64, dtype=torch.bool, device=base.device)
    m2 = torch.zeros_like(m1)
    m3 = torch.zeros_like(m1)
    for t in range(horizon):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, term, trunc, _ = wrapped.step(actions)
        m1 |= base._m1
        m2 |= base._m1 & base._m2
        m3 |= base._m1 & base._m2 & base._m3
    succ = base.episode_success.clone()
    m1_total += int(m1.sum()); m2_total += int(m2.sum())
    m3_total += int(m3.sum()); succ_total += int(succ.sum()); n_total += 64
    print(f"round {r}: M1={int(m1.sum())}/64 M2={int(m2.sum())}/64 "
          f"M3={int(m3.sum())}/64 success={int(succ.sum())}/64", flush=True)

p1 = m1_total / n_total
p21 = (m2_total / m1_total) if m1_total else float("nan")
p32 = (m3_total / m2_total) if m2_total else float("nan")
print(f"STAGEEVAL ckpt={os.path.basename(args_cli.checkpoint)} seed={args_cli.eval_seed} "
      f"n={n_total} P(M1)={p1:.4f} ({m1_total}/{n_total}) "
      f"P(M2|M1)={p21:.4f} ({m2_total}/{m1_total if m1_total else 0}) "
      f"P(M3|M2)={p32:.4f} success={succ_total}/{n_total}", flush=True)
env.close()
app.close()
