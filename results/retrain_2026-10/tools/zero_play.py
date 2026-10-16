"""Record an environment preview with zero actions (no policy). For README tasks without a baseline."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--video_length", type=int, default=1800)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
# render-only wide camera for scenes whose layout extends beyond the default view
if os.environ.get("USVB_EYE"):
    env_cfg.viewer.eye = tuple(float(v) for v in os.environ["USVB_EYE"].split(","))
    env_cfg.viewer.lookat = tuple(float(v) for v in os.environ.get("USVB_LOOKAT", "0,0,0").split(","))
if os.environ.get("USVB_STYLE"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_style
    _f = os.environ.get("USVB_FOLLOW")
    render_style.configure(args_cli.task, env_cfg, eye=[float(v) for v in _f.split(",")] if _f else None)
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
if os.environ.get("USVB_STYLE"):
    env = render_style.TrailWrapper(env, width=float(os.environ.get("USVB_TRAIL_W", "0.07")))
env = gym.wrappers.RecordVideo(env, video_folder=args_cli.out_dir, step_trigger=lambda s: s == 0,
                               video_length=args_cli.video_length, disable_logger=True)
env.reset()
act_dim = env.unwrapped.single_action_space.shape[0]
zeros = torch.zeros((args_cli.num_envs, act_dim), device=env.unwrapped.device)
for _ in range(args_cli.video_length):
    env.step(zeros)
env.close()
print("ZERO_PLAY_DONE")
simulation_app.close()
