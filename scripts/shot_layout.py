"""Capture one frozen-level Isaac task layout from the viewport camera."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--level", type=int, required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--warmup-frames", type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level

env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
base = env.unwrapped
env.reset()

origin = base.scene.env_origins[0]
ox, oy = float(origin[0]), float(origin[1])
base.sim.set_camera_view(
    eye=(ox, oy, 95.0),
    target=(ox, oy, 0.0),
)

render_product = rep.create.render_product("/OmniverseKit_Persp", (1280, 1280))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])
for _ in range(args_cli.warmup_frames):
    base.sim.render()

frame = None
for _ in range(4):
    base.sim.render()
    data = np.asarray(rgb_annotator.get_data())
    if data.size == 0:
        continue
    candidate = data[..., :3] if data.ndim == 3 and data.shape[2] == 4 else data
    if float(candidate.mean()) >= 2.0:
        frame = candidate
        break

if frame is None:
    env.close()
    app.close()
    raise RuntimeError("replicator returned no non-dark RGB frame after 4 renders")

out_path = os.path.abspath(args_cli.out)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(out_path, format="PNG")
print(
    f"layout shot -> {out_path} "
    f"(task={args_cli.task}, frozen_level={args_cli.level})",
    flush=True,
)
env.close()
app.close()
