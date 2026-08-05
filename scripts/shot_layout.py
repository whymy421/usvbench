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
# Boot EXACTLY like demos/render_probe.py, which renders fine on the same
# dual-GPU machine where the parse_known_args + sys.argv-strip variant dies
# at AppLauncher with "Vulkan: Flags 0x6 must be the same for both device
# and instance". Kit reads sys.argv during boot; do not strip it.
args_cli = parser.parse_args()
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

# No render_mode: rgb_array makes DirectRLEnv build its own viewport render
# product, and on the dual-GPU 4080 that internal engine lands on the AMD
# iGPU and Hydra dies ("failed creating scene renderer", deviceMask 1) --
# every probe that skips env-owned rendering and captures via an explicit
# replicator product on /OmniverseKit_Persp works on the same machine.
env = gym.make(args_cli.task, cfg=env_cfg)
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
# The env's viewer controller re-applies its own (robot-follow, close-up)
# eye during rendering -- the same fight the demo recorder hit. Keep
# re-asserting the top-down framing through the warmup, and once more
# right before capture.
for i in range(args_cli.warmup_frames):
    if i % 10 == 0:
        base.sim.set_camera_view(eye=(ox, oy, 95.0), target=(ox, oy, 0.0))
    base.sim.render()
base.sim.set_camera_view(eye=(ox, oy, 95.0), target=(ox, oy, 0.0))

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
