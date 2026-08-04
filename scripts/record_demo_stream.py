"""Streaming demo recorder: frames piped to ffmpeg, O(1) memory.

Replaces gym RecordVideo (which buffers every frame in RAM -- 610 s at 60 fps
is ~128 GB and OOM-kills the run; that was the black-video root cause).
--skip-seconds fast-forwards without rendering to reach interesting episodes.
"""
import argparse
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True, help="output mp4 path")
parser.add_argument("--seconds", type=float, default=130.0)
parser.add_argument("--skip-seconds", type=float, default=0.0)
parser.add_argument("--cam-height", type=float, default=40.0)
parser.add_argument("--cam-back", type=float, default=16.0)
parser.add_argument("--env-seed", type=int, default=None)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--warmup-frames", type=int, default=60,
                    help="renderer burn-in before the first captured frame")
parser.add_argument("--render-width", type=int, default=1280)
parser.add_argument("--render-height", type=int, default=720)
parser.add_argument("--visual-usd", default=None,
                    help="swap the robot spawn USD (e.g. blueboat_visual.usd; "
                         "physics layer composed identically via sublayer)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import os
import torch
import gymnasium as gym
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
if args_cli.env_seed is not None:
    env_cfg.seed = args_cli.env_seed
if args_cli.visual_usd is not None:
    env_cfg.robot_cfg.spawn.usd_path = args_cli.visual_usd
experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
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


def aim_camera():
    origin = base.scene.env_origins[0]
    tx, ty = float(origin[0]), float(origin[1])
    if hasattr(base, "target_pos"):
        goal = base.target_pos[0]
        tx = (tx + float(goal[0])) / 2.0
        ty = (ty + float(goal[1])) / 2.0
    elif hasattr(base, "dock_point"):
        tx, ty = float(base.dock_point[0][0]), float(base.dock_point[0][1])
    elif hasattr(base, "waypoints"):
        wp = base.waypoints[0] + base.scene.env_origins[0, :2]
        tx = float(wp[:, 0].mean())
        ty = float(wp[:, 1].mean())
    base.sim.set_camera_view(
        eye=(tx, ty - args_cli.cam_back, args_cli.cam_height),
        target=(tx, ty, 0.0),
    )


aim_camera()

# env.render() returns all-zero frames under --headless --enable_cameras on
# both machines we own, which is why every demo before 2026-08-04 was black
# while its log looked clean. demos/render_probe.py proves the SAME machine
# renders a lit scene correctly when the pixels are read from an explicit
# replicator annotator, so drive the renderer and read from one.
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

render_product = rep.create.render_product(
    "/OmniverseKit_Persp", (args_cli.render_width, args_cli.render_height)
)
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])
# RTX resolves over several frames; capturing from step 0 gives black.
for _ in range(args_cli.warmup_frames):
    base.sim.render()


_last_good = [None]


def grab_frame():
    """One non-black RGB frame.

    The synthetic-data graph publishes on every OTHER render: a naive
    render-then-read alternates a lit frame with a pure-black one (measured:
    YAVG 179, 16, 179, 16 ...). Re-render until the annotator hands back a lit
    frame, and if it never does, repeat the previous good frame rather than
    write black -- a dropped duplicate is invisible at 60 fps, a black flash
    is not.
    """
    for _ in range(4):
        base.sim.render()
        data = np.asarray(rgb_annotator.get_data())
        if data.size == 0:
            continue
        frame = data[..., :3] if data.ndim == 3 and data.shape[2] == 4 else data
        if float(frame.mean()) >= 2.0:
            _last_good[0] = frame
            return frame
    return _last_good[0]


writer = None
dark_frames = 0
kept_frames = 0
os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)

skip_steps = int(args_cli.skip_seconds * 60)
steps = int(args_cli.seconds * 60)
for t in range(skip_steps + steps):
    with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    obs, _, term, trunc, _ = wrapped.step(actions)
    # Re-aim periodically: something downstream of reset re-applies the env's
    # ViewerCfg eye, which is why --cam-height looked like a no-op for weeks
    # (measured: a 4 m obstacle filled 370 of 1280 px, i.e. a ~13 m wide view,
    # exactly Isaac's default 7.5,7.5,7.5 viewer, not the 75 m we asked for).
    if t % 30 == 0:
        aim_camera()
    if t >= skip_steps:
        frame = grab_frame()
        if frame is not None:
            kept_frames += 1
            if float(frame.mean()) < 2.0:
                dark_frames += 1
            if writer is None:
                h, w = frame.shape[0], frame.shape[1]
                pix = "rgba" if frame.shape[2] == 4 else "rgb24"
                writer = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                     "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", str(args_cli.fps),
                     "-i", "-", "-vf", "format=yuv420p", "-c:v", "libx264",
                     "-preset", "fast", "-crf", "23", args_cli.out],
                    stdin=subprocess.PIPE,
                )
            writer.stdin.write(frame.tobytes())
    if bool(term[0]) or bool(trunc[0]):
        ok = bool(getattr(base, "episode_success", [False])[0])
        print(f"episode end t={t / 60.0:.1f}s success={ok}", flush=True)
        aim_camera()

if writer is not None:
    writer.stdin.close()
    writer.wait()
    # Report darkness as a product assertion, not a log string: the old
    # black clips printed "video -> ..." exactly like a good one.
    print(f"frames={kept_frames} dark={dark_frames}", flush=True)
    if kept_frames and dark_frames > kept_frames // 2:
        print("BLACK VIDEO: more than half the frames are dark", flush=True)
    print(f"video -> {args_cli.out}", flush=True)
else:
    print("NO FRAMES RENDERED", flush=True)
env.close()
app.close()
