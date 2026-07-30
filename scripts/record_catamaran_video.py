# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E402  # Isaac Sim must launch before importing simulation modules
"""Record a deterministic playback video of a catamaran-patrol checkpoint.

The script is deliberately independent of the task code: waypoint markers,
lighting and the camera rig are added from the outside, so the exact same
command can be pointed at the original branch and at a fixed branch and the
resulting clips differ only in the task implementation.

Example:
    python scripts/record_catamaran_video.py \
        --task Isaac-Catamaran-Patrol-Direct-v1 \
        --checkpoint tasks/catamaran_patrol/checkpoints/catamaran_p1_s42.pt \
        --video_dir videos/before --tag before --cam chase --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a catamaran-patrol playback video.")
parser.add_argument("--task", type=str, default="Isaac-Catamaran-Patrol-Direct-v1")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint.")
parser.add_argument("--video_dir", type=str, required=True, help="Output folder for the mp4.")
parser.add_argument("--tag", type=str, default="catamaran", help="File-name prefix for the clip.")
parser.add_argument("--video_length", type=int, default=1800, help="Recorded length in policy steps (60 steps = 1 s).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=2026, help="Scenario seed; keep identical across compared runs.")
parser.add_argument("--cam", type=str, default="chase", choices=["chase", "top"], help="Camera rig.")
parser.add_argument("--cam_height", type=float, default=None, help="Override camera height (m).")
parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the policy mean.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# rendering is mandatory for video capture
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import imageio
import torch
from skrl.utils.runner.torch import Runner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = "skrl_cfg_entry_point"


def _load_checkpoint_compat(agent, checkpoint_path):
    """Load a checkpoint, tolerating the skrl observation/state preprocessor rename."""
    agent.load(checkpoint_path)
    preprocessor = None
    for attr in ("_observation_preprocessor", "_state_preprocessor", "observation_preprocessor", "state_preprocessor"):
        candidate = getattr(agent, attr, None)
        if candidate is not None and hasattr(candidate, "load_state_dict"):
            preprocessor = candidate
            break
    if preprocessor is None:
        return
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        return
    for key in ("observation_preprocessor", "state_preprocessor", "_state_preprocessor"):
        if key in raw and isinstance(raw[key], dict) and raw[key]:
            try:
                preprocessor.load_state_dict(raw[key])
                print(f"[INFO] Loaded observation scaler from checkpoint key '{key}'")
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                print(f"[WARN] Could not load scaler from '{key}': {exc}")
            return


def _marker_cfg(goal_radius: float) -> VisualizationMarkersCfg:
    """Flat discs of exactly goal_radius: the active waypoint plus the pending ones.

    Discs rather than spheres, so the success region stays readable from both the
    chase and the top-down camera and never hides the vessel.
    """
    return VisualizationMarkersCfg(
        prim_path="/Visuals/patrol_waypoints",
        markers={
            "active": sim_utils.CylinderCfg(
                radius=goal_radius,
                height=0.25,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.35, 0.10), emissive_color=(0.60, 0.18, 0.02), opacity=0.65
                ),
            ),
            "pending": sim_utils.CylinderCfg(
                radius=goal_radius,
                height=0.12,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.45, 0.95), emissive_color=(0.04, 0.12, 0.35), opacity=0.45
                ),
            ),
        },
    )


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, experiment_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed

    # Camera rigs, both tracking the hull: "chase" for how the vessel moves, "top" high
    # enough that the surrounding waypoints stay in frame next to it.
    height = args_cli.cam_height if args_cli.cam_height is not None else (14.0 if args_cli.cam == "chase" else 90.0)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    if args_cli.cam == "chase":
        env_cfg.viewer.eye = (-22.0, -22.0, height)
    else:
        env_cfg.viewer.eye = (0.0, -0.01, height)

    checkpoint = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    env_cfg.log_dir = os.path.dirname(os.path.dirname(checkpoint))

    video_dir = os.path.abspath(args_cli.video_dir)
    os.makedirs(video_dir, exist_ok=True)

    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["agent"]["experiment"]["wandb"] = False
    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    _load_checkpoint_compat(runner.agent, checkpoint)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")

    # the unwrapped task env — the skrl wrapper forwards attribute reads but its own
    # render() is a no-op, so frame grabbing has to go through the task env directly
    base = gym_env.unwrapped

    # Lighting and a visual water sheet are added from here, so that a task without
    # its own dome light still renders and the vessel has a horizon to move against.
    try:
        light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.55, 0.70, 0.88))
        light_cfg.func("/World/RecordLight", light_cfg)
        water_cfg = sim_utils.CuboidCfg(
            size=(600.0, 600.0, 0.10),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.20, 0.38), roughness=0.25, metallic=0.05
            ),
        )
        # the hull floats at z ~= -0.4 (buoyancy equilibrium), so the water sheet sits
        # slightly lower — otherwise the vessel is hidden under it
        water_cfg.func("/World/RecordWater", water_cfg, translation=(0.0, 0.0, -0.60))
    except Exception as exc:  # noqa: BLE001 - purely cosmetic, never fail the recording
        print(f"[INFO] Skipped record light/water: {exc}")

    goal_radius = float(getattr(base.cfg, "goal_radius", 3.0))
    markers = VisualizationMarkers(_marker_cfg(goal_radius))
    heading_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/vessel_heading",
            markers={
                "bow": sim_utils.ConeCfg(
                    radius=0.7,
                    height=2.4,
                    axis="X",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.85, 0.10), emissive_color=(0.55, 0.45, 0.02)
                    ),
                )
            },
        )
    )
    heading_offset = torch.tensor([[0.0, 0.0, 1.6]], device=base.device)

    observations, _ = env.reset()

    n_wp = base.waypoint_pos.shape[1]
    reached_start = int(getattr(base, "reached_count", 0))
    min_dist = float("inf")

    # Frames are grabbed here instead of through gym.wrappers.RecordVideo: the RTX
    # viewport hands back an empty buffer on every second render call, which makes
    # the wrapper produce a clip where half the frames are black.
    fps = int(round(float(base.metadata.get("render_fps", 60))))
    video_path = os.path.join(video_dir, f"{args_cli.tag}.mp4")
    writer = imageio.get_writer(video_path, fps=fps, quality=8, macro_block_size=None)
    blank_retries = 0

    def _grab_frame():
        nonlocal blank_retries
        for _ in range(3):
            frame = base.render()
            if frame is not None and frame.any():
                return frame
            blank_retries += 1
        return frame

    for step in range(args_cli.video_length):
        with torch.inference_mode():
            outputs = runner.agent.act(observations, timestep=0, timesteps=0)
            actions = outputs[0]
            info = outputs[-1] if isinstance(outputs[-1], dict) else {}
            if not args_cli.stochastic:
                actions = info.get("mean_actions", actions)
            observations, _, _, _, _ = env.step(actions)

            # marker update: waypoints of env 0, active one highlighted
            wps = base.waypoint_pos[0].clone()
            wps[:, 2] = -0.35  # just above the water sheet so the discs stay visible
            indices = torch.ones(n_wp, dtype=torch.long, device=wps.device)
            indices[int(base.wp_idx[0].item())] = 0
            markers.visualize(translations=wps, marker_indices=indices.tolist())

            # yellow cone above the hull, aligned with body +X — makes heading readable
            heading_marker.visualize(
                translations=base.robot.data.root_pos_w[0:1] + heading_offset,
                orientations=base.robot.data.root_quat_w[0:1],
            )

            pos = base.robot.data.root_pos_w[0, :2]
            target = base.waypoint_pos[0, int(base.wp_idx[0].item()), :2]
            min_dist = min(min_dist, float(torch.norm(pos - target).item()))

        writer.append_data(_grab_frame())

        if step % 300 == 0:
            print(f"[REC] step {step:5d}  reached={int(base.reached_count) - reached_start}  min_dist={min_dist:6.3f}")

    writer.close()

    print("\n" + "=" * 64)
    print(f"  clip              : {args_cli.tag} ({args_cli.cam} camera)")
    print(f"  task              : {args_cli.task}")
    print(f"  checkpoint        : {os.path.basename(checkpoint)}")
    print(f"  steps recorded    : {args_cli.video_length} (~{args_cli.video_length / fps:.0f} s at {fps} fps)")
    print(f"  waypoints reached : {int(base.reached_count) - reached_start}")
    print(f"  closest approach  : {min_dist:.3f} m  (success radius {goal_radius:.1f} m)")
    print(f"  blank-frame retries: {blank_retries}")
    print(f"  output            : {video_path}")
    print("=" * 64 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
