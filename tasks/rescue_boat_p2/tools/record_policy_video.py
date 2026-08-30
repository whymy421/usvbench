"""
record_policy_video.py — Record presentation-quality video of a trained policy.

The training videos are unusable for the poster or oral exam: they render no water
surface and no casualty markers, so they show a dark wireframe grid and a distant
speck. This script turns on the visualisation added to rescue_boat_env.py and
frames the camera on a single boat.

What you see:
  - translucent water surface at z = 0 (visual prim only, no collider)
  - casualty markers coloured by urgency:
        green  -> under 50% of timer elapsed
        amber  -> over 50%
        red    -> over 80%
        grey   -> lost
  - the boat, followed by a chase camera

The urgency colouring is the point: it makes the deadline-aware prioritisation
legible on screen. Without it the task looks like undifferentiated waypoint
chasing.

Usage:
  python record_policy_video.py --checkpoint="<path to best_agent.pt>" --num_envs=4

Output goes to videos/ next to the checkpoint's run folder.
"""

import argparse
import os
import sys

# Must be set before rescue_boat_env is imported — it reads this at module scope.
os.environ["VISUALISE"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record video of a trained policy.")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Keep small so the camera frames one boat rather than a 200 m grid.")
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--video_length", type=int, default=1800,
                    help="Frames to record. 1800 at 60 fps = 30 s of simulated time.")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--cam_dist", type=float, default=45.0, help="Chase camera distance (m).")
parser.add_argument("--cam_height", type=float, default=28.0, help="Chase camera height (m).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Cameras must be enabled for rgb_array rendering.
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = "skrl_cfg_entry_point"


def _base_env(env):
    e = env
    for _ in range(10):
        if hasattr(e, "reached_count"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _get_obs_prep(agent):
    for attr in ("_obs_preprocessor", "_observation_preprocessor", "observation_preprocessor"):
        c = getattr(agent, attr, None)
        if c is not None and callable(c):
            return c
    return None


def _restore_preprocessor(agent, ckpt):
    """Restore the normaliser explicitly — agent.load() does not do it reliably."""
    prep = _get_obs_prep(agent)
    if prep is None or not hasattr(prep, "load_state_dict"):
        return "none"
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"unreadable: {e}"
    for k in ("observation_preprocessor", "_observation_preprocessor",
              "state_preprocessor", "_state_preprocessor"):
        if k in raw:
            try:
                prep.load_state_dict(raw[k])
                return f"restored '{k}'"
            except Exception as e:
                return f"'{k}' failed: {e}"
    return "no preprocessor key"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    run_dir = os.path.dirname(os.path.dirname(ckpt))
    out_dir = args_cli.out_dir or os.path.join(run_dir, "videos", "presentation")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None

    # Chase camera locked to the boat in env 0, rather than a fixed world view.
    try:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        d, h = args_cli.cam_dist, args_cli.cam_height
        env_cfg.viewer.eye = (-d, -d * 0.6, h)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.resolution = (1920, 1080)
    except Exception as e:
        print(f"[WARN] could not configure viewer: {e}")

    print()
    print("=" * 70)
    print("  RECORDING POLICY VIDEO")
    print("=" * 70)
    print(f"  checkpoint : {os.path.basename(ckpt)}")
    print(f"  envs       : {args_cli.num_envs}")
    print(f"  frames     : {args_cli.video_length}")
    print(f"  output     : {out_dir}")
    print(f"  VISUALISE  : {os.environ.get('VISUALISE')}  (water + casualty markers)")
    print("=" * 70)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=out_dir,
        step_trigger=lambda step: step == 0,   # one clip, from the start
        video_length=args_cli.video_length,
        disable_logger=True,
        name_prefix="rescue_policy",
    )
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)
    agent = runner.agent
    agent.load(ckpt)
    print(f"  preprocessor: {_restore_preprocessor(agent, ckpt)}")
    for m in agent.models.values():
        if hasattr(m, "eval"):
            m.eval()

    obs_prep = _get_obs_prep(agent)
    base = _base_env(env)
    max_ep = int(getattr(base, "max_episode_length", 7200))
    n_cas = int(getattr(base.cfg, "n_casualties", 4))

    obs, _ = env.reset()
    rescues = 0
    prev = int(getattr(base, "reached_count", 0))
    steps = min(args_cli.video_length + 60, max_ep * args_cli.episodes)

    print()
    for i in range(steps):
        with torch.inference_mode():
            o = obs["policy"] if isinstance(obs, dict) else obs
            oi = obs_prep(o, train=False) if obs_prep is not None else o
            _a, _inf = agent.policy.act({"observations": oi}, role="policy")
            obs, _, _, _, _ = env.step(_inf.get("mean_actions", _a))

        cur = int(getattr(base, "reached_count", 0))
        if cur > prev:
            rescues += cur - prev
        prev = cur

        if (i + 1) % 600 == 0:
            print(f"  frame {i + 1}/{steps}   rescues so far: {rescues}")

    print()
    print("-" * 70)
    print(f"  {rescues} rescues across {args_cli.num_envs} boats "
          f"({args_cli.num_envs * n_cas} casualties)")
    print(f"  Video: {out_dir}")
    print("-" * 70)
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
