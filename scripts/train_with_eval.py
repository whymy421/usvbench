# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train RL agent with skrl + auto upload training videos to wandb.

Usage:
    python train_with_eval.py --task=Isaac-My-First-Task-Direct-v0 \
        --video --video_interval 50000 --video_length 200
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with skrl + video upload to wandb.")
# video default-on, but NO_VIDEO=1 turns it off (boat sweep sets this so two concurrent
# Isaac Sim instances don't crash fighting over the syntheticdata/render pipeline).
parser.add_argument("--video", action="store_true", default=(os.environ.get("NO_VIDEO") != "1"), help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=50000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default=None,
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework", type=str, default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm", type=str, default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

AppLauncher.add_app_launcher_args(parser)
_original_command = " ".join(sys.argv)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import glob
import random
from datetime import datetime

import omni
import skrl
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent + auto video upload to wandb."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training not supported on CPU.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # logging directories
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line: {log_dir}")
    # Sweep: 用环境变量覆盖wandb run name
    sweep_name = os.environ.get('WANDB_NAME', None)
    if sweep_name:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sweep_name = f"{timestamp}_{sweep_name}"
        agent_cfg["agent"]["experiment"]["experiment_name"] = sweep_name
        log_dir = os.path.join(log_root_path, sweep_name)

    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn("IO descriptors only supported for manager based envs.")

    env_cfg.log_dir = log_dir

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Video recording setup
    video_dir = None
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "train")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate runner. E8 uses a true tanh-squashed Gaussian;
    # skrl 1.4.3's clip_actions option is only a hard clamp and does not correct
    # the log-probability, so it cannot be used for this experiment.
    runner_class = Runner
    model_factory = agent_cfg.get("model_factory")
    if model_factory == "e8_squashed_gaussian":
        if not args_cli.ml_framework.startswith("torch"):
            raise ValueError("E8 squashed Gaussian is implemented for torch only")
        from isaaclab_tasks.direct.my_first_task_e8.agents.squashed_gaussian import E8SquashedRunner

        runner_class = E8SquashedRunner
    elif model_factory is not None:
        raise ValueError(f"Unknown custom model_factory: {model_factory}")

    runner = runner_class(env, agent_cfg)

    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update({"command": _original_command})
    except:
        pass

    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # ============================================
    # Video-only upload hook (no eval, no state mutation)
    # ============================================
    if args_cli.video and video_dir is not None:
        import wandb
        # 清空旧视频
        for old_mp4 in glob.glob(os.path.join(video_dir, "*.mp4")):
            os.remove(old_mp4)
        uploaded_videos = set()
        agent = runner.agent
        original_post_interaction = agent.post_interaction

        def post_interaction_with_video_upload(timestep, timesteps):
            original_post_interaction(timestep, timesteps)

            if timestep % 1000 == 0:
                try:
                    mp4_files = glob.glob(os.path.join(video_dir, "*.mp4"))
                    for mp4_path in mp4_files:
                        if mp4_path not in uploaded_videos:
                            file_size = os.path.getsize(mp4_path)
                            if file_size < 1000:
                                continue
                            if wandb.run is not None:
                                wandb.log({"Train/video": wandb.Video(mp4_path, format="mp4")})
                                uploaded_videos.add(mp4_path)
                                print(f"[VIDEO] Uploaded to wandb: {os.path.basename(mp4_path)}")
                except Exception:
                    pass

        agent.post_interaction = post_interaction_with_video_upload
        print(f"\n[INFO] Auto video upload enabled. Video dir: {video_dir}\n")

    # run training
    runner.run()

    # Final upload check
    if args_cli.video and video_dir is not None:
        import wandb
        try:
            mp4_files = glob.glob(os.path.join(video_dir, "*.mp4"))
            for mp4_path in mp4_files:
                if mp4_path not in uploaded_videos:
                    if wandb.run is not None:
                        wandb.log({"Train/video": wandb.Video(mp4_path, format="mp4")})
                        print(f"[VIDEO] Final upload: {os.path.basename(mp4_path)}")
        except Exception:
            pass

    # Explicit wandb cleanup BEFORE simulation_app.close()
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            print("[INFO] wandb run finished cleanly.")
    except Exception:
        pass

    # close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
