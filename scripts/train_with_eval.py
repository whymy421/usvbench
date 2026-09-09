# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E402  # Isaac Sim must launch before importing simulation modules

"""
Train an skrl agent, optionally record/upload videos, and optionally select the
best periodic checkpoint with a deterministic post-training evaluation sweep.

Usage:
    python train_with_eval.py --task=Isaac-My-First-Task-Direct-v0 \
        --video --video_interval 50000 --video_length 200

    python train_with_eval.py --task=Isaac-USVBench-Boat-Calm-Direct-v1 \
        --num_envs=64 --headless --max_iterations=9375 --seed=42 \
        --eval_mini_steps=1500 --eval_sweep_seed=2026

The sweep is disabled by default. ``--eval_mini_steps`` is retained for
compatibility with Arif's debug command; it means vector-environment steps per
checkpoint, not an interval between training iterations.
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
parser.add_argument(
    "--eval_mini_steps", "--eval_sweep_steps", dest="eval_sweep_steps", type=int, default=0,
    help="Post-training deterministic evaluation steps per checkpoint. Set to 0 to disable.",
)
parser.add_argument(
    "--eval_sweep_seed", type=int, default=2026,
    help="Fixed environment seed reused for every checkpoint in the sweep.",
)
parser.add_argument(
    "--no_eval_sweep", action="store_true", default=False,
    help="Disable checkpoint selection even if --eval_mini_steps is non-zero.",
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

import csv
import glob
import json
import random
import re
import shutil
from datetime import datetime

import gymnasium as gym
import omni
import skrl
import torch
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )

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


def _base_env(env):
    """Return the underlying Isaac Lab environment."""
    e = env
    for _ in range(12):
        if hasattr(e, "reached_count"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "_env"):
            e = e._env
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return e


def _set_eval_mode(agent):
    """Support both skrl 1.x and 2.x evaluation-mode APIs."""
    if hasattr(agent, "set_running_mode"):
        agent.set_running_mode("eval")
    elif hasattr(agent, "set_training_mode"):
        agent.set_training_mode(False)
    else:
        for model in agent.models.values():
            if hasattr(model, "eval"):
                model.eval()


def _observation_preprocessor(agent):
    """Locate the observation scaler across skrl naming versions."""
    for attr in (
        "_observation_preprocessor",
        "_state_preprocessor",
        "observation_preprocessor",
        "state_preprocessor",
    ):
        preprocessor = getattr(agent, attr, None)
        if preprocessor is not None and hasattr(preprocessor, "load_state_dict"):
            return preprocessor
    return None


def _load_checkpoint_compat(agent, checkpoint_path):
    """Load a checkpoint and migrate the skrl 1.x observation-scaler key."""
    agent.load(checkpoint_path)
    preprocessor = _observation_preprocessor(agent)
    if preprocessor is None:
        return False

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "observation_preprocessor" in raw:
        return False

    for old_key in ("state_preprocessor", "_state_preprocessor"):
        if old_key in raw:
            # Always reload for every candidate. Checking for an all-zero mean is
            # incorrect after the first old-format checkpoint in a sweep.
            preprocessor.load_state_dict(raw[old_key])
            return True
    return False


def _seeded_full_reset(env, seed, common_step_counter):
    """Force a real, reproducible reset of an skrl-wrapped Isaac Lab env."""
    base = _base_env(env)
    if hasattr(base, "common_step_counter"):
        base.common_step_counter = common_step_counter
    if hasattr(base, "reached_count"):
        base.reached_count = 0
    if hasattr(base, "episode_count"):
        base.episode_count = 0
    if hasattr(base, "_prev_distance"):
        base._prev_distance = None
    if hasattr(base, "_buf_idx"):
        base._buf_idx = 0
    if hasattr(base, "_action_buffer"):
        base._action_buffer = torch.zeros_like(base._action_buffer)
    if hasattr(base, "_prev_action_obs"):
        base._prev_action_obs = torch.zeros_like(base._prev_action_obs)
    if hasattr(base, "seed"):
        base.seed(seed)

    # skrl's IsaacLabWrapper intentionally resets only once. Re-arm that reset
    # so every candidate starts from the same seeded initial conditions.
    if hasattr(env, "_reset_once"):
        env._reset_once = True
        env._seed = seed
    else:
        raw_env = getattr(env, "_env", None)
        if raw_env is not None and hasattr(raw_env, "seed"):
            raw_env.seed(seed)
    # skrl's trainer leaves the simulator tensors in inference mode after the
    # rollout. IsaacLab reset writes into those tensors, so keep the reset in
    # the same mode instead of triggering PyTorch's inference-tensor guard.
    with torch.inference_mode():
        return env.reset()


def _deterministic_action(agent, observations):
    """Call the agent API so its own observation preprocessor is applied."""
    policy_observations = observations.get("policy", observations) if isinstance(observations, dict) else observations
    try:
        outputs = agent.act(policy_observations, None, timestep=0, timesteps=0)
    except TypeError:
        outputs = agent.act(policy_observations, timestep=0, timesteps=0)

    actions = outputs[0]
    info = outputs[-1] if isinstance(outputs[-1], dict) else {}
    return info.get("mean_actions", actions)


def _step_reaches(base, previous_count):
    """Count target events exactly, with a fallback for tasks without event masks."""
    reached_mask = getattr(base, "_last_reached_mask", None)
    current_count = int(getattr(base, "reached_count", previous_count))
    if isinstance(reached_mask, torch.Tensor):
        return int(reached_mask.sum().item()), current_count
    return max(current_count - previous_count, 0), current_count


def _mini_eval(agent, env, n_steps, num_envs, eval_seed, common_step_counter):
    """Evaluate one checkpoint from identical seeded initial conditions."""
    base = _base_env(env)
    max_episode_length = float(getattr(base, "max_episode_length", n_steps))
    observations, _ = _seeded_full_reset(env, eval_seed, common_step_counter)
    previous_count = int(getattr(base, "reached_count", 0))
    total_targets = 0
    oob_events = 0
    timeout_events = 0

    for _ in range(n_steps):
        with torch.inference_mode():
            actions = _deterministic_action(agent, observations)
            observations, _, terminated, truncated, _ = env.step(actions)

        reached_now, previous_count = _step_reaches(base, previous_count)
        total_targets += reached_now
        terminated = torch.as_tensor(terminated).bool()
        truncated = torch.as_tensor(truncated).bool()
        oob_events += int(terminated.sum().item())
        timeout_events += int((truncated & ~terminated).sum().item())

    episode_equivalents = (num_envs * n_steps) / max(max_episode_length, 1.0)
    return {
        "targets_per_episode": total_targets / max(episode_equivalents, 1e-9),
        "total_targets": total_targets,
        "episode_equivalents": episode_equivalents,
        "oob_events": oob_events,
        "timeout_events": timeout_events,
    }


def _checkpoint_sort_key(path):
    match = re.fullmatch(r"agent_(\d+)\.pt", os.path.basename(path))
    return (0, int(match.group(1))) if match else (1, os.path.basename(path))


def _run_checkpoint_sweep(runner, env, log_dir, n_steps, eval_seed, num_envs):
    """Select best_agent.pt by deterministic target throughput."""
    if not args_cli.ml_framework.startswith("torch"):
        raise ValueError("Checkpoint evaluation sweep is currently supported for torch agents only")

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    candidates = glob.glob(os.path.join(checkpoint_dir, "agent_*.pt"))
    reward_best_path = os.path.join(checkpoint_dir, "best_agent.pt")
    if os.path.isfile(reward_best_path):
        candidates.append(reward_best_path)
    candidates = sorted(set(candidates), key=_checkpoint_sort_key)

    if not candidates:
        print("[EVAL SWEEP] No checkpoints found; checkpoint selection skipped.")
        return

    base = _base_env(env)
    common_step_counter = int(getattr(base, "common_step_counter", 0))
    results = []
    best_path = None
    best_score = -float("inf")

    print(
        f"\n[EVAL SWEEP] {len(candidates)} checkpoints x {n_steps} steps "
        f"with fixed env seed {eval_seed}"
    )
    for checkpoint_path in candidates:
        used_compat = _load_checkpoint_compat(runner.agent, checkpoint_path)
        _set_eval_mode(runner.agent)
        metrics = _mini_eval(
            runner.agent,
            env,
            n_steps=n_steps,
            num_envs=num_envs,
            eval_seed=eval_seed,
            common_step_counter=common_step_counter,
        )
        row = {
            "checkpoint": os.path.basename(checkpoint_path),
            "eval_seed": eval_seed,
            "eval_steps": n_steps,
            "old_preprocessor_compat": int(used_compat),
            **metrics,
        }
        results.append(row)
        score = metrics["targets_per_episode"]
        marker = ""
        if score > best_score:
            best_score = score
            best_path = checkpoint_path
            marker = " <- best so far"
        print(
            f"  {row['checkpoint']:30s} {score:7.3f} tgt/ep "
            f"({row['total_targets']} targets){marker}"
        )

    csv_path = os.path.join(log_dir, "eval_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    reward_backup_path = os.path.join(checkpoint_dir, "best_agent_by_reward.pt")
    if os.path.abspath(best_path) != os.path.abspath(reward_best_path):
        # Keep the trainer's reward-selected checkpoint for auditability, but
        # always materialize the sweep winner at the launcher's stable path.
        # skrl's auto checkpointing may not create best_agent.pt at all.
        if os.path.isfile(reward_best_path):
            shutil.copy2(reward_best_path, reward_backup_path)
        shutil.copy2(best_path, reward_best_path)

    selection = {
        "selected_checkpoint": os.path.basename(best_path),
        "selected_targets_per_episode": best_score,
        "eval_seed": eval_seed,
        "eval_steps": n_steps,
        "num_envs": num_envs,
        "reward_best_backup": os.path.basename(reward_backup_path) if os.path.isfile(reward_backup_path) else None,
    }
    with open(os.path.join(log_dir, "best_agent_selection.json"), "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)

    print(f"[EVAL SWEEP] Selected {selection['selected_checkpoint']} at {best_score:.3f} tgt/ep")
    print(f"[EVAL SWEEP] Results: {csv_path}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl and optionally select a checkpoint by fixed-seed eval."""
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
    except Exception:
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

    eval_sweep_steps = 0 if args_cli.no_eval_sweep else args_cli.eval_sweep_steps
    if eval_sweep_steps < 0:
        raise ValueError("--eval_mini_steps/--eval_sweep_steps must be >= 0")
    if eval_sweep_steps:
        num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        _run_checkpoint_sweep(
            runner,
            env,
            log_dir=log_dir,
            n_steps=eval_sweep_steps,
            eval_seed=args_cli.eval_sweep_seed,
            num_envs=num_envs,
        )

    # close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
