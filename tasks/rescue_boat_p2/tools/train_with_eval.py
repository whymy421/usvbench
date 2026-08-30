"""
train_with_eval.py — USVBench training with post-training eval sweep.

Identical to Isaac Lab's train.py for the training phase, then automatically
sweeps all intermediate checkpoints with a fast inline eval and saves the one
with the highest targets_per_episode as best_agent.pt.

This replicates what Yutong's train_with_eval.py does: select best_agent.pt
by actual eval performance (targets/ep), not by training reward.

Usage: identical to train.py
  python train_with_eval.py --task=Isaac-My-First-Task-Calm-Boat-Direct-v1 \
      --num_envs=64 --headless --max_iterations=9375 --seed=42

Env-var config is printed at startup for diagnostics (OOB_PENALTY etc.).
Pass --no_eval_sweep to skip the sweep and keep the training-reward best_agent.pt.
Pass --eval_mini_steps=N to adjust sweep speed (default 500; full eval uses 6000).
"""

"""Launch Isaac Sim Simulator first."""
import argparse
import sys
import os

# ── Startup diagnostics ───────────────────────────────────────────────────────
_DIAG_VARS = [
    "REWARD_VARIANT", "OBS_DIM", "OBS_EXTENDED",
    "OOB_PENALTY", "REACH_BONUS", "SPEED_COUPLE",
    "FORWARD_TRANSIT", "USVBENCH_ASSETS",
]
print("\n[train_with_eval] Env-var config at startup:")
for _v in _DIAG_VARS:
    print(f"  {_v:22s} = {os.environ.get(_v, '(not set)')}")
print()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train RL agent with skrl + post-training eval sweep.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
# Eval sweep args
parser.add_argument("--eval_mini_steps", type=int, default=500,
                    help="Steps per checkpoint during post-training eval sweep (default 500).")
parser.add_argument("--no_eval_sweep", action="store_true", default=False,
                    help="Skip post-training eval sweep; keep training-reward best_agent.pt.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import glob
import logging
import random
import shutil
import time
from datetime import datetime

import gymnasium as gym
import skrl
import torch
from packaging import version

SKRL_VERSION = "2.0.0"
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
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# ── Eval helpers ──────────────────────────────────────────────────────────────

def _base_env(env):
    """Unwrap wrappers to find the Isaac env that has reached_count."""
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
    """Return the agent's observation preprocessor, or None."""
    for attr in ("_obs_preprocessor", "_observation_preprocessor", "observation_preprocessor"):
        cand = getattr(agent, attr, None)
        if cand is not None and callable(cand):
            return cand
    return None


def _restore_preprocessor(agent, ckpt_path):
    """
    Restore the observation normaliser explicitly. Returns (ok, message).

    The previous implementation only acted when running_mean was exactly zero and
    only looked for the legacy 'state_preprocessor' key. These checkpoints store
    it as 'observation_preprocessor', so the fallback never fired, and a
    checkpoint could be scored while normalised for a DIFFERENT checkpoint. A
    diagnostic showed this produced 0.000 for most checkpoints and made scores
    depend on evaluation order.
    """
    obs_prep = _get_obs_prep(agent)
    if obs_prep is None or not hasattr(obs_prep, "load_state_dict"):
        return True, "no stateful preprocessor"
    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"checkpoint unreadable: {e}"
    for k in ("observation_preprocessor", "_observation_preprocessor",
              "state_preprocessor", "_state_preprocessor"):
        if k in raw:
            try:
                obs_prep.load_state_dict(raw[k])
                return True, f"restored '{k}'"
            except Exception as e:
                return False, f"'{k}' present but load failed: {e}"
    return False, "no preprocessor key in checkpoint"


def _mini_eval(agent, env, n_steps, num_envs, n_casualties=4, burn_in=60):
    """
    Score a checkpoint over WHOLE episodes and return (rescue_rate, tgt_per_ep).

    Two corrections against the original implementation:

    1. Whole episodes, correct denominator. The old version ran 1500 steps but
       divided by (num_envs * 1500) / max_ep = 13.33 "episode-equivalents". 1500
       steps is 25 s of a 120 s episode, and nearly every rescue that will occur
       has already occurred by then, so the numerator was near-complete while the
       denominator was scaled down -- inflating rescue rate by about 4.6x. Here
       each env runs complete episodes and the metric is the definition the P2
       bar refers to:  rescues / (num_envs * n_casualties * episodes).

    2. Burn-in after reset. Scores previously depended on which checkpoint had
       been evaluated before, so a discarded burn-in guarantees nothing counted
       here originated in a previous evaluation.
    """
    obs_prep = _get_obs_prep(agent)
    base = _base_env(env)
    max_ep = int(getattr(base, "max_episode_length", n_steps))

    def _act(obs):
        obs_raw = obs["policy"] if isinstance(obs, dict) else obs
        obs_in = obs_prep(obs_raw, train=False) if obs_prep is not None else obs_raw
        _a, _i = agent.policy.act({"observations": obs_in}, role="policy")
        return _i.get("mean_actions", _a)

    obs, _ = env.reset()
    for _ in range(burn_in):
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(_act(obs))
    obs, _ = env.reset()

    episodes = max(1, int(round(n_steps / max(max_ep, 1))))
    total_steps = max_ep * episodes

    reached_total = 0
    reached_prev = int(getattr(base, "reached_count", 0))

    for _ in range(total_steps):
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(_act(obs))
        cur = int(getattr(base, "reached_count", 0))
        if cur > reached_prev:
            reached_total += cur - reached_prev
        reached_prev = cur

    denom = num_envs * n_casualties * episodes
    rescue_rate = reached_total / max(denom, 1)
    tgt_per_ep = reached_total / max(num_envs * episodes, 1)

    # A rate above 1.0 is impossible and means the counting window has spilled
    # into a second episode, where fresh casualties spawn. Clamp and warn rather
    # than reporting a value that cannot be true; use eval_fixed.py, which samples
    # per-episode counts at episode end, for any number that will be reported.
    if rescue_rate > 1.0:
        print(f"  [WARN] rescue_rate {rescue_rate:.4f} > 1.0 — counting window "
              f"crossed an episode boundary. Re-score with eval_fixed.py.")
        rescue_rate = min(rescue_rate, 1.0)

    return rescue_rate, tgt_per_ep


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl, then sweep checkpoints to select best by eval score."""

    # ── Setup (identical to train.py) ─────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported with CPU device.")
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

    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. Skipping."
        )

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    runner = Runner(env, agent_cfg)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # ── Training ──────────────────────────────────────────────────────────────
    runner.run()
    train_time = round(time.time() - start_time, 2)
    print(f"Training time: {train_time} seconds")

    # ── Log recorded videos to wandb ───────────────────────────────────────────
    if args_cli.video:
        try:
            import wandb as _wv
            if _wv.run is not None:
                _vid_dir = os.path.join(log_dir, "videos", "train")
                _mp4s = sorted(glob.glob(os.path.join(_vid_dir, "*.mp4"))) if os.path.exists(_vid_dir) else []
                if _mp4s:
                    print(f"\n[VIDEO] Logging {len(_mp4s)} video(s) to wandb ...")
                    for _i, _vp in enumerate(_mp4s):
                        try:
                            _wv.log({f"Video/episode_{_i:04d}": _wv.Video(_vp, fps=30, format="mp4")})
                            print(f"  logged: {os.path.basename(_vp)}")
                        except Exception as _e:
                            print(f"  [WARN] could not log {_vp}: {_e}")
                else:
                    print("[VIDEO] No mp4 files found to log.")
        except ImportError:
            pass

    # ── Post-training eval sweep ───────────────────────────────────────────────
    if args_cli.no_eval_sweep:
        print("[EVAL SWEEP] Skipped (--no_eval_sweep). Keeping training-reward best_agent.pt.")
        env.close()
        return

    num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    n_steps = args_cli.eval_mini_steps
    ckpt_dir = os.path.join(log_dir, "checkpoints")

    ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "agent_*.pt")),
        key=lambda p: int(os.path.basename(p).replace("agent_", "").replace(".pt", ""))
    )

    if not ckpts:
        print("[EVAL SWEEP] No intermediate checkpoints found — keeping best_agent.pt from training.")
        env.close()
        return

    print(f"\n[EVAL SWEEP] Sweeping {len(ckpts)} checkpoints × {n_steps} steps each ...")
    print(f"  (adjust with --eval_mini_steps=N; skip with --no_eval_sweep)\n")

    # Put models in eval mode
    for _m in runner.agent.models.values():
        if hasattr(_m, "eval"):
            _m.eval()

    n_cas = int(getattr(env_cfg, "n_casualties", 4))

    best_score = -float("inf")
    best_ckpt_path = None
    eval_rows = []

    for ckpt_path in ckpts:
        runner.agent.load(ckpt_path)
        ok, msg = _restore_preprocessor(runner.agent, ckpt_path)
        label = os.path.basename(ckpt_path)

        if not ok:
            # Scoring a policy against another policy's normalisation is
            # meaningless, so skip rather than record a misleading number.
            print(f"  {label:30s}  SKIPPED — {msg}")
            eval_rows.append((label, None, None))
            continue

        rescue_rate, tgt_per_ep = _mini_eval(
            runner.agent, env, n_steps=n_steps, num_envs=num_envs, n_casualties=n_cas
        )
        is_best = rescue_rate > best_score
        eval_rows.append((label, rescue_rate, tgt_per_ep))
        marker = " ← best so far" if is_best else ""
        print(f"  {label:30s}  rescue_rate {rescue_rate:.4f}  "
              f"({tgt_per_ep:.3f} tgt/ep){marker}")

        if is_best:
            best_score = rescue_rate
            best_ckpt_path = ckpt_path

    # ── Save best as best_agent.pt ─────────────────────────────────────────────
    best_agent_path = os.path.join(ckpt_dir, "best_agent.pt")
    if best_ckpt_path:
        shutil.copy2(best_ckpt_path, best_agent_path)
        print(f"\n[EVAL SWEEP] Best: {os.path.basename(best_ckpt_path)}  "
              f"rescue_rate {best_score:.4f}")
        print(f"[EVAL SWEEP] P2 bar 0.70 → "
              f"{'MET' if best_score >= 0.70 else f'{best_score / 0.70 * 100:.0f}% of bar'}")
        print(f"[EVAL SWEEP] Saved as best_agent.pt → {best_agent_path}")
    else:
        print("[EVAL SWEEP] No checkpoint evaluated — keeping existing best_agent.pt.")

    # Flag a flat curve explicitly. A spread this small across the whole run means
    # training is not progressing, which is easy to miss in a column of numbers.
    scored = [r[1] for r in eval_rows if r[1] is not None]
    if len(scored) >= 3:
        spread = max(scored) - min(scored)
        if spread < 0.05:
            print(f"\n[EVAL SWEEP] WARNING: rescue_rate spans only {spread:.4f} across "
                  f"{len(scored)} checkpoints.")
            print("[EVAL SWEEP] That is a flat learning curve — the policy is not improving.")
            print("[EVAL SWEEP] Check the environment before tuning hyperparameters.")

    # ── Save sweep CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(log_dir, "eval_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("checkpoint,rescue_rate,targets_per_episode\n")
        for name, rr, tpe in eval_rows:
            if rr is None:
                f.write(f"{name},,\n")
            else:
                f.write(f"{name},{rr:.4f},{tpe:.4f}\n")
    print(f"[EVAL SWEEP] Full results → {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
