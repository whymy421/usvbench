"""Frozen-level eval of a phase-switching composite policy.

Layouts come from the env's layout RNG (seeded by cfg seed) and reset timing
is fixed-horizon, so composite policies evaluated with the same --eval-seed
see the IDENTICAL episode/layout stream -> per-episode records support paired
comparisons offline.
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-USV-HarborMissionKin-Direct-v1")
parser.add_argument("--nav-ckpt", required=True)
parser.add_argument("--nav-id", required=True)
parser.add_argument("--dock-ckpt", required=True)
parser.add_argument("--dock-id", required=True)
parser.add_argument("--switch-phase", type=int, default=2)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument("--out", default=None, help="JSON per-episode records")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math
import os

import gymnasium as gym
import numpy as np
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# obs_bridge sits next to this script; running it by absolute path from a .bat
# leaves the script directory off sys.path, so put it there explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_bridge  # noqa: E402

# Scenario-protocol stamping, on the same dual import scripts/
# eval_v6_frozen.py:74-80 uses so the script keeps working from the repo and
# from the deployed task tree.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
try:
    from tasks._shared.scenario_draws import (
        episode_scenario_hashes_for,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )
except ImportError:
    from isaaclab_tasks.direct._shared.scenario_draws import (
        episode_scenario_hashes_for,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )

TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env_cfg.seed = args_cli.eval_seed
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")


def _set_runner_observation_space(native_id):
    """Advertise one champion's input width while Runner builds its models."""

    native_dim = len(obs_bridge._layout(native_id))
    native_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(native_dim,),
        dtype=np.float32,
    )
    if hasattr(wrapped, "_observation_space"):
        wrapped._observation_space = native_space
    else:
        wrapped.observation_space = native_space


def _make_runner(native_id, checkpoint):
    experiment_cfg = load_cfg_from_registry(native_id, "skrl_cfg_entry_point")
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["agent"]["experiment"]["wandb"] = False
    _set_runner_observation_space(native_id)
    runner = Runner(wrapped, experiment_cfg)
    runner.agent.load(os.path.abspath(checkpoint))
    runner.agent.set_running_mode("eval")
    return runner


# Runner only uses the wrapper's spaces to construct its models. Both agents
# retain their independently sized models/preprocessors and share the one real
# env; evaluation calls agent.act directly, never either Runner's trainer.
task_observation_space = wrapped.observation_space
try:
    nav_runner = _make_runner(args_cli.nav_id, args_cli.nav_ckpt)
    dock_runner = _make_runner(args_cli.dock_id, args_cli.dock_ckpt)
finally:
    if hasattr(wrapped, "_observation_space"):
        wrapped._observation_space = task_observation_space
    else:
        wrapped.observation_space = task_observation_space


def _native_observation(observation, native_id):
    if native_id == TASK:
        return observation
    return obs_bridge.project(observation, TASK, native_id)


base = env.unwrapped
# Does THIS env carry the scenario protocol?  Same guard, same marker and same
# warning as scripts/eval_v6_frozen.py:148-162, so this certificate and a
# frozen-eval certificate at the same --eval-seed can be scenario-paired.
scenario_protocol = scenario_protocol_stamp(base)
_notice = scenario_protocol_notice(TASK, scenario_protocol)
if _notice:
    print(_notice, flush=True)
obs, _ = wrapped.reset()
records = []
ep_counter = torch.zeros(base.num_envs, dtype=torch.long)
phase = base.phase.reshape(-1)
episode_steps = torch.zeros_like(phase, dtype=torch.long)
switch_steps = torch.full_like(phase, -1, dtype=torch.long)
max_steps = (args_cli.episodes // base.num_envs + 3) * base.max_episode_length
step = 0
while len(records) < args_cli.episodes and step < max_steps:
    phase = base.phase.reshape(-1)
    dock_mask = phase >= args_cli.switch_phase
    newly_switched = dock_mask & (switch_steps < 0)
    switch_steps[newly_switched] = episode_steps[newly_switched]
    nav_obs = _native_observation(obs, args_cli.nav_id)
    dock_obs = _native_observation(obs, args_cli.dock_id)
    with torch.inference_mode():
        nav_outputs = nav_runner.agent.act(nav_obs, timestep=0, timesteps=0)
        nav_actions = nav_outputs[-1].get("mean_actions", nav_outputs[0])
        dock_outputs = dock_runner.agent.act(dock_obs, timestep=0, timesteps=0)
        dock_actions = dock_outputs[-1].get("mean_actions", dock_outputs[0])
        action_mask = dock_mask
        while action_mask.dim() < nav_actions.dim():
            action_mask = action_mask.unsqueeze(-1)
        actions = torch.where(action_mask, dock_actions, nav_actions)
    # d0_per_env is rewritten inside _reset_idx, which runs during step(), so
    # reading it alongside the other episode stats would report the NEXT
    # episode's straight-line distance. Snapshot it while it still belongs to
    # the episode that is about to end.
    d0_prev = (base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None)
    obs, _, term, trunc, _ = wrapped.step(actions)
    step += 1
    episode_steps += 1
    done = term | trunc
    done = done.squeeze(-1) if done.dim() > 1 else done
    ids = torch.nonzero(done).flatten()
    for i in ids.tolist():
        episode_contact_steps = getattr(base, "episode_contact_steps", None)
        episode_contact_longest_steps = getattr(
            base, "episode_contact_longest_steps", None
        )
        episode_contact_depth_sum = getattr(
            base, "episode_contact_depth_sum", None
        )
        episode_max_phase = getattr(base, "episode_max_phase", None)
        rec = {
            "env": i,
            "ep": int(ep_counter[i]),
            "success": bool(base.episode_success[i]),
            "tts_s": (None if math.isnan(float(base.time_to_success[i]))
                      else float(base.time_to_success[i])),
            "min_clearance_m": float(base.episode_min_clearance[i]),
            "path_length_m": float(base.episode_path_length[i]),
            "switch_step": (None if int(switch_steps[i]) < 0
                            else int(switch_steps[i])),
        }
        # Per-primitive digests of the scenario the FINISHED episode ran,
        # latched at reset exactly like episode_min_clearance.  Absent on
        # families not yet on the scenario protocol, and empty until an env has
        # completed its first episode; both cases are guarded inside the
        # helper, which returns None for "write no field".
        scenario_hashes = episode_scenario_hashes_for(base, i)
        if scenario_hashes is not None:
            rec["scenario_hashes"] = scenario_hashes
        if (
            episode_contact_steps is not None
            and episode_contact_longest_steps is not None
            and episode_contact_depth_sum is not None
        ):
            contact_steps = float(episode_contact_steps[i])
            rec["contact_steps"] = int(contact_steps)
            rec["contact_seconds"] = contact_steps * base.control_step_s
            rec["contact_longest_seconds"] = (
                float(episode_contact_longest_steps[i]) * base.control_step_s
            )
            rec["contact_depth_mean_m"] = (
                float(episode_contact_depth_sum[i]) / contact_steps
                if contact_steps > 0.0
                else 0.0
            )
        if episode_max_phase is not None:
            rec["max_phase"] = int(episode_max_phase[i])
        if hasattr(base, "episode_gates_passed"):
            rec["gates"] = int(base.episode_gates_passed[i])
        if d0_prev is not None:
            rec["d0_m"] = float(d0_prev[i])
        records.append(rec)
        ep_counter[i] += 1
    episode_steps[ids] = 0
    switch_steps[ids] = -1

records = records[: args_cli.episodes]
n = len(records)
succ = [r for r in records if r["success"]]
tts = sorted(r["tts_s"] for r in succ if r["tts_s"] is not None)
collided = sum(1 for r in records if r["min_clearance_m"] < 0.0)
clr = sorted(r["min_clearance_m"] for r in records)


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[k]


sr = len(succ) / max(n, 1)
print(f"EVAL task={TASK} level={args_cli.level} seed={args_cli.eval_seed} "
      f"nav_ckpt={os.path.basename(args_cli.nav_ckpt)} "
      f"dock_ckpt={os.path.basename(args_cli.dock_ckpt)}")
print(f"  episodes={n} SR={sr:.4f} ({len(succ)}/{n})")
print(f"  tts median={pct(tts, 0.5):.1f}s p90={pct(tts, 0.9):.1f}s" if tts
      else "  tts: no successes")
print(f"  collision_episodes={collided}/{n} ({collided / max(n, 1):.3f}) "
      f"min_clearance p10={pct(clr, 0.10):.2f}m")
if getattr(base, "episode_max_phase", None) is not None:
    phase_distribution = {
        phase: sum(r.get("max_phase") == phase for r in records)
        for phase in range(4)
    }
    print(f"  stages: reached_phase distribution {phase_distribution}")
switched = [r for r in records if r["switch_step"] is not None]
post_switch_success = sum(r["success"] for r in switched)
print(f"  composite: switched={len(switched)}/{n} "
      f"post_switch_success={post_switch_success}/{len(switched)}")
if records and all("contact_steps" in r for r in records):
    contact_records = [r for r in records if r["contact_steps"] > 0]
    contact_seconds = sorted(r["contact_seconds"] for r in contact_records)
    contact_longest_seconds = sorted(
        r["contact_longest_seconds"] for r in contact_records
    )
    total_contact_steps = sum(r["contact_steps"] for r in contact_records)
    depth_mean = (
        sum(
            r["contact_depth_mean_m"] * r["contact_steps"]
            for r in contact_records
        ) / total_contact_steps
        if total_contact_steps > 0
        else float("nan")
    )
    print(
        f"  contact: episodes_with_contact={len(contact_records)}/{n} "
        f"total_s median={pct(contact_seconds, 0.5):.2f} "
        f"p90={pct(contact_seconds, 0.9):.2f}  "
        f"longest_s median={pct(contact_longest_seconds, 0.5):.2f}  "
        f"depth_mean={depth_mean:.3f} m"
    )
# Success rate alone has no discriminative power on a task where detouring
# works: the shifted-potential run certified 100%/100% on Task A while walking
# a 68.6 m median path against a 29.7 m straight line -- a BIGGER detour than
# the baseline policy the validity audit had already flagged as going around.
# Only the path length made that visible, so certification prints it.
path = sorted(r["path_length_m"] for r in succ) or sorted(
    r["path_length_m"] for r in records)
print(f"  path_len median={pct(path, 0.5):.1f}m p90={pct(path, 0.9):.1f}m", end="")
ratios = sorted(r["path_length_m"] / r["d0_m"] for r in succ
                if r.get("d0_m", 0.0) > 0.0)
print(f"  detour median={pct(ratios, 0.5):.2f}x straight" if ratios else "")
if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "level": args_cli.level,
                   "seed": args_cli.eval_seed,
                   # The header block when the env carries the protocol
                   # object, the off-protocol literal when it does not --
                   # never unconditionally the header.
                   "scenario_protocol": scenario_protocol,
                   "nav_checkpoint": os.path.abspath(args_cli.nav_ckpt),
                   "nav_id": args_cli.nav_id,
                   "dock_checkpoint": os.path.abspath(args_cli.dock_ckpt),
                   "dock_id": args_cli.dock_id,
                   "switch_phase": args_cli.switch_phase,
                   "records": records}, f, indent=1)
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
