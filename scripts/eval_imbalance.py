"""Frozen-level eval at one scalar thrust-imbalance value.

This mirrors eval_v6_frozen.py so a single checkpoint can be evaluated on
paired layout streams while the Suite D training-choice randomizer is off.
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-USV-HazardCross-Direct-v1")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument("--imbalance", type=float, default=None)
# Suite D grew from one axis to five. Same evaluator, one axis at a time:
# --axis names the cfg field, --value is the held-out perturbation. The
# original --imbalance form still works and means --axis thrust_imbalance.
parser.add_argument("--axis", default=None,
                    help="cfg field to override, e.g. mass_scale, drag_scale, "
                         "thrust_cap_scale, motor_tau_s, thrust_imbalance")
parser.add_argument("--value", type=float, default=None)
parser.add_argument("--out", default=None, help="JSON per-episode records")
parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="extra top-level cfg overrides (repeatable), e.g. "
                         "--set layout_max_attempts=150")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

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
if args_cli.axis is not None:
    axis, value = args_cli.axis, args_cli.value
    if value is None:
        raise SystemExit("--axis requires --value")
elif args_cli.imbalance is not None:
    axis, value = "thrust_imbalance", args_cli.imbalance
else:
    raise SystemExit("give --axis/--value or --imbalance")
if not hasattr(env_cfg, axis):
    raise SystemExit(f"cfg has no field {axis!r}")
setattr(env_cfg, axis, value)
# Kill the per-episode randomization for EVERY axis, not just the one under
# test: a held-out evaluation must vary exactly one thing.
for choices in ("thrust_imbalance_choices", "mass_scale_choices",
                "drag_scale_choices", "thrust_cap_scale_choices",
                "motor_tau_s_choices"):
    if hasattr(env_cfg, choices):
        setattr(env_cfg, choices, ())
for assignment in args_cli.extra_sets:
    field, _, raw = assignment.partition("=")
    if not _ or not hasattr(env_cfg, field):
        raise SystemExit(f"--set target {field!r} is not a cfg field")
    current = getattr(env_cfg, field)
    caster = type(current) if isinstance(current, (int, float, bool)) else str
    setattr(env_cfg, field, caster(raw) if caster is not bool
            else raw.lower() in ("1", "true", "yes"))
    print(f"set {field}={getattr(env_cfg, field)}", flush=True)
print(f"axis={axis} value={value}", flush=True)
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
experiment_cfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False
runner = Runner(wrapped, experiment_cfg)
runner.agent.load(os.path.abspath(args_cli.checkpoint))
runner.agent.set_running_mode("eval")

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
max_steps = (args_cli.episodes // base.num_envs + 3) * base.max_episode_length
step = 0
while len(records) < args_cli.episodes and step < max_steps:
    with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    # d0_per_env is rewritten inside _reset_idx, which runs during step(), so
    # reading it alongside the other episode stats would report the NEXT
    # episode's straight-line distance. Snapshot it while it still belongs to
    # the episode that is about to end.
    d0_prev = (base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None)
    obs, _, term, trunc, _ = wrapped.step(actions)
    step += 1
    done = term | trunc
    done = done.squeeze(-1) if done.dim() > 1 else done
    ids = torch.nonzero(done).flatten()
    for i in ids.tolist():
        rec = {
            "env": i,
            "ep": int(ep_counter[i]),
            "success": bool(base.episode_success[i]),
            "tts_s": (None if math.isnan(float(base.time_to_success[i]))
                      else float(base.time_to_success[i])),
            **(
                {"min_clearance_m": float(base.episode_min_clearance[i])}
                if hasattr(base, "episode_min_clearance")
                else {}
            ),
            "path_length_m": float(base.episode_path_length[i]),
        }
        # Per-primitive digests of the scenario the FINISHED episode ran,
        # latched at reset exactly like episode_min_clearance.  Absent on
        # families not yet on the scenario protocol, and empty until an env has
        # completed its first episode; both cases are guarded inside the
        # helper, which returns None for "write no field".
        scenario_hashes = episode_scenario_hashes_for(base, i)
        if scenario_hashes is not None:
            rec["scenario_hashes"] = scenario_hashes
        if hasattr(base, "episode_gates_passed"):
            rec["gates"] = int(base.episode_gates_passed[i])
        if d0_prev is not None:
            rec["d0_m"] = float(d0_prev[i])
        records.append(rec)
        ep_counter[i] += 1

records = records[: args_cli.episodes]
n = len(records)
succ = [r for r in records if r["success"]]
tts = sorted(r["tts_s"] for r in succ if r["tts_s"] is not None)
clr = sorted(r["min_clearance_m"] for r in records if "min_clearance_m" in r)
collided = sum(1 for c in clr if c < 0.0)


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[k]


sr = len(succ) / max(n, 1)
print(f"EVAL task={TASK} level={args_cli.level} seed={args_cli.eval_seed} "
      f"ckpt={os.path.basename(args_cli.checkpoint)}")
print(f"imbalance={axis}={value}")
print(f"  episodes={n} SR={sr:.4f} ({len(succ)}/{n})")
print(f"  tts median={pct(tts, 0.5):.1f}s p90={pct(tts, 0.9):.1f}s" if tts
      else "  tts: no successes")
print(f"  collision_episodes={collided}/{n} ({collided / max(n, 1):.3f}) "
      f"min_clearance p10={pct(clr, 0.10):.2f}m")
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
                   "imbalance": args_cli.imbalance,
                   "checkpoint": os.path.abspath(args_cli.checkpoint),
                   "records": records}, f, indent=1)
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
