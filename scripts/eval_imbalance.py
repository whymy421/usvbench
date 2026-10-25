"""Frozen-level eval at one scalar thrust-imbalance value.

This mirrors eval_v6_frozen.py so a single checkpoint can be evaluated on
paired layout streams while the Suite D training-choice randomizer is off.
"""
import argparse
import json
import os
import sys

ORIGINAL_ARGV = list(sys.argv)
try:
    from cfg_override import apply_overrides
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cfg_override import apply_overrides
from cert_v3 import CertificateRun
from eval_protocol import add_protocol_arguments, preflight_protocol, resolve_protocol

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
                    help="extra cfg overrides (repeatable); supports dotted "
                         "paths and comma-separated tuples, e.g. "
                         "--set layout_max_attempts=150")
add_protocol_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
preflight_protocol(args_cli, ORIGINAL_ARGV, parser)
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
        episode_appearance_fields_for,
        episode_scenario_hashes_for,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )
except ImportError:
    from isaaclab_tasks.direct._shared.scenario_draws import (
        episode_appearance_fields_for,
        episode_scenario_hashes_for,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )

TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.eval_seed
# Every cfg field this run actually writes, as {field: value read back off the
# cfg}. Same key name and same flat {path: value} shape as the "overrides"
# stamp scripts/eval_v6_frozen.py:340 writes from apply_overrides(), so one
# reader serves both certificates even though this script pins its knobs by a
# different mechanism. Before this, gcert_nc5376_drag_scale_v20_e123.json
# carried {checkpoint, imbalance, level, records, seed, task} and the 20 lived
# in the FILENAME alone: copy or rename the file and the condition it certifies
# is gone.
applied_overrides = {}
# Which of those fields is the Suite D axis under test, or None when no axis
# was pinned. Bound BEFORE the branch below, not inside it: the branch happens
# to exit when neither --axis nor --imbalance is given, so `axis` is in fact
# always bound by the time the blob is built, but that is a property of a
# SystemExit far upstream, and a certificate writer must not be one edit to
# that branch away from dying on a NameError after the episodes have run.
pinned_axis = None
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
# The shared resolver applies the dedicated axis with typed coercion.
pinned_axis = axis
# Read the knob back off the cfg instead of recording args_cli.value: what the
# certificate has to prove is the value the EPISODES ran at, and a cfg that
# coerces or clamps on assignment would otherwise be certified with the number
# that was asked for rather than the one that was used.
# Final axis value is read back by resolve_protocol below.
# Kill the per-episode randomization for EVERY axis, not just the one under
# test: a held-out evaluation must vary exactly one thing.
for choices in ("thrust_imbalance_choices", "mass_scale_choices",
                "drag_scale_choices", "thrust_cap_scale_choices",
                "motor_tau_s_choices"):
    if hasattr(env_cfg, choices):
        setattr(env_cfg, choices, ())
        # Record the blanking too. A cell whose choices tuple was emptied held
        # one knob value for every episode; a cell that still carries
        # (0.80, 1.00, 1.30) drew a fresh one per episode, and the two are not
        # the same evidence -- attributing a success rate to a knob VALUE is
        # only legitimate for the first kind. A reader who cannot see which
        # kind a certificate is cannot check that attribution, and an absent
        # key is ambiguous between "not blanked" and "written by a script that
        # never blanked".
        applied_overrides[choices] = getattr(env_cfg, choices)
applied_overrides.update(resolve_protocol(env_cfg, args_cli))
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
cert = CertificateRun(
    base=base, env_cfg=env_cfg, evaluator=__file__, task=TASK,
    controller="learned",
    controller_config={
        "cfg_entry_point": "skrl_cfg_entry_point",
        "experiment": experiment_cfg,
        "runtime_agent_config": getattr(runner.agent, "cfg", None),
        "runtime_models": {
            name: str(model)
            for name, model in getattr(runner.agent, "models", {}).items()
        },
        "action_selection": "mean_actions_if_available",
    },
    checkpoint=args_cli.checkpoint, eval_seed=args_cli.eval_seed,
    level=args_cli.level, episodes_requested=args_cli.episodes,
    overrides=applied_overrides, argv=ORIGINAL_ARGV, cli_args=vars(args_cli),
)
# Does THIS env carry the scenario protocol?  Same guard, same marker and same
# warning as scripts/eval_v6_frozen.py:148-162, so this certificate and a
# frozen-eval certificate at the same --eval-seed can be scenario-paired.
scenario_protocol = scenario_protocol_stamp(base)
_notice = scenario_protocol_notice(TASK, scenario_protocol)
if _notice:
    print(_notice, flush=True)
obs, _ = wrapped.reset()
cert.after_reset()
records = []
ep_counter = torch.zeros(base.num_envs, dtype=torch.long)
# PER-ENV QUOTA, not a pooled count (fixed 2026-08-31, owner-directed).
#
# This loop used to stop the instant `len(records)` reached --episodes, which
# collected THE FIRST N EPISODES TO COMPLETE across desynchronising envs.  Fast
# episodes -- quick successes and quick collisions -- were over-sampled, and an
# env still mid-episode at the cutoff contributed nothing at all, so timeouts
# (i.e. failures) were deleted preferentially.  Measured over 2077 finished
# certificates, 37.4% were materially affected, the worst reporting 68.8% where
# the unbiased figure was 45.3%, and the sign was controller-dependent -- which
# is worse than a constant offset, because it moves comparisons.  Re-running one
# cell at 128 and at 512 episodes moved the headline by up to 10 points while
# the first-episode-per-env estimate was identical to the record seven times out
# of seven.
#
# Quota form: every env runs exactly `quota` episodes and each contributes
# exactly `quota` records, so the sample is a fixed (env, ep) grid that no
# completion order can reweight.  This is by construction what the 669 already
# clean certificates in the corpus look like.  The step budget is unchanged.
quota = cert.quota
max_steps = (quota + 3) * base.max_episode_length
step = 0
while bool((ep_counter < quota).any()) and step < max_steps:
    with cert.time_controller(), torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    # d0_per_env is rewritten inside _reset_idx, which runs during step(), so
    # reading it alongside the other episode stats would report the NEXT
    # episode's straight-line distance. Snapshot it while it still belongs to
    # the episode that is about to end.
    d0_prev = (base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None)
    with cert.time_environment():
        obs, _, term, trunc, _ = wrapped.step(actions)
    cert.after_step()
    step += 1
    done = term | trunc
    done = done.squeeze(-1) if done.dim() > 1 else done
    ids = torch.nonzero(done).flatten()
    for i in ids.tolist():
        # Past quota this env keeps running (the others are not done yet) but
        # its extra episodes are NOT recorded.  Dropping them here rather than
        # truncating afterwards is the whole fix: truncation is what let
        # completion order decide the sample.
        if int(ep_counter[i]) >= quota:
            cert.discard_episode(i)
            ep_counter[i] += 1
            continue
        rec = {
            "env": i,
            "ep": int(ep_counter[i]),
            "success": bool(base.episode_success[i]),
            "terminated": bool(term.reshape(-1)[i]),
            "truncated": bool(trunc.reshape(-1)[i]),
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
        # What the mid-episode appearance actually did.  Empty dict on
        # families without the axis, so unrelated certificates are
        # byte-identical.  Without these, a 128/128 on the appearance axis
        # mixes "avoided it" with "the episode ended before it existed",
        # since t0 ~ U[5, 30] s and a champion finishes in 18 to 21 s.
        rec.update(episode_appearance_fields_for(base, i))
        if hasattr(base, "episode_gates_passed"):
            rec["gates"] = int(base.episode_gates_passed[i])
        if d0_prev is not None:
            rec["d0_m"] = float(d0_prev[i])
        # The six fields the model-based evaluators have always written and
        # this one never did (added 2026-08-31).  Their absence was not
        # cosmetic: an assembler defaulting a missing field to zero produced
        # "contact = 0" and "route_geodesic = 0" readings for the whole
        # learned column that were artefacts of the schema, not measurements,
        # and one of them was reported as a finding.  Same expressions and the
        # same guards as scripts/eval_mppi.py, so the two columns are
        # comparable field by field.
        rec["controller"] = "learned"
        episode_contact_steps = getattr(base, "episode_contact_steps", None)
        episode_contact_longest = getattr(
            base, "episode_contact_longest_steps", None
        )
        episode_contact_depth = getattr(base, "episode_contact_depth_sum", None)
        if (
            episode_contact_steps is not None
            and episode_contact_longest is not None
            and episode_contact_depth is not None
        ):
            contact_steps = float(episode_contact_steps[i])
            rec["contact_steps"] = int(contact_steps)
            rec["contact_seconds"] = contact_steps * base.control_step_s
            rec["contact_longest_seconds"] = (
                float(episode_contact_longest[i]) * base.control_step_s
            )
            rec["contact_depth_mean_m"] = (
                float(episode_contact_depth[i]) / contact_steps
                if contact_steps > 0.0
                else 0.0
            )
        if hasattr(base, "route_geodesic_length"):
            geodesic = float(base.route_geodesic_length[i])
            if geodesic > 0.0:
                rec["route_geodesic_m"] = geodesic
        records.append(cert.finish_episode(i, rec))
        ep_counter[i] += 1

# No truncation: the loop above already produced an exact (env, ep) grid.  The
# only way to fall short is the step budget running out, which is a real event
# and is now recorded rather than hidden by slicing.
grid_complete = bool((ep_counter >= quota).all())
if not grid_complete:
    short = int((ep_counter < quota).sum())
    print(
        f"[INCOMPLETE] step budget exhausted with {short} of {int(base.num_envs)} "
        f"envs short of the {quota}-episode quota; the certificate records this "
        "and the cell must not be read as a full sample",
        flush=True,
    )
n = len(records)
succ = [r for r in records if r["success"]]
tts = sorted(r["tts_s"] for r in succ if r["tts_s"] is not None)
clr = sorted(r["min_clearance_m"] for r in records
             if r.get("min_clearance_m") is not None)
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
path = sorted(r["path_length_m"] for r in succ if r.get("path_length_m") is not None) or sorted(
    r["path_length_m"] for r in records if r.get("path_length_m") is not None)
print(f"  path_len median={pct(path, 0.5):.1f}m p90={pct(path, 0.9):.1f}m", end="")
ratios = sorted(r["path_length_m"] / r["d0_m"] for r in succ
                if r.get("d0_m") is not None and r["d0_m"] > 0.0
                and r.get("path_length_m") is not None)
print(f"  detour median={pct(ratios, 0.5):.2f}x straight" if ratios else "")
if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump(cert.build({"task": TASK, "level": args_cli.level,
                   "seed": args_cli.eval_seed,
                   # Sampling frame (added 2026-08-31).  None of these three
                   # existed before, which is exactly why the completion-order
                   # defect survived a 30-cell transcription audit: a reader
                   # could not tell how many envs a cell used, how many
                   # episodes were requested, or whether the run finished.  A
                   # certificate that cannot describe its own sample cannot be
                   # audited.
                   "num_envs": int(base.num_envs),
                   "episodes_requested": int(args_cli.episodes),
                   "episodes_per_env": int(quota),
                   "complete": bool(grid_complete),
                   # The header block when the env carries the protocol
                   # object, the off-protocol literal when it does not --
                   # never unconditionally the header.
                   "scenario_protocol": scenario_protocol,
                   "imbalance": args_cli.imbalance,
                   "checkpoint": os.path.abspath(args_cli.checkpoint),
                   # WHICH entry of "overrides" is the Suite D axis under
                   # test. Not recoverable from the flat dict alone: with
                   # --axis drag_scale --value 20 --set mass_scale=1.1 both
                   # keys are plain cfg fields carrying plain floats, and the
                   # attribution claim ("this cell isolates drag_scale")
                   # depends on knowing which one was swept. The name only --
                   # the value stays in "overrides" and is not duplicated
                   # here, so the two can never drift apart.
                   "axis": pinned_axis,
                   "overrides": applied_overrides,
                   "records": records}), f, indent=1, allow_nan=False)
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
