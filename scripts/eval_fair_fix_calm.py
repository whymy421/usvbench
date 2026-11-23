"""Calm-water evaluation of a journal PPO checkpoint (diagnostic branch yutong/ppo-fair-fix, 2026-11-12).

Why a separate evaluator: the release evaluators (eval_v6_frozen / run_cell) resolve the pinned D7 pool files of
the evaluation seed, and those bundles are not in git. A calm evaluation needs no D7 object, so this script builds
the env exactly as train_journal_ppo.run_training builds its calm arm (no condition suite, the family's calm D7
adapter = the D24 D0 object on every key) on a layout seed that no compared policy trained on.

Same exam for every policy: the layout of (seed, env, episode) is a pure function of the key (journal contract),
and every env runs a fixed quota of episodes (per-env quota, as eval_v6_frozen), so two checkpoints evaluated with
the same --eval-seed / --num-envs / --episodes-per-env sit the same 64 x k layouts episode by episode.
Action = the policy mean (mean_actions), as eval_v6_frozen. The env runs the release reward (no --fair-fix): the
reward does not enter the evaluation, and the observation is the journal 42-D contract either way.

usage: python -B scripts/eval_fair_fix_calm.py --task journal:fortress --vessel blueboat --checkpoint PATH
           --eval-seed 3001 --out JSON [--num-envs 64] [--episodes-per-env 2] [AppLauncher args, e.g. --headless]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import seed_registry  # noqa: E402
import train_journal_ppo as T  # noqa: E402  (CPU helpers: task_modules, family_pool_module)
from eval_protocol import resolve_journal_task  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--vessel", required=True, choices=("blueboat", "wamv"))
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--eval-seed", type=int, required=True, help="a registry training seed none of the policies used")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes-per-env", type=int, default=2)
parser.add_argument("--out", required=True)
parser.add_argument("--policy", choices=("mlp", "gru"), default="mlp",
                    help="gru = a PPO-GRU checkpoint (scripts/ppo_memory_models.py): the hidden state is carried "
                         "step to step and zeroed at each episode end")
parser.add_argument("--action-dist", choices=("clip", "tanh"), default="clip",
                    help="tanh = a checkpoint trained with the bounded tanh-Gaussian head (scripts/ppo_squashed_models"
                         ".py); the deterministic action is then tanh(mean)")
parser.add_argument("--geometry-tier", type=int, default=None, choices=(0, 1, 2, 3),
                    help="diagnostic (round 8): evaluate at this gap-width tier instead of the journal tier 2")
parser.add_argument("--route-hint", action="store_true",
                    help="round 10: the checkpoint was trained with --route-hint; install the same hook "
                         "(scripts/ppo_route_hint.py) so the policy sees the planner's waypoint bearing")
parser.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many env steps (a quick mean-action probe; the success table is partial)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from eval_protocol import register_journal_tasks  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mean_probe(outside, samples, steps, num_envs, key):
    """|policy mean| on the visited states: fraction above 1 (every step) and quantiles (every 10th step)."""
    if outside is None or not samples:
        return None
    flat = torch.cat(samples, dim=0).float()
    q = torch.tensor([0.5, 0.9, 0.99])
    return {"quantity": "|%s| per action dimension (the env clamps commands to [-1, 1])" % key,
            "steps": int(steps), "envs": int(num_envs),
            "fraction_above_1": [round(float(v) / float(steps * num_envs), 4) for v in outside.tolist()],
            "quantiles_p50_p90_p99": [[round(float(x), 3) for x in torch.quantile(flat[:, d], q).tolist()]
                                      for d in range(flat.shape[1])],
            "max": [round(float(v), 3) for v in flat.max(dim=0).values.tolist()]}


def main():
    info = resolve_journal_task(args.task, args.vessel)
    if info is None:
        raise SystemExit("not a journal task: %r" % args.task)
    registry = seed_registry.load_registry()
    owner = seed_registry.namespace_for_seed(args.eval_seed, registry)
    purpose = seed_registry.purpose_of(owner, registry) if owner else None
    if purpose not in seed_registry.TRAINING_PURPOSES:
        raise SystemExit("--eval-seed must be a registry training seed (the calm adapter refuses others); "
                         "got %r in namespace %r" % (args.eval_seed, owner))
    module = register_journal_tasks()
    root = module.__name__.rsplit(".", 1)[0]
    mods = T.task_modules(root)
    pool_module = T.family_pool_module(root, info["carrier"])
    journal_id = info["task_id"]
    env_cfg = parse_env_cfg(journal_id, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = int(args.eval_seed)
    if args.geometry_tier is not None:                    # round 8: set before the pools read it
        env_cfg.journal_geometry_tier = int(args.geometry_tier)
    resolved = mods.journal_config.resolve_journal_conditions(presets=[], seed=int(args.eval_seed))
    mods.journal_config.admit_journal_conditions(env_cfg, resolved)
    mods.journal_config.apply_journal_conditions(env_cfg, resolved)
    cache = mods.hazards.TrainingHazardCache(prefetch=False)
    pools = pool_module.training_pools(env_cfg, None, cache=cache, purpose=purpose, registry=registry,
                                       num_envs=args.num_envs)
    env_cfg.journal_d7_pool = pools["d7"]
    agent_cfg = load_cfg_from_registry(journal_id, "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent_cfg["agent"]["experiment"]["wandb"] = False
    env = gym.make(journal_id, cfg=env_cfg, render_mode=None)
    hint = None
    if args.route_hint:                                   # round 10: before the first reset
        import ppo_route_hint as RH
        hint = RH.install(env.unwrapped, env_cfg, root, torch)
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    recurrent = args.policy == "gru"
    squash = args.action_dist == "tanh"
    if recurrent:
        import ppo_memory_models as PM
        runner = PM.RecurrentRunner(wrapped, agent_cfg, squash=squash)
    elif squash:
        import ppo_squashed_models as PS
        runner = PS.SquashedRunner(wrapped, agent_cfg)
    else:
        runner = Runner(wrapped, agent_cfg)
    runner.agent.load(os.path.abspath(args.checkpoint))
    runner.agent.set_running_mode("eval")
    agent = runner.agent
    base = env.unwrapped

    quota = int(args.episodes_per_env)
    count = torch.zeros(base.num_envs, dtype=torch.long)
    max_steps = (quota + 3) * int(base.max_episode_length)
    if args.max_steps is not None:
        max_steps = min(max_steps, int(args.max_steps))
    records, step, t0 = [], 0, time.monotonic()
    # mean-action probe on the visited states: the policy's own mean (the pre-tanh latent for the tanh head)
    latent_key = "latent_mean_actions" if squash else "mean_actions"
    outside, abs_samples = None, []
    obs, _ = wrapped.reset()
    while bool((count < quota).any()) and step < max_steps:
        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            latent = outputs[-1].get(latent_key, actions).abs()
            outside = (latent > 1.0).sum(dim=0) if outside is None else outside + (latent > 1.0).sum(dim=0)
            if step % 10 == 0:
                abs_samples.append(latent.detach().to("cpu"))
            if recurrent:      # PPO_RNN advances its state only in record_transition, which this loop does not call
                agent._rnn_initial_states["policy"] = [s.clone() for s in outputs[-1]["rnn"]]
        obs, _, term, trunc, _ = wrapped.step(actions)
        step += 1
        done = (term | trunc).reshape(-1)
        if recurrent and bool(done.any()):
            ended = torch.nonzero(done).flatten()
            with torch.inference_mode():   # the states are inference tensors (created under inference_mode above)
                for state in agent._rnn_initial_states["policy"]:
                    state[:, ended] = 0.0
        for i in torch.nonzero(done).flatten().tolist():
            if int(count[i]) >= quota:
                count[i] += 1
                continue
            completed = base.journal_completed_records[i] or {}
            tts = float(base.time_to_success[i])
            records.append({
                "env": i, "ep": int(count[i]),
                "success": bool(base.episode_success[i]),
                "termination_code": int(base.episode_termination_reason[i]),
                "termination_name": completed.get("termination_name"),
                "scenario_episode": completed.get("scenario_episode"),
                "time_to_success_s": None if math.isnan(tts) else tts,
                "path_length_m": float(base.episode_path_length[i]),
                "min_clearance_m": float(base.episode_min_clearance[i]),
            })
            count[i] += 1
    n = len(records)
    k = sum(r["success"] for r in records)
    outcomes = {}
    for r in records:
        outcomes[r["termination_name"]] = outcomes.get(r["termination_name"], 0) + 1
    times = sorted(r["time_to_success_s"] for r in records if r["time_to_success_s"] is not None)
    summary = {
        "schema": "usvbench.fair_fix_calm_eval/1", "task": info["journal_task"], "journal_id": journal_id,
        "vessel": args.vessel, "checkpoint": os.path.abspath(args.checkpoint), "eval_seed": int(args.eval_seed),
        "eval_seed_namespace": owner, "num_envs": int(args.num_envs), "episodes_per_env": quota,
        "episodes_expected": int(args.num_envs) * quota, "episodes": n, "successes": k,
        "success_rate": (k / n) if n else None, "wilson95": wilson(k, n), "outcomes": outcomes,
        "median_time_to_success_s": times[len(times) // 2] if times else None,
        "steps": step, "wall_s": round(time.monotonic() - t0, 1),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "action_selection": "mean_actions", "conditions": "calm (no suite), D7 = calm-arm D0 adapter",
        "policy": args.policy, "action_dist": args.action_dist,
        "geometry_tier": int(getattr(base.cfg, "journal_geometry_tier", 2)),
        "mean_action_probe": mean_probe(outside, abs_samples, step, base.num_envs, latent_key),
        "route_hint": hint.summary() if hint is not None else None,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("EVAL-DONE %s %s seed %d: %d/%d success (%s) outcomes %s -> %s"
          % (info["journal_task"], args.vessel, args.eval_seed, k, n,
             "%.3f" % (k / n) if n else "n/a", outcomes, args.out), flush=True)


EXIT_CODE = 1
try:
    main()
    EXIT_CODE = 0
except BaseException:  # noqa: BLE001 - report, then exit non-zero
    import traceback
    traceback.print_exc()
finally:
    # The results are on disk before this point. SimulationApp.close() hung after EVAL-DONE in the 2026-11-12
    # CPU smoke (one core busy for minutes), so exit hard instead of closing the app (train_journal_ppo closes
    # under a watchdog for the same reason).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_CODE)
