"""Calm-water evaluation of the journal classical planner, with and without its ray memory
(diagnostic branch yutong/ppo-fair-fix, 2026-11-12).

Same env, same layouts and same records as scripts/eval_fair_fix_calm.py (calm, the calm-arm D7 adapter, a layout
seed no compared policy trained on, a fixed per-env quota), so a PPO checkpoint and a planner sit the same exam
episode by episode.

  --controller memory   JournalMemoryPlannerController = planner_los_pid_mem_v2, the campaign's classical
                        controller (current + remembered ray returns, the circumscribed-radius ladder, hold/escape)
  --controller scan     JournalPlannerLOSPIDController = the v2.5 journal planner the campaign used before G22/G25:
                        the CURRENT ray scan only (no memory), the capsule-radius ladder, straight-leg fallback.
                        The information-matched reference for a memoryless policy.
Both read only the D5 estimate (position, rays) and the heading, with the frozen per-vessel gains
(scripts/classical_gains_v2.json) and the evaluator's default planner settings scaled by the vessel length.

usage: python -B scripts/eval_classical_calm.py --task journal:fortress --vessel blueboat --controller memory
           --eval-seed 3001 --out JSON [--num-envs 64] [--episodes-per-env 2] [AppLauncher args]
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
import train_journal_ppo as T  # noqa: E402
from eval_protocol import resolve_journal_task  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--vessel", required=True, choices=("blueboat", "wamv"))
parser.add_argument("--controller", required=True, choices=("memory", "scan"))
parser.add_argument("--eval-seed", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes-per-env", type=int, default=2)
parser.add_argument("--out", required=True)
# classical_planner_pid CLI defaults (reference lengths, scaled by the vessel length like its journal branch)
REPLAN_EVERY, WAYPOINT_SPACING_REF, ACCEPT_RADIUS_REF, PLAN_CELL_REF = 60, 2.0, 1.0, 0.25
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from eval_protocol import register_journal_tasks  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    import classical_planner_pid as C
    import journal_controllers as jc
    info = resolve_journal_task(args.task, args.vessel)
    if info is None or info["carrier"] not in ("hazard_nav", "path_hazard"):
        raise SystemExit("journal hazard_nav / path_hazard tasks only (got %r)" % args.task)
    registry = seed_registry.load_registry()
    owner = seed_registry.namespace_for_seed(args.eval_seed, registry)
    purpose = seed_registry.purpose_of(owner, registry) if owner else None
    if purpose not in seed_registry.TRAINING_PURPOSES:
        raise SystemExit("--eval-seed must be a registry training seed (the calm adapter refuses others)")
    module = register_journal_tasks()
    root = module.__name__.rsplit(".", 1)[0]
    # classical_planner_pid._geometry tries isaaclab_tasks.direct.<family> FIRST. On this machine that is the NavRL
    # research tree's own hazard_nav (a different hazard_geometry), not the release copy the journal env samples its
    # layouts from. Pin the planner to the env's module before any plan is made.
    import importlib
    for family, geometry in (("hazard_nav", "hazard_geometry"), ("path_hazard", "path_hazard_geometry")):
        C._GEOMETRY_CACHE[family] = importlib.import_module("%s.%s.%s" % (root, family, geometry))
    print("[eval_classical_calm] planner geometry: %s" % C._GEOMETRY_CACHE["hazard_nav"].__file__, flush=True)
    mods = T.task_modules(root)
    pool_module = T.family_pool_module(root, info["carrier"])
    journal_id = info["task_id"]
    env_cfg = parse_env_cfg(journal_id, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = int(args.eval_seed)
    resolved = mods.journal_config.resolve_journal_conditions(presets=[], seed=int(args.eval_seed))
    mods.journal_config.admit_journal_conditions(env_cfg, resolved)
    mods.journal_config.apply_journal_conditions(env_cfg, resolved)
    cache = mods.hazards.TrainingHazardCache(prefetch=False)
    pools = pool_module.training_pools(env_cfg, None, cache=cache, purpose=purpose, registry=registry,
                                       num_envs=args.num_envs)
    env_cfg.journal_d7_pool = pools["d7"]
    env = gym.make(journal_id, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    vessel = str(env_cfg.vehicle)
    scale = jc.length_scale(vessel)
    gains = jc.load_gains(vessel)
    settings = dict(replan_every=REPLAN_EVERY, waypoint_spacing_m=WAYPOINT_SPACING_REF * scale,
                    accept_radius_m=ACCEPT_RADIUS_REF * scale, plan_cell_m=PLAN_CELL_REF * scale,
                    gains=gains, max_range_m=jc.ray_max_range(env_cfg))
    if args.controller == "memory":
        ladder = C.journal_memory_ladder(vessel)
        controller = C.JournalMemoryPlannerController(base, ladder=ladder, vessel=vessel, **settings)
        name = "planner_los_pid_mem_v2 (ray memory)"
    else:
        ladder = C.journal_ladder(vessel)
        controller = C.JournalPlannerLOSPIDController(base, ladder=ladder, **settings)
        name = "journal planner_los_pid v2.5 (current scan only)"

    quota = int(args.episodes_per_env)
    count = torch.zeros(base.num_envs, dtype=torch.long)
    max_steps = (quota + 3) * int(base.max_episode_length)
    records, step, t0 = [], 0, time.monotonic()
    env.reset()
    while bool((count < quota).any()) and step < max_steps:
        with torch.inference_mode():
            actions = controller.act()
        _obs, _rew, term, trunc, _info = env.step(actions)
        step += 1
        done = (term | trunc).reshape(-1)
        controller.reset(done)
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
    telemetry = controller.telemetry() if hasattr(controller, "telemetry") else {
        "straight_fallbacks": int(getattr(controller, "straight_fallbacks", 0)),
        "plan_calls": int(getattr(controller, "plan_calls", 0))}
    summary = {
        "schema": "usvbench.fair_fix_calm_eval/1", "task": info["journal_task"], "journal_id": journal_id,
        "vessel": args.vessel, "controller": args.controller, "controller_name": name,
        "settings": {k_: (list(v) if isinstance(v, tuple) else v) for k_, v in settings.items() if k_ != "gains"},
        "gains": gains, "inflation_ladder_m": list(ladder), "eval_seed": int(args.eval_seed),
        "eval_seed_namespace": owner, "num_envs": int(args.num_envs), "episodes_per_env": quota,
        "episodes_expected": int(args.num_envs) * quota, "episodes": n, "successes": k,
        "success_rate": (k / n) if n else None, "wilson95": wilson(k, n), "outcomes": outcomes,
        "median_time_to_success_s": times[len(times) // 2] if times else None,
        "steps": step, "wall_s": round(time.monotonic() - t0, 1),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "conditions": "calm (no suite), D7 = calm-arm D0 adapter",
        "telemetry": {k_: v for k_, v in telemetry.items() if k_ not in ("episode_rows", "unfinished_rows")},
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("EVAL-DONE classical-%s %s %s seed %d: %d/%d success (%s) outcomes %s -> %s"
          % (args.controller, info["journal_task"], args.vessel, args.eval_seed, k, n,
             "%.3f" % (k / n) if n else "n/a", outcomes, args.out), flush=True)


EXIT_CODE = 1
try:
    main()
    EXIT_CODE = 0
except BaseException:  # noqa: BLE001
    import traceback
    traceback.print_exc()
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_CODE)      # results are on disk; SimulationApp.close() can hang (evaluator_exit history)
