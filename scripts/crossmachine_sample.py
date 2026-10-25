"""Stratified cross-machine calibration: plan the sample, then compare machines.

Owner decision (2026-09-30): cross-machine calibration uses a stratified
sample, not the full table. The plan crosses both vessels x three controllers
(PPO, MPPI, classical) x short/long task x low/high dose = 24 fixed-policy
cells, each ONE operational batch (default 128 episodes = 64 envs x 2), run
on each machine with the same calibration seed block. PPO uses one ADMITTED
training seed per (task, vessel), declared in the manifest before any
calibration outcome is seen.

Build input (JSON):
  {"gate_ledger": "ledger.json",
   "tasks": [{"vessel": "blueboat", "duration": "short", "task": "REGISTERED-ID",
              "deadline_s": 120, "level": 0, "checkpoint": "policy.pt",
              "training_seed": 42, "overrides": {},
              "doses": {"low": {"sea_state.hs_range": [0.075, 0.075]},
                        "high": {"sea_state.hs_range": [0.3, 0.3]}}}, ...]}
Exactly one task per (blueboat, wamv) x (short, long). IDs, checkpoints and
dose literals are supplied by the study owner; this module never invents
WAM-V registrations, assets or doses.

  crossmachine_sample.py build manifest.json --machine boxA --machine boxB --out plan.json
  crossmachine_sample.py compare a.json b.json
  crossmachine_sample.py compare --pairs pairs.json --out report.json

Pairs JSON: {"cells": [{"cell_id": "...", "a": "a.json", "b": "b.json"}]}.

Report per cell: success counts and Wilson intervals on each machine, the
paired 2x2 table, per-episode agreement, the paired success difference with a
Tango score interval (independent scenario pairs) or a cluster bootstrap
(--cluster-field), exact McNemar p, and paired continuous differences
(bootstrap). A pooled block summarises all cells; its interval resamples
whole cells because cells share scenario keys. Nothing here claims
equivalence: agreement statistics are descriptive.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from fixed_grid import episode_quota
from paired_intervals import METHOD as PAIRED_METHOD, paired_rate_interval, wilson_interval
from run_cell import assignment_value, parser_for_cell, read_json, resolve_cell
import seed_registry as registry_module

VESSELS = ("blueboat", "wamv")
CONTROLLERS = ("ppo", "mppi", "classical")
DURATIONS = ("short", "long")
DOSES = ("low", "high")
HEADER_MUST_MATCH = ("task_id", "vessel", "controller", "overrides", "suite_presets",
                     "seed_namespace", "evaluation_protocol_version", "scientific_environment",
                     "speed_reference", "resolved_deadline_s", "level", "eval_seed",
                     "num_envs", "episodes_per_env")


def build_sample(manifest, machines, episodes=128, num_envs=64,
                 seed_block="calibration-01", seed_registry=None):
    """Build the fully crossed 24-cell plan; every command is run_cell-resolved."""
    episode_quota(episodes, num_envs)
    registry = registry_module.load_registry(seed_registry)
    stamp = registry_module.resolve_seed_block(seed_block, "crossmachine_calibration",
                                               "calibration", registry)
    if len(machines) != 2 or len(set(machines)) != 2 or any(not item for item in machines):
        raise ValueError("exactly two distinct machine names are required")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), list):
        raise ValueError("manifest requires a tasks list")
    ledger = manifest.get("gate_ledger")
    if not isinstance(ledger, str) or not ledger:
        raise ValueError("manifest requires gate_ledger (PPO calibration uses admitted seeds)")
    entries = {}
    for task in manifest["tasks"]:
        if not isinstance(task, dict):
            raise ValueError("each task manifest entry must be an object")
        key = task.get("vessel"), task.get("duration")
        if key[0] not in VESSELS or key[1] not in DURATIONS or key in entries:
            raise ValueError("task strata must be unique vessel x short/long pairs")
        if not isinstance(task.get("task"), str) or not task["task"]:
            raise ValueError("every stratum requires an explicit registered task ID")
        deadline = task.get("deadline_s")
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("every stratum requires a positive finite deadline_s")
        if type(task.get("level")) is not int or task["level"] < 0:
            raise ValueError("every stratum requires a nonnegative integer level")
        if not isinstance(task.get("checkpoint"), str) or not task["checkpoint"]:
            raise ValueError("every stratum requires an explicit PPO checkpoint")
        if type(task.get("training_seed")) is not int:
            raise ValueError("every stratum requires the admitted PPO training_seed")
        doses = task.get("doses")
        if not isinstance(doses, dict) or set(doses) != set(DOSES):
            raise ValueError("every task requires explicit low and high dose overrides")
        if any(not isinstance(doses[name], dict) or not doses[name] for name in DOSES):
            raise ValueError("dose overrides must be nonempty objects")
        if doses["low"] == doses["high"]:
            raise ValueError("low and high doses must differ")
        overrides = task.get("overrides", {})
        if not isinstance(overrides, dict) or set(overrides) & set(doses["low"]) \
                or set(overrides) & set(doses["high"]):
            raise ValueError("task overrides must be an object disjoint from the dose fields")
        entries[key] = task
    if set(entries) != {(vessel, duration) for vessel in VESSELS for duration in DURATIONS}:
        raise ValueError("manifest must cover both vessels and both task durations")
    for vessel in VESSELS:
        short, long = entries[vessel, "short"], entries[vessel, "long"]
        if short["deadline_s"] >= long["deadline_s"] or short["task"] == long["task"]:
            raise ValueError("short/long strata require distinct tasks and ordered deadlines")
    cells = []
    parser = parser_for_cell()
    for vessel in VESSELS:
        for controller in CONTROLLERS:
            for duration in DURATIONS:
                task = entries[vessel, duration]
                for dose in DOSES:
                    cell_id = "/".join((vessel, controller, duration, dose))
                    commands = {}
                    for machine in machines:
                        argv = ["--task", task["task"], "--vessel", vessel,
                                "--controller", controller, "--level", str(task["level"]),
                                "--episodes", str(episodes), "--num-envs", str(num_envs),
                                "--eval-seed-block", stamp["block_id"],
                                "--seed-namespace", stamp["namespace"],
                                "--run-purpose", "calibration", "--out",
                                str(Path(machine) / (cell_id.replace("/", "_") + ".json"))]
                        if seed_registry is not None:
                            argv.extend(("--seed-registry", str(seed_registry)))
                        if controller == "ppo":
                            argv.extend(("--checkpoint", task["checkpoint"],
                                         "--training-seed", str(task["training_seed"]),
                                         "--gate-ledger", ledger))
                        combined = dict(task.get("overrides", {}))
                        combined.update(task["doses"][dose])
                        for field, value in sorted(combined.items()):
                            argv.extend(("--set", field + "=" + assignment_value(value)))
                        for name in task.get("allow_env", []):
                            argv.extend(("--allow-env", name))
                        resolved = resolve_cell(parser.parse_args(argv))
                        commands[machine] = {"run_cell_argv": argv,
                                             "command": resolved["command"],
                                             "cell_config_sha256": resolved["cell_config_sha256"]}
                    cells.append({"cell_id": cell_id, "task_id": task["task"],
                                  "vessel": vessel, "controller": controller,
                                  "duration": duration, "declared_deadline_s": task["deadline_s"],
                                  "dose": dose, "commands": commands})
    return {"format_version": 2, "purpose": "crossmachine_calibration",
            "machines": list(machines), "batches_per_machine": len(cells),
            "episodes_per_batch": episodes, "seed_registration": stamp,
            "strata": {"vessels": list(VESSELS), "controllers": list(CONTROLLERS),
                       "durations": list(DURATIONS), "doses": list(DOSES)},
            "deferred_runtime_checks": "task registration, vessel identity and cfg types are "
                                       "checked at launch by each evaluator",
            "cells": cells}


def _percentile_interval(estimates):
    estimates = sorted(estimates)
    k = len(estimates)
    low, high = estimates[int(0.025 * k)], estimates[max(0, int(0.975 * k) - 1)]
    # A degenerate empirical distribution cannot establish precision.
    return None if low == high else [low, high]


def _cluster_bootstrap_mean(values, cluster_ids, iterations=10000, seed=31001):
    groups = {}
    for value, cluster in zip(values, cluster_ids):
        groups.setdefault(str(cluster), []).append(value)
    if len(groups) < 2:
        return None
    groups = list(groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [rng.choice(groups) for _ in groups]
        estimates.append(sum(sum(group) for group in sample) / sum(map(len, sample)))
    return _percentile_interval(estimates)


def _resolved_differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        out = []
        for key in keys:
            if key not in a or key not in b:
                out.append(prefix + key)
            else:
                out.extend(_resolved_differences(a[key], b[key], prefix + key + "."))
        return out
    return [] if a == b else [prefix.rstrip(".") or "<root>"]


def compare_certificates(a, b, cluster_field=None):
    """Validate complete pairs first; only the machine identity may differ."""
    from analysis_input import validate_analysis_input, validate_pair

    for document in (a, b):
        validate_analysis_input(document, allowed_purposes=("calibration",))
    pairs = validate_pair(a, b)
    ha, hb = a["header"], b["header"]
    for field in HEADER_MUST_MATCH:
        left, right = ha.get(field), hb.get(field)
        if field == "controller":
            left, right = left.get("configuration"), right.get("configuration")
        if left != right:
            raise ValueError("cross-machine cell mismatch: " + field)
    if ha["checkpoint"]["sha256"] != hb["checkpoint"]["sha256"]:
        raise ValueError("cross-machine cell mismatch: checkpoint SHA-256")
    differing = _resolved_differences(ha["resolved_config"], hb["resolved_config"])
    if differing:
        raise ValueError("cross-machine cell mismatch: resolved_config differs at "
                         + ", ".join(differing[:12]))
    hostname_a = ha.get("runtime", {}).get("hostname")
    hostname_b = hb.get("runtime", {}).get("hostname")
    if not hostname_a or not hostname_b or hostname_a == hostname_b:
        raise ValueError("comparison requires certificates identifying two different machines")
    n = len(pairs)
    outcomes = {(False, False): 0, (False, True): 0, (True, False): 0, (True, True): 0}
    episode_agreement = []
    for ra, rb in pairs:
        if ra.get("scenario_hashes") != rb.get("scenario_hashes"):
            raise ValueError("cross-machine calibration requires equality of every scenario primitive")
        outcomes[ra["success"], rb["success"]] += 1
        episode_agreement.append({"env": ra["env"], "ep": ra["ep"],
                                  "success_a": ra["success"], "success_b": rb["success"],
                                  "reason_a": ra["termination_reason"],
                                  "reason_b": rb["termination_reason"],
                                  "agree": ra["success"] == rb["success"]})
    b_only, a_only = outcomes[False, True], outcomes[True, False]
    successes_a = outcomes[True, True] + a_only
    successes_b = outcomes[True, True] + b_only
    clusters = [(ra.get(cluster_field) if cluster_field else index)
                for index, (ra, _) in enumerate(pairs)]
    if cluster_field and any(cluster is None for cluster in clusters):
        raise ValueError("every episode requires the declared cluster field")
    if cluster_field and any(ra.get(cluster_field) != rb.get(cluster_field) for ra, rb in pairs):
        raise ValueError("paired cluster labels differ")
    rate = paired_rate_interval([(rb["success"], ra["success"]) for ra, rb in pairs])
    difference_interval, method = rate["interval"], PAIRED_METHOD
    if cluster_field:
        difference_interval = _cluster_bootstrap_mean(
            [int(rb["success"]) - int(ra["success"]) for ra, rb in pairs], clusters)
        method = "paired cluster percentile bootstrap; 10000 draws; null if degenerate"
    continuous = {}
    for field in ("path_length_success_prefix_m", "path_length_full_episode_m", "tts_s",
                  "min_clearance_m", "max_hold_s", "xte_rms_m", "episode_elapsed_s"):
        measured = [(index, rb.get(field) - ra.get(field)) for index, (ra, rb) in enumerate(pairs)
                    if ra.get(field) is not None and rb.get(field) is not None]
        values = [value for _, value in measured]
        continuous[field] = {
            "available_pairs": len(values), "total_pairs": n,
            "mean_b_minus_a": statistics.mean(values) if values else None,
            "interval_95": (_cluster_bootstrap_mean(values, [clusters[i] for i, _ in measured])
                            if values else None),
            "method": "paired percentile bootstrap over "
                      + ("clusters" if cluster_field else "episodes")
                      + "; 10000 draws; null if unavailable or degenerate",
            "population": "pairs measured on both machines"}
    reasons_agree = sum(1 for row in episode_agreement if row["reason_a"] == row["reason_b"])
    latency = {name: header.get("controller_latency") for name, header in (("a", ha), ("b", hb))}
    return {"machine_a": hostname_a, "machine_b": hostname_b, "episodes": n,
            "gpu_a": ha["runtime"].get("gpu_name"), "gpu_b": hb["runtime"].get("gpu_name"),
            "successes_a": successes_a, "successes_b": successes_b,
            "success_rate_a_wilson_95": None if cluster_field else wilson_interval(successes_a, n),
            "success_rate_b_wilson_95": None if cluster_field else wilson_interval(successes_b, n),
            "agreement_count": n - b_only - a_only, "agreement_fraction": (n - b_only - a_only) / n,
            "termination_reason_agreement": reasons_agree,
            "outcome_table": {"neither": outcomes[False, False], "b_only": b_only,
                              "a_only": a_only, "both": outcomes[True, True]},
            "success_difference_b_minus_a": (b_only - a_only) / n,
            "success_difference_interval_95": difference_interval,
            "difference_interval_method": method,
            "mcnemar_exact_p": rate["mcnemar_exact_p"],
            "independent_clusters": len(set(map(str, clusters))), "cluster_field": cluster_field,
            "controller_latency": latency,
            "episode_agreement": episode_agreement, "continuous_differences": continuous,
            "equivalence_claim": False}


def pooled_summary(reports, iterations=10000, seed=31002):
    """Across cells: totals, and a cell-resampling interval for the pooled difference."""
    if not reports:
        return None
    n = sum(r["episodes"] for r in reports)
    a_only = sum(r["outcome_table"]["a_only"] for r in reports)
    b_only = sum(r["outcome_table"]["b_only"] for r in reports)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [rng.choice(reports) for _ in reports]
        total = sum(r["episodes"] for r in sample)
        estimates.append(sum(r["outcome_table"]["b_only"] - r["outcome_table"]["a_only"]
                             for r in sample) / total)
    return {"cells": len(reports), "episodes": n, "agreement_fraction": (n - a_only - b_only) / n,
            "a_only": a_only, "b_only": b_only,
            "success_difference_b_minus_a": (b_only - a_only) / n,
            "interval_95": _percentile_interval(estimates) if len(reports) > 1 else None,
            "method": "percentile bootstrap over whole cells (cells share scenario keys); "
                      "descriptive, not an equivalence test"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("manifest", type=Path)
    build.add_argument("--machine", action="append", required=True)
    build.add_argument("--episodes", type=int, default=128)
    build.add_argument("--num-envs", type=int, default=64)
    build.add_argument("--seed-block", default="calibration-01")
    build.add_argument("--seed-registry", type=Path)
    build.add_argument("--out", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("a", nargs="?", type=Path)
    compare.add_argument("b", nargs="?", type=Path)
    compare.add_argument("--pairs", type=Path)
    compare.add_argument("--cluster-field")
    compare.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "build":
            result = build_sample(read_json(args.manifest), args.machine, args.episodes,
                                  args.num_envs, args.seed_block, args.seed_registry)
        else:
            from analysis_input import load_analysis_input
            if args.pairs:
                if args.a or args.b:
                    raise ValueError("use either positional certificates or --pairs")
                cells = read_json(args.pairs).get("cells")
                if not isinstance(cells, list) or not cells:
                    raise ValueError("pairs manifest requires a nonempty cells list")
            else:
                if not args.a or not args.b:
                    raise ValueError("provide two certificates or --pairs")
                cells = [{"cell_id": "single", "a": args.a, "b": args.b}]
            reports, seen = [], set()
            for cell in cells:
                if cell["cell_id"] in seen:
                    raise ValueError("duplicate cell in pairs manifest")
                seen.add(cell["cell_id"])
                report = compare_certificates(load_analysis_input(cell["a"]),
                                              load_analysis_input(cell["b"]), args.cluster_field)
                reports.append({"cell_id": cell["cell_id"], **report})
            result = {"format_version": 2, "cells": reports, "pooled": pooled_summary(reports)}
        output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            if args.out.exists():
                raise ValueError("refusing to overwrite " + str(args.out))
            args.out.write_text(output, encoding="ascii")
        else:
            print(output, end="")
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
