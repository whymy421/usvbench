"""Deployment-readiness metrics over per-episode gcert JSON records.

Task C of the ICRA-2027 statistics upgrade (2026-08-23). Pure CPU, pure
post-processing: reads certification JSONs that scripts/eval_v6_frozen.py or
scripts/classical_baseline.py already wrote, never launches Isaac, never
touches the registry.

Input records (one JSON per arm = one task x one controller x one eval seed):
    {"task": ..., "seed": ..., "records": [{"env", "ep", "success",
     "tts_s", "path_length_m", "min_clearance_m"?, "d0_m"?,
     "route_geodesic_m"?}, ...]}
Fields marked ? are present only on suites that record them; every consumer
below degrades to blank output when a field is absent, never crashes.

Frozen metric definitions (pre-registered; do not edit without a prereg bump)
----------------------------------------------------------------------------
Success rate: SR = n_success / n with a Wilson 95% score interval,
    z = 1.959963984540054.

Per-episode deployment LOSS (lower is better), for episode i with success
indicator s_i in {0,1} and time-to-success tts_i (seconds, recorded only when
s_i = 1):

    L_i = (1 - s_i) * 1.0  +  s_i * min(tts_i / T_budget, 1.0)

  * A failure costs the maximum loss 1.0 regardless of failure mode
    (collision and timeout are deliberately equal-loss in v1; disclosed).
  * A success costs its normalised time-to-success in [0, 1].
  * T_budget is the FROZEN per-task episode budget (episode_length_s from the
    task cfg), not any data-dependent quantity:
        default                                   120.0 s
        Isaac-USV-PathHazard-Direct-v1            150.0 s
        Isaac-USV-HarborMissionKin-Direct-v1      240.0 s
        Isaac-USV-HarborStage1-Direct-v1           60.0 s
        Isaac-USV-HarborStage2-Direct-v1          150.0 s
        Isaac-USV-HarborDockPhase-Direct-v1        60.0 s
    Unknown task ids fall back to the default 120.0 with a stderr warning;
    --budget TASK=SECONDS overrides (repeatable).

CVaR_alpha (expected loss in the worst alpha-tail), alpha in {0.25, 0.10}:
    sort L descending, k = ceil(alpha * n), CVaR_alpha = mean of the k
    largest losses. Discrete upper-tail convention, no fractional weighting.

Bootstrap CI on CVaR: B = 1000 resamples of the n episodes with replacement,
    percentile interval [2.5, 97.5]. RNG is numpy default_rng seeded with
    (base_seed + crc32(arm_name)) mod 2**32, base_seed default 20260823, so
    every arm is reproducible independently of file ordering.

Near-miss rate: among SUCCESS episodes that carry min_clearance_m, the share
    with min_clearance_m < 0.20 m (the C_SOFT margin of usv10k_score.py).
    Successes without the field are excluded and counted.

Worst-episode block: max loss and its env/ep id, slowest success (max tts_s),
    global min clearance (any episode, failures included), failure count.

Output: one tidy CSV (one row per arm) and one Markdown table.

Usage:
    python scripts/deployment_metrics.py gcerts_pid gcerts_cache \\
        --out-csv outputs/taskC_stats/deployment_metrics.csv \\
        --out-md  outputs/taskC_stats/deployment_metrics.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import zlib

import numpy as np

Z95 = 1.959963984540054
DEFAULT_ALPHAS = (0.25, 0.10)
DEFAULT_BOOT = 1000
DEFAULT_BOOT_SEED = 20260823
NEAR_MISS_M = 0.20

EPISODE_BUDGET_S = {
    "Isaac-USV-PathHazard-Direct-v1": 150.0,
    "Isaac-USV-HarborMissionKin-Direct-v1": 240.0,
    "Isaac-USV-HarborStage1-Direct-v1": 60.0,
    "Isaac-USV-HarborStage2-Direct-v1": 150.0,
    "Isaac-USV-HarborDockPhase-Direct-v1": 60.0,
}
DEFAULT_BUDGET_S = 120.0


def wilson_ci(k, n, z=Z95):
    """Wilson score 95% interval for a binomial proportion. Returns (lo, hi)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def episode_loss(rec, t_budget, warn_sink=None):
    """The frozen deployment loss L_i of one record (see module docstring)."""
    if rec.get("success"):
        tts = rec.get("tts_s")
        if tts is None:
            if warn_sink is not None:
                warn_sink.append("success with tts_s missing -> loss 1.0")
            return 1.0
        return min(float(tts) / float(t_budget), 1.0)
    return 1.0


def cvar(losses, alpha):
    """Discrete upper-tail CVaR: mean of the ceil(alpha*n) largest losses."""
    arr = np.sort(np.asarray(losses, dtype=float))[::-1]
    n = arr.size
    if n == 0:
        return float("nan")
    k = int(math.ceil(alpha * n))
    k = max(1, min(k, n))
    return float(arr[:k].mean())


def arm_seed(arm_name, base_seed):
    return (int(base_seed) + zlib.crc32(arm_name.encode("utf-8"))) % (2 ** 32)


def bootstrap_cvar_ci(losses, alpha, n_boot, seed):
    """Percentile bootstrap CI [2.5, 97.5] on the CVaR of ``losses``."""
    arr = np.asarray(losses, dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.empty(n_boot, dtype=float)
    k = max(1, min(int(math.ceil(alpha * n)), n))
    for b in range(n_boot):
        sample = arr[idx[b]]
        # partition is O(n) and equivalent to sorting for a tail mean
        part = np.partition(sample, n - k)[n - k:]
        stats[b] = part.mean()
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def load_arm(path):
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
        raise ValueError("%s: not a gcert-style JSON (no records list)" % path)
    return doc


def analyse_arm(path, alphas, n_boot, base_seed, budget_overrides):
    doc = load_arm(path)
    recs = doc["records"]
    arm = os.path.splitext(os.path.basename(path))[0]
    task = doc.get("task", "?")
    t_budget = budget_overrides.get(
        task, EPISODE_BUDGET_S.get(task, DEFAULT_BUDGET_S))
    if task not in EPISODE_BUDGET_S and task not in budget_overrides:
        # the 120 s default is correct for every remaining family;
        # surfaced so a genuinely unknown task id cannot pass silently
        sys.stderr.write("[budget] %s -> default %.0f s (%s)\n"
                         % (task, DEFAULT_BUDGET_S, arm))

    warns = []
    losses = [episode_loss(r, t_budget, warns) for r in recs]
    n = len(recs)
    n_succ = sum(1 for r in recs if r.get("success"))
    sr = n_succ / n if n else float("nan")
    sr_lo, sr_hi = wilson_ci(n_succ, n)

    clipped = sum(1 for r in recs
                  if r.get("success") and r.get("tts_s") is not None
                  and float(r["tts_s"]) > t_budget)
    if clipped:
        warns.append("%d success tts_s above budget clipped to 1.0" % clipped)

    row = {
        "arm": arm,
        "file": path.replace("\\", "/"),
        "task": task,
        "controller": doc.get("controller")
        or ("policy:" + os.path.basename(str(doc.get("checkpoint")))
            if doc.get("checkpoint") else "?"),
        "eval_seed": doc.get("seed", "?"),
        "n": n,
        "n_success": n_succ,
        "sr": round(sr, 6),
        "sr_lo95": round(sr_lo, 6),
        "sr_hi95": round(sr_hi, 6),
        "t_budget_s": t_budget,
    }

    seed = arm_seed(arm, base_seed)
    for alpha in alphas:
        tag = ("%g" % (alpha * 100)).replace(".", "p")
        lo, hi = bootstrap_cvar_ci(losses, alpha, n_boot, seed)
        row["cvar%s" % tag] = round(cvar(losses, alpha), 6)
        row["cvar%s_lo95" % tag] = round(lo, 6)
        row["cvar%s_hi95" % tag] = round(hi, 6)

    succ_clear = [float(r["min_clearance_m"]) for r in recs
                  if r.get("success") and r.get("min_clearance_m") is not None]
    if succ_clear:
        nm = sum(1 for c in succ_clear if c < NEAR_MISS_M)
        row["near_miss_rate"] = round(nm / len(succ_clear), 6)
        row["near_miss_n"] = "%d/%d" % (nm, len(succ_clear))
    else:
        row["near_miss_rate"] = ""
        row["near_miss_n"] = ""

    worst_i = max(range(n), key=lambda i: (losses[i],
                                           -recs[i].get("env", 0))) if n else -1
    row["worst_loss"] = round(max(losses), 6) if n else ""
    row["worst_ep"] = ("env%s/ep%s" % (recs[worst_i].get("env", "?"),
                                       recs[worst_i].get("ep", "?"))
                       if n else "")
    succ_tts = [float(r["tts_s"]) for r in recs
                if r.get("success") and r.get("tts_s") is not None]
    row["slowest_success_s"] = round(max(succ_tts), 2) if succ_tts else ""
    all_clear = [float(r["min_clearance_m"]) for r in recs
                 if r.get("min_clearance_m") is not None]
    row["min_clearance_m"] = round(min(all_clear), 4) if all_clear else ""
    row["n_fail"] = n - n_succ
    row["warnings"] = "; ".join(sorted(set(warns)))
    return row


def collect_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.json"))))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            matched = sorted(glob.glob(item))
            if not matched:
                raise FileNotFoundError(item)
            paths.extend(matched)
    seen, out = set(), []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


MD_COLS = ["arm", "n", "sr", "sr_lo95", "sr_hi95", "cvar25", "cvar25_lo95",
           "cvar25_hi95", "cvar10", "cvar10_lo95", "cvar10_hi95",
           "near_miss_rate", "worst_loss", "n_fail"]


def write_outputs(rows, out_csv, out_md, cmdline):
    if not rows:
        raise SystemExit("no arms analysed")
    cols = list(rows[0].keys())
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    if out_md:
        os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
        with open(out_md, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# Deployment metrics (SR + Wilson95, CVaR + bootstrap95,"
                     " near-miss, worst episode)\n\n")
            fh.write("Command: `%s`\n\n" % cmdline)
            fh.write("Loss: L_i = 1 for failure; tts_i/T_budget (clipped at 1)"
                     " for success. CVaR_a = mean of ceil(a*n) largest losses."
                     " Near-miss: min_clearance < %.2f m among successes.\n\n"
                     % NEAR_MISS_M)
            cols_p = [c for c in MD_COLS if c in cols]
            fh.write("| " + " | ".join(cols_p) + " |\n")
            fh.write("|" + "|".join("---" for _ in cols_p) + "|\n")
            for r in rows:
                fh.write("| " + " | ".join(str(r.get(c, "")) for c in cols_p)
                         + " |\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+",
                    help="gcert JSON files, directories, or globs")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=list(DEFAULT_ALPHAS))
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    ap.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    ap.add_argument("--budget", action="append", default=[],
                    metavar="TASK=SECONDS",
                    help="override the frozen per-task episode budget")
    args = ap.parse_args(argv)

    overrides = {}
    for spec in args.budget:
        task, _, secs = spec.partition("=")
        overrides[task] = float(secs)

    rows = []
    for path in collect_paths(args.inputs):
        try:
            rows.append(analyse_arm(path, tuple(args.alphas), args.boot,
                                    args.boot_seed, overrides))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            sys.stderr.write("[skip] %s: %s\n" % (path, exc))

    cmdline = "python scripts/deployment_metrics.py " + " ".join(
        argv if argv is not None else sys.argv[1:])
    write_outputs(rows, args.out_csv, args.out_md, cmdline)

    cols_p = [c for c in MD_COLS if rows and c in rows[0]]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
              for c in cols_p}
    print("  ".join(c.ljust(widths[c]) for c in cols_p))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols_p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
