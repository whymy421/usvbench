"""CPI verdicts over the paired convergence curves.

Implements the run-level core of the CPI criterion on the curve JSONs that
convergence_curve.py produces:

  DEF 1  delta-stabilization: every rung in the terminal window W (last w=9)
         must be TOST-equivalent to the final rung within +/-delta = 0.10 at
         level alpha2 = 0.02, on PAIRED per-layout differences. Pairing uses
         each env's FIRST episode, which faces an identical layout across
         rungs because the sweep re-seeds the layout RNG before every rung.
  DEF 2  regime guard: approximate PELT via binary segmentation with binomial
         segment cost and a BIC-style penalty; any change point strictly
         inside W vetoes stabilization (one-directional: NO only).
  DEF 3  quality labels from Delta_init (first rung as the initialization
         proxy) and Delta_peak (best rung), Newcombe intervals.

This is the run-level verdict only. Configuration-level reproducibility and
the racing ceiling are separate machinery with their own episode budgets.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

DELTA = 0.10
W = 9
ALPHA2 = 0.02
Z = 2.054  # z_{1 - alpha2}


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def newcombe(p1, n1, p2, n2):
    """CI for p1 - p2 (unpaired Newcombe; adequate for labelling)."""
    l1, u1 = wilson(p1, n1)
    l2, u2 = wilson(p2, n2)
    return (p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def first_per_env(rung):
    seen = {}
    for rec in rung["records"]:
        env = rec["env"]
        if env not in seen:
            seen[env] = 1 if rec["success"] else 0
    return seen


def binom_cost(successes, count):
    if count == 0:
        return 0.0
    p = successes / count
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -2 * count * (p * math.log(p) + (1 - p) * math.log(1 - p))


def segment_cost(series, lo, hi):
    s = sum(x for x, _ in series[lo:hi])
    n = sum(m for _, m in series[lo:hi])
    return binom_cost(s, n)


def change_points(series, penalty):
    """Binary segmentation, binomial cost. Returns split indices."""
    splits = []

    def recurse(lo, hi):
        if hi - lo < 4:
            return
        base = segment_cost(series, lo, hi)
        best_gain, best_at = 0.0, None
        for mid in range(lo + 2, hi - 2):
            gain = base - segment_cost(series, lo, mid) - segment_cost(series, mid, hi)
            if gain > best_gain:
                best_gain, best_at = gain, mid
        if best_at is not None and best_gain > penalty:
            splits.append(best_at)
            recurse(lo, best_at)
            recurse(best_at, hi)

    recurse(0, len(series))
    return sorted(splits)


def verdict(path):
    with open(path, encoding="utf-8") as handle:
        curve = json.load(handle)
    rungs = curve["rungs"]
    if len(rungs) < W + 1:
        return {"curve": os.path.basename(path), "verdict": "NOT-CERTIFIABLE",
                "note": f"only {len(rungs)} rungs (< w+1)"}

    srs = [r["sr"] for r in rungs]
    steps = [r["step"] for r in rungs]
    window = rungs[-W:]
    anchor = window[-1]
    anchor_envs = first_per_env(anchor)

    # DEF 1: equivalence of every window rung against the anchor. Paired TOST
    # when enough env-first pairs survive; otherwise an unpaired Newcombe TOST
    # on the full rung proportions. Pairing thins out on contact-terminating
    # families because fast-crashing envs recycle and displace slow envs'
    # first episodes from the 64 collected records.
    pairs_report = []
    stabilized = True
    for rung in window[:-1]:
        envs = first_per_env(rung)
        common = sorted(set(envs) & set(anchor_envs))
        diffs = [envs[e] - anchor_envs[e] for e in common]
        n = len(diffs)
        if n >= 30:
            mean = sum(diffs) / n
            var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
            half = Z * math.sqrt(var / n)
            ok = abs(mean) + half < DELTA
            stabilized &= ok
            pairs_report.append((rung["step"], round(mean, 4), n,
                                 "paired pass" if ok
                                 else f"paired FAIL |{mean:+.3f}|+{half:.3f}>=0.10"))
        else:
            lo, hi = newcombe(rung["sr"], rung["episodes"],
                              anchor["sr"], anchor["episodes"])
            ok = -DELTA < lo and hi < DELTA
            stabilized &= ok
            pairs_report.append((rung["step"], round(rung["sr"] - anchor["sr"], 4), n,
                                 "unpaired pass" if ok
                                 else f"unpaired FAIL CI[{lo:+.3f},{hi:+.3f}]"))

    # DEF 2: regime guard.
    series = [(r["successes"], r["episodes"]) for r in rungs]
    penalty = 3.0 * math.log(len(series))
    cps = change_points(series, penalty)
    veto = any(cp >= len(rungs) - W for cp in cps)
    if veto:
        stabilized = False

    # DEF 3: quality.
    first = rungs[0]
    peak = max(rungs, key=lambda r: r["sr"])
    terminal_sr = sum(r["successes"] for r in window) / sum(r["episodes"] for r in window)
    lo_i, hi_i = newcombe(first["sr"], first["episodes"], anchor["sr"], anchor["episodes"])
    lo_p, hi_p = newcombe(peak["sr"], peak["episodes"], anchor["sr"], anchor["episodes"])

    # delta_init CI is (first - anchor): terminal BELOW init <=> lo_i > 0.
    if stabilized:
        if lo_i > 0:
            label = "stabilized-DESTRUCTIVE (below its initialization)"
        elif lo_p > DELTA:
            label = "stabilized-REGRESSIVE (peak-then-decay)"
        elif hi_i < 0:
            label = "stabilized-improved"
        else:
            label = "stabilized-neutral"
    else:
        tail_trend = srs[-1] - srs[max(0, len(srs) - W)]
        label = ("not-stabilized-rising" if tail_trend > DELTA
                 else "not-stabilized-decaying" if tail_trend < -DELTA
                 else "not-stabilized-oscillating")
        if veto:
            label += " [regime-change veto in window]"

    return {
        "curve": os.path.basename(path).replace("curve_", "").replace(".json", ""),
        "rungs": len(rungs),
        "stabilized": stabilized,
        "label": label,
        "terminal_window_sr": round(terminal_sr, 4),
        "peak": {"step": peak["step"], "sr": round(peak["sr"], 4),
                 "frac_of_budget": round(peak["step"] / steps[-1], 3)},
        "delta_init_ci": [round(lo_i, 3), round(hi_i, 3)],
        "delta_peak_ci": [round(lo_p, 3), round(hi_p, 3)],
        "change_points_steps": [steps[cp] for cp in cps],
        "pairs": pairs_report,
    }


def main(root):
    results = []
    for path in sorted(glob.glob(os.path.join(root, "*", "curve_*.json"))):
        if "smoke" in path:
            continue
        results.append(verdict(path))
    out = os.path.join(root, "cpi_verdicts.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, ensure_ascii=False)
    for r in results:
        if "note" in r:
            print(f"{r['curve']:22s} {r['verdict']}: {r['note']}")
            continue
        print(f"{r['curve']:22s} stab={str(r['stabilized']):5s} {r['label']:44s} "
              f"tail={r['terminal_window_sr']:.3f} peak={r['peak']['sr']:.3f}"
              f"@{r['peak']['frac_of_budget']:.0%} cps={r['change_points_steps']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\BRADY\usvbench\curves")
