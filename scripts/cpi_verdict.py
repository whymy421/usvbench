"""CPI verdicts over the paired convergence curves.

Implements the run-level core of the CPI criterion on the curve JSONs that
convergence_curve.py produces:

  DEF 1  delta-stabilization: every rung in the terminal window W (last w=9)
         must be TOST-equivalent to the final rung within +/-delta = 0.10 at
         level alpha2 = 0.02, on PAIRED per-layout differences. Pairing uses
         the complete predeclared (env, ep) grid, with every unchanged
         scenario primitive checked across rungs before analysis.
  DEF 2  regime guard: approximate PELT via binary segmentation with binomial
         segment cost and a BIC-style penalty; any change point strictly
         inside W vetoes stabilization (one-directional: NO only).
  DEF 3  quality labels from Delta_init (first rung as the initialization
         proxy) and Delta_peak (best rung), Tango paired score intervals
         (scripts/paired_intervals.py).

This is the run-level verdict only. Configuration-level reproducibility and
the racing ceiling are separate machinery with their own episode budgets.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

from analysis_input import AnalysisInputError, validate_analysis_input, validate_pair
from paired_intervals import METHOD as PAIRED_METHOD, paired_rate_interval
from validate_cert_v3 import load_json

DELTA = 0.10
W = 9
ALPHA2 = 0.02


def checked_rungs(curve):
    """Require certified full populations and verified pairing for every rung."""
    if curve.get("paired") is not True or curve.get("pairing") != "scenario-protocol":
        raise AnalysisInputError("CPI requires a paired scenario-protocol curve")
    rungs = curve.get("rungs")
    if not isinstance(rungs, list) or not rungs:
        raise AnalysisInputError("CPI requires nonempty certified rungs")
    for rung in rungs:
        validate_analysis_input(rung)
        if rung.get("opened_fresh") is not True or rung.get("opened_episode_index") != [0, 0]:
            raise AnalysisInputError("CPI requires fresh scenario-zero rung openings")
        n = len(rung["records"])
        successes = sum(r["success"] for r in rung["records"])
        if (rung.get("episodes") != n or rung.get("successes") != successes
                or rung.get("sr") != successes / n):
            raise AnalysisInputError("rung summary differs from its complete episode grid")
        validate_pair(rungs[0], rung)
    return rungs


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
    curve = load_json(path)
    rungs = checked_rungs(curve)
    if len(rungs) < W + 1:
        return {"curve": os.path.basename(path), "verdict": "NOT-CERTIFIABLE",
                "note": f"only {len(rungs)} rungs (< w+1)"}

    srs = [r["sr"] for r in rungs]
    steps = [r["step"] for r in rungs]
    window = rungs[-W:]
    anchor = window[-1]
    # Every assigned scenario contributes once. Invalid pairing is an error;
    # there is no intersection population or unpaired fallback.
    pairs_report = []
    stabilized = True
    for rung in window[:-1]:
        pairs = validate_pair(rung, anchor)
        interval = paired_rate_interval(
            [(a["success"], b["success"]) for a, b in pairs],
            confidence=1 - 2 * ALPHA2,
        )
        lo, hi = interval["interval"]
        mean, n = interval["difference"], interval["n"]
        ok = -DELTA < lo and hi < DELTA
        stabilized &= ok
        pairs_report.append((rung["step"], round(mean, 4), n,
                             "paired pass" if ok
                             else f"paired FAIL CI[{lo:+.3f},{hi:+.3f}]"))

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
    lo_i, hi_i = paired_rate_interval(
        [(a["success"], b["success"]) for a, b in validate_pair(first, anchor)]
    )["interval"]
    lo_p, hi_p = paired_rate_interval(
        [(a["success"], b["success"]) for a, b in validate_pair(peak, anchor)]
    )["interval"]

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
        "interval_method": PAIRED_METHOD,
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
