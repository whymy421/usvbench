"""Rank-stability analysis between a nominal and a stress evaluation arm.

Task C of the ICRA-2027 statistics upgrade (2026-08-23). Pure CPU
post-processing of existing gcert JSONs; never launches Isaac.

Question answered (hypothesis A of the converged plan): does the ranking of a
controller set under NOMINAL conditions survive a STRESS condition? We
compute, over the same controller set,

  * Spearman rho and Kendall tau-b between the nominal-SR column and the
    stress-SR column, each with a bootstrap CI (episode-level resampling),
  * a paired sign-flip listing of EVERY rank inversion: controller pair
    (i, j) is inverted when (SR_nom_i - SR_nom_j) * (SR_str_i - SR_str_j) < 0,
    reported with a bootstrap flip probability P_flip = share of resamples in
    which the pair's order differs between the two columns (ties count as
    "not inverted"; pairs tied in the point estimate are listed separately).

Bootstrap protocol (frozen): B = 1000 replicates; in each replicate every
controller x arm cell resamples its OWN pooled episode success vector with
replacement (cell sizes preserved, cells independent -- exactly the binomial
sampling noise of the certification protocol); rho / tau are recomputed on
the resampled SR columns. Replicates where a column is constant have
undefined rho / tau; they are dropped from the CI and counted in the output
(n_valid). RNG: numpy default_rng seeded with
(base_seed + crc32(manifest name)) mod 2**32, base_seed default 20260823.

With only two controllers rho / tau are degenerate (+/-1 or undefined); the
script still runs and the sign-flip listing carries the content. A stderr
note flags k < 4.

Manifest JSON:
    {
      "name": "sk_calm_vs_current",
      "nominal_label": "calm",
      "stress_label": "current 2.0-2.5 m/s",
      "controllers": {
        "LOS-PID":  {"nominal": ["gcerts_pid/gcert_pid_sk_e42.json", ...],
                     "stress":  ["gcerts_pid/gcert_pid_skc_e42.json", ...]},
        "PPO-champ": {...}
      },
      "notes": "free text, printed into the report"
    }
Paths are relative to the manifest's directory (or absolute). Episodes are
pooled across the listed files of a cell (e.g. both eval seeds); the pooled n
per cell is printed so mixed-n cells cannot hide.

Usage:
    python scripts/ranking_reversal.py MANIFEST.json \\
        --out-md outputs/taskC_stats/rank_sk_calm_current.md \\
        --out-csv outputs/taskC_stats/rank_sk_calm_current_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import zlib

import numpy as np
from scipy import stats as sstats

Z95 = 1.959963984540054
DEFAULT_BOOT = 1000
DEFAULT_BOOT_SEED = 20260823


def wilson_ci(k, n, z=Z95):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_success_vector(paths, base_dir):
    """Pooled 0/1 episode success vector across the given gcert files."""
    vec = []
    files = []
    for rel in paths:
        path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        recs = doc.get("records")
        if not isinstance(recs, list) or not recs:
            raise ValueError("%s: no records" % path)
        vec.extend(1 if r.get("success") else 0 for r in recs)
        files.append(os.path.basename(path))
    return np.asarray(vec, dtype=np.int8), files


def rank_corr(col_a, col_b):
    """(spearman_rho, kendall_taub); nan when either column is constant."""
    if len(set(col_a)) < 2 or len(set(col_b)) < 2:
        return (float("nan"), float("nan"))
    rho = sstats.spearmanr(col_a, col_b).statistic
    tau = sstats.kendalltau(col_a, col_b, variant="b").statistic
    return (float(rho), float(tau))


def analyse(manifest_path, n_boot, base_seed):
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    name = man.get("name") or os.path.basename(manifest_path)
    ctrls = man["controllers"]
    order = list(ctrls.keys())
    if len(order) < 4:
        sys.stderr.write("[note] %s: only %d controllers -- rho/tau are "
                         "coarse; read the sign-flip listing\n"
                         % (name, len(order)))

    cells = {}
    for c in order:
        for armkey in ("nominal", "stress"):
            vec, files = load_success_vector(ctrls[c][armkey], base_dir)
            cells[(c, armkey)] = {"vec": vec, "files": files,
                                  "n": int(vec.size),
                                  "k": int(vec.sum()),
                                  "sr": float(vec.mean())}

    sr_nom = [cells[(c, "nominal")]["sr"] for c in order]
    sr_str = [cells[(c, "stress")]["sr"] for c in order]
    rho, tau = rank_corr(sr_nom, sr_str)

    rng = np.random.default_rng(
        (int(base_seed) + zlib.crc32(name.encode("utf-8"))) % (2 ** 32))
    boot_nom = np.empty((n_boot, len(order)))
    boot_str = np.empty((n_boot, len(order)))
    for j, c in enumerate(order):
        for arr, armkey in ((boot_nom, "nominal"), (boot_str, "stress")):
            vec = cells[(c, armkey)]["vec"]
            n = vec.size
            idx = rng.integers(0, n, size=(n_boot, n))
            arr[:, j] = vec[idx].mean(axis=1)

    rhos, taus = [], []
    for b in range(n_boot):
        r, t = rank_corr(list(boot_nom[b]), list(boot_str[b]))
        if not math.isnan(r):
            rhos.append(r)
        if not math.isnan(t):
            taus.append(t)

    def pct_ci(vals):
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5)))

    result = {
        "name": name,
        "nominal_label": man.get("nominal_label", "nominal"),
        "stress_label": man.get("stress_label", "stress"),
        "notes": man.get("notes", ""),
        "controllers": order,
        "cells": cells,
        "spearman": rho, "spearman_ci": pct_ci(rhos),
        "spearman_n_valid": len(rhos),
        "kendall": tau, "kendall_ci": pct_ci(taus),
        "kendall_n_valid": len(taus),
        "n_boot": n_boot,
    }

    pairs = []
    for i, j in itertools.combinations(range(len(order)), 2):
        d_nom = sr_nom[i] - sr_nom[j]
        d_str = sr_str[i] - sr_str[j]
        inverted = d_nom * d_str < 0
        tied = (d_nom == 0) or (d_str == 0)
        flips = np.mean(
            (boot_nom[:, i] - boot_nom[:, j])
            * (boot_str[:, i] - boot_str[:, j]) < 0)
        pairs.append({
            "pair": "%s vs %s" % (order[i], order[j]),
            "sr_nom_i": round(sr_nom[i], 4), "sr_nom_j": round(sr_nom[j], 4),
            "sr_str_i": round(sr_str[i], 4), "sr_str_j": round(sr_str[j], 4),
            "delta_nominal": round(d_nom, 4),
            "delta_stress": round(d_str, 4),
            "inverted": bool(inverted),
            "tied_point_estimate": bool(tied),
            "p_flip_boot": round(float(flips), 4),
        })
    result["pairs"] = pairs
    return result


def render_md(res):
    lines = []
    lines.append("# Ranking reversal: %s" % res["name"])
    lines.append("")
    lines.append("Nominal = %s; stress = %s. Bootstrap B=%d over episodes."
                 % (res["nominal_label"], res["stress_label"], res["n_boot"]))
    if res["notes"]:
        lines.append("")
        lines.append("Notes: %s" % res["notes"])
    lines.append("")
    lines.append("## Per-controller SR (pooled episodes, Wilson 95%)")
    lines.append("")
    lines.append("| controller | arm | files | n | k | SR | lo95 | hi95 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in res["controllers"]:
        for armkey in ("nominal", "stress"):
            cell = res["cells"][(c, armkey)]
            lo, hi = wilson_ci(cell["k"], cell["n"])
            lines.append("| %s | %s | %s | %d | %d | %.4f | %.4f | %.4f |"
                         % (c, armkey, ";".join(cell["files"]), cell["n"],
                            cell["k"], cell["sr"], lo, hi))
    lines.append("")
    lines.append("## Rank correlation (nominal column vs stress column)")
    lines.append("")
    lines.append("| stat | value | boot lo95 | boot hi95 | valid replicates |")
    lines.append("|---|---|---|---|---|")
    lines.append("| Spearman rho | %.4f | %.4f | %.4f | %d/%d |"
                 % (res["spearman"], res["spearman_ci"][0],
                    res["spearman_ci"][1], res["spearman_n_valid"],
                    res["n_boot"]))
    lines.append("| Kendall tau-b | %.4f | %.4f | %.4f | %d/%d |"
                 % (res["kendall"], res["kendall_ci"][0],
                    res["kendall_ci"][1], res["kendall_n_valid"],
                    res["n_boot"]))
    lines.append("")
    lines.append("## Paired sign-flip listing (every controller pair)")
    lines.append("")
    lines.append("| pair | SR nom (i/j) | SR stress (i/j) | d_nom | d_stress"
                 " | inverted | tie | P_flip(boot) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in res["pairs"]:
        lines.append("| %s | %.4f / %.4f | %.4f / %.4f | %+.4f | %+.4f |"
                     " %s | %s | %.4f |"
                     % (p["pair"], p["sr_nom_i"], p["sr_nom_j"],
                        p["sr_str_i"], p["sr_str_j"], p["delta_nominal"],
                        p["delta_stress"],
                        "YES" if p["inverted"] else "no",
                        "tie" if p["tied_point_estimate"] else "",
                        p["p_flip_boot"]))
    inv = sum(1 for p in res["pairs"] if p["inverted"])
    lines.append("")
    lines.append("%d/%d pairs inverted under stress." % (inv, len(res["pairs"])))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", help="manifest JSON (see module docstring)")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-csv", default=None,
                    help="pair listing as CSV")
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    ap.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    args = ap.parse_args(argv)

    res = analyse(args.manifest, args.boot, args.boot_seed)
    md = render_md(res)
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(res["pairs"][0].keys()))
            w.writeheader()
            w.writerows(res["pairs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
