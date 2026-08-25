"""Robustness-cliff detector: 1-knot segmented fit on a dose-response series.

Task C of the ICRA-2027 statistics upgrade (2026-08-23). Pure CPU
post-processing; never launches Isaac.

Question answered (hypothesis B of the converged plan): where does a
controller's success rate break as a stress dose ramps, and how steep is the
break? Average SR across doses hides this structure; the knot makes it a
number with a CI.

Model (continuous piecewise-linear hinge, one knot):

    SR(d) = b0 + b1 * d + b2 * max(0, d - c)

  * pre-slope  = b1        (d < c side, in SR per dose unit)
  * post-slope = b1 + b2   (d > c side)
  * c is chosen by GRID SEARCH minimising the episode-weighted SSE
    sum_i n_i * (sr_i - fit_i)^2 over a dense grid of candidate knots
    restricted to [2nd smallest dose, 2nd largest dose] so both segments
    keep >= 2 points (201 grid points by default).
  * The descriptive fit is not clipped to [0, 1]; it is a shape estimate,
    not a probability model.

Bootstrap CI (frozen): B = 1000 replicates. Each dose point resamples its
own episodes: when the manifest lists gcert files, the pooled 0/1 success
vector is resampled with replacement; when it lists a bare (sr, n) pair the
replicate draws k ~ Binomial(n, sr) (identical distribution for Bernoulli
data). The full grid search is re-run per replicate; the knot and both
slopes are re-extracted; CIs are percentile [2.5, 97.5]. The share of
replicates whose knot lands on a grid boundary is reported -- a large share
means the knot is weakly identified, read the CI accordingly. RNG: numpy
default_rng seeded with (base_seed + crc32(series name)) mod 2**32,
base_seed default 20260823.

The dose axis is used exactly as given (aperture metres, gate metres, thrust
cap fraction, ...): for axes where SMALLER dose = harsher stress the cliff
shows as a positive post-to-pre slope break at the low end; the knot location
is direction-agnostic.

Manifest JSON (paths relative to the manifest's directory):
    {"name": "fortress_aperture", "dose_label": "aperture_m",
     "points": [
        {"dose": 8.0, "files": ["gcerts_cache/probe_fort_ap8p0_e42.json",
                                 "gcerts_cache/probe_fort_ap8p0_e123.json"],
         "source": "probe files, n=128 x 2 eval seeds"},
        {"dose": 6.0, "sr": 0.328, "n": 128,
         "source": "work log E25 tier0 (no per-episode file located)"},
        ...],
     "notes": "free text"}

Quick mode without a manifest:
    python scripts/cliff_detector.py --name gate --points 4.5:1.0:128 \\
        3.6:1.0:128 2.7:0.977:128 1.8:0.609:128 1.575:0.008:128

Usage:
    python scripts/cliff_detector.py MANIFEST.json --out-md out.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib

import numpy as np

DEFAULT_BOOT = 1000
DEFAULT_BOOT_SEED = 20260823
DEFAULT_GRID = 201


def load_points(manifest_path):
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    points = []
    for p in man["points"]:
        entry = {"dose": float(p["dose"]), "source": p.get("source", "")}
        if "files" in p:
            vec = []
            names = []
            for rel in p["files"]:
                path = rel if os.path.isabs(rel) else os.path.join(base_dir,
                                                                   rel)
                with open(path, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                vec.extend(1 if r.get("success") else 0
                           for r in doc["records"])
                names.append(os.path.basename(path))
            entry["vec"] = np.asarray(vec, dtype=np.int8)
            entry["n"] = int(entry["vec"].size)
            entry["sr"] = float(entry["vec"].mean())
            entry["basis"] = "episodes:" + ";".join(names)
        else:
            entry["sr"] = float(p["sr"])
            entry["n"] = int(p["n"])
            entry["vec"] = None
            entry["basis"] = "sr+n (documented)"
        points.append(entry)
    points.sort(key=lambda e: e["dose"])
    return man, points


def _design(doses, knot):
    return np.column_stack([np.ones_like(doses), doses,
                            np.maximum(0.0, doses - knot)])


def fit_grid(doses, weights, grid):
    """Precompute per-knot solve (M) and hat (H) operators for weighted LSQ.

    beta(y) = M @ y ; fitted(y) = H @ y. Linear in y, so the bootstrap only
    needs matrix products.
    """
    W = np.diag(weights)
    Ms, Hs = [], []
    for c in grid:
        X = _design(doses, c)
        XtW = X.T @ W
        # pinv for numerical safety on collinear grids (knot at a data point)
        M = np.linalg.pinv(XtW @ X) @ XtW
        Ms.append(M)
        Hs.append(X @ M)
    return np.stack(Ms), np.stack(Hs)


def best_fit(y, weights, grid, Ms, Hs):
    fitted = Hs @ y                      # (G, n)
    resid = y[None, :] - fitted
    sse = (weights[None, :] * resid ** 2).sum(axis=1)
    g = int(np.argmin(sse))
    beta = Ms[g] @ y
    return {"knot": float(grid[g]), "b0": float(beta[0]),
            "pre_slope": float(beta[1]),
            "post_slope": float(beta[1] + beta[2]),
            "sse_w": float(sse[g]), "grid_index": g,
            "fitted": fitted[g]}


def linear_sse(y, doses, weights):
    X = np.column_stack([np.ones_like(doses), doses])
    W = np.diag(weights)
    beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W) @ y
    resid = y - X @ beta
    return float((weights * resid ** 2).sum())


def analyse(name, points, n_boot, base_seed, grid_size):
    doses = np.array([p["dose"] for p in points], dtype=float)
    y = np.array([p["sr"] for p in points], dtype=float)
    n_eps = np.array([p["n"] for p in points], dtype=float)
    if doses.size < 4:
        raise SystemExit("need >= 4 dose points for a 1-knot fit, got %d"
                         % doses.size)
    lo, hi = np.sort(doses)[1], np.sort(doses)[-2]
    grid = (np.array([lo]) if lo == hi
            else np.linspace(lo, hi, grid_size))
    Ms, Hs = fit_grid(doses, n_eps, grid)
    point_fit = best_fit(y, n_eps, grid, Ms, Hs)
    sse_lin = linear_sse(y, doses, n_eps)

    rng = np.random.default_rng(
        (int(base_seed) + zlib.crc32(name.encode("utf-8"))) % (2 ** 32))
    knots = np.empty(n_boot)
    pres = np.empty(n_boot)
    posts = np.empty(n_boot)
    edge_hits = 0
    for b in range(n_boot):
        yb = np.empty_like(y)
        for i, p in enumerate(points):
            if p["vec"] is not None:
                vec = p["vec"]
                yb[i] = vec[rng.integers(0, vec.size, size=vec.size)].mean()
            else:
                yb[i] = rng.binomial(p["n"], p["sr"]) / p["n"]
        fb = best_fit(yb, n_eps, grid, Ms, Hs)
        knots[b] = fb["knot"]
        pres[b] = fb["pre_slope"]
        posts[b] = fb["post_slope"]
        if fb["grid_index"] in (0, grid.size - 1):
            edge_hits += 1

    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    return {
        "name": name, "doses": doses, "sr": y, "n": n_eps,
        "points": points, "fit": point_fit, "sse_linear": sse_lin,
        "knot_ci": ci(knots), "pre_ci": ci(pres), "post_ci": ci(posts),
        "edge_share": edge_hits / n_boot, "n_boot": n_boot,
        "grid": (float(grid[0]), float(grid[-1]), int(grid.size)),
    }


def render_md(res, dose_label, notes):
    f = res["fit"]
    lines = []
    lines.append("# Cliff detector: %s" % res["name"])
    lines.append("")
    lines.append("Model: SR(d) = b0 + b1*d + b2*max(0, d-knot); knot by grid"
                 " search over [%.4g, %.4g] (%d candidates), episode-weighted"
                 " SSE; bootstrap B=%d."
                 % (res["grid"][0], res["grid"][1], res["grid"][2],
                    res["n_boot"]))
    if notes:
        lines.append("")
        lines.append("Notes: %s" % notes)
    lines.append("")
    lines.append("## Dose points")
    lines.append("")
    lines.append("| %s | SR | n episodes | fitted SR | basis | source |"
                 % dose_label)
    lines.append("|---|---|---|---|---|---|")
    for i, p in enumerate(res["points"]):
        lines.append("| %.4g | %.4f | %d | %.4f | %s | %s |"
                     % (p["dose"], p["sr"], p["n"], f["fitted"][i],
                        p["basis"], p["source"]))
    lines.append("")
    lines.append("## Fit")
    lines.append("")
    lines.append("| quantity | estimate | boot lo95 | boot hi95 |")
    lines.append("|---|---|---|---|")
    lines.append("| knot (%s) | %.4f | %.4f | %.4f |"
                 % (dose_label, f["knot"], res["knot_ci"][0],
                    res["knot_ci"][1]))
    lines.append("| pre-slope (SR per unit, d < knot) | %.4f | %.4f | %.4f |"
                 % (f["pre_slope"], res["pre_ci"][0], res["pre_ci"][1]))
    lines.append("| post-slope (SR per unit, d > knot) | %.4f | %.4f | %.4f |"
                 % (f["post_slope"], res["post_ci"][0], res["post_ci"][1]))
    lines.append("")
    lines.append("Weighted SSE: segmented %.4f vs straight-line %.4f "
                 "(improvement %.1f%%). Knot-on-grid-edge share of bootstrap:"
                 " %.1f%% (large = weakly identified knot)."
                 % (f["sse_w"], res["sse_linear"],
                    100.0 * (1.0 - f["sse_w"] / res["sse_linear"])
                    if res["sse_linear"] > 0 else float("nan"),
                    100.0 * res["edge_share"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", nargs="?", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--points", nargs="+", default=None,
                    metavar="DOSE:SR:N", help="quick mode without files")
    ap.add_argument("--dose-label", default="dose")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    ap.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID)
    args = ap.parse_args(argv)

    if args.manifest:
        man, points = load_points(args.manifest)
        name = man.get("name") or os.path.basename(args.manifest)
        dose_label = man.get("dose_label", args.dose_label)
        notes = man.get("notes", "")
    elif args.points:
        points = []
        for spec in args.points:
            d, s, n = spec.split(":")
            points.append({"dose": float(d), "sr": float(s), "n": int(n),
                           "vec": None, "basis": "sr+n (cli)", "source": ""})
        points.sort(key=lambda e: e["dose"])
        name = args.name or "cli_series"
        dose_label = args.dose_label
        notes = ""
    else:
        ap.error("give a manifest or --points")

    res = analyse(name, points, args.boot, args.boot_seed, args.grid)
    md = render_md(res, dose_label, notes)
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
