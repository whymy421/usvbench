"""No-Isaac validation of the Task C statistics trio.

Covers scripts/deployment_metrics.py, scripts/ranking_reversal.py and
scripts/cliff_detector.py with synthetic data whose answers are known in
closed form, plus the house source-hygiene check (AST parse, ASCII only, no
control characters, no tabs). Nothing here launches Isaac or reads real
certification data.

Run:  python scripts/test_metrics_stats.py
  or: python -m pytest scripts/test_metrics_stats.py -v
"""
from __future__ import annotations

import ast
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cliff_detector as cd          # noqa: E402
import deployment_metrics as dm      # noqa: E402
import ranking_reversal as rr        # noqa: E402

CHECKED_SOURCES = ["deployment_metrics.py", "ranking_reversal.py",
                   "cliff_detector.py", "test_metrics_stats.py"]


def test_source_hygiene():
    for fname in CHECKED_SOURCES:
        path = os.path.join(HERE, fname)
        raw = open(path, "rb").read()
        for b in raw:
            assert not (b < 32 and b not in (10, 13)), \
                "%s contains control byte %d" % (fname, b)
            assert b < 128, "%s contains non-ASCII byte %d" % (fname, b)
        ast.parse(raw.decode("ascii"), filename=fname)


def test_wilson_known_values():
    # classic textbook case: 50/100 -> (0.40383, 0.59617)
    lo, hi = dm.wilson_ci(50, 100)
    assert abs(lo - 0.40383) < 1e-4 and abs(hi - 0.59617) < 1e-4
    # k = 0: lower bound exactly 0, upper bound z^2/(n+z^2)
    lo0, hi0 = dm.wilson_ci(0, 128)
    z2 = dm.Z95 ** 2
    assert lo0 == 0.0 and abs(hi0 - z2 / (128 + z2)) < 1e-12
    # symmetry: k = n mirrors k = 0
    lo1, hi1 = dm.wilson_ci(128, 128)
    assert abs((1.0 - lo1) - hi0) < 1e-12 and hi1 == 1.0
    # interval always contains the point estimate
    for k, n in [(1, 7), (13, 128), (127, 128), (32, 32)]:
        lo, hi = dm.wilson_ci(k, n)
        assert lo <= k / n <= hi


def test_cvar_exact():
    losses = [i / 100.0 for i in range(1, 101)]        # 0.01 .. 1.00
    # worst 25% of 100 pts = 0.76..1.00, mean = 0.88
    assert abs(dm.cvar(losses, 0.25) - 0.88) < 1e-12
    # worst 10% = 0.91..1.00, mean = 0.955
    assert abs(dm.cvar(losses, 0.10) - 0.955) < 1e-12
    # fractional k rounds UP: n=10, alpha=0.25 -> k=3
    ten = [0.1 * i for i in range(1, 11)]
    assert abs(dm.cvar(ten, 0.25) - (0.8 + 0.9 + 1.0) / 3) < 1e-12
    # degenerate: all failures -> 1.0
    assert dm.cvar([1.0] * 5, 0.1) == 1.0


def test_loss_mapping():
    warns = []
    assert dm.episode_loss({"success": False}, 120.0, warns) == 1.0
    assert dm.episode_loss({"success": True, "tts_s": 60.0}, 120.0) == 0.5
    # clipping above budget
    assert dm.episode_loss({"success": True, "tts_s": 130.0}, 120.0) == 1.0
    # success without tts_s -> max loss, warning recorded
    assert dm.episode_loss({"success": True}, 120.0, warns) == 1.0
    assert any("tts_s missing" in w for w in warns)


def test_bootstrap_deterministic_and_sane():
    rng = np.random.default_rng(7)
    losses = np.clip(rng.normal(0.5, 0.2, size=200), 0, 1)
    a = dm.bootstrap_cvar_ci(losses, 0.25, 400, seed=99)
    b = dm.bootstrap_cvar_ci(losses, 0.25, 400, seed=99)
    c = dm.bootstrap_cvar_ci(losses, 0.25, 400, seed=100)
    assert a == b and a != c
    point = dm.cvar(losses, 0.25)
    assert a[0] <= point <= a[1]
    assert dm.arm_seed("x", 1) == dm.arm_seed("x", 1) != dm.arm_seed("y", 1)


def _hinge_sr(dose, knot=6.0, base=0.9, post=-0.15):
    return base + post * max(0.0, knot - dose)   # cliff on the LOW-dose side


def test_segmented_exact_recovery():
    # noise-free hinge data with the knot exactly on the candidate grid:
    # doses 1..10, grid over [2, 9] with 8 candidates = the integers.
    points = [{"dose": float(d), "sr": _hinge_sr(d), "n": 128, "vec": None,
               "basis": "synthetic", "source": ""} for d in range(1, 11)]
    res = cd.analyse("synthetic_exact", points, n_boot=50,
                     base_seed=1, grid_size=8)
    fit = res["fit"]
    assert abs(fit["knot"] - 6.0) < 1e-9
    assert abs(fit["pre_slope"] - 0.15) < 1e-9      # rising toward the knot
    assert abs(fit["post_slope"] - 0.0) < 1e-9      # flat above it
    assert fit["sse_w"] < 1e-18


def test_segmented_bootstrap_ci_contains_truth():
    rng = np.random.default_rng(20260823)
    points = []
    for d in range(1, 11):
        p = _hinge_sr(d)
        vec = (rng.random(128) < p).astype(np.int8)
        points.append({"dose": float(d), "sr": float(vec.mean()),
                       "n": 128, "vec": vec, "basis": "synthetic-episodes",
                       "source": ""})
    res = cd.analyse("synthetic_noisy", points, n_boot=300,
                     base_seed=5, grid_size=8)
    lo, hi = res["knot_ci"]
    assert lo <= 6.0 <= hi, "knot CI %s misses truth 6.0" % str((lo, hi))
    # the pre-knot climb of 0.15/unit must be visibly nonzero
    assert res["pre_ci"][0] > 0.05


def _write_gcert(path, task, n_success, n_total, seed=42):
    recs = []
    for i in range(n_total):
        rec = {"env": i, "ep": 0, "success": i < n_success,
               "path_length_m": 10.0}
        rec["tts_s"] = 30.0 + i * 0.1 if rec["success"] else None
        recs.append(rec)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"task": task, "seed": seed, "controller": "synthetic",
                   "records": recs}, fh)


def test_ranking_reversal_perfect_inversion():
    with tempfile.TemporaryDirectory() as tmp:
        spec = {"A": (900, 200), "B": (600, 500), "C": (300, 800)}
        man = {"name": "synthetic_reversal", "controllers": {}}
        for cname, (k_nom, k_str) in spec.items():
            fn = os.path.join(tmp, "%s_nom.json" % cname)
            fs = os.path.join(tmp, "%s_str.json" % cname)
            _write_gcert(fn, "Isaac-USV-HazardNav-Direct-v9", k_nom, 1000)
            _write_gcert(fs, "Isaac-USV-HazardNav-Direct-v9", k_str, 1000)
            man["controllers"][cname] = {"nominal": [fn], "stress": [fs]}
        mp = os.path.join(tmp, "man.json")
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(man, fh)
        res = rr.analyse(mp, n_boot=300, base_seed=3)
    assert res["spearman"] == -1.0 and res["kendall"] == -1.0
    assert all(p["inverted"] for p in res["pairs"])
    ac = [p for p in res["pairs"] if p["pair"] == "A vs C"][0]
    assert ac["p_flip_boot"] > 0.99   # 0.9 vs 0.3 flipping to 0.2 vs 0.8
    # bootstrap CI of a perfect reversal stays at the floor
    assert res["spearman_ci"][1] <= -0.5


def test_rank_corr_degenerate_columns():
    rho, tau = rr.rank_corr([0.5, 0.5, 0.5], [0.1, 0.2, 0.3])
    assert math.isnan(rho) and math.isnan(tau)


def _run_all():
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  %s" % name)
            except AssertionError as exc:
                fails += 1
                print("FAIL  %s: %s" % (name, exc))
            except Exception as exc:  # noqa: BLE001 - surfaced, not hidden
                fails += 1
                print("ERROR %s: %r" % (name, exc))
    return fails


if __name__ == "__main__":
    raise SystemExit(_run_all())
