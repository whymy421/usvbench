"""Paired per-episode comparison of two certificate cells.

Why paired
----------
The unpaired confidence intervals on the wave ladder are underpowered: at
n=128 the success-rate intervals for Hs=0.075 and Hs=0.60 overlap heavily,
and even the effort-metric (path length) intervals only just separate on one
of the two eval seeds. Read unpaired, the wave dose response is suggestive
rather than established.

But the ladder is a CONTROLLED comparison: across rungs at the same eval seed
the per-episode ``spawn_pose`` hashes are identical while the ``wave`` hashes
differ. Only the sea changes. The two cells are the SAME scenarios run under
two conditions, so the comparison should be paired. Pairing removes the
spawn-geometry variance, which is the dominant term.

Guards (v3plus, C1 finding 50)
------------------------------
Both inputs must be complete, unique, schema-v3 certificate grids from
compatible code/physics/runtime releases (scripts/analysis_input.py). EVERY
recorded scenario primitive must match between the two cells, except the
primitives declared, before inspection, as the treatment (repeated
--changed-primitive NAME, or matching certified header.pairing declarations).
Missing hashes, differing primitive inventories, incomplete or duplicated
grids and legacy files are refused (exit 2); there is no intersection of
populations and no --allow-spawn-mismatch bypass any more.

An endpoint may be unavailable (null) on some episodes: the denominators and
the missing counts are printed, never silently dropped.

Bootstrap
---------
The median interval resamples paired differences with Python's
random.Random (MT19937), seeded by --bootstrap-seed. This REPLACES the old
power-of-two-modulus LCG, whose low bits made every resample at n=128 a
permutation of the data (point-mass intervals; see the v3 NOTES). Intervals
printed by the previous version at power-of-two n are artefacts and are not
reproduced. Episodes must be independent scenario units; reused layouts or
several training seeds need a separate predeclared cluster analysis.

Usage:
    python scripts/paired_dose_test.py baseline.json treatment.json \
        --changed-primitive wave
    python scripts/paired_dose_test.py a.json b.json --field tts_s

Pure standard library (plus the v3 validator's jsonschema).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys

from analysis_input import finite_number, load_analysis_input, validate_metric, validate_pair
from paired_intervals import paired_rate_interval

# Fraction of exactly-zero differences past which a group's numbers get a
# printed caveat: enough for the header "n=" to overstate the sign test's real
# sample size by a third.
TIE_WARN_FRACTION = 0.25
BOOTSTRAP_BACKEND = "python.random.Random/MT19937-independent-index-v1"


def sign_test(differences):
    """Two-sided exact sign test; p is None when every difference is tied."""
    for value in differences:
        finite_number(value, "paired difference")
    pos = sum(value > 0 for value in differences)
    neg = sum(value < 0 for value in differences)
    n = pos + neg
    if not n:
        return 0, 0, None
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1))
    # Exact integer ratio: no float overflow above n = 1023.
    return pos, neg, min(1.0, 2.0 * (tail / (1 << n)))


def bootstrap_median_ci(values, iterations=4000, seed=20260827):
    """Percentile interval from independent paired-episode index resamples."""
    for value in values:
        finite_number(value, "paired difference")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("bootstrap iterations must be a positive integer")
    if len(values) < 4:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                     for _ in range(iterations))

    def quantile(q):
        position = (len(medians) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return medians[lo] + (medians[hi] - medians[lo]) * (position - lo)

    return quantile(0.025), quantile(0.975)


def degeneracy_notes(diffs, lo, hi, pos, neg):
    """Reasons a group's numbers must not be read at face value."""
    notes = []
    n = len(diffs)
    ties = sum(value == 0 for value in diffs)
    if n and ties >= TIE_WARN_FRACTION * n:
        notes.append(f"{ties}/{n} pairs differ by EXACTLY zero; the sign test drops "
                     f"ties, so it rests on {pos + neg} pairs, not {n}")
    if lo is not None and lo == hi:
        notes.append("point-mass empirical median interval: the resampled median never "
                     "moved; this is not evidence of precision or equivalence")
    if pos + neg == 0:
        notes.append("every pair is a tie, so the sign test is undefined and is NOT "
                     "evidence that the two arms agree")
    if n < 4:
        notes.append(f"n={n} is below the bootstrap floor of 4; interval unavailable")
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("treatment")
    parser.add_argument("--field", default="path_length_m",
                        help="per-episode field to compare (default path_length_m; see also "
                             "path_length_success_prefix_m / path_length_full_episode_m)")
    parser.add_argument("--changed-primitive", action="append", default=[], metavar="NAME",
                        help="scenario primitive intentionally changed by the treatment "
                             "(repeatable); every other primitive must match")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    args = parser.parse_args(argv)
    try:
        if args.bootstrap <= 0:
            raise ValueError("--bootstrap must be positive")
        base = load_analysis_input(args.baseline)
        treat = load_analysis_input(args.treatment)
        pairs = validate_pair(base, treat, args.changed_primitive)
        validate_metric(base, args.field)
        validate_metric(treat, args.field)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("REFUSING to pair: " + str(exc), file=sys.stderr)
        return 2

    print("baseline  : " + os.path.basename(args.baseline))
    print("treatment : " + os.path.basename(args.treatment))
    print(f"complete paired episodes: {len(pairs)}; every unchanged primitive hash equal"
          + (f" (treatment primitives: {', '.join(sorted(args.changed_primitive))})"
             if args.changed_primitive else ""))
    counts = {"both succeeded": 0, "baseline only": 0, "treatment only": 0, "neither": 0}
    both_ok, all_pairs = [], []
    missing_both, missing_all = 0, 0
    for b, t in pairs:
        bs, ts = b["success"], t["success"]
        label = ("both succeeded" if bs and ts else "baseline only" if bs
                 else "treatment only" if ts else "neither")
        counts[label] += 1
        bv, tv = b.get(args.field), t.get(args.field)
        if bv is None or tv is None:
            missing_all += 1
            missing_both += int(bs and ts)
            continue
        all_pairs.append(tv - bv)
        if bs and ts:
            both_ok.append(tv - bv)
    print()
    print("outcome breakdown (all assigned episodes)")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    flips = counts["baseline only"] + counts["treatment only"]
    print(f"  outcome flips: {flips} ({flips / len(pairs) * 100:.1f}%)")
    rate = paired_rate_interval([(t["success"], b["success"]) for b, t in pairs])
    lo, hi = rate["interval"]
    print(f"  success-rate difference treatment-minus-baseline: {rate['difference']:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] ({rate['method']}); "
          f"exact McNemar p={rate['mcnemar_exact_p']:.4g}")
    print(f"  unavailable {args.field}: {missing_all}/{len(pairs)} overall; "
          f"{missing_both}/{counts['both succeeded']} both-succeeded")
    print("bootstrap backend: " + BOOTSTRAP_BACKEND)
    if not all_pairs:
        print(f"{args.field} is unavailable on every pair; no test performed", file=sys.stderr)
        return 2
    for label, diffs in (("BOTH-SUCCEEDED pairs", both_ok), ("ALL observed pairs", all_pairs)):
        print()
        if not diffs:
            print(f"{label}: NO pair carries {args.field} on both sides -- NOT tested")
            continue
        pos, neg, p = sign_test(diffs)
        lo, hi = bootstrap_median_ci(diffs, args.bootstrap, args.bootstrap_seed)
        print(f"{label}  (n={len(diffs)})   field={args.field}")
        print(f"  median treatment-minus-baseline: {statistics.median(diffs):+.6g}")
        interval = "unavailable" if lo is None else f"[{lo:+.6g}, {hi:+.6g}]"
        print("  paired bootstrap 95% CI: " + interval)
        ptext = "undefined (all ties)" if p is None else f"{p:.6g}"
        print(f"  exact sign test: {pos} up / {neg} down; p={ptext}")
        if p is None or p >= 0.05:
            verdict = "no detectable difference"
        else:
            verdict = ("treatment INCREASES the field" if pos > neg
                       else "treatment DECREASES the field")
        print(f"  verdict: {verdict}")
        for note in degeneracy_notes(diffs, lo, hi, pos, neg):
            print("  CAVEAT: " + note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
