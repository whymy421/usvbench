"""Paired per-episode comparison of two certificate cells.

Why paired
----------
The unpaired confidence intervals on the wave ladder are underpowered: at
n=128 the success-rate intervals for Hs=0.075 and Hs=0.60 overlap heavily,
and even the effort-metric (path length) intervals only just separate on one
of the two eval seeds. Read unpaired, the wave dose response is suggestive
rather than established.

But the ladder is a CONTROLLED comparison and we verified it: across rungs at
the same eval seed the per-episode ``spawn_pose`` hashes are identical 128/128
while the ``wave`` hashes differ 128/128. Only the sea changes. That means the
two cells are not two independent samples -- they are the SAME 128 scenarios
run under two sea states, so the comparison should be paired.

Pairing removes the spawn-geometry variance, which is the dominant term: spawn
distance varies up to 15 m across episodes while the wave effect is a couple
of metres. A sign test on the per-episode differences therefore has far more
power than comparing two medians.

Guards
------
The script REFUSES to pair cells whose spawn hashes do not match, because that
is precisely the case where pairing would manufacture a result. It also
reports the success-outcome breakdown, since path length means different
things for a held station and a lost one.

Usage:
    python scripts/paired_dose_test.py baseline.json treatment.json
    python scripts/paired_dose_test.py a.json b.json --field tts_s

Pure standard library.
"""

import argparse
import json
import math
import os
import statistics
import sys


# Fraction of exactly-zero differences past which a group's numbers get a
# printed caveat. Set at a quarter because that is already enough for the "n="
# on the header line to overstate the sign test's real sample size by a third,
# which is the reading error the caveat exists to prevent.
TIE_WARN_FRACTION = 0.25


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _records(blob):
    """The records list of a certificate, or None if this is not one.

    A glob over a results directory picks up score summaries and other JSON
    that happens to live beside the certificates (gcerts/usv10k_scores.json is
    a bare list), and a run killed mid-write leaves a truncated file. Both
    should say so rather than raise AttributeError three frames down.
    """
    if not isinstance(blob, dict):
        return None
    records = blob.get("records")
    return records if isinstance(records, list) else None


def _key(record):
    return (record.get("env"), record.get("ep"))


def _sorted_keys(keys):
    """Order (env, ep) keys, tolerating a certificate that omits either field.

    ``sorted`` raises TypeError the moment one record carries env=None and
    another env=0, which is exactly the shape a half-written certificate has.
    The order is cosmetic, so fall back to the text form instead of dying on a
    file that can still be analysed. The natural sort is tried first so the
    normal case keeps its numeric order.
    """
    try:
        return sorted(keys)
    except TypeError:
        return sorted(keys, key=lambda key: tuple(str(part) for part in key))


def _spawn(record):
    hashes = record.get("scenario_hashes") or {}
    return hashes.get("spawn_pose")


def sign_test(differences):
    """Two-sided exact sign test. Returns (n_pos, n_neg, p_value).

    Zero differences are dropped, the standard convention. Exact binomial
    tail, so it stays valid at the small paired-sample sizes that appear once
    both-succeeded pairs are required.
    """
    pos = sum(1 for d in differences if d > 0)
    neg = sum(1 for d in differences if d < 0)
    n = pos + neg
    if n == 0:
        return 0, 0, float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    if n < 1024:
        p = min(1.0, 2.0 * tail / (2.0 ** n))
    else:
        # 2.0 ** n overflows the float exponent above n = 1023, which a cell
        # run with a few thousand episodes reaches. The exact integer ratio is
        # correctly rounded and underflows to 0.0 instead of raising, so the
        # tail stays reportable. The float form is kept below 1024 so that no
        # p-value reachable at today's episode counts changes by even an ulp.
        p = min(1.0, 2.0 * (tail / (1 << n)))
    return pos, neg, p


def bootstrap_median_ci(values, iterations=4000, seed=20260827):
    """Percentile bootstrap CI for the median of the paired differences.

    Deterministic: the RNG is seeded from a fixed constant so the reported
    interval is reproducible from the same inputs.

    KNOWN LIMITATION, detected but deliberately not repaired here: _Lcg has a
    power-of-two modulus, so below(n) returns its low bits, which are a
    full-period LCG on 2**k in their own right. At a power-of-two n -- 128 is
    the default episode count -- n consecutive draws therefore visit every
    index exactly once, every resample is a permutation of the input, and this
    function returns [median, median] whatever the data are. Changing the
    resampler would move every bootstrap interval this tool has ever printed,
    so degeneracy_notes flags the affected groups instead and the choice is
    left to whoever owns the published numbers.
    """
    if len(values) < 4:
        return float("nan"), float("nan")
    rng = _Lcg(seed)
    n = len(values)
    medians = []
    for _ in range(iterations):
        sample = [values[rng.below(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo = medians[int(0.025 * len(medians))]
    hi = medians[int(0.975 * len(medians)) - 1]
    return lo, hi


class _Lcg:
    """Tiny deterministic PRNG so the bootstrap needs no numpy and no random."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def below(self, n):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state % n


def degeneracy_notes(diffs, lo, hi, pos, neg):
    """Reasons a group's numbers must not be read at face value.

    Observed on the max_hold_s wave ladder: most episodes never reach the hold
    circle at all, so max_hold_s is exactly 0 on BOTH arms and the paired
    difference is exactly 0. The sign test drops those pairs, so its n is far
    smaller than the n printed on the header line, and the bootstrap resamples
    a vector that is mostly zeros -- almost every resample returns the same
    tied median, which collapsed the 95% CI to the point mass [-0.008,-0.008].
    A point-mass interval reads as an extraordinarily precise measurement when
    it means the opposite: the estimator had almost nothing to vary over.

    That run's load-bearing analysis was the both-succeeded subset, but the
    all-pairs verdict printed in the same shape and carried the same apparent
    authority, which is exactly the failure mode an unattended log invites.
    These notes exist so the caveat travels with the numbers.
    """
    notes = []
    n = len(diffs)
    ties = sum(1 for d in diffs if d == 0.0)
    if n and ties >= TIE_WARN_FRACTION * n:
        notes.append(f"{ties}/{n} pairs ({ties / n * 100:.1f}%) differ by "
                     f"EXACTLY zero; the sign test drops ties, so it rests on "
                     f"{pos + neg} pairs, not {n}")
    if not math.isnan(lo) and lo == hi:
        notes.append(f"the bootstrap 95% CI is a POINT MASS at {lo:+.3f} -- "
                     f"the resampled median never moved, so this interval "
                     f"measures nothing and must not be quoted as precision")
    if n >= 4 and n & (n - 1) == 0:
        # _Lcg is a power-of-two-modulus LCG, whose low bits are themselves a
        # full-period LCG on 2**k. When n is a power of two, below(n) returns
        # the low bits, so n consecutive draws hit every index EXACTLY once:
        # each "resample" is a permutation of the data, its median is the
        # sample median, and the CI is a point mass no matter what the data
        # look like. n=128 is this benchmark's default episode count, so this
        # fires on the common case and the CI above is an artefact.
        notes.append(f"n={n} is a power of two, where the bootstrap resampler "
                     f"draws each index exactly once and every resample is a "
                     f"permutation -- the CI above is an artefact of the "
                     f"index generator, NOT a property of the data")
    if pos + neg == 0:
        notes.append("every pair is a tie, so the sign test is undefined "
                     "(p=nan) and the verdict below is NOT evidence that the "
                     "two arms agree")
    if n < 4:
        notes.append(f"n={n} is below the bootstrap floor of 4, so the CI is "
                     f"reported as nan rather than computed")
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("treatment")
    parser.add_argument("--field", default="path_length_m",
                        help="per-episode field to compare (default path_length_m)")
    parser.add_argument("--allow-spawn-mismatch", action="store_true",
                        help="pair anyway; only for cells you KNOW are paired "
                             "by construction without carrying spawn hashes")
    args = parser.parse_args(argv)

    try:
        base = _load(args.baseline)
        treat = _load(args.treatment)
    except (OSError, ValueError) as exc:
        print(f"cannot read certificate: {exc}", file=sys.stderr)
        return 2

    base_records, treat_records = _records(base), _records(treat)
    if base_records is None or treat_records is None:
        which = args.baseline if base_records is None else args.treatment
        print(f"{os.path.basename(which)} carries no top-level 'records' list "
              f"-- it is not a certificate", file=sys.stderr)
        return 2

    base_rows = {_key(r): r for r in base_records if isinstance(r, dict)}
    treat_rows = {_key(r): r for r in treat_records if isinstance(r, dict)}
    shared = _sorted_keys(set(base_rows) & set(treat_rows))
    if not shared:
        print("no shared (env, ep) keys -- these cells cannot be paired",
              file=sys.stderr)
        return 2

    matched = sum(1 for k in shared
                  if _spawn(base_rows[k]) is not None
                  and _spawn(base_rows[k]) == _spawn(treat_rows[k]))
    have_hashes = sum(1 for k in shared if _spawn(base_rows[k]) is not None)

    print(f"baseline  : {os.path.basename(args.baseline)}")
    print(f"treatment : {os.path.basename(args.treatment)}")
    print(f"shared episodes: {len(shared)}   "
          f"identical spawn hash: {matched}/{have_hashes}")

    # A record that lost its env/ep fields hashes to (None, None), so a
    # damaged certificate collapses to ONE row and everything below would run
    # on a single fabricated pair while still printing a confident verdict.
    for label, rows, records in (("baseline", base_rows, base_records),
                                 ("treatment", treat_rows, treat_records)):
        if len(rows) != len(records):
            print(f"WARNING: {label} holds {len(records)} records but only "
                  f"{len(rows)} distinct (env, ep) keys -- duplicate or "
                  f"missing episode identifiers, so the pairing below is "
                  f"unreliable")

    # The spawn-hash guard can only refuse what it can see. Absent hashes are
    # not evidence of a valid pairing, and a run where only a handful of
    # episodes carry them is a guard that checked a handful of episodes.
    if have_hashes == 0:
        print("WARNING: not one shared episode carries a spawn hash, so the "
              "guard below cannot fire -- the pairing is UNVERIFIED and rests "
              "entirely on the two cells being paired by construction")
    elif have_hashes < len(shared):
        print(f"WARNING: only {have_hashes}/{len(shared)} shared episodes "
              f"carry a spawn hash; the remaining "
              f"{len(shared) - have_hashes} pair(s) are unverified")

    if have_hashes and matched != have_hashes and not args.allow_spawn_mismatch:
        print()
        print("REFUSING to pair: the spawn scenarios differ, so a paired test "
              "would attribute spawn variance to the treatment. Pass "
              "--allow-spawn-mismatch only if you know the pairing holds by "
              "construction.", file=sys.stderr)
        return 3

    both_ok, base_only, treat_only, neither = [], 0, 0, 0
    all_pairs = []
    for k in shared:
        b, t = base_rows[k], treat_rows[k]
        bv, tv = b.get(args.field), t.get(args.field)
        bs, ts = bool(b.get("success")), bool(t.get("success"))
        if bs and ts:
            if bv is not None and tv is not None:
                both_ok.append(float(tv) - float(bv))
        elif bs and not ts:
            base_only += 1
        elif ts and not bs:
            treat_only += 1
        else:
            neither += 1
        if bv is not None and tv is not None:
            all_pairs.append(float(tv) - float(bv))

    print()
    print("outcome breakdown")
    print(f"  both succeeded      : {len(both_ok)}")
    print(f"  baseline only       : {base_only}")
    print(f"  treatment only      : {treat_only}")
    print(f"  neither             : {neither}")
    print(f"  outcome flips       : {base_only + treat_only} "
          f"({(base_only + treat_only) / max(1, len(shared)) * 100:.1f}%)")

    # "both succeeded" counts pairs that BOTH succeeded AND carry the field,
    # because it doubles as the n of the first analysis below. The other three
    # lines count every pair. A certificate written before the field existed
    # therefore reports zeros across the board, which reads as "no episode did
    # anything" rather than "this cell cannot answer the question asked".
    unaccounted = len(shared) - (len(both_ok) + base_only + treat_only + neither)
    if unaccounted:
        print(f"  NOTE: {unaccounted} pair(s) where both arms succeeded are "
              f"missing from the count above because {args.field} is absent "
              f"on one or both sides")

    for label, diffs in (("BOTH-SUCCEEDED pairs", both_ok),
                         ("ALL pairs", all_pairs)):
        if not diffs:
            # Silently skipping a group means a certificate that never wrote
            # the field produces a clean exit 0 with no test in it at all.
            print()
            print(f"{label}: NO pair carries {args.field} on both sides -- "
                  f"this group was NOT tested")
            continue
        pos, neg, p = sign_test(diffs)
        lo, hi = bootstrap_median_ci(diffs)
        print()
        print(f"{label}  (n={len(diffs)})   field={args.field}")
        print(f"  median difference : {statistics.median(diffs):+.3f}")
        print(f"  bootstrap 95% CI  : [{lo:+.3f}, {hi:+.3f}]")
        print(f"  sign test         : {pos} up / {neg} down   p = {p:.3g}")
        verdict = ("treatment INCREASES the field" if p < 0.05 and pos > neg
                   else "treatment DECREASES the field" if p < 0.05
                   else "no detectable difference")
        print(f"  verdict           : {verdict}")
        for note in degeneracy_notes(diffs, lo, hi, pos, neg):
            print(f"  CAVEAT            : {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
