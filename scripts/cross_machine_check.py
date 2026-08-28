"""Compare two certificate JSONs produced on DIFFERENT machines.

Why per-episode and not just the success rate
---------------------------------------------
Two runs can agree on the success rate while disagreeing about which episodes
succeeded -- errors that cancel. On a benchmark whose whole selling point is
that a number means the same thing wherever it was produced, "the totals
matched" is not the claim worth making; "every episode landed the same way" is.

This matters more than it might look. The two boxes carry different GPU
architectures, and PhysX is not obliged to give bit-identical results across
them. So agreement here is a measurement, not a foregone conclusion, and a
DISagreement would be a real finding rather than a bug in this script: it would
mean certificates are machine-dependent and must be stamped with the hardware
they came from.

Checks, in increasing strictness:
  1. success rate equal
  2. the same episodes succeeded (per (env, ep) key)
  3. scenario hashes equal -- proves both ran the SAME exam paper
  4. continuous fields agree within a tolerance, and the worst offender named

Usage:
    python scripts/cross_machine_check.py boxA.json boxB.json
    python scripts/cross_machine_check.py a.json b.json --tol 1e-3

Pure standard library. Exit 0 if every check passes, 1 otherwise.
"""

import argparse
import json
import math
import os
import sys


CONTINUOUS = ("path_length_m", "tts_s", "max_hold_s", "min_clearance_m",
              "xte_rms_m")


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _records(blob):
    """The records list of a certificate, or None if this is not one."""
    if not isinstance(blob, dict):
        return None
    records = blob.get("records")
    return records if isinstance(records, list) else None


def _by_key(records):
    return {(r.get("env"), r.get("ep")): r for r in records
            if isinstance(r, dict)}


def _sorted_keys(keys):
    """Order (env, ep) keys, tolerating a certificate that omits either field.

    ``sorted`` raises TypeError as soon as one record carries env=None and
    another env=0, which is what a half-written certificate looks like. The
    order is cosmetic, so fall back to the text form rather than dying on a
    disagreement this tool exists to report. The natural sort is tried first so
    the normal case keeps its numeric order.
    """
    try:
        return sorted(keys)
    except TypeError:
        return sorted(keys, key=lambda key: tuple(str(part) for part in key))


def compare(path_a, path_b, tol):
    name_a = os.path.basename(path_a)
    name_b = os.path.basename(path_b)

    # A missing or truncated certificate is the most likely way this gate is
    # reached unattended, and a traceback in a queue log is a worse answer than
    # a named failure -- the exit status is the same either way.
    try:
        a, b = _load(path_a), _load(path_b)
    except (OSError, ValueError) as exc:
        print(f"cannot read certificate: {exc}", file=sys.stderr)
        print("CROSS-MACHINE GATE=FAIL")
        return False

    records_a, records_b = _records(a), _records(b)
    if records_a is None or records_b is None:
        which = name_a if records_a is None else name_b
        print(f"{which} carries no top-level 'records' list -- not a "
              f"certificate", file=sys.stderr)
        print("CROSS-MACHINE GATE=FAIL")
        return False

    ra, rb = _by_key(records_a), _by_key(records_b)
    shared = _sorted_keys(set(ra) & set(rb))

    print(f"A: {name_a}   ({len(ra)} episodes)")
    print(f"B: {name_b}   ({len(rb)} episodes)")
    print(f"shared (env, ep) keys: {len(shared)}")

    # Records that lost env/ep all hash to (None, None), so a damaged file
    # collapses to a single key and the counts printed above are keys rather
    # than episodes -- every check below would then run on one row while
    # reading like a full comparison.
    for label, rows, records in (("A", ra, records_a), ("B", rb, records_b)):
        if len(rows) != len(records):
            print(f"WARNING: {label} holds {len(records)} records but only "
                  f"{len(rows)} distinct (env, ep) keys -- duplicate or "
                  f"missing episode identifiers")

    if not shared:
        print("NOTHING TO COMPARE", file=sys.stderr)
        print("CROSS-MACHINE GATE=FAIL")
        return False

    # Every check below is computed over the OVERLAP, so a pass on a partial
    # overlap certifies only the episodes both boxes actually ran.
    if len(shared) < len(ra) or len(shared) < len(rb):
        print(f"WARNING: {len(ra) - len(shared)} episode(s) only in A and "
              f"{len(rb) - len(shared)} only in B are not compared; the "
              f"verdict below covers the {len(shared)} shared episode(s) only")

    ok = True

    sr_a = sum(1 for k in shared if ra[k].get("success")) / len(shared)
    sr_b = sum(1 for k in shared if rb[k].get("success")) / len(shared)
    same_sr = abs(sr_a - sr_b) < 1e-12
    print(f"\n1. success rate      A={sr_a:.4f}  B={sr_b:.4f}   "
          f"{'EQUAL' if same_sr else 'DIFFER'}")
    ok &= same_sr

    flips = [k for k in shared
             if bool(ra[k].get("success")) != bool(rb[k].get("success"))]
    print(f"2. per-episode outcome  disagreements: {len(flips)} / {len(shared)}"
          f"   {'IDENTICAL' if not flips else 'DIFFER'}")
    if flips:
        ok = False
        for k in flips[:5]:
            print(f"     env={k[0]} ep={k[1]}: A={ra[k].get('success')} "
                  f"B={rb[k].get('success')}")

    have_hash = [k for k in shared if ra[k].get("scenario_hashes")
                 and rb[k].get("scenario_hashes")]
    if have_hash:
        mismatched = [k for k in have_hash
                      if ra[k]["scenario_hashes"] != rb[k]["scenario_hashes"]]
        print(f"3. scenario hashes      mismatches: {len(mismatched)} / "
              f"{len(have_hash)}   "
              f"{'SAME EXAM PAPER' if not mismatched else 'DIFFERENT SCENARIOS'}")
        if mismatched:
            ok = False
            k = mismatched[0]
            print(f"     env={k[0]} ep={k[1]}: A={ra[k]['scenario_hashes']}")
            print(f"     env={k[0]} ep={k[1]}: B={rb[k]['scenario_hashes']}")
        if len(have_hash) < len(shared):
            print(f"       {len(shared) - len(have_hash)} shared episode(s) "
                  f"carry no hash on one or both sides and were not checked")
    else:
        print("3. scenario hashes      absent on one or both sides -- skipped")

    print("4. continuous fields")
    for field in CONTINUOUS:
        pairs = [(k, ra[k].get(field), rb[k].get(field)) for k in shared
                 if ra[k].get(field) is not None and rb[k].get(field) is not None]
        if not pairs:
            continue
        worst_key, worst = None, 0.0
        both_nan, half_nan = 0, []
        for k, va, vb in pairs:
            fa, fb = float(va), float(vb)
            # Every comparison against NaN is False, so a naive max-delta scan
            # skips NaN pairs entirely and reports EQUAL on records that carry
            # no number at all. NaN on both sides is a legitimate "not
            # applicable" and is only counted; NaN on exactly one side is a
            # genuine disagreement between the boxes and must fail the gate.
            if math.isnan(fa) or math.isnan(fb):
                if math.isnan(fa) and math.isnan(fb):
                    both_nan += 1
                else:
                    half_nan.append(k)
                continue
            delta = abs(fa - fb)
            if delta > worst:
                worst, worst_key = delta, k
        verdict = "EQUAL" if worst <= tol and not half_nan else "DIFFER"
        if worst > tol or half_nan:
            ok = False
        print(f"     {field:<16} n={len(pairs):<5} max |A-B| = {worst:.6g}"
              f"   {verdict}"
              + (f"   (env={worst_key[0]} ep={worst_key[1]})" if worst > tol else ""))
        if half_nan:
            k = half_nan[0]
            print(f"       {len(half_nan)} pair(s) NaN on exactly one side "
                  f"-- counted as DIFFER (first: env={k[0]} ep={k[1]})")
        if both_nan:
            print(f"       {both_nan} pair(s) NaN on both sides -- not "
                  f"comparable, excluded from max |A-B|")

    print()
    print("CROSS-MACHINE GATE=" + ("PASS -- the two boxes produced the same "
                                   "certificate" if ok else "FAIL"))
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="absolute tolerance for continuous fields "
                             "(default 1e-6)")
    args = parser.parse_args(argv)
    return 0 if compare(args.a, args.b, args.tol) else 1


if __name__ == "__main__":
    raise SystemExit(main())
