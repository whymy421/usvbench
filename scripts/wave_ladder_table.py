"""Build the wave-ladder table from certificate JSONs, with GRADED metrics.

Why graded metrics
------------------
Across the first wave ladder every controller showed the same shape: a large
step the moment a sea state is switched on, then a nearly FLAT response to
wave height. The calm-water champion scores 0.59/0.54 at Hs = 0.075 m and
0.57/0.51 at Hs = 0.60 m -- an eight-fold dose increase for three or four
points of success rate.

That is a property of the success CRITERION, not of the sea. Station keeping
requires a continuous 60 s inside a 2 m circle within a 120 s episode, and the
hold clock resets on every excursion. So the metric is close to binary: either
a long enough excursion-free window exists or it does not. Binary metrics
cannot resolve a dose.

The certificates already carry two continuous quantities that can:

  path_length_m  how far the hull actually travelled. A boat that holds
                 station travels little; a boat that cannot travels a lot
                 while being pushed around. In one spot-checked cell the
                 successes ran ~21 m and the failures ~76-83 m.
  tts_s          time to first success, recorded only for episodes that
                 succeeded. If waves delay the hold without preventing it,
                 this rises while success rate stays flat.

Reporting these alongside SR turns a flat ladder into a measurable one, using
runs that are already banked -- no extra GPU.

Usage:
    python scripts/wave_ladder_table.py "C:/usvb/gcert_wrl_*.json"
    python scripts/wave_ladder_table.py "C:/usvb/gcert_w*_*.json" --csv out.csv

Pure standard library. No Isaac, no torch, no numpy.
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys


def _quantiles(values):
    """Return (median, p90) with a graceful path for tiny samples."""
    if not values:
        return float("nan"), float("nan")
    ordered = sorted(values)
    median = statistics.median(ordered)
    if len(ordered) < 2:
        return median, ordered[-1]
    index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return median, ordered[index]


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a binomial proportion.

    Reported because the headline claim depends on it: across the wave ladder
    the success rate moves by ~2 points while the effort metric moves ~27%.
    Two points is only meaningful if it is outside the sampling noise, and at
    n=128 it is not -- the interval is roughly +/-8.6 points. Stating that
    explicitly turns "the success rate barely moved" into "the success rate
    CANNOT resolve this dose at this sample size", which is the actual claim.
    """
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, centre - half), min(1.0, centre + half)


def median_interval(values, z=1.96):
    """Distribution-free 95% CI for a median, from order statistics.

    No normality assumption, which matters here: path length is strongly
    right-skewed (a few episodes wander a very long way).
    """
    n = len(values)
    if n < 8:
        return float("nan"), float("nan")
    ordered = sorted(values)
    half = z * math.sqrt(n) / 2.0
    lo = int(math.floor(n / 2.0 - half))
    hi = int(math.ceil(n / 2.0 + half))
    lo = max(0, lo)
    hi = min(n - 1, hi)
    return ordered[lo], ordered[hi]


def _protocol_tag(blob):
    proto = blob.get("scenario_protocol")
    if proto is None:
        return "NO-PROTOCOL"
    if isinstance(proto, str):
        return proto
    version = proto.get("version") if isinstance(proto, dict) else None
    return f"v{version}" if version is not None else "present"


def summarise(path):
    with open(path, "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    records = blob.get("records") or []
    if not records:
        return None
    successes = [r for r in records if r.get("success")]
    failures = [r for r in records if not r.get("success")]

    def _paths(rows):
        return [float(r["path_length_m"]) for r in rows
                if r.get("path_length_m") is not None]

    tts = [float(r["tts_s"]) for r in successes if r.get("tts_s") is not None]
    ok_path_median, _ = _quantiles(_paths(successes))
    bad_path_median, _ = _quantiles(_paths(failures))
    all_path_median, all_path_p90 = _quantiles(_paths(records))
    tts_median, tts_p90 = _quantiles(tts)

    sr_lo, sr_hi = wilson_interval(len(successes), len(records))
    ok_paths = _paths(successes)
    path_lo, path_hi = median_interval(ok_paths)

    return {
        "cell": os.path.splitext(os.path.basename(path))[0],
        "task": blob.get("task", ""),
        "protocol": _protocol_tag(blob),
        "n": len(records),
        "sr": len(successes) / len(records),
        "tts_median_s": tts_median,
        "tts_p90_s": tts_p90,
        "path_median_m": all_path_median,
        "path_p90_m": all_path_p90,
        "path_ok_median_m": ok_path_median,
        "path_fail_median_m": bad_path_median,
        "sr_lo": sr_lo,
        "sr_hi": sr_hi,
        "sr_halfwidth": (sr_hi - sr_lo) / 2.0,
        "path_ok_lo": path_lo,
        "path_ok_hi": path_hi,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+",
                        help="glob(s) over certificate JSONs")
    parser.add_argument("--csv", default=None, help="also write a CSV here")
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("no files matched", file=sys.stderr)
        return 2

    rows = []
    dropped = []
    for path in paths:
        try:
            row = summarise(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{os.path.basename(path)}: UNREADABLE ({exc})", file=sys.stderr)
            dropped.append((os.path.basename(path), "unreadable"))
            continue
        if row:
            rows.append(row)
        else:
            dropped.append((os.path.basename(path), "no records"))

    header = (f"{'cell':<30}{'n':>5}{'SR':>8}{'SR 95% CI':>18}"
              f"{'path_ok':>9}{'path_ok 95% CI':>18}{'tts_med':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        sr_ci = f"[{row['sr_lo']:.3f},{row['sr_hi']:.3f}]"
        p_ci = f"[{row['path_ok_lo']:.1f},{row['path_ok_hi']:.1f}]"
        print(f"{row['cell']:<30}{row['n']:>5}"
              f"{row['sr']:>8.4f}{sr_ci:>18}"
              f"{row['path_ok_median_m']:>9.1f}{p_ci:>18}"
              f"{row['tts_median_s']:>9.1f}")

    off = [r["cell"] for r in rows if r["protocol"] == "NO-PROTOCOL"]
    if off:
        print()
        print(f"WARNING: {len(off)} cell(s) carry NO protocol stamp and must not "
              f"be quoted as dual-seed evidence:")
        for cell in off[:10]:
            print(f"  {cell}")

    if dropped:
        # A cell that is missing from the table above is not a cell that scored
        # zero, and the difference is invisible to whoever reads the log the
        # next morning: the table simply has fewer lines. Name them.
        print()
        print(f"WARNING: {len(dropped)} matched file(s) contributed no row and "
              f"are absent from the table above, which is NOT the same as "
              f"scoring zero:")
        for name, why in dropped[:10]:
            print(f"  {name} ({why})")

    if not rows:
        # Every matched file was unreadable or empty. Falling through used to
        # mean an IndexError on rows[0] in the CSV writer, and on the plain
        # path an empty table with exit 0 -- a sweep whose certificates were
        # all truncated would have reported success to the queue script that
        # launched it. This is the same "nothing to analyse" condition as an
        # empty glob, so it exits the same way (2) rather than inventing a code.
        print()
        print("ERROR: no certificate produced a row -- nothing to tabulate")
        print(f"{len(paths)} file(s) matched, none readable with records",
              file=sys.stderr)
        return 2

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
