#!/usr/bin/env python3
"""Wave height distribution at the CCO Hayling Island buoy (2025) against the wave doses.

usage: python real_data/wave_cdf.py <data folder>

Valid record: Hs, Tp, Tz, peak direction and spread all present (no 9999). The QC flag is not
used as a filter here (the R4 period selection does use it). Percentiles use numpy's default
(linear) interpolation; a dose's position is the share of records with Hs <= the dose.
Expected output (2026-10-25 notes, section 5): 16990 records; 5/25/50/75/90/99 % =
0.20 / 0.35 / 0.54 / 0.90 / 1.33 / 2.31 m; D1 0.3 %, D2 ~12 %, D3 ~46 %, D4 ~80 %.
"""
import sys

import numpy as np

import _common as C


def main(data):
    _, w = C.read_waves(data)
    valid = np.ones(len(w["hs"]), bool)
    for c in ("hs", "tp", "tz", "dirp", "spread"):
        valid &= w[c] < C.MISSING
    hs = w["hs"][valid]
    print("CCO Hayling Island wave buoy, 2025 QC file: %d records, %d valid" % (len(valid), valid.sum()))
    q = [5, 25, 50, 75, 90, 99]
    print("Hs percentiles (m): " + ", ".join("%d%% %.2f" % (p, v) for p, v in zip(q, np.percentile(hs, q))))
    for d, v in C.WAVE_HS_DOSES_M.items():
        s = C.share_at_or_below(hs, v)
        print("  %s Hs %.3f m: records at or below the dose %s, above it %s" % (d, v, C.pct(s), C.pct(1 - s)))
    print("share of records with Hs > 1.0 m (above D4): %s" % C.pct(float(np.mean(hs > 1.0))))
    tp = w["tp"][valid]
    print("Tp percentiles (s): " + ", ".join("%d%% %.1f" % (p, v) for p, v in zip(q, np.percentile(tp, q))))
    edges = [0, 3, 5, 7, 10, 16, 30]
    hist = np.histogram(tp, bins=edges)[0]
    print("Tp histogram: " + ", ".join("%d-%d s %s" % (a, b, C.pct(n / len(tp))) for a, b, n in zip(edges, edges[1:], hist)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
