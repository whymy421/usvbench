#!/usr/bin/env python3
"""10-minute mean wind at the CCO Lymington and Sandown Pier met stations (2025) against the wind doses.

usage: python real_data/wind_cdf.py <data folder>

QC rule (as used on 2026-10-25): a record is kept when its flag (last column) is 0, wind < 90
and gust < 90 m/s. A dose's position is the share of kept records with wind <= the dose. Gust
factor = gust / 10-minute mean over kept records with mean wind > 3 m/s (the D1 dose).
Expected output (2026-10-25 notes, section 6), Lymington / Sandown: about 51 thousand records
each; D1 ~18 / 26 %, D2 ~55 / 61 %, D3 ~81 / 84 %, D4 ~95 / 95 %; gust factor median
1.21 / 1.43, 90th percentile 1.37 / 1.85.
Note: the dose ladder is a 10 m wind speed (U10); the station anemometer heights are UNVERIFIED
(see the CCO metadata PDF), so no height correction is applied.
"""
import sys

import numpy as np

import _common as C


def main(data):
    for station in C.MET_FILES:
        _, m = C.read_met(data, station)
        valid = (m["flag"] == 0) & (m["wind"] < C.MET_MAX_MPS) & (m["gust"] < C.MET_MAX_MPS)
        u = m["wind"][valid]
        print("%s: %d records, %d valid (flag 0); median %.1f m/s" % (station, len(valid), valid.sum(), np.median(u)))
        for d, v in C.WIND_U10_DOSES_MPS.items():
            print("  %s %4.1f m/s: records at or below %s" % (d, v, C.pct(C.share_at_or_below(u, v))))
        g = valid & (m["wind"] > C.WIND_U10_DOSES_MPS["D1"])
        ratio = m["gust"][g] / m["wind"][g]
        p50, p90 = np.percentile(ratio, [50, 90])
        print("  gust factor (mean wind > 3 m/s, n=%d): median %.2f, 90th percentile %.2f" % (g.sum(), p50, p90))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
