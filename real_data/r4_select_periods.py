#!/usr/bin/env python3
"""R4: pick the two real-condition periods (a typical hour and a strong-wind hour) by fixed rules.

usage: python real_data/r4_select_periods.py <data folder>

The rules were written down on 2026-10-25 before any real-condition case was run, and are not to
be changed after the runs:

Eligible hour h (GMT, [h:00, h+1:00)) of 2025 -- all three sources present:
  waves    both half-hour records of the hour have QC flag 0 (and Hs < 50 m, Tp < 50 s);
  wind     Lymington has at least 6 ten-minute records in the hour with flag 0 (wind, gust < 90 m/s);
  current  the Copernicus Hayling cell has a value for the hour.
Hourly values: Hs and Tp = mean of the two half-hour records; wind = mean of the 10-minute means;
gust = maximum gust in the hour; current = the hourly model value.
Reference medians = medians of Hs, wind and current over ALL eligible hours (day and night).
Candidates: eligible daytime hours, starting 08:00 to 17:00 GMT (window 08-18 GMT).
  Typical hour   : smallest |Hs/med_Hs - 1| + |U/med_U - 1| + |c/med_c - 1|; ties -> earliest.
  Strong wind    : every valid 10-minute mean of the hour >= 9 m/s (wind D3) and Hs <= 1.0 m
                   (wave D4, no green water); largest mean wind; ties -> earliest.
Expected output (2026-10-25 notes, section 8): 7627 eligible hours; medians Hs 0.56 m, wind
5.8 m/s, current 0.39 m/s; typical 2025-05-10 09:00-10:00 GMT (Hs 0.54 m, Tp 4.0 s, wind 5.9 m/s,
gust max 13.8, current 0.39); strong wind 2025-06-06 15:00-16:00 GMT (Hs 0.98 m, Tp 3.5 s,
wind 13.4 m/s, gust max 15.9, current 0.43).
"""
import datetime as dt
import sys

import numpy as np

import _common as C

DAY_FIRST_HOUR, DAY_LAST_HOUR = 8, 17          # hours starting 08:00 .. 17:00 GMT
GALE_MIN_10MIN_MPS = C.WIND_U10_DOSES_MPS["D3"]
GALE_MAX_HS_M = C.WAVE_HS_DOSES_M["D4"]
MIN_WIND_RECORDS = 6


def hourly_table(data):
    wt, w = C.read_waves(data)
    mt, m = C.read_met(data, "Lymington")
    _, cur = C.read_current_csv(data, "Hayling")
    waves, met = {}, {}
    for k, t in enumerate(wt):
        waves.setdefault(C.hour_index(t), []).append(k)
    for k, t in enumerate(mt):
        met.setdefault(C.hour_index(t), []).append(k)
    rows = []
    for h in range(C.HOURS_2025):
        wk = waves.get(h, [])
        if len(wk) != 2:
            continue
        if not all(w["flag"][k] == 0 and w["hs"][k] < C.WAVE_MAX_M and w["tp"][k] < C.WAVE_MAX_S for k in wk):
            continue
        mk = [k for k in met.get(h, []) if m["flag"][k] == 0 and m["wind"][k] < C.MET_MAX_MPS and m["gust"][k] < C.MET_MAX_MPS]
        if len(mk) < MIN_WIND_RECORDS or not np.isfinite(cur[h]):
            continue
        u = m["wind"][mk]
        gust = m["gust"][mk]
        rows.append({"h": h, "hs": float(np.mean(w["hs"][wk])), "tp": float(np.mean(w["tp"][wk])),
                     "u": float(np.mean(u)), "u_min": float(u.min()), "gust": float(gust.max()),
                     "cur": float(cur[h])})
    return rows


def describe(r):
    t = C.YEAR_START + dt.timedelta(hours=r["h"])
    return ("%s-%s GMT: Hs %.2f m, Tp %.1f s, wind %.1f m/s (gust max %.1f), current %.2f m/s"
            % (t.strftime("%Y-%m-%d %H:%M"), (t + dt.timedelta(hours=1)).strftime("%H:%M"),
               r["hs"], r["tp"], r["u"], r["gust"], r["cur"]))


def main(data):
    rows = hourly_table(data)
    med = {k: float(np.median([r[k] for r in rows])) for k in ("hs", "u", "cur")}
    print("eligible hours (all three sources, day and night): %d" % len(rows))
    print("reference medians: Hs %.2f m, wind %.1f m/s, current %.2f m/s" % (med["hs"], med["u"], med["cur"]))
    day = [r for r in rows if DAY_FIRST_HOUR <= r["h"] % 24 <= DAY_LAST_HOUR]
    print("daytime candidates (hours starting %02d:00-%02d:00 GMT): %d" % (DAY_FIRST_HOUR, DAY_LAST_HOUR, len(day)))
    score = [abs(r["hs"] / med["hs"] - 1) + abs(r["u"] / med["u"] - 1) + abs(r["cur"] / med["cur"] - 1) for r in day]
    typical = day[int(np.argmin(score))]          # argmin returns the first (earliest) of equal scores
    print("typical hour : " + describe(typical) + "  (score %.3f)" % min(score))
    gale = [r for r in day if r["u_min"] >= GALE_MIN_10MIN_MPS and r["hs"] <= GALE_MAX_HS_M]
    print("strong-wind candidates (every 10-min mean >= %.0f m/s, Hs <= %.1f m): %d"
          % (GALE_MIN_10MIN_MPS, GALE_MAX_HS_M, len(gale)))
    if gale:
        best = gale[int(np.argmax([r["u"] for r in gale]))]   # argmax returns the earliest of equal means
        print("strong wind  : " + describe(best))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
