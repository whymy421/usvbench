#!/usr/bin/env python3
"""Hourly surface current speed (Copernicus NWS model, 2025) at two cells against the current doses.

usage: python real_data/current_cdf.py <data folder> [--netcdf]

Product: cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i (NWSHELF analysis/forecast, 1.5 km, hourly
surface uo / vo), 2025-01-01 00:00 to 2025-12-31 23:00 GMT, two 3 x 2-cell subsets: near the
Hayling Island buoy and off the Portsmouth harbour entrance. In each subset the cell with the
largest annual mean speed is used; its hourly speed is stored in speed_<site>_2025.csv.

Default: read the two CSV files (numpy only). --netcdf: recompute the cell choice and the speeds
from the .nc files (needs xarray with an HDF5/netCDF4 engine) and report the largest difference
to the CSV (the CSV is rounded to 4 decimals).
Expected output (2026-10-25 notes, section 7), Hayling / Portsmouth entrance: median 0.39 / 0.30,
99th percentile 0.67 / 0.64, maximum 0.76 / 0.73 m/s; D1 ~10 / 39 %, D2 ~84 / 86 %; D3 and D4 are
never reached.
"""
import sys
from pathlib import Path

import numpy as np

import _common as C


def from_netcdf(data, site):
    import xarray as xr  # optional dependency, only for --netcdf
    ds = xr.open_dataset(Path(data) / C.CURRENT_NC[site])
    speed = np.sqrt(ds["uo"] ** 2 + ds["vo"] ** 2).squeeze()
    mean = speed.mean("time").values
    i, j = np.unravel_index(np.nanargmax(mean), mean.shape)
    cell = speed.isel(latitude=i, longitude=j)
    t0, t1 = str(ds["time"].values[0])[:16], str(ds["time"].values[-1])[:16]
    return float(cell.latitude), float(cell.longitude), cell.values.astype(float), (t0, t1, speed.sizes["time"])


def main(data, use_nc):
    for site in C.CURRENT_CSV:
        header, s = C.read_current_csv(data, site)
        print("%s: %s" % (site, header.lstrip("# ")))
        if use_nc:
            lat, lon, s_nc, (t0, t1, n) = from_netcdf(data, site)
            print("  netcdf: %d hours %s .. %s; chosen cell %.4fN %.4fE; max |nc - csv| = %.1e m/s"
                  % (n, t0, t1, lat, lon, np.nanmax(np.abs(s_nc - s))))
        v = s[np.isfinite(s)]
        p50, p99 = np.percentile(v, [50, 99])
        print("  %d hours: median %.2f, 99th percentile %.2f, maximum %.2f m/s" % (len(v), p50, p99, v.max()))
        for d, x in C.CURRENT_DOSES_MPS.items():
            print("  %s %.2f m/s: hours at or below %s%s" % (d, x, C.pct(C.share_at_or_below(v, x)),
                                                        "  (never reached)" if v.max() < x else ""))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--netcdf"]
    if len(args) != 1:
        sys.exit(__doc__)
    main(args[0], "--netcdf" in sys.argv[1:])
