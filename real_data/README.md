# Real-data checks (UK south coast, 2025)

Small scripts that place the journal dose ladders (release v2.6.6,
`tasks/_shared/conditions/presets_v3.py`) inside one year of real
conditions off Hayling Island / the Portsmouth harbour entrance, and pick the
two real-condition periods (rule set "R4") before any real-condition case is
run. They reproduce the analysis of 2026-10-25.

**No data is in git.** Each script takes the data folder as its only
argument; the expected file layout is in `_common.py`. Requirements: Python 3
and numpy (`current_cdf.py --netcdf` also needs xarray with a netCDF4/HDF5
engine).

```
python real_data/wave_cdf.py          <data folder>
python real_data/wind_cdf.py          <data folder>
python real_data/current_cdf.py       <data folder> [--netcdf]
python real_data/r4_select_periods.py <data folder>
```

| Script | Source | What it prints |
|---|---|---|
| `wave_cdf.py` | Channel Coastal Observatory (CCO) Hayling Island directional wave buoy, 2025 QC file (half-hourly). Open Government Licence. | Hs percentiles and the share of records at or below each wave dose (Hs 0.125 / 0.25 / 0.5 / 1.0 m), Tp spread. |
| `wind_cdf.py` | CCO Lymington and Sandown Pier met stations, 2025 QC files (10-minute means). | Share of records at or below each wind dose (3 / 6 / 9 / 12 m/s), gust factor. |
| `current_cdf.py` | Copernicus Marine `cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i` (model, 1.5 km, hourly surface current), 2025, two 3 x 2-cell subsets. | Speed percentiles and the share of hours at or below each current dose (0.25 / 0.5 / 1.0 / 1.5 m/s). |
| `r4_select_periods.py` | the three above, aligned per hour | The typical hour and the strong-wind hour chosen by the R4 rules (rules in the script's docstring). |

## Results on the 2026-10-25 data (checked by running the scripts)

| Quantity | Result |
|---|---|
| Waves, valid records | 16,990 of 17,520 |
| Hs 5 / 25 / 50 / 75 / 90 / 99 % | 0.20 / 0.35 / 0.54 / 0.90 / 1.33 / 2.31 m |
| Wave doses D1 / D2 / D3 / D4, share at or below | 0.3 / 11.7 / 45.6 / 80.0 % |
| Wind doses D1-D4, Lymington | 18.0 / 54.8 / 81.3 / 95.0 % (51,204 records) |
| Wind doses D1-D4, Sandown Pier | 25.7 / 60.6 / 83.8 / 94.7 % (51,931 records) |
| Gust factor median / 90 %, Lymington / Sandown | 1.21 / 1.37, 1.43 / 1.85 |
| Current median / 99 % / max, Hayling / Portsmouth entrance | 0.39 / 0.67 / 0.76, 0.30 / 0.64 / 0.73 m/s |
| Current doses D1 / D2, share at or below, Hayling / Portsmouth | 10.5 / 84.3 %, 39.1 / 86.4 %; D3, D4 never reached |
| R4 eligible hours (day and night) / daytime candidates | 7,627 / 3,183 |
| R4 reference medians | Hs 0.56 m, wind 5.8 m/s, current 0.39 m/s |
| R4 typical hour | 2025-05-10 09:00-10:00 GMT: Hs 0.54 m, Tp 4.0 s, wind 5.9 m/s (gust max 13.8), current 0.39 m/s |
| R4 strong-wind hour | 2025-06-06 15:00-16:00 GMT: Hs 0.98 m, Tp 3.5 s, wind 13.4 m/s (gust max 15.9), current 0.43 m/s |

## Notes

- These scripts were rewritten for the repository; the original one-off
  analysis code was not kept. Two filters were chosen because they reproduce
  the recorded numbers exactly and are stated in the scripts: a wave record is
  valid when Hs, Tp, Tz, peak direction and spread are all present (the QC flag
  is not a filter for the wave distribution), and the gust factor uses
  10-minute records with mean wind above 3 m/s.
- The current CSV files hold the hourly speed of the chosen cell, rounded to
  4 decimals; `--netcdf` recomputes the cell choice (largest annual mean speed
  in each subset) and the speeds from the `.nc` files (largest difference
  5e-5 m/s).
- The wind ladder is a 10 m wind speed; the station anemometer heights have
  not been checked yet, so no height correction is applied.
- The open-water currents are model output; the much stronger tidal streams in
  the narrow harbour entrances are not resolved by a 1.5 km grid.
