"""Shared readers and constants for the real-data checks (stdlib + numpy only).

The data files are not in the repository. Every script takes the data folder as its first
argument; the layout below is the one used on 2026-10-25:

  <data>/HaylingIsland_CCO/extract_2025/HgI_waves2025.txt          CCO Hayling Island wave buoy, 2025 QC file
  <data>/CCO_met/Lymington_met_2025/Lym_met2025.txt                CCO Lymington met station, 2025 QC file
  <data>/CCO_met/SandownPier_met_2025/SdP_met2025.txt              CCO Sandown Pier met station, 2025 QC file
  <data>/Copernicus_currents/speed_Hayling_2025.csv                hourly surface speed at the chosen cell
  <data>/Copernicus_currents/speed_PortsmouthEntrance_2025.csv     (written from the .nc files, see current_cdf.py)
  <data>/Copernicus_currents/cur_Hayling_2025.nc                   cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i subsets
  <data>/Copernicus_currents/cur_PortsmouthEntrance_2025.nc        (3 x 2 cells each, uo / vo, 2025 hourly)

CCO files are tab separated, one header line, time "dd-Mon-YYYY HH:MM:SS" (GMT), 9999 = missing.
"""
import datetime as dt
from pathlib import Path

import numpy as np

MISSING = 9000.0          # CCO writes 9999 / 9999.0 for a missing value; anything >= this is treated as missing

# QC rules used on 2026-10-25 (CCO flag column; values at or above the bound are missing / invalid):
WAVE_HS_FLAGS = (0, 2, 3, 4, 6, 8)   # Hs (and Tz) kept for these flags and Hs < WAVE_MAX_M
WAVE_TP_FLAGS = (0, 3, 4, 6, 8)      # Tp kept for these flags and Tp < WAVE_MAX_S
WAVE_MAX_M = 50.0
WAVE_MAX_S = 50.0
MET_MAX_MPS = 90.0                   # wind records kept when flag == 0, wind < 90 and gust < 90
YEAR_START = dt.datetime(2025, 1, 1)
HOURS_2025 = 8760

# Dose ladders of the journal suite (release v2.6.6, tasks/_shared/conditions/presets_v3.py):
WAVE_HS_DOSES_M = {"D1": 0.125, "D2": 0.25, "D3": 0.5, "D4": 1.0}    # A_HS_M, line 57 (Tp fixed 4 s, line 56)
WIND_U10_DOSES_MPS = {"D1": 3.0, "D2": 6.0, "D3": 9.0, "D4": 12.0}  # B_U10_MPS, line 63
CURRENT_DOSES_MPS = {"D1": 0.25, "D2": 0.5, "D3": 1.0, "D4": 1.5}   # _CURRENT, line 149

WAVE_FILE = "HaylingIsland_CCO/extract_2025/HgI_waves2025.txt"
MET_FILES = {"Lymington": "CCO_met/Lymington_met_2025/Lym_met2025.txt",
             "Sandown Pier": "CCO_met/SandownPier_met_2025/SdP_met2025.txt"}
CURRENT_CSV = {"Hayling": "Copernicus_currents/speed_Hayling_2025.csv",
               "Portsmouth entrance": "Copernicus_currents/speed_PortsmouthEntrance_2025.csv"}
CURRENT_NC = {"Hayling": "Copernicus_currents/cur_Hayling_2025.nc",
              "Portsmouth entrance": "Copernicus_currents/cur_PortsmouthEntrance_2025.nc"}

# column order after the time column
WAVE_COLS = ("lat", "lon", "flag", "hs", "hmax", "tp", "tz", "dirp", "spread", "sst")
MET_COLS = ("wind", "wind_dir", "gust", "baro", "tair", "rain", "solar", "uv", "rh", "flag")


def read_cco(path, cols):
    """Return (times, {col: float array}) for a CCO tab-separated QC file."""
    times, rows = [], []
    with open(path, encoding="latin-1") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(cols) + 1:
                raise ValueError("unexpected column count in %s: %r" % (path, line))
            times.append(dt.datetime.strptime(parts[0].strip(), "%d-%b-%Y %H:%M:%S"))
            rows.append([float(x) for x in parts[1:]])
    a = np.array(rows, dtype=float)
    return times, {c: a[:, i] for i, c in enumerate(cols)}


def read_waves(data):
    return read_cco(Path(data) / WAVE_FILE, WAVE_COLS)


def read_met(data, station):
    return read_cco(Path(data) / MET_FILES[station], MET_COLS)


def read_current_csv(data, site):
    """Hourly speed (m/s) for 2025, row i = 2025-01-01 00:00 GMT + i hours (checked against the .nc time axis)."""
    path = Path(data) / CURRENT_CSV[site]
    lines = path.read_text(encoding="ascii").splitlines()
    header = lines[0]
    s = np.array([float(x) if x.strip().lower() != "nan" else np.nan for x in lines[1:] if x.strip()])
    if len(s) != HOURS_2025:
        raise ValueError("%s: expected %d hourly values, got %d" % (path, HOURS_2025, len(s)))
    return header, s


def hour_index(t):
    """Hour of 2025 (0 = 2025-01-01 00:00-01:00 GMT) that time t falls in."""
    return int((t - YEAR_START).total_seconds() // 3600)


def share_at_or_below(x, value):
    """Fraction of x <= value (the 'percentile position' of a dose)."""
    return float(np.mean(x <= value))


def pct(p):
    return "%.1f%%" % (100.0 * p)
