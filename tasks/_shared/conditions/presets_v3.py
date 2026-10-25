"""Journal dose presets (schema 3, hydro_v2) as named, resolvable data (E3).

Names: ``journal.<suite>[.<axis>].D<k>`` (k = 0..4), plus the ladder-B wave
alternative ``journal.waves.B.D<k>`` and the combinations
``journal.combo.<pair>.D<k>``. Every preset carries provenance text. The
wave ladder A (CONTEXT_R3 decision 8) is owner-decided (2026-10-01, B1); the
other ladders (decisions 9, 11) stay PROVISIONAL until the dose freeze after
the full-dose smoke (STATUS); PRESETS_E3.md lists them with sources.

Sources (read-only inputs of this task):
  C7  D:/claude code/codex_tasks/20260930_c7_strategy_redteam/work/STRATEGY.md
      section 5 (lines 173-219)
  D9  D:/claude code/codex_tasks/20260930_d9_waves_v2_doses/work/DOSE_DESIGN.md
      and presets/wave_presets_v2.py
  R3  D:/claude code/codex_tasks/20260930_nc_shared/CONTEXT_R3.md decisions 8, 9, 11

Admission. Every wave preset is checked when this module is imported
(``_build``): DNV-RP-C205 3.5.4 Sp <= 1/15 and Ss <= 1/10 with the
gamma-consistent Tz, plus the generator's 1/7 guard. A failing preset makes
the import fail (fail closed). ``ADMISSION`` holds the computed values.

F0 (CONTEXT_R4 D4, "perception x4 + bundle"): the labelled sensor bundle
``journal.perception.bundle.D<k>`` = velocity white noise {0, .02, .05, .10,
.20} m/s on surge/sway plus range white noise {0, .01, .03, .10, .30} m, the
C7 STRATEGY.md:219 bundle values (C7: a bundle, never attributed to position
noise). The yaw-rate component stays 0 (C7:219 forbids reusing a linear
velocity number and names no yaw-rate value; PENDING owner decision; it can
be set explicitly with ``perception.yaw_rate_noise_sigma_rad_s``).

G10 (owner decision 2026-10-01, NC 34_full-dose smoke plan section 7): the
supplementary wave preset ``journal.waves.supp075`` (Hs 0.75 m, Tp 4 s, every
other setting as ladder A) measures the BlueBoat operating boundary between
A3 (0.5 m) and A4 (1.0 m) on a few tasks. It is a SUPPLEMENT, never a
main-table level: its entry carries ``role = "SUPPLEMENT"`` and
``main_table = False`` (stamped by provenance() and the green-water header),
and it is selected by name (``--suite journal.waves.supp075``), like the
combination presets. The main ladder is unchanged.

D0 semantics. D0 is the zero added dose. Waves (Hs = 0), current (0 m/s),
actuator (all neutral) and perception (all zero) D0 presets are ENABLED with
a zero dose and are bit-identical to the disabled suite (tested). Wind D0 is
the module OFF (decision 9: zero dose is not U = 0).
"""
from __future__ import annotations

from copy import deepcopy
import math

from .config_v3 import WavesV3, check_steepness, steepness

STATUS = "PROVISIONAL"
REGISTRY = "usvbench.journal_presets.v1"
LEVELS = (0, 1, 2, 3, 4)

# ------------------------------------------------------------------ waves --
A_TP_S = 4.0
A_HS_M = (0.0, 0.125, 0.25, 0.5, 1.0)
GAMMA = 3.3
SPREADING_RAD = math.pi / 4.0
BAND_FP = (0.2, 4.0)
N_COMPONENTS = 1024
RECURRENCE_MIN_S = 480.0     # 2 x the longest episode (harbour, 240 s; D9 DOSE_DESIGN.md:44)
B_U10_MPS = (0.0, 3.0, 6.0, 9.0, 12.0)
B_FETCH_M = 25000.0
B_PRIOR_DURATION_S = 36000.0
CALM_TP_PLACEHOLDER_S = 4.0
G = 9.81


def cem_fetch_limited(u10_mps, fetch_m=B_FETCH_M, prior_duration_s=B_PRIOR_DURATION_S):
    """CEM EM 1110-2-1100 Part II Eqs. II-2-36/37/38 (deep water, steady wind).

    Transcribed from D9 presets/wave_presets_v2.py:68-107 (same arithmetic):
    CD = 0.001(1.1 + 0.035 U10), u*^2 = CD U10^2; g Hm0/u*^2 = 4.13e-2
    (gX/u*^2)^(1/2); g Tp/u* = 0.651 (gX/u*^2)^(1/3); caps 211.5 / 239.8;
    duration limit gX/u*^2 = 5.23e-3 (g t/u*)^(3/2). Returns (hs, tp) or
    (0, None) for U10 = 0.
    """
    if u10_mps == 0:
        return 0.0, None
    cd = 0.001 * (1.1 + 0.035 * u10_mps)
    u_star = u10_mps * math.sqrt(cd)
    xhat = G * fetch_m / u_star ** 2
    xhat_duration = 5.23e-3 * (G * prior_duration_s / u_star) ** 1.5
    effective = min(xhat, xhat_duration)
    hs = min(0.0413 * u_star ** 2 / G * math.sqrt(effective), 211.5 * u_star ** 2 / G)
    tp = min(0.651 * u_star / G * effective ** (1.0 / 3.0), 239.8 * u_star / G)
    return hs, tp


def _waves(hs, tp):
    return {"waves": {"enabled": True, "hs_m": float(hs), "tp_s": float(tp), "gamma": GAMMA,
                      "spreading_rad": SPREADING_RAD, "band_min_over_fp": BAND_FP[0],
                      "band_max_over_fp": BAND_FP[1], "band_policy": "raise_to_vessel",
                      "n_components": N_COMPONENTS, "recurrence_min_s": RECURRENCE_MIN_S,
                      "direction_mode": "strata", "steepness_rule": "dnv_rp_c205_3_5_4"}}


JOURNAL_PRESETS = {}


def _add(name, suite, axis, level, values, provenance):
    if name in JOURNAL_PRESETS:
        raise ValueError("duplicate preset " + name)
    JOURNAL_PRESETS[name] = {"suite": suite, "axis": axis, "level": level, "values": values,
                             "provenance": provenance, "status": STATUS}


for _k, _hs in enumerate(A_HS_M):
    _add(f"journal.waves.D{_k}", "waves", "hs_ladder_A", _k, _waves(_hs, A_TP_S),
         "Ladder A (owner decision 2026-10-01, B1 / D-1): D9 DOSE_DESIGN.md section 5 / "
         "wave_presets_v2.py:41-42,168-176 and CONTEXT_R3 decision 8; JONSWAP gamma 3.3, "
         "uniform spreading +/-45 deg (DNV cos^4 variance), band [0.2, 4.0] fp raised to the "
         "vessel requirement, 1024 components; anchors WMO 2/3, UFC 4-152-07, BlueBoat 1 m.")

# G10: supplementary BlueBoat boundary level (never a main-table level; selected by name).
SUPP075_HS_M = 0.75
SUPPLEMENT_PRESETS = ("journal.waves.supp075",)
_add("journal.waves.supp075", "waves", "hs_supplement", None, _waves(SUPP075_HS_M, A_TP_S),
     "SUPPLEMENT (never a main-table level; owner decision 2026-10-01, NC 34_full-dose smoke plan section 7): "
     "BlueBoat operating-boundary level between ladder A3 (0.5 m) and A4 (1.0 m), Hs 0.75 m at the ladder-A "
     "period Tp 4 s; JONSWAP gamma 3.3, uniform spreading +/-45 deg, band [0.2, 4.0] fp raised to the vessel "
     "requirement, 1024 components, group repeat >= 480 s, compass-8 strata: every setting and rule of "
     "ladder A (D9 DOSE_DESIGN.md section 5). DNV-RP-C205 3.5.4 admission at import: Sp = 0.0300 <= 1/15, "
     "Ss = 0.0496 <= 1/10 (Tz = 3.111 s).")
JOURNAL_PRESETS["journal.waves.supp075"].update(role="SUPPLEMENT", main_table=False)

for _k, _u10 in enumerate(B_U10_MPS):
    _hs, _tp = cem_fetch_limited(_u10)
    _add(f"journal.waves.B.D{_k}", "waves", "cem_ladder_B", _k,
         _waves(_hs, CALM_TP_PLACEHOLDER_S if _tp is None else _tp),
         "Ladder B (alternative): CEM EM 1110-2-1100 Part II Eqs. II-2-36/37/38 fetch-limited "
         "wind sea, U10 = %g m/s, fetch 25 km, 10 h preceding wind (D9 DOSE_DESIGN.md section 5, "
         "wave_presets_v2.py:43-45,178-193)%s." % (
             _u10, "; D0 Tp is a placeholder" if _tp is None else ""))

_WIND_U10 = (0.0, 3.0, 6.0, 9.0, 12.0)
for _k, _u10 in enumerate(_WIND_U10):
    values = ({"wind": {"enabled": False}} if _u10 == 0.0 else
              {"wind": {"enabled": True, "u10_mps": _u10, "reference": "u10", "gust_intensity": 0.15,
                        "gust_correlation_time_s": 2.0, "direction_mode": "strata"}})
    _add(f"journal.wind.D{_k}", "wind", "u10", _k, values,
         "C7 STRATEGY.md:180 ladder 0/3/6/9/12 m/s re-based to U10 (CONTEXT_R3 decision 9, "
         "PENDING): neutral log profile z0 = 2e-4 m to the vessel lateral-centroid height (D5 "
         "registry); gusts C7:180 SD = 15 % of the mean, correlation time 2 s (proposed test "
         "parameters, not measured climatology); Beaufort 0/2/4/5/6 anchor (Met Office). "
         "D0 = module OFF.")

_CURRENT = (0.0, 0.25, 0.50, 1.00, 1.50)
for _k, _u in enumerate(_CURRENT):
    _add(f"journal.current.D{_k}", "current", "speed", _k,
         {"current": {"enabled": True, "speed_mps": _u, "direction_mode": "strata"}},
         "C7 STRATEGY.md:181 ladder 0/0.25/0.50/1.00/1.50 m/s (0.49/0.97/1.94/2.92 kn), "
         "sheltered/coastal test magnitudes linked to NOAA current information; uniform "
         "water velocity (CONTEXT_R3 decision 10).")

_ACTUATOR = {
    "cap": ("thrust_cap", (1.0, 0.90, 0.75, 0.50, 0.25),
            "C7 STRATEGY.md:199 available fraction of BOTH per-motor forward and reverse limits"),
    "gain": ("motor_gain_one_sided", (1.0, 0.95, 0.85, 0.70, 0.50),
             "C7 STRATEGY.md:201 one-sided delivered-thrust gain, healthy motor exactly 1, "
             "failing side balanced over scenario-ordinal pairs"),
    "lag": ("additional_lag_s", (0.0, 0.05, 0.10, 0.25, 0.50),
            "C7 STRATEGY.md:200 additional first-order time constant (s), cascaded after the "
            "nominal motor pole"),
    "resistance": ("resistance_scale", (1.0, 1.10, 1.25, 1.50, 2.00),
                   "C7 STRATEGY.md:198 multiplier on surge/sway/yaw damping (fouling mechanism, "
                   "Schultz 2007); heave/roll/pitch unchanged"),
}
for _axis, (_field, _ladder, _text) in _ACTUATOR.items():
    for _k, _value in enumerate(_ladder):
        _add(f"journal.actuator.{_axis}.D{_k}", "actuator", _axis, _k,
             {"actuator": {"enabled": True, _field: _value}},
             _text + "; CONTEXT_R3 decision 11 (four live axes under hydro_v2; centred mass "
             "and eccentric payload rejected).")

_PERCEPTION = {
    "noise": ("pos_noise_sigma_m", (0.0, 0.02, 0.10, 0.40, 1.00),
              "C7 STRATEGY.md:214 per-axis position-estimate noise sigma (m), correlation "
              "time 0.2 s (stationary Gauss-Markov); u-blox F9P anchor (CEP = 1.177 sigma)"),
    "bias": ("pos_bias_m", (0.0, 0.05, 0.20, 0.50, 1.00),
             "C7 STRATEGY.md:215 fixed position-bias magnitude (m) with scenario-balanced "
             "direction (compass-8 strata relative to the task axis)"),
    "latency": ("latency_s", (0.0, 0.05, 0.10, 0.25, 0.50),
                "C7 STRATEGY.md:216 added latency (s), applied consistently to every "
                "observed estimate group; ArduPilot EKF2 delay anchor"),
    "dropout": ("outage_s", (0.0, 0.10, 0.30, 1.00, 3.00),
                "C7 STRATEGY.md:217 one contiguous outage (s) holding the last estimate; start at the "
                "task's critical phase (station keeping) or, every other task, at f x T_ref with "
                "T_ref = reference route length / reference speed and f keyed uniform in [0.40, 0.60] "
                "(G28 rule, owner decision 2026-10-04)"),
}
for _axis, (_field, _ladder, _text) in _PERCEPTION.items():
    for _k, _value in enumerate(_ladder):
        values = {"perception": {"enabled": True, _field: _value}}
        if _axis == "noise":
            values["perception"]["pos_noise_correlation_s"] = 0.2
        _add(f"journal.perception.{_axis}.D{_k}", "perception", _axis, _k, values, _text + ".")

# F0: labelled sensor bundle (C7 STRATEGY.md:219), velocity + range white noise.
_BUNDLE_VEL_MPS = (0.0, 0.02, 0.05, 0.10, 0.20)
_BUNDLE_RAY_M = (0.0, 0.01, 0.03, 0.10, 0.30)
for _k, (_vel, _ray) in enumerate(zip(_BUNDLE_VEL_MPS, _BUNDLE_RAY_M)):
    _add(f"journal.perception.bundle.D{_k}", "perception", "bundle", _k,
         {"perception": {"enabled": True, "vel_noise_sigma_mps": _vel, "ray_noise_sigma_m": _ray}},
         "C7 STRATEGY.md:219 labelled sensor bundle: velocity sigma {.02,.05,.10,.20} m/s (surge and "
         "sway), ray sigma {.01,.03,.10,.30} m; yaw-rate sigma 0 (C7 gives no yaw-rate value and forbids "
         "reusing the linear one; PENDING owner decision). F0 addition (CONTEXT_R4 D4).")

COMBINATIONS = {
    "journal.combo.wave_wind.D2": ("journal.waves.D2", "journal.wind.D2"),
    "journal.combo.wave_wind.D3": ("journal.waves.D3", "journal.wind.D3"),
    "journal.combo.wave_current.D2": ("journal.waves.D2", "journal.current.D2"),
    "journal.combo.wave_current.D3": ("journal.waves.D3", "journal.current.D3"),
    "journal.combo.wave_wind_local.D2": ("journal.waves.B.D2", "journal.wind.D2"),
    "journal.combo.wave_wind_local.D3": ("journal.waves.B.D3", "journal.wind.D3"),
}
COMBINATION_METADATA = {
    "journal.combo.wave_wind.D2": ("PRIMARY", "D9 combo.wave_wind.D2: A2 (0.25 m, 4 s) + U10 6 m/s; "
                                   "C3 class B; shared A2 marginal (D9 D-7)"),
    "journal.combo.wave_wind.D3": ("PRIMARY", "D9 combo.wave_wind.D3: A3 (0.50 m, 4 s) + U10 9 m/s"),
    "journal.combo.wave_current.D2": ("PRIMARY", "D9 combo.wave_current.D2: A2 + 0.50 m/s; "
                                      "no wave-current interaction modelled"),
    "journal.combo.wave_current.D3": ("PRIMARY", "D9 combo.wave_current.D3: A3 + 1.00 m/s"),
    "journal.combo.wave_wind_local.D2": ("ALTERNATIVE", "D9 combo.wave_wind_local.D2: B2 + U10 6 m/s, "
                                         "local CEM wind sea"),
    "journal.combo.wave_wind_local.D3": ("ALTERNATIVE", "D9 combo.wave_wind_local.D3: B3 + U10 9 m/s; "
                                         "outside the provisional WAM-V wetness envelope"),
}
COMBINATION_RULE = ("components share the scenario keys and the compass-8 stratum, so the partner "
                    "flows co-directed with the mean wave direction (D9 DOSE_DESIGN.md section 9); "
                    "run (0,0), (wave,0), (0,partner), (wave,partner) with common keys (C7:96)")


def expand_presets(names):
    """[(component_name, suite_values)] for preset or combination names."""
    result = []
    for name in names:
        if name in COMBINATIONS:
            for component in COMBINATIONS[name]:
                result.append((component, deepcopy(JOURNAL_PRESETS[component]["values"])))
        elif name in JOURNAL_PRESETS:
            result.append((name, deepcopy(JOURNAL_PRESETS[name]["values"])))
        else:
            raise ValueError(f"unknown journal condition preset: {name}")
    return result


def provenance(name):
    """JSON-native provenance record of a preset or combination."""
    if name in COMBINATIONS:
        role, text = COMBINATION_METADATA[name]
        return {"preset": name, "role": role, "provenance": text, "rule": COMBINATION_RULE,
                "components": [provenance(c) for c in COMBINATIONS[name]], "status": STATUS}
    entry = JOURNAL_PRESETS[name]
    record = {"preset": name, "suite": entry["suite"], "axis": entry["axis"], "level": entry["level"],
              "values": deepcopy(entry["values"]), "provenance": entry["provenance"],
              "status": entry["status"], "registry": REGISTRY}
    if "role" in entry:          # G10: only the supplementary preset carries a role (others unchanged)
        record.update(role=entry["role"], main_table=entry["main_table"])
    return record


def _build():
    admission = {}
    for name, entry in JOURNAL_PRESETS.items():
        waves = entry["values"].get("waves")
        if waves is None:
            continue
        cfg = WavesV3(**waves)
        check_steepness(cfg)                       # raises on failure (fail closed)
        admission[name] = dict(steepness(cfg.hs_m, cfg.tp_s, cfg.gamma), hs_m=cfg.hs_m, tp_s=cfg.tp_s)
    for name, components in COMBINATIONS.items():
        suites = [JOURNAL_PRESETS[c]["suite"] for c in components]
        if len(set(suites)) != len(suites):
            raise ValueError("a combination must pair different suites: " + name)
    return admission


ADMISSION = _build()

__all__ = ["JOURNAL_PRESETS", "COMBINATIONS", "COMBINATION_METADATA", "ADMISSION", "LEVELS",
           "expand_presets", "provenance", "cem_fetch_limited", "STATUS", "REGISTRY", "SUPPLEMENT_PRESETS"]
