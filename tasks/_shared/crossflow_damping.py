"""Cross-flow sway and yaw damping of the VRX WAM-V pontoons (G4-A5; pure Python, no imports).

Controller decision G4-A5 (2026-10-01, from the G1 damping sweep variant
"B2q_xf1_quadyaw", 20261001_g1_damping_sweep/work/SWEEP_REPORT.md Part B and
tools_g1/common.py sarda_y_vv / crossflow_yaw_quadratic): the journal default
WAM-V keeps the VRX-classic LINEAR sway (40 N s/m) and yaw (400 N m s/rad)
terms and ADDS

  sway  Y_v|v| = 0.5 rho [1.1 + 0.0045 L/T - 0.1 B/T + 0.016 (B/T)^2] (pi T L / 2)
        (Sarda, Qu, Bertaska, von Ellenrieder, Ocean Eng. 127 (2016) 305-324,
        Table 3, row Y_v, preprint arXiv:1702.04941 p.10; D5 copy
        20260930_d5_vessels_v2/work/evidence/web/sarda2016_text.txt:371-380;
        D5 PARAMS.md section 5), magnitude of Y_v / |v| in N s2/m2;
  yaw   N_r|r| = Y_v|v| L^3 / 32  (strip theory: the same cross-flow
        coefficient per unit length, Y_v|v| / L, integrated as
        int_{-L/2}^{L/2} |x|^3 dx = L^4 / 32 along a hull centred on the COM;
        UNVERIFIED approximation: the VRX pontoon centre is about 0.1 m from the
        COM), N m s2/rad2.

The functions below take the geometry as arguments; the profile layer
(hydro_profiles.py, rule ``swayyaw_damping``) reads every input from the v2
registry (vehicles_v2.py) and stamps formula, inputs and sources. Literature
coefficients live here because hydro_profiles.py carries no vessel number
(scripts/tests_hydro_v2/test_profiles_runtime.py test_no_vessel_numbers_in_hydro_profiles).
"""
from __future__ import annotations

import math

# Sarda et al. 2016 Table 3, row Y_v (sarda2016_text.txt:371-380).
SARDA_Y_V_HALF = 0.5
SARDA_Y_V_CONSTANT = 1.1
SARDA_Y_V_L_OVER_T = 0.0045
SARDA_Y_V_B_OVER_T = -0.1
SARDA_Y_V_B_OVER_T_SQUARED = 0.016
# Strip-theory integral of |x|^3 over [-L/2, L/2] divided by L^4.
STRIP_YAW_DIVISOR = 32.0

SARDA_SOURCE = ("Sarda, Qu, Bertaska, von Ellenrieder, Ocean Eng. 127 (2016) 305-324, Table 3 row Y_v "
                "(preprint arXiv:1702.04941 p.10; D5 evidence/web/sarda2016_text.txt:371-380)")
SARDA_FORMULA = "Y_v|v| = 0.5 rho [1.1 + 0.0045 L/T - 0.1 B/T + 0.016 (B/T)^2] (pi T L / 2)"
STRIP_FORMULA = "N_r|r| = Y_v|v| L^3 / 32"
STRIP_SOURCE = ("strip theory with the Sarda cross-flow coefficient per unit length, hull centred on the COM "
                "(G1 tools_g1/common.py crossflow_yaw_quadratic; UNVERIFIED: VRX pontoon centre ~0.1 m from "
                "the COM)")
DECISION = ("controller decision G4-A5 from the G1 damping sweep, variant B2q_xf1_quadyaw "
            "(20261001_g1_damping_sweep/work/SWEEP_REPORT.md Part B)")


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be a positive finite number")
    return float(value)


def sarda_coefficient(length_m, draft_m, beam_m):
    """Bracket [1.1 + 0.0045 L/T - 0.1 B/T + 0.016 (B/T)^2] of Sarda 2016 Table 3."""
    length_m = _positive(length_m, "hull length")
    draft_m = _positive(draft_m, "draft")
    beam_m = _positive(beam_m, "hull beam")
    b_over_t = beam_m / draft_m
    return (SARDA_Y_V_CONSTANT + SARDA_Y_V_L_OVER_T * length_m / draft_m
            + SARDA_Y_V_B_OVER_T * b_over_t + SARDA_Y_V_B_OVER_T_SQUARED * b_over_t * b_over_t)


def sarda_sway_quadratic(length_m, draft_m, beam_m, water_density_kg_m3):
    """|Y_v| / |v| of Sarda 2016 Table 3 (N s2/m2)."""
    rho = _positive(water_density_kg_m3, "water density")
    coefficient = sarda_coefficient(length_m, draft_m, beam_m)
    return SARDA_Y_V_HALF * rho * coefficient * (math.pi * float(draft_m) * float(length_m) / 2.0)


def strip_yaw_quadratic(sway_quadratic, length_m):
    """N_r|r| = Y_v|v| L^3 / 32 (N m s2/rad2)."""
    sway_quadratic = _positive(sway_quadratic, "sway quadratic damping")
    length_m = _positive(length_m, "hull length")
    return sway_quadratic * length_m ** 3 / STRIP_YAW_DIVISOR


__all__ = ["sarda_coefficient", "sarda_sway_quadratic", "strip_yaw_quadratic", "SARDA_FORMULA", "STRIP_FORMULA",
           "SARDA_SOURCE", "STRIP_SOURCE", "DECISION"]
