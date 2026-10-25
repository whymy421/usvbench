"""Diagnostic-only override of the wind-load coefficients (G16, T1.20). Pure Python.

The wind suite applies the first-harmonic Fossen law of tasks/_shared/wind.py
(FossenCoefficients): C_X = -C_X0 cos(gamma), C_Y = C_Y0 sin(gamma),
C_N = C_N0 sin(2 gamma), with the vessel's PINNED coefficients from the D5 registry
(tasks/_shared/vehicles_v2.py Windage.c_x0 / c_y0 / c_n0). A diagnostic run may
replace any of the three with suite tokens of the pseudo-suite ``windcoef``:

    --suite wind=D4 windcoef.c_n0=-0.2

and may switch on the CENTROID-OFFSET yaw term (G16 follow-up), default off:

    --suite wind=D4 windcoef.c_n0=0.11 windcoef.offset_sl=0.039

which adds the lever term of Blendermann (1994), C_N = [s_L/L - 0.18 (gamma - pi/2)] C_Y,
that the first harmonic cannot carry: C_N(gamma) = C_N0 sin(2 gamma) + (s/L) C_Y(gamma), with
s/L the lateral-area centroid ahead of the COM over the yaw reference length (the moment of the
whole lateral force acting at the centroid, N = s F_y; G16 COEFFICIENTS.md sections 3.2 and 4;
measured s/L: WAM-V 0.039, BlueBoat 0.035, vehicles_v2_data lateral_centroid_x_from_com_m / L).
Off, the backend keeps wind.py's FossenCoefficients (byte-identical behaviour).

Rules (fail closed):
  * only the names c_x0, c_y0, c_n0, offset_sl; JSON numbers only (no bool, no string), finite;
    c_x0 and c_y0 nonnegative (wind.py _validate_cfg); |coefficient| <= 5 and |offset_sl| <= 0.5
    (sanity bounds); one value per name (a repeated name with another value is a C1 conflict);
  * the tokens are NOT part of the schema-3 condition configuration: the parameter
    digest, the JournalConditions stamp and the condition stamps of a run at the
    pinned values are unchanged; the override reaches the condition layer through
    ``env_cfg.journal_wind_coefficients`` (set ONLY when an override is given);
  * eval_protocol accepts them only in DIAGNOSTIC runs (seed-registry purpose
    "diagnostic", e.g. DIAG42; reference_environment.is_diagnostic) and refuses
    them in every official run (selection, pilot, calibration, final_test), before
    the simulator starts; run_cell (which plans official purposes only) refuses them;
    cert_v3 refuses to certify an official run whose coefficients are overridden;
    validate_cert_v3 reports such a certificate as an error (non-official);
  * every journal certificate header carries ``wind_coefficients`` (schema
    usvbench.wind_coefficients/2): the pinned values, the values actually used, the
    centroid-offset term (enabled, s/L, form), whether the law is the pinned one, the
    requested override and whether wind is enabled (with wind disabled the
    coefficients are inert and the stamp says so).
"""
from __future__ import annotations

import json
import math

PSEUDO_SUITE = "windcoef"
NAMES = ("c_x0", "c_y0", "c_n0")
OFFSET = "offset_sl"
ALL_NAMES = NAMES + (OFFSET,)
NONNEGATIVE = ("c_x0", "c_y0")
BOUND = 5.0
OFFSET_BOUND = 0.5
STAMP_SCHEMA = "usvbench.wind_coefficients/2"
OFFSET_FORM = ("C_N(gamma) = C_N0 sin(2 gamma) + (s/L) C_Y(gamma), C_Y = C_Y0 sin(gamma): the lever term "
               "s_L/L of Blendermann (1994) C_N = [s_L/L - 0.18 (gamma - pi/2)] C_Y, moment about the COM")
DIAGNOSTIC_RULE = ("windcoef.* overrides are diagnostic only (seed-registry purpose 'diagnostic', e.g. DIAG42); "
                   "official runs are refused before the simulator starts and an official certificate with "
                   "overridden coefficients is invalid (G16; CONTEXT_R4 D23/D28 pattern)")


class WindCoefficientError(ValueError):
    """An invalid or conflicting wind-coefficient override."""


def is_override_token(token):
    """True for a ``windcoef.<name>=<value>`` token (string test only, no validation)."""
    if not isinstance(token, str) or "=" not in token:
        return False
    key = token.split("=", 1)[0].strip()
    return key.split(".", 1)[0] == PSEUDO_SUITE


def _number(name, value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise WindCoefficientError(f"{PSEUDO_SUITE}.{name}: {value!r} is not a JSON number") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WindCoefficientError(f"{PSEUDO_SUITE}.{name}: {value!r} is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise WindCoefficientError(f"{PSEUDO_SUITE}.{name} must be finite")
    bound = OFFSET_BOUND if name == OFFSET else BOUND
    if abs(value) > bound:
        raise WindCoefficientError(f"{PSEUDO_SUITE}.{name} = {value!r} exceeds the sanity bound {bound}")
    if name in NONNEGATIVE and value < 0.0:
        raise WindCoefficientError(f"{PSEUDO_SUITE}.{name} must be nonnegative (wind.py _validate_cfg)")
    return value


def validate_override(override):
    """Return a canonical {name: float} (sorted by name) or None for no override."""
    if override is None:
        return None
    if not isinstance(override, dict) or not override:
        raise WindCoefficientError("a wind-coefficient override is a nonempty mapping")
    out = {}
    for name in sorted(override):
        if name not in ALL_NAMES:
            raise WindCoefficientError(f"unknown wind coefficient {name!r}; allowed: {', '.join(ALL_NAMES)}")
        out[name] = _number(name, override[name])
    return out


def split_tokens(tokens):
    """(other tokens in order, override {name: float} or None) from a token list."""
    rest, override = [], {}
    for token in tokens or ():
        if not is_override_token(token):
            rest.append(token)
            continue
        key, raw = (part.strip() for part in token.split("=", 1))
        parts = key.split(".")
        if len(parts) != 2 or parts[1] not in ALL_NAMES:
            raise WindCoefficientError(f"{token!r}: use {PSEUDO_SUITE}.<{'|'.join(ALL_NAMES)}>=<number>")
        value = _number(parts[1], raw)
        if parts[1] in override and override[parts[1]] != value:
            raise WindCoefficientError(f"declared conflict C1 (two values for one field): {key} given twice "
                                       f"({override[parts[1]]!r} and {value!r})")
        override[parts[1]] = value
    return rest, validate_override(override) if override else None


def coefficient_override(override):
    """The c_x0 / c_y0 / c_n0 part of a validated override (dict, possibly empty)."""
    return {k: v for k, v in (validate_override(override) or {}).items() if k in NAMES}


def offset_value(override):
    """s/L of the centroid-offset term, or None when the term is off."""
    return (validate_override(override) or {}).get(OFFSET)


def pinned_values(windage):
    """{name: float} of a registry Windage (vehicles_v2) or a C3 windage-kwargs mapping."""
    get = windage.get if isinstance(windage, dict) else (lambda key: getattr(windage, key))
    return {name: float(get(name)) for name in NAMES}


def used_values(pinned, override):
    used = dict(pinned)
    used.update(coefficient_override(override))
    return used


class OffsetFossenCoefficients:
    """FossenCoefficients plus the centroid-offset lever term (s/L) C_Y (diagnostic option).

    cx and cy are computed exactly as wind.py FossenCoefficients does; cn adds offset_sl * cy.
    No torch import: tensor methods only."""

    def __init__(self, c_x0, c_y0, c_n0, offset_sl):
        self.c_x0, self.c_y0, self.c_n0, self.offset_sl = float(c_x0), float(c_y0), float(c_n0), float(offset_sl)

    def __call__(self, gamma):
        cx = -self.c_x0 * gamma.cos()
        cy = self.c_y0 * gamma.sin()
        cn = self.c_n0 * (2.0 * gamma).sin() + self.offset_sl * cy
        return cx, cy, cn


def stamp(vessel_profile, pinned, override, wind_enabled):
    """The header block ``wind_coefficients`` (JSON-native)."""
    override = validate_override(override)
    used = used_values(pinned, override)
    offset = offset_value(override)
    return {
        "schema": STAMP_SCHEMA,
        "vessel_profile": vessel_profile,
        "pinned": {name: float(pinned[name]) for name in NAMES},
        "used": {name: float(used[name]) for name in NAMES},
        "centroid_offset": {"enabled": offset is not None, "offset_sl": offset,
                            "form": OFFSET_FORM if offset is not None else "off (pinned law: C_N0 sin(2 gamma) only)"},
        "pinned_values": offset is None and all(used[name] == float(pinned[name]) for name in NAMES),
        "override": override,
        "wind_enabled": bool(wind_enabled),
        "effect": ("applied in every wind wrench (tasks/_shared/wind.py law: CX=-C_X0 cos g, CY=C_Y0 sin g, "
                   "CN=C_N0 sin 2g" + (" + (s/L) CY" if offset is not None else "") + ")" if wind_enabled else
                   "none: the wind suite is disabled, the coefficients are inert"),
        "rule": DIAGNOSTIC_RULE,
    }


__all__ = ["PSEUDO_SUITE", "NAMES", "OFFSET", "ALL_NAMES", "STAMP_SCHEMA", "OFFSET_FORM", "DIAGNOSTIC_RULE",
           "WindCoefficientError", "is_override_token", "validate_override", "split_tokens", "coefficient_override",
           "offset_value", "pinned_values", "used_values", "OffsetFossenCoefficients", "stamp"]
