"""Diagnostic-only override of the roll/pitch damping ratio (T-base v2.5, owner decision T1.35b). Pure Python.

The journal hull model damps roll and pitch with the per-vessel journal default of
tasks/_shared/hydro_profiles.py (``rollpitch_damping`` rule, decision 2, PENDING): BlueBoat
"zeta:0.1", WAM-V "registry" (the VRX values). Owner decision T1.35 (2026-10-02): the WAM-V
switch to "zeta:0.1" is NOT applied yet; it is to be tested first, with a per-run diagnostic
override of the damping ratio given as a suite token of the pseudo-suite ``hydro``:

    --suite waves=D4 hydro.rollpitch_zeta=0.2

The run then uses the existing ``zeta:<ratio>`` rule of hydro_profiles.py for roll AND pitch
(c_axis = 2 zeta sqrt(K_axis I_axis) with the registry's total stiffness and inertia diagonal);
yaw, surge, sway, heave, quadratic terms, mass, inertia and thrust are unchanged.

Rules (fail closed; the pattern of G16's windcoef.* tokens, tasks/_shared/conditions/wind_coefficients.py):
  * only the name rollpitch_zeta; a JSON number (no bool, no string), finite, 0 < zeta <= 2
    (ZETA_BOUND, a sanity bound: 1 is critical damping); one value per run (a repeated token with
    another value is a C1 conflict);
  * the token is NOT part of the schema-3 condition configuration: the parameter digest, the
    JournalConditions stamp and the condition stamps of a run without it are unchanged; the
    override reaches the env only through ``env_cfg.journal_rollpitch_zeta`` (set ONLY when an
    override is given), which journal_env_mixin.journal_setup turns into the profile option
    rollpitch_damping = "zeta:<value>" (refused when cfg.journal_vehicle_options already names
    rollpitch_damping);
  * eval_protocol accepts it only in DIAGNOSTIC runs (seed-registry purpose "diagnostic", e.g.
    DIAG42; reference_environment.is_diagnostic) and refuses it in every official run (selection,
    pilot, calibration, final_test), before the simulator starts (preflight) and again when the
    suites are installed (apply_journal_suites); run_cell (official purposes only) refuses it;
    cert_v3 refuses to certify an official run whose roll/pitch damping is overridden;
    validate_cert_v3 reports such a certificate as an error (non-official);
  * every journal certificate header carries ``rollpitch_damping`` (schema
    usvbench.rollpitch_damping/1): the journal-default rule and values, the rule and values the
    runtime actually used, whether they are the pinned ones and the requested override;
  * MPPI's internal prediction model (scripts/mppi_controller.py) keeps the journal default; the
    override changes the simulated plant only (the T1.35b sensitivity batch uses classical
    controllers).
"""
from __future__ import annotations

import json
import math

PSEUDO_SUITE = "hydro"
NAME = "rollpitch_zeta"
TOKEN_KEY = PSEUDO_SUITE + "." + NAME
ZETA_BOUND = 2.0
STAMP_SCHEMA = "usvbench.rollpitch_damping/1"
CFG_FIELD = "journal_rollpitch_zeta"
DECISION = "owner decision T1.35b, 2026-10-02: WAM-V roll/pitch damping zeta 0.1 to be tested first (diagnostic override)"
DIAGNOSTIC_RULE = ("hydro.rollpitch_zeta overrides are diagnostic only (seed-registry purpose 'diagnostic', e.g. "
                   "DIAG42); official runs are refused before the simulator starts and an official certificate "
                   "with an overridden roll/pitch damping is invalid (T1.35b; G16 windcoef pattern, CONTEXT_R4 "
                   "D23/D28)")
EFFECT = ("roll and pitch linear damping of the hydro_v2 hull model, c_axis = 2 zeta sqrt(K_axis I_axis) "
          "(hydro_profiles.py rollpitch_damping 'zeta:<ratio>' rule, registry total stiffness and inertia diagonal); "
          "nothing else changes")


class RollPitchOverrideError(ValueError):
    """An invalid or conflicting roll/pitch damping override."""


def is_override_token(token):
    """True for a ``hydro.<name>=<value>`` token (string test only, no validation)."""
    if not isinstance(token, str) or "=" not in token:
        return False
    key = token.split("=", 1)[0].strip()
    return key.split(".", 1)[0] == PSEUDO_SUITE


def _number(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise RollPitchOverrideError(f"{TOKEN_KEY}: {value!r} is not a JSON number") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RollPitchOverrideError(f"{TOKEN_KEY}: {value!r} is not a number")
    value = float(value)
    if not math.isfinite(value):
        raise RollPitchOverrideError(f"{TOKEN_KEY} must be finite")
    if not 0.0 < value <= ZETA_BOUND:
        raise RollPitchOverrideError(f"{TOKEN_KEY} = {value!r} must lie in (0, {ZETA_BOUND}]")
    return value


def validate_override(override):
    """Return a canonical {"rollpitch_zeta": float} or None for no override."""
    if override is None:
        return None
    if not isinstance(override, dict) or set(override) != {NAME}:
        raise RollPitchOverrideError(f"a roll/pitch damping override is {{{NAME!r}: <number>}}, got {override!r}")
    return {NAME: _number(override[NAME])}


def split_tokens(tokens):
    """(other tokens in order, override {"rollpitch_zeta": float} or None) from a token list."""
    rest, value = [], None
    for token in tokens or ():
        if not is_override_token(token):
            rest.append(token)
            continue
        key, raw = (part.strip() for part in token.split("=", 1))
        if key != TOKEN_KEY:
            raise RollPitchOverrideError(f"{token!r}: use {TOKEN_KEY}=<number>")
        number = _number(raw)
        if value is not None and value != number:
            raise RollPitchOverrideError(f"declared conflict C1 (two values for one field): {key} given twice "
                                         f"({value!r} and {number!r})")
        value = number
    return rest, ({NAME: value} if value is not None else None)


def rule_label(override):
    """The hydro_profiles rollpitch_damping rule of a validated override ("zeta:<ratio>")."""
    return "zeta:%r" % validate_override(override)[NAME]


def cfg_override(env_cfg):
    """The validated override installed on a cfg (``journal_rollpitch_zeta``), or None when absent."""
    if not hasattr(env_cfg, CFG_FIELD):
        return None
    return validate_override({NAME: getattr(env_cfg, CFG_FIELD)})


def vehicle_options(env_cfg):
    """resolve_journal_vehicle options of a journal cfg with the override applied (a new dict).

    Without an override this is exactly dict(cfg.journal_vehicle_options or {})."""
    options = dict(getattr(env_cfg, "journal_vehicle_options", None) or {})
    override = cfg_override(env_cfg)
    if override is not None:
        if "rollpitch_damping" in options:
            raise RollPitchOverrideError("cfg.journal_vehicle_options names rollpitch_damping=%r; the diagnostic "
                                         "%s override cannot be combined with it"
                                         % (options["rollpitch_damping"], TOKEN_KEY))
        options["rollpitch_damping"] = rule_label(override)
    return options


def profile_values(profile):
    """{"rule", "roll_nms_rad", "pitch_nms_rad"} of a hydro_profiles.VesselProfileV2."""
    roll, pitch = profile.hydro.angular_linear_damping[:2]
    return {"rule": str(profile.sources["options"]["rollpitch_damping"]), "roll_nms_rad": float(roll),
            "pitch_nms_rad": float(pitch)}


def stamp(vessel_profile, pinned, used, override):
    """The header block ``rollpitch_damping`` (JSON-native); pinned / used = profile_values(...)."""
    override = validate_override(override)
    pinned, used = dict(pinned), dict(used)
    if override is not None and used["rule"] != rule_label(override):
        raise RollPitchOverrideError("runtime roll/pitch rule %r is not the override %r"
                                     % (used["rule"], rule_label(override)))
    if override is None and used != pinned:
        raise RollPitchOverrideError("runtime roll/pitch damping %r differs from the journal default %r without "
                                     "an override" % (used, pinned))
    return {
        "schema": STAMP_SCHEMA,
        "vessel_profile": str(vessel_profile),
        "pinned": pinned,
        "used": used,
        "pinned_values": override is None and used == pinned,
        "override": override,
        "effect": EFFECT,
        "decision": DECISION,
        "rule": DIAGNOSTIC_RULE,
    }


__all__ = ["PSEUDO_SUITE", "NAME", "TOKEN_KEY", "ZETA_BOUND", "STAMP_SCHEMA", "CFG_FIELD", "DECISION",
           "DIAGNOSTIC_RULE", "EFFECT", "RollPitchOverrideError", "is_override_token", "validate_override",
           "split_tokens", "rule_label", "cfg_override", "vehicle_options", "profile_values", "stamp"]
