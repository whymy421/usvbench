"""JSON-native certificate records with a hash of FULL resolved parameters.

Schema "usvbench.conditions/2" (D1): parameters are the complete resolved
ConditionConfig (schema_version 2). The digest is SHA-256 of the canonical
JSON (sorted keys, compact separators, ASCII) of ``parameters``; the
certificate validator recomputes it. Semantics/context/runtime are not part
of the digest (they describe the implementation and the realization).

Schema "usvbench.conditions/3" (E3): the hydro_v2 condition layer. Same
block layout and the SAME digest rule over ``parameters`` (the complete
resolved schema-3 JournalConditionConfig), so a validator recomputes both
schemas identically. ``context`` (header) carries the hydrodynamics model,
the vessel profile, the rotational-damping reference, the Froude-Krylov
flag, the saturation policy, the footprint filter, the resolved wave band
and component count, spreading, direction strata, every resolved parameter
and the PENDING owner decisions; ``runtime`` (per episode) carries the
realisation (stratum, keys, digests, actuator side). Per-episode stamps
have the header's parameters, hence the header's digest. Schema-2 stamps
stay readable (``config_from_stamp`` accepts both).
"""
from dataclasses import asdict
import hashlib
import json
from .config import SCHEMA_VERSION, resolve

STAMP_SCHEMA = f"usvbench.conditions/{SCHEMA_VERSION}"
STAMP_SCHEMA_V3 = "usvbench.conditions/3"
READABLE_SCHEMAS = (STAMP_SCHEMA, STAMP_SCHEMA_V3)

SEMANTICS_V3 = {
    "hydrodynamics_model": "hydro_v2 (D3 HydroV2Runtime): one water-relative hull law, implicit damping",
    "water_composition": ("water velocity u_w = current (uniform, world XY) + footprint-filtered wave "
                          "orbital velocity, damped once in the hull law; no separate current/wave drag"),
    "waves": ("SpectralWaves.from_sea_state(C2 SeaState physics_version=2) keyed by (seed, env, episode); "
              "never SeaState.forces; band raised to the vessel requirement (check_wave_band enforced)"),
    "current": "water velocity only, evaluated at the runtime tick clock; never an incremental force",
    "wind": ("C3 Fossen first harmonic with D5 windage, air-relative hull COM velocity, physical bow "
             "heading, heel lever about the COM; U10 dose; OU gusts protocol usvbench.wind.ou.v2; "
             "zero dose = module off; known forcing in the runtime predictor"),
    "actuator": ("per-motor mixer faults latched per episode: thrust cap (both directions), one-sided "
                 "gain balanced over scenario-ordinal pairs, additional lag; resistance multiplier on "
                 "surge/sway/yaw damping; centred mass and payload rejected"),
    "perception": ("C7 axes on declared measurement groups: correlated position noise, balanced bias, "
                   "latency, one contiguous outage (hold last); independent keyed streams"),
    "clock": "HydroV2Runtime integer ticks only",
    "observation": "waves, current and wind are never observed by the controller",
    "dynamic_hazards": "D7 package (record field dynamic_hazards); C4 progress trigger inert",
}


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def parameters_digest(parameters):
    """Canonical SHA-256 of a JSON-native parameter mapping."""
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def stamp_conditions(certificate, cfg, *, runtime=None, context=None):
    """Return a detached certificate copy; do not mutate the caller's data."""
    parameters = _json_copy(asdict(cfg))
    physics_version = (context or {}).get("physics_version", 1)
    wave_semantics = (
        "SeaState physics v2 owns the only calm-increment subtraction; per-realization band "
        "[v2_f_min_over_fp, v2_f_max_over_fp]*fp with >=256 components; legacy f_min/f_max unused"
        if physics_version == 2 else
        "SeaState physics v1 fixed band; the condition adapter owns the only calm-drag-floor subtraction"
    )
    block = {
        "schema": STAMP_SCHEMA,
        "parameters": parameters,
        "parameters_sha256": parameters_digest(parameters),
        "dose_status": "prototype_uncalibrated",
        "semantics": {
            "water_composition": "additive_incremental_wrenches; not combined relative velocity",
            "waves": wave_semantics,
            "clock": "one integer per-env physics clock (C2 WaveClock); one advance per physics substep",
            "actuator": ("legacy external-wrench surrogate; USD mass, gravity and contacts unchanged; "
                         "physics v2 takes drag/mass scales in the implicit hull damping only"),
            "perception": "legacy physical-channel noise/bias/delay/hold-last; no truth mutation",
            "wind": ("C3 Fossen first-harmonic windage, air-relative hull COM velocity, physical bow "
                     "heading, exact keyed OU gusts per physics substep; not observed by the controller"),
            "dynamic_hazards": "task adapter required",
        },
        "context": _json_copy(context or {}),
        "runtime": _json_copy(runtime or {}),
    }
    result = _json_copy(certificate)
    result["conditions"] = block
    return result


def stamp_conditions_v3(cfg, *, runtime=None, context=None):
    """Schema-3 (hydro_v2) condition block; the digest covers ``parameters`` only."""
    if getattr(cfg, "schema_version", None) != 3:
        raise ValueError("stamp_conditions_v3 needs a schema-3 JournalConditionConfig")
    parameters = _json_copy(asdict(cfg))
    return {
        "schema": STAMP_SCHEMA_V3,
        "parameters": parameters,
        "parameters_sha256": parameters_digest(parameters),
        "dose_status": "journal_presets_provisional_owner_decisions_pending",
        "semantics": dict(SEMANTICS_V3),
        "context": _json_copy(context or {}),
        "runtime": _json_copy(runtime or {}),
    }


def config_from_stamp(certificate):
    """Resolved config of a schema-2 (legacy) or schema-3 (hydro_v2) stamp."""
    block = certificate["conditions"]
    if block["schema"] not in READABLE_SCHEMAS:
        raise ValueError("unsupported condition certificate schema")
    parameters = block["parameters"]
    if parameters_digest(parameters) != block["parameters_sha256"]:
        raise ValueError("condition parameter hash mismatch")
    if block["schema"] == STAMP_SCHEMA_V3:
        from .config_v3 import resolve_v3
        return resolve_v3(overrides=parameters)
    return resolve(overrides=parameters)
