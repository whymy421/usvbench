"""Pairing primitives for journal certificates (CONTEXT_R4 D11; F0).

Pure Python (json + hashlib); no torch, no simulator, importable by path from
analysis tools. Every finished-episode record of a journal env carries

    record["scenario_hashes"] = {primitive: sha256 hex}

with ONE entry per exogenous primitive. The inventory is fixed (an inactive
suite hashes its neutral value, so the key set never depends on the doses):

  condition primitives (derived here from the schema-3 episode stamp, so any
  reader can recompute them with ``condition_primitive_hashes``):
    direction_stratum      compass-8 stratum, world angle, task axis, rotation
    waves                  effective wave dose + realisation digest (calm if off/Hs 0)
    wind                   effective wind dose + stream identity + direction
    current                effective current + world direction
    actuator.cap / actuator.gain / actuator.lag / actuator.resistance
    perception.noise / perception.bias / perception.latency /
    perception.dropout / perception.bundle
  D7 primitive (from the E4 bridge stamp, ``d7_primitive_hash``):
    d7_assignment          candidate digest of the admitted assignment
  appearing primitive (G3; only in envs that carry the appearing suite,
  from the appearing stamp, ``appearing_primitive_hash``):
    appearing_assignment   dose-independent position digest + dose, key, task
                           (a not-admitted key: its dose-independent
                           not-admitted digest at dose 0, identical at every
                           appearing dose, D20 form)
  family primitives (supplied by the family, e.g. layout, spawn_pose, route,
  berth, rotation; FAMILY_PRIMITIVES lists the minimum per carrier).

Only EXOGENOUS content enters a hash: values that depend on the trajectory
(runtime ticks, wind steps and gust at the stamp, perception frame counters,
estimate counts) are excluded, so two controllers or two doses that end an
episode at different times still pair. Inactive axes hash their EFFECTIVE
neutral value ("off"), because the E3 layer is bit-identical for "disabled"
and "enabled at the zero dose" (E3 tests), so D0 and "suite absent" pair.

``validate_pair(stamp_a, stamp_b, treatment=...)`` accepts two finished-episode
records (``{"conditions", "scenario_hashes"[, "dynamic_hazards"]}``) of the
SAME scenario and returns the pairing summary only if every primitive outside
the declared treatment is identical and the treatment primitive(s) differ;
anything else raises PairingError with the primitive that differs.
``pairing_declaration`` produces the header ``pairing`` block in the format
scripts/analysis_input.validate_pair reads (unchanged_primitives /
changed_primitives).
"""
from __future__ import annotations

import hashlib
import json

PAIRING_SCHEMA = "usvbench.pairing/1"
HASH_TAG = "usvbench.primitive/1"

CONDITION_PRIMITIVES = (
    "direction_stratum", "waves", "wind", "current",
    "actuator.cap", "actuator.gain", "actuator.lag", "actuator.resistance",
    "perception.noise", "perception.bias", "perception.latency", "perception.dropout",
    "perception.bundle",
)
D7_PRIMITIVE = "d7_assignment"
APPEARING_PRIMITIVE = "appearing_assignment"

# treatment name -> primitives it changes
SUITE_PRIMITIVES = {
    "waves": ("waves",),
    "wind": ("wind",),
    "current": ("current",),
    "actuator": ("actuator.cap", "actuator.gain", "actuator.lag", "actuator.resistance"),
    "perception": ("perception.noise", "perception.bias", "perception.latency", "perception.dropout",
                   "perception.bundle"),
    "dynamic": (D7_PRIMITIVE,),
    "appearing": (APPEARING_PRIMITIVE,),
}
for _name in CONDITION_PRIMITIVES[4:]:
    SUITE_PRIMITIVES[_name] = (_name,)
SUITE_PRIMITIVES[D7_PRIMITIVE] = (D7_PRIMITIVE,)
SUITE_PRIMITIVES[APPEARING_PRIMITIVE] = (APPEARING_PRIMITIVE,)
del _name

# minimum family-supplied primitives per carrier family (families may add more)
FAMILY_PRIMITIVES = {
    "hazard_nav": ("layout", "rotation", "spawn_pose"),
    "path_following": ("route", "spawn_pose"),
    "station_keeping": ("layout", "spawn_pose"),
    "docking": ("berth", "spawn_pose"),
    "path_hazard": ("layout", "route", "spawn_pose"),
    "harbor_mission": ("layout", "route", "berth", "spawn_pose"),
}
RESERVED = frozenset(CONDITION_PRIMITIVES) | {D7_PRIMITIVE, APPEARING_PRIMITIVE, "scenario"}


class PairingError(ValueError):
    """Two records do not form a clean single-treatment pair."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def primitive_hash(name, content):
    """SHA-256 of the tagged canonical JSON of one primitive's content."""
    text = canonical({"tag": HASH_TAG, "primitive": name, "content": content})
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def is_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


# ------------------------------------------------------------ condition side --
def _effective(stamp):
    """Exogenous content of every condition primitive of one episode stamp."""
    if not isinstance(stamp, dict) or stamp.get("schema") != "usvbench.conditions/3":
        raise PairingError("pairing needs a schema-3 (usvbench.conditions/3) episode stamp")
    p, rt = stamp["parameters"], stamp.get("runtime") or {}
    if "episode_index" not in rt:
        raise PairingError("the stamp has no episode runtime block (header stamps cannot be paired)")
    seed = p["seed"]
    stratum = rt["direction_stratum"]
    out = {"direction_stratum": {"seed": seed, "strata_set": p["direction_strata"],
                                 "index": stratum["index"], "world_flow_to_rad": stratum["world_flow_to_rad"],
                                 "reference_axis_rad": stratum["reference_axis_rad"]}}
    w = p["waves"]
    if w["enabled"] and w["hs_m"] > 0:
        waves = rt["waves"]
        out["waves"] = {"dose": {k: w[k] for k in sorted(w) if k != "enabled"},
                        "mean_direction_rad": waves["mean_direction_rad"],
                        "n_components": waves["n_components"],
                        "realisation_sha256": waves["realisation_sha256"],
                        "realisation_key": waves["realisation_key"]}
    else:
        out["waves"] = "calm"
    wi = p["wind"]
    if wi["enabled"]:
        wind = rt["wind"]
        out["wind"] = {"dose": {k: wi[k] for k in sorted(wi) if k != "enabled"},
                       "stream": {"seed": seed, "env": rt["env_index"], "episode": wind["episode_index"],
                                  "protocol": wind["rng_protocol"]},
                       "world_flow_to_rad": wind["world_flow_to_rad"],
                       "applied_vessel_height_speed_mps": wind["applied_vessel_height_speed_mps"]}
    else:
        out["wind"] = "off"
    c = p["current"]
    if c["enabled"] and (c["speed_mps"] > 0 or c["speed_table"]):
        out["current"] = {"speed_mps": c["speed_mps"], "speed_table": c["speed_table"],
                          "world_flow_to_rad": rt["current"]["world_flow_to_rad"]}
    else:
        out["current"] = "still"
    a = p["actuator"]
    on = a["enabled"]
    act = rt.get("actuator") or {}
    cap = a["thrust_cap"] if on else 1.0
    out["actuator.cap"] = "nominal" if cap == 1.0 else {"thrust_cap": cap}
    gain = a["motor_gain_one_sided"] if on else 1.0
    out["actuator.gain"] = ("nominal" if gain == 1.0 else
                            {"motor_gain_one_sided": gain, "failing_side": act.get("failing_side"),
                             "motor_gain_port_starboard": act.get("motor_gain_port_starboard")})
    lag = a["additional_lag_s"] if on else 0.0
    out["actuator.lag"] = "nominal" if lag == 0.0 else {"additional_lag_s": lag}
    res = a["resistance_scale"] if on else 1.0
    out["actuator.resistance"] = "nominal" if res == 1.0 else {"resistance_scale": res}
    q = p["perception"]
    on = q["enabled"]
    per = rt.get("perception") or {}
    groups = per.get("groups") or {}

    def key(group):
        return (groups.get(group) or {}).get("stream_key")

    if on and q["pos_noise_sigma_m"] > 0:
        out["perception.noise"] = {"sigma_m": q["pos_noise_sigma_m"],
                                   "correlation_s": q["pos_noise_correlation_s"], "stream_key": key("pos")}
    else:
        out["perception.noise"] = "off"
    if on and q["pos_bias_m"] > 0:
        out["perception.bias"] = {"magnitude_m": q["pos_bias_m"],
                                  "direction_world_rad": per.get("bias_direction_world_rad")}
    else:
        out["perception.bias"] = "off"
    frames = per.get("latency_frames", 0) if on else 0
    out["perception.latency"] = "off" if not frames else {"latency_frames": frames}
    if on and per.get("outage_frames", 0):
        out["perception.dropout"] = {"outage_frames": per["outage_frames"],
                                     "start_frame": per.get("outage_start_frame"),
                                     "source": per.get("outage_source"),
                                     "stream_key": per.get("outage_stream_key")}
    else:
        out["perception.dropout"] = "off"
    bundle = {}
    if on and (q["vel_noise_sigma_mps"] > 0 or q["yaw_rate_noise_sigma_rad_s"] > 0):
        bundle["velocity"] = {"sigma_mps": q["vel_noise_sigma_mps"],
                              "yaw_rate_sigma_rad_s": q["yaw_rate_noise_sigma_rad_s"], "stream_key": key("vel")}
    if on and q["ray_noise_sigma_m"] > 0:
        bundle["ray"] = {"sigma_m": q["ray_noise_sigma_m"], "stream_key": key("ray")}
    out["perception.bundle"] = bundle or "off"
    if set(out) != set(CONDITION_PRIMITIVES):
        raise AssertionError("condition primitive inventory drifted")
    return out


def condition_primitive_hashes(stamp):
    """{primitive: sha256} for the condition primitives of one episode stamp."""
    return {name: primitive_hash(name, content) for name, content in sorted(_effective(stamp).items())}


# ------------------------------------------------------------------ D7 side --
def d7_primitive_hash(envelope):
    """Hash of the admitted D7 assignment (candidate digest) of one episode.

    ``envelope`` is the E4 bridge stamp ``{"payload", "sha256"}`` returned by
    HazardBridge.finish; its digest is verified first.
    """
    from .certificate import verify_dynamic_hazard_stamp
    value = verify_dynamic_hazard_stamp(envelope)
    assignment = value["payload"]["assignment"]
    content = {"candidate_digest": assignment["candidate_digest"], "task": assignment["task"],
               "dose": assignment["dose"], "key": assignment["key"]}
    # G23c (REV-G23b F7): the runtime rule that timed the object is part of the D7 scenario; stamps without a
    # rule block (v2.5, D0, moving basin, not-admitted keys) hash exactly as before
    runtime = value["payload"].get("runtime")
    rule = runtime.get("runtime_rule") if isinstance(runtime, dict) else None
    if isinstance(rule, dict) and rule.get("version") is not None:
        content["runtime_rule"] = rule["version"]
    return primitive_hash(D7_PRIMITIVE, content)


def appearing_primitive_hash(envelope):
    """Hash of the appearing assignment of one episode (G3 rule 7).

    ``envelope`` is the appearing stamp ``{"payload", "sha256"}`` of
    conditions/appearing/runtime.AppearingBridge.finish; its digest and its
    D16/D27 invariants are verified first. The content is the dose-independent
    position digest plus dose, key and task, so two runs of one key pair for
    every other suite at a fixed appearing dose and differ only in the dose
    under the appearing treatment (D0 = "suite absent" for every other primitive).
    """
    from .appearing.stamp import read_appearing_stamp
    value = read_appearing_stamp(envelope)
    assignment = value["payload"]["assignment"]
    return primitive_hash(APPEARING_PRIMITIVE, {"position_digest": assignment["position_digest"],
                                                "task": assignment["task"], "dose": assignment["dose"],
                                                "key": assignment["key"]})


def validate_family_hashes(family_hashes, family=None):
    """Family-supplied primitives: hex digests, no reserved names, carrier minimum present."""
    if not isinstance(family_hashes, dict):
        raise PairingError("family scenario hashes must be a mapping {primitive: sha256 hex}")
    clash = RESERVED & set(family_hashes)
    if clash:
        raise PairingError(f"family primitives use reserved names: {sorted(clash)}")
    for name, value in family_hashes.items():
        if not isinstance(name, str) or not name or not is_hash(value):
            raise PairingError(f"family primitive {name!r} must map a nonempty name to a sha256 hex digest")
    if family is not None:
        missing = set(FAMILY_PRIMITIVES.get(family, ())) - set(family_hashes)
        if missing:
            raise PairingError(f"family {family!r} must hash at least {sorted(missing)}")
    return dict(family_hashes)


def episode_scenario_hashes(condition_stamp, dynamic_envelope, family_hashes, family=None):
    """The complete D11 map of one finished episode (all three sources)."""
    hashes = condition_primitive_hashes(condition_stamp)
    if dynamic_envelope is None:
        raise PairingError("journal episodes always carry a D7 assignment stamp (D0 included)")
    hashes[D7_PRIMITIVE] = d7_primitive_hash(dynamic_envelope)
    hashes.update(validate_family_hashes(family_hashes, family))
    return dict(sorted(hashes.items()))


# --------------------------------------------------------------- treatments --
def treatment_primitives(treatment):
    """Primitive names a declared treatment changes (a name or a tuple of names)."""
    names = (treatment,) if isinstance(treatment, str) else tuple(treatment or ())
    if not names:
        raise PairingError("declare the treatment (a suite, a suite axis or a tuple of them)")
    out = []
    for name in names:
        if name not in SUITE_PRIMITIVES:
            raise PairingError(f"unknown treatment {name!r}; one of {sorted(SUITE_PRIMITIVES)}")
        out.extend(SUITE_PRIMITIVES[name])
    return tuple(sorted(set(out)))


def pairing_declaration(primitives, treatment=None):
    """Header ``pairing`` block (analysis_input.validate_pair format)."""
    names = sorted(set(primitives))
    changed = list(treatment_primitives(treatment)) if treatment is not None else []
    unknown = set(changed) - set(names)
    if unknown:
        raise PairingError(f"treatment primitives absent from the inventory: {sorted(unknown)}")
    return {"schema": PAIRING_SCHEMA, "primitives": names, "changed_primitives": changed,
            "unchanged_primitives": [n for n in names if n not in changed]}


def _identity(record):
    rt = record["conditions"]["runtime"]
    return {"seed": record["conditions"]["parameters"]["seed"], "env_index": rt["env_index"],
            "episode_index": rt["episode_index"], "scenario_ordinal": rt["scenario_ordinal"]}


def _checked_hashes(record, label):
    if not isinstance(record, dict) or "conditions" not in record or "scenario_hashes" not in record:
        raise PairingError(f"{label}: a finished-episode record with conditions and scenario_hashes is required")
    from .stamp import parameters_digest
    block = record["conditions"]
    if parameters_digest(block["parameters"]) != block["parameters_sha256"]:
        raise PairingError(f"{label}: condition parameter digest mismatch")
    hashes = record["scenario_hashes"]
    if not isinstance(hashes, dict) or not all(is_hash(v) for v in hashes.values()):
        raise PairingError(f"{label}: scenario_hashes must map names to sha256 hex digests")
    recomputed = condition_primitive_hashes(block)
    for name, value in recomputed.items():
        if hashes.get(name) != value:
            raise PairingError(f"{label}: scenario hash {name!r} does not match its condition stamp")
    dynamic = record.get("dynamic_hazards")
    if dynamic is not None and hashes.get(D7_PRIMITIVE) != d7_primitive_hash(dynamic):
        raise PairingError(f"{label}: scenario hash {D7_PRIMITIVE!r} does not match its D7 stamp")
    appearing = record.get("appearing_hazards")
    if appearing is not None:
        try:
            value = appearing_primitive_hash(appearing)
        except ValueError as error:
            raise PairingError(f"{label}: appearing stamp rejected: {error}") from None
        if hashes.get(APPEARING_PRIMITIVE) != value:
            raise PairingError(f"{label}: scenario hash {APPEARING_PRIMITIVE!r} does not match its appearing stamp")
    elif APPEARING_PRIMITIVE in hashes:
        raise PairingError(f"{label}: an {APPEARING_PRIMITIVE!r} hash needs the record's appearing stamp")
    return hashes


def validate_pair(stamp_a, stamp_b, treatment):
    """Check that two finished-episode records form a clean single-treatment pair.

    Returns {"treatment", "changed_primitives", "unchanged_primitives",
    "identity"}. Raises PairingError when the records are different
    scenarios, when any primitive outside the treatment differs, when the
    treatment primitives are identical (no treatment difference), or when a
    stored hash does not match its stamp.
    """
    changed = set(treatment_primitives(treatment))
    ha, hb = _checked_hashes(stamp_a, "stamp_a"), _checked_hashes(stamp_b, "stamp_b")
    if set(ha) != set(hb):
        raise PairingError(f"primitive inventories differ: {sorted(set(ha) ^ set(hb))}")
    missing = changed - set(ha)
    if missing:
        raise PairingError(f"treatment primitives are not in the inventory: {sorted(missing)}")
    ia, ib = _identity(stamp_a), _identity(stamp_b)
    if ia != ib:
        raise PairingError(f"different scenarios: {ia} vs {ib}")
    ca, cb = stamp_a["conditions"]["context"], stamp_b["conditions"]["context"]
    for field in ("vehicle", "family", "observation_contract", "physics_dt_s", "control_dt_s",
                  "hydrodynamics_model", "vessel_profile"):
        if ca.get(field) != cb.get(field):
            raise PairingError(f"context differs outside the treatment: {field}")
    differing = sorted(name for name in ha if ha[name] != hb[name])
    outside = [name for name in differing if name not in changed]
    if outside:
        raise PairingError(f"unpaired difference outside the treatment {treatment!r}: {outside}")
    if not differing:
        raise PairingError(f"no treatment difference: every primitive of {treatment!r} is identical")
    return {"schema": PAIRING_SCHEMA, "treatment": treatment,
            "changed_primitives": differing,
            "unchanged_primitives": sorted(name for name in ha if name not in differing),
            "identity": ia}


__all__ = ["PAIRING_SCHEMA", "CONDITION_PRIMITIVES", "D7_PRIMITIVE", "APPEARING_PRIMITIVE", "SUITE_PRIMITIVES",
           "FAMILY_PRIMITIVES", "PairingError", "primitive_hash", "condition_primitive_hashes",
           "d7_primitive_hash", "appearing_primitive_hash", "validate_family_hashes", "episode_scenario_hashes",
           "treatment_primitives", "pairing_declaration", "validate_pair", "is_hash"]
