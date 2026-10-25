"""Evaluator boundary for condition stamps; simulator-independent.

``certificate_fields(env)`` keeps C4's legacy behaviour (empty for a cfg that
never mentions conditions). Certificate schema 3.1 calls it with
``include_off=True`` so every header carries an explicit, fully resolved
stamp, and ``episode_certificate_fields`` for per-episode records.

E3 additions:
* hydro_v2 configurations (schema-3 layer) always have an active layer; an
  evaluator that finds none raises instead of writing a schema-2 off stamp.
* D7 per-episode dynamic-hazard stamps (D7 INTEGRATION_SPEC.md B09). Field
  ``DYNAMIC_HAZARD_FIELD = "dynamic_hazards"`` beside ``conditions`` in each
  finished-episode record; its value is the D7/E4 stamp envelope embedded
  UNCHANGED: ``{"payload": {...}, "sha256": <hex>}`` produced by
  ``conditions/dynamic/stamp.py`` ``make_stamp``. Digest rule (identical to
  D7 stamp.py:29-44): sha256 over the ASCII bytes of
  ``json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)``
  after dataclasses/tuples are converted to plain dicts/lists. The env
  latches ``env.episode_dynamic_hazards[i]`` from ``bridge.finish(i, reason)``
  BEFORE the row is reset (B08), next to ``env.episode_conditions[i]``.
  ``verify_dynamic_hazard_stamp`` re-derives the digest (it does not import
  the D7 package); the D7 ``read_stamp`` additionally checks its schema and
  model versions. Header-level D7 data (pool digest, per-dose admission
  counts) is taken verbatim from ``env.dynamic_hazard_header`` into
  ``DYNAMIC_HAZARD_HEADER_FIELD``.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json

from .config import resolve
from .stamp import stamp_conditions

DYNAMIC_HAZARD_FIELD = "dynamic_hazards"
DYNAMIC_HAZARD_HEADER_FIELD = "dynamic_hazards_header"


def _physics_version(cfg):
    physics = getattr(cfg, "underwater_physics_cfg", None)
    return int(getattr(physics, "physics_version", getattr(cfg, "physics_version", 1)) or 1)


def _hydro_v2(cfg):
    physics = getattr(cfg, "underwater_physics_cfg", None)
    return getattr(physics, "hydrodynamics_model", "legacy") == "hydro_v2"


def _plain(value):
    """Same conversion as D7 stamp.plain: dataclasses/tuples to JSON natives."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def dynamic_hazard_digest(payload):
    """D7 digest rule: SHA-256 of the canonical ASCII JSON of the payload."""
    text = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def verify_dynamic_hazard_stamp(envelope):
    """Return a detached copy of a D7 stamp envelope after checking its digest."""
    value = json.loads(envelope) if isinstance(envelope, str) else json.loads(
        json.dumps(_plain(envelope), sort_keys=True, allow_nan=False))
    if not isinstance(value, dict) or set(value) != {"payload", "sha256"}:
        raise ValueError("invalid dynamic-hazard stamp envelope (keys must be payload, sha256)")
    if dynamic_hazard_digest(value["payload"]) != value["sha256"]:
        raise ValueError("dynamic-hazard stamp digest mismatch")
    return value


def _with_dynamic(env, index, result):
    stamps = getattr(env, "episode_dynamic_hazards", None)
    if stamps is not None and stamps[index]:
        result[DYNAMIC_HAZARD_FIELD] = verify_dynamic_hazard_stamp(stamps[index])
    return result


def certificate_fields(env, env_cfg=None, *, include_off=False):
    """Header stamp: the active layer's, or an explicit all-off stamp."""
    layer = getattr(env, "_conditions", None)
    header = getattr(env, "dynamic_hazard_header", None)
    extra = {} if header is None else {DYNAMIC_HAZARD_HEADER_FIELD: json.loads(
        json.dumps(_plain(header), sort_keys=True, allow_nan=False))}
    if layer is not None:
        return {"conditions": deepcopy(layer.certificate_stamp()), **extra}
    cfg = env_cfg if env_cfg is not None else env.cfg
    raw = getattr(cfg, "conditions", None)
    presets = getattr(cfg, "condition_presets", ())
    if raw is None and not presets and not include_off:
        return {}
    if _hydro_v2(cfg):
        raise ValueError("hydro_v2 configuration has no active condition layer (the schema-3 layer "
                         "always exists under hydro_v2)")
    resolved = resolve(presets, raw, seed=getattr(cfg, "seed", None) or 0)
    if resolved.any_enabled:
        raise ValueError("enabled condition configuration has no active condition layer")
    return {**stamp_conditions({}, resolved, context={
        "layer_active": False,
        "legacy_behavior_retained": True,
        "physics_version": _physics_version(cfg),
        "note": "All condition suites off; the declared task physics is unchanged, including legacy non-calm variants.",
    }), **extra}


def episode_certificate_fields(env, index, env_cfg=None, *, include_off=False):
    """Finished-episode stamp latched at reset, never live post-reset state.

    Without a layer, ``include_off`` returns the header's off stamp with a
    runtime block naming the env, so every v3.1 record carries a stamp.
    A latched D7 stamp (``env.episode_dynamic_hazards[index]``) is added
    under ``DYNAMIC_HAZARD_FIELD`` after its digest is verified.
    """
    values = getattr(env, "episode_conditions", None)
    if values is not None and values[index]:
        return _with_dynamic(env, index, {"conditions": deepcopy(values[index])})
    if not include_off:
        return _with_dynamic(env, index, {})
    if getattr(env, "_conditions", None) is not None:
        raise ValueError(f"env {index}: condition layer active but no finished-episode stamp was latched")
    header = certificate_fields(env, env_cfg, include_off=True)["conditions"]
    header["runtime"] = {"layer_active": False, "env_index": int(index)}
    return _with_dynamic(env, index, {"conditions": header})
