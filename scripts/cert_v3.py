"""Certificate v3 provenance and episode accounting; no simulator imports.

Timers measure synchronized wall time for a complete vectorized call. Each
participating episode receives that batch latency, never latency / num_envs.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from fixed_grid import FixedEpisodeGrid
from episode_path_observer import EpisodePathObserver
import code_tree
import reference_environment
import d7_exposure
import appearing_exposure
from episode_outcome import TERMINATION_REASON_NAMES, episode_outcome


# C5 codes 0-5 plus D4's appended validity codes 6-9 (CONTEXT_R3 decision 17;
# names verbatim from tasks/_shared/journal_contracts_v2.py:44-46).
REASONS = dict(TERMINATION_REASON_NAMES)
# Schema 3.1 (D1): header.conditions and records[].conditions are required
# condition-layer stamps ("usvbench.conditions/2") with a checked digest.
# Schema 3.2 (E1): release code-tree identity, validated physics version,
# hydrodynamics model, vessel profile, docking level distance and the
# per-episode outcome (see CERT_SCHEMA_v3.md "Schema 3.2").
SCHEMA_MINOR_VERSION = 2
PHYSICS_VERSIONS = ("1", "2")
HYDRODYNAMICS_MODELS = ("legacy", "hydro_v2")
# Families whose success does NOT end the episode (fixed horizon: every
# episode stops by timeout; C5 station_keeping_env.py/docking_env.py pass
# terminated = zeros) versus families that stop on success.
FIXED_HORIZON_FAMILIES = ("station_keeping", "station_keeping_boat", "docking")
STOP_ON_SUCCESS_FAMILIES = ("blueboat_calm_nav", "boat_calm_nav", "rov_calm_nav", "hazard_nav",
                            "path_following", "path_hazard", "harbor_mission")


def _condition_certificate_module(base):
    """Import the condition stamp helpers from the SAME task tree as the env.

    Tries the package of the env's live condition layer, then the env class's
    own task package, then the repository and deployed layouts, finally the
    repository root next to this scripts/ folder. Pure Python and torch only.
    """
    candidates = []
    layer = getattr(base, "_conditions", None)
    if layer is not None:
        candidates.append(type(layer).__module__.rsplit(".", 1)[0] + ".certificate")
    module = getattr(type(base), "__module__", "") or ""
    if module.count(".") >= 2:
        candidates.append(module.rsplit(".", 2)[0] + "._shared.conditions.certificate")
    candidates += ["tasks._shared.conditions.certificate",
                   "isaaclab_tasks.direct._shared.conditions.certificate"]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("tasks._shared.conditions.certificate")


def _green_water_rule_module(conditions_api):
    """G10: the green-water rule (verdict + cell summary) from the SAME conditions package as the env."""
    package = conditions_api.__name__.rsplit(".", 1)[0]
    return importlib.import_module(package + ".green_water_rule")


def journal_waves_enabled(conditions):
    """G10: True when a journal header condition stamp includes the waves suite (enabled, any Hs)."""
    try:
        return bool(conditions["parameters"]["waves"]["enabled"])
    except (KeyError, TypeError):
        return False


def declared_physics_version(env_cfg):
    """C2 ``underwater_physics_cfg.physics_version`` as the string "1" or "2", else None.

    D2 item 8 / F8: the version is read from the nested C2 switch (then the
    legacy top-level attribute) and validated to exactly {"1", "2"}. The string
    form is kept (not an integer) so schema 3.1 documents, which D1 stamped as
    "1"/"2", keep their meaning; schema 3.2 constrains the value to that enum.
    """
    physics = getattr(env_cfg, "underwater_physics_cfg", None)
    version = getattr(physics, "physics_version", None)
    if version is None:
        version = getattr(env_cfg, "physics_version", None)
    if version is None:
        return None
    if isinstance(version, bool) or str(version) not in PHYSICS_VERSIONS:
        raise ValueError("unsupported physics_version %r (expected 1 or 2)" % (version,))
    return str(version)


def _cfg_field(env_cfg, name):
    """(value, source) of a field on underwater_physics_cfg, else on the env cfg."""
    physics = getattr(env_cfg, "underwater_physics_cfg", None)
    if physics is not None and getattr(physics, name, None) is not None:
        return getattr(physics, name), "env_cfg.underwater_physics_cfg." + name
    if getattr(env_cfg, name, None) is not None:
        return getattr(env_cfg, name), "env_cfg." + name
    return None, None


def declared_hydrodynamics_model(env_cfg):
    """D3 selector ``underwater_physics_cfg.hydrodynamics_model``; absent -> "legacy"."""
    value, _ = _cfg_field(env_cfg, "hydrodynamics_model")
    value = "legacy" if value is None else value
    if value not in HYDRODYNAMICS_MODELS:
        raise ValueError("unsupported hydrodynamics_model %r" % (value,))
    return value


def declared_vessel_profile(env_cfg):
    """Vessel profile name/status declared by the env cfg, else None.

    Read (in this order) ``vessel_profile`` on ``underwater_physics_cfg`` or on
    the env cfg: a registry name string, or an object with ``name`` and an
    optional ``status``; plus an optional ``vessel_profile_status`` string.
    Nothing is inferred from the vessel name.
    """
    value, source = _cfg_field(env_cfg, "vessel_profile")
    if value is None:
        return None
    if isinstance(value, str):
        name, status = value, None
    else:
        name, status = getattr(value, "name", None), getattr(value, "status", None)
    explicit, _ = _cfg_field(env_cfg, "vessel_profile_status")
    if explicit is not None:
        status = explicit
    if not isinstance(name, str) or not name or (status is not None and not isinstance(status, str)):
        raise ValueError("vessel_profile must declare a name string and an optional status string")
    return {"name": name, "status": status, "source": source}


def docking_level_distance(env_cfg, base, physics_version):
    """Docking only: the level table in force and the realised spawn distance.

    Mirrors tasks/docking/docking_env.py:80-92 (v2 reads
    eval_level_distances_v2, index clamped, spawn_distance_override_m > 0 wins)
    and reads the realised value from ``base.curriculum.spawn_distance``.
    """
    if not hasattr(env_cfg, "eval_level_distances"):
        return None
    in_force = getattr(env_cfg, "docking_level_table_in_force", None)
    if callable(in_force) and in_force() == "journal_plan":
        # F5 (D2 item 14): journal docking reads no level table; the env stamps the plan.
        stamp = dict(getattr(base, "docking_level_table", None) or {"docking_level_table_in_force": "journal_plan"})
        stamp.setdefault("table_field", "journal_plan")
        return stamp
    field = "eval_level_distances"
    if physics_version == "2" and hasattr(env_cfg, "eval_level_distances_v2"):
        field = "eval_level_distances_v2"
    table = [float(x) for x in getattr(env_cfg, field)]
    level = int(getattr(env_cfg, "eval_level", 0))
    index = min(max(level, 0), len(table) - 1)
    override = float(getattr(env_cfg, "spawn_distance_override_m", 0.0) or 0.0)
    expected = override if override > 0.0 else table[index]
    realised = getattr(getattr(base, "curriculum", None), "spawn_distance", None)
    try:
        realised = float(realised) if realised is not None else None
    except (TypeError, ValueError):
        realised = None
    return {"table_field": field, "table_m": table, "eval_level": level, "table_index": index,
            "spawn_distance_override_m": override, "expected_distance_m": expected,
            "realised_distance_m": realised,
            "realised_source": "base.curriculum.spawn_distance" if realised is not None else None}


def journal_certificate_mode(base, env_cfg):
    """F7: True for journal cfgs (journal_contract_version == 2), which MUST come
    from an env composing JournalEnvMixin (tasks/_shared/journal_env_mixin.py):
    the header takes journal_certificate_header() and every record takes
    journal_episode_record(i). A journal cfg without that API fails closed."""
    if int(getattr(env_cfg, "journal_contract_version", 0) or 0) != 2:
        return False
    for name in ("journal_certificate_header", "journal_episode_record"):
        if not callable(getattr(base, name, None)):
            raise ValueError("journal cfg (journal_contract_version 2) but the env has no %s(); the "
                             "family env must compose JournalEnvMixin" % name)
    return True


def journal_header_block(env_cfg):
    """header.journal: the journal task, its carrier family, the vessel and the contract."""
    task = getattr(env_cfg, "journal_task", None)
    carrier = getattr(env_cfg, "journal_carrier", None)
    if carrier is None:
        plans = _journal_plans_module()
        carrier = plans.TASK_TABLE[task]["carrier"]
    return {"task": str(task), "carrier": str(carrier), "vehicle": str(getattr(env_cfg, "vehicle")),
            "contract_version": 2}


POOL_MANIFEST_ATTRIBUTES = ("journal_d7_pool_manifest_sha256", "d7_pool_manifest_sha256")


def wind_coefficients_official_check(block, registration):
    """G16: refuse to certify an official (non-diagnostic) run with overridden wind coefficients."""
    if block is None:
        return
    overridden = block.get("override") is not None or block.get("pinned_values") is not True
    if overridden and not reference_environment.is_diagnostic(registration):
        purpose = registration.get("purpose") if isinstance(registration, dict) else None
        raise ValueError("official run (seed purpose %r) with wind coefficients %r instead of the pinned %r: "
                         "windcoef.* overrides are diagnostic only (G16)"
                         % (purpose, block.get("used"), block.get("pinned")))


def rollpitch_damping_official_check(block, registration):
    """T25 (owner decision T1.35b): refuse to certify an official run with an overridden roll/pitch damping."""
    if block is None:
        return
    overridden = block.get("override") is not None or block.get("pinned_values") is not True
    if overridden and not reference_environment.is_diagnostic(registration):
        purpose = registration.get("purpose") if isinstance(registration, dict) else None
        raise ValueError("official run (seed purpose %r) with roll/pitch damping %r instead of the journal default "
                         "%r: hydro.rollpitch_zeta overrides are diagnostic only (T1.35b)"
                         % (purpose, block.get("used"), block.get("pinned")))


def appearing_gap_override_official_check(block, registration):
    """G20b (owner decision 2026-10-03): refuse to certify an official run with the appearing.gap_m override."""
    if not isinstance(block, dict) or block.get("gap_override_m") is None:
        return
    if not reference_environment.is_diagnostic(registration):
        purpose = registration.get("purpose") if isinstance(registration, dict) else None
        raise ValueError("official run (seed purpose %r) with the appearing trigger gap overridden to %r m: "
                         "appearing.gap_m is diagnostic only (calibration sweep)" % (purpose, block["gap_override_m"]))


def appearing_supplementary_dose_official_check(block, registration):
    """G20c (REV-G20b m5): refuse to certify an official run at the supplementary appearing dose (D5 under ladder M,
    the calibration rung; diagnostic only)."""
    dose = block.get("dose") if isinstance(block, dict) else None
    if type(dose) is not int or dose <= 4:        # D5.. lie outside the main table D0..D4 in every ladder option
        return
    if not reference_environment.is_diagnostic(registration):
        purpose = registration.get("purpose") if isinstance(registration, dict) else None
        raise ValueError("official run (seed purpose %r) at the supplementary appearing dose D%r: the calibration "
                         "rung is diagnostic only" % (purpose, block["dose"]))


def journal_pool_manifest_sha256(base, env_cfg, header_block):
    """D19 (F7 round 4b): the family's PINNED D7 pool manifest digest.

    Read from, in any combination (all present values must agree): the env's D7
    header block ``pool_manifest_sha256``, the env attribute or method
    ``journal_d7_pool_manifest_sha256`` / ``d7_pool_manifest_sha256``, or the cfg
    attribute of the same names. Stamped as dynamic_hazards_header.pool_manifest_sha256;
    a journal env that publishes none fails closed.
    """
    found = {}
    if isinstance(header_block, dict) and header_block.get("pool_manifest_sha256") is not None:
        found["dynamic_hazards_header"] = header_block["pool_manifest_sha256"]
    for owner, label in ((base, "env"), (env_cfg, "cfg")):
        for name in POOL_MANIFEST_ATTRIBUTES:
            value = getattr(owner, name, None)
            if callable(value):
                value = value()
            if value is not None:
                found[label + "." + name] = value
    values = set(found.values())
    if not values:
        raise ValueError("journal env publishes no pinned D7 pool manifest digest (D19): set one of "
                         "dynamic_hazard_header['pool_manifest_sha256'], env/cfg "
                         + " / ".join(POOL_MANIFEST_ATTRIBUTES))
    if len(values) > 1:
        raise ValueError("conflicting D7 pool manifest digests: %s" % found)
    value = values.pop()
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        raise ValueError("D7 pool manifest digest must be a SHA-256 hex string, got %r" % (value,))
    return value


def journal_pool_pinned_check(header_block):
    """Round T D23: journal certificates come only from pinned pools.

    ``dynamic_hazards_header.pool_pinned is False`` (F3: a pool given by cfg path plus
    sidecar; F6: a pool absent from the pinned manifest) cannot be reproduced from the
    release (D19), so the certificate is refused. A missing flag means pinned (F1, F2, F5
    always pin and do not publish it).
    """
    if isinstance(header_block, dict) and header_block.get("pool_pinned", True) is False:
        raise ValueError("journal certificate refused: dynamic_hazards_header.pool_pinned is False; the D7 "
                         "pool is not in the family's pinned manifest and cannot be reproduced from the "
                         "release (CONTEXT_R4 D23); re-admit and pin it in the family manifest")


def journal_offered_grid(header_block):
    """Round T D24 (c): the pool's offered admission grid {"num_envs": N, "episodes": E}."""
    grid = header_block.get("grid") if isinstance(header_block, dict) else None
    if grid is None:
        raise ValueError("journal certificate refused: dynamic_hazards_header has no offered grid "
                         "{'num_envs', 'episodes'} (CONTEXT_R4 D24 (c); fail closed)")
    if not isinstance(grid, dict) or any(type(grid.get(k)) is not int or grid.get(k) <= 0
                                         for k in ("num_envs", "episodes")):
        raise ValueError("dynamic_hazards_header.grid must be {'num_envs': N, 'episodes': E} with positive "
                         "integers (CONTEXT_R4 D24 (c)), got %r" % (grid,))
    return {"num_envs": grid["num_envs"], "episodes": grid["episodes"]}


def journal_grid_check(header_block, num_envs, quota):
    """Round T D24 (d) preflight: the run must lie inside the pool's offered admission grid.

    The evaluators keep stepping every env until the slowest reaches its per-env quota;
    episodes beyond the grid run at D0 and are never counted (D24 (a)). A run whose
    per-env quota exceeds the grid's episodes, or whose num_envs exceeds the grid's
    envs, would have to count such episodes, so it is refused before it starts.
    Returns the grid.
    """
    grid = journal_offered_grid(header_block)
    if int(quota) > grid["episodes"]:
        raise ValueError("journal run refused: per-env quota %d exceeds the D7 pool's offered grid "
                         "episodes %d (CONTEXT_R4 D24 (d)); admit a pool for this grid"
                         % (int(quota), grid["episodes"]))
    if int(num_envs) > grid["num_envs"]:
        raise ValueError("journal run refused: num_envs %d exceeds the D7 pool's offered grid num_envs %d "
                         "(CONTEXT_R4 D24 (d)); admit a pool for this grid" % (int(num_envs), grid["num_envs"]))
    return grid


def journal_contract_fields(base, env):
    """F5/F6 contract metrics of a finished episode (F7 round 4b item 3), or None.

    ``env.journal_contract_record(i)`` (F6) and/or ``env.journal_episode_results[i]``
    (F1-F5: E5 EpisodeResult or its evaluator_fields() dict), both latched by the env
    before the reset, exported beside journal_episode_record(i).
    """
    out = {}
    record = getattr(base, "journal_contract_record", None)
    if callable(record):
        out["journal_contract_record"] = json_value(record(int(env)))
    results = getattr(base, "journal_episode_results", None)
    if results is not None:
        try:
            value = results[int(env)]
        except (IndexError, KeyError, TypeError):
            value = None
        fields = getattr(value, "evaluator_fields", None)
        if callable(fields):
            value = fields()
        out["journal_episode_results"] = json_value(value)
    return out or None


def _journal_plans_module():
    for name in ("tasks._shared.journal_cfg_v2", "isaaclab_tasks.direct._shared.journal_cfg_v2"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("tasks._shared.journal_cfg_v2")


def outcome_semantics(base):
    """Whether success ends an episode in this family (see episode_outcome.py)."""
    declared = getattr(base, "stop_on_success", None)
    if isinstance(declared, bool):
        return {"fixed_horizon": not declared, "source": "base.stop_on_success"}
    module = getattr(type(base), "__module__", "") or ""
    for family in module.split("."):
        if family in FIXED_HORIZON_FAMILIES:
            return {"fixed_horizon": True, "source": "task family " + family}
        if family in STOP_ON_SUCCESS_FAMILIES:
            return {"fixed_horizon": False, "source": "task family " + family}
    return {"fixed_horizon": None, "source": "unknown task family"}
# D8: stamped protocol version and the fixed real-time budget used for the
# controller deadline-miss counts (60 Hz = the 1/120 s x decimation-2 control
# period of every shipped task; the actual period is stamped beside it).
EVALUATION_PROTOCOL_VERSION = "d8-evaluation-v1"
CONTROLLER_DEADLINE_S = 1.0 / 60.0
# G47 (v2.6.4): records[].min_clearance_m includes the D7 intruder. The env contracts sample the clearance to the
# STATIC geometry only (walls, obstacles); the D7 runtime samples the intruder's surface gap to the footprint while
# it is physically present (payload.runtime.min_sampled_surface_clearance_m, digest-sealed in the D7 stamp). The
# reported metric is the smaller of the two. Metric only: rewards, termination, controllers, the contract's own
# value (journal_contract.journal_episode_results) and the validator are unchanged; a fold that would change the
# episode outcome is NOT applied and the record is listed as unverified instead.
D7_CLEARANCE_RULE = ("min_clearance_m = min(static clearance of the env contract, D7 intruder surface clearance "
                     "payload.runtime.min_sampled_surface_clearance_m); v2.6.4 (G47)")


def d7_min_surface_clearance(envelope):
    """The D7 runtime's sampled minimum surface clearance of the intruder (m), or None (no runtime / no sample)."""
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    runtime = payload.get("runtime") if isinstance(payload, dict) else None
    value = runtime.get("min_sampled_surface_clearance_m") if isinstance(runtime, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def d7_clearance_fold(record, envelope, fixed_horizon):
    """(min_clearance_m to report, note): the record's value folded with the D7 intruder clearance (G47). note is
    None, or the reason the fold was NOT applied (it would change the episode outcome: never beyond the metric)."""
    current = record.get("min_clearance_m")
    gap = d7_min_surface_clearance(envelope)
    if gap is None:
        return current, None
    finite = (isinstance(current, (int, float)) and not isinstance(current, bool) and math.isfinite(float(current)))
    if finite and float(current) <= gap:
        return current, None
    candidate = dict(record, min_clearance_m=gap)
    if episode_outcome(candidate, fixed_horizon) != episode_outcome(record, fixed_horizon):
        return current, ("D7 intruder clearance %r not folded into min_clearance_m %r: it would change the "
                         "outcome" % (gap, current))
    return gap, None


NULLABLE_METRICS = (
    "tts_s", "path_length_m", "min_clearance_m", "contact_steps",
    "contact_seconds", "contact_longest_seconds", "contact_depth_mean_m",
    "d0_m", "route_geodesic_m",
)
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"}
# G6: cfg members kept under a dunder key so the Isaac Lab dict utilities skip them
# (tasks/_shared/journal_plan_slot.HIDDEN_CFG_MEMBERS; checked equal by
# scripts/tests_g6/test_g6_runtime_cfg.py). json_value shows them under their public names at their
# original position, so resolved_config is byte-identical to the plain attribute.
HIDDEN_CFG_MEMBERS = {"__journal_plan__": "journal_plan"}


def _isaac_member_view(to_dict, item):
    """One member as Isaac Lab class_to_dict renders it (dict.py:61-71), non-callables only."""
    if callable(item):
        raise TypeError("hidden cfg members are never callables")
    if hasattr(item, "__dict__") or isinstance(item, dict):
        return to_dict(item)
    if isinstance(item, (list, tuple)):
        return type(item)([to_dict(v) for v in item])
    return item


def _restore_hidden(value, rendered):
    """Re-insert HIDDEN_CFG_MEMBERS into a cfg's to_dict() view in __dict__ order."""
    members = getattr(value, "__dict__", None)
    if not isinstance(rendered, dict) or not members or not any(k in HIDDEN_CFG_MEMBERS for k in members):
        return rendered
    to_dict = type(value).to_dict                     # configclass's plain function: class_to_dict(obj)
    out = {}
    for key, item in members.items():
        if key in HIDDEN_CFG_MEMBERS:
            out[HIDDEN_CFG_MEMBERS[key]] = _isaac_member_view(to_dict, item)
        elif key in rendered:
            out[key] = rendered[key]
    for key, item in rendered.items():
        out.setdefault(key, item)
    return out


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_value(value, _seen=None):
    """Snapshot nested cfg objects, preserving inherited public cfg fields.

Callables are qualified names. Non-finite numbers become JSON null. Cyclic
objects and opaque objects are explicit markers rather than address-bearing
repr strings. No application module is imported by this serializer.
"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    if callable(value):
        return {"__callable__": getattr(value, "__module__", "") + "." +
                getattr(value, "__qualname__", type(value).__qualname__)}
    seen = set() if _seen is None else _seen
    kind = type(value).__module__ + "." + type(value).__qualname__
    if id(value) in seen:
        return {"__cycle__": kind}
    seen.add(id(value))
    try:
        if isinstance(value, dict):
            return {str(k): json_value(v, seen) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(v, seen) for v in value]
        if isinstance(value, (set, frozenset)):
            return [json_value(v, seen) for v in sorted(value, key=str)]
        if hasattr(value, "tolist"):
            return json_value(value.tolist(), seen)
        if hasattr(value, "to_dict"):
            return json_value(_restore_hidden(value, value.to_dict()), seen)
        fields = {}
        for cls in reversed(type(value).__mro__):
            for name, item in vars(cls).items():
                if not name.startswith("_") and not callable(item) and not isinstance(item, property):
                    fields[name] = getattr(value, name)
        if hasattr(value, "__dict__"):
            fields.update({HIDDEN_CFG_MEMBERS.get(k, k): v for k, v in vars(value).items()
                           if not HIDDEN_CFG_MEMBERS.get(k, k).startswith("_")})
        if fields:
            return {k: json_value(v, seen) for k, v in fields.items()}
        return {"__opaque_type__": kind}
    finally:
        seen.remove(id(value))


def read_path(cfg, path):
    value = cfg
    for component in path.split("."):
        value = value[component] if isinstance(value, dict) else getattr(value, component)
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path):
    """SHA256 of sorted 'relative POSIX path NUL file SHA256 LF' entries."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = []
    for file in root.rglob("*"):
        relative = file.relative_to(root)
        if any(part in CACHE_DIRS for part in relative.parts) or file.suffix in (".pyc", ".pyo"):
            continue
        if file.is_file():
            entries.append((relative.as_posix(), sha256_file(file)))
    digest = hashlib.sha256()
    for name, fingerprint in sorted(entries):
        digest.update((name + "\0" + fingerprint + "\n").encode("utf-8"))
    return digest.hexdigest()


LEGACY_HASH_ALGORITHM = "sha256-path-nul-content-v1"


def legacy_tree_digest(tasks_sha256, scripts_sha256):
    """C5 combined digest (cert_v3.py C5 :131-133); recomputed by the validator."""
    return hashlib.sha256(
        ("tasks\0" + tasks_sha256 + "\nscripts\0" + scripts_sha256 + "\n").encode("ascii")
    ).hexdigest()


def _environment_source(base):
    try:
        return Path(inspect.getfile(type(base))).resolve()
    except (TypeError, OSError):
        return None


def code_provenance(evaluator, base, release=None, modules=None):
    """Release code identity (usvbench-code-tree-v1; D2 item 9, CONTEXT_R3 decision 16).

    With a release descriptor (``release`` = code_tree.load_release(...)), the
    certificate carries the release's ``release_id``/``code_tree_sha256`` and
    every loaded USVBench module (evaluator scripts and task families, also
    when Isaac Lab imports a staged copy from ``isaaclab_tasks/direct``) must
    hash to its manifest entry, else CodeTreeError (no certificate).

    Without a descriptor the manifest is computed over the working tree
    ``<evaluator>/../{tasks,scripts}`` with the same inclusion rules; loaded
    modules are compared with it and any mismatch is reported (the writer
    records it as UNVERIFIED instead of refusing a development run).
    Checkpoints, media and upstream Isaac Lab tasks are never hashed.
    """
    evaluator = Path(evaluator).resolve()
    scripts_root = evaluator.parent
    code_root = scripts_root.parent
    source = _environment_source(base)
    task_roots = [code_root / "tasks"]
    if source is not None and source.name.endswith("_env.py"):
        # <task-tree>/<family>/<name>_env.py: the repository tasks/ folder or
        # Isaac Lab's staged direct/ folder; only manifest families map.
        task_roots.append(source.parent.parent)
    modules = sys.modules if modules is None else modules
    evaluator_bytes = code_tree.normalize_content(evaluator.read_bytes())
    common = {"hash_algorithm": code_tree.HASH_ALGORITHM, "inclusion": dict(code_tree.INCLUSION_SPEC),
              "evaluator_path": str(evaluator), "evaluator_sha256": sha256_file(evaluator),
              "evaluator_member": "scripts/" + evaluator.name,
              "environment_source": str(source) if source else None}
    if release is not None:
        records = release["records"]
        member = records.get(common["evaluator_member"])
        if member is None or member[2] != code_tree.sha256_bytes(evaluator_bytes):
            raise code_tree.CodeTreeError("the running evaluator is not the release's "
                                          + common["evaluator_member"])
        loaded = code_tree.verify_loaded_modules(records, scripts_root, task_roots, modules)
        if not loaded["verified"]:
            raise code_tree.CodeTreeError("loaded modules differ from release %s: %s"
                                          % (release["release_id"], "; ".join(loaded["mismatches"][:10])))
        return dict(common, source="release_descriptor", code_tree_sha256=release["code_tree_sha256"],
                    file_count=len(records), release_id=release["release_id"],
                    git_commit=release["git_commit"], git_tree=release["git_tree"],
                    release_descriptor={"path": release["descriptor_path"],
                                        "sha256": release["descriptor_sha256"]},
                    manifest_root=str(code_root), loaded_modules=loaded)
    try:
        _, records, digest = code_tree.manifest_from_tree(code_root)
    except code_tree.CodeTreeError as exc:
        return dict(common, source="unavailable", code_tree_sha256=None, file_count=0, release_id=None,
                    git_commit=None, git_tree=None, release_descriptor=None, manifest_root=str(code_root),
                    loaded_modules={"checked": 0, "verified": False, "mismatches": [str(exc)],
                                    "mismatch_count": 1})
    loaded = code_tree.verify_loaded_modules(records, scripts_root, task_roots, modules)
    return dict(common, source="working_tree", code_tree_sha256=digest, file_count=len(records),
                release_id=None, git_commit=None, git_tree=None, release_descriptor=None,
                manifest_root=str(code_root), loaded_modules=loaded)


def legacy_directory_provenance(evaluator, base):
    """C5's former code hash (every file under the env's parent folder).

    Superseded by code_provenance (D2 item 9: it hashed checkpoints, videos,
    raw line endings and upstream Isaac Lab tasks). Kept only so historical
    certificates can be re-derived; no writer calls it.
    """
    evaluator = Path(evaluator).resolve()
    task_root = evaluator.parent.parent / "tasks"
    source = None
    try:
        source = Path(inspect.getfile(type(base))).resolve()
        # All supplied concrete environments live in <task-tree>/<family>/*.py.
        if source.name.endswith("_env.py"):
            task_root = source.parent.parent
    except (TypeError, OSError):
        pass
    tasks_digest = tree_hash(task_root)
    scripts_digest = tree_hash(evaluator.parent)
    return {"tasks_root": str(task_root), "scripts_root": str(evaluator.parent),
            "environment_source": str(source) if source else None,
            "tasks_sha256": tasks_digest, "scripts_sha256": scripts_digest,
            "tree_sha256": legacy_tree_digest(tasks_digest, scripts_digest), "evaluator_path": str(evaluator),
            "evaluator_sha256": sha256_file(evaluator),
            "hash_algorithm": LEGACY_HASH_ALGORITHM}


def _distribution_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def runtime_metadata(base):
    """Best-effort runtime queries; absence is null, never a guessed version."""
    result = {"hostname": socket.gethostname(), "gpu_name": None, "gpu_device": str(base.device),
              "driver": None, "isaac_sim_version": _distribution_version("isaacsim", "isaacsim-core"),
              "isaac_lab_version": _distribution_version("isaaclab"), "physx_version": None}
    for module_name, field in (("isaaclab", "isaac_lab_version"), ("isaacsim", "isaac_sim_version")):
        module = sys.modules.get(module_name)
        version = getattr(module, "__version__", None)
        if isinstance(version, str):
            result[field] = version
    # Distribution/extension version is not necessarily the PhysX SDK version.
    physx = sys.modules.get("omni.physx")
    version_fn = getattr(physx, "get_physx_version", None)
    if callable(version_fn):
        try:
            result["physx_version"] = str(version_fn())
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None and str(base.device).startswith("cuda"):
        try:
            result["gpu_name"] = torch.cuda.get_device_name(base.device)
            # Driver version is host-wide; do not infer physical index from a
            # CUDA-visible index (CUDA_VISIBLE_DEVICES may have remapped it).
            query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=5, check=True)
            drivers = set(query.stdout.strip().splitlines())
            if len(drivers) == 1:
                result["driver"] = drivers.pop().strip()
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass
    # G6b / CONTEXT_R4 D28 (b): the eight RE-1 versions as this process sees them and whether they
    # match the pinned scripts/reference_environment.json (validate_cert_v3 refuses an official
    # certificate without the block or with match false).
    result["reference_environment"] = reference_environment.reference_block()
    return result


def timing_stats(samples):
    """Population std and linearly interpolated quantiles, in seconds."""
    values = sorted(float(x) for x in samples)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("timing samples must be finite and nonnegative")
    if not values:
        return dict(n=0, total=0.0, mean=None, std=None, min=None, p50=None, p95=None, p99=None, max=None)
    def percentile(p):
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(n=len(values), total=sum(values), mean=statistics.fmean(values),
                std=statistics.pstdev(values), min=values[0], p50=percentile(.50),
                p95=percentile(.95), p99=percentile(.99), max=values[-1])


def vessel_family(name):
    """Map an exact runtime vessel registry name to the study vessel label.

    Every WAM-V variant ('wamv', a future 'wamv_vrx...' profile) belongs to the
    'wamv' family; the exact runtime name stays in header.vessel, so the legacy
    0.8-scale placeholder and a VRX-derived profile remain distinguishable.
    """
    if name is None:
        return None
    text = str(name).lower().replace("-", "_")
    if text.startswith("wamv"):
        return "wamv"
    # E1: the D5 v2 profile 'blueboat_v2' is the same study vessel as 'blueboat'.
    return "blueboat" if text.startswith("blueboat") else text


def vehicles_v2_registry(base=None):
    """D5's v2 vessel registry (tasks/_shared/vehicles_v2.py) from the env's task tree, or None."""
    candidates = []
    module = (getattr(type(base), "__module__", "") or "") if base is not None else ""
    if module.count(".") >= 2:
        candidates.append(module.rsplit(".", 2)[0] + "._shared.vehicles_v2")
    candidates += ["tasks._shared.vehicles_v2", "isaaclab_tasks.direct._shared.vehicles_v2"]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module("tasks._shared.vehicles_v2")
    except ImportError:
        return None


def nominal_speed_reference(spec, name=None, registry=None):
    """Nominal (calm, undegraded) speed reference of the runtime vessel, or None.

    E1 / USV-10K: a vessel name registered in D5's v2 registry
    (``registry.VEHICLES_V2``) uses that profile's ``speeds.top_model_m_s``,
    the terminal speed of the profile's own thrust and drag laws (BlueBoat v2
    3.000 m/s with 2 x 40.207 N; WAM-V opt1 2.259 m/s). A spec object that
    carries ``speeds.top_model_m_s`` itself uses it. Legacy names keep the
    legacy value: solve thrust_fwd_n = surge_lin * v + surge_quad * v**2 for
    v > 0 (BlueBoat 80 N, 9.1, 5.9 -> 2.991 m/s, tasks/_shared/vehicles.py).
    Actuator/drag dose modifiers are deliberately NOT applied: this is the
    'nominal' basis. A condition-adjusted reference must be supplied
    explicitly and is labelled as such.
    """
    name = name if name is not None else getattr(spec, "name", None)
    table = getattr(registry, "VEHICLES_V2", None) if registry is not None else None
    if table is not None and name in table:
        value = float(table[name].speeds.top_model_m_s)
        if math.isfinite(value) and value > 0:
            return {"value_m_s": value,
                    "source": "vehicles_v2 registry %s speeds.top_model_m_s (D5 profile terminal speed)" % name}
    speeds = getattr(spec, "speeds", None)
    if speeds is not None and getattr(speeds, "top_model_m_s", None) is not None:
        value = float(speeds.top_model_m_s)
        if math.isfinite(value) and value > 0:
            return {"value_m_s": value, "source": "vessel spec speeds.top_model_m_s (%s)" % (name,)}
    try:
        thrust = float(spec.thrust_fwd_n)
        lin = float(spec.surge_lin)
        quad = float(spec.surge_quad)
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (thrust, lin, quad)) or thrust <= 0 or lin < 0 or quad < 0:
        return None
    if quad > 0:
        value = (-lin + math.sqrt(lin * lin + 4.0 * quad * thrust)) / (2.0 * quad)
    elif lin > 0:
        value = thrust / lin
    else:
        return None
    return {"value_m_s": value,
            "source": "vehicle_spec terminal speed: thrust_fwd_n=%r surge_lin=%r surge_quad=%r"
                      % (thrust, lin, quad)}


def grid_complete(records, num_envs, quota):
    pairs = [(r.get("env"), r.get("ep")) for r in records]
    return (len(pairs) == num_envs * quota and len(set(pairs)) == len(pairs)
            and all(type(env) is int and type(ep) is int and 0 <= env < num_envs and 0 <= ep < quota
                    for env, ep in pairs))


class CertificateRun:
    def __init__(self, *, base, env_cfg, evaluator, task, controller, controller_config,
                 checkpoint, eval_seed, level, episodes_requested, overrides,
                 argv, cli_args, suite_presets=None, provenance=None, runtime=None,
                 synchronize=None, clock=None):
        self.base = base
        self.num_envs = int(base.num_envs)
        if self.num_envs <= 0 or int(episodes_requested) <= 0:
            raise ValueError("num_envs and episodes_requested must be positive")
        self.grid = FixedEpisodeGrid(self.num_envs, episodes_requested)
        self.quota = self.grid.quota
        self._paths = EpisodePathObserver(base)
        self._controller_batches = []
        self._env_cfg = env_cfg
        self._frozen_level = getattr(env_cfg, "eval_level", None)
        self._clock = clock or time.perf_counter
        self._sync = synchronize or self._synchronize
        self._last = {}
        self._samples = [[[], []] for _ in range(self.num_envs)]
        self._initial_distances = None
        self._reset_distances = None
        self.unverified = []
        config = json_value(env_cfg)
        pool = getattr(env_cfg, "journal_d7_pool", None)
        if pool is not None and isinstance(config, dict) and "journal_d7_pool" in config:
            # F7: the admitted D7 pool is a ledger (every candidate); the
            # certificate identifies it by digest (header.dynamic_hazards_header).
            config["journal_d7_pool"] = {"pool_digest": json_value(getattr(pool, "pool_digest", None)),
                                         "keys_admitted": json_value(getattr(pool, "keys_admitted", None)),
                                         "snapshot": "digest only (full ledger in the admission evidence)"}
        actual_overrides = {}
        for key in overrides:
            actual_overrides[key] = json_value(read_path(env_cfg, key))
        # The baseline also accepts Hydra env.* assignments after argparse.
        # Capture their resolved readback, including nested/dict cfg paths.
        for token in argv:
            if token.startswith("env.") and "=" in token:
                key = token[4:].partition("=")[0]
                actual_overrides[key] = json_value(read_path(env_cfg, key))
        for key in ("seed", "curriculum_frozen", "eval_level", "scene.num_envs", "sim.device"):
            try:
                actual_overrides[key] = json_value(read_path(env_cfg, key))
            except (AttributeError, KeyError):
                pass
        presets = {k: v for k, v in config.items()
                   if "suite" in k.lower() or "preset" in k.lower()}
        presets.update(json_value(suite_presets or {}))
        journal_suites = cli_args.get("journal_suites")
        if cli_args.get("suite") or cli_args.get("suite_file") or journal_suites:
            # F7: the --suite command (validate_cert_v3 checks it against argv)
            # and the resolved JournalConditions stamp (F0 journal_config).
            presets["suite"] = list(cli_args.get("suite") or [])
            presets["suite_file"] = cli_args.get("suite_file")
            if journal_suites:
                presets["journal_conditions"] = json_value(journal_suites.get("stamp"))
        if cli_args.get("dose_preset") is not None:
            presets.update(dose_preset=cli_args["dose_preset"], preset_sha256=cli_args.get("preset_sha256"))
        runtime = json_value(runtime if runtime is not None else runtime_metadata(base))
        for key, value in runtime.items():
            if value is None:
                self.unverified.append("runtime." + key)
        # E1 (D2 item 9): release code identity. The descriptor path comes from
        # eval_protocol.preflight_protocol (--release-descriptor or the
        # USVBENCH_RELEASE_DESCRIPTOR environment variable), which verified the
        # release tree before the simulator started.
        self._release = None
        self._evaluator = evaluator
        descriptor = cli_args.get("release_descriptor")
        if provenance is None and descriptor:
            self._release = code_tree.load_release(descriptor)
            expected = cli_args.get("release_descriptor_sha256")
            if expected is not None and expected != self._release["descriptor_sha256"]:
                raise code_tree.CodeTreeError("release descriptor changed after preflight")
        provenance = provenance if provenance is not None else code_provenance(
            evaluator, base, release=self._release)
        if provenance.get("hash_algorithm") == code_tree.HASH_ALGORITHM:
            if provenance.get("source") != "release_descriptor":
                self.unverified.append("code identity: %s without a release descriptor"
                                       % provenance.get("source"))
            loaded = provenance.get("loaded_modules") or {}
            if not loaded.get("verified", False):
                self.unverified.append("code identity: loaded modules differ from the hashed tree: "
                                       + "; ".join(loaded.get("mismatches", [])[:5]))
        vessel = getattr(getattr(base, "vehicle_spec", None), "name", None)
        vessel = vessel or getattr(env_cfg, "vehicle", None)
        if vessel is None:
            # Explicitly labelled inference for older registered BlueBoat cfgs.
            lower = (task + " " + type(env_cfg).__name__).lower()
            vessel = "blueboat" if "blueboat" in lower else ("rov" if "rov" in lower else None)
            self.unverified.append("vessel: inferred from task/cfg name" if vessel else "vessel")
        requested_vessel = cli_args.get("vessel")
        if requested_vessel and vessel_family(vessel) != requested_vessel:
            # Finding 46: the vessel is fixed by the task id, never by a late
            # cfg mutation; a mismatch means the wrong cell would be certified.
            raise ValueError(f"requested vessel {requested_vessel!r} differs from runtime {vessel!r}")
        physics_version = declared_physics_version(env_cfg)
        if physics_version is None:
            self.unverified.append("physics_version: no declared physics release; use code hashes")
        # E1 (D2 item 8): hydrodynamics model and vessel profile from the cfg.
        hydrodynamics = declared_hydrodynamics_model(env_cfg)
        if hydrodynamics == "hydro_v2" and physics_version != "2":
            raise ValueError("hydrodynamics_model 'hydro_v2' requires physics_version 2")
        vessel_profile = declared_vessel_profile(env_cfg)
        level_distance = docking_level_distance(env_cfg, base, physics_version)
        # G6b: the journal docking stamp (table_field "journal_plan", docking_level_distance above)
        # carries no realised distance by design (validate_cert_v3.level_distance_errors); only the
        # legacy level-table stamp is checked here (a KeyError refused every journal docking run).
        if (level_distance is not None and level_distance.get("table_field") != "journal_plan"
                and level_distance["realised_distance_m"] is None):
            self.unverified.append("level_distance: realised docking spawn distance unavailable")
        self._outcome_semantics = outcome_semantics(base)
        checkpoint_path = str(Path(checkpoint).resolve()) if checkpoint else None
        controller_snapshot = json_value(controller_config)
        if "runtime_agent_config" in controller_snapshot and controller_snapshot["runtime_agent_config"] is None:
            self.unverified.append("controller.configuration.runtime_agent_config")
        def find_opaque(value, path):
            if isinstance(value, dict):
                if "__opaque_type__" in value or "__cycle__" in value:
                    self.unverified.append(path + ": opaque/cyclic configuration value")
                else:
                    for key, child in value.items():
                        find_opaque(child, path + "." + key)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    find_opaque(child, path + "." + str(index))
        find_opaque(config, "resolved_config")
        find_opaque(controller_snapshot, "controller.configuration")
        self.env_cfg = env_cfg
        self._conditions_api = _condition_certificate_module(base)
        # F7 (API_F0 section 7): journal envs certify through the mixin.
        self._journal = journal_certificate_mode(base, env_cfg)
        if self._journal:
            journal_header = base.journal_certificate_header(cli_args.get("pairing_treatment"))
            conditions = journal_header["conditions"]
        else:
            conditions = self._conditions_api.certificate_fields(base, env_cfg, include_off=True)["conditions"]
        self.header = dict(
            conditions=json_value(conditions),
            task_id=task, vessel=vessel, physics_version=physics_version,
            hydrodynamics_model=hydrodynamics, vessel_profile=vessel_profile,
            level_distance=level_distance, outcome_semantics=dict(self._outcome_semantics),
            controller={"name": controller, "configuration": controller_snapshot},
            checkpoint={"path": checkpoint_path, "sha256": sha256_file(checkpoint_path) if checkpoint_path else None},
            eval_seed=int(getattr(env_cfg, "seed", eval_seed)),
            level=int(getattr(env_cfg, "eval_level", level)), num_envs=self.num_envs,
            episodes_per_env=self.quota, episodes_requested=int(episodes_requested), complete=False,
            overrides=actual_overrides, suite_presets=presets, resolved_config=config,
            code=json_value(provenance), runtime=runtime, started_at=utc_now(), ended_at=None,
            command={"argv": list(argv), "parsed_args": json_value(cli_args)},
            timing={"clock": "perf_counter", "scope": "vectorized_batch_wall_seconds",
                    "cuda_synchronized": str(getattr(base, "device", "cpu")).startswith("cuda"),
                    "includes_warmup": True, "environment_includes_autoreset": True,
                    "amortized_per_env": False}, unverified=self.unverified)
        if self._journal:
            self.header["pairing"] = json_value(journal_header["pairing"])
            dynamic_block = json_value(journal_header["dynamic_hazards_header"])
            # round 4b: D16 per-dose counts include not_admitted (fail before the run, not after)
            d7_exposure.per_dose_counts(dynamic_block)
            # Round T D23: only pinned pools; D24 (d): the run lies inside the offered grid
            journal_pool_pinned_check(dynamic_block)
            journal_grid_check(dynamic_block, self.num_envs, self.quota)
            dynamic_block["pool_manifest_sha256"] = journal_pool_manifest_sha256(
                base, env_cfg, journal_header["dynamic_hazards_header"])
            self.header["dynamic_hazards_header"] = dynamic_block
            # G3: the appearing channel (present only with the appearing suite): pinned pool, D22 triples,
            # D24 (d) grid preflight, exactly like the D7 block
            if "appearing_hazards_header" in journal_header:
                appearing_block = json_value(journal_header["appearing_hazards_header"])
                appearing_exposure.per_dose_counts(appearing_block)
                appearing_exposure.pinned_check(appearing_block)
                appearing_exposure.grid_check(appearing_block, self.num_envs, self.quota)
                self.header["appearing_hazards_header"] = appearing_block
            # D19: runtime pool-path cfg fields (never science overrides)
            self.header["runtime_paths"] = json_value(cli_args.get("runtime_paths") or {})
            self.header["journal_mixin"] = str(journal_header["journal_mixin"])
            self.header["journal"] = journal_header_block(env_cfg)
            # G16: pinned vs used wind coefficients (the diagnostic-only windcoef.* override), read from the
            # env's hydro_v2 condition layer (journal_env_mixin's header blocks stay unchanged)
            wind_stamp = getattr(getattr(base, "_conditions", None), "wind_coefficient_stamp", None)
            if callable(wind_stamp):
                self.header["wind_coefficients"] = json_value(wind_stamp())
            # T25 (owner decision T1.35b): journal-default vs used roll/pitch damping (the diagnostic-only
            # hydro.rollpitch_zeta override), read from the env (journal_env_mixin's header blocks stay unchanged)
            rollpitch_stamp = getattr(base, "journal_rollpitch_damping_stamp", None)
            if callable(rollpitch_stamp):
                self.header["rollpitch_damping"] = json_value(rollpitch_stamp())
            # G10: green-water exclusion rule (A + B); required whenever the stack includes waves
            if "green_water_header" in journal_header:
                self._green_water_rule = _green_water_rule_module(self._conditions_api)
                self.header["green_water"] = json_value(journal_header["green_water_header"])
            elif journal_waves_enabled(conditions):
                raise ValueError("journal certificate with waves but the env publishes no green_water_header "
                                 "(G10 green-water detection; the family env must compose JournalEnvMixin)")
        # ---- D8 (v3plus) protocol stamps -------------------------------
        self.header["evaluation_protocol_version"] = EVALUATION_PROTOCOL_VERSION
        registration = cli_args.get("seed_registration")
        if not isinstance(registration, dict):
            # Direct evaluator call without eval_protocol.preflight_protocol:
            # explicitly a diagnostic, never an analysis-namespace run.
            registration = {"namespace": "diagnostic", "purpose": "diagnostic",
                            "seed": int(eval_seed), "block_id": None,
                            "registry_sha256": None}
            self.unverified.append("seed_namespace: preflight registration absent")
        self.header["seed_namespace"] = json_value(registration)
        # G16 (CONTEXT_R4 D23/D28 pattern): a run whose wind coefficients are not the vessel's pinned
        # values is a diagnostic run; an official run must never produce an official certificate.
        wind_coefficients_official_check(self.header.get("wind_coefficients"), registration)
        # T25 (T1.35b): the same for the roll/pitch damping override
        rollpitch_damping_official_check(self.header.get("rollpitch_damping"), registration)
        # G20b: the same for the appearing.gap_m trigger-gap override
        appearing_gap_override_official_check(self.header.get("appearing_hazards_header"), registration)
        # G20c (REV-G20b m5): and for the supplementary appearing dose D5
        appearing_supplementary_dose_official_check(self.header.get("appearing_hazards_header"), registration)
        self.header["scientific_environment"] = json_value(
            cli_args.get("scientific_environment") or {})
        self.header["runtime_environment"] = json_value(
            cli_args.get("runtime_environment") or {})
        self.header["override_precedence"] = (
            "task defaults < dedicated flags < --set (preset literals precede CLI --set "
            "in run_cell); protocol fields are fixed and conflicting values are rejected")
        # Verified run_cell plan ({path, sha256, configuration}) from preflight.
        cell = cli_args.get("cell_config_record")
        self.header["cell_config"] = json_value(cell) if cell is not None else None
        dt = getattr(base, "control_step_s", None)
        if dt is None:
            sim = getattr(env_cfg, "sim", None)
            dt = (getattr(sim, "dt", 0) or 0) * (getattr(env_cfg, "decimation", 0) or 0)
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            dt = 0.0
        self.control_step_s = dt if math.isfinite(dt) and dt > 0 else None
        horizon = getattr(base, "max_episode_length", None)
        if horizon is not None and self.control_step_s is not None:
            deadline = float(horizon) * self.control_step_s
            source = "base.max_episode_length * control_step_s"
        else:
            deadline = getattr(env_cfg, "episode_length_s", None)
            source = "env_cfg.episode_length_s"
        try:
            deadline = float(deadline) if deadline is not None else None
        except (TypeError, ValueError):
            deadline = None
        if deadline is None or not math.isfinite(deadline) or deadline <= 0:
            deadline, source = None, None
            self.unverified.append("resolved_deadline_s: no finite positive episode horizon")
        self.header["resolved_deadline_s"] = deadline
        self.header["resolved_deadline_source"] = source
        self.header["speed_reference"] = self._speed_reference(base, vessel, cli_args)
        self.header["path_length_semantics"] = {
            "path_length_m": "legacy_task_specific_window",
            "path_length_success_prefix_m": "reset_to_first_qualified_success",
            "path_length_full_episode_m": "reset_to_termination_or_truncation",
            "sampling": "horizontal_root_position_at_control_steps_before_autoreset"}
        self.header["timing"].update(controller_deadline_s=CONTROLLER_DEADLINE_S,
                                     deadline_comparison="strictly_greater",
                                     path_telemetry_in_environment_time=True)
        if not self._paths.enabled:
            self.unverified.append("separate path windows: pre-autoreset telemetry unavailable")

    def _speed_reference(self, base, vessel, cli_args):
        """Per-vessel speed reference for USV-10K efficiency (finding 51).

        Default: nominal terminal speed of the runtime vessel spec. An explicit
        CLI value must agree with it under basis 'nominal'; a different value
        is accepted only as basis 'condition_adjusted'. Missing -> null.
        """
        nominal = nominal_speed_reference(getattr(base, "vehicle_spec", None), name=vessel,
                                          registry=vehicles_v2_registry(base))
        speed = cli_args.get("speed_reference_m_s")
        basis = cli_args.get("speed_reference_basis") or "nominal"
        if speed is not None:
            speed = float(speed)
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError("speed reference must be finite and positive")
            if basis == "nominal":
                if nominal is None:
                    self.unverified.append("speed_reference: nominal value supplied, not derived from vessel spec")
                    return {"vessel": vessel, "value_m_s": speed, "basis": "nominal",
                            "source": "command line (vessel spec unavailable)"}
                if not math.isclose(speed, nominal["value_m_s"], rel_tol=1e-6):
                    raise ValueError("nominal speed reference %r differs from vessel spec value %r; "
                                     "use --speed-reference-basis condition_adjusted"
                                     % (speed, nominal["value_m_s"]))
            elif basis != "condition_adjusted":
                raise ValueError("unknown speed reference basis: %r" % (basis,))
            else:
                return {"vessel": vessel, "value_m_s": speed, "basis": basis,
                        "source": "command line"}
        if nominal is None:
            self.unverified.append("speed_reference: runtime vessel spec unavailable")
            return None
        return {"vessel": vessel, "value_m_s": nominal["value_m_s"], "basis": "nominal",
                "source": nominal["source"]}

    def _check_frozen(self):
        """Finding 56: reject any curriculum/level drift during certification."""
        for cfg in (self._env_cfg, getattr(self.base, "cfg", None)):
            if cfg is None:
                continue
            if hasattr(cfg, "curriculum_frozen") and cfg.curriculum_frozen is not True:
                raise RuntimeError("curriculum changed during certification")
            if hasattr(cfg, "eval_level") and cfg.eval_level != self._frozen_level:
                raise RuntimeError("evaluation level changed during certification")

    def _synchronize(self):
        torch = sys.modules.get("torch")
        if str(getattr(self.base, "device", "cpu")).startswith("cuda"):
            if torch is None:
                raise RuntimeError("CUDA timing requires the evaluator's loaded torch module")
            torch.cuda.synchronize(self.base.device)

    @contextmanager
    def _measure(self, name):
        self._sync()
        begin = self._clock()
        try:
            yield
        finally:
            self._sync()
            self._last[name] = self._clock() - begin

    def time_controller(self):
        return self._measure("controller")

    def time_environment(self):
        return self._measure("environment")

    def _distance_snapshot(self):
        values = getattr(self.base, "d0_per_env", None)
        if values is None:
            distance = getattr(self.base, "_horizontal_distance", None)
            values = distance() if callable(distance) else None
        return [json_value(float(value)) for value in values] if values is not None else None

    def after_reset(self):
        """Call after the initial reset, before any automatic reset can replace d0."""
        self._initial_distances = self._distance_snapshot()
        self._reset_distances = None
        self._paths.after_reset()

    def after_step(self):
        self._check_frozen()
        if set(self._last) != {"controller", "environment"}:
            raise RuntimeError("each step must time both controller and environment exactly once")
        for controller, environment in self._samples:
            controller.append(self._last["controller"])
            environment.append(self._last["environment"])
        self._controller_batches.append(self._last["controller"])
        self._paths.after_step()
        self._last.clear()
        self._reset_distances = None

    def discard_episode(self, env):
        env = int(env)
        self._samples[env] = [[], []]
        if self._initial_distances is not None:
            if self._reset_distances is None:
                self._reset_distances = self._distance_snapshot()
            if self._reset_distances is not None:
                self._initial_distances[env] = self._reset_distances[env]

    def next_episode(self, env):
        """The one shared fixed-grid collector rule (finding 48).

        Returns the preassigned episode index of the episode env just finished,
        or None when env has already delivered its quota; in that case the
        finished episode is discarded (its timing/distance state reset) and must
        not be recorded. Completion order can never replace a slow episode.
        """
        env = int(env)
        if not 0 <= env < self.num_envs:
            raise ValueError("environment index outside the declared grid")
        episode = self.grid.counts[env]
        if episode >= self.quota:
            self.discard_episode(env)
            return None
        return episode

    def finish_episode(self, env, record):
        env = int(env)
        expected = self.grid.counts[env]
        if record.get("env") != env or record.get("ep") != expected or expected >= self.quota:
            raise ValueError("episode record does not match the next assigned grid key")
        result = json_value(record)
        reasons = getattr(self.base, "termination_reason", None)
        reason = REASONS.get(int(reasons[env]), "unknown") if reasons is not None else "unknown"
        if reason == "running":
            reason = "unknown"
        result["termination_reason"] = reason
        if reason == "unknown":
            self.unverified.append(f"termination_reason: env={env} ep={record.get('ep')}")
        for key in NULLABLE_METRICS:
            result.setdefault(key, None)
        if result["d0_m"] is None and self._initial_distances is not None:
            result["d0_m"] = self._initial_distances[env]
        if result["route_geodesic_m"] is None:
            route = getattr(self.base, "episode_route_length", None)
            if route is not None:
                result["route_geodesic_m"] = json_value(float(route[env]))
        for key in ("terminated", "truncated"):
            if key not in result:
                raise ValueError("writer must supply actual step mask: " + key)
        result["controller_compute_time_s"] = timing_stats(self._samples[env][0])
        result["environment_step_time_s"] = timing_stats(self._samples[env][1])
        result["controller_deadline_misses_60hz"] = sum(
            x > CONTROLLER_DEADLINE_S for x in self._samples[env][0])
        result["episode_elapsed_s"] = (len(self._samples[env][0]) * self.control_step_s
                                       if self.control_step_s is not None else None)
        paths = self._paths.take(env, bool(result.get("success")))
        if (paths["path_length_success_prefix_m"] is None and result.get("success") is True
                and reason == "success"):
            # The episode stopped on its success step, so the complete episode
            # IS the success prefix (reach tasks have no _success latch).
            paths["path_length_success_prefix_m"] = paths["path_length_full_episode_m"]
        result.update(paths)
        stage = {}
        for attr, key in (("episode_max_phase", "max_phase"), ("episode_gates_passed", "gates_passed")):
            values = getattr(self.base, attr, None)
            if values is not None:
                stage[key] = int(values[env])
        result["stage_attainment"] = stage or None
        # E1 (D2 item 21): one mapping table, shared with the validator.
        result["outcome"] = episode_outcome(result, self._outcome_semantics["fixed_horizon"])
        if result["outcome"] is None and reason != "unknown":
            self.unverified.append(f"outcome: reason {reason!r} contradicts success={result.get('success')!r} "
                                   f"env={env} ep={record.get('ep')}")
        # Finished-episode condition stamp latched by the env at reset (C4),
        # or the explicit off stamp when the env has no condition layer.
        # F7: journal envs latch {conditions, dynamic_hazards, scenario_hashes}
        # BEFORE the reset (journal_env_mixin.journal_dones); the record takes
        # all three, the D7 envelope digest verified here and by the validator.
        if self._journal:
            journal_record = json_value(self.base.journal_episode_record(env))
            latched = journal_record["conditions"]
            envelope = journal_record.get("dynamic_hazards")
            if envelope is None:
                raise ValueError(f"env {env}: journal episode without a D7 stamp (D0 included, D7 B08)")
            self._conditions_api.verify_dynamic_hazard_stamp(envelope)
            # G47 (v2.6.4): the reported clearance includes the D7 intruder (metric only; outcome unchanged)
            folded, note = d7_clearance_fold(result, envelope, self._outcome_semantics["fixed_horizon"])
            if note is not None:
                self.unverified.append(f"min_clearance_m: env={env} ep={record.get('ep')} {note}")
            result["min_clearance_m"] = folded
            # D16/D20: admitted or an explicit not_admitted stamp in the families' wire form
            exposure, d7_reasons = d7_exposure.exposure_of(envelope)
            if exposure == d7_exposure.NOT_ADMITTED and d7_exposure.BEYOND_OFFERED_GRID in d7_reasons:
                # Round T D24 (d): an episode beyond the offered grid is never counted
                raise ValueError(f"env {env} ep {expected}: counted journal record of a key beyond the offered "
                                 f"D7 grid (reason {d7_exposure.BEYOND_OFFERED_GRID!r}, CONTEXT_R4 D24)")
            for key in ("dynamic_hazards", "scenario_hashes"):
                if key in result and result[key] != journal_record[key]:
                    raise ValueError(f"writer-supplied {key} for env {env} differs from the journal latch")
                result[key] = journal_record[key]
            # G3: the appearing stamp (digest verified; a counted beyond-grid key is refused, D24 (d))
            if "appearing_hazards_header" in self.header:
                appearing = journal_record.get("appearing_hazards")
                if appearing is None:
                    raise ValueError(f"env {env}: journal episode without an appearing stamp (appearing suite on)")
                kind, appearing_reasons = appearing_exposure.exposure_of(appearing)
                if kind == appearing_exposure.NOT_ADMITTED and \
                        appearing_exposure.BEYOND_OFFERED_GRID in appearing_reasons:
                    raise ValueError(f"env {env} ep {expected}: counted journal record of a key beyond the offered "
                                     "appearing grid (CONTEXT_R4 D24)")
                if "appearing_hazards" in result and result["appearing_hazards"] != appearing:
                    raise ValueError(f"writer-supplied appearing_hazards for env {env} differs from the journal latch")
                result["appearing_hazards"] = appearing
            # G10: the per-episode green-water record (waves in the stack), latched before the reset
            if "green_water" in self.header:
                green = journal_record.get("green_water")
                if not isinstance(green, dict):
                    raise ValueError(f"env {env}: journal episode with waves but no green_water record (G10)")
                if "green_water" in result and result["green_water"] != green:
                    raise ValueError(f"writer-supplied green_water for env {env} differs from the journal latch")
                result["green_water"] = green
                # G18: read-only diagnostics beside the rule/1 record (never part of the verdict)
                diagnostics = journal_record.get("green_water_diagnostics")
                if diagnostics is not None:
                    result["green_water_diagnostics"] = diagnostics
                # T25 (owner decision T1.35d): report-only point sets beside it (never part of any verdict)
                report = journal_record.get("green_water_report")
                if report is not None:
                    result["green_water_report"] = report
            contract = journal_contract_fields(self.base, env)
            if contract is not None:
                if "journal_episode_results" in contract and contract["journal_episode_results"] is None:
                    self.unverified.append(f"journal_episode_results: env={env} ep={record.get('ep')} not latched")
                result["journal_contract"] = contract
        else:
            latched = json_value(self._conditions_api.episode_certificate_fields(
                self.base, env, self.env_cfg, include_off=True)["conditions"])
        if "conditions" in result and result["conditions"] != latched:
            raise ValueError(f"writer-supplied condition stamp for env {env} differs from the env latch")
        result["conditions"] = latched
        # Advance the shared grid only once the record is fully assembled.
        self.grid.accept(env)
        self.discard_episode(env)
        return result

    def _canonical_aliases(self, payload):
        """Legacy top-level controller/checkpoint aliases must name the header's values.

        D2 item 10: an absolute checkpoint alias that resolves to the header's
        (resolved) checkpoint is rewritten to that exact string; a relative
        alias must be a path suffix of it (convergence_curve.py writes the
        bare file name); anything else is a contradiction and is refused.
        """
        if "controller" in payload and payload["controller"] != self.header["controller"]["name"]:
            raise ValueError("legacy controller alias %r differs from header controller %r"
                             % (payload["controller"], self.header["controller"]["name"]))
        if "checkpoint" not in payload:
            return
        alias, claimed = payload["checkpoint"], self.header["checkpoint"]["path"]
        if alias is None or claimed is None:
            if alias is not None or claimed is not None:
                raise ValueError("legacy checkpoint alias and header checkpoint disagree on presence")
            return
        text = str(alias)
        if os.path.isabs(text):
            if os.path.normcase(os.path.realpath(text)) != os.path.normcase(claimed):
                raise ValueError("legacy checkpoint alias %r is not the certified checkpoint %r" % (text, claimed))
            payload["checkpoint"] = claimed
        elif not claimed.replace("\\", "/").endswith("/" + text.replace("\\", "/")):
            raise ValueError("legacy checkpoint alias %r is not a suffix of %r" % (text, claimed))

    def build(self, legacy_payload):
        payload = json_value(legacy_payload)
        records = payload.get("records", [])
        self._canonical_aliases(payload)
        if self._release is not None:
            # C6 6.4: re-verify after the job; a changed tree never certifies.
            code = self.header["code"]
            evaluator = Path(self._evaluator).resolve()
            task_roots = [evaluator.parent.parent / "tasks"]
            if code.get("environment_source") and code["environment_source"].endswith("_env.py"):
                task_roots.append(Path(code["environment_source"]).parent.parent)
            after = code_tree.verify_loaded_modules(self._release["records"], evaluator.parent,
                                                    task_roots, sys.modules)
            if not after["verified"]:
                raise code_tree.CodeTreeError("release files changed during the run: "
                                              + "; ".join(after["mismatches"][:10]))
        self.header["ended_at"] = utc_now()
        self.header["complete"] = grid_complete(records, self.num_envs, self.quota)
        if "green_water" in self.header:
            # G10: cell summary over the COUNTED episodes: episodes with green water / episodes, the share,
            # the formula probability and the verdict (rule in conditions/green_water_rule.py)
            self.header["green_water"]["cell"] = self._green_water_rule.cell_summary(
                self.header["green_water"]["formula"]["probability"], [r["green_water"] for r in records])
            # G18: non-binding diagnostics over the same counted episodes (settled share, event rate,
            # simulated per-cycle exceedance), only when every counted record carries them
            diagnostics = [r.get("green_water_diagnostics") for r in records]
            if records and all(isinstance(d, dict) for d in diagnostics):
                labels = [p["label"] for p in self.header["green_water"]["detection"]["points"]]
                self.header["green_water"]["diagnostics"] = self._green_water_rule.diagnostics_summary(
                    self.header["green_water"]["formula"]["probability"], [r["green_water"] for r in records],
                    diagnostics, labels)
            # T25 (owner decision T1.35d): non-binding summary of the report-only point sets (WAM-V deck edge)
            reports = [r.get("green_water_report") for r in records]
            if "report_points" in self.header["green_water"] and records and all(isinstance(x, dict) for x in reports):
                self.header["green_water"]["report"] = self._green_water_rule.report_summary(
                    self.header["green_water"]["report_points"], reports)
            # T25 (owner decision T1.35c): the BINDING exclusion verdict is rule/2 (formula or simulated per-cycle
            # exceedance after the settling window, worst point, both > 0.05); rule/1 stays in the cell, reported
            self.header["green_water"]["exclusion"] = self._green_water_rule.exclusion_block(
                self.header["green_water"]["formula"]["probability"], self.header["green_water"].get("diagnostics"),
                self.header["green_water"]["cell"], self.header["green_water"]["vessel"])   # G28: + probe stamp
        if isinstance(self.header.get("dynamic_hazards_header"), dict):
            # G23b (owner decision 2026-10-03, NC 09): exposure metric A of the cell over the COUNTED episodes
            # (threatened share, realized TTC / t_avoid at the first detection), recomputed by validate_cert_v3
            self.header["dynamic_hazards_header"]["exposure_summary"] = d7_exposure.exposure_summary(
                [r["dynamic_hazards"] for r in records if "dynamic_hazards" in r],
                self.header["dynamic_hazards_header"]["suite"]["dose"])
        self.header["controller_latency"] = {
            "controller": self.header["controller"]["name"],
            "statistics_s": timing_stats(self._controller_batches),
            "deadline_s": CONTROLLER_DEADLINE_S,
            "control_step_s": self.control_step_s,
            "deadline_misses_60hz": sum(x > CONTROLLER_DEADLINE_S for x in self._controller_batches),
            "scope": "each_vectorized_controller_call_once_including_quota_tail"}
        payload.update(schema_version=3, schema_minor_version=SCHEMA_MINOR_VERSION,
                       header=json_value(self.header))
        # Retain existing reader entry points with authoritative values.
        payload.update(task=self.header["task_id"], seed=self.header["eval_seed"],
                       level=self.header["level"], num_envs=self.num_envs,
                       episodes_per_env=self.quota, episodes_requested=self.header["episodes_requested"],
                       complete=self.header["complete"], overrides=self.header["overrides"])
        return payload
