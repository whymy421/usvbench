"""Shared ``--set FIELD=VALUE`` parsing for the evaluation entry points.

Three scripts grew their own copy of this loop (eval_obs_bridge, eval_mppi,
eval_v6_frozen) and each one only reached TOP-LEVEL cfg attributes. The wave
ladder needs the opposite: its rungs live one level down, in
``cfg.sea_state.hs_range`` and ``cfg.sea_state.tp_range``, so a top-level-only
override cannot pin a rung at all.

This module is the single implementation. It adds

  * dotted paths, so ``sea_state.hs_range`` resolves through the owner chain;
  * tuple/list values, so a range can be pinned (``0.6,0.6``);
  * type coercion driven by the value ALREADY on the cfg, so a field never
    silently changes type;
  * refusal -- never a silent no-op -- when a path, an arity or a cast is
    wrong. A typo in a queue script must stop the run, not quietly evaluate
    the default band and file the result as a rung.

Behaviour contract: with an empty assignment list this module touches nothing,
so every certified command line that predates it is bit-identical.

Pure Python. Imports neither Isaac Lab nor torch, so the tests run on CPU.
"""

from __future__ import annotations

import math

from typing import Any, Iterable, Sequence


__all__ = [
    "OverrideError",
    "JOURNAL_PROTECTED",
    "JOURNAL_RUNTIME_FIELDS",
    "apply_overrides",
    "check_journal_assignments",
    "coerce_like",
    "is_journal_cfg",
    "journal_protected_reason",
    "journal_validate",
    "resolve_owner",
    "split_assignment",
    "split_runtime_assignments",
    "validate_assignments",
]

# D12 (CONTEXT_R4, F7): fields a --set may never change on a journal cfg
# (journal_contract_version == 2). The vessel, the asset and every physics
# field are built at construction (C1 finding 46); the journal_* fields, the
# episode cap, the layout, the obstacle capacity, the legacy appearance and
# sea-state observation switches and the time step are part of the frozen task
# contract (journal_runtime_cfg_v2.journal_validate). Conditions are set with
# --suite (tasks/_shared/conditions/journal_config.py), never by --set, so the
# resolved JournalConditions and cfg.conditions can never disagree.
JOURNAL_PROTECTED = {
    "vehicle": "the vessel is selected by the vessel-specific journal id",
    "robot_cfg": "asset field (built at construction)",
    "underwater_physics_cfg": "physics field (hydro_v2 profile, physics_version 2)",
    "physics_version": "physics field",
    "hydrodynamics_model": "physics field",
    "vessel_profile": "physics field (vessel profile stamp)",
    "vessel_profile_status": "physics field (vessel profile stamp)",
    "journal_vehicle_options": "physics field (resolve_journal_vehicle options)",
    "episode_length_s": "the episode cap is derived from the frozen route budget",
    "layout_mode": "the journal layout is frozen",
    "max_obstacles": "obstacle capacity is frozen (static capacity + D7 slot)",
    "appear_count": "legacy appearance is disabled; D7 owns dynamic hazards",
    "obs_sea_state": "journal controllers never observe the sea state",
    "decimation": "control decimation is frozen at 2 (CONTEXT_R4 D2)",
    "sim.dt": "physics step is frozen at 1/120 s (CONTEXT_R4 D2)",
    "sim.gravity": "physics field",
    "conditions": "condition suites are set with --suite (journal_config), not --set",
    "condition_presets": "condition suites are set with --suite (journal_config), not --set",
    "observation_contract": "the journal observation contract is frozen",
}


# D19 (CONTEXT_R4 round 4b): where a family loads its admitted D7 pool from is a RUNTIME
# choice, not science: the env checks the pinned pool/manifest SHA-256 whatever the path,
# and the certificate stamps the pool digest and the pinned manifest digest
# (dynamic_hazards_header.pool_digest / pool_manifest_sha256). These cfg fields are
# therefore allow-listed on journal cfgs (never refused by D12), kept out of the science
# overrides and stamped separately (header.runtime_paths). F1 journal_d7_pool_dir, F3/F6
# journal_d7_pool_path, F5 d7_pool_dir / d7_pool_file.
JOURNAL_RUNTIME_FIELDS = {
    "journal_d7_pool_path": "D7 pool file or directory (F3, F6)",
    "journal_d7_pool_dir": "D7 pool directory (F1)",
    "d7_pool_dir": "D7 pool directory (F5)",
    "d7_pool_file": "D7 pool file name inside d7_pool_dir (F5)",
}


def split_runtime_assignments(assignments: Iterable[str] | None) -> tuple[list[str], dict[str, str]]:
    """(science assignments, {runtime field: value}) of a --set list (D19)."""
    science: list[str] = []
    runtime: dict[str, str] = {}
    for assignment in assignments or ():
        path, raw = split_assignment(assignment)
        if path in JOURNAL_RUNTIME_FIELDS:
            if path in runtime and runtime[path] != raw:
                raise OverrideError(f"conflicting repeated --set {path}")
            if not raw:
                raise OverrideError(f"--set {path} needs a non-empty path")
            runtime[path] = raw
        else:
            science.append(assignment)
    return science, runtime


def is_journal_cfg(cfg: Any) -> bool:
    return int(getattr(cfg, "journal_contract_version", 0) or 0) == 2


def journal_protected_reason(path: str) -> str | None:
    """Why ``path`` may not be overridden on a journal cfg, or None."""
    head = path.split(".", 1)[0]
    if path in JOURNAL_RUNTIME_FIELDS:
        return None
    if head.startswith("journal_"):
        return "journal_* fields are the frozen task contract"
    for key, reason in JOURNAL_PROTECTED.items():
        if path == key or path.startswith(key + ".") or (("." not in key) and head == key):
            return reason
    return None


def check_journal_assignments(assignments: Iterable[str] | None) -> None:
    """Refuse every D12-protected path (no cfg needed; used before launch too)."""
    for assignment in assignments or ():
        path, _ = split_assignment(assignment)
        reason = journal_protected_reason(path)
        if reason is not None:
            raise OverrideError(f"--set {path} is refused on journal cfgs (D12): {reason}")


def journal_validate(cfg: Any) -> None:
    """Run the journal cfg's own journal_validate (journal_runtime_cfg_v2), fail closed.

    Looked up in the module that defines the cfg class (every journal cfg
    class lives in tasks/_shared/journal_runtime_cfg_v2.py, which defines
    journal_validate); a journal cfg without one is refused.
    """
    import sys

    module = sys.modules.get(type(cfg).__module__)
    validator = getattr(module, "journal_validate", None) if module is not None else None
    if not callable(validator):
        raise OverrideError(f"journal cfg {type(cfg).__name__} has no journal_validate in "
                            f"{type(cfg).__module__}; refusing to evaluate an unvalidated journal cfg")
    try:
        validator(cfg)
    except (TypeError, ValueError, AttributeError) as exc:
        raise OverrideError(f"journal cfg failed journal_validate: {exc}") from exc


class OverrideError(ValueError):
    """A --set assignment could not be applied. Always fatal, never skipped."""


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def split_assignment(assignment: str) -> tuple[str, str]:
    """Split ``FIELD=VALUE`` into its two halves, or refuse."""
    path, sep, raw = assignment.partition("=")
    path = path.strip()
    if not sep or not path:
        raise OverrideError(
            f"--set expects FIELD=VALUE, got {assignment!r}"
        )
    return path, raw.strip()


def resolve_owner(cfg: Any, path: str) -> tuple[Any, str]:
    """Walk a dotted path and return the (owner, leaf_name) that holds it.

    ``resolve_owner(cfg, "sea_state.hs_range")`` returns ``(cfg.sea_state,
    "hs_range")``. Every intermediate hop must already exist: this never
    creates attributes, because a path that does not exist on the frozen cfg
    is a typo, not a request.
    """
    parts = path.split(".")
    if any(not part.isidentifier() or part.startswith("_") for part in parts):
        raise OverrideError(f"invalid public cfg path: {path!r}")
    owner = cfg
    for index, part in enumerate(parts[:-1]):
        if not hasattr(owner, part):
            walked = ".".join(parts[: index + 1])
            raise OverrideError(
                f"--set path {path!r} is not a cfg field: {walked!r} does not exist"
            )
        owner = getattr(owner, part)
        if owner is None:
            walked = ".".join(parts[: index + 1])
            raise OverrideError(
                f"--set path {path!r} cannot be resolved: {walked!r} is None"
            )
    leaf = parts[-1]
    if not hasattr(owner, leaf):
        raise OverrideError(
            f"--set path {path!r} is not a cfg field: {leaf!r} does not exist "
            f"on {type(owner).__name__}"
        )
    return owner, leaf


def _coerce_scalar(raw: str, current: Any, path: str) -> Any:
    """Cast one token to the type of the value already on the cfg."""
    if isinstance(current, bool):
        lowered = raw.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise OverrideError(
            f"--set {path}: {raw!r} is not a boolean "
            f"(use one of {_TRUE + _FALSE})"
        )
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            raise OverrideError(f"--set {path}: {raw!r} is not an int") from None
    if isinstance(current, float):
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError('nonfinite')
            return value
        except ValueError:
            raise OverrideError(f"--set {path}: {raw!r} is not a float") from None
    if isinstance(current, str):
        return raw
    raise OverrideError(
        f"--set {path}: cannot infer a cast from the current value "
        f"{current!r} of type {type(current).__name__}"
    )


def coerce_like(raw: str, current: Any, path: str = "<field>") -> Any:
    """Cast ``raw`` to the shape and type of ``current``.

    Sequences keep their container type and their arity: pinning a two-element
    range takes two values (``0.6,0.6``). Surrounding brackets or parentheses
    are accepted so a shell-quoted ``(0.6,0.6)`` works too. A mismatched arity
    is refused rather than padded -- silently broadcasting one value across a
    range is exactly how a ladder rung ends up mislabelled.
    """
    if isinstance(current, (tuple, list)):
        stripped = raw.strip()
        for opener, closer in (("(", ")"), ("[", "]")):
            if stripped.startswith(opener) and stripped.endswith(closer):
                stripped = stripped[1:-1]
                break
        tokens = [token.strip() for token in stripped.split(",") if token.strip()]
        if len(tokens) != len(current):
            raise OverrideError(
                f"--set {path}: expected {len(current)} comma-separated values "
                f"to match the current {type(current).__name__} {tuple(current)!r}, "
                f"got {len(tokens)}"
            )
        values = [
            _coerce_scalar(token, element, path)
            for token, element in zip(tokens, current)
        ]
        return type(current)(values)
    return _coerce_scalar(raw, current, path)


def apply_overrides(
    cfg: Any,
    assignments: Iterable[str] | None,
    *,
    echo: bool = True,
    label: str = "set",
) -> list[tuple[str, Any]]:
    """Apply every ``FIELD=VALUE`` assignment to ``cfg`` in order.

    Returns the ``(path, value)`` pairs actually written, so a caller can
    stamp them into its result record. After the complete non-empty batch, a
    cfg-provided ``validate_after_overrides`` method is called exactly once.
    Raises ``OverrideError`` on a bad assignment or final validation failure:
    the run is meant to die here, not to continue on an invalid cfg.
    """
    applied: list[tuple[str, Any]] = []
    journal = is_journal_cfg(cfg)
    if journal:
        # D12: refuse the whole batch before anything is written.
        check_journal_assignments(assignments)
    for assignment in assignments or ():
        path, raw = split_assignment(assignment)
        owner, leaf = resolve_owner(cfg, path)
        current = getattr(owner, leaf)
        if current is None:
            raise OverrideError(
                f"--set {path}: the current value is None, so there is no "
                f"type to cast {raw!r} to"
            )
        value = coerce_like(raw, current, path)
        setattr(owner, leaf, value)
        applied.append((path, value))
        if echo:
            print(f"{label} {path}={value}", flush=True)
    if applied and hasattr(cfg, "validate_after_overrides"):
        try:
            cfg.validate_after_overrides()
        except (TypeError, ValueError) as exc:
            raise OverrideError(f"invalid cfg after --set: {exc}") from exc
    if applied and journal:
        journal_validate(cfg)
    return applied


def validate_assignments(
    assignments: Sequence[str] | None, fail
) -> None:
    """Pre-flight the FIELD=VALUE shape before Isaac Sim is launched.

    ``fail`` is the caller's error sink (``parser.error``), so a malformed
    flag exits 2 in argparse's own idiom instead of after a minute of app
    startup. The cfg itself does not exist yet here, so only the shape is
    checked; path resolution happens later in ``apply_overrides``.
    """
    for assignment in assignments or ():
        try:
            split_assignment(assignment)
        except OverrideError as exc:
            fail(str(exc))
