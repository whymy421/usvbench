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

from typing import Any, Iterable, Sequence


__all__ = [
    "OverrideError",
    "apply_overrides",
    "coerce_like",
    "resolve_owner",
    "split_assignment",
    "validate_assignments",
]


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
            return float(raw)
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
    stamp them into its result record. Raises ``OverrideError`` on the first
    bad assignment, leaving the earlier ones applied: the run is meant to die
    here, not to continue on a half-configured cfg.
    """
    applied: list[tuple[str, Any]] = []
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
