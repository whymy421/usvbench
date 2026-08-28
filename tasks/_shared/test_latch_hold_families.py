"""CPU test for the PER-EPISODE RECORDING LATCH of the three HOLD-SEMANTICS families.

WHAT THIS FILE OWNS
-------------------
One line in each of the three task families whose success criterion is a
sustained hold rather than an arrival::

    tasks/station_keeping/station_keeping_env.py
    tasks/station_keeping_boat/station_keeping_boat_env.py
    tasks/docking/docking_env.py
        self._episode_finished.copy_(time_out)

WHY THIS FILE EXISTS
--------------------
``tasks/_shared/test_episode_latch_families.py`` documents what narrowing this
latch already cost the project once: on ``Isaac-USV-PathHazard-Direct-v2`` the
form ``copy_(time_out)`` meant every episode that ended on its OUTCOME skipped
the ``if len(completed_ids) > 0:`` recording block, so each certificate row
carried the PREVIOUS episode's success, time, clearance and scenario hash.

That test covers hazard_nav, harbor_mission and path_hazard.  It does NOT cover
the three families above -- and those three are exactly the ones that write
``episode_max_hold_s``, the graded quantity the three-tier station-keeping
report is computed from.  A latch that fires on the wrong episodes would put a
previous episode's hold time next to this episode's success flag, and the
consistency auditor could not see it: both fields would be shifted together.

The three families are SAFE TODAY, but only because each one hard-codes
``terminated = torch.zeros_like(time_out)`` -- they never end early, so
``copy_(time_out)`` and ``copy_(terminated | time_out)`` are the same set.  That
is a property nothing currently pins.  Add one early termination (a berth-wall
contact that ends the episode, an out-of-bounds abort, a stop-on-success) and
the narrow latch silently becomes the bug that already happened once.

So this test asserts the DISJUNCTION rather than either branch:

    either ``terminated`` is provably the constant zero tensor,
    or the latch condition mentions ``terminated``.

Whichever way a future edit goes, one of the two arms must hold.

Static analysis only -- parses the sources with ``ast`` and never imports them,
so it runs anywhere without Isaac Sim.

Run: python tasks/_shared/test_latch_hold_families.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FAMILIES = {
    "station_keeping": REPO / "tasks" / "station_keeping" / "station_keeping_env.py",
    "station_keeping_boat": (
        REPO / "tasks" / "station_keeping_boat" / "station_keeping_boat_env.py"
    ),
    "docking": REPO / "tasks" / "docking" / "docking_env.py",
}

# The families the SIBLING tests already own, kept here only so a reader can see
# that the two files together cover every env that latches per-episode records.
ALREADY_COVERED = ("hazard_nav", "harbor_mission", "path_hazard")


def _dones_body(path: Path) -> list[ast.stmt]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_dones":
            return list(node.body)
    raise AssertionError(f"{path.name}: no _get_dones found")


def _latch_call(body: list[ast.stmt]) -> ast.Call:
    """The `self._episode_finished.copy_(...)` call inside _get_dones."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "copy_":
            continue
        owner = func.value
        if isinstance(owner, ast.Attribute) and owner.attr == "_episode_finished":
            return node
    raise AssertionError("no self._episode_finished.copy_(...) in _get_dones")


def _terminated_is_constant_zero(body: list[ast.stmt]) -> bool:
    """True when every binding of `terminated` is torch.zeros_like(...).

    Requiring EVERY binding (not just the last) means a future edit that adds a
    conditional real termination alongside the zeros default still trips the
    test rather than sneaking through on the final assignment.
    """
    bindings = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "terminated":
                bindings.append(node.value)
    if not bindings:
        return False
    for value in bindings:
        if not isinstance(value, ast.Call):
            return False
        func = value.func
        if not (isinstance(func, ast.Attribute) and func.attr == "zeros_like"):
            return False
    return True


def test_every_hold_family_has_a_sound_latch() -> None:
    for name, path in FAMILIES.items():
        assert path.exists(), f"{name}: {path} is missing"
        body = _dones_body(path)
        call = _latch_call(body)
        mentions_terminated = any(
            isinstance(sub, ast.Name) and sub.id == "terminated"
            for sub in ast.walk(call)
        )
        constant_zero = _terminated_is_constant_zero(body)
        assert mentions_terminated or constant_zero, (
            f"{name}: the per-episode latch is `copy_(time_out)` while "
            f"`terminated` is no longer provably zero. Episodes that end on "
            f"their outcome would skip the recording block and every "
            f"certificate row would carry the PREVIOUS episode's fields -- the "
            f"exact regression documented in test_episode_latch_families.py. "
            f"Either widen the latch to `terminated | time_out` or keep "
            f"`terminated` a constant zeros_like tensor."
        )
        print(
            f"PASS {name}: "
            + (
                "latch mentions terminated"
                if mentions_terminated
                else "terminated is constant zeros_like"
            )
        )


def test_hold_families_export_the_graded_quantity() -> None:
    """The three-tier report needs episode_max_hold_s wherever a hold exists.

    A family that tracks `_max_hold_steps` but never latches it into
    `episode_max_hold_s` can only ever report a bare success rate: the
    never-arrived / arrived-but-short / met split is not recoverable after the
    fact, because the counter is zeroed on reset.
    """
    for name, path in FAMILIES.items():
        source = path.read_text(encoding="utf-8")
        assert "_max_hold_steps" in source, f"{name}: no hold counter at all"
        assert "episode_max_hold_s" in source, (
            f"{name}: tracks _max_hold_steps but never exports "
            f"episode_max_hold_s, so its certificates cannot support the "
            f"three-tier report (never-arrived / arrived-but-short / met)."
        )
        print(f"PASS {name}: exports episode_max_hold_s")


def test_latch_is_read_before_the_counter_is_cleared() -> None:
    """Within _reset_idx, the export must precede the counter's zeroing.

    Reading `_max_hold_steps` after it is cleared reports the NEXT episode's
    value -- zero -- for every completed episode, which would look like a
    plausible "nobody ever arrived" column rather than an obvious failure.
    """
    for name, path in FAMILIES.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        reset = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx":
                reset = node
                break
        assert reset is not None, f"{name}: no _reset_idx"

        export_line = None
        clear_line = None
        for node in ast.walk(reset):
            if isinstance(node, ast.Attribute) and node.attr == "episode_max_hold_s":
                if export_line is None or node.lineno < export_line:
                    export_line = node.lineno
            # The clear is written as a SUBSCRIPT ASSIGNMENT in every family
            # today (`self._max_hold_steps[env_ids] = 0`), not as a .zero_()
            # call. An earlier draft of this test only looked for the method
            # form and therefore passed vacuously on all three files -- the
            # assertion existed but could never fire. Both forms are matched
            # here so the test fails loudly if either one moves.
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    inner = target.value if isinstance(target, ast.Subscript) else target
                    if (
                        isinstance(inner, ast.Attribute)
                        and inner.attr == "_max_hold_steps"
                    ):
                        if clear_line is None or node.lineno < clear_line:
                            clear_line = node.lineno
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {
                    "zero_",
                    "fill_",
                }:
                    owner = func.value
                    inner = owner.value if isinstance(owner, ast.Subscript) else owner
                    if (
                        isinstance(inner, ast.Attribute)
                        and inner.attr == "_max_hold_steps"
                    ):
                        if clear_line is None or node.lineno < clear_line:
                            clear_line = node.lineno

        assert export_line is not None, (
            f"{name}: _reset_idx never writes episode_max_hold_s"
        )
        if clear_line is None:
            print(f"PASS {name}: export at line {export_line}, no in-reset clear")
            continue
        assert export_line < clear_line, (
            f"{name}: episode_max_hold_s is latched at line {export_line} but "
            f"_max_hold_steps is cleared at line {clear_line}. Reading the "
            f"counter after it is cleared records zero for every completed "
            f"episode."
        )
        print(
            f"PASS {name}: latch line {export_line} precedes clear line {clear_line}"
        )


def main() -> int:
    tests = [
        test_every_hold_family_has_a_sound_latch,
        test_hold_families_export_the_graded_quantity,
        test_latch_is_read_before_the_counter_is_cleared,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(
        f"\ntest_latch_hold_families: {len(tests)} tests, {failures} failure(s); "
        f"sibling coverage: {', '.join(ALREADY_COVERED)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
