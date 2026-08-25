"""Every registered gym id must have a native layout in the frozen contract.

The registry drifted silently twice: the id count was misquoted externally
(47 vs the true 52), and a freshly registered task
(Isaac-USV-HarborDockPhase-Direct-v1) crashed the composite evaluator with a
KeyError because nothing forced its author to add a NATIVE_LAYOUTS entry.
This test closes that hole at the source level -- no isaaclab import, so it
runs on any machine.

Run: python tasks/_shared/test_registry_drift.py
"""
from __future__ import annotations

from pathlib import Path
import re

SHARED = Path(__file__).resolve().parent
TASKS = SHARED.parent


def gym_registered_ids():
    """Literal ids plus regex patterns for f-string loop registrations."""
    ids = set()
    patterns = []
    for init in TASKS.glob("*/__init__.py"):
        text = init.read_text(encoding="utf-8")
        for match in re.finditer(r'id="(Isaac-[^"]+)"', text):
            ids.add(match.group(1))
        # e.g. id=f"Isaac-USV-HarborStage{_stage}-Direct-v1"
        for match in re.finditer(r'id=f"(Isaac-[^"]+)"', text):
            patterns.append(re.compile(
                re.escape(re.sub(r"\{[^}]*\}", "\0", match.group(1))).replace("\0", ".+")))
    return ids, patterns


def contract_ids():
    text = (SHARED / "obs_superset.py").read_text(encoding="utf-8")
    start = text.index("NATIVE_LAYOUTS")
    return set(re.findall(r'"(Isaac-[^"]+)"', text[start:]))


def test_every_gym_id_has_a_contract_entry():
    literals, _ = gym_registered_ids()
    missing = literals - contract_ids()
    assert not missing, (
        "gym ids missing from NATIVE_LAYOUTS (add an entry or an explicit "
        f"unsupported marker): {sorted(missing)}"
    )


def test_no_orphan_contract_entries():
    literals, patterns = gym_registered_ids()
    orphans = {
        entry for entry in contract_ids()
        if entry not in literals
        and not any(p.fullmatch(entry) for p in patterns)
    }
    assert not orphans, (
        f"NATIVE_LAYOUTS entries with no gym registration: {sorted(orphans)}"
    )


def test_registered_id_count_is_pinned():
    # Update these numbers ON PURPOSE when adding a task; a surprise mismatch
    # means an id appeared or vanished without anyone deciding it should.
    literals, patterns = gym_registered_ids()
    # 2026-08-23: 56 -> 61 for the five Suite S structural-generalization ids
    # (SingleRow/StaggeredRows/DiagonalRow/Clusters/GapWall).
    assert (len(literals), len(patterns)) == (62, 2), (
        f"literal ids: {len(literals)}, f-string registrations: {len(patterns)}; "
        "pinned (62, 2)"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
