"""Pure-CPU acceptance test for the Suite S runtime wiring.

No Isaac Lab, no gym.make, no GPU: this exercises exactly the code the env
uses at reset time -- ``load_public_layouts`` (with its SHA-256 verification)
and ``rotation_index`` -- against the 50 committed frozen assets, then
AST-reads the cfg file so the numbers asserted here are the numbers the env
will actually run with (importing the cfg would pull isaaclab).

Run: python tasks/hazard_nav/test_suite_s_wiring.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile

import numpy as np

try:
    from .suite_s_layouts import (
        SUITE_S_CLASSES,
        SUITE_S_PUBLIC_INDICES,
        load_public_layouts,
        rotation_index,
    )
except ImportError:  # Direct execution: python tasks/hazard_nav/test_suite_s_wiring.py
    from suite_s_layouts import (
        SUITE_S_CLASSES,
        SUITE_S_PUBLIC_INDICES,
        load_public_layouts,
        rotation_index,
    )

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "assets" / "suite_s"
SUITE_S_LEVEL = 2
# The cfg's D0 contract (min/max_goal_distance_m) and the generator's arena.
D0_CONTRACT_M = (20.0, 40.0)
ARENA_GOAL_RANGE_M = (24.0, 36.0)


def _suite_s_cfg_class_fields() -> dict[str, dict]:
    """AST-read the Suite S cfg class bodies without importing isaaclab."""
    tree = ast.parse(
        (HERE / "hazard_nav_env_cfg.py").read_text(encoding="utf-8")
    )
    classes: dict[str, dict] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("HazardSuiteS"):
            fields = {}
            for stmt in node.body:
                target = None
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    target = stmt.target.id
                elif isinstance(stmt, ast.Assign) and isinstance(
                    stmt.targets[0], ast.Name
                ):
                    target = stmt.targets[0].id
                if target is not None and stmt.value is not None:
                    try:
                        fields[target] = ast.literal_eval(stmt.value)
                    except ValueError:
                        fields[target] = "<non-literal>"
            bases = [getattr(base, "id", "?") for base in node.bases]
            classes[node.name] = {"bases": bases, "fields": fields}
    return classes


def test_cfg_family_shape():
    classes = _suite_s_cfg_class_fields()
    base = classes["HazardSuiteSEnvCfg"]
    # The champions (crossing / v9 / v12) all trained on the HazardNavV3EnvCfg
    # 42-D observation contract; Suite S must derive from that family.
    assert base["bases"] == ["HazardNavV3EnvCfg"], base["bases"]
    assert base["fields"]["suite_s_class"] == "single_row"
    assert base["fields"]["suite_s_index_rotation"] is True
    assert base["fields"]["suite_s_level"] == SUITE_S_LEVEL
    assert base["fields"]["max_obstacles"] == 24
    expected = {
        "HazardSuiteSStaggeredRowsEnvCfg": "staggered_rows",
        "HazardSuiteSDiagonalRowEnvCfg": "diagonal_row",
        "HazardSuiteSClustersEnvCfg": "clusters",
        "HazardSuiteSGapWallEnvCfg": "gap_wall",
    }
    for name, class_name in expected.items():
        info = classes[name]
        assert info["bases"] == ["HazardSuiteSEnvCfg"], (name, info["bases"])
        assert info["fields"] == {"suite_s_class": class_name}, (
            f"{name} must override ONLY suite_s_class: {info['fields']}"
        )
    return f"5 cfg classes, cap 24, defaults verified via AST"


def test_all_fifty_assets_load_through_the_env_path():
    cap = _suite_s_cfg_class_fields()["HazardSuiteSEnvCfg"]["fields"][
        "max_obstacles"
    ]
    total = 0
    lines = []
    for class_name in SUITE_S_CLASSES:
        layouts = load_public_layouts(
            class_name, level=SUITE_S_LEVEL, asset_dir=str(ASSETS)
        )
        assert len(layouts) == len(SUITE_S_PUBLIC_INDICES), class_name
        counts = []
        for expected_index, layout in zip(SUITE_S_PUBLIC_INDICES, layouts):
            assert layout.index == expected_index
            assert layout.class_name == class_name
            n = layout.obstacle_count
            counts.append(n)
            # Obstacle counts: real, positive, and inside the cfg's buffer cap
            # (a count above the cap would silently truncate at reset).
            assert 1 <= n <= cap, (class_name, expected_index, n, cap)
            assert layout.centers.shape == (n, 2)
            assert layout.radii.shape == (n,)
            assert np.all(np.isfinite(layout.centers))
            assert np.all(layout.radii > 0.0)
            # Start/goal sanity: local frame starts at the origin with the
            # goal on +X (scatter convention), D0 inside the cfg contract.
            assert np.allclose(layout.start, 0.0)
            assert abs(float(layout.goal[1])) < 1.0e-9
            d0 = float(np.linalg.norm(layout.goal - layout.start))
            assert ARENA_GOAL_RANGE_M[0] - 1.0e-9 <= d0 <= ARENA_GOAL_RANGE_M[1] + 1.0e-9
            assert D0_CONTRACT_M[0] < d0 < D0_CONTRACT_M[1]
            # The latched route basis must be a real detour measurement.
            assert layout.geodesic_length >= d0 - 1.0e-6
            assert layout.direct_blocked
            total += 1
        lines.append(f"{class_name:15s} n={counts}")
    assert total == 50, total
    return "\n    ".join([f"{total} layouts verified (checksums included)"] + lines)


def test_checksum_verification_is_live_in_the_loader():
    source = ASSETS / f"suite_s_single_row_L{SUITE_S_LEVEL}_00.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["radii"][0] = float(payload["radii"][0]) + 0.25  # 25 cm cheat
    with tempfile.TemporaryDirectory() as tmp:
        tampered_dir = Path(tmp)
        for index in SUITE_S_PUBLIC_INDICES:
            name = f"suite_s_single_row_L{SUITE_S_LEVEL}_{index:02d}.json"
            (tampered_dir / name).write_text(
                (ASSETS / name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        (tampered_dir / source.name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        try:
            load_public_layouts(
                "single_row", level=SUITE_S_LEVEL, asset_dir=str(tampered_dir)
            )
        except ValueError as exc:
            assert "checksum mismatch" in str(exc), exc
            return "a 25 cm radius edit is refused with a checksum mismatch"
        raise AssertionError("tampered asset was accepted")


def test_rotation_covers_all_ten_indices():
    # Ten consecutive fake episodes of one env cover every public index.
    for env_index in (0, 3, 64):
        seen = [
            rotation_index(env_index, episode)
            for episode in range(len(SUITE_S_PUBLIC_INDICES))
        ]
        assert sorted(seen) == list(SUITE_S_PUBLIC_INDICES), (env_index, seen)
    # Rotation off pins each env to env % 10, forever.
    for env_index in (0, 7, 23):
        pinned = {
            rotation_index(env_index, episode, rotation=False)
            for episode in range(25)
        }
        assert pinned == {env_index % 10}, (env_index, pinned)
    # The certification shape: 64 envs x 2 episodes = 128. 64 % 10 != 0, so
    # exact uniformity is impossible; the achievable "even" is 12-14 per
    # layout (ideal 12.8), pinned here as the exact expected tally.
    tally = np.zeros(10, dtype=int)
    for env_index in range(64):
        for episode in range(2):
            tally[rotation_index(env_index, episode)] += 1
    assert tally.sum() == 128
    assert tally.tolist() == [13, 14, 14, 14, 13, 12, 12, 12, 12, 12], (
        tally.tolist()
    )
    return f"10-episode coverage exact; 128-episode tally {tally.tolist()}"


def main() -> None:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                detail = fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            else:
                print(f"PASS {name}: {detail}")
    if failures:
        raise SystemExit(1)
    print("PASS: Suite S wiring verified on CPU without Isaac")


if __name__ == "__main__":
    main()
