"""Pure-Python acceptance tests for observation contract v2 and its bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


obs = _load_by_path(
    "test_obs_superset_contract", _REPO_ROOT / "tasks" / "_shared" / "obs_superset.py"
)
bridge = _load_by_path(
    "test_obs_bridge", _REPO_ROOT / "scripts" / "obs_bridge.py"
)


# Frozen verbatim from observation contract v1.  Do not derive this table from
# the implementation: it is the regression oracle for every v1-era id.
V1_NATIVE_LAYOUTS = {
    "Isaac-My-First-Task-Calm-Direct-v1": [0, 1, 2],
    "Isaac-My-First-Task-Calm-Boat-Direct-v1": [0, 1, 2],
    "Isaac-USV-BlueBoat-Calm-Direct-v1": [0, 1, 2],
    "Isaac-USV-StationKeep-Direct-v1": [0, 1, 2],
    "Isaac-USV-StationKeep-BlueBoat-Direct-v1": [0, 1, 2],
    "Isaac-USV-StationKeep-BlueBoat-Current-Direct-v1": [0, 1, 2],
    "Isaac-USV-StationKeep-Boat-Direct-v1": [0, 1, 2],
    "Isaac-USV-PathFollow-Direct-v1": [0, 1, 2, 3, 4, 5, 6],
    "Isaac-USV-PathFollow-BlueBoat-Direct-v1": [0, 1, 2, 3, 4, 5, 6],
    "Isaac-USV-Dock-Direct-v1": [0, 1, 2, 7, 8, 9],
    "Isaac-USV-Dock-BlueBoat-Direct-v1": [0, 1, 2, 7, 8, 9],
    "Isaac-USV-Dock-BlueBoat-Current-Direct-v1": [0, 1, 2, 7, 8, 9],
    "Isaac-USV-HazardNav-Direct-v1": [0, 1, 2] + list(range(10, 46)),
    "Isaac-USV-HazardNav-Direct-v2": [0, 1, 2, 3, 9] + list(range(10, 46)),
    "Isaac-USV-PathHazard-Direct-v1": list(range(7)) + list(range(10, 46)),
    "Isaac-USV-HarborMission-Direct-v1": (
        [0, 1, 2]
        + list(range(10, 46))
        + list(range(46, 50))
        + [7, 8, 50]
    ),
}


# Dimensions are read from the named config declaration.  Inherited configs
# cite the declaration that fixes the inherited observation_space value.
EXPECTED_V2_DIMS = {
    "Isaac-USV-HazardNav-Direct-v3": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardRing-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardNav-Direct-v4": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardNav-Direct-v5": 15,  # hazard_nav_env_cfg.py:491
    "Isaac-USV-HazardNav-Direct-v6": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardNav-Direct-v7": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardNav-Direct-v8": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardRingSealed-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardCross-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardNav-Direct-v9": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardNav-Direct-v10": 15,  # hazard_nav_env_cfg.py:688
    "Isaac-USV-HazardNav-Direct-v12": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardBasin-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardCrossDemo-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardCrossImb-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardRing2-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardFortress-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardFortress2-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardBandFort-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HazardBandFortSoft-Direct-v1": 42,  # hazard_nav_env_cfg.py:439
    "Isaac-USV-HarborStage1-Direct-v1": 49,  # harbor_mission_env_cfg.py:286
    "Isaac-USV-HarborStage2-Direct-v1": 49,  # harbor_mission_env_cfg.py:298
    "Isaac-USV-HarborStage3-Direct-v1": 49,  # harbor_mission_env_cfg.py:309
    "Isaac-USV-HarborStage2Warm-Direct-v1": 49,  # harbor_mission_env_cfg.py:298
    "Isaac-USV-HarborStage2AllRew-Direct-v1": 49,  # harbor_mission_env_cfg.py:298
    "Isaac-USV-PathHazard-Direct-v2": 46,  # path_hazard_env_cfg.py:299
    "Isaac-USV-Dock-BlueBoat-Kin-Direct-v1": 9,  # docking_env_cfg.py:299
    "Isaac-USV-Dock-BlueBoat-Current-Kin-Direct-v1": 9,  # docking_env_cfg.py:307
    "Isaac-USV-DockWall-BlueBoat-Direct-v1": 42,  # docking_env_cfg.py:289
    "Isaac-USV-DockWall-BlueBoat-Kin-Direct-v1": 45,  # docking_env_cfg.py:315
    "Isaac-USV-StationKeep-BlueBoat-Kin-Direct-v1": 6,  # station_keeping_env_cfg.py:250
    "Isaac-USV-StationKeep-BlueBoat-Current-Kin-Direct-v1": 6,  # station_keeping_env_cfg.py:258
    "Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1": 9,  # station_keeping_env_cfg.py:273
}


V1_SLICES = {
    "nav": slice(0, 3),
    "stage_norm": slice(3, 4),
    "lookahead": slice(4, 7),
    "dock": slice(7, 10),
    "rays": slice(10, 46),
    "phase": slice(46, 50),
    "dwell_fraction": slice(50, 51),
}


def _assert_raises(expected_type, text: str, function, *args) -> None:
    try:
        function(*args)
    except expected_type as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {expected_type.__name__} containing {text!r}")


def test_v1_regression() -> None:
    assert obs.SUPERSET_DIM == 51
    for name, expected in V1_SLICES.items():
        assert obs.SLICES[name] == expected, (name, obs.SLICES[name])

    source = np.arange(51)
    for gym_id, expected_indices in V1_NATIVE_LAYOUTS.items():
        assert obs.NATIVE_LAYOUTS[gym_id] == expected_indices, gym_id
        actual = obs.extract_native(source, gym_id)
        assert np.array_equal(actual, source[expected_indices]), gym_id


def test_v2_dimensions_and_roundtrips() -> tuple[int, int]:
    assert obs.SUPERSET_DIM_V2 == 64
    assert obs.SLICES["kinematics"] == slice(51, 54)
    assert obs.SLICES["task_specific"] == slice(54, 58)
    assert obs.SLICES["reserved"] == slice(58, 64)
    assert set(obs.NATIVE_LAYOUTS) == set(V1_NATIVE_LAYOUTS) | set(EXPECTED_V2_DIMS)

    unsupported = 0
    roundtripped = 0
    for gym_id, layout in obs.NATIVE_LAYOUTS.items():
        expected_dim = (
            len(V1_NATIVE_LAYOUTS[gym_id])
            if gym_id in V1_NATIVE_LAYOUTS
            else EXPECTED_V2_DIMS[gym_id]
        )
        if isinstance(layout, obs.UnsupportedNativeLayout):
            unsupported += 1
            assert layout.native_dim == expected_dim, gym_id
            _assert_raises(
                ValueError,
                "feasibility-pooled sectors",
                obs.extract_native,
                np.zeros(64),
                gym_id,
            )
            _assert_raises(
                ValueError,
                "feasibility-pooled sectors",
                obs.native_to_superset,
                np.zeros(expected_dim),
                gym_id,
            )
            continue

        assert len(layout) == expected_dim, (gym_id, len(layout), expected_dim)
        assert len(set(layout)) == len(layout), gym_id
        assert all(isinstance(index, int) for index in layout), gym_id
        assert all(0 <= index < 64 for index in layout), gym_id
        assert all(index < 58 for index in layout), f"reserved slot claimed by {gym_id}"

        native = np.arange(2 * expected_dim, dtype=np.float32).reshape(2, expected_dim)
        native += 0.25
        superset = obs.native_to_superset(native, gym_id)
        assert superset.shape == (2, 64), gym_id
        assert np.count_nonzero(superset[..., 58:64]) == 0, gym_id
        restored = obs.extract_native(superset, gym_id)
        assert np.array_equal(restored, native), gym_id
        roundtripped += 1
    return roundtripped, unsupported


def test_hazard_to_harbor_project() -> None:
    source = np.arange(1, 43, dtype=np.float32)
    projected = bridge.project(
        source,
        "Isaac-USV-HazardNav-Direct-v3",
        "Isaac-USV-HarborStage2-Direct-v1",
    )
    assert projected.shape == (49,)
    # First, centre, and last ray: superset 10/27/45 are native 6/23/41 in
    # both v3 hazard and kinematic harbor orders.
    assert projected[6] == source[6]
    assert projected[23] == source[23]
    assert projected[41] == source[41]
    # Harbor-only phase, dock-alignment, and dwell channels are absent in A.
    assert np.array_equal(projected[42:], np.zeros(7, dtype=np.float32))


def main() -> None:
    test_v1_regression()
    print(f"PASS v1 regression: {len(V1_NATIVE_LAYOUTS)} frozen layouts unchanged")
    roundtripped, unsupported = test_v2_dimensions_and_roundtrips()
    print(
        f"PASS v2 dimensions/round-trips: {roundtripped} supported ids; "
        f"{unsupported} pooled ids explicitly unsupported"
    )
    test_hazard_to_harbor_project()
    print("PASS project: HazardNav-v3 -> HarborStage2 is 49-D with aligned rays")


if __name__ == "__main__":
    main()
