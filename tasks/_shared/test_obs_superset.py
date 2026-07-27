"""Pure-Python acceptance test for the frozen observation superset."""

from __future__ import annotations

import numpy as np
import torch

try:
    from .obs_superset import NATIVE_LAYOUTS, SLICES, SUPERSET_DIM, extract_native
except ImportError:  # Direct execution from the repository root.
    from obs_superset import NATIVE_LAYOUTS, SLICES, SUPERSET_DIM, extract_native


# These are deliberately hardcoded instead of importing Isaac task configs:
# this test must remain runnable without Isaac Sim or Isaac Lab.
EXPECTED_DIMS = {
    "Isaac-My-First-Task-Calm-Direct-v1": 3,
    "Isaac-USVBench-ROV-Calm-Direct-v1": 3,
    "Isaac-My-First-Task-Calm-Boat-Direct-v1": 3,
    "Isaac-USVBench-Boat-Calm-Direct-v1": 3,
    "Isaac-USV-BlueBoat-Calm-Direct-v1": 3,
    "Isaac-USV-StationKeep-Direct-v1": 3,
    "Isaac-USV-StationKeep-BlueBoat-Direct-v1": 3,
    "Isaac-USV-StationKeep-BlueBoat-Current-Direct-v1": 3,
    "Isaac-USV-StationKeep-Boat-Direct-v1": 3,
    "Isaac-USV-PathFollow-Direct-v1": 7,
    "Isaac-USV-PathFollow-BlueBoat-Direct-v1": 7,
    "Isaac-USV-Dock-Direct-v1": 6,
    "Isaac-USV-Dock-BlueBoat-Direct-v1": 6,
    "Isaac-USV-Dock-BlueBoat-Current-Direct-v1": 6,
    "Isaac-USV-HazardNav-Direct-v1": 39,
    "Isaac-USV-HazardNav-Direct-v2": 41,
    "Isaac-USV-PathHazard-Direct-v1": 43,
    "Isaac-USV-HarborMission-Direct-v1": 46,
}

EXPECTED_SLICES = {
    "nav": (0, 3),
    "stage_norm": (3, 4),
    "lookahead": (4, 7),
    "dock": (7, 10),
    "rays": (10, 46),
    "phase": (46, 50),
    "dwell_fraction": (50, 51),
}


def main() -> None:
    assert SUPERSET_DIM == 51
    assert set(SLICES) == set(EXPECTED_SLICES)
    for name, (start, stop) in EXPECTED_SLICES.items():
        assert SLICES[name] == slice(start, stop), (name, SLICES[name])

    assert set(NATIVE_LAYOUTS) == set(EXPECTED_DIMS)
    numpy_superset = np.arange(2 * SUPERSET_DIM).reshape(2, SUPERSET_DIM)
    torch_superset = torch.arange(2 * SUPERSET_DIM).reshape(2, SUPERSET_DIM)

    for gym_id, expected_dim in EXPECTED_DIMS.items():
        indices = NATIVE_LAYOUTS[gym_id]
        assert isinstance(indices, list), gym_id
        assert len(indices) == expected_dim, (gym_id, len(indices), expected_dim)
        assert len(set(indices)) == len(indices), gym_id
        assert all(isinstance(index, int) for index in indices), gym_id
        assert all(0 <= index < SUPERSET_DIM for index in indices), gym_id

        numpy_native = extract_native(numpy_superset, gym_id)
        torch_native = extract_native(torch_superset, gym_id)
        assert numpy_native.shape == (2, expected_dim), gym_id
        assert torch_native.shape == (2, expected_dim), gym_id
        assert np.array_equal(numpy_native, numpy_superset[..., indices]), gym_id
        assert torch.equal(torch_native, torch_superset[..., indices]), gym_id

    print(
        "PASS: frozen 51-D superset and native layouts "
        "(3, 3, 7, 6, 39, 41, 43, 46) are valid"
    )


if __name__ == "__main__":
    main()
