"""Frozen cross-task policy-observation layout for USVBench.

This module deliberately depends on neither Isaac Lab nor Gymnasium so that
checkpoint adapters and schema tests can import it in a plain Python process.
The index assignments are a compatibility contract: append new channels after
the existing layout rather than reordering these entries.
"""

from __future__ import annotations

from typing import TypeVar


SUPERSET_DIM = 51

# Frozen v1 layout.  Slices use an exclusive stop, matching NumPy/Torch.
SLICES = {
    "nav": slice(0, 3),
    "stage_norm": slice(3, 4),
    "lookahead": slice(4, 7),
    "dock": slice(7, 10),
    "rays": slice(10, 46),
    "phase": slice(46, 50),
    "dwell_fraction": slice(50, 51),
}

# Docking defines speed_norm using a 2 m/s scale.  Reusing that scale is what
# makes the superset speed channel semantically identical across tasks.
SPEED_SCALE_MPS = 2.0

_NAV = [0, 1, 2]
_PATH = [0, 1, 2, 3, 4, 5, 6]
_DOCK = [0, 1, 2, 7, 8, 9]
_HAZARD = [0, 1, 2] + list(range(10, 46))
_PATH_HAZARD = _PATH + list(range(10, 46))
_HARBOR = (
    [0, 1, 2]
    + list(range(10, 46))
    + list(range(46, 50))
    + [7, 8]
    + [50]
)

# Every currently registered Gym id is listed explicitly.  Vehicle/current
# variants share their task family's native policy-observation order.
NATIVE_LAYOUTS = {
    "Isaac-My-First-Task-Calm-Direct-v1": list(_NAV),
    "Isaac-My-First-Task-Calm-Boat-Direct-v1": list(_NAV),
    "Isaac-USV-BlueBoat-Calm-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-BlueBoat-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-BlueBoat-Current-Direct-v1": list(_NAV),
    "Isaac-USV-StationKeep-Boat-Direct-v1": list(_NAV),
    "Isaac-USV-PathFollow-Direct-v1": list(_PATH),
    "Isaac-USV-PathFollow-BlueBoat-Direct-v1": list(_PATH),
    "Isaac-USV-Dock-Direct-v1": list(_DOCK),
    "Isaac-USV-Dock-BlueBoat-Direct-v1": list(_DOCK),
    "Isaac-USV-Dock-BlueBoat-Current-Direct-v1": list(_DOCK),
    "Isaac-USV-HazardNav-Direct-v1": list(_HAZARD),
    "Isaac-USV-PathHazard-Direct-v1": list(_PATH_HAZARD),
    "Isaac-USV-HarborMission-Direct-v1": list(_HARBOR),
}


_Array = TypeVar("_Array")


def extract_native(superset_vec: _Array, gym_id: str) -> _Array:
    """Extract ``gym_id``'s native observation order from a 51-D tensor/array."""
    try:
        indices = NATIVE_LAYOUTS[gym_id]
    except KeyError as exc:
        known = ", ".join(sorted(NATIVE_LAYOUTS))
        raise KeyError(f"Unknown USVBench Gym id {gym_id!r}; known ids: {known}") from exc

    shape = getattr(superset_vec, "shape", None)
    if shape is None or len(shape) == 0 or shape[-1] != SUPERSET_DIM:
        raise ValueError(
            f"Expected a NumPy/Torch value with final dimension {SUPERSET_DIM}, "
            f"got shape {shape!r}"
        )
    return superset_vec[..., indices]


__all__ = [
    "NATIVE_LAYOUTS",
    "SLICES",
    "SPEED_SCALE_MPS",
    "SUPERSET_DIM",
    "extract_native",
]
