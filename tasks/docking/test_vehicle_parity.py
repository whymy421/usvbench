"""Plain-data regression guard for docking's pre-registry WAM-V literals.

This script intentionally imports no Isaac Lab modules. Float values are
compared by their packed IEEE-754 bytes, not with a tolerance.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks._shared.vehicles import VEHICLES, get_vehicle


WAMV_SNAPSHOT = {
    "name": "wamv",
    "usd_relpath": "boat_physics.usdc",
    "mass_kg": 100.0,
    "thrust_fwd_n": 500.0,
    "thrust_rev_n": 200.0,
    "yaw_torque_nm": 400.0,
    "surge_lin": 50.0,
    "surge_quad": 100.0,
    "sway_lin": 50.0,
    "sway_quad": 65.0,
    "heave_damping": 300.0,
    "yaw_lin": 300.0,
    "yaw_quad": 250.0,
    "restoring_stiffness_roll": 5000.0,
    "restoring_stiffness_pitch": 5000.0,
    "rollpitch_rate_damping": 2000.0,
    "displaced_volume_m3": 0.2,
    "hull_height_m": 1.0,
    "buoyancy_center_offset_m": 0.0,
    "bow_body_axis": "-x",
}


def _equal_exact(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, float) and struct.pack("!d", actual) == struct.pack(
            "!d", expected
        )
    return actual == expected


def main() -> None:
    expected_names = {"rov", "wamv", "blueboat"}
    if set(VEHICLES) != expected_names:
        raise AssertionError(
            f"registry names differ: actual={sorted(VEHICLES)}, "
            f"expected={sorted(expected_names)}"
        )

    wamv = get_vehicle("wamv")
    mismatches = {
        field: (getattr(wamv, field), expected)
        for field, expected in WAMV_SNAPSHOT.items()
        if not _equal_exact(getattr(wamv, field), expected)
    }
    if mismatches:
        raise AssertionError(f"WAM-V registry parity failed: {mismatches}")

    print("Vehicle registry keys: PASS (blueboat, rov, wamv)")
    print(
        f"WAM-V docking parity: PASS ({len(WAMV_SNAPSHOT)} fields; "
        "float bytes exact)"
    )


if __name__ == "__main__":
    main()
