"""Plain-data regression guard for station keeping's pre-registry ROV literals."""

from __future__ import annotations

import struct
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks._shared.vehicles import VEHICLES, get_vehicle


ROV_SNAPSHOT = {
    "name": "rov",
    "asset_kind": "articulation",
    "usd_relpath": "ROV_rigged.usd",
    "mass_kg": 100.0,
    "thrust_fwd_n": 400.0,
    "thrust_rev_n": 400.0,
    "yaw_torque_nm": 200.0,
    "surge_lin": 30.0,
    "surge_quad": 150.0,
    "sway_lin": None,
    "sway_quad": None,
    "heave_damping": 400.0,
    "yaw_lin": 20.0,
    "yaw_quad": 180.0,
    "restoring_stiffness_roll": 5000.0,
    "restoring_stiffness_pitch": 5000.0,
    "rollpitch_rate_damping": 2000.0,
    "displaced_volume_m3": 0.5,
    "hull_height_m": 0.6,
    "buoyancy_center_offset_m": -0.1,
    "bow_body_axis": "+y",
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

    rov = get_vehicle("rov")
    mismatches = {
        field: (getattr(rov, field), expected)
        for field, expected in ROV_SNAPSHOT.items()
        if not _equal_exact(getattr(rov, field), expected)
    }
    if mismatches:
        raise AssertionError(f"ROV registry parity failed: {mismatches}")

    print("Vehicle registry keys: PASS (blueboat, rov, wamv)")
    print(
        f"ROV station-keeping parity: PASS ({len(ROV_SNAPSHOT)} fields; "
        "float bytes exact)"
    )


if __name__ == "__main__":
    main()
