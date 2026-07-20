"""Vehicle parameter registry shared by USVBench tasks.

The registry is intentionally plain Python data so vehicle definitions can be
validated without importing Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BowBodyAxis = Literal["+x", "-x", "+y", "-y"]


@dataclass(frozen=True)
class VehicleSpec:
    """USD asset and hull-specific actuator/hydrodynamic parameters."""

    name: str
    usd_relpath: str
    mass_kg: float | None
    thrust_fwd_n: float
    thrust_rev_n: float
    yaw_torque_nm: float
    surge_lin: float
    surge_quad: float
    sway_lin: float | None
    sway_quad: float | None
    heave_damping: float
    yaw_lin: float
    yaw_quad: float
    restoring_stiffness_roll: float
    restoring_stiffness_pitch: float
    rollpitch_rate_damping: float
    displaced_volume_m3: float
    hull_height_m: float
    buoyancy_center_offset_m: float
    bow_body_axis: BowBodyAxis
    notes: str

    @property
    def mass_kg_or_None(self) -> float | None:
        """Explicit alias documenting that ``None`` preserves USD-authored mass."""
        return self.mass_kg


VEHICLES: dict[str, VehicleSpec] = {
    "rov": VehicleSpec(
        name="rov",
        usd_relpath="ROV_rigged.usd",
        mass_kg=100.0,
        thrust_fwd_n=400.0,
        thrust_rev_n=400.0,
        yaw_torque_nm=200.0,
        surge_lin=30.0,
        surge_quad=150.0,
        sway_lin=None,
        sway_quad=None,
        heave_damping=400.0,
        yaw_lin=20.0,
        yaw_quad=180.0,
        restoring_stiffness_roll=5000.0,
        restoring_stiffness_pitch=5000.0,
        rollpitch_rate_damping=2000.0,
        displaced_volume_m3=0.5,
        hull_height_m=0.6,
        buoyancy_center_offset_m=-0.1,
        bow_body_axis="+y",
        notes="BlueROV2 hydrodynamics: von Benzon et al. 2022.",
    ),
    "wamv": VehicleSpec(
        name="wamv",
        usd_relpath="boat_physics.usdc",
        mass_kg=100.0,
        thrust_fwd_n=500.0,
        thrust_rev_n=200.0,
        yaw_torque_nm=400.0,
        surge_lin=50.0,
        surge_quad=100.0,
        sway_lin=50.0,
        sway_quad=65.0,
        heave_damping=300.0,
        yaw_lin=300.0,
        yaw_quad=250.0,
        restoring_stiffness_roll=5000.0,
        restoring_stiffness_pitch=5000.0,
        rollpitch_rate_damping=2000.0,
        displaced_volume_m3=0.2,
        hull_height_m=1.0,
        buoyancy_center_offset_m=0.0,
        bow_body_axis="-x",
        notes="WAM-V VRX coefficients, Froude-scaled with lambda=0.8.",
    ),
    "blueboat": VehicleSpec(
        name="blueboat",
        usd_relpath="blueboat_physics.usd",
        mass_kg=None,
        thrust_fwd_n=80.0,
        thrust_rev_n=48.0,
        yaw_torque_nm=23.0,
        surge_lin=9.1,
        surge_quad=5.9,
        sway_lin=27.0,
        sway_quad=18.0,
        heave_damping=200.0,
        yaw_lin=4.0,
        yaw_quad=6.0,
        restoring_stiffness_roll=280.0,
        restoring_stiffness_pitch=141.0,
        rollpitch_rate_damping=470.0,
        displaced_volume_m3=0.0346,
        hull_height_m=0.376,
        buoyancy_center_offset_m=-0.05,
        bow_body_axis="+x",
        notes="BlueBoat CAD and Blue Robotics datasheet.",
    ),
}


def get_vehicle(name: str) -> VehicleSpec:
    """Return a vehicle spec, listing valid registry names on lookup failure."""
    try:
        return VEHICLES[name]
    except KeyError:
        valid_names = ", ".join(sorted(VEHICLES))
        raise ValueError(
            f"Unknown vehicle {name!r}. Valid vehicle names: {valid_names}."
        ) from None
