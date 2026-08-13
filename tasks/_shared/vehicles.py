"""Vehicle parameter registry shared by USVBench tasks.

The registry is intentionally plain Python data so vehicle definitions can be
validated without importing Isaac Lab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


BowBodyAxis = Literal["+x", "-x", "+y", "-y"]
AssetKind = Literal["articulation", "rigid_object"]


@dataclass(frozen=True)
class VehicleSpec:
    """USD asset and hull-specific actuator/hydrodynamic parameters."""

    name: str
    asset_kind: AssetKind
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
    # Plan-form geometry, used to sample the water surface at several points
    # across the hull rather than once at the origin. Different in kind from
    # displaced_volume/hull_height above: those are hydrostatic equivalents
    # fitted to the buoyancy model, these are measured lengths.
    hull_length_m: float = 0.0
    hull_spacing_m: float = 0.0

    @property
    def mass_kg_or_None(self) -> float | None:
        """Explicit alias documenting that ``None`` preserves USD-authored mass."""
        return self.mass_kg

    @property
    def waterplane_area_m2(self) -> float:
        """Heave stiffness over rho*g, as implied by the buoyancy model.

        ``submerged_ratio`` spans 0 to 1 across one hull height, so the model's
        heave stiffness is rho*g*V/h, making this V/h. It is an equivalent
        area, not an integrated waterplane; for a real hull the two differ.
        """
        return self.displaced_volume_m3 / self.hull_height_m

    def surface_sample_offsets(
        self, water_density: float = 1000.0, gravity: float = 9.81
    ) -> list[tuple[float, float]]:
        """Body-frame (surge, sway) offsets at which to sample the surface.

        Two sides by three stations. Spreading the samples over the hull is
        what makes a head sea produce pitch and a beam sea produce roll without
        either being an authored gain, and it averages out waves shorter than
        the vessel instead of letting them drive the whole rigid body.

        The offsets are *hydrodynamically equivalent* radii, not the hull's
        plan-form coordinates: they are solved so the six stations reproduce
        this vehicle's calibrated ``restoring_stiffness_roll/pitch``. Placing
        them at the geometric hull centrelines instead would lump each hull's
        buoyancy onto its centreline and drop the waterplane inertia the hull's
        own width contributes, which for BlueBoat loses well over half the roll
        stiffness. Same character as ``displaced_volume_m3`` and
        ``hull_height_m``, which are also fitted equivalents rather than
        integrals of the CAD.

        Returns a single centre sample when either the stiffnesses or the
        plan-form are missing, which reproduces point sampling -- and with it
        the fact that a single point yields no attitude response at all.
        """
        if self.hull_length_m <= 0.0 or self.hull_spacing_m <= 0.0:
            return [(0.0, 0.0)]
        if self.restoring_stiffness_roll <= 0.0 or self.restoring_stiffness_pitch <= 0.0:
            return [(0.0, 0.0)]

        # Each of the n stations carries V/n of the displaced volume, so its
        # heave stiffness is rho*g*(V/n)/h. Summed against the offsets:
        #   K_roll  = rho*g*V/h * r_roll^2          (all n stations offset)
        #   K_pitch = rho*g*V/h * (n_end/n) * r_pitch^2  (end stations only)
        heave_stiffness = (
            water_density * gravity * self.displaced_volume_m3 / self.hull_height_m
        )
        stations, end_stations = 6, 4
        roll_radius = math.sqrt(self.restoring_stiffness_roll / heave_stiffness)
        pitch_radius = math.sqrt(
            self.restoring_stiffness_pitch
            * stations
            / (end_stations * heave_stiffness)
        )
        return [
            (station * pitch_radius, side * roll_radius)
            for station in (-1.0, 0.0, 1.0)
            for side in (-1.0, 1.0)
        ]


VEHICLES: dict[str, VehicleSpec] = {
    "rov": VehicleSpec(
        name="rov",
        asset_kind="articulation",
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
        asset_kind="rigid_object",
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
        asset_kind="rigid_object",
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
        # 0.7214 m is the CAD hull/thruster spacing, cross-checked by the
        # 0.3607 m lever arm behind the 23 N*m yaw limit. Length is the product
        # designation (BB120, and the shipped BB120_official_visual_only.usd),
        # not a CAD integration -- refine it if the drawing becomes available.
        hull_length_m=1.20,
        hull_spacing_m=0.7214,
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
