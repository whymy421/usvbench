"""Vehicle parameter registry shared by USVBench tasks.

The registry is intentionally plain Python data so vehicle definitions can be
validated without importing Isaac Lab.
"""

from __future__ import annotations

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

    @property
    def mass_kg_or_None(self) -> float | None:
        """Explicit alias documenting that ``None`` preserves USD-authored mass."""
        return self.mass_kg


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
        notes="BlueBoat CAD and Blue Robotics datasheet.",
    ),
    "catamaran": VehicleSpec(
        name="catamaran",
        asset_kind="rigid_object",
        usd_relpath="catamaran.usd",
        mass_kg=120.0,
        # Sized for a fast patrol: 850 N gives a 2.50 m/s terminal surge against the
        # damping below (thrust-to-weight 0.72, above the wamv's 0.51), and 740 N*m gives
        # a 1.00 rad/s terminal yaw, i.e. a 2.5 m turning radius at cruise. Reverse keeps
        # the wamv's 0.4 forward/reverse ratio.
        thrust_fwd_n=850.0,
        thrust_rev_n=340.0,
        yaw_torque_nm=740.0,
        # Surge and yaw: VRX WAM-V coefficients Froude-scaled the way the wamv entry is,
        # lambda = (120/195)^(1/3) = 0.850 for this 120 kg hull. Both are anchored by a
        # measured terminal value (2.500 m/s, 1.000 rad/s), so they are calibrated, not
        # assumed.
        surge_lin=65.0,
        surge_quad=110.0,
        # Sway follows the blueboat convention, ~3x surge, and carries the same caveat:
        # it is a lateral-bluffness rule of thumb, not system identification. Taking the
        # Froude-scaled VRX value instead would give 70, i.e. LESS resistance sideways
        # than forwards, which no hull has; that error made the vessel skate through its
        # turns at a 17.4 deg mean drift angle. blueboat is the right reference here
        # because it is also a displacement catamaran.
        sway_lin=195.0,
        sway_quad=330.0,
        heave_damping=330.0,
        yaw_lin=385.0,
        yaw_quad=355.0,
        # Hydrostatics measured off the hull mesh, blueboat's method. The waterline that
        # displaces m/rho = 0.12 m^3 sits at a 0.353 m draft, where the waterplane is
        # 0.402 m^2 with I_T = 0.0396 and I_L = 0.3121 m^4, giving BM_T = 0.330 m,
        # BM_L = 2.600 m and KB = 0.195 m. k = rho*g*V*GM at KG = 0.30 m.
        #
        # KG is the one assumed quantity, and it matters: PhysX derives this hull's
        # inertia from a uniform-density convex decomposition and puts the COM at
        # mid-height (measured KG = 0.50 m), which would give GM_T = 0.025 m and a
        # k_roll of 29 -- a marginally stable hull. That is an artefact of uniform
        # density, not a property of a real catamaran, whose machinery sits low. KG =
        # 0.30 m is the working assumption pending an inclining test; it gives 265,
        # against blueboat's 280 for a comparable catamaran.
        restoring_stiffness_roll=265.0,
        restoring_stiffness_pitch=2934.0,
        # blueboat's scaling, c ~ sqrt(k), which holds the damping ratio: it gives
        # zeta = 3.0 against the measured 21.75 kg*m^2 roll inertia.
        rollpitch_rate_damping=460.0,
        # rov_volume/rov_height feed the sim's linear submersion proxy, not geometry:
        # 0.4 * rho * g * 0.3 balances the 1176 N weight. The hull's real displaced
        # volume is 0.12 m^3 and its real draft 0.353 m, both above.
        displaced_volume_m3=0.3,
        hull_height_m=1.0,
        buoyancy_center_offset_m=0.0,
        bow_body_axis="+x",
        notes=(
            "Surge/yaw Froude-scaled from VRX WAM-V (lambda=0.85) and confirmed against "
            "measured terminal values; sway is blueboat's ~3x-surge rule pending sysid; "
            "restoring from mesh hydrostatics at an assumed KG=0.30 m."
        ),
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
