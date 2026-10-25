"""USVBench vessel profiles v2 (schema 2): BlueBoat and the VRX WAM-V at 1:1.

Pure Python, no simulator/torch/numpy import, so it can be validated on CPU.
Legacy entries in vehicles.py are untouched; this is a separate registry.

Conventions
-----------
* SI units. Body/actor axes FLU: +x bow, +y port, +z up. The actor origin of
  every v2 asset is the aggregate COM (USD MassAPI centerOfMass = 0), so the
  COM is the single physical reference point (C1 finding 25).
* Six-vectors are ordered (surge, sway, heave, roll, pitch, yaw). Damping
  coefficients are positive magnitudes of the opposing load
  tau_i = -(linear_i + quadratic_i*|nu_i|)*nu_i in hull axes on the
  WATER-RELATIVE velocity (the law of vrx_gz/src/SimpleHydrodynamics.cc:259-273).
* Restoring stiffnesses are TOTAL small-angle hydrostatic stiffnesses about the
  COM (they already contain the -m*g*z_G term); do not add a separate
  buoyancy-centre couple (C1 finding 4).
* Thrust limits are DELIVERED per-motor forces; reverse limits are positive
  magnitudes. Command laws map each source's native command to delivered
  thrust and are metadata for the runtime mixer, which works in newtons.

Profiles (all registered; the controller/owner chooses which ones run)
--------
blueboat_v2            BlueBoat, legacy 17.26 kg asset canonicalized.
wamv_vrx_v2_opt1       WAM-V, VRX competition-default assembly (283.385 kg;
                       vrx_gz/src/vrx_gz/model.py:239-240), geometry, mass,
                       inertia and buoyancy layout from osrf/vrx 7609d1bd;
                       damping + thrust from the gazebo_classic branch
                       c9b93883 (surge drag after Sarda et al. 2016).
                       Report option 1 (recommended there).
wamv_vrx_v2_opt2       Same assembly; damping + thrust from 7609d1bd itself
                       (report option 2).
wamv_vrx_v2_opt1_bare  Option 1 on the 258 kg platform (xacro defaults).
wamv_vrx_v2_opt2_bare  Option 2 on the 258 kg platform.

Owner decisions pending (see FINAL_REPORT.md): option, mass configuration,
BlueBoat roll/pitch damping, nominal motor lag, yaw request scale, pitch
coupling of the thrust line, wind profile.

Planar thrust arm. Both hulls' motors are mirror-symmetric about the HULL
CENTRELINE. Hull resistance acts on the centreline, while the v2 hydro kernel
applies damping at the COM. With an off-centre COM (BlueBoat: 0.047 m to
starboard) a literal r x F about the COM would add an unbalanced yaw moment
(about 3.8 N m at full thrust, 17 percent of yaw authority) that the real hull
cancels with its offset resistance. The mixer therefore uses the centreline
arm; the true 3-D positions are kept in Motor.position_com_m.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from types import MappingProxyType
from typing import Mapping

try:  # package import inside tasks/_shared
    from . import vehicles_v2_data as _data
except ImportError:  # path-loaded CPU tests; never import the tasks package
    import vehicles_v2_data as _data


AXES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
SCHEMA_VERSION = 2
RHO_WATER = 1000.0   # fresh water; vrx_gz/src/Surface.cc:63 default, blueboat_hydrostatics.py:18
GRAVITY = 9.81       # benchmark gravity (D3 hydro_v2.HydroSpec default); VRX worlds use 9.8
RHO_AIR = 1.225      # ISA sea level; C3 wind.py WindCfg default
KGF = 9.80665        # standard gravity, kgf -> N conversion only

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Six = tuple[float, float, float, float, float, float]
Matrix3 = tuple[Vec3, Vec3, Vec3]


# ============================================================== schema
@dataclass(frozen=True)
class Provenance:
    parameter: str
    citation: str
    status: str


@dataclass(frozen=True)
class Damping:
    linear: Six
    quadratic: Six
    source_set: str


@dataclass(frozen=True)
class Hydrostatics:
    water_density_kg_m3: float
    gravity_m_s2: float
    displaced_volume_m3: float          # = mass / rho at calm equilibrium
    maximum_volume_m3: float            # full-immersion (saturation) model volume
    waterplane_area_m2: float           # dV/d(immersion) at equilibrium
    draft_m: float                      # reference hull immersion (see model)
    freeboard_m: float
    com_height_above_water_m: float     # COM above the calm mean waterline
    stiffness_heave_n_m: float          # = rho*g*waterplane
    stiffness_roll_nm_rad: float        # total, about the COM
    stiffness_pitch_nm_rad: float       # total, about the COM
    heave_pitch_coupling_n_rad: float   # off-diagonal metadata, not in D3's diagonal model
    gm_roll_m: float                    # = K_roll / (m g)
    gm_pitch_m: float                   # = K_pitch / (m g)
    orbital_depth_m: float              # depth at which wave orbital velocity is sampled
    wave_footprint_length_m: float
    wave_footprint_beam_m: float
    static_roll_rad: float              # source-model equilibrium attitude (metadata, not applied)
    static_pitch_rad: float
    model: str


@dataclass(frozen=True)
class AddedMass:
    diagonal: Six
    active_in_v2: bool
    policy: str


@dataclass(frozen=True)
class Motor:
    name: str
    position_com_m: Vec3                # thrust application point, actor frame (origin = COM)
    direction_body: Vec3
    forward_limit_n: float              # delivered, per motor
    reverse_limit_n: float              # delivered magnitude, per motor
    command_unit: str
    command_to_thrust: tuple[tuple[float, float], ...]   # native command -> delivered N
    time_constant_s: float
    law_note: str


@dataclass(frozen=True)
class Propulsion:
    motors: tuple[Motor, Motor]         # (port, starboard)
    planar_arm_m: float                 # |y| of each motor from the hull centreline
    yaw_command_limit_nm: float         # policy yaw request scale = arm*(fwd+rev)
    pure_yaw_limit_nm: float            # zero-surge yaw = 2*arm*min(fwd,rev)
    thrust_line_height_com_m: float     # geometric z of thrust points relative to COM
    mixer_mount_z_m: float              # pitch lever passed to the D3 mixer (default 0)
    note: str


@dataclass(frozen=True)
class Windage:
    frontal_area_m2: float
    lateral_area_m2: float
    length_m: float                     # yaw reference length (LOA)
    lateral_centroid_height_above_water_m: float
    frontal_centroid_height_above_water_m: float
    roll_lever_com_m: float             # lateral centroid z minus COM z
    lateral_centroid_x_com_m: float     # geometric metadata; NOT an extra yaw lever
    c_x0: float
    c_y0: float
    c_n0: float
    air_density_kg_m3: float
    u10_profile: str                    # "neutral_log"
    roughness_length_m: float
    status: str


@dataclass(frozen=True)
class CollisionFootprint:
    polygon_com_xy_m: tuple[Vec2, ...]  # counter-clockwise, about the COM
    capsule_center_com_xy_m: Vec2
    capsule_segment_half_length_m: float
    capsule_radius_m: float
    note: str


@dataclass(frozen=True)
class SpeedReference:
    cruise_m_s: float
    reference_m_s: float
    top_model_m_s: float                # terminal speed of this profile's own laws
    top_published_m_s: float | None
    note: str


@dataclass(frozen=True)
class VesselSpecV2:
    name: str
    schema_version: int
    option: str
    configuration: str
    asset_relpath: str                  # file name under the USVBench assets directory
    asset_sha256: str
    asset_kind: str
    scale: float
    bow_body_axis: str
    mass_kg: float
    com_actor_m: Vec3
    inertia_com_kg_m2: Matrix3
    length_overall_m: float
    beam_overall_m: float
    height_overall_m: float
    damping: Damping
    hydrostatics: Hydrostatics
    added_mass: AddedMass
    propulsion: Propulsion
    windage: Windage
    collision: CollisionFootprint
    speeds: SpeedReference
    provenance: tuple[Provenance, ...]

    @property
    def motors(self) -> tuple[Motor, Motor]:
        return self.propulsion.motors


# ============================================================== source parameters
# ---- VRX, pinned jazzy head 7609d1bd (local checkout D:/claude code/scratch/vrx)
VRX_J = "osrf/vrx@7609d1bd90ce7edb29d040a082f949e8b089c864"
VRX_C = "osrf/vrx@c9b9388308f8976c724da4af6685f69c9c378983 (gazebo_classic)"
assert _data.VRX_COMMIT == "7609d1bd90ce7edb29d040a082f949e8b089c864"
# engines: wamv_aft_thrusters.xacro:4-5; propeller joint offset engine.xacro:59
WAMV_ENGINE_BASE_M = (-2.373776, 1.027135, 0.318237)
WAMV_PROP_OFFSET_M = (-0.278156, 0.0, -0.509371)
WAMV_PONTOON_Y_M = 1.03             # wamv_gazebo_dynamics_plugin.xacro:15-16,32-33
WAMV_HULL_LENGTH_M = 4.9            # wamv_gazebo_dynamics_plugin.xacro:11
WAMV_HULL_RADIUS_M = 0.213          # wamv_gazebo_dynamics_plugin.xacro:12
WAMV_PONTOON_TOP_BASE_Z_M = 0.4     # wamv_base.urdf.xacro:9-14 main float r=0.2 at z=0.2

# J hydrodynamics: wamv_gazebo_dynamics_plugin.xacro:45-59; zWW omitted -> 0
# (SimpleHydrodynamics.cc:146).
WAMV_DAMPING_J = Damping(linear=(100.0, 100.0, 500.0, 300.0, 900.0, 800.0),
                         quadratic=(150.0, 100.0, 0.0, 600.0, 900.0, 800.0),
                         source_set=VRX_J + " wamv_gazebo_dynamics_plugin.xacro:49-59 (after PR #706)")
# Classic hydrodynamics: gazebo_classic wamv_gazebo_dynamics_plugin.xacro:16-24;
# kPP, mQQ, zWW omitted -> 0 (usv_gazebo_dynamics_plugin.cc:103,105,107).
WAMV_DAMPING_C = Damping(linear=(51.3, 40.0, 500.0, 50.0, 50.0, 400.0),
                         quadratic=(72.4, 0.0, 0.0, 0.0, 0.0, 0.0),
                         source_set=VRX_C + " wamv_gazebo_dynamics_plugin.xacro:16-24")
# J thrust command limit: wamv_gazebo_thruster_config.xacro:6-8,19, per motor.
WAMV_J_FORWARD_N = ((51.3 + 72.4 * 7.71667) * 7.71667) / 2.0
# J reverse: min_thrust_cmd omitted -> gz-sim8 Thruster.cc:118-119 default -1000.
WAMV_J_REVERSE_N = 1000.0
# Classic GLF (usv_gazebo_thrust_plugin.cc:318-341, mappingType 1, maxCmd 1,
# maxForceFwd 250, maxForceRev -100: wamv_gazebo_thruster_config.xacro:14-17).
_GLF_FWD = (0.01, 59.82, 5.0, 0.38, 0.56, 0.28)
_GLF_REV = (-199.13, -0.09, 8.84, 5.34, 0.99, -0.57)
WAMV_C_CLIP_FWD_N = 250.0
WAMV_C_CLIP_REV_N = 100.0

# ---- BlueBoat (manufacturer and legacy calibration)
BB_HALF_SPACING_M = 0.3607          # blueboat_calm_nav_env_cfg.py:139-143; blueboat_mass_budget.py:50
BB_STATIC_THRUST_KGF = 8.2          # bluerobotics.com/store/boat/blueboat/blueboat/ (Maximum Static Thrust)
BB_M200_16V_KGF = (5.63, 2.81)      # bluerobotics.com/learn/technical-reference/ (M200 w/ weedless, fwd/rev @16V)
BB_MOTOR_HULL_M = (-0.52, 0.06)     # blueboat_mass_budget.py:70-72 (x aft of hull midpoint, z above keel), estimated
# Legacy calibration: blueboat_calm_nav_env_cfg.py:73-97; vehicles.py:107-116.
BB_DAMPING = Damping(linear=(9.1, 27.0, 200.0, 470.0, 470.0, 4.0),
                     quadratic=(5.9, 18.0, 0.0, 0.0, 0.0, 6.0),
                     source_set="USVBench legacy calibration, blueboat_calm_nav_env_cfg.py:73-97")


# ============================================================== laws
def _f32(x: float) -> float:
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def _glf(x: float, p) -> float:
    a, k, b, v, c, m = (_f32(t) for t in p)
    return a + (k - a) / math.pow(c + math.exp(-b * (x - m)), 1.0 / v)


def vrx_classic_thrust(command: float) -> float:
    """Delivered per-motor thrust of the classic VRX plugin for cmd in [-1, 1].

    usv_gazebo_thrust_plugin.cc:325-341: cmd > 0.01 forward GLF clipped at 250,
    cmd < 0.01 reverse GLF clipped at -100, cmd == 0.01 gives 0. The plugin's
    own value at cmd = 0 is +0.041 N (reverse curve); the v2 knot table below
    pins command 0 to exactly 0 N.
    """
    if command > 0.01:
        return min(_glf(command, _GLF_FWD), WAMV_C_CLIP_FWD_N)
    if command < 0.01:
        return max(_glf(command, _GLF_REV), -WAMV_C_CLIP_REV_N)
    return 0.0


def _classic_knots() -> tuple[tuple[float, float], ...]:
    commands = [round(-1.0 + 0.025 * i, 6) for i in range(81)]
    knots = []
    for c in commands:
        t = 0.0 if c == 0.0 else vrx_classic_thrust(c)
        if 0.0 < c < 0.0125:
            continue
        knots.append((c, t))
    return tuple(knots)


def terminal_speed(linear: float, quadratic: float, force: float) -> float:
    if force <= 0:
        return 0.0
    if quadratic == 0:
        return force / linear
    return (-linear + math.sqrt(linear * linear + 4 * quadratic * force)) / (2 * quadratic)


# ============================================================== profile builders
def _provenance(**items: tuple[str, str]) -> tuple[Provenance, ...]:
    return tuple(Provenance(k, c, s) for k, (c, s) in items.items())


def _blueboat() -> VesselSpecV2:
    d = _data.BLUEBOAT
    mass = d["mass_kg"]
    com_h = d["com_hull_frame_m"]                  # COM in the hull frame (origin hull midpoint at waterline)
    fwd = BB_STATIC_THRUST_KGF * KGF / 2.0         # equal split of whole-vessel static thrust
    rev = fwd * BB_M200_16V_KGF[1] / BB_M200_16V_KGF[0]
    z_motor_hull = BB_MOTOR_HULL_M[1] - d["draft_m"]
    motors = tuple(
        Motor(name=side,
              position_com_m=(BB_MOTOR_HULL_M[0] - com_h[0], sign * BB_HALF_SPACING_M - com_h[1],
                              z_motor_hull - com_h[2]),
              direction_body=(1.0, 0.0, 0.0), forward_limit_n=fwd, reverse_limit_n=rev,
              command_unit="normalized delivered-thrust demand u in [-1, 1]",
              command_to_thrust=((-1.0, -rev), (0.0, 0.0), (1.0, fwd)),
              time_constant_s=0.0,
              law_note=("Engineering law F=fwd*u (u>=0), F=rev*u (u<0); not a measured PWM curve "
                        "(PWM 1100-1900 us, 1500+/-25 us stop). Nominal lag 0 s is a declared "
                        "UNVERIFIED default; actuator-lag doses add on top."))
        for side, sign in (("port", 1.0), ("starboard", -1.0)))
    k_heave = RHO_WATER * GRAVITY * d["waterplane_m2"]
    k_roll = mass * GRAVITY * d["gm_roll_m"]
    k_pitch = mass * GRAVITY * d["gm_pitch_m"]
    # Heel from the lateral COM offset and trim from COM x vs buoyancy centre x (metadata).
    static_roll = mass * GRAVITY * (-com_h[1]) / k_roll
    static_pitch = mass * GRAVITY * (com_h[0] - d["buoyancy_centre_x_hull_m"]) / k_pitch
    hydro = Hydrostatics(
        water_density_kg_m3=RHO_WATER, gravity_m_s2=GRAVITY, displaced_volume_m3=mass / RHO_WATER,
        maximum_volume_m3=d["maximum_volume_m3"], waterplane_area_m2=d["waterplane_m2"],
        draft_m=d["draft_m"], freeboard_m=d["freeboard_m"], com_height_above_water_m=com_h[2],
        stiffness_heave_n_m=k_heave, stiffness_roll_nm_rad=k_roll, stiffness_pitch_nm_rad=k_pitch,
        heave_pitch_coupling_n_rad=0.0, gm_roll_m=d["gm_roll_m"], gm_pitch_m=d["gm_pitch_m"],
        orbital_depth_m=d["draft_m"] / 2.0,
        wave_footprint_length_m=d["waterline_length_m"], wave_footprint_beam_m=d["waterline_footprint_beam_m"],
        static_roll_rad=static_roll, static_pitch_rad=static_pitch,
        model=("CAD convex horizontal sections of hull STL PUB-BR-101381-001 mirrored at +/-0.3607 m, "
               "flat water, upright tangent values at 17.26 kg; heave-pitch coupling not computed (set 0)"))
    lo, hi = d["bounds_com_m"]
    arm = BB_HALF_SPACING_M
    wind = Windage(
        frontal_area_m2=d["frontal_area_m2"], lateral_area_m2=d["lateral_area_m2"], length_m=1.20,
        lateral_centroid_height_above_water_m=d["lateral_centroid_height_above_water_m"],
        frontal_centroid_height_above_water_m=d["frontal_centroid_height_above_water_m"],
        roll_lever_com_m=d["lateral_centroid_height_above_water_m"] - com_h[2],
        lateral_centroid_x_com_m=d["lateral_centroid_x_from_com_m"],
        c_x0=0.9, c_y0=0.9, c_n0=0.1, air_density_kg_m3=RHO_AIR, u10_profile="neutral_log",
        roughness_length_m=2.0e-4,
        status=("hull silhouettes from CAD (C3 method) + UNVERIFIED superstructure band estimate; "
                "coefficients 0.9/0.9/0.1 are C3 engineering assumptions"))
    rect = tuple(tuple(p) for p in d["footprint_rectangle_com_xy_m"])
    cap_c, cap_a, cap_r = d["capsule"]
    top_model = terminal_speed(BB_DAMPING.linear[0], BB_DAMPING.quadratic[0], 2 * fwd)
    return VesselSpecV2(
        name="blueboat_v2", schema_version=SCHEMA_VERSION, option="blueboat",
        configuration=("Legacy authored 17.26 kg model (two historical 1.4 kg batteries, no payload); "
                       "hull-only USD canonicalized to FLU and COM origin"),
        asset_relpath=d["asset"], asset_sha256=d["asset_sha256"], asset_kind="rigid_object", scale=1.0,
        bow_body_axis="+x", mass_kg=mass, com_actor_m=(0.0, 0.0, 0.0),
        inertia_com_kg_m2=d["inertia_com_kg_m2"],
        length_overall_m=1.20, beam_overall_m=0.93, height_overall_m=0.46,
        damping=BB_DAMPING, hydrostatics=hydro,
        added_mass=AddedMass((0.0,) * 6, False,
                             "No BlueBoat added-mass identification exists; disabled (not a physical zero)."),
        propulsion=Propulsion(
            motors=motors, planar_arm_m=arm, yaw_command_limit_nm=arm * (fwd + rev),
            pure_yaw_limit_nm=2 * arm * min(fwd, rev),
            thrust_line_height_com_m=z_motor_hull - com_h[2], mixer_mount_z_m=0.0,
            note=("forward = 8.2 kgf whole-vessel static thrust split equally; reverse derated by the "
                  "M200-weedless 16 V ratio 2.81/5.63 (UNVERIFIED for the assembled vessel); motor "
                  "points estimated (blueboat_mass_budget.py:70-72)")),
        windage=wind,
        collision=CollisionFootprint(rect, tuple(cap_c), cap_a, cap_r,
                                     "public 1.20 x 0.93 m deployed envelope centred on the hull, expanded to "
                                     "contain every authored vertex; capsule = minimum-area capsule containing it"),
        speeds=SpeedReference(cruise_m_s=1.0, reference_m_s=1.0, top_model_m_s=top_model, top_published_m_s=3.0,
                              note=("cruise = manufacturer endurance point (1 m/s at about 30 W); top 3 m/s is the "
                                    "published two-battery no-payload maximum, also used to fit surge drag, so the "
                                    "agreement of top_model with it is circular")),
        provenance=_provenance(
            mass=("source USD /World/BlueBoat MassAPI (blueboat_build_usd.py:35,136-140); C1 AUDIT_physics.md:17",
                  "authored value; physical mass of the operated configuration UNVERIFIED"),
            com=("blueboat_build_usd.py:36 authored (0.006,0.047,0.0068) in source axes (bow -X, starboard +Y, "
                 "blueboat_build_usd.py:3-5,53); canonical axes by 180 deg yaw; z re-referenced to the 17.26 kg "
                 "waterline", "authored from blueboat_mass_budget.py estimate; UNVERIFIED"),
            inertia=("blueboat_build_usd.py:37 authored diagonal (1.789,1.094,2.376)",
                     "authored diagonal; the budget tensor has products up to 0.082 kg m2 (UNVERIFIED either way)"),
            dimensions=("bluerobotics.com/store/boat/blueboat/blueboat/ 120 x 93 x 46 cm deployed",
                        "manufacturer"),
            damping=("blueboat_calm_nav_env_cfg.py:73-97; vehicles.py:107-116",
                     "legacy two-point surge fit + engineering estimates; roll/pitch 470 UNVERIFIED and strongly "
                     "overdamped"),
            hydrostatics=("evidence/derive_blueboat.py:23-70 on PUB-BR-101381-001.STL; assets/build_vessels_v2.py",
                          "CAD-derived; checked against published 160 mm freeboard (155 mm at 16.9 kg)"),
            added_mass=("none available", "UNVERIFIED; disabled"),
            motors=("store page 8.2 kgf; technical reference M200 weedless 5.63/2.81 kgf @16 V; "
                    "blueboat_mass_budget.py:70-72", "forward manufacturer; reverse and positions UNVERIFIED"),
            motor_lag=("no measurement found", "0 s declared default; UNVERIFIED"),
            windage=("C3 WINDAGE.md hull silhouettes re-clipped (assets/build_vessels_v2.py build_blueboat); "
                     "superstructure band 0.46 m public height", "hull DERIVED; superstructure UNVERIFIED"),
            collision=("store page 1.20 x 0.93 m; asset vertex bounds", "conservative envelope"),
            speeds=("store page: 3 m/s max, endurance at 1 m/s (~30 W)", "manufacturer")))


def _wamv(option: str, config: str) -> VesselSpecV2:
    d = _data.WAMV[config]
    mass = d["mass_kg"]
    com_b = d["com_base_link_m"]
    damping = WAMV_DAMPING_C if option == "opt1" else WAMV_DAMPING_J
    if option == "opt1":
        fwd, rev = vrx_classic_thrust(1.0), WAMV_C_CLIP_REV_N
        unit = "VRX classic normalized command in [-1, 1] (maxCmd 1.0)"
        knots = _classic_knots()
        law = ("VRX classic GLF (usv_gazebo_thrust_plugin.cc:318-341) sampled every 0.025; per-motor "
               "+242.67 N at +1, -101.85 N at -1 clipped to -100 N. Per Sarda 2016 Table 2 this curve "
               "equals the MEASURED TOTAL of two thrusters (inference, report 6.4).")
        thrust_src = VRX_C + " wamv_gazebo_thruster_config.xacro:14-17; usv_gazebo_thrust_plugin.cc:318-341"
    else:
        fwd, rev = WAMV_J_FORWARD_N, WAMV_J_REVERSE_N
        unit = "N (gz-sim Thruster force command, clamped)"
        knots = ((-rev, -rev), (0.0, 0.0), (fwd, fwd))
        law = ("gz-sim8 Thruster: force command clamped to [min_thrust_cmd, max_thrust_cmd] and applied "
               "directly (Thruster.cc:517-526); +2353.53 N is sized from the OLD classic drag at 15 kn, "
               "-1000 N is the plugin default, not a VRX design value.")
        thrust_src = VRX_J + " wamv_gazebo_thruster_config.xacro:6-8,19; gz-sim8 Thruster.cc:113-119,430-451"
    motors = []
    for side, sign in (("port", 1.0), ("starboard", -1.0)):
        p_base = (WAMV_ENGINE_BASE_M[0] + WAMV_PROP_OFFSET_M[0], sign * WAMV_ENGINE_BASE_M[1],
                  WAMV_ENGINE_BASE_M[2] + WAMV_PROP_OFFSET_M[2])
        motors.append(Motor(
            name=side, position_com_m=tuple(p - c for p, c in zip(p_base, com_b)),
            direction_body=(1.0, 0.0, 0.0), forward_limit_n=fwd, reverse_limit_n=rev,
            command_unit=unit, command_to_thrust=knots, time_constant_s=0.0,
            law_note=law + " VRX applies thrust instantly (no motor pole); physical lag UNVERIFIED."))
    arm = WAMV_ENGINE_BASE_M[1]
    k_heave, k_roll, k_pitch = d["stiffness_heave_n_m"], d["stiffness_roll_nm_rad"], d["stiffness_pitch_nm_rad"]
    mean_depth = sum(d["point_depths_m"]) / 4.0
    hydro = Hydrostatics(
        water_density_kg_m3=RHO_WATER, gravity_m_s2=GRAVITY, displaced_volume_m3=mass / RHO_WATER,
        maximum_volume_m3=d["maximum_volume_m3"], waterplane_area_m2=d["waterplane_m2"],
        draft_m=mean_depth, freeboard_m=WAMV_PONTOON_TOP_BASE_Z_M - mean_depth,
        com_height_above_water_m=d["com_height_above_water_m"],
        stiffness_heave_n_m=k_heave, stiffness_roll_nm_rad=k_roll, stiffness_pitch_nm_rad=k_pitch,
        heave_pitch_coupling_n_rad=d["heave_pitch_coupling_n_rad"],
        gm_roll_m=k_roll / (mass * GRAVITY), gm_pitch_m=k_pitch / (mass * GRAVITY),
        orbital_depth_m=mean_depth / 2.0,
        wave_footprint_length_m=WAMV_HULL_LENGTH_M,
        wave_footprint_beam_m=2.0 * (WAMV_PONTOON_Y_M + WAMV_HULL_RADIUS_M),
        static_roll_rad=d["static_roll_rad"], static_pitch_rad=d["static_pitch_rad"],
        model=("VRX Surface buoyancy layout (4 points, half-hull 2.45 m circular segments R=0.213, depth "
               "clipped to [0,R], Surface.cc:229-261,302-306) solved for 3-DOF equilibrium at rho=1000, "
               "g=9.81; tangent stiffness about the COM; draft = mean point immersion"))
    lo, hi = d["visual_bounds_com_m"]
    elo, ehi = d["envelope_bounds_com_m"]
    rect = ((elo[0], elo[1]), (ehi[0], elo[1]), (ehi[0], ehi[1]), (elo[0], ehi[1]))
    cap_c, cap_a, cap_r = d["capsule"]
    wind = Windage(
        frontal_area_m2=d["frontal_area_m2"], lateral_area_m2=d["lateral_area_m2"], length_m=hi[0] - lo[0],
        lateral_centroid_height_above_water_m=d["lateral_centroid_height_above_water_m"],
        frontal_centroid_height_above_water_m=d["frontal_centroid_height_above_water_m"],
        roll_lever_com_m=d["lateral_centroid_z_from_com_m"],
        lateral_centroid_x_com_m=d["lateral_centroid_x_from_com_m"],
        c_x0=0.9, c_y0=0.9, c_n0=0.1, air_density_kg_m3=RHO_AIR, u10_profile="neutral_log",
        roughness_length_m=2.0e-4,
        status=("silhouette union of every visual mesh/primitive of the " + config + " assembly above the upright "
                "equilibrium waterline; coefficients 0.9/0.9/0.1 are C3 engineering assumptions; VRX's own "
                "dimensional wind gains (0.5,0.5,0.33) are a different model"))
    top_model = terminal_speed(damping.linear[0], damping.quadratic[0], 2 * fwd)
    label = {"opt1": "option 1: J geometry/mass/buoyancy + classic damping/thrust",
             "opt2": "option 2: all from J"}[option]
    cfg_text = {"competition": ("VRX competition default: vrx_sensors_enabled + H thrusters (model.py:239-240); "
                                "base, 2 batteries, 2 engines, 2 propellers, CPU cases, 3 cameras, GPS, IMU, "
                                "3-D lidar, acoustic receiver, ball shooter; joints frozen at zero"),
                "bare": ("VRX platform default (wamv_gazebo.urdf.xacro:5-15 all sensor args false): base, "
                         "2 batteries, 2 engines, 2 propellers; joints frozen at zero")}[config]
    name = "wamv_vrx_v2_" + option + ("" if config == "competition" else "_bare")
    return VesselSpecV2(
        name=name, schema_version=SCHEMA_VERSION, option=label, configuration=cfg_text,
        asset_relpath=d["asset"], asset_sha256=d["asset_sha256"], asset_kind="rigid_object", scale=1.0,
        bow_body_axis="+x", mass_kg=mass, com_actor_m=(0.0, 0.0, 0.0),
        inertia_com_kg_m2=d["inertia_com_kg_m2"],
        length_overall_m=hi[0] - lo[0], beam_overall_m=hi[1] - lo[1], height_overall_m=hi[2] - lo[2],
        damping=damping, hydrostatics=hydro,
        added_mass=AddedMass(
            (0.0, 0.0, 0.1, 0.1, 0.1, 0.0), False,
            ("VRX values: xDotU=yDotV=nDotR=0 explicit (J dynamics xacro:45-47; classic :12-14), zDotW=kDotP=mDotQ "
             "plugin default 0.1 (SimpleHydrodynamics.cc:137-139). Negligible (<0.05 percent of rigid mass); the v2 "
             "kernel has no effective-mass solve, so they are metadata only.")),
        propulsion=Propulsion(
            motors=tuple(motors), planar_arm_m=arm, yaw_command_limit_nm=arm * (fwd + rev),
            pure_yaw_limit_nm=2 * arm * min(fwd, rev),
            thrust_line_height_com_m=motors[0].position_com_m[2], mixer_mount_z_m=0.0,
            note=("two fixed aft engines of the H layout, steering frozen at 0 (VRX engines can steer: "
                  "engine.xacro:47-53); thrust at the propeller link origin")),
        windage=wind,
        collision=CollisionFootprint(rect, tuple(cap_c), cap_a, cap_r,
                                     "axis-aligned envelope of every visual mesh and collision primitive (includes "
                                     "the inter-hull gap and the 0.24 m propeller collider); capsule = minimum-area "
                                     "capsule containing it"),
        speeds=SpeedReference(
            cruise_m_s=1.0, reference_m_s=1.0, top_model_m_s=top_model, top_published_m_s=None,
            note=("cruise = typical USV16 transit speed (Sarda et al. 2016 preprint p.10, 'nominal surge speed of "
                  "U = 1 m/s'); the real USV16 with 2x120 N thrusters reached about 1.5 m/s (preprint p.11); "
                  "top_model is this profile's own terminal speed")),
        provenance=_provenance(
            mass=("vehicles_v2_data.WAMV[" + repr(config) + "] from assets/build_vessels_v2.py expanding " + VRX_J
                  + " xacros (wamv_base.urdf.xacro:119-123, battery.xacro:17-27, engine.xacro:23-26,41-44, "
                  "wamv_gazebo.urdf.xacro:44-45,156-184, components/*.xacro)", "DERIVED from source links"),
            com=("same aggregation; COM in base_link = com_base_link_m", "DERIVED"),
            inertia=("parallel-axis sum of every link tensor rotated to base axes", "DERIVED; full tensor kept"),
            dimensions=("visual mesh union bounds (WAM-V-Base.dae unit 0.01 m, engine/propeller unit 1 m)",
                        "DERIVED"),
            damping=(damping.source_set, "VRX simulator values; classic surge after Sarda 2016 fit, others "
                     "UNVERIFIED" if option == "opt1" else "VRX simulator values inflated for Gazebo stability "
                     "(PR #706); not physical identification"),
            hydrostatics=("wamv_gazebo_dynamics_plugin.xacro:7-38 + Surface.cc:229-261,302-306 solved in "
                          "repo/vessel_physics_ref_v2.py", "DERIVED from the VRX model (VRX waterplane 3.6 m2 vs "
                          "USV16 1.6 m2 measured: VRX floats shallower)"),
            added_mass=("SimpleHydrodynamics.cc:135-140; dynamics xacro:45-47", "source values; disabled"),
            motors=(thrust_src + "; positions wamv_aft_thrusters.xacro:4-5 + engine.xacro:59",
                    "simulator command law; delivered physical thrust UNVERIFIED"),
            motor_lag=("VRX applies thrust without a motor pole (Thruster.cc:517-526; classic "
                       "usv_gazebo_thrust_plugin.cc:369-391)", "0 s; physical lag UNVERIFIED"),
            windage=("assets/build_vessels_v2.py silhouettes (C3 method, windage_geometry.py:36-48)",
                     "geometry DERIVED; coefficients UNVERIFIED"),
            collision=("envelope of visuals and xacro collision primitives", "conservative envelope"),
            speeds=("Sarda et al. 2016 arXiv:1702.04941", "cruise literature; top model-derived")))


# ============================================================== validation
def _finite(value: float, label: str, *, minimum=None, strict=False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number")
    if minimum is not None and (value < minimum or (strict and value == minimum)):
        raise ValueError(label + " is out of range")


def _vector(value, length: int, label: str, *, minimum=None) -> None:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(label + " must be an immutable tuple of length " + str(length))
    for number in value:
        _finite(number, label, minimum=minimum)


def _det3(m) -> float:
    a, b, c = m
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def _close(a: float, b: float, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


REQUIRED_PROVENANCE = frozenset({"mass", "com", "inertia", "dimensions", "damping", "hydrostatics",
                                  "added_mass", "motors", "motor_lag", "windage", "collision", "speeds"})


def validate_vehicle(spec: VesselSpecV2) -> None:
    """Raise ValueError for a nonphysical, inconsistent or incomplete profile. No I/O."""
    if spec.schema_version != SCHEMA_VERSION or spec.scale != 1.0:
        raise ValueError("v2 profiles require schema 2 and scale 1.0")
    if spec.asset_kind != "rigid_object" or not spec.asset_relpath or len(spec.asset_sha256) != 64:
        raise ValueError("v2 profiles require one rigid-object asset with a SHA-256")
    if spec.bow_body_axis != "+x" or spec.com_actor_m != (0.0, 0.0, 0.0):
        raise ValueError("v2 assets are FLU with the actor origin at the COM")
    for key in ("mass_kg", "length_overall_m", "beam_overall_m", "height_overall_m"):
        _finite(getattr(spec, key), key, minimum=0.0, strict=True)
    I = spec.inertia_com_kg_m2
    if not isinstance(I, tuple) or len(I) != 3:
        raise ValueError("inertia must be a tuple of three rows")
    for row in I:
        _vector(row, 3, "inertia row")
    if any(not _close(I[i][j], I[j][i], 0, 1e-9) for i in range(3) for j in range(3)):
        raise ValueError("inertia must be symmetric")
    if not (I[0][0] > 0 and I[0][0] * I[1][1] - I[0][1] ** 2 > 0 and _det3(I) > 0):
        raise ValueError("inertia must be positive definite")
    half = sum(I[i][i] for i in range(3)) / 2       # triangle inequality: tr(I)/2*E - I >= 0
    second = tuple(tuple((half if i == j else 0.0) - I[i][j] for j in range(3)) for i in range(3))
    tol = 1e-9 * max(1.0, half)
    if (any(second[i][i] < -tol for i in range(3)) or _det3(second) < -tol * half * half
            or any(second[i][i] * second[j][j] - second[i][j] ** 2 < -tol * half for i in range(3) for j in range(i))):
        raise ValueError("inertia violates the triangle inequalities")
    _vector(spec.damping.linear, 6, "linear damping", minimum=0.0)
    _vector(spec.damping.quadratic, 6, "quadratic damping", minimum=0.0)
    if any(spec.damping.linear[i] + spec.damping.quadratic[i] <= 0 for i in range(6)):
        raise ValueError("every axis needs positive damping")
    _vector(spec.added_mass.diagonal, 6, "added mass", minimum=0.0)
    if spec.added_mass.active_in_v2 or not spec.added_mass.policy:
        raise ValueError("added mass must be disabled metadata with an explicit policy in schema 2")
    h = spec.hydrostatics
    for key, value in asdict(h).items():
        if key != "model":
            _finite(value, "hydrostatics." + key)
    for key in ("water_density_kg_m3", "gravity_m_s2", "displaced_volume_m3", "maximum_volume_m3",
                "waterplane_area_m2", "draft_m", "freeboard_m", "stiffness_heave_n_m", "stiffness_roll_nm_rad",
                "stiffness_pitch_nm_rad", "gm_roll_m", "gm_pitch_m", "wave_footprint_length_m",
                "wave_footprint_beam_m"):
        _finite(getattr(h, key), key, minimum=0.0, strict=True)
    _finite(h.orbital_depth_m, "orbital depth", minimum=0.0)
    if not _close(h.displaced_volume_m3 * h.water_density_kg_m3, spec.mass_kg, 1e-9):
        raise ValueError("equilibrium displacement must support the configured mass")
    if h.maximum_volume_m3 <= h.displaced_volume_m3:
        raise ValueError("full-immersion volume must exceed the equilibrium displacement")
    if not _close(h.stiffness_heave_n_m, h.water_density_kg_m3 * h.gravity_m_s2 * h.waterplane_area_m2, 1e-9):
        raise ValueError("heave stiffness must equal rho*g*waterplane")
    for k, gm in ((h.stiffness_roll_nm_rad, h.gm_roll_m), (h.stiffness_pitch_nm_rad, h.gm_pitch_m)):
        if not _close(k, spec.mass_kg * h.gravity_m_s2 * gm, 1e-9):
            raise ValueError("roll/pitch stiffness must equal m*g*GM")
    if not -spec.height_overall_m < h.com_height_above_water_m < spec.height_overall_m:
        raise ValueError("COM height above water is outside the vessel")
    if abs(h.static_roll_rad) > 0.1 or abs(h.static_pitch_rad) > 0.1:
        raise ValueError("source equilibrium attitude exceeds the small-angle model (0.1 rad)")
    p = spec.propulsion
    if not isinstance(p.motors, tuple) or len(p.motors) != 2 or [m.name for m in p.motors] != ["port", "starboard"]:
        raise ValueError("schema 2 supports exactly (port, starboard) fixed motors")
    _finite(p.planar_arm_m, "planar arm", minimum=0.0, strict=True)
    port, stbd = p.motors
    for motor in p.motors:
        _vector(motor.position_com_m, 3, "motor position")
        if motor.direction_body != (1.0, 0.0, 0.0):
            raise ValueError("schema 2 motors push along +x")
        _finite(motor.forward_limit_n, "forward limit", minimum=0.0, strict=True)
        _finite(motor.reverse_limit_n, "reverse limit", minimum=0.0, strict=True)
        _finite(motor.time_constant_s, "motor lag", minimum=0.0)
        knots = motor.command_to_thrust
        if not isinstance(knots, tuple) or len(knots) < 3:
            raise ValueError("command law needs at least three knots")
        for knot in knots:
            _vector(knot, 2, "command knot")
        if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(knots, knots[1:])):
            raise ValueError("command law must be strictly increasing in command and monotone in thrust")
        if (0.0, 0.0) not in knots or not knots[0][0] < 0.0 < knots[-1][0]:
            raise ValueError("command zero must map to zero thrust, with commands on both sides")
        if not (_close(knots[0][1], -motor.reverse_limit_n, 1e-12) and _close(knots[-1][1], motor.forward_limit_n, 1e-12)):
            raise ValueError("command-law end points must equal the delivered thrust limits")
    for a, b in ((port.forward_limit_n, stbd.forward_limit_n), (port.reverse_limit_n, stbd.reverse_limit_n)):
        if not _close(a, b, 1e-12):
            raise ValueError("schema 2 nominal motors are identical (asymmetry belongs to degradation doses)")
    # mirror symmetry about the hull centreline
    centre = 0.5 * (port.position_com_m[1] + stbd.position_com_m[1])
    if not (_close(port.position_com_m[1] - centre, p.planar_arm_m, 0, 1e-9)
            and _close(centre - stbd.position_com_m[1], p.planar_arm_m, 0, 1e-9)
            and _close(port.position_com_m[0], stbd.position_com_m[0], 0, 1e-12)
            and _close(port.position_com_m[2], stbd.position_com_m[2], 0, 1e-12)):
        raise ValueError("motors must be mirror images about the centreline at +/- planar_arm_m")
    if not _close(p.yaw_command_limit_nm, p.planar_arm_m * (port.forward_limit_n + port.reverse_limit_n), 1e-12):
        raise ValueError("yaw request scale must be arm*(fwd+rev)")
    if not _close(p.pure_yaw_limit_nm, 2 * p.planar_arm_m * min(port.forward_limit_n, port.reverse_limit_n), 1e-12):
        raise ValueError("pure-yaw limit must be 2*arm*min(fwd,rev)")
    if not _close(p.thrust_line_height_com_m, port.position_com_m[2], 0, 1e-12):
        raise ValueError("thrust-line height must be the motors' z about the COM")
    _finite(p.mixer_mount_z_m, "mixer mount z")
    w = spec.windage
    for key in ("frontal_area_m2", "lateral_area_m2", "length_m", "lateral_centroid_height_above_water_m",
                "frontal_centroid_height_above_water_m", "c_x0", "c_y0", "air_density_kg_m3", "roughness_length_m"):
        _finite(getattr(w, key), key, minimum=0.0, strict=True)
    _finite(w.c_n0, "c_n0")
    _finite(w.lateral_centroid_x_com_m, "lateral centroid x")
    if not _close(w.roll_lever_com_m, w.lateral_centroid_height_above_water_m - h.com_height_above_water_m, 0, 1e-9):
        raise ValueError("wind roll lever must be measured from the COM, not from the waterline")
    if w.u10_profile != "neutral_log" or not w.roughness_length_m < w.lateral_centroid_height_above_water_m:
        raise ValueError("U10 conversion must be a neutral log law with z0 below the reference height")
    c = spec.collision
    _vector(c.capsule_center_com_xy_m, 2, "capsule centre")
    _finite(c.capsule_segment_half_length_m, "capsule half length", minimum=0.0)
    _finite(c.capsule_radius_m, "capsule radius", minimum=0.0, strict=True)
    pts = c.polygon_com_xy_m
    if not isinstance(pts, tuple) or len(pts) < 3:
        raise ValueError("collision polygon needs at least three vertices")
    for q in pts:
        _vector(q, 2, "polygon vertex")
    for i, q in enumerate(pts):
        b, d_ = pts[(i + 1) % len(pts)], pts[(i + 2) % len(pts)]
        if (b[0] - q[0]) * (d_[1] - b[1]) - (b[1] - q[1]) * (d_[0] - b[0]) <= 0:
            raise ValueError("collision polygon must be strictly convex and counter-clockwise")
    if not (min(q[0] for q in pts) < 0 < max(q[0] for q in pts) and min(q[1] for q in pts) < 0 < max(q[1] for q in pts)):
        raise ValueError("collision polygon must contain the COM")
    if (max(q[0] for q in pts) - min(q[0] for q in pts) < spec.length_overall_m - 1e-6
            or max(q[1] for q in pts) - min(q[1] for q in pts) < spec.beam_overall_m - 1e-6):
        raise ValueError("collision polygon must enclose the overall length and beam")
    for q in pts:   # capsule must contain the polygon (conservative alternative)
        dx = max(0.0, abs(q[0] - c.capsule_center_com_xy_m[0]) - c.capsule_segment_half_length_m)
        dy = q[1] - c.capsule_center_com_xy_m[1]
        if math.hypot(dx, dy) > c.capsule_radius_m + 1e-9:
            raise ValueError("capsule must contain the collision polygon")
    s = spec.speeds
    for key in ("cruise_m_s", "reference_m_s", "top_model_m_s"):
        _finite(getattr(s, key), key, minimum=0.0, strict=True)
    if s.top_published_m_s is not None:
        _finite(s.top_published_m_s, "published top speed", minimum=0.0, strict=True)
    if max(s.cruise_m_s, s.reference_m_s) >= s.top_model_m_s:
        raise ValueError("cruise/reference speed must be below the modelled top speed")
    groups = {q.parameter for q in spec.provenance if q.citation and q.status}
    if not REQUIRED_PROVENANCE <= groups:
        raise ValueError("missing provenance: " + ", ".join(sorted(REQUIRED_PROVENANCE - groups)))


# ============================================================== laws and references
def command_to_thrust(motor: Motor, command: float) -> float:
    """Native command -> delivered thrust by the monotone knot table (clipped)."""
    _finite(command, "command")
    knots = motor.command_to_thrust
    command = min(max(command, knots[0][0]), knots[-1][0])
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if command <= x1:
            return y0 + (command - x0) * (y1 - y0) / (x1 - x0)
    return knots[-1][1]


def thrust_to_command(motor: Motor, thrust_n: float) -> float:
    """Smallest-magnitude native command delivering thrust_n (clipped to the limits)."""
    _finite(thrust_n, "thrust")
    force = min(max(thrust_n, -motor.reverse_limit_n), motor.forward_limit_n)
    if force == 0.0:
        return 0.0
    knots = motor.command_to_thrust
    if force > 0:
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if y1 >= force and y1 > y0 and x1 > 0:
                x0, y0 = (0.0, 0.0) if x0 < 0 else (x0, y0)
                return x0 + (force - y0) * (x1 - x0) / (y1 - y0)
        return knots[-1][0]
    for (x0, y0), (x1, y1) in reversed(list(zip(knots, knots[1:]))):
        if y0 <= force and y1 > y0 and x0 < 0:
            x1, y1 = (0.0, 0.0) if x1 > 0 else (x1, y1)
            return x1 - (y1 - force) * (x1 - x0) / (y1 - y0)
    return knots[0][0]


def motor_wrench(spec: VesselSpecV2, thrusts_n: tuple[float, float], *, reference: str = "centreline",
                 pitch_coupling: bool = False) -> Six:
    """Body wrench (Fx,Fy,Fz,Mx,My,Mz) about the COM from delivered (port, starboard) thrust.

    reference="centreline": yaw lever = +/-planar_arm_m (hull resistance also acts on the
    centreline; see the module note). reference="com_geometric": literal r x F with the
    true 3-D motor positions about the COM (for audits only).
    """
    port, stbd = spec.propulsion.motors
    if len(thrusts_n) != 2:
        raise ValueError("two delivered thrusts (port, starboard) are required")
    for motor, t in zip((port, stbd), thrusts_n):
        _finite(t, "thrust")
        if not -motor.reverse_limit_n - 1e-9 <= t <= motor.forward_limit_n + 1e-9:
            raise ValueError("delivered thrust outside the physical motor limits")
    P, S = thrusts_n
    if reference == "centreline":
        arm = spec.propulsion.planar_arm_m
        lever = spec.propulsion.mixer_mount_z_m if not pitch_coupling else spec.propulsion.thrust_line_height_com_m
        return (P + S, 0.0, 0.0, 0.0, lever * (P + S), arm * (S - P))
    if reference == "com_geometric":
        w = [0.0] * 6
        for motor, t in zip((port, stbd), thrusts_n):
            r = motor.position_com_m
            w[0] += t
            w[3] += 0.0
            w[4] += r[2] * t
            w[5] += -r[1] * t
        return tuple(w)
    raise ValueError("reference must be 'centreline' or 'com_geometric'")


@dataclass(frozen=True)
class Allocation:
    thrusts_n: tuple[float, float]
    scale: float
    saturated: bool
    wrench: Six


def allocate_planar(spec: VesselSpecV2, surge_n: float, yaw_nm: float) -> Allocation:
    """CPU reference inverse allocation with radial (direction-preserving) saturation.

    P + S = X, arm*(S - P) = N; both requests are scaled by one common factor
    until every motor is inside [-rev, fwd]. D3's thruster_mixer.py owns the
    batched runtime version; this reference is for analysis and tests.
    """
    _finite(surge_n, "surge request")
    _finite(yaw_nm, "yaw request")
    arm = spec.propulsion.planar_arm_m
    desired = (0.5 * (surge_n - yaw_nm / arm), 0.5 * (surge_n + yaw_nm / arm))
    scale = 1.0
    for motor, t in zip(spec.propulsion.motors, desired):
        if t > motor.forward_limit_n:
            scale = min(scale, motor.forward_limit_n / t)
        elif t < -motor.reverse_limit_n:
            scale = min(scale, motor.reverse_limit_n / -t)
    thrusts = (desired[0] * scale, desired[1] * scale)
    return Allocation(thrusts, scale, scale < 1.0, motor_wrench(spec, thrusts))


def motor_lag_step(motor: Motor, previous_n: float, target_n: float, dt_s: float) -> float:
    """Exact zero-order-hold first-order lag (reference; tau = 0 returns the target)."""
    _finite(dt_s, "dt", minimum=0.0, strict=True)
    target = min(max(target_n, -motor.reverse_limit_n), motor.forward_limit_n)
    if motor.time_constant_s == 0.0:
        return target
    alpha = -math.expm1(-dt_s / motor.time_constant_s)
    return previous_n + alpha * (target - previous_n)


def wind_speed_at_height(u10_m_s: float, height_m: float, roughness_m: float = 2.0e-4) -> float:
    """Neutral marine log profile U(z) = U10 ln(z/z0) / ln(10/z0).

    z0 = 2e-4 m is the open-sea roughness class (Davenport-Wieringa, WMO-No. 8);
    heights below about 1 m lie inside the wave boundary layer, where the law
    is a declared extrapolation (UNVERIFIED).
    """
    _finite(u10_m_s, "U10", minimum=0.0)
    _finite(height_m, "height", minimum=0.0, strict=True)
    if not height_m > roughness_m:
        raise ValueError("height must exceed the roughness length")
    return u10_m_s * math.log(height_m / roughness_m) / math.log(10.0 / roughness_m)


def vessel_wind_speed_from_u10(spec: VesselSpecV2, u10_m_s: float) -> float:
    w = spec.windage
    return wind_speed_at_height(u10_m_s, w.lateral_centroid_height_above_water_m, w.roughness_length_m)


# ============================================================== adapters (no imports of D3/C3)
def to_d3_hydro_kwargs(spec: VesselSpecV2, *, rollpitch_damping: float | None = None) -> dict:
    """Keyword arguments for D3 hydro_v2.HydroSpec (contract as of 2026-09-30).

    rollpitch_damping optionally overrides the roll AND pitch linear damping
    (owner decision for BlueBoat); quadratic roll/pitch terms are kept.
    """
    validate_vehicle(spec)
    h, d = spec.hydrostatics, spec.damping
    ang_lin = d.linear[3:]
    if rollpitch_damping is not None:
        _finite(rollpitch_damping, "roll/pitch damping", minimum=0.0, strict=True)
        ang_lin = (float(rollpitch_damping), float(rollpitch_damping), d.linear[5])
    return {
        "mass_kg": spec.mass_kg, "inertia_kgm2": spec.inertia_com_kg_m2,
        "linear_damping": d.linear[:3], "quadratic_damping": d.quadratic[:3],
        "angular_linear_damping": tuple(ang_lin), "angular_quadratic_damping": d.quadratic[3:],
        "maximum_volume_m3": h.maximum_volume_m3, "waterplane_m2": h.waterplane_area_m2,
        "roll_stiffness_nm": h.stiffness_roll_nm_rad, "pitch_stiffness_nm": h.stiffness_pitch_nm_rad,
        "length_m": h.wave_footprint_length_m, "beam_m": h.wave_footprint_beam_m,
        "orbital_depth_m": h.orbital_depth_m, "equilibrium_com_z_m": h.com_height_above_water_m,
        "water_density": h.water_density_kg_m3, "gravity": h.gravity_m_s2,
    }


def to_d3_thruster_kwargs(spec: VesselSpecV2, *, pitch_coupling: bool = False) -> dict:
    """Keyword arguments for D3 thruster_mixer.ThrusterSpec (contract as of 2026-09-30)."""
    validate_vehicle(spec)
    p = spec.propulsion
    port, stbd = p.motors
    return {
        "forward_limit_n": (port.forward_limit_n, stbd.forward_limit_n),
        "reverse_limit_n": (port.reverse_limit_n, stbd.reverse_limit_n),
        "arm_m": p.planar_arm_m, "yaw_command_limit_nm": p.yaw_command_limit_nm,
        "nominal_lag_s": (port.time_constant_s, stbd.time_constant_s),
        "mount_z_m": p.thrust_line_height_com_m if pitch_coupling else p.mixer_mount_z_m,
    }


def to_c3_windage_kwargs(spec: VesselSpecV2) -> dict:
    """Keyword arguments for C3 wind.VesselWindage. centroid_height_m is the roll
    lever about the COM (C3 WINDAGE.md: use z_CP - z_COM, not the waterline height)."""
    validate_vehicle(spec)
    w = spec.windage
    return {"frontal_area_m2": w.frontal_area_m2, "lateral_area_m2": w.lateral_area_m2,
            "length_m": w.length_m, "centroid_height_m": w.roll_lever_com_m,
            "longitudinal_cp_m": w.lateral_centroid_x_com_m,
            "c_x0": w.c_x0, "c_y0": w.c_y0, "c_n0": w.c_n0}


# ============================================================== pinned derived values (G7)
# Platform-stable registry (G7 item A, 2026-10-01). Every registry value the
# builders COMPUTE (rather than copy from vehicles_v2_data or a declared
# constant above) is pinned as a declared constant in
# vehicles_v2_data.PINNED_DERIVED. At import the builders still compute each
# value; the computed value must agree with its pin within PIN_REL_TOL /
# PIN_ABS_TOL, and the PIN is what the profile (and therefore every vessel
# record, layout id and GEOMETRY_VERSION_DIGEST) carries. Reason: the last ulp
# of a computed value depends on the platform. Found case: WAM-V draft_m =
# sum(point_depths_m) / 4 is 0.10916531789468821 under Python >= 3.12, whose
# builtin sum() of floats is compensated (Neumaier summation), and
# 0.1091653178946882 under Python 3.11 (plain left-to-right addition);
# vrx_classic_thrust also goes through libm exp/pow. A registry change must be
# re-pinned (render_pinned_derived), otherwise the import fails.
PIN_REL_TOL = 1.0e-9
PIN_ABS_TOL = 1.0e-12
PINS_SCHEMA = "usvbench.vehicles_v2.pinned_derived/1"
_MOTOR_PINS = tuple("propulsion.motors[%d].%s" % (i, leaf) for i in (0, 1)
                    for leaf in ("position_com_m", "forward_limit_n", "reverse_limit_n", "command_to_thrust"))
_COMMON_PINS = (("hydrostatics.displaced_volume_m3", "hydrostatics.orbital_depth_m") + _MOTOR_PINS
                + ("propulsion.yaw_command_limit_nm", "propulsion.pure_yaw_limit_nm",
                   "propulsion.thrust_line_height_com_m", "speeds.top_model_m_s"))
# Paths each builder computes (everything else is copied from the data module or a declared literal).
PINNED_PATHS = MappingProxyType({
    "blueboat": _COMMON_PINS + ("hydrostatics.stiffness_heave_n_m", "hydrostatics.stiffness_roll_nm_rad",
                                "hydrostatics.stiffness_pitch_nm_rad", "hydrostatics.static_roll_rad",
                                "hydrostatics.static_pitch_rad", "windage.roll_lever_com_m"),
    "wamv": _COMMON_PINS + ("length_overall_m", "beam_overall_m", "height_overall_m", "hydrostatics.draft_m",
                            "hydrostatics.freeboard_m", "hydrostatics.gm_roll_m", "hydrostatics.gm_pitch_m",
                            "hydrostatics.wave_footprint_beam_m", "windage.length_m"),
})


def _path_parts(path: str):
    parts = []
    for token in path.split("."):
        name, bracket, rest = token.partition("[")
        parts.append(name)
        while bracket:
            index, _, rest = rest.partition("]")
            parts.append(int(index))
            _, bracket, rest = rest.partition("[")
    return parts


def _get_path(obj, path: str):
    for part in _path_parts(path):
        obj = obj[part] if isinstance(part, int) else getattr(obj, part)
    return obj


def _set_parts(obj, parts, value):
    if not parts:
        return value
    head, rest = parts[0], parts[1:]
    if isinstance(head, int):
        items = list(obj)
        items[head] = _set_parts(items[head], rest, value)
        return tuple(items)
    return replace(obj, **{head: _set_parts(getattr(obj, head), rest, value)})


def _float_leaves(value):
    if isinstance(value, tuple):
        out = []
        for item in value:
            out.extend(_float_leaves(item))
        return out
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("pinned registry values must be numbers or tuples of numbers")
    return [float(value)]


def _shape(value):
    return tuple(_shape(v) for v in value) if isinstance(value, tuple) else None


def pinned_value_agrees(computed, pinned) -> bool:
    """True when every number of ``computed`` lies within the pin tolerance of ``pinned``."""
    if _shape(computed) != _shape(pinned):
        return False
    return all(math.isclose(a, b, rel_tol=PIN_REL_TOL, abs_tol=PIN_ABS_TOL)
               for a, b in zip(_float_leaves(computed), _float_leaves(pinned)))


def _pin_profile(spec: VesselSpecV2, kind: str) -> VesselSpecV2:
    """Replace every computed value of ``spec`` by its declared pin (checked)."""
    paths = PINNED_PATHS[kind]
    table = getattr(_data, "PINNED_DERIVED", None)
    if hasattr(table, "record_pin"):        # tools/tests only: capture computed values (registry not pinned)
        for path in paths:
            table.record_pin(spec.name, path, _get_path(spec, path))
        return spec
    if not isinstance(table, Mapping) or spec.name not in table:
        raise ValueError("vehicles_v2_data.PINNED_DERIVED has no pins for profile %r; re-pin the registry"
                         % (spec.name,))
    pins = table[spec.name]
    if set(pins) != set(paths):
        raise ValueError("pinned derived paths of %r differ from the builder's computed paths: missing %s, "
                         "stale %s" % (spec.name, sorted(set(paths) - set(pins)), sorted(set(pins) - set(paths))))
    for path in paths:
        computed, pinned = _get_path(spec, path), pins[path]
        if not pinned_value_agrees(computed, pinned):
            raise ValueError("registry value %s.%s = %r disagrees with its pin %r beyond rel %g / abs %g; "
                             "re-pin the registry" % (spec.name, path, computed, pinned, PIN_REL_TOL, PIN_ABS_TOL))
        spec = _set_parts(spec, _path_parts(path), pinned)
    return spec


def render_pinned_derived(values) -> str:
    """vehicles_v2_data.py text block for {profile: {path: value}} (exact repr of every number)."""
    lines = ["# BEGIN PINNED_DERIVED (G7 item A; written by vehicles_v2.render_pinned_derived; do not edit by hand)",
             "PINNED_DERIVED_SCHEMA = %r" % PINS_SCHEMA, "PINNED_DERIVED = {"]
    for name in sorted(values):
        lines.append("    %r: {" % name)
        for path in sorted(values[name]):
            lines.append("        %r: %r," % (path, values[name][path]))
        lines.append("    },")
    lines += ["}", "# END PINNED_DERIVED"]
    return "\n".join(lines) + "\n"


# ============================================================== registry
def _build_registry() -> Mapping[str, VesselSpecV2]:
    profiles = [_pin_profile(_blueboat(), "blueboat")]
    for option in ("opt1", "opt2"):
        for config in ("competition", "bare"):
            profiles.append(_pin_profile(_wamv(option, config), "wamv"))
    for spec in profiles:
        validate_vehicle(spec)
    return MappingProxyType({spec.name: spec for spec in profiles})


VEHICLES_V2: Mapping[str, VesselSpecV2] = _build_registry()
# "pinned": every computed value is its declared pin; "recorded": a tool captured the computed
# values instead (never a contract registry; journal_cfg_v2 refuses it).
PINS_STATUS = "recorded" if hasattr(getattr(_data, "PINNED_DERIVED", None), "record_pin") else "pinned"
# Controller recommendation pending the owner's decision (report 9.1; R2 decision 5).
DEFAULT_WAMV_PROFILE = "wamv_vrx_v2_opt1"


def get_vehicle_v2(name: str) -> VesselSpecV2:
    try:
        return VEHICLES_V2[name]
    except KeyError:
        raise ValueError("Unknown v2 vehicle " + repr(name) + ". Valid names: "
                         + ", ".join(sorted(VEHICLES_V2)) + ".") from None


__all__ = ["AXES", "VesselSpecV2", "Motor", "Propulsion", "Hydrostatics", "Damping", "AddedMass",
           "Windage", "CollisionFootprint", "SpeedReference", "Provenance", "VEHICLES_V2",
           "DEFAULT_WAMV_PROFILE", "get_vehicle_v2", "validate_vehicle", "command_to_thrust",
           "thrust_to_command", "motor_wrench", "allocate_planar", "motor_lag_step",
           "wind_speed_at_height", "vessel_wind_speed_from_u10", "vrx_classic_thrust",
           "to_d3_hydro_kwargs", "to_d3_thruster_kwargs", "to_c3_windage_kwargs", "terminal_speed",
           "PINNED_PATHS", "PIN_REL_TOL", "PIN_ABS_TOL", "PINS_STATUS", "pinned_value_agrees",
           "render_pinned_derived"]
