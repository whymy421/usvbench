"""Frozen journal task specifications, contract version 2 (CPU importable).

Plain Python data; no Isaac Lab imports. Isaac cfg subclasses live in
journal_runtime_cfg_v2.py (never imported by CPU tests).

Vessel numbers have ONE source (CONTEXT_R3 decision 1): D5's registry
vehicles_v2.py + vehicles_v2_data.py (tasks/_shared/, owned by the vessel
task; this module only reads it). Every VesselRecord below is COMPUTED at
import from vehicles_v2.get_vehicle_v2(name); no vessel number is copied into
this file. Changing the registry changes the records, every resolved task
parameter, every layout hash and GEOMETRY_VERSION_DIGEST (a new geometry
version, never a silent edit). Derivation (derive_vessel_record):
  * collision polygon and capsule: registry ``collision`` (COM-relative, bow
    +x; every v2 asset has its actor origin at the COM, vehicles_v2.py:8-10);
  * L, B, inscribed / circumscribed radius about the COM: from that polygon;
  * mass; stop-distance bound u*tau + u^2/(2a), a = total delivered reverse
    thrust / mass, tau = the registry's nominal motor lag (largest motor);
  * heave validity band from the registry draft / freeboard;
  * top speed = registry ``speeds.top_model_m_s`` (terminal speed of the
    profile's own laws, i.e. what the simulated vessel can reach).
Task-contract parameters owned HERE (CONTEXT_R3 decision 14; one place,
configurable, labelled in PENDING_DECISIONS):
  * cruise (reference) speed: registry ``speeds.cruise_m_s`` (1.0 m/s for
    BOTH vessels; owner-decided, CONTEXT_R6 section 3) unless
    CRUISE_SPEED_OVERRIDES_M_S names another value;
  * design turning radius R = L (UNVERIFIED; D6 measures it; still open);
  * tilt (capsize) limit 60 deg.
Default WAM-V profile: the registry's DEFAULT_WAMV_PROFILE
(``wamv_vrx_v2_opt1``, competition 283.4 kg; CONTEXT_R3 decision 5, decided by
the owner 2026-10-01 00:0x, item A4; the WAM-V model audit G27 is done, controller decisions J1-J3).
Every registered v2 profile is supported (REGISTERED_VESSELS, vessel_record,
plan_class); the 24 frozen plans use BlueBoat and the default WAM-V.

Scaling rule (one rule for every task): reference lengths below are the
BlueBoat values (length scale s = 1, continuity with the legacy benchmark in
metres); every other vessel multiplies LENGTHS by s = L/L_REF (L = COM-polygon
length, L_REF = the registry BlueBoat polygon length), multiplies SPEED
tolerances by its cruise-speed ratio, and keeps angles, dwell/hold times and
counts unchanged. G28 (NC 39 section 2 item 1, owner 2026-10-04: fix the
document, keep the code): both journal vessels cruise at 1.0 m/s, so the ratio
is exactly 1 and the docking / harbour speed tolerance is 0.3 m/s for BOTH
vessels (``speed_tolerance_m_s = 0.3 * speed_ratio``); D4 TASK_CONTRACTS_v2.md
:107-108 ("x2 for WAM-V", written when the WAM-V cruise was 2 m/s) is
superseded. A Froude-similar x2.1 would also have to scale the reference
speed; it was not adopted. Obstacle-passage gaps instead follow the shared tier
definition g = B + w (journal_geometry_v2.physical_tier) so the vessel's own
beam enters exactly. Turning radius and stopping distance are admission
constraints, not scale factors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import importlib.util
import json
import math
import os
import sys
from types import MappingProxyType
from typing import ClassVar

CONTRACT_VERSION = 2
PHYSICS_VERSION = 2
CONTROL_DT_S = 1.0 / 60.0          # policy step: sim dt 1/120 s x decimation 2
PHYSICS_DT_S = 1.0 / 120.0
ROUTE_TIME_FACTOR = 2.0            # declared slack on nominal route time
TASKS = ("reach", "path_following", "station_keeping", "docking", "path_hazard", "scatter",
         "ring", "fortress", "forced_crossing", "iceberg", "moving_basin", "harbour_mission")

# Observation boundary (owner decision 2026-09-30): no controller family
# observes waves, current or wind. Journal cfgs must switch these off and the
# runtime guard must reject any of them being enabled.
FORBIDDEN_OBSERVATION_CHANNELS = ("sea_state", "wave", "current", "wind")
FORBIDDEN_CFG_FLAGS_TRUE = ("obs_sea_state",)   # station_keeping_env_cfg.py:178 (J-base)
REQUIRED_ENV_OVERRIDES_ABSENT = ("OBS_MODE", "OBS_DIM", "OBS_EXTENDED", "GOAL_RADIUS_OVERRIDE",
                                 "ACTION_DELAY", "WAVE_MODE")

# Every task carries the analytic 36-ray range sensor (CONTEXT_R3 decision 14),
# so no task is "unsensed" for dynamic hazards. Reference range 30 m is the
# existing HazardNav / PathHazard / HarborMission ray fan (D7 visibility
# SensorSpec default); like every task length it scales with s (E5 decision).
JOURNAL_RAY_SENSOR_ALL_TASKS = True
RAY_SENSOR_RAY_COUNT = 36
RAY_SENSOR_MAX_RANGE_REF_M = 30.0

# Task-contract vessel parameters (CONTEXT_R3 decision 14; PENDING owner decisions).
TILT_LIMIT_RAD = math.pi / 3.0                  # 60 deg capsize validity limit (declared)
TURNING_RADIUS_RULE = "overall_length"          # design R = L (D4 default; UNVERIFIED)
CRUISE_SPEED_OVERRIDES_M_S = MappingProxyType({})   # empty = registry speeds.cruise_m_s
BLUEBOAT_PROFILE = "blueboat_v2"

_MISSING = object()


def _load_geometry():
    here = os.path.dirname(os.path.abspath(__file__))
    name = "journal_geometry_v2_for_cfg"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, "journal_geometry_v2.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = _load_geometry()


def _path_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_registry():
    """D5's vessel registry, the ONE source of vessel numbers.

    Package import (runtime, ``tasks._shared.journal_cfg_v2``): the sibling
    module ``tasks._shared.vehicles_v2``. Path load (CPU tests, layouts): the
    same two files loaded by path under private names; vehicles_v2.py falls
    back to ``import vehicles_v2_data`` (vehicles_v2.py:56-59), which is
    served from a private copy and the previous sys.modules entry restored.
    """
    if __package__:
        from . import vehicles_v2
        return vehicles_v2
    name = "journal_cfg_v2_vehicles_v2"
    if name in sys.modules:
        return sys.modules[name]
    data = _path_module("journal_cfg_v2_vehicles_v2_data", "vehicles_v2_data.py")
    saved = sys.modules.get("vehicles_v2_data", _MISSING)
    sys.modules["vehicles_v2_data"] = data
    try:
        return _path_module(name, "vehicles_v2.py")
    finally:
        if saved is _MISSING:
            sys.modules.pop("vehicles_v2_data", None)
        else:
            sys.modules["vehicles_v2_data"] = saved


registry = _load_registry()


def _lf_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


# Provenance stamp of the registry source (LF-normalised file hashes).
REGISTRY_SOURCE_SHA256 = MappingProxyType({
    "vehicles_v2.py": _lf_sha256(registry.__file__),
    "vehicles_v2_data.py": _lf_sha256(registry._data.__file__)})
DEFAULT_WAMV_PROFILE = registry.DEFAULT_WAMV_PROFILE

PENDING_DECISIONS = MappingProxyType({
    "wamv_profile": "DECIDED (owner 2026-10-01 00:0x, A4; CONTEXT_R3 decision 5): default %s = registry "
                    "DEFAULT_WAMV_PROFILE; WAM-V model audit G27 done (controller decisions J1-J3)" % DEFAULT_WAMV_PROFILE,
    "cruise_speed": "DECIDED (owner; CONTEXT_R6 section 3: 1.0 m/s both boats; CONTEXT_R3 decision 14): registry "
                    "speeds.cruise_m_s (1.0 m/s both vessels) unless CRUISE_SPEED_OVERRIDES_M_S; overrides=%s"
                    % dict(CRUISE_SPEED_OVERRIDES_M_S),
    "design_turning_radius": "PENDING/UNVERIFIED (CONTEXT_R3 decision 14): R = L; D6 must measure it",
    "tilt_limit": "declared 60 deg capsize validity limit (D4 TASK_CONTRACTS_v2 section 2.8)",
    "ray_sensor": "36-ray fan in every task (CONTEXT_R3 decision 14); range 30 m x s (E5)",
})


def _vessel_class(name):
    if name.startswith("blueboat"):
        return "BlueBoat"
    if name.startswith("wamv_vrx"):
        return "WAMVVRX"
    raise ValueError("no journal vessel class for registry profile %r" % (name,))


def _positive_speed(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value <= 0.0:
        raise ValueError(name + " must be a finite positive speed")
    return float(value)


@dataclass(frozen=True)
class VesselRecord:
    """Contract view of ONE registry profile (every number derived, never copied).

    polygon_com_xy_m / capsule_*: registry ``collision`` (COM frame, bow +x).
    reference_length_m / reference_cruise_speed_m_s: the BlueBoat profile of
    the same registry (length scale 1, speed ratio 1).
    """

    key: str
    vessel_class: str
    polygon_com_xy_m: tuple
    capsule_center_com_xy_m: tuple
    capsule_segment_half_length_m: float
    capsule_radius_m: float
    mass_kg: float
    cruise_speed_m_s: float
    top_speed_m_s: float
    reverse_thrust_total_n: float
    motor_time_constant_s: float
    draft_m: float
    freeboard_m: float
    com_height_above_waterline_m: float
    reference_length_m: float
    reference_cruise_speed_m_s: float
    cruise_speed_status: str
    source: str
    turning_radius_design_m: float | None = None
    tilt_limit_rad: float = TILT_LIMIT_RAD

    # frame bookkeeping kept for D4 callers: the v2 actor origin IS the COM
    @property
    def body_polygon_m(self):
        return self.polygon_com_xy_m

    @property
    def com_body_xy_m(self):
        return (0.0, 0.0)

    @property
    def footprint(self):
        return geometry.PolygonFootprint(self.polygon_com_xy_m)

    @property
    def capsule(self):
        return geometry.CapsuleFootprint(self.capsule_center_com_xy_m, self.capsule_segment_half_length_m,
                                         self.capsule_radius_m)

    @property
    def length_m(self):
        x0, x1, _, _ = self.footprint.bounds
        return x1 - x0

    @property
    def beam_m(self):
        _, _, y0, y1 = self.footprint.bounds
        return y1 - y0

    @property
    def inscribed_radius_m(self):
        return self.footprint.inscribed_radius_m

    @property
    def circumscribed_radius_m(self):
        return self.footprint.radius_m

    @property
    def length_scale(self):
        return self.length_m / self.reference_length_m

    @property
    def stop_distance_m(self):
        """u*tau + u^2/(2a) with a = total reverse thrust / mass. Drag is
        ignored (it only helps), so this bounds the model stopping distance
        from cruise; the physical value is UNVERIFIED (D6 braking test)."""
        a = self.reverse_thrust_total_n / self.mass_kg
        u = self.cruise_speed_m_s
        return u * self.motor_time_constant_s + u * u / (2.0 * a)

    @property
    def turning_radius_m(self):
        """Declared design turning radius (default: overall length). UNVERIFIED;
        D6 must measure the steady turning radius at cruise and confirm it does
        not exceed this value, else the geometry version is regenerated."""
        return self.length_m if self.turning_radius_design_m is None else self.turning_radius_design_m

    @property
    def hull_depth_m(self):
        return self.draft_m + self.freeboard_m

    @property
    def heave_limits_m(self):
        """Validity band for heave = COM height above the local surface minus
        its calm equilibrium: sinking below -(freeboard + hull depth), airborne
        above draft + hull depth (one full hull depth beyond full immersion /
        full emergence). Engineering exclusion, not a calibrated prediction."""
        h = self.hull_depth_m
        return (-(self.freeboard_m + h), self.draft_m + h)

    @property
    def geometry(self):
        return geometry.VesselGeometry(self.key, self.footprint, self.turning_radius_m,
                                       self.cruise_speed_m_s, self.stop_distance_m,
                                       capsule_footprint=self.capsule,
                                       reference_length_m=self.reference_length_m)

    def derived_numbers(self):
        """Every number the contract derives from the registry (one mapping)."""
        lo, hi = self.heave_limits_m
        return dict(length_m=self.length_m, beam_m=self.beam_m, inscribed_radius_m=self.inscribed_radius_m,
                    circumscribed_radius_m=self.circumscribed_radius_m, length_scale=self.length_scale,
                    mass_kg=self.mass_kg, stop_distance_m=self.stop_distance_m,
                    turning_radius_m=self.turning_radius_m, tilt_limit_rad=self.tilt_limit_rad,
                    min_heave_m=lo, max_heave_m=hi, cruise_speed_m_s=self.cruise_speed_m_s,
                    top_speed_m_s=self.top_speed_m_s,
                    com_height_above_waterline_m=self.com_height_above_waterline_m)

    @property
    def digest(self):
        """SHA-256 over every contract-relevant field and derived number."""
        values = {f.name: getattr(self, f.name) for f in fields(self)
                  if f.name not in ("source", "cruise_speed_status")}
        values.update(("derived_" + k, v) for k, v in self.derived_numbers().items())
        return geometry.geometry_digest(record_schema="journal-vessel-record-v1", **values)


def derive_vessel_record(spec, *, reference_spec, cruise_speed_m_s=None, reference_cruise_speed_m_s=None):
    """Contract record of one registry VesselSpecV2 (pure; no copies).

    reference_spec is the BlueBoat profile of the SAME registry (length scale
    and speed-ratio reference). cruise_speed_m_s overrides the registry cruise
    (PENDING contract parameter); it must stay below the modelled top speed.
    """
    for s in (spec, reference_spec):
        if tuple(s.com_actor_m) != (0.0, 0.0, 0.0) or s.bow_body_axis != "+x":
            raise ValueError("journal footprints require v2 assets with the origin at the COM, bow +x")
    top = _positive_speed(spec.speeds.top_model_m_s, "top_model_m_s")
    if cruise_speed_m_s is None:
        cruise = _positive_speed(spec.speeds.cruise_m_s, "cruise_m_s")
        status = "registry speeds.cruise_m_s (PENDING contract default, CONTEXT_R3 decision 14)"
    else:
        cruise = _positive_speed(cruise_speed_m_s, "cruise override")
        status = "contract override %.6g m/s (PENDING owner decision)" % cruise
    if not cruise < top:
        raise ValueError("cruise speed must be below the profile's modelled top speed %.4f m/s" % top)
    ref_poly = tuple(tuple(map(float, p)) for p in reference_spec.collision.polygon_com_xy_m)
    ref_len = max(p[0] for p in ref_poly) - min(p[0] for p in ref_poly)
    ref_cruise = (_positive_speed(reference_spec.speeds.cruise_m_s, "reference cruise")
                  if reference_cruise_speed_m_s is None
                  else _positive_speed(reference_cruise_speed_m_s, "reference cruise"))
    motors = tuple(spec.propulsion.motors)
    col = spec.collision
    return VesselRecord(
        key=spec.name, vessel_class=_vessel_class(spec.name),
        polygon_com_xy_m=tuple(tuple(map(float, p)) for p in col.polygon_com_xy_m),
        capsule_center_com_xy_m=tuple(map(float, col.capsule_center_com_xy_m)),
        capsule_segment_half_length_m=float(col.capsule_segment_half_length_m),
        capsule_radius_m=float(col.capsule_radius_m),
        mass_kg=float(spec.mass_kg), cruise_speed_m_s=cruise, top_speed_m_s=top,
        reverse_thrust_total_n=float(sum(m.reverse_limit_n for m in motors)),
        motor_time_constant_s=float(max(m.time_constant_s for m in motors)),
        draft_m=float(spec.hydrostatics.draft_m), freeboard_m=float(spec.hydrostatics.freeboard_m),
        com_height_above_waterline_m=float(spec.hydrostatics.com_height_above_water_m),
        reference_length_m=float(ref_len), reference_cruise_speed_m_s=ref_cruise,
        cruise_speed_status=status,
        source=("vehicles_v2.get_vehicle_v2(%r): collision.polygon_com_xy_m / capsule_*, mass_kg, "
                "speeds.cruise_m_s / top_model_m_s, sum(motor.reverse_limit_n), max(motor.time_constant_s), "
                "hydrostatics.draft_m / freeboard_m / com_height_above_water_m; reference %r"
                % (spec.name, reference_spec.name)))


def build_vessel_records(reg, cruise_overrides=None):
    """Records for EVERY registered profile of registry module ``reg``."""
    if getattr(reg, "PINS_STATUS", None) != "pinned":
        raise ValueError("journal vessel records need a registry whose computed values are pinned "
                         "(vehicles_v2.PINS_STATUS == 'pinned'; G7 item A)")
    overrides = dict(CRUISE_SPEED_OVERRIDES_M_S if cruise_overrides is None else cruise_overrides)
    unknown = set(overrides) - set(reg.VEHICLES_V2)
    if unknown:
        raise ValueError("cruise overrides for unregistered profiles: %s" % sorted(unknown))
    ref = reg.get_vehicle_v2(BLUEBOAT_PROFILE)
    ref_cruise = overrides.get(BLUEBOAT_PROFILE)
    ref_cruise = ref.speeds.cruise_m_s if ref_cruise is None else ref_cruise
    return MappingProxyType({
        name: derive_vessel_record(reg.get_vehicle_v2(name), reference_spec=ref,
                                   cruise_speed_m_s=overrides.get(name),
                                   reference_cruise_speed_m_s=ref_cruise)
        for name in sorted(reg.VEHICLES_V2)})


REGISTERED_VESSELS = build_vessel_records(registry)
BLUEBOAT = REGISTERED_VESSELS[BLUEBOAT_PROFILE]
WAMV = REGISTERED_VESSELS[DEFAULT_WAMV_PROFILE]
# The two contract vessels of the 24 frozen plans (BlueBoat, default WAM-V).
VESSELS = MappingProxyType({BLUEBOAT.key: BLUEBOAT, WAMV.key: WAMV})


def vessel_record(vessel_key, cruise_speed_m_s=None):
    """Record of any registered profile; cruise_speed_m_s is an ANALYSIS
    override (e.g. the cruise trade-off survey), never the contract default."""
    if vessel_key not in REGISTERED_VESSELS:
        raise ValueError("unknown journal vessel %r; registered v2 profiles: %s"
                         % (vessel_key, ", ".join(sorted(REGISTERED_VESSELS))))
    record = REGISTERED_VESSELS[vessel_key]
    if cruise_speed_m_s is None:
        return record
    cruise = _positive_speed(cruise_speed_m_s, "cruise override")
    if cruise == record.cruise_speed_m_s:
        return record
    if not cruise < record.top_speed_m_s:
        raise ValueError("cruise speed must be below the profile's modelled top speed")
    return replace(record, cruise_speed_m_s=cruise,
                   cruise_speed_status="analysis override %.6g m/s (not the contract default)" % cruise)


# --------------------------------------------------------------------------
# task table (BlueBoat reference values; see module docstring for scaling)
# --------------------------------------------------------------------------

TASK_TABLE = MappingProxyType({
    "reach": dict(carrier="hazard_nav", dwell_s=0.0, success="goal_disk", terminates_on_success=True,
                  count_mode="exact", budget_factor=1.10),
    "path_following": dict(carrier="path_following", dwell_s=0.0, success="ordered_gates",
                           terminates_on_success=True, count_mode="exact", budget_factor=1.10),
    "station_keeping": dict(carrier="station_keeping", dwell_s=80.0, success="late_interval_hold",
                            terminates_on_success=False, count_mode="exact", budget_factor=1.10),
    "docking": dict(carrier="docking", dwell_s=5.0, success="contained_directional_dock",
                    terminates_on_success=True, count_mode="recorded", budget_factor=1.40),
    "path_hazard": dict(carrier="path_hazard", dwell_s=0.0, success="ordered_gates",
                        terminates_on_success=True, count_mode="exact", budget_factor=1.60),
    "scatter": dict(carrier="hazard_nav", dwell_s=0.0, success="goal_disk", terminates_on_success=True,
                    count_mode="exact", budget_factor=1.60),
    "ring": dict(carrier="hazard_nav", dwell_s=0.0, success="goal_disk", terminates_on_success=True,
                 count_mode="recorded", budget_factor=1.60),
    "fortress": dict(carrier="hazard_nav", dwell_s=0.0, success="goal_disk", terminates_on_success=True,
                     count_mode="recorded", budget_factor=2.60),
    "forced_crossing": dict(carrier="hazard_nav", dwell_s=0.0, success="goal_disk",
                            terminates_on_success=True, count_mode="recorded", budget_factor=1.60),
    "iceberg": dict(carrier="hazard_nav", dwell_s=0.0, success="goal_disk", terminates_on_success=True,
                    count_mode="exact", budget_factor=1.80),
    "moving_basin": dict(carrier="hazard_nav", dwell_s=0.0, success="goal_disk",
                         terminates_on_success=True, count_mode="recorded", budget_factor=1.40),
    "harbour_mission": dict(carrier="harbor_mission", dwell_s=5.0, success="ordered_stages_then_dock",
                            terminates_on_success=True, count_mode="exact", budget_factor=1.60),
})

# Static obstacle-slot capacity per task (all tiers, both vessels), with
# headroom over the realized maxima observed in the CPU sweep; the test suite
# asserts sampled layouts never exceed it. D7 adds one reserved intruder slot
# for every task except moving basin, whose native movers ARE the dynamic axis.
STATIC_CAPACITY = MappingProxyType({
    "reach": 0, "path_following": 0, "station_keeping": 0, "docking": 64, "path_hazard": 14,
    "scatter": 14, "ring": 24, "fortress": 72, "forced_crossing": 80, "iceberg": 3,
    "moving_basin": 128, "harbour_mission": 80})
D7_RESERVED_SLOTS = MappingProxyType({task: (0 if task == "moving_basin" else 1) for task in TASKS})

FIELD_COUNTS = (4, 8, 12, 14)          # scatter, path hazard, harbour transit field
BLOCKER_COUNTS = (1, 2, 3, 4)          # path hazard designated line blockers
ICEBERG_COUNTS = (1, 2, 3, 3)          # legacy ICEBERG_TIERS counts (hazard_geometry.py:712-717)
ICEBERG_RADII = ((5.0, 8.0), (6.0, 10.0), (7.0, 12.0), (9.0, 14.0))
MOVING_DOSE_MU = (0.0, 0.04, 0.08, 0.16, 0.30)   # C7 STRATEGY.md:227; D0 static, D2 native


def _resolve(task, tier_level, record):
    """All physical parameters for one (task, tier, vessel record)."""
    if task not in TASKS:
        raise ValueError("unknown journal task %r" % (task,))
    vessel = record.geometry
    tier = geometry.physical_tier(tier_level, vessel)
    k = tier.level
    s = vessel.length_scale
    speed_ratio = record.cruise_speed_m_s / record.reference_cruise_speed_m_s
    p = dict(task=task, vessel=record.key, tier=k, length_scale=s,
             physical_surface_gap_m=tier.physical_surface_gap_m,
             hull_centre_width_m=tier.hull_centre_width_m,
             planning_margin_m=tier.planning_margin_m,
             planned_centre_width_m=tier.planned_centre_width_m,
             centred_heading_tolerance_rad=tier.centred_heading_tolerance_rad,
             goal_radius_m=2.0 * s, region_padding_m=10.0 * s, grid_cell_m=0.25 * s,
             dwell_s=TASK_TABLE[task]["dwell_s"], count_mode=TASK_TABLE[task]["count_mode"],
             cruise_speed_m_s=record.cruise_speed_m_s,
             ray_sensor_ray_count=RAY_SENSOR_RAY_COUNT,
             ray_sensor_max_range_m=RAY_SENSOR_MAX_RANGE_REF_M * s)
    if task == "reach":
        d = (10.0, 15.0, 20.0, 30.0)[k] * s
        p.update(straight_distance_m=d, reference_length_m=d, requested_count=0)
    elif task in ("path_following", "path_hazard"):
        turn = (30.0, 45.0, 60.0, 75.0)[k] if task == "path_following" else 45.0
        p.update(segment_count=4, segment_length_m=16.0 * s, heading_change_deg=turn,
                 gate_radius_m=2.0 * s, reference_length_m=64.0 * s)
        if task == "path_following":
            p.update(requested_count=0)
        else:
            p.update(requested_count=FIELD_COUNTS[k], blocker_count=BLOCKER_COUNTS[k],
                     radius_range_m=(0.8 * s, 1.5 * s), route_band_m=6.0 * s,
                     blocker_offset_max_m=0.5 * s, blocker_fraction=(0.30, 0.70),
                     gate_clear_radius_m=3.0 * s, min_pair_surface_gap_m=tier.physical_surface_gap_m)
    elif task == "station_keeping":
        # settle_s = 20 s (G28 rationale, NC 39 section 2 item 2; value unchanged): a DECLARED allowance (D4
        # TASK_CONTRACTS_v2 section 2.7, class-2 item B18), not fitted and not derived from a model; dwell_s 80 =
        # settle 20 + scored hold 60, so the scored window [T - 60 s, T] starts 2 L_nom / u + 20 s after the reset,
        # i.e. L_nom / u (the route slack) + 20 s after the nominal arrival. Post-hoc scale check (does not set the
        # value): the time to cross one hold radius at the reference speed, r / u, is 1-3 s (BlueBoat) and
        # 4.5-13.4 s (WAM-V; 6.7 s at the main tier 2); the registry stop distances are 0.22 m / 0.71 m (< 1.5 s at
        # 1 m/s); the slowest settling mode reported for the WAM-V is the heave transient of about 3 s (G18 REPORT
        # Q1). So 20 s >= 1.5 x the largest radius transit (WAM-V tier 0) and about 3 x the main-tier WAM-V one;
        # time quantities are not length-scaled (module docstring), which is why the WAM-V is the binding case.
        # Changing it would re-run every station-keeping cell.
        p.update(hold_radius_m=(3.0, 2.0, 1.5, 1.0)[k] * s, approach_distance_m=10.0 * s,
                 hold_s=60.0, settle_s=20.0, final_window_s=60.0, reference_length_m=10.0 * s,
                 requested_count=0)
    elif task in ("docking", "harbour_mission"):
        berth = dict(slip_free_width_m=tier.physical_surface_gap_m, slip_opening_x_m=-7.0 * s,
                     slip_back_face_x_m=vessel.forward_extent_m + 0.25 * s,
                     wall_radius_m=0.35 * s, seal_overlap_m=0.05 * s,
                     position_tolerance_m=2.5 * s, heading_tolerance_rad=math.radians(15.0),
                     speed_tolerance_m_s=0.3 * speed_ratio, hold_s=5.0)
        p.update(berth)
        if task == "docking":
            p.update(approach_distance_m=12.0 * s, lateral_offset_max_m=3.0 * s,
                     heading_offset_max_rad=math.radians(45.0),
                     reference_length_m=(12.0 + 7.0) * s)
        else:
            p.update(departure_gate_x_m=(5.0 * s, 15.0 * s), virtual_gate_width_m=6.0 * s,
                     field_start_x_m=20.0 * s, field_length_m=30.0 * s,
                     requested_count=FIELD_COUNTS[k], radius_range_m=(0.8 * s, 1.8 * s),
                     field_half_width_m=12.0 * s, endpoint_clear_m=5.0 * s,
                     min_pair_surface_gap_m=tier.physical_surface_gap_m,
                     berth_offset_m=15.0 * s, gate_approach_m=1.0 * s,
                     gate_min_normal_speed_m_s=0.2 * speed_ratio,
                     reference_length_m=(20.0 + 30.0 + 15.0 + 7.0) * s)
    elif task == "scatter":
        p.update(straight_distance_m=30.0 * s, requested_count=FIELD_COUNTS[k],
                 radius_range_m=(0.8 * s, 2.0 * s), endpoint_clear_m=5.0 * s,
                 corridor_half_width_m=12.0 * s, blocker_fraction=(0.30, 0.70),
                 min_pair_surface_gap_m=tier.physical_surface_gap_m, reference_length_m=30.0 * s)
    elif task == "ring":
        p.update(ring_radius_m=10.5 * s, obstacle_radius_m=1.75 * s, goal_distance_m=30.0 * s,
                 goal_bearing_offset_max_rad=math.radians(60.0), seal_overlap_m=0.05 * s,
                 reference_length_m=30.0 * s)
    elif task == "fortress":
        r_obs = 1.2 * s
        inner = 9.0 * s
        outer = inner + 2.0 * r_obs + 6.0 * s
        p.update(inner_ring_radius_m=inner, outer_ring_radius_m=outer, obstacle_radius_m=r_obs,
                 band_surface_separation_m=6.0 * s, gate_stagger_rad=(math.pi / 2.0, math.pi),
                 start_distance_m=outer + r_obs + 8.0 * s,
                 start_bearing_offset_max_rad=math.radians(45.0), seal_overlap_m=0.05 * s,
                 reference_length_m=outer + r_obs + 8.0 * s)
    elif task == "forced_crossing":
        p.update(goal_distance_m=27.0 * s, basin_half_width_m=7.5 * s, end_margin_m=4.8 * s,
                 perimeter_radius_m=1.2 * s, bulkhead_radius_m=0.8 * s,
                 wall_surface_overlap_m=0.25 * s, wall_x_jitter_m=3.0 * s,
                 bulkhead_min_side_m=2.0 * s, endpoint_clear_m=2.5 * s, seal_overlap_m=0.05 * s,
                 reference_length_m=27.0 * s)
    elif task == "iceberg":
        lo, hi = ICEBERG_RADII[k]
        p.update(requested_count=ICEBERG_COUNTS[k], radius_range_m=(lo * s, hi * s),
                 min_edge_gap_m=max(8.0 * s, tier.physical_surface_gap_m),
                 straight_distance_m=34.0 * s, endpoint_clear_m=5.0 * s,
                 reference_length_m=34.0 * s)
    elif task == "moving_basin":
        approach, sep, buffer_, orbit, runout = 6.0 * s, 8.0 * s, 10.0 * s, 6.0 * s, 8.0 * s
        goal_x = approach + sep + buffer_ + 2.0 * orbit + runout
        p.update(row_gap_m=tier.physical_surface_gap_m, row_radius_m=0.8 * s,
                 approach_m=approach, row_separation_m=sep, buffer_m=buffer_, orbit_radius_m=orbit,
                 runout_m=runout, basin_half_width_m=10.5 * s, end_margin_m=4.8 * s,
                 perimeter_radius_m=1.2 * s, wall_surface_overlap_m=0.25 * s,
                 endpoint_clear_m=2.5 * s, top_speed_m_s=record.top_speed_m_s,
                 goal_distance_m=goal_x, reference_length_m=goal_x,
                 dose_mu_ladder=MOVING_DOSE_MU, admission_samples_per_period=12)
    # Budget = detour factor x reference length + a start-alignment allowance
    # of pi*R (the witness charges a start pivot of up to pi as R*angle).
    p["turning_radius_m"] = record.turning_radius_m
    p["route_budget_m"] = (TASK_TABLE[task]["budget_factor"] * p["reference_length_m"]
                           + math.pi * record.turning_radius_m)
    p["vessel_digest"] = record.digest
    return MappingProxyType(p)


def resolve_task_geometry(task, tier_level, vessel_key, *, cruise_speed_m_s=None):
    """All physical parameters for one (task, tier, vessel), frozen mapping.

    Reference numbers are the legacy BlueBoat metre values (hazard_geometry.py
    :24-27, :712-719, :840-866, :1908-1918, :2813-2829; path_hazard_geometry.py
    :18-37; berth_geometry.py:25-28 and docking_env_cfg.py:161-166,:218-219;
    harbor_geometry.py:42-54; station_keeping_env_cfg.py:196-199) except where
    the journal contract deliberately changes them (see TASK_CONTRACTS_v2.md).
    vessel_key is any registered v2 profile; cruise_speed_m_s is an analysis
    override (vessel_record).
    """
    if task not in TASKS:
        raise ValueError("unknown journal task %r" % (task,))
    return _resolve(task, tier_level, vessel_record(vessel_key, cruise_speed_m_s))


def deadline_steps(nominal_route_length_m, cruise_speed_m_s, dwell_s):
    """T = 2 * L_nom / u + dwell, rounded UP once to a whole control step."""
    for value, name in ((nominal_route_length_m, "L_nom"), (cruise_speed_m_s, "cruise")):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError(name + " must be finite and positive")
    if isinstance(dwell_s, bool) or not math.isfinite(dwell_s) or dwell_s < 0.0:
        raise ValueError("dwell must be finite and nonnegative")
    seconds = ROUTE_TIME_FACTOR * nominal_route_length_m / cruise_speed_m_s + dwell_s
    return int(math.ceil(seconds / CONTROL_DT_S - 1.0e-9))


def deadline_s(nominal_route_length_m, cruise_speed_m_s, dwell_s):
    return deadline_steps(nominal_route_length_m, cruise_speed_m_s, dwell_s) * CONTROL_DT_S


# --------------------------------------------------------------------------
# geometry version digest (changes whenever the registry or the contract does)
# --------------------------------------------------------------------------

def _jsonable(value):
    """Canonical identity form (G7 item A): every float becomes its declared
    quantisation text "f:<quanta>" (journal_geometry_v2.canonical_identity_value,
    IDENTITY_QUANTUM 1e-6 SI units), never the exact repr of its last ulp."""
    return geometry.canonical_identity_value(value)


def geometry_version_digest(records=None):
    """SHA-256 over the contract constants, every contract vessel record and
    every resolved (task, tier, vessel) parameter set. ``records`` defaults to
    the two contract vessels; pass records built from another registry
    (build_vessel_records) to see that a registry change is a new version."""
    records = VESSELS if records is None else records
    resolved = {"%s|%d|%s" % (task, tier, key): _jsonable(_resolve(task, tier, rec))
                for task in TASKS for tier in geometry.TIER_LEVELS for key, rec in records.items()}
    return geometry.geometry_digest(
        schema="journal-geometry-version-v1", contract_version=CONTRACT_VERSION,
        identity_canon=geometry.IDENTITY_CANON_VERSION,
        physics_version=PHYSICS_VERSION, control_dt_s=CONTROL_DT_S, route_time_factor=ROUTE_TIME_FACTOR,
        tier_centre_width_ref_m=list(geometry.TIER_CENTRE_WIDTH_REF_M),
        planning_margin_ref_m=geometry.PLANNING_MARGIN_REF_M,
        task_table=_jsonable(TASK_TABLE), static_capacity=_jsonable(STATIC_CAPACITY),
        d7_reserved_slots=_jsonable(D7_RESERVED_SLOTS), field_counts=list(FIELD_COUNTS),
        blocker_counts=list(BLOCKER_COUNTS), iceberg_counts=list(ICEBERG_COUNTS),
        iceberg_radii=_jsonable(ICEBERG_RADII), moving_dose_mu=_jsonable(MOVING_DOSE_MU),
        vessel_records={key: rec.digest for key, rec in records.items()},
        resolved=json.dumps(resolved, sort_keys=True))


GEOMETRY_VERSION_DIGEST = geometry_version_digest()


@dataclass(frozen=True)
class JournalTaskPlan:
    """Immutable experiment specification for one (task, vessel, tier).

    Condition doses are NOT inputs: geometry and deadlines are frozen before
    any dose is drawn and reused at every dose and for every controller.
    """

    task: ClassVar[str] = ""
    vessel: ClassVar[str] = ""
    contract_version: ClassVar[int] = CONTRACT_VERSION
    physics_version: ClassVar[int] = PHYSICS_VERSION
    geometry_tier: int = 2

    def __post_init__(self):
        if self.task not in TASKS or self.vessel not in REGISTERED_VESSELS:
            raise ValueError("use one of the named task/vessel plan classes (plan_class)")
        if isinstance(self.geometry_tier, bool) or not isinstance(self.geometry_tier, int) \
                or self.geometry_tier not in geometry.TIER_LEVELS:
            raise ValueError("geometry_tier must be the integer 0, 1, 2 or 3")

    @property
    def record(self):
        return vessel_record(self.vessel)

    @property
    def params(self):
        return resolve_task_geometry(self.task, self.geometry_tier, self.vessel)

    @property
    def route_budget_m(self):
        return self.params["route_budget_m"]

    @property
    def max_deadline_s(self):
        """Episode cap (cfg.episode_length_s); per-scenario T is <= this."""
        return deadline_s(self.route_budget_m, self.record.cruise_speed_m_s, self.params["dwell_s"])

    def scenario_deadline_s(self, nominal_route_length_m):
        if nominal_route_length_m > self.route_budget_m + 1.0e-9:
            raise ValueError("admitted nominal route exceeds the frozen route budget")
        return deadline_s(nominal_route_length_m, self.record.cruise_speed_m_s, self.params["dwell_s"])

    @property
    def validity_limits(self):
        lo, hi = self.record.heave_limits_m
        return dict(capsize_angle_rad=self.record.tilt_limit_rad, min_heave_m=lo, max_heave_m=hi)

    @property
    def nominal_com_height_m(self):
        """RigidState.nominal_com_height_m: calm equilibrium COM height above
        the waterline (registry hydrostatics.com_height_above_water_m)."""
        return self.record.com_height_above_waterline_m

    def assert_vehicle_override(self, requested_vehicle):
        if requested_vehicle != self.vessel:
            raise ValueError("choose the vessel-specific journal cfg before construction; "
                             "late vehicle overrides are rejected")


def _plan_class(task, vessel_key, name):
    return dataclass(frozen=True)(type(name, (JournalTaskPlan,), {"task": task, "vessel": vessel_key,
                                                                    "__module__": __name__}))


_CAMEL = {"reach": "Reach", "path_following": "PathFollowing", "station_keeping": "StationKeeping",
          "docking": "Docking", "path_hazard": "PathHazard", "scatter": "Scatter", "ring": "Ring",
          "fortress": "Fortress", "forced_crossing": "ForcedCrossing", "iceberg": "Iceberg",
          "moving_basin": "MovingBasin", "harbour_mission": "HarbourMission"}


def _vessel_tag(vessel_key):
    """Class-name tag: BlueBoat, WAMV (default WAM-V profile), WAMV<Profile> otherwise."""
    if vessel_key == BLUEBOAT_PROFILE:
        return "BlueBoat"
    if vessel_key == DEFAULT_WAMV_PROFILE:
        return "WAMV"
    rest = vessel_key.split("wamv_vrx_v2", 1)[-1] if vessel_key.startswith("wamv_vrx_v2") else vessel_key
    return "WAMV" + "".join(part.capitalize() for part in rest.split("_") if part)


JOURNAL_PLAN_CLASSES = {}
for _task in TASKS:
    for _vessel in VESSELS:
        _name = "%s%sJournalPlan" % (_CAMEL[_task], _vessel_tag(_vessel))
        _cls = _plan_class(_task, _vessel, _name)
        globals()[_name] = _cls
        JOURNAL_PLAN_CLASSES[(_task, _vessel)] = _cls
JOURNAL_PLAN_CLASSES = MappingProxyType(JOURNAL_PLAN_CLASSES)
del _task, _vessel, _name, _cls
_EXTRA_PLAN_CLASSES = {}


def plan_class(task, vessel_key):
    """Plan class for any task x REGISTERED profile: the 24 frozen classes for
    the contract vessels, a cached class for other profiles (sensitivity runs
    such as ``wamv_vrx_v2_opt2``)."""
    key = (task, vessel_key)
    if key in JOURNAL_PLAN_CLASSES:
        return JOURNAL_PLAN_CLASSES[key]
    if task not in TASKS or vessel_key not in REGISTERED_VESSELS:
        raise ValueError("unknown journal task/vessel %r" % (key,))
    if key not in _EXTRA_PLAN_CLASSES:
        _EXTRA_PLAN_CLASSES[key] = _plan_class(task, vessel_key, "%s%sJournalPlan"
                                               % (_CAMEL[task], _vessel_tag(vessel_key)))
    return _EXTRA_PLAN_CLASSES[key]


# --------------------------------------------------------------------------
# journal gym ids (proposal; registration happens in part F, not here)
# --------------------------------------------------------------------------
# One scheme: Isaac-USV-{Task}-{BlueBoat|WAMVVRX}-Journal-Direct-v1 (C4/D1
# pattern). Task tokens follow the legacy family ids (Isaac-USV-PathFollow-,
# -StationKeep-, -Dock-, -PathHazard-, -HazardRing-, -HazardFortress-,
# -HazardCross-, -Iceberg-, -HazardMovingBasin-, -HarborMission-); the scatter
# task is "HazardScatter" (the HazardNav family carries seven journal tasks);
# reach is "Reach". The WAM-V profile is cfg data (PENDING), not part of the id.
JOURNAL_ID_TASK_TOKENS = MappingProxyType({
    "reach": "Reach", "path_following": "PathFollow", "station_keeping": "StationKeep",
    "docking": "Dock", "path_hazard": "PathHazard", "scatter": "HazardScatter", "ring": "HazardRing",
    "fortress": "HazardFortress", "forced_crossing": "HazardCross", "iceberg": "Iceberg",
    "moving_basin": "HazardMovingBasin", "harbour_mission": "HarborMission"})


def journal_env_id(task, vessel_key):
    if task not in TASKS or vessel_key not in VESSELS:
        raise ValueError("journal ids exist for the 12 tasks x the two contract vessels only")
    return "Isaac-USV-%s-%s-Journal-Direct-v1" % (JOURNAL_ID_TASK_TOKENS[task],
                                                  VESSELS[vessel_key].vessel_class)


JOURNAL_ENV_IDS = MappingProxyType({(t, v): journal_env_id(t, v) for t in TASKS for v in VESSELS})
