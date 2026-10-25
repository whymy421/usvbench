"""Condition-config schema 3: the hydro_v2 (journal) condition layer (E3).

Schema 2 (``config.py``) stays the legacy composition (physics v1 and C2
physics v2 without hydro_v2) and is unchanged. Schema 3 is selected by an env
cfg whose ``underwater_physics_cfg.hydrodynamics_model == "hydro_v2"``.
Every dose is a physical quantity at 1:1 scale; there are no force scales.

Semantics (see DESIGN_E3.md):
* waves: JONSWAP point dose (Hs, Tp, gamma), uniform spreading, per-
  realisation band [band_min, band_max] * fp (the upper edge is RAISED to the
  vessel's hydro_v2 footprint requirement when ``band_policy`` is
  ``raise_to_vessel``), component count raised so the group-envelope repeat
  period 1/df is at least ``recurrence_min_s``; consumed by HydroV2Runtime as
  a SpectralWaves water-velocity source (never ``SeaState.forces``).
* current: uniform water velocity (world XY, Z = 0) from a constant speed or
  a piecewise-linear table evaluated at the runtime tick clock; consumed as
  water velocity (never a drag wrench).
* wind: dose U10 (mean wind at 10 m) converted to the vessel's lateral
  centroid height with a neutral log profile (vessel roughness length), C3
  Fossen law with D5 windage, heel lever about the COM; OU gusts
  (``gust_intensity`` x the vessel-height mean speed, correlation time).
  Zero dose = module OFF.
* actuator: four live axes (CONTEXT_R3 decision 11): thrust cap, one-sided
  motor gain (side balanced by scenario ordinal pairs), additional first-order
  lag, resistance multiplier (surge/sway/yaw damping). Centred mass and
  eccentric payload are REJECTED (deferred).
* perception: C7 STRATEGY.md section 5 primary axes (position-estimate noise
  with correlation time, fixed-magnitude position bias with balanced
  direction, latency, one contiguous outage) plus the labelled sensor bundle
  (velocity/yaw-rate/range white noise).
* directions: ``direction_mode="strata"`` assigns the balanced compass-8
  strata (strata.py) relative to the task reference axis.
* F0 (CONTEXT_R4 D4): every directional suite shares the scenario's compass-8
  stratum; ``direction_offset_rad`` (waves, current, wind) and
  ``bias_direction_offset_rad`` (perception bias) add a declared relative
  offset to that shared direction (e.g. wind 30 deg off the waves). The
  offset lies in (-pi, pi], is stamped with the parameters, and is REJECTED
  on a disabled suite/axis or with ``direction_mode="fixed"`` (two values for
  one direction).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping

SCHEMA_VERSION_V3 = 3
PRESET_REGISTRY_VERSION_V3 = 1
HYDRODYNAMICS_MODEL = "hydro_v2"
DIRECTION_MODES = ("strata", "fixed")
BAND_POLICIES = ("raise_to_vessel", "fixed")
STEEPNESS_RULES = ("dnv_rp_c205_3_5_4", "generator_1_7")
WIND_REFERENCES = ("u10", "vessel_height")
WIND_RNG_PROTOCOL = "usvbench.wind.ou.v2"
PERCEPTION_PROTOCOL = "usvbench.perception.keyed.v1"
DNV_SP_LIMIT = 1.0 / 15.0
DNV_SS_LIMIT = 1.0 / 10.0
GENERATOR_LIMIT = 1.0 / 7.0
GRAVITY = 9.81


@dataclass(frozen=True)
class WavesV3:
    enabled: bool = False
    hs_m: float = 0.0
    tp_s: float = 4.0
    gamma: float = 3.3
    spreading_rad: float = math.pi / 4.0
    band_min_over_fp: float = 0.2
    band_max_over_fp: float = 4.0
    band_policy: str = "raise_to_vessel"
    n_components: int = 1024
    recurrence_min_s: float = 480.0
    direction_mode: str = "strata"
    direction_rad: float = 0.0
    steepness_rule: str = "dnv_rp_c205_3_5_4"
    direction_offset_rad: float = 0.0


@dataclass(frozen=True)
class CurrentV3:
    enabled: bool = False
    speed_mps: float = 0.0
    speed_table: tuple = ()
    direction_mode: str = "strata"
    direction_rad: float = 0.0
    direction_offset_rad: float = 0.0


@dataclass(frozen=True)
class WindV3:
    enabled: bool = False
    u10_mps: float = 0.0
    reference: str = "u10"
    gust_intensity: float = 0.15
    gust_correlation_time_s: float = 2.0
    direction_mode: str = "strata"
    direction_rad: float = 0.0
    rng_protocol: str = WIND_RNG_PROTOCOL
    direction_offset_rad: float = 0.0


@dataclass(frozen=True)
class ActuatorV3:
    enabled: bool = False
    thrust_cap: float = 1.0
    motor_gain_one_sided: float = 1.0
    additional_lag_s: float = 0.0
    resistance_scale: float = 1.0


@dataclass(frozen=True)
class PerceptionV3:
    enabled: bool = False
    pos_noise_sigma_m: float = 0.0
    pos_noise_correlation_s: float = 0.2
    pos_bias_m: float = 0.0
    bias_direction_mode: str = "strata"
    latency_s: float = 0.0
    outage_s: float = 0.0
    outage_window_s: tuple = (5.0, 20.0)   # legacy/direct-use fallback only; journal tasks: G28 route rule
    vel_noise_sigma_mps: float = 0.0
    yaw_rate_noise_sigma_rad_s: float = 0.0
    ray_noise_sigma_m: float = 0.0
    protocol: str = PERCEPTION_PROTOCOL
    bias_direction_offset_rad: float = 0.0


@dataclass(frozen=True)
class JournalConditionConfig:
    waves: WavesV3 = field(default_factory=WavesV3)
    current: CurrentV3 = field(default_factory=CurrentV3)
    wind: WindV3 = field(default_factory=WindV3)
    actuator: ActuatorV3 = field(default_factory=ActuatorV3)
    perception: PerceptionV3 = field(default_factory=PerceptionV3)
    seed: int = 0
    presets: tuple = ()
    preset_version: int = PRESET_REGISTRY_VERSION_V3
    direction_strata: str = "compass8_v1"
    hydrodynamics_model: str = HYDRODYNAMICS_MODEL
    schema_version: int = SCHEMA_VERSION_V3

    @property
    def any_enabled(self):
        return any(getattr(self, name).enabled for name in SUITES_V3)


SUITES_V3 = {"waves": WavesV3, "current": CurrentV3, "wind": WindV3,
             "actuator": ActuatorV3, "perception": PerceptionV3}

# Schema-2 keys that have no meaning under hydro_v2, with the reason. Any of
# them in a hydro_v2 configuration is an error (never silently ignored).
LEGACY_KEYS = {
    "waves": {
        "hs_range": "schema 3 uses point doses hs_m/tp_s/gamma",
        "tp_range": "schema 3 uses point doses hs_m/tp_s/gamma",
        "gamma_range": "schema 3 uses point doses hs_m/tp_s/gamma",
        "orbital_drag_coeff": "wave force scale: hydro_v2 uses the hull's own water-relative law",
        "heave_force_scale": "wave force scale: hydro_v2 hydrostatics use the local elevation",
        "slope_moment_scale": "wave force scale: hydro_v2 restoring aligns to the surface normal",
        "hs_norm_ref": "sea-state observation scale: waves are never observed",
        "f_min": "legacy v1 frozen band", "f_max": "legacy v1 frozen band",
        "v2_f_min_over_fp": "renamed band_min_over_fp", "v2_f_max_over_fp": "renamed band_max_over_fp",
        "direction_spread_rad": "renamed spreading_rad",
    },
    "current": {
        "drag_coeff": "current drag law: hydro_v2 current is water velocity only",
        "force_mode": "current force mode: hydro_v2 current is water velocity only",
        "ramp_end_mps": "use speed_table", "ramp_duration_s": "use speed_table",
    },
    "wind": {
        "speed_mps": "renamed u10_mps (dose axis = U10)",
        "roughness_length_m": "vessel windage data (D5 registry) owns the roughness length",
        "air_density_kg_m3": "vessel windage data (D5 registry) owns the air density",
        "windage_profile": "hydro_v2 wind uses the D5 registry windage of the task vessel",
        "balanced_directions_rad": "use direction_mode='strata' (compass8_v1)",
        "direction_drift_rad_s": "direction drift is not a journal dose",
        "implementation": "fixed: C3 Fossen law with rng_protocol usvbench.wind.ou.v2",
    },
    "actuator": {
        "mass_scale": "centred-mass dose is REJECTED under hydro_v2 (deferred, decision 11)",
        "payload_mass_kg": "eccentric-payload dose is REJECTED under hydro_v2 (deferred, decision 11)",
        "payload_offset_x_m": "eccentric-payload dose is REJECTED under hydro_v2",
        "payload_offset_y_m": "eccentric-payload dose is REJECTED under hydro_v2",
        "reference_mass_kg": "mass/payload surrogate is REJECTED under hydro_v2",
        "reference_izz_kgm2": "mass/payload surrogate is REJECTED under hydro_v2",
        "reference_vehicle": "the vessel comes from the D5 registry via the task cfg",
        "drag_scale": "renamed resistance_scale (surge/sway/yaw damping multiplier)",
        "thrust_cap_scale": "renamed thrust_cap (per-motor forward and reverse limit fraction)",
        "motor_tau_s": "renamed additional_lag_s (cascade after the nominal motor pole)",
        "thrust_imbalance": "surge/yaw split semantics replaced by motor_gain_one_sided",
        "half_beam_m": "surge/yaw/half-beam semantics replaced by the per-motor mixer arm",
    },
    "perception": {
        "noise_pos_m": "renamed pos_noise_sigma_m (correlated, C7)",
        "noise_vel_mps": "renamed vel_noise_sigma_mps (sensor bundle)",
        "noise_range_m": "renamed ray_noise_sigma_m (sensor bundle)",
        "bias_pos_m": "renamed pos_bias_m (fixed magnitude, balanced direction)",
        "bias_vel_mps": "velocity bias is not a journal axis",
        "bias_range_m": "range bias is not a journal axis",
        "delay_steps": "renamed latency_s (seconds, multiple of the control period)",
        "dropout_p": "Bernoulli dropout replaced by one contiguous outage (outage_s)",
        "dropout_mode": "outage holds the last estimate",
    },
    "dynamic_hazards": {
        "*": "dynamic hazards are the D7 package (conditions/dynamic); the C4 progress "
             "trigger stays inert in journal mode (D7 B01)",
    },
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def zero_crossing_period(tp_s: float, gamma: float) -> float:
    """DNV-RP-C205 3.5.5.4: Tz/Tp = 0.6673 + 0.05037 g - 0.006230 g^2 + 0.0003341 g^3."""
    g = float(gamma)
    return float(tp_s) * (0.6673 + 0.05037 * g - 0.006230 * g ** 2 + 0.0003341 * g ** 3)


def steepness(hs_m: float, tp_s: float, gamma: float) -> dict:
    """Significant steepness Sp (Tp) and Ss (gamma-consistent Tz), DNV 3.5.4."""
    tz = zero_crossing_period(tp_s, gamma)
    sp = 2.0 * math.pi * float(hs_m) / (GRAVITY * float(tp_s) ** 2)
    ss = 2.0 * math.pi * float(hs_m) / (GRAVITY * tz ** 2)
    return {"sp": sp, "ss": ss, "tz_s": tz, "dnv_admitted": sp <= DNV_SP_LIMIT and ss <= DNV_SS_LIMIT,
            "generator_admitted": sp <= GENERATOR_LIMIT}


def check_steepness(waves: WavesV3) -> dict:
    """Raise if the wave dose fails its admission rule (the 1/7 guard always applies)."""
    report = steepness(waves.hs_m, waves.tp_s, waves.gamma)
    if waves.hs_m > 0:
        if not report["generator_admitted"]:
            raise ValueError("wave dose exceeds the generator breaking guard Sp <= 1/7")
        if waves.steepness_rule == "dnv_rp_c205_3_5_4" and not report["dnv_admitted"]:
            raise ValueError("wave dose fails DNV-RP-C205 3.5.4 admission (Sp <= 1/15 and "
                             "Ss <= 1/10): Sp=%.4f Ss=%.4f" % (report["sp"], report["ss"]))
    return report


def _check_direction(mode, value, name):
    if mode not in DIRECTION_MODES:
        raise ValueError(f"{name}.direction_mode must be one of {DIRECTION_MODES}")
    _finite(value, name + ".direction_rad")


def check_offset(offset, name, *, active, mode="strata"):
    """F0 / D4: a relative offset to the shared stratum direction.

    Declared conflicts (rejected with the reason): an offset outside (-pi, pi]
    (two spellings of one direction), an offset on a disabled suite/axis (it
    would silently do nothing) and an offset with direction_mode="fixed"
    (the fixed world angle and the offset would be two values for one
    direction).
    """
    _finite(offset, name)
    if not -math.pi < offset <= math.pi:
        raise ValueError(f"{name} must lie in (-pi, pi] (one spelling per direction)")
    if offset != 0.0 and not active:
        raise ValueError(f"declared conflict: {name} is set but its suite/axis is disabled "
                         "(an offset on an inactive disturbance has no effect)")
    if offset != 0.0 and mode != "strata":
        raise ValueError(f"declared conflict: {name} needs direction_mode='strata' (a fixed world "
                         "direction plus an offset would be two values for one direction)")


def validate_v3(cfg: JournalConditionConfig) -> JournalConditionConfig:
    if type(cfg.seed) is not int or not -(1 << 63) <= cfg.seed < (1 << 63):
        raise ValueError("seed must be a signed 64-bit integer")
    if type(cfg.schema_version) is not int or cfg.schema_version != SCHEMA_VERSION_V3:
        raise ValueError("unsupported condition schema version (hydro_v2 requires 3)")
    if cfg.hydrodynamics_model != HYDRODYNAMICS_MODEL:
        raise ValueError("schema 3 is the hydro_v2 condition layer")
    if type(cfg.preset_version) is not int or cfg.preset_version != PRESET_REGISTRY_VERSION_V3:
        raise ValueError("unsupported journal preset registry version")
    if cfg.direction_strata != "compass8_v1":
        raise ValueError("direction_strata must be 'compass8_v1'")
    from .presets_v3 import JOURNAL_PRESETS, COMBINATIONS  # late: presets import this module
    if not isinstance(cfg.presets, tuple) or any(
            p not in JOURNAL_PRESETS and p not in COMBINATIONS for p in cfg.presets):
        raise ValueError("invalid journal preset provenance")
    for name in SUITES_V3:
        if type(getattr(cfg, name).enabled) is not bool:
            raise ValueError(f"{name}.enabled must be bool")
    w = cfg.waves
    for key in ("hs_m", "tp_s", "gamma", "spreading_rad", "band_min_over_fp", "band_max_over_fp",
                "recurrence_min_s"):
        _finite(getattr(w, key), "waves." + key)
    if w.hs_m < 0 or w.tp_s <= 0 or w.gamma < 1.0 or not 0 <= w.spreading_rad <= math.pi:
        raise ValueError("invalid wave dose (hs >= 0, tp > 0, gamma >= 1, 0 <= spreading <= pi)")
    if not (0.0 < w.band_min_over_fp < 1.0 and 2.5 <= w.band_max_over_fp <= 50.0):
        raise ValueError("wave band requires 0 < min/fp < 1 and 2.5 <= max/fp <= 50")
    if type(w.n_components) is not int or not 256 <= w.n_components <= 16384:
        raise ValueError("waves.n_components must be an int in [256, 16384]")
    if w.band_policy not in BAND_POLICIES or w.steepness_rule not in STEEPNESS_RULES:
        raise ValueError("invalid wave band policy or steepness rule")
    if w.recurrence_min_s < 0:
        raise ValueError("waves.recurrence_min_s must be nonnegative")
    _check_direction(w.direction_mode, w.direction_rad, "waves")
    check_offset(w.direction_offset_rad, "waves.direction_offset_rad", active=w.enabled,
                 mode=w.direction_mode)
    check_steepness(w)
    c = cfg.current
    _finite(c.speed_mps, "current.speed_mps")
    if c.speed_mps < 0:
        raise ValueError("current speed must be nonnegative")
    if not isinstance(c.speed_table, tuple):
        raise ValueError("current.speed_table must be a tuple of (seconds, m/s) rows")
    previous = -1.0
    for row in c.speed_table:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError("speed_table rows are (seconds, m/s)")
        _finite(row[0], "table time"); _finite(row[1], "table speed")
        if row[0] < 0 or row[0] <= previous or row[1] < 0:
            raise ValueError("table times must strictly increase and speeds be nonnegative")
        previous = row[0]
    if c.speed_table and c.speed_mps != 0.0:
        raise ValueError("choose speed_mps OR speed_table")
    _check_direction(c.direction_mode, c.direction_rad, "current")
    check_offset(c.direction_offset_rad, "current.direction_offset_rad", active=c.enabled,
                 mode=c.direction_mode)
    wi = cfg.wind
    for key in ("u10_mps", "gust_intensity", "gust_correlation_time_s"):
        _finite(getattr(wi, key), "wind." + key)
    if wi.u10_mps < 0 or wi.gust_intensity < 0 or wi.gust_correlation_time_s <= 0:
        raise ValueError("invalid wind dose")
    if wi.reference not in WIND_REFERENCES:
        raise ValueError(f"wind.reference must be one of {WIND_REFERENCES}")
    if wi.rng_protocol != WIND_RNG_PROTOCOL:
        raise ValueError("wind.rng_protocol must be " + WIND_RNG_PROTOCOL)
    if wi.enabled and wi.u10_mps == 0.0:
        raise ValueError("zero wind dose means the wind module is OFF (enabled=False), "
                         "not U=0 (CONTEXT_R3 decision 9)")
    _check_direction(wi.direction_mode, wi.direction_rad, "wind")
    check_offset(wi.direction_offset_rad, "wind.direction_offset_rad", active=wi.enabled,
                 mode=wi.direction_mode)
    a = cfg.actuator
    for key in ("thrust_cap", "motor_gain_one_sided", "additional_lag_s", "resistance_scale"):
        _finite(getattr(a, key), "actuator." + key)
    if not 0.0 <= a.thrust_cap <= 1.0 or not 0.0 <= a.motor_gain_one_sided <= 1.0:
        raise ValueError("actuator cap and one-sided gain must lie in [0, 1]")
    if a.additional_lag_s < 0 or a.resistance_scale < 0:
        raise ValueError("actuator lag and resistance scale must be nonnegative")
    p = cfg.perception
    for key in ("pos_noise_sigma_m", "pos_noise_correlation_s", "pos_bias_m", "latency_s",
                "outage_s", "vel_noise_sigma_mps", "yaw_rate_noise_sigma_rad_s", "ray_noise_sigma_m"):
        _finite(getattr(p, key), "perception." + key)
        if getattr(p, key) < 0:
            raise ValueError(f"perception.{key} must be nonnegative")
    if p.pos_noise_correlation_s <= 0:
        raise ValueError("perception.pos_noise_correlation_s must be positive")
    if p.bias_direction_mode != "strata":
        raise ValueError("perception.bias_direction_mode must be 'strata'")
    window = p.outage_window_s
    if not isinstance(window, tuple) or len(window) != 2:
        raise ValueError("perception.outage_window_s must be (start_lo_s, start_hi_s)")
    for value in window:
        _finite(value, "perception.outage_window_s")
    if window[0] < 0 or window[1] < window[0]:
        raise ValueError("perception.outage_window_s must satisfy 0 <= lo <= hi")
    if p.protocol != PERCEPTION_PROTOCOL:
        raise ValueError("perception.protocol must be " + PERCEPTION_PROTOCOL)
    check_offset(p.bias_direction_offset_rad, "perception.bias_direction_offset_rad",
                 active=p.enabled and p.pos_bias_m > 0)
    return cfg


def _tupleize(value):
    if isinstance(value, list):
        return tuple(_tupleize(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_tupleize(v) for v in value)
    return value


def _reject_legacy(suite, values):
    reasons = LEGACY_KEYS.get(suite, {})
    if "*" in reasons:
        if not isinstance(values, Mapping) or set(values) - {"enabled"} or values.get("enabled", False):
            raise ValueError(f"{suite}: {reasons['*']}")
        return True
    for key in values:
        if key in reasons:
            raise ValueError(f"legacy key {suite}.{key} is rejected under hydro_v2: {reasons[key]}")
    return False


def resolve_v3(presets=(), overrides=None, *, seed=None) -> JournalConditionConfig:
    """Resolve journal preset names plus explicit nested overrides (schema 3).

    Presets are applied in order; a combination name expands to its
    component presets. Two presets writing the same field is an error, so the
    order can never silently change a run; ``enabled`` may be set by several.
    Legacy (schema-2) keys and unknown keys are rejected.
    """
    from .presets_v3 import expand_presets  # late import (presets import this module)
    if isinstance(overrides, JournalConditionConfig):
        if presets:
            raise ValueError("a resolved JournalConditionConfig cannot also take presets")
        return validate_v3(overrides)
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("hydro_v2 conditions must be a mapping or a JournalConditionConfig "
                         "(a schema-2 ConditionConfig is rejected)")
    data = asdict(JournalConditionConfig())
    owners = {}
    names = tuple(presets)
    for preset_name, suite_values in expand_presets(names):
        for suite, values in suite_values.items():
            for key, value in values.items():
                if key != "enabled":
                    owner = owners.setdefault((suite, key), preset_name)
                    if owner != preset_name:
                        raise ValueError(f"presets {owner!r} and {preset_name!r} both set {suite}.{key}")
                data[suite][key] = value
    for name, value in (overrides or {}).items():
        if name in LEGACY_KEYS and name not in data:
            if _reject_legacy(name, value):
                continue
        if name not in data:
            raise ValueError(f"unknown hydro_v2 condition key: {name}")
        if name in SUITES_V3:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} override must be a mapping")
            _reject_legacy(name, value)
            unknown = set(value) - set(data[name])
            if unknown:
                raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
            data[name].update(value)
        else:
            data[name] = value
    if seed is not None and "seed" not in (overrides or {}):
        data["seed"] = seed
    if names:
        data["presets"] = names
    for name, cls in SUITES_V3.items():
        data[name] = cls(**{key: _tupleize(val) for key, val in data[name].items()})
    data["presets"] = tuple(data["presets"])
    return validate_v3(JournalConditionConfig(**data))


__all__ = ["SCHEMA_VERSION_V3", "PRESET_REGISTRY_VERSION_V3", "HYDRODYNAMICS_MODEL",
           "WavesV3", "CurrentV3", "WindV3", "ActuatorV3", "PerceptionV3",
           "JournalConditionConfig", "SUITES_V3", "LEGACY_KEYS", "steepness",
           "zero_crossing_period", "check_steepness", "check_offset", "validate_v3", "resolve_v3",
           "DNV_SP_LIMIT", "DNV_SS_LIMIT", "GENERATOR_LIMIT", "WIND_RNG_PROTOCOL",
           "PERCEPTION_PROTOCOL"]
