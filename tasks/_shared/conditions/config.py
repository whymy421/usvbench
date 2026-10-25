"""Versioned, simulator-independent condition configuration and presets.

Numbers below are engineering test doses, not a calibrated journal ladder.
No preset changes an existing task registration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from typing import Mapping

# Condition-layer schema. 1 = C4 prototype; 2 = D1 J-base (C3 wind backend,
# v2 wave band fields). Stamps carry "usvbench.conditions/<SCHEMA_VERSION>".
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class WaveConfig:
    enabled: bool = False
    hs_range: tuple = (0.15, 0.15)
    tp_range: tuple = (3.0, 3.0)
    gamma_range: tuple = (3.3, 3.3)
    direction_spread_rad: float = math.pi / 6
    n_components: int = 104
    f_min: float = 0.04
    f_max: float = 1.60
    orbital_drag_coeff: float = 8.0
    heave_force_scale: float = 100.0
    slope_moment_scale: float = 200.0
    hs_norm_ref: float = 0.60
    # Physics v2 only (C2 SeaState): per-realization band [lo, hi] * fp with
    # at least 256 components. n_components/f_min/f_max describe the frozen
    # v1 grid and are ignored by v2 (see SeaState.__init__).
    v2_f_min_over_fp: float = 0.2
    v2_f_max_over_fp: float = 3.0


@dataclass(frozen=True)
class CurrentConfig:
    enabled: bool = False
    speed_mps: float = 0.0
    direction_rad: float | None = None
    drag_coeff: float = 8.0
    ramp_end_mps: float | None = None
    ramp_duration_s: float = 120.0
    speed_table: tuple = ()
    force_mode: str = "incremental_from_calm"


@dataclass(frozen=True)
class WindConfig:
    """C3 Fossen/OU wind (D1). Directions are air FLOW-TO azimuths, CCW.

    speed_mps is the REQUESTED mean speed at ``reference``: "u10" (10 m
    marine reference, converted to the vessel reference height with a
    neutral log profile of roughness ``roughness_length_m``; declared
    assumption, owner decision pending) or "vessel_height" (applied as is).
    direction_mode "fixed" uses direction_rad in the world frame; "uniform"
    draws a keyed per-episode offset on [0, 2*pi) from the task reference
    axis; "balanced" assigns balanced_directions_rad offsets from that axis.
    windage_profile "auto" resolves from the task vehicle when enabled.
    """
    enabled: bool = False
    speed_mps: float = 0.0
    reference: str = "u10"
    roughness_length_m: float = 0.0002
    direction_mode: str = "fixed"
    direction_rad: float = 0.0
    balanced_directions_rad: tuple = tuple(k * math.pi / 4 for k in range(8))
    direction_drift_rad_s: float = 0.0
    gust_intensity: float = 0.0
    gust_correlation_time_s: float = 2.0
    air_density_kg_m3: float = 1.225
    windage_profile: str = "auto"
    implementation: str = "c3_fossen_ou_v1"


@dataclass(frozen=True)
class ActuatorConfig:
    enabled: bool = False
    mass_scale: float = 1.0
    drag_scale: float = 1.0
    thrust_cap_scale: float = 1.0
    motor_tau_s: float = 0.0
    thrust_imbalance: float = 0.0
    payload_mass_kg: float = 0.0
    payload_offset_x_m: float = 0.0
    payload_offset_y_m: float = 0.0
    reference_vehicle: str = "blueboat"
    reference_mass_kg: float = 17.26
    reference_izz_kgm2: float = 2.376
    half_beam_m: float = 0.45


@dataclass(frozen=True)
class PerceptionConfig:
    enabled: bool = False
    noise_pos_m: float = 0.0
    noise_vel_mps: float = 0.0
    noise_range_m: float = 0.0
    bias_pos_m: float = 0.0
    bias_vel_mps: float = 0.0
    bias_range_m: float = 0.0
    delay_steps: int = 0
    dropout_p: float = 0.0
    dropout_mode: str = "hold_last"


@dataclass(frozen=True)
class DynamicHazardConfig:
    enabled: bool = False
    definition: str = "per_task_pending"
    trigger_progress: float = 0.5
    radius_m: float = 0.5
    speed_mps: float = 0.25


@dataclass(frozen=True)
class ConditionConfig:
    waves: WaveConfig = field(default_factory=WaveConfig)
    current: CurrentConfig = field(default_factory=CurrentConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    dynamic_hazards: DynamicHazardConfig = field(default_factory=DynamicHazardConfig)
    seed: int = 0
    presets: tuple = ()
    preset_version: int = 1
    # 2 (D1): real C3 wind suite, v2 wave band fields; stamp schema
    # "usvbench.conditions/2". Schema 1 (C4 prototype) is not accepted.
    schema_version: int = SCHEMA_VERSION

    @property
    def any_enabled(self):
        return any(getattr(self, name).enabled for name in SUITES)


SUITES = {
    "waves": WaveConfig, "current": CurrentConfig, "wind": WindConfig,
    "actuator": ActuatorConfig, "perception": PerceptionConfig,
    "dynamic_hazards": DynamicHazardConfig,
}

# One preset owns one suite. Duplicate suite selections are errors, so order
# cannot silently change a run; explicit overrides are applied after presets.
PRESETS = {
    "wave.mild": {"waves": {"enabled": True, "hs_range": (0.075, 0.075)}},
    "wave.medium": {"waves": {"enabled": True, "hs_range": (0.15, 0.15)}},
    "wave.heavy": {"waves": {"enabled": True, "hs_range": (0.30, 0.30)}},
    "current.mild": {"current": {"enabled": True, "speed_mps": 0.25}},
    "current.medium": {"current": {"enabled": True, "speed_mps": 0.5}},
    "current.heavy": {"current": {"enabled": True, "speed_mps": 1.0}},
    "current.ramp": {"current": {"enabled": True, "speed_mps": 0.25,
                                    "ramp_end_mps": 1.0}},
    "actuator.mass": {"actuator": {"enabled": True, "mass_scale": 2.0}},
    "actuator.drag": {"actuator": {"enabled": True, "drag_scale": 2.0}},
    "actuator.cap": {"actuator": {"enabled": True, "thrust_cap_scale": 0.5}},
    "actuator.lag": {"actuator": {"enabled": True, "motor_tau_s": 0.5}},
    "actuator.imbalance": {"actuator": {"enabled": True, "thrust_imbalance": 0.2}},
    "actuator.payload": {"actuator": {"enabled": True, "payload_mass_kg": 4.0,
                                      "payload_offset_x_m": 0.2}},
    "perception.noise": {"perception": {"enabled": True, "noise_pos_m": 0.1,
                                        "noise_vel_mps": 0.05}},
    "perception.bias": {"perception": {"enabled": True, "bias_pos_m": 0.1}},
    "perception.delay": {"perception": {"enabled": True, "delay_steps": 3}},
    "perception.dropout": {"perception": {"enabled": True, "dropout_p": 0.2}},
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def validate(cfg):
    if type(cfg.seed) is not int or not -(1 << 63) <= cfg.seed < (1 << 63):
        raise ValueError("seed must be a signed 64-bit integer")
    if type(cfg.schema_version) is not int or cfg.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported condition schema version")
    if type(cfg.preset_version) is not int or cfg.preset_version != 1:
        raise ValueError("unsupported preset registry version")
    if not isinstance(cfg.presets, (tuple, list)) or any(preset not in PRESETS for preset in cfg.presets):
        raise ValueError("invalid preset provenance")
    for name in SUITES:
        suite = getattr(cfg, name)
        if type(suite.enabled) is not bool:
            raise ValueError(f"{name}.enabled must be bool")
        non_numeric = {"enabled", "hs_range", "tp_range", "gamma_range", "speed_table",
                       "force_mode", "implementation", "reference_vehicle", "dropout_mode", "definition",
                       "reference", "direction_mode", "balanced_directions_rad", "windage_profile"}
        for key, value in asdict(suite).items():
            if key in non_numeric:
                continue
            if name == "current" and key in ("direction_rad", "ramp_end_mps") and value is None:
                continue
            _finite(value, f"{name}.{key}")
    c = cfg.current
    for key in ("speed_mps", "drag_coeff"):
        if getattr(c, key) < 0:
            raise ValueError(f"current.{key} must be nonnegative")
    if c.ramp_end_mps is not None and c.ramp_end_mps < 0:
        raise ValueError("negative ramp speed")
    if c.ramp_duration_s <= 0 or c.force_mode not in ("incremental_from_calm", "legacy_relative_drag"):
        raise ValueError("invalid current duration or force mode")
    if c.speed_table and c.ramp_end_mps is not None:
        raise ValueError("choose a ramp OR a speed table")
    previous = -1.0
    for pair in c.speed_table:
        if len(pair) != 2:
            raise ValueError("speed_table rows are (seconds, m/s)")
        t, speed = pair
        _finite(t, "table time"); _finite(speed, "table speed")
        if t < 0 or t <= previous or speed < 0:
            raise ValueError("table times must strictly increase and speeds be nonnegative")
        previous = t
    a = cfg.actuator
    if min(a.mass_scale, a.reference_mass_kg, a.reference_izz_kgm2, a.half_beam_m) <= 0:
        raise ValueError("actuator mass, inertia, scale and lever must be positive")
    if not isinstance(a.reference_vehicle, str) or not a.reference_vehicle:
        raise ValueError("actuator.reference_vehicle must name its calibrated vessel")
    if min(a.drag_scale, a.motor_tau_s, a.payload_mass_kg) < 0:
        raise ValueError("actuator drag, lag and payload must be nonnegative")
    if not 0 <= a.thrust_cap_scale <= 1 or not -1 <= a.thrust_imbalance <= 1:
        raise ValueError("invalid actuator cap or imbalance")
    p = cfg.perception
    if type(p.delay_steps) is not int or p.delay_steps < 0 or not 0 <= p.dropout_p <= 1:
        raise ValueError("invalid perception delay/dropout")
    if p.dropout_mode != "hold_last":
        raise ValueError("only legacy hold_last perception is implemented")
    if any(getattr(p, key) < 0 for key in asdict(p) if key.startswith(("noise_", "bias_"))):
        raise ValueError("perception sigmas must be nonnegative")
    w = cfg.waves
    for key in ("hs_range", "tp_range", "gamma_range"):
        values = getattr(w, key)
        if len(values) != 2:
            raise ValueError(f"waves.{key} requires two values")
        for v in values:
            _finite(v, key)
        if values[0] > values[1] or values[0] < 0 or (key != "hs_range" and values[0] <= 0):
            raise ValueError(f"invalid waves.{key}")
    if type(w.n_components) is not int or w.n_components < 1 or not 0 < w.f_min < w.f_max:
        raise ValueError("invalid wave frequency grid")
    if min(w.direction_spread_rad, w.orbital_drag_coeff, w.heave_force_scale, w.slope_moment_scale) < 0 or w.hs_norm_ref <= 0:
        raise ValueError("invalid wave coupling")
    if not (0.0 < w.v2_f_min_over_fp < 1.0 and w.v2_f_max_over_fp >= 2.5):
        raise ValueError("v2 wave band requires 0 < f_min/fp < 1 and f_max/fp >= 2.5")
    d = cfg.dynamic_hazards
    if not 0 <= d.trigger_progress <= 1 or d.radius_m <= 0 or d.speed_mps < 0:
        raise ValueError("invalid dynamic hazard parameters")
    wind = cfg.wind
    if wind.speed_mps < 0:
        raise ValueError("negative wind speed")
    if wind.reference not in ("u10", "vessel_height"):
        raise ValueError("wind.reference must be 'u10' or 'vessel_height'")
    if not 0.0 < wind.roughness_length_m < 10.0:
        raise ValueError("wind.roughness_length_m must be in (0, 10) m")
    if wind.direction_mode not in ("fixed", "uniform", "balanced"):
        raise ValueError("wind.direction_mode must be 'fixed', 'uniform' or 'balanced'")
    choices = wind.balanced_directions_rad
    if not isinstance(choices, (tuple, list)) or not choices:
        raise ValueError("wind.balanced_directions_rad must be a non-empty sequence")
    for value in choices:
        _finite(value, "wind.balanced_directions_rad")
    if len(set(float(v) % (2 * math.pi) for v in choices)) != len(choices):
        raise ValueError("wind.balanced_directions_rad must not repeat a direction")
    if wind.gust_intensity < 0 or wind.gust_correlation_time_s <= 0 or wind.air_density_kg_m3 <= 0:
        raise ValueError("invalid wind gust or air density")
    if not isinstance(wind.windage_profile, str) or not wind.windage_profile:
        raise ValueError("wind.windage_profile must name a profile or be 'auto'")
    if wind.implementation != "c3_fossen_ou_v1":
        raise ValueError("wind.implementation must be 'c3_fossen_ou_v1'")
    return cfg


def resolve(presets=(), overrides=None, *, seed=None):
    """Resolve names plus explicit nested overrides; reject misspelled keys."""
    if isinstance(overrides, ConditionConfig):
        if presets:
            raise ValueError("a resolved ConditionConfig cannot also take presets")
        return validate(overrides)
    data = asdict(ConditionConfig())
    selected = set()
    for preset in presets:
        if preset not in PRESETS:
            raise ValueError(f"unknown condition preset: {preset}")
        for name, values in PRESETS[preset].items():
            if name in selected:
                raise ValueError(f"multiple presets for suite {name}; use explicit overrides")
            selected.add(name)
            data[name].update(values)
    for name, value in (overrides or {}).items():
        if name not in data:
            raise ValueError(f"unknown condition key: {name}")
        if name in SUITES:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} override must be a mapping")
            unknown = set(value) - set(data[name])
            if unknown:
                raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
            data[name].update(value)
        else:
            data[name] = value
    if seed is not None and "seed" not in (overrides or {}):
        data["seed"] = seed
    if presets:
        data["presets"] = tuple(presets)
    for name, cls in SUITES.items():
        for key, val in data[name].items():
            if isinstance(val, list):
                data[name][key] = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        data[name] = cls(**data[name])
    data["presets"] = tuple(data["presets"])
    return validate(ConditionConfig(**data))
