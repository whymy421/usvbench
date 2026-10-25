"""Opt-in condition layer. Safe to load by package path without Isaac Lab.

Legacy composition (schema 2): ConditionLayer via build_conditions for
physics v1 / C2 physics v2. hydro_v2 composition (schema 3, E3):
HydroV2ConditionLayer via the same build_conditions when
``underwater_physics_cfg.hydrodynamics_model == "hydro_v2"``.
"""
from .config import (
    ConditionConfig, WaveConfig, CurrentConfig, WindConfig, ActuatorConfig,
    PerceptionConfig, DynamicHazardConfig, PRESETS, SCHEMA_VERSION, resolve,
)
from .config_v3 import (
    JournalConditionConfig, WavesV3, CurrentV3, WindV3, ActuatorV3, PerceptionV3,
    SCHEMA_VERSION_V3, LEGACY_KEYS, resolve_v3, validate_v3,
)
from .presets_v3 import JOURNAL_PRESETS, COMBINATIONS, ADMISSION
from .layer import ConditionLayer, build_conditions
from .hydro_v2_layer import HydroV2ConditionLayer, PhysicsInputs, build_hydro_v2_conditions
from .stamp import (STAMP_SCHEMA, STAMP_SCHEMA_V3, stamp_conditions, stamp_conditions_v3,
                    config_from_stamp, parameters_digest)
from .certificate import (certificate_fields, episode_certificate_fields, DYNAMIC_HAZARD_FIELD,
                          DYNAMIC_HAZARD_HEADER_FIELD, dynamic_hazard_digest,
                          verify_dynamic_hazard_stamp)
from .windage import WINDAGE_PROFILES, WINDAGE_PROFILE_VERSION, get_profile

__all__ = [
    "ConditionConfig", "WaveConfig", "CurrentConfig", "WindConfig",
    "ActuatorConfig", "PerceptionConfig", "DynamicHazardConfig", "PRESETS",
    "SCHEMA_VERSION", "resolve", "ConditionLayer", "build_conditions",
    "STAMP_SCHEMA", "stamp_conditions", "config_from_stamp", "parameters_digest",
    "certificate_fields", "episode_certificate_fields",
    "WINDAGE_PROFILES", "WINDAGE_PROFILE_VERSION", "get_profile",
    "JournalConditionConfig", "WavesV3", "CurrentV3", "WindV3", "ActuatorV3", "PerceptionV3",
    "SCHEMA_VERSION_V3", "LEGACY_KEYS", "resolve_v3", "validate_v3",
    "JOURNAL_PRESETS", "COMBINATIONS", "ADMISSION",
    "HydroV2ConditionLayer", "PhysicsInputs", "build_hydro_v2_conditions",
    "STAMP_SCHEMA_V3", "stamp_conditions_v3",
    "DYNAMIC_HAZARD_FIELD", "DYNAMIC_HAZARD_HEADER_FIELD", "dynamic_hazard_digest",
    "verify_dynamic_hazard_stamp",
]
