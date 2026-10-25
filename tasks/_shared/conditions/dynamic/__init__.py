"""D7 dynamic hazards: pre-admitted, controller-independent assignments, timed at runtime by the runtime
rule (release.py; G23c F11). Pure Python; no torch, no Isaac Lab.

Load this package BY PATH for CPU tests; do not import the repository's
top-level `tasks` package outside Isaac Lab. Environment hooks are specified
in D7's INTEGRATION_SPEC.md (plus the E4 addendum in E4 FINAL_REPORT.md);
no existing environment file is edited here. Vessel envelopes come only from
D5's final registry (vessels.envelope_from_spec_v2; E4, CONTEXT_R3 decision 1).
"""
from .model import (DOSE_TTC_RATIOS, EpisodeKey, Footprint, VesselEnvelope, Pose, TimedState,
                    Circle, Intruder, Encounter)
from .kinematics import integrate
from .reference import NominalPlan, build_transit_plan, build_stationary_plan
from .trajectory import PlacementError, generate_encounter, fit_motion_window, strata
from .geometry import ttc_linear, swept_interval_safe, point_clearance
from .feasibility import WitnessRequest, WitnessResult, find_witness, replay_witness
from .capacity import CapacityPlan, SlotState
from .native import NativeMotion, NATIVE_RATIOS, native_encounter
from .visibility import SensorSpec, RevealEvidence, validate_reveal, detectable, JOURNAL_RAY_SENSOR
from .vessels import (VesselPhysics, envelope_from_physics, envelope_from_spec_v2, physics_from_spec_v2,
                      footprint_from_polygon, two_motor_capacities, centreline_capacities,
                      load_registry_by_path, JOURNAL_VESSELS, DEFAULT_WAMV_PROFILE,
                      JOURNAL_REFERENCE_SPEED_MPS, JOURNAL_SPEED_CAP_FRACTION, JOURNAL_REACTION_S,
                      PENDING_PARAMETERS)
from .config import DynamicSuiteConfig, D7_SUITE_VERSION
from .adapters import (TASKS, PROFILES, ADAPTERS, TERMINATION_REASONS, LEGACY_RANGE_SENSOR, TaskProfile,
                       TaskContext, StageSite, Candidate, Assignment, AdmissionPool, Rejection,
                       DynamicRuntimeConfig, RuntimeFrame, ObstacleSample, build_candidate,
                       build_ladder, admit_ladders, admit_grid, resolve_sensor, assignment_summary,
                       make_adapter)
from .bridge import HazardBridge
from .release import (RUNTIME_RULE_VERSION, PINNED_RULE_VERSION, RUNTIME_RULES, ReleaseRoute, ReleaseState,
                      ReleasedIntruder, rule_block_for, REVEAL_RULE_VERSION, DEFAULT_RULE, DIAGNOSTIC_RULES,
                      HYBRID_RULE_VERSION, V3_RULE_VERSION)
from .reveal import RevealState, certified_segment
from .exposure import ExposureTracker
from .stamp import make_stamp, read_stamp, digest
from .dose_table import vessel_dose_table, DETECTION_PATH_TYPES
