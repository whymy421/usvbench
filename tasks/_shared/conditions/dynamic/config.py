"""Typed, immutable D7 suite configuration (the journal replacement for C4's
DynamicHazardConfig progress trigger, conditions/config.py:80-86).

C4's DynamicHazardConfig carries trigger_progress/radius_m/speed_mps and its
DynamicHazardCondition fires on a controller-dependent voyage fraction
(conditions/dynamic_hazards.py:46-58). D7 never uses that path: the world is
pre-generated and admitted offline. to_c4_override() only records D7 identity
in C4's stamp block so the C4 parameter digest changes with the D7 dose; the
C4 layer must NOT construct DynamicHazardCondition for a D7 run
(INTEGRATION_SPEC.md B01-B03).

E4 (suite version d7_ttc_dose_v2): journal_ray_sensor_all_tasks (default
True, CONTEXT_R3 decision 14, PENDING) makes every task sensed with the
standard 36-ray 30 m fan (visibility.JOURNAL_RAY_SENSOR) when its TaskContext
has no existing sensor; False restores D7's fail-closed behaviour
(allow_unsensed then decides between rejection and unperceivable cells).
"""

from dataclasses import asdict, dataclass

from .model import REVEAL_KINDS, UNFORCED_WINDOW_RULE, check_number
from .native import NU_POLICIES

# v3 (G7 item B, owner-approved 2026-10-01): the unforced D1 window counts a nominal-plan
# command as zero below 1e-6 rad/s and 1e-6 m/s^2 (model.UNFORCED_WINDOW_RULE); v2 required
# exact zeros. Pools record the version (and the rule) that admitted them.
D7_SUITE_VERSION = "d7_ttc_dose_v3"


@dataclass(frozen=True)
class DynamicSuiteConfig:
    enabled: bool = False
    dose: int = 0
    vessel_id: str = ""
    reveal_kind: str = "detection_horizon"
    intruder_speed_mps: float = 0.5        # C7 STRATEGY.md:225 (provisional)
    drift_speed_mps: float = 0.1           # "drifting" debris; declared engineering value
    intruder_size_beams: float = 0.5       # diameter in own beams, C7 STRATEGY.md:225
    static_margin_m: float = 0.05          # intruder-to-wall clearance for fit_motion_window
    plan_dt_s: float = 0.25
    witness_dt_s: float = 0.25
    witness_beam_width: int = 48
    witness_clearance_m: float = 0.02
    witness_substeps: int = 4
    visibility_sample_dt_s: float = 0.05
    joint_dose_admission: bool = True      # admit a key only if D0..D4 all admit
    allow_unsensed: bool = False           # only used when no sensor resolves (flag below False)
    native_nu_policy: str = "tied"
    native_fixed_nu: float = 0.0
    verify_witness_on_reset: bool = False
    # CONTEXT_R3 decision 14 (PENDING): the journal 36-ray 30 m sensor on every task.
    journal_ray_sensor_all_tasks: bool = True
    version: str = D7_SUITE_VERSION
    # Declared, not configurable: recorded so every pool / header / suite identity names its window rule.
    unforced_window_rule: str = UNFORCED_WINDOW_RULE

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in (
                "enabled", "joint_dose_admission", "allow_unsensed", "verify_witness_on_reset",
                "journal_ray_sensor_all_tasks")):
            raise ValueError("boolean flags must be bool")
        if type(self.dose) is not int or not 0 <= self.dose <= 4:
            raise ValueError("dose must be an integer D0..D4")
        if not self.enabled and self.dose != 0:
            raise ValueError("a disabled suite must have dose 0")
        if self.reveal_kind not in REVEAL_KINDS:
            raise ValueError("unknown reveal kind")
        if self.native_nu_policy not in NU_POLICIES:
            raise ValueError("unknown native nu policy")
        for name in ("intruder_speed_mps", "drift_speed_mps", "intruder_size_beams", "plan_dt_s",
                     "witness_dt_s", "visibility_sample_dt_s"):
            check_number(name, getattr(self, name), positive=True)
        for name in ("static_margin_m", "witness_clearance_m", "native_fixed_nu"):
            check_number(name, getattr(self, name), minimum=0.0)
        if type(self.witness_beam_width) is not int or not 1 <= self.witness_beam_width <= 4096:
            raise ValueError("witness_beam_width must be an integer in [1, 4096]")
        if type(self.witness_substeps) is not int or not 1 <= self.witness_substeps <= 256:
            raise ValueError("witness_substeps must be an integer in [1, 256]")
        if self.plan_dt_s > self.witness_dt_s:
            raise ValueError("plan_dt_s must not exceed witness_dt_s (the witness replays the plan prefix)")
        if self.version != D7_SUITE_VERSION:
            raise ValueError("unsupported D7 suite version")
        if self.unforced_window_rule != UNFORCED_WINDOW_RULE:
            raise ValueError("the unforced-window rule is declared, not configurable (model.UNFORCED_WINDOW_RULE)")
        if self.enabled and not self.vessel_id:
            raise ValueError("an enabled suite must name its resolved vessel envelope")

    def at_dose(self, dose):
        values = asdict(self)
        values["dose"] = dose
        values["enabled"] = values["enabled"] or dose > 0
        return DynamicSuiteConfig(**values)

    def to_c4_override(self):
        """Identity-only mapping into C4's DynamicHazardConfig fields (see module doc)."""
        return {"enabled": False,
                "definition": "%s;dose=D%d;vessel=%s;journal_rays=%d;window=%s" % (
                    self.version, self.dose, self.vessel_id or "none", int(self.journal_ray_sensor_all_tasks),
                    self.unforced_window_rule)}
