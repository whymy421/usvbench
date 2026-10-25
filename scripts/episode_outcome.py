"""Per-episode outcome: ONE mapping table (D2 review item 21). Standard library only.

The termination reason says WHY an episode stopped (C5 codes 0-5 plus D4's
appended validity codes 6-9, tasks/_shared/journal_contracts_v2.py:28-46). In
fixed-horizon families (station keeping, docking) every episode stops by
timeout, so the reason cannot separate a contact-voided run from one that
never achieved the task. ``outcome`` is derived from the published reason and
the task latches recorded in the same record, by the first matching row of
OUTCOME_TABLE; the certificate writer (cert_v3.py) and the validator
(validate_cert_v3.py) both call ``episode_outcome``.

Latches used:
  success       record["success"] (the task's own success latch)
  contact       record["contact_steps"] > 0 or record["min_clearance_m"] < 0;
                None when the family records neither (no contact telemetry)
  fixed_horizon header["outcome_semantics"]["fixed_horizon"]: True when
                success does not end the episode (station keeping, docking),
                False when it does, None when the family is unknown
"""
from __future__ import annotations

# Certificate string for every published integer termination reason. 0-5 are
# C5 (tasks/_shared/termination_reason.py); 6-9 are D4's appended validity
# reasons with D4's own names (journal_contracts_v2.TERMINATION_REASON_NAMES).
TERMINATION_REASON_NAMES = {0: "running", 1: "success", 2: "collision", 3: "timeout",
                            4: "out_of_bounds", 5: "failure", 6: "capsize", 7: "sinking",
                            8: "airborne", 9: "non_finite_state"}
RECORD_REASONS = tuple(name for code, name in sorted(TERMINATION_REASON_NAMES.items()) if code) + ("unknown",)
OUTCOMES = ("success", "contact_voided", "never_achieved", "out_of_bounds", "capsized", "sinking",
            "airborne", "non_finite_state", "timeout")
ANY = "any"

# (termination_reason, success, contact, fixed_horizon) -> outcome; first match wins.
# Precedence follows D4 section 2.8 / journal_contracts_v2.termination_decision:
# validity > out_of_bounds > collision > success > failure > timeout.
OUTCOME_TABLE = (
    ("capsize", ANY, ANY, ANY, "capsized"),
    ("sinking", ANY, ANY, ANY, "sinking"),
    ("airborne", ANY, ANY, ANY, "airborne"),
    ("non_finite_state", ANY, ANY, ANY, "non_finite_state"),
    ("out_of_bounds", ANY, ANY, ANY, "out_of_bounds"),
    ("collision", ANY, ANY, ANY, "contact_voided"),
    ("success", True, ANY, ANY, "success"),
    ("timeout", True, ANY, ANY, "success"),          # fixed-horizon success
    ("failure", False, True, ANY, "contact_voided"),  # e.g. contacted arrival, contact stop off
    ("failure", False, ANY, ANY, "never_achieved"),
    ("timeout", False, True, ANY, "contact_voided"),
    ("timeout", False, ANY, True, "never_achieved"),  # fixed horizon: criterion never met
    ("timeout", False, ANY, ANY, "timeout"),          # stop-on-success family ran out of time
)


def contact_latch(record):
    """True/False from the recorded contact telemetry, None when none is recorded."""
    steps = record.get("contact_steps")
    clearance = record.get("min_clearance_m")
    observed = False
    if isinstance(steps, (int, float)) and not isinstance(steps, bool):
        if steps > 0:
            return True
        observed = True
    if isinstance(clearance, (int, float)) and not isinstance(clearance, bool):
        if clearance < 0:
            return True
        observed = True
    return False if observed else None


def _matches(pattern, value):
    return pattern == ANY or pattern == value


def episode_outcome(record, fixed_horizon):
    """Outcome string, or None for an unknown reason or an impossible combination."""
    reason = record.get("termination_reason")
    success = record.get("success")
    contact = contact_latch(record)
    for row_reason, row_success, row_contact, row_fixed, outcome in OUTCOME_TABLE:
        if (row_reason == reason and _matches(row_success, success)
                and _matches(row_contact, contact) and _matches(row_fixed, fixed_horizon)):
            return outcome
    return None
