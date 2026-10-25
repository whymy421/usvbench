"""G47 (v2.6.4): records[].min_clearance_m includes the dynamic D7 intruder (CPU only; pytest).

Bug (NC 00 progress 2026-10-14 06:xx; G12 dose smoke, docking BlueBoat classical dyn_D1, cell ds_docking_bb_cl_dyn_D1):
collisions with the intruder at a closest surface gap of 0.004-0.02 m (runtime.exposure_record, runtime
min_sampled_surface_clearance_m) were certified with min_clearance_m 3-4 m. Cause: every family's env contract samples
the clearance to the STATIC geometry only (docking: berth walls, docking_env._journal_clearance; hazard_nav:
_journal_static_clearance), and cert_v3 copied that value. Fix (metric only, in the certificate writer): min_clearance_m
= min(contract value, the D7 runtime's sampled intruder surface clearance). Covers:
  * the docking evidence record (env 25 ep 0: 4.3942 m -> 0.0038649 m), outcome unchanged (collision -> contact_voided);
  * no D7 runtime sample / a larger D7 gap / D0 stamps leave the value unchanged; a missing contract value takes the
    D7 gap;
  * a fold that would change the outcome is NOT applied (note returned, the writer lists the record as unverified);
  * the writer integration: CertificateRun.finish_episode on a fake journal env reports the folded value and the
    same outcome, and the contract's own value stays in journal_contract.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cert_v3  # noqa: E402
from episode_outcome import episode_outcome  # noqa: E402


def envelope(gap):
    return {"payload": {"runtime": {"min_sampled_surface_clearance_m": gap, "sampled_contact": gap is not None
                                    and gap <= 0.0}}, "sha256": "0" * 64}


# the evidence record (fields used by the outcome table), docking BlueBoat classical dyn_D1, env 25 ep 0
EVIDENCE = {"env": 25, "ep": 0, "success": False, "termination_reason": "collision", "contact_steps": None,
            "min_clearance_m": 4.3942389488220215}
EVIDENCE_GAP = 0.0038649030218381175


def test_evidence_record_reports_the_intruder_gap_and_keeps_the_outcome():
    before = episode_outcome(EVIDENCE, True)
    value, note = cert_v3.d7_clearance_fold(EVIDENCE, envelope(EVIDENCE_GAP), True)
    assert note is None and value == EVIDENCE_GAP
    assert episode_outcome(dict(EVIDENCE, min_clearance_m=value), True) == before == "contact_voided"


def test_unchanged_cases():
    rec = dict(EVIDENCE, min_clearance_m=0.5, termination_reason="timeout", success=True)
    assert cert_v3.d7_clearance_fold(rec, envelope(None), True) == (0.5, None)          # no D7 sample (D0 etc.)
    assert cert_v3.d7_clearance_fold(rec, envelope(0.9), True) == (0.5, None)           # the walls are closer
    assert cert_v3.d7_clearance_fold(rec, {"payload": {}}, True) == (0.5, None)          # no runtime block
    assert cert_v3.d7_clearance_fold(rec, None, True) == (0.5, None)
    assert cert_v3.d7_clearance_fold(rec, envelope(float("nan")), True) == (0.5, None)
    assert cert_v3.d7_clearance_fold(rec, envelope(True), True) == (0.5, None)          # a bool is no number
    none = dict(rec, min_clearance_m=None)                                              # no static geometry
    assert cert_v3.d7_clearance_fold(none, envelope(1.25), True) == (1.25, None)
    succ = dict(rec, min_clearance_m=3.0)
    value, note = cert_v3.d7_clearance_fold(succ, envelope(0.2), True)
    assert (value, note) == (0.2, None) and episode_outcome(dict(succ, min_clearance_m=value), True) == "success"


def test_a_fold_that_would_change_the_outcome_is_not_applied():
    rec = {"success": False, "termination_reason": "timeout", "contact_steps": None, "min_clearance_m": 2.0}
    assert episode_outcome(rec, True) == "never_achieved"
    value, note = cert_v3.d7_clearance_fold(rec, envelope(-0.01), True)                 # would read as a contact
    assert value == 2.0 and note is not None and "outcome" in note


class _FakeBase:
    def __init__(self, gap):
        import types
        self.termination_reason = [2]                       # collision
        self._gap = gap
        self.journal_episode_results = [{"min_clearance_m": 4.3942389488220215}]
        self.types = types

    def journal_episode_record(self, env):
        return {"conditions": {"schema": "x"}, "dynamic_hazards": envelope(self._gap), "scenario_hashes": {}}


def test_writer_integration_on_a_fake_journal_env():
    writer = cert_v3.CertificateRun.__new__(cert_v3.CertificateRun)   # only finish_episode's collaborators
    base = _FakeBase(EVIDENCE_GAP)

    class Grid:
        counts = [0]

        def accept(self, env):
            self.counts[env] += 1

    class Paths:
        def take(self, env, success):
            return {"path_length_full_episode_m": 6.9, "path_length_success_prefix_m": None}

    class Api:
        def verify_dynamic_hazard_stamp(self, env):
            return None

    writer.grid, writer.quota, writer.base, writer.unverified = Grid(), 1, base, []
    writer._initial_distances, writer._samples, writer.control_step_s = None, {0: ([0.01], [0.05])}, 1.0 / 60.0
    writer._paths, writer._outcome_semantics, writer._journal = Paths(), {"fixed_horizon": True}, True
    writer._conditions_api, writer.header = Api(), {}
    writer.discard_episode = lambda env: None
    original = copy.deepcopy(cert_v3.d7_exposure.exposure_of)
    cert_v3.d7_exposure.exposure_of = lambda env: ("admitted", [])
    try:
        rec = writer.finish_episode(0, {"env": 0, "ep": 0, "success": False, "terminated": True, "truncated": False,
                                        "min_clearance_m": 4.3942389488220215})
    finally:
        cert_v3.d7_exposure.exposure_of = original
    assert rec["min_clearance_m"] == EVIDENCE_GAP and rec["outcome"] == "contact_voided" and writer.unverified == []
    assert rec["journal_contract"]["journal_episode_results"]["min_clearance_m"] == 4.3942389488220215
