"""G23b/G23c: runtime-rule stamps, metric A and the realized-exposure record in certificates. CPU only.

* the REAL cert_v3 writer over the REAL F0 mixin (test_journal_evaluators.journal_document, the reach
  carrier with a dynamic dose) stamps dynamic_hazards_header.runtime_rule (G23c F7) and .exposure_summary
  (schema /3, G23d), every admitted D1..D4 record carries the runtime fields (runtime_rule, release,
  finite_ttc_at_first_detection, TTC / t_avoid, exposure_record), D0 records carry no rule (G23c F13), and
  validate_cert_v3 accepts the certificate;
* the validator rejects a tampered summary, a missing summary with runtime-rule records, a metric-A flag that
  contradicts its TTC, a diagnostic rule in a non-diagnostic certificate (G23c F10, REV T3/T5), tampered rule
  constants after re-hashing (REV T11), an erased detection after re-hashing (REV T8), and a header rule that
  differs from the records; v2.5-style stamps (no runtime_rule) are read as rule None; G23d (REV-G23c T15/T16): a
  header declaring the default rule over admitted records without its block is rejected;
* d7_exposure.runtime_exposure / exposure_summary on hand-made stamp payloads.
Run: python -B scripts/tests_g23b/test_g23b_exposure.py   (or pytest)
"""
from __future__ import annotations

import copy
import os
import sys

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import d7_exposure  # noqa: E402
import test_journal_evaluators as tje  # noqa: E402

RULE = d7_exposure.DEFAULT_RULE
CAUSE = d7_exposure.RULE_CAUSES[RULE][0]
UNRELEASED = d7_exposure.UNRELEASED_REASONS[RULE]


def _doc(dose=2):
    return tje.journal_document(("dynamic=D%d" % dose,))


def test_writer_stamps_rule_and_summary_and_validator_accepts():
    for dose in (0, 2, 4):
        doc = _doc(dose)
        errors = tje._check(doc)
        assert not errors, (dose, errors)
        block = doc["header"]["dynamic_hazards_header"]
        rule = block["runtime_rule"]["version"]
        assert block["runtime_rule"]["motion_rule"] == rule
        summary = block["exposure_summary"]
        assert summary["schema"] == d7_exposure.SUMMARY_SCHEMA and summary["dose"] == dose
        assert summary["records"] == len(doc["records"])
        envelopes = [r["dynamic_hazards"] for r in doc["records"]]
        assert summary == d7_exposure.exposure_summary(envelopes, dose)
        for envelope in envelopes:
            runtime = envelope["payload"]["runtime"]
            if dose == 0:
                assert runtime["runtime_rule"] is None and runtime["release"] is None
                assert summary["diagnostic"]["finite_ttc_at_detection_share"] is None and summary["rule"] is None
            else:
                assert runtime["runtime_rule"]["version"] == rule == summary["rule"]
                assert runtime["release"]["rule"] == rule and runtime["release"]["plan_kind"] == "transit"
                assert runtime["exposure_record"]["schema"] == d7_exposure.EXPOSURE_RECORD_SCHEMA
                assert runtime["exposure_record"]["first_visible_s"] == runtime["first_detection_s"]
                assert runtime["non_exposure_reason"] in d7_exposure.NON_EXPOSURE_REASONS


def test_validator_rejects_tampering():
    doc = _doc(2)
    bad = copy.deepcopy(doc)
    bad["header"]["dynamic_hazards_header"]["exposure_summary"]["admitted"] += 1
    assert any("exposure_summary differs" in e for e in tje._check(bad))
    bad = copy.deepcopy(doc)
    bad["header"]["dynamic_hazards_header"].pop("exposure_summary")
    assert any("exposure_summary is missing" in e for e in tje._check(bad))
    bad = tje.restamp(copy.deepcopy(doc), 0, lambda p: p["runtime"].__setitem__(
        "finite_ttc_at_first_detection", not p["runtime"]["finite_ttc_at_first_detection"]))
    assert any("runtime exposure rejected" in e for e in tje._check(bad))
    # REV T3/T5: a diagnostic rule outside a diagnostic run (also differs from the header)
    bad = tje.restamp(copy.deepcopy(doc), 1, lambda p: p["runtime"].__setitem__(
        "runtime_rule", {"version": "pinned_v0", "diagnostic_only": True}))
    errors = tje._check(bad)                 # the fixture is a DIAG42 (diagnostic) run: mixing still refused
    assert any("differ from the header" in e for e in errors), errors
    import validate_cert_v3
    official = copy.deepcopy(bad)
    official["header"]["seed_namespace"] = dict(official["header"]["seed_namespace"], purpose="evaluation")
    assert any("diagnostic only" in e for e in validate_cert_v3.dynamic_exposure_errors(official))
    diagnostic_only = tje.restamp(copy.deepcopy(doc), 1, lambda p: None)
    assert not any("diagnostic only" in e for e in validate_cert_v3.dynamic_exposure_errors(diagnostic_only))
    # REV T11: rule constants tampered after re-hashing
    bad = tje.restamp(copy.deepcopy(doc), 0, lambda p: p["runtime"]["runtime_rule"]["constants"].__setitem__(
        "release_margin_s", 5.0))
    assert any("runtime_rule block differs" in e for e in tje._check(bad))
    bad = copy.deepcopy(doc)
    bad["header"]["dynamic_hazards_header"]["runtime_rule"]["constants"]["release_margin_s"] = 5.0
    assert any("dynamic_hazards_header.runtime_rule differs" in e for e in tje._check(bad))


def test_erased_detection_is_caught():
    # REV T8: a self-consistent lie about the detection (all metric-A fields erased, re-hashed): the detected
    # sample count and the exposure record still carry the detection (G23c F10)
    lie = _payload()
    lie["payload"]["runtime"].update(finite_ttc_at_first_detection=False, realized_ttc_at_first_detection_s=None,
                                     realized_ttc_at_first_detection_over_t_avoid=None, first_detection_s=None,
                                     non_exposure_reason="not_detected_at_sampled_times")
    for variant in ("as_is", "samples_zeroed"):
        bad = copy.deepcopy(lie)
        if variant == "samples_zeroed":
            bad["payload"]["runtime"]["detected_samples"] = 0
        try:
            d7_exposure.runtime_exposure(bad)
            raise AssertionError("erased detection accepted (%s)" % variant)
        except d7_exposure.ExposureError:
            pass


def test_legacy_certificate_without_runtime_rule_is_not_checked():
    doc = _doc(2)
    doc["header"]["dynamic_hazards_header"].pop("exposure_summary")
    doc["header"]["dynamic_hazards_header"].pop("runtime_rule")
    import validate_cert_v3
    assert any("exposure_summary is missing" in e for e in validate_cert_v3.dynamic_exposure_errors(doc))
    for index in range(len(doc["records"])):          # v2.5 stamps: no runtime_rule, no release block
        doc = tje.restamp(doc, index, lambda p: [p["runtime"].pop(k) for k in (
            "runtime_rule", "release", "finite_ttc_at_first_detection", "exposure_record",
            "realized_ttc_at_first_detection_over_t_avoid")])
    assert validate_cert_v3.dynamic_exposure_errors(doc) == []


def test_header_rule_requires_the_rule_block_in_every_admitted_record():
    # REV-G23c T15 / T16 (G23d H5): the header declares the default (non-diagnostic) rule while the admitted D1..D4
    # records carry no rule block (v2.5-shaped, or stripped of the rule fields and re-hashed), with or without a
    # summary: rejected
    import validate_cert_v3
    for keep_summary in (True, False):
        doc = _doc(2)
        if not keep_summary:
            doc["header"]["dynamic_hazards_header"].pop("exposure_summary")
        for index in range(len(doc["records"])):
            doc = tje.restamp(doc, index, lambda p: [p["runtime"].pop(k, None) for k in (
                "runtime_rule", "release", "finite_ttc_at_first_detection", "exposure_record",
                "realized_ttc_at_first_detection_over_t_avoid")])
        errors = validate_cert_v3.dynamic_exposure_errors(doc)
        assert any("lacks the header's runtime_rule" in e for e in errors), (keep_summary, errors)


def _payload(dose=2, released=True, first=10.0, ttc=5.0, reason=None, cause=CAUSE, release_time=8.0):
    release = None if dose == 0 else {
        "rule": RULE, "plan_kind": "transit", "applies": True, "t_hold_s": 8.0,
        "release_time_s": release_time if released else None, "release_tick_time_s": release_time if released else None,
        "cause": cause if released else None, "released": released,
        # G23d L4 (H2): the independent dose fields (G23e layer B: D2 = 3 x t_avoid 2.5 = 7.5 s; the v3 reader checks
        # dose_ttc_s = ratio x t_avoid, which the earlier 5 s (ratio 2) did not satisfy)
        "dose_ttc_s": 7.5, "ttc_at_reveal_s": 7.5, "ttc_cv_at_reveal_s": 7.5 if released else None,
        "dose_achieved": bool(released), "dose_achieved_tol": 0.05, "clamped": False,
        # G23e layer B: the v3 independent-measure fields (the default rule is d7_reveal_sync_v3)
        "ttc_independent_at_reveal_s": 7.5 if released else None, "dose_undefined": not released,
        "independent_measure": "ttc_acceleration_aware_heading", "cpa": None}
    record = None if dose == 0 else {
        "schema": d7_exposure.EXPOSURE_RECORD_SCHEMA, "first_visible_s": first,
        "first_ray_return_s": first, "closest_after_moving": None,
        "closest_after_visible": None if first is None else {"time_s": 20.0, "surface_gap_m": -0.1,
                                                             "along_track_m": 1.0, "cross_track_m": 0.0},
        "designed_contact_s": None, "sync_error_s": None, "unsynchronizable_slack_s": None}
    return {"payload": {"assignment": {"dose": dose, "task": "reach", "t_avoid_s": 2.5, "key": {}, "intruders": []},
                        "runtime": {"runtime_rule": None if dose == 0 else {"version": RULE},
                                    "first_detection_s": first, "detected_samples": 0 if first is None else 3,
                                    "realized_ttc_at_first_detection_s": ttc,
                                    "finite_ttc_at_first_detection": ttc is not None,
                                    "realized_ttc_at_first_detection_over_t_avoid": None if ttc is None else ttc / 2.5,
                                    "non_exposure_reason": reason, "release": release, "exposure_record": record}},
            "sha256": "0" * 64}


def test_runtime_exposure_reader_and_summary():
    ok = d7_exposure.runtime_exposure(_payload())
    assert ok["finite_ttc"] and ok["ttc_over_t_avoid"] == 2.0 and ok["release_cause"] == CAUSE
    assert ok["contact_after_visible"] is True and ok["ray_return_lag_s"] == 0.0
    quiet = d7_exposure.runtime_exposure(_payload(released=False, first=None, ttc=None, reason=UNRELEASED))
    assert quiet["released"] is False and quiet["non_exposure_reason"] == UNRELEASED
    seen = d7_exposure.runtime_exposure(_payload(ttc=None))      # detected, constant-velocity TTC infinite
    assert seen["detected"] and not seen["finite_ttc"]
    for bad in (_payload(first=7.0),                                          # detected before the release
                _payload(released=False, first=None, ttc=None, reason="not_detected_at_sampled_times"),
                _payload(reason=UNRELEASED),                              # detected yet "not released"
                _payload(cause="teleport")):
        try:
            d7_exposure.runtime_exposure(bad)
        except d7_exposure.ExposureError:
            continue
        raise AssertionError("accepted %r" % bad)
    tampered = _payload()
    tampered["payload"]["runtime"]["exposure_record"] = None
    try:
        d7_exposure.runtime_exposure(tampered)
        raise AssertionError("missing exposure record accepted")
    except d7_exposure.ExposureError:
        pass
    cell = [_payload(), _payload(ttc=7.5), _payload(ttc=None),
            _payload(released=False, first=None, ttc=None, reason=UNRELEASED)]
    summary = d7_exposure.exposure_summary(cell, 2)
    assert (summary["admitted"], summary["released"], summary["detected"],
            summary["diagnostic"]["finite_ttc_at_detection"]) == (4, 3, 3, 2)
    assert summary["diagnostic"]["finite_ttc_at_detection_share"] == 0.5 and summary["released_share"] == 0.75
    assert summary["diagnostic"]["ttc_over_t_avoid_at_detection"]["median"] == 2.5
    assert summary["dose_achieved"] == 3 and summary["dose_achieved_share"] == 0.75   # G23d H8 headline
    assert summary["rule_ttc_at_first_detection_over_t_avoid"]["median"] == (7.5 - 2.0) / 2.5   # G23e layer B: rule TTC 7.5 s (D2 = 3 x 2.5)
    lying = _payload()
    lying["payload"]["runtime"]["release"]["dose_achieved"] = False     # disagrees with ttc_cv / dose_ttc
    try:
        d7_exposure.runtime_exposure(lying)
        raise AssertionError("inconsistent dose_achieved accepted")
    except d7_exposure.ExposureError:
        pass
    assert summary["realized"]["contact_after_visible"] == 3
    assert summary["outcomes"] == {"detected": 3, UNRELEASED: 1}
    d0 = d7_exposure.exposure_summary([_payload(dose=0, first=None, ttc=None, reason="zero_dose")], 0)
    assert d0["diagnostic"]["finite_ttc_at_detection_share"] is None and d0["admitted"] == 1
    legacy = _payload()
    legacy["payload"]["runtime"].pop("runtime_rule")
    assert d7_exposure.runtime_exposure(legacy)["rule"] is None


def run_all():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print("TEST " + name, flush=True)
            function()
    print("ALL G23b/G23c EXPOSURE TESTS PASSED", flush=True)


if __name__ == "__main__":
    run_all()
