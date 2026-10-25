"""Round T D20 (CONTEXT_R4): the D16 reader against REAL family not-admitted stamps. CPU only.

The stamps in scripts/fixtures_d16/<family>.json were produced by each family's own code on
the merged tree (scripts/make_d16_fixtures.py --write; scripts/test_d16_fixtures_fresh.py
re-runs the family code and compares). Per family (hazard_nav, path_hazard,
path_following, docking, harbor_mission):

* d7_exposure.exposure_of returns ("not_admitted", the family's E4 reasons without the
  joint echo "joint_ladder_member_rejected", de-duplicated, first-seen order), read from the
  location this family uses;
* d7_exposure.d7_primitive equals F0 pairing.d7_primitive_hash and the family env's own
  scenario_hashes["d7_assignment"];
* the REAL cert_v3 writer accepts the record and validate_cert_v3 accepts the minimal
  journal certificate containing it;
* analysis_input reports the pair "unexposed" under the dynamic treatment and "paired"
  under the waves, wind, current, actuator and perception treatments, and (CONTEXT_R4 D27)
  the unexposed pair is coverage, not part of the effect pairs returned by validate_pair;
* one negative test per D20 rule on the family's stamp: disagreeing claims, non-zero
  dose, intruders present, no reason left after dropping the echo, an envelope with a
  third key (plus a missing pairing key and an unreadable reason item).

Run: python -B scripts/test_d16_real_stamps.py   (or pytest)
"""
from __future__ import annotations

import copy
import os
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import torch  # noqa: E402

import analysis_input  # noqa: E402
import d7_exposure  # noqa: E402
import test_journal_evaluators as tje  # noqa: E402
import validate_cert_v3  # noqa: E402

FAMILIES = tje.D16_FAMILIES
ECHO = d7_exposure.JOINT_ECHO
OTHER_TREATMENTS = ("waves", "wind", "current", "actuator", "perception")


def _modules():
    certificate, pairing, d7 = validate_cert_v3.conditions_modules()
    return certificate, pairing, d7


def _dedup(names):
    out = []
    for name in names:
        if name != ECHO and name not in out:
            out.append(name)
    return tuple(out)


def expected_reasons(family, payload):
    """The family's E4 reasons read from the family's OWN field (independent of d7_exposure)."""
    if family == "hazard_nav":          # journal_hazard_nav.not_admitted_stamp: [dose, reason, n] rows
        return _dedup(row[1] for row in payload["not_admitted"]["reasons"])
    if family == "path_hazard":         # NotAdmittedBridge.finish: [dose, reason, n] rows
        return _dedup(row[1] for row in payload["not_admitted_reasons"])
    if family == "path_following":      # not_admitted_stamp: assignment.rejection_reasons (strings)
        return _dedup(payload["assignment"]["rejection_reasons"])
    if family == "docking":             # journal_d7_exposure.not_admitted_stamp: {dose, reason, sampled_states}
        return _dedup(item["reason"] for item in payload["not_admitted"]["reasons"])
    if family == "harbor_mission":      # NotAdmitted.stamp_assignment: {dose, reason, sampled_states}
        return _dedup(item["reason"] for item in payload["assignment"]["not_admitted_reasons"])
    raise KeyError(family)


def stamp_of(family):
    return tje.d16_fixture(family)["stamp"]


def redigest(payload):
    _, _, d7 = _modules()
    return {"payload": payload, "sha256": d7.digest(payload)}


def mutated(family, change):
    payload = copy.deepcopy(stamp_of(family)["payload"])
    change(payload)
    return redigest(payload)


def reason_location(payload):
    """(parent dict, key) of the first non-empty reason location (the one the reader uses)."""
    for path in d7_exposure.REASON_SOURCES:
        block = payload
        for name in path[:-1]:
            block = block.get(name) if isinstance(block, dict) else None
        if isinstance(block, dict) and block.get(path[-1]):
            return block, path[-1]
    raise AssertionError("no reason location")


def replace_reason(item, name):
    """The family's item shape with its reason string replaced by ``name``."""
    if isinstance(item, str):
        return name
    if isinstance(item, dict):
        return dict(item, reason=name)
    return [name if isinstance(x, str) else x for x in item]


def _raises(text, function, *args):
    try:
        function(*args)
    except d7_exposure.ExposureError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError("expected ExposureError containing %r" % text)


# ------------------------------------------------------------------ reader --
def test_fixtures_are_real_family_stamps_with_provenance():
    certificate, _, d7 = _modules()
    for family in FAMILIES:
        doc = tje.d16_fixture(family)
        assert doc["schema"] == "usvbench.t0.d16_fixture/1" and doc["family"] == family
        assert doc["generator"] == "scripts/make_d16_fixtures.py"
        assert doc["producer"].startswith("tasks/" + family + "/")
        assert all(len(v) == 64 for v in doc["source_sha256"].values()) and doc["source_sha256"]
        stamp = doc["stamp"]
        certificate.verify_dynamic_hazard_stamp(stamp)            # E3 digest check
        d7.read_stamp(stamp)                                     # E4 schema/model version check
        assert set(stamp) == {"payload", "sha256"}


def test_exposure_of_reads_every_family_form():
    for family in FAMILIES:
        payload = stamp_of(family)["payload"]
        kind, reasons = d7_exposure.exposure_of(stamp_of(family))
        assert kind == "not_admitted", family
        assert reasons and ECHO not in reasons, (family, reasons)
        assert reasons == expected_reasons(family, payload), (family, reasons)
        assert len(set(reasons)) == len(reasons)
        assert payload["assignment"]["dose"] == 0 and payload["assignment"].get("intruders", []) == []
        assert not d7_exposure.is_beyond_offered_grid(stamp_of(family))
    # path_hazard's real stamp carries the joint echo next to its reason rows: dropped, de-duplicated
    rows = stamp_of("path_hazard")["payload"]["not_admitted_reasons"]
    assert any(row[1] == ECHO for row in rows) and len(rows) > 1
    assert len(d7_exposure.exposure_of(stamp_of("path_hazard"))[1]) == len({r[1] for r in rows} - {ECHO})
    # cross-checks against the families' own summaries of the same reasons
    hazard = stamp_of("hazard_nav")["payload"]
    assert hazard["not_admitted"]["primary_reason"] in d7_exposure.exposure_of(stamp_of("hazard_nav"))[1]
    harbour = stamp_of("harbor_mission")["payload"]
    assert set(harbour["assignment"]["not_admitted_failing_reasons"]) == set(
        d7_exposure.exposure_of(stamp_of("harbor_mission"))[1])
    docking = stamp_of("docking")["payload"]
    assert docking["not_admitted"]["top_reason"] in d7_exposure.exposure_of(stamp_of("docking"))[1]


def test_d7_primitive_is_the_f0_pairing_hash_of_the_family_stamp():
    _, pairing, _ = _modules()
    for family in FAMILIES:
        doc = tje.d16_fixture(family)
        value = d7_exposure.d7_primitive(pairing, doc["stamp"])
        assert value == pairing.d7_primitive_hash(doc["stamp"]), family
        assert value == doc["scenario_hashes"]["d7_assignment"], family      # the family env's own hash


# ------------------------------------------------------ writer + validator --
def test_writer_and_validator_accept_a_journal_certificate_with_the_record():
    _, pairing, _ = _modules()
    for family in FAMILIES:
        stamp = stamp_of(family)
        doc = tje.journal_document(("waves=D1",), not_admitted=(1,), stamp=stamp)
        assert doc["records"][1]["dynamic_hazards"] == stamp, family
        assert doc["records"][1]["scenario_hashes"]["d7_assignment"] == pairing.d7_primitive_hash(stamp)
        errors = tje._check(doc)
        assert not errors, (family, errors)
        assert analysis_input.dynamic_exposure(doc["records"][1]) == "not_admitted"


# ---------------------------------------------------------------- analysis --
def test_analysis_unexposed_for_dynamic_and_paired_for_every_other_suite():
    for family in FAMILIES:
        stamp = stamp_of(family)
        for suite in OTHER_TREATMENTS:
            base = tje.journal_document((suite + "=D1",), not_admitted=(1,), stamp=stamp)
            treated = tje.journal_document((suite + "=D2",), not_admitted=(1,), stamp=stamp)
            detail = analysis_input.validate_journal_pair_detailed(base, treated, (suite,))
            assert detail["status"] == ["paired", "paired"], (family, suite, detail["status"])
            assert detail["not_admitted"] == {"baseline": 1, "treatment": 1} and detail["unexposed"] == 0
            assert len(analysis_input.validate_pair(base, treated, (suite,))) == 2     # both rows in the effect
        d0 = tje.journal_document(("dynamic=D0",), not_admitted=(1,), stamp=stamp, treatment="dynamic")
        d2 = tje.journal_document(("dynamic=D2",), not_admitted=(1,), stamp=stamp, treatment="dynamic")
        assert not tje._check(d2), (family, tje._check(d2))
        detail = analysis_input.validate_journal_pair_detailed(d0, d2, ("dynamic",))
        assert detail["status"] == ["paired", "unexposed"], (family, detail["status"])
        assert detail["unexposed"] == 1 and detail["dynamic_treatment"]
        # CONTEXT_R4 D27: the not-admitted pair is coverage, never a zero difference in the effect
        assert [(a["env"], b["env"]) for a, b in detail["effect_pairs"]] == [(0, 0)]
        assert [(a["env"], b["env"]) for a, b in analysis_input.validate_pair(d0, d2, ("dynamic",))] == [(0, 0)]


# --------------------------------------------------------------- negatives --
def test_negative_disagreeing_claims():
    for family in FAMILIES:
        bad = mutated(family, lambda p: p["runtime"].__setitem__("exposure", "admitted"))
        _raises("disagree", d7_exposure.exposure_of, bad)
        bad = mutated(family, lambda p: p.__setitem__("exposure", "partly"))
        _raises("exposure must be one of", d7_exposure.exposure_of, bad)


def test_negative_non_zero_dose():
    for family in FAMILIES:
        for dose in (1, 4, 0.0, None):
            bad = mutated(family, lambda p: p["assignment"].__setitem__("dose", dose))
            _raises("dose 0", d7_exposure.exposure_of, bad)


def test_negative_intruders_present():
    for family in FAMILIES:
        for value in ([{"slot": 0}], None):
            bad = mutated(family, lambda p: p["assignment"].__setitem__("intruders", value))
            _raises("no intruders", d7_exposure.exposure_of, bad)
        # absent intruders are accepted (D20 b)
        assert d7_exposure.exposure_of(mutated(family, lambda p: p["assignment"].pop("intruders")))[0] == \
            "not_admitted"


def test_negative_no_reason_left_after_dropping_the_echo():
    for family in FAMILIES:
        def echo_only(payload):
            block, key = reason_location(payload)
            block[key] = [replace_reason(item, ECHO) for item in block[key]]
        _raises("joint-ladder echo", d7_exposure.exposure_of, mutated(family, echo_only))

        def no_reason(payload):
            for path in d7_exposure.REASON_SOURCES:
                block = payload
                for name in path[:-1]:
                    block = block.get(name) if isinstance(block, dict) else None
                if isinstance(block, dict) and path[-1] in block:
                    block[path[-1]] = []
        _raises("rejection reasons", d7_exposure.exposure_of, mutated(family, no_reason))

        def unreadable(payload):
            block, key = reason_location(payload)
            block[key] = [{"why": "x"}]
        _raises("reason", d7_exposure.exposure_of, mutated(family, unreadable))


def test_negative_envelope_with_a_third_key():
    for family in FAMILIES:
        bad = copy.deepcopy(stamp_of(family))
        bad["exposure"] = "not_admitted"
        _raises("exactly {payload, sha256}", d7_exposure.exposure_of, bad)
        doc = tje.journal_document(("waves=D1",), not_admitted=(1,), stamp=stamp_of(family))
        doc["records"][1]["dynamic_hazards"]["exposure"] = "not_admitted"
        assert tje._check(doc), family


def test_negative_missing_pairing_key_and_validator_rejection():
    for family in FAMILIES:
        for field in d7_exposure.NOT_ADMITTED_ASSIGNMENT_KEYS:
            bad = mutated(family, lambda p: p["assignment"].pop(field))
            _raises(field, d7_exposure.exposure_of, bad)
        doc = tje.journal_document(("waves=D1",), not_admitted=(1,), stamp=stamp_of(family))
        bad = tje.restamp(copy.deepcopy(doc), 1, lambda p: p["assignment"].__setitem__("dose", 2))
        assert any("dynamic_hazards envelope rejected" in e for e in tje._check(bad)), family


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                with patch.dict("os.environ", {}, clear=False):
                    function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(1 if run_all() else 0)
