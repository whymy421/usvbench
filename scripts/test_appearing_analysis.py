"""G3 phase 1b: analysis_input treats a not-admitted appearing pair as unexposed coverage (D27 analogue). CPU only.

The certificates are REAL journal certificates written by the REAL cert_v3 writer over the REAL mixin
(test_journal_evaluators.journal_document, 2 envs x 1 episode). The appearing channel is grafted onto
them with REAL appearing stamps: the REAL AppearingBridge finishes rows whose assignments come from a
scatter BlueBoat appearing pool (G20b: the 16 x 2 reference grid admitted in memory with the active ladder,
the production pools being in the pool bundle; one admitted key, one not-admitted key); the record's
``appearing_assignment`` hash is F0 pairing.appearing_primitive_hash of that stamp, the header gets the
appearing inventory, the pairing declaration and the pool's appearing header block. The grafted documents
pass validate_cert_v3 first. Checks:
* appearing treatment D0 vs D3: the admitted-key pair is "paired", the not-admitted-key pair is
  "unexposed" (coverage, never in the effect pairs returned by validate_pair);
* every other treatment (waves D1 vs D2 at appearing D3): both pairs are "paired" (the not-admitted
  appearing key hashes identically, D20 form);
* an appearing header that differs outside the dose under another treatment is refused; one certificate
  with the appearing suite and one without is refused.
Run: python -B scripts/test_appearing_analysis.py   (or pytest)
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import torch  # noqa: E402

import analysis_input  # noqa: E402
import test_journal_evaluators as tje  # noqa: E402
import validate_cert_v3  # noqa: E402

sys.path.insert(0, str(Path(_SCRIPTS).parent / "tasks" / "_shared" / "conditions" / "appearing" / "tests"))
import g3_fixtures as G  # noqa: E402

_, PAIRING, _ = validate_cert_v3.conditions_modules()
POOL = G.reference_pool("scatter", G.BB)
RECORD = {"manifest_sha256": "0" * 64, "sha256": "0" * 64}      # pin fields of the header block (format only)


def keys():
    admitted = next((e, k) for e, k in POOL.offered if POOL.is_admitted(e, k))
    rejected = next((e, k) for e, k in POOL.offered if not POOL.is_admitted(e, k))
    return admitted, rejected


def stamp(env, episode, dose):
    envelope = G.envelope(G.BB)
    bridge = G.runtime.AppearingBridge("scatter", 15, 1.0 / 120.0, envelope.top_speed_mps, envelope.footprint,
                                       envelope.t_avoid_s)
    bridge.reset_rows((env,), {env: POOL.assignment(G.key("scatter", env, episode), dose)})
    bridge.step({env: 0}, {env: G.dynamic.Pose(0.0, 0.0, 0.0, 0.0)})
    return bridge.finish(env, "timeout")


def graft(doc, dose):
    """Add the appearing channel (real stamps of the pinned pool) to a real journal certificate."""
    doc = copy.deepcopy(doc)
    header = doc["header"]
    primitives = sorted(set(header["pairing"]["primitives"]) | {PAIRING.APPEARING_PRIMITIVE})
    header["pairing"] = PAIRING.pairing_declaration(primitives, header["pairing"]["changed_primitives"] or None)
    t_avoid = G.envelope(G.BB).t_avoid_s
    block = {"schema": "usvbench.appearing_header/1", "dose": dose, "pool_pinned": True,
             "pool_manifest_sha256": RECORD["manifest_sha256"], "pool_sha256": RECORD["sha256"],
             # G20c (REV-G20b M1): mandatory ladder fields of the header
             "suite": {"vessel_id": G.BB}, "suite_version": G.config.APPEARING_SUITE_VERSION,
             "ladder_option": G.config.LADDER_OPTION, "t_avoid_s": t_avoid,
             "trigger_gaps_m": G.config.trigger_gaps_m(t_avoid, G.BB), "hazard_group": G.config.hazard_group("scatter")}
    block.update(POOL.header_block())
    header["appearing_hazards_header"] = block
    for record, (env, episode) in zip(doc["records"], keys()):
        envelope = stamp(env, episode, dose)
        record["appearing_hazards"] = envelope
        record["scenario_hashes"][PAIRING.APPEARING_PRIMITIVE] = PAIRING.appearing_primitive_hash(envelope)
        record["scenario_hashes"] = dict(sorted(record["scenario_hashes"].items()))
    return doc


def declare(doc, treatment):
    doc = copy.deepcopy(doc)
    doc["header"]["pairing"] = PAIRING.pairing_declaration(doc["header"]["pairing"]["primitives"], treatment)
    return doc


def test_grafted_certificates_validate():
    for dose in (0, 3):
        doc = graft(tje.journal_document(("waves=D1",)), dose)
        assert not tje._check(doc), tje._check(doc)
        kinds = [analysis_input.appearing_record_exposure(r) for r in doc["records"]]
        assert kinds == ["admitted", "not_admitted"], kinds


def test_not_admitted_appearing_pair_is_unexposed_under_the_appearing_treatment():
    base = tje.journal_document(("waves=D1",))
    d0 = declare(graft(base, 0), "appearing")
    d3 = declare(graft(base, 3), "appearing")
    detail = analysis_input.validate_journal_pair_detailed(d0, d3, ("appearing",))
    assert detail["status"] == ["paired", "unexposed"], detail["status"]
    assert detail["appearing_treatment"] and detail["unexposed"] == 1
    assert detail["appearing_not_admitted"] == {"baseline": 1, "treatment": 1}
    assert [(a["env"], b["env"]) for a, b in detail["effect_pairs"]] == [(0, 0)]       # D27: coverage only
    assert [(a["env"], b["env"]) for a, b in analysis_input.validate_pair(d0, d3, ("appearing",))] == [(0, 0)]
    # the unexposed pair carries identical appearing hashes (dose-independent not-admitted digest)
    a, b = d0["records"][1], d3["records"][1]
    assert a["scenario_hashes"][PAIRING.APPEARING_PRIMITIVE] == b["scenario_hashes"][PAIRING.APPEARING_PRIMITIVE]


def test_not_admitted_appearing_pair_pairs_for_every_other_treatment():
    w1 = declare(graft(tje.journal_document(("waves=D1",)), 3), "waves")
    w2 = declare(graft(tje.journal_document(("waves=D2",)), 3), "waves")
    detail = analysis_input.validate_journal_pair_detailed(w1, w2, ("waves",))
    assert detail["status"] == ["paired", "paired"], detail["status"]
    assert not detail["appearing_treatment"] and detail["unexposed"] == 0
    assert len(analysis_input.validate_pair(w1, w2, ("waves",))) == 2


def test_appearing_header_and_presence_mismatches_are_refused():
    w1 = declare(graft(tje.journal_document(("waves=D1",)), 3), "waves")
    w2 = declare(graft(tje.journal_document(("waves=D2",)), 2), "waves")       # appearing dose differs too
    try:
        analysis_input.validate_journal_pair_detailed(w1, w2, ("waves",))
    except analysis_input.AnalysisInputError as exc:
        assert "appearing headers differ" in str(exc), exc
    else:
        raise AssertionError("an appearing dose difference under the waves treatment must be refused")
    plain = declare(tje.journal_document(("waves=D2",)), "waves")
    try:
        analysis_input.validate_journal_pair_detailed(w1, plain, ("waves",))
    except analysis_input.AnalysisInputError:
        pass
    else:
        raise AssertionError("one certificate with and one without the appearing suite must be refused")


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(1 if run_all() else 0)
