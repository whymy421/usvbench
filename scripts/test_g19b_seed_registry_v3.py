"""G19b (IMPLEMENTATION_PLAN P7): seed registry v3 - selection modes (O1), the training_smoke namespace,
resolve_training_block, the B9 fallback and gate ledger version 2 (version 1 readable). CPU only.

Run: python -B -m pytest -p no:cacheprovider scripts/test_g19b_seed_registry_v3.py -q
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


seeds = load("seed_registry")
TASK = "Isaac-USV-Journal-Scatter-BlueBoat-v1"
SHA = {c: c * 64 for c in "abcdef"}


def frozen(registry):
    out = copy.deepcopy(registry)
    out["status"] = "frozen"
    out["competence_gate"]["status"] = "frozen"
    return seeds.validate_registry(out)

def proposal(registry):
    """G52 (v2.6.6): the pre-freeze state of a registry (status proposal, gate pending), for the refusal tests of
    an unfrozen registry; the in-tree registry is frozen since v2.6.6 (owner 2026-10-14)."""
    out = copy.deepcopy(registry)
    out["status"] = "proposal"
    out["competence_gate"]["status"] = "OWNER_DECISION_PENDING"
    return seeds.validate_registry(out)


def record(registry, seed, successes=None, status="completed", ledger_version=1, training_sha=SHA["a"],
           mixture_sha=SHA["b"]):
    gate = registry["competence_gate"]
    evidence = None
    if successes is not None:
        evidence = {"path": "gate.json", "sha256": SHA["d"], "seed_namespace": gate["seed_namespace"],
                    "seed_block": gate["seed_block"], "episodes": gate["episodes"], "num_envs": gate["num_envs"],
                    "successes": successes, "complete": True}
    out = {"task_id": TASK, "vessel": "blueboat", "training_seed": seed,
           "candidate_index": registry["ppo_training"]["candidate_order"].index(seed),
           "training": {"status": status, "budget_timesteps": 96000, "completed_timesteps": 96000, "note": ""},
           "checkpoint": {"path": "final.pt", "sha256": SHA["c"]}, "gate_certificate": evidence,
           "verdict": None, "reason": None}
    if ledger_version == 2:
        out["training_record_sha256"] = training_sha
        out["mixture_sha256"] = mixture_sha
    out["verdict"], out["reason"] = seeds.gate_verdict(gate, out)
    return out


def ledger(registry, records, version=1):
    out = seeds.empty_ledger(registry, version)
    out["records"] = records
    return out


# ------------------------------------------------------------------ registry file

def test_registry_v3_encodes_mode_b_proposal_and_smoke_namespace():
    registry = seeds.load_registry()
    assert registry["registry_version"] == 3 == seeds.REGISTRY_VERSION
    training = registry["ppo_training"]
    assert training["selection_mode"] == "fixed_first_n"
    assert training["selection_mode_status"] == ("owner confirmed 2026-10-03 17:0x (decision record 09, decision "
                                                 "pack 02 item 6)")
    note = registry["owner_decisions_encoded"]["ppo_selection_update"]     # verify2: the owner's own record
    assert "controller message" not in json.dumps(registry)
    assert "NC\u671f\u520a/09_\u51b3\u7b56\u8bb0\u5f55.md" in note and "item 6" in note and "17:0x" in note
    assert training["fallback_rule"] == "first_in_order_flagged" and "exhaustion_policy" not in training
    assert set(training["selection_modes"]) == set(seeds.SELECTION_MODES)
    smoke = registry["namespaces"]["training_smoke"]
    assert smoke["purpose"] == "training_smoke"
    assert [b["seed"] for b in smoke["blocks"]] == [90001, 90002]
    every = [b["seed"] for entry in registry["namespaces"].values() for b in entry["blocks"]]
    assert len(every) == len(set(every))
    for seed in (90001, 90002):
        assert seeds.namespace_for_seed(seed, registry) == "training_smoke"
    # G52 (v2.6.6): the owner froze the registry and the competence gate on 2026-10-14 (threshold 0.2 unchanged)
    assert registry["status"] == "frozen" and registry["competence_gate"]["status"] == "frozen"
    assert registry["competence_gate"]["min_success_rate"] == 0.2


def test_version_2_registries_stay_readable_and_v3_fields_are_required():
    registry = seeds.load_registry()
    v2 = copy.deepcopy(registry)
    v2["registry_version"] = 2
    del v2["namespaces"]["training_smoke"]
    for name in ("selection_mode", "selection_mode_status", "selection_modes", "fallback_rule",
                 "fallback_rule_text"):
        del v2["ppo_training"][name]
    v2["ppo_training"]["exhaustion_policy"] = "report the admitted count"
    seeds.validate_registry(v2)
    assert seeds.selection_mode(v2) == "first_passing"
    bad = copy.deepcopy(v2)
    bad["namespaces"]["training_smoke"] = copy.deepcopy(registry["namespaces"]["training_smoke"])
    with pytest.raises(ValueError, match="version 3"):
        seeds.validate_registry(bad)
    for name, value in (("selection_mode", "best_of"), ("selection_mode_status", ""), ("fallback_rule", "relax")):
        broken = copy.deepcopy(registry)
        broken["ppo_training"][name] = value
        with pytest.raises(ValueError, match=name):
            seeds.validate_registry(broken)
    broken = copy.deepcopy(registry)
    broken["ppo_training"]["exhaustion_policy"] = "x"
    with pytest.raises(ValueError, match="exhaustion_policy"):
        seeds.validate_registry(broken)
    broken = copy.deepcopy(registry)
    broken["namespaces"]["smoke_b"] = {"purpose": "training_smoke", "blocks": [{"id": "smoke-b", "seed": 90003}]}
    with pytest.raises(ValueError, match="training_smoke"):
        seeds.validate_registry(broken)
    broken = copy.deepcopy(registry)
    broken["namespaces"]["training_smoke"]["blocks"][0]["seed"] = 41041
    with pytest.raises(ValueError, match="reused"):
        seeds.validate_registry(broken)


# ------------------------------------------------------------------ resolve_training_block

def test_smoke_seeds_resolve_unfrozen_and_campaign_seeds_need_the_freeze():
    in_tree = seeds.load_registry()                       # G52: frozen since v2.6.6
    assert seeds.resolve_training_block(42, "training", in_tree)["registry_status"] == "frozen"
    registry = proposal(in_tree)
    stamp = seeds.resolve_training_block(90001, "training_smoke", registry)
    assert stamp["namespace"] == "training_smoke" and stamp["block_id"] == "smoke-90001"
    assert stamp["registry_status"] == "proposal" and stamp["selection_mode"] == "fixed_first_n"
    assert stamp["registry_sha256"] == seeds.registry_digest(registry)
    with pytest.raises(ValueError, match="frozen"):
        seeds.resolve_training_block(42, "training", registry)
    stamp = seeds.resolve_training_block(1001, "training", frozen(registry))
    assert stamp["candidate_index"] == 1 and stamp["registry_status"] == "frozen"
    assert stamp["gate_status"] == "frozen"


@pytest.mark.parametrize("seed,purpose", [
    (11001, "training"), (11001, "training_smoke"), (21001, "training"), (31001, "training_smoke"),
    (41041, "training"), (41161, "training_smoke"),           # evaluation seeds: never trained on
    (90001, "training"), (42, "training_smoke"),               # the two training purposes stay apart
    (7, "training"), (123, "training_smoke"), (55555, "training"),   # legacy / undeclared
])
def test_non_training_seeds_are_refused(seed, purpose):
    with pytest.raises(ValueError):
        seeds.resolve_training_block(seed, purpose, frozen(seeds.load_registry()))


def test_training_purposes_only_and_smoke_is_never_an_evaluation_block():
    registry = seeds.load_registry()
    for purpose in seeds.EVALUATION_PURPOSES + ("diagnostic",):
        with pytest.raises(ValueError, match="training purpose"):
            seeds.resolve_training_block(90001, purpose, registry)
    for purpose in seeds.EVALUATION_PURPOSES:
        with pytest.raises(ValueError):
            seeds.resolve_seed_block("smoke-90001", "training_smoke", purpose, registry)
    with pytest.raises(ValueError):
        seeds.resolve_seed_block("smoke-90001", "training_smoke", "training_smoke", registry)
    # a labelled diagnostic evaluation of a smoke checkpoint stays possible (as for seed 42)
    stamp = seeds.diagnostic_registration(90001, registry)
    assert stamp["purpose"] == "diagnostic" and stamp["also_registered_as"] == "training_smoke"


# ------------------------------------------------------------------ selection modes

def test_mode_a_fallback_two_of_ten_pass():
    registry = seeds.load_registry()
    order = registry["ppo_training"]["candidate_order"]
    passing = {3001: 40, 7001: 90}
    records = [record(registry, s, successes=passing.get(s, 2)) for s in order if s != 1001]
    records.insert(1, record(registry, 1001, status="diverged"))
    result = seeds.admission(ledger(registry, records), registry, TASK, "blueboat")
    assert result["status"] == "exhausted"
    assert [a["training_seed"] for a in result["admitted"]] == [3001, 7001]
    fallback = result["fallback"]
    assert fallback["rule"] == "first_in_order_flagged" and fallback["complete"] is True
    assert [(d["training_seed"], d["flag"]) for d in fallback["delivered"]] == \
        [(42, "below_gate"), (2003, "below_gate"), (3001, "pass")]
    out = seeds.delivery(ledger(registry, records), registry, TASK, "blueboat", mode="first_passing")
    assert out["mode"] == "first_passing" and out["status"] == "exhausted"
    assert out["delivered"] == fallback["delivered"]
    # a version-2 registry has no fallback: only the admitted seeds are reported
    v2 = copy.deepcopy(registry)
    v2["registry_version"] = 2
    del v2["namespaces"]["training_smoke"]
    for name in ("selection_mode", "selection_mode_status", "selection_modes", "fallback_rule",
                 "fallback_rule_text"):
        del v2["ppo_training"][name]
    records_v2 = copy.deepcopy(records)
    assert "fallback" not in seeds.admission(ledger(v2, records_v2), v2, TASK, "blueboat")
    out = seeds.delivery(ledger(v2, records_v2), v2, TASK, "blueboat")
    assert [d["training_seed"] for d in out["delivered"]] == [3001, 7001]


def test_mode_a_complete_and_pending():
    registry = seeds.load_registry()
    records = [record(registry, 42, successes=80), record(registry, 1001, successes=3),
               record(registry, 2003, successes=26), record(registry, 3001, successes=100)]
    out = seeds.delivery(ledger(registry, records), registry, TASK, "blueboat", mode="first_passing")
    assert out["status"] == "complete"
    assert [(d["training_seed"], d["flag"]) for d in out["delivered"]] == [(42, "pass"), (2003, "pass"),
                                                                        (3001, "pass")]
    assert [d["training_seed"] for d in out["dead"]] == [1001]
    pending = seeds.delivery(ledger(registry, records[:2]), registry, TASK, "blueboat", mode="first_passing")
    assert pending["status"] == "pending" and pending["waiting_on_candidate"] == 2003
    assert [(d["training_seed"], d["flag"]) for d in pending["delivered"]] == [(42, "pass")]   # admission is final


def test_mode_b_train_three_deliver_three_flags_only():
    registry = seeds.load_registry()
    assert seeds.selection_mode(registry) == "fixed_first_n"
    records = [record(registry, 42, successes=2), record(registry, 1001, status="infrastructure_failure")]
    out = seeds.delivery(ledger(registry, records), registry, TASK, "blueboat")
    assert out["mode"] == "fixed_first_n" and out["status"] == "pending"
    assert out["waiting_on_candidate"] == 1001                     # rerun the same seed after the failure
    assert [(d["training_seed"], d["flag"]) for d in out["delivered"]] == [(42, "below_gate")]   # no selection
    records += [record(registry, 1001, successes=70), record(registry, 2003, status="diverged"),
                record(registry, 3001, successes=128)]
    out = seeds.delivery(ledger(registry, records), registry, TASK, "blueboat")
    assert out["status"] == "complete"
    assert [(d["training_seed"], d["candidate_index"], d["flag"]) for d in out["delivered"]] == \
        [(42, 0, "below_gate"), (1001, 1, "pass"), (2003, 2, "diverged")]
    assert out["records_beyond_rule"] == [3001]          # trained anyway: reported, never delivered
    with pytest.raises(ValueError, match="selection mode"):
        seeds.delivery(ledger(registry, records), registry, TASK, "blueboat", mode="best_three")


# ------------------------------------------------------------------ ledger v2

def test_ledger_v2_round_trip_and_v1_readable():
    registry = seeds.load_registry()
    v2 = ledger(registry, [record(registry, s, successes=50, ledger_version=2) for s in (42, 1001, 2003)], 2)
    text = json.dumps(v2, sort_keys=True)
    back = json.loads(text)
    seeds.validate_ledger(back, registry)
    out = seeds.delivery(back, registry, TASK, "blueboat")
    assert all(d["training_record_sha256"] == SHA["a"] and d["mixture_sha256"] == SHA["b"]
               for d in out["delivered"])
    v1 = ledger(registry, [record(registry, s, successes=50) for s in (42, 1001, 2003)], 1)
    seeds.validate_ledger(v1, registry)
    assert seeds.delivery(v1, registry, TASK, "blueboat")["status"] == "complete"
    mixed = ledger(registry, [record(registry, 42, successes=50)], 2)          # v1 record in a v2 ledger
    with pytest.raises(ValueError, match="exactly the fields"):
        seeds.validate_ledger(mixed, registry)
    missing = ledger(registry, [record(registry, 42, successes=50, ledger_version=2, training_sha=None)], 2)
    with pytest.raises(ValueError, match="training_record_sha256"):
        seeds.validate_ledger(missing, registry)
    bad = ledger(registry, [record(registry, 42, successes=50, ledger_version=2, mixture_sha="xyz")], 2)
    with pytest.raises(ValueError, match="mixture_sha256"):
        seeds.validate_ledger(bad, registry)
    infra = ledger(registry, [record(registry, 42, status="infrastructure_failure", ledger_version=2,
                                     training_sha=None, mixture_sha=None)], 2)
    seeds.validate_ledger(infra, registry)                       # no record for an interrupted run
    with pytest.raises(ValueError, match="version"):
        seeds.validate_ledger(dict(v2, ledger_version=3), registry)
    with pytest.raises(ValueError):
        seeds.empty_ledger(registry, 3)


def test_calm_arm_runs_in_a_v2_ledger():
    """Follow-up m9: a calm-arm (smoke control) run has a training record but no mixture."""
    registry = seeds.load_registry()
    rec = record(registry, 42, successes=50, ledger_version=2, mixture_sha=None)
    with pytest.raises(ValueError, match="mixture_sha256"):
        seeds.validate_ledger(ledger(registry, [rec], 2), registry)
    rec["training"]["arm"] = "calm"
    seeds.validate_ledger(ledger(registry, [rec], 2), registry)
    with_mixture = record(registry, 1001, successes=50, ledger_version=2)
    with_mixture["training"]["arm"] = "calm"
    with pytest.raises(ValueError, match="calm-arm"):
        seeds.validate_ledger(ledger(registry, [with_mixture], 2), registry)
    built = seeds.gate_record_from_certificate(
        None, registry=registry, task_id=TASK, vessel="blueboat", training_seed=42, training_status="diverged",
        budget_timesteps=96000, completed_timesteps=4000, ledger_version=2, training_record_sha256=SHA["e"],
        mixture_sha256=None, arm="calm")
    assert built["training"]["arm"] == "calm"
    seeds.validate_ledger(ledger(registry, [built], 2), registry)


def test_gate_record_builder_v2():
    registry = seeds.load_registry()
    rec = seeds.gate_record_from_certificate(
        None, registry=registry, task_id=TASK, vessel="blueboat", training_seed=42, training_status="diverged",
        budget_timesteps=96000, completed_timesteps=4000, ledger_version=2, training_record_sha256=SHA["e"],
        mixture_sha256=SHA["f"])
    assert set(rec) == set(seeds.GATE_RECORD_FIELDS_V2) and rec["verdict"] == "fail"
    assert rec["training_record_sha256"] == SHA["e"] and rec["mixture_sha256"] == SHA["f"]
    old = seeds.gate_record_from_certificate(
        None, registry=registry, task_id=TASK, vessel="blueboat", training_seed=42, training_status="diverged",
        budget_timesteps=96000, completed_timesteps=4000)
    assert set(old) == set(seeds.GATE_RECORD_FIELDS)
    with pytest.raises(ValueError, match="SHA-256"):
        seeds.gate_record_from_certificate(
            None, registry=registry, task_id=TASK, vessel="blueboat", training_seed=42,
            training_status="diverged", budget_timesteps=96000, completed_timesteps=4000, ledger_version=2,
            training_record_sha256="nope")


def test_cli_resolve_training_ledger_init_and_delivery():
    def run(argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(io.StringIO()):
            code = seeds.main(argv)
        return code, stream.getvalue()
    code, text = run(["resolve-training", "--seed", "90002", "--purpose", "training_smoke"])
    assert code == 0 and json.loads(text)["block_id"] == "smoke-90002"
    assert run(["resolve-training", "--seed", "41041", "--purpose", "training_smoke"])[0] == 2
    assert run(["resolve-training", "--seed", "42", "--purpose", "training"])[0] == 0    # G52: frozen in-tree registry
    code, text = run(["ledger-init"])
    assert code == 0 and json.loads(text)["ledger_version"] == 2
    registry = seeds.load_registry()
    root = SCRIPTS.parents[1] / "test_scratch"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g19b-reg-", dir=root) as directory:
        path = Path(directory) / "ledger.json"
        path.write_text(json.dumps(ledger(registry, [record(registry, s, successes=9, ledger_version=2)
                                                     for s in (42, 1001, 2003)], 2)), encoding="ascii")
        code, text = run(["delivery", "--ledger", str(path), "--task", TASK, "--vessel", "blueboat"])
        assert code == 0
        assert [d["flag"] for d in json.loads(text)["delivered"]] == ["below_gate"] * 3
        code, text = run(["delivery", "--ledger", str(path), "--task", TASK, "--vessel", "blueboat",
                          "--selection-mode", "first_passing"])
        assert code == 0 and json.loads(text)["status"] == "pending"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
