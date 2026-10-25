"""Pre-declared, disjoint seed namespaces and the PPO seed-admission rule.

Pure Python; imports no simulator package. The registry file
(seed_registry.json) declares, BEFORE any journal training run:

  * training      the ordered PPO training-seed candidates (owner, 2026-09-30:
                  PPO is the only controller with training seeds; 3 per task
                  and vessel; 42 stays a candidate but is retrained);
  * selection     validation blocks; 'selection-gate-01' carries the
                  predeclared competence gate;
  * dose_pilots, crossmachine_calibration, final_test
                  evaluation blocks, never chosen by performance;
  * training_smoke (registry v3, G19b) diagnostic training-smoke seeds
                  (90001, 90002; TRAINING_DESIGN section 7): never a campaign
                  candidate, never an evaluation seed, usable while the
                  registry is a proposal (resolve_training_block).

Registry v3 (G19b, IMPLEMENTATION_PLAN P7) adds to ppo_training:
  selection_mode   "first_passing" (Mode A: first admit_first_passing passing
                   candidates; B9 fallback "first_in_order_flagged") or
                   "fixed_first_n" (Mode B: train the first N, deliver them,
                   each flagged pass / below_gate); owner decision O1:
                   "fixed_first_n" (owner 10-01 10:1x), CONFIRMED by the
                   owner 2026-10-03 17:0x (decision pack 02 item 6: train 3,
                   deliver 3, no picking); G28 adds the structured
                   selection_mode_confirmation {status owner_confirmed,
                   confirmed_mode, rule, by, date, source, freeze_implied
                   false}: validated, its confirmed_mode must equal
                   selection_mode (a mode change needs a new confirmation),
                   and it never freezes the registry or the gate;
  fallback_rule    "first_in_order_flagged" (replaces exhaustion_policy).
Version-2 registries stay readable (they behave as "first_passing" without a
fallback). delivery() applies the registry's mode; admission() is the Mode A
rule unchanged (plus a 'fallback' block when the candidates are exhausted).

Every seed integer appears in exactly one namespace, so a final-test block can
never be relabelled as a selection or pilot run (run_cell.py and
eval_protocol.py refuse such calls), and no evaluation seed equals a training
seed.

Gate ledger (one JSON file per study; see GATE_RECORD_FIELDS):
  {"ledger_version": 1|2, "registry_sha256": "...", "gate_sha256": "...",
   "records": [record, ...]}
Version 2 (G19b) records carry two more fields, "training_record_sha256" and
"mixture_sha256" (GATE_RECORD_FIELDS_V2; SHA-256 of the PPO training record
and the training mixture digest); version-1 ledgers stay readable.
Each record is one training attempt of one candidate on one (task, vessel):
  {"task_id": str, "vessel": "blueboat"|"wamv", "training_seed": int,
   "candidate_index": int,
   "training": {"status": "completed"|"diverged"|"infrastructure_failure",
                "budget_timesteps": int, "completed_timesteps": int,
                "note": str},
   "checkpoint": {"path": str|null, "sha256": str|null},
   "gate_certificate": null | {"path": str, "sha256": str,
                "seed_namespace": "selection", "seed_block": str,
                "episodes": int, "num_envs": int, "successes": int,
                "complete": bool},
   "verdict": "pass"|"fail"|"invalid", "reason": str}
The verdict is recomputed from the gate definition by gate_verdict(); a ledger
whose stored verdict differs is rejected (no hand-edited admissions). Use
'gate-record' to build a record from a validated v3 gate certificate.

CLI:
  seed_registry.py show
  seed_registry.py resolve --block final-01 --namespace final_test --purpose final_test
  seed_registry.py gate-record --certificate gate.json --task T --vessel V
        --training-seed 42 --training-status completed --budget-timesteps N
        --completed-timesteps N [--checkpoint ckpt.pt]
  seed_registry.py admission --ledger ledger.json --task T --vessel V
  seed_registry.py delivery --ledger ledger.json --task T --vessel V [--selection-mode M]
  seed_registry.py resolve-training --seed 90001 --purpose training_smoke
  seed_registry.py ledger-init [--ledger-version 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


DEFAULT_REGISTRY = Path(__file__).with_name("seed_registry.json")
REGISTRY_VERSION = 3
REGISTRY_VERSIONS = (2, 3)            # G19b: version-2 registries stay readable
PURPOSES = ("training", "training_smoke", "selection", "pilot", "calibration", "final_test")
EVALUATION_PURPOSES = ("selection", "pilot", "calibration", "final_test")
TRAINING_PURPOSES = ("training", "training_smoke")
SELECTION_MODES = ("first_passing", "fixed_first_n")
FALLBACK_RULES = ("first_in_order_flagged",)
VESSELS = ("blueboat", "wamv")
TRAINING_STATUSES = ("completed", "diverged", "infrastructure_failure")
VERDICTS = ("pass", "fail", "invalid")
GATE_RECORD_FIELDS = ("task_id", "vessel", "training_seed", "candidate_index", "training",
                      "checkpoint", "gate_certificate", "verdict", "reason")
# G19b: ledger version 2 adds the training record and mixture digests (IMPLEMENTATION_PLAN P7)
GATE_RECORD_FIELDS_V2 = GATE_RECORD_FIELDS + ("training_record_sha256", "mixture_sha256")
LEDGER_FIELDS = {1: GATE_RECORD_FIELDS, 2: GATE_RECORD_FIELDS_V2}


class RegistryError(ValueError):
    """The registry, a seed use or a gate ledger violates the declared rules."""


def _strict_constant(value):
    raise RegistryError("non-finite JSON constant: " + value)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=_strict_constant, object_pairs_hook=_no_duplicates)


def digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def registry_digest(registry):
    return digest(registry)


def load_registry(path=None):
    """Read and validate the complete registry, including cross-namespace reuse."""
    registry = read_json(path or DEFAULT_REGISTRY)
    validate_registry(registry)
    return registry


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise RegistryError("%s must be an integer >= %d" % (label, minimum))
    return value


def validate_registry(registry):
    if not isinstance(registry, dict) or registry.get("registry_version") not in REGISTRY_VERSIONS:
        raise RegistryError("unsupported seed registry version")
    if not isinstance(registry.get("study_version"), str) or not registry["study_version"]:
        raise RegistryError("seed registry requires study_version")
    if registry.get("status") not in ("proposal", "frozen"):
        raise RegistryError("registry status must be 'proposal' or 'frozen'")
    namespaces = registry.get("namespaces")
    if not isinstance(namespaces, dict) or not namespaces:
        raise RegistryError("seed registry requires namespaces")
    excluded = registry.get("legacy_seeds_excluded", [])
    if not isinstance(excluded, list) or any(type(s) is not int for s in excluded):
        raise RegistryError("legacy_seeds_excluded must be a list of integers")
    seeds, ids = {}, set()
    purposes = {}
    for namespace, declaration in namespaces.items():
        if not isinstance(namespace, str) or not namespace or not isinstance(declaration, dict):
            raise RegistryError("invalid namespace declaration")
        purpose = declaration.get("purpose")
        if purpose not in PURPOSES:
            raise RegistryError("invalid namespace purpose: " + namespace)
        purposes.setdefault(purpose, []).append(namespace)
        blocks = declaration.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RegistryError("namespace has no declared blocks: " + namespace)
        for block in blocks:
            if not isinstance(block, dict):
                raise RegistryError("seed block must be an object")
            seed = _int(block.get("seed"), "seed")
            block_id = block.get("id")
            if seed in seeds:
                raise RegistryError("seed %s reused in %s and %s" % (seed, seeds[seed], namespace))
            if seed in excluded:
                raise RegistryError("legacy seed %s may not be declared" % seed)
            if not isinstance(block_id, str) or not block_id or block_id in ids:
                raise RegistryError("seed block IDs must be nonempty and globally unique")
            if block_id.isdigit():
                raise RegistryError("block IDs must not look like seed integers: " + block_id)
            seeds[seed] = namespace
            ids.add(block_id)
    if len(purposes.get("training", [])) != 1:
        raise RegistryError("exactly one training namespace is required")
    if purposes.get("training_smoke") and registry["registry_version"] < 3:
        raise RegistryError("a training_smoke namespace needs registry version 3")
    if len(purposes.get("training_smoke", [])) > 1:
        raise RegistryError("at most one training_smoke namespace is allowed")
    training = registry.get("ppo_training")
    if not isinstance(training, dict):
        raise RegistryError("ppo_training declaration required")
    order = training.get("candidate_order")
    if not isinstance(order, list) or not order or len(set(order)) != len(order):
        raise RegistryError("ppo_training.candidate_order must list unique seeds")
    declared = [b["seed"] for b in namespaces[purposes["training"][0]]["blocks"]]
    if declared != order:
        raise RegistryError("training namespace blocks must equal candidate_order, in order")
    admit = _int(training.get("admit_first_passing"), "admit_first_passing", 1)
    if admit > len(order):
        raise RegistryError("admit_first_passing exceeds the number of candidates")
    if registry["registry_version"] >= 3:
        if training.get("selection_mode") not in SELECTION_MODES:
            raise RegistryError("ppo_training.selection_mode must be one of " + ", ".join(SELECTION_MODES))
        if not isinstance(training.get("selection_mode_status"), str) or not training["selection_mode_status"]:
            raise RegistryError("ppo_training.selection_mode_status required (owner decision O1)")
        if training.get("fallback_rule") not in FALLBACK_RULES:
            raise RegistryError("ppo_training.fallback_rule must be one of " + ", ".join(FALLBACK_RULES))
        if "exhaustion_policy" in training:
            raise RegistryError("registry v3 replaces exhaustion_policy by fallback_rule")
        if "selection_mode_confirmation" in training:          # G28
            check_selection_confirmation(training)
    gate = registry.get("competence_gate")
    validate_gate_definition(gate, registry)
    if registry["status"] == "frozen" and gate.get("status") != "frozen":
        raise RegistryError("a frozen registry requires a frozen competence gate "
                            "(owner decision on the threshold)")
    controller = registry.get("controller_seeds", {})
    if not isinstance(controller, dict) or any(type(v) is not int for v in controller.values()):
        raise RegistryError("controller_seeds must map controller names to integers")
    for block in namespaces.get("final_test", {}).get("blocks", []):
        if block.get("cohort") not in ("main", "predeclared_topup", "reserve"):
            raise RegistryError("final_test blocks need cohort main/predeclared_topup/reserve")
    return registry


CONFIRMATION_STATUSES = ("owner_confirmed", "pending_owner_confirmation")
CONFIRMATION_FIELDS = ("status", "confirmed_mode", "rule", "by", "date", "source", "freeze_implied")


def check_selection_confirmation(training):
    """G28: ppo_training.selection_mode_confirmation (owner confirmation of the selection rule)."""
    confirmation = training["selection_mode_confirmation"]
    if not isinstance(confirmation, dict):
        raise RegistryError("ppo_training.selection_mode_confirmation must be an object")
    missing = [k for k in CONFIRMATION_FIELDS if k not in confirmation]
    if missing:
        raise RegistryError("ppo_training.selection_mode_confirmation lacks " + ", ".join(missing))
    if confirmation["status"] not in CONFIRMATION_STATUSES:
        raise RegistryError("selection_mode_confirmation.status must be one of " + ", ".join(CONFIRMATION_STATUSES))
    if confirmation["confirmed_mode"] != training.get("selection_mode"):
        raise RegistryError("selection_mode_confirmation.confirmed_mode %r differs from selection_mode %r: a changed "
                            "selection mode needs a new owner confirmation"
                            % (confirmation["confirmed_mode"], training.get("selection_mode")))
    for key in ("rule", "by", "date", "source"):
        if not isinstance(confirmation[key], str) or not confirmation[key]:
            raise RegistryError("selection_mode_confirmation.%s must be a nonempty string" % key)
    if confirmation["freeze_implied"] is not False:
        raise RegistryError("selection_mode_confirmation.freeze_implied must be false: confirming the selection rule "
                            "never freezes the registry or the competence gate")
    status_text = training.get("selection_mode_status", "")
    if confirmation["status"] == "owner_confirmed" and not status_text.startswith("owner confirmed"):
        raise RegistryError("selection_mode_status must say 'owner confirmed ...' when selection_mode_confirmation "
                            "is owner_confirmed")
    return confirmation


def validate_gate_definition(gate, registry):
    if not isinstance(gate, dict) or gate.get("gate_version") != 1:
        raise RegistryError("competence_gate version 1 required")
    namespace = gate.get("seed_namespace")
    declaration = registry["namespaces"].get(namespace)
    if not isinstance(declaration, dict) or declaration.get("purpose") != "selection":
        raise RegistryError("competence gate must use a selection namespace")
    if gate.get("seed_block") not in [b["id"] for b in declaration["blocks"]]:
        raise RegistryError("competence gate block is not declared in its namespace")
    episodes = _int(gate.get("episodes"), "gate episodes", 1)
    envs = _int(gate.get("num_envs"), "gate num_envs", 1)
    if episodes % envs:
        raise RegistryError("gate episodes must be divisible by gate num_envs")
    rate = gate.get("min_success_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise RegistryError("gate min_success_rate must be in [0, 1]")
    if gate.get("checkpoint") != "final":
        raise RegistryError("only the 'final' checkpoint gate is implemented")
    if gate.get("status") not in ("OWNER_DECISION_PENDING", "frozen"):
        raise RegistryError("gate status must be OWNER_DECISION_PENDING or frozen")
    return gate


def gate_min_successes(gate):
    """Integer success threshold: ceil(min_success_rate * episodes)."""
    return int(math.ceil(gate["min_success_rate"] * gate["episodes"] - 1e-9))


def namespace_for_seed(seed, registry=None):
    registry = load_registry() if registry is None else validate_registry(registry)
    for namespace, declaration in registry["namespaces"].items():
        if any(item["seed"] == seed for item in declaration["blocks"]):
            return namespace
    return None


def purpose_of(namespace, registry):
    declaration = registry["namespaces"].get(namespace)
    return None if declaration is None else declaration["purpose"]


def selection_mode(registry):
    """'first_passing' or 'fixed_first_n' (a version-2 registry is 'first_passing' without a fallback)."""
    return registry["ppo_training"].get("selection_mode", "first_passing")


def resolve_training_block(seed, purpose, registry=None):
    """G19b (P7): stamp of a TRAINING seed for a PPO training run (purpose 'training' or 'training_smoke').

    Refused: a purpose outside TRAINING_PURPOSES; a seed not declared in the namespace of that purpose (an
    evaluation seed is named with its namespace in the error); a campaign ('training') seed while the
    registry or its competence gate is not frozen (the smoke namespace is usable unfrozen).
    """
    registry = load_registry() if registry is None else validate_registry(registry)
    if purpose not in TRAINING_PURPOSES:
        raise RegistryError("training purpose must be one of " + ", ".join(TRAINING_PURPOSES))
    if type(seed) is not int:
        raise RegistryError("seed must be an integer")
    owner = namespace_for_seed(seed, registry)
    if owner is None:
        raise RegistryError("seed %d is not declared in the seed registry (purpose %s)" % (seed, purpose))
    owner_purpose = purpose_of(owner, registry)
    if owner_purpose != purpose:
        raise RegistryError("seed %d belongs to the %s namespace (purpose %s), not to %s; training never "
                            "uses evaluation or other-purpose seeds" % (seed, owner, owner_purpose, purpose))
    if purpose == "training" and (registry["status"] != "frozen"
                                  or registry["competence_gate"].get("status") != "frozen"):
        raise RegistryError("campaign training seeds need a frozen registry and competence gate (status %r, "
                            "gate %r); the owner freezes them before any campaign run"
                            % (registry["status"], registry["competence_gate"].get("status")))
    item = [b for b in registry["namespaces"][owner]["blocks"] if b["seed"] == seed][0]
    stamp = {"registry_version": registry["registry_version"],
             "registry_sha256": registry_digest(registry),
             "registry_status": registry["status"],
             "gate_status": registry["competence_gate"].get("status"),
             "gate_sha256": digest(registry["competence_gate"]),
             "study_version": registry["study_version"], "namespace": owner,
             "block_id": item["id"], "seed": seed, "purpose": purpose,
             "selection_mode": selection_mode(registry),
             "selection_mode_status": registry["ppo_training"].get("selection_mode_status")}
    if purpose == "training":
        stamp["candidate_index"] = registry["ppo_training"]["candidate_order"].index(seed)
    return stamp


def resolve_seed_block(block, namespace, purpose, registry=None):
    """Resolve a named block (or its root seed) only in the requested split."""
    registry = load_registry() if registry is None else validate_registry(registry)
    if purpose not in EVALUATION_PURPOSES:
        raise RegistryError("evaluation purpose must be one of " + ", ".join(EVALUATION_PURPOSES))
    if namespace not in registry["namespaces"]:
        raise RegistryError("undeclared seed namespace: %r" % (namespace,))
    declaration = registry["namespaces"][namespace]
    if purpose != declaration["purpose"]:
        raise RegistryError("namespace %s is for %s, not %s" %
                            (namespace, declaration["purpose"], purpose))
    found = [item for item in declaration["blocks"]
             if str(block) in (item["id"], str(item["seed"]))]
    if len(found) != 1:
        owner = namespace_for_seed(_maybe_int(block), registry) if _maybe_int(block) is not None else None
        detail = " (it belongs to namespace %s)" % owner if owner else ""
        raise RegistryError("seed block %r is not declared in namespace %s%s"
                            % (block, namespace, detail))
    item = found[0]
    stamp = {"registry_version": registry["registry_version"],
             "registry_sha256": registry_digest(registry),
             "registry_status": registry["status"],
             "study_version": registry["study_version"], "namespace": namespace,
             "block_id": item["id"], "seed": item["seed"], "purpose": purpose}
    for key in ("cohort", "role"):
        if key in item:
            stamp[key] = item[key]
    return stamp


def _maybe_int(value):
    try:
        return int(str(value))
    except ValueError:
        return None


def validate_seed(seed, namespace, purpose, block_id=None, registry=None):
    if type(seed) is not int:
        raise RegistryError("seed must be an integer")
    stamp = resolve_seed_block(block_id if block_id is not None else seed,
                               namespace, purpose, registry)
    if stamp["seed"] != seed:
        raise RegistryError("seed does not match the declared seed block")
    return stamp


def diagnostic_registration(seed, registry=None):
    """Stamp for a direct evaluator call outside the registered namespaces.

    Refused for seeds owned by an evaluation namespace: those runs must name
    their namespace and purpose (and final tests must come from run_cell). A
    seed that is only a PPO training candidate (e.g. 42) may be used for a
    labelled diagnostic run; diagnostic certificates are not analysis inputs
    for any registered purpose.
    """
    registry = load_registry() if registry is None else validate_registry(registry)
    if type(seed) is not int:
        raise RegistryError("seed must be an integer")
    owner = namespace_for_seed(seed, registry)
    if owner is not None and purpose_of(owner, registry) in EVALUATION_PURPOSES:
        raise RegistryError("seed %d belongs to the %s namespace; pass --seed-namespace, "
                            "--seed-block and --run-purpose (final tests: use run_cell.py)"
                            % (seed, owner))
    return {"namespace": "diagnostic", "purpose": "diagnostic", "seed": seed,
            "block_id": None, "registry_sha256": registry_digest(registry),
            "registry_status": registry["status"],
            "also_registered_as": owner,
            "legacy_excluded": seed in registry.get("legacy_seeds_excluded", [])}


# ---------------------------------------------------------------- gate / ledger

def gate_verdict(gate, record):
    """Recompute a record's verdict from the declared gate; returns (verdict, reason)."""
    training = record.get("training")
    if not isinstance(training, dict) or training.get("status") not in TRAINING_STATUSES:
        raise RegistryError("training.status must be one of " + ", ".join(TRAINING_STATUSES))
    status = training["status"]
    if status == "infrastructure_failure":
        return "invalid", "infrastructure failure: rerun the same seed"
    if status == "diverged":
        return "fail", "training diverged (dead seed)"
    budget = _int(training.get("budget_timesteps"), "budget_timesteps", 1)
    done = _int(training.get("completed_timesteps"), "completed_timesteps", 0)
    if done != budget:
        return "invalid", "completed training must reach its declared budget"
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, dict) or not _is_sha256(checkpoint.get("sha256")):
        return "invalid", "final checkpoint SHA-256 missing"
    evidence = record.get("gate_certificate")
    if not isinstance(evidence, dict):
        return "invalid", "gate certificate missing"
    if evidence.get("seed_namespace") != gate["seed_namespace"] or \
            evidence.get("seed_block") != gate["seed_block"]:
        return "invalid", "gate evaluated outside the declared gate block"
    if evidence.get("episodes") != gate["episodes"] or evidence.get("num_envs") != gate["num_envs"]:
        return "invalid", "gate evaluated on a different grid"
    if evidence.get("complete") is not True:
        return "invalid", "gate certificate grid incomplete"
    if not _is_sha256(evidence.get("sha256")):
        return "invalid", "gate certificate SHA-256 missing"
    successes = _int(evidence.get("successes"), "gate successes", 0)
    if successes > gate["episodes"]:
        raise RegistryError("gate successes exceed episodes")
    needed = gate_min_successes(gate)
    if successes >= needed:
        return "pass", "%d/%d successes >= %d" % (successes, gate["episodes"], needed)
    return "fail", "%d/%d successes < %d (dead seed)" % (successes, gate["episodes"], needed)


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_ledger(ledger, registry):
    """Reject malformed, out-of-order, retried or hand-edited ledgers."""
    registry = validate_registry(registry)
    if not isinstance(ledger, dict) or ledger.get("ledger_version") not in LEDGER_FIELDS:
        raise RegistryError("gate ledger version 1 or 2 required")
    fields = LEDGER_FIELDS[ledger["ledger_version"]]
    if ledger.get("registry_sha256") != registry_digest(registry):
        raise RegistryError("gate ledger was written against a different registry")
    gate = registry["competence_gate"]
    if ledger.get("gate_sha256") != digest(gate):
        raise RegistryError("gate ledger was written against a different gate definition")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise RegistryError("gate ledger requires a records list")
    order = registry["ppo_training"]["candidate_order"]
    final = {}
    for index, record in enumerate(records):
        label = "records[%d]" % index
        if not isinstance(record, dict) or set(record) != set(fields):
            raise RegistryError(label + " must have exactly the fields " + ", ".join(fields))
        if ledger["ledger_version"] >= 2:
            for name in ("training_record_sha256", "mixture_sha256"):
                value = record[name]
                if value is not None and not _is_sha256(value):
                    raise RegistryError(label + "." + name + " must be a SHA-256 hex digest or null")
            training = record.get("training") if isinstance(record.get("training"), dict) else {}
            arm = training.get("arm", "mixture")
            if arm not in ("mixture", "calm"):
                raise RegistryError(label + ".training.arm must be 'mixture' or 'calm'")
            if arm == "calm" and record["mixture_sha256"] is not None:
                raise RegistryError(label + ": a calm-arm run carries no mixture_sha256")
            if training.get("status") == "completed" and (
                    record["training_record_sha256"] is None or (arm == "mixture" and record["mixture_sha256"] is None)):
                raise RegistryError(label + ": a completed training run needs training_record_sha256 and (mixture "
                                    "arm) mixture_sha256 (ledger version 2)")
        if not isinstance(record["task_id"], str) or not record["task_id"]:
            raise RegistryError(label + ".task_id required")
        if record["vessel"] not in VESSELS:
            raise RegistryError(label + ".vessel must be blueboat or wamv")
        seed = record["training_seed"]
        if seed not in order:
            raise RegistryError(label + ": training seed %r is not a declared candidate" % (seed,))
        if record["candidate_index"] != order.index(seed):
            raise RegistryError(label + ": candidate_index disagrees with candidate_order")
        verdict, reason = gate_verdict(gate, record)
        if record["verdict"] != verdict:
            raise RegistryError(label + ": stored verdict %r differs from recomputed %r"
                                % (record["verdict"], verdict))
        key = (record["task_id"], record["vessel"], seed)
        if key in final:
            raise RegistryError(label + ": candidate already has a final verdict; retries "
                                "of pass/fail outcomes are not allowed")
        if verdict in ("pass", "fail"):
            final[key] = record
    return ledger


def admission(ledger, registry, task_id, vessel):
    """Apply the Mode A rule: first N passing candidates, in declared order.

    G19b: when the candidates are exhausted (status 'exhausted') and the registry declares the B9 fallback
    (v3 fallback_rule 'first_in_order_flagged'), the result also carries ``fallback`` (see _fallback)."""
    validate_ledger(ledger, registry)
    order = registry["ppo_training"]["candidate_order"]
    needed = registry["ppo_training"]["admit_first_passing"]
    mine = [r for r in ledger["records"] if r["task_id"] == task_id and r["vessel"] == vessel]
    admitted, dead, history = [], [], []
    status, waiting_on = None, None
    for seed in order:
        if len(admitted) == needed:
            break
        attempts = [r for r in mine if r["training_seed"] == seed]
        history.extend(attempts)
        final = [r for r in attempts if r["verdict"] in ("pass", "fail")]
        if not final:
            status, waiting_on = "pending", seed
            break
        entry = {"training_seed": seed, "candidate_index": order.index(seed),
                 "checkpoint_sha256": final[0]["checkpoint"].get("sha256"),
                 "reason": final[0]["reason"]}
        (admitted if final[0]["verdict"] == "pass" else dead).append(entry)
    if status is None:
        status = "complete" if len(admitted) == needed else "exhausted"
    considered = {r["training_seed"] for r in history}
    unused = sorted({r["training_seed"] for r in mine} - considered,
                    key=order.index)
    result = {"task_id": task_id, "vessel": vessel, "status": status,
              "admitted": admitted, "dead": dead, "waiting_on_candidate": waiting_on,
              "records_beyond_rule": unused, "required": needed,
              "registry_sha256": registry_digest(registry), "gate_sha256": digest(registry["competence_gate"])}
    if status == "exhausted" and registry["ppo_training"].get("fallback_rule") == "first_in_order_flagged":
        result["fallback"] = _fallback(mine, order, needed)
    return result


def _final_record(records, seed):
    final = [r for r in records if r["training_seed"] == seed and r["verdict"] in ("pass", "fail")]
    return final[0] if final else None


def _delivered_entry(record, order):
    status = record["training"]["status"]
    flag = "diverged" if status == "diverged" else ("pass" if record["verdict"] == "pass" else "below_gate")
    entry = {"training_seed": record["training_seed"], "candidate_index": order.index(record["training_seed"]),
             "flag": flag, "checkpoint_sha256": record["checkpoint"].get("sha256"), "reason": record["reason"]}
    for name in ("training_record_sha256", "mixture_sha256"):
        if name in record:
            entry[name] = record[name]
    return entry


def _fallback(records, order, needed):
    """B9 fallback 'first_in_order_flagged' (Mode A, candidates exhausted): the first ``needed`` candidates in
    declared order whose training COMPLETED (a diverged run has no usable policy), each flagged by its own
    gate verdict ('pass' or 'below_gate')."""
    delivered = []
    for seed in order:
        record = _final_record(records, seed)
        if record is not None and record["training"]["status"] == "completed":
            delivered.append(_delivered_entry(record, order))
        if len(delivered) == needed:
            break
    return {"rule": "first_in_order_flagged", "delivered": delivered, "complete": len(delivered) == needed}


def delivery(ledger, registry, task_id, vessel, mode=None):
    """G19b (P7, owner decision O1): the policies delivered for (task, vessel) under the selection mode.

    ``mode`` defaults to the registry's ppo_training.selection_mode.
      first_passing  Mode A: admission(); the admitted seeds (flag 'pass') are delivered as soon as they are
                     admitted (an admission in declared order is final, also while later candidates are pending);
                     'exhausted' delivers the B9 fallback (v3 registries) or only the admitted seeds (v2).
      fixed_first_n  Mode B: candidates 1..N of candidate_order; each is delivered as soon as it has its own final
                     gate verdict (no selection: a candidate never depends on another one), with flag 'pass',
                     'below_gate' or 'diverged'; the status stays 'pending' until all N have one (an infrastructure
                     failure is rerun with the same seed); nothing is replaced.
    Returns {task_id, vessel, mode, status ('complete'|'pending'|'exhausted'), delivered [...],
    waiting_on_candidate, records_beyond_rule, dead, required, registry_sha256, gate_sha256}."""
    validate_ledger(ledger, registry)
    mode = selection_mode(registry) if mode is None else mode
    if mode not in SELECTION_MODES:
        raise RegistryError("selection mode must be one of " + ", ".join(SELECTION_MODES))
    order = registry["ppo_training"]["candidate_order"]
    needed = registry["ppo_training"]["admit_first_passing"]
    mine = [r for r in ledger["records"] if r["task_id"] == task_id and r["vessel"] == vessel]
    base = {"task_id": task_id, "vessel": vessel, "mode": mode, "required": needed,
            "registry_sha256": registry_digest(registry), "gate_sha256": digest(registry["competence_gate"])}
    if mode == "first_passing":
        result = admission(ledger, registry, task_id, vessel)
        if result["status"] == "exhausted" and "fallback" in result:
            delivered = result["fallback"]["delivered"]
        else:
            delivered = [_delivered_entry(_final_record(mine, a["training_seed"]), order)
                         for a in result["admitted"]]
        return dict(base, status=result["status"], delivered=delivered,
                    waiting_on_candidate=result["waiting_on_candidate"],
                    records_beyond_rule=result["records_beyond_rule"], dead=result["dead"])
    delivered, waiting_on = [], None
    for seed in order[:needed]:
        record = _final_record(mine, seed)
        if record is None:
            waiting_on = seed if waiting_on is None else waiting_on
            continue
        delivered.append(_delivered_entry(record, order))
    status = "pending" if waiting_on is not None else "complete"
    beyond = sorted({r["training_seed"] for r in mine} - set(order[:needed]), key=order.index)
    return dict(base, status=status, delivered=delivered, waiting_on_candidate=waiting_on,
                records_beyond_rule=beyond, dead=[])


def gate_record_from_certificate(document, *, registry, task_id, vessel, training_seed,
                                 training_status, budget_timesteps, completed_timesteps,
                                 checkpoint_sha256=None, checkpoint_path=None,
                                 certificate_path=None, certificate_sha256=None, note="",
                                 ledger_version=1, training_record_sha256=None, mixture_sha256=None,
                                 arm=None):
    """Build a ledger record whose gate evidence is read from a validated v3 certificate.

    G19b: ``ledger_version=2`` adds training_record_sha256 and mixture_sha256 (P7)."""
    if ledger_version not in LEDGER_FIELDS:
        raise RegistryError("ledger version 1 or 2 required")
    registry = validate_registry(registry)
    order = registry["ppo_training"]["candidate_order"]
    if training_seed not in order:
        raise RegistryError("training seed %r is not a declared candidate" % (training_seed,))
    gate = registry["competence_gate"]
    evidence = None
    if document is not None:
        from analysis_input import validate_analysis_input
        validate_analysis_input(document, allowed_purposes=("selection",))
        header = document["header"]
        registration = header["seed_namespace"]
        if header["task_id"] != task_id:
            raise RegistryError("gate certificate task differs from the record task")
        from cert_v3 import vessel_family
        if vessel_family(header["vessel"]) != vessel:
            raise RegistryError("gate certificate vessel differs from the record vessel")
        if header["controller"]["name"] != "learned":
            raise RegistryError("gate certificates must come from the PPO evaluator")
        cell = header.get("cell_config") or {}
        planned = cell.get("configuration") or {}
        if planned.get("training_seed") != training_seed:
            raise RegistryError("gate certificate was not planned for this training seed")
        if planned.get("preset") is not None or planned.get("effective_override_literals"):
            raise RegistryError("gate must be evaluated at the nominal condition "
                                "(no dose preset, no --set)")
        if checkpoint_sha256 is not None and header["checkpoint"]["sha256"] != checkpoint_sha256:
            raise RegistryError("gate certificate checkpoint differs from the declared checkpoint")
        checkpoint_sha256 = header["checkpoint"]["sha256"]
        checkpoint_path = checkpoint_path or header["checkpoint"]["path"]
        evidence = {"path": certificate_path, "sha256": certificate_sha256,
                    "seed_namespace": registration.get("namespace"),
                    "seed_block": registration.get("block_id"),
                    "episodes": header["episodes_requested"], "num_envs": header["num_envs"],
                    "successes": sum(1 for r in document["records"] if r["success"] is True),
                    "complete": header["complete"]}
    record = {"task_id": task_id, "vessel": vessel, "training_seed": training_seed,
              "candidate_index": order.index(training_seed),
              "training": {"status": training_status, "budget_timesteps": budget_timesteps,
                           "completed_timesteps": completed_timesteps, "note": note},
              "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha256},
              "gate_certificate": evidence, "verdict": None, "reason": None}
    if arm is not None:                      # G19b follow-up m9: the training arm of a v2 record
        record["training"]["arm"] = arm
    if ledger_version >= 2:
        for name, value in (("training_record_sha256", training_record_sha256), ("mixture_sha256", mixture_sha256)):
            if value is not None and not _is_sha256(value):
                raise RegistryError(name + " must be a SHA-256 hex digest")
            record[name] = value
    record["verdict"], record["reason"] = gate_verdict(gate, record)
    return record


def empty_ledger(registry, ledger_version=1):
    """New gate ledger. G19b: ``ledger_version=2`` is the campaign format (training record and mixture
    digests per record); the default stays 1 so existing callers are unchanged."""
    if ledger_version not in LEDGER_FIELDS:
        raise RegistryError("ledger version 1 or 2 required")
    return {"ledger_version": ledger_version, "registry_sha256": registry_digest(registry),
            "gate_sha256": digest(registry["competence_gate"]), "records": []}


# ---------------------------------------------------------------- test blocks

def next_reserve_block(registry, used_block_ids):
    """Deterministic replacement for an invalid final-test block (never outcome-based)."""
    for block in registry["namespaces"]["final_test"]["blocks"]:
        if block["cohort"] == "reserve" and block["id"] not in used_block_ids:
            return block
    raise RegistryError("no unused reserve final-test block remains")


def scenario_overlap(documents_a, documents_b):
    """Whole-scenario digests present in both certificate sets (validity check).

    A final-test block must share no complete scenario with training-side
    selection/pilot/calibration evaluations of the same task.
    """
    def digests(documents):
        found = set()
        for document in documents:
            for record in document.get("records", []):
                hashes = record.get("scenario_hashes")
                if isinstance(hashes, dict) and isinstance(hashes.get("scenario"), str):
                    found.add((document["header"]["task_id"], hashes["scenario"]))
        return found
    return sorted(digests(documents_a) & digests(documents_b))


# ---------------------------------------------------------------- CLI

def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("show")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--block", required=True)
    resolve.add_argument("--namespace", required=True)
    resolve.add_argument("--purpose", required=True)
    record = sub.add_parser("gate-record")
    record.add_argument("--certificate", type=Path)
    record.add_argument("--task", required=True)
    record.add_argument("--vessel", required=True, choices=VESSELS)
    record.add_argument("--training-seed", type=int, required=True)
    record.add_argument("--training-status", required=True, choices=TRAINING_STATUSES)
    record.add_argument("--budget-timesteps", type=int, required=True)
    record.add_argument("--completed-timesteps", type=int, required=True)
    record.add_argument("--checkpoint", type=Path)
    record.add_argument("--note", default="")
    record.add_argument("--training-record", type=Path,
                        help="G19b: PPO training record (ledger v2 record: its SHA-256 and mixture SHA-256)")
    admit = sub.add_parser("admission")
    admit.add_argument("--ledger", type=Path, required=True)
    admit.add_argument("--task", required=True)
    admit.add_argument("--vessel", required=True, choices=VESSELS)
    deliver = sub.add_parser("delivery")
    deliver.add_argument("--ledger", type=Path, required=True)
    deliver.add_argument("--task", required=True)
    deliver.add_argument("--vessel", required=True, choices=VESSELS)
    deliver.add_argument("--selection-mode", choices=SELECTION_MODES)
    training = sub.add_parser("resolve-training")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--purpose", required=True, choices=TRAINING_PURPOSES)
    init = sub.add_parser("ledger-init")
    init.add_argument("--ledger-version", type=int, default=2, choices=sorted(LEDGER_FIELDS))
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        if args.mode == "show":
            result = {"registry_sha256": registry_digest(registry),
                      "gate_sha256": digest(registry["competence_gate"]),
                      "gate_min_successes": gate_min_successes(registry["competence_gate"]),
                      "registry": registry}
        elif args.mode == "resolve":
            result = resolve_seed_block(args.block, args.namespace, args.purpose, registry)
        elif args.mode == "gate-record":
            document = cert_sha = None
            if args.certificate is not None:
                from validate_cert_v3 import load_json
                document = load_json(args.certificate)
                cert_sha = hashlib.sha256(args.certificate.read_bytes()).hexdigest()
            checkpoint_sha = None
            if args.checkpoint is not None and args.checkpoint.is_file():
                checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            v2 = {}
            if args.training_record is not None:
                training_record = read_json(args.training_record)
                if training_record.get("training_seed") != args.training_seed:
                    raise RegistryError("the training record belongs to another training seed")
                v2 = {"ledger_version": 2, "arm": training_record.get("arm", "mixture"),
                      "training_record_sha256": hashlib.sha256(args.training_record.read_bytes()).hexdigest(),
                      "mixture_sha256": (training_record.get("mixture") or {}).get("mixture_sha256")}
            result = gate_record_from_certificate(
                document, registry=registry, task_id=args.task, vessel=args.vessel,
                training_seed=args.training_seed, training_status=args.training_status,
                budget_timesteps=args.budget_timesteps, completed_timesteps=args.completed_timesteps,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_path=str(args.checkpoint.resolve()) if args.checkpoint else None,
                certificate_path=str(args.certificate.resolve()) if args.certificate else None,
                certificate_sha256=cert_sha, note=args.note, **v2)
        elif args.mode == "delivery":
            result = delivery(read_json(args.ledger), registry, args.task, args.vessel, args.selection_mode)
        elif args.mode == "resolve-training":
            result = resolve_training_block(args.seed, args.purpose, registry)
        elif args.mode == "ledger-init":
            result = empty_ledger(registry, args.ledger_version)
        else:
            result = admission(read_json(args.ledger), registry, args.task, args.vessel)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, KeyError) as exc:
        print("seed_registry: error: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
