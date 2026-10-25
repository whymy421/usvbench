"""G28 item 3: seed registry v3 - selection_mode fixed_first_n is owner-confirmed (train 3, deliver 3, no picking);
the registry and the competence gate stay unfrozen, so campaign training seeds stay refused (CPU, pure Python).

Run: python -B scripts/test_g28_seed_registry_confirmation.py   (or pytest)
"""
from __future__ import annotations

import copy
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import seed_registry as seeds  # noqa: E402

# G52 (v2.6.6): the owner froze the registry on 2026-10-14; these are the v2.6.5 values of the fields the freeze
# changed, so the digest pin below still checks that G28 changed only the confirmation fields.
OLD_CHECKPOINT_RULE = ("the final checkpoint of the fixed, predeclared training budget; no checkpoint search (OWNER DECISION: C7 section 6 alternatively proposed a 25/50/75/100% schedule)")
OLD_RATIONALE = ("PROPOSED DEFAULT, OWNER DECISION. A low absolute bar detects collapsed (dead) training runs without ranking healthy seeds by performance; it is the same for every task and vessel so it can be declared before training.")


def before_g52(registry):
    out = copy.deepcopy(registry)
    out["status"] = "proposal"
    del out["freeze_record"]
    gate = out["competence_gate"]
    gate["status"], gate["threshold_is_proposed_default"] = "OWNER_DECISION_PENDING", True
    gate["checkpoint_rule"], gate["rationale"] = OLD_CHECKPOINT_RULE, OLD_RATIONALE
    final = out["namespaces"]["final_test"]
    final["planned_main_blocks"] = 4
    del final["main_set_rule"]
    for block in final["blocks"]:
        if block["id"] in ("final-02", "final-03", "final-04"):
            block["cohort"] = "main"
    return out


def test_in_tree_registry_encodes_the_owner_confirmation():
    registry = seeds.load_registry()
    training = registry["ppo_training"]
    confirmation = training["selection_mode_confirmation"]
    assert confirmation["status"] == "owner_confirmed"
    assert confirmation["confirmed_mode"] == training["selection_mode"] == seeds.selection_mode(registry) \
        == "fixed_first_n"
    assert confirmation["by"] == "owner" and confirmation["date"] == "2026-10-03"
    assert "item 6" in confirmation["source"] and "17:0x" in confirmation["source"]
    assert "train 3, deliver 3, no picking" in confirmation["rule"]
    assert confirmation["freeze_implied"] is False
    assert training["admit_first_passing"] == 3
    assert training["selection_mode_status"].startswith("owner confirmed")
    # G52 (v2.6.6): frozen by the owner on 2026-10-14 (a separate decision; the confirmation implies no freeze)
    assert registry["status"] == "frozen" and registry["freeze_record"]["date"] == "2026-10-14"
    assert registry["competence_gate"]["status"] == "frozen"
    assert seeds.check_selection_confirmation(training) is confirmation


def test_campaign_seeds_stay_refused_smoke_seeds_stay_usable():
    for seed in seeds.load_registry()["ppo_training"]["candidate_order"][:3]:      # G52: frozen in-tree registry
        assert seeds.resolve_training_block(seed, "training")["registry_status"] == "frozen"
    registry = before_g52(seeds.load_registry())         # the unfrozen (v2.6.5) registry still refuses them
    for seed in registry["ppo_training"]["candidate_order"][:3]:
        with pytest.raises(ValueError, match="frozen registry and competence gate"):
            seeds.resolve_training_block(seed, "training", registry)
    stamp = seeds.resolve_training_block(90001, "training_smoke", registry)
    assert stamp["selection_mode"] == "fixed_first_n" and stamp["registry_status"] == "proposal"


def test_validator_rejects_inconsistent_confirmations():
    registry = seeds.load_registry()
    cases = []
    changed = copy.deepcopy(registry)
    changed["ppo_training"]["selection_mode"] = "first_passing"          # mode changed without a new confirmation
    cases.append((changed, "new owner confirmation"))
    for field in seeds.CONFIRMATION_FIELDS:
        broken = copy.deepcopy(registry)
        del broken["ppo_training"]["selection_mode_confirmation"][field]
        cases.append((broken, "lacks " + field))
    frozen = copy.deepcopy(registry)
    frozen["ppo_training"]["selection_mode_confirmation"]["freeze_implied"] = True
    cases.append((frozen, "freeze_implied must be false"))
    status = copy.deepcopy(registry)
    status["ppo_training"]["selection_mode_confirmation"]["status"] = "confirmed-ish"
    cases.append((status, "status must be one of"))
    text = copy.deepcopy(registry)
    text["ppo_training"]["selection_mode_status"] = "pending owner confirmation"
    cases.append((text, "must say 'owner confirmed"))
    empty = copy.deepcopy(registry)
    empty["ppo_training"]["selection_mode_confirmation"]["source"] = ""
    cases.append((empty, "source must be a nonempty string"))
    not_object = copy.deepcopy(registry)
    not_object["ppo_training"]["selection_mode_confirmation"] = "yes"
    cases.append((not_object, "must be an object"))
    for case, message in cases:
        with pytest.raises(ValueError, match=message):
            seeds.validate_registry(case)


def test_confirmation_is_optional_and_v2_registries_stay_readable():
    registry = seeds.load_registry()
    v3_without = copy.deepcopy(registry)
    del v3_without["ppo_training"]["selection_mode_confirmation"]
    seeds.validate_registry(v3_without)
    pending = copy.deepcopy(registry)
    pending["ppo_training"]["selection_mode_confirmation"]["status"] = "pending_owner_confirmation"
    pending["ppo_training"]["selection_mode_status"] = "pending owner confirmation"
    seeds.validate_registry(pending)
    v2 = copy.deepcopy(registry)
    v2["registry_version"] = 2
    del v2["namespaces"]["training_smoke"]
    for name in ("selection_mode", "selection_mode_status", "selection_modes", "fallback_rule", "fallback_rule_text",
                 "selection_mode_confirmation"):
        del v2["ppo_training"][name]
    v2["ppo_training"]["exhaustion_policy"] = "report the admitted count"
    seeds.validate_registry(v2)
    assert seeds.selection_mode(v2) == "first_passing"


def test_registry_digest_changed_only_by_the_confirmation_fields():
    registry = seeds.load_registry()
    stripped = before_g52(registry)                       # G52: undo the 2026-10-14 freeze first
    del stripped["ppo_training"]["selection_mode_confirmation"]
    # G19b verify2 digest of the registry before G28 (FINAL_REPORT section 8 of G19b)
    assert seeds.registry_digest(stripped) == "ab842288b843f2a92c40eba37ed82e17964a92191e9a65749b8fca4a245eb90a"
    assert seeds.registry_digest(registry) != seeds.registry_digest(stripped)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("PASS " + name, flush=True)
