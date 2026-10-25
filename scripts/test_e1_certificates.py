"""E1 certificate regression tests (schema 3.2). CPU only; no Isaac import.

Covers D2 review items 8, 9, 10, 20 and 21 (REVIEW.md), CONTEXT_R3 decisions
16/17 and the E1 brief: release code identity, physics/hydrodynamics/vessel
stamps, docking level distance, validator recomputation, statistics
consistency, the per-episode outcome table, D4's appended termination codes,
the USV-10K speed reference, the evaluator environment allow-list and the
run_cell harbour refusal. D2's saved counterexamples (REPRO\\c5_malformed_*.json,
copied byte-identical into scripts/fixtures_e1_d2/) are regression inputs.
"""
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))


def load(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cert = load("cert_v3")
validator = load("validate_cert_v3")
outcome = load("episode_outcome")
code_tree = load("code_tree")
protocol = load("eval_protocol")
inputs = load("analysis_input")
runner = load("run_cell")
fixtures = load("test_cert_v3")
D2 = SCRIPTS / "fixtures_e1_d2"


def step(run):
    fixtures.step(run)


def rec(env, ep, **extra):
    record = fixtures.sample_record(env, ep)
    record.update(extra)
    return record


def small_doc(n=2, quota=1, base=None, env_cfg=None, records=None, **kwargs):
    """Writer document on an n x quota grid with the test_cert_v3 fixture."""
    options = {}
    if base is not None:
        options["base"] = base
    if env_cfg is not None:
        options["env_cfg"] = env_cfg
        options.setdefault("overrides", {})
        options.setdefault("argv", ["eval.py"])
    options.update(kwargs)
    run = fixtures.make_run(n, n * quota, **options)
    rows = []
    for ep in range(quota):
        step(run)
        for env in range(n):
            rows.append(run.finish_episode(env, (records or {}).get((env, ep), rec(env, ep))))
    return run, run.build({"records": rows})


def check(doc, **kwargs):
    header = doc["header"]
    return validator.validate_certificate(doc, num_envs=header["num_envs"],
                                          episodes_per_env=header["episodes_per_env"], **kwargs)


def cfg(**fields):
    base = dict(seed=42, eval_level=0, curriculum_frozen=True, vehicle="blueboat", scene=SimpleNamespace(num_envs=2))
    base.update(fields)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------- item 21 + decision 17

def test_termination_names_are_c5_plus_d4_verbatim():
    c5 = load("e1_termination_reason", REPO / "tasks" / "_shared" / "termination_reason.py")
    d4 = load("e1_journal_contracts_v2", REPO / "tasks" / "_shared" / "journal_contracts_v2.py")
    assert {k: v for k, v in outcome.TERMINATION_REASON_NAMES.items() if k <= 5} == c5.TERMINATION_REASON_NAMES
    assert outcome.TERMINATION_REASON_NAMES == dict(d4.TERMINATION_REASON_NAMES)
    assert cert.REASONS == outcome.TERMINATION_REASON_NAMES
    schema = json.loads((SCRIPTS / "cert_v3.schema.json").read_text(encoding="ascii"))
    assert schema["$defs"]["episode"]["properties"]["termination_reason"]["enum"] == list(outcome.RECORD_REASONS)
    assert schema["$defs"]["episode"]["properties"]["outcome"]["enum"] == list(outcome.OUTCOMES) + [None]


@pytest.mark.parametrize("reason,success,contact,fixed,expected", [
    ("capsize", False, None, None, "capsized"), ("sinking", True, True, True, "sinking"),
    ("airborne", False, False, False, "airborne"), ("non_finite_state", False, None, True, "non_finite_state"),
    ("out_of_bounds", True, None, False, "out_of_bounds"), ("collision", False, True, None, "contact_voided"),
    ("success", True, True, False, "success"), ("timeout", True, None, True, "success"),
    ("failure", False, True, False, "contact_voided"), ("failure", False, None, False, "never_achieved"),
    ("timeout", False, True, True, "contact_voided"), ("timeout", False, False, True, "never_achieved"),
    ("timeout", False, None, True, "never_achieved"), ("timeout", False, False, False, "timeout"),
    ("timeout", False, None, None, "timeout"), ("success", False, None, None, None),
    ("unknown", False, None, None, None),
])
def test_outcome_table(reason, success, contact, fixed, expected):
    record = {"termination_reason": reason, "success": success, "contact_steps": None, "min_clearance_m": None}
    if contact is not None:
        record["contact_steps"] = 3 if contact else 0
    assert outcome.episode_outcome(record, fixed) == expected
    if contact:  # a negative clearance is contact evidence too
        record.update(contact_steps=None, min_clearance_m=-0.1)
        assert outcome.episode_outcome(record, fixed) == expected


def _family_class(family):
    return type("FakeEnv", (), {"__module__": "isaaclab_tasks.direct.%s.%s_env" % (family, family)})


def _family_base(family, reasons, n=2):
    base = _family_class(family)()
    base.num_envs, base.device, base.termination_reason = n, "cpu", list(reasons)
    return base


def test_writer_maps_d4_validity_codes_and_validator_accepts_them():
    base = _family_base("hazard_nav", [6, 7])
    records = {(0, 0): rec(0, 0, terminated=True, truncated=False), (1, 0): rec(1, 0, terminated=True, truncated=False)}
    run, doc = small_doc(base=base, env_cfg=cfg(), records=records)
    assert [r["termination_reason"] for r in doc["records"]] == ["capsize", "sinking"]
    assert [r["outcome"] for r in doc["records"]] == ["capsized", "sinking"]
    assert doc["schema_minor_version"] == 2 and not check(doc)
    for code, name in ((8, "airborne"), (9, "non_finite_state")):
        base = _family_base("hazard_nav", [code, code])
        _, doc = small_doc(base=base, env_cfg=cfg(), records=records)
        assert doc["records"][0]["termination_reason"] == name and not check(doc)
    doc["records"][0]["terminated"] = False
    doc["records"][0]["truncated"] = True
    assert any("terminal reason requires terminated=true" in e for e in check(doc))


def test_fixed_horizon_families_separate_never_achieved_from_contact_voided():
    base = _family_base("station_keeping", [3, 3])
    records = {(0, 0): rec(0, 0, contact_steps=0), (1, 0): rec(1, 0, contact_steps=1, contact_seconds=1 / 60.0,
                                                               contact_longest_seconds=1 / 60.0,
                                                               contact_depth_mean_m=0.1)}
    base.control_step_s = 1 / 60.0
    run, doc = small_doc(base=base, env_cfg=cfg(), records=records)
    assert doc["header"]["outcome_semantics"] == {"fixed_horizon": True, "source": "task family station_keeping"}
    assert [r["outcome"] for r in doc["records"]] == ["never_achieved", "contact_voided"]
    assert not check(doc)
    hazard = _family_base("hazard_nav", [3, 3])
    _, doc = small_doc(base=hazard, env_cfg=cfg())
    assert [r["outcome"] for r in doc["records"]] == ["timeout", "timeout"]
    doc["records"][0]["outcome"] = "never_achieved"
    assert any("outcome 'never_achieved' differs from the mapping table" in e for e in check(doc))
    doc["records"][0]["outcome"] = "timeout"
    doc["header"]["outcome_semantics"] = {"fixed_horizon": True, "source": "tampered"}
    assert any("differs from the mapping table" in e for e in check(doc))


def test_fixed_horizon_family_cannot_stop_on_success():
    base = _family_base("docking", [1, 3])
    base.curriculum = SimpleNamespace(spawn_distance=25.0)
    records = {(0, 0): rec(0, 0, success=True, terminated=True, truncated=False, tts_s=1.0)}
    _, doc = small_doc(base=base, env_cfg=cfg(eval_level_distances=(25.0, 15.0, 8.0, 2.0)), records=records)
    assert any("fixed-horizon family cannot stop on success" in e for e in check(doc))


# ---------------------------------------------------------------- item 8

def test_physics_version_is_validated_string():
    nested = cfg(underwater_physics_cfg=SimpleNamespace(physics_version=2))
    assert cert.declared_physics_version(nested) == "2"
    assert cert.declared_physics_version(cfg(physics_version="1")) == "1"
    assert cert.declared_physics_version(cfg()) is None
    for bad in (3, "v2", True, 0):
        with pytest.raises(ValueError, match="physics_version"):
            cert.declared_physics_version(cfg(underwater_physics_cfg=SimpleNamespace(physics_version=bad)))
    _, doc = small_doc(env_cfg=nested)
    assert doc["header"]["physics_version"] == "2" and not check(doc)
    tampered = copy.deepcopy(doc)
    tampered["header"]["physics_version"] = "1"
    assert any("physics_version" in e for e in check(tampered))
    tampered["header"]["physics_version"] = "3"
    assert any("schema header/physics_version" in e for e in check(tampered))
    missing = copy.deepcopy(doc)
    missing["header"]["physics_version"] = None
    assert any("header has none" in e for e in check(missing))


def test_hydrodynamics_model_and_vessel_profile_from_cfg():
    physics = SimpleNamespace(physics_version=2, hydrodynamics_model="hydro_v2", vessel_profile="blueboat_v2",
                              vessel_profile_status="provisional: D5 registry")
    # E-base (controller, 2026-09-30): under hydro_v2 the env ALWAYS carries the
    # schema-3 condition layer (E3 conditions/certificate.py refuses a hydro_v2
    # cfg without one), so the fake env gets a minimal layer whose stamps are
    # real schema-3 stamps of an all-off JournalConditionConfig.
    from tasks._shared.conditions.config_v3 import resolve_v3
    from tasks._shared.conditions.stamp import stamp_conditions_v3
    off = resolve_v3(seed=42)
    layer = SimpleNamespace(certificate_stamp=lambda: stamp_conditions_v3(off, context={"layer_active": True}))
    hydro_base = SimpleNamespace(num_envs=2, device="cpu", termination_reason=[3] * 2, _conditions=layer,
                                 episode_conditions=[stamp_conditions_v3(off, runtime={"env_index": i},
                                                                         context={"layer_active": True})
                                                     for i in range(2)])
    _, doc = small_doc(base=hydro_base, env_cfg=cfg(underwater_physics_cfg=physics))
    header = doc["header"]
    assert header["hydrodynamics_model"] == "hydro_v2"
    assert header["vessel_profile"] == {"name": "blueboat_v2", "status": "provisional: D5 registry",
                                        "source": "env_cfg.underwater_physics_cfg.vessel_profile"}
    assert not check(doc)
    _, legacy = small_doc(env_cfg=cfg())
    assert legacy["header"]["hydrodynamics_model"] == "legacy" and legacy["header"]["vessel_profile"] is None
    profile_object = cfg(vessel_profile=SimpleNamespace(name="wamv_vrx_v2_opt1", status="PENDING owner"))
    assert cert.declared_vessel_profile(profile_object) == {"name": "wamv_vrx_v2_opt1", "status": "PENDING owner",
                                                            "source": "env_cfg.vessel_profile"}
    with pytest.raises(ValueError, match="requires physics_version 2"):
        small_doc(env_cfg=cfg(underwater_physics_cfg=SimpleNamespace(physics_version=1,
                                                                     hydrodynamics_model="hydro_v2")))
    with pytest.raises(ValueError, match="hydrodynamics_model"):
        cert.declared_hydrodynamics_model(cfg(hydrodynamics_model="c3"))
    for field, value, message in (("hydrodynamics_model", "legacy", "hydrodynamics_model"),
                                  ("vessel_profile", {"name": "other", "status": None, "source": "x"},
                                   "vessel_profile"),
                                  ("vessel", "wamv", "vessel")):
        tampered = copy.deepcopy(doc)
        tampered["header"][field] = value
        assert any(message in e for e in check(tampered)), field


@pytest.mark.parametrize("version,level,override,expected,field", [
    (1, 0, 0.0, 25.0, "eval_level_distances"), (2, 0, 0.0, 2.0, "eval_level_distances_v2"),
    (2, 3, 0.0, 25.0, "eval_level_distances_v2"), (2, 9, 0.0, 25.0, "eval_level_distances_v2"),
    (1, 1, 12.5, 12.5, "eval_level_distances"),
])
def test_docking_level_distance_is_stamped_and_checked(version, level, override, expected, field):
    config = cfg(eval_level=level, eval_level_distances=(25.0, 15.0, 8.0, 2.0),
                 eval_level_distances_v2=(2.0, 8.0, 15.0, 25.0), spawn_distance_override_m=override,
                 underwater_physics_cfg=SimpleNamespace(physics_version=version))
    base = _family_base("docking", [3, 3])
    base.curriculum = SimpleNamespace(spawn_distance=expected)
    _, doc = small_doc(base=base, env_cfg=config, level=level, eval_seed=42)
    stamp = doc["header"]["level_distance"]
    assert (stamp["table_field"], stamp["expected_distance_m"], stamp["realised_distance_m"]) == (field, expected, expected)
    assert not check(doc)
    doc["header"]["level_distance"]["realised_distance_m"] = expected + 1.0
    assert any("realised docking spawn distance" in e for e in check(doc))
    doc["header"]["level_distance"] = None
    assert any("lacks the level_distance stamp" in e for e in check(doc))


def test_level_distance_unavailable_is_unverified_and_non_docking_has_none():
    config = cfg(eval_level_distances=(25.0, 15.0, 8.0, 2.0))
    run, doc = small_doc(env_cfg=config)
    assert doc["header"]["level_distance"]["realised_distance_m"] is None
    assert any("level_distance" in note for note in run.unverified)
    _, plain = small_doc(env_cfg=cfg())
    assert plain["header"]["level_distance"] is None
    plain["header"]["level_distance"] = doc["header"]["level_distance"]
    assert any("without a docking level table" in e for e in check(plain))


def test_override_of_the_inactive_docking_table_is_refused():
    def docking_cfg(version):
        return SimpleNamespace(seed=0, eval_level=0, curriculum_frozen=False, vehicle="blueboat",
                               eval_level_distances=(25.0, 15.0, 8.0, 2.0), eval_level_distances_v2=(2.0, 8.0, 15.0, 25.0),
                               underwater_physics_cfg=SimpleNamespace(physics_version=version),
                               scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device="cpu"))
    args = SimpleNamespace(eval_seed=5, level=0, num_envs=2, episodes=4, device="cpu",
                           extra_sets=["eval_level_distances=30,20,10,3"])
    with pytest.raises(ValueError, match="no effect under physics_version 2"):
        protocol.resolve_protocol(docking_cfg(2), args)
    applied = protocol.resolve_protocol(docking_cfg(1), args)
    assert applied["eval_level_distances"] == (30.0, 20.0, 10.0, 3.0)
    args.extra_sets = ["eval_level_distances_v2=3,10,20,30"]
    with pytest.raises(ValueError, match="no effect under physics_version 1"):
        protocol.resolve_protocol(docking_cfg(1), args)


# ---------------------------------------------------------------- items 10 and 20: D2 counterexamples

D2_CASES = {
    "impossible_timing_std": ("timing std", lambda d: d["records"][0]["controller_compute_time_s"].update(std=0.0)),
    "contradictory_code_hash": ("tree_sha256 is inconsistent",
                                lambda d: d["header"]["code"].update(tree_sha256=cert.legacy_tree_digest("a" * 64, "a" * 64))),
    "wrong_legacy_controller": ("legacy alias disagrees with header: controller", lambda d: d.update(controller="test")),
    "wrong_legacy_checkpoint": ("legacy alias disagrees with header: checkpoint", lambda d: d.update(checkpoint=None)),
    "contradictory_physics_release": ("physics_version",
                                      lambda d: d["header"]["resolved_config"].update(physics_version="v2")),
    "contradictory_cuda_timing": ("cuda_synchronized contradicts",
                                  lambda d: d["header"]["runtime"].update(gpu_device="synthetic_fixture")),
    "impossible_contact_duration": ("must be zero when contact_steps is zero",
                                    lambda d: d["records"][0].update(contact_seconds=None, contact_longest_seconds=None,
                                                                     contact_depth_mean_m=None, contact_steps=None)),
}


@pytest.mark.parametrize("name", sorted(D2_CASES))
def test_d2_counterexamples_rejected_and_their_baseline_valid(name):
    message, revert = D2_CASES[name]
    document = validator.load_json(D2 / ("c5_malformed_%s.json" % name))
    assert "schema_minor_version" not in document  # schema 3.0 documents
    errors = validator.validate_certificate(document, num_envs=1, episodes_per_env=1, require_verified=True)
    assert any(message in e for e in errors), errors
    revert(document)  # D2's single mutation undone: D2's own baseline, still valid 3.0
    assert validator.validate_certificate(document, num_envs=1, episodes_per_env=1, require_verified=True) == []


@pytest.mark.parametrize("mutation,message", [
    (lambda s: s.update(std=(s["max"] - s["min"]) * 0.75 + 1e-3), "exceeds (max - min) / 2"),
    (lambda s: s.update(std=0.0), "zero exactly when min == max"),
    (lambda s: s.update(n=1, total=s["mean"]), "impossible for n=1"),
])
def test_timing_statistics_consistency(mutation, message):
    run = fixtures.make_run(1, 1)
    for delay in (0.001, 0.003, 0.004):
        run._clock = iter([0.0, delay]).__next__
        with run.time_controller():
            pass
        run._clock = iter([0.0, 0.002]).__next__
        with run.time_environment():
            pass
        run.after_step()
    doc = run.build({"records": [run.finish_episode(0, rec(0, 0))]})
    assert not check(doc)
    mutation(doc["records"][0]["controller_compute_time_s"])
    assert any(message in e for e in check(doc))


def test_contact_durations_consistent_with_steps_and_episode():
    base = _family_base("hazard_nav", [3, 3])
    base.control_step_s = 0.5
    records = {(0, 0): rec(0, 0, contact_steps=2, contact_seconds=1.0, contact_longest_seconds=0.5,
                           contact_depth_mean_m=0.1)}
    run = fixtures.make_run(2, 4, base=base, env_cfg=cfg(), overrides={}, argv=["eval.py"])
    for _ in range(3):
        step(run)
    rows = [run.finish_episode(env, records.get((env, 0), rec(env, 0))) for env in range(2)]
    for _ in range(3):
        step(run)
    rows += [run.finish_episode(env, rec(env, 1)) for env in range(2)]
    doc = run.build({"records": rows})
    assert doc["records"][0]["episode_elapsed_s"] == 1.5 and not check(doc)
    bad = copy.deepcopy(doc)
    bad["records"][0]["contact_seconds"] = 1.25
    assert any("contact_seconds != contact_steps * control_step_s" in e for e in check(bad))
    bad = copy.deepcopy(doc)
    bad["records"][0].update(contact_steps=8, contact_seconds=4.0)
    assert any("exceeds the episode length" in e for e in check(bad))


# ---------------------------------------------------------------- item 10: aliases

def test_writer_canonicalises_and_refuses_checkpoint_and_controller_aliases(tmp_path):
    checkpoint = tmp_path / "agent.pt"
    checkpoint.write_bytes(b"weights")
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    run = fixtures.make_run(1, 1, checkpoint=checkpoint)
    step(run)
    row = run.finish_episode(0, rec(0, 0))
    spelled = str(tmp_path / "sub" / ".." / "agent.pt")
    doc = run.build({"records": [row], "checkpoint": spelled, "controller": "test"})
    assert doc["checkpoint"] == str(checkpoint.resolve()) and not check(doc)
    for alias, message in ((str(other), "is not the certified checkpoint"), ("elsewhere/agent.pt", "not a suffix"),
                           (None, "presence")):
        with pytest.raises(ValueError, match=message):
            run.build({"records": [row], "checkpoint": alias})
    assert run.build({"records": [row], "checkpoint": "agent.pt"})["checkpoint"] == "agent.pt"
    with pytest.raises(ValueError, match="controller alias"):
        run.build({"records": [row], "controller": "mppi"})
    strict = copy.deepcopy(doc)
    strict["checkpoint"] = str(other.resolve())
    assert any("disagrees with header: checkpoint" in e for e in check(strict))


def test_legacy_minor_checkpoint_alias_basename_leniency():
    document = validator.load_json(D2 / "c5_malformed_wrong_legacy_checkpoint.json")
    document["header"]["checkpoint"] = {"path": "E:/Storage/runs/agent.pt", "sha256": "b" * 64}
    document["checkpoint"] = "D:/claude code/runs/agent.pt"  # abspath vs resolve() of one junction
    assert validator.alias_errors(document, 0) == []
    assert validator.alias_errors(document, 2) == ["legacy alias disagrees with header: checkpoint"]


# ---------------------------------------------------------------- item 9: code identity

def _synthetic_tree(root, env_source=b"class FamEnv:\n    pass\n"):
    files = {"scripts/eval_fake.py": b"# evaluator\r\n", "scripts/helper.py": b"X = 1\n",
             "tasks/_shared/__init__.py": b"", "tasks/fam/__init__.py": b"# family\n",
             "tasks/fam/fam_env.py": env_source, "tasks/fam/checkpoints/agent.pt": b"\x00w",
             "tasks/fam/agents/ppo.yaml": b"seed: 1\n", "assets/boat.usd": b"usd"}
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def _env_base(path, name):
    module = load(name, path)
    base = module.FamEnv()
    base.num_envs, base.device, base.termination_reason = 1, "cpu", [3]
    return base


def _write_release(tree, out, commit="c" * 40):
    manifest, records, digest = code_tree.manifest_from_tree(tree)
    out.mkdir(parents=True, exist_ok=True)
    rid = commit[:12] + "-" + digest[:12]
    (out / ("usvbench-%s.manifest" % rid)).write_bytes(manifest)
    descriptor = {"format": "usvbench-release-v1", "release_id": rid, "git_commit": commit, "git_tree": "d" * 40,
                  "code_tree_sha256": digest, "manifest_file": "usvbench-%s.manifest" % rid,
                  "inclusion": code_tree.INCLUSION_SPEC}
    path = out / ("usvbench-%s.release.json" % rid)
    path.write_text(json.dumps(descriptor, sort_keys=True, indent=1) + "\n", encoding="ascii")
    return path, rid, digest


def _run(evaluator, base, **cli):
    runtime = {key: "test" for key in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                      "isaac_lab_version", "physx_version")}
    return cert.CertificateRun(base=base, env_cfg=cfg(scene=SimpleNamespace(num_envs=1)), evaluator=evaluator,
                               task="Test-BlueBoat", controller="test", controller_config={}, checkpoint=None,
                               eval_seed=42, level=0, episodes_requested=1, overrides={}, argv=["eval.py"],
                               cli_args=dict(cli), runtime=runtime, synchronize=lambda: None,
                               clock=iter(i / 1000 for i in range(1000)).__next__)


def _finish(run):
    step(run)
    return run.build({"records": [run.finish_episode(0, rec(0, 0))]})


def test_working_tree_identity_is_recomputed_by_the_validator(tmp_path):
    tree = _synthetic_tree(tmp_path / "tree")
    base = _env_base(tree / "tasks" / "fam" / "fam_env.py", "e1_fam_env_a")
    run = _run(tree / "scripts" / "eval_fake.py", base)
    doc = _finish(run)
    code = doc["header"]["code"]
    assert code["hash_algorithm"] == "usvbench-code-tree-v1" and code["source"] == "working_tree"
    assert code["code_tree_sha256"] == code_tree.manifest_from_tree(tree)[2] and code["file_count"] == 6
    assert code["loaded_modules"]["checked"] == 1 and code["loaded_modules"]["verified"] is True
    assert code["environment_source"].endswith("fam_env.py")
    assert any("without a release descriptor" in note for note in doc["header"]["unverified"])
    assert not check(doc) and not check(doc, code_root=tree)
    (tree / "scripts" / "helper.py").write_bytes(b"X = 2\n")
    assert any("recomputed code tree" in e for e in check(doc, code_root=tree))
    assert any("UNVERIFIED code identity: working_tree" in e for e in check(doc, require_verified=True))


def test_release_identity_staged_copy_and_post_run_drift(tmp_path):
    tree = _synthetic_tree(tmp_path / "release")
    descriptor, rid, digest = _write_release(tree, tmp_path / "records")
    # Isaac Lab stages the families into direct/ (which also holds upstream tasks)
    direct = tmp_path / "IsaacLab" / "direct"
    (direct / "fam").mkdir(parents=True)
    (direct / "fam" / "__init__.py").write_bytes(b"# family\n")
    (direct / "fam" / "fam_env.py").write_bytes(b"class FamEnv:\n    pass\n")
    (direct / "cartpole").mkdir()
    (direct / "cartpole" / "cartpole_env.py").write_bytes(b"UPSTREAM = 1\n")
    load("e1_upstream_cartpole", direct / "cartpole" / "cartpole_env.py")
    base = _env_base(direct / "fam" / "fam_env.py", "e1_fam_env_staged")
    run = _run(tree / "scripts" / "eval_fake.py", base, release_descriptor=str(descriptor))
    code = run.header["code"]
    assert (code["source"], code["release_id"], code["code_tree_sha256"]) == ("release_descriptor", rid, digest)
    assert code["loaded_modules"]["checked"] == 1  # the staged env; upstream cartpole ignored
    assert not any("code identity" in note for note in run.unverified)
    manifest = tmp_path / "records" / ("usvbench-%s.manifest" % rid)
    doc = _finish(run)
    assert not check(doc, release_manifest=manifest)
    tampered = copy.deepcopy(doc)
    tampered["header"]["code"]["release_id"] = "0" * 12 + "-" + digest[:12]
    assert any("release_id does not match" in e for e in check(tampered))
    # a staged file that differs from the release is refused at construction
    (direct / "fam" / "fam_env.py").write_bytes(b"class FamEnv:\n    pass\n# local edit\n")
    with pytest.raises(ValueError, match="loaded modules differ from release"):
        _run(tree / "scripts" / "eval_fake.py", base, release_descriptor=str(descriptor))
    # ... and a change during the run is refused when the certificate is built
    (direct / "fam" / "fam_env.py").write_bytes(b"class FamEnv:\n    pass\n")
    run = _run(tree / "scripts" / "eval_fake.py", base, release_descriptor=str(descriptor))
    (direct / "fam" / "fam_env.py").write_bytes(b"class FamEnv:\n    pass\n# drift\n")
    with pytest.raises(ValueError, match="changed during the run"):
        _finish(run)


def test_evaluator_outside_release_and_descriptor_change_refused(tmp_path):
    tree = _synthetic_tree(tmp_path / "release")
    descriptor, _, _ = _write_release(tree, tmp_path / "records")
    base = _env_base(tree / "tasks" / "fam" / "fam_env.py", "e1_fam_env_b")
    (tree / "scripts" / "unreleased.py").write_bytes(b"# not in the manifest\n")
    with pytest.raises(ValueError, match="not the release's scripts/unreleased.py"):
        _run(tree / "scripts" / "unreleased.py", base, release_descriptor=str(descriptor))
    with pytest.raises(ValueError, match="changed after preflight"):
        _run(tree / "scripts" / "eval_fake.py", base, release_descriptor=str(descriptor),
             release_descriptor_sha256="0" * 64)
    old = json.loads(descriptor.read_text(encoding="ascii"))
    old.pop("inclusion")
    descriptor.write_text(json.dumps(old), encoding="ascii")
    with pytest.raises(ValueError, match="inclusion specification"):
        code_tree.load_release(descriptor)


def test_legacy_code_block_is_flagged_in_strict_minor_2():
    doc = fixtures.document(2, 1)
    assert doc["schema_minor_version"] == 2 and not validator.validate_certificate(doc, num_envs=2, episodes_per_env=1)
    errors = validator.validate_certificate(doc, num_envs=2, episodes_per_env=1, require_verified=True)
    assert any("legacy directory hash is not a release code tree" in e for e in errors)


def test_analysis_compares_the_release_code_tree(tmp_path):
    tree = _synthetic_tree(tmp_path / "tree")
    base = _env_base(tree / "tasks" / "fam" / "fam_env.py", "e1_fam_env_c")
    a = _finish(_run(tree / "scripts" / "eval_fake.py", base))
    b = copy.deepcopy(a)
    _compare_code_only(a, b)  # identical code tree: compatible
    b["header"]["code"]["code_tree_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="incompatible code.code_tree_sha256"):
        _compare_code_only(a, b)


def _compare_code_only(a, b):
    """The code-identity rule of analysis_input.validate_compatible, applied alone."""
    with patch.object(inputs, "validate_analysis_input", lambda doc: doc):
        inputs.validate_compatible([a, b])


def test_three_one_documents_remain_valid():
    doc = fixtures.document(2, 1)
    for key in ("hydrodynamics_model", "vessel_profile", "level_distance", "outcome_semantics"):
        doc["header"].pop(key)
    for row in doc["records"]:
        row.pop("outcome")
    doc["schema_minor_version"] = 1
    assert not validator.validate_certificate(doc, num_envs=2, episodes_per_env=1)
    doc["schema_minor_version"] = 2
    assert any("schema" in e for e in validator.validate_certificate(doc, num_envs=2, episodes_per_env=1))
    doc["schema_minor_version"] = 3
    assert any("schema" in e for e in validator.validate_certificate(doc, num_envs=2, episodes_per_env=1))


# ---------------------------------------------------------------- USV-10K speed reference (brief item 4)

def test_speed_reference_uses_d5_registry_for_v2_names():
    registry = cert.vehicles_v2_registry()
    assert registry is not None and set(registry.VEHICLES_V2) >= {"blueboat_v2", "wamv_vrx_v2_opt1"}
    blueboat = registry.VEHICLES_V2["blueboat_v2"]
    assert blueboat.propulsion.motors[0].forward_limit_n == pytest.approx(40.207, abs=5e-4)
    v2 = cert.nominal_speed_reference(None, name="blueboat_v2", registry=registry)
    assert v2["value_m_s"] == blueboat.speeds.top_model_m_s == pytest.approx(3.000, abs=5e-4)
    wamv = cert.nominal_speed_reference(None, name="wamv_vrx_v2_opt1", registry=registry)
    assert wamv["value_m_s"] == pytest.approx(2.259, abs=5e-4)
    legacy = load("e1_vehicles", REPO / "tasks" / "_shared" / "vehicles.py").get_vehicle("blueboat")
    assert cert.nominal_speed_reference(legacy, registry=registry)["value_m_s"] == pytest.approx(2.991, abs=5e-4)
    assert cert.nominal_speed_reference(SimpleNamespace(name="custom", speeds=SimpleNamespace(top_model_m_s=1.7)),
                                        registry=registry)["value_m_s"] == 1.7
    assert cert.vessel_family("blueboat_v2") == "blueboat" and cert.vessel_family("wamv_vrx_v2_opt1") == "wamv"
    base = SimpleNamespace(num_envs=2, device="cpu", termination_reason=[3, 3], vehicle_spec=SimpleNamespace(name="blueboat_v2"))
    _, doc = small_doc(base=base, env_cfg=cfg(vehicle="blueboat_v2"), cli_args={"vessel": "blueboat"})
    assert doc["header"]["speed_reference"]["value_m_s"] == pytest.approx(3.0003, abs=1e-3)
    assert "vehicles_v2 registry blueboat_v2" in doc["header"]["speed_reference"]["source"]


# ---------------------------------------------------------------- brief item 5: environment allow-list

def test_rental_kit_environment_is_stamped_not_refused():
    kit_environment = {"OMNI_KIT_ACCEPT_EULA": "YES", "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
                       "USVBENCH_ASSETS": "/root/autodl-tmp/usvb/releases/x/assets",
                       "VK_ICD_FILENAMES": "/root/autodl-tmp/usvb/vulkan/nvidia_icd.json"}
    with patch.dict("os.environ", kit_environment, clear=True):
        args = SimpleNamespace(eval_seed=5, level=0, num_envs=2, episodes=4, extra_sets=[], device="cpu",
                               allow_env=[], seed_namespace=None)
        protocol.preflight_protocol(args, ["eval.py"])
    assert args.runtime_environment == kit_environment and args.scientific_environment == {}
    with patch.dict("os.environ", dict(kit_environment, GOAL_RADIUS_OVERRIDE="3"), clear=True):
        with pytest.raises(ValueError, match="GOAL_RADIUS_OVERRIDE"):
            protocol.preflight_protocol(SimpleNamespace(eval_seed=5, level=0, num_envs=2, episodes=4, extra_sets=[],
                                                        allow_env=[], seed_namespace=None), ["eval.py"])


# ---------------------------------------------------------------- brief item 6 and run_cell release passthrough

def _cell_args(**extra):
    values = ["--task", "Isaac-USV-HarborMission-Direct-v1", "--vessel", "blueboat", "--controller", "classical",
              "--eval-seed-block", "pilot-01", "--seed-namespace", "dose_pilots", "--run-purpose", "pilot",
              "--level", "0", "--out", "x.json"]
    for key, value in extra.items():
        values += ["--" + key.replace("_", "-"), value]
    return runner.parser_for_cell().parse_args(values)


def test_harbour_classical_refusal_names_the_missing_controller():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError) as error:
            runner.resolve_cell(_cell_args())
    text = str(error.value)
    assert "no declared classical controller for task family 'harbor_mission'" in text
    assert "classical harbour-mission controller" in text and "HarborMissionEpisode" in text


def test_run_cell_plans_the_release_and_evaluator_checks_it(tmp_path):
    descriptor, rid, digest = _write_release(REPO, tmp_path / "records")
    with patch.dict("os.environ", {}, clear=True):
        args = _cell_args(task="Isaac-USV-HazardCross-Direct-v1", release_descriptor=str(descriptor))
        plan = runner.resolve_cell(args)
    assert plan["configuration"]["release"]["release_id"] == rid
    command = plan["command"]
    assert command[command.index("--release-descriptor") + 1] == str(descriptor.resolve())
    cell = {"configuration": plan["configuration"]}
    launched = SimpleNamespace(task="Isaac-USV-HazardCross-Direct-v1", level=0, episodes=128, num_envs=64,
                               seed_registration=plan["configuration"]["seed_registration"],
                               extra_sets=[], checkpoint=None, release_id=rid)
    protocol.check_cell_config(cell, launched, [str(SCRIPTS / "classical_planner_pid.py")])
    launched.release_id = None
    with pytest.raises(ValueError, match="planned for release"):
        protocol.check_cell_config(cell, launched, [str(SCRIPTS / "classical_planner_pid.py")])
