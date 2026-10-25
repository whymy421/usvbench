"""CPU-only regression tests for strict analysis and metric availability."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

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


cert = load("cert_v3")
inputs = load("analysis_input")
deploy = load("deployment_metrics")
score = load("usv10k_score")
paired = load("paired_dose_test")


@pytest.fixture
def tmp_path():
    # Keep fixtures under this task even when pytest has no --basetemp flag.
    scratch = SCRIPTS.parents[1] / "test_scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="analysis-", dir=scratch) as directory:
        yield Path(directory)


def certificate(n=2, quota=2, *, success=True):
    """Real writer, fake CPU environment, synthetic explicit provenance."""
    config = SimpleNamespace(seed=41041, eval_level=0, curriculum_frozen=True,
                             vehicle="blueboat", physics_version="2",
                             episode_length_s=240.0, scene=SimpleNamespace(num_envs=n),
                             sea_state=SimpleNamespace(enable=True))
    base = SimpleNamespace(num_envs=n, device="cpu", termination_reason=[1 if success else 3] * n)
    provenance = {key: "a" * 64 for key in
                  ("tasks_sha256", "scripts_sha256", "tree_sha256", "evaluator_sha256")}
    provenance.update(tasks_root="fixture/tasks", scripts_root="fixture/scripts",
                      environment_source="fixture/env.py", evaluator_path="fixture/eval.py",
                      hash_algorithm="sha256-path-nul-content-v1")
    # E1: synthetic legacy digests must be self-consistent (validator recomputes tree_sha256, D2 item 10).
    provenance["tree_sha256"] = cert.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {key: "fixture" for key in ("hostname", "gpu_name", "gpu_device", "driver",
                                         "isaac_sim_version", "isaac_lab_version", "physx_version")}
    ticks = iter(i / 1000 for i in range(10000))
    run = cert.CertificateRun(base=base, env_cfg=config, evaluator=__file__,
                              task="Isaac-USV-StationKeep-BlueBoat-Direct-v1", controller="fake",
                              controller_config={}, checkpoint=None, eval_seed=41041, level=0,
                              episodes_requested=n * quota, overrides={}, argv=["fake.py"],
                              cli_args={"speed_reference_m_s": 2.0, "speed_reference_basis": "nominal"},
                              provenance=provenance, runtime=runtime, synchronize=lambda: None,
                              clock=lambda: next(ticks))
    records = []
    for ep in range(quota):
        with run.time_controller():
            pass
        with run.time_environment():
            pass
        run.after_step()
        for env in range(n):
            row = run.finish_episode(env, dict(env=env, ep=ep, success=success,
                                      terminated=success, truncated=not success,
                                      tts_s=180.0 if success else None, path_length_m=200.0,
                                      min_clearance_m=0.3, d0_m=10.0, route_geodesic_m=10.0,
                                      scenario_hashes={"spawn_pose": f"spawn-{env}-{ep}",
                                                       "wave": f"wave-{env}-{ep}",
                                                       "scenario": f"all-{env}-{ep}"}))
            row["path_length_success_prefix_m"] = 20.0 if success else None
            row["path_length_full_episode_m"] = 200.0
            records.append(row)
    document = run.build({"records": records, "scenario_protocol": {"version": 2, "hash": "blake2b"}})
    inputs.validate_analysis_input(document)
    return document


def write(tmp_path, document, name="gcert_fake_e41041.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document, allow_nan=False), encoding="ascii")
    return path


def test_valid_complete_grid_and_host_variation():
    a = certificate()
    assert inputs.validate_analysis_input(a) is a
    b = copy.deepcopy(a)
    b["records"].reverse()
    b["header"]["runtime"]["hostname"] = "other-host"
    b["header"]["runtime"]["gpu_name"] = "other-GPU"
    pairs = inputs.validate_pair(a, b)
    assert [(x["env"], x["ep"]) for x, _ in pairs] == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("mutate", [
    lambda d: d["records"].pop(),
    lambda d: d["records"].append(copy.deepcopy(d["records"][0])),
    lambda d: d["records"][0].update(env=99),
    lambda d: d["records"][0].update(env=True),
    lambda d: d["records"][0].update(ep=0.0),
    lambda d: d["header"].update(complete=False),
    lambda d: d["header"].update(episodes_requested=3),
    lambda d: d["records"][0].update(success="false"),
    lambda d: d["records"][0].update(tts_s="180"),
    lambda d: d["records"][0].update(min_clearance_m=float("nan")),
    lambda d: d["records"][0].update(path_length_full_episode_m=float("inf")),
    lambda d: d["records"][0].update(path_length_success_prefix_m=True),
    lambda d: d["records"][0].update(max_hold_s="4"),
    lambda d: d["records"][0].update(max_phase=1.5),
])
def test_bad_analysis_inputs_rejected(mutate):
    doc = certificate()
    mutate(doc)
    with pytest.raises(ValueError):
        inputs.validate_analysis_input(doc)


def test_legacy_and_duplicate_json_keys_rejected(tmp_path):
    with pytest.raises(ValueError, match="legacy"):
        inputs.validate_analysis_input({"task": "unknown-id", "records": []})
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":3,"schema_version":3}', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate JSON"):
        inputs.load_analysis_input(path)
    path.write_text('{"measurement":NaN}', encoding="ascii")
    with pytest.raises(ValueError, match="non-JSON"):
        inputs.load_analysis_input(path)


@pytest.mark.parametrize("field,value", [("physics_version", "99"), ("evaluation_protocol_version", "other")])
def test_incompatible_versions_rejected(field, value):
    a, b = certificate(), certificate()
    b["header"][field] = value
    with pytest.raises(ValueError, match=field):
        inputs.validate_pair(a, b)


@pytest.mark.parametrize("category,field", [("code", "tasks_sha256"), ("code", "scripts_sha256"),
                                             ("runtime", "physx_version")])
def test_incompatible_science_software_rejected(category, field):
    a, b = certificate(), certificate()
    b["header"][category][field] = "b" * 64 if category == "code" else "other"
    if category == "code":  # E1: keep the legacy digest self-consistent (the validator recomputes it)
        b["header"]["code"]["tree_sha256"] = cert.legacy_tree_digest(b["header"]["code"]["tasks_sha256"],
                                                                     b["header"]["code"]["scripts_sha256"])
    with pytest.raises(ValueError, match="incompatible"):
        inputs.validate_pair(a, b)


def test_all_primitives_checked_not_just_spawn():
    a, b = certificate(), certificate()
    b["records"][1]["scenario_hashes"]["wave"] = "different"
    with pytest.raises(ValueError, match="unchanged primitive differs: wave"):
        inputs.validate_pair(a, b)
    b["records"][1]["scenario_hashes"]["scenario"] = "different aggregate"
    assert len(inputs.validate_pair(a, b, ["wave"])) == 4
    with pytest.raises(ValueError, match="derived aggregate"):
        inputs.validate_pair(a, b, ["scenario"])


def test_missing_primitives_even_on_both_sides_rejected():
    a, b = certificate(), certificate()
    for doc in (a, b):
        for row in doc["records"]:
            del row["scenario_hashes"]["wave"]
    with pytest.raises(ValueError, match="primitive is missing"):
        inputs.validate_pair(a, b)
    for doc in (a, b):
        for row in doc["records"]:
            row.pop("scenario_hashes")
    with pytest.raises(ValueError, match="hash maps"):
        inputs.validate_pair(a, b)


def test_partial_hash_coverage_or_primitive_inventory_rejected():
    a, b = certificate(), certificate()
    del b["records"][0]["scenario_hashes"]["wave"]
    with pytest.raises(ValueError, match="inventories differ"):
        inputs.validate_pair(a, b)


def test_declared_unchanged_primitive_cannot_be_exempted():
    a, b = certificate(), certificate()
    for doc in (a, b):
        doc["header"]["pairing"] = {"unchanged_primitives": ["spawn_pose", "wave"],
                                   "changed_primitives": []}
    with pytest.raises(ValueError, match="differs from certified"):
        inputs.validate_pair(a, b, ["wave"])


def test_deadline_is_resolved_header_then_c5_snapshot(tmp_path):
    doc = certificate()
    doc["header"]["resolved_deadline_s"] = 240.0
    row = deploy.analyse_arm(str(write(tmp_path, doc)), (0.25,), 10, 1, {})
    assert row["t_budget_s"] == 240.0
    assert row["cvar25"] == 0.75
    del doc["header"]["resolved_deadline_s"]
    assert inputs.resolved_deadline(doc) == 240.0
    doc["header"]["resolved_deadline_s"] = None
    with pytest.raises(ValueError, match="deadline"):
        inputs.resolved_deadline(doc)


def test_no_guessed_deadline_or_posthoc_budget_override(tmp_path):
    doc = certificate()
    path = write(tmp_path, doc)
    with pytest.raises(ValueError, match="certified deadline"):
        deploy.analyse_arm(str(path), (0.25,), 10, 1, {doc["task"]: 120})
    doc["header"].pop("resolved_deadline_s")
    doc["header"]["resolved_config"].pop("episode_length_s")
    with pytest.raises(ValueError, match="deadline"):
        inputs.resolved_deadline(doc)
    # Frozen v1 loss definition (pre-registered): an unmeasured success costs
    # 1.0 with a warning; a present non-finite or coerced TTS is rejected.
    warns = []
    assert deploy.episode_loss({"success": True, "tts_s": None}, 240, warns) == 1.0 and warns
    for bad in (float("nan"), "180", True, -1.0):
        with pytest.raises(ValueError):
            deploy.episode_loss({"success": True, "tts_s": bad}, 240)


def test_deployment_does_not_skip_invalid_population(tmp_path):
    good = write(tmp_path, certificate())
    doc = certificate()
    doc["records"].pop()
    bad = write(tmp_path, doc, "gcert_bad.json")
    output = tmp_path / "metrics.csv"
    assert deploy.main([str(good), str(bad), "--out-csv", str(output), "--boot", "10"]) == 2
    assert not output.exists()


def test_score_missing_components_not_perfect_credit():
    result = score.episode_score({"success": True})
    assert result["score"] is None
    assert result["efficiency"] is None and result["safety"] is None
    assert result["r_task"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.3", True])
def test_score_rejects_nonfinite_or_coerced_clearance(value):
    with pytest.raises(ValueError):
        score.episode_score({"success": True, "min_clearance_m": value})


def test_scoring_uses_certified_speed_and_success_prefix():
    doc = certificate()
    for row in doc["records"]:
        row["tts_s"] = 10.0
    assert score.score_certificate(doc)["score"] == 2500.0
    doc["header"]["vessel"] = "wamv"
    doc["header"]["resolved_config"]["vehicle"] = "wamv"  # E1: validator cross-checks vessel vs cfg
    doc["header"]["speed_reference"] = {"vessel": "wamv", "value_m_s": 1.0, "basis": "condition_adjusted"}
    assert score.score_certificate(doc)["score"] == 5000.0
    doc["header"]["speed_reference"]["vessel"] = "blueboat"
    with pytest.raises(ValueError, match="vessel"):
        score.score_certificate(doc)


@pytest.mark.parametrize("missing", ["path_length_success_prefix_m", "min_clearance_m", "tts_s", "route_geodesic_m"])
def test_partial_component_null_blocks_aggregate_rank(missing):
    doc = certificate()
    doc["records"][0][missing] = None
    if missing == "route_geodesic_m":
        doc["records"][0]["d0_m"] = None
    result = score.score_certificate(doc)
    assert result["score"] is None and result["score_complete"] is False
    assert result["n"] == 4


def test_missing_speed_is_unavailable_not_blueboat_default():
    doc = certificate()
    doc["header"]["speed_reference"] = None
    result = score.score_certificate(doc)
    assert result["score"] is None
    assert "speed_reference" in result["unavailable_components"]


def test_only_terminal_success_c5_path_fallback():
    rec = certificate()["records"][0]
    del rec["path_length_success_prefix_m"]
    assert score.episode_score(rec, speed_reference_m_s=2)["score"] is not None
    rec["termination_reason"] = "timeout"
    assert score.episode_score(rec, speed_reference_m_s=2)["score"] is None


def test_bootstrap_power_of_two_is_not_permutation_artifact():
    lo, hi = paired.bootstrap_median_ci(list(range(128)), iterations=300)
    assert lo < 63.5 < hi
    assert paired.bootstrap_median_ci(list(range(128)), iterations=300) == (lo, hi)
    assert paired.sign_test([0.0] * 8) == (0, 0, None)
    assert paired.bootstrap_median_ci([1.0]) == (None, None)


def test_paired_cli_rejects_incomplete_not_intersection(tmp_path):
    a, b = certificate(), certificate()
    b["records"].pop()
    ap, bp = write(tmp_path, a), write(tmp_path, b, "gcert_b.json")
    assert paired.main([str(ap), str(bp)]) == 2


def test_paired_cli_counts_successes_independent_of_endpoint(tmp_path, capsys):
    a, b = certificate(), certificate()
    a["records"][0]["path_length_full_episode_m"] = None
    ap, bp = write(tmp_path, a), write(tmp_path, b, "gcert_b.json")
    assert paired.main([str(ap), str(bp), "--bootstrap", "20",
                        "--field", "path_length_full_episode_m"]) == 0
    output = capsys.readouterr().out
    assert "both succeeded: 4" in output
    assert "1/4 overall" in output
    assert "undefined (all ties)" in output


def test_analysis_imports_no_isaac():
    assert not any(name == "isaaclab" or name.startswith("isaaclab.") for name in sys.modules)
