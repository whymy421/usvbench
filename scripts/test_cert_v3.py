"""CPU-only v3 contract/provenance/validator tests, loaded directly by path."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cert = load("cert_v3")
validator = load("validate_cert_v3")


def make_run(n=64, requested=128, **kwargs):
    config = SimpleNamespace(seed=42, eval_level=0, curriculum_frozen=True, vehicle="blueboat",
                             sea_state=SimpleNamespace(hs_range=(0.6, 0.6)), drag_scale=2.0,
                             scene=SimpleNamespace(num_envs=n), suite_s_class="single_row")
    base = SimpleNamespace(num_envs=n, device="cpu", termination_reason=[3] * n)
    provenance = {key: "a" * 64 for key in ("tasks_sha256", "scripts_sha256", "tree_sha256", "evaluator_sha256")}
    provenance.update(tasks_root="test/tasks", scripts_root="test/scripts", environment_source="test/tasks/family/env.py",
                      evaluator_path="test/scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
    # E1: synthetic legacy digests must be self-consistent (validator recomputes tree_sha256, D2 item 10).
    provenance["tree_sha256"] = cert.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {key: "test" for key in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                      "isaac_lab_version", "physx_version")}
    # G6b / CONTEXT_R4 D28: official certificate fixtures come from an (injected) RE-1 environment.
    runtime["reference_environment"] = cert.reference_environment.reference_block(
        cert.reference_environment.load_required()[1])
    counter = iter(i / 1000 for i in range(10000))
    options = dict(base=base, env_cfg=config, evaluator=__file__, task="Test-BlueBoat", controller="test",
                   controller_config={"kp": 2.0}, checkpoint=None, eval_seed=42, level=0,
                   episodes_requested=requested, overrides={"sea_state.hs_range": (0.6, 0.6), "drag_scale": 2.0},
                   argv=["eval.py", "--set", "sea_state.hs_range=0.6,0.6"], cli_args={},
                   provenance=provenance, runtime=runtime, synchronize=lambda: None, clock=lambda: next(counter))
    options.update(kwargs)
    return cert.CertificateRun(**options)


def sample_record(env=0, ep=0):
    return dict(env=env, ep=ep, success=False, terminated=False, truncated=True,
                tts_s=None, path_length_m=1.0, min_clearance_m=float("nan"),
                d0_m=2.0, route_geodesic_m=None)


def step(run):
    with run.time_controller():
        pass
    with run.time_environment():
        pass
    run.after_step()


def document(n=64, quota=2):
    run = make_run(n, n * quota)
    records = []
    for ep in range(quota):
        step(run)
        for env in range(n):
            records.append(run.finish_episode(env, sample_record(env, ep)))
    return run.build({"records": records})


def test_complete_default_grid_and_json_is_strict():
    doc = document()
    assert not validator.validate_certificate(doc)
    json.dumps(doc, allow_nan=False)
    assert doc["header"]["suite_presets"] == {"suite_s_class": "single_row"}
    assert doc["records"][0]["min_clearance_m"] is None
    assert doc["records"][0]["contact_steps"] is None


@pytest.mark.parametrize("mutation,message", [
    (lambda d: d["records"].append(copy.deepcopy(d["records"][0])), "duplicate"),
    (lambda d: d["records"].pop(), "incomplete grid"),
    (lambda d: d["records"][0].update(env=64), "outside declared grid"),
    (lambda d: d["records"][0].update(ep=2), "outside declared grid"),
    (lambda d: d["records"][0].update(ep=True), "schema"),
    (lambda d: d["records"][0].update(termination_reason="crash"), "schema"),
    (lambda d: d["records"][0].update(termination_reason="success"), "terminal reason"),
    (lambda d: d["records"][0].update(tts_s=3.0), "failed episode"),
    (lambda d: d["records"][0].update(terminated=False, truncated=False), "neither stop mask"),
    (lambda d: d["header"].update(complete=False), "complete stamp"),
    (lambda d: d["header"].update(episodes_requested=64), "ceil"),
    (lambda d: d.update(complete=False), "legacy alias"),
    (lambda d: d["header"]["checkpoint"].update(path="x.pt"), "checkpoint path"),
    (lambda d: d["header"].update(ended_at="1900-01-01T00:00:00Z"), "end time"),
    (lambda d: d["header"]["code"].update(tasks_sha256="bad"), "schema"),
    (lambda d: d["records"][0]["controller_compute_time_s"].update(n=0), "schema"),
    (lambda d: d["records"][0]["controller_compute_time_s"].update(total=100), "total !="),
    (lambda d: d["records"][0]["environment_step_time_s"].update(n=2), "counts differ"),
    (lambda d: d["records"][0]["controller_compute_time_s"].update(min=2), "inconsistent"),
    (lambda d: d["records"][0].update(d0_m=float("nan")), "non-finite"),
])
def test_reject_corrupt_certificates(mutation, message):
    doc = document()
    mutation(doc)
    assert any(message in error for error in validator.validate_certificate(doc))


def test_incomplete_is_explicit_and_default_grid_is_fixed():
    doc = document(2, 1)
    assert any("expected 64 x 2" in e for e in validator.validate_certificate(doc))
    assert not validator.validate_certificate(doc, num_envs=2, episodes_per_env=1)
    doc["records"].pop()
    doc["header"]["complete"] = doc["complete"] = False
    assert not validator.validate_certificate(doc, num_envs=2, episodes_per_env=1, allow_incomplete=True)
    assert validator.validate_certificate(doc, num_envs=2, episodes_per_env=1)


def test_override_command_actual_readback_last_wins_and_precedence():
    doc = document()
    argv = ["python", "eval.py", "--axis", "drag_scale", "--value", "8", "--set=drag_scale=3",
            "--set", "drag_scale=2", "--set", "sea_state.hs_range=(0.6,0.6)"]
    assert not validator.validate_certificate(doc, command=argv)
    assert any("differs from stamp" in e for e in validator.validate_certificate(doc, command=["eval.py", "--set", "drag_scale=3"]))
    del doc["header"]["overrides"]["sea_state.hs_range"]
    doc["overrides"] = copy.deepcopy(doc["header"]["overrides"])
    assert any("no stamp" in e for e in validator.validate_certificate(doc))


def test_override_stamp_must_match_config_and_seed_is_resolved():
    run = make_run()
    run.header["overrides"]["drag_scale"] = 999
    doc = run.build({"records": []})
    assert any("differs from resolved_config" in e for e in validator.validate_certificate(doc))
    cfg = SimpleNamespace(seed=99, eval_level=3, vehicle="blueboat", drag_scale=2.,
                          sea_state=SimpleNamespace(hs_range=(.6, .6)))
    run = make_run(env_cfg=cfg, argv=["eval.py", "--eval-seed", "42", "--level", "0", "--set", "seed=99", "--set", "eval_level=3"])
    assert run.header["eval_seed"] == 99 and run.header["level"] == 3
    assert not validator.override_errors(run.header, run.header["command"]["argv"])


def test_timeout_success_and_simultaneous_masks_are_independent():
    doc = document()
    fixed = doc["header"]["outcome_semantics"]["fixed_horizon"]
    doc["records"][0].update(success=True, tts_s=1.0)
    # E1: records carry the derived outcome (D2 item 21); keep it consistent with the mutation.
    doc["records"][0]["outcome"] = validator.episode_outcome(doc["records"][0], fixed)
    assert not validator.validate_certificate(doc)
    doc["records"][0].update(terminated=True, truncated=True, termination_reason="collision")
    doc["records"][0]["outcome"] = validator.episode_outcome(doc["records"][0], fixed)
    assert not validator.validate_certificate(doc)


def test_timing_and_stage_latches_clear_only_finished_env():
    run = make_run(2, 4)
    run.base.episode_max_phase = [2, 1]
    run.base.episode_gates_passed = [3, 2]
    run.base.episode_route_length = [12.0, 13.0]
    step(run)
    a = run.finish_episode(0, sample_record(0))
    assert a["stage_attainment"] == {"max_phase": 2, "gates_passed": 3}
    assert a["route_geodesic_m"] == 12.0
    step(run)
    b = run.finish_episode(1, sample_record(1))
    assert a["controller_compute_time_s"]["n"] == 1
    assert b["controller_compute_time_s"]["n"] == 2
    run.discard_episode(0)
    step(run)
    c = run.finish_episode(0, sample_record(0, 1))
    assert c["environment_step_time_s"]["n"] == 1


def test_timer_synchronization_and_missing_measurement_rejected():
    calls = []
    run = make_run(synchronize=lambda: calls.append("sync"))
    with run.time_controller():
        calls.append("act")
    assert calls == ["sync", "act", "sync"]
    with pytest.raises(RuntimeError):
        run.after_step()


def test_timing_population_statistics_and_interpolation():
    result = cert.timing_stats([1, 2, 3, 4])
    assert result["mean"] == 2.5 and result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["std"] == pytest.approx(1.25 ** .5)
    assert cert.timing_stats([])["mean"] is None
    with pytest.raises(ValueError):
        cert.timing_stats([float("nan")])


def test_snapshot_inherited_config_cycles_and_nonfinite():
    class Cfg:
        inherited = (1, 2)
        def __init__(self):
            self.value = float("inf")
            self.child = self
    snapshot = cert.json_value(Cfg())
    assert snapshot["inherited"] == [1, 2]
    assert snapshot["value"] is None
    assert "__cycle__" in snapshot["child"]
    json.dumps(snapshot, allow_nan=False)


def test_hashes_cover_file_bytes_paths_and_ignore_caches(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="ascii")
    before = cert.tree_hash(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/junk.pyc").write_bytes(b"cache")
    assert before == cert.tree_hash(tmp_path)
    (tmp_path / "a.py").rename(tmp_path / "b.py")
    assert before != cert.tree_hash(tmp_path)
    moved = cert.tree_hash(tmp_path)
    (tmp_path / "b.py").write_text("x = 2\n", encoding="ascii")
    assert moved != cert.tree_hash(tmp_path)


def test_checkpoint_and_code_provenance(tmp_path):
    scripts = tmp_path / "scripts"
    tasks = tmp_path / "tasks"
    scripts.mkdir()
    tasks.mkdir()
    evaluator = scripts / "eval.py"
    evaluator.write_text("pass\n", encoding="ascii")
    (tasks / "task.py").write_text("pass\n", encoding="ascii")
    provenance = cert.code_provenance(evaluator, SimpleNamespace())
    assert provenance["evaluator_sha256"] == cert.sha256_file(evaluator)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"cpu fixture only")
    run = make_run(checkpoint=checkpoint)
    assert run.header["checkpoint"]["sha256"] == cert.sha256_file(checkpoint)
    with pytest.raises(FileNotFoundError):
        make_run(checkpoint=tmp_path / "missing.pt")


def test_legacy_report_is_read_only_and_missing_is_not_fabricated(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"task":"Old","seed":42,"records":[{"env":0,"ep":0,"success":false,"min_clearance_m":NaN}]}', encoding="ascii")
    original = path.read_bytes()
    report = validator.legacy_report(validator.load_json(path, legacy=True))
    assert "code" in report["missing_header_fields"]
    assert report["missing_record_fields"]["termination_reason"]["count"] == 1
    assert report["available_header"]["task_id"] == "Old"
    assert report["certificate_created"] is False
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        validator.load_json(path)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"complete":true,"complete":false}', encoding="ascii")
    with pytest.raises(ValueError, match="duplicate"):
        validator.load_json(path)


def test_cli_prints_one_line_and_fails_mismatch(tmp_path):
    path = tmp_path / "cert.json"
    path.write_text(json.dumps(document()), encoding="ascii")
    command = [sys.executable, "-B", str(SCRIPTS / "validate_cert_v3.py"), str(path)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0 and len(result.stdout.splitlines()) == 1
    result = subprocess.run(command + ["--command-argv", "eval.py", "--set", "drag_scale=9"], text=True, capture_output=True)
    assert result.returncode == 1 and result.stdout.startswith("FAIL ")
    assert "differs from stamp" in result.stderr


def test_runtime_unknowns_are_explicit():
    run = make_run(runtime={"hostname": "test", "gpu_name": None, "gpu_device": "cpu", "driver": None,
                            "isaac_sim_version": None, "isaac_lab_version": None, "physx_version": None})
    assert "runtime.physx_version" in run.header["unverified"]
    doc = document()
    assert validator.validate_certificate(doc, require_verified=True)


def test_invalid_requested_counts_and_unknown_termination():
    with pytest.raises(ValueError):
        make_run(1, 0)
    run = make_run(1, 1)
    run.base.termination_reason = [0]
    step(run)
    result = run.finish_episode(0, sample_record())
    assert result["termination_reason"] == "unknown"
    assert any("termination_reason" in note for note in run.unverified)


@pytest.mark.parametrize("bad", [{"header": []}, {"header": None}, {"header": "bad"}, [], None])
def test_malformed_input_has_one_line_summary(bad, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="ascii")
    result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "validate_cert_v3.py"), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 1 and len(result.stdout.splitlines()) == 1
    assert "Traceback" not in result.stderr


def test_large_invalid_grid_is_bounded():
    doc = document()
    doc["header"]["num_envs"] = 10 ** 12
    doc["header"]["episodes_per_env"] = 10 ** 12
    assert any("expected 64 x 2" in error for error in validator.validate_certificate(doc))


def test_strict_mode_derives_unknown_provenance_and_cross_checks_seed():
    doc = document()
    doc["header"]["unverified"] = []
    doc["header"]["runtime"]["driver"] = None
    assert any("missing provenance" in error for error in validator.validate_certificate(doc, require_verified=True))
    doc["header"]["eval_seed"] = 99
    assert any("resolved_config/header" in error for error in validator.validate_certificate(doc))


def test_hydra_override_after_boolean_flag_is_stamped_and_checked():
    run = make_run(argv=["eval.py", "--headless", "env.sea_state.hs_range=[0.6,0.6]"])
    assert run.header["overrides"]["sea_state.hs_range"] == [.6, .6]
    assert not validator.override_errors(run.header, run.header["command"]["argv"])
    errors = validator.override_errors(run.header, ["eval.py", "--headless", "env.sea_state.hs_range=[0.1,0.1]"])
    assert any("differs from stamp" in error for error in errors)


def test_legacy_converter_nested_missing_and_input_type():
    report = validator.legacy_report({"checkpoint": "old.pt", "controller": "pid", "records": []})
    assert "checkpoint.sha256" in report["missing_header_fields"]
    assert "controller.configuration" in report["missing_header_fields"]
    with pytest.raises(ValueError):
        validator.legacy_report([])


def test_spawn_distance_legacy_convenience_field():
    run = make_run()
    run.header["resolved_config"]["curriculum_start_distance_m"] = 8.0
    run.header["overrides"]["curriculum_start_distance_m"] = 8.0
    assert not validator.override_errors(run.header, ["eval.py", "--spawn-distance", "8"])


def test_initial_distance_fallback_belongs_to_finished_episode():
    run = make_run(2, 4)
    distances = [10., 20.]
    run.base._horizontal_distance = lambda: list(distances)
    run.after_reset()
    step(run)
    distances[:] = [100., 20.]
    record = sample_record(0)
    record["d0_m"] = None
    assert run.finish_episode(0, record)["d0_m"] == 10.
    step(run)
    distances[:] = [200., 300.]
    # D8: finish_episode now enforces the next preassigned (env, ep) key, so
    # each finished episode carries its own ep index (fixture data only; the
    # asserted d0 values are the C5 expectations, unchanged).
    record["ep"] = 1
    assert run.finish_episode(0, record)["d0_m"] == 100.
    record["env"], record["ep"] = 1, 0
    assert run.finish_episode(1, record)["d0_m"] == 20.


def test_version_probe_never_labels_extension_as_physx_sdk(monkeypatch):
    def absent(name):
        raise cert.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(cert.importlib.metadata, "version", absent)
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(__version__="lab-test"))
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "omni.physx", SimpleNamespace(__version__="extension-only"))
    metadata = cert.runtime_metadata(SimpleNamespace(device="cpu"))
    assert metadata["isaac_lab_version"] == "lab-test"
    assert metadata["physx_version"] is None and metadata["isaac_sim_version"] is None
    assert metadata["gpu_name"] is None and metadata["driver"] is None
