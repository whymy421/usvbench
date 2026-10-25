"""G6b: reference environment RE-1 (CONTEXT_R4 D28) - stamp, official-run refusal, validator.

This CPU box runs Python 3.13 / numpy 2.3.5, so the versions are INJECTED (reference_block(versions)
or a patched observed_versions); the real observation code runs once (block shape only).
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g6/test_g6b_reference_environment.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load("reference_environment")
protocol = _load("eval_protocol")
fixtures = _load("test_cert_v3")           # its document() is the REAL writer over a fake env
validator = fixtures.validator
cert = fixtures.cert
D28 = {"python": "3.11.15", "numpy": "1.26.0", "torch": "2.7.0+cu128", "isaacsim": "5.1.0.0",
       "isaaclab": "2.3.2", "hydra-core": "1.3.4", "omegaconf": "2.3.1", "skrl": "1.4.3"}


def test_pinned_file_holds_exactly_the_eight_d28_versions():
    env_id, required = R.load_required()
    assert env_id == "RE-1" and required == D28
    document = json.loads((SCRIPTS / "reference_environment.json").read_text(encoding="utf-8"))
    assert set(document["versions"]) == set(D28)


def test_block_match_and_mismatch_strings():
    good = R.reference_block(D28)
    assert good["match"] is True and good["mismatches"] == [] and good["id"] == "RE-1"
    bad = R.reference_block(dict(D28, numpy="2.3.5", skrl=None))
    assert bad["match"] is False
    assert bad["mismatches"] == ["numpy=2.3.5 (required 1.26.0)", "skrl=None (required 1.4.3)"]


def test_observed_block_has_every_name():
    block = R.reference_block()                      # real observation on this box (mismatching)
    assert set(block["versions"]) == set(D28) and block["versions"]["python"] == sys.version.split()[0]
    assert block["match"] is False and block["mismatches"]


def test_isaaclab_release_version_comes_from_the_checkout_version_file(tmp_path):
    """Checkout install: distribution metadata says the extension version 0.54.2, VERSION says
    2.3.2 (the Isaac Lab 2.3.2 reference copy: extension.toml version 0.54.2, VERSION 2.3.2)."""
    package = tmp_path / "IsaacLab" / "source" / "isaaclab" / "isaaclab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "IsaacLab" / "VERSION").write_text("2.3.2" + chr(10), encoding="utf-8")
    fake = SimpleNamespace(__file__=str(package / "__init__.py"))
    metadata = {"isaaclab": "0.54.2"}
    with patch.dict(sys.modules, {"isaaclab": fake}):
        with patch.object(R, "_distribution", lambda *names: metadata.get(names[0])):
            assert R.observed_versions()["isaaclab"] == "2.3.2"
    with patch.dict(sys.modules, {"isaaclab": SimpleNamespace(__file__=None)}):
        with patch.object(R.importlib.util, "find_spec", lambda name: None):
            with patch.object(R, "_distribution", lambda *names: {"isaaclab": "2.3.2"}.get(names[0])):
                assert R.observed_versions()["isaaclab"] == "2.3.2"      # pip wheel: release version


def official_args():
    """A registered pilot run (seed block pilot-01; scripts/seed_registry.json)."""
    return SimpleNamespace(task="Isaac-USV-HazardCross-Direct-v1", episodes=128, num_envs=64, level=0,
                           eval_seed=21001, seed_namespace="dose_pilots", seed_block="pilot-01",
                           run_purpose="pilot", extra_sets=[], allow_env=[], device="cuda:0", seed_registry=None,
                           speed_reference_m_s=None)


def diagnostic_args():
    return SimpleNamespace(task="Isaac-USV-HazardCross-Direct-v1", episodes=2, num_envs=2, level=0, eval_seed=42,
                           extra_sets=[], allow_env=[], device="cuda:0", seed_registry=None,
                           speed_reference_m_s=None)


def test_official_run_matching_passes_preflight_and_is_stamped():
    with patch.object(protocol.reference_environment, "observed_versions", lambda: dict(D28)):
        args = protocol.preflight_protocol(official_args(), ["eval_mppi.py"])
    assert args.reference_environment["match"] is True


def test_official_run_mismatching_is_refused_before_launch():
    with patch.object(protocol.reference_environment, "observed_versions", lambda: dict(D28, python="3.13.7")):
        with pytest.raises(ValueError, match=r"official run \(seed purpose 'pilot'\) outside the reference "
                                             r"environment RE-1 .*python=3.13.7 \(required 3.11.15\)"):
            protocol.preflight_protocol(official_args(), ["eval_mppi.py"])


def test_official_run_mismatching_exits_2_through_the_evaluator_parser():
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(SystemExit(2)))
    with patch.object(protocol.reference_environment, "observed_versions", lambda: dict(D28, numpy="2.3.5")):
        with pytest.raises(SystemExit) as raised:
            protocol.preflight_protocol(official_args(), ["eval_mppi.py"], parser)
    assert raised.value.code == 2


def test_diagnostic_run_mismatching_is_stamped_and_allowed():
    args = protocol.preflight_protocol(diagnostic_args(), ["classical_planner_pid.py"])   # real versions: mismatch
    assert args.seed_registration["purpose"] == "diagnostic"
    assert args.reference_environment["match"] is False and args.reference_environment["mismatches"]


def test_cert_runtime_metadata_stamps_the_block():
    with patch.object(cert.reference_environment, "observed_versions", lambda: dict(D28)):
        runtime = cert.runtime_metadata(SimpleNamespace(device="cpu"))
    assert runtime["reference_environment"] == R.reference_block(D28)


def _document(purpose, block):
    doc = copy.deepcopy(fixtures.document())
    doc["header"]["seed_namespace"]["purpose"] = purpose
    if purpose != "diagnostic":
        doc["header"]["seed_namespace"]["namespace"] = "dose_pilots"
    if block is None:
        doc["header"]["runtime"].pop("reference_environment", None)
    else:
        doc["header"]["runtime"]["reference_environment"] = block
    return doc


def _re_errors(doc):
    return [e for e in validator.validate_certificate(doc) if "reference_environment" in e or "D28" in e]


def test_validator_official_and_diagnostic_certificates():
    match, mismatch = R.reference_block(D28), R.reference_block(dict(D28, torch="2.9.1+cu126"))
    assert _re_errors(_document("pilot", match)) == []
    assert any("lacks runtime.reference_environment" in e for e in _re_errors(_document("pilot", None)))
    assert any("outside the reference environment RE-1" in e for e in _re_errors(_document("pilot", mismatch)))
    assert _re_errors(_document("diagnostic", mismatch)) == []
    assert _re_errors(_document("diagnostic", None)) == []
    forged = dict(mismatch, match=True, mismatches=[])
    assert any("disagree with its versions" in e for e in _re_errors(_document("diagnostic", forged)))
    schema_bad = dict(match, extra=1)
    assert any("schema" in e for e in validator.validate_certificate(_document("pilot", schema_bad)))
