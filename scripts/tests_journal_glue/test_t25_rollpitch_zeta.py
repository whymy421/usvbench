"""T25 (owner decision T1.35b): diagnostic-only roll/pitch damping-ratio override hydro.rollpitch_zeta=<x>.

* token grammar and validation (conditions/rollpitch_override.py), resolution beside the suite stack
  (journal_config.resolve_from_args_with_overrides) with an unchanged JournalConditions;
* the REAL mixin + HydroV2Runtime (glue_fixtures.FakeJournalEnv): the override selects the existing
  hydro_profiles "zeta:<x>" rule for roll and pitch (c = 2 zeta sqrt(K I)), nothing else; without it the
  cfg carries no attribute and the profile is the journal default; the header stamp says pinned vs used;
* official runs are refused: eval_protocol preflight (before the RE-1 check, i.e. before the simulator),
  eval_protocol.apply_journal_suites, run_cell, cert_v3 (rollpitch_damping_official_check), and the REAL
  evaluator scripts exit 2 before the app;
* validate_cert_v3: an overridden certificate is non-official (error when its seed purpose is official);
  consistency with the registry and the condition stamp; tampering is caught; the schema accepts the stamp.
Same pattern and harness as scripts/tests_journal_glue/test_g16_wind_coefficients.py (G16). CPU only.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
import glue_fixtures as G  # noqa: E402
import torch  # noqa: E402

RP = G.shared("conditions.rollpitch_override")
JC = G.journal_config
HP = G.shared("hydro_profiles")


def _zeta_pair(vessel, zeta):
    spec = G.vehicles_v2.get_vehicle_v2(vessel)
    h, inertia = spec.hydrostatics, spec.inertia_com_kg_m2
    return (2.0 * zeta * math.sqrt(h.stiffness_roll_nm_rad * inertia[0][0]),
            2.0 * zeta * math.sqrt(h.stiffness_pitch_nm_rad * inertia[1][1]))


# ------------------------------------------------------------------ grammar --
def test_split_tokens_and_validation():
    rest, override = RP.split_tokens(["waves=D4", "hydro.rollpitch_zeta=0.2", "wind=D1"])
    assert rest == ["waves=D4", "wind=D1"] and override == {"rollpitch_zeta": 0.2}
    assert RP.split_tokens(["waves=D2"]) == (["waves=D2"], None)
    assert RP.split_tokens(["hydro.rollpitch_zeta=0.1", "hydro.rollpitch_zeta=0.1"])[1] == {"rollpitch_zeta": 0.1}
    for bad in (["hydro.rollpitch_zeta=0.1", "hydro.rollpitch_zeta=0.2"], ["hydro.roll_zeta=0.1"], ["hydro=0.1"],
                ["hydro.rollpitch_zeta=0"], ["hydro.rollpitch_zeta=-0.1"], ["hydro.rollpitch_zeta=2.5"],
                ["hydro.rollpitch_zeta=NaN"], ["hydro.rollpitch_zeta=true"], ["hydro.rollpitch_zeta=\"0.1\""],
                ["hydro.rollpitch_zeta=abc"], ["hydro.rollpitch_zeta.x=0.1"]):
        with pytest.raises(RP.RollPitchOverrideError):
            RP.split_tokens(bad)
    assert RP.is_override_token("hydro.rollpitch_zeta=0.3") and not RP.is_override_token("waves=D2")
    assert RP.rule_label({"rollpitch_zeta": 0.05}) == "zeta:0.05"
    assert RP.validate_override(None) is None
    with pytest.raises(RP.RollPitchOverrideError):
        RP.validate_override({"rollpitch_zeta": 0.1, "x": 1})


def test_resolution_leaves_the_suite_stack_unchanged(tmp_path):
    plain = JC.resolve_from_args(SimpleNamespace(suite=["waves=D4", "wind=D1"], suite_file=None), seed=42)
    both, wind, rollpitch = JC.resolve_from_args_with_overrides(
        SimpleNamespace(suite=["waves=D4", "hydro.rollpitch_zeta=0.2", "wind=D1", "windcoef.c_n0=0.11"],
                        suite_file=None), seed=42)
    assert rollpitch == {"rollpitch_zeta": 0.2} and wind == {"c_n0": 0.11}
    assert both == plain and both.stamp() == plain.stamp() and both.tokens() == plain.tokens()
    none, no_wind, no_rp = JC.resolve_from_args_with_overrides(
        SimpleNamespace(suite=["waves=D4", "wind=D1"], suite_file=None), seed=42)
    assert no_wind is None and no_rp is None and none == plain
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps({"suite": ["waves=D2", "hydro.rollpitch_zeta=0.05"]}), encoding="ascii")
    from_file, _, file_rp = JC.resolve_from_args_with_overrides(SimpleNamespace(suite=[], suite_file=str(suite_file)),
                                                                seed=42)
    assert file_rp == {"rollpitch_zeta": 0.05}
    assert from_file == JC.resolve_from_args(SimpleNamespace(suite=["waves=D2"], suite_file=None), seed=42)
    with pytest.raises(JC.SuiteConflict):
        JC.resolve_from_args_with_overrides(SimpleNamespace(
            suite=["hydro.rollpitch_zeta=0.1", "hydro.rollpitch_zeta=0.2"], suite_file=None), seed=42)
    # the plain resolver and G16's resolver never accept it (admission tools and fixtures stay fail-closed)
    with pytest.raises(JC.SuiteConflict):
        JC.resolve_from_args(SimpleNamespace(suite=["hydro.rollpitch_zeta=0.1"], suite_file=None), seed=42)
    with pytest.raises(JC.SuiteConflict):
        JC.resolve_from_args_with_wind_coefficients(SimpleNamespace(suite=["hydro.rollpitch_zeta=0.1"],
                                                                    suite_file=None), seed=42)
    cfg = SimpleNamespace()
    JC.apply_rollpitch_override(cfg, {"rollpitch_zeta": 0.2})
    assert cfg.journal_rollpitch_zeta == 0.2 and RP.cfg_override(cfg) == {"rollpitch_zeta": 0.2}
    with pytest.raises(ValueError):
        JC.apply_rollpitch_override(SimpleNamespace(), None)


# ---------------------------------------------------------- real runtime --
def _env(vessel, tokens, zeta=None, num_envs=2, options=None):
    cfg = G.journal_cfg(vessel, tokens=tokens)
    if options is not None:
        cfg.journal_vehicle_options = dict(options)
    if zeta is not None:
        JC.apply_rollpitch_override(cfg, {"rollpitch_zeta": zeta})
    env = G.FakeJournalEnv(cfg, num_envs)
    env.reset()
    return env


@pytest.mark.parametrize("vessel", ["blueboat_v2", "wamv_vrx_v2_opt1"])
def test_override_selects_the_zeta_rule_for_roll_and_pitch_only(vessel):
    default = _env(vessel, ["waves=D2"])
    over = _env(vessel, ["waves=D2"], zeta=0.2)
    assert not hasattr(default.cfg, "journal_rollpitch_zeta")
    reference = HP.profile_for_vehicle(vessel)
    hd, ho = default._hydro_v2.profile.hydro, over._hydro_v2.profile.hydro
    assert hd == reference.hydro                                   # absent: the journal default profile
    assert tuple(ho.angular_linear_damping[:2]) == _zeta_pair(vessel, 0.2)
    assert ho.angular_linear_damping[2] == hd.angular_linear_damping[2]
    for field in ("mass_kg", "inertia_kgm2", "linear_damping", "quadratic_damping", "angular_quadratic_damping",
                  "roll_stiffness_nm", "pitch_stiffness_nm", "equilibrium_com_z_m"):
        assert getattr(ho, field) == getattr(hd, field), field
    assert over._hydro_v2.profile.thrusters == default._hydro_v2.profile.thrusters
    assert over.journal_vehicle.hydro_profile is over._hydro_v2.profile
    sd, so = default.journal_rollpitch_damping_stamp(), over.journal_rollpitch_damping_stamp()
    assert sd["pinned_values"] is True and sd["override"] is None and sd["used"] == sd["pinned"]
    # the journal default of hydro_profiles (decision 2; T1.35b switch patch changes the WAM-V entry)
    assert sd["pinned"]["rule"] == (HP._BLUEBOAT_DEFAULT_RULE if vessel == "blueboat_v2" else HP._WAMV_DEFAULT_RULE)
    assert so["pinned_values"] is False and so["override"] == {"rollpitch_zeta": 0.2}
    assert so["pinned"] == sd["pinned"] and so["used"]["rule"] == "zeta:0.2"
    assert (so["used"]["roll_nms_rad"], so["used"]["pitch_nms_rad"]) == _zeta_pair(vessel, 0.2)
    assert so["schema"] == RP.STAMP_SCHEMA and so["vessel_profile"] == vessel
    rot = over.journal_certificate_header()["conditions"]["context"]["rotational_damping"]
    assert rot["angular_linear_damping"][:2] == list(_zeta_pair(vessel, 0.2))
    # the mixin's own header blocks keep their key set (G10 neutrality test) - the stamp is read by cert_v3
    assert set(over.journal_certificate_header()) == set(default.journal_certificate_header())
    # in waves the damping changes the trajectory
    for env in (default, over):
        for _ in range(30):
            env.step(torch.tensor([[0.5, 0.5], [0.3, -0.3]]))
    assert not torch.equal(default.robot.data.root_link_pose_w, over.robot.data.root_link_pose_w)


def test_absent_override_is_bitwise_the_default():
    cfg = G.journal_cfg("wamv_vrx_v2_opt1", tokens=["waves=D3"])
    assert RP.vehicle_options(cfg) == dict(getattr(cfg, "journal_vehicle_options", None) or {})
    assert RP.cfg_override(cfg) is None
    a = _env("wamv_vrx_v2_opt1", ["waves=D3"])
    b = _env("wamv_vrx_v2_opt1", ["waves=D3"])
    for k in range(25):
        action = torch.tensor([[0.4 * math.sin(k), 0.6], [0.2, -0.5 * math.cos(k)]])
        a.step(action)
        b.step(action)
    assert all(torch.equal(x, y) for x, y in zip(a.observations, b.observations))
    assert a.journal_rollpitch_damping_stamp() == b.journal_rollpitch_damping_stamp()


def test_override_conflicts_with_an_explicit_profile_option():
    with pytest.raises(ValueError, match="cannot be combined"):
        _env("blueboat_v2", [], zeta=0.3, options={"rollpitch_damping": "registry"})
    # an explicit option alone is the pinned rule of that cfg (no override, pinned values)
    env = _env("blueboat_v2", [], options={"rollpitch_damping": "legacy_470"})
    stamp = env.journal_rollpitch_damping_stamp()
    assert stamp["pinned"]["rule"] == "legacy_470" and stamp["pinned_values"] is True


def test_stamp_refuses_inconsistent_profiles():
    env = _env("blueboat_v2", [], zeta=0.3)
    pinned = RP.profile_values(HP.profile_for_vehicle("blueboat_v2"))
    with pytest.raises(RP.RollPitchOverrideError):
        RP.stamp("blueboat_v2", pinned, pinned, {"rollpitch_zeta": 0.3})          # used is not the override
    other = RP.profile_values(HP.profile_for_vehicle("blueboat_v2", rollpitch_damping="zeta:0.3"))
    with pytest.raises(RP.RollPitchOverrideError):
        RP.stamp("blueboat_v2", pinned, other, None)                                # changed without an override
    env._hydro_v2.profile = HP.profile_for_vehicle("blueboat_v2")
    with pytest.raises(RuntimeError):
        env.journal_rollpitch_damping_stamp()


# ------------------------------------------------------- official refusals --
def _scripts():
    import eval_protocol
    import run_cell
    import cert_v3
    return eval_protocol, run_cell, cert_v3


def _preflight_args(**extra):
    base = dict(episodes=2, num_envs=2, allow_env=[], release_descriptor=None, seed_registry=None,
                speed_reference_m_s=None, dose_preset=None, extra_sets=[], task="Isaac-USV-StationKeep-WAMVVRX-"
                "Journal-Direct-v1", vessel="wamv", cell_config=None, suite=["waves=D4", "hydro.rollpitch_zeta=0.2"],
                suite_file=None, level=0, device="cpu")
    base.update(extra)
    return argparse.Namespace(**base)


def test_preflight_refuses_official_runs_before_the_simulator(monkeypatch):
    eval_protocol, _, _ = _scripts()
    for name in list(os.environ):
        if name.startswith(eval_protocol.SCIENTIFIC_PREFIXES) and name not in eval_protocol.RUNTIME_ENV:
            monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(eval_protocol.reference_environment, "require_official_match",
                        lambda registration, block: calls.append(registration) or block)
    for purpose, namespace, block in (("pilot", "dose_pilots", "pilot-01"),):
        official = _preflight_args(eval_seed=21001, seed_namespace=namespace, seed_block=block, run_purpose=purpose)
        with pytest.raises(eval_protocol.OverrideError, match="diagnostic only"):
            eval_protocol.preflight_protocol(official, ["eval.py"])
    assert calls == []                      # refused before the RE-1 check (D28)
    # the journal-default value itself is refused too (the override is diagnostic only)
    with pytest.raises(eval_protocol.OverrideError, match="rollpitch"):
        eval_protocol.preflight_protocol(_preflight_args(eval_seed=21001, seed_namespace="dose_pilots",
                                                         seed_block="pilot-01", run_purpose="pilot",
                                                         suite=["hydro.rollpitch_zeta=0.1"]), ["eval.py"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        path.write_text(json.dumps({"suite": ["waves=D2", "hydro.rollpitch_zeta=0.2"]}), encoding="ascii")
        with pytest.raises(eval_protocol.OverrideError):
            eval_protocol.preflight_protocol(_preflight_args(eval_seed=21001, seed_namespace="dose_pilots",
                                                             seed_block="pilot-01", run_purpose="pilot",
                                                             suite=[], suite_file=str(path)), ["eval.py"])
    diag = _preflight_args(eval_seed=42, seed_namespace=None, seed_block=None, run_purpose=None)
    eval_protocol.preflight_protocol(diag, ["eval.py"])
    assert diag.seed_registration["purpose"] == "diagnostic" and len(calls) == 1
    plain = _preflight_args(eval_seed=21001, seed_namespace="dose_pilots", seed_block="pilot-01",
                            run_purpose="pilot", suite=["waves=D4"])
    eval_protocol.preflight_protocol(plain, ["eval.py"])


def test_apply_journal_suites_installs_or_refuses():
    eval_protocol, _, _ = _scripts()
    cfg = G.journal_cfg("wamv_vrx_v2_opt1")
    cfg.seed = 42
    args = SimpleNamespace(suite=["waves=D4", "hydro.rollpitch_zeta=0.2"], suite_file=None,
                           seed_registration={"namespace": "diagnostic", "purpose": "diagnostic", "seed": 42})
    eval_protocol.apply_journal_suites(cfg, args)
    assert cfg.journal_rollpitch_zeta == 0.2
    assert args.journal_suites["rollpitch_damping_override"] == {"rollpitch_zeta": 0.2}
    assert "hydro.rollpitch_zeta=0.2" not in args.journal_suites["tokens"]
    assert "wind_coefficients" not in args.journal_suites
    cfg2 = G.journal_cfg("wamv_vrx_v2_opt1")
    cfg2.seed = 42
    plain = SimpleNamespace(suite=["waves=D4"], suite_file=None, seed_registration={"purpose": "pilot"})
    eval_protocol.apply_journal_suites(cfg2, plain)
    assert not hasattr(cfg2, "journal_rollpitch_zeta") and "rollpitch_damping_override" not in plain.journal_suites
    assert plain.journal_suites["stamp"] == args.journal_suites["stamp"]
    for registration in ({"purpose": "pilot"}, {"purpose": "final_test"}, None):
        cfg3 = G.journal_cfg("wamv_vrx_v2_opt1")
        cfg3.seed = 42
        with pytest.raises(eval_protocol.OverrideError, match="diagnostic only"):
            eval_protocol.apply_journal_suites(cfg3, SimpleNamespace(
                suite=["waves=D4", "hydro.rollpitch_zeta=0.1"], suite_file=None, seed_registration=registration))
        assert not hasattr(cfg3, "journal_rollpitch_zeta")


def test_run_cell_refuses_the_override(tmp_path):
    _, run_cell, _ = _scripts()
    argv = ["--task", "journal:station_keeping", "--vessel", "wamv", "--controller", "classical",
            "--run-purpose", "pilot", "--seed-namespace", "dose_pilots", "--eval-seed-block", "pilot-01",
            "--level", "0", "--episodes", "64", "--out", str(tmp_path / "c.json"), "--dry-run",
            "--suite", "waves=D4", "hydro.rollpitch_zeta=0.2"]
    with pytest.raises(run_cell.RunCellError, match="diagnostic only"):
        run_cell.resolve_cell(run_cell.parser_for_cell().parse_args(argv))


def test_cert_v3_refuses_an_official_certificate_with_an_override():
    _, _, cert_v3 = _scripts()
    pinned = RP.profile_values(HP.profile_for_vehicle("wamv_vrx_v2_opt1"))
    used = RP.profile_values(HP.profile_for_vehicle("wamv_vrx_v2_opt1", rollpitch_damping="zeta:0.1"))
    over = RP.stamp("wamv_vrx_v2_opt1", pinned, used, {"rollpitch_zeta": 0.1})
    clean = RP.stamp("wamv_vrx_v2_opt1", pinned, pinned, None)
    for registration in ({"purpose": "pilot"}, {"purpose": "final_test"}, {"purpose": "calibration"}, None):
        with pytest.raises(ValueError, match="diagnostic only"):
            cert_v3.rollpitch_damping_official_check(over, registration)
        cert_v3.rollpitch_damping_official_check(clean, registration)
    cert_v3.rollpitch_damping_official_check(over, {"namespace": "diagnostic", "purpose": "diagnostic"})
    cert_v3.rollpitch_damping_official_check(None, {"purpose": "pilot"})


# ----------------------------------------------------------------- validator --
def _document(env, purpose):
    header = env.journal_certificate_header()
    document = {"header": {"journal": {"task": "reach", "vehicle": env.cfg.vehicle},
                           "seed_namespace": {"namespace": "diagnostic" if purpose == "diagnostic" else "dose_pilots",
                                              "purpose": purpose},
                           "conditions": header["conditions"],
                           "rollpitch_damping": env.journal_rollpitch_damping_stamp(), "runtime": {}},
                "records": []}
    return json.loads(json.dumps(document))


@pytest.mark.parametrize("vessel", ["blueboat_v2", "wamv_vrx_v2_opt1"])
def test_validator_reports_an_override_as_non_official(vessel):
    import validate_cert_v3 as V
    over = _env(vessel, ["waves=D2"], zeta=0.05)
    clean = _env(vessel, ["waves=D2"])
    diag = _document(over, "diagnostic")
    assert V.rollpitch_damping_errors(diag) == []
    status = V.certificate_status(diag)
    assert status["official"] is False and any("roll/pitch damping" in r for r in status["reasons"])
    pilot = _document(over, "pilot")
    errors = V.rollpitch_damping_errors(pilot)
    assert len(errors) == 1 and "non-official" in errors[0]
    ok = _document(clean, "pilot")
    assert V.rollpitch_damping_errors(ok) == [] and V.certificate_status(ok) == {"official": True, "reasons": []}
    # tampering
    bad = copy.deepcopy(diag)
    bad["header"]["rollpitch_damping"]["used"]["roll_nms_rad"] *= 1.5
    assert any("registry rule" in e for e in V.rollpitch_damping_errors(bad))
    lie = copy.deepcopy(pilot)
    lie["header"]["rollpitch_damping"].update(pinned_values=True, override=None,
                                              used=dict(lie["header"]["rollpitch_damping"]["pinned"]))
    assert any("condition stamp rotational_damping" in e for e in V.rollpitch_damping_errors(lie))
    wrong_rule = copy.deepcopy(diag)
    wrong_rule["header"]["rollpitch_damping"]["used"]["rule"] = "zeta:0.3"
    assert any("used.rule" in e for e in V.rollpitch_damping_errors(wrong_rule))
    flag = copy.deepcopy(ok)
    flag["header"]["rollpitch_damping"]["pinned_values"] = False
    assert any("pinned_values" in e for e in V.rollpitch_damping_errors(flag))
    vessel_lie = copy.deepcopy(ok)
    vessel_lie["header"]["rollpitch_damping"]["vessel_profile"] = (
        "wamv_vrx_v2_opt1" if vessel == "blueboat_v2" else "blueboat_v2")
    assert V.rollpitch_damping_errors(vessel_lie)
    bad_override = copy.deepcopy(diag)
    bad_override["header"]["rollpitch_damping"]["override"] = {"rollpitch_zeta": 3.0}
    assert V.rollpitch_damping_errors(bad_override)
    # certificates written before T25 carry no stamp and stay valid
    old = copy.deepcopy(ok)
    del old["header"]["rollpitch_damping"]
    assert V.rollpitch_damping_errors(old) == []


def test_schema_accepts_the_stamp():
    from jsonschema import Draft202012Validator
    schema = json.loads((HERE.parents[0] / "cert_v3.schema.json").read_text(encoding="utf-8"))
    block_schema = dict(schema["properties"]["header"]["properties"]["rollpitch_damping"])
    block_schema["$defs"] = schema["$defs"]
    validator = Draft202012Validator(block_schema)
    for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
        for zeta in (None, 0.05, 0.3):
            stamp = _env(vessel, [], zeta=zeta).journal_rollpitch_damping_stamp()
            assert not list(validator.iter_errors(json.loads(json.dumps(stamp))))
    broken = _env("blueboat_v2", [], zeta=0.2).journal_rollpitch_damping_stamp()
    broken["override"]["extra"] = 1.0
    assert list(validator.iter_errors(json.loads(json.dumps(broken))))


def _evaluator_harness():
    spec = importlib.util.spec_from_file_location("t25_g6_exit", HERE.parents[0] / "test_g6_evaluator_exit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target, seed_flag", [("classical_baseline.py", "--seed"), ("eval_mppi.py", "--eval-seed")])
def test_real_evaluator_refuses_an_official_override_before_the_app(target, seed_flag):
    """The REAL evaluator script (G6b stub harness): an official (pilot) run with the token exits 2 in
    preflight with the T25 message, before AppLauncher; the same token in a DIAG42 run launches the stub app."""
    harness = _evaluator_harness()
    prefix = ["--protocol", "mission"] if target == "classical_baseline.py" else []
    task = ["--task", "Isaac-USV-StationKeep-BlueBoat-Journal-Direct-v1"]
    suite = ["--suite", "waves=D2", "hydro.rollpitch_zeta=0.3"]
    official = prefix + task + [seed_flag, "21001", "--seed-namespace", "dose_pilots", "--seed-block", "pilot-01",
                                "--run-purpose", "pilot"] + suite
    with tempfile.TemporaryDirectory(prefix="t25_off_") as workdir:
        r = harness.run_evaluator(harness.SCRIPTS, target, workdir, stub_mode="fail_cfg", argv_override=official)
    assert r["code"] == 2, (r["code"], r["output"][-2000:])
    assert r["marks"] == [] and r["files"] == []
    assert "diagnostic only" in r["output"] and "reference environment" not in r["output"], r["output"][-2000:]
    diag = prefix + task + [seed_flag, "42"] + suite
    with tempfile.TemporaryDirectory(prefix="t25_diag_") as workdir:
        r = harness.run_evaluator(harness.SCRIPTS, target, workdir, stub_mode="fail_cfg", argv_override=diag)
    assert "launch" in r["marks"], r["output"][-3000:]
    assert "diagnostic only" not in r["output"]
