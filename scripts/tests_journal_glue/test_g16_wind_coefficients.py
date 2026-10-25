"""G16 (T1.20): diagnostic-only wind-coefficient override (windcoef.c_x0 / c_y0 / c_n0) and the diagnostic
centroid-offset yaw term (windcoef.offset_sl, follow-up).

* token grammar and validation (conditions/wind_coefficients.py), resolution beside the suite stack
  (journal_config.resolve_from_args_with_wind_coefficients) with an unchanged JournalConditions;
* the REAL mixin + condition layer + C3 wind backend (glue_fixtures.FakeJournalEnv): the override reaches
  the wind wrench (yaw scales with C_N0, force unchanged), the header stamp says pinned vs used, the
  condition stamp carries the used values; with wind off the override is inert (stamps and every
  observation bitwise equal to the pinned run);
* official runs are refused: eval_protocol preflight (before the RE-1 check, i.e. before the simulator),
  eval_protocol.apply_journal_suites, run_cell, cert_v3 (wind_coefficients_official_check);
* validate_cert_v3: a certificate with overridden coefficients is non-official (error when its seed
  purpose is official), consistency of the stamp with the condition stamps and the registry.
CPU only.
"""
from __future__ import annotations

import argparse
import importlib.util
import copy
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

WC = G.shared("conditions.wind_coefficients")
JC = G.journal_config
PINNED = {"c_x0": 0.9, "c_y0": 0.9, "c_n0": 0.1}


# ------------------------------------------------------------------ grammar --
def test_split_tokens_and_validation():
    rest, override = WC.split_tokens(["wind=D4", "windcoef.c_n0=-0.2", "waves=D1", "windcoef.c_x0=0.5"])
    assert rest == ["wind=D4", "waves=D1"]
    assert override == {"c_n0": -0.2, "c_x0": 0.5} and list(override) == ["c_n0", "c_x0"]
    assert WC.split_tokens(["wind=D2"]) == (["wind=D2"], None)
    assert WC.split_tokens(["windcoef.c_n0=0.2", "windcoef.c_n0=0.2"])[1] == {"c_n0": 0.2}
    for bad in (["windcoef.c_n0=0.2", "windcoef.c_n0=0.3"], ["windcoef.c_z0=0.1"], ["windcoef=0.1"],
                ["windcoef.c_n0=NaN"], ["windcoef.c_n0=Infinity"], ["windcoef.c_x0=-0.1"], ["windcoef.c_n0=true"],
                ["windcoef.c_n0=\"0.1\""], ["windcoef.c_n0=abc"], ["windcoef.c_n0=9"], ["windcoef.c_n0.x=0.1"]):
        with pytest.raises(WC.WindCoefficientError):
            WC.split_tokens(bad)
    assert WC.is_override_token("windcoef.c_n0=0") and not WC.is_override_token("wind=D2")


def test_resolution_leaves_the_suite_stack_unchanged(tmp_path):
    plain = JC.resolve_from_args(SimpleNamespace(suite=["wind=D4", "waves=D1"], suite_file=None), seed=42)
    with_coef, override = JC.resolve_from_args_with_wind_coefficients(
        SimpleNamespace(suite=["wind=D4", "windcoef.c_n0=-0.2", "waves=D1"], suite_file=None), seed=42)
    assert override == {"c_n0": -0.2}
    assert with_coef == plain and with_coef.stamp() == plain.stamp() and with_coef.tokens() == plain.tokens()
    none, no_override = JC.resolve_from_args_with_wind_coefficients(
        SimpleNamespace(suite=["wind=D4", "waves=D1"], suite_file=None), seed=42)
    assert no_override is None and none == plain
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps({"suite": ["wind=D2", "windcoef.c_n0=0.11"]}), encoding="ascii")
    from_file, file_override = JC.resolve_from_args_with_wind_coefficients(
        SimpleNamespace(suite=[], suite_file=str(suite_file)), seed=42)
    assert file_override == {"c_n0": 0.11}
    assert from_file == JC.resolve_from_args(SimpleNamespace(suite=["wind=D2"], suite_file=None), seed=42)
    with pytest.raises(JC.SuiteConflict):
        JC.resolve_from_args_with_wind_coefficients(
            SimpleNamespace(suite=["windcoef.c_n0=0.1", "windcoef.c_n0=0.2"], suite_file=None), seed=42)
    # the plain resolver never accepts them (admission tools and fixtures stay fail-closed)
    with pytest.raises(JC.SuiteConflict):
        JC.resolve_from_args(SimpleNamespace(suite=["windcoef.c_n0=0.1"], suite_file=None), seed=42)
    cfg = SimpleNamespace()
    JC.apply_wind_coefficients(cfg, {"c_n0": -0.2})
    assert cfg.journal_wind_coefficients == {"c_n0": -0.2}
    with pytest.raises(ValueError):
        JC.apply_wind_coefficients(SimpleNamespace(), None)


# ------------------------------------------------------- real layer + wind --
def _env(vessel, tokens, override=None, num_envs=2):
    cfg = G.journal_cfg(vessel, tokens=tokens)
    if override is not None:
        JC.apply_wind_coefficients(cfg, override)
    env = G.FakeJournalEnv(cfg, num_envs)
    env.reset()
    return env


@pytest.mark.parametrize("vessel", ["blueboat_v2", "wamv_vrx_v2_opt1"])
def test_override_reaches_the_wind_wrench(vessel):
    pinned = _env(vessel, ["wind=D4"])
    over = _env(vessel, ["wind=D4"], {"c_n0": -0.2})
    spec = G.vehicles_v2.get_vehicle_v2(vessel)
    assert WC.pinned_values(spec.windage) == PINNED
    assert not hasattr(pinned.cfg, "journal_wind_coefficients")
    stamp_p = pinned._conditions.wind_coefficient_stamp()
    stamp_o = over._conditions.wind_coefficient_stamp()
    assert stamp_p["pinned_values"] is True and stamp_p["override"] is None and stamp_p["used"] == PINNED
    assert stamp_o["pinned_values"] is False and stamp_o["override"] == {"c_n0": -0.2}
    assert stamp_o["used"] == {"c_x0": 0.9, "c_y0": 0.9, "c_n0": -0.2} and stamp_o["wind_enabled"] is True
    assert stamp_o["vessel_profile"] == vessel and stamp_o["schema"] == WC.STAMP_SCHEMA
    assert over._conditions.wind.coefficient_model.c_n0 == -0.2
    assert pinned._conditions.wind.coefficient_model.c_n0 == 0.1
    wind_p = pinned.journal_certificate_header()["conditions"]["context"]["resolved"]["wind"]
    wind_o = over.journal_certificate_header()["conditions"]["context"]["resolved"]["wind"]
    assert "coefficient_override" not in wind_p and wind_o["coefficient_override"] == {"c_n0": -0.2}
    assert wind_p["windage"]["c_n0"] == 0.1 and wind_o["windage"]["c_n0"] == -0.2
    assert {k: v for k, v in wind_p.items() if k != "windage"} == \
        {k: v for k, v in wind_o.items() if k not in ("windage", "coefficient_override")}
    # same air state, same hull state: force identical, yaw scaled by -0.2 / 0.1
    dtype = pinned._conditions.wind.dtype
    vel = torch.tensor([[0.3, -0.1], [0.0, 0.2]], dtype=dtype)
    for heading in (0.4, 1.1, 2.0, -0.7):
        h = torch.full((2,), heading, dtype=dtype)
        fp, tp = pinned._conditions.wind.forces(vel, h)
        fo, to = over._conditions.wind.forces(vel, h)
        assert torch.equal(fp, fo) and torch.equal(tp[:, :2], to[:, :2])
        assert torch.allclose(to[:, 2], -2.0 * tp[:, 2], rtol=1e-5, atol=1e-7)
        assert bool((tp[:, 2].abs() > 0).all())
    # the override changes the trajectory (a wind-on run is not inert)
    for env in (pinned, over):
        for _ in range(10):
            env.step(torch.tensor([[0.4, 0.4], [0.2, -0.2]]))
    assert not torch.equal(pinned.robot.data.root_link_pose_w, over.robot.data.root_link_pose_w)


def test_override_is_inert_without_wind():
    for tokens in ([], ["waves=D1", "current=D2"]):
        pinned = _env("wamv_vrx_v2_opt1", tokens)
        over = _env("wamv_vrx_v2_opt1", tokens, {"c_n0": -0.2, "c_x0": 0.5})
        assert over._conditions.wind is None
        hp, ho = pinned.journal_certificate_header(), over.journal_certificate_header()
        assert hp == ho and "wind_coefficients" not in ho          # the mixin header blocks are unchanged
        so = over._conditions.wind_coefficient_stamp()
        assert so["wind_enabled"] is False and so["effect"].startswith("none")
        actions = [torch.tensor([[0.5 * math.sin(k), 0.3], [-0.2, 0.6 * math.cos(k)]]) for k in range(40)]
        for a in actions:
            pinned.step(a)
            over.step(a)
        assert len(pinned.observations) == len(over.observations)
        for a, b in zip(pinned.observations, over.observations):
            assert torch.equal(a, b)
        assert torch.equal(pinned.robot.data.root_link_pose_w, over.robot.data.root_link_pose_w)


def test_stamp_cross_checks_the_backend():
    env = _env("blueboat_v2", ["wind=D2"], {"c_n0": 0.11})
    assert env._conditions.wind_coefficient_stamp()["used"]["c_n0"] == 0.11
    env._conditions.wind.coefficient_model = G.shared("wind").FossenCoefficients(0.9, 0.9, 0.3)
    with pytest.raises(RuntimeError):
        env._conditions.wind_coefficient_stamp()


# ------------------------------------------------------- official refusals --
def _scripts():
    import eval_protocol
    import run_cell
    import cert_v3
    return eval_protocol, run_cell, cert_v3


def _preflight_args(**extra):
    base = dict(episodes=2, num_envs=2, allow_env=[], release_descriptor=None, seed_registry=None,
                speed_reference_m_s=None, dose_preset=None, extra_sets=[], task="Isaac-USV-StationKeep-WAMVVRX-"
                "Journal-Direct-v1", vessel="wamv", cell_config=None, suite=["wind=D4", "windcoef.c_n0=-0.2"],
                suite_file=None, level=0, device="cpu")
    base.update(extra)
    return argparse.Namespace(**base)


def test_preflight_refuses_official_runs_before_the_simulator(monkeypatch):
    eval_protocol, _, _ = _scripts()
    for name in list(os.environ):
        if name.startswith(eval_protocol.SCIENTIFIC_PREFIXES) and name not in eval_protocol.RUNTIME_ENV:
            monkeypatch.delenv(name, raising=False)
    official = _preflight_args(eval_seed=21001, seed_namespace="dose_pilots", seed_block="pilot-01",
                               run_purpose="pilot")
    with pytest.raises(eval_protocol.OverrideError, match="diagnostic only"):
        eval_protocol.preflight_protocol(official, ["eval.py"])
    # the refusal comes before the RE-1 check (D28): an RE-1 mismatch alone would raise a different error
    calls = []
    monkeypatch.setattr(eval_protocol.reference_environment, "require_official_match",
                        lambda registration, block: calls.append(registration) or block)
    with pytest.raises(eval_protocol.OverrideError, match="diagnostic only"):
        eval_protocol.preflight_protocol(_preflight_args(eval_seed=21001, seed_namespace="dose_pilots",
                                                         seed_block="pilot-01", run_purpose="pilot"), ["eval.py"])
    assert calls == []
    # the same tokens at the pinned value are refused too (the override itself is diagnostic only)
    with pytest.raises(eval_protocol.OverrideError):
        eval_protocol.preflight_protocol(_preflight_args(eval_seed=21001, seed_namespace="dose_pilots",
                                                         seed_block="pilot-01", run_purpose="pilot",
                                                         suite=["wind=D4", "windcoef.c_n0=0.1"]), ["eval.py"])
    # a suite file naming windcoef is refused in an official run
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        path.write_text(json.dumps({"suite": ["wind=D2", "windcoef.c_n0=0.2"]}), encoding="ascii")
        with pytest.raises(eval_protocol.OverrideError):
            eval_protocol.preflight_protocol(_preflight_args(eval_seed=21001, seed_namespace="dose_pilots",
                                                             seed_block="pilot-01", run_purpose="pilot",
                                                             suite=[], suite_file=str(path)), ["eval.py"])
    # DIAG42: diagnostic registration, accepted
    diag = _preflight_args(eval_seed=42, seed_namespace=None, seed_block=None, run_purpose=None)
    eval_protocol.preflight_protocol(diag, ["eval.py"])
    assert diag.seed_registration["purpose"] == "diagnostic" and len(calls) == 1
    # an official run without override tokens is unaffected by G16
    plain = _preflight_args(eval_seed=21001, seed_namespace="dose_pilots", seed_block="pilot-01",
                            run_purpose="pilot", suite=["wind=D4"])
    eval_protocol.preflight_protocol(plain, ["eval.py"])


def test_apply_journal_suites_installs_or_refuses():
    eval_protocol, _, _ = _scripts()
    cfg = G.journal_cfg("wamv_vrx_v2_opt1")
    cfg.seed = 42
    args = SimpleNamespace(suite=["wind=D4", "windcoef.c_n0=-0.2"], suite_file=None,
                           seed_registration={"namespace": "diagnostic", "purpose": "diagnostic", "seed": 42})
    eval_protocol.apply_journal_suites(cfg, args)
    assert cfg.journal_wind_coefficients == {"c_n0": -0.2}
    assert args.journal_suites["wind_coefficients"] == {"c_n0": -0.2}
    assert "windcoef.c_n0=-0.2" not in args.journal_suites["tokens"]
    cfg2 = G.journal_cfg("wamv_vrx_v2_opt1")
    cfg2.seed = 42
    plain = SimpleNamespace(suite=["wind=D4"], suite_file=None, seed_registration={"purpose": "pilot"})
    eval_protocol.apply_journal_suites(cfg2, plain)
    assert not hasattr(cfg2, "journal_wind_coefficients") and "wind_coefficients" not in plain.journal_suites
    assert plain.journal_suites["stamp"] == args.journal_suites["stamp"]
    for registration in ({"purpose": "pilot"}, {"purpose": "final_test"}, None):
        cfg3 = G.journal_cfg("wamv_vrx_v2_opt1")
        cfg3.seed = 42
        with pytest.raises(eval_protocol.OverrideError, match="diagnostic only"):
            eval_protocol.apply_journal_suites(cfg3, SimpleNamespace(
                suite=["wind=D4", "windcoef.c_n0=0.2"], suite_file=None, seed_registration=registration))
        assert not hasattr(cfg3, "journal_wind_coefficients")


def test_run_cell_refuses_the_override(tmp_path):
    _, run_cell, _ = _scripts()
    argv = ["--task", "journal:station_keeping", "--vessel", "wamv", "--controller", "classical",
            "--run-purpose", "pilot", "--seed-namespace", "dose_pilots", "--eval-seed-block", "pilot-01",
            "--level", "0", "--episodes", "64", "--out", str(tmp_path / "c.json"), "--dry-run",
            "--suite", "wind=D4", "windcoef.c_n0=-0.2"]
    with pytest.raises(run_cell.RunCellError, match="diagnostic only"):
        run_cell.resolve_cell(run_cell.parser_for_cell().parse_args(argv))


def test_cert_v3_refuses_an_official_certificate_with_overridden_coefficients():
    _, _, cert_v3 = _scripts()
    over = WC.stamp("wamv_vrx_v2_opt1", PINNED, {"c_n0": -0.2}, True)
    pinned = WC.stamp("wamv_vrx_v2_opt1", PINNED, None, True)
    same_value = WC.stamp("wamv_vrx_v2_opt1", PINNED, {"c_n0": 0.1}, True)
    diag = {"namespace": "diagnostic", "purpose": "diagnostic"}
    for registration in ({"purpose": "pilot"}, {"purpose": "final_test"}, {"purpose": "calibration"}, None):
        with pytest.raises(ValueError, match="diagnostic only"):
            cert_v3.wind_coefficients_official_check(over, registration)
        with pytest.raises(ValueError):
            cert_v3.wind_coefficients_official_check(same_value, registration)
        cert_v3.wind_coefficients_official_check(pinned, registration)
    cert_v3.wind_coefficients_official_check(over, diag)
    cert_v3.wind_coefficients_official_check(None, {"purpose": "pilot"})


# ----------------------------------------------------------------- validator --
def _document(env, purpose):
    header = env.journal_certificate_header()
    document = {"header": {"journal": {"task": "reach", "vehicle": env.cfg.vehicle},
                           "seed_namespace": {"namespace": "diagnostic" if purpose == "diagnostic" else "dose_pilots",
                                              "purpose": purpose},
                           "conditions": header["conditions"], "wind_coefficients": env._conditions.wind_coefficient_stamp(),
                           "runtime": {}},
                "records": []}
    return json.loads(json.dumps(document))


def test_validator_reports_overridden_coefficients_as_non_official():
    import validate_cert_v3 as V
    over = _env("wamv_vrx_v2_opt1", ["wind=D4"], {"c_n0": -0.2})
    pinned = _env("wamv_vrx_v2_opt1", ["wind=D4"])
    diag = _document(over, "diagnostic")
    assert V.wind_coefficient_errors(diag) == []
    assert V.certificate_status(diag)["official"] is False
    assert any("wind coefficients" in r for r in V.certificate_status(diag)["reasons"])
    pilot = _document(over, "pilot")
    errors = V.wind_coefficient_errors(pilot)
    assert len(errors) == 1 and "non-official" in errors[0]
    clean = _document(pinned, "pilot")
    assert V.wind_coefficient_errors(clean) == [] and V.certificate_status(clean) == {"official": True, "reasons": []}
    # tampering is caught
    bad = copy.deepcopy(diag)
    bad["header"]["wind_coefficients"]["used"]["c_n0"] = 0.3
    assert any("used" in e for e in V.wind_coefficient_errors(bad))
    lie = copy.deepcopy(pilot)
    lie["header"]["wind_coefficients"].update(pinned_values=True, override=None,
                                              used=dict(PINNED))
    assert any("differ from header.wind_coefficients.used" in e for e in V.wind_coefficient_errors(lie))
    wrong_pin = copy.deepcopy(clean)
    wrong_pin["header"]["wind_coefficients"]["pinned"]["c_n0"] = 0.2
    wrong_pin["header"]["wind_coefficients"]["used"]["c_n0"] = 0.2
    assert any("registry" in e for e in V.wind_coefficient_errors(wrong_pin))
    # a certificate without the stamp (pre-G16) is checked against the registry when it is official
    old = copy.deepcopy(pilot)
    del old["header"]["wind_coefficients"]
    assert any("instead of the pinned" in e for e in V.wind_coefficient_errors(old))
    old_clean = copy.deepcopy(clean)
    del old_clean["header"]["wind_coefficients"]
    assert V.wind_coefficient_errors(old_clean) == []
    # records carry the same coefficients
    rec = copy.deepcopy(diag)
    record_stamp = copy.deepcopy(rec["header"]["conditions"])
    record_stamp["context"]["resolved"]["wind"]["windage"]["c_n0"] = 0.1
    rec["records"] = [{"conditions": record_stamp}]
    assert any("records/0" in e for e in V.wind_coefficient_errors(rec))
    # wind off: the override is inert, still non-official in an official purpose
    off = _document(_env("wamv_vrx_v2_opt1", [], {"c_n0": 0.2}), "pilot")
    assert off["header"]["wind_coefficients"]["wind_enabled"] is False
    assert any("non-official" in e for e in V.wind_coefficient_errors(off))
    assert V.wind_coefficient_errors(_document(_env("wamv_vrx_v2_opt1", [], {"c_n0": 0.2}), "diagnostic")) == []


def test_schema_accepts_the_stamp():
    from jsonschema import Draft202012Validator
    schema = json.loads((HERE.parents[0] / "cert_v3.schema.json").read_text(encoding="utf-8"))
    block_schema = dict(schema["properties"]["header"]["properties"]["wind_coefficients"])
    block_schema["$defs"] = schema["$defs"]
    validator = Draft202012Validator(block_schema)
    for override in (None, {"c_n0": -0.2}, {"c_x0": 0.5, "c_y0": 0.7, "c_n0": 0.0}):
        assert not list(validator.iter_errors(json.loads(json.dumps(WC.stamp("blueboat_v2", PINNED, override, True)))))
    broken = WC.stamp("blueboat_v2", PINNED, {"c_n0": 0.2}, True)
    broken["override"]["c_z0"] = 1.0
    assert list(validator.iter_errors(json.loads(json.dumps(broken))))


def test_heading_statistics_only_in_override_runs():
    pinned = _env("wamv_vrx_v2_opt1", ["wind=D4"])
    over = _env("wamv_vrx_v2_opt1", ["wind=D4"], {"c_n0": 0.2})
    assert pinned._conditions._wind_heading_stats is None
    for env in (pinned, over):
        for _ in range(15):
            env.step(torch.tensor([[0.3, 0.1], [0.2, -0.3]]))
    live = over._conditions._wind_heading_record(0)
    assert live["ticks"] == int(over._hydro_v2.ticks[0]) == 30
    for env in (pinned, over):          # end episode 0 (airborne, code 7) so the mixin latches its stamp
        env.robot.data.root_link_pose_w[0, 2] += 50.0
        env.step(torch.tensor([[0.3, 0.1], [0.2, -0.3]]))
    sp = pinned.episode_conditions[0]["runtime"]["wind"]
    so = over.episode_conditions[0]["runtime"]["wind"]
    assert "heading_relative_wind" not in sp
    stats = so["heading_relative_wind"]
    assert stats["ticks"] == so["physics_steps"] == 32
    assert 0.0 <= stats["mean_abs_gamma_rad"] <= math.pi and -1.0 <= stats["mean_cos_2gamma"] <= 1.0
    assert abs(stats["beam_tendency_rad"]) <= 0.25 * math.pi + 1e-12
    assert set(so) - set(sp) == {"heading_relative_wind"}
    # a new episode restarts the statistics
    assert over._conditions._wind_heading_record(0)["ticks"] == 0
    # the statistic itself: wind flowing to +x (from -x); bow pointing -x (into the wind) -> gamma = 0;
    # bow +y (CCW 90 deg from the wind-from direction... psi = pi/2, psi_from = pi) -> gamma = -pi/2
    layer = over._conditions
    layer._wind_heading_stats.zero_()
    layer.wind.direction_rad[:] = 0.0
    layer.wind.time_s[:] = 0.0
    dtype = layer._wind_heading_stats.dtype
    layer._accumulate_wind_heading(torch.tensor([math.pi, 0.5 * math.pi], dtype=dtype))
    rec0, rec1 = layer._wind_heading_record(0), layer._wind_heading_record(1)
    assert abs(rec0["mean_gamma_rad"]) < 1e-12 and abs(rec0["mean_cos_2gamma"] - 1.0) < 1e-12
    assert abs(rec0["beam_tendency_rad"] + 0.25 * math.pi) < 1e-12
    assert abs(rec1["mean_gamma_rad"] + 0.5 * math.pi) < 1e-12 and abs(rec1["mean_cos_2gamma"] + 1.0) < 1e-12
    assert abs(rec1["beam_tendency_rad"] - 0.25 * math.pi) < 1e-12
    # a non-finite heading (code-9 episode) is counted apart and never poisons the means or the JSON stamp
    layer._accumulate_wind_heading(torch.tensor([float("nan"), 0.5 * math.pi], dtype=dtype))
    rec0, rec1 = layer._wind_heading_record(0), layer._wind_heading_record(1)
    assert rec0["ticks"] == 1 and rec0["nonfinite_ticks"] == 1 and abs(rec0["mean_cos_2gamma"] - 1.0) < 1e-12
    assert rec1["ticks"] == 2 and rec1["nonfinite_ticks"] == 0
    json.dumps(rec0, allow_nan=False)


def _evaluator_harness():
    spec = importlib.util.spec_from_file_location("g16_g6_exit", HERE.parents[0] / "test_g6_evaluator_exit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target, seed_flag", [("classical_baseline.py", "--seed"), ("eval_mppi.py", "--eval-seed")])
def test_real_evaluator_refuses_an_official_override_before_the_app(target, seed_flag):
    """The REAL evaluator script (G6b stub harness, scripts/test_g6_evaluator_exit.py run_evaluator): an official
    (pilot) run with windcoef tokens exits 2 in preflight with the G16 message, before AppLauncher (no 'launch'
    mark) and before the RE-1 check; the same tokens in a DIAG42 run pass preflight and launch the (stub) app."""
    harness = _evaluator_harness()
    prefix = ["--protocol", "mission"] if target == "classical_baseline.py" else []
    task = ["--task", "Isaac-USV-StationKeep-BlueBoat-Journal-Direct-v1"]          # the harness adds --vessel blueboat
    suite = ["--suite", "wind=D4", "windcoef.c_n0=-0.2"]
    official = prefix + task + [seed_flag, "21001", "--seed-namespace", "dose_pilots", "--seed-block", "pilot-01",
                                "--run-purpose", "pilot"] + suite
    with tempfile.TemporaryDirectory(prefix="g16_off_") as workdir:
        r = harness.run_evaluator(harness.SCRIPTS, target, workdir, stub_mode="fail_cfg", argv_override=official)
    assert r["code"] == 2, (r["code"], r["output"][-2000:])
    assert r["marks"] == [] and r["files"] == []
    assert "diagnostic only" in r["output"] and "reference environment" not in r["output"], r["output"][-2000:]
    diag = prefix + task + [seed_flag, "42"] + suite
    with tempfile.TemporaryDirectory(prefix="g16_diag_") as workdir:
        r = harness.run_evaluator(harness.SCRIPTS, target, workdir, stub_mode="fail_cfg", argv_override=diag)
    assert "launch" in r["marks"], r["output"][-3000:]
    assert "diagnostic only" not in r["output"]


# ------------------------------------------------- centroid-offset yaw term (G16 follow-up) --
def test_offset_token_grammar_and_stamp():
    rest, override = WC.split_tokens(["wind=D4", "windcoef.c_n0=0.11", "windcoef.offset_sl=0.039"])
    assert rest == ["wind=D4"] and override == {"c_n0": 0.11, "offset_sl": 0.039}
    for bad in (["windcoef.offset_sl=0.6"], ["windcoef.offset_sl=NaN"], ["windcoef.offset_sl=true"],
                ["windcoef.offset_sl=0.03", "windcoef.offset_sl=0.04"]):
        with pytest.raises(WC.WindCoefficientError):
            WC.split_tokens(bad)
    only = WC.stamp("wamv_vrx_v2_opt1", PINNED, {"offset_sl": 0.039}, True)
    assert only["used"] == PINNED and only["pinned_values"] is False
    assert only["centroid_offset"] == {"enabled": True, "offset_sl": 0.039, "form": WC.OFFSET_FORM}
    off = WC.stamp("wamv_vrx_v2_opt1", PINNED, None, True)
    assert off["centroid_offset"]["enabled"] is False and off["centroid_offset"]["offset_sl"] is None
    assert off["pinned_values"] is True


def test_offset_model_and_equilibrium():
    gamma = torch.linspace(-math.pi, math.pi, 721, dtype=torch.float64)
    base = G.shared("wind").FossenCoefficients(0.9, 0.9, 0.11)(gamma)
    with_offset = WC.OffsetFossenCoefficients(0.9, 0.9, 0.11, 0.039)(gamma)
    assert torch.equal(base[0], with_offset[0]) and torch.equal(base[1], with_offset[1])
    assert torch.allclose(with_offset[2] - base[2], 0.039 * base[1], rtol=0, atol=1e-15)
    # stable heading (C_N crosses zero from + to - for gamma in (0, pi)): cos g* = -(s/L) C_Y0 / (2 C_N0)
    cn = with_offset[2][360:].tolist()
    g = gamma[360:].tolist()
    cross = next(g[i] for i in range(1, len(cn)) if cn[i - 1] > 0 >= cn[i])
    assert abs(math.degrees(cross) - math.degrees(math.acos(-0.039 * 0.9 / 0.22))) < 0.5
    assert 98.5 < math.degrees(cross) < 100.0                  # 99.2 deg vs 90 deg without the term


def test_offset_reaches_the_wrench_and_is_stamped():
    off = _env("wamv_vrx_v2_opt1", ["wind=D4"], {"c_n0": 0.11})
    on = _env("wamv_vrx_v2_opt1", ["wind=D4"], {"c_n0": 0.11, "offset_sl": 0.039})
    assert type(off._conditions.wind.coefficient_model).__name__ == "FossenCoefficients"
    assert on._conditions.wind.coefficient_model.offset_sl == 0.039
    stamp = on._conditions.wind_coefficient_stamp()
    assert stamp["centroid_offset"]["enabled"] is True and stamp["pinned_values"] is False
    assert stamp["used"] == {"c_x0": 0.9, "c_y0": 0.9, "c_n0": 0.11}
    wind = on.journal_certificate_header()["conditions"]["context"]["resolved"]["wind"]
    assert wind["coefficient_override"] == {"c_n0": 0.11, "offset_sl": 0.039}
    length = G.vehicles_v2.get_vehicle_v2("wamv_vrx_v2_opt1").windage.length_m
    dtype = off._conditions.wind.dtype
    vel = torch.tensor([[0.3, -0.1], [0.0, 0.2]], dtype=dtype)
    for heading in (0.4, 1.6, 2.5, -1.2):
        h = torch.full((2,), heading, dtype=dtype)
        f_off, t_off = off._conditions.wind.forces(vel, h)
        f_on, t_on = on._conditions.wind.forces(vel, h)
        assert torch.equal(f_off, f_on) and torch.equal(t_off[:, :2], t_on[:, :2])
        c, s = math.cos(heading), math.sin(heading)
        sway = -s * f_on[:, 0] + c * f_on[:, 1]                 # body lateral force (+ port)
        assert torch.allclose(t_on[:, 2] - t_off[:, 2], 0.039 * length * sway, rtol=1e-4, atol=1e-4)
    import validate_cert_v3 as V
    doc = _document(on, "pilot")
    assert any("non-official" in e for e in V.wind_coefficient_errors(doc))
    assert V.wind_coefficient_errors(_document(on, "diagnostic")) == []
    tampered = copy.deepcopy(_document(on, "diagnostic"))
    tampered["header"]["wind_coefficients"]["centroid_offset"]["offset_sl"] = 0.035
    assert any("centroid_offset" in e for e in V.wind_coefficient_errors(tampered))


def test_offset_is_inert_without_wind_and_refused_in_official_runs(monkeypatch):
    pinned = _env("blueboat_v2", ["current=D1"])
    over = _env("blueboat_v2", ["current=D1"], {"offset_sl": 0.035})
    for _ in range(20):
        a = torch.tensor([[0.4, 0.2], [0.1, -0.3]])
        pinned.step(a)
        over.step(a)
    assert all(torch.equal(x, y) for x, y in zip(pinned.observations, over.observations))
    assert over._conditions.wind_coefficient_stamp()["centroid_offset"]["enabled"] is True
    eval_protocol, run_cell, _ = _scripts()
    for name in list(os.environ):
        if name.startswith(eval_protocol.SCIENTIFIC_PREFIXES) and name not in eval_protocol.RUNTIME_ENV:
            monkeypatch.delenv(name, raising=False)
    with pytest.raises(eval_protocol.OverrideError, match="diagnostic only"):
        eval_protocol.preflight_protocol(_preflight_args(eval_seed=21001, seed_namespace="dose_pilots",
                                                         seed_block="pilot-01", run_purpose="pilot",
                                                         suite=["wind=D4", "windcoef.offset_sl=0.039"]), ["eval.py"])
    cfg = G.journal_cfg("wamv_vrx_v2_opt1")
    cfg.seed = 42
    with pytest.raises(eval_protocol.OverrideError, match="diagnostic only"):
        eval_protocol.apply_journal_suites(cfg, SimpleNamespace(suite=["wind=D4", "windcoef.offset_sl=0.039"],
                                                                suite_file=None, seed_registration={"purpose": "pilot"}))
    _, _, cert_v3 = _scripts()
    with pytest.raises(ValueError, match="diagnostic only"):
        cert_v3.wind_coefficients_official_check(WC.stamp("x", PINNED, {"offset_sl": 0.039}, True), {"purpose": "pilot"})
