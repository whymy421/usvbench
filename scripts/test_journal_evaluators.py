"""CPU tests of the journal evaluator seams (F7 items 3, 5, 6). No Isaac.

* item 5: vessel-specific journal id selection (eval_protocol), run_cell
  mapping of the "journal" registration directory to the TASK_TABLE carrier
  (classical harbour dispatch; legacy harbour still refused), --suite
  pass-through, D12 override protection + journal_validate;
* item 6: journal certificates written by the REAL cert_v3 writer from the
  REAL mixin (F0 FakeJournalEnv: schema-3 stamps, D7 envelopes, D11 hashes)
  validate; tampered D7 envelopes / primitive hashes / suite stamps are
  rejected; analysis_input pairs two certificates differing in ONE suite
  level and rejects any other primitive difference.
* round 4b (F_ROUND_4B.md F7): D19 runtime pool-path variables/cfg fields are
  allow-listed, stamped and never refused; the pinned pool manifest digest is
  stamped in the D7 header; D16 not_admitted records validate, count in the
  header per dose, pair for every other suite and are unexposed for the
  dynamic treatment; F5/F6 contract metrics are exported; F5's docking
  level-table proposal (journal_plan) is honoured by protocol, writer and
  validator.
* round T (CONTEXT_R4 D20, D23, D24): not-admitted rows carry a REAL family
  stamp (scripts/fixtures_d16, scripts/make_d16_fixtures.py; F7's empty-assignment
  form is withdrawn) and their d7_assignment primitive is F0's
  pairing.d7_primitive_hash; the fake env publishes the D24 offered grid
  (dynamic_hazard_header["grid"]) that the writer now requires.

Run: python -B scripts/test_journal_evaluators.py   (or pytest)
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import analysis_input  # noqa: E402
import cert_v3  # noqa: E402
import cfg_override  # noqa: E402
import condition_args  # noqa: E402
import eval_protocol  # noqa: E402
import run_cell  # noqa: E402
import validate_cert_v3  # noqa: E402

TASK_ID = "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1"


def _raises(exc_type, text, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exc_type as exc:
        assert text in str(exc), str(exc)
        return exc
    raise AssertionError(f"expected {exc_type.__name__} containing {text!r}")


# ----------------------------------------------------------- id selection --
def test_vessel_specific_journal_id_is_selected_before_the_cfg():
    args = SimpleNamespace(task="journal:docking", vessel="wamv")
    info = eval_protocol.select_task_id(args)
    assert args.task == "Isaac-USV-Dock-WAMVVRX-Journal-Direct-v1"
    assert (info["carrier"], info["vehicle"], info["journal_task"]) == ("docking", "wamv_vrx_v2_opt1", "docking")
    _raises(cfg_override.OverrideError, "--vessel wamv selects", eval_protocol.select_task_id,
            SimpleNamespace(task="Isaac-USV-Dock-BlueBoat-Journal-Direct-v1", vessel="wamv"))
    _raises(cfg_override.OverrideError, "needs --vessel", eval_protocol.select_task_id,
            SimpleNamespace(task="journal:reach", vessel=None))
    _raises(cfg_override.OverrideError, "unknown journal task", eval_protocol.select_task_id,
            SimpleNamespace(task="journal:nope", vessel="blueboat"))
    legacy = SimpleNamespace(task="Isaac-USV-HazardCross-Direct-v1", vessel="blueboat")
    assert eval_protocol.select_task_id(legacy) is None and legacy.task == "Isaac-USV-HazardCross-Direct-v1"
    plans = eval_protocol.journal_plans()
    for (task, vessel), gym_id in plans.JOURNAL_ENV_IDS.items():
        resolved = eval_protocol.resolve_journal_task(gym_id)
        assert resolved["carrier"] == plans.TASK_TABLE[task]["carrier"] and resolved["vehicle"] == vessel


def _cell(*extra, task="journal:reach", vessel="blueboat", controller="classical"):
    values = ["--task", task, "--vessel", vessel, "--controller", controller, "--eval-seed-block", "pilot-01",
              "--seed-namespace", "dose_pilots", "--run-purpose", "pilot", "--level", "0", "--episodes", "128",
              "--out", os.path.join(_SCRIPTS, "..", "test_scratch_f7_cell.json"), "--dry-run"]
    return run_cell.parser_for_cell().parse_args(values + list(extra))


def test_run_cell_maps_the_journal_directory_to_the_carrier_family():
    tasks = run_cell.registered_tasks()
    plans = eval_protocol.journal_plans()
    assert all(tasks[gym_id] == "journal" for gym_id in plans.JOURNAL_ENV_IDS.values())
    expected = {"reach": "classical_planner_pid.py", "scatter": "classical_planner_pid.py",
                "path_hazard": "classical_planner_pid.py", "path_following": "classical_baseline.py",
                "station_keeping": "classical_baseline.py", "docking": "classical_baseline.py",
                "harbour_mission": "classical_harbour.py"}
    with patch.dict("os.environ", {}, clear=True):
        for task, script in expected.items():
            for vessel in ("blueboat", "wamv"):
                plan = run_cell.resolve_cell(_cell(task="journal:" + task, vessel=vessel))
                command = plan["command"]
                assert os.path.basename(command[1]) == script, (task, command[1])
                gym_id = command[command.index("--task") + 1]
                assert gym_id == plans.JOURNAL_ENV_IDS[(task, plans.BLUEBOAT_PROFILE if vessel == "blueboat"
                                                         else plans.DEFAULT_WAMV_PROFILE)]
                journal = plan["configuration"]["journal"]
                assert journal["carrier_family"] == plans.TASK_TABLE[task]["carrier"]
        mppi = run_cell.resolve_cell(_cell(controller="mppi", task=TASK_ID))
        assert os.path.basename(mppi["command"][1]) == "eval_mppi.py"
        assert "36 rays" in mppi["configuration"]["controller_information"]
        suites = run_cell.resolve_cell(_cell("--suite", "waves=D2", "wind=D1", controller="mppi"))
        command = suites["command"]
        assert command[command.index("--suite"):command.index("--suite") + 3] == ["--suite", "waves=D2", "wind=D1"]
        assert suites["configuration"]["journal"]["suite"] == ["waves=D2", "wind=D1"]
        _raises(run_cell.RunCellError, "--vessel wamv selects", run_cell.resolve_cell, _cell(task=TASK_ID, vessel="wamv"))
        _raises(ValueError, "D12", run_cell.resolve_cell, _cell("--set", "vehicle=wamv_vrx_v2_opt1", controller="mppi"))
        _raises(ValueError, "D12", run_cell.resolve_cell, _cell("--set", "episode_length_s=9", controller="mppi"))
        _raises(run_cell.RunCellError, "journal ids only", run_cell.resolve_cell,
                _cell("--suite", "waves=D2", task="Isaac-USV-HazardCross-Direct-v1", controller="mppi"))
        error = _raises(run_cell.RunCellError, "classical harbour-mission controller", run_cell.resolve_cell,
                        _cell(task="Isaac-USV-HarborMission-Direct-v1"))
        assert "HarborMissionEpisode" in str(error) and "LEGACY" in str(error)


# ----------------------------------------------------------------- D12 --
_STUB_MODULE = types.ModuleType("_f7_stub_journal_cfg")
_STUB_MODULE.calls = []


def _stub_journal_validate(cfg):
    _STUB_MODULE.calls.append(cfg)
    if cfg.max_obstacles != 15:
        raise ValueError("capacity changed")


_STUB_MODULE.journal_validate = _stub_journal_validate
sys.modules[_STUB_MODULE.__name__] = _STUB_MODULE


class JournalCfgStub:
    def __init__(self):
        self.journal_contract_version = 2
        self.journal_task = "reach"
        self.vehicle = "blueboat_v2"
        self.seed = 42
        self.goal_radius = 2.0
        self.episode_length_s = 88.0
        self.max_obstacles = 15
        self.layout_mode = "journal"
        self.appear_count = 0
        self.obs_sea_state = False
        self.decimation = 2
        self.scene = SimpleNamespace(num_envs=4)
        self.sim = SimpleNamespace(dt=1.0 / 120.0, device="cpu", gravity=(0.0, 0.0, -9.81))
        self.underwater_physics_cfg = SimpleNamespace(physics_version=2, hydrodynamics_model="hydro_v2")
        self.robot_cfg = SimpleNamespace(spawn=SimpleNamespace(scale=1.0))
        self.journal_geometry_tier = 2
        self.conditions = None
        self.condition_presets = ()


JournalCfgStub.__module__ = _STUB_MODULE.__name__


def test_d12_refuses_protected_fields_and_runs_journal_validate():
    for path in ("vehicle", "robot_cfg.spawn.scale", "underwater_physics_cfg.physics_version", "journal_task",
                 "journal_geometry_tier", "episode_length_s", "layout_mode", "max_obstacles", "appear_count",
                 "obs_sea_state", "sim.dt", "decimation", "conditions", "sim.gravity"):
        cfg = JournalCfgStub()
        _raises(cfg_override.OverrideError, "D12", cfg_override.apply_overrides, cfg, [path + "=1"], echo=False)
    cfg = JournalCfgStub()
    _STUB_MODULE.calls.clear()
    assert cfg_override.apply_overrides(cfg, ["goal_radius=3.0"], echo=False) == [("goal_radius", 3.0)]
    assert _STUB_MODULE.calls == [cfg]
    legacy = SimpleNamespace(max_obstacles=14)
    assert cfg_override.apply_overrides(legacy, ["max_obstacles=20"], echo=False) == [("max_obstacles", 20)]
    # resolve_protocol: suites at the protocol seed, then journal_validate
    cfg = JournalCfgStub()
    args = SimpleNamespace(eval_seed=42, level=0, num_envs=4, episodes=8, device="cpu", extra_sets=[],
                           suite=["waves=D2", "dynamic=D1"], suite_file=None)
    _STUB_MODULE.calls.clear()
    applied = eval_protocol.resolve_protocol(cfg, args)
    assert applied["seed"] == 42 and _STUB_MODULE.calls == [cfg]
    assert cfg.conditions.waves.enabled and cfg.journal_conditions.dynamic_dose == 1
    assert args.journal_suites["requested"] == ["waves=D2", "dynamic=D1"]
    _raises(cfg_override.OverrideError, "journal ids only", eval_protocol.apply_journal_suites,
            SimpleNamespace(journal_contract_version=0), SimpleNamespace(suite=["waves=D1"], suite_file=None))
    _raises(ValueError, "--suite", condition_args.apply_condition_args, JournalCfgStub(), ("waves_W1",))


# ------------------------------------------------ journal certificates --
_DOCS = {}
MANIFEST_SHA = "3" * 64
# Round T D24 (c): the offered admission grid every family header publishes; the glue pool
# gf.d7_pool("blueboat_v2", 2) admits 2 envs x 4 episodes.
GRID = {"num_envs": 2, "episodes": 4}
# Round T D20 (e): REAL family not-admitted stamps (scripts/make_d16_fixtures.py --write).
FIXTURE_DIR = os.path.join(_SCRIPTS, "fixtures_d16")
D16_FAMILIES = ("hazard_nav", "path_hazard", "path_following", "docking", "harbor_mission")
_FIXTURES = {}


def d16_fixture(family="hazard_nav"):
    """The saved real stamp document of one family (the reach carrier is hazard_nav)."""
    if family not in _FIXTURES:
        with open(os.path.join(FIXTURE_DIR, family + ".json"), encoding="ascii") as handle:
            _FIXTURES[family] = json.load(handle)
    return copy.deepcopy(_FIXTURES[family])


class PoolView:
    """A family pool as the round-4b families publish it: the admitted E4 pool plus
    per-dose counts that include not_admitted ([offered, admitted, not_admitted])."""

    def __init__(self, pool, not_admitted=0):
        self._pool = pool
        self.pool_digest, self.keys_offered, self.joint = pool.pool_digest, pool.keys_offered, pool.joint
        self.keys_admitted = pool.keys_admitted - not_admitted
        self.per_dose_counts = tuple((o, o - not_admitted, not_admitted) for o, _ in pool.per_dose_counts)

    def assignment(self, key, dose):
        return self._pool.assignment(key, dose)


def not_admitted_record(record, stamp):
    """A latched record whose D7 envelope is a family's REAL not-admitted stamp (D20 wire form)."""
    _, pairing, _ = validate_cert_v3.conditions_modules()
    out = copy.deepcopy(record)
    out["dynamic_hazards"] = copy.deepcopy(stamp)
    out["scenario_hashes"] = dict(out["scenario_hashes"])
    out["scenario_hashes"][pairing.D7_PRIMITIVE] = pairing.d7_primitive_hash(stamp)
    return out


def journal_env(tokens=(), seed=42, *, not_admitted=(), manifest=MANIFEST_SHA, contract=False, stamp=None,
                header=None):
    """(env, cfg): the F0 FakeJournalEnv (REAL mixin) as a round-T family env.

    not_admitted: env indices whose finished episode carries a family's REAL
    not-admitted stamp (``stamp``, default the hazard_nav fixture); manifest: the
    pinned pool manifest digest the env publishes (None = none); contract: the
    env also latches F5/F6 contract metrics; header: D7 header fields to set
    (value None = remove), applied after the D24 grid.
    """
    import glue_fixtures as gf

    cfg = gf.journal_cfg("blueboat_v2", tokens=tokens, seed=seed)
    pool = PoolView(gf.d7_pool("blueboat_v2", 2, seed=seed), len(not_admitted))
    env = gf.FakeJournalEnv(cfg, 2, deadline_s=0.5, pool=pool)
    env.dynamic_hazard_header["grid"] = dict(GRID)
    for field, value in (header or {}).items():
        if value is None:
            env.dynamic_hazard_header.pop(field, None)
        else:
            env.dynamic_hazard_header[field] = copy.deepcopy(value)
    if manifest is not None:
        env.journal_d7_pool_manifest_sha256 = manifest
    latched = env.journal_episode_record
    stamp = d16_fixture()["stamp"] if stamp is None else stamp
    env.journal_episode_record = (lambda i: not_admitted_record(latched(i), stamp) if i in not_admitted
                                  else latched(i))
    if contract:
        env.journal_episode_results = [{"berth_entry": True, "dwell_s": 1.5 + i, "heading_error_rad": 0.01}
                                       for i in range(2)]
        env.journal_contract_record = lambda i: {"phase_reached": 3, "env": i}
    return env, cfg


def journal_run(env, cfg, tokens=(), seed=42, *, runtime_paths=None, treatment=None, episodes_requested=2):
    """The REAL cert_v3 writer constructed over ``env`` (preflight checks run here)."""
    provenance = {k: "a" * 64 for k in ("tasks_sha256", "scripts_sha256", "evaluator_sha256")}
    provenance.update(tasks_root="t/tasks", scripts_root="t/scripts", environment_source="t/tasks/x/env.py",
                      evaluator_path="t/scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
    provenance["tree_sha256"] = cert_v3.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {k: "test" for k in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                   "isaac_lab_version", "physx_version")}
    ticks = itertools.count()
    argv = ["eval.py"] + (["--suite", *tokens] if tokens else [])
    cli_args = {"suite": list(tokens)} if tokens else {}
    for field, value in sorted((runtime_paths or {}).items()):
        argv += ["--set", field + "=" + value]
    if runtime_paths:
        cli_args["runtime_paths"] = dict(runtime_paths)
    if treatment is not None:
        cli_args["pairing_treatment"] = treatment
    return cert_v3.CertificateRun(base=env, env_cfg=cfg, evaluator=__file__, task=TASK_ID, controller="test",
                                  controller_config={}, checkpoint=None, eval_seed=seed, level=0,
                                  episodes_requested=episodes_requested, overrides={}, argv=argv,
                                  cli_args=cli_args, provenance=provenance,
                                  runtime=runtime, synchronize=lambda: None, clock=lambda: next(ticks) / 1000.0)


def journal_document(tokens=(), seed=42, *, not_admitted=(), manifest=MANIFEST_SHA, contract=False,
                     runtime_paths=None, treatment=None, stamp=None, header=None):
    """A 2 x 1 journal certificate from the REAL writer over the REAL mixin (journal_env)."""
    key = (tuple(tokens), seed, tuple(not_admitted), manifest, contract,
           tuple(sorted((runtime_paths or {}).items())), treatment,
           None if stamp is None else stamp["sha256"], json.dumps(header, sort_keys=True))
    if key in _DOCS:
        return copy.deepcopy(_DOCS[key])
    env, cfg = journal_env(tokens, seed, not_admitted=not_admitted, manifest=manifest, contract=contract,
                           stamp=stamp, header=header)
    run = journal_run(env, cfg, tokens, seed, runtime_paths=runtime_paths, treatment=treatment)
    env.reset()
    run.after_reset()
    records, done_envs = [], set()
    while len(done_envs) < 2:
        with run.time_controller():
            action = torch.tensor([[0.6, 0.1], [0.5, -0.1]])
        with run.time_environment():
            _, terminated, truncated = env.step(action)
        run.after_step()
        for i in (terminated | truncated).nonzero().flatten().tolist():
            if i in done_envs:
                run.discard_episode(i)
                continue
            records.append(run.finish_episode(i, dict(env=i, ep=0, success=False, terminated=bool(terminated[i]),
                                                      truncated=bool(truncated[i]), tts_s=None, path_length_m=1.0)))
            done_envs.add(i)
    doc = run.build({"records": records})
    _DOCS[key] = doc
    return copy.deepcopy(doc)


def _check(doc):
    return validate_cert_v3.validate_certificate(doc, num_envs=2, episodes_per_env=1)


def test_journal_certificate_validates_with_schema3_d7_and_pairing():
    doc = journal_document(("waves=D1",))
    assert not _check(doc), _check(doc)
    header = doc["header"]
    assert header["conditions"]["schema"] == "usvbench.conditions/3"
    assert header["pairing"]["schema"] == "usvbench.pairing/1" and header["journal_mixin"].startswith("usvbench.")
    assert header["journal"] == {"task": "reach", "carrier": "hazard_nav", "vehicle": "blueboat_v2",
                                 "contract_version": 2}
    assert header["suite_presets"]["suite"] == ["waves=D1"]
    for record in doc["records"]:
        assert set(record["scenario_hashes"]) == set(header["pairing"]["primitives"])
        assert set(record["dynamic_hazards"]) == {"payload", "sha256"}


def test_tampered_journal_certificates_are_rejected():
    doc = journal_document(("waves=D1",))
    bad = copy.deepcopy(doc)
    bad["records"][0]["dynamic_hazards"]["payload"]["runtime"]["tampered"] = True
    assert any("dynamic_hazards envelope rejected" in e for e in _check(bad))
    bad = copy.deepcopy(doc)
    bad["records"][1]["scenario_hashes"]["waves"] = "0" * 64
    assert any("does not match its stamp" in e for e in _check(bad))
    bad = copy.deepcopy(doc)
    bad["records"][0]["scenario_hashes"].pop("layout")
    assert any("inventory differs" in e for e in _check(bad))
    bad = copy.deepcopy(doc)
    bad["header"]["suite_presets"]["suite"] = ["waves=D2"]
    assert any("--suite tokens differ" in e for e in _check(bad))
    bad = copy.deepcopy(doc)
    bad["records"][0]["conditions"]["schema"] = "usvbench.conditions/2"
    assert _check(bad)


def test_single_suite_level_difference_pairs_and_other_differences_are_rejected():
    base = journal_document(("waves=D1",))
    treated = journal_document(("waves=D2",))
    pairs = analysis_input.validate_pair(base, treated, ("waves",))
    assert len(pairs) == 2
    # the treatment declaration must be the changed suite
    _raises(analysis_input.AnalysisInputError, "", analysis_input.validate_pair, base, treated, ("wind",))
    # a second suite that also differs is an unpaired difference
    other = journal_document(("waves=D2", "wind=D1"))
    _raises(analysis_input.AnalysisInputError, "wind", analysis_input.validate_pair, base, other, ("waves",))
    # a different evaluation seed is a different scenario
    reseeded = journal_document(("waves=D2",), seed=43)
    _raises(analysis_input.AnalysisInputError, "", analysis_input.validate_pair, base, reseeded, ("waves",))
    # a family primitive that changed outside the treatment is rejected
    forged = copy.deepcopy(treated)
    forged["records"][0]["scenario_hashes"]["layout"] = "f" * 64
    _raises(analysis_input.AnalysisInputError, "", analysis_input.validate_pair, base, forged, ("waves",))
    required = analysis_input._required_primitives(base["header"], set())
    assert required == set(base["header"]["pairing"]["primitives"])


# ------------------------------------------------------------ round 4b --
def test_runtime_pool_paths_are_allow_listed_stamped_and_never_refused():
    """D19: pool/asset variables are RUNTIME_ENV; pool-path cfg fields are runtime --set."""
    names = ("USVBENCH_D7_POOL_DIR", "USVBENCH_HARBOUR_D7_POOL", "USVBENCH_ASSETS_V2")
    assert set(names) <= set(eval_protocol.RUNTIME_ENV)
    environ = {"USVBENCH_D7_POOL_DIR": "/pools", "USVBENCH_HARBOUR_D7_POOL": "/pools/h.json",
               "USVBENCH_ASSETS_V2": "/assets_v2", "PATH": "/bin"}
    assert eval_protocol.ambient_environment((), environ) == {}
    assert eval_protocol.runtime_environment(environ) == {k: environ[k] for k in names}
    _raises(cfg_override.OverrideError, "--allow-env", eval_protocol.ambient_environment, (),
            {"USVBENCH_SOMETHING_ELSE": "1"})
    for field in cfg_override.JOURNAL_RUNTIME_FIELDS:
        assert cfg_override.journal_protected_reason(field) is None
        cfg_override.check_journal_assignments([field + "=/somewhere"])
    _raises(cfg_override.OverrideError, "D12", cfg_override.check_journal_assignments, ["journal_d7_pool=x"])
    _raises(cfg_override.OverrideError, "D12", cfg_override.check_journal_assignments, ["journal_task=x"])
    cfg = JournalCfgStub()
    args = SimpleNamespace(eval_seed=42, level=0, num_envs=4, episodes=8, device="cpu", suite=None, suite_file=None,
                           extra_sets=["journal_d7_pool_path=/x/pool.json", "goal_radius=3.0", "d7_pool_dir=/p"])
    applied = eval_protocol.resolve_protocol(cfg, args)
    assert cfg.journal_d7_pool_path == "/x/pool.json" and cfg.d7_pool_dir == "/p" and cfg.goal_radius == 3.0
    assert args.runtime_paths == {"d7_pool_dir": "/p", "journal_d7_pool_path": "/x/pool.json"}
    assert "journal_d7_pool_path" not in applied and "goal_radius" in applied
    typed = JournalCfgStub()
    typed.d7_pool_dir = 3
    _raises(cfg_override.OverrideError, "not a path", eval_protocol.resolve_protocol, typed,
            SimpleNamespace(eval_seed=42, level=0, num_envs=4, episodes=8, device="cpu", suite=None,
                            suite_file=None, extra_sets=["d7_pool_dir=/p"]))
    with patch.dict("os.environ", {}, clear=True):
        plan = run_cell.resolve_cell(_cell("--set", "d7_pool_dir=/pools", controller="mppi"))
    configuration, command = plan["configuration"], plan["command"]
    assert configuration["runtime_paths"] == {"d7_pool_dir": "/pools"}
    assert "d7_pool_dir" not in configuration["effective_override_literals"]
    assert command[command.index("d7_pool_dir=/pools") - 1] == "--set"
    cell = {"configuration": dict(configuration, effective_override_literals={})}
    launched = SimpleNamespace(level=0, episodes=128, num_envs=64, task=TASK_ID, extra_sets=["d7_pool_dir=/else"],
                               suite=None, seed_registration=configuration["seed_registration"],
                               release_id=None, checkpoint=None)
    eval_protocol.check_cell_config(cell, launched, [os.path.join(_SCRIPTS, "eval_mppi.py")])
    doc = journal_document(("waves=D1",), runtime_paths={"d7_pool_dir": "/pools"})
    assert doc["header"]["runtime_paths"] == {"d7_pool_dir": "/pools"}
    assert "d7_pool_dir" not in doc["header"]["overrides"]
    assert not _check(doc), _check(doc)
    bad = copy.deepcopy(doc)
    bad["header"]["runtime_paths"] = {"d7_pool_dir": "/other"}
    assert any("runtime path differs" in e for e in _check(bad))


def test_pool_manifest_digest_is_stamped_in_the_d7_header():
    doc = journal_document(("waves=D1",))
    assert doc["header"]["dynamic_hazards_header"]["pool_manifest_sha256"] == MANIFEST_SHA
    counts = doc["header"]["dynamic_hazards_header"]["per_dose_counts"]
    assert all(len(row) == 3 and row[0] == row[1] + row[2] for row in counts)
    _raises(ValueError, "pool manifest", journal_document, ("waves=D1",), manifest=None)
    _raises(ValueError, "SHA-256", journal_document, ("waves=D1",), manifest="not-a-digest")
    bad = copy.deepcopy(doc)
    del bad["header"]["dynamic_hazards_header"]["pool_manifest_sha256"]
    assert any("pool_manifest_sha256" in e for e in _check(bad))
    bad = copy.deepcopy(doc)
    bad["header"]["dynamic_hazards_header"]["per_dose_counts"] = [row[:2] for row in counts]
    assert any("not_admitted count" in e for e in _check(bad))
    bad = copy.deepcopy(doc)
    bad["header"]["dynamic_hazards_header"]["per_dose_counts"][0][1] += 1
    assert any("offered" in e for e in _check(bad))


def restamp(document, index, change):
    """Re-digest record ``index``'s D7 payload after ``change(payload)`` (E4 digest rule)."""
    _, _, d7 = validate_cert_v3.conditions_modules()
    payload = copy.deepcopy(document["records"][index]["dynamic_hazards"]["payload"])
    change(payload)
    document["records"][index]["dynamic_hazards"] = {"payload": payload, "sha256": d7.digest(payload)}
    return document


def test_not_admitted_records_validate_and_are_counted():
    doc = journal_document(("waves=D1",), not_admitted=(1,))
    assert not _check(doc), _check(doc)
    record = doc["records"][1]
    assert record["dynamic_hazards"] == d16_fixture()["stamp"]                # the REAL F1 stamp
    assert record["dynamic_hazards"]["payload"]["runtime"]["exposure"] == "not_admitted"
    assert analysis_input.dynamic_exposure(record) == "not_admitted"
    assert analysis_input.dynamic_exposure(doc["records"][0]) == "admitted"
    assert doc["header"]["dynamic_hazards_header"]["per_dose_counts"][0][2] == 1

    def no_reasons(payload):
        import d7_exposure
        for path in d7_exposure.REASON_SOURCES:          # every location the reader consults
            block = payload
            for name in path[:-1]:
                block = block.get(name) if isinstance(block, dict) else None
            if isinstance(block, dict) and path[-1] in block:
                block[path[-1]] = []

    bad = restamp(copy.deepcopy(doc), 1, no_reasons)
    assert any("rejection reasons" in e for e in _check(bad)), _check(bad)
    bad = restamp(copy.deepcopy(doc), 1, lambda p: p["assignment"].pop("candidate_digest"))
    assert any("candidate_digest" in e for e in _check(bad)), _check(bad)
    bad = restamp(copy.deepcopy(doc), 1, lambda p: p["runtime"].__setitem__("exposure", "partly"))
    assert any("exposure" in e for e in _check(bad))
    bad = copy.deepcopy(doc)
    bad["records"][1]["dynamic_hazards"]["exposure"] = "not_admitted"      # outside the digest
    assert _check(bad)
    bad = copy.deepcopy(doc)
    bad["records"][1]["scenario_hashes"]["d7_assignment"] = doc["records"][0]["scenario_hashes"]["d7_assignment"]
    assert any("d7_assignment" in e for e in _check(bad))
    # the header must count every not-admitted key at the run dose
    unc = journal_document(("waves=D1",))
    bad = copy.deepcopy(doc)
    bad["header"]["dynamic_hazards_header"]["per_dose_counts"] = unc["header"]["dynamic_hazards_header"][
        "per_dose_counts"]
    assert any("not-admitted keys" in e for e in _check(bad))


def test_not_admitted_pairs_for_other_suites_and_is_unexposed_for_the_dynamic_treatment():
    base = journal_document(("waves=D1",), not_admitted=(1,))
    treated = journal_document(("waves=D2",), not_admitted=(1,))
    detail = analysis_input.validate_journal_pair_detailed(base, treated, ("waves",))
    assert detail["status"] == ["paired", "paired"] and detail["unexposed"] == 0
    assert detail["not_admitted"] == {"baseline": 1, "treatment": 1}
    assert len(analysis_input.validate_pair(base, treated, ("waves",))) == 2
    # a key admitted in one certificate and not in the other differs outside a waves treatment
    admitted = journal_document(("waves=D2",))
    _raises(analysis_input.AnalysisInputError, "", analysis_input.validate_pair, base, admitted, ("waves",))
    # dynamic treatment: the not-admitted key ran D0 (no intruder) -> unexposed, not treated
    d0 = journal_document(("dynamic=D0",), not_admitted=(1,), treatment="dynamic")
    d2 = journal_document(("dynamic=D2",), not_admitted=(1,), treatment="dynamic")
    assert not _check(d2), _check(d2)
    detail = analysis_input.validate_journal_pair_detailed(d0, d2, ("dynamic",))
    assert detail["status"] == ["paired", "unexposed"] and detail["unexposed"] == 1 and detail["dynamic_treatment"]
    # CONTEXT_R4 D27: the unexposed pair is coverage, not a zero difference in the effect population
    assert detail["effect_pairs"] == [detail["pairs"][0]] and len(detail["pairs"]) == 2
    effect = analysis_input.validate_pair(d0, d2, ("dynamic",))
    assert [(a["env"], b["env"]) for a, b in effect] == [(0, 0)]
    # a not-admitted pair still rejects a difference outside the treatment
    forged = copy.deepcopy(d2)
    forged["records"][1]["scenario_hashes"]["layout"] = "f" * 64
    _raises(analysis_input.AnalysisInputError, "", analysis_input.validate_pair, d0, forged, ("dynamic",))


def test_d7_exposure_reader_and_the_pairing_primitive():
    """Round T D20 (d): the D11 primitive is always F0 pairing.d7_primitive_hash (F7's rule withdrawn)."""
    import d7_exposure

    _, pairing, d7 = validate_cert_v3.conditions_modules()
    stamp = d16_fixture()["stamp"]
    kind, reasons = d7_exposure.exposure_of(stamp)
    assert kind == "not_admitted" and reasons and "joint_ladder_member_rejected" not in reasons
    assert d7_exposure.d7_primitive(pairing, stamp) == pairing.d7_primitive_hash(stamp)
    assert d7_exposure.d7_primitive(pairing, stamp) == d16_fixture()["scenario_hashes"]["d7_assignment"]
    assert not hasattr(d7_exposure, "NOT_ADMITTED_CONTENT")
    stub = SimpleNamespace(d7_primitive_hash=lambda envelope: "e" * 64)
    assert d7_exposure.d7_primitive(stub, stamp) == "e" * 64
    both = copy.deepcopy(stamp)
    both["payload"]["exposure"] = "admitted"
    both["sha256"] = d7.digest(both["payload"])
    _raises(d7_exposure.ExposureError, "disagree", d7_exposure.exposure_of, both)
    empty = d7.make_stamp({}, {"exposure": "not_admitted", "rejection_reasons": ["x"]})
    _raises(d7_exposure.ExposureError, "needs the keys", d7_exposure.exposure_of, empty)   # withdrawn form
    counts = {"per_dose_counts": [[4, 4]] * 5, "per_dose_not_admitted": [0, 1, 0, 0, 0]}
    _raises(d7_exposure.ExposureError, "offered", d7_exposure.per_dose_counts, counts)
    counts["per_dose_counts"] = [[4, 4], [4, 3], [4, 4], [4, 4], [4, 4]]
    assert d7_exposure.per_dose_counts(counts)[1] == {"dose": 1, "offered": 4, "admitted": 3, "not_admitted": 1}
    rows = [{"offered": 2, "admitted": 1, "not_admitted": 1}] * 5
    assert d7_exposure.per_dose_counts({"per_dose_counts": rows})[4]["not_admitted"] == 1


def test_contract_metrics_are_exported_next_to_the_journal_record():
    doc = journal_document(("waves=D1",), contract=True)
    assert not _check(doc), _check(doc)
    for record in doc["records"]:
        block = record["journal_contract"]
        assert block["journal_contract_record"] == {"phase_reached": 3, "env": record["env"]}
        assert block["journal_episode_results"]["dwell_s"] == 1.5 + record["env"]
    plain = journal_document(("waves=D1",))
    assert all("journal_contract" not in record for record in plain["records"])


def test_docking_level_table_in_force_is_read_from_the_cfg():
    """F5 CROSS_OWNER_PROPOSAL_f7_docking_level_table (applied, round 4b)."""
    class DockCfg(JournalCfgStub):
        def __init__(self, in_force):
            super().__init__()
            self.journal_task = "docking"
            self.eval_level_distances = [4.0, 8.0]
            self.eval_level_distances_v2 = [5.0, 9.0]
            self.eval_level = 0
            self._in_force = in_force

        def docking_level_table_in_force(self):
            return self._in_force

    DockCfg.__module__ = _STUB_MODULE.__name__
    for table in ("eval_level_distances", "eval_level_distances_v2"):
        cfg = DockCfg("journal_plan")
        args = SimpleNamespace(eval_seed=42, level=0, num_envs=4, episodes=8, device="cpu", suite=None,
                               suite_file=None, extra_sets=[table + "=1.0,2.0"])
        _raises(cfg_override.OverrideError, "journal_plan", eval_protocol.resolve_protocol, cfg, args)
    cfg = DockCfg("eval_level_distances_v2")
    args = SimpleNamespace(eval_seed=42, level=0, num_envs=4, episodes=8, device="cpu", suite=None,
                           suite_file=None, extra_sets=["eval_level_distances=1.0,2.0"])
    _raises(cfg_override.OverrideError, "in force: eval_level_distances_v2", eval_protocol.resolve_protocol, cfg, args)
    stamp = {"docking_level_table_in_force": "journal_plan", "approach_distance_m": 12.0}
    base = SimpleNamespace(docking_level_table=stamp)
    got = cert_v3.docking_level_distance(DockCfg("journal_plan"), base, "2")
    assert got["table_field"] == "journal_plan" and got["approach_distance_m"] == 12.0
    header = {"resolved_config": {"eval_level_distances": [4.0], "journal_contract_version": 2},
              "level_distance": got, "journal": {"task": "docking"}, "physics_version": "2", "level": 0}
    assert validate_cert_v3.level_distance_errors(header) == []
    no_journal = {k: v for k, v in header.items() if k != "journal"}
    assert any("non-journal" in e for e in validate_cert_v3.level_distance_errors(no_journal))
    legacy_stamp = dict(header, level_distance={"table_field": "eval_level_distances"})
    assert any("journal_plan" in e for e in validate_cert_v3.level_distance_errors(legacy_stamp))


def test_journal_native_layouts_carry_the_ray_block_of_their_contract():
    """obs_superset.NATIVE_LAYOUTS journal entries vs F0 JOURNAL_TASK_CONTRACTS (item 6)."""
    import glue_fixtures as gf

    obs = gf.shared("obs_superset")
    groups = gf.observation_groups
    plans = eval_protocol.journal_plans()
    rays = set(range(10, 46))
    for (task, _), gym_id in plans.JOURNAL_ENV_IDS.items():
        layout = obs.NATIVE_LAYOUTS[gym_id]
        family, key = groups.JOURNAL_TASK_CONTRACTS[task]
        contract = groups.OBSERVATION_CONTRACTS[(family, key)]
        assert plans.TASK_TABLE[task]["carrier"] == family
        assert rays <= set(layout), gym_id
        assert set(layout[:3]) == {0, 1, 2}, gym_id          # position-derived nav channels first
        if contract.vel_channels == groups.FULL:
            assert {51, 52, 53} <= set(layout), gym_id       # surge, sway, yaw rate
        elif contract.vel_channels == groups.PLANAR:
            assert 9 in layout, gym_id                       # docking speed channel
        else:
            assert not ({51, 52, 53} & set(layout)), gym_id
    assert obs.NATIVE_LAYOUTS["Isaac-USV-HazardMovingBasin-Direct-v1"] == obs.NATIVE_LAYOUTS[
        "Isaac-USV-HazardNav-Direct-v3"]


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                with patch.dict("os.environ", {}, clear=False):
                    function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(1 if run_all() else 0)
