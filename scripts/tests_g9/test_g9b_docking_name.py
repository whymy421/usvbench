"""G9b (task T1.22): run_cell declares the controller name the docking certificates carry.

classical_baseline.py stamps "docking_3phase_pid" in every docking certificate (mode "docking":
classical_baseline.py:1103 and the controller branch at :1189) and "los_pid" for the other
baseline families; run_cell.py used to declare "los_pid" for docking too, so the plan recorded a
name the certificate never carries. Checks:
* run_cell plans for every legacy docking id and both journal docking ids (BlueBoat, WAM-V)
  record controller_name "docking_3phase_pid", with the same evaluator script, seed flag, extra
  arguments and information text as before;
* every other family and controller keeps its name (los_pid, planner_los_pid, staged_harbour,
  mppi, learned);
* the declared names equal what classical_baseline.py stamps (source check of the mode rule).
G17 update (owner decision B10 supplement, option B): the JOURNAL path following, station keeping
and docking cells run the base law wrapped by the obstacle-avoidance layer and declare
los_pid_avoid_v1 / docking_3phase_pid_avoid_v1 (journal_avoidance.CONTROLLER_NAMES, stamped by
classical_baseline.py on journal ids); every legacy id keeps los_pid / docking_3phase_pid.
G25 update (owner decision 2026-10-03, NC 09): the JOURNAL hazard_nav / path_hazard cells declare
planner_los_pid_mem_v1 (planner_memory.CONTROLLER_NAME); legacy hazard ids keep planner_los_pid.
G24b update (owner decisions 2026-10-04): one journal name per task -- path following los_pid_avoid_v2,
station keeping los_pid_cont_avoid_v2 (continuous_v2 hold), docking docking_runup_pid_avoid_v2 (run-up law),
planner planner_los_pid_mem_v2 (memory without expiry); every legacy name unchanged.
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g9/test_g9b_docking_name.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import pytest  # noqa: E402

import eval_protocol  # noqa: E402
import run_cell  # noqa: E402

DOCKING = "docking_3phase_pid"
JOURNAL_LOS = "los_pid_avoid_v2"                  # G17 -> G24b (layer memory without expiry)
JOURNAL_STATION = "los_pid_cont_avoid_v2"         # G24b (continuous_v2 hold law)
JOURNAL_DOCKING = "docking_runup_pid_avoid_v2"    # G24b (berth-axis run-up law)
JOURNAL_PLANNER = "planner_los_pid_mem_v2"        # G25 -> G24b (memory without expiry)
OUT = SCRIPTS.parents[1] / "test_scratch" / "g9b_cell.json"


def _plan(task, controller="classical", vessel="blueboat"):
    values = ["--task", task, "--vessel", vessel, "--controller", controller, "--eval-seed-block", "pilot-01",
              "--seed-namespace", "dose_pilots", "--run-purpose", "pilot", "--level", "0", "--episodes", "128",
              "--out", str(OUT), "--dry-run"]
    with patch.dict("os.environ", {}, clear=True):
        return run_cell.resolve_cell(run_cell.parser_for_cell().parse_args(values))


def _legacy_ids(family):
    return sorted(task for task, fam in run_cell.registered_tasks().items() if fam == family)


def test_maps_declare_the_docking_name_and_nothing_else_changes():
    assert run_cell.CLASSICAL_BY_FAMILY["docking"]["controller_name"] == DOCKING
    assert run_cell.JOURNAL_CLASSICAL_BY_CARRIER["docking"]["controller_name"] == JOURNAL_DOCKING
    assert run_cell.JOURNAL_DOCKING_BASELINE["controller_name"] == DOCKING      # pre-G17 base-law record
    for entry, base in ((run_cell.CLASSICAL_BY_FAMILY["docking"], run_cell.BASELINE),
                        (run_cell.JOURNAL_DOCKING_BASELINE, run_cell.JOURNAL_BASELINE)):
        assert {k: v for k, v in entry.items() if k != "controller_name"} == \
            {k: v for k, v in base.items() if k != "controller_name"}
    # G24b: the journal station-keeping / docking entries differ from the path-following one in name and the
    # information text only (the information names the base law)
    for entry in (run_cell.JOURNAL_CLASSICAL_BY_CARRIER["docking"], run_cell.JOURNAL_CLASSICAL_BY_CARRIER["station_keeping"]):
        assert {k: v for k, v in entry.items() if k not in ("controller_name", "information")} == \
            {k: v for k, v in run_cell.JOURNAL_AVOID_BASELINE.items() if k not in ("controller_name", "information")}
        assert entry["information"].startswith(run_cell.JOURNAL_AVOID_BASELINE["information"])
    assert run_cell.BASELINE["controller_name"] == run_cell.JOURNAL_BASELINE["controller_name"] == "los_pid"
    assert run_cell.JOURNAL_AVOID_BASELINE["controller_name"] == JOURNAL_LOS
    assert {k: v for k, v in run_cell.JOURNAL_AVOID_BASELINE.items() if k not in ("controller_name", "information")}         == {k: v for k, v in run_cell.JOURNAL_BASELINE.items() if k not in ("controller_name", "information")}
    names = {family: entry["controller_name"] for family, entry in run_cell.CLASSICAL_BY_FAMILY.items()}
    assert names == {"hazard_nav": "planner_los_pid", "path_hazard": "planner_los_pid",
                     "blueboat_calm_nav": "los_pid", "path_following": "los_pid", "station_keeping": "los_pid",
                     "docking": DOCKING}
    journal = {carrier: entry["controller_name"] for carrier, entry in run_cell.JOURNAL_CLASSICAL_BY_CARRIER.items()}
    assert journal == {"hazard_nav": JOURNAL_PLANNER, "path_hazard": JOURNAL_PLANNER,
                       "path_following": JOURNAL_LOS, "station_keeping": JOURNAL_STATION, "docking": JOURNAL_DOCKING,
                       "harbor_mission": "staged_harbour"}


@pytest.mark.parametrize("task", _legacy_ids("docking"))
def test_legacy_docking_plans_record_docking_3phase_pid(task):
    plan = _plan(task)
    config = plan["configuration"]
    assert config["task_family"] == "docking" and config["controller_name"] == DOCKING
    assert config["controller_information"] == "observation-only LOS/PID"
    command = plan["command"]
    assert Path(command[1]).name == "classical_baseline.py" and command[2:4] == ["--protocol", "mission"]
    assert "--seed" in command


@pytest.mark.parametrize("vessel", ["blueboat", "wamv"])
def test_journal_docking_plans_record_docking_runup_pid_avoid_v2(vessel):
    plan = _plan("journal:docking", vessel=vessel)
    config = plan["configuration"]
    assert config["journal"]["carrier_family"] == "docking" and config["controller_name"] == JOURNAL_DOCKING
    assert config["controller_information"] == run_cell.JOURNAL_AVOID_DOCKING_BASELINE["information"]
    assert Path(plan["command"][1]).name == "classical_baseline.py"


@pytest.mark.parametrize("task,controller,name", [
    ("Isaac-USV-StationKeep-BlueBoat-Direct-v1", "classical", "los_pid"),
    ("Isaac-USV-HazardCross-Direct-v1", "classical", "planner_los_pid"),
    ("Isaac-USV-Dock-BlueBoat-Direct-v1", "mppi", "mppi"),
    ("journal:path_following", "classical", JOURNAL_LOS),
    ("journal:station_keeping", "classical", JOURNAL_STATION),
    ("journal:reach", "classical", JOURNAL_PLANNER),
    ("journal:path_hazard", "classical", JOURNAL_PLANNER),
    ("journal:harbour_mission", "classical", "staged_harbour"),
    ("journal:docking", "mppi", "mppi"),
])
def test_other_families_and_controllers_keep_their_names(task, controller, name):
    assert _plan(task, controller=controller)["configuration"]["controller_name"] == name


def test_declared_names_equal_what_classical_baseline_stamps():
    """classical_baseline.py: controller_name = "docking_3phase_pid" if mode == "docking" else "los_pid"."""
    tree = ast.parse((SCRIPTS / "classical_baseline.py").read_text(encoding="utf-8"))
    rules = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "controller_name" for t in n.targets)
             and isinstance(n.value, ast.IfExp)]
    assert len(rules) == 1
    rule = rules[0]
    assert ast.unparse(rule.test) == "mode == 'docking'"
    assert (rule.body.value, rule.orelse.value) == (DOCKING, run_cell.BASELINE["controller_name"])
    modes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "JOURNAL_TASK_MODES"
                                                for t in node.targets):
            modes = ast.literal_eval(node.value)
    plans = eval_protocol.journal_plans()
    for task, spec in plans.TASK_TABLE.items():
        if spec["carrier"] == "docking":
            assert modes.get(task) == "docking", task       # journal docking certificates take the docking name


def test_g17_journal_names_equal_what_classical_baseline_stamps_on_journal_ids():
    """G17: classical_baseline.py stamps journal_avoidance.CONTROLLER_NAMES[mode] on the journal ids of the three
    JOURNAL_AVOIDANCE_MODES; run_cell declares the same names for those carriers."""
    import journal_avoidance as ja
    tree = ast.parse((SCRIPTS / "classical_baseline.py").read_text(encoding="utf-8"))
    stamps = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "controller_name" for t in n.targets)
              and isinstance(n.value, ast.Subscript)]
    assert stamps == ["ja.CONTROLLER_NAMES[mode]"]
    modes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("JOURNAL_TASK_MODES",
                                                                                     "JOURNAL_AVOIDANCE_MODES")
                                                for t in node.targets):
            modes[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(modes["JOURNAL_AVOIDANCE_MODES"]) == set(modes["JOURNAL_TASK_MODES"].values()) == set(ja.CONTROLLER_NAMES)
    for task, mode in modes["JOURNAL_TASK_MODES"].items():
        assert run_cell.JOURNAL_CLASSICAL_BY_CARRIER[task]["controller_name"] == ja.CONTROLLER_NAMES[mode]
    assert ja.CONTROLLER_NAMES == {"path_follow": JOURNAL_LOS, "station_keep": JOURNAL_STATION, "docking": JOURNAL_DOCKING}
