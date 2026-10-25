"""G6 (scope additions of 2026-10-01): evaluator <-> journal env interfaces, CPU only.

* tasks/_shared/body_wrench.apply_body_wrench: the Isaac Lab 2.3.2 replacement call when the
  robot has a permanent_wrench_composer (keyword forces/torques, defaults for env/body ids,
  positions and is_global, exactly what the deprecated method passes), else the old call;
  every journal _journal_apply_action uses it, no legacy method does.
* classical_baseline._task_mode: journal docking / station keeping / path following map
  from cfg.journal_task; legacy cfgs keep the attribute probe.
(The published dock_point / dock_heading / hold_point are tested on the real journal envs in
tasks/docking/test_journal_docking_env.py and tasks/station_keeping/tests_journal/
test_station_journal_env.py, class G6PublishedTargets.)
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g6/test_g6_journal_interfaces.py -q
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("g6_body_wrench", REPO / "tasks" / "_shared" / "body_wrench.py")
BW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BW)


class Composer:
    def __init__(self):
        self.calls = []

    def set_forces_and_torques(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class NewRobot:
    def __init__(self):
        self.permanent_wrench_composer = Composer()

    def set_external_force_and_torque(self, *args, **kwargs):
        raise AssertionError("deprecated call used although the composer exists")


class OldRobot:
    def __init__(self):
        self.calls = []

    def set_external_force_and_torque(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_apply_body_wrench_uses_the_composer_with_the_deprecated_defaults():
    forces, torques = object(), object()
    robot = NewRobot()
    BW.apply_body_wrench(robot, forces, torques)
    assert robot.permanent_wrench_composer.calls == [((), {"forces": forces, "torques": torques})]
    old = OldRobot()
    BW.apply_body_wrench(old, forces, torques)
    assert old.calls == [((forces, torques), {})]


JOURNAL_SITES = {
    "tasks/docking/docking_env.py": "_journal_apply_action",
    "tasks/harbor_mission/harbor_mission_env.py": "_journal_apply_action",
    "tasks/hazard_nav/hazard_nav_env.py": "_journal_apply_action",
    "tasks/path_following/journal_path_following.py": "_journal_apply_action",
    "tasks/path_hazard/path_hazard_env.py": "_journal_apply_action",
    "tasks/station_keeping/station_keeping_env.py": "_journal_apply_action",
}


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Name) and n.func.id == name)
        or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def test_journal_apply_action_uses_the_helper_and_legacy_methods_do_not():
    for relative, method in JOURNAL_SITES.items():
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        journal = [f for f in functions if f.name == method]
        assert len(journal) == 1, relative
        assert len(_calls(journal[0], "apply_body_wrench")) == 1, relative
        assert _calls(journal[0], "set_external_force_and_torque") == [], relative
        legacy = [f for f in functions if f.name == "_apply_action"]
        for function in legacy:                                     # legacy paths keep the old call
            assert _calls(function, "apply_body_wrench") == [], relative


def _task_mode():
    path = REPO / "scripts" / "classical_baseline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "_task_mode") or (
        isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "JOURNAL_TASK_MODES" for t in n.targets))]
    assert len(body) == 2
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_task_mode"]


def test_classical_baseline_task_mode_journal_mapping_and_legacy_probe():
    mode = _task_mode()
    bare = SimpleNamespace()
    for task, expected in (("docking", "docking"), ("station_keeping", "station_keep"),
                           ("path_following", "path_follow")):
        assert mode(bare, SimpleNamespace(journal_contract_version=2, journal_task=task)) == expected
    # journal tasks outside the mapping and legacy cfgs: the attribute probe, unchanged
    assert mode(bare, SimpleNamespace(journal_contract_version=2, journal_task="reach")) == "point_nav"
    legacy_dock = SimpleNamespace(dock_point=1, dock_heading=1, hold_timer=1, episode_path_length=1)
    assert mode(legacy_dock, SimpleNamespace()) == "docking"
    assert mode(bare, SimpleNamespace(hold_radius=2.0)) == "station_keep"
    assert mode(bare, SimpleNamespace(journal_contract_version=1, journal_task="docking")) == "point_nav"
