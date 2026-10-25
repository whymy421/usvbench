"""G6: CPU tests of scripts/gpu_build_check.py (the parent side; the worker needs Isaac Sim).

Covers the id selection, the per-id cfg path, the worker command, the per-id process
with its named timeout, PASS/FAIL parsing, the summary line and the exit code, with fake
worker processes. Run: python -B -m pytest -p no:cacheprovider scripts/test_g6_gpu_build_check.py -q
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("gpu_build_check", SCRIPTS / "gpu_build_check.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def args(**extra):
    values = dict(ids=None, vessel="both", steps=5, timeout_s=60, cfg_path="auto", suite=[], extra_sets=[],
                  d7_pool_dir=None, release_descriptor=None, log_dir=None, headless=True, device="cuda:0")
    values.update(extra)
    return SimpleNamespace(**values)


def test_ascii_parseable_and_no_isaac_import_at_module_level():
    data = (SCRIPTS / "gpu_build_check.py").read_bytes()
    assert all(b in (9, 10, 13) or 32 <= b <= 126 for b in data)
    tree = ast.parse(data.decode("ascii"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(name.split(".")[0] in ("isaaclab", "isaacsim", "omni", "torch", "gymnasium") for name in names)
    assert "isaaclab" not in sys.modules


def test_ids_and_cfg_paths():
    rows = G.journal_ids("both")
    assert len(rows) == 24 and len({r[2] for r in rows}) == 24
    assert len(G.journal_ids("blueboat")) == 12 and len(G.journal_ids("wamv")) == 12
    paths = {task: G.cfg_path_for(task, "auto") for task, _key, _id in rows}
    assert {t for t, p in paths.items() if p == "hydra"} == {"path_following", "station_keeping", "docking"}
    assert G.cfg_path_for("docking", "parse") == "parse"


def test_worker_command_mirrors_the_p5_cell():
    cmd = G.worker_command(args(d7_pool_dir="C:/pools", suite=["waves=D1", "dynamic=D1"],
                                release_descriptor="rel.json", extra_sets=["foo=1"]),
                           "docking", "Isaac-USV-Dock-WAMVVRX-Journal-Direct-v1", "hydra")
    text = " ".join(cmd)
    for part in ("--worker", "--task Isaac-USV-Dock-WAMVVRX-Journal-Direct-v1", "--cfg-path hydra",
                 "--num_envs 2", "--episodes 2", "--level 0", "--eval-seed 42", "--vessel wamv", "--headless",
                 "--device cuda:0", "--release-descriptor rel.json", "--set foo=1",
                 "--set d7_pool_dir=C:/pools", "--suite waves=D1 dynamic=D1"):
        assert part in text, part
    other = " ".join(G.worker_command(args(d7_pool_dir="C:/pools"), "reach",
                                      "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1", "parse"))
    assert "d7_pool_dir" not in other and "--vessel blueboat" in other


def fake(script):
    def command(_args, _task, _task_id, _cfg_path):
        return [sys.executable, "-B", "-c", script]
    return command


OK_SCRIPT = ("import json; print('kit noise'); print(%r + json.dumps({'stage': 'done', 'ok': True, 'error': '', "
             "'steps': 5}))" % G.RESULT_TAG)
FAIL_SCRIPT = ("import json, sys; print(%r + json.dumps({'stage': 'cfg', 'ok': False, 'error': "
               "'AttributeError: no setter', 'steps': 0})); sys.exit(1)" % G.RESULT_TAG)
CRASH_SCRIPT = "import sys; print('Traceback (most recent call last):'); print('RuntimeError: boom'); sys.exit(3)"
HANG_SCRIPT = "import time; time.sleep(60)"


def test_run_one_pass_fail_crash_timeout(tmp_path):
    a = args(timeout_s=3)
    ok = G.run_one(a, "reach", "id-ok", tmp_path, command_for=fake(OK_SCRIPT))
    assert ok["ok"] and ok["exit_code"] == 0 and ok["steps"] == 5
    bad = G.run_one(a, "reach", "id-bad", tmp_path, command_for=fake(FAIL_SCRIPT))
    assert not bad["ok"] and bad["stage"] == "cfg" and "no setter" in bad["error"]
    crash = G.run_one(a, "reach", "id-crash", tmp_path, command_for=fake(CRASH_SCRIPT))
    assert not crash["ok"] and crash["exit_code"] == 3 and "RuntimeError: boom" in crash["error"]
    hang = G.run_one(a, "reach", "id-hang", tmp_path, command_for=fake(HANG_SCRIPT))
    assert not hang["ok"] and hang["error"].startswith("TIMEOUT after 3 s")
    assert (tmp_path / "id-ok.log").read_text(encoding="utf-8").startswith("COMMAND: ")


def test_parent_summary_and_exit_code(tmp_path, monkeypatch, capsys):
    ids = ["Isaac-USV-Reach-BlueBoat-Journal-Direct-v1", "Isaac-USV-Dock-BlueBoat-Journal-Direct-v1"]
    monkeypatch.setattr(G, "worker_command", fake(OK_SCRIPT))
    assert G.parent_main(args(ids=ids, log_dir=str(tmp_path / "a"))) == 0
    out = capsys.readouterr().out
    assert "PASS Isaac-USV-Reach-BlueBoat-Journal-Direct-v1 [parse]" in out
    assert "PASS Isaac-USV-Dock-BlueBoat-Journal-Direct-v1 [hydra]" in out
    assert "SUMMARY: 2/2 PASS, 0 FAIL" in out
    monkeypatch.setattr(G, "worker_command", fake(FAIL_SCRIPT))
    assert G.parent_main(args(ids=ids[:1], log_dir=str(tmp_path / "b"))) == 1
    out = capsys.readouterr().out
    assert "FAIL Isaac-USV-Reach-BlueBoat-Journal-Direct-v1 [parse]" in out and "no setter" in out
    assert "SUMMARY: 0/1 PASS, 1 FAIL" in out
    with pytest.raises(SystemExit):
        G.parent_main(args(ids=["Isaac-USV-Nope-v1"], log_dir=str(tmp_path / "c")))



# ------------------------------------------------------------------ worker with stub simulator packages
OK_PARSE = '''"""G6 stand-in isaaclab_tasks.utils for a SUCCESSFUL worker run (fake cfg)."""
from types import SimpleNamespace


def parse_env_cfg(task_name, device="cuda:0", num_envs=None, use_fabric=None):
    return SimpleNamespace(seed=None, scene=SimpleNamespace(num_envs=num_envs), sim=SimpleNamespace(device=device))


def load_cfg_from_registry(task_name, entry_point_key):
    raise AttributeError("unused")
'''
OK_GYM = '''"""G6 stand-in gymnasium: make() returns a fake env that counts steps."""
import os
import types

registry = {}


def register(id, **kwargs):
    registry[id] = kwargs


class _Space:
    shape = (2, 2)


class _Env:
    def __init__(self):
        self.unwrapped = self
        self.action_space = _Space()
        self.device = "cpu"
        self.target_pos = None              # a goal-family env for eval_mppi._detect_mode

    def reset(self):
        return {}, {}

    def step(self, actions):
        with open(os.environ["G6_APP_MARK"], "a", encoding="ascii") as stream:
            stream.write("step%s\\n" % (tuple(actions.shape),))
        return None

    def close(self):
        with open(os.environ["G6_APP_MARK"], "a", encoding="ascii") as stream:
            stream.write("env-close\\n")


def make(task, cfg=None, render_mode=None):
    return _Env()


spaces = types.SimpleNamespace(Box=object, Space=object)
'''


def _worker(tmp_path, overrides, task_id="Isaac-USV-Reach-BlueBoat-Journal-Direct-v1", steps=3, stub_mode="fail_cfg"):
    import os
    import subprocess
    import test_g6_evaluator_exit as X
    stubs = tmp_path / "stubs"
    X.write_stubs(stubs)
    for relative, text in overrides.items():
        (stubs / relative).write_text(text, encoding="ascii")
    mark = tmp_path / "mark.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith(X.SCIENTIFIC_PREFIXES)}
    env.update(PYTHONPATH=str(stubs), G6_APP_MARK=str(mark), G6_STUB_MODE=stub_mode, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    cmd = G.worker_command(args(steps=steps), "reach", task_id, "parse")
    result = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    marks = mark.read_text(encoding="ascii").split("\n") if mark.exists() else []
    lines = [line for line in result.stdout.splitlines() if line.startswith(G.RESULT_TAG)]
    return result.returncode, marks, lines, result.stdout + result.stderr


def test_worker_reports_cfg_failure_and_exits_nonzero(tmp_path):
    code, marks, lines, output = _worker(tmp_path, {})
    assert code == 1, output[-2000:]
    assert len(lines) == 1 and '"stage": "cfg"' in lines[0] and "no setter" in lines[0], output[-2000:]
    assert not any(m.startswith("close") for m in marks) and "shutdown-with-open-app" not in marks   # G6b: no close() on failure


def test_worker_success_path_resets_steps_closes(tmp_path):
    code, marks, lines, output = _worker(tmp_path, {"isaaclab_tasks/utils/__init__.py": OK_PARSE,
                                                    "gymnasium/__init__.py": OK_GYM}, steps=3, stub_mode="ok")
    assert code == 0, output[-3000:]
    assert len(lines) == 1 and '"ok": true' in lines[0] and '"steps": 3' in lines[0], output[-2000:]
    assert '"mppi_mode": "goal"' in lines[0]
    assert [m for m in marks if m.startswith("step")] == ["step(2, 2)"] * 3
    closes = [i for i, m in enumerate(marks) if m.startswith("close:")]
    assert len(closes) == 1 and marks.index("env-close") < closes[0]


def test_worker_fails_at_the_interface_when_eval_mppi_refuses_the_env(tmp_path):
    refused = OK_GYM.replace("        self.target_pos = None              # a goal-family env for eval_mppi._detect_mode\n", "")
    assert refused != OK_GYM
    code, marks, lines, output = _worker(tmp_path, {"isaaclab_tasks/utils/__init__.py": OK_PARSE,
                                                    "gymnasium/__init__.py": refused}, stub_mode="ok")
    assert code == 1, output[-2000:]
    assert len(lines) == 1 and '"stage": "interface"' in lines[0] and "not a supported task family" in lines[0]
    assert not any(m.startswith("close") for m in marks)                        # G6b: no close() on failure
