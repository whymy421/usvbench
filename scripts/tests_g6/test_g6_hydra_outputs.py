"""G6: Hydra-decorated evaluators write nothing outside their output folder.

P5 run 1: classical_baseline.py (Isaac Lab hydra_task_config) created ./outputs/<date>/<time>/
(.hydra/ snapshot + job log) in its working directory, the release root. The evaluators now
append scripts/hydra_outputs.run_dir_overrides(<output path>, hydra_args) to the sys.argv they
already rebuild for hydra.main.

* With an Isaac Lab 2.3.2 checkout (REAL isaaclab_tasks/utils/hydra.py, real hydra-core): a
  child process runs hydra_task_config(<task>, "skrl_cfg_entry_point") exactly as the
  evaluators call it, in a temporary working directory, once with the evaluator's overrides
  (nothing may appear outside the --out folder, nothing new inside it) and once without them
  (control: Hydra's default outputs/ appears in the working directory, the P5 behaviour).
* Always: every Hydra-decorated evaluator builds its sys.argv through run_dir_overrides, and
  the overrides themselves (quoting, caller-given keys left alone).
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g6/test_g6_hydra_outputs.py -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for entry in (str(HERE), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
import hydra_outputs  # noqa: E402
import isaaclab_reference as R  # noqa: E402

REF, WHY = R.find_reference()
HYDRA_EVALUATORS = {
    "classical_baseline.py": "hydra_outputs.run_dir_overrides(args_cli.out or args_cli.csv, hydra_args)",
    "eval_benchmark.py": "hydra_outputs.run_dir_overrides(args_cli.csv, hydra_args)",
    "eval_hazard_nav.py": "hydra_outputs.run_dir_overrides(args_cli.output_json, hydra_args)",
    "eval_cross_champion.py": "hydra_outputs.run_dir_overrides(args_cli.out, hydra_args)",
}

CHILD = r'''
import json, os, sys
sys.dont_write_bytecode = True
sys.path[:0] = [SCRIPTS_DIR, HERE_DIR]
import isaaclab_reference as R
restore, modules = R.install(REF_ROOT)
hydra_mod = R.install_task_utils(REF_ROOT)
import gymnasium as gym
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import hydra_outputs

@configclass
class TinyEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    observation_space = 3
    action_space = 2
    state_space = 0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=10.0)
    journal_like_field: float = 1.5

module = type(sys)("g6_tiny_task")
module.TinyEnvCfg = TinyEnvCfg
sys.modules["g6_tiny_task"] = module
gym.register(id="G6-Tiny-v0", entry_point="g6_tiny_task:Nothing",
             kwargs={"env_cfg_entry_point": "g6_tiny_task:TinyEnvCfg", "skrl_cfg_entry_point": AGENT_YAML})
seen = {}

def task(env_cfg, agent_cfg):
    seen["field"] = env_cfg.journal_like_field
    seen["agent"] = agent_cfg

sys.argv = [sys.argv[0]] + (hydra_outputs.run_dir_overrides(OUT_PATH, []) if WITH_OVERRIDES else [])
hydra_mod.hydra_task_config("G6-Tiny-v0", "skrl_cfg_entry_point")(task)()
print("G6-RESULT " + json.dumps(seen))
'''


def _tree(root):
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in Path(root).rglob("*"))


def _run_child(tmp_path, with_overrides):
    pytest.importorskip("hydra")
    work = tmp_path / "work"
    out_dir = tmp_path / "out"
    work.mkdir()
    out_dir.mkdir()
    agent = tmp_path / "agent.yaml"
    agent.write_text("seed: 42\nagent:\n  experiment:\n    directory: logs\n", encoding="ascii")
    script = tmp_path / "child.py"
    header = "\n".join([f"SCRIPTS_DIR = {str(SCRIPTS)!r}", f"HERE_DIR = {str(HERE)!r}", f"REF_ROOT = {str(REF)!r}",
                        f"AGENT_YAML = {str(agent)!r}", f"OUT_PATH = {str(out_dir / 'cert.json')!r}",
                        f"WITH_OVERRIDES = {bool(with_overrides)!r}"])
    script.write_text(header + CHILD, encoding="utf-8")
    before = {"work": _tree(work), "out": _tree(out_dir), "tmp": _tree(tmp_path)}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("HYDRA_FULL_ERROR", None)
    result = subprocess.run([sys.executable, "-B", str(script)], cwd=str(work), env=env, capture_output=True,
                            text=True, timeout=600)
    lines = [line for line in result.stdout.splitlines() if line.startswith("G6-RESULT ")]
    assert result.returncode == 0 and len(lines) == 1, result.stdout[-2000:] + result.stderr[-3000:]
    seen = json.loads(lines[0][len("G6-RESULT "):])
    assert seen["field"] == 1.5 and seen["agent"]["seed"] == 42
    after = {"work": _tree(work), "out": _tree(out_dir), "tmp": _tree(tmp_path)}
    return before, after


@pytest.mark.skipif(REF is None, reason="Isaac Lab 2.3.2 reference not found: " + str(WHY))
def test_evaluator_hydra_configuration_writes_nothing_outside_the_out_folder(tmp_path):
    before, after = _run_child(tmp_path, with_overrides=True)
    assert after["work"] == before["work"] == [], after["work"]          # nothing in the working directory
    assert after["out"] == before["out"] == [], after["out"]             # and nothing new beside --out
    assert set(after["tmp"]) == set(before["tmp"]), sorted(set(after["tmp"]) - set(before["tmp"]))


@pytest.mark.skipif(REF is None, reason="Isaac Lab 2.3.2 reference not found: " + str(WHY))
def test_control_without_the_overrides_hydra_writes_into_the_working_directory(tmp_path):
    _, after = _run_child(tmp_path, with_overrides=False)
    assert any(path.startswith("outputs/") and path.endswith(".hydra/config.yaml") for path in after["work"]), \
        after["work"]


def test_every_hydra_evaluator_builds_its_argv_through_the_overrides():
    for name, call in HYDRA_EVALUATORS.items():
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "hydra_task_config" in source
        assert source.count("sys.argv = [sys.argv[0]] + hydra_args + " + call) == 1, name
        assert "sys.argv = [sys.argv[0]] + hydra_args\n" not in source.replace("\r\n", "\n"), name


def test_overrides_quote_paths_and_leave_caller_keys():
    overrides = hydra_outputs.run_dir_overrides("C:/some dir/cells/cert.json")
    assert overrides[0].startswith("hydra.run.dir='") and overrides[0].endswith("/some dir/cells'")
    assert overrides[1:] == ["hydra.output_subdir=null", "hydra/job_logging=disabled", "hydra/hydra_logging=disabled"]
    kept = hydra_outputs.run_dir_overrides("x/cert.json", ["hydra.run.dir=elsewhere", "+hydra/job_logging=stdout"])
    assert kept == ["hydra.output_subdir=null", "hydra/hydra_logging=disabled"]
    assert hydra_outputs.run_dir_overrides(None)[0].startswith("hydra.run.dir='")
