"""G6 / G6b: evaluator exit codes and certificate publication (CPU only, no Isaac import here).

P5 (2026-10-01): classical_planner_pid.py exited 0 without a certificate after a cfg traceback.
P5 v3 (release v2, G6b): (1) Isaac Sim 5.1's SimulationApp.close() ended every SUCCESSFUL run with
code 0 before the first guard renamed ``<out>.partial`` to ``<out>``; (2) after an exception in
cert_v3 (classical_baseline, Hydra path) close() hung for 26 minutes. scripts/evaluator_exit.py
now publishes first and closes under a watchdog on success, and never closes on failure.

* unit: evaluator_exit (publish before close, verify, watchdog, no close on failure, bytes).
* source: each evaluator writes its certificate only through the guard and never closes the app
  itself; classical_baseline makes Hydra re-raise (HYDRA_FULL_ERROR=1).
* subprocess, all five evaluators, against stub isaaclab / isaaclab_tasks / gymnasium / skrl
  packages and a stub app whose close() (a) returns, (b) ends the process with os._exit(0) like
  Isaac Sim 5.1, (c) blocks forever:
    SUCCESS (stub cert writer with quota 0, so the real evaluator runs its real certificate write,
    guard and close) leaves ``<out>`` (published BEFORE close was called) and exits 0 in all three;
    FAILURE (cfg stage; for classical_baseline also inside the Hydra-decorated main, both with
    Hydra re-raising and with Hydra's sys.exit(1) path) leaves no file, exits 1, never calls
    close() and does not hang.
Run: python -B -m pytest -p no:cacheprovider scripts/test_g6_evaluator_exit.py -q
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import evaluator_exit  # noqa: E402

EVALUATORS = ("classical_planner_pid.py", "classical_baseline.py", "classical_harbour.py",
              "eval_mppi.py", "eval_v6_frozen.py")
CLOSE_MODES = ("normal", "exit0", "block")
P5_CFG_ERROR = "property 'obstacle_slot_count' of 'ReachBlueBoatJournalEnvCfg' object has no setter"
TEST_WATCHDOG_S = 3            # the subprocesses shorten evaluator_exit.CLOSE_WATCHDOG_S (sitecustomize)
SUBPROCESS_TIMEOUT_S = 120     # a hang is a test failure, never a stuck test run


# ------------------------------------------------------------------ unit
class HardExit(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class FakeApp:
    def __init__(self, out=None, fail=False):
        self.closed = 0
        self.fail = fail
        self.out = out
        self.cert_present_at_close = None

    def close(self):
        self.closed += 1
        self.cert_present_at_close = None if self.out is None else Path(self.out).exists()
        if self.fail:
            raise RuntimeError("close failed")


@pytest.fixture
def hard_exit(monkeypatch):
    def fake(code):
        raise HardExit(code)
    monkeypatch.setattr(evaluator_exit, "_hard_exit", fake)


def _body_writing(guard, path, payload, then=None):
    def body():
        with guard.output(path).open() as stream:
            json.dump(payload, stream, indent=1)
        if then is not None:
            then()
    return body


def test_success_publishes_identical_bytes_before_close(tmp_path, hard_exit):
    payload = {"header": {"a": [1, 2.5, None], "b": "x"}, "records": [{"env": 0}]}
    legacy = tmp_path / "legacy.json"
    with open(legacy, "w", encoding="utf-8") as stream:      # the evaluators' former call
        json.dump(payload, stream, indent=1)
    out = tmp_path / "cert.json"
    guard, app = evaluator_exit.Guard(), FakeApp(out)
    assert evaluator_exit.run(_body_writing(guard, out, payload), guard, app=app) == 0
    assert app.closed == 1 and app.cert_present_at_close is True      # published BEFORE close()
    assert out.read_bytes() == legacy.read_bytes()
    assert not Path(str(out) + evaluator_exit.PARTIAL_SUFFIX).exists()


@pytest.mark.parametrize("failure, code", [
    (RuntimeError("episode loop"), 1),
    (SystemExit(1), 1),
    (SystemExit(3), 3),
    (SystemExit("eval_mppi: control_step_s mismatch"), 1),
    (KeyboardInterrupt(), 1),
])
def test_failure_never_closes_and_leaves_nothing(tmp_path, hard_exit, failure, code):
    out = tmp_path / "cert.json"
    guard, app = evaluator_exit.Guard(), FakeApp(out)

    def boom():
        raise failure
    with pytest.raises(HardExit) as raised:
        evaluator_exit.run(_body_writing(guard, out, {"x": 1}, then=boom), guard, app=app)
    assert raised.value.code == code
    assert app.closed == 0                                            # never close() on failure
    assert not out.exists() and not Path(str(out) + evaluator_exit.PARTIAL_SUFFIX).exists()


def test_certificate_build_refusal_leaves_no_file(tmp_path, hard_exit):
    out = tmp_path / "cert.json"
    guard, app = evaluator_exit.Guard(), FakeApp(out)

    def refused():
        raise ValueError("journal env publishes no pinned D7 pool manifest digest (D19)")

    def body():
        with guard.output(out).open() as stream:
            json.dump(refused(), stream, indent=1)
    with pytest.raises(HardExit) as raised:
        evaluator_exit.run(body, guard, app=app)
    assert raised.value.code == 1 and app.closed == 0
    assert not out.exists() and not Path(str(out) + evaluator_exit.PARTIAL_SUFFIX).exists()


def test_unreadable_partial_is_a_failure_and_is_deleted(tmp_path, hard_exit):
    out = tmp_path / "cert.json"
    guard, app = evaluator_exit.Guard(), FakeApp(out)

    def body():
        with guard.output(out).open() as stream:
            stream.write('{"truncated": ')
    with pytest.raises(HardExit) as raised:
        evaluator_exit.run(body, guard, app=app)
    assert raised.value.code == 1 and app.closed == 0
    assert not out.exists() and not Path(str(out) + evaluator_exit.PARTIAL_SUFFIX).exists()


def test_close_raising_after_publication_still_exits_zero(tmp_path, hard_exit):
    out = tmp_path / "cert.json"
    guard, app = evaluator_exit.Guard(), FakeApp(out, fail=True)
    assert evaluator_exit.run(_body_writing(guard, out, {"x": 1}), guard, app=app) == 0
    assert app.closed == 1 and out.exists()


def test_failure_before_app_launch_and_without_out(hard_exit):
    guard = evaluator_exit.Guard()

    def body():
        raise ValueError("cfg")
    with pytest.raises(HardExit) as raised:
        evaluator_exit.run(body, guard)
    assert raised.value.code == 1


def test_finish_runs_once(tmp_path, hard_exit):
    guard, app = evaluator_exit.Guard(), FakeApp()
    guard.attach_app(app)
    assert guard.finish() == 0 and guard.finish() == 0 and app.closed == 1


def test_exit_code_mapping_and_named_watchdog():
    assert [evaluator_exit.exit_code(v) for v in (None, 0, 2, True, False)] == [0, 0, 2, 1, 0]
    assert evaluator_exit.CLOSE_WATCHDOG_S == 120


# ------------------------------------------------------------------ source
def _source(name):
    return (SCRIPTS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", EVALUATORS)
def test_evaluator_writes_and_closes_only_through_the_guard(name):
    source = _source(name)
    tree = ast.parse(source)
    assert "_EXIT_GUARD = evaluator_exit.Guard()" in source
    assert 'open(args_cli.out, "w"' not in source
    assert source.count("_EXIT_GUARD.output(args_cli.out).open()") == 1
    closes = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "close" and isinstance(n.func.value, ast.Name)
              and n.func.value.id in ("app", "simulation_app")]
    assert closes == [], name
    if name == "eval_v6_frozen.py":
        assert "_EXIT_GUARD.install_process_hooks()" in source and "\n_EXIT_GUARD.finish()" in source
        assert "_EXIT_GUARD.attach_app(app)" in source
    elif name == "classical_baseline.py":
        assert "evaluator_exit.run(main, _EXIT_GUARD, app=simulation_app)" in source
        assert 'os.environ["HYDRA_FULL_ERROR"] = "1"' in source
    else:
        assert "evaluator_exit.run(_run_isaac_eval, _EXIT_GUARD)" in source
        assert "_EXIT_GUARD.attach_app(app)" in source


# ------------------------------------------------------------------ subprocess stubs
APP_STUB = '''"""G6b test stand-in for isaaclab.app (no Kit)."""
import atexit
import os
import time

MARK = os.environ["G6_APP_MARK"]
MODE = os.environ.get("G6_APP_CLOSE_MODE", "normal")


def _mark(text):
    with open(MARK, "a", encoding="ascii") as stream:
        stream.write(text + "\\n")


class _App:
    def __init__(self):
        self._closed = False
        # The OBSERVED P5 behaviour: an interpreter shutdown with the Kit app still open ended
        # the process with code 0 after the traceback.
        atexit.register(self._shutdown_with_open_app)

    def _shutdown_with_open_app(self):
        if not self._closed:
            _mark("shutdown-with-open-app")
            os._exit(0)

    def close(self):
        out = os.environ.get("G6_OUT")
        _mark("close:cert-present" if out and os.path.exists(out) else "close:cert-absent")
        self._closed = True
        if MODE == "exit0":           # Isaac Sim 5.1 on the success path (P5 v3)
            _mark("close-exit0")
            os._exit(0)
        if MODE == "block":           # the P5 v3 failure-path hang
            _mark("close-block")
            while True:
                time.sleep(1.0)


class AppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", type=str, default=None)

    def __init__(self, args):
        _mark("launch")
        self.app = _App()
'''
CFG_HELPERS = '''
import os
from types import SimpleNamespace


def ok_cfg():
    """A minimal NON-journal env cfg (resolve_protocol's protected fields and what the evaluators read)."""
    return SimpleNamespace(seed=None, scene=SimpleNamespace(num_envs=2), decimation=2, vehicle="blueboat_v2",
                           sim=SimpleNamespace(device="cuda:0", dt=1.0 / 120.0), hold_radius=2.0,
                           goal_radius=2.0, success_position_tolerance_m=2.5, conditions=None,
                           condition_presets=())


def mode():
    return os.environ.get("G6_STUB_MODE", "fail_cfg")
'''
TASK_UTILS_STUB = CFG_HELPERS + '''
P5 = %r


def parse_env_cfg(task_name, device="cuda:0", num_envs=None, use_fabric=None):
    if mode() == "fail_cfg":
        raise AttributeError(P5)
    return ok_cfg()


def load_cfg_from_registry(task_name, entry_point_key):
    if mode() == "fail_cfg":
        raise AttributeError(P5)
    return {"seed": 42, "trainer": {}, "agent": {"experiment": {}}}
''' % P5_CFG_ERROR
HYDRA_STUB = CFG_HELPERS + '''
import dataclasses
import sys
import traceback

P5 = %r


def hydra_task_config(task_name, agent_cfg_entry_point):
    """hydra.main's run_and_report (hydra-core 1.3.5, hydra/_internal/utils.py:218-260): the task
    function's exception is re-raised when HYDRA_FULL_ERROR=1, else printed + sys.exit(1)."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if mode() == "fail_cfg":
                    raise dataclasses.FrozenInstanceError("cannot assign to field 'geometry_tier'")
                func(ok_cfg(), {"seed": 42}, *args, **kwargs)
            except Exception:
                if os.environ.get("HYDRA_FULL_ERROR") == "1" and mode() != "fail_hydra_sysexit":
                    raise
                traceback.print_exc()
                print("Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.",
                      file=sys.stderr)
                sys.exit(1)
        return wrapper
    return decorator
''' % P5_CFG_ERROR
GYM_STUB = CFG_HELPERS + '''
import types

import torch

registry = {}


def register(id, **kwargs):
    registry[id] = kwargs


class _Space:
    shape = (2, 2)


class FakeEnv:
    """Enough of a DirectRLEnv for the evaluators' setup and certificate tail (no episode runs:
    the stub writer's quota is 0)."""

    def __init__(self, cfg):
        n = 2
        self.cfg = cfg
        self.unwrapped = self
        self.num_envs = n
        self.device = "cpu"
        self.control_step_s = 1.0 / 60.0
        self.max_episode_length = 10
        self.action_space = _Space()
        self.target_pos = torch.zeros((n, 2))
        self.episode_success = torch.zeros(n, dtype=torch.bool)
        self.time_to_success = torch.full((n,), float("nan"))
        self.episode_path_length = torch.zeros(n)
        self.vehicle_spec = types.SimpleNamespace(name="blueboat")
        self.scene = types.SimpleNamespace(env_origins=torch.zeros((n, 3)))

    def reset(self):
        return {"policy": torch.zeros((self.num_envs, 42))}, {}

    def step(self, actions):
        n = self.num_envs
        return ({"policy": torch.zeros((n, 42))}, torch.zeros(n), torch.ones(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool), {})

    def close(self):
        pass


def make(task, cfg=None, render_mode=None):
    if mode() in ("fail_env", "fail_hydra_sysexit"):
        raise RuntimeError("stub gym.make: env build failed")
    return FakeEnv(cfg)


spaces = types.SimpleNamespace(Box=object, Space=object)
'''
SKRL_WRAPPER_STUB = '''
class SkrlVecEnvWrapper:
    def __init__(self, env, ml_framework="torch"):
        self._env = env
        self.unwrapped = env.unwrapped

    def reset(self):
        return self._env.reset()

    def step(self, actions):
        return self._env.step(actions)

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)
'''
RUNNER_STUB = '''
class _Agent:
    cfg = {}
    models = {}

    def load(self, path):
        pass

    def set_running_mode(self, mode):
        pass


class Runner:
    def __init__(self, env, cfg):
        self.agent = _Agent()

    def _component(self, name):
        raise KeyError(name)
'''
# Only in G6_STUB_MODE=ok: a cert_v3 / journal_controllers / mppi_controller stand-in that lets the
# REAL evaluator reach its REAL certificate write, guard and close without episodes (quota 0).
SITECUSTOMIZE = '''
import os
import sys
import types

if os.environ.get("G6_STUB_MODE") == "ok":
    import contextlib

    class CertificateRun:
        quota = 0

        def __init__(self, **kwargs):
            self.header = {"controller": {"configuration": {}}}

        def after_reset(self):
            pass

        def after_step(self):
            pass

        @contextlib.contextmanager
        def time_controller(self):
            yield

        @contextlib.contextmanager
        def time_environment(self):
            yield

        def discard_episode(self, i):
            pass

        def finish_episode(self, i, record):
            return record

        def build(self, payload):
            return {"header": {"stub": "g6b"}, "records": payload.get("records", []),
                    "payload_keys": sorted(payload)}

    cert = types.ModuleType("cert_v3")
    cert.CertificateRun = CertificateRun
    sys.modules["cert_v3"] = cert

    jc = types.ModuleType("journal_controllers")
    jc.INFORMATION_DIET = "stub"

    class _Law:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, *args, **kwargs):
            pass

        def __getattr__(self, name):          # gains / radii read by the controllers' stamp()
            return 0.0

    jc.LOSController = jc.DockingLaw = jc.WaypointTracker = _Law
    jc.load_gains = lambda vessel, path=None: {"los_heading_pid": {}, "dock_yaw_pd": {}}
    jc.length_scale = lambda vessel: 1.0
    jc.ray_max_range = lambda cfg: 30.0
    jc.controller_stamp = lambda cfg, gains, extra=None: {"stub": True}
    _hull = types.SimpleNamespace(collision=types.SimpleNamespace(capsule_radius_m=0.4), beam_overall_m=0.9)
    jc.shared_module = lambda name: types.SimpleNamespace(get_vehicle_v2=lambda vessel: _hull)
    sys.modules["journal_controllers"] = jc

    import torch

    mc = types.ModuleType("mppi_controller")

    class MPPIParams:
        def __init__(self, **kwargs):
            self._values = dict(kwargs)

        def as_dict(self):
            return dict(self._values)

    class _Dynamics:
        vessel = "blueboat_v2"
        dtype = torch.float32
        control_dt_s = 1.0 / 60.0
        params = {"model": "stub", "physics_dt_s": 1.0 / 120.0, "decimation": 2, "swayyaw_damping": 0.0,
                  "surge_lin": 0.0, "surge_quad": 0.0, "sway_lin": 0.0, "sway_quad": 0.0, "yaw_lin": 0.0,
                  "yaw_quad": 0.0}

        def __init__(self, *args, **kwargs):
            pass

    class MPPIController:
        dtype = torch.float32

        def __init__(self, **kwargs):
            pass

        def noise_metadata(self):
            return {"stub": True}

    mc.MPPIParams = MPPIParams
    mc.MPPIController = MPPIController
    mc.NOMINAL_BLUEBOAT = {"physics_dt_s": 1.0 / 120.0, "decimation": 2}
    mc.NominalBlueBoatDynamics = _Dynamics
    mc.dynamics_for_cfg = lambda cfg, device=None: _Dynamics()
    mc.journal_mppi_params = lambda vessel, params: (params, {"stub": True})
    mc.PLANAR_DAMPING_KEYS = ()
    mc.profile_planar_damping = lambda profile: {}
    mc._shared_module = lambda name: types.SimpleNamespace(
        resolve_journal_vehicle=lambda vessel, **options: types.SimpleNamespace(hydro_profile=None))
    sys.modules["mppi_controller"] = mc

if os.environ.get("G6_WATCHDOG_S"):
    import evaluator_exit
    evaluator_exit.CLOSE_WATCHDOG_S = float(os.environ["G6_WATCHDOG_S"])
'''
STUBS = {
    "sitecustomize.py": SITECUSTOMIZE,
    "isaaclab/__init__.py": "",
    "isaaclab/app.py": APP_STUB,
    "isaaclab/envs/__init__.py": ("class DirectMARLEnv:\n    pass\n\n\nclass DirectMARLEnvCfg:\n    pass\n\n\n"
                                  "class DirectRLEnvCfg:\n    pass\n\n\nclass ManagerBasedRLEnvCfg:\n    pass\n\n\n"
                                  "def multi_agent_to_single_agent(env):\n    return env\n"),
    "isaaclab_rl/__init__.py": "",
    "isaaclab_rl/skrl.py": SKRL_WRAPPER_STUB,
    "isaaclab_tasks/__init__.py": "",
    "isaaclab_tasks/utils/__init__.py": TASK_UTILS_STUB,
    "isaaclab_tasks/utils/hydra.py": HYDRA_STUB,
    "gymnasium/__init__.py": GYM_STUB,
    "skrl/__init__.py": "import logging\nlogger = logging.getLogger('skrl-stub')\n",
    "skrl/models/__init__.py": "",
    "skrl/models/torch.py": "class GaussianMixin:\n    pass\n\n\nclass Model:\n    pass\n",
    "skrl/utils/__init__.py": "",
    "skrl/utils/spaces/__init__.py": "",
    "skrl/utils/spaces/torch.py": "def unflatten_tensorized_space(*args, **kwargs):\n    raise RuntimeError('stub')\n",
    "skrl/utils/runner/__init__.py": "",
    "skrl/utils/runner/torch.py": RUNNER_STUB,
}
COMMON = ["--num_envs", "2", "--episodes", "2", "--level", "0", "--vessel", "blueboat", "--device", "cuda:0",
          "--headless"]
ARGV = {   # failure runs: the journal ids of P5; success runs: see SUCCESS_TASK
    "classical_planner_pid.py": ["--task", "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1", "--eval-seed", "42"],
    "classical_baseline.py": ["--protocol", "mission", "--task", "Isaac-USV-PathFollow-BlueBoat-Journal-Direct-v1",
                              "--seed", "42"],
    "classical_harbour.py": ["--task", "Isaac-USV-HarborMission-BlueBoat-Journal-Direct-v1", "--eval-seed", "42"],
    "eval_mppi.py": ["--task", "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1", "--eval-seed", "42"],
    "eval_v6_frozen.py": ["--task", "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1", "--eval-seed", "42",
                          "--checkpoint", "<CHECKPOINT>"],
}
# Success runs keep the evaluator's journal branch where the stub writer suffices, else a legacy id:
# the certificate tail (guard write, publication, close) is the same code either way.
SUCCESS_TASK = {
    "classical_planner_pid.py": "Isaac-USV-HazardCross-Direct-v1",
    "classical_baseline.py": "Isaac-USV-StationKeep-BlueBoat-Direct-v1",
    "classical_harbour.py": "Isaac-USV-HarborMission-BlueBoat-Journal-Direct-v1",
    "eval_mppi.py": "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1",
    "eval_v6_frozen.py": "Isaac-USV-HazardCross-Direct-v1",
}
SCIENTIFIC_PREFIXES = ("USVBENCH_", "OBS_", "WAVE_", "SPAWN_")


def write_stubs(root):
    for relative, text in STUBS.items():
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="ascii")


def run_evaluator(scripts_dir, target, workdir, *, stub_mode="fail_cfg", close_mode="normal",
                  timeout=SUBPROCESS_TIMEOUT_S, argv_override=None):
    """Run one evaluator against the stubs; returns a dict (exit code, app marks, files, output, seconds)."""
    workdir = Path(workdir)
    stubs = workdir / "stubs"
    write_stubs(stubs)
    mark = workdir / (target + ".mark")
    out = workdir / (target + ".cert.json")
    checkpoint = workdir / "checkpoint.pt"
    checkpoint.write_bytes(b"stub")
    argv = [str(checkpoint) if a == "<CHECKPOINT>" else a for a in (argv_override or ARGV[target])]
    if stub_mode == "ok":
        argv[argv.index("--task") + 1] = SUCCESS_TASK[target]
    argv += COMMON + ["--out", str(out)]
    env = {k: v for k, v in os.environ.items() if not k.startswith(SCIENTIFIC_PREFIXES) and k != "HYDRA_FULL_ERROR"}
    env.update(PYTHONPATH=os.pathsep.join([str(stubs), str(scripts_dir)]), G6_APP_MARK=str(mark), G6_OUT=str(out),
               G6_STUB_MODE=stub_mode, G6_APP_CLOSE_MODE=close_mode, G6_WATCHDOG_S=str(TEST_WATCHDOG_S),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    start = time.monotonic()
    try:
        result = subprocess.run([sys.executable, "-B", str(Path(scripts_dir) / target)] + argv, cwd=str(workdir),
                                env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout)
        code, output = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        code, output = "HANG", str(exc.stdout) + str(exc.stderr)
    marks = mark.read_text(encoding="ascii").split() if mark.exists() else []
    files = sorted(p.name for p in workdir.glob(target + ".cert.json*"))
    return {"code": code, "marks": marks, "files": files, "output": output,
            "seconds": round(time.monotonic() - start, 1), "out": out}


@pytest.mark.parametrize("close_mode", CLOSE_MODES)
@pytest.mark.parametrize("target", EVALUATORS)
def test_success_publishes_before_close_and_exits_zero(target, close_mode):
    with tempfile.TemporaryDirectory(prefix="g6b_ok_") as workdir:
        r = run_evaluator(SCRIPTS, target, workdir, stub_mode="ok", close_mode=close_mode)
        document = json.loads(r["out"].read_text(encoding="utf-8")) if r["out"].exists() else None
    assert r["code"] == 0, (r["code"], r["marks"], r["output"][-3000:])
    assert r["files"] == [target + ".cert.json"], r["files"]                   # published, no partial left
    assert document is not None and document["header"] == {"stub": "g6b"}
    assert "close:cert-present" in r["marks"], r["marks"]                       # renamed BEFORE close()
    if close_mode == "block":
        assert "close-block" in r["marks"] and "did not return within CLOSE_WATCHDOG_S" in r["output"]


@pytest.mark.parametrize("close_mode", CLOSE_MODES)
@pytest.mark.parametrize("target", EVALUATORS)
def test_cfg_failure_exits_nonzero_without_certificate_or_close(target, close_mode):
    with tempfile.TemporaryDirectory(prefix="g6b_fail_") as workdir:
        r = run_evaluator(SCRIPTS, target, workdir, stub_mode="fail_cfg", close_mode=close_mode)
    assert "launch" in r["marks"], r["output"][-3000:]                         # preflight passed, app launched
    assert r["code"] == 1, (r["code"], r["marks"], r["output"][-3000:])
    assert not any(m.startswith("close") for m in r["marks"]), r["marks"]       # close() never called
    assert "shutdown-with-open-app" not in r["marks"]
    assert r["files"] == [], r["files"]
    expected = "FrozenInstanceError" if target == "classical_baseline.py" else P5_CFG_ERROR
    assert expected in r["output"], r["output"][-3000:]


@pytest.mark.parametrize("stub_mode", ("fail_env", "fail_hydra_sysexit"))
@pytest.mark.parametrize("close_mode", CLOSE_MODES)
def test_classical_baseline_failure_inside_the_hydra_main(stub_mode, close_mode):
    """P5 v3 finding 2: an exception inside the Hydra-decorated main (here the env build), both with
    Hydra re-raising it (HYDRA_FULL_ERROR=1, set by classical_baseline) and with Hydra's own
    print + sys.exit(1) path: exit 1, no file, no close(), no hang."""
    with tempfile.TemporaryDirectory(prefix="g6b_hydra_") as workdir:
        r = run_evaluator(SCRIPTS, "classical_baseline.py", workdir, stub_mode=stub_mode, close_mode=close_mode)
    assert r["code"] == 1, (r["code"], r["marks"], r["output"][-3000:])
    assert not any(m.startswith("close") for m in r["marks"]), r["marks"]
    assert r["files"] == []
    assert "stub gym.make: env build failed" in r["output"]
    assert ("Set the environment variable HYDRA_FULL_ERROR=1" in r["output"]) == (stub_mode == "fail_hydra_sysexit")


OFFICIAL = ["--eval-seed", "21001", "--seed-namespace", "dose_pilots", "--seed-block", "pilot-01",
            "--run-purpose", "pilot"]


@pytest.mark.parametrize("target", ("classical_planner_pid.py", "eval_mppi.py"))
def test_official_run_outside_re1_is_refused_before_the_app_starts(target):
    """CONTEXT_R4 D28 (b): this CPU box is not RE-1 (Python 3.13): an official (pilot) run exits 2
    in preflight, before AppLauncher (no 'launch' mark), with the reference-environment message."""
    with tempfile.TemporaryDirectory(prefix="g6b_re1_") as workdir:
        r = run_evaluator(SCRIPTS, target, workdir, stub_mode="ok",
                          argv_override=["--task", "Isaac-USV-HazardCross-Direct-v1"] + OFFICIAL)
    assert r["code"] == 2, (r["code"], r["output"][-2000:])
    assert r["marks"] == [] and r["files"] == []
    assert "outside the reference environment RE-1" in r["output"]
