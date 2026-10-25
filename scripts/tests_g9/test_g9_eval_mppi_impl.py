"""G9: eval_mppi selects, wires, stamps and fails closed on the MPPI implementation.

* CLI: --mppi-impl auto|eager|graph (default auto; literal copies equal to mppi_controller);
  every pre-G9 argument keeps its option string, destination, default and choices; graph with
  the legacy noise stream is refused before the app starts;
* resolution: auto -> cuda_graph_v1 for journal ids on a CUDA device, eager otherwise; eager ->
  eager; graph -> cuda_graph_v1 on CUDA, a nonzero exit elsewhere (never an eager fallback);
* wiring (source): the loop calls only the runner (rollout and reset) through the shared
  rollout_inputs reads; no exception handler in _run_isaac_eval other than ImportError; the
  stamp is written into the certificate header before cert.build; build_mppi_setup builds
  the frozen journal controller and the journal bridge;
* the REAL eval_mppi in a subprocess against stub isaaclab / isaaclab_tasks / gymnasium /
  cert_v3 / journal_controllers / mppi_controller packages (one control step, quota 1):
  graph by default on a CUDA env device (stamp cuda_graph_v1 in the published certificate),
  --mppi-impl eager on CUDA (eager stamp, no graph built), eager by default on a CPU device,
  a failed capture and --mppi-impl graph on a CPU device exit 1 with the reason, no
  certificate and no app close (G6b failure path).
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g9/test_g9_eval_mppi_impl.py -q
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import g9_fixtures as fx  # noqa: E402
from g9_fixtures import mc, torch  # noqa: E402

SCRIPTS = fx.SCRIPTS
sys.path.insert(0, str(SCRIPTS / "tests_journal_glue"))
import eval_mppi  # noqa: E402

SUBPROCESS_TIMEOUT_S = 180

# eval_mppi._build_parser() of T-base v2.2 (before G9): (option strings, dest, default, choices).
PRE_G9_ARGUMENTS = (
    (("--task",), "task", "Isaac-USV-PathFollow-BlueBoat-Direct-v1", None),
    (("--episodes",), "episodes", 128, None),
    (("--num_envs",), "num_envs", 64, None),
    (("--level",), "level", 0, None),
    (("--eval-seed",), "eval_seed", 42, None),
    (("--out",), "out", None, None),
    (("--K",), "K", 256, None),
    (("--H",), "H", 60, None),
    (("--sigma-thrust",), "sigma_thrust", 0.5, None),
    (("--sigma-yaw",), "sigma_yaw", 0.8, None),
    (("--lambda",), "lambda_", 0.3, None),
    (("--w-goal",), "w_goal", 1.0, None),
    (("--w-gate",), "w_gate", 50.0, None),
    (("--w-obstacle",), "w_obstacle", 40.0, None),
    (("--clearance-margin",), "clearance_margin", 0.6, None),
    (("--collision-cost",), "collision_cost", 2000.0, None),
    (("--w-effort",), "w_effort", 0.02, None),
    (("--w-terminal",), "w_terminal", 20.0, None),
    (("--w-speed-hold",), "w_speed_hold", 0.6, None),
    (("--w-dock-heading",), "w_dock_heading", 0.15, None),
    (("--w-dock-speed",), "w_dock_speed", 1.64, None),
    (("--mppi-seed",), "mppi_seed", 0, None),
    (("--mppi-noise-backend",), "mppi_noise_backend", "splitmix64-boxmuller-f64-v2",
     ("splitmix64-boxmuller-f64-v2", "legacy-shared-generator-v1")),
    (("--set",), "extra_sets", [], None),
)


def _arguments(parser):
    return [(tuple(a.option_strings), a.dest, a.default, None if a.choices is None else tuple(a.choices))
            for a in parser._actions if a.dest != "help"]


# ------------------------------------------------------------------ CLI and resolution
def test_existing_arguments_unchanged_and_one_new_option():
    current = _arguments(eval_mppi._build_parser())
    new = [row for row in current if row not in PRE_G9_ARGUMENTS]
    assert [row for row in PRE_G9_ARGUMENTS if row not in current] == []
    assert new == [(("--mppi-impl",), "mppi_impl", "auto", ("auto", "eager", "graph"))]


def test_literal_copies_equal_the_controller_constants():
    assert eval_mppi.MPPI_IMPL_CHOICES == mc.MPPI_IMPL_CHOICES
    assert (eval_mppi.MPPI_IMPL_EAGER, eval_mppi.MPPI_IMPL_GRAPH) == (mc.MPPI_IMPL_EAGER, mc.MPPI_IMPL_GRAPH)
    import validate_cert_v3
    assert validate_cert_v3.MPPI_IMPLEMENTATIONS == mc.MPPI_IMPLEMENTATIONS
    assert validate_cert_v3.MPPI_LEGACY_NOISE_BACKEND == mc.LEGACY_NOISE_BACKEND


@pytest.mark.parametrize("requested,device,journal,expected", [
    ("auto", "cuda:0", True, "cuda_graph_v1"),
    ("auto", "cuda", True, "cuda_graph_v1"),
    ("auto", "cuda:0", False, "eager"),
    ("auto", "cpu", True, "eager"),
    ("auto", "cpu", False, "eager"),
    ("eager", "cuda:0", True, "eager"),
    ("eager", "cpu", False, "eager"),
    ("graph", "cuda:0", True, "cuda_graph_v1"),
    ("graph", "cuda:0", False, "cuda_graph_v1"),
])
def test_resolution(requested, device, journal, expected):
    assert eval_mppi.resolve_mppi_impl(requested, device, journal) == expected


@pytest.mark.parametrize("requested,device", [("graph", "cpu"), ("graph", torch.device("cpu")), ("fast", "cuda:0")])
def test_resolution_refusals_end_the_run(requested, device):
    with pytest.raises(SystemExit) as raised:
        eval_mppi.resolve_mppi_impl(requested, device, True)
    assert raised.value.code and isinstance(raised.value.code, str)
    if requested == "graph":
        assert "needs a CUDA device" in raised.value.code and "never falls back" in raised.value.code


def test_graph_with_the_legacy_noise_stream_is_refused_before_launch():
    parser = eval_mppi._build_parser()
    args = parser.parse_args(["--mppi-impl", "graph", "--mppi-noise-backend", "legacy-shared-generator-v1"])
    with pytest.raises(SystemExit) as raised:
        eval_mppi._validate_args(parser, args)
    assert raised.value.code == 2
    for impl in ("auto", "eager"):
        ok = parser.parse_args(["--mppi-impl", impl, "--mppi-noise-backend", "legacy-shared-generator-v1"])
        eval_mppi._validate_args(parser, ok)          # diagnostic legacy reproduction stays eager


def test_runner_and_stamp_helpers():
    dynamics, params = fx.frozen("blueboat_v2", K=16)
    controller = mc.MPPIController(num_envs=2, params=params, dynamics=dynamics)
    assert eval_mppi._mppi_runner(controller, "eager") is controller
    assert eval_mppi.mppi_implementation_stamp(controller, "eager") == mc.eager_implementation_stamp()
    graph = eval_mppi._mppi_runner(controller, "cuda_graph_v1", backend=fx.ReplayBackend())
    assert isinstance(graph, mc.CudaGraphMPPI) and graph.controller is controller
    fx.call(graph, fx.make_inputs("goal", 2, 0))
    assert eval_mppi.mppi_implementation_stamp(graph, "cuda_graph_v1") == {
        "mode": "cuda_graph_v1", "captured_graphs": 1, "recaptures": 0}
    with pytest.raises(SystemExit):
        eval_mppi.mppi_implementation_stamp(graph, "eager_v2")
    with pytest.raises(SystemExit):
        eval_mppi._mppi_runner(controller, "eager_v2")


class _Bridge:
    """Records the bridge reads in order."""

    def __init__(self):
        self.calls = []
        self.goal_radius, self.gate_half_width_m, self.gate_min_normal_speed_mps = 2.5, 3.0, 0.2

    def _r(self, name, value):
        self.calls.append(name)
        return value

    def state(self):
        return self._r("state", "S")

    def targets(self):
        return self._r("targets", ("T", "I"))

    def obstacles(self):
        return self._r("obstacles", ("C", "R", "A"))

    def dock_heading(self):
        return self._r("dock_heading", "D")

    def gate_normals(self):
        return self._r("gate_normals", "G")


@pytest.mark.parametrize("mode", fx.MODES)
def test_rollout_inputs_are_the_former_loop_reads(mode):
    bridge = _Bridge()
    state, kwargs = eval_mppi.rollout_inputs(bridge, mode)
    expected = {"targets": "T", "active_index": "I", "goal_radius": 2.5, "mode": mode, "obstacle_centers": "C",
                "obstacle_radii": "R", "obstacle_active": "A"}
    order = ["state", "targets", "obstacles"]
    if mode in ("dock", "mission"):
        expected["dock_heading"] = "D"
        order.append("dock_heading")
    if mode == "mission":
        expected.update(gate_normals="G", gate_half_width_m=3.0, gate_min_normal_speed_mps=0.2)
        order.append("gate_normals")
    assert state == "S" and kwargs == expected and bridge.calls == order


# ------------------------------------------------------------------ wiring (source)
def _function(name):
    tree = ast.parse((SCRIPTS / "eval_mppi.py").read_text(encoding="ascii"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node):
    out = []
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            out.append(func.value.id + "." + func.attr)
        elif isinstance(func, ast.Name):
            out.append(func.id)
    return out


def test_loop_calls_only_the_runner_and_never_falls_back():
    body = _function("_run_isaac_eval")
    calls = _calls(body)
    assert calls.count("runner.rollout") == 1 and calls.count("runner.reset") == 1
    assert "controller.rollout" not in calls and "controller.reset" not in calls
    assert calls.count("rollout_inputs") == 1 and calls.count("build_mppi_setup") == 1
    assert calls.count("resolve_mppi_impl") == 1 and calls.count("_mppi_runner") == 1
    handlers = [h for h in ast.walk(body) if isinstance(h, ast.ExceptHandler)]
    assert handlers and all(isinstance(h.type, ast.Name) and h.type.id == "ImportError" for h in handlers)


def test_stamp_is_written_into_the_header_before_the_certificate_is_built():
    source = (SCRIPTS / "eval_mppi.py").read_text(encoding="ascii")
    stamp = source.index('cert.header["controller"]["configuration"]["mppi_implementation"] = implementation_stamp')
    assert source.index("implementation_stamp = mppi_implementation_stamp(runner, implementation)") < stamp
    assert stamp < source.index("cert.build({")


def test_build_mppi_setup_builds_the_frozen_journal_controller():
    import glue_fixtures as gf

    cfg = gf.journal_cfg("blueboat_v2", goal_radius=2.0, ray_max_range_m=30.0)
    base = SimpleNamespace(cfg=cfg, device="cpu", num_envs=2, control_step_s=1.0 / 60.0)
    args = eval_mppi._build_parser().parse_args([])
    journal = {"journal_task": "reach", "vehicle": "blueboat_v2"}
    setup = eval_mppi.build_mppi_setup(args, cfg, base, "goal", journal, torch, mc.MPPIController,
                                       mc.MPPIParams, mc.NOMINAL_BLUEBOAT, mc.NominalBlueBoatDynamics)
    expected, stamp = mc.journal_mppi_params("blueboat_v2", mc.MPPIParams())
    assert setup.params == expected and setup.frozen_parameters == stamp
    assert isinstance(setup.controller, mc.MPPIController) and setup.controller.params is setup.params
    assert isinstance(setup.dynamics, mc.VesselDynamicsV2) and setup.controller.dynamics is setup.dynamics
    assert type(setup.bridge) is eval_mppi.JournalEnvBridge and setup.bridge.max_range_m == 30.0
    assert setup.reference_params == mc.MPPIParams().as_dict() and setup.jc is not None
    assert setup.journal_notes == []
    bad = SimpleNamespace(cfg=cfg, device="cpu", num_envs=2, control_step_s=1.0 / 30.0)
    with pytest.raises(SystemExit, match="control_step_s"):
        eval_mppi.build_mppi_setup(args, cfg, bad, "goal", journal, torch, mc.MPPIController, mc.MPPIParams,
                                   mc.NOMINAL_BLUEBOAT, mc.NominalBlueBoatDynamics)


# ------------------------------------------------------------------ the real evaluator, stubbed simulator
APP_STUB = '''"""G9 test stand-in for isaaclab.app (no Kit)."""
import atexit
import os


def _mark(text):
    with open(os.environ["G9_MARK"], "a", encoding="ascii") as stream:
        stream.write(text + "\\n")


class _App:
    def __init__(self):
        self._closed = False
        atexit.register(self._shutdown_with_open_app)

    def _shutdown_with_open_app(self):
        if not self._closed:
            _mark("shutdown-with-open-app")
            os._exit(0)

    def close(self):
        out = os.environ.get("G9_OUT")
        _mark("close:cert-present" if out and os.path.exists(out) else "close:cert-absent")
        self._closed = True


class AppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", type=str, default=None)

    def __init__(self, args):
        _mark("launch")
        self.app = _App()
'''
TASK_UTILS_STUB = '''
from types import SimpleNamespace


def parse_env_cfg(task_name, device="cuda:0", num_envs=None, use_fabric=None):
    return SimpleNamespace(seed=None, scene=SimpleNamespace(num_envs=2), decimation=2, vehicle="blueboat_v2",
                           sim=SimpleNamespace(device="cuda:0", dt=1.0 / 120.0), goal_radius=2.0,
                           conditions=None, condition_presets=())
'''
GYM_STUB = '''
import os
import types

import torch

registry = {}


def register(id, **kwargs):
    registry[id] = kwargs


class _Space:
    shape = (2, 2)


class FakeEnv:
    """A goal-family env with a journal-like device; every step finishes every env."""

    def __init__(self, cfg):
        n = 2
        self.cfg = cfg
        self.unwrapped = self
        self.num_envs = n
        self.device = os.environ["G9_ENV_DEVICE"]
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
    return FakeEnv(cfg)


spaces = types.SimpleNamespace(Box=object, Space=object)
'''
SITECUSTOMIZE = '''
import contextlib
import os
import sys
import types

if os.environ.get("G9_STUB") == "1":
    import torch

    def _mark(text):
        with open(os.environ["G9_MARK"], "a", encoding="ascii") as stream:
            stream.write(text + "\\n")

    class CertificateRun:
        quota = 1

        def __init__(self, **kwargs):
            self.header = {"controller": {"name": kwargs["controller"],
                                          "configuration": {"device": kwargs["controller_config"]["device"]}}}

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
            return {"header": self.header, "records": payload.get("records", [])}

    cert = types.ModuleType("cert_v3")
    cert.CertificateRun = CertificateRun
    sys.modules["cert_v3"] = cert

    jc = types.ModuleType("journal_controllers")
    jc.INFORMATION_DIET = "stub"
    jc.ray_max_range = lambda cfg: 30.0
    jc.read_journal_inputs = lambda env: types.SimpleNamespace(
        position_world=torch.zeros((2, 2)), yaw=torch.zeros(2), velocity=torch.zeros((2, 3)),
        rays=torch.full((2, 36), 30.0))
    jc.ray_obstacles = lambda position, yaw, rays, max_range: (
        torch.zeros((2, 36, 2)), torch.zeros((2, 36)), torch.zeros((2, 36), dtype=torch.bool))
    sys.modules["journal_controllers"] = jc

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
        params = {"model": "stub", "physics_dt_s": 1.0 / 120.0, "decimation": 2, "swayyaw_damping": "stub",
                  "surge_lin": 0.0, "surge_quad": 0.0, "sway_lin": 0.0, "sway_quad": 0.0, "yaw_lin": 0.0,
                  "yaw_quad": 0.0}

    class MPPIController:
        dtype = torch.float32

        def __init__(self, **kwargs):
            self.num_envs = kwargs["num_envs"]

        def noise_metadata(self):
            return {"backend": "splitmix64-boxmuller-f64-v2"}

        def rollout(self, state, **kwargs):
            _mark("eager-rollout")
            return torch.zeros((self.num_envs, 2)), {"ess": torch.ones(self.num_envs)}

        def reset(self, done, episode_ids=None):
            _mark("eager-reset")

    class GraphCaptureError(RuntimeError):
        pass

    class CudaGraphMPPI:
        implementation = "cuda_graph_v1"

        def __init__(self, controller, backend=None, log=None):
            _mark("graph-constructed")
            self.controller, self.log, self.captured_graphs = controller, log, 0

        def rollout(self, state, **kwargs):
            if os.environ.get("G9_GRAPH") == "fail":
                message = "CUDA-graph MPPI capture failed (stub); graph mode never falls back to the eager rollout"
                self.log("[MPPI-GRAPH] " + message)
                raise GraphCaptureError(message)
            _mark("graph-rollout")
            self.captured_graphs = 1
            return torch.zeros((self.controller.num_envs, 2)), {"ess": torch.ones(self.controller.num_envs)}

        def reset(self, done, episode_ids=None):
            _mark("graph-reset")

        def implementation_stamp(self):
            return {"mode": "cuda_graph_v1", "captured_graphs": self.captured_graphs, "recaptures": 0}

    mc.MPPIParams, mc.MPPIController, mc.CudaGraphMPPI = MPPIParams, MPPIController, CudaGraphMPPI
    mc.GraphCaptureError = GraphCaptureError
    mc.NOMINAL_BLUEBOAT = {"physics_dt_s": 1.0 / 120.0, "decimation": 2}
    mc.NominalBlueBoatDynamics = _Dynamics
    mc.dynamics_for_cfg = lambda cfg, device=None: _Dynamics()
    mc.journal_mppi_params = lambda vessel, params: (params, {"stub": True})
    mc.PLANAR_DAMPING_KEYS = ()
    mc.profile_planar_damping = lambda profile: {}
    mc._shared_module = lambda name: types.SimpleNamespace(
        resolve_journal_vehicle=lambda vessel, **options: types.SimpleNamespace(hydro_profile=None))
    sys.modules["mppi_controller"] = mc
'''
STUBS = {"sitecustomize.py": SITECUSTOMIZE, "isaaclab/__init__.py": "", "isaaclab/app.py": APP_STUB,
         "isaaclab_tasks/__init__.py": "", "isaaclab_tasks/utils/__init__.py": TASK_UTILS_STUB,
         "gymnasium/__init__.py": GYM_STUB}
JOURNAL_TASK = "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1"


def run_eval(device, extra=(), graph="ok"):
    with tempfile.TemporaryDirectory(prefix="g9_eval_") as workdir:
        workdir = Path(workdir)
        for relative, text in STUBS.items():
            path = workdir / "stubs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="ascii")
        mark, out = workdir / "marks.txt", workdir / "cert.json"
        argv = ["--task", JOURNAL_TASK, "--eval-seed", "42", "--num_envs", "2", "--episodes", "2", "--level", "0",
                "--vessel", "blueboat", "--headless", "--out", str(out), *extra]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("USVBENCH_", "OBS_", "WAVE_", "SPAWN_"))}
        env.update(PYTHONPATH=os.pathsep.join([str(workdir / "stubs"), str(SCRIPTS)]), G9_STUB="1",
                   G9_MARK=str(mark), G9_OUT=str(out), G9_ENV_DEVICE=device, G9_GRAPH=graph,
                   PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "eval_mppi.py"), *argv], cwd=str(workdir),
                                env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=SUBPROCESS_TIMEOUT_S)
        marks = mark.read_text(encoding="ascii").split() if mark.exists() else []
        files = sorted(p.name for p in workdir.glob("cert.json*"))
        document = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return {"code": result.returncode, "output": result.stdout + result.stderr, "marks": marks, "files": files,
            "document": document}


def _stamp(run):
    return run["document"]["header"]["controller"]["configuration"]["mppi_implementation"]


def test_subprocess_journal_on_cuda_defaults_to_the_graph_and_stamps_it():
    run = run_eval("cuda:0")
    assert run["code"] == 0, run["output"][-3000:]
    assert run["files"] == ["cert.json"] and "close:cert-present" in run["marks"]
    assert "graph-constructed" in run["marks"] and "graph-rollout" in run["marks"]
    assert "eager-rollout" not in run["marks"] and "graph-reset" in run["marks"]
    assert _stamp(run) == {"mode": "cuda_graph_v1", "captured_graphs": 1, "recaptures": 0}
    assert "[MPPI-IMPL] requested=auto resolved=cuda_graph_v1 device=cuda:0" in run["output"]


def test_subprocess_eager_option_forces_eager_on_cuda():
    run = run_eval("cuda:0", ["--mppi-impl", "eager"])
    assert run["code"] == 0, run["output"][-3000:]
    assert "graph-constructed" not in run["marks"] and "eager-rollout" in run["marks"]
    assert _stamp(run) == {"mode": "eager", "captured_graphs": 0, "recaptures": 0}


def test_subprocess_cpu_device_defaults_to_eager():
    run = run_eval("cpu")
    assert run["code"] == 0, run["output"][-3000:]
    assert "graph-constructed" not in run["marks"]
    assert _stamp(run) == {"mode": "eager", "captured_graphs": 0, "recaptures": 0}


def test_subprocess_failed_capture_exits_nonzero_without_certificate_or_close():
    run = run_eval("cuda:0", graph="fail")
    assert run["code"] == 1, run["output"][-3000:]
    assert run["files"] == [] and run["document"] is None
    assert "launch" in run["marks"] and not any(m.startswith("close") for m in run["marks"])
    assert "shutdown-with-open-app" not in run["marks"] and "eager-rollout" not in run["marks"]
    assert "[MPPI-GRAPH] CUDA-graph MPPI capture failed" in run["output"]
    assert "GraphCaptureError" in run["output"] and "no certificate written" in run["output"]


def test_subprocess_graph_on_a_cpu_device_exits_nonzero_without_certificate_or_close():
    run = run_eval("cpu", ["--mppi-impl", "graph"])
    assert run["code"] == 1, run["output"][-3000:]
    assert run["files"] == [] and not any(m.startswith("close") for m in run["marks"])
    assert "graph-constructed" not in run["marks"] and "eager-rollout" not in run["marks"]
    assert "--mppi-impl graph needs a CUDA device" in run["output"]


def test_subprocess_graph_with_legacy_noise_is_refused_before_launch():
    run = run_eval("cuda:0", ["--mppi-impl", "graph", "--mppi-noise-backend", "legacy-shared-generator-v1"])
    assert run["code"] == 2 and run["marks"] == [] and run["files"] == []
    assert "replays the keyed noise stream only" in run["output"]
