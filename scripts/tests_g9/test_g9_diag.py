"""G9: the GPU equivalence diagnostic (scripts/diag_mppi_graph.py, scripts/diag_mppi_graph.ps1) on CPU.

* argument refusals (report outside the release root and never overwritten, steps, tolerance,
  first-k, no --mppi-impl), the comparison helpers (inf / nan aware), the summary and its
  verdict, the atomic report writer, the release-module check;
* side_by_side with REAL controllers (frozen journal BlueBoat parameters, the release's
  CudaGraphMPPI on the replay stand-in backend) and a fake journal env: EQUIVALENT, bit-identical,
  keys equal, resets on both controllers; a perturbed graph and a key mismatch are DIFFERENT;
* the REAL diag_mppi_graph.py in a subprocess against stub isaaclab / isaaclab_tasks / gymnasium
  (journal-like cfg and env), the real eval_protocol preflight, eval_mppi.build_mppi_setup, the
  real VesselDynamicsV2 / journal_mppi_params / MPPIController / CudaGraphMPPI (CPU replay class
  patched in for CudaGraphBackend): exit 0 + report; a perturbed graph exits 3 with the report
  and without closing the app; a failed capture exits 1 without report or close; refusals exit 2
  before the app starts;
* source: the diagnostic builds the controller through eval_mppi, never writes a certificate;
  the PowerShell grid parses, is pure ASCII, names every timeout with its reason, refuses an
  output folder inside the release and writes nothing with -DescribeOnly.
Run: python -B -m pytest -p no:cacheprovider scripts/tests_g9/test_g9_diag.py -q
"""
from __future__ import annotations

import ast
import json
import math
import os
from pathlib import Path
import re
import shutil
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
REPO = SCRIPTS.parent
import diag_mppi_graph as diag  # noqa: E402

SUBPROCESS_TIMEOUT_S = 300
PS1 = SCRIPTS / "diag_mppi_graph.ps1"


# ------------------------------------------------------------------ unit
def _args(tmp_path, **kw):
    values = dict(steps=10, tolerance=1e-6, first_k=2, mppi_impl="auto", out=str(tmp_path / "r.json"))
    values.update(kw)
    return SimpleNamespace(**values)


def test_argument_refusals(tmp_path):
    assert diag.validate_diag_args(_args(tmp_path), REPO) == Path(os.path.realpath(tmp_path / "r.json"))
    for kw, text in ((dict(steps=0), "--steps"), (dict(tolerance=0.0), "--tolerance"),
                     (dict(tolerance=float("nan")), "--tolerance"), (dict(first_k=10), "--first-k"),
                     (dict(mppi_impl="eager"), "both implementations"), (dict(out=None), "required"),
                     (dict(out=str(REPO / "scripts" / "r.json")), "outside the release root"),
                     (dict(out=str(tmp_path / "missing" / "r.json")), "does not exist")):
        with pytest.raises(diag.DiagError, match=text):
            diag.validate_diag_args(_args(tmp_path, **kw), REPO)
    (tmp_path / "r.json").write_text("{}", encoding="ascii")
    with pytest.raises(diag.DiagError, match="never overwrites"):
        diag.validate_diag_args(_args(tmp_path), REPO)


def test_tensor_difference_is_inf_and_nan_aware():
    inf, nan = math.inf, math.nan
    same = diag.tensor_difference(torch, torch.tensor([1.0, inf, nan]), torch.tensor([1.0, inf, nan]))
    assert same == {"max": 0.0, "mean": 0.0, "bitwise": True}
    differ = diag.tensor_difference(torch, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]))
    assert differ["max"] == 0.5 and differ["mean"] == 0.25 and differ["bitwise"] is False
    assert diag.tensor_difference(torch, torch.tensor([inf]), torch.tensor([1.0]))["max"] == inf
    assert diag.tensor_difference(torch, torch.zeros(2), torch.zeros(3))["max"] == inf


def test_summary_and_verdict():
    row = {"action_max": 0.0, "action_mean": 0.0, "action_bitwise": True, "info_max": 0.0, "warm_start_max": 0.0,
           "actions_finite": True, "keys_match": True, "resets": 0, "env_s": 0.1}
    rows = [dict(row, step=i, eager_s=0.36 + 0.001 * i, graph_s=0.03) for i in range(6)]
    rows[0]["graph_s"] = 4.0                                          # the capture step
    summary = diag.summarize(rows, 1e-6, 1)
    assert summary["verdict"] == "EQUIVALENT" and summary["failed_checks"] == []
    assert summary["bit_identical_steps"] == 6 and summary["graph_first_call_s"] == 4.0
    assert summary["eager_call_s"]["median"] == pytest.approx(0.363) and summary["graph_call_s"]["median"] == 0.03
    assert summary["speedup_controller_call"] == pytest.approx(0.363 / 0.03)
    import cert_v3
    assert summary["eager_call_s"]["p95"] == pytest.approx(cert_v3.timing_stats([r["eager_s"] for r in rows[1:]])["p95"])
    for field, value, check in (("action_max", 2e-6, "steps_over_tolerance"), ("keys_match", False, "key_mismatch_steps"),
                                ("info_max", 1.0, "info_steps_over_tolerance"),
                                ("warm_start_max", 1.0, "warm_start_steps_over_tolerance"),
                                ("actions_finite", False, "nonfinite_action_steps"), ("action_max", math.nan,
                                                                                      "steps_over_tolerance")):
        bad = [dict(r) for r in rows]
        bad[3][field] = value
        result = diag.summarize(bad, 1e-6, 1)
        assert result["verdict"] == "DIFFERENT" and check in result["failed_checks"], (field, result)


def test_report_writer_is_atomic_and_strict(tmp_path):
    out = tmp_path / "r.json"
    diag.write_report(out, diag.json_safe({"a": math.inf, "b": [1.0, -math.inf], "c": {"d": math.nan}}))
    assert json.loads(out.read_text(encoding="ascii")) == {"a": "inf", "b": [1.0, "-inf"], "c": {"d": "nan"}}
    assert not Path(str(out) + ".partial").exists()
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "x.json", {"a": math.inf})


def test_release_module_check():
    assert diag.release_module_check(mc, SCRIPTS).endswith("mppi_controller.py")
    with pytest.raises(RuntimeError, match="not from the release scripts folder"):
        diag.release_module_check(mc, REPO / "tasks")
    with pytest.raises(RuntimeError, match="not a G9 release"):
        diag.release_module_check(SimpleNamespace(__file__=str(SCRIPTS / "mppi_controller.py")), SCRIPTS)


class _JournalBridge:
    """eval_mppi.rollout_inputs-compatible bridge over the fake env (goal mode)."""

    goal_radius, gate_half_width_m, gate_min_normal_speed_mps = 2.0, 0.0, 0.0

    def __init__(self, env):
        self.env = env

    def state(self):
        e = self.env
        return torch.stack((e.pos[:, 0], e.pos[:, 1], e.yaw, e.vel[:, 0], e.vel[:, 1], e.vel[:, 2]), dim=-1)

    def targets(self):
        return self.env.target[:, None, :], None

    def obstacles(self):
        import journal_controllers as jc
        return jc.ray_obstacles(self.env.pos, self.env.yaw, self.env.rays, 30.0)


class _FakeEnv:
    def __init__(self, n=2, done_at=(2,)):
        g = torch.Generator().manual_seed(4)
        self.n, self.t, self.done_at = n, 0, done_at
        self.pos = torch.randn((n, 2), generator=g) * 3.0
        self.yaw = torch.rand((n,), generator=g)
        self.vel = torch.zeros((n, 3))
        self.target = torch.full((n, 2), 15.0)
        self.rays = torch.full((n, 36), 30.0)
        self.rays[:, :5] = 6.0
        self.actions = []

    def step(self, actions):
        self.actions.append(actions.clone())
        self.pos = self.pos + 0.1 * torch.stack((actions[:, 0], actions[:, 1]), dim=-1)
        self.yaw = self.yaw + 0.05 * actions[:, 1]
        self.vel = torch.cat((actions, actions[:, :1]), dim=-1) * 0.2
        term = torch.zeros(self.n, dtype=torch.bool)
        if self.t in self.done_at:
            term[0] = True
        self.t += 1
        return {}, torch.zeros(self.n), term, torch.zeros(self.n, dtype=torch.bool), {}


def _loop(steps=5, perturb=0.0, break_keys=False):
    dynamics, params = fx.frozen("blueboat_v2")
    eager, inner = fx.controllers(2, params, dynamics)
    graph = mc.CudaGraphMPPI(inner, backend=fx.ReplayBackend())
    if perturb or break_keys:
        rollout = graph.rollout

        def tampered(state, **kwargs):
            action, info = rollout(state, **kwargs)
            if break_keys:
                graph.last_keys = list(reversed(graph.last_keys))
            return action + perturb, info
        graph.rollout = tampered
    env = _FakeEnv()
    ticks = iter(range(10000))
    rows = diag.side_by_side(env, _JournalBridge(env), "goal", eager, graph, steps, torch, lambda: None,
                             lambda: next(ticks) * 0.01)
    return rows, env, eager, inner, graph


def test_side_by_side_real_controllers_equivalent_and_eager_drives_the_env():
    rows, env, eager, inner, graph = _loop()
    summary = diag.summarize(rows, 1e-6, 1)
    assert summary["verdict"] == "EQUIVALENT", summary
    assert summary["bit_identical_steps"] == 5 and summary["max_abs_action_diff"] == 0.0
    assert summary["key_mismatch_steps"] == 0 and summary["env_resets"] == 1
    assert eager.episode_ids == inner.episode_ids == [1, 0]
    assert len(env.actions) == 5 and graph.captured_graphs == 1


def test_side_by_side_detects_a_difference_and_a_key_mismatch():
    rows, *_ = _loop(steps=3, perturb=1e-5)
    summary = diag.summarize(rows, 1e-6, 1)
    assert summary["verdict"] == "DIFFERENT" and summary["steps_over_tolerance"] == 3
    rows, *_ = _loop(steps=3, break_keys=True)
    summary = diag.summarize(rows, 1e-6, 1)
    assert summary["verdict"] == "DIFFERENT" and summary["key_mismatch_steps"] == 3


# ------------------------------------------------------------------ source
def test_source_builds_through_eval_mppi_and_writes_no_certificate():
    source = (SCRIPTS / "diag_mppi_graph.py").read_text(encoding="ascii")
    tree = ast.parse(source)
    calls = set()
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            calls.add(f.value.id + "." + f.attr)
        elif isinstance(f, ast.Name):
            calls.add(f.id)
    for name in ("eval_mppi.build_mppi_setup", "eval_mppi._detect_mode", "eval_mppi._validate_args",
                 "eval_mppi._build_parser", "preflight_protocol", "register_journal_tasks", "parse_env_cfg",
                 "resolve_protocol", "mc.CudaGraphMPPI", "mc.MPPIController", "release_module_check",
                 "code_tree.verify_loaded_modules", "evaluator_exit.run"):
        assert name in calls, name
    assert "eval_mppi.rollout_inputs" in source and 'parse_env_cfg(task, device="cuda:0"' in source
    assert "CertificateRun" not in source and "import cert_v3" not in source and "from cert_v3" not in source
    assert "evaluator_exit.run(_run_diag, _GUARD)" in source
    assert all(b == 10 or 32 <= b <= 126 for b in source.encode("ascii"))


def _ps_available():
    return shutil.which("powershell") is not None


def test_powershell_grid_is_ascii_with_named_timeouts():
    data = PS1.read_bytes()
    assert all(b in (10, 13) or 32 <= b <= 126 for b in data)
    text = data.decode("ascii")
    names = re.findall(r"^\$(DIAG_[A-Z_]+_S(?:_[A-Z]+)?) = ", text, flags=re.M)
    assert set(names) == {"DIAG_STARTUP_BUDGET_S", "DIAG_CAPTURE_BUDGET_S", "DIAG_STEP_BUDGET_S_BLUEBOAT",
                          "DIAG_STEP_BUDGET_S_WAMV", "DIAG_GRID_CEILING_S"}
    for name in names:
        assert re.search(r"^# %s: \S" % name, text, flags=re.M), name     # each one with its reason
    assert "WaitForExit([int]($row.timeout_s * 1000))" in text and "taskkill.exe /PID $proc.Id /T /F" in text


@pytest.mark.skipif(not _ps_available(), reason="Windows PowerShell not available")
def test_powershell_grid_parses_refuses_and_describe_only_writes_nothing():
    parse = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "$e=$null; $t=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
                            "$env:G9_PS1, [ref]$t, [ref]$e); $e.Count"],
                           env=dict(os.environ, G9_PS1=str(PS1)), capture_output=True, text=True, timeout=120)
    assert parse.stdout.strip() == "0", parse.stdout + parse.stderr
    with tempfile.TemporaryDirectory(prefix="g9_ps_") as workdir:
        out = Path(workdir) / "grid"
        base = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(PS1),
                "-Python", sys.executable, "-ReleaseRoot", str(REPO)]
        described = subprocess.run(base + ["-OutDir", str(out), "-NumEnvs", "2,16", "-DescribeOnly"],
                                   capture_output=True, text=True, timeout=120)
        assert described.returncode == 0, described.stdout + described.stderr
        assert "rows=4" in described.stdout and described.stdout.count("PLAN ") == 4
        assert "timeout=1560s (DIAG_STARTUP_BUDGET_S + DIAG_CAPTURE_BUDGET_S + 300 x DIAG_STEP_BUDGET_S_BLUEBOAT)" \
            in described.stdout
        assert "timeout=3300s (DIAG_STARTUP_BUDGET_S + DIAG_CAPTURE_BUDGET_S + 300 x DIAG_STEP_BUDGET_S_WAMV)" \
            in described.stdout
        assert "nothing was written" in described.stdout and not out.exists()
        inside = subprocess.run(base + ["-OutDir", str(REPO / "g9_out"), "-DescribeOnly"], capture_output=True,
                                text=True, timeout=120)
        assert inside.returncode == 2 and "outside the release root" in inside.stdout
        assert not (REPO / "g9_out").exists()
        bad = subprocess.run(base + ["-OutDir", str(out), "-NumEnvs", "2,x", "-DescribeOnly"], capture_output=True,
                             text=True, timeout=120)
        assert bad.returncode == 2 and "NumEnvs entries must be integers" in bad.stdout and not out.exists()


# ------------------------------------------------------------------ the real diagnostic, stubbed simulator
APP_STUB = '''
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
        _mark("close")
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
    return SimpleNamespace(seed=42, scene=SimpleNamespace(num_envs=num_envs), decimation=2, vehicle="blueboat_v2",
                           journal_contract_version=2, journal_vehicle_options={}, journal_conditions=None,
                           underwater_physics_cfg=SimpleNamespace(physics_version=2, hydrodynamics_model="hydro_v2"),
                           sim=SimpleNamespace(device="cpu", dt=1.0 / 120.0), goal_radius=2.0, ray_max_range_m=30.0)
'''
GYM_STUB = '''
import types

import torch

registry = {}


def register(id, **kwargs):
    registry[id] = kwargs


class JournalFakeEnv:
    def __init__(self, cfg):
        n = cfg.scene.num_envs
        self.cfg, self.unwrapped, self.num_envs, self.device = cfg, self, n, "cpu"
        self.control_step_s, self.max_episode_length, self.t = 1.0 / 60.0, 100, 0
        self.target_pos = torch.full((n, 2), 12.0)
        self.scene = types.SimpleNamespace(env_origins=torch.zeros((n, 3)))
        pose = torch.zeros((n, 7))
        pose[:, 3] = 1.0
        self.robot = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pose_w=pose))
        rays = torch.full((n, 36), 30.0)
        rays[:, :4] = 5.0
        self.journal_estimate = types.SimpleNamespace(position=torch.zeros((n, 2)), velocity=torch.zeros((n, 3)),
                                                      rays=rays)

    def reset(self):
        return {}, {}

    def step(self, actions):
        est = self.journal_estimate
        est.position = est.position + 0.05 * actions
        est.velocity = torch.cat((actions, actions[:, :1]), dim=-1) * 0.3
        term = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.t == 1:
            term[0] = True
        self.t += 1
        return {}, torch.zeros(self.num_envs), term, torch.zeros(self.num_envs, dtype=torch.bool), {}

    def close(self):
        pass


def make(task, cfg=None, render_mode=None):
    return JournalFakeEnv(cfg)
'''
SITECUSTOMIZE = '''
import os

if os.environ.get("G9_STUB") == "1":
    import eval_protocol
    import mppi_controller as mc

    eval_protocol.resolve_protocol = lambda cfg, args, assignments=None: {}

    class CpuReplayBackend:
        """CPU stand-in for CudaGraphBackend (the module global CudaGraphMPPI instantiates)."""
        name = "cpu_replay_stand_in"
        capture_error_mode = "n/a"

        def __init__(self, device):
            self.device = device

        def warmup(self, fn, iterations):
            for _ in range(int(iterations)):
                fn()

        def capture(self, fn):
            if os.environ.get("G9_DIAG_MODE") == "fail":
                raise RuntimeError("operation not permitted when stream is capturing (stub)")
            outputs = fn()
            return {"fn": fn, "outputs": outputs}, outputs

        def replay(self, handle):
            fresh = handle["fn"]()
            for static, new in zip(handle["outputs"], fresh):
                static.copy_(new)
            if os.environ.get("G9_DIAG_MODE") == "perturb":
                handle["outputs"][0].add_(1.0e-3)

        def release(self, handle):
            pass

    mc.CudaGraphBackend = CpuReplayBackend
'''
STUBS = {"sitecustomize.py": SITECUSTOMIZE, "isaaclab/__init__.py": "", "isaaclab/app.py": APP_STUB,
         "isaaclab_tasks/__init__.py": "", "isaaclab_tasks/utils/__init__.py": TASK_UTILS_STUB,
         "gymnasium/__init__.py": GYM_STUB}


def run_diag(extra=(), mode="ok", task="journal:reach", out_inside=False):
    with tempfile.TemporaryDirectory(prefix="g9_diag_") as workdir:
        workdir = Path(workdir)
        for relative, text in STUBS.items():
            path = workdir / "stubs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="ascii")
        mark = workdir / "marks.txt"
        out = (REPO / "g9_report_never.json") if out_inside else workdir / "report.json"
        argv = ["--task", task, "--vessel", "blueboat", "--num_envs", "2", "--steps", "4", "--first-k", "1",
                "--headless", "--out", str(out), *extra]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("USVBENCH_", "OBS_", "WAVE_", "SPAWN_"))}
        env.update(PYTHONPATH=os.pathsep.join([str(workdir / "stubs"), str(SCRIPTS)]), G9_STUB="1",
                   G9_MARK=str(mark), G9_DIAG_MODE=mode, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "diag_mppi_graph.py"), *argv], cwd=str(workdir),
                                env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=SUBPROCESS_TIMEOUT_S)
        marks = mark.read_text(encoding="ascii").split() if mark.exists() else []
        report = json.loads(out.read_text(encoding="ascii")) if out.exists() and not out_inside else None
        leftovers = sorted(p.name for p in workdir.glob("report.json*"))
    return {"code": result.returncode, "output": result.stdout + result.stderr, "marks": marks, "report": report,
            "files": leftovers}


def test_subprocess_diag_equivalent_report_and_exit_zero():
    run = run_diag()
    assert run["code"] == 0, run["output"][-4000:]
    assert run["marks"] == ["launch", "close"] and run["files"] == ["report.json"]
    report = run["report"]
    summary = report["summary"]
    assert summary["verdict"] == "EQUIVALENT" and summary["steps"] == 4 and summary["bit_identical_steps"] == 4
    assert summary["key_mismatch_steps"] == 0 and summary["env_resets"] == 1
    assert report["writes_certificate"] is False and report["task_id"] == "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1"
    assert report["mppi"]["K"] == 256 and report["mppi"]["H"] == 60 and report["mode"] == "goal"
    assert report["implementation_stamp"] == {"mode": "cuda_graph_v1", "captured_graphs": 1, "recaptures": 0}
    assert report["graph"]["backend"] == "cpu_replay_stand_in"
    assert Path(report["mppi_controller"]["path"]).parent == SCRIPTS
    line = next(l for l in run["output"].splitlines() if l.startswith(diag.RESULT_TAG))
    assert json.loads(line[len(diag.RESULT_TAG):])["ok"] is True


def test_subprocess_diag_difference_exits_3_with_report_and_without_close():
    run = run_diag(mode="perturb")
    assert run["code"] == 3, run["output"][-4000:]
    assert "launch" in run["marks"] and "close" not in run["marks"] and "shutdown-with-open-app" not in run["marks"]
    summary = run["report"]["summary"]
    assert summary["verdict"] == "DIFFERENT" and summary["steps_over_tolerance"] == 4
    assert summary["max_abs_action_diff"] == pytest.approx(1e-3, rel=1e-3)


def test_subprocess_diag_failed_capture_exits_1_without_report_or_close():
    run = run_diag(mode="fail")
    assert run["code"] == 1, run["output"][-4000:]
    assert run["report"] is None and run["files"] == []
    assert "close" not in run["marks"] and "GraphCaptureError" in run["output"]
    assert "[MPPI-GRAPH] CUDA-graph MPPI capture failed" in run["output"]


@pytest.mark.parametrize("kwargs,text", [
    (dict(out_inside=True), "outside the release root"),
    (dict(task="Isaac-USV-HazardCross-Direct-v1"), "journal ids only"),
    (dict(extra=("--mppi-impl", "eager")), "both implementations"),
])
def test_subprocess_diag_refusals_before_launch(kwargs, text):
    run = run_diag(**kwargs)
    assert run["code"] == 2 and run["marks"] == [], run["output"][-3000:]
    assert text in run["output"]
    assert not (REPO / "g9_report_never.json").exists()
