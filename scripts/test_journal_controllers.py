"""CPU tests of the journal controller inputs, laws and frozen gains (F7). No Isaac.

* item 4: every journal controller reads env.journal_estimate (D5) and the ray
  block; a perception dose changes the controller INPUT (and action) while
  the truth state is unchanged (REAL mixin + REAL condition layer through the
  F0 FakeJournalEnv); no controller touches D7 objects, the static obstacle
  list, the condition layer, sea, current or wind (every such member raises);
* ray returns become obstacle discs at the right place;
* item 2: the frozen per-vessel gains carry the pre-declared procedure's
  digest and are reproduced by re-running the procedure; a gain block that is
  not frozen makes the controllers refuse to run;
* the journal planner plans around an obstacle it only sees through the rays;
  the baseline laws reach, hold and dock on the registry model.

Run: python -B scripts/test_journal_controllers.py   (or pytest)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import journal_controllers as jc  # noqa: E402
from mppi_controller import VesselDynamicsV2  # noqa: E402
from test_classical_harbour import ForbiddenAccess, ray_cast  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
PROCEDURE = Path(_SCRIPTS) / "TUNING_PROCEDURE_V2.md"


class Guarded:
    """Proxy that forwards to an env but raises on every forbidden member."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name):
        if name in jc.FORBIDDEN_ENV_MEMBERS:
            raise ForbiddenAccess("journal controller read forbidden env member " + name)
        return getattr(object.__getattribute__(self, "_env"), name)


# ------------------------------------------------------------------ item 4 --
def _fake_env(tokens=()):
    import glue_fixtures as gf

    cfg = gf.journal_cfg("blueboat_v2", tokens=tokens, goal_radius=2.0)
    env = gf.FakeJournalEnv(cfg, 2)
    env.reset()
    env.target_pos = torch.tensor([[30.0, 0.0], [80.0, 0.0]])       # world goals (env origins 0, 50)
    return env


def test_perception_dose_changes_controller_input_while_truth_is_unchanged():
    import eval_mppi

    clean = _fake_env()
    biased = _fake_env(("journal.perception.bias.D3",))           # 0.5 m position bias
    for step in range(3):
        if step:
            action = torch.tensor([[0.6, 0.2], [0.4, -0.3]])
            clean.step(action)
            biased.step(action)
        # truth identical (perception never touches the physics) ...
        assert torch.equal(clean.robot.data.root_link_pose_w, biased.robot.data.root_link_pose_w)
        assert torch.equal(clean.robot.data.root_com_vel_w, biased.robot.data.root_com_vel_w)
        a, b = jc.read_journal_inputs(Guarded(clean)), jc.read_journal_inputs(Guarded(biased))
        # ... while the controller input carries the 0.5 m bias
        offset = torch.linalg.vector_norm(b.position_world - a.position_world, dim=-1)
        assert torch.allclose(offset, torch.full_like(offset, 0.5), atol=1e-5), offset
        assert torch.equal(a.yaw, b.yaw)
        truth = clean.robot.data.root_link_pose_w[:, :2]
        assert torch.allclose(a.position_world, truth.to(a.position_world.dtype), atol=1e-6)
    # every controller class sees the difference
    gains = jc.load_gains("blueboat_v2")
    los_a = jc.LOSController(2, "cpu", torch.float32, 1 / 60, gains).act(a, torch.tensor([[30.0, 1.0], [80.0, 1.0]]))
    los_b = jc.LOSController(2, "cpu", torch.float32, 1 / 60, gains).act(b, torch.tensor([[30.0, 1.0], [80.0, 1.0]]))
    assert not torch.equal(los_a, los_b)
    bridge_a = eval_mppi.JournalEnvBridge(torch, Guarded(clean), "goal", 30.0)
    bridge_b = eval_mppi.JournalEnvBridge(torch, Guarded(biased), "goal", 30.0)
    assert not torch.equal(bridge_a.state(), bridge_b.state())
    assert torch.equal(bridge_a.state()[:, 2:], bridge_b.state()[:, 2:])


def test_no_journal_controller_reads_forbidden_members():
    import eval_mppi
    from classical_planner_pid import JournalPlannerLOSPIDController, journal_ladder

    env = _fake_env()
    guarded = Guarded(env)
    gains = jc.load_gains("blueboat_v2")
    bridge = eval_mppi.JournalEnvBridge(torch, guarded, "goal", 30.0)
    bridge.state(), bridge.targets(), bridge.obstacles()
    env.control_step_s = 1.0 / 60.0
    planner = JournalPlannerLOSPIDController(guarded, replan_every=60, waypoint_spacing_m=2.0, accept_radius_m=1.0,
                                             plan_cell_m=0.5, ladder=journal_ladder("blueboat_v2"), gains=gains,
                                             max_range_m=30.0)
    assert planner.act().shape == (2, 2)
    env.hold_point = env.target_pos.clone()
    env.dock_point = env.target_pos.clone()
    env.dock_heading = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    for mode in ("station_keep", "docking", "point_nav"):
        controller = jc.JournalBaselineController(guarded, mode, gains, dt=1 / 60, zone_radius_m=2.0,
                                                  slow_radius_m=5.0, tolerance_m=2.5, scale=1.0)
        assert controller.act(None).shape == (2, 2)
    try:
        guarded.obstacle_centers
    except ForbiddenAccess:
        pass
    else:
        raise AssertionError("the guard must trap forbidden members")


def test_ray_returns_become_discs_at_the_surface_point():
    position = torch.tensor([[10.0, -5.0]])
    yaw = torch.tensor([math.pi / 2])
    rays = torch.full((1, 36), 30.0)
    rays[0, 0] = 4.0          # dead ahead (world +y)
    rays[0, 9] = 2.0          # 90 deg to port (world -x)
    centers, radii, active = jc.ray_obstacles(position, yaw, rays, 30.0)
    assert active[0].tolist() == [k in (0, 9) for k in range(36)]
    assert torch.allclose(centers[0, 0], torch.tensor([10.0, -1.0]), atol=1e-5)
    assert torch.allclose(centers[0, 9], torch.tensor([8.0, -5.0]), atol=1e-5)
    assert math.isclose(float(radii[0, 0]), 4.0 * math.tan(math.pi / 36), rel_tol=1e-6)
    assert float(radii[0, 5]) == 0.0
    _, bare, _ = jc.ray_obstacles(position, yaw, rays, 30.0, fill_gaps=False)
    assert float(bare.abs().max()) == 0.0
    cfg = SimpleNamespace(journal_plan=SimpleNamespace(params={"ray_sensor_max_range_m": 134.3}))
    assert jc.ray_max_range(cfg) == 134.3
    try:
        jc.ray_max_range(SimpleNamespace())
    except ValueError:
        pass
    else:
        raise AssertionError("a cfg without a ray range must be refused")


# ------------------------------------------------------------------ item 2 --
def _lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_gains_are_frozen_by_the_declared_procedure_and_reproducible():
    import tune_classical_v2 as tune

    data = json.loads(jc.GAINS_PATH.read_text(encoding="ascii"))
    assert data["format"] == jc.GAINS_FORMAT
    assert data["procedure_sha256"] == _lf(PROCEDURE)
    for vessel in VESSELS:
        gains = jc.load_gains(vessel)
        assert gains["procedure_sha256"] == _lf(PROCEDURE)
        assert gains["los_heading_pid"]["ki"] == 0.0
    blue = data["vessels"]["blueboat_v2"]
    # R1: the legacy LOS gains are kept (within 10 % of the grid optimum) ...
    assert blue["los_heading_pid"]["rule"] == "R1"
    assert blue["los_heading_pid"]["gains"] == {"kp": 2.0, "ki": 0.0, "kd": 0.5}
    # G4-A5: tuned on the current calm nominal model; the superseded freeze is kept
    import mppi_controller
    for vessel in VESSELS:
        assert data["model_damping"][vessel] == mppi_controller.model_damping_record(vessel)
        assert jc.load_gains(vessel)["model_damping"] == data["model_damping"][vessel]
    assert data["model_damping"]["wamv_vrx_v2_opt1"]["swayyaw_damping"] == "crossflow_sarda_strip"
    last = data["history"][-1]
    assert last["sha256"] == "c713bba0351350512583668c88b26e3d5731c6a408ce78c0d6323cf6d0edda08"
    assert set(last["vessels"]) == {"wamv_vrx_v2_opt1"}
    # ... and every choice is what the procedure yields when re-run now.
    fresh, _ = tune.tune(PROCEDURE)
    assert fresh["model_damping"] == data["model_damping"]
    for vessel in VESSELS:
        for name in ("los_heading_pid", "dock_yaw_pd"):
            a, b = fresh["vessels"][vessel][name], data["vessels"][vessel][name]
            assert (a["rule"], a["status"], a.get("gains")) == (b["rule"], b["status"], b.get("gains")), \
                (vessel, name, a, b)
            assert math.isclose(a["objective_s"], b["objective_s"], rel_tol=1e-9)


def test_controllers_refuse_unfrozen_gains():
    data = json.loads(jc.GAINS_PATH.read_text(encoding="ascii"))
    data["vessels"]["wamv_vrx_v2_opt1"]["dock_yaw_pd"]["status"] = "FAILED"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "gains.json"
        path.write_text(json.dumps(data), encoding="ascii")
        try:
            jc.load_gains("wamv_vrx_v2_opt1", path)
        except RuntimeError as exc:
            assert "not frozen" in str(exc)
        else:
            raise AssertionError("a FAILED gain block must be refused")
        assert jc.load_gains("blueboat_v2", path)["vessel"] == "blueboat_v2"
        try:
            jc.load_gains("blueboat_v2", Path(folder) / "missing.json")
        except RuntimeError:
            pass
        else:
            raise AssertionError("a missing gains file must be refused")
        # G4-A5: gains tuned on another calm nominal model (here the VRX-classic WAM-V) are refused
        data = json.loads(jc.GAINS_PATH.read_text(encoding="ascii"))
        data["model_damping"]["wamv_vrx_v2_opt1"].update(sway_quad=0.0, yaw_quad=0.0, swayyaw_damping="vrx_classic")
        path.write_text(json.dumps(data), encoding="ascii")
        try:
            jc.load_gains("wamv_vrx_v2_opt1", path)
        except RuntimeError as exc:
            assert "model damping" in str(exc)
        else:
            raise AssertionError("gains tuned on another model must be refused")
        assert jc.load_gains("blueboat_v2", path)["vessel"] == "blueboat_v2"


# --------------------------------------------------------- closed-loop laws --
class ModelEnv:
    """Registry-model world exposing the journal inputs (estimate = truth)."""

    def __init__(self, vessel, circles=(), n=1):
        self.num_envs, self.device, self.control_step_s = n, torch.device("cpu"), 1.0 / 60.0
        self.s = jc.length_scale(vessel)
        self.cfg = SimpleNamespace(vehicle=vessel)
        self.scene = SimpleNamespace(env_origins=torch.zeros((n, 3)))
        self.dyn = VesselDynamicsV2(vessel, dtype=torch.float64)
        self.state = torch.zeros((n, 6), dtype=torch.float64)
        self.circles = list(circles)
        self.refresh()

    @property
    def robot(self):
        yaw = self.state[:, 2]
        quat = torch.stack((torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)), -1)
        return SimpleNamespace(data=SimpleNamespace(root_link_pose_w=torch.cat(
            (self.state[:, :2], torch.zeros((self.num_envs, 1), dtype=torch.float64), quat), -1)))

    def refresh(self):
        rays = ray_cast(self.state[:, :2], self.state[:, 2], self.circles, 30.0 * self.s)
        self.journal_estimate = SimpleNamespace(position=self.state[:, :2].float(),
                                                velocity=self.state[:, 3:6].float(), rays=rays.float())

    def step(self, action):
        self.state = self.dyn.step(self.state, action.double())
        self.refresh()


def test_journal_planner_avoids_an_obstacle_seen_only_through_the_rays():
    from classical_planner_pid import JournalPlannerLOSPIDController, journal_ladder

    for vessel in VESSELS:
        probe = ModelEnv(vessel)
        s = probe.s
        env = ModelEnv(vessel, circles=[(15.0 * s, 0.4 * s, 2.0 * s)])
        env.target_pos = torch.tensor([[30.0 * s, 0.0]])
        guarded = Guarded(env)
        planner = JournalPlannerLOSPIDController(guarded, replan_every=30, waypoint_spacing_m=2.0 * s,
                                                 accept_radius_m=1.0 * s, plan_cell_m=0.25 * s,
                                                 ladder=journal_ladder(vessel), gains=jc.load_gains(vessel),
                                                 max_range_m=30.0 * s)
        clearance, reached = math.inf, False
        radius = env.dyn.params["capsule_radius_m"]
        for _ in range(int((90.0 if vessel == "blueboat_v2" else 240.0) * 60)):
            env.step(planner.act())
            p = env.state[0, :2].tolist()
            clearance = min(clearance, math.dist(p, env.circles[0][:2]) - env.circles[0][2] - radius)
            if math.dist(p, (30.0 * s, 0.0)) <= 2.0 * s:
                reached = True
                break
        assert reached and clearance > 0.0, (vessel, reached, clearance)
        assert planner.straight_fallbacks == 0, vessel


def test_baseline_laws_reach_hold_and_dock_on_the_model():
    for vessel in VESSELS:
        gains = jc.load_gains(vessel)
        s = jc.length_scale(vessel)
        # station keeping: arrive and stay inside the hold zone (deadband law)
        env = ModelEnv(vessel)
        env.hold_point = torch.tensor([[12.0 * s, 6.0 * s]])
        law = jc.JournalBaselineController(env, "station_keep", gains, dt=1 / 60, zone_radius_m=2.0 * s,
                                           slow_radius_m=5.0 * s, tolerance_m=2.5 * s, scale=s)
        inside = 0
        for _ in range(int(120 * 60)):
            env.step(law.act())
            inside = inside + 1 if math.dist(env.state[0, :2].tolist(), (12.0 * s, 6.0 * s)) <= 2.0 * s else 0
        assert inside >= 30 * 60, (vessel, "hold", inside)
        # docking: berth reached with heading and speed inside the tolerances for 5 s
        env = ModelEnv(vessel)
        env.dock_point = torch.tensor([[20.0 * s, -4.0 * s]])
        env.dock_heading = torch.tensor([[0.0, -1.0]])
        law = jc.JournalBaselineController(env, "docking", gains, dt=1 / 60, zone_radius_m=2.5 * s,
                                           slow_radius_m=6.0 * s, tolerance_m=2.5 * s, scale=s)
        held = 0
        for _ in range(int(180 * 60)):
            env.step(law.act())
            p, yaw = env.state[0, :2].tolist(), float(env.state[0, 2])
            ok = (math.dist(p, (20.0 * s, -4.0 * s)) <= 2.5 * s and math.cos(yaw + math.pi / 2) >= math.cos(math.radians(15))
                  and float(torch.linalg.vector_norm(env.state[0, 3:5])) <= 0.3)
            held = held + 1 if ok else 0
            if held >= 5 * 60:
                break
        assert held >= 5 * 60, (vessel, "dock", held)


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(1 if run_all() else 0)
