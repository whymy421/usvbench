"""G24b CPU tests: the journal station-keeping hold law continuous_v2 and the docking law RunupDockingLaw. No Isaac.

* LOSController "continuous_v2" (journal station keeping, owner decision "continuous, not deadband"): defect D1
  removed (no forward thrust inside the zone with the station abeam or astern; aligned thrust min(d / zone, 0.6));
  defect D2 removed (aligned thrust outside the zone >= 0.6: monotone and continuous at the zone edge); the 0.2
  creep outside the zone while |e| >= pi / 4 and the PID unchanged; the legacy "deadband" / "continuous"
  branches are bitwise the G25b code (verbatim reference copies below) on random inputs;
* continuous_v2 holds the registry model inside the hold zone in calm water and against a steady 0.3 m/s drift;
* RunupDockingLaw: DockingLaw itself is bitwise the G25b code; a vessel already past the run-up point gets
  exactly DockingLaw's action; otherwise the APPROACH steers at the run-up point first (latched, reset clears),
  the ALIGN / HOLD actions stay DockingLaw's; from a lateral offset the model passes the run-up point, crosses the
  mouth line near the berth axis and docks within the tolerances.

Run: python -B scripts/test_g24b_classical_laws.py   (or pytest)
"""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import journal_controllers as jc  # noqa: E402
from mppi_controller import VesselDynamicsV2  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


# ------------------------------------------------------------ G25b reference copies (verbatim laws) --
def g25b_los_hold(pid, inputs, targets, zone, slow, hold_mode):
    """journal_controllers.LOSController.act of the G25b tree (hold-zone branches), verbatim."""
    error = jc.bearing_error(inputs.position_world, inputs.yaw, targets)
    distance = torch.linalg.vector_norm(targets - inputs.position_world, dim=-1)
    inside = distance <= zone
    profile = ((distance - zone) / (slow - zone)).clamp(0.0, 1.0)
    thrust = torch.where(error.abs() < math.pi / 4.0, profile, torch.full_like(profile, 0.2))
    if hold_mode == "deadband":
        yaw = pid(error, active=~inside)
        yaw = torch.where(inside, torch.zeros_like(yaw), yaw)
        pid.reset(inside)
        thrust = torch.where(inside, torch.zeros_like(thrust), thrust)
    else:
        yaw = pid(error)
        ki = pid.ki
        if not torch.is_tensor(ki) and ki > 0.0:
            pid.integral.clamp_(-1.0 / ki, 1.0 / ki)
        thrust = torch.where(inside, (distance / max(zone, 1e-6)).clamp(0.0, 0.6), thrust)
    return torch.stack((thrust, yaw), dim=-1)


def g25b_dock_act(law, inputs, berth_point, berth_heading):
    """journal_controllers.DockingLaw.act of the G25b tree, verbatim (law = a DockingLaw for its state)."""
    heading = berth_heading / torch.linalg.vector_norm(berth_heading, dim=-1, keepdim=True).clamp_min(1e-6)
    to_berth = berth_point - inputs.position_world
    distance = torch.linalg.vector_norm(to_berth, dim=-1)
    los_error = jc.bearing_error(inputs.position_world, inputs.yaw, berth_point)
    dock_error = jc.wrap(torch.atan2(heading[..., 1], heading[..., 0]) - inputs.yaw)
    speed = torch.linalg.vector_norm(inputs.velocity[..., :2], dim=-1)
    previous = law.phase.clone()
    law.phase[(previous == law.APPROACH) & (distance <= law.align_radius)] = law.ALIGN
    ready = (distance <= law.align_radius) & (dock_error.abs() <= math.radians(10.0)) & (speed < 0.2)
    law.phase[(previous == law.ALIGN) & ready] = law.HOLD
    law.phase[(previous == law.HOLD) & ~ready] = law.ALIGN
    law.phase[(previous != law.APPROACH) & (distance > law.exit_radius)] = law.APPROACH
    error = torch.where(law.phase == law.APPROACH, los_error, dock_error)
    yaw = (law.kp * error - law.k_rate * inputs.velocity[..., 2]).clamp(-1.0, 1.0)
    deadband = (law.phase == law.HOLD) & (dock_error.abs() < math.radians(8.0))
    yaw = torch.where(deadband, torch.zeros_like(yaw), yaw)
    approach = ((distance - law.stop_radius) / (law.brake_radius - law.stop_radius)).clamp(0.0, 1.0)
    approach = approach * torch.cos(los_error).clamp(0.0, 1.0)
    thrust = torch.where(law.phase == law.APPROACH, approach, torch.zeros_like(approach))
    return torch.stack((thrust, yaw), dim=-1)


def random_inputs(gen, n, spread):
    position = (torch.rand((n, 2), generator=gen, dtype=torch.float64) - 0.5) * 2.0 * spread
    yaw = (torch.rand(n, generator=gen, dtype=torch.float64) - 0.5) * 2.0 * math.pi
    velocity = (torch.rand((n, 3), generator=gen, dtype=torch.float64) - 0.5) * 0.6
    return jc.JournalInputs(position, position, yaw, velocity, torch.zeros((n, 36), dtype=torch.float64))


# ------------------------------------------------------------------------------ continuous_v2 law --
def _los(vessel, n):
    return jc.LOSController(n, "cpu", torch.float64, 1.0 / 60.0, jc.load_gains(vessel))


def test_legacy_hold_branches_are_bitwise_the_g25b_code():
    gen = torch.Generator().manual_seed(11)
    for vessel in VESSELS:
        for mode in ("deadband", "continuous"):
            a, b = _los(vessel, 64), _los(vessel, 64)
            target = torch.zeros((64, 2), dtype=torch.float64)
            for _ in range(50):
                inputs = random_inputs(gen, 64, 8.0)
                out = a.act(inputs, target, hold_zone=(2.0, 5.0), hold_mode=mode)
                ref = g25b_los_hold(b.pid, inputs, target, 2.0, 5.0, mode)
                assert torch.equal(out, ref), (vessel, mode)


def test_continuous_v2_removes_d1_and_d2_and_keeps_the_rest():
    zone, slow = 2.0, 5.0
    law = _los("blueboat_v2", 1)
    target = torch.zeros((1, 2), dtype=torch.float64)

    def act(x, y, yaw):
        p = torch.tensor([[x, y]], dtype=torch.float64)
        inputs = jc.JournalInputs(p, p, torch.tensor([yaw], dtype=torch.float64), torch.zeros((1, 3), dtype=torch.float64),
                                  torch.zeros((1, 36), dtype=torch.float64))
        law.reset(torch.tensor([True]))
        return law.act(inputs, target, hold_zone=(zone, slow), hold_mode="continuous_v2")[0]

    # D1: inside the zone with the station astern / abeam -> no forward thrust (legacy: min(d / zone, 0.6) > 0)
    assert float(act(1.0, 0.0, 0.0)[0]) == 0.0                      # station astern (bow +x, station at -x)
    assert abs(float(act(0.0, 1.5, 0.0)[0])) < 1e-12                # station abeam (cos(pi / 2) ~ 6e-17)
    # aligned inside: min(d / zone, 0.6) (legacy value kept)
    assert abs(float(act(1.0, 0.0, math.pi)[0]) - 0.5) < 1e-12
    assert abs(float(act(1.9, 0.0, math.pi)[0]) - 0.6) < 1e-12
    # D2: aligned just outside the zone >= the in-zone edge value (legacy: profile ~ 0 there)
    just_in = float(act(zone - 1e-6, 0.0, math.pi)[0])
    just_out = float(act(zone + 1e-6, 0.0, math.pi)[0])
    assert just_in == 0.6 and just_out == 0.6
    assert abs(float(act(4.0, 0.0, math.pi)[0]) - (4.0 - zone) / (slow - zone)) < 1e-12    # profile above 0.6 kept
    assert float(act(9.0, 0.0, math.pi)[0]) == 1.0
    # unaligned outside: the 0.2 creep (unchanged)
    assert abs(float(act(4.0, 0.0, 0.0)[0]) - 0.2) < 1e-12
    # monotone restoring thrust along the radial line, aligned
    values = [float(act(d, 0.0, math.pi)[0]) for d in [0.1 * k for k in range(1, 80)]]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    # yaw: the heading PID on the bearing error, as the legacy continuous branch
    law_b = _los("blueboat_v2", 1)
    p = torch.tensor([[3.0, 1.0]], dtype=torch.float64)
    inputs = jc.JournalInputs(p, p, torch.tensor([0.3], dtype=torch.float64), torch.zeros((1, 3), dtype=torch.float64),
                              torch.zeros((1, 36), dtype=torch.float64))
    law.reset(torch.tensor([True]))
    ref = g25b_los_hold(law_b.pid, inputs, target, zone, slow, "continuous")
    out = law.act(inputs, target, hold_zone=(zone, slow), hold_mode="continuous_v2")
    assert torch.equal(out[:, 1], ref[:, 1])
    try:
        law.act(inputs, target, hold_zone=(zone, slow), hold_mode="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown hold mode must be refused")


class ModelEnv:
    """Registry-model world (calm) exposing the journal inputs; ``drift`` = a steady world-frame set (m/s)."""

    def __init__(self, vessel, drift=(0.0, 0.0)):
        self.num_envs, self.device = 1, torch.device("cpu")
        self.s = jc.length_scale(vessel)
        self.scene = SimpleNamespace(env_origins=torch.zeros((1, 3)))
        self.dyn = VesselDynamicsV2(vessel, dtype=torch.float64)
        self.state = torch.zeros((1, 6), dtype=torch.float64)
        self.drift = torch.tensor(drift, dtype=torch.float64)
        self.refresh()

    @property
    def robot(self):
        yaw = self.state[:, 2]
        quat = torch.stack((torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)), -1)
        return SimpleNamespace(data=SimpleNamespace(root_link_pose_w=torch.cat(
            (self.state[:, :2], torch.zeros((1, 1), dtype=torch.float64), quat), -1)))

    def refresh(self):
        self.journal_estimate = SimpleNamespace(position=self.state[:, :2].float(), velocity=self.state[:, 3:6].float(),
                                                rays=torch.full((1, 36), 30.0 * self.s))

    def step(self, action):
        self.state = self.dyn.step(self.state, action.double())
        self.state[:, :2] += self.drift / 60.0
        self.refresh()


def _hold_run(vessel, hold_mode, drift):
    gains = jc.load_gains(vessel)
    env = ModelEnv(vessel, drift=tuple(d * 1.0 for d in drift))
    s = env.s
    station = (12.0 * s, 6.0 * s)
    env.hold_point = torch.tensor([station])
    law = jc.JournalBaselineController(env, "station_keep", gains, dt=1 / 60, zone_radius_m=1.5 * s,
                                       slow_radius_m=5.0 * s, tolerance_m=2.5 * s, scale=s, hold_mode=hold_mode)
    inside, worst = 0, 0.0
    steps = int((120 if vessel == "blueboat_v2" else 240) * 60)
    for k in range(steps):
        env.step(law.act())
        d = math.dist(env.state[0, :2].tolist(), station)
        inside = inside + 1 if d <= 1.5 * s else 0
        if k >= steps - 60 * 60:
            worst = max(worst, d)
    return inside, worst


def test_continuous_v2_holds_the_model_in_calm_and_against_a_steady_drift():
    for vessel in VESSELS:
        s = jc.length_scale(vessel)
        for drift in ((0.0, 0.0), (0.3, 0.0), (0.0, -0.3)):
            inside, worst = _hold_run(vessel, "continuous_v2", drift)
            assert inside >= 60 * 60 and worst <= 1.5 * s, (vessel, drift, inside, worst)


# ----------------------------------------------------------------------------------- run-up docking --
def _dock_laws(vessel, n, runup_x):
    gains = jc.load_gains(vessel)
    s = jc.length_scale(vessel)
    base = jc.DockingLaw(n, "cpu", torch.float64, 1 / 60, gains, tolerance_m=2.5 * s, scale=s)
    runup = jc.RunupDockingLaw(n, "cpu", torch.float64, 1 / 60, gains, tolerance_m=2.5 * s, scale=s,
                               runup_x_m=runup_x, accept_m=1.0 * s)
    return base, runup, s


def test_docking_law_is_bitwise_the_g25b_code():
    gen = torch.Generator().manual_seed(5)
    for vessel in VESSELS:
        gains = jc.load_gains(vessel)
        s = jc.length_scale(vessel)
        a = jc.DockingLaw(64, "cpu", torch.float64, 1 / 60, gains, tolerance_m=2.5 * s, scale=s)
        b = jc.DockingLaw(64, "cpu", torch.float64, 1 / 60, gains, tolerance_m=2.5 * s, scale=s)
        berth = torch.zeros((64, 2), dtype=torch.float64)
        heading = torch.tensor([[1.0, 0.0]], dtype=torch.float64).repeat(64, 1)
        for _ in range(60):
            inputs = random_inputs(gen, 64, 4.0 * s)
            assert torch.equal(a.act(inputs, berth, heading), g25b_dock_act(b, inputs, berth, heading)), vessel
            assert torch.equal(a.phase, b.phase)


def test_runup_law_equals_the_docking_law_once_past_the_runup_point_and_aims_at_it_before():
    for vessel in VESSELS:
        base, law, s = _dock_laws(vessel, 2, runup_x=-9.4)
        berth = torch.zeros((2, 2), dtype=torch.float64)
        heading = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        # env 0 already past the run-up point on the axis (x_b = -5 >= -9.4); env 1 before it, laterally offset
        p = torch.tensor([[-5.0, 1.0], [-15.0, 3.0]], dtype=torch.float64)
        inputs = jc.JournalInputs(p, p, torch.tensor([0.2, 0.0], dtype=torch.float64),
                                  torch.zeros((2, 3), dtype=torch.float64), torch.zeros((2, 36), dtype=torch.float64))
        out, ref = law.act(inputs, berth, heading), base.act(inputs, berth, heading)
        assert torch.equal(out[0], ref[0]) and bool(law.past_runup[0]) and not bool(law.past_runup[1])
        e = jc.bearing_error(p[1:], inputs.yaw[1:], torch.tensor([[-9.4, 0.0]], dtype=torch.float64))
        assert torch.allclose(out[1, 1], (law.kp * e).clamp(-1.0, 1.0)[0])         # LOS at the run-up point
        assert not torch.equal(out[1], ref[1])
        # within accept_m of the run-up point: latched on the berth, stays latched when pushed back
        p2 = torch.tensor([[-5.0, 1.0], [-9.4 - 0.5 * s, 0.2 * s]], dtype=torch.float64)
        law.act(jc.JournalInputs(p2, p2, inputs.yaw, inputs.velocity, inputs.rays), berth, heading)
        assert bool(law.past_runup[1])
        law.act(inputs, berth, heading)
        assert bool(law.past_runup[1])
        law.reset(torch.tensor([False, True]))
        assert bool(law.past_runup[0]) and not bool(law.past_runup[1])
        # ALIGN / HOLD phases: DockingLaw's action
        p3 = torch.tensor([[0.1, 0.0], [0.2, 0.1]], dtype=torch.float64)
        i3 = jc.JournalInputs(p3, p3, torch.tensor([0.5, -0.4], dtype=torch.float64),
                              torch.zeros((2, 3), dtype=torch.float64), torch.zeros((2, 36), dtype=torch.float64))
        out, ref = law.act(i3, berth, heading), base.act(i3, berth, heading)
        assert bool((law.phase != law.APPROACH).all()) and torch.equal(out, ref)
        for bad in (dict(runup_x_m=1.0, accept_m=1.0), dict(runup_x_m=-1.0, accept_m=0.0)):
            try:
                jc.RunupDockingLaw(1, "cpu", torch.float64, 1 / 60, jc.load_gains(vessel), tolerance_m=2.5, scale=1.0,
                                   **bad)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid run-up parameters must be refused")


def test_runup_law_docks_the_model_from_a_lateral_offset_through_the_runup_point():
    """Journal geometry (journal_cfg_v2 docking plan, tier 2): start at x_open - approach, lateral offset the plan's
    maximum, heading at the mouth; the model passes within accept_m of the run-up point (or past it on the axis
    while near it), crosses the mouth line within a quarter slip width of the axis and docks (2.5 s m, 15 deg,
    0.3 m/s for 5 s)."""
    plans = jc.shared_module("journal_cfg_v2")
    for vessel in VESSELS:
        p = plans.plan_class("docking", vessel)(geometry_tier=2).params
        s = jc.length_scale(vessel)
        x_open = float(p["slip_opening_x_m"])
        runup_x = x_open - float(plans.vessel_record(vessel).geometry.approach_runup_m)
        env = ModelEnv(vessel)
        start = (x_open - float(p["approach_distance_m"]), float(p["lateral_offset_max_m"]))
        env.state[0, 0], env.state[0, 1] = start
        env.state[0, 2] = math.atan2(-start[1], x_open - start[0])
        env.refresh()
        env.dock_point = torch.zeros((1, 2))
        env.dock_heading = torch.tensor([[1.0, 0.0]])
        tol = float(p["position_tolerance_m"])
        law = jc.JournalBaselineController(env, "docking", jc.load_gains(vessel), dt=1 / 60, zone_radius_m=tol,
                                           slow_radius_m=6.0 * s, tolerance_m=tol, scale=s,
                                           runup={"runup_x_m": runup_x, "accept_m": 1.0 * s})
        assert isinstance(law.dock, jc.RunupDockingLaw)
        near_runup, mouth_y, held = math.inf, None, 0
        for _ in range(int(240 * 60)):
            x0 = float(env.state[0, 0])
            env.step(law.act())
            x, y, yaw = (float(v) for v in env.state[0, :3])
            near_runup = min(near_runup, math.hypot(x - runup_x, y))
            if mouth_y is None and x0 < x_open <= x:
                mouth_y = y
            ok = (math.hypot(x, y) <= tol and math.cos(yaw) >= math.cos(math.radians(15))
                  and float(torch.linalg.vector_norm(env.state[0, 3:5])) <= 0.3)
            held = held + 1 if ok else 0
            if held >= 5 * 60:
                break
        assert near_runup <= 1.0 * s + 1e-9, (vessel, near_runup)
        assert mouth_y is not None and abs(mouth_y) <= 0.25 * float(p["slip_free_width_m"]), (vessel, mouth_y)
        assert held >= 5 * 60, (vessel, held)


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(1 if run_all() else 0)
