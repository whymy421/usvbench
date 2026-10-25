"""CPU tests of the staged classical harbour controller (F7). No Isaac.

A fake journal harbour env drives the REAL controller on the registry vessel
model (mppi_controller.VesselDynamicsV2 as the world): departure gates in
order, the transit gate, then berth-mouth entry into a walled slip + a 5 s
dwell, with an obstacle and the slip walls seen only through a point-return
ray fan. The fake env publishes F6's phase-target dict (schema
usvbench.harbour_phase_targets/1, PHASE_TARGETS_F6.md; env-local COM frame).
Every env member a journal controller must never read raises
(journal_controllers.FORBIDDEN_ENV_MEMBERS).

Run: python -B scripts/test_classical_harbour.py   (or pytest)
"""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import torch  # noqa: E402

import classical_harbour as ch  # noqa: E402
import journal_controllers as jc  # noqa: E402
from mppi_controller import VesselDynamicsV2  # noqa: E402

RAYS = 36


class ForbiddenAccess(AssertionError):
    pass


def ray_cast(position, yaw, circles, max_range):
    """[N,R] ranges from the COM to the first circle surface along each ray."""
    n = position.shape[0]
    out = torch.full((n, RAYS), float(max_range), dtype=torch.float64)
    angles = torch.arange(RAYS, dtype=torch.float64) * (2.0 * math.pi / RAYS)
    for i in range(n):
        for k in range(RAYS):
            a = float(yaw[i]) + float(angles[k])
            d = (math.cos(a), math.sin(a))
            for cx, cy, r in circles:
                fx, fy = cx - float(position[i, 0]), cy - float(position[i, 1])
                along = fx * d[0] + fy * d[1]
                perp2 = fx * fx + fy * fy - along * along
                if along <= 0 or perp2 > r * r:
                    continue
                hit = along - math.sqrt(r * r - perp2)
                if 0.0 <= hit < out[i, k]:
                    out[i, k] = hit
    return out


class FakeHarbourEnv:
    """Minimal journal harbour env: gates in order, transit gate, berth + dwell."""

    def __init__(self, vessel="blueboat_v2", num_envs=1, obstacle=True):
        s = jc.length_scale(vessel)
        self.s = s
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.control_step_s = 1.0 / 60.0
        self.cfg = SimpleNamespace(vehicle=vessel, success_position_tolerance_m=2.5 * s,
                                   journal_plan=SimpleNamespace(params={"ray_sensor_max_range_m": 30.0 * s}))
        self.scene = SimpleNamespace(env_origins=torch.stack(
            (torch.arange(num_envs, dtype=torch.float64) * 1000.0, torch.zeros(num_envs, dtype=torch.float64),
             torch.zeros(num_envs, dtype=torch.float64)), -1))
        self.dyn = VesselDynamicsV2(vessel, dtype=torch.float64)
        self.state = torch.zeros((num_envs, 6), dtype=torch.float64)
        self.gates = [((20 * s, 0.0), (1.0, 0.0)), ((40 * s, 8 * s), (0.8, 0.6)), ((60 * s, 10 * s), (1.0, 0.0))]
        self.berth = (80 * s, 10 * s)
        self.berth_heading = (1.0, 0.0)
        self.mouth = (74 * s, 10 * s)
        self.slip_half = 2.5 * s
        self.circles = [(31 * s, 4.5 * s, 1.5 * s)] if obstacle else []
        # slip walls (disks) on both sides of the berth axis, mouth to back
        for k in range(10):
            for side in (-1.0, 1.0):
                self.circles.append(((74 + k) * s, 10 * s + side * (self.slip_half + 0.3 * s), 0.3 * s))
        self.mouth_latched = [False] * num_envs
        self.half_width, self.approach, self.min_speed = 3.0 * s, 1.0 * s, 0.2
        self.stage_index = [0] * num_envs            # gates passed
        self.armed = [False] * num_envs
        self.dwell = [0.0] * num_envs
        self.complete = [False] * num_envs
        self.stage_log = [[] for _ in range(num_envs)]
        self.min_clearance = [math.inf] * num_envs
        self.t = 0.0
        self._update_estimate()

    # every forbidden member raises: controllers must never read them
    def __getattr__(self, name):
        if name in jc.FORBIDDEN_ENV_MEMBERS:
            raise ForbiddenAccess("journal controller read forbidden env member " + name)
        raise AttributeError(name)

    @property
    def robot(self):
        yaw = self.state[:, 2]
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        pose = torch.cat((self.state[:, :2] + self.scene.env_origins[:, :2],
                          torch.zeros((self.num_envs, 1), dtype=torch.float64), quat), -1)
        return SimpleNamespace(data=SimpleNamespace(root_link_pose_w=pose))

    def _update_estimate(self):
        rays = ray_cast(self.state[:, :2], self.state[:, 2], self.circles, 30.0 * self.s)
        self.journal_estimate = SimpleNamespace(position=self.state[:, :2].clone(),
                                                velocity=self.state[:, 3:6].clone(), rays=rays)

    def _stage(self, i):
        if self.complete[i]:
            return ch.STAGE_COMPLETE
        k = self.stage_index[i]
        if k >= len(self.gates):
            return ch.STAGE_DOCKING
        return ch.STAGE_DEPARTURE if k < len(self.gates) - 1 else ch.STAGE_TRANSIT

    def journal_phase_targets(self):
        """F6 schema usvbench.harbour_phase_targets/1 (the keys F7 reads)."""
        n = self.num_envs
        mids = torch.tensor([g[0] for g in self.gates] + [self.mouth], dtype=torch.float64)
        normals = torch.tensor([g[1] for g in self.gates] + [self.berth_heading], dtype=torch.float64)
        remaining = torch.zeros((n, 4), dtype=torch.bool)
        index = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            for k in range(4):
                passed = k < self.stage_index[i] if k < 3 else self.mouth_latched[i]
                remaining[i, k] = not passed
            index[i] = int(remaining[i].float().argmax()) if bool(remaining[i].any()) else 4
        berth = torch.tensor([[self.berth[0], self.berth[1], 0.0]] * n, dtype=torch.float64)
        active_mid = torch.stack([berth[i, :2] if index[i] == 4 else mids[index[i]] for i in range(n)])
        active_nrm = torch.stack([normals[3] if index[i] == 4 else normals[index[i]] for i in range(n)])
        return {"schema": "usvbench.harbour_phase_targets/1", "frame": "env_local_com_xy",
                "stage": torch.tensor([self._stage(i) for i in range(n)]), "active_gate_index": index,
                "active_gate_midpoint": active_mid, "active_gate_normal": active_nrm,
                "active_gate_half_width_m": torch.full((n,), self.half_width, dtype=torch.float64),
                "gate_midpoints": mids.expand(n, 4, 2).clone(), "gate_normals": normals.expand(n, 4, 2).clone(),
                "remaining_gates": remaining,
                "gate_approach_m": torch.full((n, 3), self.approach, dtype=torch.float64),
                "berth_pose": berth,
                "berth_interior": torch.tensor([[-6.0 * self.s, 3.0 * self.s, self.slip_half]] * n,
                                               dtype=torch.float64)}

    def step(self, action):
        previous = self.state.clone()
        self.state = self.dyn.step(self.state, action.to(torch.float64))
        self.t += self.control_step_s
        for i in range(self.num_envs):
            p0, p1 = previous[i, :2].tolist(), self.state[i, :2].tolist()
            for cx, cy, r in self.circles:
                self.min_clearance[i] = min(self.min_clearance[i], math.dist(p1, (cx, cy)) - r
                                            - self.dyn.params["capsule_radius_m"])
            k = self.stage_index[i]
            if k < len(self.gates):
                (mx, my), (nx, ny) = self.gates[k]
                s0 = (p0[0] - mx) * nx + (p0[1] - my) * ny
                s1 = (p1[0] - mx) * nx + (p1[1] - my) * ny
                if not self.armed[i]:
                    self.armed[i] = s1 <= -self.approach
                elif s0 <= 0.0 <= s1 and s1 - s0 > 1e-12:
                    f = -s0 / (s1 - s0)
                    xi = (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]))
                    lateral = abs((xi[0] - mx) * -ny + (xi[1] - my) * nx)
                    c, sn = math.cos(float(self.state[i, 2])), math.sin(float(self.state[i, 2]))
                    vx = float(self.state[i, 3]) * c - float(self.state[i, 4]) * sn
                    vy = float(self.state[i, 3]) * sn + float(self.state[i, 4]) * c
                    if lateral <= self.half_width and vx * nx + vy * ny > self.min_speed:
                        self.stage_index[i] += 1
                        self.armed[i] = False
                        self.stage_log[i].append((k, round(self.t, 3)))
            elif not self.mouth_latched[i]:
                (mx, my), (nx, ny) = self.mouth, self.berth_heading
                s0 = (p0[0] - mx) * nx + (p0[1] - my) * ny
                s1 = (p1[0] - mx) * nx + (p1[1] - my) * ny
                lateral = abs((p1[0] - mx) * -ny + (p1[1] - my) * nx)
                if s0 <= 0.0 <= s1 and lateral <= self.slip_half:
                    self.mouth_latched[i] = True
                    self.stage_log[i].append(("mouth", round(self.t, 3)))
            elif not self.complete[i]:
                d = math.dist(p1, self.berth)
                heading_ok = math.cos(float(self.state[i, 2])) >= math.cos(math.radians(15.0))
                speed = float(torch.linalg.vector_norm(self.state[i, 3:5]))
                if d <= self.cfg.success_position_tolerance_m and heading_ok and speed <= 0.3:
                    self.dwell[i] += self.control_step_s
                else:
                    self.dwell[i] = 0.0
                if self.dwell[i] >= 5.0:
                    self.complete[i] = True
                    self.stage_log[i].append(("docked", round(self.t, 3)))
        self._update_estimate()


def _run_mission(vessel, limit_s):
    env = FakeHarbourEnv(vessel)
    gains = jc.load_gains(vessel)
    controller = ch.StagedHarbourController(env, gains=gains, max_range_m=30.0 * env.s, scale=env.s)
    steps = int(limit_s / env.control_step_s)
    for _ in range(steps):
        action = controller.act()
        assert action.shape == (1, 2) and bool((action.abs() <= 1.0).all())
        env.step(action)
        if env.complete[0]:
            break
    return env, controller


def test_blueboat_full_mission_in_order_and_docks():
    env, controller = _run_mission("blueboat_v2", 240.0)
    log = env.stage_log[0]
    assert [entry[0] for entry in log] == [0, 1, 2, "mouth", "docked"], log
    assert env.min_clearance[0] > 0.0, env.min_clearance
    assert controller.straight_fallbacks == 0


def test_wamv_full_mission_in_order_and_docks():
    env, controller = _run_mission("wamv_vrx_v2_opt1", 600.0)
    log = env.stage_log[0]
    assert [entry[0] for entry in log] == [0, 1, 2, "mouth", "docked"], log
    assert env.min_clearance[0] > 0.0, env.min_clearance


def test_phase_targets_fail_closed_and_frames():
    env = FakeHarbourEnv("blueboat_v2", num_envs=2, obstacle=False)
    phase = ch.phase_targets_world(env)
    assert torch.allclose(phase.gate_midpoint[1], torch.tensor(env.gates[0][0], dtype=torch.float64)
                          + env.scene.env_origins[1, :2])
    assert torch.allclose(phase.berth_point[1], torch.tensor(env.berth, dtype=torch.float64)
                          + env.scene.env_origins[1, :2])
    good = env.journal_phase_targets()
    mutations = ((lambda d: d.pop("berth_pose"), "lacks"), (lambda d: d.update(schema="x/0"), "schema"),
                 (lambda d: d.update(frame="world"), "frame"))
    for mutate, text in mutations:
        raw = dict(good)
        mutate(raw)
        broken = SimpleNamespace(journal_phase_targets=lambda raw=raw: raw, scene=env.scene)
        try:
            ch.phase_targets_world(broken)
        except RuntimeError as exc:
            assert text in str(exc), str(exc)
        else:
            raise AssertionError("a malformed phase-target dict must be refused: " + text)
    try:
        ch.phase_targets_world(SimpleNamespace(scene=env.scene))
    except RuntimeError as exc:
        assert "journal_phase_targets" in str(exc)
    else:
        raise AssertionError("an env without phase targets must be refused")


def test_complete_stage_holds_still_and_reset_clears_plans():
    env = FakeHarbourEnv("blueboat_v2", obstacle=False)
    controller = ch.StagedHarbourController(env, gains=jc.load_gains("blueboat_v2"), max_range_m=30.0,
                                            scale=1.0)
    controller.act()
    assert controller.tracker.plans[0] is not None
    env.complete[0] = True
    assert torch.equal(controller.act(), torch.zeros((1, 2), dtype=controller.act().dtype))
    controller.reset(torch.tensor([True]))
    assert controller.tracker.plans[0] is None and controller.planned_key[0] is None


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001 - report every failure
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if run_all() else 0)
