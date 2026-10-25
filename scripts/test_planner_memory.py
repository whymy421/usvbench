"""G25 / G24b CPU tests: ray-return memory, passed-return rule, circumscribed-radius ladder, certificate stamp. No Isaac.

* memory (scripts/planner_memory.py): a wall of cylinders seen obliquely by a moving observer is closed by the
  remembered returns where the current ray discs leave a hole the planner routes through; a static scene is
  never carved (S1 needs water on both sides of a remembered point); a moving disc leaves no ghost (its trail
  is carved by S1: "passed" = seen through on both sides); G24b: NO expiry (an entry no ray sees through stays for
  any number of steps, then S1 drops it exactly when rays pass on both sides); capacity (B) is exact and
  deterministic; reset is per env (partial resets);
* controller (classical_planner_pid.JournalMemoryPlannerController): updates every env's memory every control
  step from the journal estimate only, plans on current + remembered discs, clears exactly the done envs;
* ladder (journal_memory_ladder): first rung >= the circumscribed radius about the COM for both vessels, the
  BlueBoat and WAM-V values, journal_ladder unchanged (G17 / staged harbour freeze it);
* certificate: the evaluator's ray_memory block round-trips through JSON and passes the schema $def and
  validate_cert_v3.planner_memory_errors; every inconsistency is reported; run_cell declares the same name.

Run: python -B -m pytest -p no:cacheprovider scripts/test_planner_memory.py -q
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import planner_memory as pm  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
RAYS = 36


# ------------------------------------------------------------------ geometry helpers --
def cast(origin, yaw, centers, radii, max_range, count=RAYS):
    """Exact ray / circle ranges (numpy), ray k at yaw + k 2 pi / count from origin; max_range = no return."""
    origin = np.asarray(origin, dtype=np.float64)
    angles = yaw + np.arange(count) * (2.0 * math.pi / count)
    d = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    out = np.full(count, float(max_range))
    for c, r in zip(np.asarray(centers, dtype=np.float64).reshape(-1, 2), np.asarray(radii).reshape(-1)):
        m = origin - c
        b = d @ m
        disc = b * b - (m @ m - r * r)
        ok = disc >= 0.0
        t = -b - np.sqrt(np.where(ok, disc, 0.0))
        t = np.where(ok & (t >= 0.0), t, np.inf)
        out = np.minimum(out, t)
    return np.minimum(out, float(max_range))


def wall(x0, x1, y, radius, gap):
    xs = np.arange(x0, x1 + 1e-9, 2.0 * radius + gap)
    return np.stack((xs, np.full(xs.size, float(y))), axis=-1), np.full(xs.size, float(radius))


def surface_distance(points, centers, radii):
    if len(points) == 0:
        return np.zeros(0)
    return np.min(np.abs(np.linalg.norm(points[:, None, :] - centers[None], axis=2) - radii[None]), axis=1)


def memory(num_envs=1, cell=0.25, replan=60, scale=1.0, **rules):
    """BlueBoat-scale memory: range 30 m, u_ref 1.0 m/s, 60 Hz (G24b: no expiry horizon)."""
    r = dict(pm.RULE_CONSTANTS)
    r.update(rules)
    return pm.RayReturnMemory(num_envs, plan_cell_m=cell, replan_every=replan, length_scale=scale,
                              ray_max_range_m=30.0 * scale, reference_speed_mps=1.0, control_dt_s=1.0 / 60.0,
                              rules=r)


def ray_discs(origin, yaw, ranges, max_range):
    """journal_controllers.ray_obstacles in numpy (discs of radius range tan(pi / R) at the returns)."""
    angles = yaw + np.arange(ranges.size) * (2.0 * math.pi / ranges.size)
    hit = ranges < max_range * (1.0 - 1.0e-6)
    pts = np.asarray(origin) + ranges[:, None] * np.stack((np.cos(angles), np.sin(angles)), -1)
    return pts[hit], ranges[hit] * math.tan(math.pi / ranges.size)


def crosses_wall(path, y, x_lo, x_hi):
    for a, b in zip(path[:-1], path[1:]):
        if (a[1] - y) * (b[1] - y) <= 0.0 and abs(b[1] - a[1]) > 1e-12:
            x = a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            if x_lo <= x <= x_hi:
                return True
    return False


# ------------------------------------------------------------------ memory rules --
def _oblique_wall_run(with_memory):
    """BlueBoat-scale wall of r 1.2 m cylinders, 0.75 m gaps (< beam: impassable, rays see through), seen
    obliquely while the observer drives along y = 0 from x = -25 to x = 25; then plan from the final pose
    to a goal behind the wall at the circumscribed-radius inflation."""
    import classical_planner_pid as cp

    centers, radii = wall(-30.0, 30.0, 10.0, 1.2, 0.75)
    mem = memory()
    max_range = 30.0
    for step in range(1001):
        origin = np.array((-25.0 + 0.05 * step, 0.0))
        ranges = cast(origin, 0.0, centers, radii, max_range)
        mem.update(0, origin, 0.0, ranges, max_range)
    rc, rr = ray_discs(origin, 0.0, ranges, max_range)
    if with_memory:
        mc, mr = mem.discs(0)
        rc, rr = np.vstack((rc, mc)), np.concatenate((rr, mr))
    path = cp.plan_grid_path(origin, np.array((0.0, 20.0)), rc, rr, cell_m=0.25, inflation_m=0.794,
                             snap_start_m=1.5)
    return path, mem, centers, radii


def test_oblique_wall_hole_exists_without_memory_and_is_closed_with_it():
    path, _mem, _c, _r = _oblique_wall_run(with_memory=False)
    assert path is not None and crosses_wall(path, 10.0, -28.0, 28.0)      # the G22 defect: a hole in the wall
    path, mem, centers, radii = _oblique_wall_run(with_memory=True)
    assert path is None or not crosses_wall(path, 10.0, -28.0, 28.0)
    points, _ = mem.discs(0)
    assert len(points) > 100
    assert float(surface_distance(points, centers, radii).max()) < 1e-9    # remembered returns are surface points
    assert mem.counts["carved"] == 0                                        # a static wall is never carved


def test_static_scene_seen_from_everywhere_is_never_carved():
    """S1 static safety: a cylinder wall with sub-beam gaps plus free discs, seen from a full circle around
    them with a yawing fan (rays grazing every face and passing through every gap), is never carved."""
    rng = np.random.default_rng(3)
    centers, radii = wall(-20.0, 20.0, 12.0, 1.2, 0.75)
    free = []
    while len(free) < 8:
        c, r = rng.uniform(-12.0, 8.0, size=2), float(rng.uniform(0.6, 2.0))
        if (np.all(np.linalg.norm(centers - c, axis=1) > radii + r + 0.5)
                and all(np.linalg.norm(fc - c) > fr + r + 0.5 for fc, fr in free)):
            free.append((c, r))
    centers = np.vstack((centers, [c for c, _ in free]))
    radii = np.concatenate((radii, [r for _, r in free]))
    mem = memory()
    for step in range(3000):
        a = 2.0 * math.pi * step / 3000.0
        origin = np.array((30.0 * math.cos(a), 30.0 * math.sin(a)))
        assert np.all(np.linalg.norm(centers - origin, axis=1) > radii)       # the observer is in open water
        yaw = a + math.pi / 2.0 + 0.3 * math.sin(0.05 * step)
        mem.update(0, origin, yaw, cast(origin, yaw, centers, radii, 45.0), 45.0)
    assert mem.counts["carved"] == 0 and "expired" not in mem.counts and mem.counts["inserted"] > 1000
    points, _ = mem.discs(0)
    assert float(surface_distance(points, centers, radii).max()) < 1e-9


def _moving_disc_run(**rules):
    """A disc (D7-intruder-like, r 0.5) crosses ahead of a slowly moving, yawing observer for 400 steps and
    is then gone; ghost count = remembered returns farther than 5 cm from every current surface."""
    static_c, static_r = np.array([[30.0, 25.0]]), np.array([3.0])
    mem = memory(**rules)
    ghosts = []
    for step in range(1100):
        origin = np.array((0.02 * step, 0.0))
        yaw = 0.25 * math.sin(0.02 * step)
        if step < 400:
            disc = np.array((12.0, -8.0 + 0.04 * step))                    # 0.04 m / step = 2.4 m/s at 60 Hz
            centers, radii = np.vstack((static_c, disc)), np.concatenate((static_r, [0.5]))
        else:
            centers, radii = static_c, static_r
        mem.update(0, origin, yaw, cast(origin, yaw, centers, radii, 30.0), 30.0)
        points, _ = mem.discs(0)
        ghosts.append(int(np.sum(surface_distance(points, centers, radii) > 0.05)))
    cleared = next((k for k in range(400, len(ghosts)) if all(g == 0 for g in ghosts[k:])), None)
    return mem, ghosts, cleared


def test_moving_disc_leaves_no_ghost():
    mem, ghosts, cleared = _moving_disc_run()
    assert max(ghosts[:400]) > 0 and ghosts[400] > 0                     # there WAS a trail to clear
    assert not hasattr(mem, "ttl") and "ttl_sensor_crossings" not in mem.rules
    assert cleared is not None and cleared - 400 < 100, cleared         # carved in < 100 steps (G25: 35)
    assert mem.counts["carved"] > 0
    # G24b: without S1 nothing else removes the trail (no expiry): it survives to the end of the run
    mem0, ghosts0, cleared0 = _moving_disc_run(carve_margin_cells=1.0e9)
    assert mem0.counts["carved"] == 0 and cleared0 is None and ghosts0[-1] > 0, cleared0


def test_no_expiry_an_unseen_entry_persists_until_passed():
    """G24b (owner decision: no timeout expiry). An entry no ray sees through is kept for 20 x G25b's horizon
    (36000 steps at 1800 per sensor crossing); it is dropped exactly when S1 calls its object PASSED: a current ray
    runs past it on one side, beyond it by more than the carve margin, and a later ray does so on the other side."""
    centers, radii = np.array([[10.0, 0.0]]), np.array([1.0])
    mem = memory()
    mem.update(0, (0.0, 0.0), 0.0, cast((0.0, 0.0), 0.0, centers, radii, 30.0), 30.0)
    n0 = mem.size(0)
    assert n0 == 1                                                        # ray 0 returns (9, 0)
    point = mem.discs(0)[0][0].copy()
    nothing = np.full(RAYS, 30.0)
    away = np.array((0.0, -50.0))
    for _ in range(36000):                                               # no ray passes near the entry
        mem.update(0, away, 0.05, nothing, 30.0)
    assert mem.size(0) == n0 and mem.counts["carved"] == 0
    assert np.array_equal(mem.discs(0)[0][0], point)
    # one side: ray 0 passes 0.06 m to the right of the point (lateral <= carve_lateral 0.125 m), water beyond
    origin = np.array((0.0, 0.0))
    d = float(np.linalg.norm(point - origin))
    bearing = math.atan2(point[1] - origin[1], point[0] - origin[0])
    mem.update(0, origin, bearing + math.asin(0.06 / d), nothing, 30.0)
    assert mem.size(0) == n0                                              # one side is not enough (static safety)
    for _ in range(1000):                                                 # still no expiry in between
        mem.update(0, away, 0.05, nothing, 30.0)
    assert mem.size(0) == n0
    mem.update(0, origin, bearing - math.asin(0.06 / d), nothing, 30.0)  # the other side: passed -> dropped
    assert mem.size(0) == 0 and mem.counts["carved"] == n0
    # determinism: identical inputs -> identical arrays
    runs = []
    for _ in range(2):
        m = memory()
        for step in range(200):
            o = (0.1 * step, 0.0)
            m.update(0, o, 0.01 * step, cast(o, 0.01 * step, centers, radii, 30.0), 30.0)
        runs.append((m._keys[0].copy(), m._points[0].copy(), m._seen[0].copy()))
    assert all(np.array_equal(a, b) for a, b in zip(*runs))


def test_capacity_evicts_least_recently_seen_first():
    """B: a row of small discs swept by a sideways-moving observer overflows a 10-entry memory; the kept
    entries are always the most recently returned ones and stay sorted by bin key."""
    mem = memory(capacity_entries=10)
    reference = memory()                                                # unbounded twin
    centers = np.array([[x, 6.0] for x in range(-30, 31, 2)], dtype=np.float64)
    radii = np.full(len(centers), 0.3)
    for step in range(60):
        o = (-20.0 + 0.7 * step, 0.0)
        ranges = cast(o, 0.0, centers, radii, 30.0)
        mem.update(0, o, 0.0, ranges, 30.0)
        reference.update(0, o, 0.0, ranges, 30.0)
        assert mem.size(0) <= 10 and np.all(np.diff(mem._keys[0]) > 0)
        seen_all = np.sort(reference._seen[0])[::-1]
        if reference.size(0) > 10:
            assert mem._seen[0].min() >= seen_all[9]                    # nothing newer than a kept entry was dropped
    assert mem.counts["evicted"] > 0 and mem.peak_entries == 10


def test_reset_is_per_env():
    centers, radii = np.array([[10.0, 0.0]]), np.array([1.0])
    mem = memory(num_envs=3)
    for i in range(3):
        for _ in range(5):
            mem.update(i, (0.0, 0.0), 0.0, cast((0.0, 0.0), 0.0, centers, radii, 30.0), 30.0)
    sizes = [mem.size(i) for i in range(3)]
    assert min(sizes) > 0
    mem.reset([1])
    assert [mem.size(i) for i in range(3)] == [sizes[0], 0, sizes[2]]
    assert list(mem.step) == [5, 0, 5]


def test_derived_constants_scale_with_the_vessel():
    """Derived constants scale with s; G24b: no expiry horizon (no ttl constant, no ttl_steps)."""
    assert "ttl_sensor_crossings" not in pm.RULE_CONSTANTS and not hasattr(pm, "ttl_steps")
    for vessel in VESSELS:
        terms = pm.vessel_terms(vessel)
        scale = terms["length_scale"]
        assert terms["reference_speed_mps"] == 1.0 and terms["ray_max_range_m"] == 30.0 * scale
        cell = 0.25 * scale
        c = pm.derive_constants(pm.RULE_CONSTANTS, cell, 60, scale, terms["ray_max_range_m"],
                                terms["reference_speed_mps"], terms["control_dt_s"])
        assert c["quantum_m"] == pm.RULE_CONSTANTS["quantum_cells"] * cell
        assert c["disc_radius_m"] == pm.RULE_CONSTANTS["disc_radius_cells"] * cell
        assert c["carve_lateral_m"] == 0.125 * scale and c["carve_margin_m"] == 0.25 * scale
        assert "ttl_steps" not in c and c["capacity_entries"] == 8192
        # S1 static-safety bound (module docstring): l < 0.293 R for the smallest static journal radius 0.8 s
        assert c["carve_lateral_m"] < (1.0 - math.cos(math.pi / 4.0)) * 0.8 * scale
    with pytest.raises(ValueError):
        pm.derive_constants(pm.RULE_CONSTANTS, 0.0, 60, 1.0, 30.0, 1.0, 1.0 / 60.0)
    with pytest.raises(ValueError):
        pm.derive_constants(pm.RULE_CONSTANTS, 0.25, 60, 1.0, 30.0, 0.0, 1.0 / 60.0)


# ------------------------------------------------------------------ controller --
class FakeEnv:
    def __init__(self, n, max_range=30.0):
        import torch

        self.num_envs = n
        self.device = "cpu"
        self.control_step_s = 1.0 / 60.0
        self.scene = SimpleNamespace(env_origins=torch.zeros(n, 3))
        self.robot = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]] * n)))
        self.target_pos = torch.tensor([[40.0, 0.0]] * n)
        self.centers = np.array([[15.0, 0.0]])
        self.radii = np.array([2.0])
        self.max_range = max_range
        self.pos = np.zeros((n, 2))
        self.refresh()

    def refresh(self):
        import torch

        rays = np.stack([cast(p, 0.0, self.centers, self.radii, self.max_range) for p in self.pos])
        self.journal_estimate = SimpleNamespace(position=torch.as_tensor(self.pos, dtype=torch.float32),
                                                velocity=torch.zeros(self.num_envs, 3),
                                                rays=torch.as_tensor(rays, dtype=torch.float32))

    def __getattr__(self, name):
        import journal_controllers as jc
        if name in jc.FORBIDDEN_ENV_MEMBERS:
            raise AssertionError("memory planner read forbidden env member " + name)
        raise AttributeError(name)


def _controller(env):
    import classical_planner_pid as cp
    import journal_controllers as jc

    return cp.JournalMemoryPlannerController(env, replan_every=60, waypoint_spacing_m=2.0, accept_radius_m=1.0,
                                             plan_cell_m=0.25, ladder=cp.journal_memory_ladder("blueboat_v2"),
                                             gains=jc.load_gains("blueboat_v2"), max_range_m=env.max_range,
                                             vessel="blueboat_v2")


def test_controller_updates_memory_every_step_and_resets_only_done_envs():
    import torch

    env = FakeEnv(2)
    ctl = _controller(env)
    for step in range(5):
        env.pos[:, 0] = 0.1 * step
        env.refresh()
        action = ctl.act()
        assert action.shape == (2, 2)
    assert list(ctl.memory.step) == [5, 5] and ctl.memory.size(0) > 0 and ctl.memory.size(1) > 0
    centers, radii = ctl._obstacles_np(0)
    current = int(ctl._ray_active[0].sum())
    assert len(centers) == current + ctl.memory.size(0) and len(radii) == len(centers)
    ctl.reset(torch.tensor([False, True]))
    assert ctl.memory.size(0) > 0 and ctl.memory.size(1) == 0 and list(ctl.memory.step) == [5, 0]
    assert ctl.need_plan[1] and not ctl.need_plan[0]
    telemetry = ctl.telemetry()
    assert telemetry["legs_planned"] == sum(telemetry["legs_by_rung"]) + telemetry["unplannable_legs"]
    assert telemetry["resets"] == 1 and telemetry["updates"] == 10


def test_map_gives_current_returns_the_memory_disc_radius():
    """G25b M2: current returns are discs of the memory disc radius (no range * tan(pi/36) fattening)."""
    env = FakeEnv(1)
    ctl = _controller(env)
    ctl.act()
    centers, radii = ctl._obstacles_np(0)
    current = int(ctl._ray_active[0].sum())
    assert current > 0 and np.all(radii == ctl.memory.disc_radius)
    assert np.allclose(centers[:current], ctl._ray_centers[0][ctl._ray_active[0]])
    assert np.all(ctl._ray_radii[0][ctl._ray_active[0]] > ctl.memory.disc_radius)   # v2.5 discs were fat


def test_unplannable_leg_holds_instead_of_driving_a_straight_leg():
    """G25b fallback: a goal inside the inflated map (no rung plans) -> plan ends at the boat, env holds."""
    import torch
    import classical_planner_pid as cp
    import journal_controllers as jc

    env = FakeEnv(2)
    env.target_pos = torch.tensor([[20.0, 0.0], [40.0, 0.0]])
    env.centers = np.array([[20.8, 0.0], [15.0, 6.0]])                 # the goal of env 0 is 0.4 m behind a face
    env.radii = np.array([1.2, 2.0])
    env.refresh()
    ctl = _controller(env)
    action = ctl.act()
    assert ctl.unplannable_legs >= 1 and bool(ctl.truncated[0]) and not bool(ctl.truncated[1])
    assert len(ctl.plans[0]) == 1 and ctl.straight_fallbacks == 0
    assert torch.equal(action[0], torch.zeros(2)) and not torch.equal(action[1], torch.zeros(2))
    assert ctl.telemetry()["hold_steps"] == 1
    ctl.reset(torch.tensor([True, False]))
    assert not bool(ctl.truncated[0])
    # the v2.5 planner on the same inputs drives the straight leg into the obstacle
    base = cp.JournalPlannerLOSPIDController(env, replan_every=60, waypoint_spacing_m=2.0, accept_radius_m=1.0,
                                             plan_cell_m=0.25, ladder=cp.journal_memory_ladder("blueboat_v2"),
                                             gains=jc.load_gains("blueboat_v2"), max_range_m=30.0)
    base._obstacles_np = ctl._obstacles_np                              # same thin map, v2.5 fallback rule
    base._ray_centers = ctl._ray_centers
    base.act()
    assert base.straight_fallbacks >= 1 and len(base.plans[0]) > 1


def test_base_journal_planner_is_unchanged_without_memory():
    """JournalPlannerLOSPIDController keeps planning on the current ray discs only (v2.5 behaviour)."""
    import classical_planner_pid as cp
    import journal_controllers as jc

    env = FakeEnv(1)
    base = cp.JournalPlannerLOSPIDController(env, replan_every=60, waypoint_spacing_m=2.0, accept_radius_m=1.0,
                                             plan_cell_m=0.25, ladder=cp.journal_ladder("blueboat_v2"),
                                             gains=jc.load_gains("blueboat_v2"), max_range_m=30.0)
    base.act()
    centers, _ = base._obstacles_np(0)
    assert len(centers) == int(base._ray_active[0].sum()) and not hasattr(base, "memory")


# ------------------------------------------------------------------ ladder --
@pytest.mark.parametrize("vessel", VESSELS)
def test_memory_ladder_first_rung_covers_the_circumscribed_radius(vessel):
    import classical_planner_pid as cp
    import journal_controllers as jc

    record = jc.shared_module("journal_cfg_v2").vessel_record(vessel)
    footprint = np.asarray(record.geometry.footprint.vertices_m, dtype=np.float64)
    hull = float(np.linalg.norm(footprint, axis=1).max())               # COM-relative polygon (journal_cfg_v2)
    assert hull == pytest.approx(float(record.circumscribed_radius_m), abs=1e-12)
    ladder = cp.journal_memory_ladder(vessel)
    assert ladder[0] >= hull and ladder[1] == pytest.approx(hull, abs=1e-12)
    assert list(ladder) == sorted(ladder, reverse=True)
    terms = cp.journal_memory_ladder_terms(vessel)
    assert terms["circumscribed_radius_m"] == hull


def test_memory_ladder_values_and_journal_ladder_unchanged():
    import classical_planner_pid as cp

    assert np.allclose(cp.journal_memory_ladder("blueboat_v2"), (0.9939657, 0.7939657, 0.515), atol=1e-6)
    assert np.allclose(cp.journal_memory_ladder("wamv_vrx_v2_opt1"), (3.9871429, 3.0915599, 1.4675248), atol=1e-6)
    assert np.allclose(cp.journal_ladder("blueboat_v2"), (0.7790846, 0.5790846, 0.515), atol=1e-6)
    assert np.allclose(cp.journal_ladder("wamv_vrx_v2_opt1"), (2.2258849, 1.3303019, 1.4675248), atol=1e-6)


# ------------------------------------------------------------------ certificate --
# G25b: the certificate block tests (round trip, schema, pins, every REV-G25 tamper case) are in
# scripts/test_planner_memory_pins.py.


def test_run_cell_and_evaluator_declare_the_memory_controller():
    import run_cell

    assert run_cell.JOURNAL_CLASSICAL_BY_CARRIER["hazard_nav"]["controller_name"] == pm.CONTROLLER_NAME
    assert run_cell.JOURNAL_CLASSICAL_BY_CARRIER["path_hazard"]["controller_name"] == pm.CONTROLLER_NAME
    assert run_cell.JOURNAL_MEMORY_PLANNER["script"] == "classical_planner_pid.py"
    assert run_cell.CLASSICAL_BY_FAMILY["hazard_nav"]["controller_name"] == "planner_los_pid"   # legacy ids
    assert run_cell.JOURNAL_PLANNER["controller_name"] == pm.BASE_CONTROLLER_NAME
    import validate_cert_v3 as v
    assert v.PLANNER_MEMORY_CONTROLLERS == {pm.CONTROLLER_NAME: pm.BASE_CONTROLLER_NAME}
    assert v.PLANNER_MEMORY_FORMAT == pm.MEMORY_FORMAT and pm.MEMORY_VERSION in v.PLANNER_MEMORY_VERSIONS
    source = (SCRIPTS / "classical_planner_pid.py").read_text(encoding="ascii")
    tree = ast.parse(source)
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "JournalMemoryPlannerController" in calls and "journal_memory_ladder" in calls
    assert "controller_name = pm.CONTROLLER_NAME" in source


def test_windowed_occupancy_raster_is_bit_identical_to_the_oracle_raster():
    import classical_planner_pid as cp

    hg = cp.hazard_geometry()
    rng = np.random.default_rng(25)
    for trial in range(60):
        cell = float(rng.choice([0.25, 0.25 * 4.47791504231185, 0.5]))
        n = int(rng.integers(0, 400))
        centers = rng.uniform(-40.0, 40.0, size=(n, 2))
        radii = np.where(rng.random(n) < 0.5, 0.5 * cell, rng.uniform(0.0, 6.0, size=n))
        infl = float(rng.choice([0.0, 0.515, 1.4675248, 3.9871429]))
        x0 = np.floor(rng.uniform(-60, -30) / cell) * cell
        y0 = np.floor(rng.uniform(-60, -30) / cell) * cell
        xs = np.arange(x0, x0 + rng.uniform(30, 120) + 0.5 * cell, cell)
        ys = np.arange(y0, y0 + rng.uniform(30, 120) + 0.5 * cell, cell)
        a = hg._occupancy_grid(centers, radii, xs, ys, infl)
        b = cp.occupancy_grid_windowed(centers, radii, xs, ys, infl)
        assert a.shape == b.shape and np.array_equal(a, b), trial
    # discs centred exactly on cell centres and touching cells at exactly the inflated radius
    xs = ys = np.arange(-5.0, 5.0 + 0.125, 0.25)
    centers = np.array([[0.0, 0.0], [0.25, -1.0], [-4.75, 4.75], [6.0, 0.0]])
    radii = np.array([0.5, 0.25, 0.125, 0.75])
    for infl in (0.0, 0.25, 1.0):
        assert np.array_equal(hg._occupancy_grid(centers, radii, xs, ys, infl),
                              cp.occupancy_grid_windowed(centers, radii, xs, ys, infl))


def test_candidate_ray_carving_equals_testing_every_ray():
    """S1 tests only the rays within 15 deg of a far point's bearing; the decisions equal testing all 36."""
    class AllRays(pm.RayReturnMemory):
        CANDIDATE_ANGLE = 1.0e-12                                        # every point is "near": all rays

    rng = np.random.default_rng(7)
    static_c = rng.uniform(-25.0, 25.0, size=(10, 2))
    static_r = rng.uniform(0.8, 3.0, size=10)
    fast = memory()
    full = AllRays(1, plan_cell_m=0.25, replan_every=60, length_scale=1.0, ray_max_range_m=30.0,
                   reference_speed_mps=1.0, control_dt_s=1.0 / 60.0)
    for step in range(1500):
        movers = np.array([[10.0 * math.cos(0.004 * step), 8.0 * math.sin(0.006 * step)],
                           [-12.0 + 0.02 * step, -14.0]])
        centers = np.vstack((static_c, movers))
        radii = np.concatenate((static_r, [0.6, 1.0]))
        origin = np.array((3.0 * math.sin(0.003 * step), 2.0 * math.cos(0.002 * step)))
        if np.any(np.linalg.norm(centers - origin, axis=1) <= radii):
            continue
        yaw = 0.7 * math.sin(0.01 * step)
        ranges = cast(origin, yaw, centers, radii, 30.0)
        fast.update(0, origin, yaw, ranges, 30.0)
        full.update(0, origin, yaw, ranges, 30.0)
        assert np.array_equal(fast._keys[0], full._keys[0]) and np.array_equal(fast._side[0], full._side[0]), step
    assert fast.counts == full.counts and fast.counts["carved"] > 50
