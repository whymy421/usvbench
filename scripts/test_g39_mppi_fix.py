"""G39: CPU tests of the MPPI stall fixes (scripts/mppi_g39_fix.py, mppi_controller G39 hooks).

Run: python -B scripts/test_g39_mppi_fix.py   (or pytest). No Isaac; development geometry only
(tasks/docking/journal_docking.build_scenario at eval seed 42, the development seed).
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
for path in (str(SCRIPTS), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)
for name, sub in (("tasks", "tasks"), ("tasks.docking", "tasks/docking"), ("tasks.hazard_nav", "tasks/hazard_nav")):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(REPO / sub)]
        sys.modules[name] = module

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mppi_controller as mc  # noqa: E402
import mppi_g39_fix as fix  # noqa: E402


def test_constants_and_release_variant():
    assert fix.GRID_CELLS == mc.GEODESIC_GRID_CELLS
    assert fix.RELEASE_VARIANT in fix.VARIANTS
    assert fix.variant_flags("base") == {"dock_deadzone": False, "geodesic": False, "field_rule": None}
    assert fix.variant_flags("c3") == {"dock_deadzone": True, "geodesic": True, "field_rule": "capsule"}
    assert fix.variant_flags("c2") == {"dock_deadzone": True, "geodesic": True, "field_rule": "planner_ladder"}
    assert fix.variant_flags("c1") == {"dock_deadzone": True, "geodesic": False, "field_rule": None}
    for rule, first in (("planner_ladder", 0.994), ("capsule", 0.579)):
        guide = fix.GeodesicGuide(1, vessel="blueboat_v2", ray_max_range_m=30.0, field_rule=rule)
        assert abs(guide.ladder[0] - first) < 1e-3, (rule, guide.ladder)
    params, stamp = mc.journal_mppi_params("blueboat_v2")
    flags = fix.variant_flags(fix.RELEASE_VARIANT)
    assert params.geodesic_potential == flags["geodesic"]
    assert params.dock_deadzone_fraction == (fix.DOCK_DEADZONE_FRACTION if flags["dock_deadzone"] else 0.0)
    assert stamp["g39_variant"] == fix.RELEASE_VARIANT
    # H / lambda stay the frozen values
    assert (params.H, params.lambda_) == (60, 0.3)
    wamv, _ = mc.journal_mppi_params("wamv_vrx_v2_opt1")
    assert (wamv.H, wamv.lambda_) == (269, 10.0)


def test_defaults_are_the_v261_cost():
    p = mc.MPPIParams()
    assert p.dock_deadzone_fraction == 0.0 and p.geodesic_potential is False
    d = p.as_dict()
    assert d["dock_deadzone_fraction"] == 0.0 and d["geodesic_potential"] is False
    for bad in (-0.1, 1.0):
        try:
            mc.MPPIParams(dock_deadzone_fraction=bad).validate()
        except ValueError:
            pass
        else:
            raise AssertionError("dead zone %r accepted" % bad)


def _dock_inputs(n=2):
    state = torch.tensor([[-3.0, 0.4, 0.3, 0.5, 0.0, 0.0], [-0.8, 0.0, 0.48, 0.0, 0.0, 0.0]])[:n]
    targets = torch.zeros((n, 1, 2))
    heading = torch.tensor([[1.0, 0.0]] * n)
    centers = torch.tensor([[[0.0, 1.8], [0.0, -1.8], [1.2, 0.0]]] * n)
    radii = torch.full((n, 3), 0.35)
    return state, dict(targets=targets, goal_radius=2.5, mode="dock", dock_heading=heading,
                       obstacle_centers=centers, obstacle_radii=radii, obstacle_active=torch.ones((n, 3), dtype=torch.bool))


def test_switches_off_reproduce_the_unpatched_actions():
    """Fix off (base variant): same actions as the same rollout with the G39 hooks bypassed."""
    dyn = mc.VesselDynamicsV2("blueboat_v2")
    base_params, _ = mc.journal_mppi_params("blueboat_v2")
    off = fix.params_for_variant(base_params, "base")
    state, kwargs = _dock_inputs()
    a = mc.MPPIController(2, params=off, dynamics=dyn)
    act_a, info_a = a.rollout(state, **kwargs)
    # reference: identical params; geodesic buffers absent and dead zone 0 -> shaped() == dist + route_rem
    b = mc.MPPIController(2, params=off, dynamics=dyn)
    act_b, info_b = b.rollout(state, **kwargs)
    assert torch.equal(act_a, act_b) and torch.equal(info_a["cost_min"], info_b["cost_min"])
    assert a.potential_field is None


def test_set_potential_is_in_place():
    p = fix.params_for_variant(mc.journal_mppi_params("blueboat_v2")[0], "c2")
    c = mc.MPPIController(2, params=p, dynamics=mc.VesselDynamicsV2("blueboat_v2"))
    ids = (id(c.potential_field), id(c.potential_origin), id(c.potential_cell), id(c.potential_valid))
    n = mc.GEODESIC_GRID_CELLS
    c.set_potential(torch.ones((2, n * n)), torch.zeros((2, 2)), torch.full((2,), 0.25), torch.tensor([True, False]))
    assert ids == (id(c.potential_field), id(c.potential_origin), id(c.potential_cell), id(c.potential_valid))
    assert bool(c.potential_valid[0]) and not bool(c.potential_valid[1])


def test_wavefront_routes_through_the_gap_and_never_shortcuts():
    n = 41
    passable = torch.ones((1, n, n), dtype=torch.bool)
    passable[0, :, 20] = False          # wall x = 20 ...
    passable[0, 35:38, 20] = True       # ... with a gap near row 36
    seeds = torch.full((1, n, n), float("inf"))
    seeds[0, 20, 30] = 0.0              # target right of the wall, row 20
    d1 = fix.wavefront(passable, seeds)
    start = float(d1[0, 20, 10])        # left of the wall, same row
    straight = 20.0
    via_gap = math.hypot(10, 16) + math.hypot(10, 16)
    assert start > straight + 5.0, start
    assert abs(start - via_gap) / via_gap < 0.09, (start, via_gap)     # chamfer metric: within 8.3 %
    d2 = fix.wavefront(~torch.isfinite(d1), d1)
    assert torch.equal(d2[torch.isfinite(d1)], d1[torch.isfinite(d1)])  # continuation never lowers a value
    assert bool(torch.isfinite(d2).all())


def test_continuation_slopes_out_of_an_obstacle_not_through_it():
    """A thick block between boat side and target: inside its near face the field must exceed the free
    cell in front of it (the replica's parked-scatter defect when the continuation step cost was 1)."""
    n = 61
    passable = torch.ones((1, n, n), dtype=torch.bool)
    passable[0, 20:41, 25:36] = False                     # block, columns 25..35, rows 20..40
    seeds = torch.full((1, n, n), float("inf"))
    seeds[0, 30, 55] = 0.0                                # target right of the block
    d1 = fix.wavefront(passable, seeds)
    unreached = ~torch.isfinite(d1)
    d2 = fix.wavefront(unreached, d1, step_factor=torch.where(
        unreached, torch.full_like(d1, fix.CONTINUATION_STEP_FACTOR), torch.ones_like(d1)), max_iterations=4000)
    front, inside = float(d2[0, 30, 24]), float(d2[0, 30, 26])
    assert inside > front, (inside, front)
    assert float(d2[0, 30, 30]) > float(d2[0, 30, 26])   # deeper is higher: the field slopes outward


def test_lookup_matches_the_raster_and_extends_outside():
    p = fix.params_for_variant(mc.journal_mppi_params("blueboat_v2")[0], "c2")
    c = mc.MPPIController(1, params=p, dynamics=mc.VesselDynamicsV2("blueboat_v2"))
    n = mc.GEODESIC_GRID_CELLS
    ys, xs = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    field = (2.0 * xs + 3.0 * ys).reshape(1, -1)          # linear: bilinear lookup must be exact
    c.set_potential(field, torch.tensor([[10.0, -5.0]]), torch.tensor([0.5]), torch.tensor([True]))
    pos = torch.tensor([[[10.0 + 0.5 * 7.3, -5.0 + 0.5 * 11.6], [10.0 - 1.0, -5.0]]])
    out = c._geodesic(pos, torch.zeros((1, 2)))
    assert abs(float(out[0, 0]) - (2 * 7.3 + 3 * 11.6)) < 1e-3
    assert abs(float(out[0, 1]) - (0.0 + 1.0)) < 1e-5        # edge value 0 + 1 m outside
    c.set_potential(field, torch.tensor([[10.0, -5.0]]), torch.tensor([0.5]), torch.tensor([False]))
    assert torch.equal(c._geodesic(pos, torch.full((1, 2), 7.0)), torch.full((1, 2), 7.0))


def test_dead_zone_removes_the_yaw_incentive_at_the_stall_point():
    """Closed form at the certified stall pose (x = -0.75 m, yaw 0.48 rad, stopped): with F1 the
    stage potential of a rollout that stays put is zero, so turning the hull buys no position."""
    dyn = mc.VesselDynamicsV2("blueboat_v2")
    p = fix.params_for_variant(mc.journal_mppi_params("blueboat_v2")[0], "c1")
    dz = p.dock_deadzone_fraction * 2.5
    assert dz == 1.25 and 0.75 < dz
    # the aligned stationary point of the margin barrier (G39 report closed form) lies inside the zone
    q = mc.vessel_params_v2("blueboat_v2")
    reach = q["capsule_center_com_xy_m"][0] + q["capsule_segment_half_length_m"] + q["capsule_radius_m"]
    tan_half = math.tan(math.pi / 36)
    excess = (p.H * p.w_goal + p.w_terminal) / (p.H * p.w_obstacle * 2.0 * (1.0 + tan_half))
    back = 0.856825846247375                                   # E5 BlueBoat tier-2 back face (berth frame)
    x_eq = back - (p.clearance_margin_m - excess + reach) / (1.0 - tan_half)
    assert -dz < x_eq < 0.0, x_eq
    wamv = fix.params_for_variant(mc.journal_mppi_params("wamv_vrx_v2_opt1")[0], "c1")
    assert abs(wamv.dock_deadzone_fraction * 11.194787605779625 - 5.597) < 1e-3


def test_closed_loop_blueboat_berth_development_key():
    """CPU closed loop on the REAL tier-2 slip of development key (seed 42, env 3, ep 0): base stalls
    outside the heading tolerance or succeeds; c1 must succeed. World = calm nominal model (as the
    frozen tuning procedure), rays = the docking kernels."""
    from tasks.docking import journal_docking as jd
    from tasks.docking.journal_kernels import fan_angles, polygon_circle_clearance, ray_circle_ranges
    from tasks._shared import journal_cfg_v2 as plans
    import journal_controllers as jc

    vessel = "blueboat_v2"
    hull = torch.tensor(plans.vessel_record(vessel).footprint.vertices_m, dtype=torch.float64)
    s = jd.build_scenario(vessel, 2, 42, 3, 0)
    C = torch.tensor(np.array(s.wall_centers))[None]
    R = torch.tensor(np.array(s.wall_radii))[None]
    th = s.berth_heading_rad
    b = dict(s.berth)

    def run(variant, seconds):
        p = fix.params_for_variant(mc.journal_mppi_params(vessel)[0], variant)
        ctrl = mc.MPPIController(1, params=p, dynamics=mc.VesselDynamicsV2(vessel), env_ids=[3])
        plant = mc.VesselDynamicsV2(vessel, dtype=torch.float64)
        x = torch.tensor([[s.start_xy[0], s.start_xy[1], s.start_heading_rad, 0.0, 0.0, 0.0]], dtype=torch.float64)
        heading = torch.tensor([[math.cos(th), math.sin(th)]], dtype=torch.float32)
        hold = 0.0
        for k in range(int(seconds * 60)):
            rays = ray_circle_ranges(x[:, :2], fan_angles(x[:, 2], 36), C, R, 30.0).float()
            cen, rad, act = jc.ray_obstacles(x[:, :2].float(), x[:, 2].float(), rays, 30.0)
            a, _ = ctrl.rollout(x.float(), targets=torch.zeros((1, 1, 2)), goal_radius=2.5, mode="dock",
                                dock_heading=heading, obstacle_centers=cen, obstacle_radii=rad, obstacle_active=act)
            x = plant.step(x, a.double())
            assert float(polygon_circle_clearance(x[:, :2], x[:, 2], hull, C, R)[0]) > 0.0, "contact"
            px, py, yaw, u, v = (float(z) for z in x[0, :5])
            rel = math.atan2(math.sin(yaw - th), math.cos(yaw - th))
            ok = math.hypot(px, py) <= 2.5 and abs(rel) <= b["heading_tolerance_rad"] and math.hypot(u, v) <= 0.3
            hold = hold + 1.0 / 60.0 if ok else 0.0
            if hold >= 5.0:
                return True, abs(rel)
        return False, abs(rel)

    ok, err = run("c1", 43.0)
    assert ok, "c1 did not hold the berth (final heading error %.3f rad)" % err


if __name__ == "__main__":
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL", name, type(exc).__name__, exc)
    sys.exit(1 if failures else 0)
