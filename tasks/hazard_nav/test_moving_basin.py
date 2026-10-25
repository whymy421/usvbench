# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the P5 moving-obstacle basin (layer three).

WHAT THE TASK IS
----------------
A closed basin with three segments along the track:

  (1) two staggered rows of cylinders sliding sideways  -- tests WHEN to go
  (buffer) 10 m of empty water, so the two segments can be attributed apart
  (2) one cylinder orbiting a fixed centre at constant rate -- tests
      anticipating circular motion
  (3) a clear run-out to the goal

Both segments express their dose as a fraction of the boat's own top speed,
mu = u / v_max for the rows and nu = omega * R / v_max for the orbit, so the
same rung number means the same physical demand in either segment.

WHY THESE TESTS AND NOT OTHERS
------------------------------
The appearance axis was certified for two days on an argument whose direction
had been reversed, and the defect survived because the fairness reasoning
lived in a comment rather than in an assertion.  So every fairness claim this
task makes is executed here:

* the transition dose mu* below which no lateral tracking is needed is
  RECOMPUTED, not quoted, and the ladder is asserted to straddle it;
* the constructive strategy above mu* is simulated arithmetically at every
  rung rather than asserted to exist;
* the wait-then-cross bound for the orbit is recomputed and the top rung is
  asserted to sit well inside it;
* feasibility is checked at sampled times ACROSS a period, because a moving
  field can be open when sampled and sealed a moment later -- a failure mode
  no static sampler can have;
* rung 0 is asserted to be genuinely static, which is what makes it a valid
  parent baseline.

HOW IT RUNS WITHOUT ISAAC
-------------------------
Every layout sampler in this tree is Isaac-free by design, and so is this one.
The module is loaded by path, exactly as the other CPU suites do it.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _directory in (_HERE, _HERE.parent / "_shared"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

try:
    from .hazard_geometry import (  # noqa: F401
        FORCED_SEAL_CELL_M,
        HALF_BEAM_M,
        HULL_BEAM_M,
        KIND_ORBIT,
        KIND_STATIC,
        KIND_TRANSLATE,
        MIN_OBSTACLE_RADIUS_M,
        MOVING_ADMISSION_SAMPLES,
        MOVING_APPROACH_M,
        MOVING_BUFFER_M,
        MOVING_DOSE_LADDER,
        MOVING_ENDPOINT_CLEAR_M,
        MOVING_GAP_BEAMS,
        MOVING_HALF_WIDTH_RANGE_M,
        MOVING_MAX_CYLINDERS,
        MOVING_ORBIT_RADIUS_M,
        MOVING_ROW_SEPARATION_M,
        MOVING_TOP_SPEED_MS,
        bfs_geodesic_length,
        moving_basin_positions,
        moving_basin_tracking_feasible,
        moving_orbit_window_bound,
        moving_row_period_m,
        moving_transition_mu,
        rotate_moving_params,
        sample_moving_basin_layout,
        start_goal_disks_clear,
    )
except ImportError:  # direct execution
    from hazard_geometry import (  # type: ignore  # noqa: F401
        FORCED_SEAL_CELL_M,
        HALF_BEAM_M,
        HULL_BEAM_M,
        KIND_ORBIT,
        KIND_STATIC,
        KIND_TRANSLATE,
        MIN_OBSTACLE_RADIUS_M,
        MOVING_ADMISSION_SAMPLES,
        MOVING_APPROACH_M,
        MOVING_BUFFER_M,
        MOVING_DOSE_LADDER,
        MOVING_ENDPOINT_CLEAR_M,
        MOVING_GAP_BEAMS,
        MOVING_HALF_WIDTH_RANGE_M,
        MOVING_MAX_CYLINDERS,
        MOVING_ORBIT_RADIUS_M,
        MOVING_ROW_SEPARATION_M,
        MOVING_TOP_SPEED_MS,
        bfs_geodesic_length,
        moving_basin_positions,
        moving_basin_tracking_feasible,
        moving_orbit_window_bound,
        moving_row_period_m,
        moving_transition_mu,
        rotate_moving_params,
        sample_moving_basin_layout,
        start_goal_disks_clear,
    )

GAP_M = MOVING_GAP_BEAMS * HULL_BEAM_M
PERIOD_M = moving_row_period_m(GAP_M)

ENV_PATH = _HERE / "hazard_nav_env.py"
CFG_PATH = _HERE / "hazard_nav_env_cfg.py"
INIT_PATH = _HERE / "__init__.py"
MOVING_ID = "Isaac-USV-HazardMovingBasin-Direct-v1"


# --------------------------------------------------------------- AST lifting --
def _env_method_ast(name: str) -> ast.FunctionDef:
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HazardNavEnv":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"hazard_nav_env.py: no HazardNavEnv.{name}")


def _lift_env_method(name: str):
    """Compile one SHIPPED HazardNavEnv method into a callable."""
    item = _env_method_ast(name)
    module = ast.Module(body=[item], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "torch": torch,
        "math": math,
        "KIND_TRANSLATE": KIND_TRANSLATE,
        "KIND_ORBIT": KIND_ORBIT,
        "moving_basin_positions": moving_basin_positions,
    }
    exec(compile(module, str(ENV_PATH), "exec"), namespace)  # noqa: S102
    return namespace[name]


class _Fake:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _moving_env(mu, nu, *, num_envs=3, angle=0.0, steps=(0, 0, 0), seed=7):
    """A fabricated env carrying exactly what _update_moving_obstacles reads."""
    layout, spec = _sample(mu, nu, seed=seed)
    count = layout.obstacle_count
    slots = count + 4
    cosine, sine = math.cos(angle), math.sin(angle)
    rot = np.array(((cosine, -sine), (sine, cosine)))
    base_local = layout.centers @ rot.T
    params = rotate_moving_params(spec.params, spec.kind, angle)

    origins = torch.tensor([[100.0, -50.0], [-30.0, 40.0], [0.0, 0.0]])[:num_envs]
    kind_t = torch.zeros((num_envs, slots), dtype=torch.long)
    params_t = torch.zeros((num_envs, slots, 7))
    base_t = torch.zeros((num_envs, slots, 2))
    for env in range(num_envs):
        kind_t[env, :count] = torch.as_tensor(spec.kind, dtype=torch.long)
        params_t[env, :count] = torch.as_tensor(params, dtype=torch.float32)
        base_t[env, :count] = torch.as_tensor(base_local, dtype=torch.float32)

    fake = _Fake(
        num_envs=num_envs,
        control_step_s=1.0 / 60.0,
        episode_length_buf=torch.tensor(steps[:num_envs], dtype=torch.long),
        _moving_kind=kind_t,
        _moving_params=params_t,
        _moving_base=base_t,
        obstacle_centers=torch.zeros((num_envs, slots, 2)),
        scene=_Fake(env_origins=origins),
        _obstacle_translate_ops=[],
    )
    # The display refresh is render-only and is exercised by its own test; here
    # it must exist so the lifted method can call it, and do nothing.
    fake._refresh_moving_markers = lambda _local: None
    return fake, layout, spec, count, base_local, params, origins



def _sample(mu, nu, seed=7):
    return sample_moving_basin_layout(0, np.random.default_rng(seed), mu=mu, nu=nu)


# ------------------------------------------------------------------ geometry --
def test_gap_width_is_three_hull_beams_and_the_period_follows():
    assert abs(GAP_M - 2.697) < 1.0e-3, f"gap is {GAP_M}, the design says 2.697 m"
    assert abs(PERIOD_M - (GAP_M + 2.0 * MIN_OBSTACLE_RADIUS_M)) < 1.0e-12, (
        "the row period must be the free gap plus two radii, or the measured "
        "opening is not the opening the difficulty rating names"
    )


def test_the_transition_dose_is_recomputed_not_quoted():
    """mu* is where passing without any lateral tracking stops working."""
    mu_star = moving_transition_mu(GAP_M)
    depth = 2.0 * MIN_OBSTACLE_RADIUS_M + 2.0 * HALF_BEAM_M
    margin = GAP_M - HULL_BEAM_M
    assert abs(mu_star - margin / (2.0 * depth)) < 1.0e-12
    assert abs(mu_star - 0.360) < 0.002, (
        f"mu* is {mu_star:.4f}; the design derives 0.360 and the whole ladder "
        "is placed around it"
    )
    assert abs(mu_star * MOVING_TOP_SPEED_MS - 1.076) < 0.005


def test_the_ladder_straddles_the_transition():
    """The mistake the appearance axis made was living in ONE regime."""
    below = [m for m in MOVING_DOSE_LADDER if 0.0 < m < moving_transition_mu(GAP_M)]
    above = [m for m in MOVING_DOSE_LADDER if m > moving_transition_mu(GAP_M)]
    assert below, "no rung below the transition: the ladder never tests timing alone"
    assert above, "no rung above the transition: the ladder never tests tracking"
    assert any(abs(m - moving_transition_mu(GAP_M)) < 0.005 for m in MOVING_DOSE_LADDER), (
        "the transition itself must be a rung, otherwise the mechanism change "
        "is inferred from two sides rather than measured at the point"
    )


def test_the_ladder_is_strictly_monotone():
    assert list(MOVING_DOSE_LADDER) == sorted(MOVING_DOSE_LADDER)
    assert len(set(MOVING_DOSE_LADDER)) == len(MOVING_DOSE_LADDER)
    assert MOVING_DOSE_LADDER[0] == 0.0, "rung 0 must be the undosed parent"
    assert MOVING_DOSE_LADDER[-1] < 1.0, (
        "a dose at or above 1.0 makes the gap outrun the boat, which is "
        "unsolvable rather than hard"
    )


# --------------------------------------------------------------- feasibility --
def test_every_rung_has_a_constructive_pass_through_a_row():
    """The fairness argument is executed, not asserted in a comment."""
    for mu in MOVING_DOSE_LADDER:
        ok, crossing_s, reversal_s = moving_basin_tracking_feasible(mu, GAP_M)
        assert ok, (
            f"rung mu={mu} has no constant-heading pass: crossing takes "
            f"{crossing_s:.3f} s but the row reverses every {reversal_s:.3f} s"
        )
        if mu > 0.0:
            assert crossing_s < reversal_s


def test_the_top_rung_is_tight_but_not_impossible():
    """Named so a later reader does not mistake the margin for comfort."""
    ok, crossing_s, reversal_s = moving_basin_tracking_feasible(0.80, GAP_M)
    assert ok
    window = reversal_s - crossing_s
    assert 0.2 < window < 0.8, (
        f"the top rung window is {window:.3f} s; the design expects it to be "
        "demanding, around four tenths of a second, and a large window would "
        "mean the top rung is not actually testing anything"
    )


def test_the_orbit_bound_is_recomputed_and_the_ladder_sits_inside_it():
    bound = moving_orbit_window_bound()
    crossing = 2.0 * MOVING_ORBIT_RADIUS_M / MOVING_TOP_SPEED_MS
    assert abs(bound - (math.pi / crossing) * MOVING_ORBIT_RADIUS_M / MOVING_TOP_SPEED_MS) < 1.0e-12
    assert abs(bound - 1.5708) < 0.002
    assert max(MOVING_DOSE_LADDER) < 0.55 * bound, (
        "the top orbit rung should sit around half the bound so that a "
        "wait-then-cross window always exists with room to spare"
    )


def test_a_dose_outside_the_bounds_is_refused_not_clamped():
    """A clamp would run one dose and certify another. That bug cost two days."""
    for bad in (1.0, 1.5):
        try:
            _sample(bad, 0.0)
        except ValueError as exc:
            assert "mu" in str(exc)
        else:
            raise AssertionError(f"mu={bad} was accepted")
    try:
        _sample(0.0, moving_orbit_window_bound() + 0.1)
    except ValueError as exc:
        assert "nu" in str(exc)
    else:
        raise AssertionError("an unsolvable orbit rate was accepted")


# -------------------------------------------------------------------- motion --
def test_positions_at_time_zero_reproduce_the_sampled_layout():
    """Anything else means the recorded layout is not what the episode starts from."""
    for mu in (0.0, 0.36, 0.80):
        layout, spec = _sample(mu, mu)
        at_zero = moving_basin_positions(layout.centers, spec.kind, spec.params, 0.0)
        assert np.allclose(at_zero, layout.centers, atol=1.0e-12), (
            f"mu={mu}: the layout centres are not the t=0 positions"
        )


def test_rung_zero_is_genuinely_static():
    """It is the parent baseline, so it must not move at all, ever."""
    layout, spec = _sample(0.0, 0.0)
    assert not spec.moves
    assert np.all(spec.kind == KIND_STATIC)
    for t in (0.0, 3.7, 61.0, 119.9):
        moved = moving_basin_positions(layout.centers, spec.kind, spec.params, t)
        assert np.array_equal(moved, layout.centers), f"something moved at t={t}"


def test_a_cylinder_never_teleports():
    """A wrapping saw would jump at the wrap and break contact continuity."""
    layout, spec = _sample(0.80, 0.80)
    dt = 1.0 / 60.0
    previous = moving_basin_positions(layout.centers, spec.kind, spec.params, 0.0)
    worst = 0.0
    for step in range(1, 2000):
        now = moving_basin_positions(
            layout.centers, spec.kind, spec.params, step * dt
        )
        worst = max(worst, float(np.abs(now - previous).max()))
        previous = now
    # One control step at the top dose moves a row by u*dt and the orbit by
    # omega*R*dt; both are mu*v_max*dt = 0.80 * 2.991 / 60 = 0.0399 m.
    limit = 1.5 * 0.80 * MOVING_TOP_SPEED_MS * dt
    assert worst < limit, (
        f"largest single-step jump is {worst:.4f} m against a kinematic limit "
        f"of {limit:.4f} m, so something teleports"
    )


def test_the_orbit_stays_on_its_circle():
    layout, spec = _sample(0.0, 0.55)
    idx = np.nonzero(spec.kind == KIND_ORBIT)[0]
    assert idx.size == 1
    for t in np.linspace(0.0, 120.0, 97):
        moved = moving_basin_positions(layout.centers, spec.kind, spec.params, t)
        r = float(np.linalg.norm(moved[idx[0]] - spec.orbit_center))
        assert abs(r - MOVING_ORBIT_RADIUS_M) < 1.0e-9


def test_each_row_sweeps_exactly_one_spatial_period():
    """Less and some y is never open; more and the comb overshoots pointlessly.

    Asserted analytically from the stored bounds, because a sampled sweep can
    only reach within one sampling step of the turning point and would give a
    misleading near miss.
    """
    layout, spec = _sample(0.55, 0.0)
    rows = np.nonzero(spec.kind == KIND_TRANSLATE)[0]
    assert rows.size > 0
    for i in rows:
        span = float(spec.params[i, 6])
        assert abs(span - PERIOD_M) < 1.0e-12, (
            f"cylinder {i} sweeps {span:.6f} m, the period is {PERIOD_M:.6f} m"
        )
        direction = spec.params[i, 2:4]
        assert abs(float(np.linalg.norm(direction)) - 1.0) < 1.0e-12, (
            f"cylinder {i} has a non unit motion direction {direction}"
        )
    # And the sampled motion really does visit both ends, to within the step.
    one = rows[0]
    step_s = 4.0 * PERIOD_M / spec.translate_speed_ms / 400.0
    tol = spec.translate_speed_ms * step_s
    ys = [
        float(moving_basin_positions(layout.centers, spec.kind, spec.params, t)[one, 1])
        for t in np.linspace(0.0, 4.0 * PERIOD_M / spec.translate_speed_ms, 401)
    ]
    assert abs((max(ys) - min(ys)) - PERIOD_M) < tol


# ----------------------------------------------------------------- admission --
def test_the_field_is_feasible_across_a_whole_period_not_just_at_zero():
    """The failure mode a static sampler cannot have."""
    for mu in (0.24, 0.55, 0.80):
        layout, spec = _sample(mu, mu)
        period = 2.0 * PERIOD_M / spec.translate_speed_ms
        for t in np.linspace(0.0, period, MOVING_ADMISSION_SAMPLES + 1):
            moved = moving_basin_positions(
                layout.centers, spec.kind, spec.params, float(t)
            )
            g = bfs_geodesic_length(
                layout.start, layout.goal, moved, layout.radii,
                cell_m=FORCED_SEAL_CELL_M,
            )
            assert g is not None, f"mu={mu} sealed at t={t:.2f} s"


def test_the_endpoints_stay_clear_at_every_time():
    for mu in (0.0, 0.36, 0.80):
        layout, spec = _sample(mu, mu)
        for t in np.linspace(0.0, 60.0, 61):
            moved = moving_basin_positions(
                layout.centers, spec.kind, spec.params, float(t)
            )
            assert start_goal_disks_clear(
                layout.start, layout.goal, moved, layout.radii,
                clear_radius_m=MOVING_ENDPOINT_CLEAR_M, inflation_m=0.0,
            ), f"mu={mu}: an endpoint disk was invaded at t={t:.1f} s"


def test_the_rows_seal_against_the_basin_wall_at_every_phase():
    """Otherwise the boat slips round the end of the comb, which the difficulty
    rating does not account for."""
    layout, spec = _sample(0.80, 0.0)
    rows = np.nonzero(spec.kind == KIND_TRANSLATE)[0]
    xs = sorted({round(float(layout.centers[i, 0]), 6) for i in rows})
    assert len(xs) == 2, f"expected two rows, found x positions {xs}"
    period = 2.0 * PERIOD_M / spec.translate_speed_ms
    for t in np.linspace(0.0, period, 25):
        moved = moving_basin_positions(
            layout.centers, spec.kind, spec.params, float(t)
        )
        for x in xs:
            band = [i for i in rows if abs(moved[i, 0] - x) < 1.0e-6]
            ys = sorted(float(moved[i, 1]) for i in band)
            # Consecutive cylinders are one period apart by construction; what
            # has to hold is that the comb still REACHES past the wall on both
            # sides at this phase.
            assert ys[0] <= -MOVING_HALF_WIDTH_RANGE_M[1], (
                f"row at x={x} stops at y={ys[0]:.2f} on the low side at "
                f"t={t:.2f} s, which may leave a gap against the wall"
            )
            assert ys[-1] >= MOVING_HALF_WIDTH_RANGE_M[1], (
                f"row at x={x} stops at y={ys[-1]:.2f} on the high side at "
                f"t={t:.2f} s"
            )


def _row_offset_at(layout, spec, t):
    """Circular distance between the two combs at time t, in metres."""
    rows = np.nonzero(spec.kind == KIND_TRANSLATE)[0]
    pos = moving_basin_positions(layout.centers, spec.kind, spec.params, float(t))
    xs = sorted({round(float(pos[i, 0]), 6) for i in rows})
    assert len(xs) == 2
    first = {}
    for x in xs:
        first[x] = min(float(pos[i, 1]) for i in rows if abs(pos[i, 0] - x) < 1.0e-6)
    d = (first[xs[1]] - first[xs[0]]) % PERIOD_M
    return min(d, PERIOD_M - d)


def test_the_two_rows_stay_staggered_at_every_instant():
    """The stagger is only a stagger if it does not wander.

    Implementing the design note literally, a half period phase difference on
    top of a half period spatial stagger, does NOT hold the offset: with a
    triangle wave tri(theta + P) = P - tri(theta), so the relative offset
    becomes 3P/2 - 2 tri(theta) and sweeps the whole period.  This test pins
    the property the task actually needs.
    """
    layout, spec = _sample(0.55, 0.0)
    period = 2.0 * PERIOD_M / spec.translate_speed_ms
    for t in np.linspace(0.0, 2.0 * period, 121):
        offset = _row_offset_at(layout, spec, t)
        assert abs(offset - 0.5 * PERIOD_M) < 1.0e-6, (
            f"at t={t:.2f} s the rows are offset by {offset:.4f} m rather than "
            f"the half period {0.5 * PERIOD_M:.4f} m"
        )


def test_a_straight_shot_through_both_rows_never_clears():
    """This is what the stagger is FOR, so it is asserted numerically.

    A single bearing clears both rows only if their gaps overlap by at least a
    hull beam, i.e. only if the offset is within GAP_M - HULL_BEAM_M.
    """
    slack = GAP_M - HULL_BEAM_M
    layout, spec = _sample(0.55, 0.0)
    period = 2.0 * PERIOD_M / spec.translate_speed_ms
    worst = -math.inf   # NOT 0.0: seeding at zero makes max() swallow every
                        # negative margin and the assertion below unfalsifiable
    for t in np.linspace(0.0, 2.0 * period, 241):
        worst = max(worst, slack - _row_offset_at(layout, spec, t))
    assert worst < 0.0, (
        f"a straight shot clears both rows somewhere in the cycle: the offset "
        f"falls to within {slack:.3f} m of alignment, worst margin {worst:+.4f} m"
    )
    assert 0.5 * PERIOD_M > slack, (
        f"half a period is {0.5 * PERIOD_M:.4f} m and the straight shot slack "
        f"is {slack:.4f} m; the stagger only bites while the first exceeds the second"
    )


def test_the_layout_fits_the_cylinder_cap():
    for mu in MOVING_DOSE_LADDER:
        layout, _ = _sample(mu, mu)
        assert layout.obstacle_count <= MOVING_MAX_CYLINDERS
        assert layout.obstacle_count > 50, (
            "far too few cylinders: the basin walls are not being built"
        )


def test_two_seeds_are_two_exam_papers_and_one_seed_reproduces():
    a1, s1 = _sample(0.55, 0.55, seed=11)
    a2, s2 = _sample(0.55, 0.55, seed=11)
    assert np.array_equal(a1.centers, a2.centers)
    assert np.array_equal(s1.params, s2.params)
    b1, _ = _sample(0.55, 0.55, seed=12)
    assert not np.array_equal(a1.centers, b1.centers), (
        "two eval seeds produced the same layout, so they are one exam paper"
    )


def test_segment_geometry_matches_the_design():
    layout, spec = _sample(0.36, 0.36)
    assert abs(spec.row_x[0] - MOVING_APPROACH_M) < 1.0e-9
    assert abs(spec.row_x[1] - spec.row_x[0] - MOVING_ROW_SEPARATION_M) < 1.0e-9
    assert abs(spec.orbit_center[0] - (spec.row_x[1] + MOVING_BUFFER_M + MOVING_ORBIT_RADIUS_M)) < 1.0e-9
    assert abs(spec.orbit_center[1]) < 1.0e-12, "the orbit must sit on the basin axis"
    gap_between = spec.orbit_center[0] - MOVING_ORBIT_RADIUS_M - spec.row_x[1]
    assert abs(gap_between - MOVING_BUFFER_M) < 1.0e-9, (
        f"the buffer between the segments is {gap_between:.3f} m, the design "
        f"requires {MOVING_BUFFER_M} m so the two can be attributed apart"
    )


def test_rotating_the_params_equals_rotating_the_points():
    """The refactor exists for this identity; if it fails the scene turns but
    the motion does not."""
    layout, spec = _sample(0.55, 0.55)
    for angle in (0.0, 0.3, 1.7, math.pi, 5.9):
        c, s_ = math.cos(angle), math.sin(angle)
        rot = np.array(((c, -s_), (s_, c)))
        turned_params = rotate_moving_params(spec.params, spec.kind, angle)
        turned_centers = layout.centers @ rot.T
        for t in (0.0, 2.5, 11.0, 47.3):
            direct = moving_basin_positions(
                layout.centers, spec.kind, spec.params, t
            ) @ rot.T
            viaparams = moving_basin_positions(
                turned_centers, spec.kind, turned_params, t
            )
            assert np.allclose(direct, viaparams, atol=1.0e-9), (
                f"angle {angle:.2f} rad, t={t}: rotating then moving differs "
                "from moving then rotating"
            )


def test_rotation_preserves_the_orbit_radius_and_the_row_spacing():
    layout, spec = _sample(0.55, 0.55)
    angle = 2.1
    c, s_ = math.cos(angle), math.sin(angle)
    rot = np.array(((c, -s_), (s_, c)))
    turned_params = rotate_moving_params(spec.params, spec.kind, angle)
    turned_centers = layout.centers @ rot.T
    orbit = np.nonzero(spec.kind == KIND_ORBIT)[0]
    assert orbit.size == 1
    turned_orbit_centre = spec.orbit_center @ rot.T
    for t in np.linspace(0.0, 40.0, 41):
        pos = moving_basin_positions(
            turned_centers, spec.kind, turned_params, float(t)
        )
        r = float(np.linalg.norm(pos[orbit[0]] - turned_orbit_centre))
        assert abs(r - MOVING_ORBIT_RADIUS_M) < 1.0e-9, (
            f"the orbit radius became {r:.6f} m after rotation"
        )


# ------------------------------------------------------------- env wiring --
def test_the_step_hook_reproduces_the_geometry_function_exactly():
    """The env must not re-derive the motion; it must evaluate the same law."""
    update = _lift_env_method("_update_moving_obstacles")
    for angle in (0.0, 1.3):
        steps = (0, 137, 4021)
        fake, layout, spec, count, base_local, params, origins = _moving_env(
            0.55, 0.55, angle=angle, steps=steps
        )
        update(fake)
        for env, step in enumerate(steps):
            t = step / 60.0
            expected = moving_basin_positions(base_local, spec.kind, params, t)
            got = (fake.obstacle_centers[env, :count] - origins[env]).numpy()
            assert np.allclose(got, expected, atol=2.0e-4), (
                f"angle {angle}, env {env}, step {step}: the env and the "
                "geometry function disagree"
            )


def test_the_step_hook_at_time_zero_reproduces_the_sampled_layout():
    update = _lift_env_method("_update_moving_obstacles")
    fake, layout, spec, count, base_local, _params, origins = _moving_env(
        0.80, 0.80, steps=(0, 0, 0)
    )
    update(fake)
    for env in range(fake.num_envs):
        got = (fake.obstacle_centers[env, :count] - origins[env]).numpy()
        assert np.allclose(got, base_local, atol=2.0e-4), (
            f"env {env}: the first step already differs from the sampled layout"
        )


def test_the_step_hook_leaves_rung_zero_completely_still():
    update = _lift_env_method("_update_moving_obstacles")
    fake, layout, spec, count, base_local, _p, origins = _moving_env(
        0.0, 0.0, steps=(0, 900, 7200)
    )
    assert np.all(spec.kind == KIND_STATIC)
    update(fake)
    for env in range(fake.num_envs):
        got = (fake.obstacle_centers[env, :count] - origins[env]).numpy()
        assert np.allclose(got, base_local, atol=1.0e-5), (
            f"env {env} moved at rung zero, so it is not a valid parent baseline"
        )


def test_each_env_advances_on_its_own_episode_clock():
    """Envs reset independently, so a shared clock would desynchronise them."""
    update = _lift_env_method("_update_moving_obstacles")
    fake, _l, spec, count, base_local, params, origins = _moving_env(
        0.55, 0.0, steps=(0, 600, 1200)
    )
    update(fake)
    rows = np.nonzero(spec.kind == KIND_TRANSLATE)[0]
    ys = [
        float((fake.obstacle_centers[env, :count] - origins[env])[rows[0], 1])
        for env in range(3)
    ]
    assert len({round(y, 6) for y in ys}) == 3, (
        f"three different episode times produced positions {ys}; the hook is "
        "reading a shared clock rather than episode_length_buf"
    )


def test_the_display_refresh_is_render_only_and_costs_headless_nothing():
    """Without it the recorded demo shows frozen cylinders and a dodging boat.

    _update_render_only_hazard_markers runs once per RESET, so the visual
    prims of a moving task would sit at their t = 0 poses for the whole
    episode.  The refresh must exist, must be called from the step hook, and
    must return before doing any work when nothing is being drawn.
    """
    method = _env_method_ast("_update_moving_obstacles")
    calls = [
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "_refresh_moving_markers" in calls, (
        "the step hook does not refresh the display prims, so a recording of "
        "this task would show the cylinders frozen where they started"
    )
    refresh = _env_method_ast("_refresh_moving_markers")
    first = refresh.body[1] if len(refresh.body) > 1 else refresh.body[0]
    guard = ast.unparse(first)
    assert "self._obstacle_translate_ops" in guard and "return" in ast.unparse(first), (
        f"the refresh does not bail out first when nothing is drawn; it starts "
        f"with {guard!r}"
    )
    # pxr is only importable after Isaac boots, so the import must be inside
    # the method and after the guard, never at module scope.
    imports = [n for n in ast.walk(refresh) if isinstance(n, ast.ImportFrom)]
    assert any(n.module == "pxr" for n in imports), (
        "Gf is used without a local pxr import; on a rendering run this raises "
        "NameError, and a headless test cannot see it"
    )
    source = ENV_PATH.read_text(encoding="utf-8")
    module_level = [
        line for line in source.splitlines()
        if line.startswith("from pxr import") or line.startswith("import pxr")
    ]
    assert not module_level, "pxr must not be imported at module scope"


def test_the_hook_is_guarded_and_runs_before_the_contact_predicate():
    method = _env_method_ast("_get_dones")
    hook = None
    for node in method.body:
        if not isinstance(node, ast.If):
            break
        calls = [
            inner.func.attr
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
        ]
        if "_update_moving_obstacles" in calls:
            hook = node
            break
    assert hook is not None, (
        "_get_dones does not call _update_moving_obstacles from a leading "
        "guarded statement"
    )
    guard = ast.unparse(hook.test).replace(" ", "")
    assert "getattr(self,'_moving_enabled',False)" == guard, (
        f"the moving hook guard is {guard!r}; it must be an inert-by-default "
        "getattr so every certified cfg keeps its exact step path"
    )
    clearance = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_clearance"
    ]
    assert clearance and hook.lineno < min(clearance), (
        "the cylinders must be moved before the contact predicate reads them"
    )


def test_the_reset_path_samples_and_rotates():
    """The rotation is the bug this parameterisation exists to prevent."""
    source = ENV_PATH.read_text(encoding="utf-8")
    assert 'layout_mode == "moving_basin"' in source, (
        "no moving_basin branch in the layout dispatch"
    )
    assert "sample_moving_basin_layout(" in source
    assert "rotate_moving_params(" in source, (
        "the reset path does not rotate the motion, so the cylinders will "
        "keep sliding along the old world axis after the scene is turned"
    )
    method = _env_method_ast("_reset_idx")
    body = ast.unparse(method)
    assert "rotate_moving_params" in body and "goal_angle" in body


def test_buffers_are_allocated_only_for_the_moving_id():
    source = ENV_PATH.read_text(encoding="utf-8")
    assert 'self._moving_enabled = (' in source
    assert 'getattr(self.cfg, "layout_mode", "") == "moving_basin"' in source, (
        "the allocation guard must read the cfg through getattr so cfgs that "
        "predate this task are untouched"
    )


# ------------------------------------------------------------- cfg and id --
def test_the_cfg_class_exists_with_the_right_parent_and_fields():
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HazardMovingBasinEnvCfg":
            found = node
    assert found is not None, "HazardMovingBasinEnvCfg is not defined"
    bases = [b.id for b in found.bases if isinstance(b, ast.Name)]
    assert bases == ["HazardNavV3EnvCfg"], (
        f"parent is {bases}; it must be HazardNavV3EnvCfg so observation_space "
        "stays 42 and certified archives load zero-shot"
    )
    fields = {
        item.target.id: ast.unparse(item.value)
        for item in found.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }
    assert fields.get("layout_mode") == "'moving_basin'"
    assert fields.get("moving_mu") == "0.0", "the default dose must be undosed"
    assert fields.get("moving_nu") == "0.0"
    cap = int(fields["max_obstacles"])
    assert cap > MOVING_MAX_CYLINDERS, (
        f"max_obstacles is {cap} against a sampler cap of {MOVING_MAX_CYLINDERS}; "
        "the cfg must leave headroom above whatever the sampler can build"
    )


def test_every_declared_layout_mode_is_in_the_validator_allow_list():
    """The bug this test exists for, stated plainly.

    __post_init__ validates layout_mode against a literal tuple.  A new task
    that declares a mode without being added to that tuple constructs fine on
    CPU, passes every test here, deploys cleanly, and then dies on the GPU the
    instant the cfg is built, because the cfg cannot be imported off Isaac and
    so nothing on this side ever runs __post_init__.  That is exactly what
    happened to the first P5 queue: three screening cells died six seconds
    apart with "layout_mode must be one of ...".

    The check is stated generally rather than as a literal for moving_basin,
    so the next task to add a mode is covered without anyone remembering to.
    """
    source = CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    declared = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            target = None
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                target = item.target.id
            elif (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
            ):
                target = item.targets[0].id
            if target != "layout_mode" or item.value is None:
                continue
            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                declared.add(item.value.value)
    assert declared, "no cfg class declares layout_mode; the parser is wrong"

    allowed = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not (
            isinstance(node.left, ast.Attribute)
            and node.left.attr == "layout_mode"
            and any(isinstance(op, ast.NotIn) for op in node.ops)
        ):
            continue
        candidate = node.comparators[0]
        if not isinstance(candidate, ast.Tuple):
            continue
        names = {
            elt.value for elt in candidate.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        # The reward_progress_geodesic guard is a different, deliberately
        # narrower tuple; take the widest one, which is the mode validator.
        if allowed is None or len(names) > len(allowed):
            allowed = names
    assert allowed, "no layout_mode allow-list found in the cfg"

    missing = sorted(declared - allowed)
    assert not missing, (
        f"these layout modes are declared by a cfg class but are not in the "
        f"__post_init__ allow-list, so every run of them dies at cfg "
        f"construction on the GPU: {missing}"
    )


def test_the_gym_id_is_registered_against_the_right_cfg():
    source = INIT_PATH.read_text(encoding="utf-8")
    assert MOVING_ID in source, f"{MOVING_ID} is not registered"
    where = source.index(MOVING_ID)
    window = source[where : where + 600]
    assert "HazardMovingBasinEnvCfg" in window
    assert "hazard_nav_env:HazardNavEnv" in window


def test_no_certified_id_was_touched():
    """append-only: the ids that already have certificates must be untouched."""
    source = INIT_PATH.read_text(encoding="utf-8")
    for certified in (
        "Isaac-USV-HazardNav-Direct-v3",
        "Isaac-USV-HazardCross-Direct-v1",
        "Isaac-USV-HazardCrossAppear-Direct-v1",
        "Isaac-USV-Iceberg-Direct-v1",
    ):
        assert certified in source, f"{certified} vanished from the registry"


def test_sources_are_pure_ascii():
    for path in (
        _HERE / "hazard_geometry.py",
        ENV_PATH,
        CFG_PATH,
        INIT_PATH,
        Path(__file__).resolve(),
    ):
        payload = path.read_bytes()
        assert all(byte < 128 for byte in payload), f"{path.name}: non-ASCII byte"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
