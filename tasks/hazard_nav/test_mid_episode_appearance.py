# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the mid-episode obstacle appearance axis (sudden terrain change).

WHAT THE AXIS IS
----------------
At a per-episode time t0, one or more EXTRA cylinders flip from inactive to
active in ``obstacle_active``.  The buffers are read every step already, so
the flip needs no data-structure change; what needs proof is FAIRNESS and
INERTNESS:

* the appearance azimuth is aligned to a ray of the 36-ray sensor at the
  moment of appearance, so the obstacle is guaranteed visible the instant it
  exists;
* the centre distance from the CURRENT boat position is clamped to
  [9.2, 25] m (9.2 m = r/sin(5 deg) guaranteed-visibility bound at the 0.8 m
  minimum radius -- braking needs only 0.891 m, so visibility binds);
* admission reruns the layout sampler's own check -- BFS feasibility at
  planning inflation -- on the POST-appearance field, resamples the azimuth
  on rejection (bounded), and SKIPS the appearance, recording the skip, when
  every azimuth is inadmissible;
* the appearance time rides the scenario protocol under its own primitive
  group, so eval seeds control it and two controllers get the same t0 for the
  same (env, episode);
* ``appear_count == 0`` (every certified id) is bit-identical: no buffer
  written, no stream consumed, no step-path branch taken.

HOW IT IS TESTED WITHOUT ISAAC
------------------------------
The decision logic lives in ``hazard_geometry.plan_obstacle_appearance``
(Isaac-free by design, like every layout sampler) and is exercised directly.
The env wiring -- ``_reset_appearance`` and ``_maybe_appear_obstacles`` on
``HazardNavEnv`` -- cannot be imported here (the env module imports isaaclab
at module scope), so both methods are AST-LIFTED out of the real source file
and executed against fabricated ``self`` objects, the technique
``tasks/_shared/test_hazard_env_wiring.py`` established.  Editing the env
source changes what these tests run; a stub could not.

Run: python tasks/hazard_nav/test_mid_episode_appearance.py
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SHARED = _HERE.parent / "_shared"
for _directory in (_HERE, _SHARED):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

try:
    from .hazard_geometry import (
        APPEAR_MAX_DISTANCE_M,
        APPEAR_MIN_DISTANCE_M,
        HALF_BEAM_M,
        MIN_OBSTACLE_RADIUS_M,
        _basin_cylinders,
        bfs_geodesic_length,
        plan_obstacle_appearance,
        ray_circle_ranges,
    )
except ImportError:  # direct execution
    from hazard_geometry import (  # type: ignore
        APPEAR_MAX_DISTANCE_M,
        APPEAR_MIN_DISTANCE_M,
        HALF_BEAM_M,
        MIN_OBSTACLE_RADIUS_M,
        _basin_cylinders,
        bfs_geodesic_length,
        plan_obstacle_appearance,
        ray_circle_ranges,
    )

from obs_superset import NATIVE_LAYOUTS  # noqa: E402
from scenario_draws import make_scenario_rng  # noqa: E402
from scenario_draws_hazard import (  # noqa: E402
    GROUP_APPEARANCE,
    GROUP_LAYOUT,
    appearance_stream,
)
from scenario_rng import ScenarioRNG  # noqa: E402

DEVICE = torch.device("cpu")
ENV_PATH = _HERE / "hazard_nav_env.py"
CFG_PATH = _HERE / "hazard_nav_env_cfg.py"
INIT_PATH = _HERE / "__init__.py"
RAY_COUNT = 36
APPEAR_ID = "Isaac-USV-HazardCrossAppear-Direct-v1"
PARENT_ID = "Isaac-USV-HazardCross-Direct-v1"

# Two reset schedules = two termination patterns, the pair every scenario
# wiring test in this tree uses: by the time each reaches env i's k-th
# episode they have consumed different draw counts from any SHARED stream.
SCHEDULE_A = [[0, 1, 2, 3], [1, 3], [0, 2], [1, 3], [0, 2]]
SCHEDULE_B = [[0, 1, 2, 3], [0], [2], [1], [3], [0, 1, 2, 3]]


# --------------------------------------------------------------- AST lifting --
def _lift_env_method(name: str):
    """Compile one SHIPPED HazardNavEnv method into a callable."""
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HazardNavEnv":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    module = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(module)
                    namespace = {
                        "np": np,
                        "torch": torch,
                        "math": math,
                        "plan_obstacle_appearance": plan_obstacle_appearance,
                        "appearance_stream": appearance_stream,
                        "GROUP_APPEARANCE": GROUP_APPEARANCE,
                    }
                    exec(  # noqa: S102 - executing the shipped source is the point
                        compile(module, str(ENV_PATH), "exec"), namespace
                    )
                    return namespace[name]
    raise AssertionError(f"hazard_nav_env.py: no HazardNavEnv.{name}")


def _env_method_ast(name: str) -> ast.FunctionDef:
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HazardNavEnv":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"hazard_nav_env.py: no HazardNavEnv.{name}")


class _Fake:
    """A stand-in env/cfg: only the attributes the lifted code reads."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _Tripwire:
    """A stand-in whose UNLISTED attributes may not be read at all."""

    def __init__(self, **allowed):
        self.__dict__.update(allowed)

    def __getattr__(self, name):
        raise AssertionError(
            f"the inert (appear_count == 0) path read {name!r}; certified ids "
            "must not have their buffers or streams touched by this axis"
        )


def _ray_directions(forward_xy: np.ndarray) -> np.ndarray:
    """The env's ray direction formula: cos(a)*forward + sin(a)*left."""
    forward = np.asarray(forward_xy, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    left = np.array((-forward[1], forward[0]))
    angles = 2.0 * np.pi * np.arange(RAY_COUNT) / RAY_COUNT
    return (
        np.cos(angles)[:, None] * forward[None, :]
        + np.sin(angles)[:, None] * left[None, :]
    )


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ============================================================================
# 1. Fairness floor: ray alignment, to float precision
# ============================================================================
def test_appearance_azimuth_equals_a_ray_azimuth_to_float_precision():
    rng = np.random.default_rng(20260826)
    goal = np.array((60.0, 60.0))
    for _ in range(200):
        boat = rng.uniform(-20.0, 20.0, size=2)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        forward = np.array((math.cos(heading), math.sin(heading)))
        order = rng.permutation(RAY_COUNT)
        plan = plan_obstacle_appearance(
            boat,
            forward,
            goal,
            np.zeros((0, 2)),
            np.zeros(0),
            distance_m=float(rng.uniform(9.2, 25.0)),
            radius_m=0.8,
            candidate_ray_indices=order,
        )
        assert plan is not None, "open water must admit an appearance"
        directions = _ray_directions(forward)
        offset = plan.center - boat
        bearing = math.atan2(offset[1], offset[0])
        ray_azimuth = math.atan2(
            directions[plan.ray_index][1], directions[plan.ray_index][0]
        )
        assert abs(_wrap(bearing - ray_azimuth)) < 1.0e-9, (
            f"appearance bearing {bearing} is not ray {plan.ray_index}'s "
            f"azimuth {ray_azimuth}: the fairness floor requires exact "
            "alignment at the moment of appearance"
        )
        assert abs(_wrap(plan.azimuth_rad - ray_azimuth)) < 1.0e-9
        # And the aligned ray SEES it: unit dot between the offset and the
        # ray direction is 1 to float precision.
        cosine = float(
            np.dot(offset / np.linalg.norm(offset), directions[plan.ray_index])
        )
        assert cosine > 1.0 - 1.0e-12


def test_aligned_appearance_is_visible_to_the_env_ray_sensor():
    """ray_circle_ranges (the shipped sensor) returns a hit on the aligned ray."""
    rng = np.random.default_rng(7)
    for distance in (9.2, 12.0, 20.0, 25.0):
        boat = rng.uniform(-5.0, 5.0, size=2)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        forward = np.array((math.cos(heading), math.sin(heading)))
        plan = plan_obstacle_appearance(
            boat,
            forward,
            np.array((80.0, 0.0)),
            np.zeros((0, 2)),
            np.zeros(0),
            distance_m=distance,
            radius_m=0.8,
            candidate_ray_indices=rng.permutation(RAY_COUNT),
        )
        assert plan is not None
        directions = torch.as_tensor(
            _ray_directions(forward)[None, :, :], dtype=torch.float64
        )
        ranges = ray_circle_ranges(
            torch.as_tensor(boat[None, :], dtype=torch.float64),
            directions,
            torch.as_tensor(plan.center[None, None, :], dtype=torch.float64),
            torch.full((1, 1), plan.radius_m, dtype=torch.float64),
            max_range_m=30.0,
            active_mask=torch.ones((1, 1), dtype=torch.bool),
        )[0]
        expected = plan.distance_m - plan.radius_m
        assert abs(float(ranges[plan.ray_index]) - expected) < 1.0e-6, (
            "the sensor does not see the appeared cylinder at surface range "
            f"{expected} on the aligned ray -- visibility is not guaranteed"
        )
        assert float(ranges[plan.ray_index]) < 30.0 - 1.0e-6


# ============================================================================
# 2. Fairness floor: the [9.2, 25] m distance clamp
# ============================================================================
def test_appearance_distance_clamps_to_the_fairness_interval():
    boat = np.array((3.0, -2.0))
    forward = np.array((0.6, 0.8))
    cases = (
        (5.0, APPEAR_MIN_DISTANCE_M),
        (9.2, 9.2),
        (9.5, 9.5),
        (12.0, 12.0),
        (15.0, 15.0),
        (20.0, 20.0),
        (25.0, 25.0),
        (40.0, APPEAR_MAX_DISTANCE_M),
    )
    for requested, expected in cases:
        plan = plan_obstacle_appearance(
            boat,
            forward,
            np.array((90.0, 0.0)),
            np.zeros((0, 2)),
            np.zeros(0),
            distance_m=requested,
            radius_m=0.8,
            candidate_ray_indices=list(range(RAY_COUNT)),
        )
        assert plan is not None
        assert plan.distance_m == expected, (
            f"requested {requested} m must clamp to {expected} m, "
            f"got {plan.distance_m}"
        )
        realised = float(np.linalg.norm(plan.center - boat))
        assert abs(realised - expected) < 1.0e-9, (
            f"centre landed {realised} m from the boat, not the clamped "
            f"{expected} m"
        )
    assert APPEAR_MIN_DISTANCE_M == 9.2 and APPEAR_MAX_DISTANCE_M == 25.0, (
        "the frozen fairness interval is [9.2, 25] m"
    )


# ============================================================================
# 3. Admission: goal disc, post-appearance BFS, exhaustion -> None
# ============================================================================
def test_goal_disc_overlap_is_rejected_and_resampled():
    boat = np.zeros(2)
    forward = np.array((1.0, 0.0))
    goal = np.array((12.0, 0.0))  # dead ahead, inside the appearance band
    plan = plan_obstacle_appearance(
        boat,
        forward,
        goal,
        np.zeros((0, 2)),
        np.zeros(0),
        distance_m=12.0,
        radius_m=0.8,
        candidate_ray_indices=[0, 18],
        goal_radius_m=2.0,
    )
    assert plan is not None
    assert plan.ray_index == 18, (
        "ray 0 would drop the cylinder ON the goal disc; admission must "
        "resample to the next candidate azimuth"
    )
    assert plan.attempts == 2
    assert float(np.linalg.norm(plan.center - goal)) > plan.radius_m + 2.0


def test_admission_runs_on_the_post_appearance_field():
    """The azimuth that would seal the corridor is rejected; another admits."""
    centers, radii = _basin_cylinders(
        -20.0, 30.0, 3.0, 0.0, 0.0, 0.0, include_bulkhead=False
    )
    boat = np.zeros(2)
    forward = np.array((1.0, 0.0))
    goal = np.array((25.0, 0.0))
    # Sanity: the corridor is feasible before any appearance.
    assert bfs_geodesic_length(boat, goal, centers, radii, cell_m=0.25) is not None
    plan = plan_obstacle_appearance(
        boat,
        forward,
        goal,
        centers,
        radii,
        distance_m=9.5,
        radius_m=2.0,
        candidate_ray_indices=[0, 18],
    )
    assert plan is not None
    assert plan.ray_index == 18, (
        "a 2.0 m cylinder 9.5 m dead ahead seals the corridor to the goal; "
        "BFS admission on the POST-appearance field must reject ray 0 and "
        "admit the astern azimuth instead"
    )
    # The accepted field really is feasible, the rejected one really is not.
    ahead = boat + 9.5 * forward
    sealed = bfs_geodesic_length(
        boat,
        goal,
        np.vstack((centers, ahead[None, :])),
        np.concatenate((radii, (2.0,))),
        cell_m=0.25,
    )
    assert sealed is None
    admitted = bfs_geodesic_length(
        boat,
        goal,
        np.vstack((centers, plan.center[None, :])),
        np.concatenate((radii, (plan.radius_m,))),
        cell_m=0.25,
    )
    assert admitted is not None


def _sealed_box_field():
    """A basin that seals the boat in while the goal sits outside it."""
    centers, radii = _basin_cylinders(
        -5.0, 5.0, 5.0, 0.0, 0.0, 0.0, include_bulkhead=False
    )
    return centers, radii, np.array((30.0, 0.0))


def test_every_azimuth_infeasible_returns_none():
    centers, radii, goal = _sealed_box_field()
    boat = np.zeros(2)
    plan = plan_obstacle_appearance(
        boat,
        np.array((1.0, 0.0)),
        goal,
        centers,
        radii,
        distance_m=12.0,
        radius_m=0.8,
        candidate_ray_indices=list(range(RAY_COUNT)),
    )
    assert plan is None, (
        "the boat is sealed away from the goal, so EVERY candidate azimuth "
        "fails BFS admission; the planner must return None (skip), never "
        "force a spawn"
    )


# ============================================================================
# 4. Env wiring: the shipped _maybe_appear_obstacles, executed
# ============================================================================
def _appearance_env(
    *,
    num_envs: int = 2,
    appear_count: int = 1,
    appear_distance_m: float = 12.0,
    appear_radius_m: float = 0.8,
    max_obstacles: int = 8,
    steps: tuple = (601, 10),
    time_s: tuple = (10.0, 10.0),
    boat_world=None,
    heading_rad: tuple = (0.5, 0.0),
    goal_local=(25.0, 0.0),
    rng_seed: int = 7,
):
    """A fabricated env carrying exactly what the lifted methods read."""
    origins = torch.tensor([[100.0, -50.0], [-30.0, 40.0]])[:num_envs]
    if boat_world is None:
        boat_world = origins + torch.tensor([[3.0, 2.0], [0.0, 0.0]])[:num_envs]
    forward = torch.stack(
        [
            torch.tensor([math.cos(a), math.sin(a)])
            for a in heading_rad[:num_envs]
        ]
    )
    fake = _Fake(
        _appear_count=appear_count,
        num_envs=num_envs,
        device=DEVICE,
        control_step_s=1.0 / 60.0,
        goal_radius=2.0,
        episode_length_buf=torch.tensor(steps[:num_envs], dtype=torch.long),
        _appear_pending=torch.ones(num_envs, dtype=torch.bool),
        _appear_time_s=torch.tensor(time_s[:num_envs]),
        _appear_happened=torch.zeros(num_envs, dtype=torch.bool),
        _appear_skipped=torch.zeros(num_envs, dtype=torch.bool),
        _appear_spawned=torch.zeros(num_envs, dtype=torch.long),
        _appear_azimuth_rad=torch.full((num_envs,), torch.nan),
        _appear_distance_m=torch.full((num_envs,), torch.nan),
        _appear_rngs=[np.random.default_rng(rng_seed + i) for i in range(num_envs)],
        obstacle_centers=torch.zeros((num_envs, max_obstacles, 2)),
        obstacle_radii=torch.zeros((num_envs, max_obstacles)),
        obstacle_active=torch.zeros((num_envs, max_obstacles), dtype=torch.bool),
        obstacle_count=torch.zeros(num_envs, dtype=torch.long),
        target_pos=origins + torch.tensor(goal_local),
        scene=_Fake(env_origins=origins),
        cfg=_Fake(
            max_obstacles=max_obstacles,
            ray_count=RAY_COUNT,
            appear_max_attempts=RAY_COUNT,
            appear_distance_m=appear_distance_m,
            appear_radius_m=appear_radius_m,
            appear_bfs_cell_m=0.25,
            half_beam_m=HALF_BEAM_M,
        ),
    )
    fake._com_xy = lambda: boat_world
    fake._forward_2d = lambda: forward
    return fake, origins, boat_world, forward


def test_shipped_step_hook_flips_a_ray_aligned_clamped_cylinder():
    appear = _lift_env_method("_maybe_appear_obstacles")
    fake, origins, boat_world, forward = _appearance_env()
    appear(fake)

    # Env 1 was not due (10 steps = 0.17 s < t0): untouched.
    assert not bool(fake.obstacle_active[1].any())
    assert bool(fake._appear_pending[1])

    # Env 0 was due: exactly one extra cylinder, in the first reserved slot.
    assert int(fake.obstacle_active[0].sum()) == 1
    assert bool(fake.obstacle_active[0, 0])
    assert not bool(fake._appear_pending[0])
    assert bool(fake._appear_happened[0])
    assert not bool(fake._appear_skipped[0])
    assert int(fake._appear_spawned[0]) == 1

    # The committed centre replays the planner exactly: same stream state,
    # same inputs, same plan.
    replay_rng = np.random.default_rng(7)
    boat_local = (boat_world[0] - origins[0]).numpy().astype(np.float64)
    plan = plan_obstacle_appearance(
        boat_local,
        forward[0].numpy().astype(np.float64),
        (fake.target_pos[0] - origins[0]).numpy().astype(np.float64),
        np.zeros((0, 2)),
        np.zeros(0),
        distance_m=12.0,
        radius_m=0.8,
        candidate_ray_indices=replay_rng.permutation(RAY_COUNT),
    )
    assert plan is not None
    committed_local = (fake.obstacle_centers[0, 0] - origins[0]).numpy()
    assert np.allclose(committed_local, plan.center, atol=1.0e-5), (
        "the committed centre does not replay plan_obstacle_appearance on "
        "the same stream -- the env is not executing the planner's decision"
    )
    assert abs(float(fake.obstacle_radii[0, 0]) - 0.8) < 1.0e-6
    assert abs(float(fake._appear_azimuth_rad[0]) - plan.azimuth_rad) < 1.0e-6
    assert abs(float(fake._appear_distance_m[0]) - 12.0) < 1.0e-6

    # Fairness, measured on the committed tensors: clamped distance and a
    # bearing on one of the 36 ray azimuths of the CURRENT heading.
    offset = (fake.obstacle_centers[0, 0] - boat_world[0]).numpy().astype(np.float64)
    assert abs(float(np.linalg.norm(offset)) - 12.0) < 1.0e-3
    directions = _ray_directions(forward[0].numpy())
    bearing = math.atan2(offset[1], offset[0])
    best = min(
        abs(_wrap(bearing - math.atan2(d[1], d[0]))) for d in directions
    )
    assert best < 1.0e-3, (
        f"committed bearing {bearing} is {best} rad from the nearest ray "
        "azimuth; the flip is not ray-aligned at the moment of appearance"
    )

    # The flip is once per episode: a second call must change nothing.
    before = fake.obstacle_active.clone()
    appear(fake)
    assert torch.equal(before, fake.obstacle_active)
    assert int(fake._appear_spawned[0]) == 1


def test_shipped_step_hook_skips_and_records_when_no_azimuth_is_admissible():
    appear = _lift_env_method("_maybe_appear_obstacles")
    centers, radii, goal = _sealed_box_field()
    count = len(radii)
    max_obstacles = count + 2
    fake, origins, _, _ = _appearance_env(
        num_envs=1,
        max_obstacles=max_obstacles,
        steps=(601,),
        time_s=(10.0,),
        heading_rad=(0.0,),
        goal_local=tuple(goal.tolist()),
    )
    fake.obstacle_centers[0, :count] = origins[0] + torch.as_tensor(
        centers, dtype=torch.float32
    )
    fake.obstacle_radii[0, :count] = torch.as_tensor(radii, dtype=torch.float32)
    fake.obstacle_active[0, :count] = True
    fake.obstacle_count[0] = count
    appear(fake)
    assert bool(fake._appear_skipped[0]), (
        "every azimuth is infeasible in the sealed box, so the env must "
        "record the SKIP for this episode"
    )
    assert not bool(fake._appear_happened[0])
    assert int(fake._appear_spawned[0]) == 0
    assert int(fake.obstacle_active[0].sum()) == count, (
        "a skipped appearance may not flip anything active"
    )
    assert not bool(fake._appear_pending[0])


def test_multiple_appearances_admit_against_the_grown_field():
    """appear_count = 2: both flip, both aligned, both jointly feasible."""
    appear = _lift_env_method("_maybe_appear_obstacles")
    fake, origins, boat_world, forward = _appearance_env(
        num_envs=1,
        appear_count=2,
        steps=(601,),
        time_s=(10.0,),
        heading_rad=(0.5,),
    )
    appear(fake)
    assert int(fake._appear_spawned[0]) == 2
    assert bool(fake._appear_happened[0])
    assert not bool(fake._appear_skipped[0])
    assert int(fake.obstacle_active[0].sum()) == 2
    assert bool(fake.obstacle_active[0, 0]) and bool(fake.obstacle_active[0, 1])
    directions = _ray_directions(forward[0].numpy())
    committed = []
    for slot in (0, 1):
        offset = (
            (fake.obstacle_centers[0, slot] - boat_world[0])
            .numpy().astype(np.float64)
        )
        assert abs(float(np.linalg.norm(offset)) - 12.0) < 1.0e-3
        bearing = math.atan2(offset[1], offset[0])
        best = min(
            abs(_wrap(bearing - math.atan2(d[1], d[0]))) for d in directions
        )
        assert best < 1.0e-3, f"slot {slot} is not ray-aligned"
        committed.append(
            (fake.obstacle_centers[0, slot] - origins[0]).numpy().astype(np.float64)
        )
    # The joint post-appearance field is feasible at planning inflation from
    # the boat to the goal -- the second admission ran against the field
    # INCLUDING the first cylinder.
    boat_local = (boat_world[0] - origins[0]).numpy().astype(np.float64)
    goal_local = (fake.target_pos[0] - origins[0]).numpy().astype(np.float64)
    assert (
        bfs_geodesic_length(
            boat_local,
            goal_local,
            np.vstack(committed),
            np.full(2, 0.8),
            cell_m=0.25,
        )
        is not None
    )


class _StubRng:
    """Hands the env a scripted candidate order per appearing cylinder."""

    def __init__(self, sequences):
        self._sequences = iter(sequences)

    def permutation(self, count):
        order = np.asarray(next(self._sequences), dtype=np.int64)
        assert np.all((0 <= order) & (order < count))
        return order


def test_second_admission_sees_the_first_appearance():
    """A pincer: ray 35 admits alone but jointly seals with cylinder one.

    Corridor free band is |y| < 2.15 m at planning inflation. Cylinder one
    (r = 2.0 m, 9.5 m out on ray 1) leaves a 1.15 m gap below itself, so it
    admits. Its mirror on ray 35 ALSO admits against the walls alone -- but
    together the two inflated discs cover the whole band for ~4 m of x, so
    on the grown field ray 35 must be rejected and the scripted fallback
    (ray 18, astern) taken instead. An env that reruns admission on the
    PRE-appearance field accepts ray 35 and this test fails.
    """
    appear = _lift_env_method("_maybe_appear_obstacles")
    centers, radii = _basin_cylinders(
        -15.0, 35.0, 4.0, 0.0, 0.0, 0.0, include_bulkhead=False
    )
    count = len(radii)
    fake, origins, boat_world, forward = _appearance_env(
        num_envs=1,
        appear_count=2,
        appear_distance_m=9.5,
        appear_radius_m=2.0,
        max_obstacles=count + 4,
        steps=(601,),
        time_s=(10.0,),
        heading_rad=(0.0,),
        goal_local=(25.0, 0.0),
    )
    # The construction assumes the boat on the corridor axis at local (0, 0).
    boat_world = origins.clone()
    fake._com_xy = lambda: boat_world
    fake.obstacle_centers[0, :count] = origins[0] + torch.as_tensor(
        centers, dtype=torch.float32
    )
    fake.obstacle_radii[0, :count] = torch.as_tensor(radii, dtype=torch.float32)
    fake.obstacle_active[0, :count] = True
    fake.obstacle_count[0] = count
    fake._appear_rngs = [_StubRng([[1], [35, 18]])]
    appear(fake)
    assert int(fake._appear_spawned[0]) == 2
    assert not bool(fake._appear_skipped[0])
    first_local = (
        (fake.obstacle_centers[0, count] - origins[0]).numpy().astype(np.float64)
    )
    second_local = (
        (fake.obstacle_centers[0, count + 1] - origins[0])
        .numpy()
        .astype(np.float64)
    )
    assert first_local[0] > 0.0 and abs(first_local[1] - 1.65) < 0.1
    assert second_local[0] < 0.0, (
        "cylinder two took ray 35 (the pincer seal): its admission BFS ran "
        "on the PRE-appearance field instead of the field including "
        "cylinder one"
    )
    # The scripted fallback really was the astern ray, and the joint field
    # really is feasible.
    assert abs(second_local[0] + 9.5) < 1.0e-3 and abs(second_local[1]) < 1.0e-3
    joint_centers = np.vstack((centers, first_local, second_local))
    joint_radii = np.concatenate((radii, (2.0, 2.0)))
    boat_local = (boat_world[0] - origins[0]).numpy().astype(np.float64)
    goal_local = (fake.target_pos[0] - origins[0]).numpy().astype(np.float64)
    assert (
        bfs_geodesic_length(
            boat_local, goal_local, joint_centers, joint_radii, cell_m=0.25
        )
        is not None
    )
    # And the pincer construction has teeth: ray 35's centre WOULD have been
    # feasible on the pre-appearance field.
    mirror = np.array((first_local[0], -first_local[1]))
    assert (
        bfs_geodesic_length(
            boat_local,
            goal_local,
            np.vstack((centers, mirror[None, :])),
            np.concatenate((radii, (2.0,))),
            cell_m=0.25,
        )
        is not None
    )


def test_shipped_step_hook_does_nothing_before_t0():
    appear = _lift_env_method("_maybe_appear_obstacles")
    fake, _, _, _ = _appearance_env(steps=(10, 10))
    appear(fake)
    assert not bool(fake.obstacle_active.any())
    assert bool(fake._appear_pending.all())


# ============================================================================
# 5. Inert-by-default bit-identity
# ============================================================================
def test_inert_step_path_reads_nothing_but_the_guard():
    appear = _lift_env_method("_maybe_appear_obstacles")
    appear(_Tripwire(_appear_count=0))  # any other read raises


def test_inert_reset_path_reads_nothing_draws_nothing_stamps_nothing():
    reset_appearance = _lift_env_method("_reset_appearance")
    stamp = reset_appearance(
        _Tripwire(_appear_count=0),
        torch.tensor([0, 1], dtype=torch.long),
        [0, 1],
    )
    assert stamp == {}, (
        "the inert path must contribute NO scenario group, or every "
        "certified id's scenario hash key set changes"
    )


def test_flip_runs_before_the_contact_predicate():
    method = _env_method_ast("_get_dones")
    first = method.body[0]
    assert isinstance(first, ast.If), (
        "the guarded appearance hook must be the FIRST statement of "
        "_get_dones so the appeared cylinder is part of this step's contact "
        "predicate, reward field and observation rays"
    )
    guard = ast.unparse(first.test).replace(" ", "")
    assert guard == "getattr(self.cfg,'appear_count',0)>0", (
        f"the _get_dones hook guard is {guard!r}, not the inert-by-default "
        "appear_count gate"
    )
    hook_calls = [
        node
        for node in ast.walk(first)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_maybe_appear_obstacles"
    ]
    assert len(hook_calls) == 1, "the guard must call _maybe_appear_obstacles"
    clearance_lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_clearance"
    ]
    assert clearance_lines and first.lineno < min(clearance_lines)


def test_reset_idx_calls_the_appearance_reset_and_stamps_it():
    method = _env_method_ast("_reset_idx")
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_reset_appearance"
    ]
    assert len(calls) == 1, "_reset_idx must draw the appearance time once"
    stamp_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "stamp_scenario"
    ]
    assert len(stamp_calls) == 1
    assert calls[0].lineno < stamp_calls[0].lineno, (
        "the appearance draw must precede the scenario stamp that records it"
    )
    # The ** spread of the appearance block into the stamp dict:
    spreads = [
        value.id
        for node in ast.walk(stamp_calls[0])
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if key is None and isinstance(value, ast.Name)
    ]
    assert "appear_scenario" in spreads, (
        "the appearance stamp block never reaches stamp_scenario, so the "
        "drawn t0 would not be recorded in the certificate"
    )


# ============================================================================
# 6. Scenario protocol: its own group, seed-controlled, reproducible
# ============================================================================
def test_appearance_stream_is_the_keyed_protocol_stream():
    scenario = ScenarioRNG(num_envs=4, device=DEVICE, eval_seed=515)
    scenario.reset_idx(torch.arange(4))
    fallback = np.random.default_rng(515)
    for env_index in range(4):
        stream = appearance_stream(scenario, fallback, env_index)
        assert stream is not fallback
        expected = scenario.numpy_rng(GROUP_APPEARANCE, env_index)
        assert stream.bit_generator.state == expected.bit_generator.state, (
            "appearance_stream is not the keyed 'appearance' stream of this "
            "(env, episode)"
        )
        layout = scenario.numpy_rng(GROUP_LAYOUT, env_index)
        assert stream.bit_generator.state != layout.bit_generator.state, (
            "the appearance primitive must own its OWN group, not ride the "
            "layout stream"
        )
    # Unseeded fallback: a PRIVATE child (mid-episode draws may not advance
    # the shared generator), costing exactly one draw off the fallback.
    before = fallback.bit_generator.state
    child = appearance_stream(None, fallback, 0)
    assert child is not fallback
    assert fallback.bit_generator.state != before


def _run_reset_schedule(seed: int, schedule) -> dict:
    """Drive the SHIPPED _reset_appearance through one termination pattern."""
    reset_appearance = _lift_env_method("_reset_appearance")
    scenario = make_scenario_rng(_Fake(seed=seed), 4, DEVICE)
    fake = _Fake(
        _appear_count=1,
        _scenario=scenario,
        _layout_rng=np.random.default_rng(seed),
        device=DEVICE,
        cfg=_Fake(appear_time_min_s=5.0, appear_time_max_s=30.0),
        _appear_rngs=[None] * 4,
        _appear_time_s=torch.full((4,), torch.inf),
        _appear_pending=torch.zeros(4, dtype=torch.bool),
        _appear_happened=torch.zeros(4, dtype=torch.bool),
        _appear_skipped=torch.zeros(4, dtype=torch.bool),
        _appear_spawned=torch.zeros(4, dtype=torch.long),
        _appear_azimuth_rad=torch.full((4,), torch.nan),
        _appear_distance_m=torch.full((4,), torch.nan),
    )
    seen = {}
    counters = {index: -1 for index in range(4)}
    for env_ids in schedule:
        scenario.reset_idx(torch.as_tensor(env_ids, dtype=torch.long))
        stamp = reset_appearance(
            fake, torch.as_tensor(env_ids, dtype=torch.long), list(env_ids)
        )
        times = stamp[GROUP_APPEARANCE]["time_s"]
        assert len(times) == len(env_ids)
        for row, env_index in enumerate(env_ids):
            assert 5.0 <= times[row] <= 30.0
            counters[env_index] += 1
            seen[(env_index, counters[env_index])] = times[row]
    return seen


def test_two_eval_seeds_are_two_exam_papers_and_one_seed_reproduces():
    first = _run_reset_schedule(42, SCHEDULE_A)
    again = _run_reset_schedule(42, SCHEDULE_A)
    assert first == again, "the same eval seed must reproduce every t0 exactly"
    other = _run_reset_schedule(123, SCHEDULE_A)
    assert any(first[key] != other[key] for key in first), (
        "eval seeds 42 and 123 drew identical appearance times: the axis is "
        "not under eval-seed control"
    )


def test_appearance_time_is_controller_independent():
    seen_a = _run_reset_schedule(2026, SCHEDULE_A)
    seen_b = _run_reset_schedule(2026, SCHEDULE_B)
    shared = sorted(set(seen_a) & set(seen_b))
    assert len(shared) >= 8
    for key in shared:
        assert seen_a[key] == seen_b[key], (
            f"env {key[0]} episode {key[1]} drew a different t0 under two "
            "termination patterns at the same eval seed -- the appearance "
            "time is riding a consumption-ordered stream, not the protocol"
        )


def test_shipped_reset_keeps_the_generator_it_drew_the_time_from():
    """The kept rng continues the SAME keyed stream past the t0 draw."""
    reset_appearance = _lift_env_method("_reset_appearance")
    scenario = make_scenario_rng(_Fake(seed=99), 4, DEVICE)
    scenario.reset_idx(torch.arange(4))
    fake = _Fake(
        _appear_count=1,
        _scenario=scenario,
        _layout_rng=np.random.default_rng(99),
        device=DEVICE,
        cfg=_Fake(appear_time_min_s=5.0, appear_time_max_s=30.0),
        _appear_rngs=[None] * 4,
        _appear_time_s=torch.full((4,), torch.inf),
        _appear_pending=torch.zeros(4, dtype=torch.bool),
        _appear_happened=torch.zeros(4, dtype=torch.bool),
        _appear_skipped=torch.zeros(4, dtype=torch.bool),
        _appear_spawned=torch.zeros(4, dtype=torch.long),
        _appear_azimuth_rad=torch.full((4,), torch.nan),
        _appear_distance_m=torch.full((4,), torch.nan),
    )
    reset_appearance(fake, torch.arange(4), [0, 1, 2, 3])
    reference = scenario.numpy_rng(GROUP_APPEARANCE, 2)
    reference.uniform(5.0, 30.0)  # burn the t0 draw
    assert (
        fake._appear_rngs[2].bit_generator.state
        == reference.bit_generator.state
    ), (
        "the stored generator is not the one the time was drawn from, so the "
        "azimuth permutation would come off a different (or restarted) stream"
    )


# ============================================================================
# 7. Registry pin and native layout
# ============================================================================
def test_new_id_carries_the_parent_native_layout():
    assert APPEAR_ID in NATIVE_LAYOUTS, (
        "the appearance id must have a NATIVE_LAYOUTS entry (sanctioned "
        "append flow; test_registry_drift enforces the same)"
    )
    assert NATIVE_LAYOUTS[APPEAR_ID] == NATIVE_LAYOUTS[PARENT_ID], (
        "the observation is unchanged, so the id must carry its certified "
        "crossing parent's layout verbatim"
    )


def test_new_id_is_registered_on_the_appearance_cfg():
    init_text = INIT_PATH.read_text(encoding="utf-8")
    assert f'id="{APPEAR_ID}"' in init_text
    block = init_text.split(f'id="{APPEAR_ID}"')[1].split("gym.register")[0]
    assert "HazardCrossAppearEnvCfg" in block

    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"), filename=str(CFG_PATH))
    base_default = None
    appear_cfg = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HazardNavEnvCfg":
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == "appear_count"
                ):
                    base_default = ast.literal_eval(item.value)
        if isinstance(node, ast.ClassDef) and node.name == "HazardCrossAppearEnvCfg":
            appear_cfg = node
    assert base_default == 0, (
        "the base cfg must default appear_count to 0 -- the axis has to be "
        "inert on every certified id"
    )
    assert appear_cfg is not None, "HazardCrossAppearEnvCfg is missing"
    parents = {
        base.id for base in appear_cfg.bases if isinstance(base, ast.Name)
    }
    assert "HazardForcedCrossingEnvCfg" in parents, (
        "the appearance id's parent must be the certified crossing cfg"
    )
    overrides = {
        item.target.id: ast.literal_eval(item.value)
        for item in appear_cfg.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }
    assert overrides.get("appear_count") == 1


def test_sources_are_pure_ascii():
    for path in (
        _HERE / "hazard_geometry.py",
        ENV_PATH,
        CFG_PATH,
        INIT_PATH,
        _SHARED / "scenario_draws_hazard.py",
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
    raise SystemExit(1 if failures else 0)
