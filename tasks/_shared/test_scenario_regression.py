# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Revert-detecting regression tests for the scenario protocol.

WHY THIS FILE EXISTS
--------------------
``tasks/_shared/test_scenario_rng.py`` proves the stream algebra and
``tasks/_shared/test_scenario_wiring.py`` proves the shared draw helpers in
``tasks/_shared/scenario_draws.py`` behave.  Neither reads a single line of an
ENV.  A reviewer reverted five separate fixes -- one per file below -- and every
suite in the tree stayed green, because the tests transcribe what the envs are
SUPPOSED to do instead of executing what they actually do.

So every test here runs the SHIPPED STATEMENTS.  The env modules import
``isaaclab`` at module scope and cannot be imported on a machine without Isaac
Sim, so each test AST-lifts the exact function, method or statement span it is
about out of the real source file and executes it against a fabricated ``self``.
That is the trick ``scripts/test_certificate_scenario_stamp.py:27-34`` and
``scripts/test_classical_baseline_math.py:110-125`` already use.  The
consequence that matters: editing the env source changes what these tests run,
so a reverted fix fails here instead of passing quietly.

WHAT EACH TEST OWNS (and what reverting it looks like)
-----------------------------------------------------
1. ``tasks/path_hazard/path_hazard_env.py:937-955`` ``_layout_generator``.
   Revert = ``return self._layout_rng`` unconditionally, i.e. the historical
   single numpy generator advanced per reset.  Detected by running two
   controllers whose episodes end at different times: the layout for
   (eval seed, env, episode) must not move.
2. ``tasks/station_keeping_boat/station_keeping_boat_env.py`` ``_reset_idx``
   spawn block.  Revert = three bare ``torch.rand`` calls.  Detected because
   the global torch RNG then moves the spawn and the eval seed does not.
3. ``tasks/path_following/path_following_env.py`` ``stamp_scenario`` call.
   Revert = stamping ``headings_rad`` (the running sum) instead of the raw
   ``first_heading_rad`` / ``heading_change_rad`` draws.  Detected with two
   ORDINARY float32 unit draws whose absolute headings are byte-identical.
4. ``tasks/station_keeping/station_keeping_env.py`` ``_resample_current`` and
   its ``scenario_groups`` block.  Revert = stamping ``current_vec`` (the draw
   after the trigonometry).  Detected with two currents whose vector is
   byte-identical and whose (speed, direction) draw is not.
5. ``scripts/eval_v6_frozen.py:154-158``.  Revert = stamping the protocol
   header unconditionally.  Detected with an env carrying no ``_scenario``.

Plus one guard that does NOT exist yet and must be reported rather than
patched here: ``ScenarioRNG.numpy_rng`` has no pre-reset check while the torch
draw path does (``tasks/_shared/scenario_rng.py:592-597``), so calling it before
any reset silently keys on episode index -1.

Run: python tasks/_shared/test_scenario_regression.py
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .scenario_draws import (
        SCENARIO_PROTOCOL_OFF,
        TWO_PI,
        make_scenario_rng,
        path_following_route,
        scenario_protocol_header,
        spawn_heading,
        stamp_scenario,
        station_keeping_current,
        station_keeping_spawn,
    )
    from .scenario_rng import ScenarioRNG
except ImportError:  # direct execution
    from scenario_draws import (
        SCENARIO_PROTOCOL_OFF,
        TWO_PI,
        make_scenario_rng,
        path_following_route,
        scenario_protocol_header,
        spawn_heading,
        stamp_scenario,
        station_keeping_current,
        station_keeping_spawn,
    )
    from scenario_rng import ScenarioRNG


DEVICE = torch.device("cpu")
NUM_ENVS = 4

_SHARED = Path(__file__).resolve().parent
_TASKS = _SHARED.parent
_REPO = _TASKS.parent

PATH_HAZARD_ENV = _TASKS / "path_hazard" / "path_hazard_env.py"
PATH_HAZARD_GEOMETRY = _TASKS / "path_hazard" / "path_hazard_geometry.py"
PATH_HAZARD_CFG = _TASKS / "path_hazard" / "path_hazard_env_cfg.py"
PATH_FOLLOWING_ENV = _TASKS / "path_following" / "path_following_env.py"
PATH_FOLLOWING_CFG = _TASKS / "path_following" / "path_following_env_cfg.py"
STATION_KEEPING_ENV = _TASKS / "station_keeping" / "station_keeping_env.py"
STATION_KEEPING_CFG = _TASKS / "station_keeping" / "station_keeping_env_cfg.py"
BOAT_ENV = _TASKS / "station_keeping_boat" / "station_keeping_boat_env.py"
BOAT_CFG = _TASKS / "station_keeping_boat" / "station_keeping_boat_env_cfg.py"
EVAL_V6_FROZEN = _REPO / "scripts" / "eval_v6_frozen.py"

SOURCES_UNDER_TEST = (
    PATH_HAZARD_ENV,
    PATH_FOLLOWING_ENV,
    STATION_KEEPING_ENV,
    BOAT_ENV,
    EVAL_V6_FROZEN,
    Path(__file__).resolve(),
)


# --------------------------------------------------------------- AST lifting --
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assigned_names(node: ast.stmt) -> set[str]:
    """Every plain name this statement binds (``a``, ``a, b``, ``a[i]`` -> a)."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return set()
    names = set()
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
    return names


def _calls(node: ast.AST, name: str) -> bool:
    """Does this subtree call the bare function ``name``?"""
    return any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == name
        for inner in ast.walk(node)
    )


def _calls_method(node: ast.AST, name: str) -> bool:
    """Does this subtree call the method ``.name(...)`` on anything?"""
    return any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == name
        for inner in ast.walk(node)
    )


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{path.name}: no {class_name}.{method_name}")


def _exec_statements(statements, path: Path, namespace: dict) -> dict:
    """Execute shipped statements verbatim, with their real line numbers."""
    module = ast.Module(body=list(statements), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def _lift_callable(path: Path, class_name: str, method_name: str,
                   namespace: dict):
    """Compile one shipped method into a plain function of (self, ...)."""
    _exec_statements([_method(path, class_name, method_name)], path, namespace)
    return namespace[method_name]


def _span(function: ast.FunctionDef, path: Path, starts, ends):
    """The shipped statements from the first ``starts`` to the last ``ends``."""
    body = function.body
    first = [i for i, stmt in enumerate(body) if starts(stmt)]
    last = [i for i, stmt in enumerate(body) if ends(stmt)]
    assert first, f"{path.name}: no start statement in {function.name}"
    assert last, f"{path.name}: no end statement in {function.name}"
    assert last[-1] >= first[0], (
        f"{path.name}: {function.name} end statement precedes the start"
    )
    return body[first[0]:last[-1] + 1]


def _module_constant(path: Path, name: str):
    """A module-level ``NAME = <literal>`` read out of the source."""
    values = [
        node.value.value
        for node in _tree(path).body
        if isinstance(node, ast.Assign)
        and name in _assigned_names(node)
        and isinstance(node.value, ast.Constant)
    ]
    assert len(values) == 1, f"{path.name}: {name} defined {len(values)} times"
    return values[0]


def _cfg_default(path: Path, field: str):
    """A dataclass cfg default (``field: type = <literal>``) from the source.

    Using the SHIPPED value keeps the range checks honest: the test asserts the
    draws land inside the range the benchmark actually configures, not inside a
    range the test invented.
    """
    values = []
    for node in ast.walk(_tree(path)):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == field
            and isinstance(node.value, ast.Constant)
        ):
            values.append(node.value.value)
    assert values, f"{path.name}: no cfg default for {field}"
    assert len(set(values)) == 1, (
        f"{path.name}: {field} has conflicting defaults {sorted(set(values))}"
    )
    return values[0]


def _load_module(path: Path, name: str):
    """Import an Isaac-free module by FILE, bypassing its package __init__.

    ``tasks/path_hazard/__init__.py:6`` imports gymnasium and registers gym ids,
    so ``import tasks.path_hazard.path_hazard_geometry`` cannot run here even
    though the geometry module itself is numpy/torch only.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GEOMETRY = _load_module(PATH_HAZARD_GEOMETRY, "_scenario_regression_ph_geometry")
sample_layout = _GEOMETRY.sample_layout


class _Fake:
    """A stand-in env: only the attributes the lifted statements read."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _Cfg:
    """The single attribute make_scenario_rng reads off an env cfg."""

    def __init__(self, seed):
        self.seed = seed


class _CannedScenario:
    """Feeds PINNED U[0, 1) draws through the shipped draw helpers.

    Same call shape as ``ScenarioRNG.uniform`` (see
    ``tasks/_shared/scenario_draws.py:142``), so the helper under test applies
    its real arithmetic and its real ranges to numbers the test chose.  That is
    how a float32 collision can be exhibited with draws a real ``torch.rand``
    could have produced.
    """

    def __init__(self, draws):
        self._draws = {group: list(values) for group, values in draws.items()}

    def uniform(self, group, env_ids, low=0.0, high=1.0, size=()):
        queue = self._draws.get(group)
        assert queue, f"no canned draw left for group {group!r}"
        value = queue.pop(0)
        expected = (len(env_ids), *tuple(int(n) for n in size))
        assert tuple(value.shape) == expected, (
            f"canned {group} draw has shape {tuple(value.shape)}, the helper "
            f"asked for {expected}"
        )
        assert (low, high) == (0.0, 1.0), "helpers must draw raw U[0, 1)"
        return value


def _skrl_clobber():
    """Reproduce skrl Runner.__init__ -> set_seed(agent yaml seed).

    The agent YAMLs pin a constant and the Runner is built AFTER the env, so the
    global torch RNG sits at the SAME state for every evaluation seed by the
    time the first episode is drawn.  Without this line a two-seed comparison
    passes even on the broken code, because the global stream simply keeps
    advancing between the two calls.
    """
    torch.manual_seed(42)


# ============================================================================
# 1. path_hazard: the layout must be keyed, not stream-ordered
# ============================================================================
GROUP_LAYOUT = _module_constant(PATH_HAZARD_ENV, "GROUP_LAYOUT")
PH_OBSTACLE_COUNT = _cfg_default(PATH_HAZARD_CFG, "obstacle_count")
PH_MAX_ATTEMPTS = _cfg_default(PATH_HAZARD_CFG, "layout_max_attempts")


def layout_generator_for(scenario, layout_rng):
    """The SHIPPED ``PathHazardEnv._layout_generator``, bound to a fake self."""
    namespace = {"np": np, "torch": torch, "GROUP_LAYOUT": GROUP_LAYOUT}
    shipped = _lift_callable(
        PATH_HAZARD_ENV, "PathHazardEnv", "_layout_generator", namespace
    )
    base = _Fake(_scenario=scenario, _layout_rng=layout_rng)
    return lambda env_index: shipped(base, env_index)


def _layout_signature(layout):
    """Everything about a layout a certificate would have to agree on."""
    return {
        "waypoints": np.asarray(layout.waypoints, dtype=np.float64).copy(),
        "centers": np.asarray(layout.centers, dtype=np.float64).copy(),
        "radii": np.asarray(layout.radii, dtype=np.float64).copy(),
        "route_length": float(layout.route_length),
        "obstacle_count": int(layout.obstacle_count),
    }


def _same_layout(left, right):
    return (
        np.array_equal(left["waypoints"], right["waypoints"])
        and np.array_equal(left["centers"], right["centers"])
        and np.array_equal(left["radii"], right["radii"])
        and left["route_length"] == right["route_length"]
        and left["obstacle_count"] == right["obstacle_count"]
    )


def _path_hazard_episodes(seed, schedule):
    """Transcribe path_hazard_env._reset_idx's per-row layout loop.

    ``schedule`` is the list of env-id batches in the order a controller
    happened to end episodes.  The layout loop
    (``tasks/path_hazard/path_hazard_env.py:1044-1057``) calls
    ``self._layout_generator(reset_env_indices[row])`` once per ROW, which is
    what the lifted generator is driven with here; ``sample_layout`` and its
    attempt budget are the shipped ones.
    """
    scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
    generator_for = layout_generator_for(scenario, np.random.default_rng(seed))
    seen = {}
    counters = {index: -1 for index in range(NUM_ENVS)}
    for env_ids in schedule:
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        scenario.reset_idx(ids)
        headings = spawn_heading(scenario, ids, DEVICE)
        for row, env_index in enumerate(env_ids):
            layout = sample_layout(
                rng=generator_for(env_index),
                max_attempts=PH_MAX_ATTEMPTS,
                obstacle_count=PH_OBSTACLE_COUNT,
            )
            counters[env_index] += 1
            signature = _layout_signature(layout)
            signature["spawn_heading"] = float(headings[row])
            seen[(env_index, counters[env_index])] = signature
    return seen


def test_path_hazard_layout_survives_a_different_termination_order():
    """Fix 1. Two controllers, two reset orders, one layout per (env, episode).

    Controller A runs to the horizon in lockstep pairs; controller B ends
    episodes one env at a time and finishes an extra round.  The number of
    layouts drawn BEFORE env i's k-th therefore differs between them, which is
    exactly what a single generator advanced per reset cannot survive.
    """
    seed = 2026
    schedule_a = [[0, 1, 2, 3], [1, 3], [0, 2], [1, 3], [0, 2]]
    schedule_b = [[0, 1, 2, 3], [0], [2], [1], [3], [0, 1, 2, 3]]

    seen_a = _path_hazard_episodes(seed, schedule_a)
    seen_b = _path_hazard_episodes(seed, schedule_b)

    shared = sorted(set(seen_a) & set(seen_b))
    assert len(shared) >= 8, f"only {len(shared)} shared episodes to compare"
    for key in shared:
        assert _same_layout(seen_a[key], seen_b[key]), (
            f"env {key[0]} episode {key[1]}: the path_hazard LAYOUT depends on "
            "when the episode happened, so two controllers at one eval seed "
            "did not navigate the same obstacle field -- route_geodesic_m and "
            "d0_m cannot be paired across certificates"
        )
        assert seen_a[key]["spawn_heading"] == seen_b[key]["spawn_heading"], (
            f"env {key[0]} episode {key[1]}: spawn heading moved with the "
            "reset order"
        )


def test_path_hazard_layout_actually_varies():
    """Teeth for the test above: a constant layout would satisfy it trivially."""
    seen = _path_hazard_episodes(2026, [[0, 1, 2, 3], [0, 1, 2, 3]])
    assert not _same_layout(seen[(0, 0)], seen[(1, 0)]), (
        "two different envs drew the identical layout"
    )
    assert not _same_layout(seen[(0, 0)], seen[(0, 1)]), (
        "two consecutive episodes of one env drew the identical layout"
    )
    other = _path_hazard_episodes(7, [[0, 1, 2, 3]])
    assert not _same_layout(seen[(0, 0)], other[(0, 0)]), (
        "the path_hazard layout is identical under two evaluation seeds"
    )


# ============================================================================
# 2. station_keeping_boat: the spawn must come off the protocol
# ============================================================================
BOAT_MIN_SPAWN = _cfg_default(BOAT_CFG, "min_spawn_distance")
BOAT_MAX_SPAWN = _cfg_default(BOAT_CFG, "max_spawn_distance")

# The span starts at the episode-advance guard, not at the first draw: the
# ORDER (reset_idx, then draw) is part of what makes the scenario keyed by
# episode index, so the test has to execute the shipped ordering rather than
# supply its own reset. The ``num_resets`` alternative keeps the span findable
# on a build where the guard is gone, so such a build fails on the property
# under test rather than on statement selection.
_BOAT_SPAWN_STATEMENTS = _span(
    _method(BOAT_ENV, "StationKeepingBoatEnv", "_reset_idx"),
    BOAT_ENV,
    starts=lambda stmt: _calls_method(stmt, "reset_idx")
    or "num_resets" in _assigned_names(stmt),
    ends=lambda stmt: "headings" in _assigned_names(stmt),
)


def _boat_spawn(scenario, env_ids, num_envs=NUM_ENVS):
    """Run the SHIPPED station_keeping_boat spawn statements once.

    The shipped statements advance the episode counter themselves, so callers
    must NOT call ``reset_idx`` around this helper.
    """
    base = _Fake(
        _scenario=scenario,
        device=DEVICE,
        num_envs=num_envs,
        cfg=_Fake(
            min_spawn_distance=BOAT_MIN_SPAWN,
            max_spawn_distance=BOAT_MAX_SPAWN,
        ),
    )
    namespace = {
        "torch": torch,
        "station_keeping_spawn": station_keeping_spawn,
        "self": base,
        "env_ids": torch.as_tensor(env_ids, dtype=torch.long),
    }
    _exec_statements(_BOAT_SPAWN_STATEMENTS, BOAT_ENV, namespace)
    return {
        "spawn_distance": namespace["distances"],
        "spawn_angle": namespace["spawn_angles"],
        "spawn_heading": namespace["headings"],
    }


def _same(left, right):
    return all(torch.equal(left[name], right[name]) for name in left)


def test_station_keeping_boat_spawn_is_on_the_protocol():
    """Fix 2. The eval seed must reach the spawn; the global RNG must not."""
    all_ids = list(range(NUM_ENVS))

    drawn = {}
    for seed in (123, 42):
        scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
        _skrl_clobber()
        drawn[seed] = _boat_spawn(scenario, all_ids)
    for name in drawn[123]:
        assert not torch.equal(drawn[123][name], drawn[42][name]), (
            f"station_keeping_boat {name} is identical under eval seeds 123 "
            "and 42: the spawn is still on the global torch RNG, which skrl's "
            "Runner reseeds to the agent YAML constant after the env is built"
        )

    scenario = make_scenario_rng(_Cfg(555), NUM_ENVS, DEVICE)
    torch.manual_seed(0)
    protected = _boat_spawn(scenario, all_ids)
    scenario = make_scenario_rng(_Cfg(555), NUM_ENVS, DEVICE)
    torch.manual_seed(12345)
    again = _boat_spawn(scenario, all_ids)
    assert _same(protected, again), (
        "the station_keeping_boat spawn moved when the global torch RNG was "
        "reseeded"
    )

    # Negative control: the historical path is exactly what a global reseed
    # destroys, so this proves the assertion above can tell fixed from broken.
    torch.manual_seed(0)
    old_a = _boat_spawn(None, all_ids)
    torch.manual_seed(12345)
    old_b = _boat_spawn(None, all_ids)
    assert not _same(old_a, old_b), (
        "the pre-fix global-RNG path did not react to the global seed, so this "
        "test cannot distinguish fixed from broken"
    )


def test_station_keeping_boat_advances_the_episode_before_drawing():
    """reset_idx must run BEFORE the spawn draw, or the key names last episode."""
    first = _BOAT_SPAWN_STATEMENTS[0]
    assert _calls_method(first, "reset_idx"), (
        "the first statement of the station_keeping_boat spawn block is not the "
        "scenario reset, so the three draws are keyed on the PREVIOUS episode "
        "index (or on -1 for an env that has never reset)"
    )
    drawing = [
        index for index, stmt in enumerate(_BOAT_SPAWN_STATEMENTS)
        if _calls(stmt, "station_keeping_spawn")
    ]
    assert drawing, (
        "the station_keeping_boat spawn no longer goes through "
        "scenario_draws.station_keeping_spawn, so it is not on the protocol"
    )
    assert min(drawing) > 0, "the spawn is drawn before the episode advances"


def test_station_keeping_boat_spawn_is_controller_independent():
    """Env i's k-th spawn is the same under two reset/termination orders."""
    seed = 4242
    schedules = {
        "a": [[0, 1, 2, 3], [1, 3], [0, 2], [1, 3]],
        "b": [[0, 1, 2, 3], [0], [2], [1], [3], [0, 1, 2, 3]],
    }
    seen = {}
    for label, schedule in schedules.items():
        scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
        counters = {index: -1 for index in range(NUM_ENVS)}
        for env_ids in schedule:
            drawn = _boat_spawn(scenario, env_ids)
            for row, env_index in enumerate(env_ids):
                counters[env_index] += 1
                seen[(label, env_index, counters[env_index])] = {
                    name: value[row].clone() for name, value in drawn.items()
                }
    keys_a = {key[1:] for key in seen if key[0] == "a"}
    keys_b = {key[1:] for key in seen if key[0] == "b"}
    shared = sorted(keys_a & keys_b)
    assert len(shared) >= 8, f"only {len(shared)} shared episodes to compare"
    for env_index, episode in shared:
        for name in ("spawn_distance", "spawn_angle", "spawn_heading"):
            assert torch.equal(
                seen[("a", env_index, episode)][name],
                seen[("b", env_index, episode)][name],
            ), (
                f"env {env_index} episode {episode}: station_keeping_boat "
                f"{name} differs between two controllers at one eval seed"
            )


def test_station_keeping_boat_spawn_ranges_are_unchanged():
    """Only the SOURCE of the numbers changed -- never a range or a shape.

    Bounds are the shipped cfg defaults, read out of
    ``station_keeping_boat_env_cfg.py`` rather than retyped here, so this fails
    if the migration quietly adopted the sibling family's ring.
    """
    wide = 512
    scenario = ScenarioRNG(num_envs=wide, device=DEVICE, eval_seed=3)
    ids = torch.arange(wide)
    drawn = _boat_spawn(scenario, ids, num_envs=wide)

    distances = drawn["spawn_distance"]
    assert distances.shape == (wide,), distances.shape
    assert bool((distances >= BOAT_MIN_SPAWN).all()), float(distances.min())
    assert bool((distances < BOAT_MAX_SPAWN).all()), float(distances.max())
    span = BOAT_MAX_SPAWN - BOAT_MIN_SPAWN
    midpoint = BOAT_MIN_SPAWN + 0.5 * span
    assert abs(float(distances.mean()) - midpoint) < 0.03 * span, (
        f"spawn distance mean {float(distances.mean())} is not the midpoint "
        f"{midpoint} of [{BOAT_MIN_SPAWN}, {BOAT_MAX_SPAWN})"
    )
    assert float(distances.min()) < BOAT_MIN_SPAWN + 0.02 * span
    assert float(distances.max()) > BOAT_MAX_SPAWN - 0.02 * span

    for name in ("spawn_angle", "spawn_heading"):
        value = drawn[name]
        assert value.shape == (wide,), (name, value.shape)
        assert bool((value >= 0.0).all() and (value < TWO_PI).all()), (
            f"{name} left [0, 2*pi)"
        )
        assert abs(float(value.mean()) - math.pi) < 0.03 * TWO_PI, (
            f"{name} mean {float(value.mean())} is not pi"
        )


# ============================================================================
# 3. path_following: the stamp must carry the RAW heading draws
# ============================================================================
PF_NUM_WAYPOINTS = _cfg_default(PATH_FOLLOWING_CFG, "num_waypoints")
PF_SEGMENT_MIN = _cfg_default(PATH_FOLLOWING_CFG, "segment_length_min")
PF_SEGMENT_MAX = _cfg_default(PATH_FOLLOWING_CFG, "segment_length_max")
PF_HEADING_CHANGE_MAX_DEG = _cfg_default(
    PATH_FOLLOWING_CFG, "heading_change_max_deg"
)


def _stamp_loop(path: Path, class_name: str):
    """The shipped ``for ... in stamp_scenario(...)`` statement of a reset."""
    loops = [
        stmt
        for stmt in _method(path, class_name, "_reset_idx").body
        if isinstance(stmt, ast.For) and _calls(stmt.iter, "stamp_scenario")
    ]
    assert len(loops) == 1, (
        f"{path.name}: expected one stamp_scenario loop, found {len(loops)}"
    )
    return loops[0]


_PF_STAMP_LOOP = _stamp_loop(PATH_FOLLOWING_ENV, "PathFollowingEnv")


def _path_following_route_from(unit_segments, unit_first, unit_changes):
    """Drive the SHIPPED route helper with pinned U[0, 1) draws."""
    scenario = _CannedScenario(
        {"route": [unit_segments, unit_first, unit_changes]}
    )
    max_change = torch.deg2rad(
        torch.tensor(PF_HEADING_CHANGE_MAX_DEG, device=DEVICE)
    )
    return path_following_route(
        scenario,
        torch.zeros(unit_first.shape[0], dtype=torch.long),
        DEVICE,
        PF_NUM_WAYPOINTS,
        PF_SEGMENT_MIN,
        PF_SEGMENT_MAX,
        max_change,
    )


def _path_following_stamp(segments, first_headings, heading_changes,
                          spawn_headings):
    """Run the SHIPPED path_following stamp_scenario loop once."""
    count = first_headings.shape[0]
    # The two lines path_following_env.py:704-705 use to build the absolute
    # heading chain, so a stamp that reverted to the derived value still finds
    # the name it reads.
    headings = torch.empty((count, PF_NUM_WAYPOINTS), device=DEVICE)
    headings[:, 0] = first_headings
    headings[:, 1:] = headings[:, 0:1] + torch.cumsum(heading_changes, dim=1)

    segment_buffer = torch.zeros((count, PF_NUM_WAYPOINTS), device=DEVICE)
    env_ids = torch.arange(count, dtype=torch.long)
    segment_buffer[env_ids] = segments
    base = _Fake(
        segment_lengths=segment_buffer,
        waypoints=torch.zeros((count, PF_NUM_WAYPOINTS, 2), device=DEVICE),
        route_length=torch.zeros(count, device=DEVICE),
        _scenario_params=[{} for _ in range(count)],
        _scenario_hashes=[{} for _ in range(count)],
    )
    namespace = {
        "torch": torch,
        "stamp_scenario": stamp_scenario,
        "self": base,
        "env_ids": env_ids,
        "segment_lengths": segments,
        "first_headings": first_headings,
        "heading_changes": heading_changes,
        "headings": headings,
        "spawn_headings": spawn_headings,
        # What the shipped reset produces for every cfg without a sea_state
        # field (the certified ids): the Wave-variant block above the stamp
        # leaves wave_scenario empty, so the splice adds no group -- the same
        # fabrication discipline as _sea=None on the station-keeping stamp.
        "wave_scenario": {},
    }
    _exec_statements([_PF_STAMP_LOOP], PATH_FOLLOWING_ENV, namespace)
    return headings, base._scenario_params, base._scenario_hashes


def test_path_following_stamps_the_raw_heading_draws():
    """Fix 3. Two routes with the same ABSOLUTE headings must not share a hash.

    The unit draws below are ordinary float32 values a real ``torch.rand``
    produces -- 0.75 and its float32 neighbour -- not the sub-ulp tails the
    source comment uses for its hand case.  Through the shipped
    ``path_following_route`` they give heading changes that differ, and an
    absolute heading chain that is byte-identical, because the third change is
    lost in the ulp of the running total.  Stamping the running total therefore
    hands two different exam papers one digest.
    """
    unit_segments = torch.full((1, PF_NUM_WAYPOINTS), 0.5, dtype=torch.float32)
    unit_first = torch.tensor([0.75], dtype=torch.float32)
    unit_changes_a = torch.full(
        (1, PF_NUM_WAYPOINTS - 1), 0.75, dtype=torch.float32
    )
    unit_changes_b = unit_changes_a.clone()
    unit_changes_b[0, -1] = torch.tensor(
        0.75, dtype=torch.float32
    ) + 2.0 ** -24
    assert not torch.equal(unit_changes_a, unit_changes_b), (
        "the two unit draws collapsed onto one float32 value"
    )

    segments_a, first_a, changes_a = _path_following_route_from(
        unit_segments, unit_first, unit_changes_a
    )
    segments_b, first_b, changes_b = _path_following_route_from(
        unit_segments, unit_first, unit_changes_b
    )
    spawn = torch.tensor([1.5], device=DEVICE)

    headings_a, resolved_a, hashes_a = _path_following_stamp(
        segments_a, first_a, changes_a, spawn
    )
    headings_b, resolved_b, hashes_b = _path_following_stamp(
        segments_b, first_b, changes_b, spawn
    )

    # The premise, asserted rather than assumed: same derived chain, different
    # draw. If float32 ever stops collapsing these the test says so instead of
    # passing vacuously.
    assert torch.equal(headings_a, headings_b), (
        "the two routes no longer share an absolute heading chain, so this "
        "test can no longer tell a raw stamp from a derived one"
    )
    assert torch.equal(segments_a, segments_b) and torch.equal(first_a, first_b)
    assert not torch.equal(changes_a, changes_b), "the raw draws are identical"

    assert hashes_a[0]["route"] != hashes_b[0]["route"], (
        "two path_following routes with different heading DRAWS share a route "
        "digest: the stamp is recording the running sum (headings_rad), not "
        "the raw first_heading_rad / heading_change_rad the stream produced"
    )
    assert hashes_a[0]["scenario"] != hashes_b[0]["scenario"], (
        "the whole-scenario digest is blind to the heading draw"
    )

    # And the raw form loses nothing: the absolute chain is recoverable.
    route = resolved_a[0]["route"]
    for name in ("first_heading_rad", "heading_change_rad"):
        assert name in route, (
            f"the stamped route has no {name}; the certificate cannot identify "
            f"the draw. Stamped keys: {sorted(route)}"
        )
    rebuilt = torch.empty((1, PF_NUM_WAYPOINTS), device=DEVICE)
    rebuilt[:, 0] = torch.tensor([route["first_heading_rad"]], device=DEVICE)
    rebuilt[:, 1:] = rebuilt[:, 0:1] + torch.cumsum(
        torch.tensor([route["heading_change_rad"]], device=DEVICE), dim=1
    )
    assert torch.equal(rebuilt, headings_a), (
        "the stamped raw draws do not reconstruct the absolute headings"
    )


# ============================================================================
# 4. station_keeping: the stamp must carry the RAW current draw
# ============================================================================
SK_CURRENT_MIN = _cfg_default(STATION_KEEPING_CFG, "current_speed_min")
SK_CURRENT_MAX = _cfg_default(STATION_KEEPING_CFG, "current_speed_max")

_SK_RESAMPLE = _method(STATION_KEEPING_ENV, "StationKeepingEnv",
                       "_resample_current")
_SK_CURRENT_VEC_WRITES = [
    stmt
    for stmt in _SK_RESAMPLE.body
    if isinstance(stmt, ast.Assign)
    and any(
        isinstance(sub, ast.Attribute) and sub.attr == "current_vec"
        for target in stmt.targets
        for sub in ast.walk(target)
    )
]
assert len(_SK_CURRENT_VEC_WRITES) == 2, (
    "station_keeping_env._resample_current no longer writes current_vec twice"
)

_SK_STAMP_STATEMENTS = _span(
    _method(STATION_KEEPING_ENV, "StationKeepingEnv", "_reset_idx"),
    STATION_KEEPING_ENV,
    starts=lambda stmt: "scenario_groups" in _assigned_names(stmt),
    ends=lambda stmt: isinstance(stmt, ast.For)
    and _calls(stmt.iter, "stamp_scenario"),
)


def _resample_current(scenario, env_ids, speed_min, speed_max, num_envs):
    """Call the SHIPPED ``StationKeepingEnv._resample_current``."""
    namespace = {
        "torch": torch,
        "station_keeping_current": station_keeping_current,
    }
    shipped = _lift_callable(
        STATION_KEEPING_ENV, "StationKeepingEnv", "_resample_current", namespace
    )
    base = _Fake(
        _scenario=scenario,
        device=DEVICE,
        num_envs=num_envs,
        current_vec=torch.zeros((num_envs, 2), device=DEVICE),
        physics_cfg=_Fake(
            current_speed_min=speed_min, current_speed_max=speed_max
        ),
    )
    returned = shipped(base, torch.as_tensor(env_ids, dtype=torch.long))
    return base, returned


def _station_keeping_stamp(speeds, directions, distances, spawn_angles,
                           headings):
    """Run the SHIPPED station_keeping scenario_groups block once.

    ``current_vec`` is written by the two shipped assignment statements lifted
    out of ``_resample_current``, so a stamp that reverted to the vector reads
    exactly the bytes the env would have produced.
    """
    count = speeds.shape[0]
    env_ids = torch.arange(count, dtype=torch.long)
    base = _Fake(
        _sea=None,
        _scenario_params=[{} for _ in range(count)],
        _scenario_hashes=[{} for _ in range(count)],
        current_vec=torch.zeros((count, 2), device=DEVICE),
        physics_cfg=_Fake(enable_current=True),
    )
    _exec_statements(
        _SK_CURRENT_VEC_WRITES,
        STATION_KEEPING_ENV,
        {"torch": torch, "self": base, "env_ids": env_ids,
         "speeds": speeds, "directions": directions},
    )
    namespace = {
        "torch": torch,
        "stamp_scenario": stamp_scenario,
        "self": base,
        "env_ids": env_ids,
        "current_speeds": speeds,
        "current_directions": directions,
        "distances": distances,
        "spawn_angles": spawn_angles,
        "headings": headings,
    }
    _exec_statements(_SK_STAMP_STATEMENTS, STATION_KEEPING_ENV, namespace)
    return base.current_vec.clone(), base._scenario_params, base._scenario_hashes


def test_station_keeping_resample_current_returns_the_raw_draw():
    """Fix 4a. The caller cannot stamp the draw if the method hides it."""
    scenario = make_scenario_rng(_Cfg(11), NUM_ENVS, DEVICE)
    ids = torch.arange(NUM_ENVS)
    scenario.reset_idx(ids)
    base, returned = _resample_current(
        scenario, ids, SK_CURRENT_MIN, SK_CURRENT_MAX, NUM_ENVS
    )
    assert returned is not None, (
        "_resample_current returned None: the raw (speed, direction) draw never "
        "reaches _reset_idx, so the certificate can only stamp current_vec"
    )
    speeds, directions = returned
    assert speeds.shape == (NUM_ENVS,) and directions.shape == (NUM_ENVS,)
    assert bool((speeds >= SK_CURRENT_MIN).all()
                and (speeds < SK_CURRENT_MAX).all()), "current speed range"
    assert bool((directions >= 0.0).all() and (directions < TWO_PI).all())
    # The physics path is untouched: current_vec is still speed * (cos, sin).
    assert torch.equal(base.current_vec[:, 0], speeds * torch.cos(directions))
    assert torch.equal(base.current_vec[:, 1], speeds * torch.sin(directions))


def test_station_keeping_stamps_the_raw_current_draw():
    """Fix 4b. Two currents with one vector must not share a digest.

    At ``current_speed_min = 0.0`` -- which nothing in the cfg schema forbids --
    every direction collapses onto the same ``(0.0, 0.0)`` vector, so a
    certificate that stamps the vector cannot tell those episodes apart.  Both
    directions here are first-quadrant, so the zeros are both POSITIVE zero and
    the collision is exact rather than a signed-zero artefact.
    """
    count = 1
    distances = torch.tensor([7.5], device=DEVICE)
    spawn_angles = torch.tensor([1.0], device=DEVICE)
    headings = torch.tensor([2.0], device=DEVICE)

    speeds = torch.zeros(count, device=DEVICE)
    direction_a = torch.tensor([0.1], device=DEVICE) * TWO_PI
    direction_b = torch.tensor([0.2], device=DEVICE) * TWO_PI

    vec_a, _resolved_a, hashes_a = _station_keeping_stamp(
        speeds, direction_a, distances, spawn_angles, headings
    )
    vec_b, resolved_b, hashes_b = _station_keeping_stamp(
        speeds, direction_b, distances, spawn_angles, headings
    )

    assert torch.equal(vec_a, vec_b), (
        "the two currents no longer share a vector, so this test can no longer "
        "tell a raw stamp from a derived one"
    )
    assert not torch.equal(direction_a, direction_b)
    assert hashes_a[0]["current"] != hashes_b[0]["current"], (
        "two currents with different DRAWN directions share a current digest: "
        "the stamp is recording current_vec (the draw after the trigonometry), "
        "not the raw speed_mps / direction_rad"
    )
    assert hashes_a[0]["scenario"] != hashes_b[0]["scenario"], (
        "the whole-scenario digest is blind to the current draw"
    )
    current = resolved_b[0]["current"]
    for name in ("speed_mps", "direction_rad"):
        assert name in current, (
            f"the stamped current has no {name}. Stamped keys: {sorted(current)}"
        )


def test_station_keeping_stamps_the_raw_current_draw_inside_the_shipped_range():
    """The same defect at the shipped speed range, not only at zero.

    ``0.5000017285346985`` and ``0.5000017881393433`` are adjacent float32
    directions; at 0.2 m/s -- the shipped ``current_speed_min`` -- both
    components of ``speed * (cos, sin)`` round to the same float32, so the
    vector is byte-identical while the drawn direction is not.
    """
    speeds = torch.full((1,), SK_CURRENT_MIN, device=DEVICE)
    direction_a = torch.tensor([0.5000017285346985], device=DEVICE)
    direction_b = torch.tensor([0.5000017881393433], device=DEVICE)
    distances = torch.tensor([7.5], device=DEVICE)
    spawn_angles = torch.tensor([1.0], device=DEVICE)
    headings = torch.tensor([2.0], device=DEVICE)

    vec_a, _params_a, hashes_a = _station_keeping_stamp(
        speeds, direction_a, distances, spawn_angles, headings
    )
    vec_b, _params_b, hashes_b = _station_keeping_stamp(
        speeds, direction_b, distances, spawn_angles, headings
    )
    assert not torch.equal(direction_a, direction_b)
    assert torch.equal(vec_a, vec_b), (
        "the pinned float32 pair no longer collides on this build; pick a new "
        "pair rather than dropping the check"
    )
    assert hashes_a[0]["current"] != hashes_b[0]["current"], (
        "two currents inside the shipped speed range share a digest because "
        "the stamp records the post-trigonometry vector"
    )


# ============================================================================
# 5. eval_v6_frozen: the header must be honest about an off-protocol env
# ============================================================================
_EVAL_PROTOCOL_STATEMENTS = [
    stmt
    for stmt in _tree(EVAL_V6_FROZEN).body
    if isinstance(stmt, ast.Assign) and "scenario_protocol" in _assigned_names(stmt)
]


def _eval_v6_stamp(base):
    """Run eval_v6_frozen.py's OWN scenario_protocol statement(s)."""
    assert len(_EVAL_PROTOCOL_STATEMENTS) == 1, (
        f"eval_v6_frozen.py assigns scenario_protocol "
        f"{len(_EVAL_PROTOCOL_STATEMENTS)} times"
    )
    namespace = {
        "base": base,
        "scenario_protocol_header": scenario_protocol_header,
        "SCENARIO_PROTOCOL_OFF": SCENARIO_PROTOCOL_OFF,
    }
    _exec_statements(_EVAL_PROTOCOL_STATEMENTS, EVAL_V6_FROZEN, namespace)
    return namespace["scenario_protocol"]


def test_eval_v6_frozen_header_is_honest_about_an_off_protocol_env():
    """Fix 5. No ``_scenario`` on the env -> the off-protocol marker."""
    never_migrated = _Fake(num_envs=2)
    assert not hasattr(never_migrated, "_scenario")
    assert _eval_v6_stamp(never_migrated) == SCENARIO_PROTOCOL_OFF, (
        "eval_v6_frozen.py stamped the protocol header on an env that carries "
        "no ScenarioRNG; every certificate would then claim episodes are "
        "paired across evaluation seeds when they are not"
    )

    unseeded = _Fake(num_envs=2, _scenario=None)
    assert _eval_v6_stamp(unseeded) == SCENARIO_PROTOCOL_OFF, (
        "an env built with cfg.seed None fell back to the historical global-RNG "
        "line, so it is off protocol too"
    )

    on_protocol = _Fake(
        num_envs=2,
        _scenario=ScenarioRNG(num_envs=2, device=DEVICE, eval_seed=1),
    )
    assert _eval_v6_stamp(on_protocol) == scenario_protocol_header(), (
        "a migrated, seeded env must stamp the header block"
    )
    # A plain string, never a dict: an auditor must not be able to read the
    # off-protocol marker as compliance.
    assert isinstance(_eval_v6_stamp(never_migrated), str)
    assert isinstance(_eval_v6_stamp(on_protocol), dict)


def test_eval_v6_frozen_certificate_writes_the_computed_stamp():
    """The guard is worthless if the JSON stamps the header literal anyway."""
    values = []
    for node in ast.walk(_tree(EVAL_V6_FROZEN)):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        if "records" not in keys or "scenario_protocol" not in keys:
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "scenario_protocol":
                values.append(value)
    assert len(values) == 1, (
        f"expected one certificate header carrying scenario_protocol, found "
        f"{len(values)}"
    )
    assert isinstance(values[0], ast.Name) and values[0].id == "scenario_protocol", (
        "the certificate stamps something other than the guarded "
        "scenario_protocol variable, so the off-protocol marker never reaches "
        f"the JSON (got {ast.dump(values[0])[:80]})"
    )


# ============================================================================
# 6. The guard that does not exist yet (reported, not patched)
# ============================================================================
def test_numpy_rng_refuses_to_draw_before_the_first_reset():
    """ScenarioRNG.numpy_rng must guard the pre-reset case like the torch path.

    ``ScenarioRNG._draw`` raises when an env has no episode yet
    (``tasks/_shared/scenario_rng.py:592-597``) because episode index -1 is not
    a scenario, it is "never reset".  ``numpy_rng``
    (``tasks/_shared/scenario_rng.py:381-394``) reads the same counter and
    hands -1 straight to the key, so a numpy-layout family that ever drew
    before its first ``reset_idx`` would silently run a real, reproducible,
    WRONG episode -- and every episode-index-based pairing after it would be
    off by one with nothing to show for it.

    This test is expected to FAIL until scenario_rng.py grows the guard; the
    source change is not made here on purpose.
    """
    scenario = ScenarioRNG(num_envs=2, device=DEVICE, eval_seed=7)
    assert int(scenario.episode_indices()[0]) == -1

    torch_path_raised = False
    try:
        scenario.uniform("layout", [0])
    except RuntimeError:
        torch_path_raised = True
    assert torch_path_raised, "the torch draw path lost its pre-reset guard"

    try:
        scenario.numpy_rng("layout", 0)
    except RuntimeError:
        return
    raise AssertionError(
        "ScenarioRNG.numpy_rng returned a generator for an env that has never "
        "been reset: it keyed on episode_index=-1 instead of raising the way "
        "_draw does. NEEDED SOURCE CHANGE in tasks/_shared/scenario_rng.py, "
        "numpy_rng, after the episode_index is resolved:\n"
        "    if episode_index < 0:\n"
        "        raise RuntimeError(\n"
        "            f'env {env_index} has no episode yet: call reset_idx(...) '\n"
        "            f'before drawing group {group!r}'\n"
        "        )"
    )


# ============================================================================
# hygiene
# ============================================================================
def test_sources_are_pure_ascii_without_control_characters():
    for path in SOURCES_UNDER_TEST:
        text = path.read_text(encoding="utf-8")
        bad = {
            ord(ch) for ch in text
            if ord(ch) < 32 and ch not in ("\n", "\r")
        }
        assert not bad, f"{path.name} carries control characters {sorted(bad)}"


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
            except Exception as exc:  # noqa: BLE001 - report, do not abort
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
