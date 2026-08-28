# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests that EXECUTE the hazard_nav / harbor_mission reset call sites.

WHY THIS FILE EXISTS ALONGSIDE test_scenario_wiring_hazard.py
-------------------------------------------------------------
``tasks/_shared/test_scenario_wiring_hazard.py`` proves the shared helpers in
``tasks/_shared/scenario_draws_hazard.py`` behave, by TRANSCRIBING the two
envs' reset paths into ``_reset_hazard_nav`` / ``_reset_harbor_mission``.  A
transcription cannot notice that the env stopped matching it.  Its one
source-level test, ``test_envs_expose_what_the_evaluators_read``, closes part
of that hole by string-matching a handful of lines, but string matching only
sees the SPELLING of a call site, never its effect.  Measured on this tree, by
reverting one call site at a time in a scratch copy and re-running every suite,
these all left the whole tree green:

  * ``layout_rng = episode_layout_rng(None, self._layout_rng, ...)``
    -- the spelling survives, the protocol object never reaches the helper, and
    the layout is back on the single shared generator;
  * ``episode_layout_rng(self._scenario, self._layout_rng, 0)``
    -- every env keyed on env 0, so all 64 envs run the same layout;
  * ``goal_angle = float(self._layout_rng.uniform(0.0, 2.0 * math.pi))``
    -- the rotation off the protocol entirely;
  * ``choice_indices = self._layout_rng.integers(n, size=num_resets)``
    -- an actuator knob off the protocol;
  * the harbor spawn jitter back on the shared generator, either spelling;
  * ``for completed in env_ids.tolist():`` in the recording block -- the
    per-episode scenario latched for envs that did NOT finish, so a running
    episode's digest is written over a finished one's.

So every test below EXECUTES shipped code.  The env modules import ``isaaclab``
at module scope and cannot be imported here, so each test AST-lifts the exact
call expression or statement span it is about out of the real source file and
evaluates it against a fabricated ``self`` -- the technique
``tasks/_shared/test_scenario_regression.py`` uses for the round-2 families.
Editing the env source changes what these tests run.

WHAT IS CLAIMED, AND HOW STRONGLY
---------------------------------
EXECUTION strength (the shipped expression is evaluated and its VALUE checked):
  * the layout generator, the layout rotation, the five actuator knobs and the
    harbor spawn jitter are keyed by (eval seed, env, episode) -- proved by
    running two reset schedules, as two controllers ending episodes at
    different times would, and requiring identical draws in the shared
    ``(env, episode)`` slots, with the pre-fix shared generator as the negative
    control that must NOT agree;
  * the per-episode scenario digest is latched for the envs that finished and
    for no others -- proved by running the shipped recording block over a
    mixed ``_episode_finished`` mask.

SOURCE strength (an AST property, not an executed one):
  * ``ScenarioRNG.reset_idx`` is called inside ``_reset_idx`` and its statement
    precedes every protocol draw in the method.  Ordering inside one shipped
    method cannot be established by evaluating the pieces separately, so this
    half is an AST line-number comparison.  The accompanying execution test
    shows the reseed is load-bearing: without it the shipped draw raises.

NOTHING ABOUT THE TASK DEFINITION IS TOUCHED HERE.  No range, distribution,
reward, observation, termination rule or physics constant is asserted or
changed; every claim is about which STREAM a draw comes off and which env a
record belongs to.

Run: python tasks/_shared/test_hazard_env_wiring.py
"""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path

import numpy as np
import torch

try:
    from .scenario_draws import GROUP_SPAWN, make_scenario_rng
    from .scenario_draws_hazard import (
        ACTUATOR_GROUPS,
        GROUP_DRAG_SCALE,
        GROUP_LAYOUT,
        GROUP_MASS_SCALE,
        GROUP_MOTOR_TAU_S,
        GROUP_PAYLOAD_MASS_KG,
        GROUP_ROTATION,
        GROUP_THRUST_CAP_SCALE,
        GROUP_THRUST_IMBALANCE,
        TWO_PI,
        actuator_choice_indices,
        episode_layout_rng,
        layout_rotation_angle,
        spawn_jitter_offsets,
    )
    from .scenario_rng import ScenarioRNG
except ImportError:  # direct execution
    from scenario_draws import GROUP_SPAWN, make_scenario_rng
    from scenario_draws_hazard import (
        ACTUATOR_GROUPS,
        GROUP_DRAG_SCALE,
        GROUP_LAYOUT,
        GROUP_MASS_SCALE,
        GROUP_MOTOR_TAU_S,
        GROUP_PAYLOAD_MASS_KG,
        GROUP_ROTATION,
        GROUP_THRUST_CAP_SCALE,
        GROUP_THRUST_IMBALANCE,
        TWO_PI,
        actuator_choice_indices,
        episode_layout_rng,
        layout_rotation_angle,
        spawn_jitter_offsets,
    )
    from scenario_rng import ScenarioRNG


DEVICE = torch.device("cpu")
NUM_ENVS = 4

_SHARED = Path(__file__).resolve().parent
_TASKS = _SHARED.parent

HAZARD_NAV_ENV = _TASKS / "hazard_nav" / "hazard_nav_env.py"
HARBOR_MISSION_ENV = _TASKS / "harbor_mission" / "harbor_mission_env.py"

HAZARD_NAV = (HAZARD_NAV_ENV, "HazardNavEnv")
HARBOR_MISSION = (HARBOR_MISSION_ENV, "HarborMissionEnv")

SOURCES_UNDER_TEST = (
    HAZARD_NAV_ENV,
    HARBOR_MISSION_ENV,
    Path(__file__).resolve(),
)

# Two reset schedules = two termination patterns.  Each inner list is the set
# of envs that finished together on one step.  Controller A ends them in pairs,
# controller B one at a time and then all four at once, so by the time each
# reaches env i's k-th episode they have consumed a different number of draws
# from any SHARED generator.  Same pair test_scenario_wiring_hazard.py uses.
SCHEDULE_A = [[0, 1, 2, 3], [1, 3], [0, 2], [1, 3], [0, 2]]
SCHEDULE_B = [[0, 1, 2, 3], [0], [2], [1], [3], [0, 1, 2, 3]]

# How many numbers a rejection sampler would burn off the generator it is
# handed.  The value is arbitrary; what matters is that the generator is
# CONSUMED, or a shared advancing stream would never be seen to drift.
BURN = 7
JITTER_M = 2.0
CHOICE_TABLE = (-0.15, -0.05, 0.0, 0.05, 0.15)


# --------------------------------------------------------------- AST lifting --
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{path.name}: no {class_name}.{method_name}")


def _assigned_names(node: ast.stmt) -> set[str]:
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


def _draw_calls(family, function_name: str, expected: int = 1) -> list[ast.Call]:
    """The shipped ``function_name(...)`` calls inside that env's _reset_idx."""
    path, class_name = family
    calls = [
        node
        for node in ast.walk(_method(path, class_name, "_reset_idx"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]
    assert len(calls) == expected, (
        f"{path.name}: {class_name}._reset_idx calls {function_name}(...) "
        f"{len(calls)} times, expected {expected}. A reset path that no longer "
        "goes through the shared protocol helper is drawing its scenario from "
        "somewhere this test cannot see -- which is exactly how the layout "
        "ended up on one shared generator advanced per reset"
    )
    return calls


def _evaluate(call: ast.Call, path: Path, namespace: dict):
    """Evaluate one SHIPPED call expression, with its real line number."""
    expression = ast.Expression(body=copy.deepcopy(call))
    ast.fix_missing_locations(expression)
    return eval(compile(expression, str(path), "eval"), namespace)  # noqa: S307


class _Fake:
    """A stand-in env: only the attributes the lifted expressions read."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _AutoFake:
    """A stand-in env whose UNSPECIFIED tensor buffers default to zeros.

    The recording block of these two families touches two dozen per-episode
    metric buffers that have nothing to do with the scenario latch under test.
    Defaulting them keeps this file from breaking when an unrelated metric is
    added, while every buffer the CLAIM is about is supplied explicitly.
    """

    def __init__(self, num_envs: int, **fields):
        object.__setattr__(self, "_auto_num_envs", num_envs)
        self.__dict__.update(fields)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = torch.zeros(self._auto_num_envs, device=DEVICE)
        self.__dict__[name] = value
        return value


class _AutoCfg:
    """A cfg whose ``*_choices`` tables all resolve to one fixed tuple."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, name):
        if name.endswith("_choices"):
            return CHOICE_TABLE
        raise AttributeError(name)


def _namespace(base, env_indices, row: int = 0, **extra) -> dict:
    """Every name the shipped reset-path expressions read."""
    namespace = {
        "self": base,
        "reset_env_indices": [int(index) for index in env_indices],
        "env_ids": torch.as_tensor(
            [int(index) for index in env_indices], dtype=torch.long, device=DEVICE
        ),
        "num_resets": len(env_indices),
        "row": row,
        "np": np,
        "math": math,
        "torch": torch,
        "episode_layout_rng": episode_layout_rng,
        "layout_rotation_angle": layout_rotation_angle,
        "actuator_choice_indices": actuator_choice_indices,
        "spawn_jitter_offsets": spawn_jitter_offsets,
        "GROUP_LAYOUT": GROUP_LAYOUT,
        "GROUP_ROTATION": GROUP_ROTATION,
        "GROUP_SPAWN": GROUP_SPAWN,
        "GROUP_THRUST_IMBALANCE": GROUP_THRUST_IMBALANCE,
        "GROUP_MASS_SCALE": GROUP_MASS_SCALE,
        "GROUP_DRAG_SCALE": GROUP_DRAG_SCALE,
        "GROUP_THRUST_CAP_SCALE": GROUP_THRUST_CAP_SCALE,
        "GROUP_MOTOR_TAU_S": GROUP_MOTOR_TAU_S,
        "GROUP_PAYLOAD_MASS_KG": GROUP_PAYLOAD_MASS_KG,
    }
    namespace.update(extra)
    return namespace


def _env(scenario, fallback):
    return _Fake(_scenario=scenario, _layout_rng=fallback, cfg=_AutoCfg())


def _fallback(seed=0) -> np.random.Generator:
    """The env's own generator: np.random.default_rng(layout_seed)."""
    return np.random.default_rng(seed)


# --------------------------------------------------------- schedule machinery --
def _run_schedule(schedule, seed, draw, protocol=True):
    """Drive one reset schedule; key every drawn value by (env, episode).

    ``draw(base, env_ids, row)`` evaluates a shipped call site once for the row
    that belongs to ``env_ids[row]`` and returns something comparable.  With
    ``protocol=False`` the env carries no ``ScenarioRNG``, i.e. the historical
    single advancing generator -- the negative control.
    """
    scenario = make_scenario_rng(_Fake(seed=seed), NUM_ENVS, DEVICE) if protocol else None
    fallback = _fallback(seed)
    base = _env(scenario, fallback)
    seen = {}
    counters = {index: -1 for index in range(NUM_ENVS)}
    for env_ids in schedule:
        if scenario is not None:
            scenario.reset_idx(torch.as_tensor(env_ids, dtype=torch.long))
        for row, env_index in enumerate(env_ids):
            value = draw(base, env_ids, row)
            counters[env_index] += 1
            seen[(env_index, counters[env_index])] = value
    return seen


def _same(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def _assert_controller_independent(label, draw, seed=2026):
    """The property: env i's k-th draw does not depend on WHEN it happened.

    Runs the shipped call site under two termination patterns on the protocol
    (must agree on every shared slot) and again on the pre-fix shared generator
    (must NOT), so the test cannot pass vacuously on broken code.
    """
    seen_a = _run_schedule(SCHEDULE_A, seed, draw)
    seen_b = _run_schedule(SCHEDULE_B, seed, draw)
    shared = sorted(set(seen_a) & set(seen_b))
    assert len(shared) >= 8, f"{label}: only {len(shared)} shared episodes"
    for key in shared:
        assert _same(seen_a[key], seen_b[key]), (
            f"{label}: env {key[0]} episode {key[1]} drew a different value "
            "under two controllers at the same eval seed. The shipped call "
            "site is not on the keyed protocol stream -- it is riding a "
            "generator whose position depends on how many episodes have "
            "ended so far, which is what made three certified controllers "
            "agree on only ~55% of their (env, episode) layouts"
        )

    pre_fix_a = _run_schedule(SCHEDULE_A, seed, draw, protocol=False)
    pre_fix_b = _run_schedule(SCHEDULE_B, seed, draw, protocol=False)
    disagreements = [
        key
        for key in sorted(set(pre_fix_a) & set(pre_fix_b))
        if not _same(pre_fix_a[key], pre_fix_b[key])
    ]
    assert disagreements, (
        f"{label}: the pre-fix shared-generator path agreed on every shared "
        "slot too, so this test cannot tell a keyed stream from a sequential "
        "one and its result above means nothing"
    )


def _assert_envs_differ(label, draw, seed=99):
    """Teeth: a call site that hands every env the same thing is not keyed."""
    seen = _run_schedule([[0, 1, 2, 3]], seed, draw)
    values = [seen[(index, 0)] for index in range(NUM_ENVS)]
    for index in range(1, NUM_ENVS):
        assert not _same(values[0], values[index]), (
            f"{label}: env 0 and env {index} drew the identical value on their "
            "first episode. The call site is keying every env the same way -- "
            "e.g. a literal env index instead of reset_env_indices[row] -- so "
            "the whole batch is running one scenario"
        )


def _assert_seeds_differ(label, draw):
    """Teeth: two evaluation seeds must be two different exam papers."""
    first = _run_schedule([[0, 1, 2, 3]], 123, draw)
    other = _run_schedule([[0, 1, 2, 3]], 42, draw)
    assert any(
        not _same(first[key], other[key]) for key in sorted(first)
    ), f"{label}: eval seeds 123 and 42 produced identical draws"


# ============================================================================
# 1. the layout generator must come off the protocol, in BOTH families
# ============================================================================
def _layout_draw(family):
    call = _draw_calls(family, "episode_layout_rng")[0]
    path = family[0]

    def draw(base, env_ids, row):
        generator = _evaluate(call, path, _namespace(base, env_ids, row=row))
        assert isinstance(generator, np.random.Generator), (
            f"{path.name}: the layout call site returned "
            f"{type(generator).__name__}, not a numpy Generator"
        )
        # Burn what a rejection sampler would burn, so a SHARED generator is
        # visibly advanced and the negative control has something to drift on.
        return generator.random(BURN)

    return draw


def _assert_layout_is_the_protocol_stream(family):
    """Byte-exact: the generator handed to the sampler IS numpy_rng(layout)."""
    path, class_name = family
    call = _draw_calls(family, "episode_layout_rng")[0]
    scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=515)
    indices = list(range(NUM_ENVS))
    scenario.reset_idx(torch.as_tensor(indices, dtype=torch.long))
    fallback = _fallback(515)
    base = _env(scenario, fallback)
    for row, env_index in enumerate(indices):
        generator = _evaluate(call, path, _namespace(base, indices, row=row))
        assert generator is not fallback, (
            f"{path.name}: {class_name} handed the sampler its OWN "
            "self._layout_rng -- the single generator advanced once per reset "
            "that the protocol replaced"
        )
        expected = scenario.numpy_rng(GROUP_LAYOUT, env_index)
        assert generator.bit_generator.state == expected.bit_generator.state, (
            f"{path.name}: the layout generator for row {row} is not "
            f"numpy_rng({GROUP_LAYOUT!r}, {env_index}). Either the protocol "
            "object never reaches the helper, or the row is keyed on the "
            "wrong env index, or the group was changed"
        )

    # The unseeded env keeps its historical single generator, byte for byte.
    unseeded = _env(None, fallback)
    assert _evaluate(call, path, _namespace(unseeded, indices, row=0)) is fallback, (
        f"{path.name}: an env built with cfg.seed None no longer falls back to "
        "self._layout_rng, so the pre-fix behaviour is not preserved"
    )


def test_hazard_nav_layout_generator_comes_off_the_protocol():
    _assert_layout_is_the_protocol_stream(HAZARD_NAV)
    draw = _layout_draw(HAZARD_NAV)
    _assert_controller_independent("hazard_nav layout", draw)
    _assert_envs_differ("hazard_nav layout", draw)
    _assert_seeds_differ("hazard_nav layout", draw)


def test_harbor_mission_layout_generator_comes_off_the_protocol():
    _assert_layout_is_the_protocol_stream(HARBOR_MISSION)
    draw = _layout_draw(HARBOR_MISSION)
    _assert_controller_independent("harbor_mission layout", draw)
    _assert_envs_differ("harbor_mission layout", draw)
    _assert_seeds_differ("harbor_mission layout", draw)


# ============================================================================
# 2. hazard_nav's global rotation must be its own protocol stream
# ============================================================================
def test_hazard_nav_layout_rotation_comes_off_the_protocol():
    """The rotation is applied to the ACCEPTED layout and stamped, so a
    rotation off a shared generator desynchronises the certificate exactly the
    way the layout itself did."""
    path, _class_name = HAZARD_NAV
    call = _draw_calls(HAZARD_NAV, "layout_rotation_angle")[0]

    def draw(base, env_ids, row):
        angle = _evaluate(call, path, _namespace(base, env_ids, row=row))
        assert isinstance(angle, float), type(angle)
        assert 0.0 <= angle < TWO_PI, angle
        return angle

    scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=808)
    indices = list(range(NUM_ENVS))
    scenario.reset_idx(torch.as_tensor(indices, dtype=torch.long))
    base = _env(scenario, _fallback(808))
    for row, env_index in enumerate(indices):
        angle = _evaluate(call, path, _namespace(base, indices, row=row))
        assert angle == layout_rotation_angle(scenario, None, env_index), (
            f"hazard_nav row {row}: the rotation angle is not the "
            f"{GROUP_ROTATION!r} stream of env {env_index}"
        )

    _assert_controller_independent("hazard_nav rotation", draw)
    _assert_envs_differ("hazard_nav rotation", draw)
    _assert_seeds_differ("hazard_nav rotation", draw)


# ============================================================================
# 3. every actuator knob must own a stream
# ============================================================================
def test_hazard_nav_actuator_knobs_come_off_the_protocol():
    """Five knobs, five groups, five call sites.

    scripts/eval_imbalance.py sweeps these one at a time; a knob back on the
    shared generator both breaks the pairing and re-draws its neighbours.
    """
    path, _class_name = HAZARD_NAV
    calls = _draw_calls(
        HAZARD_NAV, "actuator_choice_indices", expected=len(ACTUATOR_GROUPS)
    )
    groups = []
    for call in calls:
        names = [
            argument.id
            for argument in call.args
            if isinstance(argument, ast.Name) and argument.id.startswith("GROUP_")
        ]
        assert len(names) == 1, ast.dump(call)[:200]
        groups.append(_namespace(None, [])[names[0]])
    assert sorted(groups) == sorted(ACTUATOR_GROUPS), (
        "hazard_nav no longer draws one index per knob off that knob's own "
        f"group: {sorted(groups)} vs {sorted(ACTUATOR_GROUPS)}"
    )

    for call, group in zip(calls, groups):
        def draw(base, env_ids, row, call=call):
            chosen = _evaluate(call, path, _namespace(base, env_ids, row=row))
            return np.asarray(chosen)[row]

        _assert_controller_independent(f"hazard_nav {group}", draw)

    # And the drawn indices are the protocol's, env by env.
    scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=404)
    indices = list(range(NUM_ENVS))
    scenario.reset_idx(torch.as_tensor(indices, dtype=torch.long))
    base = _env(scenario, _fallback(404))
    chosen = _evaluate(calls[0], path, _namespace(base, indices, row=0))
    expected = actuator_choice_indices(
        scenario, None, groups[0], indices, len(CHOICE_TABLE)
    )
    assert np.array_equal(np.asarray(chosen), expected), (
        f"hazard_nav {groups[0]} indices are not the protocol's: {chosen} vs "
        f"{expected}"
    )


# ============================================================================
# 4. harbor_mission's staged spawn jitter must be its own protocol stream
# ============================================================================
def test_harbor_mission_spawn_jitter_comes_off_the_protocol():
    path, _class_name = HARBOR_MISSION
    call = _draw_calls(HARBOR_MISSION, "spawn_jitter_offsets")[0]

    def draw(base, env_ids, row):
        offsets = _evaluate(
            call, path, _namespace(base, env_ids, row=row, jitter_m=JITTER_M)
        )
        assert offsets.shape == (len(env_ids), 2), offsets.shape
        return offsets[row].copy()

    scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=77)
    indices = list(range(NUM_ENVS))
    scenario.reset_idx(torch.as_tensor(indices, dtype=torch.long))
    base = _env(scenario, _fallback(77))
    offsets = _evaluate(
        call, path, _namespace(base, indices, row=0, jitter_m=JITTER_M)
    )
    expected = spawn_jitter_offsets(scenario, None, indices, JITTER_M)
    assert np.array_equal(offsets, expected), (
        f"harbor_mission spawn jitter is not the {GROUP_SPAWN!r} stream of "
        "each env: the staged-spawn start pose is back on a generator whose "
        "position depends on the termination pattern"
    )

    _assert_controller_independent("harbor_mission spawn jitter", draw)
    _assert_envs_differ("harbor_mission spawn jitter", draw)
    _assert_seeds_differ("harbor_mission spawn jitter", draw)


# ============================================================================
# 5. reset_idx must be called, and must precede every draw
# ============================================================================
_PROTOCOL_DRAWS = (
    "episode_layout_rng",
    "layout_rotation_angle",
    "actuator_choice_indices",
    "spawn_jitter_offsets",
)


def _reseed_statement(family) -> ast.stmt:
    """The shipped ``self._scenario.reset_idx(env_ids)`` statement (with guard)."""
    path, class_name = family
    reset = _method(path, class_name, "_reset_idx")
    hits = [
        statement
        for statement in reset.body
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reset_idx"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_scenario"
            for node in ast.walk(statement)
        )
    ]
    assert len(hits) == 1, (
        f"{path.name}: {class_name}._reset_idx calls "
        f"self._scenario.reset_idx(...) {len(hits)} times, expected exactly "
        "one. Without it the per-env episode counter never advances, so every "
        "episode of a run is drawn on the key of episode 0 -- or on -1, which "
        "ScenarioRNG refuses outright"
    )
    return hits[0]


def test_reset_idx_is_called_before_every_protocol_draw():
    """SOURCE-level (AST line numbers), because order lives inside one method."""
    for family in (HAZARD_NAV, HARBOR_MISSION):
        path, class_name = family
        reseed = _reseed_statement(family)
        draws = [
            call
            for name in _PROTOCOL_DRAWS
            for call in ast.walk(_method(path, class_name, "_reset_idx"))
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == name
        ]
        assert draws, f"{path.name}: no protocol draw left in _reset_idx"
        earliest = min(call.lineno for call in draws)
        assert reseed.lineno < earliest, (
            f"{path.name}: self._scenario.reset_idx is at line "
            f"{reseed.lineno}, after the first protocol draw at line "
            f"{earliest}. Every draw of the new episode would then be keyed on "
            "the PREVIOUS episode index, so episode k of the certificate "
            "carries episode k-1's scenario"
        )


def test_the_reseed_is_what_makes_the_shipped_draw_legal():
    """EXECUTION-level companion: run the shipped reseed, then the shipped draw.

    ScenarioRNG refuses to hand out a stream for an env that has never been
    reset (episode index -1 is 'never reset', not a scenario), so the draw
    below only succeeds because the reseed ran -- which is what makes the
    ordering assertion above load-bearing rather than cosmetic.
    """
    for family in (HAZARD_NAV, HARBOR_MISSION):
        path, _class_name = family
        reseed = _reseed_statement(family)
        call = _draw_calls(family, "episode_layout_rng")[0]
        indices = list(range(NUM_ENVS))

        scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=31)
        base = _env(scenario, _fallback(31))
        assert int(scenario.episode_indices()[0]) == -1
        refused = False
        try:
            _evaluate(call, path, _namespace(base, indices, row=0))
        except RuntimeError:
            refused = True
        assert refused, (
            f"{path.name}: the shipped layout draw returned a stream for an "
            "env that has never been reset, so a missing reset_idx would be "
            "silent instead of loud"
        )

        module = ast.Module(body=[copy.deepcopy(reseed)], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = _namespace(base, indices, row=0)
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
        assert [int(value) for value in scenario.episode_indices()] == [0] * NUM_ENVS, (
            f"{path.name}: the shipped reseed statement did not advance the "
            f"per-env episode counters (got {scenario.episode_indices()})"
        )
        generator = _evaluate(call, path, _namespace(base, indices, row=0))
        assert generator.bit_generator.state == scenario.numpy_rng(
            GROUP_LAYOUT, 0
        ).bit_generator.state

        # A second reset must move to episode 1, not stand still.
        exec(compile(module, str(path), "exec"), _namespace(base, indices, row=0))
        assert [int(value) for value in scenario.episode_indices()] == [1] * NUM_ENVS


# ============================================================================
# 6. the per-episode scenario is latched for the FINISHED envs and no others
# ============================================================================
def _recording_block(family) -> list[ast.stmt]:
    """``completed_ids = ...`` plus the ``if len(completed_ids) > 0:`` block."""
    path, class_name = family
    body = _method(path, class_name, "_reset_idx").body
    starts = [
        index
        for index, statement in enumerate(body)
        if "completed_ids" in _assigned_names(statement)
    ]
    assert len(starts) == 1, (
        f"{path.name}: _reset_idx assigns completed_ids {len(starts)} times"
    )
    index = starts[0]
    guard = body[index + 1]
    assert isinstance(guard, ast.If), (
        f"{path.name}: the statement after completed_ids is no longer the "
        "`if len(completed_ids) > 0:` recording block"
    )
    return [body[index], guard]


def _run_recording_block(family, finished, live_tags, old_tags):
    """Execute the SHIPPED recording block over a mixed finished mask."""
    path, _class_name = family
    base = _AutoFake(
        NUM_ENVS,
        device=DEVICE,
        control_step_s=0.1,
        cfg=_Fake(curriculum_frozen=True),
        curriculum=_Fake(),
        extras={},
        _episode_finished=torch.as_tensor(finished, dtype=torch.bool, device=DEVICE),
        _first_success_time_s=torch.full((NUM_ENVS,), torch.nan, device=DEVICE),
        # The boolean buffers, which the float default cannot stand in for:
        # `float_buffer[ids] = bool_buffer` is a dtype error, not a cast.
        _success=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        episode_success=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        _m1=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        _m2=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        _m3=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        m1=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        m2=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        m3=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        _scenario_params=[{"tag": tag} for tag in live_tags],
        _scenario_hashes=[{"scenario": tag} for tag in live_tags],
        episode_scenario=[{"tag": tag} for tag in old_tags],
        episode_scenario_hashes=[{"scenario": tag} for tag in old_tags],
    )
    module = ast.Module(body=_recording_block(family), type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "self": base,
        "env_ids": torch.arange(NUM_ENVS, dtype=torch.long, device=DEVICE),
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return base


def _assert_latches_only_completed(family):
    path, _class_name = family
    finished = [True, False, True, False]
    live = ["live0", "live1", "live2", "live3"]
    old = ["old0", "old1", "old2", "old3"]
    base = _run_recording_block(family, finished, live, old)

    for env_index, ended in enumerate(finished):
        recorded = base.episode_scenario_hashes[env_index]["scenario"]
        if ended:
            assert recorded == live[env_index], (
                f"{path.name}: env {env_index} finished its episode but the "
                f"latched scenario digest is {recorded!r}, not the running "
                f"episode's {live[env_index]!r}. Every certificate row for "
                "that env would then name a different episode than the "
                "metrics beside it"
            )
        else:
            assert recorded == old[env_index], (
                f"{path.name}: env {env_index} did NOT finish its episode, yet "
                f"its latched digest was overwritten with {recorded!r}. A "
                "running episode's scenario has been written over a finished "
                "one's, so the certificate names an episode that had not "
                "happened yet"
            )
        assert base.episode_scenario[env_index]["tag"] == (
            live[env_index] if ended else old[env_index]
        ), f"{path.name}: episode_scenario disagrees with episode_scenario_hashes"


def test_hazard_nav_latches_the_scenario_of_completed_episodes_only():
    _assert_latches_only_completed(HAZARD_NAV)


def test_harbor_mission_latches_the_scenario_of_completed_episodes_only():
    _assert_latches_only_completed(HARBOR_MISSION)


def test_the_latch_test_can_tell_completed_from_running():
    """Teeth for the two tests above: an all-False mask must record nothing."""
    for family in (HAZARD_NAV, HARBOR_MISSION):
        base = _run_recording_block(
            family,
            [False] * NUM_ENVS,
            ["live"] * NUM_ENVS,
            ["old"] * NUM_ENVS,
        )
        assert all(
            entry["scenario"] == "old" for entry in base.episode_scenario_hashes
        ), (
            f"{family[0].name}: the recording block latched a scenario for an "
            "env whose episode has not finished"
        )


# ============================================================================
# hygiene
# ============================================================================
def test_sources_are_pure_ascii_without_control_characters():
    for path in SOURCES_UNDER_TEST:
        text = path.read_text(encoding="utf-8")
        bad = {ord(ch) for ch in text if ord(ch) < 32 and ch not in ("\n", "\r")}
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
