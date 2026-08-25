# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests that EXECUTE the path_hazard / path_following / boat reset paths.

WHY THIS FILE EXISTS ALONGSIDE test_scenario_regression.py
----------------------------------------------------------
``tasks/_shared/test_scenario_regression.py`` proves the shared helpers in
``tasks/_shared/scenario_draws.py`` behave, and for station_keeping_boat it
lifts a shipped statement span.  For path_hazard and path_following it
TRANSCRIBES the reset path instead -- ``_path_hazard_episodes`` calls
``spawn_heading(scenario, ids, DEVICE)`` itself rather than running the env's
own line.  A transcription cannot notice that the env stopped matching it.

Measured on this tree, by reverting one call site at a time in a scratch copy
outside the repository and re-running every suite, these all left the whole
tree green:

  * ``spawn_headings = spawn_heading(self._scenario, env_ids, self.device)``
    in ``tasks/path_hazard/path_hazard_env.py`` reverted to
    ``torch.rand(num_resets, device=self.device) * 2.0 * torch.pi``
    -- the bow heading back on the GLOBAL torch RNG, which skrl's ``Runner``
    reseeds to the agent-YAML constant after the env is built, so every
    evaluation seed draws the same spawn poses;
  * the identical revert in ``tasks/path_following/path_following_env.py``;
  * deleting ``self._scenario.reset_idx(env_ids)`` from either of those two
    reset paths -- every episode of a run then keyed on episode 0, or on -1,
    which is "this env has never been reset";
  * widening ``for completed in completed_ids.tolist():`` to
    ``env_ids.tolist():`` in path_hazard's recording block -- the per-episode
    scenario digest of a RUNNING episode written over a FINISHED one's, so the
    certificate row names an episode that never ran;
  * deleting the ``stamp_scenario(...)`` block from
    ``tasks/station_keeping_boat/station_keeping_boat_env.py`` -- the episode
    scenario never resolved at all, so every certificate row for that family
    carries an empty digest and ``scripts/check_scenario_independence.py`` has
    nothing to compare.

So every test below EXECUTES shipped code.  The env modules import ``isaaclab``
at module scope and cannot be imported here, so each test AST-lifts the exact
statement span it is about out of the real source file and runs it against a
fabricated ``self`` -- the technique
``tasks/_shared/test_hazard_env_wiring.py`` and
``tasks/_shared/test_scenario_regression.py`` use.  Editing the env source
changes what these tests run.

WHAT IS CLAIMED, AND HOW STRONGLY
---------------------------------
REAL TEETH (a lifted statement runs and the VALUE it produces is asserted):
  * the spawn heading of both route families IS the ``spawn_pose`` stream of
    (eval seed, env, episode), byte for byte -- checked against the protocol,
    against a global ``torch.manual_seed`` clobber with the pre-fix global-RNG
    line as the negative control, across two eval seeds, across envs, and
    across two reset schedules as two controllers would produce;
  * the reset path advances the episode counter BEFORE it draws -- proved by
    running the shipped span twice and requiring episodes 0 then 1, which a
    span with no reseed cannot do because ``ScenarioRNG`` refuses to draw for
    an env that has never been reset;
  * path_hazard's recording block latches the per-episode scenario for the envs
    that FINISHED and for no others -- proved over a mixed
    ``_episode_finished`` mask;
  * station_keeping_boat resolves and hashes its episode scenario from the
    draws it actually made -- proved by comparing the stamped digest against
    ``stamp_scenario`` applied to the values the shipped span produced.

GREP/AST-LEVEL (a source property, not an executed one):
  * ``self._scenario.reset_idx(...)`` appears in ``_reset_idx`` and its line
    precedes every protocol draw in that method.  Ordering inside one shipped
    method cannot be established by evaluating the pieces separately.  The
    execution tests above show the ordering is load-bearing rather than
    cosmetic.

NOTHING ABOUT THE TASK DEFINITION IS TOUCHED HERE.  No range, distribution,
reward, observation, termination rule or physics constant is asserted or
changed; every claim is about which STREAM a draw comes off, which episode key
it is on, and which env a record belongs to.  The range checks below assert the
SHIPPED bounds are UNCHANGED; they do not choose them.

Run: python tasks/_shared/test_scenario_env_wiring.py
"""

from __future__ import annotations

import ast
import functools
import math
from pathlib import Path

import numpy as np
import torch

try:
    from .scenario_draws import (
        GROUP_SPAWN,
        TWO_PI,
        make_scenario_rng,
        path_following_route,
        spawn_heading,
        stamp_scenario,
        station_keeping_spawn,
    )
    from .scenario_rng import ScenarioRNG
except ImportError:  # direct execution
    from scenario_draws import (
        GROUP_SPAWN,
        TWO_PI,
        make_scenario_rng,
        path_following_route,
        spawn_heading,
        stamp_scenario,
        station_keeping_spawn,
    )
    from scenario_rng import ScenarioRNG


DEVICE = torch.device("cpu")
NUM_ENVS = 4

_SHARED = Path(__file__).resolve().parent
_TASKS = _SHARED.parent

PATH_HAZARD_ENV = _TASKS / "path_hazard" / "path_hazard_env.py"
PATH_HAZARD_CFG = _TASKS / "path_hazard" / "path_hazard_env_cfg.py"
PATH_HAZARD_GEOMETRY = _TASKS / "path_hazard" / "path_hazard_geometry.py"
PATH_FOLLOWING_ENV = _TASKS / "path_following" / "path_following_env.py"
PATH_FOLLOWING_CFG = _TASKS / "path_following" / "path_following_env_cfg.py"
BOAT_ENV = _TASKS / "station_keeping_boat" / "station_keeping_boat_env.py"
BOAT_CFG = _TASKS / "station_keeping_boat" / "station_keeping_boat_env_cfg.py"

PATH_HAZARD = (PATH_HAZARD_ENV, "PathHazardEnv")
PATH_FOLLOWING = (PATH_FOLLOWING_ENV, "PathFollowingEnv")
BOAT = (BOAT_ENV, "StationKeepingBoatEnv")

SOURCES_UNDER_TEST = (
    PATH_HAZARD_ENV,
    PATH_FOLLOWING_ENV,
    BOAT_ENV,
    Path(__file__).resolve(),
)

# Two reset schedules = two termination patterns.  Each inner list is the set
# of envs that finished together on one step.  Same pair the sibling wiring
# suites use, so a stream whose position depends on how many episodes have
# ended is visibly different between them.
SCHEDULE_A = [[0, 1, 2, 3], [1, 3], [0, 2], [1, 3], [0, 2]]
SCHEDULE_B = [[0, 1, 2, 3], [0], [2], [1], [3], [0, 1, 2, 3]]


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


def _calls_method(node: ast.AST, name: str) -> bool:
    """Does this subtree call the method ``.name(...)`` on anything?"""
    return any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == name
        for inner in ast.walk(node)
    )


def _calls(node: ast.AST, name: str) -> bool:
    """Does this subtree call the bare function ``name(...)``?"""
    return any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == name
        for inner in ast.walk(node)
    )


def _exec_statements(statements, path: Path, namespace: dict) -> dict:
    """Execute shipped statements verbatim, with their real line numbers."""
    module = ast.Module(body=list(statements), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def _lift_method(path: Path, class_name: str, method_name: str, namespace: dict):
    """Compile one shipped method into a plain function of (self, ...)."""
    _exec_statements([_method(path, class_name, method_name)], path, namespace)
    return namespace[method_name]


def _module_constant(path: Path, name: str):
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


def _reset_span(family, ends) -> list[ast.stmt]:
    """The shipped ``_reset_idx`` statements from the episode advance onwards.

    The span deliberately STARTS at the episode-advance guard rather than at
    the first draw: the ORDER (reset_idx, then draw) is what keys the scenario
    by episode index, so the test has to execute the shipped ordering rather
    than supply its own reset.  ``num_resets`` is the fallback start so the span
    is still findable on a build where the guard has been deleted -- such a
    build then fails on the PROPERTY under test (its first draw is refused,
    because ScenarioRNG will not hand out a stream for an env that has never
    been reset) rather than on statement selection.
    """
    path, class_name = family
    body = _method(path, class_name, "_reset_idx").body
    starts = [
        index
        for index, statement in enumerate(body)
        if _calls_method(statement, "reset_idx")
        or "num_resets" in _assigned_names(statement)
    ]
    assert starts, (
        f"{path.name}: {class_name}._reset_idx has neither a scenario reset "
        "nor a num_resets assignment, so the reset span cannot be located"
    )
    first = starts[0]
    if ends is None:
        return body[first:]
    last = [index for index, statement in enumerate(body) if ends(statement)]
    assert last, f"{path.name}: no end statement in {class_name}._reset_idx"
    assert last[-1] >= first, (
        f"{path.name}: the reset span end precedes its start"
    )
    return body[first:last[-1] + 1]


def _assigns(name: str):
    return lambda statement: name in _assigned_names(statement)


# ------------------------------------------------------------------ fixtures --
class _Fake:
    """A stand-in env: only the attributes the lifted statements read."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _AutoFake:
    """A stand-in env whose UNSPECIFIED tensor buffers default to zeros."""

    def __init__(self, auto_num_envs: int, **fields):
        object.__setattr__(self, "_auto_num_envs", auto_num_envs)
        self.__dict__.update(fields)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = torch.zeros(self._auto_num_envs, device=DEVICE)
        self.__dict__[name] = value
        return value


class _MathUtilsStub:
    """Enough of ``isaaclab.utils.math`` for the reset span to run.

    Only the boat span reaches a quaternion, and only to fill ``root_state``,
    which nothing in this file asserts on.  The stub is shaped, not correct,
    and is documented as such so no reader mistakes it for a claim about pose.
    """

    @staticmethod
    def quat_from_angle_axis(angle, axis):
        rows = int(angle.shape[0])
        return torch.zeros((rows, 1, 4), device=DEVICE)

    @staticmethod
    def quat_apply(quat, vec):
        return torch.zeros_like(vec)


def _cfg_seed(seed):
    return _Fake(seed=seed)


def _skrl_clobber():
    """Reproduce skrl Runner.__init__ -> set_seed(agent yaml seed).

    The agent YAMLs pin a constant and the Runner is built AFTER the env, so
    the global torch RNG sits at the SAME state for every evaluation seed by
    the time the first episode is drawn.  Without this line a two-seed
    comparison passes even on the broken code, because the global stream simply
    keeps advancing between the two calls.
    """
    torch.manual_seed(42)


def _pre_fix_spawn_heading(num_resets: int) -> torch.Tensor:
    """The pre-fix line, verbatim from ``scenario_draws.spawn_heading``'s docstring.

    Used only as a NEGATIVE CONTROL, so that "the shipped draw ignored the
    global seed" cannot pass on a build where nothing reacts to it.
    """
    return torch.rand(num_resets, device=DEVICE) * 2.0 * torch.pi


# ============================================================================
# path_hazard reset path
# ============================================================================
PH_GROUP_LAYOUT = _module_constant(PATH_HAZARD_ENV, "GROUP_LAYOUT")
PH_NUM_WAYPOINTS = _cfg_default(PATH_HAZARD_CFG, "num_waypoints")
PH_OBSTACLE_COUNT = _cfg_default(PATH_HAZARD_CFG, "obstacle_count")
PH_MAX_ATTEMPTS = _cfg_default(PATH_HAZARD_CFG, "layout_max_attempts")
PH_BLOCKERS_OVERRIDE = _cfg_default(PATH_HAZARD_CFG, "on_line_blockers_override")

_PH_SPAN = _reset_span(PATH_HAZARD, _assigns("spawn_headings"))


def _load_geometry():
    """Import the Isaac-free layout sampler by FILE.

    ``tasks/path_hazard/__init__.py`` imports gymnasium and registers gym ids,
    so the package path cannot be used here.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_env_wiring_ph_geometry", PATH_HAZARD_GEOMETRY
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_env_wiring_ph_geometry"] = module
    spec.loader.exec_module(module)
    return module


_PH_SAMPLE_LAYOUT = _load_geometry().sample_layout


def _path_hazard_base(scenario, num_envs=NUM_ENVS):
    base = _AutoFake(
        num_envs,
        _scenario=scenario,
        _layout_rng=np.random.default_rng(0),
        device=DEVICE,
        num_envs=num_envs,
        cfg=_Fake(
            num_waypoints=PH_NUM_WAYPOINTS,
            obstacle_count=PH_OBSTACLE_COUNT,
            layout_max_attempts=PH_MAX_ATTEMPTS,
            on_line_blockers_override=PH_BLOCKERS_OVERRIDE,
        ),
        scene=_Fake(env_origins=torch.zeros((num_envs, 3), device=DEVICE)),
        waypoints=torch.zeros((num_envs, PH_NUM_WAYPOINTS, 2), device=DEVICE),
        segment_lengths=torch.zeros((num_envs, PH_NUM_WAYPOINTS), device=DEVICE),
        route_length=torch.zeros(num_envs, device=DEVICE),
        obstacle_centers=torch.zeros((num_envs, PH_OBSTACLE_COUNT, 2), device=DEVICE),
        obstacle_radii=torch.zeros((num_envs, PH_OBSTACLE_COUNT), device=DEVICE),
        obstacle_active=torch.zeros(
            (num_envs, PH_OBSTACLE_COUNT), dtype=torch.bool, device=DEVICE
        ),
        obstacle_count=torch.zeros(num_envs, dtype=torch.long, device=DEVICE),
    )
    # The SHIPPED _layout_generator, bound to the fabricated self, so the
    # numpy stream the rejection sampler runs on is the env's own choice.
    generator = _lift_method(
        PATH_HAZARD_ENV,
        "PathHazardEnv",
        "_layout_generator",
        {"np": np, "torch": torch, "GROUP_LAYOUT": PH_GROUP_LAYOUT},
    )
    base._layout_generator = functools.partial(generator, base)
    return base


def _run_path_hazard(base, env_ids) -> dict:
    """Execute the SHIPPED path_hazard reset span once."""
    namespace = {
        "np": np,
        "torch": torch,
        "math": math,
        "sample_layout": _PH_SAMPLE_LAYOUT,
        "spawn_heading": spawn_heading,
        "self": base,
        "env_ids": torch.as_tensor(
            [int(index) for index in env_ids], dtype=torch.long, device=DEVICE
        ),
    }
    return _exec_statements(_PH_SPAN, PATH_HAZARD_ENV, namespace)


# ============================================================================
# path_following reset path
# ============================================================================
PF_NUM_WAYPOINTS = _cfg_default(PATH_FOLLOWING_CFG, "num_waypoints")
PF_SEGMENT_MIN = _cfg_default(PATH_FOLLOWING_CFG, "segment_length_min")
PF_SEGMENT_MAX = _cfg_default(PATH_FOLLOWING_CFG, "segment_length_max")
PF_HEADING_CHANGE_MAX_DEG = _cfg_default(
    PATH_FOLLOWING_CFG, "heading_change_max_deg"
)

_PF_SPAN = _reset_span(PATH_FOLLOWING, _assigns("spawn_headings"))


def _path_following_base(scenario, num_envs=NUM_ENVS):
    return _AutoFake(
        num_envs,
        _scenario=scenario,
        device=DEVICE,
        num_envs=num_envs,
        cfg=_Fake(
            num_waypoints=PF_NUM_WAYPOINTS,
            segment_length_min=PF_SEGMENT_MIN,
            segment_length_max=PF_SEGMENT_MAX,
            heading_change_max_deg=PF_HEADING_CHANGE_MAX_DEG,
        ),
        segment_lengths=torch.zeros((num_envs, PF_NUM_WAYPOINTS), device=DEVICE),
        waypoints=torch.zeros((num_envs, PF_NUM_WAYPOINTS, 2), device=DEVICE),
        route_length=torch.zeros(num_envs, device=DEVICE),
    )


def _run_path_following(base, env_ids) -> dict:
    namespace = {
        "np": np,
        "torch": torch,
        "math": math,
        "path_following_route": path_following_route,
        "spawn_heading": spawn_heading,
        "self": base,
        "env_ids": torch.as_tensor(
            [int(index) for index in env_ids], dtype=torch.long, device=DEVICE
        ),
    }
    return _exec_statements(_PF_SPAN, PATH_FOLLOWING_ENV, namespace)


# ============================================================================
# station_keeping_boat reset path (runs to the END of _reset_idx)
# ============================================================================
BOAT_MIN_SPAWN = _cfg_default(BOAT_CFG, "min_spawn_distance")
BOAT_MAX_SPAWN = _cfg_default(BOAT_CFG, "max_spawn_distance")

# ends=None: the span runs to the last statement of _reset_idx, so DELETING the
# stamp_scenario block does not move the span -- it makes the span stop
# resolving the episode scenario, which is a VALUE the tests below read.
_BOAT_SPAN = _reset_span(BOAT, None)


def _boat_base(scenario, num_envs=NUM_ENVS):
    return _AutoFake(
        num_envs,
        _scenario=scenario,
        device=DEVICE,
        num_envs=num_envs,
        cfg=_Fake(
            min_spawn_distance=BOAT_MIN_SPAWN,
            max_spawn_distance=BOAT_MAX_SPAWN,
        ),
        scene=_Fake(env_origins=torch.zeros((num_envs, 3), device=DEVICE)),
        robot=_Fake(
            data=_Fake(default_root_state=torch.zeros((num_envs, 13), device=DEVICE)),
            write_root_state_to_sim=lambda *_args, **_kwargs: None,
        ),
        up_dir=torch.tensor([0.0, 0.0, 1.0], device=DEVICE),
        _body_yaw_from_bow_offset=0.0,
        _hold_steps=torch.zeros(num_envs, dtype=torch.long, device=DEVICE),
        _max_hold_steps=torch.zeros(num_envs, dtype=torch.long, device=DEVICE),
        _first_success_time_s=torch.full((num_envs,), torch.nan, device=DEVICE),
        hold_timer=torch.zeros(num_envs, device=DEVICE),
        path_length=torch.zeros(num_envs, device=DEVICE),
        _success=torch.zeros(num_envs, dtype=torch.bool, device=DEVICE),
        _episode_finished=torch.zeros(num_envs, dtype=torch.bool, device=DEVICE),
        _previous_xy=torch.zeros((num_envs, 2), device=DEVICE),
        _scenario_params=[{} for _ in range(num_envs)],
        _scenario_hashes=[{} for _ in range(num_envs)],
    )


def _run_boat(base, env_ids) -> dict:
    namespace = {
        "np": np,
        "torch": torch,
        "math": math,
        "math_utils": _MathUtilsStub,
        "station_keeping_spawn": station_keeping_spawn,
        "stamp_scenario": stamp_scenario,
        "self": base,
        "env_ids": torch.as_tensor(
            [int(index) for index in env_ids], dtype=torch.long, device=DEVICE
        ),
    }
    return _exec_statements(_BOAT_SPAN, BOAT_ENV, namespace)


_RESET_PATHS = {
    "path_hazard": (PATH_HAZARD, _path_hazard_base, _run_path_hazard),
    "path_following": (PATH_FOLLOWING, _path_following_base, _run_path_following),
    "station_keeping_boat": (BOAT, _boat_base, _run_boat),
}

_HEADING_FAMILIES = ("path_hazard", "path_following")


def _draw_headings(label, scenario, env_ids):
    """Run one family's SHIPPED reset span and return its spawn headings."""
    _family, make_base, run = _RESET_PATHS[label]
    base = make_base(scenario)
    namespace = run(base, env_ids)
    headings = namespace["spawn_headings"]
    assert torch.is_tensor(headings), (
        f"{label}: the shipped reset path assigned spawn_headings as "
        f"{type(headings).__name__}, not a tensor"
    )
    assert tuple(headings.shape) == (len(env_ids),), (
        f"{label}: spawn_headings has shape {tuple(headings.shape)} for "
        f"{len(env_ids)} reset envs"
    )
    return headings


# ============================================================================
# 1. the spawn heading must come off the keyed protocol, in BOTH route families
# ============================================================================
def _assert_heading_is_the_protocol_stream(label):
    """REAL TEETH: byte-exact against ScenarioRNG's spawn_pose stream."""
    indices = list(range(NUM_ENVS))
    seed = 515

    scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=seed)
    drawn = _draw_headings(label, scenario, indices)

    reference = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=seed)
    reference.reset_idx(torch.as_tensor(indices, dtype=torch.long))
    expected = spawn_heading(reference, indices, DEVICE)
    assert torch.equal(drawn, expected), (
        f"{label}: the shipped reset path did not draw its spawn heading from "
        f"the {GROUP_SPAWN!r} stream of (eval seed, env, episode). Got "
        f"{drawn.tolist()}, the protocol says {expected.tolist()}. Either the "
        "line is back on the global torch RNG -- which skrl's Runner reseeds "
        "to the agent YAML constant after the env is built, so every eval seed "
        "would draw the same spawn poses -- or the row is keyed on the wrong "
        "env, or the group was changed"
    )


def _assert_heading_ignores_the_global_rng(label):
    """REAL TEETH: a global reseed must not move the draw; the pre-fix line does."""
    indices = list(range(NUM_ENVS))

    torch.manual_seed(0)
    protected = _draw_headings(
        label, ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=555), indices
    )
    torch.manual_seed(12345)
    again = _draw_headings(
        label, ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=555), indices
    )
    assert torch.equal(protected, again), (
        f"{label}: the spawn heading moved when the GLOBAL torch RNG was "
        "reseeded, so it is not on the protocol. skrl's Runner reseeds that "
        "generator to a constant from the agent YAML after the env is built, "
        "which is how two evaluation seeds ended up drawing one exam paper"
    )

    torch.manual_seed(0)
    pre_fix_a = _pre_fix_spawn_heading(NUM_ENVS)
    torch.manual_seed(12345)
    pre_fix_b = _pre_fix_spawn_heading(NUM_ENVS)
    assert not torch.equal(pre_fix_a, pre_fix_b), (
        f"{label}: the pre-fix global-RNG line did not react to the global "
        "seed either, so the assertion above cannot tell fixed from broken"
    )

    # Two evaluation seeds must be two different exam papers.
    first = _draw_headings(
        label, ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=123), indices
    )
    _skrl_clobber()
    other = _draw_headings(
        label, ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=42), indices
    )
    assert not torch.equal(first, other), (
        f"{label}: eval seeds 123 and 42 produced identical spawn headings"
    )

    # And two envs of one batch must not share a heading.
    assert len({round(float(value), 12) for value in first}) == NUM_ENVS, (
        f"{label}: two envs drew the identical spawn heading on their first "
        f"episode ({first.tolist()}); the call site is keying every env the "
        "same way, so the whole batch is running one spawn pose"
    )


def _assert_heading_is_controller_independent(label, seed=2026):
    """REAL TEETH: env i's k-th heading does not depend on WHEN it happened."""
    seen = {}
    for tag, schedule in (("a", SCHEDULE_A), ("b", SCHEDULE_B)):
        scenario = make_scenario_rng(_cfg_seed(seed), NUM_ENVS, DEVICE)
        counters = {index: -1 for index in range(NUM_ENVS)}
        for env_ids in schedule:
            headings = _draw_headings(label, scenario, env_ids)
            for row, env_index in enumerate(env_ids):
                counters[env_index] += 1
                seen[(tag, env_index, counters[env_index])] = float(headings[row])
    keys_a = {key[1:] for key in seen if key[0] == "a"}
    keys_b = {key[1:] for key in seen if key[0] == "b"}
    shared = sorted(keys_a & keys_b)
    assert len(shared) >= 8, f"{label}: only {len(shared)} shared episodes"
    for env_index, episode in shared:
        assert seen[("a", env_index, episode)] == seen[("b", env_index, episode)], (
            f"{label}: env {env_index} episode {episode} drew a different "
            "spawn heading under two controllers at the same eval seed. The "
            "shipped call site is riding a generator whose position depends on "
            "how many episodes have ended so far"
        )


def _assert_heading_range_is_unchanged(label):
    """The migration changed the SOURCE of the numbers, never the range."""
    wide = 512
    scenario = ScenarioRNG(num_envs=wide, device=DEVICE, eval_seed=3)
    _family, make_base, run = _RESET_PATHS[label]
    base = make_base(scenario, num_envs=wide)
    headings = run(base, list(range(wide)))["spawn_headings"]
    assert bool((headings >= 0.0).all() and (headings < TWO_PI).all()), (
        f"{label}: the spawn heading left [0, 2*pi): "
        f"[{float(headings.min())}, {float(headings.max())}]"
    )
    assert abs(float(headings.mean()) - math.pi) < 0.03 * TWO_PI, (
        f"{label}: spawn heading mean {float(headings.mean())} is not pi"
    )


def test_path_hazard_spawn_heading_comes_off_the_protocol():
    """REAL TEETH. Fix 3: path_hazard_env.py:1088, the bow heading stream."""
    _assert_heading_is_the_protocol_stream("path_hazard")
    _assert_heading_ignores_the_global_rng("path_hazard")
    _assert_heading_is_controller_independent("path_hazard")


def test_path_following_spawn_heading_comes_off_the_protocol():
    """REAL TEETH. Fix 4: path_following_env.py:710, the bow heading stream."""
    _assert_heading_is_the_protocol_stream("path_following")
    _assert_heading_ignores_the_global_rng("path_following")
    _assert_heading_is_controller_independent("path_following")


def test_spawn_heading_ranges_are_unchanged():
    """REAL TEETH. Only the stream changed: [0, 2*pi) in both families, still."""
    for label in _HEADING_FAMILIES:
        _assert_heading_range_is_unchanged(label)


# ============================================================================
# 2. the reset path must advance the episode BEFORE it draws
# ============================================================================
_PROTOCOL_DRAW_NAMES = (
    "spawn_heading",
    "path_following_route",
    "station_keeping_spawn",
    "station_keeping_current",
)


def _reseed_statements(family) -> list[ast.stmt]:
    path, class_name = family
    return [
        statement
        for statement in _method(path, class_name, "_reset_idx").body
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reset_idx"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_scenario"
            for node in ast.walk(statement)
        )
    ]


def test_reset_idx_is_called_before_every_protocol_draw():
    """GREP/AST-LEVEL. Order lives inside one shipped method, so it is a
    line-number comparison; the execution test below shows it is load-bearing."""
    for label, (family, _make, _run) in _RESET_PATHS.items():
        path, class_name = family
        hits = _reseed_statements(family)
        assert len(hits) == 1, (
            f"{path.name}: {class_name}._reset_idx calls "
            f"self._scenario.reset_idx(...) {len(hits)} times, expected "
            "exactly one. Without it the per-env episode counter never "
            "advances, so every episode of a run is drawn on the key of "
            "episode 0 -- or on -1, which ScenarioRNG refuses outright"
        )
        reset = _method(path, class_name, "_reset_idx")
        draws = [
            call.lineno
            for name in _PROTOCOL_DRAW_NAMES
            for call in ast.walk(reset)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == name
        ]
        # path_hazard's layout stream is reached through a method, not a bare
        # helper call, so it is located separately.
        draws += [
            call.lineno
            for call in ast.walk(reset)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_layout_generator"
        ]
        assert draws, f"{label}: no protocol draw left in _reset_idx"
        assert hits[0].lineno < min(draws), (
            f"{path.name}: self._scenario.reset_idx is at line "
            f"{hits[0].lineno}, after the first protocol draw at line "
            f"{min(draws)}. Every draw of the new episode would then be keyed "
            "on the PREVIOUS episode index, so episode k of the certificate "
            "carries episode k-1's scenario"
        )


def _assert_span_advances_the_episode(label):
    """REAL TEETH: run the shipped span twice; it must reach episodes 0 then 1.

    ``ScenarioRNG`` refuses to hand out a stream for an env that has never been
    reset (episode -1 is "never reset", not a scenario), so a span with no
    reseed cannot even complete the first call -- which is what makes the
    ordering assertion above load-bearing rather than cosmetic.
    """
    _family, make_base, run = _RESET_PATHS[label]
    indices = list(range(NUM_ENVS))
    scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=31)
    assert [int(value) for value in scenario.episode_indices()] == [-1] * NUM_ENVS

    base = make_base(scenario)
    first = run(base, indices)
    assert [int(value) for value in scenario.episode_indices()] == [0] * NUM_ENVS, (
        f"{label}: the shipped reset span did not advance the per-env episode "
        f"counters to 0 (got {scenario.episode_indices().tolist()}). Without "
        "self._scenario.reset_idx(env_ids) every episode of the run is drawn "
        "on one key and the certificate's per-episode digests are all equal"
    )
    second = run(base, indices)
    assert [int(value) for value in scenario.episode_indices()] == [1] * NUM_ENVS, (
        f"{label}: a second reset did not move the episode counter to 1 "
        f"(got {scenario.episode_indices().tolist()})"
    )
    return first, second


def test_path_hazard_reset_advances_the_episode_before_drawing():
    """REAL TEETH. Fix 6: the self._scenario.reset_idx(env_ids) in the reset."""
    first, second = _assert_span_advances_the_episode("path_hazard")
    assert not torch.equal(first["spawn_headings"], second["spawn_headings"]), (
        "path_hazard drew the identical spawn heading for episode 0 and "
        "episode 1 of the same envs: the episode key is not advancing"
    )
    assert not np.array_equal(
        first["waypoints_np"].copy(), second["waypoints_np"]
    ), "path_hazard drew the identical layout for two consecutive episodes"


def test_path_following_reset_advances_the_episode_before_drawing():
    """REAL TEETH. Fix 7: the self._scenario.reset_idx(env_ids) in the reset."""
    first, second = _assert_span_advances_the_episode("path_following")
    assert not torch.equal(first["spawn_headings"], second["spawn_headings"]), (
        "path_following drew the identical spawn heading for episode 0 and "
        "episode 1 of the same envs: the episode key is not advancing"
    )
    assert not torch.equal(first["first_headings"], second["first_headings"]), (
        "path_following drew the identical route for two consecutive episodes"
    )


def test_station_keeping_boat_reset_advances_the_episode_before_drawing():
    """REAL TEETH. The boat reset advances the episode before it draws."""
    first, second = _assert_span_advances_the_episode("station_keeping_boat")
    assert not torch.equal(first["distances"], second["distances"]), (
        "station_keeping_boat drew the identical spawn distance for episode 0 "
        "and episode 1 of the same envs"
    )


# ============================================================================
# 3. path_hazard latches the per-episode scenario for the FINISHED envs only
# ============================================================================
def _path_hazard_recording_block() -> list[ast.stmt]:
    body = _method(PATH_HAZARD_ENV, "PathHazardEnv", "_reset_idx").body
    starts = [
        index
        for index, statement in enumerate(body)
        if "completed_ids" in _assigned_names(statement)
    ]
    assert len(starts) == 1, (
        f"path_hazard_env.py: _reset_idx assigns completed_ids {len(starts)} "
        "times, expected exactly one"
    )
    index = starts[0]
    guard = body[index + 1]
    assert isinstance(guard, ast.If), (
        "path_hazard_env.py: the statement after completed_ids is no longer "
        "the `if len(completed_ids) > 0:` recording block"
    )
    return [body[index], guard]


_PH_RECORD = _path_hazard_recording_block()


def _run_path_hazard_recording(finished, live_tags, old_tags):
    """Execute the SHIPPED path_hazard recording block over a mixed mask."""
    base = _AutoFake(
        NUM_ENVS,
        device=DEVICE,
        control_step_s=0.1,
        cfg=_Fake(curriculum_frozen=True),
        extras={},
        _episode_finished=torch.as_tensor(finished, dtype=torch.bool, device=DEVICE),
        _first_success_time_s=torch.full((NUM_ENVS,), torch.nan, device=DEVICE),
        # Boolean buffers, which the float default cannot stand in for:
        # `float_buffer[ids] = bool_buffer` is a dtype error, not a cast.
        _success=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        episode_success=torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE),
        # Distinct per-env metrics, so "wrote the wrong env" is visible.
        path_length=torch.arange(1.0, NUM_ENVS + 1.0, device=DEVICE) * 10.0,
        episode_path_length=torch.zeros(NUM_ENVS, device=DEVICE),
        _scenario_params=[{"tag": tag} for tag in live_tags],
        _scenario_hashes=[{"scenario": tag} for tag in live_tags],
        episode_scenario=[{"tag": tag} for tag in old_tags],
        episode_scenario_hashes=[{"scenario": tag} for tag in old_tags],
    )
    _exec_statements(
        _PH_RECORD,
        PATH_HAZARD_ENV,
        {
            "torch": torch,
            "self": base,
            "env_ids": torch.arange(NUM_ENVS, dtype=torch.long, device=DEVICE),
        },
    )
    return base


def test_path_hazard_latches_the_scenario_of_completed_episodes_only():
    """REAL TEETH. Fix 5: a mixed mask; running envs must keep their old row."""
    finished = [True, False, True, False]
    live = ["live0", "live1", "live2", "live3"]
    old = ["old0", "old1", "old2", "old3"]
    base = _run_path_hazard_recording(finished, live, old)

    for env_index, ended in enumerate(finished):
        recorded = base.episode_scenario_hashes[env_index]["scenario"]
        if ended:
            assert recorded == live[env_index], (
                f"path_hazard_env.py: env {env_index} finished its episode but "
                f"the latched scenario digest is {recorded!r}, not the "
                f"episode's own {live[env_index]!r}"
            )
            assert float(base.episode_path_length[env_index]) == float(
                base.path_length[env_index]
            ), f"env {env_index} finished but its metrics were not latched"
        else:
            assert recorded == old[env_index], (
                f"path_hazard_env.py: env {env_index} did NOT finish its "
                f"episode, yet its latched digest was overwritten with "
                f"{recorded!r}. A RUNNING episode's scenario has been written "
                "over a FINISHED one's, so scripts/eval_v6_frozen.py stamps "
                "the certificate row with an episode that had not happened yet "
                "-- exactly the confusion the digests exist to rule out"
            )
            assert float(base.episode_path_length[env_index]) == 0.0, (
                f"env {env_index} did not finish, yet its per-episode metrics "
                "were latched"
            )
        assert base.episode_scenario[env_index]["tag"] == (
            live[env_index] if ended else old[env_index]
        ), "episode_scenario disagrees with episode_scenario_hashes"


def test_the_path_hazard_latch_test_can_tell_completed_from_running():
    """REAL TEETH, anti-vacuity. An all-False mask must record nothing."""
    base = _run_path_hazard_recording(
        [False] * NUM_ENVS, ["live"] * NUM_ENVS, ["old"] * NUM_ENVS
    )
    assert all(
        entry["scenario"] == "old" for entry in base.episode_scenario_hashes
    ), (
        "path_hazard_env.py: the recording block latched a scenario for an env "
        "whose episode has not finished"
    )


# ============================================================================
# 4. station_keeping_boat must resolve and hash the episode scenario it drew
# ============================================================================
def test_station_keeping_boat_stamps_the_scenario_it_actually_drew():
    """REAL TEETH. Fix 8: the stamped digest is stamp_scenario over the draws.

    Deleting the ``stamp_scenario(...)`` block leaves ``_scenario_hashes``
    empty, so every certificate row for this family carries no digest at all
    and ``scripts/check_scenario_independence.py`` has nothing to compare
    between two evaluation seeds -- a pair of runs could share one exam paper
    and no auditor could see it.
    """
    indices = list(range(NUM_ENVS))
    scenario = ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=808)
    base = _boat_base(scenario)
    namespace = _run_boat(base, indices)

    for env_index in indices:
        hashes = base._scenario_hashes[env_index]
        assert hashes, (
            "station_keeping_boat_env.py: the shipped reset path finished "
            f"without resolving any scenario for env {env_index} "
            "(_scenario_hashes is empty). The stamp_scenario(...) block is "
            "gone or no longer writes _scenario_hashes, so the certificate "
            "row for every episode of this family carries no digest and two "
            "evaluation seeds cannot be shown to be two different exam papers"
        )
        assert set(hashes) == {GROUP_SPAWN, "scenario"}, (
            f"station_keeping_boat env {env_index} stamped {sorted(hashes)}, "
            f"expected the {GROUP_SPAWN!r} group digest plus the whole-scenario "
            "digest scripts/check_scenario_independence.py reads"
        )

    expected = stamp_scenario(
        torch.as_tensor(indices, dtype=torch.long, device=DEVICE),
        {
            GROUP_SPAWN: {
                "distance_m": namespace["distances"],
                "angle_rad": namespace["spawn_angles"],
                "heading_rad": namespace["headings"],
            }
        },
    )
    for env_index, resolved, hashes in expected:
        assert base._scenario_hashes[env_index] == hashes, (
            f"station_keeping_boat env {env_index} stamped a digest that is "
            "not the hash of the three draws the shipped span actually made: "
            f"{base._scenario_hashes[env_index]} vs {hashes}. The certificate "
            "would identify an episode by numbers the boat never spawned with"
        )
        assert base._scenario_params[env_index] == resolved, (
            f"station_keeping_boat env {env_index} resolved "
            f"{base._scenario_params[env_index]}, not the drawn "
            f"{resolved}"
        )

    # A constant stamp would satisfy everything above, so require that the
    # digest actually separates envs and evaluation seeds.
    digests = [base._scenario_hashes[index]["scenario"] for index in indices]
    assert len(set(digests)) == NUM_ENVS, (
        f"station_keeping_boat stamped one digest for several envs: {digests}"
    )
    other_base = _boat_base(ScenarioRNG(num_envs=NUM_ENVS, device=DEVICE, eval_seed=42))
    _run_boat(other_base, indices)
    other = [other_base._scenario_hashes[index]["scenario"] for index in indices]
    assert not set(digests) & set(other), (
        "station_keeping_boat stamped the same scenario digest under eval "
        f"seeds 808 and 42: {digests} vs {other}"
    )


def test_station_keeping_boat_spawn_ranges_are_unchanged():
    """REAL TEETH. The stamped draws stay inside the family's own shipped ring."""
    wide = 512
    scenario = ScenarioRNG(num_envs=wide, device=DEVICE, eval_seed=7)
    base = _boat_base(scenario, num_envs=wide)
    namespace = _run_boat(base, list(range(wide)))
    distances = namespace["distances"]
    assert bool((distances >= BOAT_MIN_SPAWN).all()), float(distances.min())
    assert bool((distances < BOAT_MAX_SPAWN).all()), float(distances.max())
    for name in ("spawn_angles", "headings"):
        value = namespace[name]
        assert bool((value >= 0.0).all() and (value < TWO_PI).all()), (
            f"{name} left [0, 2*pi)"
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
