# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the PER-EPISODE RECORDING LATCH of path_hazard.

WHAT THIS FILE OWNS
-------------------
One line, ``tasks/path_hazard/path_hazard_env.py``::

    self._episode_finished.copy_(time_out | terminated)

It used to read ``copy_(time_out)``.  ``_episode_finished`` is the mask
``_reset_idx`` uses to decide which envs finished an episode and therefore
which envs get their per-episode statistics and their per-episode SCENARIO
recorded.  On the v2 gym id -- ``terminate_on_outcome = True``, defined on
``PathHazardV2EnvCfg`` in ``tasks/path_hazard/path_hazard_env_cfg.py`` and
registered as ``Isaac-USV-PathHazard-Direct-v2`` -- an episode that ends on its
OUTCOME (the last gate, or a contact before it) has ``time_out`` False, so with
the old line it never entered the recording block and every latch kept the
value of whatever episode that env last TIMED OUT on:

  * a success terminates by construction under this cfg, so a successful
    episode could never latch its own success;
  * a contact terminates too, so a negative ``episode_min_clearance`` could
    never be latched either;
  * ``episode_scenario_hashes[env]``, latched in the same block, named a
    DIFFERENT episode than the certificate row it was written into
    (``scripts/eval_v6_frozen.py`` reads it for the episode that just ended),
    which is exactly the confusion the digests exist to rule out.

The damage is in the repo.  ``curves/g4080/curve_ph2_s42.json`` -- task
``Isaac-USV-PathHazard-Direct-v2``, eval seed 42 -- holds 94 rungs and 6016
episodes with ``sr`` exactly 0.0 on EVERY rung, zero episodes with a negative
``min_clearance_m``, and 1430 records that repeat an exact
``(env, path_length_m, min_clearance_m)`` triple inside a single rung.  That
file is untracked evaluation output, so nothing here reads it; the fingerprint
is reproduced from the shipped statements instead.

WHY THE TESTS LOOK LIKE THIS
----------------------------
Nothing in the test tree read ``_get_dones`` or ``_episode_finished``, and the
env module imports ``isaaclab`` at module scope so the class cannot be imported
on a machine without Isaac Sim.  So, exactly as
``tasks/_shared/test_scenario_regression.py`` does, every test below AST-lifts
the SHIPPED method or statement span out of the real source file and executes
it against a fabricated ``self``.  Editing the env source changes what these
tests run, so a reverted fix fails here instead of passing quietly.

The "old" latch is not retyped either: it is the shipped ``_get_dones`` with
the single argument of the ``self._episode_finished.copy_(...)`` call replaced
by the bare name ``time_out``, which is the pre-fix line.  A build that reverts
the fix makes the two variants identical, and
``test_the_two_latches_disagree_under_terminate_on_outcome`` says so.

NOTHING ABOUT THE TASK DEFINITION IS TOUCHED HERE.  No range, distribution,
reward, observation, termination rule or physics constant is asserted or
changed; the only claim is about which finished episodes reach the recording
block.

Run: python tasks/_shared/test_episode_latch.py
"""

from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


DEVICE = torch.device("cpu")

_SHARED = Path(__file__).resolve().parent
_TASKS = _SHARED.parent

PATH_HAZARD_ENV = _TASKS / "path_hazard" / "path_hazard_env.py"
PATH_HAZARD_CFG = _TASKS / "path_hazard" / "path_hazard_env_cfg.py"
PATH_HAZARD_GEOMETRY = _TASKS / "path_hazard" / "path_hazard_geometry.py"
PATH_HAZARD_INIT = _TASKS / "path_hazard" / "__init__.py"

SOURCES_UNDER_TEST = (PATH_HAZARD_ENV, Path(__file__).resolve())


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


def _writes_self_attribute(node: ast.AST, name: str) -> bool:
    """Does this subtree assign to ``self.<name>[...]`` or ``self.<name>``?"""
    for inner in ast.walk(node):
        if not isinstance(inner, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            continue
        targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Attribute) and sub.attr == name:
                    return True
    return False


def _exec_statements(statements, path: Path, namespace: dict) -> dict:
    """Execute shipped statements verbatim, with their real line numbers."""
    module = ast.Module(body=list(statements), type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def _cfg_default(path: Path, field: str):
    """A dataclass cfg default (``field: type = <literal>``) from the source."""
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


def _cfg_values(path: Path, field: str) -> set:
    """Every literal a cfg field is given anywhere in the file."""
    return {
        node.value.value
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == field
        and isinstance(node.value, ast.Constant)
    }


def _load_module(path: Path, name: str):
    """Import an Isaac-free module by FILE, bypassing its package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GEOMETRY = _load_module(PATH_HAZARD_GEOMETRY, "_episode_latch_ph_geometry")
analytic_min_clearance = _GEOMETRY.analytic_min_clearance


# ---------------------------------------------------- the statements under test --
_GET_DONES = _method(PATH_HAZARD_ENV, "PathHazardEnv", "_get_dones")

# The helper methods _get_dones calls. Lifted too, so the geometry the latch
# reacts to is the shipped geometry and not a restatement of it.
_HELPER_METHODS = (
    "_com_xy",
    "_waypoint_world",
    "_current_waypoint_world",
    "_active_segment_relative",
    "_clearance",
)


def _latch_calls(function: ast.FunctionDef) -> list[ast.Call]:
    """Every ``self._episode_finished.copy_(...)`` call in a function."""
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy_"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_episode_finished"
    ]


def _old_get_dones() -> ast.FunctionDef:
    """The shipped ``_get_dones`` with the PRE-FIX latch line.

    Derived from the source rather than retyped: the single argument of the
    ``self._episode_finished.copy_(...)`` call is replaced by the bare name
    ``time_out``, which is what the line read before the fix.
    """
    function = copy.deepcopy(_GET_DONES)
    calls = _latch_calls(function)
    assert len(calls) == 1, (
        f"path_hazard_env.py: _get_dones has {len(calls)} "
        "self._episode_finished.copy_(...) calls, expected exactly one"
    )
    calls[0].args = [ast.Name(id="time_out", ctx=ast.Load())]
    ast.fix_missing_locations(function)
    return function


def _record_block() -> list[ast.stmt]:
    """The per-episode recording span of the shipped ``_reset_idx``.

    ``completed_ids = env_ids[self._episode_finished[env_ids]]`` and the
    ``if len(completed_ids) > 0:`` block that follows it -- the block whose
    entry condition is the thing the latch decides.
    """
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
    assert _writes_self_attribute(guard, "episode_scenario_hashes"), (
        "path_hazard_env.py: the recording block no longer latches "
        "episode_scenario_hashes, so the certificate's per-episode digest is "
        "written somewhere this test does not see"
    )
    return [body[index], guard]


_RECORD_BLOCK = _record_block()


# ------------------------------------------------------------------- fixtures --
NUM_WAYPOINTS = _cfg_default(PATH_HAZARD_CFG, "num_waypoints")
GOAL_RADIUS_M = _cfg_default(PATH_HAZARD_CFG, "goal_radius")
HALF_BEAM_M = _cfg_default(PATH_HAZARD_CFG, "half_beam_m")

# Test fixtures, not shipped ranges: one obstacle per env, placed so that the
# gate chain below clears it and the contact script below does not.
OBSTACLE_LOCAL_M = (12.0, 3.0)
OBSTACLE_RADIUS_M = 1.0
CONTACT_LOCAL_M = (12.0, 2.0)
MAX_EPISODE_LENGTH = 6
CONTROL_STEP_S = 0.1
SEGMENT_M = 5.0


class _Fake:
    """A stand-in env: only the attributes the lifted statements read."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _PathHazard:
    """A fabricated PathHazardEnv driven through the SHIPPED statements.

    ``step`` runs the shipped ``_get_dones`` (or its pre-fix twin) and ``reset``
    runs the shipped recording span of ``_reset_idx``.  Everything else -- the
    layout, the boat position, the episode counter -- is supplied by the test,
    because the point under test is which finished episodes reach the recording
    block, not how a layout is drawn.
    """

    def __init__(self, num_envs=2, terminate_on_outcome=True, latch="shipped"):
        self.num_envs = num_envs
        origins = torch.zeros((num_envs, 3), device=DEVICE)
        origins[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 100.0
        self.base = _Fake(
            num_envs=num_envs,
            device=DEVICE,
            cfg=_Fake(
                num_waypoints=NUM_WAYPOINTS,
                half_beam_m=HALF_BEAM_M,
                terminate_on_outcome=terminate_on_outcome,
            ),
            goal_radius=float(GOAL_RADIUS_M),
            control_step_s=CONTROL_STEP_S,
            max_episode_length=MAX_EPISODE_LENGTH,
            scene=_Fake(env_origins=origins),
            robot=_Fake(
                data=_Fake(root_com_pos_w=torch.zeros((num_envs, 3), device=DEVICE))
            ),
            waypoints=torch.zeros((num_envs, NUM_WAYPOINTS, 2), device=DEVICE),
            obstacle_centers=torch.zeros((num_envs, 1, 2), device=DEVICE),
            obstacle_radii=torch.full((num_envs, 1), OBSTACLE_RADIUS_M, device=DEVICE),
            obstacle_active=torch.ones((num_envs, 1), dtype=torch.bool, device=DEVICE),
            obstacle_count=torch.ones(num_envs, dtype=torch.long, device=DEVICE),
            episode_length_buf=torch.zeros(num_envs, dtype=torch.long, device=DEVICE),
            gates_passed=torch.zeros(num_envs, dtype=torch.long, device=DEVICE),
            path_length=torch.zeros(num_envs, device=DEVICE),
            xte_rms=torch.zeros(num_envs, device=DEVICE),
            route_length=torch.zeros(num_envs, device=DEVICE),
            _previous_xy=torch.zeros((num_envs, 2), device=DEVICE),
            _xte_squared_sum=torch.zeros(num_envs, device=DEVICE),
            _xte_sample_count=torch.zeros(num_envs, dtype=torch.long, device=DEVICE),
            _min_clearance=torch.full((num_envs,), torch.inf, device=DEVICE),
            _success=torch.zeros(num_envs, dtype=torch.bool, device=DEVICE),
            _last_gate_reached=torch.zeros(num_envs, dtype=torch.bool, device=DEVICE),
            _contact_before_last_gate=torch.zeros(
                num_envs, dtype=torch.bool, device=DEVICE
            ),
            _gates_passed_this_step=torch.zeros(
                num_envs, dtype=torch.bool, device=DEVICE
            ),
            _clean_last_gate_this_step=torch.zeros(
                num_envs, dtype=torch.bool, device=DEVICE
            ),
            _first_success_time_s=torch.full((num_envs,), torch.nan, device=DEVICE),
            _pre_transition_target_distance=torch.zeros(num_envs, device=DEVICE),
            _episode_finished=torch.zeros(num_envs, dtype=torch.bool, device=DEVICE),
            _fee_steps=torch.zeros(num_envs, device=DEVICE),
            _ep_prox_cost=torch.zeros(num_envs, device=DEVICE),
            _ep_contact_entries=torch.zeros(num_envs, device=DEVICE),
            episode_success=torch.zeros(num_envs, dtype=torch.bool, device=DEVICE),
            time_to_success=torch.full((num_envs,), torch.nan, device=DEVICE),
            episode_gates_passed=torch.zeros(num_envs, dtype=torch.long, device=DEVICE),
            episode_path_length=torch.zeros(num_envs, device=DEVICE),
            episode_min_clearance=torch.full((num_envs,), torch.nan, device=DEVICE),
            episode_xte_rms=torch.zeros(num_envs, device=DEVICE),
            episode_route_length=torch.zeros(num_envs, device=DEVICE),
            _scenario_params=[{} for _ in range(num_envs)],
            _scenario_hashes=[{} for _ in range(num_envs)],
            episode_scenario=[{} for _ in range(num_envs)],
            episode_scenario_hashes=[{} for _ in range(num_envs)],
            extras={},
        )

        namespace = {
            "torch": torch,
            "np": np,
            "analytic_min_clearance": analytic_min_clearance,
        }
        for name in _HELPER_METHODS:
            _exec_statements(
                [_method(PATH_HAZARD_ENV, "PathHazardEnv", name)],
                PATH_HAZARD_ENV,
                namespace,
            )
            setattr(self.base, name, functools.partial(namespace[name], self.base))

        function = _GET_DONES if latch == "shipped" else _old_get_dones()
        _exec_statements([function], PATH_HAZARD_ENV, namespace)
        self._dones = namespace["_get_dones"]

    # -- driving ------------------------------------------------------------
    def world(self, local_xy) -> torch.Tensor:
        """Env-local XY per env -> world XY."""
        local = torch.as_tensor(local_xy, dtype=torch.float32, device=DEVICE)
        return local + self.base.scene.env_origins[:, :2]

    def step(self, world_xy):
        self.base.robot.data.root_com_pos_w[:, :2] = world_xy
        self.base.episode_length_buf += 1
        return self._dones(self.base)

    def reset(self, env_ids, tag: str, scale: float = 1.0):
        """Run the SHIPPED recording span, then start a new episode."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=DEVICE)
        _exec_statements(
            _RECORD_BLOCK,
            PATH_HAZARD_ENV,
            {"torch": torch, "self": self.base, "env_ids": ids},
        )

        base = self.base
        origins_xy = base.scene.env_origins[ids, :2]
        waypoints = torch.zeros((NUM_WAYPOINTS, 2), device=DEVICE)
        waypoints[:, 0] = (
            torch.arange(1, NUM_WAYPOINTS + 1, dtype=torch.float32) * SEGMENT_M * scale
        )
        base.waypoints[ids] = waypoints
        base.route_length[ids] = float(NUM_WAYPOINTS) * SEGMENT_M * scale
        base.obstacle_centers[ids] = origins_xy.unsqueeze(1) + torch.tensor(
            [list(OBSTACLE_LOCAL_M)], device=DEVICE
        )
        base.robot.data.root_com_pos_w[ids, :2] = origins_xy
        base.episode_length_buf[ids] = 0
        base.gates_passed[ids] = 0
        base.path_length[ids] = 0.0
        base.xte_rms[ids] = 0.0
        base._xte_squared_sum[ids] = 0.0
        base._xte_sample_count[ids] = 0
        base._success[ids] = False
        base._last_gate_reached[ids] = False
        base._gates_passed_this_step[ids] = False
        base._first_success_time_s[ids] = torch.nan
        base._episode_finished[ids] = False
        base._previous_xy[ids] = origins_xy
        initial_clearance = analytic_min_clearance(
            origins_xy,
            base.obstacle_centers[ids],
            base.obstacle_radii[ids],
            half_beam_m=HALF_BEAM_M,
            active_mask=base.obstacle_active[ids],
        )
        base._min_clearance[ids] = initial_clearance
        base._contact_before_last_gate[ids] = initial_clearance < 0.0
        for env_index in ids.tolist():
            base._scenario_params[env_index] = {"tag": tag}
            base._scenario_hashes[env_index] = {"layout": tag, "scenario": tag}

    # -- observation --------------------------------------------------------
    def live(self, env_index: int) -> dict:
        """What the RUNNING episode of one env is worth right now."""
        base = self.base
        return {
            "success": bool(base._success[env_index]),
            "gates_passed": int(base.gates_passed[env_index]),
            "path_length": float(base.path_length[env_index]),
            "min_clearance": float(base._min_clearance[env_index]),
            "xte_rms": float(base.xte_rms[env_index]),
            "route_length": float(base.route_length[env_index]),
            "scenario_hashes": dict(base._scenario_hashes[env_index]),
        }

    def recorded(self, env_index: int) -> dict:
        """What the LATCHED per-episode record of one env says."""
        base = self.base
        return {
            "success": bool(base.episode_success[env_index]),
            "gates_passed": int(base.episode_gates_passed[env_index]),
            "path_length": float(base.episode_path_length[env_index]),
            "min_clearance": float(base.episode_min_clearance[env_index]),
            "xte_rms": float(base.episode_xte_rms[env_index]),
            "route_length": float(base.episode_route_length[env_index]),
            "scenario_hashes": dict(base.episode_scenario_hashes[env_index]),
        }

    def latch_state(self) -> dict:
        base = self.base
        names = (
            "episode_success",
            "time_to_success",
            "episode_gates_passed",
            "episode_path_length",
            "episode_min_clearance",
            "episode_xte_rms",
            "episode_route_length",
        )
        state = {name: getattr(base, name).clone() for name in names}
        state["episode_scenario_hashes"] = [
            dict(entry) for entry in base.episode_scenario_hashes
        ]
        return state


# ------------------------------------------------------------------- scripts --
def _gate_chain(scale: float) -> list[tuple[float, float]]:
    """One gate per step, then a pad step so a fixed-horizon cfg can time out."""
    chain = [
        (float(index + 1) * SEGMENT_M * scale, 0.0) for index in range(NUM_WAYPOINTS)
    ]
    return chain + [chain[-1]] * (MAX_EPISODE_LENGTH - 1 - len(chain))


def _crawl(scale: float) -> list[tuple[float, float]]:
    """Never within goal_radius of the first gate: the episode times out."""
    return [(0.5 * (index + 1), 0.0) for index in range(MAX_EPISODE_LENGTH - 1)]


def _into_the_obstacle(scale: float) -> list[tuple[float, float]]:
    return [CONTACT_LOCAL_M] * (MAX_EPISODE_LENGTH - 1)


def _run_episode(env: _PathHazard, script, scale: float):
    """Step until Isaac Lab would reset, i.e. until terminated | time_out."""
    for local in script(scale):
        terminated, time_out = env.step(env.world([local] * env.num_envs))
        if bool((terminated | time_out).any()):
            return terminated, time_out
    raise AssertionError("the script ended without terminating or timing out")


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.is_floating_point():
        nan_left, nan_right = torch.isnan(left), torch.isnan(right)
        if not torch.equal(nan_left, nan_right):
            return False
        return torch.equal(left[~nan_left], right[~nan_right])
    return torch.equal(left, right)


def _same_state(left: dict, right: dict) -> bool:
    for name, value in left.items():
        other = right[name]
        if isinstance(value, torch.Tensor):
            if not _same_tensor(value, other):
                return False
        elif value != other:
            return False
    return True


# ============================================================================
# 0. the premise: the v2 gym id really does terminate on outcome
# ============================================================================
def test_terminate_on_outcome_is_a_shipped_configuration():
    """A cfg with terminate_on_outcome=True exists and is registered.

    Without this the whole file could be testing a branch nothing uses.
    """
    values = _cfg_values(PATH_HAZARD_CFG, "terminate_on_outcome")
    assert values == {False, True}, (
        "path_hazard_env_cfg.py no longer ships both a fixed-horizon and an "
        f"outcome-terminated configuration (found {sorted(values)})"
    )
    cfg_source = PATH_HAZARD_CFG.read_text(encoding="utf-8")
    outcome_class = cfg_source[cfg_source.index("class PathHazardV2EnvCfg"):]
    assert "terminate_on_outcome: bool = True" in outcome_class, (
        "PathHazardV2EnvCfg no longer sets terminate_on_outcome=True"
    )
    registration = PATH_HAZARD_INIT.read_text(encoding="utf-8")
    assert "PathHazardV2EnvCfg" in registration, (
        "no gym id points at PathHazardV2EnvCfg, so the outcome-terminated "
        "branch would be unreachable"
    )


# ============================================================================
# 1. THE test: an episode that ends on its OUTCOME records ITS OWN values
# ============================================================================
def _success_then_record(latch: str) -> dict:
    """Time out one episode, win the next, then read what got recorded.

    Returns the values latched for env 0 plus the truth about the episode that
    produced them, so the caller can assert they are the same episode.
    """
    env = _PathHazard(num_envs=2, terminate_on_outcome=True, latch=latch)

    # Episode 0: neither env reaches a gate, so both time out. This is the
    # episode whose values a stale latch would keep serving.
    env.reset([0, 1], "ep0", scale=1.0)
    terminated, time_out = _run_episode(env, _crawl, 1.0)
    assert bool(time_out.all()) and not bool(terminated.any()), (
        "the crawl script no longer times out both envs"
    )
    stale = env.live(0)

    # Episode 1: env 0 reaches every gate, so it ends on its OUTCOME.
    env.reset([0, 1], "ep1", scale=1.5)
    terminated, time_out = _run_episode(env, _gate_chain, 1.5)
    assert bool(terminated[0]) and not bool(time_out[0]), (
        "env 0 no longer ends episode 1 on the outcome rather than the horizon"
    )
    truth = env.live(0)
    assert truth["success"], "the gate chain no longer produces a success"

    # Isaac Lab resets the envs that finished; the recording happens there.
    env.reset([0], "ep2", scale=2.0)
    return {"recorded": env.recorded(0), "truth": truth, "stale": stale,
            "other_env": env.recorded(1)}


def test_outcome_terminated_episode_records_its_own_values():
    """The whole point. A win must latch ITS success, ITS metrics, ITS scenario."""
    result = _success_then_record("shipped")
    recorded, truth, stale = result["recorded"], result["truth"], result["stale"]

    # The premise, asserted rather than assumed: the two episodes really are
    # distinguishable, or "recorded the right one" would be vacuous.
    for name in ("path_length", "min_clearance", "scenario_hashes"):
        assert recorded[name] != stale[name] or truth[name] != stale[name], (
            f"episodes 0 and 1 share {name}, so this test cannot tell which "
            "one was recorded"
        )

    assert recorded["success"], (
        "an episode that ended by reaching the last gate latched "
        "episode_success=False. Under terminate_on_outcome=True a success "
        "ALWAYS terminates, so with a time_out-only latch no success can ever "
        "be recorded -- which is why curves/g4080/curve_ph2_s42.json reports "
        "sr 0.0 on all 94 rungs"
    )
    assert recorded["scenario_hashes"] == truth["scenario_hashes"], (
        "the certificate digest latched for the finished episode names a "
        f"DIFFERENT episode: recorded {recorded['scenario_hashes']}, the "
        f"episode that just ended was {truth['scenario_hashes']}"
    )
    assert recorded["scenario_hashes"] != stale["scenario_hashes"], (
        "the latched digest is still the PREVIOUS episode's"
    )
    for name in ("path_length", "min_clearance", "xte_rms", "route_length",
                 "gates_passed"):
        assert recorded[name] == truth[name], (
            f"episode_{name} latched {recorded[name]} for an "
            f"outcome-terminated episode worth {truth[name]} "
            f"(the previous timed-out episode was worth {stale[name]})"
        )

    # An env whose episode is still running must NOT be recorded early.
    assert result["other_env"]["scenario_hashes"] == {
        "layout": "ep0", "scenario": "ep0"
    }, "an unfinished env was recorded"


def test_outcome_terminated_contact_records_a_negative_clearance():
    """The second fingerprint: 0 of 6016 episodes had a negative clearance.

    A contact before the last gate also TERMINATES under this cfg, so with the
    old latch a contact episode could never latch its own min clearance and the
    certificate's clearance column could never go negative.
    """
    env = _PathHazard(num_envs=2, terminate_on_outcome=True, latch="shipped")
    env.reset([0, 1], "ep0", scale=1.0)
    _run_episode(env, _crawl, 1.0)
    stale = env.live(0)
    assert stale["min_clearance"] > 0.0, (
        "the crawl script now ends in contact, so it can no longer serve as "
        "the clean episode a stale latch would keep serving"
    )

    env.reset([0, 1], "ep1", scale=1.0)
    terminated, time_out = _run_episode(env, _into_the_obstacle, 1.0)
    assert bool(terminated[0]) and not bool(time_out[0]), (
        "the contact script no longer terminates the episode on its outcome"
    )
    truth = env.live(0)
    assert truth["min_clearance"] < 0.0 and not truth["success"], (
        "the contact script no longer produces a negative clearance"
    )

    env.reset([0], "ep2", scale=1.0)
    recorded = env.recorded(0)
    assert recorded["min_clearance"] < 0.0, (
        "an episode that ended by hitting an obstacle latched "
        f"episode_min_clearance={recorded['min_clearance']} -- the previous "
        "timed-out episode's value. Every contact terminates under "
        "terminate_on_outcome=True, so a time_out-only latch makes a negative "
        "clearance unrecordable, which is why curve_ph2_s42.json has 0 of "
        "6016 episodes below zero"
    )
    assert recorded["min_clearance"] == truth["min_clearance"], recorded
    assert not recorded["success"]
    assert recorded["scenario_hashes"] == truth["scenario_hashes"]


def test_consecutive_outcome_episodes_do_not_repeat_one_record():
    """The third fingerprint: 1430 repeated (env, path, clearance) triples.

    Four episodes that each end on their outcome, each with a different layout.
    Every latched record must be that episode's own, so no two records repeat.
    """
    env = _PathHazard(num_envs=2, terminate_on_outcome=True, latch="shipped")
    scales = (1.0, 1.25, 1.5, 1.75)
    truths, records = [], []
    for index, scale in enumerate(scales):
        env.reset([0, 1], f"ep{index}", scale=scale)
        terminated, _time_out = _run_episode(env, _gate_chain, scale)
        assert bool(terminated[0]), f"episode {index} did not end on its outcome"
        truths.append(env.live(0))
        if index:
            records.append(env.recorded(0))
    env.reset([0, 1], "final", scale=2.0)
    records.append(env.recorded(0))

    for index, record in enumerate(records):
        assert record["scenario_hashes"], (
            f"record {index} carries NO scenario digest at all: the episode "
            "ended on its outcome and never reached the recording block, so "
            "the certificate row for it would be stamped with whatever the "
            "previous timed-out episode left behind"
        )
    triples = [
        (round(record["path_length"], 6), round(record["min_clearance"], 6),
         record["scenario_hashes"]["scenario"])
        for record in records
    ]
    assert len({tuple(map(str, triple)) for triple in triples}) == len(triples), (
        "consecutive episodes latched a repeated "
        f"(path_length, min_clearance, scenario) triple: {triples}. That is "
        "the signature of a stale latch -- 1430 of 6016 records in "
        "curves/g4080/curve_ph2_s42.json repeat one inside a single rung"
    )
    for index, record in enumerate(records):
        assert record["path_length"] == truths[index]["path_length"], (
            f"record {index} carries episode {index}'s path length "
            f"{truths[index]['path_length']} as {record['path_length']}"
        )
        assert record["scenario_hashes"] == truths[index]["scenario_hashes"]


# ============================================================================
# 2. the fixed-horizon cfg must be untouched, bit for bit
# ============================================================================
_SCRIPTS = (
    ("ep0", _crawl),
    ("ep1", _gate_chain),
    ("ep2", _into_the_obstacle),
    ("ep3", _crawl),
    ("ep4", _gate_chain),
)


def _trace(latch: str, terminate_on_outcome: bool) -> list:
    """Every latch value after every step, and every record after every reset."""
    env = _PathHazard(
        num_envs=2, terminate_on_outcome=terminate_on_outcome, latch=latch
    )
    trace = []
    for tag, script in _SCRIPTS:
        env.reset([0, 1], tag, scale=1.0)
        trace.append(("record", env.latch_state()))
        for local in script(1.0):
            terminated, time_out = env.step(env.world([local] * env.num_envs))
            trace.append(("finished", env.base._episode_finished.clone()))
            if bool((terminated | time_out).any()):
                break
        else:
            raise AssertionError(f"{tag} neither terminated nor timed out")
    env.reset([0, 1], "final", scale=1.0)
    trace.append(("record", env.latch_state()))
    return trace


def test_fixed_horizon_behaviour_is_bit_identical_to_the_old_line():
    """terminate_on_outcome=False: `time_out | terminated` == `time_out`.

    On that cfg ``terminated`` is all-False by construction, so the fix must be
    a no-op -- the v1 gym id and the certified champion keep their numbers.
    Both traces are produced by the SHIPPED statements; only the one argument
    of the latch call differs.
    """
    shipped = _trace("shipped", terminate_on_outcome=False)
    old = _trace("old", terminate_on_outcome=False)
    assert len(shipped) == len(old), (len(shipped), len(old))
    for index, ((kind, left), (other_kind, right)) in enumerate(zip(shipped, old)):
        assert kind == other_kind, (index, kind, other_kind)
        if kind == "finished":
            assert torch.equal(left, right), (
                f"step {index}: _episode_finished differs between the shipped "
                "latch and the pre-fix line on a fixed-horizon cfg"
            )
        else:
            assert _same_state(left, right), (
                f"reset {index}: a per-episode record differs between the "
                "shipped latch and the pre-fix line on a fixed-horizon cfg"
            )


def test_the_two_latches_disagree_under_terminate_on_outcome():
    """Teeth for the test above, and for the whole file.

    If the fix is reverted the two variants become the same statement and this
    fails -- so the bit-identity test above can never pass vacuously.
    """
    shipped = _trace("shipped", terminate_on_outcome=True)
    old = _trace("old", terminate_on_outcome=True)
    differences = [
        index
        for index, ((kind, left), (_kind, right)) in enumerate(zip(shipped, old))
        if (
            not torch.equal(left, right)
            if kind == "finished"
            else not _same_state(left, right)
        )
    ]
    assert differences, (
        "the shipped latch and the pre-fix `copy_(time_out)` line produced "
        "identical traces under terminate_on_outcome=True. Either the fix has "
        "been reverted -- self._episode_finished.copy_(time_out | terminated) "
        "in path_hazard_env.py _get_dones -- or the two variants are no longer "
        "distinguishable and this file has lost its teeth"
    )


# ============================================================================
# 3. source-level guard (grep-strength, not execution-strength)
# ============================================================================
def test_the_latch_reads_both_termination_and_timeout():
    """The shipped latch argument must mention terminated as well as time_out."""
    (call,) = _latch_calls(_GET_DONES)
    assert len(call.args) == 1, ast.dump(call)
    names = {
        node.id for node in ast.walk(call.args[0]) if isinstance(node, ast.Name)
    }
    assert names == {"time_out", "terminated"}, (
        "self._episode_finished.copy_(...) no longer reads exactly "
        f"(time_out, terminated) but {sorted(names)}. 'Finished' has to mean "
        "'Isaac Lab is about to reset this env', which is terminated OR timed "
        "out; reading time_out alone loses every outcome-terminated episode"
    )
    assert isinstance(call.args[0], ast.BinOp) and isinstance(
        call.args[0].op, ast.BitOr
    ), "the latch argument is no longer the OR of the two masks"


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
