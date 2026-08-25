# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the PER-EPISODE RECORDING LATCH of hazard_nav and harbor_mission.

WHAT THIS FILE OWNS
-------------------
Two lines, one per family::

    tasks/hazard_nav/hazard_nav_env.py:1457
        self._episode_finished.copy_(terminated | time_out)
    tasks/harbor_mission/harbor_mission_env.py:1142
        self._episode_finished.copy_(terminated | time_out)

``tasks/_shared/test_episode_latch.py`` owns the SAME line in
``tasks/path_hazard/path_hazard_env.py`` and documents what narrowing it to
``copy_(time_out)`` already cost this project: on
``Isaac-USV-PathHazard-Direct-v2`` every episode that ended on its OUTCOME
skipped the ``if len(completed_ids) > 0:`` recording block, so each certificate
row carried the PREVIOUS episode's success, time, clearance and scenario
digest.  The corpus fingerprint was 94 checkpoints reading success rate 0.0,
zero negative-clearance episodes in 6016, and about 1430 repeated
``(env, path_length, min_clearance)`` triples.

Reverting either of the two lines above left all sixteen suites green.  That is
worse than the path_hazard hole was, because BOTH of these families terminate
episodes early BY DESIGN and on their headline gym ids:

  * hazard_nav terminates on the goal UNCONDITIONALLY -- ``terminated``
    contains ``reached_now`` on every cfg the family ships, with contact
    termination added on top wherever ``contact_terminates`` is True.  So under
    the narrowed latch a hazard_nav SUCCESS can never be recorded at all: the
    only episodes that reach the recording block are the ones that failed to
    arrive.  ``Episode/success`` would read 0.0 for a perfect policy.
  * harbor_mission terminates on the stage milestone and on contact whenever
    ``terminate_on_milestone`` / ``terminate_on_contact`` are set, which every
    registered ``Isaac-USV-HarborStage{1,2,3}-Direct-v1`` cfg does.

WHY THE TESTS LOOK LIKE THIS
----------------------------
Both env modules import ``isaaclab`` at module scope, so the classes cannot be
imported on a machine without Isaac Sim.  Exactly as
``tasks/_shared/test_episode_latch.py`` and
``tasks/_shared/test_scenario_regression.py`` do, every test below AST-lifts the
SHIPPED statements out of the real source file and executes them against a
fabricated ``self``.  Editing the env source changes what these tests run, so a
reverted fix fails here instead of passing quietly.

The "old" latch is never retyped: it is the shipped statement with the single
argument of the ``self._episode_finished.copy_(...)`` call replaced by the bare
name ``time_out``, which is the pre-fix line.  A build that reverts the fix
makes the two variants identical, and the two
``..._latches_disagree_...`` tests say so, so nothing here can pass vacuously.

HOW MUCH OF THE SHIPPED CODE RUNS -- LABELLED HONESTLY
------------------------------------------------------
hazard_nav       REAL TEETH, whole method.  The complete shipped ``_get_dones``
                 runs, together with the shipped ``_com_xy``,
                 ``_horizontal_distance`` and ``_clearance`` helpers, so the
                 success, the contact, the path length and the min clearance
                 that get latched are computed by shipped code from a boat
                 position the test drives.  The shipped recording span of
                 ``_reset_idx`` then decides what is written.

harbor_mission   REAL TEETH, tail of the method.  The shipped span from
                 ``depth = int(self.cfg.mission_depth)`` through
                 ``self._episode_finished.copy_(...)`` runs -- that is the stage
                 selection, ``self._success |= success_now``, the
                 ``_first_success_time_s`` write, the whole ``terminated``
                 construction and the latch itself.  The gate-crossing and
                 docking geometry ABOVE that span is not re-run here (it needs a
                 harbour of USD gates); the milestone booleans that feed the
                 span are supplied by the test, and the per-episode
                 ``path_length`` / ``_min_clearance`` values are supplied by the
                 test too.  The claim under test -- which finished episodes
                 reach the recording block, and whose numbers they write -- is
                 executed end to end.

NOTHING ABOUT THE TASK DEFINITION IS TOUCHED HERE.  No range, distribution,
reward, observation, termination rule or physics constant is asserted or
changed; the only claim is about which finished episodes reach the recording
block.

Run: python tasks/_shared/test_episode_latch_families.py
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

HAZARD_NAV_ENV = _TASKS / "hazard_nav" / "hazard_nav_env.py"
HAZARD_NAV_CFG = _TASKS / "hazard_nav" / "hazard_nav_env_cfg.py"
HAZARD_NAV_GEOMETRY = _TASKS / "hazard_nav" / "hazard_geometry.py"
HAZARD_NAV_INIT = _TASKS / "hazard_nav" / "__init__.py"

HARBOR_MISSION_ENV = _TASKS / "harbor_mission" / "harbor_mission_env.py"
HARBOR_MISSION_CFG = _TASKS / "harbor_mission" / "harbor_mission_env_cfg.py"
HARBOR_MISSION_INIT = _TASKS / "harbor_mission" / "__init__.py"

SOURCES_UNDER_TEST = (
    HAZARD_NAV_ENV,
    HARBOR_MISSION_ENV,
    Path(__file__).resolve(),
)


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


_HAZARD_GEOMETRY = _load_module(HAZARD_NAV_GEOMETRY, "_latch_families_hn_geometry")
analytic_min_clearance = _HAZARD_GEOMETRY.analytic_min_clearance


# -------------------------------------------------------- the latch statement --
def _latch_calls(node: ast.AST) -> list[ast.Call]:
    """Every ``self._episode_finished.copy_(...)`` call in a subtree."""
    return [
        inner
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "copy_"
        and isinstance(inner.func.value, ast.Attribute)
        and inner.func.value.attr == "_episode_finished"
    ]


def _narrow_latch(node):
    """A deep copy of ``node`` with the latch argument replaced by ``time_out``.

    Derived from the source rather than retyped: the single argument of the
    ``self._episode_finished.copy_(...)`` call becomes the bare name
    ``time_out``, which is what the line read before the fix.
    """
    clone = copy.deepcopy(node)
    holder = ast.Module(body=list(clone) if isinstance(clone, list) else [clone],
                        type_ignores=[])
    calls = _latch_calls(holder)
    assert len(calls) == 1, (
        f"expected exactly one self._episode_finished.copy_(...) call, "
        f"found {len(calls)}"
    )
    calls[0].args = [ast.Name(id="time_out", ctx=ast.Load())]
    ast.fix_missing_locations(holder)
    return holder.body if isinstance(clone, list) else holder.body[0]


def _recording_span(path: Path, class_name: str) -> list[ast.stmt]:
    """``completed_ids = ...`` plus the ``if len(completed_ids) > 0:`` block.

    The block whose ENTRY CONDITION is the thing the latch decides.
    """
    body = _method(path, class_name, "_reset_idx").body
    starts = [
        index
        for index, statement in enumerate(body)
        if "completed_ids" in _assigned_names(statement)
    ]
    assert len(starts) == 1, (
        f"{path.name}: {class_name}._reset_idx assigns completed_ids "
        f"{len(starts)} times, expected exactly one"
    )
    index = starts[0]
    guard = body[index + 1]
    assert isinstance(guard, ast.If), (
        f"{path.name}: the statement after completed_ids is no longer the "
        "`if len(completed_ids) > 0:` recording block"
    )
    assert _writes_self_attribute(guard, "episode_scenario_hashes"), (
        f"{path.name}: the recording block no longer latches "
        "episode_scenario_hashes, so the certificate's per-episode digest is "
        "written somewhere this test does not see"
    )
    assert _writes_self_attribute(guard, "episode_success"), (
        f"{path.name}: the recording block no longer latches episode_success"
    )
    return [body[index], guard]


# ------------------------------------------------------------------- fixtures --
class _Fake:
    """A stand-in env: only the attributes the lifted statements read."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _AutoFake:
    """A stand-in env whose UNSPECIFIED tensor buffers default to zeros.

    The recording blocks touch two dozen per-episode metric buffers that have
    nothing to do with the latch under test.  Defaulting them keeps this file
    from breaking when an unrelated metric is added, while every buffer the
    CLAIM is about is supplied explicitly.  Same device as
    ``tasks/_shared/test_hazard_env_wiring.py``.
    """

    def __init__(self, auto_num_envs: int, **fields):
        object.__setattr__(self, "_auto_num_envs", auto_num_envs)
        self.__dict__.update(fields)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = torch.zeros(self._auto_num_envs, device=DEVICE)
        self.__dict__[name] = value
        return value


def _bools(num_envs: int) -> torch.Tensor:
    return torch.zeros(num_envs, dtype=torch.bool, device=DEVICE)


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
# hazard_nav
# ============================================================================
HN_GOAL_RADIUS_M = _cfg_default(HAZARD_NAV_CFG, "goal_radius")
HN_HALF_BEAM_M = _cfg_default(HAZARD_NAV_CFG, "half_beam_m")

# Test fixtures, not shipped ranges: one obstacle per env, placed off the
# straight run to the goal so the clean script below clears it and the contact
# script below drives into it.
HN_GOAL_DISTANCE_M = 20.0
HN_OBSTACLE_OFFSET_M = 5.0
HN_OBSTACLE_RADIUS_M = 1.0
HN_MAX_EPISODE_LENGTH = 6
HN_CONTROL_STEP_S = 0.1

_HN_GET_DONES = _method(HAZARD_NAV_ENV, "HazardNavEnv", "_get_dones")
_HN_HELPERS = ("_com_xy", "_horizontal_distance", "_clearance")
_HN_RECORD = _recording_span(HAZARD_NAV_ENV, "HazardNavEnv")


class _HazardNav:
    """A fabricated HazardNavEnv driven through the SHIPPED statements.

    ``step`` runs the shipped ``_get_dones`` (or its pre-fix twin) and ``reset``
    runs the shipped recording span of ``_reset_idx``.  Everything else -- the
    goal position, the obstacle, the boat position -- is supplied by the test,
    because the point under test is which finished episodes reach the recording
    block, not how a layout is drawn.
    """

    def __init__(self, num_envs=2, latch="shipped", contact_terminates=True,
                 clean_goal_gate=True):
        self.num_envs = num_envs
        origins = torch.zeros((num_envs, 3), device=DEVICE)
        origins[:, 0] = torch.arange(num_envs, dtype=torch.float32) * 200.0
        self.base = _AutoFake(
            num_envs,
            num_envs=num_envs,
            device=DEVICE,
            cfg=_Fake(
                half_beam_m=HN_HALF_BEAM_M,
                contact_terminates=contact_terminates,
                clean_goal_gate=clean_goal_gate,
                curriculum_frozen=True,
            ),
            curriculum=_Fake(update=lambda *_args: None),
            goal_radius=float(HN_GOAL_RADIUS_M),
            control_step_s=HN_CONTROL_STEP_S,
            max_episode_length=HN_MAX_EPISODE_LENGTH,
            scene=_Fake(env_origins=origins),
            robot=_Fake(
                data=_Fake(root_com_pos_w=torch.zeros((num_envs, 3), device=DEVICE))
            ),
            target_pos=torch.zeros((num_envs, 2), device=DEVICE),
            obstacle_centers=torch.zeros((num_envs, 1, 2), device=DEVICE),
            obstacle_radii=torch.full((num_envs, 1), HN_OBSTACLE_RADIUS_M,
                                      device=DEVICE),
            obstacle_active=torch.ones((num_envs, 1), dtype=torch.bool,
                                       device=DEVICE),
            episode_length_buf=torch.zeros(num_envs, dtype=torch.long,
                                           device=DEVICE),
            path_length=torch.zeros(num_envs, device=DEVICE),
            _previous_xy=torch.zeros((num_envs, 2), device=DEVICE),
            _min_clearance=torch.full((num_envs,), torch.inf, device=DEVICE),
            _first_success_time_s=torch.full((num_envs,), torch.nan, device=DEVICE),
            _reached_goal=_bools(num_envs),
            _contact_before_goal=_bools(num_envs),
            _success=_bools(num_envs),
            _clean_goal_entry_this_step=_bools(num_envs),
            _episode_finished=_bools(num_envs),
            episode_success=_bools(num_envs),
            time_to_success=torch.full((num_envs,), torch.nan, device=DEVICE),
            episode_min_clearance=torch.full((num_envs,), torch.nan, device=DEVICE),
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
        for name in _HN_HELPERS:
            _exec_statements(
                [_method(HAZARD_NAV_ENV, "HazardNavEnv", name)],
                HAZARD_NAV_ENV,
                namespace,
            )
            setattr(self.base, name, functools.partial(namespace[name], self.base))

        function = _HN_GET_DONES if latch == "shipped" else _narrow_latch(_HN_GET_DONES)
        _exec_statements([function], HAZARD_NAV_ENV, namespace)
        self._dones = namespace["_get_dones"]

    # -- driving ------------------------------------------------------------
    def world(self, local_xy) -> torch.Tensor:
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
            _HN_RECORD,
            HAZARD_NAV_ENV,
            {"torch": torch, "self": self.base, "env_ids": ids},
        )

        base = self.base
        origins_xy = base.scene.env_origins[ids, :2]
        goal_distance = HN_GOAL_DISTANCE_M * scale
        base.target_pos[ids] = origins_xy + torch.tensor(
            [[goal_distance, 0.0]], device=DEVICE
        )
        base.obstacle_centers[ids] = (
            origins_xy.unsqueeze(1)
            + torch.tensor([[[0.5 * goal_distance, HN_OBSTACLE_OFFSET_M]]],
                           device=DEVICE)
        )
        base.robot.data.root_com_pos_w[ids, :2] = origins_xy
        base.episode_length_buf[ids] = 0
        base.path_length[ids] = 0.0
        base._success[ids] = False
        base._reached_goal[ids] = False
        base._contact_before_goal[ids] = False
        base._clean_goal_entry_this_step[ids] = False
        base._first_success_time_s[ids] = torch.nan
        base._episode_finished[ids] = False
        base._previous_xy[ids] = origins_xy
        initial_clearance = analytic_min_clearance(
            origins_xy,
            base.obstacle_centers[ids],
            base.obstacle_radii[ids],
            half_beam_m=HN_HALF_BEAM_M,
            active_mask=base.obstacle_active[ids],
        )
        base._min_clearance[ids] = initial_clearance
        base._contact_before_goal[ids] = initial_clearance < 0.0
        for env_index in ids.tolist():
            base._scenario_params[env_index] = {"tag": tag}
            base._scenario_hashes[env_index] = {"layout": tag, "scenario": tag}

    # -- observation --------------------------------------------------------
    def live(self, env_index: int) -> dict:
        base = self.base
        return {
            "success": bool(base._success[env_index]),
            "path_length": float(base.path_length[env_index]),
            "min_clearance": float(base._min_clearance[env_index]),
            "scenario_hashes": dict(base._scenario_hashes[env_index]),
        }

    def recorded(self, env_index: int) -> dict:
        base = self.base
        return {
            "success": bool(base.episode_success[env_index]),
            "path_length": float(base.episode_path_length[env_index]),
            "min_clearance": float(base.episode_min_clearance[env_index]),
            "scenario_hashes": dict(base.episode_scenario_hashes[env_index]),
        }

    def latch_state(self) -> dict:
        base = self.base
        names = (
            "episode_success",
            "time_to_success",
            "episode_path_length",
            "episode_min_clearance",
        )
        state = {name: getattr(base, name).clone() for name in names}
        state["episode_scenario_hashes"] = [
            dict(entry) for entry in base.episode_scenario_hashes
        ]
        return state


def _hn_crawl(scale: float) -> list[tuple[float, float]]:
    """Never within goal_radius of the goal: the episode times out."""
    return [(float(index + 1), 0.0) for index in range(HN_MAX_EPISODE_LENGTH - 1)]


def _hn_to_the_goal(scale: float) -> list[tuple[float, float]]:
    """Arrive cleanly: the episode ends on its OUTCOME, well before the horizon."""
    distance = HN_GOAL_DISTANCE_M * scale
    return [
        (0.4 * distance, 0.0),
        (0.7 * distance, 0.0),
        (distance - 0.5 * HN_GOAL_RADIUS_M, 0.0),
    ]


def _hn_into_the_obstacle(scale: float) -> list[tuple[float, float]]:
    """Drive onto the obstacle centre: contact, and contact terminates."""
    distance = HN_GOAL_DISTANCE_M * scale
    return [
        (0.25 * distance, 0.0),
        (0.5 * distance, HN_OBSTACLE_OFFSET_M),
    ]


def _run_episode(env, script, scale: float):
    """Step until Isaac Lab would reset, i.e. until terminated | time_out."""
    for local in script(scale):
        terminated, time_out = env.step(env.world([local] * env.num_envs))
        if bool((terminated | time_out).any()):
            return terminated, time_out
    raise AssertionError("the script ended without terminating or timing out")


# ---------------------------------------------------------------- premise --
def test_hazard_nav_always_terminates_on_the_goal():
    """GREP/AST-LEVEL. The premise, asserted rather than assumed.

    ``terminated`` contains ``reached_now`` on EVERY hazard_nav cfg -- there is
    no fixed-horizon variant of this family -- so under a time_out-only latch a
    hazard_nav success is unrecordable outright.
    """
    source = HAZARD_NAV_ENV.read_text(encoding="utf-8")
    body = ast.get_source_segment(source, _HN_GET_DONES)
    assert "reached_now | contact_now" in body and "else reached_now" in body, (
        "hazard_nav _get_dones no longer terminates unconditionally on the "
        "goal, so the premise of this file has changed and its claims need "
        "rereading"
    )
    values = _cfg_values(HAZARD_NAV_CFG, "contact_terminates")
    assert True in values, (
        "no hazard_nav cfg sets contact_terminates=True any more, so the "
        "contact-termination branch tested below would be unreachable"
    )
    registration = HAZARD_NAV_INIT.read_text(encoding="utf-8")
    assert "HazardNavEnvCfg" in registration, (
        "no gym id points at a hazard_nav cfg"
    )


# --------------------------------- THE test: a win records ITS OWN values --
def test_hazard_nav_outcome_terminated_episode_records_its_own_values():
    """REAL TEETH. A hazard_nav success must latch ITS success, metrics, scenario."""
    env = _HazardNav(num_envs=2, latch="shipped")

    # Episode 0: neither env gets near the goal, so both time out. This is the
    # episode whose values a stale latch would keep serving.
    env.reset([0, 1], "ep0", scale=1.0)
    terminated, time_out = _run_episode(env, _hn_crawl, 1.0)
    assert bool(time_out.all()) and not bool(terminated.any()), (
        "the crawl script no longer times out both envs"
    )
    stale = env.live(0)

    # Episode 1: env 0 arrives, so it ends on its OUTCOME.
    env.reset([0, 1], "ep1", scale=1.5)
    terminated, time_out = _run_episode(env, _hn_to_the_goal, 1.5)
    assert bool(terminated[0]) and not bool(time_out[0]), (
        "env 0 no longer ends episode 1 on the outcome rather than the horizon"
    )
    truth = env.live(0)
    assert truth["success"], "the arrival script no longer produces a success"

    # Isaac Lab resets the envs that finished; the recording happens there.
    env.reset([0], "ep2", scale=2.0)
    recorded = env.recorded(0)

    # The premise, asserted rather than assumed: the two episodes really are
    # distinguishable, or "recorded the right one" would be vacuous.
    for name in ("path_length", "min_clearance", "scenario_hashes"):
        assert truth[name] != stale[name], (
            f"episodes 0 and 1 share {name}, so this test cannot tell which "
            "one was recorded"
        )

    assert recorded["success"], (
        "a hazard_nav episode that ended by REACHING THE GOAL latched "
        "episode_success=False. hazard_nav terminates on the goal on every "
        "cfg it ships, so with a time_out-only latch no arrival can ever be "
        "recorded and Episode/success reads 0.0 for a perfect policy -- the "
        "same defect that put 94 checkpoints at sr 0.0 in "
        "curves/g4080/curve_ph2_s42.json"
    )
    assert recorded["scenario_hashes"] == truth["scenario_hashes"], (
        "the certificate digest latched for the finished episode names a "
        f"DIFFERENT episode: recorded {recorded['scenario_hashes']}, the "
        f"episode that just ended was {truth['scenario_hashes']}"
    )
    assert recorded["scenario_hashes"] != stale["scenario_hashes"], (
        "the latched digest is still the PREVIOUS episode's"
    )
    for name in ("path_length", "min_clearance"):
        assert recorded[name] == truth[name], (
            f"episode_{name} latched {recorded[name]} for an "
            f"outcome-terminated episode worth {truth[name]} (the previous "
            f"timed-out episode was worth {stale[name]})"
        )

    # An env whose episode is still running must NOT be recorded early.
    assert env.recorded(1)["scenario_hashes"] == {
        "layout": "ep0", "scenario": "ep0"
    }, "an unfinished env was recorded"


def test_hazard_nav_contact_terminated_episode_records_a_negative_clearance():
    """REAL TEETH. Corpus fingerprint 2: 0 of 6016 episodes below zero clearance.

    A contact terminates whenever ``contact_terminates`` is True, so with the
    narrowed latch a contact episode could never latch its own min clearance
    and the certificate's clearance column could never go negative.
    """
    env = _HazardNav(num_envs=2, latch="shipped", contact_terminates=True)
    env.reset([0, 1], "ep0", scale=1.0)
    _run_episode(env, _hn_crawl, 1.0)
    stale = env.live(0)
    assert stale["min_clearance"] > 0.0, (
        "the crawl script now ends in contact, so it can no longer serve as "
        "the clean episode a stale latch would keep serving"
    )

    env.reset([0, 1], "ep1", scale=1.0)
    terminated, time_out = _run_episode(env, _hn_into_the_obstacle, 1.0)
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
        "a hazard_nav episode that ended by hitting an obstacle latched "
        f"episode_min_clearance={recorded['min_clearance']} -- the previous "
        "timed-out episode's value. Every contact terminates under "
        "contact_terminates=True, so a time_out-only latch makes a negative "
        "clearance unrecordable, which is why curve_ph2_s42.json has 0 of "
        "6016 episodes below zero"
    )
    assert recorded["min_clearance"] == truth["min_clearance"], recorded
    assert not recorded["success"]
    assert recorded["scenario_hashes"] == truth["scenario_hashes"]


def test_hazard_nav_consecutive_outcome_episodes_do_not_repeat_one_record():
    """REAL TEETH. Corpus fingerprint 3: 1430 repeated (env, path, clearance) triples."""
    env = _HazardNav(num_envs=2, latch="shipped")
    scales = (1.0, 1.25, 1.5, 1.75)
    truths, records = [], []
    for index, scale in enumerate(scales):
        env.reset([0, 1], f"ep{index}", scale=scale)
        terminated, _time_out = _run_episode(env, _hn_to_the_goal, scale)
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
        "consecutive hazard_nav episodes latched a repeated "
        f"(path_length, min_clearance, scenario) triple: {triples}. That is "
        "the signature of a stale latch -- 1430 of 6016 records in "
        "curves/g4080/curve_ph2_s42.json repeat one inside a single rung"
    )
    for index, record in enumerate(records):
        assert record["path_length"] == truths[index]["path_length"], (
            f"record {index} does not carry episode {index}'s path length "
            f"{truths[index]['path_length']} (got {record['path_length']})"
        )
        assert record["scenario_hashes"] == truths[index]["scenario_hashes"]


# ------------------------------------------------- anti-vacuity + no-op --
_HN_SCRIPTS = (
    ("ep0", _hn_crawl),
    ("ep1", _hn_to_the_goal),
    ("ep2", _hn_into_the_obstacle),
    ("ep3", _hn_crawl),
    ("ep4", _hn_to_the_goal),
)


def _hn_trace(latch: str, scripts) -> list:
    """Every latch value after every step, and every record after every reset."""
    env = _HazardNav(num_envs=2, latch=latch)
    trace = []
    for tag, script in scripts:
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


def _compare_traces(shipped, old):
    assert len(shipped) == len(old), (len(shipped), len(old))
    differences = []
    for index, ((kind, left), (other_kind, right)) in enumerate(zip(shipped, old)):
        assert kind == other_kind, (index, kind, other_kind)
        same = (
            torch.equal(left, right) if kind == "finished"
            else _same_state(left, right)
        )
        if not same:
            differences.append((index, kind))
    return differences


def test_hazard_nav_horizon_only_episodes_are_bit_identical_to_the_old_line():
    """REAL TEETH. On episodes that end at the HORIZON the fix must be a no-op.

    ``terminated`` is all-False on every step of a pure time-out episode, so
    ``terminated | time_out`` and ``time_out`` are the same mask and no
    certified hazard_nav number moves.  Both traces are produced by the SHIPPED
    statements; only the one argument of the latch call differs.
    """
    horizon_only = tuple((f"ep{index}", _hn_crawl) for index in range(4))
    differences = _compare_traces(
        _hn_trace("shipped", horizon_only), _hn_trace("old", horizon_only)
    )
    assert not differences, (
        "the shipped latch and the pre-fix `copy_(time_out)` line disagree on "
        f"episodes that only ever time out: {differences}"
    )


def test_hazard_nav_latches_disagree_under_early_termination():
    """REAL TEETH, anti-vacuity. Guards the no-op test above and this whole half.

    If the fix is reverted the two variants become the same statement and this
    fails -- so the bit-identity test above can never pass vacuously.
    """
    differences = _compare_traces(
        _hn_trace("shipped", _HN_SCRIPTS), _hn_trace("old", _HN_SCRIPTS)
    )
    assert differences, (
        "the shipped latch and the pre-fix `copy_(time_out)` line produced "
        "identical traces on episodes that end by reaching the goal and by "
        "hitting an obstacle. Either the fix has been reverted -- "
        "self._episode_finished.copy_(terminated | time_out) in "
        "hazard_nav_env.py _get_dones -- or the two variants are no longer "
        "distinguishable and this file has lost its teeth"
    )


# ============================================================================
# harbor_mission
# ============================================================================
HM_MAX_EPISODE_LENGTH = 6
HM_CONTROL_STEP_S = 0.1

_HM_GET_DONES = _method(HARBOR_MISSION_ENV, "HarborMissionEnv", "_get_dones")
_HM_RECORD = _recording_span(HARBOR_MISSION_ENV, "HarborMissionEnv")


def _hm_tail() -> list[ast.stmt]:
    """The shipped ``_get_dones`` tail: stage selection through the latch.

    Starts at ``depth = int(self.cfg.mission_depth)`` -- the first statement of
    the scoring-and-termination block -- and ends at the
    ``self._episode_finished.copy_(...)`` statement.  Everything the latch
    depends on downstream of the harbour geometry therefore runs as shipped:
    which milestone scores the stage, ``self._success |= success_now``, the
    ``_first_success_time_s`` write and the whole ``terminated`` construction.
    """
    body = _HM_GET_DONES.body
    starts = [
        index for index, statement in enumerate(body)
        if "depth" in _assigned_names(statement)
    ]
    assert len(starts) == 1, (
        "harbor_mission_env.py: _get_dones assigns `depth` "
        f"{len(starts)} times, expected exactly one"
    )
    ends = [
        index for index, statement in enumerate(body)
        if _latch_calls(statement)
    ]
    assert len(ends) == 1, (
        "harbor_mission_env.py: _get_dones has "
        f"{len(ends)} statements calling self._episode_finished.copy_(...), "
        "expected exactly one"
    )
    assert ends[0] > starts[0], (
        "harbor_mission_env.py: the latch no longer follows the stage-scoring "
        "block, so this span no longer covers it"
    )
    span = body[starts[0]:ends[0] + 1]
    text = "\n".join(
        ast.dump(statement) for statement in span
    )
    assert "'_success'" in text and "'terminated'" in text, (
        "the harbor_mission tail span no longer both scores the episode and "
        "builds `terminated`, so it is the wrong span"
    )
    return span


_HM_TAIL = _hm_tail()


class _HarborMission:
    """A fabricated HarborMissionEnv driven through the SHIPPED tail + record.

    ``step`` runs the shipped ``_get_dones`` tail (or its pre-fix twin) and
    ``reset`` runs the shipped recording span of ``_reset_idx``.  The milestone
    booleans that feed the tail, and the per-episode ``path_length`` /
    ``_min_clearance``, are supplied by the test -- see the module docstring for
    what that does and does not claim.
    """

    def __init__(self, num_envs=2, latch="shipped", mission_depth=3,
                 terminate_on_milestone=True, terminate_on_contact=True,
                 stage2_full_mission_rewards=False):
        self.num_envs = num_envs
        self.base = _AutoFake(
            num_envs,
            num_envs=num_envs,
            device=DEVICE,
            cfg=_Fake(
                mission_depth=mission_depth,
                terminate_on_milestone=terminate_on_milestone,
                terminate_on_contact=terminate_on_contact,
                stage2_full_mission_rewards=stage2_full_mission_rewards,
                curriculum_frozen=True,
            ),
            control_step_s=HM_CONTROL_STEP_S,
            max_episode_length=HM_MAX_EPISODE_LENGTH,
            episode_length_buf=torch.zeros(num_envs, dtype=torch.long,
                                           device=DEVICE),
            path_length=torch.zeros(num_envs, device=DEVICE),
            _min_clearance=torch.full((num_envs,), torch.inf, device=DEVICE),
            _previous_xy=torch.zeros((num_envs, 2), device=DEVICE),
            _first_success_time_s=torch.full((num_envs,), torch.nan, device=DEVICE),
            _contact_before_dock=_bools(num_envs),
            _stage_goal_entry_this_step=_bools(num_envs),
            _success=_bools(num_envs),
            _episode_finished=_bools(num_envs),
            _m1=_bools(num_envs),
            _m2=_bools(num_envs),
            _m3=_bools(num_envs),
            m1=_bools(num_envs),
            m2=_bools(num_envs),
            m3=_bools(num_envs),
            _m1_step=torch.zeros(num_envs, device=DEVICE),
            _m2_step=torch.ones(num_envs, device=DEVICE),
            _m3_step=torch.full((num_envs,), 2.0, device=DEVICE),
            episode_success=_bools(num_envs),
            time_to_success=torch.full((num_envs,), torch.nan, device=DEVICE),
            episode_min_clearance=torch.full((num_envs,), torch.nan, device=DEVICE),
            _scenario_params=[{} for _ in range(num_envs)],
            _scenario_hashes=[{} for _ in range(num_envs)],
            episode_scenario=[{} for _ in range(num_envs)],
            episode_scenario_hashes=[{} for _ in range(num_envs)],
            extras={},
        )
        self._tail = _HM_TAIL if latch == "shipped" else _narrow_latch(_HM_TAIL)

    # -- driving ------------------------------------------------------------
    def step(self, *, milestone=False, contact=False):
        """One shipped tail evaluation. Returns (terminated, time_out)."""
        num_envs = self.num_envs
        self.base.episode_length_buf += 1
        flag = torch.full((num_envs,), bool(milestone), dtype=torch.bool,
                          device=DEVICE)
        clearance = torch.full(
            (num_envs,), -1.0 if contact else 3.0, device=DEVICE
        )
        namespace = {
            "torch": torch,
            "np": np,
            "self": self.base,
            # Locals the harbour geometry above the span would have produced.
            "current_xy": torch.zeros((num_envs, 2), device=DEVICE),
            "prefix_active": torch.ones(num_envs, dtype=torch.bool, device=DEVICE),
            "clearance": clearance,
            "gate2_crossed": flag,
            "field_exit_crossed": flag,
            "dock_completed": flag,
            "success_full": flag & ~self.base._contact_before_dock,
        }
        _exec_statements(self._tail, HARBOR_MISSION_ENV, namespace)
        return namespace["terminated"], namespace["time_out"]

    def reset(self, env_ids, tag: str, path_length: float, min_clearance: float):
        """Run the SHIPPED recording span, then start a new episode."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=DEVICE)
        _exec_statements(
            _HM_RECORD,
            HARBOR_MISSION_ENV,
            {"torch": torch, "self": self.base, "env_ids": ids},
        )

        base = self.base
        base.episode_length_buf[ids] = 0
        base.path_length[ids] = float(path_length)
        base._min_clearance[ids] = float(min_clearance)
        base._success[ids] = False
        base._contact_before_dock[ids] = False
        base._stage_goal_entry_this_step[ids] = False
        base._first_success_time_s[ids] = torch.nan
        base._episode_finished[ids] = False
        for env_index in ids.tolist():
            base._scenario_params[env_index] = {"tag": tag}
            base._scenario_hashes[env_index] = {"layout": tag, "scenario": tag}

    # -- observation --------------------------------------------------------
    def live(self, env_index: int) -> dict:
        base = self.base
        return {
            "success": bool(base._success[env_index]),
            "path_length": float(base.path_length[env_index]),
            "min_clearance": float(base._min_clearance[env_index]),
            "scenario_hashes": dict(base._scenario_hashes[env_index]),
        }

    def recorded(self, env_index: int) -> dict:
        base = self.base
        return {
            "success": bool(base.episode_success[env_index]),
            "path_length": float(base.episode_path_length[env_index]),
            "min_clearance": float(base.episode_min_clearance[env_index]),
            "scenario_hashes": dict(base.episode_scenario_hashes[env_index]),
        }

    def latch_state(self) -> dict:
        base = self.base
        names = (
            "episode_success",
            "time_to_success",
            "episode_path_length",
            "episode_min_clearance",
        )
        state = {name: getattr(base, name).clone() for name in names}
        state["episode_scenario_hashes"] = [
            dict(entry) for entry in base.episode_scenario_hashes
        ]
        return state


def _hm_run(env, *, milestone_on_step=None, contact_on_step=None):
    """Step until the shipped tail says Isaac Lab would reset."""
    for index in range(HM_MAX_EPISODE_LENGTH):
        terminated, time_out = env.step(
            milestone=(milestone_on_step == index),
            contact=(contact_on_step == index),
        )
        if bool((terminated | time_out).any()):
            return terminated, time_out
    raise AssertionError("the script ended without terminating or timing out")


# ---------------------------------------------------------------- premise --
def test_harbor_mission_ships_early_terminating_configurations():
    """GREP/AST-LEVEL. The registered stage cfgs do terminate on the milestone."""
    for field in ("terminate_on_milestone", "terminate_on_contact"):
        values = _cfg_values(HARBOR_MISSION_CFG, field)
        assert values == {False, True}, (
            f"harbor_mission_env_cfg.py no longer ships both settings of "
            f"{field} (found {sorted(values)}), so the early-termination "
            "branch this file is about may be unreachable"
        )
    cfg_source = HARBOR_MISSION_CFG.read_text(encoding="utf-8")
    stage = cfg_source[cfg_source.index("class HarborStage3EnvCfg"):]
    assert "terminate_on_milestone: bool = True" in stage, (
        "HarborStage3EnvCfg no longer sets terminate_on_milestone=True"
    )
    registration = HARBOR_MISSION_INIT.read_text(encoding="utf-8")
    assert "HarborStage" in registration, (
        "no gym id points at a HarborStage cfg, so the outcome-terminated "
        "branch would be unreachable"
    )


# --------------------------------- THE test: a win records ITS OWN values --
def test_harbor_mission_milestone_terminated_episode_records_its_own_values():
    """REAL TEETH. A docked mission must latch ITS success, metrics, scenario."""
    env = _HarborMission(num_envs=2, latch="shipped", mission_depth=3)

    # Episode 0: no milestone, so the episode runs to the horizon.
    env.reset([0, 1], "ep0", path_length=11.0, min_clearance=4.0)
    terminated, time_out = _hm_run(env)
    assert bool(time_out.all()) and not bool(terminated.any()), (
        "the idle script no longer times out both envs"
    )
    stale = env.live(0)
    assert not stale["success"]

    # Episode 1: the dock milestone fires, so the episode ends on its OUTCOME.
    env.reset([0, 1], "ep1", path_length=23.0, min_clearance=1.5)
    terminated, time_out = _hm_run(env, milestone_on_step=1)
    assert bool(terminated.all()) and not bool(time_out.any()), (
        "the milestone script no longer ends the episode on its outcome"
    )
    truth = env.live(0)
    assert truth["success"], (
        "the shipped tail no longer scores a completed depth-3 mission as a "
        "success, so this test cannot show a success failing to latch"
    )

    env.reset([0], "ep2", path_length=5.0, min_clearance=9.0)
    recorded = env.recorded(0)

    for name in ("path_length", "min_clearance", "scenario_hashes"):
        assert truth[name] != stale[name], (
            f"episodes 0 and 1 share {name}, so this test cannot tell which "
            "one was recorded"
        )

    assert recorded["success"], (
        "a harbor_mission episode that ended by COMPLETING ITS MISSION latched "
        "episode_success=False. Every HarborStage cfg sets "
        "terminate_on_milestone=True, so with a time_out-only latch the "
        "stage-terminal success is the one episode that can never be recorded "
        "-- the same defect that put 94 path_hazard checkpoints at sr 0.0"
    )
    assert recorded["scenario_hashes"] == truth["scenario_hashes"], (
        "the certificate digest latched for the finished episode names a "
        f"DIFFERENT episode: recorded {recorded['scenario_hashes']}, the "
        f"episode that just ended was {truth['scenario_hashes']}"
    )
    assert recorded["scenario_hashes"] != stale["scenario_hashes"], (
        "the latched digest is still the PREVIOUS episode's"
    )
    for name in ("path_length", "min_clearance"):
        assert recorded[name] == truth[name], (
            f"episode_{name} latched {recorded[name]} for a "
            f"milestone-terminated episode worth {truth[name]} (the previous "
            f"timed-out episode was worth {stale[name]})"
        )
    assert env.recorded(1)["scenario_hashes"] == {
        "layout": "ep0", "scenario": "ep0"
    }, "an unfinished env was recorded"


def test_harbor_mission_contact_terminated_episode_records_its_own_clearance():
    """REAL TEETH. A contact terminates too, so its clearance stays recordable."""
    env = _HarborMission(num_envs=2, latch="shipped", mission_depth=3)
    env.reset([0, 1], "ep0", path_length=11.0, min_clearance=4.0)
    _hm_run(env)
    stale = env.live(0)
    assert stale["min_clearance"] > 0.0

    env.reset([0, 1], "ep1", path_length=7.0, min_clearance=-0.75)
    terminated, time_out = _hm_run(env, contact_on_step=1)
    assert bool(terminated.all()) and not bool(time_out.any()), (
        "the contact script no longer terminates the episode on its outcome"
    )
    truth = env.live(0)

    env.reset([0], "ep2", path_length=1.0, min_clearance=6.0)
    recorded = env.recorded(0)
    assert recorded["min_clearance"] < 0.0, (
        "a harbor_mission episode that ended in CONTACT latched "
        f"episode_min_clearance={recorded['min_clearance']} -- the previous "
        "timed-out episode's value. Under terminate_on_contact=True every "
        "contact terminates, so a time_out-only latch makes a negative "
        "clearance unrecordable"
    )
    assert recorded["min_clearance"] == truth["min_clearance"]
    assert not recorded["success"]
    assert recorded["scenario_hashes"] == truth["scenario_hashes"]


def test_harbor_mission_consecutive_outcome_episodes_do_not_repeat_one_record():
    """REAL TEETH. The repeated-triple fingerprint, in the harbour family."""
    env = _HarborMission(num_envs=2, latch="shipped", mission_depth=3)
    episodes = [(13.0, 3.5), (17.0, 2.5), (21.0, 1.5), (25.0, 0.5)]
    truths, records = [], []
    for index, (path_length, clearance) in enumerate(episodes):
        env.reset([0, 1], f"ep{index}", path_length=path_length,
                  min_clearance=clearance)
        terminated, _time_out = _hm_run(env, milestone_on_step=1)
        assert bool(terminated[0]), f"episode {index} did not end on its outcome"
        truths.append(env.live(0))
        if index:
            records.append(env.recorded(0))
    env.reset([0, 1], "final", path_length=99.0, min_clearance=9.0)
    records.append(env.recorded(0))

    for index, record in enumerate(records):
        assert record["scenario_hashes"], (
            f"record {index} carries NO scenario digest at all: the episode "
            "ended on its outcome and never reached the recording block"
        )
    triples = [
        (round(record["path_length"], 6), round(record["min_clearance"], 6),
         record["scenario_hashes"]["scenario"])
        for record in records
    ]
    assert len({tuple(map(str, triple)) for triple in triples}) == len(triples), (
        "consecutive harbor_mission episodes latched a repeated "
        f"(path_length, min_clearance, scenario) triple: {triples}. That is "
        "the signature of a stale latch"
    )
    for index, record in enumerate(records):
        assert record["path_length"] == truths[index]["path_length"]
        assert record["scenario_hashes"] == truths[index]["scenario_hashes"]


# ------------------------------------------------- anti-vacuity + no-op --
def _hm_trace(latch: str, scripts, **kwargs) -> list:
    env = _HarborMission(num_envs=2, latch=latch, **kwargs)
    trace = []
    for index, (tag, milestone, contact) in enumerate(scripts):
        env.reset([0, 1], tag, path_length=10.0 + index, min_clearance=3.0 - index)
        trace.append(("record", env.latch_state()))
        for step in range(HM_MAX_EPISODE_LENGTH):
            terminated, time_out = env.step(
                milestone=(milestone == step), contact=(contact == step)
            )
            trace.append(("finished", env.base._episode_finished.clone()))
            if bool((terminated | time_out).any()):
                break
        else:
            raise AssertionError(f"{tag} neither terminated nor timed out")
    env.reset([0, 1], "final", path_length=0.0, min_clearance=0.0)
    trace.append(("record", env.latch_state()))
    return trace


_HM_SCRIPTS = (
    ("ep0", None, None),
    ("ep1", 1, None),
    ("ep2", None, 2),
    ("ep3", None, None),
    ("ep4", 0, None),
)


def test_harbor_mission_horizon_only_episodes_are_bit_identical_to_the_old_line():
    """REAL TEETH. Where nothing terminates early, the fix must be a no-op.

    The v1 gym id ``Isaac-USV-HarborMission-Direct-v1`` ships
    terminate_on_milestone=False and terminate_on_contact=False, so
    ``terminated`` is all-False by construction there and no certified harbour
    number moves.  Reproduced here with that cfg pair and idle episodes.
    """
    horizon_only = tuple((f"ep{index}", None, None) for index in range(4))
    differences = _compare_traces(
        _hm_trace("shipped", horizon_only, terminate_on_milestone=False,
                  terminate_on_contact=False),
        _hm_trace("old", horizon_only, terminate_on_milestone=False,
                  terminate_on_contact=False),
    )
    assert not differences, (
        "the shipped latch and the pre-fix `copy_(time_out)` line disagree on "
        f"a fixed-horizon harbour cfg: {differences}"
    )


def test_harbor_mission_latches_disagree_under_early_termination():
    """REAL TEETH, anti-vacuity. Guards the harbor_mission half of this file."""
    differences = _compare_traces(
        _hm_trace("shipped", _HM_SCRIPTS), _hm_trace("old", _HM_SCRIPTS)
    )
    assert differences, (
        "the shipped latch and the pre-fix `copy_(time_out)` line produced "
        "identical traces on episodes that end on the stage milestone and on "
        "contact. Either the fix has been reverted -- "
        "self._episode_finished.copy_(terminated | time_out) in "
        "harbor_mission_env.py _get_dones -- or the two variants are no longer "
        "distinguishable and this file has lost its teeth"
    )


# ============================================================================
# source-level guard (grep-strength, not execution-strength)
# ============================================================================
def test_both_latches_read_termination_as_well_as_timeout():
    """GREP/AST-LEVEL. The latch argument must mention terminated and time_out."""
    for path, class_name in (
        (HAZARD_NAV_ENV, "HazardNavEnv"),
        (HARBOR_MISSION_ENV, "HarborMissionEnv"),
    ):
        function = _method(path, class_name, "_get_dones")
        calls = _latch_calls(function)
        assert len(calls) == 1, (
            f"{path.name}: {len(calls)} self._episode_finished.copy_(...) calls"
        )
        (call,) = calls
        assert len(call.args) == 1, ast.dump(call)
        names = {
            node.id for node in ast.walk(call.args[0]) if isinstance(node, ast.Name)
        }
        assert names == {"time_out", "terminated"}, (
            f"{path.name}: self._episode_finished.copy_(...) no longer reads "
            f"exactly (time_out, terminated) but {sorted(names)}. 'Finished' "
            "has to mean 'Isaac Lab is about to reset this env', which is "
            "terminated OR timed out; reading time_out alone loses every "
            "episode that ended on its outcome"
        )
        assert isinstance(call.args[0], ast.BinOp) and isinstance(
            call.args[0].op, ast.BitOr
        ), f"{path.name}: the latch argument is no longer the OR of the two masks"


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
