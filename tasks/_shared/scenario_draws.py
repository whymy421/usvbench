# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The reset-path scenario draws, lifted out of the envs so they can be tested.

WHY THIS FILE EXISTS
--------------------
``tasks/_shared/scenario_rng.py`` owns the stream algebra: one blake2b-keyed
``torch.Generator`` per ``(protocol_version, eval_seed, env_index,
episode_index, group)``.  This file owns the *call sites*: the exact arithmetic
that station keeping, path following and path hazard used to run against the
GLOBAL torch RNG, now taking its uniforms from that stream instead.

It is deliberately Isaac-free (torch, numpy and math only), for the same reason
``tasks/station_keeping/current_ramp.py`` and ``tasks/_shared/sea_state.py``
are: the env modules import ``isaaclab`` at module scope and cannot be imported
on a machine without Isaac Sim, so the only way to unit-test a reset path on
CPU is for the env and the test to call the SAME function from here.
``tasks/_shared/test_scenario_wiring.py`` is that test.

WHAT IS AND IS NOT PRESERVED
----------------------------
Preserved: every distribution and every range.  Each helper draws raw
``U[0, 1)`` of exactly the shape ``torch.rand`` produced before and then applies
the caller's original arithmetic verbatim, so "only the source of the random
numbers changes" is checkable by eye against the before/after lines quoted in
each docstring.

Not preserved: the specific numbers.  That is the whole point -- two evaluation
seeds used to produce byte-identical scenarios for these families because
skrl's ``Runner.__init__`` reseeds the global torch RNG to the constant pinned
in the agent YAML (``tasks/station_keeping/agents/skrl_ppo_cfg.yaml:1``) after
the env has been built, throwing ``cfg.seed`` away.

UNSEEDED FALLBACK (the training path)
-------------------------------------
``make_scenario_rng`` returns ``None`` when ``cfg.seed is None``, and every
helper here treats ``scenario=None`` as "call ``torch.rand`` exactly the way
this line used to".  So an unseeded env -- a bare ``gym.make`` in a smoke
script, or a trainer that never sets a seed -- keeps running on the historical
code path, bit for bit.  A SEEDED env (every evaluator in ``scripts/`` sets
``env_cfg.seed``, and so does ``scripts/sac_train.py:92``) moves onto the
protocol.  Training therefore also changes which scenarios it sees; it does not
change WHICH DISTRIBUTION it samples them from, so a training run is
statistically the same experiment on a different draw.

SCENARIO HASHES ARE OVER VALUES, NEVER OVER THE KEY
---------------------------------------------------
``stamp_scenario`` hashes the RESOLVED parameters only -- not the eval seed,
not the env index, not the episode index.  Salting the hash with the key would
make two seeds differ by construction, and the certificate auditor
(``scripts/check_scenario_independence.py``, "Scenario-hash channels bypass the
chance guard") would then read every pair as independent even when both seeds
had genuinely drawn the same numbers.  A value-only hash keeps "these two
papers are the same paper" detectable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

# Loaded both as a package member (by the envs) and as a bare top-level module
# (tasks/_shared/sea_state.py is imported that way by its own standalone CPU
# test), so both import spellings have to work -- the idiom already used at
# tasks/_shared/test_sea_state.py:9-12.
try:
    from .scenario_rng import (
        PROTOCOL_NOTES,
        SCENARIO_PROTOCOL_VERSION,
        ScenarioRNG,
        scenario_hash,
    )
except ImportError:  # direct execution
    from scenario_rng import (
        PROTOCOL_NOTES,
        SCENARIO_PROTOCOL_VERSION,
        ScenarioRNG,
        scenario_hash,
    )

# Primitive groups used by the USV envs.  Separate groups mean adding a draw to
# one primitive cannot shift another primitive's numbers (property (d) of the
# protocol).  "route" is task geometry -- the waypoint chain a path family
# follows -- and is deliberately NOT folded into "spawn_pose".
GROUP_SPAWN = "spawn_pose"
GROUP_CURRENT = "current"
GROUP_WAVE = "wave"
GROUP_ROUTE = "route"
GROUP_ACTUATOR = "actuator"

TWO_PI = 2.0 * math.pi  # torch.pi is math.pi (checked), so this is the same
                        # constant the pre-fix lines multiplied by.


def make_scenario_rng(
    cfg: Any,
    num_envs: int,
    device: torch.device | str,
    groups: Sequence[str] | None = None,
) -> ScenarioRNG | None:
    """Build the env's scenario stream, or ``None`` for the unseeded fallback.

    ``None`` is a defined behaviour, not an error: see UNSEEDED FALLBACK in the
    module docstring.  Call this AFTER ``super().__init__`` so ``num_envs`` and
    ``device`` are resolved.
    """
    seed = getattr(cfg, "seed", None)
    if seed is None:
        return None
    return ScenarioRNG(
        num_envs=num_envs,
        device=device,
        eval_seed=int(seed),
        groups=groups,
    )


def unit_uniform(
    scenario: ScenarioRNG | None,
    group: str,
    env_ids: Any,
    device: torch.device | str,
    size: Sequence[int] = (),
) -> torch.Tensor:
    """``U[0, 1)`` shaped exactly like the ``torch.rand`` call it replaces.

    ``size=()`` gives ``(len(env_ids),)`` -- what ``torch.rand(num_resets,
    device=...)`` gave.  ``size=(n,)`` gives ``(len(env_ids), n)`` -- what
    ``torch.rand((num_resets, n), device=...)`` gave.  Row ``r`` belongs to
    ``env_ids[r]`` and comes only from that env's stream, so resetting a subset
    of the envs yields the same numbers as resetting all of them.
    """
    shape = (len(env_ids), *tuple(int(n) for n in size))
    if scenario is None:
        # Historical path, unchanged: the global torch RNG.
        return torch.rand(shape, device=device)
    return scenario.uniform(group, env_ids, 0.0, 1.0, size=size).to(device)


# ------------------------------------------------------------ station keeping --
def station_keeping_current(
    scenario: ScenarioRNG | None,
    env_ids: Any,
    device: torch.device | str,
    speed_min: float,
    speed_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One constant world-frame current per reset env: ``(speeds, directions)``.

    Before (tasks/station_keeping/station_keeping_env.py:367-373)::

        speeds = self.physics_cfg.current_speed_min + torch.rand(
            len(env_ids), device=self.device
        ) * (current_speed_max - current_speed_min)
        directions = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi

    After: the two ``torch.rand`` calls become draws on the ``current`` stream;
    the multiply-and-shift below is copied from those lines unchanged, so the
    speed stays uniform on ``[min, max)`` and the direction on ``[0, 2*pi)``.
    """
    unit_speed = unit_uniform(scenario, GROUP_CURRENT, env_ids, device)
    unit_direction = unit_uniform(scenario, GROUP_CURRENT, env_ids, device)
    speeds = speed_min + unit_speed * (speed_max - speed_min)
    directions = unit_direction * TWO_PI
    return speeds, directions


def station_keeping_spawn(
    scenario: ScenarioRNG | None,
    env_ids: Any,
    device: torch.device | str,
    min_distance: float,
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Polar spawn: ``(distances, spawn_angles, headings)``.

    Before (tasks/station_keeping/station_keeping_env.py:684-688)::

        distances = self.cfg.min_spawn_distance + torch.rand(
            num_resets, device=self.device
        ) * (self.cfg.max_spawn_distance - self.cfg.min_spawn_distance)
        spawn_angles = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        headings = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi

    After: three draws on the ``spawn_pose`` stream, same arithmetic, same
    draw ORDER (distance, then angle, then heading).
    """
    unit_distance = unit_uniform(scenario, GROUP_SPAWN, env_ids, device)
    unit_angle = unit_uniform(scenario, GROUP_SPAWN, env_ids, device)
    unit_heading = unit_uniform(scenario, GROUP_SPAWN, env_ids, device)
    distances = min_distance + unit_distance * (max_distance - min_distance)
    spawn_angles = unit_angle * TWO_PI
    headings = unit_heading * TWO_PI
    return distances, spawn_angles, headings


# -------------------------------------------------------------- path families --
def path_following_route(
    scenario: ScenarioRNG | None,
    env_ids: Any,
    device: torch.device | str,
    num_waypoints: int,
    segment_length_min: float,
    segment_length_max: float,
    max_heading_change: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Waypoint chain: ``(segment_lengths, first_headings, heading_changes)``.

    Before (tasks/path_following/path_following_env.py:641-658)::

        self.segment_lengths[env_ids] = self.cfg.segment_length_min + torch.rand(
            (num_resets, self.cfg.num_waypoints), device=self.device
        ) * (self.cfg.segment_length_max - self.cfg.segment_length_min)
        headings[:, 0] = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        heading_changes = (
            torch.rand((num_resets, self.cfg.num_waypoints - 1), device=self.device)
            * 2.0 - 1.0
        ) * max_change

    After: three draws on the ``route`` stream in that same order.
    ``max_heading_change`` is passed through untouched (the env hands over the
    0-dim ``torch.deg2rad`` tensor it already had), so the change stays uniform
    on ``[-max, +max)``.
    """
    unit_segments = unit_uniform(
        scenario, GROUP_ROUTE, env_ids, device, size=(num_waypoints,)
    )
    unit_first = unit_uniform(scenario, GROUP_ROUTE, env_ids, device)
    unit_changes = unit_uniform(
        scenario, GROUP_ROUTE, env_ids, device, size=(max(num_waypoints - 1, 0),)
    )
    segment_lengths = segment_length_min + unit_segments * (
        segment_length_max - segment_length_min
    )
    first_headings = unit_first * TWO_PI
    heading_changes = (unit_changes * 2.0 - 1.0) * max_heading_change
    return segment_lengths, first_headings, heading_changes


def spawn_heading(
    scenario: ScenarioRNG | None,
    env_ids: Any,
    device: torch.device | str,
) -> torch.Tensor:
    """Uniform bow heading on ``[0, 2*pi)``, one per reset env.

    Before (tasks/path_following/path_following_env.py:671 and
    tasks/path_hazard/path_hazard_env.py:981, identical lines)::

        spawn_headings = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
    """
    return unit_uniform(scenario, GROUP_SPAWN, env_ids, device) * TWO_PI


# ----------------------------------------------------------------- recording --
def stamp_scenario(
    env_ids: Any,
    groups: Mapping[str, Mapping[str, Any]],
) -> list[tuple[int, dict[str, dict[str, Any]], dict[str, str]]]:
    """Turn per-group batched draws into per-env resolved params and hashes.

    ``groups`` maps a primitive group name to a mapping of parameter name ->
    tensor (or sequence) whose ROW ``r`` belongs to ``env_ids[r]``.  Returns one
    ``(env_index, resolved_params, hashes)`` triple per reset env, where
    ``hashes`` carries a digest per group plus a whole-scenario digest under the
    key ``"scenario"`` -- the shape
    ``scripts/check_scenario_independence.py`` reads from
    ``record["scenario_hashes"]``.

    Each tensor is moved to the host ONCE for the whole batch, so a reset costs
    one sync per parameter rather than one per (env, parameter).
    """
    if torch.is_tensor(env_ids):
        ids = [int(value) for value in env_ids.detach().flatten().cpu().tolist()]
    else:
        ids = [int(value) for value in env_ids]

    columns: dict[str, dict[str, list]] = {}
    for group, params in groups.items():
        columns[group] = {}
        for name, value in params.items():
            rows = (
                value.detach().cpu().tolist()
                if torch.is_tensor(value)
                else list(value)
            )
            if len(rows) != len(ids):
                raise ValueError(
                    f"scenario parameter {group}.{name} has {len(rows)} rows "
                    f"for {len(ids)} envs"
                )
            columns[group][name] = rows

    out = []
    for row, env_index in enumerate(ids):
        resolved = {
            group: {name: values[row] for name, values in params.items()}
            for group, params in columns.items()
        }
        # Hash VALUES only -- never the (seed, env, episode) key.  See the
        # module docstring: a key-salted hash could never expose two seeds that
        # drew the same paper, which is the defect this plumbing exists to make
        # visible.
        hashes = {
            group: scenario_hash(params) for group, params in resolved.items()
        }
        hashes["scenario"] = scenario_hash(resolved)
        out.append((env_index, resolved, hashes))
    return out


def scenario_protocol_header() -> dict[str, Any]:
    """Certificate header block describing which protocol produced the run.

    Deliberately free of run-specific values -- no eval seed, no checkpoint.
    ``scripts/check_scenario_independence.py:736-738`` warns whenever the two
    certificates of a pair disagree on this block, and the two members of a
    pair differ precisely in their evaluation seed, so putting the seed here
    would fire that warning on every healthy pair. The evaluation seed lives at
    the certificate top level instead (``seed`` and ``eval_seed``).
    """
    return {
        "version": SCENARIO_PROTOCOL_VERSION,
        "hash": "blake2b",
        "note": PROTOCOL_NOTES[SCENARIO_PROTOCOL_VERSION],
    }


# ------------------------------------------------- certificate stamping --
# Certificate value for a run whose env never built a ScenarioRNG: the episode
# scenarios of that run do NOT descend from the evaluation seed in a
# controller-independent way, and the certificate has to say so rather than
# stay silent.  A missing key is ambiguous (old certificate?  unstamped
# writer?  off protocol?); an explicit marker is not.  It is a plain string,
# never the header dict, so no auditor can read it as compliance --
# ``scripts/check_scenario_independence.py:257`` stores the field opaquely and
# :736-738 only compares and formats it, so the marker rides through unchanged
# and a pair with one side on protocol and one side off raises the
# "scenario_protocol differs" warning it should.
#
# The literal is duplicated at ``scripts/eval_v6_frozen.py:91``, which is the
# writer this module's helpers were factored out of; the two are pinned equal
# by ``scripts/test_certificate_scenario_stamp.py``.
SCENARIO_PROTOCOL_OFF = "off-protocol"


def scenario_protocol_stamp(base: Any) -> Any:
    """The certificate's ``scenario_protocol`` value for the env ``base``.

    Transcribed from ``scripts/eval_v6_frozen.py:154-158`` so every evaluator
    stamps the same way.  Two ways an env may be off protocol: the task family
    was never migrated (no ``_scenario`` attribute at all), or it was migrated
    but built unseeded, in which case ``make_scenario_rng`` returned ``None``
    and every draw fell back to the historical global-RNG line.  Both are
    off-protocol and both are caught by this one guard.

    Returns the header block when the env carries the protocol object and the
    ``SCENARIO_PROTOCOL_OFF`` literal when it does not.  NEVER unconditionally
    the header: stamping compliance on a task whose scenarios were never on
    the protocol is the one failure mode a certificate must not have.
    """
    return (
        scenario_protocol_header()
        if getattr(base, "_scenario", None) is not None
        else SCENARIO_PROTOCOL_OFF
    )


def episode_scenario_hashes_for(
    base: Any,
    env_index: Any,
) -> dict[str, str] | None:
    """Digests of the scenario the episode env ``env_index`` just FINISHED ran.

    Transcribed from ``scripts/eval_v6_frozen.py:213-215``.  The env latches
    ``episode_scenario_hashes[env]`` at reset exactly like
    ``episode_min_clearance`` (e.g. ``tasks/path_hazard/path_hazard_env.py:952
    -954``), so this must be read for the episode that just ended, in the same
    place the other per-episode stats are read.

    Returns ``None`` -- meaning "write no ``scenario_hashes`` field" -- in the
    three cases that are not a stamped episode: a family not yet on the
    scenario protocol (no attribute), an env index the container does not
    carry, and an env that has not completed its first episode yet (the latch
    is still the empty dict every env starts with).  A copy is returned so a
    later reset cannot mutate a record already appended to the certificate.
    """
    latched = getattr(base, "episode_scenario_hashes", None)
    if latched is None:
        return None
    try:
        entry = latched[env_index]
    except (IndexError, KeyError, TypeError):
        return None
    if not entry:
        return None
    return dict(entry)


def scenario_protocol_notice(task: Any, protocol: Any) -> str | None:
    """The stdout warning for an off-protocol run, or ``None`` when on it.

    Word for word the message ``scripts/eval_v6_frozen.py:159-162`` prints, so
    an off-protocol run of ANY evaluator says the same thing in its log.
    """
    if protocol != SCENARIO_PROTOCOL_OFF:
        return None
    return (
        f"WARNING: {task} is not on the scenario protocol; this certificate "
        f"is stamped scenario_protocol={SCENARIO_PROTOCOL_OFF!r} and its "
        "episodes are NOT paired across evaluation seeds"
    )


__all__ = [
    "GROUP_ACTUATOR",
    "GROUP_CURRENT",
    "GROUP_ROUTE",
    "GROUP_SPAWN",
    "GROUP_WAVE",
    "SCENARIO_PROTOCOL_OFF",
    "TWO_PI",
    "episode_scenario_hashes_for",
    "make_scenario_rng",
    "path_following_route",
    "scenario_protocol_header",
    "scenario_protocol_notice",
    "scenario_protocol_stamp",
    "spawn_heading",
    "stamp_scenario",
    "station_keeping_current",
    "station_keeping_spawn",
    "unit_uniform",
]
