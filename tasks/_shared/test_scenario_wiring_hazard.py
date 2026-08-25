# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the ENV WIRING of hazard_nav and harbor_mission.

``tasks/_shared/test_scenario_rng.py`` proves the stream algebra.
``tasks/_shared/test_scenario_wiring.py`` proves the reset paths of the
station-keeping / path families were plumbed into it.  This file proves the
same for the two families that carry most of the benchmark: hazard_nav (38 of
62 registered gym ids, 199 of 242 certificates) and harbor_mission.

WHY THESE TWO NEED THEIR OWN FILE
---------------------------------
The other families were broken because they drew from the GLOBAL torch RNG,
which skrl reseeds to a constant.  These two were never on the global RNG: they
owned ONE numpy ``Generator`` seeded from ``cfg.seed``
(``tasks/hazard_nav/hazard_nav_env.py:116-118``,
``tasks/harbor_mission/harbor_mission_env.py:63-65``) and advanced it once per
reset.  That is seed-derived and still not controller-independent, because both
families terminate episodes EARLY, so two controllers consume the shared stream
at different rates and are handed different layouts for the same ``(eval seed,
env, episode)`` slot.

Measured, at one evaluation seed, on the crossing task: a sampling-MPC, a PPO
policy and a classical planner+PID were each certified over 128 episodes and
compared on ``route_geodesic_m`` and ``d0_m`` -- two per-episode certificate
fields that depend on the LAYOUT alone and not on the controller.  The layouts
agreed on 53.8% / 56.9% / 55.6% of shared ``(env, episode)`` slots; the multiset
of layouts overlapped on only 105 / 114 / 110 of 128.  Every paired statistic
computed from those certificates was computed on unpaired data.

``test_hazard_nav_survives_two_termination_patterns`` and its harbor twin are
that measurement, reproduced on CPU: the same two schedules are run once on the
protocol (must agree on 100% of shared slots) and once on the pre-fix shared
generator (must NOT), so the test has teeth rather than passing vacuously.

Isaac Sim is not installed here and the env modules import ``isaaclab`` at
module scope, so the env classes cannot be imported.  That is why
``tasks/_shared/scenario_draws_hazard.py`` exists: the reset-path draw logic
lives there, the envs call it, and so does this file.  ``_reset_hazard_nav``
and ``_reset_harbor_mission`` below are line-for-line transcriptions of the
draw ORDER in each env's ``_reset_idx``, and they call the REAL layout samplers
from ``hazard_geometry`` / ``harbor_geometry`` (both deliberately Isaac-free),
so what is compared here is the actual geometry a certificate would record.

Run: python tasks/_shared/test_scenario_wiring_hazard.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from .scenario_draws import make_scenario_rng, stamp_scenario
    from .scenario_draws_hazard import (
        ACTUATOR_GROUPS,
        GROUP_LAYOUT,
        GROUP_MASS_SCALE,
        GROUP_ROTATION,
        GROUP_SPAWN,
        GROUP_THRUST_IMBALANCE,
        TWO_PI,
        actuator_choice_indices,
        actuator_scenario_entry,
        episode_layout_rng,
        layout_rotation_angle,
        spawn_jitter_offsets,
    )
except ImportError:  # direct execution
    from scenario_draws import make_scenario_rng, stamp_scenario
    from scenario_draws_hazard import (
        ACTUATOR_GROUPS,
        GROUP_LAYOUT,
        GROUP_MASS_SCALE,
        GROUP_ROTATION,
        GROUP_SPAWN,
        GROUP_THRUST_IMBALANCE,
        TWO_PI,
        actuator_choice_indices,
        actuator_scenario_entry,
        episode_layout_rng,
        layout_rotation_angle,
        spawn_jitter_offsets,
    )

# The real samplers. Imported by directory rather than as package members
# because ``tasks/hazard_nav/__init__.py`` registers gym ids and would pull
# isaaclab -- the same dodge tasks/harbor_mission/harbor_geometry.py:27-39 uses
# for its own standalone test.
_TASKS = Path(__file__).resolve().parent.parent
for _directory in (_TASKS / "hazard_nav", _TASKS / "harbor_mission"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
from hazard_geometry import sample_layout  # noqa: E402
from harbor_geometry import sample_harbor_route  # noqa: E402


DEVICE = torch.device("cpu")
NUM_ENVS = 4

# Values from the shipped cfgs so the test exercises real numbers:
# hazard_nav_env_cfg.py:186 (layout_max_attempts), :179 (layout_mode scatter),
# harbor_mission_env_cfg.py:179-180, :336 (HarborDockPhase jitter).
LEVEL = 0
LAYOUT_MAX_ATTEMPTS = 20
HARBOR_MAX_ATTEMPTS = 50
HARBOR_HAZARD_MAX_ATTEMPTS = 20
JITTER_M = 2.0

# A representative pair of knob tables. The values are irrelevant to the
# properties under test; what matters is that a choice INDEX is drawn per env.
THRUST_IMBALANCE_CHOICES = (-0.15, -0.05, 0.0, 0.05, 0.15)
MASS_SCALE_CHOICES = (0.85, 1.0, 1.15)

# Two reset schedules = two simulated termination patterns. Each inner list is
# the set of envs that finished together on one step, in the order a controller
# happened to end episodes. Controller A ends them in pairs; controller B ends
# them one at a time and then all four at once. On a keyed protocol env i's k-th
# episode is the same scenario in both; on one shared advancing generator it
# cannot be, because B has consumed a different number of draws by the time it
# reaches that slot.
SCHEDULE_A = [[0, 1, 2, 3], [1, 3], [0, 2], [1, 3], [0, 2]]
SCHEDULE_B = [[0, 1, 2, 3], [0], [2], [1], [3], [0, 1, 2, 3]]


class _Cfg:
    """The single attribute make_scenario_rng reads off an env cfg."""

    def __init__(self, seed):
        self.seed = seed


def _fallback(seed=0):
    """The env's own generator: np.random.default_rng(layout_seed).

    ``layout_seed`` defaults to the constant 0 in both families
    (hazard_nav_env_cfg.py:187, harbor_mission_env_cfg.py:181), which is the
    unseeded behaviour these tests pin rather than change.
    """
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Reset-path transcriptions. The draw ORDER mirrors each env's _reset_idx.
# ---------------------------------------------------------------------------
def _reset_hazard_nav(scenario, fallback_rng, env_ids, knobs=None):
    """hazard_nav_env.py _reset_idx.

    Order: ``ScenarioRNG.reset_idx`` -> the actuator knobs in cfg order ->
    per row (layout sampler, then global rotation).  One dict per resetting env,
    row r belonging to ``env_ids[r]``.
    """
    ids = torch.as_tensor(env_ids, dtype=torch.long)
    if scenario is not None:
        scenario.reset_idx(ids)
    indices = [int(value) for value in env_ids]

    actuator = {}
    for group, table in (knobs or {}).items():
        chosen = actuator_choice_indices(
            scenario, fallback_rng, group, indices, len(table)
        )
        actuator[group] = np.asarray(table, dtype=np.float32)[chosen]

    rows = []
    for row, env_index in enumerate(indices):
        layout = sample_layout(
            LEVEL,
            rng=episode_layout_rng(scenario, fallback_rng, env_index),
            max_attempts=LAYOUT_MAX_ATTEMPTS,
        )
        angle = layout_rotation_angle(scenario, fallback_rng, env_index)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        drawn = {
            "start_m": layout.start @ rotation.T,
            "goal_m": layout.goal @ rotation.T,
            "obstacle_centers_m": layout.centers @ rotation.T,
            "obstacle_radii_m": layout.radii,
            # The two purely-layout certificate fields the cross-controller
            # audit compared (hazard_nav_env.py:1890-1891 writes them exactly
            # this way on the non-geodesic branch).
            "route_geodesic_m": float(layout.geodesic_length),
            "d0_m": float(np.linalg.norm(layout.goal - layout.start)),
            "rotation_rad": angle,
        }
        for group, values in actuator.items():
            drawn[group] = float(values[row])
        rows.append(drawn)
    return rows


def _reset_harbor_mission(scenario, fallback_rng, env_ids, jitter_m=0.0):
    """harbor_mission_env.py _reset_idx.

    Order: ``ScenarioRNG.reset_idx`` -> per row the route sampler -> the staged
    spawn jitter for the whole batch (which is where the env draws it,
    harbor_mission_env.py:1296-1308).
    """
    ids = torch.as_tensor(env_ids, dtype=torch.long)
    if scenario is not None:
        scenario.reset_idx(ids)
    indices = [int(value) for value in env_ids]

    rows = []
    for env_index in indices:
        route = sample_harbor_route(
            episode_layout_rng(scenario, fallback_rng, env_index),
            max_attempts=HARBOR_MAX_ATTEMPTS,
            hazard_max_attempts=HARBOR_HAZARD_MAX_ATTEMPTS,
        )
        rows.append(
            {
                "gate_midpoints_m": np.asarray(route.gate_midpoints),
                "berth_point_m": np.asarray(route.berth_point),
                "field_exit_m": np.asarray(route.field_exit),
                "obstacle_centers_m": np.asarray(route.obstacle_centers),
                "obstacle_radii_m": np.asarray(route.obstacle_radii),
                "field_geodesic_m": float(route.field_geodesic_length),
            }
        )

    offsets = spawn_jitter_offsets(scenario, fallback_rng, indices, jitter_m)
    for row, drawn in enumerate(rows):
        drawn["spawn_jitter_m"] = offsets[row]
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _episodes(reset, scenario, fallback_rng, schedule, **kwargs):
    """Run a reset schedule; key every drawn scenario by (env, episode).

    ``(env_index, episode_index)`` is what two controllers must AGREE on, and
    the per-env episode counter here is maintained the way the env's own reset
    count is: it advances once per reset of that env, regardless of when.
    """
    seen = {}
    counters = {index: -1 for index in range(NUM_ENVS)}
    for env_ids in schedule:
        rows = reset(scenario, fallback_rng, env_ids, **kwargs)
        for row, env_index in enumerate(env_ids):
            counters[env_index] += 1
            seen[(env_index, counters[env_index])] = rows[row]
    return seen


def _same(left, right):
    return all(np.array_equal(left[name], right[name]) for name in left)


def _agreement(seen_a, seen_b):
    """Fraction of shared (env, episode) slots whose scenarios are identical.

    This is the statistic the certificate audit reported as 53.8% / 56.9% /
    55.6% across three controllers; on the protocol it must be exactly 1.0.
    """
    shared = sorted(set(seen_a) & set(seen_b))
    assert shared, "the two schedules share no (env, episode) slot"
    agreed = sum(1 for key in shared if _same(seen_a[key], seen_b[key]))
    return agreed / len(shared), len(shared)


# ---------------------------------------------------------------------------
# (a) two eval seeds must be two different exam papers
# ---------------------------------------------------------------------------
def test_hazard_nav_two_eval_seeds_are_different_layouts():
    all_ids = list(range(NUM_ENVS))
    drawn = {}
    for seed in (123, 42):
        drawn[seed] = _reset_hazard_nav(
            make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE), _fallback(), all_ids
        )
    for row in range(NUM_ENVS):
        assert not _same(drawn[123][row], drawn[42][row]), (
            f"hazard_nav env {row} drew the identical layout under eval seeds "
            "123 and 42"
        )
    # And specifically on the two fields the certificate carries.
    for field in ("route_geodesic_m", "d0_m"):
        left = [row[field] for row in drawn[123]]
        right = [row[field] for row in drawn[42]]
        assert left != right, f"hazard_nav {field} is seed-independent"


def test_harbor_mission_two_eval_seeds_are_different_routes():
    all_ids = list(range(NUM_ENVS))
    drawn = {}
    for seed in (123, 42):
        drawn[seed] = _reset_harbor_mission(
            make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE),
            _fallback(),
            all_ids,
            jitter_m=JITTER_M,
        )
    for row in range(NUM_ENVS):
        assert not _same(drawn[123][row], drawn[42][row]), (
            f"harbor_mission env {row} drew the identical route under eval "
            "seeds 123 and 42"
        )


# ---------------------------------------------------------------------------
# (b) THE test: one eval seed, two termination patterns, same layouts
# ---------------------------------------------------------------------------
def test_hazard_nav_survives_two_termination_patterns():
    """The property the measured 53.8% / 56.9% / 55.6% failure violates.

    Two controllers at the same eval seed end episodes at different times and
    in different groupings.  Env i's k-th episode must be the same layout, the
    same rotation and the same actuator draw in both.
    """
    seed = 2026
    knobs = {
        GROUP_THRUST_IMBALANCE: THRUST_IMBALANCE_CHOICES,
        GROUP_MASS_SCALE: MASS_SCALE_CHOICES,
    }
    seen_a = _episodes(
        _reset_hazard_nav,
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE),
        _fallback(),
        SCHEDULE_A,
        knobs=knobs,
    )
    seen_b = _episodes(
        _reset_hazard_nav,
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE),
        _fallback(),
        SCHEDULE_B,
        knobs=knobs,
    )
    fraction, shared = _agreement(seen_a, seen_b)
    assert shared >= 8, f"only {shared} shared episodes to compare"
    for key in sorted(set(seen_a) & set(seen_b)):
        for name in seen_a[key]:
            assert np.array_equal(seen_a[key][name], seen_b[key][name]), (
                f"env {key[0]} episode {key[1]}: {name} differs between two "
                "controllers at the same eval seed"
            )
    assert fraction == 1.0, fraction

    # NEGATIVE CONTROL -- without it this test would pass on the broken code.
    # One shared generator advanced per reset is exactly the pre-fix path, and
    # it is what produced the ~55% agreement in the real certificates.
    pre_fix_a = _episodes(
        _reset_hazard_nav, None, _fallback(seed), SCHEDULE_A, knobs=knobs
    )
    pre_fix_b = _episodes(
        _reset_hazard_nav, None, _fallback(seed), SCHEDULE_B, knobs=knobs
    )
    pre_fix_fraction, _ = _agreement(pre_fix_a, pre_fix_b)
    assert pre_fix_fraction < 1.0, (
        "the pre-fix shared-generator path agreed on every shared slot, so "
        "this test cannot tell a keyed stream from a sequential one"
    )


def test_harbor_mission_survives_two_termination_patterns():
    """Same property, same negative control, for the ordered harbor route."""
    seed = 2026
    seen_a = _episodes(
        _reset_harbor_mission,
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE),
        _fallback(),
        SCHEDULE_A,
        jitter_m=JITTER_M,
    )
    seen_b = _episodes(
        _reset_harbor_mission,
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE),
        _fallback(),
        SCHEDULE_B,
        jitter_m=JITTER_M,
    )
    fraction, shared = _agreement(seen_a, seen_b)
    assert shared >= 8, f"only {shared} shared episodes to compare"
    for key in sorted(set(seen_a) & set(seen_b)):
        for name in seen_a[key]:
            assert np.array_equal(seen_a[key][name], seen_b[key][name]), (
                f"env {key[0]} episode {key[1]}: {name} differs between two "
                "controllers at the same eval seed"
            )
    assert fraction == 1.0, fraction

    pre_fix_a = _episodes(
        _reset_harbor_mission, None, _fallback(seed), SCHEDULE_A,
        jitter_m=JITTER_M,
    )
    pre_fix_b = _episodes(
        _reset_harbor_mission, None, _fallback(seed), SCHEDULE_B,
        jitter_m=JITTER_M,
    )
    pre_fix_fraction, _ = _agreement(pre_fix_a, pre_fix_b)
    assert pre_fix_fraction < 1.0, (
        "the pre-fix shared-generator path agreed on every shared slot, so "
        "this test cannot tell a keyed stream from a sequential one"
    )


def test_same_eval_seed_replays_exactly():
    """Two runs of the same seed are the same paper -- reproducibility."""
    all_ids = list(range(NUM_ENVS))
    first = _reset_hazard_nav(
        make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE), _fallback(), all_ids
    )
    second = _reset_hazard_nav(
        make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE), _fallback(), all_ids
    )
    assert all(_same(a, b) for a, b in zip(first, second)), (
        "hazard_nav did not replay under the same eval seed"
    )

    first = _reset_harbor_mission(
        make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE), _fallback(), all_ids,
        jitter_m=JITTER_M,
    )
    second = _reset_harbor_mission(
        make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE), _fallback(), all_ids,
        jitter_m=JITTER_M,
    )
    assert all(_same(a, b) for a, b in zip(first, second)), (
        "harbor_mission did not replay under the same eval seed"
    )


# ---------------------------------------------------------------------------
# (c) per-env isolation
# ---------------------------------------------------------------------------
def test_hazard_nav_per_env_isolation():
    """Resetting one env must not disturb another env's layout."""
    seed = 99
    together = _reset_hazard_nav(
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE), _fallback(),
        [0, 1, 2, 3],
    )

    alone_scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
    fallback = _fallback()
    # Env 2 resets by itself, after envs 0/1/3 have already reset.
    _reset_hazard_nav(alone_scenario, fallback, [0, 1, 3])
    single = _reset_hazard_nav(alone_scenario, fallback, [2])
    assert _same(together[2], single[0]), (
        "env 2's layout depends on which other envs reset alongside it"
    )

    # And two different envs must not be handed the same scenario.
    assert not _same(together[0], together[1]), (
        "env 0 and env 1 drew the same layout"
    )


def test_harbor_mission_per_env_isolation():
    seed = 99
    together = _reset_harbor_mission(
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE), _fallback(),
        [0, 1, 2, 3], jitter_m=JITTER_M,
    )

    alone_scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
    fallback = _fallback()
    _reset_harbor_mission(alone_scenario, fallback, [0, 1, 3], jitter_m=JITTER_M)
    single = _reset_harbor_mission(
        alone_scenario, fallback, [2], jitter_m=JITTER_M
    )
    assert _same(together[2], single[0]), (
        "env 2's route depends on which other envs reset alongside it"
    )
    assert not _same(together[0], together[1]), (
        "env 0 and env 1 drew the same route"
    )


# ---------------------------------------------------------------------------
# (d) substream isolation between primitives
# ---------------------------------------------------------------------------
def test_primitives_do_not_shift_each_other():
    """A knob only one variant randomises must not move the layout.

    This is why the five actuator knobs are five groups and not five ordered
    draws off one stream: scripts/eval_imbalance.py sweeps them one at a time,
    and a shared stream would re-draw the others whenever one was enabled.
    """
    ids = list(range(NUM_ENVS))

    baseline = _reset_hazard_nav(
        make_scenario_rng(_Cfg(404), NUM_ENVS, DEVICE), _fallback(), ids
    )
    with_knobs = _reset_hazard_nav(
        make_scenario_rng(_Cfg(404), NUM_ENVS, DEVICE), _fallback(), ids,
        knobs={
            GROUP_THRUST_IMBALANCE: THRUST_IMBALANCE_CHOICES,
            GROUP_MASS_SCALE: MASS_SCALE_CHOICES,
        },
    )
    for row in range(NUM_ENVS):
        for name in baseline[row]:
            assert np.array_equal(
                baseline[row][name], with_knobs[row][name]
            ), f"enabling the actuator knobs shifted {name} on env {row}"

    # One knob alone must give the same values as that knob among several.
    one = _reset_hazard_nav(
        make_scenario_rng(_Cfg(404), NUM_ENVS, DEVICE), _fallback(), ids,
        knobs={GROUP_MASS_SCALE: MASS_SCALE_CHOICES},
    )
    for row in range(NUM_ENVS):
        assert one[row][GROUP_MASS_SCALE] == with_knobs[row][GROUP_MASS_SCALE], (
            "mass_scale moved when thrust_imbalance was also enabled; the two "
            "knobs are sharing a stream"
        )

    # Every group name is distinct, or "its own stream" is a fiction.
    assert len(set(ACTUATOR_GROUPS)) == len(ACTUATOR_GROUPS)
    assert GROUP_LAYOUT not in ACTUATOR_GROUPS
    assert len({GROUP_LAYOUT, GROUP_ROTATION, GROUP_SPAWN}) == 3


def test_rotation_is_not_a_draw_off_the_layout_stream():
    """The rotation must not move when the layout sampler burns more draws.

    The Suite S branch (hazard_nav_env.py:1790-1795) consumes NO layout draws while
    every other layout_mode consumes many, so a rotation taken off the layout
    stream would differ between them for the same key.
    """
    ids = list(range(NUM_ENVS))
    scenario = make_scenario_rng(_Cfg(808), NUM_ENVS, DEVICE)
    scenario.reset_idx(torch.as_tensor(ids, dtype=torch.long))
    without_layout = [
        layout_rotation_angle(scenario, None, index) for index in ids
    ]

    scenario = make_scenario_rng(_Cfg(808), NUM_ENVS, DEVICE)
    drawn = _reset_hazard_nav(scenario, _fallback(), ids)
    with_layout = [row["rotation_rad"] for row in drawn]
    assert without_layout == with_layout, (
        "the rotation angle moved when a layout was sampled first, so it is "
        "riding the layout stream"
    )


# ---------------------------------------------------------------------------
# Unseeded fallback and preserved ranges
# ---------------------------------------------------------------------------
def test_unseeded_fallback_is_the_historical_line():
    """cfg.seed None keeps the pre-fix code path, bit for bit.

    Both families fall back to cfg.layout_seed, which defaults to the CONSTANT
    0, so an unseeded run replays one fixed sequence. That is the behaviour
    being preserved -- not endorsed; it is documented at
    hazard_nav_env.py:88-96 and harbor_mission_env.py:44-51.
    """
    assert make_scenario_rng(_Cfg(None), NUM_ENVS, DEVICE) is None

    ids = list(range(NUM_ENVS))

    # The layout generator is the env's own generator, unwrapped.
    shared = _fallback(5)
    assert episode_layout_rng(None, shared, 3) is shared
    assert episode_layout_rng(None, shared, 3, GROUP_ROTATION) is shared

    # The actuator draw is the ORIGINAL batched integers() call.
    expected = _fallback(5).integers(len(THRUST_IMBALANCE_CHOICES), size=NUM_ENVS)
    actual = actuator_choice_indices(
        None, _fallback(5), GROUP_THRUST_IMBALANCE, ids,
        len(THRUST_IMBALANCE_CHOICES),
    )
    assert np.array_equal(actual, expected), (
        "the unseeded actuator draw is no longer the batched "
        "self._layout_rng.integers(n, size=num_resets)"
    )

    # The rotation is the original scalar uniform.
    rng = _fallback(5)
    expected_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    assert layout_rotation_angle(None, _fallback(5), 0) == expected_angle

    # The jitter is the original pair of batched uniforms, in that order.
    rng = _fallback(5)
    angles = rng.uniform(0.0, 2.0 * math.pi, NUM_ENVS)
    radii = JITTER_M * np.sqrt(rng.uniform(0.0, 1.0, NUM_ENVS))
    expected_offsets = np.zeros((NUM_ENVS, 2), dtype=np.float32)
    expected_offsets[:, 0] = radii * np.cos(angles)
    expected_offsets[:, 1] = radii * np.sin(angles)
    assert np.array_equal(
        spawn_jitter_offsets(None, _fallback(5), ids, JITTER_M),
        expected_offsets,
    ), "the unseeded spawn jitter is no longer the historical batched draw"

    # And an unseeded env still RUNS: the whole reset path works with None.
    rows = _reset_hazard_nav(None, _fallback(), ids)
    assert len(rows) == NUM_ENVS and math.isfinite(rows[0]["d0_m"])
    rows = _reset_harbor_mission(None, _fallback(), ids, jitter_m=JITTER_M)
    assert len(rows) == NUM_ENVS and math.isfinite(rows[0]["field_geodesic_m"])

# The PRE-migration reset paths, transcribed verbatim from the code these tests
# replaced, so "the unseeded path is bit-identical" is a comparison against the
# real thing and not against a restatement of the new code.
def _pre_fix_reset_hazard_nav(rng, env_ids, knobs=None):
    num_resets = len(env_ids)
    actuator = {}
    for group, table in (knobs or {}).items():
        choice_indices = rng.integers(len(table), size=num_resets)
        actuator[group] = np.asarray(table, dtype=np.float32)[choice_indices]
    rows = []
    for row in range(num_resets):
        layout = sample_layout(LEVEL, rng=rng, max_attempts=LAYOUT_MAX_ATTEMPTS)
        goal_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        cosine, sine = math.cos(goal_angle), math.sin(goal_angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        drawn = {
            "start_m": layout.start @ rotation.T,
            "goal_m": layout.goal @ rotation.T,
            "obstacle_centers_m": layout.centers @ rotation.T,
            "obstacle_radii_m": layout.radii,
            "route_geodesic_m": float(layout.geodesic_length),
            "d0_m": float(np.linalg.norm(layout.goal - layout.start)),
            "rotation_rad": goal_angle,
        }
        for group, values in actuator.items():
            drawn[group] = float(values[row])
        rows.append(drawn)
    return rows


def _pre_fix_reset_harbor_mission(rng, env_ids, jitter_m=0.0):
    num_resets = len(env_ids)
    rows = []
    for _row in range(num_resets):
        route = sample_harbor_route(
            rng,
            max_attempts=HARBOR_MAX_ATTEMPTS,
            hazard_max_attempts=HARBOR_HAZARD_MAX_ATTEMPTS,
        )
        rows.append(
            {
                "gate_midpoints_m": np.asarray(route.gate_midpoints),
                "berth_point_m": np.asarray(route.berth_point),
                "field_exit_m": np.asarray(route.field_exit),
                "obstacle_centers_m": np.asarray(route.obstacle_centers),
                "obstacle_radii_m": np.asarray(route.obstacle_radii),
                "field_geodesic_m": float(route.field_geodesic_length),
            }
        )
    offsets = np.zeros((num_resets, 2), dtype=np.float32)
    if jitter_m > 0.0:
        angles = rng.uniform(0.0, 2.0 * math.pi, num_resets)
        radii_j = jitter_m * np.sqrt(rng.uniform(0.0, 1.0, num_resets))
        offsets[:, 0] = radii_j * np.cos(angles)
        offsets[:, 1] = radii_j * np.sin(angles)
    for row, drawn in enumerate(rows):
        drawn["spawn_jitter_m"] = offsets[row]
    return rows


def test_unseeded_reset_sequence_is_bit_identical_to_the_pre_fix_code():
    """Not just each helper -- the whole multi-reset SEQUENCE, and the RNG state.

    A helper can be individually faithful and the reset still drift, if the
    migration reordered the calls or changed how many values one of them
    consumes. Comparing the generator's bit_generator state afterwards catches
    exactly that: the states are equal only if the same draws were taken, in
    the same order, in the same shapes.
    """
    knobs = {
        GROUP_THRUST_IMBALANCE: THRUST_IMBALANCE_CHOICES,
        GROUP_MASS_SCALE: MASS_SCALE_CHOICES,
    }
    new_rng, old_rng = _fallback(4242), _fallback(4242)
    for env_ids in SCHEDULE_A:
        new_rows = _reset_hazard_nav(None, new_rng, env_ids, knobs=knobs)
        old_rows = _pre_fix_reset_hazard_nav(old_rng, env_ids, knobs=knobs)
        for new_row, old_row in zip(new_rows, old_rows):
            assert _same(new_row, old_row), (
                "the unseeded hazard_nav reset drifted from the pre-fix code"
            )
    assert new_rng.bit_generator.state == old_rng.bit_generator.state, (
        "the unseeded hazard_nav reset consumed a different number of draws"
    )

    new_rng, old_rng = _fallback(4242), _fallback(4242)
    for env_ids in SCHEDULE_A:
        new_rows = _reset_harbor_mission(
            None, new_rng, env_ids, jitter_m=JITTER_M
        )
        old_rows = _pre_fix_reset_harbor_mission(
            old_rng, env_ids, jitter_m=JITTER_M
        )
        for new_row, old_row in zip(new_rows, old_rows):
            assert _same(new_row, old_row), (
                "the unseeded harbor_mission reset drifted from the pre-fix "
                "code"
            )
    assert new_rng.bit_generator.state == old_rng.bit_generator.state, (
        "the unseeded harbor_mission reset consumed a different number of draws"
    )



def test_ranges_and_distributions_are_unchanged():
    """Only the source of the numbers changed, never a range or a shape."""
    wide_envs = 512
    ids = list(range(wide_envs))
    scenario = make_scenario_rng(_Cfg(3), wide_envs, DEVICE)
    scenario.reset_idx(torch.as_tensor(ids, dtype=torch.long))

    angles = np.array(
        [layout_rotation_angle(scenario, None, index) for index in ids]
    )
    assert angles.min() >= 0.0 and angles.max() < TWO_PI, "rotation left [0, 2pi)"
    assert abs(angles.mean() - math.pi) < 0.2, angles.mean()

    chosen = actuator_choice_indices(
        scenario, None, GROUP_THRUST_IMBALANCE, ids,
        len(THRUST_IMBALANCE_CHOICES),
    )
    assert chosen.shape == (wide_envs,) and chosen.dtype == np.int64
    assert chosen.min() >= 0 and chosen.max() < len(THRUST_IMBALANCE_CHOICES)
    counts = np.bincount(chosen, minlength=len(THRUST_IMBALANCE_CHOICES))
    assert counts.min() > 0, f"a choice was never drawn in 512 envs: {counts}"

    offsets = spawn_jitter_offsets(scenario, None, ids, JITTER_M)
    assert offsets.shape == (wide_envs, 2) and offsets.dtype == np.float32
    radii = np.linalg.norm(offsets.astype(np.float64), axis=1)
    assert radii.max() <= JITTER_M, "jitter left the disc"
    # Uniform over the DISC, not over the radius: E[r] = 2R/3.
    assert abs(radii.mean() - 2.0 * JITTER_M / 3.0) < 0.15, radii.mean()


# ---------------------------------------------------------------------------
# Certificate stamping
# ---------------------------------------------------------------------------
def _stamp_hazard_nav(seed):
    """Exactly the stamp_scenario call hazard_nav_env.py makes at reset."""
    ids = list(range(NUM_ENVS))
    knobs = {GROUP_MASS_SCALE: MASS_SCALE_CHOICES}
    rows = _reset_hazard_nav(
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE), _fallback(), ids,
        knobs=knobs,
    )
    groups = {
        GROUP_LAYOUT: {
            "start_m": [row["start_m"].tolist() for row in rows],
            "goal_m": [row["goal_m"].tolist() for row in rows],
            "obstacle_centers_m": [
                row["obstacle_centers_m"].tolist() for row in rows
            ],
            "obstacle_radii_m": [row["obstacle_radii_m"].tolist() for row in rows],
            "route_geodesic_m": [row["route_geodesic_m"] for row in rows],
            "d0_m": [row["d0_m"] for row in rows],
        },
        GROUP_ROTATION: {"angle_rad": [row["rotation_rad"] for row in rows]},
    }
    groups.update(
        actuator_scenario_entry(
            GROUP_MASS_SCALE, [row[GROUP_MASS_SCALE] for row in rows]
        )
    )
    return stamp_scenario(torch.as_tensor(ids), groups)


def test_stamp_matches_the_certificate_schema():
    """Per-primitive hashes: value-only, seed-sensitive, replayable, JSON-able."""
    first = _stamp_hazard_nav(123)
    replay = _stamp_hazard_nav(123)
    other = _stamp_hazard_nav(42)

    assert [row[0] for row in first] == list(range(NUM_ENVS))
    for (_, _, hashes), (_, _, replay_hashes) in zip(first, replay):
        assert hashes == replay_hashes, "same seed did not reproduce the hashes"
    for (_, _, hashes), (_, _, other_hashes) in zip(first, other):
        # Continuous primitives: every env must differ between the two seeds.
        for group in (GROUP_LAYOUT, GROUP_ROTATION, "scenario"):
            assert hashes[group] != other_hashes[group], (
                f"{group} hash is identical under eval seeds 123 and 42"
            )
    # A CATEGORICAL primitive cannot be asserted per env: mass_scale picks one
    # of three values, so a third of the envs are expected to draw the same
    # value under two seeds and hash identically. The claim that has content is
    # that the seed moves the assignment across the batch.
    actuator_first = [hashes[GROUP_MASS_SCALE] for _, _, hashes in first]
    actuator_other = [hashes[GROUP_MASS_SCALE] for _, _, hashes in other]
    assert actuator_first != actuator_other, (
        "the actuator choice is identical for every env under eval seeds 123 "
        "and 42"
    )

    _, resolved, hashes = first[0]
    assert set(hashes) == {
        GROUP_LAYOUT, GROUP_ROTATION, GROUP_MASS_SCALE, "scenario"
    }
    # The evaluator json.dumps the record that carries these; the resolved
    # params must therefore be plain Python, never numpy scalars.
    json.dumps(resolved)
    assert isinstance(resolved[GROUP_LAYOUT]["d0_m"], float)
    assert isinstance(resolved[GROUP_ROTATION]["angle_rad"], float)

    # VALUE-ONLY, not key-salted: two keys resolving to the same numbers must
    # hash the same, or scripts/check_scenario_independence.py could never see
    # two seeds that genuinely drew the same paper.
    fixed = {GROUP_ROTATION: {"angle_rad": [1.25] * NUM_ENVS}}
    rows = stamp_scenario(torch.as_tensor(list(range(NUM_ENVS))), fixed)
    assert len({row[2][GROUP_ROTATION] for row in rows}) == 1, (
        "identical parameter values hashed differently, so the hash is salted "
        "with the key -- that would hide real duplicates"
    )


def test_actuator_group_names_map_to_readable_parameters():
    entry = actuator_scenario_entry(GROUP_MASS_SCALE, [1.0, 1.15])
    assert entry == {GROUP_MASS_SCALE: {"mass_scale": [1.0, 1.15]}}
    for group in ACTUATOR_GROUPS:
        (name,) = actuator_scenario_entry(group, []) [group]
        assert group == f"actuator_{name}", (group, name)


def test_envs_expose_what_the_evaluators_read():
    """Source-level: the two attributes every evaluator discovers by name.

    scripts/eval_v6_frozen.py:154-158 stamps the protocol header off
    ``base._scenario``, and :213-215 reads ``base.episode_scenario_hashes[i]``
    for the episode that just ended; the shared helpers
    scenario_draws.scenario_protocol_stamp / episode_scenario_hashes_for read
    the same two names for every other evaluator. An env that draws from the
    protocol but never exposes these certifies as off-protocol.
    """
    for name in ("hazard_nav/hazard_nav_env.py", "harbor_mission/harbor_mission_env.py"):
        text = (_TASKS / name).read_text(encoding="utf-8")
        reset = text[text.index("def _reset_idx"):]
        assert "self._scenario = make_scenario_rng(" in text, name
        assert "self.episode_scenario_hashes = [" in text, name
        assert "self._scenario.reset_idx(env_ids)" in reset, name
        # The reseed must precede every draw, or a whole episode is drawn on
        # the previous episode's key.
        assert reset.index("self._scenario.reset_idx(env_ids)") < reset.index(
            "episode_layout_rng("
        ), f"{name}: a layout is drawn before the streams are reseeded"
        assert "self._scenario_hashes[env_index] = hashes" in reset, name
        assert "self.episode_scenario_hashes[completed] = self._scenario_hashes[" in reset, name


def test_sources_are_pure_ascii_without_control_characters():
    """No stray control character can ride into a source file I touched."""
    shared = Path(__file__).resolve().parent
    targets = [
        shared / "scenario_draws_hazard.py",
        shared / "test_scenario_wiring_hazard.py",
        _TASKS / "hazard_nav" / "hazard_nav_env.py",
        _TASKS / "harbor_mission" / "harbor_mission_env.py",
    ]
    for path in targets:
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
    raise SystemExit(1 if failures else 0)
