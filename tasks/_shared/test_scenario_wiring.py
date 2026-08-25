# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU tests for the ENV WIRING of the scenario protocol.

``tasks/_shared/test_scenario_rng.py`` proves the stream algebra.  This file
proves the reset paths were plumbed into it correctly, which is a different
claim and a different failure mode: a correct ScenarioRNG wired to the wrong
group, drawn in the wrong order, or drawn before ``reset_idx`` would still pass
every test in that file.

Isaac Sim is not installed here and the env classes import ``isaaclab`` at
module scope, so the env classes themselves cannot be imported.  That is why
``tasks/_shared/scenario_draws.py`` exists: the reset-path arithmetic lives
there, the envs call it, and so does this file.  ``_reset_episode`` below is a
line-for-line transcription of the draw ORDER in each env's ``_reset_idx``, so
a change to that order shows up here.

Run: python tasks/_shared/test_scenario_wiring.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

try:
    from .test_scenario_regression import (
        PH_MAX_ATTEMPTS,
        PH_OBSTACLE_COUNT,
        layout_generator_for,
        sample_layout,
    )
except ImportError:  # direct execution
    from test_scenario_regression import (
        PH_MAX_ATTEMPTS,
        PH_OBSTACLE_COUNT,
        layout_generator_for,
        sample_layout,
    )

try:
    from .scenario_draws import (
        GROUP_ACTUATOR,
        make_scenario_rng,
        path_following_route,
        scenario_protocol_header,
        spawn_heading,
        stamp_scenario,
        station_keeping_current,
        station_keeping_spawn,
        unit_uniform,
    )
    from .scenario_rng import SCENARIO_PROTOCOL_VERSION
    from .sea_state import SeaState, SeaStateCfg
except ImportError:  # direct execution
    from scenario_draws import (
        GROUP_ACTUATOR,
        make_scenario_rng,
        path_following_route,
        scenario_protocol_header,
        spawn_heading,
        stamp_scenario,
        station_keeping_current,
        station_keeping_spawn,
        unit_uniform,
    )
    from scenario_rng import SCENARIO_PROTOCOL_VERSION
    from sea_state import SeaState, SeaStateCfg


DEVICE = torch.device("cpu")
NUM_ENVS = 4

# Ranges taken from the shipped cfgs so the test exercises real numbers.
# tasks/station_keeping/station_keeping_env_cfg.py values are irrelevant to the
# properties under test, but using plausible ones keeps the bound checks honest.
SPAWN_MIN_M, SPAWN_MAX_M = 1.5, 4.0
CURRENT_MIN_MPS, CURRENT_MAX_MPS = 0.10, 0.60
NUM_WAYPOINTS = 4
SEGMENT_MIN_M, SEGMENT_MAX_M = 3.0, 6.0
HEADING_CHANGE_MAX_RAD = math.radians(45.0)


class _Cfg:
    """The single attribute make_scenario_rng reads off an env cfg."""

    def __init__(self, seed):
        self.seed = seed


def _sea() -> SeaState:
    cfg = SeaStateCfg(enable=True, hs_range=(0.3, 0.6), tp_range=(1.5, 3.0))
    return SeaState(cfg, NUM_ENVS, DEVICE)


# ---------------------------------------------------------------------------
# Reset-path transcriptions. The draw ORDER here mirrors each env's _reset_idx.
# ---------------------------------------------------------------------------
def _reset_station_keeping(scenario, sea, env_ids, reset=True):
    """station_keeping_env.py _reset_idx: reset_idx -> current -> spawn -> sea."""
    ids = torch.as_tensor(env_ids, dtype=torch.long)
    if scenario is not None and reset:
        scenario.reset_idx(ids)
    speeds, directions = station_keeping_current(
        scenario, ids, DEVICE, CURRENT_MIN_MPS, CURRENT_MAX_MPS
    )
    distances, angles, headings = station_keeping_spawn(
        scenario, ids, DEVICE, SPAWN_MIN_M, SPAWN_MAX_M
    )
    sea.resample(ids, scenario=scenario)
    return {
        "current_speed": speeds,
        "current_direction": directions,
        "spawn_distance": distances,
        "spawn_angle": angles,
        "spawn_heading": headings,
        "wave_hs": sea.hs[ids].clone(),
        "wave_tp": sea.tp[ids].clone(),
        "wave_gamma": sea.gamma[ids].clone(),
        "wave_mean_direction": sea.mean_direction[ids].clone(),
        "wave_phase": sea.phase[ids].clone(),
        "wave_direction": sea.direction[ids].clone(),
    }


def _reset_path_following(scenario, env_ids):
    """path_following_env.py _reset_idx: reset_idx -> route -> spawn heading."""
    ids = torch.as_tensor(env_ids, dtype=torch.long)
    if scenario is not None:
        scenario.reset_idx(ids)
    max_change = torch.deg2rad(torch.tensor(45.0, device=DEVICE))
    segments, first_headings, heading_changes = path_following_route(
        scenario, ids, DEVICE, NUM_WAYPOINTS, SEGMENT_MIN_M, SEGMENT_MAX_M,
        max_change,
    )
    spawn = spawn_heading(scenario, ids, DEVICE)
    return {
        "segment_lengths": segments,
        "first_heading": first_headings,
        "heading_changes": heading_changes,
        "spawn_heading": spawn,
    }


def _reset_path_hazard(scenario, env_ids):
    """path_hazard_env.py _reset_idx: reset_idx -> numpy layout -> heading.

    This used to stop at the heading and merely MENTION the layout in a
    comment, so the numpy layout stream -- the thing that decides
    route_geodesic_m and d0_m, i.e. the two fields a cross-controller pairing
    audit compares -- was never executed by any test in the tree.  It now runs
    the shipped ``PathHazardEnv._layout_generator`` (lifted out of the env
    source by ``test_scenario_regression.layout_generator_for``, because the env
    module imports isaaclab) and the shipped ``sample_layout``, once per reset
    env, in the same per-row order as
    ``tasks/path_hazard/path_hazard_env.py:1044-1057``.

    ``waypoints`` is the fixed-shape part of the layout (NUM_SEGMENTS x 2), so
    it stacks; ``route_length`` is the scalar the certificate compares.  The
    obstacle field can shrink through the K-reduction ladder and is therefore
    ragged, so it is checked in ``test_scenario_regression`` instead of here.
    """
    ids = torch.as_tensor(env_ids, dtype=torch.long)
    if scenario is not None:
        scenario.reset_idx(ids)
    # Mirrors the env's unseeded fallback: one shared generator advanced per
    # reset when cfg.seed is None, the keyed per-(env, episode) stream when not.
    generator_for = layout_generator_for(scenario, np.random.default_rng(0))
    layouts = [
        sample_layout(
            rng=generator_for(int(env_index)),
            max_attempts=PH_MAX_ATTEMPTS,
            obstacle_count=PH_OBSTACLE_COUNT,
        )
        for env_index in ids.tolist()
    ]
    return {
        "layout_waypoints": torch.as_tensor(
            np.stack([np.asarray(layout.waypoints, dtype=np.float64)
                      for layout in layouts])
        ),
        "layout_route_length": torch.as_tensor(
            np.array([float(layout.route_length) for layout in layouts])
        ),
        "spawn_heading": spawn_heading(scenario, ids, DEVICE),
    }


def _episodes(scenario, sea, schedule, extra_group=False):
    """Run a reset schedule and collect every env's per-episode scenario.

    ``schedule`` is a list of env-id lists, in the order a controller happened
    to finish episodes.  The return key is (env_index, episode_index), which is
    what two controllers must AGREE on.  ``extra_group=True`` also draws from a
    primitive group the other controller never touches.
    """
    seen = {}
    counters = {i: -1 for i in range(NUM_ENVS)}
    for env_ids in schedule:
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        if scenario is not None:
            scenario.reset_idx(ids)
            if extra_group:
                # Group names are hashed as TEXT, so a primitive only one
                # variant randomises must not shift any other stream.
                unit_uniform(scenario, GROUP_ACTUATOR, ids, DEVICE, size=(2,))
        # reset_idx already advanced the counters, so the transcription below
        # must not advance them again: pass the scenario with resets disabled.
        drawn = _reset_station_keeping(scenario, sea, env_ids, reset=False)
        for row, env_index in enumerate(env_ids):
            counters[env_index] += 1
            seen[(env_index, counters[env_index])] = {
                name: value[row].clone() for name, value in drawn.items()
            }
    return seen


def _same(left, right):
    return all(torch.equal(left[name], right[name]) for name in left)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _skrl_clobber():
    """Reproduce skrl Runner.__init__ -> set_seed(agent yaml seed).

    The agent YAMLs pin a constant (tasks/station_keeping/agents/
    skrl_ppo_cfg.yaml:1 is ``seed: 42``), and the Runner is built AFTER the env,
    so the global torch RNG is at this same state for EVERY evaluation seed by
    the time the first episode is drawn.  Without this line a test comparing two
    eval seeds passes even on the broken code, because the global stream just
    keeps advancing between the two calls -- which is precisely the false
    negative that let the defect survive.
    """
    torch.manual_seed(42)


def test_different_eval_seeds_give_different_scenarios():
    """The defect itself: seeds 123 and 42 must not be the same exam paper."""
    all_ids = list(range(NUM_ENVS))
    drawn = {}
    for seed in (123, 42):
        scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
        _skrl_clobber()
        drawn[seed] = _reset_station_keeping(scenario, _sea(), all_ids)
    for name in drawn[123]:
        assert not torch.equal(drawn[123][name], drawn[42][name]), (
            f"station keeping {name} is identical under eval seeds 123 and 42"
        )

    for seed in (123, 42):
        scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
        _skrl_clobber()
        drawn[seed] = _reset_path_following(scenario, all_ids)
    for name in drawn[123]:
        assert not torch.equal(drawn[123][name], drawn[42][name]), (
            f"path following {name} is identical under eval seeds 123 and 42"
        )

    for seed in (123, 42):
        scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
        _skrl_clobber()
        drawn[seed] = _reset_path_hazard(scenario, all_ids)
    for name in drawn[123]:
        assert not torch.equal(drawn[123][name], drawn[42][name]), (
            f"path hazard {name} is identical under eval seeds 123 and 42"
        )


def test_path_hazard_layout_replays_under_the_same_eval_seed():
    """The numpy layout stream is seed-derived and reproducible, like the rest.

    Guards the other half of the stub repair: a layout drawn off a generator
    that ignored the eval seed would still differ between two seeds by pure
    stream drift, so "differs across seeds" alone is not evidence.
    """
    all_ids = list(range(NUM_ENVS))
    first = _reset_path_hazard(make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE),
                               all_ids)
    torch.manual_seed(999)  # the global RNG must not reach the layout either
    second = _reset_path_hazard(make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE),
                                all_ids)
    assert _same(first, second), "the path hazard layout did not replay"
    assert not torch.equal(
        first["layout_route_length"][0:1], first["layout_route_length"][1:2]
    ), "two envs drew the identical route length"


def test_same_eval_seed_replays_exactly():
    """Two runs of the same seed are the same paper -- reproducibility."""
    all_ids = list(range(NUM_ENVS))
    first = _reset_station_keeping(
        make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE), _sea(), all_ids
    )
    second = _reset_station_keeping(
        make_scenario_rng(_Cfg(7), NUM_ENVS, DEVICE), _sea(), all_ids
    )
    assert _same(first, second), "same eval seed did not replay"


def test_controller_independence_under_different_consumption_order():
    """The property a single advancing generator cannot have.

    Controller A and controller B end episodes at different times, reset
    different subsets together, and B additionally draws from a group A never
    touches (an actuator randomisation, say).  Env i's k-th episode must be the
    same scenario in both.
    """
    seed = 2026
    schedule_a = [[0, 1, 2, 3], [1, 3], [0, 2], [1, 3], [0, 2]]
    schedule_b = [[0, 1, 2, 3], [0], [2], [1], [3], [0, 1, 2, 3]]

    seen_a = _episodes(
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE), _sea(), schedule_a
    )
    seen_b = _episodes(
        make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE), _sea(), schedule_b,
        extra_group=True,
    )

    shared = set(seen_a) & set(seen_b)
    assert len(shared) >= 8, f"only {len(shared)} shared episodes to compare"
    for key in sorted(shared):
        for name in seen_a[key]:
            assert torch.equal(seen_a[key][name], seen_b[key][name]), (
                f"env {key[0]} episode {key[1]}: {name} differs between two "
                "controllers at the same eval seed"
            )


def test_per_env_isolation():
    """Resetting one env must not disturb another env's stream."""
    seed = 99
    everyone = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
    together = _reset_station_keeping(everyone, _sea(), [0, 1, 2, 3])

    alone = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
    sea = _sea()
    # Env 2 resets by itself, after envs 0/1/3 have already reset repeatedly.
    _reset_station_keeping(alone, sea, [0, 1, 3])
    single = _reset_station_keeping(alone, sea, [2])
    for name, value in together.items():
        assert torch.equal(value[2:3], single[name]), (
            f"env 2 {name} depends on which other envs reset alongside it"
        )

    # And two different envs must not be handed the same scenario.
    assert not torch.equal(
        together["spawn_distance"][0], together["spawn_distance"][1]
    ), "env 0 and env 1 drew the same spawn distance"


def test_global_torch_rng_cannot_move_the_scenario():
    """The exact clobber skrl's Runner performs, plus its negative control."""
    all_ids = list(range(NUM_ENVS))
    scenario = make_scenario_rng(_Cfg(555), NUM_ENVS, DEVICE)
    torch.manual_seed(0)
    protected = _reset_station_keeping(scenario, _sea(), all_ids)

    scenario = make_scenario_rng(_Cfg(555), NUM_ENVS, DEVICE)
    torch.manual_seed(12345)  # skrl Runner.__init__ -> set_seed(agent yaml)
    again = _reset_station_keeping(scenario, _sea(), all_ids)
    assert _same(protected, again), (
        "the scenario moved when the global torch RNG was reseeded"
    )

    # Negative control: the OLD path (scenario=None) is exactly what the global
    # reseed destroys, so this asserts the test above has teeth.
    torch.manual_seed(0)
    old_a = _reset_station_keeping(None, _sea(), all_ids)
    torch.manual_seed(12345)
    old_b = _reset_station_keeping(None, _sea(), all_ids)
    assert not _same(old_a, old_b), (
        "the pre-fix global-RNG path did not react to the global seed, so this "
        "test cannot distinguish fixed from broken"
    )


def test_unseeded_fallback_is_the_historical_line():
    """cfg.seed None keeps training bit-identical to the pre-fix code."""
    cfg = _Cfg(None)
    assert make_scenario_rng(cfg, NUM_ENVS, DEVICE) is None

    ids = torch.arange(NUM_ENVS)
    torch.manual_seed(31337)
    expected_scalar = torch.rand(NUM_ENVS, device=DEVICE)
    expected_matrix = torch.rand((NUM_ENVS, 5), device=DEVICE)

    torch.manual_seed(31337)
    assert torch.equal(
        unit_uniform(None, "spawn_pose", ids, DEVICE), expected_scalar
    ), "unseeded unit_uniform is not torch.rand(count)"
    assert torch.equal(
        unit_uniform(None, "spawn_pose", ids, DEVICE, size=(5,)),
        expected_matrix,
    ), "unseeded unit_uniform is not torch.rand((count, n))"


def test_ranges_and_shapes_are_unchanged():
    """Only the source of the numbers changed, never a range or a shape."""
    ids = torch.arange(NUM_ENVS)
    scenario = make_scenario_rng(_Cfg(11), NUM_ENVS, DEVICE)
    scenario.reset_idx(ids)

    speeds, directions = station_keeping_current(
        scenario, ids, DEVICE, CURRENT_MIN_MPS, CURRENT_MAX_MPS
    )
    distances, angles, headings = station_keeping_spawn(
        scenario, ids, DEVICE, SPAWN_MIN_M, SPAWN_MAX_M
    )
    assert speeds.shape == (NUM_ENVS,) and distances.shape == (NUM_ENVS,)
    assert bool(
        (speeds >= CURRENT_MIN_MPS).all() and (speeds < CURRENT_MAX_MPS).all()
    ), "current speed left [min, max)"
    assert bool(
        (distances >= SPAWN_MIN_M).all() and (distances < SPAWN_MAX_M).all()
    ), "spawn distance left [min, max)"
    for name, value in (("angle", angles), ("heading", headings),
                        ("direction", directions)):
        assert bool((value >= 0.0).all() and (value < 2.0 * math.pi).all()), (
            f"{name} left [0, 2*pi)"
        )

    scenario = make_scenario_rng(_Cfg(11), NUM_ENVS, DEVICE)
    scenario.reset_idx(ids)
    segments, first_headings, changes = path_following_route(
        scenario, ids, DEVICE, NUM_WAYPOINTS, SEGMENT_MIN_M, SEGMENT_MAX_M,
        torch.tensor(HEADING_CHANGE_MAX_RAD),
    )
    assert segments.shape == (NUM_ENVS, NUM_WAYPOINTS), segments.shape
    assert changes.shape == (NUM_ENVS, NUM_WAYPOINTS - 1), changes.shape
    assert first_headings.shape == (NUM_ENVS,), first_headings.shape
    assert bool(
        (segments >= SEGMENT_MIN_M).all() and (segments < SEGMENT_MAX_M).all()
    ), "segment length left [min, max)"
    assert bool(
        (changes >= -HEADING_CHANGE_MAX_RAD).all()
        and (changes < HEADING_CHANGE_MAX_RAD).all()
    ), "heading change left [-max, +max)"

    # Uniformity is the claim that matters most: a mis-scaled stream would show
    # up as a mean far from the midpoint over a large sample.
    wide = make_scenario_rng(_Cfg(3), 512, DEVICE)
    wide_ids = torch.arange(512)
    wide.reset_idx(wide_ids)
    sample, _ = station_keeping_current(wide, wide_ids, DEVICE, 0.0, 1.0)
    assert abs(float(sample.mean()) - 0.5) < 0.03, float(sample.mean())
    assert float(sample.min()) < 0.02 and float(sample.max()) > 0.98


def test_substreams_are_isolated_between_primitives():
    """Adding a draw to one primitive must not shift another primitive."""
    ids = torch.arange(NUM_ENVS)

    baseline = make_scenario_rng(_Cfg(404), NUM_ENVS, DEVICE)
    baseline.reset_idx(ids)
    _, _, spawn_baseline = station_keeping_spawn(
        baseline, ids, DEVICE, SPAWN_MIN_M, SPAWN_MAX_M
    )

    perturbed = make_scenario_rng(_Cfg(404), NUM_ENVS, DEVICE)
    perturbed.reset_idx(ids)
    # Two extra current draws and an actuator group that did not exist before.
    station_keeping_current(perturbed, ids, DEVICE, 0.0, 1.0)
    unit_uniform(perturbed, GROUP_ACTUATOR, ids, DEVICE, size=(3,))
    _, _, spawn_perturbed = station_keeping_spawn(
        perturbed, ids, DEVICE, SPAWN_MIN_M, SPAWN_MAX_M
    )
    assert torch.equal(spawn_baseline, spawn_perturbed), (
        "drawing from current/actuator shifted the spawn_pose stream"
    )


def test_stamp_scenario_matches_the_certificate_schema():
    """Per-primitive hashes: value-only, seed-sensitive, replayable."""
    all_ids = list(range(NUM_ENVS))

    def stamped(seed):
        scenario = make_scenario_rng(_Cfg(seed), NUM_ENVS, DEVICE)
        _skrl_clobber()
        drawn = _reset_station_keeping(scenario, _sea(), all_ids)
        return stamp_scenario(
            torch.as_tensor(all_ids),
            {
                "spawn_pose": {
                    "distance_m": drawn["spawn_distance"],
                    "angle_rad": drawn["spawn_angle"],
                    "heading_rad": drawn["spawn_heading"],
                },
                "current": {"speed_mps": drawn["current_speed"]},
                "wave": {"hs_m": drawn["wave_hs"],
                         "phase_rad": drawn["wave_phase"]},
            },
        )

    first = stamped(123)
    replay = stamped(123)
    other = stamped(42)

    assert [row[0] for row in first] == all_ids, "env indices are misaligned"
    for (_, _, hashes), (_, _, replay_hashes) in zip(first, replay):
        assert hashes == replay_hashes, "same seed did not reproduce the hashes"
    for (_, _, hashes), (_, _, other_hashes) in zip(first, other):
        for group in ("spawn_pose", "current", "wave", "scenario"):
            assert hashes[group] != other_hashes[group], (
                f"{group} hash is identical under eval seeds 123 and 42"
            )

    # Every group named in the schema is present, and the resolved params are
    # plain JSON-serialisable values (the certificate has to survive json.dump).
    _, resolved, hashes = first[0]
    assert set(hashes) == {"spawn_pose", "current", "wave", "scenario"}
    assert isinstance(resolved["spawn_pose"]["distance_m"], float)
    assert isinstance(resolved["wave"]["phase_rad"], list)

    # VALUE-ONLY, not key-salted. Two different (env, episode) keys that
    # resolve to the same numbers must hash the SAME, otherwise
    # scripts/check_scenario_independence.py could never see two seeds that
    # genuinely drew the same paper.
    fixed = {"spawn_pose": {"distance_m": torch.full((NUM_ENVS,), 2.5)}}
    rows = stamp_scenario(torch.as_tensor(all_ids), fixed)
    digests = {row[2]["spawn_pose"] for row in rows}
    assert len(digests) == 1, (
        "identical parameter values hashed differently, so the hash is salted "
        "with the key -- that would hide real duplicates"
    )


def test_certificate_header_block():
    header = scenario_protocol_header()
    assert header["version"] == SCENARIO_PROTOCOL_VERSION
    assert header["hash"] == "blake2b"
    # No run-specific value may live here: the auditor warns when the two
    # certificates of a pair disagree on this block, and they differ exactly in
    # their evaluation seed.
    assert "eval_seed" not in header and "seed" not in header


def test_sources_are_pure_ascii_without_control_characters():
    """No stray control character can ride into a source file I touched."""
    shared = Path(__file__).resolve().parent
    tasks = shared.parent
    targets = [
        shared / "scenario_draws.py",
        shared / "scenario_rng.py",
        shared / "sea_state.py",
        shared / "test_scenario_wiring.py",
        shared / "test_scenario_regression.py",
        tasks / "station_keeping" / "station_keeping_env.py",
        tasks / "station_keeping_boat" / "station_keeping_boat_env.py",
        tasks / "path_following" / "path_following_env.py",
        tasks / "path_hazard" / "path_hazard_env.py",
        tasks.parent / "scripts" / "eval_v6_frozen.py",
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
