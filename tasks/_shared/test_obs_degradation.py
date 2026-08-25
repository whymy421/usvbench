# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU acceptance tests for the observation-degradation OOD hooks.

Two layers, no isaaclab import:

1. Behavioral tests of ObsDegrader on fabricated tensors -- noise statistics,
   per-episode bias, exact k-step delay, dropout sample-and-hold with the
   clean-first-frame rule, reset isolation, RNG independence, and the
   zero-config identity (same tensor OBJECT back).
2. AST regression over the two carrier envs and their cfgs: with every
   obs_* field at its zero default the env-side hooks must be a no-op --
   every degrader call sits under an ``is not None`` guard, the frozen
   assignments are exactly the pre-hook ones, and the cfg defaults are 0.
   (The envs cannot be booted here, so the guard STRUCTURE is the contract.)

Run: python tasks/_shared/test_obs_degradation.py
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

try:
    from .kinematics import body_planar_kinematics
    from .obs_degradation import (
        ObsChannelGroup,
        ObsDegrader,
        build_obs_degrader,
    )
    from .obs_superset import SPEED_SCALE_MPS
except ImportError:  # direct execution
    from kinematics import body_planar_kinematics
    from obs_degradation import ObsChannelGroup, ObsDegrader, build_obs_degrader
    from obs_superset import SPEED_SCALE_MPS

SHARED = Path(__file__).resolve().parent
TASKS = SHARED.parent


def _group(dim, noise=0.0, bias=0.0, clamp=None):
    return ObsChannelGroup(
        dim=dim,
        noise_sigma=(noise,) * dim if isinstance(noise, float) else noise,
        bias_sigma=(bias,) * dim if isinstance(bias, float) else bias,
        clamp_min=None if clamp is None else clamp[0],
        clamp_max=None if clamp is None else clamp[1],
    )


# --------------------------------------------------------------- behavioral --
def test_noise_stats_match_sigma():
    num_envs, dim, steps = 128, 3, 400
    sigma = (0.5, 1.0, 2.0)
    degrader = ObsDegrader(
        num_envs, "cpu", 1234, {"g": _group(dim, noise=sigma)}
    )
    degrader.reset(range(num_envs))
    x = torch.zeros(num_envs, dim)
    stacked = torch.stack(
        [degrader.apply("g", x) for _ in range(steps)]
    ).reshape(-1, dim)
    count = stacked.shape[0]
    std = stacked.std(dim=0)
    mean = stacked.mean(dim=0)
    for c in range(dim):
        assert abs(float(std[c]) - sigma[c]) < 0.03 * sigma[c], (c, std)
        assert abs(float(mean[c])) < 5.0 * sigma[c] / math.sqrt(count), (c, mean)


def test_bias_fixed_within_episode_and_resampled_across():
    num_envs, dim = 64, 2
    degrader = build_obs_degrader(
        num_envs, "cpu", 7, {"g": _group(dim, bias=1.5)}
    )
    degrader.reset(torch.arange(num_envs))
    x = torch.zeros(num_envs, dim)
    first = degrader.apply("g", x).clone()
    second = degrader.apply("g", x).clone()
    third = degrader.apply("g", x).clone()
    # Constant within the episode, nonzero, env-decorrelated, sigma-scaled.
    assert torch.equal(first, second) and torch.equal(first, third)
    assert float(first.abs().max()) > 0.0
    assert not torch.equal(first[0], first[1])
    std = float(first.reshape(-1).std())
    assert abs(std - 1.5) < 0.45, std
    # Resetting env 0 redraws ITS bias; env 1 keeps its episode untouched.
    degrader.reset(torch.tensor([0]))
    renewed = degrader.apply("g", x).clone()
    assert not torch.equal(renewed[0], first[0])
    assert torch.equal(renewed[1], first[1])


def test_delay_exact_shift():
    num_envs, dim, k = 4, 2, 3
    degrader = build_obs_degrader(
        num_envs, "cpu", 11, {"g": _group(dim)}, delay_steps=k
    )
    degrader.reset(range(num_envs))

    def frame(t):
        base = 100.0 * torch.arange(num_envs).unsqueeze(-1)
        return (base + t + 0.5 * torch.arange(dim)).float()

    outs = [degrader.apply("g", frame(t)).clone() for t in range(10)]
    for t in range(10):
        expected = frame(0) if t < k else frame(t - k)
        assert torch.equal(outs[t], expected), (t, outs[t], expected)


def test_dropout_holds_clean_first_frame():
    # p = 1: every frame drops, so the output is the CLEAN initial
    # measurement forever -- even though the noise axis is switched on.
    num_envs = 8
    degrader = build_obs_degrader(
        num_envs, "cpu", 21, {"g": _group(1, noise=2.0)}, dropout_p=1.0
    )
    degrader.reset(range(num_envs))
    x0 = torch.arange(num_envs).unsqueeze(-1).float()
    assert torch.equal(degrader.apply("g", x0), x0)
    for t in range(1, 6):
        assert torch.equal(degrader.apply("g", x0 + t), x0), t


def test_dropout_sample_and_hold_identity():
    # 0 < p < 1 with no other axis: every output is either this step's frame
    # (kept) or the previous output (held); both branches must occur.
    num_envs = 16
    degrader = build_obs_degrader(
        num_envs, "cpu", 3, {"g": _group(1)}, dropout_p=0.4
    )
    degrader.reset(range(num_envs))
    held = kept = 0
    previous = None
    for t in range(200):
        x = (10.0 * torch.arange(num_envs).unsqueeze(-1) + t).float()
        out = degrader.apply("g", x).clone()
        if previous is None:
            assert torch.equal(out, x)
        else:
            was_kept = out == x
            was_held = out == previous
            assert bool((was_kept | was_held).all()), t
            kept += int(was_kept.sum())
            held += int(was_held.sum())
        previous = out
    assert kept > 0 and held > 0, (kept, held)


def test_reset_clears_delay_buffers():
    num_envs, k = 2, 2
    degrader = build_obs_degrader(
        num_envs, "cpu", 5, {"g": _group(1)}, delay_steps=k
    )
    degrader.reset([0, 1])
    pre_values = []
    for t in range(5):
        x = torch.tensor([[10.0 + t], [20.0 + t]])
        pre_values.append(x)
        degrader.apply("g", x)
    degrader.reset([0])
    # Env 0 starts over: warm-up replays ITS new clean init, never the stale
    # ring. Env 1 continues its shifted pre-reset stream uninterrupted.
    post = []
    for t in range(4):
        x = torch.tensor([[1000.0 + t], [20.0 + 5 + t]])
        post.append(degrader.apply("g", x).clone())
    assert float(post[0][0, 0]) == 1000.0  # t<k: clean init frame
    assert float(post[1][0, 0]) == 1000.0
    assert float(post[2][0, 0]) == 1000.0  # t==k: delayed frame 0
    assert float(post[3][0, 0]) == 1001.0
    stale = {float(v[0, 0]) for v in pre_values}
    assert all(float(p[0, 0]) not in stale for p in post)
    # env 1: at post step t its frame index is 5+t, output = frame (5+t-k).
    for t, p in enumerate(post):
        assert float(p[1, 0]) == 20.0 + 5 + t - k, (t, p)


def test_zero_config_identity():
    assert build_obs_degrader(
        4, "cpu", 0, {"g": _group(3)}, delay_steps=0, dropout_p=0.0
    ) is None
    degrader = build_obs_degrader(
        4, "cpu", 0, {"idle": _group(2), "hot": _group(3, noise=0.1)}
    )
    degrader.reset(range(4))
    x = torch.randn(4, 2)
    assert degrader.apply("idle", x) is x  # identity: same OBJECT, no copy
    assert degrader.wants("idle") is False
    assert degrader.wants("hot") is True
    assert not torch.equal(degrader.apply("hot", torch.zeros(4, 3)),
                           torch.zeros(4, 3))


def test_rng_isolated_and_deterministic():
    torch.manual_seed(999)
    before = torch.randn(5)
    torch.manual_seed(999)
    degrader = build_obs_degrader(8, "cpu", 42, {"g": _group(2, noise=1.0)})
    degrader.reset(range(8))
    for _ in range(3):
        degrader.apply("g", torch.zeros(8, 2))
    after = torch.randn(5)
    # The degrader must never advance torch's global generator.
    assert torch.equal(before, after)

    def run(seed):
        d = build_obs_degrader(8, "cpu", seed, {"g": _group(2, noise=1.0)})
        d.reset(range(8))
        return torch.stack(
            [d.apply("g", torch.zeros(8, 2)).clone() for _ in range(5)]
        )

    same_a, same_b, other = run(5), run(5), run(6)
    assert torch.equal(same_a, same_b)
    assert not torch.equal(same_a, other)
    assert not torch.equal(same_a[:, 0], same_a[:, 1])  # per-env streams


def test_ray_clamp_bounds():
    degrader = build_obs_degrader(
        8, "cpu", 13, {"ray": _group(4, noise=50.0, clamp=(0.0, 30.0))}
    )
    degrader.reset(range(8))
    x = torch.full((8, 4), 15.0)
    outs = torch.stack([degrader.apply("ray", x) for _ in range(50)])
    assert float(outs.min()) >= 0.0 and float(outs.max()) <= 30.0
    assert bool((outs == 0.0).any()) and bool((outs == 30.0).any())


def test_kinematics_split_is_bit_identical():
    # normalize(rates(...)) must reproduce the pre-refactor closed form bit
    # for bit; this is the frozen path every certified kinematic id runs.
    torch.manual_seed(0)
    forward = torch.nn.functional.normalize(torch.randn(64, 2), dim=-1)
    velocity = torch.randn(64, 3)
    yaw = torch.randn(64)
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    surge = torch.sum(velocity[:, :2] * forward, dim=-1, keepdim=True)
    sway = torch.sum(velocity[:, :2] * left, dim=-1, keepdim=True)
    legacy = torch.hstack(
        (
            torch.clamp(surge / SPEED_SCALE_MPS, -1.0, 1.0),
            torch.clamp(sway / SPEED_SCALE_MPS, -1.0, 1.0),
            torch.clamp(yaw.reshape(-1, 1) / 0.7, -1.0, 1.0),
        )
    )
    current = body_planar_kinematics(
        forward, velocity, yaw, yaw_rate_scale_rad_s=0.7
    )
    assert torch.equal(legacy, current)


# ----------------------------------------------------------- AST regression --
def _parse(relpath):
    return ast.parse((TASKS / relpath).read_text(encoding="utf-8"))


def _function(tree, cls, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _parents(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_degrader_ref(node):
    if isinstance(node, ast.Name) and node.id == "degrader":
        return True
    return isinstance(node, ast.Attribute) and node.attr == "_obs_degrader"


def _test_guards_none(test, extra_names=()):
    """True when the If test contains `<degrader-ref> is not None`."""
    for node in ast.walk(test):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(
            node.ops[0], ast.IsNot
        ):
            comparator = node.comparators[0]
            if not (isinstance(comparator, ast.Constant)
                    and comparator.value is None):
                continue
            if _is_degrader_ref(node.left):
                return True
            if isinstance(node.left, ast.Name) and node.left.id in extra_names:
                return True
    return False


def _under_guard(node, parents, extra_names=()):
    """True when the node only runs while a degrader guard holds.

    Membership must be in the If's BODY (or short-circuited inside its own
    test); the ORELSE branch is the frozen default path and stays unguarded.
    """
    prev, current = node, parents.get(node)
    while current is not None:
        if isinstance(current, ast.If) and _test_guards_none(
            current.test, extra_names
        ):
            if prev is current.test or prev in current.body:
                return True
        prev, current = current, parents.get(current)
    return False


def _assert_degrader_calls_guarded(func, extra_names=()):
    parents = _parents(func)
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("apply", "wants", "reset")
            and _is_degrader_ref(node.func.value)
        ):
            assert _under_guard(node, parents, extra_names), (
                f"unguarded degrader .{node.func.attr} at line {node.lineno}"
            )


def _unguarded_assign_counts(func, names, extra_names=()):
    parents = _parents(func)
    counts = {name: 0 for name in names}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in counts:
                if not _under_guard(node, parents, extra_names):
                    counts[target.id] += 1
    return counts


def test_env_hooks_are_noop_at_defaults():
    for relpath, cls in (
        ("hazard_nav/hazard_nav_env.py", "HazardNavEnv"),
        ("station_keeping/station_keeping_env.py", "StationKeepingEnv"),
    ):
        tree = _parse(relpath)
        # __init__: degrader is None unless the any(<knobs>) guard fires.
        init = _function(tree, cls, "__init__")
        plain_none, guarded_build = 0, 0
        parents = _parents(init)
        for node in ast.walk(init):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == "_obs_degrader"
                for t in node.targets
            ):
                inside_if = any(
                    isinstance(p, ast.If)
                    for p in _ancestors(node, parents)
                )
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    assert not inside_if, f"{relpath}: None-init must be unconditional"
                    plain_none += 1
                else:
                    assert inside_if, (
                        f"{relpath}: degrader build must sit under the "
                        "any(<knobs>) guard"
                    )
                    guarded_build += 1
        assert plain_none == 1 and guarded_build == 1, (relpath, plain_none,
                                                        guarded_build)
        # Observation and reset hooks: guarded degrader calls only.
        _assert_degrader_calls_guarded(
            _function(tree, cls, "_get_observations"), ("degraded_kin",)
        )
        _assert_degrader_calls_guarded(_function(tree, cls, "_reset_idx"))

    # Frozen assignments pinned: at defaults exactly the pre-hook statements
    # write these locals, so the emitted observation is byte-identical.
    hazard = _parse("hazard_nav/hazard_nav_env.py")
    hazard_obs = _function(hazard, "HazardNavEnv", "_get_observations")
    assert _unguarded_assign_counts(
        hazard_obs,
        ("position_to_goal", "distance", "bearing_distance", "ranges",
         "speed_norm"),
        ("degraded_kin",),
    ) == {"position_to_goal": 2, "distance": 2, "bearing_distance": 2,
          "ranges": 1, "speed_norm": 1}
    station = _parse("station_keeping/station_keeping_env.py")
    station_obs = _function(station, "StationKeepingEnv", "_get_observations")
    assert _unguarded_assign_counts(
        station_obs, ("rpos", "observation")
    ) == {"rpos": 1, "observation": 3}

    # degraded_kin: None unconditionally, real value only under the guard.
    parents = _parents(hazard_obs)
    for node in ast.walk(hazard_obs):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "degraded_kin"
            for t in node.targets
        ):
            is_none = (isinstance(node.value, ast.Constant)
                       and node.value.value is None)
            guarded = _under_guard(node, parents)
            assert is_none != guarded, (
                "degraded_kin: None-assign must be unguarded, "
                "apply-assign must be guarded"
            )

    # Feasibility pooling must consume the SAME (possibly degraded) ranges
    # tensor, not a fresh _ray_ranges_m() call that would bypass the hook.
    for node in ast.walk(hazard_obs):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "feasibility_pool"):
            first = node.args[0]
            assert isinstance(first, ast.Name) and first.id == "ranges", (
                "feasibility_pool must pool the shared `ranges` tensor"
            )

    # The reward path stays a CLEAN reader: no-arg _body_motion_observation.
    rewards = _function(hazard, "HazardNavEnv", "_get_rewards")
    for node in ast.walk(rewards):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_body_motion_observation"):
            assert not node.args and not node.keywords, (
                "reward path must read the clean measurement"
            )
    body_motion = _function(hazard, "HazardNavEnv", "_body_motion_observation")
    args = body_motion.args
    assert [a.arg for a in args.args] == ["self", "measurement"]
    assert len(args.defaults) == 1 and isinstance(
        args.defaults[0], ast.Constant
    ) and args.defaults[0].value is None


def _ancestors(node, parents):
    current = node
    while current in parents:
        current = parents[current]
        yield current


def test_cfg_degradation_fields_default_to_zero():
    expected = {
        ("hazard_nav/hazard_nav_env_cfg.py", "HazardNavEnvCfg"): (
            "obs_noise_sigma_pos", "obs_noise_sigma_vel",
            "obs_noise_sigma_ray", "obs_bias_sigma_pos",
            "obs_bias_sigma_vel", "obs_bias_sigma_ray",
            "obs_delay_steps", "obs_dropout_p",
        ),
        ("station_keeping/station_keeping_env_cfg.py",
         "StationKeepingEnvCfg"): (
            "obs_noise_sigma_pos", "obs_noise_sigma_vel",
            "obs_bias_sigma_pos", "obs_bias_sigma_vel",
            "obs_delay_steps", "obs_dropout_p",
        ),
    }
    for (relpath, cls), fields in expected.items():
        tree = _parse(relpath)
        class_node = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == cls
        )
        found = {}
        for item in class_node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(
                item.target, ast.Name
            ) and item.target.id in fields:
                assert isinstance(item.value, ast.Constant), item.target.id
                found[item.target.id] = item.value.value
        assert set(found) == set(fields), (relpath, sorted(found))
        for name, value in found.items():
            assert value == 0, (relpath, name, value)


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
