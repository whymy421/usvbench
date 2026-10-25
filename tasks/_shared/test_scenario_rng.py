# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU acceptance tests for the scenario RNG (no isaaclab, no GPU).

The bug this guards: 25 of 79 paired certificates had byte-identical
per-episode success patterns at ``--eval-seed 123`` and ``--eval-seed 42``,
because the station-keeping family drew its scenario from the GLOBAL torch
RNG (``tasks/station_keeping/station_keeping_env.py:684-688``,
``tasks/_shared/sea_state.py:106-118``) and an skrl Runner reseeds that global
RNG from a constant in the agent YAML
(``tasks/station_keeping/agents/skrl_ppo_cfg.yaml:1`` is ``seed: 42``) after
``scripts/eval_v6_frozen.py:74`` has already applied the eval seed.

So the properties under test are not cosmetic:

* the eval seed must actually reach the scenario (and two seeds must differ);
* the scenario must NOT move when something reseeds the global torch RNG;
* two controllers that consume randomness differently must still sit the
  SAME exam paper -- tested by replaying with the group draws interleaved in
  a different order and with an extra group one run never touches;
* resetting one env must not disturb another env's paper;
* the key hash must be stable ACROSS PROCESSES -- pinned to hard-coded
  digests here, and re-checked in subprocesses under two PYTHONHASHSEED
  values, which is what Python's salted builtin ``hash()`` would fail;
* every bit of the key must reach the DRAWN VALUES.  The first version of
  this file only ever compared seed INTEGERS, and that hole let a second
  defect through: ``torch.Generator.manual_seed`` accepts 64 bits and keeps
  32, so two keys could differ as integers and still emit an identical
  stream.  Assertions about keying therefore compare what the generator
  emits (see ``_draws``), never the seed integer alone.

Run: python tasks/_shared/test_scenario_rng.py
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import torch

try:
    from .scenario_rng import (
        PROTOCOL_NOTES,
        SCENARIO_GROUPS,
        SCENARIO_PROTOCOL_VERSION,
        ScenarioRNG,
        as_torch_seed,
        numpy_generator,
        scenario_hash,
        stable_stream_seed,
    )
except ImportError:  # direct execution
    from scenario_rng import (
        PROTOCOL_NOTES,
        SCENARIO_GROUPS,
        SCENARIO_PROTOCOL_VERSION,
        ScenarioRNG,
        as_torch_seed,
        numpy_generator,
        scenario_hash,
        stable_stream_seed,
    )

SHARED = Path(__file__).resolve().parent
SOURCE = SHARED / "scenario_rng.py"

# Pinned scenario parameters used by the hash tests.
PINNED_PARAMS = {
    "spawn_distance_m": 4.25,
    "spawn_angle_rad": 1.5,
    "hs_m": 0.6,
    "n": 3,
}


def _draws(key, n=4):
    """The floats the scenario stream for one 64-bit KEY actually emits.

    This -- not the seed integer -- is the observable the benchmark cares
    about, so it is the observable the keying tests compare.
    """
    generator = torch.Generator()
    generator.manual_seed(as_torch_seed(key))
    return torch.rand(n, generator=generator).tolist()


def _raw_draws(seed, n=4):
    """``_draws`` WITHOUT the fold: what torch emits from a seed handed to it.

    Used to prove the fold is not a no-op.  ``torch.Generator.manual_seed``
    accepts 64 bits and keeps the low 32, so ``_raw_draws(key)`` is exactly
    what the pre-fix ``as_torch_seed`` (a straight pass-through of the digest)
    produced, and it is byte-identical to ``_raw_draws(key & 0xFFFFFFFF)``.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.rand(n, generator=generator).tolist()


def _key_draws(protocol_version, eval_seed, env_index, episode_index, group):
    """``_draws`` of the key built from one full scenario key tuple."""
    return _draws(
        stable_stream_seed(
            protocol_version, eval_seed, env_index, episode_index, group
        )
    )


def _paper(eval_seed, order, extra_group=None, num_envs=4, episode=None):
    """One controller's view of the exam paper.

    ``order`` is the order in which this controller happens to consume the
    primitive groups; ``extra_group`` is a group it draws from that another
    controller never touches (an actuator-noise model, say).  Neither may
    change the values any group returns.
    """
    rng = ScenarioRNG(num_envs=num_envs, device="cpu", eval_seed=eval_seed)
    rng.reset_idx(torch.arange(num_envs), episode_index=episode)
    ids = list(range(num_envs))
    drawn = {}
    for group in order:
        if extra_group is not None:
            # Interleaved BEFORE each group: a shared stream would desync here.
            rng.uniform(extra_group, ids, -1.0, 1.0, size=(5,))
            rng.normal(extra_group, [0], size=(3,))
        drawn[group] = rng.uniform(group, ids, 0.0, 10.0, size=(3,))
    return drawn


# ------------------------------------------------------------------- keying --
def test_stable_seed_is_pinned():
    # Hard-coded on purpose: these exact integers are the benchmark's exam
    # papers.  If a future edit changes the hashing, this line fails instead
    # of silently reshuffling every scenario in the suite.
    assert stable_stream_seed(2, 123, 0, 0, "spawn_pose") == 9603477983582327822
    assert stable_stream_seed(2, 42, 0, 0, "spawn_pose") == 14805704900687423346
    assert stable_stream_seed(2, 123, 7, 3, "wave") == 5538925527437492987
    assert stable_stream_seed(1, 0, 0, 0, "spawn_pose") == 15641016452370946757

    # RE-PINNED.  as_torch_seed used to be a two's-complement reinterpretation
    # of the 64-bit key (9603477983582327822 -> -8843266090127223794), which
    # handed torch bits it silently discards.  It now folds the whole key
    # through numpy SeedSequence into [0, 2**32); see TORCH SEED WIDTH in
    # scenario_rng.py.  Every scenario in the benchmark re-rolled once more.
    assert as_torch_seed(9603477983582327822) == 1725929043
    assert as_torch_seed(5538925527437492987) == 2279917495
    assert 0 <= as_torch_seed((1 << 64) - 1) < (1 << 32)

    # Pin the WHOLE chain, key -> torch seed -> floats, so a change anywhere
    # along it fails here rather than quietly reshuffling the suite.
    assert _draws(9603477983582327822, 3) == [
        0.5591599345207214, 0.5856941938400269, 0.9985114336013794
    ]
    assert SCENARIO_PROTOCOL_VERSION == 2 and 1 in PROTOCOL_NOTES


def test_seed_is_stable_across_processes():
    """The key, the folded torch seed AND the floats must cross a fork intact.

    Carrying the drawn floats matters as much as the key: the fold added a
    numpy SeedSequence hop between them, and a hop that was not reproducible
    across processes would reintroduce the salted-hash class of bug at the
    stage nobody was looking at.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import torch;"
        "from scenario_rng import stable_stream_seed as s, as_torch_seed as t;"
        "k = s(2, 123, 0, 0, 'spawn_pose');"
        "g = torch.Generator(); g.manual_seed(t(k));"
        "v = ','.join(repr(x) for x in torch.rand(3, generator=g).tolist());"
        "print(k, t(k), v, hash('spawn_pose'))" % str(SHARED)
    )
    seen = {}
    for salt in ("0", "1", "123456"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = salt
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env, check=True,
        ).stdout.split()
        seen[salt] = (int(out[0]), int(out[1]), out[2], int(out[3]))
    stream = {value[0] for value in seen.values()}
    assert stream == {9603477983582327822}, seen
    folded = {value[1] for value in seen.values()}
    assert folded == {1725929043}, seen
    drawn = {value[2] for value in seen.values()}
    assert drawn == {"0.5591599345207214,0.5856941938400269,0.9985114336013794"}, seen
    builtin = {value[3] for value in seen.values()}
    print(
        f"  cross-process: blake2b key {stream.pop()} -> torch seed "
        f"{folded.pop()} -> {drawn.pop()} identical under {len(seen)} "
        f"PYTHONHASHSEED values; builtin hash() took "
        f"{len(builtin)} different values"
    )


def test_key_fields_are_all_load_bearing():
    """Every key field must move the DRAWN VALUES, not just the seed integer.

    This test used to compare ``stable_stream_seed`` integers, which is the
    wrong observable: nothing in the benchmark consumes a seed integer, and an
    integer that differs says nothing about the stream that follows it.  It
    now asserts on what the generator emits, so it stays honest no matter what
    the seed plumbing downstream of the digest does.

    STRENGTHENED, and it is worth saying exactly what was wrong before: the
    first half of this test (a field changes the values) is carried by blake2b's
    diffusion, which changes the digest's low 32 bits as well as its high ones.
    It therefore passed unchanged against the TRUNCATING build -- the one that
    handed the raw 64-bit digest to ``manual_seed`` and lost bits 32..63 -- so
    reverting that fix left this test green and the file's own docstring said as
    much.  A test that cannot fail on the revert it names does not protect it.

    The second half closes that: for the key each FIELD produces, the drawn
    values must differ from the values of the same key with its high half
    thrown away.  On the truncating build those are the SAME 32 bits and hence
    the same stream, so the first such comparison fails and the test goes red.
    The field coverage and the truncation regression now live in one test
    instead of one real check plus one decoration.
    """
    base_key = stable_stream_seed(2, 123, 4, 5, "wave")
    base = _draws(base_key)
    variants = {
        "protocol_version": stable_stream_seed(3, 123, 4, 5, "wave"),
        "eval_seed": stable_stream_seed(2, 124, 4, 5, "wave"),
        "env_index": stable_stream_seed(2, 123, 5, 5, "wave"),
        "episode_index": stable_stream_seed(2, 123, 4, 6, "wave"),
        "group_name": stable_stream_seed(2, 123, 4, 5, "wav"),
    }
    for field, key in variants.items():
        assert _draws(key) != base, field
    # Length-prefixed group names: no concatenation aliasing, in the values.
    assert _key_draws(2, 123, 4, 5, "ab") != _key_draws(2, 123, 4, 5, "a")

    # THE TEETH.  Every field's key must reach the values through all 64 bits,
    # not through the low 32 torch would otherwise keep.  ``key & 0xFFFFFFFF``
    # is what the truncating build effectively drew from, so equality here IS
    # that build.
    for field, key in {"base": base_key, **variants}.items():
        low_half = key & 0xFFFFFFFF
        assert key != low_half, f"{field}: pick a key with a non-zero high half"
        assert _draws(key) != _draws(low_half), (
            f"{field}: the key's high 32 bits do not reach the drawn values, so "
            "torch is being seeded from the low half of the digest alone"
        )
        # And the fold is not the identity: handing torch the raw key is the
        # pre-fix behaviour, and it must not be what the stream sees.
        assert _draws(key) != _raw_draws(key), (
            f"{field}: as_torch_seed is a pass-through, so manual_seed silently "
            "truncates the key"
        )

    # Eval seeds differing only ABOVE bit 31 (0 vs 2**32, 42 vs 2**32 + 42).
    for low in (0, 42, 123):
        assert _key_draws(2, low, 4, 5, "wave") != _key_draws(
            2, low + (1 << 32), 4, 5, "wave"
        ), low

    # The integer-level property still holds; it is now a supporting check,
    # not the whole test.
    assert stable_stream_seed(2, 124, 4, 5, "wave") != base_key
    assert stable_stream_seed(2, 123, 4, 5, "ab") != stable_stream_seed(
        2, 123, 4, 5, "a"
    )


def test_every_high_key_bit_reaches_the_drawn_values():
    """The exact regression: bits 32..63 used to fall off at ``manual_seed``.

    ``torch.Generator.manual_seed`` takes a 64-bit argument and seeds CPU
    MT19937 from its low 32 bits.  Measured against the previous
    implementation, which passed the raw key through: 32 of 32 high-bit flips
    left the drawn values UNCHANGED for every key below, so all 160 of those
    assertions failed -- while every seed-integer assertion in this file
    passed.  This is the test that would have caught the defect.
    """
    keys = (
        0x0123456789ABCDEF,
        0x0000000000000000,
        0xFFFFFFFFFFFFFFFF,
        stable_stream_seed(2, 123, 0, 0, "spawn_pose"),
        stable_stream_seed(2, 42, 7, 3, "wave"),
    )
    for key in keys:
        base = _draws(key)
        for bit in range(32, 64):
            assert _draws(key ^ (1 << bit)) != base, (hex(key), bit)
        # The low half is not exempt from scrutiny just because torch reads
        # it: check the boundary bits on that side too.
        for bit in (0, 1, 30, 31):
            assert _draws(key ^ (1 << bit)) != base, (hex(key), bit)

    # Why the fold is needed at all, reported rather than asserted: torch may
    # widen manual_seed one day, and the fold stays correct either way.
    low, high = torch.Generator(), torch.Generator()
    low.manual_seed(0x00000000ABCDEF01)
    high.manual_seed(0x12345678ABCDEF01)
    truncated = torch.equal(
        torch.rand(4, generator=low), torch.rand(4, generator=high)
    )
    print(
        f"  torch {torch.__version__}: manual_seed keeps only the low 32 bits "
        f"= {truncated}; the SeedSequence fold makes all 64 count regardless"
    )


def test_torch_fold_is_a_separate_branch_from_the_numpy_stream():
    """The torch handoff must be DERIVED, and on a branch of its own.

    Two independent ways the handoff can be wrong, and this test now fails on
    both:

    * Raw key material.  The pre-fix ``as_torch_seed`` returned the digest
      unchanged and ``manual_seed`` kept its low 32 bits, so the torch stream
      was literally ``key & 0xFFFFFFFF``.  The previous version of this test
      compared ``as_torch_seed(key)`` against the numpy word only, which a
      pass-through satisfies trivially -- it stayed green against that build.
    * A shared branch.  ``numpy_generator`` seeds PCG64 from
      ``SeedSequence(key)`` and ``generate_state(1)`` is a PREFIX of what PCG64
      consumes, so dropping ``_TORCH_SPAWN_KEY`` would make one group's torch
      and numpy streams share their first 32 bits of entropy.

    Both halves are asserted on the DRAWN VALUES as well as on the integer,
    which is the rule this file sets itself in its module docstring.
    """
    import numpy as np

    keys = (
        stable_stream_seed(2, 123, 0, 0, "wave"),
        stable_stream_seed(2, 42, 5, 2, "current"),
        stable_stream_seed(2, 0, 0, 0, "spawn_pose"),
        stable_stream_seed(2, 2026, 63, 127, "layout"),
        0x0123456789ABCDEF,
    )
    for key in keys:
        unsalted = int(np.random.SeedSequence(key).generate_state(1, np.uint32)[0])
        assert as_torch_seed(key) != unsalted, hex(key)
        assert _draws(key) != _raw_draws(unsalted), (
            f"{hex(key)}: the torch stream is the numpy generator's own first "
            "word, so the two streams for this group share entropy"
        )

        # Not raw key material either: manual_seed keeps the low 32 bits, so a
        # pass-through fold makes these three the same stream.
        assert as_torch_seed(key) != (key & 0xFFFFFFFF), hex(key)
        assert as_torch_seed(key) != (key >> 32), hex(key)
        assert _draws(key) != _raw_draws(key), (
            f"{hex(key)}: as_torch_seed hands torch the key itself, which "
            "manual_seed then truncates to its low 32 bits"
        )
        assert _draws(key) != _raw_draws(key & 0xFFFFFFFF), hex(key)

    # The fold must still be a FUNCTION of the key, not of anything ambient:
    # calling it twice, and calling it after the global RNGs move, is stable.
    torch.manual_seed(999)
    np.random.seed(999)
    assert [as_torch_seed(key) for key in keys] == [
        as_torch_seed(key) for key in keys
    ]


def test_source_uses_hashlib_not_builtin_hash():
    """AST, not grep: the docstring is allowed to NAME the banned function."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    builtin = [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "hash"
    ]
    assert not builtin, f"builtin hash() is salted per process; lines {builtin}"
    blake = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "blake2b"
    ]
    assert blake, "the keying must go through hashlib.blake2b"


# ------------------------------------------------------- the reported defect --
def test_different_eval_seeds_give_different_scenarios():
    a = _paper(123, ("spawn_pose", "wave", "current"))
    b = _paper(42, ("spawn_pose", "wave", "current"))
    for group in a:
        assert not torch.equal(a[group], b[group]), group
        assert not torch.allclose(a[group], b[group]), group


def test_same_eval_seed_is_bit_identical():
    a = _paper(123, ("spawn_pose", "wave", "current"))
    b = _paper(123, ("spawn_pose", "wave", "current"))
    for group in a:
        assert torch.equal(a[group], b[group]), group


def test_controller_independence_order_and_extra_group():
    """Property (b): the exam paper cannot depend on the examinee.

    Run B consumes the groups in a different order AND draws from a group run
    A never touches, interleaved between run A's groups.  A single per-env
    stream advanced by draws would desynchronise here; per (env, group)
    streams keyed by episode do not.
    """
    a = _paper(123, ("spawn_pose", "wave", "current"))
    b = _paper(123, ("current", "spawn_pose", "wave"), extra_group="actuator")
    for group in a:
        assert torch.equal(a[group], b[group]), (
            group, a[group][:2].tolist(), b[group][:2].tolist()
        )


def test_global_torch_rng_cannot_move_the_scenario():
    """Property (c): this is exactly what skrl's Runner does to the world."""
    clean = _paper(123, ("spawn_pose", "wave"))

    rng = ScenarioRNG(num_envs=4, device="cpu", eval_seed=123)
    rng.reset_idx(torch.arange(4))
    torch.manual_seed(42)  # skrl Runner.__init__ -> set_seed(cfg["seed"])
    poisoned = {}
    for group in ("spawn_pose", "wave"):
        torch.manual_seed(42)
        torch.rand(1000)
        poisoned[group] = rng.uniform(group, list(range(4)), 0.0, 10.0, size=(3,))
    for group in clean:
        assert torch.equal(clean[group], poisoned[group]), group

    # ... and the traffic is one-way: we do not disturb the global stream.
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    side = ScenarioRNG(num_envs=4, device="cpu", eval_seed=999)
    side.reset_idx(torch.arange(4))
    side.uniform("spawn_pose", [0, 1, 2, 3], 0.0, 1.0, size=(16,))
    side.normal("wave", [0, 1, 2, 3], size=(16,))
    assert torch.equal(expected, torch.rand(3))


# -------------------------------------------------------------- isolation ---
def test_resetting_one_env_does_not_disturb_others():
    def draw(reset_env3):
        rng = ScenarioRNG(num_envs=8, device="cpu", eval_seed=20260824)
        rng.reset_idx(torch.arange(8))
        if reset_env3:
            rng.reset_idx(torch.tensor([3]))
        return (
            rng.uniform("spawn_pose", [0, 5], 0.0, 1.0, size=(4,)),
            rng.uniform("spawn_pose", [3], 0.0, 1.0, size=(4,)),
        )

    quiet_others, quiet_three = draw(False)
    reset_others, reset_three = draw(True)
    assert torch.equal(quiet_others, reset_others)          # envs 0 and 5 fixed
    assert not torch.equal(quiet_three, reset_three)        # env 3 moved on


def test_subset_draw_equals_full_batch_draw():
    full = ScenarioRNG(num_envs=8, device="cpu", eval_seed=5)
    full.reset_idx(torch.arange(8))
    every = full.uniform("spawn_pose", torch.arange(8), 2.0, 9.0, size=(3,))

    lone = ScenarioRNG(num_envs=8, device="cpu", eval_seed=5)
    lone.reset_idx(torch.arange(8))
    just_five = lone.uniform("spawn_pose", [5], 2.0, 9.0, size=(3,))
    assert torch.equal(every[5], just_five[0])

    pair = ScenarioRNG(num_envs=8, device="cpu", eval_seed=5)
    pair.reset_idx(torch.arange(8))
    two = pair.uniform("spawn_pose", [7, 1], 2.0, 9.0, size=(3,))
    assert torch.equal(two[0], every[7]) and torch.equal(two[1], every[1])


def test_groups_are_independent_substreams():
    rng = ScenarioRNG(num_envs=4, device="cpu", eval_seed=11)
    rng.reset_idx(torch.arange(4))
    ids = [0, 1, 2, 3]
    spawn = rng.uniform("spawn_pose", ids, 0.0, 1.0, size=(4,))
    wave = rng.uniform("wave", ids, 0.0, 1.0, size=(4,))
    assert not torch.equal(spawn, wave)
    seeds = {g: rng.seed_for(g, 0) for g in SCENARIO_GROUPS}
    assert len(set(seeds.values())) == len(SCENARIO_GROUPS), seeds

    # Adding a primitive later must not renumber existing streams: group ids
    # are hashed NAMES, unlike the positional enumerate(sorted(...)) ids in
    # tasks/_shared/obs_degradation.py:163.
    late = ScenarioRNG(
        num_envs=4, device="cpu", eval_seed=11,
        groups=("aaa_new_primitive", "spawn_pose", "zzz_new_primitive"),
    )
    late.reset_idx(torch.arange(4))
    assert torch.equal(spawn, late.uniform("spawn_pose", ids, 0.0, 1.0, size=(4,)))


# ---------------------------------------------------------------- episodes --
def test_episode_index_advances_and_replays():
    rng = ScenarioRNG(num_envs=4, device="cpu", eval_seed=77)
    ids = [0, 1, 2, 3]
    rng.reset_idx(torch.arange(4))
    assert rng.episode_indices().tolist() == [0, 0, 0, 0]
    episode0 = rng.uniform("spawn_pose", ids, 0.0, 1.0, size=(3,))

    rng.reset_idx(torch.arange(4))
    assert rng.episode_indices().tolist() == [1, 1, 1, 1]
    episode1 = rng.uniform("spawn_pose", ids, 0.0, 1.0, size=(3,))
    assert not torch.equal(episode0, episode1)

    rng.reset_idx(torch.arange(4), episode_index=0)
    replay = rng.uniform("spawn_pose", ids, 0.0, 1.0, size=(3,))
    assert torch.equal(episode0, replay)

    # Per-env explicit indices, and the counter is genuinely per env.
    rng.reset_idx([1, 2], episode_index=[0, 5])
    assert rng.episode_indices().tolist() == [0, 0, 5, 0]


def test_draw_before_reset_is_refused():
    rng = ScenarioRNG(num_envs=2, device="cpu", eval_seed=1)
    try:
        rng.uniform("spawn_pose", [0], 0.0, 1.0)
    except RuntimeError as exc:
        assert "reset_idx" in str(exc), exc
    else:
        raise AssertionError("drawing from an env with no episode must raise")


# ------------------------------------------------------------------ shapes --
def test_shapes_dtype_and_scatter_helpers():
    rng = ScenarioRNG(num_envs=6, device="cpu", eval_seed=3)
    rng.reset_idx(torch.arange(6))
    flat = rng.uniform("spawn_pose", [0, 2, 4], 1.0, 3.0)
    assert flat.shape == (3,) and flat.dtype is torch.float32
    assert float(flat.min()) >= 1.0 and float(flat.max()) < 3.0
    assert rng.uniform("wave", [1], size=7).shape == (1, 7)
    assert rng.normal("wave", [1, 3], size=(2, 5)).shape == (2, 2, 5)
    assert rng.uniform("wave", [], size=(4,)).shape == (0, 4)

    # A degenerate range is legal and constant (calm sea, fixed spawn).
    assert torch.equal(
        rng.uniform("current", [0, 1], 2.5, 2.5, size=(3,)),
        torch.full((2, 3), 2.5),
    )

    buffer = torch.zeros(6, 3)
    rng.uniform_(buffer, "current", [1, 4], 5.0, 6.0)
    touched = torch.tensor([1, 4])
    assert float(buffer[touched].min()) >= 5.0
    assert float(buffer[[0, 2, 3, 5]].abs().max()) == 0.0
    rng.normal_(buffer, "actuator", [0], mean=100.0, std=0.0)
    assert torch.equal(buffer[0], torch.full((3,), 100.0))

    # 1-D buffers are the sea-state call shape (self.hs[env_ids] = ...,
    # tasks/_shared/sea_state.py:107) and must scatter the same way.
    hs = torch.zeros(6)
    rng.uniform_(hs, "wave", torch.tensor([2, 5]), 0.4, 0.8)
    assert float(hs[torch.tensor([2, 5])].min()) >= 0.4
    assert float(hs[[0, 1, 3, 4]].abs().max()) == 0.0
    try:
        rng.uniform_(torch.zeros(5, 2), "wave", [0])
    except ValueError as exc:
        assert "num_envs" in str(exc), exc
    else:
        raise AssertionError("a dest with the wrong leading dim must raise")

    # numpy side (hazard_nav-style layouts) rides the same key.
    same = numpy_generator(SCENARIO_PROTOCOL_VERSION, 3, 2, 0, "spawn_pose")
    assert (rng.numpy_rng("spawn_pose", 2).random(4) == same.random(4)).all()


def test_guardrails():
    rng = ScenarioRNG(num_envs=4, device="cpu", eval_seed=1)
    rng.reset_idx(torch.arange(4))
    cases = [
        (IndexError, lambda: rng.uniform("spawn_pose", [9])),
        (ValueError, lambda: rng.uniform("", [0])),
        (ValueError, lambda: rng.uniform("spawn_pose", [0], 3.0, 1.0)),
        (ValueError, lambda: rng.normal("spawn_pose", [0], std=-1.0)),
        (ValueError, lambda: ScenarioRNG(0, "cpu", 1)),
        (ValueError, lambda: ScenarioRNG(4, "cpu", 1, protocol_version=0)),
        (ValueError, lambda: rng.reset_idx([0], episode_index=-1)),
        (TypeError, lambda: stable_stream_seed(2, 1.5, 0, 0, "spawn_pose")),
        (TypeError, lambda: stable_stream_seed(2, 1, 0, 0, b"spawn_pose")),
        # The fold takes an UNSIGNED 64-bit key; a signed leftover or an
        # oversized value is a wiring mistake, not something to wrap silently.
        (ValueError, lambda: as_torch_seed(-1)),
        (ValueError, lambda: as_torch_seed(1 << 64)),
        (TypeError, lambda: as_torch_seed(1.5)),
        (TypeError, lambda: as_torch_seed(True)),
        (TypeError, lambda: scenario_hash(["not", "a", "mapping"])),
        (TypeError, lambda: scenario_hash({"weird": {1, 2}})),
    ]
    for expected, call in cases:
        try:
            call()
        except expected:
            continue
        except Exception as exc:  # noqa: BLE001 - wrong error type is a failure
            raise AssertionError(f"expected {expected.__name__}, got {exc!r}")
        raise AssertionError(f"expected {expected.__name__}, nothing raised")


# ----------------------------------------------------------- scenario hash --
def test_scenario_hash_is_pinned_and_order_invariant():
    assert scenario_hash(PINNED_PARAMS) == "d0e3d7d8f05a51bb"
    shuffled = {k: PINNED_PARAMS[k] for k in reversed(list(PINNED_PARAMS))}
    assert scenario_hash(shuffled) == scenario_hash(PINNED_PARAMS)
    nested_a = {"wave": {"hs_m": 0.6, "tp_s": 5.0}, "spawn": {"d_m": 4.25}}
    nested_b = {"spawn": {"d_m": 4.25}, "wave": {"tp_s": 5.0, "hs_m": 0.6}}
    assert scenario_hash(nested_a) == scenario_hash(nested_b)
    assert len(scenario_hash(PINNED_PARAMS)) == 16
    assert len(scenario_hash(PINNED_PARAMS, digest_size=4)) == 8


def test_scenario_hash_moves_for_every_parameter():
    base = scenario_hash(PINNED_PARAMS)
    for key in PINNED_PARAMS:
        changed = dict(PINNED_PARAMS)
        changed[key] = PINNED_PARAMS[key] + 1e-6
        assert scenario_hash(changed) != base, key
    # Structure changes too: extra key, dropped key, renamed key.
    assert scenario_hash({**PINNED_PARAMS, "extra": 0.0}) != base
    trimmed = {k: v for k, v in PINNED_PARAMS.items() if k != "n"}
    assert scenario_hash(trimmed) != base
    # Types are not conflated, and sequence ORDER is information.
    assert scenario_hash({"x": 1}) != scenario_hash({"x": 1.0})
    assert scenario_hash({"x": 1}) != scenario_hash({"x": True})
    assert scenario_hash({"x": [1.0, 2.0]}) != scenario_hash({"x": [2.0, 1.0]})
    # A real per-primitive comparison: identical spawn, different wave.
    run_a = {"spawn": {"d_m": 4.25}, "wave": {"hs_m": 0.60}}
    run_b = {"spawn": {"d_m": 4.25}, "wave": {"hs_m": 0.61}}
    assert scenario_hash(run_a["spawn"]) == scenario_hash(run_b["spawn"])
    assert scenario_hash(run_a["wave"]) != scenario_hash(run_b["wave"])


def test_hash_survives_the_tensor_and_numpy_types_certificates_carry():
    import numpy as np

    plain = {"d_m": 4.25, "angle": [0.5, 1.5]}
    fancy = {
        "d_m": torch.tensor(4.25, dtype=torch.float64),
        "angle": np.array([0.5, 1.5]),
    }
    assert scenario_hash(plain) == scenario_hash(fancy)


# ----------------------------------------------------------------- hygiene --
def test_sources_are_pure_ascii_without_control_characters():
    for path in (SOURCE, Path(__file__).resolve()):
        text = path.read_text(encoding="utf-8")
        bad = [
            (index, character)
            for index, character in enumerate(text)
            if ord(character) > 126 or (ord(character) < 32 and character not in "\n\r")
        ]
        assert not bad, (path.name, bad[:5])


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
