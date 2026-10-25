"""G11: the deferred EpisodePathObserver equals the eager T-base v2.3 observer bit for bit.

REFERENCE below is the T-base v2.3 scripts/episode_path_observer.py class verbatim (the eager
observer). Both observers watch twin fake envs fed the same randomized control steps (tensor
inputs: the deferred device path; list inputs: the eager path; mixed sequences; NaN / inf /
float masks / [n, 1] masks / no success latch / autoresets / take between steps / attribute
reads / forced capacity replays / after_reset mid-run / a second observer on the same env).
Every take() result, every replayed state attribute and the wrapper's return object must be
identical (floats compared by their bits). A per-step synchronisation probe checks that the
deferred path performs no host transfer per control step.
Run: python -B -m pytest -p no:cacheprovider scripts/test_g11_path_observer.py -q
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import random
import struct
import sys
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("g11_episode_path_observer", ROOT / "episode_path_observer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paths = _load()


# ------------------------------------------------- T-base v2.3 reference --
def _ref_flags(values, n):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "reshape") and not isinstance(values, (list, tuple)):
        values = values.reshape(-1)
    flat = values.tolist() if hasattr(values, "tolist") else list(values)
    flat = [v[0] if isinstance(v, (list, tuple)) else v for v in flat]
    if len(flat) != n:
        raise ValueError("mask does not have one entry per environment")
    return [bool(v) for v in flat]


class REFERENCE:
    def __init__(self, base):
        self.base = base
        self.n = int(base.num_envs)
        self.previous = None
        self.full = [0.0] * self.n
        self.prefix = [None] * self.n
        self.completed = {}
        self.pending = set()
        original = getattr(base, "_usvbench_original_get_dones", None)
        if original is None:
            original = getattr(base, "_get_dones", None)
            if callable(original):
                base._usvbench_original_get_dones = original
        self.enabled = callable(original) and self._positions() is not None
        if self.enabled:
            def observed_dones(*args, **kwargs):
                result = original(*args, **kwargs)
                self.observe(result)
                return result
            base._get_dones = observed_dones

    def _positions(self):
        robot = getattr(self.base, "robot", None)
        data = getattr(robot, "data", None)
        positions = getattr(data, "root_pos_w", None)
        if positions is None:
            return None
        if hasattr(positions, "detach"):
            positions = positions.detach()
        values = positions.tolist() if hasattr(positions, "tolist") else positions
        return [(float(row[0]), float(row[1])) for row in values]

    def after_reset(self):
        self.previous = self._positions()
        self.full = [0.0] * self.n
        self.prefix = [None] * self.n
        self.completed.clear()
        self.pending.clear()

    def observe(self, dones):
        if self.previous is None:
            return
        positions = self._positions()
        success = getattr(self.base, "_success", None)
        success = _ref_flags(success, self.n) if success is not None else None
        term, trunc = dones
        term, trunc = _ref_flags(term, self.n), _ref_flags(trunc, self.n)
        for i, (before, after) in enumerate(zip(self.previous, positions)):
            distance = math.hypot(after[0] - before[0], after[1] - before[1])
            self.full[i] += distance
            if self.prefix[i] is None and success is not None and success[i]:
                self.prefix[i] = self.full[i]
            if term[i] or trunc[i]:
                self.completed[i] = (self.full[i], self.prefix[i])
                self.pending.add(i)
        self.previous = positions

    def after_step(self):
        positions = self._positions()
        if positions is not None and self.previous is not None:
            for i in self.pending:
                self.previous[i] = positions[i]
                self.full[i] = 0.0
                self.prefix[i] = None
        self.pending.clear()

    def take(self, env, success):
        full, prefix = self.completed.pop(env, (None, None))
        if not success:
            prefix = None
        return {"path_length_full_episode_m": full if full is None or math.isfinite(full) else None,
                "path_length_success_prefix_m": prefix if prefix is None or math.isfinite(prefix) else None}


# ------------------------------------------------------------- helpers --
def bits(value):
    """Exact identity of floats (bits), None, tuples, lists, dicts, sets."""
    if isinstance(value, float):
        return ("f", struct.pack("<d", value))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(bits(v) for v in value))
    if isinstance(value, dict):
        return ("d", tuple(sorted((k, bits(v)) for k, v in value.items())))
    if isinstance(value, set):
        return ("s", tuple(sorted(value)))
    return ("o", value)


class FakeEnv:
    def __init__(self, n, tensors, dtype=torch.float32, success=True, mask_shape="flat", mask_dtype=torch.bool):
        self.num_envs = n
        self.tensors, self.dtype, self.mask_shape, self.mask_dtype = tensors, dtype, mask_shape, mask_dtype
        self.robot = NS(data=NS(root_pos_w=self._pos([[0.0, 0.0, 0.0]] * n)))
        self._success = self._mask([False] * n) if success else None
        if not success:
            del self._success
        self.next_dones = ([False] * n, [False] * n)

    def _pos(self, rows):
        return torch.tensor(rows, dtype=self.dtype) if self.tensors else [list(r) for r in rows]

    def _mask(self, values):
        if not self.tensors:
            return list(values)
        t = torch.tensor(values, dtype=self.mask_dtype)
        return t.reshape(-1, 1) if self.mask_shape == "column" else t

    def _get_dones(self):
        term, trunc = self.next_dones
        return self._mask(term), self._mask(trunc)


def drive(n, steps, seed, *, tensors=True, mixed=False, dtype=torch.float32, success=True, mask_shape="flat",
          mask_dtype=torch.bool, capacity=None, reads=True, nonfinite=True):
    rng = random.Random(seed)
    envs = [FakeEnv(n, tensors, dtype, success, mask_shape, mask_dtype) for _ in range(2)]
    ref, new = REFERENCE(envs[0]), paths.EpisodePathObserver(envs[1])
    if capacity is not None:
        paths.LOG_CAPACITY = capacity
    try:
        assert ref.enabled == new.enabled
        for obs in (ref, new):
            obs.after_reset()
        xy = [[rng.uniform(-50, 50), rng.uniform(-50, 50)] for _ in range(n)]
        latch = [False] * n
        outputs = [[], []]
        for k in range(steps):
            use_tensors = tensors if not mixed else rng.random() < 0.7
            for e in envs:
                e.tensors = use_tensors
            for i in range(n):
                xy[i][0] += rng.gauss(0.0, 0.3)
                xy[i][1] += rng.gauss(0.0, 0.3)
                if nonfinite and rng.random() < 0.01:
                    xy[i][rng.randrange(2)] = rng.choice([math.nan, math.inf, -math.inf])
                if rng.random() < 0.05:
                    latch[i] = True
            term = [rng.random() < 0.04 for _ in range(n)]
            trunc = [rng.random() < 0.02 for _ in range(n)]
            rows = [[x, y, 0.5] for x, y in xy]
            vals = latch if mask_dtype == torch.bool else [rng.choice([0.5, -1e-300, 2.0]) if v else
                                                           rng.choice([0.0, -0.0]) for v in latch]
            for e in envs:
                e.robot.data.root_pos_w = e._pos(rows)
                if success:
                    e._success = e._mask(vals)
                e.next_dones = (term, trunc)
            results = [e._get_dones() for e in envs]
            for e, r in zip(envs, results):
                assert isinstance(r, tuple)
            # autoreset: teleport finished rows, clear their latch
            for i in range(n):
                if term[i] or trunc[i]:
                    xy[i] = [rng.uniform(-50, 50), rng.uniform(-50, 50)]
                    latch[i] = False
            rows = [[x, y, 0.5] for x, y in xy]
            for e in envs:
                e.robot.data.root_pos_w = e._pos(rows)
            ref.after_step()
            new.after_step()
            for i in range(n):
                if (term[i] or trunc[i]) and rng.random() < 0.8:
                    flag = rng.random() < 0.5
                    outputs[0].append(bits(ref.take(i, flag)))
                    outputs[1].append(bits(new.take(i, flag)))
            if reads and rng.random() < 0.05:
                for name in ("previous", "full", "prefix", "completed", "pending"):
                    assert bits(getattr(ref, name)) == bits(getattr(new, name)), (k, name)
            if rng.random() < 0.01:
                for obs in (ref, new):
                    obs.after_reset()
        for i in range(n):
            outputs[0].append(bits(ref.take(i, True)))
            outputs[1].append(bits(new.take(i, True)))
        for name in ("previous", "full", "prefix", "completed", "pending"):
            assert bits(getattr(ref, name)) == bits(getattr(new, name)), name
        return outputs
    finally:
        paths.LOG_CAPACITY = 1024


@pytest.mark.parametrize("seed", range(6))
def test_deferred_tensor_path_equals_eager_reference(seed):
    a, b = drive(7, 400, seed)
    assert a == b and len(a) > 20


@pytest.mark.parametrize("seed", range(3))
def test_list_inputs_take_the_eager_path_and_match(seed):
    a, b = drive(5, 250, 100 + seed, tensors=False)
    assert a == b


@pytest.mark.parametrize("seed", range(4))
def test_mixed_tensor_and_list_steps_match(seed):
    a, b = drive(6, 300, 200 + seed, mixed=True)
    assert a == b


def test_float64_positions_float_masks_column_masks_and_capacity_replays():
    a, b = drive(4, 300, 7, dtype=torch.float64, mask_dtype=torch.float64, mask_shape="column", capacity=3)
    assert a == b
    a, b = drive(4, 300, 8, mask_dtype=torch.int64, capacity=1)
    assert a == b


def test_env_without_success_latch():
    a, b = drive(5, 300, 9, success=False)
    assert a == b


def test_no_attribute_reads_until_the_end():
    a, b = drive(64, 200, 10, reads=False)
    assert a == b


def test_wrapper_returns_the_original_object_and_second_observer_replaces_the_first():
    env = FakeEnv(3, True)
    expected = (torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
    env._get_dones = lambda: expected
    first = paths.EpisodePathObserver(env)
    second = paths.EpisodePathObserver(env)
    assert first.enabled and second.enabled
    second.after_reset()
    assert env._get_dones() is expected
    assert env._usvbench_original_get_dones is not env._get_dones
    assert len(second._log) == 1 and first._log == []


def test_mask_length_error_is_raised_at_the_step_like_the_eager_observer():
    env = FakeEnv(3, True)
    obs = paths.EpisodePathObserver(env)
    obs.after_reset()
    env.next_dones = ([False] * 3, [False] * 3)
    env._success = torch.zeros(4, dtype=torch.bool)
    with pytest.raises(ValueError, match="one entry per environment"):
        env._get_dones()


def test_deferred_step_moves_nothing_to_the_host():
    """Per control step: no tolist/item/cpu/bool on any tensor (the CUDA sync calls)."""
    env = FakeEnv(8, True)
    obs = paths.EpisodePathObserver(env)
    obs.after_reset()
    calls = []

    class Probe(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", "")
            if name in ("tolist", "item", "cpu", "numpy", "__bool__", "__float__", "__int__", "__index__",
                        "nonzero"):
                calls.append(name)
            return func(*args, **(kwargs or {}))

    with Probe():
        for k in range(50):
            env.robot.data.root_pos_w = torch.randn(8, 3)
            env.next_dones = ([k % 9 == 0] + [False] * 7, [False] * 8)
            env._get_dones()
            obs.after_step()
    assert calls == []
    assert len(obs._log) == 100
    obs.take(0, False)
    assert obs._log == []


def test_scalar_replay_when_the_state_is_not_n_rows_and_mixed_position_dtypes():
    """after_reset sees n+1 list rows (eager), the steps hand n-row tensors (deferred, float32 then float64):
    the replay falls back to the eager loop (zip truncation) and widens mixed dtypes exactly."""
    n = 3
    envs = [FakeEnv(n, False) for _ in range(2)]
    for e in envs:
        e.robot.data.root_pos_w = [[0.0, 0.0, 0.0]] * (n + 1)
    ref, new = REFERENCE(envs[0]), paths.EpisodePathObserver(envs[1])
    ref.after_reset()
    new.after_reset()
    rng = random.Random(1)
    for k in range(12):
        rows = [[rng.uniform(-5, 5), rng.uniform(-5, 5), 0.0] for _ in range(n)]
        dtype = torch.float32 if k % 2 else torch.float64
        term = [k % 5 == 4, False, k == 7]
        for e in envs:
            e.tensors = True
            e.robot.data.root_pos_w = torch.tensor(rows, dtype=dtype)
            e._success = torch.tensor([k > 6, False, k > 3])
            e.next_dones = (term, [False] * n)
            e._get_dones()
        ref.after_step()
        new.after_step()
    for name in ("previous", "full", "prefix", "completed", "pending"):
        assert bits(getattr(ref, name)) == bits(getattr(new, name)), name
    assert [bits(ref.take(i, True)) for i in range(n)] == [bits(new.take(i, True)) for i in range(n)]


def test_state_setters_replay_first():
    env = FakeEnv(2, True)
    obs = paths.EpisodePathObserver(env)
    obs.after_reset()
    env.robot.data.root_pos_w = torch.tensor([[3.0, 4.0, 0.0], [0.0, 1.0, 0.0]])
    env._get_dones()
    assert len(obs._log) == 1
    obs.full = [10.0, 20.0]
    assert obs._log == [] and obs.full == [10.0, 20.0]
    obs.prefix = [None, 1.0]
    obs.previous = [(0.0, 0.0), (0.0, 0.0)]
    assert obs.prefix == [None, 1.0] and obs.previous == [(0.0, 0.0), (0.0, 0.0)]
