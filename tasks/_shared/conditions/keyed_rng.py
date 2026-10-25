"""Counter-based keyed random draws, generated in batches (E3).

Protocol ``usvbench.keyed_counter.v1``. Used by the hydro_v2 condition layer
(wind gust innovations, perception degradations). Pure Python/NumPy/Torch,
no simulator import; safe to load by file path.

Keys. A stream is identified by (eval seed, logical env key, scenario episode
index, stream name). Its 64-bit key is the SHARED scenario seed-folding
protocol ``scenario_rng.stable_stream_seed(protocol_version, eval_seed, env,
episode, group_name)`` (BLAKE2b over a fixed-width encoding) with the group
name ``"cond3." + stream``. The key is derived ONCE per (env, episode,
stream) at reset: never per step.

Draws. Element ``c`` (c = 0, 1, 2, ...) of a stream is the c-th output of the
standard SplitMix64 sequence started at the key:
    x = key + (c + 1) * 0x9E3779B97F4A7C15            (mod 2^64)
    z = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
A uniform is ((z >> 11) + 0.5) / 2^53, strictly inside (0, 1). The normal
with index j uses uniforms 2j and 2j+1 (Box-Muller cosine branch). Values are
pure functions of (key, counter): reset order, batching, chunk size, other
envs and any global RNG cannot change them, and no global RNG is touched.

Backend. uint64 arithmetic wraps modulo 2^64 in NumPy (defined behaviour);
the float64 transcendental functions run on the CPU through NumPy. Bitwise
equality across CPU platforms / NumPy builds is UNVERIFIED (libm); within one
installation the streams are exactly reproducible.

``KeyedNormalTape`` materialises a stream in chunks of ``chunk_steps`` steps
(``width`` normals per step) per environment, so a per-physics-step or
per-control-step caller only gathers from a device tensor; refills are
vectorised NumPy over every env whose chunk ran out (no per-env Python
hashing, no per-step Python loop over environments).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from ..scenario_rng import SCENARIO_PROTOCOL_VERSION, stable_stream_seed
except ImportError:  # loaded by file path (CPU tests, wind.py fallback)
    _name = "_usvbench_keyed_rng_scenario_rng"
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            _name, Path(__file__).resolve().parents[1] / "scenario_rng.py")
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _module
        _spec.loader.exec_module(_module)
    SCENARIO_PROTOCOL_VERSION = sys.modules[_name].SCENARIO_PROTOCOL_VERSION
    stable_stream_seed = sys.modules[_name].stable_stream_seed

PROTOCOL = "usvbench.keyed_counter.v1"
GROUP_PREFIX = "cond3."
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_TWO53 = float(1 << 53)


def stream_key(eval_seed: int, env_key: int, episode_index: int, stream: str,
               protocol_version: int = SCENARIO_PROTOCOL_VERSION) -> int:
    """Unsigned 64-bit key of one (seed, env, episode, stream) stream."""
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream name must be a non-empty string")
    if int(episode_index) < 0:
        raise ValueError("stream keys need a started episode (index >= 0)")
    return stable_stream_seed(int(protocol_version), int(eval_seed), int(env_key),
                              int(episode_index), GROUP_PREFIX + stream)


def splitmix64(keys: np.ndarray, counters: np.ndarray) -> np.ndarray:
    """Element ``counters`` of the SplitMix64 sequences started at ``keys``.

    keys and counters broadcast; both are converted to uint64. Wraps mod 2^64.
    """
    k = np.asarray(keys, dtype=np.uint64)
    c = np.asarray(counters, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = k + (c + np.uint64(1)) * _GOLDEN
        z = (x ^ (x >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def uniforms(keys: np.ndarray, counters: np.ndarray) -> np.ndarray:
    """float64 uniforms strictly inside (0, 1)."""
    z = splitmix64(keys, counters)
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) / _TWO53


def normals(keys: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """float64 standard normals; normal j uses uniforms 2j and 2j+1."""
    j = np.asarray(indices, dtype=np.uint64)
    u1 = uniforms(keys, j * np.uint64(2))
    u2 = uniforms(keys, j * np.uint64(2) + np.uint64(1))
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


class KeyedNormalTape:
    """Chunked per-env normal stream: ``width`` normals per step.

    Step ``s`` lane ``l`` of env ``i`` is ``normals(key_i, s * width + l)``.
    ``reset`` installs new keys (one per reset env) and prefetches steps
    [0, chunk_steps); ``gather`` returns the normals for per-env step indices
    and refills (vectorised) only the envs whose chunk does not cover them.
    Host bookkeeping is NumPy on the CPU; the tape lives on ``device``.
    """

    def __init__(self, num_envs: int, width: int, device, dtype=torch.float32,
                 chunk_steps: int = 1024):
        if int(num_envs) <= 0 or int(width) <= 0 or int(chunk_steps) <= 0:
            raise ValueError("num_envs, width and chunk_steps must be positive")
        self.num_envs, self.width, self.chunk = int(num_envs), int(width), int(chunk_steps)
        self.device, self.dtype = torch.device(device), dtype
        self.keys = np.zeros(self.num_envs, dtype=np.uint64)
        self.has_key = np.zeros(self.num_envs, dtype=bool)
        self.start = np.zeros(self.num_envs, dtype=np.int64)
        self.tape = torch.zeros((self.num_envs, self.chunk, self.width), device=self.device,
                                dtype=self.dtype)
        self.generated = 0          # normals generated so far (diagnostic)
        self.refills = 0            # vectorised refill calls (diagnostic)

    def _fill(self, ids: np.ndarray, starts: np.ndarray) -> None:
        if ids.size == 0:
            return
        steps = starts[:, None] + np.arange(self.chunk, dtype=np.int64)[None, :]
        index = (steps[:, :, None] * self.width
                 + np.arange(self.width, dtype=np.int64)[None, None, :])
        values = normals(self.keys[ids][:, None, None], index)
        self.tape[torch.as_tensor(ids, dtype=torch.long, device=self.device)] = torch.as_tensor(
            values, dtype=self.dtype).to(self.device)
        self.start[ids] = starts
        self.generated += int(values.size)
        self.refills += 1

    def reset(self, ids, keys) -> None:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        keys = np.asarray([int(k) for k in keys], dtype=np.uint64).reshape(-1)
        if ids.size != keys.size:
            raise ValueError("one key per reset env is required")
        if ids.size == 0:
            return
        if np.unique(ids).size != ids.size or ids.min() < 0 or ids.max() >= self.num_envs:
            raise ValueError("reset ids must be unique and in bounds")
        self.keys[ids] = keys
        self.has_key[ids] = True
        self._fill(ids, np.zeros(ids.size, dtype=np.int64))

    def gather(self, ids, steps) -> torch.Tensor:
        """Normals [len(ids), width] at per-env step indices (host int64)."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        steps = np.asarray(steps, dtype=np.int64).reshape(-1)
        if ids.size != steps.size:
            raise ValueError("one step index per env is required")
        if ids.size == 0:
            return torch.zeros((0, self.width), device=self.device, dtype=self.dtype)
        if not bool(self.has_key[ids].all()):
            raise RuntimeError("keyed stream used before its episode reset")
        if bool((steps < 0).any()):
            raise ValueError("step indices must be nonnegative")
        outside = (steps < self.start[ids]) | (steps >= self.start[ids] + self.chunk)
        if bool(outside.any()):
            self._fill(ids[outside], steps[outside])
        column = steps - self.start[ids]
        row_t = torch.as_tensor(ids, dtype=torch.long).to(self.device)
        col_t = torch.as_tensor(column, dtype=torch.long).to(self.device)
        return self.tape[row_t, col_t]


__all__ = ["PROTOCOL", "GROUP_PREFIX", "stream_key", "splitmix64", "uniforms", "normals",
           "KeyedNormalTape"]
