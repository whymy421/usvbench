# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible, controller-independent episode scenarios for evaluation.

WHY THIS EXISTS
---------------
An evaluation is supposed to be two independent exam papers: the same policy
sat at ``--eval-seed 123`` and at ``--eval-seed 42``.  For several task
families the two papers came out byte-identical episode for episode, which
means they were ONE measurement reported twice.

The chain (the two middle hops live in the installed packages, not in this
tree: ``isaaclab/envs/direct_rl_env.py:99-100`` and
``skrl/utils/runner/torch/runner.py:29``):

* ``scripts/eval_v6_frozen.py:74`` sets ``env_cfg.seed = args_cli.eval_seed``,
  and Isaac Lab's ``DirectRLEnv`` seeds the GLOBAL torch RNG from ``cfg.seed``
  while the env is being constructed.
* AFTER construction the evaluator builds an skrl ``Runner``, whose
  ``__init__`` calls ``set_seed`` with the agent YAML's seed; that YAML pins a
  constant (``tasks/station_keeping/agents/skrl_ppo_cfg.yaml:1`` is literally
  ``seed: 42``).  The evaluation seed is thereby discarded.
* Families that draw their scenario from the global torch RNG are therefore
  frozen to that constant.  Station keeping draws spawn distance, spawn angle
  and heading with bare ``torch.rand``
  (``tasks/station_keeping/station_keeping_env.py:684-688``), its current with
  bare ``torch.rand`` (``:367,:373``), and the shared sea state draws H_s, T_p,
  gamma, mean direction, phases and spreading the same way
  (``tasks/_shared/sea_state.py:106-118``).
* Families that own a generator seeded from ``cfg.seed`` were immune -- e.g.
  ``tasks/hazard_nav/hazard_nav_env.py:95-96`` builds
  ``np.random.default_rng(layout_seed)`` from ``cfg.seed``, and
  ``tasks/docking/docking_env.py:45-48`` keeps a per-env CPU
  ``torch.Generator`` for exactly this reason.

FOUR PROPERTIES THIS MODULE GUARANTEES
--------------------------------------
(a) Seed-derived.  Every draw descends from the EVALUATION seed, so two eval
    seeds are two genuinely different exam papers.
(b) Controller-independent.  The scenario is keyed by
    ``(protocol_version, eval_seed, env_index, episode_index, group)`` and the
    stream for that key is (re)seeded at the START of the episode.  Two
    different controllers evaluated at the same eval seed get the SAME
    scenario for the same (env, episode) even though they end episodes at
    different times and consume different numbers of draws.  A single
    per-env generator merely advanced by resets does NOT have this property.
(c) Process-RNG independent.  Nothing here reads or writes torch's global
    generator, numpy's legacy global state, or ``random``.  A downstream
    library (skrl, an agent, a wrapper) calling ``set_seed`` cannot move it.
(d) Substream-isolated.  Each random primitive (spawn pose, current, wave,
    actuator/dynamics, observation degradation, ...) owns its own stream, so
    adding a draw to one primitive cannot shift another primitive's values.

Group names are hashed as TEXT, not as a positional index.  This is a
deliberate difference from ``tasks/_shared/obs_degradation.py:163``, which
assigns ``group_id`` by ``enumerate(sorted(groups))``: there, registering a
sixth group renumbers the others and silently changes every stream.  Here,
adding a group perturbs nothing that already existed -- which is also what
makes property (b) testable (one controller may touch a group the other never
touches).

HASHING
-------
``hashlib.blake2b`` (digest_size=8), NOT Python's builtin ``hash()``.
``hash()`` is salted per process for str/bytes (PYTHONHASHSEED), so a
builtin-hash key would silently produce a different exam paper in every
process -- the exact class of bug this module exists to kill.  blake2b is in
the standard library, is stable across processes, platforms, Python versions
and machines, and needs no third-party dependency.  ``numpy.SeedSequence``
would also have been stable; blake2b was chosen because the same primitive
then serves the certificate scenario hash, and because the key encoding stays
explicit and byte-exact (see ``_pack_int``/``_pack_text``).  SeedSequence
still does one job here -- handing the key to torch -- for the reason in the
next section.

TORCH SEED WIDTH  (why the key is 64 bits but the stream space is 2**32)
------------------------------------------------------------------------
``torch.Generator.manual_seed`` ACCEPTS a 64-bit value and then keeps only
the low 32: on torch 2.9.1 the seeds ``0x00000000ABCDEF01`` and
``0x12345678ABCDEF01`` drive the SAME CPU MT19937 stream.  An earlier
revision of this module handed ``stable_stream_seed``'s raw 64-bit digest
straight to ``manual_seed``, so bits 32..63 of every key -- carrying a large
slice of the eval seed, the env index and the episode index -- fell off the
end in silence, and any two keys agreeing in their low 32 bits sat the
identical exam paper.

``as_torch_seed`` now folds the whole key through
``numpy.random.SeedSequence(key, ...).generate_state(1, dtype=uint32)``,
whose hashmix consumes full-width entropy, so every bit of every key field
(protocol version, eval seed, env index, episode index, group name) reaches
the drawn values.  SeedSequence was preferred over simply shortening the
digest to ``blake2b(digest_size=4)``: both give the same 2**32 torch stream
space, but the fold leaves ``stable_stream_seed`` 64 bits wide, so the
certificate still records the full-width key and ``numpy_generator`` still
seeds PCG64 from all 64 bits.  Only the torch handoff -- where 32 bits is
torch's own hard limit, not a choice this module makes -- is narrowed.
SeedSequence is also the primitive the numpy path here already trusts, so
this adds no dependency and no second cross-version stability promise.
(Writing a full MT19937 state with ``Generator.set_state`` would keep all 64
bits, but that state layout is a torch internal; betting certificate
reproducibility on it is a worse trade than a 2**32 stream space.)

Honest consequence -- the torch stream space is 2**32, not 2**64, so distinct
keys are distinct streams only up to the birthday bound.  For n streams the
chance that some two share a seed is about n(n-1)/2 / 2**32:

* one 128-episode certificate over five primitive groups is about 640
  streams -> ~4.8e-5, roughly 1 in 21,000;
* the 1000-stream round number used for planning -> ~1.2e-4, 1 in 8,600;
* an entire 79-pair census, ~1e5 streams -> about one coincidental pair
  expected somewhere in the suite.

What a collision costs: two ``(env, episode, group)`` keys draw the same
numbers, i.e. one duplicated scenario.  It does NOT make two eval seeds look
alike -- that verdict would need EVERY stream in the certificate to collide
at once -- and a cross-certificate collision links two unrelated keys in two
different runs, which changes no measurement.  If 2**32 ever stops being
enough, bump ``SCENARIO_PROTOCOL_VERSION`` and move the draws onto a numpy
Generator, which takes the full key.

The fold changes every number this module has ever produced.  That is
deliberate and free: no certificate has been issued under protocol 2, so
version 2 is corrected in place rather than retired -- the same call taken
when the protocol first replaced the global-RNG draws.

DEVICE POLICY
-------------
Generators are CPU generators by default and they fill CPU tensors; the result
is copied to the consumer device afterwards.  So the invariant "the generator
lives on the same device as the tensor it fills" always holds, while the
BYTES are the CPU MT19937 stream -- identical on a 4080, a 5070, or a laptop
with no GPU at all.  ``torch.Generator(device="cuda")`` does exist, but

  * a CUDA generator passed to an op filling a CPU tensor (or vice versa)
    raises "Expected a 'cuda' device type for generator but found 'cpu'", and
  * the CUDA Philox stream is not guaranteed to give the same numbers across
    devices or launch configurations, which would break cross-machine
    certificate comparison.

Scenario draws happen only on reset and are a handful of floats per env, so
the host-to-device copy is free.  ``generator_device="cuda"`` is available for
callers who knowingly want device-resident generators; the draw path then
allocates on that same device, preserving the invariant.

Draws are made with an explicit ``dtype`` (default ``torch.float32``) so that
a global ``torch.set_default_dtype`` cannot change the byte stream.

USAGE (env side)
----------------
    self._scenario = ScenarioRNG(
        num_envs=self.num_envs, device=self.device, eval_seed=int(cfg.seed)
    )
    ...
    def _reset_idx(self, env_ids):
        self._scenario.reset_idx(env_ids)          # new episode for these envs
        d = self._scenario.uniform("spawn_pose", env_ids, d_min, d_max)
        a = self._scenario.uniform("spawn_pose", env_ids, 0.0, 2 * math.pi)

and record ``SCENARIO_PROTOCOL_VERSION`` plus ``scenario_hash({...})`` in the
certificate so two runs can be diffed primitive by primitive.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

# Protocol version 1 is the historical BROKEN behaviour: scenarios drawn from
# the global torch RNG, which skrl's Runner reseeds to a constant after the
# env is built.  Version 2 is this module.  Callers must record the version
# they ran under; results certified under different versions are not
# episode-comparable, only aggregate-comparable.
SCENARIO_PROTOCOL_VERSION = 2

PROTOCOL_NOTES = {
    1: "global torch RNG; eval seed clobbered by skrl Runner set_seed",
    2: "per (eval_seed, env, episode, group) blake2b stream; skrl-proof",
}

# Canonical primitive groups for the USV tasks.  Any string is a legal group
# name -- this tuple is a naming convention, not a whitelist.
SCENARIO_GROUPS = (
    "spawn_pose",
    "current",
    "wave",
    "actuator",
    "observation",
)

# Domain separation.  Changing this byte string changes every scenario in the
# benchmark, so it is frozen: bump SCENARIO_PROTOCOL_VERSION instead.
_KEY_NAMESPACE = b"usvbench/scenario_rng/key/v1"
_HASH_NAMESPACE = b"usvbench/scenario_rng/scenario-hash/v1"

# Domain tag for the torch fold.  ``numpy_generator`` seeds PCG64 from
# ``SeedSequence(key)`` directly, and ``generate_state(1)`` is a PREFIX of the
# state PCG64 consumes -- so without a spawn key the torch seed for a group
# would literally be the first 32 bits of that same material.  A fixed spawn
# key puts the two derivations on separate branches of the SeedSequence tree.
# Frozen like the namespaces above: changing it changes every scenario.
_TORCH_SPAWN_KEY = (0x53434E52,)  # b"SCNR"

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_TWO64 = 1 << 64


# --------------------------------------------------------------------- keys --
def _pack_int(name: str, value: Any) -> bytes:
    """Fixed-width, byte-exact encoding of one signed 64-bit key field."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    as_int = int(value)
    if not _INT64_MIN <= as_int <= _INT64_MAX:
        raise ValueError(f"{name}={as_int} does not fit in a signed 64-bit int")
    return as_int.to_bytes(8, "little", signed=True)


def _pack_text(value: str) -> bytes:
    """Length-prefixed UTF-8, so no group name can alias another key field."""
    if not isinstance(value, str):
        raise TypeError(f"group name must be a str, got {type(value).__name__}")
    if not value:
        raise ValueError("group name must not be empty")
    raw = value.encode("utf-8")
    return len(raw).to_bytes(4, "little", signed=False) + raw


def stable_stream_seed(
    protocol_version: int,
    eval_seed: int,
    env_index: int,
    episode_index: int,
    group_name: str,
) -> int:
    """Stable unsigned 64-bit stream seed for one scenario key.

    Stable means: same value in every process, on every platform, in every
    Python build, forever.  It is a blake2b digest of a fixed-width encoding
    of the key -- never Python's salted builtin ``hash()``.
    """
    payload = b"".join(
        (
            _KEY_NAMESPACE,
            _pack_int("protocol_version", protocol_version),
            _pack_int("eval_seed", eval_seed),
            _pack_int("env_index", env_index),
            _pack_int("episode_index", episode_index),
            _pack_text(group_name),
        )
    )
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def as_torch_seed(seed_u64: int) -> int:
    """Fold a 64-bit stream key into the 32-bit seed torch actually uses.

    NOT a bijection, and it cannot be one: ``torch.Generator.manual_seed``
    takes a 64-bit argument and keeps only the low 32 bits (see TORCH SEED
    WIDTH in the module docstring).  The job here is to make the 32 bits torch
    keeps depend on ALL 64 bits of the key, instead of letting bits 32..63
    fall off the end unnoticed -- which is exactly what the predecessor of
    this function did.

    numpy's ``SeedSequence`` hashmix consumes full-width entropy, so a one-bit
    change anywhere in the key changes the returned seed -- with probability
    1 - 2**-32, not certainty, because 64 bits cannot inject into 32; every
    single-bit flip of several real keys is checked in
    ``test_every_high_key_bit_reaches_the_drawn_values``.  The result lies in
    ``[0, 2**32)``, which every torch build accepts, so the portability the
    old two's-complement mapping bought is preserved for free.
    """
    if isinstance(seed_u64, (bool, np.bool_)) or not isinstance(
        seed_u64, (int, np.integer)
    ):
        raise TypeError(f"stream seed must be an integer, got {seed_u64!r}")
    key = int(seed_u64)
    if not 0 <= key < _TWO64:
        raise ValueError(f"stream seed {key} is not an unsigned 64-bit value")
    sequence = np.random.SeedSequence(key, spawn_key=_TORCH_SPAWN_KEY)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def numpy_generator(
    protocol_version: int,
    eval_seed: int,
    env_index: int,
    episode_index: int,
    group_name: str,
) -> np.random.Generator:
    """A numpy Generator on the same key, for envs whose layouts are numpy.

    ``tasks/hazard_nav/hazard_nav_env.py:96`` builds its layout RNG with
    ``np.random.default_rng``; this gives such code the same key algebra
    without forcing it onto torch.

    This path keeps the full 64-bit key -- SeedSequence feeds PCG64 all of it,
    so the 2**32 ceiling described under TORCH SEED WIDTH applies to the torch
    generators only.  The bare ``SeedSequence(seed)`` here and the spawn-keyed
    one inside ``as_torch_seed`` are different branches on purpose, so a group
    used from both sides does not run two correlated streams.
    """
    seed = stable_stream_seed(
        protocol_version, eval_seed, env_index, episode_index, group_name
    )
    return np.random.default_rng(np.random.SeedSequence(seed))


# ---------------------------------------------------------------- the class --
class ScenarioRNG:
    """One torch.Generator per (env, group), reseeded at episode start.

    See the module docstring for the four properties this enforces.  The draw
    methods loop over env indices on purpose: a per-env stream must not depend
    on which OTHER envs happen to be resetting in the same call, so a single
    vectorised draw across the batch is not an option.  Resets are rare and
    ``num_envs`` is small (64 in the certified evaluator), so the loop costs
    nothing that matters.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        eval_seed: int,
        groups: Iterable[str] | None = None,
        protocol_version: int = SCENARIO_PROTOCOL_VERSION,
        generator_device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if int(num_envs) != num_envs or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if int(protocol_version) != protocol_version or protocol_version < 1:
            raise ValueError("protocol_version must be a positive integer")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.generator_device = torch.device(generator_device)
        self.dtype = dtype
        self.eval_seed = int(eval_seed)
        self.protocol_version = int(protocol_version)
        # Episode counters live on the CPU: they are read one int at a time
        # while reseeding, and -1 means "this env has never been reset".
        self._episode = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._generators: dict[str, list[torch.Generator]] = {}
        for name in groups if groups is not None else ():
            self._ensure_group(name)

    # ----------------------------------------------------------- inspection --
    @property
    def groups(self) -> tuple[str, ...]:
        """Groups materialised so far (registration order is irrelevant)."""
        return tuple(sorted(self._generators))

    def episode_indices(self) -> torch.Tensor:
        """CPU copy of the per-env episode counter (-1 = never reset)."""
        return self._episode.clone()

    def seed_for(
        self, group: str, env_index: int, episode_index: int | None = None
    ) -> int:
        """The unsigned 64-bit stream seed for one key (for certificates)."""
        self._check_env_index(env_index)
        if episode_index is None:
            episode_index = int(self._episode[int(env_index)])
        return stable_stream_seed(
            self.protocol_version,
            self.eval_seed,
            int(env_index),
            int(episode_index),
            group,
        )

    def numpy_rng(
        self, group: str, env_index: int, episode_index: int | None = None
    ) -> np.random.Generator:
        """numpy Generator on this env/group's CURRENT episode key."""
        self._check_env_index(env_index)
        if episode_index is None:
            episode_index = int(self._episode[int(env_index)])
        if int(episode_index) < 0:
            # Same guard the torch draw path carries: an env that has never
            # been reset has no episode key, and silently keying on -1 would
            # hand out a generator nothing can reproduce.
            raise RuntimeError(
                f"env {env_index} has no episode yet: call reset_idx(...) "
                f"before drawing group {group!r}"
            )
        return numpy_generator(
            self.protocol_version,
            self.eval_seed,
            int(env_index),
            int(episode_index),
            group,
        )

    # ---------------------------------------------------------------- reset --
    def reset_idx(
        self,
        env_ids: Any,
        episode_index: int | Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Start a new episode for ``env_ids`` and reseed all their streams.

        Call this from the env's ``_reset_idx`` BEFORE drawing the new
        episode's scenario.  With ``episode_index=None`` each listed env's own
        counter advances by one (-1 -> 0 on the first reset), which is what
        makes the scenario controller-independent: env i's k-th episode is
        the same exam paper no matter when, or under which policy, it runs.
        Pass an explicit index to replay a specific episode.
        """
        ids = self._as_id_list(env_ids)
        if not ids:
            return
        if episode_index is None:
            for i in ids:
                self._episode[i] += 1
        elif isinstance(episode_index, (int, np.integer)) and not isinstance(
            episode_index, (bool, np.bool_)
        ):
            if episode_index < 0:
                raise ValueError("episode_index must be non-negative")
            for i in ids:
                self._episode[i] = int(episode_index)
        else:
            values = self._as_id_list(episode_index, bound=False)
            if len(values) != len(ids):
                raise ValueError(
                    f"episode_index has {len(values)} entries for {len(ids)} envs"
                )
            for i, value in zip(ids, values):
                if value < 0:
                    raise ValueError("episode_index must be non-negative")
                self._episode[i] = value
        for name, generators in self._generators.items():
            for i in ids:
                generators[i].manual_seed(
                    as_torch_seed(self.seed_for(name, i))
                )

    # ---------------------------------------------------------------- draws --
    def uniform(
        self,
        group: str,
        env_ids: Any,
        low: float = 0.0,
        high: float = 1.0,
        size: Sequence[int] | int = (),
    ) -> torch.Tensor:
        """Uniform in ``[low, high)``, shape ``(len(env_ids), *size)``.

        Row r corresponds to ``env_ids[r]`` and depends ONLY on that env's
        stream, so drawing for a subset gives the same numbers as drawing for
        the whole batch.  ``low``/``high`` are scalars; for a per-env range
        (a curriculum level, say) draw in [0, 1) and scale outside, which
        keeps the byte stream independent of the curriculum state.
        """
        if not high >= low:
            raise ValueError(f"uniform needs high >= low, got ({low}, {high})")
        span = float(high) - float(low)
        base = float(low)

        def sampler(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
            drawn = torch.rand(
                shape, generator=gen, device=self.generator_device, dtype=self.dtype
            )
            return drawn * span + base

        return self._draw(group, env_ids, size, sampler)

    def normal(
        self,
        group: str,
        env_ids: Any,
        mean: float = 0.0,
        std: float = 1.0,
        size: Sequence[int] | int = (),
    ) -> torch.Tensor:
        """Gaussian, shape ``(len(env_ids), *size)``; same isolation as above."""
        if std < 0.0:
            raise ValueError(f"normal needs std >= 0, got {std}")
        scale = float(std)
        shift = float(mean)

        def sampler(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
            drawn = torch.randn(
                shape, generator=gen, device=self.generator_device, dtype=self.dtype
            )
            return drawn * scale + shift

        return self._draw(group, env_ids, size, sampler)

    def uniform_(
        self,
        dest: torch.Tensor,
        group: str,
        env_ids: Any,
        low: float = 0.0,
        high: float = 1.0,
    ) -> torch.Tensor:
        """Scatter a uniform draw into rows ``env_ids`` of a full buffer.

        Mirrors the ``self.hs[env_ids] = uniform(...)`` shape of
        ``tasks/_shared/sea_state.py:107-118``, which is the call pattern the
        envs actually need.  ``dest`` must be ``(num_envs, *feature_shape)``.
        """
        ids, size = self._scatter_args(dest, env_ids)
        values = self.uniform(group, ids, low, high, size=size)
        return self._scatter(dest, ids, values)

    def normal_(
        self,
        dest: torch.Tensor,
        group: str,
        env_ids: Any,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> torch.Tensor:
        """Scatter a Gaussian draw into rows ``env_ids`` of a full buffer."""
        ids, size = self._scatter_args(dest, env_ids)
        values = self.normal(group, ids, mean, std, size=size)
        return self._scatter(dest, ids, values)

    # -------------------------------------------------------------- private --
    def _ensure_group(self, group: str) -> list[torch.Generator]:
        """Materialise a group's generators, seeded to the current episodes.

        Groups may appear at any time: because the key hashes the group NAME,
        a late arrival cannot disturb a stream that already exists.
        """
        generators = self._generators.get(group)
        if generators is not None:
            return generators
        _pack_text(group)  # validate the name before it reaches a key
        generators = []
        for i in range(self.num_envs):
            gen = torch.Generator(device=self.generator_device)
            episode = int(self._episode[i])
            # Envs that have never been reset get a placeholder seed; drawing
            # from them raises in _draw, so the placeholder is never observed.
            gen.manual_seed(
                as_torch_seed(self.seed_for(group, i)) if episode >= 0 else 0
            )
            generators.append(gen)
        self._generators[group] = generators
        return generators

    def _check_env_index(self, env_index: Any) -> int:
        index = int(env_index)
        if index != env_index or not 0 <= index < self.num_envs:
            raise IndexError(
                f"env index {env_index!r} outside [0, {self.num_envs})"
            )
        return index

    def _as_id_list(self, env_ids: Any, bound: bool = True) -> list[int]:
        if env_ids is None:
            return []
        if torch.is_tensor(env_ids):
            raw = env_ids.detach().flatten().cpu().tolist()
        elif isinstance(env_ids, np.ndarray):
            raw = env_ids.reshape(-1).tolist()
        elif isinstance(env_ids, (int, np.integer)):
            raw = [int(env_ids)]
        else:
            raw = list(env_ids)
        if bound:
            return [self._check_env_index(value) for value in raw]
        return [int(value) for value in raw]

    @staticmethod
    def _as_shape(size: Sequence[int] | int) -> tuple[int, ...]:
        if isinstance(size, (int, np.integer)):
            size = (int(size),)
        shape = tuple(int(n) for n in size)
        if any(n < 0 for n in shape):
            raise ValueError(f"size entries must be non-negative, got {shape}")
        return shape

    def _draw(
        self,
        group: str,
        env_ids: Any,
        size: Sequence[int] | int,
        sampler,
    ) -> torch.Tensor:
        ids = self._as_id_list(env_ids)
        shape = self._as_shape(size)
        generators = self._ensure_group(group)
        out = torch.empty(
            (len(ids), *shape), device=self.generator_device, dtype=self.dtype
        )
        for row, i in enumerate(ids):
            if int(self._episode[i]) < 0:
                raise RuntimeError(
                    f"env {i} has no episode yet: call reset_idx(...) before "
                    f"drawing group {group!r}"
                )
            out[row] = sampler(shape, generators[i])
        return out.to(self.device)

    def _scatter_args(
        self, dest: torch.Tensor, env_ids: Any
    ) -> tuple[list[int], tuple[int, ...]]:
        if not torch.is_tensor(dest):
            raise TypeError("dest must be a torch.Tensor")
        if dest.shape[:1] != (self.num_envs,):
            raise ValueError(
                f"dest must have leading dim num_envs={self.num_envs}, "
                f"got {tuple(dest.shape)}"
            )
        return self._as_id_list(env_ids), tuple(dest.shape[1:])

    @staticmethod
    def _scatter(
        dest: torch.Tensor, ids: list[int], values: torch.Tensor
    ) -> torch.Tensor:
        if not ids:
            return dest
        index = torch.tensor(ids, dtype=torch.long, device=dest.device)
        dest[index] = values.to(device=dest.device, dtype=dest.dtype)
        return dest


# ------------------------------------------------------------ scenario hash --
def _canonical(value: Any) -> str:
    """Injective text form of a scenario parameter.

    Length-prefixed strings and explicit type tags mean no two distinct
    parameter trees can encode to the same text.  Mappings are sorted (dict
    ORDER is not information); sequences are not (list order IS information).
    Floats use ``float.hex()``, which is exact -- 0.1 never becomes "0.1".
    """
    if value is None:
        return "n:"
    if isinstance(value, (bool, np.bool_)):
        return "b:1" if bool(value) else "b:0"
    if isinstance(value, (int, np.integer)):
        return f"i:{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"f:{float(value).hex()}"
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return f"s{len(raw)}:{value}"
    if isinstance(value, bytes):
        return f"y{len(value)}:{value.hex()}"
    if torch.is_tensor(value):
        return _canonical(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        body = "".join(_canonical(str(k)) + _canonical(v) for k, v in items)
        return f"d{len(items)}:{body}"
    if isinstance(value, (list, tuple)):
        body = "".join(_canonical(v) for v in value)
        return f"l{len(value)}:{body}"
    raise TypeError(f"scenario parameter of unsupported type {type(value).__name__}")


def scenario_hash(params: Mapping[str, Any], digest_size: int = 8) -> str:
    """Short stable hex digest of one episode's RESOLVED scenario.

    Record it per episode in the certificate: two runs can then be compared
    primitive by primitive (hash each group's sub-dict as well as the whole),
    which is how "these two eval seeds ran the identical scenarios" becomes a
    one-line check instead of a 79-certificate census.

    Invariant to dict ordering, sensitive to every value (floats compared by
    exact bits, so a float32 0.1 and a float64 0.1 are correctly different).
    """
    if not isinstance(params, Mapping):
        raise TypeError("scenario_hash expects a mapping of parameters")
    if not 1 <= digest_size <= 64:
        raise ValueError("digest_size must lie in [1, 64]")
    payload = _HASH_NAMESPACE + _canonical(params).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


__all__ = [
    "PROTOCOL_NOTES",
    "SCENARIO_GROUPS",
    "SCENARIO_PROTOCOL_VERSION",
    "ScenarioRNG",
    "as_torch_seed",
    "numpy_generator",
    "scenario_hash",
    "stable_stream_seed",
]
