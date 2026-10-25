"""Training-only hazard pools with ONLINE admission (G19b, IMPLEMENTATION_PLAN P3, shared part).

Why. A journal env asks its D7 pool (and, when the stack names the appearing
suite, its appearing pool) for an assignment at EVERY reset, D0 included
(journal_env_mixin.journal_reset / _journal_appearing_reset). Evaluation pools
are admitted offline for one seed and a 64 x 2 grid (D19, D23, D24); a PPO
training run has thousands of episodes with its own training seed, so no
evaluation pool can serve it (TRAINING_DESIGN section 3).

What. Under the option-2 mixture (conditions/training_mixture.py) a hazard
suite is ON in exactly the episodes whose component is that suite
(``TrainingMixture.hazard_dose(suite, env, episode) == 1``). The training
adapters therefore
  * serve the family's D24 beyond-grid D0 object for every other key: no
    admission, no geometry check, no intruder / no cylinder;
  * admit a component key ONCE, on its own E5 layout, with the evaluation
    admission code of the suite (E4 ``admit_grid`` for D7, non-joint, D1
    taken; ``appearing.placement.admit_key`` for the appearing cylinder, which
    is joint D1..D4 by construction) and serve its D1 assignment, or the
    family's D16 not-admitted D0 object when D1 is rejected.
Admission is a pure function of the key (seed, env, episode, group), so the
served objects do not depend on reset order, timing or the prefetch option.

This module holds the shared part: ``TrainingHazardCache`` (memo per suite and
key, optional deterministic prefetch thread, the admission log written into the
training record), the refusal of every seed whose registry purpose is not
training / training_smoke (``check_training_seed``), the common key gate
(``TrainingHazardGate``) and the shared APPEARING adapter
(``TrainingAppearingPool``, the interface journal_env_mixin._journal_appearing_setup
reads). The D7 adapters are per family (tasks/<family>/training_pool.py).

Never used by an evaluator: the adapters exist only together with a training
mixture (a layer with a mixture refuses certificate_stamp), they live only in
the in-memory cfg fields journal_d7_pool / journal_appearing_pool, which
cfg_override protects (CONTEXT_R4 D12, D25), and they refuse evaluation seeds.

Pure Python / NumPy / torch-free; imports no simulator package.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import threading
import time

TRAINING_HAZARDS_SCHEMA = "usvbench.training_hazards/1"
ADMISSION_LOG_SCHEMA = "usvbench.training_admission_log/1"
TRAINING_POOL_SOURCE = "training_online"
TRAINING_LEVEL = 1
HAZARD_SUITES = ("dynamic", "appearing")
REPO = Path(__file__).resolve().parents[2]
SEED_REGISTRY_PY = REPO / "scripts" / "seed_registry.py"
ONLINE_RULE = ("training only (G19b, IMPLEMENTATION_PLAN P3): a key whose mixture component is not this suite runs "
               "the family's D24 beyond-grid D0 object (no admission); a component key is admitted once on its E5 "
               "layout under the suite's evaluation admission code and runs its D1 assignment, or the D16 "
               "not-admitted D0 object")


class TrainingHazardError(ValueError):
    """A training hazard adapter was misconfigured or used outside training."""


def _digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


# ------------------------------------------------------------------ seed refusal --
_REGISTRY_MODULE = None


def seed_registry_module():
    """scripts/seed_registry.py of THIS tree, loaded by path (pure Python)."""
    global _REGISTRY_MODULE
    if _REGISTRY_MODULE is None:
        spec = importlib.util.spec_from_file_location("_g19b_seed_registry", SEED_REGISTRY_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REGISTRY_MODULE = module
    return _REGISTRY_MODULE


def check_training_seed(seed, purpose, registry=None):
    """Registry stamp of a training seed (seed_registry.resolve_training_block); refuses every other seed.

    ``registry`` is a registry dict (default: scripts/seed_registry.json of this tree). Evaluation seeds
    (selection, pilot, calibration, final test), undeclared seeds, a seed of the other training purpose and a
    campaign seed of an unfrozen registry raise TrainingHazardError."""
    module = seed_registry_module()
    try:
        if registry is None:
            registry = module.load_registry()
        return module.resolve_training_block(seed, purpose, registry)
    except module.RegistryError as error:
        raise TrainingHazardError("training hazard pools refuse seed %r: %s" % (seed, error)) from None


def check_mixture(mixture, seed, purpose, task, vessel):
    from .conditions.training_mixture import TRAINING_PURPOSES, TrainingMixture
    if not isinstance(mixture, TrainingMixture):
        raise TrainingHazardError("training hazard pools need the run's TrainingMixture")
    if mixture.purpose not in TRAINING_PURPOSES or mixture.purpose != purpose:
        raise TrainingHazardError("mixture purpose %r differs from the training purpose %r" % (mixture.purpose, purpose))
    if int(mixture.seed) != int(seed) or mixture.task != task or mixture.vessel != vessel:
        raise TrainingHazardError("the mixture belongs to another seed, task or vessel")
    return mixture


# ------------------------------------------------------------------ the cache --
class TrainingHazardCache:
    """Memo of online admissions per (suite, key), with an optional deterministic prefetch thread.

    ``get(suite, key, compute)`` returns ``compute()`` of the key, computing it at most once (a prefetched
    result is reused; a key still in flight is awaited). ``compute`` must be a pure function of the key (the
    admission), so the served object is identical with or without prefetch (tested). Entries of an env older
    than the episode just served are dropped (each key is needed at its own reset only).
    The admission log counts SERVED component keys only: per suite the keys served at D1, admitted, not admitted
    (with reasons), the admission wall time per key (computed in the thread or synchronously) and the time the
    training thread actually blocked on admissions (the M3 cost, TRAINING_DESIGN section 7.3).
    """

    def __init__(self, prefetch=False, keep_keys=True):
        if prefetch not in (True, False):
            raise TrainingHazardError("prefetch must be a bool")
        self.prefetch_enabled = prefetch
        self.keep_keys = bool(keep_keys)
        self._results = {}           # (suite, key) -> (value, compute_wall_s, source)
        self._futures = {}           # (suite, key) -> Future
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g19b-prefetch") if prefetch else None
        self._latest = {}            # (suite, env) -> latest served episode
        self._log = {}
        self._keys = []

    # -- internals
    @staticmethod
    def _timed(compute):
        start = time.perf_counter()
        value = compute()
        return value, time.perf_counter() - start

    def _suite_log(self, suite):
        if suite not in self._log:
            self._log[suite] = {"keys_served_d1": 0, "admitted": 0, "not_admitted": 0, "not_admitted_reasons": {},
                                "admission_wall_s_total": 0.0, "admission_wall_s_max": 0.0,
                                "blocking_wall_s_total": 0.0, "prefetched_served": 0, "d0_keys_served": 0}
        return self._log[suite]

    # -- API
    def prefetch(self, suite, key, compute):
        """Start ``compute`` for (suite, key) in the background (no-op without prefetch or when known)."""
        if self._executor is None:
            return False
        ident = (suite, key)
        with self._lock:
            if ident in self._results or ident in self._futures:
                return False
            self._futures[ident] = self._executor.submit(self._timed, compute)
        return True

    def get(self, suite, key, compute, outcome):
        """Served value of a COMPONENT key; ``outcome(value)`` -> (admitted: bool, reasons: list of str)."""
        ident = (suite, key)
        start = time.perf_counter()
        with self._lock:
            cached = self._results.get(ident)
            future = self._futures.pop(ident, None) if cached is None else None
        first = cached is None
        if cached is None:
            if future is not None:
                value, wall = future.result()
                source = "prefetch"
            else:
                value, wall = self._timed(compute)
                source = "sync"
            cached = (value, wall, source)
            with self._lock:
                self._results[ident] = cached
        blocked = time.perf_counter() - start
        value, wall, source = cached
        if first:
            entry = self._suite_log(suite)
            admitted, reasons = outcome(value)
            entry["keys_served_d1"] += 1
            entry["admitted" if admitted else "not_admitted"] += 1
            for reason in reasons if not admitted else ():
                entry["not_admitted_reasons"][reason] = entry["not_admitted_reasons"].get(reason, 0) + 1
            entry["admission_wall_s_total"] += wall
            entry["admission_wall_s_max"] = max(entry["admission_wall_s_max"], wall)
            entry["blocking_wall_s_total"] += blocked
            entry["prefetched_served"] += int(source == "prefetch")
            if self.keep_keys:
                self._keys.append([suite, int(key.env), int(key.episode), bool(admitted), round(wall, 6), source])
        self._forget_older(suite, key)
        return value

    def note_d0(self, suite, key):
        """A non-component key served its D0 object (counted once per key, never admitted)."""
        entry = self._suite_log(suite)
        if int(key.episode) > self._latest.get((suite, int(key.env)), -1):
            entry["d0_keys_served"] += 1
        self._forget_older(suite, key)

    def _forget_older(self, suite, key):
        env, episode = int(key.env), int(key.episode)
        latest = self._latest.get((suite, env), -1)
        if episode <= latest:
            return
        self._latest[(suite, env)] = episode
        with self._lock:
            for ident in [i for i in self._results if i[0] == suite and int(i[1].env) == env
                          and int(i[1].episode) < episode]:
                del self._results[ident]

    def admission_log(self):
        """JSON-native admission log (schema ADMISSION_LOG_SCHEMA) for the training record."""
        suites = {}
        for suite, entry in sorted(self._log.items()):
            out = dict(entry)
            out["not_admitted_reasons"] = dict(sorted(entry["not_admitted_reasons"].items()))
            n = entry["keys_served_d1"]
            out["admission_wall_s_mean"] = (entry["admission_wall_s_total"] / n) if n else None
            suites[suite] = out
        return {"schema": ADMISSION_LOG_SCHEMA, "rule": ONLINE_RULE, "prefetch": self.prefetch_enabled,
                "suites": suites, "keys": list(self._keys) if self.keep_keys else None}

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __deepcopy__(self, memo):
        return self                 # a runtime service (threads, locks), never configuration data


# ------------------------------------------------------------------ key gate --
class TrainingHazardGate:
    """Common key checks of one training hazard adapter (one suite, one task, one vessel, one seed).

    ``mixture=None`` is the calm-only control arm of the training smoke (TRAINING_DESIGN section 7.2): the
    stack carries no hazard suite (dose 0 on every key), every key gets the D24 D0 object, nothing is admitted.
    """

    TRAINING_ADAPTER = True         # marker read by the families and refused by every evaluator (eval_protocol)

    def __init__(self, *, suite, mixture, task, vessel, seed, num_envs, hazard_group, cache, purpose="training",
                 registry=None, registry_stamp=None):
        if suite not in HAZARD_SUITES:
            raise TrainingHazardError("unknown hazard suite %r" % (suite,))
        if type(num_envs) is not int or num_envs < 1:
            raise TrainingHazardError("num_envs must be a positive integer")
        if not isinstance(cache, TrainingHazardCache):
            raise TrainingHazardError("a TrainingHazardCache is required")
        stamp = check_training_seed(int(seed), purpose, registry)      # always resolved (follow-up M4)
        if registry_stamp is not None:
            given = dict(registry_stamp)
            for name in ("registry_sha256", "namespace", "block_id", "seed", "purpose", "registry_status"):
                if given.get(name) != stamp.get(name):
                    raise TrainingHazardError("the registry stamp passed for seed %r differs from the registry in "
                                              "%r (%r vs %r): stamps are verified, never trusted"
                                              % (seed, name, given.get(name), stamp.get(name)))
        if mixture is not None:
            check_mixture(mixture, seed, purpose, task, vessel)
        elif suite == "appearing":
            raise TrainingHazardError("the calm arm has no appearing pool (the stack carries no appearing suite)")
        if suite == "appearing" and not mixture.appearing_wired:
            raise TrainingHazardError("the mixture does not carry the appearing suite on %r" % (task,))
        self.training_suite = suite
        self.mixture = mixture
        self.task, self.vessel, self.seed = str(task), str(vessel), int(seed)
        self.num_envs = int(num_envs)
        self.hazard_group = str(hazard_group)
        self.cache = cache
        self.purpose = purpose
        self.registry_stamp = stamp
        self.adapter_digest = _digest({"schema": TRAINING_HAZARDS_SCHEMA, "suite": suite, "task": self.task,
                                       "vessel": self.vessel, "seed": self.seed, "purpose": purpose,
                                       "hazard_group": self.hazard_group, "num_envs": self.num_envs,
                                       "mixture_sha256": None if mixture is None else mixture.digest()})
        # the stack dose the env requests: the suite's training dose (D1, or the owner-decision-J10 exception,
        # BlueBoat station keeping D2; training_mixture.training_level), read from the mixture
        self.level = 0 if mixture is None else int(mixture.level(suite))

    def check_key(self, key, dose):
        """Fail closed on a key outside this run (another seed or group, env out of range) or a dose other than
        the training dose (D1, or the J10 exception)."""
        if int(key.eval_seed) != self.seed or str(key.hazard_group) != self.hazard_group:
            raise KeyError(key)
        if not 0 <= int(key.env) < self.num_envs or int(key.episode) < 0:
            raise KeyError(key)
        if int(dose) != self.level:
            raise TrainingHazardError("this training hazard pool serves dose D%d only (requested D%s)"
                                      % (self.level, dose))

    def component_on(self, key):
        if self.mixture is None:
            return False
        return self.mixture.hazard_dose(self.training_suite, int(key.env), int(key.episode)) == self.level

    def next_component_episode(self, env, after_episode):
        """Next episode index > after_episode whose component is this suite (within two blocks), or None."""
        if self.mixture is None:
            return None
        for episode in range(int(after_episode) + 1, int(after_episode) + 1 + 2 * self.mixture.block_length):
            if self.mixture.hazard_dose(self.training_suite, int(env), episode) == self.level:
                return episode
        return None

    def __deepcopy__(self, memo):
        return self                 # a runtime service, never configuration data

    def training_header(self):
        return {"schema": TRAINING_HAZARDS_SCHEMA, "suite": self.training_suite, "source": TRAINING_POOL_SOURCE,
                "rule": ONLINE_RULE, "adapter_digest": self.adapter_digest, "arm": "calm" if self.mixture is None
                else "mixture", "mixture_sha256": None if self.mixture is None else self.mixture.digest(),
                "registry": dict(self.registry_stamp), "certificate_eligible": False}


_INPUT_NAMES = {"task": "task", "vessel": "vessel", "vehicle": "vessel", "vessel_id": "vessel", "vessel_key": "vessel",
                "tier": "tier", "geometry_tier": "tier", "seed": "seed", "eval_seed": "seed", "env": "env",
                "env_index": "env", "episode": "episode", "episode_index": "episode"}


def call_admission_inputs(function, task, vessel, tier, seed, env, episode, extras=None):
    """Call a family's ``appearing_admission_inputs`` by PARAMETER NAMES (follow-up M3).

    Known names: task; vessel / vehicle / vessel_id / vessel_key; tier / geometry_tier; seed / eval_seed;
    env / env_index; episode / episode_index. Any other parameter is taken from ``extras`` (e.g. the docking
    family's berth_walls, which must equal the env's cfg.berth_walls) or left at its default; a required
    parameter with an unknown name is refused (fail closed). Families wired later (G20b) are picked up without a
    change here as long as their parameters use these names."""
    values = {"task": task, "vessel": vessel, "tier": int(tier), "seed": int(seed), "env": int(env),
              "episode": int(episode)}
    extras = dict(extras or {})
    positional, keywords = [], {}
    for p in inspect.signature(function).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in _INPUT_NAMES:
            value = values[_INPUT_NAMES[p.name]]
        elif p.name in extras:
            value = extras[p.name]
        elif p.default is not p.empty:
            continue
        else:
            raise TrainingHazardError("appearing_admission_inputs parameter %r of %s is unknown (signature %s)"
                                      % (p.name, getattr(function, "__module__", function),
                                         inspect.signature(function)))
        if p.kind == p.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[p.name] = value
    return function(*positional, **keywords)


# ------------------------------------------------------------------ appearing --
def _appearing():
    from .conditions.appearing import config, placement, pool
    from .conditions.dynamic.model import EpisodeKey
    return config, placement, pool, EpisodeKey


class TrainingAppearingPool(TrainingHazardGate):
    """Shared training adapter of the appearing suite (what journal_env_mixin._journal_appearing_setup reads).

    Interface: ``assignment(key, dose)``, ``pool_digest``, ``suite_identity``, ``task``, ``vessel_id``,
    ``seed``, ``header_block()``. ``inputs(env, episode)`` -> (TaskContext, length_scale, radius_range_m or
    None[, regions]) is the family's admission input of the key's REAL journal layout (the same function the
    offline admission tool uses; G20b docking adds the berth ``regions``, passed through to admit_key). The
    context task must be ``config.context_task(task)`` (G20b: harbour_mission -> "harbour"). A non-component key
    runs the D24 beyond-grid D0 object; a component key is admitted once with ``placement.admit_key`` (the
    evaluation admission, joint over the ladder's witnessed doses) and serves its D1 assignment, built by the
    appearing pool module's OWN construction (a one-key AppearingPool, then re-keyed), so the record fields
    follow the pool module of this tree."""

    def __init__(self, *, mixture, task, vessel, seed, tier, num_envs, inputs, cache, purpose="training",
                 registry=None, registry_stamp=None, admit=None):
        config, placement, pool_module, _ = _appearing()
        if task not in config.WIRED_TASKS:          # read at run time (G20b extends WIRED_TASKS)
            raise TrainingHazardError("the appearing suite is not wired on %r (WIRED_TASKS %s)"
                                      % (task, ", ".join(config.WIRED_TASKS)))
        super().__init__(suite="appearing", mixture=mixture, task=task, vessel=vessel, seed=seed,
                         num_envs=num_envs, hazard_group=config.hazard_group(task), cache=cache, purpose=purpose,
                         registry=registry, registry_stamp=registry_stamp)
        if not callable(inputs):
            raise TrainingHazardError("inputs(env, episode) is required")
        self.vessel_id = self.vessel
        self.tier = int(tier)
        self._inputs = inputs
        self._admit = placement.admit_key if admit is None else admit
        self.admission_cfg = config.AppearingSuiteConfig(enabled=True, dose=0, vessel_id=self.vessel)
        self.suite_identity = self.admission_cfg.identity()
        self.pool_digest = self.adapter_digest
        self.grid = {"num_envs": self.num_envs, "episodes": 0}
        self.keys_offered = None
        self.keys_admitted = None
        self.keys_not_admitted = None
        self.per_dose_counts = ()
        self.joint = True
        self.admission_calls = 0

    def _compute(self, key):
        def run():
            config = _appearing()[0]
            self.admission_calls += 1
            item = tuple(self._inputs(int(key.env), int(key.episode)))
            context, length_scale, radius_range = item[:3]
            regions = item[3] if len(item) > 3 else None
            want = config.context_task(self.task) if callable(getattr(config, "context_task", None)) else self.task
            if context.task != want or context.envelope.vessel_id != self.vessel:
                raise TrainingHazardError("appearing admission inputs belong to another task or vessel (context "
                                          "task %r, expected %r)" % (context.task, want))
            kwargs = {"length_scale": float(length_scale), "radius_range_m": radius_range}
            if regions is not None:
                kwargs["regions"] = regions
            return self._admit(context, key, self.admission_cfg, **kwargs)
        return run

    @staticmethod
    def _outcome(record):
        reasons = [str(r) for _, r in record["reasons"] if r != "joint_ladder_member_rejected"]
        return (not record["reasons"]), reasons

    def _assignment_from(self, record, key, dose):
        """The served object, built by the pool module of this tree: a one-key AppearingPool over the admission
        record (it validates the witnessed doses and digests exactly as an offline pool does), then re-keyed to
        the real key and this adapter's digest (dataclasses.replace; the record digests never contain the key)."""
        import dataclasses
        config, _, pool_module, episode_key = _appearing()
        local = pool_module.AppearingPool(
            task=self.task, vessel_id=self.vessel, seed=self.seed, tier=self.tier, suite_identity=self.suite_identity,
            num_envs=1, episodes=1, keys=[{"env": 0, "episode": 0, "geometry_id": record["geometry_id"],
                                           "placement": record["placement"],
                                           "reasons": [tuple(r) for r in record["reasons"]],
                                           "doses": dict(record.get("doses") or {})}])
        served = local.assignment(episode_key(self.seed, 0, 0, self.hazard_group), int(dose))
        return dataclasses.replace(served, key=key, pool_digest=self.pool_digest)

    def assignment(self, key, dose):
        config, _, pool_module, episode_key = _appearing()
        if not isinstance(key, episode_key):
            raise TypeError("appearing assignments are looked up by EpisodeKey")
        self.check_key(key, dose)
        if not self.component_on(key):
            self.cache.note_d0("appearing", key)
            out = pool_module.NotAdmittedAppearing(key, self.task, self.vessel, None,
                                                   ((0, config.BEYOND_OFFERED_GRID),), int(dose), self.pool_digest)
        else:
            record = self.cache.get("appearing", key, self._compute(key), self._outcome)
            out = self._assignment_from(record, key, dose)
        upcoming = self.next_component_episode(key.env, key.episode)
        if upcoming is not None:
            nxt = episode_key(self.seed, int(key.env), upcoming, self.hazard_group)
            self.cache.prefetch("appearing", nxt, self._compute(nxt))
        return out

    def header_block(self):
        return {"pool_digest": self.pool_digest, "grid": dict(self.grid), "joint": True,
                "suite_identity": dict(self.suite_identity), "training": self.training_header()}


def appearing_pool_for(family_module, *, mixture, task, vessel, seed, tier, num_envs, cache, purpose="training",
                       registry=None, registry_stamp=None, extras=None, **_ignored):
    """The shared appearing adapter of a family, or None when the mixture carries no appearing suite.

    The family's admission inputs are looked up at run time (``appearing_admission_inputs`` of its journal
    module, the function its offline admission tool uses), so a family wired later (G20b) needs no change here;
    a wired task whose family module has no such function is refused."""
    if mixture is None or not mixture.appearing_wired:
        return None
    modules = tuple(family_module) if isinstance(family_module, (tuple, list)) else (family_module,)
    function = next((getattr(m, "appearing_admission_inputs") for m in modules
                     if callable(getattr(m, "appearing_admission_inputs", None))), None)
    if function is None:
        raise TrainingHazardError("the appearing suite is wired on %r but none of %s has appearing_admission_inputs"
                                  % (task, [getattr(m, "__name__", m) for m in modules]))

    def inputs(env, episode):
        return call_admission_inputs(function, task, vessel, tier, seed, env, episode, extras)
    return TrainingAppearingPool(mixture=mixture, task=task, vessel=vessel, seed=seed, tier=tier, num_envs=num_envs,
                                 inputs=inputs, cache=cache, purpose=purpose, registry=registry,
                                 registry_stamp=registry_stamp)


__all__ = ["TRAINING_HAZARDS_SCHEMA", "ADMISSION_LOG_SCHEMA", "TRAINING_POOL_SOURCE", "ONLINE_RULE",
           "TrainingHazardError", "TrainingHazardCache", "TrainingHazardGate", "TrainingAppearingPool",
           "check_training_seed", "check_mixture", "call_admission_inputs", "seed_registry_module",
           "appearing_pool_for"]
