"""Training-only D7 (and appearing) pools of journal path hazard, online admission (G19b, P3 / F2).

``TrainingPathHazardD7Pool`` is the in-memory "PathHazardD7Pool-like object" that
path_hazard_journal.resolve_pool accepts from ``cfg.journal_d7_pool`` (``assignment``, ``is_admitted``,
``pool_digest``, ``grid``, ``check_suite``, ``vessel``, ``tier``; no env change needed); the env also reads
``key_status`` (pool_key_status), ``geometry_hashes`` and ``header_fields``:
  * a key whose mixture component is not ``dynamic``: ``key_status`` = beyond_offered_grid (no layout check)
    and the assignment is the family's D24 ``NotAdmittedAssignment``: no admission, no intruder;
  * a component key is admitted ONCE on its own E5 layout (``journal_layout`` -> ``d7_context`` with
    ``journal_envelope``, the functions of the offline ``admit_pool``) with E4 ``admit_grid`` (non-joint;
    D1 taken) and serves the E4 Assignment at D1, or the D16 ``NotAdmittedAssignment`` with the D1 reasons.
The header grid is {"num_envs": N, "episodes": 1}: check_grid needs a positive episode count; a training pool has
no offered grid (every episode index is served online), which the header's "training" block states.
"""
from __future__ import annotations

from .._shared import training_hazards as TH
from . import path_hazard_journal as J

TRAINING_ADAPTER = True


class _GeometryView:
    """``geometry_hashes.get((seed, env, episode))`` of the component keys (admission on demand)."""

    def __init__(self, pool):
        self.pool = pool

    def get(self, key3, default=None):
        seed, env, episode = (int(v) for v in key3)
        key = J.episode_key(seed, env, episode)
        if seed != self.pool.seed or not self.pool.gate.component_on(key):
            return default
        return self.pool._served(key)[2]


class TrainingPathHazardD7Pool:
    TRAINING_ADAPTER = True

    def __init__(self, *, mixture, task, vessel, seed, tier, num_envs, suite, cache, purpose="training",
                 registry=None, registry_stamp=None, admit_grid=None):
        if task != J.JOURNAL_TASK:
            raise TH.TrainingHazardError("the path-hazard family carries only %r" % (J.JOURNAL_TASK,))
        self.gate = TH.TrainingHazardGate(suite="dynamic", mixture=mixture, task=task, vessel=vessel, seed=seed,
                                          num_envs=num_envs, hazard_group="d7_" + J.JOURNAL_TASK, cache=cache,
                                          purpose=purpose, registry=registry, registry_stamp=registry_stamp)
        if mixture is None:
            if suite.enabled or int(suite.dose) != 0:
                raise TH.TrainingHazardError("the calm arm runs the dynamic suite off (D0)")
        elif not (suite.enabled and int(suite.dose) == TH.TRAINING_LEVEL and suite.vessel_id == vessel):
            raise TH.TrainingHazardError("the training D7 pool needs the env's enabled D1 DynamicSuiteConfig")
        dyn = J._dynamic()
        self.admission_suite = dyn.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
        self.suite_parameters = J.suite_parameters(self.admission_suite)
        self.vessel, self.tier, self.seed, self.num_envs = str(vessel), int(tier), int(seed), int(num_envs)
        self.eval_seed, self.hazard_group = self.seed, "d7_" + J.JOURNAL_TASK
        self._admit_grid = dyn.admit_grid if admit_grid is None else admit_grid
        self.admission_calls = 0
        self.pool_digest = self.gate.adapter_digest
        self.e4_per_dose_counts = tuple((0, 0) for _ in range(5))
        self.per_dose_counts = tuple((0, 0, 0) for _ in range(5))
        self.keys_offered = self.keys_admitted = self.keys_not_admitted = 0
        self.joint = False
        self.manifest_sha256 = None
        self.geometry_hashes = _GeometryView(self)
        self.grid = {"num_envs": self.num_envs, "episodes": 1}

    def _compute(self, key):
        def run():
            self.admission_calls += 1
            layout = J.journal_layout(self.vessel, self.tier, self.seed, int(key.env), int(key.episode))
            context = J.d7_context(layout, J.journal_envelope(self.vessel))
            pool = self._admit_grid([(context, key, self.admission_suite)], joint=False, workers=1)
            found = pool.assignments.get((key, TH.TRAINING_LEVEL))
            if found is not None:
                return ("admitted", found, layout.geometry_hash)
            reasons = tuple((int(r.dose), str(r.reason), int(r.sampled_states)) for r in pool.rejections
                            if r.key == key and int(r.dose) == TH.TRAINING_LEVEL)
            if not reasons:
                raise TH.TrainingHazardError("D1 neither admitted nor rejected for %s" % (key,))
            return ("not_admitted", reasons, layout.geometry_hash)
        return run

    @staticmethod
    def _outcome(value):
        kind, item, _ = value
        return (True, []) if kind == "admitted" else (False, [r for _, r, _ in item])

    def _served(self, key):
        return self.gate.cache.get("dynamic", key, self._compute(key), self._outcome)

    # -- interface read by resolve_pool / PathHazardEnv
    def check_suite(self, suite):
        if J.suite_parameters(suite) != self.suite_parameters:
            raise ValueError("the runtime D7 suite parameters differ from the training admission suite")
        return self

    def key_status(self, eval_seed, env_index, episode_index, hazard_group="d7_" + J.JOURNAL_TASK):
        key = J._dynamic().EpisodeKey(int(eval_seed), int(env_index), int(episode_index), str(hazard_group))
        self.gate.check_key(key, self.gate.level)          # KeyError outside the run (preflight error)
        if not self.gate.component_on(key):
            return J.BEYOND_OFFERED_GRID
        return J.ADMITTED if self._served(key)[0] == "admitted" else J.NOT_ADMITTED

    def is_admitted(self, eval_seed, env_index, episode_index):
        return self.key_status(eval_seed, env_index, episode_index) == J.ADMITTED

    def header_fields(self):
        return {"per_dose_counts": [list(r) for r in self.per_dose_counts],
                "per_dose_counts_columns": ["offered", "admitted", J.NOT_ADMITTED], "grid": dict(self.grid),
                "keys_not_admitted": 0, "not_admitted_reason_counts": {}, "pool_manifest_sha256": None,
                "not_admitted_policy": TH.ONLINE_RULE, "training": self.gate.training_header()}

    def assignment(self, key, dose):
        self.gate.check_key(key, dose)
        if not self.gate.component_on(key):
            self.gate.cache.note_d0("dynamic", key)
            out = J.NotAdmittedAssignment(key, J.BEYOND_OFFERED_GRID_REASONS, self.pool_digest)
        else:
            kind, item, _ = self._served(key)
            out = item if kind == "admitted" else J.NotAdmittedAssignment(key, item, self.pool_digest)
        upcoming = self.gate.next_component_episode(key.env, key.episode)
        if upcoming is not None:
            nxt = J.episode_key(self.seed, int(key.env), upcoming)
            self.gate.cache.prefetch("dynamic", nxt, self._compute(nxt))
        return out

    def training_header(self):
        return self.gate.training_header()

    def __deepcopy__(self, memo):
        return self                 # a runtime service, never configuration data


def training_pools(cfg, mixture, *, cache, purpose="training", registry=None, registry_stamp=None, num_envs=None):
    """{"d7": TrainingPathHazardD7Pool, "appearing": shared adapter where the suite is wired, else None}."""
    resolved = cfg.journal_conditions
    n = int(cfg.scene.num_envs if num_envs is None else num_envs)
    tier = int(getattr(cfg, "journal_geometry_tier", 2))
    common = dict(mixture=mixture, task=cfg.journal_task, vessel=cfg.vehicle, seed=int(cfg.seed), tier=tier,
                  num_envs=n, cache=cache, purpose=purpose, registry=registry, registry_stamp=registry_stamp)
    d7 = TrainingPathHazardD7Pool(suite=resolved.dynamic_config(cfg.vehicle), **common)
    return {"d7": d7, "appearing": TH.appearing_pool_for(J, **common)}


__all__ = ["TRAINING_ADAPTER", "TrainingPathHazardD7Pool", "training_pools"]
