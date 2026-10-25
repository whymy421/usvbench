"""Training-only D7 (and appearing) pools of the path-following family, online admission (G19b, P3 / F3).

``TrainingPathFollowingD7Pool`` offers the interface PathFollowingJournalMixin reads from its D7 pool
(``meta["suite"]``, ``path``, ``sha256``, ``pool_digest``, ``keys_not_admitted``, ``not_admitted_keys()``,
``grid_record()``, ``beyond_grid(key)``, ``geometry(env, episode)``, ``assignment(key, dose)``); the env takes
it from ``cfg.journal_d7_pool`` when the object carries ``TRAINING_ADAPTER = True`` (journal_path_following
_journal_construct, G19b hunk) instead of loading a pinned pool file:
  * a key whose mixture component is not ``dynamic``: ``beyond_grid`` is True (the env skips the pool geometry
    record) and the assignment is the family's D24 ``NotAdmittedAssignment`` (reason beyond_offered_grid) on the
    key's E5 layout geometry: no admission, no intruder;
  * a component key is admitted ONCE on its E5 layout (``journal_layout`` -> ``d7_task_context``) with E4
    ``admit_grid`` (non-joint; D1 taken; ``admission_suite``, the suite the offline tool uses) and serves the
    E4 Assignment at D1, or the D16 ``NotAdmittedAssignment`` with the D1 reasons.
The pattern is the family's own opt-in ``OnlineExtendedPool`` (online admission of a key on its E5 layout),
restricted to the mixture's dynamic episodes.
"""
from __future__ import annotations

from .._shared import training_hazards as TH
from . import journal_path_following as J

TRAINING_ADAPTER = True


class TrainingPathFollowingD7Pool:
    TRAINING_ADAPTER = True

    def __init__(self, *, mixture, task, vessel, seed, tier, num_envs, suite, cache, purpose="training",
                 registry=None, registry_stamp=None, admit_grid=None):
        if task != J.TASK:
            raise TH.TrainingHazardError("the path-following family carries only %r" % (J.TASK,))
        self.gate = TH.TrainingHazardGate(suite="dynamic", mixture=mixture, task=task, vessel=vessel, seed=seed,
                                          num_envs=num_envs, hazard_group=J.HAZARD_GROUP, cache=cache,
                                          purpose=purpose, registry=registry, registry_stamp=registry_stamp)
        if mixture is None:                 # calm-only control arm: the stack has no dynamic suite (D0)
            if suite.enabled or int(suite.dose) != 0:
                raise TH.TrainingHazardError("the calm arm runs the dynamic suite off (D0)")
        elif not (suite.enabled and int(suite.dose) == TH.TRAINING_LEVEL and suite.vessel_id == vessel):
            raise TH.TrainingHazardError("the training D7 pool needs the env's enabled D1 DynamicSuiteConfig")
        self.admission_suite = J.admission_suite(vessel)
        if J.suite_identity(J.admission_suite(vessel) if mixture is None else suite) != \
                J.suite_identity(self.admission_suite):
            raise TH.TrainingHazardError("the runtime D7 suite differs from the admission suite (only the dose may "
                                         "change)")
        self.vessel, self.seed, self.geometry_tier = str(vessel), int(seed), int(tier)
        self.num_envs = int(num_envs)
        self._admit_grid = J._dynamic().admit_grid if admit_grid is None else admit_grid
        self.admission_calls = 0
        self.pool_digest = self.gate.adapter_digest
        self.path, self.sha256 = None, None
        self.keys_offered, self.keys_admitted, self.keys_not_admitted = 0, 0, 0
        self.joint = False
        self.per_dose_counts = tuple((0, 0, 0) for _ in range(5))
        self.meta = {"task": J.TASK, "vessel": self.vessel, "seed": self.seed, "geometry_tier": self.geometry_tier,
                     "num_envs": self.num_envs, "episodes": 0, "hazard_group": J.HAZARD_GROUP,
                     "suite": J.suite_identity(self.admission_suite), "joint": False,
                     "source": TH.TRAINING_POOL_SOURCE, "training": self.gate.training_header()}

    # -- admission (pure function of the key)
    def _compute(self, key):
        def run():
            self.admission_calls += 1
            layout = J.journal_layout(self.vessel, self.geometry_tier, self.seed, int(key.env), int(key.episode))
            pool = self._admit_grid([(J.d7_task_context(layout, self.vessel), key, self.admission_suite)],
                                    joint=False, workers=1)
            found = pool.assignments.get((key, TH.TRAINING_LEVEL))
            if found is not None:
                return ("admitted", found, layout.geometry_hash)
            reasons = tuple((int(r.dose), str(r.reason)) for r in pool.rejections
                            if r.key == key and int(r.dose) == TH.TRAINING_LEVEL)
            if not reasons:
                raise TH.TrainingHazardError("D1 neither admitted nor rejected for %s" % (key,))
            return ("not_admitted", reasons, layout.geometry_hash)
        return run

    @staticmethod
    def _outcome(value):
        kind, item, _ = value
        return (True, []) if kind == "admitted" else (False, [r for _, r in item])

    def _served(self, key):
        return self.gate.cache.get("dynamic", key, self._compute(key), self._outcome)

    # -- interface read by PathFollowingJournalMixin
    def not_admitted_keys(self):
        return []

    def grid_record(self):
        return {"num_envs": self.num_envs, "episodes": 0}

    def covers(self, env, episode):
        return False

    def beyond_grid(self, key):
        """True for every non-component key (D24 object, no pool geometry record)."""
        self.gate.check_key(key, self.gate.level)
        return not self.gate.component_on(key)

    def is_admitted(self, env, episode):
        key = J._dynamic().EpisodeKey(self.seed, int(env), int(episode), J.HAZARD_GROUP)
        return self.gate.component_on(key) and self._served(key)[0] == "admitted"

    def geometry(self, env, episode):
        key = J._dynamic().EpisodeKey(self.seed, int(env), int(episode), J.HAZARD_GROUP)
        self.gate.check_key(key, self.gate.level)
        if not self.gate.component_on(key):
            raise ValueError("training D7 pool: key %s is not a dynamic-component key (it runs the D24 object)"
                             % (key,))
        return self._served(key)[2]

    def assignment(self, key, dose):
        self.gate.check_key(key, dose)
        if not self.gate.component_on(key):
            self.gate.cache.note_d0("dynamic", key)
            layout = J.journal_layout(self.vessel, self.geometry_tier, self.seed, int(key.env), int(key.episode))
            out = J.NotAdmittedAssignment(key, int(dose), layout.geometry_hash, self.vessel,
                                          J.BEYOND_GRID_REJECTIONS, self.pool_digest)
        else:
            kind, item, geometry = self._served(key)
            out = item if kind == "admitted" else J.NotAdmittedAssignment(key, int(dose), geometry, self.vessel,
                                                                          item, self.pool_digest)
        upcoming = self.gate.next_component_episode(key.env, key.episode)
        if upcoming is not None:
            nxt = J._dynamic().EpisodeKey(self.seed, int(key.env), upcoming, J.HAZARD_GROUP)
            self.gate.cache.prefetch("dynamic", nxt, self._compute(nxt))
        return out

    def training_header(self):
        return self.gate.training_header()

    def __deepcopy__(self, memo):
        return self                 # a runtime service, never configuration data


def appearing_inputs(task, vessel, tier, seed):
    """inputs(env, episode) of the family's appearing admission (journal_path_following.appearing_admission_inputs)."""
    function = getattr(J, "appearing_admission_inputs")

    def inputs(env, episode):
        return TH.call_admission_inputs(function, task, vessel, tier, seed, env, episode)
    return inputs


def training_pools(cfg, mixture, *, cache, purpose="training", registry=None, registry_stamp=None, num_envs=None):
    """{"d7": TrainingPathFollowingD7Pool, "appearing": TrainingAppearingPool or None}."""
    resolved = cfg.journal_conditions
    n = int(cfg.scene.num_envs if num_envs is None else num_envs)
    tier = int(getattr(cfg, "journal_geometry_tier", 2))
    common = dict(mixture=mixture, task=cfg.journal_task, vessel=cfg.vehicle, seed=int(cfg.seed), tier=tier,
                  num_envs=n, cache=cache, purpose=purpose, registry=registry, registry_stamp=registry_stamp)
    d7 = TrainingPathFollowingD7Pool(suite=resolved.dynamic_config(cfg.vehicle), **common)
    appearing = None
    if mixture is not None and mixture.appearing_wired:
        appearing = TH.TrainingAppearingPool(inputs=appearing_inputs(cfg.journal_task, cfg.vehicle, tier, cfg.seed),
                                             **common)
    return {"d7": d7, "appearing": appearing}


__all__ = ["TRAINING_ADAPTER", "TrainingPathFollowingD7Pool", "appearing_inputs", "training_pools"]
