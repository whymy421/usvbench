"""Training-only D7 (and appearing) pools of the HazardNav family, online admission (G19b, P3 / F1).

``TrainingHazardD7Pool`` is a ``journal_hazard_nav.HazardD7Pool`` (``family_pool`` returns it as is,
``resolve_d7_pool`` accepts it as an in-memory object through ``cfg.journal_d7_pool``):
  * a key whose mixture component is not ``dynamic`` gets the family's D24 object
    (``NotAdmittedAssignment(beyond_grid_record(key))``): no admission, the pool geometry check
    ``check_assignment_geometry`` is skipped for it, no intruder;
  * a component key is admitted ONCE on its real E5 scenario (``build_scenario`` -> ``d7_context``) with E4
    ``admit_grid`` (non-joint; the D1 entry is taken) and the runtime suite of the env; it serves the E4
    Assignment at D1, or ``NotAdmittedAssignment(NotAdmitted(key, D1 reasons, geometry_id))`` (D16).
``training_pools(cfg, mixture, ...)`` builds the D7 adapter and, where the appearing suite is wired
(``conditions/appearing/config.WIRED_TASKS``, read at run time) and in the mixture, the shared appearing
adapter on ``journal_hazard_nav.appearing_admission_inputs``.
"""
from __future__ import annotations

from .._shared import training_hazards as TH
from . import journal_hazard_nav as J

TRAINING_ADAPTER = True


def _dynamic():
    return J._dynamic()


class TrainingHazardD7Pool(J.HazardD7Pool):
    """HazardD7Pool interface over online admission (never constructed from an offline pool)."""

    TRAINING_ADAPTER = True

    def __init__(self, *, mixture, task, vessel, seed, tier, num_envs, suite, cache, purpose="training",
                 registry=None, registry_stamp=None, admit_grid=None):
        if task not in J.CARRIED_TASKS:
            raise TH.TrainingHazardError("task %r is not carried by HazardNav" % (task,))
        self.gate = TH.TrainingHazardGate(suite="dynamic", mixture=mixture, task=task, vessel=vessel, seed=seed,
                                          num_envs=num_envs, hazard_group="d7_" + task, cache=cache,
                                          purpose=purpose, registry=registry, registry_stamp=registry_stamp)
        if mixture is None:                 # calm-only control arm: the stack has no dynamic suite (D0)
            if suite.enabled or int(suite.dose) != 0:
                raise TH.TrainingHazardError("the calm arm runs the dynamic suite off (D0)")
        elif not (suite.enabled and int(suite.dose) == TH.TRAINING_LEVEL and suite.vessel_id == vessel):
            raise TH.TrainingHazardError("the training D7 pool needs the env's enabled D1 DynamicSuiteConfig")
        self.task, self.vessel, self.seed, self.tier = task, vessel, int(seed), int(tier)
        self.suite = suite
        self._admit_grid = _dynamic().admit_grid if admit_grid is None else admit_grid
        self.admission_calls = 0
        # HazardD7Pool attributes the env and the header read
        self.admission = None
        self.assignments = {}
        self.rejections = ()
        self.offered = ()
        self.not_admitted = {}
        self.pool_digest = self.gate.adapter_digest
        self.joint = False
        self.keys_offered = 0
        self.keys_admitted = 0
        self.keys_not_admitted = 0
        self.per_dose_counts = tuple((0, 0, 0) for _ in range(5))
        self.grid = {"eval_seed": self.seed, "hazard_group": "d7_" + task, "num_envs": int(num_envs), "episodes": 0}

    # -- admission (pure function of the key)
    def _compute(self, key):
        def run():
            self.admission_calls += 1
            scenario = J.build_scenario(self.task, self.vessel, self.tier, self.seed, int(key.env), int(key.episode))
            if J.d7_key(scenario) != key:
                raise TH.TrainingHazardError("scenario key differs from the requested key")
            pool = self._admit_grid([(J.d7_context(scenario), key, self.suite)], joint=False, workers=1)
            found = pool.assignments.get((key, TH.TRAINING_LEVEL))
            if found is not None:
                return ("admitted", found, scenario.geometry_id)
            reasons = tuple((int(r.dose), str(r.reason), int(r.sampled_states)) for r in pool.rejections
                            if r.key == key and int(r.dose) == TH.TRAINING_LEVEL)
            if not reasons:
                raise TH.TrainingHazardError("D1 neither admitted nor rejected for %s" % (key,))
            return ("not_admitted", J.NotAdmitted(key, reasons, scenario.geometry_id), scenario.geometry_id)
        return run

    @staticmethod
    def _outcome(value):
        kind, item, _ = value
        if kind == "admitted":
            return True, []
        return False, [r for _, r, _ in item.reasons]

    def is_admitted(self, key):
        return False

    def beyond_offered_grid(self, key):
        """True for every non-component key (it runs the D24 object; no pool geometry check)."""
        return not self.gate.component_on(key)

    def assignment(self, key, dose):
        self.gate.check_key(key, dose)
        if not self.gate.component_on(key):
            self.gate.cache.note_d0("dynamic", key)
            out = J.NotAdmittedAssignment(J.beyond_grid_record(key), int(dose), self.pool_digest)
        else:
            kind, item, _ = self.gate.cache.get("dynamic", key, self._compute(key), self._outcome)
            out = item if kind == "admitted" else J.NotAdmittedAssignment(item, int(dose), self.pool_digest)
        upcoming = self.gate.next_component_episode(key.env, key.episode)
        if upcoming is not None:
            nxt = _dynamic().EpisodeKey(self.seed, int(key.env), upcoming, "d7_" + self.task)
            self.gate.cache.prefetch("dynamic", nxt, self._compute(nxt))
        return out

    def training_header(self):
        return self.gate.training_header()

    def __deepcopy__(self, memo):
        return self                 # a runtime service, never configuration data


def appearing_inputs(task, vessel, tier, seed):
    """inputs(env, episode) of the family's appearing admission (journal_hazard_nav.appearing_admission_inputs)."""
    function = getattr(J, "appearing_admission_inputs")

    def inputs(env, episode):
        return TH.call_admission_inputs(function, task, vessel, tier, seed, env, episode)
    return inputs


def training_pools(cfg, mixture, *, cache, purpose="training", registry=None, registry_stamp=None, num_envs=None):
    """{"d7": TrainingHazardD7Pool, "appearing": TrainingAppearingPool or None} for a journal cfg carrying the
    resolved D1 stack (cfg.journal_conditions) and the run's seed."""
    resolved = cfg.journal_conditions
    n = int(cfg.scene.num_envs if num_envs is None else num_envs)
    suite = resolved.dynamic_config(cfg.vehicle)
    tier = int(getattr(cfg, "journal_geometry_tier", 2))
    common = dict(mixture=mixture, task=cfg.journal_task, vessel=cfg.vehicle, seed=int(cfg.seed), tier=tier,
                  num_envs=n, cache=cache, purpose=purpose, registry=registry, registry_stamp=registry_stamp)
    d7 = TrainingHazardD7Pool(suite=suite, **common)
    appearing = None
    if mixture is not None and mixture.appearing_wired:
        appearing = TH.TrainingAppearingPool(inputs=appearing_inputs(cfg.journal_task, cfg.vehicle, tier, cfg.seed),
                                             **common)
    return {"d7": d7, "appearing": appearing}


__all__ = ["TRAINING_ADAPTER", "TrainingHazardD7Pool", "appearing_inputs", "training_pools"]
