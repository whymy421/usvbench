"""Training-only D7 (and appearing) pools of journal docking, online admission (G19b, P3 / F5).

``TrainingDockingD7Pool`` offers the interface DockingEnv reads from an in-memory ``cfg.journal_d7_pool``
(docking_env._journal_init / _journal_post_setup / _journal_reset_idx; no env change needed): ``berth_walls``,
``vessel``, ``eval_seed``, ``grid``, ``suite`` (for journal_d7_pool.suite_matches), ``exposure(env, k)``,
``geometry_id(env, k)``, ``beyond_grid(env, k, seed)``, ``not_admitted_reasons``, ``header()``, ``assignment``:
  * a key whose mixture component is not ``dynamic``: ``beyond_grid`` is True (the env skips the pool geometry
    check) and the assignment is the family's D24 ``NotAdmittedAssignment`` on the key's E5 scenario geometry;
  * a component key is admitted ONCE on its own E5 scenario (``journal_docking.build_scenario`` -> ``d7_context``,
    the offline tool's functions, CPU test ``small_pool``) with E4 ``admit_grid`` (non-joint; D1 taken) and serves
    the E4 Assignment at D1, or the D16 ``NotAdmittedAssignment`` with the D1 reasons.
"""
from __future__ import annotations

from .._shared import training_hazards as TH
from . import journal_d7_pool as P
from . import journal_docking as JD

TRAINING_ADAPTER = True


def _dynamic():
    return P._dynamic()


class TrainingDockingD7Pool:
    TRAINING_ADAPTER = True

    def __init__(self, *, mixture, task, vessel, seed, tier, num_envs, suite, cache, berth_walls=True,
                 purpose="training", registry=None, registry_stamp=None, admit_grid=None):
        if task != "docking":
            raise TH.TrainingHazardError("the docking family carries only 'docking'")
        self.gate = TH.TrainingHazardGate(suite="dynamic", mixture=mixture, task=task, vessel=vessel, seed=seed,
                                          num_envs=num_envs, hazard_group=P.HAZARD_GROUP, cache=cache,
                                          purpose=purpose, registry=registry, registry_stamp=registry_stamp)
        if mixture is None:
            if suite.enabled or int(suite.dose) != 0:
                raise TH.TrainingHazardError("the calm arm runs the dynamic suite off (D0)")
        elif not (suite.enabled and int(suite.dose) == TH.TRAINING_LEVEL and suite.vessel_id == vessel):
            raise TH.TrainingHazardError("the training D7 pool needs the env's enabled D1 DynamicSuiteConfig")
        dyn = _dynamic()
        from .._shared.conditions.dynamic.stamp import plain
        self.admission_suite = dyn.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
        self.suite = dict(plain(self.admission_suite))
        self.berth_walls = bool(berth_walls)
        self.vessel, self.eval_seed, self.geometry_tier = str(vessel), int(seed), int(tier)
        self.grid = {"num_envs": int(num_envs), "episodes": 0}
        self._admit_grid = dyn.admit_grid if admit_grid is None else admit_grid
        self.admission_calls = 0
        self.pool_digest = self.gate.adapter_digest
        self.per_dose_counts = tuple((0, 0) for _ in range(5))
        self.keys_offered = self.keys_admitted = self.keys_not_admitted = 0
        self.joint = False
        self.file_sha256 = None

    def _key(self, env, k):
        return _dynamic().EpisodeKey(self.eval_seed, int(env), int(k), P.HAZARD_GROUP)

    def _compute(self, key):
        def run():
            self.admission_calls += 1
            scenario = JD.build_scenario(self.vessel, self.geometry_tier, self.eval_seed, int(key.env),
                                         int(key.episode), self.berth_walls)
            context = JD.d7_context(_dynamic(), scenario)
            pool = self._admit_grid([(context, key, self.admission_suite)], joint=False, workers=1)
            found = pool.assignments.get((key, TH.TRAINING_LEVEL))
            if found is not None:
                return ("admitted", found, scenario.geometry_id)
            reasons = tuple(sorted((int(r.dose), str(r.reason), int(r.sampled_states)) for r in pool.rejections
                                   if r.key == key and int(r.dose) == TH.TRAINING_LEVEL))
            if not reasons:
                raise TH.TrainingHazardError("D1 neither admitted nor rejected for %s" % (key,))
            return ("not_admitted", reasons, scenario.geometry_id)
        return run

    @staticmethod
    def _outcome(value):
        kind, item, _ = value
        return (True, []) if kind == "admitted" else (False, [r for _, r, _ in item])

    def _served(self, key):
        return self.gate.cache.get("dynamic", key, self._compute(key), self._outcome)

    # -- interface read by DockingEnv
    def covers(self, env, k):
        return False

    def beyond_grid(self, env, k, seed=None, hazard_group=P.HAZARD_GROUP):
        if seed is not None and int(seed) != self.eval_seed or hazard_group != P.HAZARD_GROUP:
            return False
        key = self._key(env, k)
        try:
            self.gate.check_key(key, self.gate.level)
        except KeyError:
            return False
        return not self.gate.component_on(key)

    def geometry_id(self, env, k):
        key = self._key(env, k)
        self.gate.check_key(key, self.gate.level)
        if not self.gate.component_on(key):
            raise ValueError("training D7 pool: env %d episode %d runs the D24 object (no pool record)" % (env, k))
        return self._served(key)[2]

    def exposure(self, env, k):
        key = self._key(env, k)
        if not self.gate.component_on(key):
            return P.EXPOSURE_NOT_ADMITTED
        return P.EXPOSURE_ADMITTED if self._served(key)[0] == "admitted" else P.EXPOSURE_NOT_ADMITTED

    def not_admitted_reasons(self, env, k):
        key = self._key(env, k)
        if not self.gate.component_on(key):
            return P.BEYOND_GRID_REASONS
        kind, item, _ = self._served(key)
        return () if kind == "admitted" else item

    def per_dose_exposure_counts(self):
        return [{"dose": d, "offered": 0, "admitted": 0, "not_admitted": 0} for d in range(5)]

    def header(self):
        return {"schema": P.POOL_SCHEMA, "task": "docking", "vessel": self.vessel, "geometry_tier": self.geometry_tier,
                "eval_seed": self.eval_seed, "grid": dict(self.grid), "berth_walls": self.berth_walls,
                "hazard_group": P.HAZARD_GROUP, "pool_digest": self.pool_digest, "file_sha256": None,
                "per_dose_counts": [list(x) for x in self.per_dose_counts],
                "per_dose_exposure_counts": self.per_dose_exposure_counts(), "task_episodes": 0,
                "keys_admitted": 0, "keys_not_admitted": 0, "not_admitted": [],
                "exposure_rule": TH.ONLINE_RULE, "beyond_grid_rule": P.BEYOND_GRID_RULE,
                "training": self.gate.training_header()}

    def assignment(self, key, dose):
        self.gate.check_key(key, dose)
        if not self.gate.component_on(key):
            self.gate.cache.note_d0("dynamic", key)
            geometry = JD.build_scenario(self.vessel, self.geometry_tier, self.eval_seed, int(key.env),
                                         int(key.episode), self.berth_walls).geometry_id
            out = P.NotAdmittedAssignment(key, geometry, P.BEYOND_GRID_REASONS, self.pool_digest, self.vessel,
                                          int(dose))
        else:
            kind, item, geometry = self._served(key)
            out = item if kind == "admitted" else P.NotAdmittedAssignment(key, geometry, item, self.pool_digest,
                                                                          self.vessel, int(dose))
        upcoming = self.gate.next_component_episode(key.env, key.episode)
        if upcoming is not None:
            nxt = self._key(key.env, upcoming)
            self.gate.cache.prefetch("dynamic", nxt, self._compute(nxt))
        return out

    def training_header(self):
        return self.gate.training_header()

    def __deepcopy__(self, memo):
        return self                 # a runtime service, never configuration data


def training_pools(cfg, mixture, *, cache, purpose="training", registry=None, registry_stamp=None, num_envs=None):
    """{"d7": TrainingDockingD7Pool, "appearing": shared adapter where the suite is wired, else None}."""
    resolved = cfg.journal_conditions
    n = int(cfg.scene.num_envs if num_envs is None else num_envs)
    tier = int(getattr(cfg, "journal_geometry_tier", 2))
    common = dict(mixture=mixture, task=cfg.journal_task, vessel=cfg.vehicle, seed=int(cfg.seed), tier=tier,
                  num_envs=n, cache=cache, purpose=purpose, registry=registry, registry_stamp=registry_stamp)
    d7 = TrainingDockingD7Pool(suite=resolved.dynamic_config(cfg.vehicle),
                               berth_walls=bool(getattr(cfg, "berth_walls", True)), **common)
    return {"d7": d7, "appearing": TH.appearing_pool_for((JD, P), extras={"berth_walls": d7.berth_walls},
                                                         **common)}


__all__ = ["TRAINING_ADAPTER", "TrainingDockingD7Pool", "training_pools"]
