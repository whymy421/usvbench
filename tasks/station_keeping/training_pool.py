"""Training-only D7 (and appearing) pools of journal station keeping, online admission (G19b, P3 / F4).

``TrainingStationD7Pool`` offers the interface StationKeepingEnv reads from ``cfg.journal_d7_pool`` when the
object has ``assignment`` (station_keeping_env._journal_post_init; no env change needed): ``check_env``,
``header_fields``, ``key_status``, ``bind_context``, ``assignment`` plus the mixin's ``pool_digest``,
``per_dose_counts``, ``keys_offered``, ``keys_admitted``, ``joint``:
  * a key whose mixture component is not ``dynamic``: ``key_status`` = beyond_offered_grid (the env skips the
    context bind) and the assignment is the family's D24 ``NotAdmittedAssignment`` (kind beyond_offered_grid):
    no admission, no intruder;
  * a component key is admitted ONCE on its own E5 layout (``build_station_layout`` -> ``station_task_context``,
    the functions of the offline ``admit_station_grid``) with E4 ``admit_grid`` (non-joint; D1 taken) and serves
    the E4 Assignment at D1, or the D16 ``NotAdmittedAssignment`` (kind e4_rejected) with the D1 reasons; the env's
    bound layout context must carry the same geometry id.
"""
from __future__ import annotations

from .._shared import training_hazards as TH
from . import journal_station as J

TRAINING_ADAPTER = True


class TrainingStationD7Pool:
    TRAINING_ADAPTER = True

    def __init__(self, *, mixture, task, vessel, seed, tier, num_envs, suite, cache, purpose="training",
                 registry=None, registry_stamp=None, admit_grid=None):
        if task != J.TASK:
            raise TH.TrainingHazardError("the station-keeping family carries only %r" % (J.TASK,))
        self.gate = TH.TrainingHazardGate(suite="dynamic", mixture=mixture, task=task, vessel=vessel, seed=seed,
                                          num_envs=num_envs, hazard_group=J.HAZARD_GROUP, cache=cache,
                                          purpose=purpose, registry=registry, registry_stamp=registry_stamp)
        if mixture is None:
            if suite.enabled or int(suite.dose) != 0:
                raise TH.TrainingHazardError("the calm arm runs the dynamic suite off (D0)")
        elif not (suite.enabled and int(suite.dose) == self.gate.level and suite.vessel_id == vessel):
            # owner decision J10: the training dose is D1, except BlueBoat SK = D2 (D1 not applicable by design)
            raise TH.TrainingHazardError("the training D7 pool needs the env's enabled D%d DynamicSuiteConfig"
                                         % self.gate.level)
        dyn = J._dynamic()
        self.suite = dyn.DynamicSuiteConfig(enabled=True, vessel_id=vessel)       # the offline admission suite
        self.vessel, self.seed, self.tier, self.num_envs = str(vessel), int(seed), int(tier), int(num_envs)
        self._admit_grid = dyn.admit_grid if admit_grid is None else admit_grid
        self.admission_calls = 0
        self.pool_digest = self.gate.adapter_digest
        self.joint = False
        self.keys_offered, self.keys_admitted, self.keys_not_admitted = 0, 0, 0
        self.per_dose_counts = tuple((0, 0, 0) for _ in range(5))
        self.e4_per_dose_counts = tuple((0, 0) for _ in range(5))
        self.offered_grid = {"num_envs": self.num_envs, "episodes": 0}
        self.grid = dict(self.offered_grid)
        self.source = None
        self._contexts = {}

    # -- admission (pure function of the key)
    def _compute(self, key):
        def run():
            self.admission_calls += 1
            layout = J.build_station_layout(self.tier, self.vessel, self.seed, int(key.env), int(key.episode))
            context = J.station_task_context(layout, self.vessel)
            geometry = J.layout_hash(layout)
            if context.geometry_id != geometry:
                raise TH.TrainingHazardError("station context geometry differs from the E5 layout hash")
            pool = self._admit_grid([(context, key, self.suite)], joint=False, workers=1)
            level = self.gate.level                     # J10: D1, or D2 for BlueBoat SK
            found = pool.assignments.get((key, level))
            if found is not None:
                return ("admitted", found, geometry)
            reasons = tuple((int(r.dose), str(r.reason), int(r.sampled_states)) for r in pool.rejections
                            if r.key == key and int(r.dose) == level)
            if not reasons:     # G23e: the training tier is not applicable by design (BlueBoat SK D1): no intruder
                reasons = tuple((int(r.dose), str(r.reason), 0) for r in getattr(pool, "not_applicable", ())
                                if r.key == key and int(r.dose) == level)
            if not reasons:
                raise TH.TrainingHazardError("D%d neither admitted nor rejected for %s" % (level, key))
            return ("not_admitted", reasons, geometry)
        return run

    @staticmethod
    def _outcome(value):
        kind, item, _ = value
        return (True, []) if kind == "admitted" else (False, [r for _, r, _ in item])

    def _served(self, key):
        return self.gate.cache.get("dynamic", key, self._compute(key), self._outcome)

    # -- interface read by StationKeepingEnv
    def check_env(self, *, vessel, seed, tier, num_envs, suite=None):
        if vessel != self.vessel or int(seed) != self.seed or int(tier) != self.tier or int(num_envs) > self.num_envs:
            raise ValueError("training D7 pool (%s, seed %d, tier %d, %d envs) does not match the run"
                             % (self.vessel, self.seed, self.tier, self.num_envs))
        if suite is not None:
            from dataclasses import asdict
            want = {k: v for k, v in asdict(self.suite).items() if k not in ("enabled", "dose")}
            have = {k: v for k, v in asdict(suite).items() if k not in ("enabled", "dose")}
            if want != have:
                raise ValueError("runtime D7 suite parameters differ from the training admission suite")

    def header_fields(self):
        return {"per_dose_counts": [list(r) for r in self.per_dose_counts],
                "per_dose_counts_columns": list(J.PER_DOSE_COLUMNS),
                "e4_per_dose_counts": [list(r) for r in self.e4_per_dose_counts], "keys_not_admitted": 0,
                "not_admitted": [], "not_admitted_reason_counts": {}, "grid": dict(self.offered_grid),
                "pool_file_sha256": None, "not_admitted_policy": J.NOT_ADMITTED_RULE,
                "beyond_offered_grid_policy": J.BEYOND_GRID_RULE, "training": self.gate.training_header()}

    def key_status(self, key):
        try:
            self.gate.check_key(key, self.gate.level)
        except KeyError:
            raise ValueError("training D7 pool holds no key %r (another seed, group or env range)" % (key,)) from None
        if not self.gate.component_on(key):
            return J.BEYOND_OFFERED_GRID
        return J.ADMITTED if self._served(key)[0] == "admitted" else J.NOT_ADMITTED

    def not_admitted_reasons(self, env_index, episode_index):
        key = J.episode_key(self.seed, env_index, episode_index)
        if not self.gate.component_on(key):
            return []
        kind, item, _ = self._served(key)
        return [] if kind == "admitted" else [list(r) for r in item]

    def layout_hash_for(self, env_index, episode_index):
        key = J.episode_key(self.seed, env_index, episode_index)
        return self._served(key)[2] if self.gate.component_on(key) else None

    def bind_context(self, key, context):
        if not self.gate.component_on(key):
            raise KeyError(key)
        geometry = self._served(key)[2]
        if geometry != context.geometry_id:
            raise ValueError("env layout %s for %s differs from the admitted layout %s"
                             % (context.geometry_id, key, geometry))
        self._contexts = {k: v for k, v in self._contexts.items() if k.env != key.env}
        self._contexts[key] = context

    def assignment(self, key, dose):
        self.gate.check_key(key, dose)
        if not self.gate.component_on(key):
            self.gate.cache.note_d0("dynamic", key)
            out = J.NotAdmittedAssignment(key, [[0, J.BEYOND_OFFERED_GRID, 0]], kind=J.BEYOND_OFFERED_GRID,
                                          pool_digest=self.pool_digest, vessel=self.vessel, geometry_id=None,
                                          requested_dose=int(dose))
        else:
            if key not in self._contexts:
                raise ValueError("bind the env's real layout context for %s before requesting its assignment" % (key,))
            kind, item, geometry = self._served(key)
            out = item if kind == "admitted" else J.NotAdmittedAssignment(
                key, item, kind=J.E4_REJECTED, pool_digest=self.pool_digest, vessel=self.vessel,
                geometry_id=geometry, requested_dose=int(dose))
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
    """{"d7": TrainingStationD7Pool, "appearing": shared adapter where the suite is wired, else None}."""
    resolved = cfg.journal_conditions
    n = int(cfg.scene.num_envs if num_envs is None else num_envs)
    tier = int(getattr(cfg, "journal_geometry_tier", 2))
    common = dict(mixture=mixture, task=cfg.journal_task, vessel=cfg.vehicle, seed=int(cfg.seed), tier=tier,
                  num_envs=n, cache=cache, purpose=purpose, registry=registry, registry_stamp=registry_stamp)
    d7 = TrainingStationD7Pool(suite=resolved.dynamic_config(cfg.vehicle), **common)
    return {"d7": d7, "appearing": TH.appearing_pool_for(J, **common)}


__all__ = ["TRAINING_ADAPTER", "TrainingStationD7Pool", "training_pools"]
