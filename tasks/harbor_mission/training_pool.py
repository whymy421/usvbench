"""Training-only D7 (and appearing) pools of the journal harbour mission, online admission (G19b, P3 / F6).

``TrainingHarbourD7Pool`` is the object HarborMissionEnv takes from an in-memory ``cfg.journal_d7_pool``
(harbor_mission_env._journal_init; no env change needed): ``assignment``, ``pool_digest``, counts,
``header_fields()`` (with the D24 ``grid``) and ``meta`` (``suite`` of the admission); the family bridge
(HarbourD7Bridge) recognises the family's ``NotAdmitted`` records:
  * a key whose mixture component is not ``dynamic``: the D24 ``NotAdmitted`` (beyond_grid=True, no geometry,
    no stage): the env compares nothing and the row runs at D0;
  * a component key is admitted ONCE on its own E5 layout (``harbor_journal.journal_layout`` ->
    ``d7_task_context`` with the vessel envelope, the functions of the offline ``admit_harbour_grid``) with E4
    ``admit_grid`` (non-joint; D1 taken) and serves the E4 Assignment at D1, or the D16 ``NotAdmitted`` with the
    D1 reasons, the layout geometry id and the strata stage (as the offline tool records them).
Mixin keys may carry the pre-v2 group d7_harbour_mission; they are mapped onto d7_harbour (HarbourPoolView rule).
"""
from __future__ import annotations

from importlib import import_module

from .._shared import training_hazards as TH
from .._shared import journal_cfg_v2 as plans
from .._shared import vehicles_v2
from . import harbor_d7_pool as P
from . import harbor_journal as HJ

TRAINING_ADAPTER = True


def _dynamic():
    from .._shared.conditions import dynamic
    return dynamic


def vessel_envelope(vessel):
    record = plans.vessel_record(vessel)
    envelope, _ = _dynamic().envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vessel),
                                                   reference_speed_mps=record.cruise_speed_m_s,
                                                   turning_radius_m=record.turning_radius_design_m)
    return envelope


class TrainingHarbourD7Pool:
    TRAINING_ADAPTER = True

    def __init__(self, *, mixture, task, vessel, seed, tier, num_envs, suite, cache, purpose="training",
                 registry=None, registry_stamp=None, admit_grid=None):
        if task != HJ.TASK:
            raise TH.TrainingHazardError("the harbour family carries only %r" % (HJ.TASK,))
        self.gate = TH.TrainingHazardGate(suite="dynamic", mixture=mixture, task=task, vessel=vessel, seed=seed,
                                          num_envs=num_envs, hazard_group=P.ADMITTED_GROUP, cache=cache,
                                          purpose=purpose, registry=registry, registry_stamp=registry_stamp)
        if mixture is None:
            if suite.enabled or int(suite.dose) != 0:
                raise TH.TrainingHazardError("the calm arm runs the dynamic suite off (D0)")
        elif not (suite.enabled and int(suite.dose) == TH.TRAINING_LEVEL and suite.vessel_id == vessel):
            raise TH.TrainingHazardError("the training D7 pool needs the env's enabled D1 DynamicSuiteConfig")
        from dataclasses import asdict
        dyn = _dynamic()
        self.admission_suite = dyn.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
        self.vessel, self.seed, self.tier, self.num_envs = str(vessel), int(seed), int(tier), int(num_envs)
        self.meta = {"vessel": self.vessel, "seed": self.seed, "tier": self.tier, "suite": asdict(self.admission_suite),
                     "source": TH.TRAINING_POOL_SOURCE, "training": self.gate.training_header()}
        self._admit_grid = dyn.admit_grid if admit_grid is None else admit_grid
        self.admission_calls = 0
        self.pool_digest = self.gate.adapter_digest
        self.keys_offered = self.keys_admitted = self.keys_not_admitted = 0
        self.joint = False
        self.file_sha256 = None
        self.per_dose_counts = tuple((0, 0, 0) for _ in range(5))
        self.grid = {"num_envs": self.num_envs, "episodes": 0}
        self._envelope = vessel_envelope(self.vessel)

    @staticmethod
    def admitted_key(key):
        return P.HarbourPoolView.admitted_key(key)

    def _compute(self, key):
        def run():
            self.admission_calls += 1
            dyn = _dynamic()
            layout = HJ.journal_layout(self.vessel, self.tier, self.seed, int(key.env), int(key.episode))
            context = HJ.d7_task_context(layout, self._envelope, dyn)
            adapters = import_module(dyn.__name__ + ".adapters")
            stage = adapters._choices(adapters.PROFILES[HJ.D7_TASK], context, key)["stage"].name
            pool = self._admit_grid([(context, key, self.admission_suite)], joint=False, workers=1)
            found = pool.assignments.get((key, TH.TRAINING_LEVEL))
            if found is not None:
                if found.candidate.geometry_id != layout.geometry_hash:
                    raise TH.TrainingHazardError("admitted candidate geometry differs from the layout (%r)" % (key,))
                return ("admitted", found, layout.geometry_hash, stage)
            reasons = tuple((int(r.dose), str(r.reason), int(r.sampled_states)) for r in pool.rejections
                            if r.key == key and int(r.dose) == TH.TRAINING_LEVEL)
            if not reasons:
                raise TH.TrainingHazardError("D1 neither admitted nor rejected for %s" % (key,))
            return ("not_admitted", reasons, layout.geometry_hash, stage)
        return run

    @staticmethod
    def _outcome(value):
        kind, item = value[0], value[1]
        return (True, []) if kind == "admitted" else (False, [r for _, r, _ in item])

    def assignment(self, key, dose):
        akey = self.admitted_key(key)
        self.gate.check_key(akey, dose)
        if not self.gate.component_on(akey):
            self.gate.cache.note_d0("dynamic", akey)
            out = P.NotAdmitted(akey, int(dose), None, None, ((0, P.BEYOND_GRID, 0),), self.pool_digest,
                                self.vessel, beyond_grid=True)
        else:
            kind, item, geometry, stage = self.gate.cache.get("dynamic", akey, self._compute(akey), self._outcome)
            out = item if kind == "admitted" else P.NotAdmitted(akey, int(dose), geometry, stage, item,
                                                                self.pool_digest, self.vessel)
        upcoming = self.gate.next_component_episode(akey.env, akey.episode)
        if upcoming is not None:
            nxt = type(akey)(self.seed, int(akey.env), upcoming, P.ADMITTED_GROUP)
            self.gate.cache.prefetch("dynamic", nxt, self._compute(nxt))
        return out

    def is_beyond_grid(self, key):
        akey = self.admitted_key(key)
        return not self.gate.component_on(akey)

    def header_fields(self):
        return {"per_dose_counts_fields": ["offered", "admitted", "not_admitted"], "keys_not_admitted": 0,
                "not_admitted_keys": [], "not_admitted_failing_reason_counts": {},
                "not_admitted_rule": TH.ONLINE_RULE, "pool_schema": P.POOL_SCHEMA, "pool_file_sha256": None,
                "pool_pinned": False, "pool_manifest_sha256": None, "grid": dict(self.grid),
                "beyond_grid_rule": P.BEYOND_GRID_RULE, "training": self.gate.training_header()}

    def training_header(self):
        return self.gate.training_header()

    def __deepcopy__(self, memo):
        return self                 # a runtime service, never configuration data


def training_pools(cfg, mixture, *, cache, purpose="training", registry=None, registry_stamp=None, num_envs=None):
    """{"d7": TrainingHarbourD7Pool, "appearing": shared adapter where the suite is wired, else None}."""
    resolved = cfg.journal_conditions
    n = int(cfg.scene.num_envs if num_envs is None else num_envs)
    tier = int(getattr(cfg, "journal_geometry_tier", 2))
    common = dict(mixture=mixture, task=cfg.journal_task, vessel=cfg.vehicle, seed=int(cfg.seed), tier=tier,
                  num_envs=n, cache=cache, purpose=purpose, registry=registry, registry_stamp=registry_stamp)
    d7 = TrainingHarbourD7Pool(suite=resolved.dynamic_config(cfg.vehicle), **common)
    return {"d7": d7, "appearing": TH.appearing_pool_for((HJ, P), **common)}


__all__ = ["TRAINING_ADAPTER", "TrainingHarbourD7Pool", "training_pools", "vessel_envelope"]
