"""G20: where each journal task's appearing admission inputs come from (lazy family imports, CPU only).

``FAMILY_OF_TASK[task]`` is the family folder that carries the task (its pinned pools live in
tasks/<family>/appearing_pools); ``admission_inputs(task, vessel, tier, seed, env, episode)`` returns
the family's (D7 TaskContext, length scale, radius range or None) of that key's REAL journal layout,
the same frozen geometry the env builds (CONTEXT_R4 D16). No family module is imported before it is
needed, so this module stays importable without numpy-heavy families.
"""
from importlib import import_module

from . import config

FAMILY_OF_TASK = {
    "reach": "hazard_nav", "scatter": "hazard_nav", "ring": "hazard_nav", "fortress": "hazard_nav",
    "forced_crossing": "hazard_nav", "iceberg": "hazard_nav", "moving_basin": "hazard_nav",
    "path_following": "path_following", "path_hazard": "path_hazard", "harbour_mission": "harbor_mission",
    "station_keeping": "station_keeping", "docking": "docking",
}
_MODULE = {"hazard_nav": "hazard_nav.journal_hazard_nav", "path_following": "path_following.journal_path_following",
           "path_hazard": "path_hazard.path_hazard_journal", "harbor_mission": "harbor_mission.harbor_journal",
           "station_keeping": "station_keeping.journal_station", "docking": "docking.journal_docking"}


def _root():
    # this module is <root>._shared.conditions.appearing.families
    return __name__.rsplit(".", 4)[0]


def family_module(task):
    config.check_placement_task(task)
    return import_module(_root() + "." + _MODULE[FAMILY_OF_TASK[task]])


def admission_inputs(task, vessel, tier, seed, env, episode):
    module = family_module(task)
    if FAMILY_OF_TASK[task] == "hazard_nav":
        return module.appearing_admission_inputs(task, vessel, tier, seed, env, episode)
    return module.appearing_admission_inputs(vessel, tier, seed, env, episode)


def inputs_for(task, vessel, tier, seed):
    """inputs(env, episode) closure for admission.build_pool."""
    return lambda env, episode: admission_inputs(task, vessel, tier, seed, env, episode)


__all__ = ["FAMILY_OF_TASK", "family_module", "admission_inputs", "inputs_for"]
