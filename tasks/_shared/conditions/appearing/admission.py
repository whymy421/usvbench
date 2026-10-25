"""Offline admission driver of the appearing suite (G3 rule 6; serial core, process-pool safe jobs).

``admit_job((context, key, cfg, length_scale, radius_range_m[, regions]))`` is a pure,
deterministic function of its frozen inputs (no RNG state, no controller), so a
process pool gives exactly the serial records; ``build_pool`` assembles the
records of a rectangular grid (num_envs x episodes, offer order env-major) into
an AppearingPool. The family supplies ``inputs(env, episode) -> (TaskContext,
length_scale, radius_range_m or None[, regions])`` built from its REAL journal
layout of the key (the same frozen geometry the env builds, D16); ``regions``
(G20b) is the docking family's berth rectangle ({"berth": ...}), absent elsewhere.
"""
from ..dynamic.model import EpisodeKey
from . import config
from .placement import admit_key
from .pool import AppearingPool


def admission_suite(vessel_id):
    """The AppearingSuiteConfig the offline admission uses (dose set per witnessed rung)."""
    return config.AppearingSuiteConfig(enabled=True, dose=0, vessel_id=vessel_id)


def admit_job(job):
    context, key, cfg, length_scale, radius_range_m = job[:5]
    regions = job[5] if len(job) > 5 else None
    out = admit_key(context, key, cfg, length_scale=length_scale, radius_range_m=radius_range_m, regions=regions)
    return {"env": int(key.env), "episode": int(key.episode), "geometry_id": out["geometry_id"],
            "placement": out["placement"], "reasons": [tuple(r) for r in out["reasons"]], "doses": out["doses"],
            "optional_reasons": [tuple(r) for r in out.get("optional_reasons", ())]}     # G20d (decision 5B)


def grid_jobs(task, vessel_id, seed, num_envs, episodes, inputs, cfg=None):
    cfg = admission_suite(vessel_id) if cfg is None else cfg
    jobs = []
    for env in range(int(num_envs)):
        for episode in range(int(episodes)):
            item = tuple(inputs(env, episode))
            context, length_scale, radius_range_m = item[:3]
            regions = item[3] if len(item) > 3 else None
            if context.task != config.context_task(task) or context.envelope.vessel_id != vessel_id:
                raise ValueError("admission inputs belong to another task or vessel")
            key = EpisodeKey(int(seed), env, episode, config.hazard_group(task))
            job = (context, key, cfg, float(length_scale), radius_range_m)
            jobs.append(job if regions is None else job + (regions,))
    return jobs


def build_pool(task, vessel_id, seed, tier, num_envs, episodes, inputs, cfg=None, map_fn=map, meta=None):
    """AppearingPool of the grid (serial unless ``map_fn`` is an ordered parallel map)."""
    cfg = admission_suite(vessel_id) if cfg is None else cfg
    jobs = grid_jobs(task, vessel_id, seed, num_envs, episodes, inputs, cfg)
    records = list(map_fn(admit_job, jobs))
    identity = cfg.identity()          # G20b: the suite identity (dose and diagnostic gap override excluded)
    return AppearingPool(task=task, vessel_id=vessel_id, seed=seed, tier=tier, suite_identity=identity,
                         num_envs=num_envs, episodes=episodes, keys=records, meta=meta)


__all__ = ["admission_suite", "admit_job", "grid_jobs", "build_pool"]
