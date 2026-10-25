"""G20b CPU fixture (not a test module): small appearing pools PINNED in a temporary family folder.

Since G20b the production appearing pools live in the pool bundle beside the release (G3c option B):
the in-repo manifests pin them, the files are not in the repository. The env tests therefore admit a
small key grid on the REAL journal layouts with the REAL admission (admission.build_pool, the active
ladder), write the pool files and a MANIFEST.json per family into a temporary folder and point the
pool module's family_pool_dir at it, so the env's own pinned-pool path (journal_env_mixin ->
pool.load_pinned_pool: manifest pin, size, SHA-256, digests) runs unchanged with pool_pinned true.

Every function takes the module objects of the caller's package root (each test loads the task tree
under its own root), so the env and this fixture share one pool module. Load this file by path:
    spec = importlib.util.spec_from_file_location("appearing_pin_fixture", <this path>)
"""
from __future__ import annotations

from pathlib import Path
import tempfile

_CACHE = {}


def build(admission, families, task, vessel, num_envs, episodes, seed=42, tier=2):
    """AppearingPool of a small grid (real layouts, real admission, active ladder); cached per process."""
    key = (id(admission), task, vessel, int(num_envs), int(episodes), int(seed), int(tier))
    if key not in _CACHE:
        _CACHE[key] = admission.build_pool(task, vessel, int(seed), int(tier), int(num_envs), int(episodes),
                                           families.inputs_for(task, vessel, int(tier), int(seed)))
    return _CACHE[key]


def forced(pool_module, pool, reject, reason="forced_test_rejection"):
    """Copy of ``pool`` with the keys in ``reject`` ((env, episode), ...) replaced by not-admitted records."""
    keys = []
    for ident in pool.offered:
        item = pool._keys[ident]
        record = {"env": ident[0], "episode": ident[1], "geometry_id": item["geometry_id"],
                  "placement": item["placement"], "reasons": list(item["reasons"]), "doses": dict(item["doses"])}
        if ident in set(tuple(r) for r in reject):
            record.update(reasons=[(0, reason)], doses={})
        keys.append(record)
    return pool_module.AppearingPool(task=pool.task, vessel_id=pool.vessel_id, seed=pool.seed, tier=pool.tier,
                                     suite_identity=pool.suite_identity, num_envs=pool.grid["num_envs"],
                                     episodes=pool.grid["episodes"], keys=keys)


def install(pool_module, families, pools, folder=None):
    """Write ``pools`` (AppearingPool objects) into <folder>/<family>/appearing_pools with a MANIFEST.json
    each and point pool_module.family_pool_dir there; returns (folder, restore)."""
    # tempfile picks the temporary folder itself: no literal environment read here (eval_protocol's
    # scientific-variable scan treats every literal read in the task tree as a scientific variable)
    folder = Path(folder or tempfile.mkdtemp(prefix="g20b_pins_"))
    by_family = {}
    for pool in pools:
        family = families.FAMILY_OF_TASK[pool.task]
        target = folder / family / pool_module.POOL_DIRNAME
        name = pool_module.pool_file_name(pool.task, pool.vessel_id, pool.seed)
        sha = pool_module.write_pool_file(pool, target / name, provenance={"fixture": "g20b appearing_pin_fixture"})
        by_family.setdefault(family, []).append(pool_module.manifest_entry(pool, name, sha,
                                                                           (target / name).stat().st_size))
    for family, entries in by_family.items():
        pool_module.write_manifest(folder / family / pool_module.POOL_DIRNAME, entries,
                                   provenance={"fixture": "g20b appearing_pin_fixture"})
    original = pool_module.family_pool_dir

    def family_pool_dir(family):
        return folder / family / pool_module.POOL_DIRNAME

    pool_module.family_pool_dir = family_pool_dir

    def restore():
        pool_module.family_pool_dir = original

    return folder, restore


__all__ = ["build", "forced", "install"]
