"""CPU fixtures for the G3 appearing-suite tests (never imports Isaac Lab; not a test module).

The repository task tree is loaded under the private root ``_g3t`` as bare
namespace packages (no package initializer runs, so no gym registration and no
simulator import), torch runs one thread. Pools for tests are admitted on the
REAL journal layouts of a small key grid (every task through
appearing/families.admission_inputs, the active ladder). G20b: the production
pools live in the pool bundle beside the release (only their pins are in
tasks/<family>/appearing_pools/MANIFEST.json), so tests use ``reference_pool``
(a 16 x 2 grid admitted in memory) where they used the pinned files; the two
v2.4 G3 files (ladder A) are read by ``legacy_pool`` without the suite check.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import torch  # noqa: E402

torch.set_num_threads(1)
REPO = Path(__file__).resolve().parents[5]
ROOT = "_g3t"
SEED = 42
TIER = 2
BB, WAMV = "blueboat_v2", "wamv_vrx_v2_opt1"

if ROOT not in sys.modules:
    _root = ModuleType(ROOT)
    _root.__path__ = [str(REPO / "tasks")]
    sys.modules[ROOT] = _root
    for _folder in sorted((REPO / "tasks").iterdir()):
        if _folder.is_dir() and _folder.name != "__pycache__":
            _sub = ModuleType(ROOT + "." + _folder.name)
            _sub.__path__ = [str(_folder)]
            sys.modules[ROOT + "." + _folder.name] = _sub


def load(name):
    return importlib.import_module(ROOT + "." + name)


config = load("_shared.conditions.appearing.config")
stamp = load("_shared.conditions.appearing.stamp")
placement = load("_shared.conditions.appearing.placement")
pool_module = load("_shared.conditions.appearing.pool")
runtime = load("_shared.conditions.appearing.runtime")
admission = load("_shared.conditions.appearing.admission")
dynamic = load("_shared.conditions.dynamic")
pairing = load("_shared.conditions.pairing")
families = load("_shared.conditions.appearing.families")
ladder = load("_shared.conditions.appearing.ladder")
plans = load("_shared.journal_cfg_v2")
hazard = load("hazard_nav.journal_hazard_nav")
path = load("path_following.journal_path_following")


def envelope(vessel):
    return hazard.vessel_envelope(vessel)


def inputs(task, vessel, seed=SEED, tier=TIER):
    """(context, length scale, radius range) of a key (the docking regions item is dropped; see inputs4)."""
    return lambda env, episode: tuple(families.admission_inputs(task, vessel, tier, seed, env, episode))[:3]


def inputs4(task, vessel, seed=SEED, tier=TIER):
    """The family's full admission inputs (context, scale, radius range[, regions])."""
    return lambda env, episode: tuple(families.admission_inputs(task, vessel, tier, seed, env, episode))


_POOLS = {}


def small_pool(task, vessel, num_envs=2, episodes=2, seed=SEED):
    key = (task, vessel, num_envs, episodes, seed)
    if key not in _POOLS:
        _POOLS[key] = admission.build_pool(task, vessel, seed, TIER, num_envs, episodes, inputs4(task, vessel, seed))
    return _POOLS[key]


def reference_pool(task, vessel):
    """G20b: a 16 x 2 grid of the task admitted in memory with the active ladder (the tests' stand-in for
    the production pool, which lives in the pool bundle)."""
    return small_pool(task, vessel, num_envs=16, episodes=2)


LEGACY_FAMILY = {"scatter": "hazard_nav", "path_following": "path_following"}


def legacy_pool(task, vessel):
    """(pool, sha256) of the pinned v2.4 G3 pool (ladder A, legacy name), read WITHOUT the suite check."""
    folder = pool_module.family_pool_dir(LEGACY_FAMILY[task])
    manifest, _ = pool_module.read_manifest(folder)
    name = pool_module.pool_file_name(task, vessel)
    entry = manifest["pools"][name]
    return pool_module.read_pool_file(folder / name, expected_sha256=entry["sha256"], task=task, vessel_id=vessel,
                                      seed=SEED, tier=TIER), entry["sha256"]


def legacy_placements(task, vessel):
    """{(env, episode): placement or None} of the pinned v2.4 G3 file, read from its raw document (no pool object,
    so it works under every ladder option, also one whose dose set differs from the file's)."""
    import hashlib
    import json
    import lzma
    folder = pool_module.family_pool_dir(LEGACY_FAMILY[task])
    manifest, _ = pool_module.read_manifest(folder)
    name = pool_module.pool_file_name(task, vessel)
    data = (folder / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["pools"][name]["sha256"]:
        raise ValueError("legacy pool %s does not match its pin" % name)
    doc = json.loads(lzma.decompress(data).decode("ascii"))
    return {(item["env"], item["episode"]): item["placement"] for item in doc["keys"]}


def pinned_pool(task, vessel):
    """(pool, record) through the loader: the in-repo pin; the file from $USVBENCH_D7_POOL_DIR when set."""
    runtime_dir, source = pool_module.runtime_pool_dir()
    return pool_module.load_pinned_pool(families.FAMILY_OF_TASK[task], task, vessel, SEED, TIER,
                                        runtime_dir=runtime_dir, runtime_source=source)


def key(task, env, episode, seed=SEED):
    return dynamic.EpisodeKey(seed, env, episode, config.hazard_group(task))


def first_admitted(pool):
    for env, episode in pool.offered:
        if pool.is_admitted(env, episode):
            return env, episode
    raise AssertionError("pool has no admitted key")


def first_not_admitted(pool):
    for env, episode in pool.offered:
        if not pool.is_admitted(env, episode):
            return env, episode
    return None


def straight_run(bridge, env, start, heading, speed, physics_dt, decimation, steps):
    """Drive a row along a straight line (exact poses); returns per-control-step (tick, gap, triggered)."""
    out = []
    tick = 0
    x0, y0 = start
    pose = dynamic.Pose(float(x0), float(y0), float(heading), float(speed))
    bridge.step({env: 0}, {env: pose})
    for _ in range(steps):
        for _sub in range(decimation):
            bridge.step({env: tick}, {env: pose})
            tick += 1
            d = speed * physics_dt * tick
            pose = dynamic.Pose(float(x0 + d * math.cos(heading)), float(y0 + d * math.sin(heading)),
                                float(heading), float(speed))
        bridge.step({env: tick}, {env: pose}, check_trigger=True)
        out.append((tick, pose, bridge.triggered(env)))
    return out
