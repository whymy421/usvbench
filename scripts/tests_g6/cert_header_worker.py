"""G6b worker: the REAL cert_v3.CertificateRun journal header over every journal env of one family.

usage: python -B cert_header_worker.py <family> [suite token ...]
  family: hazard_nav | path_following | station_keeping | docking | path_hazard | harbor_mission

For each (journal task, vessel) of the family: the REAL E5 template cfg and the REAL family env,
built on the family's own CPU fixtures (stubbed Isaac Lab DirectRLEnv / robot, no simulator),
with the REAL PINNED D7 pool located exactly as on the GPU host (P5): $USVBENCH_D7_POOL_DIR for
hazard_nav / path_following / path_hazard / harbor_mission, cfg.d7_pool_dir (= --set
d7_pool_dir) for docking, the release's d7_pools/INDEX.json for station keeping. Then the REAL
cert_v3.CertificateRun is constructed on the journal_header path (D16/D19/D22/D23/D24 checks run
in its constructor) and the D7 header block is checked for the required fields. One JSON line
per id: ``G6B-HEADER {...}``; exit code 0 only if every id passes.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
TASKS = REPO / "tasks"
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
HAZARD_TASKS = ("reach", "scatter", "ring", "fortress", "forced_crossing", "iceberg", "moving_basin")
HEX = set("0123456789abcdef")
TOKENS = []          # suite tokens (argv[2:]), resolved at the eval seed 42 as resolve_protocol does


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _envs_hazard_nav():
    sys.path.insert(0, str(TASKS / "hazard_nav"))
    import journal_test_support as S
    S.install_isaaclab_stubs()
    fam = S.load_family()
    for task in HAZARD_TASKS:
        for vessel in VESSELS:
            def build(task=task, vessel=vessel):
                cfg = S.journal_cfg(fam, task, vessel, tokens=TOKENS, num_envs=2)      # pool=None: the pinned file pool
                return S.make_env(fam, cfg)
            yield task, vessel, build


def _envs_path_following():
    sys.path.insert(0, str(TASKS / "path_following" / "tests_journal"))
    import f3_fixtures as F
    _cfg_module, _env_module, templates = F.env_modules()
    for vessel in VESSELS:
        def build(vessel=vessel):
            cls = {"blueboat_v2": templates.PathFollowingBlueBoatJournalEnvCfg,
                   "wamv_vrx_v2_opt1": templates.PathFollowingWAMVJournalEnvCfg}[vessel]
            cfg = cls()
            cfg.scene.num_envs = 2
            cfg.visual.enable_water = False
            cfg.visual.enable_waypoint_markers = False
            cfg.visual.enable_path_line = False
            F.journal_config.apply_journal_conditions(cfg, F.journal_config.resolve_journal_conditions(
                presets=list(TOKENS), seed=42))
            templates.journal_validate(cfg)                             # no journal_d7_pool_path: pinned
            return F.make_env(cfg)
        yield "path_following", vessel, build


def _envs_station_keeping():
    sys.path.insert(0, str(TASKS / "station_keeping" / "tests_journal"))
    import station_journal_fixtures as F
    F.install_isaaclab_stubs()
    for vessel in VESSELS:
        def build(vessel=vessel):
            return F.make_env(F.journal_cfg(vessel, num_envs=2, tokens=TOKENS))          # pool from d7_pools/INDEX.json
        yield "station_keeping", vessel, build


def _envs_docking():
    T = _load("g6b_docking_test", TASKS / "docking" / "test_journal_docking_env.py")
    for vessel in VESSELS:
        def build(vessel=vessel):
            cfg = T.CFG_CLASS[vessel]()
            cfg.sim.gravity = (0.0, 0.0, -9.81)
            cfg.scene.num_envs = 2
            cfg.visual.enable_water = False
            T.JC.apply_journal_conditions(cfg, T.JC.resolve_journal_conditions(presets=list(TOKENS), seed=42))
            cfg.journal_d7_pool = None
            bundle = os.environ.get("USVBENCH_D7_POOL_DIR")
            if bundle:
                cfg.d7_pool_dir = bundle                                 # the queue's --set d7_pool_dir
            return T.docking_env.DockingEnv(cfg)
        yield "docking", vessel, build


def _envs_path_hazard():
    sys.path.insert(0, str(TASKS / "path_hazard" / "tests_journal"))
    import ph_fixtures as F
    for vessel in VESSELS:
        def build(vessel=vessel):
            cfg = F.journal_cfg(vessel, tokens=TOKENS, num_envs=2, pool=object())
            cfg.journal_d7_pool = ""                                     # the default: the pinned file pool
            return F.make_env(cfg)
        yield "path_hazard", vessel, build


def _envs_harbor_mission():
    H = _load("g6b_harbour_fixtures", TASKS / "harbor_mission" / "harbour_journal_fixtures.py")
    for vessel in VESSELS:
        def build(vessel=vessel):
            cfg = H.journal_cfg(vessel, TOKENS, num_envs=2, pool=False)          # no object pool: the pinned file
            spec = H.vehicles_v2.get_vehicle_v2(vessel)
            H.FakeRigidObject.MASS = spec.mass_kg
            H.FakeRigidObject.INERTIA = spec.inertia_com_kg_m2
            return H.env_module().HarborMissionEnv(cfg)
        yield "harbour_mission", vessel, build


FAMILIES = {"hazard_nav": _envs_hazard_nav, "path_following": _envs_path_following,
            "station_keeping": _envs_station_keeping, "docking": _envs_docking,
            "path_hazard": _envs_path_hazard, "harbor_mission": _envs_harbor_mission}


def certificate(cert_v3, env, task_id):
    """The REAL writer on the journal_header path (constructor preflight checks run here)."""
    provenance = {k: "a" * 64 for k in ("tasks_sha256", "scripts_sha256", "evaluator_sha256")}
    provenance.update(tasks_root="t/tasks", scripts_root="t/scripts", environment_source="t/tasks/x/env.py",
                      evaluator_path="t/scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
    provenance["tree_sha256"] = cert_v3.legacy_tree_digest(provenance["tasks_sha256"], provenance["scripts_sha256"])
    runtime = {k: "test" for k in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                   "isaac_lab_version", "physx_version")}
    ticks = itertools.count()
    return cert_v3.CertificateRun(base=env, env_cfg=env.cfg, evaluator=__file__, task=task_id, controller="g6b",
                                  controller_config={}, checkpoint=None, eval_seed=42, level=0,
                                  episodes_requested=2, overrides={},
                                  argv=["eval.py"] + (["--suite", *TOKENS] if TOKENS else []),
                                  cli_args={"suite": list(TOKENS)} if TOKENS else {},
                                  provenance=provenance, runtime=runtime, synchronize=lambda: None,
                                  clock=lambda: next(ticks) / 1000.0)


def header_errors(header):
    errors = []
    block = header.get("dynamic_hazards_header")
    if not isinstance(block, dict):
        return ["no dynamic_hazards_header"]
    manifest = block.get("pool_manifest_sha256")
    if not (isinstance(manifest, str) and len(manifest) == 64 and set(manifest) <= HEX):
        errors.append("D19 pool_manifest_sha256 missing or not a SHA-256: %r" % (manifest,))
    digest = block.get("pool_digest")
    if not (isinstance(digest, str) and len(digest) == 64 and set(digest) <= HEX):
        errors.append("pool_digest missing or not a SHA-256: %r" % (digest,))
    if block.get("pool_pinned", True) is False:
        errors.append("D23 pool_pinned is False")
    grid = block.get("grid")
    if not (isinstance(grid, dict) and all(type(grid.get(k)) is int and grid[k] > 0 for k in ("num_envs", "episodes"))):
        errors.append("D24 grid missing: %r" % (grid,))
    counts = block.get("per_dose_counts")
    if not (isinstance(counts, list) and len(counts) == 5 and all(isinstance(r, list) and len(r) == 3 for r in counts)):
        errors.append("D22 per_dose_counts are not five [offered, admitted, not_admitted] rows: %r" % (counts,))
    if not isinstance(header.get("pairing"), dict):
        errors.append("no pairing block")
    return errors


def main():
    family = sys.argv[1]
    TOKENS[:] = sys.argv[2:]
    for entry in (str(SCRIPTS), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    builders = list(FAMILIES[family]())
    sys.path.insert(0, str(SCRIPTS))
    import cert_v3
    import eval_protocol
    plans = eval_protocol.journal_plans()
    failed = 0
    for task, vessel, build in builders:
        task_id = plans.JOURNAL_ENV_IDS[(task, vessel)]
        row = {"family": family, "task_id": task_id, "suite": list(TOKENS)}
        try:
            env = build()
            run = certificate(cert_v3, env, task_id)
            errors = header_errors(run.header)
            block = run.header["dynamic_hazards_header"]
            row.update(ok=not errors, errors=errors, pool_digest=block.get("pool_digest"),
                       pool_manifest_sha256=block.get("pool_manifest_sha256"), grid=block.get("grid"))
        except Exception as exc:  # noqa: BLE001 - every failure is a row
            row.update(ok=False, errors=["%s: %s" % (type(exc).__name__, exc)],
                       where=[l.strip() for l in traceback.format_exc().splitlines()[-6:]])
        failed += not row["ok"]
        print("G6B-HEADER " + json.dumps(row, sort_keys=True), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
