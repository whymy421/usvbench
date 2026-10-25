"""Offline admission of the appearing-obstacle pools (CPU only, serial, deterministic).

For each (task, vessel) -- every journal task with a declared placement rule
(conditions/appearing/config.PLACEMENT_TASKS; G3 phase 1 scatter / path_following, G20 rollout,
G20b final rules) -- every key (eval seed, env < num_envs, episode < episodes, hazard group
g3_appearing_<task>) is offered ONCE on the key's REAL E5 journal layout (the family's own builder,
conditions/appearing/families.admission_inputs) and admitted jointly over every witnessed dose of the
ACTIVE ladder (conditions/appearing/ladder.py; placement.admit_key). Rejected keys are kept as explicit
not-admitted records (CONTEXT_R4 D16). Pools are written deterministically (xz canonical JSON).

Where (G20b, G3c option B): with --bundle DIR the pool FILES go to DIR (the flat pool bundle beside the
release, never the repo) with one sidecar <file>.entry.json each; --pin then writes DIR/MANIFEST.json
(SHA-256 and bytes of every pool, the admission-tool bytes and the LF code digests, D26 pattern) and
MERGES the pins into the in-repo family manifests tasks/<family>/appearing_pools/MANIFEST.json (the
loader's authority; entries a run does not write are kept verbatim). Without --bundle the files go to
the family folder as before (CPU fixtures only).

File names: every pool this tool writes is seed-qualified (<task>__<vessel>__s<seed>.json.xz, G20); the
two v2.4 G3 pools (<task>__<vessel>.json.xz, seed 42, ladder A) keep their pins and are never rewritten.

usage:
    python -B scripts/admit_appearing_pools.py --write  --bundle DIR [--tasks T1,T2] [--seed 42
                                                        --num-envs 64 --episodes 2 --tier 2]
    python -B scripts/admit_appearing_pools.py --pin    --bundle DIR   bundle manifest + in-repo pins
    python -B scripts/admit_appearing_pools.py --check  --bundle DIR [--scratch S]  re-admit, compare bytes
    python -B scripts/admit_appearing_pools.py --verify [--bundle DIR]  load every pinned pool (all digests,
                                                         from the bundle when given) and replay every
                                                         admitted witness
    --tasks LIST    comma-separated journal tasks (default: the two G3 tasks); "all" = PLACEMENT_TASKS
    --vessels LIST  comma-separated vessels (default: blueboat_v2,wamv_vrx_v2_opt1)
    --only T__V     task__vessel filter (repeatable)
    --report PATH   write the admission report (JSON) there as well
    --workers N     G20d: admit the keys of one pool with N worker processes (ordered map of the pure admit_job:
                    the records, the pool and its bytes are identical to the serial run); default 1 = serial
The process lowers itself to BelowNormal priority and runs torch with one thread (every worker too).
Parallel runs (G20c): with --bundle, --write and --check touch only the files of their own (task, vessel),
so one process per --only T__V may run side by side; --pin runs once, after every --write has finished.
Keys the finite search does not certify are kept as not-admitted records (reason
no_witness_in_frozen_search: not certified by the finite search, not proven infeasible).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
TOOL_VERSION = "g20d-admit-appearing-pools-v5"
BUNDLE_MANIFEST_SCHEMA = "usvbench.appearing_pool_bundle/1"
ENTRY_SUFFIX = ".entry.json"
LEGACY_TASKS = ("scatter", "path_following")
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
FAMILY_SOURCES = {"hazard_nav": "tasks/hazard_nav/journal_hazard_nav.py",
                  "path_following": "tasks/path_following/journal_path_following.py",
                  "path_hazard": "tasks/path_hazard/path_hazard_journal.py",
                  "harbor_mission": "tasks/harbor_mission/harbor_journal.py",
                  "station_keeping": "tasks/station_keeping/journal_station.py",
                  "docking": "tasks/docking/journal_docking.py"}


def _low_priority():
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")                  # pointer-width handle (T-base v2 runner fix)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000):
            raise OSError("SetPriorityClass failed")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _load():
    """The task tree as bare namespace packages (no family __init__, no gym registration, no simulator)."""
    if "tasks" not in sys.modules:
        root = types.ModuleType("tasks")
        root.__path__ = [str(REPO / "tasks")]
        sys.modules["tasks"] = root
        for folder in (REPO / "tasks").iterdir():
            if folder.is_dir() and folder.name != "__pycache__":
                sub = types.ModuleType("tasks." + folder.name)
                sub.__path__ = [str(folder)]
                sys.modules["tasks." + folder.name] = sub
    imp = importlib.import_module
    return types.SimpleNamespace(
        hazard=imp("tasks.hazard_nav.journal_hazard_nav"),
        families=imp("tasks._shared.conditions.appearing.families"),
        admission=imp("tasks._shared.conditions.appearing.admission"),
        pool=imp("tasks._shared.conditions.appearing.pool"),
        config=imp("tasks._shared.conditions.appearing.config"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _content(path):
    """Pool document without its provenance block (None when the file is missing)."""
    import lzma
    path = Path(path)
    if not path.is_file():
        return None
    doc = json.loads(lzma.decompress(path.read_bytes()).decode("ascii"))
    doc.pop("provenance", None)
    return doc


def _json_bytes(doc):
    """LF-terminated canonical-indent JSON bytes (hash-pinned data, D26)."""
    return (json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("ascii")


def provenance(seed, num_envs, episodes, tier, family=None):
    sources = [*(REPO / "tasks" / "_shared" / "conditions" / "appearing").glob("*.py"),
               REPO / "tasks" / "_shared" / "journal_layouts_v2.py",
               REPO / "tasks" / "_shared" / "journal_cfg_v2.py",
               REPO / "tasks" / "_shared" / "conditions" / "dynamic" / "feasibility.py",
               REPO / "tasks" / "_shared" / "conditions" / "dynamic" / "adapters.py",
               Path(__file__).resolve()]
    families = [family] if family else ["hazard_nav", "path_following"]
    sources += [REPO / FAMILY_SOURCES[f] for f in families]
    # code digests are of LF-normalised text (line endings never change a pool, D26 d)
    code = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(sources)}
    return {"tool": TOOL_VERSION, "seed": int(seed), "num_envs": int(num_envs), "episodes": int(episodes),
            "tier": int(tier), "code_lf_sha256": code, "python": sys.version.split()[0],
            "ladder": fam_ladder()}


def fam_ladder():
    """The active ladder option and its gap ratios (read from the pure-stdlib ladder module)."""
    lad = importlib.import_module("tasks._shared.conditions.appearing.ladder")
    return {"option": lad.ACTIVE_LADDER, "kind": lad.KIND,
            "gap_ratios": None if lad.GAP_RATIOS is None else list(lad.GAP_RATIOS),
            "gaps_m": None if lad.GAPS_M is None else {v: list(g) for v, g in lad.GAPS_M.items()},
            "kinematic_floor_m": dict(lad.KINEMATIC_FLOOR_M), "calibration_rung_m": dict(lad.CALIBRATION_RUNG_M),
            "gap_override_min_m": dict(lad.GAP_OVERRIDE_MIN_M),
            "calibration_sweep_m": {v: list(g) for v, g in lad.CALIBRATION_SWEEP_M.items()},
            "suite_version": lad.SUITE_VERSION}


def file_name(fam, folder, task, vessel, seed):
    """G20b: every newly written pool is seed-qualified (the v2.4 legacy names are pinned, never rewritten)."""
    return fam.pool.pool_file_name(task, vessel, seed)


def _worker_init():
    """G20d: a --workers child loads the task tree like the parent and lowers its own priority."""
    _low_priority()
    _load()


def admit(fam, task, vessel, seed, num_envs, episodes, tier, log, workers=1):
    start = time.monotonic()
    inputs = fam.families.inputs_for(task, vessel, tier, seed)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool_exec:
            pool = fam.admission.build_pool(task, vessel, seed, tier, num_envs, episodes, inputs,
                                            map_fn=lambda fn, jobs: list(pool_exec.map(fn, jobs, chunksize=1)))
    else:
        pool = fam.admission.build_pool(task, vessel, seed, tier, num_envs, episodes, inputs)
    seconds = time.monotonic() - start
    log("%s/%s: %d offered, %d admitted, %d not admitted, reasons %s (%.1f s)"
        % (task, vessel, pool.keys_offered, pool.keys_admitted, pool.keys_not_admitted,
           pool.reason_counts(), seconds))
    return pool, seconds


def _quantiles(values):
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(round(q * (len(v) - 1))))]  # noqa: E731
    return {"n": len(v), "min": v[0], "p10": pick(0.1), "p50": pick(0.5), "p90": pick(0.9), "max": v[-1]}


def report_entry(fam, pool, file_sha=None, size=None, seconds=None):
    trig = {}
    for dose in fam.config.WITNESSED_DOSES:
        if dose in pool.not_applicable_doses:                      # G20d: never offered (decision 2B)
            trig["D%d" % dose] = {"not_applicable_by_design": True}
            continue
        times, speeds, admitted, visible, occluded, kinds = [], [], 0, 0, 0, {}
        for env, episode in pool.offered:
            if pool.is_admitted(env, episode):
                a = pool.assignment(fam.admission.EpisodeKey(pool.seed, env, episode, pool.hazard_group), dose)
                if a.candidate is None:                              # optional dose without a witness (5B)
                    continue
                admitted += 1
                if a.candidate["nominal_trigger_s"] is not None:      # None: witness-floor rung not reached
                    times.append(a.candidate["nominal_trigger_s"])
                if "nominal_trigger_speed_mps" in a.candidate:
                    speeds.append(a.candidate["nominal_trigger_speed_mps"])
                flag = a.candidate.get("nominal_visible_at_trigger")
                visible += flag is True
                occluded += flag is False
                reason = str(a.witness_reason)
                kind = ("monotone_stitched" if reason.startswith("monotone_stitched") else
                        "fallback_beam" if "_fallback_beam_" in reason else "frozen_search")
                kinds[kind] = kinds.get(kind, 0) + 1
        trig["D%d" % dose] = {"admitted_keys": admitted, "nominal_trigger_reached_keys": len(times),
                              "nominal_trigger_s_min": min(times) if times else None,
                              "nominal_trigger_s_max": max(times) if times else None,
                              "nominal_visible_at_trigger": visible, "nominal_occluded_at_trigger": occluded,
                              "witness_kinds": dict(sorted(kinds.items()))}
        if speeds:
            trig["D%d" % dose].update(nominal_trigger_speed_mps_min=min(speeds),
                                      nominal_trigger_speed_mps_max=max(speeds))
    radii = [pool.placement(e, k)["cylinder"]["radius_m"] for e, k in pool.offered if pool.placement(e, k)]
    admitted_radii = [pool.placement(e, k)["cylinder"]["radius_m"] for e, k in pool.offered if pool.is_admitted(e, k)]
    cand_index = {}
    for e, k in pool.offered:
        if pool.is_admitted(e, k):
            index = pool.placement(e, k)["site"].get("candidate_index")
            cand_index[str(index)] = cand_index.get(str(index), 0) + 1
    return {"task": pool.task, "vessel_id": pool.vessel_id, "seed": pool.seed, "tier": pool.tier,
            "ladder": fam.config.LADDER_OPTION, "suite_version": fam.config.APPEARING_SUITE_VERSION,
            "grid": dict(pool.grid), "keys_offered": pool.keys_offered, "keys_admitted": pool.keys_admitted,
            "keys_not_admitted": pool.keys_not_admitted,
            "admitted_fraction": pool.keys_admitted / float(pool.keys_offered),
            "per_dose_counts": [list(r) for r in pool.per_dose_counts],
            "per_dose_counts_fields": list(fam.config.PER_DOSE_COUNT_FIELDS),
            "reason_counts_keys": pool.reason_counts(), "pool_digest": pool.pool_digest,
            "file_sha256": file_sha, "bytes": size, "admission_seconds": seconds,
            "placed_radius_m": {"min": min(radii) if radii else None, "max": max(radii) if radii else None},
            "admitted_radius_m": _quantiles(admitted_radii),
            "admitted_candidate_index": dict(sorted(cand_index.items(), key=lambda kv: (len(kv[0]), kv[0]))),
            "admission_policy": pool.admission_policy(), "dose_applicability": pool.dose_applicability(),
            "optional_not_admitted_keys": pool.optional_not_admitted_keys(),
            "optional_reason_counts": pool.optional_reason_counts(),
            "not_admitted_keys": pool.not_admitted_keys(),
            "nominal_trigger": trig,
            "trigger_gaps_m": None}


def main(argv=None):
    _low_priority()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--pin", action="store_true")
    parser.add_argument("--bundle", type=Path, default=None, help="pool bundle folder (outside the repo)")
    parser.add_argument("--scratch", type=Path, default=None, help="--check scratch folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--tier", type=int, default=2)
    parser.add_argument("--tasks", default=",".join(LEGACY_TASKS))
    parser.add_argument("--vessels", default=",".join(VESSELS))
    parser.add_argument("--only", action="append", default=[], help="task__vessel filter (repeatable)")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1, help="G20d: worker processes for one pool (ordered map)")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be 1..8")
    fam = _load()

    def log(text):
        print(text, flush=True)

    tasks = list(fam.config.PLACEMENT_TASKS) if args.tasks == "all" else [t for t in args.tasks.split(",") if t]
    vessels = [v for v in args.vessels.split(",") if v]
    for task in tasks:
        fam.config.check_placement_task(task)
    bundle = None if args.bundle is None else args.bundle.resolve()
    if bundle is not None and (bundle == REPO or REPO in bundle.parents):
        raise SystemExit("the pool bundle must lie outside the repository (G3c option B)")
    if args.pin:
        if bundle is None:
            raise SystemExit("--pin needs --bundle")
        return pin(fam, bundle, log)
    report = {"tool": TOOL_VERSION, "mode": "write" if args.write else "check" if args.check else "verify",
              "ladder": fam_ladder(), "pools": []}
    failures = []
    by_family = {}
    for task in tasks:
        by_family.setdefault(fam.families.FAMILY_OF_TASK[task], []).append(task)
    for family, family_tasks in by_family.items():
        folder = fam.pool.family_pool_dir(family)
        entries = []
        for task in family_tasks:
            for vessel in vessels:
                if args.only and "%s__%s" % (task, vessel) not in args.only:
                    continue
                name = file_name(fam, folder, task, vessel, args.seed)
                if args.verify:
                    record_pool, record = fam.pool.load_pinned_pool(
                        family, task, vessel, args.seed, args.tier, runtime_dir=bundle,
                        runtime_source=None if bundle is None else "--bundle")
                    contexts = {}
                    for env, episode in record_pool.offered:
                        if record_pool.is_admitted(env, episode):
                            contexts[(env, episode)] = fam.families.admission_inputs(
                                task, vessel, args.tier, args.seed, env, episode)[0]
                    record_pool.verify_witnesses(contexts)
                    witnesses = sum(len(record_pool._keys[i]["doses"]) for i in record_pool.offered)
                    log("VERIFIED %s (%s, from %s): digests recompute, %d admitted keys, %d witnesses replay"
                        % (record["file"], record["sha256"], record["path"], record_pool.keys_admitted, witnesses))
                    report["pools"].append(report_entry(fam, record_pool, record["sha256"],
                                                        Path(record["path"]).stat().st_size))
                    continue
                pool, seconds = admit(fam, task, vessel, args.seed, args.num_envs, args.episodes, args.tier, log,
                                      workers=args.workers)
                prov = provenance(args.seed, args.num_envs, args.episodes, args.tier, family)
                if args.write:
                    target = folder if bundle is None else bundle
                    sha = fam.pool.write_pool_file(pool, target / name, provenance=prov)
                    size = (target / name).stat().st_size
                    entry = fam.pool.manifest_entry(pool, name, sha, size)
                    entries.append(entry)
                    log("WROTE %s sha256=%s bytes=%d" % (target / name, sha, size))
                    rep = report_entry(fam, pool, sha, size, seconds)
                    report["pools"].append(rep)
                    if bundle is not None:
                        side = {"family": family, "manifest_entry": entry, "report": rep, "provenance": prov}
                        (bundle / (name + ENTRY_SUFFIX)).write_bytes(_json_bytes(side))
                else:
                    scratch = (args.scratch or Path(tempfile.gettempdir()) / "usvbench_admit_check").resolve()
                    scratch.mkdir(parents=True, exist_ok=True)
                    sha = fam.pool.write_pool_file(pool, scratch / name, provenance=prov)
                    manifest, _ = fam.pool.read_manifest(folder)
                    pinned = manifest["pools"].get(name, {}).get("sha256")
                    source = folder if bundle is None else bundle
                    exact = pinned == sha and (source / name).is_file() and _sha(source / name) == sha
                    # G20: the file embeds the code digests (provenance), so a code change that leaves the
                    # admission unchanged still changes the bytes; the CONTENT check drops only provenance
                    content = _content(source / name) == _content(scratch / name)
                    ok = exact or content
                    log(("OK " if exact else "OK-CONTENT " if content else "DIFF ")
                        + "%s re-admitted sha256=%s pinned=%s" % (name, sha, pinned))
                    if not ok:
                        failures.append(name)
                    report["pools"].append(report_entry(fam, pool, sha, (scratch / name).stat().st_size, seconds))
        if args.write and entries and bundle is None:
            try:
                existing, _ = fam.pool.read_manifest(folder)
                kept = [e for n, e in sorted(existing["pools"].items()) if n not in {x["file"] for x in entries}]
                previous = existing.get("provenance", {})
            except FileNotFoundError:
                kept, previous = [], {}
            prov = provenance(args.seed, args.num_envs, args.episodes, args.tier, family)
            if kept:
                prov = dict(prov, kept_entries_from_previous_manifest=sorted(e["file"] for e in kept),
                            previous_provenance=previous)
            manifest_sha = fam.pool.write_manifest(folder, kept + entries, provenance=prov)
            log("WROTE %s sha256=%s (%d kept verbatim, %d written)"
                % (folder / fam.pool.MANIFEST_NAME, manifest_sha, len(kept), len(entries)))
    for entry in report["pools"]:
        gaps = fam.config.trigger_gaps_m(fam.hazard.vessel_envelope(entry["vessel_id"]).t_avoid_s, entry["vessel_id"])
        entry["trigger_gaps_m"] = gaps
    if args.write and bundle is not None:
        log("bundle files written; run --pin --bundle %s to write the bundle manifest and the in-repo pins" % bundle)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="ascii")
        log("REPORT %s" % args.report)
    if failures:
        log("FAILED: " + ", ".join(failures))
        return 1
    return 0


def pin(fam, bundle, log):
    """Bundle MANIFEST.json (every pool + sidecar, admission-tool bytes) and the merged in-repo family pins."""
    sides = sorted(bundle.glob("*" + ENTRY_SUFFIX))
    if not sides:
        raise SystemExit("no %s sidecars in %s (run --write --bundle first)" % (ENTRY_SUFFIX, bundle))
    pools, by_family = {}, {}
    for side_path in sides:
        side = json.loads(side_path.read_bytes().decode("ascii"))
        entry = side["manifest_entry"]
        name = entry["file"]
        path = bundle / name
        if not path.is_file() or _sha(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            raise SystemExit("bundle pool %s does not match its sidecar (missing, SHA-256 or size)" % name)
        pool = fam.pool.read_pool_file(path, expected_sha256=entry["sha256"], task=entry["task"],
                                       vessel_id=entry["vessel_id"], seed=entry["seed"], tier=entry["tier"])
        if pool.pool_digest != entry["pool_digest"]:
            raise SystemExit("bundle pool %s digest differs from its sidecar" % name)
        pools[name] = {"family": side["family"], "sha256": entry["sha256"], "bytes": int(entry["bytes"]),
                       "pool_digest": entry["pool_digest"], "task": entry["task"], "vessel_id": entry["vessel_id"],
                       "seed": entry["seed"], "tier": entry["tier"], "grid": entry["grid"],
                       "keys_offered": entry["keys_offered"], "keys_admitted": entry["keys_admitted"],
                       "keys_not_admitted": entry["keys_not_admitted"], "reason_counts": entry["reason_counts"],
                       "admission_seconds": side["report"].get("admission_seconds"),
                       "sidecar": side_path.name, "sidecar_sha256": _sha(side_path),
                       "code_lf_sha256": side["provenance"]["code_lf_sha256"]}
        by_family.setdefault(side["family"], []).append(entry)
    tool = Path(__file__).resolve()
    manifest = {"schema": BUNDLE_MANIFEST_SCHEMA, "tool": TOOL_VERSION, "ladder": fam_ladder(),
                "admission_tool": {"path": tool.relative_to(REPO).as_posix(), "sha256": _sha(tool),
                                   "lf_sha256": hashlib.sha256(tool.read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
                "pools": pools, "total_bytes": sum(p["bytes"] for p in pools.values())}
    data = _json_bytes(manifest)
    (bundle / "MANIFEST.json").write_bytes(data)
    log("WROTE %s sha256=%s (%d pools, %d bytes)" % (bundle / "MANIFEST.json", hashlib.sha256(data).hexdigest(),
                                                    len(pools), manifest["total_bytes"]))
    for family, entries in sorted(by_family.items()):
        folder = fam.pool.family_pool_dir(family)
        written = {e["file"] for e in entries}
        try:
            existing, _ = fam.pool.read_manifest(folder)
            kept = [e for n, e in sorted(existing["pools"].items()) if n not in written]
            previous = existing.get("provenance", {})
        except FileNotFoundError:
            kept, previous = [], {}
        for e in kept:
            stale = folder / e["file"]
            log("kept verbatim: %s/%s%s" % (family, e["file"], "" if stale.is_file() else " (file in the bundle)"))
        for e in entries:
            stale = folder / e["file"]
            if stale.is_file() and _sha(stale) != e["sha256"]:
                raise SystemExit("in-repo %s differs from the new pin; remove the stale file first" % stale)
        prov = {"tool": TOOL_VERSION, "bundle_manifest_sha256": hashlib.sha256(data).hexdigest(),
                "ladder": fam_ladder(), "pool_files": "the flat pool bundle (G3c option B); none in the repo"}
        if kept:
            prov = dict(prov, kept_entries_from_previous_manifest=sorted(e["file"] for e in kept),
                        previous_provenance=previous)
        sha = fam.pool.write_manifest(folder, kept + sorted(entries, key=lambda x: x["file"]), provenance=prov)
        log("WROTE %s sha256=%s (%d kept verbatim, %d pinned from the bundle)"
            % (folder / fam.pool.MANIFEST_NAME, sha, len(kept), len(entries)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
