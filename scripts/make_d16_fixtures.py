"""Generate the D16/D20 not-admitted stamp fixtures from the REAL family code (CPU only).

CONTEXT_R4 D20 (e): the tests of scripts/d7_exposure.py, validate_cert_v3, analysis_input
and cert_v3 use REAL not-admitted stamps produced by each family's code on the merged tree,
not hand-written dictionaries. This tool runs, for each of the five families that publish
D16 not-admitted records, one short episode of the family's REAL journal env (the family's
own stubbed-Isaac-Lab test helpers, the same recipe as the family's D16 test) whose row is
a key E4 rejected, takes the finished record's D7 envelope (the family's not-admitted
stamp producer) and the env's D7 certificate-header block, and writes

    scripts/fixtures_d16/<family>.json

with provenance (the SHA-256 of every family source file the stamp comes from, the
helper modules, the E4 stamp module, F0 pairing and this generator).

    family           stamp producer (real code)                                   recipe (family test it mirrors)
    hazard_nav       tasks/hazard_nav/journal_hazard_nav.py not_admitted_stamp    tasks/hazard_nav/test_journal_d16.py
    path_hazard      tasks/path_hazard/path_hazard_journal.py NotAdmittedBridge.finish
                                                                                  tasks/path_hazard/tests_journal/test_journal_d16.py
    path_following   tasks/path_following/journal_path_following.py not_admitted_stamp
                                                                                  tasks/path_following/tests_journal/test_f3_d16_not_admitted.py
    docking          tasks/docking/journal_d7_exposure.py not_admitted_stamp      tasks/docking/test_journal_docking_env.py
    harbor_mission   tasks/harbor_mission/harbor_d7_pool.py HarbourD7Bridge.finish + NotAdmitted.stamp_assignment
                                                                                  tasks/harbor_mission/test_journal_harbour_env.py

Every family runs in its own child process (each family installs its own isaaclab
stand-ins), with CUDA hidden, one torch thread, bytecode writing off and the simulator
packages blocked, like the D1 CPU harness. Children run one at a time.

usage:
    python -B scripts/make_d16_fixtures.py --write              regenerate and write all five fixtures
    python -B scripts/make_d16_fixtures.py --check              regenerate and compare with the saved
                                                                 fixtures (stamp and D7 header), exit 1
                                                                 on any difference
    python -B scripts/make_d16_fixtures.py --family docking ... one family only
Set TEMP/TMP to a scratch folder first if temporary files must stay in a given place
(the path_following helper writes its test pool files to a temporary directory).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import types

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
FIXTURE_DIR = SCRIPTS / "fixtures_d16"
SCHEMA = "usvbench.t0.d16_fixture/1"
FAMILIES = ("hazard_nav", "path_hazard", "path_following", "docking", "harbor_mission")
SHARED_SOURCES = ("tasks/_shared/conditions/dynamic/stamp.py", "tasks/_shared/conditions/pairing.py",
                  "tasks/_shared/journal_env_mixin.py")
FAMILY = {
    "hazard_nav": {
        "producer": "tasks/hazard_nav/journal_hazard_nav.py:not_admitted_stamp",
        "recipe": "tasks/hazard_nav/test_journal_d16.py (probe pool fortress/BlueBoat 3 envs x episodes 0-1, "
                  "dynamic=D2, 2 control steps to the shortened deadline; first not-admitted row)",
        "sources": ("tasks/hazard_nav/journal_hazard_nav.py", "tasks/hazard_nav/hazard_nav_env.py",
                    "tasks/hazard_nav/journal_test_support.py", "tasks/hazard_nav/test_journal_d16.py"),
    },
    "path_hazard": {
        "producer": "tasks/path_hazard/path_hazard_journal.py:NotAdmittedBridge.finish",
        "recipe": "tasks/path_hazard/tests_journal/test_journal_d16.py (WAM-V test pool 2 envs x 3 episodes, "
                  "key (42, 0, 0) not admitted, dynamic=D2, contract timeout after 3 control steps; row 0)",
        "sources": ("tasks/path_hazard/path_hazard_journal.py", "tasks/path_hazard/path_hazard_env.py",
                    "tasks/path_hazard/tests_journal/ph_fixtures.py",
                    "tasks/path_hazard/tests_journal/test_journal_d16.py"),
    },
    "path_following": {
        "producer": "tasks/path_following/journal_path_following.py:not_admitted_stamp",
        "recipe": "tasks/path_following/tests_journal/test_f3_d16_not_admitted.py short_record (BlueBoat, "
                  "seed 51, key (51, 0, 0) not admitted, dynamic=D2, out of bounds at control step 3; row 0)",
        "sources": ("tasks/path_following/journal_path_following.py", "tasks/path_following/path_following_env.py",
                    "tasks/path_following/tests_journal/f3_fixtures.py",
                    "tasks/path_following/tests_journal/test_f3_d16_not_admitted.py"),
    },
    "docking": {
        "producer": "tasks/docking/journal_d7_exposure.py:not_admitted_stamp",
        "recipe": "tasks/docking/test_journal_docking_env.py NotAdmittedD16.run_full_episode (BlueBoat small "
                  "pool seed 42, row 1 = key (1, 0) not admitted, dynamic=D2, out of bounds after 3 steps)",
        "sources": ("tasks/docking/journal_d7_exposure.py", "tasks/docking/docking_env.py",
                    "tasks/docking/journal_d7_pool.py", "tasks/docking/level_table.py",
                    "tasks/docking/test_journal_docking_env.py"),
    },
    "harbor_mission": {
        "producer": "tasks/harbor_mission/harbor_d7_pool.py:HarbourD7Bridge.finish + NotAdmitted.stamp_assignment",
        "recipe": "tasks/harbor_mission/test_journal_harbour_env.py NotAdmittedD16._record (BlueBoat forced pool "
                  "reject (0, 1), (1, 0), dynamic=D2, 2 steps then the shortened deadline; row 1 = key (1, 0))",
        "sources": ("tasks/harbor_mission/harbor_d7_pool.py", "tasks/harbor_mission/harbor_journal.py",
                    "tasks/harbor_mission/harbor_mission_env.py", "tasks/harbor_mission/harbour_journal_fixtures.py",
                    "tasks/harbor_mission/test_journal_harbour_env.py"),
    },
}
CHILD_ENVIRONMENT = {"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8",
                     "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
MARK = "D16_FIXTURE_JSON:"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_lf(path):
    """SHA-256 of a text file with CRLF normalised to LF (CONTEXT_R4 D26 (d): line endings are not content)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


TEXT_DATA_SUFFIXES = (".json", ".sha256", ".txt", ".md")


def crlf_digest_map():
    """{raw SHA-256: LF SHA-256} of every CRLF text data file under tasks/ (empty on an LF tree).

    CONTEXT_R4 D26: the T-base v0 assembly converted hash-pinned data files (pool manifests,
    sidecars) to CRLF; the integration restores the families' LF bytes. A family header that
    stamps the digest of such a file (F3 pool_manifest_sha256) is recorded with the file's
    LF digest, so the fixtures never hold a digest of converted bytes and are identical on
    a CRLF and an LF checkout.
    """
    out = {}
    for path in (REPO / "tasks").rglob("*"):
        if path.is_file() and path.suffix in TEXT_DATA_SUFFIXES:
            raw = path.read_bytes()
            if b"\r\n" in raw and b"\0" not in raw:
                out[hashlib.sha256(raw).hexdigest()] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    return out


def lf_digests(value, mapping):
    if isinstance(value, dict):
        return {k: lf_digests(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [lf_digests(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


# --------------------------------------------------------------- child side --
class _NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"isaaclab", "isaacsim", "omni", "pxr"}:
            raise ImportError("make_d16_fixtures blocks simulator import: " + fullname)


def _guard(test_dir):
    """The D1 harness environment: no simulator, bare namespace task packages, one thread."""
    sys.dont_write_bytecode = True
    os.environ.update(CHILD_ENVIRONMENT)
    sys.meta_path.insert(0, _NoSimulator())
    sys.path[:0] = [str(test_dir), str(SCRIPTS), str(REPO)]
    for name, folder in [("tasks", REPO / "tasks")] + [
            ("tasks." + f.name, f) for f in (REPO / "tasks").iterdir() if f.is_dir() and f.name != "_shared"]:
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        module.__package__ = name
        sys.modules[name] = module
    import torch
    torch.set_num_threads(1)
    return torch


def _hazard_nav():
    torch = _guard(REPO / "tasks" / "hazard_nav")
    import journal_test_support as S
    S.install_isaaclab_stubs()
    fam = S.load_family()
    J, D = fam.journal, fam.dynamic
    task, vessel, num_envs = "fortress", "blueboat_v2", 3     # G7: ring / BlueBoat now admits every key
    pool = S.admitted_pool(fam, task, vessel, num_envs=num_envs, episodes=(0, 1))
    family_pool = J.family_pool(pool)
    rows = [i for i in range(num_envs) if not family_pool.is_admitted(D.EpisodeKey(S.SEED, i, 0, "d7_" + task))]
    cfg = S.journal_cfg(fam, task, vessel, tokens=("dynamic=D2",), num_envs=num_envs, pool=pool)
    env = S.make_env(fam, cfg)
    env.reset()
    env._journal_twin.deadline[:] = 2 * fam.plans.CONTROL_DT_S
    for _ in range(2):
        env.step(S.zero_action(env, (0.4, 0.05)))
    row = rows[0]
    record = env.journal_completed_records[row]
    return env, row, record, env.journal_certificate_header(treatment="dynamic"), {}


def _path_hazard():
    torch = _guard(REPO / "tasks" / "path_hazard" / "tests_journal")
    import ph_fixtures as F
    F.modules()
    F.install_stubs()
    vessel, row = "wamv_vrx_v2_opt1", 0
    env = F.make_env(F.journal_cfg(vessel, tokens=("dynamic=D2",), pool=F.test_pool(vessel, 2, 3)))
    env.reset()
    n = env.num_envs
    routes = torch.cat((torch.zeros((n, 1, 2), dtype=torch.float64), env._jph_waypoints), dim=1)
    env._jph_oracle.reset(torch.arange(n), routes, env._jph_gate_radius,
                          torch.full((n,), 3.0 / 60.0, dtype=torch.float64), routes[:, 0])
    for k in range(3):
        env.step(F.actions(k, n))
    return env, row, env.journal_episode_record(row), env.journal_certificate_header(treatment="dynamic"), {}


def _path_following():
    torch = _guard(REPO / "tasks" / "path_following" / "tests_journal")
    import f3_fixtures as F
    row, steps = 0, 3
    # G7: seed 51 (BlueBoat key (51, 0, 0), real E4 rejection); key (20, 0, 0) is admitted under the G7 rule
    cfg = F.journal_cfg("blueboat_v2", num_envs=1, episodes=2, tokens=("dynamic=D2",), seed=51)
    env = F.make_env(cfg)
    env.reset()
    if env._journal_d7_exposure[row] != F.jpf.EXPOSURE_NOT_ADMITTED:
        raise RuntimeError("path_following: key (51, 0, 0) is no longer a not-admitted key")
    for k in range(steps + 1):
        if k == steps:
            env.robot.data.root_link_pose_w[row, 0] += 5000.0
        env.step(F.seek_actions(env))
    if int(env.termination_reason[row]) != 4:
        raise RuntimeError("path_following: the row did not end out of bounds")
    return env, row, env.journal_episode_record(row), env.journal_certificate_header(), {}


def _docking():
    torch = _guard(REPO / "tasks" / "docking")
    import test_journal_docking_env as T
    row = 1
    env = T.make_env(tokens=("dynamic=D2",), num_envs=2)
    env.reset()
    if env._d7_bridge.exposure(row) != "not_admitted":
        raise RuntimeError("docking: row 1 is no longer a not-admitted key")
    steps = 0
    while env.journal_episode_results[row] is None:
        if steps == 3:
            T.force_out_of_bounds(env, [row])
        if steps > 50:
            raise RuntimeError("docking: the not-admitted row did not finish")
        env.step(torch.zeros(2, 2))
        steps += 1
    extra = {"docking_level_table": getattr(env, "docking_level_table", None)}
    return env, row, env.journal_episode_record(row), env.journal_certificate_header(), extra


def _harbor_mission():
    torch = _guard(REPO / "tasks" / "harbor_mission")
    import harbour_journal_fixtures as F
    row = 1
    env = F.make_env(tokens=("dynamic=D2",), reject=((0, 1), (1, 0)))
    env.reset()
    for _ in range(2):
        env.step(torch.tensor([[0.5, 0.1]] * env.num_envs))
    env._oracle.deadline_s[row] = float(env.episode_length_buf[row] + 1) * F.plans.CONTROL_DT_S
    env.step(torch.zeros(2, 2))
    if env.extras["termination_reason"].tolist()[row] != 3:
        raise RuntimeError("harbor_mission: the not-admitted row did not time out")
    return env, row, env.journal_episode_record(row), env.journal_certificate_header(), {}


RECIPES = {"hazard_nav": _hazard_nav, "path_hazard": _path_hazard, "path_following": _path_following,
           "docking": _docking, "harbor_mission": _harbor_mission}


RUNTIME_PATH = "<runtime path>/"


def normalise_paths(value):
    """Machine-specific absolute file paths (e.g. a helper's temporary pool file) -> "<runtime path>/name".

    The family headers stamp where a pool was loaded from; that location depends on the
    machine and the temporary directory, not on the family code, so it is replaced by its
    file name to keep the fixtures reproducible. Nothing else is changed.
    """
    if isinstance(value, dict):
        return {k: (RUNTIME_PATH + Path(v.replace("\\", "/")).name
                    if isinstance(v, str) and (k == "path" or k.endswith("_path")) and _absolute(v)
                    else normalise_paths(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [normalise_paths(v) for v in value]
    return value


def _absolute(text):
    return text.startswith(("/", "\\")) or (len(text) > 2 and text[1] == ":" and text[2] in ("\\", "/"))


def child(family):
    env, row, record, header, extra = RECIPES[family]()
    import cert_v3                      # the writer's own JSON conversion (header.dynamic_hazards_header)
    out = {
        "row": row,
        "stamp": record["dynamic_hazards"],
        "scenario_hashes": cert_v3.json_value(record.get("scenario_hashes")),
        "condition_runtime": cert_v3.json_value((record.get("conditions") or {}).get("runtime")),
        "dynamic_hazards_header": lf_digests(normalise_paths(cert_v3.json_value(header["dynamic_hazards_header"])),
                                             crlf_digest_map()),
        "pairing_inventory": cert_v3.json_value((header.get("pairing") or {}).get("primitives")),
        "extra": cert_v3.json_value(extra),
    }
    sys.stdout.write(MARK + json.dumps(out, sort_keys=True) + "\n")
    sys.stdout.flush()


# -------------------------------------------------------------- parent side --
def run_child(family, timeout=1800):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", family]
    environment = dict(os.environ, **CHILD_ENVIRONMENT)
    for name in [k for k in environment if k.startswith("USVBENCH_")]:
        environment.pop(name)          # families read pool locations from these; use the repo defaults
    result = subprocess.run(command, cwd=str(REPO), env=environment, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    lines = [line for line in result.stdout.splitlines() if line.startswith(MARK)]
    if result.returncode != 0 or len(lines) != 1:
        raise RuntimeError("fixture child %s failed (exit %s):\n%s\n%s"
                           % (family, result.returncode, result.stdout[-4000:], result.stderr[-4000:]))
    return json.loads(lines[0][len(MARK):])


def provenance(family):
    names = list(FAMILY[family]["sources"]) + list(SHARED_SOURCES) + ["scripts/make_d16_fixtures.py"]
    return {name: sha256_lf(REPO / name) for name in names}


def fixture_document(family, produced):
    return {
        "schema": SCHEMA,
        "family": family,
        "producer": FAMILY[family]["producer"],
        "recipe": FAMILY[family]["recipe"],
        "generator": "scripts/make_d16_fixtures.py",
        "regenerate": "python -B scripts/make_d16_fixtures.py --write",
        "source_sha256": provenance(family),
        "row": produced["row"],
        "stamp": produced["stamp"],
        "header_paths": "absolute file paths in dynamic_hazards_header replaced by '<runtime path>/<name>'",
        "header_file_digests": "digests of CRLF-converted text data files recorded as their LF digests (D26)",
        "source_digest_rule": "source_sha256 = SHA-256 of each file with CRLF normalised to LF (D26 d)",
        "scenario_hashes": produced["scenario_hashes"],
        "condition_runtime": produced["condition_runtime"],
        "dynamic_hazards_header": produced["dynamic_hazards_header"],
        "pairing_inventory": produced["pairing_inventory"],
        "extra": produced["extra"],
    }


def fixture_path(family):
    return FIXTURE_DIR / (family + ".json")


def load_fixture(family):
    return json.loads(fixture_path(family).read_text(encoding="ascii"))


def write_fixture(document):
    FIXTURE_DIR.mkdir(exist_ok=True)
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    path = fixture_path(document["family"])
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write(text)
    return path


COMPARED = ("stamp", "dynamic_hazards_header", "scenario_hashes", "row")


def check_family(family):
    """[] when a fresh run reproduces the saved fixture (stamp, D7 header, hashes), else the differences."""
    saved = load_fixture(family)
    fresh = run_child(family)
    problems = ["%s: %s differs from a fresh run of the family code" % (family, key)
                for key in COMPARED if saved.get(key) != fresh.get(key)]
    stale = sorted(name for name, digest in provenance(family).items()
                   if saved["source_sha256"].get(name) != digest)
    return problems, stale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--child", choices=FAMILIES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        child(args.child)
        return 0
    if args.write == args.check:
        parser.error("choose exactly one of --write / --check")
    failures = 0
    for family in args.family or FAMILIES:
        if args.write:
            path = write_fixture(fixture_document(family, run_child(family)))
            print("wrote %s (sha256 %s)" % (path.relative_to(REPO).as_posix(), sha256_file(path)), flush=True)
        else:
            problems, stale = check_family(family)
            for line in problems:
                print("DIFFERS " + line, flush=True)
            if stale:
                print("NOTE %s: provenance SHA-256 changed for %s (stamp still compared above)"
                      % (family, ", ".join(stale)), flush=True)
            print(("OK " if not problems else "FAIL ") + family, flush=True)
            failures += bool(problems)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
