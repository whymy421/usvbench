"""G3c / G20b: the appearing pools reach a deployed release through the runtime pool directory (CPU only).

The release code tree (scripts/code_tree.py is_included: tasks/ and scripts/ files ending in .cfg .json .py
.toml .yaml .yml) does not carry tasks/*/appearing_pools/*.json.xz, so the pools travel in the flat pool
bundle and the loader reads them from the runtime pool directory (cfg.journal_d7_pool_dir, else
USVBENCH_D7_POOL_DIR), checking every file against the in-repo MANIFEST.json pin (controller option B).
G20b: the production pools (DIAG42, every task) are bundle-only, so these mechanics are exercised on small
pools admitted on the REAL layouts with the active ladder and pinned in a temporary family folder
(appearing_pin_fixture): runtime directory present (file read from it, source stamped), absent (family-folder
file), a wrong file and a missing file in the runtime directory (fail closed, no fallback), cfg field over
environment variable, the env header stamp, and an extracted-release style tree (code-tree members only, no
.xz) whose manifests pin the bundle files. The in-repo manifests themselves pin a bundle-only file for every
task x vessel (no fallback to a repo copy).
Run: python -B test_appearing_pool_dir.py
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_fixtures as F  # noqa: E402

POOL = F.pool_module
CASES = (("hazard_nav", "scatter"), ("path_following", "path_following"), ("docking", "docking"))
_spec = importlib.util.spec_from_file_location("appearing_pin_fixture",
                                               Path(__file__).resolve().parent / "appearing_pin_fixture.py")
PIN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PIN)
FOLDER = None
UNPIN = None


def setUpModule():
    global FOLDER, UNPIN
    pools = [PIN.build(F.admission, F.families, task, vessel, 2, 1, seed=F.SEED)
             for _, task in CASES for vessel in (F.BB, F.WAMV)]
    FOLDER, UNPIN = PIN.install(POOL, F.families, pools)


def tearDownModule():
    if UNPIN is not None:
        UNPIN()


def bundle_with(folder, names=None, corrupt=None, source=None):
    """A flat bundle folder holding byte copies of the pinned pools (optionally one corrupted)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for family, task in CASES:
        for vessel in (F.BB, F.WAMV):
            name = POOL.pool_file_name(task, vessel, F.SEED)
            if names is not None and name not in names:
                continue
            data = ((source or POOL.family_pool_dir)(family) / name).read_bytes()
            if corrupt == name:
                data = data[:-1] + bytes([data[-1] ^ 0x01])
            (folder / name).write_bytes(data)
    return folder


class RuntimePoolDirectory(unittest.TestCase):
    def test_present_absent_wrong_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = bundle_with(Path(tmp) / "bundle")
            for family, task in CASES:
                for vessel in (F.BB, F.WAMV):
                    repo_pool, repo_record = POOL.load_pinned_pool(family, task, vessel, F.SEED, F.TIER)
                    self.assertEqual(repo_record["source"], "repo")
                    pool, record = POOL.load_pinned_pool(family, task, vessel, F.SEED, F.TIER, runtime_dir=bundle,
                                                         runtime_source="test bundle")
                    self.assertEqual(record["source"], "runtime_pool_dir")
                    self.assertEqual(Path(record["path"]).parent, bundle)
                    self.assertEqual(pool.pool_digest, repo_pool.pool_digest)
                    self.assertEqual(record["sha256"], repo_record["sha256"])
            wrong = POOL.pool_file_name("scatter", F.BB, F.SEED)
            bad = bundle_with(Path(tmp) / "bad", corrupt=wrong)
            with self.assertRaises(ValueError):                         # same size, wrong bytes: SHA-256 pin
                POOL.load_pinned_pool("hazard_nav", "scatter", F.BB, F.SEED, F.TIER, runtime_dir=bad)
            (bad / wrong).write_bytes((bad / wrong).read_bytes() + b"x")
            with self.assertRaises(ValueError):                         # wrong size
                POOL.load_pinned_pool("hazard_nav", "scatter", F.BB, F.SEED, F.TIER, runtime_dir=bad)
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):                  # missing: never a fallback to the repo
                POOL.load_pinned_pool("path_following", "path_following", F.WAMV, F.SEED, F.TIER, runtime_dir=empty)

    def test_runtime_directory_resolution(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(POOL.POOL_DIR_ENV, None)
            self.assertEqual(POOL.runtime_pool_dir(SimpleNamespace()), (None, None))
            self.assertEqual(POOL.runtime_pool_dir(SimpleNamespace(journal_d7_pool_dir="")), (None, None))
            os.environ[POOL.POOL_DIR_ENV] = "C:/bundle_env"
            self.assertEqual(POOL.runtime_pool_dir(SimpleNamespace()), (Path("C:/bundle_env"), "env USVBENCH_D7_POOL_DIR"))
            self.assertEqual(POOL.runtime_pool_dir(SimpleNamespace(journal_d7_pool_dir="C:/bundle_cfg")),
                             (Path("C:/bundle_cfg"), "cfg.journal_d7_pool_dir"))
        self.assertEqual(POOL.runtime_pool_dir(None, environ={}), (None, None))

    def test_in_repo_pins_point_at_bundle_only_files(self):
        """G20b: the real in-repo manifests pin the DIAG42 pools of every task x vessel; no copy is in the repo,
        so the repo alone cannot load them (fail closed with the bundle hint, never a fallback)."""
        tasks_root = F.REPO / "tasks"                # the REAL family folders (not the patched test folder)
        for task in F.config.PLACEMENT_TASKS:
            family = F.families.FAMILY_OF_TASK[task]
            folder = tasks_root / family / POOL.POOL_DIRNAME
            for vessel in (F.BB, F.WAMV):
                name = POOL.pool_file_name(task, vessel, F.SEED)
                with self.subTest(pool=name):
                    manifest, _ = POOL.read_manifest(folder)
                    self.assertIn(name, manifest["pools"])
                    self.assertFalse((folder / name).exists())
                    with self.assertRaisesRegex(FileNotFoundError, "no runtime pool directory is set"):
                        POOL.load_pinned_pool(family, task, vessel, F.SEED, F.TIER, pool_dir=folder)

    def test_extracted_release_tree_plus_bundle(self):
        """A release tree (code-tree members only, so no .xz pools) finds and verifies the pools in the bundle."""
        spec = importlib.util.spec_from_file_location("_g3c_code_tree", F.REPO / "scripts" / "code_tree.py")
        code_tree = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(code_tree)
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp) / "release"
            copied = 0
            for top in ("tasks", "scripts"):
                for path in (F.REPO / top).rglob("*"):
                    rel = path.relative_to(F.REPO).as_posix()
                    if path.is_file() and code_tree.is_included(rel):
                        target = release / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                        copied += 1
            self.assertGreater(copied, 300)
            self.assertEqual(list(release.rglob("*.xz")), [])                       # the gap G2 found
            for family, _ in CASES:
                manifest = release / "tasks" / family / POOL.POOL_DIRNAME / POOL.MANIFEST_NAME
                self.assertTrue(manifest.is_file())
                # the release pins the test pools (the real manifests pin the DIAG42 bundle the same way)
                shutil.copyfile(FOLDER / family / POOL.POOL_DIRNAME / POOL.MANIFEST_NAME, manifest)
            bundle = bundle_with(Path(tmp) / "bundle")
            name = "_g3c_release"
            root = ModuleType(name)
            root.__path__ = [str(release / "tasks")]
            sys.modules[name] = root
            for folder in (release / "tasks").iterdir():
                if folder.is_dir():
                    sub = ModuleType(name + "." + folder.name)
                    sub.__path__ = [str(folder)]
                    sys.modules[name + "." + folder.name] = sub
            released = importlib.import_module(name + "._shared.conditions.appearing.pool")
            self.assertTrue(str(released.family_pool_dir("hazard_nav")).startswith(str(release)))
            for family, task in CASES:
                for vessel in (F.BB, F.WAMV):
                    with self.assertRaises(FileNotFoundError):                        # release alone cannot load
                        released.load_pinned_pool(family, task, vessel, F.SEED, F.TIER)
                    pool, record = released.load_pinned_pool(family, task, vessel, F.SEED, F.TIER, runtime_dir=bundle,
                                                             runtime_source="env USVBENCH_D7_POOL_DIR")
                    self.assertEqual(record["source"], "runtime_pool_dir")
                    self.assertEqual(pool.pool_digest, POOL.load_pinned_pool(family, task, vessel, F.SEED,
                                                                             F.TIER)[0].pool_digest)


class EnvHeaderStamp(unittest.TestCase):
    def test_env_reads_the_bundle_and_stamps_the_source(self):
        sys.path.insert(0, str(F.REPO / "tasks" / "hazard_nav"))
        import journal_test_support as S
        restore = S.install_isaaclab_stubs()
        unpin = None
        try:
            fam = S.load_family()
            spool = importlib.import_module(S.ROOT + "._shared.conditions.appearing.pool")
            sadm = importlib.import_module(S.ROOT + "._shared.conditions.appearing.admission")
            sfams = importlib.import_module(S.ROOT + "._shared.conditions.appearing.families")
            folder, unpin = PIN.install(spool, sfams, [PIN.build(sadm, sfams, "scatter", F.BB, 2, 2, seed=F.SEED)])
            d7 = S.admitted_pool(fam, "scatter", F.BB, num_envs=2, episodes=(0, 1))
            with tempfile.TemporaryDirectory() as tmp:
                bundle = bundle_with(Path(tmp) / "bundle", names={POOL.pool_file_name("scatter", F.BB, F.SEED)},
                                     source=spool.family_pool_dir)
                cfg = S.journal_cfg(fam, "scatter", F.BB, tokens=("appearing=D2",), num_envs=2, pool=d7)
                cfg.journal_d7_pool_dir = str(bundle)          # D7 comes from the in-memory pool; appearing from here
                env = S.make_env(fam, cfg)
                header = env.journal_certificate_header()["appearing_hazards_header"]
                self.assertEqual(header["pool_source"], "runtime_pool_dir")
                self.assertEqual(header["pool_source_detail"], "cfg.journal_d7_pool_dir")
                self.assertTrue(header["pool_pinned"])
                cfg2 = S.journal_cfg(fam, "scatter", F.BB, tokens=("appearing=D2",), num_envs=2, pool=d7)
                cfg2.journal_d7_pool_dir = str(Path(tmp) / "nowhere")
                with self.assertRaises(FileNotFoundError):
                    S.make_env(fam, cfg2)
            cfg3 = S.journal_cfg(fam, "scatter", F.BB, tokens=("appearing=D2",), num_envs=2, pool=d7)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(POOL.POOL_DIR_ENV, None)
                env = S.make_env(fam, cfg3)
            self.assertEqual(env.journal_certificate_header()["appearing_hazards_header"]["pool_source"], "repo")
        finally:
            if unpin is not None:
                unpin()
            restore()


if __name__ == "__main__":
    unittest.main(verbosity=2)
