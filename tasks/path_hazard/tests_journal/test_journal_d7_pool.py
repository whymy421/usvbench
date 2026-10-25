"""F2 D7 pool tests: pinned manifest, offline admission reproducibility, codec integrity, fail-closed env.

Run: python -B tasks/path_hazard/tests_journal/test_journal_d7_pool.py   (CPU only)

D16 semantics: per key the pool holds the admitted D0..D4 ladder OR an explicit not-admitted record
with E4's rejection reasons; every key keeps its E5 layout (no redraw, no attempt streams). The pool
files live in task F2's work/evidence/d7_pool (more than 8 MB, D19); the repository carries the
pinned manifest (tasks/path_hazard/journal_d7/d7_pool_manifest.json) with every file SHA-256, the E4
pool digests, the not-admitted list and the E5 geometry hash of every key. The evidence checks run
when that folder is next to the repository (the F2 work tree) and are reported as skipped otherwise.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import ph_fixtures as F  # noqa: E402

M = F.modules()
J = M.journal
EVIDENCE = F.REPO.parent / "evidence" / "d7_pool"
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def is_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class PinnedManifest(unittest.TestCase):
    def test_manifest_covers_the_evaluation_grid(self):
        manifest = J.pinned_manifest()
        self.assertEqual(manifest["format"], J.POOL_MANIFEST_FORMAT)
        self.assertEqual(manifest["task"], "path_hazard")
        self.assertEqual((manifest["tier"], manifest["eval_seed"], manifest["envs"], manifest["episodes"]),
                         (2, 42, 64, 2))
        self.assertEqual(manifest["pool_format"], J.POOL_FORMAT)
        grid = {(42, e, p) for e in range(64) for p in range(2)}
        for vessel in VESSELS:
            entry = manifest["vessels"][vessel]
            self.assertNotIn("attempts", entry)                                   # D16: no layout attempts
            self.assertTrue(is_hex(entry["pool_digest"]))
            self.assertEqual(entry["keys_offered"], 128)
            self.assertEqual(entry["keys_admitted"] + entry["keys_not_admitted"], 128)
            self.assertEqual(entry["per_dose_counts_columns"], ["offered", "admitted", "not_admitted"])
            self.assertEqual(entry["per_dose_counts"],
                             [[128, entry["keys_admitted"], entry["keys_not_admitted"]]] * 5)
            not_admitted = {tuple(r[:3]): r[3] for r in entry["not_admitted"]}
            self.assertEqual(len(not_admitted), entry["keys_not_admitted"])
            self.assertTrue(set(not_admitted) <= grid)
            for reasons in not_admitted.values():
                self.assertTrue(J.primary_reasons(reasons))                       # a real E4 reason per key
            self.assertEqual({tuple(r[:3]) for r in entry["geometry_hashes"]}, grid)
            self.assertTrue(all(is_hex(r[3]) for r in entry["geometry_hashes"]))
            self.assertEqual(sorted(entry["files"]), ["0", "1", "2", "3", "4"])
            self.assertTrue(all(is_hex(f["sha256"]) for f in entry["files"].values()))
            suite = M.dynamic.DynamicSuiteConfig(enabled=True, vessel_id=vessel)
            self.assertEqual(entry["suite_parameters"], J.suite_parameters(suite))

    def test_in_process_admission_reproduces_the_pinned_status_and_layouts(self):
        manifest = J.pinned_manifest()
        for vessel in VESSELS:
            pool = F.test_pool(vessel, 2, 3)
            entry = manifest["vessels"][vessel]
            pinned_na = {tuple(r[:3]): r[3] for r in entry["not_admitted"]}
            geometry = {tuple(r[:3]): r[3] for r in entry["geometry_hashes"]}
            for key, h in pool.geometry_hashes.items():
                if key in geometry:
                    self.assertEqual(h, geometry[key], (vessel, key))
                    self.assertEqual(key in pool.not_admitted, key in pinned_na, (vessel, key))
                    if key in pinned_na:     # same E4 reasons (joint admission is per key)
                        self.assertEqual(pool.not_admitted[key], pinned_na[key], (vessel, key))
            self.assertEqual(pool.keys_admitted + pool.keys_not_admitted, pool.keys_offered)

    @unittest.skipUnless(EVIDENCE.is_dir(), "F2 evidence folder not next to this repository copy")
    def test_evidence_files_match_the_pins(self):
        manifest = J.pinned_manifest()
        for vessel in VESSELS:
            entry = manifest["vessels"][vessel]
            na = sorted(tuple(r[:3]) for r in entry["not_admitted"])
            adm = sorted({(42, e, p) for e in range(64) for p in range(2)} - set(na))
            for dose, spec in entry["files"].items():
                raw = (EVIDENCE / spec["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), spec["sha256"])
                suite = M.dynamic.DynamicSuiteConfig(enabled=int(dose) > 0, vessel_id=vessel, dose=int(dose))
                pool = J.resolve_pool(str(EVIDENCE), vessel=vessel, tier=2, suite=suite)
                self.assertEqual(pool.pool_digest, entry["pool_digest"])
                self.assertEqual(pool.manifest_sha256, J.pinned_manifest_sha256())
                self.assertEqual(len(pool.assignments), entry["keys_admitted"])
                self.assertEqual(pool.assignment(J.episode_key(*adm[0]), int(dose)).candidate.request.encounter.dose,
                                 int(dose))
                if na:
                    record = pool.assignment(J.episode_key(*na[0]), int(dose))
                    self.assertIsInstance(record, J.NotAdmittedAssignment)

    @unittest.skipUnless(EVIDENCE.is_dir(), "F2 evidence folder not next to this repository copy")
    def test_pool_folder_runtime_variables(self):
        suite = M.dynamic.DynamicSuiteConfig(enabled=False, vessel_id="blueboat_v2", dose=0)
        old = os.environ.get(J.POOL_DIR_ENV)
        try:
            os.environ[J.POOL_DIR_ENV] = str(EVIDENCE)
            self.assertEqual(J.pool_folder(), EVIDENCE)
            pool = J.resolve_pool("", vessel="blueboat_v2", tier=2, suite=suite)
            self.assertEqual(pool.pool_digest, J.pinned_manifest()["vessels"]["blueboat_v2"]["pool_digest"])
            with tempfile.TemporaryDirectory() as folder:     # cfg.journal_d7_pool_path wins over the variable
                with self.assertRaises(FileNotFoundError):
                    J.resolve_pool("", vessel="blueboat_v2", tier=2, suite=suite, path_variable=folder)
        finally:
            if old is None:
                os.environ.pop(J.POOL_DIR_ENV, None)
            else:
                os.environ[J.POOL_DIR_ENV] = old
        self.assertEqual(J.pool_folder("", str(EVIDENCE)), EVIDENCE)
        self.assertEqual(J.resolve_pool("", vessel="blueboat_v2", tier=2, suite=suite,
                                        path_variable=str(EVIDENCE)).keys_offered, 128)


class Codec(unittest.TestCase):
    def test_round_trip_and_tamper_detection(self):
        pool = F.test_pool("blueboat_v2", 2, 3)
        self.assertTrue(pool.not_admitted and pool.keys_admitted)      # the fixture grid holds both kinds
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pool.json.gz"
            sha = J.save_pool_dose(pool, 2, path)
            self.assertEqual(sha, J.save_pool_dose(pool, 2, Path(folder) / "again.json.gz"))   # deterministic
            loaded = J.load_pool_dose(path, vessel="blueboat_v2", tier=2, dose=2, file_sha256=sha,
                                      pool_digest=pool.pool_digest)
            self.assertEqual(loaded.not_admitted, pool.not_admitted)
            self.assertEqual(loaded.geometry_hashes, pool.geometry_hashes)
            self.assertEqual(loaded.per_dose_counts, pool.per_dose_counts)
            for (key, dose), assignment in loaded.assignments.items():
                self.assertEqual(dose, 2)
                self.assertEqual(assignment.candidate_digest, pool.assignment(key, 2).candidate_digest)
            admitted = next(k for k in pool.geometry_hashes if k not in pool.not_admitted)
            with self.assertRaises(KeyError):
                loaded.assignment(J.episode_key(*admitted), 1)             # another dose is not loaded
            with self.assertRaises(ValueError):
                J.load_pool_dose(path, vessel="blueboat_v2", tier=2, dose=2, file_sha256="0" * 64)
            with self.assertRaises(ValueError):
                J.load_pool_dose(path, vessel="blueboat_v2", tier=2, dose=3)
            with self.assertRaises(ValueError):
                J.load_pool_dose(path, vessel="wamv_vrx_v2_opt1", tier=2, dose=2)
            doc = json.loads(gzip.decompress(path.read_bytes()))
            tampered = json.loads(json.dumps(doc))
            first = tampered["assignments"]["__dict__"][0][1]["fields"]["candidate"]["fields"]
            first["geometry_id"] = "0" * 64                                  # one field of one candidate
            bad = Path(folder) / "bad.json.gz"
            bad.write_bytes(gzip.compress(json.dumps(tampered).encode("ascii"), mtime=0))
            with self.assertRaises(ValueError):
                J.load_pool_dose(bad, vessel="blueboat_v2", tier=2, dose=2)
            tampered = json.loads(json.dumps(doc))                           # hide a not-admitted key
            tampered["not_admitted"] = tampered["not_admitted"][1:]
            bad.write_bytes(gzip.compress(json.dumps(tampered).encode("ascii"), mtime=0))
            with self.assertRaises(ValueError):
                J.load_pool_dose(bad, vessel="blueboat_v2", tier=2, dose=2)

    def test_pool_resolution_fails_closed(self):
        pool = F.test_pool("blueboat_v2", 2, 3)
        suite = M.dynamic.DynamicSuiteConfig(enabled=True, vessel_id="blueboat_v2", dose=2)
        self.assertIs(J.resolve_pool(pool, vessel="blueboat_v2", tier=2, suite=suite), pool)
        with self.assertRaises(ValueError):          # admitted under other D7 suite parameters
            J.resolve_pool(pool, vessel="blueboat_v2", tier=2,
                           suite=M.dynamic.DynamicSuiteConfig(enabled=True, vessel_id="blueboat_v2", dose=2,
                                                              intruder_speed_mps=0.6))
        with self.assertRaises(ValueError):
            J.resolve_pool(pool, vessel="blueboat_v2", tier=1, suite=suite)
        with self.assertRaises(ValueError):
            J.resolve_pool(object(), vessel="blueboat_v2", tier=2, suite=suite)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                J.resolve_pool(folder, vessel="blueboat_v2", tier=2, suite=suite)


class EnvFailsClosed(unittest.TestCase):
    def test_keys_outside_the_offered_grid_refuse_to_reset(self):
        pool = F.test_pool("blueboat_v2", 2, 3)
        cfg = F.journal_cfg("blueboat_v2", num_envs=3, pool=pool)
        env = F.make_env(cfg)
        with self.assertRaisesRegex(ValueError, "outside the offered D7 grid"):
            env.reset()

    def test_missing_pool_files_refuse_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = F.journal_cfg("blueboat_v2", pool=F.test_pool("blueboat_v2", 2, 3))
            cfg.journal_d7_pool = folder
            with self.assertRaises(FileNotFoundError):
                F.make_env(cfg)
            cfg = F.journal_cfg("blueboat_v2", pool=F.test_pool("blueboat_v2", 2, 3))
            cfg.journal_d7_pool, cfg.journal_d7_pool_path = "", folder
            with self.assertRaises(FileNotFoundError):
                F.make_env(cfg)

    def test_env_layouts_are_the_e5_ones_and_the_header_counts_not_admitted(self):
        pool = F.test_pool("wamv_vrx_v2_opt1", 2, 3)
        env = F.make_env(F.journal_cfg("wamv_vrx_v2_opt1", pool=pool))
        env.reset()
        for i in range(2):
            self.assertEqual(env._jph_layouts[i].geometry_hash, pool.geometry_hashes[(42, i, 0)])
            self.assertEqual(env._jph_exposure[i],
                             "admitted" if pool.is_admitted(42, i, 0) else "not_admitted")
        header = env.dynamic_hazard_header
        self.assertEqual(header["pool_digest"], pool.pool_digest)
        self.assertEqual(header["per_dose_counts_columns"], ["offered", "admitted", "not_admitted"])
        self.assertEqual(header["per_dose_counts"], [[6, pool.keys_admitted, pool.keys_not_admitted]] * 5)
        self.assertEqual(header["keys_not_admitted"], pool.keys_not_admitted)
        self.assertEqual(header["not_admitted_reason_counts"], pool.reason_counts())
        self.assertTrue(torch.all(env.obstacle_active[:, :12]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
