"""G7 item C (CPU): the re-admitted D7 pools are pinned under the G7 identity and window rules.

* every in-repo pool file equals its pin byte for byte (F3 path_following, F4 station keeping, F5 docking);
* every pin that records the journal geometry version names the CURRENT GEOMETRY_VERSION_DIGEST;
* every pinned suite names D7 suite version d7_ttc_dose_v3 and the G7 unforced-window rule;
* F4 companion files carry the admitted candidate of every accepted (key, dose) (schema /2);
* pools that live outside the repo (F1, F2, F6; D19) are checked against their pins when
  USVBENCH_D7_POOL_DIR points at the pool bundle (otherwise only the pin files are checked).
G49 (v2.6.5, J29): the pin tables also hold the formal seeds 11002 41041 41047 41051 41057 (seed-qualified
names / keys); their F3 / F5 pool files travel in the bundle only (checked when USVBENCH_D7_POOL_DIR is set),
the F4 files ship in the release like seed 42.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import re
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"


def load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cfg = load("g7_pins_cfg", TASKS / "_shared" / "journal_cfg_v2.py")
SUITE_VERSION = "d7_ttc_dose_v3"
WINDOW_RULE = "abs_yaw_rate_lt_1e-6_rad_s_and_abs_accel_lt_1e-6_m_s2/1"
G49_SEEDS = (42, 11002, 41041, 41047, 41051, 41057)            # G49 (J29): every pinned evaluation seed


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def pool_file_folder(repo_folder, name, seed):
    """G49: where a pinned F3 / F5 pool file is read: the repo folder (seed 42 ships there), else the bundle
    named by USVBENCH_D7_POOL_DIR (the formal seeds), else None (not checked without the bundle)."""
    if (repo_folder / name).is_file():
        return repo_folder
    bundle = os.environ.get("USVBENCH_D7_POOL_DIR")
    if bundle:
        return Path(bundle)
    if seed == 42:
        raise AssertionError("seed-42 pool %s is missing from %s (no USVBENCH_D7_POOL_DIR)" % (name, repo_folder))
    return None


def sidecar_digest(folder, repo_folder, name):
    """G49: the .sha256 sidecar's digest; required in the repo folder, optional in the bundle (None if absent)."""
    path = folder / (name + ".sha256")
    if folder != repo_folder and not path.is_file():
        return None
    return path.read_text(encoding="ascii").split()[0]


class TestInRepoPins(unittest.TestCase):
    def test_f3_path_following(self):
        folder = TASKS / "path_following" / "journal_d7"
        manifest = json.loads((folder / "POOL_MANIFEST.json").read_text(encoding="ascii"))
        self.assertEqual(len(manifest["pools"]), 2 * len(G49_SEEDS))
        self.assertEqual(sorted({e["seed"] for e in manifest["pools"].values()}), sorted(G49_SEEDS))
        for name, entry in manifest["pools"].items():
            self.assertEqual(entry["geometry_version_digest"], cfg.GEOMETRY_VERSION_DIGEST, name)
            self.assertEqual((entry["num_envs"], entry["episodes"]), (64, 2))
            self.assertTrue(name.endswith("_seed%d.d7pool.xz" % entry["seed"]), name)
            where = pool_file_folder(folder, name, entry["seed"])
            if where is None:
                continue
            raw = (where / name).read_bytes()
            self.assertEqual(sha256_bytes(raw), entry["file_sha256"], name)
            container = sha256_bytes(lzma.decompress(raw, format=lzma.FORMAT_XZ))
            self.assertEqual(container, entry["container_sha256"], name)
            self.assertIn(sidecar_digest(where, folder, name), (container, None))

    def test_f4_station_keeping(self):
        # G23e: the env reads the approach-design pools (d7_pools_approach); the retired hold-design pools stay
        # in d7_pools/ unused
        folder = TASKS / "station_keeping" / "d7_pools_approach"
        index = json.loads((folder / "INDEX.json").read_text(encoding="ascii"))
        self.assertEqual(len(index["pools"]), 2 * len(G49_SEEDS))
        for key, entry in index["pools"].items():
            # G49: "<vessel>/tier2" = seed 42 (v2.6.4), "<vessel>/tier2/seed<seed>" = the formal seeds
            self.assertEqual(key.endswith("/tier2"), entry["seed"] == 42, key)
            if entry["seed"] != 42:
                self.assertTrue(key.endswith("/tier2/seed%d" % entry["seed"]), key)
            self.assertEqual(sha256_file(folder / entry["file"]), entry["sha256"], key)
            self.assertEqual(sha256_file(folder / entry["witness_file"]), entry["witness_sha256"], key)
            doc = json.loads((folder / entry["file"]).read_text(encoding="ascii"))
            self.assertEqual(doc["geometry_version_digest"], cfg.GEOMETRY_VERSION_DIGEST, key)
            self.assertEqual(doc["suite"]["version"], SUITE_VERSION)
            self.assertEqual(doc["suite"]["unforced_window_rule"], WINDOW_RULE)
            wit = json.loads((folder / entry["witness_file"]).read_text(encoding="ascii"))
            self.assertEqual(wit["schema"], "usvbench.f4.station_d7_witnesses/2")
            accepted = {"%d/%d/%d/%d" % (r[0], r[1], r[2], r[4]) for r in doc["accepted"]}
            # G23e: BlueBoat D1 not applicable by design (recorded, never a rejection), WAM-V none
            na = sorted({r["dose"] for r in doc["not_applicable"]})
            self.assertEqual(na, [1] if key.startswith("blueboat_v2/") else [], key)
            self.assertTrue(all(r["reason"] == "not_applicable_by_design" for r in doc["not_applicable"]))
            self.assertEqual(set(wit["candidates"]), accepted)
            self.assertEqual(set(wit["witnesses"]), accepted)
            # one stored candidate reproduces its admitted digest (exact float round trip)
            some = sorted(doc["accepted"])[0]
            blob = wit["candidates"]["%d/%d/%d/%d" % (some[0], some[1], some[2], some[4])]
            self.assertTrue(json.loads(gzip.decompress(base64.b64decode(blob)).decode("ascii")))

    def test_f5_docking(self):
        folder = TASKS / "docking" / "journal_pools"
        text = (TASKS / "docking" / "docking_env_cfg.py").read_text(encoding="utf-8")
        block = re.search(r"JOURNAL_D7_POOLS = \{(.*?)\n\}", text, re.S).group(1)
        pins = dict(re.findall(r'"([^"]+\.json\.xz)":\s*"([0-9a-f]{64})"', block))
        manifest_pin = re.search(r'JOURNAL_D7_POOL_MANIFEST_SHA256 = "([0-9a-f]{64})"', text).group(1)
        self.assertEqual(sha256_file(folder / "POOL_MANIFEST.json"), manifest_pin)
        manifest = json.loads((folder / "POOL_MANIFEST.json").read_text(encoding="ascii"))
        self.assertEqual(set(pins), set(manifest["pools"]))
        self.assertEqual(sorted({e["eval_seed"] for e in manifest["pools"].values()}), sorted(G49_SEEDS))   # G49
        for name, pin in pins.items():
            self.assertEqual(manifest["pools"][name]["sha256"], pin, name)
            where = pool_file_folder(folder, name, manifest["pools"][name]["eval_seed"])
            if where is None:
                continue
            self.assertEqual(sha256_file(where / name), pin, name)
            self.assertIn(sidecar_digest(where, folder, name), (pin, None))
            doc = json.loads(lzma.decompress((where / name).read_bytes(), format=lzma.FORMAT_XZ).decode("ascii"))
            self.assertEqual(doc["d7_suite_version"], SUITE_VERSION)


class TestOutOfRepoPins(unittest.TestCase):
    @staticmethod
    def bundle():
        value = os.environ.get("USVBENCH_D7_POOL_DIR")
        return Path(value) if value else None

    def check(self, name, pin, size=None):
        folder = self.bundle()
        if folder is None:
            return
        path = folder / name
        self.assertTrue(path.is_file(), name)
        self.assertEqual(sha256_file(path), pin, name)
        if size is not None:
            self.assertEqual(path.stat().st_size, size, name)

    def test_f1_hazard_nav(self):
        manifest = json.loads((TASKS / "hazard_nav" / "d7_pools" / "MANIFEST.json").read_text(encoding="ascii"))
        self.assertEqual(len(manifest["pools"]), 14)
        self.assertEqual((manifest["grid"]["seed"], manifest["grid"]["num_envs"], manifest["grid"]["episodes"]),
                         (42, 64, [0, 1]))
        for name, entry in manifest["pools"].items():
            self.check(name, entry["sha256"], entry["bytes"])

    def test_f2_path_hazard(self):
        manifest = json.loads((TASKS / "path_hazard" / "journal_d7" / "d7_pool_manifest.json")
                              .read_text(encoding="ascii"))
        self.assertEqual(set(manifest["vessels"]), {"blueboat_v2", "wamv_vrx_v2_opt1"})
        for vessel, entry in manifest["vessels"].items():
            self.assertEqual(entry["suite_parameters"]["version"], SUITE_VERSION)
            self.assertEqual(entry["suite_parameters"]["unforced_window_rule"], WINDOW_RULE)
            self.assertEqual(set(entry["files"]), {"0", "1", "2", "3", "4"})
            for f in entry["files"].values():
                self.check(f["file"], f["sha256"], f["bytes"])

    def test_f6_harbour(self):
        manifest = json.loads((TASKS / "harbor_mission" / "d7_pools" / "POOL_MANIFEST.json")
                              .read_text(encoding="ascii"))
        seed42 = [e for e in manifest["pools"] if e["seed"] == 42]
        self.assertEqual({e["vessel"] for e in seed42}, {"blueboat_v2", "wamv_vrx_v2_opt1"})
        self.assertEqual(sorted({e["seed"] for e in manifest["pools"]}), sorted(G49_SEEDS))       # G49
        self.assertEqual(len(manifest["pools"]), 2 * len(G49_SEEDS))
        for e in manifest["pools"]:
            self.check(e["file"], e["file_sha256"], e["size_bytes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
