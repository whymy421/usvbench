"""CPU tests for tasks/_shared/journal_vehicle.py (E2 journal vehicle record).

Run: python -B test_journal_vehicle.py (from this folder) or through the D1-style
runner. No Isaac import; torch is needed only because the record carries the
hydro_v2 profile (hydro_profiles imports hydro_v2 lazily).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
REPO = HERE.parents[2]
ASSETS = REPO / "assets_v2"
sys.path.insert(0, str(SHARED))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SHARED / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


registry = _load("vehicles_v2")
journal = _load("journal_vehicle")
NAMES = ("blueboat_v2", "wamv_vrx_v2_opt1", "wamv_vrx_v2_opt2", "wamv_vrx_v2_opt1_bare", "wamv_vrx_v2_opt2_bare")


class JournalVehicleRecord(unittest.TestCase):
    def test_record_fields_come_from_the_registry(self):
        for name in NAMES:
            spec = registry.get_vehicle_v2(name)
            record = journal.resolve_journal_vehicle(name)
            self.assertEqual(record.name, name)
            self.assertEqual(record.registry_schema_version, 2)
            self.assertEqual((record.asset_relpath, record.asset_sha256), (spec.asset_relpath, spec.asset_sha256))
            self.assertEqual(record.bow_axis, "+x")
            self.assertEqual(record.mass_kg, spec.mass_kg)
            self.assertEqual(record.inertia_com_kg_m2, spec.inertia_com_kg_m2)   # full tensor, off-diagonals kept
            self.assertEqual(record.com_offset_m, (0.0, 0.0, 0.0))
            self.assertEqual(record.calm_com_height_above_water_m, spec.hydrostatics.com_height_above_water_m)
            self.assertEqual(dict(record.windage_kwargs), registry.to_c3_windage_kwargs(spec))
            self.assertEqual(record.windage_reference_height_m,
                             spec.windage.lateral_centroid_height_above_water_m)
            # Top speed of the profile's own laws equals the registry's SpeedReference.
            self.assertAlmostEqual(record.top_speed_model_m_s, spec.speeds.top_model_m_s, places=12)
            self.assertEqual(record.hydro_profile.name, name)
            self.assertEqual(record.profile_status, record.hydro_profile.status)
            self.assertIn("PENDING", record.profile_status)
            self.assertTrue(record.pending and all(isinstance(item, str) for item in record.pending))
        self.assertAlmostEqual(journal.resolve_journal_vehicle("wamv_vrx_v2_opt1").inertia_com_kg_m2[0][2],
                               3.594, delta=0.01)

    def test_no_task_contract_quantities_in_the_record(self):
        # Cruise/reference speed, turning radius, scoring footprint, validity bands
        # and deadlines belong to journal_cfg_v2.py (task E5), not here.
        names = set(journal.JournalVehicle.__dataclass_fields__)
        for forbidden in ("cruise", "reference_speed", "turning", "radius", "footprint", "polygon", "capsule",
                          "deadline", "validity", "band"):
            self.assertFalse(any(forbidden in field for field in names), forbidden)

    def test_options_reach_the_profile(self):
        default = journal.resolve_journal_vehicle("blueboat_v2")
        legacy = journal.resolve_journal_vehicle("blueboat_v2", rollpitch_damping="legacy_470",
                                                 froude_krylov_horizontal=False)
        self.assertEqual(default.profile_options["rollpitch_damping"], "zeta:0.1")
        self.assertEqual(legacy.hydro_profile.hydro.angular_linear_damping[:2], (470.0, 470.0))
        self.assertFalse(legacy.hydro_profile.hydro.froude_krylov_horizontal)
        # T25 switch (b), owner decision T1.35b: the WAM-V journal default is "zeta:0.1" (was "registry")
        self.assertEqual(journal.resolve_journal_vehicle("wamv_vrx_v2_opt1").profile_options["rollpitch_damping"],
                         "zeta:0.1")

    def test_legacy_names_rejected_with_reason(self):
        for name, fragment in (("blueboat", "-X"), ("wamv", "0.8"), ("rov", "BlueROV2")):
            with self.assertRaises(ValueError) as caught:
                journal.resolve_journal_vehicle(name)
            self.assertIn("not a journal", str(caught.exception))
            self.assertIn(fragment, str(caught.exception))
        with self.assertRaises(ValueError):
            journal.resolve_journal_vehicle("wamv_vrx_v2_opt3")
        with self.assertRaises(TypeError):
            journal.resolve_journal_vehicle(None)

    def test_immutability(self):
        record = journal.resolve_journal_vehicle("blueboat_v2")
        with self.assertRaises(FrozenInstanceError):
            record.mass_kg = 1.0
        with self.assertRaises(TypeError):
            record.windage_kwargs["frontal_area_m2"] = 1.0
        with self.assertRaises(TypeError):
            record.inertia_com_kg_m2[0][0] = 1.0

    def test_verify_asset_hash_fail_closed(self):
        for name in NAMES:
            record = journal.resolve_journal_vehicle(name)
            path = record.verify_asset(ASSETS)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record.asset_sha256)
        record = journal.resolve_journal_vehicle("wamv_vrx_v2_opt1")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                record.verify_asset(folder)
            copy = Path(folder) / record.asset_relpath
            shutil.copyfile(ASSETS / record.asset_relpath, copy)
            self.assertEqual(record.verify_asset(folder), copy)
            with open(copy, "ab") as handle:
                handle.write(b"\0")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                record.verify_asset(folder)
            # Module-level form: by record or by registry name; legacy names rejected.
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                journal.verify_asset("wamv_vrx_v2_opt1", folder)
            with self.assertRaises(FileNotFoundError):
                journal.verify_asset("blueboat_v2", folder)
        for name in NAMES:
            self.assertEqual(journal.verify_asset(name, ASSETS), ASSETS / registry.get_vehicle_v2(name).asset_relpath)
        self.assertEqual(journal.verify_asset(record, ASSETS), ASSETS / record.asset_relpath)
        with self.assertRaisesRegex(ValueError, "not a journal"):
            journal.verify_asset("blueboat", ASSETS)

    def test_manifest_and_registry_agree(self):
        import json
        manifest = json.loads((ASSETS / "MANIFEST.json").read_text(encoding="ascii"))
        for name in NAMES:
            spec = registry.get_vehicle_v2(name)
            self.assertEqual(manifest["artifacts"][spec.asset_relpath]["sha256"], spec.asset_sha256)
        self.assertEqual(hashlib.sha256((ASSETS / "LICENSE_VRX.txt").read_bytes()).hexdigest(),
                         manifest["artifacts"]["LICENSE_VRX.txt"]["sha256"])

    def test_module_is_ascii_and_torch_free_at_import(self):
        (SHARED / "journal_vehicle.py").read_bytes().decode("ascii")
        text = (SHARED / "journal_vehicle.py").read_text(encoding="ascii")
        header = text.split("def resolve_journal_vehicle", 1)[0]
        self.assertNotIn("import torch", header)
        self.assertTrue(math.isfinite(journal.resolve_journal_vehicle("blueboat_v2").top_speed_model_m_s))


if __name__ == "__main__":
    unittest.main(verbosity=2)
