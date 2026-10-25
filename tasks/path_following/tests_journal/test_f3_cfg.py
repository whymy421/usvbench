"""F3: the journal path-following cfgs (E5 templates + the F3 parent edits), CPU only.

The REAL cfg modules run under isaaclab stubs WITHOUT the shims F0's
harness applies (scripts/tests_journal_glue/glue_fixtures.FAMILY_OBLIGATIONS):
the v2 vessel resolves through journal_vehicle (ANCHORS_E5 C6), the asset
hash is verified (fail closed), hydro_v2 is selected, the vessel profile is
declared for E1's certificate, the plan's 36-ray sensor and the journal
observation contract are declared and the hooks flag is set.
Run: python -B test_f3_cfg.py
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f3_fixtures as F  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
ALL_SUITES = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "dynamic=D2")


def template(vessel):
    _, _, templates = F.env_modules()
    return {"blueboat_v2": templates.PathFollowingBlueBoatJournalEnvCfg,
            "wamv_vrx_v2_opt1": templates.PathFollowingWAMVJournalEnvCfg}[vessel]()


class JournalCfgTests(unittest.TestCase):
    def test_both_vessels_resolve_through_journal_vehicle(self):
        journal_vehicle = F.load("_shared.journal_vehicle")
        observation_groups = F.observation_groups
        for vessel in VESSELS:
            cfg = template(vessel)
            plan = cfg.journal_plan
            params = plan.params
            vehicle = journal_vehicle.resolve_journal_vehicle(vessel)
            physics = cfg.underwater_physics_cfg
            self.assertEqual(physics.hydrodynamics_model, "hydro_v2")
            self.assertEqual(physics.physics_version, 2)
            self.assertEqual(physics.vessel_profile, vessel)
            self.assertEqual(physics.vessel_profile_status, vehicle.profile_status)
            self.assertFalse(physics.enable_current)
            self.assertEqual(cfg.ray_count, 36)
            self.assertEqual(cfg.ray_max_range_m, params["ray_sensor_max_range_m"])
            self.assertAlmostEqual(cfg.ray_max_range_m, 30.0 * params["length_scale"], places=9)
            self.assertEqual(cfg.observation_space, 43)
            self.assertEqual(cfg.observation_contract, "journal_nav7_ray36")
            self.assertIs(getattr(cfg, observation_groups.HOOKS_FLAG), True)
            usd = Path(cfg.robot_cfg.spawn.usd_path)
            self.assertEqual(usd.name, Path(vehicle.asset_relpath).name)
            self.assertEqual(usd.parent.name, journal_vehicle.ASSETS_DIRNAME)
            self.assertIs(cfg.robot_cfg.spawn.rigid_props.disable_gravity, False)
            self.assertFalse(hasattr(cfg.robot_cfg.spawn, "mass_props"))
            self.assertAlmostEqual(cfg.robot_cfg.init_state.pos[2], vehicle.calm_com_height_above_water_m)
            self.assertEqual(cfg.episode_length_s, plan.max_deadline_s)
            self.assertEqual(cfg.decimation, 2)
            self.assertEqual(cfg.sim.dt, 1.0 / 120.0)
            self.assertTrue(all(math.isnan(v) for v in (cfg.thrust_max_fwd, cfg.thrust_max_rev, cfg.yaw_torque_max)))
            self.assertEqual(F.journal_config.check_journal_cfg(cfg), "path_following")
            contract = observation_groups.effective_contract(
                observation_groups.lookup("path_following", cfg.observation_contract), cfg)
            self.assertTrue(contract.hooks_wired and contract.journal)
            self.assertEqual(contract.groups, ("pos", "ray"))

    def test_all_suites_admit_on_both_cfgs(self):
        for vessel in VESSELS:
            cfg = template(vessel)
            resolved = F.journal_config.resolve_journal_conditions(presets=list(ALL_SUITES), seed=F.SEED)
            report = F.journal_config.admit_journal_conditions(cfg, resolved)
            self.assertTrue(report)

    def test_asset_hash_is_verified_fail_closed(self):
        journal_vehicle = F.load("_shared.journal_vehicle")
        cfg_module, _, _ = F.env_modules()
        saved = journal_vehicle.ASSETS_DIRNAME
        tmp = Path(tempfile.mkdtemp(prefix="f3_assets_"))
        try:
            journal_vehicle.ASSETS_DIRNAME = str(tmp)           # absolute: Path / absolute = absolute
            with self.assertRaises(FileNotFoundError):
                template("blueboat_v2")
            source = F.REPO / "assets_v2" / "blueboat_v2.usda"
            corrupted = tmp / "blueboat_v2.usda"
            shutil.copyfile(source, corrupted)
            with open(corrupted, "ab") as handle:
                handle.write(b"\n# tampered\n")
            with self.assertRaises(ValueError):
                template("blueboat_v2")
        finally:
            journal_vehicle.ASSETS_DIRNAME = saved
            shutil.rmtree(tmp, ignore_errors=True)
        del cfg_module

    def test_parent_branch_requires_the_matching_plan_and_rejects_legacy_names(self):
        cfg_module, _, _ = F.env_modules()
        plan = F.plans.plan_class("path_following", "blueboat_v2")(2)
        with self.assertRaises(ValueError):
            cfg_module._journal_parent_post_init(SimpleNamespace(journal_plan=plan, vehicle="wamv_vrx_v2_opt1"))
        with self.assertRaises(ValueError):
            cfg_module._journal_parent_post_init(SimpleNamespace(journal_plan=None, vehicle="blueboat_v2"))
        journal_vehicle = F.load("_shared.journal_vehicle")
        for legacy in ("blueboat", "wamv", "rov"):
            with self.assertRaises(ValueError):
                journal_vehicle.resolve_journal_vehicle(legacy)

    def test_legacy_ids_keep_the_legacy_physics(self):
        cfg_module, _, _ = F.env_modules()
        legacy = cfg_module.PathFollowingBlueBoatEnvCfg()
        self.assertFalse(hasattr(legacy.underwater_physics_cfg, "hydrodynamics_model"))
        self.assertEqual(legacy.observation_space, 7)
        self.assertFalse(hasattr(legacy, "journal_contract_version"))
        self.assertFalse(hasattr(legacy, "ray_count"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
