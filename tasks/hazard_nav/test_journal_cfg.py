"""F1: the 14 HazardNav journal cfgs (7 tasks x 2 vessels) and the parent cfg edits (CPU only).

The REAL cfg modules run under stubbed isaaclab (journal_test_support); the
E5 template classes are constructed WITHOUT F0's family shims, so this proves
the parent edits of ANCHORS_E5 C6/C8: the v2 vessel is resolved in the parent
post-init (asset from assets_v2 with its registry SHA-256), hydro_v2, the
plan's 36-ray sensor, the layout_mode="journal" whitelist, the journal
min_obstacles floor (E5 capacity + the D7 slot), the geodesic flag rule, the
moving-basin floor (D7 N1) and the D10 renames. Each cfg also admits the
all-suites stack (F0 admit_journal_conditions) and runs one control step in
the real env on the stub DirectRLEnv.

Run: python -B tasks/hazard_nav/test_journal_cfg.py
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import journal_test_support as S  # noqa: E402

RESTORE = S.install_isaaclab_stubs()
FAM = S.load_family()
J = FAM.journal
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
ALL_ON = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "perception.bundle=D2",
          "dynamic=D2")


class JournalCfgs(unittest.TestCase):
    def test_14_journal_cfgs_construct_through_the_parent_edits(self):
        from importlib import import_module
        registry = import_module(S.ROOT + "._shared.vehicles_v2")
        for task in J.CARRIED_TASKS:
            for vessel in VESSELS:
                with self.subTest(task=task, vessel=vessel):
                    cfg = S.template_class(FAM, task, vessel)()
                    plan = cfg.journal_plan
                    params = plan.params
                    self.assertEqual(cfg.vehicle, vessel)
                    self.assertEqual(cfg.journal_task, task)
                    self.assertEqual(cfg.layout_mode, "journal")
                    self.assertEqual(cfg.underwater_physics_cfg.hydrodynamics_model, "hydro_v2")
                    self.assertEqual(cfg.underwater_physics_cfg.physics_version, 2)
                    self.assertEqual(cfg.max_obstacles, J.storage_width(task))
                    self.assertEqual(cfg.max_obstacles, max(1, FAM.plans.STATIC_CAPACITY[task]
                                                            + FAM.plans.D7_RESERVED_SLOTS[task]))
                    self.assertEqual(cfg.obstacle_slot_count, cfg.max_obstacles)
                    self.assertEqual(cfg.ray_count, 36)
                    self.assertEqual(cfg.ray_max_range_m, params["ray_sensor_max_range_m"])
                    self.assertEqual(cfg.goal_radius, params["goal_radius_m"])
                    self.assertEqual(cfg.observation_space, 42)
                    self.assertTrue(cfg.journal_observation_hooks_wired)
                    self.assertEqual(cfg.observation_contract, "journal_v3_ray36")
                    self.assertEqual(cfg.vessel_profile, vessel)
                    self.assertIsInstance(cfg.vessel_profile_status, str)
                    self.assertTrue(all(math.isnan(v) for v in (cfg.thrust_max_fwd, cfg.thrust_max_rev,
                                                                cfg.yaw_torque_max)))
                    spec = registry.get_vehicle_v2(vessel)
                    usd = Path(cfg.robot_cfg.spawn.usd_path)
                    self.assertEqual(usd.name, Path(spec.asset_relpath).name)
                    self.assertTrue(usd.is_file())
                    self.assertIs(cfg.robot_cfg.spawn.rigid_props.disable_gravity, False)
                    self.assertAlmostEqual(cfg.robot_cfg.init_state.pos[2],
                                           spec.hydrostatics.com_height_above_water_m)
                    self.assertEqual(cfg.episode_length_s, plan.max_deadline_s)
                    self.assertEqual(cfg.appear_count, 0)
                    self.assertFalse(cfg.use_official_blueboat_visual)
                    FAM.templates.journal_validate(cfg)
                    self.assertEqual(FAM.journal_config.check_journal_cfg(cfg), "hazard_nav")

    def test_every_cfg_admits_the_all_suites_stack(self):
        for task in J.CARRIED_TASKS:
            for vessel in VESSELS:
                with self.subTest(task=task, vessel=vessel):
                    cfg = S.template_class(FAM, task, vessel)()
                    cfg.sim.gravity = S.GRAVITY
                    resolved = FAM.journal_config.resolve_journal_conditions(presets=list(ALL_ON), seed=S.SEED)
                    report = FAM.journal_config.admit_journal_conditions(cfg, resolved)
                    self.assertTrue(report)

    def test_journal_validation_rejects_contract_changes(self):
        cfg = S.template_class(FAM, "scatter", "blueboat_v2")()
        for field, value in (("obs_v2", True), ("emit_superset_obs", True), ("obs_feasibility", True),
                             ("ray_max_range_m", 30.5), ("ray_count", 24), ("layout_mode", "ring"),
                             ("appear_count", 1)):
            with self.subTest(field=field):
                saved = getattr(cfg, field)
                setattr(cfg, field, value)
                with self.assertRaises((ValueError, TypeError)):
                    cfg.validate_after_overrides()
                setattr(cfg, field, saved)
        cfg.validate_after_overrides()
        cfg.max_obstacles = J.storage_width("scatter") - 1
        with self.assertRaises(ValueError):
            cfg.validate_after_overrides()

    def test_journal_allows_geodesic_progress(self):
        cfg = S.template_class(FAM, "fortress", "blueboat_v2")()
        cfg.reward_progress_geodesic = True
        cfg.validate_after_overrides()
        cfg.nav_targets_waypoint = True
        with self.assertRaises(ValueError):
            cfg.validate_after_overrides()

    def test_legacy_whitelist_and_moving_basin_floor(self):
        legacy = FAM.cfg.HazardNavV3EnvCfg()
        self.assertFalse(hasattr(legacy, "journal_contract_version"))
        legacy.layout_mode = "journal"
        with self.assertRaises(ValueError):
            legacy.validate_after_overrides()
        basin = FAM.cfg.HazardMovingBasinEnvCfg()
        self.assertEqual(basin.max_obstacles, 136)
        basin.max_obstacles = 127
        with self.assertRaises(ValueError):
            basin.validate_after_overrides()
        basin.max_obstacles = 128
        basin.validate_after_overrides()
        legacy_geo = FAM.cfg.HazardNavV3EnvCfg()
        legacy_geo.reward_progress_geodesic = True
        with self.assertRaises(ValueError):
            legacy_geo.validate_after_overrides()

    def test_d10_renames(self):
        module = FAM.cfg
        self.assertFalse(hasattr(module, "HazardNavV3JournalEnvCfg"))
        self.assertFalse(hasattr(module, "HazardForcedCrossingJournalEnvCfg"))
        for name, parent in (("HazardNavV3JBaseCalmEnvCfg", "HazardNavV3EnvCfg"),
                             ("HazardForcedCrossingJBaseCalmEnvCfg", "HazardForcedCrossingEnvCfg")):
            cls = getattr(module, name)
            self.assertIn(getattr(module, parent), cls.__mro__)
            cfg = cls()
            self.assertEqual(cfg.underwater_physics_cfg.physics_version, 2)
            self.assertTrue(cfg.journal_mode)
            self.assertEqual(cfg.vehicle, "blueboat")
            self.assertFalse(hasattr(cfg, "journal_contract_version"))
            for init in (S.REPO / "tasks" / "journal" / "__init__.py", S.REPO / "tasks" / "hazard_nav" / "__init__.py"):
                self.assertNotIn(name + '"', init.read_text(encoding="utf-8"))


class EveryTaskRuns(unittest.TestCase):
    def pool_for_first_keys(self, task, vessel):
        """Pool over key (seed, 0, 0) and (seed, 0, 1); D16: the env runs key episode 0 whatever its status."""
        return S.admitted_pool(FAM, task, vessel, num_envs=1, episodes=(0, 1))

    def test_reset_and_one_step_every_task_both_vessels(self):
        for task in J.CARRIED_TASKS:
            for vessel in VESSELS:
                with self.subTest(task=task, vessel=vessel):
                    pool = self.pool_for_first_keys(task, vessel)
                    cfg = S.journal_cfg(FAM, task, vessel, tokens=("dynamic=D2",), num_envs=1, pool=pool)
                    env = S.make_env(FAM, cfg)
                    obs, _ = env.reset()
                    self.assertEqual(obs["policy"].shape, (1, 42))
                    sc = env._journal_scenarios[0]
                    self.assertEqual((sc.env, sc.episode), (0, 0))              # D16: no schedule, no skip
                    admitted = (J.d7_key(sc), 2) in pool.assignments
                    self.assertEqual(env._d7_bridge.is_not_admitted(0), not admitted)
                    o, r, te, tr, _ = env.step(S.zero_action(env, (0.5, 0.0)))
                    self.assertTrue(torch.isfinite(o["policy"]).all())
                    self.assertEqual(int(env.obstacle_count[0]), sc.obstacle_count)
                    self.assertTrue(torch.equal(env._journal_bounds[0],
                                                torch.tensor(sc.bounds, dtype=torch.float64)))

    def test_geodesic_field_uses_the_vessel_footprint(self):
        pool = self.pool_for_first_keys("fortress", "wamv_vrx_v2_opt1")
        cfg = S.journal_cfg(FAM, "fortress", "wamv_vrx_v2_opt1", num_envs=1, pool=pool)
        cfg.reward_progress_geodesic = True
        cfg.validate_after_overrides()
        env = S.make_env(FAM, cfg)
        env.reset()
        sc = env._journal_scenarios[0]
        self.assertAlmostEqual(float(env._geodesic_field_cell[0]), sc.layout.params["grid_cell_m"], places=5)
        straight = math.dist(sc.start_xy, sc.goal_xy)
        self.assertGreater(float(env.d0_per_env[0]), straight)
        env.step(S.zero_action(env, (0.4, 0.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
