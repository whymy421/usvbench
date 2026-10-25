"""Frozen plan, vessel-record and runtime-template checks (CPU only)."""

import ast
import dataclasses
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# E5: the registry is part of the tree (tasks/_shared/vehicles_v2.py +
# vehicles_v2_data.py, D5's files borrowed unmodified); loaded independently
# of journal_cfg_v2 under test-private module names.
REGISTRY_DIR = ROOT / "tasks/_shared"


def load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_registry():
    data = load("e5_test_cfg_registry_data", REGISTRY_DIR / "vehicles_v2_data.py")
    saved = sys.modules.get("vehicles_v2_data")
    sys.modules["vehicles_v2_data"] = data
    try:
        return load("e5_test_cfg_registry", REGISTRY_DIR / "vehicles_v2.py")
    finally:
        if saved is None:
            sys.modules.pop("vehicles_v2_data", None)
        else:
            sys.modules["vehicles_v2_data"] = saved


cfg = load("d4_test_cfg", ROOT / "tasks/_shared/journal_cfg_v2.py")
reg = load_registry()


class PlanTests(unittest.TestCase):
    def test_twenty_four_frozen_plans(self):
        self.assertEqual(len(cfg.JOURNAL_PLAN_CLASSES), 24)
        self.assertEqual(set(cfg.JOURNAL_PLAN_CLASSES), {(t, v) for t in cfg.TASKS for v in cfg.VESSELS})
        for (task, vessel), cls in cfg.JOURNAL_PLAN_CLASSES.items():
            for tier in range(4):
                plan = cls(tier)
                self.assertEqual((plan.task, plan.vessel, plan.physics_version, plan.contract_version),
                                 (task, vessel, 2, 2))
                self.assertGreater(plan.max_deadline_s, 0)
                self.assertAlmostEqual(plan.max_deadline_s * 60.0, round(plan.max_deadline_s * 60.0), places=6)
        plan = cfg.JOURNAL_PLAN_CLASSES[("reach", "blueboat_v2")]()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.geometry_tier = 3
        for bad in (-1, 4, 2.0, True):
            with self.assertRaises(ValueError):
                cfg.JOURNAL_PLAN_CLASSES[("reach", "blueboat_v2")](bad)
        with self.assertRaises(TypeError):
            cfg.JOURNAL_PLAN_CLASSES[("reach", "blueboat_v2")](dose=3)
        with self.assertRaises(ValueError):
            plan.assert_vehicle_override("wamv")
        self.assertNotIn("wamv", cfg.VESSELS)
        self.assertNotIn("wamv_vrx_nominal", cfg.VESSELS)

    def test_deadline_rule(self):
        self.assertEqual(cfg.deadline_steps(30.0, 1.0, 0.0), 3600)
        self.assertEqual(cfg.deadline_steps(10.0, 1.0, 80.0), 6000)
        self.assertEqual(cfg.deadline_steps(3.001, 1.0, 0.0), 361)          # rounded up once
        for bad in ((0, 1, 0), (1, 0, 0), (1, 1, -1), (float("nan"), 1, 0), (True, 1, 0)):
            with self.assertRaises(ValueError):
                cfg.deadline_steps(*bad)
        # E5: WAM-V key = registry default profile; cruise from the registry
        # (1.0 m/s, CONTEXT_R3 decision 14) instead of D4's pinned 2.0 m/s.
        plan = cfg.JOURNAL_PLAN_CLASSES[("station_keeping", cfg.WAMV.key)](2)
        self.assertAlmostEqual(plan.scenario_deadline_s(44.7791504231185),
                               2 * 44.7791504231185 / cfg.WAMV.cruise_speed_m_s + 80.0, delta=1 / 60)
        with self.assertRaises(ValueError):
            plan.scenario_deadline_s(plan.route_budget_m * 1.01)

    def test_every_task_geometry_resolves_positive_and_scaled(self):
        for task in cfg.TASKS:
            for tier in range(4):
                b = cfg.resolve_task_geometry(task, tier, "blueboat_v2")
                w = cfg.resolve_task_geometry(task, tier, cfg.WAMV.key)
                s = w["length_scale"]
                self.assertAlmostEqual(b["length_scale"], 1.0, places=12)
                for key, value in b.items():
                    if isinstance(value, float) and key.endswith("_m") and key not in (
                            "physical_surface_gap_m", "hull_centre_width_m", "planned_centre_width_m",
                            "slip_free_width_m", "min_pair_surface_gap_m", "row_gap_m", "min_edge_gap_m",
                            "slip_back_face_x_m", "route_budget_m", "reference_length_m", "goal_distance_m",
                            "outer_ring_radius_m", "start_distance_m"):
                        self.assertAlmostEqual(w[key], value * s, places=9, msg=(task, key))
                self.assertAlmostEqual(w["route_budget_m"], b["route_budget_m"] * s, places=6)

    def test_vessel_records(self):
        # E5: expectations are recomputed from the registry (CONTEXT_R3
        # decision 1) instead of D4's pinned 15:11 numbers (258 kg WAM-V,
        # 0.1 s BlueBoat lag, cruise 2.0 m/s WAM-V).
        bb, wv = cfg.BLUEBOAT, cfg.WAMV
        sb, sw = reg.get_vehicle_v2("blueboat_v2"), reg.get_vehicle_v2(wv.key)
        self.assertAlmostEqual(bb.length_m, 1.200825846195221, places=12)
        self.assertAlmostEqual(wv.length_m, 5.37719612, places=6)
        for rec, spec in ((bb, sb), (wv, sw)):
            rev = sum(m.reverse_limit_n for m in spec.motors)
            tau = max(m.time_constant_s for m in spec.motors)
            u = spec.speeds.cruise_m_s
            self.assertAlmostEqual(rec.stop_distance_m, u * tau + u * u / (2 * rev / spec.mass_kg), places=12)
            h = spec.hydrostatics
            lo, hi = rec.heave_limits_m
            self.assertAlmostEqual(lo, -(h.freeboard_m + h.draft_m + h.freeboard_m), places=12)
            self.assertAlmostEqual(hi, h.draft_m + h.draft_m + h.freeboard_m, places=12)
            self.assertEqual(rec.com_height_above_waterline_m, h.com_height_above_water_m)
        lo, hi = bb.heave_limits_m
        self.assertAlmostEqual(lo, -(0.15302155053041147 + 0.37601467704772951), places=9)
        self.assertAlmostEqual(hi, 0.22299312651731804 + 0.37601467704772951, places=9)
        self.assertEqual(bb.turning_radius_m, bb.length_m)
        self.assertEqual(cfg.FORBIDDEN_CFG_FLAGS_TRUE, ("obs_sea_state",))
        self.assertAlmostEqual(bb.com_height_above_waterline_m, 0.007006873572833999, places=15)
        plan = cfg.JOURNAL_PLAN_CLASSES[("docking", wv.key)](1)
        self.assertEqual(plan.nominal_com_height_m, wv.com_height_above_waterline_m)
        lim = plan.validity_limits
        self.assertAlmostEqual(lim["min_heave_m"], -(sw.hydrostatics.freeboard_m + wv.hull_depth_m), places=12)
        self.assertAlmostEqual(lim["max_heave_m"], sw.hydrostatics.draft_m + wv.hull_depth_m, places=12)

    def test_blueboat_footprint_encloses_c1_verified_asset_bounds(self):
        # C1 AUDIT_physics.md:17: legacy asset bounds and COM in the legacy
        # frame; D5 v2 rotates the asset 180 deg about z (PARAMS.md:28,74).
        lo = np.array((-0.600826, -0.449360))
        hi = np.array((0.597951, 0.449360))
        com = np.array((0.006, 0.047))
        corners = np.array(((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])))
        rotated_rel = -(corners - com)          # 180 deg rotation of COM-relative points
        x0, x1, y0, y1 = cfg.BLUEBOAT.footprint.bounds
        self.assertTrue(np.all(rotated_rel[:, 0] >= x0 - 1e-6) and np.all(rotated_rel[:, 0] <= x1 + 1e-6))
        self.assertTrue(np.all(rotated_rel[:, 1] >= y0 - 1e-6) and np.all(rotated_rel[:, 1] <= y1 + 1e-6))

    def test_wamv_footprint_encloses_vrx_mesh_bounds(self):
        # WAM-V parameter report (usvbench_gazebo/NC.../WAM-V...md:100-101):
        # hull x -2.456..+2.555, y +-1.244; with outboards x from -2.8185.
        # These are VRX base_link coordinates. E5: the registry polygon is
        # COM-relative (v2 origin = COM), so shift it by the registry's COM in
        # base_link (vehicles_v2_data.WAMV[config]['com_base_link_m']).
        config = "bare" if cfg.WAMV.key.endswith("_bare") else "competition"
        com = reg._data.WAMV[config]["com_base_link_m"]
        xs = [p[0] + com[0] for p in cfg.WAMV.body_polygon_m]
        ys = [p[1] + com[1] for p in cfg.WAMV.body_polygon_m]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        self.assertLessEqual(x0, -2.8185)
        self.assertGreaterEqual(x1, 2.555)
        self.assertLessEqual(y0, -1.244)
        self.assertGreaterEqual(y1, 1.244)

    def test_cross_delivery_consistency_with_d5_registry(self):
        # E5: was an opt-in skip against D5's 15:11 draft schema; now always
        # on against the registry in the tree (every field of both records).
        for rec in (cfg.BLUEBOAT, cfg.WAMV):
            spec = reg.get_vehicle_v2(rec.key)
            self.assertEqual(tuple(map(tuple, spec.collision.polygon_com_xy_m)), rec.body_polygon_m)
            self.assertEqual(tuple(spec.com_actor_m[:2]), rec.com_body_xy_m)
            self.assertEqual(spec.mass_kg, rec.mass_kg)
            self.assertEqual(spec.speeds.cruise_m_s, rec.cruise_speed_m_s)
            self.assertEqual(spec.speeds.top_model_m_s, rec.top_speed_m_s)
            self.assertEqual(sum(m.reverse_limit_n for m in spec.motors), rec.reverse_thrust_total_n)
            self.assertEqual(max(m.time_constant_s for m in spec.motors), rec.motor_time_constant_s)
            self.assertEqual(spec.hydrostatics.draft_m, rec.draft_m)
            self.assertEqual(spec.hydrostatics.freeboard_m, rec.freeboard_m)
            self.assertEqual(spec.hydrostatics.com_height_above_water_m, rec.com_height_above_waterline_m)


# E5: parent-field checks run against the tree itself (J-base), not C2's copy.
C2_TASKS = ROOT / "tasks"


def _class_fields(module_path, class_name, seen=None):
    """Annotated field names of a class, following bases inside the same file."""
    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    fields = set()
    node = classes.get(class_name)
    while node is not None:
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields.add(stmt.target.id)
            if isinstance(stmt, ast.Assign):
                fields.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        base = node.bases[0].id if node.bases and isinstance(node.bases[0], ast.Name) else None
        node = classes.get(base)
    return fields


class RuntimeTemplateTests(unittest.TestCase):
    PARENTS = {"HazardNavV3EnvCfg": "hazard_nav/hazard_nav_env_cfg.py",
               "PathFollowingEnvCfg": "path_following/path_following_env_cfg.py",
               "StationKeepingEnvCfg": "station_keeping/station_keeping_env_cfg.py",
               "DockingBlueBoatWallEnvCfg": "docking/docking_env_cfg.py",
               "PathHazardV2EnvCfg": "path_hazard/path_hazard_env_cfg.py",
               "HarborMissionEnvCfg": "harbor_mission/harbor_mission_env_cfg.py"}

    def setUp(self):
        self.source = ROOT / "tasks/_shared/journal_runtime_cfg_v2.py"
        self.tree = ast.parse(self.source.read_text(encoding="ascii"))
        self.classes = [n for n in self.tree.body if isinstance(n, ast.ClassDef)]

    def test_24_classes_bind_known_plans_and_real_parent_fields(self):
        self.assertEqual(len(self.classes), 24)
        seen = set()
        for node in self.classes:
            base = node.bases[0].id
            self.assertIn(base, self.PARENTS)
            values = {s.target.id: ast.literal_eval(s.value) for s in node.body
                      if isinstance(s, ast.AnnAssign) and s.value is not None}
            key = (values["journal_task"], values["vehicle"])
            self.assertIn(key, cfg.JOURNAL_PLAN_CLASSES)
            seen.add(key)
            parent_fields = _class_fields(C2_TASKS / self.PARENTS[base], base)
            for name in values:
                if not name.startswith("journal_"):
                    self.assertIn(name, parent_fields, (node.name, name))
        self.assertEqual(seen, set(cfg.JOURNAL_PLAN_CLASSES))

    def test_post_init_order_and_guards_with_fake_parent(self):
        funcs = [n for n in self.tree.body if isinstance(n, ast.FunctionDef)]
        ns = {"plans": cfg, "os": SimpleNamespace(environ={})}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(self.source), "exec"), ns)
        calls = []

        def make(task, vessel):
            c = SimpleNamespace(journal_task=task, vehicle=vessel, journal_geometry_tier=2,
                                journal_contract_version=2, journal_mode=True,
                                episode_length_s=120.0, appear_count=1,
                                max_obstacles=14, layout_mode="journal", obs_sea_state=True,
                                underwater_physics_cfg=SimpleNamespace(physics_version=1))

            def parent():
                calls.append(c.underwater_physics_cfg.physics_version)
            return c, parent

        for task in cfg.TASKS:
            for vessel in cfg.VESSELS:
                c, parent = make(task, vessel)
                ns["_journal_post_init"](c, parent)
                self.assertEqual(calls[-1], 2)            # v2 selected before the parent rebuild
                self.assertEqual(c.underwater_physics_cfg.physics_version, 2)
                self.assertEqual(c.appear_count, 0)
                self.assertFalse(c.obs_sea_state)
                self.assertEqual(c.max_obstacles, max(1, cfg.STATIC_CAPACITY[task] + cfg.D7_RESERVED_SLOTS[task]))
                self.assertAlmostEqual(c.episode_length_s, c.journal_plan.max_deadline_s)
                c.vehicle = "wamv"
                with self.assertRaises(ValueError):
                    ns["journal_validate"](c)
        c, parent = make("station_keeping", "blueboat_v2")
        ns["_journal_post_init"](c, parent)
        c.obs_sea_state = True
        with self.assertRaises(ValueError):
            ns["journal_validate"](c)
        c.obs_sea_state = False
        ns["journal_validate"](c)
        c.journal_mode = False                               # E5: J-base validated layer required
        with self.assertRaisesRegex(ValueError, "journal_mode"):
            ns["journal_validate"](c)
        c.journal_mode = True
        ns["journal_validate"](c)
        # E5: another registered profile goes through plan_class (sensitivity runs)
        c2, parent2 = make("docking", "wamv_vrx_v2_opt2")
        ns["_journal_post_init"](c2, parent2)
        self.assertEqual(c2.journal_plan.vessel, "wamv_vrx_v2_opt2")
        ns["os"].environ["GOAL_RADIUS_OVERRIDE"] = "3"      # C1 finding 61
        with self.assertRaisesRegex(ValueError, "ambient overrides"):
            ns["journal_validate"](c)
        self.assertNotIn("isaaclab", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
