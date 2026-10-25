"""CPU geometry and docking-contract tests; never import Isaac Lab packages."""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import warnings

import numpy as np


sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
LEGACY = Path(os.environ.get("USVBENCH_LEGACY_ROOT", "C:/Users/BRADY/usvbench"))
GEOMETRY = Path("tasks/path_hazard/path_hazard_geometry.py")
DOCK_CFG = Path("tasks/docking/docking_env_cfg.py")
DOCK_ENV = Path("tasks/docking/docking_env.py")


def load_geometry(name, root):
    spec = importlib.util.spec_from_file_location(name, root / GEOMETRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Layout generation only uses NumPy. Do not import the tensor/raycast
    # dependency for these tests; annotations are deferred in the source.
    with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
        spec.loader.exec_module(module)
    return module


geometry = load_geometry("physics_v2_geometry", REPO)
legacy_geometry = load_geometry("physics_v2_legacy_geometry", LEGACY)


def docking_defaults(root):
    tree = ast.parse((root / DOCK_CFG).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DockingEnvCfg")
    return {
        node.target.id: ast.literal_eval(node.value)
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and node.target.id in ("eval_level_distances", "eval_level_distances_v2")
    }


def docking_frozen_block(root):
    """Execute the real selection block, excluding all Isaac imports/setup."""
    tree = ast.parse((root / DOCK_ENV).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DockingEnv")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = next(
        n for n in init.body
        if isinstance(n, ast.If)
        and "curriculum_frozen" in ast.unparse(n.test)
    )
    function = ast.parse("def apply(self):\n    pass\n").body[0]
    function.body = [copy.deepcopy(block)]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(root / DOCK_ENV), "exec"), namespace)
    return namespace["apply"]


class PathHazardV2Tests(unittest.TestCase):
    def assert_arrays_exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_v1_candidates_preserve_legacy_numbers_and_rng_all_tiers(self):
        # Legacy tiers 1/2 read unwritten allocator memory. Supply the same
        # allocator bytes to each version; no reproducibility claim is made
        # for those tiers with arbitrary real allocator contents.
        for tier in range(5):
            for seed in range(8):
                original_rng = np.random.default_rng(seed)
                copied_rng = np.random.default_rng(seed)
                controlled_numpy = types.SimpleNamespace(**{
                    **vars(np), "empty": lambda shape, dtype: np.full(shape, 999.0, dtype=dtype)
                })
                with mock.patch.object(geometry, "np", controlled_numpy), mock.patch.object(legacy_geometry, "np", controlled_numpy):
                    expected = legacy_geometry._sample_candidate(6, original_rng, tier)
                    actual = geometry._sample_candidate(6, copied_rng, tier, physics_version=1)
                self.assertEqual(original_rng.bit_generator.state, copied_rng.bit_generator.state)
                self.assertEqual(expected is None, actual is None)
                if expected is not None:
                    for result, original in zip(actual, expected):
                        self.assert_arrays_exact(result, original)

    def test_v1_complete_layouts_match_original_exactly(self):
        for tier in (0, 3):
            for seed in (0, 3, 11):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    expected = legacy_geometry.sample_layout(np.random.default_rng(seed), blockers_override=tier)
                    actual = geometry.sample_layout(np.random.default_rng(seed), blockers_override=tier, physics_version=1)
                for field in ("route_points", "centers", "radii", "on_line_mask", "blocker_segment_indices", "leg_geodesic_lengths"):
                    self.assert_arrays_exact(getattr(actual, field), getattr(expected, field))
                self.assertEqual(actual.attempts, expected.attempts)

    def test_v2_all_tiers_are_finite_feasible_and_keep_their_blockers(self):
        for tier in range(5):
            for seed in (0, 3, 11, 29):
                with self.subTest(tier=tier, seed=seed), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    layout = geometry.sample_layout(np.random.default_rng(seed), blockers_override=tier, physics_version=2)
                    self.assertTrue(np.isfinite(layout.centers).all())
                    self.assertEqual(layout.on_line_blocker_count, tier)
                    self.assertTrue(layout.chain_feasible)
                    self.assertTrue(geometry.gate_disks_clear(layout.route_points, layout.centers, layout.radii))
                    self.assertTrue(geometry._inflated_obstacles_separated(layout.centers, layout.radii))
                    measured = geometry.count_on_line_blockers(layout.route_points, layout.centers)
                    if tier == 0:
                        self.assertEqual(measured, 0)
                    else:
                        self.assertGreaterEqual(measured, tier)
                    self.assertTrue(np.all(layout.blocker_segment_indices[tier:] == -1))

    def test_v2_tier4_minimum_count_does_not_overwrite_fourth_blocker(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                layout = geometry.sample_layout(np.random.default_rng(seed), obstacle_count=4, blockers_override=4, physics_version=2)
                self.assertEqual(layout.obstacle_count, 4)
                self.assertEqual(set(layout.blocker_segment_indices), set(range(4)))
                for center, segment_index in zip(layout.centers, layout.blocker_segment_indices):
                    fraction, distance = geometry._point_segment_projection(
                        center, layout.route_points[segment_index], layout.route_points[segment_index + 1]
                    )
                    self.assertGreaterEqual(fraction, geometry.ON_LINE_FRACTION_MIN)
                    self.assertLessEqual(fraction, geometry.ON_LINE_FRACTION_MAX)
                    self.assertLessEqual(distance, geometry.ON_LINE_OFFSET_MAX_M + 1e-12)

    def test_v2_detects_an_unwritten_slot(self):
        class DroppedWriteArray(np.ndarray):
            def __setitem__(self, index, value):
                if index != 5:
                    super().__setitem__(index, value)

        original_full = np.full

        def full_with_dropped_write(shape, value, **kwargs):
            result = original_full(shape, value, **kwargs)
            return result.view(DroppedWriteArray) if shape == (6, 2) else result

        controlled_numpy = types.SimpleNamespace(**{**vars(np), "full": full_with_dropped_write})
        with mock.patch.object(geometry, "np", controlled_numpy):
            with self.assertRaisesRegex(AssertionError, "every PathHazard center slot must be written"):
                geometry._sample_candidate(6, np.random.default_rng(0), 3, physics_version=2)


class DockingV2Tests(unittest.TestCase):
    def run_selection(self, root, version, level, override=0.0, legacy_distances=None):
        defaults = docking_defaults(root)
        if legacy_distances is not None:
            defaults["eval_level_distances"] = legacy_distances
        cfg = types.SimpleNamespace(
            **defaults, curriculum_frozen=True, eval_level=level,
            spawn_distance_override_m=override,
        )
        env = types.SimpleNamespace(
            cfg=cfg, physics_cfg=types.SimpleNamespace(physics_version=version),
            curriculum=types.SimpleNamespace(spawn_distance=-1.0),
        )
        docking_frozen_block(root)(env)
        return env.curriculum.spawn_distance

    def test_v1_table_and_selection_equal_original_exactly(self):
        self.assertEqual(docking_defaults(REPO)["eval_level_distances"], docking_defaults(LEGACY)["eval_level_distances"])
        for level in range(-2, 7):
            for override in (0.0, -1.0, 3.75):
                for distances in (None, (31.0, 19.0, 7.0, 1.0)):
                    actual = self.run_selection(REPO, 1, level, override, distances)
                    expected = self.run_selection(LEGACY, 1, level, override, distances)
                    self.assertEqual(actual.hex(), expected.hex())

    def test_v2_level_table_is_easiest_to_hardest(self):
        self.assertEqual(tuple(self.run_selection(REPO, 2, i) for i in range(4)), (2.0, 8.0, 15.0, 25.0))
        self.assertEqual(self.run_selection(REPO, 2, -1), 2.0)
        self.assertEqual(self.run_selection(REPO, 2, 99), 25.0)
        self.assertEqual(self.run_selection(REPO, 2, 2, override=4.5), 4.5)


class ConfigRebuildV2Tests(unittest.TestCase):
    def test_all_vehicle_cfg_rebuilds_preserve_selected_version(self):
        # Compile real __post_init__ registry prefixes with light builders;
        # full config imports would load Isaac. super() remains a real class
        # closure, while the builders deliberately reset their output to v1.
        spec = importlib.util.spec_from_file_location(
            "physics_v2_config_review_vehicles", REPO / "tasks/_shared/vehicles.py"
        )
        vehicles = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = vehicles
        spec.loader.exec_module(vehicles)
        checked = []
        for path in (REPO / "tasks").glob("*/*env_cfg.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                method = next((
                    n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"
                    and "_build_underwater_physics_cfg" in ast.unparse(n)
                ), None)
                if method is None:
                    continue
                method = copy.deepcopy(method)
                registry_end = next(
                    i for i, node in enumerate(method.body)
                    if isinstance(node, ast.Assign)
                    and ast.unparse(node.targets[0]) == "self.yaw_torque_max"
                )
                method.body = method.body[:registry_end + 1]
                extracted = ast.parse("class TestCfg:\n    pass\n").body[0]
                extracted.body = [copy.deepcopy(method)]
                module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
                namespace = {
                    "physics_version": vehicles.physics_version,
                    "get_vehicle": vehicles.get_vehicle,
                    "_build_robot_cfg": lambda vehicle: vehicle.name,
                    "_build_underwater_physics_cfg": lambda vehicle: types.SimpleNamespace(physics_version=1),
                }
                exec(compile(module, str(path), "exec"), namespace)
                for version in (None, 1, 2):
                    with self.subTest(path=str(path.relative_to(REPO)), version=version):
                        cfg = namespace["TestCfg"]()
                        cfg.vehicle = "blueboat"
                        original = types.SimpleNamespace()
                        if version is not None:
                            original.physics_version = version
                        cfg.underwater_physics_cfg = original
                        cfg.validate_after_overrides = lambda: None
                        cfg.__post_init__()
                        self.assertIsNot(cfg.underwater_physics_cfg, original)
                        self.assertEqual(cfg.underwater_physics_cfg.physics_version, version or 1)
                        self.assertEqual(cfg.thrust_max_fwd, vehicles.get_vehicle("blueboat").thrust_fwd_n)
                checked.append(path.parent.name)
        self.assertEqual(set(checked), {
            "docking", "path_following", "station_keeping", "path_hazard",
            "hazard_nav", "harbor_mission", "station_keeping_boat",
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
