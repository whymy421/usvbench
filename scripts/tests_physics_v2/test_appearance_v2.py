"""CPU regressions for the v2 appearance slot budget; never import Isaac Lab."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
ENV_PATH = REPO / "tasks/hazard_nav/hazard_nav_env.py"
CFG_PATH = REPO / "tasks/hazard_nav/hazard_nav_env_cfg.py"
ORIGINAL_ENV = Path(r"C:\Users\BRADY\usvbench\tasks\hazard_nav\hazard_nav_env.py")


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry = load_path("appearance_v2_geometry", ENV_PATH.with_name("hazard_geometry.py"))


def method_node(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def lift(path, class_name, method_name, **extra):
    node = method_node(path, class_name, method_name)
    node.decorator_list = []
    namespace = dict(np=np, torch=torch, math=math, **extra)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


slot_count = lift(CFG_PATH, "HazardNavEnvCfg", "obstacle_slot_count")


class FakeCfg(types.SimpleNamespace):
    @property
    def obstacle_slot_count(self):
        return slot_count(self)


def config(version=2, static_slots=14, appear_count=1):
    return FakeCfg(
        max_obstacles=static_slots,
        appear_count=appear_count,
        underwater_physics_cfg=types.SimpleNamespace(physics_version=version),
        ray_count=36, appear_max_attempts=36, appear_distance_m=12.0,
        appear_radius_m=0.8, appear_bfs_cell_m=0.25, half_beam_m=0.45,
        ray_max_range_m=30.0,
    )


def fake_env(version=2, static_slots=14, static_count=14, appear_count=1):
    cfg = config(version, static_slots, appear_count)
    count = cfg.obstacle_slot_count
    env = types.SimpleNamespace(
        cfg=cfg, num_envs=1, device="cpu", _appear_count=appear_count,
        control_step_s=1.0 / 60.0, goal_radius=2.0,
        episode_length_buf=torch.tensor([601]),
        _appear_pending=torch.tensor([True]), _appear_time_s=torch.tensor([10.0]),
        _appear_happened=torch.tensor([False]), _appear_skipped=torch.tensor([False]),
        _appear_spawned=torch.tensor([0]), _appear_azimuth_rad=torch.tensor([torch.nan]),
        _appear_clearance_m=torch.tensor([torch.nan]), _appear_distance_m=torch.tensor([torch.nan]),
        _appear_rngs=[np.random.default_rng(1729)],
        obstacle_centers=torch.zeros((1, count, 2)), obstacle_radii=torch.zeros((1, count)),
        obstacle_active=torch.zeros((1, count), dtype=torch.bool),
        obstacle_count=torch.tensor([static_count]), target_pos=torch.tensor([[25.0, 0.0]]),
        scene=types.SimpleNamespace(env_origins=torch.zeros((1, 3))),
        _com_xy=lambda: torch.zeros((1, 2)),
        _forward_2d=lambda: torch.tensor([[1.0, 0.0]]),
    )
    env.obstacle_active[:, :static_count] = True
    env.obstacle_centers[0, :static_count, 0] = torch.arange(static_count)
    env.obstacle_centers[0, :static_count, 1] = 10.0
    env.obstacle_radii[:, :static_count] = 0.8
    return env


def accepting_planner(*args, **kwargs):
    # Capacity is independent of BFS admission. Use a known accepted plan so
    # the final reserved slot is always exercised, including multiple pillars.
    return types.SimpleNamespace(center=np.array([12.0, -2.0]), radius_m=0.8,
                                 azimuth_rad=0.0, distance_m=12.0)


def appearance_hook(path=ENV_PATH, planner=accepting_planner):
    return lift(path, "HazardNavEnv", "_maybe_appear_obstacles", plan_obstacle_appearance=planner)


def execute_capacity_block(method_name, env):
    """Run actual shipped allocation statements, excluding simulator setup."""
    node = method_node(ENV_PATH, "HazardNavEnv", method_name)
    start = next(i for i, item in enumerate(node.body)
                 if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id == "max_obstacles" for target in item.targets))
    if method_name == "__init__":
        # Include moving-obstacle arrays and stop before appearance bookkeeping.
        stop = next(i for i in range(start, len(node.body))
                    if isinstance(node.body[i], ast.Assign)
                    and any(isinstance(target, ast.Attribute) and target.attr == "_appear_count"
                            for target in node.body[i].targets))
    else:
        stop = next(i for i in range(start, len(node.body)) if isinstance(node.body[i], ast.If))
    namespace = dict(np=np, torch=torch, self=env, num_resets=2)
    module = ast.Module(body=node.body[start:stop], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ENV_PATH), "exec"), namespace)
    return namespace


class AppearanceV2Tests(unittest.TestCase):
    def test_version1_keeps_static_capacity_even_with_appearance(self):
        for static_slots in (14, 96, 136):
            for count in (0, 1, 3):
                self.assertEqual(config(1, static_slots, count).obstacle_slot_count, static_slots)
        cfg = config()
        cfg.underwater_physics_cfg = types.SimpleNamespace()
        self.assertEqual(cfg.obstacle_slot_count, 14)

    def test_v2_budget_follows_overrides_without_mutating_static_budget(self):
        cfg = config()
        self.assertEqual(cfg.obstacle_slot_count, 15)
        cfg.appear_count = 3
        self.assertEqual(cfg.obstacle_slot_count, 17)
        self.assertEqual(cfg.max_obstacles, 14)
        cfg.appear_count = 0
        self.assertEqual(cfg.obstacle_slot_count, 14)

    def test_constructor_and_reset_share_the_extra_capacity(self):
        for version, expected in ((1, 14), (2, 17)):
            env = types.SimpleNamespace(cfg=config(version, appear_count=3), num_envs=2, device="cpu")
            env.cfg.layout_mode = "moving_basin"
            execute_capacity_block("__init__", env)
            self.assertEqual(tuple(env.obstacle_centers.shape), (2, expected, 2))
            self.assertEqual(tuple(env.obstacle_active.shape), (2, expected))
            self.assertEqual(tuple(env._moving_params.shape), (2, expected, 7))
            reset = execute_capacity_block("_reset_idx", env)
            self.assertEqual(reset["local_centers_np"].shape, (2, expected, 2))
            self.assertEqual(reset["active_np"].shape, (2, expected))
            self.assertEqual(reset["moving_params_np"].shape, (2, expected, 7))
            self.assertFalse(reset["active_np"].any())

    def test_level3_plus_one_or_many_appearances_writes_extra_slots(self):
        layout = geometry.sample_layout(3, rng=np.random.default_rng(912), max_attempts=100)
        self.assertEqual(layout.obstacle_count, 14)
        for extra in (1, 3):
            env = fake_env(appear_count=extra)
            env.obstacle_centers[0, :14] = torch.as_tensor(layout.centers, dtype=torch.float32)
            env.obstacle_radii[0, :14] = torch.as_tensor(layout.radii, dtype=torch.float32)
            before = env.obstacle_centers[:, :14].clone()
            appearance_hook()(env)
            self.assertEqual(int(env.obstacle_active.sum()), 14 + extra)
            self.assertEqual(int(env._appear_spawned[0]), extra)
            self.assertTrue(torch.equal(before, env.obstacle_centers[:, :14]))
            self.assertTrue(bool(env._appear_happened[0]))

    def test_static_layout_cannot_consume_reserved_appearance_budget(self):
        node = method_node(ENV_PATH, "HazardNavEnv", "_reset_idx")
        loop = next(item for item in node.body if isinstance(item, ast.For)
                    and isinstance(item.target, ast.Name) and item.target.id == "row")
        count_index = next(i for i, item in enumerate(loop.body)
                           if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name)
                           and target.id == "count" for target in item.targets))
        guard = loop.body[count_index + 1]
        self.assertIsInstance(guard, ast.If)
        code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
                       str(ENV_PATH), "exec")
        for version in (1, 2):
            env = fake_env(version=version)
            exec(code, dict(self=env, count=14))
            if version == 2:
                with self.assertRaisesRegex(RuntimeError, "appearance slots are reserved separately"):
                    exec(code, dict(self=env, count=15))
            else:
                exec(code, dict(self=env, count=15))

    def test_v1_matches_original_hook_numbers_and_rng_exactly(self):
        old, new = fake_env(version=1, static_count=12), fake_env(version=1, static_count=12)
        appearance_hook(ORIGINAL_ENV)(old)
        appearance_hook()(new)
        for name, value in vars(old).items():
            if isinstance(value, torch.Tensor):
                np.testing.assert_array_equal(value.numpy(), getattr(new, name).numpy(), err_msg=name)
        self.assertEqual(old._appear_rngs[0].bit_generator.state, new._appear_rngs[0].bit_generator.state)

    def test_v1_retains_original_level3_overflow(self):
        for path in (ORIGINAL_ENV, ENV_PATH):
            with self.assertRaisesRegex(RuntimeError, "slot 14 but max_obstacles=14"):
                appearance_hook(path)(fake_env(version=1))

    def test_inert_and_skipped_appearance_do_not_consume_extra_slots(self):
        appearance_hook()(types.SimpleNamespace(_appear_count=0))
        env = fake_env()
        appearance_hook(planner=lambda *args, **kwargs: None)(env)
        self.assertEqual(int(env.obstacle_active.sum()), 14)
        self.assertEqual(int(env._appear_spawned[0]), 0)
        self.assertTrue(bool(env._appear_skipped[0]))

    def test_v2_refreshes_display_after_appearance(self):
        for version in (1, 2):
            env = fake_env(version=version, static_count=12)
            env._obstacle_translate_ops = [[object()]]
            calls = []
            env._update_render_only_hazard_markers = lambda *args: calls.append(args)
            appearance_hook()(env)
            self.assertEqual(len(calls), int(version == 2))
            if calls:
                self.assertEqual(calls[0][2].shape[1], 15)
                self.assertTrue(bool(calls[0][4][0, 12]))

    def test_render_creation_and_reset_update_use_total_capacity(self):
        class Value:
            def Set(self, value):
                self.value = value

        class Cylinder:
            def __init__(self):
                self.visible = False
                self.attrs = {}

            @classmethod
            def Define(cls, stage, path):
                return cls()

            def GetPrim(self):
                return self

            def __getattr__(self, name):
                return lambda: self.attrs.setdefault(name, Value())

            def MakeVisible(self):
                self.visible = True

            def MakeInvisible(self):
                self.visible = False

        omni = types.ModuleType("omni")
        omni.usd = types.SimpleNamespace(get_context=lambda: types.SimpleNamespace(get_stage=lambda: object()))
        pxr = types.ModuleType("pxr")
        pxr.Gf = types.SimpleNamespace(Vec3f=lambda *args: args, Vec3d=lambda *args: args)
        pxr.Vt = types.SimpleNamespace(Vec3fArray=lambda value: value)
        pxr.UsdGeom = types.SimpleNamespace(Cylinder=Cylinder, Imageable=lambda prim: prim)
        for version, expected in ((1, 14), (2, 15)):
            env = fake_env(version=version, static_count=12)
            env.physics_cfg = types.SimpleNamespace(water_surface_z=0.0)
            env.cfg.visual = types.SimpleNamespace(enable_goal_ring=False, enable_obstacles=True,
                                                  obstacle_color=(0.3, 0.3, 0.3), obstacle_height_m=2.0)
            env._goal_translate_ops, env._obstacle_translate_ops = [], []
            env._obstacle_prims, env._obstacle_radius_attrs = [], []
            env._get_or_add_translate_op = lambda prim: Value()
            with patch.dict(sys.modules, {"omni": omni, "omni.usd": omni.usd, "pxr": pxr}):
                lift(ENV_PATH, "HazardNavEnv", "_create_render_only_hazard_markers")(env)
                self.assertEqual(len(env._obstacle_prims[0]), expected)
                env.obstacle_active[0, -1] = True
                lift(ENV_PATH, "HazardNavEnv", "_update_render_only_hazard_markers")(
                    env, torch.tensor([0]), env.target_pos, env.obstacle_centers,
                    env.obstacle_radii, env.obstacle_active,
                )
                self.assertTrue(env._obstacle_prims[0][-1].visible)


if __name__ == "__main__":
    unittest.main(verbosity=2)
