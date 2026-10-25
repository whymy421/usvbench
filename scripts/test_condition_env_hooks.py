"""CPU checks of the actual env method ASTs; never import Isaac Lab.

Run from the copied repo: python -B scripts/test_condition_env_hooks.py
Baseline files are preserved at ../baseline/tasks by the C4 copy workflow.
These are tensor/hook tests, not a substitute for a future simulator run.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch


REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO.parent / "baseline"
FAMILIES = ("station_keeping", "hazard_nav")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Relative imports inside the extracted observation methods resolve only to
# pure modules; neither tasks/__init__.py nor env modules are executed.
for name, path in (
    ("_condition_cpu", REPO / "tasks"),
    ("_condition_cpu._shared", REPO / "tasks/_shared"),
):
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
OBS = load_file("_condition_cpu._shared.obs_superset", REPO / "tasks/_shared/obs_superset.py")
RESTORING = load_file("_condition_cpu._shared.restoring", REPO / "tasks/_shared/restoring.py")
CONDITIONS = load_file("_condition_cpu._shared.conditions", REPO / "tasks/_shared/conditions/__init__.py")


def quat_apply(q, v):
    qv = q[..., 1:]
    cross = torch.cross(qv, v, dim=-1)
    return v + 2.0 * (q[..., :1] * cross + torch.cross(qv, cross, dim=-1))


def quat_apply_inverse(q, v):
    qv = q[..., 1:]
    cross = torch.cross(qv, v, dim=-1)
    return v - 2.0 * q[..., :1] * cross + 2.0 * torch.cross(qv, cross, dim=-1)


def source_class(root, family):
    path = root / "tasks" / family / (family + "_env.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    name = "StationKeepingEnv" if family == "station_keeping" else "HazardNavEnv"
    return path, next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def extracted_class(root, family):
    path, original = source_class(root, family)
    wanted = {
        "_pre_physics_step", "_apply_action", "_get_observations", "_root_quat",
        "_com_xy", "_forward_2d", "_body_motion_measurement", "_body_motion_observation",
    }
    cls = ast.ClassDef(
        name="ExtractedEnv", bases=[], keywords=[], decorator_list=[],
        body=[copy.deepcopy(n) for n in original.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
    )
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {
        "__package__": "_condition_cpu." + family,
        "torch": torch,
        "math_utils": SimpleNamespace(quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse),
        "restoring_torque_body": RESTORING.restoring_torque_body,
        "SPEED_SCALE_MPS": OBS.SPEED_SCALE_MPS,
        "SUPERSET_DIM": OBS.SUPERSET_DIM,
        "SUPERSET_DIM_V2": OBS.SUPERSET_DIM_V2,
        "SLICES": OBS.SLICES,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["ExtractedEnv"]


class Robot:
    def __init__(self, data):
        self.num_bodies = 2
        self.data = data
        self.calls = []

    def set_external_force_and_torque(self, forces, torques):
        self.calls.append((forces.clone(), torques.clone()))


class Sea:
    def forces(self, position, velocity, t):
        f = torch.zeros((len(position), 3))
        f[:, :2] = 0.1 * position - 0.2 * velocity + t
        return f, -0.3 * f


class RecordingConditions:
    def __init__(self):
        self.events = []

    def action_in(self, actions):
        self.events.append("action")
        assert torch.all(torch.abs(actions) <= 1.0)
        return actions

    def reset_runtime(self, *, progress=None, env_ids=None):
        self.events.append("runtime" if env_ids is None else "reset")

    def command(self, thrust, yaw, dt, half_beam):
        self.events.append("command")
        self.dt = dt
        return thrust, yaw

    def scale_drag(self, drag):
        self.events.append("drag")
        return drag

    def environmental_wrench(self, position, velocity, dt, **context):
        self.events.append("wrench")
        assert context["orientation_wxyz"].shape == (len(position), 4)
        assert context["angular_velocity_w"].shape == (len(position), 3)
        return torch.zeros((len(position), 3)), torch.zeros((len(position), 3))

    def dynamics(self, forces, torques):
        self.events.append("dynamics")
        return forces, torques

    def wants(self, name):
        return True

    def apply(self, name, tensor):
        self.events.append("observation:" + name)
        return tensor + 0.125


def make_env(root, family, *, isotropic=False, six_velocity=False, bow_y=False, legacy=False):
    env = extracted_class(root, family)()
    n = 5
    generator = torch.Generator().manual_seed(984)
    quaternion = torch.randn((n, 4), generator=generator)
    quaternion /= torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    velocity = torch.randn((n, 6 if six_velocity else 3), generator=generator)
    position = torch.randn((n, 3), generator=generator)
    data = SimpleNamespace(
        root_link_quat_w=quaternion, root_quat_w=quaternion,
        root_com_vel_w=velocity, root_ang_vel_w=torch.randn((n, 3), generator=generator),
        root_pos_w=position, root_com_pos_w=position,
    )
    env.robot = Robot(data)
    env.num_envs = n
    env.device = "cpu"
    env.cfg = SimpleNamespace(
        thrust_max_fwd=80.0, thrust_max_rev=48.0, yaw_torque_max=23.0,
        sim=SimpleNamespace(dt=1.0 / 120.0), half_beam_m=0.45,
        thrust_cap_scale=1.0, thrust_imbalance=0.0, motor_tau_s=0.0,
        drag_scale=1.0, mass_scale=1.0, payload_mass_kg=0.0,
        payload_offset_x_m=0.2, payload_offset_y_m=-0.1,
        payload_ref_mass_kg=17.26, payload_ref_izz_kgm2=3.2,
        max_spawn_distance=20.0, obs_sea_state=False, obs_kinematic=True,
        yaw_rate_obs_scale_rad_s=1.0, nav_targets_waypoint=False,
        ray_max_range_m=30.0, obs_feasibility=False,
        obs_v3=True, obs_v2=False, emit_superset_obs=False, superset_version=1,
        ray_count=8, decimation=2, seed=47, vehicle="blueboat",
        condition_presets=(), conditions=None,
    )
    for axis in ("thrust_cap_scale", "thrust_imbalance", "motor_tau_s", "drag_scale", "mass_scale", "payload_mass_kg"):
        setattr(env.cfg, axis + "_choices", ())
    if legacy:
        env.cfg.thrust_cap_scale = 0.8
        env.cfg.thrust_imbalance = 0.15
        env.cfg.motor_tau_s = 0.2
        env.cfg.drag_scale = 1.3
        env.cfg.mass_scale = 1.4
        env.cfg.payload_mass_kg = 2.0
    env.physics_cfg = SimpleNamespace(
        sway_lin_damping=None if isotropic else 27.0,
        sway_quad_damping=None if isotropic else 18.0,
        surge_lin_damping=9.1, surge_quad_damping=5.9, heave_damping=200.0,
        yaw_lin_damping=4.0, yaw_quad_damping=6.0, rollpitch_rate_damping=470.0,
        restoring_stiffness_roll=280.0,
        restoring_stiffness_pitch=280.0 if isotropic else 141.0,
        enable_current=legacy,
    )
    env.cfg.underwater_physics_cfg = env.physics_cfg
    env._fwd_x, env._fwd_y = (0.0, 1.0) if bow_y else (1.0, 0.0)
    env._surge_axis_idx, env._sway_axis_idx = (1, 0) if bow_y else (0, 1)
    env.forward_vec = torch.tensor([env._fwd_x, env._fwd_y, 0.0]).expand(n, 3)
    env.up_dir = torch.tensor([0.0, 0.0, 1.0])
    env._sea = Sea() if legacy else None
    env.control_step_s = 1.0 / 60.0
    env.episode_length_buf = torch.arange(n)
    env._conditions = None
    env._obs_degrader = None
    env._applied_thrust_per_env = torch.zeros(n)
    env._applied_yaw_per_env = torch.zeros(n)
    env._compute_buoyancy_forces = lambda: (torch.ones((n, 3)) * 0.4, torch.ones((n, 3)) * 0.05)
    env._compute_current_forces = lambda: torch.ones((n, 3)) * 0.03
    env.hold_point = torch.ones((n, 2)) * 2.0
    env.target_pos = env.hold_point.clone()
    env.d0_per_env = torch.arange(1, n + 1).float()
    env._reached_goal = torch.zeros(n, dtype=torch.bool)
    env._ray_ranges_m = lambda: torch.arange(1, 9).float().expand(n, 8)
    return env


class EnvHookTests(unittest.TestCase):
    def test_actual_disabled_factory_equals_baseline(self):
        for family in FAMILIES:
            original, current = make_env(BASELINE, family), make_env(REPO, family)
            current.cfg.conditions = {
                "current": {"enabled": False, "speed_mps": 2.0},
                "actuator": {"enabled": False, "mass_scale": 2.0},
                "perception": {"enabled": False, "noise_pos_m": 2.0},
            }
            current._conditions = CONDITIONS.build_conditions(current.cfg, 5, "cpu", family=family)
            self.assertIsNone(current._conditions)
            for env in (original, current):
                env._pre_physics_step(torch.ones((5, 2)) * 0.375)
                env._apply_action()
            for a, b in zip(original.robot.calls[-1], current.robot.calls[-1]):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(original._get_observations()["policy"], current._get_observations()["policy"]))

    def test_actual_current_rotates_world_wrench_and_advances_physics_clock(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                original, current = make_env(BASELINE, family), make_env(REPO, family)
                current.cfg.conditions = {"current": {"enabled": True, "speed_mps": 0.5, "direction_rad": 0.0}}
                layer = current._conditions = CONDITIONS.build_conditions(current.cfg, 5, "cpu", family=family)
                layer.reset_runtime(env_ids=torch.arange(5))
                expected_world, _ = layer.current.wrench(current.robot.data.root_com_vel_w[:, :2], layer.elapsed_s)
                expected_body = quat_apply_inverse(current.robot.data.root_link_quat_w, expected_world)
                for env in (original, current):
                    env._pre_physics_step(torch.ones((5, 2)) * 0.375)
                    env._apply_action()
                    env._apply_action()
                self.assertTrue(torch.allclose(current.robot.calls[-1][0][:, 0, :] - original.robot.calls[-1][0][:, 0, :], expected_body, atol=3e-5, rtol=1e-5))
                self.assertTrue(torch.equal(current.robot.calls[-1][1], original.robot.calls[-1][1]))
                self.assertTrue(torch.all(layer.elapsed_s == 2.0 * current.cfg.sim.dt))

    def test_actual_actuator_matches_hazard_legacy_six_axes(self):
        original = make_env(BASELINE, "hazard_nav", legacy=True)
        original._sea = None
        original.physics_cfg.enable_current = False
        current = make_env(REPO, "hazard_nav")
        axes = ("thrust_cap_scale", "thrust_imbalance", "motor_tau_s", "drag_scale", "mass_scale", "payload_mass_kg", "payload_offset_x_m", "payload_offset_y_m")
        actuator = {key: getattr(original.cfg, key) for key in axes}
        actuator.update(enabled=True, reference_mass_kg=original.cfg.payload_ref_mass_kg, reference_izz_kgm2=original.cfg.payload_ref_izz_kgm2, half_beam_m=original.cfg.half_beam_m)
        current.cfg.conditions = {"actuator": actuator}
        current._conditions = CONDITIONS.build_conditions(current.cfg, 5, "cpu", family="hazard_nav")
        current._conditions.reset_runtime(env_ids=torch.arange(5))
        for tick in range(4):
            if tick % 2 == 0:
                actions = torch.tensor([[0.5, -0.4], [-0.8, 0.2], [0.1, 0.0], [0.0, -0.5], [1.0, 1.0]])
                original._pre_physics_step(actions)
                current._pre_physics_step(actions)
            original._apply_action()
            current._apply_action()
            for a, b in zip(original.robot.calls[-1], current.robot.calls[-1]):
                self.assertTrue(torch.equal(a, b))

    def test_actual_station_actuator_cap(self):
        original, current = make_env(BASELINE, "station_keeping"), make_env(REPO, "station_keeping")
        current.cfg.condition_presets = ("actuator.cap",)
        current._conditions = CONDITIONS.build_conditions(current.cfg, 5, "cpu", family="station_keeping")
        current._conditions.reset_runtime(env_ids=torch.arange(5))
        for env in (original, current):
            env._pre_physics_step(torch.ones((5, 2)) * 0.5)
            env._apply_action()
        delta = current.robot.calls[-1][0] - original.robot.calls[-1][0]
        expected = torch.zeros_like(delta)
        expected[:, 0, 0] = -20.0
        self.assertTrue(torch.allclose(delta, expected, atol=1e-5, rtol=0))
        self.assertTrue(torch.equal(current.robot.calls[-1][1], original.robot.calls[-1][1]))

    def test_actual_perception_matches_legacy_group_boundaries(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                original, current = make_env(BASELINE, family), make_env(REPO, family)
                current.cfg.conditions = {"perception": {"enabled": True, "noise_pos_m": 0.1, "noise_vel_mps": 0.05, "delay_steps": 2, "dropout_p": 0.2}}
                layer = current._conditions = CONDITIONS.build_conditions(current.cfg, 5, "cpu", family=family)
                # Independent instance of the unchanged legacy degrader with
                # matching raw-unit groups and stream keys.
                from _condition_cpu._shared.obs_degradation import build_obs_degrader
                original._obs_degrader = build_obs_degrader(num_envs=5, device="cpu", base_seed=current.cfg.seed, groups=layer.perception._groups, delay_steps=2, dropout_p=0.2)
                original._obs_degrader.reset(torch.arange(5))
                layer.reset_runtime(env_ids=torch.arange(5))
                for _ in range(4):
                    self.assertTrue(torch.equal(original._get_observations()["policy"], current._get_observations()["policy"]))
                    original.robot.data.root_pos_w += 0.05
                    current.robot.data.root_pos_w += 0.05

    def test_legacy_action_and_physics_bitwise(self):
        """Compare original methods, not a rewritten dynamics oracle."""
        for family in FAMILIES:
            for isotropic in (False, True):
                for six_velocity in (False, True):
                    for legacy in (False, True):
                        with self.subTest(family=family, isotropic=isotropic, six=six_velocity, legacy=legacy):
                            options = dict(isotropic=isotropic, six_velocity=six_velocity, bow_y=isotropic, legacy=legacy)
                            before = make_env(BASELINE, family, **options)
                            after = make_env(REPO, family, **options)
                            actions = torch.tensor([[-3.0, 2.0], [-0.25, 0.1], [0.0, 0.0], [0.125, -0.2], [4.0, -2.0]])
                            before._pre_physics_step(actions)
                            after._pre_physics_step(actions)
                            self.assertTrue(torch.equal(before.actions, after.actions))
                            for _ in range(4):
                                before._apply_action()
                                after._apply_action()
                                for expected, actual in zip(before.robot.calls[-1], after.robot.calls[-1]):
                                    self.assertTrue(torch.equal(expected, actual))

    def test_legacy_observation_bitwise(self):
        for family in FAMILIES:
            for kin in (False, True):
                for superset in (False, True):
                    with self.subTest(family=family, kin=kin, superset=superset):
                        before, after = make_env(BASELINE, family), make_env(REPO, family)
                        for env in (before, after):
                            env.cfg.obs_kinematic = kin
                            env.cfg.obs_v3 = kin
                            env.cfg.emit_superset_obs = superset
                            env.cfg.superset_version = 2 if kin else 1
                            if superset and family == "hazard_nav":
                                count = OBS.SLICES["rays"].stop - OBS.SLICES["rays"].start
                                env._ray_ranges_m = lambda: torch.arange(1, count + 1).float().expand(5, count)
                        self.assertTrue(torch.equal(before._get_observations()["policy"], after._get_observations()["policy"]))

    def test_hook_order_and_physics_rate(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                env = make_env(REPO, family)
                layer = env._conditions = RecordingConditions()
                env._pre_physics_step(torch.ones((env.num_envs, 2)) * 2.0)
                env._apply_action()
                env._apply_action()
                self.assertEqual(layer.events, ["action", "runtime"] + ["command", "drag", "drag", "wrench", "dynamics"] * 2)
                self.assertEqual(layer.dt, env.cfg.sim.dt)

    def test_perception_once_per_physical_group(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                env = make_env(REPO, family)
                clean = env._get_observations()["policy"]
                layer = env._conditions = RecordingConditions()
                degraded = env._get_observations()["policy"]
                expected = ["observation:pos", "observation:vel"]
                if family == "hazard_nav":
                    expected.append("observation:ray")
                self.assertEqual(layer.events, expected)
                self.assertFalse(torch.equal(clean, degraded))

    def test_reset_latches_stamp_before_one_layer_reset(self):
        for family in FAMILIES:
            _, cls = source_class(REPO, family)
            reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_idx")
            calls = [n for n in ast.walk(reset) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            stamps = [n for n in calls if n.func.attr == "episode_stamp"]
            resets = [n for n in calls if n.func.attr == "reset_runtime"]
            self.assertEqual(len(stamps), 1)
            self.assertEqual(len(resets), 1)
            self.assertLess(stamps[0].lineno, resets[0].lineno)
            # D1: the reset hook also passes the task reference axis (wind
            # directions) and the task scenario episode ids (cross-check).
            self.assertEqual([kw.arg for kw in resets[0].keywords],
                             ["env_ids", "reference_axis_rad", "task_episode_ids"])
            stamp_parent = next(n for n in ast.walk(reset) if isinstance(n, ast.For) and stamps[0] in list(ast.walk(n)))
            self.assertEqual(ast.unparse(stamp_parent.iter), "completed_ids.tolist()")

    def test_actual_completed_stamp_survives_async_reset(self):
        """Execute the actual latch loop and reset hook extracted from each env.

        Legacy geometry/PhysX reset statements are deliberately excluded;
        their simulator behavior is outside this CPU harness's claims.
        """
        for family in FAMILIES:
            with self.subTest(family=family):
                env = make_env(REPO, family)
                env.cfg.condition_presets = ("current.medium", "actuator.lag")
                layer = env._conditions = CONDITIONS.build_conditions(env.cfg, 5, "cpu", family=family)
                layer.reset_runtime(env_ids=torch.arange(5))
                env._pre_physics_step(torch.ones((5, 2)))
                env._apply_action()
                env.episode_conditions = [{} for _ in range(5)]
                env.episode_scenario = [{} for _ in range(5)]
                env.episode_scenario_hashes = [{} for _ in range(5)]
                env._scenario_params = [{} for _ in range(5)]
                env._scenario_hashes = [{} for _ in range(5)]
                _, cls = source_class(REPO, family)
                reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_idx")
                latch = next(n for n in ast.walk(reset) if isinstance(n, ast.For) and ast.unparse(n.iter) == "completed_ids.tolist()")
                reset_hook = next(n for n in reversed(reset.body) if isinstance(n, ast.If) and "self._conditions.reset_runtime" in ast.unparse(n))
                selected = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(latch), copy.deepcopy(reset_hook)], type_ignores=[]))
                ids = torch.tensor([1, 3])
                # D1: the reset hook reads the task reference axis inputs
                # computed earlier in _reset_idx; supply them for the 2 rows.
                env._scenario = None
                import math
                exec(compile(selected, "<actual env latch and reset>", "exec"), {
                    "self": env, "env_ids": ids, "completed_ids": ids, "torch": torch, "math": math,
                    "spawn_angles": torch.tensor([0.25, -1.0]),
                    "local_starts": torch.zeros((2, 2)), "local_goals": torch.tensor([[3.0, 4.0], [0.0, -2.0]]),
                })
                for i in (1, 3):
                    self.assertEqual(env.episode_conditions[i]["runtime"]["episode_index"], 0)
                    self.assertEqual(env.episode_conditions[i]["runtime"]["elapsed_physics_s"], env.cfg.sim.dt)
                    self.assertEqual(int(layer.episode_index[i]), 1)
                    self.assertEqual(float(layer.elapsed_s[i]), 0.0)
                    self.assertEqual(float(layer.actuator._applied_thrust[i]), 0.0)
                for i in (0, 2, 4):
                    self.assertEqual(env.episode_conditions[i], {})
                    self.assertEqual(int(layer.episode_index[i]), 0)
                    self.assertEqual(float(layer.elapsed_s[i]), env.cfg.sim.dt)
                    self.assertGreater(float(layer.actuator._applied_thrust[i]), 0.0)

    def test_no_rewards_or_terminations_rewired(self):
        for family in FAMILIES:
            _, original = source_class(BASELINE, family)
            _, current = source_class(REPO, family)
            for name in ("_get_rewards", "_get_dones"):
                a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
                b = copy.deepcopy(next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name))
                # D1: C5's termination-reason telemetry call is the only
                # permitted addition (checked by scripts/test_termination_v3.py
                # and scripts/test_jbase_integration.py); drop it here.
                b.body = [n for n in b.body if not (
                    isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "publish_termination_reason")]
                self.assertEqual(ast.dump(a, include_attributes=False), ast.dump(b, include_attributes=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
