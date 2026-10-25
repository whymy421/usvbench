"""D1 J-base semantic-reconciliation tests (CPU only, never imports Isaac Lab).

Run: python -B scripts/test_jbase_integration.py

Covers the D1 brief step 3 items and the journal configuration classes:
  a. waves: exactly one calm-increment subtraction and one per-env clock
  b. damping order and a single drag/mass source for the v2 implicit damping
  c. legacy current rejected in journal configurations
  e. wind: C3 backend through the condition layer (frames, cadence, keys,
     direction modes, dose-axis conversion, stamps, profiles)
  4. journal cfg classes (stubbed isaaclab/gymnasium, real cfg modules)
Item d (certificates) is in scripts/test_cert_v31_conditions.py.

Environment methods are the ACTUAL merged methods, extracted by AST from
tasks/<family>/<family>_env.py and executed on CPU fakes; they are tensor-
level checks, not simulator runs.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, replace
import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "tasks" / "_shared"
PACKAGE = "_d1_shared"
_parent = ModuleType(PACKAGE)
_parent.__path__ = [str(SHARED)]
sys.modules[PACKAGE] = _parent
_spec = importlib.util.spec_from_file_location(PACKAGE + ".conditions", SHARED / "conditions" / "__init__.py",
                                               submodule_search_locations=[str(SHARED / "conditions")])
api = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = api
_spec.loader.exec_module(api)
physics = sys.modules[PACKAGE + ".physics_v2"]
sea_state = sys.modules[PACKAGE + ".sea_state"]
scenario_rng = sys.modules[PACKAGE + ".scenario_rng"]
wind_backend = sys.modules[PACKAGE + ".wind"]
wind_condition = sys.modules[PACKAGE + ".conditions.wind"]
windage = sys.modules[PACKAGE + ".conditions.windage"]
restoring = importlib.import_module(PACKAGE + ".restoring")
DT = 1.0 / 120.0


# ----------------------------------------------------------------- helpers --
def quat_apply(q, v):
    qv = q[..., 1:]
    cross = torch.cross(qv, v, dim=-1)
    return v + 2.0 * (q[..., :1] * cross + torch.cross(qv, cross, dim=-1))


def quat_apply_inverse(q, v):
    qv = q[..., 1:]
    cross = torch.cross(qv, v, dim=-1)
    return v - 2.0 * q[..., :1] * cross + 2.0 * torch.cross(qv, cross, dim=-1)


def layer(overrides=None, n=3, seed=42, presets=(), physics_version=2, vehicle="blueboat",
          reset=True, **kw):
    cfg = api.resolve(presets, overrides, seed=seed)
    result = api.ConditionLayer(cfg, n, "cpu", context={"vehicle": vehicle},
                                physics_version=physics_version, physics_dt=DT, **kw)
    if reset:
        result.reset_runtime(env_ids=torch.arange(n), reference_axis_rad=torch.zeros(n))
    return result


def env_class(family):
    path = REPO / "tasks" / family / (family + "_env.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    name = "StationKeepingEnv" if family == "station_keeping" else "HazardNavEnv"
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    wanted = {"_pre_physics_step", "_apply_action", "_root_quat"}
    cls = ast.ClassDef(name="Extracted", bases=[], keywords=[], decorator_list=[],
                       body=[copy.deepcopy(n) for n in original.body
                             if isinstance(n, ast.FunctionDef) and n.name in wanted])
    namespace = {"torch": torch, "math": math,
                 "math_utils": SimpleNamespace(quat_apply=quat_apply, quat_apply_inverse=quat_apply_inverse),
                 "restoring_torque_body": restoring.restoring_torque_body,
                 "add_implicit_hull_damping": physics.add_implicit_hull_damping}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"),
         namespace)
    return namespace["Extracted"], namespace


class Robot:
    def __init__(self, data):
        self.num_bodies = 1
        self.data = data
        self.calls = []

    def set_external_force_and_torque(self, forces, torques):
        self.calls.append((forces.clone(), torques.clone()))


def make_env(family, *, physics_version=2, bow_y=False, n=5, seed=984):
    cls, namespace = env_class(family)
    env = cls()
    generator = torch.Generator().manual_seed(seed)
    quaternion = torch.randn((n, 4), generator=generator)
    quaternion /= torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    data = SimpleNamespace(
        root_link_quat_w=quaternion, root_quat_w=quaternion,
        root_com_vel_w=torch.randn((n, 6), generator=generator),
        root_ang_vel_w=torch.randn((n, 3), generator=generator),
        root_pos_w=torch.randn((n, 3), generator=generator),
        default_mass=torch.full((n, 1), 17.26),
        default_inertia=torch.diag(torch.tensor([1.789, 1.094, 2.376])).repeat(n, 1, 1).reshape(n, 9),
    )
    data.root_com_pos_w = data.root_pos_w
    env.robot = Robot(data)
    env.num_envs, env.device = n, "cpu"
    env.cfg = SimpleNamespace(
        thrust_max_fwd=80.0, thrust_max_rev=48.0, yaw_torque_max=23.0,
        sim=SimpleNamespace(dt=DT, gravity=(0.0, 0.0, -9.81)), half_beam_m=0.45,
        thrust_cap_scale=1.0, thrust_imbalance=0.0, motor_tau_s=0.0,
        drag_scale=1.0, mass_scale=1.0, payload_mass_kg=0.0,
        payload_offset_x_m=0.0, payload_offset_y_m=0.0,
        payload_ref_mass_kg=17.26, payload_ref_izz_kgm2=2.376,
        ray_count=8, ray_max_range_m=30.0, decimation=2, seed=47, vehicle="blueboat",
        condition_presets=(), conditions=None, journal_mode=True, obs_kinematic=True,
    )
    for axis in ("thrust_cap_scale", "thrust_imbalance", "motor_tau_s", "drag_scale", "mass_scale", "payload_mass_kg"):
        setattr(env.cfg, axis + "_choices", ())
    env.physics_cfg = SimpleNamespace(
        sway_lin_damping=27.0, sway_quad_damping=18.0, surge_lin_damping=9.1, surge_quad_damping=5.9,
        heave_damping=200.0, yaw_lin_damping=4.0, yaw_quad_damping=6.0, rollpitch_rate_damping=470.0,
        restoring_stiffness_roll=280.0, restoring_stiffness_pitch=141.0, enable_current=False,
        physics_version=physics_version,
    )
    env.cfg.underwater_physics_cfg = env.physics_cfg
    env._fwd_x, env._fwd_y = (0.0, 1.0) if bow_y else (1.0, 0.0)
    env._surge_axis_idx, env._sway_axis_idx = (1, 0) if bow_y else (0, 1)
    env.forward_vec = torch.tensor([env._fwd_x, env._fwd_y, 0.0]).expand(n, 3)
    env.up_dir = torch.tensor([0.0, 0.0, 1.0])
    env._sea = None
    env.control_step_s = DT * 2
    env.episode_length_buf = torch.zeros(n, dtype=torch.long)
    env._conditions = None
    env._compute_buoyancy_forces = lambda: (torch.ones((n, 3)) * 0.4, torch.ones((n, 3)) * 0.05)

    def legacy_current_called():
        raise AssertionError("legacy current path executed")
    env._compute_current_forces = legacy_current_called
    return env, namespace


def attach_layer(env, family, overrides=None, presets=()):
    env.cfg.conditions = overrides
    env.cfg.condition_presets = presets
    env._conditions = api.build_conditions(env.cfg, env.num_envs, "cpu", family=family)
    env._conditions.reset_runtime(env_ids=torch.arange(env.num_envs),
                                  reference_axis_rad=torch.zeros(env.num_envs))
    return env._conditions


FAMILIES = ("station_keeping", "hazard_nav")


# ============================================================== a. waves ====
class WaveTests(unittest.TestCase):
    WAVES = {"waves": {"enabled": True, "hs_range": (0.2, 0.3), "tp_range": (2.0, 3.0)}}

    def reference_sea(self, cfg, n, seed):
        kwargs = asdict(cfg.waves)
        kwargs["enable"] = kwargs.pop("enabled")
        kwargs["calm_water_increment"] = True
        with torch.random.fork_rng(devices=[]):
            sea = sea_state.SeaState(sea_state.SeaStateCfg(**kwargs), n, torch.device("cpu"), physics_version=2)
        scenario = scenario_rng.ScenarioRNG(num_envs=n, device="cpu", eval_seed=seed)
        scenario.reset_idx(torch.arange(n))
        sea.resample(torch.arange(n), scenario)
        return sea

    def test_v2_force_equals_c2_sea_state_single_subtraction(self):
        n, seed = 4, 11
        wave = layer(self.WAVES, n=n, seed=seed)
        self.assertEqual(wave.waves.sea.physics_version, 2)
        self.assertTrue(wave.waves.sea.cfg.calm_water_increment)
        ref = self.reference_sea(wave.cfg, n, seed)
        for name in ("hs", "tp", "gamma", "mean_direction", "amplitude", "phase", "direction", "freqs"):
            self.assertTrue(torch.equal(getattr(ref, name), getattr(wave.waves.sea, name)), name)
        clock = physics.WaveClock(n, "cpu", DT)
        generator = torch.Generator().manual_seed(5)
        for step in range(6):
            xy = torch.randn((n, 2), generator=generator) * 3
            v = torch.randn((n, 2), generator=generator)
            got_f, got_t = wave.environmental_wrench(xy, v, DT)
            ref_f, ref_t = ref.forces(xy, v, clock.time())
            clock.advance()
            self.assertTrue(torch.equal(got_f, ref_f), step)
            self.assertTrue(torch.equal(got_t, ref_t), step)
            self.assertGreater(float(got_f.abs().sum()), 0.0)
        # A second subtraction would differ from the C2 force by exactly the
        # calm drag D(-v) = -8 v|v|; show that term is nonzero here.
        calm = -8.0 * v * torch.norm(v, dim=-1, keepdim=True)
        self.assertGreater(float(calm.abs().sum()), 0.0)

    def test_hs_zero_gives_exact_zero_wrench(self):
        v = torch.tensor([[1.5, -0.7], [0.0, 2.0], [-3.0, 0.1]])
        xy = torch.randn(3, 2)
        bypass = layer({"waves": {"enabled": True, "hs_range": (0.0, 0.0), "tp_range": (2.0, 2.0)}})
        self.assertIsNone(bypass.waves.sea)
        f, t = bypass.environmental_wrench(xy, v, DT)
        self.assertEqual(torch.count_nonzero(f).item() + torch.count_nonzero(t).item(), 0)
        # A realized Hs=0 inside the v2 SeaState path (not the bypass).
        wave = layer(self.WAVES)
        wave.waves.sea.hs.zero_()
        wave.waves.sea.amplitude.zero_()
        for _ in range(3):
            f, t = wave.environmental_wrench(xy, v, DT)
            self.assertEqual(torch.count_nonzero(f).item() + torch.count_nonzero(t).item(), 0)

    def test_other_env_reset_leaves_ongoing_env_untouched(self):
        n = 3
        a, b = layer(self.WAVES, n=n), layer(self.WAVES, n=n)
        generator = torch.Generator().manual_seed(9)
        for step in range(9):
            xy = torch.randn((n, 2), generator=generator)
            v = torch.randn((n, 2), generator=generator)
            if step == 4:
                before = b.waves.sea.phase[1].clone()
                b.reset_runtime(env_ids=torch.tensor([1]), reference_axis_rad=torch.zeros(1))
                self.assertFalse(torch.equal(before, b.waves.sea.phase[1]))
                self.assertEqual(int(b.clock.ticks[1]), 0)
            fa, ta = a.environmental_wrench(xy, v, DT)
            fb, tb = b.environmental_wrench(xy, v, DT)
            for i in (0, 2):
                self.assertTrue(torch.equal(fa[i], fb[i]) and torch.equal(ta[i], tb[i]), (step, i))
        self.assertEqual(b.clock.ticks.tolist(), [9, 5, 9])
        self.assertEqual(a.clock.ticks.tolist(), [9, 9, 9])

    def test_journal_env_has_one_clock_advanced_once_per_substep(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                env, _ = make_env(family)
                layer_ = attach_layer(env, family, self.WAVES)
                self.assertFalse(hasattr(env, "_wave_clock"))
                env._pre_physics_step(torch.full((5, 2), 0.3))
                for tick in range(1, 4):
                    env._apply_action()
                    self.assertEqual(layer_.clock.ticks.tolist(), [tick] * 5)
                self.assertEqual(layer_.episode_stamp(0)["runtime"]["clock_ticks"], 3)
        # The factory refuses a legacy (env-owned) sea state, hence a second clock.
        env, _ = make_env("hazard_nav")
        env.cfg.sea_state = SimpleNamespace(enable=True)
        env.cfg.conditions = self.WAVES
        with self.assertRaises(ValueError):
            api.build_conditions(env.cfg, 5, "cpu", family="hazard_nav")

    def test_v1_prototype_path_keeps_adapter_subtraction(self):
        wave = layer(self.WAVES, physics_version=1)
        self.assertEqual(wave.waves.sea.physics_version, 1)
        xy, v = torch.randn(3, 2), torch.randn(3, 2)
        f, _ = wave.environmental_wrench(xy, v, DT)
        raw, _ = wave.waves.sea.forces(xy, v, torch.zeros(3))
        calm = -8.0 * v * torch.norm(v, dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(f[:, :2], raw[:, :2] - calm, atol=1e-6))


# ============================================================ b. damping ====
class DampingTests(unittest.TestCase):
    def fake(self, n=4):
        env, _ = make_env("hazard_nav", n=n)
        env.robot.data.root_com_vel_w = torch.tensor([[1.0, -0.5, 0.2, 0.3, -0.1, 0.8]] * n)
        return env

    def test_drag_scale_multiplies_v2_coefficients_exactly_once(self):
        env = self.fake()
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
        base = layer({"actuator": {"enabled": True, "drag_scale": 1.0}}, n=4)
        scaled = layer({"actuator": {"enabled": True, "drag_scale": 1.5}}, n=4)
        t1 = physics.hull_damping_terms(env, quat, base.hull_damping_scales())
        t15 = physics.hull_damping_terms(env, quat, scaled.hull_damping_scales())
        t_legacy = physics.hull_damping_terms(env, quat, None)
        for key in ("linear_damping", "angular_damping"):
            self.assertTrue(torch.equal(t1[key], t_legacy[key]), key)
        for i, j in ((0, 0), (1, 1)):
            self.assertTrue(torch.equal(t15["linear_damping"][:, i, j], 1.5 * t1["linear_damping"][:, i, j]))
        self.assertTrue(torch.equal(t15["angular_damping"][:, 2, 2], 1.5 * t1["angular_damping"][:, 2, 2]))
        # Heave and roll/pitch are not hull drag: unchanged.
        self.assertTrue(torch.equal(t15["linear_damping"][:, 2, 2], t1["linear_damping"][:, 2, 2]))
        self.assertTrue(torch.equal(t15["angular_damping"][:, :2, :2], t1["angular_damping"][:, :2, :2]))

    def test_divisors_reproduce_condition_dynamics_factors(self):
        cfg = {"actuator": {"enabled": True, "mass_scale": 1.3, "payload_mass_kg": 4.0,
                            "payload_offset_x_m": 0.2, "payload_offset_y_m": -0.1}}
        lay = layer(cfg, n=2)
        scales = lay.hull_damping_scales()
        force = torch.zeros(2, 1, 3)
        torque = torch.zeros(2, 1, 3)
        force[:, 0, 0] = 1.0
        f, _ = lay.dynamics(force, torque)
        self.assertTrue(torch.allclose(f[:, 0, 0], 1.0 / scales["mass_divisor"], rtol=1e-6))
        torque = torch.zeros(2, 1, 3)
        torque[:, 0, 2] = 1.0
        _, t = lay.dynamics(torch.zeros(2, 1, 3), torque)
        self.assertTrue(torch.allclose(t[:, 0, 2], 1.0 / scales["yaw_divisor"], rtol=1e-6))
        torque = torch.zeros(2, 1, 3)
        torque[:, 0, 0] = 1.0
        _, t = lay.dynamics(torch.zeros(2, 1, 3), torque)
        self.assertTrue(torch.allclose(t[:, 0, 0], 1.0 / scales["rollpitch_divisor"], rtol=1e-6))

    def test_env_v2_order_dynamics_then_damping_single_drag_source(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                env, namespace = make_env(family)
                lay = attach_layer(env, family, {"actuator": {"enabled": True, "drag_scale": 1.5, "mass_scale": 1.3}})
                events, seen = [], {}
                original_dynamics = lay.dynamics
                lay.scale_drag = lambda value: events.append("scale_drag") or value

                def dynamics(forces, torques):
                    events.append("dynamics")
                    result = original_dynamics(forces, torques)
                    seen["after_dynamics"] = (result[0].clone(), result[1].clone())
                    return result
                lay.dynamics = dynamics

                def damping(env_, forces, torques, quat, scales=None):
                    events.append("damping")
                    self.assertTrue(torch.equal(forces, seen["after_dynamics"][0]))
                    self.assertTrue(torch.equal(torques, seen["after_dynamics"][1]))
                    for key, value in lay.hull_damping_scales().items():
                        self.assertTrue(torch.equal(scales[key], value), key)
                    seen["other"] = (forces.clone(), torques.clone(), quat, scales)
                    physics.add_implicit_hull_damping(env_, forces, torques, quat, scales)
                namespace["add_implicit_hull_damping"] = damping
                env._pre_physics_step(torch.full((5, 2), 0.4))
                env._apply_action()
                self.assertEqual(events, ["dynamics", "damping"])
                other_f, other_t, quat, scales = seen["other"]
                expected_f, expected_t = other_f.clone(), other_t.clone()
                physics.add_implicit_hull_damping(env, expected_f, expected_t, quat, scales)
                applied_f, applied_t = env.robot.calls[-1]
                self.assertTrue(torch.equal(applied_f, expected_f))
                self.assertTrue(torch.equal(applied_t, expected_t))

    def test_env_v1_scales_drag_explicitly_and_never_calls_implicit(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                env, namespace = make_env(family, physics_version=1)
                env.cfg.journal_mode = False
                lay = attach_layer(env, family, {"actuator": {"enabled": True, "drag_scale": 1.5}})
                calls = []
                original = lay.scale_drag
                lay.scale_drag = lambda value: calls.append(1) or original(value)
                namespace["add_implicit_hull_damping"] = lambda *a, **k: self.fail("implicit damping in v1")
                env._pre_physics_step(torch.full((5, 2), 0.4))
                env._apply_action()
                self.assertEqual(len(calls), 2)  # planar (surge+sway) block and yaw

    def test_scales_and_legacy_axes_are_mutually_exclusive(self):
        env = self.fake()
        env.cfg.drag_scale = 1.2
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
        with self.assertRaises(ValueError):
            physics.hull_damping_terms(env, quat, layer(n=4).hull_damping_scales())

    def test_journal_calm_layer_equals_plain_v2_bitwise(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                plain, _ = make_env(family)
                journal, _ = make_env(family)
                lay = attach_layer(journal, family, None)
                self.assertFalse(lay.cfg.any_enabled)
                state = torch.random.get_rng_state()
                for env in (plain, journal):
                    env._pre_physics_step(torch.tensor([[0.5, -0.4], [-0.8, 0.2], [0.1, 0.0], [0.0, -0.5], [1.0, 1.0]]))
                    for _ in range(3):
                        env._apply_action()
                for a, b in zip(plain.robot.calls, journal.robot.calls):
                    self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


# ============================================================ c. current ====
class CurrentTests(unittest.TestCase):
    def journal_cfg(self, **kwargs):
        env, _ = make_env("station_keeping")
        for key, value in kwargs.items():
            setattr(env.cfg, key, value)
        return env.cfg

    def test_journal_factory_rejects_legacy_current_even_all_off(self):
        cases = []
        cfg = self.journal_cfg()
        cfg.underwater_physics_cfg.enable_current = True
        cases.append(cfg)
        cases.append(self.journal_cfg(current_speed_override_mps=0.5))
        cases.append(self.journal_cfg(current_ramp_start_mps=0.5, current_ramp_end_mps=3.0))
        cfg = self.journal_cfg(conditions={"current": {"enabled": True, "speed_mps": 0.5}})
        cfg.underwater_physics_cfg.enable_current = True
        cases.append(cfg)
        cases.append(self.journal_cfg(conditions={"current": {"enabled": True, "force_mode": "legacy_relative_drag"}}))
        cfg = self.journal_cfg()
        cfg.underwater_physics_cfg.physics_version = 1
        cases.append(cfg)
        for cfg in cases:
            with self.assertRaises(ValueError):
                api.build_conditions(cfg, 5, "cpu", family="station_keeping")
        with self.assertRaises(ValueError):
            api.build_conditions(self.journal_cfg(), 5, "cpu", family="docking")
        legacy = self.journal_cfg(journal_mode=False)
        legacy.underwater_physics_cfg.enable_current = True
        self.assertIsNone(api.build_conditions(legacy, 5, "cpu", family="station_keeping"))
        calm = api.build_conditions(self.journal_cfg(), 5, "cpu", family="station_keeping")
        self.assertIsNotNone(calm)
        self.assertFalse(calm.cfg.any_enabled)

    def test_journal_current_is_incremental_and_legacy_path_never_runs(self):
        env, _ = make_env("station_keeping")
        lay = attach_layer(env, "station_keeping", {"current": {"enabled": True, "speed_mps": 0.5, "direction_rad": 0.3}})
        velocity = torch.tensor([[1.0, 0.0], [0.0, -2.0], [0.5, 0.5], [0.0, 0.0], [-1.0, 0.2]])
        u = 0.5 * torch.tensor([math.cos(0.3), math.sin(0.3)])
        drag = lambda x: -8.0 * x * torch.norm(x, dim=-1, keepdim=True)
        f, _ = lay.current.wrench(velocity, lay.elapsed_s)
        self.assertTrue(torch.allclose(f[:, :2], drag(velocity - u) - drag(velocity), atol=1e-6))
        env._pre_physics_step(torch.zeros(5, 2))
        env._apply_action()  # _compute_current_forces raises if called
        zero = attach_layer(make_env("station_keeping")[0], "station_keeping",
                            {"current": {"enabled": True, "speed_mps": 0.0}})
        f, _ = zero.current.wrench(velocity, zero.elapsed_s)
        self.assertEqual(torch.count_nonzero(f).item(), 0)


# =============================================================== e. wind ====
class WindTests(unittest.TestCase):
    BASE = {"enabled": True, "speed_mps": 6.0, "reference": "vessel_height", "direction_rad": 0.3,
            "gust_intensity": 0.15, "gust_correlation_time_s": 2.0}

    def test_world_wrench_equals_c3_backend_once_per_substep(self):
        n, seed = 3, 7
        lay = layer({"wind": dict(self.BASE)}, n=n, seed=seed)
        profile = windage.get_profile("blueboat_hull_only_c3")
        ref = wind_backend.Wind(wind_backend.WindCfg(
            enable=True, mean_speed_mps=6.0, reference_height_m=profile.centroid_height_m,
            direction_rad=0.3, gust_intensity=0.15, gust_correlation_time_s=2.0,
            windage=wind_backend.VesselWindage(
                profile.frontal_area_m2, profile.lateral_area_m2, profile.length_m,
                profile.centroid_height_m, profile.longitudinal_cp_m, 0.9, 0.9, 0.1)),
            n, "cpu", eval_seed=seed)
        ref.reset(None, [0] * n)
        generator = torch.Generator().manual_seed(3)
        for step in range(7):
            v = torch.randn((n, 2), generator=generator)
            heading = torch.rand(n, generator=generator) * 6.28
            got_f, got_t = lay.environmental_wrench(torch.zeros(n, 2), v, DT, heading_rad=heading)
            ref_f, ref_t = ref.forces(v, heading)
            ref.step(DT)
            self.assertTrue(torch.equal(got_f, ref_f) and torch.equal(got_t, ref_t), step)
            self.assertEqual(lay.wind.backend._physics_steps, [step + 1] * n)
            self.assertEqual(lay.clock.ticks.tolist(), [step + 1] * n)
        self.assertEqual(float(got_f[:, 2].abs().max()), 0.0)  # planar model: no vertical force

    def test_u10_log_profile_conversion_and_stamps(self):
        lay = layer({"wind": {"enabled": True, "speed_mps": 10.0, "roughness_length_m": 0.0002}})
        z = windage.get_profile("blueboat_hull_only_c3").centroid_height_m
        expected = 10.0 * math.log(z / 0.0002) / math.log(10.0 / 0.0002)
        self.assertEqual(lay.wind.applied_speed_mps, expected)
        self.assertEqual(lay.wind.backend.cfg.mean_speed_mps, expected)
        model = lay.certificate_stamp()["context"]["wind_model"]
        self.assertEqual(model["requested_speed_mps"], 10.0)
        self.assertEqual(model["requested_reference"], "u10")
        self.assertEqual(model["applied_vessel_height_speed_mps"], expected)
        self.assertEqual(model["windage_profile"]["status"], "provisional")
        episode = lay.episode_stamp(0)["runtime"]["wind_realization"]
        self.assertEqual(episode["requested_speed_mps"], 10.0)
        self.assertEqual(episode["applied_vessel_height_speed_mps"], expected)
        direct = layer({"wind": {"enabled": True, "speed_mps": 10.0, "reference": "vessel_height"}})
        self.assertEqual(direct.wind.applied_speed_mps, 10.0)
        with self.assertRaises(ValueError):
            layer({"wind": {"enabled": True, "speed_mps": 10.0, "roughness_length_m": 0.5}})
        with self.assertRaises(ValueError):
            api.resolve((), {"wind": {"reference": "u2"}})

    def test_direction_modes_keyed_balanced_and_relative(self):
        axis = torch.tensor([0.5, -1.0, 2.0, 0.0])
        fixed = layer({"wind": {"enabled": True, "speed_mps": 3.0, "direction_rad": 1.1}}, n=4, reset=False)
        fixed.reset_runtime(env_ids=torch.arange(4), reference_axis_rad=axis)
        self.assertTrue(torch.equal(fixed.wind.backend.direction_rad, torch.full((4,), 1.1, dtype=torch.float64)))
        uniform_cfg = {"wind": {"enabled": True, "speed_mps": 3.0, "direction_mode": "uniform"}}
        a = layer(uniform_cfg, n=4, reset=False)
        a.reset_runtime(env_ids=torch.arange(4), reference_axis_rad=axis)
        b = layer(uniform_cfg, n=4, reset=False)
        b.reset_runtime(env_ids=torch.tensor([2]), reference_axis_rad=axis[2:3])
        b.reset_runtime(env_ids=torch.tensor([3, 0, 1]), reference_axis_rad=axis[[3, 0, 1]])
        self.assertTrue(torch.equal(a.wind.backend.direction_rad, b.wind.backend.direction_rad))
        self.assertTrue(torch.allclose(a.wind.direction_world_rad - axis.double(), a.wind.direction_relative_rad))
        first = a.wind.direction_relative_rad.clone()
        a.reset_runtime(env_ids=torch.tensor([0]), reference_axis_rad=axis[:1])
        self.assertNotEqual(float(first[0]), float(a.wind.direction_relative_rad[0]))
        with self.assertRaises(ValueError):
            a.reset_runtime(env_ids=torch.tensor([0]))  # relative mode needs the task axis
        n = 8
        balanced = layer({"wind": {"enabled": True, "speed_mps": 3.0, "direction_mode": "balanced"}},
                         n=n, seed=5, reset=False)
        choices = torch.tensor(balanced.cfg.wind.balanced_directions_rad, dtype=torch.float64)
        for episode in range(3):
            balanced.reset_runtime(env_ids=torch.arange(n), reference_axis_rad=torch.zeros(n))
            used = sorted(balanced.wind.direction_relative_rad.tolist())
            self.assertEqual(used, sorted(choices.tolist()), episode)
            self.assertEqual(int(balanced.wind.episode_index[0]), episode)

    def test_env_uses_forward_vec_heading_and_com_velocity(self):
        for family in FAMILIES:
            for bow_y in (False, True):
                with self.subTest(family=family, bow_y=bow_y):
                    env, _ = make_env(family, physics_version=1, bow_y=bow_y)
                    env.cfg.journal_mode = False
                    lay = attach_layer(env, family, {"wind": dict(self.BASE)})
                    seen = {}
                    original = lay.wind.wrench

                    def spy(position, velocity, elapsed=None, **kw):
                        seen.update(velocity=velocity.clone(), heading=kw["heading_rad"].clone())
                        return original(position, velocity, elapsed, **kw)
                    lay.wind.wrench = spy
                    env._pre_physics_step(torch.zeros(5, 2))
                    env._apply_action()
                    quat = env.robot.data.root_link_quat_w
                    bow = quat_apply(quat, env.forward_vec)
                    self.assertTrue(torch.equal(seen["heading"], torch.atan2(bow[:, 1], bow[:, 0])))
                    self.assertTrue(torch.equal(seen["velocity"], env.robot.data.root_com_vel_w[:, :2]))
                    # World wrench enters once, rotated once, before dynamics.
                    calm, _ = make_env(family, physics_version=1, bow_y=bow_y)
                    calm._pre_physics_step(torch.zeros(5, 2))
                    calm._apply_action()
                    replay = layer({"wind": dict(self.BASE)}, n=5, seed=47)
                    wind_f, wind_t = replay.wind.wrench(None, env.robot.data.root_com_vel_w[:, :2],
                                                        heading_rad=seen["heading"])
                    delta_f = env.robot.calls[-1][0][:, 0] - calm.robot.calls[-1][0][:, 0]
                    delta_t = env.robot.calls[-1][1][:, 0] - calm.robot.calls[-1][1][:, 0]
                    # float32 world accumulators hold O(100 N) heave drag, so
                    # the difference carries ~1e-4 N rounding; a frame or sign
                    # error would be O(1 N).
                    self.assertGreater(float(wind_f.abs().max()), 0.2)
                    self.assertTrue(torch.allclose(delta_f, quat_apply_inverse(quat, wind_f), atol=1e-3))
                    self.assertTrue(torch.allclose(delta_t, quat_apply_inverse(quat, wind_t), atol=1e-3))

    def test_disabled_is_exact_zero_without_draws(self):
        lay = layer({"current": {"enabled": True, "speed_mps": 0.0}})
        self.assertIsNone(lay.wind.backend)
        state = torch.random.get_rng_state()
        f, t = lay.wind.wrench(None, torch.ones(3, 2), heading_rad=torch.zeros(3))
        self.assertEqual(torch.count_nonzero(f).item() + torch.count_nonzero(t).item(), 0)
        lay.wind.step(DT)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(lay.episode_stamp(0)["runtime"]["wind_realization"], {})

    def test_stream_key_is_scenario_episode_and_task_counter_is_checked(self):
        lay = layer({"wind": dict(self.BASE)}, n=2)
        lay.reset_runtime(env_ids=torch.tensor([0]), reference_axis_rad=torch.zeros(1),
                          task_episode_ids=torch.tensor([1]))
        self.assertEqual(lay.wind.backend._episode_ids, [1, 0])
        self.assertEqual(lay.scenario.episode_indices().tolist(), [1, 0])
        stamp = lay.episode_stamp(0)["runtime"]["wind_realization"]
        self.assertEqual(stamp["stream_key"], {"eval_seed": 42, "env_key": 0, "episode_index": 1})
        with self.assertRaises(ValueError):
            lay.reset_runtime(env_ids=torch.tensor([1]), reference_axis_rad=torch.zeros(1),
                              task_episode_ids=torch.tensor([5]))

    def test_profiles_placeholder_and_vehicle_guards(self):
        self.assertEqual(windage.get_profile("wamv_vrx_placeholder_d5").status, "placeholder_pending_D5")
        with self.assertRaises(ValueError):
            layer({"wind": dict(self.BASE)}, vehicle="wamv_vrx")
        with self.assertRaises(ValueError):
            layer({"wind": dict(self.BASE, windage_profile="blueboat_hull_only_c3")}, vehicle="wamv_vrx")
        with self.assertRaises(ValueError):
            layer({"wind": dict(self.BASE)}, vehicle="wamv")
        bb = windage.get_profile("blueboat_hull_only_c3")
        self.assertEqual((bb.frontal_area_m2, bb.lateral_area_m2, bb.centroid_height_m),
                         (0.04968478224291942, 0.16220798892378677, 0.07621178065812223))
        lay = layer({"wind": dict(self.BASE)})
        self.assertEqual(lay.cfg.wind.windage_profile, "blueboat_hull_only_c3")
        self.assertEqual(api.config_from_stamp({"conditions": lay.certificate_stamp()}), lay.cfg)


# ================================================== 4. journal cfg classes ====
def _stub_isaaclab():
    """Minimal stand-ins so the REAL cfg modules can be executed on CPU."""
    class Cfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class WithState(Cfg):
        InitialStateCfg = Cfg

    def configclass(cls):
        def __init__(self, **kwargs):
            for klass in reversed(cls.__mro__):
                for name, value in vars(klass).items():
                    if name.startswith("__") or isinstance(value, (staticmethod, classmethod, property)):
                        continue
                    if callable(value) and not isinstance(value, type):
                        continue
                    setattr(self, name, copy.deepcopy(value))
            for name, value in kwargs.items():
                setattr(self, name, value)
            post = getattr(self, "__post_init__", None)
            if post is not None:
                post()
        cls.__init__ = __init__
        return cls

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__getattr__ = lambda attr: Cfg
        sys.modules[name] = mod
        return mod

    root = module("isaaclab")
    root.sim = module("isaaclab.sim", SimulationCfg=Cfg)
    module("isaaclab.assets", ArticulationCfg=WithState, RigidObjectCfg=WithState)
    module("isaaclab.envs", DirectRLEnvCfg=type("DirectRLEnvCfg", (), {}), ViewerCfg=Cfg)
    module("isaaclab.scene", InteractiveSceneCfg=Cfg)
    module("isaaclab.utils", configclass=configclass)
    gym = module("gymnasium")
    gym.spaces = SimpleNamespace(Box=Cfg)


def load_cfg_module(family, filename):
    _stub_isaaclab()
    for name, folder in (("_d1_tasks", REPO / "tasks"), ("_d1_tasks." + family, REPO / "tasks" / family)):
        package = ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    shared = ModuleType("_d1_tasks._shared")
    shared.__path__ = [str(SHARED)]
    sys.modules["_d1_tasks._shared"] = shared
    return importlib.import_module("_d1_tasks." + family + "." + filename)


class JournalCfgTests(unittest.TestCase):
    def setUp(self):
        # E1: these tests install stub isaaclab/gymnasium modules; restore
        # sys.modules afterwards so later tests in the same process (D8's
        # test_protocol_v3plus/test_run_cell_v3plus "no isaaclab imported"
        # checks) do not see a stub simulator.
        saved = dict(sys.modules)

        def restore():
            for name in [name for name in sys.modules if name not in saved]:
                del sys.modules[name]
            sys.modules.update(saved)
        self.addCleanup(restore)

    CASES = (
        ("station_keeping", "station_keeping_env_cfg", "StationKeepingBlueBoatJBaseCalmEnvCfg",
         "StationKeepingBlueBoatKinEnvCfg"),
        ("hazard_nav", "hazard_nav_env_cfg", "HazardNavV3JBaseCalmEnvCfg", "HazardNavV3EnvCfg"),
        ("hazard_nav", "hazard_nav_env_cfg", "HazardForcedCrossingJBaseCalmEnvCfg", "HazardForcedCrossingEnvCfg"),
    )

    def test_journal_cfgs_are_calm_v2_with_unchanged_observation_contract(self):
        for family, filename, name, parent_name in self.CASES:
            with self.subTest(cfg=name):
                module = load_cfg_module(family, filename)
                cls, parent_cls = getattr(module, name), getattr(module, parent_name)
                self.assertIn(parent_cls, cls.__mro__)
                cfg, parent = cls(), parent_cls()
                self.assertEqual(cfg.underwater_physics_cfg.physics_version, 2)
                self.assertEqual(parent.underwater_physics_cfg.physics_version, 1)
                self.assertTrue(cfg.journal_mode)
                self.assertFalse(getattr(parent, "journal_mode", False))
                self.assertIsNone(cfg.conditions)
                self.assertEqual(tuple(cfg.condition_presets), ())
                self.assertEqual(cfg.observation_space, parent.observation_space)
                self.assertEqual(cfg.vehicle, "blueboat")
                self.assertFalse(getattr(cfg, "obs_sea_state", False))
                self.assertFalse(getattr(getattr(cfg, "sea_state", None), "enable", False))
                self.assertFalse(getattr(cfg.underwater_physics_cfg, "enable_current", False))
                observed = [key for key in vars(cfg) if key.startswith("obs_")
                            and any(word in key for word in ("sea", "wave", "current", "wind")) and getattr(cfg, key)]
                self.assertEqual(observed, [])
                # Only the physics version and the journal flag differ from the parent.
                a = {k: v for k, v in vars(cfg).items() if k not in ("underwater_physics_cfg", "journal_mode")}
                b = {k: v for k, v in vars(parent).items() if k != "underwater_physics_cfg"}
                self.assertEqual(sorted(a), sorted(b))
                lay = api.build_conditions(cfg, 4, "cpu", family=family)
                self.assertIsNotNone(lay)
                self.assertFalse(lay.cfg.any_enabled)
                self.assertEqual(lay.context["physics_version"], 2)
                self.assertTrue(lay.context["journal_mode"])
                state = torch.random.get_rng_state()
                lay.reset_runtime(env_ids=torch.arange(4), reference_axis_rad=torch.zeros(4))
                f, t = lay.environmental_wrench(torch.zeros(4, 2), torch.ones(4, 2), cfg.sim.dt)
                self.assertEqual(torch.count_nonzero(f).item() + torch.count_nonzero(t).item(), 0)
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                cfg.conditions = {"wind": {"enabled": True, "speed_mps": 6.0}}
                windy = api.build_conditions(cfg, 4, "cpu", family=family)
                self.assertEqual(windy.cfg.wind.windage_profile, "blueboat_hull_only_c3")

    def test_journal_post_init_sets_version_after_registry_rebuild(self):
        for family, filename, name, _ in self.CASES:
            path = REPO / "tasks" / family / (filename + ".py")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            post = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
            body = [ast.unparse(statement) for statement in post.body if not isinstance(statement, ast.Expr)
                    or not isinstance(statement.value, ast.Constant)]
            self.assertEqual(body[0], "super().__post_init__()")
            self.assertEqual(body[1], "self.underwater_physics_cfg.physics_version = 2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
