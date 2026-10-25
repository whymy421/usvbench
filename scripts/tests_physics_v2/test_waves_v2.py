"""CPU wave regression tests; load copied and original modules without Isaac."""

import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import warnings

sys.dont_write_bytecode = True

import torch


REPO = Path(__file__).resolve().parents[2]
ORIGINAL = Path(os.environ.get("USVBENCH_ORIGINAL_ROOT", "C:/Users/BRADY/usvbench"))


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# These production helpers have no Isaac dependency. Resolve their fallback
# imports explicitly by path, without importing the tasks package.
scenario_rng = load_path("scenario_rng", REPO / "tasks/_shared/scenario_rng.py")
load_path("scenario_draws", REPO / "tasks/_shared/scenario_draws.py")
wave = load_path("physics_v2_wave", REPO / "tasks/_shared/sea_state.py")
legacy = load_path("physics_v2_wave_original", ORIGINAL / "tasks/_shared/sea_state.py")
physics = load_path("physics_v2_wave_clock", REPO / "tasks/_shared/physics_v2.py")


class WaveTests(unittest.TestCase):
    def make(self, version=2, seed=45, num_envs=4, **kwargs):
        torch.manual_seed(seed)
        cfg = wave.SeaStateCfg(**kwargs)
        return wave.SeaState(cfg, num_envs, torch.device("cpu"), physics_version=version)

    def assert_state_equal(self, old, new):
        for field in ("freqs", "omega", "k", "hs", "tp", "gamma", "mean_direction",
                      "amplitude", "phase", "direction"):
            self.assertTrue(torch.equal(getattr(old, field), getattr(new, field)), field)
        self.assertEqual(old.df, new.df)
        xy = torch.tensor([[0.3, -1.7], [-3.1, 1.0], [4.5, 0.0], [9.1, 9.2]])
        vel = torch.tensor([[0.0, 0.0], [-3.0, 0.1], [0.1, 1.0], [2.0, -1.1]])
        heading = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.6, 0.8]])
        for t in (0.0, 0.125, 11.37):
            for method in ("_phase_at", "elevation", "orbital_velocity", "surface_slope"):
                self.assertTrue(torch.equal(getattr(old, method)(xy, t),
                                            getattr(new, method)(xy, t)), method)
            old_force = old.forces(xy, vel, t)
            new_force = new.forces(xy, vel, t)
            for a, b in zip(old_force, new_force):
                self.assertTrue(torch.equal(a, b), "forces")
        self.assertTrue(torch.equal(old.observation(heading), new.observation(heading)))

    def test_v1_exact_original_numbers_rng_and_subset_reset(self):
        for cfg_values in ({}, {"hs_range": (0.0, 0.0)},
                           {"hs_range": (0.30, 0.60), "tp_range": (1.5, 3.0)}):
            with self.subTest(cfg_values=cfg_values), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                torch.manual_seed(8712)
                old = legacy.SeaState(legacy.SeaStateCfg(**cfg_values), 4, torch.device("cpu"))
                old_rng = torch.get_rng_state().clone()
                torch.manual_seed(8712)
                # Default version MUST be the legacy path.
                new = wave.SeaState(wave.SeaStateCfg(**cfg_values), 4, torch.device("cpu"))
                self.assertEqual(new.physics_version, 1)
                self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                self.assert_state_equal(old, new)
                ids = torch.tensor([1, 3])
                torch.manual_seed(117)
                old.resample(ids)
                old_rng = torch.get_rng_state().clone()
                torch.manual_seed(117)
                new.resample(ids)
                self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                self.assert_state_equal(old, new)

    def test_v2_zero_amplitude_is_exact_calm_for_moving_hulls(self):
        sea = self.make(hs_range=(0.0, 0.0))
        clock = physics.WaveClock(4, "cpu", 1.0 / 120.0)
        clock.ticks[:] = torch.tensor([0, 30, 1440, 36000])
        clocks = (clock.time(), torch.tensor([0.0, 0.25, 12.0, 300.0], dtype=torch.float64))
        for times in clocks:
            with self.subTest(clock_dtype=times.dtype):
                force, torque = sea.forces(
                    torch.tensor([[0.0, 0.0], [1.0, -3.0], [25.0, 40.0], [-3.0, 0.1]]),
                    torch.tensor([[0.0, 0.0], [0.12345, 0.678], [3.13, -2.12], [-100.137, 70.281]], dtype=torch.float32),
                    times,
                )
                self.assertTrue(torch.equal(force, torch.zeros_like(force)))
                self.assertTrue(torch.equal(torque, torch.zeros_like(torque)))

    def test_v2_calm_increment_can_be_disabled(self):
        inc = self.make(calm_water_increment=True)
        full = self.make(calm_water_increment=False)
        xy = torch.zeros((4, 2))
        vel = torch.tensor([[0.1, 0.0], [0.0, 1.0], [-3.0, 4.0], [0.0, 0.0]])
        force_inc, torque_inc = inc.forces(xy, vel, 2.3)
        force_full, torque_full = full.forces(xy, vel, 2.3)
        calm = full.cfg.orbital_drag_coeff * (-vel) * torch.norm(-vel, dim=-1, keepdim=True)
        self.assertTrue(torch.equal(force_inc[:, :2], force_full[:, :2] - calm))
        self.assertTrue(torch.equal(force_inc[:, 2], force_full[:, 2]))
        self.assertTrue(torch.equal(torque_inc, torque_full))

    def test_v1_ignores_new_increment_switch(self):
        inc = self.make(version=1, hs_range=(0.0, 0.0), calm_water_increment=True)
        full = self.make(version=1, hs_range=(0.0, 0.0), calm_water_increment=False)
        xy = torch.zeros((4, 2))
        vel = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
        force, _ = inc.forces(xy, vel, 0.0)
        self.assertTrue(torch.equal(force, full.forces(xy, vel, 0.0)[0]))
        self.assertTrue(torch.equal(force[:, 0], torch.full((4,), -8.0)))

    def test_vector_clocks_equal_individual_scalar_evaluation(self):
        sea = self.make()
        xy = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 3.0
        vel = torch.ones_like(xy)
        times = torch.tensor([0.0, 0.2, 3.5, 9.1])
        for method in ("_phase_at", "elevation", "orbital_velocity", "surface_slope"):
            actual = getattr(sea, method)(xy, times)
            expected = torch.stack([
                getattr(sea, method)(xy, float(times[i]))[i] for i in range(4)
            ])
            self.assertTrue(torch.equal(actual, expected), method)
        before = sea.forces(xy, vel, times)
        sea.resample(torch.tensor([0]))
        times[0] = 0.0
        after = sea.forces(xy, vel, times)
        for a, b in zip(before, after):
            self.assertTrue(torch.equal(a[1:], b[1:]))
        with self.assertRaises(ValueError):
            sea.elevation(xy, torch.zeros(3))

    def test_v2_band_tracks_every_draw_and_subset_reset(self):
        sea = self.make(num_envs=128, hs_range=(0.01, 0.01), tp_range=(0.5, 20.0))
        n = sea.amplitude.shape[1]
        self.assertGreaterEqual(n, 256)
        self.assertTrue(torch.allclose(sea.freqs[:, -1] + sea.df[:, 0] / 2.0,
                                       3.0 / sea.tp))
        self.assertTrue(torch.all(sea.freqs[:, -1] >= 2.5 / sea.tp))
        self.assertTrue(torch.all(sea.df[:, 0] * sea.tp <= 0.0125))
        self.assertTrue(torch.allclose(sea.k, sea.omega**2 / wave.GRAVITY))
        hs_recovered = 4.0 * torch.sqrt((sea.amplitude**2 / 2.0).sum(dim=1))
        self.assertTrue(torch.allclose(hs_recovered, sea.hs))
        original_rows = {key: getattr(sea, key).clone() for key in
                         ("freqs", "df", "omega", "k", "amplitude", "phase")}
        sea.resample(torch.tensor([0, 7, 99]))
        keep = torch.ones(128, dtype=torch.bool)
        keep[[0, 7, 99]] = False
        for key, value in original_rows.items():
            self.assertTrue(torch.equal(value[keep], getattr(sea, key)[keep]), key)
        self.assertTrue(torch.allclose(sea.freqs[:, -1] + sea.df[:, 0] / 2.0,
                                       3.0 / sea.tp))

    def test_spectral_capture_before_renormalisation_and_peak_resolution(self):
        for tp in (0.5, 1.0, 1.4, 3.0, 7.0, 20.0):
            for gamma in (1.0, 5.0):
                with self.subTest(tp=tp, gamma=gamma):
                    sea = self.make(num_envs=1, hs_range=(0.01, 0.01),
                                    tp_range=(tp, tp), gamma_range=(gamma, gamma))
                    f = sea.freqs[0].double()
                    integral = float(wave.jonswap_spectrum(f, 0.01, tp, gamma).sum()) * float(sea.df[0, 0])
                    dense = torch.linspace(0.01 / tp, 100.0 / tp, 100_000, dtype=torch.float64)
                    reference = float(torch.trapezoid(wave.jonswap_spectrum(dense, 0.01, tp, gamma), dense))
                    self.assertGreater(integral / reference, 0.98)
                    # Check discretisation separately from the tail cutoff.
                    band_dense = torch.linspace(0.2 / tp, 3.0 / tp, 20_000, dtype=torch.float64)
                    band_reference = float(torch.trapezoid(wave.jonswap_spectrum(band_dense, 0.01, tp, gamma), band_dense))
                    self.assertLess(abs(integral / band_reference - 1.0), 0.0001)

    def test_steepness_admission_is_retained_and_exception_is_v1_only(self):
        cfg = wave.SeaStateCfg(hs_range=(0.30, 0.60), tp_range=(1.5, 3.0))
        with self.assertWarns(RuntimeWarning):
            wave.validate_steepness_admission(cfg, physics_version=1)
        with self.assertRaises(ValueError):
            wave.validate_steepness_admission(cfg, physics_version=2)
        safe_hs = wave.GRAVITY * 2.0**2 / (2.0 * math.pi) / 7.0
        wave.validate_steepness_admission(wave.SeaStateCfg(hs_range=(0.0, safe_hs), tp_range=(2.0, 2.0)), 2)
        with self.assertRaises(ValueError):
            self.make(hs_range=(0.0, safe_hs * 1.001), tp_range=(2.0, 2.0))

    def test_invalid_v2_band_and_period_are_rejected(self):
        for kwargs in ({"v2_f_max_over_fp": 2.4}, {"v2_f_min_over_fp": 1.0},
                       {"v2_f_min_over_fp": float("nan")},
                       {"tp_range": (0.0, 0.0), "hs_range": (0.0, 0.0)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)
        with self.assertRaises(ValueError):
            self.make(version=3)

    def test_wave_clock_physics_dt_and_independent_reset(self):
        clock = physics.WaveClock(4, "cpu", 1.0 / 120.0)
        dtype = clock.time().dtype
        self.assertEqual(clock.ticks.dtype, torch.int64)
        self.assertTrue(torch.equal(clock.time(), torch.zeros(4, dtype=dtype)))
        clock.advance()
        self.assertTrue(torch.equal(clock.time(), torch.full((4,), 1.0 / 120.0, dtype=dtype)))
        clock.advance()
        clock.reset(torch.tensor([1, 3]))
        self.assertTrue(torch.equal(clock.time(), torch.tensor([2.0, 0.0, 2.0, 0.0], dtype=dtype) / 120.0))
        clock.ticks[:] = 123456789
        self.assertTrue(torch.allclose(clock.time(), torch.full((4,), 123456789 / 120.0, dtype=dtype)))
        for bad_dt in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(dt=bad_dt), self.assertRaises(ValueError):
                physics.WaveClock(2, "cpu", bad_dt)

    def test_six_environment_clock_wiring_executes_actual_wave_blocks(self):
        # Execute just each real wave block, clock construction and reset from
        # its AST. This exercises production wiring without importing Isaac.
        paths = (
            "docking/docking_env.py", "harbor_mission/harbor_mission_env.py",
            "hazard_nav/hazard_nav_env.py", "path_following/path_following_env.py",
            "path_hazard/path_hazard_env.py", "station_keeping/station_keeping_env.py",
        )

        def is_call(node, text):
            return isinstance(node, ast.Call) and ast.unparse(node.func) == text

        def is_version(node):
            return ast.unparse(node) in (
                "physics_version(self.physics_cfg)",
                "getattr(self.physics_cfg, 'physics_version', 1)",
            )

        def is_v2_guard(node):
            return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                    and is_version(node.test.left) and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq)
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and node.test.comparators[0].value == 2)

        def execute(node, scope):
            module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exec(compile(module, "<production-wave-block>", "exec"), scope)

        class SeaRecorder:
            def __init__(self):
                self.times = []

            def forces(self, xy, velocity, t):
                self.times.append(t.clone() if isinstance(t, torch.Tensor) else t)
                return torch.zeros((2, 3)), torch.zeros((2, 3))

        for relative in paths:
            with self.subTest(environment=relative):
                tree = ast.parse((REPO / "tasks" / relative).read_text(encoding="utf-8-sig"))
                methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                apply_action = next(node for node in methods if node.name == "_apply_action")
                wave_block = next(node for node in ast.walk(apply_action)
                                  if isinstance(node, ast.If)
                                  and ast.unparse(node.test) == "self._sea is not None")
                init_block = next(node for node in ast.walk(tree)
                                  if is_v2_guard(node)
                                  and any(is_call(child, "WaveClock") for child in ast.walk(node)))
                reset_method = next(node for node in methods if node.name == "_reset_idx")
                reset_block = next(node for node in ast.walk(reset_method)
                                   if is_v2_guard(node)
                                   and any(is_call(child, "self._wave_clock.reset") for child in ast.walk(node)))
                ctor = next(node for node in ast.walk(init_block) if is_call(node, "WaveClock"))
                self.assertEqual(ast.unparse(ctor.args[2]), "self.cfg.sim.dt")
                sea_ctor = next(node for node in ast.walk(tree) if is_call(node, "SeaState"))
                self.assertTrue(any(key.arg == "physics_version" and
                                    is_version(key.value)
                                    for key in sea_ctor.keywords))
                for version in (1, 2):
                    env = SimpleNamespace(
                        num_envs=2, device="cpu", _sea=SeaRecorder(),
                        cfg=SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 120.0)),
                        physics_cfg=SimpleNamespace(physics_version=version),
                        control_step_s=1.0 / 60.0,
                        episode_length_buf=torch.tensor([12, 222]),
                        robot=SimpleNamespace(data=SimpleNamespace(
                            root_com_pos_w=torch.zeros((2, 3)),
                            root_com_vel_w=torch.zeros((2, 6)),
                        )),
                    )
                    scope = {"self": env, "WaveClock": physics.WaveClock,
                             "physics_version": lambda cfg: cfg.physics_version,
                             "env_ids": torch.tensor([0]),
                             "force_w": torch.zeros((2, 3)), "torque_w": torch.zeros((2, 3)),
                             "force_world": torch.zeros((2, 3)), "torque_world": torch.zeros((2, 3))}
                    execute(init_block, scope)
                    execute(wave_block, scope)
                    execute(wave_block, scope)
                    execute(reset_block, scope)
                    execute(wave_block, scope)
                    if version == 1:
                        self.assertFalse(hasattr(env, "_wave_clock"))
                        self.assertEqual(env._sea.times, [0.2, 0.2, 0.2])
                    else:
                        expected = ([0.0, 0.0], [1.0, 1.0], [0.0, 2.0])
                        for actual, ticks in zip(env._sea.times, expected):
                            self.assertTrue(torch.equal(actual, torch.tensor(ticks, dtype=actual.dtype) / 120.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
