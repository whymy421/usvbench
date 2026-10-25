"""CPU-only wind acceptance tests; run: python -B test_wind.py.

The module is imported by file path, without importing an Isaac task package.
Statistical tests use a fixed independent ensemble of keyed OU trajectories,
not many correlated samples falsely counted as independent observations.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

sys.dont_write_bytecode = True

import torch

_SPEC = importlib.util.spec_from_file_location("wind_under_test", Path(__file__).with_name("wind.py"))
wind = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = wind
_SPEC.loader.exec_module(wind)

Wind, WindCfg, VesselWindage = wind.Wind, wind.WindCfg, wind.VesselWindage
DTYPE = torch.float64
GEOMETRY = VesselWindage(0.22, 0.48, 1.2, 0.18, 0.07, 0.8, 1.1, 0.12)


def config(**kwargs):
    return replace(WindCfg(enable=True, mean_speed_mps=6.0, windage=GEOMETRY), **kwargs)


def layer(cfg=None, n=1, seed=19, keys=None, episodes=None, dtype=DTYPE):
    field = Wind(config() if cfg is None else cfg, n, "cpu", eval_seed=seed,
                 env_keys=keys, dtype=dtype)
    field.reset(None, [0] * n if episodes is None else episodes)
    return field


def tensor(values):
    return torch.tensor(values, dtype=DTYPE)


class WindForceTests(unittest.TestCase):
    def test_static_formula_several_headings_and_incidence_angles(self):
        # Construct relative flow from the specified incoming gamma, instead
        # of reproducing the implementation's atan2 calculation.
        for heading in (0.0, math.pi / 3, math.pi / 2, math.pi, -1.4):
            for gamma in (0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2):
                with self.subTest(heading=heading, gamma=gamma):
                    direction = heading + math.pi - gamma
                    field = layer(config(direction_rad=direction))
                    force, torque = field.forces(tensor([[0.0, 0.0]]), tensor([heading]))
                    q = 0.5 * 1.225 * 36.0
                    x = -q * 0.22 * 0.8 * math.cos(gamma)
                    y = q * 0.48 * 1.1 * math.sin(gamma)
                    k = -y * 0.18
                    n = q * 0.48 * 1.2 * 0.12 * math.sin(2 * gamma)
                    c, s = math.cos(heading), math.sin(heading)
                    torch.testing.assert_close(force, tensor([[c*x-s*y, s*x+c*y, 0.0]]),
                                               rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(torque, tensor([[c*k, s*k, n]]),
                                               rtol=1e-12, atol=1e-12)

    def test_force_pushes_downwind_and_roll_is_right_hand_cross_product(self):
        field = layer(config(direction_rad=math.pi / 2))
        force, torque = field.forces(tensor([[0.0, 0.0]]), tensor([0.0]))
        self.assertGreater(force[0, 1].item(), 0.0)
        arm = tensor([[0.0, 0.0, GEOMETRY.centroid_height_m]])
        expected = torch.linalg.cross(arm, force)
        torch.testing.assert_close(torque[:, 0], expected[:, 0])
        self.assertLess(torque[0, 0].item(), 0.0)

    def test_relative_wind_tailwind_headwind_and_comoving(self):
        field = layer(n=4)
        velocity = tensor([[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [6.0, 0.0]])
        force, torque = field.forces(velocity, tensor([0.0] * 4))
        expected = tensor([0.5 * 1.225 * 0.22 * 0.8 * v*v for v in (6.0, 4.0, 8.0, 0.0)])
        torch.testing.assert_close(force[:, 0], expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(torch.count_nonzero(force[3]).item(), 0)
        self.assertEqual(torch.count_nonzero(torque[3]).item(), 0)

    def test_relative_wind_oblique_motion_equals_static_apparent_wind(self):
        moving = layer(config(direction_rad=0.7))
        vx, vy = 6 * math.cos(0.7) - 1.8, 6 * math.sin(0.7) + 0.9
        apparent = layer(config(mean_speed_mps=math.hypot(vx, vy),
                                direction_rad=math.atan2(vy, vx)))
        a = moving.forces(tensor([[1.8, -0.9]]), tensor([-0.4]))
        b = apparent.forces(tensor([[0.0, 0.0]]), tensor([-0.4]))
        for av, bv in zip(a, b):
            torch.testing.assert_close(av, bv, rtol=1e-12, atol=1e-12)

    def test_enabled_still_air_opposes_motion(self):
        field = layer(config(mean_speed_mps=0.0, gust_intensity=0.2), n=3)
        velocities = tensor([[3.0, 0.0], [0.0, -2.0], [1.0, 2.0]])
        force, _ = field.forces(velocities, tensor([0.0, 0.0, 0.8]))
        self.assertTrue(bool(((force[:, :2] * velocities).sum(-1) < 0).all()))
        self.assertEqual(field.normal_draw_count, 0)

    def test_swappable_coefficient_model_and_cp_not_double_counted(self):
        def model(gamma):
            return torch.full_like(gamma, 0.2), torch.full_like(gamma, -0.3), torch.full_like(gamma, 0.04)
        cfg = config(windage=replace(GEOMETRY, longitudinal_cp_m=100.0))
        field = Wind(cfg, 1, dtype=DTYPE, coefficient_model=model)
        field.reset(None, [0])
        f, t = field.forces(tensor([[0.0, 0.0]]), tensor([0.0]))
        q = 0.5 * 1.225 * 36
        torch.testing.assert_close(f, tensor([[q*0.22*0.2, q*0.48*(-0.3), 0.0]]))
        torch.testing.assert_close(t, tensor([[q*0.48*0.3*0.18, 0.0, q*0.48*1.2*0.04]]))

    def test_float32_output_contract(self):
        field = layer(dtype=torch.float32, n=3)
        result = field.forces(torch.zeros((3, 2)), torch.zeros(3))
        for array in result:
            self.assertEqual(tuple(array.shape), (3, 3))
            self.assertEqual(array.dtype, torch.float32)
            self.assertEqual(array.device.type, "cpu")


class WindStreamTests(unittest.TestCase):
    def test_ou_mean_variance_and_autocorrelation_time(self):
        count, tau, dt = 10000, 0.5, 1.0 / 120.0
        sigma = 6.0 * 0.2
        field = layer(config(gust_intensity=0.2, gust_correlation_time_s=tau), n=count)
        initial = field.gust_mps.clone()
        samples = {0: initial}
        for k in range(1, 121):
            field.step(dt)
            if k in (60, 120):
                samples[k] = field.gust_mps.clone()
        for k, sample in samples.items():
            mean = sample.mean().item()
            variance = sample.var(unbiased=True).item()
            print(f"OU t={k*dt:.3f}s: mean={mean:.6f}, variance={variance:.6f}, target={sigma*sigma:.6f}")
            self.assertLess(abs(mean), 0.04 * sigma)
            self.assertLess(abs(variance / (sigma*sigma) - 1.0), 0.05)
        for k in (60, 120):
            corr = torch.corrcoef(torch.stack((initial, samples[k])))[0, 1].item()
            expected = math.exp(-k * dt / tau)
            tau_fit = -(k * dt) / math.log(corr)
            print(f"OU lag={k*dt:.3f}s: rho={corr:.6f}, target={expected:.6f}, fitted_tau={tau_fit:.6f}s")
            self.assertLess(abs(corr - expected), 0.03)
            self.assertLess(abs(tau_fit / tau - 1.0), 0.12)

    def test_exact_ou_update_and_stationary_initialization(self):
        cfg = config(gust_intensity=0.25, gust_correlation_time_s=1.7)
        field = layer(cfg, keys=[53], episodes=[7], seed=41)
        g0 = 1.5 * wind._normal_for_key(41, 53, 7, 0)
        self.assertEqual(field.gust_mps.item(), g0)
        dt = 0.03
        field.step(dt)
        a = math.exp(-dt / 1.7)
        expected = a*g0 + 1.5*math.sqrt(-math.expm1(-2*dt/1.7))*wind._normal_for_key(41, 53, 7, 1)
        self.assertAlmostEqual(field.gust_mps.item(), expected, places=14)

    def test_streams_ignore_completion_order_batch_order_and_sharding(self):
        cfg = config(gust_intensity=0.2, direction_drift_rad_s=0.03)
        batch = layer(cfg, n=3, keys=[100, 200, 300], episodes=[2, 5, 8])
        reordered = layer(cfg, n=3, keys=[300, 100, 200], episodes=[8, 2, 5])
        alone = layer(cfg, keys=[200], episodes=[5])
        for k in range(25):
            batch.step(1/120)
            reordered.step(1/120, [2, 0, 1])
            alone.step(1/120)
            if k in (3, 9, 12):
                # Other environments may finish and reset in arbitrary order.
                batch.reset([2, 0], [k+10, k+20])
            self.assertEqual(batch.gust_mps[1].item(), alone.gust_mps[0].item())
            self.assertEqual(batch.gust_mps[1].item(), reordered.gust_mps[2].item())
            torch.testing.assert_close(batch.velocity_world()[1], alone.velocity_world()[0], rtol=0, atol=0)

    def test_replay_and_each_key_component_changes_stream(self):
        cfg = config(gust_intensity=0.3)
        field = layer(cfg, seed=2, keys=[10], episodes=[4])
        first = []
        for _ in range(10):
            field.step(0.01)
            first.append(field.gust_mps.item())
        field.reset(None, [4])
        for expected in first:
            field.step(0.01)
            self.assertEqual(field.gust_mps.item(), expected)
        g = [layer(cfg, seed=s, keys=[e], episodes=[p]).gust_mps.item()
             for s, e, p in ((2, 10, 4), (3, 10, 4), (2, 11, 4), (2, 10, 5))]
        self.assertEqual(len(set(g)), 4)

    def test_direction_clock_is_per_environment_and_force_query_is_read_only(self):
        field = layer(config(direction_rad=0.1, direction_drift_rad_s=0.2), n=2)
        field.step(0.1, [1])
        field.step(0.15, [1])
        expected = tensor([[6*math.cos(0.1), 6*math.sin(0.1)],
                           [6*math.cos(0.15), 6*math.sin(0.15)]])
        torch.testing.assert_close(field.velocity_world(), expected)
        before = (field.gust_mps.clone(), field.time_s.clone(), field._physics_steps[:])
        for _ in range(3):
            field.forces(torch.zeros((2, 2), dtype=DTYPE), torch.zeros(2, dtype=DTYPE))
        self.assertTrue(torch.equal(before[0], field.gust_mps))
        self.assertTrue(torch.equal(before[1], field.time_s))
        self.assertEqual(before[2], field._physics_steps)

    def test_no_global_rng_consumption_even_when_enabled(self):
        torch_before, python_before = torch.random.get_rng_state().clone(), random.getstate()
        field = layer(config(gust_intensity=0.2), n=3)
        field.step(0.01)
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        self.assertEqual(python_before, random.getstate())

    def test_disabled_exact_zeros_and_no_draws(self):
        cfg = config(enable=False, gust_intensity=0.8)
        with patch.object(wind, "_normal_for_key", side_effect=AssertionError("RNG consumed")):
            field = layer(cfg, n=2)
            field.reset([1], [999])
            field.step(1/120)
            force, torque = field.forces(tensor([[10.0, -30.0], [1.0, 2.0]]), tensor([0.3, -2.0]))
            self.assertTrue(torch.equal(force, torch.zeros_like(force)))
            self.assertTrue(torch.equal(torque, torch.zeros_like(torque)))
            self.assertTrue(torch.equal(field.velocity_world(), torch.zeros((2, 2), dtype=DTYPE)))
            self.assertEqual(field.normal_draw_count, 0)
            self.assertEqual(field._physics_steps, [0, 0])
            self.assertEqual(field._episode_ids, [None, None])

    def test_gust_free_enabled_case_consumes_no_draws(self):
        with patch.object(wind, "_normal_for_key", side_effect=AssertionError("RNG consumed")):
            field = layer(config(gust_intensity=0.0))
            field.step(0.01)
            field.forces(tensor([[0.0, 0.0]]), tensor([0.0]))
            self.assertEqual(field.normal_draw_count, 0)


class WindValidationTests(unittest.TestCase):
    def test_cfg_accepts_cli_setattr_and_constructor_snapshots_it(self):
        cfg = WindCfg(windage=replace(GEOMETRY))
        setattr(cfg, "enable", True)
        setattr(cfg, "mean_speed_mps", 6.0)
        setattr(cfg.windage, "c_y0", 1.2)
        field = layer(cfg)
        cfg.mean_speed_mps = 50.0
        cfg.windage.c_y0 = 10.0
        self.assertEqual(field.cfg.mean_speed_mps, 6.0)
        self.assertEqual(field.cfg.windage.c_y0, 1.2)

    def test_bad_configuration_and_keys_rejected(self):
        for change in ({"mean_speed_mps": -1}, {"gust_correlation_time_s": 0},
                       {"gust_intensity": -1}, {"air_density_kg_m3": 0},
                       {"direction_rad": math.nan}, {"windage": None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Wind(config(**change), 1)
        with self.assertRaises(ValueError):
            Wind(config(), 2, env_keys=[1, 1])
        with self.assertRaises(ValueError):
            Wind(config(), 1, eval_seed=1.5)

    def test_reset_required_and_invalid_step_or_reset_is_rejected(self):
        field = Wind(config(), 2, dtype=DTYPE)
        with self.assertRaises(RuntimeError):
            field.step(0.01)
        with self.assertRaises(RuntimeError):
            field.forces(torch.zeros((2, 2), dtype=DTYPE), torch.zeros(2, dtype=DTYPE))
        for dt in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                field.step(dt)
        for ids, episodes in (([0, 0], [1, 2]), ([2], [0]), ([0], []), ([0], [-1])):
            with self.assertRaises(ValueError):
                field.reset(ids, episodes)


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(f"Python {sys.version.split()[0]}; Torch {torch.__version__}; CPU only; float64 acceptance")
    unittest.main(verbosity=2)
