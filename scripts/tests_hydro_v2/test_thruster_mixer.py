"""CPU-only mixer properties; load files directly and disable bytecode.

E2 change (decision 1): the vessel profiles now come from the D5 registry,
whose BlueBoat per-motor limits (8.2 kgf split, M200 reverse ratio) and yaw
request scale differ from D3's provisional 40/24 N and 23 N m. The allocator
property tests below are vessel-agnostic; they keep D3's round reference
numbers through an explicit fixture ThrusterSpec (REFERENCE_SPEC, NOT a
vessel), and the registry profiles are checked in the tests named
*_registry_* and test_wamv_vrx_classic_envelope.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import torch

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "tasks" / "_shared"
sys.path.insert(0, str(SHARED))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SHARED / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Canonical module names so hydro_profiles' fallback imports reuse these objects.
load("physics_v2")
load("restoring_v2")
load("hydro_v2")
mixer = load("thruster_mixer")
registry = load("vehicles_v2")
load("journal_vehicle")
profiles = load("hydro_profiles")
DTYPE = torch.float64
# D3's reviewed round numbers (legacy BlueBoat 40/24 N per motor, arm 0.3607 m,
# 23 N m request scale) kept as a vessel-agnostic allocator fixture.
REFERENCE_SPEC = mixer.ThrusterSpec((40.0, 40.0), (24.0, 24.0), .3607, 23.0)


def blueboat_spec(nominal_lag_s=(0.0, 0.0)):
    return replace(REFERENCE_SPEC, nominal_lag_s=nominal_lag_s)


class ThrusterMixerTests(unittest.TestCase):
    def setUp(self):
        self.spec = blueboat_spec()

    def step(self, actions, **kwargs):
        actions = torch.as_tensor(actions, dtype=DTYPE)
        return mixer.step_thrusters(actions, mixer.zero_state(actions), self.spec,
                                   1 / 120, **kwargs)

    def test_reviewed_blueboat_parameters(self):
        # Fixture values (D3 review) ...
        self.assertEqual(self.spec.forward_limit_n, (40, 40))
        self.assertEqual(self.spec.reverse_limit_n, (24, 24))
        self.assertEqual(self.spec.arm_m, .3607)
        self.assertEqual(self.spec.yaw_command_limit_nm, 23)
        # ... and the registry BlueBoat v2 thrusters through the adapter.
        spec = registry.get_vehicle_v2("blueboat_v2")
        thrusters = profiles.profile_for_vehicle("blueboat_v2").thrusters
        fwd = 8.2 * 9.80665 / 2                                   # D5 PARAMS: 8.2 kgf split equally
        rev = fwd * 2.81 / 5.63                                   # M200 weedless 16 V ratio
        self.assertAlmostEqual(thrusters.forward_limit_n[0], fwd, places=12)
        self.assertAlmostEqual(thrusters.reverse_limit_n[1], rev, places=12)
        self.assertEqual(thrusters.arm_m, .3607)
        self.assertAlmostEqual(thrusters.yaw_command_limit_nm, .3607 * (fwd + rev), places=12)
        self.assertEqual(thrusters.mount_z_m, 0.0)                # decision 6
        self.assertEqual(thrusters.nominal_lag_s, (0.0, 0.0))
        self.assertEqual(spec.propulsion.planar_arm_m, thrusters.arm_m)

    def test_registry_blueboat_pure_yaw_and_envelope(self):
        thrusters = profiles.profile_for_vehicle("blueboat_v2").thrusters
        fwd, rev = thrusters.forward_limit_n[0], thrusters.reverse_limit_n[0]
        actions = torch.tensor([[0, 1], [1, 0], [-1, 0], [1, 1], [-1, -1]], dtype=DTYPE)
        result = mixer.step_thrusters(actions, mixer.zero_state(actions), thrusters, 1 / 120)
        torch.testing.assert_close(result.body_wrench[0, 5], torch.tensor(2 * rev * .3607, dtype=DTYPE))
        torch.testing.assert_close(result.body_wrench[1:3, 0], torch.tensor([2 * fwd, -2 * rev], dtype=DTYPE))
        self.assertTrue(bool((result.delivered_thrust <= fwd + 1e-12).all()))
        self.assertTrue(bool((result.delivered_thrust >= -rev - 1e-12).all()))
        self.assertAlmostEqual(registry.get_vehicle_v2("blueboat_v2").propulsion.pure_yaw_limit_nm,
                               float(result.body_wrench[0, 5]), places=12)

    def test_zero_request_exact_zero(self):
        result = self.step(torch.zeros((100, 2), dtype=DTYPE))
        self.assertTrue(torch.equal(result.body_wrench, torch.zeros((100, 6), dtype=DTYPE)))
        self.assertTrue(torch.equal(result.saturation_scale, torch.ones(100, dtype=DTYPE)))

    def test_full_surge_and_reverse(self):
        result = self.step([[1, 0], [-1, 0]])
        torch.testing.assert_close(result.delivered_thrust,
                                   torch.tensor([[40, 40], [-24, -24]], dtype=DTYPE))
        torch.testing.assert_close(result.body_wrench[:, 0], torch.tensor([80, -48], dtype=DTYPE))
        self.assertTrue(torch.equal(result.body_wrench[:, 5], torch.zeros(2, dtype=DTYPE)))

    def test_pure_yaw_is_reverse_limited(self):
        result = self.step([[0, 1], [0, -1]])
        torch.testing.assert_close(result.delivered_thrust,
                                   torch.tensor([[-24, 24], [24, -24]], dtype=DTYPE))
        torch.testing.assert_close(result.body_wrench[:, 5],
                                   torch.tensor([17.3136, -17.3136], dtype=DTYPE))
        self.assertTrue(torch.equal(result.body_wrench[:, 0], torch.zeros(2, dtype=DTYPE)))

    def test_full_surge_full_yaw_radial_corner(self):
        result = self.step([[1, 1]])
        scale = 40 / ((80 + 23 / .3607) / 2)
        torch.testing.assert_close(result.saturation_scale, torch.tensor([scale], dtype=DTYPE))
        torch.testing.assert_close(result.body_wrench[:, [0, 5]],
                                   torch.tensor([[80 * scale, 23 * scale]], dtype=DTYPE))
        self.assertLessEqual(float(result.delivered_thrust.max()), 40)
        self.assertGreaterEqual(float(result.delivered_thrust.min()), -24)

    def test_clip_is_configurable_and_not_radial(self):
        radial = self.step([[1, 1]])
        clipped = self.step([[1, 1]], saturation="clip")
        expected = torch.tensor([[(80 - 23 / .3607) / 2, 40]], dtype=DTYPE)
        torch.testing.assert_close(clipped.delivered_thrust, expected)
        self.assertGreater(float(clipped.body_wrench[0, 0]), float(radial.body_wrench[0, 0]))
        self.assertLess(float(clipped.body_wrench[0, 5]), float(radial.body_wrench[0, 5]))

    def test_random_envelope_and_radial_direction_property(self):
        gen = torch.Generator().manual_seed(19223)
        actions = 2 * torch.rand((10000, 2), dtype=DTYPE, generator=gen) - 1
        for policy in ("radial", "clip"):
            result = self.step(actions, saturation=policy)
            self.assertTrue(bool((result.delivered_thrust <= 40).all()))
            self.assertTrue(bool((result.delivered_thrust >= -24).all()))
            if policy == "radial":
                expected = result.requested_thrust * result.saturation_scale[:, None]
                torch.testing.assert_close(result.delivered_thrust, expected)

    def test_action_all_corners_stay_within_motor_envelope(self):
        result = self.step([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        self.assertTrue(bool((result.delivered_thrust <= 40).all()))
        self.assertTrue(bool((result.delivered_thrust >= -24).all()))

    def test_cap_is_limit_not_throttle_gain(self):
        degraded = mixer.ThrusterDegradation(thrust_cap=.5)
        small = self.step([[.1, 0]], degradation=degraded)
        nominal = self.step([[.1, 0]])
        torch.testing.assert_close(small.delivered_thrust, nominal.delivered_thrust)
        large = self.step([[1, 0], [-1, 0]], degradation=degraded)
        torch.testing.assert_close(large.delivered_thrust,
                                   torch.tensor([[20, 20], [-12, -12]], dtype=DTYPE))

    def test_one_sided_loss_never_boosts_healthy_motor(self):
        result = self.step([[1, 0], [-1, 0]],
                           degradation=mixer.ThrusterDegradation(motor_gain=(.5, 1)))
        torch.testing.assert_close(result.delivered_thrust,
                                   torch.tensor([[20, 40], [-12, -24]], dtype=DTYPE))
        torch.testing.assert_close(result.body_wrench[:, 5],
                                   torch.tensor([.3607 * 20, -.3607 * 12], dtype=DTYPE))

    def test_cap_and_loss_compose(self):
        result = self.step([[1, 0], [-1, 0]],
                           degradation=mixer.ThrusterDegradation(.5, (.5, 1)))
        torch.testing.assert_close(result.delivered_thrust,
                                   torch.tensor([[10, 20], [-6, -12]], dtype=DTYPE))

    def test_random_degraded_envelope_property(self):
        gen = torch.Generator().manual_seed(81223)
        actions = 2 * torch.rand((4000, 2), dtype=DTYPE, generator=gen) - 1
        cap = torch.rand((4000, 1), dtype=DTYPE, generator=gen)
        gain = torch.rand((4000, 2), dtype=DTYPE, generator=gen)
        degradation = mixer.ThrusterDegradation(cap, gain, .25)
        # Deliberately invalid prior forces model an abrupt authority change.
        state = mixer.ThrusterState(torch.full_like(actions, 100), torch.full_like(actions, -100))
        for policy in ("radial", "clip"):
            result = mixer.step_thrusters(actions, state, self.spec, .01, degradation, policy)
            for value in (result.commanded_thrust, result.delivered_thrust,
                          result.state.nominal, result.state.extra):
                self.assertTrue(bool((value <= 40 * cap * gain).all()))
                self.assertTrue(bool((value >= -24 * cap * gain).all()))

    def test_zero_cap_kills_all_stages_immediately(self):
        actions = torch.ones((1, 2), dtype=DTYPE)
        state = mixer.ThrusterState(torch.full_like(actions, 40), torch.full_like(actions, 40))
        result = mixer.step_thrusters(actions, state, self.spec, .1,
                    mixer.ThrusterDegradation(thrust_cap=0, additional_lag_s=1))
        self.assertTrue(torch.equal(result.delivered_thrust, torch.zeros_like(actions)))
        self.assertTrue(torch.equal(result.state.nominal, torch.zeros_like(actions)))

    def test_nominal_backward_euler_step_response(self):
        dt, tau = .01, .2
        spec = blueboat_spec((tau, tau))
        actions = torch.tensor([[1, 0]], dtype=DTYPE)
        state = mixer.zero_state(actions)
        for index in range(1, 101):
            result = mixer.step_thrusters(actions, state, spec, dt)
            state = result.state
            expected = 40 * (1 - (tau / (tau + dt)) ** index)
            torch.testing.assert_close(result.delivered_thrust,
                                       torch.full_like(actions, expected), atol=1e-12, rtol=1e-12)

    def test_additional_lag_is_cascade_not_sum(self):
        dt, tau_nominal, tau_extra = .01, .1, .25
        spec = blueboat_spec((tau_nominal, tau_nominal))
        actions = torch.tensor([[1, 0]], dtype=DTYPE)
        result = mixer.step_thrusters(actions, mixer.zero_state(actions), spec, dt,
                                      mixer.ThrusterDegradation(additional_lag_s=tau_extra))
        expected_first = 40 * dt / (tau_nominal + dt)
        expected_second = expected_first * dt / (tau_extra + dt)
        torch.testing.assert_close(result.delivered_thrust,
                                   torch.full_like(actions, expected_second))
        self.assertNotAlmostEqual(float(result.delivered_thrust[0, 0]),
                                  40 * dt / (tau_nominal + tau_extra + dt))

    def test_per_motor_lag_asymmetry_and_monotonicity(self):
        actions = torch.tensor([[1, 0]], dtype=DTYPE)
        spec = blueboat_spec((.1, .2))
        state = mixer.zero_state(actions)
        previous = state.extra
        for unused in range(200):
            result = mixer.step_thrusters(actions, state, spec, .01,
                                          mixer.ThrusterDegradation(additional_lag_s=(.1, .2)))
            self.assertTrue(bool((result.delivered_thrust >= previous).all()))
            self.assertTrue(bool((result.delivered_thrust <= 40).all()))
            self.assertGreater(float(result.delivered_thrust[0, 0]),
                               float(result.delivered_thrust[0, 1]))
            state, previous = result.state, result.delivered_thrust

    def test_zero_lag_exact_bypass_and_no_state_mutation(self):
        actions = torch.tensor([[.143, -.217]], dtype=DTYPE)
        state = mixer.zero_state(actions)
        result = mixer.step_thrusters(actions, state, self.spec, 1 / 120)
        self.assertTrue(torch.equal(result.commanded_thrust, result.delivered_thrust))
        self.assertTrue(torch.equal(state.nominal, torch.zeros_like(actions)))
        self.assertTrue(torch.equal(state.extra, torch.zeros_like(actions)))

    def test_keyed_sides_balanced_and_order_independent(self):
        keys = torch.arange(1000)
        sides = mixer.balanced_failure_sides(keys, 12345)
        self.assertEqual(int(sides.sum()), 500)
        self.assertTrue(torch.equal(sides[0::2] + sides[1::2], torch.ones(500, dtype=torch.int64)))
        permutation = torch.randperm(1000, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(mixer.balanced_failure_sides(keys[permutation], 12345),
                                    sides[permutation]))
        self.assertTrue(torch.equal(mixer.balanced_failure_sides(keys[91:197], 12345), sides[91:197]))

    def test_c7_one_sided_gain_ladder(self):
        keys = torch.arange(10)
        for gain in (1, .95, .85, .70, .50):
            pairs = mixer.one_sided_motor_gains(gain, keys, 9)
            torch.testing.assert_close(pairs.min(-1).values, torch.full((10,), gain, dtype=DTYPE))
            torch.testing.assert_close(pairs.max(-1).values, torch.ones(10, dtype=DTYPE))
            result = self.step(torch.tensor([[1, 0]], dtype=DTYPE).expand(10, -1),
                               degradation=mixer.ThrusterDegradation(motor_gain=pairs))
            torch.testing.assert_close(result.body_wrench[:, 0],
                                       torch.full((10,), 40 * (1 + gain), dtype=DTYPE))

    def test_vessel_agnostic_asymmetric_limits(self):
        spec = mixer.ThrusterSpec((100, 120), (50, 60), 1.2, 150)
        gen = torch.Generator().manual_seed(23)
        actions = 2 * torch.rand((1000, 2), dtype=DTYPE, generator=gen) - 1
        result = mixer.step_thrusters(actions, mixer.zero_state(actions), spec, .01)
        self.assertTrue(bool((result.delivered_thrust <= torch.tensor([100, 120])).all()))
        self.assertTrue(bool((result.delivered_thrust >= torch.tensor([-50, -60])).all()))
        torch.testing.assert_close(result.body_wrench[:, 0], result.delivered_thrust.sum(-1))
        torch.testing.assert_close(result.body_wrench[:, 5],
                                   1.2 * (result.delivered_thrust[:, 1] - result.delivered_thrust[:, 0]))

    def test_legacy_wamv_rejected_and_registry_drift_caught(self):
        # E2: the legacy registry names are rejected by name (the legacy 'wamv'
        # is the 0.8-scale 100 kg placeholder, not VRX 1:1) ...
        for legacy in ("wamv", "blueboat", "rov"):
            with self.assertRaisesRegex(ValueError, "not a journal"):
                profiles.profile_for_vehicle(legacy)
        # ... and a drifted/inconsistent registry entry fails the D5 validation
        # that every adapter runs (yaw request scale must equal arm*(fwd+rev)).
        spec = registry.get_vehicle_v2("blueboat_v2")
        drifted = replace(spec, propulsion=replace(spec.propulsion, yaw_command_limit_nm=23.0))
        with self.assertRaises(ValueError):
            registry.to_d3_thruster_kwargs(drifted)

    def test_wamv_vrx_classic_envelope(self):
        # Report 6.3/6.5/9.2: 242.67 N forward, 100 N reverse per motor, arm 1.027135 m
        # (registry opt1 = classic GLF at +1: 242.6659 N).
        spec = profiles.profile_for_vehicle("wamv_vrx_v2_opt1", pitch_coupling=True).thrusters
        fwd = spec.forward_limit_n[0]
        self.assertAlmostEqual(fwd, 242.67, delta=0.005)
        result = mixer.step_thrusters(torch.tensor([[1., 0], [-1., 0], [0, 1.], [1., 1.]], dtype=DTYPE),
                                      mixer.zero_state(torch.zeros((4, 2), dtype=DTYPE)), spec, 1/120)
        torch.testing.assert_close(result.body_wrench[:2, 0], torch.tensor([2*fwd, -200.0], dtype=DTYPE))
        torch.testing.assert_close(result.body_wrench[2, 5], torch.tensor(2*100*1.027135, dtype=DTYPE))
        self.assertTrue(bool((result.delivered_thrust <= fwd + 1e-9).all()))
        self.assertTrue(bool((result.delivered_thrust >= -100 - 1e-9).all()))
        # Thrust line about 0.459 m below the COM (propeller z in base_link minus
        # the registry COM z 0.26822): forward thrust gives bow-up (negative) pitch.
        self.assertAlmostEqual(spec.mount_z_m, 0.318237 - 0.509371
                               - registry._data.WAMV["competition"]["com_base_link_m"][2], places=9)
        self.assertLess(float(result.body_wrench[0, 4]), 0.0)
        torch.testing.assert_close(result.body_wrench[:, 4], spec.mount_z_m*result.body_wrench[:, 0])
        # Decision 6: the journal default has no thrust-line pitch coupling.
        self.assertEqual(profiles.profile_for_vehicle("wamv_vrx_v2_opt1").thrusters.mount_z_m, 0.0)
        current = profiles.profile_for_vehicle("wamv_vrx_v2_opt2").thrusters
        self.assertAlmostEqual(current.forward_limit_n[0], 2353.53, places=1)
        self.assertEqual(current.reverse_limit_n, (1000.0, 1000.0))

    def test_prepared_step_is_bit_identical_to_validating_step(self):
        # E2: the runtime calls step_thrusters_prepared with cached parameters.
        gen = torch.Generator().manual_seed(611)
        actions = 2 * torch.rand((257, 2), dtype=DTYPE, generator=gen) - 1
        spec = blueboat_spec((.05, .1))
        degradation = mixer.ThrusterDegradation(torch.rand((257, 1), dtype=DTYPE, generator=gen),
                                                torch.rand((257, 2), dtype=DTYPE, generator=gen), .25)
        state = mixer.ThrusterState(torch.randn((257, 2), dtype=DTYPE, generator=gen) * 30,
                                    torch.randn((257, 2), dtype=DTYPE, generator=gen) * 30)
        for policy in ("radial", "clip"):
            a = mixer.step_thrusters(actions, state, spec, 1 / 120, degradation, policy)
            params = mixer.prepare_thruster_parameters(spec, degradation, actions)
            b = mixer.step_thrusters_prepared(actions, state, spec, 1 / 120, params, policy)
            for x, y in ((a.body_wrench, b.body_wrench), (a.delivered_thrust, b.delivered_thrust),
                         (a.state.nominal, b.state.nominal), (a.state.extra, b.state.extra),
                         (a.saturation_scale, b.saturation_scale)):
                self.assertTrue(torch.equal(x, y))

    def test_zero_mount_height_gives_exact_zero_pitch(self):
        result = self.step(2*torch.rand((64, 2), dtype=DTYPE, generator=torch.Generator().manual_seed(5))-1)
        self.assertTrue(torch.equal(result.body_wrench[:, 1:5], torch.zeros((64, 4), dtype=DTYPE)))

    def test_input_validation(self):
        for invalid in ([[1.1, 0]], [[0, float("nan")]], [[0, float("inf")]]):
            with self.assertRaises(ValueError):
                self.step(invalid)
        for kwargs in ({"thrust_cap": 1.1}, {"thrust_cap": -1},
                       {"motor_gain": (1, 1.1)}, {"additional_lag_s": -.1}):
            with self.assertRaises(ValueError):
                self.step([[0, 0]], degradation=mixer.ThrusterDegradation(**kwargs))
        with self.assertRaises(ValueError):
            self.step([[0, 0]], saturation="unknown")
        with self.assertRaises(ValueError):
            mixer.balanced_failure_sides(torch.tensor([1.5]))
        with self.assertRaises(ValueError):
            mixer.balanced_failure_sides(torch.tensor([-1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
