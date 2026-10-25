"""CPU-only acceptance tests; invoke with python -B -m unittest discover.

Imports are by file path, so neither tasks/__init__.py nor Isaac Lab runs.
Assertions use physical examples and stream invariants, not simulator mocks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


sys.dont_write_bytecode = True
CONDITIONS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "_condition_test_" + name, CONDITIONS / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


actuator_module = load("actuator")
perception_module = load("perception")
ActuatorCondition = actuator_module.ActuatorCondition
PerceptionCondition = perception_module.PerceptionCondition
Group = perception_module.ObsChannelGroup


def actuator(**overrides):
    values = dict(
        enabled=True, mass_scale=1.0, drag_scale=1.0,
        thrust_cap_scale=1.0, motor_tau_s=0.0, thrust_imbalance=0.0,
        payload_mass_kg=0.0, payload_offset_x_m=0.0, payload_offset_y_m=0.0,
    )
    values.update(overrides)
    return ActuatorCondition(SimpleNamespace(**values), 2, "cpu")


def perception(*, seed=7, groups=None, **overrides):
    values = dict(enabled=True, delay_steps=0, dropout_p=0.0)
    values.update(overrides)
    if groups is None:
        groups = {"pos": Group(2, (0.0, 0.0), (0.0, 0.0))}
    return PerceptionCondition(
        SimpleNamespace(**values), 2, "cpu", seed, groups
    )


class ActuatorTests(unittest.TestCase):
    def test_disabled_is_object_identity_even_with_non_neutral_parameters(self):
        model = actuator(
            enabled=False, mass_scale=3.0, drag_scale=4.0,
            thrust_cap_scale=0.2, thrust_imbalance=0.5, motor_tau_s=1.0,
            payload_mass_kg=10.0, payload_offset_x_m=0.4,
        )
        action = torch.tensor([[0.2, 0.3], [-0.5, -0.9]])
        force = torch.arange(12.0).reshape(2, 2, 3)
        torque = force + 2.0
        thrust, yaw = action[:, 0], action[:, 1]
        self.assertIs(model.action_in(action), action)
        got_thrust, got_yaw = model.command(thrust, yaw, 0.1, 0.4)
        self.assertIs(got_thrust, thrust)
        self.assertIs(got_yaw, yaw)
        self.assertIs(model.scale_drag(force), force)
        got_force, got_torque = model.dynamics(force, torque, 10.0, 4.0)
        self.assertIs(got_force, force)
        self.assertIs(got_torque, torque)

    def test_enabled_neutral_is_object_identity(self):
        model = actuator()
        thrust, yaw = torch.tensor([1.1, -2.3]), torch.tensor([3.1, -0.4])
        force, torque = torch.ones(2, 1, 3), torch.ones(2, 1, 3)
        result = model.command(thrust, yaw, 1.0 / 120.0, 0.4495)
        self.assertIs(result[0], thrust)
        self.assertIs(result[1], yaw)
        self.assertIs(model.scale_drag(force), force)
        result = model.dynamics(force, torque, 17.26, 3.0)
        self.assertIs(result[0], force)
        self.assertIs(result[1], torque)

    def test_cap_imbalance_and_lag_order_in_physical_units(self):
        model = actuator(thrust_cap_scale=0.5, thrust_imbalance=0.2, motor_tau_s=0.3)
        thrust, yaw = torch.tensor([10.0, -4.0]), torch.tensor([2.0, -1.0])
        first = model.command(thrust, yaw, 0.1, 0.5)
        torch.testing.assert_close(first[0], torch.tensor([1.45, -0.6]))
        torch.testing.assert_close(first[1], torch.tensor([0.625, -0.3]))
        second = model.command(thrust, yaw, 0.1, 0.5)
        torch.testing.assert_close(second[0], torch.tensor([2.5375, -1.05]))
        torch.testing.assert_close(second[1], torch.tensor([1.09375, -0.525]))
        torch.testing.assert_close(thrust, torch.tensor([10.0, -4.0]))
        torch.testing.assert_close(yaw, torch.tensor([2.0, -1.0]))

    def test_imbalance_turns_a_straight_command_and_reverses_with_thrust(self):
        model = actuator(thrust_imbalance=0.25)
        thrust, yaw = model.command(
            torch.tensor([8.0, -8.0]), torch.zeros(2), 0.01, 0.4
        )
        torch.testing.assert_close(thrust, torch.tensor([8.0, -8.0]))
        torch.testing.assert_close(yaw, torch.tensor([0.8, -0.8]))

    def test_motor_partial_reset_leaves_other_environment_running(self):
        model = actuator(motor_tau_s=0.1)
        thrust, yaw = torch.full((2,), 10.0), torch.full((2,), 4.0)
        model.command(thrust, yaw, 0.1, 0.5)
        model.reset(torch.tensor([0]))
        output = model.command(thrust, yaw, 0.1, 0.5)
        torch.testing.assert_close(output[0], torch.tensor([5.0, 7.5]))
        torch.testing.assert_close(output[1], torch.tensor([2.0, 3.0]))
        model.reset(torch.tensor([], dtype=torch.long))
        output = model.command(thrust, yaw, 0.1, 0.5)
        torch.testing.assert_close(output[0], torch.tensor([7.5, 8.75]))

    def test_planar_and_yaw_drag_receive_same_scalar(self):
        model = actuator(drag_scale=2.5)
        value = torch.tensor([[-2.0, 4.0], [0.0, -1.0]])
        torch.testing.assert_close(
            model.scale_drag(value), torch.tensor([[-5.0, 10.0], [0.0, -2.5]])
        )
        self.assertEqual(float(value[0, 0]), -2.0)

    def test_mass_scales_all_force_and_torque_axes(self):
        model = actuator(mass_scale=2.0)
        force = torch.arange(12.0).reshape(2, 2, 3)
        torque = force + 3.0
        output = model.dynamics(force, torque, 10.0, 4.0)
        torch.testing.assert_close(output[0], force * 0.5)
        torch.testing.assert_close(output[1], torque * 0.5)

    def test_offset_payload_and_mass_compose_with_analytical_planar_model(self):
        model = actuator(
            mass_scale=2.0, payload_mass_kg=5.0,
            payload_offset_x_m=0.4, payload_offset_y_m=0.3,
        )
        force = torch.tensor([[[10.0, -5.0, 2.0]], [[-10.0, 5.0, -2.0]]])
        torque = torch.tensor([[[6.0, 8.0, 3.0]], [[6.0, 8.0, -3.0]]])
        force_before, torque_before = force.clone(), torque.clone()
        output = model.dynamics(force, torque, 10.0, 4.0)
        torch.testing.assert_close(
            output[0], torch.tensor([[[4.0, -2.0, 0.8]], [[-4.0, 2.0, -0.8]]])
        )
        torch.testing.assert_close(
            output[1], torch.tensor([[[3.0, 4.0, 16.0 / 9.0]], [[3.0, 4.0, -16.0 / 9.0]]])
        )
        self.assertTrue(torch.equal(force, force_before))
        self.assertTrue(torch.equal(torque, torque_before))

    def test_centered_payload_changes_force_but_preserves_yaw_inertia(self):
        model = actuator(payload_mass_kg=10.0)
        force, torque = torch.ones(2, 2, 3), torch.ones(2, 2, 3)
        output = model.dynamics(force, torque, 10.0, 4.0)
        torch.testing.assert_close(output[0], force * 0.5)
        torch.testing.assert_close(output[1], torque)

    def test_invalid_physics_references_fail_clearly(self):
        model = actuator()
        for dt in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                model.command(torch.ones(2), torch.ones(2), dt, 0.5)
        for mass, inertia in ((0.0, 1.0), (1.0, -1.0), (1.0, float("nan"))):
            with self.subTest(mass=mass, inertia=inertia), self.assertRaises(ValueError):
                model.dynamics(torch.ones(2, 1, 3), torch.ones(2, 1, 3), mass, inertia)


class PerceptionTests(unittest.TestCase):
    def test_disabled_and_neutral_are_object_identity_and_do_not_draw_rng(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        global_state = torch.random.get_rng_state().clone()
        disabled = perception(
            enabled=False, delay_steps=3, dropout_p=1.0,
            groups={"pos": Group(2, (5.0, 5.0), (2.0, 2.0))},
        )
        neutral = perception()
        for model in (disabled, neutral):
            model.reset(torch.tensor([0, 1]))
            self.assertFalse(model.wants("pos"))
            self.assertIs(model.apply("pos", x), x)
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

    def test_bias_is_episode_constant_and_two_seeds_are_distinct(self):
        groups = {"pos": Group(2, (0.0, 0.0), (0.4, 0.4))}
        a, b = perception(seed=17, groups=groups), perception(seed=31, groups=groups)
        x = torch.zeros(2, 2)
        for model in (a, b):
            model.reset(torch.tensor([0, 1]))
        a0, b0 = a.apply("pos", x), b.apply("pos", x)
        self.assertFalse(torch.equal(a0, b0))
        torch.testing.assert_close(a.apply("pos", x + 2.0) - 2.0, a0)
        self.assertTrue(torch.equal(x, torch.zeros(2, 2)))

    def test_seed_reproducibility_noise_changes_and_global_rng_is_untouched(self):
        groups = {"pos": Group(2, (0.3, 0.3), (0.2, 0.2))}
        a, b = perception(seed=101, groups=groups), perception(seed=101, groups=groups)
        before = torch.random.get_rng_state().clone()
        a.reset(torch.tensor([0, 1]))
        b.reset(torch.tensor([0, 1]))
        x = torch.ones(2, 2)
        first = a.apply("pos", x)
        self.assertTrue(torch.equal(first, b.apply("pos", x)))
        second = a.apply("pos", x)
        self.assertTrue(torch.equal(second, b.apply("pos", x)))
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_delay_uses_clean_initial_frame_then_exact_delayed_measurement(self):
        model = perception(delay_steps=2)
        model.reset(torch.tensor([0, 1]))
        for step, expected in enumerate((10.0, 10.0, 10.0, 11.0, 12.0)):
            out = model.apply("pos", torch.full((2, 2), 10.0 + step))
            torch.testing.assert_close(out, torch.full((2, 2), expected))

    def test_full_dropout_holds_clean_initial_then_partial_reset_refreshes(self):
        model = perception(dropout_p=1.0)
        model.reset(torch.tensor([0, 1]))
        initial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(torch.equal(model.apply("pos", initial), initial))
        self.assertTrue(torch.equal(model.apply("pos", initial + 10.0), initial))
        model.reset(torch.tensor([0]))
        out = model.apply("pos", initial + 20.0)
        self.assertTrue(torch.equal(out[0], initial[0] + 20.0))
        self.assertTrue(torch.equal(out[1], initial[1]))

    def test_partial_reset_does_not_disturb_other_environment_stream(self):
        groups = {"pos": Group(2, (0.3, 0.3), (0.1, 0.1))}
        a, b = perception(groups=groups), perception(groups=groups)
        x = torch.ones(2, 2)
        for model in (a, b):
            model.reset(torch.tensor([0, 1]))
            model.apply("pos", x)
        a.reset(torch.tensor([0]))
        result_a, result_b = a.apply("pos", x), b.apply("pos", x)
        self.assertTrue(torch.equal(result_a[1], result_b[1]))
        self.assertFalse(torch.equal(result_a[0], result_b[0]))

    def test_group_call_order_is_independent_and_clamps_are_physical(self):
        groups = {
            "a": Group(2, (0.2, 0.2), (0.0, 0.0)),
            "rays": Group(2, (0.1, 0.1), (0.0, 0.0), 0.0, 10.0),
        }
        a, b = perception(groups=groups), perception(groups=groups)
        x, rays = torch.ones(2, 2), torch.tensor([[-100.0, 100.0]] * 2)
        a.reset(torch.tensor([0, 1]))
        b.reset(torch.tensor([0, 1]))
        a_pos, a_rays = a.apply("a", x), a.apply("rays", rays)
        b_rays, b_pos = b.apply("rays", rays), b.apply("a", x)
        self.assertTrue(torch.equal(a_pos, b_pos))
        self.assertTrue(torch.equal(a_rays, b_rays))
        torch.testing.assert_close(a_rays, torch.tensor([[0.0, 10.0]] * 2))

    def test_unknown_group_and_invalid_temporal_domains_fail(self):
        with self.assertRaises(KeyError):
            perception().apply("typo", torch.ones(2, 2))
        for kwargs in ({"delay_steps": -1}, {"delay_steps": 1.5}, {"dropout_p": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                perception(**kwargs)


if __name__ == "__main__":
    unittest.main()
