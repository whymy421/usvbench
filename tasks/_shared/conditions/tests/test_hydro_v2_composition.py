"""E3: hydro_v2 composition path of the condition layer (CPU only).

Run: python -B tasks/_shared/conditions/tests/test_hydro_v2_composition.py
Covers brief work item 1 and tests: zero-dose identity (fake and real
runtime), current as water velocity only, wave source parameters and keyed
realisation, wind vs a directly driven C3 Wind with D5 windage, actuator
mapping, ordering guards and schema-3 stamps.
"""
from __future__ import annotations

import math
import sys
import unittest

sys.dont_write_bytecode = True
import torch

import e3_fixtures as F

api = F.api
ALL_D0 = ("journal.waves.D0", "journal.current.D0", "journal.wind.D0",
          "journal.actuator.cap.D0", "journal.actuator.gain.D0", "journal.actuator.lag.D0",
          "journal.actuator.resistance.D0", "journal.perception.noise.D0",
          "journal.perception.bias.D0", "journal.perception.latency.D0",
          "journal.perception.dropout.D0")


def random_state(generator, n, prof):
    z = prof.hydro.equilibrium_com_z_m
    position = torch.randn((n, 3), generator=generator) * 2.0
    position[:, 2] = z + 0.02 * torch.randn(n, generator=generator)
    velocity = torch.randn((n, 6), generator=generator) * 0.5
    angles = torch.randn((n, 3), generator=generator) * 0.1
    half = angles / 2
    cr, sr, cp, sp, cy, sy = (torch.cos(half[:, 0]), torch.sin(half[:, 0]), torch.cos(half[:, 1]),
                              torch.sin(half[:, 1]), torch.cos(half[:, 2]), torch.sin(half[:, 2]))
    quat = torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)
    actions = (torch.rand((n, 2), generator=generator) * 2 - 1)
    return position, velocity, quat, actions


def drive(layer, rt, steps, seed=3, resets=((6, (2,)),), use_layer=True):
    """Run substeps; return the list of runtime outputs (real) or calls (fake)."""
    generator = torch.Generator().manual_seed(seed)
    n = rt.num_envs
    origins = torch.zeros((n, 3))
    outputs = []
    reset_at = dict(resets)
    for k in range(steps):
        if k in reset_at:
            ids = torch.as_tensor(reset_at[k])
            rt.reset(ids)
            if use_layer:
                layer.reset_runtime(ids, reference_axis_rad=torch.full((ids.numel(),), 0.3, dtype=torch.float64),
                                    task_episode_ids=F.task_episode_ids(layer, ids))   # F0 / D7
                layer.configure_runtime(rt, ids)
        position, velocity, quat, actions = random_state(generator, n, rt.profile)
        kwargs = layer.physics_inputs(velocity, quat).as_kwargs() if use_layer else {}
        out = rt.step(actions, position, velocity, quat, origins, **kwargs)
        outputs.append(out)
        for group in ("pos", "vel"):
            # drawn in every run so that all runs see the same state sequence
            x = torch.randn((n, 2 if group == "pos" else 3), generator=generator)
            if use_layer:
                y = layer.observation_out(group, x)
                assert y is x, "zero-dose perception must return the input object"
    return outputs


def calls_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if type(x) is not type(y):
            return False
        if isinstance(x, tuple):
            if not calls_equal(x, y):
                return False
        elif isinstance(x, torch.Tensor):
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                return False
        elif x is None or isinstance(x, (int, float, str, list)):
            if x != y:
                return False
        else:  # WaveSample
            if not calls_equal(tuple(vars(x).values()), tuple(vars(y).values())):
                return False
    return True


class ZeroDoseIdentity(unittest.TestCase):
    def test_disabled_equals_dose_zero_through_fake_runtime(self):
        before = torch.random.get_rng_state().clone()
        runs = []
        for presets in ((), ALL_D0):
            layer, rt = F.build(presets, n=5)
            F.reset(layer, rt, axis=torch.full((5,), 0.3, dtype=torch.float64))
            drive(layer, rt, 12)
            runs.append(rt.calls)
            self.assertIsNone(layer.wave_source())
        self.assertTrue(calls_equal(runs[0], runs[1]))
        steps = [c for c in runs[0] if c[0] == "step"]
        self.assertEqual(len(steps), 12)
        for call in steps:
            self.assertTrue(torch.equal(call[2], torch.zeros(5, 3)))
            self.assertIsNone(call[3]); self.assertIsNone(call[4]); self.assertIsNone(call[5])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_disabled_equals_dose_zero_equals_no_layer_through_real_runtime(self):
        results = []
        for presets in ((), ALL_D0, None):
            prof = F.profile("blueboat_v2")
            rt = F.real_runtime(prof, 5)
            layer = None
            if presets is not None:
                cfg = F.env_cfg(condition_presets=presets)
                layer = api.build_conditions(cfg, 5, "cpu", family="station_keeping")
                layer.bind_runtime(rt)
                F.reset(layer, rt, axis=torch.full((5,), 0.3, dtype=torch.float64))
            outs = drive(layer, rt, 12, use_layer=presets is not None)
            results.append(torch.stack([torch.cat((o.force_actor, o.torque_actor), -1) for o in outs]))
        self.assertTrue(torch.equal(results[0], results[1]))
        self.assertTrue(torch.equal(results[0], results[2]))
        self.assertTrue(bool(torch.isfinite(results[0]).all()))
        self.assertGreater(float(results[0].abs().max()), 0.0)


class CurrentIsWaterVelocity(unittest.TestCase):
    def test_current_velocity_world_values_and_zero_z(self):
        layer, rt = F.build(("journal.current.D3",), n=8)
        axis = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
        F.reset(layer, rt, axis=axis)
        u = layer.current_velocity_world()
        world = axis + F.strata.relative_angles(layer.stratum)
        expected = torch.stack((torch.cos(world), torch.sin(world)), -1).to(torch.float32) * 1.0
        self.assertTrue(torch.equal(u[:, :2], expected))
        self.assertTrue(torch.equal(u[:, 2], torch.zeros(8)))
        self.assertEqual(u.dtype, rt.dtype)

    def test_speed_table_is_evaluated_at_the_runtime_tick_clock(self):
        layer, rt = F.build((), {"current": {"enabled": True, "speed_table": ((0.0, 0.0), (1.0, 1.0)),
                                             "direction_mode": "fixed", "direction_rad": 0.0}}, n=3)
        F.reset(layer, rt)
        quat = F.quaternion_yaw(torch.zeros(3))
        for k in range(30):
            inputs = layer.physics_inputs(torch.zeros(3, 6), quat)
            expected = rt.time().clamp(0, 1)
            self.assertTrue(torch.allclose(inputs.current_velocity_w[:, 0], expected, atol=1e-7, rtol=0))
            rt.step(torch.zeros(3, 2), torch.zeros(3, 3), torch.zeros(3, 6), quat, torch.zeros(3, 3),
                    **inputs.as_kwargs())
        rt.reset([1]); layer.reset_runtime([1], reference_axis_rad=torch.zeros(1),
                                           task_episode_ids=F.task_episode_ids(layer, [1]))   # F0 / D7
        layer.configure_runtime(rt, [1])
        u = layer.current_velocity_world()
        self.assertEqual(float(u[1, 0]), 0.0)
        self.assertAlmostEqual(float(u[0, 0]), 30 * F.DT, places=6)

    def test_legacy_current_wrench_and_legacy_hooks_are_never_used(self):
        current_module = F.module("conditions.current")
        original = current_module.CurrentCondition.wrench
        def forbidden(*args, **kwargs):
            raise AssertionError("legacy current wrench called under hydro_v2")
        current_module.CurrentCondition.wrench = forbidden
        sea_forces = F.sea_state.SeaState.forces
        F.sea_state.SeaState.forces = forbidden
        try:
            layer, rt = F.build(("journal.combo.wave_current.D3",), n=4, runtime="real")
            F.reset(layer, rt)
            drive(layer, rt, 5, resets=())
        finally:
            current_module.CurrentCondition.wrench = original
            F.sea_state.SeaState.forces = sea_forces
        for name in ("command", "scale_drag", "environmental_wrench", "dynamics", "hull_damping_scales"):
            with self.assertRaises(RuntimeError):
                getattr(layer, name)()

    def test_hull_drifting_with_the_current_feels_no_horizontal_damping(self):
        layer, rt = F.build((), {"current": {"enabled": True, "speed_mps": 1.0, "direction_mode": "fixed",
                                             "direction_rad": 0.7}}, n=2, runtime="real")
        F.reset(layer, rt)
        z = rt.profile.hydro.equilibrium_com_z_m
        position = torch.tensor([[0.0, 0.0, z], [0.0, 0.0, z]])
        velocity = torch.zeros(2, 6)
        velocity[0, 0], velocity[0, 1] = math.cos(0.7), math.sin(0.7)   # co-moving hull
        quat = F.quaternion_yaw(torch.tensor([0.7, 0.7]))
        inputs = layer.physics_inputs(velocity, quat)
        out = rt.step(torch.zeros(2, 2), position, velocity, quat, torch.zeros(2, 3), **inputs.as_kwargs())
        damping = out.hydro.damping_force_body
        self.assertLess(float(damping[0, :2].abs().max()), 1e-5)
        # hull at rest in the same current: drag pushes it downstream (+x of the hull)
        self.assertGreater(float(damping[1, 0]), 1.0)
        self.assertLess(float(damping[1, 1].abs()), 1e-4)


class WaveSource(unittest.TestCase):
    def test_band_components_spreading_gamma_and_hs(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            layer, rt = F.build(("journal.waves.D4",), n=6, vessel=vessel)
            axis = torch.linspace(0.0, 2.0, 6, dtype=torch.float64)
            F.reset(layer, rt, axis=axis)
            waves = layer.wave_source()
            self.assertIsInstance(waves, F.hydro_v2.SpectralWaves)
            sea = layer.sea
            self.assertEqual(sea.physics_version, 2)
            resolved = layer.context["resolved"]["waves"]
            fp = 0.25
            required = F.hydro_v2.required_high_frequency_hz(rt.profile.hydro, fp) * 4.0
            self.assertEqual(resolved["band_max_over_fp"], math.ceil(required * 100 - 1e-9) / 100)
            self.assertGreaterEqual(resolved["band_max_over_fp"], 4.0)
            self.assertEqual(F.hydro_v2_runtime.check_wave_band(rt.profile, layer.sea_cfg), required)
            self.assertEqual(sea.amplitude.shape[1], resolved["n_components"])
            self.assertGreaterEqual(resolved["n_components"], 1024)
            self.assertGreaterEqual(resolved["group_repeat_s"], 480.0)
            lo, hi = 0.2 * fp, resolved["band_max_over_fp"] * fp
            self.assertTrue(bool((sea.freqs > lo).all()) and bool((sea.freqs < hi).all()))
            hs = 4.0 * torch.sqrt((sea.amplitude ** 2 / 2).sum(-1))
            torch.testing.assert_close(hs, torch.full((6,), 1.0), rtol=1e-5, atol=0)
            self.assertTrue(torch.equal(sea.gamma, torch.full((6,), 3.3)))
            world = (axis + F.strata.relative_angles(layer.stratum)).to(torch.float32)
            self.assertTrue(torch.equal(sea.mean_direction, world))
            offset = sea.direction - sea.mean_direction[:, None]
            self.assertLessEqual(float(offset.abs().max()), math.pi / 4 + 1e-6)
            for name in ("amplitude", "omega", "k", "direction", "phase"):
                self.assertIs(getattr(waves, name), getattr(sea, name))

    def test_band_below_the_vessel_requirement_is_rejected(self):
        override = {"waves": {"enabled": True, "hs_m": 0.25, "tp_s": 4.0, "band_policy": "fixed"}}
        with self.assertRaises(ValueError):
            F.build((), override, n=2)

    def test_realisation_keyed_independent_of_reset_order_and_batching(self):
        seed, n = 17, 6
        a, rta = F.build(("journal.waves.D3",), n=n, seed=seed)
        F.reset(a, rta)
        b, rtb = F.build(("journal.waves.D3",), n=n, seed=seed)
        for batch in ([4, 1], [5, 0, 3], [2]):
            F.reset(b, rtb, ids=batch)
        names = ("hs", "tp", "gamma", "mean_direction", "amplitude", "phase", "direction", "freqs", "omega", "k")
        for name in names:
            self.assertTrue(torch.equal(getattr(a.sea, name), getattr(b.sea, name)), name)
        # second episode of a subset only changes that subset
        before = a.sea.phase.clone()
        F.reset(a, rta, ids=[2])
        self.assertFalse(torch.equal(a.sea.phase[2], before[2]))
        self.assertTrue(torch.equal(a.sea.phase[[0, 1, 3, 4, 5]], before[[0, 1, 3, 4, 5]]))
        # directly driven C2 SeaState with the same keyed ScenarioRNG
        scenario = F.scenario_rng.ScenarioRNG(num_envs=n, device="cpu", eval_seed=seed)
        with torch.random.fork_rng(devices=[]):
            ref = F.sea_state.SeaState(a.sea_cfg, n, torch.device("cpu"), physics_version=2)
        scenario.reset_idx(torch.arange(n))
        world = F.strata.relative_angles(F.strata.stratum_indices(torch.arange(n), seed))
        ref.resample(torch.arange(n), scenario, mean_direction_rad=world)
        scenario.reset_idx(torch.tensor([2]))
        ref.resample(torch.tensor([2]), scenario,
                     mean_direction_rad=F.strata.relative_angles(F.strata.stratum_indices(torch.tensor([n + 2]), seed)))
        for name in names:
            self.assertTrue(torch.equal(getattr(ref, name), getattr(a.sea, name)), name)

    def test_hs_zero_is_calm_and_waves_off_is_calm(self):
        for presets in (("journal.waves.D0",), ()):
            layer, rt = F.build(presets, n=2)
            F.reset(layer, rt)
            self.assertIsNone(layer.wave_source())
            self.assertIsNone(layer.sea)


class Wind(unittest.TestCase):
    def direct_wind(self, layer, vessel):
        spec = F.vehicles_v2.get_vehicle_v2(vessel)
        u10 = layer.cfg.wind.u10_mps
        cfg = F.wind_backend.WindCfg(
            enable=True, mean_speed_mps=F.vehicles_v2.vessel_wind_speed_from_u10(spec, u10),
            reference_height_m=spec.windage.lateral_centroid_height_above_water_m,
            gust_intensity=0.15, gust_correlation_time_s=2.0,
            air_density_kg_m3=spec.windage.air_density_kg_m3,
            windage=F.wind_backend.VesselWindage(**F.vehicles_v2.to_c3_windage_kwargs(spec)))
        return F.wind_backend.Wind(cfg, layer.num_envs, "cpu", eval_seed=layer.cfg.seed,
                                   rng_protocol=F.wind_backend.RNG_PROTOCOL_V2)

    def test_layer_wrench_equals_directly_driven_c3_wind_with_d5_windage(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            n = 5
            layer, rt = F.build(("journal.wind.D4",), n=n, vessel=vessel, seed=9)
            axis = torch.linspace(-0.5, 2.5, n, dtype=torch.float64)
            F.reset(layer, rt, axis=axis)
            direct = self.direct_wind(layer, vessel)
            world = axis + F.strata.relative_angles(layer.stratum)
            direct.reset(torch.arange(n), torch.zeros(n, dtype=torch.long), direction_rad=world)
            generator = torch.Generator().manual_seed(1)
            for step in range(40):
                velocity = torch.randn((n, 6), generator=generator)
                yaw = torch.rand(n, generator=generator) * 6.0 - 3.0
                quat = F.quaternion_yaw(yaw)
                force, torque = layer.wind_wrench_world(velocity, quat)
                heading = layer.heading_rad(quat)
                ref_f, ref_t = direct.forces(velocity[:, :2], heading)
                direct.step(F.DT)
                self.assertTrue(torch.equal(force, ref_f)); self.assertTrue(torch.equal(torque, ref_t))
                rt.step(torch.zeros(n, 2), torch.zeros(n, 3), velocity, quat, torch.zeros(n, 3))
            spec = F.vehicles_v2.get_vehicle_v2(vessel)
            z0 = spec.windage.roughness_length_m
            z = spec.windage.lateral_centroid_height_above_water_m
            self.assertAlmostEqual(layer.wind.cfg.mean_speed_mps, 12.0 * math.log(z / z0) / math.log(10 / z0), places=12)
            layer.episode_stamp(0)   # tick mirror consistent with the runtime

    def test_heel_lever_is_about_the_com(self):
        layer, rt = F.build(("journal.wind.D3",), n=1, seed=0)
        F.reset(layer, rt)
        spec = F.vehicles_v2.get_vehicle_v2("blueboat_v2")
        self.assertEqual(layer.wind.cfg.windage.centroid_height_m, spec.windage.roll_lever_com_m)
        self.assertNotEqual(spec.windage.roll_lever_com_m, spec.windage.lateral_centroid_height_above_water_m)
        quat = F.quaternion_yaw(torch.tensor([0.4]))
        force, torque = layer.wind_wrench_world(torch.zeros(1, 6), quat)
        c, s = math.cos(0.4), math.sin(0.4)
        sway = -s * float(force[0, 0]) + c * float(force[0, 1])
        roll = -sway * spec.windage.roll_lever_com_m
        self.assertAlmostEqual(float(torque[0, 0]), c * roll, places=5)
        self.assertAlmostEqual(float(torque[0, 1]), s * roll, places=5)

    def test_zero_dose_is_module_off(self):
        layer, rt = F.build(("journal.wind.D0",), n=2)
        F.reset(layer, rt)
        self.assertIsNone(layer.wind)
        self.assertEqual(layer.wind_wrench_world(torch.zeros(2, 6), F.quaternion_yaw(torch.zeros(2))), (None, None))


class Actuator(unittest.TestCase):
    def test_cap_limits_both_directions(self):
        layer, rt = F.build(("journal.actuator.cap.D3",), n=2)
        F.reset(layer, rt)
        deg = rt.degradation
        self.assertTrue(torch.equal(deg.thrust_cap, torch.full((2, 1), 0.5)))
        spec = rt.profile.thrusters
        state = F.thruster_mixer.ThrusterState(torch.zeros(2, 2), torch.zeros(2, 2))
        out = F.thruster_mixer.step_thrusters(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), state, spec, F.DT, deg)
        torch.testing.assert_close(out.delivered_thrust[0], torch.full((2,), 0.5 * spec.forward_limit_n[0]))
        torch.testing.assert_close(out.delivered_thrust[1], torch.full((2,), -0.5 * spec.reverse_limit_n[0]))

    def test_one_sided_gain_side_balance_over_paired_ordinals(self):
        n = 64
        layer, rt = F.build(("journal.actuator.gain.D4",), n=n, seed=123)
        for episode in range(2):
            F.reset(layer, rt)
            gain = rt.degradation.motor_gain
            failing = (gain < 1.0)
            self.assertTrue(bool((failing.sum(-1) == 1).all()))
            self.assertTrue(torch.equal(gain[failing], torch.full((n,), 0.5)))
            self.assertTrue(torch.equal(gain[~failing], torch.ones(n)))
            port = failing[:, 0]
            self.assertEqual(int(port.sum()), 32)
            self.assertTrue(bool((port[0::2] != port[1::2]).all()))   # pairs (2j, 2j+1) opposite
            sides = F.thruster_mixer.balanced_failure_sides(torch.as_tensor(layer.ordinal), 123)
            self.assertTrue(torch.equal(port.to(torch.long), (sides == 0).to(torch.long)))
        # asynchronous reset: untouched rows keep their latched sides
        before = rt.degradation.motor_gain.clone()
        F.reset(layer, rt, ids=[5, 6])
        after = rt.degradation.motor_gain
        keep = [i for i in range(n) if i not in (5, 6)]
        self.assertTrue(torch.equal(before[keep], after[keep]))

    def test_lag_and_resistance_mapping(self):
        layer, rt = F.build(("journal.actuator.lag.D3", "journal.actuator.resistance.D4"), n=3, runtime="real")
        F.reset(layer, rt)
        self.assertTrue(torch.equal(rt.degradation.additional_lag_s, torch.full((3, 1), 0.25)))
        self.assertTrue(torch.equal(rt.resistance_scale, torch.full((3,), 2.0)))
        z = rt.profile.hydro.equilibrium_com_z_m
        quat = F.quaternion_yaw(torch.zeros(3))
        out = rt.step(torch.tensor([[1.0, 0.0]] * 3), torch.tensor([[0, 0, z]] * 3), torch.zeros(3, 6), quat,
                      torch.zeros(3, 3), **layer.physics_inputs(torch.zeros(3, 6), quat).as_kwargs())
        target = rt.profile.thrusters.forward_limit_n[0]
        expected = F.DT * target / (0.25 + F.DT)       # one backward-Euler step of the extra pole
        torch.testing.assert_close(out.motors.delivered_thrust, torch.full((3, 2), expected), rtol=1e-6, atol=0)

    def test_disabled_actuator_configures_neutral_tensors(self):
        layer, rt = F.build((), n=3)
        F.reset(layer, rt)
        deg = rt.degradation
        self.assertTrue(torch.equal(deg.thrust_cap, torch.ones(3, 1)))
        self.assertTrue(torch.equal(deg.motor_gain, torch.ones(3, 2)))
        self.assertTrue(torch.equal(deg.additional_lag_s, torch.zeros(3, 1)))
        self.assertTrue(torch.equal(rt.resistance_scale, torch.ones(3)))


class Guards(unittest.TestCase):
    def test_ordering_and_clock_guards(self):
        layer, rt = F.build(("journal.wind.D2",), n=2)
        with self.assertRaises(RuntimeError):
            layer.current_velocity_world()                  # never reset
        rt.reset(); layer.reset_runtime(torch.arange(2), reference_axis_rad=torch.zeros(2),
                                        task_episode_ids=F.task_episode_ids(layer, torch.arange(2)))  # F0 / D7
        with self.assertRaises(RuntimeError):
            layer.physics_inputs(torch.zeros(2, 6), F.quaternion_yaw(torch.zeros(2)))   # not configured
        layer.configure_runtime(rt, torch.arange(2))
        quat = F.quaternion_yaw(torch.zeros(2))
        rt.step(torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(2, 6), quat, torch.zeros(2, 3))
        with self.assertRaises(RuntimeError):
            layer.episode_stamp(0)                          # wind not stepped with the runtime
        with self.assertRaises(RuntimeError):
            layer.configure_runtime(rt, [0])                # runtime tick not reset
        with self.assertRaises(RuntimeError):
            layer.bind_runtime(rt)
        other, _ = F.build((), n=2)
        with self.assertRaises(ValueError):
            other_rt = F.FakeRuntime(F.profile("wamv_vrx_v2_opt1"), 2)
            layer2 = api.build_conditions(F.env_cfg(), 2, "cpu", family="station_keeping")
            layer2.bind_runtime(other_rt)                   # profile is not the task vessel
        with self.assertRaises(ValueError):
            layer2.bind_runtime(F.FakeRuntime(F.profile(), 3))
        with self.assertRaises(ValueError):
            layer, rt = F.build(("journal.current.D2",), n=2)
            rt.reset(); layer.reset_runtime(torch.arange(2),  # stratified direction needs the task axis
                                            task_episode_ids=F.task_episode_ids(layer, torch.arange(2)))

    def test_task_episode_counter_is_cross_checked(self):
        layer, rt = F.build(("journal.current.D1",), n=3)
        rt.reset()
        with self.assertRaises(ValueError):
            layer.reset_runtime(torch.arange(3), reference_axis_rad=torch.zeros(3),
                                task_episode_ids=torch.tensor([0, 1, 0]))

    def test_progress_trigger_is_inert(self):
        layer, rt = F.build((), n=2)
        F.reset(layer, rt)
        self.assertIsNone(layer.reset_runtime(progress=torch.tensor([5.0, float("nan")])))


class Stamps(unittest.TestCase):
    def test_header_and_episode_stamps(self):
        layer, rt = F.build(("journal.combo.wave_wind.D2", "journal.actuator.gain.D3",
                             "journal.perception.noise.D2"), n=4)
        F.reset(layer, rt, axis=torch.full((4,), 0.1, dtype=torch.float64))
        quat = F.quaternion_yaw(torch.zeros(4))
        for _ in range(3):
            rt.step(torch.zeros(4, 2), torch.zeros(4, 3), torch.zeros(4, 6), quat, torch.zeros(4, 3),
                    **layer.physics_inputs(torch.zeros(4, 6), quat).as_kwargs())
        layer.observation_out("pos", torch.zeros(4, 2))
        header = layer.certificate_stamp()
        self.assertEqual(header["schema"], "usvbench.conditions/3")
        self.assertEqual(header["parameters_sha256"], api.parameters_digest(header["parameters"]))
        context = header["context"]
        for key in ("hydrodynamics_model", "vessel_profile", "rotational_damping", "froude_krylov_horizontal",
                    "saturation_policy", "footprint_filter", "resolved", "direction_strata",
                    "pending_owner_decisions", "observation_contract"):
            self.assertIn(key, context)
        self.assertEqual(context["hydrodynamics_model"], "hydro_v2")
        waves = context["resolved"]["waves"]
        for key in ("band_min_over_fp", "band_max_over_fp", "n_components", "spreading_rad", "gamma"):
            self.assertIn(key, waves)
        self.assertTrue(all(v["status"].startswith(("PENDING", "REQUIRED", "DECIDED"))     # G28: decided labels
                            for v in context["pending_owner_decisions"].values()))
        record = layer.episode_stamp(2)
        self.assertEqual(record["parameters_sha256"], header["parameters_sha256"])
        runtime = record["runtime"]
        for key in ("scenario_ordinal", "direction_stratum", "waves", "wind", "actuator", "perception"):
            self.assertIn(key, runtime)
        self.assertEqual(runtime["waves"]["n_components"], waves["n_components"])
        self.assertEqual(api.config_from_stamp({"conditions": record}), layer.cfg)
        import json
        json.dumps(record, allow_nan=False)


if __name__ == "__main__":
    print("runtime kernel:", F.RUNTIME_KERNEL, flush=True)
    unittest.main(verbosity=2)
