"""E3: keyed batched random streams, wind protocol v2, journal perception,
direction strata and the observation-group table (CPU only).

Run: python -B tasks/_shared/conditions/tests/test_keyed_streams_v3.py
Covers D2 review items 4, 7, 11 and C1 finding 31.
"""
from __future__ import annotations

import math
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np
import torch

import e3_fixtures as F

api = F.api
KR = F.keyed_rng
DT_CONTROL = F.DT * F.DECIMATION


class KeyedCounterRng(unittest.TestCase):
    def test_splitmix64_reference_values_and_uniform_range(self):
        # Standard SplitMix64 sequence from state 0 (Vigna's reference outputs).
        self.assertEqual(int(KR.splitmix64(np.uint64(0), 0)), 0xE220A8397B1DCDAF)
        self.assertEqual(int(KR.splitmix64(np.uint64(0), 1)), 0x6E789E6AA1B965F4)
        u = KR.uniforms(np.uint64(12345), np.arange(100000))
        self.assertTrue(bool((u > 0).all() and (u < 1).all()))
        z = KR.normals(np.uint64(99), np.arange(200000))
        self.assertLess(abs(float(z.mean())), 0.01)
        self.assertLess(abs(float(z.std()) - 1.0), 0.01)

    def test_stream_key_uses_the_shared_scenario_protocol(self):
        key = KR.stream_key(7, 3, 2, "wind.gust")
        expected = F.scenario_rng.stable_stream_seed(F.scenario_rng.SCENARIO_PROTOCOL_VERSION, 7, 3, 2,
                                                     "cond3.wind.gust")
        self.assertEqual(key, expected)
        self.assertNotEqual(key, KR.stream_key(7, 3, 2, "perception.pos_noise"))
        self.assertNotEqual(key, KR.stream_key(7 + (1 << 32), 3, 2, "wind.gust"))   # C1 finding 31
        with self.assertRaises(ValueError):
            KR.stream_key(7, 3, -1, "wind.gust")

    def test_tape_is_chunk_order_and_batch_independent(self):
        keys = [KR.stream_key(1, i, 0, "t") for i in range(6)]
        a = KR.KeyedNormalTape(6, 3, "cpu", chunk_steps=1000)
        a.reset(np.arange(6), keys)
        b = KR.KeyedNormalTape(6, 3, "cpu", chunk_steps=4)
        for batch in ([5, 2], [0, 4, 1, 3]):
            b.reset(batch, [keys[i] for i in batch])
        for step in range(0, 40, 3):
            steps = np.full(6, step)
            self.assertTrue(torch.equal(a.gather(np.arange(6), steps), b.gather(np.arange(6), steps)))
        direct = KR.normals(np.uint64(keys[4]), np.arange(3) + 39 * 3)
        torch.testing.assert_close(b.gather([4], [39])[0], torch.as_tensor(direct, dtype=torch.float32))


class WindProtocolV2(unittest.TestCase):
    def wind(self, n, chunk=1024, protocol=None, sigma=0.15, seed=11):
        cfg = F.wind_backend.WindCfg(enable=True, mean_speed_mps=5.0, gust_intensity=sigma,
                                     gust_correlation_time_s=2.0,
                                     windage=F.wind_backend.VesselWindage(0.08, 0.19, 1.2, 0.085))
        return F.wind_backend.Wind(cfg, n, "cpu", eval_seed=seed, chunk_steps=chunk,
                                   rng_protocol=protocol or F.wind_backend.RNG_PROTOCOL_V2)

    def test_reset_order_batching_and_chunk_size_do_not_change_the_stream(self):
        n = 7
        a = self.wind(n)
        a.reset(torch.arange(n), torch.tensor([0, 1, 2, 0, 1, 2, 3]))
        b = self.wind(n, chunk=5)
        for batch in ([6, 1], [3, 0, 5], [4, 2]):
            ids = torch.tensor(batch)
            b.reset(ids, torch.tensor([0, 1, 2, 0, 1, 2, 3])[ids])
        self.assertTrue(torch.equal(a.gust_mps, b.gust_mps))
        for _ in range(60):
            a.step(F.DT); b.step(F.DT)
            self.assertTrue(torch.equal(a.gust_mps, b.gust_mps))

    def test_exact_ou_recursion_from_keyed_normals(self):
        w = self.wind(3)
        w.reset(torch.arange(3), torch.tensor([4, 4, 5]))
        sigma = 5.0 * 0.15
        ratio = F.DT / 2.0
        a, b = math.exp(-ratio), sigma * math.sqrt(-math.expm1(-2.0 * ratio))
        keys = [KR.stream_key(11, i, e, "wind.gust") for i, e in zip(range(3), (4, 4, 5))]
        g = sigma * torch.as_tensor([KR.normals(np.uint64(k), 0) for k in keys], dtype=torch.float32)
        self.assertTrue(torch.equal(w.gust_mps, g))
        for step in range(1, 30):
            w.step(F.DT)
            n = torch.as_tensor([KR.normals(np.uint64(k), step) for k in keys], dtype=torch.float32)
            g = a * g + b * n
            self.assertTrue(torch.equal(w.gust_mps, g))
        self.assertEqual(w.physics_steps(1), 29)

    def test_no_per_step_keyed_hashing(self):
        calls = {"n": 0}
        original = KR.stable_stream_seed
        def counting(*args):
            calls["n"] += 1
            return original(*args)
        v1_draw = F.wind_backend._normal_for_key
        def forbidden(*args):
            raise AssertionError("v1 per-step BLAKE2b draw used under protocol v2")
        KR.stable_stream_seed = counting
        F.wind_backend._normal_for_key = forbidden
        try:
            w = self.wind(16, chunk=64)
            w.reset(torch.arange(16), torch.zeros(16, dtype=torch.long))
            for _ in range(500):
                w.step(F.DT)
            w.reset(torch.tensor([3, 4]), torch.tensor([1, 1]))
            for _ in range(10):
                w.step(F.DT)
        finally:
            KR.stable_stream_seed = original
            F.wind_backend._normal_for_key = v1_draw
        self.assertEqual(calls["n"], 18)          # one key per env reset, none per step
        self.assertEqual(w.key_derivations, 18)
        self.assertGreater(w._tape.refills, 1)    # chunked refills happened (vectorised)

    def test_stationary_statistics(self):
        n, steps = 64, 3000
        w = self.wind(n)
        w.reset(torch.arange(n), torch.zeros(n, dtype=torch.long))
        sigma = 0.75
        a = math.exp(-F.DT / 2.0)
        b = sigma * math.sqrt(1 - a * a)
        previous = w.gust_mps.clone().double()
        values, innovations = [previous], []
        for _ in range(steps):
            w.step(F.DT)
            current = w.gust_mps.clone().double()
            innovations.append((current - a * previous) / b)
            values.append(current)
            previous = current
        z = torch.stack(innovations)
        self.assertLess(abs(float(z.mean())), 0.01)
        self.assertLess(abs(float(z.std()) - 1.0), 0.01)
        g = torch.stack(values)
        self.assertLess(abs(float(g.std()) / sigma - 1.0), 0.08)

    def test_v1_default_is_unchanged(self):
        w = self.wind(2, protocol=F.wind_backend.RNG_PROTOCOL_V1)
        self.assertEqual(w.rng_protocol, "usvbench.wind.ou.v1")
        w.reset(torch.arange(2), torch.tensor([0, 3]))
        normals = torch.tensor([F.wind_backend._normal_for_key(11, i, e, 0) for i, e in ((0, 0), (1, 3))],
                               dtype=torch.float32)
        self.assertTrue(torch.equal(w.gust_mps, 0.75 * normals))     # C3 wind.py reset arithmetic
        default = F.wind_backend.Wind(F.wind_backend.WindCfg(), 2)
        self.assertEqual(default.rng_protocol, F.wind_backend.RNG_PROTOCOL_V1)
        with self.assertRaises(ValueError):
            self.wind(2, protocol="usvbench.wind.ou.v9")


def degrader(n=8, seed=5, contract=("hazard_nav", "v3"), **perception):
    values = {"enabled": True}
    values.update(perception)
    cfg = api.resolve_v3((), {"perception": values}).perception
    return F.perception_v3.PerceptionV3Degrader(
        cfg, n, "cpu", seed, F.observation_groups.lookup(*contract), DT_CONTROL,
        ray_count=36, ray_max_range_m=30.0)


def reset(d, ids=None, episode=0, axis=0.0, **kwargs):
    ids = torch.arange(d.num_envs) if ids is None else torch.as_tensor(ids)
    episodes = torch.full((ids.numel(),), episode, dtype=torch.long)
    ordinals = F.strata.scenario_ordinals(ids, episodes, d.num_envs)
    strata = F.strata.stratum_indices(ordinals, d.seed)
    d.reset(ids, episodes, strata, torch.full((ids.numel(),), axis, dtype=torch.float64), **kwargs)
    return strata


class Perception(unittest.TestCase):
    def test_primitives_use_independent_streams_c1_31(self):
        base = dict(pos_noise_sigma_m=0.1, outage_s=0.3)
        plain = degrader(**base)
        biased = degrader(pos_bias_m=0.5, **base)
        outage_only = degrader(outage_s=0.3)
        for d in (plain, biased, outage_only):
            reset(d, axis=0.4)
        self.assertTrue(np.array_equal(plain.outage_start, biased.outage_start))
        self.assertTrue(np.array_equal(plain.outage_start, outage_only.outage_start))
        zero = torch.zeros(8, 2)
        for _ in range(900):
            a, b = plain.apply("pos", zero), biased.apply("pos", zero)
            self.assertTrue(torch.equal(b, a - biased.bias))
        # a vel-bundle stream never perturbs the position noise stream
        with_vel = degrader(vel_noise_sigma_mps=0.2, **base)
        reference = degrader(**base)
        reset(with_vel); reset(reference)
        for _ in range(50):
            with_vel.apply("vel", torch.zeros(8, 3))
            self.assertTrue(torch.equal(with_vel.apply("pos", zero), reference.apply("pos", zero)))

    def test_noise_is_independent_of_reset_order_and_batching(self):
        a, b = degrader(pos_noise_sigma_m=0.4), degrader(pos_noise_sigma_m=0.4)
        reset(a)
        for batch in ([7, 3], [0, 6, 1], [5, 2, 4]):
            reset(b, ids=batch)
        zero = torch.zeros(8, 2)
        for _ in range(30):
            self.assertTrue(torch.equal(a.apply("pos", zero), b.apply("pos", zero)))
        reset(a, ids=[2], episode=1)
        reset(b, ids=[2], episode=1)
        self.assertTrue(torch.equal(a.apply("pos", zero), b.apply("pos", zero)))

    def test_position_noise_is_first_order_gauss_markov(self):
        d = degrader(n=64, pos_noise_sigma_m=0.4, contract=("station_keeping", "native3"))
        reset(d)
        zero = torch.zeros(64, 2)
        samples = torch.stack([-d.apply("pos", zero) for _ in range(3000)]).double()
        a = math.exp(-DT_CONTROL / 0.2)
        self.assertLess(abs(float(samples.std()) / 0.4 - 1.0), 0.05)
        x, y = samples[:-1].flatten(), samples[1:].flatten()
        corr = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
        self.assertLess(abs(corr - a), 0.02)

    def test_latency_is_an_exact_shift_with_age(self):
        d = degrader(latency_s=0.10)      # 6 control frames
        self.assertEqual(d.latency, 6)
        reset(d)
        inputs = [torch.full((8, 2), float(k)) for k in range(20)]
        outputs = [d.apply("pos", x) for x in inputs]
        for k, out in enumerate(outputs):
            expected = inputs[max(k - 6, 0)] if k >= 6 else inputs[0]
            self.assertTrue(torch.equal(out, expected))
        torch.testing.assert_close(d.measurement_age_s("pos"), torch.full((8,), 6 * DT_CONTROL))

    def test_outage_holds_the_last_estimate(self):
        d = degrader(outage_s=0.30)       # 18 frames
        self.assertEqual(d.outage, 18)
        reset(d, outage_start_s=torch.full((8,), 10 * DT_CONTROL, dtype=torch.float64))
        outputs = [d.apply("pos", torch.full((8, 2), float(k))) for k in range(40)]
        for k, out in enumerate(outputs):
            expected = 9.0 if 10 <= k < 28 else float(k)
            self.assertTrue(torch.equal(out, torch.full((8, 2), expected)), k)
        record = d.episode_parameters(0)
        self.assertEqual((record["outage_start_frame"], record["outage_end_frame"]), (10, 28))
        self.assertEqual(record["outage_source"], "task_critical_phase")
        fallback = degrader(outage_s=1.0)
        reset(fallback)
        lo, hi = 5.0 / DT_CONTROL, 20.0 / DT_CONTROL
        self.assertTrue(bool(((fallback.outage_start >= lo - 1) & (fallback.outage_start <= hi)).all()))
        self.assertEqual(fallback.episode_parameters(3)["outage_source"], "fallback_window_PENDING")

    def test_bias_magnitude_and_balanced_direction(self):
        d = degrader(n=64, pos_bias_m=0.5)
        strata = reset(d, axis=0.3)
        norm = torch.linalg.vector_norm(d.bias, dim=-1)
        torch.testing.assert_close(norm, torch.full((64,), 0.5))
        direction = torch.atan2(d.bias[:, 1], d.bias[:, 0]).double()
        expected = 0.3 + F.strata.relative_angles(strata)
        diff = torch.remainder(direction - expected + math.pi, 2 * math.pi) - math.pi
        self.assertLess(float(diff.abs().max()), 1e-6)
        self.assertLess(float(d.bias.sum(0).abs().max()), 1e-5)     # balanced: vectors cancel

    def test_sensor_bundle_channels_and_ray_clamp(self):
        d = degrader(yaw_rate_noise_sigma_rad_s=0.1)
        reset(d)
        x = torch.ones(8, 3)
        y = d.apply("vel", x)
        self.assertTrue(torch.equal(y[:, :2], x[:, :2]))
        self.assertFalse(torch.equal(y[:, 2], x[:, 2]))
        r = degrader(ray_noise_sigma_m=5.0)
        reset(r)
        out = r.apply("ray", torch.full((8, 36), 29.9))
        self.assertLessEqual(float(out.max()), 30.0)
        self.assertGreaterEqual(float(out.min()), 0.0)

    def test_tape_memory_is_capped_for_wide_groups(self):
        d = degrader(n=4096, ray_noise_sigma_m=0.1, pos_noise_sigma_m=0.1)
        ray, pos = d.groups["ray"].tape, d.groups["pos"].tape
        self.assertEqual(ray.chunk, (1 << 22) // (4096 * 36))
        self.assertLessEqual(ray.tape.numel(), 1 << 22)
        self.assertLessEqual(pos.tape.numel(), 1 << 22)

    def test_zero_dose_is_identity(self):
        d = degrader()
        reset(d)
        x = torch.randn(8, 2)
        self.assertIs(d.apply("pos", x), x)
        self.assertFalse(any(d.wants(g) for g in ("pos", "vel", "ray")))


class Strata(unittest.TestCase):
    def test_exact_balance_on_a_64_env_grid(self):
        for seed in (0, 1, 42, 12345):
            for episode in (0, 1, 2):
                ordinals = F.strata.scenario_ordinals(torch.arange(64), torch.full((64,), episode), 64)
                counts = torch.bincount(F.strata.stratum_indices(ordinals, seed), minlength=8)
                self.assertTrue(torch.equal(counts, torch.full((8,), 8)))
        self.assertEqual(sorted(a for _, a in F.strata.STRATA), [0, 45, 90, 135, 180, 225, 270, 315])
        classes = [F.strata.STRATA_CLASSES[name] for name, _ in F.strata.STRATA]
        self.assertEqual(classes.count("head"), 1); self.assertEqual(classes.count("following"), 1)
        self.assertEqual(classes.count("beam"), 2)
        self.assertEqual(classes.count("bow_quartering"), 2); self.assertEqual(classes.count("stern_quartering"), 2)

    def test_layer_assigns_the_same_balanced_stratum_to_every_directional_suite(self):
        presets = ("journal.waves.D2", "journal.current.D1")
        layer, rt = F.build(presets, n=64, seed=77)
        for episode in range(2):
            F.reset(layer, rt, axis=torch.full((64,), 1.0, dtype=torch.float64))
            counts = np.bincount(layer.stratum, minlength=8)
            self.assertTrue(np.array_equal(counts, np.full(8, 8)))
            rel = torch.remainder(layer.sea.mean_direction.double() - 1.0, 2 * math.pi)
            expected = F.strata.relative_angles(layer.stratum)
            diff = torch.remainder(rel - expected + math.pi, 2 * math.pi) - math.pi
            self.assertLess(float(diff.abs().max()), 1e-5)
            self.assertTrue(np.array_equal(layer.ordinal, np.arange(64) + 64 * episode))

    def test_ordinals_ignore_reset_order(self):
        a, rta = F.build(("journal.current.D1",), n=8)
        b, rtb = F.build(("journal.current.D1",), n=8)
        F.reset(a, rta)
        F.reset(b, rtb, ids=[6, 2, 0]); F.reset(b, rtb, ids=[1, 7, 3, 5, 4])
        self.assertTrue(np.array_equal(a.ordinal, b.ordinal))
        self.assertTrue(np.array_equal(a.stratum, b.stratum))


class ObservationGroups(unittest.TestCase):
    def test_contract_derivation_and_groups(self):
        from types import SimpleNamespace as S
        derive, lookup = F.observation_groups.derive_contract, F.observation_groups.lookup
        cases = [
            ("hazard_nav", S(), "v1", ("pos", "ray")),
            ("hazard_nav", S(obs_v2=True), "v2", ("pos", "vel", "ray")),
            ("hazard_nav", S(obs_v3=True), "v3", ("pos", "vel", "ray")),
            ("hazard_nav", S(obs_v3=True, obs_feasibility=True), "v3_pooled", ("pos", "vel", "ray")),
            ("hazard_nav", S(obs_v3=True, emit_superset_obs=True, superset_version=2), "v3_superset2",
             ("pos", "vel", "ray")),
            ("hazard_nav", S(emit_superset_obs=True, superset_version=1), "v1_superset1", ("pos", "vel", "ray")),
            ("station_keeping", S(), "native3", ("pos",)),
            ("station_keeping", S(obs_kinematic=True), "kinematic6", ("pos", "vel")),
            ("station_keeping", S(obs_sea_state=True, obs_kinematic=True), "sea_state9", ("pos", "vel")),
            ("path_following", S(observation_contract="native7"), "native7", ("pos",)),
        ]
        for family, cfg, contract, groups in cases:
            self.assertEqual(derive(family, cfg), contract)
            self.assertEqual(lookup(family, contract).groups, groups)
        self.assertTrue(lookup("station_keeping", "sea_state9").sea_state_observed)
        # speed-only layouts consume surge/sway but not the yaw rate
        self.assertEqual(lookup("hazard_nav", "v2").vel_channels, ("surge", "sway"))
        self.assertEqual(lookup("hazard_nav", "v3_superset1").vel_channels, ("surge", "sway"))
        self.assertEqual(lookup("hazard_nav", "v1_superset2").vel_channels, ("surge", "sway", "yaw_rate"))
        yaw = api.resolve_v3((), {"perception": {"enabled": True, "yaw_rate_noise_sigma_rad_s": 0.1}}).perception
        with self.assertRaisesRegex(ValueError, "yaw-rate"):
            F.observation_groups.admit_perception(lookup("hazard_nav", "v2"), yaw)
        F.observation_groups.admit_perception(lookup("hazard_nav", "v3"), yaw)
        with self.assertRaises(ValueError):
            derive("docking", S())
        with self.assertRaises(ValueError):
            lookup("hazard_nav", "v99")
        families = {family for family, _ in F.observation_groups.OBSERVATION_CONTRACTS}
        self.assertEqual(families, set(F.hydro_v2_layer.HYDRO_V2_FAMILIES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
