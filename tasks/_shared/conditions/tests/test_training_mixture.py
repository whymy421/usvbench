"""G19: training-only condition mixture (prototype; to be rebased on v2.5). CPU only.

Run: python -B tasks/_shared/conditions/tests/test_training_mixture.py
Covers: the block-balanced sampler (pure function of the key, exact shares,
axis cycles, namespace refusal, hazard doses), the D1 training stack
(resolution, stack and layer checks), and the condition-layer gates (a
gated-off row is the disabled suite exactly, through a fake and a real
HydroV2Runtime; actuator sub-axes; perception sub-axes; stamps and the
certificate refusal; attach guards).
"""
from __future__ import annotations

from fractions import Fraction
import importlib
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np
import torch

import e3_fixtures as F

tm = importlib.import_module(F.PACKAGE + ".conditions.training_mixture")
jc = importlib.import_module(F.PACKAGE + ".conditions.journal_config")

SEED = 1001
D1_PRESETS = ("journal.waves.D1", "journal.wind.D1", "journal.current.D1",
              "journal.actuator.cap.D1", "journal.actuator.gain.D1", "journal.actuator.lag.D1",
              "journal.actuator.resistance.D1", "journal.perception.noise.D1",
              "journal.perception.bias.D1", "journal.perception.latency.D1",
              "journal.perception.dropout.D1")
PHYSICS_GATES = ("waves", "wind", "current", "actuator.cap", "actuator.gain", "actuator.lag",
                 "actuator.resistance")


def mixture(appearing=True, seed=SEED, **kwargs):
    return tm.TrainingMixture(seed=seed, task="scatter", vessel="blueboat_v2", appearing_wired=appearing, **kwargs)


class Sampler(unittest.TestCase):
    def test_component_is_a_pure_function_of_the_key(self):
        a, b = mixture(), mixture()
        keys = [(e, k) for e in range(6) for k in range(40)]
        forward = [a.component(e, k) for e, k in keys]
        backward = [b.component(e, k) for e, k in reversed(keys)][::-1]
        self.assertEqual(forward, backward)
        self.assertNotEqual([mixture(seed=2003).component(e, k) for e, k in keys], forward)
        state = np.random.get_state()[1].copy()
        torch_state = torch.random.get_rng_state().clone()
        mixture().component(3, 17)
        self.assertTrue(np.array_equal(state, np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_blocks_hold_every_suite_once_and_calm_three_times(self):
        for appearing, length in ((True, 10), (False, 9)):
            m = mixture(appearing)
            self.assertEqual(m.block_length, length)
            for env in range(8):
                for block in range(12):
                    names = [m.component(env, block * length + p) for p in range(length)]
                    suites = [n.split(".")[0] for n in names]
                    self.assertEqual(suites.count("calm"), 3)
                    for suite in m.suites:
                        self.assertEqual(suites.count(suite), 1, (env, block, names))
                    self.assertEqual("appearing" in suites, appearing)

    def test_axis_cycles_cover_the_four_sub_axes(self):
        m = mixture()
        for env in range(5):
            for suite, axes in tm.AXIS_SUITES.items():
                seen = [n for n in (m.component(env, k) for k in range(m.block_length * 8))
                        if n.startswith(suite + ".")]
                self.assertEqual(len(seen), 8)
                for cycle in range(2):
                    self.assertEqual(sorted(s.split(".")[1] for s in seen[4 * cycle:4 * cycle + 4]),
                                     sorted(axes))

    def test_weights_are_exact_and_match_realised_shares(self):
        m = mixture()
        w = m.weights()
        self.assertEqual(sum(w.values()), Fraction(1))
        self.assertEqual(w["calm"], Fraction(3, 10))
        self.assertEqual(w["waves"], Fraction(1, 10))
        self.assertEqual(w["actuator.lag"], Fraction(1, 40))
        counts = {name: 0 for name in tm.COMPONENTS}
        episodes = m.block_length * 4            # 4 blocks = one full axis cycle per env
        for env in range(16):
            for k in range(episodes):
                counts[m.component(env, k)] += 1
        total = 16 * episodes
        for name in tm.COMPONENTS:
            self.assertEqual(Fraction(counts[name], total), w[name], name)
        w6 = mixture(False).weights()
        self.assertEqual(w6["appearing"], 0)
        self.assertEqual(w6["calm"], Fraction(3, 9))

    def test_gates_and_hazard_doses(self):
        m = mixture()
        ids, eps = np.arange(10), np.arange(10) * 7
        gates = m.gates(ids, eps)
        names = m.components(ids, eps)
        for row, name in enumerate(names):
            on = [g for g in tm.GATES if gates[g][row]]
            self.assertEqual(on, [] if name == "calm" else [name])
        for env in range(4):
            for k in range(30):
                name = m.component(env, k)
                self.assertEqual(m.hazard_dose("dynamic", env, k), 1 if name == "dynamic" else 0)
                self.assertEqual(m.hazard_dose("appearing", env, k), 1 if name == "appearing" else 0)
        self.assertEqual(mixture(False).hazard_dose("appearing", 0, 3), 0)
        with self.assertRaises(tm.MixtureError):
            m.hazard_dose("waves", 0, 0)

    def test_namespace_and_argument_refusals(self):
        for purpose in ("selection", "final_test", "pilot", "calibration", "diagnostic"):
            with self.assertRaises(tm.MixtureError):
                mixture(purpose=purpose)
        with self.assertRaises(tm.MixtureError):
            mixture(seed=True)
        with self.assertRaises(tm.MixtureError):
            mixture(calm_slots=-1)
        with self.assertRaises(tm.MixtureError):
            mixture().component(-1, 0)

    def test_stamp_is_self_describing_and_digest_stable(self):
        m = mixture()
        stamp = m.stamp(counts={"calm": 3})
        self.assertEqual(stamp["mixture_sha256"], mixture().digest())
        self.assertNotEqual(m.digest(), mixture(calm_slots=2).digest())
        self.assertEqual(stamp["presets"]["waves"]["values"]["waves"]["hs_m"], 0.125)
        self.assertEqual(stamp["presets"]["perception.latency"]["values"]["perception"]["latency_s"], 0.05)
        self.assertEqual(stamp["presets"]["actuator.cap"]["values"]["actuator"]["thrust_cap"], 0.9)
        self.assertEqual(stamp["weights"]["calm"], "3/10")
        self.assertEqual(stamp["realised_episode_starts"], {"calm": 3})

    def test_permutation_is_unbiased_enough(self):
        counts = np.zeros((3, 3), dtype=np.int64)
        for key in range(3000):
            order = tm.keyed_permutation(("a", "b", "c"), key * 7919 + 1)
            for position, item in enumerate(order):
                counts[position, "abc".index(item)] += 1
        self.assertTrue(np.all(np.abs(counts - 1000) < 120), counts)


class Stack(unittest.TestCase):
    def test_required_tokens_resolve_and_pass_the_checks(self):
        for appearing in (True, False):
            m = mixture(appearing)
            resolved = jc.resolve_journal_conditions(m.required_tokens(), seed=SEED)
            self.assertTrue(tm.check_stack(m, resolved))
            self.assertTrue(tm.check_layer_config(m, resolved.conditions))
            self.assertEqual(resolved.dynamic_dose, 1)
            self.assertEqual(resolved.appearing_enabled, appearing)

    def test_wrong_stacks_are_refused(self):
        m = mixture()
        bad = [["waves=D2"] + [t for t in m.required_tokens() if not t.startswith("waves")],
               [t for t in m.required_tokens() if not t.startswith("current")],
               m.required_tokens() + ["perception.bundle=D1"],
               [t for t in m.required_tokens() if not t.startswith("appearing")]]
        for tokens in bad:
            resolved = jc.resolve_journal_conditions(tokens, seed=SEED)
            with self.assertRaises(tm.MixtureError):
                tm.check_stack(m, resolved)
                tm.check_layer_config(m, resolved.conditions)
        other_seed = jc.resolve_journal_conditions(m.required_tokens(), seed=SEED + 1)
        with self.assertRaises(tm.MixtureError):
            tm.check_stack(m, other_seed)


def build_pair(n, runtime="fake", calm_slots=3):
    """(mixture layer + runtime, calm layer + runtime), both reset for every env."""
    m = mixture(calm_slots=calm_slots)
    layer, rt = F.build(D1_PRESETS, n=n, runtime=runtime, seed=SEED)
    layer.attach_training_mixture(m)
    calm, crt = F.build((), n=n, runtime=runtime, seed=SEED)
    axis = torch.full((n,), 0.3, dtype=torch.float64)
    F.reset(layer, rt, axis=axis)
    F.reset(calm, crt, axis=axis)
    return m, (layer, rt), (calm, crt)


def drive_pair(pair, steps, seed=5, resets=((7, (1, 4)), (13, (0, 2, 3)))):
    """Identical random states through both layers; returns per-run lists of per-step outputs."""
    generator = torch.Generator().manual_seed(seed)
    outs = [[], []]
    reset_at = dict(resets)
    n = pair[0][1].num_envs
    origins = torch.zeros((n, 3))
    for k in range(steps):
        if k in reset_at:
            ids = torch.as_tensor(reset_at[k])
            for layer, rt in pair:
                rt.reset(ids)
                layer.reset_runtime(ids, reference_axis_rad=torch.full((ids.numel(),), 0.3, dtype=torch.float64),
                                    task_episode_ids=F.task_episode_ids(layer, ids))
                layer.configure_runtime(rt, ids)
        position, velocity, quat, actions = F_random_state(generator, n, pair[0][1].profile)
        for run, (layer, rt) in enumerate(pair):
            kwargs = layer.physics_inputs(velocity, quat).as_kwargs()
            outs[run].append((kwargs, rt.step(actions, position, velocity, quat, origins, **kwargs)))
    return outs


def F_random_state(generator, n, prof):
    z = prof.hydro.equilibrium_com_z_m
    position = torch.randn((n, 3), generator=generator) * 2.0
    position[:, 2] = z + 0.02 * torch.randn(n, generator=generator)
    velocity = torch.randn((n, 6), generator=generator) * 0.5
    yaw = torch.randn(n, generator=generator) * 0.5
    quat = F.quaternion_yaw(yaw)
    actions = torch.rand((n, 2), generator=generator) * 2 - 1
    return position, velocity, quat, actions


def physics_off_rows(layer):
    return [i for i in range(layer.num_envs)
            if not any(bool(layer._gates[g][i]) for g in PHYSICS_GATES)]


class LayerGates(unittest.TestCase):
    N = 16

    def test_gated_off_rows_equal_the_calm_layer_through_the_real_runtime(self):
        m, mix, calm = build_pair(self.N, runtime="real", calm_slots=1)
        rows_seen = set()
        first_episode_off = physics_off_rows(mix[0])      # components until the first reset (step 7)
        outs = drive_pair((mix, calm), 20)
        for step in range(20):
            off = first_episode_off if step < 7 else None
            a, b = outs[0][step][1], outs[1][step][1]
            fa = torch.cat((a.force_actor, a.torque_actor), -1)
            fb = torch.cat((b.force_actor, b.torque_actor), -1)
            if off is not None:
                for i in off:
                    rows_seen.add(i)
                    self.assertTrue(torch.equal(fa[i], fb[i]), (step, i, mix[0].training_component(i)))
        self.assertTrue(rows_seen, "no gated-off row in the window")
        self.assertTrue(bool(torch.isfinite(fa).all()))

    def test_gated_off_rows_equal_the_calm_layer_in_every_episode(self):
        """Follow-up (review): not only the first episode - after every reset, every row whose CURRENT episode has
        all physics gates off produces bitwise the calm layer's wrench (real HydroV2Runtime, 3 episodes per row)."""
        m, mix, calm = build_pair(self.N, runtime="real", calm_slots=1)
        generator = torch.Generator().manual_seed(23)
        n = self.N
        origins = torch.zeros((n, 3))
        resets = {6: tuple(range(0, n, 2)), 12: tuple(range(n)), 18: tuple(range(1, n, 2))}
        checked_rows, later_episode_rows = 0, 0
        for k in range(24):
            if k in resets:
                ids = torch.as_tensor(resets[k])
                for layer, rt in (mix, calm):
                    rt.reset(ids)
                    layer.reset_runtime(ids, reference_axis_rad=torch.full((ids.numel(),), 0.3, dtype=torch.float64),
                                        task_episode_ids=F.task_episode_ids(layer, ids))
                    layer.configure_runtime(rt, ids)
            off = physics_off_rows(mix[0])
            position, velocity, quat, actions = F_random_state(generator, n, mix[1].profile)
            results = []
            for layer, rt in (mix, calm):
                kwargs = layer.physics_inputs(velocity, quat).as_kwargs()
                results.append(rt.step(actions, position, velocity, quat, origins, **kwargs))
            fa = torch.cat((results[0].force_actor, results[0].torque_actor), -1)
            fb = torch.cat((results[1].force_actor, results[1].torque_actor), -1)
            for i in off:
                self.assertTrue(torch.equal(fa[i], fb[i]), (k, i, mix[0].training_component(i)))
                checked_rows += 1
                later_episode_rows += int(mix[0].episode_index[i] > 0)
        self.assertGreater(later_episode_rows, 0, "no gated-off row after a reset")
        self.assertGreater(checked_rows, later_episode_rows)

    def test_physics_inputs_are_masked_per_row(self):
        m, mix, calm = build_pair(self.N, runtime="fake")
        layer = mix[0]
        current = layer.current_velocity_world()
        force, torque = layer.wind_wrench_world(torch.zeros((self.N, 6)), F.quaternion_yaw(torch.zeros(self.N)))
        waves = layer.wave_source()
        for i in range(self.N):
            name = layer.training_component(i)
            self.assertEqual(bool(current[i].abs().sum() > 0), name == "current", (i, name))
            self.assertEqual(bool(force[i].abs().sum() > 0), name == "wind", (i, name))
            self.assertEqual(bool(torque[i].abs().sum() > 0) or name != "wind", True)
            self.assertEqual(bool(waves.amplitude[i].abs().sum() > 0), name == "waves", (i, name))
        self.assertEqual(sum(layer.training_mixture_counts().values()), self.N)

    def test_actuator_sub_axes_follow_their_gates(self):
        m, (layer, rt), _ = build_pair(self.N, runtime="fake")
        state = layer._actuator
        for i in range(self.N):
            name = layer.training_component(i)
            self.assertEqual(float(state["cap"][i, 0]), 0.9 if name == "actuator.cap" else 1.0)
            self.assertAlmostEqual(float(state["lag"][i, 0]), 0.05 if name == "actuator.lag" else 0.0, places=6)
            self.assertAlmostEqual(float(state["resistance"][i]), 1.1 if name == "actuator.resistance" else 1.0,
                                   places=6)
            gains = sorted(float(v) for v in state["gain"][i])
            if name == "actuator.gain":
                self.assertAlmostEqual(gains[0], 0.95, places=6)
                self.assertEqual(gains[1], 1.0)
                self.assertIn(int(state["side"][i]), (0, 1))
            else:
                self.assertEqual(gains, [1.0, 1.0])
                self.assertEqual(int(state["side"][i]), -1)

    def test_failing_side_is_stamped_only_where_the_gain_gate_is_on(self):
        """verify2 m2: a row with actuator.gain off (side -1, gains [1, 1]) stamps no failing side; a gated-on row
        stamps its side; without a mixture every row keeps its side (evaluation unchanged)."""
        n = 160                                             # actuator.gain is 1/40 of the episodes
        m, (layer, rt), _ = build_pair(n, runtime="fake")
        seen = set()
        for i in range(n):
            name = layer.training_component(i)
            got = layer.episode_stamp(i)["runtime"]["actuator"]["failing_side"]
            side = int(layer._actuator["side"][i])
            if name == "actuator.gain":
                self.assertEqual(got, "port" if side == 0 else "starboard", i)
            else:
                self.assertIsNone(got, (i, name))
            seen.add(name == "actuator.gain")
        self.assertEqual(seen, {True, False})
        plain, prt = F.build(D1_PRESETS, n=4, seed=SEED)
        F.reset(plain, prt)
        for i in range(4):
            side = int(plain._actuator["side"][i])
            self.assertIn(side, (0, 1))
            self.assertEqual(plain.episode_stamp(i)["runtime"]["actuator"]["failing_side"],
                             "port" if side == 0 else "starboard")

    def test_trigger_gap_signature_is_detected_by_parameter_names(self):
        """verify2 item 7: no exception fallback; v2.5 (dose, t_avoid_s), G20b (dose, t_avoid_s, vessel_id=None),
        anything else refused (a TypeError inside trigger_gap_m propagates)."""
        import types as _types
        calls = []

        def v25(dose, t_avoid_s):
            calls.append(("v25", dose, t_avoid_s))
            return 1.5

        def g20b(dose, t_avoid_s, vessel_id=None):
            calls.append(("g20b", dose, t_avoid_s, vessel_id))
            return 5.0

        def odd(level, t, v):
            return 0.0

        def broken(dose, t_avoid_s, vessel_id=None):
            raise TypeError("a real bug inside trigger_gap_m")

        self.assertEqual(tm._trigger_gap_d1(_types.SimpleNamespace(trigger_gap_m=v25), 2.0, "blueboat_v2"), 1.5)
        self.assertEqual(tm._trigger_gap_d1(_types.SimpleNamespace(trigger_gap_m=g20b), 2.0, "blueboat_v2"), 5.0)
        self.assertEqual(calls, [("v25", 1, 2.0), ("g20b", 1, 2.0, "blueboat_v2")])
        with self.assertRaises(tm.MixtureError):
            tm._trigger_gap_d1(_types.SimpleNamespace(trigger_gap_m=odd), 2.0, "blueboat_v2")
        with self.assertRaises(TypeError):
            tm._trigger_gap_d1(_types.SimpleNamespace(trigger_gap_m=broken), 2.0, "blueboat_v2")

    def test_perception_sub_axes_follow_their_gates(self):
        m, (layer, rt), _ = build_pair(self.N, runtime="fake")
        generator = torch.Generator().manual_seed(11)
        inputs, outputs, vel_in, vel_out = [], [], [], []
        for k in range(30):
            x = torch.randn((self.N, 2), generator=generator)
            inputs.append(x)
            outputs.append(layer.observation_out("pos", x))
            v = torch.randn((self.N, 3), generator=generator)
            vel_in.append(v)
            vel_out.append(layer.observation_out("vel", v))
        latency = layer.perception.latency
        self.assertEqual(latency, 3)
        for i in range(self.N):          # velocity group: only the temporal axes act on it (no bundle)
            name = layer.training_component(i)
            a = torch.stack([v[i] for v in vel_in])
            b = torch.stack([v[i] for v in vel_out])
            if name == "perception.latency":
                self.assertTrue(torch.equal(b[latency:], a[:-latency]), i)
            elif name != "perception.dropout":
                self.assertTrue(torch.equal(a, b), (i, name))
        for i in range(self.N):
            name = layer.training_component(i)
            series_in = torch.stack([v[i] for v in inputs])
            series_out = torch.stack([v[i] for v in outputs])
            if not name.startswith("perception."):
                self.assertTrue(torch.equal(series_in, series_out), (i, name))
            elif name == "perception.latency":
                self.assertTrue(torch.equal(series_out[latency:], series_in[:-latency]), i)
            elif name == "perception.bias":
                diff = (series_in - series_out).norm(dim=-1)
                self.assertTrue(torch.allclose(diff, torch.full_like(diff, 0.05), atol=1e-6), i)
            elif name == "perception.noise":
                diff = (series_in - series_out).norm(dim=-1)
                self.assertGreater(float(diff.max()), 0.0)
                self.assertLess(float(diff.max()), 0.2)
            elif name == "perception.dropout":
                held = [k for k in range(1, 30) if torch.equal(series_out[k], series_out[k - 1])]
                start = int(layer.perception.outage_start[i])
                if start < 29:
                    self.assertTrue(held, i)

    def test_stamps_and_certificate_refusal(self):
        m, (layer, rt), _ = build_pair(4, runtime="fake")
        with self.assertRaises(RuntimeError):
            layer.certificate_stamp()
        for i in range(4):
            stamp = layer.episode_stamp(i)
            block = stamp["runtime"]["training_mixture"]
            self.assertEqual(block["component"], layer.training_component(i))
            self.assertEqual(block["mixture_sha256"], m.digest())
            self.assertFalse(block["certificate_eligible"])
        self.assertFalse(layer.context["certificate_eligible"])

    def test_attach_guards(self):
        layer, rt = F.build(D1_PRESETS, n=4, seed=SEED)
        with self.assertRaises(TypeError):
            layer.attach_training_mixture(object())
        with self.assertRaises(tm.MixtureError):
            layer.attach_training_mixture(mixture(seed=SEED + 1))
        F.reset(layer, rt)
        with self.assertRaises(RuntimeError):
            layer.attach_training_mixture(mixture())
        wrong, _ = F.build(("journal.waves.D2",) + D1_PRESETS[1:], n=4, seed=SEED)
        with self.assertRaises(tm.MixtureError):
            wrong.attach_training_mixture(mixture())
        with self.assertRaises((tm.MixtureError, ValueError)):     # refused at build (C4) or at attach
            bundle, _ = F.build(D1_PRESETS + ("journal.perception.bundle.D1",), n=4, seed=SEED)
            bundle.attach_training_mixture(mixture())

    def test_no_mixture_path_is_untouched(self):
        layer, rt = F.build(D1_PRESETS, n=4, seed=SEED)
        F.reset(layer, rt)
        self.assertIsNone(layer.training_mixture)
        self.assertIsNone(layer.training_component(0))
        self.assertIsNone(layer.perception._train)
        layer.certificate_stamp()
        self.assertNotIn("training_mixture", layer.episode_stamp(0)["runtime"])

    def test_factory_attaches_from_the_cfg_field(self):
        m = mixture()
        resolved = jc.resolve_journal_conditions(m.required_tokens(), seed=SEED)
        cfg = F.env_cfg(condition_presets=D1_PRESETS, seed=SEED)
        cfg.journal_training_mixture = m
        cfg.journal_conditions = resolved
        layer = F.api.build_conditions(cfg, 4, "cpu", family="station_keeping")
        self.assertIs(layer.training_mixture, m)
        cfg.journal_conditions = jc.resolve_journal_conditions(["waves=D1"], seed=SEED)
        with self.assertRaises(tm.MixtureError):
            F.api.build_conditions(cfg, 4, "cpu", family="station_keeping")


if __name__ == "__main__":
    unittest.main(verbosity=2)
