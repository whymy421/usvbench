"""G28 item 1: perception outage start = reference-route rule (CPU only).

Owner decision 2026-10-04 (NC 39 section 2 item 4): the fixed 5-20 s start window is replaced by a rule.
Station keeping keeps its key phase (outage_start_s); every other task starts the outage at f x T_ref,
T_ref = reference route length / reference speed, f keyed uniform in [0.40, 0.60] per scenario key.

* the start is a pure function of (seed, env, episode, T_ref): no measurement, no action, no other axis,
  no reset order or batching changes it (controller- and progress-independent);
* f uses draw 0 of the keyed perception.outage_start stream, i.e. the same uniform the old window used;
* the critical-phase path and the legacy fallback window are unchanged;
* invalid inputs and the ambiguous call (both rules) are refused;
* the per-episode stamp names the rule, f and T_ref;
* cells without an outage are untouched: reference_duration_s is ignored and every output is identical;
* through the condition layer (reset_runtime) the same rule applies.

Run: python -B tasks/_shared/conditions/tests/test_g28_outage_rule.py   (or pytest)
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
PV = F.perception_v3
DT_CONTROL = F.DT * F.DECIMATION


def degrader(n=8, seed=5, contract=("hazard_nav", "v3"), **perception):
    values = {"enabled": True}
    values.update(perception)
    cfg = api.resolve_v3((), {"perception": values}).perception
    return PV.PerceptionV3Degrader(cfg, n, "cpu", seed, F.observation_groups.lookup(*contract), DT_CONTROL,
                                   ray_count=36, ray_max_range_m=30.0)


def reset(d, ids=None, episode=0, axis=0.0, **kwargs):
    ids = torch.arange(d.num_envs) if ids is None else torch.as_tensor(ids)
    episodes = torch.full((ids.numel(),), episode, dtype=torch.long) if np.isscalar(episode) else \
        torch.as_tensor(episode, dtype=torch.long)
    ordinals = F.strata.scenario_ordinals(ids, episodes, d.num_envs)
    strata = F.strata.stratum_indices(ordinals, d.seed)
    d.reset(ids, episodes, strata, torch.full((ids.numel(),), axis, dtype=torch.float64), **kwargs)
    return strata


def expected_fraction(seed, env, episode):
    key = np.uint64(KR.stream_key(seed, env, episode, "perception.outage_start"))
    u = float(KR.uniforms(np.asarray([key]), np.zeros(1))[0])
    return 0.40 + u * (0.60 - 0.40), u          # the rule's own arithmetic: lo + u (hi - lo)


class RouteRule(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(PV.OUTAGE_ROUTE_FRACTION, (0.40, 0.60))
        self.assertEqual(PV.OUTAGE_ROUTE_SOURCE, "reference_route_fraction")

    def test_start_is_f_times_reference_duration_with_f_keyed_in_the_middle_band(self):
        n = 16
        d = degrader(n=n, outage_s=1.0)
        reference = torch.linspace(19.0, 330.0, n, dtype=torch.float64)      # BlueBoat docking .. WAM-V harbour
        for episode in (0, 1, 7):
            reset(d, episode=episode, reference_duration_s=reference)
            for i in range(n):
                f, _ = expected_fraction(d.seed, i, episode)
                self.assertTrue(0.40 <= f <= 0.60)
                self.assertEqual(int(d.outage_start[i]), int(math.floor(f * float(reference[i]) / DT_CONTROL)))
                record = d.episode_parameters(i)
                self.assertEqual(record["outage_source"], "reference_route_fraction")
                self.assertEqual(record["outage_start_fraction"], f)
                self.assertEqual(record["outage_reference_duration_s"], float(reference[i]))
                self.assertEqual(record["outage_fraction_window"], [0.40, 0.60])
                key = KR.stream_key(d.seed, i, episode, "perception.outage_start")
                self.assertEqual(record["outage_stream_key"], format(key, "016x"))
                self.assertEqual(record["outage_end_frame"] - record["outage_start_frame"], d.outage)

    def test_same_uniform_as_the_old_window(self):
        a, b = degrader(outage_s=0.3), degrader(outage_s=0.3)
        reset(a, episode=3)                                                    # legacy fallback window
        reset(b, episode=3, reference_duration_s=torch.full((8,), 40.0, dtype=torch.float64))
        for i in range(8):
            f, u = expected_fraction(a.seed, i, 3)
            self.assertEqual(int(a.outage_start[i]), int(math.floor((5.0 + u * 15.0) / DT_CONTROL)))
            self.assertEqual(int(b.outage_start[i]), int(math.floor(f * 40.0 / DT_CONTROL)))
            self.assertEqual(a.keys["outage"][i], b.keys["outage"][i])

    def test_deterministic_per_scenario_key_and_independent_of_order_axes_and_inputs(self):
        reference = torch.full((8,), 64.0, dtype=torch.float64)
        plain = degrader(outage_s=1.0)
        others = degrader(outage_s=1.0, pos_noise_sigma_m=0.4, pos_bias_m=0.5, latency_s=0.10)
        reset(plain, reference_duration_s=reference)
        reset(others, axis=1.3, reference_duration_s=reference)
        self.assertTrue(np.array_equal(plain.outage_start, others.outage_start))
        batched = degrader(outage_s=1.0)
        for batch in ([6, 1], [3, 0, 5], [7, 4, 2]):
            reset(batched, ids=batch, reference_duration_s=reference[: len(batch)])
        self.assertTrue(np.array_equal(plain.outage_start, batched.outage_start))
        # progress / controller independence: measurements and steps never move the start
        start = plain.outage_start.copy()
        for k in range(200):
            plain.apply("pos", torch.full((8, 2), float(k) * 0.37))
        self.assertTrue(np.array_equal(plain.outage_start, start))
        other_seed = degrader(seed=6, outage_s=1.0)
        reset(other_seed, reference_duration_s=reference)
        self.assertFalse(np.array_equal(plain.outage_start, other_seed.outage_start))
        later = degrader(outage_s=1.0)
        reset(later, episode=1, reference_duration_s=reference)
        self.assertFalse(np.array_equal(plain.outage_start, later.outage_start))

    def test_outage_lies_inside_the_reference_transit_and_before_the_deadline(self):
        d = degrader(n=64, outage_s=3.0)
        for t_ref in (8.0, 19.0, 40.0, 100.0, 330.0):
            reference = torch.full((64,), t_ref, dtype=torch.float64)
            for episode in range(3):
                reset(d, episode=episode, reference_duration_s=reference)
                start_s = d.outage_start * DT_CONTROL
                end_s = (d.outage_start + d.outage) * DT_CONTROL
                self.assertTrue(bool((start_s >= 0.40 * t_ref - DT_CONTROL).all()))
                self.assertTrue(bool((start_s <= 0.60 * t_ref).all()))
                self.assertTrue(bool((end_s < 2.0 * t_ref).all()))                # deadline >= 2 L_nom / u

    def test_critical_phase_and_fallback_paths_are_unchanged(self):
        d = degrader(outage_s=0.30)
        reset(d, outage_start_s=torch.full((8,), 10 * DT_CONTROL, dtype=torch.float64))
        self.assertEqual(d.episode_parameters(0)["outage_source"], "task_critical_phase")
        self.assertNotIn("outage_start_fraction", d.episode_parameters(0))
        self.assertTrue(bool((d.outage_start == 10).all()))
        fallback = degrader(outage_s=1.0)
        reset(fallback)
        self.assertEqual(fallback.episode_parameters(3)["outage_source"], "fallback_window_PENDING")
        self.assertNotIn("outage_start_fraction", fallback.episode_parameters(3))

    def test_refusals(self):
        d = degrader(outage_s=1.0)
        with self.assertRaises(ValueError):
            reset(d, outage_start_s=torch.zeros(8, dtype=torch.float64),
                  reference_duration_s=torch.full((8,), 10.0, dtype=torch.float64))
        for bad in (torch.zeros(8, dtype=torch.float64), torch.full((8,), float("nan"), dtype=torch.float64),
                    torch.full((7,), 10.0, dtype=torch.float64), torch.full((8,), -1.0, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                reset(d, reference_duration_s=bad)


class NonOutageCellsUnchanged(unittest.TestCase):
    def test_reference_duration_is_ignored_without_an_outage(self):
        for axes in ({}, {"pos_noise_sigma_m": 0.4}, {"pos_bias_m": 0.5}, {"latency_s": 0.10},
                     {"pos_noise_sigma_m": 0.1, "latency_s": 0.05}):
            a, b = degrader(**axes), degrader(**axes)
            reset(a, axis=0.4)
            reset(b, axis=0.4, reference_duration_s=torch.full((8,), 33.0, dtype=torch.float64))
            self.assertEqual(a.episode_parameters(2), b.episode_parameters(2))
            self.assertEqual(a.key_derivations, b.key_derivations)
            for k in range(60):
                x = torch.full((8, 2), float(k))
                if a.active:
                    self.assertTrue(torch.equal(a.apply("pos", x), b.apply("pos", x)))
            self.assertNotIn("outage_source", b.episode_parameters(0))


class ConditionLayer(unittest.TestCase):
    def test_reset_runtime_passes_the_reference_duration(self):
        layer, rt = F.build(("journal.perception.dropout.D3",), family="hazard_nav", n=4)
        reference = torch.tensor([20.0, 33.0, 64.0, 150.0], dtype=torch.float64)
        F.reset(layer, rt, reference_duration_s=reference)
        p = layer.perception
        for i in range(4):
            f, _ = expected_fraction(layer.cfg.seed, i, int(p.episode[i]))
            self.assertEqual(int(p.outage_start[i]), int(math.floor(f * float(reference[i]) / DT_CONTROL)))
            self.assertEqual(p.episode_parameters(i)["outage_source"], "reference_route_fraction")

    def test_calm_layer_ignores_the_reference_duration(self):
        a, rta = F.build((), family="hazard_nav", n=4)
        b, rtb = F.build((), family="hazard_nav", n=4)
        F.reset(a, rta)
        F.reset(b, rtb, reference_duration_s=torch.full((4,), 50.0, dtype=torch.float64))
        self.assertEqual(a.episode_stamp(1), b.episode_stamp(1))

    def test_decision_label(self):
        entry = F.hydro_v2_layer.PENDING_OWNER_DECISIONS["perception_outage_start"]
        self.assertTrue(entry["status"].startswith("DECIDED"))
        self.assertIn("[0.40, 0.60]", entry["default"])


if __name__ == "__main__":
    unittest.main()
