"""F4: tensor twin of StationKeepingEpisode vs the scalar oracle (bit for bit), CPU only.

Run: python -B tasks/station_keeping/tests_journal/test_station_twin_parity.py

* py_hypot reproduces CPython math.hypot bit for bit (the scalar oracle's
  distance primitive);
* 200 randomized multi-episode traces (validity events, collisions,
  out-of-bounds, non-finite states, in/out-of-region motion, re-resets of
  finished rows while others run): per-step codes and every completed
  EpisodeResult.evaluator_fields() equal bitwise;
* reset-row isolation and identical constructor validation.
No Isaac Lab import (pure tasks/_shared + tasks/station_keeping modules).
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import station_journal_fixtures as F

js = F.ns("station_keeping.journal_station")
contracts = F.ns("_shared.journal_contracts_v2")
plans = F.ns("_shared.journal_cfg_v2")
DTC = js.CONTROL_DT_S
LIMITS = plans.plan_class("station_keeping", "blueboat_v2")(2).validity_limits
NOMINAL = plans.plan_class("station_keeping", "blueboat_v2")(2).nominal_com_height_m


def _quat(rng, tilt_rad):
    axis = rng.normal(size=2)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    yaw = rng.uniform(-math.pi, math.pi)
    half = 0.5 * tilt_rad
    q_tilt = np.array([math.cos(half), axis[0] * math.sin(half), axis[1] * math.sin(half), 0.0])
    q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = q_tilt
    q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return q * rng.uniform(0.5, 2.0)          # unnormalised on purpose (the oracle normalises)


class Scenario:
    """One randomized row: episode parameters + a stateful random walk."""

    def __init__(self, rng):
        self.rng = rng
        self.new_episode()

    def new_episode(self):
        rng = self.rng
        self.station = (float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)))
        self.radius = float(rng.uniform(0.3, 2.0))
        steps = int(rng.integers(20, 160))
        self.deadline = steps * DTC
        self.hold = float(rng.uniform(0.05, 0.8)) * self.deadline
        start = float(rng.uniform(0.0, self.deadline - self.hold))
        self.late = (start, self.deadline)
        self.window = float(rng.uniform(0.1, 1.0)) * self.deadline
        self.mode = int(rng.integers(0, 6))
        angle, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2.5) * self.radius
        self.pos = np.array([self.station[0] + r * math.cos(angle), self.station[1] + r * math.sin(angle), NOMINAL])
        if rng.random() < 0.2:
            self.pos[:2] = self.station          # start exactly at the station
        self.start = (float(self.pos[0]), float(self.pos[1]))

    def state(self):
        rng = self.rng
        step = rng.normal(scale=0.03 * self.radius, size=2)
        if self.mode == 1:
            step += rng.normal(scale=0.3 * self.radius, size=2)      # jittery
        self.pos[:2] += step
        if rng.random() < 0.02:
            self.pos[:2] = self.station                               # snap inside
        tilt = rng.uniform(0, 0.2)
        if rng.random() < 0.003:
            tilt = rng.uniform(1.1, 2.5)                              # capsize
        z = NOMINAL + rng.normal(scale=0.02)
        if rng.random() < 0.002:
            z = NOMINAL + (rng.choice([-1, 1]) * rng.uniform(0.6, 1.5))   # sinking / airborne
        com = [float(self.pos[0]), float(self.pos[1]), float(z)]
        quat = [float(v) for v in _quat(rng, tilt)]
        lin = [float(v) for v in rng.normal(size=3)]
        ang = [float(v) for v in rng.normal(size=3)]
        water = float(rng.normal(scale=0.01))
        if rng.random() < 0.002:
            which = int(rng.integers(0, 4))
            [com, quat, lin, ang][which][int(rng.integers(0, 3))] = float("nan")
        if rng.random() < 0.0005:
            quat = [0.0, 0.0, 0.0, 0.0]
        return com, quat, lin, ang, water

    def flags(self):
        rng = self.rng
        return (bool(rng.random() < (0.004 if self.mode == 2 else 0.0005)),
                bool(rng.random() < (0.004 if self.mode == 3 else 0.0005)))


def stack_state(rows):
    com = torch.tensor([r[0] for r in rows], dtype=torch.float64)
    quat = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    lin = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    ang = torch.tensor([r[3] for r in rows], dtype=torch.float64)
    water = torch.tensor([r[4] for r in rows], dtype=torch.float64)
    return js.TwinState(com, quat, lin, ang, water, torch.full((len(rows),), NOMINAL, dtype=torch.float64))


def scalar_state(row):
    return contracts.RigidState(row[0], row[1], row[2], row[3], water_level_m=row[4], nominal_com_height_m=NOMINAL)


class HypotTests(unittest.TestCase):
    def test_py_hypot_is_bit_identical_to_math_hypot(self):
        rng = np.random.default_rng(7)
        total = 0
        for scale in (1e-6, 1e-3, 1.0, 17.0, 1e3, 1e7):
            a = rng.normal(size=100000) * scale
            b = rng.normal(size=100000) * scale * rng.uniform(0, 1, 100000)
            a[:50] = 0.0
            b[50:100] = 0.0
            b[100:150] = -a[100:150]
            got = js.py_hypot(torch.tensor(a), torch.tensor(b)).tolist()
            for x, y, z in zip(a.tolist(), b.tolist(), got):
                self.assertEqual(math.hypot(x, y), z, (x, y))
                total += 1
        self.assertEqual(total, 600000)
        special = js.py_hypot(torch.tensor([0.0, math.inf, math.nan, 3.0]), torch.tensor([0.0, 1.0, 1.0, math.inf]))
        self.assertEqual(special[0].item(), 0.0)
        self.assertEqual(special[1].item(), math.inf)
        self.assertTrue(math.isnan(special[2].item()))
        self.assertEqual(special[3].item(), math.inf)

    def test_torch_hypot_alone_would_not_be_bit_exact(self):
        # documents why the replica exists (torch.hypot differs in rare last-ulp cases)
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=200000) * 10, rng.normal(size=200000) * 10
        ours = js.py_hypot(torch.tensor(a), torch.tensor(b)).tolist()
        self.assertTrue(all(math.hypot(x, y) == z for x, y, z in zip(a.tolist(), b.tolist(), ours)))


class TwinParityTests(unittest.TestCase):
    ROWS = 200

    def run_traces(self, seed, rows, control_steps, rereset=True):
        rng = np.random.default_rng(seed)
        scen = [Scenario(np.random.default_rng(int(rng.integers(0, 2**31)))) for _ in range(rows)]
        twin = js.StationKeepingTwin(rows, "cpu", **LIMITS)
        validity = contracts.ValidityLimits(**LIMITS)
        oracles = [None] * rows
        completed_scalar, completed_twin, codes_checked = 0, 0, 0
        clocks = [0] * rows

        def reset_rows(ids):
            twin.reset(ids, [scen[i].start for i in ids], [scen[i].station for i in ids],
                       [scen[i].radius for i in ids], [scen[i].deadline for i in ids],
                       hold_s=[scen[i].hold for i in ids], late_interval_s=[scen[i].late for i in ids],
                       top_dose_start_s=0.0, final_window_s=[scen[i].window for i in ids])
            for i in ids:
                s = scen[i]
                oracle = contracts.StationKeepingEpisode(s.station, s.radius, s.deadline, hold_s=s.hold,
                                                         late_interval_s=s.late, top_dose_start_s=0.0,
                                                         final_window_s=s.window, validity_limits=validity)
                oracle.reset(s.start)
                oracles[i] = oracle
                clocks[i] = 0

        reset_rows(list(range(rows)))
        for _ in range(control_steps):
            for _sub in range(2):
                states = [s.state() for s in scen]
                twin.observe_physics(stack_state(states))
                for i, o in enumerate(oracles):
                    o.observe_physics(scalar_state(states[i]))
            states = [s.state() for s in scen]
            flags = [s.flags() for s in scen]
            for i in range(rows):
                clocks[i] += 1
            time_s = torch.tensor([c * DTC for c in clocks], dtype=torch.float64)
            codes = twin.step(time_s, stack_state(states), torch.tensor([f[0] for f in flags]),
                              torch.tensor([f[1] for f in flags]))
            finished = []
            for i, o in enumerate(oracles):
                d = o.step(clocks[i] * DTC, scalar_state(states[i]), collision=flags[i][0],
                           out_of_bounds=flags[i][1])
                expected = int(d.reason) if d.done else 0
                self.assertEqual(int(codes[i]), expected, (seed, i, clocks[i]))
                codes_checked += 1
                if d.done:
                    completed_scalar += 1
                    F.same(twin.completed_result(i), o.last_completed.evaluator_fields(), "row%d" % i)
                    completed_twin += 1
                    finished.append(i)
            if finished and rereset:
                for i in finished:
                    scen[i].new_episode()
                reset_rows(finished)
        return completed_scalar, codes_checked

    def test_200_randomized_traces_bitwise(self):
        completed, checked = self.run_traces(20260930, self.ROWS, 400)
        self.assertGreater(completed, self.ROWS)            # multi-episode rows, all causes mixed
        self.assertEqual(checked, self.ROWS * 400)

    def test_reason_coverage(self):
        rng = np.random.default_rng(5)
        seen = set()
        for seed in rng.integers(0, 2**31, size=3):
            rows = 64
            scen = [Scenario(np.random.default_rng(int(s))) for s in np.random.default_rng(int(seed)).integers(0, 2**31, rows)]
            twin = js.StationKeepingTwin(rows, "cpu", **LIMITS)
            twin.reset(range(rows), [s.start for s in scen], [s.station for s in scen], [s.radius for s in scen],
                       [s.deadline for s in scen], hold_s=[s.hold for s in scen], late_interval_s=[s.late for s in scen],
                       top_dose_start_s=0.0, final_window_s=[s.window for s in scen])
            k = 0
            while not bool(twin.done.all()):
                k += 1
                twin.observe_physics(stack_state([s.state() for s in scen]))
                flags = [s.flags() for s in scen]
                codes = twin.step(torch.full((rows,), k * DTC, dtype=torch.float64), stack_state([s.state() for s in scen]),
                                  torch.tensor([f[0] for f in flags]), torch.tensor([f[1] for f in flags]))
                seen.update(int(c) for c in codes.tolist() if c)
            seen.update(int(r) for r in twin.c_reason.tolist())
            seen.update(("success",) if bool(twin.c_success.any()) else ())
        for code in (2, 3, 4, 6, 9):
            self.assertIn(code, seen)
        self.assertIn("success", seen)

    def test_reset_row_isolation(self):
        rows = 6
        rng = np.random.default_rng(3)
        scen = [Scenario(np.random.default_rng(int(rng.integers(0, 2**31)))) for _ in range(rows)]
        twin = js.StationKeepingTwin(rows, "cpu", **LIMITS)
        args = lambda ids: ([scen[i].start for i in ids], [scen[i].station for i in ids], [scen[i].radius for i in ids],
                            [scen[i].deadline for i in ids])
        twin.reset(range(rows), *args(range(rows)), hold_s=[s.hold for s in scen],
                   late_interval_s=[s.late for s in scen], top_dose_start_s=0.0, final_window_s=[s.window for s in scen])
        for k in range(1, 6):
            twin.observe_physics(stack_state([s.state() for s in scen]))
            twin.step(torch.full((rows,), k * DTC, dtype=torch.float64), stack_state([s.state() for s in scen]))
        snapshot = {name: value.clone() for name, value in vars(twin).items() if isinstance(value, torch.Tensor)}
        fw = {name: value.clone() for name, value in twin.fw.items()}
        twin.reset([1, 4], *args([1, 4]), hold_s=[scen[1].hold, scen[4].hold],
                   late_interval_s=[scen[1].late, scen[4].late], top_dose_start_s=0.0,
                   final_window_s=[scen[1].window, scen[4].window])
        keep = torch.tensor([0, 2, 3, 5])
        for name, before in snapshot.items():
            after = getattr(twin, name)
            if after.shape[:1] == (rows,):
                self.assertTrue(torch.equal(before[keep].nan_to_num(7.0), after[keep].nan_to_num(7.0)), name)
        for name, before in fw.items():
            self.assertTrue(torch.equal(before[keep], twin.fw[name][keep]), name)

    def test_constructor_validation_is_the_scalar_one(self):
        twin = js.StationKeepingTwin(2, "cpu", **LIMITS)
        base = dict(hold_s=[60.0, 60.0], late_interval_s=[(40.0, 100.0), (40.0, 100.0)], top_dose_start_s=0.0,
                    final_window_s=[60.0, 60.0])
        twin.reset([0, 1], [(0, 0), (0, 0)], [(10, 0), (10, 0)], [1.5, 1.5], [100.0, 100.0], **base)
        bad = dict(base, late_interval_s=[(40.0, 99.0), (40.0, 100.0)])
        with self.assertRaisesRegex(ValueError, "late interval"):
            twin.reset([0, 1], [(0, 0), (0, 0)], [(10, 0), (10, 0)], [1.5, 1.5], [100.0, 100.0], **bad)
        bad = dict(base, top_dose_start_s=45.0)
        with self.assertRaisesRegex(ValueError, "ramp"):
            twin.reset([0, 1], [(0, 0), (0, 0)], [(10, 0), (10, 0)], [1.5, 1.5], [100.0, 100.0], **bad)
        with self.assertRaises(ValueError):
            js.StationKeepingTwin(2, "cpu", capsize_angle_rad=4.0, min_heave_m=-1.0, max_heave_m=1.0)

    def test_step_guards_match_the_scalar_guards(self):
        twin = js.StationKeepingTwin(1, "cpu", **LIMITS)
        twin.reset([0], [(0, 0)], [(0, 0)], [1.0], [1.0], hold_s=[0.5], late_interval_s=[(0.5, 1.0)],
                   top_dose_start_s=0.0, final_window_s=[1.0])
        state = stack_state([([0.0, 0.0, NOMINAL], [1.0, 0, 0, 0], [0, 0, 0], [0, 0, 0], 0.0)])
        with self.assertRaisesRegex(ValueError, "sample clock"):
            twin.step(torch.zeros(1, dtype=torch.float64), state)
        with self.assertRaisesRegex(ValueError, "deadline"):
            twin.step(torch.full((1,), 1.5, dtype=torch.float64), state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
