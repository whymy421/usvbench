"""F1: tensor twin of the E5 ReachEpisode oracle vs the scalar reference (CPU only).

The HazardNav journal env drives journal_hazard_nav.ReachEpisodeTensor; the
scalar journal_contracts_v2.ReachEpisode is the reference. Parity is checked
BIT FOR BIT (decisions and every float of the completed record) on 200
randomized multi-episode traces covering success, collision (flag and
clearance), non-finite / missing clearance, out-of-bounds, capsize, sinking,
airborne, non-finite states and timeouts. The env-recorded traces are
replayed in test_journal_env.py. Also: py_hypot == math.hypot bitwise.

Run: python -B tasks/hazard_nav/test_journal_contract_twin.py
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import random
import struct
import sys
from types import ModuleType
import unittest

sys.dont_write_bytecode = True
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NS = "_f1_twin"
if NS not in sys.modules:
    _root = ModuleType(NS)
    _root.__path__ = [str(REPO / "tasks")]
    sys.modules[NS] = _root
    for _folder in ("_shared", "hazard_nav"):
        _sub = ModuleType(NS + "." + _folder)
        _sub.__path__ = [str(REPO / "tasks" / _folder)]
        sys.modules[NS + "." + _folder] = _sub
J = importlib.import_module(NS + ".hazard_nav.journal_hazard_nav")
C = importlib.import_module(NS + "._shared.journal_contracts_v2")
CONTROL_DT = 1.0 / 60.0


def bits(value):
    if value is None:
        return None
    return struct.pack("<d", float(value))


def same_fields(a, b):
    if set(a) != set(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, float) or isinstance(y, float):
            if bits(x) != bits(y):
                return False
        elif x != y:
            return False
    return True


def quat_from(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


class Trace:
    """One randomized episode input stream for one row (substeps + control steps)."""

    def __init__(self, rng, limits, nominal):
        self.rng = rng
        self.goal = (rng.uniform(5, 40), rng.uniform(-10, 10))
        self.radius = rng.uniform(0.5, 3.0)
        self.steps = rng.randint(3, 40)
        self.deadline = self.steps * CONTROL_DT
        self.start = (rng.uniform(-2, 2), rng.uniform(-2, 2))
        self.limits, self.nominal = limits, nominal
        self.mode = rng.choice(["walk", "goal", "goal", "collide", "clear", "oob", "tilt", "sink", "air",
                                "nan", "noclear", "nanclear"])
        self.fire = rng.randint(1, self.steps)
        self.xy = list(self.start)

    def state(self, k, sub):
        rng = self.rng
        if self.mode == "goal" and k >= self.fire - 1:
            f = min(1.0, (k + 0.5 * sub) / max(1, self.fire))
            self.xy = [self.start[0] + f * (self.goal[0] - self.start[0]),
                       self.start[1] + f * (self.goal[1] - self.start[1])]
        else:
            self.xy = [self.xy[0] + rng.uniform(-0.05, 0.08), self.xy[1] + rng.uniform(-0.05, 0.05)]
        roll, pitch = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
        heave = rng.uniform(-0.1, 0.1)
        water = rng.uniform(-0.05, 0.05)
        if self.mode == "tilt" and k >= self.fire:
            roll = self.limits.capsize_angle_rad + rng.uniform(0.01, 0.5)
        if self.mode == "sink" and k >= self.fire:
            heave = self.limits.min_heave_m - rng.uniform(0.01, 0.5)
        if self.mode == "air" and k >= self.fire:
            heave = self.limits.max_heave_m + rng.uniform(0.01, 0.5)
        quat = quat_from(roll, pitch, rng.uniform(-math.pi, math.pi))
        scale = rng.uniform(0.9, 1.1)
        quat = tuple(q * scale for q in quat)                 # unnormalised on purpose
        com = [self.xy[0], self.xy[1], water + self.nominal + heave]
        lin = [rng.uniform(-1, 1) for _ in range(3)]
        ang = [rng.uniform(-0.3, 0.3) for _ in range(3)]
        if self.mode == "nan" and k >= self.fire and rng.random() < 0.7:
            com[rng.randint(0, 2)] = rng.choice([math.nan, math.inf])
        present = self.mode != "noclear" or rng.random() < 0.3
        clearance = rng.uniform(0.05, 3.0)
        if self.mode == "clear" and k >= self.fire:
            clearance = rng.uniform(-0.3, 0.0)
        if self.mode == "nanclear" and k >= self.fire:
            clearance = rng.choice([math.nan, math.inf, -math.inf])
        collision = self.mode == "collide" and k >= self.fire and sub == 1
        oob = self.mode == "oob" and k >= self.fire
        return dict(com=com, quat=list(quat), lin=lin, ang=ang, water=water, clearance=clearance,
                    present=present, collision=collision, oob=oob)


def scalar_state(s, nominal):
    return C.RigidState(s["com"], s["quat"], s["lin"], s["ang"], water_level_m=s["water"],
                        nominal_com_height_m=nominal)


class TwinParity(unittest.TestCase):
    def run_rows(self, seed, rows=8, episodes=4, decimation=2):
        rng = random.Random(seed)
        limits = C.ValidityLimits(capsize_angle_rad=math.pi / 3, min_heave_m=-0.53, max_heave_m=0.6)
        nominal = 0.0070068735728
        twin = J.ReachEpisodeTensor(rows, "cpu", limits, nominal)
        scalars = [None] * rows
        traces = [None] * rows
        steps = [0] * rows
        ended = [0] * rows
        checked = 0

        def start(i):
            traces[i] = Trace(rng, limits, nominal)
            t = traces[i]
            scalars[i] = C.ReachEpisode(t.goal, t.radius, t.deadline, limits)
            scalars[i].reset(t.start)
            twin.reset([i], [t.start], [t.goal], [t.radius], [t.deadline])
            steps[i] = 0

        for i in range(rows):
            start(i)
        while min(ended) < episodes:
            for sub in range(decimation):
                inputs = [traces[i].state(steps[i], sub) for i in range(rows)]
                for i in range(rows):
                    s = inputs[i]
                    scalars[i].observe_physics(scalar_state(s, nominal),
                                               clearance_m=s["clearance"] if s["present"] else None)
                twin.observe_physics(
                    torch.tensor([s["com"] for s in inputs], dtype=torch.float64),
                    torch.tensor([s["quat"] for s in inputs], dtype=torch.float64),
                    torch.tensor([s["lin"] for s in inputs], dtype=torch.float64),
                    torch.tensor([s["ang"] for s in inputs], dtype=torch.float64),
                    torch.tensor([s["water"] for s in inputs], dtype=torch.float64),
                    clearance_m=torch.tensor([s["clearance"] for s in inputs], dtype=torch.float64),
                    clearance_present=torch.tensor([s["present"] for s in inputs]))
            for i in range(rows):
                steps[i] += 1
            inputs = [traces[i].state(steps[i] - 1, 1) for i in range(rows)]
            times = [steps[i] * CONTROL_DT for i in range(rows)]
            expected = []
            for i in range(rows):
                s = inputs[i]
                d = scalars[i].step(times[i], scalar_state(s, nominal), collision=s["collision"],
                                    out_of_bounds=s["oob"], clearance_m=s["clearance"] if s["present"] else None)
                expected.append(d)
            term, trunc, codes = twin.step(
                torch.tensor(times, dtype=torch.float64),
                torch.tensor([s["com"] for s in inputs], dtype=torch.float64),
                torch.tensor([s["quat"] for s in inputs], dtype=torch.float64),
                torch.tensor([s["lin"] for s in inputs], dtype=torch.float64),
                torch.tensor([s["ang"] for s in inputs], dtype=torch.float64),
                torch.tensor([s["water"] for s in inputs], dtype=torch.float64),
                collision=torch.tensor([s["collision"] for s in inputs]),
                out_of_bounds=torch.tensor([s["oob"] for s in inputs]),
                clearance_m=torch.tensor([s["clearance"] for s in inputs], dtype=torch.float64),
                clearance_present=torch.tensor([s["present"] for s in inputs]))
            for i, d in enumerate(expected):
                self.assertEqual((bool(term[i]), bool(trunc[i])), (d.terminated, d.truncated), (seed, i))
                self.assertEqual(int(codes[i]), int(d.reason) if d.done else 0, (seed, i))
                if d.done:
                    want = scalars[i].last_completed.evaluator_fields()
                    got = twin.completed_fields(i)
                    self.assertTrue(same_fields(want, got), (seed, i, want, got))
                    checked += 1
                    ended[i] += 1
                    start(i)
        return checked

    def test_200_randomized_traces_bit_for_bit(self):
        total = 0
        for seed in range(25):
            total += self.run_rows(seed, rows=8, episodes=1)
        self.assertGreaterEqual(total, 200)
        print("twin parity: %d completed randomized episodes identical bit for bit" % total)

    def test_reasons_cover_every_code(self):
        seen = set()
        rng_seeds = range(40)
        for seed in rng_seeds:
            rng = random.Random(1000 + seed)
            limits = C.ValidityLimits(capsize_angle_rad=math.pi / 3, min_heave_m=-0.53, max_heave_m=0.6)
            twin = J.ReachEpisodeTensor(1, "cpu", limits, 0.007)
            t = Trace(rng, limits, 0.007)
            twin.reset([0], [t.start], [t.goal], [t.radius], [t.deadline])
            for k in range(t.steps):
                s = t.state(k, 1)
                term, trunc, codes = twin.step(
                    torch.tensor([(k + 1) * CONTROL_DT], dtype=torch.float64),
                    torch.tensor([s["com"]], dtype=torch.float64), torch.tensor([s["quat"]], dtype=torch.float64),
                    torch.tensor([s["lin"]], dtype=torch.float64), torch.tensor([s["ang"]], dtype=torch.float64),
                    torch.tensor([s["water"]], dtype=torch.float64), collision=torch.tensor([s["collision"]]),
                    out_of_bounds=torch.tensor([s["oob"]]),
                    clearance_m=torch.tensor([s["clearance"]], dtype=torch.float64),
                    clearance_present=torch.tensor([s["present"]]))
                if bool(term[0] | trunc[0]):
                    seen.add(int(codes[0]))
                    break
        self.assertTrue({1, 2, 3, 4, 6, 7, 8, 9} <= seen, seen)

    def test_done_rows_must_be_reset_and_clock_is_checked(self):
        limits = C.ValidityLimits()
        twin = J.ReachEpisodeTensor(1, "cpu", limits, 0.0)
        twin.reset([0], [(0.0, 0.0)], [(0.1, 0.0)], [1.0], [1.0])
        args = (torch.zeros((1, 3), dtype=torch.float64), torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64),
                torch.zeros((1, 3), dtype=torch.float64), torch.zeros((1, 3), dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64))
        with self.assertRaises(ValueError):
            twin.step(torch.tensor([0.0], dtype=torch.float64), *args)
        with self.assertRaises(ValueError):
            twin.step(torch.tensor([2.0], dtype=torch.float64), *args)
        term, _, codes = twin.step(torch.tensor([CONTROL_DT], dtype=torch.float64), *args)
        self.assertEqual(int(codes[0]), 1)
        with self.assertRaises(RuntimeError):
            twin.step(torch.tensor([2 * CONTROL_DT], dtype=torch.float64), *args)


class PyHypot(unittest.TestCase):
    def test_bitwise_equal_to_math_hypot(self):
        rng = random.Random(7)
        xs, ys = [], []
        for lo, hi in ((-3, 2), (-200, 200), (-300, 300), (-8, -2)):
            for _ in range(50000):
                xs.append(rng.choice((-1, 1)) * rng.random() * 10 ** rng.uniform(lo, hi))
                ys.append(rng.choice((-1, 1)) * rng.random() * 10 ** rng.uniform(lo, hi))
        special = [0.0, -0.0, math.inf, -math.inf, math.nan, 1e-310, 5e-324, 1.0, 3.0]
        xs += [a for a in special for _ in special]
        ys += [b for _ in special for b in special]
        out = J.py_hypot(torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)).tolist()
        bad = [(a, b) for a, b, c in zip(xs, ys, out)
               if not (math.hypot(a, b) == c or (math.isnan(math.hypot(a, b)) and math.isnan(c)))]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
