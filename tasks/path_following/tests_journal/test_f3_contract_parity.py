"""F3: the tensor twin reproduces E5's scalar OrderedPathEpisode bit for bit (CPU only).

* py_hypot == math.hypot on this interpreter (the scalar oracle's metric);
* the capsize threshold reproduces math.acos(c) > limit exactly;
* 200 randomized traces (gates passed/skipped/tunnelled, degenerate segments,
  NaN / zero-norm states, capsize, sinking, airborne, substep and step
  contact, out of bounds, timeouts, mid-batch resets of single rows): codes,
  gate events, running metrics and every completed EpisodeResult field
  bitwise equal to the scalar oracle (``journal_contracts_v2``);
* recorded traces of the REAL env (stubbed Isaac Lab, real runtime/layer/
  bridge): a full successful BlueBoat episode plus forced out-of-bounds and
  capsize rows, replayed through the scalar oracle.
Run: python -B test_f3_contract_parity.py
"""
from __future__ import annotations

import math
from pathlib import Path
import struct
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import f3_fixtures as F  # noqa: E402

J = F.jpf
C = F.contracts
DTC = F.plans.CONTROL_DT_S


def bits_equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, float) and isinstance(b, float)):
            return False
        return struct.pack("<d", a) == struct.pack("<d", b)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(bits_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def assert_results_equal(test, scalar, twin, label):
    test.assertIsNotNone(scalar, label)
    test.assertIsNotNone(twin, label)
    fa, fb = scalar.evaluator_fields(), twin.evaluator_fields()
    test.assertEqual(sorted(fa), sorted(fb), label)
    for key in fa:
        test.assertTrue(bits_equal(fa[key], fb[key]), "%s: %s scalar %r twin %r" % (label, key, fa[key], fb[key]))


def state(com, quat, lin, ang, water, nominal):
    return C.RigidState(tuple(com), tuple(quat), tuple(lin), tuple(ang), water_level_m=water,
                        nominal_com_height_m=nominal)


class NumericsTests(unittest.TestCase):
    def test_py_hypot_matches_math_hypot(self):
        rng = np.random.default_rng(7)
        a = (rng.random(200000) - 0.5) * np.exp(rng.normal(0.0, 8.0, 200000))
        b = (rng.random(200000) - 0.5) * np.exp(rng.normal(0.0, 8.0, 200000))
        b[::3] = a[::3] * 1.0e-12
        specials = [(0.0, 0.0), (-0.0, 0.0), (math.inf, math.nan), (math.nan, 1.0), (5e-324, 5e-324),
                    (1e-310, 3e-310), (1e308, 1e308), (3.0, 4.0), (-3.0, -4.0), (2.2250738585072014e-308, 0.0)]
        a = np.concatenate((a, [x for x, _ in specials]))
        b = np.concatenate((b, [y for _, y in specials]))
        got = J.py_hypot(torch.as_tensor(a), torch.as_tensor(b)).tolist()
        for x, y, g in zip(a.tolist(), b.tolist(), got):
            expected = math.hypot(x, y)
            self.assertTrue(bits_equal(expected, g) or (math.isnan(expected) and math.isnan(g)), (x, y, g))
        self.assertEqual(J.assert_hypot_parity(), 20000)

    def test_torch_hypot_is_not_enough(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=400000) * 50, rng.normal(size=400000) * 50
        th = torch.hypot(torch.as_tensor(a), torch.as_tensor(b)).tolist()
        mismatches = sum(math.hypot(x, y) != t for x, y, t in zip(a.tolist(), b.tolist(), th))
        print("torch.hypot differs from math.hypot on %d of 400000 samples (py_hypot: 0)" % mismatches)

    def test_capsize_threshold_is_exact(self):
        for limit in (math.pi / 3.0, 0.1, 1.2, 2.5, math.pi / 2.0):
            c0 = J.capsize_cos_threshold(limit)
            probe = [c0, math.nextafter(c0, -2.0), math.nextafter(c0, 2.0), math.cos(limit), -1.0, 1.0, 0.0]
            rng = np.random.default_rng(int(limit * 1000))
            probe += list(rng.uniform(-1.0, 1.0, 5000)) + list(math.cos(limit) + rng.normal(0, 1e-12, 2000))
            for c in probe:
                c = max(-1.0, min(1.0, float(c)))
                self.assertEqual(math.acos(c) > limit, c < c0, (limit, c))


# ------------------------------------------------------------ random traces --
class TraceGenerator:
    """One batch of random rows; the scalar bank and the twin receive identical float64 inputs."""

    def __init__(self, seed, rows):
        self.rng = np.random.default_rng(seed)
        self.rows = rows

    def episode(self):
        rng = self.rng
        start = np.zeros(2) if rng.random() < 0.5 else rng.uniform(-3.0, 3.0, 2)
        heading = rng.uniform(-math.pi, math.pi)
        points = [start]
        for k in range(4):
            heading += rng.uniform(-1.4, 1.4)
            length = 0.0 if rng.random() < 0.06 else rng.uniform(2.0, 10.0)
            points.append(points[-1] + length * np.array((math.cos(heading), math.sin(heading))))
        route = np.array(points)
        radius = float(rng.uniform(0.4, 2.2))
        deadline = int(rng.integers(30, 320)) * DTC
        if rng.random() < 0.5:
            limits = C.ValidityLimits()
        else:
            limits = C.ValidityLimits(capsize_angle_rad=float(rng.uniform(0.3, 1.5)),
                                      min_heave_m=-float(rng.uniform(0.1, 0.8)), max_heave_m=float(rng.uniform(0.1, 0.8)))
        return dict(route=route, radius=radius, deadline=deadline, limits=limits, nominal=float(rng.uniform(0.0, 0.4)),
                    pos=start.copy() + (rng.normal(0, 0.3, 2) if rng.random() < 0.3 else 0.0), target=1,
                    step_len=float(rng.uniform(0.05, 1.6)) * (3.0 if rng.random() < 0.15 else 1.0),
                    hazard=float(rng.uniform(0.0, 0.004)))

    def sample(self, ep, move):
        """(com xyz, quat, lin, ang, water) of one row; move advances the random walk."""
        rng = self.rng
        route = ep["route"]
        if move:
            target = route[min(ep["target"], 4)]
            delta = target - ep["pos"]
            dist = float(np.hypot(*delta))
            step = ep["step_len"] * rng.uniform(0.3, 1.2)
            if dist > 1e-9:
                ep["pos"] = ep["pos"] + delta / dist * min(step, dist * rng.uniform(0.6, 1.6)) + rng.normal(0, 0.15, 2)
            if dist < ep["radius"] * rng.uniform(0.2, 1.5) and ep["target"] < 4:
                ep["target"] += 1
            if rng.random() < 0.01:
                ep["target"] = min(4, ep["target"] + 1)        # skip a gate on purpose
        water = float(rng.normal(0.0, 0.05))
        heave = float(rng.normal(0.0, 0.05))
        h = ep["hazard"]
        if rng.random() < h:
            heave = ep["limits"].min_heave_m - rng.uniform(0.0, 0.3)
        if rng.random() < h:
            heave = ep["limits"].max_heave_m + rng.uniform(0.0, 0.3)
        yaw = rng.uniform(-math.pi, math.pi)
        roll, pitch = rng.normal(0, 0.05, 2)
        if rng.random() < h:
            roll = ep["limits"].capsize_angle_rad + rng.uniform(-1e-3, 0.5)
        cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                                  math.cos(yaw / 2), math.sin(yaw / 2))
        quat = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy]
        quat = [q * rng.uniform(0.5, 2.0) for q in quat]
        com = [float(ep["pos"][0]), float(ep["pos"][1]), water + ep["nominal"] + heave]
        lin, ang = list(rng.normal(0, 1, 3)), list(rng.normal(0, 0.3, 3))
        if rng.random() < h * 0.5:
            which = rng.integers(0, 4)
            [com, quat, lin, ang][which][int(rng.integers(0, 3))] = float(rng.choice([math.nan, math.inf, -math.inf]))
        if rng.random() < h * 0.3:
            quat = [0.0, 0.0, 0.0, 0.0]
        if rng.random() < h * 0.2:
            water = math.nan
        return com, quat, lin, ang, water


class RandomizedParityTests(unittest.TestCase):
    def run_batch(self, seed, rows=8, steps=360):
        gen = TraceGenerator(seed, rows)
        twin = J.OrderedPathTensorOracle(rows, 4, "cpu")
        eps = [gen.episode() for _ in range(rows)]
        scalars = [None] * rows
        counters = [0] * rows
        completed = 0

        def reset(i):
            ep = eps[i]
            scalars[i] = C.OrderedPathEpisode(tuple(map(tuple, ep["route"])), ep["radius"], ep["deadline"], ep["limits"])
            scalars[i].reset(tuple(ep["route"][0]))
            twin.reset([i], ep["route"][None], ep["radius"], ep["deadline"], ep["limits"], ep["route"][0][None])
            counters[i] = 0

        for i in range(rows):
            reset(i)
        for k in range(steps):
            for sub in range(2):
                samples = [gen.sample(eps[i], move=False) for i in range(rows)]
                collide = [bool(gen.rng.random() < eps[i]["hazard"] * 0.5) for i in range(rows)]
                for i in range(rows):
                    if not scalars[i].done:
                        com, quat, lin, ang, water = samples[i]
                        scalars[i].observe_physics(state(com, quat, lin, ang, water, eps[i]["nominal"]),
                                                   collision=collide[i])
                arrays = [torch.tensor([s[j] for s in samples], dtype=torch.float64) for j in range(5)]
                twin.observe_physics(*arrays[:4], arrays[4], torch.tensor([e["nominal"] for e in eps], dtype=torch.float64),
                                     collision=torch.tensor(collide))
            samples = [gen.sample(eps[i], move=True) for i in range(rows)]
            collide = [bool(gen.rng.random() < eps[i]["hazard"]) for i in range(rows)]
            oob = [bool(gen.rng.random() < eps[i]["hazard"] * 0.7) for i in range(rows)]
            times = []
            for i in range(rows):
                counters[i] += 1
                times.append(counters[i] * DTC)
            expected, gates, xtes = [], [], []
            for i in range(rows):
                scalar = scalars[i]
                if scalar.done:
                    expected.append(0)
                    gates.append(False)
                    xtes.append(None)
                    continue
                before = scalar.next_gate
                com, quat, lin, ang, water = samples[i]
                decision = scalar.step(times[i], state(com, quat, lin, ang, water, eps[i]["nominal"]),
                                       collision=collide[i], out_of_bounds=oob[i])
                expected.append(int(decision.reason) if decision.done else 0)
                gates.append(scalar.next_gate != before)
                xtes.append(scalar.metrics.previous_cross_track_error_m)
            arrays = [torch.tensor([s[j] for s in samples], dtype=torch.float64) for j in range(5)]
            codes, hit, _ = twin.step(torch.tensor(times, dtype=torch.float64), *arrays[:4], arrays[4],
                                      torch.tensor([e["nominal"] for e in eps], dtype=torch.float64),
                                      collision=torch.tensor(collide), out_of_bounds=torch.tensor(oob))
            self.assertEqual(codes.tolist(), expected, (seed, k))
            self.assertEqual(hit.tolist(), gates, (seed, k))
            for i in range(rows):
                scalar = scalars[i]
                label = "seed %d step %d row %d" % (seed, k, i)
                if expected[i] == 0 and not scalar.done:
                    m = scalar.metrics
                    self.assertTrue(bits_equal(float(twin.full_path[i]), m.full_episode_path_length_m), label)
                    self.assertTrue(bits_equal(float(twin.xte_integral[i]), m.path_cross_track_error_integral_m_s), label)
                    self.assertTrue(bits_equal(float(twin.previous_xte[i]), m.previous_cross_track_error_m), label)
                    self.assertEqual(int(twin.next_gate[i]), scalar.next_gate, label)
                    self.assertEqual(int(twin.validity[i]), int(scalar.validity.reason), label)
                if expected[i] != 0:
                    assert_results_equal(self, scalar.last_completed, twin.last_completed[i], label)
                    completed += 1
                    if gen.rng.random() < 0.6:            # reset THIS row only, the others keep running
                        eps[i] = gen.episode()
                        reset(i)
        return completed

    def test_200_randomized_traces(self):
        completed = 0
        for batch in range(25):
            completed += self.run_batch(1000 + batch)
        print("randomized parity: 200 traces (25 batches x 8 rows), %d completed episodes compared" % completed)
        self.assertGreater(completed, 200)

    def test_reason_coverage(self):
        """Every E5 reason the path contract can emit, on deterministic single-row cases (twin == scalar)."""
        seen = set()
        route = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]])
        up = (1.0, 0.0, 0.0, 0.0)
        tilted = (math.cos(0.6), math.sin(0.6), 0.0, 0.0)

        def run(points, *, quat=up, collision=False, oob=False, deadline=1.0, z=0.0, nan=False):
            twin = J.OrderedPathTensorOracle(1, 4, "cpu")
            scalar = C.OrderedPathEpisode(tuple(map(tuple, route)), 1.0, deadline)
            scalar.reset((0.0, 0.0))
            twin.reset([0], route[None], 1.0, deadline, C.ValidityLimits(), np.zeros((1, 2)))
            out = None
            for k, (x, y) in enumerate(points, start=1):
                com = (math.nan if nan else x, y, z)
                d = scalar.step(k * DTC, state(com, quat, (0, 0, 0), (0, 0, 0), 0.0, 0.0), collision=collision,
                                out_of_bounds=oob)
                code, _, _ = twin.step(torch.tensor([k * DTC], dtype=torch.float64),
                                       torch.tensor([com], dtype=torch.float64), torch.tensor([quat], dtype=torch.float64),
                                       torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
                                       torch.zeros(1, dtype=torch.float64), 0.0,
                                       collision=torch.tensor([collision]), out_of_bounds=torch.tensor([oob]))
                self.assertEqual(int(code[0]), int(d.reason) if d.done else 0)
                if d.done:
                    assert_results_equal(self, scalar.last_completed, twin.last_completed[0], str(d.reason))
                    out = int(d.reason)
                    break
            return out

        seen.add(run([(5.0, 0.0), (10.0, 0.2), (15.0, -0.3), (21.0, 0.0)]))       # success (swept last gate)
        seen.add(run([(1.0, 0.0)], collision=True))
        seen.add(run([(1.0, 0.0)], oob=True))
        seen.add(run([(1.0, 0.0)] * 70, deadline=60 * DTC))                          # timeout
        seen.add(run([(1.0, 0.0)], quat=tilted))                                     # capsize
        seen.add(run([(1.0, 0.0)], z=-1.0))                                          # sinking
        seen.add(run([(1.0, 0.0)], z=1.0))                                           # airborne
        seen.add(run([(1.0, 0.0)], nan=True))                                        # non-finite
        seen.add(run([(1.0, 0.0)], quat=(0.0, 0.0, 0.0, 0.0)))                       # zero-norm quaternion
        self.assertEqual(seen, {1, 2, 3, 4, 6, 7, 8, 9})


# ------------------------------------------------------------ recorded traces --
def record(env):
    """Wrap the env's twin to log every reset / observe_physics / step input and output."""
    oracle = env._journal_oracle
    log = []
    orig_reset, orig_observe, orig_step = oracle.reset, oracle.observe_physics, oracle.step

    def clone(x):
        return x.detach().clone() if torch.is_tensor(x) else x

    def reset(env_ids, route_xy, gate_radius_m, deadline_s, validity_limits, com_xy):
        log.append(("reset", [clone(a) for a in (env_ids, route_xy, gate_radius_m, deadline_s, com_xy)],
                    validity_limits))
        return orig_reset(env_ids, route_xy, gate_radius_m, deadline_s, validity_limits, com_xy)

    def observe(com, quat, lin, ang, water, nominal, collision=None):
        log.append(("observe", [clone(a) for a in (com, quat, lin, ang, water)], nominal, clone(collision)))
        return orig_observe(com, quat, lin, ang, water, nominal, collision)

    def step(time_s, com, quat, lin, ang, water, nominal, *, collision=None, out_of_bounds=None):
        out = orig_step(time_s, com, quat, lin, ang, water, nominal, collision=collision, out_of_bounds=out_of_bounds)
        log.append(("step", [clone(a) for a in (time_s, com, quat, lin, ang, water)], nominal,
                    clone(collision), clone(out_of_bounds), [clone(o) for o in out],
                    [r for r in oracle.last_completed]))
        return out

    oracle.reset, oracle.observe_physics, oracle.step = reset, observe, step
    return log


def replay(test, log, n):
    scalars = [None] * n
    compared = {"steps": 0, "completed": 0, "gates": 0}
    for event in log:
        if event[0] == "reset":
            (ids, route, radius, deadline, com), limits = event[1], event[2]
            ids = torch.as_tensor(ids).tolist()
            route = torch.as_tensor(route, dtype=torch.float64).reshape(len(ids), -1, 2)
            radius = torch.as_tensor(radius, dtype=torch.float64).expand(len(ids))
            deadline = torch.as_tensor(deadline, dtype=torch.float64).expand(len(ids))
            com = torch.as_tensor(com, dtype=torch.float64).reshape(len(ids), 2)
            for k, i in enumerate(ids):
                scalars[i] = C.OrderedPathEpisode(tuple(map(tuple, route[k].tolist())), float(radius[k]),
                                                  float(deadline[k]), limits)
                scalars[i].reset(tuple(com[k].tolist()))
        elif event[0] == "observe":
            (com, quat, lin, ang, water), nominal, collision = event[1], event[2], event[3]
            water = torch.as_tensor(water, dtype=torch.float64).expand(n)
            for i in range(n):
                if scalars[i] is not None and not scalars[i].done:
                    scalars[i].observe_physics(state(com[i].tolist(), quat[i].tolist(), lin[i].tolist(),
                                                     ang[i].tolist(), float(water[i]), float(nominal)),
                                               collision=bool(collision[i]) if collision is not None else False)
        else:
            (t, com, quat, lin, ang, water), nominal, collision, oob, out, completed = event[1:]
            water = torch.as_tensor(water, dtype=torch.float64).expand(n)
            codes, hit = out[0].tolist(), out[1].tolist()
            for i in range(n):
                scalar = scalars[i]
                if scalar is None or scalar.done:
                    test.assertEqual(codes[i], 0)
                    continue
                before = scalar.next_gate
                d = scalar.step(float(t[i]), state(com[i].tolist(), quat[i].tolist(), lin[i].tolist(), ang[i].tolist(),
                                                   float(water[i]), float(nominal)),
                                collision=bool(collision[i]), out_of_bounds=bool(oob[i]))
                test.assertEqual(codes[i], int(d.reason) if d.done else 0)
                test.assertEqual(hit[i], scalar.next_gate != before)
                compared["gates"] += int(hit[i])
                compared["steps"] += 1
                if d.done:
                    assert_results_equal(test, scalar.last_completed, completed[i], "recorded env %d" % i)
                    compared["completed"] += 1
    return compared


class RecordedTraceParityTests(unittest.TestCase):
    def test_env_success_trace_blueboat(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, episodes=2)
        env = F.make_env(cfg)
        log = record(env)
        env.reset()
        for k in range(1500):
            env.step(F.seek_actions(env))
            if bool(env.episode_success[0]):
                break
        self.assertTrue(bool(env.episode_success[0]), "the seek driver should complete env 0 in the toy engine")
        result = env.journal_episode_results[0]
        self.assertEqual(result.termination_reason, 1)
        self.assertEqual(result.metrics["ordered_gates_completed"], 4)
        compared = replay(self, log, env.num_envs)
        print("recorded success trace: %s" % compared)
        self.assertGreaterEqual(compared["completed"], 1)
        self.assertGreaterEqual(compared["gates"], 4)

    def test_env_forced_terminations_trace_wamv(self):
        cfg = F.journal_cfg("wamv_vrx_v2_opt1", num_envs=3, episodes=2, tokens=("waves=D2", "perception=D1"))
        env = F.make_env(cfg)
        log = record(env)
        env.reset()
        for k in range(4):
            env.step(F.seek_actions(env))
        env.robot.data.root_link_pose_w[0, 0] += 5000.0                  # out of bounds
        env.robot.data.root_link_pose_w[1, 3:7] = torch.tensor([0.5, 0.8660254, 0.0, 0.0])   # capsize
        env.robot.data.root_link_pose_w[2, 2] += 50.0                    # airborne
        env.step(F.seek_actions(env))
        self.assertEqual(env.termination_reason.tolist(), [4, 6, 8])
        for k in range(4):
            env.step(F.seek_actions(env))
        compared = replay(self, log, env.num_envs)
        print("recorded forced-termination trace: %s" % compared)
        self.assertEqual(compared["completed"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
