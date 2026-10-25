"""F2 contract parity: tensor twin vs E5's scalar OrderedPathEpisode (bit for bit), geometry twins.

Run: python -B tasks/path_hazard/tests_journal/test_journal_contract_parity.py   (CPU only)

* exact_hypot == math.hypot on 300,000 random pairs over 40 decades (bitwise);
* validity_codes == journal_contracts_v2.validity_reason on 3,000 random states;
* 200 randomized traces (8 rows in parallel, finished rows reset with the next trace, the env's
  exact call pattern: 2 substep observations with swept clearance and thrust, one pre-step thrust
  observation, then step with collision / out-of-bounds / clearance): every per-step code and every
  field of every EpisodeResult identical to the scalar reference (NaN-aware, bitwise floats);
* recorded env traces: the oracle calls of the REAL journal env (stubbed Isaac Lab, all suites D2 +
  D7 D2) are recorded and replayed into scalar oracles and a fresh tensor twin: identical;
* ThreadingPassage twin == scalar passages on 200 portal traces (active-portal rule, bitwise);
* hull clearance: pose_clearance equals journal_geometry_v2.pose_clearance (polygon footprint);
  swept_clearance is a certified lower bound of densely sampled pose clearances and is exact for a
  stationary hull.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ph_fixtures as F  # noqa: E402

M = F.modules()
J, C = M.journal, M.contracts
LIMITS = C.ValidityLimits(**M.plans.plan_class("path_hazard", "blueboat_v2")(2).validity_limits)
NOMINAL = M.plans.vessel_record("blueboat_v2").com_height_above_waterline_m
DT = 1.0 / 120.0
CONTROL_DT = DT * 2


def same_float(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1.0, a) == math.copysign(1.0, b))
    return a == b


def same_value(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and same_float(a, b)
    return a == b and type(a) is type(b)


def assert_same_result(test, scalar, tensor, label):
    test.assertIsNotNone(scalar, label)
    test.assertIsNotNone(tensor, label)
    for name in ("episode_success", "time_to_success", "success_prefix_path_length_m",
                 "full_episode_path_length_m", "termination_reason"):
        test.assertTrue(same_value(getattr(scalar, name), getattr(tensor, name)),
                        f"{label}: {name} {getattr(scalar, name)!r} != {getattr(tensor, name)!r}")
    test.assertEqual(list(scalar.metrics), list(tensor.metrics), label)
    for key in scalar.metrics:
        test.assertTrue(same_value(scalar.metrics[key], tensor.metrics[key]),
                        f"{label}: metric {key} {scalar.metrics[key]!r} != {tensor.metrics[key]!r}")


def rigid_state(com, quat, lin, ang, water, nominal):
    return C.RigidState(tuple(com), tuple(quat), tuple(lin), tuple(ang), water_level_m=float(water),
                        nominal_com_height_m=float(nominal))


class ExactHypot(unittest.TestCase):
    def test_bitwise_equal_to_math_hypot(self):
        g = torch.Generator().manual_seed(7)
        n = 300_000
        scale = torch.exp((torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 92.0)
        a = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * scale
        b = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * scale * torch.exp(
            (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 8.0)
        out = J.exact_hypot(a, b).tolist()
        bad = [(x, y) for x, y, z in zip(a.tolist(), b.tolist(), out) if math.hypot(x, y) != z]
        self.assertEqual(bad, [])
        special = [0.0, -0.0, 5e-324, 1e-310, 2.2250738585072014e-308, 1e-160, 1.0, 3.0, 4.0, 1e154, 1e308]
        for x in special:
            for y in special:
                self.assertEqual(J.exact_hypot(torch.tensor(x, dtype=torch.float64),
                                               torch.tensor(y, dtype=torch.float64)).item(), math.hypot(x, y))
        self.assertEqual(J.exact_hypot(torch.tensor(math.inf), torch.tensor(1.0)).item(), math.inf)


class Validity(unittest.TestCase):
    def test_validity_codes_match_scalar(self):
        rng = np.random.default_rng(11)
        rows = []
        for k in range(3000):
            com = rng.normal(0.0, 1.0, 3)
            com[2] = rng.choice([NOMINAL, NOMINAL + 0.7, NOMINAL - 0.7, NOMINAL + rng.normal(0, 0.4)])
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.choice([rng.uniform(0, 0.5), rng.uniform(0.9, 1.2), rng.uniform(0, math.pi)])
            quat = np.concatenate(([math.cos(angle / 2)], axis * math.sin(angle / 2))) * rng.uniform(0.5, 2.0)
            lin, ang = rng.normal(size=3), rng.normal(size=3)
            water = rng.normal(0.0, 0.2)
            if k % 97 == 0:
                com[1] = math.nan
            if k % 131 == 0:
                quat[:] = 0.0
            if k % 173 == 0:
                ang[2] = math.inf
            rows.append((com, quat, lin, ang, water))
        com = torch.tensor(np.array([r[0] for r in rows]))
        quat = torch.tensor(np.array([r[1] for r in rows]))
        lin = torch.tensor(np.array([r[2] for r in rows]))
        ang = torch.tensor(np.array([r[3] for r in rows]))
        water = torch.tensor([r[4] for r in rows], dtype=torch.float64)
        codes, _ = J.validity_codes(com, quat, lin, ang, water, NOMINAL, LIMITS)
        want = [int(C.validity_reason(rigid_state(*r, NOMINAL), LIMITS)) for r in rows]
        self.assertEqual(codes.tolist(), want)
        self.assertEqual(set(want), {0, 6, 7, 8, 9})


# ------------------------------------------------------------- random traces --
def random_route(rng):
    pts = [(0.0, 0.0)]
    heading = rng.uniform(-math.pi, math.pi)
    for _ in range(4):
        heading += rng.choice((-1.0, 1.0)) * math.radians(45.0)
        length = rng.uniform(6.0, 12.0)
        pts.append((pts[-1][0] + length * math.cos(heading), pts[-1][1] + length * math.sin(heading)))
    return pts


def trace_generator(seed):
    """Yield (route, radius, deadline_steps, per-step events) for one randomized trace."""
    rng = random.Random(seed)
    route = random_route(rng)
    radius = rng.uniform(1.0, 2.5)
    steps = rng.randint(40, 2400)
    speed = rng.uniform(1.0, 3.0)
    mode = rng.random()
    events = []
    pos = [0.0, 0.0]
    target = 1
    for k in range(steps):
        substates = []
        for _ in range(3):   # two substeps + the control sample
            tx, ty = route[min(target, 4)]
            dx, dy = tx - pos[0], ty - pos[1]
            d = math.hypot(dx, dy) or 1.0
            step = speed * DT
            pos = [pos[0] + step * dx / d + rng.gauss(0, 0.004), pos[1] + step * dy / d + rng.gauss(0, 0.004)]
            if d < radius * 0.3 and target < 4:
                target += 1
            tilt = 0.02 if rng.random() > 0.0003 else rng.uniform(1.1, 1.5)
            yaw = rng.uniform(-math.pi, math.pi)
            quat = (math.cos(tilt / 2) * math.cos(yaw / 2), math.sin(tilt / 2) * math.cos(yaw / 2),
                    math.sin(tilt / 2) * math.sin(yaw / 2), math.cos(tilt / 2) * math.sin(yaw / 2))
            z = NOMINAL + rng.gauss(0.0, 0.05)
            if rng.random() < 0.0002:
                z -= 2.0
            com = [pos[0], pos[1], z]
            if rng.random() < 0.0002:
                com[0] = math.nan
            lin = [rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 0.1)]
            ang = [rng.gauss(0, 0.2), rng.gauss(0, 0.2), rng.gauss(0, 0.5)]
            water = rng.gauss(0.0, 0.03)
            clearance = rng.uniform(0.05, 5.0) if rng.random() > (0.0002 if mode > 0.3 else 0.01) else \
                rng.choice((-0.1, 0.0, math.inf if rng.random() < 0.5 else -math.inf, math.nan))
            thrust = [rng.uniform(-40, 80), rng.uniform(-40, 80)]
            if rng.random() < 0.01:
                thrust[1] = math.nan
            substates.append((com, quat, lin, ang, water, clearance, thrust))
        collision = rng.random() < (0.0005 if mode > 0.7 else 0.0)
        oob = rng.random() < 0.0003
        events.append((substates, collision, oob))
    return route, radius, steps, events


class RandomTraces(unittest.TestCase):
    def test_200_randomized_traces_bitwise(self):
        rows, total = 8, 200
        twin = J.OrderedPathTensorOracle(rows, 4, "cpu", LIMITS, motors=2)
        queue = list(range(total))
        live = [None] * rows
        finished, codes_seen = 0, set()

        def start(r):
            seed = queue.pop(0)
            route, radius, steps, events = trace_generator(1000 + seed)
            scalar = C.OrderedPathEpisode(route, radius, steps * CONTROL_DT, LIMITS)
            scalar.reset(route[0])
            twin.reset([r], torch.tensor([route], dtype=torch.float64), radius, steps * CONTROL_DT,
                       torch.tensor([route[0]], dtype=torch.float64))
            live[r] = {"seed": seed, "scalar": scalar, "events": events, "k": 0}

        for r in range(rows):
            start(r)
        guard = 0
        while any(live):
            guard += 1
            self.assertLess(guard, 100000)
            stacked = []
            for part in range(3):
                com = torch.zeros((rows, 3), dtype=torch.float64)
                quat = torch.zeros((rows, 4), dtype=torch.float64)
                quat[:, 0] = 1.0
                lin = torch.zeros((rows, 3), dtype=torch.float64)
                ang = torch.zeros((rows, 3), dtype=torch.float64)
                water = torch.zeros(rows, dtype=torch.float64)
                clear = torch.full((rows,), 10.0, dtype=torch.float64)
                thrust = torch.zeros((rows, 2), dtype=torch.float64)
                for r, item in enumerate(live):
                    if item is None:
                        continue
                    sub = item["events"][item["k"]][0][part]
                    com[r], quat[r], lin[r], ang[r] = (torch.tensor(v, dtype=torch.float64) for v in sub[:4])
                    water[r], clear[r] = sub[4], sub[5]
                    thrust[r] = torch.tensor(sub[6], dtype=torch.float64)
                stacked.append((com, quat, lin, ang, water, clear, thrust))
            active = torch.tensor([item is not None for item in live])
            # substep 0 (no thrust: the first substep re-observes), substep 1 (thrust), pre-step thrust
            for part, use_thrust in ((0, False), (1, True)):
                com, quat, lin, ang, water, clear, thrust = stacked[part]
                twin.observe_physics(com, quat, lin, ang, water, NOMINAL, clearance=clear,
                                     motor_thrust=thrust if use_thrust else None, dt_s=DT if use_thrust else None,
                                     rows=active)
                for r, item in enumerate(live):
                    if item is not None:
                        s = rigid_state(com[r].tolist(), quat[r].tolist(), lin[r].tolist(), ang[r].tolist(),
                                        float(water[r]), NOMINAL)
                        item["scalar"].observe_physics(s, clearance_m=float(clear[r]),
                                                       motor_thrust_n=thrust[r].tolist() if use_thrust else None,
                                                       dt_s=DT if use_thrust else None)
            com, quat, lin, ang, water, clear, thrust = stacked[2]
            twin.observe_physics(com, quat, lin, ang, water, NOMINAL, motor_thrust=thrust, dt_s=DT, rows=active)
            collision = torch.zeros(rows, dtype=torch.bool)
            oob = torch.zeros(rows, dtype=torch.bool)
            times = torch.zeros(rows, dtype=torch.float64)
            for r, item in enumerate(live):
                if item is None:
                    continue
                s = rigid_state(com[r].tolist(), quat[r].tolist(), lin[r].tolist(), ang[r].tolist(),
                                float(water[r]), NOMINAL)
                item["scalar"].observe_physics(s, motor_thrust_n=thrust[r].tolist(), dt_s=DT)
                collision[r], oob[r] = item["events"][item["k"]][1], item["events"][item["k"]][2]
                times[r] = float(item["k"] + 1) * CONTROL_DT
            # rows that are not live would violate the all-rows-live rule: give them a dummy trace
            if not bool(active.all()):
                for r, item in enumerate(live):
                    if item is None:
                        twin.reset([r], torch.tensor([[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0),
                                                       (20.0, 0.0)]], dtype=torch.float64), 1.0, 1000.0,
                                   torch.zeros((1, 2), dtype=torch.float64))
                        times[r] = twin.prev_time[r] + CONTROL_DT
            codes = twin.step(times, com, quat, lin, ang, water, NOMINAL, collision=collision, out_of_bounds=oob,
                              clearance=clear)
            for r, item in enumerate(live):
                if item is None:
                    continue
                s = rigid_state(com[r].tolist(), quat[r].tolist(), lin[r].tolist(), ang[r].tolist(),
                                float(water[r]), NOMINAL)
                decision = item["scalar"].step(float(times[r]), s, collision=bool(collision[r]),
                                               out_of_bounds=bool(oob[r]), clearance_m=float(clear[r]))
                want = int(decision.reason) if decision.done else 0
                self.assertEqual(int(codes[r]), want, f"trace {item['seed']} step {item['k']}")
                item["k"] += 1
                if decision.done:
                    assert_same_result(self, item["scalar"].last_completed, twin.result(r), f"trace {item['seed']}")
                    codes_seen.add(want)
                    finished += 1
                    live[r] = None
                    if queue:
                        start(r)
        self.assertEqual(finished, total)
        self.assertTrue({1, 2, 3, 6, 9} <= codes_seen, codes_seen)


# ---------------------------------------------------------- recorded traces --
class Recorder:
    """Wraps an OrderedPathTensorOracle and records every call (float64 CPU copies)."""

    def __init__(self, twin):
        self.twin, self.calls = twin, []
        for name in ("reset", "observe_physics", "step"):
            original = getattr(twin, name)

            def wrapper(*args, _name=name, _original=original, **kwargs):
                self.calls.append((_name, _copy(args), _copy(kwargs)))
                return _original(*args, **kwargs)
            setattr(twin, name, wrapper)


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_copy(v) for v in value)
    return value


def replay(test, calls, n, limits, nominal_default):
    twin = J.OrderedPathTensorOracle(n, 4, "cpu", limits, motors=2)
    scalars = [None] * n
    results = []
    for name, args, kwargs in calls:
        if name == "reset":
            ids, route, radius, deadline, start = args
            ids = torch.as_tensor(ids).flatten().tolist()
            twin.reset(ids, route, radius, deadline, start)
            route = torch.as_tensor(route, dtype=torch.float64)
            deadline = torch.as_tensor(deadline, dtype=torch.float64).expand(len(ids))
            start = torch.as_tensor(start, dtype=torch.float64)
            for row, i in enumerate(ids):
                scalars[i] = C.OrderedPathEpisode([tuple(p) for p in route[row].tolist()], float(radius),
                                                  float(deadline[row]), limits)
                scalars[i].reset(tuple(start[row].tolist()))
            continue
        if name == "observe_physics":
            com, quat, lin, ang, water, nominal = args
            twin.observe_physics(com, quat, lin, ang, water, nominal, **kwargs)
            water = torch.as_tensor(water, dtype=torch.float64).expand(n)
            for i in range(n):
                s = rigid_state(com[i].tolist(), quat[i].tolist(), lin[i].tolist(), ang[i].tolist(),
                                float(water[i]), float(nominal))
                thrust = kwargs.get("motor_thrust")
                clearance = kwargs.get("clearance")
                mask = kwargs.get("clearance_mask")
                if mask is not None and not bool(mask[i]):
                    clearance = None
                scalars[i].observe_physics(s, clearance_m=None if clearance is None else float(clearance[i]),
                                           motor_thrust_n=None if thrust is None else thrust[i].tolist(),
                                           dt_s=kwargs.get("dt_s"))
            continue
        time_s, com, quat, lin, ang, water, nominal = args
        codes = twin.step(time_s, com, quat, lin, ang, water, nominal, **kwargs)
        water = torch.as_tensor(water, dtype=torch.float64).expand(n)
        times = torch.as_tensor(time_s, dtype=torch.float64).expand(n)
        for i in range(n):
            s = rigid_state(com[i].tolist(), quat[i].tolist(), lin[i].tolist(), ang[i].tolist(), float(water[i]),
                            float(nominal))
            mask = kwargs.get("clearance_mask")
            given = mask is None or bool(mask[i])
            decision = scalars[i].step(float(times[i]), s, collision=bool(kwargs["collision"][i]),
                                       out_of_bounds=bool(kwargs["out_of_bounds"][i]),
                                       clearance_m=float(kwargs["clearance"][i]) if given else None)
            test.assertEqual(int(codes[i]), int(decision.reason) if decision.done else 0)
            if decision.done:
                assert_same_result(test, scalars[i].last_completed, twin.result(i), f"recorded env {i}")
                results.append(int(decision.reason))
    return results


class RecordedEnvTraces(unittest.TestCase):
    def test_replay_of_the_real_env_calls_is_bitwise_identical(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            cfg = F.journal_cfg(vessel, tokens=F.ALL_ON, num_envs=2)
            env = F.make_env(cfg)
            recorder = Recorder(env._jph_oracle)
            env.reset()
            # re-arm the contract with short deadlines THROUGH the recorded reset (finish inside the test)
            _, (ids, route, radius, _, start), _ = recorder.calls[-1]
            env._jph_oracle.reset(ids, route, radius, torch.tensor([0.5, 0.75], dtype=torch.float64), start)
            for k in range(60):
                env.step(F.actions(k, 2))
            results = replay(self, recorder.calls, 2, env._jph_limits, env._jph_nominal_height)
            self.assertGreaterEqual(len(results), 2, vessel)
            self.assertIn(3, results)


class ThreadingParity(unittest.TestCase):
    def test_threading_twin_matches_scalar_passages_on_200_traces(self):
        rng = random.Random(21)
        total, batch, portals, bonus = 200, 20, 4, 25.0
        paid_seen = failed_seen = 0
        for start in range(0, total, batch):
            radius = [rng.uniform(0.8, 3.0) for _ in range(batch)]
            twin = J.ThreadingTensorTwin(batch, portals, "cpu", bonus)
            twin.reset(list(range(batch)), torch.tensor(radius, dtype=torch.float64))
            scalars = [[C.ThreadingPassage("gate_%d" % (k + 1), -r, r, r, bonus) for k in range(portals)]
                       for r in radius]
            x = [-r - rng.uniform(0.5, 4.0) for r in radius]
            y = [rng.gauss(0.0, 0.3 * r) for r in radius]
            for step in range(900):
                hx = [rng.uniform(0.4, 0.8) for _ in range(batch)]
                hy = [rng.uniform(0.3, 0.6) for _ in range(batch)]
                collision = [rng.random() < 0.003 for _ in range(batch)]
                for i in range(batch):
                    x[i] += rng.uniform(0.005, 0.06) if rng.random() > 0.002 else -0.02
                    y[i] += rng.gauss(0.0, 0.01)
                got = twin.update(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64),
                                  torch.tensor(hx, dtype=torch.float64), torch.tensor(hy, dtype=torch.float64),
                                  torch.tensor(collision))
                for i in range(batch):
                    active = next((p for p in scalars[i] if not (p.paid or p.failed)), None)
                    want = 0.0
                    for passage in scalars[i]:
                        want += passage.update(active.portal_id if active else "none", (x[i], y[i]), hx[i], hy[i],
                                               collision=collision[i])
                    self.assertEqual(float(got[i]), want, (start + i, step))
                    now = next((k for k, p in enumerate(scalars[i]) if not (p.paid or p.failed)), portals)
                    self.assertEqual(int(twin.current[i]), now)
                    if active is not None and (active.paid or active.failed):
                        x[i] = -radius[i] - rng.uniform(0.5, 4.0)       # next portal of the route
                        y[i] = rng.gauss(0.0, 0.3 * radius[i])
            for i in range(batch):
                for k, passage in enumerate(scalars[i]):
                    self.assertEqual(bool(twin.paid[i, k]), passage.paid)
                    self.assertEqual(bool(twin.failed[i, k]), passage.failed)
                    self.assertTrue(same_float(float(twin.worst[i, k]), float(passage.worst_clearance_m)))
                    self.assertEqual(bool(twin.active[i, k]), passage.active)
                    paid_seen += passage.paid
                    failed_seen += passage.failed
        self.assertGreater(paid_seen, 50)
        self.assertGreater(failed_seen, 50)


# ------------------------------------------------------------------ geometry --
class HullClearance(unittest.TestCase):
    def test_pose_clearance_matches_polygon_reference(self):
        rng = np.random.default_rng(5)
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            record = M.plans.vessel_record(vessel)
            rect = J.HullRectangle.for_vessel(vessel)
            footprint = record.geometry.footprint
            s = record.length_scale
            n, k = 400, 6
            pose = torch.tensor(np.column_stack((rng.uniform(-3 * s, 3 * s, n), rng.uniform(-3 * s, 3 * s, n),
                                                 rng.uniform(-math.pi, math.pi, n))))
            centers = torch.tensor(rng.uniform(-4 * s, 4 * s, (n, k, 2)))
            radii = torch.tensor(rng.uniform(0.2 * s, 1.5 * s, (n, k)))
            mask = torch.tensor(rng.random((n, k)) < 0.8)
            got = J.pose_clearance(rect, pose[:, 0], pose[:, 1], pose[:, 2], centers, radii, mask)
            for i in range(n):
                keep = mask[i].numpy()
                want = M.geometry.pose_clearance(footprint, pose[i].numpy(), centers[i].numpy()[keep],
                                                 radii[i].numpy()[keep]) if keep.any() else math.inf
                self.assertTrue(abs(float(got[i]) - want) <= 1e-9 * max(1.0, abs(want)) or
                                (math.isinf(want) and math.isinf(float(got[i]))), (vessel, i, float(got[i]), want))

    def test_swept_clearance_is_a_certified_lower_bound(self):
        rng = np.random.default_rng(9)
        rect = J.HullRectangle.for_vessel("blueboat_v2")
        n, k = 300, 5
        a = torch.tensor(np.column_stack((rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(-3, 3, n))))
        b = a + torch.tensor(np.column_stack((rng.normal(0, 0.03, n), rng.normal(0, 0.03, n),
                                              rng.normal(0, 0.05, n))))
        centers = torch.tensor(rng.uniform(-3, 3, (n, k, 2)))
        radii = torch.tensor(rng.uniform(0.2, 1.0, (n, k)))
        mask = torch.ones((n, k), dtype=torch.bool)
        swept = J.swept_clearance(rect, a, b, centers, radii, mask)
        dyaw = torch.atan2(torch.sin(b[:, 2] - a[:, 2]), torch.cos(b[:, 2] - a[:, 2]))
        dense = []
        for f in np.linspace(0.0, 1.0, 101):
            p = a + f * torch.stack((b[:, 0] - a[:, 0], b[:, 1] - a[:, 1], dyaw), -1)
            dense.append(J.pose_clearance(rect, p[:, 0], p[:, 1], p[:, 2], centers, radii, mask))
        dense = torch.stack(dense).min(dim=0).values
        self.assertTrue(bool((swept <= dense + 1e-12).all()))
        motion = torch.linalg.vector_norm(b[:, :2] - a[:, :2], dim=-1) + rect.core_radius_m * dyaw.abs()
        self.assertTrue(bool((swept >= dense - motion / J.SWEEP_SUBDIVISIONS - 1e-12).all()))
        still = J.swept_clearance(rect, a, a.clone(), centers, radii, mask)
        exact = J.pose_clearance(rect, a[:, 0], a[:, 1], a[:, 2], centers, radii, mask)
        self.assertTrue(torch.equal(still, exact))
        bad = a.clone()
        bad[0, 0] = math.nan
        self.assertEqual(float(J.swept_clearance(rect, bad, b, centers, radii, mask)[0]), -math.inf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
