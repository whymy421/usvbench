"""F6: bit-for-bit parity of the harbour tensor oracle with E5's scalar HarborMissionEpisode.

200 randomized traces on REAL journal layouts (both vessels, both contract
scopes) drive N scalar oracles and ONE HarborMissionTensorOracle side by side:
every control-step decision (terminated, truncated, reason), every latch
(gate times, transit time, berth entry, hold start, success, clean, validity,
contact, min clearance, path length) and every completed EpisodeResult
(evaluator_fields, NaN-aware exact equality) must match. Traces include gate
crossings with and without arming, reversed passes, berth entries, holds long
enough to succeed, timeouts, analytic contact (clearance <= 0 and non-finite
clearance), D7 contact, out of bounds, capsize, sinking, airborne, non-finite
states, missing clearance and per-motor thrust integrals. Also checks the
torch footprint clearance against journal_geometry_v2 (pose and depth-0 sweep).

CPU only; run: python -B test_journal_contract_parity.py
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import harbour_journal_fixtures as F  # noqa: E402

hj, contracts, plans, geometry = F.hj, F.contracts, F.plans, F.geometry
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def _layouts():
    out = {}
    for vessel in VESSELS:
        out[vessel] = [hj.journal_layout(vessel, 2, F.SEED, env, 0) for env in range(4)]
    return out


LAYOUTS = _layouts()


def _yaw_quat(yaw, roll=0.0, pitch=0.0):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


class Trace:
    """One synthetic COM trajectory along a layout's witness into the berth (plus random events)."""

    def __init__(self, rng, layout, record, spawn_phase):
        self.layout, self.record, self.spawn_phase = layout, record, spawn_phase
        s = float(layout.params["length_scale"])
        pts = np.asarray(layout.witness.vertices_m, dtype=np.float64)
        berth = layout.extras["berth"]
        start = np.array(hj.spawn_pose(layout, spawn_phase)[:2])
        if spawn_phase == 2:
            pts = np.vstack((start, pts[pts[:, 0] > start[0] + 1e-9]))
        self.path = pts
        seg = np.diff(pts, axis=0)
        self.cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(seg, axis=1))))
        self.speed = float(rng.uniform(0.6, 2.2)) * record.cruise_speed_m_s
        self.lateral = float(rng.choice([0.0, rng.uniform(0.0, 0.25 * s), rng.uniform(0.3 * s, 2.0 * s)]))
        self.omega = float(rng.uniform(0.05, 0.6))
        self.heading_noise = float(rng.choice([0.0, rng.uniform(0.0, 0.15), rng.uniform(0.3, 1.8)]))
        self.hold_s = float(rng.uniform(0.0, 9.0))
        self.creep = float(rng.choice([0.0, 0.0, rng.uniform(0.05, 0.6)]))   # residual speed while holding
        self.reverse_at = float(rng.uniform(0.0, 1.0)) if rng.random() < 0.1 else None
        self.berth_xy = np.array(berth["center_xy"])
        self.dt = 0.1
        self.water = float(rng.choice([0.0, 0.0, rng.uniform(-0.05, 0.05)]))
        horizon = self.cum[-1] / self.speed + self.hold_s + 4.0
        self.deadline = round(float(rng.choice([layout.deadline_s, rng.uniform(5.0, horizon)])), 1)
        steps = int(round(self.deadline / self.dt))
        self.steps = max(steps, 1)
        self.deadline = self.steps * self.dt
        self.events = {}
        for name, p in (("contact", 0.12), ("nonfinite_clear", 0.05), ("d7", 0.08), ("oob", 0.05),
                        ("capsize", 0.05), ("sink", 0.04), ("air", 0.04), ("nan", 0.04)):
            if rng.random() < p:
                self.events[name] = int(rng.integers(1, self.steps + 1))
        self.no_clearance = rng.random() < 0.1
        self.thrust = rng.random() < 0.7
        self.rng = rng

    def pose(self, t):
        travel = self.speed * t
        total = self.cum[-1]
        if self.reverse_at is not None and travel > self.reverse_at * total:
            travel = max(0.0, 2 * self.reverse_at * total - travel)
        if travel >= total:
            extra = (travel - total) / self.speed
            p = self.path[-1] + np.array((self.creep * max(0.0, extra - self.hold_s), 0.0))
            yaw = 0.0 + 0.02 * math.sin(3.0 * t) * (self.heading_noise > 0)
            return p, yaw, np.array((self.creep if extra > self.hold_s else 0.0, 0.0))
        i = min(int(np.searchsorted(self.cum, travel, side="right")) - 1, len(self.cum) - 2)
        seg = self.path[i + 1] - self.path[i]
        length = max(float(np.linalg.norm(seg)), 1e-12)
        direction = seg / length
        normal = np.array((-direction[1], direction[0]))
        fade = min(1.0, (total - travel) / max(4.0, 0.1 * total))
        offset = self.lateral * math.sin(self.omega * t) * fade
        p = self.path[i] + (travel - self.cum[i]) * direction + offset * normal
        yaw = math.atan2(direction[1], direction[0]) + self.heading_noise * math.sin(1.3 * t) * fade
        v = self.speed * direction + self.lateral * self.omega * math.cos(self.omega * t) * fade * normal
        if self.reverse_at is not None and self.speed * t > self.reverse_at * total:
            v = -v
        return p, yaw, v

    def state(self, t, k):
        p, yaw, v = self.pose(t)
        roll = pitch = 0.0
        z = self.water + self.record.com_height_above_waterline_m + 0.01 * math.sin(2.0 * t)
        ev = self.events
        if ev.get("capsize") is not None and k >= ev["capsize"]:
            roll = 1.3
        if ev.get("sink") is not None and k >= ev["sink"]:
            z -= 5.0
        if ev.get("air") is not None and k >= ev["air"]:
            z += 5.0
        q = _yaw_quat(yaw, roll, pitch)
        state = dict(com=[float(p[0]), float(p[1]), float(z)], quat=list(q), lin=[float(v[0]), float(v[1]), 0.0],
                     ang=[0.0, 0.0, 0.1 * math.sin(t)], water=self.water,
                     nominal=self.record.com_height_above_waterline_m)
        if ev.get("nan") is not None and k >= ev["nan"]:
            state["com"][0] = float("nan")
        return state

    def clearance(self, k, sub):
        if self.no_clearance:
            return None
        ev = self.events
        if ev.get("contact") is not None and k == ev["contact"] and sub == 1:
            return -0.05
        if ev.get("nonfinite_clear") is not None and k == ev["nonfinite_clear"] and sub == 0:
            return float("nan") if k % 2 else float("inf")
        return float(0.05 + 3.0 * abs(math.sin(0.37 * k + sub)))


def rigid(state):
    return contracts.RigidState(tuple(state["com"]), tuple(state["quat"]), tuple(state["lin"]), tuple(state["ang"]),
                                water_level_m=state["water"], nominal_com_height_m=state["nominal"])


def batch(states):
    return {"com": torch.tensor([s["com"] for s in states], dtype=torch.float64),
            "quat": torch.tensor([s["quat"] for s in states], dtype=torch.float64),
            "lin": torch.tensor([s["lin"] for s in states], dtype=torch.float64),
            "ang": torch.tensor([s["ang"] for s in states], dtype=torch.float64),
            "water": torch.tensor([s["water"] for s in states], dtype=torch.float64),
            "nominal": torch.tensor([s["nominal"] for s in states], dtype=torch.float64)}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def opt(v):
    return None if math.isnan(v) else v


def compare_latches(case, scalars, tensor, rows):
    for i in rows:
        s = scalars[i]
        case.assertEqual(bool(tensor.done[i]), s.done, i)
        case.assertEqual(bool(tensor.success[i]), s.success, i)
        case.assertEqual(bool(tensor.clean[i]), s.clean, i)
        case.assertEqual(int(tensor.validity[i]), int(s.validity.reason), i)
        case.assertEqual(bool(tensor.nonfinite_seen[i]), s.validity.nonfinite_seen, i)
        case.assertEqual(bool(tensor.entered[i]), s.entered, i)
        case.assertTrue(same(opt(float(tensor.hold_started[i])), s.hold_started_s), i)
        case.assertTrue(same(opt(float(tensor.entry_time[i])), s.berth_entry_time_s), i)
        case.assertTrue(same(float(tensor.min_clearance[i]), s.metrics.min_clearance_m), i)
        case.assertTrue(same(float(tensor.full_len[i]), s.metrics.full_episode_path_length_m), i)
        if s.spawn_phase == 0:
            case.assertTrue(same(tuple(opt(float(v)) for v in tensor.gate_times[i]), tuple(s.gate_times_s)), i)
            case.assertTrue(same(opt(float(tensor.transit_time[i])), s.transit_time_s), i)
            case.assertEqual([bool(v) for v in tensor.armed[i]], [bool(v) for v in s._armed], i)


def run_traces(case, vessel, count, seed, spawn_phase):
    rng = np.random.default_rng(seed)
    record = plans.vessel_record(vessel)
    lo, hi = record.heave_limits_m
    limits = contracts.ValidityLimits(record.tilt_limit_rad, lo, hi)
    traces = [Trace(rng, LAYOUTS[vessel][j % 4], record, spawn_phase) for j in range(count)]
    scalars = [hj.scalar_oracle(tr.layout, record, spawn_phase, limits, deadline_s=tr.deadline) for tr in traces]
    tensor = hj.HarborMissionTensorOracle(count, record.polygon_com_xy_m, limits)
    rows = [hj.oracle_row(tr.layout, spawn_phase, tr.deadline) for tr in traces]
    tensor.reset_rows(list(range(count)), **hj.stack_rows(rows))
    for i, tr in enumerate(traces):
        scalars[i].reset(tuple(rows[i]["com_xy"]))
    compare_latches(case, scalars, tensor, range(count))
    outcomes = {}
    max_steps = max(tr.steps for tr in traces)
    for k in range(1, max_steps + 1):
        active = [i for i in range(count) if not scalars[i].done]
        if not active:
            break
        mask = torch.zeros(count, dtype=torch.bool)
        mask[active] = True
        for sub in (0, 1):
            t_sub = (k - 1) * traces[0].dt + (sub + 1) * 0.5 * traces[0].dt
            states = [tr.state(t_sub, k) for tr in traces]
            clear = [tr.clearance(k, sub) for tr in traces]
            thrust = [[float(np.sin(k + i)), float(np.cos(k * 0.5 + i))] for i in range(count)]
            thrust_ok = [tr.thrust for tr in traces]
            for i in active:
                kwargs = dict(clearance_m=clear[i])
                if thrust_ok[i]:
                    kwargs.update(dt_s=0.05, motor_thrust_n=thrust[i])
                scalars[i].observe_physics(rigid(states[i]), **kwargs)
            has = torch.tensor([c is not None for c in clear])
            cvals = torch.tensor([0.0 if c is None else c for c in clear], dtype=torch.float64)
            tensor.observe_physics(batch(states), clearance=cvals, has_clearance=has, dt_s=0.05,
                                   motor_thrust=torch.tensor(thrust, dtype=torch.float64),
                                   thrust_mask=torch.tensor(thrust_ok), rows=mask)
        t = k * traces[0].dt
        states = [tr.state(t, k) for tr in traces]
        clear = [tr.clearance(k, 2) for tr in traces]
        d7 = [tr.events.get("d7") is not None and k >= tr.events["d7"] for tr in traces]
        oob = [tr.events.get("oob") is not None and k >= tr.events["oob"] for tr in traces]
        expected = {}
        for i in active:
            decision = scalars[i].step(t, rigid(states[i]), collision=d7[i], out_of_bounds=oob[i],
                                       clearance_m=clear[i])
            expected[i] = decision
        has = torch.tensor([c is not None for c in clear])
        cvals = torch.tensor([0.0 if c is None else c for c in clear], dtype=torch.float64)
        term, trunc, reason = tensor.step(torch.full((count,), t, dtype=torch.float64), batch(states),
                                          collision=torch.tensor(d7), out_of_bounds=torch.tensor(oob),
                                          clearance=cvals, has_clearance=has, rows=mask)
        for i in active:
            d = expected[i]
            case.assertEqual((bool(term[i]), bool(trunc[i])), (d.terminated, d.truncated), (vessel, i, k))
            case.assertEqual(int(reason[i]) if d.done else 0, int(d.reason) if d.done else 0, (vessel, i, k))
            if d.done:
                a = tensor.last_completed[i].evaluator_fields()
                b = scalars[i].last_completed.evaluator_fields()
                case.assertTrue(same(a, b), (vessel, i, k, {x: (a[x], b.get(x)) for x in a if not same(a[x], b.get(x))}))
                outcomes[i] = d.reason.name
        compare_latches(case, scalars, tensor, active)
    case.assertTrue(all(s.done for s in scalars))
    return outcomes


class TensorScalarParity(unittest.TestCase):
    def test_200_randomized_traces_bit_for_bit(self):
        seen = {}
        plan = (("blueboat_v2", 70, 11, 0), ("wamv_vrx_v2_opt1", 70, 12, 0), ("blueboat_v2", 30, 13, 2),
                ("wamv_vrx_v2_opt1", 30, 14, 2))
        total = 0
        for vessel, count, seed, phase in plan:
            outcomes = run_traces(self, vessel, count, seed, phase)
            total += len(outcomes)
            for name in outcomes.values():
                seen[name] = seen.get(name, 0) + 1
        self.assertEqual(total, 200)
        print("parity outcomes over 200 traces:", dict(sorted(seen.items())), flush=True)
        for reason in ("SUCCESS", "TIMEOUT", "COLLISION", "OUT_OF_BOUNDS", "CAPSIZE", "SINKING", "AIRBORNE",
                       "NONFINITE"):
            self.assertIn(reason, seen, reason)

    def test_reset_rows_isolated_and_errors_match_scalar(self):
        vessel = "blueboat_v2"
        record = plans.vessel_record(vessel)
        lo, hi = record.heave_limits_m
        limits = contracts.ValidityLimits(record.tilt_limit_rad, lo, hi)
        tensor = hj.HarborMissionTensorOracle(3, record.polygon_com_xy_m, limits)
        rows = [hj.oracle_row(LAYOUTS[vessel][j]) for j in range(3)]
        tensor.reset_rows([0, 1, 2], **hj.stack_rows(rows))
        before = {name: getattr(tensor, name).clone() for name in ("armed", "prev_xy", "berth", "deadline_s")}
        tensor.reset_rows([1], **hj.stack_rows([hj.oracle_row(LAYOUTS[vessel][3])]))
        for name, value in before.items():
            self.assertTrue(torch.equal(getattr(tensor, name)[[0, 2]], value[[0, 2]]), name)
        state = {"com": torch.zeros(3, 3, dtype=torch.float64), "quat": torch.tensor([[1.0, 0, 0, 0]] * 3,
                                                                                      dtype=torch.float64),
                 "lin": torch.zeros(3, 3, dtype=torch.float64), "ang": torch.zeros(3, 3, dtype=torch.float64),
                 "water": torch.zeros(3, dtype=torch.float64),
                 "nominal": torch.full((3,), record.com_height_above_waterline_m, dtype=torch.float64)}
        with self.assertRaisesRegex(ValueError, "sample clock"):
            tensor.step(torch.zeros(3, dtype=torch.float64), state)
        with self.assertRaisesRegex(ValueError, "deadline"):
            tensor.step(torch.full((3,), 1.0e6, dtype=torch.float64), state)
        with self.assertRaisesRegex(ValueError, "spawn phases"):
            tensor.reset_rows([0], **{**hj.stack_rows([rows[0]]), "spawn_phase": torch.tensor([1])})


class FootprintClearance(unittest.TestCase):
    def test_torch_pose_and_swept_clearance_match_the_numpy_reference(self):
        rng = np.random.default_rng(5)
        for vessel in VESSELS:
            record = plans.vessel_record(vessel)
            fp = record.footprint
            layout = LAYOUTS[vessel][0]
            centers, radii = np.asarray(layout.centers), np.asarray(layout.radii)
            n = 64
            lo, hi = centers.min(0) - 3, centers.max(0) + 3
            a = np.column_stack((rng.uniform(lo[0], hi[0], n), rng.uniform(lo[1], hi[1], n), rng.uniform(-3, 3, n)))
            b = a + np.column_stack((rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.normal(0, 0.05, n)))
            ta, tb = torch.tensor(a), torch.tensor(b)
            tc = torch.tensor(centers).expand(n, -1, -1)
            tr = torch.tensor(radii).expand(n, -1)
            mask = torch.ones_like(tr, dtype=torch.bool)
            hull = torch.tensor(fp.vertices_m, dtype=torch.float64)
            pose = hj.footprint_pose_clearance(hull, ta, tc, tr, mask)
            swept = hj.footprint_swept_clearance(hull, ta, tb, tc, tr, mask)
            for i in range(n):
                self.assertAlmostEqual(float(pose[i]), geometry.pose_clearance(fp, a[i], centers, radii), places=9)
                self.assertAlmostEqual(float(swept[i]), geometry.swept_clearance(fp, a[i], b[i], centers, radii,
                                                                                  max_depth=0), places=9)
                self.assertLessEqual(float(swept[i]), geometry.swept_clearance(fp, a[i], b[i], centers, radii) + 1e-9)
            empty = hj.footprint_pose_clearance(hull, ta, tc, tr, torch.zeros_like(mask))
            self.assertTrue(bool(torch.isinf(empty).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
