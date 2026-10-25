"""F5: bit parity of the docking tensor twin with E5's scalar DockingEpisode oracle (CPU only).

The env runs tasks/docking/journal_contract_torch.DockingEpisodeBatch; the
reference is E5's scalar tasks/_shared/journal_contracts_v2.DockingEpisode.
Both consume the SAME physics-substep and control-step samples (the env call
pattern: two observe_physics per control step + step). Every step compares
the codes and the internal latches (validity reason, clean, entered, hold
start, success, time to success, path lengths, minimum clearance); every
completed episode compares EpisodeResult.evaluator_fields() exactly
(NaN == NaN). Traces: a recorded set (deterministic hand-built approaches:
clean dock, wall scrape, exterior hold behind the back wall = C1 finding 27,
capsize, non-finite state, out of bounds, timeout) and 200 randomized traces.

Run: python -B tasks/docking/test_journal_docking_contract.py
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import types
import unittest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if "tasks" not in sys.modules:                      # bare namespaces: no gym registration, no Isaac Lab
    _pkg = types.ModuleType("tasks")
    _pkg.__path__ = [str(REPO / "tasks")]
    sys.modules["tasks"] = _pkg
if "tasks.docking" not in sys.modules:
    _pkg = types.ModuleType("tasks.docking")
    _pkg.__path__ = [str(REPO / "tasks" / "docking")]
    sys.modules["tasks.docking"] = _pkg

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tasks._shared import journal_cfg_v2 as plans  # noqa: E402
from tasks._shared import journal_contracts_v2 as contracts  # noqa: E402
from tasks.docking.journal_contract_torch import DockingEpisodeBatch  # noqa: E402

CONTROL_DT = 1.0 / 60.0


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b) if a == 0.0 else a == b
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b and type(a) is type(b) or (isinstance(a, bool) and isinstance(b, bool) and a == b)


def quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


class Episode:
    """One trace episode: berth, tolerances and a list of control steps of samples."""

    def __init__(self, berth, deadline_s, hold_s, steps, pos_tol=2.5, head_tol=math.radians(15.0), speed_tol=0.3):
        self.berth, self.deadline_s, self.hold_s, self.steps = berth, deadline_s, hold_s, steps
        self.pos_tol, self.head_tol, self.speed_tol = pos_tol, head_tol, speed_tol

    # each step: dict(subs=[(state, collision, clearance)] * 2, final=(state, collision, oob, clearance))


def state_tuple(x, y, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, vx=0.0, vy=0.0, water=0.0, nominal=0.0):
    return dict(com=(x, y, z), quat=quat_from_euler(roll, pitch, yaw), lin=(vx, vy, 0.0), ang=(0.0, 0.0, 0.0),
                water=water, nominal=nominal)


def to_rigid(s):
    return contracts.RigidState(s["com"], s["quat"], s["lin"], s["ang"], water_level_m=s["water"],
                                nominal_com_height_m=s["nominal"])


def batch_state(samples):
    return {name: torch.tensor([s[name] for s in samples], dtype=torch.float64)
            for name in ("com", "quat", "lin", "ang", "water", "nominal")}


HULL = plans.vessel_record("blueboat_v2").polygon_com_xy_m
LIMITS = plans.plan_class("docking", "blueboat_v2")(2).validity_limits


def approach(berth, rng, deadline_s, hold_s, *, lateral=0.0, heading_err=0.0, stop_x=-0.2, exterior=False,
             speed=1.0, events=(), rand=False):
    """Kinematic approach along the berth axis (berth frame), converted to world samples."""
    cx, cy, h = berth.center_xy[0], berth.center_xy[1], berth.heading_rad
    c, s = math.cos(h), math.sin(h)
    n = int(round(deadline_s / CONTROL_DT))
    x0 = berth.opening_x_m - 4.0
    bx, by = x0, lateral
    steps = []
    t = 0.0
    events = dict(events)
    for k in range(1, n + 1):
        subs = []
        for j in range(3):
            if exterior:
                target_x, target_y = berth.back_x_m + 1.4, 0.0
            else:
                target_x, target_y = stop_x, 0.0
            dx, dy = target_x - bx, target_y - by
            dist = math.hypot(dx, dy)
            v = min(speed, dist / 1.5)
            if rand:
                v *= 1.0 + 0.2 * (rng.random() - 0.5)
            step = v * CONTROL_DT / 2.0
            if dist > 1e-9:
                bx += dx / dist * step
                by += dy / dist * step
            yaw = h + heading_err * (1.0 if not rand else rng.random())
            vxw = v * (dx / dist if dist > 1e-9 else 0.0)
            vyw = v * (dy / dist if dist > 1e-9 else 0.0)
            wx, wy = cx + c * bx - s * by, cy + s * bx + c * by
            wvx, wvy = c * vxw - s * vyw, s * vxw + c * vyw
            st = state_tuple(wx, wy, 0.0, 0.0, 0.0, yaw, wvx, wvy)
            clearance = 0.5
            collision = False
            ev = events.get((k, j))
            if ev == "scrape":
                clearance = -0.01
            elif ev == "collide":
                collision = True
            elif ev == "capsize":
                st = state_tuple(wx, wy, 0.0, 1.3, 0.0, yaw, wvx, wvy)
            elif ev == "nan":
                st = dict(st, com=(math.nan, wy, 0.0))
            elif ev == "sink":
                st = dict(st, com=(wx, wy, -2.0))
            elif ev == "clearance_nan":
                clearance = math.nan
            subs.append((st, collision, clearance))
            if j < 2:
                continue
        oob = events.get((k, "oob"), False)
        steps.append(dict(subs=subs[:2], final=(subs[2][0], subs[2][1], oob, subs[2][2])))
        t += CONTROL_DT
    return Episode(berth, n * CONTROL_DT, hold_s, steps)


def recorded_episodes():
    rng = np.random.default_rng(7)
    base = contracts.Berth((0.0, 0.0), 0.0, -7.0, 0.8568, 1.465)
    rot = contracts.Berth((3.0, -2.0), 1.1, -7.0, 0.8568, 1.465)
    return [
        approach(base, rng, 14.0, 0.5),                                    # clean dock + dwell
        approach(rot, rng, 14.0, 0.5, lateral=0.3, heading_err=0.1),       # rotated berth
        approach(base, rng, 14.0, 0.5, events={(200, 1): "scrape"}),       # wall scrape -> collision
        approach(base, rng, 14.0, 0.5, exterior=True),                     # C1 finding 27: behind the back wall
        approach(base, rng, 6.0, 0.5, events={(100, 0): "capsize"}),       # validity: capsize latch
        approach(base, rng, 6.0, 0.5, events={(50, 1): "nan"}),            # non-finite state
        approach(base, rng, 6.0, 0.5, events={(80, 0): "sink"}),           # sinking
        approach(base, rng, 6.0, 0.5, events={(60, "oob"): True}),         # out of bounds
        approach(base, rng, 3.0, 0.5),                                     # timeout before arrival
        approach(base, rng, 14.0, 0.5, heading_err=0.5),                   # heading outside tolerance
        approach(base, rng, 14.0, 0.5, events={(30, 0): "collide"}),       # engine / D7 contact flag
        approach(base, rng, 14.0, 0.5, events={(90, 1): "clearance_nan"}), # non-finite clearance fails closed
    ]


def random_episodes(count, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        berth = contracts.Berth((float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))),
                                float(rng.uniform(-math.pi, math.pi)), -7.0, float(rng.uniform(0.6, 1.2)),
                                float(rng.uniform(1.0, 2.0)))
        events = {}
        for _ in range(int(rng.integers(0, 3))):
            k = int(rng.integers(1, 700))
            kind = str(rng.choice(["scrape", "collide", "capsize", "nan", "sink", "clearance_nan", "oob"]))
            events[(k, "oob") if kind == "oob" else (k, int(rng.integers(0, 3)))] = True if kind == "oob" else kind
        out.append(approach(berth, rng, float(rng.choice([3.0, 8.0, 12.0])), float(rng.choice([0.2, 0.5, 1.0])),
                            lateral=float(rng.uniform(-0.8, 0.8)), heading_err=float(rng.uniform(-0.4, 0.4)),
                            stop_x=float(rng.uniform(-1.5, 0.3)), exterior=bool(rng.random() < 0.15),
                            speed=float(rng.uniform(0.4, 1.4)), events=events, rand=True))
    return out


def run_parity(testcase, episodes, num_envs=4):
    """Drive N rows in lockstep; each row walks its own queue of episodes."""
    queues = [list(episodes[i::num_envs]) for i in range(num_envs)]
    batch = DockingEpisodeBatch(num_envs, "cpu", HULL)
    current = [None] * num_envs
    scalars = [None] * num_envs
    cursor = [0] * num_envs
    completed = 0

    def start(i):
        real = bool(queues[i])
        if real:
            ep = queues[i].pop(0)
        else:                       # exhausted row: an idle long episode that never ends in the loop
            ep = approach(contracts.Berth((0.0, 0.0), 0.0, -7.0, 0.8, 1.4), np.random.default_rng(0), 600.0, 0.5,
                          stop_x=-20.0)
        current[i], cursor[i] = ep, 0
        first = ep.steps[0]["subs"][0][0]
        scalar = contracts.DockingEpisode(ep.berth, HULL, ep.deadline_s, hold_s=ep.hold_s,
                                          position_tolerance_m=ep.pos_tol, heading_tolerance_rad=ep.head_tol,
                                          speed_tolerance_mps=ep.speed_tol,
                                          validity_limits=contracts.ValidityLimits(**LIMITS))
        com = first["com"][:2] if all(math.isfinite(v) for v in first["com"][:2]) else (0.0, 0.0)
        scalar.reset(com)
        scalars[i] = scalar
        dummy[i] = not real
        b = ep.berth
        batch.reset([i], torch.tensor([com], dtype=torch.float64),
                    berth=dict(center_x=b.center_xy[0], center_y=b.center_xy[1], heading_rad=b.heading_rad,
                               opening_x_m=b.opening_x_m, back_x_m=b.back_x_m, half_width_m=b.half_width_m),
                    deadline_s=ep.deadline_s, hold_s=ep.hold_s, position_tolerance_m=ep.pos_tol,
                    heading_tolerance_rad=ep.head_tol, speed_tolerance_mps=ep.speed_tol, **LIMITS)
        return real

    dummy = [False] * num_envs
    active = [start(i) for i in range(num_envs)]
    while any(active):
        for j in range(2):
            samples = [current[i].steps[cursor[i]]["subs"][j] for i in range(num_envs)]
            for i, (st, col, clr) in enumerate(samples):
                scalars[i].observe_physics(to_rigid(st), collision=col, clearance_m=clr)
            batch.observe_physics(batch_state([s[0] for s in samples]),
                                  collision=torch.tensor([s[1] for s in samples]),
                                  clearance_m=torch.tensor([s[2] for s in samples], dtype=torch.float64))
        finals = [current[i].steps[cursor[i]]["final"] for i in range(num_envs)]
        times = [(cursor[i] + 1) * CONTROL_DT for i in range(num_envs)]
        expected = []
        for i, (st, col, oob, clr) in enumerate(finals):
            d = scalars[i].step(times[i], to_rigid(st), collision=col, out_of_bounds=oob, clearance_m=clr)
            expected.append(int(d.reason) if d.done else 0)
        codes = batch.step(torch.tensor(times, dtype=torch.float64), batch_state([f[0] for f in finals]),
                           collision=torch.tensor([f[1] for f in finals]), out_of_bounds=torch.tensor([f[2] for f in finals]),
                           clearance_m=torch.tensor([f[3] for f in finals], dtype=torch.float64))
        testcase.assertEqual(codes.tolist(), expected)
        for i in range(num_envs):
            s = scalars[i]
            hold = s.hold_started_s
            got = dict(reason=int(batch.reason[i]), clean=bool(batch.clean[i]), entered=bool(batch.entered[i]),
                       hold=float(batch.hold_started[i]) if bool(batch.has_hold[i]) else None,
                       success=bool(batch.success[i]), tts=float(batch.tts[i]), full=float(batch.full_len[i]),
                       minc=float(batch.min_clearance[i]), prefix=float(batch.prefix_len[i]),
                       entry=float(batch.entry_time[i]) if bool(batch.has_entry[i]) else None)
            want = dict(reason=int(s.validity.reason), clean=s.clean, entered=s.entered, hold=hold, success=s.success,
                        tts=s.metrics.time_to_success, full=s.metrics.full_episode_path_length_m,
                        minc=s.metrics.min_clearance_m, prefix=s.metrics.success_prefix_path_length_m,
                        entry=s.berth_entry_time_s)
            for key in want:
                testcase.assertTrue(same(got[key], want[key]), (key, got[key], want[key]))
            if expected[i]:
                ref = s.last_completed.evaluator_fields()
                twin = batch.last_completed[i]
                testcase.assertEqual(sorted(ref), sorted(twin))
                for key in ref:
                    testcase.assertTrue(same(twin[key], ref[key]), (key, twin[key], ref[key]))
                if dummy[i]:
                    raise AssertionError("an idle row finished; lengthen the idle episode")
                completed += 1
                active[i] = start(i)
            else:
                cursor[i] += 1
    return completed


class Parity(unittest.TestCase):
    def test_recorded_traces_bit_identical(self):
        episodes = recorded_episodes()
        done = run_parity(self, episodes, num_envs=3)
        self.assertEqual(done, len(episodes))

    def test_recorded_trace_outcomes(self):
        """The recorded scenarios exercise every docking code path (scalar reference)."""
        outcomes = []
        for ep in recorded_episodes():
            s = contracts.DockingEpisode(ep.berth, HULL, ep.deadline_s, hold_s=ep.hold_s,
                                         validity_limits=contracts.ValidityLimits(**LIMITS))
            first = ep.steps[0]["subs"][0][0]["com"][:2]
            s.reset(first)
            for k, step in enumerate(ep.steps):
                for st, col, clr in step["subs"]:
                    s.observe_physics(to_rigid(st), collision=col, clearance_m=clr)
                st, col, oob, clr = step["final"]
                d = s.step((k + 1) * CONTROL_DT, to_rigid(st), collision=col, out_of_bounds=oob, clearance_m=clr)
                if d.done:
                    outcomes.append(int(d.reason))
                    break
        self.assertEqual(outcomes, [1, 1, 2, 3, 6, 9, 7, 4, 3, 3, 2, 2])

    def test_200_randomized_traces_bit_identical(self):
        episodes = random_episodes(200, seed=20260930)
        done = run_parity(self, episodes, num_envs=8)
        self.assertEqual(done, 200)

    def test_reset_rows_are_isolated(self):
        batch = DockingEpisodeBatch(2, "cpu", HULL)
        berth = dict(center_x=0.0, center_y=0.0, heading_rad=0.0, opening_x_m=-7.0, back_x_m=0.8, half_width_m=1.4)
        kwargs = dict(berth=berth, deadline_s=5.0, hold_s=0.5, position_tolerance_m=2.5,
                      heading_tolerance_rad=0.26, speed_tolerance_mps=0.3, **LIMITS)
        batch.reset([0, 1], torch.tensor([[-11.0, 0.0], [-11.0, 0.5]], dtype=torch.float64), **kwargs)
        state = batch_state([state_tuple(-10.9, 0.0, vx=1.0), state_tuple(-10.9, 0.5, vx=1.0)])
        batch.observe_physics(state, clearance_m=torch.tensor([-1.0, 1.0], dtype=torch.float64))
        before = (float(batch.min_clearance[0]), bool(batch.contact_since_step[0]))
        batch.reset([1], torch.tensor([[-11.0, 0.0]], dtype=torch.float64), **kwargs)
        self.assertEqual((float(batch.min_clearance[0]), bool(batch.contact_since_step[0])), before)
        self.assertEqual(float(batch.min_clearance[1]), math.inf)
        self.assertFalse(bool(batch.contact_since_step[1]))

    def test_step_on_a_done_row_raises(self):
        batch = DockingEpisodeBatch(1, "cpu", HULL)
        berth = dict(center_x=0.0, center_y=0.0, heading_rad=0.0, opening_x_m=-7.0, back_x_m=0.8, half_width_m=1.4)
        batch.reset([0], torch.tensor([[-11.0, 0.0]], dtype=torch.float64), berth=berth, deadline_s=CONTROL_DT,
                    hold_s=0.5, position_tolerance_m=2.5, heading_tolerance_rad=0.26, speed_tolerance_mps=0.3,
                    **LIMITS)
        state = batch_state([state_tuple(-11.0, 0.0)])
        self.assertEqual(batch.step(torch.tensor([CONTROL_DT], dtype=torch.float64), state).tolist(), [3])
        with self.assertRaises(RuntimeError):
            batch.step(torch.tensor([2 * CONTROL_DT], dtype=torch.float64), state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
