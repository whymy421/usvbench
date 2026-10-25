"""G11: the docking tensor twin with batched libm transfers equals the T-base v2.3 twin bit for bit.

* _libm_batch == the per-call _libm / wrapped results (bits) on random and special inputs, empty
  tensors included;
* REFERENCE_Batch below = DockingEpisodeBatch with the T-base v2.3 observe_physics and step (verbatim;
  per-call libm round trips, step recomputing _state / _yaw / _conj). Both twins are driven in lockstep
  through the recorded and 200 randomized traces of test_journal_docking_contract.py: every code, every
  tensor attribute after every observe / step, and every latched last_completed record are identical;
* DockingEnv._journal_publish_d7 / _journal_d7_contact_now: == the T-base per-field copies
  (REFERENCE_publish / REFERENCE_contact, verbatim) on random sample sets and contact sequences.
Run: python -B tasks/docking/test_g11_sync_free.py   (CPU only)
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = _load("g11_docking_parity", HERE / "test_journal_docking_contract.py")
import torch  # noqa: E402

from tasks.docking import journal_contract_torch as T  # noqa: E402
from tasks.docking.journal_contract_torch import (  # noqa: E402
    DockingEpisodeBatch, F64, RUNNING, NONFINITE, TIMEOUT, SUCCESS, COLLISION, OUT_OF_BOUNDS, _min, cos, sin, hypot,
    wrapped)


def bits(t):
    t = torch.as_tensor(t).detach().contiguous()
    return (str(t.dtype), tuple(t.shape), t.cpu().numpy().tobytes())


class REFERENCE_Batch(DockingEpisodeBatch):
    """T-base v2.3 observe_physics / step (verbatim)."""

    def observe_physics(self, state, collision=None, clearance_m=None, rows=None):
        active = ~self.done if rows is None else (~self.done & rows)
        finite, com, qn, lin, water, nominal = self._state(state)
        observed = self._validity(finite, com, qn, water, nominal)
        self.nonfinite_seen |= active & (observed == NONFINITE)
        self.reason = torch.where(active & (self.reason == RUNNING), observed, self.reason)
        hit = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if collision is None
               else torch.as_tensor(collision, dtype=torch.bool, device=self.device).clone())
        if clearance_m is not None:
            c = torch.as_tensor(clearance_m, dtype=F64, device=self.device)
            c_finite = torch.isfinite(c)
            safe_c = torch.where(c_finite, c, torch.zeros_like(c))
            self.min_clearance = torch.where(active & c_finite, _min(self.min_clearance, safe_c), self.min_clearance)
            hit = hit | ~c_finite | (c_finite & (safe_c <= 0.0))
        self.contact_since_step |= active & hit
        hit2 = (torch.zeros_like(hit) if collision is None
                else torch.as_tensor(collision, dtype=torch.bool, device=self.device))
        if clearance_m is not None:
            hit2 = hit2 | ~(torch.as_tensor(clearance_m, dtype=F64, device=self.device) > 0.0)
        conj, _ = self._conj(finite, com, qn, lin)
        broken = (self.reason != RUNNING) | hit2 | ~finite | ~conj
        self.has_hold = torch.where(active & broken, torch.zeros_like(self.has_hold), self.has_hold)

    def step(self, time_s, state, collision=None, out_of_bounds=None, clearance_m=None):
        t = torch.as_tensor(time_s, dtype=F64, device=self.device).expand(self.n).clone()
        if not bool(self.configured.all()):
            raise RuntimeError("every row must be reset before the first step")
        if bool(self.done.any()):
            raise RuntimeError("reset required after completion (rows %s)" % self.done.nonzero().flatten().tolist())
        if not bool(torch.isfinite(t).all()) or bool((t <= self.prev_t).any()):
            raise ValueError("sample clock must increase from the reset time zero")
        if bool((t > self.deadline_s + 1.0e-9).any()):
            raise ValueError("the adapter must include a sample exactly at the deadline")
        oob = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if out_of_bounds is None
               else torch.as_tensor(out_of_bounds, dtype=torch.bool, device=self.device))
        self.observe_physics(state, collision=collision, clearance_m=clearance_m)
        contact = self.contact_since_step.clone()
        self.contact_since_step = torch.zeros_like(contact)
        self.clean = self.clean & ~contact
        finite, com, qn, lin, _, _ = self._state(state)
        self.valid_prefix_only |= ~finite
        step_len = hypot(com[:, 0] - self.prev_x, com[:, 1] - self.prev_y)
        self.full_len = torch.where(finite, self.full_len + step_len, self.full_len)
        self.prev_x = torch.where(finite, com[:, 0], self.prev_x)
        self.prev_y = torch.where(finite, com[:, 1], self.prev_y)
        self.prev_t = torch.where(finite, t, self.prev_t)
        lx, ly = self._local(com[:, 0], com[:, 1], self.cx, self.cy, self.heading)
        dx = lx - self.prev_lx
        crossing = finite & ~self.entered & (self.prev_lx < self.opening) & (self.opening <= lx) & (dx > 0.0)
        safe_dx = torch.where(crossing, dx, torch.ones_like(dx))
        fraction = (self.opening - self.prev_lx) / safe_dx
        y_cross = self.prev_ly + fraction * (ly - self.prev_ly)
        yaw = self._yaw(qn)
        rel = wrapped(yaw - self.heading)
        sr, cr = sin(rel), cos(rel)
        half = (sr.unsqueeze(1) * self.hull[:, 0].unsqueeze(0) + cr.unsqueeze(1) * self.hull[:, 1].unsqueeze(0))
        half = half.abs().amax(dim=1)
        inward = lin[:, 0] * cos(self.heading) + lin[:, 1] * sin(self.heading)
        enter = crossing & self.clean & (inward > 0.0) & (cr > 0.0) & (y_cross.abs() + half <= self.half_width)
        entry_time = self.prev_lt + fraction * (t - self.prev_lt)
        self.entry_time = torch.where(enter, entry_time, self.entry_time)
        self.has_entry = self.has_entry | enter
        self.entered = self.entered | enter
        self.prev_lx = torch.where(finite, lx, self.prev_lx)
        self.prev_ly = torch.where(finite, ly, self.prev_ly)
        self.prev_lt = torch.where(finite, t, self.prev_lt)
        conj, parts = self._conj(finite, com, qn, lin)
        conj = conj & self.clean
        start = conj & ~self.has_hold
        self.hold_started = torch.where(start, t, self.hold_started)
        self.has_hold = conj.clone()
        dwell = torch.where(conj, t - self.hold_started, torch.zeros_like(t))
        candidate = conj & (dwell >= self.hold_s - 1.0e-9)
        self.conjunction, self.dwell_s = conj, dwell
        safe = (self.reason == RUNNING) & self.clean & ~oob
        latch = candidate & safe & ~self.success
        self.success = self.success | latch
        first = latch & torch.isnan(self.tts)
        self.tts = torch.where(first, t, self.tts)
        self.prefix_len = torch.where(first, self.full_len, self.prefix_len)
        timeout = t >= self.deadline_s - 1.0e-9
        codes = torch.zeros(self.n, dtype=torch.long, device=self.device)
        codes = torch.where(timeout, torch.full_like(codes, TIMEOUT), codes)
        codes = torch.where(self.success, torch.full_like(codes, SUCCESS), codes)
        codes = torch.where(contact, torch.full_like(codes, COLLISION), codes)
        codes = torch.where(oob, torch.full_like(codes, OUT_OF_BOUNDS), codes)
        codes = torch.where(self.reason != RUNNING, self.reason, codes)
        ended = codes != RUNNING
        if bool(ended.any()):
            keep = (codes == SUCCESS) | (codes == TIMEOUT)
            self.success = torch.where(ended & ~keep, torch.zeros_like(self.success), self.success)
            self.done = self.done | ended
            self._latch(ended, codes, finite, parts, lx, ly, dwell)
        return codes


def tensor_state(batch):
    return {k: bits(v) for k, v in vars(batch).items() if isinstance(v, torch.Tensor)}


def same_records(a, b):
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    return sorted(a) == sorted(b) and all(P.same(a[k], b[k]) for k in a)


def lockstep(testcase, episodes, num_envs):
    """run_parity's driving pattern (exhausted rows run an idle episode that never ends), two twins."""
    queues = [list(episodes[i::num_envs]) for i in range(num_envs)]
    new, ref = DockingEpisodeBatch(num_envs, "cpu", P.HULL), REFERENCE_Batch(num_envs, "cpu", P.HULL)
    current, cursor, finished = [None] * num_envs, [0] * num_envs, 0

    def start(i):
        real = bool(queues[i])
        if real:
            ep = queues[i].pop(0)
        else:
            ep = P.approach(P.contracts.Berth((0.0, 0.0), 0.0, -7.0, 0.8, 1.4), P.np.random.default_rng(0), 600.0,
                            0.5, stop_x=-20.0)
        current[i], cursor[i] = ep, 0
        first = ep.steps[0]["subs"][0][0]
        com = first["com"][:2] if all(math.isfinite(v) for v in first["com"][:2]) else (0.0, 0.0)
        b = ep.berth
        for batch in (new, ref):
            batch.reset([i], torch.tensor([com], dtype=torch.float64),
                        berth=dict(center_x=b.center_xy[0], center_y=b.center_xy[1], heading_rad=b.heading_rad,
                                   opening_x_m=b.opening_x_m, back_x_m=b.back_x_m, half_width_m=b.half_width_m),
                        deadline_s=ep.deadline_s, hold_s=ep.hold_s, position_tolerance_m=ep.pos_tol,
                        heading_tolerance_rad=ep.head_tol, speed_tolerance_mps=ep.speed_tol, **P.LIMITS)
        return real

    active = [start(i) for i in range(num_envs)]
    while any(active):
        for j in range(2):
            samples = [current[i].steps[cursor[i]]["subs"][j] for i in range(num_envs)]
            args = (P.batch_state([s[0] for s in samples]),)
            kwargs = dict(collision=torch.tensor([s[1] for s in samples]),
                          clearance_m=torch.tensor([s[2] for s in samples], dtype=torch.float64))
            new.observe_physics(*args, **kwargs)
            ref.observe_physics(*args, **kwargs)
            testcase.assertEqual(tensor_state(new), tensor_state(ref))
        finals = [current[i].steps[cursor[i]]["final"] for i in range(num_envs)]
        times = torch.tensor([(cursor[i] + 1) * P.CONTROL_DT for i in range(num_envs)], dtype=torch.float64)
        kwargs = dict(collision=torch.tensor([f[1] for f in finals]), out_of_bounds=torch.tensor([f[2] for f in finals]),
                      clearance_m=torch.tensor([f[3] for f in finals], dtype=torch.float64))
        a = new.step(times, P.batch_state([f[0] for f in finals]), **kwargs)
        b = ref.step(times, P.batch_state([f[0] for f in finals]), **kwargs)
        testcase.assertEqual(bits(a), bits(b))
        testcase.assertEqual(tensor_state(new), tensor_state(ref))
        if bool((a != 0).any()):
            testcase.assertEqual(new.last_code_list, a.tolist())
        else:
            testcase.assertIsNone(new.last_code_list)
        for i in range(num_envs):
            testcase.assertTrue(same_records(new.last_completed[i], ref.last_completed[i]), i)
            if int(a[i]):
                finished += 1
                active[i] = start(i)
            else:
                cursor[i] += 1
    return finished


def REFERENCE_publish(env, samples):
    rows, xs, ys, rs, phys, vis = [], [], [], [], [], []
    for e, items in samples.items():
        rows.append(int(e))
        if items:
            item = items[0]
            xs.append(float(item.x))
            ys.append(float(item.y))
            rs.append(float(item.radius_m))
            phys.append(bool(item.physical))
            vis.append(bool(item.visible))
        else:
            xs.append(0.0)
            ys.append(0.0)
            rs.append(0.0)
            phys.append(False)
            vis.append(False)
    ids = torch.tensor(rows, dtype=torch.long, device=env.device)
    env._d7_centers[ids, 0, 0] = torch.tensor(xs, dtype=torch.float64, device=env.device)
    env._d7_centers[ids, 0, 1] = torch.tensor(ys, dtype=torch.float64, device=env.device)
    env._d7_radii[ids, 0] = torch.tensor(rs, dtype=torch.float64, device=env.device)
    env._d7_physical[ids, 0] = torch.tensor(phys, dtype=torch.bool, device=env.device)
    env._d7_visible[ids, 0] = torch.tensor(vis, dtype=torch.bool, device=env.device)


class G11Docking(unittest.TestCase):
    def test_libm_batch_equals_per_call(self):
        rng = random.Random(8)
        special = [0.0, -0.0, math.pi, -math.pi, 1e-310, 1e300, math.nan, 0.5, -1.0, 1.0]
        for trial in range(40):
            n = rng.randint(0, 12)
            mk = lambda: torch.tensor([rng.choice(special) if rng.random() < 0.3 else rng.uniform(-10, 10)
                                       for _ in range(n)], dtype=torch.float64)
            a, b, c = mk(), mk(), mk().clamp(-1.0, 1.0)
            got = T._libm_batch([(math.cos, (a,)), (math.atan2, (a, b)), (T.WRAPPED, (b,)), (math.hypot, (a, b)),
                                 (math.acos, (c,)), (math.sqrt, (b.abs(),))])
            want = [cos(a), T.atan2(a, b), wrapped(b), hypot(a, b), T._libm(math.acos, c), T._libm(math.sqrt, b.abs())]
            for x, y in zip(got, want):
                self.assertEqual(bits(x), bits(y))
        for bad in (math.inf, -math.inf):
            x = torch.tensor([0.5, bad], dtype=torch.float64)
            with self.assertRaises(ValueError):
                cos(x)
            with self.assertRaises(ValueError):
                T._libm_batch([(math.hypot, (x, x)), (math.cos, (x,))])

    def test_recorded_traces_lockstep(self):
        self.assertEqual(lockstep(self, P.recorded_episodes(), 3), len(P.recorded_episodes()))

    def test_randomized_traces_lockstep(self):
        episodes = P.random_episodes(200, seed=20261002)
        self.assertEqual(lockstep(self, episodes, 8), 200)

    def test_publish_and_contact_equal_reference(self):
        env_mod = P  # noqa: F841 - parity module already put tasks/ on the import path
        docking_env = importlib.import_module("tasks.docking.docking_env") if "isaaclab" in sys.modules else None
        if docking_env is None:
            stub = _load("g11_docking_env_test", HERE / "test_journal_docking_env.py")
            docking_env = stub.docking_env
        cls = docking_env.DockingEnv
        rng = random.Random(14)
        for case in range(150):
            n = rng.randint(1, 5)

            def make():
                g = torch.Generator().manual_seed(case)
                return NS(num_envs=n, device=torch.device("cpu"), _jd_d7_slot=0,
                          _d7_centers=torch.randn(n, 1, 2, generator=g, dtype=torch.float64),
                          _d7_radii=torch.rand(n, 1, generator=g, dtype=torch.float64),
                          _d7_physical=torch.rand(n, 1, generator=g) < 0.5, _d7_visible=torch.rand(n, 1, generator=g) < 0.5)
            a, b = make(), make()
            keys = list(range(n)) if rng.random() < 0.7 else rng.sample(range(n), rng.randint(1, n))
            rng.shuffle(keys)
            samples = {i: ([] if rng.random() < 0.6 else
                           [NS(slot=0, x=rng.uniform(-9, 9), y=rng.uniform(-9, 9), radius_m=rng.uniform(0.1, 2),
                               physical=rng.random() < 0.6, visible=rng.random() < 0.6)]) for i in keys}
            cls._journal_publish_d7(a, samples)
            REFERENCE_publish(b, samples)
            for name in ("_d7_centers", "_d7_radii", "_d7_physical", "_d7_visible"):
                self.assertEqual(bits(getattr(a, name)), bits(getattr(b, name)), (case, name))
        env = NS(num_envs=3, device=torch.device("cpu"), _d7_bridge=NS(rows={}))
        for k in range(30):
            env._d7_bridge.rows = {i: {"contact": rng.random() < 0.2} for i in range(3) if rng.random() < 0.8}
            got = cls._journal_d7_contact_now(env)
            rows = env._d7_bridge.rows
            want = torch.tensor([bool(rows[i]["contact"]) if i in rows else False for i in range(3)], dtype=torch.bool)
            self.assertEqual(bits(got), bits(want))
            got[0] = True       # the caller may write in place (reset does); the cache must not see it
            self.assertEqual(bits(cls._journal_d7_contact_now(env)), bits(want))


if __name__ == "__main__":
    unittest.main(verbosity=2)
