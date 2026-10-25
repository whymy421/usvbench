"""G11: the path-following journal step without its avoidable host syncs equals T-base v2.3 bit for bit.

* OrderedPathTensorOracle._rows_f64: a Python-number nominal / water gives the bits of the T-base
  torch.as_tensor(value, float64).expand(N);
* OrderedPathTensorOracle.step: both entry checks (one transfer) raise the T-base errors in the T-base
  order (REFERENCE_entry_checks, verbatim); last_code_list == code.tolist() when a row finished, else None;
* PathFollowingJournalMixin._journal_publish_d7 (all rows present): == the T-base read-back publish
  (REFERENCE_publish, verbatim) on random sample sets, duplicate slots and NaN / -0.0 stored centres included.
Run: python -B tasks/path_following/tests_journal/test_g11_sync_free.py   (CPU only)
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import struct
import sys
from types import SimpleNamespace as NS
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import f3_fixtures as F  # noqa: E402

J = F.jpf
F64 = torch.float64


def bits(t):
    t = torch.as_tensor(t).detach().contiguous()
    return (str(t.dtype), tuple(t.shape), t.cpu().numpy().tobytes())


def REFERENCE_entry_checks(rows, previous_time, deadline, t):
    bad_clock = rows & (~torch.isfinite(t) | (t <= previous_time))
    if bool(bad_clock.any()):
        raise ValueError("sample clock must increase from the reset time zero")
    if bool((rows & (t > deadline + 1.0e-9)).any()):
        raise ValueError("the adapter must include a sample exactly at the deadline")


def REFERENCE_publish(env, samples):
    if not samples:
        return
    rows = sorted(int(e) for e in samples)
    physical = [[False] * env._d7_slots for _ in rows]
    visible = [[False] * env._d7_slots for _ in rows]
    centers = env._d7_centers_local[rows].tolist()
    radii = env._d7_radii[rows].tolist()
    for k, e in enumerate(rows):
        for item in samples[e]:
            slot = int(item.slot)
            if not 0 <= slot < env._d7_slots:
                raise ValueError("D7 sample slot %d outside the reserved storage" % slot)
            centers[k][slot] = [float(item.x), float(item.y)]
            radii[k][slot] = float(item.radius_m)
            physical[k][slot] = bool(item.physical)
            visible[k][slot] = bool(item.visible)
    index = torch.as_tensor(rows, dtype=torch.long, device=env.device)
    env._d7_centers_local[index] = torch.as_tensor(centers, dtype=F64, device=env.device)
    env._d7_radii[index] = torch.as_tensor(radii, dtype=F64, device=env.device)
    env._d7_physical[index] = torch.as_tensor(physical, dtype=torch.bool, device=env.device)
    env._d7_visible[index] = torch.as_tensor(visible, dtype=torch.bool, device=env.device)


def oracle(n=3):
    o = J.OrderedPathTensorOracle(n, 3, "cpu")
    route = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]] * n, dtype=F64)
    limits = F.contracts.ValidityLimits(capsize_angle_rad=1.0, min_heave_m=-1.0, max_heave_m=1.0)
    o.reset(torch.arange(n), route, 1.0, 5.0, limits, route[:, 0])
    return o


def state(n):
    com = torch.zeros(n, 3, dtype=F64)
    return com, torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n), torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n), 0.0


class G11PathFollowing(unittest.TestCase):
    def test_rows_f64(self):
        o = oracle(4)
        for value in (0.25, 3, -0.0, math.inf, math.nan, 1e-310):
            self.assertEqual(bits(o._rows_f64(value)),
                             bits(torch.as_tensor(value, dtype=F64).expand(4).contiguous()))
        t = torch.arange(4, dtype=torch.float32)
        self.assertEqual(bits(o._rows_f64(t)), bits(torch.as_tensor(t, dtype=F64).expand(4)))

    def test_entry_checks_raise_the_reference_errors_in_order(self):
        n = 3
        for t_value, done_row in ((0.1, None), (0.0, None), (math.nan, None), (9.0, None), (9.0, 1),
                                  ([0.1, -1.0, 9.0], None), ([0.1, 0.2, 9.0], 2), ([math.nan, 9.0, 0.1], 0)):
            o = oracle(n)
            if done_row is not None:
                o.done[done_row] = True
            t = torch.as_tensor(t_value, dtype=F64).expand(n)
            expected = got = None
            try:
                REFERENCE_entry_checks(o.armed & ~o.done, o.previous_time.clone(), o.deadline.clone(), t)
            except ValueError as exc:
                expected = str(exc)
            try:
                o.step(torch.as_tensor(t_value, dtype=F64), *state(n))
            except ValueError as exc:
                got = str(exc)
            self.assertEqual(got, expected, (t_value, done_row))

    def test_last_code_list(self):
        n = 3
        o = oracle(n)
        code, _, _ = o.step(torch.tensor(1.0 / 60.0, dtype=F64), *state(n))
        self.assertIsNone(o.last_code_list)
        self.assertFalse(bool((code != 0).any()))
        code, _, _ = o.step(torch.tensor(2.0 / 60.0, dtype=F64), *state(n),
                            out_of_bounds=torch.tensor([False, True, False]))
        self.assertEqual(o.last_code_list, code.tolist())
        self.assertNotEqual(o.last_code_list[1], 0)

    def test_publish_all_rows_equals_reference(self):
        publish = J.PathFollowingJournalMixin._journal_publish_d7
        rng = random.Random(21)
        for case in range(200):
            n, slots = rng.randint(1, 5), rng.randint(1, 3)

            def make():
                g = torch.Generator().manual_seed(case)
                centers = torch.randn(n, slots, 2, generator=g, dtype=F64) * 10
                centers[0, 0, 0] = -0.0
                if n > 1:
                    centers[1, 0, 1] = math.nan
                return NS(num_envs=n, device=torch.device("cpu"), _d7_slots=slots, _d7_centers_local=centers,
                          _d7_radii=torch.rand(n, slots, generator=g, dtype=F64),
                          _d7_physical=torch.rand(n, slots, generator=g) < 0.5,
                          _d7_visible=torch.rand(n, slots, generator=g) < 0.5,
                          _journal_publish_d7_all_rows=lambda s, r, env=None: None)
            a, b = make(), make()
            a._journal_publish_d7_all_rows = lambda s, r: J.PathFollowingJournalMixin._journal_publish_d7_all_rows(a, s, r)
            keys = list(range(n)) if rng.random() < 0.7 else rng.sample(range(n), rng.randint(0, n))
            samples = {i: [NS(slot=rng.randrange(slots), x=rng.uniform(-9, 9), y=rng.uniform(-9, 9),
                              radius_m=rng.uniform(0.1, 2), physical=rng.random() < 0.7, visible=rng.random() < 0.7)
                           for _ in range(rng.choice((0, 0, 1, 2)))] for i in keys}
            publish(a, samples)
            REFERENCE_publish(b, samples)
            for name in ("_d7_centers_local", "_d7_radii", "_d7_physical", "_d7_visible"):
                self.assertEqual(bits(getattr(a, name)), bits(getattr(b, name)), (case, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
