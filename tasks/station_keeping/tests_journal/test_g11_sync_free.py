"""G11: the station-keeping journal step without its avoidable host syncs equals T-base v2.3 bit for bit.

* StationKeepingTwin.step: both entry checks (one transfer) raise the T-base errors in the T-base order
  (REFERENCE_entry_checks, verbatim); last_any_finished == bool((codes != 0).any()) after every step;
* StationKeepingEnv._journal_publish_d7: all-empty fill path and the one-copy sample path == the T-base
  per-field copies (REFERENCE_publish, verbatim) on random sample sets (empty rows, intruders, subsets).
Run: python -B tasks/station_keeping/tests_journal/test_g11_sync_free.py   (CPU only)
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import station_journal_fixtures as F  # noqa: E402

js = F.ns("station_keeping.journal_station")


def bits(t):
    t = torch.as_tensor(t).detach().contiguous()
    return (str(t.dtype), tuple(t.shape), t.cpu().numpy().tobytes())


def REFERENCE_entry_checks(act, prev_time, deadline, t):
    if bool((act & ~(torch.isfinite(t) & (t > prev_time))).any()):
        raise ValueError("sample clock must increase from the reset time zero")
    if bool((act & (t > deadline + 1.0e-9)).any()):
        raise ValueError("the adapter must include a sample exactly at the deadline")


def REFERENCE_publish(env, samples):
    if not samples:
        return
    rows, xy, radius, physical, visible, empty = [], [], [], [], [], []
    origins = env.scene.env_origins
    for e, items in samples.items():
        if not items:
            empty.append(int(e))
            continue
        if len(items) != 1 or int(items[0].slot) != 0:
            raise ValueError("station keeping stores exactly one D7 object in slot 0 (capacity 0 + 1)")
        item = items[0]
        rows.append(int(e))
        xy.append((float(item.x), float(item.y)))
        radius.append(float(item.radius_m))
        physical.append(bool(item.physical))
        visible.append(bool(item.visible))
    if empty:
        ids = torch.as_tensor(empty, dtype=torch.long, device=env.device)
        env._d7_physical[ids, 0] = False
        env._d7_visible[ids, 0] = False
        env._d7_radii[ids, 0] = 0.0
    if rows:
        ids = torch.as_tensor(rows, dtype=torch.long, device=env.device)
        local = torch.as_tensor(xy, dtype=env._d7_centers_w.dtype, device=env.device)
        env._d7_centers_w[ids, 0] = local + origins[ids, :2].to(local.dtype)
        env._d7_radii[ids, 0] = torch.as_tensor(radius, dtype=env._d7_radii.dtype, device=env.device)
        env._d7_physical[ids, 0] = torch.as_tensor(physical, dtype=torch.bool, device=env.device)
        env._d7_visible[ids, 0] = torch.as_tensor(visible, dtype=torch.bool, device=env.device)


def twin(n):
    t = js.StationKeepingTwin(n, "cpu", capsize_angle_rad=1.0, min_heave_m=-1.0, max_heave_m=1.0)
    t.reset(torch.arange(n), torch.zeros(n, 2), torch.zeros(n, 2), torch.full((n,), 5.0), torch.full((n,), 3.0),
            hold_s=torch.full((n,), 1.0), late_interval_s=torch.tensor([[1.0, 3.0]] * n),
            top_dose_start_s=torch.zeros(n), final_window_s=torch.full((n,), 1.0))
    return t


def state(n, x=0.0):
    com = torch.zeros(n, 3, dtype=torch.float64)
    com[:, 0] = x
    return js.TwinState(com, torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, dtype=torch.float64),
                        torch.zeros(n, 3, dtype=torch.float64), torch.zeros(n, 3, dtype=torch.float64),
                        torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64))


class G11StationKeeping(unittest.TestCase):
    def test_entry_checks_raise_the_reference_errors_in_order(self):
        n = 3
        for t_value, done_row in ((0.1, None), (0.0, None), (math.nan, None), (9.0, None), (math.nan, 0),
                                  (9.0, 1), ([0.1, -1.0, 9.0], None), ([0.1, -1.0, 9.0], 1)):
            tw = twin(n)
            if done_row is not None:
                tw.done[done_row] = True
            t = torch.as_tensor(t_value, dtype=torch.float64).reshape(-1).expand(n)
            expected = got = None
            try:
                REFERENCE_entry_checks(~tw.done, tw.prev_time.clone(), tw.deadline.clone(), t)
            except ValueError as exc:
                expected = str(exc)
            try:
                tw.step(t_value, state(n))
            except ValueError as exc:
                got = str(exc)
            self.assertEqual(got, expected, (t_value, done_row))

    def test_last_any_finished_tracks_the_codes(self):
        n = 3
        tw = twin(n)
        self.assertTrue(tw.last_any_finished)          # unknown before the first step
        out = tw.step(1.0 / 60.0, state(n))
        self.assertEqual(tw.last_any_finished, bool((out != 0).any()))
        oob = torch.tensor([False, True, False])
        out = tw.step(2.0 / 60.0, state(n), out_of_bounds=oob)
        self.assertTrue(tw.last_any_finished)
        self.assertEqual(tw.last_any_finished, bool((out != 0).any()))

    def test_publish_equals_reference(self):
        F.install_isaaclab_stubs()
        publish = F.env_module().StationKeepingEnv._journal_publish_d7
        rng = random.Random(12)
        for case in range(200):
            n = rng.randint(1, 5)

            def make():
                g = torch.Generator().manual_seed(case)
                return NS(num_envs=n, device=torch.device("cpu"), scene=NS(env_origins=torch.randn(n, 3, generator=g) * 9),
                          _d7_centers_w=torch.randn(n, 1, 2, generator=g), _d7_radii=torch.rand(n, 1, generator=g),
                          _d7_physical=torch.rand(n, 1, generator=g) < 0.5, _d7_visible=torch.rand(n, 1, generator=g) < 0.5)
            a, b = make(), make()
            keys = list(range(n)) if rng.random() < 0.6 else rng.sample(range(n), rng.randint(0, n))
            rng.shuffle(keys)
            samples = {i: ([] if rng.random() < 0.6 else
                           [NS(slot=0, x=rng.uniform(-20, 20), y=rng.uniform(-20, 20), radius_m=rng.uniform(0.1, 2),
                               physical=rng.random() < 0.7, visible=rng.random() < 0.7)]) for i in keys}
            publish(a, samples)
            REFERENCE_publish(b, samples)
            for name in ("_d7_centers_w", "_d7_radii", "_d7_physical", "_d7_visible"):
                self.assertEqual(bits(getattr(a, name)), bits(getattr(b, name)), (case, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
