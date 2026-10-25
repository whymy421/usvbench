"""G11: the hazard-navigation journal step without its avoidable host syncs equals T-base v2.3 bit for bit.

* py_hypot_many == [py_hypot(x, y) ...] == math.hypot bitwise (zeros, subnormals, infinities, NaN,
  huge / tiny magnitudes; pairs with and without subnormal rows in one call);
* ReachEpisodeTensor.step: the three entry checks (one transfer) raise the T-base errors in the
  T-base order (REFERENCE_entry_checks below, verbatim), malformed time_s included;
* swept_polygon_clearance(core_radius=footprint_core_radius(v)) == the T-base call (bits);
* HazardNavEnv._journal_core_radius: computed once, recomputed after an in-place write or a new tensor;
* HazardNavEnv._journal_publish_d7: one host-to-device copy == the T-base per-field copies
  (REFERENCE_publish below, verbatim) on random sample sets.
Run: python -B tasks/hazard_nav/test_g11_sync_free.py   (CPU only)
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import random
import struct
import sys
from types import ModuleType
from types import SimpleNamespace as NS
import unittest

sys.dont_write_bytecode = True
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ROOT = "_g11_hn"
if ROOT not in sys.modules:
    _root = ModuleType(ROOT)
    _root.__path__ = [str(REPO / "tasks")]
    sys.modules[ROOT] = _root
    for _folder in ("_shared", "hazard_nav"):
        _sub = ModuleType(ROOT + "." + _folder)
        _sub.__path__ = [str(REPO / "tasks" / _folder)]
        sys.modules[ROOT + "." + _folder] = _sub
J = importlib.import_module(ROOT + ".hazard_nav.journal_hazard_nav")
C = importlib.import_module(ROOT + "._shared.journal_contracts_v2")


def bits(t):
    t = torch.as_tensor(t).detach().contiguous()
    return (str(t.dtype), tuple(t.shape), t.cpu().numpy().tobytes())


SPECIAL = [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1e-310, 1e300, -1e300, 1.7976931348623157e308,
           math.inf, -math.inf, math.nan, 1.0, 3.0, 4.0, 1e-200]


def REFERENCE_entry_checks(done, prev_time, deadline, time_s, dtype=torch.float64):
    if bool(done.any()):
        raise RuntimeError("reset required after completion")
    t = torch.as_tensor(time_s, dtype=dtype).expand(done.shape[0])
    if not bool((torch.isfinite(t) & (t > prev_time)).all()):
        raise ValueError("sample clock must increase from the reset time zero")
    if bool((t > deadline + 1.0e-9).any()):
        raise ValueError("the adapter must include a sample exactly at the deadline")


def REFERENCE_publish(env, samples):
    if not samples:
        return
    rows, slots, xs, ys, rs, physical, visible = [], [], [], [], [], [], []
    width = int(env.obstacle_centers.shape[1])
    for env_index, items in samples.items():
        for item in items:
            if not 0 <= int(item.slot) < width:
                raise ValueError(f"D7 slot {item.slot} outside the {width}-slot storage")
            rows.append(int(env_index))
            slots.append(int(item.slot))
            xs.append(float(item.x))
            ys.append(float(item.y))
            rs.append(float(item.radius_m))
            physical.append(bool(item.physical))
            visible.append(bool(item.visible))
    if not rows:
        return
    row_t = torch.as_tensor(rows, dtype=torch.long, device=env.device)
    slot_t = torch.as_tensor(slots, dtype=torch.long, device=env.device)
    origin = env.scene.env_origins[row_t, :2]
    dtype = env.obstacle_centers.dtype
    env.obstacle_centers[row_t, slot_t, 0] = torch.as_tensor(xs, dtype=dtype, device=env.device) + origin[:, 0]
    env.obstacle_centers[row_t, slot_t, 1] = torch.as_tensor(ys, dtype=dtype, device=env.device) + origin[:, 1]
    env.obstacle_radii[row_t, slot_t] = torch.as_tensor(rs, dtype=env.obstacle_radii.dtype, device=env.device)
    env._d7_physical[row_t, slot_t] = torch.as_tensor(physical, dtype=torch.bool, device=env.device)
    env._d7_visible[row_t, slot_t] = torch.as_tensor(visible, dtype=torch.bool, device=env.device)


class G11HazardNav(unittest.TestCase):
    def test_py_hypot_many_equals_py_hypot_and_math_hypot(self):
        rng = random.Random(4)
        for trial in range(60):
            pairs = []
            for _ in range(rng.randint(1, 4)):
                n = rng.randint(1, 30)
                xs = [rng.choice(SPECIAL) if rng.random() < 0.3 else rng.uniform(-1, 1) * 10 ** rng.uniform(-320, 300)
                      for _ in range(n)]
                ys = [rng.choice(SPECIAL) if rng.random() < 0.3 else rng.uniform(-1, 1) * 10 ** rng.uniform(-320, 300)
                      for _ in range(n)]
                pairs.append((torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)))
            many = J.py_hypot_many(pairs)
            for (x, y), got in zip(pairs, many):
                self.assertEqual(bits(got), bits(J.py_hypot(x, y)))
                ref = torch.tensor([math.hypot(a, b) for a, b in zip(x.tolist(), y.tolist())], dtype=torch.float64)
                self.assertEqual(bits(got), bits(ref))

    def _twin(self, n=3):
        limits = C.ValidityLimits(capsize_angle_rad=1.0, min_heave_m=-1.0, max_heave_m=1.0)
        twin = J.ReachEpisodeTensor(n, "cpu", limits, 0.1)
        twin.reset(torch.arange(n), torch.zeros(n, 2), torch.full((n, 2), 30.0), torch.ones(n), torch.full((n,), 5.0))
        return twin

    def test_step_entry_checks_raise_the_reference_errors_in_order(self):
        n = 3
        com = torch.zeros(n, 3)
        com[:, 2] = 0.1
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
        lin, ang, water = torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n)
        for done_row, time_s in ((None, 0.1), (1, 0.1), (1, "x"), (None, "x"), (None, 0.0), (None, math.nan),
                                 (None, 6.0), (0, -1.0), (None, [0.1, 0.0, 0.2]), (2, 9.0)):
            twin = self._twin(n)
            if done_row is not None:
                twin.done[done_row] = True
            expected = None
            try:
                REFERENCE_entry_checks(twin.done.clone(), twin.prev_time.clone(), twin.deadline.clone(), time_s)
            except Exception as exc:  # noqa: BLE001
                expected = (type(exc), str(exc))
            got = None
            try:
                twin.step(time_s, com, quat, lin, ang, water)
            except Exception as exc:  # noqa: BLE001
                got = (type(exc), str(exc))
            if expected is None:
                self.assertIsNone(got, (done_row, time_s))
            else:
                self.assertEqual(got, expected, (done_row, time_s))

    def test_swept_clearance_with_cached_core_radius(self):
        g = torch.Generator().manual_seed(2)
        vertices = J.footprint_vertices("blueboat_v2")
        for _ in range(20):
            n, m = 4, 6
            prev = torch.randn(n, 3, generator=g, dtype=torch.float64) * 5
            pose = prev + torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.05
            centers = torch.randn(n, m, 2, generator=g) * 6
            radii = torch.rand(n, m, generator=g) * 2
            mask = torch.rand(n, m, generator=g) < 0.6
            a = J.swept_polygon_clearance(prev, pose, vertices, centers, radii, mask)
            b = J.swept_polygon_clearance(prev, pose, vertices, centers, radii, mask,
                                          core_radius=J.footprint_core_radius(vertices))
            self.assertEqual(bits(a), bits(b))

    def test_env_core_radius_cache_and_publish(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import journal_test_support as S
        S.install_isaaclab_stubs()
        fam = S.load_family()
        env_cls = fam.env.HazardNavEnv
        holder = NS(_journal_vertices=J.footprint_vertices("blueboat_v2"))
        reads = []
        original = fam.journal.footprint_core_radius

        def counting(v):
            reads.append(1)
            return original(v)
        fam.journal.footprint_core_radius = counting
        try:
            r1 = env_cls._journal_core_radius(holder)
            r2 = env_cls._journal_core_radius(holder)
            self.assertEqual(len(reads), 1)
            self.assertEqual(r1, original(holder._journal_vertices))
            self.assertEqual(r1, r2)
            holder._journal_vertices.mul_(2.0)
            self.assertEqual(env_cls._journal_core_radius(holder), original(holder._journal_vertices))
            holder._journal_vertices = J.footprint_vertices("wamv_vrx_v2_opt1")
            self.assertEqual(env_cls._journal_core_radius(holder), original(holder._journal_vertices))
            self.assertEqual(len(reads), 3)
        finally:
            fam.journal.footprint_core_radius = original
        rng = random.Random(6)
        for case in range(150):
            n, width = rng.randint(1, 5), rng.randint(1, 4)

            def make():
                g = torch.Generator().manual_seed(case)
                return NS(device=torch.device("cpu"), obstacle_centers=torch.randn(n, width, 2, generator=g),
                          obstacle_radii=torch.rand(n, width, generator=g),
                          _d7_physical=torch.rand(n, width, generator=g) < 0.5,
                          _d7_visible=torch.rand(n, width, generator=g) < 0.5,
                          scene=NS(env_origins=torch.randn(n, 3, generator=g) * 40))
            a, b = make(), make()
            samples = {}
            for i in rng.sample(range(n), rng.randint(0, n)):
                samples[i] = [NS(slot=rng.randrange(width), x=rng.uniform(-30, 30), y=rng.uniform(-30, 30),
                                 radius_m=rng.uniform(0.1, 3), physical=rng.random() < 0.6, visible=rng.random() < 0.6)
                              for _ in range(rng.choice((0, 1, 1, 2)))]
            env_cls._journal_publish_d7(a, samples)
            REFERENCE_publish(b, samples)
            for name in ("obstacle_centers", "obstacle_radii", "_d7_physical", "_d7_visible"):
                self.assertEqual(bits(getattr(a, name)), bits(getattr(b, name)), (case, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
