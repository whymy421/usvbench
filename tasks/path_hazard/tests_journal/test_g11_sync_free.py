"""G11: the path-hazard journal step without its avoidable host syncs equals T-base v2.3 bit for bit.

REFERENCE_* below are the T-base v2.3 sources (verbatim) of the changed code:
* ThreadingTensorTwin.update (boolean-mask writes) vs the where-based write: 300 random portal traces,
  every return value and every table identical (bits);
* validity_codes with a Python-number nominal / water (torch.full) vs the as_tensor path;
* OrderedPathTensorOracle.step: the three entry checks (one transfer) raise exactly the reference
  errors in the reference order, malformed time_s included; last_code_list == code.tolist();
* PathHazardEnv._journal_publish_d7: fast all-rows clear + one host-to-device copy vs the reference
  per-field copies on random sample sets (empty rows, intruder rows, subsets of rows).
Run: python -B tasks/path_hazard/tests_journal/test_g11_sync_free.py   (CPU only)
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

import ph_fixtures as F  # noqa: E402

M = F.modules()
J, C = M.journal, M.contracts
LIMITS = C.ValidityLimits(**M.plans.plan_class("path_hazard", "blueboat_v2")(2).validity_limits)


def bits(t):
    t = t.detach().contiguous()
    return (str(t.dtype), tuple(t.shape), t.cpu().numpy().tobytes())


class REFERENCE_Threading(J.ThreadingTensorTwin):
    def update(self, x, y, hx, hy, collision, rows=None):
        live = self.current < self.g
        if rows is not None:
            live = live & torch.as_tensor(rows, dtype=torch.bool, device=self.device)
        rows = torch.arange(self.n, device=self.device)
        k = self.current.clamp(max=self.g - 1)
        x, y, hx, hy = (torch.as_tensor(v, dtype=torch.float64, device=self.device) for v in (x, y, hx, hy))
        if not bool(((torch.isfinite(hx) & (hx > 0) & torch.isfinite(hy) & (hy > 0)) | ~live).all()):
            raise ValueError("hull extents must be positive")
        collision = torch.as_tensor(collision, dtype=torch.bool, device=self.device).expand(self.n)
        g = lambda t: t[rows, k]  # noqa: E731
        active, failed, paid, worst = g(self.active), g(self.failed), g(self.paid), g(self.worst)
        prev, prev_y, prev_hx, prev_hy = g(self.prev_x), g(self.prev_y), g(self.prev_hx), g(self.prev_hy)
        entry, exit_, half = g(self.entry), g(self.exit), g(self.half_width)
        span_hx = torch.maximum(hx, torch.where(torch.isnan(prev_hx), hx, prev_hx))
        span_hy = torch.maximum(hy, torch.where(torch.isnan(prev_hy), hy, prev_hy))
        has_prev = ~torch.isnan(prev)
        enter = live & ~active & has_prev & (prev + span_hx < entry) & (entry <= x + span_hx)
        safe_den = torch.where(enter, x - prev, torch.ones_like(x))
        fraction = ((entry - prev) - span_hx) / safe_den
        entry_y = prev_y + fraction * (y - prev_y)
        entry_clear = (half - entry_y.abs()) - span_hy
        worst = torch.where(enter & (entry_clear < worst), entry_clear, worst)
        failed = torch.where(enter, entry_clear <= 0.0, failed)
        active = active | enter
        act = live & active
        clear = (half - y.abs()) - span_hy
        worst = torch.where(act & (clear < worst), clear, worst)
        failed = torch.where(act, failed | collision | (clear <= 0.0) | (has_prev & (x < prev)), failed)
        pay = act & ~failed & ((x - hx) >= exit_)
        active = torch.where(act & (failed | pay), torch.zeros_like(active), active)
        paid = paid | pay
        for name, value in (("active", active), ("failed", failed), ("paid", paid), ("worst", worst),
                            ("prev_x", x), ("prev_y", y), ("prev_hx", hx), ("prev_hy", hy)):
            table = getattr(self, name)
            table[rows[live], k[live]] = value[live]
        self.current = torch.where(live & (paid | failed), self.current + 1, self.current)
        return torch.where(pay, torch.full_like(x, self.bonus), torch.zeros_like(x))


TABLES = ("entry", "exit", "half_width", "active", "paid", "failed", "worst", "prev_x", "prev_y", "prev_hx",
          "prev_hy", "current")


def REFERENCE_validity_nominal(com, nominal):
    return torch.as_tensor(nominal, dtype=torch.float64, device=com.device).expand(com.shape[0])


def REFERENCE_publish(env, samples):
    if not samples:
        return
    rows = torch.as_tensor(sorted(samples), dtype=torch.long, device=env.device)
    slots = torch.as_tensor(env._jph_d7_slots, dtype=torch.long, device=env.device)
    env._d7_physical[rows.unsqueeze(-1), slots.unsqueeze(0)] = False
    env._d7_visible[rows.unsqueeze(-1), slots.unsqueeze(0)] = False
    e_, slot, xy, radius, physical, visible = [], [], [], [], [], []
    for i in sorted(samples):
        for s in samples[i]:
            if int(s.slot) not in env._jph_d7_slots:
                raise ValueError(f"D7 sample slot {s.slot} outside the reserved D7 slots {env._jph_d7_slots}")
            e_.append(int(i))
            slot.append(int(s.slot))
            xy.append((float(s.x), float(s.y)))
            radius.append(float(s.radius_m))
            physical.append(bool(s.physical))
            visible.append(bool(s.visible))
    if not e_:
        return
    e = torch.as_tensor(e_, dtype=torch.long, device=env.device)
    k = torch.as_tensor(slot, dtype=torch.long, device=env.device)
    local = torch.as_tensor(xy, dtype=torch.float64, device=env.device)
    r = torch.as_tensor(radius, dtype=torch.float64, device=env.device)
    env._jph_centers[e, k] = local
    env._jph_radii[e, k] = r
    env.obstacle_centers[e, k] = (local + env.scene.env_origins[e, :2].to(torch.float64)).to(
        env.obstacle_centers.dtype)
    env.obstacle_radii[e, k] = r.to(env.obstacle_radii.dtype)
    env._d7_physical[e, k] = torch.as_tensor(physical, dtype=torch.bool, device=env.device)
    env._d7_visible[e, k] = torch.as_tensor(visible, dtype=torch.bool, device=env.device)


def fake_env(n, base, width, seed):
    g = torch.Generator().manual_seed(seed)
    return NS(num_envs=n, device=torch.device("cpu"), _jph_d7_slots=tuple(range(base, width)),
              _d7_physical=torch.rand(n, width, generator=g) < 0.5, _d7_visible=torch.rand(n, width, generator=g) < 0.5,
              _jph_centers=torch.randn(n, width, 2, generator=g, dtype=torch.float64),
              _jph_radii=torch.rand(n, width, generator=g, dtype=torch.float64),
              obstacle_centers=torch.randn(n, width, 2, generator=g), obstacle_radii=torch.rand(n, width, generator=g),
              scene=NS(env_origins=torch.randn(n, 3, generator=g) * 50.0))


ENV_FIELDS = ("_d7_physical", "_d7_visible", "_jph_centers", "_jph_radii", "obstacle_centers", "obstacle_radii")


class G11SyncFree(unittest.TestCase):
    def test_threading_update_equals_reference(self):
        rng = random.Random(3)
        for trace in range(300):
            n, g = rng.randint(1, 6), rng.randint(1, 4)
            new, ref = J.ThreadingTensorTwin(n, g, "cpu", 1.5), REFERENCE_Threading(n, g, "cpu", 1.5)
            radius = torch.tensor([rng.uniform(0.5, 3.0) for _ in range(n)], dtype=torch.float64)
            for twin in (new, ref):
                twin.reset(torch.arange(n), radius)
            for step in range(rng.randint(5, 40)):
                x = torch.tensor([rng.uniform(-6, 6) for _ in range(n)], dtype=torch.float64)
                y = torch.tensor([rng.uniform(-4, 4) for _ in range(n)], dtype=torch.float64)
                hx = torch.tensor([rng.uniform(0.2, 2.0) for _ in range(n)], dtype=torch.float64)
                hy = torch.tensor([rng.uniform(0.2, 1.0) for _ in range(n)], dtype=torch.float64)
                collision = torch.tensor([rng.random() < 0.05 for _ in range(n)])
                rows = torch.tensor([rng.random() < 0.9 for _ in range(n)]) if rng.random() < 0.5 else None
                if rng.random() < 0.05:
                    x[rng.randrange(n)] = math.nan
                a = new.update(x, y, hx, hy, collision, rows=rows)
                b = ref.update(x, y, hx, hy, collision, rows=rows)
                self.assertEqual(bits(a), bits(b))
                for name in TABLES:
                    self.assertEqual(bits(getattr(new, name)), bits(getattr(ref, name)), (trace, step, name))
                if rng.random() < 0.1:
                    ids = torch.tensor([i for i in range(n) if rng.random() < 0.5], dtype=torch.long)
                    for twin in (new, ref):
                        twin.reset(ids, radius[ids])

    def test_exact_hypot_many_equals_exact_hypot_and_math_hypot(self):
        rng = random.Random(17)
        special = [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1e-310, 1e300, 1.7976931348623157e308,
                   math.inf, -math.inf, math.nan, 3.0, 4.0]
        for trial in range(60):
            pairs = []
            for _ in range(rng.randint(1, 4)):
                n = rng.randint(1, 25)
                mk = lambda: torch.tensor([rng.choice(special) if rng.random() < 0.3 else
                                           rng.uniform(-1, 1) * 10 ** rng.uniform(-320, 300) for _ in range(n)],
                                          dtype=torch.float64)
                pairs.append((mk(), mk()))
            for (x, y), got in zip(pairs, J.exact_hypot_many(pairs)):
                self.assertEqual(bits(got), bits(J.exact_hypot(x, y)))
                ref = torch.tensor([math.hypot(a, b) for a, b in zip(x.tolist(), y.tolist())], dtype=torch.float64)
                self.assertEqual(bits(got), bits(ref))

    def test_segment_and_cross_track_wrappers_equal_the_reference_formula(self):
        g = torch.Generator().manual_seed(23)
        for _ in range(30):
            v = [torch.randn(9, generator=g, dtype=torch.float64) * 10 for _ in range(7)]
            radius = torch.rand(9, generator=g, dtype=torch.float64) * 5
            p0x, p0y, p1x, p1y, cx, cy = v[:6]
            ax, ay = p0x - cx, p0y - cy
            dx, dy = p1x - p0x, p1y - p0y
            n = dx * dx + dy * dy
            raw = -(ax * dx + ay * dy) / torch.where(n == 0.0, torch.ones_like(n), n)
            t = torch.where(n == 0.0, torch.zeros_like(raw), J._py_max0_min1(raw))
            want = J.exact_hypot(ax + t * dx, ay + t * dy) <= radius
            self.assertEqual(bits(J.segment_hits_disk(p0x, p0y, p1x, p1y, cx, cy, radius)), bits(want))
            px, py, ax2, ay2, bx, by = v[:6]
            dx, dy = bx - ax2, by - ay2
            n = dx * dx + dy * dy
            raw = ((px - ax2) * dx + (py - ay2) * dy) / torch.where(n != 0.0, n, torch.ones_like(n))
            t = torch.where(n != 0.0, J._py_max0_min1(raw), torch.zeros_like(raw))
            want = J.exact_hypot(px - (ax2 + t * dx), py - (ay2 + t * dy))
            self.assertEqual(bits(J.cross_track_error(px, py, ax2, ay2, bx, by)), bits(want))

    def test_threading_update_still_validates_extents(self):
        twin = J.ThreadingTensorTwin(2, 2, "cpu", 1.0)
        twin.reset(torch.arange(2), torch.ones(2, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "hull extents must be positive"):
            twin.update(torch.zeros(2), torch.zeros(2), torch.tensor([1.0, -1.0]), torch.ones(2), torch.zeros(2, dtype=torch.bool))

    def test_number_nominal_and_water_equal_the_as_tensor_path(self):
        g = torch.Generator().manual_seed(5)
        for n in (1, 3, 8):
            com = torch.randn(n, 3, generator=g, dtype=torch.float64)
            quat = torch.nn.functional.normalize(torch.randn(n, 4, generator=g, dtype=torch.float64), dim=-1)
            lin, ang = torch.randn(n, 3, generator=g), torch.randn(n, 3, generator=g)
            for nominal in (0.123456789, 1, -0.0, math.inf, math.nan, 3.0e-310):
                water = torch.randn(n, generator=g, dtype=torch.float64)
                tensor_nominal = REFERENCE_validity_nominal(com, nominal)
                self.assertEqual(bits(J._rows_f64(nominal, com)), bits(tensor_nominal.contiguous()))
                a = J.validity_codes(com, quat, lin, ang, water, nominal, LIMITS)
                b = J.validity_codes(com, quat, lin, ang, water, tensor_nominal, LIMITS)
                c = J.validity_codes(com, quat, lin, ang, float(water[0]), nominal, LIMITS)
                d = J.validity_codes(com, quat, lin, ang, torch.as_tensor(float(water[0]), dtype=torch.float64),
                                     tensor_nominal, LIMITS)
                for x_, y_ in ((a, b), (c, d)):
                    self.assertEqual(bits(x_[0]), bits(y_[0]))
                    self.assertEqual(bits(x_[1]), bits(y_[1]))

    def _oracle(self, n=3, g=2):
        oracle = J.OrderedPathTensorOracle(n, g, "cpu", LIMITS)
        route = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]] * n, dtype=torch.float64)
        oracle.reset(torch.arange(n), route, 1.0, torch.full((n,), 5.0, dtype=torch.float64), route[:, 0])
        return oracle

    def _state(self, n=3):
        com = torch.zeros(n, 3, dtype=torch.float64)
        com[:, 2] = M.plans.vessel_record("blueboat_v2").com_height_above_waterline_m
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, dtype=torch.float64)
        return com, quat, torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n, dtype=torch.float64), float(com[0, 2])

    def test_step_entry_checks_raise_the_reference_errors_in_order(self):
        com, quat, lin, ang, water, nominal = self._state()
        cases = [
            ("done", lambda o: o.done.__setitem__(1, True), 0.1, RuntimeError, "reset required"),
            ("done+bad time", lambda o: o.done.__setitem__(1, True), "x", RuntimeError, "reset required"),
            ("bad time", None, "x", (TypeError, ValueError, RuntimeError), None),
            ("not increasing", None, 0.0, ValueError, "sample clock must increase"),
            ("nan time", None, math.nan, ValueError, "sample clock must increase"),
            ("past deadline", None, 6.0, ValueError, "sample exactly at the deadline"),
            ("done beats clock", lambda o: o.done.__setitem__(0, True), -1.0, RuntimeError, "reset required"),
        ]
        for label, prepare, time_s, error, message in cases:
            oracle = self._oracle()
            if prepare:
                prepare(oracle)
            with self.assertRaises(error, msg=label) as ctx:
                oracle.step(time_s, com, quat, lin, ang, water, nominal)
            if message:
                self.assertIn(message, str(ctx.exception), label)

    def test_last_code_list_is_the_step_codes(self):
        com, quat, lin, ang, water, nominal = self._state()
        oracle = self._oracle()
        oob = torch.tensor([False, True, False])
        code = oracle.step(1.0 / 60.0, com, quat, lin, ang, water, nominal, out_of_bounds=oob)
        self.assertEqual(oracle.last_code_list, code.tolist())
        self.assertEqual(oracle.last_code_list[1], J.OUT_OF_BOUNDS)
        self.assertIsNotNone(oracle.result(1))

    def test_publish_equals_reference(self):
        env_module = F.install_stubs()["env"]
        publish = env_module.PathHazardEnv._journal_publish_d7
        rng = random.Random(9)
        for case in range(200):
            n, base = rng.randint(1, 5), rng.randint(0, 6)
            width = base + rng.randint(1, 2)
            a, b = fake_env(n, base, width, case), fake_env(n, base, width, case)
            keys = list(range(n)) if rng.random() < 0.7 else sorted(rng.sample(range(n), rng.randint(0, n)))
            samples = {}
            for i in keys:
                items = []
                for _ in range(rng.choice((0, 0, 1, 2))):
                    items.append(NS(slot=rng.randrange(base, width), x=rng.uniform(-40, 40), y=rng.uniform(-40, 40),
                                    radius_m=rng.uniform(0.1, 3.0), physical=rng.random() < 0.7,
                                    visible=rng.random() < 0.7))
                samples[i] = items
            publish(a, samples)
            REFERENCE_publish(b, samples)
            for name in ENV_FIELDS:
                self.assertEqual(bits(getattr(a, name)), bits(getattr(b, name)), (case, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
