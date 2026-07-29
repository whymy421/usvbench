"""Acceptance tests for the one-shot half-sine threading bonus.

Two things must hold: the arc has the shape the owner specified (max on the
centreline, zero at contact, self-normalising across gap widths), and the gate
cannot be farmed by standing still, oscillating, crabbing, hugging one pillar,
or re-crossing the same gap.
"""

from __future__ import annotations

import math

import torch

try:
    from .threading_gate import ThreadingLatch, arc_payout, gap_geometry
except ImportError:  # direct execution
    from threading_gate import ThreadingLatch, arc_payout, gap_geometry

BEAM = 0.899
HALF_BEAM = 0.45
SPACING = math.radians(10.0)
MAX_R = 30.0
A = 5.0


def rays(d_left, d_right, front=MAX_R, n=36):
    """Build a 36-ray sweep with the given left/right/front readings."""
    x = torch.full((1, n), MAX_R)
    idx = torch.arange(n, dtype=torch.float32)
    ang = torch.rad2deg(idx * SPACING)
    ang = torch.where(ang > 180.0, ang - 360.0, ang)
    x[0, (ang >= 50) & (ang <= 130)] = d_left
    x[0, (ang >= -130) & (ang <= -50)] = d_right
    x[0, (ang >= -30) & (ang <= 30)] = front
    return x


def make_latch():
    return ThreadingLatch(1, torch.device("cpu"), amplitude=A)


def drive(latch, x, steps, surge=0.6, sway=0.0, yaw=0.0, dot=0.9, cross=0.1,
          contact=False):
    total = 0.0
    for _ in range(steps):
        r = latch.step(
            x, ray_spacing_rad=SPACING, half_beam_m=HALF_BEAM,
            surge_norm=torch.tensor([surge]), sway_norm=torch.tensor([sway]),
            yaw_rate_norm=torch.tensor([yaw]), goal_dot=torch.tensor([dot]),
            goal_cross=torch.tensor([cross]),
            contact=torch.tensor([contact]),
        )
        total += float(r[0])
    return total


def main() -> None:
    # --- 1. the arc has the specified shape --------------------------------
    assert abs(float(arc_payout(torch.tensor([0.0]), A)) - A) < 1e-6
    assert abs(float(arc_payout(torch.tensor([1.0]), A))) < 1e-6
    mid = float(arc_payout(torch.tensor([0.5]), A))
    assert 0 < mid < A and abs(mid - A * math.cos(math.pi / 4)) < 1e-6
    # monotone decreasing away from the centre
    us = torch.linspace(0, 1, 21)
    vals = arc_payout(us, A)
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))

    # --- 2. self-normalisation: same offset, different gaps ---------------
    # tier 1: gap 5 beams -> each side reads 2.25 m when centred
    u1, h1, _ = gap_geometry(rays(2.25 - 0.6, 2.25 + 0.6),
                             ray_spacing_rad=SPACING, half_beam_m=HALF_BEAM)
    # tier 4: gap 2 beams -> each side reads 0.90 m when centred
    u4, h4, _ = gap_geometry(rays(0.90 - 0.3, 0.90 + 0.3),
                             ray_spacing_rad=SPACING, half_beam_m=HALF_BEAM)
    r1 = float(arc_payout(u1, A)) / A
    r4 = float(arc_payout(u4, A)) / A
    assert r1 > r4, (r1, r4)
    assert abs(r1 - math.cos(math.pi / 6)) < 0.02, r1   # 0.6/1.80 -> 13% loss
    assert abs(r4 - math.cos(math.pi / 3)) < 0.02, r4   # 0.3/0.45 -> 50% loss

    # --- 3. hugging one pillar earns almost nothing -----------------------
    u_hug, _, _ = gap_geometry(rays(0.50, 2.50), ray_spacing_rad=SPACING,
                               half_beam_m=HALF_BEAM)
    assert float(arc_payout(u_hug, A)) / A < 0.15, float(arc_payout(u_hug, A))

    # --- 4. a clean centred passage pays exactly once ---------------------
    centred = rays(1.20, 1.20)
    latch = make_latch()
    paid = drive(latch, centred, 60)
    assert abs(paid - A) < 1e-4, paid          # full amplitude, once
    assert bool(latch.paid[0])
    assert drive(latch, centred, 60) == 0.0    # re-crossing pays nothing

    # --- 5. farming defences ---------------------------------------------
    # sitting still
    assert drive(make_latch(), centred, 200, surge=0.0) == 0.0
    # crabbing sideways
    assert drive(make_latch(), centred, 200, sway=0.9) == 0.0
    # spinning
    assert drive(make_latch(), centred, 200, yaw=0.9) == 0.0
    # driving away from the goal
    assert drive(make_latch(), centred, 200, dot=-0.9, cross=0.0) == 0.0
    # touching something during the passage
    assert drive(make_latch(), centred, 200, contact=True) == 0.0
    # oscillating in and out: never sustains the hold, so never completes
    latch = make_latch()
    total = 0.0
    for _ in range(40):
        total += drive(latch, centred, 5)                     # in, briefly
        total += drive(latch, rays(MAX_R, MAX_R), 5)          # out again
    assert total == 0.0, total
    # a dead end: no clearance ahead
    assert drive(make_latch(), rays(1.20, 1.20, front=0.6), 200) == 0.0
    # a slit narrower than the hull: h <= 0 forces u = 1 -> zero payout
    assert drive(make_latch(), rays(0.40, 0.40), 200) == 0.0

    print(f"一档偏 0.6 m: 拿到 {r1*100:.1f}% | 四档偏 0.3 m: 拿到 {r4*100:.1f}% "
          f"| 贴单柱: 拿到 {float(arc_payout(u_hug, A))/A*100:.1f}%")
    print("PASS: threading gate (arc shape, self-normalising, 8 farming defences)")


if __name__ == "__main__":
    main()
