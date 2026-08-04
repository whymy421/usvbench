"""Owner's porous-double-ring design, computed instead of guessed.

His spec: goal at center; TWO concentric rings of cylinders; the gap between
every pair of neighbouring cylinders is 3-5 boat-widths (every gap passable --
no searching, no fighting the potential); ring separation 3-5 boat lengths so
the hull can realign between crossings; rings biggish. Requirement to verify:
from ANY approach bearing the straight greedy run at the goal is blocked by
BOTH rings, so the boat must weave twice -- pure avoidance, zero search.

This script Monte-Carlos that property over the parameter grid and reports:
  P(block outer): straight spawn->goal segment hits >=1 outer cylinder
  P(block inner): same for the inner ring
  P(two-weave)  : both blocked  (want ~100%)
  LOS blocked   : goal invisible from spawn along the straight line (flavour)
Boat is treated as a disc of half-beam 0.45 m (inflate cylinders by it), the
same convention as every shipped admission audit.
"""
import math

import numpy as np

HALF_BEAM = 0.45
BEAM = 0.9
LOA = 1.2
R_OBS = (0.8, 2.0)          # existing cylinder radius convention
INNER_R = 9.0
SPAWN_LO, SPAWN_HI = 18.0, 24.0
N_TRIALS = 20000


def build_ring(radius_m, gap_m, rng, phase=0.0):
    """Cylinders on a circle with ~gap_m clear water between neighbours."""
    r_mean = 0.5 * (R_OBS[0] + R_OBS[1])
    period = 2.0 * r_mean + gap_m
    count = max(6, int(round((2.0 * math.pi * radius_m) / period)))
    angles = phase + np.arange(count) * (2.0 * math.pi / count)
    radii = rng.uniform(R_OBS[0], R_OBS[1], size=count)
    centers = radius_m * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return centers, radii, count


def seg_hits_any(p0, p1, centers, radii_infl):
    d = p1 - p0
    L2 = float(d @ d)
    t = np.clip(((centers - p0) @ d) / L2, 0.0, 1.0)
    closest = p0 + t[:, None] * d
    dist = np.linalg.norm(centers - closest, axis=1)
    return bool(np.any(dist < radii_infl))


def build_grid_ring(radius_m, count, rng, phase):
    """count cylinders on a fixed angular grid (radii still random)."""
    angles = phase + np.arange(count) * (2.0 * math.pi / count)
    radii = rng.uniform(R_OBS[0], R_OBS[1], size=count)
    centers = radius_m * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return centers, radii, count


def evaluate(gap_m, sep_m, mode, rng):
    outer_r = INNER_R + sep_m
    blocked_o = blocked_i = blocked_both = los = 0
    for _ in range(N_TRIALS):
        phase_in = rng.uniform(0.0, 2.0 * math.pi)
        if mode == "grid":
            # Same count on BOTH rings, outer offset by half a period: a
            # radial ray through an outer gap meets an inner cylinder at the
            # same bearing. Rays at the goal are radial (goal IS the center),
            # so the greedy straight line is blocked by construction.
            r_mean = 0.5 * (R_OBS[0] + R_OBS[1])
            count = max(6, int(round((2.0 * math.pi * INNER_R)
                                     / (2.0 * r_mean + gap_m))))
            c_in, r_in, n_in = build_grid_ring(INNER_R, count, rng, phase_in)
            c_out, r_out, n_out = build_grid_ring(
                outer_r, count, rng, phase_in + math.pi / count
            )
        elif mode == "band":
            # A "band" = two sub-rings 2.5 m apart, same count, half-period
            # staggered (brick wall). Every gap is passable via an S-turn,
            # yet no radial line clears the band: the sub-rings' angular
            # coverages interleave to 100%. Two bands = two forced weaves.
            r_mean = 0.5 * (R_OBS[0] + R_OBS[1])
            count = max(6, int(round((2.0 * math.pi * INNER_R)
                                     / (2.0 * r_mean + gap_m))))
            a1, b1, _ = build_grid_ring(INNER_R, count, rng, phase_in)
            a2, b2, _ = build_grid_ring(
                INNER_R + 2.5, count, rng, phase_in + math.pi / count
            )
            c_in, r_in = np.vstack([a1, a2]), np.concatenate([b1, b2])
            n_in = 2 * count
            band2_r = INNER_R + 2.5 + sep_m
            phase_o = rng.uniform(0.0, 2.0 * math.pi)
            count_o = max(6, int(round((2.0 * math.pi * band2_r)
                                       / (2.0 * r_mean + gap_m))))
            a3, b3, _ = build_grid_ring(band2_r, count_o, rng, phase_o)
            a4, b4, _ = build_grid_ring(
                band2_r + 2.5, count_o, rng, phase_o + math.pi / count_o
            )
            c_out, r_out = np.vstack([a3, a4]), np.concatenate([b3, b4])
            n_out = 2 * count_o
            outer_r = band2_r + 2.5
        else:
            c_in, r_in, n_in = build_ring(INNER_R, gap_m, rng, phase_in)
            phase_out = rng.uniform(0.0, 2.0 * math.pi)
            c_out, r_out, n_out = build_ring(outer_r, gap_m, rng, phase_out)
        spawn_r = rng.uniform(SPAWN_LO, SPAWN_HI)
        b = rng.uniform(0.0, 2.0 * math.pi)
        p0 = spawn_r * np.array([math.cos(b), math.sin(b)])
        p1 = np.zeros(2)
        hit_o = seg_hits_any(p0, p1, c_out, r_out + HALF_BEAM)
        hit_i = seg_hits_any(p0, p1, c_in, r_in + HALF_BEAM)
        blocked_o += hit_o
        blocked_i += hit_i
        blocked_both += hit_o and hit_i
        los += seg_hits_any(p0, p1, np.vstack([c_out, c_in]),
                            np.concatenate([r_out, r_in]))  # sight, no beam
    n = float(N_TRIALS)
    return (blocked_o / n, blocked_i / n, blocked_both / n, los / n,
            n_in, n_out, outer_r)


rng = np.random.default_rng(11)
print("多孔双环设计计算 | 内环半径 9 m | 出生带 18-24 m | 柱半径 0.8-2.0 m "
      f"| 船宽 {BEAM} m 船长 {LOA} m | 每格 {N_TRIALS} 次")
print("缝宽解释: 3-5x船宽 = 2.7/3.6/4.5 m ; 3-5x船长 = 3.6/4.8/6.0 m")
print("=" * 108)
print(f"{'缝宽':>6} {'环距':>6} {'错位':>6} {'柱数(内/外)':>12} "
      f"{'挡外环':>8} {'挡内环':>8} {'两次必避':>9} {'终点不可见':>10}")
print("-" * 108)
for gap in (2.7, 3.6, 4.5, 6.0):
    for sep in (3.6, 4.8, 6.0):
        for mode in ("band", "grid", "rand"):
            po, pi_, pb, pl, n_in, n_out, outer_r = evaluate(gap, sep, mode, rng)
            r_mean = 0.5 * (R_OBS[0] + R_OBS[1])
            gap_out = (2.0 * math.pi * outer_r / n_out) - 2.0 * r_mean
            label = {"band": "砖墙", "grid": "网格", "rand": "随机"}[mode]
            print(f"{gap:>6.1f} {sep:>6.1f} {label:>6} "
                  f"{n_in:>5d}/{n_out:<6d} "
                  f"{po:>7.1%} {pi_:>8.1%} {pb:>8.1%} {pl:>9.1%}"
                  f"   外环实缝 {gap_out:.1f} m")
print()
print("判读标准: '两次必避' 越接近 100% 越纯粹 -- 直线贪心必然失败, 但每个缝都能过,")
print("不需要绕环搜索, 不违背势函数; '终点不可见' 高说明穿进去前看不到终点(owner 的要求)。")
