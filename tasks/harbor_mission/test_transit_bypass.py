"""Does the harbor transit inherit the Task A detour hole?

The transit is pinned at both ends by 6 m gates (gate2 and field exit), but
there are no walls between them, so a lateral swing around the field is
geometrically possible. Measure its cost on real sampled routes: if the
detour costs much more than threading, the tiered field bites; if it is
cheap, Stage 2's numbers measure detour-willingness like old Task A did.
"""

from __future__ import annotations

import numpy as np

try:
    from .harbor_geometry import sample_harbor_route
except ImportError:  # direct execution
    from harbor_geometry import sample_harbor_route

HALF_BEAM_M = 0.45
N = 40


def main() -> None:
    rng = np.random.default_rng(5)
    ratios, sides = [], []
    for _ in range(N):
        r = sample_harbor_route(rng)
        a = np.asarray(r.field_start, float)
        b = np.asarray(r.field_exit, float)
        centers = np.asarray(r.obstacle_centers, float)
        radii = np.asarray(r.obstacle_radii, float)
        axis = b - a
        length = float(np.linalg.norm(axis))
        u = axis / max(length, 1e-9)
        n = np.array([-u[1], u[0]])
        lat = np.array([float(np.dot(c - a, n)) for c in centers])
        reach_pos = float((lat + radii + HALF_BEAM_M).max())
        reach_neg = float((-(lat - radii - HALF_BEAM_M)).max())
        side = min(reach_pos, reach_neg)
        detour = 2.0 * side + length
        ratios.append(detour / length)
        sides.append(side)

    med = float(np.median(ratios))
    print(f"{N} 条真实航线: 穿障段直线中位 = 门到门")
    print(f"  横向绕开整片障碍需外摆 中位 {np.median(sides):.1f} m, "
          f"绕行/直线 = 中位 {med:.2f}, p90 {np.percentile(ratios, 90):.2f}")
    if med < 2.0:
        print("=> 与旧题A 同款: 绕行代价不足两倍, 穿障段的档位不控制难度。")
        print("   题B 若保留, 穿障段需要围栏或把里程碑门收窄贴住障碍带。")
    else:
        print("=> 绕行代价显著, 双门钉位有效, 题B 穿障段合法。")


if __name__ == "__main__":
    main()
