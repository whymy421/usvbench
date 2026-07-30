"""Does the tier ladder actually control difficulty, or can the boat go around?

A seed-43 baseline certified 128/128 at tier 1 by taking 63% longer paths and
never coming within a metre of anything -- i.e. it circumvented the obstacle
field rather than threading it. If circumvention stays cheap at every tier,
then the tier parameter (gap width in hull beams) does not control difficulty
at all, and every certified Task A number is really measuring willingness to
detour.

This samples real layouts from the shipped generator and measures, per tier:
  * how far out the field extends laterally
  * the length of a detour that clears the whole field
  * the detour ratio against the straight-line distance

A tier only "bites" if threading is meaningfully cheaper than going around.
"""

from __future__ import annotations

import numpy as np

try:
    from .hazard_geometry import difficulty_for_level, sample_layout
except ImportError:  # direct execution
    from hazard_geometry import difficulty_for_level, sample_layout

HULL_BEAM_M = 0.899
HALF_BEAM_M = 0.45
N_LAYOUTS = 40


def detour_length(start, goal, centers, radii, clearance=HALF_BEAM_M):
    """Length of a two-leg path that clears every obstacle laterally.

    Straight out to one side until past the widest obstacle, along, then in.
    This is an upper bound on the true circumventing path, so if even THIS is
    cheap the field can be avoided.
    """
    axis = goal - start
    length = np.linalg.norm(axis)
    axis = axis / max(length, 1e-9)
    normal = np.array([-axis[1], axis[0]])

    lateral = np.array([np.dot(c - start, normal) for c in centers])
    reach = lateral + radii + clearance
    reach_neg = -(lateral - radii - clearance)
    side = min(reach.max(), reach_neg.max())  # cheaper side
    # out, along, back in
    return 2.0 * side + length, side


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'档位':<6}{'缝宽(船宽)':>12}{'直线(m)':>10}{'绕行(m)':>10}"
          f"{'绕行倍数':>10}{'需横移(m)':>11}")
    print("-" * 62)
    rows = []
    for level in (0, 1, 2, 3):
        diff = difficulty_for_level(level)
        straights, detours, sides = [], [], []
        for _ in range(N_LAYOUTS):
            lay = sample_layout(level, rng)
            start = np.asarray(lay.start, dtype=float)
            goal = np.asarray(lay.goal, dtype=float)
            centers = np.asarray(lay.centers, dtype=float)
            radii = np.asarray(lay.radii, dtype=float)
            d, side = detour_length(start, goal, centers, radii)
            straights.append(float(np.linalg.norm(goal - start)))
            detours.append(d)
            sides.append(side)
        s = float(np.median(straights))
        de = float(np.median(detours))
        rows.append((level, s, de, de / s, float(np.median(sides))))
        print(f"{level:<6}{diff.bottleneck_m / HULL_BEAM_M:>12.1f}"
              f"{s:>10.1f}{de:>10.1f}{de / s:>10.2f}{np.median(sides):>11.1f}")

    print()
    worst = max(r[3] for r in rows)
    best = min(r[3] for r in rows)
    print(f"绕行倍数在四个档位之间的变化: {best:.2f} -> {worst:.2f}")
    if worst < 2.0:
        print("=> 每个档位都能以不到两倍路程绕开整片障碍。")
        print("   缝宽档位控制的是'穿过去有多难',但穿过去从来不是必须的,")
        print("   所以档位阶梯并没有在控制任务难度。")
    else:
        print("=> 绕行代价随档位显著上升,档位阶梯有效。")
    # --- the ring-siege variant, for contrast ----------------------------
    # The owner proposed sealing the spawn inside a ring with exactly one
    # tier-width gap. If that design holds, no detour exists at all: the boat
    # starts inside, so any escape has to pass through the gap.
    try:
        from .hazard_geometry import sample_ring_layout
    except ImportError:
        from hazard_geometry import sample_ring_layout

    print()
    print("对照: 环形围困(出生点封在环内,唯一出口是缝)")
    for level in (0, 3):
        rng2 = np.random.default_rng(7)
        sealed = 0
        trials = 20
        for _ in range(trials):
            lay = sample_ring_layout(level, rng2)
            centers = np.asarray(lay.centers, dtype=float)
            radii = np.asarray(lay.radii, dtype=float)
            inflated = radii + HALF_BEAM_M
            # Only ADJACENT cylinders can form a passage. Ordering them by
            # angle around the ring centre gives the neighbour pairs; comparing
            # every pair would count opposite sides of the ring as a huge
            # "opening", which is what my first version got wrong.
            hub = centers.mean(axis=0)
            order = np.argsort(np.arctan2(centers[:, 1] - hub[1],
                                          centers[:, 0] - hub[0]))
            c, r = centers[order], inflated[order]
            nxt = np.roll(np.arange(len(c)), -1)
            span = np.linalg.norm(c - c[nxt], axis=-1) - r - r[nxt]
            passable = int((span > 0.0).sum())
            if passable <= 1:
                sealed += 1
        print(f"  档位 {level}: {trials} 张布局里 {sealed} 张只有一个出口")

    # --- cross-check against a real certified policy ----------------------
    # The seed-43 baseline that scored 128/128 at tier 1 travelled a median of
    # 53.6 m. If the geometry above is right, that is the detour route, not a
    # threading route -- and the two numbers should agree.
    import json
    import os
    cert = os.path.join(os.path.dirname(__file__), "..", "..",
                        "queue12", "cert_base_s43_e42.json")
    if os.path.isfile(cert):
        recs = json.load(open(cert, encoding="utf-8"))["records"]
        measured = float(np.median([r["path_length_m"] for r in recs if r["success"]]))
        predicted = rows[0][2]
        print()
        print(f"交叉验证(一档): 几何预测绕行 {predicted:.1f} m | "
              f"100% 冠军实测路径 {measured:.1f} m | 直线 {rows[0][1]:.1f} m")
        if abs(measured - predicted) < 0.35 * predicted:
            print("  => 实测路径落在绕行预测附近,该策略走的是绕行路线,不是穿缝路线。")

    print()
    print("注: 这里的绕行路径是上界(直出-平移-直入三段),真实绕行只会更短。")


if __name__ == "__main__":
    main()
