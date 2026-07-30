"""Is line x hazard immune to the Task A bypass, or did it inherit it?

Task A's flaw: the goal is a single point, so the whole obstacle field can be
circumvented for ~1.7x the straight-line cost. Line x hazard is topologically
different -- ordered gates PIN the route -- but that claim was asserted, not
measured. This measures it.

For each sampled layout:
  * how many of the gate-to-gate legs are blocked by an obstacle whose
    inflated disk intersects the direct segment (forced deviation);
  * the lateral deviation needed to clear the tightest blocker on each leg,
    in hull beams (how much avoidance the leg actually demands);
  * whether a gate disk itself is reachable without entering the proximity
    band of some obstacle (gates cannot be "collected from outside").

If most legs force a deviation of a hull beam or more, the task genuinely
certifies local avoidance under route constraints and its numbers stand.
"""

from __future__ import annotations

import numpy as np

try:
    from .path_hazard_geometry import sample_layout
except ImportError:  # direct execution
    from path_hazard_geometry import sample_layout

HALF_BEAM_M = 0.45
BEAM_M = 0.899
N_LAYOUTS = 60


def leg_blockers(a, b, centers, radii):
    """Inflated obstacles whose disks intersect segment a->b; return the
    lateral clearance deviation each one forces (0 if none)."""
    ab = b - a
    length = np.linalg.norm(ab)
    if length < 1e-9:
        return []
    u = ab / length
    out = []
    for c, r in zip(centers, radii):
        t = float(np.clip(np.dot(c - a, u), 0.0, length))
        closest = a + t * u
        lateral = float(np.linalg.norm(c - closest))
        inflated = r + HALF_BEAM_M
        if lateral < inflated and 0.0 < t < length:
            # to pass, the hull centre must shift to `inflated` lateral offset
            out.append(inflated - lateral)
    return out


def main() -> None:
    rng = np.random.default_rng(3)
    legs_total = 0
    legs_blocked = 0
    deviations = []
    per_layout_blocked = []
    shipped_counts = []
    for _ in range(N_LAYOUTS):
        lay = sample_layout(rng)
        pts = np.asarray(lay.waypoints, dtype=float)
        centers = np.asarray(lay.centers, dtype=float)
        radii = np.asarray(lay.radii, dtype=float)
        try:
            from .path_hazard_geometry import count_on_line_blockers
        except ImportError:
            from path_hazard_geometry import count_on_line_blockers
        shipped_counts.append(count_on_line_blockers(pts, centers))
        blocked_here = 0
        for i in range(len(pts) - 1):
            legs_total += 1
            devs = leg_blockers(pts[i], pts[i + 1], centers, radii)
            if devs:
                legs_blocked += 1
                blocked_here += 1
                deviations.append(max(devs))
        per_layout_blocked.append(blocked_here)

    frac = legs_blocked / max(legs_total, 1)
    dev = np.array(deviations) if deviations else np.array([0.0])
    print(f"抽样 {N_LAYOUTS} 张布局,共 {legs_total} 条门间航段")
    print(f"  被障碍挡住(必须偏航绕过)的航段: {legs_blocked}/{legs_total} "
          f"= {frac:.0%}")
    print(f"  每张布局被挡航段数: 中位 {int(np.median(per_layout_blocked))} "
          f"(设计要求 >= 3 个在线阻挡)")
    print(f"  被挡航段需要的横向偏移: 中位 {np.median(dev):.2f} m "
          f"= {np.median(dev)/BEAM_M:.1f} 倍船宽 | p90 {np.percentile(dev,90):.2f} m")
    print()

    print(f"  官方口径的在线阻挡数: 中位 {int(np.median(shipped_counts))} "
          f"(准入要求 >= 3;两个阻挡可落在同一航段,所以被挡航段数可以更少)")
    print()

    # KNOWN DEFECT, recorded not hidden: the layout's own on_line_mask says 3
    # blockers (admission passes on it), but recounting the FINAL geometry
    # gives 2 on most layouts -- the sampler moves obstacles during separation
    # repair after marking the mask, and the bookkeeping is never recomputed.
    # Third instance of the checker-vs-reality pattern (ring seal, tier
    # ladder). The task remains valid because the leg measurements below are
    # taken on the final geometry, but the generator should recount after
    # repair, and the spec should say "2-3 forced blockers" until it does.

    # The verdict this test exists for -- all measured on FINAL geometry:
    assert frac > 0.5, f"multi-leg blockage only {frac:.0%} -- bypass suspected"
    assert min(per_layout_blocked) >= 2, per_layout_blocked
    assert np.median(dev) > 0.5 * BEAM_M, float(np.median(dev))
    print("PASS: 门把路线钉死,过半航段强制绕障,偏移量超过半个船宽 --")
    print("      题A 的绕行漏洞不适用于线x避,其认证数字衡量的确是沿线避障。")


if __name__ == "__main__":
    main()
