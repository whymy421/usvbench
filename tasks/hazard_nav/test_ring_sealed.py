"""Does the ring actually seal against the REAL hull, or only against BFS?

The shipped ring places neighbours so their INFLATED disks (radius + 0.65)
overlap by 0.30 m, leaving 2*0.65 - 0.30 = 1.00 m between the physical
surfaces -- wider than the 0.899 m hull. The admission test missed it because
its BFS inflates obstacles by 0.65 while the simulator's contact predicate
uses the true half-beam 0.45: the checker was testing a different, fatter boat
than the one being simulated.

This measures the narrowest ADJACENT opening on the ring (ordering cylinders
by bearing around the ring centre, so opposite sides are never mistaken for an
opening) and requires it to be narrower than the hull everywhere except the
one designed gap.
"""

from __future__ import annotations

import numpy as np

try:
    from .hazard_geometry import (
        RING_NEIGHBOR_OVERLAP_M,
        RING_SEALED_OVERLAP_M,
        difficulty_for_level,
        sample_ring_layout,
    )
except ImportError:  # direct execution
    from hazard_geometry import (
        RING_NEIGHBOR_OVERLAP_M,
        RING_SEALED_OVERLAP_M,
        difficulty_for_level,
        sample_ring_layout,
    )

HULL_BEAM_M = 0.899
TRIALS = 15


def adjacent_openings(layout):
    """Surface-to-surface gaps between cylinders adjacent on the ring."""
    c = np.asarray(layout.centers, dtype=float)
    r = np.asarray(layout.radii, dtype=float)
    hub = c.mean(axis=0)
    order = np.argsort(np.arctan2(c[:, 1] - hub[1], c[:, 0] - hub[0]))
    c, r = c[order], r[order]
    nxt = np.roll(np.arange(len(c)), -1)
    return np.linalg.norm(c - c[nxt], axis=-1) - r - r[nxt]


def survey(level, overlap):
    rng = np.random.default_rng(11)
    escapes, widest_non_gap, counts = 0, [], []
    for _ in range(TRIALS):
        lay = sample_ring_layout(level, rng, neighbor_overlap_m=overlap)
        gaps = np.sort(adjacent_openings(lay))
        # the largest opening is the designed gap; everything else must be
        # narrower than the hull
        others = gaps[:-1]
        widest_non_gap.append(float(others.max()))
        n_escapes = int((others > HULL_BEAM_M).sum())
        counts.append(len(gaps))
        if n_escapes:
            escapes += 1
    return escapes, float(np.median(widest_non_gap)), int(np.median(counts))


def main() -> None:
    print(f"船宽 {HULL_BEAM_M} m | 每档 {TRIALS} 张布局")
    print()
    print(f"{'档位':<6}{'重叠量':>8}{'最宽的非缝开口':>16}{'能钻出去的布局':>16}{'圆柱数':>8}")
    print("-" * 56)
    for label, overlap in (("现有", RING_NEIGHBOR_OVERLAP_M),
                           ("封死", RING_SEALED_OVERLAP_M)):
        for level in (0, 3):
            esc, widest, n = survey(level, overlap)
            mark = "  <-- 漏" if esc else ""
            print(f"{label}{level:<4}{overlap:>8.2f}{widest:>16.2f}"
                  f"{esc:>13}/{TRIALS}{n:>8}{mark}")

    print()
    esc_old, widest_old, _ = survey(0, RING_NEIGHBOR_OVERLAP_M)
    esc_new, widest_new, _ = survey(0, RING_SEALED_OVERLAP_M)
    assert esc_old > 0, "shipped ring was expected to leak"
    assert esc_new == 0, f"sealed ring still leaks: widest {widest_new:.2f} m"
    assert widest_new < HULL_BEAM_M, widest_new
    print(f"PASS: 重叠 {RING_SEALED_OVERLAP_M} 时最宽的非缝开口 {widest_new:.2f} m "
          f"< 船宽 {HULL_BEAM_M} m,唯一出口是设计的缝")


if __name__ == "__main__":
    main()
