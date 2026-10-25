"""Admission tests for the ring-siege layout sampler (pure NumPy, no Isaac)."""

from __future__ import annotations

import numpy as np

try:
    from .hazard_geometry import (
        DIFFICULTIES,
        ENDPOINT_CLEAR_RADIUS_M,
        OBSTACLE_INFLATION_M,
        RING_GAP_SLACK_M,
        bfs_geodesic_length,
        inflated_radii,
        sample_ring_layout,
        start_goal_disks_clear,
    )
except ImportError:  # direct execution
    from hazard_geometry import (
        DIFFICULTIES,
        ENDPOINT_CLEAR_RADIUS_M,
        OBSTACLE_INFLATION_M,
        RING_GAP_SLACK_M,
        bfs_geodesic_length,
        inflated_radii,
        sample_ring_layout,
        start_goal_disks_clear,
    )

SAMPLES_PER_LEVEL = 50


def _adjacent_free_gaps(layout) -> np.ndarray:
    """Free inflated gaps between angularly adjacent ring obstacles."""
    angles = np.arctan2(layout.centers[:, 1], layout.centers[:, 0])
    order = np.argsort(angles)
    centers = layout.centers[order]
    radii_i = inflated_radii(layout.radii[order])
    gaps = []
    n = len(order)
    for k in range(n):
        a, b = k, (k + 1) % n
        chord = float(np.linalg.norm(centers[a] - centers[b]))
        gaps.append(chord - radii_i[a] - radii_i[b])
    return np.asarray(gaps)


def main() -> None:
    rng = np.random.default_rng(7)
    for level, diff in sorted(DIFFICULTIES.items()):
        widths = []
        counts = []
        for _ in range(SAMPLES_PER_LEVEL):
            layout = sample_ring_layout(level, rng=rng)
            assert layout.feasible, f"L{level}: infeasible layout admitted"
            assert start_goal_disks_clear(
                layout.start, layout.goal, layout.centers, layout.radii
            ), f"L{level}: endpoint disk violated"
            gaps = _adjacent_free_gaps(layout)
            passable = gaps[gaps > 0.45]  # wider than one hull half-beam pair
            assert len(passable) == 1, (
                f"L{level}: expected exactly one passable gap, got {len(passable)} "
                f"(gaps={np.round(gaps, 2)})"
            )
            width = float(passable[0])
            assert diff.bottleneck_m - 1e-6 <= width <= diff.bottleneck_m + RING_GAP_SLACK_M + 1e-6, (
                f"L{level}: gap width {width:.3f} outside "
                f"[{diff.bottleneck_m:.3f}, {diff.bottleneck_m + RING_GAP_SLACK_M:.3f}]"
            )
            assert layout.obstacle_count <= 18, f"L{level}: {layout.obstacle_count} obstacles"
            widths.append(width)
            counts.append(layout.obstacle_count)
        print(
            f"L{level}: {SAMPLES_PER_LEVEL} rings OK | gap width "
            f"{np.mean(widths):.2f}±{np.std(widths):.2f} m "
            f"(tier {diff.bottleneck_m:.2f}) | obstacles {min(counts)}-{max(counts)}"
        )
    print("PASS: ring-siege admission (sealed, single tier-width gap, clear endpoints)")


if __name__ == "__main__":
    main()
