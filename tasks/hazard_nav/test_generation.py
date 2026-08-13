"""Pure-Python acceptance test for the Hazard Navigation layout oracle."""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    from .hazard_geometry import (
        DIFFICULTIES,
        HALF_BEAM_M,
        analytic_min_clearance,
        bfs_geodesic_length,
        direct_segment_blocked,
        minimum_pairwise_inflated_gap,
        ray_circle_ranges,
        sample_layout,
        start_goal_disks_clear,
    )
except ImportError:  # Direct execution: python tasks/hazard_nav/test_generation.py
    from hazard_geometry import (
        DIFFICULTIES,
        HALF_BEAM_M,
        analytic_min_clearance,
        bfs_geodesic_length,
        direct_segment_blocked,
        minimum_pairwise_inflated_gap,
        ray_circle_ranges,
        sample_layout,
        start_goal_disks_clear,
    )


LAYOUTS_PER_LEVEL = 50


def _test_clearance_math() -> None:
    points = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    centers = torch.tensor([[[2.0, 0.0]], [[2.0, 0.0]]])
    radii = torch.tensor([[1.0], [1.0]])
    expected = torch.tensor([2.0 - 1.0 - HALF_BEAM_M] * 2)
    actual = analytic_min_clearance(points, centers, radii)
    assert torch.allclose(actual, expected), (actual, expected)

    origins = torch.tensor([[0.0, 0.0]])
    directions = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    ray_centers = torch.tensor([[[5.0, 0.0]]])
    ray_radii = torch.tensor([[1.0]])
    ranges = ray_circle_ranges(
        origins, directions, ray_centers, ray_radii, max_range_m=30.0
    )
    assert torch.allclose(ranges, torch.tensor([[4.0, 30.0]])), ranges


def main() -> None:
    _test_clearance_math()
    rng = np.random.default_rng(20260720)

    for level, difficulty in DIFFICULTIES.items():
        feasible_count = 0
        blocked_count = 0
        full_count = 0
        for _ in range(LAYOUTS_PER_LEVEL):
            layout = sample_layout(level, rng=rng)
            recomputed = bfs_geodesic_length(
                layout.start, layout.goal, layout.centers, layout.radii
            )
            assert recomputed is not None and math.isfinite(recomputed)
            assert math.isclose(recomputed, layout.geodesic_length, rel_tol=1.0e-9)
            assert np.all(2.0 * layout.radii >= 1.0)
            assert start_goal_disks_clear(
                layout.start, layout.goal, layout.centers, layout.radii
            )
            assert (
                minimum_pairwise_inflated_gap(layout.centers, layout.radii)
                + 1.0e-8
                >= difficulty.bottleneck_m
            )

            blocked = direct_segment_blocked(
                layout.start, layout.goal, layout.centers, layout.radii
            )
            assert blocked == layout.direct_blocked
            feasible_count += 1
            blocked_count += int(blocked)
            full_count += int(layout.obstacle_count == difficulty.obstacle_count)

        assert feasible_count == LAYOUTS_PER_LEVEL
        assert blocked_count / LAYOUTS_PER_LEVEL >= 0.60
        print(
            f"level {level}: feasible {feasible_count}/{LAYOUTS_PER_LEVEL}, "
            f"direct blocked {blocked_count}/{LAYOUTS_PER_LEVEL}, "
            f"full K={difficulty.obstacle_count} {full_count}/{LAYOUTS_PER_LEVEL}"
        )

    print("PASS: 150 feasible layouts; geometry and clearance assertions passed")


if __name__ == "__main__":
    main()
