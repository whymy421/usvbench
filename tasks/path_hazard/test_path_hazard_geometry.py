"""Pure-Python acceptance test for Path Hazard interaction geometry."""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    from .path_hazard_geometry import (
        HALF_BEAM_M,
        MIN_OBSTACLE_RADIUS_M,
        NUM_SEGMENTS,
        REQUESTED_OBSTACLE_COUNT,
        REQUIRED_ON_LINE_BLOCKERS,
        analytic_min_clearance,
        all_obstacles_within_route_band,
        chain_bfs_geodesic_lengths,
        count_on_line_blockers,
        gate_disks_clear,
        ray_circle_ranges,
        sample_layout,
    )
except ImportError:  # Direct execution from the repository root.
    from path_hazard_geometry import (
        HALF_BEAM_M,
        MIN_OBSTACLE_RADIUS_M,
        NUM_SEGMENTS,
        REQUESTED_OBSTACLE_COUNT,
        REQUIRED_ON_LINE_BLOCKERS,
        analytic_min_clearance,
        all_obstacles_within_route_band,
        chain_bfs_geodesic_lengths,
        count_on_line_blockers,
        gate_disks_clear,
        ray_circle_ranges,
        sample_layout,
    )


LAYOUT_COUNT = 50


def _test_analytic_geometry() -> None:
    points = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    centers = torch.tensor([[[2.0, 0.0]], [[2.0, 0.0]]])
    radii = torch.tensor([[1.0], [1.0]])
    expected = torch.tensor([2.0 - 1.0 - HALF_BEAM_M] * 2)
    assert torch.allclose(analytic_min_clearance(points, centers, radii), expected)

    ranges = ray_circle_ranges(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[5.0, 0.0]]]),
        torch.tensor([[1.0]]),
    )
    assert torch.allclose(ranges, torch.tensor([[4.0, 30.0]])), ranges


def main() -> None:
    _test_analytic_geometry()
    rng = np.random.default_rng(20260720)
    feasible_count = 0
    blocker_count = 0
    gates_clear_count = 0
    radius_floor_count = 0
    full_k_count = 0

    for _ in range(LAYOUT_COUNT):
        layout = sample_layout(rng=rng)
        recomputed = chain_bfs_geodesic_lengths(
            layout.route_points, layout.centers, layout.radii
        )
        feasible = recomputed is not None and np.all(np.isfinite(recomputed))
        assert feasible and layout.chain_feasible
        assert len(recomputed) == NUM_SEGMENTS
        assert np.allclose(recomputed, layout.leg_geodesic_lengths, rtol=1.0e-12)

        measured_blockers = count_on_line_blockers(
            layout.route_points, layout.centers
        )
        blockers_ok = (
            measured_blockers >= REQUIRED_ON_LINE_BLOCKERS
            and layout.on_line_blocker_count >= REQUIRED_ON_LINE_BLOCKERS
        )
        blocker_segments = layout.blocker_segment_indices[layout.on_line_mask]
        assert len(np.unique(blocker_segments)) == REQUIRED_ON_LINE_BLOCKERS
        gates_ok = gate_disks_clear(
            layout.route_points, layout.centers, layout.radii
        )
        assert all_obstacles_within_route_band(
            layout.route_points, layout.centers
        )
        radii_ok = bool(np.all(layout.radii >= MIN_OBSTACLE_RADIUS_M))
        full_k = layout.obstacle_count == REQUESTED_OBSTACLE_COUNT

        assert blockers_ok, measured_blockers
        assert gates_ok
        assert radii_ok, float(np.min(layout.radii))
        assert math.isfinite(layout.route_length)

        feasible_count += int(feasible)
        blocker_count += int(blockers_ok)
        gates_clear_count += int(gates_ok)
        radius_floor_count += int(radii_ok)
        full_k_count += int(full_k)

    assert feasible_count == LAYOUT_COUNT
    assert blocker_count == LAYOUT_COUNT
    assert gates_clear_count == LAYOUT_COUNT
    assert radius_floor_count == LAYOUT_COUNT
    print(f"chain-BFS feasible: {feasible_count}/{LAYOUT_COUNT}")
    print(f">=3 on-line blockers: {blocker_count}/{LAYOUT_COUNT}")
    print(f"gates clear (inflated 3 m disks): {gates_clear_count}/{LAYOUT_COUNT}")
    print(f"all radii >=0.8 m: {radius_floor_count}/{LAYOUT_COUNT}")
    print(f"full K={REQUESTED_OBSTACLE_COUNT}: {full_k_count}/{LAYOUT_COUNT}")
    print("PASS: Path Hazard geometry interaction and feasibility assertions passed")


if __name__ == "__main__":
    main()
