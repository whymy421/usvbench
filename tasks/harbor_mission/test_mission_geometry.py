"""Pure-Python acceptance test for Ordered Harbor Mission geometry."""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    from .harbor_geometry import (
        BERTH_CLEAR_RADIUS_M,
        FIELD_LENGTH_RANGE_M,
        GATE_HALF_WIDTH_M,
        MAX_OBSTACLE_COUNT,
        MAX_OBSTACLE_RADIUS_M,
        MIN_OBSTACLE_COUNT,
        MIN_OBSTACLE_RADIUS_M,
        arm_gate_approach,
        berth_clear_of_obstacles,
        gate_clear_of_obstacles,
        gate_crossing_mask,
        sample_harbor_route,
    )
except ImportError:  # Direct execution from the repository root.
    from harbor_geometry import (
        BERTH_CLEAR_RADIUS_M,
        FIELD_LENGTH_RANGE_M,
        GATE_HALF_WIDTH_M,
        MAX_OBSTACLE_COUNT,
        MAX_OBSTACLE_RADIUS_M,
        MIN_OBSTACLE_COUNT,
        MIN_OBSTACLE_RADIUS_M,
        arm_gate_approach,
        berth_clear_of_obstacles,
        gate_clear_of_obstacles,
        gate_crossing_mask,
        sample_harbor_route,
    )

try:
    from ..hazard_nav.hazard_geometry import bfs_geodesic_length
except ImportError:
    from hazard_geometry import bfs_geodesic_length


ROUTE_COUNT = 50


def _test_gate_crossing_detector() -> None:
    midpoint = torch.tensor([[0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0]])

    # A normal 60 Hz-style trajectory arms at -1 m, then crosses on adjacent
    # samples; it does not need to jump the whole approach distance in one step.
    positions = [
        torch.tensor([[-1.2, 0.5]]),
        torch.tensor([[-0.4, 0.5]]),
        torch.tensor([[0.1, 0.5]]),
        torch.tensor([[0.4, 0.5]]),
    ]
    armed = torch.tensor([False])
    crossings = 0
    for previous, current in zip(positions[:-1], positions[1:], strict=True):
        armed = arm_gate_approach(armed, previous, midpoint, normal)
        crossed = gate_crossing_mask(
            previous,
            current,
            torch.tensor([[0.5, 0.0]]),
            midpoint,
            normal,
            armed,
        )
        crossings += int(crossed.item())
    assert crossings == 1, crossings

    backward = gate_crossing_mask(
        torch.tensor([[0.2, 0.0]]),
        torch.tensor([[-0.2, 0.0]]),
        torch.tensor([[-0.5, 0.0]]),
        midpoint,
        normal,
        torch.tensor([True]),
    )
    assert not backward.item()

    slow_drift = gate_crossing_mask(
        torch.tensor([[-0.1, 0.0]]),
        torch.tensor([[0.1, 0.0]]),
        torch.tensor([[0.19, 0.0]]),
        midpoint,
        normal,
        torch.tensor([True]),
    )
    assert not slow_drift.item()

    outside_segment = gate_crossing_mask(
        torch.tensor([[-0.1, GATE_HALF_WIDTH_M + 0.1]]),
        torch.tensor([[0.1, GATE_HALF_WIDTH_M + 0.1]]),
        torch.tensor([[0.5, 0.0]]),
        midpoint,
        normal,
        torch.tensor([True]),
    )
    assert not outside_segment.item()


def main() -> None:
    _test_gate_crossing_detector()
    rng = np.random.default_rng(20260720)
    feasible_count = 0
    forced_blocker_count = 0
    ordered_gate_count = 0
    clear_berth_count = 0

    for _ in range(ROUTE_COUNT):
        route = sample_harbor_route(rng=rng)
        field_centers = route.obstacle_centers - route.field_start
        recomputed = bfs_geodesic_length(
            np.zeros(2),
            route.field_exit - route.field_start,
            field_centers,
            route.obstacle_radii,
        )
        assert recomputed is not None and math.isfinite(recomputed)
        assert math.isclose(recomputed, route.field_geodesic_length, rel_tol=1.0e-9)
        assert FIELD_LENGTH_RANGE_M[0] <= route.field_length_m <= FIELD_LENGTH_RANGE_M[1]
        assert 60.0 <= route.route_length_m <= 80.0
        assert MIN_OBSTACLE_COUNT <= route.obstacle_count <= MAX_OBSTACLE_COUNT
        assert np.all(route.obstacle_radii >= MIN_OBSTACLE_RADIUS_M)
        assert np.all(route.obstacle_radii <= MAX_OBSTACLE_RADIUS_M)
        assert route.direct_blocked

        gate_x = route.gate_midpoints[:, 0]
        assert np.all(np.diff(gate_x) > 0.0)
        assert all(
            gate_clear_of_obstacles(
                midpoint,
                normal,
                route.gate_half_width_m,
                route.obstacle_centers,
                route.obstacle_radii,
            )
            for midpoint, normal in zip(
                route.gate_midpoints, route.gate_normals, strict=True
            )
        )
        assert berth_clear_of_obstacles(
            route.berth_point,
            route.obstacle_centers,
            route.obstacle_radii,
            clear_radius_m=BERTH_CLEAR_RADIUS_M,
        )

        feasible_count += 1
        forced_blocker_count += int(route.direct_blocked)
        ordered_gate_count += 1
        clear_berth_count += 1

    assert feasible_count == ROUTE_COUNT
    print(f"routes feasible (BFS): {feasible_count}/{ROUTE_COUNT}")
    print(f"forced direct blockers: {forced_blocker_count}/{ROUTE_COUNT}")
    print(f"ordered, obstacle-clear gates: {ordered_gate_count}/{ROUTE_COUNT}")
    print(f"berth clear zones: {clear_berth_count}/{ROUTE_COUNT}")
    print("gate detector: forward=1, backward=0, slow(<0.2 m/s)=0, outside=0")
    print("PASS: 50 harbor routes and synthetic crossing trajectories")


if __name__ == "__main__":
    main()
