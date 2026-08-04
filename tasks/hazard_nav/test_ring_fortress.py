"""CPU-only admission audit for the owner-designed ring fortresses.

Run ``python tasks/hazard_nav/test_ring_fortress.py --layouts 10000`` for the
full audit. ``--layouts`` is the count per variant, spread across all tiers.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

try:
    from .hazard_geometry import (
        DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M,
        DOUBLE_RING_MIN_GAP_OFFSET_RAD,
        DOUBLE_RING_OUTER_RADIUS_M,
        FORCED_SEAL_CELL_M,
        FORTRESS_SPAWN_RADIUS_RANGE_M,
        HALF_BEAM_M,
        OBSTACLE_INFLATION_M,
        RING_OBSTACLE_RADIUS_MAX_M,
        RING_OBSTACLE_RADIUS_MIN_M,
        RING_RADIUS_M,
        RING_SEALED_OVERLAP_M,
        _ring_line_openings,
        bfs_geodesic_length,
        difficulty_for_level,
        sample_double_ring_fortress_layout,
        sample_ring_fortress_layout,
        wall_segment_cylinders,
    )
except ImportError:  # direct execution
    from hazard_geometry import (  # type: ignore
        DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M,
        DOUBLE_RING_MIN_GAP_OFFSET_RAD,
        DOUBLE_RING_OUTER_RADIUS_M,
        FORCED_SEAL_CELL_M,
        FORTRESS_SPAWN_RADIUS_RANGE_M,
        HALF_BEAM_M,
        OBSTACLE_INFLATION_M,
        RING_OBSTACLE_RADIUS_MAX_M,
        RING_OBSTACLE_RADIUS_MIN_M,
        RING_RADIUS_M,
        RING_SEALED_OVERLAP_M,
        _ring_line_openings,
        bfs_geodesic_length,
        difficulty_for_level,
        sample_double_ring_fortress_layout,
        sample_ring_fortress_layout,
        wall_segment_cylinders,
    )


LEVELS = (0, 1, 2, 3)
RAY_MAX_RANGE_M = 30.0


def _scan_and_audit_overlap(
    centers: np.ndarray,
    radii: np.ndarray,
    ring_radius_m: float,
    target_width_m: float,
) -> tuple[float, float, int, int]:
    """Measure one opening and enforce the sealed 0.55 m overlap floor."""
    openings = _ring_line_openings(centers, radii, ring_radius_m)
    assert len(openings) == 1, f"found {len(openings)} openings"
    opening = openings[0]
    assert abs(opening[0] - target_width_m) <= 0.01, (
        opening[0],
        target_width_m,
    )

    bearings = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2.0 * math.pi)
    order = np.argsort(bearings)
    for position, before in enumerate(order):
        after = int(order[(position + 1) % len(order)])
        before = int(before)
        if before == opening[2] and after == opening[3]:
            continue
        distance = float(np.linalg.norm(centers[before] - centers[after]))
        # The sealed single-ring constructor defines the floor on planning-
        # inflated disks. The double-ring pair physically overlaps and thus
        # exceeds the same floor by another 2*inflation.
        realised_overlap = float(
            radii[before]
            + radii[after]
            + 2.0 * OBSTACLE_INFLATION_M
            - distance
        )
        assert realised_overlap + 1.0e-8 >= RING_SEALED_OVERLAP_M, (
            realised_overlap,
            RING_SEALED_OVERLAP_M,
        )
    return opening


def _plug(
    ring_centers: np.ndarray,
    opening: tuple[float, float, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Tile the scanned jambs exactly like the sealed-ring audit."""
    plug_radius = 0.5 * (
        RING_OBSTACLE_RADIUS_MIN_M + RING_OBSTACLE_RADIUS_MAX_M
    )
    plug_centers = wall_segment_cylinders(
        ring_centers[opening[2]],
        ring_centers[opening[3]],
        radius_m=plug_radius,
        overlap_m=RING_SEALED_OVERLAP_M,
    )
    if len(plug_centers) > 1:
        separations = np.linalg.norm(np.diff(plug_centers, axis=0), axis=1)
        realised_overlap = 2.0 * plug_radius - separations
        assert np.all(realised_overlap + 1.0e-8 >= RING_SEALED_OVERLAP_M)
    return plug_centers, np.full(len(plug_centers), plug_radius)


def _assert_spawn_sanity(
    start: np.ndarray,
    goal: np.ndarray,
    spawn_range_m: tuple[float, float],
    outer_ring_radius_m: float,
) -> float:
    assert np.allclose(goal, 0.0), goal
    spawn_radius = float(np.linalg.norm(start))
    assert spawn_range_m[0] <= spawn_radius <= spawn_range_m[1], spawn_radius
    # Stronger than centerline-only: the spawn is >=2 m beyond the largest
    # possible physical cylinder surface on the outer ring.
    assert spawn_radius - outer_ring_radius_m - RING_OBSTACLE_RADIUS_MAX_M >= 2.0
    nearest_centerline_m = spawn_radius - outer_ring_radius_m
    assert nearest_centerline_m <= RAY_MAX_RANGE_M, nearest_centerline_m
    return nearest_centerline_m


def _assert_open_and_sealed(
    layout,
    rings: list[tuple[np.ndarray, tuple[float, float, int, int]]],
) -> None:
    open_route = bfs_geodesic_length(
        layout.start,
        layout.goal,
        layout.centers,
        layout.radii,
        cell_m=FORCED_SEAL_CELL_M,
        inflation_m=HALF_BEAM_M,
    )
    assert open_route is not None, "no true-half-beam route with gaps open"

    # Plug each ring independently. For fortress2, either sealed wall alone
    # must disconnect the outside spawn from the center goal.
    for ring_centers, opening in rings:
        plug_centers, plug_radii = _plug(ring_centers, opening)
        sealed_route = bfs_geodesic_length(
            layout.goal,
            layout.start,
            np.vstack((layout.centers, plug_centers)),
            np.concatenate((layout.radii, plug_radii)),
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        )
        assert sealed_route is None, f"plugged ring still routes in {sealed_route:.1f} m"


def _audit_fortress(layouts: int, rng: np.random.Generator) -> None:
    max_count = 0
    max_visible_distance = 0.0
    for index in range(layouts):
        level = LEVELS[index % len(LEVELS)]
        target = difficulty_for_level(level).bottleneck_m
        layout, _declared_gap = sample_ring_fortress_layout(level, rng=rng)
        opening = _scan_and_audit_overlap(
            layout.centers, layout.radii, RING_RADIUS_M, target
        )
        _assert_open_and_sealed(layout, [(layout.centers, opening)])
        max_visible_distance = max(
            max_visible_distance,
            _assert_spawn_sanity(
                layout.start,
                layout.goal,
                FORTRESS_SPAWN_RADIUS_RANGE_M,
                RING_RADIUS_M,
            ),
        )
        max_count = max(max_count, layout.obstacle_count)
    print(
        f"fortress: {layouts} layouts, 0 overlap-floor violations, "
        f"sealed/open routes verified, max cylinders {max_count}, "
        f"max nearest-ring distance {max_visible_distance:.3f} m"
    )


def _audit_fortress2(layouts: int, rng: np.random.Generator) -> None:
    max_count = 0
    max_visible_distance = 0.0
    for index in range(layouts):
        level = LEVELS[index % len(LEVELS)]
        target = difficulty_for_level(level).bottleneck_m
        layout, _declared_gaps = sample_double_ring_fortress_layout(level, rng=rng)
        radial_distance = np.linalg.norm(layout.centers, axis=1)
        inner_mask = np.isclose(radial_distance, RING_RADIUS_M, atol=1.0e-8)
        outer_mask = np.isclose(
            radial_distance, DOUBLE_RING_OUTER_RADIUS_M, atol=1.0e-8
        )
        assert int(inner_mask.sum() + outer_mask.sum()) == layout.obstacle_count
        inner_centers, inner_radii = (
            layout.centers[inner_mask],
            layout.radii[inner_mask],
        )
        outer_centers, outer_radii = (
            layout.centers[outer_mask],
            layout.radii[outer_mask],
        )
        inner_opening = _scan_and_audit_overlap(
            inner_centers, inner_radii, RING_RADIUS_M, target
        )
        outer_opening = _scan_and_audit_overlap(
            outer_centers, outer_radii, DOUBLE_RING_OUTER_RADIUS_M, target
        )
        offset = abs(
            (outer_opening[1] - inner_opening[1] + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        assert offset + 1.0e-9 >= DOUBLE_RING_MIN_GAP_OFFSET_RAD, offset
        _assert_open_and_sealed(
            layout,
            [(inner_centers, inner_opening), (outer_centers, outer_opening)],
        )
        max_visible_distance = max(
            max_visible_distance,
            _assert_spawn_sanity(
                layout.start,
                layout.goal,
                DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M,
                DOUBLE_RING_OUTER_RADIUS_M,
            ),
        )
        max_count = max(max_count, layout.obstacle_count)
    print(
        f"fortress2: {layouts} layouts, 0 overlap-floor violations, "
        f"both seals/open routes verified, max cylinders {max_count}, "
        f"max nearest-ring distance {max_visible_distance:.3f} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.layouts < 1:
        parser.error("--layouts must be at least one")

    started = time.time()
    rng = np.random.default_rng(args.seed)
    _audit_fortress(args.layouts, rng)
    _audit_fortress2(args.layouts, rng)
    print(f"PASS: {2 * args.layouts} fortress layouts, 0 violations ({time.time() - started:.1f} s)")


if __name__ == "__main__":
    main()
