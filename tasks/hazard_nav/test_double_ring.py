"""Admission audit for the owner-approved double-ring siege.

The constructor intends to make two walls, but this audit earns that claim
from the arrays the environment actually receives. Openings are independently
scanned on each ring line, physical neighbour overlap is measured, and each
gap is plugged into the shipped arrays before a true-half-beam seal test.

Run ``python tasks/hazard_nav/test_double_ring.py --layouts 1000`` for a
larger audit. The default is 25 layouts per tier.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

try:
    from .hazard_geometry import (
        DOUBLE_RING_GOAL_DISTANCE_RANGE_M,
        DOUBLE_RING_MIN_GAP_OFFSET_RAD,
        DOUBLE_RING_OUTER_RADIUS_M,
        FORCED_SEAL_CELL_M,
        HALF_BEAM_M,
        OBSTACLE_INFLATION_M,
        RING_OBSTACLE_RADIUS_MAX_M,
        RING_OBSTACLE_RADIUS_MIN_M,
        RING_RADIUS_M,
        RING_SEALED_OVERLAP_M,
        bfs_geodesic_length,
        difficulty_for_level,
        sample_double_ring_layout,
        start_goal_disks_clear,
        wall_segment_cylinders,
    )
except ImportError:  # direct execution
    from hazard_geometry import (  # type: ignore
        DOUBLE_RING_GOAL_DISTANCE_RANGE_M,
        DOUBLE_RING_MIN_GAP_OFFSET_RAD,
        DOUBLE_RING_OUTER_RADIUS_M,
        FORCED_SEAL_CELL_M,
        HALF_BEAM_M,
        OBSTACLE_INFLATION_M,
        RING_OBSTACLE_RADIUS_MAX_M,
        RING_OBSTACLE_RADIUS_MIN_M,
        RING_RADIUS_M,
        RING_SEALED_OVERLAP_M,
        bfs_geodesic_length,
        difficulty_for_level,
        sample_double_ring_layout,
        start_goal_disks_clear,
        wall_segment_cylinders,
    )

LEVELS = (0, 1, 2, 3)


def _scan_openings(
    centers: np.ndarray,
    radii: np.ndarray,
    ring_radius_m: float,
) -> list[tuple[float, float, int, int]]:
    """Independently scan uncovered runs on the planning-inflated ring line."""
    bearings = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2.0 * math.pi)
    order = np.argsort(bearings)
    bearings = bearings[order]
    inflated = radii[order] + OBSTACLE_INFLATION_M
    half_arcs = 2.0 * np.arcsin(
        np.clip(inflated / (2.0 * ring_radius_m), 0.0, 1.0)
    )
    openings: list[tuple[float, float, int, int]] = []
    for position, before_index in enumerate(order):
        next_position = (position + 1) % len(order)
        start = float(bearings[next_position] - half_arcs[next_position])
        if next_position == 0:
            start += 2.0 * math.pi
        end = float(bearings[position] + half_arcs[position])
        free_angle = start - end
        if free_angle > 1.0e-10:
            openings.append(
                (
                    ring_radius_m * free_angle,
                    (end + 0.5 * free_angle) % (2.0 * math.pi),
                    int(before_index),
                    int(order[next_position]),
                )
            )
    return openings


def _nearest_surface_gaps(centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    delta = centers[:, None, :] - centers[None, :, :]
    gaps = np.linalg.norm(delta, axis=-1) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, np.inf)
    return gaps.min(axis=1)


def _plug(
    ring_centers: np.ndarray,
    opening: tuple[float, float, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    plug_radius = 0.5 * (
        RING_OBSTACLE_RADIUS_MIN_M + RING_OBSTACLE_RADIUS_MAX_M
    )
    plug_centers = wall_segment_cylinders(
        ring_centers[opening[2]],
        ring_centers[opening[3]],
        radius_m=plug_radius,
        overlap_m=RING_SEALED_OVERLAP_M,
    )
    return plug_centers, np.full(len(plug_centers), plug_radius)


def audit_one(level: int, rng: np.random.Generator) -> dict:
    """Re-derive the six admissions in their declared order."""
    layout, _declared_gaps = sample_double_ring_layout(level, rng=rng)
    centers, radii = layout.centers, layout.radii
    radial_distance = np.linalg.norm(centers, axis=1)
    inner_mask = np.isclose(radial_distance, RING_RADIUS_M, atol=1.0e-8)
    outer_mask = np.isclose(
        radial_distance, DOUBLE_RING_OUTER_RADIUS_M, atol=1.0e-8
    )
    inner_c, inner_r = centers[inner_mask], radii[inner_mask]
    outer_c, outer_r = centers[outer_mask], radii[outer_mask]
    violations: list[str] = []
    target = difficulty_for_level(level).bottleneck_m

    if int(inner_mask.sum() + outer_mask.sum()) != len(radii):
        violations.append("a cylinder is not on either declared ring")
    if np.any(radii < RING_OBSTACLE_RADIUS_MIN_M) or np.any(
        radii > RING_OBSTACLE_RADIUS_MAX_M
    ):
        violations.append("a cylinder radius is outside [1.5, 2.0] m")

    # 1. Scan rather than trusting the gap metadata or constructor bearings.
    inner_openings = _scan_openings(inner_c, inner_r, RING_RADIUS_M)
    outer_openings = _scan_openings(outer_c, outer_r, DOUBLE_RING_OUTER_RADIUS_M)
    if len(inner_openings) != 1:
        violations.append(f"inner ring has {len(inner_openings)} openings, expected 1")
    if len(outer_openings) != 1:
        violations.append(f"outer ring has {len(outer_openings)} openings, expected 1")
    inner_gap = inner_openings[0] if len(inner_openings) == 1 else None
    outer_gap = outer_openings[0] if len(outer_openings) == 1 else None
    if inner_gap is not None and abs(inner_gap[0] - target) > 0.01:
        violations.append(f"inner gap {inner_gap[0]:.3f} != target {target:.3f}")
    if outer_gap is not None and abs(outer_gap[0] - target) > 0.01:
        violations.append(f"outer gap {outer_gap[0]:.3f} != target {target:.3f}")

    offset_deg = math.nan
    if inner_gap is not None and outer_gap is not None:
        offset = abs((outer_gap[1] - inner_gap[1] + math.pi) % (2.0 * math.pi) - math.pi)
        offset_deg = math.degrees(offset)
        if offset + 1.0e-9 < DOUBLE_RING_MIN_GAP_OFFSET_RAD:
            violations.append(f"gap offset {offset_deg:.2f} deg is below 90 deg")

    # 2. Same-ring PHYSICAL surface overlap, not planner-disk overlap.
    inner_worst = float(_nearest_surface_gaps(inner_c, inner_r).max())
    outer_worst = float(_nearest_surface_gaps(outer_c, outer_r).max())
    if inner_worst >= 0.0:
        violations.append(f"inner wall nearest-neighbour gap is {inner_worst:.3f} m")
    if outer_worst >= 0.0:
        violations.append(f"outer wall nearest-neighbour gap is {outer_worst:.3f} m")

    # 3. Planning-inflated route through both openings.
    planning_route = bfs_geodesic_length(
        layout.start,
        layout.goal,
        centers,
        radii,
        inflation_m=OBSTACLE_INFLATION_M,
    )
    if planning_route is None:
        violations.append("no route at planning inflation")

    # 4. Plug only the inner scanned gap into the SHIPPED arrays.
    inner_sealed = False
    if inner_gap is not None:
        plug_c, plug_r = _plug(inner_c, inner_gap)
        inner_leak = bfs_geodesic_length(
            layout.start,
            layout.goal,
            np.vstack((centers, plug_c)),
            np.concatenate((radii, plug_r)),
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        )
        inner_sealed = inner_leak is None
        if not inner_sealed:
            violations.append(f"plugged inner ring still routes in {inner_leak:.1f} m")

    # 5. Plug only the outer scanned gap into the SHIPPED arrays.
    outer_sealed = False
    if outer_gap is not None:
        plug_c, plug_r = _plug(outer_c, outer_gap)
        outer_leak = bfs_geodesic_length(
            layout.start,
            layout.goal,
            np.vstack((centers, plug_c)),
            np.concatenate((radii, plug_r)),
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        )
        outer_sealed = outer_leak is None
        if not outer_sealed:
            violations.append(f"plugged outer ring still routes in {outer_leak:.1f} m")

    # 6. The same true-half-beam checker routes when both gaps are open.
    open_route = bfs_geodesic_length(
        layout.start,
        layout.goal,
        centers,
        radii,
        cell_m=FORCED_SEAL_CELL_M,
        inflation_m=HALF_BEAM_M,
    )
    open_routable = open_route is not None
    if not open_routable:
        violations.append("no true-half-beam route with both gaps open")

    goal_distance = float(np.linalg.norm(layout.goal - layout.start))
    if not (
        DOUBLE_RING_GOAL_DISTANCE_RANGE_M[0]
        <= goal_distance
        <= DOUBLE_RING_GOAL_DISTANCE_RANGE_M[1]
    ):
        violations.append(f"goal distance {goal_distance:.3f} m is outside its range")
    if not start_goal_disks_clear(layout.start, layout.goal, centers, radii):
        violations.append("a shipped cylinder enters a 5 m endpoint clear disk")

    return {
        "violations": violations,
        "inner_gap_m": inner_gap[0] if inner_gap is not None else math.nan,
        "outer_gap_m": outer_gap[0] if outer_gap is not None else math.nan,
        "offset_deg": offset_deg,
        "cylinders": len(radii),
        "inner_sealed": inner_sealed,
        "outer_sealed": outer_sealed,
        "open_routable": open_routable,
        "inner_worst_gap_m": inner_worst,
        "outer_worst_gap_m": outer_worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layouts",
        type=int,
        default=25,
        help="layouts to audit per tier (default: 25)",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.layouts < 1:
        parser.error("--layouts must be at least one")

    rng = np.random.default_rng(args.seed)
    started = time.time()
    all_violations: list[str] = []
    rows: list[tuple[int, list[dict]]] = []

    print(
        f"double-ring admission audit: {args.layouts} layouts/tier x "
        f"{len(LEVELS)} tiers"
    )
    print(
        f"  planning inflation {OBSTACLE_INFLATION_M:.2f} m (route/opening scan) / "
        f"seal inflation {HALF_BEAM_M:.2f} m, cell {FORCED_SEAL_CELL_M} m"
    )
    print()
    print(
        "tier  target    inner gap range    outer gap range    offset deg "
        "(min/median)  cylinders (max)"
    )
    print("----  ------  -----------------  -----------------  -----------------------  ---------------")
    for level in LEVELS:
        stats = [audit_one(level, rng) for _ in range(args.layouts)]
        rows.append((level, stats))
        bad = [message for stat in stats for message in stat["violations"]]
        all_violations.extend(bad)
        inner = np.array([stat["inner_gap_m"] for stat in stats])
        outer = np.array([stat["outer_gap_m"] for stat in stats])
        offsets = np.array([stat["offset_deg"] for stat in stats])
        cylinders = np.array([stat["cylinders"] for stat in stats])
        target = difficulty_for_level(level).bottleneck_m
        print(
            f" {level:>1}    {target:>5.3f}   {inner.min():>6.3f}-{inner.max():<6.3f}     "
            f"{outer.min():>6.3f}-{outer.max():<6.3f}       "
            f"{offsets.min():>6.1f}/{np.median(offsets):<6.1f}          "
            f"{cylinders.min():>2}-{cylinders.max():<2} ({cylinders.max():>2})"
        )
        inner_seals = sum(stat["inner_sealed"] for stat in stats)
        outer_seals = sum(stat["outer_sealed"] for stat in stats)
        open_routes = sum(stat["open_routable"] for stat in stats)
        print(
            f"      seal verdicts: inner {inner_seals}/{args.layouts}, "
            f"outer {outer_seals}/{args.layouts}; both-open true-beam routes "
            f"{open_routes}/{args.layouts}; violations {len(bad)}"
        )
        if bad:
            for message in bad[:5]:
                print(f"        ! {message}")
    print()

    worst_cylinders = max(
        stat["cylinders"] for _level, stats in rows for stat in stats
    )
    print(f"elapsed {time.time() - started:.0f} s")
    print(f"max cylinders over every tier: {worst_cylinders} (cfg max_obstacles: 80)")
    print()
    total = args.layouts * len(LEVELS)
    if all_violations:
        print(f"=> FAIL: {len(all_violations)} violations across {total} layouts.")
        raise SystemExit(1)

    print(f"=> PASS: {total} layouts, 0 violations.")
    print("   Each shipped ring has exactly one measured tier-width opening; plugging")
    print("   either opening alone makes the goal unreachable at the true half-beam.")
    print()
    print("   What this PASS does not claim:")
    print("   - it does not claim a learned policy can discover or reliably transit")
    print("     both gaps; that is a training and evaluation result, not admission;")
    print("   - the 0.25 m occupancy-grid verdict is a conservative geometry audit,")
    print("     not a continuous-dynamics proof or a collision-free trajectory;")
    print("   - tiers control only the two opening widths. Gap bearings, route length,")
    print("     and realised control difficulty remain stochastic within every tier.")


if __name__ == "__main__":
    main()
