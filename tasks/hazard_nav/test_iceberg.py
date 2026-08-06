"""CPU-only admission audit for long-range iceberg detours."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np


GEOMETRY_PATH = Path(__file__).with_name("hazard_geometry.py")
SPEC = importlib.util.spec_from_file_location("iceberg_geometry", GEOMETRY_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load {GEOMETRY_PATH}")
sys.dont_write_bytecode = True
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


LEVELS = tuple(sorted(geometry.ICEBERG_TIERS))
GOAL_DISK_RADIUS_M = 2.0
ENDPOINT_DISK_CLEARANCE_M = 3.0


def _minimum_physical_edge_gap(
    centers: np.ndarray, radii: np.ndarray
) -> float:
    if len(radii) < 2:
        return math.inf
    delta = centers[:, None, :] - centers[None, :, :]
    gaps = np.linalg.norm(delta, axis=-1) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, math.inf)
    return float(np.min(gaps))


def _endpoint_disk_clearance(layout) -> float:
    start = np.linalg.norm(layout.centers - layout.start, axis=1)
    goal = np.linalg.norm(layout.centers - layout.goal, axis=1)
    physical_edge_distance = np.minimum(start, goal) - layout.radii
    return float(np.min(physical_edge_distance - GOAL_DISK_RADIUS_M))


def _route_bounds(layout) -> tuple[float, float, float, float]:
    inflated = layout.radii + geometry.HALF_BEAM_M
    x_min = min(
        float(layout.start[0]),
        float(layout.goal[0]),
        float(np.min(layout.centers[:, 0] - inflated)),
    )
    x_max = max(
        float(layout.start[0]),
        float(layout.goal[0]),
        float(np.max(layout.centers[:, 0] + inflated)),
    )
    y_min = min(
        float(layout.start[1]),
        float(layout.goal[1]),
        float(np.min(layout.centers[:, 1] - inflated)),
    )
    y_max = max(
        float(layout.start[1]),
        float(layout.goal[1]),
        float(np.max(layout.centers[:, 1] + inflated)),
    )
    padding = 6.0
    return x_min - padding, x_max + padding, y_min - padding, y_max + padding


def _audit_level(level: int, layouts: int, rng: np.random.Generator) -> dict:
    budget, radius_min, radius_max = geometry.ICEBERG_TIERS[level]
    ratios: list[float] = []
    d0_values: list[float] = []
    routes: list[float] = []
    minimum_gap = math.inf
    minimum_endpoint_clearance = math.inf
    maximum_count = 0

    for _ in range(layouts):
        layout = geometry.sample_iceberg_layout(
            level, rng=rng, max_attempts=60
        )
        assert layout.obstacle_count <= budget, (
            level,
            layout.obstacle_count,
            budget,
        )
        assert np.all(layout.radii >= radius_min)
        assert np.all(layout.radii <= radius_max)

        blocked = geometry.direct_segment_blocked(
            layout.start,
            layout.goal,
            layout.centers,
            layout.radii,
            inflation_m=geometry.HALF_BEAM_M,
        )
        assert blocked, f"level {level}: direct segment is not blocked"

        route = geometry.bfs_geodesic_length(
            layout.start,
            layout.goal,
            layout.centers,
            layout.radii,
            inflation_m=geometry.HALF_BEAM_M,
        )
        assert route is not None, f"level {level}: no half-beam route"

        gap = _minimum_physical_edge_gap(layout.centers, layout.radii)
        assert gap >= geometry.ICEBERG_EDGE_GAP_M - 1.0e-6, (
            level,
            gap,
        )
        endpoint_clearance = _endpoint_disk_clearance(layout)
        assert endpoint_clearance >= ENDPOINT_DISK_CLEARANCE_M - 1.0e-6, (
            level,
            endpoint_clearance,
        )

        routed = geometry.route_geodesic_length(
            layout.start,
            layout.centers,
            layout.radii,
            layout.goal,
            _route_bounds(layout),
            half_beam_m=geometry.HALF_BEAM_M,
        )
        assert routed is not None, f"level {level}: A* router found no route"
        d0 = float(np.linalg.norm(layout.goal - layout.start))
        d0_values.append(d0)
        routes.append(float(routed))
        ratios.append(float(routed) / d0)
        minimum_gap = min(minimum_gap, gap)
        minimum_endpoint_clearance = min(
            minimum_endpoint_clearance, endpoint_clearance
        )
        maximum_count = max(maximum_count, layout.obstacle_count)

    ratio_values = np.asarray(ratios)
    worst = int(np.argmax(ratio_values))
    return {
        "minimum_gap": minimum_gap,
        "minimum_endpoint_clearance": minimum_endpoint_clearance,
        "maximum_count": maximum_count,
        "d0_min": min(d0_values),
        "d0_max": max(d0_values),
        "ratio_median": float(np.median(ratio_values)),
        "ratio_p90": float(np.percentile(ratio_values, 90)),
        "ratio_max": float(ratio_values[worst]),
        "worst_d0": d0_values[worst],
        "worst_route": routes[worst],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.layouts < 1:
        parser.error("--layouts must be at least one")

    started = time.time()
    rng = np.random.default_rng(args.seed)
    failures: list[str] = []
    for level in LEVELS:
        budget, radius_min, radius_max = geometry.ICEBERG_TIERS[level]
        try:
            stats = _audit_level(level, args.layouts, rng)
            print(
                f"level {level}: layouts={args.layouts} count<={budget} "
                f"r={radius_min:.1f}..{radius_max:.1f}m "
                f"blocked={args.layouts}/{args.layouts} "
                f"routes={args.layouts}/{args.layouts} "
                f"min_gap={stats['minimum_gap']:.3f}m "
                f"endpoint_clear={stats['minimum_endpoint_clearance']:.3f}m "
                f"d0={stats['d0_min']:.2f}..{stats['d0_max']:.2f}m "
                f"ratio_med={stats['ratio_median']:.3f} "
                f"ratio_p90={stats['ratio_p90']:.3f} "
                f"worst={stats['worst_d0']:.2f}->{stats['worst_route']:.2f}m "
                f"({stats['ratio_max']:.3f}x) "
                f"max_cylinders={stats['maximum_count']} PASS"
            )
        except Exception as error:  # preserve one summary line for every tier
            failures.append(f"level {level}: {error}")
            print(
                f"level {level}: layouts={args.layouts} count<={budget} "
                f"r={radius_min:.1f}..{radius_max:.1f}m FAIL {error}"
            )

    elapsed = time.time() - started
    if failures:
        print(f"FAIL: {len(failures)} level(s) failed ({elapsed:.1f} s)")
        raise SystemExit(1)
    print(
        f"PASS: {args.layouts * len(LEVELS)} iceberg layouts, "
        f"0 violations ({elapsed:.1f} s)"
    )


if __name__ == "__main__":
    main()
