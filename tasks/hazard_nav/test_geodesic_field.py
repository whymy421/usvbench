"""CPU-only regression audit for the band-fortress geodesic field."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np


GEOMETRY_PATH = Path(__file__).with_name("hazard_geometry.py")
SPEC = importlib.util.spec_from_file_location("geodesic_field_geometry", GEOMETRY_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load {GEOMETRY_PATH}")
sys.dont_write_bytecode = True
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


LEVELS = (0, 3)
FIELD_PADDING_M = 1.0


def _field_bounds(layout: object) -> tuple[float, float, float, float]:
    inflated = layout.radii + geometry.HALF_BEAM_M
    return (
        min(layout.start[0], layout.goal[0], np.min(layout.centers[:, 0] - inflated))
        - FIELD_PADDING_M,
        max(layout.start[0], layout.goal[0], np.max(layout.centers[:, 0] + inflated))
        + FIELD_PADDING_M,
        min(layout.start[1], layout.goal[1], np.min(layout.centers[:, 1] - inflated))
        - FIELD_PADDING_M,
        max(layout.start[1], layout.goal[1], np.max(layout.centers[:, 1] + inflated))
        + FIELD_PADDING_M,
    )


def _snapped_origin(bounds: tuple[float, float, float, float]) -> np.ndarray:
    return geometry.GRID_CELL_M * np.floor(
        np.asarray((bounds[0], bounds[2])) / geometry.GRID_CELL_M
    )


def _nearest_index(
    point: np.ndarray, origin: np.ndarray, shape: tuple[int, int]
) -> tuple[int, int]:
    col, row = np.rint((point - origin) / geometry.GRID_CELL_M).astype(int)
    return int(np.clip(row, 0, shape[0] - 1)), int(np.clip(col, 0, shape[1] - 1))


def _bilinear_value(field: np.ndarray, point: np.ndarray, origin: np.ndarray) -> float:
    col_f, row_f = (point - origin) / geometry.GRID_CELL_M
    col0, row0 = int(math.floor(col_f)), int(math.floor(row_f))
    col1, row1 = min(col0 + 1, field.shape[1] - 1), min(
        row0 + 1, field.shape[0] - 1
    )
    col_t, row_t = col_f - col0, row_f - row0
    return float(
        (1.0 - row_t) * (1.0 - col_t) * field[row0, col0]
        + (1.0 - row_t) * col_t * field[row0, col1]
        + row_t * (1.0 - col_t) * field[row1, col0]
        + row_t * col_t * field[row1, col1]
    )


def _greedy_reaches_goal(
    field: np.ndarray,
    start_idx: tuple[int, int],
    goal_idx: tuple[int, int],
) -> int:
    current = start_idx
    maximum_steps = field.size
    for steps in range(maximum_steps + 1):
        if current == goal_idx:
            return steps
        row, col = current
        candidates: list[tuple[float, int, int]] = []
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                next_row, next_col = row + d_row, col + d_col
                if 0 <= next_row < field.shape[0] and 0 <= next_col < field.shape[1]:
                    candidates.append(
                        (float(field[next_row, next_col]), next_row, next_col)
                    )
        next_distance, next_row, next_col = min(candidates)
        if not next_distance < float(field[row, col]):
            raise AssertionError(
                f"local minimum at {current}: {field[row, col]} -> {next_distance}"
            )
        current = (next_row, next_col)
    raise AssertionError(f"greedy walk exceeded {maximum_steps} steps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.layouts < 1:
        parser.error("--layouts must be at least one")

    rng = np.random.default_rng(args.seed)
    failures: list[str] = []
    total_started = time.perf_counter()
    total_fields = 0
    for level in LEVELS:
        build_ms: list[float] = []
        worst_route_error = 0.0
        longest_walk = 0
        for index in range(args.layouts):
            try:
                layout = geometry.sample_band_fortress_layout(
                    level, rng=rng, max_attempts=80
                )
                bounds = _field_bounds(layout)
                started = time.perf_counter()
                field = geometry.geodesic_distance_field(
                    layout.centers,
                    layout.radii,
                    layout.goal,
                    bounds,
                    cell_m=geometry.GRID_CELL_M,
                    half_beam_m=geometry.HALF_BEAM_M,
                )
                build_ms.append(1000.0 * (time.perf_counter() - started))
                assert field.dtype == np.float32

                origin = _snapped_origin(bounds)
                start_idx = _nearest_index(layout.start, origin, field.shape)
                goal_idx = _nearest_index(layout.goal, origin, field.shape)
                route = geometry.route_geodesic_length(
                    layout.start,
                    layout.centers,
                    layout.radii,
                    layout.goal,
                    bounds,
                    cell_m=geometry.GRID_CELL_M,
                    half_beam_m=geometry.HALF_BEAM_M,
                )
                assert route is not None
                spawn_distance = _bilinear_value(field, layout.start, origin)
                route_error = abs(spawn_distance - route)
                # Bilinear interpolation spans the four surrounding cells,
                # while A* attaches the exact endpoint to its nearest cell.
                # One cell diagonal (sqrt(2)*0.5 m) bounds that discretization.
                route_tolerance = math.sqrt(2.0) * geometry.GRID_CELL_M
                assert route_error <= route_tolerance + 1.0e-6, route_error
                worst_route_error = max(worst_route_error, route_error)
                assert abs(float(field[goal_idx])) <= 1.0e-6, field[goal_idx]

                longest_walk = max(
                    longest_walk, _greedy_reaches_goal(field, start_idx, goal_idx)
                )

                outside_xy = origin + geometry.GRID_CELL_M * np.asarray(
                    (start_idx[1], start_idx[0]), dtype=np.float64
                )
                outer_extent = float(
                    np.max(np.linalg.norm(layout.centers, axis=1) + layout.radii)
                )
                assert np.linalg.norm(outside_xy) > outer_extent
                euclidean = float(np.linalg.norm(outside_xy - layout.goal))
                assert float(field[start_idx]) + 1.0e-5 >= euclidean, (
                    field[start_idx],
                    euclidean,
                )
                total_fields += 1
            except Exception as error:
                failures.append(f"level {level} layout {index}: {error}")

        if build_ms:
            print(
                f"level {level}: layouts={args.layouts} "
                f"route_error_max={worst_route_error:.3f}m "
                f"greedy_steps_max={longest_walk} "
                f"field_build_mean={np.mean(build_ms):.2f}ms "
                f"field_build_max={np.max(build_ms):.2f}ms "
                f"{'PASS' if not any(item.startswith(f'level {level} ') for item in failures) else 'FAIL'}"
            )

    elapsed = time.perf_counter() - total_started
    if failures:
        for failure in failures[:10]:
            print(f"FAIL: {failure}")
        if len(failures) > 10:
            print(f"FAIL: ... {len(failures) - 10} additional failure(s)")
        print(f"FAIL: {len(failures)}/{args.layouts * len(LEVELS)} layouts ({elapsed:.2f}s)")
        raise SystemExit(1)
    print(
        f"PASS: {total_fields} geodesic fields across levels {LEVELS}; "
        f"goal=0, router agreement<=one cell diagonal, greedy descent reached goal, "
        f"outside>=Euclidean ({elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
