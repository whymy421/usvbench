"""Isaac-free geometry and procedural layout oracle for Hazard Navigation.

Only the Python standard library, NumPy, and Torch are imported here so layout
generation and all analytic obstacle math can be tested without Isaac Lab.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import heapq
import math
import warnings

import numpy as np
import torch


HALF_BEAM_M = 0.45
COLLISION_MARGIN_M = 0.20
OBSTACLE_INFLATION_M = HALF_BEAM_M + COLLISION_MARGIN_M
HULL_BEAM_M = 0.899
ENDPOINT_CLEAR_RADIUS_M = 5.0
GRID_CELL_M = 0.5
MIN_OBSTACLE_RADIUS_M = 0.8  # v3: two-ray coverage rule at 36 rays
MAX_OBSTACLE_RADIUS_M = 2.0


@dataclass(frozen=True)
class Difficulty:
    """One curriculum rung: obstacle count and admitted inflated gap."""

    level: int
    obstacle_count: int
    bottleneck_beams: float

    @property
    def bottleneck_m(self) -> float:
        return self.bottleneck_beams * HULL_BEAM_M


DIFFICULTIES: dict[int, Difficulty] = {
    0: Difficulty(level=0, obstacle_count=4, bottleneck_beams=5.0),
    1: Difficulty(level=1, obstacle_count=8, bottleneck_beams=4.0),
    2: Difficulty(level=2, obstacle_count=12, bottleneck_beams=3.0),
    # Level 3 (user-requested ultimate tier): gaps exactly 2x hull beam --
    # threading with fenders-width margins. Curriculum cap must be raised to
    # reach it; also the per-episode randomized band used by Suite S.
    3: Difficulty(level=3, obstacle_count=14, bottleneck_beams=2.0),
}


@dataclass(frozen=True)
class HazardLayout:
    """Accepted local-frame layout; start is at zero and goal lies on +X."""

    level: int
    requested_obstacle_count: int
    start: np.ndarray
    goal: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    geodesic_length: float
    direct_blocked: bool
    attempts: int

    @property
    def obstacle_count(self) -> int:
        return int(self.radii.shape[0])

    @property
    def feasible(self) -> bool:
        return math.isfinite(self.geodesic_length)


def difficulty_for_level(level: int) -> Difficulty:
    """Return a validated curriculum difficulty."""
    try:
        return DIFFICULTIES[int(level)]
    except (KeyError, ValueError):
        valid = ", ".join(str(value) for value in sorted(DIFFICULTIES))
        raise ValueError(f"Unknown hazard level {level!r}; expected one of {valid}.") from None


def inflated_radii(
    radii: np.ndarray, inflation_m: float = OBSTACLE_INFLATION_M
) -> np.ndarray:
    """Return obstacle radii inflated for hull half-beam and safety margin."""
    return np.asarray(radii, dtype=np.float64) + float(inflation_m)


def direct_segment_blocked(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    inflation_m: float = OBSTACLE_INFLATION_M,
) -> bool:
    """Whether any inflated cylinder intersects the finite start-goal segment."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0:
        return False
    segment = goal - start
    segment_norm_sq = float(np.dot(segment, segment))
    if segment_norm_sq <= 0.0:
        raise ValueError("start and goal must be distinct")
    projections = np.clip(((centers - start) @ segment) / segment_norm_sq, 0.0, 1.0)
    closest = start + projections[:, None] * segment
    distances = np.linalg.norm(centers - closest, axis=1)
    return bool(np.any(distances <= inflated_radii(radii, inflation_m)))


def start_goal_disks_clear(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    clear_radius_m: float = ENDPOINT_CLEAR_RADIUS_M,
    inflation_m: float = OBSTACLE_INFLATION_M,
) -> bool:
    """Check that inflated obstacles do not enter either protected endpoint disk."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0:
        return True
    required = inflated_radii(radii, inflation_m) + float(clear_radius_m)
    start_distance = np.linalg.norm(centers - np.asarray(start), axis=1)
    goal_distance = np.linalg.norm(centers - np.asarray(goal), axis=1)
    return bool(np.all(start_distance >= required) and np.all(goal_distance >= required))


def minimum_pairwise_inflated_gap(
    centers: np.ndarray,
    radii: np.ndarray,
    inflation_m: float = OBSTACLE_INFLATION_M,
) -> float:
    """Minimum free gap between inflated cylinders, or infinity for fewer than two."""
    centers = np.asarray(centers, dtype=np.float64)
    radii_i = inflated_radii(radii, inflation_m)
    if len(radii_i) < 2:
        return math.inf
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    gap = distance - radii_i[:, None] - radii_i[None, :]
    gap[np.tril_indices(len(radii_i))] = math.inf
    return float(np.min(gap))


def _occupancy_grid(
    centers: np.ndarray,
    radii: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    inflation_m: float,
) -> np.ndarray:
    """Rasterize inflated cylinders at the supplied grid-cell centers."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii_i = inflated_radii(np.asarray(radii, dtype=np.float64), inflation_m)
    occupied = np.zeros((len(ys), len(xs)), dtype=np.bool_)
    if len(centers):
        xx, yy = np.meshgrid(xs, ys)
        dx = xx[..., None] - centers[:, 0]
        dy = yy[..., None] - centers[:, 1]
        occupied = np.any(dx * dx + dy * dy <= radii_i * radii_i, axis=-1)
    return occupied


def _dijkstra_distance_field(
    occupied: np.ndarray,
    goal_idx: tuple[int, int],
    cell_m: float,
) -> np.ndarray:
    """Return weighted eight-connected distances from one free goal cell."""
    if occupied[goal_idx]:
        return np.full(occupied.shape, np.inf, dtype=np.float32)

    cardinal_cost = float(cell_m)
    diagonal_cost = math.sqrt(2.0) * cardinal_cost
    height, width = occupied.shape
    stride = width + 2
    padded_occupied = np.ones((height + 2, width + 2), dtype=np.bool_)
    padded_occupied[1:-1, 1:-1] = occupied
    occupied_flat = padded_occupied.ravel()
    neighbours = (
        (1, cardinal_cost),
        (-1, cardinal_cost),
        (stride, cardinal_cost),
        (-stride, cardinal_cost),
        (stride + 1, diagonal_cost),
        (stride - 1, diagonal_cost),
        (-stride + 1, diagonal_cost),
        (-stride - 1, diagonal_cost),
    )
    # Keep heap and array costs at the same precision. Storing heap candidates
    # into float32 can round upward and repeatedly re-enqueue the same cell.
    distances = np.full(padded_occupied.size, np.inf, dtype=np.float64)
    goal_node = (goal_idx[0] + 1) * stride + goal_idx[1] + 1
    distances[goal_node] = 0.0
    queue: list[tuple[float, int]] = [(0.0, goal_node)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for offset, move_cost in neighbours:
            next_node = node + offset
            if occupied_flat[next_node]:
                continue
            candidate = distance + move_cost
            if candidate < distances[next_node]:
                distances[next_node] = candidate
                heapq.heappush(queue, (candidate, next_node))
    return distances.reshape(height + 2, width + 2)[1:-1, 1:-1].astype(
        np.float32
    )


def _bounded_grid_axes(
    bounds_m: float | tuple[float, float, float, float], cell_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return snapped cell-center axes for scalar or explicit arena bounds."""
    if np.isscalar(bounds_m):
        extent = float(bounds_m)
        if extent <= 0.0:
            raise ValueError("scalar bounds_m must be positive")
        x_min = y_min = -extent
        x_max = y_max = extent
    else:
        if len(bounds_m) != 4:
            raise ValueError(
                "bounds_m must be a scalar or (x_min, x_max, y_min, y_max)"
            )
        x_min, x_max, y_min, y_max = (float(value) for value in bounds_m)
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("bounds_m minima must be smaller than maxima")

    x_min = math.floor(x_min / cell_m) * cell_m
    x_max = math.ceil(x_max / cell_m) * cell_m
    y_min = math.floor(y_min / cell_m) * cell_m
    y_max = math.ceil(y_max / cell_m) * cell_m
    xs = np.arange(x_min, x_max + 0.5 * cell_m, cell_m, dtype=np.float64)
    ys = np.arange(y_min, y_max + 0.5 * cell_m, cell_m, dtype=np.float64)
    return xs, ys


def geodesic_distance_field(
    centers: np.ndarray,
    radii: np.ndarray,
    goal_xy: np.ndarray,
    bounds_m: float | tuple[float, float, float, float],
    cell_m: float = GRID_CELL_M,
    half_beam_m: float = HALF_BEAM_M,
) -> np.ndarray:
    """Build a float32 geodesic distance-to-goal field over an arena grid.

    A scalar ``bounds_m`` denotes the square ``[-bounds_m, bounds_m]^2``.
    Four values denote ``(x_min, x_max, y_min, y_max)``. Bounds are snapped
    outward to the cell lattice, cells are sampled at those lattice points,
    and cells inside half-beam-inflated cylinders remain ``+inf``.
    """
    if cell_m <= 0.0 or half_beam_m < 0.0:
        raise ValueError("cell_m must be positive and half_beam_m non-negative")
    xs, ys = _bounded_grid_axes(bounds_m, cell_m)
    occupied = _occupancy_grid(centers, radii, xs, ys, half_beam_m)

    goal = np.asarray(goal_xy, dtype=np.float64)
    goal_col = int(
        np.clip(round((float(goal[0]) - xs[0]) / cell_m), 0, len(xs) - 1)
    )
    goal_row = int(
        np.clip(round((float(goal[1]) - ys[0]) / cell_m), 0, len(ys) - 1)
    )
    return _dijkstra_distance_field(occupied, (goal_row, goal_col), cell_m)


def geodesic_descent_directions(field: np.ndarray) -> np.ndarray:
    """Return the lowest-neighbour descent hop for every field cell.

    Directions are ``(d_row, d_col)`` int8 pairs. A zero pair marks the goal
    or a cell with no strictly lower finite neighbour. Computing this once at
    reset keeps waypoint extraction to indexed lookups during control steps.
    """
    field = np.asarray(field)
    if field.ndim != 2 or not np.issubdtype(field.dtype, np.floating):
        raise ValueError("field must be a two-dimensional floating array")

    height, width = field.shape
    offsets = np.asarray(
        (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ),
        dtype=np.int64,
    )
    padded = np.pad(field, 1, constant_values=np.inf)
    neighbours = np.stack(
        [
            padded[
                1 + d_row : 1 + d_row + height,
                1 + d_col : 1 + d_col + width,
            ]
            for d_row, d_col in offsets
        ]
    )
    choice = np.argmin(neighbours, axis=0)
    best = np.take_along_axis(neighbours, choice[None], axis=0)[0]
    directions = np.zeros((*field.shape, 2), dtype=np.int8)
    descending = best < field
    directions[descending] = offsets[choice[descending]]
    return directions


def geodesic_field_value(
    field: np.ndarray,
    point_xy: np.ndarray,
    origin_xy: np.ndarray,
    cell_m: float = GRID_CELL_M,
) -> float:
    """Bilinearly sample a field without evaluating ``0 * inf`` corners."""
    if cell_m <= 0.0:
        raise ValueError("cell_m must be positive")
    col_f, row_f = (
        np.asarray(point_xy, dtype=np.float64)
        - np.asarray(origin_xy, dtype=np.float64)
    ) / cell_m
    col_f = float(np.clip(col_f, 0.0, field.shape[1] - 1))
    row_f = float(np.clip(row_f, 0.0, field.shape[0] - 1))
    col0, row0 = int(math.floor(col_f)), int(math.floor(row_f))
    col1, row1 = min(col0 + 1, field.shape[1] - 1), min(
        row0 + 1, field.shape[0] - 1
    )
    col_t, row_t = col_f - col0, row_f - row0
    sampled = 0.0
    for row, col, weight in (
        (row0, col0, (1.0 - row_t) * (1.0 - col_t)),
        (row0, col1, (1.0 - row_t) * col_t),
        (row1, col0, row_t * (1.0 - col_t)),
        (row1, col1, row_t * col_t),
    ):
        if weight > 0.0:
            sampled += weight * float(field[row, col])
    return sampled


def geodesic_waypoint(
    field: np.ndarray,
    descent_directions: np.ndarray,
    point_xy: np.ndarray,
    goal_xy: np.ndarray,
    origin_xy: np.ndarray,
    lookahead_m: float,
    cell_m: float = GRID_CELL_M,
) -> np.ndarray:
    """Extract a lookahead waypoint by hopping down a solved field."""
    if lookahead_m <= 0.0 or cell_m <= 0.0:
        raise ValueError("lookahead_m and cell_m must be positive")
    if descent_directions.shape != (*field.shape, 2):
        raise ValueError("descent_directions shape must be (*field.shape, 2)")

    point = np.asarray(point_xy, dtype=np.float64)
    goal = np.asarray(goal_xy, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)
    if geodesic_field_value(field, point, origin, cell_m) < lookahead_m:
        return goal.copy()

    col, row = np.rint((point - origin) / cell_m).astype(np.int64)
    row = int(np.clip(row, 0, field.shape[0] - 1))
    col = int(np.clip(col, 0, field.shape[1] - 1))
    waypoint = origin + cell_m * np.asarray((col, row), dtype=np.float64)
    remaining = float(lookahead_m)

    # Attach the exact continuous position to its nearest route cell first.
    attachment = float(np.linalg.norm(waypoint - point))
    if attachment >= remaining:
        return waypoint
    remaining -= attachment

    max_hops = int(math.ceil(lookahead_m / cell_m)) + 1
    for _ in range(max_hops):
        d_row, d_col = (
            int(value) for value in descent_directions[row, col]
        )
        hop_m = cell_m * math.hypot(d_row, d_col)
        if hop_m == 0.0:
            # This is unreachable for a valid finite non-goal Dijkstra cell.
            # The exact goal is the only safe, deterministic recovery target.
            return goal.copy()
        waypoint += cell_m * np.asarray((d_col, d_row))
        remaining -= hop_m
        if remaining <= 1.0e-9:
            return waypoint
        row += d_row
        col += d_col
    return waypoint


def route_geodesic_length(
    start_xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    goal_xy: np.ndarray,
    bounds_m: float | tuple[float, float, float, float],
    cell_m: float = GRID_CELL_M,
    half_beam_m: float = HALF_BEAM_M,
) -> float | None:
    """Independently route start-to-goal with weighted eight-connected A*."""
    if cell_m <= 0.0 or half_beam_m < 0.0:
        raise ValueError("cell_m must be positive and half_beam_m non-negative")
    xs, ys = _bounded_grid_axes(bounds_m, cell_m)
    occupied = _occupancy_grid(centers, radii, xs, ys, half_beam_m)
    start = np.asarray(start_xy, dtype=np.float64)
    goal = np.asarray(goal_xy, dtype=np.float64)

    def nearest_index(point: np.ndarray) -> tuple[int, int]:
        col = int(np.clip(round((float(point[0]) - xs[0]) / cell_m), 0, len(xs) - 1))
        row = int(np.clip(round((float(point[1]) - ys[0]) / cell_m), 0, len(ys) - 1))
        return row, col

    start_idx, goal_idx = nearest_index(start), nearest_index(goal)
    if occupied[start_idx] or occupied[goal_idx]:
        return None

    cardinal_cost = float(cell_m)
    diagonal_cost = math.sqrt(2.0) * cardinal_cost
    neighbours = (
        (0, 1, cardinal_cost),
        (0, -1, cardinal_cost),
        (1, 0, cardinal_cost),
        (-1, 0, cardinal_cost),
        (1, 1, diagonal_cost),
        (1, -1, diagonal_cost),
        (-1, 1, diagonal_cost),
        (-1, -1, diagonal_cost),
    )

    def heuristic(row: int, col: int) -> float:
        d_row = abs(goal_idx[0] - row)
        d_col = abs(goal_idx[1] - col)
        diagonal_steps = min(d_row, d_col)
        return diagonal_steps * diagonal_cost + abs(d_row - d_col) * cardinal_cost

    best = np.full(occupied.shape, np.inf, dtype=np.float64)
    best[start_idx] = 0.0
    queue: list[tuple[float, float, int, int]] = [
        (heuristic(*start_idx), 0.0, start_idx[0], start_idx[1])
    ]
    height, width = occupied.shape
    route_length: float | None = None
    while queue:
        _estimate, distance, row, col = heapq.heappop(queue)
        if distance > best[row, col]:
            continue
        if (row, col) == goal_idx:
            route_length = distance
            break
        for d_row, d_col, move_cost in neighbours:
            next_row, next_col = row + d_row, col + d_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if occupied[next_row, next_col]:
                continue
            candidate = distance + move_cost
            if candidate < best[next_row, next_col]:
                best[next_row, next_col] = candidate
                heapq.heappush(
                    queue,
                    (
                        candidate + heuristic(next_row, next_col),
                        candidate,
                        next_row,
                        next_col,
                    ),
                )
    if route_length is None:
        return None
    start_cell = np.array((xs[start_idx[1]], ys[start_idx[0]]))
    goal_cell = np.array((xs[goal_idx[1]], ys[goal_idx[0]]))
    return (
        route_length
        + float(np.linalg.norm(start - start_cell))
        + float(np.linalg.norm(goal - goal_cell))
    )


def bfs_geodesic_length(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    cell_m: float = GRID_CELL_M,
    inflation_m: float = OBSTACLE_INFLATION_M,
    padding_m: float = 6.0,
) -> float | None:
    """Run an eight-connected occupancy-grid BFS and return its path length.

    Occupancy is evaluated at 0.5 m cell centers against cylinders inflated by
    half-beam plus margin. BFS has no external planner dependency. The returned
    polyline length includes the final sub-cell connection to the exact goal.
    """
    if cell_m <= 0.0 or padding_m <= 0.0:
        raise ValueError("cell_m and padding_m must be positive")

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii_i = inflated_radii(np.asarray(radii, dtype=np.float64), inflation_m)

    extent_points = np.vstack((start, goal, centers)) if len(centers) else np.vstack((start, goal))
    if len(centers):
        x_min = min(float(start[0]), float(goal[0]), float(np.min(centers[:, 0] - radii_i)))
        x_max = max(float(start[0]), float(goal[0]), float(np.max(centers[:, 0] + radii_i)))
        y_min = min(float(start[1]), float(goal[1]), float(np.min(centers[:, 1] - radii_i)))
        y_max = max(float(start[1]), float(goal[1]), float(np.max(centers[:, 1] + radii_i)))
    else:
        x_min, y_min = np.min(extent_points, axis=0)
        x_max, y_max = np.max(extent_points, axis=0)

    x_min = math.floor((x_min - padding_m) / cell_m) * cell_m
    x_max = math.ceil((x_max + padding_m) / cell_m) * cell_m
    y_min = math.floor((y_min - padding_m) / cell_m) * cell_m
    y_max = math.ceil((y_max + padding_m) / cell_m) * cell_m
    xs = np.arange(x_min, x_max + 0.5 * cell_m, cell_m)
    ys = np.arange(y_min, y_max + 0.5 * cell_m, cell_m)

    occupied = _occupancy_grid(centers, radii, xs, ys, inflation_m)

    def nearest_index(point: np.ndarray) -> tuple[int, int]:
        col = int(np.clip(round((float(point[0]) - x_min) / cell_m), 0, len(xs) - 1))
        row = int(np.clip(round((float(point[1]) - y_min) / cell_m), 0, len(ys) - 1))
        return row, col

    start_idx = nearest_index(start)
    goal_idx = nearest_index(goal)
    if occupied[start_idx] or occupied[goal_idx]:
        return None

    parent_row = np.full(occupied.shape, -1, dtype=np.int32)
    parent_col = np.full(occupied.shape, -1, dtype=np.int32)
    visited = np.zeros(occupied.shape, dtype=np.bool_)
    visited[start_idx] = True
    queue: deque[tuple[int, int]] = deque([start_idx])
    neighbours = (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    while queue:
        row, col = queue.popleft()
        if (row, col) == goal_idx:
            break
        for d_row, d_col in neighbours:
            next_row = row + d_row
            next_col = col + d_col
            if not (0 <= next_row < len(ys) and 0 <= next_col < len(xs)):
                continue
            if occupied[next_row, next_col] or visited[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            parent_row[next_row, next_col] = row
            parent_col[next_row, next_col] = col
            queue.append((next_row, next_col))

    if not visited[goal_idx]:
        return None

    route_length = 0.0
    row, col = goal_idx
    while (row, col) != start_idx:
        previous_row = int(parent_row[row, col])
        previous_col = int(parent_col[row, col])
        route_length += cell_m * math.hypot(row - previous_row, col - previous_col)
        row, col = previous_row, previous_col

    start_cell = np.array((xs[start_idx[1]], ys[start_idx[0]]))
    goal_cell = np.array((xs[goal_idx[1]], ys[goal_idx[0]]))
    route_length += float(np.linalg.norm(start - start_cell))
    route_length += float(np.linalg.norm(goal - goal_cell))
    return route_length


def _sample_candidate(
    difficulty: Difficulty,
    obstacle_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Sample one geometrically separated candidate before the BFS check."""
    start = np.zeros(2, dtype=np.float64)
    distance = float(rng.uniform(20.0, 40.0))
    goal = np.array((distance, 0.0), dtype=np.float64)
    radii = rng.uniform(
        MIN_OBSTACLE_RADIUS_M,
        MAX_OBSTACLE_RADIUS_M,
        size=obstacle_count,
    )
    radii_i = inflated_radii(radii)
    centers = np.empty((obstacle_count, 2), dtype=np.float64)

    # Make direct obstruction the default accepted-map mode. Endpoint-disk
    # admission can shorten the nominal 30--70% interval for large cylinders.
    endpoint_buffer = ENDPOINT_CLEAR_RADIUS_M + radii_i[0]
    blocker_low = max(0.30 * distance, endpoint_buffer)
    blocker_high = min(0.70 * distance, distance - endpoint_buffer)
    if blocker_low > blocker_high:
        return None
    centers[0] = (rng.uniform(blocker_low, blocker_high), 0.0)

    # The corridor is the strip whose longitudinal projection lies between the
    # endpoints. A generous lateral width makes the level-2 packing constraint
    # feasible while the forced blocker still demands a non-straight route.
    corridor_half_width = max(12.0, 0.40 * distance)
    for index in range(1, obstacle_count):
        admitted = False
        required_endpoint_distance = ENDPOINT_CLEAR_RADIUS_M + radii_i[index]
        for _ in range(3000):
            candidate = np.array(
                (
                    rng.uniform(0.0, distance),
                    rng.uniform(-corridor_half_width, corridor_half_width),
                )
            )
            if np.linalg.norm(candidate - start) < required_endpoint_distance:
                continue
            if np.linalg.norm(candidate - goal) < required_endpoint_distance:
                continue
            separation = np.linalg.norm(centers[:index] - candidate, axis=1)
            required = radii_i[:index] + radii_i[index] + difficulty.bottleneck_m
            if np.any(separation < required):
                continue
            centers[index] = candidate
            admitted = True
            break
        if not admitted:
            return None

    return start, goal, centers, radii


def sample_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 20,
) -> HazardLayout:
    """Rejection-sample one feasible layout, falling back to fewer obstacles.

    Every requested obstacle count receives at most ``max_attempts`` complete
    layout attempts. Failure emits a warning before trying one fewer cylinder.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    rng = np.random.default_rng() if rng is None else rng
    total_attempts = 0

    for obstacle_count in range(difficulty.obstacle_count, 0, -1):
        for _ in range(max_attempts):
            total_attempts += 1
            sampled = _sample_candidate(difficulty, obstacle_count, rng)
            if sampled is None:
                continue
            start, goal, centers, radii = sampled
            if not start_goal_disks_clear(start, goal, centers, radii):
                continue
            if minimum_pairwise_inflated_gap(centers, radii) + 1.0e-9 < difficulty.bottleneck_m:
                continue
            geodesic = bfs_geodesic_length(start, goal, centers, radii)
            if geodesic is None:
                continue
            blocked = direct_segment_blocked(start, goal, centers, radii)
            return HazardLayout(
                level=difficulty.level,
                requested_obstacle_count=difficulty.obstacle_count,
                start=start,
                goal=goal,
                centers=centers,
                radii=radii,
                geodesic_length=float(geodesic),
                direct_blocked=blocked,
                attempts=total_attempts,
            )

        if obstacle_count > 1:
            warnings.warn(
                f"Hazard layout level {level}: no feasible {obstacle_count}-obstacle "
                f"map in {max_attempts} attempts; falling back to {obstacle_count - 1}.",
                RuntimeWarning,
                stacklevel=2,
            )

    raise RuntimeError(
        f"Could not generate even a one-obstacle feasible layout for level {level}."
    )


# --- Iceberg variant: a few huge obstacles that demand an early detour -------
# Kept separate from DIFFICULTIES for the same reason forced gates have their
# own table: these tiers vary physical cylinder count/radius, not gap width.
ICEBERG_TIERS: dict[int, tuple[int, float, float]] = {
    0: (1, 5.0, 8.0),
    1: (2, 6.0, 10.0),
    2: (3, 7.0, 12.0),
    3: (3, 9.0, 14.0),
}
ICEBERG_EDGE_GAP_M = 8.0
ICEBERG_GOAL_DISTANCE_RANGE_M = (28.0, 40.0)


def sample_iceberg_layout(
    level: int,
    rng: np.random.Generator | None = None,
    max_attempts: int = 60,
) -> HazardLayout:
    """Sample open water where huge cylinders force a long-range detour.

    The local-frame start/goal and protected endpoint disks deliberately use
    the scatter task's conventions. The first cylinder center lies exactly on
    the direct segment; later cylinders alternate sides and extend the field
    laterally, while leaving at least eight metres between physical edges.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    try:
        count, radius_min, radius_max = ICEBERG_TIERS[int(level)]
    except (KeyError, ValueError):
        valid = ", ".join(str(value) for value in sorted(ICEBERG_TIERS))
        raise ValueError(
            f"Unknown iceberg level {level!r}; expected one of {valid}."
        ) from None

    rng = np.random.default_rng() if rng is None else rng
    start = np.zeros(2, dtype=np.float64)
    endpoint_inflation = ENDPOINT_CLEAR_RADIUS_M + OBSTACLE_INFLATION_M

    for attempt in range(1, max_attempts + 1):
        distance = float(rng.uniform(*ICEBERG_GOAL_DISTANCE_RANGE_M))
        goal = np.array((distance, 0.0), dtype=np.float64)
        radii = rng.uniform(radius_min, radius_max, size=count)
        centers = np.empty((count, 2), dtype=np.float64)

        # Exactly-on-segment placement makes direct blockage constructive, not
        # probabilistic. The standard scatter endpoint disks remain inviolate.
        blocker_low = float(radii[0] + endpoint_inflation)
        blocker_high = float(distance - radii[0] - endpoint_inflation)
        if blocker_low > blocker_high:
            continue
        middle_low = max(blocker_low, 0.38 * distance)
        middle_high = min(blocker_high, 0.62 * distance)
        if middle_low > middle_high:
            continue
        centers[0] = (rng.uniform(middle_low, middle_high), 0.0)

        first_side = -1.0 if rng.random() < 0.5 else 1.0
        admitted = True
        for index in range(1, count):
            side = first_side if index % 2 else -first_side
            required_endpoint = float(
                radii[index] + ENDPOINT_CLEAR_RADIUS_M + OBSTACLE_INFLATION_M
            )
            placed = False
            for _ in range(600):
                # Keep the lateral iceberg near enough to form one leg of the
                # obstacle field, but never turn its gap into a bottleneck.
                x = float(rng.uniform(0.25 * distance, 0.75 * distance))
                required_from_blocker = float(
                    radii[0] + radii[index] + ICEBERG_EDGE_GAP_M
                )
                dx = x - float(centers[0, 0])
                minimum_abs_y = math.sqrt(
                    max(0.0, required_from_blocker**2 - dx**2)
                )
                abs_y = float(
                    rng.uniform(minimum_abs_y, minimum_abs_y + 4.0)
                )
                candidate = np.array((x, side * abs_y), dtype=np.float64)

                if np.linalg.norm(candidate - start) < required_endpoint:
                    continue
                if np.linalg.norm(candidate - goal) < required_endpoint:
                    continue
                separation = np.linalg.norm(centers[:index] - candidate, axis=1)
                required = radii[:index] + radii[index] + ICEBERG_EDGE_GAP_M
                if np.any(separation + 1.0e-9 < required):
                    continue
                centers[index] = candidate
                placed = True
                break
            if not placed:
                admitted = False
                break
        if not admitted:
            continue

        if not start_goal_disks_clear(start, goal, centers, radii):
            continue
        physical_gap = minimum_pairwise_inflated_gap(
            centers, radii, inflation_m=0.0
        )
        if physical_gap + 1.0e-9 < ICEBERG_EDGE_GAP_M:
            continue
        geodesic = bfs_geodesic_length(start, goal, centers, radii)
        if geodesic is None:
            continue
        if not direct_segment_blocked(start, goal, centers, radii):
            continue
        return HazardLayout(
            level=int(level),
            requested_obstacle_count=count,
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=float(geodesic),
            direct_blocked=True,
            attempts=attempt,
        )

    raise RuntimeError(
        f"Could not generate an iceberg layout for level {level} "
        f"in {max_attempts} attempts."
    )


# --- Ring-siege variant (user-requested): spawn encircled, exactly one
# passable gap whose inflated width follows the tier ladder. Escape requires
# threading from the very first second -- avoidance as a mandatory skill.
RING_RADIUS_M = 10.5  # >= ENDPOINT_CLEAR (5.0) + 2 * max inflated radius
RING_OBSTACLE_RADIUS_MIN_M = 1.5
RING_OBSTACLE_RADIUS_MAX_M = 2.0
RING_NEIGHBOR_OVERLAP_M = 0.30  # inflated neighbors overlap: sealed by construction
# Overlap needed for the physical surface gap to fall below the hull beam:
# 2*OBSTACLE_INFLATION_M - HULL_BEAM_M = 1.30 - 0.899 = 0.401. Use a margin.
RING_SEALED_OVERLAP_M = 0.55
RING_GAP_SLACK_M = 0.50  # accepted gap width band: [bottleneck, bottleneck + slack]
RING_GOAL_DISTANCE_RANGE_M = (24.0, 40.0)  # 24 keeps the goal disk clear of the ring
DOUBLE_RING_OUTER_RADIUS_M = 17.5
DOUBLE_RING_GOAL_DISTANCE_RANGE_M = (26.0, 40.0)
DOUBLE_RING_MIN_GAP_OFFSET_RAD = 0.5 * math.pi
FORTRESS_APERTURE_M: dict[int, float] = {0: 6.0, 1: 4.5, 2: 3.6, 3: 2.7}
FORTRESS_SPAWN_RADIUS_RANGE_M = (18.0, 24.0)
DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M = (25.0, 31.0)
# Hard returned-array worst cases. The sealed single-ring constructor can
# return at most 23 cylinders. The double-ring helper has a generic emergency
# cap of 63 per ring, so fortress2 explicitly rejects above 32 inner / 56 outer
# and can return at most 88 cylinders total.
FORTRESS_MAX_OBSTACLES = 23
DOUBLE_RING_FORTRESS_MAX_INNER_OBSTACLES = 32
DOUBLE_RING_FORTRESS_MAX_OUTER_OBSTACLES = 56
BAND_FORTRESS_MAX_OBSTACLES = 96
BAND_FORTRESS_INNER_RADIUS_M = 9.0
BAND_FORTRESS_SEPARATION_M = 6.0
BAND_FORTRESS_LAYER_RADIUS_RANGE_M = (0.8, 2.0)
BAND_FORTRESS_PLUG_RADIUS_RANGE_M = (1.6, 2.0)


def _ring_seal_check(
    start: np.ndarray,
    goal: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    gap_mid_xy: np.ndarray,
) -> bool:
    """Layout is a true siege iff plugging the gap makes the goal unreachable."""
    plug_radius = 0.5 * (RING_OBSTACLE_RADIUS_MIN_M + RING_OBSTACLE_RADIUS_MAX_M)
    plugged_centers = np.vstack([centers, gap_mid_xy[None, :]])
    plugged_radii = np.concatenate([radii, [plug_radius]])
    return bfs_geodesic_length(start, goal, plugged_centers, plugged_radii) is None


def sample_ring_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 40,
    neighbor_overlap_m: float | None = None,
    exact_gap_width_m: float | None = None,
) -> HazardLayout:
    """Rejection-sample a sealed ring around the spawn with one tier-width gap.

    Admission differs from the scatter sampler on purpose: ring neighbors
    intentionally violate the pairwise-bottleneck rule (they overlap after
    inflation -- that is what seals the ring), so admission is instead
    (a) exactly one passable gap with inflated width in
    [bottleneck, bottleneck + RING_GAP_SLACK_M], (b) BFS-feasible as built,
    (c) BFS-INFEASIBLE with the gap plugged, (d) endpoint disks clear.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    rng = np.random.default_rng() if rng is None else rng
    start = np.zeros(2, dtype=np.float64)

    if exact_gap_width_m is not None and exact_gap_width_m <= 0.0:
        raise ValueError("exact_gap_width_m must be positive")

    def chord_angle(chord: float) -> float:
        return 2.0 * math.asin(min(1.0, chord / (2.0 * RING_RADIUS_M)))

    # Neighbours are placed so their INFLATED disks overlap by this much, so
    # the gap between the PHYSICAL surfaces is 2*OBSTACLE_INFLATION_M minus the
    # overlap. At the shipped 0.30 that leaves 1.00 m -- wider than the 0.899 m
    # hull, so the "sealed" ring could be escaped anywhere. The seal check did
    # not catch it because its BFS inflates by 0.65 while the simulator's
    # contact test uses the true half-beam 0.45: two different hulls.
    # `neighbor_overlap_m` is the MINIMUM overlap, not the maximum: the closure
    # bisection returns some value inside [overlap_min, overlap_max], so
    # raising only the ceiling still admits loose rings. Sealing is a floor
    # condition -- surface gap = 2*OBSTACLE_INFLATION_M - overlap < hull beam.
    if neighbor_overlap_m is None:
        overlap_min = 0.05   # loosest packing that still overlaps after inflation
        overlap_max = RING_NEIGHBOR_OVERLAP_M
    else:
        overlap_min = float(neighbor_overlap_m)
        overlap_max = overlap_min + 0.20

    for attempt in range(1, max_attempts + 1):
        distance = float(rng.uniform(*RING_GOAL_DISTANCE_RANGE_M))
        goal = np.array((distance, 0.0), dtype=np.float64)
        gap_bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        gap_free = float(difficulty.bottleneck_m) + 0.5 * RING_GAP_SLACK_M

        # Grow the radii list until the circle is over-full at the TIGHTEST
        # packing; then the exact closure overlap is solved by bisection so
        # the ring closes with the gap at exactly the tier width. Closure is
        # guaranteed whenever total angle straddles 2*pi between the two
        # packing extremes -- no acceptance-window lottery.
        radii_list = [
            float(rng.uniform(RING_OBSTACLE_RADIUS_MIN_M, RING_OBSTACLE_RADIUS_MAX_M))
        ]

        def total_angle(overlap: float) -> float:
            infl = [r + OBSTACLE_INFLATION_M for r in radii_list]
            if exact_gap_width_m is None:
                gap_angle = chord_angle(gap_free + infl[-1] + infl[0])
            else:
                gap_angle = (
                    chord_angle(infl[-1])
                    + chord_angle(infl[0])
                    + exact_gap_width_m / RING_RADIUS_M
                )
            total = gap_angle
            for a, b in zip(infl[:-1], infl[1:]):
                total += chord_angle(a + b - overlap)
            return total

        # total_angle DECREASES in overlap. Grow until even the TIGHTEST
        # packing overfills the circle, then drop the last obstacle: K-1
        # obstacles satisfy total(overlap_max) < 2*pi; solvable iff their
        # loosest packing can still fill it (total(overlap_min) >= 2*pi).
        while total_angle(overlap_max) < 2.0 * math.pi and len(radii_list) <= 24:
            radii_list.append(
                float(rng.uniform(RING_OBSTACLE_RADIUS_MIN_M, RING_OBSTACLE_RADIUS_MAX_M))
            )
        if len(radii_list) > 24:
            continue
        radii_list.pop()
        if len(radii_list) < 3 or total_angle(overlap_min) < 2.0 * math.pi:
            continue  # this radii draw cannot straddle 2*pi; resample

        lo, hi = overlap_min, overlap_max  # total(lo) >= 2pi > total(hi)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if total_angle(mid) >= 2.0 * math.pi:
                lo = mid
            else:
                hi = mid
        overlap = lo

        infl = [r + OBSTACLE_INFLATION_M for r in radii_list]
        if exact_gap_width_m is None:
            gap_angle = chord_angle(gap_free + infl[-1] + infl[0])
        else:
            gap_angle = (
                chord_angle(infl[-1])
                + chord_angle(infl[0])
                + exact_gap_width_m / RING_RADIUS_M
            )
        angles = [gap_bearing + 0.5 * gap_angle]
        for a, b in zip(infl[:-1], infl[1:]):
            angles.append(angles[-1] + chord_angle(a + b - overlap))

        centers = np.stack(
            [
                RING_RADIUS_M * np.cos(np.asarray(angles)),
                RING_RADIUS_M * np.sin(np.asarray(angles)),
            ],
            axis=1,
        )
        radii = np.asarray(radii_list, dtype=np.float64)
        if not start_goal_disks_clear(start, goal, centers, radii):
            continue
        geodesic = bfs_geodesic_length(start, goal, centers, radii)
        if geodesic is None:
            continue
        gap_mid_angle = angles[0] - 0.5 * (2.0 * math.pi - (angles[-1] - angles[0]))
        gap_mid = np.array(
            (RING_RADIUS_M * math.cos(gap_mid_angle), RING_RADIUS_M * math.sin(gap_mid_angle))
        )
        if not _ring_seal_check(start, goal, centers, radii, gap_mid):
            continue
        return HazardLayout(
            level=difficulty.level,
            requested_obstacle_count=len(radii_list),
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=float(geodesic),
            direct_blocked=direct_segment_blocked(start, goal, centers, radii),
            attempts=attempt,
        )

    raise RuntimeError(
        f"Could not generate a sealed ring layout for level {level} in {max_attempts} attempts."
    )


@dataclass(frozen=True)
class RingGapSpec:
    """One ring opening, recorded from the shipped arrays after scanning."""

    ring_radius_m: float
    center: np.ndarray
    bearing_rad: float
    free_width_m: float
    requested_width_m: float


@dataclass(frozen=True)
class DoubleRingGapSpecs:
    """Measured openings for the inner and outer walls of a double siege."""

    inner: RingGapSpec
    outer: RingGapSpec

    @property
    def angular_offset_rad(self) -> float:
        delta = (self.outer.bearing_rad - self.inner.bearing_rad) % (2.0 * math.pi)
        return min(delta, 2.0 * math.pi - delta)


def _ring_line_openings(
    centers: np.ndarray,
    radii: np.ndarray,
    ring_radius_m: float,
    *,
    inflation_m: float = OBSTACLE_INFLATION_M,
) -> list[tuple[float, float, int, int]]:
    """Scan free arcs on a ring line.

    Returns ``(arc_width_m, bearing_rad, before_index, after_index)`` for
    every uncovered run. Indices address the arrays passed in. This measures
    the constructed cylinders rather than trusting their requested bearings.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(radii, dtype=np.float64)
    if len(centers) == 0:
        return []
    bearings = np.mod(np.arctan2(centers[:, 1], centers[:, 0]), 2.0 * math.pi)
    order = np.argsort(bearings)
    bearings = bearings[order]
    inflated = radii[order] + float(inflation_m)
    half_arcs = 2.0 * np.arcsin(
        np.clip(inflated / (2.0 * float(ring_radius_m)), 0.0, 1.0)
    )

    openings: list[tuple[float, float, int, int]] = []
    for position in range(len(order)):
        next_position = (position + 1) % len(order)
        current_end = float(bearings[position] + half_arcs[position])
        next_start = float(bearings[next_position] - half_arcs[next_position])
        if next_position == 0:
            next_start += 2.0 * math.pi
        free_angle = next_start - current_end
        if free_angle > 1.0e-10:
            bearing = (current_end + 0.5 * free_angle) % (2.0 * math.pi)
            openings.append(
                (
                    float(ring_radius_m) * free_angle,
                    bearing,
                    int(order[position]),
                    int(order[next_position]),
                )
            )
    return openings


def _sample_sealed_ring(
    ring_radius_m: float,
    gap_width_m: float,
    gap_bearing_rad: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pack one ring with physical neighbour overlap and one planner gap."""
    overlap_min = RING_SEALED_OVERLAP_M
    overlap_max = overlap_min + 0.35
    radii_list = [
        float(rng.uniform(RING_OBSTACLE_RADIUS_MIN_M, RING_OBSTACLE_RADIUS_MAX_M))
    ]

    def chord_angle(chord_m: float) -> float:
        return 2.0 * math.asin(min(1.0, chord_m / (2.0 * ring_radius_m)))

    def occupied_half_angle(radius_m: float) -> float:
        inflated = radius_m + OBSTACLE_INFLATION_M
        return 2.0 * math.asin(min(1.0, inflated / (2.0 * ring_radius_m)))

    def total_angle(overlap_m: float) -> float:
        gap_angle = (
            occupied_half_angle(radii_list[-1])
            + occupied_half_angle(radii_list[0])
            + gap_width_m / ring_radius_m
        )
        wall_angles = sum(
            chord_angle(first + second - overlap_m)
            for first, second in zip(radii_list[:-1], radii_list[1:])
        )
        return gap_angle + wall_angles

    while total_angle(overlap_max) < 2.0 * math.pi and len(radii_list) <= 64:
        radii_list.append(
            float(rng.uniform(RING_OBSTACLE_RADIUS_MIN_M, RING_OBSTACLE_RADIUS_MAX_M))
        )
    if len(radii_list) > 64:
        return None
    radii_list.pop()
    if len(radii_list) < 3 or total_angle(overlap_min) < 2.0 * math.pi:
        return None

    lo, hi = overlap_min, overlap_max
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if total_angle(mid) >= 2.0 * math.pi:
            lo = mid
        else:
            hi = mid
    overlap = lo

    first_half = occupied_half_angle(radii_list[0])
    last_half = occupied_half_angle(radii_list[-1])
    gap_angle = last_half + first_half + gap_width_m / ring_radius_m
    angles = [gap_bearing_rad + 0.5 * gap_angle]
    for first, second in zip(radii_list[:-1], radii_list[1:]):
        angles.append(angles[-1] + chord_angle(first + second - overlap))
    centers = np.stack(
        [
            ring_radius_m * np.cos(np.asarray(angles)),
            ring_radius_m * np.sin(np.asarray(angles)),
        ],
        axis=1,
    )
    return centers, np.asarray(radii_list, dtype=np.float64)


def _sample_double_ring_pair(
    target_width_m: float,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """Reuse the owner-approved 10.5/17.5 m sealed pair construction."""
    inner_bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    outer_bearing = inner_bearing + float(
        rng.uniform(DOUBLE_RING_MIN_GAP_OFFSET_RAD, 1.5 * math.pi)
    )
    sampled_inner = _sample_sealed_ring(
        RING_RADIUS_M, target_width_m, inner_bearing, rng
    )
    sampled_outer = _sample_sealed_ring(
        DOUBLE_RING_OUTER_RADIUS_M, target_width_m, outer_bearing, rng
    )
    if sampled_inner is None or sampled_outer is None:
        return None
    return sampled_inner, sampled_outer


def _nearest_neighbour_surface_gaps(
    centers: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    gaps = distances - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, np.inf)
    return gaps.min(axis=1)


def _ring_gap_plug(
    centers: np.ndarray,
    opening: tuple[float, float, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Tile plug cylinders between the scanned jambs of a shipped ring."""
    plug_radius = 0.5 * (RING_OBSTACLE_RADIUS_MIN_M + RING_OBSTACLE_RADIUS_MAX_M)
    before, after = opening[2], opening[3]
    plug_centers = wall_segment_cylinders(
        centers[before],
        centers[after],
        radius_m=plug_radius,
        overlap_m=RING_SEALED_OVERLAP_M,
    )
    return plug_centers, np.full(len(plug_centers), plug_radius, dtype=np.float64)


def sample_double_ring_layout(
    level: int,
    rng: np.random.Generator | None = None,
    max_attempts: int = 60,
) -> tuple[HazardLayout, DoubleRingGapSpecs]:
    """Sample two independently sealed rings whose only exits are misaligned.

    Admission is deliberately performed on the concatenated arrays returned
    to the environment. Each opening is scanned at planning inflation; each
    seal verdict plugs those same arrays and uses the true hull half-beam.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    rng = np.random.default_rng() if rng is None else rng
    start = np.zeros(2, dtype=np.float64)
    target_width = float(difficulty.bottleneck_m)

    for attempt in range(1, max_attempts + 1):
        goal = np.array(
            (float(rng.uniform(*DOUBLE_RING_GOAL_DISTANCE_RANGE_M)), 0.0),
            dtype=np.float64,
        )
        sampled_pair = _sample_double_ring_pair(target_width, rng)
        if sampled_pair is None:
            continue
        sampled_inner, sampled_outer = sampled_pair
        inner_centers, inner_radii = sampled_inner
        outer_centers, outer_radii = sampled_outer
        centers = np.vstack((inner_centers, outer_centers))
        radii = np.concatenate((inner_radii, outer_radii))

        # 1. Scan the shipped rings: exactly one tier-width opening in each.
        inner_openings = _ring_line_openings(
            inner_centers, inner_radii, RING_RADIUS_M
        )
        outer_openings = _ring_line_openings(
            outer_centers, outer_radii, DOUBLE_RING_OUTER_RADIUS_M
        )
        if len(inner_openings) != 1 or len(outer_openings) != 1:
            continue
        inner_opening, outer_opening = inner_openings[0], outer_openings[0]
        if (
            abs(inner_opening[0] - target_width) > 0.01
            or abs(outer_opening[0] - target_width) > 0.01
        ):
            continue
        angular_offset = abs(
            (outer_opening[1] - inner_opening[1] + math.pi) % (2.0 * math.pi)
            - math.pi
        )
        if angular_offset + 1.0e-9 < DOUBLE_RING_MIN_GAP_OFFSET_RAD:
            continue

        # 2. Every physical wall cylinder overlaps a same-ring neighbour.
        if (
            np.any(_nearest_neighbour_surface_gaps(inner_centers, inner_radii) >= 0.0)
            or np.any(_nearest_neighbour_surface_gaps(outer_centers, outer_radii) >= 0.0)
        ):
            continue

        # 3. The shipped layout routes at planning inflation.
        geodesic = bfs_geodesic_length(start, goal, centers, radii)
        if geodesic is None:
            continue

        # 4. Plug only the scanned inner opening in the shipped arrays.
        inner_plug_c, inner_plug_r = _ring_gap_plug(inner_centers, inner_opening)
        if bfs_geodesic_length(
            start,
            goal,
            np.vstack((centers, inner_plug_c)),
            np.concatenate((radii, inner_plug_r)),
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        ) is not None:
            continue

        # 5. Plug only the scanned outer opening in the shipped arrays.
        outer_plug_c, outer_plug_r = _ring_gap_plug(outer_centers, outer_opening)
        if bfs_geodesic_length(
            start,
            goal,
            np.vstack((centers, outer_plug_c)),
            np.concatenate((radii, outer_plug_r)),
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        ) is not None:
            continue

        # 6. Both real openings route at the same true-half-beam resolution.
        if bfs_geodesic_length(
            start,
            goal,
            centers,
            radii,
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        ) is None:
            continue
        if not start_goal_disks_clear(start, goal, centers, radii):
            continue

        inner_spec = RingGapSpec(
            ring_radius_m=RING_RADIUS_M,
            center=RING_RADIUS_M
            * np.array((math.cos(inner_opening[1]), math.sin(inner_opening[1]))),
            bearing_rad=inner_opening[1],
            free_width_m=inner_opening[0],
            requested_width_m=target_width,
        )
        outer_spec = RingGapSpec(
            ring_radius_m=DOUBLE_RING_OUTER_RADIUS_M,
            center=DOUBLE_RING_OUTER_RADIUS_M
            * np.array((math.cos(outer_opening[1]), math.sin(outer_opening[1]))),
            bearing_rad=outer_opening[1],
            free_width_m=outer_opening[0],
            requested_width_m=target_width,
        )
        layout = HazardLayout(
            level=difficulty.level,
            requested_obstacle_count=len(radii),
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=float(geodesic),
            direct_blocked=direct_segment_blocked(start, goal, centers, radii),
            attempts=attempt,
        )
        return layout, DoubleRingGapSpecs(inner=inner_spec, outer=outer_spec)

    raise RuntimeError(
        f"Could not generate a double-ring layout for level {level} "
        f"in {max_attempts} attempts."
    )


def _sample_fortress_spawn(
    radius_range_m: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    """Sample radius uniformly in the declared annulus and bearing uniformly."""
    radius = float(rng.uniform(*radius_range_m))
    bearing = float(rng.uniform(0.0, 2.0 * math.pi))
    return radius * np.array((math.cos(bearing), math.sin(bearing)))


def _ring_gap_spec(
    ring_radius_m: float,
    opening: tuple[float, float, int, int],
    requested_width_m: float,
) -> RingGapSpec:
    bearing = opening[1]
    return RingGapSpec(
        ring_radius_m=ring_radius_m,
        center=ring_radius_m
        * np.array((math.cos(bearing), math.sin(bearing))),
        bearing_rad=bearing,
        free_width_m=opening[0],
        requested_width_m=requested_width_m,
    )


def sample_ring_fortress_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 60,
) -> tuple[HazardLayout, RingGapSpec]:
    """Sample one sealed ring with the goal inside and the spawn outside.

    The 18--24 m spawn annulus puts the nearest 10.5 m ring centerline
    7.5--13.5 m away, comfortably inside the environment's 30 m ray range.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    rng = np.random.default_rng() if rng is None else rng
    goal = np.zeros(2, dtype=np.float64)
    target_width = float(difficulty.bottleneck_m)

    for attempt in range(1, max_attempts + 1):
        start = _sample_fortress_spawn(FORTRESS_SPAWN_RADIUS_RANGE_M, rng)
        sampled_layout = sample_ring_layout(
            level,
            rng=rng,
            max_attempts=max(40, max_attempts),
            neighbor_overlap_m=RING_SEALED_OVERLAP_M,
            exact_gap_width_m=target_width,
        )
        centers, radii = sampled_layout.centers, sampled_layout.radii
        if len(radii) > FORTRESS_MAX_OBSTACLES:
            continue
        openings = _ring_line_openings(centers, radii, RING_RADIUS_M)
        if len(openings) != 1 or abs(openings[0][0] - target_width) > 0.01:
            continue
        if np.any(_nearest_neighbour_surface_gaps(centers, radii) >= HULL_BEAM_M):
            continue
        if not start_goal_disks_clear(start, goal, centers, radii):
            continue
        geodesic = bfs_geodesic_length(start, goal, centers, radii)
        if geodesic is None:
            continue
        plug_centers, plug_radii = _ring_gap_plug(centers, openings[0])
        if bfs_geodesic_length(
            goal,
            start,
            np.vstack((centers, plug_centers)),
            np.concatenate((radii, plug_radii)),
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        ) is not None:
            continue
        if bfs_geodesic_length(
            start,
            goal,
            centers,
            radii,
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        ) is None:
            continue
        layout = HazardLayout(
            level=difficulty.level,
            requested_obstacle_count=len(radii),
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=float(geodesic),
            direct_blocked=direct_segment_blocked(start, goal, centers, radii),
            attempts=attempt,
        )
        return layout, _ring_gap_spec(RING_RADIUS_M, openings[0], target_width)

    raise RuntimeError(
        f"Could not generate a ring-fortress layout for level {level} "
        f"in {max_attempts} attempts."
    )


def sample_double_ring_fortress_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 80,
) -> tuple[HazardLayout, DoubleRingGapSpecs]:
    """Sample the sealed double ring with the goal inside and spawn outside.

    The 25--31 m spawn annulus puts the nearest 17.5 m outer-ring centerline
    7.5--13.5 m away, comfortably inside the environment's 30 m ray range.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    rng = np.random.default_rng() if rng is None else rng
    goal = np.zeros(2, dtype=np.float64)
    target_width = float(difficulty.bottleneck_m)

    for attempt in range(1, max_attempts + 1):
        start = _sample_fortress_spawn(
            DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M, rng
        )
        sampled_pair = _sample_double_ring_pair(target_width, rng)
        if sampled_pair is None:
            continue
        (inner_centers, inner_radii), (outer_centers, outer_radii) = sampled_pair
        if (
            len(inner_radii) > DOUBLE_RING_FORTRESS_MAX_INNER_OBSTACLES
            or len(outer_radii) > DOUBLE_RING_FORTRESS_MAX_OUTER_OBSTACLES
        ):
            continue
        centers = np.vstack((inner_centers, outer_centers))
        radii = np.concatenate((inner_radii, outer_radii))
        inner_openings = _ring_line_openings(
            inner_centers, inner_radii, RING_RADIUS_M
        )
        outer_openings = _ring_line_openings(
            outer_centers, outer_radii, DOUBLE_RING_OUTER_RADIUS_M
        )
        if len(inner_openings) != 1 or len(outer_openings) != 1:
            continue
        inner_opening, outer_opening = inner_openings[0], outer_openings[0]
        if (
            abs(inner_opening[0] - target_width) > 0.01
            or abs(outer_opening[0] - target_width) > 0.01
        ):
            continue
        angular_offset = abs(
            (outer_opening[1] - inner_opening[1] + math.pi) % (2.0 * math.pi)
            - math.pi
        )
        if angular_offset + 1.0e-9 < DOUBLE_RING_MIN_GAP_OFFSET_RAD:
            continue
        if (
            np.any(
                _nearest_neighbour_surface_gaps(inner_centers, inner_radii) >= 0.0
            )
            or np.any(
                _nearest_neighbour_surface_gaps(outer_centers, outer_radii) >= 0.0
            )
        ):
            continue
        if not start_goal_disks_clear(start, goal, centers, radii):
            continue
        geodesic = bfs_geodesic_length(start, goal, centers, radii)
        if geodesic is None:
            continue

        for ring_centers, opening in (
            (inner_centers, inner_opening),
            (outer_centers, outer_opening),
        ):
            plug_centers, plug_radii = _ring_gap_plug(ring_centers, opening)
            if bfs_geodesic_length(
                goal,
                start,
                np.vstack((centers, plug_centers)),
                np.concatenate((radii, plug_radii)),
                cell_m=FORCED_SEAL_CELL_M,
                inflation_m=HALF_BEAM_M,
            ) is not None:
                break
        else:
            if bfs_geodesic_length(
                start,
                goal,
                centers,
                radii,
                cell_m=FORCED_SEAL_CELL_M,
                inflation_m=HALF_BEAM_M,
            ) is None:
                continue
            inner_spec = _ring_gap_spec(
                RING_RADIUS_M, inner_opening, target_width
            )
            outer_spec = _ring_gap_spec(
                DOUBLE_RING_OUTER_RADIUS_M, outer_opening, target_width
            )
            layout = HazardLayout(
                level=difficulty.level,
                requested_obstacle_count=len(radii),
                start=start,
                goal=goal,
                centers=centers,
                radii=radii,
                geodesic_length=float(geodesic),
                direct_blocked=direct_segment_blocked(start, goal, centers, radii),
                attempts=attempt,
            )
            return layout, DoubleRingGapSpecs(inner=inner_spec, outer=outer_spec)

    raise RuntimeError(
        f"Could not generate a double-ring-fortress layout for level {level} "
        f"in {max_attempts} attempts."
    )


def _band_clear_of_all(
    candidate: np.ndarray,
    radius_m: float,
    placed_centers: list[np.ndarray],
    placed_radii: list[float],
    aperture_m: float,
) -> bool:
    """Whether a candidate preserves the constructive pairwise aperture."""
    if not placed_centers:
        return True
    distances = np.linalg.norm(np.asarray(placed_centers) - candidate, axis=1)
    edge_gaps = distances - np.asarray(placed_radii) - radius_m
    return bool(np.all(edge_gaps >= aperture_m - 1.0e-9))


def _build_band_layer(
    ring_radius_m: float,
    aperture_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place a chord-spaced cylinder layer, including its wrap gap."""
    radii: list[float] = []
    angles: list[float] = []
    theta = start = float(rng.uniform(0.0, 2.0 * math.pi))
    while True:
        radius = float(rng.uniform(*BAND_FORTRESS_LAYER_RADIUS_RANGE_M))
        if radii:
            theta += 2.0 * math.asin(
                min(
                    1.0,
                    (radii[-1] + aperture_m + radius)
                    / (2.0 * ring_radius_m),
                )
            )
        if radii and theta - start > 2.0 * math.pi:
            break
        if radii:
            wrap = start + 2.0 * math.pi - theta
            needed = 2.0 * math.asin(
                min(
                    1.0,
                    (radius + aperture_m + radii[0])
                    / (2.0 * ring_radius_m),
                )
            )
            if wrap < needed:
                break
        angles.append(theta)
        radii.append(radius)
        if len(radii) > 200:
            break

    angle_array = np.asarray(angles, dtype=np.float64)
    centers = ring_radius_m * np.stack(
        (np.cos(angle_array), np.sin(angle_array)), axis=1
    )
    return centers, np.asarray(radii, dtype=np.float64), angle_array


def _add_band_plugs(
    angles: np.ndarray,
    ring_radius_m: float,
    aperture_m: float,
    rng: np.random.Generator,
    all_centers: list[np.ndarray],
    all_radii: list[float],
    *,
    max_offset_m: float,
) -> int:
    """Back every layer gap with a plug and return the number skipped."""
    skipped = 0
    for index in range(len(angles)):
        next_index = (index + 1) % len(angles)
        next_angle = float(angles[next_index])
        if next_index == 0:
            next_angle += 2.0 * math.pi
        midpoint = 0.5 * (float(angles[index]) + next_angle)
        placed = False
        radius_candidates = (
            float(rng.uniform(*BAND_FORTRESS_PLUG_RADIUS_RANGE_M)),
            1.4,
            1.1,
        )
        for radius in radius_candidates:
            offset = 0.8
            while offset <= max_offset_m:
                candidate = (ring_radius_m + offset) * np.array(
                    (math.cos(midpoint), math.sin(midpoint)), dtype=np.float64
                )
                if _band_clear_of_all(
                    candidate,
                    radius,
                    all_centers,
                    all_radii,
                    aperture_m,
                ):
                    all_centers.append(candidate)
                    all_radii.append(radius)
                    placed = True
                    break
                offset += 0.25
            if placed:
                break
        if not placed:
            skipped += 1
    return skipped


def _build_band_fortress(
    aperture_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Faithful constructive port of ``demos/band_v2_calc.py``."""
    all_centers: list[np.ndarray] = []
    all_radii: list[float] = []

    inner_centers, inner_radii, inner_angles = _build_band_layer(
        BAND_FORTRESS_INNER_RADIUS_M, aperture_m, rng
    )
    all_centers.extend(inner_centers)
    all_radii.extend(inner_radii)
    inner_skipped = _add_band_plugs(
        inner_angles,
        BAND_FORTRESS_INNER_RADIUS_M,
        aperture_m,
        rng,
        all_centers,
        all_radii,
        max_offset_m=BAND_FORTRESS_SEPARATION_M - 1.0,
    )

    outer_ring_radius = (
        BAND_FORTRESS_INNER_RADIUS_M
        + BAND_FORTRESS_SEPARATION_M
        + aperture_m
        + 2.0 * BAND_FORTRESS_PLUG_RADIUS_RANGE_M[1]
    )
    outer_centers, outer_radii, outer_angles = _build_band_layer(
        outer_ring_radius, aperture_m, rng
    )
    kept_outer_centers: list[np.ndarray] = []
    kept_outer_radii: list[float] = []
    for center, radius in zip(outer_centers, outer_radii):
        if _band_clear_of_all(
            center, radius, all_centers, all_radii, aperture_m
        ):
            kept_outer_centers.append(center)
            kept_outer_radii.append(float(radius))
    all_centers.extend(kept_outer_centers)
    all_radii.extend(kept_outer_radii)
    outer_skipped = _add_band_plugs(
        outer_angles,
        outer_ring_radius,
        aperture_m,
        rng,
        all_centers,
        all_radii,
        max_offset_m=10.0,
    )

    centers = np.asarray(all_centers, dtype=np.float64).reshape(-1, 2)
    radii = np.asarray(all_radii, dtype=np.float64)
    outer_extent = float(np.max(np.linalg.norm(centers, axis=1) + radii))
    return centers, radii, inner_skipped + outer_skipped, outer_extent


def sample_band_fortress_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 80,
    aperture_override_m: float | None = None,
) -> HazardLayout:
    """Sample two constructive brick-wall bands around an origin goal."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    try:
        aperture = float(FORTRESS_APERTURE_M[int(level)])
    except (KeyError, ValueError):
        valid = ", ".join(str(value) for value in sorted(FORTRESS_APERTURE_M))
        raise ValueError(
            f"Unknown band-fortress level {level!r}; expected one of {valid}."
        ) from None
    # Tier-regrade probes: the frozen table has a measured cliff (32.8% at
    # 6.0 m straight to 0% at 4.5 m), so intermediate apertures must be
    # measurable without touching tier semantics. None/0 keeps the table.
    if aperture_override_m is not None and aperture_override_m > 0.0:
        aperture = float(aperture_override_m)
    rng = np.random.default_rng() if rng is None else rng
    goal = np.zeros(2, dtype=np.float64)

    for attempt in range(1, max_attempts + 1):
        centers, radii, _skipped, outer_extent = _build_band_fortress(
            aperture, rng
        )
        if (
            len(radii) == 0
            or len(radii) > BAND_FORTRESS_MAX_OBSTACLES
            or len(centers) != len(radii)
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(radii))
            or not math.isfinite(outer_extent)
        ):
            continue
        delta = centers[:, None, :] - centers[None, :, :]
        edge_gaps = (
            np.linalg.norm(delta, axis=-1)
            - radii[:, None]
            - radii[None, :]
        )
        np.fill_diagonal(edge_gaps, math.inf)
        if float(np.min(edge_gaps)) < aperture - 1.0e-6:
            continue

        spawn_radius = float(rng.uniform(outer_extent + 2.0, outer_extent + 8.0))
        spawn_bearing = float(rng.uniform(0.0, 2.0 * math.pi))
        start = spawn_radius * np.array(
            (math.cos(spawn_bearing), math.sin(spawn_bearing)), dtype=np.float64
        )
        geodesic = bfs_geodesic_length(
            start,
            goal,
            centers,
            radii,
            inflation_m=HALF_BEAM_M,
        )
        if geodesic is None:
            continue

        # The 150-layout-per-tier admission audit observed at most 72
        # cylinders; the 96-slot task budget therefore leaves 24 slots headroom.
        return HazardLayout(
            level=int(level),
            requested_obstacle_count=len(radii),
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=float(geodesic),
            direct_blocked=direct_segment_blocked(start, goal, centers, radii),
            attempts=attempt,
        )

    raise RuntimeError(
        f"Could not generate a band-fortress layout for level {level} "
        f"in {max_attempts} attempts."
    )


def analytic_min_clearance(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    *,
    half_beam_m: float = HALF_BEAM_M,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """COM-to-cylinder clearance ``min(||p-c|| - radius - half_beam)``."""
    if points.shape[-1] != 2 or centers.shape[-1] != 2:
        raise ValueError("points and centers must end in planar XY coordinates")
    if centers.shape[-2] == 0:
        return torch.full(points.shape[:-1], torch.inf, device=points.device, dtype=points.dtype)

    distances = torch.linalg.vector_norm(points.unsqueeze(-2) - centers, dim=-1)
    clearances = distances - radii - float(half_beam_m)
    if active_mask is not None:
        clearances = torch.where(
            active_mask,
            clearances,
            torch.full_like(clearances, torch.inf),
        )
    return torch.min(clearances, dim=-1).values


def ray_circle_ranges(
    origins: torch.Tensor,
    directions: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    *,
    max_range_m: float = 30.0,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Nearest forward ray/cylinder intersections for arbitrary batched rays."""
    if max_range_m <= 0.0:
        raise ValueError("max_range_m must be positive")
    if origins.shape[-1] != 2 or directions.shape[-1] != 2 or centers.shape[-1] != 2:
        raise ValueError("origins, directions, and centers must be planar XY")
    if centers.shape[-2] == 0:
        return torch.full(
            directions.shape[:-1], max_range_m, device=directions.device, dtype=directions.dtype
        )

    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    to_center = centers.unsqueeze(-3) - origins.unsqueeze(-2).unsqueeze(-2)
    projection = torch.sum(directions.unsqueeze(-2) * to_center, dim=-1)
    center_distance_sq = torch.sum(to_center * to_center, dim=-1)
    perpendicular_sq = center_distance_sq - projection * projection
    radius_sq = radii.unsqueeze(-2) * radii.unsqueeze(-2)
    discriminant = radius_sq - perpendicular_sq
    root = torch.sqrt(torch.clamp(discriminant, min=0.0))
    inside = center_distance_sq <= radius_sq
    entry = torch.where(inside, torch.zeros_like(projection), projection - root)
    valid = (discriminant >= 0.0) & (entry >= 0.0)
    if active_mask is not None:
        valid &= active_mask.unsqueeze(-2)
    intersections = torch.where(
        valid,
        entry,
        torch.full_like(entry, float(max_range_m)),
    )
    return torch.min(intersections, dim=-1).values.clamp(max=float(max_range_m))


# --- Forced-crossing variant: a closed basin split by a gated bulkhead -------
#
# Why this exists: every certified Task A number measures willingness to
# detour, not avoidance. The scatter arena has no boundary, terminations are
# only reached/contact/timeout, and circumventing the whole obstacle field
# costs 1.63-1.82x the straight line at EVERY tier -- the 128/128 champion
# takes that route (median path 53.6 m against a 29.7 m straight line).
#
# Here the arena is a closed rectangle divided by a wall with exactly one gate.
# Admission proves that with the gate sealed the goal is unreachable AT THE
# TRUE HULL HALF-BEAM, so there is no "around" to be willing to take: every
# success has transited the gate.
#
# Gate widths are declared in PHYSICAL metres, deliberately in their own table
# rather than reusing Difficulty.bottleneck_m. That field is an INFLATED gap
# (planner units, half-beam + margin = 0.65); silently carrying the same
# numbers across the two meanings is exactly the checker-vs-reality confusion
# that let the "sealed" ring ship with 1.00 m holes in a 0.899 m hull.
FORCED_GATE_BEAMS: dict[int, float] = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}

FORCED_PERIMETER_RADIUS_M = 1.20  # basin sides/ends: coarse, only has to seal
FORCED_BULKHEAD_RADIUS_M = 0.80  # gate jambs: MIN_OBSTACLE_RADIUS, 2-ray covered
FORCED_WALL_OVERLAP_M = 0.25  # surface overlap FLOOR between wall neighbours
FORCED_ENDPOINT_CLEAR_M = 2.50  # the 5.0 m scatter disk cannot fit in a basin
FORCED_APPROACH_MIN_M = 6.00  # endpoint-to-bulkhead room to line up
FORCED_BASIN_HALF_WIDTH_RANGE_M = (6.0, 9.0)
FORCED_END_MARGIN_RANGE_M = (3.6, 6.0)  # >= perimeter radius + endpoint clear
FORCED_GOAL_DISTANCE_RANGE_M = (20.0, 34.0)
FORCED_BULKHEAD_MIN_SIDE_M = 2.0  # each jamb must be a real wall, not a stub
FORCED_SEAL_CELL_M = 0.25  # seal BFS resolution; walls are >= 1.6 m thick
FORCED_GATE_FIT_MARGIN_M = 0.30  # gate must beat the beam by this much


@dataclass(frozen=True)
class GateSpec:
    """The single opening in the bulkhead, described as measured not requested."""

    center: np.ndarray  # midpoint of the free opening
    along: np.ndarray  # unit vector across the opening (along the wall)
    through: np.ndarray  # unit vector from the start chamber to the goal chamber
    free_width_m: float  # MEASURED between physical cylinder surfaces
    requested_width_m: float
    wall_x: float


def wall_segment_cylinders(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    radius_m: float,
    overlap_m: float = FORCED_WALL_OVERLAP_M,
) -> np.ndarray:
    """Tile a segment with cylinder centres whose surfaces overlap.

    The count comes from a ceiling, so the realised overlap is at least
    ``overlap_m``. Overlap has to be a FLOOR: when the ring generator treated
    it as a ceiling, bisection returned values near the loose end and 5/15
    rings still leaked.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    span = float(np.linalg.norm(p1 - p0))
    stride = 2.0 * radius_m - overlap_m
    if stride <= 0.0:
        raise ValueError("overlap_m must be smaller than the cylinder diameter")
    count = max(2, int(math.ceil(span / stride)) + 1)
    fractions = np.linspace(0.0, 1.0, count)[:, None]
    return p0[None, :] + fractions * (p1 - p0)[None, :]


def _basin_cylinders(
    x_back: float,
    x_front: float,
    half_width: float,
    wall_x: float,
    gate_y: float,
    gate_width: float,
    *,
    include_bulkhead: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Four perimeter walls plus (optionally) a bulkhead split by one gate.

    ``include_bulkhead=False`` builds the same basin with nothing in it. That
    is the control for the zero-shot result: the basin introduces boundaries
    AND a mandatory gate at once, and the Task A champion dies on the walls
    (median path 4.95 m) before it ever reaches the bulkhead at >= 6 m, so the
    two effects have to be separated before the number means anything.
    """
    r_p = FORCED_PERIMETER_RADIUS_M
    r_b = FORCED_BULKHEAD_RADIUS_M
    # Jamb CENTRES sit half a gate plus one radius out, so the free opening
    # between the two physical SURFACES is exactly ``gate_width``.
    jamb = 0.5 * gate_width + r_b
    perimeter = [
        ((x_back, -half_width), (x_front, -half_width)),
        ((x_back, half_width), (x_front, half_width)),
        ((x_back, -half_width), (x_back, half_width)),
        ((x_front, -half_width), (x_front, half_width)),
    ]
    bulkhead = [
        ((wall_x, -half_width), (wall_x, gate_y - jamb)),
        ((wall_x, gate_y + jamb), (wall_x, half_width)),
    ] if include_bulkhead else []

    centers: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    for segments, radius in ((perimeter, r_p), (bulkhead, r_b)):
        if not segments:
            continue
        for p0, p1 in segments:
            pts = wall_segment_cylinders(np.array(p0), np.array(p1), radius_m=radius)
            centers.append(pts)
            radii.append(np.full(len(pts), radius))
    return np.vstack(centers), np.concatenate(radii)


def _bulkhead_free_runs(
    centers: np.ndarray,
    radii: np.ndarray,
    wall_x: float,
    y_lo: float,
    y_hi: float,
    *,
    samples: int = 6000,
) -> list[tuple[float, float]]:
    """Contiguous stretches of the bulkhead line that no cylinder covers."""
    ys = np.linspace(y_lo, y_hi, samples)
    points = np.stack([np.full_like(ys, wall_x), ys], axis=1)
    distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    free = np.all(distance > radii[None, :], axis=1)
    runs: list[tuple[float, float]] = []
    start = None
    for index, is_free in enumerate(free):
        if is_free and start is None:
            start = index
        elif not is_free and start is not None:
            runs.append((float(ys[start]), float(ys[index - 1])))
            start = None
    if start is not None:
        runs.append((float(ys[start]), float(ys[-1])))
    return runs


def sample_open_basin_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 60,
) -> HazardLayout:
    """The crossing basin with the bulkhead removed: boundaries and nothing else.

    Control for the zero-shot experiment. The basin changes two things at once
    relative to Task A -- the arena acquires walls, and the route acquires a
    mandatory gate -- and the champion collides in 128/128 after a median 4.95 m,
    which is short of the >= 6 m to the bulkhead. If it also fails here, the
    walls alone explain the result and the gate contributed nothing to it.

    Same perimeter distribution as the crossing task, so the only difference
    between the two is the bulkhead.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    rng = np.random.default_rng() if rng is None else rng
    start = np.zeros(2, dtype=np.float64)

    for attempt in range(1, max_attempts + 1):
        distance = float(rng.uniform(*FORCED_GOAL_DISTANCE_RANGE_M))
        goal = np.array((distance, 0.0), dtype=np.float64)
        half_width = float(rng.uniform(*FORCED_BASIN_HALF_WIDTH_RANGE_M))
        x_back = -float(rng.uniform(*FORCED_END_MARGIN_RANGE_M))
        x_front = distance + float(rng.uniform(*FORCED_END_MARGIN_RANGE_M))

        centers, radii = _basin_cylinders(
            x_back, x_front, half_width, 0.0, 0.0, 0.0, include_bulkhead=False
        )
        if not start_goal_disks_clear(
            start, goal, centers, radii,
            clear_radius_m=FORCED_ENDPOINT_CLEAR_M, inflation_m=0.0,
        ):
            continue
        geodesic = bfs_geodesic_length(
            start, goal, centers, radii, cell_m=FORCED_SEAL_CELL_M
        )
        if geodesic is None:
            continue
        return HazardLayout(
            level=difficulty.level,
            requested_obstacle_count=int(len(radii)),
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=float(geodesic),
            direct_blocked=direct_segment_blocked(start, goal, centers, radii),
            attempts=attempt,
        )

    raise RuntimeError(
        f"Could not generate an open-basin layout for level {level} "
        f"in {max_attempts} attempts."
    )


def sample_forced_crossing_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 60,
    gate_width_override_m: float | None = None,
) -> tuple[HazardLayout, GateSpec]:
    """Sample a closed basin whose only route to the goal is through the gate.

    Admission, in the order it rejects most cheaply:

    1. the bulkhead jambs are real walls (>= FORCED_BULKHEAD_MIN_SIDE_M each);
    2. the MEASURED free opening matches the requested width and clears the
       hull beam by FORCED_GATE_FIT_MARGIN_M -- measured by scanning the wall
       line, never assumed from the construction;
    3. exactly one free run on the bulkhead line (a second one is a leak);
    4. endpoint disks clear of every cylinder;
    5. BFS at PLANNING inflation finds a route as built;
    6. BFS with the gate filled in finds NO route at the TRUE half-beam with no
       margin -- the check the ring's admission got wrong by inflating to 0.65
       while the simulator's contact predicate uses 0.45.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    difficulty = difficulty_for_level(level)
    gate_width = float(FORCED_GATE_BEAMS[difficulty.level] * HULL_BEAM_M)
    # Tier-regrade probes (e.g. a 1.5x-beam T5 candidate) pin the gate in
    # physical metres; the hull-fit guard below still applies to overrides.
    if gate_width_override_m is not None and gate_width_override_m > 0.0:
        gate_width = float(gate_width_override_m)
    if gate_width < HULL_BEAM_M + FORCED_GATE_FIT_MARGIN_M:
        raise ValueError(f"level {level} gate {gate_width:.3f} m cannot fit the hull")
    rng = np.random.default_rng() if rng is None else rng
    start = np.zeros(2, dtype=np.float64)

    for attempt in range(1, max_attempts + 1):
        distance = float(rng.uniform(*FORCED_GOAL_DISTANCE_RANGE_M))
        goal = np.array((distance, 0.0), dtype=np.float64)
        if distance < 2.0 * FORCED_APPROACH_MIN_M:
            continue
        half_width = float(rng.uniform(*FORCED_BASIN_HALF_WIDTH_RANGE_M))
        x_back = -float(rng.uniform(*FORCED_END_MARGIN_RANGE_M))
        x_front = distance + float(rng.uniform(*FORCED_END_MARGIN_RANGE_M))
        wall_x = float(
            rng.uniform(FORCED_APPROACH_MIN_M, distance - FORCED_APPROACH_MIN_M)
        )
        jamb = 0.5 * gate_width + FORCED_BULKHEAD_RADIUS_M
        gate_limit = half_width - jamb - FORCED_BULKHEAD_MIN_SIDE_M
        if gate_limit <= 0.0:
            continue
        gate_y = float(rng.uniform(-gate_limit, gate_limit))

        centers, radii = _basin_cylinders(
            x_back, x_front, half_width, wall_x, gate_y, gate_width
        )

        runs = _bulkhead_free_runs(centers, radii, wall_x, -half_width, half_width)
        if len(runs) != 1:
            continue
        measured = runs[0][1] - runs[0][0]
        if abs(measured - gate_width) > 0.01:
            continue
        if measured < HULL_BEAM_M + FORCED_GATE_FIT_MARGIN_M:
            continue

        if not start_goal_disks_clear(
            start,
            goal,
            centers,
            radii,
            clear_radius_m=FORCED_ENDPOINT_CLEAR_M,
            inflation_m=0.0,
        ):
            continue

        geodesic = bfs_geodesic_length(
            start, goal, centers, radii, cell_m=FORCED_SEAL_CELL_M
        )
        if geodesic is None:
            continue

        sealed_centers, sealed_radii = _basin_cylinders(
            x_back, x_front, half_width, wall_x, gate_y, 0.0
        )
        leak = bfs_geodesic_length(
            start,
            goal,
            sealed_centers,
            sealed_radii,
            cell_m=FORCED_SEAL_CELL_M,
            inflation_m=HALF_BEAM_M,
        )
        if leak is not None:
            continue

        gate = GateSpec(
            center=np.array((wall_x, 0.5 * (runs[0][0] + runs[0][1]))),
            along=np.array((0.0, 1.0)),
            through=np.array((1.0, 0.0)),
            free_width_m=float(measured),
            requested_width_m=gate_width,
            wall_x=wall_x,
        )
        layout = HazardLayout(
            level=difficulty.level,
            requested_obstacle_count=int(len(radii)),
            start=start,
            goal=goal,
            centers=centers,
            radii=radii,
            geodesic_length=float(geodesic),
            direct_blocked=direct_segment_blocked(start, goal, centers, radii),
            attempts=attempt,
        )
        return layout, gate

    raise RuntimeError(
        f"Could not generate a forced-crossing layout for level {level} "
        f"in {max_attempts} attempts."
    )


# --- Mid-episode obstacle appearance: the sudden-terrain-change axis ---------
#
# One or more EXTRA cylinders flip from inactive to active at a per-episode
# time t0.  The buffers are read every step already, so the flip needs no data
# structure; what needs care is FAIRNESS -- the obstacle must be guaranteed
# visible the instant it exists, and it must not make the remaining task
# impossible.  The planner below is the whole decision; the env only executes
# an accepted plan, so this function is testable on CPU without Isaac.
#
# FAIRNESS FLOOR (frozen 2026-08-26, owner's verified derivation):
#
# * The appearance azimuth is ALIGNED TO A SENSOR RAY at the moment of
#   appearance.  The 36-ray sensor has 10-degree spacing, so a cylinder of the
#   minimum admitted radius (MIN_OBSTACLE_RADIUS_M = 0.8) subtends less than
#   the 5-degree half-spacing beyond r / sin(5 deg) = 9.18 m and could fall
#   BETWEEN rays; centred on a ray it is hit at any admitted distance
#   (25 - 0.8 = 24.2 m of surface range against the 30 m ray cap).
#
# * CORRECTED 2026-08-31 (owner-directed).  The 2026-08-26 text had the
#   floor's DIRECTION backwards, and the clamp below it silently capped the
#   whole dose axis at its lightest usable rung.
#
#   r / sin(5 deg) = 9.18 m is an UPPER bound, not a floor: BEYOND it a
#   minimum-radius cylinder subtends less than the half-spacing and could
#   fall between rays.  That is exactly why the azimuth is snapped onto a
#   ray.  Snapped, the cylinder is hit at every admitted distance, so 9.18 m
#   constrains nothing here.  It is also self-inconsistent as a guarantee:
#   asin(0.8 / 9.2) = 4.9885 deg, already inside 5 deg.  The figure is kept
#   as APPEAR_RAY_VISIBILITY_LIMIT_M for the record and is NOT enforced.
#
#   The binding constraint is braking, re-derived against this repo's own
#   integrator rather than the round numbers in the old note: top speed
#   2.9910 m/s (80 N thrust against 9.1 v + 5.9 v^2 drag); a full-reverse
#   stop takes 0.7083 s and carries the hull APPEAR_STOP_DISTANCE_M = 0.868 m
#   forward.  Collision geometry is radius + half beam.  The centre floor is
#   therefore radius + half_beam + stop = 2.118 m at the minimum radius,
#   computed per radius by ``appear_min_distance_m``.  The old note's 0.891 m
#   was the braking term alone, excluding the 0.25 s reaction leg (a further
#   0.748 m) -- and it was never the binding number in any case.
#
#   Motor lag moves the floor: pass the lagged stop distance to reach 2.413 m
#   at tau = 0.10 s or 2.811 m at tau = 0.25 s.  Callers that enable the
#   motor-lag knob MUST pass it; the default is the zero-lag figure.
#
#   Out-of-range distances now RAISE.  They used to be clamped in silence, so
#   ``--set appear_distance_m=2.25`` ran at 9.2 m while the certificate's
#   ``overrides`` still recorded 2.25 -- a certificate that misstated its own
#   dose.  The ceiling APPEAR_MAX_DISTANCE_M = 25 m is unchanged and still
#   keeps the appearance inside the ray horizon with margin for the largest
#   admitted radius.
#
# * Admission is the layout samplers' own feasibility check: eight-connected
#   BFS at PLANNING inflation (half-beam + margin) over the POST-appearance
#   field, from the current boat position to the goal, plus no overlap with
#   the hull disk or the goal disc.  An inadmissible azimuth is resampled
#   (the caller supplies the bounded candidate order); exhaustion returns
#   None, and the env records the skip instead of forcing an unfair spawn.
APPEAR_RAY_VISIBILITY_LIMIT_M = 9.18
APPEAR_STOP_DISTANCE_M = 0.868
# Below this speed the velocity vector is drift noise, not a direction of
# travel, so it must not be used to aim the appearance (added 2026-08-31
# after an audit executed the hole).  About 8 percent of the 2.9910 m/s
# top speed.
APPEAR_MIN_ALIGN_SPEED_MS = 0.25
APPEAR_MAX_DISTANCE_M = 25.0


def appear_min_distance_m(
    radius_m: float,
    half_beam_m: float = HALF_BEAM_M,
    stop_distance_m: float = APPEAR_STOP_DISTANCE_M,
) -> float:
    """Smallest fair centre distance for an appearance of this radius.

    Collision geometry (radius + half beam) plus the distance the hull still
    travels under a full-reverse stop.  At the minimum admitted radius and
    zero motor lag this is 0.8 + 0.45 + 0.868 = 2.118 m.  Pass the lagged
    stop distance when the motor-lag knob is enabled; see the fairness note
    above for the tau = 0.10 and tau = 0.25 figures.
    """
    return float(radius_m) + float(half_beam_m) + float(stop_distance_m)


# The floor for the minimum admitted radius at zero motor lag, kept as a
# module constant because tests and callers quote it.  It is NOT a global
# floor: every radius gets its own via appear_min_distance_m.
APPEAR_MIN_DISTANCE_M = appear_min_distance_m(MIN_OBSTACLE_RADIUS_M)


@dataclass(frozen=True)
class AppearancePlan:
    """One admitted mid-episode appearance, described as measured."""

    center: np.ndarray  # local-frame XY of the admitted cylinder centre
    radius_m: float
    ray_index: int  # the sensor ray the azimuth is aligned to
    azimuth_rad: float  # world-frame bearing boat -> centre (== ray azimuth)
    distance_m: float  # the clamped centre distance actually used
    attempts: int  # candidates examined, including the accepted one


def route_direction_xy(
    boat_xy: np.ndarray,
    goal_xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    cell_m: float = FORCED_SEAL_CELL_M,
    inflation_m: float = OBSTACLE_INFLATION_M,
    lookahead_m: float = 8.0,
) -> np.ndarray | None:
    """Unit vector along the geodesic route the boat still has to travel.

    Added 2026-08-31.  The appearance axis can align its bearing to the
    DIRECTION OF TRAVEL (velocity) or to the DIRECTION OF THE TASK (this).
    They differ whenever the boat is detouring round an existing cylinder or
    turning onto a gate, and the difference is the whole question: an obstacle
    on the velocity vector can be stepped around by changing course, while an
    obstacle on the route has to be dealt with.

    Computed here rather than read off the env because the descent map the env
    maintains only exists when cfg.nav_targets_waypoint is set, which the
    crossing family does not set.  This version needs nothing but the geometry
    the planner is already handed, so it works on every task and is testable
    on CPU.

    The field is inflated at PLANNING inflation, so the route returned is one
    the hull can actually take.  Returns ``None`` when the boat sits inside an
    inflated cylinder, when the goal is unreachable, or when the walk cannot
    leave the starting cell -- all cases where "the direction of the task" is
    not defined and the caller must fall back rather than invent one.
    """
    boat = np.asarray(boat_xy, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    field_centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    field_radii = np.asarray(radii, dtype=np.float64).reshape(-1)

    pad = float(lookahead_m) + 4.0
    if field_centers.size:
        xs_all = np.concatenate((field_centers[:, 0], (boat[0], goal[0])))
        ys_all = np.concatenate((field_centers[:, 1], (boat[1], goal[1])))
        pad += float(field_radii.max())
    else:
        xs_all = np.asarray((boat[0], goal[0]))
        ys_all = np.asarray((boat[1], goal[1]))
    bounds = (
        float(xs_all.min()) - pad,
        float(xs_all.max()) + pad,
        float(ys_all.min()) - pad,
        float(ys_all.max()) + pad,
    )

    field = geodesic_distance_field(
        field_centers,
        field_radii,
        goal,
        bounds,
        cell_m=float(cell_m),
        half_beam_m=float(inflation_m),
    )
    if not np.isfinite(field).any():
        return None
    descent = geodesic_descent_directions(field)

    xs, ys = _bounded_grid_axes(bounds, float(cell_m))
    col = int(np.clip(round((boat[0] - xs[0]) / float(cell_m)), 0, len(xs) - 1))
    row = int(np.clip(round((boat[1] - ys[0]) / float(cell_m)), 0, len(ys) - 1))
    if not np.isfinite(field[row, col]):
        # The boat's own cell can be occupied while the boat is perfectly
        # legal: the field is inflated at PLANNING inflation (half beam plus
        # margin, 0.65 m) but the hull only needs half beam (0.45 m), so a
        # boat 0.50 m off a cylinder surface is legal and its cell is not.
        # Measured on a real level-0 crossing layout: 5.0 percent of legal
        # boat positions land in such a cell.  Returning None there is what
        # killed a whole 128-episode cell -- one env in sixty-four is enough.
        # So snap outward to the nearest cell that IS on the route, which is
        # the same thing the boat itself will do within a few control steps.
        snapped = None
        max_ring = int(math.ceil(2.0 / float(cell_m)))
        for ring in range(1, max_ring + 1):
            best = None
            for d_row in range(-ring, ring + 1):
                for d_col in range(-ring, ring + 1):
                    if max(abs(d_row), abs(d_col)) != ring:
                        continue
                    rr, cc = row + d_row, col + d_col
                    if not (0 <= rr < field.shape[0] and 0 <= cc < field.shape[1]):
                        continue
                    value = field[rr, cc]
                    if np.isfinite(value) and (best is None or value < best[0]):
                        best = (float(value), rr, cc)
            if best is not None:
                snapped = (best[1], best[2])
                break
        if snapped is None:
            return None
        row, col = snapped

    hops = max(1, int(math.ceil(float(lookahead_m) / float(cell_m))))
    walked = 0.0
    r, c = row, col
    for _ in range(hops):
        d_row, d_col = (int(v) for v in descent[r, c])
        if d_row == 0 and d_col == 0:
            break
        r, c = r + d_row, c + d_col
        walked += float(cell_m) * math.hypot(d_row, d_col)
        if walked >= float(lookahead_m):
            break
    if (r, c) == (row, col):
        return None

    target = np.array((xs[c], ys[r]), dtype=np.float64)
    delta = target - boat
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-9:
        return None
    return delta / norm


def plan_obstacle_appearance(
    boat_xy: np.ndarray,
    forward_xy: np.ndarray,
    goal_xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    distance_m: float,
    radius_m: float,
    candidate_ray_indices: Sequence[int] | np.ndarray,
    ray_count: int = 36,
    goal_radius_m: float = 2.0,
    half_beam_m: float = HALF_BEAM_M,
    bfs_cell_m: float = FORCED_SEAL_CELL_M,
    inflation_m: float = OBSTACLE_INFLATION_M,
    stop_distance_m: float = APPEAR_STOP_DISTANCE_M,
    velocity_xy: np.ndarray | None = None,
    route_xy: np.ndarray | None = None,
    strict_ray: bool = False,
    ray_max_range_m: float = 30.0,
) -> AppearancePlan | None:
    """Plan one fair, admissible mid-episode obstacle appearance.

    ``boat_xy``/``forward_xy`` are the boat's CURRENT planar position and
    heading direction; ``centers``/``radii`` are the currently ACTIVE field
    (same local frame as ``boat_xy`` and ``goal_xy``).  Candidates are taken
    from ``candidate_ray_indices`` IN ORDER -- the caller draws that order
    from its scenario stream, which is what bounds the resampling -- and the
    first candidate that passes every admission check is returned.  ``None``
    means every supplied azimuth was inadmissible: the caller must SKIP the
    appearance for this episode and record that it did.

    The candidate centre for ray ``k`` is built exactly on that ray:
    ``boat + d * (cos(a_k) * forward + sin(a_k) * left)`` with
    ``a_k = 2 * pi * k / ray_count``, the same direction formula the env's
    ray sensor uses, so alignment holds to float precision by construction.
    """
    if int(ray_count) < 1:
        raise ValueError("ray_count must be positive")
    if not MIN_OBSTACLE_RADIUS_M <= float(radius_m) <= MAX_OBSTACLE_RADIUS_M:
        raise ValueError(
            f"appearance radius {radius_m} outside "
            f"[{MIN_OBSTACLE_RADIUS_M}, {MAX_OBSTACLE_RADIUS_M}] m; the "
            "appearance floor is derived per radius by "
            "appear_min_distance_m"
        )
    boat = np.asarray(boat_xy, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    forward = np.asarray(forward_xy, dtype=np.float64).reshape(2)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 0.0:
        raise ValueError("forward_xy must be a nonzero direction")
    forward = forward / forward_norm
    left = np.array((-forward[1], forward[0]), dtype=np.float64)
    field_centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    field_radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if field_centers.shape[0] != field_radii.shape[0]:
        raise ValueError("centers and radii must have matching lengths")

    # The fairness range, measured from the CURRENT boat position.  This USED
    # to be a silent clamp, which turned an out-of-range --set into a run at
    # a different dose than the certificate recorded.  It raises now: a dose
    # the rig cannot deliver must stop the run, not be quietly rounded.
    distance = float(distance_m)
    floor = appear_min_distance_m(float(radius_m), float(half_beam_m), stop_distance_m)
    if not floor <= distance <= APPEAR_MAX_DISTANCE_M:
        raise ValueError(
            f"appearance distance {distance} m outside the fair range "
            f"[{floor:.3f}, {APPEAR_MAX_DISTANCE_M}] m for radius "
            f"{float(radius_m)} m and stop distance {stop_distance_m} m. "
            "The floor is collision geometry (radius + half beam) plus the "
            "full-reverse stopping distance; raise the stop distance when the "
            "motor-lag knob is on. This is deliberately not clamped -- a "
            "clamp would run one dose and certify another."
        )

    # --- Ray choice (added 2026-08-31, master TODO section 8.3 items 3+4) --
    #
    # The order used to be a uniform permutation of all 36 rays, i.e. the
    # obstacle could appear in ANY direction including astern.  At 20 m that
    # merely dilutes the dose; at the redesigned near rungs it destroys them,
    # because a cylinder 2.25 m BEHIND the boat is not an obstacle at all.  So
    # when the caller supplies the instantaneous velocity, candidates are
    # ordered by angular distance from the direction of travel: the aligned
    # ray first, then its neighbours alternating outward.
    #
    # This also buys section 8.3 item 2 for free.  Nothing about the episode
    # before t0 depends on the appearance distance, so the velocity at t0 --
    # and therefore the aligned ray -- is identical at every rung of the
    # ladder.  With ``strict_ray`` the aligned ray is the ONLY candidate, so a
    # rung either uses that exact ray or skips and says so; it can never
    # quietly substitute a different azimuth and break the pairing (item 4).
    if strict_ray and velocity_xy is None and route_xy is None:
        # strict_ray means USE THE ALIGNED RAY OR SKIP.  With no alignment
        # source at all there is no aligned ray, so the flag is incoherent and
        # the run would fall through to the uniform order below while the
        # certificate still recorded strict_ray.  The near-rung guard beneath
        # does not catch this above 9.18 m: on 2026-08-31 a 9.5 m route
        # aligned cell completed normally and quietly drew an unknown share of
        # its bearings uniformly, which is the same silent-degradation class
        # as the two defects fixed earlier that day.
        raise ValueError(
            "strict_ray was requested but neither velocity_xy nor route_xy "
            "was supplied, so there is no aligned ray to be strict about and "
            "the bearing would fall back to the uniform candidate order at "
            "any distance. Pass velocity_xy (cfg appear_velocity_aligned) or "
            "route_xy (cfg appear_route_aligned), or turn strict_ray off."
        )
    if (
        velocity_xy is None
        and route_xy is None
        and distance < APPEAR_RAY_VISIBILITY_LIMIT_M
    ):
        raise ValueError(
            f"appearance distance {distance} m is a NEAR rung but no alignment "
            "direction was supplied, so the bearing would be drawn uniformly "
            f"over all {int(ray_count)} rays -- a cylinder that close astern is "
            "not an obstacle, and the rung would measure nothing while looking "
            "easy. Pass velocity_xy (cfg appear_velocity_aligned) or route_xy "
            f"(cfg appear_route_aligned) below {APPEAR_RAY_VISIBILITY_LIMIT_M} m. "
            "This guard exists because the silent-clamp bug hid exactly this "
            "class of mistake for two days."
        )
    order = [int(k) for k in candidate_ray_indices]
    # Alignment source, in precedence order.  route_xy is the direction the
    # TASK still requires; velocity_xy is the direction the boat happens to be
    # going.  They diverge whenever the boat is detouring or turning onto a
    # gate -- measured at 31.6 degrees, i.e. about three rays, on a gated wall
    # whose opening is offset from the goal bearing.  An obstacle on the
    # velocity vector can be stepped around by changing course; one on the
    # route has to be dealt with, so the two are different interventions and
    # the axis reports which one it used.
    align = route_xy if route_xy is not None else velocity_xy
    if align is not None:
        velocity = np.asarray(align, dtype=np.float64).reshape(2)
        speed = float(np.linalg.norm(velocity))
        # A caller that asked for an aligned bearing must never silently get a
        # uniform one.  The old test here was `speed > 1e-6`, a divide-by-zero
        # guard masquerading as a validity guard: below it the candidate order
        # stayed the caller's uniform permutation, and strict_ray then took
        # order[0] -- ONE RANDOM RAY OF 36.  Executed: velocity (0, 0),
        # distance 3.0, strict_ray -> ray 17, centre (-2.95, 0.52), directly
        # ASTERN, recorded as a successful velocity-aligned appearance.  The
        # graded version is worse because it is common: at 0.05 m/s the
        # "direction of travel" is drift, and the run still reports itself
        # aligned.  Skipping is the honest outcome -- a near-stationary boat is
        # a legitimate episode state, not a config error, and the env already
        # records skips.  route_xy is a unit vector by construction and always
        # clears this; only a real velocity can fail it.
        if route_xy is None and speed < APPEAR_MIN_ALIGN_SPEED_MS:
            return None
        if speed > 1.0e-6:
            # Angle of travel in the SAME frame the ray formula uses.
            psi = math.atan2(
                float(np.dot(velocity, left)), float(np.dot(velocity, forward))
            )
            step = 2.0 * math.pi / float(ray_count)
            aligned = int(round(psi / step)) % int(ray_count)
            spiral = [aligned]
            for offset in range(1, int(ray_count) // 2 + 1):
                spiral.append((aligned + offset) % int(ray_count))
                spiral.append((aligned - offset) % int(ray_count))
            seen: set[int] = set()
            deduped = []
            for k in spiral:
                if k not in seen:
                    seen.add(k)
                    deduped.append(k)
            # Keep the caller's attempt budget: it is what bounds resampling.
            order = deduped[: max(1, len(order))]
    if strict_ray:
        order = order[:1]

    attempts = 0
    for ray_index in order:
        attempts += 1
        index = int(ray_index)
        if not 0 <= index < int(ray_count):
            raise ValueError(
                f"candidate ray index {index} outside [0, {ray_count})"
            )
        angle = 2.0 * math.pi * index / float(ray_count)
        direction = math.cos(angle) * forward + math.sin(angle) * left
        center = boat + distance * direction
        # Hull overlap: implied by the distance floor, but asserted
        # rather than assumed (the checker-vs-reality lesson).
        if float(np.linalg.norm(center - boat)) - float(radius_m) - float(
            half_beam_m
        ) <= 0.0:
            continue
        # Goal disc: the appeared cylinder may not overlap the success region.
        if float(np.linalg.norm(center - goal)) <= float(radius_m) + float(
            goal_radius_m
        ):
            continue
        # The samplers' own admission: BFS at planning inflation over the
        # POST-appearance field, from the current boat position to the goal.
        post_centers = (
            np.vstack((field_centers, center[None, :]))
            if field_centers.size
            else center[None, :].copy()
        )
        post_radii = np.concatenate((field_radii, (float(radius_m),)))
        remaining = bfs_geodesic_length(
            boat,
            goal,
            post_centers,
            post_radii,
            cell_m=float(bfs_cell_m),
            inflation_m=float(inflation_m),
        )
        if remaining is None:
            continue
        # --- Occlusion (added 2026-08-31, master TODO section 8.3 item 5) --
        #
        # Ray ALIGNMENT is not visibility.  The sensor reports the NEAREST
        # forward intersection along each ray, so an existing cylinder sitting
        # between the boat and the new one hides it completely: the policy sees
        # no change at the instant the obstacle appears, and the episode is
        # scored as if it had been given a warning it never received.
        #
        # Checked with ray_circle_ranges itself rather than a re-derivation, so
        # this uses the same intersection maths the observation does -- the
        # checker-vs-reality lesson, again.
        if field_centers.size:
            near_surface = float(distance) - float(radius_m)
            pre_range = float(
                ray_circle_ranges(
                    torch.as_tensor(boat, dtype=torch.float64).reshape(1, 2),
                    torch.as_tensor(direction, dtype=torch.float64).reshape(1, 1, 2),
                    torch.as_tensor(field_centers, dtype=torch.float64).reshape(1, -1, 2),
                    torch.as_tensor(field_radii, dtype=torch.float64).reshape(1, -1),
                    max_range_m=float(ray_max_range_m),
                )[0, 0]
            )
            if pre_range <= near_surface:
                continue
        return AppearancePlan(
            center=center,
            radius_m=float(radius_m),
            ray_index=index,
            azimuth_rad=float(math.atan2(direction[1], direction[0])),
            distance_m=float(distance),
            attempts=attempts,
        )
    return None


__all__ = [
    "APPEAR_MAX_DISTANCE_M",
    "APPEAR_MIN_ALIGN_SPEED_MS",
    "APPEAR_MIN_DISTANCE_M",
    "APPEAR_RAY_VISIBILITY_LIMIT_M",
    "APPEAR_STOP_DISTANCE_M",
    "appear_min_distance_m",
    "AppearancePlan",
    "BAND_FORTRESS_MAX_OBSTACLES",
    "COLLISION_MARGIN_M",
    "DIFFICULTIES",
    "DOUBLE_RING_GOAL_DISTANCE_RANGE_M",
    "DOUBLE_RING_MIN_GAP_OFFSET_RAD",
    "DOUBLE_RING_OUTER_RADIUS_M",
    "DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M",
    "DoubleRingGapSpecs",
    "ENDPOINT_CLEAR_RADIUS_M",
    "FORCED_GATE_BEAMS",
    "DOUBLE_RING_FORTRESS_MAX_INNER_OBSTACLES",
    "DOUBLE_RING_FORTRESS_MAX_OUTER_OBSTACLES",
    "FORTRESS_MAX_OBSTACLES",
    "FORTRESS_APERTURE_M",
    "FORTRESS_SPAWN_RADIUS_RANGE_M",
    "GRID_CELL_M",
    "HALF_BEAM_M",
    "HULL_BEAM_M",
    "ICEBERG_EDGE_GAP_M",
    "ICEBERG_GOAL_DISTANCE_RANGE_M",
    "ICEBERG_TIERS",
    "GateSpec",
    "HazardLayout",
    "MAX_OBSTACLE_RADIUS_M",
    "MIN_OBSTACLE_RADIUS_M",
    "OBSTACLE_INFLATION_M",
    "RING_RADIUS_M",
    "RING_SEALED_OVERLAP_M",
    "RingGapSpec",
    "analytic_min_clearance",
    "bfs_geodesic_length",
    "plan_obstacle_appearance",
    "difficulty_for_level",
    "direct_segment_blocked",
    "geodesic_descent_directions",
    "geodesic_distance_field",
    "geodesic_field_value",
    "geodesic_waypoint",
    "inflated_radii",
    "minimum_pairwise_inflated_gap",
    "ray_circle_ranges",
    "route_direction_xy",
    "route_geodesic_length",
    "sample_forced_crossing_layout",
    "sample_iceberg_layout",
    "sample_band_fortress_layout",
    "sample_double_ring_fortress_layout",
    "sample_double_ring_layout",
    "sample_layout",
    "sample_open_basin_layout",
    "sample_ring_fortress_layout",
    "start_goal_disks_clear",
    "wall_segment_cylinders",
]


# ---------------------------------------------------------------------------
# P5 layer three: the moving-obstacle basin (added 2026-08-31)
# ---------------------------------------------------------------------------
#
# Three segments in a closed basin, in order along +X:
#
#   (1) two staggered rows of cylinders sliding sideways  -- tests WHEN to go
#   (buffer) MOVING_BUFFER_M of empty water, so the two segments can be
#           attributed separately
#   (2) one cylinder orbiting a fixed centre at constant rate -- tests
#           anticipating circular motion
#   (3) a clear run-out to the goal
#
# WHY THIS IS CHEAP TO BUILD.  Both consumers of obstacle geometry read the
# buffers analytically every step and neither queries the physics engine:
# ``ray_circle_ranges`` for the 36-ray sensor and ``analytic_min_clearance``
# for contact.  So "moving" is one tensor write per step, not a new collision
# body, not a new asset, and not a PhysX change.
#
# DOSE.  Both segments express their dose as a fraction of the boat's own top
# speed, so segment (1) at 0.55 and segment (2) at 0.55 are the same physical
# demand and can be read against each other:
#
#   mu = u / v_max          (row translation speed)
#   nu = omega * R / v_max  (orbiting cylinder tangential speed)
#
# FAIRNESS, SEGMENT (1).  Clearing one row means covering
# ``L = 2 * r + 2 * half_beam`` of depth.  A boat that enters a gap dead
# centre and never moves sideways has ``(W - beam) / 2`` of margin on each
# side, so it survives without any lateral tracking exactly when
#
#     u <= v * (W - beam) / (2 * L)
#
# which at top speed and a three-beam gap is u* = 1.076 m/s, i.e. mu* = 0.360.
# The ladder deliberately straddles that point: below it the task is about
# CHOOSING A MOMENT, above it the boat has to chase the gap sideways while
# still making headway, which is a different skill.  The ladder must cross the
# transition or the whole axis lives in one regime -- the mistake the
# appearance axis made by never leaving its own.
#
# FAIRNESS ABOVE mu*.  The design note that this file implements gave no
# constructive strategy above the transition, which would have left the top
# rungs unproven.  Here is one, and ``moving_basin_tracking_feasible`` checks
# it as an executable assertion rather than a paragraph.  Hold a constant
# heading ``psi`` with ``v_max * sin(psi) = u`` so the boat tracks the gap
# exactly sideways while advancing at ``v_max * cos(psi)``.  The crossing then
# takes ``L / (v_max * cos(psi))`` and needs the row to keep a constant
# velocity throughout, i.e. no reversal mid-crossing.  Reversals are half a
# spatial period apart in time, ``P / u``, so the strategy exists when
#
#     L / (v_max * cos(psi)) < P / u,  sin(psi) = u / v_max
#
# This is a KINEMATIC proof: a point mass that can hold v_max on any heading.
# It does not model the turn rate needed to acquire that heading, so it is a
# necessary condition for fairness, not a sufficient one for any controller.
# Stated plainly because the appearance axis was certified for two days on an
# argument whose direction had been reversed.
#
# FAIRNESS, SEGMENT (2).  Crossing the orbit disc takes ``2 R / v_max``.  The
# cylinder only blocks over a small arc, so if the period exceeds twice the
# crossing time there is always a window in which the boat can wait outside
# and then cross while the cylinder is on the far side.  That bound is
# nu < 1.57; the top rung is 0.80, half of it.  The low rungs are NOT testing
# anticipation -- at nu = 0.12 the cylinder turns 14 degrees during a crossing
# and is effectively static.  They are light doses, and reports must not
# claim otherwise.

MOVING_GAP_BEAMS = 3.0          # gap width in hull beams; crossing tier 2
MOVING_ROW_SEPARATION_M = 8.0   # along-track distance between the two rows
MOVING_BUFFER_M = 10.0          # empty water between segment 1 and segment 2
MOVING_ORBIT_RADIUS_M = 6.0     # orbit radius of the segment 2 cylinder
MOVING_RUNOUT_M = 8.0           # clear water between the orbit and the goal
MOVING_APPROACH_M = 6.0         # clear water between the start and row one
MOVING_TOP_SPEED_MS = 2.9910    # repo integrator: 80 N against 9.1v + 5.9v^2
MOVING_ROW_RADIUS_M = MIN_OBSTACLE_RADIUS_M
# The orbit of segment 2 reaches R + r = 6.8 m from the basin axis, and the
# perimeter wall cylinders have radius FORCED_PERIMETER_RADIUS_M = 1.20 m with
# their centres on the half-width line, so their inner SURFACE is at
# half_width - 1.20.  A half width of 7.0 m would therefore put the orbiting
# cylinder 1.0 m INSIDE the wall.  The floor is 8.0 m and this range clears it
# by 1.5 to 3.5 m, which is also enough for the boat to pass outside the orbit
# if it chooses to, so segment 2 measures timing rather than a forced squeeze.
MOVING_HALF_WIDTH_RANGE_M = (9.5, 11.5)
MOVING_END_MARGIN_RANGE_M = (3.6, 6.0)

# Dose ladder.  Rung 0 is the parent baseline and MUST be byte-identical to
# the static basin; rung 3 is the transition derived above and may not be
# dropped from any table.
MOVING_DOSE_LADDER: tuple[float, ...] = (0.0, 0.12, 0.24, 0.36, 0.55, 0.80)

KIND_STATIC = 0
KIND_TRANSLATE = 1
KIND_ORBIT = 2


@dataclass(frozen=True)
class MovingSpec:
    """Per-cylinder motion law for one moving-basin episode.

    ``kind`` and ``params`` are parallel to the layout's ``centers``, so a
    consumer never has to match cylinders up by position.  ``params`` has seven
    columns whose meaning depends on ``kind``:

    * ``KIND_STATIC``    -- unused; the centre in the layout is the position.
    * ``KIND_TRANSLATE`` -- ``[base_x, base_y, dir_x, dir_y, phase_m,
      speed_ms, span_m]``.  The cylinder oscillates along the UNIT vector
      ``dir`` about ``base``, as a triangle wave of half-amplitude
      ``span_m / 2``.  A triangle wave rather than a wrapping saw, because a
      saw makes a cylinder teleport at the wrap and that breaks contact and
      clearance continuity mid episode.
    * ``KIND_ORBIT``     -- ``[cx, cy, radius_m, omega_rads, phase_rad, 0, 0]``.

    The motion is written as a DIRECTION rather than as an axis-aligned pair
    of bounds for one reason: every layout in this tree is rotated by a
    per-episode angle drawn from the scenario protocol
    (``layout_rotation_angle``), and an axis-aligned parameterisation silently
    keeps translating along world y after the scene has been turned.  With a
    direction vector the rotation is exact, and ``rotate_moving_params``
    performs it.
    """

    kind: np.ndarray
    params: np.ndarray
    mu: float
    nu: float
    gap_width_m: float
    row_period_m: float
    translate_speed_ms: float
    orbit_omega_rads: float
    row_x: tuple[float, float]
    orbit_center: np.ndarray

    @property
    def moves(self) -> bool:
        return bool(np.any(self.kind != KIND_STATIC))


def moving_row_period_m(
    gap_width_m: float, radius_m: float = MOVING_ROW_RADIUS_M
) -> float:
    """Centre-to-centre spacing of a row whose free gap is ``gap_width_m``."""
    return float(gap_width_m) + 2.0 * float(radius_m)


def moving_transition_mu(
    gap_width_m: float,
    *,
    radius_m: float = MOVING_ROW_RADIUS_M,
    half_beam_m: float = HALF_BEAM_M,
    beam_m: float = HULL_BEAM_M,
    top_speed_ms: float = MOVING_TOP_SPEED_MS,
) -> float:
    """The dose at which passing without any lateral tracking stops working.

    Below it the task asks WHEN to enter; above it the boat must chase the
    gap sideways.  Returned as a fraction of top speed so it can be compared
    against a ladder rung directly.
    """
    depth = 2.0 * float(radius_m) + 2.0 * float(half_beam_m)
    margin = float(gap_width_m) - float(beam_m)
    if margin <= 0.0:
        raise ValueError(
            f"gap {gap_width_m} m does not admit a {beam_m} m beam at all"
        )
    return float(margin / (2.0 * depth))


def moving_basin_tracking_feasible(
    mu: float,
    gap_width_m: float,
    *,
    radius_m: float = MOVING_ROW_RADIUS_M,
    half_beam_m: float = HALF_BEAM_M,
    beam_m: float = HULL_BEAM_M,
) -> tuple[bool, float, float]:
    """Constructive feasibility of one row at dose ``mu``.

    Returns ``(feasible, crossing_time_s, time_between_reversals_s)``.  See
    the FAIRNESS ABOVE mu* note at the head of this section: the boat holds a
    constant heading that tracks the gap exactly, and the only thing that can
    defeat it is the row reversing mid-crossing.
    """
    mu = float(mu)
    if not 0.0 <= mu < 1.0:
        raise ValueError(f"mu must lie in [0, 1); got {mu}")
    depth = 2.0 * float(radius_m) + 2.0 * float(half_beam_m)
    if mu == 0.0:
        return True, depth / MOVING_TOP_SPEED_MS, math.inf
    forward = MOVING_TOP_SPEED_MS * math.sqrt(1.0 - mu * mu)
    crossing = depth / forward
    period_half = moving_row_period_m(gap_width_m, radius_m) / (
        mu * MOVING_TOP_SPEED_MS
    )
    return bool(crossing < period_half), float(crossing), float(period_half)


def moving_orbit_window_bound(
    *,
    radius_m: float = MOVING_ORBIT_RADIUS_M,
    top_speed_ms: float = MOVING_TOP_SPEED_MS,
) -> float:
    """Largest ``nu`` at which a wait-then-cross strategy still exists.

    The orbit period must exceed twice the time to cross the disc, so that a
    boat waiting outside always gets a full crossing window with the cylinder
    on the far side.
    """
    crossing = 2.0 * float(radius_m) / float(top_speed_ms)
    omega_bound = math.pi / crossing
    return float(omega_bound * float(radius_m) / float(top_speed_ms))


def moving_basin_positions(
    base_centers: np.ndarray,
    kind: np.ndarray,
    params: np.ndarray,
    t_s: float,
) -> np.ndarray:
    """Cylinder centres at time ``t_s``, in the layout's local frame.

    Pure function of ``t_s``: the same time gives the same positions, which is
    what makes an episode reproducible from its scenario draw alone.  At
    ``t_s == 0`` it must return ``base_centers`` unchanged for every kind, and
    a test pins that.
    """
    centers = np.array(base_centers, dtype=np.float64, copy=True)
    kind = np.asarray(kind)
    params = np.asarray(params, dtype=np.float64)
    t = float(t_s)

    rows = np.nonzero(kind == KIND_TRANSLATE)[0]
    if rows.size:
        base = params[rows, 0:2]
        direction = params[rows, 2:4]
        phase = params[rows, 4]
        speed = params[rows, 5]
        span = params[rows, 6]
        # Triangle wave: fold the sawtooth of period 2*span back on itself, so
        # the cylinder reverses at each end instead of jumping.
        s = np.mod(speed * t + phase, 2.0 * span)
        offset = np.where(s <= span, s, 2.0 * span - s) - 0.5 * span
        centers[rows] = base + direction * offset[:, None]

    orbits = np.nonzero(kind == KIND_ORBIT)[0]
    if orbits.size:
        cx = params[orbits, 0]
        cy = params[orbits, 1]
        radius = params[orbits, 2]
        omega = params[orbits, 3]
        phase = params[orbits, 4]
        theta = omega * t + phase
        centers[orbits, 0] = cx + radius * np.cos(theta)
        centers[orbits, 1] = cy + radius * np.sin(theta)
    return centers


def _triangle(value: float, span: float) -> float:
    """Fold a sawtooth of period ``2 * span`` back on itself, into [0, span]."""
    s = math.fmod(float(value), 2.0 * span)
    if s < 0.0:
        s += 2.0 * span
    return s if s <= span else 2.0 * span - s


def rotate_moving_params(
    params: np.ndarray, kind: np.ndarray, angle_rad: float
) -> np.ndarray:
    """Turn a motion description into a frame rotated by ``angle_rad``.

    Every layout in this tree is rotated per episode by an angle drawn from
    the scenario protocol, and that rotation is applied to the CENTRES.  The
    motion has to be turned with them, or the cylinders keep sliding along the
    old world axis while the scene faces a new way.  The transform is exact:
    rotating the params and then evaluating gives the same points as
    evaluating and then rotating, and a test pins that.
    """
    kind = np.asarray(kind)
    out = np.array(params, dtype=np.float64, copy=True)
    c, s = math.cos(float(angle_rad)), math.sin(float(angle_rad))
    rot = np.array(((c, -s), (s, c)))
    rows = np.nonzero(kind == KIND_TRANSLATE)[0]
    if rows.size:
        out[rows, 0:2] = out[rows, 0:2] @ rot.T
        out[rows, 2:4] = out[rows, 2:4] @ rot.T
    orbits = np.nonzero(kind == KIND_ORBIT)[0]
    if orbits.size:
        out[orbits, 0:2] = out[orbits, 0:2] @ rot.T
        out[orbits, 4] = out[orbits, 4] + float(angle_rad)
    return out


MOVING_ENDPOINT_CLEAR_M = 2.50   # same rule as the crossing basin: a disk this
                                 # big around start and goal stays empty at
                                 # EVERY sampled time, not just at t = 0
MOVING_MAX_CYLINDERS = 128       # sampler refuses above this; the cfg cap must
                                 # leave room on top for the appearance axis if
                                 # the two are ever combined
MOVING_ADMISSION_SAMPLES = 12    # times sampled per period for the feasibility
                                 # check; a moving field can be feasible at
                                 # t = 0 and sealed a second later, which is a
                                 # failure mode the static samplers cannot have

# How the two rows relate to each other.  The design note asked for a half
# period phase difference so that "after passing the first row the boat cannot
# drive straight on, it has to re-aim at the second row's gap".  Implementing
# that literally does NOT produce that property, and the arithmetic says why.
# With a triangle wave, tri(theta + P) = P - tri(theta), so combining a half
# period phase difference with a half period SPATIAL stagger makes the actual
# relative offset 3P/2 - 2*tri(theta), which sweeps the whole period instead of
# staying put.  Measured on a real sampled layout at mu = 0.55: the offset
# wanders over 0.004 to 2.145 m and a straight shot clears both rows in 83
# percent of the cycle, because a straight shot only needs the two gaps to
# overlap by one beam, i.e. an offset within W - beam = 1.798 m.
#
# Rigid in-phase translation with a fixed half period stagger holds the offset
# at exactly P/2 = 2.1485 m, which is above 1.798 m at every instant, so the
# re-aim is genuinely forced.  That is the property the design asked for, so it
# is the default.  The anti-phase variant is kept selectable because "the rows
# move against each other" is a different and also interesting mechanic, but it
# must not be described as forcing a re-aim.
MOVING_ANTI_PHASE_ROWS = False


def _moving_row(
    x: float,
    centre_y: float,
    period_m: float,
    half_width: float,
    radius_m: float,
    speed_ms: float,
    phase_m: float,
) -> tuple[list, list, list, list]:
    """One comb of cylinders that slides sideways as a rigid body.

    Each cylinder oscillates by half a period either side of its own base, so
    the comb as a whole sweeps exactly one spatial period and every y is both
    covered and open at some point in the cycle.  A triangle wave rather than
    a wrapping saw: a saw would teleport a cylinder at the wrap, which breaks
    contact and clearance continuity mid episode.

    The comb is generated PAST the basin wall on both sides so that its end
    cylinders overlap the perimeter.  Without that the boat could slip between
    the last cylinder and the wall, which is not a gap the difficulty rating
    accounts for.
    """
    reach = half_width + 1.5 * period_m
    k_lo = int(math.floor((-reach - centre_y) / period_m))
    k_hi = int(math.ceil((reach - centre_y) / period_m))
    span = period_m
    offset0 = _triangle(phase_m, span) - 0.5 * span
    centers, radii, kind, params = [], [], [], []
    for k in range(k_lo, k_hi + 1):
        base_y = centre_y + k * period_m
        centers.append(np.array((x, base_y + offset0), dtype=np.float64))
        radii.append(float(radius_m))
        if speed_ms > 0.0:
            kind.append(KIND_TRANSLATE)
            params.append((x, base_y, 0.0, 1.0, phase_m, speed_ms, span))
        else:
            kind.append(KIND_STATIC)
            params.append((0.0,) * 7)
    return centers, radii, kind, params


def sample_moving_basin_layout(
    level: int,
    rng: np.random.Generator | None = None,
    *,
    mu: float = 0.0,
    nu: float = 0.0,
    max_attempts: int = 60,
    gap_width_override_m: float | None = None,
    anti_phase: bool = MOVING_ANTI_PHASE_ROWS,
) -> tuple[HazardLayout, MovingSpec]:
    """Sample the three-segment moving-obstacle basin.

    Admission, in the order it rejects most cheaply:

    1. every dose is inside its own fairness bound -- ``mu`` below 1 and
       constructively feasible per ``moving_basin_tracking_feasible``, ``nu``
       below ``moving_orbit_window_bound``;
    2. the cylinder count fits ``MOVING_MAX_CYLINDERS``;
    3. the endpoint disks are clear AT EVERY SAMPLED TIME, not just at t = 0;
    4. BFS at planning inflation finds a route AT EVERY SAMPLED TIME.

    Points 3 and 4 are the whole reason this sampler is not a copy of the
    crossing one.  A field of moving obstacles can be perfectly feasible at
    the instant it is sampled and sealed a second later, and a check at t = 0
    alone would certify a layout the boat cannot solve.  ``level`` is accepted
    and recorded for interface parity with the other samplers; the geometry of
    this task is fixed and the DOSE, not the level, carries the difficulty.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    mu = float(mu)
    nu = float(nu)
    if not 0.0 <= mu < 1.0:
        raise ValueError(f"mu must lie in [0, 1); got {mu}")
    if nu < 0.0:
        raise ValueError(f"nu must be non-negative; got {nu}")
    orbit_bound = moving_orbit_window_bound()
    if nu >= orbit_bound:
        raise ValueError(
            f"nu={nu} is at or above the wait-then-cross bound {orbit_bound:.3f}; "
            "above it no strategy is guaranteed to exist and the rung would be "
            "unsolvable rather than merely hard"
        )
    gap_width = (
        float(gap_width_override_m)
        if gap_width_override_m is not None and gap_width_override_m > 0.0
        else float(MOVING_GAP_BEAMS * HULL_BEAM_M)
    )
    if gap_width <= HULL_BEAM_M:
        raise ValueError(
            f"gap {gap_width:.3f} m does not admit a {HULL_BEAM_M} m beam"
        )
    feasible, crossing_s, reversal_s = moving_basin_tracking_feasible(mu, gap_width)
    if not feasible:
        raise ValueError(
            f"mu={mu} is not constructively feasible for a {gap_width:.3f} m gap: "
            f"a tracking crossing takes {crossing_s:.3f} s but the row reverses "
            f"every {reversal_s:.3f} s, so no constant-heading pass exists"
        )

    period_m = moving_row_period_m(gap_width)
    speed_ms = mu * MOVING_TOP_SPEED_MS
    omega = nu * MOVING_TOP_SPEED_MS / MOVING_ORBIT_RADIUS_M
    difficulty = difficulty_for_level(level)
    rng = np.random.default_rng() if rng is None else rng
    start = np.zeros(2, dtype=np.float64)

    row1_x = MOVING_APPROACH_M
    row2_x = row1_x + MOVING_ROW_SEPARATION_M
    orbit_cx = row2_x + MOVING_BUFFER_M + MOVING_ORBIT_RADIUS_M
    goal_x = orbit_cx + MOVING_ORBIT_RADIUS_M + MOVING_RUNOUT_M
    goal = np.array((goal_x, 0.0), dtype=np.float64)

    period_row = (2.0 * period_m / speed_ms) if speed_ms > 0.0 else 0.0
    period_orbit = (2.0 * math.pi / omega) if omega > 0.0 else 0.0

    def sample_times() -> list:
        spans = [p for p in (period_row, period_orbit) if p > 0.0]
        if not spans:
            return [0.0]
        out = []
        for p in spans:
            out.extend(p * i / MOVING_ADMISSION_SAMPLES
                       for i in range(MOVING_ADMISSION_SAMPLES))
        return sorted(set(out))

    times = sample_times()

    for attempt in range(1, max_attempts + 1):
        half_width = float(rng.uniform(*MOVING_HALF_WIDTH_RANGE_M))
        x_back = -float(rng.uniform(*MOVING_END_MARGIN_RANGE_M))
        x_front = goal_x + float(rng.uniform(*MOVING_END_MARGIN_RANGE_M))
        phase1 = float(rng.uniform(0.0, 2.0 * period_m))
        # The SPATIAL stagger of half a period is what stops a boat driving
        # straight through both rows on one bearing, and it only holds if the
        # rows translate together.  See the MOVING_ANTI_PHASE_ROWS note.
        phase2 = phase1 + period_m if anti_phase else phase1
        orbit_phase = float(rng.uniform(0.0, 2.0 * math.pi))

        centers, radii = _basin_cylinders(
            x_back, x_front, half_width, 0.0, 0.0, 0.0, include_bulkhead=False
        )
        centers = [np.asarray(c, dtype=np.float64) for c in centers]
        radii = [float(r) for r in radii]
        kind = [KIND_STATIC] * len(radii)
        params = [(0.0,) * 7] * len(radii)

        for x, centre_y, phase in (
            (row1_x, 0.0, phase1),
            (row2_x, 0.5 * period_m, phase2),
        ):
            c, r, k, p = _moving_row(
                x, centre_y, period_m, half_width,
                MOVING_ROW_RADIUS_M, speed_ms, phase,
            )
            centers.extend(c); radii.extend(r); kind.extend(k); params.extend(p)

        orbit_xy = np.array(
            (orbit_cx + MOVING_ORBIT_RADIUS_M * math.cos(orbit_phase),
             MOVING_ORBIT_RADIUS_M * math.sin(orbit_phase)),
            dtype=np.float64,
        )
        centers.append(orbit_xy)
        radii.append(float(MOVING_ROW_RADIUS_M))
        if omega > 0.0:
            kind.append(KIND_ORBIT)
            params.append(
                (orbit_cx, 0.0, MOVING_ORBIT_RADIUS_M, omega, orbit_phase, 0.0, 0.0)
            )
        else:
            kind.append(KIND_STATIC)
            params.append((0.0,) * 7)

        base_centers = np.vstack(centers)
        all_radii = np.asarray(radii, dtype=np.float64)
        kind_arr = np.asarray(kind, dtype=np.int64)
        param_arr = np.asarray(params, dtype=np.float64)
        if base_centers.shape[0] > MOVING_MAX_CYLINDERS:
            raise ValueError(
                f"moving basin needs {base_centers.shape[0]} cylinders but "
                f"MOVING_MAX_CYLINDERS is {MOVING_MAX_CYLINDERS}; widen the cap "
                "on the cfg as well before raising this"
            )

        ok = True
        geodesic0 = None
        for t in times:
            moved = moving_basin_positions(base_centers, kind_arr, param_arr, t)
            if not start_goal_disks_clear(
                start, goal, moved, all_radii,
                clear_radius_m=MOVING_ENDPOINT_CLEAR_M, inflation_m=0.0,
            ):
                ok = False
                break
            g = bfs_geodesic_length(
                start, goal, moved, all_radii, cell_m=FORCED_SEAL_CELL_M
            )
            if g is None:
                ok = False
                break
            if geodesic0 is None:
                geodesic0 = float(g)
        if not ok:
            continue

        layout = HazardLayout(
            level=difficulty.level,
            requested_obstacle_count=int(base_centers.shape[0]),
            start=start,
            goal=goal,
            centers=base_centers,
            radii=all_radii,
            geodesic_length=float(geodesic0),
            direct_blocked=True,
            attempts=attempt,
        )
        spec = MovingSpec(
            kind=kind_arr,
            params=param_arr,
            mu=mu,
            nu=nu,
            gap_width_m=gap_width,
            row_period_m=period_m,
            translate_speed_ms=speed_ms,
            orbit_omega_rads=omega,
            row_x=(row1_x, row2_x),
            orbit_center=np.array((orbit_cx, 0.0), dtype=np.float64),
        )
        return layout, spec

    raise RuntimeError(
        f"moving basin layout not admitted in {max_attempts} attempts at "
        f"mu={mu}, nu={nu}; this should not happen for a dose inside the "
        "fairness bounds and indicates a geometry regression"
    )
