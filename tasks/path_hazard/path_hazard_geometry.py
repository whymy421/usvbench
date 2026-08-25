"""Isaac-free geometry and layout oracle for Path Hazard.

Only the Python standard library, NumPy, and Torch are imported so the
procedural interaction guarantee can be acceptance-tested without Isaac Lab.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import warnings

import numpy as np
import torch


NUM_SEGMENTS = 4
SEGMENT_LENGTH_MIN_M = 12.0
SEGMENT_LENGTH_MAX_M = 20.0
HEADING_CHANGE_MAX_DEG = 60.0
GOAL_RADIUS_M = 2.0

REQUESTED_OBSTACLE_COUNT = 6
REQUIRED_ON_LINE_BLOCKERS = 3
MIN_OBSTACLE_RADIUS_M = 0.8
MAX_OBSTACLE_RADIUS_M = 1.5
ON_LINE_FRACTION_MIN = 0.30
ON_LINE_FRACTION_MAX = 0.70
ON_LINE_OFFSET_MAX_M = 0.5
SCATTER_ROUTE_DISTANCE_MAX_M = 6.0

HALF_BEAM_M = 0.45
COLLISION_MARGIN_M = 0.20
OBSTACLE_INFLATION_M = HALF_BEAM_M + COLLISION_MARGIN_M
GATE_CLEAR_RADIUS_M = 3.0
GRID_CELL_M = 0.5


@dataclass(frozen=True)
class PathHazardLayout:
    """One accepted route and obstacle layout in an origin-relative frame."""

    route_points: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    on_line_mask: np.ndarray
    blocker_segment_indices: np.ndarray
    leg_geodesic_lengths: np.ndarray
    requested_obstacle_count: int
    attempts: int

    @property
    def waypoints(self) -> np.ndarray:
        return self.route_points[1:]

    @property
    def segment_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.route_points, axis=0), axis=1)

    @property
    def route_length(self) -> float:
        return float(np.sum(self.segment_lengths))

    @property
    def obstacle_count(self) -> int:
        return int(self.radii.shape[0])

    @property
    def on_line_blocker_count(self) -> int:
        return int(np.count_nonzero(self.on_line_mask))

    @property
    def chain_feasible(self) -> bool:
        return bool(
            len(self.leg_geodesic_lengths) == NUM_SEGMENTS
            and np.all(np.isfinite(self.leg_geodesic_lengths))
        )


def inflated_radii(
    radii: np.ndarray, inflation_m: float = OBSTACLE_INFLATION_M
) -> np.ndarray:
    """Inflate cylinders by hull half-beam plus feasibility margin."""
    return np.asarray(radii, dtype=np.float64) + float(inflation_m)


def sample_route(rng: np.random.Generator) -> np.ndarray:
    """Sample path_following's four-segment procedural polyline."""
    lengths = rng.uniform(
        SEGMENT_LENGTH_MIN_M, SEGMENT_LENGTH_MAX_M, size=NUM_SEGMENTS
    )
    headings = np.empty(NUM_SEGMENTS, dtype=np.float64)
    headings[0] = rng.uniform(0.0, 2.0 * math.pi)
    heading_changes = rng.uniform(
        -math.radians(HEADING_CHANGE_MAX_DEG),
        math.radians(HEADING_CHANGE_MAX_DEG),
        size=NUM_SEGMENTS - 1,
    )
    headings[1:] = headings[0] + np.cumsum(heading_changes)
    vectors = np.column_stack((lengths * np.cos(headings), lengths * np.sin(headings)))
    return np.vstack((np.zeros((1, 2), dtype=np.float64), np.cumsum(vectors, axis=0)))


def _point_segment_projection(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[float, float]:
    segment = end - start
    norm_sq = float(np.dot(segment, segment))
    if norm_sq <= 0.0:
        raise ValueError("route segments must have positive length")
    fraction = float(np.clip(np.dot(point - start, segment) / norm_sq, 0.0, 1.0))
    closest = start + fraction * segment
    return fraction, float(np.linalg.norm(point - closest))


def distance_to_route(point: np.ndarray, route_points: np.ndarray) -> float:
    """Minimum Euclidean distance from a point to the finite route polyline."""
    return min(
        _point_segment_projection(point, start, end)[1]
        for start, end in zip(route_points[:-1], route_points[1:])
    )


def count_on_line_blockers(
    route_points: np.ndarray,
    centers: np.ndarray,
    *,
    fraction_min: float = ON_LINE_FRACTION_MIN,
    fraction_max: float = ON_LINE_FRACTION_MAX,
    offset_max_m: float = ON_LINE_OFFSET_MAX_M,
) -> int:
    """Count obstacles satisfying the benchmark's on-segment placement rule."""
    count = 0
    for center in np.asarray(centers, dtype=np.float64):
        matches = False
        for start, end in zip(route_points[:-1], route_points[1:]):
            fraction, distance = _point_segment_projection(center, start, end)
            if (
                fraction_min - 1.0e-9 <= fraction <= fraction_max + 1.0e-9
                and distance <= offset_max_m + 1.0e-9
            ):
                matches = True
                break
        count += int(matches)
    return count


def gate_disks_clear(
    route_points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    clear_radius_m: float = GATE_CLEAR_RADIUS_M,
    inflation_m: float = OBSTACLE_INFLATION_M,
) -> bool:
    """Require inflated hazards to stay outside 3 m disks at start and gates."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(centers) == 0:
        return True
    distances = np.linalg.norm(
        centers[:, None, :] - np.asarray(route_points, dtype=np.float64)[None, :, :],
        axis=-1,
    )
    required = inflated_radii(radii, inflation_m)[:, None] + float(clear_radius_m)
    return bool(np.all(distances >= required))


def all_obstacles_within_route_band(
    route_points: np.ndarray,
    centers: np.ndarray,
    *,
    max_distance_m: float = SCATTER_ROUTE_DISTANCE_MAX_M,
) -> bool:
    """Whether every obstacle axis lies within the configured route band."""
    return all(
        distance_to_route(center, route_points) <= max_distance_m + 1.0e-9
        for center in np.asarray(centers, dtype=np.float64)
    )


def _inflated_obstacles_separated(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    minimum_gap_m: float = 0.25,
) -> bool:
    """Reject overlapping inflated hazards so the intended detours remain legible."""
    if len(radii) < 2:
        return True
    radii_i = inflated_radii(radii)
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    required = radii_i[:, None] + radii_i[None, :] + minimum_gap_m
    upper = np.triu_indices(len(radii), k=1)
    return bool(np.all(distances[upper] >= required[upper]))


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
    """Eight-connected occupancy BFS using inflated analytic cylinders."""
    if cell_m <= 0.0 or padding_m <= 0.0:
        raise ValueError("cell_m and padding_m must be positive")

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    radii_i = inflated_radii(np.asarray(radii, dtype=np.float64), inflation_m)

    points = np.vstack((start, goal, centers)) if len(centers) else np.vstack((start, goal))
    if len(centers):
        x_min = min(float(np.min(points[:, 0])), float(np.min(centers[:, 0] - radii_i)))
        x_max = max(float(np.max(points[:, 0])), float(np.max(centers[:, 0] + radii_i)))
        y_min = min(float(np.min(points[:, 1])), float(np.min(centers[:, 1] - radii_i)))
        y_max = max(float(np.max(points[:, 1])), float(np.max(centers[:, 1] + radii_i)))
    else:
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

    x_min = math.floor((x_min - padding_m) / cell_m) * cell_m
    x_max = math.ceil((x_max + padding_m) / cell_m) * cell_m
    y_min = math.floor((y_min - padding_m) / cell_m) * cell_m
    y_max = math.ceil((y_max + padding_m) / cell_m) * cell_m
    xs = np.arange(x_min, x_max + 0.5 * cell_m, cell_m)
    ys = np.arange(y_min, y_max + 0.5 * cell_m, cell_m)

    occupied = np.zeros((len(ys), len(xs)), dtype=np.bool_)
    if len(centers):
        xx, yy = np.meshgrid(xs, ys)
        dx = xx[..., None] - centers[:, 0]
        dy = yy[..., None] - centers[:, 1]
        occupied = np.any(dx * dx + dy * dy <= radii_i * radii_i, axis=-1)

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
            next_row, next_col = row + d_row, col + d_col
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
    return (
        route_length
        + float(np.linalg.norm(start - start_cell))
        + float(np.linalg.norm(goal - goal_cell))
    )


def chain_bfs_geodesic_lengths(
    route_points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray | None:
    """Certify start->gate1->...->gate4 by chaining four BFS calls."""
    leg_lengths = []
    for start, goal in zip(route_points[:-1], route_points[1:]):
        geodesic = bfs_geodesic_length(start, goal, centers, radii)
        if geodesic is None or not math.isfinite(geodesic):
            return None
        leg_lengths.append(geodesic)
    return np.asarray(leg_lengths, dtype=np.float64)


def _sample_candidate(
    obstacle_count: int,
    rng: np.random.Generator,
    blockers: int = REQUIRED_ON_LINE_BLOCKERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    route_points = sample_route(rng)
    radii = rng.uniform(
        MIN_OBSTACLE_RADIUS_M, MAX_OBSTACLE_RADIUS_M, size=obstacle_count
    )
    centers = np.empty((obstacle_count, 2), dtype=np.float64)
    on_line_mask = np.zeros(obstacle_count, dtype=np.bool_)
    blocker_segments = np.full(obstacle_count, -1, dtype=np.int64)

    chosen_segments = rng.choice(NUM_SEGMENTS, size=blockers, replace=False)
    for obstacle_index, segment_index in enumerate(chosen_segments):
        start = route_points[segment_index]
        end = route_points[segment_index + 1]
        segment = end - start
        length = float(np.linalg.norm(segment))
        direction = segment / length
        normal = np.array((-direction[1], direction[0]))
        endpoint_distance = (
            GATE_CLEAR_RADIUS_M + radii[obstacle_index] + OBSTACLE_INFLATION_M
        )
        fraction_low = max(ON_LINE_FRACTION_MIN, endpoint_distance / length)
        fraction_high = min(ON_LINE_FRACTION_MAX, 1.0 - endpoint_distance / length)
        if fraction_low > fraction_high:
            return None
        fraction = rng.uniform(fraction_low, fraction_high)
        offset = rng.uniform(-ON_LINE_OFFSET_MAX_M, ON_LINE_OFFSET_MAX_M)
        centers[obstacle_index] = start + fraction * segment + offset * normal
        on_line_mask[obstacle_index] = True
        blocker_segments[obstacle_index] = segment_index

    # blockers == 0 (line-avoid tier 0) must scatter-place EVERY obstacle.
    # For blockers >= 1 the historical start index is kept bit-for-bit:
    # frozen tier probes measured layouts produced by exactly this loop, so
    # its RNG stream and index arithmetic must not move under them.
    scatter_start = REQUIRED_ON_LINE_BLOCKERS if blockers > 0 else 0
    for obstacle_index in range(scatter_start, obstacle_count):
        admitted = False
        for _ in range(1000):
            segment_index = int(rng.integers(0, NUM_SEGMENTS))
            start = route_points[segment_index]
            end = route_points[segment_index + 1]
            segment = end - start
            length = float(np.linalg.norm(segment))
            direction = segment / length
            normal = np.array((-direction[1], direction[0]))
            fraction = rng.uniform(0.0, 1.0)
            offset = rng.uniform(
                -SCATTER_ROUTE_DISTANCE_MAX_M, SCATTER_ROUTE_DISTANCE_MAX_M
            )
            candidate = start + fraction * segment + offset * normal
            required_gate_distance = (
                GATE_CLEAR_RADIUS_M + radii[obstacle_index] + OBSTACLE_INFLATION_M
            )
            if np.any(
                np.linalg.norm(route_points - candidate, axis=1)
                < required_gate_distance
            ):
                continue
            previous_distance = np.linalg.norm(
                centers[:obstacle_index] - candidate, axis=1
            )
            required_separation = (
                inflated_radii(radii[:obstacle_index])
                + radii[obstacle_index]
                + OBSTACLE_INFLATION_M
                + 0.25
            )
            if np.any(previous_distance < required_separation):
                continue
            centers[obstacle_index] = candidate
            admitted = True
            break
        if not admitted:
            return None

    return route_points, centers, radii, on_line_mask, blocker_segments


def sample_layout(
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 20,
    obstacle_count: int = REQUESTED_OBSTACLE_COUNT,
    blockers_override: int | None = None,
) -> PathHazardLayout:
    """Rejection-sample a chain-feasible interaction layout.

    Each K receives at most ``max_attempts`` complete candidates. Failure emits
    a warning before K is reduced, but K never falls below the blocker count
    required to preserve the task's interaction effect. ``blockers_override``
    (tier-regrade probes) may pin 0..NUM_SEGMENTS on-line blockers; None keeps
    the frozen v1 value of three. 0 is the line-avoid tier: a plain gate
    route whose off-line scatter is unchanged and whose on-line corridor
    (the ``count_on_line_blockers`` rule) is certified obstacle-free.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    blockers = REQUIRED_ON_LINE_BLOCKERS if blockers_override is None else int(blockers_override)
    if not 0 <= blockers <= NUM_SEGMENTS:
        raise ValueError(
            f"blockers must be in 0..{NUM_SEGMENTS} (at most one route segment each); got {blockers}"
        )
    if obstacle_count < blockers:
        raise ValueError("obstacle_count cannot be smaller than the blocker count")
    rng = np.random.default_rng() if rng is None else rng
    total_attempts = 0

    for accepted_count in range(obstacle_count, blockers - 1, -1):
        for _ in range(max_attempts):
            total_attempts += 1
            candidate = _sample_candidate(accepted_count, rng, blockers)
            if candidate is None:
                continue
            route_points, centers, radii, on_line_mask, blocker_segments = candidate
            if not gate_disks_clear(route_points, centers, radii):
                continue
            if not all_obstacles_within_route_band(route_points, centers):
                continue
            if not _inflated_obstacles_separated(centers, radii):
                continue
            measured_blockers = count_on_line_blockers(route_points, centers)
            if blockers == 0:
                # Line-avoid tier 0 is a plain gate route: reject any scatter
                # obstacle that strays into a segment's on-line corridor.
                if measured_blockers != 0:
                    continue
            elif measured_blockers < blockers:
                continue
            leg_lengths = chain_bfs_geodesic_lengths(route_points, centers, radii)
            if leg_lengths is None:
                continue
            return PathHazardLayout(
                route_points=route_points,
                centers=centers,
                radii=radii,
                on_line_mask=on_line_mask,
                blocker_segment_indices=blocker_segments,
                leg_geodesic_lengths=leg_lengths,
                requested_obstacle_count=obstacle_count,
                attempts=total_attempts,
            )

        if accepted_count > blockers:
            warnings.warn(
                f"PathHazard: no feasible K={accepted_count} layout in "
                f"{max_attempts} attempts; reducing to K={accepted_count - 1}.",
                RuntimeWarning,
                stacklevel=2,
            )

    raise RuntimeError(
        f"Could not generate a feasible PathHazard layout while retaining "
        f"{blockers} blockers."
    )


def analytic_min_clearance(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    *,
    half_beam_m: float = HALF_BEAM_M,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """COM-to-cylinder clearance: min(||p-c|| - radius - half_beam)."""
    if points.shape[-1] != 2 or centers.shape[-1] != 2:
        raise ValueError("points and centers must end in planar XY coordinates")
    if centers.shape[-2] == 0:
        return torch.full(
            points.shape[:-1], torch.inf, device=points.device, dtype=points.dtype
        )
    distances = torch.linalg.vector_norm(points.unsqueeze(-2) - centers, dim=-1)
    clearances = distances - radii - float(half_beam_m)
    if active_mask is not None:
        clearances = torch.where(
            active_mask, clearances, torch.full_like(clearances, torch.inf)
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
    """Nearest forward ray/cylinder intersections for batched analytic rays."""
    if max_range_m <= 0.0:
        raise ValueError("max_range_m must be positive")
    if origins.shape[-1] != 2 or directions.shape[-1] != 2 or centers.shape[-1] != 2:
        raise ValueError("origins, directions, and centers must be planar XY")
    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    to_center = centers.unsqueeze(-3) - origins.unsqueeze(-2).unsqueeze(-2)
    projection = torch.sum(directions.unsqueeze(-2) * to_center, dim=-1)
    center_distance_sq = torch.sum(to_center * to_center, dim=-1)
    radius_sq = radii.unsqueeze(-2) * radii.unsqueeze(-2)
    discriminant = radius_sq - (center_distance_sq - projection * projection)
    root = torch.sqrt(torch.clamp(discriminant, min=0.0))
    inside = center_distance_sq <= radius_sq
    entry = torch.where(inside, torch.zeros_like(projection), projection - root)
    valid = (discriminant >= 0.0) & (entry >= 0.0)
    if active_mask is not None:
        valid &= active_mask.unsqueeze(-2)
    intersections = torch.where(
        valid, entry, torch.full_like(entry, float(max_range_m))
    )
    return torch.min(intersections, dim=-1).values.clamp(max=float(max_range_m))


__all__ = [
    "COLLISION_MARGIN_M",
    "GATE_CLEAR_RADIUS_M",
    "GOAL_RADIUS_M",
    "HALF_BEAM_M",
    "HEADING_CHANGE_MAX_DEG",
    "MAX_OBSTACLE_RADIUS_M",
    "MIN_OBSTACLE_RADIUS_M",
    "NUM_SEGMENTS",
    "OBSTACLE_INFLATION_M",
    "ON_LINE_FRACTION_MAX",
    "ON_LINE_FRACTION_MIN",
    "ON_LINE_OFFSET_MAX_M",
    "PathHazardLayout",
    "REQUESTED_OBSTACLE_COUNT",
    "REQUIRED_ON_LINE_BLOCKERS",
    "SCATTER_ROUTE_DISTANCE_MAX_M",
    "SEGMENT_LENGTH_MAX_M",
    "SEGMENT_LENGTH_MIN_M",
    "all_obstacles_within_route_band",
    "analytic_min_clearance",
    "bfs_geodesic_length",
    "chain_bfs_geodesic_lengths",
    "count_on_line_blockers",
    "distance_to_route",
    "gate_disks_clear",
    "inflated_radii",
    "ray_circle_ranges",
    "sample_layout",
    "sample_route",
]
