"""Isaac-free geometry and procedural layout oracle for Hazard Navigation.

Only the Python standard library, NumPy, and Torch are imported here so layout
generation and all analytic obstacle math can be tested without Isaac Lab.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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


# --- Ring-siege variant (user-requested): spawn encircled, exactly one
# passable gap whose inflated width follows the tier ladder. Escape requires
# threading from the very first second -- avoidance as a mandatory skill.
RING_RADIUS_M = 10.5  # >= ENDPOINT_CLEAR (5.0) + 2 * max inflated radius
RING_OBSTACLE_RADIUS_MIN_M = 1.5
RING_OBSTACLE_RADIUS_MAX_M = 2.0
RING_NEIGHBOR_OVERLAP_M = 0.30  # inflated neighbors overlap: sealed by construction
RING_GAP_SLACK_M = 0.50  # accepted gap width band: [bottleneck, bottleneck + slack]
RING_GOAL_DISTANCE_RANGE_M = (24.0, 40.0)  # 24 keeps the goal disk clear of the ring


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

    def chord_angle(chord: float) -> float:
        return 2.0 * math.asin(min(1.0, chord / (2.0 * RING_RADIUS_M)))

    overlap_max = RING_NEIGHBOR_OVERLAP_M  # tightest packing (most sealed)
    overlap_min = 0.05  # loosest packing that still overlaps after inflation

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
            total = chord_angle(gap_free + infl[-1] + infl[0])
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
        angles = [gap_bearing + 0.5 * chord_angle(gap_free + infl[-1] + infl[0])]
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


__all__ = [
    "COLLISION_MARGIN_M",
    "DIFFICULTIES",
    "ENDPOINT_CLEAR_RADIUS_M",
    "GRID_CELL_M",
    "HALF_BEAM_M",
    "HULL_BEAM_M",
    "HazardLayout",
    "MAX_OBSTACLE_RADIUS_M",
    "MIN_OBSTACLE_RADIUS_M",
    "OBSTACLE_INFLATION_M",
    "analytic_min_clearance",
    "bfs_geodesic_length",
    "difficulty_for_level",
    "direct_segment_blocked",
    "inflated_radii",
    "minimum_pairwise_inflated_gap",
    "ray_circle_ranges",
    "sample_layout",
    "start_goal_disks_clear",
]
