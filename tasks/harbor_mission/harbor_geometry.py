"""Isaac-free route generation and gate geometry for Ordered Harbor Mission.

This module deliberately depends only on the Python standard library, NumPy,
Torch, and Hazard Navigation's equally Isaac-free geometry oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from ..hazard_nav.hazard_geometry import (
        DIFFICULTIES,
        OBSTACLE_INFLATION_M,
        bfs_geodesic_length,
        direct_segment_blocked,
        minimum_pairwise_inflated_gap,
        sample_layout,
        start_goal_disks_clear,
    )
except ImportError:  # Direct import from test_mission_geometry.py.
    _HAZARD_DIR = Path(__file__).resolve().parents[1] / "hazard_nav"
    if str(_HAZARD_DIR) not in sys.path:
        sys.path.insert(0, str(_HAZARD_DIR))
    from hazard_geometry import (  # type: ignore[no-redef]
        DIFFICULTIES,
        OBSTACLE_INFLATION_M,
        bfs_geodesic_length,
        direct_segment_blocked,
        minimum_pairwise_inflated_gap,
        sample_layout,
        start_goal_disks_clear,
    )


EXIT_GATE_X_M = (5.0, 15.0)
FIELD_START_X_M = 20.0
FIELD_LENGTH_RANGE_M = (25.0, 35.0)
BERTH_OFFSET_RANGE_M = (15.0, 25.0)
GATE_WIDTH_M = 6.0
GATE_HALF_WIDTH_M = GATE_WIDTH_M / 2.0
GATE_APPROACH_DISTANCE_M = 1.0
GATE_MIN_NORMAL_SPEED_MPS = 0.2
BERTH_CLEAR_RADIUS_M = 5.0
MIN_OBSTACLE_COUNT = 6
MAX_OBSTACLE_COUNT = 10
MIN_OBSTACLE_RADIUS_M = 0.8
MAX_OBSTACLE_RADIUS_M = 1.8
MID_TIER_HAZARD_LEVEL = 1


@dataclass(frozen=True)
class HarborRoute:
    """One accepted route in the environment-local +X frame."""

    gate_midpoints: np.ndarray
    gate_normals: np.ndarray
    gate_half_width_m: float
    field_start: np.ndarray
    field_exit: np.ndarray
    berth_point: np.ndarray
    dock_heading: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray
    field_geodesic_length: float
    direct_blocked: bool
    attempts: int

    @property
    def obstacle_count(self) -> int:
        return int(self.obstacle_radii.shape[0])

    @property
    def field_length_m(self) -> float:
        return float(self.field_exit[0] - self.field_start[0])

    @property
    def route_length_m(self) -> float:
        return float(self.berth_point[0])

    @property
    def stage_spans_m(self) -> np.ndarray:
        """Exit, transit, and dock longitudinal spans used by reward Phi."""
        return np.array(
            (
                EXIT_GATE_X_M[1],
                self.field_exit[0] - EXIT_GATE_X_M[1],
                self.berth_point[0] - self.field_exit[0],
            ),
            dtype=np.float64,
        )


def gate_endpoints(
    midpoint: np.ndarray, normal: np.ndarray, half_width_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered endpoints of a gate segment perpendicular to its normal."""
    midpoint = np.asarray(midpoint, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    tangent = np.array((-normal[1], normal[0]), dtype=np.float64)
    return midpoint - half_width_m * tangent, midpoint + half_width_m * tangent


def point_segment_distance(
    points: np.ndarray, endpoint_a: np.ndarray, endpoint_b: np.ndarray
) -> np.ndarray:
    """Vectorized Euclidean distance from planar points to a finite segment."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    endpoint_a = np.asarray(endpoint_a, dtype=np.float64)
    endpoint_b = np.asarray(endpoint_b, dtype=np.float64)
    segment = endpoint_b - endpoint_a
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 0.0:
        raise ValueError("gate endpoints must be distinct")
    fraction = np.clip(((points - endpoint_a) @ segment) / length_sq, 0.0, 1.0)
    closest = endpoint_a + fraction[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def gate_clear_of_obstacles(
    midpoint: np.ndarray,
    normal: np.ndarray,
    half_width_m: float,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    inflation_m: float = OBSTACLE_INFLATION_M,
) -> bool:
    """Whether inflated obstacle cylinders are disjoint from a gate segment."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(centers) == 0:
        return True
    endpoint_a, endpoint_b = gate_endpoints(midpoint, normal, half_width_m)
    required = np.asarray(radii, dtype=np.float64) + float(inflation_m)
    return bool(np.all(point_segment_distance(centers, endpoint_a, endpoint_b) > required))


def berth_clear_of_obstacles(
    berth_point: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    clear_radius_m: float = BERTH_CLEAR_RADIUS_M,
    inflation_m: float = OBSTACLE_INFLATION_M,
) -> bool:
    """Check the protected docking disk against inflated cylinders."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(centers) == 0:
        return True
    distances = np.linalg.norm(centers - np.asarray(berth_point), axis=1)
    required = np.asarray(radii, dtype=np.float64) + inflation_m + clear_radius_m
    return bool(np.all(distances > required))


def arm_gate_approach(
    armed: torch.Tensor,
    positions: torch.Tensor,
    gate_midpoints: torch.Tensor,
    gate_normals: torch.Tensor,
    *,
    approach_distance_m: float = GATE_APPROACH_DISTANCE_M,
) -> torch.Tensor:
    """Latch proof that the trajectory visited at least 1 m before the plane."""
    signed_distance = torch.sum((positions - gate_midpoints) * gate_normals, dim=-1)
    return armed | (signed_distance <= -float(approach_distance_m))


def gate_crossing_mask(
    previous_positions: torch.Tensor,
    current_positions: torch.Tensor,
    velocities: torch.Tensor,
    gate_midpoints: torch.Tensor,
    gate_normals: torch.Tensor,
    approach_armed: torch.Tensor,
    *,
    half_width_m: float = GATE_HALF_WIDTH_M,
    min_normal_speed_mps: float = GATE_MIN_NORMAL_SPEED_MPS,
) -> torch.Tensor:
    """Detect valid outward crossings of finite gate segments.

    ``approach_armed`` is a monotone latch proving that an earlier trajectory
    sample lay at least 1 m before the plane. The immediate previous/current
    samples must straddle the plane, their interpolated intersection must lie
    within the endpoints, and current normal velocity must be strictly greater
    than 0.2 m/s. Separating the latch from the adjacent-sample line crossing
    avoids requiring an impossible one-metre jump in one 60 Hz control step.
    """
    previous_signed = torch.sum(
        (previous_positions - gate_midpoints) * gate_normals, dim=-1
    )
    current_signed = torch.sum(
        (current_positions - gate_midpoints) * gate_normals, dim=-1
    )
    denominator = current_signed - previous_signed
    crosses_plane = (
        (previous_signed <= 0.0)
        & (current_signed >= 0.0)
        & (denominator > 1.0e-9)
    )
    fraction = torch.where(
        crosses_plane,
        -previous_signed / denominator.clamp_min(1.0e-9),
        torch.zeros_like(denominator),
    )
    intersection = previous_positions + fraction.unsqueeze(-1) * (
        current_positions - previous_positions
    )
    tangent = torch.stack((-gate_normals[..., 1], gate_normals[..., 0]), dim=-1)
    lateral = torch.abs(torch.sum((intersection - gate_midpoints) * tangent, dim=-1))
    normal_speed = torch.sum(velocities * gate_normals, dim=-1)
    return (
        approach_armed
        & crosses_plane
        & (lateral <= float(half_width_m))
        & (normal_speed > float(min_normal_speed_mps))
    )


def sample_harbor_route(
    rng: np.random.Generator | None = None,
    *,
    max_attempts: int = 50,
    hazard_max_attempts: int = 20,
) -> HarborRoute:
    """Generate a feasible 60--80 m ordered harbor route.

    The obstacle field is derived from ``hazard_geometry.sample_layout`` at
    level 1. Its longitudinal extent is mapped to 25--35 m and its radii are
    mapped from HazardNav's [0.8, 2.0] range to [0.8, 1.8]. The mid-tier gap,
    endpoint disks, forced direct blocker, and BFS oracle are rechecked after
    that transformation rather than assumed to survive it.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    rng = np.random.default_rng() if rng is None else rng
    difficulty = DIFFICULTIES[MID_TIER_HAZARD_LEVEL]

    for attempt in range(1, max_attempts + 1):
        base = sample_layout(
            MID_TIER_HAZARD_LEVEL,
            rng=rng,
            max_attempts=hazard_max_attempts,
        )
        if not (MIN_OBSTACLE_COUNT <= base.obstacle_count <= MAX_OBSTACLE_COUNT):
            continue

        field_length = float(rng.uniform(*FIELD_LENGTH_RANGE_M))
        scale_x = field_length / float(base.goal[0] - base.start[0])
        centers_field = np.asarray(base.centers, dtype=np.float64).copy()
        centers_field[:, 0] *= scale_x
        radii = MIN_OBSTACLE_RADIUS_M + (
            np.asarray(base.radii, dtype=np.float64) - MIN_OBSTACLE_RADIUS_M
        ) * (
            (MAX_OBSTACLE_RADIUS_M - MIN_OBSTACLE_RADIUS_M) / (2.0 - 0.8)
        )
        radii = np.clip(radii, MIN_OBSTACLE_RADIUS_M, MAX_OBSTACLE_RADIUS_M)
        field_start_zero = np.zeros(2, dtype=np.float64)
        field_exit_zero = np.array((field_length, 0.0), dtype=np.float64)

        if not start_goal_disks_clear(
            field_start_zero, field_exit_zero, centers_field, radii
        ):
            continue
        if (
            minimum_pairwise_inflated_gap(centers_field, radii) + 1.0e-9
            < difficulty.bottleneck_m
        ):
            continue
        geodesic = bfs_geodesic_length(
            field_start_zero, field_exit_zero, centers_field, radii
        )
        if geodesic is None or not math.isfinite(geodesic):
            continue
        if not direct_segment_blocked(
            field_start_zero, field_exit_zero, centers_field, radii
        ):
            continue

        field_start = np.array((FIELD_START_X_M, 0.0), dtype=np.float64)
        field_exit = field_start + field_exit_zero
        centers = centers_field + field_start
        berth_point = field_exit + np.array(
            (float(rng.uniform(*BERTH_OFFSET_RANGE_M)), 0.0), dtype=np.float64
        )
        gate_midpoints = np.array(
            (
                (EXIT_GATE_X_M[0], 0.0),
                (EXIT_GATE_X_M[1], 0.0),
                (field_exit[0], field_exit[1]),
            ),
            dtype=np.float64,
        )
        gate_normals = np.tile(np.array((1.0, 0.0)), (3, 1))
        gates_clear = all(
            gate_clear_of_obstacles(
                midpoint, normal, GATE_HALF_WIDTH_M, centers, radii
            )
            for midpoint, normal in zip(gate_midpoints, gate_normals, strict=True)
        )
        if not gates_clear or not berth_clear_of_obstacles(berth_point, centers, radii):
            continue

        return HarborRoute(
            gate_midpoints=gate_midpoints,
            gate_normals=gate_normals,
            gate_half_width_m=GATE_HALF_WIDTH_M,
            field_start=field_start,
            field_exit=field_exit,
            berth_point=berth_point,
            dock_heading=np.array((1.0, 0.0), dtype=np.float64),
            obstacle_centers=centers,
            obstacle_radii=radii,
            field_geodesic_length=float(geodesic),
            direct_blocked=True,
            attempts=attempt,
        )

    raise RuntimeError(
        f"Could not generate a feasible harbor route in {max_attempts} attempts."
    )


__all__ = [
    "BERTH_CLEAR_RADIUS_M",
    "BERTH_OFFSET_RANGE_M",
    "EXIT_GATE_X_M",
    "FIELD_LENGTH_RANGE_M",
    "FIELD_START_X_M",
    "GATE_APPROACH_DISTANCE_M",
    "GATE_HALF_WIDTH_M",
    "GATE_MIN_NORMAL_SPEED_MPS",
    "GATE_WIDTH_M",
    "HarborRoute",
    "MAX_OBSTACLE_COUNT",
    "MAX_OBSTACLE_RADIUS_M",
    "MID_TIER_HAZARD_LEVEL",
    "MIN_OBSTACLE_COUNT",
    "MIN_OBSTACLE_RADIUS_M",
    "arm_gate_approach",
    "berth_clear_of_obstacles",
    "gate_clear_of_obstacles",
    "gate_crossing_mask",
    "gate_endpoints",
    "point_segment_distance",
    "sample_harbor_route",
]
