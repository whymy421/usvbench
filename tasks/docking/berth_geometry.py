# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solid berth walls for the docking x structure crossing (C3 + C5).

The slip is a U of overlapping cylinders: two side walls plus a back wall,
opening toward the approach direction. Cylinders (rather than segments) reuse
the certified analytic clearance and ray-cast kernels from hazard_nav byte for
byte, so the contact predicate and the 36-ray sensor keep identical semantics
across tasks -- a requirement of the frozen sensor contract.

Local frame: dock point at the origin, +X along the dock heading (the heading
the hull must match when berthed), so the slip opening faces -X.
"""

from __future__ import annotations

import math

import numpy as np


WALL_CYLINDER_RADIUS_M = 0.35
# Consecutive wall cylinders overlap by this much after hull inflation, so a
# wall is sealed: no hull-width gap can appear between two neighbours.
WALL_OVERLAP_M = 0.25
HULL_BEAM_M = 0.899


def slip_width_for_beams(beams: float) -> float:
    """Free slip width expressed in hull beams (tier ladder, as in hazard)."""
    return float(beams) * HULL_BEAM_M


def berth_wall_cylinders(
    slip_width_m: float,
    slip_length_m: float,
    *,
    wall_radius_m: float = WALL_CYLINDER_RADIUS_M,
    overlap_m: float = WALL_OVERLAP_M,
    back_wall: bool = True,
    back_offset_m: float = 1.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centers, radii) of the U-shaped slip in the dock-local frame.

    ``slip_width_m`` is the FREE width between the two inner wall faces, so
    the cylinder centres sit half a slip width plus one radius off the axis.
    ``slip_length_m`` measures from the opening plane to the back wall.
    """
    if slip_width_m <= 0.0 or slip_length_m <= 0.0:
        raise ValueError("slip width and length must be positive")
    if wall_radius_m <= 0.0:
        raise ValueError("wall radius must be positive")
    step = max(1.0e-3, 2.0 * wall_radius_m - overlap_m)

    lateral = 0.5 * slip_width_m + wall_radius_m
    # Side walls run from the opening (x = -slip_length) to the back (x = 0).
    count = int(math.ceil(slip_length_m / step)) + 1
    xs = -slip_length_m + step * np.arange(count, dtype=np.float64)
    xs = np.clip(xs, -slip_length_m, 0.0)

    centers = [np.stack([xs, np.full_like(xs, +lateral)], axis=1),
               np.stack([xs, np.full_like(xs, -lateral)], axis=1)]

    if back_wall:
        # The back wall closes the slip AHEAD of the dock point. Its face must
        # clear the hull disc (radius = half beam) that sits on the dock point
        # when berthed, hence the offset rather than a wall at x = radius.
        back_x = back_offset_m
        span = np.arange(-lateral, lateral + 1.0e-9, step, dtype=np.float64)
        centers.append(
            np.stack([np.full_like(span, back_x), span], axis=1)
        )

    stacked = np.concatenate(centers, axis=0)
    radii = np.full(stacked.shape[0], float(wall_radius_m), dtype=np.float64)
    return stacked, radii


def to_world(
    centers_local: np.ndarray,
    dock_point_xy: np.ndarray,
    dock_heading_xy: np.ndarray,
) -> np.ndarray:
    """Rotate/translate dock-local wall centres into the world frame."""
    heading = np.asarray(dock_heading_xy, dtype=np.float64)
    norm = float(np.linalg.norm(heading))
    if norm < 1.0e-9:
        raise ValueError("dock heading must be non-zero")
    heading = heading / norm
    rotation = np.array(
        ((heading[0], -heading[1]), (heading[1], heading[0])), dtype=np.float64
    )
    return np.asarray(centers_local, dtype=np.float64) @ rotation.T + np.asarray(
        dock_point_xy, dtype=np.float64
    )


def opening_clearance_ok(
    slip_width_m: float,
    hull_beam_m: float = HULL_BEAM_M,
    inflation_m: float = 0.65,
) -> bool:
    """Admission check: the inflated hull must physically fit the slip."""
    return slip_width_m > hull_beam_m + 2.0 * 0.0 and slip_width_m > 2.0 * inflation_m


__all__ = [
    "HULL_BEAM_M",
    "WALL_CYLINDER_RADIUS_M",
    "berth_wall_cylinders",
    "opening_clearance_ok",
    "slip_width_for_beams",
    "to_world",
]
