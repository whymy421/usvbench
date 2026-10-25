"""Admission tests for the solid berth walls (pure NumPy, no Isaac)."""

from __future__ import annotations

import numpy as np

try:
    from .berth_geometry import (
        HULL_BEAM_M,
        berth_wall_cylinders,
        slip_width_for_beams,
        to_world,
    )
except ImportError:  # direct execution
    from berth_geometry import (
        HULL_BEAM_M,
        berth_wall_cylinders,
        slip_width_for_beams,
        to_world,
    )

INFLATION_M = 0.65  # half beam + collision margin, as in hazard_nav


def _min_pairwise_gap(centers: np.ndarray, radii: np.ndarray) -> float:
    """Smallest free gap between INFLATED neighbours (negative = sealed)."""
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    gap = distance - (radii + INFLATION_M)[:, None] - (radii + INFLATION_M)[None, :]
    np.fill_diagonal(gap, np.inf)
    return float(gap.min())


def main() -> None:
    for beams in (5.0, 4.0, 3.0, 2.5):
        width = slip_width_for_beams(beams)
        centers, radii = berth_wall_cylinders(width, slip_length_m=7.0)

        # 1. the hull fits: free width exceeds the beam with margin to spare
        assert width > HULL_BEAM_M + 0.3, (beams, width)

        # 2. walls are sealed: every consecutive pair overlaps once inflated.
        # Select ONE side wall by its constant lateral offset (the back wall
        # also has positive-y cylinders and must not be mixed in here).
        lateral = 0.5 * width + radii[0]
        side = centers[np.isclose(centers[:, 1], lateral)]
        side = side[np.argsort(side[:, 0])]
        steps = np.linalg.norm(np.diff(side, axis=0), axis=1)
        assert steps.max() < 2.0 * radii[0] + 1e-9, f"wall gap {steps.max():.3f}"

        # 3. the slip axis is clear: a hull on the centreline never contacts
        axis = np.stack([np.linspace(-7.0, 0.0, 40), np.zeros(40)], axis=1)
        d = np.linalg.norm(axis[:, None, :] - centers[None, :, :], axis=-1)
        clearance = (d - radii[None, :] - 0.45).min()
        assert clearance > 0.0, f"centreline blocked at {beams} beams: {clearance:.3f}"

        # 4. world transform preserves geometry (rigid motion)
        world = to_world(centers, np.array([12.0, -5.0]), np.array([0.3, 0.95]))
        d_local = np.linalg.norm(centers[0] - centers[-1])
        d_world = np.linalg.norm(world[0] - world[-1])
        assert abs(d_local - d_world) < 1e-9

        print(
            f"{beams:.1f} beams: slip {width:.2f} m | {len(radii)} cylinders | "
            f"centreline clearance {clearance:.2f} m | max wall step "
            f"{steps.max():.2f} m"
        )
    print("PASS: berth-wall admission (hull fits, walls sealed, axis clear)")


if __name__ == "__main__":
    main()
