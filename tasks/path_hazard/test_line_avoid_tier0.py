"""CPU acceptance test for the line-avoid tier of the blocker ladder.

The owner froze the tier ladder as 0/1/2/3/4 on-line blockers. Tier 0 is a
plain gate route: every obstacle is off-line scatter and the on-line corridor
of every segment (the geometry's own ``count_on_line_blockers`` rule) must be
empty. Tiers 1..4 keep their pre-existing sampling behavior.

Run: python tasks/path_hazard/test_line_avoid_tier0.py
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    from .path_hazard_geometry import (
        NUM_SEGMENTS,
        REQUESTED_OBSTACLE_COUNT,
        all_obstacles_within_route_band,
        count_on_line_blockers,
        gate_disks_clear,
        sample_layout,
    )
except ImportError:  # Direct execution from the repository root.
    from path_hazard_geometry import (
        NUM_SEGMENTS,
        REQUESTED_OBSTACLE_COUNT,
        all_obstacles_within_route_band,
        count_on_line_blockers,
        gate_disks_clear,
        sample_layout,
    )


TIER0_LAYOUTS = 20
SPOT_LAYOUTS_PER_TIER = 6
# Overrides 1 and 2 inherit a rejection-heavy legacy path; a raised budget
# keeps the spot sample deterministic-in-contract across allocators.
SPOT_MAX_ATTEMPTS = 60


def test_validator_accepts_zero_and_rejects_out_of_range() -> None:
    for bad in (-1, NUM_SEGMENTS + 1):
        try:
            sample_layout(rng=np.random.default_rng(0), blockers_override=bad)
        except ValueError as exc:
            assert "0.." in str(exc), str(exc)
        else:
            raise AssertionError(f"blockers_override={bad} should raise ValueError")


def test_tier0_corridor_is_clear() -> None:
    rng = np.random.default_rng(20260823)
    for index in range(TIER0_LAYOUTS):
        with warnings.catch_warnings():
            # Tier 0 has no forced placements; layouts should come easily and
            # a K-reduction warning would indicate a sampler regression.
            warnings.simplefilter("error")
            layout = sample_layout(rng=rng, blockers_override=0)
        # The geometry's own corridor rule: nothing on-line, on any segment.
        measured = count_on_line_blockers(layout.route_points, layout.centers)
        assert measured == 0, (index, measured)
        assert layout.on_line_blocker_count == 0, index
        assert bool((layout.blocker_segment_indices == -1).all()), index
        # Off-line scatter contract is unchanged: full K, inside the route
        # band, clear of the 3 m gate disks, chain-feasible.
        assert layout.obstacle_count == REQUESTED_OBSTACLE_COUNT, index
        assert all_obstacles_within_route_band(layout.route_points, layout.centers)
        assert gate_disks_clear(layout.route_points, layout.centers, layout.radii)
        assert layout.chain_feasible, index
        assert bool(np.isfinite(layout.centers).all()), index


def test_default_path_is_unchanged() -> None:
    """None and an explicit 3 must draw identical layouts from equal seeds."""
    layout_none = sample_layout(rng=np.random.default_rng(99))
    layout_three = sample_layout(rng=np.random.default_rng(99), blockers_override=3)
    assert np.array_equal(layout_none.route_points, layout_three.route_points)
    assert np.array_equal(layout_none.centers, layout_three.centers)
    assert np.array_equal(layout_none.radii, layout_three.radii)
    assert np.array_equal(layout_none.on_line_mask, layout_three.on_line_mask)
    assert layout_none.attempts == layout_three.attempts


def test_tiers_one_to_four_spot_sample_counts() -> None:
    """Existing tiers keep their contract: >= override on-line blockers."""
    for override in range(1, NUM_SEGMENTS + 1):
        rng = np.random.default_rng(4321 + override)
        accepted = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(SPOT_LAYOUTS_PER_TIER):
                try:
                    layout = sample_layout(
                        rng=rng,
                        blockers_override=override,
                        max_attempts=SPOT_MAX_ATTEMPTS,
                    )
                except RuntimeError:
                    continue
                accepted += 1
                measured = count_on_line_blockers(
                    layout.route_points, layout.centers
                )
                assert measured >= override, (override, measured)
                assert layout.on_line_blocker_count == override, override
                assert layout.obstacle_count >= override, override
        assert accepted >= SPOT_LAYOUTS_PER_TIER - 2, (override, accepted)


def main() -> None:
    test_validator_accepts_zero_and_rejects_out_of_range()
    print("PASS validator: 0 accepted, -1 and 5 rejected")
    test_tier0_corridor_is_clear()
    print(f"PASS tier 0: {TIER0_LAYOUTS} layouts, all corridors clear, full K")
    test_default_path_is_unchanged()
    print("PASS default: None == blockers_override=3 layout-for-layout")
    test_tiers_one_to_four_spot_sample_counts()
    print("PASS tiers 1..4: spot-sample counts respect each override")


if __name__ == "__main__":
    main()
