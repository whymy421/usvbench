"""Admission audit for the forced-crossing basin.

Old Task A's tiers do not control difficulty: circumventing the obstacle field
costs 1.63-1.82x the straight line at every tier and the 128/128 champion takes
that route, so its certified numbers measure willingness to detour. The basin
removes the alternative rather than making it expensive.

The claim this script has to earn is exactly one sentence: *with the gate
sealed the goal is unreachable by a hull of the true beam*. Everything else --
gate width, tier separation, cylinder budget -- is reported so the numbers that
end up in the paper come from measurement, not from the constructor's intent.

Run ``python test_forced_crossing.py --layouts 10000`` for the full audit;
the default 200 is the smoke version.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from .hazard_geometry import (
        FORCED_BULKHEAD_RADIUS_M,
        FORCED_GATE_BEAMS,
        FORCED_GATE_FIT_MARGIN_M,
        FORCED_SEAL_CELL_M,
        HALF_BEAM_M,
        HULL_BEAM_M,
        OBSTACLE_INFLATION_M,
        _bulkhead_free_runs,
        bfs_geodesic_length,
        sample_forced_crossing_layout,
        sample_open_basin_layout,
        wall_segment_cylinders,
    )
except ImportError:  # direct execution
    from hazard_geometry import (  # type: ignore
        FORCED_BULKHEAD_RADIUS_M,
        FORCED_GATE_BEAMS,
        FORCED_GATE_FIT_MARGIN_M,
        FORCED_SEAL_CELL_M,
        HALF_BEAM_M,
        HULL_BEAM_M,
        OBSTACLE_INFLATION_M,
        _bulkhead_free_runs,
        bfs_geodesic_length,
        sample_forced_crossing_layout,
        sample_open_basin_layout,
        wall_segment_cylinders,
    )

LEVELS = (0, 1, 2, 3)


def _neighbour_surface_gaps(centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Surface gap to each cylinder's nearest neighbour (negative = overlapping)."""
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    gap = distance - radii[:, None] - radii[None, :]
    np.fill_diagonal(gap, np.inf)
    return gap.min(axis=1)


def audit_one(level: int, rng: np.random.Generator) -> dict:
    """Re-derive every admission claim from the returned geometry."""
    layout, gate = sample_forced_crossing_layout(level, rng=rng)
    centers, radii = layout.centers, layout.radii
    violations: list[str] = []

    # 1. The gate is the width it claims to be, measured on the wall line.
    target = FORCED_GATE_BEAMS[level] * HULL_BEAM_M
    if abs(gate.free_width_m - target) > 0.01:
        violations.append(f"gate {gate.free_width_m:.3f} != target {target:.3f}")
    if gate.free_width_m < HULL_BEAM_M + FORCED_GATE_FIT_MARGIN_M:
        violations.append(f"gate {gate.free_width_m:.3f} too tight for the hull")

    # 2. Exactly one opening in the bulkhead.
    half_width = float(np.abs(centers[:, 1]).max())
    runs = _bulkhead_free_runs(centers, radii, gate.wall_x, -half_width, half_width)
    if len(runs) != 1:
        violations.append(f"{len(runs)} openings in the bulkhead, expected 1")

    # 3. Every wall cylinder physically overlaps a neighbour. A wall that only
    #    LOOKS closed is the ring failure: its neighbours left 1.00 m surface
    #    gaps against a 0.899 m hull and the seal check never noticed, because
    #    it inflated by 0.65 while the simulator collides at 0.45.
    worst = float(_neighbour_surface_gaps(centers, radii).max())
    if worst >= 0.0:
        violations.append(f"a wall cylinder has a {worst:.3f} m gap to its nearest neighbour")

    # 4. Routable as built, at planning inflation.
    if not np.isfinite(layout.geodesic_length):
        violations.append("no route as built")

    # 5. THE claim: seal the gate and the true-beam hull is stuck. No margin,
    #    no planner inflation -- the contact predicate's own half-beam.
    #
    #    The plug is tiled into the SHIPPED arrays rather than rebuilt by
    #    _basin_cylinders. Re-deriving the geometry from the same helper the
    #    generator used would let a shared bug pass both sides; what the
    #    simulator loads is `layout.centers`, so that is what gets tested.
    plug = wall_segment_cylinders(
        np.array((gate.wall_x, runs[0][0] - FORCED_BULKHEAD_RADIUS_M)),
        np.array((gate.wall_x, runs[0][1] + FORCED_BULKHEAD_RADIUS_M)),
        radius_m=FORCED_BULKHEAD_RADIUS_M,
    )
    sealed_c = np.vstack([centers, plug])
    sealed_r = np.concatenate([radii, np.full(len(plug), FORCED_BULKHEAD_RADIUS_M)])
    leak = bfs_geodesic_length(
        layout.start, layout.goal, sealed_c, sealed_r,
        cell_m=FORCED_SEAL_CELL_M, inflation_m=HALF_BEAM_M,
    )
    if leak is not None:
        violations.append(f"sealed basin still routes in {leak:.1f} m")

    # 5b. The plug must not be doing more than plugging: with it removed the
    #     same BFS at the same inflation has to succeed, or "no route when
    #     sealed" would be trivially true for the wrong reason.
    open_route = bfs_geodesic_length(
        layout.start, layout.goal, centers, radii,
        cell_m=FORCED_SEAL_CELL_M, inflation_m=HALF_BEAM_M,
    )
    if open_route is None:
        violations.append("no true-beam route even with the gate open")

    # 6. Endpoints are inside the basin and on opposite sides of the bulkhead.
    if not (layout.start[0] < gate.wall_x < layout.goal[0]):
        violations.append("start and goal are not separated by the bulkhead")

    straight = float(np.linalg.norm(layout.goal - layout.start))
    return {
        "violations": violations,
        "gate_m": gate.free_width_m,
        "cylinders": int(len(radii)),
        "geodesic_m": layout.geodesic_length,
        "straight_m": straight,
        "ratio": layout.geodesic_length / straight,
        "lateral_offset_m": abs(float(gate.center[1])),
        "attempts": layout.attempts,
    }


def audit_basin(rng: np.random.Generator, count: int) -> list[str]:
    """The control task's walls have to hold too.

    A leaky basin is not a milder version of the task, it is a different task:
    the boat could leave the arena and the "bounded" result would be measuring
    something else. The ring shipped with 1.00 m holes in a 0.899 m hull
    because nobody measured the surface gaps, so measure them.
    """
    violations: list[str] = []
    for _ in range(count):
        layout = sample_open_basin_layout(0, rng=rng)
        centers, radii = layout.centers, layout.radii
        worst = float(_neighbour_surface_gaps(centers, radii).max())
        if worst >= 0.0:
            violations.append(f"basin wall gap {worst:.3f} m to nearest neighbour")
        # Escape test: can a true-beam hull reach a point well outside the
        # basin? bfs pads the grid past the walls, so an unreachable exterior
        # target is a real seal, not an artefact of a clipped domain.
        outside = np.array((float(centers[:, 0].max()) + 4.0,
                            float(centers[:, 1].max()) + 4.0))
        if bfs_geodesic_length(layout.start, outside, centers, radii,
                               cell_m=FORCED_SEAL_CELL_M,
                               inflation_m=HALF_BEAM_M) is not None:
            violations.append("hull can leave the basin")
        if not np.isfinite(layout.geodesic_length):
            violations.append("no route to the goal inside the basin")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=200,
                        help="total layouts, split evenly across the four tiers")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--basin", type=int, default=0,
                        help="also audit N open-basin control layouts")
    args = parser.parse_args()

    if args.basin:
        rng = np.random.default_rng(args.seed + 1)
        bad = audit_basin(rng, args.basin)
        print(f"open-basin control: {args.basin} layouts, {len(bad)} violations")
        for line in bad[:5]:
            print(f"  ! {line}")
        print()
        if bad:
            return

    per_level = max(1, args.layouts // len(LEVELS))
    rng = np.random.default_rng(args.seed)
    started = time.time()
    all_violations: list[str] = []
    rows = []

    print(f"forced-crossing admission audit: {per_level} layouts x {len(LEVELS)} tiers")
    print(f"  planning inflation {OBSTACLE_INFLATION_M:.2f} m (route exists) / "
          f"seal inflation {HALF_BEAM_M:.2f} m (no leak), cell {FORCED_SEAL_CELL_M} m")
    print()
    for level in LEVELS:
        stats = [audit_one(level, rng) for _ in range(per_level)]
        bad = [v for s in stats for v in s["violations"]]
        all_violations.extend(bad)
        gates = np.array([s["gate_m"] for s in stats])
        ratios = np.array([s["ratio"] for s in stats])
        cyls = np.array([s["cylinders"] for s in stats])
        offs = np.array([s["lateral_offset_m"] for s in stats])
        rows.append((level, gates, ratios, cyls, offs, len(bad)))
        print(f"tier {level}  gate {FORCED_GATE_BEAMS[level]:.0f} beams = "
              f"{gates.mean():.3f} m (measured, spread {np.ptp(gates):.4f})")
        print(f"         route/straight  median {np.median(ratios):.2f}  "
              f"p90 {np.percentile(ratios, 90):.2f}")
        print(f"         gate offset from the axis  median {np.median(offs):.1f} m  "
              f"max {offs.max():.1f} m")
        print(f"         cylinders/layout  {cyls.min()}-{cyls.max()}  "
              f"(cfg max_obstacles must be >= {cyls.max()})")
        print(f"         violations: {len(bad)}")
        if bad:
            for line in bad[:5]:
                print(f"           ! {line}")
        print()

    worst_cyl = max(int(r[3].max()) for r in rows)
    print(f"elapsed {time.time() - started:.0f} s")
    print(f"max cylinders over every tier: {worst_cyl}")
    print()
    if all_violations:
        print(f"=> FAIL: {len(all_violations)} violations across "
              f"{per_level * len(LEVELS)} layouts.")
        return

    print(f"=> PASS: {per_level * len(LEVELS)} layouts, 0 violations.")
    print("   Sealing the gate makes the goal unreachable at the true half-beam in")
    print("   every one, so success on this task cannot be earned by going around.")
    print()
    print("   Two things the tier ladder does NOT yet control, stated so nobody")
    print("   reads more into the pass than it says:")
    print(f"   - route length barely moves across tiers (the gate's lateral offset")
    print(f"     drives it, not the width), so tier difficulty is entirely 'how")
    print(f"     precisely must I steer', measured only by training;")
    print(f"   - the threading arc's portal test caps mean side reading at 2.25 m,")
    print(f"     i.e. gaps up to {2 * 2.25:.2f} m. Tier 0 at "
          f"{FORCED_GATE_BEAMS[0] * HULL_BEAM_M:.3f} m sits on that boundary and")
    print(f"     will pay erratically; tiers 1-3 are comfortably inside it.")


if __name__ == "__main__":
    main()
