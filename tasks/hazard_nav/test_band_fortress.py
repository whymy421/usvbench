"""CPU-only admission audit for the constructive band fortress."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np


GEOMETRY_PATH = Path(__file__).with_name("hazard_geometry.py")
SPEC = importlib.util.spec_from_file_location("band_fortress_geometry", GEOMETRY_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load {GEOMETRY_PATH}")
sys.dont_write_bytecode = True
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


LEVELS = (0, 1, 2, 3)


def _minimum_edge_gap(centers: np.ndarray, radii: np.ndarray) -> float:
    delta = centers[:, None, :] - centers[None, :, :]
    gaps = np.linalg.norm(delta, axis=-1) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, math.inf)
    return float(np.min(gaps))


def _audit_level(
    level: int, layouts: int, rng: np.random.Generator
) -> tuple[float, float, float, int]:
    aperture = float(geometry.FORTRESS_APERTURE_M[level])
    minimum_gap = math.inf
    minimum_spawn_margin = math.inf
    maximum_spawn_margin = -math.inf
    maximum_count = 0

    for _ in range(layouts):
        layout = geometry.sample_band_fortress_layout(
            level, rng=rng, max_attempts=80
        )
        gap = _minimum_edge_gap(layout.centers, layout.radii)
        assert gap >= aperture - 1.0e-6, (level, gap, aperture)

        route = geometry.bfs_geodesic_length(
            layout.start,
            layout.goal,
            layout.centers,
            layout.radii,
            inflation_m=geometry.HALF_BEAM_M,
        )
        assert route is not None, f"level {level}: no half-beam route"
        assert np.allclose(layout.goal, 0.0), layout.goal

        outer_extent = float(
            np.max(np.linalg.norm(layout.centers, axis=1) + layout.radii)
        )
        spawn_margin = float(np.linalg.norm(layout.start)) - outer_extent
        assert spawn_margin >= 2.0 - 1.0e-6, (level, spawn_margin)
        assert spawn_margin <= 8.0 + 1.0e-6, (level, spawn_margin)
        assert layout.obstacle_count <= geometry.BAND_FORTRESS_MAX_OBSTACLES, (
            level,
            layout.obstacle_count,
        )

        minimum_gap = min(minimum_gap, gap)
        minimum_spawn_margin = min(minimum_spawn_margin, spawn_margin)
        maximum_spawn_margin = max(maximum_spawn_margin, spawn_margin)
        maximum_count = max(maximum_count, layout.obstacle_count)

    return minimum_gap, minimum_spawn_margin, maximum_spawn_margin, maximum_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.layouts < 1:
        parser.error("--layouts must be at least one")

    started = time.time()
    rng = np.random.default_rng(args.seed)
    failures: list[str] = []
    for level in LEVELS:
        aperture = geometry.FORTRESS_APERTURE_M[level]
        try:
            min_gap, spawn_min, spawn_max, max_count = _audit_level(
                level, args.layouts, rng
            )
            print(
                f"level {level}: layouts={args.layouts} A={aperture:.1f}m "
                f"min_gap={min_gap:.6f}m routes={args.layouts}/{args.layouts} "
                f"spawn_margin={spawn_min:.3f}..{spawn_max:.3f}m "
                f"max_cylinders={max_count} PASS"
            )
        except Exception as error:  # keep one summary line for every tier
            failures.append(f"level {level}: {error}")
            print(
                f"level {level}: layouts={args.layouts} A={aperture:.1f}m "
                f"FAIL {error}"
            )

    elapsed = time.time() - started
    if failures:
        print(f"FAIL: {len(failures)} level(s) failed ({elapsed:.1f} s)")
        raise SystemExit(1)
    print(
        f"PASS: {args.layouts * len(LEVELS)} band-fortress layouts, "
        f"0 violations ({elapsed:.1f} s)"
    )


if __name__ == "__main__":
    main()
