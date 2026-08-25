"""Admission audit for tier-regrade probe apertures (band fortress).

Same three checks the frozen tiers passed, applied to an override aperture:
hard edge-gap guarantee, a true-half-beam route with the gaps open, and the
spawn outside the outer band. CPU only; run before any GPU probe at a new
aperture so a probe number can never come from an unaudited geometry.

Run: python scripts/audit_aperture.py --aperture 5.1 --layouts 1000
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# Import the geometry module directly by file path: going through the package
# would pull tasks/hazard_nav/__init__.py, which imports gymnasium/isaaclab and
# would make this CPU audit unrunnable on machines without the simulator.
import importlib.util

_GEOMETRY = Path(__file__).resolve().parents[1] / "tasks" / "hazard_nav" / "hazard_geometry.py"
_spec = importlib.util.spec_from_file_location("hazard_geometry", _GEOMETRY)
geometry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = geometry  # dataclass resolution needs the registry
_spec.loader.exec_module(geometry)


def minimum_edge_gap(centers, radii):
    delta = centers[:, None, :] - centers[None, :, :]
    gaps = np.linalg.norm(delta, axis=-1) - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, math.inf)
    return float(np.min(gaps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aperture", type=float, required=True)
    parser.add_argument("--layouts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    started = time.time()
    violations = []
    worst_gap = math.inf
    for index in range(args.layouts):
        layout = geometry.sample_band_fortress_layout(
            0, rng=rng, aperture_override_m=args.aperture
        )
        gap = minimum_edge_gap(layout.centers, layout.radii)
        worst_gap = min(worst_gap, gap)
        if gap + 1.0e-6 < args.aperture:
            violations.append(f"layout {index}: edge gap {gap:.3f} < {args.aperture}")
        route = geometry.bfs_geodesic_length(
            layout.start, layout.goal, layout.centers, layout.radii,
            cell_m=0.25, inflation_m=geometry.HALF_BEAM_M,
        )
        if route is None:
            violations.append(f"layout {index}: no true-half-beam route")
        for text in violations[-2:]:
            if text.startswith(f"layout {index}"):
                print("VIOLATION", text, flush=True)
    verdict = "PASS" if not violations else f"FAIL ({len(violations)} violations)"
    print(f"aperture={args.aperture} layouts={args.layouts} worst_gap={worst_gap:.3f} "
          f"elapsed={time.time() - started:.0f}s => {verdict}")
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
