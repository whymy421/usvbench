"""Regression check for the BlueBoat hull orientation in assets/blueboat_physics.usd.

The first CAD-derived asset (commit 1cac168) kept the hull meshes in the CAD frame: bow on -x and the two
hulls swapped port/starboard, while every task drives the boat along +x. A bounding box cannot tell bow
from stern, so this script checks the hull shape itself:

  1. bow on +x       the keel rises towards the bow and the stern is the deepest point
  2. hull sides      each hull's flat wall faces the centreline and its rounded side faces outboard
  3. centre of mass  no lateral offset (the boat floats level)

Usage:  python scripts/check_blueboat_orientation.py [path/to/blueboat_physics.usd]
Exit code 0 when all checks pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = REPO_ROOT / "assets" / "blueboat_physics.usd"
HULLS = ("/World/BlueBoat/Visuals/Hull01_Starboard", "/World/BlueBoat/Visuals/Hull02_Port")
END_FRACTION = 0.12        # length fraction treated as the bow / stern end
MIN_KEEL_RISE_M = 0.05     # bow keel must sit at least this much above the stern keel
MAX_COM_LATERAL_M = 0.01


def hull_points(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise SystemExit(f"missing prim {path}")
    pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
    m = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    return (np.c_[pts, np.ones(len(pts))] @ m.T)[:, :3]


def side_variation(pts: np.ndarray, inner: bool) -> float:
    """Spread of the inboard (or outboard) wall position over depth, mid-ship slice."""
    mid = pts[np.abs(pts[:, 0]) < 0.10]
    ay = np.abs(mid[:, 1])
    edges = np.linspace(-0.18, 0.10, 15)
    wall = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mid[:, 2] >= lo) & (mid[:, 2] < hi)
        if sel.sum() >= 5:
            wall.append(ay[sel].min() if inner else ay[sel].max())
    return float(np.ptp(wall)) if len(wall) >= 5 else float("nan")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    stage = Usd.Stage.Open(str(path))
    pts = np.vstack([hull_points(stage, h) for h in HULLS])
    failed = False

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal failed
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed |= not ok

    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    span = x1 - x0
    keel_plus = pts[pts[:, 0] > x1 - END_FRACTION * span, 2].min()
    keel_minus = pts[pts[:, 0] < x0 + END_FRACTION * span, 2].min()
    report("bow on +x", keel_plus - keel_minus > MIN_KEEL_RISE_M,
           f"lowest keel point at +x end {keel_plus:+.3f} m, at -x end {keel_minus:+.3f} m")

    for h in HULLS:
        p = hull_points(stage, h)
        inner, outer = side_variation(p, inner=True), side_variation(p, inner=False)
        side_ok = (p[:, 1].mean() > 0) == ("Port" in h)        # +y is port
        report(f"{Path(h).name} placement", side_ok, f"mean y = {p[:, 1].mean():+.3f} m")
        report(f"{Path(h).name} flat wall inboard", inner < 0.5 * outer,
               f"inboard wall spread {inner:.3f} m, outboard {outer:.3f} m")

    com = stage.GetPrimAtPath("/World/BlueBoat").GetAttribute("physics:centerOfMass").Get()
    report("centre of mass lateral", abs(com[1]) < MAX_COM_LATERAL_M, f"com = {tuple(round(c, 4) for c in com)}")

    print("ORIENTATION CHECK", "FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
