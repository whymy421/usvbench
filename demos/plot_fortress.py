"""Render fortress layout pictures for the owner's go/no-go review.

Pure CPU: samples real layouts from the shipped generators and draws them with
matplotlib, so what he approves is exactly what the env will instantiate.
"""
import math
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Annulus

# Load the geometry module by file path: the package __init__ imports
# gymnasium, which the plotting python does not have and does not need.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "hazard_geometry",
    r"C:\Users\BRADY\usvbench\tasks\hazard_nav\hazard_geometry.py",
)
_geo = importlib.util.module_from_spec(_spec)
# dataclasses resolves the defining module through sys.modules at class
# creation; register the module first or exec fails with a None lookup.
sys.modules["hazard_geometry"] = _geo
_spec.loader.exec_module(_geo)
sample_ring_fortress_layout = _geo.sample_ring_fortress_layout
sample_double_ring_fortress_layout = _geo.sample_double_ring_fortress_layout
FORTRESS_SPAWN_RADIUS_RANGE_M = _geo.FORTRESS_SPAWN_RADIUS_RANGE_M
DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M = (
    _geo.DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M
)


def draw(ax, layout, spawn_range, title):
    n = layout.obstacle_count
    for (x, y), r in zip(layout.centers[:n], layout.radii[:n]):
        ax.add_patch(Circle((x, y), r, facecolor="#c8c8c8", edgecolor="#555", lw=0.8))
    lo, hi = spawn_range
    ax.add_patch(Annulus((0, 0), hi, hi - lo, facecolor="#2e7d32", alpha=0.10))
    gx, gy = layout.goal
    ax.add_patch(Circle((gx, gy), 1.5, fill=False, edgecolor="#f5a300", lw=2.5))
    ax.plot([gx], [gy], marker="*", ms=14, color="#f5a300")
    sx, sy = layout.start
    ax.plot([sx], [sy], marker="^", ms=13, color="#1565c0",
            markeredgecolor="k", zorder=5)
    ang = math.atan2(gy - sy, gx - sx)
    ax.annotate("", xy=(sx + 3.2 * math.cos(ang), sy + 3.2 * math.sin(ang)),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.8))
    ax.set_title(title, fontsize=11)
    lim = hi + 4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.4)


rng = np.random.default_rng(7)
fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))
for col, level in enumerate((0, 1, 3)):
    layout, _ = sample_ring_fortress_layout(level, rng=rng, max_attempts=80)
    draw(axes[0][col], layout, FORTRESS_SPAWN_RADIUS_RANGE_M,
         f"single fortress  tier {level}")
for col, level in enumerate((0, 1, 3)):
    layout, _ = sample_double_ring_fortress_layout(level, rng=rng, max_attempts=120)
    draw(axes[1][col], layout, DOUBLE_RING_FORTRESS_SPAWN_RADIUS_RANGE_M,
         f"double fortress  tier {level}")
fig.suptitle(
    "Ring fortress family - goal (star) sealed at center, boat (triangle) spawns "
    "in the green annulus; grey = obstacle cylinders (to scale, boat is 1.2 m)",
    fontsize=12,
)
fig.tight_layout()
out = r"C:\Users\BRADY\usvbench\demos\fortress_layouts.png"
fig.savefig(out, dpi=110)
print("SAVED", out)
