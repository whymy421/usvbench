"""Draw the brick-wall band fortress (the computed recommendation)."""
import math
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Annulus

R_OBS = (0.8, 2.0)
INNER_R = 9.0
SUB_GAP = 2.5          # radial spacing between a band's two sub-rings
BAND_SEP = 6.0         # 5x LOA between the bands' facing sub-rings
SPAWN = (22.0, 28.0)


def grid_ring(radius, count, rng, phase):
    ang = phase + np.arange(count) * (2 * math.pi / count)
    rad = rng.uniform(*R_OBS, size=count)
    return radius * np.stack([np.cos(ang), np.sin(ang)], 1), rad


def band(radius, gap, rng, phase):
    r_mean = 0.5 * sum(R_OBS)
    count = max(6, int(round(2 * math.pi * radius / (2 * r_mean + gap))))
    c1, r1 = grid_ring(radius, count, rng, phase)
    c2, r2 = grid_ring(radius + SUB_GAP, count, rng, phase + math.pi / count)
    return np.vstack([c1, c2]), np.concatenate([r1, r2])


def draw(ax, gap, seed, title):
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(0, 2 * math.pi)
    band2_r = INNER_R + SUB_GAP + BAND_SEP
    cs1, rs1 = band(INNER_R, gap, rng, p1)
    cs2, rs2 = band(band2_r, gap, rng, rng.uniform(0, 2 * math.pi))
    for rr in (INNER_R, INNER_R + SUB_GAP, band2_r, band2_r + SUB_GAP):
        ax.add_patch(Circle((0, 0), rr, fill=False, edgecolor="#90a4ae",
                            lw=0.8, ls=":", alpha=0.8))
    for (x, y), r in zip(np.vstack([cs1, cs2]),
                         np.concatenate([rs1, rs2])):
        ax.add_patch(Circle((x, y), r, facecolor="#c8c8c8",
                            edgecolor="#555", lw=0.7))
    ax.add_patch(Annulus((0, 0), SPAWN[1], SPAWN[1] - SPAWN[0],
                         facecolor="#2e7d32", alpha=0.10))
    ax.add_patch(Circle((0, 0), 1.5, fill=False, edgecolor="#f5a300", lw=2.5))
    ax.plot([0], [0], marker="*", ms=15, color="#f5a300")
    b = rng.uniform(0, 2 * math.pi)
    s = rng.uniform(*SPAWN)
    sx, sy = s * math.cos(b), s * math.sin(b)
    ax.plot([sx], [sy], marker="^", ms=12, color="#1565c0",
            markeredgecolor="k", zorder=5)
    ax.plot([sx, 0], [sy, 0], ls="--", lw=1.0, color="#d32f2f", alpha=0.7)
    ax.set_title(title, fontsize=11)
    lim = SPAWN[1] + 3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.4)


fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
draw(axes[0], 6.0, 3, "tier 0  gap 6.0 m (5x LOA)  ~55% straight lines blocked twice")
draw(axes[1], 4.5, 5, "tier 1  gap 4.5 m (5x beam)  ~84%")
draw(axes[2], 2.7, 9, "tier 3  gap 2.7 m (3x beam)  ~98.5%")
fig.suptitle(
    "Brick-wall band fortress (computed design): each band = two staggered "
    "sub-rings 2.5 m apart -- every gap passable via an S-turn, straight "
    "greedy line (red dashed) blocked by BOTH bands; no search, no potential "
    "fighting. Star=goal, triangle=boat, green=spawn annulus.",
    fontsize=11,
)
fig.tight_layout()
out = r"C:\Users\BRADY\usvbench\demos\band_fortress_layouts.png"
fig.savefig(out, dpi=110)
print("SAVED", out)
