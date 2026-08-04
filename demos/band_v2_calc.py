"""Band fortress v2: the gap is a GUARANTEE, not a mean.

Owner's eye caught the v1 flaw: layer separation 2.5 m with radii up to 2.0 m
made the true diagonal passage ~0.9 m while the label said 2.7 m. v2 builds
constructively so that EVERY pairwise cylinder edge distance >= A (the tier
aperture):

  layer:  cylinders placed sequentially around the ring, edge-to-edge arc
          gap exactly A, radii random 0.8..2.0.
  plugs:  one big cylinder (r 1.6..2.0) behind EACH layer gap, at the gap's
          bearing, pushed radially outward just far enough that its edge
          distance to BOTH gap neighbours is >= A.

A radial ray through a layer gap meets the plug; going around the plug is the
guaranteed-width S-corridor. Two such bands = two forced compound dodges.
This script builds layouts, measures (a) min pairwise aperture (must be >= A),
(b) fraction of spawn->goal straight lines blocked by both bands, (c) goal
line-of-sight, and draws one precise panel per tier.
"""
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Annulus

HALF_BEAM = 0.45
R_LAYER = (0.8, 2.0)
R_PLUG = (1.6, 2.0)
INNER_R = 9.0
BAND_SEP = 6.0            # inner-band plugs to outer-band layer, roughly
SPAWN = (24.0, 30.0)
TIERS = {0: 6.0, 1: 4.5, 2: 3.6, 3: 2.7}
N_MC = 4000


def clear_of_all(cand, r, placed_c, placed_r, A):
    if len(placed_c) == 0:
        return True
    d = np.linalg.norm(np.asarray(placed_c) - cand, axis=1)
    return bool(np.all(d - np.asarray(placed_r) - r >= A - 1e-9))


def build_layer(radius, A, rng):
    """Sequential ring placement, edge gaps >= A including the wrap gap."""
    radii, angles = [], []
    theta = start = rng.uniform(0.0, 2.0 * math.pi)
    while True:
        r = rng.uniform(*R_LAYER)
        if radii:
            # CHORD spacing, not arc: the Euclid edge gap is what the hull
            # sails through, and arc-based spacing under-delivered by ~0.4 m.
            theta += 2.0 * math.asin(
                min(1.0, (radii[-1] + A + r) / (2.0 * radius))
            )
        if radii and (theta - start) > 2.0 * math.pi:
            break
        if radii:
            wrap = (start + 2.0 * math.pi) - theta
            need = 2.0 * math.asin(min(1.0, (r + A + radii[0]) / (2.0 * radius)))
            if wrap < need:
                break
        angles.append(theta)
        radii.append(r)
        if len(radii) > 200:
            break
    ang = np.array(angles)
    centers = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    return centers, np.array(radii), ang


def add_plugs(ang, ring_r, A, rng, all_c, all_r, s_max):
    """A plug behind each gap, pushed outward until >= A from EVERYTHING
    already placed (both layers, earlier plugs). Skips a gap only if no
    radius/offset combination fits; the caller counts skips."""
    pc, pr, skipped = [], [], 0
    n = len(ang)
    for i in range(n):
        j = (i + 1) % n
        a1 = ang[j] + (2.0 * math.pi if j == 0 else 0.0)
        mid = 0.5 * (ang[i] + a1)
        placed = False
        for r in (rng.uniform(*R_PLUG), 1.4, 1.1):
            s = 0.8
            while s <= s_max:
                cand = (ring_r + s) * np.array([math.cos(mid), math.sin(mid)])
                if clear_of_all(cand, r, all_c, all_r, A):
                    pc.append(cand); pr.append(r)
                    all_c.append(cand); all_r.append(r)
                    placed = True
                    break
                s += 0.25
            if placed:
                break
        if not placed:
            skipped += 1
    return pc, pr, skipped


def build_layout(A, rng):
    all_c, all_r = [], []
    c1, r1, ang1 = build_layer(INNER_R, A, rng)
    all_c += list(c1); all_r += list(r1)
    _, _, sk1 = add_plugs(ang1, INNER_R, A, rng, all_c, all_r,
                          s_max=BAND_SEP - 1.0)
    # outer band far enough that its layer clears the deepest inner plug by A
    outer_ring_r = INNER_R + BAND_SEP + A + 2.0 * R_PLUG[1]
    c2, r2, ang2 = build_layer(outer_ring_r, A, rng)
    kept_c2, kept_r2 = [], []
    for cc, rr in zip(c2, r2):
        if clear_of_all(cc, rr, all_c, all_r, A):
            kept_c2.append(cc); kept_r2.append(rr)
    for cc, rr in zip(kept_c2, kept_r2):
        all_c.append(cc); all_r.append(rr)
    _, _, sk2 = add_plugs(ang2, outer_ring_r, A, rng, all_c, all_r, s_max=10.0)
    n1 = len(c1) + (len(ang1) - sk1)
    band1_c, band1_r = all_c[:n1], all_r[:n1]
    band2_c, band2_r = all_c[n1:], all_r[n1:]
    ac = np.array(all_c); ar = np.array(all_r)
    outer_extent = float((np.linalg.norm(ac, axis=1) + ar).max())
    return ((np.array(band1_c), np.array(band1_r)),
            (np.array(band2_c), np.array(band2_r)), sk1 + sk2, outer_extent)


def min_aperture(centers, radii):
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    d -= radii[:, None] + radii[None, :]
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def seg_hits(p0, p1, centers, radii_infl):
    d = p1 - p0
    L2 = float(d @ d)
    t = np.clip(((centers - p0) @ d) / L2, 0.0, 1.0)
    closest = p0 + t[:, None] * d
    return bool(np.any(np.linalg.norm(centers - closest, axis=1) < radii_infl))


print("砖墙堡垒 v2 | 缝宽=保证值(任意两柱边距>=A) | 层柱 0.8-2.0 m, 塞柱 1.6-2.0 m")
print(f"内环 9 m | 环带间距 ~{BAND_SEP:.0f}+A m | 出生带 {SPAWN[0]:.0f}-{SPAWN[1]:.0f} m | 每档 {N_MC} 次")
print("=" * 96)
print(f"{'tier':>4} {'A(m)':>6} {'实测最小缝':>10} {'柱数':>6} "
      f"{'挡内带':>8} {'挡外带':>8} {'两次必避':>9} {'终点不可见':>10}")
print("-" * 96)
rng = np.random.default_rng(23)
results = {}
for tier, A in TIERS.items():
    min_ap = np.inf
    bi = bo = bb = los = skips = 0
    keep = None
    exts = []
    for k in range(N_MC):
        (c1, r1), (c2, r2), sk, ext = build_layout(A, rng)
        skips += sk
        exts.append(ext)
        allc, allr = np.vstack([c1, c2]), np.concatenate([r1, r2])
        if k < 200:
            min_ap = min(min_ap, min_aperture(allc, allr))
        if k == 0:
            keep = ((c1.copy(), r1.copy()), (c2.copy(), r2.copy()))
        b = rng.uniform(0, 2 * math.pi)
        # spawn annulus floats with the actual band extent (tier 0's thicker
        # bands would otherwise swallow a fixed 24-30 m spawn ring)
        s = rng.uniform(ext + 2.0, ext + 8.0)
        p0 = s * np.array([math.cos(b), math.sin(b)])
        p1 = np.zeros(2)
        h1 = seg_hits(p0, p1, c1, r1 + HALF_BEAM)
        h2 = seg_hits(p0, p1, c2, r2 + HALF_BEAM)
        bi += h1; bo += h2; bb += h1 and h2
        los += seg_hits(p0, p1, allc, allr)
    n = float(N_MC)
    results[tier] = (A, keep, float(np.median(exts)))
    print(f"{tier:>4} {A:>6.1f} {min_ap:>9.2f}m {len(allr):>6d} "
          f"{bi / n:>7.1%} {bo / n:>8.1%} {bb / n:>8.1%} {los / n:>9.1%}"
          f"   跳塞 {skips / n:.2f}/图  外沿中位 {np.median(exts):.1f}m")

fig, axes = plt.subplots(2, 2, figsize=(17, 17))
for tier, ax in zip(TIERS, axes.flat):
    A, ((c1, r1), (c2, r2)), ext_med = results[tier]
    spawn_lo, spawn_hi = ext_med + 2.0, ext_med + 8.0
    for (x, y), r in zip(np.vstack([c1, c2]), np.concatenate([r1, r2])):
        ax.add_patch(Circle((x, y), r, facecolor="#bfc7cc",
                            edgecolor="#37474f", lw=0.9))
    ax.add_patch(Annulus((0, 0), spawn_hi, spawn_hi - spawn_lo,
                         facecolor="#2e7d32", alpha=0.10))
    ax.add_patch(Circle((0, 0), 1.5, fill=False, edgecolor="#f5a300", lw=2.5))
    ax.plot(0, 0, marker="*", ms=16, color="#f5a300")
    rng2 = np.random.default_rng(tier + 40)
    b = rng2.uniform(0, 2 * math.pi)
    s = rng2.uniform(spawn_lo, spawn_hi)
    sx, sy = s * math.cos(b), s * math.sin(b)
    ax.plot(sx, sy, marker="^", ms=13, color="#1565c0",
            markeredgecolor="k", zorder=5)
    ax.plot([sx, 0], [sy, 0], ls="--", lw=1.1, color="#d32f2f", alpha=0.75)
    boat = plt.Rectangle((-spawn_hi + 2, -spawn_hi + 1.4), 1.2, 0.9,
                         facecolor="#1565c0", alpha=0.9)
    ax.add_patch(boat)
    ax.annotate("boat to scale (1.2 x 0.9 m)",
                (-spawn_hi + 3.6, -spawn_hi + 1.6),
                fontsize=8, color="#1565c0")
    ax.set_title(f"tier {tier}   guaranteed aperture A = {A:.1f} m",
                 fontsize=13)
    lim = spawn_hi + 2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(alpha=0.2, lw=0.4)
fig.suptitle(
    "Band fortress v2 -- every pairwise cylinder gap >= A by construction; "
    "each layer gap is backed by a plug cylinder (the forced S-dodge). "
    "Star = goal, triangle = boat spawn (green annulus), red dashed = greedy line.",
    fontsize=12)
fig.tight_layout()
out = r"C:\Users\BRADY\usvbench\demos\band_v2_layouts.png"
fig.savefig(out, dpi=115)
print("SAVED", out)
