"""
make_timing_figure.py — Build fig_timing.pdf from rescue_timing.csv.

No GPU needed. Run analyse_rescue_timing.py first to produce the CSV.

Produces a three-panel figure in the same house style as the other thesis
figures:

  (a) cumulative rescues and losses against episode time
  (b) distribution of rescue and expiry times
  (c) outcome against the casualty's initial deadline  <- the triage test

Panel (c) is the one that matters. If the rescued fraction is flat across the
40-90 s deadline draw, the policy is ignoring urgency and the priority term in
the observation is doing no work. If it slopes, the policy is triaging.
"""

import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "rescue_timing.csv"
OUT = "overleaf/figures/fig_timing.pdf"

INK, MUTE, GRID = "#16202b", "#5d6b7a", "#dfe4ea"
NAVY, TEAL, BRICK, SLATE = "#1f3b57", "#2a8f83", "#b23a2f", "#7c8894"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 8.2,
    "axes.edgecolor": MUTE, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

def tidy(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, lw=0.6)
    return ax

if not os.path.exists(CSV):
    raise SystemExit(f"{CSV} not found. Run analyse_rescue_timing.py first.")

rows = list(csv.DictReader(open(CSV)))
t   = np.array([float(r["time_s"]) for r in rows])
out = np.array([r["outcome"] for r in rows])
dl  = np.array([float(r["initial_deadline_s"]) for r in rows])
n_ep = len(set(r["episode"] for r in rows))
n_env = len(set(r["env"] for r in rows))
n_cas = len(set(r["casualty"] for r in rows))
total = n_ep * n_env * n_cas
T = t.max()

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))

# (a) cumulative outcomes over episode time
ax = axes[0]
grid = np.linspace(0, T, 200)
for lab, col, mask in [("rescued", TEAL, out == "rescued"),
                       ("lost", BRICK, out == "expired")]:
    cum = [(t[mask] <= g).sum() / total for g in grid]
    ax.plot(grid, cum, color=col, lw=1.6, label=lab)
tidy(ax); ax.set_title("(a)  cumulative outcome", loc="left")
ax.set_xlabel("episode time (s)"); ax.set_ylabel("fraction of all casualties")
ax.set_xlim(0, T); ax.set_ylim(0, 1)
ax.legend(loc="upper left", handlelength=1.7)

# (b) when they happen
ax = axes[1]
bins = np.linspace(0, T, 13)
ax.hist([t[out == "rescued"], t[out == "expired"]], bins=bins, stacked=False,
        color=[TEAL, BRICK], label=["rescued", "lost"], rwidth=0.85)
tidy(ax); ax.set_title("(b)  timing distribution", loc="left")
ax.set_xlabel("episode time (s)"); ax.set_ylabel("casualties")
ax.legend(loc="upper right", handlelength=1.5)

# (c) THE TRIAGE TEST: outcome vs initial deadline
ax = axes[2]
edges = np.linspace(dl.min(), dl.max(), 6)
centres, frac, ns = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dl >= lo) & (dl < hi if hi < edges[-1] else dl <= hi)
    if m.sum() == 0: continue
    centres.append((lo + hi) / 2)
    frac.append((out[m] == "rescued").mean())
    ns.append(int(m.sum()))
ax.plot(centres, frac, color=NAVY, lw=1.6, marker="o", ms=4)
overall = (out == "rescued").mean()
ax.axhline(overall, color=SLATE, lw=1.0, ls=(0, (4, 3)))
ax.text(edges[-1], overall, f" overall {overall:.2f}", fontsize=7.4,
        color=SLATE, va="center", ha="right")
tidy(ax); ax.set_title("(c)  outcome vs. initial deadline", loc="left")
ax.set_xlabel("initial deadline (s)"); ax.set_ylabel("fraction rescued")
ax.set_ylim(0, 1)

fig.tight_layout(w_pad=2.0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=400)

print(f"wrote {OUT}")
print(f"\n  casualties        {len(rows)} over {n_ep} episodes x {n_env} envs")
print(f"  rescued           {(out=='rescued').sum()}  ({overall:.4f})")
print(f"  median rescue     {np.median(t[out=='rescued']):.1f} s of {T:.0f} s")
print(f"  median expiry     {np.median(t[out=='expired']):.1f} s")
print("\n  fraction rescued by initial deadline band:")
for c, f, n in zip(centres, frac, ns):
    print(f"    {c:5.1f} s   {f:.3f}   (n={n})")
print("\n  If (c) is flat, the policy is not using urgency and that is a")
print("  finding worth reporting. If it slopes, it is triaging.")
