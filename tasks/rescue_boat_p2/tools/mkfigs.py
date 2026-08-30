import csv, math, os, collections
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/sessions/loving-jolly-euler/mnt/Individual Project/"
OUT  = "/tmp/build/figs/"
L    = ROOT + "logs/skrl/usvbench/"
SUF  = "_ppo_torch_rescue_boat_v2_s42/"

# ---------------------------------------------------------------- house style
INK, MUTE, GRID = "#16202b", "#5d6b7a", "#dfe4ea"
NAVY, TEAL, AMBER, BRICK, SLATE = "#1f3b57", "#2a8f83", "#d8973c", "#b23a2f", "#7c8894"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 8.2,
    "axes.edgecolor": MUTE, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

def tidy(ax, ygrid=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    return ax

def thresh(ax, y=0.70, label="pass threshold  0.70", x=0.985, ha="right"):
    ax.axhline(y, color=BRICK, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.text(x, y, label, transform=ax.get_yaxis_transform(), ha=ha,
            va="bottom", fontsize=7.6, color=BRICK)

def panel_tag(ax, t):
    ax.set_title(t, loc="left", fontsize=9, color=INK, pad=5)

def rd(p):
    return list(csv.DictReader(open(p)))

def steps(name):
    return int(name.replace("agent_", "").replace(".pt", ""))

# ---------------------------------------------------------------- data
ORIG_SEEDS = ["2026-08-17_20-20-17", "2026-08-18_00-31-52", "2026-08-18_14-20-31",
              "2026-08-18_23-34-49", "2026-08-19_02-42-31"]
FIXED_SEEDS = ["2026-08-17_20-20-17", "2026-08-18_23-34-49", "2026-08-19_11-28-39"]
V4_SEEDS = ["2026-08-22_17-29-30", "2026-08-22_19-28-22", "2026-08-22_21-19-50"]
V3_SEEDS = ["2026-08-22_01-23-51", "2026-08-22_03-04-09", "2026-08-22_04-44-47"]

def orig_curve(d):
    r = rd(L + d + SUF + "eval_sweep.csv")
    return ([steps(x["checkpoint"]) for x in r],
            [float(x["targets_per_episode"]) / 4.0 for x in r])

def fixed_curve(d):
    r = rd(L + d + SUF + "eval_fixed.csv")
    return ([steps(x["checkpoint"]) for x in r],
            [float(x["rescue_rate"]) for x in r])

def v4_curve(d):
    r = rd(L + d + SUF + "eval_sweep.csv")
    return ([steps(x["checkpoint"]) for x in r],
            [float(x["rescue_rate"]) for x in r])

# ============================================================ 1. fig_metric
fig, ax = plt.subplots(figsize=(4.3, 2.85))
for i, d in enumerate(ORIG_SEEDS):
    x, y = orig_curve(d)
    ax.plot(np.array(x) / 1e3, y, color=SLATE, lw=0.9, marker="o", ms=2.6,
            alpha=0.85, zorder=2, label="original harness (5 seeds)" if i == 0 else None)
for i, d in enumerate(FIXED_SEEDS):
    x, y = fixed_curve(d)
    ax.plot(np.array(x) / 1e3, y, color=NAVY, lw=1.4, marker="s", ms=3.0,
            zorder=3, label="corrected harness (3 seeds)" if i == 0 else None)
tidy(ax)
ax.set_xlabel("training steps  ($\\times 10^{3}$)")
ax.set_ylabel("rescue rate")
ax.set_ylim(-0.03, 0.62)
ax.text(300, 0.40, "identical spikes,\nall five seeds", fontsize=7.6,
        color=MUTE, ha="left", va="center")
ax.legend(loc="upper left", handlelength=1.7, bbox_to_anchor=(-0.01, 1.05))
ax.set_ylim(-0.03, 0.72)
fig.savefig(OUT + "fig_metric.pdf"); plt.close(fig)

# ============================================================ 2. fig_physics
def probe(path, test):
    r = [x for x in rd(path) if x["test"] == test]
    t = [float(x["t_s"]) for x in r]
    z = [float(x["z_m"]) for x in r]
    roll = [abs(float(x["roll_deg"])) for x in r]
    return t, z, roll

t_b, z_b, r_b = probe(ROOT + "todays meeting/physics_probe.csv", "turn_sweep")
t_a, z_a, r_a = probe(ROOT + "physics_probe.csv", "turn_sweep")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
ax = axes[0]
ax.plot(t_b, z_b, color=BRICK, lw=1.5, label="before correction")
ax.plot(t_a, z_a, color=TEAL, lw=1.7, label="after correction")
ax.axhline(-0.60, color=MUTE, lw=0.8, ls=(0, (4, 3)))
ax.text(29.4, -0.6, "equilibrium draught", ha="right", va="bottom",
        fontsize=7.4, color=MUTE)
tidy(ax); panel_tag(ax, "(a)  hull height")
ax.set_xlabel("time (s)"); ax.set_ylabel("$z$ (m)")
ax.legend(loc="lower left", handlelength=1.7)

ax = axes[1]
ax.plot(t_b, r_b, color=BRICK, lw=1.5)
ax.plot(t_a, r_a, color=TEAL, lw=1.7)
ax.axhline(90, color=MUTE, lw=0.8, ls=(0, (4, 3)))
ax.text(0.4, 86, "beam ends ($90^{\\circ}$)", ha="left", va="top", fontsize=7.4, color=MUTE)
tidy(ax); panel_tag(ax, "(b)  roll angle")
ax.set_xlabel("time (s)"); ax.set_ylabel("$|\\phi|$ (deg)")
ax.set_ylim(-6, 185); ax.set_yticks([0, 45, 90, 135, 180])
ax.annotate("peak $1.6^{\\circ}$ through a", xy=(20.5, 1.6), xytext=(8.0, 40),
            fontsize=7.4, color=TEAL, ha="left",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))
ax.text(8.0, 33, "full sweep of heading", fontsize=7.4, color=TEAL, ha="left")
ax.annotate("hull inverts", xy=(13.6, 163), xytext=(1.5, 176), fontsize=7.4,
            color=BRICK, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=BRICK, lw=0.8))
fig.tight_layout(w_pad=2.2)
fig.savefig(OUT + "fig_physics.pdf"); plt.close(fig)

# ============================================================ 3. fig_yawinv
psi = np.linspace(0, 360, 721)
err = np.abs(psi % 360)
err = np.where(err > 180, 360 - err, err)
fig, ax = plt.subplots(figsize=(4.3, 2.6))
ax.plot(psi, err, color=NAVY, lw=1.6)
ax.fill_between(psi, 0, err, color=NAVY, alpha=0.07)
tidy(ax)
ax.set_xlabel("hull heading $\\psi$ (deg)")
ax.set_ylabel("moment direction error (deg)")
ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
ax.set_ylim(0, 195); ax.set_yticks([0, 45, 90, 135, 180])
ax.axvline(0, color=TEAL, lw=1.0, ls=(0, (4, 3)))
ax.annotate("all single-axis\ntesting sits here", xy=(2, 4), xytext=(30, 128),
            fontsize=7.6, color=TEAL, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8,
                            connectionstyle="arc3,rad=0.25"))
ax.annotate("applied moment does\nnothing to reduce tilt", xy=(90, 90),
            xytext=(150, 42), fontsize=7.6, color=BRICK, ha="left",
            arrowprops=dict(arrowstyle="->", color=BRICK, lw=0.8,
                            connectionstyle="arc3,rad=-0.2"))
fig.savefig(OUT + "fig_yawinv.pdf"); plt.close(fig)

# ============================================================ 4. fig_oracle
labels = ["original\n(r = 5 m,\nspawn $\\leq$ 50 m)",
          "intermediate\n(r = 15 m,\noriginal spawn)",
          "adopted\n(r = 15 m,\nspawn 80–200 m)"]
vals = [0.2617, 0.6680, 0.5820]
fig, ax = plt.subplots(figsize=(4.3, 2.9))
xs = np.arange(3)
bars = ax.bar(xs, vals, width=0.52, color=[SLATE, SLATE, NAVY],
              edgecolor="none", zorder=2)
for x, v in zip(xs, vals):
    ax.text(x, v + 0.018, f"{v:.3f}", ha="center", fontsize=8.2, color=INK)
ax.errorbar([2.85], [0.7695], yerr=[0.0355], color=TEAL, lw=1.2, capsize=3,
            marker="D", ms=5.5, zorder=4)
ax.text(2.85, 0.7695 + 0.075, "learned policy\n$0.770 \\pm 0.036$", fontsize=7.8,
        color=TEAL, ha="center", va="bottom")
ax.plot([2.15, 2.62], [0.582, 0.7695], color=TEAL, lw=0.7, ls=":", zorder=1)
thresh(ax, 0.70, "pass threshold  0.70", x=0.02, ha="left")
tidy(ax)
ax.set_ylabel("rescue rate, scripted controller")
ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, 3.35)
ax.set_xticks(list(xs) + [2.85])
ax.set_xticklabels(labels + ["learned\npolicy\n(adopted)"], fontsize=7.7)
fig.savefig(OUT + "fig_oracle.pdf"); plt.close(fig)

# ============================================================ 5. fig_ablation
v3_rr = [max(float(x["rescue_rate"]) for x in rd(L + d + SUF + "eval_sweep.csv"))
         for d in V3_SEEDS]
v4_rr = [max(float(x["rescue_rate"]) for x in rd(L + d + SUF + "eval_sweep.csv"))
         for d in V4_SEEDS]
metrics = [
    ("(a)  rescue rate", "rescue rate", v3_rr, v4_rr, 0.582, 0.70, "{:.3f}"),
    ("(b)  hull rotation", "revolutions per episode", [19.0], [12.0], 8.8, None, "{:.0f}"),
    ("(c)  hull vs. track", "heading error (deg)", [87.0], [59.0], 49.0, None, "{:.0f}"),
]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
for ax, (tag, ylab, a, b, ref, th, fm) in zip(axes, metrics):
    ma, mb = float(np.mean(a)), float(np.mean(b))
    ax.bar([0, 1], [ma, mb], width=0.5, color=[SLATE, NAVY], edgecolor="none", zorder=2)
    if len(a) > 1:
        ax.errorbar([0], [ma], yerr=[np.std(a)], color=INK, lw=1.0, capsize=3, zorder=4)
        ax.errorbar([1], [mb], yerr=[np.std(b)], color=INK, lw=1.0, capsize=3, zorder=4)
        ax.scatter(np.full(len(a), 0.0) + np.linspace(-.11, .11, len(a)), a,
                   s=9, color=INK, zorder=5, alpha=0.8)
        ax.scatter(np.full(len(b), 1.0) + np.linspace(-.11, .11, len(b)), b,
                   s=9, color=INK, zorder=5, alpha=0.8)
    ax.axhline(ref, color=TEAL, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(1.52, ref, "scripted\ncontroller", fontsize=7.2, color=TEAL,
            va="top", ha="left")
    top = max(ma, mb, ref) * 1.42
    if th is not None:
        ax.axhline(th, color=BRICK, lw=0.9, ls=(0, (5, 3)), zorder=1)
        ax.text(1.52, th, "threshold 0.70", fontsize=7.2, color=BRICK,
                va="bottom", ha="left")
    off = (np.std(a) if len(a) > 1 else 0.0) + top * 0.045
    ax.text(0, ma + off, fm.format(ma), ha="center", fontsize=8.2, color=INK)
    off = (np.std(b) if len(b) > 1 else 0.0) + top * 0.045
    ax.text(1, mb + off, fm.format(mb), ha="center", fontsize=8.2, color=INK)
    tidy(ax); panel_tag(ax, tag)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["without\nshaping", "with\nshaping"], fontsize=7.8)
    ax.set_ylabel(ylab); ax.set_ylim(0, top); ax.set_xlim(-0.55, 2.55)
fig.tight_layout(w_pad=2.6)
fig.savefig(OUT + "fig_ablation.pdf"); plt.close(fig)

# ============================================================ 6. fig_learning
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
ax = axes[0]
for i, d in enumerate(FIXED_SEEDS):
    x, y = fixed_curve(d)
    ax.plot(np.array(x) / 1e3, y, lw=1.3, marker="o", ms=2.8,
            color=[NAVY, TEAL, AMBER][i], label=f"seed {i+1}")
thresh(ax, 0.70, "pass threshold  0.70", x=0.02, ha="left")
tidy(ax); panel_tag(ax, "(a)  original configuration")
ax.set_xlabel("training steps  ($\\times 10^{3}$)"); ax.set_ylabel("rescue rate")
ax.set_ylim(0, 0.92)
ax.text(0.5, 0.16, "no learning across $6\\times10^{5}$ steps",
        transform=ax.transAxes, ha="center", fontsize=7.8, color=MUTE)

ax = axes[1]
for i, d in enumerate(V4_SEEDS):
    x, y = v4_curve(d)
    ax.plot(np.array(x) / 1e3, y, lw=1.3, marker="o", ms=2.8,
            color=[NAVY, TEAL, AMBER][i], label=f"seed {i+1}")
thresh(ax, 0.70, "", x=0.02, ha="left")
tidy(ax); panel_tag(ax, "(b)  corrected environment, calibrated task")
ax.set_xlabel("training steps  ($\\times 10^{3}$)")
ax.legend(loc="lower right", ncol=3, handlelength=1.5, columnspacing=1.1)
fig.tight_layout(w_pad=1.8)
fig.savefig(OUT + "fig_learning.pdf"); plt.close(fig)

print("figures:", sorted(os.listdir(OUT)))
print("V3", [round(v,4) for v in v3_rr], "V4", [round(v,4) for v in v4_rr])
