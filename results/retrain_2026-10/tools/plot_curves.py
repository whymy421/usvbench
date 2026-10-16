"""Training curves for the README: one PNG per task row, reward | primary metric, three seeds.

Reward: tfevents 'Reward / Total reward (mean)' (training, stochastic policy).
Primary metric:
  bench rows  -> deterministic eval of each periodic checkpoint (results/full/curves, 2000 steps)
  other rows  -> tfevents 'Info / Episode/success' (training-time, stochastic policy)
"""

import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

R = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(R, "figures", "curves")
SEEDS = (42, 123, 456)
# reference palette slots 1-3 (validated all-pairs in light mode) and light chrome
# UCL palette (NavRL2026 EXPERIMENT_LOG.md): purple, cyan, pink; worst CVD pair dE 11.9 (OKLab x100)
SEED_COLOR = {42: "#500778", 123: "#00BED5", 456: "#AC145A"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0D0C1D", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

META = {
    "rov_calm": ("ROV calm navigation", "Doc Ricketts ROV", "calm water", "bench"),
    "boat_calm": ("Boat calm navigation", "5.5 m monohull", "calm water", "bench"),
    "blueboat_calm": ("BlueBoat calm navigation", "BlueBoat", "calm water", "bench"),
    "hazard_v3": ("Static hazard navigation (v3)", "BlueBoat", "calm water", "success"),
    "path_rov": ("Path following", "ROV", "calm water", "success"),
    "path_blueboat": ("Path following", "BlueBoat", "calm water", "success"),
    "path_hazard": ("Path hazard", "BlueBoat", "calm water", "success"),
    "dock_boat": ("Docking", "5 m monohull", "calm water", "success"),
    "dock_blueboat": ("Docking", "BlueBoat", "calm water", "success"),
    "dock_blueboat_current": ("Docking", "BlueBoat", "cross current", "success"),
    "sk_rov": ("Station keeping", "ROV", "calm water", "success"),
    "sk_blueboat": ("Station keeping", "BlueBoat", "calm water", "success"),
    "sk_blueboat_current": ("Station keeping", "BlueBoat", "cross current", "success"),
    "sk_boat": ("Station keeping", "5 m monohull", "calm water", "success"),
    "harbor": ("Ordered harbor mission", "BlueBoat", "calm water", "success"),
    "catamaran": ("Catamaran patrol", "Catamaran", "calm water", "bench"),
    "hazard_airy": ("Hazard navigation, preview", "BlueBoat", "Airy regular wave H 0.12 m, T 3 s", "success"),
    "hazard_jonswap": ("Hazard navigation, preview", "BlueBoat", "JONSWAP Hs 0.06-0.18 m, Tp 2-4 s", "success"),
}


def logdir(key, seed):
    hits = sorted(glob.glob(os.path.join(R, "logs", "skrl", "*", f"*_{key}_s{seed}")))
    return hits[-1] if hits else None


def scalars(d, tag):
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    s = ea.Scalars(tag)
    return [x.step for x in s], [x.value for x in s]


def bench_curve(key, seed):
    p = os.path.join(R, "results", "full", "curves", f"{key}_s{seed}.csv")
    if not os.path.exists(p):
        return [], []
    pts = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            step = int(re.findall(r"agent_(\d+)\.pt", r["checkpoint"])[0])
            pts.append((step, float(r["targets_per_episode"])))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot(key):
    name, vehicle, cond, kind = META[key]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=150, facecolor=SURFACE)
    for seed in SEEDS:
        d = logdir(key, seed)
        if not d:
            continue
        x, y = scalars(d, "Reward / Total reward (mean)")
        axes[0].plot([v / 1000 for v in x], y, color=SEED_COLOR[seed], linewidth=1.5, solid_capstyle="round",
                     label=f"seed {seed}")
        if kind == "bench":
            x, y = bench_curve(key, seed)
            axes[1].plot([v / 1000 for v in x], y, color=SEED_COLOR[seed], linewidth=1.5, marker="o", markersize=4.5,
                         markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"seed {seed}")
        else:
            x, y = scalars(d, "Info / Episode/success")
            axes[1].plot([v / 1000 for v in x], y, color=SEED_COLOR[seed], linewidth=1.5, label=f"seed {seed}")
    axes[0].set_title("Mean episode return (training)", fontsize=9.5, color=INK, loc="left")
    if kind == "bench":
        axes[1].set_title("Targets per episode (deterministic eval of each checkpoint)", fontsize=9.5, color=INK, loc="left")
        axes[1].set_ylim(bottom=0)
    else:
        axes[1].set_title("Episode success (training, stochastic policy)", fontsize=9.5, color=INK, loc="left")
        axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        style(ax)
        ax.set_xlabel("Training timesteps (thousands; 64 parallel envs)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(f"{name}  ·  {vehicle}  ·  {cond}", x=0.01, y=0.99, ha="left", fontsize=11, color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{key}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for k in META:
        print(plot(k))
