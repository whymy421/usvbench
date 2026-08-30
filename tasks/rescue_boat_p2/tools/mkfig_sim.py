import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 8.5, "axes.linewidth": 0.6})
FR = "/tmp/fr2/"
CX, CY = 960, 540

def centre_crop(f, half=150, out=680, sharp=True):
    im = Image.open(FR + f).convert("RGB").crop((CX-half, CY-half, CX+half, CY+half))
    im = im.resize((out, out), Image.LANCZOS)
    if sharp:
        im = im.filter(ImageFilter.UnsharpMask(radius=2.0, percent=105, threshold=2))
    return np.array(im)

def panel(ax, arr, tag, sub=None):
    ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9aa3ad"); s.set_linewidth(0.7)
    t = tag if sub is None else f"{tag}   {sub}"
    ax.text(0.028, 0.955, t, transform=ax.transAxes, fontsize=8.4, va="top",
            color="#15202b", bbox=dict(fc="white", ec="none", alpha=0.86, pad=2.2))

fig = plt.figure(figsize=(7.1, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.30, 1.0], hspace=0.10, wspace=0.055,
                      left=0.006, right=0.994, top=0.988, bottom=0.012)

# (a) wide field of view with locator box and magnified inset
axw = fig.add_subplot(gs[0, :])
full = np.array(Image.open(FR + "g0386.png").convert("RGB"))
axw.imshow(full); axw.set_xticks([]); axw.set_yticks([])
for s in axw.spines.values():
    s.set_color("#9aa3ad"); s.set_linewidth(0.7)
H = 105
axw.add_patch(Rectangle((CX-H, CY-H), 2*H, 2*H, fill=False, ec="#c0392b", lw=1.2))
ins = axw.inset_axes([0.655, 0.075, 0.325, 0.80])
ins.imshow(centre_crop("g0386.png", half=105, out=680))
ins.set_xticks([]); ins.set_yticks([])
for s in ins.spines.values():
    s.set_color("#c0392b"); s.set_linewidth(1.2)
axw.indicate_inset([CX-H, CY-H, 2*H, 2*H], ins, ec="#c0392b", lw=0.85, alpha=0.95)
ins.annotate("", xy=(0.505, 0.545), xytext=(0.30, 0.79), xycoords="axes fraction",
             textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
ins.text(0.115, 0.845, "vessel", transform=ins.transAxes, fontsize=8.2,
         color="#c0392b")
axw.text(0.012, 0.955, "(a)   full field of view at operating range",
         transform=axw.transAxes, fontsize=8.4, va="top", color="#15202b",
         bbox=dict(fc="white", ec="none", alpha=0.86, pad=2.2))

# (b)-(d) approach at three urgency levels
rows = [("g0100.png", "(b)", "low urgency"),
        ("g0161.png", "(c)", "rising urgency"),
        ("g0191.png", "(d)", "imminent expiry")]
for k, (f, tag, sub) in enumerate(rows):
    panel(fig.add_subplot(gs[1, k]), centre_crop(f, half=150), tag, sub)


fig.savefig("/tmp/build/fig_sim_grid.png", dpi=400)
fig.savefig("/tmp/build/fig_sim_grid.pdf")
print("ok")
