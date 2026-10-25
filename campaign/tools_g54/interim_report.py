"""Interim results table from validated certificates (VALID lines in run/validation)."""
import glob, re, collections, math, json, datetime
from pathlib import Path
B = Path(__file__).resolve().parent.parent
TASKS = ["reach","path_following","station_keeping","docking","path_hazard","scatter","ring","fortress","forced_crossing","iceberg","moving_basin","harbour_mission"]
ZH = {"reach":"到达","path_following":"沿线跟踪","station_keeping":"站位保持","docking":"靠泊","path_hazard":"沿线穿障","scatter":"散点","ring":"环形",
      "fortress":"要塞","forced_crossing":"强制穿越","iceberg":"冰山","moving_basin":"移动水池","harbour_mission":"港区航次"}
SUITE_ZH = {"waves":"浪","wind":"风","cur":"流","actcap":"推力上限","actgain":"单侧推力","actlag":"电机滞后","actres":"阻力","pnoise":"位置噪声",
            "pbias":"位置偏置","plat":"延迟","pdrop":"丢帧","dyn":"动态障碍","app":"突现障碍"}
SUITES = list(SUITE_ZH)
CTRL = {"cl":"经典","mp":"MPPI"}
cells = {}
for f in glob.glob(str(B / "run" / "validation" / "*.txt")):
    for l in open(f, encoding="ascii", errors="replace"):
        m = re.match(r"VALID (fc_\S+) .* success=(\d+)/(\d+)", l)
        if m:
            cells[m.group(1)] = (int(m.group(2)), int(m.group(3)))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)
grid = collections.defaultdict(dict)
for name, (k, n) in cells.items():
    m = re.match(r"fc_(.+?)_(bb|wamv)_(cl|mp|p\d+)_(.+)_f1$", name)
    if not m: continue
    task, ves, ctl, cond = m.groups()
    grid[(task, ves, ctl)][cond] = k / n
out = ["# 全量阶段性结果（%s，已核对 %d 格）" % (datetime.date.today(), len(cells)), "",
       "> 每格 128 局成功率。只列已出结果的格子；'·' 表示还没跑到。全部来自 final-01，证书已在 Linux 上核对。",
       "> 单格成功率的 95% 区间半宽约 ±0.04（p≈0.9）到 ±0.09（p≈0.5）。", ""]
for ves in ("wamv", "bb"):
    for ctl in ("cl", "mp"):
        rows = [t for t in TASKS if (t, ves, ctl) in grid]
        if not rows: continue
        out.append("## %s · %s" % ("WAM-V" if ves == "wamv" else "BlueBoat", CTRL[ctl])); out.append("")
        out.append("| 题 | 静水 | 叠加 | " + " | ".join("%s D1→D4" % SUITE_ZH[s] for s in SUITES) + " |")
        out.append("|---|---|---|" + "---|" * len(SUITES))
        for t in rows:
            g = grid[(t, ves, ctl)]
            calm = g.get("calm"); stack = next((v for k, v in g.items() if k.startswith("allsuites")), None)
            fmt = lambda v: "·" if v is None else "%.2f" % v
            cols = []
            for s in SUITES:
                vals = [g.get("%s_D%d" % (s, d)) for d in range(1, 5)]
                cols.append("·" if all(v is None for v in vals) else "/".join(fmt(v) for v in vals))
            out.append("| %s | %s | %s | %s |" % (ZH[t], fmt(calm), fmt(stack), " | ".join(cols)))
        out.append("")
# PPO
ppo = [(k, v) for k, v in grid.items() if k[2].startswith("p")]
if ppo:
    out.append("## PPO（WAM-V，按训练种子）"); out.append("")
    out.append("| 题 | 种子 | 静水 | 叠加 | 已出单项格数 | 单项平均成功率 |"); out.append("|---|---|---|---|---|---|")
    for (t, ves, ctl), g in sorted(ppo, key=lambda x: (TASKS.index(x[0][0]), x[0][2])):
        singles = [v for k, v in g.items() if re.match(r".+_D\d$", k)]
        stack = next((v for k, v in g.items() if k.startswith("allsuites")), None)
        out.append("| %s | %s | %s | %s | %d | %s |" % (ZH[t], ctl[1:], "%.2f" % g["calm"] if "calm" in g else "·",
                   "%.2f" % stack if stack is not None else "·", len(singles), "%.2f" % (sum(singles)/len(singles)) if singles else "·"))
    out.append("")
p = Path(r"C:/Users/BRADY/usvbench_gazebo/NC期刊/53_全量阶段性结果_%s.md" % datetime.date.today().strftime("%Y%m%d"))
p.write_text("\n".join(out) + "\n", encoding="utf-8"); print(p, len(cells))
