# USVBench 交接单 — 任何一台机器上的 Claude Code 先读这份

> 这份文件在仓库里,所以 **clone 下来就等于把记忆带过去了**。
> 最后更新 2026-07-31。分支 `jinshi-brady/benchmark-v2`。

---

## 0. 你是谁、在做什么

USVBench:Isaac Lab 上的水面无人艇(BlueBoat 双体,17.3 kg,船宽 0.899 m,
差分推进,无侧推无刹车)强化学习基准,目标 NeurIPS Datasets & Benchmarks。

**当前阶段:任务有效性审计刚结束,正在按审计结论修题和重训。**

必读顺序:
1. 本文件
2. `../usvbench_gazebo/工作日志_USVBench.md` — 待办总账 + 全部结论与教训
3. `../usvbench_gazebo/竞品与定位_深读汇总_20260726.md` — 论文定位、必引文献、引用坑
4. 最近 15 条 git log

---

## 1. 现在的优先级清单(按顺序做,不要跳)

### P0 — 阻塞其他一切

| # | 任务 | 完成标准 | 机器 |
|---|---|---|---|
| P0-1 | **势函数在单一任务上跑通** | 用 `Isaac-USV-HazardNav-Direct-v8`(非负势 = 已完成路程比例)训 2 seed,认证 ≥ 基线 82%。**在拿到这个证据之前,禁止把新势函数推广到任何其他任务** | GPU |
| P0-2 | **强制穿越版题A 生成器** | 封闭航道 + 横墙 + 单门;准入必须包含"堵门后寻路无解";10,000 张布局 0 违规 | CPU |

**为什么 P0-1 排第一**:势函数已经崩过两次(0% 和 1.6%),两次都是理论正确、实跑失败。
第三个版本(v8)在打分台和回归测试上都通过,但**一次都没训过**。
在它证明自己之前,它是嫌疑品不是工具。

### P1 — 修题(P0-2 完成后)

| # | 任务 | 完成标准 |
|---|---|---|
| P1-1 | 强制穿越题A 训练 + 认证 | 2 seed 先探路,成功轨迹 ≥99% 真的穿门;四档要有可测差异 |
| P1-2 | 环形围困旧数字逐局核验 | 用 `usvbench_cloud_20260726/cert_ring_siege_*.json`:成功局最小净空若普遍 <0.5 m,说明是从漏洞钻的 → 旧数字作废重训;否则保留 |
| P1-3 | 题B 穿障段加围栏 | 复用 P0-2 的航道边界;重测绕行代价须 >3 倍 |
| P1-4 | 靠泊改固定分层出生距离 | 删自适应课程,固定 {2,5,10,15,20,25} m;96k/192k/384k 探路 |

### P2 — 验方法(题修好之后才有意义)

| # | 任务 | 备注 |
|---|---|---|
| P2-1 | **owner 的半 sin 弧穿缝奖励**(`HazardNav-Direct-v6`) | 已实现+接线+8 条防刷分测试全过,**从未训练**。必须在强制穿越题上验,旧题A 没有必须穿的缝 |
| P2-2 | 可行性池化(`v5`) | 旧题A 上 2 seed 已跑:100%/100% 和 71%/78%,只能说明"没搞坏" |
| P2-3 | 线×避 v11 配方版(`PathHazard-Direct-v2`) | 已接线未训。**注意:它不需要等新势函数**,两件事解耦 |
| P2-4 | 真终止置零(`v7`) | 风险最高(单步仍偏向"远处撞"),排最后 |

### P3 — 套件D(论文第一贡献)

五轴:质量 / 阻力 / 推力尺度 / 电机时间常数 / **左右推力不平衡**(最干净的 novelty)。
训练/内插/外推三段包必须**事先冻结**,不许看了测试结果再挪边界。

### 已推迟(不在本轮)

套件S、动态障碍+避碰规则、三交叉、大 Boss、回合中途换动力学、感知退化、
浪的严重度标定、demo 波面渲染。

### 欠的 demo(全部未录)

泊×实体墙 70% 冠军、环形穿缝、题B 出港 99%、
**最有故事性的:题A 种子42 穿行 84% vs 种子43 绕行 100% 对比**。

---

## 2. 有效性审计结论(六个任务,一张表)

| 任务 | 判决 | 依据脚本 |
|---|---|---|
| 题A | 有洞**且被利用**(绕行 1.63-1.82 倍即可,100% 冠军走的就是绕行) | `tasks/hazard_nav/test_detour_feasibility.py` |
| 线×避 | **干净**(门钉死路线,每张图强制绕障 1.4 倍船宽) | `tasks/path_hazard/test_no_bypass.py` |
| 环形围困 | 曾有洞(表面间距 1.00 m > 船宽 0.899 m),**已封死** 0/30 泄漏 | `tasks/hazard_nav/test_ring_sealed.py` |
| 题B 穿障 | 有洞(绕行 1.81 倍),**未被策略利用** | `tasks/harbor_mission/test_transit_bypass.py` |
| 靠泊系列 | 无洞,是**标签问题**(课程卡在 2-3 m) | 冻结课程评测 level 3 vs 0 |
| 保持×流 | 干净 | — |

**共同教训**:准入协议只验证了"有解",没验证"无捷径"。三次"检查器和现实不一致"
(环形密封按 0.65 m 膨胀而仿真按 0.45、题A 档位、线×避阻挡账本 3 vs 2)。

---

## 3. 铁律(违反会毁掉数据可信度)

1. **追加式**:老题号、老配置、已认证数字**永不修改**。新行为一律新题号 + 默认关闭的开关。
2. **改奖励前必须先跑打分台**:`python scripts/reward_scenarios.py`,8 条判据必须全过。
3. **多种子**:训练至少 2 seed(正式 5 seed),认证用 2 个独立评测种子各 128 局。
4. **筛选**:训练存档要用确定性动作重新筛选,不能信训练曲线(虚高)。
   训练时必须传 `agent.agent.experiment.checkpoint_interval=3200`,否则 375 个存档会让
   筛选跑 4.8 小时(训练才 29 分钟)。
5. **平台对账**:新机器第一件事是用 `tasks/hazard_nav/checkpoints/hazard_nav_v11_best_s42.pt`
   在 `Isaac-USV-HazardNav-Direct-v3` 上跑 128 局 seed 123,**必须是 75.78%**,
   对不上就别信这台机器产出的任何数字。

---

## 4. 环境安装(Windows,已验证两次)

```powershell
# 1. Python 3.11 虚拟环境(conda 会和学术加速代理打架,用 uv)
uv venv C:\usvb\env --python 3.11

# 2. torch:Blackwell(50 系)必须 cu128
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 3. 先降 setuptools,否则 flatdict 构建失败会卡死 IsaacLab 安装
pip install "setuptools<80" wheel

# 4. Isaac Sim 5.1(约 20 GB,最长的一步)
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 5. IsaacLab v2.3.2,逐包 editable 安装
git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
pip install -e source/isaaclab --no-build-isolation      # 四个包都装
pip install "skrl==1.4.3"

# 6. 任务接线:把 usvbench/tasks/* 复制进
#    IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/
#    (_shared hazard_nav path_hazard harbor_mission docking station_keeping)
```

**必设环境变量**:`OMNI_KIT_ACCEPT_EULA=YES`(不设会卡在交互确认)、
`USVBENCH_ASSETS=<repo>\assets`。

**踩过的坑,别再踩**:
- 后台启动 Isaac 会卡 EULA(隐藏窗口切断了标准输入)→ 用计划任务或前台
- PowerShell 5.1 读**不带 BOM 的 UTF-8** 会按 ANSI 解释 → **脚本必须纯 ASCII**
- 反过来,PowerShell 生成的 yaml **带 BOM** 会让 Hydra 解析失败 → 要剥 BOM
- 训练必加 `agent.agent.experiment.wandb=False`(除非那台机器登过 wandb)
- 笔记本/工作站要关睡眠:`powercfg /change standby-timeout-ac 0`
- 杀毒软件排除安装目录,否则 Isaac Sim 解压时间翻倍

---

## 5. 配置够不够(判断新机器)

| 项 | 最低 | 说明 |
|---|---|---|
| 显存 | **8 GB** | 实测训练占 5 GB(64 并行环境);12 GB 舒适 |
| 显卡 | 任意 NVIDIA,RTX 30 系以上 | **50 系(Blackwell)必须 cu128 版 torch** |
| 内存 | 16 GB | 32 GB 舒适 |
| 硬盘 | **60 GB 空闲** | Isaac Sim 约 20 GB + IsaacLab + 存档 |
| 系统 | Windows 10/11 或 Ubuntu 22.04 | 两边都验过 |

**速度参考**(一次 96k 步训练):RTX 4090 约 29 分钟,RTX 5090 约 48 分钟(云机,
CPU 较弱),RTX 5070 Ti Laptop 预计 45-60 分钟。

---

## 6. 现有机器

| 机器 | 用途 | 接入方式 |
|---|---|---|
| 台式 RTX 4090 | owner 自用为主,偶尔借 | 本地 |
| MSI 笔记本 RTX 5070 Ti 12G | **第二训练机**,环境安装中 | Tailscale `100.99.34.15`,用户 `tyzen`,免密钥已装 |
| AutoDL 云机 RTX 5090 | 已关机(15 天内可开回) | 数据已全量备份到 `usvbench_cloud_20260726/` |

---

## 7. 给另一台机器的 Claude Code 的开场白

把下面这段原样发给它:

> 读 `HANDOFF.md`,然后读 `../usvbench_gazebo/工作日志_USVBench.md` 全文和最近 15 条
> git log。我们在做 USVBench(Isaac Lab 水面艇 RL 基准)。当前阶段是任务有效性
> 审计后的修题与重训。**先做优先级清单里的 P0-1**(势函数在单一任务上跑通),
> 不要跳步、不要同时改多个变量、不要修改任何已认证的题号。
> 每个改动都要有可跑的验证脚本,不能只靠"应该对"。
