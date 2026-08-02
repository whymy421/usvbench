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
| ~~P0-1~~ | ~~势函数在 v8 上认证 ≥82%~~ | ❌ **07-31 判决:不通过,且判据本身是错的**。s42 100%/100% 但路径 68.6 m(直线 29.7),s43 35.9%/38.3% 且碰撞 30-34%。见 §1.6 | 已跑完 |
| P0-1b | **势函数改在强制穿越题上验** | `Isaac-USV-HazardCross-Direct-v1` 2 seed,**认证必须同时报路径长度**。在绕行可行的题上,成功率单独一个数字没有判别力 | GPU |
| ~~P0-2~~ | ~~强制穿越版题A 生成器~~ | ✅ **完成 07-31**:`Isaac-USV-HazardCross-Direct-v1`,10,000 张 0 违规。见 §1.5 | CPU |

**为什么 P0-1 排第一**:势函数已经崩过两次(0% 和 1.6%),两次都是理论正确、实跑失败。
第三个版本(v8)在打分台和回归测试上都通过,但**一次都没训过**。
在它证明自己之前,它是嫌疑品不是工具。

### 1.5 新的两级阶梯(07-31 建成并已零样本测过)

| 题号 | 内容 | 与上一级的唯一差别 |
|---|---|---|
| `Isaac-USV-HazardBasin-Direct-v1` | 封闭水池,里面什么都没有 | — |
| `Isaac-USV-HazardCross-Direct-v1` | 同样的水池 + 单门舱壁 | **舱壁** |

周界分布逐字相同,所以两题之差**干净地隔离出"穿缝"这一项能力**。

**v11 冠军零样本(每格 128 局 × 2 评测种子)**:两题**都是 0/128、碰撞 100%**,
路径中位 4.5-4.95 m。tier 0(门 4.50 m)和 tier 3(门 1.80 m)**一个数都不动**。
→ **墙本身足以解释全部失败,门的贡献为零。**

⚠️ **别再拿零样本去问"多少成绩靠绕行"** —— 冠军在够到门之前就死在墙上了,
这条路已经用对照实验排除。要回答只能在穿越题上真训,再比路径长度。

真实结论:**题A 的无边界场地不只是计分漏洞,它是那个策略的支撑条件** ——
"绕大圈"策略一旦有墙就自毁。

- 生成器 `tasks/hazard_nav/hazard_geometry.py:sample_forced_crossing_layout`
- 准入审计 `tasks/hazard_nav/test_forced_crossing.py` —— **10,000 张 0 违规**
- 奖励体检 `tasks/hazard_nav/test_crossing_reward_cost.py` —— 穿门最贵 0.98(tier 3)
- 档位 = 门宽 5/4/3/2 倍船宽,**物理米**,自己一张表 `FORCED_GATE_BEAMS`
- `max_obstacles = 96`(实测最多 79 根墙圆柱)

⚠️ 两个已知边界,别读大这个 PASS:路径长度四档几乎不变(难度全在"要多准");
半 sin 弧的 portal 上限是缝宽 4.50 m,**tier 0 的 4.495 m 压在边界上会时有时无**。

### 1.6 为什么 P0-1 的判据是错的(别再犯)

题A 绕行 1.63-1.82 倍即可满分,基线 s43 本来就 100%。**在这种题上设成功率门槛,
通过它反而会给"教绕行的干预"发合格证。**

同一段代码重算所有题A 认证的成功路径长度(直线 29.7 m):

| | SR (e42/e123) | path 中位 | 净空中位 |
|---|---|---|---|
| 基线 v3 s42 | 0.844 / 0.820 | 32.8 / 34.0 | 1.41 |
| 基线 v3 s43 | 1.000 / 1.000 | 53.6 / 53.4 | 2.82 |
| 可行性池化 v5 s42 | **1.000 / 1.000** | **33.3 / 31.7** | 1.00 |
| 非负势 v8 s42 | **1.000 / 1.000** | **68.6 / 69.0** | 1.77 |
| 非负势 v8 s43 | 0.359 / 0.383 | 79.1 / 75.2 | 1.91(碰撞 30-34%) |

v5 和 v8 都把 s42 打到 100%——一个 33 m 一个 69 m。**同样的头条数字,相反的行为。**

三条固化下来的规矩:
1. **认证必须报路径长度**(`eval_v6_frozen.py` 已加;`d0` 会被 `_reset_idx` 覆盖,
   所以在 `step()` 之前快照)。
2. **每个认证对必须过独立性检查** `scripts/check_eval_seed_independence.py`
   ——靠泊那次两个评测种子跑出逐位相同的 128 局,光看数字看不出来。
3. **待验假设**:折扣形式 `gamma*Phi(s')-Phi(s)` 本身会吹大路径(v4 和 v8 都涨,
   不用折扣的基线和 v5 都不涨)。同配方只改折扣开关跑一次对照即可证伪。

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

**2026-07-31 笔记本装机新增四坑**(每一个都花了半小时以上):

1. **计划任务的工作目录是 `C:\Windows\System32`,普通用户不能在那里建目录。**
   `train.py` 是 Hydra 应用,会在**当前目录**下 mkdir `outputs\<日期>\<时间>` →
   `PermissionError: [WinError 5] Access is denied: 'outputs'`,Isaac 启动一半就退出。
   **迷惑点**:`eval_v6_frozen.py` 不是 Hydra 应用,所以同一个计划任务里评测正常、
   只有训练挂,看起来像"训练专有的栈问题"。
   → 脚本开头必须 `Set-Location` 到一个可写目录(我们用 `C:\usvb\runs`)。
2. **PowerShell 5.1 的 `2>&1 | ... | Out-File` 会把原生程序的 stderr 包成 ErrorRecord**,
   **traceback 进不了日志** —— 第一次排查只看到 GPU 横幅,误判成原生崩溃。
   → 所有 python 调用一律 `cmd /c "... > log 2>&1"`,在 cmd 里重定向。
3. **`winget install` 在计划任务里静默失败**(不装成功也不报错)。
   后续 `git clone` 瞬间返回,脚本没验证就继续 pip install 一个不存在的目录。
   → 改用 GitHub API 取 PortableGit 自解压包;**每一步都验证产物存在,别只看返回码**。
4. **uv 会挪走它管理的 Python,venv 的 trampoline 随之失效。**
   `Scripts\python.exe` 是 uv trampoline,目标路径**烧死在 exe 里**,改 `pyvenv.cfg` 无效;
   旧路径 `cpython-3.11-windows-x86_64-none` 变成断掉的 junction,新路径带上了补丁号。
   → 把真解释器 `python.exe` 直接复制进 `Scripts\`(CPython venv 在 Windows 上本来就是
   这个布局,靠上一级 `pyvenv.cfg` 找 stdlib,没有烧死的路径)。
   ⚠️ **绝不要对 junction 用 `Remove-Item -Recurse`** —— 它会跟着链接删掉目标,
   而目标是这台机器上唯一一份解释器。

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
