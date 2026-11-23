# PPO 在要塞、环形 0 分的诊断实验（分支 `yutong/ppo-fair-fix`）

起点：期刊版（journal/v2.6.6，13a8874）里 PPO 在 fortress（要塞）、ring（环形）都是 0 分，经典控制器 `planner_los_pid_mem_v2` 却能做到。
这个分支逐步给 PPO 加修复，看它缺的是什么。**所有改动默认关闭，不加开关时与发布版行为完全一致。**

## 先说清楚：三个控制器的输入是一样的

PPO、MPPI、经典控制器每一步读的都是 D5 位置估计、速度、36 根测距线、船头朝向、目标位置（`scripts/run_cell.py` 的 `JOURNAL_INFORMATION`），都不读障碍或门的真实位置。
经典控制器的航点不是外部给的，是它用同样的输入自己算出来的，分三层：

1. **记忆地图**：本局所有测距线碰到的点都记下来，整局不忘（`scripts/planner_memory.py`）；
2. **找路**：在这张地图上找最短路，每 2 m 放一个航点，每 60 步重算一次；
3. **跟航点**：转向用 PID 对准当前航点；推力按固定公式 `0.2 + 0.8·cos(角度偏差)`。

所以差别不在输入，而在于：经典控制器记住了过去的测距结果并做规划，PPO（全连接网络）每一步只看当前这一帧。

## 结果（BlueBoat，静水，布局种子 3001，128 局，均值动作，缺口 2 档）

参照：经典控制器有记忆 要塞 **111/128**、环形 **128/128**；去掉记忆 要塞 0/128、环形 100/128。

### 纯 PPO（第 1–5、8、9 轮）

| 修复 | 结果 |
|---|---|
| 按绕行路程算前进分 + 每步时间成本（`--fair-fix geo_time`） | 全 0 |
| 带记忆的网络（`--policy gru`） | 全 0 |
| 动作有界（`--action-dist tanh`，修好了学习率被压到下限、GRU 发散） | 全 0 |
| 超时也按失败罚（`--timeout-penalty`） | 全 0（船开始动，但到不了） |
| 缺口宽度课程（`--curriculum` / `--curriculum-adaptive`），3 倍训练量 | 全 0 |
| 降低失败罚（`--failure-penalty-no-events`）、新格子奖励（`--novelty-bonus`），仅环形 | MLP 种子 90001 有 15/128，种子 90002 是 0/128，没复现 |

共同现象：训练里几乎从来没碰到过一次成功，所以学不到"到达"。

### 给 PPO 经典控制器的航点 / 示范（第 6、7、10、11 轮）

| 轮 | 设置 | 要塞（模仿后 → 最终） | 环形（模仿后 → 最终） |
|---|---|---|---|
| 6 | 示范热启动（`--demo-warmstart`：经典控制器开 6400 步，PPO 先模仿再训练） | MLP 9 → 0；GRU 111 → 发散 | MLP 125 → 0；GRU 128 → 发散 |
| 7 | + 第 0 步冻结观测归一化 + 递减的模仿约束（`--demo-anchor`） | MLP 9 → 0；GRU 111 → 4 | MLP 两种子 119、126；GRU 两种子 126、126 |
| 10 | 只给航点（`--route-hint`：观测第 0/1 列改成指向经典控制器当前航点），不给示范 | 0（两种奖励都是 0） | 0 |
| 11 | 航点 + 示范热启动 + 模仿约束（MLP） | **102 → 110**（种子 90001） | **128 → 128**（种子 90001） |

第 11 轮要塞逐局对照经典控制器：两边都成功 105 局、只有经典成功 6 局、只有 PPO 成功 5 局、都失败 12 局；用时、路程、离障碍距离几乎一样。
种子 90002 正在跑，结果出来后补在这里。

### 读法

- 网络本身能用这些信息：给了航点和示范，PPO 在要塞追平经典控制器（110 对 111）。
- 纯 PPO 的问题是探索：只给航点（第 10 轮）仍然一次都到不了终点。
- 给航点等于把经典控制器的"记忆 + 找路"结果交给 PPO，第 11 轮里 PPO 实际只替代了第 3 层（跟航点）。这一行必须和纯 PPO 分开写，也不能说成 PPO 自己学会的。
- 比较时要注意：经典控制器把"障碍不动、记下来再找路"的知识写进了代码，并且针对这些题改过几版；PPO 用的是标准设置。

## 新增 / 修改的文件

| 文件 | 内容 |
|---|---|
| `scripts/train_journal_ppo.py` | 上面所有开关；`--resume` 续训；`--early-stop-no-success`（训练 1/3 后，最近 1/3 一次成功都没有就停） |
| `scripts/eval_fair_fix_calm.py` / `eval_classical_calm.py` | 静水评测（PPO / 经典），记录每局结局、路程、动作均值探针 |
| `scripts/ppo_squashed_models.py`、`ppo_memory_models.py` | tanh 有界高斯（精确对数概率）、GRU |
| `scripts/ppo_training_fixes.py`、`ppo_final_g46.py` | 超时罚 |
| `scripts/ppo_demo_warmstart.py` | 示范采集、模仿、模仿约束 |
| `scripts/ppo_route_hint.py` | 航点提示（训练和评测装同一个钩子） |
| `scripts/ppo_novelty_bonus.py`、`ppo_early_stop.py` | 新格子奖励、提前停止 |
| `tasks/hazard_nav/*` | 绕行进度/时间成本、课程（默认关） |
| `scripts/test_ppo_*.py` | 离线自检（不需要 Isaac） |

## 怎么跑（例：第 11 轮要塞）

```
python scripts/train_journal_ppo.py --task journal:fortress --vessel blueboat --training-seed 90001 \
    --purpose training_smoke --arm mixture --fair-fix geo_time --action-dist tanh \
    --demo-warmstart --demo-anchor --route-hint --early-stop-no-success \
    --num-envs 64 --device cpu --out runs/tr_fortress_r11 --headless
python scripts/eval_fair_fix_calm.py --task journal:fortress --vessel blueboat \
    --checkpoint runs/tr_fortress_r11/final.pt --action-dist tanh --route-hint \
    --eval-seed 3001 --num-envs 64 --episodes-per-env 2 --device cpu --out ev_fortress_r11.json --headless
```

模仿后的存档在 `<out>/bc_init/agent_0.pt`。本机 CPU 上一条训练约 6–9 小时。
每一轮的预先判据和逐条记录（中文）在本地 `usvbench_ppo_fix_runs/PROTOCOL*.md`，没有放进仓库，需要可以单独给。
