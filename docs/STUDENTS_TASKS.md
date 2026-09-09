# USVBench — 学生任务分配（静水阶段）

> **学生**：史进（机械大三）、张年喜
> **导师**：张老师
> **目标**：扩展 USVBench calm-water benchmark，每人各自贡献 1–2 个新任务

---

## 史进的任务清单

1. **复现 ROV calm baseline**（Week 1–2）
   - 跑通 `Isaac-My-First-Task-Calm-Direct-v1`
   - 复现基线 **4.48** targets/episode（确定性评估，基线统一维护在 [`TASKS.md`](../TASKS.md#current-baselines)）
   - 确认整个流程（Isaac Lab + skrl + wandb）可用

2. **Gazebo sim-to-sim 验证**（Week 3–4）
   - 导出 Isaac Sim 训好的 ROV 策略（权重）
   - 在 Gazebo/VRX 里复现相同任务
   - 对比 Isaac vs Gazebo 的轨迹、性能
   - 交付：轨迹对比图 + transfer gap 指标

3. **丰富 benchmark — 新船体任务**（Week 5+）
   - 选一个新船体（catamaran / cruise-ship / 自己想法）
   - 验证 USD、physics 稳定
   - 设计新任务（point-nav / 航线 / 靠岸等）
   - 训练 + 3-seed baseline
   - 交付：STARTER_TASK.md + 合并 PR

---

## 张年喜的任务清单

1. **复现 boat calm V26 baseline**（Week 1–2）
   - 跑通 `Isaac-USVBench-Boat-Calm-Direct-v1`
   - 理解为什么 boat 的 reward 要特殊调（speed-coupling、reach bonus）
   - 复现基线 **3.75** targets/episode（seed 42，确定性评估，见 [`TASKS.md`](../TASKS.md#current-baselines)）

2. **补充 boat 3-seed 基线**（Week 3）
   - 跑 seed 123 / 456 两次
   - 计算 mean ± std（3-seed）
   - 更新 README/TASKS.md 的 boat 基线

3. **静水 obstacle avoidance**（Week 4+）
   - 在 calm 任务里加静态障碍（rocks / buoys）
   - 设计碰撞检测 + reward penalty
   - 训练 baseline + 3-seed 验证
   - 交付：env + STARTER + baseline

---

## 通用要求

- 每个新任务的交付物（参考 boat_calm_nav/）：
  - `tasks/<name>/my_first_task_env.py` + `my_first_task_env_cfg.py`
  - `tasks/<name>/STARTER_TASK.md`（含运行步骤、成功标准、env-vars）
  - `tasks/<name>/checkpoints/<name>_s42.pt`（reference policy）
  - wandb runs（3 seeds）+ 分数
  
- 代码质量：
  - 遵循 [`docs/GITHUB_COLLABORATION.md`](GITHUB_COLLABORATION.md) 的 git 流程
  - 单变量实验（符合项目 hard rules）
  - wandb 记到 `uclusv/usvbench`，检查 `$env:WANDB_ENTITY`
  
- 文档：
  - 解释为什么这个任务的 reward 要这样调
  - 列出 physics 参数（mass、damping、thrust）和调校原因
  - 如果有低于预期的地方，写进 "Known limitations"

---

## 进度跟踪

| 学生 | 任务 1 | 任务 2 | 任务 3 |
|---|---|---|---|
| **史进** | 复现 ROV | Gazebo 验证 | 新船体（自选） |
| **张年喜** | 复现 boat + 补 3-seed | obstacle avoidance | — |

每周五一句话状态更新；卡住了**当天**找 Yutong。
