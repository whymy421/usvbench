# USV-RL Benchmark Design Doc

> 状态:**Draft v0**,viva 后第一稿。
> 作者:Raina(UCL Mech Eng)。
> 目的:把现有 E6/E7 研究代码重构为可复用、可贡献、可发表的开源 USV RL benchmark。

---

## 1. Motivation —— 这个 benchmark 解决什么问题

来自 E6 项目 80+ 次失败 run 的复盘(详见 `EXPERIMENT_LOG.md`):

| 当前研究代码的痛点 | benchmark 必须解决 |
|--------------------|--------------------|
| Vehicle 换型号要改 5+ 文件、重调全套常数 | Vehicle 是 plugin,USD + 物理参数包统一注入 |
| Reward 一锅 8 项,出错时不知道哪项在害 | Reward 是 composable 组件,可单独开关、有 unit test |
| Obs 维度从 11→16 改 6 个文件 | Obs 声明式 yaml 选,维度自动计算 |
| Wave 模型嵌死在 env 里,不能换 sea state | Sea state 是 plugin |
| 没有公认指标,每篇 paper 自己定义 lateral_exposure | 指标库 + 标准评估协议 |

**结论**:USV RL 缺一个像 `safety-gymnasium`、`d4rl` 那样的标准 benchmark。本项目占位。

---

## 2. 目标用户

1. 研究者:想在自己的 USV/船舶算法上跑一个标准 task,报可比数字
2. 算法贡献者:想测自己的 PPO/SAC/MAPPO 变种在真实物理任务上是否 work
3. 工业:想测 sim-to-real 前的 controller 性能

**非目标**:替代 ROS、替代真实船舶仿真(Maritime Robotics 这类专业 sim)

---

## 3. 架构

```
usv_bench/
├── usv_bench/
│   ├── vehicles/           # 船 plugin(USD + dynamics 参数 + 力映射)
│   │   ├── base.py
│   │   ├── rov_proxy.py
│   │   ├── trihull_5m.py
│   │   └── catamaran_3m.py
│   ├── seas/               # 海况 plugin
│   │   ├── base.py
│   │   ├── calm.py
│   │   ├── airy.py
│   │   └── jonswap.py
│   ├── physics/            # 共享物理函数
│   │   ├── buoyancy.py
│   │   ├── damping.py
│   │   └── wave_forces.py
│   ├── observations/       # 观测组件
│   │   ├── nav_2d.py
│   │   ├── wave_current.py
│   │   ├── wave_future_ncr.py   # 论文 contribution
│   │   └── teammates.py
│   ├── rewards/            # reward 组件
│   │   ├── heading.py
│   │   ├── velocity.py
│   │   ├── reach.py
│   │   ├── wave_safety.py       # 强制 speed-coupled
│   │   ├── roll.py
│   │   ├── collision.py
│   │   └── formation.py
│   ├── tasks/              # 完整任务(组装上面所有组件)
│   │   ├── point_nav.py
│   │   ├── sar_multi.py
│   │   └── path_follow.py
│   ├── metrics/
│   │   ├── lateral_exposure.py
│   │   ├── rescue_rate.py
│   │   ├── roll_rms.py
│   │   └── energy.py
│   ├── backends/           # 仿真后端
│   │   ├── isaac.py
│   │   └── lightweight.py       # 3DOF Python sim(无 GPU 也能用)
│   └── envs/               # gym/gymnasium wrapper
├── configs/                # 锁版本的标准 task 配置
│   ├── v1_nav_calm.yaml
│   ├── v1_nav_jonswap_lowsea.yaml
│   ├── v1_nav_jonswap_highsea.yaml
│   ├── v1_sar_3usv_jonswap.yaml
│   └── v1_path_follow_curr.yaml
├── baselines/              # 参考算法实现 + checkpoint
├── examples/
├── tests/
└── docs/
```

---

## 4. 核心接口

### Vehicle

```python
class USVBase:
    """所有 USV 模型必须实现的接口。"""
    # 静态属性(从 yaml 注入)
    usd_path: str
    mass: float          # kg
    displacement: float  # m³ 满载排水
    length: float        # m
    beam: float          # m
    height: float        # m
    # 动力学
    def apply_thrust(self, action) -> tuple[forces, torques]: ...
    def compute_buoyancy(self, z, ori) -> tuple[forces, torques]: ...
    def compute_damping(self, vel, submerged) -> tuple[forces, torques]: ...
    # 船舶特有(可选)
    def metacentric_height(self) -> float: ...   # GM
    def waterplane_area(self) -> float: ...
```

### Reward / Observation 组件

```python
class RewardTerm:
    name: str
    weight: float
    def __call__(self, state) -> torch.Tensor: ...
    def documented_range(self) -> tuple[float, float]: ...
    def documented_inputs(self) -> list[str]: ...

class ObservationTerm:
    name: str
    @property
    def dim(self) -> int: ...                  # 维度自动算
    def __call__(self, state) -> torch.Tensor: ...
```

### Sea state

```python
class SeaState:
    enable: bool
    def step(self, t): ...                     # 时间推进
    def elevation(self, t, x, y): ...
    def forces_on(self, vehicle, state): ...
    def randomize(self, env_ids): ...
```

---

## 5. Task = YAML

一个 task 完全由 YAML 定义,改任务不改代码:

```yaml
# configs/v1_sar_3usv_jonswap.yaml
name: SAR-3USV-JONSWAP-v1
backend: isaac
n_agents: 3
vehicle: trihull_5m
sea: jonswap
sea_params: { hs_range: [0.3, 1.0], tp_range: [4, 7], n_components: 30 }
observations:
  - nav_2d
  - wave_current
  - wave_future_ncr: { horizons: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] }
  - teammates: { n_teammates: 2 }
rewards:
  - heading:      { weight: 1.0 }
  - velocity:     { weight: 0.3 }
  - reach:        { weight: 10.0, radius: 5.0 }
  - wave_safety:  { weight: 1.0, speed_coupled: true }
  - roll:         { weight: 0.75 }
  - collision:    { weight: 2.0, threshold: 8.0 }
metrics: [lateral_exposure, rescue_rate, roll_rms, energy]
evaluation:
  seeds: [42, 123, 456, 789, 2024]
  episodes_per_seed: 100
  episode_length_s: 120
```

---

## 6. 评估协议(让别人能复现的关键)

| 规则 | 内容 |
|------|------|
| 版本锁 | 任务一旦发布(`v1`)永不改,改了就是 `v2`,保证旧 paper 复现 |
| 标准 seeds | 固定 [42, 123, 456, 789, 2024] |
| 报数格式 | mean ± std,最少 5 seed |
| Reference baseline | 每个 task 必须提供 1 个公开 baseline 的 checkpoint + 训练曲线 |
| 提交方式 | PR + 复现命令 + wandb 公开链接 |

---

## 7. 阶段计划

| 阶段 | 时间 | 产出 |
|------|------|------|
| Phase 1 | 1 周 | 抽 `jonswap`、`wave_future_ncr`、4 个指标成独立模块,自用 |
| Phase 2 | 2 周 | 定 Vehicle/Reward/Obs 接口,把 E6 重构成 yaml 任务 |
| Phase 3 | 4 周 | 加 lightweight backend、PPO/SAC/MAPPO baseline、文档 |
| Phase 4 | 2 周 | CI、tests、release(PyPI + GitHub) |

---

## 8. 现有代码可复用清单

| 当前文件 | 复用程度 | 目标位置 |
|----------|---------|---------|
| `jonswap_wave.py` | 几乎全用 | `seas/jonswap.py` |
| `_compute_buoyancy_forces` | 函数化 | `physics/buoyancy.py` |
| `compute_future_profile` (6D NCR) | **核心 contribution** | `observations/wave_future_ncr.py` |
| E6 reward 8 项 | 拆 → 8 个独立文件 | `rewards/*.py` |
| MAPPO 多智能体 done 逻辑 | 函数化 | `tasks/sar_multi.py` |
| K-series 训练脚本 | 重写 | `baselines/mappo.py` |
| EXPERIMENT_LOG.md | motivation 来源 | `docs/lessons_learned.md` |

**不复用**:Isaac Lab 紧耦合代码(替换为 backend 抽象层)、硬编码 force 系数、env var 配置注入(改 yaml)。

---

## 9. 开源前 checklist

- [ ] README 5 分钟看懂,1 行命令跑通
- [ ] 至少 3 个 task,每个有 baseline 数字
- [ ] 至少 2 个 vehicle,2 个 sea state
- [ ] CI 跑 unit tests + 5-iter smoke training
- [ ] CITATION.cff + arXiv 链接
- [ ] LICENSE(推荐 MIT 或 Apache-2.0)
- [ ] CONTRIBUTING.md(怎么加新 vehicle/task)
- [ ] 版本锁:`v1` 配置文件永不改

---

## 10. 待决策(留给设计会议)

见 `BENCHMARK_MEETING_AGENDA.md`。

---

## 11. 命名候选

`WaveBench` / `USVBench` / `MarinaRL` / `SeaNav` / `UCL-USV-Bench`(待定)
