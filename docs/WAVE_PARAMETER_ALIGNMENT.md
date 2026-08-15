# 波浪实现及参数统一说明

后续两边统一采用最新 push 的共享波浪实现：
`tasks/_shared/waves.py`。早期各任务目录中保留的独立 JONSWAP 实现不再作为
后续实验、对比或结果汇报的依据。

## JONSWAP 参数定义

### 显著（有义）波高 `Hs`

不规则波的波高统一记为 `Hs`，单位为 m，定义为：

```text
Hs = 4 sqrt(m0)
m0 = integral S(f) df
```

其中，`S(f)` 为波浪频谱，`m0` 为频谱的零阶矩。

共享实现会根据实际使用的离散频带计算 `m0`，并重新归一化整个频谱，使
生成波场严格满足 `4 sqrt(m0) = Hs`。因此，配置中的 `hs_min_m` 和
`hs_max_m` 表示波场最终实际实现的显著波高，不是未经校正的名义尺度参数。

中文说明统一使用“显著（有义）波高”，避免使用含义不明确的“有效波高”。

### 谱峰周期 `Tp`

谱峰周期统一记为 `Tp`，单位为 s，定义为：

```text
Tp = 1 / fp
```

其中，`fp` 为 JONSWAP 频谱的目标峰值频率，单位为 Hz。`Tp` 不是平均周期、
零交叉周期或能量周期。

由于频谱采用有限个离散频点，实际最大谱值所在频点可能与目标 `fp` 有很小
偏差。如需记录该离散频点，应另写为 `f_peak_bin` 或
`T_peak_bin = 1 / f_peak_bin`，不要与配置参数 `Tp` 混用。

### 当前默认值与抽样律

HazardNav 当前共享实现的默认值如下；这些是代码默认值，不等同于双方
尚未签字的训练/认证档位：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `gamma` | `3.3` | 认证协议的固定峰增强因子；探索性实验仍可显式改用范围 |
| `f_min_hz` | `0.10` | 频带下限 |
| `f_max_hz` | `1.60` | 频带上限 |
| `spread_deg` | `30` | 总展布宽度，实际为平均波向两侧各 `15°` |
| `n_components` | `30` | 频率分量数 |
| `buoyancy_stations` | `6` | 船体浮力采样点数 |

认证默认固定 `gamma=3.3`。`Hs` 和 `Tp` 的训练/评测抽样仍应使用
`sampling_mode="levels"`，并显式填写 `hs_levels_m`、`tp_levels_s`；
认证配置可将 `gamma_levels` 写成 `(3.3,)`。探索性实验若要在范围内均匀抽样，
必须显式记录该选择。训练包、插值包、外推包的具体 `Hs×Tp` 档位仍需双方确认，
当前仓库不把任何一组未经确认的档位冻结成 baseline。

`direction_deg=None` 时，平均传播方向在 `[0, 360°)` 上确定性伪随机采样；
指定数值时所有环境固定为该方向。

### 随机相位/方向种子协议（认证必需）

相位、平均传播方向、每个分量的方向展布以及海况参数均由下面的纯函数决定：

```text
random_value = f(eval_seed, environment_index, episode_index, stream_index)
```

认证评测的参考种子为 `eval_seed=42`，与 `scripts/eval_hazard_nav.py` 的
`--seed` 默认值一致；正式报告应把实际使用的 seed 写入结果记录，不能依赖
未记录的进程默认状态。`environment_index` 是从 `0` 开始的环境编号，
`episode_index` 是每个环境独立维护的回合编号，首次 reset 固定为 `0`，
此后依次为 `1, 2, ...`；它不是需要另行抽样的海况参数。

实现使用每个环境独立的 CPU `torch.Generator`，由上述整数元组派生 seed，
不读取或推进全局 Torch RNG。HazardNav 在每个环境维护 `episode_index`：
首次 reset 为 `0`，之后每次 reset 加 `1`；reset 时把该索引显式传给共享波场。
因此，同一评测 seed、环境编号和回合编号会得到完全相同的 `Hs/Tp/gamma`、
相位和方向；每回合的波面时间也从该回合 reset 时的局部 `t=0` 开始，不受
前一回合实际结束时刻影响。即使 reset 的环境子集顺序不同，也不会改变结果。障碍布局使用
同一 `(eval_seed, environment_index, episode_index)` 协议，保证配对策略看到同一
回合流。

### 零剂量恒等

当 `mode="calm"`，或 Airy 的 `height_m=0`、JONSWAP 的 `hs_max_m=0` 时，
工厂直接返回 `CalmWater`。这条路径不采样浮力站、不施加轨道流或波面速度，
因此与静水路径逐位一致；它不是“计算了零波幅后的近似相等”。

## 波场离散方式

当前 JONSWAP 波场默认由 **30 个频率分量**叠加生成，即
`n_components = 30`。每个分量由 JONSWAP 频谱确定振幅，并具有独立的随机
相位和传播方向。

波面表达为：

```text
eta(x, y, t) = sum_n a_n cos(k_n (dx_n x + dy_n y) - omega_n t + phi_n)
a_n = sqrt(2 S(f_n) Delta_f_n)
```

波数采用深水色散关系 `k_n = omega_n^2 / g`。频点不是严格等间距：在名义
`df=(f_max-f_min)/30` 的中点网格上加入固定的小抖动，并用相邻频点中点定义
每个 `Delta_f_n`。因此不存在 `1/df` 的精确短周期重复；旧均匀网格的诊断周期
在当前默认频带中是 `1/0.05 = 20 s`，已经短于 120 s 回合，不能作为现行实现
的重复周期。非均匀频点同时在谱归一化中使用各自的 `Delta_f_n`，所以
`4 sqrt(m0) = Hs` 不变。

### 展布形状

`spread_deg` 表示总宽度，而不是单侧角度。当前实现对每个分量独立地从
`[-spread_deg/2, +spread_deg/2]` 均匀抽样，再加到平均波向上；它不是
cosine-power（`cos^s`）方向分布。因而 `spread_deg=30°` 明确表示 `±15°`。

### 有效域与深水假设

HazardNav 默认要求所有可能的组合满足：

```text
lambda_p = g Tp^2 / (2 pi)
Hs / lambda_p <= 0.05
```

这是线性深水模型的陡度上限；超出时配置会在建场阶段报错。色散关系仍是
深水关系，未加入有限水深修正；若水深 `h` 不再明显大于 `lambda_p/2`，应先
另行确认适用性，不能把当前结果称为浅水认证结果。

这里的 30 个点是频谱的**频率分量数**，不是船体上的浮力采样点数。默认频率
抖动幅度为名义 `df` 的 `0.22` 倍；它是固定的频点设计参数，不是每回合重新
抽样的随机量。

## 船体波浪耦合

当前船体采用 **6 个分布式浮力采样点**，即
`buoyancy_stations = 6`，不是 8 个。

每个采样点根据其正下方的局部波面高度计算浸没比例和浮力。各点浮力合成后，
自然产生船体的升沉、横摇和纵摇响应。水平波浪作用则通过船体与水质点轨道
速度之间的相对速度进入阻力计算，垂向阻尼同样使用船体垂向速度与波面垂向
速度之差。

这套实现不额外设置人为的波浪力或波浪力矩增益；波浪响应来自已有的浮力、
静水恢复力和阻尼参数。

需要区分以下两个数量：

| 数量 | 当前默认值 | 作用 |
|---|---:|---|
| JONSWAP 频率分量数 | 30 | 离散并合成不规则波场 |
| 船体浮力采样点数 | 6 | 计算局部浸没、浮力及横摇/纵摇力矩 |

## 规则波说明

Airy 规则波的波峰到波谷高度记为 `H`，周期记为 `T`。规则波的 `H` 与
JONSWAP 不规则海况的 `Hs` 定义不同，结果表中不能把二者使用同一个字段。

若需要构造与 JONSWAP 海况方差相同的规则波对照，应使用：

```text
H = Hs / sqrt(2)
T = Tp
```

若仅令 `H = Hs`，应明确标注为“波高数值相同”，不能称为能量等效。

## 实验记录要求

后续每个 JONSWAP 实验至少记录：

```text
model=JONSWAP
eval_seed=<integer>
environment_index=<integer>
episode_index=<integer>
Hs=<value or range> m
Tp=<value or range> s
gamma=<value or range>
sampling_mode=<uniform or levels>
Hs_levels=<explicit list when levels>
Tp_levels=<explicit list when levels>
f_band=[f_min_hz, f_max_hz] Hz
max_steepness=0.05
n_components=30
spread=<value> deg
direction=<value or random>
buoyancy_stations=6
```

按照以上字段记录后，两边使用相同配置时，对应的就是同一套波浪定义、波场
离散方式和船体耦合口径。
