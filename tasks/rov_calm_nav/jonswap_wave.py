"""
JONSWAP Irregular Wave Module for Isaac Lab
============================================
替换原有单频正弦波，实现基于JONSWAP谱的不规则波叠加。

用法:
    在 MultiUSVEnv 中替换 _init_wave_field() 和 _compute_wave_forces()

    from .jonswap_wave import JONSWAPWaveField

    # 在 _setup_scene() 或 __init__ 中:
    self.wave_field = JONSWAPWaveField(cfg=cfg.wave_cfg, num_envs=self.num_envs, device=self.device)

    # 在 _compute_wave_forces() 中:
    eta, vel_x, vel_y = self.wave_field.compute(t, positions_x, positions_y)

作者: Raina (Multi-USV MARL Project)
"""

import torch
import math


class JONSWAPWaveField:
    """基于JONSWAP谱的不规则波浪场。

    核心原理:
        η(x,y,t) = Σ aₙ · cos(kₙ·(dx·x + dy·y) - ωₙ·t + φₙ)

        其中:
        - aₙ = √(2·S(fₙ)·Δf) 为第n个谐波的振幅，由JONSWAP谱决定
        - ωₙ = 2π·fₙ 为角频率
        - kₙ = ωₙ²/g 为波数（深水近似）
        - φₙ 为随机相位 [0, 2π)
        - (dx, dy) 为波浪传播方向单位向量
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        # JONSWAP谱参数
        hs_range: tuple[float, float] = (0.3, 1.0),     # 有义波高范围 (m)
        tp_range: tuple[float, float] = (4.0, 7.0),     # 谱峰周期范围 (s)
        gamma_range: tuple[float, float] = (1.0, 5.0),   # 峰增强因子范围
        # 离散化参数
        n_components: int = 30,                            # 谐波分量数
        f_min: float = 0.04,                               # 最小频率 (Hz)
        f_max: float = 0.5,                                # 最大频率 (Hz)
        # 物理常数
        gravity: float = 9.81,
    ):
        self.num_envs = num_envs
        self.device = device
        self.n_components = n_components
        self.gravity = gravity
        self.f_min = f_min
        self.f_max = f_max
        self.hs_range = hs_range
        self.tp_range = tp_range
        self.gamma_range = gamma_range

        # 频率离散化 (所有env共享频率网格)
        self.df = (f_max - f_min) / n_components
        # (n_components,)
        self.freqs = torch.linspace(
            f_min + self.df / 2,
            f_max - self.df / 2,
            n_components,
            device=device
        )
        self.omegas = 2 * math.pi * self.freqs  # (n_components,)
        self.wave_numbers = self.omegas ** 2 / gravity  # 深水色散关系 (n_components,)

        # 每个env的参数 (在reset时随机化)
        self.hs = torch.zeros(num_envs, device=device)
        self.tp = torch.zeros(num_envs, device=device)
        self.gamma = torch.zeros(num_envs, device=device)
        self.wave_dir = torch.zeros(num_envs, 2, device=device)  # (dx, dy)

        # 每个env、每个频率分量的振幅和相位
        # (num_envs, n_components)
        self.amplitudes = torch.zeros(num_envs, n_components, device=device)
        self.phases = torch.zeros(num_envs, n_components, device=device)
        self.comp_dir_x = torch.zeros(num_envs, n_components, device=device)
        self.comp_dir_y = torch.zeros(num_envs, n_components, device=device)
        # 初始化所有env
        all_ids = torch.arange(num_envs, device=device)
        self.randomize(all_ids)

    def _compute_jonswap_spectrum(self, hs: torch.Tensor, tp: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """计算JONSWAP谱。

        S(f) = α · f^(-5) · exp(-5/4 · (fp/f)^4) · γ^r

        其中:
            α = 5/16 · Hs² · fp⁴  (Phillips常数，归一化到Hs)
            r = exp(-0.5 · ((f - fp) / (σ · fp))²)
            σ = 0.07 (f ≤ fp), 0.09 (f > fp)

        Args:
            hs: (num_envs,) 有义波高
            tp: (num_envs,) 谱峰周期
            gamma: (num_envs,) 峰增强因子

        Returns:
            S: (num_envs, n_components) 谱密度值 m²/Hz
        """
        fp = 1.0 / tp  # (num_envs,)

        # 扩展维度以便广播: (num_envs, 1) 和 (1, n_components)
        fp_e = fp.unsqueeze(1)        # (num_envs, 1)
        hs_e = hs.unsqueeze(1)        # (num_envs, 1)
        gamma_e = gamma.unsqueeze(1)  # (num_envs, 1)
        f_e = self.freqs.unsqueeze(0) # (1, n_components)

        # Phillips常数 (归一化)
        alpha = 5.0 / 16.0 * hs_e ** 2 * fp_e ** 4

        # Pierson-Moskowitz部分
        pm = alpha * f_e.pow(-5) * torch.exp(-1.25 * (fp_e / f_e).pow(4))

        # sigma: 0.07 for f <= fp, 0.09 for f > fp
        sigma = torch.where(f_e <= fp_e, 0.07, 0.09)

        # 峰增强因子
        r = torch.exp(-0.5 * ((f_e - fp_e) / (sigma * fp_e)).pow(2))

        # JONSWAP = PM × γ^r
        S = pm * gamma_e.pow(r)

        # 数值安全
        S = torch.clamp(S, min=0.0)

        return S

    def randomize(self, env_ids):
        num = len(env_ids)

        self.hs[env_ids] = torch.rand(num, device=self.device) * (self.hs_range[1] - self.hs_range[0]) + self.hs_range[0]
        self.tp[env_ids] = torch.rand(num, device=self.device) * (self.tp_range[1] - self.tp_range[0]) + self.tp_range[0]
        self.gamma[env_ids] = torch.rand(num, device=self.device) * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]

        angles = torch.rand(num, device=self.device) * 2 * math.pi
        self.wave_dir[env_ids, 0] = torch.cos(angles)
        self.wave_dir[env_ids, 1] = torch.sin(angles)

        self.phases[env_ids] = torch.rand(num, self.n_components, device=self.device) * 2 * math.pi

        # 方向展布：每个分量在主方向±30°内随机偏移
        spread_angles = (torch.rand(num, self.n_components, device=self.device) - 0.5) * (math.pi / 3)
        base_angle = torch.atan2(self.wave_dir[env_ids, 1], self.wave_dir[env_ids, 0])
        comp_angles = base_angle.unsqueeze(1) + spread_angles
        self.comp_dir_x[env_ids] = torch.cos(comp_angles)
        self.comp_dir_y[env_ids] = torch.sin(comp_angles)

        S = self._compute_jonswap_spectrum(self.hs[env_ids], self.tp[env_ids], self.gamma[env_ids])
        self.amplitudes[env_ids] = torch.sqrt(2.0 * S * self.df)

    def compute_elevation(self, t, pos_x, pos_y):
        spatial_phase = (self.comp_dir_x * pos_x.unsqueeze(1) + self.comp_dir_y * pos_y.unsqueeze(1)) * self.wave_numbers.unsqueeze(0)
        temporal_phase = self.omegas.unsqueeze(0) * t
        total_phase = spatial_phase - temporal_phase + self.phases
        eta = (self.amplitudes * torch.cos(total_phase)).sum(dim=1)
        return eta

    def compute_forces(self, t, pos_x, pos_y, forward_2d):
        eta = self.compute_elevation(t, pos_x, pos_y)

        spatial_phase = (self.comp_dir_x * pos_x.unsqueeze(1) + self.comp_dir_y * pos_y.unsqueeze(1)) * self.wave_numbers.unsqueeze(0)
        temporal_phase = self.omegas.unsqueeze(0) * t
        total_phase = spatial_phase - temporal_phase + self.phases
        deta = -(self.amplitudes * self.wave_numbers.unsqueeze(0) * torch.sin(total_phase)).sum(dim=1)

        sideways = torch.stack([-forward_2d[:, 1], forward_2d[:, 0]], dim=-1)
        lateral_exposure = torch.abs(self.wave_dir[:, 0] * sideways[:, 0] + self.wave_dir[:, 1] * sideways[:, 1])
        frontal_exposure = self.wave_dir[:, 0] * forward_2d[:, 0] + self.wave_dir[:, 1] * forward_2d[:, 1]

        heave_force = eta * 100.0
        roll_torque = lateral_exposure * deta * 200.0
        wave_drag = torch.clamp(-frontal_exposure, min=0) * torch.abs(eta) * 15.0

        return {
            "eta": eta,
            "heave_force": heave_force,
            "roll_torque": roll_torque,
            "lateral_exposure": lateral_exposure,
            "wave_drag": wave_drag,
        }

    def get_obs(self, forward_2d: torch.Tensor) -> dict[str, torch.Tensor]:
        """获取波浪相关的观测量，用于RL策略输入。

        Args:
            forward_2d: (num_envs, 2) USV前向方向

        Returns:
            dict with:
            - wave_dot: cos(船头与波浪夹角)
            - wave_cross: sin(船头与波浪夹角)
            - wave_height_norm: 归一化波高
            - hs: 当前有义波高 (给centralized critic用)
            - tp: 当前谱峰周期
        """
        wave_dot = (forward_2d[:, 0] * self.wave_dir[:, 0]
                    + forward_2d[:, 1] * self.wave_dir[:, 1])
        wave_cross = (forward_2d[:, 0] * self.wave_dir[:, 1]
                      - forward_2d[:, 1] * self.wave_dir[:, 0])
        wave_height_norm = self.hs / self.hs_range[1]  # 归一化到[0,1]

        return {
            "wave_dot": wave_dot.unsqueeze(-1),
            "wave_cross": wave_cross.unsqueeze(-1),
            "wave_height_norm": wave_height_norm.unsqueeze(-1),
            "hs": self.hs,
            "tp": self.tp,
        }


# ============================================
# 配置类（和Isaac Lab的configclass风格一致）
# ============================================

class JONSWAPWaveCfg:
    """JONSWAP波浪配置，替换原有的WavePhysicsCfg"""
    enable_wave: bool = True
    # JONSWAP谱参数范围 (训练时随机化)
    hs_min: float = 0.3        # 最小有义波高 (m)
    hs_max: float = 1.0        # 最大有义波高 (m)
    tp_min: float = 4.0        # 最小谱峰周期 (s)
    tp_max: float = 7.0        # 最大谱峰周期 (s)
    gamma_min: float = 1.0     # 最小峰增强因子
    gamma_max: float = 5.0     # 最大峰增强因子
    # 离散化
    n_components: int = 30     # 谐波分量数
    f_min: float = 0.04        # 最小频率 (Hz)
    f_max: float = 0.5         # 最大频率 (Hz)


# ============================================
# 工具函数: 用于验证和调试
# ============================================

def validate_spectrum(hs: float = 1.0, tp: float = 5.0, gamma: float = 3.3, n_components: int = 30):
    """验证JONSWAP谱实现的正确性。

    检查:
    1. 4√m₀ ≈ Hs (谱的零阶矩应满足)
    2. 谱峰位置在 fp = 1/Tp 附近
    3. 波面时间序列的统计特性

    用法:
        python -c "from jonswap_wave import validate_spectrum; validate_spectrum()"
    """
    device = torch.device("cpu")

    wave = JONSWAPWaveField(
        num_envs=1,
        device=device,
        hs_range=(hs, hs),
        tp_range=(tp, tp),
        gamma_range=(gamma, gamma),
        n_components=n_components,
    )

    # 检查谱
    S = wave._compute_jonswap_spectrum(
        torch.tensor([hs]),
        torch.tensor([tp]),
        torch.tensor([gamma])
    )

    m0 = (S * wave.df).sum().item()
    hs_check = 4 * math.sqrt(m0)

    print(f"JONSWAP Spectrum Validation")
    print(f"{'='*40}")
    print(f"Input:  Hs={hs}m, Tp={tp}s, gamma={gamma}")
    print(f"N components: {n_components}")
    print(f"Freq range: {wave.f_min}-{wave.f_max} Hz")
    print(f"{'='*40}")
    print(f"m0 = {m0:.4f} m²")
    print(f"4√m0 = {hs_check:.3f} m (should ≈ {hs} m)")
    print(f"Error: {abs(hs_check - hs)/hs*100:.1f}%")

    # 谱峰
    peak_idx = S[0].argmax().item()
    peak_freq = wave.freqs[peak_idx].item()
    print(f"Peak freq: {peak_freq:.3f} Hz (should ≈ {1/tp:.3f} Hz)")

    # 时间序列
    dt = 0.1
    t_max = 600.0
    etas = []
    for t in range(int(t_max / dt)):
        eta = wave.compute_elevation(
            t * dt,
            torch.zeros(1),
            torch.zeros(1)
        )
        etas.append(eta.item())

    etas_t = torch.tensor(etas)
    hs_ts = 4 * etas_t.std().item()
    print(f"\nTime series (600s):")
    print(f"  Hs from time series: {hs_ts:.3f} m (should ≈ {hs} m)")
    print(f"  Max elevation: {etas_t.max().item():.3f} m")
    print(f"  Min elevation: {etas_t.min().item():.3f} m")
    print(f"  Mean (should ≈ 0): {etas_t.mean().item():.4f} m")

    print(f"\n✅ Validation complete!")
    return m0, hs_check


if __name__ == "__main__":
    validate_spectrum()