"""Physics acceptance tests for the irregular wave field (pure torch)."""

from __future__ import annotations

import math

import torch

try:
    from .sea_state import GRAVITY, SeaState, SeaStateCfg
except ImportError:  # direct execution
    from sea_state import GRAVITY, SeaState, SeaStateCfg


def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)
    hs, tp = 0.30, 3.0
    cfg = SeaStateCfg(
        enable=True, significant_height_m=hs, peak_period_s=tp, n_components=8
    )
    n_envs = 64
    sea = SeaState(cfg, n_envs, device)

    # 1. Dispersion: every component satisfies omega^2 = g k (deep water).
    assert torch.allclose(sea.omega**2, GRAVITY * sea.k, rtol=1e-6)

    # 2. Significant height: H_s = 4 sqrt(m0) recovered from the realisation.
    xy = torch.zeros((n_envs, 2), device=device)
    samples = torch.stack(
        [sea.elevation(xy, t) for t in torch.linspace(0, 120, 4000).tolist()]
    )
    m0 = samples.var(dim=0).mean()
    hs_measured = 4.0 * math.sqrt(float(m0))
    assert abs(hs_measured - hs) / hs < 0.15, (hs_measured, hs)

    # 3. Zero mean elevation over many periods (no spurious set-up).
    assert abs(float(samples.mean())) < 0.02 * hs

    # 4. Determinism: same phases -> identical field; resample -> different.
    a = sea.elevation(xy, 7.0).clone()
    b = sea.elevation(xy, 7.0)
    assert torch.equal(a, b)
    sea.resample(torch.arange(n_envs))
    assert not torch.allclose(a, sea.elevation(xy, 7.0))

    # 5. Forces are finite, bounded, and vanish for a calm sea.
    vel = torch.zeros((n_envs, 2), device=device)
    force, torque = sea.forces(xy, vel, 3.5)
    assert torch.isfinite(force).all() and torch.isfinite(torque).all()
    peak_force = float(force[:, :2].norm(dim=-1).max())
    assert peak_force < 200.0, peak_force  # BlueBoat thrust is 80 N

    calm = SeaState(SeaStateCfg(significant_height_m=0.0, n_components=4), 8, device)
    f0, t0 = calm.forces(torch.zeros((8, 2)), torch.zeros((8, 2)), 5.0)
    assert float(f0.abs().max()) < 1e-9 and float(t0.abs().max()) < 1e-9

    # 6. Spatial coherence: the field is smooth, so a hull-length step sees
    # almost the same surface (a white-noise "wave" would fail this).
    lam_peak = 2.0 * math.pi / float(sea.k[len(sea.k) // 2])
    p1 = torch.zeros((n_envs, 2))
    p2 = torch.zeros((n_envs, 2))
    p2[:, 0] = 1.2  # one BlueBoat hull length
    e1, e2 = sea.elevation(p1, 0.0), sea.elevation(p2, 0.0)
    corr = float(torch.corrcoef(torch.stack([e1, e2]))[0, 1])
    assert corr > 0.7, (corr, lam_peak)

    print(
        f"H_s target {hs:.2f} m -> measured {hs_measured:.3f} m | "
        f"peak orbital force {peak_force:.1f} N | spatial corr {corr:.2f}"
    )
    print("PASS: sea state (dispersion, H_s, zero mean, determinism, bounds)")


if __name__ == "__main__":
    main()
