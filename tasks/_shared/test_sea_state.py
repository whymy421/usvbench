"""Physics acceptance tests for the merged JONSWAP sea state (pure torch)."""

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
    n_envs = 64

    # Pin H_s / T_p / gamma so the spectrum can be checked against theory.
    hs, tp = 0.60, 5.0
    cfg = SeaStateCfg(
        enable=True, hs_range=(hs, hs), tp_range=(tp, tp), gamma_range=(3.3, 3.3)
    )
    sea = SeaState(cfg, n_envs, device)

    # 1. Deep-water dispersion holds for every discretised component.
    assert torch.allclose(sea.omega**2, GRAVITY * sea.k, rtol=1e-6)

    # 2. Spectral energy: H_s = 4 sqrt(m0) with m0 = sum(a_i^2 / 2).
    m0_spec = float((sea.amplitude[0] ** 2 / 2.0).sum())
    hs_spec = 4.0 * math.sqrt(m0_spec)
    assert abs(hs_spec - hs) / hs < 0.10, (hs_spec, hs)

    # 3. The realised time series reproduces the same H_s.
    xy = torch.zeros((n_envs, 2), device=device)
    samples = torch.stack(
        [sea.elevation(xy, t) for t in torch.linspace(0, 300, 6000).tolist()]
    )
    hs_measured = 4.0 * math.sqrt(float(samples.var(dim=0).mean()))
    assert abs(hs_measured - hs) / hs < 0.15, (hs_measured, hs)
    assert abs(float(samples.mean())) < 0.02 * hs  # no spurious set-up

    # 4. Determinism, then genuine re-randomisation on reset.
    a = sea.elevation(xy, 7.0).clone()
    assert torch.equal(a, sea.elevation(xy, 7.0))
    sea.resample(torch.arange(n_envs))
    assert not torch.allclose(a, sea.elevation(xy, 7.0))

    # 5. Forces finite and bounded; a calm sea produces exactly nothing.
    vel = torch.zeros((n_envs, 2), device=device)
    force, torque = sea.forces(xy, vel, 3.5)
    assert torch.isfinite(force).all() and torch.isfinite(torque).all()
    peak_planar = float(force[:, :2].norm(dim=-1).max())
    peak_yawless_moment = float(torque[:, :2].norm(dim=-1).max())
    assert peak_planar < 400.0, peak_planar

    calm = SeaState(SeaStateCfg(hs_range=(0.0, 0.0)), 8, device)
    f0, t0 = calm.forces(torch.zeros((8, 2)), torch.zeros((8, 2)), 5.0)
    assert float(f0.abs().max()) < 1e-9 and float(t0.abs().max()) < 1e-9

    # 6. Spatial smoothness: one hull length apart the surface barely changes.
    p2 = torch.zeros((n_envs, 2))
    p2[:, 0] = 1.2
    corr = float(
        torch.corrcoef(torch.stack([sea.elevation(xy, 0.0), sea.elevation(p2, 0.0)]))[
            0, 1
        ]
    )
    assert corr > 0.7, corr

    # 7. Sea-state observation: bow-relative encoding is a unit-circle pair.
    forward = torch.zeros((n_envs, 2))
    forward[:, 0] = 1.0
    obs = sea.observation(forward)
    assert obs.shape == (n_envs, 3)
    unit = obs[:, 0] ** 2 + obs[:, 1] ** 2
    assert torch.allclose(unit, torch.ones_like(unit), atol=1e-5)

    # 8. Monotone severity: a bigger sea pushes harder (admission sanity).
    big = SeaState(SeaStateCfg(hs_range=(1.5, 1.5), tp_range=(tp, tp)), n_envs, device)
    fb, _ = big.forces(xy, vel, 3.5)
    assert float(fb[:, :2].norm(dim=-1).mean()) > float(
        force[:, :2].norm(dim=-1).mean()
    )

    print(
        f"H_s {hs:.2f} m -> spectral {hs_spec:.3f} m, realised {hs_measured:.3f} m | "
        f"peak planar force {peak_planar:.1f} N | trim moment {peak_yawless_moment:.1f} N*m "
        f"| spatial corr {corr:.2f}"
    )
    print("PASS: JONSWAP sea state (spectrum, H_s, determinism, bounds, obs)")


if __name__ == "__main__":
    main()
