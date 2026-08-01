"""Pure-Python acceptance test for the shared wave fields.

Runnable without Isaac Sim or Isaac Lab:

    python tasks/_shared/test_waves.py
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

try:
    from .waves import (
        AiryWaveField,
        CalmWater,
        JONSWAPWaveField,
        make_wave_field,
    )
except ImportError:  # Direct execution from the repository root.
    from waves import (
        AiryWaveField,
        CalmWater,
        JONSWAPWaveField,
        make_wave_field,
    )


DEVICE = torch.device("cpu")


def _positions(num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(num_envs, generator=generator) * 40.0 - 20.0
    y = torch.rand(num_envs, generator=generator) * 40.0 - 20.0
    return x, y


def test_calm_is_exactly_zero() -> None:
    field = CalmWater(num_envs=8, device=DEVICE)
    x, y = _positions(8)
    assert torch.equal(field.elevation(3.7, x, y), torch.zeros(8))
    assert torch.equal(field.slope(3.7, x, y), torch.zeros(8, 2))
    print("calm water: elevation and slope are exact zeros")


def test_airy_amplitude_and_dispersion() -> None:
    height = 0.8
    period = 5.0
    field = AiryWaveField(
        num_envs=64,
        device=DEVICE,
        height_m=height,
        period_s=period,
        direction_deg=0.0,
    )
    # Deep-water dispersion: k = omega^2 / g.
    expected_k = (2.0 * math.pi / period) ** 2 / 9.81
    assert abs(field.wave_number - expected_k) < 1e-9

    # Sweep a full period at the origin; the envelope must be the amplitude.
    x = torch.zeros(64)
    y = torch.zeros(64)
    peak = max(
        field.elevation(t * period / 200.0, x, y).max().item()
        for t in range(201)
    )
    assert abs(peak - height / 2.0) < 1e-3, peak
    print(f"airy: peak elevation {peak:.4f} m matches amplitude {height / 2:.4f} m")


def test_jonswap_spectrum_recovers_hs() -> None:
    """4 * sqrt(m0) must return the significant wave height that built it.

    Swept across the corners of the band actually configured for the task, not
    just a comfortable mid-band point: the closed-form gamma normalisation this
    replaced held at Tp = 5 s / gamma = 3.3 but drifted ~8% low at Tp = 4 s /
    gamma = 5, which is inside the shipped range.
    """
    worst = 0.0
    for hs in (0.06, 0.18, 1.0):
        for tp, band in ((2.0, (0.10, 1.60)), (4.0, (0.10, 1.60)), (5.0, (0.04, 0.50))):
            for gamma in (1.0, 3.3, 5.0):
                field = JONSWAPWaveField(
                    num_envs=1,
                    device=DEVICE,
                    hs_range=(hs, hs),
                    tp_range=(tp, tp),
                    gamma_range=(gamma, gamma),
                    f_min=band[0],
                    f_max=band[1],
                )
                m0 = (field.amplitudes[0] ** 2 / 2.0).sum().item()
                recovered = 4.0 * math.sqrt(m0)
                error = abs(recovered - hs) / hs
                worst = max(worst, error)
                assert error < 0.01, (hs, tp, gamma, band, recovered, error)
    print(f"jonswap: Hs recovered within {worst:.2%} across the whole "
          f"Hs x Tp x gamma x band grid")


def test_slope_matches_numerical_gradient() -> None:
    """The analytic slope must equal a central difference of the elevation.

    This is the term that tilts the restoring equilibrium, so a sign or
    direction error here silently produces plausible-looking but wrong roll.
    """
    num_envs = 32
    x, y = _positions(num_envs)
    t = 12.34
    # Central differences in float32 over coordinates of order 10 m bottom out
    # near 1e-4; the tolerance sits above that floor and still catches any sign,
    # direction, or factor error, which are all O(1).
    eps = 1e-2
    tolerance = 1e-3

    fields = {
        "airy": AiryWaveField(
            num_envs=num_envs, device=DEVICE, height_m=1.0, period_s=6.0
        ),
        "jonswap": JONSWAPWaveField(num_envs=num_envs, device=DEVICE),
    }
    for name, field in fields.items():
        analytic = field.slope(t, x, y)
        numeric_x = (
            field.elevation(t, x + eps, y) - field.elevation(t, x - eps, y)
        ) / (2.0 * eps)
        numeric_y = (
            field.elevation(t, x, y + eps) - field.elevation(t, x, y - eps)
        ) / (2.0 * eps)
        err_x = (analytic[:, 0] - numeric_x).abs().max().item()
        err_y = (analytic[:, 1] - numeric_y).abs().max().item()
        assert err_x < tolerance, (name, err_x)
        assert err_y < tolerance, (name, err_y)
        print(f"{name}: analytic slope matches central difference "
              f"(max err {max(err_x, err_y):.2e})")


def test_pinned_direction_is_shared_by_all_envs() -> None:
    """A pinned heading is what a controlled beam-sea sweep needs."""
    field = JONSWAPWaveField(num_envs=16, device=DEVICE, direction_deg=90.0)
    assert torch.allclose(
        field.direction[:, 1], torch.ones(16), atol=1e-6
    ), field.direction
    random_field = JONSWAPWaveField(num_envs=64, device=DEVICE)
    assert random_field.direction.std(dim=0).min().item() > 0.1
    print("direction: pinned envs share a heading, unpinned ones spread")


def test_randomize_only_touches_requested_envs() -> None:
    field = JONSWAPWaveField(num_envs=8, device=DEVICE)
    before = field.hs.clone()
    field.randomize(torch.tensor([1, 3]))
    changed = (field.hs != before).nonzero().flatten().tolist()
    assert set(changed) <= {1, 3}, changed
    print("randomize: untouched envs keep their sea state")


def test_station_sampling_separates_roll_from_pitch() -> None:
    """The whole point of sampling across the hull rather than at a point.

    A beam sea must put a height difference across the beam (which becomes
    roll) and almost none along the length; a head sea must do the opposite.
    With one sample at the origin both are identically zero and no attitude
    response exists at all.
    """
    num_envs = 32
    field = AiryWaveField(
        num_envs=num_envs, device=DEVICE, height_m=0.3, period_s=3.0,
        direction_deg=0.0,  # travelling along +x
    )
    half_len, half_beam = 0.60, 0.3607  # BlueBoat: 1.2 m by 0.7214 m

    # Bow along +y => the +x wave arrives on the beam.
    beam_across = torch.tensor([[-half_len, 0.0], [half_len, 0.0]])
    # Bow along +x => the same wave arrives head on.
    head_along = torch.tensor([[-half_len, 0.0], [half_len, 0.0]])

    def spread(offsets: torch.Tensor) -> float:
        xs = offsets[:, 0].unsqueeze(0).expand(num_envs, -1)
        ys = offsets[:, 1].unsqueeze(0).expand(num_envs, -1)
        diffs = [
            (field.elevation_at(t * 0.05, xs, ys)[:, 1]
             - field.elevation_at(t * 0.05, xs, ys)[:, 0])
            for t in range(120)
        ]
        return torch.cat(diffs).std().item()

    # Stations separated along the wave direction see a large difference;
    # stations separated across it see none.
    along = spread(beam_across)
    across = spread(torch.stack([
        torch.tensor([0.0, -half_beam]), torch.tensor([0.0, half_beam])
    ]))
    assert along > 10.0 * across, (along, across)
    print(f"stations: along-wave spread {along:.4f} m vs across-wave "
          f"{across:.6f} m")


def test_orbital_velocity_matches_airy_kinematics() -> None:
    """Surface orbital velocity is a*omega along the direction of travel."""
    num_envs = 16
    height, period = 0.4, 3.0
    field = AiryWaveField(
        num_envs=num_envs, device=DEVICE, height_m=height, period_s=period,
        direction_deg=0.0,
    )
    x = torch.zeros(num_envs)
    y = torch.zeros(num_envs)
    samples = torch.stack(
        [field.orbital_velocity(t * period / 240.0, x, y)[0] for t in range(241)]
    )
    expected_peak = (height / 2.0) * (2.0 * math.pi / period)
    assert abs(samples[:, 0].abs().max().item() - expected_peak) < 1e-3
    # Motion is along +x only, since the wave travels along +x.
    assert samples[:, 1].abs().max().item() < 1e-6
    print(f"orbital: peak {samples[:, 0].abs().max():.4f} m/s matches "
          f"a*omega = {expected_peak:.4f} m/s, no cross-track component")


def test_factory_dispatch() -> None:
    cfg = SimpleNamespace(
        mode="calm",
        direction_deg=None,
        airy_height_m=0.5,
        airy_period_s=5.0,
        hs_min_m=0.3,
        hs_max_m=1.0,
        tp_min_s=4.0,
        tp_max_s=7.0,
        gamma_min=1.0,
        gamma_max=5.0,
        n_components=30,
        f_min_hz=0.04,
        f_max_hz=0.5,
        spread_deg=30.0,
    )
    assert isinstance(make_wave_field(cfg, 4, DEVICE), CalmWater)
    cfg.mode = "airy"
    assert isinstance(make_wave_field(cfg, 4, DEVICE), AiryWaveField)
    cfg.mode = "jonswap"
    assert isinstance(make_wave_field(cfg, 4, DEVICE), JONSWAPWaveField)
    cfg.mode = "typo"
    try:
        make_wave_field(cfg, 4, DEVICE)
    except ValueError:
        pass
    else:  # A silent fallback to calm would fake a passing wave experiment.
        raise AssertionError("unknown wave mode must raise")
    print("factory: modes dispatch and an unknown mode raises")


if __name__ == "__main__":
    test_calm_is_exactly_zero()
    test_airy_amplitude_and_dispersion()
    test_jonswap_spectrum_recovers_hs()
    test_slope_matches_numerical_gradient()
    test_pinned_direction_is_shared_by_all_envs()
    test_randomize_only_touches_requested_envs()
    test_station_sampling_separates_roll_from_pitch()
    test_orbital_velocity_matches_airy_kinematics()
    test_factory_dispatch()
    print("\nAll wave-field checks passed.")
