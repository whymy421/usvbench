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
    """4 * sqrt(m0) must return the significant wave height that built it."""
    for hs in (0.5, 1.0, 2.0):
        field = JONSWAPWaveField(
            num_envs=1,
            device=DEVICE,
            hs_range=(hs, hs),
            tp_range=(5.0, 5.0),
            gamma_range=(3.3, 3.3),
        )
        # m0 = sum(a_n^2 / 2) for a component sum.
        m0 = (field.amplitudes[0] ** 2 / 2.0).sum().item()
        recovered = 4.0 * math.sqrt(m0)
        error = abs(recovered - hs) / hs
        assert error < 0.05, (hs, recovered, error)
        print(f"jonswap: Hs={hs} m recovered as {recovered:.3f} m ({error:.1%})")


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


def test_roll_torque_follows_the_hull_not_the_world() -> None:
    """Beam seas must roll the boat whatever heading it is on.

    Guards the bug this coupling was rewritten to avoid: driving roll from a
    mean-direction magnitude about a fixed world axis makes the same sea roll
    the hull on one heading and pitch it on another.
    """
    num_envs = 64
    field = JONSWAPWaveField(num_envs=num_envs, device=DEVICE, direction_deg=0.0)
    x = torch.zeros(num_envs)
    y = torch.zeros(num_envs)

    # Beam-on: bow along +y, so a wave travelling along +x hits the beam.
    beam = torch.zeros(num_envs, 2)
    beam[:, 1] = 1.0
    # Head-on: bow along +x, straight into the same wave.
    head = torch.zeros(num_envs, 2)
    head[:, 0] = 1.0

    gains = dict(heave_gain=100.0, roll_gain=200.0, drag_gain=15.0)
    beam_roll = torch.cat(
        [field.compute_forces(t * 0.05, x, y, beam, **gains)["roll_torque"]
         for t in range(200)]
    )
    head_roll = torch.cat(
        [field.compute_forces(t * 0.05, x, y, head, **gains)["roll_torque"]
         for t in range(200)]
    )
    assert beam_roll.std() > 5.0 * head_roll.std(), (
        beam_roll.std().item(), head_roll.std().item()
    )
    print(f"roll: beam-on RMS {beam_roll.std():.2f} N m vs head-on "
          f"{head_roll.std():.2f} N m")


def test_drag_only_opposes_a_head_sea() -> None:
    num_envs = 32
    field = AiryWaveField(
        num_envs=num_envs, device=DEVICE, height_m=0.5, direction_deg=0.0
    )
    x = torch.zeros(num_envs)
    y = torch.zeros(num_envs)
    gains = dict(heave_gain=100.0, roll_gain=200.0, drag_gain=15.0)

    into = torch.zeros(num_envs, 2)
    into[:, 0] = -1.0  # bow towards -x, wave travels +x: head sea
    following = torch.zeros(num_envs, 2)
    following[:, 0] = 1.0  # bow with the wave

    into_drag = torch.cat(
        [field.compute_forces(t * 0.05, x, y, into, **gains)["wave_drag"]
         for t in range(100)]
    )
    follow_drag = torch.cat(
        [field.compute_forces(t * 0.05, x, y, following, **gains)["wave_drag"]
         for t in range(100)]
    )
    assert into_drag.max() > 0.0
    assert torch.equal(follow_drag, torch.zeros_like(follow_drag))
    print(f"drag: head sea peaks at {into_drag.max():.2f} N, following sea 0")


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
    test_roll_torque_follows_the_hull_not_the_world()
    test_drag_only_opposes_a_head_sea()
    test_factory_dispatch()
    print("\nAll wave-field checks passed.")
