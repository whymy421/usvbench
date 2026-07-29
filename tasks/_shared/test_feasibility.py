"""Acceptance tests for feasibility pooling.

The property that matters: a gap the hull FITS THROUGH must read as open water,
and a gap it does not fit through must read as blocked at the gap.
"""

from __future__ import annotations

import math

import torch

try:
    from .feasibility import feasibility_pool
except ImportError:  # direct execution
    from feasibility import feasibility_pool

BEAM = 0.899
SPACING = math.radians(10.0)  # 36 rays over 360 deg
MAX_RANGE = 30.0


def sector_case(readings, n_sectors=1, width=BEAM, mult=1.0):
    x = torch.tensor([readings], dtype=torch.float32)
    return float(
        feasibility_pool(
            x, n_sectors=n_sectors, vessel_width_m=width,
            ray_spacing_rad=SPACING, width_multiplier=mult
        )[0, 0]
    )


def main() -> None:
    # 1. Wide open: every ray at max range -> feasible = max range.
    assert abs(sector_case([MAX_RANGE] * 6) - MAX_RANGE) < 1e-4

    # 2. Solid wall at 5 m across the whole sector -> blocked at 5 m.
    assert abs(sector_case([5.0] * 6) - 5.0) < 1e-4

    # 3. A PASSABLE gap: two pillars at 6 m with four open rays between them.
    #    At 6 m, four rays subtend 4 * 6 * 10deg = 4.19 m >> 0.9 m beam,
    #    so the hull fits and the sector must read open, NOT 6 m.
    passable = [6.0, MAX_RANGE, MAX_RANGE, MAX_RANGE, MAX_RANGE, 6.0]
    got = sector_case(passable)
    assert got > 20.0, f"passable gap read as blocked at {got:.2f} m"

    # 4. The same geometry with min-pooling would report 6.0 -- the failure
    #    mode Meyer et al. warn about. Show the two disagree.
    assert min(passable) == 6.0 and got > 20.0

    # 5. An IMPASSABLE slit: pillars at 1 m with ONE open ray between them.
    #    At 1 m a single ray subtends 1 * 10deg = 0.17 m < 0.9 m beam.
    slit = [1.0, 1.0, MAX_RANGE, 1.0, 1.0, 1.0]
    got = sector_case(slit)
    assert got <= 1.0 + 1e-4, f"impassable slit read as open at {got:.2f} m"

    # 6. Monotone in hull width: a wider vessel can never travel further.
    narrow = sector_case(passable, width=0.5)
    wide = sector_case(passable, width=8.0)
    assert wide <= narrow + 1e-6, (narrow, wide)

    # 7. The SAME ray pattern means different things at different distances,
    #    because an angular gap subtends more metres the further away it is.
    #    Four open rays span 4 * d * 10deg: at d = 1 m that is 0.70 m, narrower
    #    than the 0.899 m beam, so the hull is blocked; at d = 6 m it is 4.19 m
    #    and the hull passes. This distance dependence is the whole point --
    #    a raw ray reading of "1.0" cannot express it.
    near_gap = [1.0, 6.0, 6.0, 6.0, 6.0, 1.0]
    far_gap = [6.0, MAX_RANGE, MAX_RANGE, MAX_RANGE, MAX_RANGE, 6.0]
    assert sector_case([1.0] * 6) <= 1.0 + 1e-4
    assert sector_case(near_gap) <= 1.0 + 1e-4, sector_case(near_gap)
    assert sector_case(far_gap) > 20.0, sector_case(far_gap)

    # 8. Batched and multi-sector shapes come back right and finite.
    x = torch.rand(16, 36) * MAX_RANGE
    out = feasibility_pool(
        x, n_sectors=9, vessel_width_m=BEAM, ray_spacing_rad=SPACING
    )
    assert out.shape == (16, 9) and torch.isfinite(out).all()
    assert (out <= x.max() + 1e-4).all() and (out >= 0).all()

    print(
        f"可通行缝(6 m 处 4 条射线): 池化后 {sector_case(passable):.1f} m "
        f"(取最小值会报 {min(passable):.1f} m)"
    )
    print(f"不可通行细缝(1 m 处 1 条射线): 池化后 {sector_case(slit):.2f} m")
    print("PASS: feasibility pooling (passable=open, impassable=blocked, width-monotone)")


if __name__ == "__main__":
    main()
