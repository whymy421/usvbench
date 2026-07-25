"""Acceptance tests for the shared body-frame kinematics block."""

from __future__ import annotations

import math

import torch

try:
    from .kinematics import body_planar_kinematics
    from .obs_superset import SPEED_SCALE_MPS
except ImportError:  # direct execution
    from kinematics import body_planar_kinematics
    from obs_superset import SPEED_SCALE_MPS


def main() -> None:
    # 1. Pure forward motion: surge only.
    forward = torch.tensor([[1.0, 0.0]])
    vel = torch.tensor([[1.0, 0.0, 0.0]])
    yaw = torch.zeros(1)
    out = body_planar_kinematics(forward, vel, yaw)
    assert abs(float(out[0, 0]) - 1.0 / SPEED_SCALE_MPS) < 1e-6, out
    assert abs(float(out[0, 1])) < 1e-6 and abs(float(out[0, 2])) < 1e-6

    # 2. Pure sideways motion: sway only, and the sign follows the left normal.
    vel = torch.tensor([[0.0, 1.0, 0.0]])
    out = body_planar_kinematics(forward, vel, yaw)
    assert abs(float(out[0, 0])) < 1e-6
    assert abs(float(out[0, 1]) - 1.0 / SPEED_SCALE_MPS) < 1e-6

    # 3. Frame equivariance: rotating hull AND velocity leaves the block fixed.
    for angle in (0.3, 1.1, -2.4):
        c, s = math.cos(angle), math.sin(angle)
        rot = torch.tensor([[c, -s], [s, c]])
        f2 = (rot @ forward[0]).unsqueeze(0)
        v2 = torch.zeros((1, 3))
        v2[0, :2] = rot @ torch.tensor([0.7, -0.4])
        base = body_planar_kinematics(
            forward, torch.tensor([[0.7, -0.4, 0.0]]), yaw
        )
        rotated = body_planar_kinematics(f2, v2, yaw)
        assert torch.allclose(base, rotated, atol=1e-6), (angle, base, rotated)

    # 4. Yaw rate passes through its own scale and clamps.
    out = body_planar_kinematics(forward, vel, torch.tensor([2.5]),
                                 yaw_rate_scale_rad_s=1.0)
    assert float(out[0, 2]) == 1.0
    out = body_planar_kinematics(forward, vel, torch.tensor([-0.5]),
                                 yaw_rate_scale_rad_s=2.0)
    assert abs(float(out[0, 2]) + 0.25) < 1e-6

    # 5. Fast motion clamps instead of exploding the observation range.
    out = body_planar_kinematics(forward, torch.tensor([[50.0, 0.0, 0.0]]), yaw)
    assert float(out[0, 0]) == 1.0

    # 6. Batched shape and finiteness.
    n = 64
    f = torch.nn.functional.normalize(torch.randn(n, 2), dim=-1)
    v = torch.randn(n, 3)
    w = torch.randn(n)
    out = body_planar_kinematics(f, v, w)
    assert out.shape == (n, 3) and torch.isfinite(out).all()

    print("PASS: body-frame kinematics (surge/sway/yaw, equivariance, clamps)")


if __name__ == "__main__":
    main()
