"""Plain-assert tests for the pure-Torch hydrostatic restoring torque."""

import math

import torch

from restoring import restoring_torque_body


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Reference scalar-first quaternion rotation without Isaac imports."""
    quat_vector = quat[..., 1:]
    cross = torch.cross(quat_vector, vector, dim=-1)
    return vector + 2.0 * quat[..., :1] * cross + 2.0 * torch.cross(
        quat_vector, cross, dim=-1
    )


def quat_apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Reference inverse scalar-first quaternion rotation."""
    conjugate = torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)
    return quat_apply(conjugate, vector)


def axis_angle_quat(axis: tuple[float, float, float], angle_deg: float) -> torch.Tensor:
    half_angle = math.radians(angle_deg) / 2.0
    axis_tensor = torch.tensor(axis, dtype=torch.float64)
    axis_tensor /= torch.linalg.vector_norm(axis_tensor)
    return torch.cat(
        (
            torch.tensor([math.cos(half_angle)], dtype=torch.float64),
            axis_tensor * math.sin(half_angle),
        )
    )


def quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compose scalar-first quaternions as left rotation after right."""
    lw, lv = left[:1], left[1:]
    rw, rv = right[:1], right[1:]
    return torch.cat(
        (
            lw * rw - torch.dot(lv, rv),
            lw * rv + rw * lv + torch.cross(lv, rv, dim=0),
        )
    )


def test_restores_roll() -> None:
    roll_deg = 10.0
    quat = axis_angle_quat((1.0, 0.0, 0.0), roll_deg).unsqueeze(0)
    torque = restoring_torque_body(quat, 5000.0, 3000.0)[0]
    assert torque[0] * math.radians(roll_deg) < 0.0
    assert torch.allclose(
        torque[1:], torch.zeros(2, dtype=torque.dtype), atol=1.0e-12
    )


def test_restores_pitch() -> None:
    pitch_deg = 10.0
    quat = axis_angle_quat((0.0, 1.0, 0.0), pitch_deg).unsqueeze(0)
    torque = restoring_torque_body(quat, 5000.0, 3000.0)[0]
    assert torque[1] * math.radians(pitch_deg) < 0.0
    assert torch.allclose(
        torque[[0, 2]], torch.zeros(2, dtype=torque.dtype), atol=1.0e-12
    )


def test_yaw_invariance() -> None:
    tilt = axis_angle_quat((1.0, 2.0, 0.0), 10.0)
    torques = []
    for yaw_deg in (0.0, 45.0, 90.0, 135.0, 180.0, 270.0):
        yaw = axis_angle_quat((0.0, 0.0, 1.0), yaw_deg)
        quat = quat_multiply(yaw, tilt).unsqueeze(0)
        torques.append(restoring_torque_body(quat, 5000.0, 3000.0)[0])
    expected = torques[0]
    assert all(
        torch.allclose(torque, expected, atol=1.0e-10)
        for torque in torques[1:]
    )


def test_isotropic_matches_v9c() -> None:
    generator = torch.Generator().manual_seed(9)
    quats = torch.randn((20, 4), generator=generator, dtype=torch.float64)
    quats /= torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    world_up = torch.tensor([0.0, 0.0, 1.0], dtype=quats.dtype).expand(20, 3)

    up_body_in_world = quat_apply(quats, world_up)
    torque_world_v9c = 4100.0 * torch.stack(
        (
            up_body_in_world[:, 1],
            -up_body_in_world[:, 0],
            torch.zeros(20, dtype=quats.dtype),
        ),
        dim=-1,
    )
    expected_body = quat_apply_inverse(quats, torque_world_v9c)
    actual_body = restoring_torque_body(quats, 4100.0, 4100.0)
    assert torch.allclose(actual_body, expected_body, atol=1.0e-10, rtol=1.0e-10)


def test_upside_down_bounded() -> None:
    stiffness = 5000.0
    for roll_deg in (100.0, 135.0, 170.0):
        quat = axis_angle_quat((1.0, 0.0, 0.0), roll_deg).unsqueeze(0)
        torque = restoring_torque_body(quat, stiffness, 3000.0)[0]
        assert torch.isfinite(torque).all()
        assert abs(torque[0]) <= stiffness
        assert torque[0] < 0.0


def main() -> None:
    test_restores_roll()
    test_restores_pitch()
    test_yaw_invariance()
    test_isotropic_matches_v9c()
    test_upside_down_bounded()
    print("restoring tests: PASS")


if __name__ == "__main__":
    main()
