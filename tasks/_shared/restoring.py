"""Pure-Torch hydrostatic restoring torque shared by environments and tests.

The configured roll and pitch values are effective stiffnesses. When
``buoyancy_center_offset != 0``, total effective stiffness additionally includes
the buoyancy-center couple ``rho * g * V * offset * sin(tilt)``. Double-count
audit conclusion: code-level buoyancy is the only buoyancy source because
PhysX has no native water, so there is no engine-side double counting.
"""

from __future__ import annotations

import torch


def _quat_apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of scalar-first unit quaternions."""
    quat_vector = quat[..., 1:]
    cross = torch.cross(quat_vector, vector, dim=-1)
    return vector - 2.0 * quat[..., :1] * cross + 2.0 * torch.cross(
        quat_vector, cross, dim=-1
    )


def restoring_torque_body(
    quat: torch.Tensor,
    k_roll: float,
    k_pitch: float,
    up_world: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return anisotropic roll/pitch restoring torque in the body frame.

    ``quat`` is a scalar-first unit quaternion rotating body vectors to world.
    If ``b`` is world-up expressed in body coordinates, the restoring direction
    is ``e3_body x b = (-b_y, b_x, 0)``. This is the restoring-sign negative of
    ``b x e3_body`` and exactly matches V9c when both stiffnesses are equal.

    ``up_world`` overrides the direction the hull is restored toward with a
    per-env unit vector, which is how a wave field tilts the equilibrium to the
    local surface normal. Omitting it restores toward global up, the calm-water
    behaviour every pre-wave checkpoint was trained under.
    """
    if up_world is None:
        world_up = torch.zeros_like(quat[..., 1:])
        world_up[..., 2] = 1.0
    else:
        world_up = up_world
    up_world_in_body = _quat_apply_inverse(quat, world_up)

    torque = torch.zeros_like(up_world_in_body)
    torque[..., 0] = -k_roll * up_world_in_body[..., 1]
    torque[..., 1] = k_pitch * up_world_in_body[..., 0]
    return torque
