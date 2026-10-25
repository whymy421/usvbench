"""Conservative total hydrostatic righting moment; pure Python and torch.

Axes are a right-handed hull basis: x forward, y port, z up. ``rotation`` maps
hull vectors into world coordinates. The vessel's k_roll and k_pitch are TOTAL
small-angle metacentric stiffnesses about its COM, in N m/rad. Do not add a
separate buoyancy-center cross-force couple to this model. Heave buoyancy is
applied at the COM; all modeled orientation dependence is in this potential.

For q = R.T @ n, m = (k_roll + k_pitch)/2 and d = (k_pitch-k_roll)/2,
    U = m (1-q_z) + d/2 (q_x**2-q_y**2).
Its exact negative rotational gradient is tau_body = q cross grad_q(U).
Near alignment, U = (k_roll*roll**2 + k_pitch*pitch**2)/2 + O(tilt**4).
Unlike independent roll/pitch component scaling, the full gradient includes
the coupled body-z moment. This makes tau_world dot n identically zero.

For calm water n=world_up: U is yaw invariant, and there is no world-vertical
moment at ANY attitude. For a fixed tilted wave normal the symmetry axis is n,
not world_up: a physical world-vertical component is allowed. Requiring both
alignment with a fixed tilted normal and world-yaw invariance is impossible:
yaw rotation would have to preserve every energy minimum while rotating its
aligned normal away from n. Jointly rotating R and n preserves this model.

This is a smooth large-angle continuation of a metacentric small-angle model,
not measured large-angle righting-arm geometry. Positive stiffnesses make
alignment the sole stable orientation (apart from unconstrained yaw about n).
"""
from __future__ import annotations

import torch


def _inputs(rotation, k_roll, k_pitch, normal_world):
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in (3, 3)")
    kr = torch.as_tensor(k_roll, dtype=rotation.dtype, device=rotation.device)
    kp = torch.as_tensor(k_pitch, dtype=rotation.dtype, device=rotation.device)
    if not (torch.isfinite(kr).all() and torch.isfinite(kp).all()):
        raise ValueError("restoring stiffness must be finite")
    if not ((kr >= 0).all() and (kp >= 0).all()):
        raise ValueError("restoring stiffness must be nonnegative")
    if normal_world is None:
        normal = torch.zeros_like(rotation[..., 0])
        normal[..., 2] = 1.0
    else:
        normal = torch.as_tensor(normal_world, dtype=rotation.dtype,
                                 device=rotation.device)
        if normal.shape[-1:] != (3,):
            raise ValueError("normal_world must end in 3")
        norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        if not (torch.isfinite(normal).all() and (norm > 0).all()):
            raise ValueError("normal_world must be finite and nonzero")
        normal = normal / norm
    q = (rotation.transpose(-1, -2) @ normal.unsqueeze(-1)).squeeze(-1)
    return q, (kr + kp) / 2, (kp - kr) / 2


def restoring_potential(rotation: torch.Tensor, k_roll, k_pitch,
                        normal_world=None) -> torch.Tensor:
    """Return righting potential in joules; accepts arbitrary batch dimensions.

    ``rotation`` must be a proper orthonormal matrix. ``normal_world`` defaults
    to world up and is normalized if supplied. Coefficients may be tensors
    broadcastable to the rotation's batch shape. A wave slope (sx,sy) gives
    the upward normal ``(-sx, -sy, 1)``.
    """
    q, mean, difference = _inputs(rotation, k_roll, k_pitch, normal_world)
    return mean * (1 - q[..., 2]) + 0.5 * difference * (
        q[..., 0].square() - q[..., 1].square())


def restoring_moment(rotation: torch.Tensor, k_roll, k_pitch,
                     normal_world=None) -> torch.Tensor:
    """Return the exact negative potential gradient as a COM body moment.

    Returning all three body components is essential even for flat water:
    zero WORLD vertical moment generally requires a nonzero BODY z component.
    The wave normal is held fixed during this orientation derivative. Its
    spatial/time changes represent prescribed external wave excitation.
    """
    q, mean, difference = _inputs(rotation, k_roll, k_pitch, normal_world)
    gx, gy, gz = torch.broadcast_tensors(difference * q[..., 0],
                                       -difference * q[..., 1], -mean)
    gradient = torch.stack((gx, gy, gz), dim=-1)
    return torch.cross(q, gradient, dim=-1)


restoring_torque_body = restoring_moment
