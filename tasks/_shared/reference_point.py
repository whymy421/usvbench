"""The COM is the one physical reference for task and hydrodynamic state.

Use actual COM world position/velocity for observations, goals, success,
distances, trajectory length, wave sampling and relative fluid velocities.
Spawn targets and collision geometry are also COM referenced. Actor/root
origins are engine pose handles, never an alternative task reference.

``rotation`` maps hull/actor axes to world axes and ``com_offset_body`` is the
vector from the actor origin to the COM expressed in those axes. A changed
payload COM must update this offset and inertia consistently. These helpers
do not change mass properties and contain no Isaac-specific imports.

When engine data already supplies ``root_com_pos_w``/``root_com_vel_w``, use
those directly; do NOT apply these conversions a second time. The name of an
engine root velocity API does not establish its reference point: the caller
must verify that API before choosing a conversion.

E2 / controller decision 12 (COM source for v2 assets). Every v2 asset
(tasks/_shared/vehicles_v2.py, validate_vehicle) has its actor origin AT the
aggregate COM, so the root-link pose IS the COM pose. Use
com_state_from_root_link: position and orientation from ``root_link_pose_w``
and velocity from ``root_com_vel_w``. Reason (Isaac Lab 2.3.2 source,
C:/Users/BRADY/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object/):
``root_com_pose_w`` / ``root_com_pos_w`` is a lazily derived buffer
(rigid_object_data.py:164-180, recomputed only when its timestamp is older
than the simulation clock), and a pose write refreshes ``root_link_pose_w``
(rigid_object.py:263) but not ``root_com_pose_w`` (rigid_object.py:246-282),
so right after a reset the COM position can be the pre-reset value until the
next scene update (D6 FINAL_REPORT finding 5). ``root_com_vel_w`` is the
PhysX velocity buffer itself (rigid_object_data.py:182-193) and a velocity
write refreshes it (rigid_object.py:351). For an origin = COM asset the link
and COM velocities coincide (rigid_object_data.py:151-157 with a zero
body_com_pos_b). The identity is only valid for a ZERO COM offset, which
require_zero_com_offset enforces (HydroV2Runtime checks it at construction).
Whether the installed engine really reports body_com_pos_b = 0 for these
assets is a GPU readback gate (UNVERIFIED on CPU).
"""
from __future__ import annotations

import math

import torch


REFERENCE_POINT = "center_of_mass"
COM_SOURCES = ("root_link", "root_com")


def require_zero_com_offset(com_offset_body) -> None:
    """Fail closed unless the declared actor-origin-to-COM offset is exactly zero.

    Accepts the profile's Python tuple (never a device tensor: this is a
    construction-time check without host synchronisation).
    """
    if isinstance(com_offset_body, torch.Tensor):
        raise ValueError("pass the profile's declared COM offset tuple, not a tensor")
    values = tuple(com_offset_body)
    if len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in values):
        raise ValueError("COM offset must be three finite numbers")
    if any(v != 0 for v in values):
        raise ValueError("root-link COM source requires an asset whose actor origin is the COM "
                         "(declared offset %r is not zero); use actor/COM conversions instead"
                         % (values,))


def com_state_from_root_link(root_link_pose_w: torch.Tensor, root_com_vel_w: torch.Tensor,
                             com_offset_body=(0.0, 0.0, 0.0)):
    """COM state of an origin = COM asset from the engine's fresh buffers.

    root_link_pose_w [N,7] = (x, y, z, qw, qx, qy, qz) and root_com_vel_w
    [N,6] = (linear, angular) in world axes, as named in Isaac Lab 2.3.2.
    Returns (com_position_w [N,3], com_velocity_w [N,6], actor_quat_wxyz [N,4])
    as views/slices (no copy, no host synchronisation). ``com_offset_body``
    must be the profile's zero offset (checked; see module docstring).
    """
    require_zero_com_offset(com_offset_body)
    if root_link_pose_w.shape[-1:] != (7,) or root_com_vel_w.shape[-1:] != (6,) \
            or root_link_pose_w.shape[:-1] != root_com_vel_w.shape[:-1]:
        raise ValueError("expected root_link_pose_w [N,7] and root_com_vel_w [N,6]")
    return root_link_pose_w[..., :3], root_com_vel_w, root_link_pose_w[..., 3:7]


def _rotate(rotation, vector):
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


def com_position_world(actor_position_world: torch.Tensor,
                       rotation: torch.Tensor,
                       com_offset_body: torch.Tensor) -> torch.Tensor:
    """Actor pose -> COM position, x_com = x_actor + R r_com."""
    return actor_position_world + _rotate(rotation, com_offset_body)


def actor_position_from_com(com_position: torch.Tensor,
                            rotation: torch.Tensor,
                            com_offset_body: torch.Tensor) -> torch.Tensor:
    """Desired COM spawn -> actor pose passed to an actor-pose engine API."""
    return com_position - _rotate(rotation, com_offset_body)


def com_velocity_world(actor_linear_velocity_world: torch.Tensor,
                       angular_velocity_world: torch.Tensor,
                       rotation: torch.Tensor,
                       com_offset_body: torch.Tensor) -> torch.Tensor:
    """v_com = v_actor + omega_world cross (R r_com)."""
    return actor_linear_velocity_world + torch.cross(
        angular_velocity_world, _rotate(rotation, com_offset_body), dim=-1)


def actor_velocity_from_com(com_velocity: torch.Tensor,
                            angular_velocity_world: torch.Tensor,
                            rotation: torch.Tensor,
                            com_offset_body: torch.Tensor) -> torch.Tensor:
    """COM velocity -> actor-origin velocity, only for APIs needing the latter."""
    return com_velocity - torch.cross(
        angular_velocity_world, _rotate(rotation, com_offset_body), dim=-1)


def com_centered_vertices_body(actor_vertices_body: torch.Tensor,
                               com_offset_body: torch.Tensor) -> torch.Tensor:
    """Rebase actor-frame collision vertices around the COM.

    Vertices end in (vertex_count,3); offsets end in 3. Per-environment
    vertex sets and offsets may use broadcastable leading batch dimensions.
    A symmetric actor bounding box is generally asymmetric around the COM.
    """
    return actor_vertices_body - com_offset_body.unsqueeze(-2)


def collision_vertices_world(com_position: torch.Tensor,
                             rotation: torch.Tensor,
                             actor_vertices_body: torch.Tensor,
                             com_offset_body: torch.Tensor) -> torch.Tensor:
    """Transform the actual footprint from COM through FULL hull attitude.

    Include 3-D collision vertices/extreme points, then project for planar
    tests. Yaw-only rotation of a flat rectangle misses tilted hull extents.
    This transforms supplied geometry; a convex hull/collision test remains
    the caller's responsibility. Mesh holes and concavity are not invented.
    """
    vertices = com_centered_vertices_body(actor_vertices_body, com_offset_body)
    return vertices @ rotation.transpose(-1, -2) + com_position.unsqueeze(-2)


def collision_footprint_world_xy(com_position: torch.Tensor,
                                 rotation: torch.Tensor,
                                 actor_vertices_body: torch.Tensor,
                                 com_offset_body: torch.Tensor) -> torch.Tensor:
    """World XY projection of the supplied COM-referenced collision vertices."""
    return collision_vertices_world(com_position, rotation, actor_vertices_body,
                                    com_offset_body)[..., :2]


def environment_local_position(com_position: torch.Tensor,
                                env_origin_world: torch.Tensor) -> torch.Tensor:
    """World COM -> independent-clone coordinates for tasks and wave phase."""
    return com_position - env_origin_world
