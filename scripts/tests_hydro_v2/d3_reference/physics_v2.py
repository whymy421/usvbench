"""CPU-testable implicit hull damping and per-environment wave clocks.

No Isaac imports. The adapter consumes Isaac Lab data and body-frame wrench
buffers. Default inertia is about the COM, expressed in the actor frame (see
RigidObjectData.default_inertia). Only single rigid hulls are supported in v2.
"""
from __future__ import annotations

import math
import torch


def rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Actor-to-world matrix from Isaac's unit (w,x,y,z) quaternion."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack((
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)


def implicit_damping_wrench(velocity, mass_matrix, damping_matrix, dt,
                            other_wrench=None, water_velocity=None):
    """Return a wrench realizing one backward-Euler damping step.

    Freeze the nonnegative quadratic-drag coefficient at the current speed.
    Solve (M + dt D) v_next = M v + dt f_other; return -D v_next.

    D3 extension (optional ``water_velocity`` u, same frame as ``velocity``):
    solve (M + dt D) v_next = M v + dt (f_other + D u) and return
    -D (v_next - u), i.e. damping of the velocity RELATIVE to the water, with
    D u entering as known forcing. Omitting u (None) executes exactly the C2
    arithmetic. D must be frozen at the water-relative state by the caller.
    Applying f_other plus that wrench through the usual force API produces
    the same step. For damping alone, in mass-normalized coordinates the
    amplification eigenvalues are 1/(1+dt*lambda), lambda >= 0. Hence energy
    cannot grow for ANY nonnegative damping or positive dt (no explicit
    c*dt/I < 2 condition). This claim excludes springs, contacts, gyroscopic
    integration, and external forcing; these still require simulation tests.
    """
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("physics dt must be positive and finite")
    rhs = (mass_matrix @ velocity.unsqueeze(-1)).squeeze(-1)
    if other_wrench is not None:
        rhs = rhs + dt * other_wrench
    if water_velocity is not None:
        rhs = rhs + dt * (damping_matrix @ water_velocity.unsqueeze(-1)).squeeze(-1)
    next_velocity = torch.linalg.solve(
        mass_matrix + dt * damping_matrix, rhs.unsqueeze(-1)
    )
    if water_velocity is not None:
        next_velocity = next_velocity - water_velocity.unsqueeze(-1)
    return -(damping_matrix @ next_velocity).squeeze(-1)


def _per_env(env, name, default):
    if getattr(env.cfg, name + "_choices", ()):
        return getattr(env, "_" + name + "_per_env")
    return torch.full((env.num_envs,), float(getattr(env.cfg, name, default)),
                      device=env.device, dtype=env.robot.data.default_mass.dtype)


def add_implicit_hull_damping(env, forces, torques, quat):
    """Add v2 damping AFTER existing actuator mass/payload wrench scaling.

    Surge/sway are actor-axis drag; heave remains world vertical. Roll/pitch
    and yaw rates in v2 use actor axes, consistent with the hull inertia and
    anisotropic restoring moments. Scaling each drag coefficient after the
    legacy mass/payload divisions preserves their intended acceleration
    factors. Payload-offset coupling of damping is not approximated as an
    additional torque: v2 drag is applied about the authored COM.
    """
    if env.robot.num_bodies != 1:
        raise ValueError("physics v2 requires a single rigid hull")
    cfg = env.physics_cfg
    names = ("surge_lin_damping", "surge_quad_damping", "sway_lin_damping",
             "sway_quad_damping", "heave_damping", "yaw_lin_damping",
             "yaw_quad_damping", "rollpitch_rate_damping")
    if not getattr(env, "_v2_damping_validated", False):
        for name in names:
            value = getattr(cfg, name, None)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(name + " must be finite and nonnegative in v2")
        mass = env.robot.data.default_mass.reshape(env.num_envs)
        inertia = env.robot.data.default_inertia.reshape(env.num_envs, 3, 3)
        if not torch.isfinite(mass).all() or not (mass > 0).all():
            raise ValueError("v2 requires positive finite authored mass")
        if not torch.isfinite(inertia).all() or not torch.allclose(
                inertia, inertia.transpose(-1, -2)):
            raise ValueError("v2 requires a symmetric finite authored inertia")
        if not (torch.linalg.eigvalsh(inertia) > 0).all():
            raise ValueError("v2 requires positive definite authored inertia")
        env._v2_damping_validated = True
    mass = env.robot.data.default_mass.reshape(env.num_envs)
    inertia = env.robot.data.default_inertia.reshape(env.num_envs, 3, 3)
    rot = rotation_matrix(quat)
    inverse = rot.transpose(-1, -2)
    vel_w = env.robot.data.root_com_vel_w
    lin_w = vel_w[:, :3]
    ang_w = vel_w[:, 3:] if vel_w.shape[-1] == 6 else env.robot.data.root_ang_vel_w
    lin_b = (inverse @ lin_w.unsqueeze(-1)).squeeze(-1)
    ang_b = (inverse @ ang_w.unsqueeze(-1)).squeeze(-1)
    drag_scale = _per_env(env, "drag_scale", 1.0)
    hull_scale = _per_env(env, "mass_scale", 1.0)
    payload = _per_env(env, "payload_mass_kg", 0.0)
    if not all(torch.isfinite(value).all() for value in (drag_scale, hull_scale, payload)):
        raise ValueError("v2 drag, mass scale and payload must be finite")
    if not (drag_scale >= 0).all() or not (hull_scale > 0).all() or not (payload >= 0).all():
        raise ValueError("v2 requires nonnegative drag/payload and positive mass scale")
    mass_divisor = hull_scale.clone()
    yaw_divisor = hull_scale.clone()
    if getattr(env.cfg, "payload_mass_kg_choices", ()) or getattr(env.cfg, "payload_mass_kg", 0):
        ref_mass = float(env.cfg.payload_ref_mass_kg)
        ref_izz = float(env.cfg.payload_ref_izz_kgm2)
        hull_mass = hull_scale * ref_mass
        total_mass = hull_mass + payload
        mass_divisor = hull_scale * (total_mass / hull_mass)
        reduced_mass = hull_mass * payload / total_mass
        offset_sq = env.cfg.payload_offset_x_m**2 + env.cfg.payload_offset_y_m**2
        yaw_divisor = hull_scale * (1 + reduced_mass*offset_sq/(hull_scale*ref_izz))
    drag = torch.zeros((env.num_envs, 3, 3), device=env.device, dtype=lin_b.dtype)
    surge_axis = getattr(env, "_surge_axis_idx", 0)
    sway_axis = getattr(env, "_sway_axis_idx", 1)
    if getattr(cfg, "sway_lin_damping", None) is None or getattr(cfg, "sway_quad_damping", None) is None:
        speed = torch.linalg.vector_norm(lin_w[:, :2], dim=-1)
        coeff = (cfg.surge_lin_damping + cfg.surge_quad_damping*speed)*drag_scale/mass_divisor
        planar_w = torch.diag_embed(torch.stack((coeff, coeff, coeff*0), dim=-1))
        drag = inverse @ planar_w @ rot
    else:
        drag[:, surge_axis, surge_axis] = (cfg.surge_lin_damping + cfg.surge_quad_damping*lin_b[:, surge_axis].abs())*drag_scale/mass_divisor
        drag[:, sway_axis, sway_axis] = (cfg.sway_lin_damping + cfg.sway_quad_damping*lin_b[:, sway_axis].abs())*drag_scale/mass_divisor
    vertical_b = rot[:, 2, :]
    drag += (cfg.heave_damping/mass_divisor)[:, None, None] * vertical_b.unsqueeze(-1) * vertical_b.unsqueeze(-2)
    angular_diag = torch.empty_like(ang_b)
    angular_diag[:, :2] = (cfg.rollpitch_rate_damping/hull_scale)[:, None]
    angular_diag[:, 2] = (cfg.yaw_lin_damping + cfg.yaw_quad_damping*ang_b[:, 2].abs())*drag_scale/yaw_divisor
    angular_drag = torch.diag_embed(angular_diag)
    mass_matrix = torch.diag_embed(mass[:, None].expand(-1, 3))
    gravity = torch.as_tensor(getattr(env.cfg.sim, "gravity", (0.0, 0.0, -9.81)),
                              device=env.device, dtype=lin_b.dtype).expand_as(lin_b)
    gravity_b = (inverse @ gravity.unsqueeze(-1)).squeeze(-1)
    force = implicit_damping_wrench(lin_b, mass_matrix, drag, env.cfg.sim.dt,
                                    forces[:, 0, :] + mass[:, None]*gravity_b)
    torque = implicit_damping_wrench(ang_b, inertia, angular_drag, env.cfg.sim.dt,
                                     torques[:, 0, :])
    forces[:, 0, :] += force
    torques[:, 0, :] += torque


class WaveClock:
    """Integer physics ticks avoid drift and reset only selected environments."""
    def __init__(self, num_envs, device, dt):
        if dt <= 0 or not math.isfinite(float(dt)):
            raise ValueError("wave clock dt must be positive and finite")
        self.ticks = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.dt = float(dt)

    def time(self):
        return self.ticks.to(torch.float32) * self.dt

    def advance(self):
        self.ticks += 1

    def reset(self, env_ids):
        self.ticks[env_ids] = 0
