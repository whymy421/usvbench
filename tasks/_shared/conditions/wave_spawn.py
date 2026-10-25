"""Wave-following reset state (G18 option; ON by default since T25, owner decision T1.35a, 2026-10-02).

Why. Every journal family spawns the hull at the CALM equilibrium height,
level and at rest (e.g. hazard_nav_env.py "3. spawn: COM at water level +
equilibrium COM height, at rest"), while the condition layer draws a new
wave realisation whose clock starts at 0 (HydroV2Runtime.reset). The hull is
therefore released off its wave-following state: heave offset = the
footprint-filtered COM elevation eta_f (rms Hs/4 in long waves), attitude
offset = the filtered slope, velocity offset = the filtered orbital and
surface velocities. The released hull slams towards the surface (initial
vertical acceleration w_n^2 eta_f, several g at Hs 1 m) and, in a share of
episodes that grows with Hs, the local crest already stands above a deck edge
at t = 0 (G18 REPORT section 1).

What. ``wave_following_root_state`` returns, for the reset rows, the root
state with
  * COM height  z = z_spawn + eta_f(c, 0)   (the buoyancy equilibrium of
    hydro_v2.hydrostatic_wrench: water + eta_f + z_eq - z_COM = 0),
  * attitude    body z axis = the filtered surface normal at the spawn
    heading (the restoring target of hydro_v2.hydrostatic_wrench),
  * velocity    v_spawn + (filtered horizontal orbital velocity at the
    orbital sampling depth, d eta_f / dt) and the filtered surface angular
    velocity (the water-relative motion of hydro_wrench is then zero),
all from the SAME SpectralWaves realisation, footprint filter, heading and
clock (t = 0) that HydroV2Runtime.step uses on the first substep. Horizontal
position and heading are unchanged, so layouts, D7 and the appearing channel
see the same spawn.

How it is switched on: cfg.journal_spawn_on_waves (journal_env_mixin
.journal_reset calls it after configure_runtime, before the D7 tick-0 step,
and writes the rows with robot.write_root_state_to_sim). Calm water / Hs = 0
(no wave source) is a no-op. Off (cfg.journal_spawn_on_waves = False): nothing
is read or written.

Default (T25): owner decision T1.35a (2026-10-02) approved G18's recommendation
3: ``JOURNAL_SPAWN_ON_WAVES_DEFAULT = True``, i.e. every journal id (the only
envs that compose journal_env_mixin) spawns on the filtered wave surface unless
its cfg sets journal_spawn_on_waves = False. This changes the simulation of
every wave cell (Hs > 0); calm cells and Hs = 0 stay bitwise unchanged.
"""
from __future__ import annotations

import math

import torch

SPAWN_VERSION = "usvbench.wave_spawn/1"
SPAWN_RULE = ("reset rows rewritten onto the footprint-filtered wave surface at t = 0: z = z_spawn + eta_f(COM), "
              "attitude normal to the filtered surface at the spawn heading, velocity = spawn velocity + filtered "
              "orbital velocity (horizontal) + d eta_f/dt (vertical), angular velocity = filtered surface "
              "angular velocity; x, y and heading unchanged; same realisation, filter and clock as the first "
              "runtime substep")


# Owner decision T1.35a, 2026-10-02 (G18 REPORT recommendation 3): spawn on the filtered wave surface by default.
JOURNAL_SPAWN_ON_WAVES_DEFAULT = True
SPAWN_DEFAULT_DECISION = ("owner decision T1.35a, 2026-10-02: journal ids spawn on the footprint-filtered wave "
                          "surface by default (cfg.journal_spawn_on_waves defaults to True)")


def enabled(cfg):
    value = getattr(cfg, "journal_spawn_on_waves", JOURNAL_SPAWN_ON_WAVES_DEFAULT)
    if not isinstance(value, bool):
        raise ValueError("cfg.journal_spawn_on_waves must be a bool")
    return value


def stamp():
    return {"version": SPAWN_VERSION, "mode": "wave_following", "rule": SPAWN_RULE}


def _hydro_v2():
    from importlib import import_module
    return import_module(__package__.rsplit(".", 1)[0] + ".hydro_v2")


def _quat_from_euler(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)


def normal_attitude(normal, yaw):
    """(roll, pitch) of the Z-Y-X attitude with heading yaw whose body z axis is the unit vector ``normal``."""
    nx = normal[:, 0] * torch.cos(yaw) + normal[:, 1] * torch.sin(yaw)
    ny = -normal[:, 0] * torch.sin(yaw) + normal[:, 1] * torch.cos(yaw)
    return torch.asin(torch.clamp(-ny, -1.0, 1.0)), torch.atan2(nx, normal[:, 2])


def wave_following_root_state(waves, profile, root_pose_w, root_vel_w, env_origins_w, env_ids):
    """[k,13] root state (pos, quat wxyz, lin vel, ang vel; world) of the rows ``env_ids`` on the waves.

    ``waves`` is the layer's SpectralWaves ([N,M] rows), ``profile`` the runtime's vessel profile (its hydro
    spec gives the footprint and the orbital depth; its actor_to_hull basis must be the identity, as for
    every v2 asset), ``root_pose_w`` [N,7] / ``root_vel_w`` [N,6] the spawned state (calm height, level).
    """
    hv2 = _hydro_v2()
    basis = torch.as_tensor(profile.actor_to_hull, dtype=torch.float64)
    if not torch.equal(basis, torch.eye(3, dtype=torch.float64)):
        raise ValueError("wave_spawn needs an actor frame equal to the hull frame (v2 assets)")
    spec = profile.hydro
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=root_pose_w.device).flatten()
    dtype = waves.amplitude.dtype
    rows = hv2.SpectralWaves(*(getattr(waves, name)[ids] if getattr(waves, name).ndim == 2 else getattr(waves, name)
                               for name in ("amplitude", "omega", "k", "direction", "phase")))
    pose = root_pose_w[ids].to(dtype)
    vel = root_vel_w[ids].to(dtype)
    origins = env_origins_w[ids].to(dtype)
    w, x, y, z = pose[:, 3:7].unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    sample = rows.sample(pose[:, :2], origins[:, :2], torch.zeros((), dtype=dtype, device=pose.device), yaw, spec,
                         footprint=True)
    kx, ky = rows.k * torch.cos(rows.direction), rows.k * torch.sin(rows.direction)
    local = pose[:, :2] - origins[:, :2]
    theta = kx * local[:, 0:1] + ky * local[:, 1:2] + rows.phase              # t = 0
    transfer = hv2.footprint_transfer(kx, ky, yaw, spec.length_m, spec.beam_m)
    heave_rate = (transfer * rows.amplitude * rows.omega * torch.sin(theta)).sum(-1)
    roll, pitch = normal_attitude(hv2.surface_normal(sample.slope), yaw)
    out = torch.empty((ids.numel(), 13), dtype=root_pose_w.dtype, device=root_pose_w.device)
    out[:, 0:2] = root_pose_w[ids, 0:2]
    out[:, 2] = (pose[:, 2] + sample.elevation).to(out.dtype)
    out[:, 3:7] = _quat_from_euler(roll, pitch, yaw).to(out.dtype)
    lin = vel[:, :3].clone()
    lin[:, 0:2] = lin[:, 0:2] + sample.orbital_velocity_world[:, 0:2]
    lin[:, 2] = lin[:, 2] + heave_rate
    out[:, 7:10] = lin.to(out.dtype)
    out[:, 10:13] = (vel[:, 3:6] + sample.surface_angular_velocity_world).to(out.dtype)
    if not bool(torch.isfinite(out).all()):
        raise ValueError("non-finite wave-following spawn state")
    return out


__all__ = ["SPAWN_VERSION", "SPAWN_RULE", "JOURNAL_SPAWN_ON_WAVES_DEFAULT", "SPAWN_DEFAULT_DECISION", "enabled",
           "stamp", "normal_attitude", "wave_following_root_state"]
