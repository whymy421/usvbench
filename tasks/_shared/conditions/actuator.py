"""Scalar actuator conditions, extracted from hazard_nav's six legacy axes.

The caller converts the bounded action to physical surge force / yaw torque.
``command`` then runs ONCE PER PHYSICS STEP, including decimated steps. Drag
scaling belongs on planar and yaw drag only, before ``dynamics`` receives the
complete body-frame external wrench. The payload approximation deliberately
matches the legacy external-wrench model; it does not edit the USD inertia.
"""

from __future__ import annotations

import math

import torch


class ActuatorCondition:
    """Six independently composable scalar axes with exact neutral branches.

    Config attributes: enabled, mass_scale, drag_scale, thrust_cap_scale,
    motor_tau_s, thrust_imbalance, payload_mass_kg, payload_offset_x_m,
    payload_offset_y_m. Config validation is owned by conditions.config.
    """

    def __init__(self, cfg, num_envs: int, device: str | torch.device) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._applied_thrust = torch.zeros(self.num_envs, device=self.device)
        self._applied_yaw = torch.zeros(self.num_envs, device=self.device)

    def action_in(self, actions: torch.Tensor) -> torch.Tensor:
        """Preserve the action object; physical command filtering is per tick.

        Action clipping and conversion remain the task's authored contract.
        Filtering here would incorrectly advance lag once per control step.
        """
        return actions

    def command(
        self,
        thrust: torch.Tensor,
        yaw: torch.Tensor,
        dt: float,
        half_beam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cap, differential thrust imbalance, then backward Euler lag."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("physics dt must be finite and positive")
        if not math.isfinite(half_beam) or half_beam <= 0.0:
            raise ValueError("half_beam must be finite and positive")
        if not self.cfg.enabled:
            return thrust, yaw
        if thrust.shape != (self.num_envs,) or yaw.shape != (self.num_envs,):
            raise ValueError("thrust and yaw must have shape (num_envs,)")
        if self.cfg.thrust_cap_scale != 1.0:
            thrust = thrust * self.cfg.thrust_cap_scale
        if self.cfg.thrust_imbalance != 0.0:
            # Preserve the legacy operation order, including its lever floor.
            lever = max(float(half_beam), 1.0e-6)
            half_yaw = yaw / lever
            t_stbd = 0.5 * (thrust + half_yaw)
            t_port = 0.5 * (thrust - half_yaw)
            t_port = t_port * (1.0 - self.cfg.thrust_imbalance)
            t_stbd = t_stbd * (1.0 + self.cfg.thrust_imbalance)
            thrust = t_port + t_stbd
            yaw = (t_stbd - t_port) * lever
        if self.cfg.motor_tau_s != 0.0:
            alpha = dt / (self.cfg.motor_tau_s + dt)
            thrust = self._applied_thrust + (
                thrust - self._applied_thrust
            ) * alpha
            yaw = self._applied_yaw + (yaw - self._applied_yaw) * alpha
            self._applied_thrust.copy_(thrust)
            self._applied_yaw.copy_(yaw)
        return thrust, yaw

    def scale_drag(self, value: torch.Tensor) -> torch.Tensor:
        """Scale task-authored planar/yaw drag, preserving the neutral object."""
        if not self.cfg.enabled or self.cfg.drag_scale == 1.0:
            return value
        return value * self.cfg.drag_scale

    def dynamics(
        self,
        force_body: torch.Tensor,
        torque_body: torch.Tensor,
        reference_mass_kg: float,
        reference_izz_kgm2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply payload COM/mass/yaw inertia, then common mass scaling.

        Inputs are (num_envs, num_bodies, 3); the authored hull is body zero.
        Payload shifts the hull's planar COM arm and yaw inertia only. All
        external force components receive its mass ratio; all external force
        and torque components receive mass_scale. Inputs are never mutated.
        """
        if not self.cfg.enabled:
            return force_body, torque_body
        if (
            not math.isfinite(reference_mass_kg)
            or reference_mass_kg <= 0.0
            or not math.isfinite(reference_izz_kgm2)
            or reference_izz_kgm2 <= 0.0
        ):
            raise ValueError("reference mass and yaw inertia must be positive")
        if (
            force_body.ndim != 3
            or force_body.shape[0] != self.num_envs
            or force_body.shape[1] < 1
            or force_body.shape[2] != 3
            or torque_body.shape != force_body.shape
        ):
            raise ValueError("body wrenches must have shape (num_envs, bodies, 3)")
        forces, torques = force_body, torque_body
        if self.cfg.payload_mass_kg != 0.0:
            # Tensor arithmetic matches the scalar legacy branch exactly.
            payload_mass = torch.full(
                (self.num_envs,), float(self.cfg.payload_mass_kg),
                device=self.device,
            )
            hull_scale = torch.full(
                (self.num_envs,), float(self.cfg.mass_scale),
                device=self.device,
            )
            hull_mass = hull_scale * float(reference_mass_kg)
            hull_izz = hull_scale * float(reference_izz_kgm2)
            total_mass = hull_mass + payload_mass
            offset_x = float(self.cfg.payload_offset_x_m)
            offset_y = float(self.cfg.payload_offset_y_m)
            delta_x = payload_mass * offset_x / total_mass
            delta_y = payload_mass * offset_y / total_mass
            reduced_mass = hull_mass * payload_mass / total_mass
            offset_sq = offset_x * offset_x + offset_y * offset_y
            mass_ratio = total_mass / hull_mass
            izz_ratio = (hull_izz + reduced_mass * offset_sq) / hull_izz
            arm_torque = delta_y * forces[:, 0, 0] - delta_x * forces[:, 0, 1]
            loaded_torque_z = (torques[:, 0, 2] + arm_torque) / izz_ratio
            torques = torques.clone()
            torques[:, 0, 2] = loaded_torque_z
            forces = forces / mass_ratio.unsqueeze(-1).unsqueeze(-1)
        if self.cfg.mass_scale != 1.0:
            forces = forces / self.cfg.mass_scale
            torques = torques / self.cfg.mass_scale
        return forces, torques

    def hull_damping_scales(
        self, reference_mass_kg: float, reference_izz_kgm2: float,
    ) -> dict[str, torch.Tensor]:
        """Physics-v2 implicit-damping scales: the ONE drag/mass source (D1).

        In physics v2 the env applies no explicit hull drag, so ``scale_drag``
        is never called; ``drag_scale`` below multiplies the planar and yaw
        damping coefficients exactly once inside the implicit solve. The
        divisors reproduce the acceleration factors that ``dynamics`` applies
        to every other external load: linear ``mass_scale * mass_ratio``,
        yaw ``mass_scale * izz_ratio``, roll/pitch ``mass_scale`` (same
        ratio formulas as ``dynamics``). Disabled is exactly neutral.
        """
        ones = torch.ones(self.num_envs, device=self.device)
        if not self.cfg.enabled:
            return {"drag_scale": ones, "mass_divisor": ones.clone(),
                    "yaw_divisor": ones.clone(), "rollpitch_divisor": ones.clone()}
        if (
            not math.isfinite(reference_mass_kg)
            or reference_mass_kg <= 0.0
            or not math.isfinite(reference_izz_kgm2)
            or reference_izz_kgm2 <= 0.0
        ):
            raise ValueError("reference mass and yaw inertia must be positive")
        scale = float(self.cfg.mass_scale)
        mass_divisor = ones * scale
        yaw_divisor = ones * scale
        if self.cfg.payload_mass_kg != 0.0:
            payload_mass = torch.full(
                (self.num_envs,), float(self.cfg.payload_mass_kg), device=self.device
            )
            hull_scale = torch.full((self.num_envs,), scale, device=self.device)
            hull_mass = hull_scale * float(reference_mass_kg)
            hull_izz = hull_scale * float(reference_izz_kgm2)
            total_mass = hull_mass + payload_mass
            offset_x = float(self.cfg.payload_offset_x_m)
            offset_y = float(self.cfg.payload_offset_y_m)
            reduced_mass = hull_mass * payload_mass / total_mass
            offset_sq = offset_x * offset_x + offset_y * offset_y
            mass_ratio = total_mass / hull_mass
            izz_ratio = (hull_izz + reduced_mass * offset_sq) / hull_izz
            mass_divisor = mass_ratio * scale
            yaw_divisor = izz_ratio * scale
        return {"drag_scale": ones * float(self.cfg.drag_scale), "mass_divisor": mass_divisor,
                "yaw_divisor": yaw_divisor, "rollpitch_divisor": ones * scale}

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset only the requested environments' motor state."""
        self._applied_thrust[env_ids] = 0.0
        self._applied_yaw[env_ids] = 0.0


__all__ = ["ActuatorCondition"]
