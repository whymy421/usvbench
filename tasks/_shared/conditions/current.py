"""Shared station/docking quadratic current law in WORLD coordinates."""
import math
import torch


def relative_drag(velocity_xy, current_xy, coefficient):
    """Exact legacy order: -c * (v-u) * norm(v-u)."""
    relative = velocity_xy - current_xy
    return -coefficient * relative * torch.norm(relative, dim=1, keepdim=True)


class CurrentCondition:
    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.direction = torch.zeros(num_envs, device=device)

    def reset(self, env_ids, scenario):
        if not self.cfg.enabled:
            return
        if self.cfg.direction_rad is None:
            self.direction[env_ids] = scenario.uniform(
                "condition_current", env_ids, 0.0, 2 * math.pi).to(self.direction.device)
        else:
            self.direction[env_ids] = self.cfg.direction_rad

    def speed(self, elapsed_s):
        c = self.cfg
        if c.speed_table:
            result = torch.full_like(elapsed_s, c.speed_table[0][1])
            for (start, lo), (end, hi) in zip(c.speed_table, c.speed_table[1:]):
                fraction = ((elapsed_s - start) / (end - start)).clamp(0, 1)
                result = torch.where(elapsed_s >= start, lo + (hi - lo) * fraction, result)
            return result
        if c.ramp_end_mps is not None:
            fraction = (elapsed_s / c.ramp_duration_s).clamp(0, 1)
            return c.speed_mps + (c.ramp_end_mps - c.speed_mps) * fraction
        return torch.full_like(elapsed_s, c.speed_mps)

    def velocity(self, elapsed_s):
        speed = self.speed(elapsed_s)
        return speed[:, None] * torch.stack((torch.cos(self.direction), torch.sin(self.direction)), dim=-1)

    def wrench(self, velocity_xy, elapsed_s):
        force = velocity_xy.new_zeros((len(velocity_xy), 3))
        torque = torch.zeros_like(force)
        if not self.cfg.enabled:
            return force, torque
        # Clock precision must not silently promote the extracted float32 law.
        current = self.velocity(elapsed_s.to(velocity_xy.dtype))
        value = relative_drag(velocity_xy, current, self.cfg.drag_coeff)
        if self.cfg.force_mode == "incremental_from_calm":
            value = value - relative_drag(velocity_xy, torch.zeros_like(current), self.cfg.drag_coeff)
        force[:, :2] = value
        return force, torque
