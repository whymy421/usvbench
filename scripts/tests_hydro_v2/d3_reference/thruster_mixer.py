"""Physical two-motor allocation and reproducible actuator faults, without Isaac.

All vectors end in (port, starboard). The returned wrench is about the COM in
the hull frame: x forward, y port, z up. Both motors push along x at y=+/-arm
and at height mount_z_m relative to the COM; therefore X=P+S, N=arm*(S-P) and
the pitch moment is M=mount_z_m*(P+S) (r x F with r=(x,+/-arm,z), F=(T,0,0)).
Thrust-axis tilt and steering (azimuth) are not modelled. An adapter must
rotate this hull-frame wrench to a differently authored actor frame; never
reinterpret port/starboard from an asset's axis names.

No vessel numbers live in this module: every limit, arm, height and lag comes
from a ThrusterSpec built by the vessel profile (see hydro_profiles.py).

The policy asks for nominal surge and yaw independently. Inverse allocation
can exceed the physical envelope. Default radial saturation scales BOTH motor
requests by one common number, retaining the requested wrench direction before
fault gains. Optional clipping prioritizes each motor independently, so it can
change that direction. Gains are applied AFTER allocation: a one-sided fault
loses thrust without secretly boosting or compensating the healthy motor.

The nominal actuator lag and C7's additional lag are separate backward-Euler
stages, not a single time constant equal to their sum. Zero nominal lag is a
declared uncalibrated default; supply a measured vessel value when available.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


@dataclass(frozen=True)
class ThrusterSpec:
    """Vessel-owned physical values; reverse limits are positive magnitudes.

    The vessel profile (hydro_profiles.py, later D5) supplies per-motor
    forward and reverse limits, the transverse arm, the thrust-line height
    relative to the COM (negative = below the COM), the policy yaw request
    scale and nominal step-response poles. Shared/mirrored mounting arm is the
    supported two-thruster geometry.
    """

    forward_limit_n: tuple[float, float]
    reverse_limit_n: tuple[float, float]
    arm_m: float
    yaw_command_limit_nm: float
    nominal_lag_s: tuple[float, float] = (0.0, 0.0)
    mount_z_m: float = 0.0

    def __post_init__(self):
        for name in ("forward_limit_n", "reverse_limit_n", "nominal_lag_s"):
            pair = getattr(self, name)
            if len(pair) != 2 or any(not math.isfinite(float(x)) for x in pair):
                raise ValueError(name + " must contain two finite values")
            if any(x < 0 for x in pair):
                raise ValueError(name + " must be nonnegative")
        if any(x <= 0 for x in self.forward_limit_n + self.reverse_limit_n):
            raise ValueError("nominal motor limits must be positive")
        for name in ("arm_m", "yaw_command_limit_nm"):
            if not math.isfinite(float(getattr(self, name))) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive and finite")
        if not math.isfinite(float(self.mount_z_m)):
            raise ValueError("mount_z_m must be finite")


@dataclass(frozen=True)
class ThrusterDegradation:
    """Faults held fixed per episode, or explicitly reconfigured with clamping.

    cap: available fraction of both forward/reverse limits, in [0,1]. A cap
    leaves small commands unchanged; it is not a throttle gain. motor_gain:
    multiplicative delivered-thrust loss, in [0,1], applied to all commands.
    additional_lag_s: extra cascade pole, in seconds, >=0. Parameters accept a
    scalar, shared (2,) motor pair, or batch_shape+(1|2,) tensor. Use (N,1) for
    per-environment scalars to avoid ambiguity with the two-motor dimension.
    """

    thrust_cap: object = 1.0
    motor_gain: object = (1.0, 1.0)
    additional_lag_s: object = 0.0


@dataclass(frozen=True)
class ThrusterState:
    nominal: torch.Tensor
    extra: torch.Tensor


@dataclass(frozen=True)
class ThrusterOutput:
    body_wrench: torch.Tensor
    commanded_thrust: torch.Tensor
    delivered_thrust: torch.Tensor
    state: ThrusterState
    saturation_scale: torch.Tensor
    requested_thrust: torch.Tensor


def zero_state(actions: torch.Tensor) -> ThrusterState:
    """Zero BOTH stages at episode reset, retaining shape/device/precision."""
    _validate_actions(actions)
    return ThrusterState(torch.zeros_like(actions), torch.zeros_like(actions))


def _validate_actions(actions):
    if not isinstance(actions, torch.Tensor) or not actions.is_floating_point():
        raise ValueError("actions must be a floating-point torch tensor")
    if actions.ndim < 1 or actions.shape[-1] != 2:
        raise ValueError("actions must end in (surge, yaw)")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("actions must be finite")
    if bool((actions.abs() > 1).any()):
        raise ValueError("policy actions must lie in [-1,1]; clamp at the policy boundary")


def _parameter(value, like, name, upper=None):
    value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    try:
        value = torch.broadcast_to(value, like.shape)
    except RuntimeError as exc:
        raise ValueError(name + " must broadcast to batch_shape+(2,)") from exc
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(name + " must be finite and nonnegative")
    if upper is not None and bool((value > upper).any()):
        raise ValueError(name + " must not exceed " + str(upper))
    return value


def motor_wrench(thrust: torch.Tensor, spec: ThrusterSpec) -> torch.Tensor:
    """Convert physical motor forces to [Fx,Fy,Fz,Tx,Ty,Tz] about the COM.

    A zero mount_z_m yields an exact zero pitch component (0*x == 0).
    """
    port, starboard = thrust.unbind(-1)
    zero = torch.zeros_like(port)
    total = port + starboard
    return torch.stack((total, zero, zero, zero, spec.mount_z_m * total,
                        spec.arm_m * (starboard - port)), dim=-1)


def step_thrusters(actions: torch.Tensor, state: ThrusterState,
                   spec: ThrusterSpec, dt: float,
                   degradation: ThrusterDegradation | None = None,
                   saturation: Literal["radial", "clip"] = "radial") -> ThrusterOutput:
    """Allocate one action and advance both per-motor poles by one physics dt.

    For an action (a_X,a_N), X=a_X*sum(F_forward) when a_X>=0 and
    X=a_X*sum(F_reverse) otherwise; N=a_N*yaw_command_limit_nm. Inverse:
    P=(X-N/arm)/2, S=(X+N/arm)/2. For radial saturation let lambda be the
    minimum of 1 and each sign-appropriate capped limit / |requested motor|;
    both commands are scaled by lambda. A zero request imposes no constraint.

    Gains then reduce the scaled commands. Each pole uses the convex update
    y_next=(tau*y_old+dt*target)/(tau+dt); tau=0 is EXACT bypass. Every stage
    and target is bounded by the cap*gain envelope, even if a cap/gain changes
    abruptly. This represents instantaneous authority loss with no stored
    over-limit thrust. State outside the envelope is projected into it before
    stepping. No state is mutated and no global RNG is consumed.

    saturation_scale is the common lambda for radial mode; for clip it is
    the smaller motor retention ratio (diagnostic only, not a common scale).
    requested_thrust is the infeasible diagnostic request, NOT a command.
    """
    _validate_actions(actions)
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("physics dt must be positive and finite")
    if saturation not in ("radial", "clip"):
        raise ValueError("saturation must be 'radial' or 'clip'")
    for tensor in (state.nominal, state.extra):
        if (tensor.shape != actions.shape or tensor.device != actions.device
                or tensor.dtype != actions.dtype or not bool(torch.isfinite(tensor).all())):
            raise ValueError("state stages must match finite actions in shape, dtype and device")
    degradation = degradation or ThrusterDegradation()
    forward = _parameter(spec.forward_limit_n, actions, "forward_limit_n")
    reverse = _parameter(spec.reverse_limit_n, actions, "reverse_limit_n")
    cap = _parameter(degradation.thrust_cap, actions, "thrust_cap", 1.0)
    gain = _parameter(degradation.motor_gain, actions, "motor_gain", 1.0)
    nominal_tau = _parameter(spec.nominal_lag_s, actions, "nominal_lag_s")
    extra_tau = _parameter(degradation.additional_lag_s, actions, "additional_lag_s")

    surge_action, yaw_action = actions.unbind(-1)
    surge = surge_action * torch.where(surge_action >= 0,
                                       forward.sum(-1), reverse.sum(-1))
    differential = yaw_action * (spec.yaw_command_limit_nm / spec.arm_m)
    requested = torch.stack(((surge - differential) / 2,
                             (surge + differential) / 2), dim=-1)
    positive_cap, negative_cap = forward * cap, reverse * cap
    limits = torch.where(requested >= 0, positive_cap, negative_cap)
    # Guard the division before where: eager torch evaluates both branches.
    nonzero = requested != 0
    ratio = torch.where(nonzero, limits / torch.where(nonzero, requested.abs(),
                                                     torch.ones_like(requested)),
                        torch.ones_like(requested)).clamp(max=1)
    scale = ratio.amin(-1)
    if saturation == "radial":
        limited = requested * scale.unsqueeze(-1)
    else:
        limited = torch.minimum(torch.maximum(requested, -negative_cap), positive_cap)
    lower, upper = -negative_cap * gain, positive_cap * gain

    def bounded(value):
        return torch.minimum(torch.maximum(value, lower), upper)

    target = bounded(limited * gain)
    nominal = bounded(torch.where(nominal_tau == 0, target,
                      (nominal_tau * bounded(state.nominal) + dt * target)
                      / (nominal_tau + dt)))
    extra = bounded(torch.where(extra_tau == 0, nominal,
                    (extra_tau * bounded(state.extra) + dt * nominal)
                    / (extra_tau + dt)))
    next_state = ThrusterState(nominal, extra)
    return ThrusterOutput(motor_wrench(extra, spec), target, extra, next_state,
                          scale, requested)


def balanced_failure_sides(scenario_keys: torch.Tensor, balance_seed: int = 0):
    """Stable 0=port/1=starboard assignment, balanced in each paired key block.

    Keys must be nonnegative integer scenario ORDINALS shared across doses;
    (2*j,2*j+1) always receive opposite sides. A seeded integer permutation
    chooses each pair's orientation. This is deterministic under reordering,
    resets or repartitioning into environment batches, and consumes no RNG.
    Arbitrary subsets need not balance: freeze complete key pairs in the study
    manifest. Never pass an ephemeral vector-environment index as the key.
    """
    keys = torch.as_tensor(scenario_keys)
    if keys.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                          torch.uint8) or bool((keys < 0).any()):
        raise ValueError("scenario keys must be nonnegative integer ordinals")
    if not isinstance(balance_seed, int) or isinstance(balance_seed, bool):
        raise ValueError("balance_seed must be an integer")
    keys = keys.to(torch.int64)
    prime = 2147483647
    pair_hash = ((keys.div(2, rounding_mode="floor") % prime) * 48271
                 + balance_seed % prime) % prime
    return (pair_hash + keys % 2) % 2


def one_sided_motor_gains(loss_gain, scenario_keys: torch.Tensor,
                          balance_seed: int = 0, *, dtype=torch.float64):
    """C7 gain ladder: one motor in [0,1], other EXACTLY 1, balanced by key."""
    sides = balanced_failure_sides(scenario_keys, balance_seed)
    gains = torch.ones((*sides.shape, 2), dtype=dtype, device=sides.device)
    loss = torch.as_tensor(loss_gain, dtype=dtype, device=sides.device)
    if not bool(torch.isfinite(loss).all()) or bool(((loss < 0) | (loss > 1)).any()):
        raise ValueError("loss_gain must be finite and in [0,1]")
    loss = torch.broadcast_to(loss, sides.shape)
    return gains.scatter(-1, sides.unsqueeze(-1), loss.unsqueeze(-1))
