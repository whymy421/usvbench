"""Suite D dynamics axes: algebra and frozen packs, checked without Isaac.

Every modifier must have an exact legacy identity, a physically monotonic
non-default effect, and a pre-registered pack that this test can parse.

Run: python tasks/hazard_nav/test_thrust_imbalance.py
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np


HALF_BEAM_M = 0.45
THRUST_FWD_N = 80.0
THRUST_REV_N = 48.0
YAW_TORQUE_NM = 23.0
SURGE_LIN = 9.1
SURGE_QUAD = 5.9
SWAY_LIN = 27.0
SWAY_QUAD = 18.0
YAW_LIN = 4.0
YAW_QUAD = 6.0
PHYSICS_DT_S = 1.0 / 120.0


def split_recombine(thrust, yaw, imbalance, lever=HALF_BEAM_M):
    """Mirror of the env's _apply_action imbalance block."""
    half_yaw = yaw / lever
    t_stbd = 0.5 * (thrust + half_yaw)
    t_port = 0.5 * (thrust - half_yaw)
    t_port = t_port * (1.0 - imbalance)
    t_stbd = t_stbd * (1.0 + imbalance)
    return t_port + t_stbd, (t_stbd - t_port) * lever


def _commands_to_wrench(command: np.ndarray) -> np.ndarray:
    thrust = np.where(
        command[..., 0] >= 0.0,
        command[..., 0] * THRUST_FWD_N,
        command[..., 0] * THRUST_REV_N,
    )
    yaw = command[..., 1] * YAW_TORQUE_NM
    return np.stack((thrust, yaw), axis=-1)


def axis_free_plant(
    command: np.ndarray, velocity_body: np.ndarray, yaw_rate: np.ndarray
) -> np.ndarray:
    """Legacy planar applied (surge force, sway force, yaw torque)."""
    actuator = _commands_to_wrench(command)
    surge = velocity_body[..., 0]
    sway = velocity_body[..., 1]
    surge_drag = -(SURGE_LIN + SURGE_QUAD * np.abs(surge)) * surge
    sway_drag = -(SWAY_LIN + SWAY_QUAD * np.abs(sway)) * sway
    yaw_drag = -(YAW_LIN + YAW_QUAD * np.abs(yaw_rate)) * yaw_rate
    return np.stack(
        (
            actuator[..., 0] + surge_drag,
            sway_drag,
            actuator[..., 1] + yaw_drag,
        ),
        axis=-1,
    )


def modified_plant(
    command: np.ndarray,
    velocity_body: np.ndarray,
    yaw_rate: np.ndarray,
    actuator_prev: np.ndarray,
    *,
    mass_scale: float = 1.0,
    drag_scale: float = 1.0,
    thrust_cap_scale: float = 1.0,
    motor_tau_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy mirror of the four Suite D branches in _apply_action."""
    actuator = _commands_to_wrench(command)
    if thrust_cap_scale != 1.0:
        actuator[..., 0] = actuator[..., 0] * thrust_cap_scale
    if motor_tau_s != 0.0:
        alpha = PHYSICS_DT_S / (motor_tau_s + PHYSICS_DT_S)
        actuator = actuator_prev + (actuator - actuator_prev) * alpha

    surge = velocity_body[..., 0]
    sway = velocity_body[..., 1]
    drag = np.stack(
        (
            -(SURGE_LIN + SURGE_QUAD * np.abs(surge)) * surge,
            -(SWAY_LIN + SWAY_QUAD * np.abs(sway)) * sway,
            -(YAW_LIN + YAW_QUAD * np.abs(yaw_rate)) * yaw_rate,
        ),
        axis=-1,
    )
    if drag_scale != 1.0:
        drag = drag * drag_scale
    wrench = np.stack(
        (
            actuator[..., 0] + drag[..., 0],
            drag[..., 1],
            actuator[..., 1] + drag[..., 2],
        ),
        axis=-1,
    )
    if mass_scale != 1.0:
        wrench = wrench / mass_scale
    return wrench, actuator


def assert_default_identities() -> None:
    """Compare 200 random trajectories using exact, not tolerant, equality."""
    rng = np.random.default_rng(20260806)
    sequence_count = 200
    sequence_steps = 32
    commands = rng.uniform(-1.0, 1.0, (sequence_count, sequence_steps, 2))
    velocities = rng.uniform(-3.0, 3.0, (sequence_count, sequence_steps, 2))
    yaw_rates = rng.uniform(-2.0, 2.0, (sequence_count, sequence_steps))

    legacy = axis_free_plant(commands, velocities, yaw_rates)
    axes = {
        "mass_scale=1.0": {"mass_scale": 1.0},
        "drag_scale=1.0": {"drag_scale": 1.0},
        "thrust_cap_scale=1.0": {"thrust_cap_scale": 1.0},
        "motor_tau_s=0.0": {"motor_tau_s": 0.0},
    }
    for name, override in axes.items():
        actual = np.empty_like(legacy)
        for sequence in range(sequence_count):
            actuator_prev = np.zeros(2)
            for step in range(sequence_steps):
                wrench, actuator_prev = modified_plant(
                    commands[sequence, step],
                    velocities[sequence, step],
                    yaw_rates[sequence, step],
                    actuator_prev,
                    **override,
                )
                actual[sequence, step] = wrench
        assert np.array_equal(actual, legacy), name
        print(f"   {name:<24} PASS (200 x {sequence_steps} exact samples)")


def steady_peak_speed(thrust_cap_scale: float, duration_s: float = 60.0) -> float:
    """Integrate the registry surge plant; return its peak forward speed."""
    # The USD authors the actual mass. The displaced-water value supplies a
    # conservative standalone acceleration scale; ordering is independent of
    # that choice and the production simulator remains authoritative.
    effective_mass_kg = 34.6
    speed = 0.0
    peak = 0.0
    for _ in range(round(duration_s / PHYSICS_DT_S)):
        drag = (SURGE_LIN + SURGE_QUAD * abs(speed)) * speed
        acceleration = (THRUST_FWD_N * thrust_cap_scale - drag) / effective_mass_kg
        speed += acceleration * PHYSICS_DT_S
        peak = max(peak, speed)
    return peak


def response_time_63(motor_tau_s: float) -> float:
    """Measure elapsed substep time until a unit step reaches 1-exp(-1)."""
    if motor_tau_s == 0.0:
        return 0.0
    target = 1.0 - np.exp(-1.0)
    applied = 0.0
    alpha = PHYSICS_DT_S / (motor_tau_s + PHYSICS_DT_S)
    for step in range(1, 1_000_001):
        applied = applied + (1.0 - applied) * alpha
        if applied >= target:
            return step * PHYSICS_DT_S
    raise AssertionError(f"63% response not reached for tau={motor_tau_s}")


def assert_monotonic_effects() -> None:
    command = np.array([1.0, 0.5])
    velocity = np.zeros(2)
    yaw_rate = np.array(0.0)
    previous = np.zeros(2)

    nominal, _ = modified_plant(command, velocity, yaw_rate, previous)
    heavy, _ = modified_plant(
        command, velocity, yaw_rate, previous, mass_scale=1.30
    )
    nominal_accel_proxy = np.linalg.norm(nominal)
    heavy_accel_proxy = np.linalg.norm(heavy)
    assert heavy_accel_proxy < nominal_accel_proxy
    print(
        "   mass: applied-wrench acceleration proxy "
        f"{nominal_accel_proxy:.3f} -> {heavy_accel_proxy:.3f}"
    )

    moving = np.array([2.0, -1.0])
    turning = np.array(0.75)
    zero_command = np.zeros(2)
    nominal_drag, _ = modified_plant(
        zero_command, moving, turning, previous, drag_scale=1.0
    )
    doubled_drag, _ = modified_plant(
        zero_command, moving, turning, previous, drag_scale=2.0
    )
    assert np.linalg.norm(doubled_drag) > np.linalg.norm(nominal_drag)
    print(
        "   drag: opposing-wrench magnitude "
        f"{np.linalg.norm(nominal_drag):.3f} -> "
        f"{np.linalg.norm(doubled_drag):.3f}"
    )

    nominal_peak = steady_peak_speed(1.0)
    derated_peak = steady_peak_speed(0.50)
    assert derated_peak < nominal_peak
    print(
        f"   thrust cap: 60 s peak speed {nominal_peak:.3f} -> "
        f"{derated_peak:.3f} m/s"
    )

    taus = (0.0, 0.10, 0.25, 1.00)
    response_times = tuple(response_time_63(tau) for tau in taus)
    assert all(b > a for a, b in zip(response_times, response_times[1:]))
    print(f"   tau: 63% response delays {dict(zip(taus, response_times))}")


EXPECTED_PACKS = {
    "Mass scale": {
        "Train": (0.80, 1.00, 1.30),
        "Interpolation": (0.90, 1.10, 1.20),
        "Extrapolation": (1.75, 3.00, 5.00),
    },
    "Drag scale": {
        "Train": (0.75, 1.00, 1.50),
        "Interpolation": (0.875, 1.20, 1.35),
        "Extrapolation": (3.00, 8.00, 20.00),
    },
    "Thrust-cap scale": {
        "Train": (0.75, 0.875, 1.00),
        "Interpolation": (0.8125, 0.90, 0.95),
        "Extrapolation": (0.50, 0.20, 0.05),
    },
    "Motor time constant": {
        "Train": (0.00, 0.10, 0.25),
        "Interpolation": (0.05, 0.15, 0.20),
        "Extrapolation": (1.00, 3.00, 10.00),
    },
}


def parse_frozen_packs() -> dict[str, dict[str, tuple[float, ...]]]:
    path = Path(__file__).with_name("suite_d_packs_v3.md")
    text = path.read_text(encoding="utf-8")
    assert "Frozen 2026-08-06 BEFORE any v3 evaluation ran" in text
    sections = {}
    for chunk in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        title, body = chunk.split("\n", 1)
        if title not in EXPECTED_PACKS:
            continue
        values_by_pack = {}
        for pack_name in ("Train", "Interpolation", "Extrapolation"):
            match = re.search(
                rf"^- {pack_name}: `\{{([^}}]+)\}}`$", body, re.MULTILINE
            )
            assert match is not None, (title, pack_name)
            values_by_pack[pack_name] = tuple(
                float(value.strip()) for value in match.group(1).split(",")
            )
        sections[title] = values_by_pack
    return sections


def assert_frozen_packs_parse() -> None:
    parsed = parse_frozen_packs()
    assert parsed == EXPECTED_PACKS, (parsed, EXPECTED_PACKS)
    for axis, packs in parsed.items():
        print(f"   {axis:<20} PASS {packs}")


def assert_imbalance_axis() -> None:
    rng = np.random.default_rng(0)
    thrust = rng.uniform(-40.0, 80.0, size=4096)
    yaw = rng.uniform(-12.0, 12.0, size=4096)
    out_t, out_y = split_recombine(thrust, yaw, 0.0)
    assert np.max(np.abs(out_t - thrust)) < 1.0e-9
    assert np.max(np.abs(out_y - yaw)) < 1.0e-9

    for imbalance in (0.02, 0.05, 0.10, 0.20):
        _, out_yaw = split_recombine(
            np.array([60.0]), np.array([0.0]), imbalance
        )
        assert abs(float(out_yaw[0])) > 1.0e-6
    print("   imbalance=0 identity and nonzero veer PASS")


def main() -> None:
    print("1) exact default identities over random applied-wrench trajectories")
    assert_default_identities()
    print("\n2) monotonic non-default physical effects")
    assert_monotonic_effects()
    print("\n3) frozen v3 packs parse exactly")
    assert_frozen_packs_parse()
    print("\n4) shipped imbalance regression")
    assert_imbalance_axis()
    print("\n=> PASS")


if __name__ == "__main__":
    main()
