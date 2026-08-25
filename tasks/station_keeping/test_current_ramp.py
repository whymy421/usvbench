"""CPU unit test for the within-episode current-speed ramp.

Exercises the standalone modulation function with fabricated timestep
tensors -- no Isaac Lab, no gym registration, no env instantiation. The
asserted endpoint values 0.5 / 3.0 m/s mirror the defaults of
StationKeepingBlueBoatRampCurrentEnvCfg.

Run: python tasks/station_keeping/test_current_ramp.py
"""

from __future__ import annotations

import torch

try:
    from .current_ramp import ramped_current_vec
except ImportError:  # Direct execution from the repository root.
    from current_ramp import ramped_current_vec


# 120 s episode at the 60 Hz control rate, final-step normalizer as the env
# passes it: max_episode_length - 1.
MAX_STEPS = 7199
RAMP_START = 0.5
RAMP_END = 3.0


def _speeds(vec: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(vec, dim=-1)


def test_ramp_endpoints_and_midpoint() -> None:
    # Three fixed per-episode directions with arbitrary sampled magnitudes.
    base = torch.tensor([[2.3, 0.0], [0.0, -2.1], [1.0, 1.0]])
    for step_value, expected_speed in (
        (0, RAMP_START),
        (MAX_STEPS // 2, (RAMP_START + RAMP_END) / 2.0),
        (MAX_STEPS, RAMP_END),
    ):
        steps = torch.full((base.shape[0],), step_value, dtype=torch.long)
        ramped = ramped_current_vec(base, steps, MAX_STEPS, RAMP_START, RAMP_END)
        expected = torch.full((base.shape[0],), expected_speed)
        # MAX_STEPS is odd, so the integer midpoint sits half a step below
        # T/2; that offset is (END-START)/(2*T) in speed.
        tolerance = (RAMP_END - RAMP_START) / (2.0 * MAX_STEPS) + 1.0e-6
        assert torch.allclose(_speeds(ramped), expected, atol=tolerance), (
            step_value,
            _speeds(ramped),
        )
        # Direction is untouched: ramped vector is parallel to the base.
        base_dir = base / _speeds(base).unsqueeze(-1)
        ramp_dir = ramped / _speeds(ramped).unsqueeze(-1)
        assert torch.allclose(ramp_dir, base_dir, atol=1.0e-6), step_value


def test_ramp_exact_even_grid() -> None:
    # An even step count makes all three checkpoints exact.
    base = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    steps = torch.tensor([0, 300, 600], dtype=torch.long)
    ramped = ramped_current_vec(base, steps, 600, RAMP_START, RAMP_END)
    assert torch.allclose(
        _speeds(ramped), torch.tensor([0.5, 1.75, 3.0]), atol=1.0e-6
    ), _speeds(ramped)


def test_ramp_is_clamped_outside_the_episode() -> None:
    base = torch.tensor([[0.0, 4.0]])
    below = ramped_current_vec(
        base, torch.tensor([-5], dtype=torch.long), MAX_STEPS, RAMP_START, RAMP_END
    )
    beyond = ramped_current_vec(
        base,
        torch.tensor([MAX_STEPS + 500], dtype=torch.long),
        MAX_STEPS,
        RAMP_START,
        RAMP_END,
    )
    assert torch.allclose(_speeds(below), torch.tensor([RAMP_START]), atol=1.0e-6)
    assert torch.allclose(_speeds(beyond), torch.tensor([RAMP_END]), atol=1.0e-6)


def test_degenerate_and_invalid_inputs() -> None:
    # A zero base vector has no direction: output stays zero, never NaN.
    zero = ramped_current_vec(
        torch.zeros((2, 2)),
        torch.tensor([0, MAX_STEPS], dtype=torch.long),
        MAX_STEPS,
        RAMP_START,
        RAMP_END,
    )
    assert torch.equal(zero, torch.zeros((2, 2)))

    # A flat ramp (start == end) is a constant-speed current.
    flat = ramped_current_vec(
        torch.tensor([[3.0, 0.0]]),
        torch.tensor([1234], dtype=torch.long),
        MAX_STEPS,
        1.5,
        1.5,
    )
    assert torch.allclose(_speeds(flat), torch.tensor([1.5]), atol=1.0e-6)

    for bad_call in (
        lambda: ramped_current_vec(
            torch.ones((1, 2)), torch.tensor([0]), 0, RAMP_START, RAMP_END
        ),
        lambda: ramped_current_vec(
            torch.ones((1, 2)), torch.tensor([0]), MAX_STEPS, -0.1, RAMP_END
        ),
        lambda: ramped_current_vec(
            torch.ones((1, 2)), torch.tensor([0]), MAX_STEPS, RAMP_START, -1.0
        ),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def main() -> None:
    test_ramp_endpoints_and_midpoint()
    print("PASS ramp endpoints/midpoint at t=0, T/2, T (direction preserved)")
    test_ramp_exact_even_grid()
    print("PASS exact values on an even step grid: 0.5 / 1.75 / 3.0 m/s")
    test_ramp_is_clamped_outside_the_episode()
    print("PASS clamped to endpoint speeds outside [0, T]")
    test_degenerate_and_invalid_inputs()
    print("PASS zero-vector, flat-ramp, and invalid-argument handling")


if __name__ == "__main__":
    main()
