"""CPU-only tests for per-episode station-error instrumentation.

Run: python -m pytest tasks/station_keeping/test_station_error_metrics.py -v
"""

from __future__ import annotations

from pathlib import Path

import torch

try:
    from .station_error_metrics import StationErrorAccumulator
except ImportError:  # Direct execution from the repository root.
    from station_error_metrics import StationErrorAccumulator


_REPO = Path(__file__).resolve().parents[2]
_CERTIFICATE_FIELDS = (
    "station_error_rms_m",
    "station_error_p95_m",
    "station_error_final_60s_rms_m",
    "station_error_final_60s_p95_m",
)


def _accumulator(num_envs: int = 1) -> StationErrorAccumulator:
    return StationErrorAccumulator(
        num_envs=num_envs,
        max_episode_samples=240,
        control_step_s=1.0,
        device="cpu",
    )


def test_stationary_vessel_has_zero_error() -> None:
    metrics = _accumulator()
    for _ in range(120):
        metrics.record(torch.tensor([0.0]))

    metrics.latch(torch.tensor([0]))

    assert metrics.episode_rms_m.item() == 0.0
    assert metrics.episode_p95_m.item() == 0.0
    assert metrics.episode_final_60s_rms_m.item() == 0.0
    assert metrics.episode_final_60s_p95_m.item() == 0.0


def test_constant_displacement_is_reported_exactly() -> None:
    metrics = _accumulator()
    for _ in range(120):
        metrics.record(torch.tensor([3.25]))

    metrics.latch(torch.tensor([0]))

    for result in (
        metrics.episode_rms_m,
        metrics.episode_p95_m,
        metrics.episode_final_60s_rms_m,
        metrics.episode_final_60s_p95_m,
    ):
        assert result.item() == 3.25


def test_final_60_seconds_excludes_earlier_samples() -> None:
    metrics = _accumulator()
    for _ in range(60):
        metrics.record(torch.tensor([10.0]))
    for _ in range(60):
        metrics.record(torch.tensor([2.0]))

    metrics.latch(torch.tensor([0]))

    assert metrics.episode_rms_m.item() > 2.0
    assert metrics.episode_p95_m.item() == 10.0
    assert metrics.episode_final_60s_rms_m.item() == 2.0
    assert metrics.episode_final_60s_p95_m.item() == 2.0


def test_mid_episode_partial_reset_does_not_leak_samples() -> None:
    metrics = _accumulator(num_envs=2)
    metrics.record(torch.tensor([9.0, 2.0]))
    metrics.record(torch.tensor([9.0, 2.0]))

    metrics.reset(torch.tensor([0]))
    metrics.record(torch.tensor([3.0, 2.0]))
    metrics.record(torch.tensor([3.0, 2.0]))
    metrics.latch(torch.tensor([0, 1]))

    expected = torch.tensor([3.0, 2.0])
    assert torch.equal(metrics.episode_rms_m, expected)
    assert torch.equal(metrics.episode_p95_m, expected)
    assert torch.equal(metrics.episode_final_60s_rms_m, expected)
    assert torch.equal(metrics.episode_final_60s_p95_m, expected)


def test_p95_uses_linear_quantile_interpolation() -> None:
    metrics = _accumulator()
    for error in (0.0, 1.0, 2.0, 3.0, 4.0):
        metrics.record(torch.tensor([error]))

    metrics.latch(torch.tensor([0]))

    expected = torch.quantile(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]), 0.95)
    torch.testing.assert_close(metrics.episode_p95_m[0], expected)
    torch.testing.assert_close(metrics.episode_final_60s_p95_m[0], expected)


def test_station_evaluators_serialize_all_four_fields() -> None:
    env_source = (
        _REPO / "tasks" / "station_keeping" / "station_keeping_env.py"
    ).read_text(encoding="utf-8")
    for field in _CERTIFICATE_FIELDS:
        assert f"episode_{field}" in env_source

    for relative_path in (
        "scripts/eval_v6_frozen.py",
        "scripts/eval_obs_bridge.py",
        "scripts/eval_mppi.py",
        "scripts/classical_baseline.py",
    ):
        evaluator_source = (_REPO / relative_path).read_text(encoding="utf-8")
        for field in _CERTIFICATE_FIELDS:
            assert f'rec["{field}"]' in evaluator_source, (relative_path, field)


def test_environment_records_after_existing_done_logic_and_resets_per_env() -> None:
    source = (
        _REPO / "tasks" / "station_keeping" / "station_keeping_env.py"
    ).read_text(encoding="utf-8")
    dones = source[source.index("    def _get_dones"):source.index("    def _reset_idx")]
    reset = source[source.index("    def _reset_idx"):]

    assert dones.index("self._episode_finished.copy_(time_out)") < dones.index(
        "self._station_error.record"
    )
    assert reset.index("self._station_error.latch(completed_ids)") < reset.index(
        "self._station_error.reset(env_ids)"
    )
    assert reset.index("self.path_length[env_ids] = 0.0") < reset.index(
        "self._station_error.reset(env_ids)"
    )


def main() -> None:
    tests = (
        test_stationary_vessel_has_zero_error,
        test_constant_displacement_is_reported_exactly,
        test_final_60_seconds_excludes_earlier_samples,
        test_mid_episode_partial_reset_does_not_leak_samples,
        test_p95_uses_linear_quantile_interpolation,
        test_station_evaluators_serialize_all_four_fields,
        test_environment_records_after_existing_done_logic_and_resets_per_env,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} station-error tests passed")


if __name__ == "__main__":
    main()
