"""What does the shipped reward actually pay for threading a basin gate?

The basin changes the geometry the reward sees, not the reward itself -- but
that is exactly how the last two shaping attempts died. Both were correct on
paper and collapsed in training because nobody added up what the terms paid in
the situation the policy would actually be in (idling far from the goal earned
144 units against a progress budget of 20).

The worry here is concrete: the per-ray proximity field has a 1.35 m band, and
threading a 2-beam gate puts a wall inside that band on BOTH sides at once for
the whole transit. If the accumulated cost is comparable to the goal bonus, the
task pays the boat to refuse the gate and time out instead.

This walks a nominal transit through real generated layouts and applies the
env's own ``ray_circle_ranges`` and the exact proximity expression from
``_get_rewards``, so the number is measured rather than estimated.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from .hazard_geometry import (
        FORCED_GATE_BEAMS,
        HULL_BEAM_M,
        ray_circle_ranges,
        sample_forced_crossing_layout,
    )
except ImportError:  # direct execution
    from hazard_geometry import (  # type: ignore
        FORCED_GATE_BEAMS,
        HULL_BEAM_M,
        ray_circle_ranges,
        sample_forced_crossing_layout,
    )

# Mirrors HazardNavV3EnvCfg. Kept as literals so a cfg drift shows up as a
# disagreement here rather than being silently absorbed.
RAY_COUNT = 36
RAY_MAX_RANGE_M = 30.0
PROX_SCALE = 4.1
PROX_FLOOR_M = 0.45
SAFE_CLEARANCE_M = 0.90
HALF_BEAM_M = 0.45
CONTROL_STEP_S = (1.0 / 120.0) * 2  # sim.dt * decimation
GOAL_ENTRY_BONUS = 50.0
PROGRESS_BUDGET = 20.0
CRUISE_MPS = 1.0


def transit_prox_cost(layout, gate, *, samples: int = 400) -> tuple[float, float]:
    """Integrate the proximity cost along a straight run through the gate.

    Returns (total cost over the transit, worst single-step cost).
    """
    # Start -> gate centre -> goal, which is what a competent policy flies.
    waypoints = np.array([layout.start, gate.center, layout.goal])
    legs = np.diff(waypoints, axis=0)
    lengths = np.linalg.norm(legs, axis=1)
    total_len = float(lengths.sum())

    path = []
    for leg_index, leg in enumerate(legs):
        n = max(2, int(samples * lengths[leg_index] / total_len))
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            path.append(waypoints[leg_index] + t * leg)
    points = torch.tensor(np.array(path), dtype=torch.float64)

    # Heading along the path, so the ray fan is oriented the way it would be.
    headings = torch.tensor(
        np.array([legs[0] / lengths[0]] * len(path)), dtype=torch.float64
    )
    for i, p in enumerate(path):
        leg = legs[0] if p[0] <= gate.center[0] else legs[1]
        headings[i] = torch.tensor(leg / np.linalg.norm(leg))

    angles = torch.arange(RAY_COUNT, dtype=torch.float64) * (2 * np.pi / RAY_COUNT)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    hx = headings[:, 0:1]
    hy = headings[:, 1:2]
    directions = torch.stack(
        [hx * cos_a - hy * sin_a, hx * sin_a + hy * cos_a], dim=-1
    )

    centers = torch.tensor(layout.centers, dtype=torch.float64)
    radii = torch.tensor(layout.radii, dtype=torch.float64)
    ranges = ray_circle_ranges(
        points, directions, centers, radii, max_range_m=RAY_MAX_RANGE_M
    )

    cap = SAFE_CLEARANCE_M + HALF_BEAM_M
    per_step = (
        -PROX_SCALE
        * CONTROL_STEP_S
        * torch.log(ranges.clamp(PROX_FLOOR_M, cap) / cap).mean(dim=-1)
    )
    # Each sampled point stands for the distance between samples, walked at
    # CRUISE_MPS; convert the path integral into control steps.
    step_m = total_len / len(path)
    steps_per_sample = (step_m / CRUISE_MPS) / CONTROL_STEP_S
    return float(per_step.sum() * steps_per_sample), float(per_step.max())


def main() -> None:
    rng = np.random.default_rng(11)
    print("proximity cost of one nominal transit (start -> gate -> goal)")
    print(f"  band {SAFE_CLEARANCE_M + HALF_BEAM_M:.2f} m, scale {PROX_SCALE}, "
          f"{CONTROL_STEP_S * 1000:.1f} ms/step, cruise {CRUISE_MPS} m/s")
    print()
    worst_total = 0.0
    for level in (0, 1, 2, 3):
        totals, peaks = [], []
        for _ in range(12):
            layout, gate = sample_forced_crossing_layout(level, rng=rng)
            total, peak = transit_prox_cost(layout, gate)
            totals.append(total)
            peaks.append(peak)
        med = float(np.median(totals))
        worst_total = max(worst_total, float(np.max(totals)))
        print(f"tier {level} (gate {FORCED_GATE_BEAMS[level]:.0f} beams = "
              f"{FORCED_GATE_BEAMS[level] * HULL_BEAM_M:.2f} m)")
        print(f"  transit proximity cost  median {med:.2f}  max {max(totals):.2f}")
        print(f"  worst single step       {max(peaks):.4f}")
    print()
    print(f"worst transit cost over every tier: {worst_total:.2f}")
    print(f"  against goal entry bonus {GOAL_ENTRY_BONUS:.0f} "
          f"(+ up to 50 more for speed) and a progress budget of {PROGRESS_BUDGET:.0f}")
    if worst_total < 0.25 * PROGRESS_BUDGET:
        print("=> PASS: threading costs a small fraction of what arriving pays, so")
        print("   the reward does not pay the boat to refuse the gate and time out.")
    else:
        print("=> FAIL: the transit tax is large enough to compete with arriving.")
        print("   Band-limit the proximity field or widen the jamb radius before")
        print("   training anything on this task.")


if __name__ == "__main__":
    main()
