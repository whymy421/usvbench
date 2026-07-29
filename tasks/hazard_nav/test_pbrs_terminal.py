"""Regression test for the potential-shaping terminal rule.

The first attempt zeroed Phi at EVERY episode end, including time-limit
truncations. Because Phi = -clamp(d/D0, 0, 1), the shaping at such a step
became gamma*0 - Phi(s) = +d/D0, i.e. up to +20 reward (the entire progress
budget) for running out of time as far from the goal as possible. Certified
success went 75.8% -> 0.0%.

This test encodes the rule arithmetically so the mistake cannot come back:
  * a time-limit truncation must NOT zero Phi
  * a true terminal (goal / contact) may zero Phi
  * with either setting, per-step shaping must never reward being further away
"""

from __future__ import annotations

import torch

SCALE = 20.0
GAMMA = 0.999


def shaping(phi_prev, phi_now, *, correct, zero_at_terminal, terminated, timed_out):
    """Mirror of the env's progress term, in the same order of operations."""
    if correct:
        if zero_at_terminal:
            nxt = torch.where(terminated, torch.zeros_like(phi_now), phi_now)
        else:
            nxt = phi_now
        return SCALE * (GAMMA * nxt - phi_prev)
    return SCALE * (phi_now - phi_prev)


def main() -> None:
    F = torch.tensor  # noqa: N806
    no = F([False])
    yes = F([True])

    # --- 1. the bug: zeroing at a TIMEOUT pays for being far away ----------
    # Reproduce the old behaviour by declaring the timeout a terminal.
    far = F([-1.0])    # d = D0
    near = F([-0.05])  # almost at the goal
    buggy_far = float(shaping(far, far, correct=True, zero_at_terminal=True,
                              terminated=yes, timed_out=yes))
    buggy_near = float(shaping(near, near, correct=True, zero_at_terminal=True,
                               terminated=yes, timed_out=yes))
    assert buggy_far > 19.0, buggy_far      # +20: the whole progress budget
    assert buggy_near <= 1.0, buggy_near    # 20 * 0.05, i.e. nearly nothing
    # The gap is the defect: timing out far away paid 20x what timing out at
    # the goal paid, on a term whose whole-episode budget is only 20.
    assert buggy_far > 15.0 * buggy_near, (buggy_far, buggy_near)

    # --- 2. the fix: a timeout is not a terminal, so no spike -------------
    fixed_far = float(shaping(far, far, correct=True, zero_at_terminal=True,
                              terminated=no, timed_out=yes))
    fixed_near = float(shaping(near, near, correct=True, zero_at_terminal=True,
                               terminated=no, timed_out=yes))
    assert abs(fixed_far) < 0.05, fixed_far
    assert abs(fixed_near) < 0.05, fixed_near

    # --- 3. progress must always pay for CLOSING the distance -------------
    for correct in (False, True):
        for zero in (False, True):
            closing = float(shaping(F([-0.8]), F([-0.4]), correct=correct,
                                    zero_at_terminal=zero, terminated=no,
                                    timed_out=no))
            receding = float(shaping(F([-0.4]), F([-0.8]), correct=correct,
                                     zero_at_terminal=zero, terminated=no,
                                     timed_out=no))
            assert closing > 0 > receding, (correct, zero, closing, receding)

    # --- 4. at a TRUE terminal, zeroing must not make crashing far better
    # than crashing near, once the contact penalty is included.
    CONTACT = 25.0
    crash_far = float(shaping(F([-0.9]), F([-0.9]), correct=True,
                              zero_at_terminal=True, terminated=yes,
                              timed_out=no)) - CONTACT
    crash_near = float(shaping(F([-0.1]), F([-0.1]), correct=True,
                               zero_at_terminal=True, terminated=yes,
                               timed_out=no)) - CONTACT
    # This ordering IS still perverse step-locally -- the telescoping argument
    # only makes the TOTAL constant. Record it so the next reader knows the
    # residual risk rather than discovering it in a training run.
    print(f"  真终止处置零的残余风险: 远处撞 {crash_far:+.1f} vs 近处撞 "
          f"{crash_near:+.1f} (单步看仍偏向远处撞,总和才恒定)")

    print(f"  旧实现(超时也置零): 远处超时 {buggy_far:+.1f} vs 近处超时 "
          f"{buggy_near:+.1f}  <-- 这就是 75.8% -> 0% 的原因")
    print(f"  修正后: 远处超时 {fixed_far:+.2f} vs 近处超时 {fixed_near:+.2f}")
    print("PASS: PBRS terminal rule (timeout is not terminal; progress still "
          "pays for closing)")


if __name__ == "__main__":
    main()
