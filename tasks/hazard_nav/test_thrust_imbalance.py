"""Suite D, thrust-imbalance axis: the algebra, checked without Isaac.

Two things have to hold before this axis can carry a benchmark result:

1. **Zero imbalance must be an exact identity.** The env rewrites the lumped
   (surge, yaw) command into port/starboard shares and back. If that round trip
   is not bit-exact at imbalance 0, every certified id silently changes the
   moment the field exists -- the same class of defect as the ring's inflated
   seal check, where the checker and the simulator disagreed about the hull.

2. **A non-zero imbalance must actually VEER the boat**, not merely slow it.
   Scaling the net force would make the task uniformly harder and would be a
   worse version of the mass axis. The point of this axis is that it breaks a
   symmetry: holding a straight line now costs continuous correction.

Run: python tasks/hazard_nav/test_thrust_imbalance.py
"""

from __future__ import annotations

import numpy as np

HALF_BEAM_M = 0.45


def split_recombine(thrust, yaw, imbalance, lever=HALF_BEAM_M):
    """Mirror of the env's _apply_action imbalance block."""
    half_yaw = yaw / lever
    t_stbd = 0.5 * (thrust + half_yaw)
    t_port = 0.5 * (thrust - half_yaw)
    t_port = t_port * (1.0 - imbalance)
    t_stbd = t_stbd * (1.0 + imbalance)
    return t_port + t_stbd, (t_stbd - t_port) * lever


def main() -> None:
    rng = np.random.default_rng(0)
    thrust = rng.uniform(-40.0, 80.0, size=4096)
    yaw = rng.uniform(-12.0, 12.0, size=4096)

    print("1) zero imbalance must be an exact identity")
    out_t, out_y = split_recombine(thrust, yaw, 0.0)
    dt = np.abs(out_t - thrust).max()
    dy = np.abs(out_y - yaw).max()
    print(f"   max |dthrust| = {dt:.3e}   max |dyaw| = {dy:.3e}")
    ok_identity = dt < 1e-9 and dy < 1e-9
    print(f"   {'PASS' if ok_identity else 'FAIL'}\n")

    print("2) a non-zero imbalance must VEER, not just slow down")
    print("   command: pure straight-ahead thrust, zero yaw")
    print(f"   {'imbalance':>10} {'net thrust':>12} {'net yaw (N m)':>14} {'veers?':>8}")
    ok_veer = True
    for imb in (0.0, 0.02, 0.05, 0.10, 0.20):
        t, y = split_recombine(np.array([60.0]), np.array([0.0]), imb)
        veer = abs(float(y)) > 1e-6
        if imb > 0 and not veer:
            ok_veer = False
        print(f"   {imb:>10.2f} {float(t):>12.2f} {float(y):>14.3f} {str(veer):>8}")
    print(f"   {'PASS' if ok_veer else 'FAIL'}\n")

    print("3) how hard is each pack? (yaw a straight-ahead command produces)")
    # Net thrust is unchanged by the imbalance under a pure-surge command --
    # the port loss and starboard gain cancel -- so the axis adds a turning
    # moment without changing available speed. That is what makes it clean.
    for name, vals in (("train", (0.00, 0.02, 0.04)),
                       ("interp", (0.01, 0.03, 0.05)),
                       ("extrap", (0.08, 0.12, 0.16))):
        ys = [abs(float(split_recombine(np.array([60.0]), np.array([0.0]), v)[1]))
              for v in vals]
        print(f"   {name:>7}: imbalance {vals} -> yaw {['%.2f' % y for y in ys]} N m")

    print()
    print("=> PASS" if (ok_identity and ok_veer) else "=> FAIL")
    print("   The packs above are FROZEN in suite_d_packs.md before any run.")


if __name__ == "__main__":
    main()
