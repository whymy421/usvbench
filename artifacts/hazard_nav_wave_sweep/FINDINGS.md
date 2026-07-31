# hazard_nav wave sweep — first run (2026-07-31)

Zero-shot evaluation of the calm-trained v11 champion
(`hazard_nav_v11_best_s42.pt`) under waves. Eval seed 42, level 1,
64 episodes x 64 envs, deterministic policy, `scripts/eval_hazard_nav.py`.

**The wave numbers below are not usable as a wave-robustness baseline yet.**
The failure mode is not dose-dependent, which a physical sensitivity would be.
See "Open question" — this needs one more experiment before the numbers mean
anything.

## Headline

| run | successes | SPL | path length | wave |
|---|---|---|---|---|
| calm (`Direct-v3`) | 21/64 | 0.2916 | 27.4 m | none |
| airy | 1/64 | 0.0087 | 120.5 m | H=0.12 m, T=3.0 s |
| jonswap | 1/64 | 0.0090 | 125.9 m | Hs 0.06-0.18 m, Tp 2-4 s |

The calm run reproduces `v11_best_level1_seed42.json` bit for bit
(0.291623609103239), so the wave code does not disturb the calm path.

## What was ruled out

**Buoyancy saturation.** The first attempt reused the rov/boat sea states
(Airy 0.5 m, Hs 0.3-1.0 m). BlueBoat's hull is 0.376 m tall, so the surface
only has to move +-0.188 m for the boat to leave the water entirely; those
settings saturated buoyancy 46% (airy) and 25% (jonswap) of the time. Sea
states were rescaled to the hull (saturation now 0.0%) and a startup check was
added that prints the saturated fraction and warns above 5%. **Rescaling did
not change the scores** (SPL 0.0092 -> 0.0087), so saturation was not the cause.

**Wave time not advancing.** Ruled out by direct instrumentation: t advances
one sim step (0.0083 s) per call and eta tracks it.

**Broken hull physics.** Also ruled out. Under waves the boat holds a steady
2.2-2.6 deg tilt, z stays within +-0.017 m, and speed holds 1.9 m/s. It does
not capsize, fly, or oscillate wildly.

**The `up_world` code path.** Feeding `restoring_torque_body` an exact
(0, 0, 1) — mathematically identical to omitting the argument — restores
21/64 and SPL 0.2916 exactly. The argument plumbing is sound.

## What the failure actually looks like

Distance to goal grows monotonically while the boat cruises at 1.9 m/s:

```
t=  5.0s  dist 32.4 m    t= 35.0s  dist  66.1 m
t= 15.0s  dist 40.6 m    t= 50.0s  dist  85.2 m
t= 30.0s  dist 59.4 m    t= 70.0s  dist 104.4 m
```

The boat is not being pushed around by waves; it is driving away from the goal
under power. Mean min-clearance rises from 0.26 m (calm) to 2.15 m, i.e. it
also stops threading the obstacle field.

## Open question — why this is not yet a result

Scaling the slope-torque channel gives a switch, not a dose response:

| `slope_torque_scale` | successes | SPL | path |
|---|---|---|---|
| 0.00 | 21/64 | 0.2916 | 27.4 m |
| 0.05 | 1/64 | 0.0090 | 125.8 m |
| 0.20 | 1/64 | 0.0087 | 120.8 m |
| 0.50 | 1/64 | 0.0087 | 120.6 m |
| 1.00 | 1/64 | 0.0087 | 120.5 m |

At 0.05 the commanded surface tilt is about 0.07 deg and the torque differs
from the calm case by 0.03-0.12 N m, yet the outcome is already identical to
full strength. A physical sensitivity would degrade gradually.

Two explanations remain, and they are not distinguished by anything measured
so far:

1. **The policy is the fragile part.** v11 was trained in perfectly flat water,
   spends 89% of its pre-goal steps in reverse, and passes obstacles at 0.26 m.
   Any persistent attitude disturbance may push it into a single failure
   attractor — which would explain why every non-zero scale lands on the same
   numbers. If so, this is a real and reportable wave-robustness result.
2. **The coupling has a positive feedback not yet found.** The measured tilt
   (2.5 deg, ~7.8 N m of restoring) is 20-35x larger than the 0.36 N m the
   0.05-scale slope excites directly, and the hull is heavily overdamped
   (zeta ~ 16 in pitch), so a quasi-static response of ~0.07 deg was expected.
   That gap is unexplained.

**Next experiment to separate them:** disturb the same checkpoint with a
constant small tilt that has nothing to do with waves. If it fails the same
way, explanation 1 holds and the wave numbers are real. If it survives,
the coupling is at fault.

## Files

`calm_s42.json`, `airy_s42.json`, `jonswap_s42.json` are the three headline
runs. `_slope0`, `_scan_*`, `_nullslope` are the diagnostic sweep above.
