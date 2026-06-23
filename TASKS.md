# USVBench — Task Roadmap

Standardized calm-water navigation baselines for a **surface vehicle (boat)** and an
**underwater vehicle (ROV)** in Isaac Sim. This file tracks planned work and who owns it.

## Current baselines
Continuous 360° waypoint navigation, calm water, evaluated with `eval_benchmark.py`
(targets reached per 7200-step episode, mean over seeds 42/123/456):

| vehicle | targets/episode | OOB rate |
|---|---|---|
| ROV  | ~27 | 0% |
| boat | 3.24 ± 0.48 (distance-progress reward) | ~600% |

The boat's main remaining weakness is the high out-of-bounds rate (it still leaves the
arena ~6×/episode). Reward fix that got it from 1.05 → 3.24: potential-based
distance-progress shaping (`PROGRESS_COEF`, reward closing distance to target).

---

## A. Physics cleanup  — owner: Raina
These change the physics, so they invalidate current checkpoints → require re-train + re-eval.

- **A1 — Boat center of mass.** Measured COM is offset ~0.9 m laterally (should be on the
  centerline). A `COM_CENTER=1` env var already authors a centered COM at scene setup;
  decide whether to bake it into the USD. Re-train + re-eval the boat baseline on the
  corrected COM and compare targets/OOB.
- **A2 — Boat collision.** The boat collision is a triangle mesh, so PhysX falls back to a
  convex hull (startup warning; not a faithful hull). Author a proper approximation
  (`convexHull` / `convexDecomposition` / simplified primitive) in the boat USD; confirm
  the warning is gone; re-train + compare.

## B. Benchmark enrichment  — students
- **B1 — Wave navigation task (highest priority).** All current tasks are *calm*; they do
  not test wave-robust navigation. Add a navigation task under JONSWAP/Airy waves (the env
  already has a wave field, currently disabled for calm). Deliver ROV + boat baselines.
  Success: a `wave_nav` task + baseline numbers (mean±std) + a wave-robustness metric.
- **B2 — Obstacle avoidance (in waves).** Add static/dynamic obstacles; baseline + a
  collision metric.
- **B3 — Multi-target / coverage variants.**

  *Each task must ship ROV + boat baselines, a standardized eval, and a README entry.*

## C. Sim-to-sim validation (Gazebo)  — students
- **C1.** Export a trained Isaac Sim policy and replay it in Gazebo on an equivalent USV
  model. Compare trajectories/behavior across simulators to validate physics + policy
  transfer. Success: side-by-side Isaac-vs-Gazebo trajectory comparison + a transfer-gap
  metric.

---

## Conventions
- Train in the `isaaclab` conda env; keep wandb video logging on to inspect behaviour.
- Evaluate with `eval_benchmark.py`; report **mean ± std over seeds 42 / 123 / 456**.
- New vehicles are just geometry — they need rigid-body + collision + mass set up before
  use (see A1/A2 for the kind of issues that arise).
