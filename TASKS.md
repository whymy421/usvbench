# USVBench — Task Roadmap

Standardized calm-water navigation baselines for a **surface vehicle (boat)** and an
**underwater vehicle (ROV)** in Isaac Sim. This file tracks planned work and who owns it.

## Current baselines

**This table is the single source of truth for baseline numbers.** Every other document
in the repo links here instead of repeating figures. If you change physics or a reward,
retrain, re-measure, and update this table together with its date.

Continuous 360° point navigation, calm water. Measured on the checkpoints shipped in
this repo, which are the **final checkpoint of a 3000-iteration seed-42 run** — a fixed
budget, with no selection on the evaluation metric:

```bash
python scripts/eval_benchmark.py --task <task-id> --checkpoint <shipped .pt> \
    --num_envs 64 --eval_steps 6000 --seed 2026 --headless
```

| vehicle | checkpoint | targets/episode | mean speed | oob/episode |
|---|---|---|---|---|
| ROV  | `rov_calm_s42.pt` | **4.48** | 1.48 m/s | 0.02 |
| boat | `boat_calm_v26_s42.pt` | **3.75** | 1.33 m/s | 0.00 |

*Measured 2026-07-31 on the realistic-dynamics physics, eval seed 2026, single seed (42);
3-seed means still pending for both.*

> **These replace the old 27.1 / 5.1 figures and are not comparable to them.** Those were
> measured before the physics merge, on a model where the drag actually acting on the ROV
> came from a PhysX body damping of 4.0 authored inside `ROV_rigged.usd` rather than from
> the configured hydrodynamics, where `max_angular_velocity` pinned both vessels to a
> 5 deg/s turn rate, and where the boat's world-frame wrench was rotated twice. The
> vessels used to cruise at 6.1 / 4.6 m/s; they now settle at the terminal speeds the
> damping model is designed for (ROV 1.54 m/s at 400 N, boat 2.00 m/s), so throughput
> against targets 10–30 m away drops roughly in proportion. The new numbers are lower and
> physically calibrated; the old ones were higher and came from a model nobody designed.

> Two different numbers get called `targets_per_episode`: the **deterministic eval** score
> above, and the **training-time** wandb metric logged from a stochastic policy. Always
> say which one you mean.

> skrl's `best_agent.pt` is selected on episode return, which does not track the primary
> metric reliably — on these runs it scores 5.10 for the ROV (better) but 3.26 for the
> boat (worse). The shipped references are the final checkpoints, not `best_agent`.

The shipped boat reference is the **V26 speed-coupled** reward
(`tasks/boat_calm_nav/STARTER_TASK.md`). An experimental potential-based
**distance-progress** reward (`PROGRESS_COEF`, default off) was tried — it lifted a
from-scratch run 1.05 → 3.24 but scored *below* the V26 reference on the old physics, so
it stays an opt-in experiment, not the baseline. Worth re-checking now that the physics
has changed.

> The reward recipes (ROV E7, boat V26) and the 3000-iteration budget were both tuned
> against the old physics. Both runs above plateau by the end — the boat's episode return
> peaks mid-run and declines — so the recipes are worth revisiting rather than assuming
> they are still optimal.

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
- **A3 — Realistic dynamics.** ✅ merged 2026-07-31 from
  `jinshi-brady/realistic-dynamics` (ROV) and `jinshi-brady/realistic-dynamics-boat`
  (boat); both baselines retrained and the shipped checkpoints replaced. This brought in
  per-DOF linear+quadratic damping, actuator limits in the cfg with actions clipped to
  [-1,1], and three fixes worth calling out because they were silent:
  - the wrench frame — world-frame buoyancy/drag/current/wave were handed to
    `set_external_force_and_torque()`, which defaults to `is_global=False` and rotated
    them by the hull attitude again. Exactly zero error at yaw = 0, which is where every
    episode starts because `_reset_idx` never randomises heading, and growing from there:
    turned to 60°, the velocity ended up 42.7° off the bow.
  - `max_angular_velocity` is in **degrees per second**, not rad/s. At `5.0` both vessels
    were pinned to 0.087 rad/s under full yaw torque — a ~35 m turning circle for the boat
    at its old cruise speed. Now 573 (= 10 rad/s), deliberately non-binding, so the turn
    rate comes from the damping model.
  - `ROV_rigged.usd` authored a PhysX body damping of 4.0/4.0 that silently dominated all
    the configured hydrodynamic drag. Now explicitly zeroed.

  `scripts/check_wrench_frame.py` reproduces the first one in about two minutes with no
  policy involved — turn the hull, then thrust, and watch the drift.
- **A4 — Defaults did not match the shipped checkpoints.** ✅ fixed 2026-07-31.
  `OBS_DIM` defaulted to 7 for the ROV (its STARTER says 3) and to 3 for the boat (whose
  V26 recipe needs 9 plus `OBS_EXTENDED=1`, which itself defaulted to 0), so a clean clone
  running either task without env vars crashed on checkpoint load. Defaults now match the
  STARTERs.
- **A5 — `boat_physics.usdc` references a missing layer.** Loading it prints
  `Failed to open layer @<assets>/ROV_TEST.usd@`. Nothing depends on it and the task runs,
  but the dangling reference should be removed from the asset.

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
