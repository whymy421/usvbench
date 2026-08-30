# Rescue Boat P2 — Task Notes

Task documentation for anyone picking up or extending this task.
Arif Abubhakkar, August 2026.

---

## Task

`Isaac-RescueBoat-Direct-v1`

Four casualties spawn with independent countdown deadlines. The vessel provably
cannot reach all of them, so the policy has to choose which to abandon. This is
what separates P2 from the navigation tasks: P1 is navigation toward a fixed
objective, P2 is triage under a deadline.

The task is scaled so that servicing all four exceeds the shortest deadline. A
four-point tour spans roughly 500 m and needs about 42 s of transit against
deadlines starting at 40 s, so prioritisation is forced rather than optional.

## Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| n_casualties | 4 | |
| rescue_radius | 15.0 m | must exceed hull length and turning radius |
| casualty_timer_min/max | 40.0 / 90.0 s | independent per casualty, uniform |
| casualty_spawn_r_min | 80.0 m | **must exceed rescue_radius**, see pitfall 4 |
| casualty_spawn_r_max | 200.0 m | scaled to vessel speed |
| episode_length_s | 120.0 s | |
| OBS_DIM | 14 | top-2 casualties only, so the dimension is independent of n_casualties |
| URGENCY_POW | 2.0 | priority `u^2 / (d + eps)`, `u = 1 - t_rem/t_init` |
| PROGRESS_COEF | 3.0 | dense, on closing distance |
| RESCUE_BONUS | 500.0 | terminal, per casualty |
| LOSE_PENALTY | 50.0 | terminal, per casualty expired |
| HEADING_COEF | 0.05 | `cos(delta)`, locomotion shaping |
| YAWRATE_COEF | 0.03 | `-omega_z^2`, locomotion shaping |
| PROX_COEF | 0.0 | deliberately zero, see reward history |
| MAX_THRUST | 500 N | body +X |
| MAX_TORQUE | 800 N·m | body +Z |
| mass | 300 kg | ARENA RIB 8.50 m hull |
| rov_volume | 0.75 m³ | vs 0.30 for neutral buoyancy, so ~150% reserve |
| max_linear_damping | 30.0 | RIB has less beam drag than the catamaran |
| max_angular_damping | 60.0 | high yaw damping for realistic steering |
| max_angular_velocity | **60.0** | **DEGREES/s**, see pitfall 1 |
| env_spacing | 1200.0 m | rule of thumb: >= 3 x casualty_spawn_r_max |
| RIGHTING_K / C / MAX | 1500 / 400 / 2500 | restoring moment |

## Metric

```
rescue_rate = rescues / (N_env x N_cas x N_ep)
```

Counted over **complete episodes**, not over a sampling window. See pitfall 3
for why that distinction matters.

## Results

Bar for P2: `rescue_rate >= 0.70` across three seeds.

| Seed | Run directory | Checkpoint | Committed as | rescue_rate |
|------|---------------|-----------|--------------|-------------|
| 42 | 2026-08-22_17-29-30 | agent_60006 | `checkpoints/rescue_boat_p2_s42_best.pt` | 0.777 |
| 123 | 2026-08-22_19-28-22 | agent_300030 | `checkpoints/rescue_boat_p2_s123_best.pt` | 0.809 |
| 456 | 2026-08-22_21-19-50 | agent_180018 | `checkpoints/rescue_boat_p2_s456_best.pt` | 0.723 |
| **mean** | | | | **0.770 ± 0.036** |
| scripted pure-pursuit baseline | | | | 0.582 |

All three clear the bar. Margin over the scripted baseline is 0.188. The ± is a
population standard deviation across three seeds, not a standard error, and no
significance test is claimed at n = 3.

**Two things to know before quoting these numbers.**

The checkpoint for each seed was selected on the same episodes used to report
it, which makes this an optimistic estimator. Re-scoring one selected checkpoint
over independent episodes returned 0.872 against the 0.934 that had selected it,
so the selection bias is of order 0.06. A fully held-out figure would select on
one evaluation seed and report on another.

The run directory names are all `..._s42` and that is wrong. The launch script
did not substitute the seed into the directory name. The real seed for each run
is in `params/agent.yaml` and is 42, 123 and 456 respectively. The committed
checkpoints are named by the real seed, not by the directory.

## Running it

```
set USVBENCH_ASSETS=<repo_root>/assets
set USVBENCH_PYTHON=<isaac lab python.exe>

# 1. Verify the vehicle model before trusting anything downstream.
#    Phase 3 sweeps thrust and yaw through all headings.
python tools/probe_physics.py --headless

# 2. Score a committed checkpoint.
python tools/eval_fixed.py --checkpoint=checkpoints/rescue_boat_p2_s123_best.pt --headless
```

**Versions:** skrl 2.1.0, Isaac Lab 0.54.4, Isaac Sim 5.1.0.

The checkpoints store the observation normaliser under the key
`observation_preprocessor`. On skrl 1.4.x the equivalent key is
`_state_preprocessor`. `eval_fixed.py` searches both and **aborts** rather than
scoring if it finds neither, because a silently skipped normaliser produces
plausible-looking numbers from a policy being fed unscaled observations.

## Physics implementation

- Buoyancy is proportional to submerged fraction and applied at the centre of
  mass. A point force at the COM generates no torque, so an explicit restoring
  moment is required for any roll stability at all.
- Restoring moment uses the **cross-product form**, which is invariant under
  heading: `T = K (u_hull x u_world) - C omega_perp`, clamped at `RIGHTING_MAX`.
  Damping acts only on the angular velocity component perpendicular to world
  up, so yaw authority is untouched.
- The wrench is converted to the **body frame** before
  `set_external_force_and_torque`, which expects body frame. Assembling in world
  coordinates and passing directly applies a second rotation.
- Forward axis: body +X.
- Every run prints its actuation model, restoring-moment form and task
  parameters at startup, so provenance travels with the output.

---

## Known pitfalls

These are recorded because each one cost real time on this task, each produced
a plausible-looking wrong answer rather than an error, and each is likely to
recur on any new vessel or task added to the benchmark.

### 1. `max_angular_velocity` is in degrees per second, not radians

**Symptom.** The vessel appears to train normally but cannot complete turns. A
value set as though it were radians silently caps the hull far below the turning
rate the task needs. At a setting of 3.0 the effective cap is 3 deg/s, roughly
0.05 rad/s, which makes any circuit with a tight goal radius physically
unsolvable regardless of the policy.

**Why it survives.** Nothing errors. The vessel moves, the reward changes, the
training curve looks like a hard task rather than an impossible one.

**Detection.** Compare the configured value against the yaw rate measured by the
open-loop probe. If they differ by a factor near 57, this is the cause.

**Note.** This was diagnosed independently on the catamaran task before it
appeared here. It recurred because the local working tree and the committed tree
had diverged, so the fix did not propagate. Keeping vehicle specifications in
the shared registry rather than per-task is the structural fix.

### 2. Forces that are exactly correct at zero heading

**Symptom.** The hull capsizes or sinks whenever it sustains a turn, while every
open-loop test passes.

**Cause.** Two independent faults of this shape were found here. A restoring
moment computed from Euler roll and pitch but applied about world X and Y
coincides with the correct moment only at yaw = 0; once the hull turns, the roll
correction acts partly about the pitch axis and the two pump each other. A
wrench assembled in world coordinates and passed to a body-frame API is
similarly correct only when body and world frames align.

**Why it survives.** Every episode begins at zero heading, and single-axis
open-loop testing operates there by construction. Both faults are invisible at
exactly the attitude where all conventional testing sits.

**Detection.** `tools/probe_physics.py` phase 3 drives thrust and yaw
simultaneously, sweeps through all headings, and asserts peak tilt <= 5° and
depth > -5 m. Before correction the hull reached 166° of roll and -43 m; after,
peak tilt is 1.77° through 2230° of cumulative heading change.

**Attribution.** Both were identified by Yutong Song, the project PGTA, after
they had survived the author's own testing for the reason above.

### 3. Evaluation code is where faults hide

**Symptom.** Across five independent seeds, every checkpoint scored exactly
0.0000 except at two identical checkpoint steps. Independent seeds do not peak
together, so a pattern identical across seeds is a property of the measurement,
not of the policies.

**Cause.** Two faults, and they concealed each other. The observation normaliser
was not restored on checkpoint load, so policies were scored under statistics
they had not been trained with. Separately, a 1500-step scoring window was
divided by a full-episode denominator, inflating the rate by roughly 4.6x
because casualties are reached early in the episode.

**Why it survives.** An incapacitated hull scored by an inflated metric returns
roughly 0.5, which reads as underperformance rather than as error. Either fault
alone would have been conspicuous: the corrected metric on a capsizing hull
gives 0.10, and the inflated metric on a working hull exceeds 1.0.

**Detection.** `tools/diagnose_eval_sweep.py` applies four checks that any sound
evaluator must pass: score one checkpoint twice, score A then B then A again,
force-restore the normaliser, and compare windowed against full-episode scoring.
All four failed here. `tools/eval_fixed.py` now refuses to score if the
normaliser cannot be restored and asserts `rescue_rate <= 1.0`.

**General point.** Evaluation code is ordinary software written under the same
time pressure as everything else, and receives a fraction of the scrutiny given
to the policy. A systematic measurement fault does not become visible by running
more seeds.

### 4. Thresholds asserted rather than demonstrated

**Symptom.** Negative results that cannot be interpreted. If the pass threshold
sits above anything any controller can reach, a failing policy and an impossible
task look identical.

**Detection.** Build a scripted controller with perfect state access and no
learning, and measure what it achieves. Under the original task parameters here
that controller reached 0.262 against a 0.70 threshold, which showed the
parameters, not the policy, were the problem: the 5 m capture radius was smaller
than both the 8.5 m hull and the ~7.6 m minimum turning radius, so an overshoot
could never be corrected.

**Caveat on interpretation.** A scripted controller establishes a *demonstrated
attainable lower bound*, not a ceiling. It has fixed gains and no route
planning, so a threshold above it may still be reachable by a better controller.
What it does establish is that a threshold set at more than twice anything
demonstrated is an assertion rather than a specification.

**Related constraint.** `casualty_spawn_r_min` must exceed `rescue_radius`, or
some casualties begin the episode already inside the capture zone and are
counted as rescued at t = 0. This produced a `rescue_rate` above 1.0, which is
how it was caught. A configuration yielding 0.94 instead of 1.02 would not have
been caught by anything in the pipeline.

### 5. Workarounds outliving their cause

The casualty spawn radius was narrowed early to address what appeared to be
reward sparsity. It was in fact compensating for the physics fault in pitfall 2,
which was capping effective speed. The workaround survived six weeks past the
removal of its cause, and every subsequent parameter decision inherited a task
easier than intended.

A workaround introduced to compensate for a suspected fault should carry an
explicit condition for when it is removed, not only a note of when it was added.

---

## Reward design history

Recorded so the dead ends are not repeated on other tasks.

| Version | Term | Intent | Learned behaviour |
|---------|------|--------|-------------------|
| V2 | proximity | be near casualties | hovered without rescuing: ~34,000 shaping reward per episode against 1,200 available from the task |
| V3 | progress only | close distance | spiralled while translating: 19 revolutions per episode, hull 87° off its direction of travel |
| V4 | progress + heading + yaw-rate | drive forward | 12 revolutions, 59° off track, rescue rate up 0.075 |

Each of the first two named a *correlate* of the goal rather than the goal
itself, and the optimiser maximised the correlate.

**Sizing.** Dense terms accumulate over 7200 steps per episode. The V4
locomotion terms contribute at most 598 per episode against 2000 available from
recovering all four casualties. Coefficients an order of magnitude larger would
have contributed 3600, making the shaping term the effective objective.

A cheap diagnostic follows from this: before trusting a training run, compare
the reward a shaping term accumulates over an episode against the reward
available from actually completing the task.

## Asset provenance

`assets/rescue_boat.usd` is built from an ARENA RIB 8.50 m mesh, 8.5 m × 2.82 m,
simulated at 300 kg as a single rigid body.

**Licence status is unresolved.** The source CAD was obtained during the project
and its redistribution terms were never confirmed. The asset is committed here
so the task runs from a clean clone within this internal registry, but it should
not be included in any public release or archived under a DOI until the terms
are established. `docs/ASSET_PROVENANCE.md` records what needs checking.

For a shared benchmark, a parametric hull generated from published dimensions
would be the better long-term answer, since it removes the dependency entirely.
`tools/make_rescue_boat_usd.py` is the starting point.

## Open items

- The restoring moment uses hand-set constants rather than the shared
  implementation in `tasks/_shared/restoring.py`. The shared version is
  preferable for a benchmark, since consistency across vessels matters more than
  per-task tuning. Treat the constants here as a placeholder.
- `tools/eval_robustness.py` is implemented but has never been run.
- The final policy still completes roughly 12 hull revolutions per 120 s
  episode, down from 19 before locomotion shaping. That is an improvement, not
  physical vessel behaviour. The residual traces to the priority ordering
  re-ranking mid-transit, measured at 2.6 target changes per episode, which
  affects any controller obeying that ordering including the scripted baseline
  at 8.8. Damping it would require hysteresis in the priority function.
- The task parameters were recalibrated late in the project. An earlier
  configuration used a 20 m capture radius, 60–120 s deadlines and a 35 m spawn
  radius, under which the vessel covered roughly 1440 m per episode against a
  task spanning about 60 m and the scripted controller sat above the threshold.
  The current values force a genuine tour. Any comparison against numbers
  produced before that recalibration is not like for like.
