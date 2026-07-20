# USV Boat Docking

The water, tolerance ring, and berth-heading marker form a render-only visual layer that is excluded from physics by construction.

> **Vehicle:** realistic boat hull (`boat_physics.usdc`)
> **Condition:** calm water (no waves or current)
> **Gym id:** `Isaac-USV-Dock-Direct-v1`

## Task

Each 120 s episode asks the boat to dock at the local environment origin with
its bow pointing along the per-environment dock heading (world `+X` by default).
The boat starts at rest on the current curriculum radius. Its bearing from the
dock is sampled uniformly from
`[-60, +60]` degrees relative to the dock heading, and its bow heading is the
dock heading plus a uniformly sampled offset in `[-45, +45]` degrees. Both are
rotated by the per-environment dock heading. The boat asset's bow/forward axis
is body `-X`.

Task geometry statement: docking is evaluated as an approach-sector task (vessels enter a berth from the water side); this is part of the task geometry, not a curriculum trick.

The fixed observation is
`(dot, cross, dist/25, dock_dot, dock_cross, planar_speed/2)`. The first two
components describe the dock position relative to the boat's forward direction;
`dock_dot` and `dock_cross` align boat forward with the dock heading. Actions
are forward thrust and yaw torque in `[-1, 1]^2`.

## Crossing variant

The BlueBoat current crossing exercises the C3xC6 interaction: docking requires a crab-angle approach that compensates for lateral drift. The per-episode current is reportable as `current_vec` and `Episode/current_speed`, but it is intentionally absent from policy observations under the asymmetric-information principle because a real boat must infer current from its drift rather than sense the current field directly. Current magnitudes await cross-validation with the upstream fixed-direction flow implementation, and the two force paths must not be double counted when that implementation is merged.

## Reward-free success predicate

All three conditions must hold simultaneously for 5 consecutive seconds:

- horizontal position error `<= 2.5 m`;
- absolute heading error from dock heading `<= 15 degrees`;
- planar speed `<= 0.3 m/s`.

Breaking any condition resets only the 5 s timer. Every episode runs for the
full 120 s. Episode success is an achieved-once metric: it is true when the
conjunction has held for at least 5 consecutive seconds at any point in the
episode, even if a later violation resets the current timer. Time to success is
the elapsed time when that streak first completes, or NaN if it never does.
Success is computed only from the logged state trajectory and does not depend
on reward.

## Official curriculum

The success-rate EMA starts at zero and is updated once per finished episode.
The spawn radius never decreases.

| Parameter | Value |
|---|---:|
| Initial spawn radius | 2.0 m |
| Success-rate EMA decay | 0.99 |
| Advancement threshold | 0.6 |
| Radius increment | 2.5 m |
| Maximum spawn radius | 25.0 m |

When the updated EMA is at least 0.6, that episode advances the radius by one
2.5 m stage, capped at 25 m. The EMA is retained across stages.

## Reference reward

The included reward is **reference-only** and is not part of success:

```text
-distance/25 + 0.5 * dock_dot * exp(-distance/2.5)
    + 0.4 * exp(-distance/2.5) * clamp(1 - planar_speed/1.0, 0, 1)
    + 3.0 * [instantaneous success predicate is true]
```

## Reference training recipe history

V1 omitted the braking and hold-progress terms. A full 3000-iteration training
run produced 0% episode success, a zero success-rate EMA, and no advancement
beyond its then-initial 5.0 m curriculum stage: the policy learned to remain
close and aligned while still moving. V2 adds near-dock deceleration credit and
progress credit during the consecutive hold. This documents reward sensitivity
only; the scored success predicate did not change.

V2.1 keeps the V2 reward and adds a bounded action space, `clip_actions: True`,
and `initial_log_std: -1.0`. This prevents the Gaussian policy mean from locking
beyond the action rails, where hard-clipped samples become indistinguishable,
while reducing disruptive exploration during the 5 s hold. V2 and V2.1 training
also scored 0% episode success.

V3 adds approach-sector spawns, an in-zone 2.0 m curriculum start, and
proximity-gated alignment. The diagnosis was that the alignment term was
antagonistic to distance-closing approaches on the far side of the dock, while
the previous 3.0 m start placed zero-exploration episodes outside the position
tolerance. The scored success predicate remains unchanged.

V4 fixes reward milking: zero-action scored 39% success while trained PPO scored 0%, showing that the policy learned to avoid success termination and keep collecting the near-dock dense stream.
Design rule for all success-terminated mission tasks: add a one-time terminal success bonus large enough to dominate the discounted value of continuing to milk dense rewards; V4 uses `150.0` and reduces `initial_log_std` to `-1.5` so exploration is less likely to break the speed condition.

V5 fixes the V3/V4 geometry, which spawned the boat past the berth facing away;
noise drift under asymmetric thrust destroyed all bootstrap successes, and the
required reverse-parking recovery was unlearnable. V5 spawns on the approach
lane so drift, thrust authority, alignment shaping, and the predicate all point
the same way. Design rule: spawn geometry must make the success maneuver lie
along the vehicle's strong actuation axis.

V6 fixes an exploration infeasibility that remained after the V5 geometry
change. The success predicate requires planar speed `<= 0.3 m/s` for 300
consecutive 60 Hz control steps. Under Gaussian exploration with
`initial_log_std: -1.5` (sigma approximately 0.22) on a 500 N thruster, the
velocity random-walk RMS is approximately 0.4 m/s. The probability that the
instantaneous speed conjunction survives all 300 steps is therefore
effectively zero, so stochastic rollouts never sample success or its terminal
bonus even though deterministic zero-action evaluation scores 39%. V6 reduces
`initial_log_std` to `-2.5` (sigma approximately 0.08, velocity RMS
approximately 0.07 m/s), making bootstrap spawn-successes samplable, and raises
the terminal success reward from `150.0` to `500.0`. With `gamma=0.99`, the
milkable dense stream is worth approximately 140 from the best hover state, so
the larger bonus makes completion strictly dominant.

Design rule: a hold-based success predicate bounds the exploration noise a
solver can use. Benchmark tasks with consecutive instantaneous-speed
conditions must either be paired with low-noise exploration or use an
averaged-speed criterion.

V13 makes every episode fixed-horizon. Removing success termination eliminates
timer-dependent termination, so the environment remains Markov without adding
the hold timer to observations; the timer now exists only for evaluation
logging. The reward is a pure function of instantaneous physical state
(position, heading, and speed), while the consecutive-5 s requirement is
evaluated from logged trajectories. Terminal-scored docking has precedent in
Patil et al. (2021), and the separation follows the shaping and time-limit
principles of Ng et al. (1999) and Pardo et al. (2018).

Design rule: with no early termination, per-step conjunction pay makes holding
the successful state the optimum. The success-termination milking trap is
therefore structurally impossible because incentives and the achieved-once
metric point in the same direction. This protocol change invalidates direct
comparison with V12 returns and requires matched-seed retraining (queued).

## Train

Copy this folder into the Isaac Lab direct-task package, set `USVBENCH_ASSETS`
if the repository is not at `~/usvbench`, activate Isaac Lab, then run:

```bash
python <IsaacLab>/scripts/reinforcement_learning/skrl/train_with_eval.py \
  --task=Isaac-USV-Dock-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42 \
  --video --video_interval 50000 --video_length 200
```

## Smoke tests

The curriculum tests and curriculum-only smoke do not require Isaac Lab or a
GPU:

```powershell
python .\tasks\docking\test_curriculum.py
python .\tasks\docking\smoke.py --curriculum-only
```

Full Isaac Lab smoke:

```powershell
$env:USVBENCH_ASSETS = (Resolve-Path .\assets).Path
python .\tasks\docking\smoke.py --headless
```

V13b ablation (negative result, documented): capping KLAdaptiveLR at
max_lr=3e-4 did NOT cure the recurring training cliff -- the LR trace
confirms the cap engaged (peak 2.9e-4) and KL cuts fired, yet success
still oscillated (1.00 -> 0.23 -> 0.88 -> 0.00). The instability is not
pure LR growth; value-function dynamics are implicated. Early-stop
checkpoint harvest (dense interval + eval ladder) remains the
documented recipe. V13 reference (92.2%) stands.
