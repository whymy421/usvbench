# Task A: Static Hazard Navigation

> **Gym id:** `Isaac-USV-HazardNav-Direct-v1`  
> **Environment:** `HazardNavEnv` (`DirectRLEnv`)  
> **Default vehicle:** `vehicle="blueboat"`, resolved through
> `tasks/_shared/vehicles.py`  
> **Condition:** calm water, static analytic hazards

## Episode and scene

Every episode lasts exactly 120 s. Physics runs at 120 Hz (`dt=1/120`) with
decimation 2, so the policy controls at 60 Hz. Reaching the goal and colliding
are trajectory events only: neither terminates the episode early, and
`terminated` is always false. Time-limit truncation is the only done signal.

The vehicle starts at rest with its planar center of mass (COM) at the local
environment origin. A goal direction is uniform on `[0, 2*pi)` and its initial
COM distance is `D0 ~ U[20, 40] m`. The orange goal circle has radius 2.0 m.
Obstacles are vertical cylinders with radii sampled from `U[0.5, 2.0] m` in the
corridor strip whose longitudinal projection lies between start and goal.
Start and goal each have a 5 m obstacle-free disk.

## Reward-free C1 + C5 predicate

Let `tau_g` be the first trajectory sample at which the COM enters the closed
2.0 m goal circle. For cylinder `i`, define analytic hull clearance

```text
d_i = ||COM_xy - cylinder_axis_i|| - cylinder_radius_i - 0.45
d_min = min_i d_i
```

A contact sample is `simulator_contact OR (d_min < 0)`. V1 has no physical
obstacle colliders or contact sensors, so the analytic branch is authoritative
and no simulator obstacle contact is expected. Episode success uses prefix
semantics:

```text
success = (tau_g exists) AND (there is no contact on [0, tau_g])
```

Contact after `tau_g` cannot revoke an achieved success. Contact before or on
the first goal-entry sample permanently makes that episode unsuccessful.
Success is computed from the logged state trajectory and never from reward.

The environment snapshots these completed-episode attributes before automatic
reset: `episode_success`, `time_to_success` (NaN on failure),
`episode_path_length`, `episode_min_clearance`, and
`route_geodesic_length`. Path length and minimum clearance are accumulated over
the pre-goal prefix (or the full episode when no goal entry exists).
`goal_radius=2.0` and `_horizontal_distance()` mirror the evaluator discovery
interface. `route_geodesic_length` is the oracle's `L*` for hazard-aware SPL.

## Procedural feasibility oracle

`hazard_geometry.py` is importable without Isaac Lab and has no planner
dependency. It generates each candidate in a goal-aligned local frame, then
rotates the accepted layout into the sampled world direction.

1. Inflate every cylinder by the BlueBoat half-beam `0.45 m` (beam
   `0.899 m`) plus a `0.20 m` margin, for `0.65 m` total inflation.
2. Place the first cylinder directly on the start-goal segment, uniformly in
   the admissible part of its 30--70% interval, then scatter the remaining
   cylinders in the corridor. This makes direct-line obstruction the default
   and targets at least 70% blocked accepted maps.
3. Enforce the 5 m protected endpoint disks and the current curriculum's
   minimum pairwise free gap between inflated cylinders.
4. Rasterize the inflated cylinders on a 0.5 m occupancy grid and run
   eight-connected BFS from start to goal. Reject candidates with no path.
5. Record the accepted BFS polyline length as `L*`.

Each obstacle count receives at most 20 full-layout attempts. If all fail, the
sampler emits a `RuntimeWarning` and retries with one fewer obstacle until a
feasible layout is found. The fixed-seed acceptance test currently admits all
three rungs without fallback.

## Observation

The policy observation has 19 values:

```text
(goal_dot, goal_cross, clamp(distance_to_goal / D0, 0, 1), ray_0, ..., ray_15)
```

`goal_dot` and signed `goal_cross` use the registry-authored bow axis and the
COM-to-goal direction, matching the station-keeping convention. The planar
rangefinder is an analytic ray/cylinder intersection sensor: 16 rays cover
360 degrees in the body frame, ray 0 starts at the bow, adjacent rays are
22.5 degrees apart, maximum range is 30 m, and every return is normalized to
`[0, 1]` as `range/30`. It uses actual cylinder radii, not feasibility
inflation. V1 has no range noise or dropout.

Sensor sizing/admission constraint: cylinders have diameter at least 1.0 m,
and collision-critical presentations at distance `<=7 m` are required to
register on at least two ray samples. With 16 rays, the latter is a
ray-phase/scene-admission constraint rather than a diameter-only guarantee
(1.0 m at 7 m subtends about 8.2 degrees, less than the 22.5-degree spacing).
The included generator enforces the 1.0 m diameter floor; benchmark scene
admission should also count analytic ray hits when enforcing the two-ray
presentation condition. The upgrade path is the debated 72-ray sensor plus
explicit sensor-admission tests, followed by measured noise and dropout.

## Actions and calm-water physics

The action is a bounded `Box[-1, 1]^2` and is hard-clipped in
`_pre_physics_step`:

- action 0: bow-axis thrust, with the BlueBoat registry limits of 80 N forward
  and 48 N reverse;
- action 1: yaw torque, limited to 23 N m.

The selected registry entry also supplies the USD, bow axis, hydrodynamic drag,
buoyancy, and restoring parameters. BlueBoat is the official default.

## Ledger-capped reference reward

At one control transition (`dt_ctrl=1/60 s`):

```text
Phi(s) = -clip(dist_to_goal / D0, 0, 1)
r = Phi(s_next) - Phi(s)
    - 0.05 * dt_ctrl * clamp((0.9 - d_min) / 0.9, 0, 1)^2
    - 2.0 * 1[d_min < 0 on this sample]
```

For the fixed-horizon implementation, `dist_to_goal` in `Phi` is floored at
the 2.0 m goal radius. This is a memoryless plateau throughout the goal region
and prevents post-arrival reward circulation. There is no terminal bonus and
no success conjunction in reward. The maximum cumulative positive potential
gain is approximately 1.0 (strictly below it with the goal-radius plateau),
which is less than the 2.0 cost of one contact sample. This is the reward
ledger property; repeated contact samples can each incur the same penalty.

## Official curriculum

The environment imports `DockingCurriculum` rather than copying it. Its scalar
distance is interpreted as an integer level (`start=0`, `increment=1`,
`max=2`), preserving the same EMA reset/advance cooldown and low-EMA retreat
semantics.

| Level | Cylinders K | Minimum inflated pairwise gap |
|---:|---:|---:|
| 0 | 4 | 5 beams = 4.495 m |
| 1 | 8 | 4 beams = 3.596 m |
| 2 | 12 | 3 beams = 2.697 m |

The episodic success EMA uses decay 0.99 and advances at 0.60. Retreat uses
the imported curriculum's 400-episode patience and 0.05 threshold.

## V1 visual/physics boundary and upgrade paths

The water plane, orange 2 m goal ring, and gray extruded cylinders are guarded,
render-only USD geometry. No collision schema is authored for an obstacle.
Physics, observations, reward, and scoring read only analytic tensors. Thus the
boat visibly sails through a cylinder after contact, but the prefix episode is
scored failed. This V1 simplification makes PhysX contact dynamics unnecessary
and keeps thousands of procedural obstacles cheap.

The physical-obstacle upgrade path is to author collider instances, enable
contact reporting, add `simulator_contact OR analytic_contact` parity tests,
and validate that impact dynamics do not change the task predicate. The sensor
upgrade path is 72 rays, then calibrated noise/dropout. Neither upgrade may
silently change V1 results; each needs a new task version.

## Pairwise-control admission protocol

Any hazard-method claim must include obstacle-off paired control scenes. For
each seed, hold the vehicle, `D0`, goal direction, initial pose, horizon, and
calm-water dynamics fixed, remove only the hazards/rays, and evaluate the
corresponding `blueboat_calm_nav` scene at the same `D0`. Admit a training or
evaluation batch only when the paired scene identities match. Report both the
hazard result and this obstacle-off control so transit ability is not confused
with hazard avoidance.

## Reference recipe history

V1 deliberately uses 16 analytic rays rather than the debated 72 for training
tractability, keeps normal transit exploration (`initial_log_std=-1.9`), and
never clips critic predictions. The map oracle forces a direct blocker before
rejection sampling so an accepted corpus cannot collapse into straight-line
navigation. Potential shaping is per-environment `D0`-normalized and
ledger-capped; collision-free success remains trajectory judged. Analytic
obstacles and noiseless sensing are explicit V1 abstractions, not hidden
fidelity claims.

## Verify and train

The geometry acceptance test requires only NumPy and Torch:

```powershell
python .\tasks\hazard_nav\test_generation.py
```

In an Isaac Lab environment, train with:

```bash
python <IsaacLab>/scripts/reinforcement_learning/skrl/train.py \
  --task=Isaac-USV-HazardNav-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42
```
