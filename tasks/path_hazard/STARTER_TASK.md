# C4 x C5: Path Hazard

> **Gym id:** `Isaac-USV-PathHazard-Direct-v1`  
> **Vehicle:** `vehicle="blueboat"`, from `tasks/_shared/vehicles.py`  
> **Condition:** calm water, ordered route, static analytic hazards  
> **Version:** v1, fixed difficulty with no curriculum

## Task and interaction effect

Path Hazard crosses ordered path following (C4) with obstacle avoidance (C5).
Each episode samples the same four-segment polyline as `path_following`: segment
lengths are uniform on 12-20 m, the first heading is uniform over 360 degrees,
and each later heading changes uniformly by at most +/-60 degrees. The four
segment endpoints are ordered 2.0 m gates.

The interaction is deliberate rather than incidental. Six cylinders have radii
uniform on 0.8-1.5 m. Three distinct route segments receive a forced blocker at
30-70% of the segment with at most 0.5 m centerline offset. The remaining
cylinders are scattered within 6 m of the polyline. Because each forced
blocker's radius plus the BlueBoat half-beam exceeds its possible line offset,
following the direct reference line produces contact. A successful controller
must leave the line, clear a hazard, and re-acquire the active segment before
continuing through the ordered gates.

This creates the intended XTE-versus-clearance conflict. Plain path following
rewards reference progress and evaluates cross-track error, while safe transit
requires temporary non-zero XTE. Random nearby obstacles would not reliably
activate that conflict; on-line placement guarantees it in every accepted
scene.

## Fixed horizon and success certificate

Physics runs at 120 Hz with decimation 2, giving a 60 Hz control rate. Every
episode lasts exactly 150 s. `terminated` is always false; time-limit truncation
is the only done signal. Gate entry and contact are achieved-once trajectory
events and never end the episode early.

Let `tau_4` be the first sample at which all four 2.0 m gates have been entered
in order. Analytic hull clearance is

```text
d_min = min_i (||COM_xy - center_i|| - radius_i - 0.45)
contact = (d_min < 0)
success = (tau_4 exists) AND (no contact on [0, tau_4])
```

Contact on the last-gate entry sample counts against success. Contact after
`tau_4` cannot revoke success, matching `hazard_nav` prefix semantics. Reward is
never consulted by this predicate.

Completed-episode snapshots are `episode_success`, `time_to_success` (NaN on
failure), `episode_gates_passed`, `episode_path_length`,
`episode_min_clearance`, and `episode_xte_rms`. Path length, minimum clearance,
and XTE RMS use the pre-last-gate prefix (or the full episode if gate four is
never reached). XTE is sampled against the segment active at the start of the
control step. Live `route_length` and the scalar `goal_radius=2.0` are exposed
for evaluator compatibility; `episode_route_length` preserves the completed
route across automatic reset.

## Feasible procedural layouts

`path_hazard_geometry.py` depends only on the standard library, NumPy, and
Torch. Layout admission is:

1. Sample the four-segment route and six radii.
2. Put three hazards on three distinct route segments using the 30-70% and
   0.5 m limits; scatter the other three within 6 m of the polyline.
3. Inflate every cylinder by `0.45 + 0.20 = 0.65 m` for feasibility. Protect
   the start and every gate with obstacle-free 3 m disks, and reject overlapping
   inflated obstacles.
4. Run the 0.5 m, eight-connected BFS oracle separately for
   `start->gate1`, `gate1->gate2`, ..., `gate3->gate4`, with all obstacles
   present. Accept only if every leg has a collision-free corridor.
5. Try at most 20 candidates at `K=6`. If exhausted, emit a `RuntimeWarning`
   and reduce K one at a time while retaining all three forced blockers.

The fixed-seed 50-layout acceptance test is expected to retain full `K=6` in
all layouts.

## Observation and action

The action is `(bow_thrust, yaw_torque)` in `[-1, 1]^2`, hard-clipped before
application. The BlueBoat registry supplies its asset, `+X` bow axis, 80 N
forward/48 N reverse limits, 23 N m yaw limit, and calm-water dynamics.

The policy observation has 43 values. Values 0-6 copy `path_following` v4b:

| Index | Field |
| ---: | --- |
| 0-2 | dot, signed cross, and distance/20 m to the current gate |
| 3 | `gates_passed / 4` |
| 4-6 | dot, signed cross, and distance/20 m to the next gate |
| 7-42 | 36 normalized analytic ray ranges |

At the final gate, the next-gate index is clamped so next equals current. It is
not padded with the out-of-distribution `(1, 0, 0)` triplet. The 36 noiseless
rays cover 360 degrees in 10-degree increments from the bow, intersect actual
cylinder radii analytically, use a 30 m maximum range, and are normalized to
`[0, 1]`.

## Reference reward

For one 1/60 s control step:

```text
progress = previous_distance_to_step_start_gate
           - current_distance_to_step_start_gate
proximity = clamp((0.9 - d_min) / 0.9, 0, 1)
r = 20 * progress
    + 10 * 1[ordered gate entered this step]
    - 1.0 * (1/60) * proximity^2
    - 25 * 1[contact rising edge]
    - 1 * 1[contact dwell sample]
```

The progress implementation is copied from `path_following`: it is a raw
distance potential delta and is **not** normalized by a per-environment D0.
After the reward is calculated, distance is re-anchored to the possibly new
active gate. This prevents mixed old-gate/new-gate deltas. Each gate bonus is
paid once. There is no terminal bonus because episodes have a fixed horizon.
The barrier, contact-entry latch, and contact-dwell terms are the `hazard_nav`
v3 ledger family.

## Bidirectional certificate plan

The interaction claim requires two negative-transfer certificates in matched
scenes:

- Run the `path_following` champion without hazard-aware adaptation. It should
  track the reference line and contact the forced on-line cylinders.
- Run the `hazard_nav` champion without ordered-route adaptation. It may avoid
  local hazards, but should miss ordered gates and/or produce poor active-segment
  XTE.

A positive Path Hazard result is meaningful only alongside both failures; this
distinguishes solving the crossed capability from inheriting either parent.

## Pairwise control and curriculum upgrade

Every benchmark batch should include obstacle-off matched scenes: preserve the
route, vehicle, initial pose, horizon, seed, and calm-water dynamics, but remove
only obstacles and rays. These are plain `path_following` controls and isolate
the incremental cost of the interaction from ordinary route-following ability.

V1 has no curriculum: every requested layout is fixed at `K=6` with three
on-line blockers. A future version may stage-unlock difficulty only by creating
explicit rungs, for example obstacle-off route acquisition, scattered hazards,
one on-line blocker, then three on-line blockers. Promotion should use a
held-out success EMA and must retain matched-scene identities. Such a change is
a new task version; it must not silently alter v1.

## Visual boundary and verification

Water, env 0's ordered-color gate rings and white route line, and gray obstacle
cylinders are guarded render-only USD geometry. Obstacles have no collision
schemas. The marker helper reuses an existing translate xform op before adding
one, which keeps cloned env trees safe. Physics, observation, reward, contact,
and scoring read only analytic tensors.

The pure-Python acceptance test is:

```powershell
python .\tasks\path_hazard\test_path_hazard_geometry.py
```

Train in an Isaac Lab environment with:

```bash
python <IsaacLab>/scripts/reinforcement_learning/skrl/train.py \
  --task=Isaac-USV-PathHazard-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42
```

The PPO recipe copies `hazard_nav` (`initial_log_std=-1.9`), uses experiment
name `path_hazard_v1`, and checkpoints every 3200 trainer intervals.
