# Catamaran P1 — review of `arif/catamaran-p1` @ `3f2d799`

Everything below was reproduced locally on a clean clone of the branch
(RTX 5080, Isaac Lab 2.7.0+cu128, skrl 1.4.3) before any change was made.

## Starting point

`scripts/eval_benchmark.py`, 64 envs x 6000 steps, evaluation seed 2026,
deterministic policy, checkpoint `catamaran_p1_s42.pt`:

| metric | value |
|---|---|
| targets_per_episode | **0.000** |
| total targets | 0 |
| mean speed | 4.400 m/s |
| closest approach to the active waypoint, over all 64 envs | 3.001 m |
| envs that ever entered the 3.0 m success region | 0 / 64 |

The reported `1.35 tgt/ep` does not reproduce under the standardized protocol.

## Bugs fixed on this branch

### 1. External wrench applied in the wrong frame (decisive)

`_apply_action` rotated thrust and torque into the world frame and then called
`set_external_force_and_torque(...)`, which defaults to `is_global=False` and
therefore treats the wrench as body-local — the rotation was applied twice, and
the world-frame buoyancy/drag terms were rotated out of world frame as well.

Open-loop check, full forward thrust for 6 s (`scripts/check_catamaran_physics.py`):

| | drift angle between velocity and bow |
|---|---|
| before | **71.2 deg** |
| after | **0.0 deg** |

Fix: pass `is_global=True`, since the assembled wrench is already world-frame.

### 2. `max_angular_velocity` is in degrees per second (decisive)

`RigidBodyPropertiesCfg.max_angular_velocity = 5.0` reads as 5 deg/s, i.e.
0.087 rad/s. Under full yaw torque the hull sat exactly on that cap, giving a
**49 m turning circle at cruise speed** while the patrol circuit has a 12 m
radius and a 3 m goal — the task was not physically solvable, whatever the
policy did. `MAX_TORQUE = 100 N.m` against angular damping 40 implies an
intended turn rate of ~2.5 rad/s, so the cap must not be the binding limit.

Fix: `max_angular_velocity = 90.0` deg/s (~1.57 rad/s, ~2.8 m turning radius).
The value is inherited from the boat reference task, so that task is worth
re-checking too.

### 3. Heading reward paid for standing still

`heading_rew` paid up to 0.5 every step for merely pointing at the waypoint.
Over a 7200-step episode that is up to 3600 reward available without moving,
against 150 per waypoint. `SPEED_COUPLE` did not couple anything — it added a
separate speed bonus. The best available policy was therefore to stop just
outside the goal radius and aim at it, which is exactly what the checkpoint
does (closest approach 3.001 m, success needs < 3.000 m).

Fix: `heading_rew = align * clamp(fwd_speed / SPEED_REF, 0, 1) * HEADING_W`,
the same speed-coupled form as the boat reference task (V23). The standalone
speed bonus is gone; `HEADING_W` and `SPEED_REF` are env-var tunable.

### 4. Unbounded patrol-progress observation

Observation [8] was `wps_done / N_WAYPOINTS`. The circuit repeats, so this grew
past 1 and kept growing for the whole episode — a drifting input the policy
cannot normalise. Replaced with `wp_idx / N_WAYPOINTS`, which is the position
inside the current lap and stays in [0, 1).

### 5. The hull mesh is rolled 90 degrees

The vessel renders lying on its side. A mirror-symmetry test on the mesh settles
it: a hull has exactly one mirror plane, the port-starboard centreline, and the
mismatch score is 0.254 about Z against 1.717 about X and 1.861 about Y. So the
mesh's beam axis is Z and its vertical axis is Y, while Isaac Sim treats +Z as
world up. The stage metadata says `upAxis = Z`, but the geometry inside was
authored Y-up and the conversion never compensated.

Fix: a +90 degree rotation about X on the `/catamaran_scaled/geometry` xform.
Root-frame extent goes from `3.0 x 1.014 x 1.101` (length x height x beam) to
`3.0 x 1.101 x 1.014` (length x beam x height). No inertia is authored in the
asset — PhysX derives it from the convex decomposition — so the inertia tensor
is recomputed correctly by the rotation.

Impact: mostly visual, but not only. The yaw inertia the simulator was using was
97.06 where it should be 103.77 (7%), `rov_height` was effectively describing
the beam rather than the height (8%), and any later hydrodynamics that depend on
hull geometry — added mass, wave loads, wetted area — would have been wrong.

Worth noting: the asset itself sets `physxRigidBody:maxAngularVelocity = 286`
deg/s, which is 5 rad/s and perfectly reasonable. The turn-rate cap in item 2
was introduced entirely by the task configuration overriding it with 5.0.

### 6. Heave oscillation: buoyancy had almost no vertical damping

Buoyancy is modelled as a linear spring of stiffness `rho*g*V/height` =
2940 N/m. Against 120 kg that is a 1.27 s natural period, and the vertical
damping reused the surge coefficient of 40 N.s/m, giving a damping ratio of
0.034. The hull bobbed continuously; measured over 16 envs the heave was
`std 0.126 m`, ranging `-0.761 .. -0.002 m`, still oscillating visibly after
30 s.

Fix: `heave_damping = 700.0` as its own coefficient (zeta = 0.59). This is also
the physically right shape — heave damping on a real hull is far larger than
surge damping. Measured after the change: `std 0.045 m`, range
`-0.439 .. -0.002 m`, settled at the -0.400 m equilibrium draft within 1.5 s.

### 7. Smaller items

- Yaw-rate observation used the world-frame rate; now body-frame, matching the
  docstring and the rest of the observation vector.
- Nav metrics were pushed to wandb on every one of the 60 control steps per
  second; now every `WANDB_EVERY` steps (default 60).
- `_setup_scene` created no light, so `--video` and the GUI viewport rendered
  black. Added the same dome light as the boat reference task.
- `_last_reached_mask` is now exposed for `eval_benchmark.py`'s per-step
  waypoint counting.
- `observation_space` used `__import__('os')` although `_os` was already
  imported.
- NOTES.md said the waypoint radius jitter is +/-40%; the code samples
  `0.8..1.2 x patrol_radius`, i.e. +/-20%.

## Result after the fixes

Retrained on this branch with **Arif's own PPO configuration unchanged**
(`skrl_ppo_cfg.yaml`, seed 42, 64 envs) — only the environment was fixed.
4000 iterations = 256k vector steps, about 1 h on an RTX 5080. Episode return
rose from 1520 to 6065 and was still climbing at the end.

Same standardized evaluation, 64 envs x 6000 steps, evaluation seed 2026:

| metric | before (`3f2d799`) | after |
|---|---|---|
| targets_per_episode | 0.000 | **19.406** |
| total targets | 0 | 1035 |
| mean speed | 4.400 m/s | 3.663 m/s |
| out-of-bounds per episode | 0.000 | 0.000 |

(An earlier run with items 1-4 fixed but before the hull rotation and the heave
damping scored 21.469; the corrected hull inertia and the damped heave cost a
little throughput and are the physically right configuration.)

The P1 bar is 2.0 targets/episode averaged over three seeds and a mean speed
above 1 m/s; seed 42 alone clears both by a wide margin. Seeds 123 and 456
still need to be run before P1 can be signed off.

## Still open, for Arif

- Where does `1.35 tgt/ep` come from? It is not the deterministic
  `eval_benchmark.py` number. If it is the wandb training metric
  `Metrics/targets_per_episode`, say so in NOTES.md and report the
  deterministic number next to it.
- The checkpoint stores `observation_preprocessor`, while skrl 1.4.3 uses
  `_state_preprocessor`; `eval_benchmark.py` only handles the opposite
  direction, so the observation scaler was silently skipped. Please pin the
  exact Isaac Lab and skrl versions used for training.
- The submitted policy is saturated: mean absolute raw action ~26 before the
  environment clips to [-1, 1], and the policy `log_std` sits at the 2.0 cap.
- No righting moment is modelled. The boat reference task adds roll and pitch
  restoring torques; this task has none, so attitude is only damped, never
  restored. Measured roll after the fixes is `std 1.9 deg, max 8.3 deg`, which
  is a lot for a catamaran. Left alone here on purpose — it is a task-design
  decision, not a bug introduced by the branch.

## How to re-check

```bash
# open-loop physics, no policy involved
python scripts/check_catamaran_physics.py --headless

# standardized deterministic evaluation
python scripts/eval_benchmark.py --task Isaac-Catamaran-Patrol-Direct-v1 \
    --checkpoint <ckpt.pt> --num_envs 64 --eval_steps 6000 --seed 2026 \
    --train_seed 42 --headless

# playback video (chase and top-down camera rigs)
python scripts/record_catamaran_video.py --task Isaac-Catamaran-Patrol-Direct-v1 \
    --checkpoint <ckpt.pt> --video_dir videos --tag run --cam chase --headless
```
