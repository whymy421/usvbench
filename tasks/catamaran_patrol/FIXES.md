# Catamaran P1 — physics review at commit `3f2d799`

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

> These are the defects as found, with the fix that was applied at the time. Three of the
> values below were later superseded by the physics port documented further down —
> `is_global=True` became body-frame assembly with the API default, the 90 deg/s cap
> became 573, and `heave_damping` 700 became 330. The diagnosis in each item still stands;
> read the port section for what the code does today.

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

## Ported onto the reference physics (2026-07-31)

After the fixes below were validated, the task was ported onto the same physics model
the reference tasks now use on `main`, so its score is comparable to theirs and it is not
the odd vessel out:

- **Per-DOF linear+quadratic hydrodynamic damping** replaces the single isotropic
  coefficient. Coefficients are Froude-scaled from the VRX WAM-V exactly the way
  `boat_calm_nav` derives its own — lambda = (120/195)^(1/3) = 0.850 for this 120 kg hull.
- **Actuator limits moved into the cfg** (`thrust_max_fwd` 850 N, `thrust_max_rev` 340 N,
  `yaw_torque_max` 740 N.m), replacing the hardcoded `MAX_THRUST` / `MAX_TORQUE` module
  constants, with asymmetric forward/reverse thrust.
- **Velocity caps made non-binding** (`max_angular_velocity` 573 deg/s). The 90 deg/s in
  the first round of fixes was still clamping the hull: the damping-limited rate is
  2.5 rad/s and 90 deg/s is 1.57.
- **Attitude restoring spring added.** The task modelled no righting moment at all;
  measured roll was `std 1.9 deg, max 8.3 deg`. See the capsize note below — the first
  version of this spring was itself wrong.
- **Frame discipline matched to the boat**: body-frame terms go straight into the force
  tensor, world-frame terms accumulate separately and are rotated in once at the end,
  with the API default `is_global=False`.

Sizing is documented in the cfg and verified open-loop:

| | design | measured |
|---|---|---|
| terminal surge | 2.50 m/s | **2.500** |
| terminal yaw | 1.00 rad/s | **1.000** |
| drift vs bow under thrust | 0 deg | **0.0** |
| turning radius at cruise | 2.5 m (same as the boat reference) | 2.5 m |

Two knock-on constants had to move with the terminal speed, and both would have failed
silently: `SPEED_REF` in the speed-coupled heading reward (5.0 -> 2.5, otherwise the
reward is permanently scaled to 40% of its design value) and the observation normalisers
`max_speed` / `max_yaw` (8.0/3.0 -> 4.0/2.0).

Known limitation, stated rather than hidden: the Froude scaling is anchored on mass,
following the boat. By mass the catamaran is the larger vessel (lambda 0.850 vs 0.800),
but its hull is 3.0 m against the boat's 5 m — heavier and shorter — so mass-anchored
scaling overstates its length scale. No catamaran hull-form correction is applied at all.
This is a documented first approximation, not measured hydrodynamics.

### Capsize-by-turning: the attitude spring was not yaw-invariant

The first version of the attitude spring above extracted roll and pitch from the
quaternion and applied the restoring torque about the fixed **world** X and Y axes.
Those angles are body quantities and the axes are world axes; the two coincide only at
yaw = 0. Once the hull turns, the roll correction is applied about what is by then partly
the pitch axis, roll and pitch pump each other, and the vessel tumbles.

Measured under the trained policy, 16 envs, 30 s:

| | broken spring | fixed |
|---|---|---|
| cumulative roll about body X | 2.16 turns (worst env 5.04) | **0.00** |
| roll angle range | -180.00 .. 179.99 deg | -0.06 .. 0.06 deg |
| pitch angle range | -89.71 .. 89.78 deg | -0.00 .. 0.02 deg |
| peak roll rate | 7.59 rad/s | 0.01 rad/s |

Fix: the yaw-invariant form both reference tasks already use,
`k * (hull_up x world_up)`, i.e. `torque_w[:, 0] += k*up_w[:, 1]`,
`torque_w[:, 1] += -k*up_w[:, 0]`. `boat_calm_nav` carries a comment describing exactly
this failure; the Euler-angle form is the one it had already abandoned.

This bug and the double-rotated wrench (item 1) share a signature worth naming: **both are
exactly correct at yaw = 0**, which is where every episode starts and where a zero-action
test sits. An attitude measurement taken with zero actions showed `roll 0.000 deg` while
the hull was tumbling under the policy. `check_catamaran_physics.py` therefore gained a
phase 3 that drives thrust and yaw torque together, sweeping the hull through every
heading, and fails if the tilt off vertical exceeds 5 deg. All three phases pass:
worst tilt 0.02 deg.

## Result after the fixes

Retrained on this branch with the **submitted PPO configuration unchanged**
(`skrl_ppo_cfg.yaml`, seed 42, 64 envs) — only the environment was fixed.
4000 iterations = 256k vector steps, about 1 h on an RTX 5080. Episode return
rose from 1520 to 6065 and was still climbing at the end.

Same standardized evaluation, 64 envs x 6000 steps, evaluation seed 2026:

| metric | before (`3f2d799`) | after the bug fixes | after the physics port |
|---|---|---|---|
| targets_per_episode | 0.000 | 19.406 | **20.175** |
| total targets | 0 | 1035 | 1076 |
| mean speed | 4.400 m/s | 3.663 m/s | 2.465 m/s |
| out-of-bounds per episode | 0.000 | 0.000 | 0.000 |

Only the last column is comparable to the rest of the benchmark; the middle column is on
the old physics. The port lowers the score because it lowers the speed — the vessel now
runs at the 2.5 m/s its damping model is designed for instead of 3.7 m/s, so it covers the
circuit less often. That is the same trade the reference tasks made when they were
re-baselined.

Three intermediate numbers, recorded so they are not mistaken for results: 21.469 with
items 1-4 fixed but before the hull rotation and heave damping; 15.600 after the port but
with the broken attitude spring, i.e. with the hull tumbling; and 17.850 after the spring
was fixed but with the hull still skating sideways through its turns. All three are void.

The final figure exceeds the 19.406 measured on the old physics despite a 1.2 m/s lower
cruise speed, because the hull now carves its turns instead of sliding through them.

`mean_speed` is measured by `eval_benchmark.py` as `root_com_vel_w` while the open-loop
check measures `root_lin_vel_w`; on a turning hull those differ by the omega x r term, so
the two are not expected to agree exactly.

The P1 bar is still TBD — the old 2.0 was calibrated against
reference baselines roughly 5x higher than they are now. Seeds 123 and 456 still need to
be run before P1 can be signed off either way.

### Sway damping was the wrong way round (found by watching the vessel)

Reported symptom: the hull fishtails around the waypoint, gets half its length inside the
goal without the reach test firing, and circles three or four times. Measured on the
then-current checkpoint, 64 envs, 30 s:

| | measured | a displacement hull in a steady turn |
|---|---|---|
| drift angle, velocity vs bow | mean 17.4 deg, p95 35.0 | 5-10 deg |
| sway speed | mean 0.76, peak 1.53 m/s | far lower |
| surge speed | 2.37 m/s | |
| time spent turning >0.5 rad/s | 75% | |

The hull was skating, not carving. Cause: `sway_quad_damping` was 70 against
`surge_quad_damping` 110, i.e. the model resisted sideways motion *less* than forward
motion. That is unphysical for any hull — the lateral underwater area is several times
the frontal area and meets the flow bluff-on. It came straight from the Froude-scaled VRX
coefficients (`yVV=100` vs `xUU=150`), and the port had recorded "no catamaran hull-form
correction is applied" as an accepted approximation. It was not a benign one.

Fix: the standard crossflow-drag estimate for the sway quadratic term,
`0.5 * rho * Cd * A_lateral` with `Cd = 1.0` and `A_lateral = L * T = 3.0 m x 0.400 m`
(the measured equilibrium draft) `= 1.20 m^2`, giving **600**. Sway/surge quadratic is
then 5.5, inside the 3-10 band real hulls sit in.

What makes this specific to sway rather than a problem with the scaling as a whole is the
same method applied to the other two axes:

| axis | crossflow estimate | value in use | ratio |
|---|---|---|---|
| surge | 48 | 110 | 0.4x — same order, and *lower*, so the method is not inflating everything |
| yaw | 506 | 355 | 1.4x — inside the model's uncertainty |
| **sway** | **600** | **70** | **8.6x** — the outlier |

Yaw is deliberately left alone: at 1.4x it is not clearly wrong, and raising it would move
the terminal yaw rate and hence the turning radius, which is a separate open decision.

After the fix and a retrain, drift angle mean 17.4 -> **7.2 deg** (p95 35.0 -> 15.0), sway
peak 1.53 -> **0.64 m/s**, time turning hard 75% -> 36%. The straight-line and pure-turn
design points are unchanged, since sway does not enter either balance: phases 1-3 of the
physics check still give 2.500 m/s, 1.000 rad/s, 0.02 deg tilt.

A tail remains: max drift angle is 89.3 deg against a p95 of 15. Those are transients at
low speed, where a small sway velocity gives a large angle; the p95 is the number that
describes the steady behaviour.

### Resolved by the sway fix: turning radius against goal radius

The mean hides a bimodal distribution. Measured over 64 envs for 30 s on the final
checkpoint, per-env waypoint counts are:

```
count:  0   1   2   3   4   5   6
envs:  10   0   0   0   1  24  28      (+1 above 6)
```

10 of 64 envs never score. They are **not** the original pathology — the check for "parked
near the goal and barely moving" returns 0. All ten are running at full speed (2.61-2.64
m/s) with a closest approach of 3.06-4.33 m against the 3.0 m goal: they are missing, not
loitering.

**This resolved itself when the sway damping was fixed above.** Once the hull carves its
turns instead of skating through them, the effective turn radius drops to the kinematic
value and every env scores. Re-measured after the fix: 64/64 envs score, minimum 3
waypoints per env in 30 s, distribution `0 0 0 0 1 20 40 3`. The three ways out listed
below are therefore no longer needed — recorded because the reasoning about turning radius
versus goal radius still applies if the actuator sizing is ever changed.

The mechanism most consistent with the measurement is a sizing conflict I introduced.
Minimum turning radius at cruise is `v / yaw_rate = 2.556 / 1.0 = 2.56 m`, against a goal
radius of 3.0 m. A loop at that radius has a 5.1 m diameter, wider than the goal, so a
vessel that arrives at a bad angle can settle into a stable orbit just outside the success
region. The speed-coupled heading reward pays for speed, so the policy holds full throttle
and never slows down to tighten the turn (radius scales with v).

I sized `yaw_torque_max` to match the boat reference's 2.5 m turning radius without
checking it against *this* task's 3.0 m goal radius. Three ways out, none of them
obviously right, and all of them change task difficulty:

- raise `yaw_torque_max` so the turn tightens (but the hull is already at
  torque-to-weight well above the boat);
- lower `thrust_max_fwd` so cruise speed drops and the radius with it (but the task is
  meant to be a fast patrol);
- raise `goal_radius` above the turning diameter (but that changes the benchmark).

Leaving this for Yutong to decide rather than picking one, since all three move the bar.

## Wired into the shared vehicle registry

The hull's parameters were hardcoded in this task's own cfg, the way `boat_calm_nav` and
`rov_calm_nav` still do it. The newer tasks (`docking`, `station_keeping`,
`path_following`, ...) instead resolve everything from `tasks/_shared/vehicles.py`. The
catamaran now does the same: a `catamaran` entry in the registry, and a `__post_init__`
that resolves asset, mass, actuator limits, damping and restoring stiffness from it.
Swapping hull is a one-line subclass, and the current/wave variants those tasks define
apply here unchanged.

The private attitude spring is gone too, replaced by `tasks/_shared/restoring.py`'s
`restoring_torque_body(quat, k_roll, k_pitch)`. That module is yaw-invariant by
construction and takes separate roll and pitch stiffnesses, which this hull needs.

### Restoring stiffness, measured instead of inherited

The 5000 N.m/rad used until now was copied from the boat and ROV, where it is a
stabilisation device rather than a hydrostatic quantity. blueboat derives its stiffness
from CAD hydrostatics, so the same was done here, off the hull mesh:

| | |
|---|---|
| waterline that displaces m/rho = 0.12 m^3 | draft **0.353 m** |
| waterplane area | 0.402 m^2 |
| I_T / I_L | 0.0396 / 0.3121 m^4 |
| BM_T / BM_L | 0.330 / 2.600 m |
| KB | 0.195 m |
| KG, measured from the simulator | **0.50 m** (inertia diag 21.75 / 97.06 / 103.77) |

That measurement is worth stating plainly: PhysX derives this hull's inertia from a
uniform-density convex decomposition and puts the COM at mid-height, which gives
GM_T = 0.025 m and a roll stiffness of 29 N.m/rad -- a marginally stable vessel. That is
an artefact of uniform density, not a property of a catamaran, whose machinery sits low.
KG = 0.30 m is the working assumption pending an inclining test; it gives
k_roll = 265 against blueboat's 280 for a comparable catamaran, and
k_pitch = 2934. Damping follows blueboat's c ~ sqrt(k) scaling: 460, giving zeta = 3.0
against the measured 21.75 kg*m^2 roll inertia.

Four earlier attempts at this measurement were discarded, all for the same root cause:
they sliced the hull at a height that was not the waterline. `-0.400 m` is the rigid
body's root height at equilibrium, not a draft, and the sim's `rov_volume = 0.3` is a
spring parameter tuned so `0.4 * rho * g * V` balances the 1176 N weight -- not the
hull's displaced volume. Two of those attempts also measured the shell wall thickness
rather than the hull section, reporting 0.015 m demihull beams on a hull 1.101 m in beam.

### Sway recalibrated to the blueboat convention

`sway_quad` was 600, from a crossflow estimate with an assumed Cd = 1.0. blueboat states
its own sway as "lateral bluffness approximated as ~3x surge pending sway system
identification" -- an explicit placeholder, the same epistemic status as the crossflow
number and not a measurement. Since blueboat is itself a displacement catamaran and is
the best-documented entry in the registry, its convention is adopted: 3x surge, giving
195 / 330.

Measured after retraining, the change is not visible in the manoeuvring: drift angle mean
7.2 deg either way (p95 15.0 at 600, 18.9 at 330), sway peak 0.64 vs 0.68 m/s. The 3x
convention is sufficient for this hull.

## Bang-bang steering: the actuator had no dynamics

Reported from watching the vessel: it swings noticeably and often while running straight.
The hull cannot do that on its own -- yaw has damping but no restoring torque, so the
open-loop yaw dynamics are overdamped first order and cannot oscillate. Any weaving is
closed-loop, from the command.

Measured against `boat_calm_nav`'s own checkpoint on the same metrics, 64 envs, 30 s:

| | boat | catamaran, before |
|---|---|---|
| yaw command sign flips | 0.17 /s | **1.72 /s** |
| yaw rate sign flips | 0.03 /s | **1.53 /s** |
| steps with abs(a1) > 0.9 | 21.9% | 49.9% |
| straight-leg mean abs(yaw rate) | 0.287 rad/s | 0.585 rad/s |
| raw command abs(a1), p95 | 16.05 | 13.27 |

Two things separate out here. Action saturation is a benchmark-wide property, not a
catamaran defect -- boat's raw commands are larger than the catamaran's, and neither
task's reward penalises control effort. But **the weaving is specific to this task**:
boat, equally saturated, still tracks straight. A circuit of radius 12 m at 2.46 m/s needs
a steady 0.205 rad/s; the catamaran was averaging 0.585.

Two candidate explanations were checked and neither holds: boat has no action penalty in
its reward either, and its `ACTION_DELAY` defaults to 0.

The actual gap is that the task modelled no actuator dynamics at all. A real propeller
cannot reverse thrust instantly -- motor, shaft inertia and water column all take time,
and a T200/M200-class unit sits around 0.1-0.2 s -- but the model was a perfect
zero-order hold, so the policy was free to slam the command rail to rail every few steps,
and it learned to.

Fix: a first-order lag `thruster_tau` on both action channels, integrated at sim dt so the
constant is independent of decimation, and cleared on reset. This is an addition to the
model rather than a bug fix; the task never had actuator dynamics.

Tuning it mattered more than expected:

| | no lag | tau = 0.15 s | **tau = 0.06 s** |
|---|---|---|---|
| targets/episode | 19.762 | 14.344 | **20.175** |
| yaw command sign flips | 1.72 /s | 0.19 /s | 0.62 /s |
| yaw rate sign flips | 1.53 /s | 0.09 /s | 0.31 /s |
| steps with abs(a1) > 0.9 | 49.9% | 33.5% | **20.7%** |
| straight-leg abs(yaw rate) | 0.585 | 0.381 | 0.317 rad/s |
| envs scoring | 64/64 | **57/64** | 64/64 |
| worst approach | 3.017 m | **5.853 m** | 3.014 m |

0.15 s fixed the weaving outright -- sign flips landed on boat's 0.17/s -- but cost 27% of
the score, with 7 of 64 envs no longer scoring, because the hull could no longer correct
tightly near the goal. That distinguished the two explanations for the loss: at 0.06 s the
score comes back in full, so it was steering bandwidth, not an under-trained policy facing
harder dynamics.

At 0.06 s the vessel is quieter on the helm than the boat reference by the saturation
measure (20.7% against 21.9%) and within a factor of two on flip rate, while scoring
slightly above the no-lag baseline. Time spent turning hard fell from 50% to 22%.

## Known follow-ups

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
- ~~No righting moment is modelled.~~ Done during the physics port — an attitude
  spring is now in place. Measured roll under the trained policy is `-0.06 .. 0.06 deg`,
  against `std 1.9 deg, max 8.3 deg` before.

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
