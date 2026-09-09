# Task B: Ordered Harbor Mission (C7)

> **Gym id:** `Isaac-USV-HarborMission-Direct-v1`  
> **Environment:** `HarborMissionEnv` (`DirectRLEnv`)  
> **Default vehicle:** `vehicle="blueboat"`, resolved through
> `tasks/_shared/vehicles.py`  
> **Condition:** calm water, fixed mid-tier analytic hazards

## Mission and fixed horizon

Every episode lasts exactly 240 s. Physics runs at 120 Hz (`dt=1/120`) with
decimation 2, so control runs at 60 Hz. `terminated` is always false and the
time-limit truncation is the only done signal. Completing the mission changes
the monotone task state to `done`, but never ends the episode early.

The local route is axis-aligned along `+X` and is generated independently per
environment:

1. **Exit:** two 6 m gates at `x=5 m` and `x=15 m`; the boat must cross gate 1
   outward before gate 2.
2. **Hazard transit:** the obstacle field begins at `x=20 m`, spans a sampled
   25--35 m, and ends at a 6 m directional exit gate. The field reuses
   `hazard_geometry.sample_layout(level=1)`, contains 6--10 cylinders (normally
   the mid-tier eight), maps radii into 0.8--1.8 m, forces a direct-line
   blocker, and re-runs the inflated-cylinder BFS feasibility oracle after the
   harbor transformation.
3. **Dock:** the berth lies 15--25 m beyond the field exit, with heading `+X`.

The berth is therefore 60--80 m from spawn. All three gate segments are
checked against inflated cylinders, and a 5 m protected berth disk is checked
at generation time.

## Explicit monotone task automaton

The task state is stored by the environment and can only move to the right:

```text
                     valid gate 1        valid gate 2
q=0 EXIT, gate=1  ----------------->  q=0 EXIT, gate=2  -----------+
                                                                  |
                                                                  v  M_exit
                         valid field-exit crossing          q=1 TRANSIT
                    +----------------------------------------------+
                    |
                    v  M_transit
                q=2 DOCK  -- docking conjunction held 5 s -->  q=3 DONE
                                                           M_dock

Any invalid/out-of-order/backward crossing: self-loop
Any later backtracking across a completed boundary: self-loop
q=3: absorbing task state (episode clock continues)
```

For the currently active gate, a latch first records that the trajectory has
visited at least 1 m before the gate plane. A crossing then requires adjacent
previous/current COM samples to straddle the finite gate line segment, the
linearly interpolated intersection to lie between its endpoints, and current
velocity along the outward normal to be strictly greater than 0.2 m/s.
Backward crossings, slow drift, endpoint misses, and out-of-order gates do not
advance the automaton. The separate approach latch preserves the required 1 m
pre-crossing evidence without requiring an impossible 1 m displacement during
one 1/60 s control transition.

### Why position-triggered stage switching was rejected

The C7 adversarial design debate rejected rules such as "enter a radius" or
"be past an X boundary" as stage transitions. Two identical physical states
can require different next actions depending on which ordered milestones were
already achieved; physical position alone therefore aliases task state.
Reversible boundary predicates also permit boundary farming: oscillating
around a switch can repeatedly collect stage progress or transition bonuses.
The explicit irreversible automaton makes milestone history part of state,
admits no reverse transition, and evaluates progress using the phase active at
the start of the step.

## Docking conjunction

During `q=2`, all three instantaneous conditions must hold simultaneously:

```text
COM distance to berth <= 2.5 m
absolute bow error from dock heading <= 15 degrees
planar COM speed <= 0.3 m/s
```

They must remain true for 5 consecutive seconds (300 control samples).
Breaking any condition resets the current dwell timer to zero. Completing the
streak changes `q` to 3 exactly once.

## Markov augmented observation

The policy observation has exactly 46 values:

```text
3 current-target values:
    target_dot, target_cross, clamp(distance/current_phase_span, 0, 1)
36 analytic cylinder rays:
    full 360 degrees, 10 degree spacing, range/30 m
4 task-state values:
    one_hot(q), q in {exit, transit, dock, done}
2 dock alignment values:
    dock_dot, dock_cross in q=2; zero otherwise
1 dwell value:
    current_hold_time/5 s in q=2; zero otherwise
```

The current target is exit gate 1 or 2 in `q=0`, the field-exit midpoint in
`q=1`, and the berth point in `q=2`. The done-state target triplet is zero.
Rays reuse HazardNav's analytic ray/circle code and see obstacle cylinders
only. The active target triplet distinguishes the two exit substates.

This is a memoryless policy over the **augmented** task state: the physical
state is supplemented with irreversible phase/active-target information and
the otherwise history-dependent dwell fraction. This follows the state/time
augmentation principle discussed by Pardo et al., *Time Limits in
Reinforcement Learning* (ICML 2018): hidden clocks or task memory must be made
state when they affect transitions. V1 keeps the fixed episode clock as an
external truncation rather than a success transition; remaining-horizon input
is an explicit upgrade if finite-horizon time-dependent policies are studied.

## Ledger-capped reference reward

Let the stage longitudinal intervals be spawn-to-gate-2, gate-2-to-field-exit,
and field-exit-to-berth. For phase `q` in `{0,1,2}`:

```text
Phi_q(s) = clamp((COM_x - stage_start_x) / stage_span, 0, 1)
r_prog = (1/3) * (Phi_q(s') - Phi_q(s))
```

The implementation equivalently stores `Phi_q/3` and multiplies its delta by
`progress_scale=20`. The phase is sampled at the **start** of the transition,
so a gate crossing receives only old-phase progress. After the automaton
advances, the next phase's potential baseline is initialized at the crossing
state; there is no target-switch jump. Each phase contributes at most `20/3`,
and total mission potential progress is at most 20.

With analytic hull clearance
`d_min = min_i(||COM_xy-c_i|| - radius_i - 0.45)`, the full control-step reward
before done is:

```text
20 * r_prog
- 1.0 * dt_ctrl * clamp((0.9 - d_min)/0.9, 0, 1)^2
- 25 * 1[analytic contact entry]
- 1  * 1[analytic contact dwell sample]
+ 3  * 1[q=2 and the instantaneous docking conjunction holds]
```

This copies HazardNav v3's event-contact ledger and exact scales. A contact
entry dominates the full capped progress ledger. Dock pay stops on the sample
after the 5 s dwell completes because `q=3` is absorbing. In `q=3`, progress,
contact entry/dwell, and dock pay are all frozen to zero; only the continuous
clearance barrier remains, so a drifting completed boat has nothing left to
farm.

## Reward-free, log-judged success

Let `M_exit`, `M_transit`, and `M_dock` be their first achievement samples.
Success is judged from the event ledger, not reward:

```text
success = (M_exit < M_transit < M_dock)
          AND no analytic obstacle contact on [episode start, M_dock]
```

The contact prefix includes the sample on which docking dwell completes.
Contact after `M_dock` cannot revoke success. Every episode still runs for the
full horizon.

Before auto-reset the environment snapshots these evaluator-facing fields:

- `episode_success`;
- `time_to_success` (the time `q` first hit 3; NaN on failure);
- `stage_reached` (`0..3` at episode end);
- `episode_path_length` and `episode_min_clearance`, accumulated through the
  `M_dock` prefix or the full episode when docking never completes;
- `goal_radius=2.5` for evaluation compatibility;
- per-stage booleans `m1`, `m2`, and `m3`.

The episode logger reports `P(M1)`, `P(M2|M1)`, and `P(M3|M2)`. These
conditional metrics localize whether a method fails at ordered exit, hazard
transit, or docking instead of collapsing the ladder into one success number.

## Curriculum decision

V1 has no within-task curriculum. This is deliberate: the benchmark task
ladder is the curriculum, and exit control, HazardNav, and docking are
separately trainable components. A future version may add an explicit
stage-unlock curriculum (exit, then transit, then dock), but it must retain the
same final route distribution and be reported separately from direct V1
training.

## Render-only visuals and V1 simplifications

The guarded visual layer contains flat water, paired vertical posts plus a thin
water-level crossbar for every gate, gray obstacle cylinders, and Docking's
berth ring, `+X` heading arrow, and berth-box outline. Every movable USD prim
uses the get-or-add translate-op helper. No visual state feeds physics,
observations, reward, or scoring.

V1 deliberately uses analytic, non-colliding obstacles; noiseless 36-ray
sensing; calm water; an axis-aligned local route; one fixed mid-tier hazard
distribution; and no current, waves, traffic, range dropout, or curriculum.
Upgrade paths are physical collider/contact parity, calibrated ray noise and
dropout, randomized route heading, current/wave disturbances, moving traffic,
and the separately reported stage-unlock curriculum.

## Bidirectional certificate plan

Mission claims should include two negative-transfer certificates in addition
to a trained mission policy:

1. Evaluate the HazardNav champion unchanged. It should clear transit often
   enough to prove perception transfer, then fail in the dock phase because it
   lacks align/brake/hold competence.
2. Evaluate the Docking champion unchanged. It should demonstrate docking
   competence when delivered near the berth, but die in transit because its
   observation/policy lacks hazard-field avoidance.

Report stage-conditionals for both. The paired failures certify that Harbor
Mission is a genuine ordered conjunction rather than a renamed copy of either
component task.

## Verify and train

The geometry test requires only NumPy and Torch:

```powershell
python .\tasks\harbor_mission\test_mission_geometry.py
```

In an Isaac Lab environment:

```bash
python <IsaacLab>/scripts/reinforcement_learning/skrl/train.py \
  --task=Isaac-USV-HarborMission-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42
```

The copied HazardNav PPO recipe uses `initial_log_std=-1.9` and writes under
`experiment_name=harbor_mission_v1`.

## v1 campaign ledger (honest)

3000-iter: P(M1) peaked 0.61. 9000-iter: P(M1) reached 0.94 but
P(M2|M1) stayed exactly 0.00 for 288k steps. Diagnosis probe: blind
full-throttle reaches phase 2 in <30 s in 8/8 envs -- the automaton is
sound end-to-end; the trained policy (correctly) refuses to plow
through the field because success forbids contact, and clean transit
embeds the hazard-navigation capability, itself the open-problem tier
(14.8% standalone). The composed task is capability-gated exactly as
the bidirectional-certificate model predicts: solving stage 2 requires
solving hazard_nav first. RL reference pending that capability;
cross-champion certificates additionally await the frozen observation
superset (component champions have incompatible obs dims).
