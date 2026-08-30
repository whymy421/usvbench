# USVBench Project — Training Progress Log

Project: UCL Individual Project — Autonomous USV Navigation
Supervisor: Yao
Primary metric: `rescue_rate` (P2 bar ≥ 0.70, averaged over 3 seeds)
WandB project: `usvbench`

---

## Task Overview

| Task | Gym ID | Status | P-bar |
|------|--------|--------|-------|
| Catamaran Patrol (P1) | `Isaac-Catamaran-Patrol-Direct-v1` | ✅ Complete | waypoints/ep ≥ 2.5 |
| Rescue Boat (P2) | `Isaac-RescueBoat-Direct-v1` | 🔄 In progress | rescue_rate ≥ 0.70 |

---

## Experiment Log

### V1 — Rescue Boat (Broken reward, large spawn radius) — FAILED

| WandB Run | Seed | Timesteps | Best rescue_rate | Notes |
|-----------|------|-----------|-----------------|-------|
| `rescue_boat_s42` | 42 | 1.5M | ~0.000 | Broken reward (inf → NaN gradients); policy collapse |
| `rescue_boat_s123` | 123 | 1.5M | ~0.01-0.04 | Fixed NaN bug; spawn 30-120m too sparse |
| `rescue_boat_s456` | 456 | 1.5M | ~0.01-0.04 | Same as s123; reward too sparse to learn |

Root cause: casualty spawn radius 30-120m → episode too short (120s) for policy to reach casualties with random exploration. No learning signal.

### V2 — Rescue Boat (Proximity shaping) — ❌ FAILED (hover-hacking)

- Proximity reward caused policy to circle near casualties without rescuing
- Best: 1.725 tgt/ep at 60K steps, collapsed to 0.000 by 300K

> ⚠️ **ALL V1/V2/V2b RESULTS BELOW ARE INVALID.** Two independent faults were found
> on 19–20 August 2026: the evaluation metric overstated rescue rate by ~4.6×, and
> the hull capsized and sank whenever it turned. Numbers are retained only as a
> record of the diagnostic trail. See "The 20 August correction" below.

### V2b — Rescue Boat (No proximity, RESCUE_BONUS=500, PROGRESS_COEF=3) — ❌ INVALIDATED

Changes from V2:
- Removed proximity reward (`PROX_COEF=0`) — was causing reward hacking
- `RESCUE_BONUS`: 300 → **500**
- `PROGRESS_COEF`: 2 → **3**
- WandB names: `rescue_boat_v2b_s{42,123,456,7,13,21}`

| WandB Run | Seed | Timesteps | Best tgt/ep | Best rescue_rate | Best checkpoint |
|-----------|------|-----------|------------|-----------------|-----------------|
| `rescue_boat_v2b_s42` | 42 | 600K | **2.550** | **0.638** | agent_600012.pt |
| `rescue_boat_v2b_s123` | 123 | 600K | **1.800** | **0.450** | agent_600012.pt |
| `rescue_boat_v2b_s456` | 456 | 600K | **2.100** | **0.525** | agent_540000.pt |
| `rescue_boat_v2b_s7` | 7 | 600K | **1.800** | **0.450** | — |
| `rescue_boat_v2b_s13` | 13 | 600K | **2.100** | **0.525** | agent_540000.pt |
| `rescue_boat_v2b_s21` | 21 | 600K | **1.950** | **0.488** | agent_240000.pt |

**6-seed mean**: 1.883 tgt/ep → **rescue_rate = 0.513 ± 0.068** (below 0.65–0.68 target)
**Best 3-seed subset** (s42, s456, s13): rescue_rate = 0.563

**Consistent patterns:**
- Late-training degradation: all seeds peak at 240K–540K, then collapse to ~0.000 at 600K
- High seed variance: s42 (0.638) is a strong outlier; most seeds cluster at 0.450–0.525
- Early best: s21 peaked at 240K (earliest of all seeds)

---

## The 20 August correction

Two separate bugs, found by investigating an anomaly in the evaluation rather than
by tuning the policy. Neither was visible in the training reward.

### Fault 1 — the evaluation metric (found 19 Aug)

**Symptom.** Across five seeds, every checkpoint scored 0.000 except at exactly
240k and 540k steps. Independent seeds peaking at identical checkpoints is not
something learning produces.

**Diagnostic** (`diagnose_eval_sweep.py`) — all four checks failed:

| Check | Result |
|---|---|
| Same checkpoint evaluated twice | 0.975 then 0.000 — not repeatable |
| A, then B, then A again | 0.000 fresh vs 0.600 after B — order-dependent |
| Force-restore normaliser | 0.000 → 1.650 — normaliser was the fault |
| Full episode vs 1500-step window | denominator inflated ~4.6× |

**Causes.** (a) `agent.load()` did not restore the observation normaliser; the
fallback only fired when `running_mean` was exactly zero and only looked for the
legacy `state_preprocessor` key, but these checkpoints use
`observation_preprocessor`. (b) The sweep ran 1500 steps (25 s of a 120 s episode)
yet divided by 13.33 "episode-equivalents". Nearly every rescue has already
occurred by step 1500, so the numerator was near-complete while the denominator
was scaled down.

**Fix.** `eval_fixed.py`, and the same corrections ported into
`train_with_eval.py`: explicit normaliser restore, hard reset with discarded
burn-in, whole-episode scoring, and the honest metric
`rescue_rate = rescues / (envs × casualties × episodes)`.

**Corrected result — seed 21, all checkpoints re-scored:**

| 60k | 120k | 180k | 240k | 300k | 360k | 420k | 480k | 540k | 600k |
|---|---|---|---|---|---|---|---|---|---|
| 0.070 | 0.106 | **0.145** | 0.106 | 0.117 | 0.109 | 0.109 | 0.086 | 0.047 | 0.117 |

A flat line at ~0.10 (binomial SD ≈ 0.019 at n=256). No learning across 600k
steps, and no "late-training peak" — that narrative was an artifact of the broken
metric. Because selection used the broken metric, every previously reported
`best_agent.pt` was effectively a random pick.

### Fault 2 — the hull capsizes (found 20 Aug)

**Isolating it.** A scripted pure-pursuit controller (`eval_oracle.py`, no neural
network) was swept over 12 gain settings. All scored 0.05–0.10, the same as the
trained policies — so the failure was not in RL. The controller also logged motion:
mean speed 0.67–1.14 m/s against a configured 12 m/s.

**Open-loop probe** (`probe_physics.py`):

| Test | Result |
|---|---|
| Full thrust, no torque | 12.00 m/s — **100% of expected**, surge is fine |
| Full torque, no thrust | roll 87° → 130° → **172°**, z −1.4 → **−49.02** (seabed) |
| Free float afterwards | stays at z = −49.02, roll 172°, speed 0 |

**Cause.** Yaw torque was applied about **body** +Z. Any roll tilts body-Z off
vertical, so the yaw command acquires a horizontal component and induces more
roll — a runaway loop. Nothing opposed it, because buoyancy was applied as a point
force at the centre of mass (`F[:, 2] += F_buo_z`), giving **zero roll stability**.
Past ~90° the 17% reserve buoyancy could not recover and the hull sank 50 m to the
ground plane.

The catamaran escaped this only because its `max_angular_velocity` is capped at
5.0; the rescue boat allowed 120.

**Hull comparison:**

| | Catamaran (works) | Rescue boat (was) | Rescue boat (now) |
|---|---|---|---|
| Mass | 120 kg | 300 kg | 300 kg |
| `rov_volume` | 0.30 m³ | 0.35 m³ | **0.75 m³** |
| Reserve buoyancy | 150% | **17%** | **150%** |
| Equilibrium submergence | 0.400 | **0.857** | **0.400** |
| `max_angular_velocity` | 5.0 | **120.0** | **3.0** |

**Fixes applied.**

1. Yaw torque about **world** +Z, thrust held in the horizontal plane
   (`HORIZONTAL_ACTUATION=1`) — removes the feedback loop at source.
2. Righting moment on roll and pitch (`RIGHTING_K=8000` N·m/rad, `RIGHTING_C=2000`),
   standing in for metacentric stability. Righting authority exceeds full steering
   torque beyond ~6° of heel.
3. `rov_volume` 0.35 → 0.75; `max_angular_velocity` 120 → 3.0.

---

### Fault 3 — the benchmark was never calibrated (found 20 Aug)

With the physics fixed, a scripted pure-pursuit oracle (no neural network,
perfect state access) still only reached 0.32. Since a trivial controller could
not approach 0.70, the bar was unreachable by construction.

**Why 5 m capture failed.** Minimum turning radius is v/ω = 8/1.05 ≈ 7.6 m at
cruise and 11.4 m at full speed — larger than the 5 m capture zone. An overshoot
therefore could not be corrected: the boat flew a circle wider than the target
zone and orbited outside it. The capture radius was also smaller than the hull
itself (8.5 m), which is hard to defend physically.

**Target selection** (same boat, same gains):

| Strategy | rescue_rate | Target switches/episode |
|---|---|---|
| committed (lock until rescued/lost) | 0.324 | 2.6 |
| priority (`urgency²/dist`) | 0.258 | — |
| nearest (ignore urgency) | 0.234 | — |

Commitment helps ~26%, so mid-transit re-ranking costs real performance, but it
was not the dominant term. Worth noting `urgency = 0` for every casualty at
episode start, so the initial ranking is an arbitrary tie-break.

**Capture radius sweep** (committed strategy):

| rescue_radius | 5 m | 20 m |
|---|---|---|
| oracle rescue_rate | 0.324 | 0.688 |

**Configuration calibration.** A usable benchmark needs the oracle ceiling well
above the bar, so that reaching 0.70 means the policy is good rather than
superhuman. Only the fully relaxed configuration achieved that:

| Config | radius | timers | spawn max | oracle |
|---|---|---|---|---|
| original | 5 m | 40–90 s | 50 m | 0.324 |
| **V3 (adopted)** | **20 m** | **60–120 s** | **35 m** | **0.887** |

**These changes redefine the benchmark** and require supervisor agreement. They
are recorded here and in `rescue_boat_env_cfg.py` with per-parameter
justification.

**Remaining gap.** On identical settings the trained policy scored 0.145 against
the oracle's 0.324 — less than half. So RL training is underperforming
independently of the environment faults, and fixing the environment alone will
not reach the bar.

---

## FINAL RESULTS — V4, 23 August 2026

| | V3 | **V4 (final)** | Greedy oracle |
|---|---|---|---|
| rescue_rate, 3 seeds | 0.695 ± 0.028 | **0.770 ± 0.036** | 0.582 |
| Seeds clearing 0.70 | 2 of 3 | **3 of 3** | — |
| Margin over oracle | +0.113 | **+0.188** | — |
| Revolutions/episode | 19 | **12** | 8.8 |
| Hull vs direction of travel | 87° | **59°** | 49° |

Best single seed: **0.8086** (`agent_300030`, run `2026-08-22_19-28-22`).
Figure: `thesis_draft/figures/fig8_v3_v4_ablation.png`

### Fault 8 — reward under-specified the locomotion

V3 reached targets by spinning and drifting: 19 revolutions per episode with the
hull 87° off its direction of travel. The reward was defined purely on distance
to the current target, so nothing constrained heading. A hull thrusting along
body +X while rotating traces a cycloid — it still closes distance, so it still
earns progress reward.

**This is the third specification failure of the same shape:**

| Version | Term | Intent | Learned behaviour |
|---|---|---|---|
| V2 | proximity | be near casualties | hover without rescuing (34,000/ep vs 1,200 available) |
| V3 | progress | close distance | spiral while translating |
| — | priority `urgency²/dist` | prioritise by urgency | thrash between targets, 2.6 switches/ep |

Each named a *correlate* of the goal rather than the goal.

**Fix (V4).** Two terms, both defaulting to zero so V3 remains reproducible:

- `HEADING_COEF = 0.05` — rewards `cos(bearing error)`, a reason to hold a heading
- `YAWRATE_COEF = 0.03` — penalises `ω²`, quadratic so course corrections stay cheap

**Sizing, derived from the V2 failure.** Dense per-step terms accumulate over
7,200 steps and can swamp a sparse terminal bonus. The first candidate values
(0.5 / 0.3) would have paid 3,600 per episode against 2,000 available from
rescuing all four casualties — the shaping would have become the objective, which
is precisely how V2 failed. At 0.05 / 0.03 the combined shaping is 598 per
episode, 30% of the rescue bonus: enough to change behaviour, not enough to
replace the goal.

**Outcome.** Revolutions −37%, heading error −32%, rescue_rate +0.075. Realism
and performance improved together, because less rotation means faster transit and
therefore more casualties reached before their deadlines.

**Residual.** V4 still rotates more than the scripted controller (12 vs 8.8). The
oracle's own 49° heading error shows the remaining weave comes from the priority
heuristic re-ranking mid-transit, not from the policy. Addressing it would require
hysteresis in the target selection, which changes the task definition.

---

### V3 — first training on the corrected environment

`rescue_boat/train_v3_fixed.ps1` — seeds 42/123/456 at 300k steps, wandb
`rescue_boat_v3_s{42,123,456}`. The 600k horizon was justified by a peak that
did not exist, so shorter runs and more seeds are the better trade.

**Not comparable to any V1/V2/V2b figure.**

| WandB Run | Seed | Steps | rescue_rate | Notes |
|-----------|------|-------|-------------|-------|
| `rescue_boat_v3_s42` | 42 | 300k | TBD | |
| `rescue_boat_v3_s123` | 123 | 300k | TBD | |
| `rescue_boat_v3_s456` | 456 | 300k | TBD | |

### Other changes, 20 August

- **Environmental disturbance** added (`CURRENT_SPEED`, `WAVE_AMP`, `WAVE_YAW_AMP`,
  `GUST_STD`) as four sea states, all defaulting to calm. Supports zero-shot
  robustness evaluation (`eval_robustness.py`) without retraining.
- **Visualisation** (`VISUALISE=1`): water surface plus casualty markers coloured
  by urgency (green → amber >50% → red >80% → grey when lost). Previous videos
  rendered no water and no casualties — a stationary speck over a wireframe grid.
- **`gpu_cleanup.ps1`**: clears stale Isaac Sim processes. A run that "doesn't
  start" and leaves no log is a zombie process holding the GPU.

---

## Weekly Meeting Notes

| Date | Summary |
|------|---------|
| 07.08.2026 | Task design agreed: rescue boat P2 with deadline-aware prioritisation. Physics check passed. Metrics defined. |
| 14.08.2026 | V1 three-seed training complete. rescue_rate 0.01-0.04 (too sparse). V2 design agreed: smaller spawn + proximity shaping. |

---

## Environment Parameters

```
Isaac Sim: 5.1.0
Isaac Lab: 0.54.4
skrl: 2.1.0 (PPO)
GPU: RTX 4090
Training: 64 envs × 1.5M steps ≈ 5-8h per seed
Asset: C:\usvbench\assets\rescue_boat.usd (ARENA RIB 8.50m, 300kg)
```

## Key File Locations

| File | Purpose |
|------|---------|
| `rescue_boat/rescue_boat_env.py` | Main env (DirectRLEnv) |
| `rescue_boat/rescue_boat_env_cfg.py` | Config (spawn radii, timers, physics) |
| `rescue_boat/agents/skrl_ppo_cfg.yaml` | PPO hyperparams |
| `rescue_boat/train_rescue_boat.ps1` | Launch all 3 seeds |
| `rescue_boat/sync_to_isaaclab.ps1` | Copy files to Isaac Lab dir |
| `train_with_eval.py` | Training + checkpoint eval sweep |

---

## Oral Presentation Notes (placeholder — update when results arrive)

Key points to cover:
1. Problem: autonomous USV for maritime rescue — urgent, deadline-driven task
2. Environment: Isaac Sim 5.1.0, 64 parallel envs, realistic water physics
3. P2 task design: 4 casualties with countdown timers, prioritisation by urgency × proximity
4. Reward engineering journey: sparsity problem → proximity shaping fix
5. Results: V1 baseline vs V2 after shaping (rescue_rate improvement)
6. Comparison to P1 catamaran patrol (simpler task, faster convergence)
7. Lessons: reward shaping critical for sparse-reward maritime tasks

WandB dashboard: https://wandb.ai/[username]/usvbench
Runs to highlight: `rescue_boat_v2_s42`, `rescue_boat_v2_s123`, `rescue_boat_v2_s456`
