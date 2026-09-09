# Task B — Boat Calm-Water Navigation

> **Vessel**: 5 m monohull (`boat_physics.usdc`)
> **Condition**: calm water (no waves, no current)
> **Algorithm**: PPO (single agent)
> **Status**: ✅ validated — V26 = **3.75** targets/episode on the deterministic eval
> (`boat_calm_v26_s42.pt`, final checkpoint of a 3000-iteration seed-42 run).
> Baselines are maintained in [`TASKS.md`](../../TASKS.md#current-baselines).
> **Gym id**: `Isaac-My-First-Task-Calm-Boat-Direct-v1`
> **wandb**: https://wandb.ai/whymysong321-university-of-southampton/usvbench/runs/si1f8sq1

Same task as Task A (point navigation, calm water) but on a **real boat hull**. The
boat is harder: its mass distribution and body-axis orientation differ from the ROV,
so the simple ROV reward does **not** work here. This task ships with the tuned
"side-approach" reward (V40) that solves it.

---

## ⚠️ Read this first — why the reward is different from Task A

The ROV trains fine with `forward_speed × exp(alignment)`. On the boat that recipe
**fails** (the boat learns to point ~150° away from the target and barely scores) because:

- `exp(alignment)` is always positive → moving *backwards* still earns reward.
- The boat's body axes are non-standard (`body-X = stern`, `body-Y = starboard`).

The fix (V26): **speed-coupled reward** (`REWARD_VARIANT=V23`, `SPEED_COUPLE=1`) —
the heading reward is multiplied by forward speed (so pointing the right way only pays
off while actually moving forward), plus a large reach bonus (+50) to beat the cost of
travelling to a freshly-spawned target. See `my_first_task_env.py` `_get_rewards()`,
the `V23` + `speed_coupling` branch.

> There is a variant, **V40**, that adds `SIDE_APPROACH=1` to drop the bow-alignment
> requirement. On the pre-2026-07-31 physics it scored ~18% higher than V26, but the boat
> learned to *reverse* into targets, which looks unnatural, so V26 stays the reference.
> **That comparison has not been re-run on the current physics** — the old figures (V40
> 5.64, V26 4.76) came from a model with a different damping and actuator system, so no
> number is quoted here until someone re-measures. If you want to try V40, add
> `SIDE_APPROACH=1` to the env vars below and report what you get.

---

## What it is

| Spec | Value |
|------|-------|
| Observation | 9D — nav (3) + self-state (speed, yaw-rate, etc.) via `OBS_EXTENDED=1` |
| Action | 2D — `(forward_thrust, yaw_torque)`, both in [-1, 1] |
| Reward | V23 speed-coupled nav + reach (`REACH_BONUS=50`) |
| Episode | 120 s, continuous re-targeting |

---

## Install

1. Copy this folder and the shared modules into your Isaac Lab tasks directory:
   ```
   <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/boat_calm_nav/
   <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/_shared/
   ```
2. Asset path: handled by `USVBENCH_ASSETS` (train script sets it to `<repo>/assets`).
3. Activate your Isaac Lab conda env.

---

## Run

**Easiest** — use the provided script (edit the `$ISAACLAB` line first):
```powershell
.\scripts\train_boat_calm.ps1
```

**Manual** — the exact env vars that define V26:
```bash
OBS_DIM=9 OBS_EXTENDED=1 REWARD_VARIANT=V23 SPEED_COUPLE=1 \
REACH_BONUS=50.0 \
python <IsaacLab>/scripts/reinforcement_learning/skrl/train_with_eval.py \
  --task=Isaac-My-First-Task-Calm-Boat-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42 \
  --video --video_interval 50000 --video_length 200
```
**Do NOT set `FORWARD_TRANSIT=1`** — that was experiment V41 and it made the boat flee
the target almost completely. (`SIDE_APPROACH=1` gives V40, which scores higher but
reverses into targets — see the note above. Neither variant has been re-measured on the
current physics.)

Takes ~30 min on an RTX 5080 (measured 2026-07-31, 3000 iterations, 64 envs).

---

## Play my trained checkpoint (no training needed)

The V26 reference policy is shipped at `checkpoints/boat_calm_v26_s42.pt`:
```bash
USVBENCH_ASSETS=<repo>/assets OBS_DIM=9 OBS_EXTENDED=1 REWARD_VARIANT=V23 SPEED_COUPLE=1 REACH_BONUS=50.0 \
python <IsaacLab>/scripts/reinforcement_learning/skrl/play.py \
  --task=Isaac-My-First-Task-Calm-Boat-Direct-v1 --num_envs=16 \
  --checkpoint=<repo>/tasks/boat_calm_nav/checkpoints/boat_calm_v26_s42.pt
```

---

## Evaluate (standardized protocol)

Score any checkpoint with the benchmark eval (deterministic policy, fixed budget):
```bash
USVBENCH_ASSETS=<repo>/assets OBS_DIM=9 OBS_EXTENDED=1 \
python scripts/eval_benchmark.py --task=Isaac-My-First-Task-Calm-Boat-Direct-v1 \
  --num_envs=64 --eval_steps=6000 --headless \
  --checkpoint=<repo>/tasks/boat_calm_nav/checkpoints/boat_calm_v26_s42.pt
```
Add `--seed 2026` to match [`TASKS.md`](../../TASKS.md#current-baselines).
Reference: `targets_per_episode` = **3.75** (seed 42). Note: the boat has higher
seed-variance than the ROV — see *Known limitation* below.

---

## What success looks like

Reference: V26 (`boat_calm_V26_speedcouple_s42`, wandb `si1f8sq1`).

| Metric | Reference | Meaning |
|--------|-----------|---------|
| `Metrics/targets_per_episode` | training-time metric, stochastic policy | boat reaches several targets per episode |
| `Nav/speed` | ~5 m/s (cap) | boat is moving at full cruise |
| `Episode / Total timesteps (mean)` | ~6700 / 7200 | survives most of the episode |

Reproduce within ~20%. With V26 the boat navigates **bow-first** toward targets
(unlike the V40 variant, which reverses in).

**If you land more than ~20% below the reference in [`TASKS.md`](../../TASKS.md#current-baselines), or the boat points consistently away → message Yutong.**

---

## Env-var toggles

| env var | default here | purpose |
|---------|--------------|---------|
| `OBS_DIM` | `9` | obs size (nav + self-state) |
| `OBS_EXTENDED` | `1` | enable self-state channels |
| `REWARD_VARIANT` | `V23` | speed-coupled nav reward |
| `SPEED_COUPLE` | `1` | reward only when moving |
| `REACH_BONUS` | `50` | per-target reward (must be large for the slow boat) |
| `SIDE_APPROACH` | unset | set to `1` for the V40 variant (reverses into targets; not re-measured on the current physics) |
| `FORWARD_TRANSIT` | unset | ⚠️ leave unset (V41 regression) |

---

## Known limitation — an open problem (optional to improve)

The boat is the **weak spot** of the benchmark: 3.75 targets/ep vs the ROV's 4.48. It
took a lot of reward tuning (V11→V26→V40) just to get here, and neither baseline is
clean: V26 navigates bow-first but is slow/modest, V40 scores higher but reverses into
targets. Root causes are the boat's non-marine-grade physics (mass/volume are a float-
balance hack, non-standard body axes) and the fact that the simple ROV reward fails on it.

**If you have ideas to make the boat navigate more cleanly/efficiently** (better reward,
obs, or physics), feel free to try — it'd be a welcome contribution. But this is
**optional**: it's not your assigned task, so don't sink days into it. Reproducing the
V26 baseline is all P0 requires; your real deliverables are the catamaran and cruise-ship
tasks (see `docs/ARIF_TASKS.md`).
