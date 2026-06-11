# Task B — Boat Calm-Water Navigation

> **Vessel**: 5 m monohull (`boat_physics.usdc`)
> **Condition**: calm water (no waves, no current)
> **Algorithm**: PPO (single agent)
> **Status**: ✅ validated — V26 = 4.76 targets/episode @ 3000 iter (Yutong, seed 42)
> **Gym id**: `Isaac-My-First-Task-Calm-Boat-Direct-v1`
> **wandb**: https://wandb.ai/whymysong321-university-of-southampton/usvbench/runs/si1f8sq1

Same task as Task A (point navigation, calm water) but on a **real boat hull**. The
boat is harder: its mass distribution and body-axis orientation differ from the ROV,
so the simple ROV reward does **not** work here. This task ships with the tuned
"side-approach" reward (V40) that solves it.

---

## ⚠️ Read this first — why the reward is different from Task A

The ROV trains fine with `forward_speed × exp(alignment)`. On the boat that recipe
**fails** (the boat learns to point ~150° away from the target, ~0.1 tgt/ep) because:

- `exp(alignment)` is always positive → moving *backwards* still earns reward.
- The boat's body axes are non-standard (`body-X = stern`, `body-Y = starboard`).

The fix (V26): **speed-coupled reward** (`REWARD_VARIANT=V23`, `SPEED_COUPLE=1`) —
the heading reward is multiplied by forward speed (so pointing the right way only pays
off while actually moving forward), plus a large reach bonus (+50) to beat the cost of
travelling to a freshly-spawned target. See `my_first_task_env.py` `_get_rewards()`,
the `V23` + `speed_coupling` branch.

> There's a higher-scoring variant, **V40** (5.64 tgt/ep), that adds `SIDE_APPROACH=1`
> to drop the bow-alignment requirement. It scores ~18% higher but the boat learns to
> *reverse* into targets, which looks unnatural. We use **V26** as the reference because
> bow-forward navigation is the cleaner baseline. If you want to reproduce V40, add
> `SIDE_APPROACH=1` to the env vars below.

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

1. Copy this folder into your Isaac Lab tasks directory:
   ```
   <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/boat_calm_nav/
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
**Do NOT set `FORWARD_TRANSIT=1`** — that was experiment V41 and it made the boat
flee the target (0.04 tgt/ep). (Adding `SIDE_APPROACH=1` gives V40, ~5.6 tgt/ep, but
the boat reverses into targets — see the note above.)

Takes ~1.5 h on an RTX 5080.

---

## What success looks like

Reference: V26 (`boat_calm_V26_speedcouple_s42`, wandb `si1f8sq1`).

| Metric | Reference | Meaning |
|--------|-----------|---------|
| `Metrics/targets_per_episode` | **~4.8** | boat reaches several targets per episode |
| `Nav/speed` | ~5 m/s (cap) | boat is moving at full cruise |
| `Episode / Total timesteps (mean)` | ~6700 / 7200 | survives most of the episode |

Reproduce within ~20%. With V26 the boat navigates **bow-first** toward targets
(unlike the V40 variant, which reverses in).

**If `targets_per_episode` < 2 or the boat points consistently away → message Yutong.**

---

## Env-var toggles

| env var | default here | purpose |
|---------|--------------|---------|
| `OBS_DIM` | `9` | obs size (nav + self-state) |
| `OBS_EXTENDED` | `1` | enable self-state channels |
| `REWARD_VARIANT` | `V23` | speed-coupled nav reward |
| `SPEED_COUPLE` | `1` | reward only when moving |
| `REACH_BONUS` | `50` | per-target reward (must be large for the slow boat) |
| `SIDE_APPROACH` | unset | set to `1` for the V40 variant (reverses in, ~5.6 tgt/ep) |
| `FORWARD_TRANSIT` | unset | ⚠️ leave unset (V41 regression) |
