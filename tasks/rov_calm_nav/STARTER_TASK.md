# Task A — ROV Calm-Water Navigation

> **Vessel**: Doc Ricketts ROV (`ROV_rigged.usd`)
> **Condition**: calm water (no waves, no current)
> **Algorithm**: PPO (single agent)
> **Status**: ✅ validated — 24 targets/episode @ 3000 iter (Yutong, seed 42)
> **Gym id**: `Isaac-My-First-Task-Calm-Direct-v1`

This is the **easiest task in USVBench** and the recommended Day-1 warmup. The ROV
trains cleanly with the simplest possible reward, so you can confirm your whole
Isaac Lab + skrl pipeline works before touching anything harder.

---

## What it is

A single ROV navigates to a point target on flat water. When it reaches the target,
a new target spawns — so a good policy reaches **many** targets per episode.

| Spec | Value |
|------|-------|
| Observation | 3D — `(dot, cross, dist_norm)` (heading-vs-target alignment + normalized distance) |
| Action | 2D — `(forward_thrust, yaw_torque)`, both in [-1, 1] |
| Reward | `forward_speed × exp(alignment) + reach_bonus` (the "E7" variant, default) |
| Episode | 120 s, continuous re-targeting |

---

## Install

1. Copy this folder into your Isaac Lab tasks directory:
   ```
   <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/rov_calm_nav/
   ```
2. The vessel USD is auto-located via `USVBENCH_ASSETS` (the train script sets it
   to `<repo>/assets` for you). If you run manually, set it yourself:
   ```powershell
   $env:USVBENCH_ASSETS = "C:\path\to\usvbench\assets"   # PowerShell
   export USVBENCH_ASSETS=/path/to/usvbench/assets        # bash
   ```
3. Activate your Isaac Lab conda env (`conda activate isaaclab`).

---

## Run

**Easiest** — use the provided script (edit the `$ISAACLAB` line at the top first):
```powershell
.\scripts\train_rov_calm.ps1
```

**Manual** — the exact command (cross-platform):
```bash
OBS_DIM=3 \
python <IsaacLab>/scripts/reinforcement_learning/skrl/train_with_eval.py \
  --task=Isaac-My-First-Task-Calm-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42 \
  --video --video_interval 50000 --video_length 200
```
(On Windows PowerShell set `$env:OBS_DIM="3"` on its own line instead of the inline prefix.)

Takes ~30 min on an RTX 5080.

---

## What success looks like

Reference run (Yutong, seed 42, 3000 iter): **`rov_calm_benchmark_s42`** on the
`usvbench` wandb project.

| Metric | Reference | Meaning |
|--------|-----------|---------|
| `Metrics/targets_per_episode` | **~24** | reaches a new target every ~5 s |
| `Reward / Instantaneous reward (mean)` | ~15 | policy strongly navigating |
| `Nav/distance` | < 17 m | consistently closing on targets |
| `Nav/speed` | > 4 m/s | moving fast, not stuck |

You should reproduce within ~20%. Watch the auto-uploaded wandb videos to confirm
the ROV drives smoothly to targets (not spinning in place).

**If `targets_per_episode` < 5 or `speed` ≈ 0 after 3000 iter → message Yutong before continuing.**

---

## Env-var toggles

| env var | default | purpose |
|---------|---------|---------|
| `OBS_DIM` | `3` | observation size (keep 3 for this task) |
| `REWARD_VARIANT` | `E7` | leave unset; `E7` = the working recipe |
| `REACH_BONUS` | `10` | reward per target reached |
| `USVBENCH_ASSETS` | `~/usvbench/assets` | where the vessel USD lives |
