# Task A — ROV Calm-Water Navigation

> **Vessel**: Doc Ricketts ROV (`ROV_rigged.usd`)
> **Condition**: calm water (no waves, no current)
> **Algorithm**: PPO (single agent)
> **Status**: current-physics baseline validated at 6.863 targets/episode (seed 42)
> **Gym id**: `Isaac-USVBench-ROV-Calm-Direct-v1`

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

**Easiest** - use the reproducible current-physics launcher:
```powershell
.\scripts\train_rov_baseline_current.ps1 `
  -IsaacLabRoot "C:\path\to\IsaacLab" `
  -Seed 42 -MaxIterations 3000 `
  -EvalSweepSteps 1500 -BenchmarkSteps 6000 -EvalSeed 2026
```

**Manual** — the exact command (cross-platform):
```bash
OBS_DIM=3 \
python <IsaacLab>/scripts/reinforcement_learning/skrl/train_with_eval.py \
  --task=Isaac-USVBench-ROV-Calm-Direct-v1 \
  --num_envs=64 --headless --max_iterations=3000 --seed=42 \
  --video --video_interval 50000 --video_length 200
```
(On Windows PowerShell set `$env:OBS_DIM="3"` on its own line instead of the inline prefix.)

Takes ~30 min on an RTX 5080.

---

## Play my trained checkpoint (no training needed)

Use `checkpoints/rov_calm_current_s42.pt` with the current task code:
```bash
USVBENCH_ASSETS=<repo>/assets OBS_DIM=3 \
python <IsaacLab>/scripts/reinforcement_learning/skrl/play.py \
  --task=Isaac-USVBench-ROV-Calm-Direct-v1 --num_envs=16 \
  --checkpoint=<repo>/tasks/rov_calm_nav/checkpoints/rov_calm_current_s42.pt
```

`rov_calm_s42.pt` is the legacy June checkpoint. It loads with the current
3D policy architecture, but it predates action clipping and the realistic
hydrodynamics introduced in July. Do not attach its historical 27.1 score to
the current task physics.

---

## Evaluate (standardized protocol)

Score any checkpoint with the benchmark eval (deterministic policy, fixed budget):
```bash
USVBENCH_ASSETS=<repo>/assets OBS_DIM=3 \
python scripts/eval_benchmark.py --task=Isaac-USVBench-ROV-Calm-Direct-v1 \
  --num_envs=64 --eval_steps=6000 --headless \
  --seed=2026 \
  --checkpoint=<repo>/tasks/rov_calm_nav/checkpoints/rov_calm_current_s42.pt
```

| Checkpoint | Physics used for evaluation | Fixed eval result |
|---|---|---:|
| `rov_calm_current_s42.pt` | current clipped-action, realistic-drag task | **6.863 tgt/ep**, 1.495 m/s, 0 OOB |
| `rov_calm_s42.pt` | current task (compatibility check only) | **6.769 tgt/ep**, 1.525 m/s, 0 OOB |
| `rov_calm_s42.pt` | legacy task physics | **27.1 +/- 2.4 tgt/ep** (historical documentation) |

The current-task rows use 64 environments, 6000 evaluation steps, and fixed
evaluation seed 2026. The legacy and current scores are not directly comparable.

---

## What success looks like

Current-physics reference run: seed 42, 3000 iterations, with the best checkpoint
selected by a 1500-step fixed-seed sweep.

| Metric | Reference | Meaning |
|--------|-----------|---------|
| `targets_per_episode` | **6.863** | standardized 6000-step benchmark |
| `mean_speed` | **1.495 m/s** | near the 1.54 m/s thrust/drag terminal speed |
| `oob_per_episode` | **0.000** | no boundary terminations |

Watch a rendered evaluation to confirm the ROV drives smoothly to targets rather
than spinning in place.

**If `targets_per_episode` < 5 or `speed` ≈ 0 after 3000 iter → message Yutong before continuing.**

---

## Env-var toggles

| env var | default | purpose |
|---------|---------|---------|
| `OBS_DIM` | `3` | observation size (keep 3 for this task) |
| `REWARD_VARIANT` | `E7` | leave unset; `E7` = the working recipe |
| `REACH_BONUS` | `10` | reward per target reached |
| `USVBENCH_ASSETS` | `~/usvbench/assets` | where the vessel USD lives |
