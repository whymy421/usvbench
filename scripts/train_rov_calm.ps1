# =============================================================
# USVBench — Train: ROV calm-water navigation (reference task A)
# Current physics: 6.863 targets/episode @ 3000 iter, seed 42 (RTX 5080, ~30 min)
# =============================================================
#
# Before running:
#   1. Copy  tasks/rov_calm_nav/  into your Isaac Lab tasks dir:
#        <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/rov_calm_nav/
#   2. Edit the $ISAACLAB line below to point at YOUR Isaac Lab root.
#   3. conda activate isaaclab  (or your Isaac Lab env)
#   4. .\scripts\train_rov_calm.ps1
# -------------------------------------------------------------

$ErrorActionPreference = "Stop"

# ---- EDIT THIS: path to your Isaac Lab repo root ----
$ISAACLAB = "C:\Users\Yutong\NavRL\IsaacLab"
# -----------------------------------------------------

# Asset dir auto-resolved to this repo's assets/ folder (no manual setup needed)
$env:USVBENCH_ASSETS = (Resolve-Path "$PSScriptRoot\..\assets").Path
$env:PYTHONIOENCODING = "utf-8"

# Task A config: ROV, calm water, E7 reward (forward_speed x exp(alignment) + reach)
$env:OBS_DIM = "3"          # 3D nav obs: (dot, cross, dist_norm)
# REWARD_VARIANT unset  -> defaults to E7 (the working recipe for ROV)
# REACH_BONUS unset     -> defaults to 10

$env:WANDB_PROJECT = "usvbench"
$env:WANDB_NAME = "rov_calm_nav_s42"

$TRAIN = "$ISAACLAB\scripts\reinforcement_learning\skrl\train_with_eval.py"

python $TRAIN `
    --task=Isaac-USVBench-ROV-Calm-Direct-v1 `
    --num_envs=64 --headless --max_iterations=3000 --seed=42 `
    --video --video_interval 50000 --video_length 200
