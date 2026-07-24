# =============================================================
# USVBench — P0 Task B: Boat calm nav baseline (3 seeds)
# Task: Isaac-My-First-Task-Calm-Boat-Direct-v1
# Target: >= 4.0 targets_per_episode per seed
# =============================================================
# Before running:
#   1. conda activate env_isaaclab
#   2. Confirm task is installed (see below if not):
#      xcopy /E /I /Y C:\usvbench\tasks\boat_calm_nav "C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\boat_calm_nav"
#   3. .\train_boat_baseline_multiseed.ps1
# -------------------------------------------------------------
$ErrorActionPreference = "Stop"

$ISAACLAB = "C:\IsaacLab"
$TRAIN    = "C:\Users\arifa\Desktop\UCL\Individual Project\train_with_eval.py"
$PYTHON   = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"

# Boat calm-nav config (V23 — do NOT set FORWARD_TRANSIT=1)
$env:REWARD_VARIANT = "V23"
$env:OBS_DIM        = "9"
$env:OBS_EXTENDED   = "1"
$env:OOB_PENALTY    = "10"
$env:REACH_BONUS    = "50"
$env:SPEED_COUPLE   = "1"

$env:WANDB_PROJECT  = "usvbench"

$TASK = "Isaac-My-First-Task-Calm-Boat-Direct-v1"
$SEEDS = @(42, 456)  # seed 123 done (6.019 tgt/ep): 2026-07-23_11-55-48  [rerun #2 — OOB reverted to +20, eval_mini_steps=1500]

foreach ($SEED in $SEEDS) {
    $env:WANDB_NAME = "boat_calm_nav_V23_s$SEED"
    Write-Host "============================================================"
    Write-Host "  Starting seed $SEED  (target >= 4.0 targets/ep)"
    Write-Host "============================================================"
    & $PYTHON $TRAIN `
        --task=$TASK `
        --num_envs=64 `
        --headless `
        --max_iterations=9375 `
        --eval_mini_steps=1500 `
        --seed=$SEED
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Seed $SEED failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "  Seed $SEED complete."
}

Write-Host ""
Write-Host "All 3 seeds done. Check wandb for targets_per_episode."
