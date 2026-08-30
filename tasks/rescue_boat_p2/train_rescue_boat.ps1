# =============================================================
# USVBench — Train: Rescue Boat P2 task (deadline-aware rescue)
# Runs seeds 42, 123, 456 sequentially (each ~5-8h on RTX 4090)
# =============================================================
# Before running:
#   1. Copy rescue_boat/ into Isaac Lab direct tasks dir, alongside catamaran_patrol/
#      (same folder that contains catamaran_patrol/__init__.py)
#   2. conda activate env_isaaclab
#   3. cd "C:\Users\arifa\Desktop\UCL\Individual Project"
#   4. .\rescue_boat\train_rescue_boat.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON  = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$TRAIN   = "C:\Users\arifa\Desktop\UCL\Individual Project\train_with_eval.py"
$LOGDIR  = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"

# Task hyperparameters (v2b: PROX disabled to prevent hover-hacking, bigger bonus)
$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"    # disabled — caused reward hacking in v2 probe
$env:PROX_RADIUS   = "25.0"   # unused

$env:WANDB_PROJECT = "usvbench"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

Write-Host "=================================================" -ForegroundColor Green
Write-Host "  Rescue Boat P2 v2b training  seeds=99,200,777" -ForegroundColor Green
Write-Host "  RESCUE_BONUS=$($env:RESCUE_BONUS)  PROGRESS_COEF=$($env:PROGRESS_COEF)  PROX=off  spawn=10-50m" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

$SEEDS = @(200)   # run one seed at a time to avoid GPU hang between seeds

foreach ($SEED in $SEEDS) {
    $runName = "rescue_boat_v2b_s$SEED"
    $env:WANDB_NAME = $runName
    $trainLog = "$LOGDIR\$runName.train.log"

    Write-Host ""
    Write-Host ">>> TRAIN $runName" -ForegroundColor Cyan

    & $PYTHON $TRAIN `
        --task=Isaac-RescueBoat-Direct-v1 `
        --num_envs=64 `
        --headless `
        --video `
        --video_length=300 `
        --video_interval=100000 `
        --max_iterations=9375 `
        --eval_mini_steps=1500 `
        --seed=$SEED `
        2>&1 | Tee-Object -FilePath $trainLog

    Write-Host "Seed $SEED done. Log: $trainLog" -ForegroundColor Green

    # Give Isaac Sim time to fully release GPU/CUDA resources before next seed
    if ($SEED -ne $SEEDS[-1]) {
        Write-Host "Waiting 60s for GPU cleanup before next seed..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
}

Write-Host ""
Write-Host "All 3 seeds complete. Check wandb 'usvbench' project for rescue_rate." -ForegroundColor Green
Write-Host "Primary metric: Metrics/rescue_rate (P2 bar >= 0.70 averaged across seeds)" -ForegroundColor Green
