# =============================================================
# train_v4_locomotion.ps1 — V4: reward shaped on HOW the vessel moves
#
# V3 reached targets but did so by spinning and drifting:
#   19 revolutions per episode
#   hull 87 deg off its direction of travel
#   (scripted pure pursuit: 8.8 revolutions, 49 deg)
#
# The v3 reward was defined purely on distance to the current target, so
# nothing constrained heading. Two terms added:
#
#   HEADING_COEF  +cos(bearing error) — a reason to hold a heading
#   YAWRATE_COEF  -w^2               — sustained rotation costs
#
# SIZING. Dense per-step terms accumulate over 7200 steps and can swamp a
# sparse terminal bonus. This is exactly how V2 failed: its proximity term
# paid ~34,000 per episode against 1,200 available from actually rescuing,
# so the agent hovered instead. Sized against the same budget here:
#
#   heading  0.05 x 7200          =  360 per episode
#   spin     0.03 x 1.10 x 7200   =  238 per episode
#   combined                       =  598  (30% of the 2,000 rescue bonus)
#
# Shaping influences behaviour without becoming the objective.
#
#   .\rescue_boat\gpu_cleanup.ps1
#   .\rescue_boat\sync_to_isaaclab.ps1
#   .\rescue_boat\train_v4_locomotion.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$TRAIN  = "C:\Users\arifa\Desktop\UCL\Individual Project\train_with_eval.py"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_PROJECT    = "usvbench"

# Task reward — unchanged from v3
$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"

# NEW in v4: locomotion shaping
$env:HEADING_COEF  = "0.05"
$env:YAWRATE_COEF  = "0.03"

# Corrected hull physics
$env:HORIZONTAL_ACTUATION = "0"
$env:WRENCH_WORLD         = "0"
$env:RIGHTING_K           = "1500.0"
$env:RIGHTING_C           = "400.0"
$env:RIGHTING_MAX         = "2500.0"

# Calm water
$env:CURRENT_SPEED = "0.0"
$env:WAVE_AMP      = "0.0"
$env:WAVE_YAW_AMP  = "0.0"
$env:GUST_STD      = "0.0"
$env:VISUALISE     = "0"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

$MAX_ITER = 4688     # 300k steps
$SEEDS    = @(42, 123, 456)

Write-Host "=================================================" -ForegroundColor Green
Write-Host "  V4 - LOCOMOTION SHAPING"                          -ForegroundColor Green
Write-Host "  HEADING_COEF=0.05  YAWRATE_COEF=0.03"            -ForegroundColor Green
Write-Host "  Target: fewer revolutions, hull aligned to track" -ForegroundColor Green
Write-Host "  Watch rescue_rate does not fall below v3 (0.715)" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Green

foreach ($SEED in $SEEDS) {
    $runName = "rescue_boat_v4_s$SEED"
    $env:WANDB_NAME = $runName
    $trainLog = "$LOGDIR\$runName.train.log"

    Write-Host ""
    Write-Host ">>> TRAIN $runName" -ForegroundColor Cyan

    & $PYTHON $TRAIN `
        --task=Isaac-RescueBoat-Direct-v1 `
        --num_envs=64 `
        --headless `
        --max_iterations=$MAX_ITER `
        --eval_mini_steps=7200 `
        --seed=$SEED `
        2>&1 | Tee-Object -FilePath $trainLog

    Write-Host "Seed $SEED done -> $trainLog" -ForegroundColor Green

    if ($SEED -ne $SEEDS[-1]) {
        Write-Host "Cooling down 60s..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  V4 COMPLETE"                                     -ForegroundColor Green
Write-Host "  Next: analyse_behaviour.py on the best checkpoint" -ForegroundColor Green
Write-Host "  v3 reference: 19 rev/ep, 87 deg heading error"    -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
