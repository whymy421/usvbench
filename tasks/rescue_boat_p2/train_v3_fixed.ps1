# =============================================================
# train_v3_fixed.ps1 — V3: training on the CORRECTED environment
#
# Everything before this ran on a hull that capsized and sank the
# moment it turned, so no V1/V2/V2b result is comparable to these.
#
# Environment fixes now in place:
#   - yaw torque about world +Z (was body +Z -> roll feedback -> capsize)
#   - righting moment on roll/pitch (buoyancy at COM gave no restoring torque)
#   - rov_volume 0.35 -> 0.75  (reserve buoyancy 17% -> 150%)
#   - max_angular_velocity 120 -> 3.0
#
# Evaluation fixes now in place:
#   - observation normaliser restored explicitly on every checkpoint load
#   - whole-episode scoring with the correct denominator
#     (the old metric overstated rescue_rate by ~4.6x)
#
# Shorter runs than before: 300k steps rather than 600k. The old
# justification for 600k was a "late-training peak" that turned out to be
# an artifact of the broken metric, and with limited time remaining more
# seeds at 300k is worth more than fewer seeds at 600k.
#
#   .\rescue_boat\gpu_cleanup.ps1
#   .\rescue_boat\sync_to_isaaclab.ps1
#   .\rescue_boat\train_v3_fixed.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$TRAIN  = "C:\Users\arifa\Desktop\UCL\Individual Project\train_with_eval.py"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_PROJECT    = "usvbench"

# Task reward (unchanged from v2b - the reward was never the problem)
$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"
$env:PROX_RADIUS   = "25.0"

# Corrected hull physics
# Full 6-DOF actuation. Was "1" (planar) as a workaround for the capsize;
# that masked the frame defects, so it must stay off.
$env:HORIZONTAL_ACTUATION = "0"
$env:WRENCH_WORLD         = "0"   # body-frame wrench (corrected)
# 8000/2000 diverged numerically within a single 1/120 s step and crashed
# the machine. These are the clamped, stable gains.
$env:RIGHTING_K           = "1500.0"
$env:RIGHTING_C           = "400.0"
$env:RIGHTING_MAX         = "2500.0"

# Calm water for the baseline
$env:CURRENT_SPEED = "0.0"
$env:WAVE_AMP      = "0.0"
$env:WAVE_YAW_AMP  = "0.0"
$env:GUST_STD      = "0.0"

# Markers off during training; enable only when recording video
$env:VISUALISE = "0"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# 300k steps = 4688 iterations x 64 rollouts, roughly 20 min per seed
$MAX_ITER = 4688
$SEEDS    = @(42, 123, 456)

Write-Host "=================================================" -ForegroundColor Green
Write-Host "  V3 TRAINING - CORRECTED ENVIRONMENT"             -ForegroundColor Green
Write-Host "  seeds: $($SEEDS -join ', ')   300k steps each"   -ForegroundColor Green
Write-Host "  NOT comparable to any V1/V2/V2b result"          -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Green

foreach ($SEED in $SEEDS) {
    $runName = "rescue_boat_v3_s$SEED"
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

    # Isaac Sim does not always release the GPU promptly; without this the
    # next seed dies during startup before it can open a log.
    if ($SEED -ne $SEEDS[-1]) {
        Write-Host "Cooling down 60s..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  V3 COMPLETE"                                     -ForegroundColor Green
Write-Host "  Check the WARNING line in each sweep: a flat"    -ForegroundColor Green
Write-Host "  rescue_rate means the env is still wrong."       -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
