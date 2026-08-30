# =============================================================
# run_probe.ps1 — Physics probe (syncs first, then runs)
#
# Prints the ACTUAL mass and inertia tensor the simulator is using,
# then applies full thrust / full torque / no command open-loop.
#
# This is the run that identifies why the boat will not turn.
#
#   .\rescue_boat\run_probe.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$PROBE  = "C:\Users\arifa\Desktop\UCL\Individual Project\probe_physics.py"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

# Sync first so the probe tests the current code, not a stale copy
Write-Host "Syncing to Isaac Lab..." -ForegroundColor Cyan
& "C:\Users\arifa\Desktop\UCL\Individual Project\rescue_boat\sync_to_isaaclab.ps1"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_MODE       = "disabled"

$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"

# Full 6-DOF actuation. Was "1" (planar) as a workaround for the capsize;
# that masked the frame defects, so it must stay off.
$env:HORIZONTAL_ACTUATION = "0"
$env:WRENCH_WORLD         = "0"   # body-frame wrench (corrected)
$env:RIGHTING_K           = "1500.0"
$env:RIGHTING_C           = "400.0"
$env:RIGHTING_MAX         = "2500.0"
$env:VISUALISE            = "0"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

Write-Host ""
Write-Host "Running probe (4 envs, ~2 min)..." -ForegroundColor Cyan

& $PYTHON $PROBE `
    --task=Isaac-RescueBoat-Direct-v1 `
    --num_envs=4 `
    --headless `
    2>&1 | Tee-Object -FilePath "$LOGDIR\probe.log"

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  Send me the ACTUAL RIGID BODY PROPERTIES block" -ForegroundColor Green
Write-Host "  and the full_torque peak yaw line."             -ForegroundColor Green
Write-Host "  Full log: $LOGDIR\probe.log"                    -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
