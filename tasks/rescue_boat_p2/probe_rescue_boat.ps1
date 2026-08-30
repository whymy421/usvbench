# probe_rescue_boat.ps1 — Quick 300K-step probe to validate v2 reward shaping
# Run this BEFORE the full 1.5M training to confirm the policy is learning.
# Takes ~1 hour. Check wandb: rescue_rate should rise above 0.05 by step 200K.
# If still flat at 0.01-0.02 after 300K → stop and debug before wasting 15+ hours.

$ErrorActionPreference = "Continue"

$PYTHON  = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$TRAIN   = "C:\Users\arifa\Desktop\UCL\Individual Project\train_with_eval.py"
$LOGDIR  = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:URGENCY_POW      = "2.0"
$env:PROGRESS_COEF    = "3.0"
$env:RESCUE_BONUS     = "500.0"
$env:LOSE_PENALTY     = "50.0"
$env:PROX_COEF        = "0.0"   # disabled — was causing hover-hacking
$env:PROX_RADIUS      = "25.0"
$env:WANDB_PROJECT    = "usvbench"
$env:WANDB_NAME       = "rescue_boat_v2b_probe_s42"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

Write-Host "=================================================" -ForegroundColor Yellow
Write-Host "  PROBE RUN: 300K steps, seed=42 (~1 hour)"      -ForegroundColor Yellow
Write-Host "  Watch wandb: rescue_rate should climb > 0.05"   -ForegroundColor Yellow
Write-Host "  If flat at 0.02 after 300K -> stop and debug"   -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Yellow

& $PYTHON $TRAIN `
    --task=Isaac-RescueBoat-Direct-v1 `
    --num_envs=64 `
    --headless `
    --max_iterations=4688 `
    --eval_mini_steps=1500 `
    --seed=42 `
    2>&1 | Tee-Object -FilePath "$LOGDIR\probe_v2_s42.log"

Write-Host ""
Write-Host "Probe done. Check wandb run: rescue_boat_v2_probe_s42" -ForegroundColor Green
Write-Host "If rescue_rate > 0.05 at end -> run full train_rescue_boat.ps1" -ForegroundColor Green
