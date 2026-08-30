# =============================================================
# run_diagnostic.ps1 — Diagnose the P2 eval sweep
#
# Tests whether the 0.000 scores across most checkpoints are a
# checkpoint-loading bug rather than real policy degradation.
# Takes roughly 15-25 minutes (add ~10 min if phase 4 runs).
#
#   conda activate env_isaaclab
#   cd "C:\Users\arifa\Desktop\UCL\Individual Project"
#   .\rescue_boat\run_diagnostic.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$DIAG   = "C:\Users\arifa\Desktop\UCL\Individual Project\diagnose_eval_sweep.py"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"

# Must match the values the checkpoints were trained under
$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"
$env:PROX_RADIUS   = "25.0"

# Do not log the diagnostic to wandb
$env:WANDB_MODE = "disabled"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# Seed 99 run (best result so far: 2.250 tgt/ep at agent_540000)
$RUN = "C:\Users\arifa\Desktop\UCL\Individual Project\logs\skrl\usvbench\2026-08-19_11-28-39_ppo_torch_rescue_boat_v2_s42"

Write-Host "=================================================" -ForegroundColor Yellow
Write-Host "  EVAL SWEEP DIAGNOSTIC"                           -ForegroundColor Yellow
Write-Host "  Testing agent_300000 (scores 0.000) against"     -ForegroundColor Yellow
Write-Host "  agent_540000 (scores 2.250)"                     -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Yellow

& $PYTHON $DIAG `
    --task=Isaac-RescueBoat-Direct-v1 `
    --num_envs=64 `
    --headless `
    --log_dir="$RUN" `
    --ckpt_a=agent_300000.pt `
    --ckpt_b=agent_540000.pt `
    --eval_steps=1500 `
    2>&1 | Tee-Object -FilePath "$LOGDIR\eval_diagnostic.log"

Write-Host ""
Write-Host "Diagnostic complete. Log: $LOGDIR\eval_diagnostic.log" -ForegroundColor Green
Write-Host "Read the SUMMARY block at the end." -ForegroundColor Green
