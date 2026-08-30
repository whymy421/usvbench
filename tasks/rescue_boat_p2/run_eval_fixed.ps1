# =============================================================
# run_eval_fixed.ps1 — Re-score existing checkpoints with the corrected metric
#
# No retraining. Re-evaluates checkpoints already on disk using
# whole episodes and the correct denominator.
#
# The old sweep selected best_agent.pt using a broken metric, so the
# "best" checkpoint for every seed was effectively a random pick.
# Better policies may already exist in these folders.
#
#   .\rescue_boat\gpu_cleanup.ps1
#   .\rescue_boat\run_eval_fixed.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$EVAL   = "C:\Users\arifa\Desktop\UCL\Individual Project\eval_fixed.py"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"
$RUNS   = "C:\Users\arifa\Desktop\UCL\Individual Project\logs\skrl\usvbench"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_MODE       = "disabled"

# Must match training values
$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"
$env:PROX_RADIUS   = "25.0"

# Calm water - disturbance off
$env:CURRENT_SPEED = "0.0"
$env:WAVE_AMP      = "0.0"
$env:WAVE_YAW_AMP  = "0.0"
$env:GUST_STD      = "0.0"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# Best-performing seeds first, so partial results are still useful if
# you have to stop early.
$RUNS_TO_SCORE = [ordered]@{
    "s42" = "$RUNS\2026-08-17_20-20-17_ppo_torch_rescue_boat_v2_s42"
    "s99" = "$RUNS\2026-08-19_11-28-39_ppo_torch_rescue_boat_v2_s42"
    "s21" = "$RUNS\2026-08-18_23-34-49_ppo_torch_rescue_boat_v2_s42"
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  CORRECTED CHECKPOINT EVALUATION"                 -ForegroundColor Cyan
Write-Host "  10 checkpoints x $($RUNS_TO_SCORE.Count) runs, no retraining"  -ForegroundColor Cyan
Write-Host "  ~30-40 min per run"                              -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

foreach ($label in $RUNS_TO_SCORE.Keys) {
    $run = $RUNS_TO_SCORE[$label]

    if (-not (Test-Path "$run\checkpoints")) {
        Write-Host "!! No checkpoints for $label at $run" -ForegroundColor Red
        continue
    }

    Write-Host ""
    Write-Host ">>> $label" -ForegroundColor Cyan

    & $PYTHON $EVAL `
        --task=Isaac-RescueBoat-Direct-v1 `
        --num_envs=64 `
        --headless `
        --log_dir="$run" `
        --episodes=1 `
        --update_best `
        2>&1 | Tee-Object -FilePath "$LOGDIR\eval_fixed_$label.log"

    Write-Host "  -> $run\eval_fixed.csv" -ForegroundColor Green
    Start-Sleep -Seconds 20
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  DONE - compare eval_fixed.csv against eval_sweep.csv" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
