# =============================================================
# run_robustness.ps1 — Zero-shot robustness sweep across sea states
#
# Evaluates calm-water-trained policies under increasing disturbance
# WITHOUT retraining. Produces robustness_results.csv.
#
# Each (checkpoint x sea state) is one eval pass in its own process,
# because the disturbance env vars are read at module import.
#
#   conda activate env_isaaclab
#   cd "C:\Users\arifa\Desktop\UCL\Individual Project"
#   .\rescue_boat\run_robustness.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$EVAL   = "C:\Users\arifa\Desktop\UCL\Individual Project\eval_robustness.py"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"
$OUT    = "C:\Users\arifa\Desktop\UCL\Individual Project\robustness_results.csv"
$RUNS   = "C:\Users\arifa\Desktop\UCL\Individual Project\logs\skrl\usvbench"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_MODE       = "disabled"

# Task hyperparameters must match what the checkpoints were trained under
$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"
$env:PROX_RADIUS   = "25.0"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# Best checkpoint per seed. Edit these as stronger runs land.
# Format: label = path to checkpoint
$CHECKPOINTS = [ordered]@{
    "s42"  = "$RUNS\2026-08-17_20-20-17_ppo_torch_rescue_boat_v2_s42\checkpoints\best_agent.pt"
    "s99"  = "$RUNS\2026-08-19_11-28-39_ppo_torch_rescue_boat_v2_s42\checkpoints\best_agent.pt"
    "s21"  = "$RUNS\2026-08-18_23-34-49_ppo_torch_rescue_boat_v2_s42\checkpoints\best_agent.pt"
}

$SEA_STATES = @("calm", "slight", "moderate", "rough")

# Full episode. The 1500-step window used by the training sweep only samples the
# opening of an episode and biases rescue rate downward.
$EVAL_STEPS = 7200

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  ZERO-SHOT ROBUSTNESS SWEEP"                      -ForegroundColor Cyan
Write-Host "  $($CHECKPOINTS.Count) checkpoints x $($SEA_STATES.Count) sea states" -ForegroundColor Cyan
Write-Host "  No retraining - evaluation only"                 -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

if (Test-Path $OUT) {
    $bak = "$OUT.bak"
    Copy-Item $OUT $bak -Force
    Remove-Item $OUT -Force
    Write-Host "Previous results backed up to $bak" -ForegroundColor DarkGray
}

$total = $CHECKPOINTS.Count * $SEA_STATES.Count
$i = 0

foreach ($label in $CHECKPOINTS.Keys) {
    $ckpt = $CHECKPOINTS[$label]

    if (-not (Test-Path $ckpt)) {
        Write-Host "!! Missing checkpoint for $label : $ckpt" -ForegroundColor Red
        continue
    }

    foreach ($sea in $SEA_STATES) {
        $i++
        Write-Host ""
        Write-Host ">>> [$i/$total] $label @ $sea" -ForegroundColor Cyan

        & $PYTHON $EVAL `
            --task=Isaac-RescueBoat-Direct-v1 `
            --num_envs=64 `
            --headless `
            --checkpoint="$ckpt" `
            --sea_state=$sea `
            --eval_steps=$EVAL_STEPS `
            --label=$label `
            --out="$OUT" `
            2>&1 | Tee-Object -FilePath "$LOGDIR\robust_${label}_${sea}.log"

        # Let Isaac Sim fully release GPU resources before the next process
        Start-Sleep -Seconds 20
    }
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  SWEEP COMPLETE"                                  -ForegroundColor Green
Write-Host "  Results: $OUT"                                   -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

if (Test-Path $OUT) {
    Write-Host ""
    Import-Csv $OUT | Format-Table label, sea_state, targets_per_episode, rescue_rate -AutoSize
}
