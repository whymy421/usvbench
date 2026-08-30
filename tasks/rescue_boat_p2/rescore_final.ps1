# =============================================================
# rescore_final.ps1 — Final protocol: all three seeds, same method
#
# Re-scores each seed's best checkpoint with eval_fixed.py over 3
# episodes (768 casualties). This is the number that goes in the
# thesis: one protocol, no mixing of 1-episode and 3-episode figures.
#
#   .\rescue_boat\gpu_cleanup.ps1
#   .\rescue_boat\rescore_final.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$EVAL   = "C:\Users\arifa\Desktop\UCL\Individual Project\eval_fixed.py"
$RUNS   = "C:\Users\arifa\Desktop\UCL\Individual Project\logs\skrl\usvbench"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_MODE       = "disabled"

# Must match the training configuration exactly
$env:URGENCY_POW   = "2.0"
$env:PROGRESS_COEF = "3.0"
$env:RESCUE_BONUS  = "500.0"
$env:LOSE_PENALTY  = "50.0"
$env:PROX_COEF     = "0.0"

$env:HORIZONTAL_ACTUATION = "0"
$env:WRENCH_WORLD         = "0"
$env:RIGHTING_K           = "1500.0"
$env:RIGHTING_C           = "400.0"
$env:RIGHTING_MAX         = "2500.0"
$env:VISUALISE            = "0"

$env:CURRENT_SPEED = "0.0"
$env:WAVE_AMP      = "0.0"
$env:WAVE_YAW_AMP  = "0.0"
$env:GUST_STD      = "0.0"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# seed label -> run folder, best checkpoint from the 1-episode sweep
$SEEDS = [ordered]@{
    "s42"  = @("2026-08-22_01-23-51_ppo_torch_rescue_boat_v2_s42", "agent_300030.pt")
    "s123" = @("2026-08-22_03-04-09_ppo_torch_rescue_boat_v2_s42", "agent_180018.pt")
    "s456" = @("2026-08-22_04-44-47_ppo_torch_rescue_boat_v2_s42", "agent_270027.pt")
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  FINAL RE-SCORE - 3 episodes, 768 casualties"     -ForegroundColor Cyan
Write-Host "  Same protocol for every seed"                    -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

foreach ($k in $SEEDS.Keys) {
    $run  = Join-Path $RUNS $SEEDS[$k][0]
    $ckpt = Join-Path $run "checkpoints\$($SEEDS[$k][1])"

    if (-not (Test-Path $ckpt)) {
        Write-Host "!! missing $k : $ckpt" -ForegroundColor Red
        continue
    }

    Write-Host ""
    Write-Host ">>> $k  $($SEEDS[$k][1])" -ForegroundColor Cyan

    & $PYTHON $EVAL `
        --task=Isaac-RescueBoat-Direct-v1 `
        --num_envs=64 `
        --headless `
        --log_dir="$run" `
        --checkpoint="$ckpt" `
        --episodes=3 `
        --out="$run\eval_final_$k.csv" `
        2>&1 | Tee-Object -FilePath "$LOGDIR\rescore_$k.log"

    Start-Sleep -Seconds 20
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  DONE - collecting results"                       -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

$rows = @()
foreach ($k in $SEEDS.Keys) {
    $run = Join-Path $RUNS $SEEDS[$k][0]
    $csv = "$run\eval_final_$k.csv"
    if (Test-Path $csv) {
        $r = Import-Csv $csv | Select-Object -First 1
        $rows += [pscustomobject]@{
            Seed        = $k
            Checkpoint  = $r.checkpoint
            RescueRate  = [double]$r.rescue_rate
            Rescues     = "$($r.rescues)/$($r.denominator)"
        }
    }
}
if ($rows.Count -gt 0) {
    $rows | Format-Table -AutoSize
    $m = ($rows | Measure-Object RescueRate -Average).Average
    Write-Host ("  mean over {0} seeds : {1:N4}" -f $rows.Count, $m) -ForegroundColor Green
    Write-Host ("  P2 bar             : 0.70") -ForegroundColor Green
    Write-Host ("  greedy oracle      : 0.582") -ForegroundColor Green
    Write-Host ("  margin over oracle : +{0:N3}" -f ($m - 0.582)) -ForegroundColor Green
}
