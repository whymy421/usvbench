# =============================================================
# record_video.ps1 — Record presentation video of the V3 policy
#
# Uses the best seed (42, rescue_rate 0.934). Enables the water
# surface and urgency-coloured casualty markers, which are off
# during training.
#
# Casualty colours: green -> amber (>50% of timer) -> red (>80%) -> grey (lost)
# That colouring is what makes the prioritisation visible on screen.
#
#   .\rescue_boat\gpu_cleanup.ps1
#   .\rescue_boat\record_video.ps1
# =============================================================
$ErrorActionPreference = "Continue"

$PYTHON = "C:\Users\arifa\anaconda3\envs\env_isaaclab\python.exe"
$REC    = "C:\Users\arifa\Desktop\UCL\Individual Project\record_policy_video.py"
$RUNS   = "C:\Users\arifa\Desktop\UCL\Individual Project\logs\skrl\usvbench"
$LOGDIR = "C:\Users\arifa\Desktop\UCL\Individual Project\sweep_logs"

$env:USVBENCH_ASSETS  = "C:\usvbench\assets"
$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_MODE       = "disabled"

# Must match training
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

# Calm water
$env:CURRENT_SPEED = "0.0"
$env:WAVE_AMP      = "0.0"
$env:WAVE_YAW_AMP  = "0.0"
$env:GUST_STD      = "0.0"

# The whole point of this script: water surface, casualty markers and lighting.
# record_policy_video.py also sets this, but set it here so the intent is explicit
# and so the startup banner records it.
$env:VISUALISE = "1"

New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

# Best V3 result on the rescaled task: 0.7148 (549/768), agent_270027
# V4 best: rescue_rate 0.8086, with locomotion shaping (HEADING_COEF=0.05,
# YAWRATE_COEF=0.03). Revolutions 19 -> 12, hull alignment 87 -> 59 deg.
$CKPT = "$RUNS\2026-08-22_19-28-22_ppo_torch_rescue_boat_v2_s42\checkpoints\agent_300030.pt"

if (-not (Test-Path $CKPT)) {
    Write-Host "Checkpoint not found: $CKPT" -ForegroundColor Red
    Write-Host "Available runs:" -ForegroundColor Yellow
    Get-ChildItem $RUNS -Directory | Sort-Object LastWriteTime -Descending |
        Select-Object -First 6 Name | Format-Table -AutoSize
    exit 1
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  RECORDING V3 POLICY (best seed, 0.715)"           -ForegroundColor Cyan
Write-Host "  Water surface + urgency-coloured casualties ON" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

& $PYTHON $REC `
    --task=Isaac-RescueBoat-Direct-v1 `
    --num_envs=4 `
    --checkpoint="$CKPT" `
    --video_length=7200 `
    --cam_dist=170 `
    --cam_height=120 `
    2>&1 | Tee-Object -FilePath "$LOGDIR\record_video.log"

Write-Host ""
Write-Host "Video folder:" -ForegroundColor Green
Write-Host "  $(Split-Path (Split-Path $CKPT))\videos\presentation" -ForegroundColor Green
