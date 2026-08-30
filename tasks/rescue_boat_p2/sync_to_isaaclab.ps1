# sync_to_isaaclab.ps1 — Copy rescue_boat task files to Isaac Lab
# Run from: Individual Project folder or anywhere
# Usage: .\rescue_boat\sync_to_isaaclab.ps1

$SRC = "C:\Users\arifa\Desktop\UCL\Individual Project\rescue_boat"
$DST = "C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\rescue_boat"

if (-not (Test-Path $DST)) {
    Write-Host "ERROR: Isaac Lab rescue_boat dir not found at $DST" -ForegroundColor Red
    exit 1
}

Copy-Item "$SRC\rescue_boat_env.py"      "$DST\" -Force
Copy-Item "$SRC\rescue_boat_env_cfg.py"  "$DST\" -Force
Copy-Item "$SRC\agents\skrl_ppo_cfg.yaml" "$DST\agents\" -Force

Write-Host "Synced rescue_boat v2 files to Isaac Lab:" -ForegroundColor Green
Write-Host "  rescue_boat_env.py       (proximity shaping, spawn 10-50m)" -ForegroundColor Cyan
Write-Host "  rescue_boat_env_cfg.py   (spawn 10-50m, timers 40-90s, spacing 200m)" -ForegroundColor Cyan
Write-Host "  skrl_ppo_cfg.yaml        (experiment_name -> rescue_boat_v2_s42)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Now run: .\rescue_boat\train_rescue_boat.ps1" -ForegroundColor Yellow
