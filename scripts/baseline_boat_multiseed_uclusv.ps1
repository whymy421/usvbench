# =============================================================
# Boat (USVBench) multi-seed baseline -> uclusv, with REAL eval scoring + mean/std.
#
# Why multi-seed: boat training is RUN-unstable (2026-06-16: same V26 config + seed 42 +
# 3000 iter gave eval 5.119 on 2026-06-03 but 0.656 today — bistable converge~5 / collapse~0.6,
# oob 7.5% vs 452%). So a single run is meaningless. This trains seeds 42/123/456, evals each
# with the standardized protocol (the wandb training metric is NOT reliable — must use eval),
# and reports mean +/- std so Arif sees the real spread.
#
# Run in isaaclab conda env:  conda activate isaaclab ; .\baseline_boat_multiseed_uclusv.ps1
# ~2-3 h (3 trains + 3 evals). Safe to run alongside the E6 Hungarian (NO_VIDEO avoids the
# syntheticdata crash). Each python runs in its OWN console because the boat task's exit
# broadcasts a Ctrl+C that would otherwise kill this loop.
# =============================================================

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"          # boat env prints an emoji on load -> needs utf-8
$env:WANDB_PROJECT = "usvbench"
$env:WANDB_ENTITY  = "uclusv"            # new team shared with Arif
$env:NO_VIDEO = "1"                       # coexist with E6 without the render-pipeline crash

if ($env:CONDA_DEFAULT_ENV -ne "isaaclab") {
    Write-Host "!! Wrong conda env: '$($env:CONDA_DEFAULT_ENV)'. Run 'conda activate isaaclab' first." -ForegroundColor Red
    exit 1
}

# --- HARDENING: the boat task broadcasts a Ctrl+C when Isaac exits, which on 2026-06-16
# killed this loop AFTER seed 42 trained (only s42 train ran; no eval, no s123/s456).
# The per-seed `cmd /c` console (below) was NOT enough on its own. Treat Ctrl+C as input
# so the broadcast can't terminate this orchestrator. (Restored at the end of the script;
# to abort manually, close the window or use Ctrl+Break.)
try { $script:prevTreatCtrlC = [Console]::TreatControlCAsInput; [Console]::TreatControlCAsInput = $true } catch { }

$ISAACLAB = "C:\Users\Yutong\NavRL\IsaacLab"
$TRAIN    = "$ISAACLAB\scripts\reinforcement_learning\skrl\train_with_eval.py"
$EVAL     = "C:\Users\Yutong\NavRL\NavRL2026\eval_benchmark.py"
$LOGROOT  = "C:\Users\Yutong\NavRL\NavRL2026\logs\skrl"
$LOGDIR   = "C:\Users\Yutong\NavRL\NavRL2026\sweep_logs"
$CSV      = "C:\Users\Yutong\NavRL\NavRL2026\baseline_boat_multiseed.csv"
$TASK     = "Isaac-My-First-Task-Calm-Boat-Direct-v1"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
Remove-Item $CSV -ErrorAction SilentlyContinue

# V26 base (the config whose seed-42 run hit eval 5.119)
$env:OBS_DIM = "9"; $env:OBS_EXTENDED = "1"; $env:REWARD_VARIANT = "V23"
$env:SPEED_COUPLE = "1"; $env:REACH_BONUS = "50.0"

$SEEDS = @(42, 123, 456)

function Invoke-Isaac($ArgString, $LogFile) {
    # own console (NO -NoNewWindow) -> boat's Ctrl+C-on-exit can't reach this loop; Minimized
    # just blips in the taskbar; output -> log (tail it for live progress).
    Start-Process -FilePath cmd -ArgumentList "/c python $ArgString > `"$LogFile`" 2>&1" -WindowStyle Minimized -Wait | Out-Null
}
function Find-Ckpt($Name) {
    $d = Get-ChildItem -Path $LOGROOT -Recurse -Directory -Filter "*$Name*" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime | Select-Object -Last 1
    if ($d -and (Test-Path (Join-Path $d.FullName "checkpoints\best_agent.pt"))) { return (Join-Path $d.FullName "checkpoints\best_agent.pt") }
    return $null
}

Write-Host "==================================================" -ForegroundColor Green
Write-Host " Boat multi-seed baseline -> uclusv/usvbench" -ForegroundColor Green
Write-Host " seeds = $($SEEDS -join ', ')  (real eval scoring)" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

foreach ($seed in $SEEDS) {
    $runName = "boat_ms_s$seed"
    $env:WANDB_NAME = $runName
    Write-Host "`n>>> TRAIN $runName  -> log: $LOGDIR\$runName.train.log" -ForegroundColor Cyan
    Invoke-Isaac "`"$TRAIN`" --task=$TASK --num_envs=64 --headless --max_iterations=3000 --seed=$seed" "$LOGDIR\$runName.train.log"
    $ck = Find-Ckpt $runName
    if (-not $ck) { Write-Host "!! no checkpoint for $runName (see log)" -ForegroundColor Red; continue }
    Write-Host ">>> EVAL  $runName  -> log: $LOGDIR\$runName.eval.log" -ForegroundColor Cyan
    Invoke-Isaac "`"$EVAL`" --task=$TASK --num_envs=64 --eval_steps=6000 --headless --seed=$seed --checkpoint=`"$ck`" --csv=`"$CSV`"" "$LOGDIR\$runName.eval.log"
    $row = Import-Csv $CSV -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($row) { Write-Host "   seed $seed -> targets/ep = $($row.targets_per_episode)  (oob $([math]::Round([double]$row.oob_rate*100))%)" -ForegroundColor Magenta }
}

Write-Host "`n========== SUMMARY (real eval, mean +/- std) ==========" -ForegroundColor Green
python -c @"
import csv, statistics as st
vals=[]
try:
    with open(r'$CSV', newline='') as f:
        for r in csv.DictReader(f):
            vals.append(float(r['targets_per_episode']))
except FileNotFoundError:
    pass
if vals:
    m=st.mean(vals); s=st.pstdev(vals) if len(vals)>1 else 0.0
    print(f'targets/ep over {len(vals)} seeds: {m:.2f} +/- {s:.2f}   raw={[round(v,2) for v in vals]}')
    print('NOTE: spread vs prior same-config seed-42 runs (5.12 on 06-03, 0.66 on 06-16) = the run-instability story.')
else:
    print('no eval rows - check sweep_logs')
"@
Write-Host "`nDone. View runs at: https://wandb.ai/uclusv/usvbench  | raw CSV: $CSV" -ForegroundColor Green

# restore prior Ctrl+C handling
try { if ($null -ne $script:prevTreatCtrlC) { [Console]::TreatControlCAsInput = $script:prevTreatCtrlC } } catch { }
