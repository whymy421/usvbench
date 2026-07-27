[CmdletBinding()]
param(
    [string]$IsaacLabRoot = $env:ISAACLAB_ROOT,
    [int]$Seed = 42,
    [int]$MaxIterations = 3000,
    [int]$EvalSweepSteps = 1500,
    [int]$BenchmarkSteps = 6000,
    [int]$EvalSeed = 2026,
    [string]$WandbEntity = "uclusv"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $IsaacLabRoot) {
    throw "Pass -IsaacLabRoot or set the ISAACLAB_ROOT environment variable."
}

$IsaacLabBat = Join-Path $IsaacLabRoot "isaaclab.bat"
$TrainScript = Join-Path $RepoRoot "scripts\train_with_eval.py"
$EvalScript = Join-Path $RepoRoot "scripts\eval_benchmark.py"
$TaskSource = Join-Path $RepoRoot "tasks\rov_calm_nav"
$TaskDestination = Join-Path $IsaacLabRoot "source\isaaclab_tasks\isaaclab_tasks\direct\rov_calm_nav"
$Task = "Isaac-USVBench-ROV-Calm-Direct-v1"
$RunName = "rov_current_s$Seed"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RepoRoot "outputs\rov_baseline_current_$RunStamp"
$CsvPath = Join-Path $OutputRoot "benchmark.csv"

if (-not (Test-Path -LiteralPath $IsaacLabBat)) {
    throw "Isaac Lab launcher not found: $IsaacLabBat"
}
if (-not (Test-Path -LiteralPath $TrainScript)) { throw "Training script not found: $TrainScript" }
if (-not (Test-Path -LiteralPath $EvalScript)) { throw "Evaluation script not found: $EvalScript" }

& python -c "import isaaclab, skrl, wandb; print('Isaac Lab Python environment OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Activate the Isaac Lab conda environment before running this launcher."
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $TaskDestination -Force | Out-Null
Get-ChildItem -LiteralPath $TaskSource -Force | Copy-Item -Destination $TaskDestination -Recurse -Force

# Remove boat and experimental knobs that may leak in from another shell.
$VariablesToClear = @(
    "OBS_EXTENDED", "OBS_MODE", "OOB_PENALTY", "FORWARD_TRANSIT", "SIDE_APPROACH",
    "SPAWN_CONE_DEG", "CONE_ANNEAL_STEPS", "THRUST_SCALE", "TORQUE_SCALE",
    "BACKWARD_DRAG_FACTOR", "ANG_DAMP_SCALE", "DIST_SPEED_COEF", "DIST_PENALTY_COEF",
    "PROGRESS_COEF", "ACTION_DELAY", "ALIVE_DIRECTIONAL", "BACKWARD_COST", "PHYS_CLEAN"
)
foreach ($Name in $VariablesToClear) {
    Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
}

$env:PYTHONIOENCODING = "utf-8"
$env:NO_VIDEO = "1"
$env:USVBENCH_ASSETS = Join-Path $RepoRoot "assets"
$env:WANDB_PROJECT = "usvbench"
$env:WANDB_NAME = $RunName
if ($WandbEntity) { $env:WANDB_ENTITY = $WandbEntity }

$env:OBS_DIM = "3"
$env:REWARD_VARIANT = "E7"
$env:REACH_BONUS = "10.0"

function Invoke-IsaacPython {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    $QuotedArgs = @($Arguments | ForEach-Object {
        $text = [string]$_
        if ($text -match '[\s"]') { '"' + $text.Replace('"', '\"') + '"' } else { $text }
    }) -join ' '
    $CommandLine = '"{0}" -p "{1}" {2} >> "{3}" 2>&1' -f $IsaacLabBat, $Script, $QuotedArgs, $LogPath
    & cmd.exe /d /c $CommandLine
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "Isaac Lab command failed with exit code $ExitCode. See $LogPath"
    }
}

function Find-LatestCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][datetime]$StartedAfter
    )

    $LogRoot = Join-Path $RepoRoot "logs\skrl"
    $RunDirectory = Get-ChildItem -LiteralPath $LogRoot -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*$Name*" -and $_.LastWriteTime -ge $StartedAfter.AddMinutes(-1) } |
        Sort-Object LastWriteTime |
        Select-Object -Last 1
    if (-not $RunDirectory) { return $null }

    $Checkpoint = Join-Path $RunDirectory.FullName "checkpoints\best_agent.pt"
    if (Test-Path -LiteralPath $Checkpoint) { return $Checkpoint }
    return $null
}

$Commit = git -C $RepoRoot rev-parse HEAD
@(
    "commit=$Commit"
    "task=$Task"
    "training_seed=$Seed"
    "eval_seed=$EvalSeed"
    "max_iterations=$MaxIterations"
    "eval_sweep_steps=$EvalSweepSteps"
    "benchmark_steps=$BenchmarkSteps"
) | Set-Content -LiteralPath (Join-Path $OutputRoot "run_manifest.txt") -Encoding utf8

Write-Host "============================================================" -ForegroundColor Green
Write-Host " Current-physics ROV baseline: seed $Seed" -ForegroundColor Green
Write-Host " Output: $OutputRoot" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$TrainLog = Join-Path $OutputRoot "$RunName.train.log"
$EvalLog = Join-Path $OutputRoot "$RunName.eval.log"
$StartedAt = Get-Date

Write-Host "`n>>> TRAIN seed $Seed" -ForegroundColor Cyan
$TrainArguments = @(
    "--task=$Task",
    "--num_envs=64",
    "--headless",
    "--max_iterations=$MaxIterations",
    "--eval_mini_steps=$EvalSweepSteps",
    "--eval_sweep_seed=$EvalSeed",
    "--seed=$Seed"
)
Invoke-IsaacPython -Script $TrainScript -Arguments $TrainArguments -LogPath $TrainLog

$Checkpoint = Find-LatestCheckpoint -Name $RunName -StartedAfter $StartedAt
if (-not $Checkpoint) {
    throw "No selected best_agent.pt found for $RunName. See $TrainLog"
}

$ExportedCheckpoint = Join-Path $OutputRoot "rov_calm_current_s$Seed.pt"
Copy-Item -LiteralPath $Checkpoint -Destination $ExportedCheckpoint -Force

Write-Host ">>> BENCHMARK seed $Seed using fixed eval seed $EvalSeed" -ForegroundColor Cyan
$EvalArguments = @(
    "--task=$Task",
    "--num_envs=64",
    "--eval_steps=$BenchmarkSteps",
    "--headless",
    "--train_seed=$Seed",
    "--seed=$EvalSeed",
    "--checkpoint=$Checkpoint",
    "--csv=$CsvPath"
)
Invoke-IsaacPython -Script $EvalScript -Arguments $EvalArguments -LogPath $EvalLog

$Row = Import-Csv -LiteralPath $CsvPath | Select-Object -Last 1
Write-Host "`n==================== SUMMARY ====================" -ForegroundColor Green
Write-Host (
    "train seed {0}: {1:N3} tgt/ep | mean speed {2:N3} m/s | oob/events {3:N3}" -f
    $Row.train_seed, [double]$Row.targets_per_episode, [double]$Row.mean_speed, [double]$Row.oob_per_episode
)
Write-Host "Checkpoint: $ExportedCheckpoint" -ForegroundColor Green
Write-Host "CSV: $CsvPath" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
