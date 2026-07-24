[CmdletBinding()]
param(
    [string]$IsaacLabRoot = $env:ISAACLAB_ROOT,
    [int[]]$Seeds = @(42, 123, 456),
    [int]$MaxIterations = 9375,
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
$TaskSource = Join-Path $RepoRoot "tasks\boat_calm_nav"
$TaskDestination = Join-Path $IsaacLabRoot "source\isaaclab_tasks\isaaclab_tasks\direct\boat_calm_nav"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RepoRoot "outputs\boat_baseline_fixed_$RunStamp"
$CsvPath = Join-Path $OutputRoot "benchmark.csv"
$Task = "Isaac-USVBench-Boat-Calm-Direct-v1"

if (-not (Test-Path -LiteralPath $IsaacLabBat)) {
    throw "Isaac Lab launcher not found: $IsaacLabBat. Pass -IsaacLabRoot or set ISAACLAB_ROOT."
}
if (-not (Test-Path -LiteralPath $TrainScript)) { throw "Training script not found: $TrainScript" }
if (-not (Test-Path -LiteralPath $EvalScript)) { throw "Evaluation script not found: $EvalScript" }

& python -c "import isaaclab, skrl, wandb; print('Isaac Lab Python environment OK')"
if ($LASTEXITCODE -ne 0) {
    throw "The active Python environment cannot import isaaclab/skrl/wandb. Activate the Isaac Lab conda environment first."
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $TaskDestination -Force | Out-Null
Get-ChildItem -LiteralPath $TaskSource -Force | Copy-Item -Destination $TaskDestination -Recurse -Force

# Remove experiment knobs that may leak in from a previous shell. The corrected
# reference physics uses cfg constants rather than THRUST_SCALE-style overrides.
$VariablesToClear = @(
    "FORWARD_TRANSIT", "SIDE_APPROACH", "SPAWN_CONE_DEG", "CONE_ANNEAL_STEPS",
    "THRUST_SCALE", "TORQUE_SCALE", "BACKWARD_DRAG_FACTOR", "ANG_DAMP_SCALE",
    "DIST_SPEED_COEF", "DIST_PENALTY_COEF", "PROGRESS_COEF", "ACTION_DELAY",
    "ALIVE_DIRECTIONAL", "BACKWARD_COST", "PHYS_CLEAN"
)
foreach ($Name in $VariablesToClear) {
    Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
}

$env:PYTHONIOENCODING = "utf-8"
$env:NO_VIDEO = "1"
$env:USVBENCH_ASSETS = Join-Path $RepoRoot "assets"
$env:WANDB_PROJECT = "usvbench"
if ($WandbEntity) { $env:WANDB_ENTITY = $WandbEntity }

# Exact boat baseline configuration. Both the OOB terminal condition and this
# penalty now use max_spawn_distance + 20 m in the reference environment.
$env:REWARD_VARIANT = "V23"
$env:OBS_DIM = "9"
$env:OBS_EXTENDED = "1"
$env:OOB_PENALTY = "10.0"
$env:REACH_BONUS = "50.0"
$env:SPEED_COUPLE = "1"

function Invoke-IsaacPython {
    param(
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    # Redirect both native streams in cmd.exe. W&B's helper process can inherit
    # stdout/stderr; reading a PowerShell redirection file before that helper
    # exits would block the launcher from reaching eval.
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
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)][datetime]$StartedAfter
    )

    $LogRoot = Join-Path $RepoRoot "logs\skrl"
    $RunDirectory = Get-ChildItem -LiteralPath $LogRoot -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*$RunName*" -and $_.LastWriteTime -ge $StartedAfter.AddMinutes(-1) } |
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
    "training_seeds=$($Seeds -join ',')"
    "eval_seed=$EvalSeed"
    "max_iterations=$MaxIterations"
    "eval_sweep_steps=$EvalSweepSteps"
    "benchmark_steps=$BenchmarkSteps"
) | Set-Content -LiteralPath (Join-Path $OutputRoot "run_manifest.txt") -Encoding utf8

Write-Host "============================================================" -ForegroundColor Green
Write-Host " Corrected boat baseline: fixed physics + fixed-seed selection" -ForegroundColor Green
Write-Host " Training seeds: $($Seeds -join ', ') | eval seed: $EvalSeed" -ForegroundColor Green
Write-Host " Output: $OutputRoot" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

foreach ($Seed in $Seeds) {
    $RunName = "boat_fixed_s$Seed"
    $env:WANDB_NAME = $RunName
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

    $Checkpoint = Find-LatestCheckpoint -RunName $RunName -StartedAfter $StartedAt
    if (-not $Checkpoint) {
        throw "No selected best_agent.pt found for $RunName. See $TrainLog"
    }

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
}

$Rows = Import-Csv -LiteralPath $CsvPath
$Scores = @($Rows | ForEach-Object { [double]$_.targets_per_episode })
$Mean = ($Scores | Measure-Object -Average).Average
$Variance = 0.0
if ($Scores.Count -gt 1) {
    $Variance = ($Scores | ForEach-Object { [math]::Pow($_ - $Mean, 2) } | Measure-Object -Average).Average
}
$Std = [math]::Sqrt($Variance)

Write-Host "`n==================== SUMMARY ====================" -ForegroundColor Green
foreach ($Row in $Rows) {
    $OobFraction = if ($Row.oob_fraction_completed) {
        "{0:P1}" -f [double]$Row.oob_fraction_completed
    } else {
        "n/a"
    }
    Write-Host (
        "train seed {0}: {1:N3} tgt/ep | oob/events {2:N3} | OOB completed {3}" -f
        $Row.train_seed, [double]$Row.targets_per_episode, [double]$Row.oob_per_episode, $OobFraction
    )
}
Write-Host ("mean +/- population std: {0:N3} +/- {1:N3} tgt/ep" -f $Mean, $Std) -ForegroundColor Magenta
Write-Host "CSV: $CsvPath" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
