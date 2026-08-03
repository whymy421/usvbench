param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$Repo,

    [Parameter(Mandatory = $true)]
    [string]$RunDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:beaconPath = $null

$trainTask = "Isaac-USV-HazardCrossImb-Direct-v1"
$evalTask = "Isaac-USV-HazardCross-Direct-v1"
$repoParent = Split-Path -Parent $Repo
$trainScript = Join-Path $repoParent "IsaacLab\scripts\reinforcement_learning\skrl\train.py"
$screenScript = Join-Path $Repo "scripts\screen_v6_ladder.py"
$evalScript = Join-Path $Repo "scripts\eval_imbalance.py"

function Stop-Fatal {
    param([Parameter(Mandatory = $true)][string]$Reason)
    if ($script:beaconPath -and (Test-Path -LiteralPath $script:beaconPath -PathType Leaf)) {
        Add-Content -LiteralPath $script:beaconPath -Value "FATAL: $Reason" -Encoding Ascii
    }
    throw "FATAL: $Reason"
}

function Assert-Leaf {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Stop-Fatal "$Description is missing: $Path"
    }
}

function Assert-Directory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        Stop-Fatal "$Description is missing: $Path"
    }
}

Assert-Leaf -Path $Python -Description "Python executable"
Assert-Directory -Path $Repo -Description "repository"
Assert-Leaf -Path $trainScript -Description "Isaac Lab training script"
Assert-Leaf -Path $screenScript -Description "checkpoint screening script"
Assert-Leaf -Path $evalScript -Description "imbalance evaluation script"

if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) {
    New-Item -ItemType Directory -Path $RunDir | Out-Null
}
Assert-Directory -Path $RunDir -Description "Suite D run directory"

$script:beaconPath = Join-Path $RunDir "suite_d_beacon.txt"
Set-Content -LiteralPath $script:beaconPath -Value "SUITE D START $(Get-Date -Format o)" -Encoding Ascii
Assert-Leaf -Path $script:beaconPath -Description "Suite D beacon"

function Write-Beacon {
    param([Parameter(Mandatory = $true)][string]$Message)
    Add-Content -LiteralPath $script:beaconPath `
        -Value "$(Get-Date -Format HH:mm:ss) $Message" -Encoding Ascii
}

foreach ($cmdValue in @($Python, $Repo, $RunDir, $trainScript, $screenScript, $evalScript)) {
    if ($cmdValue.Contains('"') -or $cmdValue.Contains('%')) {
        Stop-Fatal "cmd paths cannot contain a double quote or percent sign: $cmdValue"
    }
}

function Invoke-Logged {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CommandLine,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $cmdPath = Join-Path $RunDir "$Name.cmd"
    $cmdLines = @(
        "@echo off",
        "set `"OMNI_KIT_ACCEPT_EULA=YES`"",
        "cd /d `"$Repo`"",
        "$CommandLine > `"$LogPath`" 2>&1",
        "exit /b %ERRORLEVEL%"
    )
    Set-Content -LiteralPath $cmdPath -Value $cmdLines -Encoding Ascii
    Assert-Leaf -Path $cmdPath -Description "$Name command file"

    $process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList @("/d", "/c", "`"$cmdPath`"") `
        -WorkingDirectory $Repo -WindowStyle Hidden -Wait -PassThru
    if ($null -eq $process) {
        Stop-Fatal "$Name did not return a process"
    }
    Assert-Leaf -Path $LogPath -Description "$Name log"
    if ($process.ExitCode -ne 0) {
        Stop-Fatal "$Name exited with code $($process.ExitCode); log=$LogPath"
    }
}

function Find-RunDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentName,
        [Parameter(Mandatory = $true)][datetime]$NotBefore
    )
    $roots = @(
        (Join-Path $Repo "logs\skrl\usvbench"),
        (Join-Path $repoParent "IsaacLab\logs\skrl\usvbench")
    )
    $candidates = @()
    foreach ($root in $roots) {
        if (Test-Path -LiteralPath $root -PathType Container) {
            $candidates += Get-ChildItem -LiteralPath $root -Directory |
                Where-Object {
                    $_.Name -like "*$ExperimentName*" -and
                    $_.LastWriteTime -ge $NotBefore.AddMinutes(-1)
                }
        }
    }
    return $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Read-LadderRows {
    param([Parameter(Mandatory = $true)][string]$Path)
    Assert-Leaf -Path $Path -Description "ladder log"
    $rows = @()
    $hits = Select-String -LiteralPath $Path `
        -Pattern 'LADDER (agent_(\d+)\.pt): SR=([\d.]+) \(\d+/64\) median_tts=([\d.]+|nan)s' `
        -AllMatches
    foreach ($hit in $hits) {
        foreach ($match in $hit.Matches) {
            $ttsText = $match.Groups[4].Value
            $tts = if ($ttsText -eq "nan") { [double]::PositiveInfinity } else { [double]$ttsText }
            $rows += [pscustomobject]@{
                Checkpoint = $match.Groups[1].Value
                Step = [int]$match.Groups[2].Value
                Sr = [double]$match.Groups[3].Value
                Tts = $tts
            }
        }
    }
    return $rows
}

function Select-LadderChampion {
    param([Parameter(Mandatory = $true)][object[]]$Rows)
    if ($Rows.Count -eq 0) {
        Stop-Fatal "screen produced no LADDER rows"
    }
    return $Rows | Sort-Object `
        @{Expression = { $_.Sr }; Descending = $true}, `
        @{Expression = { $_.Tts }; Ascending = $true}, `
        @{Expression = { $_.Step }; Ascending = $true} |
        Select-Object -First 1
}

$denseChampions = @()
foreach ($seed in @(42, 43)) {
    $name = "suite_d_imb_s$seed"
    $trainLog = Join-Path $RunDir "train_$name.log"
    $started = Get-Date
    Write-Beacon "TRAIN seed=$seed task=$trainTask steps=48000"
    $trainCommand = (
        "`"$Python`" `"$trainScript`" --task $trainTask --num_envs 64 " +
        "--seed $seed --headless " +
        "agent.agent.experiment.experiment_name=$name " +
        "agent.agent.experiment.wandb=False " +
        "agent.agent.experiment.checkpoint_interval=256 " +
        "agent.trainer.timesteps=48000"
    )
    Invoke-Logged -Name "train_$name" -CommandLine $trainCommand -LogPath $trainLog

    $actualRun = Find-RunDirectory -ExperimentName $name -NotBefore $started
    if ($null -eq $actualRun) {
        Stop-Fatal "training run directory was not created for $name"
    }
    $checkpointDir = Join-Path $actualRun.FullName "checkpoints"
    Assert-Directory -Path $checkpointDir -Description "$name checkpoint directory"
    $checkpoints = @(Get-ChildItem -LiteralPath $checkpointDir -File |
        Where-Object { $_.Name -match '^agent_(\d+)\.pt$' })
    if ($checkpoints.Count -lt 12) {
        Stop-Fatal "$name produced only $($checkpoints.Count) checkpoints"
    }
    Write-Beacon "TRAIN seed=$seed run_dir=$($actualRun.FullName) checkpoints=$($checkpoints.Count)"

    $coarseLog = Join-Path $RunDir "coarse_$name.log"
    $coarseCommand = (
        "`"$Python`" `"$screenScript`" --run-dir `"$($actualRun.FullName)`" " +
        "--task $trainTask --every 12 --level 0 --eval-seed 42 --headless"
    )
    Invoke-Logged -Name "coarse_$name" -CommandLine $coarseCommand -LogPath $coarseLog
    $coarseRows = @(Read-LadderRows -Path $coarseLog)
    $coarseChampion = Select-LadderChampion -Rows $coarseRows
    Write-Beacon "COARSE seed=$seed checkpoint=$($coarseChampion.Checkpoint) SR=$($coarseChampion.Sr)"

    $windowLow = [math]::Max(0, $coarseChampion.Step - 2560)
    $windowHigh = [math]::Min(48000, $coarseChampion.Step + 2560)
    $denseSteps = @($checkpoints | ForEach-Object {
        if ($_.Name -match '^agent_(\d+)\.pt$') { [int]$Matches[1] }
    } | Where-Object { $_ -ge $windowLow -and $_ -le $windowHigh } |
        Sort-Object -Unique)
    if ($denseSteps.Count -eq 0) {
        Stop-Fatal "no dense checkpoints found for seed $seed in [$windowLow,$windowHigh]"
    }
    $only = $denseSteps -join ","
    $denseLog = Join-Path $RunDir "dense_$name.log"
    $denseCommand = (
        "`"$Python`" `"$screenScript`" --run-dir `"$($actualRun.FullName)`" " +
        "--task $trainTask --only $only --level 0 --eval-seed 42 --headless"
    )
    Invoke-Logged -Name "dense_$name" -CommandLine $denseCommand -LogPath $denseLog
    $denseRows = @(Read-LadderRows -Path $denseLog)
    if ($denseRows.Count -ne $denseSteps.Count) {
        Stop-Fatal "dense screen seed $seed reported $($denseRows.Count) of $($denseSteps.Count) requested checkpoints"
    }
    $denseChampion = Select-LadderChampion -Rows $denseRows
    $denseChampions += [pscustomobject]@{
        Seed = $seed
        Checkpoint = Join-Path $checkpointDir $denseChampion.Checkpoint
        Step = $denseChampion.Step
        Sr = $denseChampion.Sr
        Tts = $denseChampion.Tts
    }
    Write-Beacon "DENSE seed=$seed checkpoint=$($denseChampion.Checkpoint) SR=$($denseChampion.Sr)"
}

$champion = Select-LadderChampion -Rows $denseChampions
Assert-Leaf -Path $champion.Checkpoint -Description "dense champion checkpoint"
Write-Beacon "CHAMPION seed=$($champion.Seed) step=$($champion.Step) SR=$($champion.Sr) checkpoint=$($champion.Checkpoint)"

$packs = @(
    [pscustomobject]@{ Name = "train"; Values = @("0.0", "0.02", "0.04") },
    [pscustomobject]@{ Name = "interp"; Values = @("0.01", "0.03", "0.05") },
    [pscustomobject]@{ Name = "extrap"; Values = @("0.08", "0.12", "0.16") }
)

foreach ($pack in $packs) {
    foreach ($imbalance in $pack.Values) {
        $imbalanceStem = $imbalance.Replace(".", "p")
        foreach ($evalSeed in @(42, 123)) {
            $stem = "$($pack.Name)_i${imbalanceStem}_e$evalSeed"
            $evalLog = Join-Path $RunDir "eval_$stem.log"
            $evalJson = Join-Path $RunDir "eval_$stem.json"
            Write-Beacon "EVAL START pack=$($pack.Name) imbalance=$imbalance seed=$evalSeed"
            $evalCommand = (
                "`"$Python`" `"$evalScript`" --checkpoint `"$($champion.Checkpoint)`" " +
                "--task $evalTask --episodes 128 --level 0 --eval-seed $evalSeed " +
                "--imbalance $imbalance --out `"$evalJson`" --headless"
            )
            Invoke-Logged -Name "eval_$stem" -CommandLine $evalCommand -LogPath $evalLog
            Assert-Leaf -Path $evalJson -Description "$stem evaluation JSON"
            $product = Get-Content -LiteralPath $evalJson -Raw | ConvertFrom-Json
            if (@($product.records).Count -ne 128) {
                Stop-Fatal "$stem JSON has $(@($product.records).Count) records instead of 128"
            }
            $summaryLines = @(Get-Content -LiteralPath $evalLog | Where-Object {
                $_ -match '^(EVAL task=|imbalance=|  episodes=|  tts|  collision_episodes=|  path_len|  records ->)'
            })
            foreach ($requiredPattern in @(
                '^imbalance=', '  episodes=.*SR=', '  collision_episodes=', '  path_len'
            )) {
                if (-not ($summaryLines | Where-Object { $_ -match $requiredPattern })) {
                    Stop-Fatal "$stem log is missing summary pattern: $requiredPattern"
                }
            }
            foreach ($line in $summaryLines) {
                Write-Beacon "EVAL pack=$($pack.Name) seed=$evalSeed $($line.Trim())"
            }
        }
    }
}

Write-Beacon "SUITE D DONE $(Get-Date -Format o)"
