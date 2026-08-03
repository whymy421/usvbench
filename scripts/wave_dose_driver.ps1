param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$RepoScripts,

    [Parameter(Mandatory = $true)]
    [string]$AssetsDir,

    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$cells = @(
    [pscustomobject]@{ Height = "0";    Period = "3.0"; Stem = "h0p00_t3p0" },
    [pscustomobject]@{ Height = "0.02"; Period = "3.0"; Stem = "h0p02_t3p0" },
    [pscustomobject]@{ Height = "0.04"; Period = "3.0"; Stem = "h0p04_t3p0" },
    [pscustomobject]@{ Height = "0.08"; Period = "3.0"; Stem = "h0p08_t3p0" },
    [pscustomobject]@{ Height = "0.12"; Period = "3.0"; Stem = "h0p12_t3p0" },
    [pscustomobject]@{ Height = "0.12"; Period = "2.0"; Stem = "h0p12_t2p0" },
    [pscustomobject]@{ Height = "0.12"; Period = "1.5"; Stem = "h0p12_t1p5" },
    [pscustomobject]@{ Height = "0.12"; Period = "1.0"; Stem = "h0p12_t1p0" }
)

if (-not (Test-Path -LiteralPath $OutDir -PathType Container)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
if (-not (Test-Path -LiteralPath $OutDir -PathType Container)) {
    throw "FATAL: failed to create output directory: $OutDir"
}

$beaconPath = Join-Path $OutDir "wave_dose_beacon.txt"
Set-Content -LiteralPath $beaconPath -Value "WAVE DOSE DRIVER START $(Get-Date -Format o)" -Encoding Ascii
if (-not (Test-Path -LiteralPath $beaconPath -PathType Leaf)) {
    throw "FATAL: failed to create beacon file: $beaconPath"
}

function Write-Beacon {
    param([Parameter(Mandatory = $true)][string]$Message)
    Add-Content -LiteralPath $beaconPath -Value $Message -Encoding Ascii
}

function Stop-Fatal {
    param([Parameter(Mandatory = $true)][string]$Reason)
    Write-Beacon "FATAL: $Reason"
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

function Read-Json {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    Assert-Leaf -Path $Path -Description $Description
    try {
        return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
    }
    catch {
        Stop-Fatal "$Description is not valid JSON: $Path; $($_.Exception.Message)"
    }
}

foreach ($namedPath in @(
    [pscustomobject]@{ Path = $Python; Description = "Python executable" },
    [pscustomobject]@{ Path = $Checkpoint; Description = "checkpoint" }
)) {
    Assert-Leaf -Path $namedPath.Path -Description $namedPath.Description
}

foreach ($namedDirectory in @(
    [pscustomobject]@{ Path = $RepoScripts; Description = "scripts directory" },
    [pscustomobject]@{ Path = $AssetsDir; Description = "assets directory" },
    [pscustomobject]@{ Path = $RunDir; Description = "run directory" }
)) {
    if (-not (Test-Path -LiteralPath $namedDirectory.Path -PathType Container)) {
        Stop-Fatal "$($namedDirectory.Description) is missing: $($namedDirectory.Path)"
    }
}

$sweepScript = Join-Path $RepoScripts "wave_dose_sweep.py"
Assert-Leaf -Path $sweepScript -Description "wave dose sweep script"

foreach ($batchValue in @($Python, $AssetsDir, $RunDir, $Checkpoint, $sweepScript, $OutDir)) {
    if ($batchValue.Contains('"') -or $batchValue.Contains('%')) {
        Stop-Fatal "batch paths cannot contain a double quote or percent sign: $batchValue"
    }
}

$mergedRows = @()
$mergedTask = $null

foreach ($cell in $cells) {
    $jsonPath = Join-Path $OutDir ("cell_{0}.json" -f $cell.Stem)
    $logPath = Join-Path $OutDir ("cell_{0}.log" -f $cell.Stem)
    $batPath = Join-Path $OutDir ("cell_{0}.bat" -f $cell.Stem)

    foreach ($stalePath in @($jsonPath, $logPath, $batPath)) {
        if (Test-Path -LiteralPath $stalePath) {
            Remove-Item -LiteralPath $stalePath -Force
        }
        if (Test-Path -LiteralPath $stalePath) {
            Stop-Fatal "failed to remove stale cell product: $stalePath"
        }
    }

    $batchLines = @(
        "@echo off",
        "set `"OMNI_KIT_ACCEPT_EULA=YES`"",
        "set `"USVBENCH_ASSETS=$AssetsDir`"",
        "cd /d `"$RunDir`"",
        "`"$Python`" `"$sweepScript`" --checkpoint `"$Checkpoint`" --wave-height $($cell.Height) --wave-period $($cell.Period) --num-envs 16 --episodes 8 --level 1 --eval-seed 42 --headless --out `"$jsonPath`" > `"$logPath`" 2>&1",
        "exit /b %ERRORLEVEL%"
    )
    Set-Content -LiteralPath $batPath -Value $batchLines -Encoding Ascii
    Assert-Leaf -Path $batPath -Description "cell batch file"

    Write-Beacon "CELL h=$($cell.Height) t=$($cell.Period) START $(Get-Date -Format o)"
    try {
        $process = Start-Process -FilePath $batPath -WorkingDirectory $RunDir -WindowStyle Hidden -PassThru
    }
    catch {
        Stop-Fatal "failed to launch h=$($cell.Height) t=$($cell.Period): $($_.Exception.Message)"
    }
    if ($null -eq $process) {
        Stop-Fatal "Start-Process returned no process for h=$($cell.Height) t=$($cell.Period)"
    }

    $finished = $process.WaitForExit(15 * 60 * 1000)
    if (-not $finished) {
        Write-Beacon "CELL h=$($cell.Height) t=$($cell.Period) TIMEOUT after 900 seconds"
        $killProcess = $null
        $killReason = $null
        try {
            $killProcess = Start-Process -FilePath "taskkill.exe" `
                -ArgumentList @("/PID", [string]$process.Id, "/T", "/F") `
                -WindowStyle Hidden -Wait -PassThru
        }
        catch {
            $killReason = $_.Exception.Message
        }
        if ($null -eq $killProcess -or $killProcess.ExitCode -ne 0) {
            if ($null -ne $killProcess) {
                $killReason = "taskkill exited with code $($killProcess.ExitCode)"
            }
            try {
                $process.Kill()
                $process.WaitForExit()
            }
            catch {
                $killReason = "$killReason; fallback kill failed: $($_.Exception.Message)"
            }
            Stop-Fatal "process-tree kill failed for timed out h=$($cell.Height) t=$($cell.Period): $killReason"
        }
        $process.WaitForExit()
        if (-not $process.HasExited) {
            Stop-Fatal "timed out process is still running for h=$($cell.Height) t=$($cell.Period)"
        }
        Stop-Fatal "15-minute watchdog expired for h=$($cell.Height) t=$($cell.Period)"
    }

    Assert-Leaf -Path $logPath -Description "cell log"
    Write-Beacon "CELL h=$($cell.Height) t=$($cell.Period) EXIT code=$($process.ExitCode) log=$logPath"
    if ($process.ExitCode -ne 0) {
        Stop-Fatal "cell h=$($cell.Height) t=$($cell.Period) exited with code $($process.ExitCode)"
    }

    $cellJson = Read-Json -Path $jsonPath -Description "cell JSON"
    if ($null -eq $cellJson.grid -or @($cellJson.grid).Count -ne 1) {
        Stop-Fatal "cell JSON must contain exactly one grid row: $jsonPath"
    }
    $row = @($cellJson.grid)[0]
    if ([double]$row.height_m -ne [double]$cell.Height -or
        [double]$row.period_s -ne [double]$cell.Period) {
        Stop-Fatal "cell JSON coordinates do not match h=$($cell.Height) t=$($cell.Period): $jsonPath"
    }
    if ($null -eq $mergedTask) {
        $mergedTask = [string]$cellJson.task
    }
    elseif ([string]$cellJson.task -ne $mergedTask) {
        Stop-Fatal "cell JSON task does not match earlier cells: $jsonPath"
    }
    $mergedRows += $row
    Write-Beacon "CELL h=$($cell.Height) t=$($cell.Period) OK json=$jsonPath"
}

$mergedPath = Join-Path $OutDir "dose_merged.json"
$mergedJson = [ordered]@{ task = $mergedTask; grid = $mergedRows } | ConvertTo-Json -Depth 10
Set-Content -LiteralPath $mergedPath -Value $mergedJson -Encoding UTF8
$verifiedMerged = Read-Json -Path $mergedPath -Description "merged JSON"
if ($null -eq $verifiedMerged.grid -or @($verifiedMerged.grid).Count -ne $cells.Count) {
    Stop-Fatal "merged JSON does not contain $($cells.Count) grid rows: $mergedPath"
}

Write-Beacon ""
Write-Beacon "RESULTS"
Write-Beacon ("  {0,7} {1,6} {2,9} {3,8} {4,9}" -f "H (m)", "T (s)", "lambda/L", "SR", "path med")
foreach ($row in @($verifiedMerged.grid)) {
    Write-Beacon ("  {0,7:F2} {1,6:F1} {2,9:F1} {3,8:F3} {4,9:F1}" -f
        [double]$row.height_m,
        [double]$row.period_s,
        [double]$row.lambda_over_L,
        [double]$row.sr,
        [double]$row.path_median_m)
}
Write-Beacon "MERGED OK $mergedPath"
Write-Beacon "WAVE DOSE DRIVER DONE $(Get-Date -Format o)"
