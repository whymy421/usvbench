# R3 pull (controller-run; adapted from G12 launcher\pull_dose.ps1, SHA-256 6B59B448...2449): copy one executor queue
# directory <data root>/usvb/dose_<batch_id>_a<attempt> and <data root>/usvb/queues_v2 from one rental into a NEW local
# folder and verify every file against a remote listing (relative path, size, SHA-256), laid out as pull_dose.ps1 does:
# launcher\pulls\<QueueDir>_<HostKey>_<stamp>_try<k>\<QueueDir> and ...\queues_v2.
# usage: pull_rental.ps1 -SshHost <host> [-SshPort <port>] [-SshUser root] -SshKey <key file> -DataRoot <dir>
#            -HostKey <dispatch host name> -QueueDir dose_<batch_id>_a<attempt> [-DescribeOnly]
# Copy method: scp -r (no rsync assumption on Windows; Windows PowerShell 5.1 cannot pipe binary tar output). A failed or
# incomplete copy is retried into a fresh local folder (PULL_TRIES); nothing local or remote is deleted or overwritten.
# Run it after the batch is terminal (dose_FINISHED.txt or dose_ABORTED.txt); an interim pull of a running batch is a
# snapshot and may differ from the listing.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [ValidateRange(1, 65535)][int]$SshPort = 22,
    [string]$SshUser = 'root',
    [Parameter(Mandatory=$true)][string]$SshKey,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]{1,32}$')][string]$HostKey,
    [Parameter(Mandatory=$true)][ValidatePattern('^dose_[A-Za-z0-9_-]{1,60}_a[1-9]$')][string]$QueueDir,
    [switch]$DescribeOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\r3_local_common.ps1')
Set-Variable -Name PULL_TRIES -Value 4 -Option Constant
$endpoint = New-R3Endpoint $SshHost $SshPort $SshUser $SshKey $DataRoot -Describe:$DescribeOnly
$executorDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'executor'
$toolFiles = @((Join-Path $executorDir 'r3_remote.py'), (Join-Path $executorDir 'usvb_dose_executor.py'))
$runStamp = if ($DescribeOnly) { 'STAMP' } else { New-R3Stamp }
$pullsRoot = Join-Path $PSScriptRoot 'pulls'
$runsRoot = Join-Path $PSScriptRoot 'runs'
if (-not $DescribeOnly) {
    foreach ($folder in @($pullsRoot, $runsRoot)) { if (-not (Test-Path -LiteralPath $folder)) { New-Item -Path $folder -ItemType Directory | Out-Null } }
    $script:r3LocalLog = Join-Path $runsRoot "pull_$($QueueDir)_$($HostKey)_$runStamp.log"
    [IO.File]::WriteAllText($script:r3LocalLog, "R3 pull $QueueDir host=$HostKey endpoint=$($endpoint.Target):$($endpoint.Port)`r`n", [Text.Encoding]::ASCII)
}
$stagingRoot = Join-Path $runsRoot "pull_toolkit_$($QueueDir)_$runStamp"
if (-not $DescribeOnly) { New-Item -Path $stagingRoot -ItemType Directory -ErrorAction Stop | Out-Null }
$toolkit = Send-R3Toolkit $endpoint "pull_$HostKey" $toolFiles @{} $stagingRoot -Describe:$DescribeOnly
foreach ($remoteName in @($QueueDir, 'queues_v2')) {
    Invoke-R3Step $endpoint "$toolkit/r3_remote.py" @('listing', '--data-root', $DataRoot, '--name', $remoteName) @(0) $STEP_TRIES -Describe:$DescribeOnly
    if ($DescribeOnly) {
        "----- PULL scp -r $(@(Get-R3ScpOptions $endpoint) -join ' ') $($endpoint.Target):$($endpoint.Usvb)/$remoteName <work>\launcher\pulls\$($QueueDir)_$($HostKey)_<stamp>_try<k> ; each file compared with the listing (size, SHA-256); up to $PULL_TRIES tries -----"
        continue
    }
    $listing = @{}
    foreach ($line in @($script:r3LastLines)) {
        $match = [regex]::Match($line, '^FILE ([0-9A-F]{64}) ([0-9]+) (\S+)$')
        if ($match.Success) { $listing[$match.Groups[3].Value] = @($match.Groups[1].Value, [long]$match.Groups[2].Value) }
    }
    if (@($script:r3LastLines | Where-Object { $_ -cmatch '^LISTING-OK ' }).Count -ne 1) { throw "Listing of $remoteName did not end with LISTING-OK" }
    Write-R3Log "LISTING $remoteName files=$($listing.Count)"
    $pulled = $false
    for ($pullTry = 1; $pullTry -le $PULL_TRIES -and -not $pulled; $pullTry++) {
        $dest = Join-Path $pullsRoot "$($QueueDir)_$($HostKey)_$($runStamp)_try$pullTry"
        if (-not (Test-Path -LiteralPath $dest)) { New-Item -Path $dest -ItemType Directory | Out-Null }
        $copy = Invoke-R3LocalNative 'scp' (@('-r') + @(Get-R3ScpOptions $endpoint) + @("$($endpoint.Target):$($endpoint.Usvb)/$remoteName", $dest))
        foreach ($line in @($copy.Lines)) { Write-R3Log "[scp $remoteName] $line" }
        $localRoot = Join-Path $dest $remoteName
        $bad = [Collections.Generic.List[string]]::new()
        foreach ($relative in @($listing.Keys)) {
            $path = Join-Path $localRoot ($relative.Replace('/', '\'))
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $bad.Add("missing $relative"); continue }
            if ((Get-Item -LiteralPath $path).Length -ne $listing[$relative][1] -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -cne $listing[$relative][0]) { $bad.Add("differs $relative") }
        }
        if ($copy.Exit -eq 0 -and $bad.Count -eq 0) {
            $pulled = $true
            Write-R3Log "PULL-OK $remoteName files=$($listing.Count) local=$localRoot"
        } else {
            Write-R3Log "PULL-INCOMPLETE $remoteName try=$pullTry scp_exit=$($copy.Exit) problems=$($bad.Count) first=$(@($bad.ToArray() | Select-Object -First 3) -join '; ')"
            if ($pullTry -lt $PULL_TRIES) { Start-Sleep -Milliseconds $STEP_WAIT_MS }
        }
    }
    if (-not $pulled) { throw "Pull of $remoteName failed $PULL_TRIES times" }
}
if ($DescribeOnly) { 'DESCRIBE-ONLY: no ssh, no scp, nothing written' } else { Write-R3Log "R3 PULL COMPLETE $QueueDir host=$HostKey (judge it with the G12 judge, update 4 or later: judge\dose_judge.py --pull-dir <the PULL-OK local folder of $QueueDir>)" }
