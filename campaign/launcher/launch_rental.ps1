# R3 launcher (controller-run; adapted from G12 launcher\launch_dose.ps1, SHA-256 BF7B5F17...DB94): install the Linux
# executor executor\usvb_dose_executor.py with ONE slot's plan list (and the batch files it names) on one rental and start
# that slot detached (setsid + nohup on the rental; tmux is not used, so nothing depends on it being installed).
# usage: launch_rental.ps1 -SshHost <host> [-SshPort <port>] [-SshUser root] -SshKey <operator key file> -DataRoot <dir>
#            -HostKey <dispatch host name, e.g. rental01> -Record <record> -RecordSha256 <hash> -ExecutorSha256 <reviewed hash>
#            -Slots <K 1..4> -Slot <0..K-1> -PlanList <local .list file> [-GpuMemoryWaitMs <ms>] [-ExecutorPath <file>]
#            [-DescribeOnly]
# The plan list is used as tools\dispatch.py wrote it (format usvbench-g12-planlist-v1; the batch files sit next to it);
# a list named dispatch_<seq>_<host>_k<K>s<slot>.list must name this -HostKey, -Slots and -Slot. Every file is
# hash-checked here, again on the rental after the copy (verify-toolkit: SHA-256 of every file against the toolkit
# manifest), again by the remote launch step and again by the executor. The remote step refuses a foreign python that
# persists for 120 s, installs <usvb>/queues_v2/<task>.py, <task>_plans\ and the wrapper <task>.sh (all CreateNew),
# starts the wrapper detached and proves the executor alive (its process and the first batch's progress file within
# 240 s). The task is named dose_<host>_k<K>s<slot>_<stamp> (never reused). -GpuMemoryWaitMs goes unchanged to the
# executor (0 = its named default: 10 min for K <= 2, 120 min for K >= 3; otherwise 60000..14400000 ms).
# -DescribeOnly prints the plan only (no ssh, no scp, nothing written).
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [ValidateRange(1, 65535)][int]$SshPort = 22,
    [string]$SshUser = 'root',
    [Parameter(Mandatory=$true)][string]$SshKey,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]{1,32}$')][string]$HostKey,
    [Parameter(Mandatory=$true)][string]$Record,
    [Parameter(Mandatory=$true)][string]$RecordSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9A-Fa-f]{64}$')][string]$ExecutorSha256,
    [Parameter(Mandatory=$true)][ValidateRange(1, 4)][int]$Slots,
    [Parameter(Mandatory=$true)][ValidateRange(0, 3)][int]$Slot,
    [Parameter(Mandatory=$true)][string]$PlanList,
    [ValidateRange(0, 14400000)][int]$GpuMemoryWaitMs = 0,
    [string]$ExecutorPath = '',
    [switch]$DescribeOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\r3_local_common.ps1')
if ($Slot -ge $Slots) { throw "Slot $Slot must be below Slots $Slots" }
if ($GpuMemoryWaitMs -gt 0 -and $GpuMemoryWaitMs -lt 60000) { throw '-GpuMemoryWaitMs must be 0 (named default) or at least 60000' }
$gpuMemWaitText = if ($GpuMemoryWaitMs -gt 0) { "$GpuMemoryWaitMs (parameter)" } elseif ($Slots -ge 3) { '7200000 (GPUMEMWAITMS_DEFAULT_K3)' } else { '600000 (GPUMEMWAITMS_DEFAULT_K2)' }
$endpoint = New-R3Endpoint $SshHost $SshPort $SshUser $SshKey $DataRoot -Describe:$DescribeOnly
$rec = Read-R3LocalRecord $Record $RecordSha256
$executorLocal = if ($ExecutorPath) { $ExecutorPath } else { Join-Path (Split-Path -Parent $PSScriptRoot) 'executor\usvb_dose_executor.py' }
if ((Split-Path -Leaf $executorLocal) -cne 'usvb_dose_executor.py') { throw '-ExecutorPath must name a file called usvb_dose_executor.py' }
$executorHash = (Get-FileHash -LiteralPath $executorLocal -Algorithm SHA256).Hash
if ($executorHash -cne $ExecutorSha256.ToUpperInvariant()) { throw "Local executor differs from the reviewed hash: $executorHash" }
$remoteTool = Join-Path (Split-Path -Parent $executorLocal) 'r3_remote.py'
if (-not (Test-Path -LiteralPath $remoteTool -PathType Leaf)) { throw "r3_remote.py missing next to the executor: $remoteTool" }
$listItem = Get-Item -LiteralPath $PlanList -ErrorAction Stop
if ($listItem.Name -cnotmatch '^[A-Za-z0-9_.\-]+\.list$') { throw "Plan list name not allowed: $($listItem.Name)" }
if ($listItem.Name -cmatch '^dispatch_[0-9]{4}_([A-Za-z0-9_-]+)_k([1-4])s([0-3])\.list$') {
    if ($Matches[1] -cne $HostKey -or [int]$Matches[2] -ne $Slots -or [int]$Matches[3] -ne $Slot) { throw "Dispatch list $($listItem.Name) is not for host $HostKey K=$Slots slot=$Slot" }
}
$listHash = (Get-FileHash -LiteralPath $listItem.FullName -Algorithm SHA256).Hash
$listLines = @([IO.File]::ReadAllLines($listItem.FullName, [Text.Encoding]::ASCII) | Where-Object { $_ })
if ($listLines.Count -lt 2 -or $listLines[0] -cne 'format=usvbench-g12-planlist-v1') { throw "Unexpected plan list format: $PlanList" }
$sendFiles = [Collections.Generic.List[string]]::new()
$sendFiles.Add($executorLocal)
$sendFiles.Add($remoteTool)
$sendFiles.Add((Get-Item -LiteralPath $Record).FullName)
$sendFiles.Add($listItem.FullName)
$firstOut = ''
foreach ($listLine in @($listLines | Select-Object -Skip 1)) {
    if ($listLine -cnotmatch '^batch\|([A-Za-z0-9_.-]+\.plan)\|([0-9A-Fa-f]{64})\|([1-9])\|([1-9][0-9]{0,5})$') { throw "Bad plan list line: $listLine" }
    $batchName = $Matches[1]; $batchSha = $Matches[2].ToUpperInvariant(); $batchAttempt = $Matches[3]; $batchFrom = $Matches[4]
    $batchPath = Join-Path $listItem.DirectoryName $batchName
    if ((Get-FileHash -LiteralPath $batchPath -Algorithm SHA256).Hash -cne $batchSha) { throw "Batch file differs from the list: $batchPath" }
    $batchId = @([IO.File]::ReadAllLines($batchPath, [Text.Encoding]::ASCII) | Where-Object { $_.StartsWith('batch_id=') })
    if ($batchId.Count -ne 1 -or $batchId[0] -cnotmatch '^batch_id=[A-Za-z0-9_-]{1,60}$') { throw "No valid batch_id in $batchPath" }
    $rentalTimeouts = @([IO.File]::ReadAllLines($batchPath, [Text.Encoding]::ASCII) | Where-Object { $_.StartsWith('ceiling|RENTAL|') })
    if ($rentalTimeouts.Count -ne 1) { throw "Batch file declares no RENTAL ceiling: $batchPath" }
    if (-not $firstOut) { $firstOut = "$($endpoint.Usvb)/dose_$($batchId[0].Substring(9))_a$batchAttempt" }
    Write-Host "BATCH $batchName sha256=$batchSha attempt=$batchAttempt from_order=$batchFrom $($batchId[0])"
    $sendFiles.Add($batchPath)
}
$runStamp = if ($DescribeOnly) { '20261002T000000Z_000000' } else { New-R3Stamp }
$taskName = "dose_$($HostKey)_k$($Slots)s$($Slot)_$runStamp"
$runsRoot = Join-Path $PSScriptRoot 'runs'
$stagingRoot = Join-Path $runsRoot "launch_$taskName"
if (-not $DescribeOnly) {
    if (-not (Test-Path -LiteralPath $runsRoot)) { New-Item -Path $runsRoot -ItemType Directory | Out-Null }
    New-Item -Path $stagingRoot -ItemType Directory -ErrorAction Stop | Out-Null
    $script:r3LocalLog = Join-Path $runsRoot "launch_$taskName.log"
    [IO.File]::WriteAllText($script:r3LocalLog, "R3 launch $taskName executor_sha256=$executorHash record_sha256=$($RecordSha256.ToUpperInvariant()) planlist_sha256=$listHash`r`n", [Text.Encoding]::ASCII)
}
Write-R3Log "LAUNCH-PLAN host=$HostKey endpoint=$($endpoint.Target):$($endpoint.Port) data_root=$DataRoot slot=$Slot/$Slots task=$taskName plan_list=$($listItem.Name) planlist_sha256=$listHash first_queue_dir=$firstOut executor_sha256=$executorHash release=$($rec.release_id) gpumemwaitms=$gpuMemWaitText detach=setsid+nohup"
$renames = @{ (Split-Path -Leaf $Record) = 'release_record_v2.json' }
$toolkit = Send-R3Toolkit $endpoint "launch_$HostKey" @($sendFiles.ToArray()) $renames $stagingRoot -Describe:$DescribeOnly
$stepArgs = @('launch', '--data-root', $DataRoot, '--toolkit', $toolkit, '--host-key', $HostKey, '--record-sha256', $RecordSha256.ToUpperInvariant(),
    '--executor-sha256', $executorHash, '--task', $taskName, '--slots', [string]$Slots, '--slot', [string]$Slot,
    '--plan-list-file', $listItem.Name, '--plan-list-sha256', $listHash, '--gpu-memory-wait-ms', [string]$GpuMemoryWaitMs)
Invoke-R3Step $endpoint "$toolkit/r3_remote.py" $stepArgs @(0) 1 -Describe:$DescribeOnly
if ($DescribeOnly) { 'DESCRIBE-ONLY: no ssh, no scp, nothing written' }
else { Write-R3Log "LAUNCHED task=$taskName first_queue_dir=$firstOut console=$($endpoint.Usvb)/queues_v2/$($taskName)_console.log status: status_rental.ps1 -Task $taskName" }
