# R3 abort (controller-run): SIGTERM to ONE slot task's executor on a rental (r3_remote.py abort, the copy installed with
# that task). The executor stops the current cell's process group (SIGTERM, SIGKILL after its grace), writes
# dose_ABORTED.txt with reason INTERRUPTED and resume_from_order, stops its list and releases its locks. The remote tool
# waits up to 180 s and never sends SIGKILL itself. Exit 0 aborted, 3 nothing was running, 1 still running after the wait.
# A second slot of the same host is never touched (only processes carrying this task's executor path and its usvbexec
# tag are signalled). Resume: pull the aborted directory, then dispatch.py --reattempt <batch>:<resume_from_order> on the
# SAME host and launch the new list.
# usage: abort_rental.ps1 -SshHost <host> [-SshPort <port>] [-SshUser root] -SshKey <key file> -DataRoot <dir>
#            -Task <dose_<host>_k<K>s<s>_<stamp>> [-DescribeOnly]
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [ValidateRange(1, 65535)][int]$SshPort = 22,
    [string]$SshUser = 'root',
    [Parameter(Mandatory=$true)][string]$SshKey,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^dose_[A-Za-z0-9_-]{1,32}_k[1-4]s[0-3]_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{6}$')][string]$Task,
    [switch]$DescribeOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\r3_local_common.ps1')
$endpoint = New-R3Endpoint $SshHost $SshPort $SshUser $SshKey $DataRoot -Describe:$DescribeOnly
if (-not $DescribeOnly) {
    $runsRoot = Join-Path $PSScriptRoot 'runs'
    if (-not (Test-Path -LiteralPath $runsRoot)) { New-Item -Path $runsRoot -ItemType Directory | Out-Null }
    $script:r3LocalLog = Join-Path $runsRoot "abort_$($Task)_$(New-R3Stamp).log"
    [IO.File]::WriteAllText($script:r3LocalLog, "R3 abort $Task endpoint=$($endpoint.Target):$($endpoint.Port)`r`n", [Text.Encoding]::ASCII)
}
Invoke-R3Step $endpoint "$($endpoint.Usvb)/queues_v2/$($Task)_plans/r3_remote.py" @('abort', '--data-root', $DataRoot, '--task', $Task) @(0, 1, 3) 1 -Describe:$DescribeOnly
if ($DescribeOnly) { 'DESCRIBE-ONLY: no ssh, nothing written' }
else {
    Write-R3Log "ABORT-STEP-EXIT $($script:r3LastExit) task=$Task (0 aborted, 3 nothing running, 1 still running after the wait)"
    exit $script:r3LastExit
}
