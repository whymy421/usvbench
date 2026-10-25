# R3 status (controller-run, read-only): ONE line per slot task for the controller's watcher, from the rental's
# r3_remote.py status (the copy installed with that task in <usvb>/queues_v2/<task>_plans/). Fields: state
# (RUNNING | STARTING | FINISHED | ABORTED | DEAD | NOT-STARTED), current batch, batches done, cells ended / selected,
# success, errors, timeouts, not-started-ceiling, current cell and its age (s, from its INFLIGHT file), age of the last
# log write (s, progress file and current cell log / heartbeat), GPU utilisation and memory (nvidia-smi), executor PIDs.
# An unreachable rental prints state=UNREACHABLE with the ssh exit code. Writes nothing locally or remotely.
# usage: status_rental.ps1 -SshHost <host> [-SshPort <port>] [-SshUser root] -SshKey <key file> -DataRoot <dir>
#            -Task <dose_<host>_k<K>s<s>_<stamp>>[,<task>...] [-DescribeOnly]
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [ValidateRange(1, 65535)][int]$SshPort = 22,
    [string]$SshUser = 'root',
    [Parameter(Mandatory=$true)][string]$SshKey,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][string[]]$Task,
    [switch]$DescribeOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\r3_local_common.ps1')
$endpoint = New-R3Endpoint $SshHost $SshPort $SshUser $SshKey $DataRoot -Describe:$DescribeOnly
$Task = @($Task | ForEach-Object { $_.Split(',') } | Where-Object { $_ })
foreach ($oneTask in @($Task)) {
    if ($oneTask -cnotmatch '^dose_[A-Za-z0-9_-]{1,32}_k[1-4]s[0-3]_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{6}$') { throw "Task name not allowed: $oneTask" }
    $toolArgs = Get-R3ToolCommand $endpoint "$($endpoint.Usvb)/queues_v2/$($oneTask)_plans/r3_remote.py" @('status', '--data-root', $DataRoot, '--task', $oneTask)
    if ($DescribeOnly) { "----- STATUS ssh $(@(Get-R3SshOptions $endpoint) -join ' ') $($endpoint.Target) $(@($toolArgs) -join ' ') -----"; continue }
    $result = Invoke-R3LocalNative 'ssh' (@(Get-R3SshOptions $endpoint) + @($endpoint.Target) + @($toolArgs))
    $statusLines = @($result.Lines | Where-Object { $_ -cmatch '^R3-STATUS ' })
    if ($result.Exit -eq 0 -and $statusLines.Count -eq 1) { $statusLines[0] }
    else { "R3-STATUS task=$oneTask state=UNREACHABLE ssh_exit=$($result.Exit) detail=$(@($result.Lines | Select-Object -Last 2) -join ' | ')" }
}
if ($DescribeOnly) { 'DESCRIBE-ONLY: no ssh, nothing written' }
