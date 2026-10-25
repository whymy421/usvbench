# R3 local helpers (controller side, Windows PowerShell 5.1), dot-sourced by launch_rental.ps1, pull_rental.ps1,
# status_rental.ps1 and abort_rental.ps1. Adapted from G12 launcher\lib\v2_local_common.ps1 (a verbatim G2 copy,
# SHA-256 4FF76608...49B21F7): the endpoint comes from parameters (host, port, user, key path) instead of the record's
# host entry, scp takes -P for the port, remote arguments are checked for the rental's bash, and every remote step is
# r3_remote.py run as '<data root>/usvb/env/bin/python -X usvbexec=r3tool -B -I <tool> <command> ...'.
# Pure ASCII. Every remote step works on a fresh toolkit folder <data root>/usvb/r3kit_<purpose>_<stamp> (scp -r of a
# new local folder; a new stamp per try, so nothing is ever copied over) whose files are checked by SHA-256 on the
# rental (verify-toolkit) before use. Native calls are judged by exit code only. No argument ends with a backslash.
Set-Variable -Name SSH_ALIVE_INTERVAL_S -Value 15 -Option Constant -Scope Script
Set-Variable -Name SSH_ALIVE_COUNT_MAX -Value 4 -Option Constant -Scope Script
Set-Variable -Name SSH_CONNECT_TIMEOUT_S -Value 30 -Option Constant -Scope Script
Set-Variable -Name SSH_LINK_FAILURE_EXIT -Value 255 -Option Constant -Scope Script
Set-Variable -Name TOOLKIT_TRIES -Value 4 -Option Constant -Scope Script
Set-Variable -Name STEP_TRIES -Value 3 -Option Constant -Scope Script
Set-Variable -Name STEP_WAIT_MS -Value 20000 -Option Constant -Scope Script

function Write-R3Log([string]$logLine) {
    $stamped = "$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) $logLine"
    Write-Host $stamped
    if ($script:r3LocalLog) {
        if ($stamped -match '[^\x20-\x7E]') { $stamped = [regex]::Replace($stamped, '[^\x20-\x7E]', '?') }
        [IO.File]::AppendAllText($script:r3LocalLog, $stamped + "`r`n", [Text.Encoding]::ASCII)
    }
}

function Read-R3LocalRecord([string]$recordPath, [string]$recordSha256) {
    if ($recordSha256 -cnotmatch '^[0-9A-Fa-f]{64}$') { throw 'RecordSha256 must be 64 hex characters' }
    $measured = (Get-FileHash -LiteralPath $recordPath -Algorithm SHA256 -ErrorAction Stop).Hash
    if ($measured -cne $recordSha256.ToUpperInvariant()) { throw "RECORD-HASH-MISMATCH $recordPath measured=$measured given=$recordSha256" }
    $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json -ErrorAction Stop
    if ($record.format -cne 'usvbench-release-record-v2') { throw "Unexpected record format: $($record.format)" }
    return $record
}

function New-R3Endpoint([string]$sshHost, [int]$sshPort, [string]$sshUser, [string]$sshKey, [string]$dataRoot, [switch]$Describe) {
    if ($sshHost -cnotmatch '^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$') { throw "SshHost not allowed: $sshHost" }
    if ($sshPort -lt 1 -or $sshPort -gt 65535) { throw "SshPort out of range: $sshPort" }
    if ($sshUser -cnotmatch '^[a-z_][a-z0-9_-]{0,31}$') { throw "SshUser not allowed: $sshUser" }
    if ($dataRoot -cnotmatch '^/[A-Za-z0-9_./:+-]+$' -or $dataRoot.EndsWith('/') -or $dataRoot.Contains('//')) { throw "DataRoot must be an absolute POSIX path of [A-Za-z0-9_./:+-] without a trailing slash: $dataRoot" }
    if (-not $Describe -and -not (Test-Path -LiteralPath $sshKey -PathType Leaf)) { throw "ssh key missing: $sshKey (the key belongs to the operator; only its location is passed)" }
    return [pscustomobject]@{ Target="$sshUser@$sshHost"; Port=[string]$sshPort; Key=$sshKey; DataRoot=$dataRoot; Usvb="$dataRoot/usvb"; Python="$dataRoot/usvb/env/bin/python" }
}

# G40 copy: the only change against R3 3CC04E18... is this file name for the known-hosts file, passed to every ssh and scp
# call, because the AutoDL host keys were accepted into it (not into the default known_hosts).
$G40_KNOWN_HOSTS = 'C:/Users/BRADY/.ssh/known_hosts_autodl'

function Get-R3SshOptions($endpoint) {
    return @('-i', $endpoint.Key, '-p', $endpoint.Port, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', "UserKnownHostsFile=$G40_KNOWN_HOSTS",
        '-o', "ServerAliveInterval=$SSH_ALIVE_INTERVAL_S", '-o', "ServerAliveCountMax=$SSH_ALIVE_COUNT_MAX",
        '-o', "ConnectTimeout=$SSH_CONNECT_TIMEOUT_S")
}

# G51 copy: scp gets '-l <kbit/s>' when G51_SCP_LIMIT_KBIT is a positive integer (owner upload rule, daytime cap).
function Get-R3ScpOptions($endpoint) {
    $limit = @()
    if ($env:G51_SCP_LIMIT_KBIT -cmatch '^[1-9][0-9]{0,6}$') { $limit = @('-l', $env:G51_SCP_LIMIT_KBIT) }
    return @($limit) + @('-i', $endpoint.Key, '-P', $endpoint.Port, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', "UserKnownHostsFile=$G40_KNOWN_HOSTS",
        '-o', "ServerAliveInterval=$SSH_ALIVE_INTERVAL_S", '-o', "ServerAliveCountMax=$SSH_ALIVE_COUNT_MAX",
        '-o', "ConnectTimeout=$SSH_CONNECT_TIMEOUT_S")
}

function Invoke-R3LocalNative([string]$exe, [string[]]$arguments) {
    foreach ($argument in @($arguments)) { if ($argument.EndsWith('\')) { throw "Argument ends with a backslash (PowerShell 5.1 mangles it): $argument" } }
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = @(& $exe @arguments 2>&1 | ForEach-Object { "$_" })
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $saved }
    return [pscustomobject]@{ Exit=$code; Lines=@($lines) }
}

function Assert-R3RemoteArgs([string[]]$remoteArgs) {
    # ssh joins the remote arguments with spaces into ONE command line that the rental's bash parses: only
    # shell-neutral tokens pass (letters, digits, _ . : = / , + -); no space, quote, backslash, $ ; & | < > ( ) * ? ~ #.
    foreach ($remoteArg in @($remoteArgs)) {
        if ($remoteArg -cnotmatch '^[A-Za-z0-9_.:=/,+-]+$') { throw "Remote argument is not shell-neutral: '$remoteArg'" }
    }
}

function New-R3Stamp {
    return (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 6)
}

function Get-R3ToolCommand($endpoint, [string]$toolPath, [string[]]$toolArgs) {
    $remoteArgs = @($endpoint.Python, '-X', 'usvbexec=r3tool', '-B', '-I', $toolPath) + @($toolArgs)
    Assert-R3RemoteArgs $remoteArgs
    return @($remoteArgs)
}

function Send-R3Toolkit($endpoint, [string]$purpose, [string[]]$files, [hashtable]$renames, [string]$stagingRoot, [switch]$Describe) {
    # Returns the remote toolkit folder after r3_remote.py verify-toolkit confirmed every file's SHA-256 on the rental.
    if ($Describe) {
        Write-Host "----- TOOLKIT ${purpose}: scp -r $(@(Get-R3ScpOptions $endpoint) -join ' ') <new local folder r3kit_$($purpose)_<stamp>> $($endpoint.Target):$($endpoint.Usvb) ; files: $(@($files | ForEach-Object { $leaf = Split-Path -Leaf $_; if ($renames.ContainsKey($leaf)) { $renames[$leaf] } else { $leaf } }) -join ', ') + TOOLKIT_MANIFEST.txt (name=SHA256 lines); then ssh ... $($endpoint.Python) -X usvbexec=r3tool -B -I <toolkit>/r3_remote.py verify-toolkit --dir <toolkit> --manifest-sha256 <SHA-256 of TOOLKIT_MANIFEST.txt> -----"
        return "$($endpoint.Usvb)/r3kit_DESCRIBE"
    }
    for ($toolkitTry = 1; $toolkitTry -le $TOOLKIT_TRIES; $toolkitTry++) {
        $folderName = "r3kit_$($purpose)_$(New-R3Stamp)"
        $local = Join-Path $stagingRoot $folderName
        New-Item -Path $local -ItemType Directory -ErrorAction Stop | Out-Null
        foreach ($file in @($files)) {
            $leaf = Split-Path -Leaf $file
            $target = if ($renames.ContainsKey($leaf)) { $renames[$leaf] } else { $leaf }
            Copy-Item -LiteralPath $file -Destination (Join-Path $local $target)
        }
        $manifestLines = @(Get-ChildItem -LiteralPath $local -File | Sort-Object Name | ForEach-Object { "$($_.Name)=$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)" })
        $manifestLocal = Join-Path $local 'TOOLKIT_MANIFEST.txt'
        [IO.File]::WriteAllText($manifestLocal, ((@($manifestLines) -join "`r`n") + "`r`n"), [Text.Encoding]::ASCII)
        $manifestHash = (Get-FileHash -LiteralPath $manifestLocal -Algorithm SHA256).Hash
        $remote = "$($endpoint.Usvb)/$folderName"
        $checkArgs = Get-R3ToolCommand $endpoint "$remote/r3_remote.py" @('verify-toolkit', '--dir', $remote, '--manifest-sha256', $manifestHash)
        $copy = Invoke-R3LocalNative 'scp' (@('-r') + @(Get-R3ScpOptions $endpoint) + @($local, "$($endpoint.Target):$($endpoint.Usvb)"))
        foreach ($line in @($copy.Lines)) { Write-R3Log "[scp toolkit] $line" }
        if ($copy.Exit -eq 0) {
            $check = Invoke-R3LocalNative 'ssh' (@(Get-R3SshOptions $endpoint) + @($endpoint.Target) + @($checkArgs))
            foreach ($line in @($check.Lines)) { Write-R3Log "[verify-toolkit] $line" }
            if ($check.Exit -eq 0) { Write-R3Log "TOOLKIT-READY $remote"; return $remote }
            Write-R3Log "TOOLKIT-CHECK-FAILED try=$toolkitTry exit=$($check.Exit) folder=$remote (kept on the rental; a new folder is used next)"
        } else { Write-R3Log "TOOLKIT-SCP-FAILED try=$toolkitTry exit=$($copy.Exit) folder=$remote (kept on the rental if partial)" }
        if ($toolkitTry -lt $TOOLKIT_TRIES) { Start-Sleep -Milliseconds $STEP_WAIT_MS }
    }
    throw "Toolkit transfer failed $TOOLKIT_TRIES times"
}

function Invoke-R3Step($endpoint, [string]$toolPath, [string[]]$toolArgs, [int[]]$acceptedExits, [int]$linkTries, [switch]$Describe) {
    # Runs r3_remote.py on the rental; retried only on an ssh link failure (exit 255).
    $script:r3LastExit = 0
    $script:r3LastLines = @()
    $remoteArgs = Get-R3ToolCommand $endpoint $toolPath $toolArgs
    if ($Describe) {
        "----- STEP ssh $(@(Get-R3SshOptions $endpoint) -join ' ') $($endpoint.Target) $(@($remoteArgs) -join ' ') (accepted exits $(@($acceptedExits) -join ','); link tries $linkTries; arguments shell-neutral) -----"
        return
    }
    for ($stepTry = 1; $stepTry -le $linkTries; $stepTry++) {
        $result = Invoke-R3LocalNative 'ssh' (@(Get-R3SshOptions $endpoint) + @($endpoint.Target) + @($remoteArgs))
        foreach ($line in @($result.Lines)) { Write-R3Log "[$($toolArgs[0])] $line" }
        if (@($acceptedExits) -contains $result.Exit) {
            $script:r3LastExit = $result.Exit
            $script:r3LastLines = @($result.Lines)
            return
        }
        if ($result.Exit -ne $SSH_LINK_FAILURE_EXIT) { throw "Remote step $($toolArgs[0]) failed: exit $($result.Exit)" }
        Write-R3Log "LINK-FAILURE step=$($toolArgs[0]) try=$stepTry/$linkTries"
        if ($stepTry -lt $linkTries) { Start-Sleep -Milliseconds $STEP_WAIT_MS }
    }
    throw "Remote step $($toolArgs[0]): link failed $linkTries time(s)"
}
