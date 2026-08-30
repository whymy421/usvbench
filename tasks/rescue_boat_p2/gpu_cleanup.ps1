# =============================================================
# gpu_cleanup.ps1 — Clear stale Isaac Sim / CUDA processes
#
# Isaac Sim does not always exit cleanly. A leftover python.exe keeps
# the GPU context alive, and the next run then dies during startup
# before it can even open its log file — which is why a failed run
# leaves no log at all.
#
# Run this before any training / eval script, or whenever a run
# "doesn't start".
#
#   .\rescue_boat\gpu_cleanup.ps1
# =============================================================

Write-Host "=================================================" -ForegroundColor Yellow
Write-Host "  GPU / PROCESS CLEANUP"                           -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Yellow

# ── 1. What is currently on the GPU ──────────────────────────────────────────
Write-Host ""
Write-Host "Processes currently holding the GPU:" -ForegroundColor Cyan
$smi = "C:\Windows\System32\nvidia-smi.exe"
if (Test-Path $smi) {
    & $smi --query-compute-apps=pid,process_name,used_memory --format=csv
} else {
    Write-Host "  nvidia-smi not found at $smi" -ForegroundColor DarkGray
}

# ── 2. Stale python / Isaac Sim processes ────────────────────────────────────
Write-Host ""
Write-Host "Python / Isaac Sim processes:" -ForegroundColor Cyan
$procs = Get-Process -Name "python", "kit", "isaac-sim*" -ErrorAction SilentlyContinue

if (-not $procs) {
    Write-Host "  none found - GPU should be free" -ForegroundColor Green
} else {
    $procs | Select-Object Id, ProcessName,
        @{N = "MemMB"; E = { [math]::Round($_.WorkingSet64 / 1MB) } },
        @{N = "Started"; E = { $_.StartTime } } | Format-Table -AutoSize

    Write-Host "Killing them..." -ForegroundColor Yellow
    foreach ($p in $procs) {
        try {
            Stop-Process -Id $p.Id -Force -ErrorAction Stop
            Write-Host "  killed PID $($p.Id) ($($p.ProcessName))" -ForegroundColor Green
        } catch {
            Write-Host "  could not kill PID $($p.Id): $_" -ForegroundColor Red
        }
    }
}

# ── 3. Wait for the driver to release memory ─────────────────────────────────
Write-Host ""
Write-Host "Waiting 20s for the driver to release VRAM..." -ForegroundColor Cyan
Start-Sleep -Seconds 20

if (Test-Path $smi) {
    Write-Host ""
    Write-Host "GPU state after cleanup:" -ForegroundColor Cyan
    & $smi --query-gpu=memory.used,memory.total --format=csv
}

Write-Host ""
Write-Host "Cleanup done - safe to start the next run." -ForegroundColor Green
