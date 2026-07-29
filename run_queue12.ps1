$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$py = "E:\anaconda\envs\isaaclab\python.exe"
$repo = "C:\Users\BRADY\usvbench"
$train = "C:\Users\BRADY\IsaacLab\scripts\reinforcement_learning\skrl\train.py"
$out = "$repo\queue12"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $repo

function Beacon($msg) {
    "$(Get-Date -Format 'HH:mm:ss') $msg" | Out-File -Append -Encoding UTF8 "$out\progress.txt"
}

function TrainAndScreen($task, $seed, $name, $steps, $screenTask) {
    Beacon "TRAIN $name task=$task seed=$seed steps=$steps"
    & $py $train --task $task --num_envs 64 --seed $seed --headless `
        agent.agent.experiment.experiment_name=$name `
        agent.agent.experiment.wandb=False `
        agent.trainer.timesteps=$steps *> "$out\train_$name.log"
    $rd = Get-ChildItem "$repo\logs\skrl\usvbench" -Directory -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -like "*$name*" } | Sort-Object CreationTime -Desc | Select-Object -First 1
    if (-not $rd) {
        $rd = Get-ChildItem "C:\Users\BRADY\IsaacLab\logs\skrl\usvbench" -Directory -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -like "*$name*" } | Sort-Object CreationTime -Desc | Select-Object -First 1
    }
    if (-not $rd) { Beacon "  $name : NO RUN DIR"; return }
    Beacon "  run_dir=$($rd.FullName)"
    & $py "$repo\scripts\screen_v6_ladder.py" --run-dir $rd.FullName --task $screenTask `
        --every 2 --level 0 --eval-seed 42 --headless *> "$out\screen_$name.log"
    $best = Select-String -Path "$out\screen_$name.log" -Pattern 'LADDER (agent_\d+\.pt): SR=([\d.]+)' -AllMatches |
            ForEach-Object { $_.Matches } | ForEach-Object { [PSCustomObject]@{ ck = $_.Groups[1].Value; sr = [double]$_.Groups[2].Value } } |
            Sort-Object sr -Descending | Select-Object -First 1
    if ($best) {
        Beacon "  $name BEST $($best.ck) SR=$($best.sr)"
        foreach ($es in 42, 123) {
            & $py "$repo\scripts\eval_v6_frozen.py" --checkpoint "$($rd.FullName)\checkpoints\$($best.ck)" `
                --task $screenTask --episodes 128 --level 0 --eval-seed $es `
                --out "$out\cert_${name}_e$es.json" --headless *> "$out\cert_${name}_e$es.log"
            $line = Select-String -Path "$out\cert_${name}_e$es.log" -Pattern 'episodes=.*SR=' | Select-Object -First 1
            Beacon "  CERT $name e${es}: $($line.Line.Trim())"
        }
    } else {
        Beacon "  $name : no ladder lines"
    }
}

Beacon "QUEUE12 START"

# --- Block 1: does the policy-invariant potential change behaviour? ---------
# Pre-registered: v4 vs the v3 baseline (75.8% SR / 24.2% collision). The raw
# difference adds a standing "cut the straight-line distance now" pressure, so
# the prediction is a LOWER collision rate at comparable success.
foreach ($s in 42, 43) {
    TrainAndScreen "Isaac-USV-HazardNav-Direct-v4" $s "pbrs_s$s" 96000 "Isaac-USV-HazardNav-Direct-v4"
}
Beacon "=== BLOCK 1 DONE (PBRS) ==="

# --- Block 2: baseline re-run for a matched comparison ---------------------
# Same budget, same seeds, uncorrected potential. Without this the v4 numbers
# have nothing to be compared against on this machine.
foreach ($s in 42, 43) {
    TrainAndScreen "Isaac-USV-HazardNav-Direct-v3" $s "v3base_s$s" 96000 "Isaac-USV-HazardNav-Direct-v3"
}
Beacon "=== BLOCK 2 DONE (matched baseline) ==="

# --- Block 3: feasibility pooling in the observation -----------------------
# Pre-registered: if the failure is "a passable gap looks like two threats",
# pooling should raise success without raising collisions.
foreach ($s in 42, 43) {
    TrainAndScreen "Isaac-USV-HazardNav-Direct-v5" $s "feas_s$s" 96000 "Isaac-USV-HazardNav-Direct-v5"
}
Beacon "=== BLOCK 3 DONE (feasibility pooling) ==="

# --- Block 4: the owner's one-shot half-sine threading bonus ---------------
# Pre-registered: v11 already commits (all its failures are crashes, zero
# timeouts), so the prediction here is a LOWER collision rate rather than a
# higher commit rate -- the bonus only pays for CLEAN passages.
foreach ($s in 42, 43) {
    TrainAndScreen "Isaac-USV-HazardNav-Direct-v6" $s "thread_s$s" 96000 "Isaac-USV-HazardNav-Direct-v6"
}
Beacon "=== BLOCK 4 DONE (threading bonus) ==="

Beacon "QUEUE12 DONE"
