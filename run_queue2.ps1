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

function TrainAndScreen($task, $seed, $name, $steps) {
    Beacon "TRAIN $name task=$task seed=$seed"
    # checkpoint_interval matters a LOT here: the v11 agent cfg saves every 256
    # steps, which produced 375 checkpoints and turned screening into a 4.8-hour
    # job (longer than the 29-minute training). 3200 gives 30 checkpoints.
    & $py $train --task $task --num_envs 64 --seed $seed --headless `
        agent.agent.experiment.experiment_name=$name `
        agent.agent.experiment.wandb=False `
        agent.agent.experiment.checkpoint_interval=3200 `
        agent.trainer.timesteps=$steps *> "$out\train_$name.log"
    $rd = Get-ChildItem "$repo\logs\skrl\usvbench" -Directory -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -like "*$name*" } | Sort-Object CreationTime -Desc | Select-Object -First 1
    if (-not $rd) { Beacon "  $name : NO RUN DIR"; return }
    $n = (Get-ChildItem "$($rd.FullName)\checkpoints" -Filter '*.pt' -ErrorAction SilentlyContinue).Count
    Beacon "  $name run_dir ok, $n checkpoints"
    & $py "$repo\scripts\screen_v6_ladder.py" --run-dir $rd.FullName --task $task `
        --every 2 --level 0 --eval-seed 42 --headless *> "$out\screen_$name.log"
    $best = Select-String -Path "$out\screen_$name.log" -Pattern 'LADDER (agent_\d+\.pt): SR=([\d.]+)' -AllMatches |
            ForEach-Object { $_.Matches } | ForEach-Object { [PSCustomObject]@{ ck = $_.Groups[1].Value; sr = [double]$_.Groups[2].Value } } |
            Sort-Object sr -Descending | Select-Object -First 1
    if (-not $best) { Beacon "  $name : no ladder lines"; return }
    Beacon "  $name BEST $($best.ck) SR=$($best.sr)"
    foreach ($es in 42, 123) {
        & $py "$repo\scripts\eval_v6_frozen.py" --checkpoint "$($rd.FullName)\checkpoints\$($best.ck)" `
            --task $task --episodes 128 --level 0 --eval-seed $es `
            --out "$out\cert_${name}_e$es.json" --headless *> "$out\cert_${name}_e$es.log"
        $line = Select-String -Path "$out\cert_${name}_e$es.log" -Pattern 'episodes=.*SR=' | Select-Object -First 1
        Beacon "  CERT $name e${es}: $($line.Line.Trim())"
    }
}

Beacon "QUEUE2 START (corrected PBRS + faster screening)"

# Block A: matched baseline first, so every later number has a reference that
# was produced on this machine with this budget.
foreach ($s in 42, 43) { TrainAndScreen "Isaac-USV-HazardNav-Direct-v3" $s "base_s$s" 96000 }
Beacon "=== A DONE (baseline v3) ==="

# Block B: discounted potential difference ONLY. Single change vs baseline.
# The first attempt bundled this with terminal zeroing at timeouts and scored
# 0.0%; that defect is now covered by test_pbrs_terminal.py.
foreach ($s in 42, 43) { TrainAndScreen "Isaac-USV-HazardNav-Direct-v4" $s "pbrsB_s$s" 96000 }
Beacon "=== B DONE (discounted difference) ==="

# Block C: feasibility pooling in the observation.
foreach ($s in 42, 43) { TrainAndScreen "Isaac-USV-HazardNav-Direct-v5" $s "feas_s$s" 96000 }
Beacon "=== C DONE (feasibility pooling) ==="

# Block D: the owner's one-shot half-sine threading arc.
foreach ($s in 42, 43) { TrainAndScreen "Isaac-USV-HazardNav-Direct-v6" $s "thread_s$s" 96000 }
Beacon "=== D DONE (threading arc) ==="

# Block E: difference + zeroing at TRUE terminals only. Runs last because the
# unit test shows a residual step-local pull toward "crash far", so this is the
# variant most likely to misbehave.
TrainAndScreen "Isaac-USV-HazardNav-Direct-v7" 42 "pbrsE_s42" 96000
Beacon "=== E DONE (true-terminal zeroing) ==="

Beacon "QUEUE2 DONE"
