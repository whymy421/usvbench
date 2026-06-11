# =============================================================
# USVBench — Train: Boat calm-water navigation (reference task B)
# Reference: V40 = 5.64 targets/episode @ 3000 iter, seed 42 (RTX 5080, ~1.5 h)
# wandb: https://wandb.ai/whymysong321-university-of-southampton/usvbench/runs/9yupkmf6
# =============================================================
#
# Before running:
#   1. Copy  tasks/boat_calm_nav/  into your Isaac Lab tasks dir:
#        <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/boat_calm_nav/
#   2. Edit the $ISAACLAB line below to point at YOUR Isaac Lab root.
#   3. conda activate isaaclab
#   4. .\scripts\train_boat_calm.ps1
# -------------------------------------------------------------

$ErrorActionPreference = "Stop"

# ---- EDIT THIS: path to your Isaac Lab repo root ----
$ISAACLAB = "C:\Users\Yutong\NavRL\IsaacLab"
# -----------------------------------------------------

# Asset dir auto-resolved to this repo's assets/ folder (no manual setup needed)
$env:USVBENCH_ASSETS = (Resolve-Path "$PSScriptRoot\..\assets").Path
$env:PYTHONIOENCODING = "utf-8"

# Task B config: boat, calm water, V40 "side-approach" reward
#   The simple ROV recipe (forward x exp(alignment)) FAILS on the boat (points
#   backwards, ~0.1 tgt/ep). The boat needs the side-approach reward below.
$env:OBS_DIM = "9"           # 9D obs: nav + self-state (speed, yaw-rate, ...)
$env:OBS_EXTENDED = "1"      # enable the extra self-state channels
$env:REWARD_VARIANT = "V23"  # speed-coupled nav reward
$env:SPEED_COUPLE = "1"      # reward only counts when the boat is moving
$env:REACH_BONUS = "50.0"    # boat is slow; +10 gets eaten by respawn travel cost
$env:SIDE_APPROACH = "1"     # don't require bow-alignment; side/stern entry counts as reach
# NOTE: do NOT set FORWARD_TRANSIT=1 — that was V41 and it made the boat flee (0.04 tgt/ep).

$env:WANDB_PROJECT = "usvbench"
$env:WANDB_NAME = "boat_calm_nav_s42"

$TRAIN = "$ISAACLAB\scripts\reinforcement_learning\skrl\train_with_eval.py"

python $TRAIN `
    --task=Isaac-My-First-Task-Calm-Boat-Direct-v1 `
    --num_envs=64 --headless --max_iterations=3000 --seed=42 `
    --video --video_interval 50000 --video_length 200
