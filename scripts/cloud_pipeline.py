"""One-shot cloud orchestrator: platform check -> Task B curriculum -> crossings.

Runs training/eval as subprocesses, screens checkpoints, chains champions
between curriculum stages, and writes an append-only beacon file so progress
survives disconnects. Every stage records its numbers in results.jsonl.

Usage (on the cloud box):
    nohup python scripts/cloud_pipeline.py > pipeline.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

D = os.environ.get("USVB_ROOT", "/root/autodl-tmp")
PY = f"{D}/envs/isaaclab/bin/python"
TRAIN = f"{D}/IsaacLab/scripts/reinforcement_learning/skrl/train.py"
REPO = f"{D}/usvbench"
LOGS = f"{D}/runlogs"
BEACON = f"{D}/pipeline_progress.txt"
RESULTS = f"{D}/results.jsonl"
SKRL_LOGS = None  # discovered at runtime


def beacon(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(BEACON, "a") as handle:
        handle.write(line + "\n")


def record(**kw) -> None:
    with open(RESULTS, "a") as handle:
        handle.write(json.dumps(kw) + "\n")


def run(cmd: list[str], log_name: str, timeout_s: int = 14400) -> str:
    os.makedirs(LOGS, exist_ok=True)
    path = f"{LOGS}/{log_name}.log"
    with open(path, "w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                              timeout=timeout_s)
    with open(path, errors="ignore") as handle:
        out = handle.read()
    beacon(f"  [{log_name}] exit={proc.returncode}")
    return out


def newest_run_dir(name_fragment: str) -> str | None:
    """Find the skrl run directory created for an experiment name."""
    roots = [f"{D}/logs/skrl/usvbench", f"{D}/IsaacLab/logs/skrl/usvbench",
             os.path.expanduser("~/logs/skrl/usvbench")]
    best, best_t = None, -1.0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            if name_fragment in entry:
                full = os.path.join(root, entry)
                t = os.path.getctime(full)
                if t > best_t:
                    best, best_t = full, t
    return best


def train(task: str, seed: int, name: str, timesteps: int = 96000,
          num_envs: int = 64, warm_start: str | None = None) -> str | None:
    beacon(f"TRAIN {name} task={task} seed={seed} warm={bool(warm_start)}")
    cmd = [PY, TRAIN, "--task", task, "--num_envs", str(num_envs),
           "--seed", str(seed), "--headless",
           f"agent.agent.experiment.experiment_name={name}",
           f"agent.trainer.timesteps={timesteps}"]
    if warm_start:
        cmd += ["--checkpoint", warm_start]
    run(cmd, f"train_{name}")
    rd = newest_run_dir(name)
    beacon(f"  run_dir={rd}")
    return rd


def screen(run_dir: str, task: str, every: int = 6, level: int = 0,
           tag: str = "screen") -> tuple[str, float] | None:
    """Ladder-screen checkpoints; return (best_ckpt_path, best_sr)."""
    if not run_dir:
        return None
    out = run([PY, f"{REPO}/scripts/screen_v6_ladder.py", "--run-dir", run_dir,
               "--task", task, "--every", str(every), "--level", str(level),
               "--eval-seed", "42", "--headless"], f"{tag}")
    best, best_sr = None, -1.0
    for match in re.finditer(r"LADDER (agent_\d+\.pt): SR=([\d.]+)", out):
        sr = float(match.group(2))
        if sr > best_sr:
            best, best_sr = match.group(1), sr
    if best is None:
        beacon(f"  {tag}: no ladder lines parsed")
        return None
    path = os.path.join(run_dir, "checkpoints", best)
    beacon(f"  {tag}: best={best} SR={best_sr:.3f}")
    return path, best_sr


def certify(ckpt: str, task: str, tag: str, level: int = 0,
            episodes: int = 128) -> dict:
    """128-episode frozen eval on both the selection seed and a fresh one."""
    numbers = {}
    for eval_seed in (42, 123):
        out = run([PY, f"{REPO}/scripts/eval_v6_frozen.py", "--checkpoint", ckpt,
                   "--task", task, "--episodes", str(episodes), "--level",
                   str(level), "--eval-seed", str(eval_seed), "--out",
                   f"{D}/cert_{tag}_e{eval_seed}.json", "--headless"],
                  f"cert_{tag}_e{eval_seed}")
        m = re.search(r"SR=([\d.]+) \((\d+)/(\d+)\)", out)
        t = re.search(r"tts median=([\d.]+)s", out)
        c = re.search(r"collision_episodes=(\d+)/(\d+)", out)
        if m:
            numbers[eval_seed] = {
                "sr": float(m.group(1)), "hits": int(m.group(2)),
                "n": int(m.group(3)),
                "tts_median_s": float(t.group(1)) if t else None,
                "collision_rate": (int(c.group(1)) / int(c.group(2))) if c else None,
            }
    beacon(f"  CERT {tag}: {numbers}")
    record(stage=tag, task=task, checkpoint=ckpt, certify=numbers)
    return numbers


def stage_chain(seeds=(42, 43)) -> None:
    """Task B: exit -> exit+transit -> full mission, champion warm-started."""
    warm = None
    for stage in (1, 2, 3):
        task = f"Isaac-USV-HarborStage{stage}-Direct-v1"
        steps = {1: 96000, 2: 96000, 3: 96000}[stage]
        candidates = []
        for seed in seeds:
            name = f"hb_stage{stage}_s{seed}"
            rd = train(task, seed, name, timesteps=steps, warm_start=warm)
            got = screen(rd, task, every=6, tag=f"screen_s{stage}_{seed}")
            if got:
                candidates.append(got)
        if not candidates:
            beacon(f"STAGE {stage}: no candidate survived -- chain stops")
            return
        champ, sr = max(candidates, key=lambda x: x[1])
        beacon(f"STAGE {stage} CHAMPION {champ} screen_SR={sr:.3f}")
        certify(champ, task, f"harbor_stage{stage}")
        warm = champ  # chain into the next stage


def crossing(task: str, tag: str, seeds=(42, 43), timesteps=96000) -> None:
    candidates = []
    for seed in seeds:
        rd = train(task, seed, f"{tag}_s{seed}", timesteps=timesteps)
        got = screen(rd, task, every=6, tag=f"screen_{tag}_{seed}")
        if got:
            candidates.append(got)
    if not candidates:
        beacon(f"{tag}: no candidate")
        return
    champ, sr = max(candidates, key=lambda x: x[1])
    beacon(f"{tag} CHAMPION {champ} screen_SR={sr:.3f}")
    certify(champ, task, tag)


def platform_check() -> None:
    """Reproduce the local v11 number; a mismatch means the box is not equal."""
    ckpt = f"{REPO}/tasks/hazard_nav/checkpoints/hazard_nav_v11_best_s42.pt"
    if not os.path.isfile(ckpt):
        beacon("PLATFORM CHECK SKIPPED (no v11 checkpoint in repo)")
        return
    out = run([PY, f"{REPO}/scripts/eval_v6_frozen.py", "--checkpoint", ckpt,
               "--task", "Isaac-USV-HazardNav-Direct-v3", "--episodes", "128",
               "--level", "0", "--eval-seed", "123", "--headless"],
              "platform_check")
    m = re.search(r"SR=([\d.]+)", out)
    sr = float(m.group(1)) if m else -1.0
    ok = abs(sr - 0.7578) < 0.08
    beacon(f"PLATFORM CHECK sr={sr:.4f} (local 0.7578) -> {'OK' if ok else 'MISMATCH'}")
    record(stage="platform_check", sr=sr, local_reference=0.7578, ok=ok)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-platform", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma list: platform,taskb,line,ring,dockcur,dockwall")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None

    def want(x: str) -> bool:
        return only is None or x in only

    beacon("PIPELINE START")
    if want("platform") and not args.skip_platform:
        platform_check()
    if want("taskb"):
        beacon("=== BLOCK: Task B staged curriculum ===")
        stage_chain()
    if want("line"):
        beacon("=== BLOCK: line x hazard (v6 reward) ===")
        crossing("Isaac-USV-PathHazard-Direct-v1", "line_hazard")
    if want("ring"):
        beacon("=== BLOCK: ring siege (v11 recipe) ===")
        crossing("Isaac-USV-HazardRing-Direct-v1", "ring_siege")
    if want("dockcur"):
        beacon("=== BLOCK: dock x current (v11-style retrain) ===")
        crossing("Isaac-USV-Dock-BlueBoat-Current-Direct-v1", "dock_current")
    if want("dockwall"):
        beacon("=== BLOCK: dock x solid walls (C3xC5) ===")
        crossing("Isaac-USV-DockWall-BlueBoat-Direct-v1", "dock_wall")
    beacon("PIPELINE DONE")


if __name__ == "__main__":
    main()
