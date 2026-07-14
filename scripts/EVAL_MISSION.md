# Mission evaluation

## Mission eval protocol

Mission tasks end each episode in success or failure. Evaluate a deterministic
policy with `eval_mission.py` for a fixed number of **completed** episodes.

- **SR (success rate)** is successful completed episodes divided by all counted
  completed episodes.
- **SPL (success weighted by path length)** averages
  `success * (d0 - R) / max(path_length, d0 - R)`, where `d0` is the initial
  horizontal distance and `R` is `hold_radius` (or `goal_radius`). Failed
  episodes contribute zero.
- Once the requested completion count is reached, every episode still in flight
  is discarded. If a vector step produces surplus simultaneous completions,
  only the lowest env indices needed to reach the exact count are included.

Each run prints a reproducibility header and, when `--csv` is set, appends the
same header as a `#` comment row. It records the git commit, task ID, checkpoint
filename (or `zero`), requested episode count, seed, and a short SHA1 of sorted
protocol-relevant config scalars: episode length, hold/goal radius, and available
spawn-distance fields.

Checkpoint evaluation:

```powershell
python scripts/eval_mission.py --task Isaac-USV-StationKeep-Direct-v1 `
  --checkpoint path/to/best_agent.pt --num_envs 64 --episodes 128 --seed 42 `
  --headless --csv mission_results.csv
```

No-checkpoint plumbing dry run:

```powershell
python scripts/eval_mission.py --task Isaac-USV-StationKeep-Direct-v1 `
  --policy zero --num_envs 2 --episodes 2 --seed 42 --headless
```
