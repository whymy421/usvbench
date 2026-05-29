# Boat Calm-Water Navigation — Starter Task

> **For**: Arif (and anyone learning Isaac Lab + RL for USV)
> **Author**: Yutong
> **Status**: physics verified, RL training not yet validated by Yutong
> **Date**: 2026-05-28

## What this task is

A **single boat** navigating to a **single point target** in **calm water** (no waves, no current). It's the simplest possible USV RL task — designed for you to get familiar with the Isaac Lab + skrl loop without any complexity from waves or multi-agent.

**Algorithm**: PPO (single-agent)
**Observation**: 3D — `(dot, cross, dist_norm)` where dot/cross measures alignment between heading and target direction, dist_norm is normalized distance.
**Action**: 2D — `(forward_thrust, yaw_torque)` both in [-1, 1]
**Reward**: nav + reach (no wave terms)
**Episode**: 120 seconds, ends when target reached or timeout

## Files in this folder

| File | Purpose |
|------|---------|
| `my_first_task_env.py` | The env (physics + observations + reward) |
| `my_first_task_env_cfg.py` | All hyperparameters, USD path, physics constants |
| `__init__.py` | Registers the Gym task name `Isaac-Boat-Calm-Direct-v0` |
| `agents/skrl_ppo_cfg.yaml` | skrl's PPO config (network, learning rate, etc) |
| `learned_reward.py` | (unused, vestigial from E4) |

## Setup

1. Put boat USD at `C:\Users\Yutong\NavRL\NavRL2026\isaac_underwater\boat_physics.usdc`
   (or change the `usd_path` in `my_first_task_env_cfg.py` line 19 to your path)
2. Make sure your `isaaclab` conda env is activated.

## Run commands

### Quick visualization (5 min, see the boat float and a random policy thrash around)

```powershell
$env:PYTHONIOENCODING="utf-8"
python "C:\Users\Yutong\NavRL\IsaacLab\scripts\reinforcement_learning\skrl\train.py" `
  --task=Isaac-Boat-Calm-Direct-v0 --num_envs=4 --max_iterations=200
```

### "Pure float" demo (no RL action, see physics only)

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:NO_ACTION="1"
python "C:\Users\Yutong\NavRL\IsaacLab\scripts\reinforcement_learning\skrl\train.py" `
  --task=Isaac-Boat-Calm-Direct-v0 --num_envs=4 --max_iterations=200
```

The boat sits still at z=0 (half submerged), no oscillation. This proves the physics is stable.

### Drop-from-height demo (see buoyancy work)

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:NO_ACTION="1"
$env:INIT_Z="1.0"
python "C:\Users\Yutong\NavRL\IsaacLab\scripts\reinforcement_learning\skrl\train.py" `
  --task=Isaac-Boat-Calm-Direct-v0 --num_envs=4 --max_iterations=200
```

Boat drops from 1m, splashes in, settles to z=0.

### Real training with auto video upload to wandb (3-4 hours, RECOMMENDED)

This uses `train_with_eval.py` which records short videos every 50k timesteps and uploads them to wandb. You'll see training videos in your wandb run page (very useful for debugging + showing reviewers later).

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:WANDB_NAME="boat_calm_nav_s42"
python "C:\Users\Yutong\NavRL\IsaacLab\scripts\reinforcement_learning\skrl\train_with_eval.py" `
  --task=Isaac-Boat-Calm-Direct-v0 --num_envs=64 --headless --max_iterations=3000 --seed=42 `
  --video --video_interval 50000 --video_length 200
```

Video parameters:
- `--video_length 200` — each video is 200 simulation steps (~3 seconds)
- `--video_interval 50000` — record a video every 50000 timesteps (~3 videos total per 3000 iter run)
- Add `--enable_cameras` if you get "no camera" errors (the script tries to add it automatically)

### Real training without video (faster, fallback option)

If your GPU can't handle the camera/render overhead, use the basic trainer:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:WANDB_NAME="boat_calm_nav_s42"
python "C:\Users\Yutong\NavRL\IsaacLab\scripts\reinforcement_learning\skrl\train.py" `
  --task=Isaac-Boat-Calm-Direct-v0 --num_envs=64 --headless --max_iterations=3000 --seed=42
```

## What "success" looks like

After 3000 iter (around 3-4 hours on RTX 5080) you should see in wandb:

| Metric | Expected (rough) | What it means |
|--------|------------------|---------------|
| `Reward / Instantaneous reward (mean)` | > 5 and still rising | policy is learning |
| `Train/distance` | < 5 m | boat consistently gets near targets |
| `Train/speed` | > 0.3 m/s | boat is moving, not stuck |
| `Train/heading_error` | < 30° | boat points at target |

If after 3000 iter `distance` is still 20+ m or `speed` is 0, something's wrong — message Yutong.

## Environment variable toggles

| env var | default | purpose |
|---------|---------|---------|
| `NO_ACTION` | unset | set to `1` to disable RL actions (pure physics demo) |
| `INIT_Z` | `0.0` | initial z of boat at episode reset. Set `1.0` for drop-from-height demo |
| `DEBUG_Z` | unset | set to `1` to print z position every 30 steps (debugging) |

## Known facts about the physics (don't be surprised)

1. **Mass = 100 kg, displacement volume = 0.2 m³**. This is **not physically realistic** for a 5.5m trihull (real one would be ~1000 kg, 2.0 m³ displacement). It's a numerical hack so the boat balances at 50% submerged with 100N thrust. We'll do a real-physics refactor later.
2. **No articulation**. The boat USD is a single rigid body — thrust and torques are applied as external forces to the hull. No moving thruster joints.
3. **No collision between boats** (only one boat in this task).
4. **Water surface is flat** at z=0. No waves are computed (`enable_wave=False` in cfg).
5. **Damping is PhysX-native** (`linear_damping=15` in `rigid_props`). This is much more stable than Python force-based damping we used previously.

## When you get stuck

- Boat sinks completely → `rov_volume` too small for this mass, see cfg line 53
- Boat flies up out of water → `rov_volume` too big, same line
- Boat oscillates up and down → check that `enable_wave=False` (water mesh animation was the culprit before)
- Boat drifts sideways without action → check `enable_current=False` in cfg
- Training reward not improving → first try `num_envs=64 --headless` (we tested at 4 for viz, too few for actual learning)

## Next steps after you get this running

1. Repeat at different seeds (123, 456) — reproducibility check
2. Bump up to 5000 iter, see if metrics improve further
3. Talk to Yutong about adding waves and going multi-boat (E6 territory)
