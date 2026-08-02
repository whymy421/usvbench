"""Is the wave failure dose-dependent, and in which variable?

Yutong's sweep scanned `slope_torque_scale` and found a switch, not a dose
response (0.00 -> 21/64, 0.05..1.00 -> 1/64 identically). Her own config
explains why: in plant v2, with distributed buoyancy active, that field is
documented as "Legacy channel ... Unused". The scan was moving a knob that
does nothing in the active plant, so of course it had no dose response.

Under plant v2 the wave reaches the hull through buoyancy sampled at six
stations, so the dose knobs are the sea state itself:

  height  -- how far the surface moves. A real physical sensitivity should
             degrade smoothly with it.
  period  -- via deep-water dispersion, lambda = g T^2 / 2pi. At T = 3.0 s
             lambda is 14.1 m against a 1.2 m hull, so all six stations sit in
             nearly the same phase and the boat rides a quasi-static tilt
             rather than being excited. T ~ 1.0 s puts lambda at 1.5 m, which
             is comparable to the hull and to the station spacing.

If height gives a smooth degradation, the wave numbers are a real robustness
result. If height is also a switch, the policy has a failure attractor that
any persistent disturbance triggers, and the wave magnitude is not what is
being measured.

    python scripts/wave_dose_sweep.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-USV-HazardNav-Airy-Direct-v3")
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--level", type=int, default=1)
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

# (height_m, period_s). Row 1 is calm-equivalent; the height column holds the
# period at her value, the period column holds the height at hers.
GRID = [
    (0.00, 3.0), (0.02, 3.0), (0.04, 3.0), (0.08, 3.0), (0.12, 3.0),
    (0.12, 2.0), (0.12, 1.5), (0.12, 1.0),
]

experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False

results = []
print(f"WAVE DOSE SWEEP  task={args_cli.task} "
      f"ckpt={os.path.basename(args_cli.checkpoint)}")
print(f"  {'H (m)':>7} {'T (s)':>6} {'lambda/L':>9} {'SR':>8} {'path med':>9}")

for height, period in GRID:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0",
                            num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.eval_seed
    if hasattr(env_cfg, "curriculum_frozen"):
        env_cfg.curriculum_frozen = True
        env_cfg.eval_level = args_cli.level
    env_cfg.wave.airy_height_m = height
    env_cfg.wave.airy_period_s = period
    if height <= 0.0:
        env_cfg.wave.mode = "calm"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, experiment_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.set_running_mode("eval")
    base = env.unwrapped

    # Stop as soon as `episodes` have COMPLETED rather than running a full
    # horizon. Wave physics samples buoyancy at six stations every step and is
    # roughly an order of magnitude slower than calm: a full 7200-step horizon
    # did not finish inside a 30 minute watchdog, which is also why the first
    # 64x64 attempt looked like a four-hour hang when it was merely slow.
    obs, _ = wrapped.reset()
    done_flags = []
    paths_done = []
    steps = 0
    max_steps = int(base.max_episode_length)
    with torch.inference_mode():
        while len(done_flags) < args_cli.episodes and steps < max_steps:
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            obs, _, term, trunc, _ = wrapped.step(
                outputs[-1].get("mean_actions", outputs[0])
            )
            steps += 1
            done = term | trunc
            done = done.squeeze(-1) if done.dim() > 1 else done
            for i in torch.nonzero(done).flatten().tolist():
                done_flags.append(bool(base.episode_success[i]))
                paths_done.append(float(base.episode_path_length[i]))
    n_done = max(len(done_flags), 1)
    sr = sum(done_flags) / n_done
    ok = sorted(p for p, s in zip(paths_done, done_flags) if s)
    med = ok[len(ok) // 2] if ok else float("nan")
    lam = 9.81 * period * period / (2 * 3.14159265) / 1.2
    print(f"  {height:>7.2f} {period:>6.1f} {lam:>9.1f} {sr:>8.3f} {med:>9.1f}")
    results.append({"height_m": height, "period_s": period,
                    "lambda_over_L": lam, "sr": sr, "path_median_m": med})
    env.close()

print("\n  If SR falls smoothly with height -> real physical sensitivity.")
print("  If SR is 0 at every non-zero height -> a failure attractor that any")
print("  persistent disturbance triggers, and the wave size is not the story.")
if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"task": args_cli.task, "grid": results}, f, indent=1)
sys.stdout.flush()
app.close()
