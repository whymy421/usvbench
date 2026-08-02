"""Where does a policy actually spend its time, relative to the obstacles?

`arc_breakeven.py` prices an open-water tax but cannot pick its RADIUS: that
depends on how far from obstacles a detouring policy actually travels, which is
a measurement, not an assumption. `min_clearance` in the certification records
is the episode MINIMUM -- the single closest approach -- and says nothing about
where the boat spends the other 99% of its steps.

This walks a checkpoint through real episodes recording per-step clearance, and
reports, for a range of candidate radii, what fraction of pre-goal steps would
be taxed and what the tax would cost over an episode. Run it on a threading
policy and a detouring policy: the radius to pick is the one that separates
them, and a radius that taxes the threading policy too is the wrong radius.

    python scripts/measure_open_water.py --checkpoint <ckpt> --task <id>
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
parser.add_argument("--task", default="Isaac-USV-HazardNav-Direct-v3")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--level", type=int, default=0)
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

TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env_cfg.seed = args_cli.eval_seed
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
experiment_cfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False

env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
runner = Runner(wrapped, experiment_cfg)
runner.agent.load(os.path.abspath(args_cli.checkpoint))
runner.agent.set_running_mode("eval")
base = env.unwrapped

RADII = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]
counts = {r: 0 for r in RADII}
pre_goal_steps = 0
clearances: list[float] = []

obs, _ = wrapped.reset()
horizon = int(base.max_episode_length)
with torch.inference_mode():
    for _ in range(horizon):
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, _, _, _ = wrapped.step(actions)
        clr = base._clearance()
        goal_d = torch.norm(base.target_pos - base._com_xy(), dim=-1)
        # Same gate the reward uses: pre-goal and off final approach.
        active = (~base._reached_goal) & (goal_d > 6.0)
        pre_goal_steps += int(active.sum())
        for r in RADII:
            counts[r] += int(((clr > r) & active).sum())
        sample = clr[active]
        if sample.numel():
            clearances.extend(sample[:8].detach().cpu().tolist())

clearances.sort()
n = max(pre_goal_steps, 1)
step_s = float(base.control_step_s)
print(f"MEASURE task={TASK} level={args_cli.level} "
      f"ckpt={os.path.basename(args_cli.checkpoint)}")
print(f"  pre-goal steps sampled: {pre_goal_steps}")
if clearances:
    def q(p):
        return clearances[min(len(clearances) - 1, int(p * (len(clearances) - 1)))]
    print(f"  per-step clearance  p10 {q(.10):.2f}  median {q(.50):.2f}  "
          f"p90 {q(.90):.2f}  max {clearances[-1]:.2f} m")
print(f"  {'radius':>8} {'taxed steps':>12} {'cost @1.0/s':>12}")
for r in RADII:
    frac = counts[r] / n
    # cost per episode for a policy that spends this fraction out in the open
    cost = frac * horizon * step_s
    print(f"  {r:>8.1f} {frac:>11.1%} {cost:>12.1f}")
print("  (cost scales linearly with reward_open_water_scale)")

if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "checkpoint": os.path.abspath(args_cli.checkpoint),
                   "level": args_cli.level, "pre_goal_steps": pre_goal_steps,
                   "taxed_fraction": {str(r): counts[r] / n for r in RADII}}, f, indent=1)
sys.stdout.flush()
env.close()
app.close()
