"""Off-policy training entry point for the baseline matrix.

Isaac Lab's stock train.py only accepts AMP/PPO/IPPO/MAPPO, so SAC (and any
other skrl agent) needs its own launcher. This drives skrl's Runner directly
with the same env boot sequence the certified evaluators use, so a checkpoint
produced here loads in eval_v6_frozen.py without special handling.

The config comes from the task registry under ``skrl_sac_cfg_entry_point``,
keeping algorithm hyperparameters versioned next to the task like PPO's.

Run: python scripts/sac_train.py --task Isaac-USV-HazardCross-Direct-v1 \
         --seed 42 --timesteps 48000 --experiment-name sac_cross_s42 --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=None,
                    help="override the trainer timesteps in the yaml")
parser.add_argument("--checkpoint-interval", type=int, default=None)
parser.add_argument("--experiment-name", default=None)
parser.add_argument("--warm-start", default=None,
                    help="checkpoint to load into the agent before training "
                         "(policy+value+optimizer; same-family agents only)")
parser.add_argument("--cfg-entry-point", default="skrl_sac_cfg_entry_point")
# Off-policy tuning without minting a yaml per variant. SAC/TD3 have no
# value preprocessor (PPO does), so raw rewards of +-50 with gamma=0.999
# drive Q targets toward 1e5 and the critic diverges; rewards_shaper_scale
# and discount_factor are the two knobs that fix it.
parser.add_argument("--agent-set", dest="agent_sets", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override an agent config key, repeatable, e.g. "
                         "--agent-set discount_factor=0.99")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import os
from datetime import datetime

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# --------------------------------------------------------------- v4 policy hook
# skrl's Runner._component (skrl/utils/runner/torch/runner.py, read from the
# installed 1.4.3) is a hard-coded if/elif chain over skrl's OWN class names
# that ends in `raise ValueError(f"Unknown component '{name}' in runner cfg")`.
# It is a closed whitelist: a yaml `class:` entry can therefore never select a
# model defined outside skrl, no matter how it is spelled. Selecting this
# repo's tanh-squashed Gaussian policy (skrl_sac_v4_cfg.yaml) requires adding
# exactly one branch, which means subclassing.
#
# Every other name falls straight through to super(), so a stock PPO/SAC/TD3
# config resolves through the untouched skrl code path and an unknown class
# still raises the same ValueError. "squashedgaussianmixin" is a name stock
# skrl would have rejected anyway, so no existing config can change meaning.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    # append, not insert(0): nothing already importable changes resolution.
    sys.path.append(_REPO_ROOT)
try:
    from tasks._shared.squashed_gaussian import squashed_gaussian_model
except ImportError:
    # Deployed layout -- the queue scripts' SyncPackage copies tasks/<pkg> into
    # IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/<pkg>, so on a box
    # the same module also answers to this name.
    from isaaclab_tasks.direct._shared.squashed_gaussian import squashed_gaussian_model


class SquashedRunner(Runner):
    """Runner whose component lookup also knows ``SquashedGaussianMixin``."""

    def _component(self, name: str):
        if name.lower() == "squashedgaussianmixin":
            return squashed_gaussian_model
        return super()._component(name)


TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed

experiment_cfg = load_cfg_from_registry(TASK, args_cli.cfg_entry_point)
experiment_cfg["seed"] = args_cli.seed
if args_cli.timesteps is not None:
    experiment_cfg["trainer"]["timesteps"] = args_cli.timesteps
# Match Isaac Lab train.py's layout exactly: runs land in
# <cwd>/logs/skrl/usvbench/<timestamp>_<algo>_torch_<name>. The screening,
# certification and RunDir helpers all glob that shape, so an off-policy run
# is indistinguishable from a PPO run to every downstream tool.
experiment_cfg["agent"]["experiment"]["directory"] = os.path.join(
    os.getcwd(), "logs", "skrl", "usvbench"
)
# The house PPO yaml carries wandb settings but the boxes have no wandb
# package; leaving it on kills the Runner at construction time -- and
# Isaac's teardown hook then LIES with exit code 0 (one silent night
# lost to exactly that). Eval scripts force this off; so does training.
experiment_cfg["agent"]["experiment"]["wandb"] = False
if args_cli.experiment_name:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo = str(experiment_cfg["agent"].get("class", "sac")).lower()
    experiment_cfg["agent"]["experiment"]["experiment_name"] = (
        f"{stamp}_{algo}_torch_{args_cli.experiment_name}"
    )
if args_cli.checkpoint_interval is not None:
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = (
        args_cli.checkpoint_interval
    )
for assignment in args_cli.agent_sets:
    key, _, raw = assignment.partition("=")
    if not _:
        raise SystemExit(f"--agent-set needs KEY=VALUE, got {assignment!r}")
    try:
        value = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
    except ValueError:
        value = raw
    experiment_cfg["agent"][key] = value
    print(f"agent override {key}={value}", flush=True)

# The replay buffer is per-environment: skrl allocates memory_size transitions
# for EACH parallel env, so the yaml value is a per-env depth, not a total.
memory_size = experiment_cfg["memory"].get("memory_size", 0)
print(f"SAC train task={TASK} seed={args_cli.seed} envs={args_cli.num_envs} "
      f"timesteps={experiment_cfg['trainer']['timesteps']} "
      f"replay/env={memory_size}", flush=True)

env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

torch.manual_seed(args_cli.seed)
runner = SquashedRunner(wrapped, experiment_cfg)
if args_cli.warm_start:
    # Restore the full agent state (networks + optimizer + preprocessors)
    # so training continues rather than restarts. Used for the fortress
    # warm+tax stabilizer experiment: resume the 77%-peak policy inside
    # the taxed reward field and see whether the collapse still happens.
    runner.agent.load(os.path.abspath(args_cli.warm_start))
    print(f"warm-started from {args_cli.warm_start}", flush=True)
runner.run("train")

env.close()
app.close()
