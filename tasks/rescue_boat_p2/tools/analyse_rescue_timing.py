"""
analyse_rescue_timing.py — When, within an episode, does the policy rescue and
when does it lose?

WHY THIS EXISTS
rescue_rate says how many casualties were recovered. It says nothing about the
thing the task was built to test: whether the policy triages. A policy that
recovers 0.77 by servicing whichever casualties happen to be nearest scores the
same as one that reads the deadlines and goes for the ones about to expire.

Three questions this answers, none of which the current figures address:

  1. WHEN do rescues happen relative to the 120 s episode? A triaging policy
     should front-load: reach the urgent ones early, then mop up. A greedy one
     spreads rescues evenly and loses casualties in a late cluster.

  2. WHEN do casualties expire? Expiries bunched at the end mean the vessel ran
     out of time. Expiries spread throughout mean it was going to the wrong ones.

  3. Does outcome depend on the INITIAL deadline? This is the direct test of
     prioritisation. Casualties drawn a short deadline (40 s) should be either
     rescued early or lost early; if survival is flat across the 40-90 s draw,
     the policy is ignoring urgency entirely and the priority observation is
     doing no work.

WHAT IT RECORDS
Per casualty, per environment: the simulated time at which it was rescued or
expired, its initial deadline, and its spawn distance. Written to
rescue_timing.csv, one row per casualty.

The environment does not currently log any of this; the script reconstructs it
by watching the casualty_alive flag flip and classifying the transition by
whether the vessel was inside the capture radius at that step.

USAGE
  python analyse_rescue_timing.py --checkpoint="<path>/best_agent.pt" \
      --num_envs=64 --headless

  Then:  python make_timing_figure.py     (no GPU needed, reads the CSV)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record rescue and expiry times.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--seed", type=int, default=2026,
                    help="Evaluation seed. Use one NOT used for checkpoint "
                         "selection if you want a held-out result.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--out", type=str, default="rescue_timing.csv")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = "skrl_cfg_entry_point"


def _base_env(env):
    e = env
    for _ in range(10):
        if hasattr(e, "casualty_alive"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _get_obs_prep(agent):
    for a in ("_obs_preprocessor", "_observation_preprocessor",
              "observation_preprocessor"):
        c = getattr(agent, a, None)
        if c is not None and callable(c):
            return c
    return None


def _restore_prep(agent, ckpt):
    """The normaliser must be restored or the policy is scored under the wrong
    statistics. This was defect 1; see the thesis Section 5.2."""
    prep = _get_obs_prep(agent)
    if prep is None or not hasattr(prep, "load_state_dict"):
        return "none"
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"unreadable: {e}"
    for k in ("observation_preprocessor", "_observation_preprocessor",
              "state_preprocessor", "_state_preprocessor"):
        if k in raw:
            prep.load_state_dict(raw[k])
            return f"restored '{k}'"
    return "no key"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    ckpt = os.path.abspath(args_cli.checkpoint)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env)
    base = _base_env(env)

    runner = Runner(env, agent_cfg)
    runner.agent.load(ckpt)
    runner.agent.set_running_mode("eval")
    status = _restore_prep(runner.agent, ckpt)
    print(f"[timing] normaliser: {status}")
    if status in ("none", "no key") or status.startswith("unreadable"):
        raise SystemExit("[timing] refusing to score: normaliser not restored. "
                         "See thesis Section 5.2 for why this matters.")

    dt = base.cfg.sim.dt * base.cfg.decimation
    cap_r = base.cfg.rescue_radius
    max_ep = int(base.max_episode_length)
    N, C = base.num_envs, base.cfg.n_casualties

    rows = []
    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        prev_alive = base.casualty_alive.clone()
        init_timer = base.casualty_init_timer.clone()
        spawn_d = torch.linalg.norm(
            base.casualty_pos[:, :, :2] - base.robot.data.root_pos_w[:, None, :2],
            dim=-1).clone()
        resolved = torch.zeros_like(prev_alive)

        for step in range(max_ep):
            with torch.inference_mode():
                actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                obs, _, _, _, _ = env.step(actions)

            alive = base.casualty_alive
            flipped = prev_alive & (~alive) & (~resolved)
            if flipped.any():
                boat_xy = base.robot.data.root_pos_w[:, None, :2]
                dist = torch.linalg.norm(base.casualty_pos[:, :, :2] - boat_xy, dim=-1)
                # A casualty leaves the alive set either because the vessel
                # closed inside the capture radius, or because its timer ran out.
                was_rescue = dist <= cap_r * 1.05
                idx = flipped.nonzero(as_tuple=False)
                for e, c in idx.tolist():
                    rows.append({
                        "episode": ep,
                        "env": e,
                        "casualty": c,
                        "outcome": "rescued" if bool(was_rescue[e, c]) else "expired",
                        "time_s": round(step * dt, 3),
                        "episode_frac": round(step / max_ep, 4),
                        "initial_deadline_s": round(float(init_timer[e, c]), 2),
                        "spawn_distance_m": round(float(spawn_d[e, c]), 1),
                    })
                resolved |= flipped
            prev_alive = alive.clone()

        n_r = sum(1 for r in rows if r["episode"] == ep and r["outcome"] == "rescued")
        print(f"[timing] episode {ep}: {n_r}/{N*C} rescued "
              f"({n_r/(N*C):.4f})")

    with open(args_cli.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    resc = sum(1 for r in rows if r["outcome"] == "rescued")
    print(f"\n[timing] wrote {total} casualty outcomes to {args_cli.out}")
    print(f"[timing] overall rescue rate {resc}/{args_cli.episodes*N*C} = "
          f"{resc/(args_cli.episodes*N*C):.4f}")
    print("[timing] now run: python make_timing_figure.py")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
