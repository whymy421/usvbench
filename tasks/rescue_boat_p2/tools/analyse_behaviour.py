"""
analyse_behaviour.py — Quantify HOW the policy moves, not just how well it scores.

Prompted by video showing the vessel rotating continuously while transiting to
casualties. rescue_rate cannot detect this: the reward is defined on distance to
the target and says nothing about heading, so a policy that spins while closing
distance scores the same as one that drives straight.

Metrics:

  path efficiency   net displacement / distance travelled
                    1.0 = straight line, 0.3 = travelling three times as far as
                    it needs to. This is the single number that exposes spinning.

  mean |yaw rate|   sustained rotation. Efficient pursuit turns to acquire a
                    heading and then holds it, so this should be low with
                    occasional spikes, not a high steady value.

  heading error     angle between the hull's forward axis and its velocity
                    vector. A vessel driving properly has this near zero; a
                    spinning one sweeps through all values.

  revolutions       cumulative heading change / 360, per episode.

Compares the trained policy against the scripted pure-pursuit oracle, which by
construction drives straight, so it provides the reference values.

Usage:
  python analyse_behaviour.py --checkpoint="<path>.pt" --num_envs=16 --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Analyse policy locomotion behaviour.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=7200, help="One full episode.")
parser.add_argument("--compare_oracle", action="store_true", default=True)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
        if hasattr(e, "reached_count"):
            return e
        if hasattr(e, "unwrapped") and e.unwrapped is not e:
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _get_obs_prep(agent):
    for a in ("_obs_preprocessor", "_observation_preprocessor", "observation_preprocessor"):
        c = getattr(agent, a, None)
        if c is not None and callable(c):
            return c
    return None


def _restore_prep(agent, ckpt):
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


def yaw_of(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(env, base, act_fn, steps, label):
    obs, _ = env.reset()
    n = base.num_envs
    # Stay inside one episode. At exactly max_episode_length the environment has
    # auto-reset and the vessel is back at spawn, which makes net displacement
    # zero and path efficiency meaningless.
    max_ep = int(getattr(base, "max_episode_length", steps))
    steps = min(int(steps), max_ep - 120)

    start = base.robot.data.root_pos_w[:, :2].clone()
    prev_pos = start.clone()
    prev_yaw = yaw_of(base.robot.data.root_quat_w).clone()

    path_len = torch.zeros(n, device=base.device)
    yaw_abs = torch.zeros(n, device=base.device)
    cum_turn = torch.zeros(n, device=base.device)
    head_err_sum = torch.zeros(n, device=base.device)
    speed_sum = torch.zeros(n, device=base.device)
    peak_net = torch.zeros(n, device=base.device)
    samples = 0

    for i in range(steps):
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(act_fn(obs, base))

        pos = base.robot.data.root_pos_w[:, :2]
        path_len += (pos - prev_pos).norm(dim=-1)
        prev_pos = pos.clone()

        y = yaw_of(base.robot.data.root_quat_w)
        d = y - prev_yaw
        d = torch.atan2(torch.sin(d), torch.cos(d))
        cum_turn += d.abs()
        prev_yaw = y.clone()

        vel = base.robot.data.root_lin_vel_w[:, :2]
        spd = vel.norm(dim=-1)
        yaw_abs += base.robot.data.root_ang_vel_w[:, 2].abs()
        speed_sum += spd

        # Angle between hull forward axis and velocity direction.
        moving = spd > 0.5
        vel_dir = torch.atan2(vel[:, 1], vel[:, 0])
        he = torch.atan2(torch.sin(vel_dir - y), torch.cos(vel_dir - y)).abs()
        head_err_sum += torch.where(moving, he, torch.zeros_like(he))
        cur_net = (pos - start).norm(dim=-1)
        peak_net = torch.maximum(peak_net, cur_net)
        samples += 1

    net = (base.robot.data.root_pos_w[:, :2] - start).norm(dim=-1)
    # Peak excursion is the fairer numerator: a vessel that services several
    # targets legitimately returns toward its start, which would understate
    # efficiency if only the final position were used.
    eff = (peak_net / path_len.clamp(min=1e-6)).mean().item()

    return dict(
        label=label,
        path_eff=eff,
        path_len=path_len.mean().item(),
        net=net.mean().item(),
        peak=peak_net.mean().item(),
        yaw_rate=(yaw_abs / samples).mean().item(),
        revolutions=(cum_turn / (2 * math.pi)).mean().item(),
        head_err=math.degrees((head_err_sum / samples).mean().item()),
        speed=(speed_sum / samples).mean().item(),
    )


def report(r):
    print(f"\n  {r['label']}")
    print(f"    mean speed            {r['speed']:.2f} m/s")
    print(f"    distance travelled    {r['path_len']:.0f} m")
    print(f"    final displacement    {r['net']:.0f} m")
    print(f"    peak excursion        {r['peak']:.0f} m")
    print(f"    PATH EFFICIENCY       {r['path_eff']:.3f}   (1.0 = straight line)")
    print(f"    mean |yaw rate|       {r['yaw_rate']:.3f} rad/s")
    print(f"    revolutions/episode   {r['revolutions']:.1f}")
    print(f"    heading vs velocity   {r['head_err']:.0f} deg   (0 = driving forward)")


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
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = _base_env(env)

    runner = Runner(env, agent_cfg)
    agent = runner.agent
    agent.load(ckpt)
    print(f"\n  preprocessor: {_restore_prep(agent, ckpt)}")
    for m in agent.models.values():
        if hasattr(m, "eval"):
            m.eval()
    prep = _get_obs_prep(agent)

    def policy_act(obs, base):
        o = obs["policy"] if isinstance(obs, dict) else obs
        oi = prep(o, train=False) if prep is not None else o
        a, inf = agent.policy.act({"observations": oi}, role="policy")
        return inf.get("mean_actions", a)

    def oracle_act(obs, base):
        o = obs["policy"] if isinstance(obs, dict) else obs
        err = torch.atan2(o[:, 1], o[:, 0])
        return torch.stack([torch.clamp(torch.cos(err), 0.0, 1.0),
                            torch.clamp(2.0 * err, -1.0, 1.0)], dim=-1)

    print()
    print("=" * 70)
    print("  LOCOMOTION BEHAVIOUR ANALYSIS")
    print("=" * 70)
    print("  rescue_rate measures WHETHER targets are reached, not HOW.")
    print("  The reward is defined on distance alone, so nothing penalises")
    print("  rotating while translating.")
    print("=" * 70)

    rp = run(env, base, policy_act, args_cli.steps, "TRAINED POLICY")
    report(rp)

    ro = None
    if args_cli.compare_oracle:
        ro = run(env, base, oracle_act, args_cli.steps, "SCRIPTED PURE PURSUIT (reference)")
        report(ro)

    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if rp["path_eff"] < 0.5:
        print(f"  Path efficiency {rp['path_eff']:.3f}: the vessel travels roughly "
              f"{1/max(rp['path_eff'],1e-6):.1f}x further than necessary.")
    if rp["revolutions"] > 3:
        print(f"  {rp['revolutions']:.0f} revolutions per episode — the hull is "
              f"spinning continuously, not steering.")
    if rp["head_err"] > 45:
        print(f"  Hull points {rp['head_err']:.0f} deg away from its direction of "
              f"travel on average; it is not driving forward.")
    if ro:
        print(f"\n  Policy path efficiency {rp['path_eff']:.3f} vs pure pursuit "
              f"{ro['path_eff']:.3f}")
        if rp["path_eff"] < ro["path_eff"] * 0.7:
            print("  The policy reaches targets despite its locomotion, not because")
            print("  of it. The reward under-specifies the behaviour: closing")
            print("  distance is rewarded, holding a heading is not.")
    print("=" * 70)
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
