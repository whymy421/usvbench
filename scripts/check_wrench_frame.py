# ruff: noqa: E402
"""Turn first, then thrust, and measure where the vessel actually goes.

A wrench assembled in world coordinates but handed to
set_external_force_and_torque() with the default is_global=False is invisible at
yaw = 0 and grows with heading, so the vessel has to be turned away from its
spawn attitude before the error shows up.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--fwd_x", type=float, default=1.0, help="+1 if body +X is the bow, -1 if it is the stern.")
parser.add_argument("--bow_axis", type=str, default="x", choices=["x", "y"], help="Which body axis the thrust acts on.")
parser.add_argument("--turn_s", type=float, default=6.0)
parser.add_argument("--thrust_s", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_rotate

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    with torch.inference_mode():
        base.reset()

    n = args_cli.num_envs
    device = base.device
    bow_vec = [args_cli.fwd_x, 0.0, 0.0] if args_cli.bow_axis == "x" else [0.0, args_cli.fwd_x, 0.0]
    bow_local = torch.tensor([bow_vec], device=device).repeat(n, 1)

    def heading_deg():
        q = base.robot.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.rad2deg(torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def run(action, seconds):
        acts = torch.tensor(action, device=device).repeat(n, 1)
        for _ in range(max(1, int(seconds / base.step_dt))):
            with torch.inference_mode():
                base.step(acts)

    def drift():
        vel = base.robot.data.root_lin_vel_w[:, :2]
        bow = quat_rotate(base.robot.data.root_quat_w, bow_local)[:, :2]
        speed = torch.norm(vel, dim=-1)
        cos = (vel * bow).sum(-1) / (speed * torch.norm(bow, dim=-1) + 1e-9)
        d = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
        m = speed > 0.05
        return (d[m].mean().item() if m.any() else float("nan"), speed.mean().item())

    print("\n" + "=" * 70)
    sign = "+" if args_cli.fwd_x > 0 else "-"
    print(f"  Wrench-frame audit: {args_cli.task}   (bow = body {sign}{args_cli.bow_axis.upper()})")
    print("=" * 70)

    print(f"  spawn heading  : {heading_deg().mean().item():7.1f} deg")
    run([0.0, 0.0], 1.0)
    run([1.0, 0.0], args_cli.thrust_s)
    d0, s0 = drift()
    print(f"  A) thrust from the spawn attitude    : drift {d0:6.1f} deg   speed {s0:5.2f} m/s")

    with torch.inference_mode():
        base.reset()
    run([0.0, 1.0], args_cli.turn_s)
    # let the yaw rate decay first: a vessel still rotating while it accelerates has its
    # velocity lag its bow for reasons that have nothing to do with the wrench frame
    for _ in range(60):
        run([0.0, 0.0], 0.5)
        if base.robot.data.root_ang_vel_w[:, 2].abs().max().item() < 0.005:
            break
    yaw_rate = base.robot.data.root_ang_vel_w[:, 2].abs().max().item()
    turned = heading_deg().mean().item()
    run([1.0, 0.0], args_cli.thrust_s)
    d1, s1 = drift()
    print(f"  B) turn to {turned:7.1f} deg, then thrust : drift {d1:6.1f} deg   speed {s1:5.2f} m/s")
    print(f"     (residual yaw rate before thrusting: {yaw_rate:.4f} rad/s)")

    print("-" * 70)
    if abs(d1 - d0) < 5.0:
        print("  Frame OK: the drift does not depend on heading.")
    else:
        print(f"  FRAME BUG: drift changes by {abs(d1 - d0):.1f} deg once the vessel is turned.")
        print("  A world-frame wrench is being applied as a body-frame one (is_global).")
    print("=" * 70 + "\n")
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    simulation_app.close()
