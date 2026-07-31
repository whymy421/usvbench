# Copyright (c) 2022-2025, USVBench Contributors.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E402  # Isaac Sim must launch before importing simulation modules
"""Open-loop physics check for the catamaran task — no policy involved.

Holds full forward thrust for a few seconds and measures where the vessel actually
goes relative to its bow, then holds pure yaw torque and measures the turn rate.
A correct thruster model gives a drift angle near 0 deg; a wrench applied in the
wrong frame shows up here immediately and independently of any training run.

Example:
    python scripts/check_catamaran_physics.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open-loop physics check for the catamaran task.")
parser.add_argument("--task", type=str, default="Isaac-Catamaran-Patrol-Direct-v1")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=6.0, help="Duration of each open-loop phase.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

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
    base.reset()

    steps = max(1, int(args_cli.seconds / base.step_dt))
    device = base.device
    bow_local = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(args_cli.num_envs, 1)
    up_local  = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(args_cli.num_envs, 1)

    def run(action, label, track_attitude=False):
        actions = torch.tensor(action, device=device).repeat(args_cli.num_envs, 1)
        worst_tilt = 0.0
        for _ in range(steps):
            with torch.inference_mode():
                base.step(actions)
                if track_attitude:
                    # angle between the hull's up axis and world up, i.e. total tilt
                    up_w = quat_rotate(base.robot.data.root_quat_w, up_local)
                    tilt = torch.rad2deg(torch.acos(up_w[:, 2].clamp(-1.0, 1.0)))
                    worst_tilt = max(worst_tilt, tilt.max().item())
        if track_attitude:
            print(f"\n  {label}")
            print(f"    worst tilt off vertical : {worst_tilt:7.2f} deg")
            print(f"    final yaw rate          : {base.robot.data.root_ang_vel_w[:, 2].mean().item():7.3f} rad/s")
            return worst_tilt
        vel_w = base.robot.data.root_lin_vel_w
        ang_w = base.robot.data.root_ang_vel_w
        bow_w = quat_rotate(base.robot.data.root_quat_w, bow_local)

        speed = torch.norm(vel_w[:, :2], dim=-1)
        moving = speed > 0.05
        cos = (vel_w[:, :2] * bow_w[:, :2]).sum(-1) / (speed * torch.norm(bow_w[:, :2], dim=-1) + 1e-9)
        drift = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
        drift_report = drift[moving].mean().item() if moving.any() else float("nan")

        print(f"\n  {label}")
        print(f"    mean speed              : {speed.mean().item():7.3f} m/s")
        print(f"    drift angle vs bow      : {drift_report:7.1f} deg   (0 = thrust pushes the bow forward)")
        print(f"    mean yaw rate           : {ang_w[:, 2].mean().item():7.3f} rad/s")
        print(f"    mean |roll+pitch| rate  : {ang_w[:, :2].abs().mean().item():7.3f} rad/s")
        return drift_report

    print("\n" + "=" * 68)
    print(f"  Open-loop physics check: {args_cli.task}")
    print(f"  {args_cli.num_envs} envs, {args_cli.seconds:.1f} s per phase, dt = {base.step_dt:.4f} s")
    print("=" * 68)

    drift_fwd = run([1.0, 0.0], "phase 1 — full forward thrust, zero torque")
    run([0.0, 1.0], "phase 2 — zero thrust, full yaw torque")
    # Phase 3 exists because the two worst bugs this task has had — the double-rotated
    # wrench and the Euler-angle attitude spring — were both exactly correct at yaw = 0
    # and only wrong once the hull turned. Every episode starts at yaw = 0 and any
    # zero-action test sits there, so neither showed up until something drove a turn.
    # Thrust and yaw torque together sweep the hull through every heading.
    tilt = run([1.0, 1.0], "phase 3 — full thrust AND full yaw torque (sweeps all headings)",
               track_attitude=True)

    print("\n" + "-" * 68)
    ok = True
    if math.isfinite(drift_fwd) and drift_fwd < 15.0:
        print("  PASS: thrust drives the vessel along its bow axis.")
    else:
        ok = False
        print("  FAIL: the vessel does not move along its bow — check the wrench frame")
        print("        passed to set_external_force_and_torque (is_global).")
    if tilt < 5.0:
        print("  PASS: attitude stays upright through a full sweep of headings.")
    else:
        ok = False
        print("  FAIL: the hull tilts while turning — the attitude spring is not")
        print("        yaw-invariant. Use k*(hull_up x world_up), never body roll/pitch")
        print("        angles applied about fixed world axes.")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    print("-" * 68 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
