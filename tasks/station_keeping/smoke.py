"""Headless runtime smoke test for the station-keeping task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-USV-StationKeep-Direct-v1"
NUM_ENVS = 16

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Register this task whether the script is run from USVBench or after the folder
# is copied into isaaclab_tasks/direct.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import station_keeping  # noqa: F401, E402

from station_keeping.station_keeping_env_cfg import StationKeepingEnvCfg  # noqa: E402


def main() -> bool:
    env = None
    try:
        env_cfg = StationKeepingEnvCfg()
        env_cfg.scene.num_envs = NUM_ENVS
        env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
        base = env.unwrapped

        obs, _ = env.reset()
        obs_shape = tuple(obs["policy"].shape)
        reward_min = float("inf")
        reward_max = float("-inf")
        finite = torch.isfinite(obs["policy"]).all().item()

        for _ in range(300):
            actions = torch.empty((NUM_ENVS, 2), device=base.device).uniform_(-1.0, 1.0)
            obs, reward, _, _, _ = env.step(actions)
            reward_min = min(reward_min, reward.min().item())
            reward_max = max(reward_max, reward.max().item())
            finite = finite and torch.isfinite(obs["policy"]).all().item()
            finite = finite and torch.isfinite(reward).all().item()

        print(f"obs shape: {obs_shape}")
        print(f"reward range over 300 random steps: [{reward_min:.6f}, {reward_max:.6f}]")
        print(
            "hold_timer after random steps (s): "
            f"min={base.hold_timer.min().item():.6f}, "
            f"mean={base.hold_timer.mean().item():.6f}, "
            f"max={base.hold_timer.max().item():.6f}"
        )

        # Zero-action probe from fresh task spawns. It checks every sample rather
        # than assuming the passive hull remains outside the hold zone.
        env.reset()
        spawn_distance = torch.norm(
            base.robot.data.root_pos_w[:, :2] - base.hold_point, dim=-1
        )
        previous_timer = base.hold_timer.clone()
        outside_growth_violations = 0
        outside_nonzero_violations = 0

        zero_actions = torch.zeros((NUM_ENVS, 2), device=base.device)
        for _ in range(120):
            env.step(zero_actions)
            distance = torch.norm(
                base.robot.data.root_pos_w[:, :2] - base.hold_point, dim=-1
            )
            inside = distance <= base.cfg.hold_radius
            grew = base.hold_timer > previous_timer + 1.0e-7

            outside_growth_violations += int((grew & ~inside).sum().item())
            outside_nonzero_violations += int(
                ((~inside) & (base.hold_timer > 1.0e-7)).sum().item()
            )
            previous_timer.copy_(base.hold_timer)

        outside_probe_ok = (
            outside_growth_violations == 0 and outside_nonzero_violations == 0
        )

        # Positive control: place env 0 one metre from its hold point, still at
        # rest, and keep zero actions. This makes the "grows inside" half of the
        # timer predicate observable without relying on a random policy arrival.
        probe_id = torch.tensor([0], dtype=torch.long, device=base.device)
        inside_state = base.robot.data.default_root_state[probe_id].clone()
        inside_state[:, :3] += base.scene.env_origins[probe_id]
        inside_state[:, 0] += 1.0
        inside_state[:, 7:] = 0.0
        base.robot.write_root_state_to_sim(inside_state, probe_id)
        base._hold_steps[probe_id] = 0
        base.hold_timer[probe_id] = 0.0

        inside_growth_samples = 0
        previous_inside_timer = base.hold_timer[probe_id].clone()
        for _ in range(60):
            env.step(zero_actions)
            inside_distance = torch.norm(
                base.robot.data.root_pos_w[probe_id, :2] - base.hold_point[probe_id], dim=-1
            )
            inside = inside_distance <= base.cfg.hold_radius
            grew = base.hold_timer[probe_id] > previous_inside_timer + 1.0e-7
            inside_growth_samples += int((inside & grew).sum().item())
            previous_inside_timer.copy_(base.hold_timer[probe_id])

        timer_only_grows_inside = outside_probe_ok and inside_growth_samples > 0
        print(
            "zero-action spawn distance (m): "
            f"min={spawn_distance.min().item():.6f}, "
            f"max={spawn_distance.max().item():.6f}"
        )
        print(
            "zero-action probe: hold_timer grows only inside 2 m: "
            f"{timer_only_grows_inside} "
            f"(inside growth samples={inside_growth_samples}, "
            f"outside violations={outside_growth_violations + outside_nonzero_violations})"
        )

        spawn_ok = bool(
            (spawn_distance >= base.cfg.min_spawn_distance - 1.0e-3).all()
            and (spawn_distance <= base.cfg.max_spawn_distance + 1.0e-3).all()
        )
        return obs_shape == (NUM_ENVS, 3) and finite and spawn_ok and timer_only_grows_inside
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    passed = False
    failure = None
    try:
        passed = main()
    except Exception as exc:  # keep a clean summary line even on runtime failure
        failure = exc
        print(f"smoke exception: {type(exc).__name__}: {exc}")

    # print + flush BEFORE closing kit: on this machine simulation_app.close()
    # hard-kills the process and unflushed stdout is lost.
    if passed:
        print("SMOKE RESULT: PASS")
    elif failure is None:
        print("SMOKE RESULT: FAIL")
    else:
        print(f"SMOKE RESULT: FAIL ({type(failure).__name__})")
    sys.stdout.flush()

    simulation_app.close()
    raise SystemExit(0 if passed else 1)
