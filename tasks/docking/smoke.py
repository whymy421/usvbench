"""Headless runtime smoke test for the boat docking task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TASK_ID = "Isaac-USV-Dock-Direct-v1"
NUM_ENVS = 16
TASK_DIR = Path(__file__).resolve().parent

# Import the dependency-free module directly so this check can run without
# importing gymnasium, torch, Isaac Lab, or starting a simulation application.
sys.path.insert(0, str(TASK_DIR))
from curriculum import DockingCurriculum  # noqa: E402


def curriculum_smoke_check() -> bool:
    curriculum = DockingCurriculum()
    for _ in range(20):
        curriculum.update(False)
    stayed_at_start = curriculum.spawn_distance == 3.0

    for _ in range(200):
        curriculum.update(True)
        if curriculum.spawn_distance > 3.0:
            break
    advanced_one_stage = curriculum.spawn_distance == 5.5
    return stayed_at_start and advanced_one_stage


if "--curriculum-only" in sys.argv:
    curriculum_ok = curriculum_smoke_check()
    print(f"curriculum fake-success advance: {curriculum_ok}")
    print(f"SMOKE RESULT: {'PASS' if curriculum_ok else 'FAIL'}")
    sys.stdout.flush()
    raise SystemExit(0 if curriculum_ok else 1)


from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

# Register this task whether run from USVBench or after the folder is copied
# into isaaclab_tasks/direct.
sys.path.insert(0, str(TASK_DIR.parent))
import docking  # noqa: F401, E402

from docking.docking_env_cfg import DockingEnvCfg  # noqa: E402


def main() -> bool:
    env = None
    try:
        curriculum_ok = curriculum_smoke_check()
        env_cfg = DockingEnvCfg()
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

        print(f"curriculum fake-success advance: {curriculum_ok}")
        print(f"obs shape: {obs_shape}")
        print(f"reward range over 300 random steps: [{reward_min:.6f}, {reward_max:.6f}]")
        print(
            "hold_timer after random steps (s): "
            f"min={base.hold_timer.min().item():.6f}, "
            f"mean={base.hold_timer.mean().item():.6f}, "
            f"max={base.hold_timer.max().item():.6f}"
        )

        env.reset()
        spawn_distance = torch.norm(
            base.robot.data.root_pos_w[:, :2] - base.dock_point, dim=-1
        )
        previous_timer = base.hold_timer.clone()
        predicate_violations = 0

        zero_actions = torch.zeros((NUM_ENVS, 2), device=base.device)
        for _ in range(120):
            env.step(zero_actions)
            instantaneous_success = base._task_state()[3]
            grew = base.hold_timer > previous_timer + 1.0e-7
            predicate_violations += int((grew & ~instantaneous_success).sum().item())
            predicate_violations += int(
                ((~instantaneous_success) & (base.hold_timer > 1.0e-7)).sum().item()
            )
            previous_timer.copy_(base.hold_timer)

        # Positive control: put env 0 one metre from the dock, aligned with
        # world +X and at rest. Body -X points +X at a pi-radian body yaw.
        probe_id = torch.tensor([0], dtype=torch.long, device=base.device)
        inside_state = base.robot.data.default_root_state[probe_id].clone()
        inside_state[:, :3] += base.scene.env_origins[probe_id]
        inside_state[:, 0] += 1.0
        dock_aligned_yaw = torch.full((1, 1), torch.pi, device=base.device)
        inside_state[:, 3:7] = math_utils.quat_from_angle_axis(
            dock_aligned_yaw, base.up_dir
        ).reshape(1, 4)
        inside_state[:, 7:] = 0.0
        base.robot.write_root_state_to_sim(inside_state, probe_id)
        base._hold_steps[probe_id] = 0
        base.hold_timer[probe_id] = 0.0

        inside_growth_samples = 0
        previous_inside_timer = base.hold_timer[probe_id].clone()
        for _ in range(60):
            env.step(zero_actions)
            predicate_true = base._task_state()[3][probe_id]
            grew = base.hold_timer[probe_id] > previous_inside_timer + 1.0e-7
            inside_growth_samples += int((predicate_true & grew).sum().item())
            previous_inside_timer.copy_(base.hold_timer[probe_id])

        # Break heading alignment and verify the consecutive timer resets.
        broken_state = base.robot.data.root_state_w[probe_id].clone()
        broken_state[:, 3:7] = base.robot.data.default_root_state[probe_id, 3:7]
        broken_state[:, 7:] = 0.0
        base.robot.write_root_state_to_sim(broken_state, probe_id)
        env.step(zero_actions)
        break_resets_timer = base.hold_timer[probe_id].item() == 0.0

        predicate_timer_ok = (
            predicate_violations == 0
            and inside_growth_samples > 0
            and break_resets_timer
        )
        spawn_ok = bool(
            torch.allclose(
                spawn_distance,
                torch.full_like(spawn_distance, base.current_spawn_distance),
                atol=1.0e-3,
                rtol=0.0,
            )
        )
        print(
            "spawn distance (m): "
            f"min={spawn_distance.min().item():.6f}, "
            f"max={spawn_distance.max().item():.6f}"
        )
        print(
            "three-factor predicate timer: "
            f"{predicate_timer_ok} "
            f"(inside growth samples={inside_growth_samples}, "
            f"violations={predicate_violations}, break reset={break_resets_timer})"
        )

        return (
            obs_shape == (NUM_ENVS, 6)
            and finite
            and curriculum_ok
            and spawn_ok
            and predicate_timer_ok
        )
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

    # Print and flush BEFORE closing Kit: on this machine close() hard-kills the
    # process and unflushed stdout is lost.
    if passed:
        print("SMOKE RESULT: PASS")
    elif failure is None:
        print("SMOKE RESULT: FAIL")
    else:
        print(f"SMOKE RESULT: FAIL ({type(failure).__name__})")
    sys.stdout.flush()

    simulation_app.close()
    raise SystemExit(0 if passed else 1)
