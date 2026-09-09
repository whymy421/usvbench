"""Headless runtime smoke test for the T1 ROV path-following task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-USV-PathFollow-Direct-v1"
NUM_ENVS = 16

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# Register this task whether run in USVBench or after copying the folder into
# isaaclab_tasks/direct.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import path_following  # noqa: F401, E402

from path_following.path_following_env_cfg import PathFollowingEnvCfg  # noqa: E402


def main() -> bool:
    env = None
    try:
        env_cfg = PathFollowingEnvCfg()
        env_cfg.scene.num_envs = NUM_ENVS
        env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None)
        base = env.unwrapped

        obs, _ = env.reset()
        obs_shape = tuple(obs["policy"].shape)
        reward_min = float("inf")
        reward_max = float("-inf")
        finite = bool(torch.isfinite(obs["policy"]).all().item())

        for _ in range(300):
            actions = torch.empty((NUM_ENVS, 2), device=base.device).uniform_(-1.0, 1.0)
            obs, reward, _, _, _ = env.step(actions)
            reward_min = min(reward_min, reward.min().item())
            reward_max = max(reward_max, reward.max().item())
            finite = finite and bool(torch.isfinite(obs["policy"]).all().item())
            finite = finite and bool(torch.isfinite(reward).all().item())

        print(f"obs shape: {obs_shape}")
        print(f"reward range over 300 random steps: [{reward_min:.6f}, {reward_max:.6f}]")

        # Fresh paths for a deterministic geometry and gate-order probe.
        env.reset()
        segment_vectors = torch.empty_like(base.waypoints)
        segment_vectors[:, 0] = base.waypoints[:, 0]
        segment_vectors[:, 1:] = base.waypoints[:, 1:] - base.waypoints[:, :-1]
        segment_lengths = torch.norm(segment_vectors, dim=-1)
        headings = torch.atan2(segment_vectors[:, :, 1], segment_vectors[:, :, 0])
        heading_deltas = torch.atan2(
            torch.sin(headings[:, 1:] - headings[:, :-1]),
            torch.cos(headings[:, 1:] - headings[:, :-1]),
        )
        geometry_ok = bool(
            (segment_lengths >= base.cfg.segment_length_min - 1.0e-5).all().item()
            and (segment_lengths <= base.cfg.segment_length_max + 1.0e-5).all().item()
            and (
                heading_deltas.abs()
                <= torch.deg2rad(
                    torch.tensor(base.cfg.heading_change_max_deg, device=base.device)
                )
                + 1.0e-5
            )
            .all()
            .item()
        )

        probe_id = torch.tensor([0], dtype=torch.long, device=base.device)
        probe_waypoints = base.waypoints[0].clone()
        probe_origin = base.scene.env_origins[0, :2].clone()
        zero_actions = torch.zeros((NUM_ENVS, 2), device=base.device)
        ordered_increments = True
        success_only_on_last = True
        gate_trace = []

        for gate_index in range(base.cfg.num_waypoints):
            gate_state = base.robot.data.default_root_state[probe_id].clone()
            gate_state[:, :3] += base.scene.env_origins[probe_id]
            gate_state[:, :2] = probe_origin + probe_waypoints[gate_index]
            gate_state[:, 7:] = 0.0
            base.robot.write_root_state_to_sim(gate_state, probe_id)
            # Exclude intentional smoke-test teleports from distance accumulation.
            base._previous_xy[probe_id] = gate_state[:, :2]

            _, reward, terminated, truncated, _ = env.step(zero_actions)
            fired = bool(torch.as_tensor(terminated)[0].item())
            timed_out = bool(torch.as_tensor(truncated)[0].item())
            expected_success = gate_index == base.cfg.num_waypoints - 1
            success_only_on_last = (
                success_only_on_last and fired == expected_success and not timed_out
            )

            if expected_success:
                observed_gates = int(base.episode_gates_passed[0].item())
                ordered_increments = ordered_increments and observed_gates == base.cfg.num_waypoints
                ordered_increments = ordered_increments and bool(base.episode_success[0].item())
            else:
                observed_gates = int(base.gates_passed[0].item())
                ordered_increments = ordered_increments and observed_gates == gate_index + 1

            gate_trace.append(
                f"{gate_index + 1}:{observed_gates},done={fired},reward={reward[0].item():.3f}"
            )

        waypoint_probe_ok = ordered_increments and success_only_on_last
        print(
            "procedural path bounds: "
            f"length=[{segment_lengths.min().item():.6f}, "
            f"{segment_lengths.max().item():.6f}] m, "
            f"max |heading delta|={torch.rad2deg(heading_deltas.abs()).max().item():.6f} deg, "
            f"ok={geometry_ok}"
        )
        print(f"waypoint advance trace: {' | '.join(gate_trace)}")
        print(
            "waypoint advance probe: increments in order and success only at gate 4: "
            f"{waypoint_probe_ok}"
        )

        metric_shapes_ok = (
            tuple(base.episode_success.shape) == (NUM_ENVS,)
            and tuple(base.time_to_success.shape) == (NUM_ENVS,)
            and tuple(base.path_length.shape) == (NUM_ENVS,)
            and tuple(base.gates_passed.shape) == (NUM_ENVS,)
            and base.gates_passed.dtype == torch.long
            and tuple(base.xte_rms.shape) == (NUM_ENVS,)
        )
        return (
            obs_shape == (NUM_ENVS, 7)
            and finite
            and geometry_ok
            and waypoint_probe_ok
            and metric_shapes_ok
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
