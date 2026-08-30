"""
eval_oracle.py — Scripted pure-pursuit controller, no neural network.

Purpose: establish the CEILING for the rescue task and separate two very
different explanations for the observed ~0.10 rescue rate.

  If the oracle scores high (say > 0.6):
      the physics and task parameters are fine and reachable, and the failure is
      in RL training (learning rate, horizon, reward scale, exploration).

  If the oracle also scores low (say < 0.3):
      the task as configured is not winnable by ANY controller, and the problem
      is in the environment -- most likely control authority, rescue radius,
      casualty timers or spawn distances. No amount of training would have fixed
      it, and the honest result is that the P2 bar was mis-specified.

The controller is deliberately simple, because a simple controller ought to be
enough for this task:

    heading_error = atan2(dir_y, dir_x)        # body-frame bearing to target
    yaw    = clamp(K_YAW * heading_error, -1, 1)
    thrust = clamp(cos(heading_error), 0, 1) * THRUST_SCALE

It drives hardest when pointing at the target and eases off while turning.

It also measures the achieved speed and yaw rate, which tells us directly
whether control authority is sane.

Usage:

  python eval_oracle.py --task=Isaac-RescueBoat-Direct-v1 --num_envs=64 --headless
  python eval_oracle.py --k_yaw=2.0 --thrust_scale=0.6 --headless    # tuning
  python eval_oracle.py --sweep --headless                           # gain sweep
"""

import argparse
import csv
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted oracle controller baseline.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--k_yaw", type=float, default=2.0,
                    help="Proportional gain on heading error.")
parser.add_argument("--thrust_scale", type=float, default=1.0,
                    help="Fraction of max thrust to command (1.0 = full).")
parser.add_argument("--sweep", action="store_true", default=False,
                    help="Sweep K_YAW x THRUST_SCALE instead of a single run.")
parser.add_argument("--strategy", type=str, default="priority",
                    choices=["priority", "nearest", "committed", "all"],
                    help="Target selection. 'priority' uses the env's urgency^2/dist ranking; "
                         "'nearest' ignores urgency; 'committed' locks on until rescued or lost.")
parser.add_argument("--rescue_radius", type=float, default=None,
                    help="Override cfg.rescue_radius (m). Default keeps the config value.")
parser.add_argument("--max_yaw_deg", type=float, default=None,
                    help="Override max_angular_velocity (DEGREES/s). Needs a fresh process.")
parser.add_argument("--radius_sweep", action="store_true", default=False,
                    help="Sweep rescue_radius with the committed strategy. Turning radius is "
                         "v/omega, so if capture radius is below it the boat cannot correct an "
                         "overshoot and orbits outside the zone.")
parser.add_argument("--config_sweep", action="store_true", default=False,
                    help="Compare candidate task configurations. A benchmark bar should sit "
                         "well BELOW the oracle ceiling, so that reaching it means the policy "
                         "is good rather than superhuman.")
parser.add_argument("--out", type=str, default="oracle_results.csv")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

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


def oracle_action(obs, k_yaw, thrust_scale):
    """
    Pure pursuit on the highest-priority casualty.

    obs[:, 0:2] is the body-frame unit vector to priority-0 (see the observation
    layout documented in rescue_boat_env.py), so the bearing is just atan2 of it.
    """
    o = obs["policy"] if isinstance(obs, dict) else obs
    dir_x = o[:, 0]
    dir_y = o[:, 1]

    heading_err = torch.atan2(dir_y, dir_x)

    yaw = torch.clamp(k_yaw * heading_err, -1.0, 1.0)
    # Full ahead when pointing at the target, backing off through the turn.
    thrust = torch.clamp(torch.cos(heading_err), 0.0, 1.0) * thrust_scale

    return torch.stack([thrust, yaw], dim=-1)


def select_target(base, strategy, locked):
    """
    Choose a casualty index per env and return its body-frame bearing.

    'priority' reproduces the environment's own urgency^2/dist ranking, which is
    what the observation exposes to the policy. Note that urgency is 0 for every
    casualty at episode start, so all scores tie at 0 and the initial choice is
    arbitrary; thereafter the ranking re-orders as timers run down at different
    rates, which can pull the boat off a target mid-transit.

    'nearest' ignores urgency entirely.

    'committed' keeps the chosen target until it is rescued or lost, which
    removes mid-transit switching without changing how the target is first
    chosen.
    """
    pos = base.robot.data.root_pos_w
    dp = base.casualty_pos - pos.unsqueeze(1)
    dist = torch.norm(dp[:, :, :2], dim=-1)
    alive = base.casualty_alive

    if strategy == "nearest":
        d = dist.masked_fill(~alive, float("inf"))
        idx = d.argmin(dim=-1)
    else:
        urgency = (1.0 - base.casualty_timer / (base.casualty_init_timer + 1e-6)).clamp(0, 1)
        score = urgency.pow(2.0) / (dist + 1e-3)
        score = score.masked_fill(~alive, float("-inf"))
        idx = score.argmax(dim=-1)

        if strategy == "committed" and locked is not None:
            # Keep the previous target while it is still alive.
            still_ok = torch.gather(alive, 1, locked.unsqueeze(1)).squeeze(1)
            idx = torch.where(still_ok, locked, idx)

    n = base.num_envs
    ar = torch.arange(n, device=base.device)
    tgt = base.casualty_pos[ar, idx]
    delta = tgt - pos

    # World bearing -> body bearing via yaw only (hull stays level).
    q = base.robot.data.root_quat_w
    w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    bearing = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
    bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))  # wrap to [-pi, pi]
    return bearing, idx


def run_trial(env, base, k_yaw, thrust_scale, num_envs, n_cas, episodes, max_ep,
              verbose=True, strategy="priority"):
    obs, _ = env.reset()

    total_steps = int(max_ep) * int(episodes)
    rescues = 0
    prev = int(getattr(base, "reached_count", 0))

    speed_sum = 0.0
    yaw_abs_sum = 0.0
    yaw_abs_max = 0.0
    samples = 0

    locked = None
    switches = 0
    prev_idx = None

    for i in range(total_steps):
        with torch.inference_mode():
            if strategy == "priority":
                act = oracle_action(obs, k_yaw, thrust_scale)
                _, idx = select_target(base, "priority", None)
            else:
                bearing, idx = select_target(base, strategy, locked)
                locked = idx
                yaw_cmd = torch.clamp(k_yaw * bearing, -1.0, 1.0)
                thr_cmd = torch.clamp(torch.cos(bearing), 0.0, 1.0) * thrust_scale
                act = torch.stack([thr_cmd, yaw_cmd], dim=-1)

            # Count target changes: thrashing shows up here even when the
            # rescue rate alone does not explain what went wrong.
            if prev_idx is not None:
                switches += int((idx != prev_idx).sum())
            prev_idx = idx

            obs, _, _, _, _ = env.step(act)

        cur = int(getattr(base, "reached_count", 0))
        if cur > prev:
            rescues += cur - prev
        prev = cur

        if i % 20 == 0:
            try:
                v = base.robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
                w = base.robot.data.root_ang_vel_w[:, 2].abs()
                speed_sum += float(v.mean())
                yaw_abs_sum += float(w.mean())
                yaw_abs_max = max(yaw_abs_max, float(w.max()))
                samples += 1
            except Exception:
                pass

        if verbose and (i + 1) % 2400 == 0:
            print(f"      step {i + 1}/{total_steps}  rescues {rescues}")

    denom = num_envs * n_cas * episodes
    rate = rescues / max(denom, 1)
    mean_speed = speed_sum / max(samples, 1)
    mean_yaw = yaw_abs_sum / max(samples, 1)
    # Switches per env per episode: how often the chosen target changed.
    sw_per_ep = switches / max(num_envs * episodes, 1)
    return rate, rescues, denom, mean_speed, mean_yaw, yaw_abs_max, sw_per_ep


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None

    if args_cli.rescue_radius is not None:
        env_cfg.rescue_radius = args_cli.rescue_radius
    if args_cli.max_yaw_deg is not None:
        # Must be set before the asset spawns; PhysX reads it at creation.
        env_cfg.robot_cfg.spawn.rigid_props.max_angular_velocity = args_cli.max_yaw_deg

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    base = _base_env(env)
    max_ep = float(getattr(base, "max_episode_length", 7200))
    n_cas = int(getattr(base.cfg, "n_casualties", 4))
    ne = args_cli.num_envs
    eps = args_cli.episodes

    # Theoretical control limits, for comparison against what is achieved.
    phys = base.cfg.underwater_physics_cfg
    try:
        from isaaclab_tasks.direct.rescue_boat.rescue_boat_env import MAX_THRUST, MAX_TORQUE
    except Exception:
        MAX_THRUST, MAX_TORQUE = 500.0, 800.0
    v_term = MAX_THRUST / max(phys.max_linear_damping, 1e-6)
    w_term = MAX_TORQUE / max(phys.max_angular_damping, 1e-6)
    # max_angular_velocity is in DEGREES/s and is what actually binds, not the
    # damping-limited value above.
    import math as _m
    w_cap = _m.radians(float(base.cfg.robot_cfg.spawn.rigid_props.max_angular_velocity))
    w_term_capped = min(w_term, w_cap)

    print()
    print("=" * 78)
    print("  ORACLE CONTROLLER BASELINE  (scripted, no neural network)")
    print("=" * 78)
    print(f"  envs {ne}   casualties {n_cas}   episodes/env {eps}   denom {ne * n_cas * eps}")
    print(f"  episode {max_ep:.0f} steps")
    print()
    print("  Control authority implied by the config:")
    print(f"    max thrust {MAX_THRUST:.0f} N / linear damping {phys.max_linear_damping:.0f}"
          f"  -> terminal speed {v_term:.1f} m/s")
    print(f"    max torque {MAX_TORQUE:.0f} Nm / angular damping {phys.max_angular_damping:.0f}"
          f"  -> terminal yaw {w_term:.1f} rad/s  ({w_term / (2 * math.pi):.2f} rev/s)")
    if w_term > 2.0:
        print(f"    NOTE: a real 8.5 m RIB turns at well under 1 rad/s.")
        print(f"          {w_term:.1f} rad/s is roughly {w_term / 0.8:.0f}x realistic and may")
        print(f"          make fine heading control impossible for a learned policy.")
    print("=" * 78)
    print()

    rows = []

    if args_cli.config_sweep:
        # rescue_radius, timer_min, timer_max, spawn_max, label
        # (capture, timer_min, timer_max, spawn_min, spawn_max, label)
        #
        # CONSTRAINT: spawn_min must exceed capture, or some casualties start
        # already rescued. An earlier sweep violated this (capture 20 m against a
        # 10 m spawn minimum), which put 40% of casualties inside the capture
        # radius at t=0 and produced a meaningless 1.0000.
        configs = [
            # The vessel covers 1440 m in a 120 s episode. With casualties inside
            # 60 m every one is reachable ~18x over, so nothing has to be given up
            # and the task does not test prioritisation at all. These configs scale
            # the task to the vehicle: a four-point tour runs ~2.5x the outer
            # radius, so the tour must be long enough to exceed some deadlines.
            #
            # V1 originally used 30-120 m. That was reduced to 10-50 m to address
            # "reward sparsity" which was in fact a symptom of a hull limited to
            # 0.8 m/s by the capsize defect. The workaround outlived its cause.
            (15.0, 40.0, 90.0,  60.0, 150.0, "spawn 60-150, deadlines 40-90"),
            (15.0, 40.0, 90.0,  80.0, 200.0, "spawn 80-200, deadlines 40-90"),
            (20.0, 40.0, 90.0, 100.0, 250.0, "spawn 100-250, deadlines 40-90"),
            (20.0, 30.0, 70.0, 100.0, 250.0, "spawn 100-250, deadlines 30-70"),
            (20.0, 40.0,100.0, 150.0, 350.0, "spawn 150-350, deadlines 40-100"),
            (20.0, 30.0, 80.0, 150.0, 350.0, "spawn 150-350, deadlines 30-80"),
        ]
        for c in configs:
            assert c[3] > c[0], f"invalid config {c[5]}: spawn_min <= capture"
        print("  A benchmark bar should sit BELOW the oracle ceiling with headroom.")
        print("  Target: oracle ~0.85 so that a 0.70 bar means 'good, not perfect'.\n")
        print(f"  {'config':<38s} {'capture':>6s} {'deadlines':>11s} {'spawn':>9s} {'oracle':>8s}")
        print("  " + "-" * 76)

        for rr, tmin, tmax, smin, smax, label in configs:
            base.cfg.rescue_radius = rr
            base.cfg.casualty_timer_min = tmin
            base.cfg.casualty_timer_max = tmax
            base.cfg.casualty_spawn_r_min = smin
            base.cfg.casualty_spawn_r_max = smax
            rate, resc, denom, spd, yaw, ymax, sw = run_trial(
                env, base, 2.0, 0.7, ne, n_cas, eps, max_ep, verbose=False,
                strategy="committed")
            head = ""
            if rate >= 0.85:
                head = "  <- good headroom"
            elif rate >= 0.75:
                head = "  <- workable"
            print(f"  {label:<38s} {rr:>5.0f}m {tmin:>4.0f}-{tmax:<4.0f}s "
                  f"{smin:>3.0f}-{smax:<3.0f}m {rate:>8.4f}{head}")
            rows.append((label, 0.7, rate, resc, denom, spd, yaw, ymax))

        print()
        print("=" * 78)
        ok = [r for r in rows if r[2] >= 0.85]
        if ok:
            pick = min(ok, key=lambda r: r[2])   # least relaxed config that clears 0.85
            print(f"  Least-relaxed config with headroom: {pick[0]}  ({pick[2]:.4f})")
            print("  Use this, and keep the 0.70 bar. Changing task parameters changes the")
            print("  benchmark definition, so it needs your supervisor's agreement and must")
            print("  be stated clearly in the methods.")
        else:
            best = max(rows, key=lambda r: r[2])
            print(f"  Nothing reaches 0.85. Best is {best[0]} at {best[2]:.4f}.")
            print("  Either drop n_casualties from 4 to 3, or lower the bar to ~0.55 and")
            print("  report the oracle ceiling alongside it as the empirical maximum.")
        print("=" * 78)

    elif args_cli.radius_sweep:
        # A pursuit controller can only capture a target if it can turn inside the
        # capture zone. Minimum turning radius is v/omega, so when rescue_radius
        # falls below that the boat overshoots and orbits rather than converging.
        v_cruise = 8.0
        r_turn = v_cruise / max(w_term_capped, 1e-6)
        print(f"  Cruise ~{v_cruise:.0f} m/s at {w_term_capped:.2f} rad/s")
        print(f"  -> minimum turning radius {r_turn:.1f} m")
        print(f"  -> current rescue_radius  {base.cfg.rescue_radius:.1f} m")
        print(f"  -> hull length            8.5 m\n")

        for rr in [5.0, 8.0, 12.0, 16.0, 20.0]:
            base.cfg.rescue_radius = rr
            rate, resc, denom, spd, yaw, ymax, sw = run_trial(
                env, base, 2.0, 0.7, ne, n_cas, eps, max_ep, verbose=False,
                strategy="committed")
            flag = "  <- turning radius" if abs(rr - r_turn) < 2.5 else ""
            print(f"  rescue_radius {rr:5.1f} m   rescue_rate {rate:.4f}  "
                  f"({resc}/{denom}){flag}")
            rows.append((f"r={rr}", 0.7, rate, resc, denom, spd, yaw, ymax))
        print()
        best_r = max(rows, key=lambda r: r[2])
        print("=" * 78)
        print(f"  Best: {best_r[0]}  ->  {best_r[2]:.4f}")
        if best_r[2] >= 0.85:
            print("  Clear headroom above the 0.70 bar once the capture zone exceeds the")
            print("  turning radius. The original 5 m was smaller than the hull (8.5 m).")
        elif best_r[2] >= 0.70:
            print("  The bar is reachable but sits AT the oracle ceiling, so a learned")
            print("  policy would have to match a hand-tuned controller with perfect state")
            print("  access. Relax the deadlines or spawn distances for headroom.")
        else:
            print("  Still short of 0.70 even with a generous capture zone, so the")
            print("  deadlines or spawn distances are also binding.")
        print("=" * 78)

    elif args_cli.strategy == "all":
        # Same boat, same gains, three target-selection rules. Isolates task
        # DESIGN from task DIFFICULTY: if 'committed' or 'nearest' scores far
        # above 'priority', the shortfall is the priority heuristic re-ranking
        # mid-transit, not the physics or the deadlines.
        print("  Comparing target-selection strategies (K_YAW=2.0, THRUST=0.7)\n")
        best_by_strategy = {}
        for strat in ["priority", "nearest", "committed"]:
            print(f"  strategy = {strat}")
            rate, resc, denom, spd, yaw, ymax, sw = run_trial(
                env, base, 2.0, 0.7, ne, n_cas, eps, max_ep, verbose=False, strategy=strat)
            print(f"      rescue_rate {rate:.4f}  ({resc}/{denom})   speed {spd:.2f} m/s"
                  f"   target switches/episode {sw:.1f}")
            best_by_strategy[strat] = rate
            rows.append((strat, 0.7, rate, resc, denom, spd, yaw, ymax))

        print()
        print("=" * 78)
        print("  STRATEGY COMPARISON")
        print("=" * 78)
        for s, r in sorted(best_by_strategy.items(), key=lambda kv: -kv[1]):
            print(f"  {s:<12s} {r:.4f}")
        p = best_by_strategy["priority"]
        c = max(best_by_strategy["committed"], best_by_strategy["nearest"])
        print()
        if c > p * 1.4:
            print(f"  Target thrashing is the bottleneck: {p:.3f} -> {c:.3f} once the")
            print("  controller stops re-ranking mid-transit. The priority heuristic, not")
            print("  the physics or the deadlines, is costing most of the performance.")
            print("  Note urgency = 0 for every casualty at episode start, so the initial")
            print("  ranking is an arbitrary tie-break.")
        else:
            print(f"  Target selection is not the main bottleneck ({p:.3f} vs {c:.3f}).")
            print("  The task parameters themselves are likely too tight: check")
            print("  rescue_radius, casualty timers and spawn radii.")
        print("=" * 78)

    elif args_cli.sweep:
        k_yaws = [0.5, 1.0, 2.0, 4.0]
        scales = [0.4, 0.7, 1.0]
        print(f"  Sweeping {len(k_yaws)} gains x {len(scales)} thrust scales\n")
        for k in k_yaws:
            for s in scales:
                print(f"  K_YAW={k}  THRUST={s}")
                rate, resc, denom, spd, yaw, ymax, _sw = run_trial(
                    env, base, k, s, ne, n_cas, eps, max_ep, verbose=False,
                    strategy=args_cli.strategy)
                print(f"      rescue_rate {rate:.4f}  ({resc}/{denom})   "
                      f"speed {spd:.2f} m/s   yaw |{yaw:.2f}| max {ymax:.2f} rad/s")
                rows.append((k, s, rate, resc, denom, spd, yaw, ymax))
        print()
        rows.sort(key=lambda r: r[2], reverse=True)
        print("=" * 78)
        print("  RANKED")
        print("=" * 78)
        print(f"  {'K_YAW':>6s} {'THRUST':>7s} {'rate':>8s} {'speed':>7s} {'yaw':>7s}")
        for k, s, rate, resc, denom, spd, yaw, ymax in rows:
            print(f"  {k:>6.1f} {s:>7.2f} {rate:>8.4f} {spd:>7.2f} {yaw:>7.2f}")
        best = rows[0]
        print()
        print(f"  Best oracle: K_YAW={best[0]} THRUST={best[1]} -> {best[2]:.4f}")
    else:
        rate, resc, denom, spd, yaw, ymax, _sw = run_trial(
            env, base, args_cli.k_yaw, args_cli.thrust_scale, ne, n_cas, eps, max_ep,
            strategy=args_cli.strategy)
        rows.append((args_cli.k_yaw, args_cli.thrust_scale, rate, resc, denom, spd, yaw, ymax))
        print()
        print("=" * 78)
        print(f"  ORACLE rescue_rate {rate:.4f}   ({resc}/{denom})")
        print(f"  mean speed {spd:.2f} m/s (terminal {v_term:.1f})")
        print(f"  mean |yaw rate| {yaw:.2f} rad/s, peak {ymax:.2f} (terminal {w_term:.1f})")
        print("=" * 78)

    best_rate = max(r[2] for r in rows)
    print()
    print("  INTERPRETATION")
    print("  " + "-" * 60)
    print(f"  Best oracle rescue_rate : {best_rate:.4f}")
    print(f"  (compare against your own trained policy separately)")
    print(f"  P2 bar                  : 0.70")
    print()
    if best_rate >= 0.60:
        print("  The task IS winnable. Physics and parameters are fine, and the gap")
        print("  is entirely in RL training. Fix hyperparameters, not the environment.")
    elif best_rate >= 0.30:
        print("  Partially winnable. A simple controller beats the learned policy")
        print("  substantially, so training is failing, but the 0.70 bar also looks")
        print("  optimistic for this configuration.")
    else:
        print("  The task is NOT winnable as configured -- even a direct-pursuit")
        print("  controller cannot reach the casualties in time. The problem is the")
        print("  ENVIRONMENT, not the policy. Check control authority, rescue_radius,")
        print("  casualty timers and spawn radii. No training run could have passed.")
    print()

    out = os.path.abspath(args_cli.out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k_yaw", "thrust_scale", "rescue_rate", "rescues", "denominator",
                    "mean_speed_mps", "mean_yaw_rate", "peak_yaw_rate"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", r[3], r[4],
                        f"{r[5]:.3f}", f"{r[6]:.3f}", f"{r[7]:.3f}"])
    print(f"  Written: {out}")
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
