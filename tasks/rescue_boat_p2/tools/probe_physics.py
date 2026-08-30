"""
probe_physics.py — Open-loop physics probe for the rescue boat.

The oracle sweep showed every scripted controller topping out around 0.8 m/s when
the configuration implies a terminal speed of 12 m/s. That is a ~15x shortfall, so
before touching the policy or the reward we need to know what the hull is actually
doing.

This applies a CONSTANT command with NO feedback and logs the resulting motion.
Any shortfall is therefore purely the physics, with the controller removed from
the picture entirely.

Three tests:

  1. FULL THRUST, ZERO TORQUE
     Straight-line acceleration. Should approach 500/30 = 16.7 m/s (capped at
     12 by max_linear_velocity) with a time constant of m/c = 10 s.
     Reports the speed actually reached, and the heave, roll and pitch behaviour
     along the way.

  2. ZERO THRUST, FULL TORQUE
     Yaw step response. Should approach 800/60 = 13.3 rad/s. What it actually
     reaches reveals the effective rotational inertia.

  3. FREE FLOAT (no command)
     Pure heave response from the spawn state. The buoyancy restoring force is
     ~2287 N/m against 300 kg, giving a ~2.3 s period at a damping ratio near
     0.018 -- essentially undamped. If the hull oscillates through the surface
     it will alternate between water damping (30) and air damping (0.5), and any
     pitch it acquires tilts the thrust vector out of horizontal.

Usage:
  python probe_physics.py --task=Isaac-RescueBoat-Direct-v1 --num_envs=4 --headless
"""

import argparse
import csv
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open-loop physics probe.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Isaac-RescueBoat-Direct-v1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=1800, help="Steps per test (1800 = 30 s).")
parser.add_argument("--out", type=str, default="physics_probe.csv")

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


def quat_to_rpy(q):
    """[N,4] wxyz -> roll, pitch, yaw in degrees."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr, cosr)
    sinp = (2 * (w * y - z * x)).clamp(-1, 1)
    pitch = torch.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny, cosy)
    return (torch.rad2deg(roll), torch.rad2deg(pitch), torch.rad2deg(yaw))


def run_test(env, base, thrust, torque, steps, name, rows):
    env.reset()
    print(f"\n  {name}   thrust={thrust:+.1f}  torque={torque:+.1f}")
    print(f"  {'t(s)':>6s} {'speed':>8s} {'z':>8s} {'roll':>8s} {'pitch':>8s} {'yawrate':>9s}")
    print("  " + "-" * 54)

    n = base.num_envs
    dev = base.device
    act = torch.zeros(n, 2, device=dev)
    act[:, 0] = thrust
    act[:, 1] = torque

    peak_speed = 0.0
    z_min, z_max = 1e9, -1e9
    roll_max = pitch_max = 0.0
    peak_yaw = 0.0

    for i in range(steps):
        with torch.inference_mode():
            env.step(act)

        if i % 60 == 0:  # once per simulated second
            pos = base.robot.data.root_pos_w
            vel = base.robot.data.root_lin_vel_w
            ang = base.robot.data.root_ang_vel_w
            q = base.robot.data.root_quat_w

            spd = float(vel[:, :2].norm(dim=-1).mean())
            z = float(pos[:, 2].mean())
            roll, pitch, _ = quat_to_rpy(q)
            r = float(roll.abs().mean())
            p = float(pitch.abs().mean())
            yr = float(ang[:, 2].abs().mean())

            peak_speed = max(peak_speed, spd)
            z_min, z_max = min(z_min, z), max(z_max, z)
            roll_max = max(roll_max, r)
            pitch_max = max(pitch_max, p)
            peak_yaw = max(peak_yaw, yr)

            t = i / 60.0
            if i % 300 == 0:  # print every 5 s
                print(f"  {t:>6.1f} {spd:>8.2f} {z:>8.2f} {r:>8.1f} {p:>8.1f} {yr:>9.2f}")

            rows.append([name, f"{t:.2f}", f"{spd:.3f}", f"{z:.3f}",
                         f"{r:.2f}", f"{p:.2f}", f"{yr:.3f}"])

    print("  " + "-" * 54)
    print(f"  peak speed {peak_speed:.2f} m/s   peak |yaw| {peak_yaw:.2f} rad/s")
    print(f"  z range [{z_min:.2f}, {z_max:.2f}]  (heave amplitude {(z_max - z_min) / 2:.2f} m)")
    print(f"  peak |roll| {roll_max:.1f} deg   peak |pitch| {pitch_max:.1f} deg")
    return dict(peak_speed=peak_speed, peak_yaw=peak_yaw, z_min=z_min, z_max=z_max,
                roll_max=roll_max, pitch_max=pitch_max)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = _base_env(env)

    phys = base.cfg.underwater_physics_cfg
    try:
        from isaaclab_tasks.direct.rescue_boat.rescue_boat_env import MAX_THRUST, MAX_TORQUE
    except Exception:
        MAX_THRUST, MAX_TORQUE = 500.0, 800.0

    mass = 300.0
    v_term = MAX_THRUST / phys.max_linear_damping
    w_term = MAX_TORQUE / phys.max_angular_damping
    tau = mass / phys.max_linear_damping

    weight = mass * phys.gravity
    buoy_full = phys.water_density * phys.gravity * phys.rov_volume
    sub_eq = weight / buoy_full
    z_eq = -sub_eq * phys.rov_height
    k_heave = buoy_full / phys.rov_height
    omega = math.sqrt(k_heave / mass)
    zeta = phys.max_linear_damping / (2 * math.sqrt(k_heave * mass))

    # ── What the simulator ACTUALLY has, as opposed to what the config asks for ──
    # Yaw response depends on rotational inertia, which comes from the USD asset
    # and is not set by MassPropertiesCfg. If the mesh implies a large inertia
    # tensor, full torque produces almost no angular acceleration and the hull
    # cannot turn regardless of damping or velocity limits.
    print()
    print("=" * 70)
    print("  ACTUAL RIGID BODY PROPERTIES (read from the simulator)")
    print("=" * 70)
    izz = None
    try:
        view = base.robot.root_physx_view
        masses = view.get_masses()
        inertias = view.get_inertias()
        print(f"  mass            : {float(masses[0].sum()):.1f} kg   (config asks 300)")
        inert = inertias[0].reshape(-1)
        if inert.numel() >= 9:
            ixx, iyy, izz = float(inert[0]), float(inert[4]), float(inert[8])
            print(f"  inertia Ixx/Iyy/Izz : {ixx:.1f} / {iyy:.1f} / {izz:.1f} kg m^2")
            L, W = 8.5, 3.0
            izz_expect = 300.0 * (L * L + W * W) / 12.0
            print(f"  expected Izz for an {L} x {W} m hull at 300 kg : {izz_expect:.0f} kg m^2")
            if izz > 5 * izz_expect:
                print(f"  *** Izz is {izz / izz_expect:.0f}x larger than expected ***")
                print(f"      angular accel at full torque = {800.0 / izz:.4f} rad/s^2")
                print(f"      time to reach 1.0 rad/s      = {1.0 / (800.0 / izz):.0f} s")
                print("      This alone would prevent the boat from turning.")
            else:
                print(f"  angular accel at full torque : {800.0 / izz:.3f} rad/s^2")
                print(f"  time to reach 1.0 rad/s      : {izz / 800.0:.1f} s")
        else:
            print(f"  inertia (raw): {inert.tolist()[:9]}")
    except Exception as e:
        print(f"  could not read rigid body properties: {e}")
    print(f"  HORIZONTAL_ACTUATION = {os.environ.get('HORIZONTAL_ACTUATION', '1')}")
    print(f"  max_angular_velocity (cfg, DEGREES/s) = "
          f"{base.cfg.robot_cfg.spawn.rigid_props.max_angular_velocity}")
    print("=" * 70)

    print()
    print("=" * 70)
    print("  OPEN-LOOP PHYSICS PROBE")
    print("=" * 70)
    print("  Predicted from config:")
    print(f"    terminal speed      {v_term:6.1f} m/s   (time constant {tau:.1f} s)")
    print(f"    terminal yaw rate   {w_term:6.1f} rad/s")
    print(f"    float equilibrium   z = {z_eq:6.2f} m   (submergence {sub_eq:.3f})")
    print(f"    heave period        {2 * math.pi / omega:6.2f} s")
    print(f"    heave damping ratio {zeta:6.3f}", end="")
    print("   <- essentially undamped" if zeta < 0.1 else "")
    print("=" * 70)

    rows = []
    r1 = run_test(env, base, 1.0, 0.0, args_cli.steps, "full_thrust", rows)
    r2 = run_test(env, base, 0.0, 1.0, args_cli.steps, "full_torque", rows)
    r3 = run_test(env, base, 0.0, 0.0, args_cli.steps, "free_float", rows)

    # ── PHASE 3 — combined thrust and yaw, swept through heading ─────────────
    # Phases 1 and 2 are not sufficient. Both an incorrectly-framed restoring
    # torque and a double-rotated wrench are exactly correct at yaw = 0, which is
    # where every episode starts and where a single-axis open-loop test sits. The
    # earlier probe therefore reported a healthy yaw rate while the hull rolled to
    # 172 degrees, because it only ever sampled the one attitude where the bug is
    # invisible.
    #
    # Driving thrust and yaw together forces the hull through every heading, so a
    # yaw-dependent fault shows up as tilt that grows with turn angle.
    print()
    print("=" * 70)
    print("  PHASE 3 — combined thrust + yaw, swept through heading")
    print("=" * 70)
    print("  Any heading-dependent fault in the restoring torque appears here.")
    print(f"  {'t(s)':>6s} {'heading':>9s} {'tilt':>8s} {'roll':>8s} {'pitch':>8s} {'z':>8s}")
    print("  " + "-" * 56)

    env.reset()
    n = base.num_envs
    dev = base.device
    act = torch.zeros(n, 2, device=dev)
    act[:, 0] = 1.0     # full thrust
    act[:, 1] = 0.6     # sustained turn

    tilt_max = 0.0
    tilt_at = 0.0
    z_min3 = 1e9
    prev_yaw = None
    cum_yaw = 0.0

    for i in range(args_cli.steps):
        with torch.inference_mode():
            env.step(act)
        if i % 60 == 0:
            q = base.robot.data.root_quat_w
            pos = base.robot.data.root_pos_w
            roll, pitch, yaw = quat_to_rpy(q)

            # Tilt = angle between hull +Z and world +Z. Heading-independent by
            # construction, so it is the right quantity to assert on.
            w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            up_z = 1 - 2 * (x_ * x_ + y_ * y_)
            tilt = torch.rad2deg(torch.acos(up_z.clamp(-1, 1)))

            t_mean = float(tilt.mean())
            yaw_mean = float(yaw.mean())
            if prev_yaw is not None:
                d = yaw_mean - prev_yaw
                if d > 180: d -= 360
                if d < -180: d += 360
                cum_yaw += abs(d)
            prev_yaw = yaw_mean

            if t_mean > tilt_max:
                tilt_max, tilt_at = t_mean, cum_yaw
            z_min3 = min(z_min3, float(pos[:, 2].mean()))

            if i % 300 == 0:
                print(f"  {i/60.0:>6.1f} {yaw_mean:>9.1f} {t_mean:>8.2f} "
                      f"{float(roll.abs().mean()):>8.2f} {float(pitch.abs().mean()):>8.2f} "
                      f"{float(pos[:,2].mean()):>8.2f}")
            rows.append(["turn_sweep", f"{i/60.0:.2f}", "", f"{float(pos[:,2].mean()):.3f}",
                         f"{float(roll.abs().mean()):.2f}", f"{float(pitch.abs().mean()):.2f}",
                         f"{t_mean:.3f}"])

    print("  " + "-" * 56)
    print(f"  cumulative heading change : {cum_yaw:.0f} deg ({cum_yaw/360:.2f} turns)")
    print(f"  peak tilt                 : {tilt_max:.2f} deg (after {tilt_at:.0f} deg of turn)")
    print(f"  minimum z                 : {z_min3:.2f} m")
    TILT_LIMIT = 5.0
    phase3_pass = (tilt_max <= TILT_LIMIT) and (z_min3 > -5.0)
    print()
    print(f"  PHASE 3: {'PASS' if phase3_pass else 'FAIL'}  "
          f"(tilt limit {TILT_LIMIT} deg, z limit -5 m)")
    if not phase3_pass:
        print("  The restoring torque is heading-dependent. Check that it uses the")
        print("  yaw-invariant form K*(hull_up x world_up) rather than Euler roll and")
        print("  pitch applied about world axes.")
    print("=" * 70)

    print()
    print("=" * 70)
    print("  DIAGNOSIS")
    print("=" * 70)
    eff = r1["peak_speed"] / max(min(v_term, 12.0), 1e-6)
    print(f"  surge : reached {r1['peak_speed']:.2f} m/s of {min(v_term, 12.0):.1f} expected"
          f"  ({eff * 100:.0f}%)")
    effw = r2["peak_yaw"] / max(w_term, 1e-6)
    print(f"  yaw   : reached {r2['peak_yaw']:.2f} rad/s of {w_term:.1f} expected"
          f"  ({effw * 100:.0f}%)")
    heave = (r3["z_max"] - r3["z_min"]) / 2
    print(f"  heave : amplitude {heave:.2f} m about z={z_eq:.2f}, "
          f"peak pitch {r3['pitch_max']:.1f} deg")
    print()

    if eff < 0.5:
        print("  SURGE IS THE BOTTLENECK. The hull cannot reach its configured speed,")
        print("  so no controller and no policy can complete the task in 120 s.")
        if r1["pitch_max"] > 15 or r1["roll_max"] > 15:
            print(f"  Attitude is the likely cause: peak roll {r1['roll_max']:.0f} deg /")
            print(f"  pitch {r1['pitch_max']:.0f} deg means body-frame +X thrust is being")
            print("  rotated substantially out of horizontal, so much of the 500 N is")
            print("  pushing the hull down or up rather than forward.")
        if r3["z_max"] > 0:
            print("  The hull also breaches the surface during heave, alternating between")
            print(f"  water damping ({phys.max_linear_damping}) and air damping "
                  f"({phys.air_linear_damping}), which makes the dynamics discontinuous.")
        print()
        print("  Candidate fixes, cheapest first:")
        print("    - damp heave: raise a vertical damping term so the hull settles")
        print("    - constrain roll/pitch, or project thrust onto the horizontal plane")
        print("    - raise rov_volume so the hull rides higher and more stably")
    else:
        print("  Surge is fine. The bottleneck is elsewhere -- look at turning and at")
        print("  how much time is lost re-acquiring targets.")
    print("=" * 70)

    out = os.path.abspath(args_cli.out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test", "t_s", "speed_mps", "z_m", "roll_deg", "pitch_deg", "yaw_rate_or_tilt"])
        w.writerows(rows)
    print(f"\n  Written: {out}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
