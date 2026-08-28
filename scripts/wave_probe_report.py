"""Read a wave_probe_dump .npz and answer the two questions it was run for.

Question 1 -- what does the sea actually do to this hull?
    Distributions of tilt, tilt rate, heave, wave force and wave torque, in
    physical units, so the magnitudes in a paper come from a measurement
    rather than from a linear-wave hand calculation.

Question 2 -- what normalisation should the attitude observation block use?
    The planned tasks/_shared/attitude.py block divides each channel by a
    FROZEN scale and clamps to [-1, 1], exactly as body_planar_kinematics
    does. Pick the scale too small and the channel saturates and carries no
    information; too large and it never leaves the noise floor. This script
    reports the p99 of each candidate channel, which is the defensible place
    to put the clamp: 1% of steps saturate, 99% stay informative.

Usage:
    python scripts/wave_probe_report.py probe_w600_zero.npz [more.npz ...]

Pure NumPy. No Isaac, no torch.
"""

import json
import math
import sys

import numpy as np


CHANNELS = (
    # (label, extractor, unit, suggested rounding for a frozen scale)
    ("roll", lambda a: a["roll_pitch"][:, :, 0], "rad", 0.05),
    ("pitch", lambda a: a["roll_pitch"][:, :, 1], "rad", 0.05),
    ("tilt_magnitude", lambda a: np.sqrt((a["roll_pitch"] ** 2).sum(-1)), "rad", 0.05),
    ("roll_rate_p", lambda a: a["ang_vel_w"][:, :, 0], "rad/s", 0.25),
    ("pitch_rate_q", lambda a: a["ang_vel_w"][:, :, 1], "rad/s", 0.25),
    ("yaw_rate_r", lambda a: a["ang_vel_w"][:, :, 2], "rad/s", 0.25),
    ("heave_dev", lambda a: a["pos_w"][:, :, 2] - a["pos_w"][:, :, 2].mean(), "m", 0.05),
    ("vert_vel", lambda a: a["lin_vel_w"][:, :, 2], "m/s", 0.1),
    ("surge", lambda a: a["body_uvr"][:, :, 0], "m/s", 0.5),
    ("sway", lambda a: a["body_uvr"][:, :, 1], "m/s", 0.5),
    ("wave_force_xy", lambda a: np.sqrt((a["wave_force"][:, :, :2] ** 2).sum(-1)), "N", 1.0),
    ("wave_force_z", lambda a: a["wave_force"][:, :, 2], "N", 1.0),
    ("wave_torque_xy", lambda a: np.sqrt((a["wave_torque"][:, :, :2] ** 2).sum(-1)), "N.m", 5.0),
    ("wave_eta", lambda a: a["wave_eta"], "m", 0.05),
    ("hold_err", lambda a: a["hold_err"], "m", 1.0),
)


def _round_up(value, step):
    """Round a scale UP to a clean multiple, so the clamp is never tighter."""
    if step <= 0 or not math.isfinite(value):
        return value
    return math.ceil(value / step) * step


def report(path):
    data = np.load(path, allow_pickle=False)
    summary = json.loads(str(data["summary"]))
    print("=" * 78)
    print(f"FILE {path}")
    print(f"  task={summary['task']}  zero_action={summary['zero_action']}")
    print(f"  checkpoint={summary.get('checkpoint')}")
    print(f"  overrides={summary.get('applied_overrides')}")
    print(f"  steps={summary['recorded_steps']}  envs={summary['num_envs']}  "
          f"hold_radius={summary['hold_radius_m']} m")
    hs = summary.get("hs_m") or []
    tp = summary.get("tp_s") or []
    if hs:
        print(f"  Hs={min(hs):.3f}-{max(hs):.3f} m   Tp={min(tp):.2f}-{max(tp):.2f} s")
    print(f"  inside_hold_frac={summary['inside_hold_frac']:.4f}   "
          f"hold_err_rms={summary['hold_err_rms_m']:.3f} m")
    print()
    print(f"{'channel':<16}{'unit':>6}{'rms':>10}{'p50':>10}{'p95':>10}"
          f"{'p99':>10}{'max':>10}{'scale':>9}")
    print("-" * 78)
    for label, fn, unit, step in CHANNELS:
        try:
            x = np.asarray(fn(data), dtype=np.float64).ravel()
        except KeyError:
            continue
        if x.size == 0:
            continue
        absx = np.abs(x)
        rms = float(np.sqrt((x ** 2).mean()))
        p50, p95, p99 = (float(np.quantile(absx, q)) for q in (0.5, 0.95, 0.99))
        mx = float(absx.max())
        print(f"{label:<16}{unit:>6}{rms:>10.4f}{p50:>10.4f}{p95:>10.4f}"
              f"{p99:>10.4f}{mx:>10.4f}{_round_up(p99, step):>9.3g}")
    # ---- drift from the SPAWN point, not distance to the hold point --------
    # hold_err starts at the spawn offset (up to max_spawn_distance), so for a
    # zero-action run it is dominated by "the boat never went anywhere",
    # NOT by "the sea carried it away". The honest passive-drift measure is
    # displacement from each env's own starting position.
    pos = data["pos_w"]
    drift = np.linalg.norm(pos[:, :, :2] - pos[0:1, :, :2], axis=-1)
    print()
    print("PASSIVE DRIFT (displacement from each env's own start, horizontal)")
    print(f"  final    mean={drift[-1].mean():.3f} m   max={drift[-1].max():.3f} m")
    print(f"  peak     mean={drift.max(axis=0).mean():.3f} m   "
          f"max={drift.max():.3f} m")
    print(f"  hold_err at t=0: mean={data['hold_err'][0].mean():.3f} m  "
          f"(this is the SPAWN offset, not a wave effect)")

    # ---- is the attitude channel physics or numerical chatter? -------------
    # A lightly damped stiff spring rings near its own natural frequency; if
    # that frequency is a large fraction of the physics rate, the "tilt rate"
    # channel is mostly integrator noise and must not be fed to a policy.
    dt = float(summary["control_dt_s"])
    for label, series in (("roll", data["roll_pitch"][:, :, 0]),
                          ("pitch", data["roll_pitch"][:, :, 1])):
        x = series - series.mean(axis=0, keepdims=True)
        # zero-crossing rate -> dominant frequency, robust and cheap
        crossings = (np.diff(np.signbit(x), axis=0) != 0).sum(axis=0)
        f_dom = crossings / (2.0 * x.shape[0] * dt)
        print(f"  {label:<6} dominant freq ~{f_dom.mean():.2f} Hz "
              f"(wave peak is {1.0 / max(tp) if tp else float('nan'):.2f} Hz, "
              f"control rate {1.0 / dt:.0f} Hz)")
    rate = np.abs(data["ang_vel_w"][:, :, :2])
    sat = float((rate > 9.9).mean())
    print(f"  angular-rate |p|,|q| above 9.9 rad/s: {sat * 100:.1f}% of samples "
          f"(a solver clamp at 10 rad/s would show up exactly here)")

    print()
    print("SUGGESTED FROZEN SCALES for the attitude observation block")
    print("  (p99 rounded up; 1% of steps saturate the clamp, 99% stay informative)")
    picks = {
        "roll/pitch": ("tilt_magnitude", 0.05),
        "roll/pitch rate": ("pitch_rate_q", 0.25),
        "vertical velocity": ("vert_vel", 0.1),
    }
    for name, (chan, step) in picks.items():
        for label, fn, unit, _s in CHANNELS:
            if label != chan:
                continue
            x = np.abs(np.asarray(fn(data), dtype=np.float64).ravel())
            val = _round_up(float(np.quantile(x, 0.99)), step)
            print(f"  {name:<20} -> {val:.3g} {unit}   (from {chan} p99)")
    print()
    print("NOTE: a zero-action trace is the PASSIVE-HULL envelope. A controlled")
    print("hull sees smaller excursions, so scales chosen here are conservative")
    print("(they will not saturate under control). Re-check against the champion")
    print("trace before freezing the contract.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for path in sys.argv[1:]:
        report(path)
