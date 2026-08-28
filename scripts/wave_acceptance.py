"""D2 physics acceptance probe for the JONSWAP wave task. ONE Isaac boot.

Boots --task (default the wave station-keeping id) with --num-envs envs and,
for each configured (Hs, Tp) pair, pins the per-episode JONSWAP draw to that
pair, runs --steps zero-action control steps, and reports:

* measured 4*sigma of the free-surface elevation vs nominal Hs,
* the spectral peak location (Hann-windowed FFT, averaged over envs) vs 1/Tp,
* whether 1/Tp is representable inside the component band [f_min, f_max]
  (since the 2026-08-26 band fix SeaStateCfg.f_max defaults to 1.60 Hz, so
  every sanctioned Tp >= 1.5 s peak is in-band; the flag remains live for
  overridden cfgs),
* a PRE-NORMALISATION ANALYTIC-SPECTRUM CAPTURE gate per (Hs, Tp) pair:
  captured = sum(S_analytic(f_i) * df) over the component grid, divided by
  the analytic total m0 (high-resolution quadrature), FAIL below 95%, taken
  at the worst-case gamma of the cfg's gamma_range.  This is computed from
  the ANALYTIC spectrum, NEVER from the renormalised amplitudes: the Hs
  renormalisation nearly cancels truncation (measured scale factor 1.024 at
  Tp=2.25/gamma=3.3 on the old 0.50 Hz band while 37% of the analytic
  variance was missing), so an amplitude-based check can never detect it,
* an aliasing sanity check of the component band against the control dt,
* the SLOPE-CHANNEL DISCLOSURE (roll/pitch moment integrand ~ f^4 * S(f)
  does not converge with f_max; the band extension roughly doubles slope RMS
  at unchanged labels, so pre/post-band moment channels are not comparable),
* hull heave/roll/pitch response statistics, and the 1:10 Froude table.

Elevation source: the wave physics (tasks/_shared/sea_state.py, spectrum from
Yutong's jonswap_wave.py) exposes elevation directly - the env holds a
SeaState at StationKeepingEnv._sea and the probe samples _sea.elevation(xy, t)
at the FIXED env origins with t = k * control_step_s, the same closed-form
field and clock the hull forcing integrates. Hull heave/roll/pitch are
recorded as secondary response PROXIES, not as the elevation measurement.

Extra cfg overrides use the eval_imbalance --set FIELD=VALUE pattern, extended
with dotted paths and tuples for the nested sea-state block, e.g.
  --set sea_state.gamma_range=3.3,3.3   --set sea_state.f_max=0.8

Example:
  python scripts/wave_acceptance.py --num-envs 16 --steps 3000 \
    --pairs 0.30:1.5,0.45:2.25,0.60:3.0 --out wave_d2.json --headless
"""
import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

FROUDE_SCALE = 10.0  # geometric scale 1:10


def _parse_pairs(text):
    """Parse "HS:TP,HS:TP" into [(hs, tp), ...]; raise ValueError if bad."""
    pairs = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        head, sep, tail = token.partition(":")
        if not sep:
            raise ValueError(f"pair {token!r} is not HS:TP")
        hs, tp = float(head), float(tail)
        if hs <= 0.0 or tp <= 0.0:
            raise ValueError(f"pair {token!r} must have positive Hs and Tp")
        pairs.append((hs, tp))
    if not pairs:
        raise ValueError("no pairs given")
    return pairs


def _froude_rows(pairs):
    """Model-scale -> full-scale rows at 1:10 (time and speed x sqrt(10))."""
    root = math.sqrt(FROUDE_SCALE)
    return [
        {
            "model_hs_m": hs,
            "model_tp_s": tp,
            "full_hs_m": hs * FROUDE_SCALE,
            "full_tp_s": tp * root,
        }
        for hs, tp in pairs
    ]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task",
                    default="Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=3000,
                    help="zero-action control steps recorded per (Hs, Tp) pair")
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument("--pairs", default=None,
                    help="comma list HS:TP, e.g. 0.30:1.5,0.60:3.0; default "
                         "= lo/mid/hi corners of the cfg (hs_range, tp_range)")
parser.add_argument("--out", default=None, help="JSON report")
parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="cfg overrides (repeatable); dotted paths and comma "
                         "tuples allowed, e.g. --set sea_state.f_max=0.8")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs <= 0:
    parser.error("--num-envs must be positive")
if args_cli.steps < 256:
    parser.error("--steps must be >= 256 (FFT needs a real window)")
cli_pairs = None
if args_cli.pairs is not None:
    try:
        cli_pairs = _parse_pairs(args_cli.pairs)
    except ValueError as exc:
        parser.error(f"--pairs: {exc}")

sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _apply_set(env_cfg, assignment):
    """eval_imbalance --set pattern, extended: dotted paths + tuple values."""
    field, sep, raw = assignment.partition("=")
    if not sep:
        raise SystemExit(f"--set {assignment!r} is not FIELD=VALUE")
    target = env_cfg
    parts = field.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise SystemExit(f"--set target {field!r} is not a cfg field")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise SystemExit(f"--set target {field!r} is not a cfg field")
    current = getattr(target, leaf)
    if isinstance(current, bool):
        value = raw.lower() in ("1", "true", "yes")
    elif isinstance(current, tuple) or isinstance(current, list):
        value = tuple(float(item) for item in raw.split(","))
    elif isinstance(current, (int, float)):
        value = type(current)(raw)
    else:
        value = raw
    setattr(target, leaf, value)
    print(f"set {field}={getattr(target, leaf)}", flush=True)


TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.eval_seed
for assignment in args_cli.extra_sets:
    _apply_set(env_cfg, assignment)

env = gym.make(TASK, cfg=env_cfg, render_mode=None)
base = env.unwrapped
sea = getattr(base, "_sea", None)
if sea is None:
    raise SystemExit(
        f"task {TASK!r} has no enabled SeaState (env attribute _sea is None); "
        "the D2 probe needs cfg.sea_state.enable=True - use the wave gym id "
        "or --set sea_state.enable=True")

# F5: the analytic-capture gate reuses the SAME sea-state module the env
# built its field from (single source of truth for the JONSWAP formula and
# the capture quadrature; CPU-tested in tasks/_shared/test_wave_fix_package
# .py).  sys.modules holds it regardless of the package path Isaac used.
sea_state_mod = sys.modules[type(sea).__module__]
CAPTURE_THRESHOLD = 0.95

SLOPE_DISCLOSURE = (
    "SLOPE-CHANNEL DISCLOSURE: the roll/pitch moment forcing integrand "
    "~ f^4 * S(f) does not converge with f_max; the 2026-08-26 band "
    "extension (0.50 -> 1.60 Hz) roughly DOUBLES surface-slope RMS at "
    "unchanged (Hs, Tp) labels (measured 0.088 -> 0.178 at Hs=0.45, "
    "Tp=2.25, gamma=3.3). Physics decision, not a bug: roll/pitch moment "
    "channels recorded before and after the band change are NOT comparable.")

control_dt = float(base.control_step_s)
fs_hz = 1.0 / control_dt
nyquist_hz = fs_hz / 2.0
f_min = float(sea.cfg.f_min)
f_max = float(sea.cfg.f_max)
n_steps = int(args_cli.steps)
window_s = n_steps * control_dt
df_fft = 1.0 / window_s
num_envs = int(base.num_envs)
all_ids = torch.arange(num_envs, device=base.device)
fixed_xy = base.scene.env_origins[:, :2].clone()
try:
    action_dim = int(base.cfg.action_space)  # station family: plain int (2)
except (TypeError, ValueError):
    action_dim = int(base.single_action_space.shape[-1])
zero_actions = torch.zeros((num_envs, action_dim), device=base.device)

if cli_pairs is not None:
    pairs = cli_pairs
else:
    hs_lo, hs_hi = (float(v) for v in sea.cfg.hs_range)
    tp_lo, tp_hi = (float(v) for v in sea.cfg.tp_range)
    pairs = [
        (hs_lo, tp_lo),
        ((hs_lo + hs_hi) / 2.0, (tp_lo + tp_hi) / 2.0),
        (hs_hi, tp_hi),
    ]

print(f"D2 WAVE ACCEPTANCE task={TASK} seed={args_cli.eval_seed} "
      f"envs={num_envs} steps={n_steps} window={window_s:.1f}s")
print(f"  control_dt={control_dt:.6f}s (fs={fs_hz:.1f} Hz, "
      f"Nyquist={nyquist_hz:.1f} Hz)  FFT df={df_fft:.4f} Hz")
print(f"  component band: [{f_min:.3f}, {f_max:.3f}] Hz "
      f"({sea.cfg.n_components} components)  gamma_range={sea.cfg.gamma_range}"
      " (gamma stays cfg-random; the 4*sigma identity is gamma-independent)")
alias_ok = f_max <= 0.25 * fs_hz
print(f"  aliasing check: f_max={f_max:.3f} Hz vs fs/4={fs_hz / 4.0:.2f} Hz "
      f"-> {'PASS' if alias_ok else 'FAIL'} "
      f"(shortest component sampled {fs_hz / f_max:.0f}x per period; forcing "
      "is zero-order-held over the decimation substeps)")
if n_steps >= int(base.max_episode_length):
    print(f"  WARN: steps >= max_episode_length ({int(base.max_episode_length)}); "
          "the fixed-horizon auto-reset will redraw phases mid-window (Hs/Tp "
          "stay pinned) and smear the spectrum slightly")

print("  " + SLOPE_DISCLOSURE)
print("  Froude scaling 1:10 -> length x10, time x sqrt(10)=3.162, "
      "speed x sqrt(10)=3.162")
print("    model Hs (m)  model Tp (s)  full Hs (m)  full Tp (s)")
for row in _froude_rows(pairs):
    print(f"    {row['model_hs_m']:12.2f}  {row['model_tp_s']:12.2f}  "
          f"{row['full_hs_m']:11.1f}  {row['full_tp_s']:11.2f}")

report_pairs = []
overall_pass = alias_ok
for hs_nom, tp_nom in pairs:
    env.reset()
    # Pin the per-episode draw AFTER the reset resample, then redraw so every
    # env carries exactly (hs_nom, tp_nom) with independent phases/directions.
    sea.cfg.hs_range = (hs_nom, hs_nom)
    sea.cfg.tp_range = (tp_nom, tp_nom)
    sea.resample(all_ids)
    pinned_hs = float(sea.hs.mean())
    pinned_tp = float(sea.tp.mean())

    eta = torch.zeros((n_steps, num_envs), device=base.device)
    heave = torch.zeros((n_steps, num_envs), device=base.device)
    roll = torch.zeros((n_steps, num_envs), device=base.device)
    pitch = torch.zeros((n_steps, num_envs), device=base.device)
    # Plain stepping, no grad/inference context: the pattern the task's own
    # runtime smoke (tasks/station_keeping/smoke.py) certifies.
    for k in range(n_steps):
        # Same clock the forcing uses: episode_length_buf is k during
        # step k, and _apply_action evaluates the field at k*control_dt.
        t = k * control_dt
        eta[k] = sea.elevation(fixed_xy, t)
        env.step(zero_actions)
        heave[k] = base.robot.data.root_pos_w[:, 2]
        quat = base.robot.data.root_link_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll[k] = torch.atan2(
            2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch[k] = torch.asin(
            torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))

    eta_np = eta.cpu().numpy()
    eta_np = eta_np - eta_np.mean(axis=0, keepdims=True)
    hs_meas = 4.0 * eta_np.std(axis=0)
    hs_mean = float(hs_meas.mean())
    hs_std = float(hs_meas.std())
    hs_ratio = hs_mean / hs_nom
    hs_ok = 0.9 <= hs_ratio <= 1.1

    window = np.hanning(n_steps)[:, None]
    spectrum = np.abs(np.fft.rfft(eta_np * window, axis=0)) ** 2
    psd_mean = spectrum.mean(axis=1)
    freqs = np.fft.rfftfreq(n_steps, d=control_dt)
    searchable = freqs >= 0.5 * f_min
    peak_index = int(np.argmax(np.where(searchable, psd_mean, 0.0)))
    f_peak = float(freqs[peak_index])
    fp_nom = 1.0 / tp_nom

    # F5 analytic-spectrum capture gate, PRE-normalisation.  gamma is not
    # pinned by the probe (the 4*sigma identity is gamma-independent), so
    # gate on the worst case over the cfg's gamma_range; capture grows with
    # gamma (peak enhancement concentrates energy inside the band), but both
    # ends are evaluated rather than assumed.
    gamma_lo, gamma_hi = (float(v) for v in sea.cfg.gamma_range)
    capture_by_gamma = {
        g: float(sea_state_mod.analytic_spectrum_capture(
            int(sea.cfg.n_components), f_min, f_max, hs_nom, tp_nom, g))
        for g in sorted({gamma_lo, gamma_hi})
    }
    capture_min = min(capture_by_gamma.values())
    capture_ok = capture_min >= CAPTURE_THRESHOLD
    overall_pass = overall_pass and capture_ok

    representable = f_min <= fp_nom <= f_max
    peak_tol = max(2.0 * df_fft, 0.15 * fp_nom)
    peak_ok = abs(f_peak - fp_nom) <= peak_tol
    if representable:
        peak_verdict = "PASS" if peak_ok else "FAIL"
        overall_pass = overall_pass and peak_ok
    else:
        peak_verdict = ("NOT-REPRESENTABLE (nominal fp outside component "
                        f"band [{f_min:.3f}, {f_max:.3f}] Hz - the JONSWAP "
                        "peak is truncated and the measured peak sits at the "
                        "band edge, NOT at 1/Tp; the renormalised Hs still "
                        "holds but the sea is band-limited)")
    overall_pass = overall_pass and hs_ok

    heave_np = heave.cpu().numpy()
    heave_np = heave_np - heave_np.mean(axis=0, keepdims=True)
    heave_4sigma = float(4.0 * heave_np.std(axis=0).mean())
    roll_rms_deg = float(np.degrees(roll.cpu().numpy().std(axis=0).mean()))
    pitch_rms_deg = float(np.degrees(pitch.cpu().numpy().std(axis=0).mean()))

    print(f"PAIR Hs={hs_nom:.2f}m Tp={tp_nom:.2f}s "
          f"(pinned draw: Hs={pinned_hs:.3f} Tp={pinned_tp:.3f})")
    print(f"  Hs: 4*sigma measured={hs_mean:.3f}m +/- {hs_std:.3f} "
          f"vs nominal {hs_nom:.2f}m (ratio {hs_ratio:.3f}) -> "
          f"{'PASS' if hs_ok else 'FAIL'}")
    print(f"  peak: measured f_peak={f_peak:.3f} Hz vs nominal "
          f"1/Tp={fp_nom:.3f} Hz (tol {peak_tol:.3f}) -> {peak_verdict}")
    print("  analytic capture (pre-normalisation, vs analytic m0): "
          + ", ".join(f"gamma={g:g}: {c * 100.0:.2f}%"
                      for g, c in sorted(capture_by_gamma.items()))
          + f" (threshold {CAPTURE_THRESHOLD * 100.0:.0f}%) -> "
          f"{'PASS' if capture_ok else 'FAIL'}")
    print(f"  hull response proxies: heave 4*sigma={heave_4sigma:.3f}m  "
          f"roll RMS={roll_rms_deg:.2f}deg  pitch RMS={pitch_rms_deg:.2f}deg")

    keep = freqs <= min(2.0 * f_max, nyquist_hz)
    report_pairs.append({
        "hs_nominal_m": hs_nom,
        "tp_nominal_s": tp_nom,
        "pinned_hs_m": pinned_hs,
        "pinned_tp_s": pinned_tp,
        "hs_measured_4sigma_mean_m": hs_mean,
        "hs_measured_4sigma_std_m": hs_std,
        "hs_ratio": hs_ratio,
        "hs_pass": hs_ok,
        "fp_nominal_hz": fp_nom,
        "f_peak_measured_hz": f_peak,
        "peak_tolerance_hz": peak_tol,
        "peak_representable_in_band": representable,
        "peak_pass": (peak_ok if representable else None),
        "analytic_capture_by_gamma": {
            f"{g:g}": c for g, c in sorted(capture_by_gamma.items())
        },
        "analytic_capture_min": capture_min,
        "analytic_capture_threshold": CAPTURE_THRESHOLD,
        "analytic_capture_pass": capture_ok,
        "heave_4sigma_m": heave_4sigma,
        "roll_rms_deg": roll_rms_deg,
        "pitch_rms_deg": pitch_rms_deg,
        "psd_freqs_hz": [float(v) for v in freqs[keep]],
        "psd_mean": [float(v) for v in psd_mean[keep]],
    })

print(f"D2 ACCEPTANCE: {'PASS' if overall_pass else 'FAIL'} "
      "(Hs identity + representable peaks + aliasing + analytic capture "
      ">= 95%; NOT-REPRESENTABLE pairs are reported, not failed - decide "
      "their fate explicitly)")

if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({
            "task": TASK,
            "seed": args_cli.eval_seed,
            "num_envs": num_envs,
            "steps": n_steps,
            "control_dt_s": control_dt,
            "sample_rate_hz": fs_hz,
            "nyquist_hz": nyquist_hz,
            "fft_df_hz": df_fft,
            "component_band_hz": [f_min, f_max],
            "aliasing_pass": alias_ok,
            "analytic_capture_threshold": CAPTURE_THRESHOLD,
            "slope_channel_disclosure": SLOPE_DISCLOSURE,
            "elevation_source": "SeaState.elevation at fixed env origins",
            "froude_scale": FROUDE_SCALE,
            "froude_table": _froude_rows(pairs),
            "overall_pass": overall_pass,
            "pairs": report_pairs,
        }, f, indent=1)
    print(f"  report -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
