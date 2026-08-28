"""Per-step telemetry dump for one wave episode batch: WHICH channel breaks us.

Why this exists
---------------
Station keeping holds 100% in calm water and ~56% under Hs = 0.60 m waves, and
nothing in the certification record says why. Three mechanisms are consistent
with that drop and they call for three different fixes:

  1. the policy chases wave-frequency surge/sway it cannot predict;
  2. roll/pitch tilt misaligns thrust from heading, so commanded yaw and
     delivered yaw diverge;
  3. quadratic drag on (orbital - hull) velocity rectifies into a mean drift
     force that walks the boat out of the hold radius.

A success rate cannot separate them. Per-step telemetry can. Every certified
evaluator in this repo records one row per EPISODE, so this script is the
missing per-STEP instrument.

Deliberately NOT a certification tool: it writes an .npz, never a gcert JSON,
and it defaults to the dev eval seed (7) so its output can never be mistaken
for a headline number.

Cross-task checkpoints
----------------------
The station-keeping champion was trained on the calm 3-D observation while this
probe runs the 9-D wave task, so a Runner sized from the wave env cannot load
it at all. Passing --train-task turns on the same observation bridge
scripts/eval_obs_bridge.py certifies with: the model is sized to the train
task's native width, and every step's observation is scattered into the frozen
64-D superset and gathered back in the train task's column order. Omit the flag
(or give it the same id as --task) and nothing bridges -- the probe behaves
exactly as it did before the bridge existed.

What it records, per step, per environment
------------------------------------------
  pos_w        (3)  world position; z is the heave channel
  quat_w       (4)  world orientation, scalar-first
  roll_pitch   (2)  extracted tilt, radians -- the channel the POLICY CANNOT SEE
  lin_vel_w    (3)  world linear velocity
  ang_vel_w    (3)  world angular velocity; [0]=p [1]=q [2]=r
  body_uvr     (3)  body surge / sway / yaw-rate (what the policy DOES see)
  action       (A)  the commanded action
  wave_force   (3)  world-frame wave force actually applied this step
  wave_torque  (3)  world-frame wave torque actually applied this step
  wave_eta     (1)  surface elevation at the hull point
  wave_slope   (2)  surface slope at the hull point
  hold_err     (1)  horizontal distance to the hold point
  inside_hold  (1)  whether that distance is within the hold radius

Derived summaries printed at the end (all per-environment, then aggregated):
  * action-velocity coherence -- is the policy tracking the wave or fighting it
  * thrust-heading misalignment -- how far tilt rotates the thrust axis
  * mean horizontal wave impulse -- the rectified drift the boat must resist
  * tilt exceedance -- fraction of steps past a roll/pitch threshold

Usage
-----
    python scripts/wave_probe_dump.py \
        --task Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1 \
        --checkpoint C:/usvb/sk_champion.pt --train-task <native id> \
        --set sea_state.hs_range=0.60,0.60 \
        --set sea_state.tp_range=2.25,2.25 \
        --episodes 8 --num_envs 8 --headless --out probe_w600.npz

    # zero-action reference (what the sea alone does to a passive hull):
    python scripts/wave_probe_dump.py --task ... --zero-action ...
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# CPU-only contract import, deliberately ABOVE the isaaclab import: a task pair
# the frozen observation superset cannot bridge is refused before Isaac boots,
# so a mistyped --train-task costs a second rather than the minutes a headless
# launch takes. scripts/eval_obs_bridge.py:41 places it for the same reason.
from tasks._shared.obs_superset import (  # noqa: E402
    NATIVE_LAYOUTS,
    SUPERSET_DIM_V2,
    UnsupportedNativeLayout,
    extract_native,
    native_to_superset,
)

try:
    from cfg_override import apply_overrides
except ImportError:  # invoked from a cwd that is not scripts/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cfg_override import apply_overrides

# Either branch above leaves scripts/ importable, so the shared-column map is
# taken from scripts/obs_bridge.py rather than copied a third time: that is the
# same function embed_checkpoint resizes real checkpoints with, and a probe
# whose columns disagreed with it would be tracing a policy nobody ran. Its
# arguments read (src, dst) -- here the source is the EVAL task and the
# destination the TRAIN task, because the map is indexed by train-native
# column. Also CPU-only, so it stays above the isaaclab import.
from obs_bridge import _column_mapping  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="wave-enabled gym id")
parser.add_argument("--checkpoint", default=None,
                    help="policy checkpoint; omit with --zero-action")
parser.add_argument("--train-task", default=None,
                    help="registry id the checkpoint was trained on; defaults "
                         "to --task when the shapes already match. When it "
                         "names a DIFFERENT id the observation bridge turns "
                         "on: the model is sized to that task's native width "
                         "and every observation is remapped into its column "
                         "order, the same mechanism scripts/eval_obs_bridge.py "
                         "certifies with")
parser.add_argument("--cfg-entry-point", default="skrl_cfg_entry_point")
parser.add_argument("--zero-action", action="store_true",
                    help="drive zero action: the passive-hull reference that "
                         "says what the sea does with no control at all")
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=7,
                    help="DEV seed by default: this is a diagnostic, and its "
                         "numbers must never enter a certification table")
parser.add_argument("--stride", type=int, default=1,
                    help="record every Nth control step (1 = every step)")
parser.add_argument("--tilt-threshold-deg", type=float, default=15.0)
parser.add_argument("--out", default="wave_probe.npz")
parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="cfg overrides (repeatable); dotted paths reach "
                         "nested cfgs, e.g. --set sea_state.hs_range=0.6,0.6")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.zero_action and not args_cli.checkpoint:
    parser.error("give --checkpoint, or --zero-action for the passive reference")
if args_cli.episodes < 1 or args_cli.num_envs < 1:
    parser.error("--episodes and --num_envs must be >= 1")
if args_cli.stride < 1:
    parser.error("--stride must be >= 1")


# ---- observation bridge (ported from scripts/eval_obs_bridge.py) -----------
# The station-keeping champion was trained on the 3-D calm observation while
# this probe runs the 9-D wave task, so a Runner sized from the wave env cannot
# take the checkpoint at all ("Error(s) in loading state_dict for SharedModel")
# and the trace that was supposed to explain the wave drop never starts. The
# certified answer is the frozen superset: size the model to the TRAIN task's
# native width, and remap the eval task's native observation into the train
# task's column order before every policy call.


def _bridge_layout(gym_id, fail):
    """Resolve a supported native layout or refuse with the contract's story."""
    try:
        layout = NATIVE_LAYOUTS[gym_id]
    except KeyError:
        known = ", ".join(sorted(NATIVE_LAYOUTS))
        fail(f"{gym_id!r} has no frozen native layout in NATIVE_LAYOUTS; "
             f"known ids: {known}")
        raise
    if isinstance(layout, UnsupportedNativeLayout):
        fail(f"{gym_id!r} is explicitly unsupported by the observation "
             f"bridge: {layout.reason}")
        raise AssertionError("unreachable")
    return list(layout)


# The bridge is OFF unless --train-task names a different id: absent or equal,
# every line below runs exactly as it did before the bridge existed, which is
# what keeps the existing single-task traces reproducible. --zero-action loads
# no checkpoint at all, so it has nothing to bridge and already ignores
# --train-task further down; it keeps ignoring it here.
BRIDGE = (not args_cli.zero_action
          and args_cli.train_task is not None
          and args_cli.train_task != args_cli.task)
train_layout = eval_layout = mapping = None
bridge_summary = None
if BRIDGE:
    train_layout = _bridge_layout(args_cli.train_task, parser.error)
    eval_layout = _bridge_layout(args_cli.task, parser.error)
    mapping = _column_mapping(eval_layout, train_layout)
    shared = sum(index is not None for index in mapping)
    zero_filled = [slot for slot, index in zip(train_layout, mapping)
                   if index is None]
    dropped = [slot for slot in eval_layout if slot not in set(train_layout)]
    if shared == 0:
        parser.error(
            f"train task {args_cli.train_task!r} and eval task "
            f"{args_cli.task!r} share NO superset column: the bridge would "
            "feed the policy nothing but zeros. Refusing.")
    # Recorded into the .npz so a bridged trace can never be mistaken for a
    # native one months later: which columns the policy actually saw is part
    # of what the trace means.
    bridge_summary = {
        "train_layout": train_layout,
        "eval_layout": eval_layout,
        "mapping": mapping,
        "shared": shared,
        "zero_filled_superset_slots": zero_filled,
        "dropped_eval_superset_slots": dropped,
    }
    print(f"[probe] BRIDGE {args_cli.task} ({len(eval_layout)}D native) -> "
          f"{args_cli.train_task} ({len(train_layout)}D native)")
    print(f"  shared columns: {shared}/{len(train_layout)}  "
          f"zero-filled train channels: {len(zero_filled)}  "
          f"dropped eval channels: {len(dropped)}")
    print("  train_native  superset  eval_native")
    for train_native, (slot, eval_native) in enumerate(
            zip(train_layout, mapping)):
        source = "ZERO" if eval_native is None else str(eval_native)
        print(f"  {train_native:12d}  {slot:8d}  {source:>11}")
    if zero_filled:
        print(f"  WARNING: train channels at superset slots {zero_filled} are "
              "zero-filled every step; the policy never trained on zeros there "
              "unless those channels were rest-state zero in training.")

sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def _roll_pitch_from_quat(quat):
    """Return (roll, pitch) in radians from a scalar-first wxyz quaternion.

    Derived the same way the physics does it: world-up expressed in body
    coordinates, b = R^T * e3, which is the third ROW of R. Then

        roll  = atan2( b_y, b_z)
        pitch = atan2(-b_x, b_z)

    The signs are pinned by tasks/_shared/restoring.py, which builds the
    restoring torque as tau_x = -k_roll * b_y and tau_y = +k_pitch * b_x. A
    restoring torque must oppose the tilt it is correcting, so positive roll
    must carry positive b_y -- which fixes both signs above. Getting either
    one backwards would not crash anything; it would hand back a clean-looking
    trace that answers the roll-vs-drift question the wrong way round.

    Reading tilt off this vector (rather than off Euler angles in the world
    frame) is also what makes it yaw-invariant -- the property whose absence
    caused the July "capsize by turning" bug in the attitude spring.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    bx = 2.0 * (x * z - w * y)
    by = 2.0 * (y * z + w * x)
    bz = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(by, bz)
    pitch = torch.atan2(-bx, bz)
    return roll, pitch


def _body_uvr(quat, lin_vel_w, ang_vel_w):
    """Body surge/sway/yaw-rate -- the motion channels the policy DOES see."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Bow direction = first column of R, projected to the horizontal plane.
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    norm = torch.sqrt(fx * fx + fy * fy).clamp(min=1.0e-6)
    fx, fy = fx / norm, fy / norm
    vx, vy = lin_vel_w[:, 0], lin_vel_w[:, 1]
    surge = vx * fx + vy * fy
    sway = -vx * fy + vy * fx
    return torch.stack((surge, sway, ang_vel_w[:, 2]), dim=-1), torch.stack((fx, fy), dim=-1)


def _set_native_runner_space(env, native_dim: int):
    """Give Runner the checkpoint's input size without changing target output.

    Copied verbatim from scripts/eval_obs_bridge.py:188-220 (which took it
    from scripts/eval_cross_champion.py) so the two can be diffed line for
    line: the skrl Runner sizes its models from env.observation_space, so
    overriding the wrapper's space with the TRAIN task's native width lets the
    unchanged checkpoint load.
    """
    native_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(native_dim,), dtype=np.float32,
    )
    # skrl's IsaacLab wrapper exposes observation_space as a READ-ONLY property
    # recomputed from the underlying env on every access, so neither writing
    # _observation_space nor assigning the attribute has any effect (measured
    # while building the certified bridge: the Runner still built a 9-input
    # model for a 3-input checkpoint). The only robust override for a
    # class-level property is a dynamic subclass whose property returns the
    # native space.
    cls = type(env)
    patched = type(
        cls.__name__ + "Bridged",
        (cls,),
        {"observation_space": property(lambda self: native_space)},
    )
    env.__class__ = patched
    got = env.observation_space
    if tuple(got.shape) != (native_dim,):
        raise RuntimeError(
            f"observation-space override failed: wrapper still reports "
            f"{got.shape} instead of ({native_dim},)")
    # The override must live ONLY through Runner construction: skrl's wrapper
    # also uses observation_space to slice step() outputs, so leaving the patch
    # in place truncates the eval-native observation to the train width
    # (measured: the wave task "emitted 3 channels"). In this probe that would
    # be quiet and fatal at once -- the sea-state channels the trace exists to
    # explain would be the first ones cut. Caller restores.
    return cls


def _bridge_obs(observation):
    """Map one batch of eval-native observations into train-native order.

    scripts/eval_obs_bridge.py:223 with --eval-task spelled --task; the scatter
    zero-fills any train channel this task never emits, and drops any channel
    the train task never saw.
    """
    if observation.shape[-1] != len(eval_layout):
        raise RuntimeError(
            f"task {args_cli.task!r} emitted {observation.shape[-1]} channels "
            f"but NATIVE_LAYOUTS declares {len(eval_layout)}; the frozen "
            "contract and the env disagree - refusing to bridge a mislabeled "
            "observation.")
    superset = native_to_superset(
        observation, args_cli.task, dim=SUPERSET_DIM_V2)
    return extract_native(superset, args_cli.train_task)


TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.eval_seed
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
applied_overrides = dict(apply_overrides(env_cfg, args_cli.extra_sets))

env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
base = env.unwrapped
device = base.device

sea = getattr(base, "_sea", None)
if sea is None:
    raise SystemExit(
        f"{TASK} carries no sea state; this probe exists to explain a WAVE "
        f"drop, so refusing rather than dumping a calm-water trace"
    )

policy = None
if not args_cli.zero_action:
    from skrl.utils.runner.torch import Runner  # noqa: E402

    # The v4 squashed-Gaussian policy class is not in skrl's closed _component
    # whitelist, so a stock Runner cannot resolve it and dies at model
    # construction -- the same crash that once produced a bogus "below gate"
    # verdict on the first v4 queue. Both certified evaluators carry this hook
    # and the probe now carries it too. For a plain Gaussian config the
    # override never fires, so this changes nothing about the traces already
    # taken. It lives inside this branch because the Runner import is
    # deliberately lazy: a --zero-action reference run touches no policy
    # machinery at all.
    try:
        from tasks._shared.squashed_gaussian import squashed_gaussian_model
    except ImportError:
        from isaaclab_tasks.direct._shared.squashed_gaussian import (
            squashed_gaussian_model,
        )

    class SquashedRunner(Runner):
        def _component(self, name: str):
            if name.lower() == "squashedgaussianmixin":
                return squashed_gaussian_model
            return super()._component(name)

    train_task = args_cli.train_task or TASK
    experiment_cfg = load_cfg_from_registry(train_task, args_cli.cfg_entry_point)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    # skrl's agent base imports wandb during init unless this is switched off,
    # and this environment has no wandb. Both certified evaluators
    # (eval_v6_frozen.py, eval_obs_bridge.py) carry the same line; the probe's
    # first run died here without it.
    experiment_cfg["agent"]["experiment"]["wandb"] = False
    # With the bridge on, the models must be sized to the CHECKPOINT's input
    # width rather than the wave env's, or the state_dict simply does not fit.
    # Off, no subclass is installed and the Runner sees the env exactly as
    # before.
    orig_wrapper_cls = (_set_native_runner_space(wrapped, len(train_layout))
                        if BRIDGE else None)
    runner = SquashedRunner(wrapped, experiment_cfg)
    if orig_wrapper_cls is not None:
        # Models are sized; restore the wrapper here and nowhere later. skrl's
        # wrapper also slices step() output by observation_space, so a patch
        # left in place would hand the policy a 3-channel stump of the 9-D wave
        # observation every step -- no crash, just a trace of a policy nobody
        # ran. eval_obs_bridge.py:254 reverts at exactly this point.
        wrapped.__class__ = orig_wrapper_cls
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")
    policy = runner.agent

hold_radius = float(getattr(base.cfg, "hold_radius", 2.0))
control_dt = float(base.cfg.sim.dt) * int(getattr(base.cfg, "decimation", 1))
max_steps = int(base.max_episode_length)

rows = {k: [] for k in (
    "pos_w", "quat_w", "roll_pitch", "lin_vel_w", "ang_vel_w", "body_uvr",
    "action", "wave_force", "wave_torque", "wave_eta", "wave_slope",
    "hold_err", "inside_hold", "step_index",
)}

obs, _ = wrapped.reset()
if BRIDGE:
    obs = _bridge_obs(obs)
print(f"[probe] task={TASK} hold_radius={hold_radius} m  dt={control_dt:.5f} s  "
      f"max_steps={max_steps}  envs={args_cli.num_envs}", flush=True)
print(f"[probe] sea: Hs={sea.hs[:4].tolist()} Tp={sea.tp[:4].tolist()}", flush=True)

with torch.inference_mode():
    for step in range(max_steps):
        if policy is None:
            action = torch.zeros(
                (args_cli.num_envs, base.cfg.action_space), device=device
            )
        else:
            # Deterministic eval action, extracted exactly the way
            # scripts/eval_v6_frozen.py does it so the probe drives the hull
            # with the same policy output the certification measures.
            outputs = policy.act(obs, timestep=0, timesteps=0)
            action = outputs[-1].get("mean_actions", outputs[0])

        if step % args_cli.stride == 0:
            quat = base.robot.data.root_link_quat_w
            pos = base.robot.data.root_pos_w
            # root_com_vel_w is the 6-vector every task in this repo reads:
            # [:, :3] linear, [:, 3:] angular (yaw rate is [:, 5]).
            vel6 = base.robot.data.root_com_vel_w
            lin = vel6[:, :3]
            ang = vel6[:, 3:]
            roll, pitch = _roll_pitch_from_quat(quat)
            uvr, _fwd = _body_uvr(quat, lin, ang)

            # Ask the sea for exactly what it is about to apply, at the hull
            # point and the CURRENT sim time -- not a re-derivation.
            t = float(base.episode_length_buf[0]) * control_dt
            xy = pos[:, :2]
            wf, wt = sea.forces(xy, lin[:, :2], t)
            eta = sea.elevation(xy, t)
            slope = sea.surface_slope(xy, t)

            hold_point = getattr(base, "hold_point", None)
            if hold_point is not None:
                err = torch.norm(hold_point - xy, dim=-1)
            else:
                err = torch.full((args_cli.num_envs,), float("nan"), device=device)

            rows["pos_w"].append(pos.cpu().numpy())
            rows["quat_w"].append(quat.cpu().numpy())
            rows["roll_pitch"].append(
                torch.stack((roll, pitch), dim=-1).cpu().numpy())
            rows["lin_vel_w"].append(lin.cpu().numpy())
            rows["ang_vel_w"].append(ang.cpu().numpy())
            rows["body_uvr"].append(uvr.cpu().numpy())
            rows["action"].append(action.cpu().numpy())
            rows["wave_force"].append(wf.cpu().numpy())
            rows["wave_torque"].append(wt.cpu().numpy())
            rows["wave_eta"].append(eta.reshape(-1).cpu().numpy())
            rows["wave_slope"].append(slope.cpu().numpy())
            rows["hold_err"].append(err.cpu().numpy())
            rows["inside_hold"].append((err <= hold_radius).cpu().numpy())
            rows["step_index"].append(
                np.full((args_cli.num_envs,), step, dtype=np.int32))

        obs, _rew, _term, _trunc, _info = wrapped.step(action)
        if BRIDGE:
            # Remap before the NEXT policy call, never before the telemetry
            # above: every recorded channel is read off the env, not off obs.
            obs = _bridge_obs(obs)

arrays = {k: np.stack(v, axis=0) for k, v in rows.items()}

# ---- derived summaries ----------------------------------------------------
roll_pitch = arrays["roll_pitch"]
tilt = np.sqrt((roll_pitch ** 2).sum(axis=-1))
thresh = np.deg2rad(args_cli.tilt_threshold_deg)
wave_force_xy = arrays["wave_force"][:, :, :2]
mean_impulse = wave_force_xy.mean(axis=0) * control_dt * arrays["pos_w"].shape[0]
surge = arrays["body_uvr"][:, :, 0]
thrust_cmd = arrays["action"][:, :, 0]


def _coherence(a, b):
    """Zero-lag correlation per environment, NaN-safe."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=0) / denom, np.nan)


summary = {
    "task": TASK,
    "eval_seed": args_cli.eval_seed,
    "zero_action": bool(args_cli.zero_action),
    "checkpoint": args_cli.checkpoint,
    # Null on a native run, the full column map on a bridged one -- the reader
    # of an .npz has to be able to tell which observation the policy was fed.
    "train_task": args_cli.train_task,
    "bridge": bridge_summary,
    "applied_overrides": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in applied_overrides.items()},
    "hold_radius_m": hold_radius,
    "control_dt_s": control_dt,
    "recorded_steps": int(arrays["pos_w"].shape[0]),
    "num_envs": int(args_cli.num_envs),
    "hs_m": sea.hs.cpu().numpy().tolist(),
    "tp_s": sea.tp.cpu().numpy().tolist(),
    "tilt_deg_rms": float(np.rad2deg(np.sqrt((tilt ** 2).mean()))),
    "tilt_deg_p99": float(np.rad2deg(np.quantile(tilt, 0.99))),
    "tilt_exceedance_frac": float((tilt > thresh).mean()),
    "tilt_threshold_deg": args_cli.tilt_threshold_deg,
    "heave_rms_m": float(np.sqrt((arrays["pos_w"][:, :, 2] -
                                  arrays["pos_w"][:, :, 2].mean()) ** 2).mean()),
    "wave_force_xy_rms_N": float(np.sqrt((wave_force_xy ** 2).sum(-1).mean())),
    "wave_force_xy_mean_N": wave_force_xy.mean(axis=(0, 1)).tolist(),
    "mean_horizontal_impulse_Ns": mean_impulse.mean(axis=0).tolist(),
    "wave_torque_rms_Nm": float(np.sqrt((arrays["wave_torque"][:, :, :2] ** 2)
                                        .sum(-1).mean())),
    "hold_err_rms_m": float(np.sqrt((arrays["hold_err"] ** 2).mean())),
    "inside_hold_frac": float(arrays["inside_hold"].mean()),
    "coherence_thrust_vs_surge": np.nanmean(_coherence(thrust_cmd, surge)).item(),
    "coherence_thrust_vs_wavefx": np.nanmean(
        _coherence(thrust_cmd, wave_force_xy[:, :, 0])).item(),
    "coherence_holderr_vs_tilt": np.nanmean(_coherence(arrays["hold_err"], tilt)).item(),
}

np.savez_compressed(args_cli.out, summary=json.dumps(summary), **arrays)
print(json.dumps(summary, indent=2), flush=True)
print(f"[probe] wrote {args_cli.out}", flush=True)

env.close()
app.close()
