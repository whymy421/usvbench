"""Frozen-level eval of a task-A checkpoint INSIDE task B via the obs bridge.

eval_v6_frozen variant (same per-episode record schema, same paired layout
stream) that evaluates a checkpoint trained on --train-task while the episodes
run on --eval-task. Before every policy call the eval task's native
observation is mapped into the train task's native layout through the frozen
observation-superset contract (tasks/_shared/obs_superset.py):

    eval native -> scatter into the 64-D superset -> gather train native

Channels the train task never saw are dropped; channels the train task expects
but the eval task does not emit are zero-filled (the superset scatter zeroes
them). This is the SAME mechanism scripts/obs_bridge.py uses to embed
checkpoints and scripts/eval_cross_champion.py used to carry a champion onto
superset-emitting targets zero-shot; here it runs per-step on the eval task's
NATIVE observation, so the target env needs no emit_superset_obs support.

The script refuses (exit 2, before Isaac boots) when either Gym id has no
frozen native layout, when a layout is explicitly unsupported (pooled), or
when the two layouts share no superset column at all.

Example (calm station champion on the wave task):
  python scripts/eval_obs_bridge.py \
    --checkpoint tasks/station_keeping/checkpoints/sk_blueboat_s42.pt \
    --train-task Isaac-USV-StationKeep-BlueBoat-Direct-v1 \
    --eval-task Isaac-USV-StationKeep-BlueBoat-Wave-Direct-v1 \
    --episodes 128 --eval-seed 42 --level 0 --out bridge_sk_wave_e42.json \
    --headless
"""
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# CPU-only contract import: must stay ABOVE the isaaclab import so the refusal
# path never needs Isaac and the stubbed argparse tests can exercise it.
from tasks._shared.obs_superset import (  # noqa: E402
    NATIVE_LAYOUTS,
    SUPERSET_DIM_V2,
    UnsupportedNativeLayout,
    extract_native,
    native_to_superset,
)

# Shared --set FIELD=VALUE parsing (scripts/cfg_override.py). Dotted paths
# reach nested cfgs, which the wave ladder needs: its rungs live at
# cfg.sea_state.hs_range, one level below anything a top-level-only override
# could touch. Also CPU-only, so it stays above the isaaclab import.
try:
    from cfg_override import apply_overrides  # noqa: E402
except ImportError:  # invoked from a cwd that is not scripts/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cfg_override import apply_overrides  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--train-task", required=True,
                    help="Gym id the checkpoint was trained on (task A)")
parser.add_argument("--eval-task", required=True,
                    help="Gym id the episodes run on (task B)")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
# Off-policy checkpoints (SAC/TD3) carry different model roles than the
# PPO SharedModel, so the runner must be built from THEIR config or the
# state_dict load fails on missing policy_layer/value_layer keys. The config
# is looked up on the TRAIN task: the architecture must match the checkpoint.
parser.add_argument("--cfg-entry-point", default="skrl_cfg_entry_point",
                    help="registry key for the skrl config on the TRAIN task, "
                         "e.g. skrl_sac_cfg_entry_point")
parser.add_argument("--out", default=None, help="JSON per-episode records")
parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="extra EVAL-task cfg overrides (repeatable); dotted "
                         "paths reach nested cfgs and comma tuples pin ranges, "
                         "e.g. --set current_speed_override_mps=2.25 or one "
                         "wave rung --set sea_state.hs_range=0.6,0.6")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


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


def _bridge_mapping(train_layout, eval_layout):
    """For each train-native index: the eval-native index sharing its
    superset slot, or None (zero-fill). Mirrors obs_bridge._column_mapping."""
    eval_native_by_slot = {slot: i for i, slot in enumerate(eval_layout)}
    return [eval_native_by_slot.get(slot) for slot in train_layout]


train_layout = _bridge_layout(args_cli.train_task, parser.error)
eval_layout = _bridge_layout(args_cli.eval_task, parser.error)
mapping = _bridge_mapping(train_layout, eval_layout)
shared = sum(index is not None for index in mapping)
zero_filled = [slot for slot, index in zip(train_layout, mapping)
               if index is None]
if shared == 0:
    parser.error(
        f"train task {args_cli.train_task!r} and eval task "
        f"{args_cli.eval_task!r} share NO superset column: the bridge would "
        "feed the policy nothing but zeros. Refusing.")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive")
checkpoint_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(checkpoint_path):
    parser.error(f"--checkpoint does not exist: {checkpoint_path}")

dropped = [slot for slot in eval_layout if slot not in set(train_layout)]
print(f"BRIDGE {args_cli.eval_task} ({len(eval_layout)}D native) -> "
      f"{args_cli.train_task} ({len(train_layout)}D native)")
print(f"  shared columns: {shared}/{len(train_layout)}  "
      f"zero-filled train channels: {len(zero_filled)}  "
      f"dropped eval channels: {len(dropped)}")
print("  train_native  superset  eval_native")
for train_native, (slot, eval_native) in enumerate(zip(train_layout, mapping)):
    source = "ZERO" if eval_native is None else str(eval_native)
    print(f"  {train_native:12d}  {slot:8d}  {source:>11}")
if zero_filled:
    print(f"  WARNING: train channels at superset slots {zero_filled} are "
          "zero-filled every step; the policy never trained on zeros there "
          "unless those channels were rest-state zero in training.")

sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from skrl.utils.runner.torch import Runner  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

# Scenario-protocol stamping, on the same dual import scripts/
# eval_v6_frozen.py:74-80 uses so the script keeps working from the repo and
# from the deployed task tree.  _REPO_ROOT is already on sys.path (module top).
try:
    from tasks._shared.scenario_draws import (  # noqa: E402
        episode_scenario_hashes_for,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )
except ImportError:
    from isaaclab_tasks.direct._shared.scenario_draws import (  # noqa: E402
        episode_scenario_hashes_for,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )

# ---- squashed-policy hook (mirrors scripts/eval_v6_frozen.py) --------------
try:
    from tasks._shared.squashed_gaussian import squashed_gaussian_model
except ImportError:
    from isaaclab_tasks.direct._shared.squashed_gaussian import squashed_gaussian_model


class SquashedRunner(Runner):
    def _component(self, name: str):
        if name.lower() == "squashedgaussianmixin":
            return squashed_gaussian_model
        return super()._component(name)
# ---------------------------------------------------------------------------


def _set_native_runner_space(env, native_dim: int):
    """Give Runner the checkpoint's input size without changing target output.

    Reused from scripts/eval_cross_champion.py: the skrl Runner sizes its
    models from env.observation_space, so overriding the wrapper's space with
    the TRAIN task's native width lets the unchanged checkpoint load.
    """
    native_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(native_dim,), dtype=np.float32,
    )
    # skrl's IsaacLab wrapper exposes observation_space as a READ-ONLY property
    # recomputed from the underlying env on every access, so neither writing
    # _observation_space nor assigning the attribute has any effect (measured:
    # the Runner still built a 9-input model for a 3-input checkpoint). The
    # only robust override for a class-level property is a dynamic subclass
    # whose property returns the native space.
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
    # also uses observation_space to slice step() outputs, so leaving the
    # patch in place truncates the eval-native observation to the train width
    # (measured: the wave task "emitted 3 channels"). Caller restores.
    return cls


def _bridge_obs(observation: torch.Tensor) -> torch.Tensor:
    """Map one batch of eval-native observations into train-native order."""
    if observation.shape[-1] != len(eval_layout):
        raise RuntimeError(
            f"eval task {args_cli.eval_task!r} emitted "
            f"{observation.shape[-1]} channels but NATIVE_LAYOUTS declares "
            f"{len(eval_layout)}; the frozen contract and the env disagree - "
            "refusing to bridge a mislabeled observation.")
    superset = native_to_superset(
        observation, args_cli.eval_task, dim=SUPERSET_DIM_V2)
    return extract_native(superset, args_cli.train_task)


TASK = args_cli.eval_task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env_cfg.seed = args_cli.eval_seed
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
applied_overrides = dict(apply_overrides(env_cfg, args_cli.extra_sets))
experiment_cfg = load_cfg_from_registry(
    args_cli.train_task, args_cli.cfg_entry_point)
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
_orig_wrapper_cls = _set_native_runner_space(wrapped, len(train_layout))
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False
runner = SquashedRunner(wrapped, experiment_cfg)
# Models are sized; restore the wrapper so step() returns eval-native width.
wrapped.__class__ = _orig_wrapper_cls
runner.agent.load(checkpoint_path)
runner.agent.set_running_mode("eval")

base = env.unwrapped
# Does THIS env carry the scenario protocol?  Same guard, same marker and same
# warning as scripts/eval_v6_frozen.py:148-162, so this certificate and a
# frozen-eval certificate at the same --eval-seed can be scenario-paired.
scenario_protocol = scenario_protocol_stamp(base)
_notice = scenario_protocol_notice(TASK, scenario_protocol)
if _notice:
    print(_notice, flush=True)
obs, _ = wrapped.reset()
obs = _bridge_obs(obs)
records = []
ep_counter = torch.zeros(base.num_envs, dtype=torch.long)
max_steps = (args_cli.episodes // base.num_envs + 3) * base.max_episode_length
step = 0
while len(records) < args_cli.episodes and step < max_steps:
    with torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    # d0_per_env is rewritten inside _reset_idx, which runs during step(), so
    # reading it alongside the other episode stats would report the NEXT
    # episode's straight-line distance. Snapshot it while it still belongs to
    # the episode that is about to end.
    d0_prev = (base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None)
    obs, _, term, trunc, _ = wrapped.step(actions)
    obs = _bridge_obs(obs)
    step += 1
    done = term | trunc
    done = done.squeeze(-1) if done.dim() > 1 else done
    ids = torch.nonzero(done).flatten()
    for i in ids.tolist():
        episode_contact_steps = getattr(base, "episode_contact_steps", None)
        episode_contact_longest_steps = getattr(
            base, "episode_contact_longest_steps", None
        )
        episode_contact_depth_sum = getattr(
            base, "episode_contact_depth_sum", None
        )
        episode_max_phase = getattr(base, "episode_max_phase", None)
        rec = {
            "env": i,
            "ep": int(ep_counter[i]),
            "success": bool(base.episode_success[i]),
            "tts_s": (None if math.isnan(float(base.time_to_success[i]))
                      else float(base.time_to_success[i])),
            "path_length_m": float(base.episode_path_length[i]),
        }
        # Longest continuous hold, the graded quantity the binary success
        # criterion thresholds (station_keeping_env.py: _success is
        # _max_hold_steps >= required_hold_steps). Defined for every episode,
        # so a dose response can be read off it without conditioning on the
        # outcome -- conditioning on "both cells succeeded" was shown to flip
        # the SIGN of a p~1e-19 result on the deadband-PID wave ladder.
        if hasattr(base, "episode_max_hold_s"):
            rec["max_hold_s"] = float(base.episode_max_hold_s[i])
        # Station keeping and path following have no obstacle field, hence no
        # clearance buffer; every hazard/docking family records it.
        if hasattr(base, "episode_min_clearance"):
            rec["min_clearance_m"] = float(base.episode_min_clearance[i])
        # Cross-track error, mirrored from scripts/eval_v6_frozen.py so every
        # controller writes the same record schema (the parity tests enforce it).
        if hasattr(base, "episode_xte_rms"):
            rec["xte_rms_m"] = float(base.episode_xte_rms[i])
        # Per-primitive digests of the scenario the FINISHED episode ran,
        # latched at reset exactly like episode_min_clearance.  Absent on
        # families not yet on the scenario protocol, and empty until an env has
        # completed its first episode; both cases are guarded inside the
        # helper, which returns None for "write no field".
        scenario_hashes = episode_scenario_hashes_for(base, i)
        if scenario_hashes is not None:
            rec["scenario_hashes"] = scenario_hashes
        if (
            episode_contact_steps is not None
            and episode_contact_longest_steps is not None
            and episode_contact_depth_sum is not None
        ):
            contact_steps = float(episode_contact_steps[i])
            rec["contact_steps"] = int(contact_steps)
            rec["contact_seconds"] = contact_steps * base.control_step_s
            rec["contact_longest_seconds"] = (
                float(episode_contact_longest_steps[i]) * base.control_step_s
            )
            rec["contact_depth_mean_m"] = (
                float(episode_contact_depth_sum[i]) / contact_steps
                if contact_steps > 0.0
                else 0.0
            )
        if episode_max_phase is not None:
            rec["max_phase"] = int(episode_max_phase[i])
        if hasattr(base, "episode_gates_passed"):
            rec["gates"] = int(base.episode_gates_passed[i])
        if d0_prev is not None:
            rec["d0_m"] = float(d0_prev[i])
        # The shortest feasible route for THIS layout. The env has computed and
        # latched it per episode since the geodesic reward work; it was simply
        # never written out, so no past certification can be rescored against
        # an oracle. Every future one can.
        if hasattr(base, "route_geodesic_length"):
            geodesic = float(base.route_geodesic_length[i])
            if geodesic > 0.0:
                rec["route_geodesic_m"] = geodesic
        records.append(rec)
        ep_counter[i] += 1

records = records[: args_cli.episodes]
n = len(records)
succ = [r for r in records if r["success"]]
tts = sorted(r["tts_s"] for r in succ if r["tts_s"] is not None)
clr = sorted(r["min_clearance_m"] for r in records if "min_clearance_m" in r)
collided = sum(1 for c in clr if c < 0.0)


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[k]


sr = len(succ) / max(n, 1)
print(f"EVAL task={TASK} level={args_cli.level} seed={args_cli.eval_seed} "
      f"ckpt={os.path.basename(args_cli.checkpoint)}")
print(f"  bridge: trained_on={args_cli.train_task} "
      f"({len(train_layout)}D) evaluated_on={args_cli.eval_task} "
      f"({len(eval_layout)}D) shared={shared} zero_filled={len(zero_filled)}")
print(f"  episodes={n} SR={sr:.4f} ({len(succ)}/{n})")
print(f"  tts median={pct(tts, 0.5):.1f}s p90={pct(tts, 0.9):.1f}s" if tts
      else "  tts: no successes")
print(f"  collision_episodes={collided}/{n} ({collided / max(n, 1):.3f}) "
      f"min_clearance p10={pct(clr, 0.10):.2f}m")
if getattr(base, "episode_max_phase", None) is not None:
    phase_distribution = {
        phase: sum(r.get("max_phase") == phase for r in records)
        for phase in range(4)
    }
    print(f"  stages: reached_phase distribution {phase_distribution}")
if records and all("contact_steps" in r for r in records):
    contact_records = [r for r in records if r["contact_steps"] > 0]
    contact_seconds = sorted(r["contact_seconds"] for r in contact_records)
    contact_longest_seconds = sorted(
        r["contact_longest_seconds"] for r in contact_records
    )
    total_contact_steps = sum(r["contact_steps"] for r in contact_records)
    depth_mean = (
        sum(
            r["contact_depth_mean_m"] * r["contact_steps"]
            for r in contact_records
        ) / total_contact_steps
        if total_contact_steps > 0
        else float("nan")
    )
    print(
        f"  contact: episodes_with_contact={len(contact_records)}/{n} "
        f"total_s median={pct(contact_seconds, 0.5):.2f} "
        f"p90={pct(contact_seconds, 0.9):.2f}  "
        f"longest_s median={pct(contact_longest_seconds, 0.5):.2f}  "
        f"depth_mean={depth_mean:.3f} m"
    )
# Success rate alone has no discriminative power on a task where detouring
# works; certification prints the path length for the same reason
# eval_v6_frozen does.
path = sorted(r["path_length_m"] for r in succ) or sorted(
    r["path_length_m"] for r in records)
print(f"  path_len median={pct(path, 0.5):.1f}m p90={pct(path, 0.9):.1f}m", end="")
ratios = sorted(r["path_length_m"] / r["d0_m"] for r in succ
                if r.get("d0_m", 0.0) > 0.0)
print(f"  detour median={pct(ratios, 0.5):.2f}x straight" if ratios else "")
if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "level": args_cli.level,
                   "seed": args_cli.eval_seed,
                   # The header block when the env carries the protocol
                   # object, the off-protocol literal when it does not --
                   # never unconditionally the header.
                   "scenario_protocol": scenario_protocol,
                   # Every --set actually written to the cfg, under the same
                   # key scripts/eval_v6_frozen.py:340 uses so one schema
                   # serves both. A wave-ladder cell is identified by this
                   # field, not by whatever the caller named the file.
                   "overrides": applied_overrides,
                   "checkpoint": checkpoint_path,
                   "train_task": args_cli.train_task,
                   "eval_task": args_cli.eval_task,
                   "bridge": {
                       "train_layout": train_layout,
                       "eval_layout": eval_layout,
                       "mapping": mapping,
                       "shared": shared,
                       "zero_filled_superset_slots": zero_filled,
                       "dropped_eval_superset_slots": dropped,
                   },
                   "records": records}, f, indent=1)
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
