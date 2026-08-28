"""Frozen-level eval with per-episode records for paired statistics.

Paired statistics (McNemar, paired bootstrap) require that two policies at the
same --eval-seed sat the SAME exam paper, episode by episode. On a family that
carries the scenario protocol they do: every primitive is keyed by (protocol
version, eval seed, env index, per-env episode index, primitive group), so the
scenario for env i's k-th episode does not depend on when that episode happened
to start or on how many draws the controller consumed before it. Such a run
stamps ``scenario_protocol`` with the protocol header block.

That claim is NOT free, and this script used to assert it for every task. The
older wording here -- "layouts come from the env's layout RNG and reset timing
is fixed-horizon" -- holds only while nothing terminates early; on a family that
ends episodes on an outcome, a single advancing generator hands two controllers
different scenarios at the same reset index. So a run whose env carries no
scenario protocol object now stamps the literal SCENARIO_PROTOCOL_OFF instead of
the header, and its per-episode records must be treated as unpaired.
"""
import argparse
import json
import os
import sys

# Shared --set FIELD=VALUE parsing (scripts/cfg_override.py). Dotted paths
# reach nested cfgs, which the wave ladder needs: its rungs live at
# cfg.sea_state.hs_range, one level below anything a top-level-only override
# could touch. Pure Python, no Isaac import, so it is safe before the launcher.
try:
    from cfg_override import apply_overrides
except ImportError:  # invoked from a cwd that is not scripts/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cfg_override import apply_overrides

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-USV-HazardNav-Direct-v1")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
# Off-policy checkpoints (SAC/TD3) carry different model roles than the
# PPO SharedModel, so the runner must be built from THEIR config or the
# state_dict load fails on missing policy_layer/value_layer keys.
parser.add_argument("--cfg-entry-point", default="skrl_cfg_entry_point",
                    help="registry key for the skrl config, e.g. "
                         "skrl_sac_cfg_entry_point")
parser.add_argument("--out", default=None, help="JSON per-episode records")
parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="extra cfg overrides (repeatable); dotted paths reach "
                         "nested cfgs and comma tuples pin ranges, e.g. "
                         "--set obs_noise_sigma_ray=0.18 or one wave rung "
                         "--set sea_state.hs_range=0.6,0.6; same coercion "
                         "rules as eval_imbalance.py")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# ---- squashed-policy hook (mirrors scripts/sac_train.py) -------------------
# skrl's Runner._component is a closed whitelist, so the v4 squashed-Gaussian
# policy class (skrl_sac_v4_cfg.yaml) can never be resolved by the stock
# Runner. Without this hook, screening/certifying a v4 checkpoint dies at
# model construction -- which is exactly how the first v4 queue produced a
# bogus "below gate" verdict from an SR=-1 crash.
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
try:
    from tasks._shared.squashed_gaussian import squashed_gaussian_model
except ImportError:
    from isaaclab_tasks.direct._shared.squashed_gaussian import squashed_gaussian_model
# Scenario-protocol stamp for the certificate header, on the same dual import
# so the script keeps working from the repo and from the deployed task tree.
try:
    from tasks._shared.scenario_draws import scenario_protocol_header
except ImportError:
    from isaaclab_tasks.direct._shared.scenario_draws import (
        scenario_protocol_header,
    )

# Certificate value for a run whose env never built a ScenarioRNG: the episode
# scenarios of that run do NOT descend from --eval-seed in a controller-
# independent way, and the certificate has to say so rather than stay silent.
# A missing key is ambiguous (old certificate? unstamped writer? off protocol?);
# an explicit marker is not. It is a plain string, never the header dict, so no
# auditor can read it as compliance -- check_scenario_independence.py:257 stores
# the field opaquely and :736-738 only compares and formats it, so the marker
# rides through unchanged and a pair with one side on protocol and one side off
# raises the "scenario_protocol differs" warning it should.
SCENARIO_PROTOCOL_OFF = "off-protocol"


class SquashedRunner(Runner):
    def _component(self, name: str):
        if name.lower() == "squashedgaussianmixin":
            return squashed_gaussian_model
        return super()._component(name)
# ---------------------------------------------------------------------------


TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env_cfg.seed = args_cli.eval_seed
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
# Generic eval-time cfg overrides (exact pattern and type coercion of
# scripts/eval_imbalance.py --set). The observation-degradation dose axes
# (obs_noise_sigma_*/obs_bias_sigma_*/obs_delay_steps/obs_dropout_p) ride
# this, so degraded runs never mint a gym id; every applied override is
# echoed into the output JSON top level.
applied_overrides = dict(apply_overrides(env_cfg, args_cli.extra_sets))
# Resolved degradation dose actually carried by this run (defaults included),
# recorded so a gcert JSON is self-describing even when no --set was passed.
OBS_DEGRADATION_FIELDS = (
    "obs_noise_sigma_pos", "obs_noise_sigma_vel", "obs_noise_sigma_ray",
    "obs_bias_sigma_pos", "obs_bias_sigma_vel", "obs_bias_sigma_ray",
    "obs_delay_steps", "obs_dropout_p",
)
obs_degradation = {
    field: getattr(env_cfg, field)
    for field in OBS_DEGRADATION_FIELDS
    if hasattr(env_cfg, field)
}
experiment_cfg = load_cfg_from_registry(TASK, args_cli.cfg_entry_point)
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False
runner = SquashedRunner(wrapped, experiment_cfg)
runner.agent.load(os.path.abspath(args_cli.checkpoint))
runner.agent.set_running_mode("eval")

base = env.unwrapped
# Does THIS env actually carry the scenario protocol? Two ways it may not: the
# task family was never migrated (no _scenario attribute at all), or it was
# migrated but built unseeded, in which case make_scenario_rng returned None and
# every draw fell back to the historical global-RNG line. Both cases are
# off-protocol and both are caught by the same test. env_cfg.seed is set from
# --eval-seed above, so for a migrated family this resolves to the header.
scenario_protocol = (
    scenario_protocol_header()
    if getattr(base, "_scenario", None) is not None
    else SCENARIO_PROTOCOL_OFF
)
if scenario_protocol == SCENARIO_PROTOCOL_OFF:
    print(f"WARNING: {TASK} is not on the scenario protocol; this certificate "
          f"is stamped scenario_protocol={SCENARIO_PROTOCOL_OFF!r} and its "
          "episodes are NOT paired across evaluation seeds", flush=True)
obs, _ = wrapped.reset()
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
        # Cross-track error: the path families have maintained this per episode
        # all along (path_following_env.py:619, path_hazard_env.py:883) and it
        # was simply never written out, so no past certificate can be rescored
        # on tracking quality. Every future one can.
        if hasattr(base, "episode_xte_rms"):
            rec["xte_rms_m"] = float(base.episode_xte_rms[i])
        # Per-primitive digests of the scenario the FINISHED episode ran,
        # latched at reset exactly like episode_min_clearance. Absent on
        # families not yet on the scenario protocol, and empty until an env has
        # completed its first episode, so both cases are guarded.
        episode_scenario_hashes = getattr(base, "episode_scenario_hashes", None)
        if episode_scenario_hashes is not None and episode_scenario_hashes[i]:
            rec["scenario_hashes"] = dict(episode_scenario_hashes[i])
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
# works: the shifted-potential run certified 100%/100% on Task A while walking
# a 68.6 m median path against a 29.7 m straight line -- a BIGGER detour than
# the baseline policy the validity audit had already flagged as going around.
# Only the path length made that visible, so certification prints it.
path = sorted(r["path_length_m"] for r in succ) or sorted(
    r["path_length_m"] for r in records)
print(f"  path_len median={pct(path, 0.5):.1f}m p90={pct(path, 0.9):.1f}m", end="")
ratios = sorted(r["path_length_m"] / r["d0_m"] for r in succ
                if r.get("d0_m", 0.0) > 0.0)
print(f"  detour median={pct(ratios, 0.5):.2f}x straight" if ratios else "")
if args_cli.out:
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"task": TASK, "level": args_cli.level,
                   # The evaluation seed. Kept under the existing name and NOT
                   # duplicated under "eval_seed": check_scenario_independence
                   # .py:442-449 discards exactly the key "seed" from its
                   # metadata diff, so a second copy of the same number would
                   # warn "metadata also differs on: eval_seed" on every
                   # healthy pair -- the two members of a pair differ precisely
                   # in their evaluation seed.
                   "seed": args_cli.eval_seed,
                   # The header block when the env carries the protocol object,
                   # the SCENARIO_PROTOCOL_OFF literal when it does not. Never
                   # unconditionally the header: stamping compliance on a task
                   # whose scenarios were never on the protocol is the one
                   # failure mode a certificate must not have.
                   "scenario_protocol": scenario_protocol,
                   "checkpoint": os.path.abspath(args_cli.checkpoint),
                   "overrides": applied_overrides,
                   "obs_degradation": obs_degradation,
                   "records": records}, f, indent=1)
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
