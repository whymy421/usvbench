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

ORIGINAL_ARGV = list(sys.argv)

# Shared --set FIELD=VALUE parsing (scripts/cfg_override.py). Dotted paths
# reach nested cfgs, which the wave ladder needs: its rungs live at
# cfg.sea_state.hs_range, one level below anything a top-level-only override
# could touch. Pure Python, no Isaac import, so it is safe before the launcher.
try:
    from cfg_override import apply_overrides
    from condition_args import apply_condition_args
except ImportError:  # invoked from a cwd that is not scripts/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cfg_override import apply_overrides
    from condition_args import apply_condition_args

# G6: fail-closed exit codes and certificate publication (scripts/evaluator_exit.py).
import evaluator_exit  # noqa: E402

_EXIT_GUARD = evaluator_exit.Guard()

from cert_v3 import CertificateRun
from eval_protocol import add_protocol_arguments, preflight_protocol, resolve_protocol

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
parser.add_argument("--condition-preset", action="append", default=[],
                    help="opt-in condition preset, repeat for different suites")
parser.add_argument("--condition-config", default=None,
                    help="JSON file of explicit nested condition overrides")
parser.add_argument("--set", dest="extra_sets", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="extra cfg overrides (repeatable); dotted paths reach "
                         "nested cfgs and comma tuples pin ranges, e.g. "
                         "--set obs_noise_sigma_ray=0.18 or one wave rung "
                         "--set sea_state.hs_range=0.6,0.6; same coercion "
                         "rules as eval_imbalance.py")
add_protocol_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
preflight_protocol(args_cli, ORIGINAL_ARGV, parser)
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app
_EXIT_GUARD.attach_app(app)
# G6: the body below runs at module level; an uncaught exception (or a shutdown that skips
# _EXIT_GUARD.finish) ends the process nonzero with no certificate (evaluator_exit).
_EXIT_GUARD.install_process_hooks()

import math
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# F7: journal ids (preflight_protocol already selected the vessel-specific id)
# are registered by tasks/journal (D10); the certificate takes the journal
# mixin header/records (cert_v3) and --suite is applied by resolve_protocol.
if getattr(args_cli, "journal_task_info", None) is not None:
    from eval_protocol import register_journal_tasks

    register_journal_tasks()

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
    from tasks._shared.scenario_draws import (
        episode_appearance_fields_for,
        scenario_protocol_header,
    )
except ImportError:
    from isaaclab_tasks.direct._shared.scenario_draws import (
        episode_appearance_fields_for,
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

# Additive certificate hook; imports the same shared module as the env.
try:
    from tasks._shared.conditions import certificate_fields, episode_certificate_fields
except ImportError:
    from isaaclab_tasks.direct._shared.conditions import certificate_fields, episode_certificate_fields


class SquashedRunner(Runner):
    def _component(self, name: str):
        if name.lower() == "squashedgaussianmixin":
            return squashed_gaussian_model
        return super()._component(name)
# ---------------------------------------------------------------------------


TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
applied_overrides = resolve_protocol(env_cfg, args_cli)
# E1 merge: D1/C4 condition presets are applied after the D8 protocol batch,
# i.e. after every --set as in J-base. They only replace cfg.conditions and
# cfg.condition_presets (no protocol field), so the batch readback above is
# unaffected; CertificateRun stamps them as header.conditions.
apply_condition_args(env_cfg, args_cli.condition_preset, args_cli.condition_config)
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
cert = CertificateRun(
    base=base, env_cfg=env_cfg, evaluator=__file__, task=TASK,
    controller="learned",
    controller_config={
        "cfg_entry_point": args_cli.cfg_entry_point,
        "experiment": experiment_cfg,
        "runtime_agent_config": getattr(runner.agent, "cfg", None),
        "runtime_models": {
            name: str(model)
            for name, model in getattr(runner.agent, "models", {}).items()
        },
        "action_selection": "mean_actions_if_available",
    },
    checkpoint=args_cli.checkpoint, eval_seed=args_cli.eval_seed,
    level=args_cli.level, episodes_requested=args_cli.episodes,
    overrides=applied_overrides, argv=ORIGINAL_ARGV, cli_args=vars(args_cli),
)
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
cert.after_reset()
records = []
ep_counter = torch.zeros(base.num_envs, dtype=torch.long)
# PER-ENV QUOTA, not a pooled count (fixed 2026-08-31, owner-directed).
# The loop used to stop the instant len(records) reached --episodes, which
# collected the first N episodes to COMPLETE and therefore deleted long
# episodes, i.e. timeouts, preferentially.  Quota form gives a fixed
# (env, ep) grid that completion order cannot reweight.  Same reasoning and
# the same evidence as scripts/eval_imbalance.py; step budget unchanged.
quota = cert.quota
max_steps = (quota + 3) * base.max_episode_length
step = 0
while bool((ep_counter < quota).any()) and step < max_steps:
    with cert.time_controller(), torch.inference_mode():
        outputs = runner.agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
    # d0_per_env is rewritten inside _reset_idx, which runs during step(), so
    # reading it alongside the other episode stats would report the NEXT
    # episode's straight-line distance. Snapshot it while it still belongs to
    # the episode that is about to end.
    d0_prev = (base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None)
    with cert.time_environment():
        obs, _, term, trunc, _ = wrapped.step(actions)
    cert.after_step()
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
        # Past quota this env keeps running (others are not done yet) but its
        # extra episodes are NOT recorded.  Dropping them here rather than
        # truncating afterwards is the whole fix: truncation is what let
        # completion order decide the sample.
        if int(ep_counter[i]) >= quota:
            cert.discard_episode(i)
            ep_counter[i] += 1
            continue
        rec = {
            "env": i,
            "ep": int(ep_counter[i]),
            "success": bool(base.episode_success[i]),
            "terminated": bool(term.reshape(-1)[i]),
            "truncated": bool(trunc.reshape(-1)[i]),
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
        if hasattr(base, "episode_station_error_rms_m"):
            rec["station_error_rms_m"] = float(base.episode_station_error_rms_m[i])
            rec["station_error_p95_m"] = float(base.episode_station_error_p95_m[i])
            rec["station_error_final_60s_rms_m"] = float(
                base.episode_station_error_final_60s_rms_m[i]
            )
            rec["station_error_final_60s_p95_m"] = float(
                base.episode_station_error_final_60s_p95_m[i]
            )
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
        # What the mid-episode appearance actually did.  Empty dict on
        # families without the axis, so unrelated certificates are
        # byte-identical.  Without these, a 128/128 on the appearance axis
        # mixes "avoided it" with "the episode ended before it existed",
        # since t0 ~ U[5, 30] s and a champion finishes in 18 to 21 s.
        rec.update(episode_appearance_fields_for(base, i))
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
        rec.update(episode_certificate_fields(base, i))
        records.append(cert.finish_episode(i, rec))
        ep_counter[i] += 1

# No truncation: the loop already produced an exact (env, ep) grid.
grid_complete = bool((ep_counter >= quota).all())
if not grid_complete:
    print(
        f"[INCOMPLETE] step budget exhausted with "
        f"{int((ep_counter < quota).sum())} of {int(base.num_envs)} envs "
        f"short of the {quota}-episode quota",
        flush=True,
    )
n = len(records)
succ = [r for r in records if r["success"]]
tts = sorted(r["tts_s"] for r in succ if r["tts_s"] is not None)
clr = sorted(r["min_clearance_m"] for r in records
             if r.get("min_clearance_m") is not None)
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
if records and all(all(r.get(key) is not None for key in (
    "contact_steps", "contact_seconds", "contact_longest_seconds", "contact_depth_mean_m"
)) for r in records):
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
path = sorted(r["path_length_m"] for r in succ if r.get("path_length_m") is not None) or sorted(
    r["path_length_m"] for r in records if r.get("path_length_m") is not None)
print(f"  path_len median={pct(path, 0.5):.1f}m p90={pct(path, 0.9):.1f}m", end="")
ratios = sorted(r["path_length_m"] / r["d0_m"] for r in succ
                if r.get("d0_m") is not None and r["d0_m"] > 0.0
                and r.get("path_length_m") is not None)
print(f"  detour median={pct(ratios, 0.5):.2f}x straight" if ratios else "")
if args_cli.out:
    with _EXIT_GUARD.output(args_cli.out).open() as f:
        json.dump(cert.build({"task": TASK, "level": args_cli.level,
                   # Sampling frame (added 2026-08-31).  None of these existed before,
                   # which is why the completion-order defect survived a 30-cell audit:
                   # a reader could not tell how many envs a cell used, how many episodes
                   # were requested, or whether the run finished.
                   "num_envs": int(base.num_envs),
                   "episodes_requested": int(args_cli.episodes),
                   "episodes_per_env": int(quota),
                   "complete": bool(grid_complete),
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
                   "records": records, **certificate_fields(base)}), f, indent=1, allow_nan=False)
    print(f"  records -> {args_cli.out}")
sys.stdout.flush()
env.close()
# G6b: publish the certificate (verify + rename), THEN close the app under a watchdog;
# a failure never calls close() and exits nonzero (evaluator_exit.py docstring).
_EXIT_GUARD.finish()
