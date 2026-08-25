"""Paired checkpoint-ladder sweep that persists per-rung, per-episode records.

Why this exists rather than screen_v6_ladder.py: that script advances the layout
RNG across checkpoints inside one process, so every rung faces a DIFFERENT batch
of layouts and rung-to-rung differences carry layout variance on top of policy
variance. For studying the shape of a learning curve -- where the question is
"has this stopped improving" rather than "which checkpoint is best" -- the layout
draw is a nuisance variable, so by default every rung here is put back at the
START of the scenario stream and the comparison becomes paired.

How that pairing is done depends on the family (see restart_scenarios): on a
family carrying the scenario protocol the protocol object is rebuilt, which
returns every per-env episode counter to its starting value, so each rung enters
the sweep from the same protocol state and env i's j-th episode carries the same
KEY -- hence the same scenario -- in every rung; off-protocol families keep the
legacy layout-RNG rewind. The mechanism actually used is printed and written to
the curve JSON as "pairing", and on a protocol family the per-episode scenario
digests in the records make the claim checkable instead of asserted.

The five Suite S ids draw their obstacles from a frozen layout library instead
of a sampler, indexed by a SECOND per-env counter that the protocol object knows
nothing about (tasks/hazard_nav/hazard_nav_env.py:2071-2080). Rebuilding the
protocol leaves that counter where the previous rung stopped, so on those ids
rung k and rung k+1 used to face different frozen layouts while the JSON said
paired. restart_suite_s_rotation rewinds it alongside the protocol, and the
per-rung "suite_s_rewound_from" field records how far it had run.

Every rung is OPENED by a reset of the unwrapped env (open_rung) rather than by
the skrl wrapper's reset, because that wrapper is understood to reset only on
its first call: opening through it left rungs 2..N starting on an episode
carried over from the previous checkpoint -- run partly under the previous
policy, on a scenario drawn under the pre-rewind counters -- and that carried
episode is exactly the one scripts/cpi_verdict.py:51-56 pairs the rungs on.
"wrapper_reset_one_shot" in the JSON is the MEASURED answer to that question on
the machine that ran the sweep, and per-rung "opened_fresh" /
"opened_episode_index" say whether the open actually landed the rung at the
start of the stream.

Episode records use exactly the collection logic of eval_v6_frozen.py, so a rung
here and a certification there are the same measurement at different N.

Records are flushed after every rung: a watchdog kill still leaves usable data.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--task", default="Isaac-USV-HazardNav-Direct-v1")
parser.add_argument("--every", type=int, default=1, help="sweep every Nth checkpoint")
parser.add_argument("--only", default=None,
                    help="comma-separated step numbers; overrides --every")
parser.add_argument("--episodes", type=int, default=64, help="episodes per rung")
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=42)
parser.add_argument("--out", required=True, help="JSON curve file")
parser.add_argument("--unpaired", action="store_true",
                    help="let the layout RNG run on, reproducing screen_v6_ladder")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import json
import math
import os
import re
import time

import gymnasium as gym
import numpy as np
import torch

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# Scenario-protocol stamping, on the same dual import scripts/
# eval_v6_frozen.py:74-80 uses so the script keeps working from the repo and
# from the deployed task tree.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
try:
    from tasks._shared.scenario_draws import (
        episode_scenario_hashes_for,
        make_scenario_rng,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )
except ImportError:
    from isaaclab_tasks.direct._shared.scenario_draws import (
        episode_scenario_hashes_for,
        make_scenario_rng,
        scenario_protocol_notice,
        scenario_protocol_stamp,
    )

ckpt_dir = os.path.join(args_cli.run_dir, "checkpoints")
cks = sorted(
    (f for f in os.listdir(ckpt_dir) if re.fullmatch(r"agent_\d+\.pt", f)),
    key=lambda f: int(re.findall(r"\d+", f)[0]),
)
if args_cli.only:
    wanted = {int(s) for s in args_cli.only.split(",")}
    ladder = [c for c in cks if int(re.findall(r"\d+", c)[0]) in wanted]
else:
    ladder = cks[:: args_cli.every]
    if cks and cks[-1] not in ladder:
        ladder.append(cks[-1])
print(f"curve: {len(ladder)}/{len(cks)} checkpoints, {args_cli.episodes} eps each, "
      f"paired={not args_cli.unpaired}", flush=True)

TASK = args_cli.task
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env_cfg.seed = args_cli.eval_seed
if hasattr(env_cfg, "curriculum_frozen"):
    env_cfg.curriculum_frozen = True
    env_cfg.eval_level = args_cli.level
experiment_cfg = load_cfg_from_registry(TASK, "skrl_cfg_entry_point")
env = gym.make(TASK, cfg=env_cfg, render_mode=None)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
experiment_cfg["trainer"]["close_environment_at_exit"] = False
experiment_cfg["agent"]["experiment"]["write_interval"] = 0
experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
experiment_cfg["agent"]["experiment"]["wandb"] = False
runner = Runner(wrapped, experiment_cfg)
base = env.unwrapped
# Does THIS env carry the scenario protocol?  Same guard, same marker and same
# warning as scripts/eval_v6_frozen.py:148-162.  On protocol the per-rung
# scenario hashes make the "paired" claim in this file's docstring CHECKABLE
# rather than asserted: two rungs are paired iff their hashes agree episode by
# episode.  NOTE: rung records carry no "ep" key, so they align by env only.
scenario_protocol = scenario_protocol_stamp(base)
_notice = scenario_protocol_notice(TASK, scenario_protocol)
if _notice:
    print(_notice, flush=True)


# One spelling of the "the caller switched pairing off" mechanism, so the
# check in restart_scenarios cannot drift away from the string the JSON
# carries -- a drifted copy would silently start rewinding under --unpaired.
UNPAIRED = "off (--unpaired)"


def pairing_mechanism(base):
    """Which rewind this run can use to put the rungs on the same scenarios.

    "scenario-protocol"  the env carries a ScenarioRNG, so a scenario is a pure
                         function of (protocol version, eval seed, env index,
                         per-env episode index, primitive group) and rewinding
                         the episode counters repeats the scenarios exactly.
    "layout-rng-rewind"  no protocol object; the legacy single advancing numpy
                         generator is the only thing there is to rewind.
    "off (--unpaired)"   the caller asked for screen_v6_ladder behaviour.
    "none"               neither source exists; the rungs cannot be paired and
                         the JSON says so rather than claiming paired=true.

    This names the SCENARIO-STREAM mechanism only. The Suite S frozen-layout
    rotation is a second, independent source that rides along with any of the
    first three (see restart_suite_s_rotation) and is reported separately.
    """
    if args_cli.unpaired:
        return UNPAIRED
    if getattr(base, "_scenario", None) is not None:
        return "scenario-protocol"
    if hasattr(base, "_layout_rng"):
        return "layout-rng-rewind"
    return "none"


def restart_suite_s_rotation(base):
    """Rewind the Suite S frozen-layout rotation: the SECOND scenario source.

    The five Suite S ids (tasks/hazard_nav/__init__.py:458, 470, 482, 494, 506)
    replace the scatter draw with a frozen, checksum-verified layout library
    and pick this episode's member with ``rotation_index(env, that env's own
    completed-reset count)`` (tasks/hazard_nav/hazard_nav_env.py:2074-2080).
    That count lives in ``_suite_s_episode_counter``, a plain numpy array on
    the env -- it is NOT part of the ScenarioRNG, so rebuilding the protocol
    object in restart_scenarios does not touch it.

    Without this rewind the counter simply kept running across rungs: rung 1
    ended env i somewhere at k, rung 2 opened env i at k+1, and env i's first
    episode of rung 2 was layout ``(i + k + 1) % 10`` against rung 1's
    ``(i + 0) % 10``. The curve then compared checkpoints on different exams on
    exactly the five ids where the exam is a fixed, named artifact -- and
    scripts/cpi_verdict.py:51-56 pairs rungs on each env's FIRST episode, i.e.
    on the record the shift hits hardest.

    Zeroing is the exact analogue of the protocol rebuild: the env builds this
    array as ``np.zeros(num_envs)`` and post-increments it, so 0 means "this
    env has completed no episode", which is what ``ScenarioRNG``'s -1 means on
    its own counter (tasks/_shared/scenario_rng.py:351). Nothing about which
    layout an index maps to, nor about the rotation rule, is touched -- and
    since this function is called only from restart_scenarios, which only this
    sweep calls, a normal training or certification run counts exactly as it
    did before.

    Returns the largest counter value that was zeroed -- 0 on the first rung,
    where the env has not built the array yet (it is created lazily, inside
    ``_suite_s_layout_for``, at the first reset) -- or None on a family that
    has no Suite S rotation at all.
    """
    counter = getattr(base, "_suite_s_episode_counter", None)
    if counter is None:
        # No array yet. Two different situations, and the cfg field that is the
        # only thing which ever builds one tells them apart: a Suite S id
        # before its first reset (0 = already at the start of the rotation)
        # versus any other family (None = there is no rotation here).
        return 0 if getattr(base.cfg, "suite_s_class", "") else None
    highest = int(np.max(counter))
    counter[...] = 0
    return highest


def restart_scenarios(base, mechanism):
    """Put the env back at the start of its scenario stream, before a rung.

    WHAT USED TO BE HERE, AND WHY IT SILENTLY STOPPED PAIRING ANYTHING
    -----------------------------------------------------------------
    This was one line -- ``base._layout_rng = np.random.default_rng(eval_seed)``
    -- which rewound the single advancing numpy generator the layout used to be
    drawn from. The scenario-protocol migration made that line dead on exactly
    the families it was written for: this script always sets ``env_cfg.seed``
    (:100), so ``make_scenario_rng`` returns a ScenarioRNG, and a migrated env
    then draws its layout from that object and consults ``_layout_rng`` ONLY as
    the unseeded fallback (``tasks/path_hazard/path_hazard_env.py:952-955``).
    Rewinding ``_layout_rng`` therefore changed nothing at all, and the rungs
    went back to facing different scenarios while the JSON still said
    ``paired: true``.

    THE MECHANISM NOW
    -----------------
    Under the protocol the scenario for an episode is decided by its KEY, not by
    how many draws were consumed before it, and the only part of that key that
    moves during a run is the per-env episode index. Those counters only ever
    advance (``ScenarioRNG.reset_idx``), so without intervention rung 2 would
    carry on from wherever rung 1 stopped and face different scenarios.
    Rebuilding the protocol object with the SAME call the env itself used
    (``tasks/path_hazard/path_hazard_env.py:173``, and its three sibling envs)
    returns every counter to "never reset", so each rung enters ``collect`` from
    an identical protocol state and env i's j-th episode of one rung carries the
    same key -- hence the same scenario -- as env i's j-th episode of every
    other rung.

    Nothing about the sampled ranges, distributions or the env's own reset logic
    is touched: the rebuild only decides which keys the streams are seeded from.

    THE SECOND SOURCE. On the five Suite S ids the layout does not come from
    the protocol at all: it is looked up in a frozen library by a separate
    per-env counter on the env. Rebuilding the protocol object leaves that
    counter running, so those rungs stayed unpaired while everything else was
    fixed. restart_suite_s_rotation rewinds it, and this function returns what
    it found so the rung record can say how far it had run.

    WHAT THIS DOES NOT CONTROL is the env's PHYSICAL state -- whether the rung
    opens on a fresh episode or on one carried over from the previous
    checkpoint. That is not a stream problem and no scenario RNG can fix it; it
    is handled at the other end, by open_rung, which resets the unwrapped env
    itself instead of relying on the skrl wrapper's reset. Both must happen, in
    this order: this function decides WHICH keys the next episodes are drawn
    from, open_rung decides WHEN they are drawn.

    OFF-PROTOCOL FAMILIES keep the legacy rewind, which is the best available
    there and is not equivalent: it restores the layout STREAM, but on a family
    that ends episodes early (hazard_nav, harbor_mission) two checkpoints
    consume that stream at different rates, so the rungs share a stream without
    sharing per-episode scenarios.

    NOT rewound: an ``ObsDegrader``, which keys its own (env, episode) streams
    (``tasks/_shared/obs_degradation.py:160``). This script passes no
    degradation dose and every cfg default is zero
    (``tasks/station_keeping/station_keeping_env_cfg.py:187-192``), so
    ``build_obs_degrader`` returns ``None`` on every run of this script and
    there is nothing to rewind; a future dose flag here would have to grow one.

    Returns restart_suite_s_rotation's verdict, for the rung record.
    """
    if mechanism == UNPAIRED:
        # screen_v6_ladder behaviour was asked for: rewind nothing, the Suite S
        # rotation included, and report that nothing was rewound.
        return None
    if mechanism == "scenario-protocol":
        rebuilt = make_scenario_rng(base.cfg, base.num_envs, base.device)
        if rebuilt is None:
            # cfg.seed went away after construction. Carrying on would repeat
            # the exact failure this function exists to fix -- an unpaired sweep
            # labelled paired -- so stop loudly instead.
            raise RuntimeError(
                "the env carries a scenario protocol object but cfg.seed is "
                "None, so it cannot be rebuilt; the rungs would be unpaired "
                "while the curve JSON claimed otherwise")
        base._scenario = rebuilt
    elif mechanism == "layout-rng-rewind":
        base._layout_rng = np.random.default_rng(args_cli.eval_seed)
    # Independent of which of the three branches above applied: the Suite S
    # rotation is a counter on the ENV, part of neither stream, so it has to be
    # rewound on its own or those five ids stay unpaired.
    return restart_suite_s_rotation(base)


def policy_observation(reset_return):
    """The observation tensor an agent step sees, out of a gym reset return.

    ``DirectRLEnv.reset`` returns ``(self._get_observations(), self.extras)``
    and every env in this repo builds that first element as ``{"policy":
    tensor}`` (e.g. tasks/hazard_nav/hazard_nav_env.py:1066), which is the same
    tensor the skrl wrapper hands back from ``step``. Anything else is a
    wrapper or env that transforms observations on the way out, in which case
    feeding this straight to the agent would be wrong -- so it stops rather
    than guessing.
    """
    obs = reset_return[0] if isinstance(reset_return, tuple) else reset_return
    if isinstance(obs, dict):
        if "policy" not in obs:
            raise RuntimeError(
                "env.reset() returned an observation dict without a 'policy' "
                f"group (keys: {sorted(obs)}); open_rung cannot tell which one "
                "the agent consumes")
        obs = obs["policy"]
    return obs


def check_observation_matches(obs, template):
    """Refuse to run a rung on an observation the agent would not recognise."""
    if template is None:
        return
    same = (
        type(obs) is type(template)
        and getattr(obs, "shape", None) == getattr(template, "shape", None)
        and getattr(obs, "dtype", None) == getattr(template, "dtype", None)
    )
    if not same:
        raise RuntimeError(
            "the unwrapped env's reset observation "
            f"({type(obs).__name__}, {getattr(obs, 'shape', None)}, "
            f"{getattr(obs, 'dtype', None)}) does not match what the skrl "
            f"wrapper returns ({type(template).__name__}, "
            f"{getattr(template, 'shape', None)}, "
            f"{getattr(template, 'dtype', None)}); this wrapper transforms "
            "observations, so open_rung must be taught its transform before "
            "the sweep can open rungs through the unwrapped env")


def open_rung():
    """Open one rung on a FRESH episode, whatever the wrapper's reset does.

    THE BUG THIS REPLACES. The rung used to open with ``wrapped.reset()``.
    skrl's Isaac Lab wrapper is understood to reset the underlying env only on
    its FIRST call and to return cached observations afterwards; that could not
    be verified where this was written (skrl is not installed there), so both
    behaviours have to be assumed possible, and they differ:

      resets every call   every rung starts all 64 envs on episode 0 of the
                          rewound counters and runs 0, 1, 2, ... -- the indices
                          a fresh certification process uses, so a rung and a
                          certificate line up episode for episode.
      resets once         rung 1 behaves as above; rungs 2..N do not reset at
                          all. The previous rung stopped the step its last
                          record landed, leaving every env either mid-episode
                          or one step into a fresh one, so the FIRST episode
                          each env finishes in the new rung is one whose
                          scenario was drawn under the PREVIOUS rung's
                          counters -- and, for the envs that were mid-episode,
                          whose opening steps were steered by the PREVIOUS
                          checkpoint. Every later record is paired, so the
                          rung's records are shifted by one per env -- and
                          scripts/cpi_verdict.py:51-56 pairs the rungs on
                          precisely each env's first record.

    THE FIX. ``base.reset()`` is Gymnasium's reset on the unwrapped env, and
    ``DirectRLEnv.reset`` unconditionally resets every env and returns fresh
    observations (isaaclab/envs/direct_rl_env.py:292-331 -- it has no
    once-only flag to be defeated by). Opening the rung through it makes both
    wrapper behaviours produce the same rung: all envs fresh, all counters at
    their first episode. The wrapper is still what steps the sweep, and one
    ``wrapped.reset()`` runs before the ladder so a wrapper that wants its
    first-call reset gets it; after that its reset is never used again, so
    which of the two behaviours it has stops mattering.

    Must run AFTER restart_scenarios: this is the reset that consumes the keys
    that function rewinds.

    --unpaired gets the fresh open too. Carrying an episode across a checkpoint
    boundary is a measurement error, not a pairing choice -- the record would
    be credited to a checkpoint that only steered part of it -- so the control
    arm is entitled to it as much as the paired one. What --unpaired still
    switches off is every rewind, which is what makes it the control.
    """
    obs = policy_observation(base.reset())
    check_observation_matches(obs, OBS_TEMPLATE)
    return obs


def rung_open_state():
    """Evidence that the open landed the rung at the start of the stream.

    ``(opened_fresh, [lowest, highest] episode index)``, either entry None when
    the env does not carry the thing it is read from. ``opened_fresh`` is
    ``episode_length_buf`` all-zero, i.e. no env is mid-episode
    (isaaclab/envs/direct_rl_env.py:197,388,628); the episode indices come off
    the protocol counters and should read ``[0, 0]`` on every rung of a paired
    protocol sweep. Recorded rather than asserted so a sweep on a family with
    neither still produces a curve, and so a reader can tell WHICH of the two
    guarantees a suspicious curve lost.
    """
    lengths = getattr(base, "episode_length_buf", None)
    fresh = None if lengths is None else bool(int(lengths.max()) == 0)
    scenario = getattr(base, "_scenario", None)
    span = None
    if scenario is not None:
        indices = scenario.episode_indices()
        span = [int(indices.min()), int(indices.max())]
    return fresh, span


def collect(target, obs):
    """One rung. Identical per-episode semantics to eval_v6_frozen.py.

    ``obs`` comes from open_rung, not from a reset in here: see open_rung for
    why the rung boundary must not go through the wrapper.
    """
    records = []
    max_steps = (target // base.num_envs + 3) * base.max_episode_length
    step = 0
    while len(records) < target and step < max_steps:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
        # d0_per_env is rewritten inside _reset_idx during step(); snapshot it
        # while it still belongs to the episode that is about to end.
        d0_prev = base.d0_per_env.clone() if hasattr(base, "d0_per_env") else None
        obs, _, term, trunc, _ = wrapped.step(actions)
        step += 1
        done = term | trunc
        done = done.squeeze(-1) if done.dim() > 1 else done
        for i in torch.nonzero(done).flatten().tolist():
            tts = float(base.time_to_success[i])
            rec = {
                "env": i,
                "success": bool(base.episode_success[i]),
                "tts_s": None if math.isnan(tts) else tts,
                "min_clearance_m": float(base.episode_min_clearance[i]),
                "path_length_m": float(base.episode_path_length[i]),
            }
            # Per-primitive digests of the scenario the FINISHED episode
            # ran, latched at reset exactly like episode_min_clearance.
            # Absent on families not yet on the scenario protocol, and empty
            # until an env has completed its first episode; both cases are
            # guarded inside the helper, which returns None for "no field".
            scenario_hashes = episode_scenario_hashes_for(base, i)
            if scenario_hashes is not None:
                rec["scenario_hashes"] = scenario_hashes
            if hasattr(base, "episode_max_phase"):
                rec["max_phase"] = int(base.episode_max_phase[i])
            if hasattr(base, "episode_contact_steps"):
                rec["contact_steps"] = int(base.episode_contact_steps[i])
            if d0_prev is not None:
                rec["d0_m"] = float(d0_prev[i])
            if hasattr(base, "route_geodesic_length"):
                geodesic = float(base.route_geodesic_length[i])
                if geodesic > 0.0:
                    rec["route_geodesic_m"] = geodesic
            records.append(rec)
    return records[:target]


# Decided once: the mechanism cannot change between rungs. Printed as well as
# stored, so a sweep log says on its face whether "paired" meant anything on
# this family -- the previous version of this script printed paired=True while
# rewinding a generator the env no longer read.
PAIRING = pairing_mechanism(base)
print(f"curve: pairing mechanism = {PAIRING}", flush=True)

# Whether this id rotates a frozen Suite S library on top of its scenario
# stream, and therefore whether "suite_s_rewound_from" in the rungs means
# anything. Read off the cfg, which is what builds the counter in the first
# place; the counter itself does not exist until the first reset.
SUITE_S_CLASS = getattr(base.cfg, "suite_s_class", "") or None
if SUITE_S_CLASS is None:
    SUITE_S_ROTATION = "n/a (not a Suite S id)"
elif PAIRING == UNPAIRED:
    SUITE_S_ROTATION = "left running (--unpaired)"
else:
    SUITE_S_ROTATION = "rewound per rung"
print(f"curve: suite S rotation = {SUITE_S_ROTATION}", flush=True)

# One wrapper reset before the ladder, for a wrapper that wants its first-call
# reset; from here on rungs are opened by open_rung through the unwrapped env.
# The observation it returns is the template every rung's opening observation
# is checked against, so a wrapper that transforms observations is caught at
# the first rung instead of quietly feeding the agent something else. Both
# sides of that comparison go through policy_observation, so the check is about
# the TENSOR and not about which of the two hands back the containing dict.
OBS_TEMPLATE = policy_observation(wrapped.reset())

# MEASURED, not assumed: does a second wrapped.reset() reach the env at all?
# A real reset advances every per-env episode counter by one, a cached one
# advances nothing, so the protocol counters answer the question the old
# restart_scenarios docstring could only speculate about. The probe costs one
# extra reset before the ladder loop, and on a paired sweep every rung rewinds
# the counters afterwards, so no rung sees a different scenario for it -- which
# is why it is skipped under --unpaired, where nothing is rewound and an extra
# reset really would shift the layout stream the control arm exists to keep.
# None whenever it was skipped or there is no protocol counter to read it off.
WRAPPER_RESET_ONE_SHOT = None
if PAIRING != UNPAIRED and getattr(base, "_scenario", None) is not None:
    _before_probe = base._scenario.episode_indices().clone()
    wrapped.reset()
    WRAPPER_RESET_ONE_SHOT = bool(
        torch.equal(_before_probe, base._scenario.episode_indices())
    )
    print(f"curve: wrapper reset is one-shot = {WRAPPER_RESET_ONE_SHOT} "
          "(rungs are opened through the unwrapped env either way)", flush=True)

curve = {
    "task": TASK,
    "run_dir": os.path.abspath(args_cli.run_dir),
    "level": args_cli.level,
    "eval_seed": args_cli.eval_seed,
    # The header block when the env carries the protocol object, the
    # off-protocol literal when it does not -- never unconditionally
    # the header.
    "scenario_protocol": scenario_protocol,
    "episodes_per_rung": args_cli.episodes,
    "paired": not args_cli.unpaired,
    # HOW the rungs were paired, not just that pairing was asked for. "paired"
    # above records the flag; a reader cannot tell from a flag whether the
    # mechanism behind it was actually available on this family, and a rung
    # sweep whose pairing silently did nothing is what this field exists to
    # make visible.
    "pairing": PAIRING,
    # The SECOND scenario source, reported separately because it is rewound by
    # its own mechanism and was the one the "pairing" field above used to
    # overstate: on the five Suite S ids "scenario-protocol" pairing was true
    # of the stream and false of the frozen layout the episode actually ran.
    "suite_s_class": SUITE_S_CLASS,
    "suite_s_rotation": SUITE_S_ROTATION,
    # How a rung starts, and what the wrapper's repeat reset was measured to
    # do on this machine. Together these say whether each rung began on fresh
    # episodes -- the property cpi_verdict.py's first-episode pairing needs and
    # which an opening wrapped.reset() silently failed to provide.
    "rung_open": "unwrapped-env-reset",
    "wrapper_reset_one_shot": WRAPPER_RESET_ONE_SHOT,
    "total_checkpoints": len(cks),
    "rungs": [],
}

for ck in ladder:
    # The first deployment of this script omitted this load: 12 sweeps then
    # measured a randomly-initialized policy 94 times each and returned all
    # zeros, including on runs with certified 100% champions. Nothing in the
    # rung loop may run before the weights are in.
    runner.agent.load(os.path.join(ckpt_dir, ck))
    runner.agent.set_running_mode("eval")
    # Same episode scenarios for every checkpoint -> rung differences are
    # paired. Both rewinds must run BEFORE open_rung, which is the reset that
    # draws the rung's first episodes off the counters they put back.
    suite_s_rewound_from = restart_scenarios(base, PAIRING)
    started = time.time()
    obs0 = open_rung()
    opened_fresh, opened_episode_index = rung_open_state()
    if opened_fresh is False:
        # Cannot happen through DirectRLEnv.reset; if it ever does, the rung
        # carries episodes started under the previous checkpoint and the
        # first-episode pairing cpi_verdict.py does is invalid for this curve.
        print(f"WARNING {ck}: rung did not open on fresh episodes; its first "
              "record per env is carried over from the previous rung",
              flush=True)
    records = collect(args_cli.episodes, obs0)
    n = len(records)
    successes = sum(1 for r in records if r["success"])
    collided = sum(1 for r in records if r["min_clearance_m"] < 0.0)
    sr = successes / max(n, 1)
    curve["rungs"].append({
        "checkpoint": ck,
        "step": int(re.findall(r"\d+", ck)[0]),
        "episodes": n,
        "successes": successes,
        "sr": sr,
        "collision_episodes": collided,
        # Per-rung evidence for the two pairing claims the header makes, so a
        # reader checks them instead of trusting them. suite_s_rewound_from is
        # the highest rotation counter this rung put back: None off Suite S, 0
        # on the first rung, and on later rungs the number of episodes the
        # previous rung had rotated through -- i.e. exactly the layout drift
        # that would otherwise have been carried into this rung. Then whether
        # the rung opened with no env mid-episode, and where the protocol
        # counters stood at the open ([0, 0] on a paired protocol sweep).
        "suite_s_rewound_from": suite_s_rewound_from,
        "opened_fresh": opened_fresh,
        "opened_episode_index": opened_episode_index,
        "seconds": round(time.time() - started, 1),
        "records": records,
    })
    with open(args_cli.out, "w", encoding="utf-8") as handle:
        json.dump(curve, handle)
    print(f"LADDER {ck}: SR={sr:.4f} ({successes}/{n}) "
          f"coll={collided} {time.time() - started:.0f}s", flush=True)

best = max(curve["rungs"], key=lambda r: r["sr"], default=None)
if best:
    print(f"CURVE DONE rungs={len(curve['rungs'])} "
          f"best={best['checkpoint']} SR={best['sr']:.4f} -> {args_cli.out}")
sys.stdout.flush()
env.close()
app.close()
