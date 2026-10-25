"""Journal PPO training entry point (G19b, IMPLEMENTATION_PLAN P5): one task, one vessel, one training seed.

Algorithm: Proximal Policy Optimization (Schulman, Wolski, Dhariwal, Radford and Klimov 2017, "Proximal Policy
Optimization Algorithms", arXiv:1707.06347), the skrl implementation named by the registration's
skrl_cfg_entry_point agent configuration (agent class PPO); G28 adds the citation (NC 39).

usage:
  python -B scripts/train_journal_ppo.py --task <journal id | journal:<task>> --vessel blueboat|wamv
        --training-seed N --out NEW_DIR [--arm mixture|calm] [--purpose training|training_smoke]
        [--budget-timesteps N] [--num-envs 64] [--checkpoint-every N] [--calm-slots 3] [--prefetch]
        [--release-record PATH] [--host-class NAME] [--concurrency K] [--device cuda:0] [--dry-run]
        [--preflight-out PATH] [AppLauncher arguments, e.g. --headless]

What it does (TRAINING_DESIGN sections 2, 3, 5; CONTEXT_R4 D8 namespaces, D12, D25, D28):
* PREFLIGHT, without the simulator (also the whole of ``--dry-run``): the journal id (eval_protocol
  resolve_journal_task); the seed (seed_registry.resolve_training_block: purpose 'training' needs the FROZEN
  registry and gate, 'training_smoke' (90001, 90002) runs while the registry is a proposal; every evaluation
  seed is refused); RE-1 (reference_environment: a campaign run outside RE-1 is refused, a smoke run is stamped);
  the code-tree hash (code_tree); the mixture (conditions/training_mixture.py, appearing wired iff the task is in
  conditions/appearing/config.WIRED_TASKS at run time; arm 'calm' = no stack, the smoke's control); the stack
  resolved (journal_config.resolve_journal_conditions), checked against the mixture and ADMITTED on a CPU plan
  cfg (journal_config.admit_journal_conditions); the agent yaml of the registration (tasks/journal) and the
  stamped runtime overrides; the family's training pool module (tasks/<carrier>/training_pool.py); the output
  directory must not exist. Any refusal exits 2 before anything is written.
* RUN (Isaac Lab, skrl): out/ is created and out/preflight.json written FIRST; parse_env_cfg; cfg.seed = the
  training seed; the stack admitted on the REAL cfg and applied; cfg.journal_training_mixture = the mixture;
  cfg.journal_d7_pool / journal_appearing_pool = the family's online-admission adapters (P3); agent cfg =
  load_cfg_from_registry(task, 'skrl_cfg_entry_point') with ONLY the overrides of
  ppo_training_record.ALLOWED_AGENT_OVERRIDES (seed, trainer.timesteps, trainer.close_environment_at_exit False,
  experiment directory / name, checkpoint_interval, write_interval, wandb False), all stamped; the env must
  carry the mixture and the adapters (checked after gym.make); skrl Runner(...).run('train').
  A heartbeat file (out/heartbeat.json, atomic replace) every iteration (= rollouts env steps) and a progress
  line (stdout and out/progress.log) every 1 % of the budget. G35 (controller decision J19, v2.6.1): a heartbeat
  whose atomic replace fails (OSError, e.g. a Windows reader holding the file) is retried up to
  HEARTBEAT_REPLACE_ATTEMPTS times HEARTBEAT_REPLACE_WAIT_S apart and then SKIPPED (counted, warned on stderr),
  never fatal; an appearing episode whose trigger speed lies outside the vessel envelope is recorded in its stamp
  (never refused) and counted in divergence_checks.appearing_trigger_speed of the training record.
  Code 9 (follow-up C1, controller decision of REV-G19b verify2): a row that terminates with the journal code 9
  (non_finite_state, a LEGAL episode outcome) carries a non-finite reward computed from its non-finite state; the
  wrapper replaces it by the FAMILY'S terminal failure penalty (the reward the family assigns to a collision
  termination: -env_cfg.reward_contact_entry_penalty for hazard_nav / path_hazard / docking / harbor_mission;
  0.0 for path_following / station_keeping, whose rewards carry no termination penalty; the table is
  ppo_training_record.CODE9_PENALTY_FIELD) before the agent sees it, stamps the value and its source and counts
  it in the record. Code-9 alarm: more than max(3, 1e-4 x finished episodes) code-9 rows stop the run as
  'diverged' (ppo_training_record.code9_alarm_limit). Divergence (exit 4, record status 'diverged') = a
  non-finite action (checked before every env.step), a non-finite reward on any other row, a non-finite
  observation (a device-side flag over all rows every step, read at every step with a terminating row and at
  every iteration end), the code-9 alarm, a non-finite loss (skrl track_data hook) or non-finite parameters /
  optimizer / preprocessor state (checked after EVERY update, and after any exception of the trainer, together
  with the last observations and actions: a policy NaN surfacing as a torch ValueError is 'diverged', not an
  infrastructure failure).
  Any other exception or interruption is an infrastructure failure (out/STATUS.json, no training record,
  exit 3: rerun the SAME seed, seed_registry rule).
* AFTER: the final checkpoint is agent_<budget>.pt of the skrl run (the last-checkpoint rule), copied to
  out/final.pt; its parameters must be finite (else 'diverged'); SHA-256 of every checkpoint; the training
  record (scripts/ppo_training_record.py, schema usvbench.ppo_training_record/1) is written LAST.
* G44 (v2.6.3, owner decision 2026-10-13; scripts/ppo_training_fixes.py): R2 every row that terminates with a
  failure code (2, 4, 5, 6, 7, 8, 9) gets the failure terminal penalty -(A + B x D) on top of its step reward
  (D = the episode's layout bounds diagonal; A, B from the family's per-step reward range and the horizon), so
  failing is never worth more than continuing to the time limit; R1 the state normaliser is frozen after
  5 % of the budget, KLAdaptiveLR gets the floor 0.1 x the yaml learning rate, and out/ppo_fix.json (listed in
  log_digests) records all three. The periodic checkpoints are evaluated outside the trainer
  (scripts/ppo_checkpoint_eval.py: best by evaluation beside the final checkpoint). Record format unchanged.
* v2.6.4 PPO final (owner decision 2026-10-14; plan 46 config P1, scripts/ppo_final_g46.py) SUPERSEDES two G44
  parts: NO learning-rate floor (KLAdaptiveLR exactly as in the registration yaml; the G46 smoke switch
  USVB_G46_LR is removed and refused when set) and the failure penalty is the DISCOUNTED bound -(H x L + E)
  (H = min(ceil(1/(1-gamma)), N), L at the vessel's top speed); normaliser freeze at 5 %; checkpoints every
  2 % of the budget are the DEFAULT (CHECKPOINT_PARTS = 50; a campaign run refuses another interval); the
  untrained agent is saved to out/g46_init/agent_0.pt; every periodic checkpoint is evaluated on the validation
  seed and the best is selected outside the trainer (scripts/ppo_checkpoint_eval.py, G46.VALIDATION_RULE:
  campaign runs selection-02 / seed 11002 only). The record stamps agent_config.ppo_final.
* v2.6.4 (G47): an EMPTY USVBENCH_ASSETS_V2 is refused at preflight (G45/G46 S1 failure: an exported empty value
  made the env cfg resolve the vessel asset relative to the cwd; unset means <release>/assets_v2).
Exit codes: 0 completed, 2 refused (preflight), 3 infrastructure failure, 4 diverged. Follow-up M1 (the
evaluator_exit.py pattern; Isaac Sim 5.1's SimulationApp.close() ends the process with code 0): a non-zero code
ends the process with os._exit(code) WITHOUT close(); code 0 closes the app under a watchdog. Every exit of main()
goes through finish() (any exception before the run is exit 3, or 2 for a refusal). Launchers judge a run by its
training record (status), never by the exit code alone.
Seed registry (follow-up M4, verify2): EVERY training run (campaign 'training' and 'training_smoke') uses the
in-tree registry (scripts/seed_registry.json of the running tree, whose code-tree hash the release record pins);
a --seed-registry whose digest differs from it is refused at preflight for every purpose.
CPU only for --dry-run; the run itself needs the RE-1 GPU stack. No training happens in the CPU tests (they
drive run_training with a fake backend).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import sys
import threading
import time
import traceback
import types
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import code_tree  # noqa: E402
import ppo_training_record as R  # noqa: E402
import reference_environment  # noqa: E402
import seed_registry  # noqa: E402
import ppo_training_fixes as FX  # noqa: E402
import ppo_final_g46 as G46  # noqa: E402
from eval_protocol import OverrideError, resolve_journal_task  # noqa: E402

TRAINER_VERSION = "g19b-train-journal-ppo-v1"
DEFAULT_BUDGET = {"blueboat": 96000, "wamv": 192000}     # timesteps per env (TRAINING_DESIGN section 5 item 2)
CHECKPOINT_PARTS = G46.CHECKPOINT_PARTS                   # v2.6.4: checkpoints every 2 % of the budget (50)
EXIT_OK, EXIT_REFUSED, EXIT_INFRA, EXIT_DIVERGED = 0, 2, 3, 4
TRAINING_POOL_MODULE = "training_pool"
HEARTBEAT_SCHEMA = "usvbench.train_heartbeat/1"
PREFLIGHT_SCHEMA = "usvbench.train_preflight/1"
REFUSED_APP_PREFIXES = ("env.", "agent.", "+", "~")
IN_TREE_REGISTRY = SCRIPTS / "seed_registry.json"          # follow-up M4: the registry campaign training uses
CODE9 = 9                                                  # journal_contracts_v2 TERMINATION_TABLE non_finite_state
CODE9_RULE = ("follow-up C1: a row terminating with code 9 (non_finite_state, a legal episode outcome) has a "
              "non-finite reward computed from its non-finite state; the training wrapper replaces it by the "
              "family's terminal failure penalty (replacement_source) before the agent sees it; every replacement "
              "is counted; more than the alarm limit is a divergence; any other non-finite action, reward, "
              "observation, loss or parameter is a divergence")
OBSERVATION_CHECK_RULE = ("all rows every step into a device-side flag, read at every step with a terminating row "
                          "and at every iteration end (rollouts env steps); actions checked before every env.step")
CLOSE_WATCHDOG_S = 120                                     # as evaluator_exit.CLOSE_WATCHDOG_S
# G35 (J19): heartbeat replace retry (then skip, never fatal) and the appearing trigger-speed count
HEARTBEAT_REPLACE_ATTEMPTS = 5
HEARTBEAT_REPLACE_WAIT_S = 0.2
APPEARING_SPEED_FLAG = "trigger_speed_outside_envelope"     # = conditions/appearing/stamp.SPEED_ENVELOPE_FLAG
APPEARING_SPEED_RECORD = "trigger_speed_envelope"           # = conditions/appearing/stamp.SPEED_ENVELOPE_RECORD
APPEARING_SPEED_EXAMPLES = 20                               # first episodes listed in the record
APPEARING_SPEED_RULE = ("J19 (v2.6.1): an appearing trigger whose pose speed lies outside the vessel envelope is "
                        "recorded in the episode's stamp and counted here; never an infrastructure failure")


class TrainingRefusal(ValueError):
    """A preflight rule refuses the training run (exit 2, nothing written)."""


class Divergence(RuntimeError):
    """Non-finite training values: the run ends as 'diverged'."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ arguments
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="journal id or journal:<task>")
    p.add_argument("--vessel", required=True, choices=("blueboat", "wamv"))
    p.add_argument("--training-seed", type=int, required=True)
    p.add_argument("--purpose", choices=seed_registry.TRAINING_PURPOSES,
                   help="default: the purpose of the seed's registry namespace")
    p.add_argument("--arm", choices=R.ARMS, default="mixture",
                   help="mixture (option 2, campaign) or calm (calm-only control of the training smoke)")
    p.add_argument("--budget-timesteps", type=int, help="default BlueBoat 96000, WAM-V 192000 (per env)")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--checkpoint-every", type=int, help="default budget / 50 (2 %%; must divide the budget; "
                   "campaign runs refuse any other value)")
    p.add_argument("--calm-slots", type=int, default=3, help="owner decision O3 (pending): calm episodes per block")
    p.add_argument("--prefetch", action="store_true", help="admission prefetch thread (results identical)")
    p.add_argument("--out", type=Path, required=True, help="NEW run directory (refused if it exists)")
    p.add_argument("--release-record", type=Path, help="release record JSON (required for campaign seeds)")
    p.add_argument("--seed-registry", type=Path, help="registry file (default scripts/seed_registry.json)")
    p.add_argument("--host-class", default="unspecified", help="e.g. 5070, 4080, rental_4090 (stamped)")
    p.add_argument("--concurrency", type=int, default=1, help="training copies sharing the GPU (stamped)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true", help="preflight only (no simulator, nothing written)")
    p.add_argument("--preflight-out", type=Path, help="dry run: also write the preflight JSON here")
    p.add_argument("--agent-set", action="append", default=[],
                   help="REFUSED: agent hyperparameters are the registration yaml; only the stamped runtime "
                        "overrides are applied")
    return p


# ------------------------------------------------------------------ CPU loading (no family __init__)
_NAMESPACE = None


def namespace_tasks():
    """Private namespace package over tasks/ (no family __init__, no gym registration, no simulator)."""
    global _NAMESPACE
    if _NAMESPACE is None:
        root = "_g19b_train_tasks"
        if root not in sys.modules:
            package = types.ModuleType(root)
            package.__path__ = [str(REPO / "tasks")]
            sys.modules[root] = package
            for folder in (REPO / "tasks").iterdir():
                if folder.is_dir() and folder.name != "__pycache__":
                    sub = types.ModuleType(root + "." + folder.name)
                    sub.__path__ = [str(folder)]
                    sys.modules[root + "." + folder.name] = sub
        _NAMESPACE = root
    return _NAMESPACE


def task_modules(root):
    imp = lambda name: importlib.import_module(root + "." + name)  # noqa: E731
    return types.SimpleNamespace(journal_config=imp("_shared.conditions.journal_config"),
                                 mixture=imp("_shared.conditions.training_mixture"),
                                 appearing=imp("_shared.conditions.appearing.config"),
                                 hazards=imp("_shared.training_hazards"),
                                 plans=imp("_shared.journal_cfg_v2"),
                                 groups=imp("_shared.conditions.observation_groups"))


def family_pool_module(root, carrier):
    name = "%s.%s.%s" % (root, carrier, TRAINING_POOL_MODULE)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            raise TrainingRefusal("no training pool adapter for carrier %r (tasks/%s/%s.py, IMPLEMENTATION_PLAN "
                                  "P3)" % (carrier, carrier, TRAINING_POOL_MODULE)) from None
        raise


def plan_cfg(mods, task, vehicle, seed):
    """CPU stand-in of a journal cfg (the fields admission reads; same shape as the F0 glue fixture)."""
    plan = mods.plans.plan_class(task, vehicle)(2)
    params = plan.params
    cfg = types.SimpleNamespace(
        journal_task=task, journal_contract_version=2, journal_mode=True, journal_geometry_tier=2,
        journal_plan=plan, vehicle=vehicle, seed=int(seed), conditions=None, condition_presets=(),
        sim=types.SimpleNamespace(dt=1.0 / 120.0, gravity=(0.0, 0.0, -9.81)), decimation=2,
        underwater_physics_cfg=types.SimpleNamespace(physics_version=2, hydrodynamics_model="hydro_v2",
                                                     enable_current=False),
        robot_cfg=types.SimpleNamespace(spawn=types.SimpleNamespace(
            rigid_props=types.SimpleNamespace(disable_gravity=False))),
        obs_sea_state=False, appear_count=0, ray_count=int(params["ray_sensor_ray_count"]),
        ray_max_range_m=float(params["ray_sensor_max_range_m"]))
    setattr(cfg, mods.groups.HOOKS_FLAG, True)
    return cfg


# ------------------------------------------------------------------ agent yaml
def registration_agent_entry(task_id):
    """'<carrier>.agents:<yaml>' of a journal id, read from tasks/journal/__init__.py (AST; no import)."""
    import ast
    tree = ast.parse((REPO / "tasks" / "journal" / "__init__.py").read_text(encoding="ascii"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "register":
            kwargs = {k.arg: k.value for k in node.keywords}
            if isinstance(kwargs.get("id"), ast.Constant) and kwargs["id"].value == task_id:
                for key, value in zip(kwargs["kwargs"].keys, kwargs["kwargs"].values):
                    if isinstance(key, ast.Constant) and key.value == "skrl_cfg_entry_point":
                        tail = "".join(v.value for v in value.values if isinstance(v, ast.Constant))
                        return tail.lstrip(".")
    raise TrainingRefusal("journal id %s has no skrl_cfg_entry_point in tasks/journal" % task_id)


def load_agent_yaml(entry):
    import yaml
    module, _, name = entry.partition(":")
    carrier = module.split(".")[0]
    path = REPO / "tasks" / carrier / "agents" / name
    return yaml.safe_load(path.read_text(encoding="utf-8")), path


def agent_overrides(agent_cfg, *, seed, budget, directory, experiment_name, checkpoint_interval, write_interval):
    """Apply ONLY the stamped runtime overrides (R.ALLOWED_AGENT_OVERRIDES, in order); returns their list."""
    values = {"seed": int(seed), "trainer.timesteps": int(budget), "trainer.close_environment_at_exit": False,
              "agent.experiment.directory": str(directory),
              "agent.experiment.experiment_name": str(experiment_name),
              "agent.experiment.checkpoint_interval": int(checkpoint_interval),
              "agent.experiment.write_interval": int(write_interval), "agent.experiment.wandb": False}
    out = []
    for path in R.ALLOWED_AGENT_OVERRIDES:
        R.set_path(agent_cfg, path, values[path])
        out.append({"path": path, "value": values[path]})
    return out


# ------------------------------------------------------------------ preflight
def _release(args):
    if args.release_record is None:
        return None
    path = Path(args.release_record)
    if not path.is_file():
        raise TrainingRefusal("release record %s is missing" % path)
    data = path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    release_id = document.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise TrainingRefusal("release record %s carries no release_id" % path)
    tree = document.get("code_tree_sha256")
    return {"release_id": release_id, "record_path": str(path.resolve()),
            "record_sha256": hashlib.sha256(data).hexdigest(),
            "code_tree_sha256": tree if isinstance(tree, str) else None}


def preflight(args, *, root=None, registry=None, reference_block=None, code_tree_sha256=None):
    """Every refusal before the simulator; returns the JSON-native plan (raises TrainingRefusal)."""
    if args.agent_set:
        raise TrainingRefusal("agent hyperparameters are the registration yaml; --agent-set is refused (only the "
                              "stamped runtime overrides %s are applied)" % (R.ALLOWED_AGENT_OVERRIDES,))
    try:
        info = resolve_journal_task(args.task, args.vessel)
    except OverrideError as exc:
        raise TrainingRefusal(str(exc)) from None
    if info is None:
        raise TrainingRefusal("only journal ids train here (got %r); legacy ids keep their own tooling" % args.task)
    if registry is None and args.seed_registry is not None and not Path(args.seed_registry).is_file():
        raise TrainingRefusal("seed registry %s is missing" % args.seed_registry)
    try:
        registry = seed_registry.load_registry(args.seed_registry) if registry is None else registry
        owner = seed_registry.namespace_for_seed(args.training_seed, registry)
        purpose = args.purpose or (seed_registry.purpose_of(owner, registry) if owner else None)
        if purpose not in seed_registry.TRAINING_PURPOSES:
            raise seed_registry.RegistryError("seed %r is not a training seed (namespace %s)"
                                              % (args.training_seed, owner))
        stamp = seed_registry.resolve_training_block(args.training_seed, purpose, registry)
    except seed_registry.RegistryError as exc:
        raise TrainingRefusal(str(exc)) from None
    in_tree = seed_registry.load_registry(IN_TREE_REGISTRY)
    if seed_registry.registry_digest(registry) != seed_registry.registry_digest(in_tree):
        raise TrainingRefusal("training (every purpose, %s included) uses the in-tree seed registry (%s, sha256 %s) "
                              "whose code tree the release pins; the registry given has sha256 %s (follow-up M4, "
                              "verify2)" % (purpose, IN_TREE_REGISTRY, seed_registry.registry_digest(in_tree),
                                            seed_registry.registry_digest(registry)))
    vessel = info["vessel"]
    budget = int(args.budget_timesteps or DEFAULT_BUDGET[vessel])
    every = int(args.checkpoint_every or budget // CHECKPOINT_PARTS)
    if budget <= 0 or every <= 0 or budget % every:
        raise TrainingRefusal("checkpoint interval %d must divide the budget %d" % (every, budget))
    problem = G46.checkpoint_every_error(purpose, budget, every) or G46.assets_env_error()      # v2.6.4
    if problem:
        raise TrainingRefusal(problem)
    if args.num_envs < 1 or args.concurrency < 1:
        raise TrainingRefusal("num_envs and concurrency must be positive")
    if Path(args.out).exists():
        raise TrainingRefusal("output directory %s exists (never overwrite a run)" % args.out)
    release = _release(args)
    if purpose == "training" and release is None:
        raise TrainingRefusal("campaign training needs --release-record (the release it trains on)")
    block = reference_environment.reference_block() if reference_block is None else reference_block
    if purpose == "training" and not block.get("match"):
        raise TrainingRefusal("campaign training runs only in RE-1 (CONTEXT_R4 D28): %s"
                              % "; ".join(block.get("mismatches") or ["no match"]))
    tree_sha = code_tree.manifest_from_tree(REPO)[2] if code_tree_sha256 is None else code_tree_sha256
    if release is not None and release.get("code_tree_sha256") not in (None, tree_sha) and purpose == "training":
        raise TrainingRefusal("the release record pins code tree %s, this tree is %s (the registry and the code "
                              "must be the release's; follow-up M4)" % (release["code_tree_sha256"], tree_sha))
    root = namespace_tasks() if root is None else root
    mods = task_modules(root)
    task, vehicle, carrier = info["journal_task"], info["vehicle"], info["carrier"]
    if carrier not in R.CODE9_PENALTY_FIELD:
        raise TrainingRefusal("no declared code-9 replacement rule for carrier %r (ppo_training_record."
                              "CODE9_PENALTY_FIELD)" % carrier)
    pool_module = family_pool_module(root, carrier)
    wired = task in mods.appearing.WIRED_TASKS                    # read at run time (G20b extends it)
    mixture = None
    tokens = []
    if args.arm == "mixture":
        try:
            mixture = mods.mixture.TrainingMixture(seed=int(args.training_seed), task=task, vessel=vehicle,
                                                   appearing_wired=wired, calm_slots=int(args.calm_slots),
                                                   purpose=purpose)
        except mods.mixture.MixtureError as exc:
            raise TrainingRefusal(str(exc)) from None
        tokens = mixture.required_tokens()
    resolved = mods.journal_config.resolve_journal_conditions(presets=tokens, seed=int(args.training_seed))
    if mixture is not None:
        mods.mixture.check_stack(mixture, resolved)
    try:
        admission = mods.journal_config.admit_journal_conditions(plan_cfg(mods, task, vehicle, args.training_seed),
                                                                 resolved)
    except ValueError as exc:
        raise TrainingRefusal("the training stack does not admit on %s: %s" % (info["task_id"], exc)) from None
    entry = registration_agent_entry(info["task_id"])
    agent_cfg, agent_path = load_agent_yaml(entry)
    rollouts = int(agent_cfg["agent"]["rollouts"])
    if budget % rollouts:
        raise TrainingRefusal("the budget %d is not a whole number of %d-step iterations" % (budget, rollouts))
    planned = copy.deepcopy(agent_cfg)
    try:                                                             # v2.6.4 PPO final (KLAdaptiveLR only)
        g46 = G46.block(purpose, agent_cfg, vehicle, budget)
    except FX.FixError as exc:
        raise TrainingRefusal(str(exc)) from None
    plan_params = mods.plans.plan_class(task, vehicle)(2).params
    g44 = {"version": FX.G44_VERSION, "failure_codes": list(FX.FAILURE_CODES),
           "warmup_timesteps": int(math.ceil(FX.WARMUP_FRACTION * budget)),
           "lr_floor_fraction": FX.LR_FLOOR_FRACTION,
           "base_learning_rate": float(agent_cfg["agent"]["learning_rate"]),
           "plan_params": {k: plan_params.get(k) for k in ("length_scale", "gate_radius_m", "segment_count")}}
    out = Path(args.out).resolve()
    experiment = "ppo_%s_%s_s%d_%s" % (task, vessel, int(args.training_seed), args.arm)
    overrides = agent_overrides(planned, seed=args.training_seed, budget=budget, directory=out / "skrl",
                                experiment_name=experiment, checkpoint_interval=every,
                                write_interval=max(1, budget // 100))
    return {"schema": PREFLIGHT_SCHEMA, "trainer_version": TRAINER_VERSION, "utc": utc_now(),
            "task": task, "journal_id": info["task_id"], "vessel": vessel, "vehicle": vehicle, "carrier": carrier,
            "arm": args.arm, "training_seed": int(args.training_seed), "purpose": purpose, "registry": stamp,
            "release": release, "reference_environment": block, "re1_match": bool(block.get("match")),
            "code_tree_sha256": tree_sha, "budget_timesteps": budget, "checkpoint_every": every,
            "num_envs": int(args.num_envs), "rollouts": rollouts, "iterations": budget // rollouts,
            "calm_slots": int(args.calm_slots), "appearing_wired": wired, "stack_tokens": list(resolved.tokens()),
            "stack_parameters_sha256": resolved.stamp()["conditions_parameters_sha256"],
            "admission": json.loads(json.dumps(admission, default=str)),
            "mixture": None if mixture is None else mixture.stamp(),
            "agent": {"entry_point": entry, "path": agent_path.relative_to(REPO).as_posix(),
                      "yaml_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
                      "base_config_sha256": R.digest(agent_cfg), "planned_config_sha256": R.digest(planned),
                      "overrides": overrides, "experiment_name": experiment},
            "training_pool_module": pool_module.__name__.split(".", 1)[1],
            "prefetch": bool(args.prefetch), "host_class": args.host_class, "concurrency": int(args.concurrency),
            "out": str(out), "device": args.device,
            "code9": {"penalty_field": R.CODE9_PENALTY_FIELD[carrier], "rule": R.CODE9_REPLACEMENT_RULE,
                      "alarm": R.code9_alarm_block()},
            "in_tree_registry_sha256": seed_registry.registry_digest(in_tree), "g44": g44, "g46": g46}


# ------------------------------------------------------------------ run (backend-driven)
def code9_replacement(env_cfg, carrier):
    """C1 controller decision (REV-G19b verify2 item 4): the family's terminal failure penalty, read from the
    run's env cfg (the reward the family assigns to a collision termination); returns (value, source)."""
    if carrier not in R.CODE9_PENALTY_FIELD:
        raise RuntimeError("no declared code-9 replacement rule for carrier %r" % (carrier,))
    field = R.CODE9_PENALTY_FIELD[carrier]
    if field is None:
        if hasattr(env_cfg, "reward_contact_entry_penalty"):
            raise RuntimeError("carrier %s is declared without a termination penalty but its cfg carries "
                               "reward_contact_entry_penalty (CODE9_PENALTY_FIELD is stale)" % carrier)
        return 0.0, {"carrier": carrier, "field": None, "cfg_value": None, "value_rule": R.CODE9_NO_PENALTY_RULE}
    value = getattr(env_cfg, field, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError("env cfg %s.%s must be a positive finite penalty (got %r)" % (carrier, field, value))
    return -float(value), {"carrier": carrier, "field": field, "cfg_value": float(value),
                           "value_rule": "-env_cfg.%s" % field}


def install_g44(plan, env_cfg, base, agent):
    """G44: the failure penalty (from the run's env cfg and the plan params), the normaliser freeze and the
    learning-rate floor, attached to the skrl agent; raises FixError when a fix cannot be derived."""
    return G46.install(plan, env_cfg, base, agent)        # G46 P1: discounted penalty, freeze, LR log (no floor)


def atomic_write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    os.replace(tmp, path)


def retrying_atomic_write_json(path, value, attempts=HEARTBEAT_REPLACE_ATTEMPTS, wait_s=HEARTBEAT_REPLACE_WAIT_S,
                               sleep=time.sleep, write=None):
    """G35 (J19): atomic_write_json retried up to ``attempts`` times ``wait_s`` apart on OSError (a Windows reader
    holding the target makes os.replace fail with PermissionError); returns None on success, else the last error
    (the caller skips this write). Only the heartbeat uses it: every other file keeps atomic_write_json."""
    write = atomic_write_json if write is None else write
    attempts = max(1, int(attempts))
    error = None
    for attempt in range(attempts):
        try:
            write(path, value)
            return None
        except OSError as exc:
            error = exc
            if attempt + 1 < attempts:
                sleep(wait_s)
    return error


def host_block(torch_module=None):
    gpu = driver = None
    try:
        if torch_module is not None and torch_module.cuda.is_available():
            gpu = torch_module.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - host description is best effort
        gpu = None
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        driver = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:  # noqa: BLE001
        driver = None
    return {"gpu": gpu, "driver": driver, "cpu": platform.processor() or platform.machine(),
            "os": platform.platform(), "python": sys.version.split()[0],
            "hostname_sha256": hashlib.sha256(platform.node().encode("utf-8")).hexdigest()}


def _flat(value):
    try:
        return [float(v) for v in value.detach().reshape(-1).to("cpu").tolist()]
    except AttributeError:
        return [float(v) for v in value]


def _finite_state(obj):
    """True when every floating tensor / number in a (nested) checkpoint object is finite."""
    import torch
    if isinstance(obj, torch.Tensor):
        return (not obj.is_floating_point()) or bool(torch.isfinite(obj).all())
    if isinstance(obj, dict):
        return all(_finite_state(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite_state(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


def agent_state_problem(agent):
    """Name of the first agent state (checkpoint modules: models, optimizer, preprocessors) holding a non-finite
    value, or None. skrl registers every model, the optimizer and the preprocessors in checkpoint_modules."""
    modules = dict(getattr(agent, "checkpoint_modules", None) or {}) or dict(getattr(agent, "models", None) or {})
    for name, module in sorted(modules.items()):
        state = module.state_dict() if callable(getattr(module, "state_dict", None)) else None
        if state is not None and not _finite_state(state):
            return name
    return None


def _obs_tensor(obs):
    import torch
    if isinstance(obs, dict):
        parts = [v.reshape(v.shape[0], -1).float() for v in obs.values() if isinstance(v, torch.Tensor)]
        return torch.cat(parts, -1) if parts else None
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    return None


class StepMonitor:
    """Wraps the skrl-wrapped env's step: timestep count, heartbeat, progress, returns, code-9 rewards and the
    divergence checks (follow-up C1, M2, m7)."""

    def __init__(self, wrapped, base, plan, out, agent_getter, clock=time.monotonic):
        self.wrapped, self.base, self.plan, self.out = wrapped, base, plan, Path(out)
        self.agent_getter = agent_getter
        self.clock = clock
        self.budget, self.rollouts = plan["budget_timesteps"], plan["rollouts"]
        self.code9_reward = float(plan["code9_reward"])
        if not math.isfinite(self.code9_reward):
            raise ValueError("the code-9 replacement reward must be finite")
        self.code9_source = plan.get("code9_source")
        self.code9_rows = 0
        self.episodes_finished = 0
        self.last_obs = self.last_actions = None
        self._obs_bad = None
        self.code9_per_component = {}
        self.updates_checked = 0
        self.loss_values_checked = 0
        self.timestep = 0
        self.next_percent = 1
        self.start = clock()
        self.tracker = R.ReturnTracker(self.budget, plan["num_envs"])
        self.components = None
        self.heartbeats_skipped = 0                               # G35 (J19)
        self.heartbeat_sleep = time.sleep
        self.appearing_stamps_seen = 0
        self.appearing_speed_outside = 0
        self.appearing_speed_examples = []
        self.g44 = None                                          # G44: install_g44() result (None = off)
        self.original = wrapped.step
        self.progress = open(self.out / "progress.log", "a", encoding="ascii")
        wrapped.step = self.step

    def _layer(self):
        return getattr(self.base, "_conditions", None)

    def _current_components(self):
        layer = self._layer()
        n = self.plan["num_envs"]
        if layer is None or getattr(layer, "training_mixture", None) is None:
            return ["calm"] * n
        return [layer.training_component(i) for i in range(n)]

    def heartbeat(self, status="running"):
        error = retrying_atomic_write_json(self.out / "heartbeat.json", {
            "schema": HEARTBEAT_SCHEMA, "pid": os.getpid(), "utc": utc_now(), "status": status,
            "timestep": self.timestep, "budget_timesteps": self.budget,
            "percent": round(100.0 * self.timestep / self.budget, 3), "iteration": self.timestep // self.rollouts,
            "wall_s": round(self.clock() - self.start, 3)}, sleep=self.heartbeat_sleep)
        if error is not None:                                     # G35 (J19): skip, never kill the run
            self.heartbeats_skipped += 1
            print("train_journal_ppo: WARNING heartbeat skipped at timestep %d after %d attempts: %s: %s"
                  % (self.timestep, HEARTBEAT_REPLACE_ATTEMPTS, type(error).__name__, error), file=sys.stderr,
                  flush=True)

    def _count_appearing(self, dones):
        """G35 (J19): count the finished episodes' appearing stamps whose trigger speed lay outside the envelope
        (journal_dones latched them before the reset, after the stamp reader verified each one)."""
        stamps = getattr(self.base, "episode_appearing_hazards", None)
        if stamps is None:
            return
        for i, done in enumerate(dones):
            stamp = stamps[i] if done and i < len(stamps) else None
            if not isinstance(stamp, dict):
                continue
            self.appearing_stamps_seen += 1
            runtime = (stamp.get("payload") or {}).get("runtime") or {}
            if runtime.get(APPEARING_SPEED_FLAG) is True:
                self.appearing_speed_outside += 1
                if len(self.appearing_speed_examples) < APPEARING_SPEED_EXAMPLES:
                    record = runtime.get(APPEARING_SPEED_RECORD) or {}
                    component = self.components[i] if self.components is not None and i < len(self.components) \
                        else None
                    self.appearing_speed_examples.append({
                        "timestep": self.timestep, "env": i, "component": component,
                        "speed_mps": record.get("speed_mps"), "envelope_mps": record.get("envelope_mps")})

    # -- agent hooks (follow-up M2): every update, not skrl's periodically cleared tracking_data
    def attach_agent(self, agent):
        update = getattr(agent, "_update", None)
        track = getattr(agent, "track_data", None)
        if not callable(update) or not callable(track):
            raise RuntimeError("the skrl agent has no _update / track_data: the divergence hooks cannot be installed")
        monitor = self

        def checked_update(*args, **kwargs):
            result = update(*args, **kwargs)
            monitor.updates_checked += 1
            if monitor.g44 is not None and monitor.g44.get("floor") is not None:   # G44 R1b: lr per update
                monitor.g44["floor"].observe()
            where = agent_state_problem(agent)
            if where is not None:
                raise Divergence("non-finite agent state (%s) after update %d at timestep %d"
                                 % (where, monitor.updates_checked, monitor.timestep))
            return result

        def checked_track(tag, value, *args, **kwargs):
            if str(tag).startswith("Loss"):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = 0.0
                monitor.loss_values_checked += 1
                if not math.isfinite(number):
                    raise Divergence("non-finite %s at timestep %d" % (tag, monitor.timestep))
            return track(tag, value, *args, **kwargs)

        agent._update = checked_update
        agent.track_data = checked_track

    def _codes(self):
        codes = getattr(self.base, "termination_reason", None)            # journal_dones publisher (D9)
        if codes is None:
            codes = getattr(self.base, "episode_termination_reason", None)
        return None if codes is None else [int(v) for v in _flat(codes)]

    def step(self, actions):
        import torch
        if isinstance(actions, torch.Tensor) and actions.is_floating_point() and not bool(
                torch.isfinite(actions).all()):                                 # verify2 F: before env.step
            raise Divergence("non-finite action at timestep %d (checked before env.step)" % self.timestep)
        self.last_actions = actions
        if self.components is None:
            self.components = self._current_components()
        if self.g44 is not None and self.g44["failure"].values is None:
            self.g44["failure"].refresh()                          # G44: penalties of the first episodes
        out = self.original(actions)
        obs, rew, terminated, truncated = out[0], out[1], out[2], out[3]
        rewards = _flat(rew)
        dones = [bool(a or b) for a, b in zip(_flat(terminated), _flat(truncated))]
        codes = self._codes()
        bad = [i for i, v in enumerate(rewards) if not math.isfinite(v)]
        code9 = [i for i in bad if dones[i] and codes is not None and codes[i] == CODE9]
        other = [i for i in bad if i not in code9]
        if other:
            raise Divergence("non-finite reward on env rows %s (not a code-9 termination) at timestep %d"
                             % (other, self.timestep))
        if code9:
            mask = torch.zeros(len(rewards), dtype=torch.bool)
            mask[code9] = True
            mask = mask.to(rew.device).reshape(rew.shape)
            rew = torch.where(mask, torch.full_like(rew, self.code9_reward), rew)
            out = (out[0], rew) + tuple(out[2:])
            for i in code9:
                rewards[i] = self.code9_reward
                self.code9_rows += 1
                name = self.components[i]
                self.code9_per_component[name] = self.code9_per_component.get(name, 0) + 1
        if self.g44 is not None:                                  # G44 R2: failure terminal penalty
            penalties = self.g44["failure"].penalties(dones, codes)
            if penalties:
                add = torch.zeros(len(rewards), dtype=rew.dtype)
                for i, value in penalties.items():
                    add[i] = value
                    rewards[i] += value
                rew = rew + add.to(rew.device).reshape(rew.shape)
                out = (out[0], rew) + tuple(out[2:])
            if any(dones):                                        # the new episodes' bounds (auto-reset)
                self.g44["failure"].refresh([i for i, d in enumerate(dones) if d])
        self.episodes_finished += sum(1 for d in dones if d)
        if any(dones):
            self._count_appearing(dones)                          # G35 (J19)
        if code9:
            limit = R.code9_alarm_limit(self.episodes_finished)
            if self.code9_rows > limit:
                raise Divergence("code-9 alarm: %d code-9 (non_finite_state) terminations in %d finished episodes "
                                 "exceed the limit %s = %.4g at timestep %d"
                                 % (self.code9_rows, self.episodes_finished, R.CODE9_ALARM_RULE, limit,
                                    self.timestep))
        flat_obs = _obs_tensor(obs)
        if flat_obs is not None:
            self.last_obs = flat_obs
            bad = ~torch.isfinite(flat_obs).all()                     # device scalar: no host sync here (m7)
            self._obs_bad = bad if self._obs_bad is None else (self._obs_bad | bad)
            if ((self.timestep + 1) % self.rollouts == 0 or any(dones)) and bool(self._obs_bad):
                raise Divergence("non-finite observation at timestep %d (all rows since the last check)"
                                 % self.timestep)
        self.timestep += 1
        if self.g44 is not None and self.g44.get("freeze") is not None:   # G44 R1a: freeze after warm-up
            self.g44["freeze"].maybe_freeze(self.timestep)
        self.tracker.step(self.timestep - 1, rewards, dones, self.components, codes)
        if any(dones):
            now = self._current_components()
            self.components = [now[i] if d else c for i, (c, d) in enumerate(zip(self.components, dones))]
        if self.timestep % self.rollouts == 0:
            self.heartbeat()
        while self.next_percent <= 100 and self.timestep * 100 >= self.next_percent * self.budget:
            wall = self.clock() - self.start
            line = "PROGRESS %3d%% timestep %d/%d iteration %d wall %.1f s eta %.1f s" % (
                self.next_percent, self.timestep, self.budget, self.timestep // self.rollouts, wall,
                wall * (self.budget - self.timestep) / max(1, self.timestep))
            print(line, flush=True)
            self.progress.write(line + "\n")
            self.progress.flush()
            self.next_percent += 1
        return out

    def last_inputs_problem(self):
        """verify2 M2 residual: 'observation' / 'action' when the last observations / actions hold a non-finite
        value (read after a trainer exception with finite agent state), else None."""
        import torch
        for name, value in (("observation", self.last_obs), ("action", self.last_actions)):
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                return name
        return None

    def nonfinite_terminal(self):
        return {"rule": CODE9_RULE, "termination_code": CODE9, "replacement_reward": self.code9_reward,
                "replacement_source": self.code9_source, "rows_replaced": self.code9_rows,
                "episodes_finished": self.episodes_finished, "alarm": R.code9_alarm_block(),
                "per_component": dict(sorted(self.code9_per_component.items()))}

    def divergence_checks(self):
        return {"updates_checked": self.updates_checked, "loss_values_checked": self.loss_values_checked,
                "observation_check": OBSERVATION_CHECK_RULE,
                "agent_state": "models, optimizer and preprocessor state after every update and after any trainer "
                               "exception",
                # G35 (J19): non-fatal findings, counted
                "heartbeats_skipped": self.heartbeats_skipped,
                "appearing_trigger_speed": {"rule": APPEARING_SPEED_RULE, "stamps_seen": self.appearing_stamps_seen,
                                            "outside_envelope": self.appearing_speed_outside,
                                            "examples": list(self.appearing_speed_examples)}}

    def close(self):
        self.wrapped.step = self.original
        if not self.progress.closed:
            self.progress.close()


def check_attached(base, mixture, pools):
    """Follow-up m5: the env must carry the run's mixture and adapters (fail loudly otherwise)."""
    layer = getattr(base, "_conditions", None)
    if layer is None:
        raise RuntimeError("the env has no condition layer (_conditions): not a journal env")
    if getattr(layer, "training_mixture", None) is not mixture:
        raise RuntimeError("the env's condition layer does not carry the run's training mixture")
    if getattr(base, "_d7_pool", None) is not pools["d7"]:
        raise RuntimeError("the env does not use the run's training D7 pool")
    if pools["appearing"] is not None and getattr(base, "_appearing_pool", None) is not pools["appearing"]:
        raise RuntimeError("the env does not use the run's training appearing pool")


def _infrastructure_failure(out, plan, error, started, clock_start):
    out = Path(out)
    if out.is_dir():
        atomic_write_json(out / "STATUS.json", {
            "schema": "usvbench.train_status/1", "status": "infrastructure_failure", "utc": utc_now(),
            "started_utc": started, "wall_time_s": round(time.monotonic() - clock_start, 3),
            "error": "%s: %s" % (type(error).__name__, error), "journal_id": plan["journal_id"],
            "training_seed": plan["training_seed"], "rule": ("not a learning outcome: rerun the SAME seed "
                                                            "(seed_registry infrastructure_failure_policy)")})
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    print("train_journal_ppo: INFRASTRUCTURE FAILURE: %s: %s" % (type(error).__name__, error), file=sys.stderr)
    return EXIT_INFRA


def run_training(plan, args, backend):
    """The run (see the module docstring). ``backend`` supplies package_root, parse_env_cfg, load_agent_cfg,
    make_env, wrap, make_runner, torch; returns the exit code."""
    out = Path(plan["out"])
    started, clock_start = utc_now(), time.monotonic()
    out.mkdir(parents=True, exist_ok=False)                     # never overwrite a run
    atomic_write_json(out / "preflight.json", plan)               # manifest first
    monitor = cache = None
    status, divergence = "completed", None
    try:
        registry = seed_registry.load_registry(getattr(args, "seed_registry", None))
        if seed_registry.registry_digest(registry) != plan["registry"]["registry_sha256"]:
            raise RuntimeError("the seed registry changed between the preflight and the run")
        mods = task_modules(backend.package_root)
        pool_module = family_pool_module(backend.package_root, plan["carrier"])
        env_cfg = backend.parse_env_cfg(plan["journal_id"], plan["device"], plan["num_envs"])
        env_cfg.seed = int(plan["training_seed"])
        mixture = None
        tokens = []
        if plan["arm"] == "mixture":
            mixture = mods.mixture.TrainingMixture(seed=plan["training_seed"], task=plan["task"],
                                                   vessel=plan["vehicle"], appearing_wired=plan["appearing_wired"],
                                                   calm_slots=plan["calm_slots"], purpose=plan["purpose"])
            if mixture.digest() != plan["mixture"]["mixture_sha256"]:
                raise RuntimeError("the run's mixture differs from the preflight mixture")
            tokens = mixture.required_tokens()
        resolved = mods.journal_config.resolve_journal_conditions(presets=tokens, seed=plan["training_seed"])
        mods.journal_config.admit_journal_conditions(env_cfg, resolved)
        mods.journal_config.apply_journal_conditions(env_cfg, resolved)
        if mixture is not None:
            env_cfg.journal_training_mixture = mixture
        cache = mods.hazards.TrainingHazardCache(prefetch=plan["prefetch"])
        pools = pool_module.training_pools(env_cfg, mixture, cache=cache, purpose=plan["purpose"],
                                           registry=registry, registry_stamp=plan["registry"],
                                           num_envs=plan["num_envs"])
        env_cfg.journal_d7_pool = pools["d7"]
        if pools["appearing"] is not None:
            env_cfg.journal_appearing_pool = pools["appearing"]
        agent_cfg = backend.load_agent_cfg(plan["journal_id"])
        if R.digest(agent_cfg) != plan["agent"]["base_config_sha256"]:
            raise RuntimeError("the registry agent config differs from the preflight yaml")
        overrides = agent_overrides(agent_cfg, seed=plan["training_seed"], budget=plan["budget_timesteps"],
                                    directory=out / "skrl", experiment_name=plan["agent"]["experiment_name"],
                                    checkpoint_interval=plan["checkpoint_every"],
                                    write_interval=max(1, plan["budget_timesteps"] // 100))
        problem = G46.removed_switch_error() or G46.assets_env_error()     # v2.6.4: as at preflight
        if problem:
            raise RuntimeError(problem)
        if overrides != plan["agent"]["overrides"] or R.digest(agent_cfg) != plan["agent"]["planned_config_sha256"]:
            raise RuntimeError("the applied agent overrides differ from the preflight plan")
        env = backend.make_env(plan["journal_id"], env_cfg)
        base = getattr(env, "unwrapped", env)
        check_attached(base, mixture, pools)                                 # follow-up m5
        code9_value, code9_source = code9_replacement(env_cfg, plan["carrier"])       # verify2 C1
        wrapped = backend.wrap(env)
        holder = {}
        monitor = StepMonitor(wrapped, base, dict(plan, code9_reward=code9_value, code9_source=code9_source), out,
                              lambda: getattr(holder.get("runner"), "agent", None))
        monitor.heartbeat("starting")
        agent_snapshot = json.loads(R.canonical(agent_cfg))     # the stamped config (skrl may rewrite its copy)
        runner = backend.make_runner(wrapped, copy.deepcopy(agent_cfg))
        holder["runner"] = runner
        monitor.attach_agent(runner.agent)                                   # follow-up M2
        g44 = install_g44(plan, env_cfg, base, runner.agent)                 # G44 R1a + G46 P1
        monitor.g44 = g44
        g46_init = G46.save_init(runner.agent, out, backend.torch)           # G46: the untrained agent
        try:
            runner.run("train")
        except Divergence as exc:
            status, divergence = "diverged", {"reason": str(exc), "timestep": monitor.timestep}
        except Exception as exc:  # noqa: BLE001 - a NaN policy surfaces as e.g. a torch ValueError in act()
            where = agent_state_problem(runner.agent)
            if where is not None:
                reason = "non-finite agent state (%s) after %s: %s" % (where, type(exc).__name__, str(exc)[:300])
            else:
                inputs = monitor.last_inputs_problem()                       # verify2 M2 residual
                if inputs is None:
                    raise
                reason = "non-finite last %s after %s: %s" % (inputs, type(exc).__name__, str(exc)[:300])
            status, divergence = "diverged", {"reason": reason, "timestep": monitor.timestep}
        completed = monitor.timestep
        monitor.heartbeat("finishing")
        monitor.close()
        fix = G46.sidecar(plan["g46"], g44, g46_init)                        # v2.6.4 evidence (log_digests)
        atomic_write_json(out / "ppo_fix.json", fix)
        if status == "completed" and not G46.effective(fix):
            raise RuntimeError("v2.6.4 PPO fix not effective: %s" % FX.canonical(
                {"normaliser_freeze": fix["normaliser_freeze"], "learning_rate_log": fix["learning_rate_log"]}))
        run_dir = out / "skrl" / plan["agent"]["experiment_name"]
        checkpoints = sorted(run_dir.glob("checkpoints/agent_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
        checkpoints = [p for p in checkpoints if p.stem.split("_")[1].isdigit()]
        entries = [{"timestep": int(p.stem.split("_")[1]), "path": p.relative_to(out).as_posix(),
                    "sha256": R.file_sha256(p)} for p in checkpoints]
        final = None
        if status == "completed":
            if completed != plan["budget_timesteps"]:
                raise RuntimeError("the trainer stopped at timestep %d of %d" % (completed, plan["budget_timesteps"]))
            source = run_dir / "checkpoints" / ("agent_%d.pt" % plan["budget_timesteps"])
            if not source.is_file():
                raise RuntimeError("final checkpoint %s is missing (last-checkpoint rule)" % source)
            try:                                        # our own checkpoint (skrl modules dict)
                state = backend.torch.load(str(source), map_location="cpu", weights_only=False)
            except TypeError:
                state = backend.torch.load(str(source), map_location="cpu")
            if not _finite_state(state):
                status, divergence = "diverged", {"reason": "non-finite parameters in the final checkpoint",
                                                  "timestep": completed}
            else:
                shutil.copyfile(source, out / "final.pt")
                final = {"path": "final.pt", "sha256": R.file_sha256(out / "final.pt"),
                         "source_path": source.relative_to(out).as_posix(), "source_sha256": R.file_sha256(source)}
        layer = getattr(base, "_conditions", None)
        counts = layer.training_mixture_counts() if (layer is not None and mixture is not None) else None
        logs = {}
        for path in sorted(list(run_dir.glob("**/events.out.tfevents.*")) + [out / "progress.log",
                                                                           out / "heartbeat.json",
                                                                           out / "ppo_fix.json"]):
            if path.is_file():
                logs[path.relative_to(out).as_posix()] = R.file_sha256(path)
        record = {
            "schema": R.SCHEMA, "task": plan["task"], "journal_id": plan["journal_id"], "vessel": plan["vessel"],
            "vehicle": plan["vehicle"], "carrier_family": plan["carrier"], "arm": plan["arm"],
            "training_seed": plan["training_seed"], "candidate_index": plan["registry"].get("candidate_index"),
            "purpose": plan["purpose"], "registry": plan["registry"], "release": plan["release"],
            "code_tree_sha256": plan["code_tree_sha256"], "reference_environment": plan["reference_environment"],
            "re1_match": plan["re1_match"], "host": host_block(getattr(backend, "torch", None)),
            "host_class": plan["host_class"], "concurrency": plan["concurrency"],
            "preflight_sha256": R.file_sha256(out / "preflight.json"),
            "agent_config": {"entry_point": plan["agent"]["entry_point"], "yaml_path": plan["agent"]["path"],
                             "yaml_sha256": plan["agent"]["yaml_sha256"],
                             "base_config_sha256": plan["agent"]["base_config_sha256"], "config": agent_snapshot,
                             "sha256": R.digest(agent_snapshot), "overrides": overrides,
                             "ppo_final": {"version": plan["g46"]["version"], "lr_mode": plan["g46"]["lr_mode"],
                                           "lr_floor": None, "checkpoint_every": plan["checkpoint_every"],
                                           "validation_rule": plan["g46"]["validation_rule"]}},
            "budget_timesteps": plan["budget_timesteps"], "completed_timesteps": completed,
            "num_envs": plan["num_envs"], "rollouts": plan["rollouts"], "iterations": completed // plan["rollouts"],
            "mixture": None if mixture is None else mixture.stamp(counts or {}),
            "hazard_admission": cache.admission_log(), "checkpoints": entries, "final_checkpoint": final,
            "status": status, "divergence": divergence, "nonfinite_terminal": monitor.nonfinite_terminal(),
            "divergence_checks": monitor.divergence_checks(),
            "wall_time_s": round(time.monotonic() - clock_start, 3),
            "started_utc": started, "finished_utc": utc_now(), "training_returns": monitor.tracker.summary(),
            "log_digests": logs}
        sha = R.write_record(record, out / "training_record.json")       # LAST
        print("train_journal_ppo: %s, record %s (sha256 %s)" % (status.upper(), out / "training_record.json", sha),
              flush=True)
        return EXIT_OK if status == "completed" else EXIT_DIVERGED
    except BaseException as exc:  # noqa: BLE001 - everything else is an infrastructure failure (exit 3)
        if monitor is not None:
            try:
                monitor.close()
            except Exception:  # noqa: BLE001
                pass
        return _infrastructure_failure(out, plan, exc, started, clock_start)
    finally:
        if cache is not None:
            cache.close()


# ------------------------------------------------------------------ Isaac Lab backend
def app_arguments(app_argv, device):
    """AppLauncher arguments: Hydra / agent tokens refused; --device passed through (follow-up m6)."""
    for token in app_argv:
        if token.startswith(REFUSED_APP_PREFIXES) or "=" in token and token.split("=", 1)[0].startswith(
                ("env.", "agent.")):
            raise TrainingRefusal("Hydra / agent overrides are refused (%r); only the stamped overrides apply" % token)
    app_argv = list(app_argv)
    given = [t for t in app_argv if t == "--device" or t.startswith("--device=")]
    if not given:
        app_argv += ["--device", str(device)]
    return app_argv


_LAUNCHED = {}                                              # the app once AppLauncher started it (verify2 M1)


def launched_backend():
    """A finish()-able stand-in when the app started but the backend was never returned (import failure)."""
    return types.SimpleNamespace(app=_LAUNCHED["app"]) if _LAUNCHED.get("app") is not None else None


def isaac_backend(app_argv, device):
    app_argv = app_arguments(app_argv, device)
    from isaaclab.app import AppLauncher
    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(app_argv)
    app = AppLauncher(app_args).app
    _LAUNCHED["app"] = app
    import gymnasium as gym
    import torch
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from skrl.utils.runner.torch import Runner
    from eval_protocol import register_journal_tasks
    module = register_journal_tasks()
    root = module.__name__.rsplit(".", 1)[0]
    return types.SimpleNamespace(
        app=app, torch=torch, package_root=root,
        parse_env_cfg=lambda task, dev, n: parse_env_cfg(task, device=dev, num_envs=n),
        load_agent_cfg=lambda task: load_cfg_from_registry(task, "skrl_cfg_entry_point"),
        make_env=lambda task, cfg: gym.make(task, cfg=cfg, render_mode=None),
        wrap=lambda env: SkrlVecEnvWrapper(env, ml_framework="torch"),
        make_runner=lambda wrapped, cfg: Runner(wrapped, cfg))


def finish(code, backend, exit_fn=os._exit, timer_cls=threading.Timer):
    """Follow-up M1 (scripts/evaluator_exit.py pattern). Without an app (CPU tests): return the code. With an app:
    a non-zero code ends the process with os._exit(code) WITHOUT close() (Kit's close() ends the process with 0
    or hangs); code 0 closes the app under a watchdog (CLOSE_WATCHDOG_S) and then ends with 0."""
    app = getattr(backend, "app", None) if backend is not None else None
    if app is None:
        return code
    sys.stdout.flush()
    sys.stderr.flush()
    if code != 0:
        exit_fn(int(code))
        return code
    watchdog = timer_cls(CLOSE_WATCHDOG_S, lambda: exit_fn(0))
    watchdog.daemon = True
    watchdog.start()
    try:
        app.close()
    except BaseException:  # noqa: BLE001 - the record is written; close() problems never change the outcome
        traceback.print_exc()
    exit_fn(0)
    return code


def main(argv=None, backend=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    try:
        if args.dry_run:
            if rest:
                raise TrainingRefusal("unknown arguments in a dry run: %s" % " ".join(rest))
            plan = preflight(args)                     # no simulator: private namespace over tasks/
            text = json.dumps(plan, indent=1, sort_keys=True, allow_nan=False)
            if args.preflight_out is not None:
                Path(args.preflight_out).write_text(text + "\n", encoding="ascii")
            print(text)
            print("train_journal_ppo: DRY RUN OK %s %s seed %d (%s, arm %s)" % (
                plan["journal_id"], plan["vessel"], plan["training_seed"], plan["purpose"], plan["arm"]))
            return EXIT_OK
        if Path(args.out).exists():                    # cheap refusal before the app starts
            raise TrainingRefusal("output directory %s exists (never overwrite a run)" % args.out)
        if args.agent_set:
            raise TrainingRefusal("--agent-set is refused (agent hyperparameters are the registration yaml)")
        if backend is None:
            # the app first (Isaac Lab imports simulator-dependent modules only after AppLauncher), then the SAME
            # preflight on the real task package, before any output is written
            backend = isaac_backend(rest, args.device)
        elif rest:
            raise TrainingRefusal("unknown arguments: %s" % " ".join(rest))
        plan = preflight(args, root=backend.package_root)
    except (TrainingRefusal, ValueError) as exc:
        print("train_journal_ppo: REFUSED: %s" % exc, file=sys.stderr)
        return finish(EXIT_REFUSED, backend if backend is not None else launched_backend())
    except SystemExit as exc:                                   # e.g. AppLauncher's argparse (verify2 M1)
        code = exc.code if isinstance(exc.code, int) else EXIT_REFUSED
        print("train_journal_ppo: REFUSED: exit %r before the run" % (exc.code,), file=sys.stderr)
        return finish(code, backend if backend is not None else launched_backend())
    except BaseException as exc:  # noqa: BLE001 - verify2 M1: nothing escapes main() without finish()
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        print("train_journal_ppo: INFRASTRUCTURE FAILURE before the run: %s: %s" % (type(exc).__name__, exc),
              file=sys.stderr)
        return finish(EXIT_INFRA, backend if backend is not None else launched_backend())
    try:
        code = run_training(plan, args, backend)
    except BaseException as exc:  # noqa: BLE001 - run_training reports its own failures; this is a last guard
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        print("train_journal_ppo: INFRASTRUCTURE FAILURE: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        code = EXIT_INFRA
    return finish(code, backend)


if __name__ == "__main__":
    raise SystemExit(main())
