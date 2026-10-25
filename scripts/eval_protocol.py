"""One fail-closed evaluation configuration contract. No simulator imports.

Used by every certificate writer (eval_v6_frozen, eval_mppi, classical_planner_pid,
classical_baseline, eval_imbalance, eval_obs_bridge, eval_composite,
eval_hazard_nav, eval_cross_champion, convergence_curve) and by run_cell.py.

Precedence (finding 53), identical for every evaluator:
    task-cfg defaults  <  dedicated convenience flags  <  --set FIELD=VALUE
A dedicated flag and a --set naming the same field with different values is
rejected, never silently resolved. Protocol fields (seed, eval_level,
curriculum_frozen, scene.num_envs, sim.device) are fixed by the evaluator's own
arguments; a --set that disagrees is rejected. 'vehicle' can never be changed
by --set (finding 46: the asset/hull are built at cfg construction).

Ambient environment variables (finding 61): every name the task tree reads via
os.environ/getenv (scanned from source, plus a frozen list for deployed
layouts) and every USVBENCH_/OBS_/WAVE_/SPAWN_ variable is refused unless
allowed with --allow-env NAME; allowed values are stamped. Operational runtime
selectors (RUNTIME_ENV: CUDA_VISIBLE_DEVICES, thread counts, and since E1 the
rental-kit variables OMNI_KIT_ACCEPT_EULA, WANDB_MODE, PYTHONDONTWRITEBYTECODE,
USVBENCH_ASSETS, VK_ICD_FILENAMES and USVBENCH_RELEASE_DESCRIPTOR; since F7 round 4b
the D19 pool/asset locations USVBENCH_D7_POOL_DIR, USVBENCH_HARBOUR_D7_POOL and
USVBENCH_ASSETS_V2) are always stamped and never refused. The families' pool-path cfg
fields (cfg_override.JOURNAL_RUNTIME_FIELDS) are likewise runtime on journal cfgs:
accepted by --set, stamped in header.runtime_paths, never science literals.

Release identity (E1): with --release-descriptor or $USVBENCH_RELEASE_DESCRIPTOR
the deployed tasks/ and scripts/ must equal the release manifest before launch
(code_tree.py); the certificate then stamps release_id and code_tree_sha256.

Episode budgets must be divisible by the environment count (finding 57).

Journal ids (F7, CONTEXT_R4 D10/D12): ``select_task_id`` resolves a journal id
(``Isaac-USV-<Task>-<BlueBoat|WAMVVRX>-Journal-Direct-v1``) or the vessel-neutral
alias ``journal:<task>`` + ``--vessel`` to the VESSEL-SPECIFIC journal id before
any cfg is parsed (a mismatching --vessel is refused); run_cell maps the
registration directory ``journal`` to the real carrier family through
journal_cfg_v2.TASK_TABLE. For journal cfgs (journal_contract_version == 2) every
D12-protected field is refused by --set (cfg_override.JOURNAL_PROTECTED, checked
before launch and again on the cfg) and ``journal_validate`` runs after the
protocol batch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

from cfg_override import (JOURNAL_RUNTIME_FIELDS, OverrideError, apply_overrides, check_journal_assignments,
                          coerce_like, is_journal_cfg, journal_validate, resolve_owner, split_assignment,
                          split_runtime_assignments)
import code_tree
import reference_environment
from fixed_grid import episode_quota

VERSION = "d8-evaluation-v1"
# G19b (IMPLEMENTATION_PLAN P4): training-only cfg content that no evaluator may carry. The training entry point
# (scripts/train_journal_ppo.py) sets these in-memory objects; a layer with a mixture also refuses
# certificate_stamp (conditions/hydro_v2_layer.py), so a certificate can never come from a training env.
TRAINING_ONLY_CFG_FIELDS = ("journal_training_mixture",)
TRAINING_POOL_CFG_FIELDS = ("journal_d7_pool", "journal_appearing_pool")
# Operational variables: stamped in header.runtime_environment, never refused.
# E1 adds the rental-kit environment (D10 install.sh usvb_env.sh: Kit EULA,
# W&B off, bytecode off, asset root, kit-local Vulkan ICD; D10 FINAL_REPORT
# decision 6) and the release descriptor selector (code_tree.py).
RUNTIME_ENV = ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS", "PYTHONHASHSEED", "NVIDIA_TF32_OVERRIDE",
               "OMNI_KIT_ACCEPT_EULA", "WANDB_MODE", "PYTHONDONTWRITEBYTECODE", "USVBENCH_ASSETS",
               "VK_ICD_FILENAMES", "USVBENCH_RELEASE_DESCRIPTOR",
               # F7 round 4b (D19): the families' D7 pool / v2 asset locations are
               # runtime selectors; the pinned digests are checked by the envs and
               # stamped in the certificate (dynamic_hazards_header).
               "USVBENCH_D7_POOL_DIR", "USVBENCH_HARBOUR_D7_POOL", "USVBENCH_ASSETS_V2")
SCIENTIFIC_PREFIXES = ("USVBENCH_", "OBS_", "WAVE_", "SPAWN_")
# Literal os.environ/getenv reads in the 2026-09-30 task tree (non-test files),
# kept here so a deployed layout without a readable tasks/ tree still refuses
# them. The live scan below adds any new name.
KNOWN_SCIENTIFIC_ENV = (
    "ACTION_DELAY", "ALIVE_DIRECTIONAL", "AXIS_FLIP", "BACKWARD_COST", "CONE_ANNEAL_STEPS",
    "DIST_PENALTY_COEF", "DIST_SPEED_COEF", "DIST_SPEED_D0", "EARLYSTOP_AFTER",
    "EARLYSTOP_MIN_TARGETS", "EARLYSTOP_PATIENCE", "FORWARD_TRANSIT", "FWD_TRANSIT_COEF",
    "GOAL_RADIUS_OVERRIDE", "MAX_LIN_VEL", "OBS_DIM", "OBS_EXTENDED", "OBS_MODE",
    "OOB_PENALTY", "PHYS_CLEAN", "PROGRESS_COEF", "REACH_BONUS", "REWARD_VARIANT",
    "SIDE_APPROACH", "SPAWN_CONE_DEG", "SPEED_COUPLE", "USVBENCH_ASSETS",
    "USVBENCH_WATER_USD", "WAVE_COEF", "WAVE_MODE",
)
PROTECTED_FORBIDDEN = ("vehicle",)
_READ = re.compile(r"(?:environ\.get|environ\.setdefault|environ\.pop|getenv)\(\s*['\"]([^'\"]+)['\"]"
                   r"|environ\s*\[\s*['\"]([^'\"]+)['\"]\s*\]")


def tasks_root():
    return Path(__file__).resolve().parents[1] / "tasks"


# ---------------------------------------------------------------- journal ids (F7)
JOURNAL_TASK_ALIAS = "journal:"
JOURNAL_ID_MARKER = "-Journal-"
_JOURNAL_PLANS = {}


def journal_plans(root=None):
    """tasks/_shared/journal_cfg_v2 loaded by path (numpy only; no torch, no simulator).

    A private package over <tasks root>/_shared is used so that neither the
    ``tasks`` family initialisers (gym registration) nor tasks/_shared/__init__
    run; journal_cfg_v2 reads the D5 registry through its own relative import.
    """
    import importlib
    import sys
    import types
    shared = ((tasks_root() if root is None else Path(root)) / "_shared").resolve()
    key = str(shared)
    if key not in _JOURNAL_PLANS:
        private = "_f7_protocol_shared_%d" % len(_JOURNAL_PLANS)
        package = types.ModuleType(private)
        package.__path__ = [key]
        sys.modules[private] = package
        _JOURNAL_PLANS[key] = importlib.import_module(private + ".journal_cfg_v2")
    return _JOURNAL_PLANS[key]


def _vessel_label(vehicle):
    return "wamv" if str(vehicle).lower().startswith("wamv") else "blueboat"


def resolve_journal_task(task_id, vessel=None, root=None):
    """{task_id, journal_task, vehicle, vessel, carrier} of a journal id or alias, else None.

    ``journal:<task>`` needs ``vessel`` ('blueboat' -> journal_cfg_v2.BLUEBOAT_PROFILE,
    'wamv' -> DEFAULT_WAMV_PROFILE). A concrete journal id fixes its vessel; a
    different ``vessel`` is refused (the cfg class, hence the asset, is chosen
    by the id, C1 finding 46 / D12).
    """
    if not task_id:
        return None
    text = str(task_id)
    if not (text.startswith(JOURNAL_TASK_ALIAS) or JOURNAL_ID_MARKER in text):
        return None
    plans = journal_plans(root)
    if text.startswith(JOURNAL_TASK_ALIAS):
        name = text[len(JOURNAL_TASK_ALIAS):]
        if name not in plans.TASKS:
            raise OverrideError("unknown journal task %r (one of %s)" % (name, ", ".join(plans.TASKS)))
        if vessel not in ("blueboat", "wamv"):
            raise OverrideError("journal:<task> needs --vessel blueboat or wamv to pick the journal id")
        key = plans.BLUEBOAT_PROFILE if vessel == "blueboat" else plans.DEFAULT_WAMV_PROFILE
        concrete = plans.JOURNAL_ENV_IDS[(name, key)]
    else:
        reverse = {value: pair for pair, value in plans.JOURNAL_ENV_IDS.items()}
        if text not in reverse:
            raise OverrideError("unknown journal id %r (not in journal_cfg_v2.JOURNAL_ENV_IDS)" % text)
        (name, key), concrete = reverse[text], text
    label = _vessel_label(key)
    if vessel is not None and vessel != label:
        other = plans.BLUEBOAT_PROFILE if vessel == "blueboat" else plans.DEFAULT_WAMV_PROFILE
        raise OverrideError("journal id %s is the %s cfg; --vessel %s selects %s" % (
            concrete, label, vessel, plans.JOURNAL_ENV_IDS[(name, other)]))
    return {"task_id": concrete, "journal_task": name, "vehicle": key, "vessel": label,
            "carrier": plans.TASK_TABLE[name]["carrier"]}


def select_task_id(args, root=None):
    """Replace args.task by the vessel-specific journal id (before parse_env_cfg)."""
    info = resolve_journal_task(_task_of(args), getattr(args, "vessel", None), root)
    if info is not None:
        for name in ("task", "eval_task", "target_task"):
            if getattr(args, name, None):
                setattr(args, name, info["task_id"])
                break
    args.journal_task_info = info
    return info


def register_journal_tasks():
    """Import the literal journal registrations (tasks/journal, D10) once."""
    import importlib
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.append(root)
    errors = []
    for name in ("tasks.journal", "isaaclab_tasks.direct.journal"):
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append("%s: %s" % (name, exc))
    raise OverrideError("cannot import the journal registration package: " + "; ".join(errors))


def scientific_environment_names(root=None):
    """Frozen list plus literal reads discovered in the task tree (no import)."""
    names = set(KNOWN_SCIENTIFIC_ENV)
    root = tasks_root() if root is None else Path(root)
    if root.is_dir():
        for source in root.rglob("*.py"):
            if source.name.startswith("test"):
                continue
            for match in _READ.finditer(source.read_text(encoding="utf-8-sig")):
                names.add(match.group(1) or match.group(2))
    return names


def ambient_environment(allowed=(), environ=None):
    """Return {name: value} of allowed scientific variables; refuse the rest."""
    values = os.environ if environ is None else environ
    known = scientific_environment_names()
    allowed = set(allowed or ())
    # Only names the task code actually reads may be allowed (typos fail); an
    # unread prefixed variable in the environment must be unset instead.
    unknown = allowed - known
    if unknown:
        raise OverrideError("unknown --allow-env names: " + ", ".join(sorted(unknown)))
    present = {name: value for name, value in values.items()
               if (name in known or name.startswith(SCIENTIFIC_PREFIXES)) and name not in RUNTIME_ENV}
    forbidden = set(present) - allowed
    if forbidden:
        raise OverrideError("ambient scientific variables require explicit --allow-env: "
                            + ", ".join(sorted(forbidden)))
    return dict(sorted(present.items()))


def runtime_environment(environ=None):
    values = os.environ if environ is None else environ
    return {name: values[name] for name in RUNTIME_ENV if name in values}


def add_protocol_arguments(parser):
    def add(flag, **kwargs):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, **kwargs)
    add("--set", dest="extra_sets", action="append", default=[])
    add("--num_envs", type=int, default=64)
    add("--seed-namespace", default=None)
    add("--seed-block", default=None)
    add("--run-purpose", default=None)
    add("--seed-registry", default=None, help="registry file (default scripts/seed_registry.json)")
    add("--vessel", choices=("blueboat", "wamv"), default=None)
    add("--speed-reference-m-s", type=float, default=None)
    add("--speed-reference-basis", choices=("nominal", "condition_adjusted"), default="nominal")
    add("--allow-env", action="append", default=[])
    add("--dose-preset", default=None)
    add("--preset-sha256", default=None)
    add("--cell-config", default=None, help="resolved cell configuration written by run_cell.py")
    add("--cell-config-sha256", default=None)
    add("--release-descriptor", default=None,
        help="usvbench-release-v1 descriptor of the deployed release (else $%s)" % code_tree.RELEASE_DESCRIPTOR_ENV)
    # F7: journal condition stacking (F0 journal_config.add_suite_arguments,
    # same flags and semantics; declared here so no torch is imported before
    # AppLauncher). Journal ids only; applied by resolve_protocol.
    add("--suite", nargs="+", action="extend", default=[], metavar="TOKEN",
        help="journal suite levels/parameters, e.g. waves=D2 wind=D3 actuator.cap=D1 dynamic=D2")
    add("--suite-file", default=None, metavar="PATH", help="JSON/YAML file with the same tokens")


def _seed_of(args):
    for name in ("eval_seed", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            return value
    return None


def _task_of(args):
    return getattr(args, "task", None) or getattr(args, "eval_task", None) or getattr(args, "target_task", None)


def _protected_from_args(args):
    level = getattr(args, "level", 0)
    protected = {"seed": _seed_of(args),
                 "eval_level": 0 if level is None else level,
                 "curriculum_frozen": True,
                 "scene.num_envs": getattr(args, "num_envs", 64)}
    if getattr(args, "device", None) is not None:
        protected["sim.device"] = args.device
    return protected


def refuse_training_assignments(assignments):
    """G19b (P4): no --set may name a training-only cfg field, on any task (pre-launch)."""
    for item in assignments or ():
        path, _ = split_assignment(item)
        if path.split(".")[0] in TRAINING_ONLY_CFG_FIELDS + TRAINING_POOL_CFG_FIELDS:
            raise OverrideError(f"--set {path} is refused: training-only cfg content never reaches an evaluator "
                                "(G19b P4)")


def refuse_training_cfg(cfg):
    """G19b (P4): an evaluator refuses a cfg carrying a training mixture or a training hazard pool (before the
    env is built). Called first in resolve_protocol, which every certificate writer runs on its cfg."""
    for field in TRAINING_ONLY_CFG_FIELDS:
        if getattr(cfg, field, None) is not None:
            raise OverrideError(f"cfg.{field} is set: a training cfg never produces evaluation evidence (G19b P4)")
    for field in TRAINING_POOL_CFG_FIELDS:
        if getattr(getattr(cfg, field, None), "TRAINING_ADAPTER", False) is True:
            raise OverrideError(f"cfg.{field} holds a training-only online-admission pool (G19b P4)")


def _check_generic_assignments(assignments, protected):
    """Shape/protocol checks that do not need the cfg (pre-launch)."""
    refuse_training_assignments(assignments)                     # G19b P4
    seen = {}
    for item in assignments:
        path, raw = split_assignment(item)
        if path in PROTECTED_FORBIDDEN or path.split(".")[0] in PROTECTED_FORBIDDEN:
            raise OverrideError(f"--set {path} is not allowed: select the vessel by task id")
        if path in protected and protected[path] is not None:
            typed = coerce_like(raw, protected[path], path)
            if typed != protected[path]:
                raise OverrideError(f"--set {path} conflicts with protocol value {protected[path]!r}")
        if path in seen and seen[path] != raw:
            raise OverrideError(f"conflicting repeated --set {path}")
        seen[path] = raw
    return seen


def load_cell_config(path, expected_sha256):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is None or actual != expected_sha256:
        raise OverrideError("--cell-config SHA-256 mismatch or --cell-config-sha256 missing")
    config = json.loads(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("format") != "usvbench-cell-v1":
        raise OverrideError("--cell-config is not a run_cell configuration")
    return {"path": str(Path(path).resolve()), "sha256": actual, "configuration": config}


def check_cell_config(cell, args, argv):
    """The launched evaluator must be exactly the cell run_cell planned."""
    config = cell["configuration"]
    registration = getattr(args, "seed_registration", {}) or {}
    script = Path(argv[0]).name if argv else None
    expected = {
        "evaluator": script,
        "task_id": _task_of(args),
        "level": getattr(args, "level", None),
        "episodes_requested": getattr(args, "episodes", None),
        "num_envs": getattr(args, "num_envs", None),
    }
    for key, value in expected.items():
        if value is not None and config.get(key) != value:
            raise OverrideError(f"cell configuration {key}={config.get(key)!r} differs from the "
                                f"launched evaluator ({value!r})")
    planned = config.get("seed_registration") or {}
    for key in ("namespace", "block_id", "seed", "purpose", "registry_sha256"):
        if planned.get(key) != registration.get(key):
            raise OverrideError("cell configuration seed registration differs: " + key)
    # D19: runtime pool-path fields are not science literals (never compared, never refused)
    literals = {k: v for k, v in (split_assignment(item) for item in getattr(args, "extra_sets", ()) or ())
                if k not in JOURNAL_RUNTIME_FIELDS}
    planned_literals = {k: v for k, v in (config.get("effective_override_literals") or {}).items()
                        if k not in JOURNAL_RUNTIME_FIELDS}
    if literals != planned_literals:
        raise OverrideError("cell configuration --set literals differ from the launched evaluator")
    planned_suites = (config.get("journal") or {}).get("suite", [])
    if list(getattr(args, "suite", None) or []) != list(planned_suites or []):
        raise OverrideError("cell configuration --suite tokens differ from the launched evaluator")
    planned_release = config.get("release")
    if planned_release is not None and planned_release.get("release_id") != getattr(args, "release_id", None):
        raise OverrideError("cell configuration was planned for release %r, launched with %r"
                            % (planned_release.get("release_id"), getattr(args, "release_id", None)))
    checkpoint = getattr(args, "checkpoint", None)
    planned_sha = (config.get("checkpoint") or {}).get("sha256")
    if planned_sha is not None:
        if not checkpoint or hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != planned_sha:
            raise OverrideError("launched checkpoint differs from the planned checkpoint SHA-256")


def preflight_protocol(args, argv=None, parser=None):
    """Pre-launch validation; with a parser, errors exit 2 via parser.error."""
    try:
        return _preflight(args, list(argv or ()))
    except (OverrideError, ValueError, OSError) as exc:
        if parser is not None:
            parser.error(str(exc))
        raise


def resolve_release(args, environ=None, code_root=None):
    """E1: find, load and verify the release descriptor before the simulator starts.

    The descriptor is ``--release-descriptor`` or ``$USVBENCH_RELEASE_DESCRIPTOR``
    (both given and different: refused). The deployed tree
    ``<code_root>/{tasks,scripts}`` (default: the folder holding this scripts/)
    must equal the release manifest exactly (missing, extra or changed member
    files refused). Sets args.release_descriptor (absolute path or None),
    args.release_descriptor_sha256 and args.release_id; returns the release or None.
    """
    values = os.environ if environ is None else environ
    given = getattr(args, "release_descriptor", None)
    ambient = values.get(code_tree.RELEASE_DESCRIPTOR_ENV) or None
    if given and ambient and Path(given).resolve() != Path(ambient).resolve():
        raise OverrideError("--release-descriptor and $%s name different descriptors"
                            % code_tree.RELEASE_DESCRIPTOR_ENV)
    chosen = given or ambient
    args.release_descriptor, args.release_descriptor_sha256, args.release_id = None, None, None
    if not chosen:
        return None
    release = code_tree.load_release(chosen)
    root = Path(__file__).resolve().parents[1] if code_root is None else Path(code_root)
    errors = code_tree.compare_tree(root, release["records"])
    if errors:
        raise OverrideError("deployed tree %s is not release %s (%d differences: %s)"
                            % (root, release["release_id"], len(errors), "; ".join(errors[:10])))
    args.release_descriptor = release["descriptor_path"]
    args.release_descriptor_sha256 = release["descriptor_sha256"]
    args.release_id = release["release_id"]
    return release


def _preflight(args, argv):
    from seed_registry import diagnostic_registration, load_registry, validate_seed
    episode_quota(args.episodes, getattr(args, "num_envs", 64))
    args.scientific_environment = ambient_environment(getattr(args, "allow_env", ()))
    args.runtime_environment = runtime_environment()
    resolve_release(args)
    # Only the shared --set language may mutate the environment. Hydra's
    # independent parsing/precedence would otherwise bypass typed validation.
    for token in argv[1:]:
        if token.lstrip("+~").startswith("env.") and "=" in token:
            raise OverrideError("Hydra env.* overrides are unsupported; use --set FIELD=VALUE")
    value = getattr(args, "speed_reference_m_s", None)
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise OverrideError("speed reference must be finite and positive")
    if getattr(args, "dose_preset", None) is not None and not getattr(args, "preset_sha256", None):
        raise OverrideError("--dose-preset requires --preset-sha256 (use run_cell.py)")
    _check_generic_assignments(getattr(args, "extra_sets", ()) or (), _protected_from_args(args))
    # F7: the vessel-specific journal id is selected before the cfg exists;
    # D12 refusals are checked here already (and again on the cfg).
    if select_task_id(args) is not None:
        check_journal_assignments(getattr(args, "extra_sets", ()) or ())
    registry = load_registry(getattr(args, "seed_registry", None))
    seed = _seed_of(args)
    namespace = getattr(args, "seed_namespace", None)
    if namespace is not None or getattr(args, "seed_block", None) is not None \
            or getattr(args, "run_purpose", None) is not None:
        if namespace is None or getattr(args, "run_purpose", None) is None:
            raise OverrideError("--seed-namespace and --run-purpose must be given together")
        args.seed_registration = validate_seed(seed, namespace, args.run_purpose,
                                               getattr(args, "seed_block", None), registry)
    else:
        args.seed_registration = diagnostic_registration(seed, registry)
    cell_path = getattr(args, "cell_config", None)
    args.cell_config_record = None
    if cell_path is not None:
        cell = load_cell_config(cell_path, getattr(args, "cell_config_sha256", None))
        check_cell_config(cell, args, argv)
        args.cell_config_record = cell
    elif args.seed_registration.get("purpose") == "final_test":
        raise OverrideError("final_test evaluations must be planned by run_cell.py (--cell-config)")
    # G16: the windcoef.* wind-coefficient override is diagnostic only; refused here, before the
    # simulator starts, for every official purpose (selection, pilot, calibration, final_test).
    wind_coefficient_preflight(args)
    # T25 (owner decision T1.35b): the hydro.rollpitch_zeta roll/pitch damping override is diagnostic only;
    # refused here, before the simulator starts, for every official purpose (same rule as windcoef.*).
    rollpitch_override_preflight(args)
    # G20b (owner decision 2026-10-03): the appearing.gap_m trigger-gap override (calibration sweep) is
    # diagnostic only; refused here, before the simulator starts, for every official purpose.
    appearing_gap_override_preflight(args)
    # G6b / CONTEXT_R4 D28 (b): the eight RE-1 versions are stamped (cert_v3 runtime block); an
    # official run outside RE-1 is refused here, before the simulator starts; diagnostic runs pass.
    args.reference_environment = reference_environment.require_official_match(
        args.seed_registration, reference_environment.reference_block())
    return args


WIND_COEFFICIENT_SUITE = "windcoef"


def _is_wind_coefficient_token(token):
    return isinstance(token, str) and "=" in token and         token.split("=", 1)[0].strip().split(".", 1)[0] == WIND_COEFFICIENT_SUITE


def wind_coefficient_requested(args):
    """G16: (windcoef.* command-line tokens, True if the suite file names windcoef). Fail closed:
    any occurrence of 'windcoef' in a readable suite file counts as a request."""
    tokens = [t for t in (getattr(args, "suite", None) or []) if _is_wind_coefficient_token(t)]
    in_file = False
    suite_file = getattr(args, "suite_file", None)
    if suite_file:
        try:
            in_file = WIND_COEFFICIENT_SUITE in Path(suite_file).read_text(encoding="utf-8")
        except OSError:
            in_file = False          # an unreadable suite file is refused when the suites are resolved
    return tokens, in_file


def _wind_coefficient_refusal(registration, what):
    purpose = registration.get("purpose") if isinstance(registration, dict) else None
    return OverrideError("%s: the windcoef.* wind-coefficient override is diagnostic only (seed purpose "
                         "'diagnostic', e.g. a DIAG42 run without --seed-namespace); refused for seed purpose %r "
                         "(official runs use the vessel's pinned coefficients; G16)" % (what, purpose))


def wind_coefficient_preflight(args):
    """G16: refuse a wind-coefficient override in an official run before the simulator starts."""
    tokens, in_file = wind_coefficient_requested(args)
    registration = getattr(args, "seed_registration", None)
    if (tokens or in_file) and not reference_environment.is_diagnostic(registration):
        raise _wind_coefficient_refusal(registration, " ".join(tokens) or "--suite-file names windcoef")


ROLLPITCH_OVERRIDE_SUITE = "hydro"
ROLLPITCH_OVERRIDE_NAME = "rollpitch_zeta"


def _is_rollpitch_override_token(token):
    return (isinstance(token, str) and "=" in token
            and token.split("=", 1)[0].strip().split(".", 1)[0] == ROLLPITCH_OVERRIDE_SUITE)


def rollpitch_override_requested(args):
    """T25 (T1.35b): (hydro.* command-line tokens, True if the suite file names rollpitch_zeta). Fail
    closed: any occurrence of 'rollpitch_zeta' in a readable suite file counts as a request."""
    tokens = [t for t in (getattr(args, "suite", None) or []) if _is_rollpitch_override_token(t)]
    in_file = False
    suite_file = getattr(args, "suite_file", None)
    if suite_file:
        try:
            in_file = ROLLPITCH_OVERRIDE_NAME in Path(suite_file).read_text(encoding="utf-8")
        except OSError:
            in_file = False          # an unreadable suite file is refused when the suites are resolved
    return tokens, in_file


def _rollpitch_override_refusal(registration, what):
    purpose = registration.get("purpose") if isinstance(registration, dict) else None
    return OverrideError("%s: the hydro.rollpitch_zeta roll/pitch damping override is diagnostic only (seed "
                         "purpose 'diagnostic', e.g. a DIAG42 run without --seed-namespace); refused for seed "
                         "purpose %r (official runs use the journal-default roll/pitch damping; owner decision "
                         "T1.35b)" % (what, purpose))


def rollpitch_override_preflight(args):
    """T25 (T1.35b): refuse a roll/pitch damping override in an official run before the simulator starts."""
    tokens, in_file = rollpitch_override_requested(args)
    registration = getattr(args, "seed_registration", None)
    if (tokens or in_file) and not reference_environment.is_diagnostic(registration):
        raise _rollpitch_override_refusal(registration, " ".join(tokens) or "--suite-file names rollpitch_zeta")


APPEARING_GAP_OVERRIDE_KEY = "appearing.gap_m"


APPEARING_SUITE_KEY = "appearing"


def _token_key_value(token):
    if not isinstance(token, str) or "=" not in token:
        return None, None
    key, value = token.split("=", 1)
    return key.strip(), value.strip()


def _decoded_suite_file_tokens(suite_file):
    """G20c (REV-G20b m4): the suite file's tokens as the stack resolver will see them (journal_config.load_suite_file:
    JSON with \\u escapes decoded, YAML), or None when the file cannot be read or decoded (it is refused when the
    suites are resolved)."""
    try:
        journal_config = _journal_config_module(None)
        tokens, _seed = journal_config.load_suite_file(suite_file)
    except (OSError, ValueError, TypeError, OverrideError):
        return None
    return [str(t) for t in tokens]


def _suite_file_requests(args, wanted, raw_marker):
    """(tokens of the decoded suite file for which wanted(key, value) holds, True if any or, fail closed, if the
    raw text names raw_marker)."""
    suite_file = getattr(args, "suite_file", None)
    if not suite_file:
        return False
    decoded = _decoded_suite_file_tokens(suite_file)
    if decoded is not None and any(wanted(*_token_key_value(t)) for t in decoded):
        return True
    try:
        return raw_marker is not None and raw_marker in Path(suite_file).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False          # an unreadable suite file is refused when the suites are resolved


def _is_gap_override(key, value):
    return key == APPEARING_GAP_OVERRIDE_KEY


def appearing_gap_override_requested(args):
    """G20b/G20c: (appearing.gap_m command-line tokens, True if the suite file sets it). Fail closed: the suite file
    is decoded with journal_config.load_suite_file (so a \\u-escaped key is seen, REV-G20b m4) and any raw occurrence
    of 'gap_m' counts too."""
    tokens = [t for t in (getattr(args, "suite", None) or []) if _is_gap_override(*_token_key_value(t))]
    return tokens, _suite_file_requests(args, _is_gap_override, "gap_m")


def _is_supplementary_appearing_dose(key, value):
    """appearing=D<k> with k above the main table (D5 under ladder M: the calibration rung, diagnostic only)."""
    match = re.match(r"^D([0-9]+)$", value or "")
    return key == APPEARING_SUITE_KEY and match is not None and int(match.group(1)) > 4


def appearing_supplementary_dose_requested(args):
    """G20c (REV-G20b m5): (appearing=D5.. command-line tokens, True if the decoded suite file sets one)."""
    tokens = [t for t in (getattr(args, "suite", None) or []) if _is_supplementary_appearing_dose(*_token_key_value(t))]
    return tokens, _suite_file_requests(args, _is_supplementary_appearing_dose, None)


def _appearing_gap_override_refusal(registration, what):
    purpose = registration.get("purpose") if isinstance(registration, dict) else None
    return OverrideError("%s: the appearing.gap_m trigger-gap override is diagnostic only (calibration sweep; seed "
                         "purpose 'diagnostic', e.g. DIAG42); refused for seed purpose %r (official runs use the "
                         "pinned ladder, owner decision 2026-10-03)" % (what, purpose))


def _appearing_supplementary_dose_refusal(registration, what):
    purpose = registration.get("purpose") if isinstance(registration, dict) else None
    return OverrideError("%s: the supplementary appearing dose (D5 = the calibration rung, outside the main table) is "
                         "diagnostic only (seed purpose 'diagnostic', e.g. DIAG42); refused for seed purpose %r "
                         "(G20c, REV-G20b m5)" % (what, purpose))


def appearing_gap_override_preflight(args):
    """G20b/G20c: refuse the appearing.gap_m override and the supplementary dose appearing=D5 in an official run
    before the simulator starts."""
    tokens, in_file = appearing_gap_override_requested(args)
    registration = getattr(args, "seed_registration", None)
    if (tokens or in_file) and not reference_environment.is_diagnostic(registration):
        raise _appearing_gap_override_refusal(registration, " ".join(tokens) or "--suite-file names gap_m")
    tokens, in_file = appearing_supplementary_dose_requested(args)
    if (tokens or in_file) and not reference_environment.is_diagnostic(registration):
        raise _appearing_supplementary_dose_refusal(registration, " ".join(tokens)
                                                    or "--suite-file sets a supplementary appearing dose")


def _dedicated(cfg, args):
    pairs = []
    for flag, path in (("gate_width_override", "gate_width_override_m"),
                       ("fortress_aperture_override", "fortress_aperture_override_m"),
                       ("on_line_blockers_override", "on_line_blockers_override"),
                       ("current_speed", "current_speed_override_mps")):
        value = getattr(args, flag, None)
        if value is not None:
            pairs.append((path, str(value)))
    distance = getattr(args, "spawn_distance", None)
    if distance is not None:
        path = ("spawn_distance_override_m" if hasattr(cfg, "spawn_distance_override_m")
                else "curriculum_start_distance_m")
        pairs.append((path, str(distance)))
    axis, value = getattr(args, "axis", None), getattr(args, "value", None)
    imbalance = getattr(args, "imbalance", None)
    if axis is not None and imbalance is not None:
        raise OverrideError("--axis and --imbalance are mutually exclusive")
    if axis is not None:
        if value is None:
            raise OverrideError("--axis requires --value")
        pairs.append((axis, str(value)))
    elif imbalance is not None:
        pairs.append(("thrust_imbalance", str(imbalance)))
    elif value is not None:
        raise OverrideError("--value requires --axis")
    return pairs


def _journal_config_module(cfg):
    """conditions.journal_config from the SAME task tree as the journal cfg class.

    The mixin checks ``isinstance(cfg.journal_conditions, JournalConditions)``
    against its own import, so the module must be the one next to the cfg's
    module (tasks._shared.* or isaaclab_tasks.direct._shared.*).
    """
    import importlib
    module = type(cfg).__module__ or ""
    candidates = []
    if "._shared." in module:
        candidates.append(module.split("._shared.", 1)[0] + "._shared.conditions.journal_config")
    candidates += ["tasks._shared.conditions.journal_config",
                   "isaaclab_tasks.direct._shared.conditions.journal_config"]
    errors = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append("%s: %s" % (name, exc))
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.append(root)
        try:
            return importlib.import_module("tasks._shared.conditions.journal_config")
        except ImportError as exc:
            errors.append("repository root: %s" % exc)
    raise OverrideError("cannot import conditions.journal_config: " + "; ".join(errors))


def apply_journal_suites(cfg, args):
    """F7 / CONTEXT_R4 D4: install the --suite / --suite-file stack on a journal cfg.

    Resolved with F0 journal_config.resolve_from_args at the cfg's evaluation
    seed and installed with apply_journal_conditions (cfg.conditions,
    cfg.journal_conditions). Legacy cfgs refuse suite flags. Returns the
    JournalConditions stamp (also left on args.journal_suites for the
    certificate), or None.
    """
    tokens = list(getattr(args, "suite", None) or [])
    suite_file = getattr(args, "suite_file", None)
    if not is_journal_cfg(cfg):
        if tokens or suite_file:
            raise OverrideError("--suite/--suite-file apply to journal ids only (legacy cfgs use their own "
                                "condition switches)")
        args.journal_suites = None
        return None
    journal_config = _journal_config_module(cfg)
    try:
        # G16: windcoef.* tokens are split off before the stack is resolved (the stack, its tokens
        # and its digest are those of the same command without them); T25 (owner decision T1.35b):
        # the same for the diagnostic-only hydro.rollpitch_zeta token
        resolved, wind_override, rollpitch_override = journal_config.resolve_from_args_with_overrides(
            args, seed=int(cfg.seed))
        journal_config.apply_journal_conditions(cfg, resolved)
        if getattr(resolved, "appearing_gap_m", None) is not None:     # G20b: diagnostic only
            registration = getattr(args, "seed_registration", None)
            if not reference_environment.is_diagnostic(registration):
                raise _appearing_gap_override_refusal(registration, "appearing.gap_m=%r" % (resolved.appearing_gap_m,))
        if getattr(resolved, "appearing_enabled", False) and int(getattr(resolved, "appearing_dose", 0)) > 4:
            registration = getattr(args, "seed_registration", None)     # G20c (REV-G20b m5): D5 diagnostic only
            if not reference_environment.is_diagnostic(registration):
                raise _appearing_supplementary_dose_refusal(registration, "appearing=D%d" % resolved.appearing_dose)
        if wind_override is not None:
            registration = getattr(args, "seed_registration", None)
            if not reference_environment.is_diagnostic(registration):
                raise _wind_coefficient_refusal(registration, "windcoef override %r" % (wind_override,))
            journal_config.apply_wind_coefficients(cfg, wind_override)
        if rollpitch_override is not None:
            registration = getattr(args, "seed_registration", None)
            if not reference_environment.is_diagnostic(registration):
                raise _rollpitch_override_refusal(registration, "hydro.rollpitch_zeta override %r"
                                                  % (rollpitch_override,))
            journal_config.apply_rollpitch_override(cfg, rollpitch_override)
    except (TypeError, ValueError) as exc:
        raise OverrideError("journal suites rejected: %s" % exc) from exc
    stamp = resolved.stamp()
    args.journal_suites = {"tokens": list(resolved.tokens()), "stamp": stamp,
                           "requested": tokens, "suite_file": suite_file}
    if wind_override is not None:                    # G16: absent at the pinned values
        args.journal_suites["wind_coefficients"] = dict(wind_override)
    if rollpitch_override is not None:               # T25 (T1.35b): absent without the override
        args.journal_suites["rollpitch_damping_override"] = dict(rollpitch_override)
    return stamp


def apply_runtime_paths(cfg, runtime):
    """D19: set the allow-listed pool-path cfg fields (strings); returns the stamp dict.

    Never refused on a journal cfg and never part of the science overrides; the env
    checks the pinned pool / manifest SHA-256 whatever the path. A field the cfg
    already holds with a non-string, non-None value is refused (type safety).
    """
    stamp = {}
    for field, raw in sorted((runtime or {}).items()):
        if field not in JOURNAL_RUNTIME_FIELDS:
            raise OverrideError(f"{field} is not a runtime pool-path field")
        current = getattr(cfg, field, None)
        if current is not None and not isinstance(current, (str, os.PathLike)):
            raise OverrideError(f"--set {field}: the cfg holds a {type(current).__name__}, not a path")
        setattr(cfg, field, str(raw))
        stamp[field] = str(raw)
    return stamp


def resolve_protocol(cfg, args, assignments=None):
    """Apply defaults < dedicated flags < --set as ONE validated batch.

    Returns {dotted path: value read back from the final cfg} for every
    applied science override and every protocol field.
    """
    refuse_training_cfg(cfg)                                     # G19b P4: before anything is applied
    seed = _seed_of(args)
    level = getattr(args, "level", 0)
    level = 0 if level is None else level
    protected = {"seed": seed, "scene.num_envs": getattr(args, "num_envs", 64)}
    if hasattr(cfg, "curriculum_frozen"):
        protected["curriculum_frozen"] = True
    if hasattr(cfg, "eval_level"):
        protected["eval_level"] = level
    elif level != 0:
        raise OverrideError("task has no evaluation level; --level must be 0")
    device = getattr(args, "device", None)
    if device is not None:
        protected["sim.device"] = device
    generic = list(getattr(args, "extra_sets", ()) if assignments is None else assignments)
    _check_generic_assignments(generic, protected)
    runtime_paths = {}
    if is_journal_cfg(cfg):
        # D19: pool-path fields are runtime (stamped in header.runtime_paths), not science
        generic, runtime_paths = split_runtime_assignments(generic)
    args.runtime_paths = apply_runtime_paths(cfg, runtime_paths)
    # Protocol values are resolved first so task-specific validation and
    # derivation see the final seed/level/slot count.
    for path, value in protected.items():
        owner, leaf = resolve_owner(cfg, path)
        setattr(owner, leaf, value)
    merged = {}
    for path, raw in _dedicated(cfg, args):
        merged[path] = raw
    for item in generic:
        path, raw = split_assignment(item)
        if path in merged:
            owner, leaf = resolve_owner(cfg, path)
            current = getattr(owner, leaf)
            if coerce_like(raw, current, path) != coerce_like(merged[path], current, path):
                raise OverrideError(f"--set {path}={raw} conflicts with a dedicated flag value "
                                    f"{merged[path]}")
        merged[path] = raw
    # apply_overrides validates the cfg exactly once after the complete batch.
    applied = dict(apply_overrides(cfg, [f"{key}={value}" for key, value in merged.items()]))
    # E1 (D2 F14): docking reads eval_level_distances_v2 under physics v2 and
    # eval_level_distances under v1 (docking_env.py:84-86); an override of the
    # table that is NOT in force would be silently ignored, so it is refused.
    if hasattr(cfg, "eval_level_distances") and hasattr(cfg, "eval_level_distances_v2"):
        physics = getattr(cfg, "underwater_physics_cfg", None)
        version = getattr(physics, "physics_version", getattr(cfg, "physics_version", 1))
        # F5 (D2 item 14): the docking cfg exposes the table in force; journal docking
        # cfgs (contract 2) read no level table at all ("journal_plan").
        in_force = getattr(cfg, "docking_level_table_in_force", None)
        in_force = in_force() if callable(in_force) else (
            "eval_level_distances_v2" if str(version) == "2" else "eval_level_distances")
        for inactive in ("eval_level_distances", "eval_level_distances_v2"):
            if inactive != in_force and inactive in merged:
                # F7 adaptation of F5's hunk: keep E1's wording ("no effect under
                # physics_version N", asserted unchanged by test_e1_certificates) and add
                # the table in force.
                raise OverrideError(f"--set {inactive} has no effect under physics_version {version} "
                                    f"(docking level table in force: {in_force}); override the table in force")
    if not merged and hasattr(cfg, "validate_after_overrides"):
        try:
            cfg.validate_after_overrides()
        except (TypeError, ValueError) as exc:
            raise OverrideError(f"invalid cfg: {exc}") from exc
    for path, value in protected.items():
        owner, leaf = resolve_owner(cfg, path)
        if getattr(owner, leaf) != value:
            raise OverrideError(f"configuration derivation changed protocol field {path}")
        applied[path] = getattr(owner, leaf)
    for path in list(applied):
        owner, leaf = resolve_owner(cfg, path)
        applied[path] = getattr(owner, leaf)
    if is_journal_cfg(cfg):
        # D4 suites at the protocol seed, then D12: the journal cfg re-checks
        # its frozen contract after the batch (also without science overrides).
        apply_journal_suites(cfg, args)
        journal_validate(cfg)
    elif getattr(args, "suite", None) or getattr(args, "suite_file", None):
        apply_journal_suites(cfg, args)
    episode_quota(args.episodes, cfg.scene.num_envs)
    return applied
