"""Resolve, check and launch ONE evaluation cell; --dry-run imports no simulator.

A cell is (task id, vessel, controller, checkpoint, eval seed block, level or
dose preset, episodes). run_cell dispatches to the evaluator for that
controller with the same quota rule (episodes divisible by num_envs), the same
override language and precedence (eval_protocol.py), and v3 stamping. Future
queue scripts call this, never the evaluators directly.

  python -B scripts/run_cell.py --task Isaac-USV-HazardCross-Direct-v1 \
      --vessel blueboat --controller mppi --run-purpose pilot \
      --seed-namespace dose_pilots --eval-seed-block pilot-01 --level 0 \
      --episodes 128 --out cells/cross_mppi_p01.json --dry-run

Seed policy (seed_registry.json, owner decisions of 2026-09-30):
  * the block must be declared in --seed-namespace and the namespace purpose
    must equal --run-purpose (a final-test block cannot be used for selection,
    pilots or calibration, and vice versa); training seeds are never
    evaluation seeds;
  * final_test cells require a FROZEN registry;
  * PPO needs --training-seed (a declared candidate) and --checkpoint;
    selection runs are the competence gate itself (gate block, gate grid,
    nominal condition only); calibration and final_test runs need
    --gate-ledger and the seed must be ADMITTED (first passing candidates in
    declared order) with the same checkpoint SHA-256; pilots check admission
    when a ledger is given;
  * MPPI runs with the registry's fixed controller seed; MPPI and classical
    controllers may not use the selection namespace (it belongs to the PPO
    gate); tune them on pilot blocks.

Dose presets (data; D9 owns their content):
  {"format": "usvbench-dose-presets-v1", "presets": {"W2": {"level": 0,
   "overrides": {"sea_state.enable": true, "sea_state.hs_range": [0.15, 0.15]}}}}
Preset literals precede CLI --set; a CLI --set of the same field wins and is
recorded. Protocol fields (seed, level, num_envs, device, frozen curriculum)
cannot be changed by either.

Journal ids (F7, CONTEXT_R4 D10): tasks/journal/__init__.py registers the 24
journal ids, so registered_tasks maps them to the directory "journal"; the real
carrier family comes from journal_cfg_v2.TASK_TABLE and selects the classical
controller (harbour_mission -> classical_harbour.py). --task accepts a journal id
or the alias journal:<task>, resolved with --vessel to the vessel-specific id
before anything else (a journal id of the other vessel is refused). G17 / G24b: the
journal path following / station keeping / docking classical cells declare
los_pid_avoid_v2 / los_pid_cont_avoid_v2 / docking_runup_pid_avoid_v2 (base law + avoidance
layer; G17's v1 names retired). G25 / G24b: the journal hazard_nav / path_hazard classical cells
declare planner_los_pid_mem_v2 (journal planner + ray-return memory without expiry +
circumscribed-radius ladder).

Without --dry-run the resolved configuration is written to OUT.cell.json and
passed to the evaluator (--cell-config + SHA-256), which re-checks that it is
the planned cell and stamps it into the certificate. Existing OUT files are
never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from cfg_override import OverrideError, split_assignment, split_runtime_assignments
from eval_protocol import (_check_generic_assignments, ambient_environment, resolve_journal_task,
                           resolve_release, runtime_environment, tasks_root)
from cfg_override import check_journal_assignments
from eval_protocol import rollpitch_override_requested, wind_coefficient_requested
from eval_protocol import appearing_gap_override_requested, appearing_supplementary_dose_requested
from fixed_grid import episode_quota
import seed_registry as registry_module

RUN_CELL_VERSION = "d8-run-cell-v1"
CELL_FORMAT = "usvbench-cell-v1"
SCRIPTS = Path(__file__).resolve().parent
PPO = {"script": "eval_v6_frozen.py", "seed_flag": "--eval-seed", "extra": [],
       "controller_name": "learned", "information": "native observation"}
MPPI = {"script": "eval_mppi.py", "seed_flag": "--eval-seed", "extra": [],
        "controller_name": "mppi", "information": "privileged state and map"}
PLANNER = {"script": "classical_planner_pid.py", "seed_flag": "--eval-seed", "extra": [],
           "controller_name": "planner_los_pid", "information": "privileged map (planner + LOS/PID)"}
BASELINE = {"script": "classical_baseline.py", "seed_flag": "--seed",
            "extra": ["--protocol", "mission"], "controller_name": "los_pid",
            "information": "observation-only LOS/PID"}
# G9b (T1.22): classical_baseline.py names its docking controller "docking_3phase_pid" in every
# docking certificate (classical_baseline.py:1103, :1189); the plan declares the same name.
DOCKING_BASELINE = dict(BASELINE, controller_name="docking_3phase_pid")
# Declared classical controller per task family (OWNER DECISION: which
# classical controller represents a family in the paper). The LEGACY harbour
# mission ids have no declared classical controller and are refused.
CLASSICAL_BY_FAMILY = {"hazard_nav": PLANNER, "path_hazard": PLANNER,
                       "blueboat_calm_nav": BASELINE, "path_following": BASELINE,
                       "station_keeping": BASELINE, "docking": DOCKING_BASELINE}
# E1: the refusal names the controller that is missing. F7: the staged
# controller exists for the JOURNAL harbour ids (scripts/classical_harbour.py,
# on the D4 contract journal_contracts_v2.HarborMissionEpisode and F6's
# journal_phase_targets()); the legacy harbour ids publish no phase-target
# interface, so they stay refused.
MISSING_CLASSICAL = {
    "harbor_mission": ("missing controller: a classical harbour-mission controller for the LEGACY harbour "
                       "ids (they publish no phase-target interface). The journal harbour ids "
                       "(Isaac-USV-HarborMission-<BlueBoat|WAMVVRX>-Journal-Direct-v1) run "
                       "scripts/classical_harbour.py on the D4 harbour task contract "
                       "(tasks/_shared/journal_contracts_v2.py HarborMissionEpisode)"),
}
# F7: journal cells (registration directory "journal", carrier from TASK_TABLE).
JOURNAL_INFORMATION = "journal estimate (position, velocity, 36 rays; D5) + heading + task targets"
JOURNAL_PPO = dict(PPO, information="native journal observation (estimate-derived + 36 rays)")
JOURNAL_MPPI = dict(MPPI, information=JOURNAL_INFORMATION + "; MPPI on the D5 registry vessel model")
JOURNAL_PLANNER = dict(PLANNER, information=JOURNAL_INFORMATION + "; ray-map planner + LOS/PID")
JOURNAL_BASELINE = dict(BASELINE, information=JOURNAL_INFORMATION + "; LOS/PID")
JOURNAL_DOCKING_BASELINE = dict(JOURNAL_BASELINE, controller_name="docking_3phase_pid")
# G17 (owner decision B10 supplement, option B): the journal path following, station keeping and
# docking ids run the same classical_baseline.py with the obstacle-avoidance layer
# (scripts/journal_avoidance.py) around the unchanged base law, for every cell (calm included);
# classical_baseline.py stamps these names (journal_avoidance.CONTROLLER_NAMES). The two pre-G17
# journal entries above stay as the base-law records the layer names in its certificate block.
JOURNAL_AVOID_INFORMATION = (JOURNAL_INFORMATION + "; LOS/PID + G17 avoidance layer (rays only; docking berth "
                             "walls rebuilt from the published berth pose and the frozen plan)")
# G24b: one name per journal task (owner decisions 2026-10-04): path following los_pid_avoid_v2 (layer memory
# without expiry), station keeping los_pid_cont_avoid_v2 (continuous_v2 hold law), docking
# docking_runup_pid_avoid_v2 (berth-axis run-up point first); G17's los_pid_avoid_v1 /
# docking_3phase_pid_avoid_v1 are retired.
JOURNAL_AVOID_BASELINE = dict(JOURNAL_BASELINE, controller_name="los_pid_avoid_v2",
                              information=JOURNAL_AVOID_INFORMATION)
JOURNAL_AVOID_STATION_BASELINE = dict(JOURNAL_AVOID_BASELINE, controller_name="los_pid_cont_avoid_v2",
                                      information=JOURNAL_AVOID_INFORMATION + "; continuous_v2 hold law")
JOURNAL_AVOID_DOCKING_BASELINE = dict(JOURNAL_AVOID_BASELINE, controller_name="docking_runup_pid_avoid_v2",
                                      information=JOURNAL_AVOID_INFORMATION + "; berth-axis run-up docking law")
# G25 (owner decision 2026-10-03, NC 09; G22 options A2 + A1): the journal hazard_nav and path_hazard cells run
# classical_planner_pid.py's JournalMemoryPlannerController (the journal planner with the ray-return memory of
# scripts/planner_memory.py and the circumscribed-radius ladder journal_memory_ladder); classical_planner_pid.py
# stamps planner_memory.CONTROLLER_NAME. JOURNAL_PLANNER above stays as the base-planner record.
JOURNAL_MEMORY_PLANNER = dict(JOURNAL_PLANNER, controller_name="planner_los_pid_mem_v2",
                              information=JOURNAL_INFORMATION + "; ray-map planner with remembered ray returns "
                              "(G25; no expiry, G24b) + LOS/PID")
HARBOUR = {"script": "classical_harbour.py", "seed_flag": "--eval-seed", "extra": [],
           "controller_name": "staged_harbour",
           "information": JOURNAL_INFORMATION + " + F6 phase targets; ray-map planner + LOS/PID, docking law"}
JOURNAL_CLASSICAL_BY_CARRIER = {"hazard_nav": JOURNAL_MEMORY_PLANNER, "path_hazard": JOURNAL_MEMORY_PLANNER,
                                "path_following": JOURNAL_AVOID_BASELINE, "station_keeping": JOURNAL_AVOID_STATION_BASELINE,
                                "docking": JOURNAL_AVOID_DOCKING_BASELINE, "harbor_mission": HARBOUR}
_ID = re.compile(r"""\bid\s*=\s*(['"])(Isaac-[A-Za-z0-9_.\-]+)\1""")


class RunCellError(ValueError):
    """The requested cell violates the evaluation protocol or seed policy."""


def _reject_constant(value):
    raise RunCellError("non-finite JSON constant: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)


def registered_tasks(root=None):
    """{literal gym id: task family directory}, parsed statically (no import)."""
    root = tasks_root() if root is None else Path(root)
    found = {}
    for init in sorted(root.glob("*/__init__.py")):
        for match in _ID.finditer(init.read_text(encoding="utf-8-sig")):
            task = match.group(2)
            if task in found and found[task] != init.parent.name:
                raise RunCellError("task id registered by two families: " + task)
            found[task] = init.parent.name
    return found


def assignment_value(value):
    """Encode a JSON primitive in the one shared --set syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if not value or any(isinstance(item, (list, tuple, dict)) for item in value):
            raise RunCellError("preset override sequences must be nonempty and flat")
        return ",".join(assignment_value(item) for item in value)
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise RunCellError("preset override must be finite")
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return value
    raise RunCellError("preset overrides require JSON scalar or flat sequence values")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser_for_cell():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--vessel", choices=("blueboat", "wamv"), required=True)
    parser.add_argument("--controller", choices=("ppo", "mppi", "classical"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--gate-ledger", type=Path)
    parser.add_argument("--eval-seed-block", required=True)
    parser.add_argument("--seed-namespace", required=True)
    parser.add_argument("--run-purpose", required=True,
                        choices=registry_module.EVALUATION_PURPOSES)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--level", type=int)
    choice.add_argument("--dose-preset")
    parser.add_argument("--presets-file", type=Path)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=64)
    parser.add_argument("--set", dest="extra_sets", action="append", default=[])
    parser.add_argument("--suite", nargs="+", action="extend", default=[], metavar="TOKEN",
                        help="journal ids only: condition suite tokens (F0 journal_config), passed through")
    parser.add_argument("--suite-file", default=None, help="journal ids only: suite token file")
    parser.add_argument("--allow-env", action="append", default=[])
    parser.add_argument("--speed-reference-m-s", type=float)
    parser.add_argument("--speed-reference-basis", choices=("nominal", "condition_adjusted"),
                        default="nominal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--controller-seed", type=int,
                        help="MPPI only; must equal the registry's declared value")
    parser.add_argument("--seed-registry", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--release-descriptor", default=None,
                        help="release descriptor (else $USVBENCH_RELEASE_DESCRIPTOR); the deployed tree "
                             "must equal it, and the evaluator is launched with it")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _evaluator_for(controller, family, journal=False):
    if journal:
        if controller == "ppo":
            return JOURNAL_PPO
        if controller == "mppi":
            return JOURNAL_MPPI
        if family not in JOURNAL_CLASSICAL_BY_CARRIER:
            raise RunCellError("no declared classical controller for journal carrier %r" % family)
        return JOURNAL_CLASSICAL_BY_CARRIER[family]
    if controller == "ppo":
        return PPO
    if controller == "mppi":
        return MPPI
    if family not in CLASSICAL_BY_FAMILY:
        detail = MISSING_CLASSICAL.get(family)
        raise RunCellError("no declared classical controller for task family %r%s"
                           % (family, ": " + detail if detail else ""))
    return CLASSICAL_BY_FAMILY[family]


def _preset(args):
    if args.dose_preset is None:
        if args.presets_file is not None:
            raise RunCellError("--presets-file requires --dose-preset")
        return None, None, []
    if args.presets_file is None:
        raise RunCellError("--dose-preset requires --presets-file")
    presets = read_json(args.presets_file)
    if not isinstance(presets, dict) or not isinstance(presets.get("presets"), dict):
        raise RunCellError("presets file needs a 'presets' object")
    if presets.get("format", "usvbench-dose-presets-v1") != "usvbench-dose-presets-v1":
        raise RunCellError("unsupported presets format")
    preset = presets["presets"].get(args.dose_preset)
    if not isinstance(preset, dict) or type(preset.get("level")) is not int:
        raise RunCellError("unknown dose preset or preset without an integer level: "
                           + args.dose_preset)
    if set(preset) - {"level", "overrides", "description", "reference"}:
        raise RunCellError("unknown keys in dose preset " + args.dose_preset)
    overrides = preset.get("overrides", {})
    if not isinstance(overrides, dict):
        raise RunCellError("preset overrides must be an object")
    digest = hashlib.sha256(json.dumps(preset, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode("ascii")).hexdigest()
    literals = [path + "=" + assignment_value(value) for path, value in sorted(overrides.items())]
    return preset, digest, literals


def _literals(preset_literals, cli_sets):
    cli = {}
    for assignment in cli_sets:
        path, raw = split_assignment(assignment)
        if path in cli and raw != cli[path]:
            raise RunCellError("conflicting repeated --set " + path)
        cli[path] = raw
    final = {}
    for assignment in list(preset_literals) + list(cli_sets):
        path, raw = split_assignment(assignment)
        for token in raw.strip("()[]").split(","):
            if token.strip().lower() in ("nan", "+nan", "-nan", "inf", "+inf", "-inf",
                                          "infinity", "+infinity", "-infinity"):
                raise RunCellError("non-finite --set value for " + path)
        final[path] = raw
    return final


def _seed_policy(args, registry, stamp, controller, level, preset, literals):
    """Owner seed rules; returns (checkpoint block, admission block, controller seed)."""
    purpose = stamp["purpose"]
    if purpose == "final_test" and registry["status"] != "frozen":
        raise RunCellError("final_test cells require a frozen seed registry (status is %r)"
                           % registry["status"])
    order = registry["ppo_training"]["candidate_order"]
    gate = registry["competence_gate"]
    checkpoint, admission, controller_seed = None, None, None
    if controller == "ppo":
        if args.training_seed is None or args.training_seed not in order:
            raise RunCellError("PPO cells need --training-seed from the declared candidate order")
        if args.checkpoint is None:
            raise RunCellError("PPO cells require --checkpoint")
        if not args.checkpoint.is_file():
            raise RunCellError("checkpoint is not a file: " + str(args.checkpoint))
        checkpoint = {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint)}
        if args.controller_seed is not None:
            raise RunCellError("--controller-seed applies only to MPPI")
        if purpose == "selection":
            if stamp["block_id"] != gate["seed_block"]:
                raise RunCellError("selection runs are the competence gate: use block %s"
                                   % gate["seed_block"])
            if args.episodes != gate["episodes"] or args.num_envs != gate["num_envs"]:
                raise RunCellError("the competence gate uses %d episodes on %d envs"
                                   % (gate["episodes"], gate["num_envs"]))
            if preset is not None or literals:
                raise RunCellError("the competence gate runs the nominal condition "
                                   "(no dose preset, no --set)")
            if args.gate_ledger is not None:
                raise RunCellError("the gate run creates ledger evidence; do not pass --gate-ledger")
            admission = {"status": "gate_evaluation", "training_seed": args.training_seed}
        elif args.gate_ledger is None:
            if purpose in ("calibration", "final_test"):
                raise RunCellError(purpose + " PPO cells need --gate-ledger (admitted seeds only)")
            admission = {"status": "not_checked", "reason": "pilot without a gate ledger",
                         "training_seed": args.training_seed}
        else:
            ledger = registry_module.read_json(args.gate_ledger)
            # G19b follow-up m3: the registry's declared selection rule (seed_registry.delivery: fixed_first_n
            # delivers candidates 1..3 with their flags; first_passing the first passing ones / B9 fallback)
            result = registry_module.delivery(ledger, registry, args.task, args.vessel)
            admitted = {entry["training_seed"]: entry for entry in result["delivered"]}
            if args.training_seed not in admitted:
                raise RunCellError("training seed %d is not admitted for %s/%s under selection mode %s "
                                   "(delivered: %s; dead: %s; status %s)"
                                   % (args.training_seed, args.task, args.vessel, result["mode"],
                                      sorted(admitted), [d["training_seed"] for d in result["dead"]],
                                      result["status"]))
            if admitted[args.training_seed]["checkpoint_sha256"] != checkpoint["sha256"]:
                raise RunCellError("checkpoint differs from the admitted checkpoint of seed %d"
                                   % args.training_seed)
            admission = {"status": "admitted", "ledger": {"path": str(args.gate_ledger.resolve()),
                                                          "sha256": sha256_file(args.gate_ledger)},
                         "selection_mode": result["mode"], "flag": admitted[args.training_seed]["flag"],
                         "rule_result": result, "training_seed": args.training_seed}
    else:
        if args.training_seed is not None or args.checkpoint is not None or args.gate_ledger is not None:
            raise RunCellError("--training-seed/--checkpoint/--gate-ledger apply only to PPO "
                               "(the only trained controller)")
        if purpose == "selection":
            raise RunCellError("the selection namespace belongs to the PPO competence gate; "
                               "tune MPPI/classical controllers on pilot blocks")
        if controller == "mppi":
            declared = registry.get("controller_seeds", {}).get("mppi")
            if declared is None:
                raise RunCellError("registry declares no MPPI controller seed")
            if args.controller_seed is not None and args.controller_seed != declared:
                raise RunCellError("MPPI controller seed is fixed at %d by the registry" % declared)
            controller_seed = declared
        elif args.controller_seed is not None:
            raise RunCellError("--controller-seed applies only to MPPI")
    return checkpoint, admission, controller_seed


def resolve_cell(args):
    """Return {command, configuration, cell_config_bytes}; imports no simulator."""
    try:
        return _resolve(args)
    except (OverrideError, registry_module.RegistryError) as exc:
        raise RunCellError(str(exc)) from exc


def _resolve(args):
    registry = registry_module.load_registry(args.seed_registry)
    quota = episode_quota(args.episodes, args.num_envs)
    tasks = registered_tasks()
    # F7: a journal alias/id is resolved to the vessel-specific id first.
    journal = resolve_journal_task(args.task, args.vessel, root=tasks_root())
    if journal is not None:
        args.task = journal["task_id"]
    if args.task not in tasks:
        raise RunCellError("unknown task id (not a literal registration under tasks/): " + args.task)
    family = tasks[args.task]
    if family == "journal" and journal is None:
        raise RunCellError("task %s is registered by tasks/journal but is not a journal_cfg_v2 id" % args.task)
    if journal is not None and family != "journal":
        raise RunCellError("journal id %s is not registered by tasks/journal (found %s)" % (args.task, family))
    carrier = journal["carrier"] if journal is not None else family
    evaluator = _evaluator_for(args.controller, carrier, journal=journal is not None)
    if journal is not None:
        check_journal_assignments(args.extra_sets)
    elif args.suite or args.suite_file:
        raise RunCellError("--suite/--suite-file apply to journal ids only")
    # G16: run_cell plans official purposes only (selection, pilot, calibration, final_test); the
    # windcoef.* wind-coefficient override is diagnostic only (a direct DIAG42 evaluator call).
    wind_tokens, wind_in_file = wind_coefficient_requested(args)
    if wind_tokens or wind_in_file:
        raise RunCellError("the windcoef.* wind-coefficient override is diagnostic only; run_cell plans official "
                           "cells (purpose %s), which use the vessel's pinned coefficients (G16): %s"
                           % (args.run_purpose, " ".join(wind_tokens) or "--suite-file names windcoef"))
    # T25 (owner decision T1.35b): the hydro.rollpitch_zeta roll/pitch damping override is diagnostic only too.
    rollpitch_tokens, rollpitch_in_file = rollpitch_override_requested(args)
    if rollpitch_tokens or rollpitch_in_file:
        raise RunCellError("the hydro.rollpitch_zeta roll/pitch damping override is diagnostic only; run_cell plans "
                           "official cells (purpose %s), which use the journal-default roll/pitch damping "
                           "(T1.35b): %s" % (args.run_purpose, " ".join(rollpitch_tokens)
                                             or "--suite-file names rollpitch_zeta"))
    # G20b (owner decision 2026-10-03): the appearing.gap_m trigger-gap override (calibration sweep) is diagnostic
    # only; run_cell plans official cells, which use the pinned ladder.
    gap_tokens, gap_in_file = appearing_gap_override_requested(args)
    if gap_tokens or gap_in_file:
        raise RunCellError("the appearing.gap_m trigger-gap override is diagnostic only; run_cell plans official "
                           "cells (purpose %s), which use the pinned appearing ladder: %s"
                           % (args.run_purpose, " ".join(gap_tokens) or "--suite-file names gap_m"))
    # G20c (REV-G20b m5): the supplementary appearing dose (D5 = the calibration rung) is diagnostic only too.
    supp_tokens, supp_in_file = appearing_supplementary_dose_requested(args)
    if supp_tokens or supp_in_file:
        raise RunCellError("the supplementary appearing dose (D5, the calibration rung) is diagnostic only; run_cell "
                           "plans official cells (purpose %s), which use the main appearing table D0..D4: %s"
                           % (args.run_purpose, " ".join(supp_tokens) or "--suite-file sets a supplementary dose"))
    stamp = registry_module.resolve_seed_block(args.eval_seed_block, args.seed_namespace,
                                               args.run_purpose, registry)
    preset, preset_sha, preset_literals = _preset(args)
    level = preset["level"] if preset is not None else args.level
    if type(level) is not int or level < 0:
        raise RunCellError("evaluation level must be a nonnegative integer")
    science_sets, runtime_paths = (split_runtime_assignments(args.extra_sets) if journal is not None
                                   else (list(args.extra_sets), {}))
    literals = _literals(preset_literals, science_sets)
    assignments = [path + "=" + raw for path, raw in literals.items()]
    protected = {"seed": stamp["seed"], "eval_level": level, "curriculum_frozen": True,
                 "scene.num_envs": args.num_envs, "sim.device": args.device}
    _check_generic_assignments(assignments, protected)
    allowed = ambient_environment(args.allow_env)
    release = resolve_release(args)
    checkpoint, admission, controller_seed = _seed_policy(
        args, registry, stamp, args.controller, level, preset, literals)
    if args.speed_reference_m_s is not None and (not math.isfinite(args.speed_reference_m_s)
                                                 or args.speed_reference_m_s <= 0):
        raise RunCellError("speed reference must be finite and positive")
    out = args.out.resolve()
    cell_path = Path(str(out) + ".cell.json")
    configuration = {
        "format": CELL_FORMAT, "run_cell_version": RUN_CELL_VERSION,
        "task_id": args.task, "task_family": family, "vessel": args.vessel,
        "controller": args.controller, "evaluator": evaluator["script"],
        "controller_name": evaluator["controller_name"],
        "controller_information": evaluator["information"],
        "checkpoint": checkpoint, "training_seed": args.training_seed,
        "admission": admission, "controller_seed": controller_seed,
        "level": level, "preset": args.dose_preset, "preset_definition": preset,
        "preset_sha256": preset_sha, "effective_override_literals": literals,
        "runtime_paths": runtime_paths,
        "override_precedence": "task defaults < preset literals < CLI --set; protocol fields fixed",
        "episodes_requested": args.episodes, "num_envs": args.num_envs,
        "episodes_per_env": quota, "seed_registration": stamp,
        "allowed_environment": allowed, "runtime_environment": runtime_environment(),
        "speed_reference_m_s": args.speed_reference_m_s,
        "speed_reference_basis": args.speed_reference_basis,
        "device": args.device, "out": str(out), "schema_version": 3,
        "release": None if release is None else {
            "release_id": release["release_id"], "code_tree_sha256": release["code_tree_sha256"],
            "descriptor_path": release["descriptor_path"],
            "descriptor_sha256": release["descriptor_sha256"]},
        **({} if journal is None else {"journal": {
            "journal_task": journal["journal_task"], "carrier_family": carrier,
            "vehicle": journal["vehicle"], "registration": "tasks/journal/__init__.py",
            "suite": list(args.suite), "suite_file": args.suite_file}}),
        "deferred_runtime_checks": [
            "cfg field existence and types for every --set (cfg_override at launch)",
            "runtime vessel equals --vessel (cert_v3 at launch)",
            "curriculum frozen at the level for the whole run (cert_v3 per step)",
        ],
    }
    cell_bytes = (json.dumps(configuration, indent=2, sort_keys=True, allow_nan=False)
                  + "\n").encode("ascii")
    cell_sha = hashlib.sha256(cell_bytes).hexdigest()
    command = [str(args.python), str(SCRIPTS / evaluator["script"]), *evaluator["extra"],
               "--task", args.task, "--num_envs", str(args.num_envs),
               "--episodes", str(args.episodes), "--level", str(level),
               evaluator["seed_flag"], str(stamp["seed"]),
               "--seed-namespace", stamp["namespace"], "--seed-block", stamp["block_id"],
               "--run-purpose", stamp["purpose"], "--vessel", args.vessel,
               "--device", args.device, "--headless", "--out", str(out),
               "--cell-config", str(cell_path), "--cell-config-sha256", cell_sha]
    if args.seed_registry is not None:
        command.extend(("--seed-registry", str(args.seed_registry.resolve())))
    if release is not None:
        command.extend(("--release-descriptor", release["descriptor_path"]))
    if checkpoint is not None:
        command.extend(("--checkpoint", checkpoint["path"]))
    if controller_seed is not None:
        command.extend(("--mppi-seed", str(controller_seed)))
    if args.dose_preset is not None:
        command.extend(("--dose-preset", args.dose_preset, "--preset-sha256", preset_sha))
    for assignment in assignments:
        command.extend(("--set", assignment))
    for field, raw in sorted(runtime_paths.items()):
        command.extend(("--set", field + "=" + raw))
    if args.suite:
        command.extend(("--suite", *args.suite))
    if args.suite_file:
        command.extend(("--suite-file", args.suite_file))
    for name in args.allow_env:
        command.extend(("--allow-env", name))
    if args.speed_reference_m_s is not None:
        command.extend(("--speed-reference-m-s", repr(args.speed_reference_m_s),
                        "--speed-reference-basis", args.speed_reference_basis))
    return {"command": command, "command_display": subprocess.list2cmdline(command),
            "configuration": configuration, "cell_config_path": str(cell_path),
            "cell_config_sha256": cell_sha, "cell_config_bytes": cell_bytes}


def main(argv=None):
    parser = parser_for_cell()
    args = parser.parse_args(argv)
    try:
        resolved = resolve_cell(args)
        printable = {key: value for key, value in resolved.items() if key != "cell_config_bytes"}
        if args.dry_run:
            print(json.dumps(printable, indent=2, sort_keys=True, allow_nan=False))
            return 0
        cell_path = Path(resolved["cell_config_path"])
        for path in (args.out, cell_path):
            if Path(path).exists():
                raise RunCellError("refusing to overwrite existing file: " + str(path))
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        with cell_path.open("xb") as stream:
            stream.write(resolved["cell_config_bytes"])
        return subprocess.run(resolved["command"], check=False).returncode
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
