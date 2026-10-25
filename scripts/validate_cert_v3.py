"""Validate v3 certificates without importing torch or Isaac Lab.

Default acceptance requires the complete 64 x 2 benchmark grid. Use explicit
--num-envs/--episodes-per-env for smoke grids. --legacy-report is read-only:
it converts available legacy data into a diagnostic view, never a certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import sys

from jsonschema import Draft202012Validator, FormatChecker

try:
    from cfg_override import JOURNAL_RUNTIME_FIELDS, coerce_like, split_assignment
except ImportError:
    # The deliverable at work/ can run alongside the copied repo/.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "scripts"))
    from cfg_override import JOURNAL_RUNTIME_FIELDS, coerce_like, split_assignment
try:
    import code_tree
    import d7_exposure
    import appearing_exposure
    import reference_environment
    from episode_outcome import episode_outcome
except ImportError:  # invoked by path from another working directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import code_tree
    import d7_exposure
    import appearing_exposure
    import reference_environment
    from episode_outcome import episode_outcome


SUPPORTED_MINOR_VERSIONS = (0, 1, 2)
CONDITION_STAMP_SCHEMA = "usvbench.conditions/2"
# F7 (CONTEXT_R4 E-base seam): hydro_v2 certificates carry E3's schema-3 stamp
# (tasks/_shared/conditions/stamp.py stamp_conditions_v3), same block layout and
# the same digest rule over ``parameters``; both are accepted, never mixed.
CONDITION_STAMP_SCHEMA_V3 = "usvbench.conditions/3"
CONDITION_STAMP_SCHEMAS = (CONDITION_STAMP_SCHEMA, CONDITION_STAMP_SCHEMA_V3)
# F0 pairing declaration of journal certificates (conditions/pairing.py).
PAIRING_SCHEMA = "usvbench.pairing/1"
LEGACY_HASH_ALGORITHM = "sha256-path-nul-content-v1"
TERMINAL_REASONS = ("success", "collision", "out_of_bounds", "failure",
                    "capsize", "sinking", "airborne", "non_finite_state")
REL_TOL = 1e-9
ABS_TOL = 1e-12


# G9: MPPI control-step implementations (mppi_controller.MPPI_IMPLEMENTATIONS) and what each
# --mppi-impl value resolves to (eval_mppi.resolve_mppi_impl; "auto" depends on journal + device).
MPPI_IMPLEMENTATIONS = ("eager", "cuda_graph_v1")
MPPI_IMPL_REQUESTS = {"eager": "eager", "graph": "cuda_graph_v1"}
MPPI_LEGACY_NOISE_BACKEND = "legacy-shared-generator-v1"


def mppi_implementation_errors(header):
    """G9: header.controller.configuration.mppi_implementation of MPPI certificates.

    Required for journal MPPI certificates whose command carries --mppi-impl (every G9
    eval_mppi records it in command.parsed_args); journal MPPI certificates written before
    G9 carry neither and stay valid. When present: an eager run captured no graph; a
    cuda_graph_v1 run captured at least one graph, every capture after the first replaced
    the live graph (recaptures = captured_graphs - 1), ran on a CUDA controller device with
    the keyed noise stream; and the mode is what --mppi-impl resolves to (auto: the graph for
    journal ids on a CUDA device, else eager). Only MPPI certificates carry the stamp.
    """
    controller = header.get("controller") or {}
    configuration = controller.get("configuration") or {}
    stamp = configuration.get("mppi_implementation")
    if controller.get("name") != "mppi":
        return [] if stamp is None else ["controller.configuration.mppi_implementation on a %r certificate "
                                         "(only MPPI certificates stamp it)" % (controller.get("name"),)]
    parsed = (header.get("command") or {}).get("parsed_args") or {}
    journal = "journal" in header
    if stamp is None:
        if journal and "mppi_impl" in parsed:
            return ["journal MPPI certificate lacks controller.configuration.mppi_implementation (G9)"]
        return []
    errors = []
    mode, graphs, recaptures = stamp["mode"], stamp["captured_graphs"], stamp["recaptures"]
    device = configuration.get("device")
    cuda = isinstance(device, str) and device.startswith("cuda")
    if mode == "eager":
        if graphs != 0 or recaptures != 0:
            errors.append("mppi_implementation: an eager run captures no CUDA graph "
                          "(captured_graphs %r, recaptures %r)" % (graphs, recaptures))
    else:
        if graphs < 1:
            errors.append("mppi_implementation: a cuda_graph_v1 run captured no graph")
        elif recaptures != graphs - 1:
            errors.append("mppi_implementation: cuda_graph_v1 keeps one live graph, so recaptures (%r) must equal "
                          "captured_graphs - 1 (%r)" % (recaptures, graphs - 1))
        if not cuda:
            errors.append("mppi_implementation: cuda_graph_v1 on controller device %r (needs CUDA)" % (device,))
        noise = (configuration.get("sampling_noise") or {}).get("backend")
        if noise == MPPI_LEGACY_NOISE_BACKEND:
            errors.append("mppi_implementation: cuda_graph_v1 with the legacy shared-generator noise stream")
    if "mppi_impl" in parsed:
        requested = parsed["mppi_impl"]
        if requested == "auto":
            expected = "cuda_graph_v1" if (journal and cuda) else "eager"
        else:
            expected = MPPI_IMPL_REQUESTS.get(requested)
        if expected is None:
            errors.append("unknown --mppi-impl %r in command.parsed_args" % (requested,))
        elif expected != mode:
            errors.append("mppi_implementation mode %r contradicts --mppi-impl %s (resolves to %r on %s, %s)"
                          % (mode, requested, expected, device, "journal" if journal else "legacy id"))
    return errors


# G17: the journal classical controllers with the obstacle-avoidance layer (scripts/journal_avoidance.py
# CONTROLLER_NAMES) and the journal task each one belongs to; every such certificate carries the
# layer block header.controller.configuration.avoidance (constants + run telemetry).
AVOIDANCE_FORMAT = "usvbench-classical-avoidance-v1"
# G24b: layer g24b_avoidance_v2 (memory without expiry) and one name per journal task (station keeping
# holds with continuous_v2, docking runs the run-up law); the G17 names / layer version are not accepted.
AVOIDANCE_LAYER_VERSIONS = ("g24b_avoidance_v2",)
AVOIDANCE_CONTROLLERS = {"los_pid_avoid_v2": {"path_following": "los_pid"},
                         "los_pid_cont_avoid_v2": {"station_keeping": "los_pid_cont"},
                         "docking_runup_pid_avoid_v2": {"docking": "docking_runup_pid"}}
AVOIDANCE_MEMORY_RULE = "passed"                  # G24b: no time expiry (journal_avoidance obstacle_memory_rule)
AVOIDANCE_STATION_HOLD_MODE = "continuous_v2"     # G24b: journal station-keeping hold law
AVOIDANCE_TELEMETRY_COUNTS = ("activations", "handbacks", "active_steps", "plans", "plan_failures",
                              "fallback_steps", "max_simultaneous_active", "episodes_finished",
                              "episodes_with_activation")


def avoidance_layer_errors(header):
    """G17: header.controller.configuration.avoidance of the avoidance-layer certificates.

    A certificate named los_pid_avoid_v2 / los_pid_cont_avoid_v2 / docking_runup_pid_avoid_v2 must carry the block and
    no other controller may; the block names the same controller, its journal task and base law,
    the vessel of the header, and constants that obey their own derivation rules (gaps = ratio x
    t_avoid x u_ref, clear gap > trigger gap, lengths = reference x length scale). A finished
    certificate's telemetry holds nonnegative integer counts that are mutually consistent.
    Certificates of other controllers and pre-G17 journal certificates stay valid.
    """
    controller = header.get("controller") or {}
    name = controller.get("name")
    block = (controller.get("configuration") or {}).get("avoidance")
    if name not in AVOIDANCE_CONTROLLERS:
        return [] if block is None else ["controller.configuration.avoidance on a %r certificate (only the "
                                         "avoidance-layer controllers stamp it)" % (name,)]
    if not isinstance(block, dict):
        return ["%s certificate lacks controller.configuration.avoidance (G17)" % name]
    errors = []
    if block.get("format") != AVOIDANCE_FORMAT or block.get("layer_version") not in AVOIDANCE_LAYER_VERSIONS:
        errors.append("avoidance: unknown format / layer version %r / %r" % (block.get("format"),
                                                                            block.get("layer_version")))
    if block.get("controller_name") != name:
        errors.append("avoidance: block names controller %r, header %r" % (block.get("controller_name"), name))
    task = block.get("task")
    if task not in AVOIDANCE_CONTROLLERS[name]:
        errors.append("avoidance: task %r is not a %s task" % (task, name))
    elif block.get("base_controller_name") != AVOIDANCE_CONTROLLERS[name][task]:
        errors.append("avoidance: base controller %r of %s on %s" % (block.get("base_controller_name"), name, task))
    journal = header.get("journal")
    if isinstance(journal, dict) and journal.get("task") != task:
        errors.append("avoidance: block task %r differs from header.journal.task %r" % (task, journal.get("task")))
    rules = block.get("rule_constants") or {}
    vessel = block.get("vessel_constants") or {}
    task_constants = block.get("task_constants") or {}
    if vessel.get("vessel") != header.get("vessel"):
        errors.append("avoidance: constants of vessel %r on a %r certificate" % (vessel.get("vessel"),
                                                                              header.get("vessel")))
    if task_constants.get("task") != task:
        errors.append("avoidance: task constants of %r" % (task_constants.get("task"),))
    try:
        t_avoid, u_ref, scale = vessel["t_avoid_s"], vessel["reference_speed_mps"], vessel["length_scale"]
        derived = {
            "trigger_gap_m": rules["trigger_ttc_ratio"] * t_avoid * u_ref,
            "clear_gap_m": rules["clear_ttc_ratio"] * t_avoid * u_ref,
            "obstacle_radius_m": rules["obstacle_radius_length_factor"] * scale,
            "plan_cell_m": rules["plan_cell_ref_m"] * scale,
            "waypoint_spacing_m": rules["waypoint_spacing_ref_m"] * scale,
            "accept_radius_m": rules["accept_radius_ref_m"] * scale,
            "replan_every_steps": rules["replan_every_steps"],
            "keep_inflation_m": min(vessel["inflation_ladder_m"]),
            "avoidance_speed_mps": rules["avoidance_speed_ratio"] * u_ref,
            "closing_speed_mps": rules["closing_speed_ratio"] * u_ref}
        for key, value in derived.items():
            if vessel.get(key) != value:
                errors.append("avoidance: %s %r differs from its rule (%r)" % (key, vessel.get(key), value))
        if not (0.0 < vessel["trigger_gap_m"] < vessel["clear_gap_m"]):
            errors.append("avoidance: the clear gap must exceed the trigger gap (hysteresis)")
        if not vessel["static_mask_tolerance_m"] > 0.0:
            errors.append("avoidance: static mask tolerance must be positive")
        # G24b pins: memory without expiry; station keeping holds with continuous_v2 (rule and the configuration's
        # hold_mode); docking run-up point = slip opening x - the registry approach_runup_m
        if rules.get("obstacle_memory_rule") != AVOIDANCE_MEMORY_RULE:
            errors.append("avoidance: obstacle_memory_rule %r is not %r" % (rules.get("obstacle_memory_rule"),
                                                                         AVOIDANCE_MEMORY_RULE))
        if rules.get("station_hold_mode") != AVOIDANCE_STATION_HOLD_MODE:
            errors.append("avoidance: station_hold_mode %r is not %r" % (rules.get("station_hold_mode"),
                                                                      AVOIDANCE_STATION_HOLD_MODE))
        if task == "station_keeping":
            hold = (controller.get("configuration") or {}).get("hold_mode")
            if hold != AVOIDANCE_STATION_HOLD_MODE:
                errors.append("avoidance: station keeping ran hold_mode %r, not %r" % (hold, AVOIDANCE_STATION_HOLD_MODE))
        if task == "docking":
            pm = _planner_memory_module()
            runup = float(pm.shared_module("journal_cfg_v2").vessel_record(header.get("vessel")).geometry.approach_runup_m)
            if task_constants.get("approach_runup_m") != runup:
                errors.append("avoidance: approach_runup_m %r differs from the registry (%r)"
                              % (task_constants.get("approach_runup_m"), runup))
            if task_constants.get("runup_x_m") != task_constants["slip_opening_x_m"] - runup:
                errors.append("avoidance: runup_x_m %r is not slip_opening_x_m - approach_runup_m"
                              % (task_constants.get("runup_x_m"),))
    except (KeyError, TypeError) as exc:
        errors.append("avoidance: constants incomplete (%s)" % (exc,))
    telemetry = block.get("telemetry")
    if telemetry is None:
        if header.get("complete"):
            errors.append("avoidance: a complete certificate lacks the layer telemetry")
        return errors
    if not isinstance(telemetry, dict):
        return errors + ["avoidance: telemetry is not an object"]
    counts = {}
    for key in AVOIDANCE_TELEMETRY_COUNTS:
        value = telemetry.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append("avoidance: telemetry %s %r is not a nonnegative integer" % (key, value))
        counts[key] = value if isinstance(value, int) else 0
    if counts["handbacks"] > counts["activations"]:
        errors.append("avoidance: more hand-backs than activations")
    if counts["episodes_with_activation"] > counts["episodes_finished"]:
        errors.append("avoidance: more episodes with an activation than finished episodes")
    if counts["activations"] == 0 and (counts["active_steps"] or counts["plans"] or counts["episodes_with_activation"]):
        errors.append("avoidance: active steps, plans or activated episodes without any activation")
    rows = list(telemetry.get("episode_rows") or []) + list(telemetry.get("unfinished_rows") or [])
    if len(telemetry.get("episode_rows") or []) != counts["episodes_with_activation"]:
        errors.append("avoidance: episode_rows count differs from episodes_with_activation")
    if sum(int(r[2]) for r in rows) != counts["active_steps"] or sum(int(r[3]) for r in rows) != counts["activations"]:
        errors.append("avoidance: episode rows do not add up to the active steps / activations")
    return errors


# G17 pins (G25b, REV-G25 MAJOR-2): the avoidance block must carry exactly the frozen constants of
# scripts/classical_avoidance_v2.json for its vessel and task, no extra keys, and (with a code root) the file digest.
AVOIDANCE_FILE = "scripts/classical_avoidance_v2.json"
AVOIDANCE_BLOCK_KEYS = ("format", "layer_version", "controller_name", "base_controller_name", "base_controller_class",
                        "task", "mode", "information", "motion_prediction", "static_geometry", "file", "file_sha256",
                        "rule_constants", "vessel_constants", "task_constants", "hold_zone_m", "telemetry")
AVOIDANCE_TASK_OF_MODE = {"path_follow": "path_following", "station_keep": "station_keeping", "docking": "docking"}
AVOIDANCE_TELEMETRY_KEYS = AVOIDANCE_TELEMETRY_COUNTS + ("wall_registrations", "closing_steps", "memory_steps",
                                                         "memory_dropped",          # G24b
                                                         "episode_rows_fields", "episode_rows", "unfinished_rows",
                                                         "control_step_s")


def _lf_sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _scripts_dir(code_root):
    return (Path(code_root) / "scripts") if code_root is not None else Path(__file__).resolve().parent


def avoidance_pin_errors(header, code_root=None):
    """G25b: pin the G17 avoidance block to the frozen constants file (complements avoidance_layer_errors).

    The block holds exactly AVOIDANCE_BLOCK_KEYS; its rule / vessel / task constants equal the frozen
    scripts/classical_avoidance_v2.json (of ``code_root`` when given, else of this validator's tree) for the
    header's vessel and the block's task; mode and task agree; motion_prediction is False; the telemetry
    holds exactly AVOIDANCE_TELEMETRY_KEYS; with a code root the file digest equals the file's.
    """
    controller = header.get("controller") or {}
    block = (controller.get("configuration") or {}).get("avoidance")
    if controller.get("name") not in AVOIDANCE_CONTROLLERS or not isinstance(block, dict):
        return []
    errors = []
    keys = set(block)
    if keys != set(AVOIDANCE_BLOCK_KEYS):
        errors.append("avoidance: block keys differ (extra %s, missing %s)" % (sorted(keys - set(AVOIDANCE_BLOCK_KEYS)),
                                                                          sorted(set(AVOIDANCE_BLOCK_KEYS) - keys)))
    if AVOIDANCE_TASK_OF_MODE.get(block.get("mode")) != block.get("task"):
        errors.append("avoidance: mode %r does not belong to task %r" % (block.get("mode"), block.get("task")))
    if block.get("motion_prediction") is not False:
        errors.append("avoidance: motion_prediction must be false")
    if block.get("file") != AVOIDANCE_FILE:
        errors.append("avoidance: constants file %r, expected %r" % (block.get("file"), AVOIDANCE_FILE))
    path = _scripts_dir(code_root) / Path(AVOIDANCE_FILE).name
    try:
        frozen = json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError) as exc:
        return errors + ["avoidance: cannot read the frozen constants %s (%s)" % (path, exc)]
    if code_root is not None and block.get("file_sha256") != _lf_sha256(path):
        errors.append("avoidance: file_sha256 differs from %s" % (path,))
    if block.get("format") != frozen.get("format") or block.get("layer_version") != frozen.get("layer_version"):
        errors.append("avoidance: format / layer version differ from the frozen file")
    if block.get("rule_constants") != frozen.get("rule_constants"):
        errors.append("avoidance: rule_constants differ from the frozen file")
    vessel_block = (frozen.get("vessels") or {}).get(header.get("vessel"))
    if not isinstance(vessel_block, dict):
        return errors + ["avoidance: no frozen constants for vessel %r" % (header.get("vessel"),)]
    if block.get("vessel_constants") != {k: v for k, v in vessel_block.items() if k != "tasks"}:
        errors.append("avoidance: vessel_constants differ from the frozen file")
    if block.get("task_constants") != (vessel_block.get("tasks") or {}).get(block.get("task")):
        errors.append("avoidance: task_constants differ from the frozen file")
    zone = block.get("hold_zone_m")
    if not (isinstance(zone, list) and len(zone) == 2 and all(type(z) in (int, float) for z in zone)
            and 0.0 <= zone[0] < zone[1]):
        errors.append("avoidance: hold_zone_m %r is not [zone, slow] with 0 <= zone < slow" % (zone,))
    telemetry = block.get("telemetry")
    if isinstance(telemetry, dict) and set(telemetry) != set(AVOIDANCE_TELEMETRY_KEYS):
        errors.append("avoidance: telemetry keys differ (extra %s, missing %s)"
                      % (sorted(set(telemetry) - set(AVOIDANCE_TELEMETRY_KEYS)),
                         sorted(set(AVOIDANCE_TELEMETRY_KEYS) - set(telemetry))))
    return errors


# G25 / G25b: the journal planner with the ray-return memory and the circumscribed-radius ladder
# (classical_planner_pid.JournalMemoryPlannerController, scripts/planner_memory.py CONTROLLER_NAME); every such
# certificate carries header.controller.configuration.ray_memory (rule constants, derived constants, ladder rule
# and values, run telemetry). G25b (REV-G25 MAJOR-2): every value is pinned to planner_memory and the registry.
PLANNER_MEMORY_FORMAT = "usvbench-planner-memory-v1"
# G24b: memory g24b_ray_memory_v3 / planner_los_pid_mem_v2 without the expiry rule S2 (no ttl constant, no
# "expired" counter); G25b's g25b_ray_memory_v2 / planner_los_pid_mem_v1 are not accepted.
PLANNER_MEMORY_VERSIONS = ("g24b_ray_memory_v3",)
PLANNER_MEMORY_CONTROLLERS = {"planner_los_pid_mem_v2": "planner_los_pid"}
PLANNER_MEMORY_COUNTS = ("inserted", "refreshed", "carved", "evicted", "resets", "updates",
                         "peak_entries", "legs_planned", "unplannable_legs", "hold_steps",
                         "escape_plans", "escape_failures")         # G24b hold escape (REV-G25b C1)
PLANNER_MEMORY_EPISODE_FIELDS = ("hold_steps", "unplannable_legs", "escape_plans", "escape_failures")
PLANNER_MEMORY_ROW_KEYS = ("legs_by_rung", "episode_rows_fields", "episode_rows", "unfinished_rows")
PLANNER_MEMORY_BLOCK_KEYS = ("format", "memory_version", "rule_constants", "constants", "module_sha256",
                             "controller_name", "base_controller_name", "vessel", "ladder", "telemetry")
PLANNER_MEMORY_CONSTANT_KEYS = ("plan_cell_m", "replan_every_steps", "length_scale", "ray_max_range_m",
                                "reference_speed_mps", "control_dt_s", "quantum_m", "disc_radius_m",
                                "carve_lateral_m", "carve_margin_m", "capacity_entries")
PLANNER_MEMORY_LADDER_KEYS = ("rule", "circumscribed_radius_m", "half_beam_m", "length_scale", "margin_lengths",
                              "narrow_margin_lengths", "inflation_ladder_m")


def _planner_memory_module():
    try:
        import planner_memory
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import planner_memory
    return planner_memory


def planner_memory_errors(header, code_root=None):
    """G25b: header.controller.configuration.ray_memory of the memory-planner certificates.

    A certificate named planner_los_pid_mem_v2 must carry the block and no other controller may. The block
    belongs to a journal hazard_nav / path_hazard certificate, holds exactly PLANNER_MEMORY_BLOCK_KEYS (and
    exact key sets inside), names the controller, its base planner and the header's vessel, and every value
    is PINNED: rule_constants == planner_memory.RULE_CONSTANTS (same types); ladder terms ==
    planner_memory.ladder_terms(vessel) from the registry and the ladder == memory_ladder(terms), rung 1 >= C;
    plan cell == 0.25 s, replan_every == 60, ray range == 30 s, u_ref and control step from the registry;
    derived constants == planner_memory.derive_constants(...) and the configuration's own plan_cell_m /
    replan_every / inflation_ladder_m equal them; with a code root, module_sha256 == that tree's
    scripts/planner_memory.py. A complete certificate's telemetry holds exactly PLANNER_MEMORY_COUNTS +
    legs_by_rung, nonnegative integers that are mutually consistent.
    """
    controller = header.get("controller") or {}
    name = controller.get("name")
    configuration = controller.get("configuration") or {}
    block = configuration.get("ray_memory")
    if name not in PLANNER_MEMORY_CONTROLLERS:
        return [] if block is None else ["controller.configuration.ray_memory on a %r certificate (only the "
                                         "memory planner stamps it)" % (name,)]
    if not isinstance(block, dict):
        return ["%s certificate lacks controller.configuration.ray_memory (G25)" % name]
    pm = _planner_memory_module()
    errors = []
    journal = header.get("journal")
    carrier = journal.get("carrier") if isinstance(journal, dict) else None
    if carrier not in pm.CARRIERS:
        errors.append("ray_memory: %s runs only on journal hazard_nav / path_hazard ids (carrier %r)" % (name, carrier))
    if configuration.get("controller") != name:
        errors.append("ray_memory: configuration.controller %r differs from header %r"
                      % (configuration.get("controller"), name))
    keys = set(block)
    if keys != set(PLANNER_MEMORY_BLOCK_KEYS):
        errors.append("ray_memory: block keys differ (extra %s, missing %s)"
                      % (sorted(keys - set(PLANNER_MEMORY_BLOCK_KEYS)), sorted(set(PLANNER_MEMORY_BLOCK_KEYS) - keys)))
    if block.get("format") != PLANNER_MEMORY_FORMAT or block.get("memory_version") not in PLANNER_MEMORY_VERSIONS:
        errors.append("ray_memory: unknown format / memory version %r / %r" % (block.get("format"),
                                                                              block.get("memory_version")))
    if block.get("controller_name") != name:
        errors.append("ray_memory: block names controller %r, header %r" % (block.get("controller_name"), name))
    if block.get("base_controller_name") != PLANNER_MEMORY_CONTROLLERS[name]:
        errors.append("ray_memory: base controller %r of %s" % (block.get("base_controller_name"), name))
    vessel = header.get("vessel")
    if block.get("vessel") != vessel:
        errors.append("ray_memory: block of vessel %r on a %r certificate" % (block.get("vessel"), vessel))
    rules = block.get("rule_constants")
    if not (isinstance(rules, dict) and set(rules) == set(pm.RULE_CONSTANTS)
            and all(type(rules[k]) is type(v) and rules[k] == v for k, v in pm.RULE_CONSTANTS.items())):
        errors.append("ray_memory: rule_constants %r differ from planner_memory.RULE_CONSTANTS" % (rules,))
    if code_root is not None:
        module = _scripts_dir(code_root) / "planner_memory.py"
        try:
            if block.get("module_sha256") != pm.module_sha256(module):
                errors.append("ray_memory: module_sha256 differs from %s" % (module,))
        except OSError as exc:
            errors.append("ray_memory: cannot read %s (%s)" % (module, exc))
    try:
        terms = pm.ladder_terms(vessel)
        vterms = pm.vessel_terms(vessel)
    except Exception as exc:  # unknown vessel / registry failure
        return errors + ["ray_memory: registry terms of vessel %r unavailable (%s)" % (vessel, exc)]
    ladder = block.get("ladder")
    expected_ladder = pm.memory_ladder(terms)
    if not isinstance(ladder, dict) or set(ladder) != set(PLANNER_MEMORY_LADDER_KEYS):
        errors.append("ray_memory: ladder keys differ from %s" % (list(PLANNER_MEMORY_LADDER_KEYS),))
    else:
        if {k: ladder[k] for k in terms} != terms:
            errors.append("ray_memory: ladder terms %r differ from the registry terms %r"
                          % ({k: ladder.get(k) for k in terms}, terms))
        if ladder.get("inflation_ladder_m") != expected_ladder:
            errors.append("ray_memory: ladder %r differs from its rule on the registry (%r)"
                          % (ladder.get("inflation_ladder_m"), expected_ladder))
    if not expected_ladder or expected_ladder[0] < terms["circumscribed_radius_m"]:
        errors.append("ray_memory: first ladder rung below the circumscribed radius")
    if configuration.get("inflation_ladder_m") != expected_ladder:
        errors.append("ray_memory: configuration inflation_ladder_m differs from the registry ladder")
    scale = vterms["length_scale"]
    cell, replan = pm.PLAN_CELL_REF_M * scale, pm.REPLAN_EVERY_STEPS
    if configuration.get("plan_cell_m") != cell:
        errors.append("ray_memory: plan_cell_m %r is not 0.25 s = %r" % (configuration.get("plan_cell_m"), cell))
    if configuration.get("replan_every") != replan:
        errors.append("ray_memory: replan_every %r is not %d" % (configuration.get("replan_every"), replan))
    expected_constants = pm.derive_constants(pm.RULE_CONSTANTS, cell, replan, scale, vterms["ray_max_range_m"],
                                             vterms["reference_speed_mps"], vterms["control_dt_s"])
    constants = block.get("constants")
    if constants != expected_constants:
        errors.append("ray_memory: constants %r differ from planner_memory.derive_constants on the registry (%r)"
                      % (constants, expected_constants))
    if not isinstance(constants, dict):
        constants = {}
    telemetry = block.get("telemetry")
    if telemetry is None:
        if header.get("complete"):
            errors.append("ray_memory: a complete certificate lacks the memory telemetry")
        return errors
    if not isinstance(telemetry, dict):
        return errors + ["ray_memory: telemetry is not an object"]
    if set(telemetry) != set(PLANNER_MEMORY_COUNTS) | set(PLANNER_MEMORY_ROW_KEYS):
        errors.append("ray_memory: telemetry keys differ (extra %s, missing %s)"
                      % (sorted(set(telemetry) - set(PLANNER_MEMORY_COUNTS) - set(PLANNER_MEMORY_ROW_KEYS)),
                         sorted(set(PLANNER_MEMORY_COUNTS) | set(PLANNER_MEMORY_ROW_KEYS) - set(telemetry))))
    counts = {}
    for key in PLANNER_MEMORY_COUNTS:
        value = telemetry.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append("ray_memory: telemetry %s %r is not a nonnegative integer" % (key, value))
        counts[key] = value if isinstance(value, int) else 0
    by_rung = telemetry.get("legs_by_rung")
    if (not isinstance(by_rung, list) or len(by_rung) != len(expected_ladder)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in by_rung)):
        errors.append("ray_memory: legs_by_rung %r is not one nonnegative count per ladder rung" % (by_rung,))
    elif sum(by_rung) + counts["unplannable_legs"] != counts["legs_planned"]:
        errors.append("ray_memory: legs per rung + unplannable legs differ from the legs planned")
    if counts["carved"] + counts["evicted"] > counts["inserted"]:
        errors.append("ray_memory: more entries removed than inserted")
    if counts["peak_entries"] > constants.get("capacity_entries", 0):
        errors.append("ray_memory: peak entries above the capacity")
    if counts["inserted"] and not counts["updates"]:
        errors.append("ray_memory: entries inserted without any update")
    if counts["hold_steps"] and not counts["unplannable_legs"]:
        errors.append("ray_memory: held control steps without an unplannable leg")
    # G24b (REV-G25b C1 / m2): escape counts and per-episode rows add up to the run totals
    if counts["escape_plans"] + counts["escape_failures"] > counts["unplannable_legs"]:
        errors.append("ray_memory: more escape replans than unplannable legs")
    if telemetry.get("episode_rows_fields") != ["env", "episode"] + list(PLANNER_MEMORY_EPISODE_FIELDS):
        errors.append("ray_memory: episode_rows_fields %r" % (telemetry.get("episode_rows_fields"),))
    rows = list(telemetry.get("episode_rows") or []) + list(telemetry.get("unfinished_rows") or [])
    if not all(isinstance(r, list) and len(r) == 6 and all(type(x) is int and x >= 0 for x in r) for r in rows):
        errors.append("ray_memory: episode rows are not [env, episode, 4 nonnegative counts]")
    else:
        for k, key in enumerate(PLANNER_MEMORY_EPISODE_FIELDS):
            if sum(r[2 + k] for r in rows) != counts[key]:
                errors.append("ray_memory: episode rows do not add up to the run total %s" % key)
    return errors


# G24b (REV-G25b M3): every field of the changed controllers' configuration blocks is pinned, not only the memory /
# avoidance sub-blocks. Torch-free: the frozen gains are read from scripts/classical_gains_v2.json and the information
# text from scripts/journal_controllers.py (ast), both of the code root when given, else of this validator's tree.
PLANNER_CONFIGURATION_KEYS = ("controller", "mode", "replan_every", "waypoint_spacing_m", "accept_radius_m",
                              "plan_cell_m", "inflation_ladder_m", "kp", "ki", "kd", "privileged", "journal",
                              "control_step_s", "ray_memory", "plan_margin_m", "planning_padding_m", "snap_start_m",
                              "turn_creep_thrust", "aligned_thrust_gain", "action_limits", "planning_dtype", "pid_dtype",
                              "num_envs", "device")
PLANNER_JOURNAL_KEYS = ("journal", "vessel", "information", "gains", "length_scale", "ray_max_range_m", "obstacles")
PLANNER_OBSTACLES_TEXT = "current and remembered ray returns as memory discs (G25b planner_memory)"
PLANNER_MODE_OF_CARRIER = {"hazard_nav": "goal", "path_hazard": "gates"}
AVOIDANCE_CONFIGURATION_KEYS = ("journal", "vessel", "information", "gains", "length_scale", "ray_max_range_m", "mode",
                                "hold_mode", "zone_radius_m", "slow_radius_m", "docking_tolerance_m", "controller_class",
                                "avoidance", "protocol", "num_envs", "device", "dt", "action_limits", "dtype")
AVOIDANCE_MODE_OF_TASK = {"path_following": "path_follow", "station_keeping": "station_keep", "docking": "docking"}
PATH_FOLLOWING_CFG_GOAL_RADIUS_M = 2.0      # journal path-following cfg goal_radius (unused by the pass-through law)


def _frozen_gains(vessel, code_root=None):
    """journal_controllers.load_gains(vessel) without torch: the frozen block of scripts/classical_gains_v2.json
    (every key except the host-specific absolute "file" path, whose basename is returned instead)."""
    import hashlib
    path = _scripts_dir(code_root) / "classical_gains_v2.json"
    data = json.loads(path.read_text(encoding="ascii"))
    block = data["vessels"][vessel]
    return {"vessel": vessel, "los_heading_pid": dict(block["los_heading_pid"]["gains"]),
            "dock_yaw_pd": dict(block["dock_yaw_pd"]["gains"]),
            "rules": {k: block[k]["rule"] for k in ("los_heading_pid", "dock_yaw_pd")},
            "procedure_sha256": data.get("procedure_sha256"),
            "file_sha256": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "model_damping": (data.get("model_damping") or {}).get(vessel)}, path.name


def _information_diet(code_root=None):
    import ast
    tree = ast.parse((_scripts_dir(code_root) / "journal_controllers.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INFORMATION_DIET"
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError("INFORMATION_DIET")


def _gains_errors(prefix, gains, vessel, code_root):
    try:
        expected, name = _frozen_gains(vessel, code_root)
    except (OSError, ValueError, KeyError) as exc:
        return ["%s: frozen gains unavailable (%s)" % (prefix, exc)]
    if not isinstance(gains, dict):
        return ["%s: gains is not an object" % prefix]
    errors = []
    if set(gains) != set(expected) | {"file"}:
        errors.append("%s: gains keys differ" % prefix)
    if {k: v for k, v in gains.items() if k != "file"} != expected:
        errors.append("%s: gains differ from the frozen scripts/classical_gains_v2.json" % prefix)
    if Path(str(gains.get("file", ""))).name != name:
        errors.append("%s: gains file %r is not %s" % (prefix, gains.get("file"), name))
    return errors


def planner_configuration_errors(header, code_root=None):
    """G24b: every field of a planner_los_pid_mem_v2 certificate's controller.configuration (REV-G25b M3)."""
    controller = header.get("controller") or {}
    if controller.get("name") not in PLANNER_MEMORY_CONTROLLERS:
        return []
    c = controller.get("configuration") or {}
    pm = _planner_memory_module()
    vessel = header.get("vessel")
    errors = []
    if set(c) != set(PLANNER_CONFIGURATION_KEYS):
        errors.append("planner configuration: keys differ (extra %s, missing %s)"
                      % (sorted(set(c) - set(PLANNER_CONFIGURATION_KEYS)), sorted(set(PLANNER_CONFIGURATION_KEYS) - set(c))))
    try:
        terms = pm.vessel_terms(vessel)
    except Exception as exc:  # unknown vessel
        return errors + ["planner configuration: registry terms of %r unavailable (%s)" % (vessel, exc)]
    s = terms["length_scale"]
    journal = header.get("journal")
    carrier = journal.get("carrier") if isinstance(journal, dict) else None
    expected = {"mode": PLANNER_MODE_OF_CARRIER.get(carrier), "waypoint_spacing_m": 2.0 * s, "accept_radius_m": 1.0 * s,
                "privileged": False, "control_step_s": terms["control_dt_s"], "planning_padding_m": 6.0,
                "snap_start_m": pm.SNAP_START_REF_M * s, "turn_creep_thrust": 0.2, "aligned_thrust_gain": 0.8,
                "action_limits": [-1.0, 1.0], "planning_dtype": "numpy.float64", "plan_margin_m": 0.15}
    for key, value in expected.items():
        if c.get(key) != value:
            errors.append("planner configuration: %s %r is not %r" % (key, c.get(key), value))
    try:
        frozen, _name = _frozen_gains(vessel, code_root)
        for key in ("kp", "ki", "kd"):
            if c.get(key) != frozen["los_heading_pid"][key]:
                errors.append("planner configuration: %s %r differs from the frozen LOS gain" % (key, c.get(key)))
    except (OSError, ValueError, KeyError) as exc:
        errors.append("planner configuration: frozen gains unavailable (%s)" % (exc,))
    if c.get("pid_dtype") not in ("torch.float32", "torch.float64"):
        errors.append("planner configuration: pid_dtype %r" % (c.get("pid_dtype"),))
    if isinstance(c.get("num_envs"), bool) or not isinstance(c.get("num_envs"), int) or c.get("num_envs") < 1:
        errors.append("planner configuration: num_envs %r" % (c.get("num_envs"),))
    if not isinstance(c.get("device"), str):
        errors.append("planner configuration: device %r" % (c.get("device"),))
    j = c.get("journal")
    if not isinstance(j, dict) or set(j) != set(PLANNER_JOURNAL_KEYS):
        return errors + ["planner configuration: journal block keys differ from %s" % (list(PLANNER_JOURNAL_KEYS),)]
    try:
        info = _information_diet(code_root)
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        info = None
        errors.append("planner configuration: information text unavailable (%s)" % (exc,))
    for key, value in (("journal", True), ("vessel", vessel), ("length_scale", s),
                       ("ray_max_range_m", terms["ray_max_range_m"]), ("obstacles", PLANNER_OBSTACLES_TEXT),
                       ("information", info)):
        if j.get(key) != value:
            errors.append("planner configuration: journal.%s %r is not %r" % (key, j.get(key), value))
    errors.extend(_gains_errors("planner configuration: journal", j.get("gains"), vessel, code_root))
    return errors


def avoidance_configuration_errors(header, code_root=None):
    """G24b: every field of a los_pid_avoid_v2 / los_pid_cont_avoid_v2 / docking_runup_pid_avoid_v2 certificate's
    controller.configuration outside the avoidance block (REV-G25b M3; classical_baseline.py journal block)."""
    controller = header.get("controller") or {}
    name = controller.get("name")
    if name not in AVOIDANCE_CONTROLLERS:
        return []
    c = controller.get("configuration") or {}
    block = c.get("avoidance") if isinstance(c.get("avoidance"), dict) else {}
    task = block.get("task")
    vessel = header.get("vessel")
    errors = []
    if set(c) != set(AVOIDANCE_CONFIGURATION_KEYS):
        errors.append("avoidance configuration: keys differ (extra %s, missing %s)"
                      % (sorted(set(c) - set(AVOIDANCE_CONFIGURATION_KEYS)), sorted(set(AVOIDANCE_CONFIGURATION_KEYS) - set(c))))
    if task not in AVOIDANCE_MODE_OF_TASK:
        return errors + ["avoidance configuration: unknown task %r" % (task,)]
    pm = _planner_memory_module()
    try:
        terms = pm.vessel_terms(vessel)
        params = pm.shared_module("journal_cfg_v2").plan_class(task, vessel)(geometry_tier=2).params
    except Exception as exc:  # unknown vessel / plan
        return errors + ["avoidance configuration: registry / plan of %r unavailable (%s)" % (vessel, exc)]
    s = terms["length_scale"]
    zone = {"station_keeping": params.get("hold_radius_m"), "docking": params.get("position_tolerance_m"),
            "path_following": PATH_FOLLOWING_CFG_GOAL_RADIUS_M}[task]
    expected = {"journal": True, "vessel": vessel, "length_scale": s, "ray_max_range_m": terms["ray_max_range_m"],
                "mode": AVOIDANCE_MODE_OF_TASK[task],
                "hold_mode": AVOIDANCE_STATION_HOLD_MODE if task == "station_keeping" else "deadband",
                "zone_radius_m": zone, "slow_radius_m": (6.0 if task == "docking" else 5.0) * s,
                "docking_tolerance_m": params["position_tolerance_m"] if task == "docking" else 2.5 * s,
                "controller_class": "journal_avoidance.AvoidanceLayerController", "protocol": "mission",
                "dt": terms["control_dt_s"], "action_limits": [-1.0, 1.0]}
    try:
        expected["information"] = _information_diet(code_root)
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        errors.append("avoidance configuration: information text unavailable (%s)" % (exc,))
    for key, value in expected.items():
        if c.get(key) != value:
            errors.append("avoidance configuration: %s %r is not %r" % (key, c.get(key), value))
    if c.get("dtype") not in ("torch.float32", "torch.float64"):
        errors.append("avoidance configuration: dtype %r" % (c.get("dtype"),))
    if isinstance(c.get("num_envs"), bool) or not isinstance(c.get("num_envs"), int) or c.get("num_envs") < 1:
        errors.append("avoidance configuration: num_envs %r" % (c.get("num_envs"),))
    if not isinstance(c.get("device"), str):
        errors.append("avoidance configuration: device %r" % (c.get("device"),))
    errors.extend(_gains_errors("avoidance configuration", c.get("gains"), vessel, code_root))
    return errors


def legacy_tree_digest(tasks_sha256, scripts_sha256):
    """C5 combined code digest; must match cert_v3.legacy_tree_digest."""
    import hashlib
    return hashlib.sha256(("tasks\0" + tasks_sha256 + "\nscripts\0" + scripts_sha256 + "\n")
                          .encode("ascii")).hexdigest()


def code_errors(document, minor, *, code_root=None, release_manifest=None, require_verified=False):
    """D2 items 9/10: recompute or cross-check the code identity block."""
    errors = []
    code = document["header"]["code"]
    if code["hash_algorithm"] == LEGACY_HASH_ALGORITHM:
        if legacy_tree_digest(code["tasks_sha256"], code["scripts_sha256"]) != code["tree_sha256"]:
            errors.append("code.tree_sha256 is inconsistent with its tasks/scripts component digests")
        if require_verified and minor >= 2:
            errors.append("UNVERIFIED code identity: legacy directory hash is not a release code tree")
        if code_root is not None or release_manifest is not None:
            errors.append("a legacy directory hash cannot be recomputed as a usvbench-code-tree-v1 digest")
        return errors
    if code["inclusion"] != code_tree.INCLUSION_SPEC:
        errors.append("code.inclusion differs from the %s specification" % code_tree.INCLUSION_VERSION)
    loaded = code["loaded_modules"]
    if loaded["verified"] != (loaded["mismatch_count"] == 0) or loaded["mismatch_count"] < len(loaded["mismatches"]):
        errors.append("code.loaded_modules verdict is inconsistent with its mismatch list")
    release_fields = [code[key] for key in ("release_id", "git_commit", "git_tree", "release_descriptor")]
    if code["source"] == "release_descriptor":
        if any(value is None for value in release_fields) or code["code_tree_sha256"] is None:
            errors.append("release code identity requires release_id, git ids, descriptor and code tree")
        elif code["release_id"] != "%s-%s" % (code["git_commit"][:12], code["code_tree_sha256"][:12]):
            errors.append("code.release_id does not match its git_commit and code_tree_sha256")
        if not loaded["verified"]:
            errors.append("release code identity with unverified loaded modules")
    else:
        if any(value is not None for value in release_fields):
            errors.append("release fields are set although the code source is %s" % code["source"])
        if (code["source"] == "unavailable") != (code["code_tree_sha256"] is None):
            errors.append("code.code_tree_sha256 presence contradicts code.source")
        if require_verified:
            errors.append("UNVERIFIED code identity: %s, not a release" % code["source"])
    if code_root is not None:
        try:
            _, records, digest = code_tree.manifest_from_tree(code_root)
        except (OSError, code_tree.CodeTreeError) as exc:
            errors.append("cannot recompute the code tree at %s: %s" % (code_root, exc))
        else:
            if digest != code["code_tree_sha256"] or len(records) != code["file_count"]:
                errors.append("recomputed code tree %s (%d files) differs from the certificate"
                              % (digest, len(records)))
    if release_manifest is not None:
        try:
            records, digest = code_tree.parse_manifest(Path(release_manifest).read_bytes())
        except (OSError, code_tree.CodeTreeError) as exc:
            errors.append("cannot read the release manifest: %s" % exc)
        else:
            if digest != code["code_tree_sha256"] or len(records) != code["file_count"]:
                errors.append("release manifest %s (%d files) differs from the certificate"
                              % (digest, len(records)))
    return errors


def _resolved_physics(config):
    nested = config.get("underwater_physics_cfg")
    nested = nested if isinstance(nested, dict) else {}
    version = nested.get("physics_version", config.get("physics_version"))
    hydro = nested.get("hydrodynamics_model", config.get("hydrodynamics_model"))
    return nested, version, hydro


def physics_vessel_errors(header, minor):
    """D2 items 8/10: physics version, vessel, hydrodynamics model and profile vs resolved_config."""
    errors = []
    config = header["resolved_config"]
    nested, version, hydro = _resolved_physics(config)
    declared = header["physics_version"]
    if declared is not None and version is not None and str(version) != str(declared):
        errors.append("header physics_version %r differs from resolved_config %r" % (declared, version))
    if minor >= 2 and declared is None and version is not None:
        errors.append("resolved_config declares physics_version %r but the header has none" % (version,))
    vehicle = config.get("vehicle")
    if isinstance(vehicle, str) and header["vessel"] is not None and header["vessel"] != vehicle:
        errors.append("header vessel %r differs from resolved_config vehicle %r" % (header["vessel"], vehicle))
    if minor >= 2:
        expected_hydro = "legacy" if hydro is None else hydro
        if header["hydrodynamics_model"] != expected_hydro:
            errors.append("header hydrodynamics_model %r differs from resolved_config %r"
                          % (header["hydrodynamics_model"], expected_hydro))
        if header["hydrodynamics_model"] == "hydro_v2" and declared != "2":
            errors.append("hydrodynamics_model hydro_v2 requires physics_version 2")
        profile = header["vessel_profile"]
        declared_profile = nested.get("vessel_profile", config.get("vessel_profile"))
        if profile is None:
            if declared_profile is not None:
                errors.append("resolved_config declares a vessel_profile but the header has none")
        else:
            name = declared_profile if isinstance(declared_profile, str) else (
                declared_profile.get("name") if isinstance(declared_profile, dict) else None)
            if name != profile["name"]:
                errors.append("header vessel_profile %r differs from resolved_config %r" % (profile["name"], name))
    return errors


def reference_environment_errors(header):
    """G6b / CONTEXT_R4 D28 (b): an official (non-diagnostic) certificate needs the RE-1 block with
    match true; the block, when present, must be internally consistent (match iff no mismatch,
    mismatches recomputed from versions vs required). Diagnostic certificates are accepted."""
    block = header["runtime"].get("reference_environment")
    registration = header.get("seed_namespace")
    if registration is None and block is None:
        return []                     # schema 3.0 (pre-D8) document: no registration, no RE-1 rule
    official = (registration or {}).get("purpose") != reference_environment.DIAGNOSTIC_PURPOSE
    errors = []
    if block is None:
        if official:
            errors.append("official certificate lacks runtime.reference_environment (CONTEXT_R4 D28)")
        return errors
    required, versions = block.get("required") or {}, block.get("versions") or {}
    names = set(reference_environment.NAMES)
    if set(required) != names or set(versions) != names:
        errors.append("runtime.reference_environment must list exactly %s" % ", ".join(reference_environment.NAMES))
        return errors
    expected = reference_environment.mismatches(versions, required)
    if block["mismatches"] != expected or block["match"] != (not expected):
        errors.append("runtime.reference_environment match/mismatches disagree with its versions")
    if official and not block["match"]:
        errors.append("official certificate from outside the reference environment %s (CONTEXT_R4 D28): %s"
                      % (block.get("id"), "; ".join(block["mismatches"])))
    return errors


def level_distance_errors(header):
    """D2 item 8: docking level table in force and realised distance (minor 2)."""
    errors = []
    config = header["resolved_config"]
    stamp = header["level_distance"]
    if "eval_level_distances" not in config:
        if stamp is not None:
            errors.append("level_distance stamped for a task without a docking level table")
        return errors
    if stamp is None:
        return ["docking certificate lacks the level_distance stamp"]
    if stamp.get("table_field") == "journal_plan":
        # F5 D2 item 14 (applied by F7 round 4b): journal docking reads no level table;
        # the env stamps the frozen journal plan instead of a distance.
        if int(config.get("journal_contract_version", 0) or 0) != 2 or "journal" not in header:
            errors.append("level_distance journal_plan stamped for a non-journal docking certificate")
        if stamp.get("docking_level_table_in_force") != "journal_plan":
            errors.append("level_distance journal_plan without docking_level_table_in_force journal_plan")
        return errors
    if int(config.get("journal_contract_version", 0) or 0) == 2:
        return ["journal docking certificate must stamp level_distance table_field journal_plan "
                "(docking_level_table_in_force)"]
    field = "eval_level_distances"
    if header["physics_version"] == "2" and "eval_level_distances_v2" in config:
        field = "eval_level_distances_v2"
    table = [float(x) for x in config[field]]
    index = min(max(int(stamp["eval_level"]), 0), len(table) - 1)
    override = float(config.get("spawn_distance_override_m", 0.0) or 0.0)
    expected = override if override > 0.0 else table[index]
    if (stamp["table_field"], stamp["table_m"], stamp["table_index"]) != (field, table, index):
        errors.append("level_distance table differs from the resolved configuration (%s)" % field)
    if stamp["eval_level"] != header["level"]:
        errors.append("level_distance eval_level differs from header level")
    if not math.isclose(stamp["expected_distance_m"], expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        errors.append("level_distance expected distance differs from the resolved table")
    realised = stamp["realised_distance_m"]
    if realised is not None and not math.isclose(realised, expected, rel_tol=1e-6, abs_tol=1e-6):
        errors.append("realised docking spawn distance %r differs from the level table value %r"
                      % (realised, expected))
    return errors


def statistics_errors(stats, label):
    """D2 item 20: internal consistency of one timing statistics object."""
    if stats.get("n") in (0, None):
        return []
    n = stats["n"]
    lo, hi, mean, std = stats["min"], stats["max"], stats["mean"], stats["std"]
    errors = []
    ordered = [stats[k] for k in ("min", "p50", "p95", "p99", "max")]
    if ordered != sorted(ordered) or not lo <= mean <= hi:
        errors.append(label + "timing statistics are inconsistent")
    if not math.isclose(stats["total"], mean * n, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        errors.append(label + "timing total != mean * n")
    spread = hi - lo
    if n == 1 and (std != 0 or spread != 0):
        errors.append(label + "timing std/min/max impossible for n=1")
    if std < 0 or std > spread / 2 * (1 + REL_TOL) + ABS_TOL:
        errors.append(label + "timing std exceeds (max - min) / 2")
    if (spread > 0) != (std > 0):
        errors.append(label + "timing std must be zero exactly when min == max")
    return errors


def contact_errors(record, header, prefix):
    """D2 item 20: contact durations consistent with steps and within the episode."""
    errors = []
    steps, seconds, longest = record["contact_steps"], record["contact_seconds"], record["contact_longest_seconds"]
    latency = header.get("controller_latency") or {}
    dt = latency.get("control_step_s")
    if steps is not None:
        for name, value in (("contact_seconds", seconds), ("contact_longest_seconds", longest)):
            if value is not None and steps == 0 and value != 0:
                errors.append(prefix + name + " must be zero when contact_steps is zero")
        if steps > 0 and seconds is not None and seconds <= 0:
            errors.append(prefix + "contact_seconds must be positive when contact_steps is positive")
        if dt is not None and seconds is not None and not math.isclose(
                seconds, steps * dt, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            errors.append(prefix + "contact_seconds != contact_steps * control_step_s")
    if seconds is not None:
        tolerance = (dt or 0.0) + 1e-9
        elapsed = record.get("episode_elapsed_s")
        if elapsed is not None and seconds > elapsed + tolerance:
            errors.append(prefix + "contact duration exceeds the episode length")
        deadline = header.get("resolved_deadline_s")
        if deadline is not None and seconds > deadline + tolerance:
            errors.append(prefix + "contact duration exceeds the episode deadline")
    return errors


def alias_errors(document, minor):
    """D2 item 10: legacy controller/checkpoint aliases must name the header's values."""
    errors = []
    header = document["header"]
    if "controller" in document and document["controller"] != header["controller"]["name"]:
        errors.append("legacy alias disagrees with header: controller")
    if "checkpoint" in document:
        alias, claimed = document["checkpoint"], header["checkpoint"]["path"]
        if (alias is None) != (claimed is None):
            errors.append("legacy alias disagrees with header: checkpoint (presence)")
        elif alias is not None:
            text, full = str(alias).replace("\\", "/"), claimed.replace("\\", "/")
            absolute = text.startswith("/") or (len(text) > 1 and text[1] == ":")
            if absolute:
                same = text == full if minor >= 2 else text.rsplit("/", 1)[-1] == full.rsplit("/", 1)[-1]
            else:
                same = full.endswith("/" + text)
            if not same:
                errors.append("legacy alias disagrees with header: checkpoint")
    return errors


def cell_config_errors(header):
    """Strict mode: the run_cell plan digest recomputed from its canonical JSON."""
    import hashlib
    cell = header.get("cell_config")
    if cell is None:
        return []
    canonical = (json.dumps(cell["configuration"], indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != cell["sha256"]:
        return ["cell_config.sha256 does not match its configuration"]
    return []


def condition_digest(parameters):
    """Canonical digest; must match tasks/_shared/conditions/stamp.py."""
    import hashlib
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


_PURE_MODULES = {}


def conditions_modules():
    """(certificate, pairing, d7 stamp) modules of tasks/_shared/conditions, pure Python.

    Loaded by path under private names (no torch, no simulator, and the
    ``tasks`` package initialisers are never executed): certificate.py for
    verify_dynamic_hazard_stamp (E3), pairing.py for the D11 primitive hashes
    (F0), dynamic/stamp.py for read_stamp (D7/E4 schema and model versions).
    """
    if _PURE_MODULES:
        return _PURE_MODULES["certificate"], _PURE_MODULES["pairing"], _PURE_MODULES["d7"]
    import importlib
    import importlib.util
    import types
    root = Path(__file__).resolve().parents[1] / "tasks" / "_shared" / "conditions"
    if not (root / "pairing.py").is_file() or not (root / "dynamic" / "stamp.py").is_file():
        raise ValueError("cannot verify journal certificate stamps: %s is missing" % root)
    private = "_f7_validator_conditions"
    if private not in sys.modules:
        package = types.ModuleType(private)
        package.__path__ = [str(root)]
        sys.modules[private] = package
    certificate = importlib.import_module(private + ".certificate")
    pairing = importlib.import_module(private + ".pairing")
    spec = importlib.util.spec_from_file_location("_f7_validator_d7_stamp", root / "dynamic" / "stamp.py")
    d7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d7)
    _PURE_MODULES.update(certificate=certificate, pairing=pairing, d7=d7)
    return certificate, pairing, d7


# G16: wind-coefficient stamp of journal certificates (tasks/_shared/conditions/wind_coefficients.py)
WIND_COEFFICIENT_SCHEMA = "usvbench.wind_coefficients/2"
WIND_COEFFICIENT_NAMES = ("c_x0", "c_y0", "c_n0")
_REGISTRY = {}


def vessel_registry_module():
    """tasks/_shared/vehicles_v2.py (pure Python) loaded by path under a private package name."""
    if "vehicles_v2" not in _REGISTRY:
        import importlib
        import types
        root = Path(__file__).resolve().parents[1] / "tasks" / "_shared"
        if not (root / "vehicles_v2.py").is_file():
            raise ValueError("cannot verify pinned wind coefficients: %s is missing" % root)
        private = "_g16_validator_shared"
        if private not in sys.modules:
            package = types.ModuleType(private)
            package.__path__ = [str(root)]
            sys.modules[private] = package
        _REGISTRY["vehicles_v2"] = importlib.import_module(private + ".vehicles_v2")
    return _REGISTRY["vehicles_v2"]


def registry_pinned_wind_coefficients(profile):
    windage = vessel_registry_module().get_vehicle_v2(profile).windage
    return {name: float(getattr(windage, name)) for name in WIND_COEFFICIENT_NAMES}


def _resolved_wind(stamp):
    context = stamp.get("context") if isinstance(stamp, dict) else None
    resolved = context.get("resolved") if isinstance(context, dict) else None
    wind = resolved.get("wind") if isinstance(resolved, dict) else None
    return wind if isinstance(wind, dict) else {}


def _stamped_coefficients(wind):
    windage = wind.get("windage") if isinstance(wind.get("windage"), dict) else {}
    return {name: windage.get(name) for name in WIND_COEFFICIENT_NAMES}


def wind_coefficients_overridden(header):
    """G16: True when the certificate's wind coefficients are not the vessel's pinned values."""
    block = header.get("wind_coefficients")
    return isinstance(block, dict) and (block.get("override") is not None or block.get("pinned_values") is not True)


# T25 (owner decision T1.35b): roll/pitch damping stamp of journal certificates
# (tasks/_shared/conditions/rollpitch_override.py, hydro_profiles.py rollpitch_damping rule)
ROLLPITCH_SCHEMA = "usvbench.rollpitch_damping/1"
ROLLPITCH_ZETA_BOUND = 2.0


def rollpitch_damping_overridden(header):
    """T25: True when the certificate's roll/pitch damping is not the journal default."""
    block = header.get("rollpitch_damping")
    return isinstance(block, dict) and (block.get("override") is not None or block.get("pinned_values") is not True)


def registry_rollpitch_values(profile, rule):
    """(roll, pitch) linear damping of a hydro_profiles rollpitch_damping rule label, from the registry."""
    spec = vessel_registry_module().get_vehicle_v2(profile)
    if isinstance(rule, str) and rule.startswith("zeta:"):
        ratio = float(rule[len("zeta:"):])
        h, inertia = spec.hydrostatics, spec.inertia_com_kg_m2
        return (2.0 * ratio * math.sqrt(h.stiffness_roll_nm_rad * inertia[0][0]),
                2.0 * ratio * math.sqrt(h.stiffness_pitch_nm_rad * inertia[1][1]))
    if rule in ("registry", "legacy_470"):
        return float(spec.damping.linear[3]), float(spec.damping.linear[4])
    if isinstance(rule, str) and rule.startswith("value:"):
        value = float(rule[len("value:"):])
        return value, value
    raise ValueError("unknown roll/pitch damping rule %r" % (rule,))


def rollpitch_damping_errors(document):
    """T25: the rollpitch_damping stamp agrees with itself, the registry and the condition stamp, and an
    official (non-diagnostic) certificate uses the journal-default roll/pitch damping. Certificates written
    before T25 carry no stamp and stay valid."""
    header = document["header"]
    block = header.get("rollpitch_damping")
    if block is None:
        return []
    errors = []
    if "journal" not in header:
        errors.append("header.rollpitch_damping on a non-journal certificate")
    if not isinstance(block, dict) or block.get("schema") != ROLLPITCH_SCHEMA:
        return errors + ["header.rollpitch_damping.schema must be %s" % ROLLPITCH_SCHEMA]
    registration = header.get("seed_namespace")
    official = not reference_environment.is_diagnostic(registration)
    purpose = registration.get("purpose") if isinstance(registration, dict) else None
    pinned, used, override = block.get("pinned"), block.get("used"), block.get("override")
    if not (isinstance(pinned, dict) and isinstance(used, dict)):
        return errors + ["header.rollpitch_damping.pinned / used must be objects"]
    zeta = None
    if override is not None:
        value = override.get("rollpitch_zeta") if isinstance(override, dict) else None
        if not isinstance(override, dict) or set(override) != {"rollpitch_zeta"} or isinstance(value, bool) or \
                not isinstance(value, (int, float)) or not math.isfinite(value) or \
                not 0.0 < value <= ROLLPITCH_ZETA_BOUND:
            return errors + ["header.rollpitch_damping.override %r is not {'rollpitch_zeta': 0 < zeta <= %r}"
                             % (override, ROLLPITCH_ZETA_BOUND)]
        zeta = float(value)
    expected_rule = ("zeta:%r" % zeta) if zeta is not None else pinned.get("rule")
    if used.get("rule") != expected_rule:
        errors.append("header.rollpitch_damping.used.rule %r is not %r" % (used.get("rule"), expected_rule))
    if block.get("pinned_values") is not (override is None and used == pinned):
        errors.append("header.rollpitch_damping.pinned_values contradicts used vs pinned / override")
    profile = block.get("vessel_profile")
    stamp = header.get("conditions") if isinstance(header.get("conditions"), dict) else {}
    context = stamp.get("context") if isinstance(stamp.get("context"), dict) else {}
    if isinstance(context.get("vehicle"), str) and profile != context.get("vehicle"):
        errors.append("header.rollpitch_damping.vessel_profile %r differs from the condition stamp vehicle %r"
                      % (profile, context.get("vehicle")))
    for side, values in (("pinned", pinned), ("used", used)):
        try:
            expected = registry_rollpitch_values(profile, values.get("rule"))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append("cannot verify header.rollpitch_damping.%s of %r: %s" % (side, profile, exc))
            continue
        if (values.get("roll_nms_rad"), values.get("pitch_nms_rad")) != expected:
            errors.append("header.rollpitch_damping.%s values %r differ from the registry rule %r (%r)"
                          % (side, values, values.get("rule"), expected))
    rotational = context.get("rotational_damping") if isinstance(context.get("rotational_damping"), dict) else None
    if rotational is not None:
        stamped = list(rotational.get("angular_linear_damping") or [])[:2]
        if stamped != [used.get("roll_nms_rad"), used.get("pitch_nms_rad")]:
            errors.append("condition stamp rotational_damping %r differs from header.rollpitch_damping.used" % stamped)
    if official and (override is not None or block.get("pinned_values") is not True):
        errors.append("official certificate (seed purpose %r) with roll/pitch damping %r instead of the journal "
                      "default %r: non-official (the hydro.rollpitch_zeta override is diagnostic only, T1.35b)"
                      % (purpose, used, pinned))
    return errors


def certificate_status(document):
    """G16: {"official": bool, "reasons": [...]}: why a certificate is not an official one.

    Non-official reasons: seed purpose 'diagnostic' (e.g. DIAG42), overridden wind coefficients
    (windcoef.*), a run outside the reference environment (D28). validate_certificate reports an
    error when a certificate claims an official purpose but has one of the other two reasons.
    """
    header = document.get("header", {}) if isinstance(document, dict) else {}
    reasons = []
    registration = header.get("seed_namespace")
    if reference_environment.is_diagnostic(registration):
        reasons.append("seed purpose diagnostic")
    if wind_coefficients_overridden(header):
        block = header["wind_coefficients"]
        reasons.append("wind coefficients %r instead of the pinned %r" % (block.get("used"), block.get("pinned")))
    if rollpitch_damping_overridden(header):         # T25 (T1.35b)
        block = header["rollpitch_damping"]
        reasons.append("roll/pitch damping %r instead of the journal default %r" % (block.get("used"),
                                                                                   block.get("pinned")))
    appearing = header.get("appearing_hazards_header")
    if isinstance(appearing, dict) and appearing.get("gap_override_m") is not None:     # G20b
        reasons.append("appearing trigger gap overridden to %r m (appearing.gap_m, diagnostic)"
                       % (appearing["gap_override_m"],))
    if isinstance(appearing, dict) and type(appearing.get("dose")) is int and appearing["dose"] > 4:     # G20c
        reasons.append("supplementary appearing dose D%d (calibration rung, diagnostic)" % appearing["dose"])
    block = (header.get("runtime") or {}).get("reference_environment")
    if isinstance(block, dict) and block.get("match") is False:
        reasons.append("outside the reference environment %s" % block.get("id"))
    return {"official": not reasons, "reasons": reasons}


def wind_coefficient_errors(document):
    """G16: the wind_coefficients stamp is consistent with the condition stamps and the registry, and
    an official (non-diagnostic) certificate uses the vessel's pinned coefficients."""
    errors = []
    header = document["header"]
    block = header.get("wind_coefficients")
    registration = header.get("seed_namespace")
    official = not reference_environment.is_diagnostic(registration)
    purpose = registration.get("purpose") if isinstance(registration, dict) else None
    stamp = header.get("conditions") if isinstance(header.get("conditions"), dict) else {}
    wind = _resolved_wind(stamp)
    context = stamp.get("context") if isinstance(stamp.get("context"), dict) else {}
    profile = context.get("vehicle")
    journal = "journal" in header
    if block is None:
        # A certificate without the stamp (code before G16 had no override): the coefficients of an
        # enabled wind suite must be the registry's pinned values in an official certificate.
        if journal and wind.get("enabled") and official and isinstance(profile, str):
            try:
                pinned = registry_pinned_wind_coefficients(profile)
            except (ValueError, KeyError) as exc:
                return ["cannot verify the wind coefficients of %r: %s" % (profile, exc)]
            if _stamped_coefficients(wind) != pinned:
                errors.append("official certificate (seed purpose %r) with wind coefficients %r instead of the "
                              "pinned %r (G16)" % (purpose, _stamped_coefficients(wind), pinned))
        return errors
    if not journal:
        errors.append("header.wind_coefficients on a non-journal certificate")
    if block.get("schema") != WIND_COEFFICIENT_SCHEMA:
        return errors + ["header.wind_coefficients.schema must be %s" % WIND_COEFFICIENT_SCHEMA]
    pinned, used, override = block["pinned"], block["used"], block["override"]
    expected = dict(pinned)
    expected.update({k: v for k, v in (override or {}).items() if k in WIND_COEFFICIENT_NAMES})
    offset = (override or {}).get("offset_sl")
    centroid = block.get("centroid_offset") if isinstance(block.get("centroid_offset"), dict) else {}
    if centroid.get("enabled") is not (offset is not None) or centroid.get("offset_sl") != offset:
        errors.append("header.wind_coefficients.centroid_offset %r contradicts the override %r" % (centroid, override))
    if used != expected:
        errors.append("header.wind_coefficients.used %r is not the pinned values updated by the override %r"
                      % (used, expected))
    if block["pinned_values"] != (used == pinned and offset is None):
        errors.append("header.wind_coefficients.pinned_values contradicts used vs pinned")
    if isinstance(profile, str) and block["vessel_profile"] != profile:
        errors.append("header.wind_coefficients.vessel_profile %r differs from the condition stamp vehicle %r"
                      % (block["vessel_profile"], profile))
    try:
        registry = registry_pinned_wind_coefficients(block["vessel_profile"])
        if registry != pinned:
            errors.append("header.wind_coefficients.pinned %r differs from the registry %r" % (pinned, registry))
    except (ValueError, KeyError) as exc:
        errors.append("cannot verify the pinned wind coefficients of %r: %s" % (block["vessel_profile"], exc))
    if block["wind_enabled"] != bool(wind.get("enabled")):
        errors.append("header.wind_coefficients.wind_enabled differs from the condition stamp")
    if wind.get("enabled"):
        if _stamped_coefficients(wind) != used:
            errors.append("condition stamp windage coefficients %r differ from header.wind_coefficients.used %r"
                          % (_stamped_coefficients(wind), used))
        if wind.get("coefficient_override") != override:
            errors.append("condition stamp coefficient_override differs from header.wind_coefficients.override")
    for index, record in enumerate(document["records"]):
        record_wind = _resolved_wind(record.get("conditions"))
        if record_wind.get("enabled") and _stamped_coefficients(record_wind) != used:
            errors.append("records/%d: condition stamp windage coefficients differ from "
                          "header.wind_coefficients.used" % index)
    if official and (override is not None or block["pinned_values"] is not True):
        errors.append("official certificate (seed purpose %r) with wind coefficients %r instead of the pinned %r: "
                      "non-official (the windcoef.* override is diagnostic only, G16)" % (purpose, used, pinned))
    return errors


def green_water_rule_module():
    """G10: tasks/_shared/conditions/green_water_rule.py (pure Python), loaded like conditions_modules()."""
    import importlib
    conditions_modules()
    return importlib.import_module("_f7_validator_conditions.green_water_rule")


def green_water_errors(document):
    """G10 (owner rule A + B): journal certificates whose condition stack includes waves carry
    header.green_water (rule, formula stamp, detection definition, cell summary) and a green_water
    record per episode; the cell summary and the verdict are recomputed here
    (conditions/green_water_rule.header_errors) and the formula inputs must be the stamped wave dose.
    Certificates without waves (legacy, calm, wave-free journal cells) carry neither and stay valid."""
    header = document["header"]
    block = header.get("green_water")
    stamp = header["conditions"]
    waves = (stamp.get("parameters") or {}).get("waves") or {}
    waves_on = waves.get("enabled") is True
    declaration = header.get("pairing")
    journal = "journal" in header or (isinstance(declaration, dict) and declaration.get("schema") == PAIRING_SCHEMA)
    carried = [i for i, r in enumerate(document["records"]) if "green_water" in r]
    if block is None:
        errors = []
        if journal and waves_on and stamp.get("schema") == CONDITION_STAMP_SCHEMA_V3:
            errors.append("journal certificate with waves lacks header.green_water (G10 green-water rule A + B)")
        if carried:
            errors.append("records/%d: green_water record without header.green_water" % carried[0])
        return errors
    if not waves_on:
        return ["header.green_water on a certificate whose condition stack has no waves"]
    try:
        rule = green_water_rule_module()
    except (ValueError, ImportError) as exc:
        return ["cannot verify header.green_water: %s" % exc]
    errors = list(rule.header_errors(block, document["records"]))
    formula = block.get("formula") if isinstance(block.get("formula"), dict) else {}
    inputs = formula.get("inputs") if isinstance(formula.get("inputs"), dict) else {}
    keys = ("hs_m",) if waves.get("hs_m") == 0 else ("hs_m", "tp_s", "gamma", "spreading_rad")
    for key in keys:
        if inputs.get(key) != waves.get(key):
            errors.append("header.green_water.formula.inputs.%s = %r differs from the condition stamp %r"
                          % (key, inputs.get(key), waves.get(key)))
    if waves.get("hs_m") == 0 and formula.get("probability") != 0.0:
        errors.append("header.green_water.formula.probability must be 0 at Hs = 0 (calm surface)")
    journal_block = header.get("journal")
    if isinstance(journal_block, dict) and formula.get("vessel") != journal_block.get("vehicle"):
        errors.append("header.green_water.formula.vessel differs from header.journal.vehicle")
    if isinstance(journal_block, dict) and block.get("vessel") != journal_block.get("vehicle"):
        errors.append("header.green_water.vessel differs from header.journal.vehicle")
    return errors


def dynamic_hazard_errors(document):
    """F7: every per-record D7 envelope re-derives its digest and has a readable version."""
    errors = []
    records = [(i, r) for i, r in enumerate(document["records"]) if "dynamic_hazards" in r]
    if not records:
        return errors
    try:
        certificate, _, d7 = conditions_modules()
    except ValueError as exc:
        return [str(exc)]
    for index, record in records:
        try:
            certificate.verify_dynamic_hazard_stamp(record["dynamic_hazards"])
            d7.read_stamp(record["dynamic_hazards"])
            # round 4b (D16) / round T (D20): an explicit not_admitted stamp is a valid record
            exposure, reasons = d7_exposure.exposure_of(record["dynamic_hazards"])
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"records/{index}: dynamic_hazards envelope rejected: {exc}")
            continue
        if exposure == d7_exposure.NOT_ADMITTED and d7_exposure.BEYOND_OFFERED_GRID in reasons:
            # round T D24 (d): a counted record beyond the offered grid is a hard error
            errors.append(f"records/{index}: counted journal record of a key beyond the offered D7 grid "
                          f"(reason {d7_exposure.BEYOND_OFFERED_GRID!r}; CONTEXT_R4 D24)")
    return errors


_D7_RULES = {}


def d7_rule_block(version):
    """The runtime_rule block this code tree stamps for a D7 runtime rule version (conditions/dynamic, by path)."""
    if not _D7_RULES:
        import importlib
        conditions_modules()
        _D7_RULES["release"] = importlib.import_module("_f7_validator_conditions.dynamic.release")
    return _D7_RULES["release"].rule_block_for(version)


def dynamic_exposure_errors(document):
    """G23b/G23c: the D7 runtime rule of a certificate.

    * every admitted runtime-rule stamp is self-consistent (d7_exposure.runtime_exposure, including the G23c
      realized-exposure record and the detected_samples / first_detection agreement);
    * one certificate holds one runtime rule (records and the env header's runtime_rule block);
    * G23c F10: a diagnostic rule (d7_exposure.DIAGNOSTIC_RULES) is refused outside a diagnostic run (seed purpose
      diagnostic); the runtime_rule block of every record and of the header equals the block this code tree
      stamps for that version (constants included);
    * the header's exposure_summary equals the one recomputed from the records, and is required when records
      carry a non-diagnostic rule. Records without a runtime_rule block (v2.5, D0, moving basin) are read as rule
      None."""
    header = document["header"]
    block = header.get("dynamic_hazards_header")
    if not isinstance(block, dict):
        return []
    errors = []
    diagnostic = reference_environment.is_diagnostic(header.get("seed_namespace"))
    rules = set()
    envelopes = []
    for index, record in enumerate(document["records"]):
        envelope = record.get("dynamic_hazards")
        if envelope is None:
            continue
        envelopes.append(envelope)
        try:
            reading = d7_exposure.runtime_exposure(envelope)
        except (d7_exposure.ExposureError, KeyError, TypeError) as exc:
            errors.append(f"records/{index}: dynamic_hazards runtime exposure rejected: {exc}")
            continue
        if reading is None:
            continue
        rules.add(reading["rule"])
        if reading["rule"] is None:
            continue
        if reading["rule"] in d7_exposure.DIAGNOSTIC_RULES and not diagnostic:
            errors.append(f"records/{index}: D7 runtime rule {reading['rule']!r} is diagnostic only (seed purpose "
                          f"diagnostic required, G23c F10)")
        try:
            expected_block = d7_rule_block(reading["rule"])
        except (ValueError, ImportError) as exc:
            errors.append(f"records/{index}: D7 runtime rule {reading['rule']!r}: {exc}")
            continue
        if envelope["payload"]["runtime"].get("runtime_rule") != expected_block:
            errors.append(f"records/{index}: runtime_rule block differs from the one this code tree stamps for "
                          f"{reading['rule']!r} (constants, causes or text; G23c F10)")
    declared = block.get("runtime_rule")
    if declared is not None:
        version = declared.get("version") if isinstance(declared, dict) else None
        try:
            expected_block = dict(d7_rule_block(version), motion_rule=version)
        except (ValueError, ImportError, TypeError) as exc:
            expected_block = None
            errors.append("dynamic_hazards_header.runtime_rule: %s" % exc)
        if expected_block is not None and declared != expected_block:
            errors.append("dynamic_hazards_header.runtime_rule differs from the block this code tree stamps (G23c F10)")
        if version in d7_exposure.DIAGNOSTIC_RULES and not diagnostic:
            errors.append("dynamic_hazards_header.runtime_rule %r is diagnostic only (G23c F10)" % (version,))
        stray = {r for r in rules if r is not None and r != version}
        if stray:
            errors.append("D7 records of runtime rule(s) %s differ from the header's %r" % (sorted(stray), version))
        # G23d L4 (REV-G23c H5, T15/T16): a header declaring a non-diagnostic rule needs that rule's block in every
        # admitted D1..D4 record; a record stripped of its rule fields (v2.5-shaped) is rejected
        if version is not None and version not in d7_exposure.DIAGNOSTIC_RULES:
            for index, record in enumerate(document["records"]):
                envelope = record.get("dynamic_hazards")
                if envelope is None:
                    continue
                try:
                    kind, _ = d7_exposure.exposure_of(envelope)
                except (d7_exposure.ExposureError, KeyError, TypeError, AttributeError):
                    continue                        # malformed stamps are reported by dynamic_hazard_errors
                if kind == d7_exposure.NOT_ADMITTED:
                    continue
                payload = envelope.get("payload") if isinstance(envelope, dict) else None
                assignment = payload.get("assignment") if isinstance(payload, dict) else None
                runtime = payload.get("runtime") if isinstance(payload, dict) else None
                dose = assignment.get("dose") if isinstance(assignment, dict) else None
                task = assignment.get("task") if isinstance(assignment, dict) else None
                if type(dose) is not int or not 1 <= dose <= 4 or task == "moving_basin":
                    continue
                block_r = runtime.get("runtime_rule") if isinstance(runtime, dict) else None
                if not isinstance(block_r, dict) or block_r.get("version") != version:
                    errors.append(f"records/{index}: an admitted D{dose} record lacks the header's runtime_rule "
                                  f"{version!r} block (G23d H5)")
    if len(rules) > 1:
        errors.append("one certificate holds one D7 runtime rule; the records carry %s" % sorted(str(r) for r in rules))
    summary = block.get("exposure_summary")
    if summary is None:
        if any(r is not None and r not in d7_exposure.DIAGNOSTIC_RULES for r in rules):
            errors.append("dynamic_hazards_header.exposure_summary is missing (records of a D7 runtime rule)")
        return errors
    suite = block.get("suite") if isinstance(block.get("suite"), dict) else {}
    try:
        expected = d7_exposure.exposure_summary(envelopes, suite.get("dose"))
    except (d7_exposure.ExposureError, KeyError, TypeError) as exc:
        return errors + ["dynamic_hazards_header.exposure_summary cannot be recomputed: %s" % exc]
    if summary != expected:
        errors.append("dynamic_hazards_header.exposure_summary differs from the summary recomputed from the records")
    return errors


def dynamic_header_errors(document):
    """Round 4b (D16/D19): D7 header per-dose counts include not_admitted, the family's
    pinned pool manifest digest is stamped, and not-admitted records agree with the counts.
    Round T (D23): a header with pool_pinned False is an error."""
    header = document["header"]
    block = header.get("dynamic_hazards_header")
    if not isinstance(block, dict):
        return []
    errors = []
    if block.get("pool_pinned", True) is False:
        # round T D23: journal certificates come only from pinned pools (missing flag = pinned)
        errors.append("dynamic_hazards_header.pool_pinned is False: the D7 pool is not in the family's "
                      "pinned manifest and cannot be reproduced from the release (CONTEXT_R4 D23)")
    manifest = block.get("pool_manifest_sha256")
    if not (isinstance(manifest, str) and len(manifest) == 64 and all(c in "0123456789abcdef" for c in manifest)):
        errors.append("dynamic_hazards_header.pool_manifest_sha256 must be the family's pinned pool manifest "
                      "SHA-256 (D19)")
    try:
        counts = d7_exposure.per_dose_counts(block)
    except d7_exposure.ExposureError as exc:
        return errors + [str(exc)]
    suite = block.get("suite") if isinstance(block.get("suite"), dict) else {}
    dose = suite.get("dose")
    if type(dose) is not int or not 0 <= dose < len(counts):
        return errors + ["dynamic_hazards_header.suite.dose must be an integer D0..D4"]
    try:                                                # G23e: tiers not applicable by design
        excluded = d7_exposure.not_applicable_doses(block)
    except d7_exposure.ExposureError as exc:
        return errors + [str(exc)]
    if dose in excluded:
        errors.append("D%d is not applicable by design for this pool (dynamic_hazards_header.not_applicable_doses "
                      "%s): no certificate exists for it (G23e)" % (dose, list(excluded)))
    keys = set()
    for index, record in enumerate(document["records"]):
        envelope = record.get("dynamic_hazards")
        if envelope is None:
            continue
        try:
            kind, _ = d7_exposure.exposure_of(envelope)
        except d7_exposure.ExposureError:
            continue  # reported by dynamic_hazard_errors
        if kind == d7_exposure.NOT_ADMITTED:
            runtime = record["conditions"].get("runtime") or {}
            keys.add((runtime.get("env_index", record["env"]), runtime.get("episode_index", record["ep"])))
    if len(keys) > counts[dose]["not_admitted"]:
        errors.append("%d not-admitted keys in the records but dynamic_hazards_header counts %d not_admitted "
                      "at D%d" % (len(keys), counts[dose]["not_admitted"], dose))
    return errors


def journal_pairing_errors(document):
    """F7 / D11: journal records carry the complete, recomputable primitive map.

    The header ``pairing`` block (conditions/pairing.pairing_declaration) names
    the primitive inventory; every record's ``scenario_hashes`` must have
    exactly that inventory, its condition primitives must be recomputable from
    the record's schema-3 episode stamp and its ``d7_assignment`` from the
    record's D7 envelope (conditions/pairing.py semantics).
    """
    header = document["header"]
    declaration = header["pairing"]
    errors = []
    primitives = declaration.get("primitives")
    changed = declaration.get("changed_primitives", [])
    unchanged = declaration.get("unchanged_primitives", [])
    if (not isinstance(primitives, list) or not primitives or primitives != sorted(set(primitives))
            or any(not isinstance(p, str) or not p for p in primitives)):
        return ["pairing.primitives must be a sorted list of unique primitive names"]
    if not isinstance(changed, list) or not isinstance(unchanged, list) or \
            sorted(set(changed) | set(unchanged)) != primitives or set(changed) & set(unchanged):
        errors.append("pairing changed/unchanged primitives do not partition pairing.primitives")
    if header["conditions"]["schema"] != CONDITION_STAMP_SCHEMA_V3:
        errors.append("a journal pairing declaration requires a usvbench.conditions/3 header stamp")
    for field in ("dynamic_hazards_header", "journal_mixin"):
        if field not in header:
            errors.append("journal certificate header lacks " + field)
    try:
        _, pairing, _ = conditions_modules()
    except ValueError as exc:
        return errors + [str(exc)]
    for index, record in enumerate(document["records"]):
        prefix = f"records/{index}: "
        hashes = record.get("scenario_hashes")
        if not isinstance(hashes, dict) or "dynamic_hazards" not in record:
            errors.append(prefix + "journal records need scenario_hashes and a dynamic_hazards envelope")
            continue
        if sorted(hashes) != primitives:
            errors.append(prefix + "scenario_hashes inventory differs from header pairing.primitives")
            continue
        if not all(pairing.is_hash(value) for value in hashes.values()):
            errors.append(prefix + "scenario_hashes values must be SHA-256 hex digests")
            continue
        try:
            recomputed = pairing.condition_primitive_hashes(record["conditions"])
            # round T (D20 d): every record, admitted or not, hashes by F0 pairing.d7_primitive_hash
            recomputed[pairing.D7_PRIMITIVE] = d7_exposure.d7_primitive(pairing, record["dynamic_hazards"])
            # G3: the appearing primitive, recomputed from the record's appearing stamp when declared
            if getattr(pairing, "APPEARING_PRIMITIVE", None) in primitives:
                recomputed[pairing.APPEARING_PRIMITIVE] = pairing.appearing_primitive_hash(record["appearing_hazards"])
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(prefix + "cannot recompute the pairing primitives: %s" % exc)
            continue
        for name, value in sorted(recomputed.items()):
            if hashes.get(name) != value:
                errors.append(prefix + "scenario hash %r does not match its stamp" % name)
    return errors


def condition_errors(document):
    """Schema 3.1: recompute every condition digest and cross-check records.

    The header digest must match its parameters; every record carries the
    finished episode's stamp whose parameters (hence digest) must equal the
    header's; a record's runtime env_index, when present, must equal its env;
    a legacy top-level ``conditions`` alias must equal the header stamp; the
    stamp's physics version must equal the header's declared version.

    F7: schema-2 (legacy) and schema-3 (hydro_v2, E3) stamps are both
    accepted with the same digest rule; records must use the header's
    schema; per-record D7 envelopes are verified (dynamic_hazard_errors);
    journal certificates (header pairing schema usvbench.pairing/1) must carry
    recomputable primitive maps (journal_pairing_errors).
    """
    errors = []
    header = document["header"]
    stamp = header["conditions"]
    if stamp.get("schema") not in CONDITION_STAMP_SCHEMAS:
        return ["unsupported condition stamp schema %r" % (stamp.get("schema"),)]
    digest = condition_digest(stamp["parameters"])
    if digest != stamp["parameters_sha256"]:
        errors.append("header condition digest does not match its parameters")
    declared = header.get("physics_version")
    stamped = stamp["context"].get("physics_version")
    if declared is not None and stamped is not None and str(stamped) != str(declared):
        errors.append("condition stamp physics_version differs from header physics_version")
    for index, record in enumerate(document["records"]):
        prefix = f"records/{index}: "
        block = record["conditions"]
        if block.get("schema") != stamp["schema"]:
            errors.append(prefix + "condition stamp schema differs from the header stamp")
            continue
        record_digest = condition_digest(block["parameters"])
        if record_digest != block["parameters_sha256"]:
            errors.append(prefix + "condition digest does not match its parameters")
        elif record_digest != stamp["parameters_sha256"]:
            errors.append(prefix + "condition parameters differ from the header stamp")
        env_index = block["runtime"].get("env_index")
        if env_index is not None and env_index != record["env"]:
            errors.append(prefix + "condition runtime env_index differs from record env")
    if "conditions" in document and document["conditions"] != stamp:
        errors.append("legacy alias disagrees with header: conditions")
    errors.extend(dynamic_hazard_errors(document))
    declaration = header.get("pairing")
    if isinstance(declaration, dict) and declaration.get("schema") == PAIRING_SCHEMA:
        errors.extend(journal_pairing_errors(document))
        errors.extend(dynamic_header_errors(document))
    errors.extend(dynamic_exposure_errors(document))         # G23b: runtime rule S and exposure metric A
    # G3 appearing suite: per-record stamps and the header block (present only with the suite)
    errors.extend(appearing_exposure.record_errors(document))
    if "appearing_hazards_header" in header:
        errors.extend(appearing_exposure.header_errors(header["appearing_hazards_header"], document["records"]))
        registration = header.get("seed_namespace")
        purpose = registration.get("purpose") if isinstance(registration, dict) else None
        # G20b: the diagnostic gap override is never official; G20c (REV-G20b m4/m5): a missing registration counts
        # as official, and the supplementary dose D5 is diagnostic only too
        for check in (appearing_exposure.gap_override_official_check,
                      appearing_exposure.supplementary_dose_official_check):
            try:
                check(header["appearing_hazards_header"], purpose)
            except ValueError as exc:
                errors.append(str(exc))
    # G10 green-water exclusion rule (journal certificates with waves)
    errors.extend(green_water_errors(document))
    return errors


def load_json(path, *, legacy=False):
    def no_constant(value):
        raise ValueError("non-JSON numeric constant: " + value)
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key: " + key)
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=(lambda _: None) if legacy else no_constant,
                         object_pairs_hook=no_duplicates)


def _path(config, path):
    for part in path.split("."):
        config = config[part]
    return config


def command_options(argv):
    """Read flag occurrences from tokenized argv; duplicate flags keep order."""
    result = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            key, separator, value = token.partition("=")
            next_is_hydra = (index + 1 < len(argv) and argv[index + 1].startswith("env.")
                            and "=" in argv[index + 1])
            if not separator and index + 1 < len(argv) and not argv[index + 1].startswith("--") and not next_is_hydra:
                index += 1
                value = argv[index]
            result.setdefault(key, []).append(value)
        elif token.startswith("env.") and "=" in token:
            result.setdefault("--set", []).append(token[4:])
        index += 1
    return result


def journal_suite_arguments(argv):
    """(every --suite token in order, last --suite-file value or None) of an argv."""
    tokens, suite_file = [], None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--suite":
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                tokens.append(argv[index])
                index += 1
            continue
        if token.startswith("--suite="):
            tokens.append(token.partition("=")[2])
        elif token == "--suite-file" and index + 1 < len(argv):
            suite_file = argv[index + 1]
            index += 1
        elif token.startswith("--suite-file="):
            suite_file = token.partition("=")[2]
        index += 1
    return tokens, suite_file


def override_errors(header, argv):
    errors = []
    options = command_options(argv)
    overrides = header["overrides"]
    config = header["resolved_config"]
    # All stamps must equal the final configuration, even stamps imposed by
    # evaluator defaults rather than written explicitly on the command line.
    for path, value in overrides.items():
        try:
            if _path(config, path) != value:
                errors.append("override differs from resolved_config: " + path)
        except (KeyError, TypeError):
            errors.append("override missing from resolved_config: " + path)
    expected = {}
    for flag, path in (("--gate-width-override", "gate_width_override_m"),
                       ("--fortress-aperture-override", "fortress_aperture_override_m"),
                       ("--on-line-blockers-override", "on_line_blockers_override"),
                       ("--current-speed", "current_speed_override_mps")):
        if flag in options:
            expected[path] = options[flag][-1]
    if "--spawn-distance" in options:
        path = "spawn_distance_override_m" if "spawn_distance_override_m" in config else "curriculum_start_distance_m"
        expected[path] = options["--spawn-distance"][-1]
    if "--axis" in options:
        if "--value" not in options:
            errors.append("command --axis requires --value")
        else:
            expected[options["--axis"][-1]] = options["--value"][-1]
    elif "--imbalance" in options:
        expected["thrust_imbalance"] = options["--imbalance"][-1]
    runtime_paths = header.get("runtime_paths") or {}
    journal = "journal" in header
    supplied = set()
    for assignment in options.get("--set", []):
        try:
            path, raw = split_assignment(assignment)
            if journal and path in JOURNAL_RUNTIME_FIELDS:
                # D19: runtime pool-path fields are stamped in header.runtime_paths, never
                # as science overrides
                supplied.add(path)
                if runtime_paths.get(path) != raw:
                    errors.append("command runtime path differs from header.runtime_paths: " + path)
                continue
            expected[path] = raw
        except ValueError as exc:
            errors.append(str(exc))
    if journal:
        for path in sorted(set(runtime_paths) - supplied):
            errors.append("header.runtime_paths entry without a command --set: " + path)
    for path in overrides:
        if journal and path in JOURNAL_RUNTIME_FIELDS:
            errors.append("runtime path stamped as a science override: " + path)
    for path, raw in expected.items():
        try:
            resolved = _path(config, path)
            desired = coerce_like(raw, resolved, path)
            if path not in overrides:
                errors.append("command override has no stamp: " + path)
            elif desired != overrides[path]:
                errors.append("command override differs from stamp: " + path)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"cannot verify command override {path}: {exc}")
    for flags, field, caster in (
        (("--task", "--eval-task"), "task_id", str),
        (("--eval-seed", "--seed"), "eval_seed", int),
        (("--level",), "level", int),
        (("--num_envs", "--num-envs"), "num_envs", int),
        (("--episodes",), "episodes_requested", int),
    ):
        for flag in flags:
            if flag in options:
                try:
                    value = options[flag][-1]
                    cfg_field = {"eval_seed": "seed", "level": "eval_level", "num_envs": "scene.num_envs"}.get(field)
                    if cfg_field in expected:
                        value = expected[cfg_field]
                    if caster(value) != header[field]:
                        errors.append("command/header mismatch: " + flag)
                except ValueError:
                    errors.append("invalid command value: " + flag)
    if "--checkpoint" in options:
        # Absolute paths are platform-specific; compare normalized separators
        # and permit the same relative suffix in cross-host audit workflows.
        claimed = (header["checkpoint"]["path"] or "").replace("\\", "/")
        supplied = options["--checkpoint"][-1].replace("\\", "/")
        if not (claimed == supplied or (not supplied.startswith("/") and ":" not in supplied
                                       and claimed.endswith("/" + supplied))):
            errors.append("command/header mismatch: --checkpoint")
    # F7: journal --suite takes several tokens per occurrence (nargs "+");
    # the stamp must list every token of every occurrence, in order.
    suite_tokens, suite_file = journal_suite_arguments(argv)
    presets = header["suite_presets"]
    if suite_tokens or "suite" in presets:
        if list(presets.get("suite") or []) != suite_tokens:
            errors.append("command --suite tokens differ from header.suite_presets.suite")
    if suite_file is not None or presets.get("suite_file") is not None:
        if presets.get("suite_file") != suite_file:
            errors.append("command --suite-file differs from header.suite_presets.suite_file")
    # Supplied suite/preset switches must be represented, never silently lost.
    for flag, values in options.items():
        if flag in ("--suite", "--suite-file"):
            continue
        if "suite" in flag.lower() or "preset" in flag.lower():
            name = flag.lstrip("-").replace("-", "_")
            if name not in header["suite_presets"] or str(header["suite_presets"][name]) != values[-1]:
                errors.append("command suite/preset has no matching stamp: " + flag)
    return errors


def validate_certificate(document, *, schema=None, num_envs=64, episodes_per_env=2,
                         command=None, allow_incomplete=False, require_verified=False,
                         code_root=None, release_manifest=None):
    if schema is None:
        schema = load_json(Path(__file__).with_name("cert_v3.schema.json"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = ["schema " + "/".join(map(str, e.absolute_path)) + ": " + e.message
              for e in sorted(validator.iter_errors(document), key=lambda e: str(list(e.absolute_path)))]
    if errors:
        return errors
    def finite_errors(value, path=""):
        if isinstance(value, float) and not math.isfinite(value):
            return ["non-finite number: " + path]
        if isinstance(value, dict):
            return [error for key, item in value.items() for error in finite_errors(item, path + "/" + key)]
        if isinstance(value, list):
            return [error for index, item in enumerate(value) for error in finite_errors(item, path + "/" + str(index))]
        return []
    errors.extend(finite_errors(document))
    if errors:
        return errors
    minor = document.get("schema_minor_version", 0)
    if minor not in SUPPORTED_MINOR_VERSIONS:
        return [f"unsupported schema_minor_version {minor!r}"]
    if minor >= 1:
        errors.extend(condition_errors(document))
    header, records = document["header"], document["records"]
    # E1: D2 items 8, 9, 10, 20, 21 (all minors unless noted in the helper).
    errors.extend(code_errors(document, minor, code_root=code_root, release_manifest=release_manifest,
                              require_verified=require_verified))
    errors.extend(physics_vessel_errors(header, minor))
    errors.extend(alias_errors(document, minor))
    errors.extend(reference_environment_errors(header))
    errors.extend(wind_coefficient_errors(document))
    errors.extend(rollpitch_damping_errors(document))          # T25 (T1.35b)
    errors.extend(mppi_implementation_errors(header))
    errors.extend(avoidance_layer_errors(header))
    errors.extend(avoidance_pin_errors(header, code_root=code_root))     # G25b
    errors.extend(planner_memory_errors(header, code_root=code_root))    # G25 / G25b
    errors.extend(planner_configuration_errors(header, code_root=code_root))     # G24b (REV-G25b M3)
    errors.extend(avoidance_configuration_errors(header, code_root=code_root))   # G24b (REV-G25b M3)
    device = header["runtime"].get("gpu_device")
    if isinstance(device, str) and header["timing"]["cuda_synchronized"] != device.startswith("cuda"):
        errors.append("timing.cuda_synchronized contradicts runtime.gpu_device")
    latency = header.get("controller_latency")
    if isinstance(latency, dict) and isinstance(latency.get("statistics_s"), dict):
        errors.extend(statistics_errors(latency["statistics_s"], "controller_latency: "))
    if minor >= 2:
        errors.extend(level_distance_errors(header))
        fixed = header["outcome_semantics"]["fixed_horizon"]
        for index, record in enumerate(records):
            if record["outcome"] != episode_outcome(record, fixed):
                errors.append(f"records/{index}: outcome {record['outcome']!r} differs from the mapping "
                              f"table value {episode_outcome(record, fixed)!r}")
            if fixed is True and record["termination_reason"] == "success":
                errors.append(f"records/{index}: a fixed-horizon family cannot stop on success")
    if require_verified:
        errors.extend(cell_config_errors(header))
    n, quota = header["num_envs"], header["episodes_per_env"]
    if (n, quota) != (num_envs, episodes_per_env):
        errors.append(f"expected {num_envs} x {episodes_per_env} grid, header has {n} x {quota}")
    if quota != -(-header["episodes_requested"] // n):
        errors.append("episodes_per_env must be ceil(episodes_requested / num_envs)")
    pairs = [(r["env"], r["ep"]) for r in records]
    actual = set(pairs)
    outside = {(env, ep) for env, ep in actual if env >= n or ep >= quota}
    if len(actual) != len(pairs):
        errors.append("duplicate (env, ep)")
    if outside:
        errors.append("records outside declared grid: " + str(sorted(outside)[:8]))
    is_complete = not outside and len(actual) == len(pairs) == n * quota
    if header["complete"] != is_complete:
        errors.append("complete stamp disagrees with record grid")
    if not is_complete and not allow_incomplete:
        errors.append(f"incomplete grid: {n * quota - len(actual - outside)} missing episodes")
    for alias, canonical in (("task", "task_id"), ("seed", "eval_seed"), ("level", "level"),
                             ("complete", "complete"), ("num_envs", "num_envs"),
                             ("episodes_requested", "episodes_requested"), ("episodes_per_env", "episodes_per_env"),
                             ("overrides", "overrides")):
        if alias in document and document[alias] != header[canonical]:
            errors.append("legacy alias disagrees with header: " + alias)
    checkpoint = header["checkpoint"]
    if (checkpoint["path"] is None) != (checkpoint["sha256"] is None):
        errors.append("checkpoint path and SHA256 must both be null or both be present")
    for path, field in (("seed", "eval_seed"), ("eval_level", "level"), ("scene.num_envs", "num_envs")):
        try:
            if _path(header["resolved_config"], path) != header[field]:
                errors.append("resolved_config/header mismatch: " + path)
        except KeyError:
            pass
    if datetime.fromisoformat(header["ended_at"].replace("Z", "+00:00")) < datetime.fromisoformat(header["started_at"].replace("Z", "+00:00")):
        errors.append("end time precedes start time")
    for index, record in enumerate(records):
        prefix = f"records/{index}: "
        reason = record["termination_reason"]
        if not (record["terminated"] or record["truncated"]):
            errors.append(prefix + "completed episode has neither stop mask")
        if reason == "timeout" and (not record["truncated"] or record["terminated"]):
            errors.append(prefix + "timeout must be a truncation without termination")
        if reason in TERMINAL_REASONS and not record["terminated"]:
            errors.append(prefix + "terminal reason requires terminated=true")
        if reason == "success" and not record["success"]:
            errors.append(prefix + "success termination requires success=true")
        if not record["success"] and record["tts_s"] is not None:
            errors.append(prefix + "failed episode must have null time to success")
        times = [record["controller_compute_time_s"], record["environment_step_time_s"]]
        if times[0]["n"] != times[1]["n"]:
            errors.append(prefix + "controller/environment timing counts differ")
        for stats in times:
            errors.extend(statistics_errors(stats, prefix))
        if record["contact_seconds"] is not None and record["contact_longest_seconds"] is not None:
            if record["contact_longest_seconds"] > record["contact_seconds"] + 1e-9:
                errors.append(prefix + "longest contact exceeds total contact duration")
        errors.extend(contact_errors(record, header, prefix))
    errors.extend(override_errors(header, header["command"]["argv"]))
    if command is not None:
        errors.extend(override_errors(header, command))
    if require_verified:
        if header["unverified"]:
            errors.append("UNVERIFIED provenance: " + "; ".join(header["unverified"]))
        if any(r["termination_reason"] == "unknown" for r in records):
            errors.append("UNVERIFIED episode termination reason")
        missing = [key for key in ("vessel", "physics_version") if header[key] is None]
        missing += ["runtime." + key for key, value in header["runtime"].items() if value is None]
        if header["code"]["environment_source"] is None:
            missing.append("code.environment_source")
        if missing:
            errors.append("UNVERIFIED missing provenance: " + ", ".join(missing))
    return list(dict.fromkeys(errors))


def legacy_report(document, schema=None):
    """Return a diagnostic conversion; never fabricate a v3 header or write input."""
    if schema is None:
        schema = load_json(Path(__file__).with_name("cert_v3.schema.json"))
    if not isinstance(document, dict):
        raise ValueError("legacy certificate must be a JSON object")
    if not isinstance(document.get("records", []), list) or any(not isinstance(r, dict) for r in document.get("records", [])):
        raise ValueError("legacy records must be an array of objects")
    available_header = dict(document.get("header") or {})
    for old, new in (("task", "task_id"), ("seed", "eval_seed"), ("level", "level"),
                     ("num_envs", "num_envs"), ("episodes_requested", "episodes_requested"),
                     ("episodes_per_env", "episodes_per_env"), ("complete", "complete"),
                     ("overrides", "overrides")):
        if old in document:
            available_header.setdefault(new, document[old])
    checkpoint = document.get("checkpoint")
    if isinstance(checkpoint, str):
        available_header.setdefault("checkpoint", {"path": checkpoint})
    controller = document.get("controller")
    if isinstance(controller, str):
        available_header.setdefault("controller", {"name": controller})
    def missing_required(value, definition, prefix=""):
        paths = []
        for key in definition.get("required", []):
            path = prefix + key
            if not isinstance(value, dict) or key not in value:
                paths.append(path)
            else:
                paths.extend(missing_required(value[key], definition.get("properties", {}).get(key, {}), path + "."))
        return paths
    required_record = schema["$defs"]["episode"]["required"]
    records = document.get("records", [])
    missing_records = {}
    for key in required_record:
        missing = [i for i, record in enumerate(records) if key not in record]
        if missing:
            missing_records[key] = {"count": len(missing), "record_indices": missing}
    return {"report_type": "legacy_read_only_conversion", "certificate_created": False,
            "available_header": available_header,
            "missing_header_fields": missing_required(available_header, schema["properties"]["header"]),
            "missing_record_fields": missing_records, "record_count": len(records),
            "unknown_provenance": "Cannot recover missing code, timing, commands or termination causes from legacy records."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("cert_v3.schema.json"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes-per-env", type=int, default=2)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--legacy-report", action="store_true")
    parser.add_argument("--code-root", type=Path, default=None,
                        help="recompute the usvbench-code-tree-v1 digest over <root>/{tasks,scripts}")
    parser.add_argument("--release-manifest", type=Path, default=None,
                        help="check the digest against a release manifest file")
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument("--command", help="Shell command string (use --command-json for Windows paths)")
    command_group.add_argument("--command-json", type=Path, help="JSON array of exact original argv tokens")
    command_group.add_argument("--command-argv", nargs=argparse.REMAINDER, help="Exact command tokens; must be last option")
    args = parser.parse_args(argv)
    try:
        schema = load_json(args.schema)
        Draft202012Validator.check_schema(schema)
        document = load_json(args.certificate, legacy=args.legacy_report)
        if args.legacy_report:
            print(json.dumps(legacy_report(document, schema), indent=2, allow_nan=False))
            return 0
        command = args.command_argv
        if args.command_json:
            command = load_json(args.command_json)
        elif args.command:
            command = shlex.split(args.command)
        if command is not None and (not isinstance(command, list) or not all(isinstance(x, str) for x in command)):
            raise ValueError("command must be an array of argv strings")
        errors = validate_certificate(document, schema=schema, num_envs=args.num_envs,
                                      episodes_per_env=args.episodes_per_env, command=command,
                                      allow_incomplete=args.allow_incomplete, require_verified=args.require_verified,
                                      code_root=args.code_root, release_manifest=args.release_manifest)
        header = document.get("header", {}) if isinstance(document, dict) else {}
        header = header if isinstance(header, dict) else {}
        records = document.get("records", []) if isinstance(document, dict) else []
        records = records if isinstance(records, list) else []
        unverified = header.get("unverified", [])
        unverified = unverified if isinstance(unverified, list) else []
        minor = document.get("schema_minor_version", 0) if isinstance(document, dict) else 0
        print(f"{'FAIL' if errors else 'PASS'} {args.certificate.name} schema=v3.{minor} "
              f"grid={header.get('num_envs', '?')}x{header.get('episodes_per_env', '?')} "
              f"records={len(records)} complete={header.get('complete', '?')} "
              f"unverified={len(unverified)} errors={len(errors)}")
        for error in errors:
            print(error.replace("\n", " "), file=sys.stderr)
        return 1 if errors else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"FAIL {args.certificate.name} {str(exc).replace(chr(10), ' ')}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
