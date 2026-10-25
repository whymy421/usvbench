"""G12 full-dose smoke: the cell plan (pure Python, ASCII, no third-party imports).

Shared by the planner (tools/plan_dose_smoke.py: timing model, named timeouts, sharding) and the judge
(judge/dose_judge.py: independent reconstruction of every manifest row). The PowerShell queue builds the
same plan with its own code (queue/dose_queue.ps1 BuildPlan); the judge compares the two row by row.

Sources (brief 20261002_g12_dose_smoke/prompt.md "The cells"; NC 34 section 9):
  * groups: 12 journal tasks x {BlueBoat, WAM-V} = 24 journal ids (tasks/_shared/journal_cfg_v2.py TASKS,
    JOURNAL_ENV_IDS; the judge reads them from the release and checks the table below against it);
  * 49 single-suite conditions: calm; waves, wind, current D1..D4; actuator.{cap,gain,lag,resistance} D1..D4;
    perception.{noise,bias,latency,dropout} D1..D4; dynamic D1..D4 (token grammar of
    tasks/_shared/conditions/journal_config.py: one suite or sub-axis at a time). The appearing suite is NOT
    in this queue;
  * combinations: CODE-PATH CHECK ONLY (controller message, owner decision 2026-10-02 12:xx: the real-sea combinations
    are a side-table experiment on tasks chosen later, not the main table): journal.combo.wave_wind.D3,
    journal.combo.wave_current.D3 and journal.combo.wave_wind_local.D3 by name (presets_v3.COMBINATIONS), each on
    station_keeping BlueBoat and station_keeping WAM-V, classical controller only (6 cells; the WAM-V
    wave_wind_local.D3 cell is expected to be excluded by the green-water formula, part of the check);
  * supplement journal.waves.supp075 (role SUPPLEMENT) on BlueBoat station_keeping, docking, forced_crossing,
    path_following;
  * classical on every cell above; MPPI on calm and the all-suites stack of every group, plus every condition
    above (49 conditions + the supplement where it exists; the combination check is classical only) for
    station_keeping, forced_crossing and docking on both vessels (deduplicated: calm once).

G32 v2.6 COPY (work_v26/dose_plan.py; the original work/dose_plan.py is untouched). Additions only (no existing value
or function changes, so every earlier plan, batch and judgement is reproduced byte for byte):
  * NAMES_V24 / NAMES_V25 / NAMES_V26 and RELEASE_GENERATIONS: the classical certificate names of each release
    generation (v2.6 = G24b CONTROLLER_NAMES.md: planner_los_pid_mem_v2, los_pid_avoid_v2, los_pid_cont_avoid_v2,
    docking_runup_pid_avoid_v2, staged_harbour);
  * names_generation(names) / release_generation(release_root): the generation whose table equals the names that
    the release's own scripts/run_cell.py declares (read statically by release_controller_names);
  * release_default_names(release_root): the release's names when its generation is v2.6 or later, else None
    (= the v2.4 table, the unchanged behaviour of every v2.4 / v2.5 judgement).
"""

TASKS = ("reach", "path_following", "station_keeping", "docking", "path_hazard", "scatter", "ring", "fortress",
         "forced_crossing", "iceberg", "moving_basin", "harbour_mission")
TASK_CODE = {"reach": "RE", "path_following": "PF", "station_keeping": "SK", "docking": "DK", "path_hazard": "PH",
             "scatter": "SC", "ring": "RG", "fortress": "FO", "forced_crossing": "FC", "iceberg": "IB",
             "moving_basin": "MB", "harbour_mission": "HM"}
# (vessel key, --vessel flag, journal vehicle) in plan order.
VESSELS = (("bb", "blueboat", "blueboat_v2"), ("wamv", "wamv", "wamv_vrx_v2_opt1"))
JOURNAL_ID = {
    ("reach", "bb"): "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1",
    ("path_following", "bb"): "Isaac-USV-PathFollow-BlueBoat-Journal-Direct-v1",
    ("station_keeping", "bb"): "Isaac-USV-StationKeep-BlueBoat-Journal-Direct-v1",
    ("docking", "bb"): "Isaac-USV-Dock-BlueBoat-Journal-Direct-v1",
    ("path_hazard", "bb"): "Isaac-USV-PathHazard-BlueBoat-Journal-Direct-v1",
    ("scatter", "bb"): "Isaac-USV-HazardScatter-BlueBoat-Journal-Direct-v1",
    ("ring", "bb"): "Isaac-USV-HazardRing-BlueBoat-Journal-Direct-v1",
    ("fortress", "bb"): "Isaac-USV-HazardFortress-BlueBoat-Journal-Direct-v1",
    ("forced_crossing", "bb"): "Isaac-USV-HazardCross-BlueBoat-Journal-Direct-v1",
    ("iceberg", "bb"): "Isaac-USV-Iceberg-BlueBoat-Journal-Direct-v1",
    ("moving_basin", "bb"): "Isaac-USV-HazardMovingBasin-BlueBoat-Journal-Direct-v1",
    ("harbour_mission", "bb"): "Isaac-USV-HarborMission-BlueBoat-Journal-Direct-v1",
    ("reach", "wamv"): "Isaac-USV-Reach-WAMVVRX-Journal-Direct-v1",
    ("path_following", "wamv"): "Isaac-USV-PathFollow-WAMVVRX-Journal-Direct-v1",
    ("station_keeping", "wamv"): "Isaac-USV-StationKeep-WAMVVRX-Journal-Direct-v1",
    ("docking", "wamv"): "Isaac-USV-Dock-WAMVVRX-Journal-Direct-v1",
    ("path_hazard", "wamv"): "Isaac-USV-PathHazard-WAMVVRX-Journal-Direct-v1",
    ("scatter", "wamv"): "Isaac-USV-HazardScatter-WAMVVRX-Journal-Direct-v1",
    ("ring", "wamv"): "Isaac-USV-HazardRing-WAMVVRX-Journal-Direct-v1",
    ("fortress", "wamv"): "Isaac-USV-HazardFortress-WAMVVRX-Journal-Direct-v1",
    ("forced_crossing", "wamv"): "Isaac-USV-HazardCross-WAMVVRX-Journal-Direct-v1",
    ("iceberg", "wamv"): "Isaac-USV-Iceberg-WAMVVRX-Journal-Direct-v1",
    ("moving_basin", "wamv"): "Isaac-USV-HazardMovingBasin-WAMVVRX-Journal-Direct-v1",
    ("harbour_mission", "wamv"): "Isaac-USV-HarborMission-WAMVVRX-Journal-Direct-v1",
}
CARRIER = {"reach": "hazard_nav", "path_following": "path_following", "station_keeping": "station_keeping",
           "docking": "docking", "path_hazard": "path_hazard", "scatter": "hazard_nav", "ring": "hazard_nav",
           "fortress": "hazard_nav", "forced_crossing": "hazard_nav", "iceberg": "hazard_nav",
           "moving_basin": "hazard_nav", "harbour_mission": "harbor_mission"}
# run_cell.py JOURNAL_CLASSICAL_BY_CARRIER (v2.4 scripts/run_cell.py:110-112; docking controller name
# docking_3phase_pid since G9b, run_cell.py:77-81 DOCKING_BASELINE, :106 JOURNAL_DOCKING_BASELINE): script, seed flag, extra args, certificate name.
CLASSICAL_BY_CARRIER = {
    "hazard_nav": ("classical_planner_pid.py", "--eval-seed", (), "planner_los_pid"),
    "path_hazard": ("classical_planner_pid.py", "--eval-seed", (), "planner_los_pid"),
    "path_following": ("classical_baseline.py", "--seed", ("--protocol", "mission"), "los_pid"),
    "station_keeping": ("classical_baseline.py", "--seed", ("--protocol", "mission"), "los_pid"),
    "docking": ("classical_baseline.py", "--seed", ("--protocol", "mission"), "docking_3phase_pid"),
    "harbor_mission": ("classical_harbour.py", "--eval-seed", (), "staged_harbour"),
}
MPPI_EVALUATOR = ("eval_mppi.py", "--eval-seed", (), "mppi")
NUM_ENVS, EPISODES, EPISODES_PER_ENV, DIAG_SEED, LEVEL = 32, 32, 1, 42, 0
# Single-suite axes in plan order: (axis key, condition id prefix).
AXES = (("waves", "waves"), ("wind", "wind"), ("current", "cur"),
        ("actuator.cap", "actcap"), ("actuator.gain", "actgain"), ("actuator.lag", "actlag"),
        ("actuator.resistance", "actres"),
        ("perception.noise", "pnoise"), ("perception.bias", "pbias"), ("perception.latency", "plat"),
        ("perception.dropout", "pdrop"), ("dynamic", "dyn"))
AXIS_KEYS = tuple(a for a, _p in AXES)
DOSES = (1, 2, 3, 4)
# Combination code-path check (owner decision 2026-10-02 12:xx): three D3 combinations, classical, station keeping.
COMBINATIONS = (("journal.combo.wave_wind.D3", "cwavewind_D3"), ("journal.combo.wave_current.D3", "cwavecur_D3"),
                ("journal.combo.wave_wind_local.D3", "cwwlocal_D3"))
COMBINATION_TASKS = ("station_keeping",)       # both vessels, classical controller only
# Wave component of each combination (presets_v3.py:207-213): (ladder, level).
COMBO_WAVES = {"journal.combo.wave_wind.D2": ("A", 2), "journal.combo.wave_wind.D3": ("A", 3),
               "journal.combo.wave_current.D2": ("A", 2), "journal.combo.wave_current.D3": ("A", 3),
               "journal.combo.wave_wind_local.D2": ("B", 2), "journal.combo.wave_wind_local.D3": ("B", 3)}
SUPPLEMENT = "journal.waves.supp075"
SUPPLEMENT_TASKS = ("station_keeping", "docking", "forced_crossing", "path_following")   # BlueBoat only
ALLSUITES = ("waves=D1", "wind=D1", "current=D1", "actuator=D1", "perception=D1", "dynamic=D1")
# Canonical tokens the stamp records for the all-suites stack (G2 g2_smoke_judge.py CANONICAL_SUITE).
ALLSUITES_CANONICAL = ("actuator.cap=D1", "actuator.gain=D1", "actuator.lag=D1", "actuator.resistance=D1",
                       "current=D1", "dynamic=D1", "perception.bias=D1", "perception.dropout=D1",
                       "perception.latency=D1", "perception.noise=D1", "waves=D1", "wind=D1")
MPPI_FULL_TASKS = ("station_keeping", "forced_crossing", "docking")
SIX_GRADIENT_AXES = ("wind", "current", "waves", "actuator.cap", "perception.latency", "dynamic")


def conditions(task, vkey, ctrl):
    """Ordered condition list of one block: dicts {cond, kind, axis, dose, tokens, canonical, role}."""
    out = [{"cond": "calm", "kind": "calm", "axis": None, "dose": 0, "tokens": (), "canonical": (),
            "role": "MAIN"}]
    full = ctrl == "cl" or task in MPPI_FULL_TASKS
    if full:
        for axis, prefix in AXES:
            for dose in DOSES:
                token = "%s=D%d" % (axis, dose)
                out.append({"cond": "%s_D%d" % (prefix, dose), "kind": "single", "axis": axis, "dose": dose,
                            "tokens": (token,), "canonical": (token,), "role": "MAIN"})
        for name, cond in (COMBINATIONS if ctrl == "cl" and task in COMBINATION_TASKS else ()):
            out.append({"cond": cond, "kind": "combo", "axis": None, "dose": None, "tokens": (name,),
                        "canonical": (name,), "role": "COMBINATION"})
        if vkey == "bb" and task in SUPPLEMENT_TASKS:
            out.append({"cond": "supp075", "kind": "supplement", "axis": None, "dose": None,
                        "tokens": (SUPPLEMENT,), "canonical": (SUPPLEMENT,), "role": "SUPPLEMENT"})
    if ctrl == "mp":
        out.append({"cond": "allsuites", "kind": "allsuites", "axis": None, "dose": None, "tokens": ALLSUITES,
                    "canonical": ALLSUITES_CANONICAL, "role": "ALLSUITES"})
    return out


def blocks():
    """Every (task, vessel key, controller) block in plan order: classical blocks first, then MPPI."""
    for ctrl in ("cl", "mp"):
        for vkey, _flag, _vehicle in VESSELS:
            for task in TASKS:
                yield (task, vkey, ctrl)


def block_id(task, vkey, ctrl):
    return "%s_%s_%s" % (TASK_CODE[task], vkey.upper(), ctrl.upper())


def has_waves(cond):
    """True when the cell's stack holds a wave component (green-water header and records required, G10)."""
    return cond["kind"] in ("combo", "supplement", "allsuites") or cond["axis"] == "waves"


def timeout_name(task, vkey, ctrl, kind, host):
    """Named timeout CELLMS_<task code>_<BB|WAMV>_<CL|MP>_<CALM|SUITE>_<4080|5070>; every non-calm cell
    (single suite, combination, supplement, all-suites stack) uses SUITE."""
    return "CELLMS_%s_%s_%s_%s_%s" % (TASK_CODE[task], "BB" if vkey == "bb" else "WAMV", ctrl.upper(),
                                      "CALM" if kind == "calm" else "SUITE", host)


def all_cells():
    """Every cell of the plan in plan order (the shard of a cell is its block's shard, plan JSON)."""
    cells = []
    for task, vkey, ctrl in blocks():
        flag, vehicle = [(f, v) for k, f, v in VESSELS if k == vkey][0]
        for cond in conditions(task, vkey, ctrl):
            name = "ds_%s_%s_%s_%s" % (task, vkey, ctrl, cond["cond"])
            cells.append(dict(cond, name=name, task=task, vkey=vkey, vessel=flag, vehicle=vehicle, ctrl=ctrl,
                              jid=JOURNAL_ID[(task, vkey)], carrier=CARRIER[task], block=block_id(task, vkey, ctrl),
                              waves=has_waves(cond)))
    if R4_STATE["enabled"]:                       # G32 r4: appended at the end, so no existing order moves
        for task, vkey, ctrl, cond in R4_EXTRA_CELLS:
            flag, vehicle = [(f, v) for k, f, v in VESSELS if k == vkey][0]
            cells.append(dict(cond, name="ds_%s_%s_%s_%s" % (task, vkey, ctrl, cond["cond"]), task=task, vkey=vkey, vessel=flag,
                              vehicle=vehicle, ctrl=ctrl, jid=JOURNAL_ID[(task, vkey)], carrier=CARRIER[task],
                              block=block_id(task, vkey, ctrl), waves=has_waves(cond)))
    return cells


# ---------------------------------------------------------------- G32 r4 (owner decision 2026-10-08)
# BlueBoat station keeping has no dynamic D1 by design (v2.6 G23e: the intruder would have to appear before the start;
# tasks/station_keeping/journal_station.py:93); its all-suites stack therefore uses the lowest offered dynamic tier, D2,
# with every other suite at D1 (the release accepts per-suite tiers in one stack: journal_config.resolve_from_tokens
# gives dynamic_dose 2 and the canonical tokens below). The appearing suite is not part of this smoke's all-suites stack
# (ALLSUITES above), so nothing changes for it. R4_STATE["enabled"] is set only for plans that carry 'g32_r4'; every
# other plan sees exactly the cells it saw before.
R4_STATE = {"enabled": False}
R4_EXTRA_CELLS = (("station_keeping", "bb", "mp",
                   {"cond": "allsuites_dynD2", "kind": "allsuites", "axis": None, "dose": None,
                    "tokens": tuple(t for t in ALLSUITES if not t.startswith("dynamic=")) + ("dynamic=D2",),
                    "canonical": tuple(sorted(("dynamic=D2" if t == "dynamic=D1" else t) for t in ALLSUITES_CANONICAL)),
                    "role": "ALLSUITES", "allsuites_dynamic_dose": 2}),)
# Not applicable by design on v2.6 (BlueBoat station keeping dynamic D1 and the D1 all-suites stack).
R4_NOT_APPLICABLE = ("ds_station_keeping_bb_cl_dyn_D1", "ds_station_keeping_bb_mp_dyn_D1", "ds_station_keeping_bb_mp_allsuites")


def set_r4(enabled):
    R4_STATE["enabled"] = bool(enabled)


def r4_dropped(cell):
    """Owner decision 2026-10-08: the WAM-V MPPI smoke runs only D1 and D4 of every suite (single-suite ladders)."""
    return cell["vkey"] == "wamv" and cell["ctrl"] == "mp" and cell["kind"] == "single" and cell["dose"] in (2, 3)


def r5_dropped(cell):
    """Owner decision 2026-10-08 evening: WAM-V classical too - every WAM-V single-suite D2 / D3 cell (MPPI and classical)."""
    return cell["vkey"] == "wamv" and cell["kind"] == "single" and cell["dose"] in (2, 3)


def expected_controller(cell, names=None):
    """Certificate controller name of a cell. names = {carrier: classical controller name} of a release
    (release_controller_names); None = the v2.4 table above (unchanged behaviour for plan v3 and the running smoke)."""
    if cell["ctrl"] == "mp":
        return "mppi"
    if names is not None:
        return names[cell["carrier"]]
    return CLASSICAL_BY_CARRIER[cell["carrier"]][3]


# ---------------------------------------------------------------- release-aware controller names (G12 update 5)
# G17 (avoidance layer) renames the classical controller of path_following / station_keeping (los_pid_avoid_v1) and
# docking (docking_3phase_pid_avoid_v1) from v2.5 on; v2.4 keeps los_pid / docking_3phase_pid. The names come from the
# release tree itself: scripts/run_cell.py JOURNAL_CLASSICAL_BY_CARRIER (v2.4 run_cell.py:110-112), read STATICALLY
# (ast; nothing of the release is imported or executed). Script, seed flag and extra arguments must equal
# CLASSICAL_BY_CARRIER (the executor's command builder is not release-aware); otherwise the release is refused.
class ReleaseNamesError(ValueError):
    pass


def _static_value(node, env):
    import ast
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ReleaseNamesError("name %s is not a module-level constant" % node.id)
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_static_value(e, env) for e in node.elts]
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ReleaseNamesError("dict unpacking is not supported")
        return {_static_value(k, env): _static_value(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_value(node.left, env), _static_value(node.right, env)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        raise ReleaseNamesError("only string concatenation is supported")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and len(node.args) <= 1:
        out = dict(_static_value(node.args[0], env)) if node.args else {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ReleaseNamesError("dict(**x) is not supported")
            out[kw.arg] = _static_value(kw.value, env)
        return out
    raise ReleaseNamesError("unsupported expression %s" % type(node).__name__)


def release_controller_names(release_root):
    """{carrier: classical controller name} and the SHA-256 of the release's scripts/run_cell.py."""
    import ast
    import hashlib
    from pathlib import Path as _Path
    path = _Path(release_root) / "scripts" / "run_cell.py"
    data = path.read_bytes()
    tree = ast.parse(data.decode("utf-8"), filename=str(path))
    env, mapping = {}, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = _static_value(node.value, env)
            except ReleaseNamesError:
                continue
            if node.targets[0].id == "JOURNAL_CLASSICAL_BY_CARRIER":
                mapping = env["JOURNAL_CLASSICAL_BY_CARRIER"]
    if not isinstance(mapping, dict):
        raise ReleaseNamesError("%s: JOURNAL_CLASSICAL_BY_CARRIER not found as a static dict" % path)
    names = {}
    for carrier, (script, seed_flag, extra, _old) in CLASSICAL_BY_CARRIER.items():
        entry = mapping.get(carrier)
        if not isinstance(entry, dict):
            raise ReleaseNamesError("%s: no classical entry for carrier %s" % (path, carrier))
        if (entry.get("script"), entry.get("seed_flag"), tuple(entry.get("extra") or ())) != (script, seed_flag, tuple(extra)):
            raise ReleaseNamesError("%s: carrier %s runs %r %r %r, the executor builds %r %r %r" % (
                path, carrier, entry.get("script"), entry.get("seed_flag"), entry.get("extra"), script, seed_flag, extra))
        name = entry.get("controller_name")
        if not isinstance(name, str) or not name or len(name) > 60 or not all(ch.islower() or ch.isdigit() or ch == "_" for ch in name) or not name[0].isalpha():
            raise ReleaseNamesError("%s: bad controller name %r for carrier %s" % (path, name, carrier))
        names[carrier] = name
    return names, hashlib.sha256(data).hexdigest().upper()



# ---------------------------------------------------------------- G32 v2.6: release generations (additions only)
# Classical certificate names per release generation, as each release's scripts/run_cell.py declares them
# (JOURNAL_CLASSICAL_BY_CARRIER): v2.4 = CLASSICAL_BY_CARRIER above (release 6dfbb709f9c5-0a91174ae258); v2.5 = G17
# avoidance layer (release 97af8d40982b-d8c1be39a8b0); v2.6 = G24b (20261006_g24b_classical_fixes/work/CONTROLLER_NAMES.md;
# release f7eec671ce0d-0d729fdf5560, run_cell.py SHA-256 00912927032D7F4C...).
NAMES_V24 = {carrier: entry[3] for carrier, entry in CLASSICAL_BY_CARRIER.items()}
NAMES_V25 = dict(NAMES_V24, path_following="los_pid_avoid_v1", station_keeping="los_pid_avoid_v1",
                 docking="docking_3phase_pid_avoid_v1")
NAMES_V26 = {"hazard_nav": "planner_los_pid_mem_v2", "path_hazard": "planner_los_pid_mem_v2",
             "path_following": "los_pid_avoid_v2", "station_keeping": "los_pid_cont_avoid_v2",
             "docking": "docking_runup_pid_avoid_v2", "harbor_mission": "staged_harbour"}
RELEASE_GENERATIONS = (("v2.4", NAMES_V24), ("v2.5", NAMES_V25), ("v2.6", NAMES_V26))
GENERATION_RANK = {label: rank for rank, (label, _names) in enumerate(RELEASE_GENERATIONS)}


def names_generation(names):
    """'v2.4' / 'v2.5' / 'v2.6' when names equal that generation's table exactly, else None."""
    for label, table in RELEASE_GENERATIONS:
        if names == table:
            return label
    return None


def at_least(generation, label):
    return generation is not None and GENERATION_RANK[generation] >= GENERATION_RANK[label]


def release_generation(release_root):
    """(generation or None, names, run_cell.py SHA-256) of a release tree (ReleaseNamesError / OSError propagate)."""
    names, sha = release_controller_names(release_root)
    return names_generation(names), names, sha


def release_default_names(release_root):
    """Expected classical names for cells that carry none: the release's own names from v2.6 on; None (the v2.4
    table, exactly the behaviour before G32) for v2.4, v2.5, unknown or unreadable releases."""
    try:
        generation, names, _sha = release_generation(release_root)
    except (ReleaseNamesError, OSError, SyntaxError, UnicodeDecodeError):
        return None
    return dict(names) if at_least(generation, "v2.6") else None

def counts():
    cells = all_cells()
    cl = [c for c in cells if c["ctrl"] == "cl"]
    mp = [c for c in cells if c["ctrl"] == "mp"]
    return {"classical": len(cl), "mppi": len(mp), "total": len(cells),
            "classical_calm_plus_single": sum(1 for c in cl if c["kind"] in ("calm", "single")),
            "classical_combination": sum(1 for c in cl if c["kind"] == "combo"),
            "classical_supplement": sum(1 for c in cl if c["kind"] == "supplement"),
            "mppi_calm_and_allsuites": sum(1 for c in mp if c["kind"] in ("calm", "allsuites")),
            "mppi_single_combination_supplement": sum(1 for c in mp if c["kind"] not in ("calm", "allsuites")),
            "mppi_supplement": sum(1 for c in mp if c["kind"] == "supplement"),
            "blocks": len(list(blocks()))}


if __name__ == "__main__":
    import json
    print(json.dumps(counts(), indent=1))
