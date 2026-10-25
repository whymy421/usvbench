"""G25b CPU tests: pinned certificate blocks of the memory planner (ray_memory) and of the G17 layer (avoidance).

REV-G25 MAJOR-2 (work/merge/evidence/tamper_summary.txt of 20261003_rev_g25): validate_cert_v3 accepted 16 of 39
tampered ray_memory blocks per vessel because it re-derived values only from the block's own numbers. Every tamper
case of that summary (T01-T39, both vessels) must now be reported; T00 (valid) and T12 (no telemetry on an
incomplete certificate) stay valid. The G17 avoidance block gets the same pinning against the frozen
scripts/classical_avoidance_v2.json (extra keys, rule / vessel / task constants, file digest with a code root).

Run: python -B -m pytest -p no:cacheprovider scripts/test_planner_memory_pins.py -q
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pytest  # noqa: E402

import planner_memory as pm  # noqa: E402
import validate_cert_v3 as v  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
OTHER = {"blueboat_v2": "wamv_vrx_v2_opt1", "wamv_vrx_v2_opt1": "blueboat_v2"}
ROOT = SCRIPTS.parent


def memory_header(vessel="blueboat_v2", complete=True, task="fortress", carrier="hazard_nav"):
    """The evaluator's certificate header parts (classical_planner_pid._run_isaac_eval, journal branch)."""
    import classical_planner_pid as cp

    terms = pm.vessel_terms(vessel)
    scale = terms["length_scale"]
    cell = pm.PLAN_CELL_REF_M * scale
    ladder = list(cp.journal_memory_ladder(vessel))
    mem = pm.RayReturnMemory(1, plan_cell_m=cell, replan_every=pm.REPLAN_EVERY_STEPS, length_scale=scale,
                             ray_max_range_m=terms["ray_max_range_m"], reference_speed_mps=terms["reference_speed_mps"],
                             control_dt_s=terms["control_dt_s"])
    block = {**mem.stamp(), "controller_name": pm.CONTROLLER_NAME, "base_controller_name": pm.BASE_CONTROLLER_NAME,
             "vessel": vessel, "ladder": {**cp.journal_memory_ladder_terms(vessel), "inflation_ladder_m": ladder},
             "telemetry": {**mem.telemetry(), "legs_planned": 7, "unplannable_legs": 1, "hold_steps": 3,
                           "escape_plans": 1, "escape_failures": 0, "legs_by_rung": [4, 2, 0][:len(ladder)],
                           "episode_rows_fields": ["env", "episode", "hold_steps", "unplannable_legs", "escape_plans",
                                                   "escape_failures"],
                           "episode_rows": [[0, 0, 3, 1, 1, 0]], "unfinished_rows": []}}
    # G24b (REV-G25b M3): the full configuration classical_planner_pid._run_isaac_eval writes on a journal id
    import journal_controllers as jc
    gains = jc.load_gains(vessel)
    configuration = {"controller": pm.CONTROLLER_NAME, "mode": "gates" if carrier == "path_hazard" else "goal",
                     "replan_every": pm.REPLAN_EVERY_STEPS, "waypoint_spacing_m": 2.0 * scale,
                     "accept_radius_m": 1.0 * scale, "plan_cell_m": cell, "inflation_ladder_m": ladder,
                     **gains["los_heading_pid"], "privileged": False,
                     "journal": {"journal": True, "vessel": vessel, "information": jc.INFORMATION_DIET, "gains": gains,
                                 "length_scale": jc.length_scale(vessel), "ray_max_range_m": terms["ray_max_range_m"],
                                 "obstacles": "current and remembered ray returns as memory discs (G25b planner_memory)"},
                     "control_step_s": terms["control_dt_s"], "ray_memory": block, "plan_margin_m": 0.15,
                     "planning_padding_m": 6.0, "snap_start_m": pm.SNAP_START_REF_M * scale, "turn_creep_thrust": 0.2,
                     "aligned_thrust_gain": 0.8, "action_limits": [-1.0, 1.0], "planning_dtype": "numpy.float64",
                     "pid_dtype": "torch.float32", "num_envs": 32, "device": "cuda:0"}
    header = {"vessel": vessel, "complete": complete, "journal": {"task": task, "carrier": carrier},
              "controller": {"name": pm.CONTROLLER_NAME, "configuration": configuration}}
    return json.loads(json.dumps(header, allow_nan=False))             # the certificate's JSON round trip


def _blk(h):
    return h["controller"]["configuration"]["ray_memory"]


def _set_ladder(h, ladder):
    _blk(h)["ladder"]["inflation_ladder_m"] = list(ladder)
    h["controller"]["configuration"]["inflation_ladder_m"] = list(ladder)


def _recompute_ladder(h):
    _set_ladder(h, pm.memory_ladder(_blk(h)["ladder"]))


def tamper_cases(vessel):
    """(id, description, header) for REV-G25's T01-T39 (T00 / T12 are the valid controls)."""
    out = []

    def case(tid, text, fn):
        h = memory_header(vessel)
        fn(h)
        out.append((tid, text, h))

    case("T01", "rule quantum_cells 0.5 -> 0.4", lambda h: _blk(h)["rule_constants"].update(quantum_cells=0.4))
    # G24b: T02 / T04 / T31 re-add the removed expiry (G25b's S2) instead of changing its value
    case("T02", "rule ttl_sensor_crossings re-added (G25b S2)",
         lambda h: _blk(h)["rule_constants"].update(ttl_sensor_crossings=1.0))
    case("T03", "rule capacity 8192 -> 100", lambda h: _blk(h)["rule_constants"].update(capacity_entries=100))
    case("T04", "constants ttl_steps re-added (G25b S2)", lambda h: _blk(h)["constants"].update(ttl_steps=1800))

    def t05(h):
        lad = list(_blk(h)["ladder"]["inflation_ladder_m"])
        lad[0] = _blk(h)["ladder"]["circumscribed_radius_m"] - 0.1
        _set_ladder(h, lad)
    case("T05", "ladder first rung = C - 0.1", t05)

    def t06(h):
        _blk(h)["ladder"]["margin_lengths"] = -0.5
        _recompute_ladder(h)
    case("T06", "ladder margin -0.5 s, ladder recomputed", t06)

    def t07(h):
        _blk(h)["ladder"]["circumscribed_radius_m"] *= 0.5
        _recompute_ladder(h)
    case("T07", "circumscribed radius x 0.5, consistent", t07)

    def t08(h):
        _blk(h)["ladder"]["margin_lengths"] = 0.0
        _blk(h)["ladder"]["narrow_margin_lengths"] = 0.0
        _recompute_ladder(h)
    case("T08", "ladder margins -> 0, consistent", t08)
    case("T09", "block vessel swapped only", lambda h: _blk(h).update(vessel=OTHER[vessel]))

    def t10(h):
        h["vessel"] = OTHER[vessel]
        _blk(h)["vessel"] = OTHER[vessel]
    case("T10", "header + block vessel swapped, original constants", t10)
    case("T11", "telemetry dropped on a complete cert", lambda h: _blk(h).update(telemetry=None))
    case("T13", "legs_by_rung one entry too many", lambda h: _blk(h)["telemetry"]["legs_by_rung"].append(0))
    case("T14", "legs per rung + unplannable != legs planned", lambda h: _blk(h)["telemetry"].update(legs_planned=99))
    case("T15", "telemetry inserted = True", lambda h: _blk(h)["telemetry"].update(inserted=True))

    def t16(h):
        t = _blk(h)["telemetry"]
        t["legs_by_rung"] = [3, True, 0][:len(t["legs_by_rung"])]
    case("T16", "legs_by_rung holds a bool (sum kept)", t16)
    case("T17", "telemetry evicted = -1", lambda h: _blk(h)["telemetry"].update(evicted=-1))
    case("T18", "telemetry peak_entries = 40.0", lambda h: _blk(h)["telemetry"].update(peak_entries=40.0))
    case("T19", "telemetry key refreshed missing", lambda h: _blk(h)["telemetry"].pop("refreshed"))
    case("T20", "telemetry extra key", lambda h: _blk(h)["telemetry"].update(extra=0))
    case("T21", "block on a planner_los_pid cert", lambda h: h["controller"].update(name="planner_los_pid"))
    case("T22", "block missing on a mem_v2 cert", lambda h: h["controller"]["configuration"].pop("ray_memory"))
    case("T23", "base controller name wrong", lambda h: _blk(h).update(base_controller_name="los_pid"))
    case("T24", "block controller_name = base name", lambda h: _blk(h).update(controller_name="planner_los_pid"))
    case("T25", "module_sha256 = 64 zeros (code root given)", lambda h: _blk(h).update(module_sha256="0" * 64))
    case("T26", "module_sha256 malformed", lambda h: _blk(h).update(module_sha256="ABC"))
    case("T27", "rule_constants extra key", lambda h: _blk(h)["rule_constants"].update(extra=1))
    case("T28", "rule key carve_margin_cells missing", lambda h: _blk(h)["rule_constants"].pop("carve_margin_cells"))
    case("T29", "constants extra key", lambda h: _blk(h)["constants"].update(extra=1))

    def t30(h):
        _blk(h)["rule_constants"]["capacity_entries"] = 100
        _blk(h)["constants"]["capacity_entries"] = 100
    case("T30", "capacity 100 in rules AND constants", t30)

    def t31(h):
        r, c = _blk(h)["rule_constants"], _blk(h)["constants"]
        r.update(ttl_sensor_crossings=0.01, carve_lateral_cells=0.0, carve_margin_cells=0.0)
        c.update(ttl_steps=36, carve_lateral_m=0.0, carve_margin_m=0.0)
    case("T31", "short ttl re-added, no carve, consistent", t31)
    case("T32", "rule quantum_cells as a string", lambda h: _blk(h)["rule_constants"].update(quantum_cells="0.5"))

    def t33(h):
        h["controller"]["configuration"]["replan_every"] = 10
        _blk(h)["constants"]["replan_every_steps"] = 10
    case("T33", "replan_every 10, consistent", t33)

    def t34(h):
        c = _blk(h)["constants"]
        h["controller"]["configuration"]["plan_cell_m"] *= 2.0
        for k in ("plan_cell_m", "quantum_m", "disc_radius_m", "carve_lateral_m", "carve_margin_m"):
            c[k] *= 2.0
    case("T34", "plan cell x 2, consistent", t34)
    case("T35", "format wrong", lambda h: _blk(h).update(format="usvbench-planner-memory-v2"))

    def t36(h):
        h["journal"] = {"task": "path_following", "carrier": "path_following"}
    case("T36", "memory controller on a path_following header", t36)
    case("T37", "configuration.controller = base name",
         lambda h: h["controller"]["configuration"].update(controller="planner_los_pid"))
    case("T38", "telemetry is a list", lambda h: _blk(h).update(telemetry=[1, 2]))

    def t39(h):
        b = _blk(h)
        b["ladder"]["length_scale"] *= 1.5
        b["constants"]["length_scale"] *= 1.5
        _recompute_ladder(h)
    case("T39", "length scale x 1.5 everywhere, consistent", t39)
    case("T40", "memory_version of G25 (v1, superseded)", lambda h: _blk(h).update(memory_version="g25_ray_memory_v1"))
    case("T41", "block key extra", lambda h: _blk(h).update(note="x"))
    # G24b: the G25b versions (with the expiry) are not accepted any more
    case("T42", "memory_version of G25b (v2, with expiry)", lambda h: _blk(h).update(memory_version="g25b_ray_memory_v2"))

    def t43(h):
        h["controller"]["name"] = "planner_los_pid_mem_v1"
        h["controller"]["configuration"]["controller"] = "planner_los_pid_mem_v1"
        _blk(h)["controller_name"] = "planner_los_pid_mem_v1"
    case("T43", "G25b certificate name planner_los_pid_mem_v1, consistent", t43)

    def t44(h):
        t = _blk(h)["telemetry"]
        t["expired"] = 0
    case("T44", "telemetry carries G25b's expired counter", t44)
    return out


@pytest.mark.parametrize("vessel", VESSELS)
def test_valid_block_passes_with_and_without_code_root(vessel):
    header = memory_header(vessel)
    assert v.planner_memory_errors(header) == []
    assert v.planner_memory_errors(header, code_root=ROOT) == []
    for task, carrier in (("path_hazard", "path_hazard"), ("reach", "hazard_nav"), ("moving_basin", "hazard_nav")):
        assert v.planner_memory_errors(memory_header(vessel, task=task, carrier=carrier)) == []
    unfinished = memory_header(vessel, complete=False)                 # T12: telemetry None on an incomplete cert
    _blk(unfinished)["telemetry"] = None
    assert v.planner_memory_errors(unfinished, code_root=ROOT) == []
    schema = json.loads((SCRIPTS / "cert_v3.schema.json").read_text(encoding="ascii"))
    from jsonschema import Draft202012Validator
    sub = {"$defs": schema["$defs"], "$ref": "#/$defs/planner_memory"}
    assert list(Draft202012Validator(sub).iter_errors(_blk(header))) == []
    plain = {"vessel": vessel, "controller": {"name": "planner_los_pid", "configuration": {}}}
    assert v.planner_memory_errors(plain) == []                         # v2.5 planner certificates stay valid


@pytest.mark.parametrize("vessel", VESSELS)
def test_every_rev_tamper_case_is_reported(vessel):
    schema = json.loads((SCRIPTS / "cert_v3.schema.json").read_text(encoding="ascii"))
    from jsonschema import Draft202012Validator
    sub = {"$defs": schema["$defs"], "$ref": "#/$defs/planner_memory"}
    silent = []
    for tid, text, header in tamper_cases(vessel):
        errors = v.planner_memory_errors(header, code_root=ROOT)
        block = header["controller"]["configuration"].get("ray_memory")
        if tid == "T26" and isinstance(block, dict):                     # the schema pattern reports it too
            errors = errors + [e.message for e in Draft202012Validator(sub).iter_errors(block)]
        if not errors:
            silent.append((tid, text))
    assert silent == [], silent
    assert len(tamper_cases(vessel)) == 43


def test_registry_pins_reject_another_vessels_terms():
    header = memory_header("blueboat_v2")
    _blk(header)["ladder"] = copy.deepcopy(_blk(memory_header("wamv_vrx_v2_opt1"))["ladder"])
    assert any("registry" in e for e in v.planner_memory_errors(header))


# ------------------------------------------------------------------------ G17 avoidance block pins --
MODES = {"path_follow": "path_following", "station_keep": "station_keeping", "docking": "docking"}


def avoidance_header(mode="path_follow", vessel="blueboat_v2"):
    import journal_avoidance as ja

    avoidance = ja.load_avoidance(vessel, MODES[mode])
    block = ja.controller_block(mode, avoidance, hold_zone=(2.0, 5.0), berth_walls=(mode == "docking"))
    counters = {k: 0 for k in v.AVOIDANCE_TELEMETRY_COUNTS + ("wall_registrations", "closing_steps", "memory_steps",  # noqa
                                                             "memory_dropped")}
    block["telemetry"] = {**counters, "episode_rows_fields": ["env", "episode", "active_steps", "activations"],
                          "episode_rows": [], "unfinished_rows": [], "control_step_s": 1.0 / 60.0}
    # classical_baseline.py's journal block (G24b, REV-G25b M3: every field pinned)
    import journal_controllers as jc
    import planner_memory as pmm
    s = jc.length_scale(vessel)
    params = ja.journal_plan(MODES[mode], vessel).params
    zone = {"station_keep": params.get("hold_radius_m"), "docking": params.get("position_tolerance_m"),
            "path_follow": 2.0}[mode]
    configuration = {"journal": True, "vessel": vessel, "information": jc.INFORMATION_DIET, "gains": jc.load_gains(vessel),
                     "length_scale": s, "ray_max_range_m": 30.0 * s, "mode": mode, "hold_mode": ja.journal_hold_mode(mode),
                     "zone_radius_m": zone, "slow_radius_m": (6.0 if mode == "docking" else 5.0) * s,
                     "docking_tolerance_m": params["position_tolerance_m"] if mode == "docking" else 2.5 * s,
                     "controller_class": "journal_avoidance.AvoidanceLayerController", "avoidance": block,
                     "protocol": "mission", "num_envs": 32, "device": "cuda:0",
                     "dt": pmm.vessel_terms(vessel)["control_dt_s"], "action_limits": [-1.0, 1.0], "dtype": "torch.float32"}
    header = {"controller": {"name": ja.CONTROLLER_NAMES[mode], "configuration": configuration},
              "vessel": vessel, "journal": {"task": MODES[mode]}, "complete": True}
    return json.loads(json.dumps(header))


def _ablk(h):
    return h["controller"]["configuration"]["avoidance"]


@pytest.mark.parametrize("vessel", VESSELS)
@pytest.mark.parametrize("mode", sorted(MODES))
def test_valid_avoidance_block_passes_both_checks(mode, vessel):
    header = avoidance_header(mode, vessel)
    assert v.avoidance_layer_errors(header) == []
    assert v.avoidance_pin_errors(header) == [] and v.avoidance_pin_errors(header, code_root=ROOT) == []


def avoidance_tamper_cases():
    out = []

    def case(tid, text, fn, mode="path_follow", vessel="blueboat_v2"):
        h = avoidance_header(mode, vessel)
        fn(h)
        out.append((tid, text, h))

    case("A01", "block extra key", lambda h: _ablk(h).update(note=1))
    case("A02", "block key information missing", lambda h: _ablk(h).pop("information"))
    case("A03", "rule constant changed", lambda h: _ablk(h)["rule_constants"].update(trigger_ttc_ratio=5.0))
    case("A04", "rule constants extra key", lambda h: _ablk(h)["rule_constants"].update(extra=1))
    case("A05", "vessel constant changed consistently with its rule",
         lambda h: _ablk(h)["vessel_constants"].update(static_mask_tolerance_m=1.0))
    case("A06", "vessel constants extra key", lambda h: _ablk(h)["vessel_constants"].update(extra=1))
    case("A07", "task constants of another task", lambda h: _ablk(h).update(
        task_constants=copy.deepcopy(avoidance_header("station_keep")["controller"]["configuration"]["avoidance"]
                                     ["task_constants"])))
    case("A08", "mode does not match task", lambda h: _ablk(h).update(mode="docking"))
    case("A09", "motion_prediction true", lambda h: _ablk(h).update(motion_prediction=True))
    case("A10", "file path changed", lambda h: _ablk(h).update(file="scripts/other.json"))
    case("A11", "file_sha256 = zeros (code root given)", lambda h: _ablk(h).update(file_sha256="0" * 64))
    case("A12", "hold zone inverted", lambda h: _ablk(h).update(hold_zone_m=[5.0, 2.0]))
    case("A13", "telemetry extra key", lambda h: _ablk(h)["telemetry"].update(extra=0))
    case("A14", "WAM-V block with BlueBoat vessel constants", lambda h: (
        h.update(vessel="wamv_vrx_v2_opt1"), _ablk(h)["vessel_constants"].update(vessel="wamv_vrx_v2_opt1")))
    case("A15", "layer version changed", lambda h: _ablk(h).update(layer_version="g17_avoidance_v2"))
    # G24b pins
    case("A16", "station keeping ran the deadband", lambda h: h["controller"]["configuration"].update(hold_mode="deadband"),
         mode="station_keep")
    case("A17", "docking run-up point moved 1 m", lambda h: _ablk(h)["task_constants"].update(
        runup_x_m=_ablk(h)["task_constants"]["runup_x_m"] + 1.0), mode="docking")
    case("A18", "memory rule steps (G17 expiry)", lambda h: _ablk(h)["rule_constants"].update(obstacle_memory_rule="steps"))

    def a19(h):
        h["controller"]["name"] = "los_pid_avoid_v1"
        _ablk(h)["controller_name"] = "los_pid_avoid_v1"
    case("A19", "G17 name los_pid_avoid_v1, consistent", a19)
    case("A20", "G17 layer version", lambda h: _ablk(h).update(layer_version="g17_avoidance_v1"))
    case("A21", "station keeping block under the path-following name", lambda h: (
        h["controller"].update(name="los_pid_avoid_v2"), _ablk(h).update(controller_name="los_pid_avoid_v2")),
         mode="station_keep")
    return out


def test_every_avoidance_tamper_case_is_reported():
    silent = []
    for tid, text, header in avoidance_tamper_cases():
        errors = v.avoidance_layer_errors(header) + v.avoidance_pin_errors(header, code_root=ROOT)
        if not errors:
            silent.append((tid, text))
    assert silent == [], silent
    # A05 / A01 / A04 / A06 / A13 pass the G17 derivation check alone; only the pins catch them
    pinned_only = [tid for tid, _t, h in avoidance_tamper_cases()
                   if not v.avoidance_layer_errors(h) and v.avoidance_pin_errors(h, code_root=ROOT)]
    assert {"A01", "A04", "A13"} <= set(pinned_only), pinned_only


# ------------------------------------------------------------- G24b: every configuration field pinned (REV-G25b M3) --
@pytest.mark.parametrize("vessel", VESSELS)
def test_full_configuration_blocks_pass(vessel):
    for task, carrier in (("fortress", "hazard_nav"), ("path_hazard", "path_hazard")):
        header = memory_header(vessel, task=task, carrier=carrier)
        assert v.planner_configuration_errors(header) == [] and v.planner_configuration_errors(header, code_root=ROOT) == []
        assert v.planner_memory_errors(header, code_root=ROOT) == []
    for mode in MODES:
        header = avoidance_header(mode, vessel)
        assert v.avoidance_configuration_errors(header) == [], (mode, v.avoidance_configuration_errors(header))
        assert v.avoidance_configuration_errors(header, code_root=ROOT) == []


def test_every_rev_g25b_configuration_tamper_is_reported():
    """REV-G25b M3: the 12 configuration fields that passed G25b's validator when tampered, plus the avoidance block's."""
    def cfg(h):
        return h["controller"]["configuration"]
    planner = [
        ("waypoint_spacing_m x3", lambda h: cfg(h).update(waypoint_spacing_m=cfg(h)["waypoint_spacing_m"] * 3)),
        ("accept_radius_m x3", lambda h: cfg(h).update(accept_radius_m=cfg(h)["accept_radius_m"] * 3)),
        ("kp x10", lambda h: cfg(h).update(kp=cfg(h)["kp"] * 10)),
        ("kd = 0", lambda h: cfg(h).update(kd=0.0)),
        ("privileged true", lambda h: cfg(h).update(privileged=True)),
        ("control_step_s 0.5", lambda h: cfg(h).update(control_step_s=0.5)),
        ("journal ray range x0.5", lambda h: cfg(h)["journal"].update(ray_max_range_m=cfg(h)["journal"]["ray_max_range_m"] * 0.5)),
        ("journal gains changed", lambda h: cfg(h)["journal"]["gains"]["los_heading_pid"].update(kp=9.0)),
        ("obstacles text of G25", lambda h: cfg(h)["journal"].update(obstacles="current and remembered ray returns")),
        ("mode swapped", lambda h: cfg(h).update(mode="gates")),
        ("extra key", lambda h: cfg(h).update(note=1)),
        ("hold_steps 1e9", lambda h: h["controller"]["configuration"]["ray_memory"]["telemetry"].update(hold_steps=10 ** 9)),
        ("snap start not scaled", lambda h: cfg(h).update(snap_start_m=1.5)),
        ("escape count above unplannable", lambda h: cfg(h)["ray_memory"]["telemetry"].update(escape_plans=5)),
        ("episode rows do not add up", lambda h: cfg(h)["ray_memory"]["telemetry"].update(episode_rows=[])),
        ("escape rule changed", lambda h: cfg(h)["ray_memory"]["rule_constants"].update(hold_escape_failed_replans=3)),
    ]
    silent = []
    for text, fn in planner:
        h = memory_header("wamv_vrx_v2_opt1")
        fn(h)
        if not (v.planner_configuration_errors(h, code_root=ROOT) + v.planner_memory_errors(h, code_root=ROOT)):
            silent.append(text)
    avoidance = [
        ("station_keep", "zone radius x2", lambda h: cfg(h).update(zone_radius_m=cfg(h)["zone_radius_m"] * 2)),
        ("station_keep", "slow radius changed", lambda h: cfg(h).update(slow_radius_m=1.0)),
        ("docking", "docking tolerance changed", lambda h: cfg(h).update(docking_tolerance_m=1.0)),
        ("path_follow", "gains changed", lambda h: cfg(h)["gains"]["los_heading_pid"].update(kd=0.0)),
        ("path_follow", "controller class", lambda h: cfg(h).update(controller_class="x")),
        ("path_follow", "dt changed", lambda h: cfg(h).update(dt=0.5)),
        ("docking", "mode swapped", lambda h: cfg(h).update(mode="path_follow")),
        ("path_follow", "extra key", lambda h: cfg(h).update(note=1)),
        ("path_follow", "information changed", lambda h: cfg(h).update(information="x")),
        ("docking", "hold mode continuous on docking", lambda h: cfg(h).update(hold_mode="continuous")),
    ]
    for mode, text, fn in avoidance:
        h = avoidance_header(mode, "wamv_vrx_v2_opt1")
        fn(h)
        if not v.avoidance_configuration_errors(h, code_root=ROOT):
            silent.append(mode + ": " + text)
    assert silent == [], silent
