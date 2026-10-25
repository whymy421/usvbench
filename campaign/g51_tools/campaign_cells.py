"""G51 campaign cell universe (pure Python, ASCII, stdlib only). Shared by make_campaign_plan.py, the scheduler simulation,
the orchestrator and the campaign judge.

Main table (the paper's main results), on release v2.6.x, formal test blocks final-01..04 (seed registry final_test,
planned_main_blocks = 4, 128 episodes = 64 envs x 2 per block):
  12 journal tasks x 2 vessels x controllers {classical, MPPI c3, PPO x 3 training candidates (fixed_first_n: seeds
  42, 1001, 2003 = the first 3 of candidate_order)} x 53 conditions (calm + 13 suites x D1..D4: waves, wind, current,
  actuator.cap/.gain/.lag/.resistance, perception.noise/.bias/.latency/.dropout, dynamic, appearing) - the G13 'MT'
  definition (20261002_g13_campaign_scheduler/work/catalog.py:14) - plus one all-suites stack per group (the smoke's
  ALLSUITES = every suite at D1 except appearing; BlueBoat station keeping uses dynamic D2: work_v26/dose_plan.py
  R4_EXTRA_CELLS) as the STACK stratum.
Not applicable by design (never planned; any certificate for them is a plan error):
  * station_keeping BlueBoat dynamic D1 (v2.6 G23e; work_v26/dose_plan.py R4_NOT_APPLICABLE; journal_station.py:93);
  * appearing D1 where the release's tasks/_shared/conditions/appearing/ladder.py NOT_APPLICABLE_DOSES says so
    (v2.6.5: station_keeping BlueBoat D1 = owner 2B 2026-10-04; docking BlueBoat D1 = owner 2026-10-14 11:xx), read from
    the release tree at plan time.
Real-sea combinations are a side table on tasks still to be chosen (owner 2026-10-02 12:xx) and are NOT in this plan; the
BlueBoat waves supplement supp075 was a smoke check and is NOT in the main table.
"""
import ast
from pathlib import Path
import sys

WORK_V26 = Path(r"D:\claude code\codex_tasks\20261002_g12_dose_smoke\work_v26")
if str(WORK_V26) not in sys.path:
    sys.path.insert(0, str(WORK_V26))
import dose_plan as P  # noqa: E402  (TASKS, TASK_CODE, VESSELS, JOURNAL_ID, CARRIER, AXES, ALLSUITES, names)

SEED_BLOCKS = (("final-01", 41041), ("final-02", 41047), ("final-03", 41051), ("final-04", 41057))
PPO_SEEDS = (42, 1001, 2003)          # seed_registry.json ppo_training.candidate_order[:admit_first_passing=3], fixed_first_n
AXES = tuple(P.AXES) + (("appearing", "app"),)
DOSES = (1, 2, 3, 4)
SK_BB_DYN_D1 = ("station_keeping", "bb", "dynamic", 1)
CTRL_CODE = {"cl": "CL", "mp": "MP", "ppo": "PPO"}
VESSEL_ID = {"bb": "blueboat_v2", "wamv": "wamv_vrx_v2_opt1"}


def release_not_applicable(release_root):
    """{(task, vkey): {dose: reason}} for the appearing suite, read statically from the release's ladder.py."""
    path = Path(release_root) / "tasks" / "_shared" / "conditions" / "appearing" / "ladder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    table = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "NOT_APPLICABLE_DOSES":
            table = ast.literal_eval(node.value)
    if not isinstance(table, dict):
        raise SystemExit("NOT_APPLICABLE_DOSES not found as a literal in %s" % path)
    out = {}
    back = {v: k for k, v in VESSEL_ID.items()}
    for (task, vehicle), doses in table.items():
        out[(task, back[vehicle])] = dict(doses)
    return out


def ctrl_label(ctrl, ppo_seed):
    return "p%d" % ppo_seed if ctrl == "ppo" else ctrl


def block_id(task, vkey, ctrl, ppo_seed, seed_index):
    code = "P%d" % ppo_seed if ctrl == "ppo" else CTRL_CODE[ctrl]
    return "%s_%s_%s_F%d" % (P.TASK_CODE[task], vkey.upper(), code, seed_index)


def table_block(task, vkey, ctrl, ppo_seed=None):
    """The paper's table column a cell belongs to (one machine class per table block)."""
    code = "P%d" % ppo_seed if ctrl == "ppo" else CTRL_CODE[ctrl]
    return "%s_%s_%s" % (P.TASK_CODE[task], vkey.upper(), code)


def conditions(task, vkey, na):
    """Ordered conditions of one (task, vessel): calm, the 13 ladders, the all-suites stack; n/a cells removed."""
    out = [{"cond": "calm", "kind": "calm", "axis": None, "dose": 0, "tokens": (), "stratum": "MT"}]
    for axis, prefix in AXES:
        for dose in DOSES:
            if (task, vkey, axis, dose) == SK_BB_DYN_D1:
                continue
            if axis == "appearing" and dose in na.get((task, vkey), {}):
                continue
            token = "%s=D%d" % (axis, dose)
            out.append({"cond": "%s_D%d" % (prefix, dose), "kind": "single", "axis": axis, "dose": dose,
                        "tokens": (token,), "stratum": "MT"})
    if (task, vkey) == ("station_keeping", "bb"):
        extra = P.R4_EXTRA_CELLS[0][3]
        out.append({"cond": extra["cond"], "kind": "allsuites", "axis": None, "dose": None, "tokens": tuple(extra["tokens"]),
                    "stratum": "STACK"})
    else:
        out.append({"cond": "allsuites", "kind": "allsuites", "axis": None, "dose": None, "tokens": tuple(P.ALLSUITES),
                    "stratum": "STACK"})
    return out


def not_applicable_list(na):
    rows = [("station_keeping", "bb", "dynamic", 1, "v2.6 G23e: the intruder would appear before the start (dose_plan R4)")]
    for (task, vkey), doses in sorted(na.items()):
        for dose, reason in sorted(doses.items()):
            rows.append((task, vkey, "appearing", dose, reason))
    return rows


def all_cells(release_root, names, controllers=("cl", "mp", "ppo"), seed_blocks=SEED_BLOCKS, ppo_seeds=PPO_SEEDS):
    """Every main-table cell in plan order: seed block, then controller, vessel, task, condition."""
    na = release_not_applicable(release_root)
    cells = []
    for s_index, (blk, seed) in enumerate(seed_blocks, start=1):
        for ctrl in controllers:
            for ppo_seed in (ppo_seeds if ctrl == "ppo" else (None,)):
                for vkey, flag, vehicle in P.VESSELS:
                    for task in P.TASKS:
                        for cond in conditions(task, vkey, na):
                            label = ctrl_label(ctrl, ppo_seed)
                            name = "fc_%s_%s_%s_%s_f%d" % (task, vkey, label, cond["cond"], s_index)
                            carrier = P.CARRIER[task]
                            expected = "mppi" if ctrl == "mp" else ("learned" if ctrl == "ppo" else names[carrier])
                            cells.append(dict(cond, name=name, task=task, vkey=vkey, vessel=flag, vehicle=vehicle, ctrl=ctrl,
                                              ppo_seed=ppo_seed, seed_block=blk, seed=seed, seed_index=s_index,
                                              jid=P.JOURNAL_ID[(task, vkey)], carrier=carrier,
                                              block=block_id(task, vkey, ctrl, ppo_seed, s_index),
                                              table_block=table_block(task, vkey, ctrl, ppo_seed),
                                              expected_controller=expected, waves=P.has_waves(dict(cond, kind=cond["kind"]))))
    return cells, na


# REPRO-DUP stratified sample (seed_registry crossmachine_rule; scripts/crossmachine_sample.py design: vessels x
# controllers x short/long task x low/high dose = 24 cells): short = path_following, long = scatter, low/high = current D1/D4
# (no pool and no green-water dependence). The duplicates re-run the campaign's own final-01 cells under new names
# 'xm<target>_<main name>' on a host of a different class (5070, later 4080) or a different rental; they never count as
# main-table data. PPO rows are added when the PPO candidate 42 of those tasks has its delivered checkpoint.
REPRO_TASKS = ("path_following", "scatter")
REPRO_CONDS = ("cur_D1", "cur_D4")


def repro_dup_cells(main_cells, controllers=("cl", "mp")):
    want = {(t, v, c, cond) for t in REPRO_TASKS for v in ("bb", "wamv") for c in controllers for cond in REPRO_CONDS}
    return [c for c in main_cells if c["seed_block"] == "final-01" and c["ppo_seed"] in (None, 42)
            and (c["task"], c["vkey"], c["ctrl"], c["cond"]) in want]
