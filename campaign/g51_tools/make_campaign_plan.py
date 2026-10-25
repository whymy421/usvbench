#!/usr/bin/env python3
"""G51: the formal campaign as dispatchable batch files (FINAL mode, run_cell.py route) for the G51 Linux executor.

usage:
  python -B make_campaign_plan.py --name c1 --release-root <v2.6.x extract> --release-id <id> --timing <timing_model_g51.json>
      --predictions <predictions file> --controllers cl,mp [--seed-blocks final-01,final-02,final-03,final-04]
      [--batch-hours 6] --out-json plan/campaign_c1.json --out-text plan/campaign_c1.txt --out-dir plan/campaign_c1
  PPO (one delivered checkpoint per run, after its training and gate):
  python -B make_campaign_plan.py --name p-SC-WAMV-42 ... --controllers ppo --ppo-run scatter:wamv:42
      --ppo-checkpoint <remote abs path> --ppo-checkpoint-sha256 <hex> --gate-ledger <remote abs path> --gate-ledger-sha256 <hex>
  REPRO-DUP (cross-machine duplicates of the stratified sample): add --repro <target>, e.g. --repro W (5070) or R (rental).
  Projection only (counts and hours, no files): --project.

Plan format: usvbench-g12-dose-plan-v3 with 'rerun_set' (so the unchanged G12/G32 dispatch.py dispatches it, one
ledger per plan) and a 'g51_campaign' block (seed blocks, controllers, n/a list, timing model sha, predictions sha). Batch
files: format usvbench-g12-batch-v3 + mode=FINAL, seed_namespace/seed_block/run_purpose (+ PPO fields), grid 64 x 2, level
0, eval_seed = the block's seed. One seed block per batch; a dose ladder (D1..D4 of one suite) is never split; the calm and
the stack cell lead batch 1 of their (task, vessel, controller, seed block); batches are cut to <= --batch-hours expected
on a rental slot (K = 2). Affinity id = the batch id (any host of the right class may take any batch; the orchestrator keeps
each table block - task x vessel x controller - on ONE machine class, and PPO batches on the host that trained the run).
Named timeouts CELLMS_<task>_<vessel>_<CL|MP|PPO>_<CALM|SUITE>_<host class> = ceil_min(max(3 x high estimate, 30 min)) of
the timing model; expected per-slot seconds = the model's central value for that class at its operational K (RENTAL 2,
5070 3, 4080 2). Never overwrites.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign_cells as C  # noqa: E402
P = C.P
HOSTS = ("4080", "5070", "RENTAL")
CONSTANTS = {"EXITWAITMS": 30000, "GPUMEMWAITMS": 600000, "PYEXITWAITMS": 120000, "VERIFYMS": 600000}   # = executor values
TIMEOUT_FACTOR, TIMEOUT_FLOOR_S = 3.0, 1800.0


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest().upper()


def ceil_min_ms(seconds):
    return int(math.ceil(seconds / 60.0)) * 60000


class Timing(object):
    """Central / high per-slot hours of one 64 x 2 cell per host class, from timing_model_g51.json (tools/make_timing_g51.py)."""

    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        self.sha = sha_bytes(self.raw)
        self.doc = json.loads(self.raw.decode("ascii"))
        self.table = self.doc["cells"]

    def key(self, cell):
        return "%s_%s_%s" % (P.TASK_CODE[cell["task"]], cell["vkey"].upper(), C.CTRL_CODE[cell["ctrl"]])

    def hours(self, cell, host, which="mid"):
        entry = self.table[self.key(cell)]
        if self.doc.get("format") == "g51-timing-v1":            # per-condition table (tools/make_timing_g51.py)
            return entry[cell["cond"]][host][{"mid": 0, "low": 1, "high": 2}[which]]
        entry = entry[host]                                        # stub format (tests): calm / suite
        kind = "calm" if cell["kind"] == "calm" else "suite"
        return entry[kind][which]


def unit_key(cell):
    return "base" if cell["kind"] in ("calm", "allsuites") else cell["axis"]


def cut(units, hours_of, limit_h):
    """Greedy, order-preserving: a unit (ladder) is never split; a batch closes when the next unit would pass limit_h."""
    groups, current, load = [], [], 0.0
    for unit in units:
        h = sum(hours_of(c) for c in unit)
        if current and load + h > limit_h:
            groups.append(current)
            current, load = [], 0.0
        current.extend(unit)
        load += h
    if current:
        groups.append(current)
    return groups


def build(args, timing, cells, names, run_cell_sha, na, predictions):
    for order, cell in enumerate(cells, start=1):
        cell["order"] = order
    timeouts, reasons = {}, {}
    for cell in cells:
        cell["timeout_name"], cell["exp_s"] = {}, {}
        for host in HOSTS:
            kind = "CALM" if cell["kind"] == "calm" else "SUITE"
            tname = "CELLMS_%s_%s_%s_%s_%s" % (P.TASK_CODE[cell["task"]], cell["vkey"].upper(), C.CTRL_CODE[cell["ctrl"]], kind, host)
            hi_s = 3600.0 * timing.hours(cell, host, "high")          # the name's value = the max over its cells
            ms = ceil_min_ms(max(TIMEOUT_FACTOR * hi_s, TIMEOUT_FLOOR_S))
            if tname not in timeouts or timeouts[tname] < ms:
                timeouts[tname] = ms
                reasons[tname] = ("G51 campaign: ceil_min(max(%.1f x high estimate %.0f s, %.0f s)) of timing model %s (64 x 2, %s per-slot "
                                  "at its operational K)" % (TIMEOUT_FACTOR, hi_s, TIMEOUT_FLOOR_S, timing.sha[:16], host))
            cell["timeout_name"][host] = tname
            cell["exp_s"][host] = round(3600.0 * timing.hours(cell, host, "mid"), 1)
    batches = []
    blocks = list(dict.fromkeys(c["block"] for c in cells))
    for block in blocks:
        mine = [c for c in cells if c["block"] == block]
        units = []
        for cell in mine:
            key = unit_key(cell)
            match = [u for u in units if u[0] == key]
            if match:
                match[0][1].append(cell)
            else:
                units.append((key, [cell]))
        units.sort(key=lambda u: (u[0] != "base", min(c["order"] for c in u[1])))
        groups = cut([u[1] for u in units], lambda c: c["exp_s"]["RENTAL"] / 3600.0, args.batch_hours)
        for index, group in enumerate(groups, start=1):
            group.sort(key=lambda c: c["order"])
            bid = "%s-%s-b%d" % (args.name, block, index)
            if len(bid) > 60:
                raise SystemExit("batch id too long: %s" % bid)
            entry = {"id": bid, "affinity": bid, "block": block, "table_block": group[0]["table_block"], "index": index,
                     "of": len(groups), "cells": [c["name"] for c in group], "per_host": {}, "seed_block": group[0]["seed_block"],
                     "ctrl": group[0]["ctrl"], "vkey": group[0]["vkey"], "task": group[0]["task"], "ppo_seed": group[0]["ppo_seed"]}
            for host in HOSTS:
                exp = sum(c["exp_s"][host] for c in group)
                worst = (sum(timeouts[c["timeout_name"][host]] for c in group)
                         + len(group) * (CONSTANTS["EXITWAITMS"] + 2 * CONSTANTS["PYEXITWAITMS"] + CONSTANTS["GPUMEMWAITMS"])
                         + 2 * CONSTANTS["VERIFYMS"])
                entry["per_host"][host] = {"expected_two_slot_h": exp / 3600.0, "worst_case_h": worst / 3600000.0,
                                           "ceiling_ms": int(math.ceil(worst / 3600000.0)) * 3600000}
            for c in group:
                c["batch"] = bid
            batches.append(entry)
    ctrls = sorted({c["ctrl"] for c in cells})
    plan = {"format": "usvbench-g12-dose-plan-v3", "hosts": list(HOSTS),
            "counts": {"total": len(cells), "classical": sum(1 for c in cells if c["ctrl"] == "cl"),
                       "mppi": sum(1 for c in cells if c["ctrl"] == "mp"), "ppo": sum(1 for c in cells if c["ctrl"] == "ppo"),
                       "blocks": len(blocks), "batches": len(batches)},
            "rerun_set": {"name": args.name, "selectors": ["G51 campaign plan tools/make_campaign_plan.py"], "release_id": args.release_id,
                          "release_root": str(args.release_root), "run_cell_sha256": run_cell_sha, "controller_names": names,
                          "release_generation": P.names_generation(names), "source_plan_json": "G51 campaign_cells.py",
                          "source_plan_json_sha256": sha_bytes((HERE / "campaign_cells.py").read_bytes())},
            "g51_campaign": {"mode": "FINAL", "grid": [64, 2], "level": 0, "seed_blocks": [list(x) for x in args.blocks],
                             "controllers": ctrls, "ppo_run": args.ppo_run_spec, "repro": args.repro,
                             "not_applicable": [list(r) for r in C.not_applicable_list(na)],
                             "timing_model_sha256": timing.sha, "predictions_sha256": sha_bytes(predictions["raw"]),
                             "batch_hours": args.batch_hours, "timeout_rule": "ceil_min(max(%.1f x high, %d s))" % (TIMEOUT_FACTOR, TIMEOUT_FLOOR_S)},
            "timeouts_ms": dict(sorted(timeouts.items())), "timeout_reasons": dict(sorted(reasons.items())), "constants": CONSTANTS,
            "batches": batches,
            "cells": [{"name": c["name"], "order": c["order"], "block": c["block"], "batch": c["batch"], "kind": c["kind"],
                       "axis": c["axis"], "dose": c["dose"], "tokens": list(c["tokens"]), "timeout_name": c["timeout_name"],
                       "expected_two_slot_s": c["exp_s"], "expected_controller": c["expected_controller"], "task": c["task"],
                       "vkey": c["vkey"], "ctrl": c["ctrl"], "ppo_seed": c["ppo_seed"], "seed_block": c["seed_block"],
                       "seed": c["seed"], "cond": c["cond"], "stratum": c["stratum"], "table_block": c["table_block"],
                       "jid": c["jid"], "carrier": c["carrier"], "vessel": c["vessel"]} for c in cells]}
    return plan


def batch_lines(plan, plan_hash, batch, cells, predictions, ppo):
    rs = plan["rerun_set"]
    seed = cells[0]["seed"]
    lines = ["format=usvbench-g12-batch-v3", "batch_id=%s" % batch["id"], "affinity=%s" % batch["affinity"], "mode=FINAL",
             "num_envs=64", "episodes_per_env=2", "eval_seed=%d" % seed, "level=0", "seed_namespace=final_test",
             "seed_block=%s" % batch["seed_block"], "run_purpose=final_test"]
    if ppo:
        lines += ["ppo_training_seed=%d" % ppo["seed"], "ppo_checkpoint=%s" % ppo["checkpoint"],
                  "ppo_checkpoint_sha256=%s" % ppo["checkpoint_sha256"].upper(), "gate_ledger=%s" % ppo["ledger"],
                  "gate_ledger_sha256=%s" % ppo["ledger_sha256"].upper()]
    lines += ["plan_json_sha256=%s" % plan_hash, "release_id=%s" % rs["release_id"], "controller_names_sha256=%s" % rs["run_cell_sha256"]]
    lines += ["PREDICTION " + p for p in predictions["lines"]]
    lines += ["ceiling|%s|%d" % (h, batch["per_host"][h]["ceiling_ms"]) for h in HOSTS]
    for name in sorted({c["timeout_name"][h] for c in cells for h in HOSTS}):
        lines.append("timeout|%s|%d|%s" % (name, plan["timeouts_ms"][name], plan["timeout_reasons"][name]))
    for c in cells:
        lines.append("|".join(["cell", str(c["order"]), c["name"], c["block"], c["jid"], c["vessel"], c["carrier"], c["ctrl"], c["kind"],
                               "" if c["axis"] is None else str(c["axis"]), "" if c["dose"] is None else str(c["dose"]), ",".join("%s=%s" % (h, c["timeout_name"][h]) for h in HOSTS),
                               c["expected_controller"], " ".join(c["tokens"])]))
    for line in lines:
        line.encode("ascii")
        if re.search(r"[^\x20-\x7E]", line):
            raise SystemExit("non-printable text in a batch line: %r" % line[:80])
    return lines


def read_predictions(path):
    raw = Path(path).read_bytes()
    lines = [l[len("PRED "):] for l in raw.decode("ascii").splitlines() if l.startswith("PRED ")]
    if not lines:
        raise SystemExit("no 'PRED ' lines in %s" % path)
    return {"raw": raw, "lines": lines}


def summary(plan, timing):
    lines = ["G51 campaign plan %s: release %s, timing model %s, predictions %s" % (
        plan["rerun_set"]["name"], plan["rerun_set"]["release_id"], timing.sha[:16], plan["g51_campaign"]["predictions_sha256"][:16]),
        "cells: %s" % json.dumps(plan["counts"], sort_keys=True),
        "not applicable by design: %s" % "; ".join("%s %s %s D%s" % tuple(r[:4]) for r in plan["g51_campaign"]["not_applicable"])]
    by = {}
    for c in plan["cells"]:
        k = (c["ctrl"], c["vkey"])
        d = by.setdefault(k, {"n": 0, "RENTAL": 0.0, "5070": 0.0, "4080": 0.0})
        d["n"] += 1
        for h in HOSTS:
            d[h] += c["expected_two_slot_s"][h] / 3600.0
    lines.append("expected slot-hours (per slot at the class's operational K: RENTAL K=2, 5070 K=3, 4080 K=2):")
    for (ctrl, vkey), d in sorted(by.items()):
        lines.append("  %-4s %-5s cells %5d  RENTAL %8.1f  5070 %8.1f  4080 %8.1f" % (ctrl, vkey, d["n"], d["RENTAL"], d["5070"], d["4080"]))
    tot = {h: sum(d[h] for d in by.values()) for h in HOSTS}
    lines.append("  all        cells %5d  RENTAL %8.1f  5070 %8.1f  4080 %8.1f" % (len(plan["cells"]), tot["RENTAL"], tot["5070"], tot["4080"]))
    lines.append("batches: %d (<= %.1f h expected on a rental slot, ladders never split)" % (len(plan["batches"]), plan["g51_campaign"]["batch_hours"]))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--release-id", required=True)
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--controllers", default="cl,mp")
    ap.add_argument("--seed-blocks", default="final-01,final-02,final-03,final-04")
    ap.add_argument("--batch-hours", type=float, default=6.0)
    ap.add_argument("--ppo-run")
    ap.add_argument("--ppo-checkpoint")
    ap.add_argument("--ppo-checkpoint-sha256")
    ap.add_argument("--gate-ledger")
    ap.add_argument("--gate-ledger-sha256")
    ap.add_argument("--repro")
    ap.add_argument("--project", action="store_true")
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-text", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9-]{1,24}", args.name):
        raise SystemExit("name: 1-24 of letters, digits, '-'")
    if not re.fullmatch(r"[0-9a-f]{12}-[0-9a-f]{12}", args.release_id):
        raise SystemExit("release id must look like <12 hex>-<12 hex>")
    wanted = args.seed_blocks.split(",")
    args.blocks = [b for b in C.SEED_BLOCKS if b[0] in wanted]
    if len(args.blocks) != len(wanted):
        raise SystemExit("unknown seed block in %s" % args.seed_blocks)
    ctrls = tuple(args.controllers.split(","))
    if not set(ctrls) <= {"cl", "mp", "ppo"}:
        raise SystemExit("controllers: cl, mp, ppo")
    timing = Timing(args.timing)
    predictions = read_predictions(args.predictions)
    names, run_cell_sha = P.release_controller_names(args.release_root)
    ppo = None
    args.ppo_run_spec = None
    ppo_seeds = C.PPO_SEEDS
    cells, na = C.all_cells(args.release_root, names, controllers=ctrls, seed_blocks=args.blocks, ppo_seeds=ppo_seeds)
    if args.ppo_run:
        if ctrls != ("ppo",):
            raise SystemExit("--ppo-run needs --controllers ppo")
        task, vkey, seed = args.ppo_run.split(":")
        seed = int(seed)
        if task not in P.TASKS or vkey not in ("bb", "wamv") or seed not in C.PPO_SEEDS:
            raise SystemExit("bad --ppo-run %s" % args.ppo_run)
        cells = [c for c in cells if c["task"] == task and c["vkey"] == vkey and c["ppo_seed"] == seed]
        if not args.project:
            for key in ("ppo_checkpoint", "ppo_checkpoint_sha256", "gate_ledger", "gate_ledger_sha256"):
                if not getattr(args, key):
                    raise SystemExit("--ppo-run needs --%s" % key.replace("_", "-"))
            ppo = {"seed": seed, "checkpoint": args.ppo_checkpoint, "checkpoint_sha256": args.ppo_checkpoint_sha256,
                   "ledger": args.gate_ledger, "ledger_sha256": args.gate_ledger_sha256}
        args.ppo_run_spec = args.ppo_run
    elif "ppo" in ctrls and not args.project:
        raise SystemExit("PPO batches are generated per delivered run (--ppo-run ...); use --project for PPO counts")
    if args.repro:
        if not re.fullmatch(r"[A-Z0-9]{1,6}", args.repro):
            raise SystemExit("--repro target: 1-6 of A-Z0-9 (e.g. W = 5070, R = rental, F = 4080)")
        cells = C.repro_dup_cells(cells, controllers=tuple(c for c in ctrls if c != "ppo") or ("ppo",))
        for c in cells:
            c["name"] = "xm%s_%s" % (args.repro.lower(), c["name"])
            c["block"] = "XM%s_%s" % (args.repro, c["block"])
            c["stratum"] = "REPRO"
    if not cells:
        raise SystemExit("no cell selected")
    plan = build(args, timing, cells, names, run_cell_sha, na, predictions)
    text = json.dumps(plan, indent=1, sort_keys=True) + "\n"
    lines = summary(plan, timing)
    if args.project:
        print("\n".join(lines))
        print("PROJECTION-ONLY nothing written")
        return 0
    for path in (args.out_json, args.out_text, args.out_dir):
        if path is None:
            raise SystemExit("give --out-json, --out-text and --out-dir (or --project)")
        if path.exists():
            raise SystemExit("refusing to overwrite %s" % path)
    args.out_json.write_bytes(text.encode("ascii"))
    plan_hash = sha_bytes(text.encode("ascii"))
    args.out_dir.mkdir(parents=True)
    by_name = {c["name"]: c for c in cells}
    list_lines = ["format=usvbench-g12-planlist-v1"]
    for batch in plan["batches"]:
        group = [by_name[n] for n in batch["cells"]]
        body = ("\n".join(batch_lines(plan, plan_hash, batch, group, predictions, ppo)) + "\n").encode("ascii")
        path = args.out_dir / ("%s.plan" % batch["id"])
        with open(path, "xb") as handle:
            handle.write(body)
        list_lines.append("batch|%s|%s|1|1" % (path.name, sha_bytes(body)))
    with open(args.out_dir / "all_batches.list", "xb") as handle:
        handle.write(("\n".join(list_lines) + "\n").encode("ascii"))
    args.out_text.write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    print("\n".join(lines))
    print("CAMPAIGN-PLAN %s plan_sha256=%s batches=%d cells=%d" % (args.name, plan_hash, len(plan["batches"]), len(cells)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
