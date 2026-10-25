#!/usr/bin/env python3
"""G51: campaign schedule simulation (stdlib, deterministic seeds) with THE dispatch policy the orchestrator uses
(policy_g51.py). Predicts makespan with uncertainty for a host fleet.

usage: python -B sim_campaign.py --release-root <extract> --timing timing/timing_model_g51.json [--draws 300]
           [--seed-blocks final-01,...] [--rentals 12] [--rental-online-h 0,0,0,0,0,0,24,24,24,24,24,24] [--w5070 yes|no]
           [--out <json>]
Model:
  * jobs = the campaign batches (cl, mp from make_campaign_plan.build; PPO eval batches per run, released when that run's
    training + gate finish, bound to the training host), PPO training runs (72: 24 task-vessels x seeds 42/1001/2003);
  * a job's true duration = expected(host class) x f_table x f_job: f_table ~ triangular(low/mid, 1, high/mid) of the timing
    model, drawn once per (task, vessel, controller) and draw (correlated errors), f_job ~ triangular(0.85, 1, 1.2);
  * training: BlueBoat 2.8 h [1.95, 2.92] (measured G46, rental K=2, inline evaluations), WAM-V 6.0 h [4.5, 9.0] (UNVERIFIED:
    2 x budget + 51 WAM-V evaluations of 555-1556 s at 3 in parallel) + gate evaluation (one 64 x 2 PPO cell, calm);
  * hosts: rentals K=2 (online at the given hours), the 5070 K=3 (only when --w5070 yes: needs the Windows FINAL executor
    port, NOT built) taking classical and MPPI-BlueBoat table blocks only (WAM-V MPPI and PPO digests differ on the 5070).
Policy (policy_g51.choose): see policy_g51.py.
"""
import argparse
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign_cells as C  # noqa: E402
import make_campaign_plan as M  # noqa: E402
import policy_g51 as POL  # noqa: E402
P = C.P


CENTRAL = [False]


def tri(rng, lo, mode, hi):
    if CENTRAL[0]:
        return mode
    lo, hi = min(lo, mode), max(hi, mode)
    return mode if hi <= lo else rng.triangular(lo, hi, mode)


def build_jobs(args, timing):
    names, sha = P.release_controller_names(args.release_root)
    blocks = [b for b in C.SEED_BLOCKS if b[0] in args.seed_blocks.split(",")]
    ns = argparse.Namespace(name="sim", release_id="0" * 12 + "-" + "0" * 12, release_root=args.release_root, blocks=blocks,
                            batch_hours=args.batch_hours, ppo_run_spec=None, repro=None)
    pred = {"raw": b"sim", "lines": ["sim"]}
    jobs = []
    cells, na = C.all_cells(args.release_root, names, controllers=("cl", "mp", "ppo"), seed_blocks=blocks)
    plan = M.build(ns, timing, cells, names, sha, na, pred)
    by_name = {c["name"]: c for c in plan["cells"]}
    for b in plan["batches"]:
        c0 = by_name[b["cells"][0]]
        rng_key = "%s_%s_%s" % (c0["task"], c0["vkey"], c0["ctrl"])
        lo = sum(timing.hours(dict(by_name[n], kind=by_name[n]["kind"]), "RENTAL", "low") for n in b["cells"])
        hi = sum(timing.hours(dict(by_name[n], kind=by_name[n]["kind"]), "RENTAL", "high") for n in b["cells"])
        mid = b["per_host"]["RENTAL"]["expected_two_slot_h"]
        jobs.append({"id": b["id"], "kind": "ppo_eval" if c0["ctrl"] == "ppo" else "eval", "table": c0["table_block"],
                     "task": c0["task"], "vkey": c0["vkey"], "ctrl": c0["ctrl"], "ppo_seed": c0["ppo_seed"],
                     "run": "%s_%s_%s" % (P.TASK_CODE[c0["task"]], c0["vkey"].upper(), c0["ppo_seed"]) if c0["ctrl"] == "ppo" else None,
                     "h": {h: b["per_host"][h]["expected_two_slot_h"] for h in M.HOSTS}, "lo": lo / mid if mid else 1, "hi": hi / mid if mid else 1,
                     "rkey": rng_key, "cells": len(b["cells"])})
    for task in P.TASKS:
        for vkey in ("bb", "wamv"):
            gate = timing.hours({"task": task, "vkey": vkey, "ctrl": "ppo", "cond": "calm", "kind": "calm"}, "RENTAL", "mid")
            for seed in C.PPO_SEEDS:
                mid, lo, hi = (2.8, 1.95, 2.92) if vkey == "bb" else (6.0, 4.5, 9.0)
                jobs.append({"id": "train_%s_%s_%d" % (P.TASK_CODE[task], vkey.upper(), seed), "kind": "train",
                             "table": "%s_%s_P%d" % (P.TASK_CODE[task], vkey.upper(), seed), "task": task, "vkey": vkey, "ctrl": "ppo",
                             "ppo_seed": seed, "run": "%s_%s_%d" % (P.TASK_CODE[task], vkey.upper(), seed),
                             "h": {"RENTAL": mid + gate, "4080": (mid + gate) * 1.0, "5070": (mid + gate) * 1.94}, "lo": lo / mid, "hi": hi / mid,
                             "rkey": "train_%s" % vkey, "cells": 0})
    return jobs


def simulate(jobs, hosts, rng, w5070):
    ftab = {}
    for j in jobs:
        if j["rkey"] not in ftab:
            ftab[j["rkey"]] = tri(rng, j["lo"], 1.0, j["hi"])
    true = {j["id"]: ftab[j["rkey"]] * tri(rng, 0.85, 1.0, 1.2) for j in jobs}
    state = POL.State(jobs)
    slots = []
    for h in hosts:
        for s in range(h["K"]):
            slots.append({"host": h["key"], "cls": h["cls"], "free_at": h["online_h"], "busy": 0.0})
    now, events, idle_nothing = 0.0, [], 0.0
    import heapq
    heap = [(s["free_at"], i) for i, s in enumerate(slots)]
    heapq.heapify(heap)
    finish = 0.0
    pending_release = []          # (time, run, host)
    while heap:
        t, i = heapq.heappop(heap)
        while pending_release and pending_release[0][0] <= t:
            _, run, host = heapq.heappop(pending_release)
            state.release_run(run, host)
        slot = slots[i]
        if slot["cls"] == "5070" and not w5070:
            continue
        job = POL.choose(state, slot["host"], slot["cls"])
        if job is None:
            if state.remaining() == 0 and not pending_release:
                continue
            nxt = pending_release[0][0] if pending_release else None
            if nxt is None:
                continue                  # nothing will ever come for this slot (class lock / binding)
            idle_nothing += nxt - t
            heapq.heappush(heap, (nxt, i))
            continue
        d = job["h"][slot["cls"]] * true[job["id"]]
        end = t + d
        slot["busy"] += d
        finish = max(finish, end)
        if job["kind"] == "train":
            heapq.heappush(pending_release, (end, job["run"], slot["host"]))
        heapq.heappush(heap, (end, i))
    left = state.remaining()
    online = sum(max(0.0, finish - s["free_at"]) for s in slots if s["cls"] != "5070" or w5070)
    busy = sum(s["busy"] for s in slots)
    return {"makespan_h": finish, "unassigned_jobs": left, "slot_h_online": online, "slot_h_busy": busy, "idle_waiting_h": idle_nothing}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed-blocks", default="final-01,final-02,final-03,final-04")
    ap.add_argument("--rentals", type=int, default=12)
    ap.add_argument("--rental-online-h", default="")
    ap.add_argument("--w5070", default="no")
    ap.add_argument("--batch-hours", type=float, default=6.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--central", action="store_true", help="every duration = the model's central value (one draw)")
    args = ap.parse_args(argv)
    CENTRAL[0] = args.central
    if args.central:
        args.draws = 1
    timing = M.Timing(args.timing)
    jobs = build_jobs(args, timing)
    online = [float(x) for x in args.rental_online_h.split(",")] if args.rental_online_h else [0.0] * args.rentals
    hosts = [{"key": "rental%02d" % (i + 1), "cls": "RENTAL", "K": 2, "online_h": online[i] if i < len(online) else 0.0} for i in range(args.rentals)]
    hosts.append({"key": "5070", "cls": "5070", "K": 3, "online_h": 0.0})
    res = []
    for d in range(args.draws):
        rng = random.Random(20261014 * 1000 + d)
        res.append(simulate(jobs, hosts, rng, args.w5070 == "yes"))
    ms = sorted(r["makespan_h"] for r in res)
    q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]
    tot = {k: sum(j["h"]["RENTAL"] for j in jobs if j["kind"] == k) for k in ("eval", "ppo_eval", "train")}
    out = {"seed_blocks": args.seed_blocks, "rentals": args.rentals, "online_h": online, "w5070": args.w5070, "draws": args.draws,
           "jobs": len(jobs), "expected_rental_slot_h": tot, "makespan_h": {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9)},
           "makespan_days": {"p10": q(0.1) / 24, "p50": q(0.5) / 24, "p90": q(0.9) / 24},
           "busy_fraction_mean": sum(r["slot_h_busy"] / r["slot_h_online"] for r in res) / len(res),
           "idle_waiting_slot_h_mean": sum(r["idle_waiting_h"] for r in res) / len(res),
           "unassigned_max": max(r["unassigned_jobs"] for r in res)}
    text = json.dumps(out, indent=1, sort_keys=True)
    print(text)
    if args.out:
        if args.out.exists():
            raise SystemExit("refusing to overwrite %s" % args.out)
        args.out.write_bytes((text + "\n").encode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
