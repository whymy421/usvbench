#!/usr/bin/env python3
"""G12 dispatch helper (Python stdlib, CPU only): assign plan v3 batches to a host slot and write that slot's plan list.

usage (one slot):
  python -B tools/dispatch.py --host 4080 --slots 2 --slot 0 --batch smoke-v3-RE_BB_CL-b1 --batch smoke-v3-RE_BB_MP-b1 ...
  python -B tools/dispatch.py --host rental01 --host-class RENTAL --slots 2 --slot 1 --batch ...
usage (assignment file, host -> slot -> batch ids; every slot of every host in one all-or-nothing step):
  python -B tools/dispatch.py --assignment assign.json
      {"hosts": {"4080": {"host_class": "4080", "slots": 2, "slot_batches": {"0": ["id", ...], "1": ["id", ...]}},
                 "rental01": {"host_class": "RENTAL", "slots": 2, "slot_batches": {"0": [...]}}}}
usage (re-run on the SAME host after an abort: next attempt, resume from the ABORTED marker's resume_from_order):
  python -B tools/dispatch.py --host 4080 --slots 2 --slot 1 --batch smoke-v3-RE_BB_CL-b1 --reattempt smoke-v3-RE_BB_CL-b1:17
usage (state): python -B tools/dispatch.py --status
Common options: --plan (default plan/dose_plan_v3.json), --batch-dir (default plan/batches_v3; the batch files, and the
folder the list is written to, since list entries are names relative to the list), --ledger (default
dispatch/ledger.jsonl), --dry-run (check and print, write nothing).

--host is the host identity the affinity rule is about (4080, 5070, or a rental name such as rental01); --host-class is
the timing class (4080, 5070, RENTAL; defaults to --host when that is 4080 or 5070). Output per slot: a new plan list
plan/batches_v3/dispatch_<seq>_<host>_k<K>s<slot>.list (format usvbench-g12-planlist-v1: 'batch|<file>|<sha256>|<attempt>|
<from_order>' lines, in the given order), its sha256 (the -PlanListSha256 of the launch), cells, the expected hours on
that host class and the worst case. Expected hours: K = 2 the plan's two-slot expected; K = 1 that / 1.09 (G8 P4 slot
factor); K >= 3 UNMEASURED (the two-slot value is printed as a lower bound).

Ledger: append-only JSON lines, one per assigned batch, each carrying prev_sha256 = SHA-256 of the ledger bytes before
it (a chain; an edited or truncated ledger is refused). A lock file <ledger>.lock (exclusive create) serialises runs.
Refusals (nothing is written when any refusal fires):
  BATCH-ALREADY-ASSIGNED     the batch is in the ledger (a re-run needs --reattempt on the same host)
  AFFINITY-ON-OTHER-HOST     a batch of the same affinity id (block) is assigned to another host, in the ledger or in this run
  plus unknown batch, batch twice in one run, bad host name / class / K / slot, host class differing from the ledger's,
  attempt above 9, batch file digest or content not matching the plan, ledger chain broken, lock held.

G32 v2.6 COPY (work_v26/tools/dispatch.py; the original tools/dispatch.py and its ledger are untouched):
  * defaults: --plan plan/rerun_v26r2.json, --batch-dir plan/rerun_v26r2 (the v2.6 full smoke: batch-v3 files with
    the v2.6 release names and G24b's timeouts), --ledger dispatch/ledger.jsonl inside work_v26 (a new ledger);
  * K-CAP refusal: host class 4080 at most K = 2, 5070 at most K = 3 (controller rule for G24b's timeouts =
    measured x 1.5 x 2); RENTAL keeps 1..4;
  * V2-BATCH refusal: only plans with 'rerun_set' (batch-v3 files) are dispatched, because the work_v26 executor runs
    only batch-v3 batches;
  * the ledger lock is an OS byte-range lock (msvcrt.locking / fcntl.lockf) on <ledger>.lock, which is created once
    and never deleted (no file is removed by this tool); a lock held by another process refuses with LEDGER-LOCKED,
    a crashed run leaves no stale lock.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parent.parent
HOST_CLASSES = ("4080", "5070", "RENTAL")
SLOT_FACTOR = 1.09          # G8 P4 two-slot slowdown, as in tools/plan_dose_smoke.py SLOT_EXPECTED
MAX_ATTEMPT = 9             # executor plan list regex: attempt [1-9]
ZERO = "0" * 64
HOST_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_SLOTS = {"4080": 2, "5070": 3, "RENTAL": 4}     # G32 v2.6 K cap


def lock_ledger(lock):
    """G32: exclusive OS lock on the persistent lock file; returns the open handle (never deletes the file)."""
    handle = open(lock, "a+b")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 0)
    except OSError:
        handle.close()
        raise Refusal("LEDGER-LOCKED %s is locked by another dispatch run" % lock)
    return handle


def unlock_ledger(handle):
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.lockf(handle.fileno(), fcntl.LOCK_UN, 1, 0)
    finally:
        handle.close()


class Refusal(Exception):
    pass


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest().upper()


def load_ledger(path):
    """Return the ledger records after checking the prev_sha256 chain."""
    if not path.exists():
        return [], b""
    data = path.read_bytes()
    records, offset = [], 0
    for raw in data.split(b"\n"):
        if not raw:
            offset += 1
            continue
        record = json.loads(raw.decode("ascii"))
        want = sha_bytes(data[:offset]) if offset else ZERO
        if record.get("prev_sha256") != want:
            raise Refusal("LEDGER-CHAIN-BROKEN at line %d of %s (edited or truncated ledger)" % (len(records) + 1, path))
        records.append(record)
        offset += len(raw) + 1
    if data and not data.endswith(b"\n"):
        raise Refusal("LEDGER-CHAIN-BROKEN: %s does not end with a newline" % path)
    return records, data


def ledger_state(records):
    batches, affinity_host, host_class = {}, {}, {}
    for r in records:
        batches[r["batch_id"]] = r
        affinity_host.setdefault(r["affinity"], r["host"])
        host_class.setdefault(r["host"], r["host_class"])
    return batches, affinity_host, host_class


def expected_hours(two_slot_h, slots):
    if slots == 1:
        return two_slot_h / SLOT_FACTOR, "single-slot estimate (two-slot / %.2f)" % SLOT_FACTOR
    if slots == 2:
        return two_slot_h, "two-slot expected"
    return two_slot_h, "UNMEASURED at K=%d (two-slot value shown, a lower bound)" % slots


def requests_from_args(args):
    if args.assignment:
        doc = json.loads(Path(args.assignment).read_text(encoding="ascii"))
        out = []
        for host, spec in doc.get("hosts", {}).items():
            for slot, ids in sorted(spec.get("slot_batches", {}).items(), key=lambda kv: int(kv[0])):
                out.append({"host": host, "host_class": spec.get("host_class") or (host if host in ("4080", "5070") else None),
                            "slots": int(spec.get("slots", 0)), "slot": int(slot), "batches": list(ids)})
        if not out:
            raise Refusal("assignment file names no host slot")
        return out
    if args.host is None or args.slots is None or args.slot is None or not args.batch:
        raise Refusal("give --host, --slots, --slot and one or more --batch (or --assignment, or --status)")
    return [{"host": args.host, "host_class": args.host_class or (args.host if args.host in ("4080", "5070") else None),
             "slots": args.slots, "slot": args.slot, "batches": list(args.batch)}]


def plan_and_batches(plan_path, batch_dir):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes.decode("ascii"))
    if plan.get("format") != "usvbench-g12-dose-plan-v3":
        raise Refusal("plan v3 expected: %s" % plan_path)
    return plan, sha_bytes(plan_bytes), {b["id"]: b for b in plan["batches"]}


def check_batch_file(batch_dir, batch, plan_hash, plan=None):
    path = batch_dir / ("%s.plan" % batch["id"])
    if not path.exists():
        raise Refusal("batch file missing: %s" % path)
    data = path.read_bytes()
    lines = data.decode("ascii").splitlines()
    fmt = "format=usvbench-g12-batch-v3" if (plan or {}).get("rerun_set") else "format=usvbench-g12-batch-v2"   # update 5: re-run plans
    for want in (fmt, "batch_id=%s" % batch["id"], "affinity=%s" % batch["affinity"],
                 "plan_json_sha256=%s" % plan_hash):
        if want not in lines:
            raise Refusal("batch file %s lacks '%s' (not of this plan)" % (path.name, want))
    if sum(1 for l in lines if l.startswith("cell|")) != len(batch["cells"]):
        raise Refusal("batch file %s cell count differs from the plan" % path.name)
    return path, sha_bytes(data)


def plan_dispatch(args, plan, plan_hash, by_id, records):
    batches_done, affinity_host, ledger_class = ledger_state(records)
    reattempt = {}
    for item in args.reattempt or []:
        bid, _, frm = item.partition(":")
        reattempt[bid] = int(frm) if frm else 1
    requests = requests_from_args(args)
    seen, run_affinity, out = set(), {}, []
    for req in requests:
        host, cls, slots, slot = req["host"], req["host_class"], req["slots"], req["slot"]
        if not HOST_RE.match(host or ""):
            raise Refusal("bad host name %r (letters, digits, _ and -, at most 32)" % host)
        if cls not in HOST_CLASSES:
            raise Refusal("host %s: host class must be one of %s (give --host-class)" % (host, ", ".join(HOST_CLASSES)))
        if host in ("4080", "5070") and cls != host:
            raise Refusal("host %s must use host class %s" % (host, host))
        if ledger_class.get(host, cls) != cls:
            raise Refusal("host %s is class %s in the ledger, not %s" % (host, ledger_class[host], cls))
        if not 1 <= slots <= 4 or not 0 <= slot < slots:
            raise Refusal("host %s: K must be 1..4 and slot 0..K-1 (got K=%s slot=%s)" % (host, slots, slot))
        if slots > MAX_SLOTS[cls]:
            raise Refusal("K-CAP host %s (class %s): at most K = %d slots with the v2.6 timeouts (G24b measured x 1.5 x 2), got K=%d"
                          % (host, cls, MAX_SLOTS[cls], slots))
        if not req["batches"]:
            raise Refusal("host %s slot %d: no batch" % (host, slot))
        entries = []
        for bid in req["batches"]:
            batch = by_id.get(bid)
            if batch is None:
                raise Refusal("unknown batch %s (not in plan v3)" % bid)
            if bid in seen:
                raise Refusal("batch %s listed twice in this run" % bid)
            seen.add(bid)
            aff = batch["affinity"]
            owner = affinity_host.get(aff)
            if owner is not None and owner != host:
                raise Refusal("AFFINITY-ON-OTHER-HOST batch %s: affinity %s is assigned to host %s, not %s" % (bid, aff, owner, host))
            if run_affinity.setdefault(aff, host) != host:
                raise Refusal("AFFINITY-ON-OTHER-HOST batch %s: affinity %s goes to host %s earlier in this run, not %s"
                              % (bid, aff, run_affinity[aff], host))
            previous = batches_done.get(bid)
            if previous is not None and bid not in reattempt:
                raise Refusal("BATCH-ALREADY-ASSIGNED batch %s: host %s slot %s/%s attempt %d (list %s); a re-run on the same "
                              "host needs --reattempt %s:<from_order>" % (bid, previous["host"], previous["slot"], previous["slots"],
                                                                          previous["attempt"], previous["list_file"], bid))
            if bid in reattempt and previous is None:
                raise Refusal("--reattempt %s: the batch was never assigned" % bid)
            attempt = previous["attempt"] + 1 if previous else 1
            if attempt > MAX_ATTEMPT:
                raise Refusal("batch %s: attempt %d above %d" % (bid, attempt, MAX_ATTEMPT))
            from_order = reattempt.get(bid, 1)
            orders = [c["order"] for c in plan["cells"] if c["batch"] == bid]
            if from_order != 1 and from_order not in orders:
                raise Refusal("--reattempt %s:%d: order not in the batch (%d..%d)" % (bid, from_order, min(orders), max(orders)))
            path, digest = check_batch_file(args.batch_dir, batch, plan_hash, plan)
            selected = [c for c in plan["cells"] if c["batch"] == bid and c["order"] >= from_order]
            entries.append({"batch": batch, "file": path.name, "sha": digest, "attempt": attempt, "from_order": from_order,
                            "cells": len(selected), "expected_s": sum(c["expected_two_slot_s"][cls] for c in selected),
                            "full_cells": len(batch["cells"])})
        out.append(dict(req, entries=entries))
    unused = [b for b in reattempt if b not in seen]
    if unused:
        raise Refusal("--reattempt names batches not dispatched in this run: %s" % ", ".join(unused))
    return out


def run(args):
    plan, plan_hash, by_id = plan_and_batches(args.plan, args.batch_dir)
    if args.status:
        records, _ = load_ledger(args.ledger)
        return status(plan, records)
    if not plan.get("rerun_set"):
        raise Refusal("V2-BATCH %s has no rerun_set: the G32 v2.6 executor runs only batch-v3 batches (release names); "
                      "make the plan with work_v26 tools/plan_rerun_set.py" % args.plan)
    lock = Path(str(args.ledger) + ".lock")
    held = None
    if not args.dry_run:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        held = lock_ledger(lock)
    try:
        records, data = load_ledger(args.ledger)
        slots_out = plan_dispatch(args, plan, plan_hash, by_id, records)
        seq = 1 + max([r["dispatch_seq"] for r in records] or [0])
        new_lines, chain = [], data
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lists = []
        for req in slots_out:
            name = "dispatch_%04d_%s_k%ds%d.list" % (seq, req["host"], req["slots"], req["slot"])
            list_path = args.batch_dir / name
            if list_path.exists():
                raise Refusal("list exists: %s" % list_path)
            text = "\n".join(["format=usvbench-g12-planlist-v1"] + ["batch|%s|%s|%d|%d" % (e["file"], e["sha"], e["attempt"], e["from_order"])
                                                                    for e in req["entries"]]) + "\n"
            lists.append((list_path, text.encode("ascii"), req))
        for list_path, body, req in lists:
            list_sha = sha_bytes(body)
            for e in req["entries"]:
                record = {"prev_sha256": sha_bytes(chain) if chain else ZERO, "dispatch_seq": seq, "time_utc": stamp,
                          "batch_id": e["batch"]["id"], "affinity": e["batch"]["affinity"], "host": req["host"],
                          "host_class": req["host_class"], "slots": req["slots"], "slot": req["slot"], "attempt": e["attempt"],
                          "from_order": e["from_order"], "batch_file": e["file"], "batch_sha256": e["sha"],
                          "list_file": list_path.name, "list_sha256": list_sha, "plan_json_sha256": plan_hash}
                line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
                new_lines.append(line)
                chain += line
        report(lists, args.dry_run)
        if args.dry_run:
            print("DRY-RUN nothing written")
            return 0
        for list_path, body, _ in lists:
            with open(list_path, "xb") as handle:
                handle.write(body)
        with open(args.ledger, "ab") as handle:
            handle.write(b"".join(new_lines))
        print("LEDGER %s appended %d records (dispatch %d)" % (args.ledger, len(new_lines), seq))
        return 0
    finally:
        if held is not None:
            unlock_ledger(held)


def report(lists, dry):
    for list_path, body, req in lists:
        exp2 = sum(e["expected_s"] for e in req["entries"]) / 3600.0
        hours, label = expected_hours(exp2, req["slots"])
        worst = sum(e["batch"]["per_host"][req["host_class"]]["worst_case_h"] for e in req["entries"])
        print("SLOT-LIST host=%s class=%s K=%d slot=%d list=%s sha256=%s batches=%d cells=%d expected_h=%.1f (%s) worst_case_h=%.1f%s"
              % (req["host"], req["host_class"], req["slots"], req["slot"], list_path.name, sha_bytes(body), len(req["entries"]),
                 sum(e["cells"] for e in req["entries"]), hours, label, worst, " [dry run]" if dry else ""))
        for e in req["entries"]:
            print("  batch|%s|%s|%d|%d cells=%d/%d expected_two_slot_h=%.2f affinity=%s"
                  % (e["file"], e["sha"], e["attempt"], e["from_order"], e["cells"], e["full_cells"], e["expected_s"] / 3600.0,
                     e["batch"]["affinity"]))


def status(plan, records):
    batches_done, affinity_host, host_class = ledger_state(records)
    print("LEDGER records=%d batches_assigned=%d of %d" % (len(records), len(batches_done), len(plan["batches"])))
    for host in sorted(host_class):
        mine = [b for b in plan["batches"] if b["id"] in batches_done and batches_done[b["id"]]["host"] == host]
        print("HOST %s class=%s batches=%d affinities=%s expected_two_slot_h=%.1f" % (
            host, host_class[host], len(mine), ",".join(sorted(a for a, h in affinity_host.items() if h == host)),
            sum(b["per_host"][host_class[host]]["expected_two_slot_h"] for b in mine)))
    free = [b for b in plan["batches"] if b["id"] not in batches_done]
    print("UNASSIGNED batches=%d expected_two_slot_h 4080=%.1f 5070=%.1f RENTAL=%.1f" % (
        len(free), *[sum(b["per_host"][h]["expected_two_slot_h"] for b in free) for h in HOST_CLASSES]))
    for b in free:
        tied = affinity_host.get(b["affinity"])
        print("  %s affinity=%s 4080_h=%.2f%s" % (b["id"], b["affinity"], b["per_host"]["4080"]["expected_two_slot_h"],
                                                (" must go to host %s" % tied) if tied else ""))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=WORK / "plan" / "rerun_v26r2.json")
    parser.add_argument("--batch-dir", type=Path, default=WORK / "plan" / "rerun_v26r2")
    parser.add_argument("--ledger", type=Path, default=WORK / "dispatch" / "ledger.jsonl")
    parser.add_argument("--host")
    parser.add_argument("--host-class", choices=HOST_CLASSES)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--batch", action="append")
    parser.add_argument("--assignment")
    parser.add_argument("--reattempt", action="append", help="<batch id>[:<from_order>] re-run on the same host, next attempt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Refusal as refusal:
        print("DISPATCH-REFUSED %s" % refusal)
        return 3


if __name__ == "__main__":
    sys.exit(main())
