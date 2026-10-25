#!/usr/bin/env python3
# campaign/orchestrator.py -- the USVBench journal campaign orchestrator, VERSION fix32.
# Copied unchanged (apart from these comment lines) from the controller PC file
# g52_orchestrator_fix32.py (G52 campaign, release v2.6.6 2c04042a7a99-64e5932b737f) on 2026-10-25;
# SHA-256 of that file at copy time: af82b707b4c3c91bb3c21ffc15694dee62fa14034b1c56fe7249a30a0ffc308a
# (drop lines 2-9 to get it back byte for byte). Its config is campaign_config_v266_final_fix32.json.
# The hard-coded Windows paths (D:/claude code/codex_tasks/...) are the controller PC's layout;
# earlier fixN versions are not in this repository (history: the campaign notes). See campaign/README.md.
# Internal codes in comments (G51, G52, FIX17, ...) are working notes, to be cleaned in a later version.
"""G52 COPY (2026-10-14) of G51 g51_orchestrator.py: only (1) TOOLS = the G51 tools folder and (2) PPO test plans get
--seed-blocks from the config key seed_blocks (owner 10-14: main test set = final-01 only). State: this folder, run/.

G51 unattended campaign orchestrator (runs on the controller PC; Python 3 stdlib; no LLM; BelowNormal priority).

usage:
  python -B g51_orchestrator.py --config campaign_config.json [--once] [--dry-run] [--interval-min 10]
Every poll (default every 10 minutes):
  1. PROBE each enabled host with ONE read-only ssh 'bash -s' script (no python on the host): GPU, disk, qualification file of
     the release, every campaign queue folder (dose_<plan>-*_a<k>: FINISHED / ABORTED / running, SUCCESS and problem counts),
     every G51 training queue (_train_*: RUN-DONE lines, FINISHED / ABORTED), the running executors and training launchers with
     their slot (K, s).
  2. PULL every terminal batch not yet pulled (launcher pull_rental.ps1: remote listing + scp download + size/SHA-256 check; its
     small toolkit upload is rate-limited in the day window) and every finished training run's queue folder (scp download).
  3. VALIDATE newly pulled batches on Linux (WSL Ubuntu-22.04, ~/usvb_validate venv, tools/validate_campaign_certs.py = the
     release's validate_cert_v3 at 64 x 2 + identity / seed registration / suite / release checks); one result file per batch.
  4. PPO: for every finished run with a gate record, append the record to the study gate ledger (a NEW version file, validated
     with the release's seed_registry.validate_ledger and delivery()), generate that run's test plan (make_campaign_plan.py
     --ppo-run) and mark it dispatchable.
  5. DISPATCH: for every idle slot of a qualified host, policy_g51.choose() (the simulated policy) picks one job; evaluation
     batches go through the unchanged G12 dispatch.py (one ledger per plan) and the G51 launcher launch_rental.ps1; a PPO
     training run is started with linux/g51_train_slot.sh (one run per launch); a PPO test batch on another host first gets the
     delivered checkpoint and the ledger version (small uploads, rate-limited).
  6. STATUS: a new status file status/status_<stamp>.txt and one line in status/status_log.txt (never overwritten).
  7. WAKE the controller ONLY on: ERROR (batch ABORTED, cell ERROR/TIMEOUT, INVALID certificate, executor gone with unfinished
     batches, launch / pull / validation failure), host unreachable > 1 h, a slot idle with nothing to dispatch, milestones
     (first valid certificate of each plan, 10/25/50/75/100 % of planned cells valid, all PPO training done). A wake is a new file
     wake/WAKE_<stamp>_<kind>.txt (each event key once: wake/wake_keys.txt); tools/g51_wait_wake.py blocks until one appears.
Rules kept: never kills a process, never deletes a file (local or remote), never overwrites a local file (new names only),
never starts a python on a host besides the launcher's tagged r3 tool and the G51 training launcher's slot-tagged pythons,
uploads in 09:00-01:00 local time only with scp -l <day_scp_limit_kbit> and at most <max_launches_per_poll_day> launches per
poll; no bulk transfer from this PC (checkpoints < 5 MB, plan files, ledgers only). --dry-run: probes for real (read-only),
dispatch.py --dry-run, launch_rental.ps1 -DescribeOnly, no pull / validation / ledger / wake files (writes only dryrun/)."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TOOLS = Path(r"D:/claude code/codex_tasks/20261014_g51_campaign/work/tools")   # G52 copy: the G51 tools, unchanged
sys.path.insert(0, str(TOOLS))
import policy_g51 as POL  # noqa: E402

PROBE = r'''
U=__U__; R=__R__; H=__H__
echo "PROBE-TIME $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GPU $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
echo "DISK $(df -Pk /root/autodl-tmp 2>/dev/null | awk 'NR==2{print $4}')"
q=$(ls $U/qualification/qualification_${R}_${H}_*.json 2>/dev/null | tail -1)
[ -n "$q" ] && grep -q QUALIFIED "$q" && echo "QUAL yes $q" || echo "QUAL no"
for d in $U/dose_*_a[0-9]; do
  [ -d "$d" ] || continue
  m=RUNNING; [ -e "$d/dose_FINISHED.txt" ] && m=FINISHED; [ -e "$d/dose_ABORTED.txt" ] && m=ABORTED
  s=$(grep -c " SUCCESS " "$d/progress_dose.txt" 2>/dev/null); e=$(grep -cE " (ERROR|TIMEOUT|NOT-STARTED-CEILING|PY-FOREIGN |GPU-MEMORY-LOW|ABORT)" "$d/progress_dose.txt" 2>/dev/null)
  echo "Q $(basename $d) $m ${s:-0} ${e:-0} $(stat -c %Y "$d/progress_dose.txt" 2>/dev/null)"
done
for d in $U/ppo/*/formal/_train_*; do      # FIX26: runs on other releases (v2.6.8/9) too
  [ -d "$d" ] || continue
  m=RUNNING; [ -e "$d/train_FINISHED.txt" ] && m=FINISHED; [ -e "$d/train_ABORTED.txt" ] && m=ABORTED
  echo "T $(basename $d) $m $(grep -c RUN-DONE "$d/progress_train.txt" 2>/dev/null) $(ls $d/run_*_DONE.txt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ',')"
done
pgrep -af 'usvbexec=ds-' | grep -v pgrep | sed -n 's/.*--slots \([0-9]\) --slot \([0-9]\).*/EXEC \1 \2/p'
pgrep -af g51_train_slot.sh | grep -v pgrep | sed -n 's/.*--slot-k \([0-9]\) --slot \([0-9]\).*/TRAINL \1 \2/p'
for f in $U/g51/ppo/*/best_eval.pt $U/g51/ledgers/*.json; do [ -f "$f" ] && echo "HAVE $f"; done
echo PROBE-END
'''


def now_local():
    return datetime.datetime.now()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="ascii", errors="replace", newline="\n") as f:
        f.write(text)


def append(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="ascii", errors="replace", newline="\n") as f:
        f.write(text)


def lowprio():
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = wintypes.HANDLE
            k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)      # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:  # noqa: BLE001
            pass


class Orchestrator(object):
    def __init__(self, cfg_path, dry):
        self.cfg_path = Path(cfg_path)
        self.cfg = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        self.dry = dry
        self.base = HERE / ("dryrun" if dry else "run")
        self.hosts = json.loads(Path(self.cfg["hosts_file"]).read_text(encoding="utf-8"))["hosts"]
        self.events = self.base / "events.jsonl"
        self.wake_dir = self.base / "wake"
        self.status_dir = self.base / "status"
        self.uploads = 0

    # ------------------------------------------------------------ small helpers
    def log(self, kind, **kw):
        line = json.dumps(dict(kw, time_utc=stamp(), kind=kind), sort_keys=True)
        print(line)
        append(self.events, line + "\n")

    def day_window(self):
        up = self.cfg["upload"]
        h = now_local().hour
        start, end = up["day_start_hour"], up["day_end_hour"]      # 9 and 1: day = 09:00 .. 24:59 (until 01:00)
        return h >= start or h < end

    def ssh_base(self, host):
        return ["ssh", "-4", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                "-o", "UserKnownHostsFile=%s" % self.cfg["known_hosts"], "-i", self.cfg["ssh_key"], "-p", str(host["port"]),
                "root@%s" % host["ssh_host"]]

    def run(self, argv, timeout=600, stdin=None, env=None):
        try:      # bytes in / out: Windows text mode would turn the probe script's LF into CRLF for the remote bash
            p = subprocess.run(argv, input=None if stdin is None else stdin.encode("ascii"), capture_output=True, timeout=timeout, env=env)
            return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            return 124, "", "TIMEOUT"

    def wake(self, key, kind, text):
        keys = self.wake_dir / "wake_keys.txt"
        seen = keys.read_text(encoding="ascii").split("\n") if keys.exists() else []
        if key in seen:
            return
        self.log("WAKE", key=key, wake_kind=kind, text=text[:500])
        if self.dry:
            return
        n = 0
        while (self.wake_dir / ("WAKE_%s_%s%s.txt" % (stamp(), kind, "_%d" % n if n else ""))).exists():   # FIX13: several wakes in one second
            n += 1
        write_new(self.wake_dir / ("WAKE_%s_%s%s.txt" % (stamp(), kind, "_%d" % n if n else "")), "kind=%s\nkey=%s\n%s\n" % (kind, key, text))
        append(keys, key + "\n")

    # ------------------------------------------------------------ plans, ledgers, jobs
    def plans(self):
        out = []
        for p in self.cfg["plans"]:
            out.append(dict(p, doc=json.loads(Path(p["plan_json"]).read_text(encoding="ascii"))))
        for f in sorted([x for x in (self.base / "ppo_plans").glob("*.json") if not x.name.endswith(".plan.json")]) if (self.base / "ppo_plans").exists() else []:
            spec = json.loads(f.read_text(encoding="ascii"))
            out.append(dict(spec, doc=json.loads(Path(spec["plan_json"]).read_text(encoding="ascii"))))
        return out

    def ledger_records(self, ledger):
        path = Path(ledger)
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="ascii").splitlines() if l.strip()]

    def jobs(self, plans):
        jobs = []
        cell_batch, batch_host = {}, {}
        for p in plans:
            if p.get("repro"):
                continue
            for c in p["doc"]["cells"]:
                cell_batch[c["name"]] = c["batch"]
            for r in self.ledger_records(p["ledger"]):
                batch_host.setdefault(r["batch_id"], r["host"])
        for p in plans:
            doc = p["doc"]
            cells = {c["name"]: c for c in doc["cells"]}
            if p.get("repro"):                       # REPRO-DUP plan: held until every original cell is dispatched
                for b in doc["batches"]:
                    origs = [n.split("_", 1)[1] for n in b["cells"]]
                    hosts = [batch_host.get(cell_batch.get(n)) for n in origs]
                    jobs.append({"id": b["id"], "kind": "repro", "table": "REPRO", "task": "", "vkey": "", "ctrl": "", "ppo_seed": None, "run": None,
                                 "h": {h: b["per_host"][h]["expected_two_slot_h"] for h in ("RENTAL", "4080", "5070")}, "plan": p["name"],
                                 "cells": len(b["cells"]), "target_cls": {"R": "RENTAL", "W": "5070", "F": "4080"}[p["repro"]],
                                 "avoid_hosts": None if None in hosts else sorted(set(hosts))})
                continue
            for b in doc["batches"]:
                c0 = cells[b["cells"][0]]
                jobs.append({"id": b["id"], "kind": "ppo_eval" if c0["ctrl"] == "ppo" else "eval", "table": c0["table_block"], "task": c0["task"],
                             "vkey": c0["vkey"], "ctrl": c0["ctrl"], "ppo_seed": c0["ppo_seed"],
                             "run": ("%s_%s_%s" % (b["block"].split("_")[0], c0["vkey"].upper(), c0["ppo_seed"])) if c0["ctrl"] == "ppo" else None,
                             "h": {h: b["per_host"][h]["expected_two_slot_h"] for h in ("RENTAL", "4080", "5070")}, "plan": p["name"],
                             "cells": len(b["cells"])})
        bf = self.base / "blocked_train_runs.txt"        # FIX20: training families with a known start-up bug are not launched
        blocked = [x for x in bf.read_text(encoding="ascii").split() if not x.startswith("#")] if bf.exists() else []
        for run in self.cfg["ppo"]["runs"]:
            if any(run == b or (b.endswith("*") and run.startswith(b[:-1])) for b in blocked):
                continue
            tc, ves, seed = run.split("-")
            exp = self.cfg["ppo"]["expected_h"][ves]
            jobs.append({"id": "train_%s" % run, "kind": "train", "table": "%s_%s_P%s" % (tc, ves, seed), "task": tc, "vkey": ves.lower(),
                         "ctrl": "ppo", "ppo_seed": int(seed), "run": "%s_%s_%s" % (tc, ves, seed), "h": {"RENTAL": exp, "4080": exp, "5070": exp * 1.94},
                         "plan": None, "cells": 0})
        return jobs

    def policy_state(self, plans, jobs):
        state = POL.State(jobs, rentals_online=sum(1 for h in self.hosts if h["enabled"] and h["class"] == "RENTAL"))
        byid = {j["id"]: j for j in jobs}
        for p in plans:
            for r in self.ledger_records(p["ledger"]):
                j = byid.get(r["batch_id"])
                if j and j["id"] not in state.taken:
                    state.take(j, r["host"], r["host_class"])
        for r in self.ledger_records(self.base / "train_ledger.jsonl"):
            j = byid.get("train_%s" % r["run"])
            if j and j["id"] not in state.taken:
                state.take(j, r["host"], "RENTAL")
            if r.get("released"):
                state.release_run(j["run"], r["host"])
        for f in [x for x in (self.base / "ppo_plans").glob("*.json") if not x.name.endswith(".plan.json")] if (self.base / "ppo_plans").exists() else []:
            spec = json.loads(f.read_text(encoding="ascii"))
            state.bound[spec["run"]] = spec["train_host"]
        return state

    # ------------------------------------------------------------ probe
    def probe(self, host):
        script = (PROBE.replace("__U__", self.cfg["data_root"] + "/usvb").replace("__R__", self.cfg["release_id"])
                  .replace("__H__", host["key"]))
        rc, out, err = self.run(self.ssh_base(host) + ["bash", "-s"], timeout=180, stdin=script)
        info = {"ok": rc == 0 and "PROBE-END" in out, "rc": rc, "queues": {}, "trains": {}, "exec": set(), "trainl": set(), "have": set(),
                "qual": False, "gpu": "", "disk_kb": 0, "err": err.strip()[-300:]}
        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Q" and len(parts) >= 5:
                info["queues"][parts[1]] = {"state": parts[2], "success": int(parts[3]), "problems": int(parts[4])}
            elif parts[0] == "T" and len(parts) >= 4:
                info["trains"][parts[1]] = {"state": parts[2], "done": int(parts[3] or 0), "files": parts[4] if len(parts) > 4 else ""}
            elif parts[0] == "EXEC":
                info["exec"].add((int(parts[1]), int(parts[2])))
            elif parts[0] == "TRAINL":
                info["trainl"].add((int(parts[1]), int(parts[2])))
            elif parts[0] == "HAVE":
                info["have"].add(parts[1])
            elif parts[0] == "QUAL":
                info["qual"] = parts[1] == "yes"
            elif parts[0] == "GPU" and len(parts) > 1:
                info["gpu"] = parts[1]
            elif parts[0] == "DISK" and len(parts) > 1 and parts[1].isdigit():
                info["disk_kb"] = int(parts[1])
        return info

    def reach_check(self, host, info):
        path = self.base / "reach.jsonl"
        if not self.dry:
            append(path, json.dumps({"host": host["key"], "ok": info["ok"], "time_utc": stamp()}) + "\n")
        last_ok, first_fail = None, None
        if path.exists():
            for l in path.read_text(encoding="ascii").splitlines():
                r = json.loads(l)
                if r["host"] != host["key"]:
                    continue
                if r["ok"]:
                    last_ok, first_fail = r["time_utc"], None
                elif first_fail is None:
                    first_fail = r["time_utc"]
        if not info["ok"] and first_fail:
            t0 = datetime.datetime.strptime(first_fail, "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc)
            mins = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds() / 60.0
            if mins > self.cfg["wake"]["unreachable_min"]:
                self.wake("unreachable:%s:%s" % (host["key"], first_fail), "UNREACHABLE", "host %s unreachable since %s (%.0f min); last ok %s; ssh rc=%s %s"
                          % (host["key"], first_fail, mins, last_ok, info["rc"], info["err"]))

    # ------------------------------------------------------------ pull + validate
    def pulled_dirs(self, qdir, host, any_state=False):
        root = Path(self.cfg["launcher_dir"]) / "pulls"
        ok = lambda q: (q / "dose_FINISHED.txt").exists() or (q / "dose_ABORTED.txt").exists() or (any_state and (q / "progress_dose.txt").exists())
        return sorted([d / qdir for d in root.glob("%s_%s_*" % (qdir, host["key"])) if ok(d / qdir)]) if root.exists() else []

    def ps_launcher(self, script, args, env_extra=None, timeout=3600):
        env = dict(os.environ)
        if self.day_window():
            env["G51_SCP_LIMIT_KBIT"] = str(self.cfg["upload"]["day_scp_limit_kbit"])
        if env_extra:
            env.update(env_extra)
        argv = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(Path(self.cfg["launcher_dir"]) / script)] + args
        return self.run(argv, timeout=timeout, env=env)

    def host_args(self, host):
        return ["-SshHost", host["ssh_host"], "-SshPort", str(host["port"]), "-SshUser", "root", "-SshKey", self.cfg["ssh_key"],
                "-DataRoot", self.cfg["data_root"]]

    def prefetch_pulls(self, deferred):
        """FIX30 (10-24): pulls run in a persistent background pool and never hold up a cycle (FIX28 still waited for the
        slowest pull, so cycles lasted 2-3 h and freed slots sat idle). At most one pull in flight per host; each cycle submits
        new pulls, harvests finished ones (logged here), and pull_and_validate runs only for queues whose folder has arrived."""
        if self.dry:
            return set()
        from concurrent.futures import ThreadPoolExecutor
        if not hasattr(self, "_pool"):
            self._pool, self._inflight, self._pullfails = ThreadPoolExecutor(max_workers=8), {}, {}
        def work(h, q):
            t0 = time.time()
            try:
                rc, o, e = self.ps_launcher("pull_rental.ps1", self.host_args(h) + ["-HostKey", h["key"], "-QueueDir", q], timeout=10800)
            except Exception as exc:  # noqa: BLE001
                rc, o, e = -1, repr(exc), ""
            return rc, o.strip()[-300:], (e or "")[-300:], round(time.time() - t0, 1)
        for q, (hk, fut) in list(self._inflight.items()):          # harvest
            if fut.done():
                rc, tail, err, wall = fut.result()
                del self._inflight[q]
                self.log("PULL-BG", host=hk, queue=q, rc=rc, wall_s=wall, tail=tail)
                if rc != 0:
                    n = self._pullfails[q] = self._pullfails.get(q, 0) + 1
                    if n >= 3:
                        self.wake("pullfail:%s:%s" % (hk, q), "ERROR", "background pull of %s from %s failed %d times rc=%s: %s %s" % (q, hk, n, rc, tail, err))
        busy = set(hk for hk, _ in self._inflight.values())
        for kind, h, a, b in deferred:                              # submit
            if kind == "pull" and a not in self._inflight and h["key"] not in busy and not self.pulled_dirs(a, h):
                self._inflight[a] = (h["key"], self._pool.submit(work, h, a))
                busy.add(h["key"])
                self.log("PULL-BG-START", host=h["key"], queue=a)
        return set(self._inflight)

    def pull_and_validate(self, host, qdir, qinfo, plans, recover=False):
        if self.dry:
            self.log("DRY-PULL", host=host["key"], queue=qdir, state=qinfo["state"])
            return
        if not self.pulled_dirs(qdir, host, any_state=recover):
            rc, out, err = self.ps_launcher("pull_rental.ps1", self.host_args(host) + ["-HostKey", host["key"], "-QueueDir", qdir])
            self.log("PULL", host=host["key"], queue=qdir, rc=rc, tail=out.strip()[-300:])
            if rc != 0 or not self.pulled_dirs(qdir, host, any_state=recover):
                self.wake("pullfail:%s:%s" % (host["key"], qdir), "ERROR", "pull of %s from %s failed rc=%s: %s %s" % (qdir, host["key"], rc, out[-500:], err[-300:]))
                return
        local = self.pulled_dirs(qdir, host, any_state=recover)[-1]
        vdir = self.base / "validation"
        if list(vdir.glob("%s_%s_*.txt" % (qdir, host["key"]))) if vdir.exists() else []:
            return
        plan = [p for p in plans if qdir.startswith("dose_%s-" % p["name"])]
        if not plan:
            self.wake("noplan:%s" % qdir, "ERROR", "pulled %s matches no plan" % qdir)
            return
        w = self.cfg["wsl"]
        to_wsl = lambda p: "/mnt/" + str(Path(p).resolve()).replace("\\", "/")[0].lower() + str(Path(p).resolve()).replace("\\", "/")[2:]
        cmd = ["wsl", "-d", w["distro"], "--", w["python"], "-B", to_wsl(TOOLS / "validate_campaign_certs.py"), "--release-root", w["release_root"],
               "--manifest", w["manifest"], "--plan", to_wsl(plan[0]["plan_json"]), "--record", to_wsl(self.cfg["record"]), to_wsl(local)]
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        rc, out, err = self.run(cmd, timeout=7200, env=env)
        write_new(vdir / ("%s_%s_%s.txt" % (qdir, host["key"], stamp())), out + ("\n# stderr\n" + err[-2000:] if err.strip() else ""))
        bad = [l for l in out.splitlines() if l.startswith("INVALID")]
        summary = [l for l in out.splitlines() if l.startswith("G51-VALIDATE")]
        self.log("VALIDATE", host=host["key"], queue=qdir, rc=rc, summary=summary[-1] if summary else "none", invalid=len(bad))
        if rc != 0 or bad or not summary:
            self.wake("invalid:%s:%s" % (host["key"], qdir), "ERROR", "validation of %s (%s): rc=%s %s\n%s" % (qdir, host["key"], rc,
                      summary[-1] if summary else "no summary", "\n".join(bad[:20])))
        m = re.search(r"missing=(\d+)", summary[-1]) if summary else None
        if not recover and not self.is_recovery_queue(qdir) and ((m and int(m.group(1)) > 0) or qinfo["state"] == "ABORTED" or qinfo["problems"] > 0):
            self.wake("batchprob:%s:%s" % (host["key"], qdir), "ERROR", "batch %s on %s ended %s with %d problem lines; %s"
                      % (qdir, host["key"], qinfo["state"], qinfo["problems"], summary[-1] if summary else ""))

    # ------------------------------------------------------------ FIX13: queues killed by a host power-off (10-17)
    def recovery_records(self):
        return self.ledger_records(self.base / "recovered.jsonl")

    def is_recovery_queue(self, qdir):
        """A queue listed for recovery or the follow-up attempt it launched: missing cells there are expected."""
        recs = self.recovery_records()
        f = self.base / "recover_queues.txt"
        listed = [l.split()[-1] for l in f.read_text(encoding="ascii").splitlines() if l.strip() and not l.startswith("#")] if f.exists() else []
        return qdir in listed or any(r["qdir"] == qdir or r.get("next_qdir") == qdir for r in recs)

    def recover_dead_queues(self, host, info, plans, jobs, idle):
        if host.get("no_eval"):
            return idle, 0
        idle, nrel = self.relaunch_listed(host, jobs, plans, idle)
        f = self.base / "recover_queues.txt"
        if not f.exists():
            return idle, nrel
        listed = [l.split() for l in f.read_text(encoding="ascii").splitlines() if l.strip() and not l.startswith("#")]
        done = set(r["qdir"] for r in self.recovery_records())
        byid = {j["id"]: j for j in jobs}
        launched = 0
        for hk, qdir in listed:
            if hk != host["key"] or qdir in done or qdir not in info["queues"]:
                continue
            self.pull_and_validate(host, qdir, info["queues"][qdir], plans, recover=True)
            dirs = self.pulled_dirs(qdir, host, any_state=True)
            if not dirs:
                continue
            prog = (dirs[-1] / "progress_dose.txt").read_text(encoding="ascii", errors="replace").splitlines()
            ok = set(l.split()[3] for l in prog if len(l.split()) > 3 and l.split()[2] == "SUCCESS")
            starts = [(l.split()[3], int(l.split("order=")[1].split()[0])) for l in prog
                      if len(l.split()) > 3 and l.split()[2] == "START-CELL" and "order=" in l]
            todo = [o for n, o in starts if n not in ok]
            bid, att = qdir[len("dose_"):].rsplit("_a", 1)
            rec = {"qdir": qdir, "host": host["key"], "done_cells": len(ok), "time_utc": stamp()}
            if not todo:
                # no unfinished started cell: resume after the last started one if the batch has more cells
                plan = [p for p in plans if bid in [b["id"] for b in p["doc"]["batches"]]]
                cells = [b["cells"] for b in plan[0]["doc"]["batches"] if b["id"] == bid][0] if plan else []
                if len(ok) >= len(cells):
                    rec["action"] = "complete"
                    append(self.base / "recovered.jsonl", json.dumps(rec) + "\n")
                    self.log("RECOVER-COMPLETE", host=host["key"], queue=qdir)
                    continue
                todo = [max(o for _, o in starts) + 1] if starts else [None]
            if todo[0] is None or not idle or bid not in byid:
                continue
            slot = idle.pop(0)
            rec.update(action="reattempt", from_order=todo[0], slot=slot, next_qdir="dose_%s_a%d" % (bid, int(att) + 1))
            append(self.base / "recovered.jsonl", json.dumps(rec) + "\n")
            self.log("RECOVER-REATTEMPT", host=host["key"], queue=qdir, from_order=todo[0], slot=slot)
            self.dispatch_eval(host, slot, byid[bid], plans, reattempt=todo[0])
            launched += 1
        return idle, launched + nrel

    def relaunch_listed(self, host, jobs, plans, idle):
        """FIX16: batches whose recovery launch was refused (cloned host not yet re-qualified): run.recover_relaunch.txt lines
        '<host> <batch id> <from order>', each relaunched once on its own host with dispatch.py --reattempt <batch>:<order>."""
        idle, nre = self.rehost_listed(host, jobs, plans, idle)
        f = self.base / "recover_relaunch.txt"
        if not f.exists():
            return idle, nre
        done = set(r.get("relaunch") for r in self.recovery_records())
        byid = {j["id"]: j for j in jobs}
        n = 0
        for l in f.read_text(encoding="ascii").splitlines():
            p = l.split()
            if len(p) != 3 or l.startswith("#") or p[0] != host["key"] or l.strip() in done or p[1] not in byid or not idle:
                continue
            slot = idle.pop(0)
            append(self.base / "recovered.jsonl", json.dumps({"relaunch": l.strip(), "qdir": "relaunch:" + l.strip(), "host": host["key"],
                                                              "slot": slot, "time_utc": stamp()}) + "\n")
            self.log("RECOVER-RELAUNCH", host=host["key"], batch=p[1], from_order=int(p[2]), slot=slot)
            self.dispatch_eval(host, slot, byid[p[1]], plans, reattempt=int(p[2]))
            n += 1
        return idle, n + nre

    def rehost_listed(self, host, jobs, plans, idle):
        """FIX17: run/rehost_relaunch.txt lines '<batch id> <from order> <lost host>': batches stranded on a host that will not
        come back; each goes once to the first idle slot of ANY usable host via dispatch_g54 --rehost (ledger: rehost_from)."""
        f = self.base / "rehost_relaunch.txt"
        if not f.exists():
            return idle, 0
        done = set(r.get("rehost") for r in self.recovery_records())
        byid = {j["id"]: j for j in jobs}
        n = 0
        for l in f.read_text(encoding="ascii").splitlines():
            p = l.split()
            if len(p) != 3 or l.startswith("#") or l.strip() in done or p[0] not in byid or not idle:
                continue
            slot = idle.pop(0)
            append(self.base / "recovered.jsonl", json.dumps({"rehost": l.strip(), "qdir": "rehost:" + l.strip(), "host": host["key"],
                                                              "slot": slot, "time_utc": stamp()}) + "\n")
            self.log("RECOVER-REHOST", host=host["key"], batch=p[0], from_order=int(p[1]), lost_host=p[2], slot=slot)
            self.dispatch_eval(host, slot, byid[p[0]], plans, reattempt=int(p[1]), rehost=True)
            n += 1
        return idle, n

    def valid_cells(self):
        valid = set()
        vdir = self.base / "validation"
        for f in vdir.glob("*.txt") if vdir.exists() else []:
            for l in f.read_text(encoding="ascii", errors="replace").splitlines():
                if l.startswith("VALID "):
                    valid.add(l.split()[1])
        return valid

    # ------------------------------------------------------------ PPO post-processing
    def train_runs_done(self, host, info):
        """Finished training runs (run_<id>_DONE.txt) not yet processed -> pull the queue folder, ledger append, PPO plan."""
        failed = set((r["queue"], r["run"]) for r in self.ledger_records(self.base / "train_failed_seen.jsonl"))
        for tq, t in info["trains"].items():
            for f in [x for x in t["files"].split(",") if x]:
                run = f[len("run_"):-len("_DONE.txt")]
                spec_path = self.base / "ppo_plans" / ("%s.json" % run)
                if (tq, run) in failed:            # FIX2: a failed run is pulled and reported once, not every poll
                    continue
                if spec_path.exists() or self.dry:
                    if self.dry:
                        self.log("DRY-PPO-DONE", host=host["key"], run=run, queue=tq)
                    continue
                self.process_run(host, tq, run, spec_path)

    def process_run(self, host, tq, run, spec_path):
        u = self.cfg["data_root"] + "/usvb"
        rq = "%s/ppo/%s/formal/%s" % (u, self.run_override(run).get("release_id", self.cfg["release_id"]), tq)   # FIX26
        dest = self.base / "train_pulls" / ("%s_%s_%s" % (tq, host["key"], stamp()))
        dest.mkdir(parents=True)
        o = ["-4", "-o", "BatchMode=yes", "-o", "UserKnownHostsFile=%s" % self.cfg["known_hosts"], "-i", self.cfg["ssh_key"], "-P", str(host["port"])]
        rc, out, err = self.run(["scp", "-r"] + o + ["root@%s:%s" % (host["ssh_host"], rq), str(dest)], timeout=3600)
        local = dest / tq
        done = local / ("run_%s_DONE.txt" % run)
        if rc != 0 or not done.exists():
            self.wake("trainpull:%s" % run, "ERROR", "pull of training queue %s (%s) failed rc=%s %s" % (tq, host["key"], rc, err[-300:]))
            return
        kv = dict(re.findall(r"(\w+)=(\S+)", done.read_text(encoding="ascii", errors="replace")))
        grec = local / "gate" / ("%s_gate_record.json" % run)
        best = local / ("best_%s" % run) / "best_eval.pt"
        if not grec.exists() or grec.stat().st_size == 0 or not best.exists():   # FIX9: empty record = failed gate-record
            self.wake("trainfail:%s:%s" % (run, tq), "ERROR", "run %s on %s ended without a delivered checkpoint or gate record: %s" % (run, host["key"], done.read_text()[:800]))
            append(self.base / "train_failed_seen.jsonl", json.dumps({"queue": tq, "run": run, "host": host["key"], "time_utc": stamp()}) + "\n")
            txt = done.read_text()
            if "state=HEARTBEAT-STALE" in txt or "state=TIMEOUT" in txt:   # FIX21: time-limit kills retry too   # FIX4: any watchdog kill (no delivered model) -> clean restart, same seed   # FIX2: hung before its first checkpoint -> one clean retry, any rental
                prev = [r for r in self.ledger_records(self.base / "train_retry.jsonl") if r["run"] == run and not r["launched"]]
                if len(prev) < 2:                  # FIX3: up to attempt 3
                    append(self.base / "train_retry.jsonl", json.dumps({"run": run, "attempt": len(prev) + 2, "failed_host": host["key"], "launched": False, "time_utc": stamp()}) + "\n")
                    self.log("TRAIN-RETRY-QUEUED", run=run, failed_host=host["key"])
            return
        ledger_path, ledger_sha = self.ledger_append(json.loads(grec.read_text(encoding="utf-8")), run)
        if ledger_path is None:
            return
        tc, ves, seed = run.split("-")
        task = {v: k for k, v in POL_TASK_CODE().items()}[tc]
        name = "p-%s-%s-%s" % (tc, ves, seed)
        pdir = self.base / "ppo_plans" / name
        pdir.parent.mkdir(parents=True, exist_ok=True)   # FIX10: folder did not exist -> plan tool failed
        remote_ckpt = "%s/g51/ppo/%s/best_eval.pt" % (u, run)
        remote_ledger = "%s/g51/ledgers/%s" % (u, ledger_path.name)
        cmd = [sys.executable, "-B", str(TOOLS / "make_campaign_plan.py"), "--name", name, "--release-root", self.cfg["release_root"],
               "--release-id", self.cfg["release_id"], "--timing", self.cfg["timing"], "--predictions", self.cfg["predictions"],
               "--controllers", "ppo", "--seed-blocks", self.cfg.get("seed_blocks", "final-01,final-02,final-03,final-04"), "--ppo-run", "%s:%s:%s" % (task, ves.lower(), seed), "--ppo-checkpoint", remote_ckpt,
               "--ppo-checkpoint-sha256", sha_file(best), "--gate-ledger", remote_ledger, "--gate-ledger-sha256", ledger_sha,
               "--out-json", str(pdir) + ".plan.json", "--out-text", str(pdir) + ".txt", "--out-dir", str(pdir)]
        rc, out, err = self.run(cmd, timeout=600)
        if rc != 0:
            self.wake("ppoplan:%s" % run, "ERROR", "PPO plan for %s failed: %s %s" % (run, out[-400:], err[-400:]))
            return
        spec = {"name": name, "plan_json": str(pdir) + ".plan.json", "batch_dir": str(pdir), "ledger": str(self.base / "dispatch" / ("ledger_%s.jsonl" % name)),
                "run": "%s_%s_%s" % (tc, ves, seed), "train_host": host["key"], "local_checkpoint": str(best), "remote_checkpoint": remote_ckpt,
                "local_ledger": str(ledger_path), "remote_ledger": remote_ledger, "train_queue": tq}
        write_new(spec_path, json.dumps(spec, indent=1, sort_keys=True) + "\n")
        append(self.base / "train_ledger.jsonl", json.dumps({"run": run, "host": host["key"], "released": True, "time_utc": stamp()}) + "\n")
        self.log("PPO-RELEASED", run=run, host=host["key"], plan=name, ledger=ledger_path.name)

    def ledger_append(self, record, run):
        ldir = self.base / "ledgers"
        ldir.mkdir(parents=True, exist_ok=True)
        versions = sorted(ldir.glob("gate_ledger_v*.json"))
        sys.path.insert(0, str(Path(self.cfg["release_root"]) / "scripts"))
        import seed_registry as SR  # the release's own (pure python)
        registry = SR.load_registry(Path(self.cfg["release_root"]) / "scripts" / "seed_registry.json")
        ledger = json.loads(versions[-1].read_text(encoding="ascii")) if versions else SR.empty_ledger(registry, 2)
        if versions and record in ledger["records"]:      # FIX10: same record already delivered (plan step failed earlier) -> reuse
            return versions[-1], sha_file(versions[-1])
        ledger["records"].append(record)
        try:
            SR.validate_ledger(ledger, registry)
            SR.delivery(ledger, registry, record["task_id"], record["vessel"])
        except Exception as exc:  # noqa: BLE001
            self.wake("ledger:%s" % run, "ERROR", "gate record of %s refused by the release's ledger rules: %s" % (run, exc))
            return None, None
        path = ldir / ("gate_ledger_v%03d.json" % (len(versions) + 1))
        write_new(path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        return path, sha_file(path)

    # ------------------------------------------------------------ dispatch
    def idle_slots(self, host, info):
        busy = info["exec"] | info["trainl"]
        return [s for s in range(host["K"]) if (host["K"], s) not in busy]

    def stranded(self, host, info, plans):
        """FIX1 (10-14): batches whose launch was refused by the rental's slot lock (nothing written there) -> re-run on the same
        host with dispatch.py --reattempt <batch>:1 (next attempt). One retry per refused launch event (run/stranded_retry.jsonl)."""
        done = set(r["event"] for r in self.ledger_records(self.base / "stranded_retry.jsonl"))
        last = {}
        for p in plans:
            for r in self.ledger_records(p["ledger"]):
                last[r["batch_id"]] = r
        out = []
        for r in self.ledger_records(self.base / "events.jsonl"):
            if r.get("kind") != "LAUNCH" or r.get("host") != host["key"] or "slot lock busy" not in r.get("tail", ""):
                continue
            ev = "%s|%s" % (r["batch"], r["time_utc"])
            lr = last.get(r["batch"])
            if ev in done or lr is None or lr["host"] != host["key"] or ("dose_%s_a%d" % (r["batch"], lr["attempt"])) in info["queues"]:
                continue
            if any(ev.split("|")[0] == o[0] for o in out):
                continue
            out.append((r["batch"], ev))
        return out

    def dispatch_eval(self, host, slot, job, plans, reattempt=None, rehost=False):
        plan = [p for p in plans if p["name"] == job["plan"]][0]
        dpy = self.cfg["dispatch_py_rehost"] if rehost else self.cfg["dispatch_py"]   # FIX17: G54 copy only for --rehost
        dargs = [sys.executable, "-B", dpy, "--plan", plan["plan_json"], "--batch-dir", plan["batch_dir"], "--ledger", plan["ledger"],
                 "--host", host["key"], "--host-class", host["class"], "--slots", str(host["K"]), "--slot", str(slot), "--batch", job["id"]]
        if reattempt:
            dargs += ["--reattempt", "%s:%d" % (job["id"], reattempt if isinstance(reattempt, int) and not isinstance(reattempt, bool) else 1)]
        if rehost:
            dargs += ["--rehost", job["id"]]
        if self.dry:
            dargs.append("--dry-run")
        rc, out, err = self.run(dargs, timeout=600)
        m = re.search(r"SLOT-LIST .* list=(\S+) sha256=", out)
        self.log("DISPATCH", host=host["key"], slot=slot, batch=job["id"], rc=rc, out=out.strip()[-300:])
        if rc != 0 or not m:
            self.wake("dispatchfail:%s" % job["id"], "ERROR", "dispatch.py refused %s for %s s%d: %s %s" % (job["id"], host["key"], slot, out[-400:], err[-200:]))
            return False
        if job["kind"] == "ppo_eval" and not self.dry:
            spec = [p for p in plans if p["name"] == job["plan"]][0]
            if not self.stage_ppo_inputs(host, spec):
                return False
        largs = self.host_args(host) + ["-HostKey", host["key"], "-Record", self.cfg["record"], "-RecordSha256", self.cfg["record_sha256"],
                                        "-ExecutorSha256", self.cfg["executor_sha256"], "-Slots", str(host["K"]), "-Slot", str(slot),
                                        "-PlanList", str(Path(plan["batch_dir"]) / m.group(1))]
        if self.dry:
            # dispatch.py --dry-run writes no list: describe the launch from a copy of the batch and a one-line list in dryrun/lists
            ldir = self.base / "lists" / ("%s_%s_s%d" % (stamp(), host["key"], slot))
            ldir.mkdir(parents=True)
            src = Path(plan["batch_dir"]) / ("%s.plan" % job["id"])
            data = src.read_bytes()
            (ldir / src.name).write_bytes(data)
            write_new(ldir / "describe.list", "format=usvbench-g12-planlist-v1\nbatch|%s|%s|1|1\n" % (src.name, hashlib.sha256(data).hexdigest().upper()))
            largs[-1] = str(ldir / "describe.list")
            rc, out, err = self.ps_launcher("launch_rental.ps1", largs + ["-DescribeOnly"])
            write_new(ldir / "describe_output.txt", out + "\n# stderr\n" + err)
            self.log("DRY-LAUNCH", host=host["key"], slot=slot, batch=job["id"], rc=rc, tail=out.strip()[-200:])
            return rc == 0
        rc, out, err = self.ps_launcher("launch_rental.ps1", largs)
        self.uploads += 1
        self.log("LAUNCH", host=host["key"], slot=slot, batch=job["id"], rc=rc, tail=out.strip()[-400:])
        if rc != 0 or "LAUNCHED task=" not in out:
            self.wake("launchfail:%s" % job["id"], "ERROR", "launch of %s on %s s%d failed rc=%s: %s %s" % (job["id"], host["key"], slot, rc, out[-600:], err[-300:]))
            return False
        return True

    def scp_up(self, host, local, remote_dir):
        lim = ["-l", str(self.cfg["upload"]["day_scp_limit_kbit"])] if self.day_window() else []
        rc, out, err = self.run(self.ssh_base(host) + ["mkdir -p %s" % remote_dir], timeout=120)
        if rc != 0:
            return False
        o = lim + ["-4", "-o", "BatchMode=yes", "-o", "UserKnownHostsFile=%s" % self.cfg["known_hosts"], "-i", self.cfg["ssh_key"], "-P", str(host["port"])]
        name = Path(local).name
        rc, out, err = self.run(self.ssh_base(host) + ["test ! -e %s/%s" % (remote_dir, name)], timeout=120)
        if rc == 1:
            return True                            # already there (checked by the executor's hash pin before every PPO cell)
        if rc != 0:
            return False                           # FIX33 (10-25): ssh failure (255) or timeout used to count as "already there"
        rc, out, err = self.run(["scp"] + o + [str(local), "root@%s:%s/%s" % (host["ssh_host"], remote_dir, name)], timeout=1800)
        self.uploads += 1
        return rc == 0

    def stage_ppo_inputs(self, host, spec):
        u = self.cfg["data_root"] + "/usvb"
        ok = self.scp_up(host, spec["local_checkpoint"], str(Path(spec["remote_checkpoint"]).parent).replace("\\", "/")) and \
            self.scp_up(host, spec["local_ledger"], "%s/g51/ledgers" % u)
        if not ok:
            self.wake("ppostage:%s:%s" % (spec["run"], host["key"]), "ERROR", "could not stage checkpoint / ledger of %s on %s" % (spec["run"], host["key"]))
        return ok

    def run_override(self, run):
        for pre, ov in (self.cfg.get("run_release_overrides") or {}).items():
            if run.startswith(pre):
                return ov
        return {}

    def train_allowed_on(self, run, host):
        return host["key"] not in self.run_override(run).get("exclude_hosts", [])

    def pending_train_retries(self):
        recs = self.ledger_records(self.base / "train_retry.jsonl")
        launched = set((r["run"], r["attempt"]) for r in recs if r["launched"])
        return [r for r in recs if not r["launched"] and (r["run"], r["attempt"]) not in launched]

    # FIX29 (10-24): a training slot (trainer + up to 4 evals) used the whole container CPU quota (cgroup cpu.max, 16-25 cores);
    # the cgroup was throttled every period and the other slot's eval cells crawled (17 min cells hit 1.8 h timeouts on rental02).
    # Training slots now run under taskset on quota-3 cores, leaving about 3 cores for the eval slot.
    def dispatch_train(self, host, slot, job, attempt=1, reuse=None, gate_only_from=None):
        run = job["id"][len("train_"):]
        c = self.cfg
        u = c["data_root"] + "/usvb"
        tdir = u + "/" + c.get("train_tools_dir", "usvb_tools_g51")
        ov = self.run_override(run)          # FIX22: path_hazard trains on v2.6.8 (training-pool fix), owner 10-19
        args = ("--host-key %s --slot-k %d --slot %d --runs %s --release-id %s --record-sha256 %s --predictions %s/%s --predictions-sha256 %s"
                % (host["key"], host["K"], slot, run, ov.get("release_id", c["release_id"]), ov.get("record_sha256", c["record_sha256"]).lower(),
                   tdir, Path(c["predictions"]).name,
                   sha_file(c["predictions"]).lower()))
        args += " --eval-par-train %d" % c.get("eval_par_train", 3)   # FIX6: fewer evals beside the trainer (starvation)
        if ov.get("release_id"):              # FIX23: keep the campaign slot tag for runs on another release
            args += " --slot-tag-release %s" % c["release_id"]
        if attempt > 1:
            args += " --attempt %d" % attempt
        if reuse:                                  # FIX5: re-evaluate an already trained run on its own host
            args += " --reuse-run-dir %s" % reuse
        if gate_only_from:                         # FIX8: gate only, best checkpoint from the earlier queue
            args += " --gate-only-from %s" % gate_only_from
        if ov.get("gate_release", c.get("gate_release")):   # FIX8: competence gate on v2.6.7 (seed-11001 pools)
            args += " --gate-release %s" % ov.get("gate_release", c.get("gate_release"))
        if self.dry:
            self.log("DRY-TRAIN", host=host["key"], slot=slot, run=run, cmd="nohup setsid bash %s/g51_train_slot.sh %s" % (tdir, args))
            return True
        if not (self.scp_up(host, c["train_launcher"], tdir) and self.scp_up(host, c["predictions"], tdir)):
            self.wake("trainstage:%s" % run, "ERROR", "could not stage the training launcher on %s" % host["key"])
            return False
        rc, out, err = self.run(self.ssh_base(host) + ["cd %s && %sQ=$(awk '{print ($1==\"max\")?32:int($1/100000)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 32); [ \"$Q\" -ge 8 ] || Q=8; EXP_WAMV=%d EVAL_TO_WAMV=%d EXP_BB=%d EVAL_TO_BB=%d setsid -f nohup taskset -c 0-$((Q-4)) bash %s/g51_train_slot.sh %s > %s/train_%s_%s.nohup 2>&1 < /dev/null; echo STARTED"
                                                       % (tdir, ("export STALE_S_OVERRIDE=21600; " if run.startswith("HM-WAMV") else ""), int(c.get("exp_wamv_s", 21600)), int(c.get("eval_to_wamv_s", 5400)), int(c.get("exp_bb_s", 9000)), int(c.get("eval_to_bb_s", 3600)), tdir, args, tdir, run, stamp())], timeout=120)
        append(self.base / "train_ledger.jsonl", json.dumps({"run": run, "host": host["key"], "slot": slot, "released": False, "time_utc": stamp()}) + "\n")
        self.log("TRAIN-LAUNCH", host=host["key"], slot=slot, run=run, rc=rc, out=out.strip()[-200:])
        if rc != 0 or "STARTED" not in out:
            self.wake("trainlaunch:%s" % run, "ERROR", "training launch %s on %s failed: %s %s" % (run, host["key"], out, err[-300:]))
            return False
        return True

    # ------------------------------------------------------------ one poll
    def sync_hosts_from_table(self):
        """FIX14 (10-17): the housekeeping window keeps the rental table (cfg rental_table, E: drive, 连接表_最新.md) current
        every 5 min. Address, port and usability come from that table; class / K / key from the hosts file. A host is used
        only when the table says 运行中, the data-disk copy is 完成 or 不需要, and the environment check is OK (or 都在)."""
        p = Path(self.cfg.get("rental_table", ""))
        if not p.exists():
            return
        rows = {}
        for l in p.read_text(encoding="utf-8").splitlines():
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            if len(c) >= 7 and c[0].startswith("rental"):
                rows[c[0]] = c
        for h in self.hosts:
            r = rows.get(h["key"])
            if not r or h.get("class") != "RENTAL":
                continue
            addr = r[2]
            if ":" in addr:
                hostname, port = addr.rsplit(":", 1)
                h["ssh_host"], h["port"] = hostname, int(port)
            usable = ("运行中" in r[4]) and (("完成" in r[5]) or ("不需要" in r[5])) and (("OK" in r[6]) or ("都在" in r[6]))
            if h.get("enabled") != usable:
                self.log("HOST-TABLE", host=h["key"], enabled=usable, addr=addr, state=r[4], copy=r[5], env=r[6])
            force = self.base / "force_ready.txt"          # FIX18: hosts the controller set up and qualified itself (table says 待)
            if not usable and "运行中" in r[4] and force.exists() and h["key"] in force.read_text(encoding="ascii").split():
                usable = True
            hold = self.base / "hold_hosts.txt"           # FIX16: hosts waiting for re-qualification after a clone
            h["no_eval"] = bool(usable and hold.exists() and h["key"] in hold.read_text(encoding="ascii").split())   # FIX19: held = trains only
            h["enabled"] = usable

    def auto_list_dead(self, host, info):
        """FIX14: on a host with no executor and no training launcher, every non-terminal campaign queue is dead (power-off)."""
        if info["exec"] or info["trainl"]:
            return
        dead = [q for q, v in info["queues"].items() if v["state"] not in ("FINISHED", "ABORTED")]
        f = self.base / "recover_queues.txt"
        have = f.read_text(encoding="ascii") if f.exists() else ""
        new = [q for q in dead if (" %s" % q) not in have]
        if new:
            append(f, "".join("%s %s\n" % (host["key"], q) for q in new))
            self.log("RECOVER-LISTED", host=host["key"], queues=new)

    def poll(self):
        self.sync_hosts_from_table()
        plans = self.plans()
        jobs = self.jobs(plans)
        state = self.policy_state(plans, jobs)
        lines, launches = [], 0
        deferred = []          # FIX27: pulls/validations and finished-training processing run AFTER all dispatching
        cap = self.cfg["upload"]["max_launches_per_poll_day" if self.day_window() else "max_launches_per_poll_night"]
        dispatched_queues = set()
        for p in plans:
            for r in self.ledger_records(p["ledger"]):
                dispatched_queues.add(("dose_%s_a%d" % (r["batch_id"], r["attempt"]), r["host"]))
        for host in self.hosts:
            if not host["enabled"]:
                continue
            if host["class"] != "RENTAL":
                lines.append("%s class=%s SKIPPED (no G51 FINAL executor port for Windows hosts yet)" % (host["key"], host["class"]))
                continue
            info = self.probe(host)
            self.reach_check(host, info)
            if not info["ok"]:
                lines.append("%s UNREACHABLE rc=%s %s" % (host["key"], info["rc"], info["err"][:120]))
                continue
            names = tuple("dose_%s-" % p["name"] for p in plans)
            info["queues"] = {k: v for k, v in info["queues"].items() if k.startswith(names)}
            if not self.dry:
                self.auto_list_dead(host, info)
            for qdir, q in sorted(info["queues"].items()):
                if q["state"] in ("FINISHED", "ABORTED"):
                    deferred.append(("pull", host, qdir, q))
                elif not any(k == host["K"] for k, _ in info["exec"]) and (qdir, host["key"]) in dispatched_queues and not self.is_recovery_queue(qdir):
                    self.wake("dead:%s:%s" % (host["key"], qdir), "ERROR", "queue %s on %s is not terminal but no executor runs" % (qdir, host["key"]))
            for tq, t in info["trains"].items():
                if t["state"] == "ABORTED":
                    self.wake("trainabort:%s" % tq, "ERROR", "training queue %s on %s ABORTED" % (tq, host["key"]))
                elif t["state"] == "FINISHED" and t["done"] == 0:
                    self.wake("trainnorun:%s" % tq, "ERROR", "training queue %s on %s FINISHED without a RUN-DONE (dry-run refusal, "
                              "claim skip or crash): read its progress_train.txt" % (tq, host["key"]))
            deferred.append(("train", host, info, None))
            idle = self.idle_slots(host, info) if (info["qual"] or self.dry) else []     # dry run: describe even unqualified hosts
            exf = self.base / "exclusive_hosts.txt"      # FIX7: hosts reserved for one slow PPO training (owner 10-15: harbour WAM-V alone)
            if idle and exf.exists() and host["key"] in exf.read_text(encoding="ascii").split():
                lines.append("%s EXCLUSIVE (no new jobs) idle=%s" % (host["key"], idle))
                mine = [r for r in self.pending_train_retries() if r.get("pin_host") == host["key"]]
                if not mine and len(idle) == host["K"] and not info["exec"] and not info["trainl"]:   # FIX12: never sit idle silently
                    self.wake("exclusive-idle:%s" % host["key"], "ERROR", "exclusive host %s is fully idle and has no pinned run" % host["key"])
                if not (mine and len(idle) == host["K"] and not info["exec"] and not info["trainl"]):
                    idle = []                            # the pinned slow run starts only on a fully empty host
                else:
                    idle = idle[:1]
            if idle and not self.dry:
                idle, nrec = self.recover_dead_queues(host, info, plans, jobs, idle)
                launches += nrec
                for tr in self.pending_train_retries():
                    if not idle or launches >= cap:
                        break
                    if tr.get("pin_host") and tr["pin_host"] != host["key"]:
                        continue
                    if not self.train_allowed_on(tr["run"], host):
                        continue
                    slot = idle.pop(0)
                    append(self.base / "train_retry.jsonl", json.dumps({"run": tr["run"], "attempt": tr["attempt"], "host": host["key"], "slot": slot, "launched": True, "time_utc": stamp()}) + "\n")
                    self.log("TRAIN-RETRY", host=host["key"], slot=slot, run=tr["run"], attempt=tr["attempt"])
                    self.dispatch_train(host, slot, {"id": "train_%s" % tr["run"]}, attempt=tr["attempt"], reuse=tr.get("reuse_dir"), gate_only_from=tr.get("gate_only_from"))
                    launches += 1
                    if not hasattr(self, "_retry_launched"):
                        self._retry_launched = {}
                    self._retry_launched[host["key"]] = time.time()     # FIX31: not yet visible to the probe
                byid = {j["id"]: j for j in jobs}
                for bid, ev in ([] if host.get("no_eval") else self.stranded(host, info, plans)):
                    if not idle or launches >= cap or bid not in byid:
                        break
                    slot = idle.pop(0)
                    append(self.base / "stranded_retry.jsonl", json.dumps({"event": ev, "host": host["key"], "slot": slot, "time_utc": stamp()}) + "\n")
                    self.log("STRANDED-RETRY", host=host["key"], slot=slot, batch=bid)
                    self.dispatch_eval(host, slot, byid[bid], plans, reattempt=True)
                    launches += 1
            # FIX31 (10-24): the policy's per-host "training running" count came from the ledger and was only decremented on the
            # host that released the run, so hosts whose run was retried elsewhere stayed "busy" forever and the per-host cap
            # blocked the rest: 21 BlueBoat trainings were never launched. Use what the probe actually sees on the host instead.
            # FIX32 (10-25): evals now wait on trainings, so the second slot sat idle while trainings queued; allow two
            # trainings per host (both slots). Each slot is already pinned to quota-3 cores (FIX29).
            state.train_running[host["key"]] = max(0, len(info["trainl"]) + (1 if time.time() - getattr(self, "_retry_launched", {}).get(host["key"], 0) < 1800 else 0) - 1)
            state.train_count[host["key"]] = 0
            for slot in idle:
                if launches >= cap:
                    break
                job = POL.choose(state, host["key"], host["class"])
                if job is not None and job["kind"] == "train" and not self.train_allowed_on(job["id"][len("train_"):], host):
                    lines.append("%s train %s not allowed here (release missing)" % (host["key"], job["id"]))
                    break
                if job is not None and host.get("no_eval") and job["kind"] != "train":   # FIX19: not re-qualified yet
                    lines.append("%s NO-EVAL (awaiting re-qualification) idle=%s" % (host["key"], idle))
                    break
                if job is None:
                    self.wake("idle:%s:%d:%s" % (host["key"], slot, len(state.taken)), "IDLE",
                              "slot %s s%d is idle and nothing is dispatchable (taken %d of %d jobs)" % (host["key"], slot, len(state.taken), len(state.jobs)))
                    break
                ok = self.dispatch_train(host, slot, job) if job["kind"] == "train" else self.dispatch_eval(host, slot, job, plans)
                launches += 1
            lines.append("%s qual=%s gpu=%s disk_gb=%.0f exec=%s train=%s queues=%d (%s) idle_slots=%s"
                         % (host["key"], info["qual"], info["gpu"], info["disk_kb"] / 1e6, sorted(info["exec"]), sorted(info["trainl"]), len(info["queues"]),
                            ",".join("%s:%s" % (k.split("-")[-2] if "-" in k else k, v["state"][0]) for k, v in sorted(info["queues"].items())[-6:]),
                            idle))
        inflight = self.prefetch_pulls(deferred)   # FIX30: pulls in a background pool; validate only what has arrived
        for kind, h, a, b in deferred:
            try:
                if kind == "pull":
                    if a in inflight or not self.pulled_dirs(a, h):
                        continue
                    self.pull_and_validate(h, a, b, plans)
                else:
                    self.train_runs_done(h, a)
            except Exception as exc:  # noqa: BLE001
                self.log("DEFERRED-EXCEPTION", host=h["key"], kind=kind, error=repr(exc)[:300])
        valid = self.valid_cells()
        planned = sum(len(p["doc"]["cells"]) for p in plans)
        total_main = self.cfg.get("planned_cells_total", planned)
        frac = len(valid) / float(total_main) if total_main else 0.0
        for m in (0.10, 0.25, 0.50, 0.75, 1.0):
            if frac >= m:
                self.wake("milestone:%d" % int(m * 100), "MILESTONE", "%d of %d planned cells valid (%.1f %%)" % (len(valid), total_main, 100 * frac))
        for p in plans:
            if any(c["name"] in valid for c in p["doc"]["cells"]):
                self.wake("firstvalid:%s" % p["name"], "MILESTONE", "first valid certificate of plan %s" % p["name"])
        released = len(list([x for x in (self.base / "ppo_plans").glob("*.json") if not x.name.endswith(".plan.json")])) if (self.base / "ppo_plans").exists() else 0
        if released >= len(self.cfg["ppo"]["runs"]) > 0:
            self.wake("ppo:alltrained", "MILESTONE", "all %d PPO runs trained, gated and released" % released)
        head = ("G51 status %s (local %s) day_window=%s launches=%d valid_cells=%d/%d (%.2f %%) jobs_taken=%d/%d ppo_released=%d dry=%s"
                % (stamp(), now_local().strftime("%Y-%m-%d %H:%M"), self.day_window(), launches, len(valid), total_main, 100 * frac,
                   len(state.taken), len(state.jobs), released, self.dry))
        text = head + "\n" + "\n".join(lines) + "\n"
        write_new(self.status_dir / ("status_%s.txt" % stamp()), text)
        append(self.status_dir / "status_log.txt", head + "\n")
        print(text)


def POL_TASK_CODE():
    import campaign_cells as C
    return C.P.TASK_CODE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval-min", type=float, default=10.0)
    a = ap.parse_args(argv)
    lowprio()
    orch = Orchestrator(a.config, a.dry_run)
    while True:
        t0 = time.time()
        try:
            orch.poll()
        except Exception as exc:  # noqa: BLE001
            orch.log("POLL-EXCEPTION", error=repr(exc)[:500])
            orch.wake("exception:%s" % stamp()[:11], "ERROR", "orchestrator poll raised %r" % exc)
        if a.once:
            return 0
        time.sleep(max(60.0, a.interval_min * 60.0 - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
