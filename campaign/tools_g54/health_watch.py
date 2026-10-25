"""Owner 10-25 ("鲁棒一点"): two alarms the campaign lacked, every 10 min, read-only, WAKE files into run/wake.

1. IDLE-SHARE: over the last 6 orchestrator status files (about one hour), more than 20 % of the slots on non-reserved
   hosts were idle in every file. (Status lines are written before dispatch, so one busy cycle does not trigger it.)
2. STUCK-RUN: a PPO run that was launched, is not released, has no pending retry, and whose last training host shows
   no training running for at least 2 h (the FO-WAMV runs sat like this for 10 days after a gate failure)."""
import json, re, time
from datetime import datetime, timezone
from pathlib import Path

ORCH = Path(r"D:/claude code/codex_tasks/20261014_g52_v266_start/work/orchestrator")
RUN = ORCH / "run"
SEEN = RUN / "health_warned.txt"
LOG = RUN / "health_watch_log.txt"


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ts(s):
    return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def wake(key, kind, text):
    seen = set(SEEN.read_text(encoding="ascii").split()) if SEEN.exists() else set()
    if key in seen:
        return
    (RUN / "wake" / ("WAKE_%s_%s.txt" % (stamp(), kind))).write_text("kind=%s\nkey=%s\n%s\n" % (kind, key, text), encoding="utf-8")
    with open(SEEN, "a", encoding="ascii") as fh:
        fh.write(key + "\n")


def status_files(n):
    return sorted((RUN / "status").glob("status_2*.txt"))[-n:]


def idle_share():
    reserved = set((RUN / "exclusive_hosts.txt").read_text(encoding="ascii").split()) if (RUN / "exclusive_hosts.txt").exists() else set()
    shares = []
    for f in status_files(6):
        tot = idle = 0
        for l in f.read_text(encoding="ascii", errors="replace").splitlines():
            m = re.match(r"(rental\d+) .*idle_slots=\[([^\]]*)\]", l)
            if m and m.group(1) not in reserved:
                tot += 2
                idle += len([x for x in m.group(2).split(",") if x.strip()])
        if tot:
            shares.append(idle / float(tot))
    if len(shares) == 6 and min(shares) > 0.20:
        hour = stamp()[:11]
        wake("idleshare-%s" % hour, "IDLESHARE", "idle share above 20 %% in each of the last 6 status files: %s"
             % ", ".join("%.0f%%" % (100 * s) for s in shares))
    return shares


def host_training(hk):
    f = status_files(1)
    if not f:
        return True
    for l in f[0].read_text(encoding="ascii", errors="replace").splitlines():
        if l.startswith(hk + " ") and "train=" in l:      # reserved hosts also get an "EXCLUSIVE" line without train=
            m = re.search(r"train=\[([^\]]*)\]", l)
            return bool(m and m.group(1).strip())
    return False        # host not in the status at all (off): nothing of ours runs there


def stuck_runs():
    led = [json.loads(l) for l in (RUN / "train_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ret = [json.loads(l) for l in (RUN / "train_retry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    released = set(r["run"] for r in led if r.get("released")) | set(p.stem for p in (RUN / "ppo_plans").glob("*.json") if not p.name.endswith(".plan.json"))
    launched_att = set((r["run"], r["attempt"]) for r in ret if r.get("launched"))
    pending = set(r["run"] for r in ret if not r.get("launched") and (r["run"], r["attempt"]) not in launched_att)
    last = {}
    for r in led:
        if not r.get("released"):
            last[r["run"]] = r
    for r in ret:
        if r.get("launched"):
            last[r["run"]] = {"run": r["run"], "host": r["host"], "time_utc": r["time_utc"]}
    now = time.time()
    out = []
    for run, r in sorted(last.items()):
        if run in released or run in pending or now - ts(r["time_utc"]) < 7200:
            continue
        if not host_training(r["host"]):
            out.append(run)
            wake("stuck-%s-%s" % (run, stamp()[:8]), "STUCKRUN", "PPO run %s: launched on %s at %s, not released, no pending retry, "
                 "and no training is running on that host" % (run, r["host"], r["time_utc"]))
    return out


if __name__ == "__main__":
    while True:
        try:
            sh = idle_share()
            st = stuck_runs()
            with open(LOG, "a", encoding="ascii") as fh:
                fh.write("%s idle=%s stuck=%s\n" % (stamp(), ",".join("%.0f" % (100 * s) for s in sh), ",".join(st)))
        except Exception as exc:  # noqa: BLE001
            with open(LOG, "a", encoding="ascii") as fh:
                fh.write("%s ERROR %r\n" % (stamp(), exc))
        time.sleep(600)
