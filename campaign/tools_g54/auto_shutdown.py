"""Owner 10-25: power off rental03/04/05/06/11/12 once their current wave is done ("等这波跑完再关机").

Every 10 min, for each listed host that is still up:
  * nothing of ours runs there (no dose executor, no training slot, no eval),
  * no unlaunched training retry is pinned to it,
  * every finished dose queue and training queue on it has been pulled to this PC by the orchestrator,
  * the above held on two consecutive checks (gives the orchestrator a cycle to pull).
Then it runs AutoDL's own `shutdown` in the container (stops the instance and its billing), marks the row in the
rental table as shut down (so the orchestrator stops probing it) and leaves a WAKE file. Never deletes anything."""
import json, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ORCH = Path(r"D:/claude code/codex_tasks/20261014_g52_v266_start/work/orchestrator")
TABLE = Path(r"E:/Storage/Projects/USVBench/租机/连接表_最新.md")
PULLS = Path(r"D:/claude code/codex_tasks/20261014_g51_campaign/work/r3copy_g51/launcher/pulls")
TPULLS = ORCH / "run" / "train_pulls"
WAKE = ORCH / "run" / "wake"
LOG = ORCH / "run" / "auto_shutdown_log.txt"
KEY = "C:/Users/BRADY/.ssh/usvb_rental_ed25519"
KH = "C:/Users/BRADY/.ssh/known_hosts_autodl"
HOSTS = ["rental03", "rental04", "rental05", "rental06", "rental11", "rental12"]
U = "/root/autodl-tmp/usvb"
# The pattern is assembled from quoted pieces so that the probe shell's own command line never matches it.
# 10-25: aborted training queues (refused at start, nothing produced) are not waited for; a run that matters still blocks via unreleased_here.
REMOTE = ("P=\"usvb\"\"exec=|g51_train\"\"_slot.sh|usvb\"\"train=|run_\"\"cell.py|eval_v6\"\"_frozen.py|eval_\"\"mppi.py|ppo_checkpoint\"\"_eval\"; "
          "echo BUSY $(pgrep -fc \"$P\"); "
          "echo GPU $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); "
          "for q in %s/dose_*; do [ -e $q/dose_FINISHED.txt ] || [ -e $q/dose_ABORTED.txt ] && echo DQ $(basename $q); done; "
          "for q in %s/ppo/*/formal/_train_*; do [ -e $q/train_FINISHED.txt ] && echo TQ $(basename $q); done" % (U, U))


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("%s %s\n" % (stamp(), msg))


def table_rows():
    out = {}
    for l in TABLE.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 7 and c[0].startswith("rental") and ":" in c[2]:
            out[c[0]] = c
    return out


def mark_off(hk):
    raw = TABLE.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"          # keep the table's own line endings
    lines = raw.splitlines()
    for i, l in enumerate(lines):
        c = l.split("|")
        if len(c) > 7 and c[1].strip() == hk:
            c[5] = " 关机 "
            c[7] = " %s 自动关机（owner 10-25：这波跑完关；数据已拉回） " % stamp()
            lines[i] = "|".join(c)
    TABLE.write_bytes((nl.join(lines) + nl).encode("utf-8"))


def pinned_pending(hk):
    recs = [json.loads(l) for l in (ORCH / "run" / "train_retry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    launched = set((r["run"], r["attempt"]) for r in recs if r.get("launched"))
    return [r["run"] for r in recs if not r.get("launched") and r.get("pin_host") == hk and (r["run"], r["attempt"]) not in launched]


def unreleased_here(hk):
    """Runs whose latest training ran on this host, not released and with no pending retry: their files may only exist here."""
    led = [json.loads(l) for l in (ORCH / "run" / "train_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ret = [json.loads(l) for l in (ORCH / "run" / "train_retry.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    released = set(r["run"] for r in led if r.get("released")) | set(p.stem for p in (ORCH / "run" / "ppo_plans").glob("*.json") if not p.name.endswith(".plan.json"))
    launched = set((r["run"], r["attempt"]) for r in ret if r.get("launched"))
    pending = set(r["run"] for r in ret if not r.get("launched") and (r["run"], r["attempt"]) not in launched)
    last = {}
    for r in sorted(led + [dict(x, host=x["host"]) for x in ret if x.get("launched")], key=lambda r: r["time_utc"]):
        if not r.get("released"):
            last[r["run"]] = r["host"]
    return sorted(run for run, h in last.items() if h == hk and run not in released and run not in pending)


def pulled_dose(q, hk):
    return any((d / q / "dose_FINISHED.txt").exists() or (d / q / "dose_ABORTED.txt").exists() for d in PULLS.glob("%s_%s_*" % (q, hk)))


def pulled_train(q, hk):
    return any(TPULLS.glob("%s_%s_*" % (q, hk)))


def check(hk, row):
    h, p = row[2].rsplit(":", 1)
    r = subprocess.run(["ssh", "-4", "-n", "-i", KEY, "-o", "UserKnownHostsFile=" + KH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        "-p", p, "root@" + h, REMOTE], capture_output=True, text=True, timeout=120)
    if "BUSY" not in r.stdout:
        return None, "unreachable"
    busy = int(re.search(r"BUSY (\d+)", r.stdout).group(1))
    gpu = int(re.search(r"GPU (\d+)", r.stdout).group(1)) if "GPU " in r.stdout else 99
    dq = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("DQ ")]
    tq = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("TQ ")]
    dq = [q for q in dq if q.startswith(("dose_c266-", "dose_p-", "dose_xmR-"))]   # campaign queues only (older sweeps/smokes were pulled elsewhere)
    miss = [q for q in dq if not pulled_dose(q, hk)] + [q for q in tq if not pulled_train(q, hk)]
    pend = pinned_pending(hk)
    unrel = unreleased_here(hk)
    ok = busy == 0 and gpu == 0 and not miss and not pend and not unrel
    return ok, "busy=%d gpu=%d unpulled=%s pinned=%s unreleased=%s" % (busy, gpu, miss[:4], pend, unrel)


def main():
    ready = {}
    while True:
        rows = table_rows()
        todo = [hk for hk in HOSTS if hk in rows and "运行中" in rows[hk][4]]
        if not todo:
            log("all listed hosts are off; exiting")
            return
        for hk in todo:
            try:
                ok, why = check(hk, rows[hk])
            except subprocess.TimeoutExpired:
                ok, why = None, "probe timeout"
            log("%s ok=%s %s" % (hk, ok, why))
            if not ok:
                ready.pop(hk, None)
                continue
            ready[hk] = ready.get(hk, 0) + 1
            if ready[hk] < 2:
                continue
            h, p = rows[hk][2].rsplit(":", 1)
            subprocess.run(["ssh", "-4", "-n", "-i", KEY, "-o", "UserKnownHostsFile=" + KH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                            "-p", p, "root@" + h, "nohup shutdown > /dev/null 2>&1 &"], capture_output=True, text=True, timeout=60)
            mark_off(hk)
            log("%s SHUTDOWN sent; table row marked 关机" % hk)
            (WAKE / ("WAKE_%s_SHUTDOWN.txt" % stamp())).write_text("kind=SHUTDOWN\nkey=shutdown:%s\nhost %s powered off after its last wave (data pulled)\n" % (hk, hk), encoding="utf-8")
            ready.pop(hk, None)
        time.sleep(600)


if __name__ == "__main__":
    main()
