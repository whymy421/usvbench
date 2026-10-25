"""G54 training-timeout early warning (owner 10-19): every 2 h, for every running formal PPO training on every usable rental
(addresses from the rental table), project the total training wall time from its timeline (wall_s / percent) and compare
with the queue's time limit (START-CELL TIMEOUT_<run>=<ms>). Writes a WAKE_*_ETA.txt into the orchestrator wake folder
once per (queue) when the projection exceeds 90 % of the limit. Read-only on the rentals; never kills anything."""
import json, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ORCH = Path(r"D:/claude code/codex_tasks/20261014_g52_v266_start/work/orchestrator")
TABLE = Path(r"E:/Storage/Projects/USVBench/租机/连接表_最新.md")
WAKE = ORCH / "run" / "wake"
SEEN = ORCH / "run" / "eta_warned.txt"
LOG = ORCH / "run" / "eta_watch_log.txt"
KEY = "C:/Users/BRADY/.ssh/usvb_rental_ed25519"
KH = "C:/Users/BRADY/.ssh/known_hosts_autodl"
FORMAL = "/root/autodl-tmp/usvb/ppo/2c04042a7a99-64e5932b737f/formal"
REMOTE = ("for q in /root/autodl-tmp/usvb/ppo/*/formal/_train_*; do [ -e $q/train_FINISHED.txt ] && continue; [ -e $q/train_ABORTED.txt ] && continue; "
          "s=$(grep -m1 START-CELL $q/progress_train.txt); [ -z \"$s\" ] && continue; "
          "echo \"Q $(basename $q) | $s | $(tail -1 $q/timeline.csv)\"; done")


def hosts():
    out = {}
    for l in TABLE.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 7 and c[0].startswith("rental") and "运行中" in c[4] and ":" in c[2]:
            out[c[0]] = c[2].rsplit(":", 1)
    return out


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def once():
    seen = set(SEEN.read_text(encoding="ascii").split()) if SEEN.exists() else set()
    lines = []
    for hk, (h, p) in sorted(hosts().items()):
        try:
            r = subprocess.run(["ssh", "-4", "-i", KEY, "-o", "UserKnownHostsFile=" + KH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                                "-p", p, "root@" + h, REMOTE], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        for l in r.stdout.splitlines():
            if not l.startswith("Q "):
                continue
            q, start, tl = [x.strip() for x in l[2:].split("|", 2)]
            m = re.search(r"START-CELL (\S+) TIMEOUT_\S+=(\d+)", start)
            f = tl.split(",")
            if not m or len(f) < 8 or f[3] != "progress.log":
                continue
            run, limit_s = m.group(1), int(m.group(2)) / 1000.0
            try:
                pct, wall = float(f[5]), float(f[7])
            except ValueError:
                continue
            if pct < 3:
                continue
            try:   # queues left behind by the 10-17 power-off are not running: skip timelines older than 2 h
                age = (datetime.now(timezone.utc) - datetime.strptime(f[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds()
            except ValueError:
                age = 0
            if age > 7200:
                continue
            proj = wall / pct * 100.0
            lines.append("%s %s %s %s %.0f%% wall=%.1fh projected=%.1fh limit=%.1fh" % (stamp(), hk, run, q, pct, wall / 3600, proj / 3600, limit_s / 3600))
            if proj > 0.9 * limit_s and q not in seen:
                (WAKE / ("WAKE_%s_ETA.txt" % stamp())).write_text(
                    "kind=ETA\nkey=eta:%s\ntraining %s on %s is at %.0f%% after %.1f h; projected %.1f h vs time limit %.1f h (queue %s)\n"
                    % (q, run, hk, pct, wall / 3600, proj / 3600, limit_s / 3600, q), encoding="ascii")
                with open(SEEN, "a", encoding="ascii") as fh:
                    fh.write(q + "\n")
    with open(LOG, "a", encoding="ascii") as fh:
        fh.write("".join(x + "\n" for x in lines) or "%s no running trainings seen\n" % stamp())


def idle_check():
    """Wake when a host shows an idle slot in each of the last 3 orchestrator status files (silent idling)."""
    sdir = ORCH / "run" / "status"
    files = sorted(sdir.glob("status_2*.txt"))[-3:]
    if len(files) < 3:
        return
    idle = None
    for f in files:
        cur = set()
        for l in f.read_text(encoding="ascii", errors="replace").splitlines():
            m = re.match(r"(rental\d+) .*idle_slots=\[(\d[^\]]*)\]", l)
            if m:
                cur.add(m.group(1))
        idle = cur if idle is None else idle & cur
    seen = set(SEEN.read_text(encoding="ascii").split()) if SEEN.exists() else set()
    hold_f = ORCH / "run" / "hold_hosts.txt"      # held hosts are idle on purpose (awaiting re-qualification)
    held = set(hold_f.read_text(encoding="ascii").split()) if hold_f.exists() else set()
    # a host that received a launch in the last 40 min is being refilled (status lines are written before dispatch)
    recent = set()
    cutoff = (datetime.now(timezone.utc).timestamp() - 2400)
    try:
        for l in (ORCH / "run" / "events.jsonl").read_text(encoding="ascii", errors="replace").splitlines()[-400:]:
            if '"kind": "LAUNCH"' in l or '"kind": "TRAIN-LAUNCH"' in l:
                r = json.loads(l)
                t = datetime.strptime(r["time_utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
                if t >= cutoff:
                    recent.add(r.get("host"))
    except (OSError, ValueError, KeyError):
        pass
    for hk in sorted((idle or set()) - held - recent):
        key = "idle-%s-%s" % (hk, files[-1].stem[-16:-5])     # at most one wake per host per hour
        if key in seen:
            continue
        (WAKE / ("WAKE_%s_IDLESLOT.txt" % stamp())).write_text(
            "kind=IDLESLOT\nkey=%s\nhost %s had an idle slot in the last 3 status files (%s)\n" % (key, hk, ", ".join(x.name for x in files)),
            encoding="ascii")
        with open(SEEN, "a", encoding="ascii") as fh:
            fh.write(key + "\n")


if __name__ == "__main__":
    n = 0
    while True:
        if n % 12 == 0:
            once()
        idle_check()
        n += 1
        time.sleep(600)
