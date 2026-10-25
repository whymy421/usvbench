"""Owner 10-25: keep the campaign's own tools running even when the Claude session that started them ends.

Started detached (WMI process create), it runs each tool below as a child and restarts any that exits, after 60 s.
Edit TOOLS and restart the keeper to change versions. Writes run/keeper_log.txt. Stop it by killing keeper.py."""
import subprocess, time
from datetime import datetime, timezone
from pathlib import Path

W = Path(r"D:/claude code/codex_tasks/20261014_g52_v266_start/work/orchestrator")
PY = r"C:\Python313\python.exe"
TOOLS = {
    "orchestrator": [PY, "-B", str(W / "g52_orchestrator_fix32.py"), "--config", str(W / "campaign_config_v266_final_fix32.json"), "--interval-min", "10"],
    "eta_watch": [PY, "-B", str(W / "tools_g54" / "train_eta_watch.py")],
    "health_watch": [PY, "-B", str(W / "tools_g54" / "health_watch.py")],
    "auto_shutdown": [PY, "-B", str(W / "tools_g54" / "auto_shutdown.py")],
}
ONESHOT = {"auto_shutdown"}      # exits on its own once all its hosts are off; not restarted after a clean exit
LOG = W / "run" / "keeper_log.txt"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("%s %s\n" % (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), msg))


def start(name):
    out = open(W / ("run_%s_keeper_console.txt" % name), "a", encoding="utf-8")
    p = subprocess.Popen(TOOLS[name], cwd=str(W), stdout=out, stderr=subprocess.STDOUT, creationflags=0x08000000)  # CREATE_NO_WINDOW
    log("start %s pid=%d" % (name, p.pid))
    return p


def main():
    procs = {n: start(n) for n in TOOLS}
    done = set()
    beat = 0
    while True:
        time.sleep(60)
        beat += 1
        if beat % 60 == 0:
            log("alive; running=%s" % sorted(procs))
        try:
            loop_once(procs, done)
        except Exception as exc:  # noqa: BLE001  (10-25: the first keeper died silently; never let one bad tick end it)
            log("tick error %r" % (exc,))


def loop_once(procs, done):
    if True:
        for n, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            if n in ONESHOT and rc == 0:
                log("%s finished (rc=0), not restarted" % n)
                done.add(n)
                del procs[n]
                continue
            log("%s exited rc=%s; restarting" % (n, rc))
            procs[n] = start(n)


if __name__ == "__main__":
    import atexit, sys
    sys.stdout = sys.stderr = open(W / "run" / "keeper_stderr.txt", "a", encoding="utf-8", buffering=1)   # tracebacks land here
    atexit.register(lambda: log("keeper exiting (atexit)"))
    log("keeper start pid=%d" % __import__("os").getpid())
    main()
