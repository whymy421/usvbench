"""G54 one-off: pull + validate the queues stranded on rental10 / rental13 (power-off), print each queue's resume order."""
import sys, json, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("orch", str(Path(__file__).resolve().parent.parent / "g52_orchestrator_fix16.py"))
orch = importlib.util.module_from_spec(spec); spec.loader.exec_module(orch)
o = orch.Orchestrator(str(Path(__file__).resolve().parent.parent / "campaign_config_v266_final_fix16.json"), False)
plans = o.plans()
# REDACTED for the public repository: the rental ssh gateway host and the two ssh ports (set to 0).
jobs = {
    ("rental13", 0): ["dose_c266-MB_WAMV_MP_F1-b9_a1", "dose_c266-RG_WAMV_MP_F1-b8_a1"],
    ("rental10", 0): ["dose_c266-HM_WAMV_CL_F1-b5_a1", "dose_c266-MB_WAMV_CL_F1-b5_a1"],
}
for (hk, port), qs in jobs.items():
    host = {"key": hk, "ssh_host": "REDACTED_RENTAL_SSH_HOST", "port": port, "class": "RENTAL", "K": 2, "enabled": False}
    for q in qs:
        o.pull_and_validate(host, q, {"state": "R", "problems": 0}, plans, recover=True)
        d = o.pulled_dirs(q, host, any_state=True)
        prog = (d[-1] / "progress_dose.txt").read_text(encoding="ascii", errors="replace").splitlines()
        ok = set(l.split()[3] for l in prog if len(l.split()) > 3 and l.split()[2] == "SUCCESS")
        starts = [(l.split()[3], int(l.split("order=")[1].split()[0])) for l in prog if len(l.split()) > 3 and l.split()[2] == "START-CELL"]
        todo = [od for n, od in starts if n not in ok]
        print("RESULT", hk, q, "done", len(ok), "resume_order", todo[0] if todo else "next-after-%d" % max(od for _, od in starts))
