#!/usr/bin/env python3
"""G51: convert the timing agent's work/timing/timing_model.json into the planner's per-cell table timing_model_g51.json.

Per (task code, vessel, CL|MP|PPO) and per condition (G51 names: calm, waves_D1.., app_D1.., allsuites, allsuites_dynD2) the
per-slot hours of ONE 64 x 2 cell, [central, low, high], for RENTAL (K=2), 4080 (K=2), 5070 (K=3):
  RENTAL = per_condition_rental_s (measured rental 64 x 2, else scaled 32 x 1 smoke, else inferred; source tag kept);
  4080 / 5070 = RENTAL x the host factor of the controller class (fits H4080_cl 1.01 [0.85, 1.16], H4080_mp 1.48 [1.00, 1.62],
  H5070 cl 2.42 [2.0, 3.5], mp 2.2 [1.87, 2.9]); low uses low x low, high uses high x high;
  PPO = classical x 1.10 [0.50, 4.16] (timing_model.md: two BlueBoat calm cells; UNVERIFIED), host factors as classical.
Never overwrites."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign_cells as C  # noqa: E402
P = C.P
SRC = HERE.parent / "timing" / "timing_model.json"
OUT = HERE.parent / "timing" / "timing_model_g51.json"
HOSTF = {"cl": {"4080": (1.014, 0.852, 1.163), "5070": (2.42, 2.0, 3.5)},
         "mp": {"4080": (1.48, 1.001, 1.615), "5070": (2.2, 1.87, 2.9)}}
PPO_F = (1.10, 0.50, 4.16)


def main():
    raw = SRC.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    pc = doc["per_condition_rental_s"]
    cells = {}
    for (task, vkey, ctrl) in [(t, v, c) for t in P.TASKS for v in ("bb", "wamv") for c in ("cl", "mp")]:
        src = pc["%s|%s|%s" % (task, vkey, ctrl)]
        table = {}
        for cond, val in src.items():
            name = cond.replace("appearing_", "app_")
            mid, lo, hi = val[0] / 3600.0, val[1] / 3600.0, val[2] / 3600.0
            entry = {"RENTAL": [mid, lo, hi], "source": val[3] if len(val) > 3 else ""}
            for host in ("4080", "5070"):
                f = HOSTF[ctrl][host]
                entry[host] = [mid * f[0], lo * f[1], hi * f[2]]
            table[name] = entry
        table["allsuites_dynD2"] = table["allsuites"]
        key = "%s_%s_%s" % (P.TASK_CODE[task], vkey.upper(), ctrl.upper())
        cells[key] = table
        if ctrl == "cl":
            ppo = {}
            for name, entry in table.items():
                ppo[name] = {h: [entry[h][0] * PPO_F[0], entry[h][1] * PPO_F[1], entry[h][2] * PPO_F[2]] for h in ("RENTAL", "4080", "5070")}
                ppo[name]["source"] = "classical x %.2f [%.2f, %.2f] (UNVERIFIED)" % PPO_F
            cells["%s_%s_PPO" % (P.TASK_CODE[task], vkey.upper())] = ppo
    out = {"format": "g51-timing-v1", "source": str(SRC), "source_sha256": hashlib.sha256(raw).hexdigest().upper(),
           "units": "per-slot hours of one 64 x 2 cell at the class's operational K (RENTAL 2, 4080 2, 5070 3): [central, low, high]",
           "host_factors": HOSTF, "ppo_factor": PPO_F, "ppo_training": doc.get("ppo"), "cells": cells}
    data = (json.dumps(out, indent=1, sort_keys=True) + "\n").encode("ascii")
    if OUT.exists():
        if OUT.read_bytes() == data:
            print("SAME %s" % OUT)
            return 0
        raise SystemExit("refusing to overwrite %s" % OUT)
    with open(OUT, "xb") as handle:
        handle.write(data)
    print("WROTE %s sha256=%s" % (OUT, hashlib.sha256(data).hexdigest().upper()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
