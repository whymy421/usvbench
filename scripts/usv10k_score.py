"""USV-10K v1.0 scoring over certification records.

Frozen decisions (owner, 2026-08-13): the 10000 ceiling is the PHYSICAL-LIMIT
reading (A = E = Q = 1 means shortest feasible route, flat-out speed, zero
contact; structurally unexceedable); beta = 0.10; failure-branch proximity
credit requires the min-geodesic-remaining field, which certification does not
record yet, so in this v1 table a failed episode earns its milestone credit
only (staged missions) or zero -- disclosed per row in the "basis" column.

    Score_i = 10000 * A_i * E_i * Q_i
    10000 - Score_i = R_task + R_eff + R_safe   (exact, no cross terms)

A: success -> 1; staged missions -> 0.2*(phase>=1) + 0.4*(phase>=2) on failure.
E (success branch only): eta_d * eta_t, with eta_d = L*/max(p, L*) against the
   per-episode feasible geodesic, and eta_t = T*/max(t, T*), T* = L*/V_MAX.
Q: hull penetration (min clearance < 0) -> 0; linear ramp to 1 at 0.20 m;
   clearance above 0.20 m earns nothing (rewarding clearance rewards detours).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

V_MAX = 2.991      # m/s, drag-limited top speed from the hydro calibration
C_SOFT = 0.20      # m, the collision margin; clearance beyond this earns nothing
BETA = 0.10        # failure-branch cap (unused in v1: proximity not recorded)


def episode_score(rec, spawn_phase=0):
    success = bool(rec.get("success"))
    phase = rec.get("max_phase")
    if success:
        attain = 1.0
    elif phase is not None:
        # Milestones are only worth points if the boat EARNED them: a staged
        # spawn (HarborDockPhase starts at phase 2) hands the early phases
        # out for free, and the first draft of this table paid 6000 for
        # standing still because of exactly that.
        attain = (0.2 * (phase >= 1 and 1 > spawn_phase)
                  + 0.4 * (phase >= 2 and 2 > spawn_phase))
    else:
        attain = 0.0

    geodesic = rec.get("route_geodesic_m") or 0.0
    reference = geodesic if geodesic > 0 else (rec.get("d0_m") or 0.0)
    basis = "geo" if geodesic > 0 else ("d0" if reference > 0 else "none")

    efficiency = 1.0
    if success and reference > 0:
        path = max(rec.get("path_length_m") or reference, 1e-6)
        eta_d = reference / max(path, reference)
        tts = rec.get("tts_s")
        if tts:
            t_star = reference / V_MAX
            eta_t = t_star / max(tts, t_star)
        else:
            eta_t = 1.0
        efficiency = eta_d * eta_t

    clearance = rec.get("min_clearance_m")
    if clearance is None:
        safety = 1.0
    elif clearance < 0.0:
        safety = 0.0
    elif clearance < C_SOFT:
        safety = clearance / C_SOFT
    else:
        safety = 1.0

    score = 10000.0 * attain * efficiency * safety
    return {
        "score": score,
        "r_task": 10000.0 * (1.0 - attain),
        "r_eff": 10000.0 * attain * (1.0 - efficiency),
        "r_safe": 10000.0 * attain * efficiency * (1.0 - safety),
        "basis": basis,
    }


def main(root):
    rows = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "gcert_*.json"))):
        name = os.path.basename(path)
        match = re.fullmatch(r"gcert_(.+)_e(\d+)\.json", name)
        if not match:
            continue
        policy, seed = match.group(1), match.group(2)
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        records = data.get("records") or []
        if not records:
            continue
        spawn = 2 if "DockPhase" in data.get("task", "") else 0
        scored = [episode_score(r, spawn) for r in records]
        n = len(scored)
        mean = lambda key: sum(s[key] for s in scored) / n
        bases = {s["basis"] for s in scored}
        # A family with no efficiency reference in its records scores on
        # attainment alone; label it so nobody reads a flat 10000 as
        # "physically perfect" -- for these rows E is simply not measured yet.
        if bases == {"none"}:
            bases = {"A-only"}
        rows[policy][seed] = {
            "n": n,
            "sr": sum(1 for r in records if r.get("success")) / n,
            "score": mean("score"),
            "r_task": mean("r_task"),
            "r_eff": mean("r_eff"),
            "r_safe": mean("r_safe"),
            "basis": "/".join(sorted(bases)),
            "task": data.get("task", "?"),
        }

    order = sorted(rows, key=lambda p: -rows[p].get("123", rows[p].get("42", {})).get("score", 0))
    print(f"{'policy':18s} {'task':42s} {'SR123':>6s} {'e123':>6s} {'e42':>6s} "
          f"{'R_task':>7s} {'R_eff':>6s} {'R_safe':>6s}  basis")
    out_rows = []
    for policy in order:
        head = rows[policy].get("123") or rows[policy].get("42")
        alt = rows[policy].get("42") if "123" in rows[policy] else None
        print(f"{policy:18s} {head['task'][:42]:42s} {head['sr']:6.1%} "
              f"{head['score']:6.0f} {alt['score'] if alt else float('nan'):6.0f} "
              f"{head['r_task']:7.0f} {head['r_eff']:6.0f} {head['r_safe']:6.0f}  {head['basis']}")
        out_rows.append({"policy": policy, **{f"e{s}": v for s, v in rows[policy].items()}})
    with open(os.path.join(root, "usv10k_scores.json"), "w", encoding="utf-8") as handle:
        json.dump(out_rows, handle, indent=1)
    print(f"\n-> {os.path.join(root, 'usv10k_scores.json')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\BRADY\usvbench\gcerts")
