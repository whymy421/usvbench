"""USV-10K scoring over v3 certificates, with explicit component availability.

Frozen decisions (owner, 2026-08-13), unchanged: the 10000 ceiling is the
PHYSICAL-LIMIT reading (A = E = Q = 1 means shortest feasible route, flat-out
speed, zero contact); beta = 0.10; a failed episode earns its milestone
credit only (staged missions) or zero.

    Score_i = 10000 * A_i * E_i * Q_i
    10000 - Score_i = R_task + R_eff + R_safe   (exact, no cross terms)

A: success -> 1; staged missions -> 0.2*(phase>=1) + 0.4*(phase>=2) on
   failure, counting only phases the boat earned (spawn_phase from the cfg).
E (success branch only): eta_d * eta_t, eta_d = L*/max(p, L*) with L* the
   feasible geodesic (else d0) and p the SUCCESS-PREFIX path (finding 60);
   eta_t = T*/max(t, T*), T* = L*/V_ref. Failed episodes keep the v1
   convention E = 1 (labelled 'failure-convention').
Q: clearance < 0 -> 0; linear ramp to 1 at 0.20 m; beyond 0.20 m no credit.

v3plus changes (C1 finding 51): a missing or non-finite component is
UNAVAILABLE (null), never perfect credit; NaN is rejected; V_ref is the
certificate's per-vessel speed_reference (header.speed_reference, basis
'nominal' = vessel-spec terminal speed or 'condition_adjusted'), never the
BlueBoat 2.991 m/s constant for every hull. A cell with any unavailable
component has score null ('partial') and is never ranked beside full scores.
Output rows are sorted by file name, not ranked.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from analysis_input import (AnalysisInputError, finite_number, load_analysis_input,
                            speed_reference, validate_analysis_input,
                            validate_compatible, validate_record_metrics)

C_SOFT = 0.20
SCORE_VERSION = "usv10k-v3plus-availability-1"


def episode_score(rec, spawn_phase=0, *, speed_reference_m_s=None):
    validate_record_metrics(rec)
    if type(spawn_phase) is not int or spawn_phase < 0:
        raise AnalysisInputError("spawn_phase must be a nonnegative integer")
    if speed_reference_m_s is not None:
        finite_number(speed_reference_m_s, "speed reference", positive=True)
    success = rec["success"]
    phase = rec.get("max_phase")
    if phase is None and isinstance(rec.get("stage_attainment"), dict):
        phase = rec["stage_attainment"].get("max_phase")
    if phase is not None and (type(phase) is not int or phase < 0):
        raise AnalysisInputError("max_phase must be a nonnegative integer")
    if success:
        attain = 1.0
    elif phase is not None:
        attain = (0.2 * (phase >= 1 and 1 > spawn_phase)
                  + 0.4 * (phase >= 2 and 2 > spawn_phase))
    else:
        attain = 0.0

    geodesic, distance = rec.get("route_geodesic_m"), rec.get("d0_m")
    reference = geodesic if geodesic is not None and geodesic > 0 else distance
    basis = "geo" if geodesic is not None and geodesic > 0 else "d0"
    if reference is None or reference <= 0:
        reference, basis = None, "unavailable"

    unavailable = []
    efficiency = 1.0 if not success else None
    efficiency_basis = "failure-convention" if not success else "success-prefix"
    if success:
        path = rec.get("path_length_success_prefix_m")
        # C5's full episode is also its prefix only when success terminates it.
        # A new explicit null prefix never falls back to the legacy field.
        if "path_length_success_prefix_m" not in rec and rec.get("termination_reason") == "success":
            path = rec.get("path_length_m")
            efficiency_basis = "terminal-success-full-equals-prefix"
        tts = rec.get("tts_s")
        for name, value in (("route_reference", reference), ("success_prefix_path", path),
                            ("tts", tts), ("speed_reference", speed_reference_m_s)):
            if value is None or (name != "tts" and value <= 0):
                unavailable.append(name)
        if tts is not None and tts < 0:
            raise AnalysisInputError("tts_s cannot be negative")
        if not unavailable:
            t_star = reference / speed_reference_m_s
            efficiency = reference / max(path, reference) * t_star / max(tts, t_star)

    clearance = rec.get("min_clearance_m")
    safety = None if clearance is None else min(1.0, max(0.0, clearance / C_SOFT))
    if safety is None:
        unavailable.append("clearance")
    complete = efficiency is not None and safety is not None
    score = 10000.0 * attain * efficiency * safety if complete else None
    return {
        "score": score,
        "complete": complete,
        "attainment": attain,
        "efficiency": efficiency,
        "safety": safety,
        "r_task": 10000.0 * (1.0 - attain),
        "r_eff": None if efficiency is None else 10000.0 * attain * (1.0 - efficiency),
        "r_safe": None if not complete else 10000.0 * attain * efficiency * (1.0 - safety),
        "basis": basis,
        "efficiency_basis": efficiency_basis,
        "unavailable_components": unavailable,
    }


def score_certificate(document):
    validate_analysis_input(document)
    header = document["header"]
    reference = speed_reference(document)
    cfg = header["resolved_config"]
    # Harbour cfgs declare spawn_phase (tasks/harbor_mission/harbor_mission_env_cfg.py:188);
    # every other family has no staged spawn. Never inferred from the task name.
    spawn = cfg.get("spawn_phase", 0)
    scored = [episode_score(record, spawn,
                            speed_reference_m_s=None if reference is None else reference["value_m_s"])
              for record in document["records"]]
    n = len(scored)

    def available_mean(key):
        values = [row[key] for row in scored]
        return None if any(value is None for value in values) else sum(values) / n

    complete = all(row["complete"] for row in scored) and reference is not None
    return {
        "score_version": SCORE_VERSION,
        "task": header["task_id"], "vessel": header["vessel"],
        "controller": header["controller"]["name"],
        "checkpoint_sha256": header["checkpoint"]["sha256"],
        "eval_seed": header["eval_seed"], "level": header["level"],
        "n": n, "n_success": sum(record["success"] for record in document["records"]),
        "sr": sum(record["success"] for record in document["records"])/n,
        "score_complete": complete,
        "score": available_mean("score") if complete else None,
        "r_task": available_mean("r_task"), "r_eff": available_mean("r_eff"),
        "r_safe": available_mean("r_safe"),
        "n_complete_episodes": sum(row["complete"] for row in scored),
        "speed_reference": reference,
        "basis": sorted({row["basis"] for row in scored}),
        "unavailable_components": sorted({name for row in scored for name in row["unavailable_components"]}
                                          | ({"speed_reference"} if reference is None else set())),
    }


def main(root):
    paths = sorted(glob.glob(os.path.join(root, "gcert_*.json")))
    documents = [load_analysis_input(path) for path in paths]
    validate_compatible(documents)
    rows = [{"file": os.path.basename(path), **score_certificate(doc)}
            for path, doc in zip(paths, documents)]
    print("file  vessel  task  n  successes  score  availability")
    for row in rows:
        score = "unavailable" if row["score"] is None else f"{row['score']:.1f}"
        print(f"{row['file']}  {row['vessel']}  {row['task']}  {row['n']}  "
              f"{row['n_success']}  {score}  "
              f"{'full' if row['score_complete'] else 'partial'}")
    output = os.path.join(root, "usv10k_scores.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("-> " + output)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: usv10k_score.py CERTIFICATE_DIRECTORY")
    try:
        raise SystemExit(main(sys.argv[1]))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("[reject] " + str(exc), file=sys.stderr)
        raise SystemExit(2)
