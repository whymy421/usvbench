"""Pre-declared MPPI per-vessel horizon / temperature procedure (F7 round 4b; CPU only).

Implements F7 work/MPPI_TUNING_PROCEDURE.md (copy: scripts/MPPI_TUNING_PROCEDURE_V2.md),
declared before the first run; its LF-normalised SHA-256 is stamped into the output. The
calm nominal registry model (mppi_controller.VesselDynamicsV2) is the ONLY world; the
controller is the evaluators' MPPIController with the eval_mppi parser defaults after the
declared per-vessel scaling (mppi_params_for_vessel). Nothing here touches Isaac, an
evaluation seed, a journal layout or an evaluation pack.

  one grid point (a worker):
    python -B scripts/tune_mppi_v2.py point --vessel V --lambda L --out <dir>/V_L.json
  all points in parallel, then freeze:
    python -B scripts/tune_mppi_v2.py all --procedure <proc.md> --points <dir> \
        --out scripts/mppi_params_v2.json --evidence <work>/evidence/mppi_tuning_v2_evidence.json
  a re-freeze keeps the superseded record (G4-A5):
    ... all ... --supersedes <old mppi_params_v2.json> --supersede-reason "<why>"
  The frozen file records the tuning model's planar damping per vessel (``model_damping``,
  mppi_controller.model_damping_record); journal_mppi_params refuses a file whose record
  differs from the current calm nominal model.
  the reported WAM-V long-abeam case (procedure section 6):
    python -B scripts/tune_mppi_v2.py abeam --lambda L [--H H] --out <file>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

FORMAT = "usvbench-mppi-params-v1"
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
LAMBDA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
SHIPPED = {"H": 60, "lambda": 0.3}
GOAL_BEARINGS_DEG = (45.0, 90.0, 135.0, 180.0)
GOAL_DISTANCE = 12.0
O0_GOAL = (20.0, 0.0)
O0_OBSTACLE = (10.0, 0.3, 1.5)
RADIUS = 2.0
CRUISE = 1.0
REPLICAS = 2
R1_TOLERANCE = 0.10
CONTROL_HZ = 60


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def scenarios(scale):
    """[(name, goal (x, y), obstacle (x, y, r) or None, deadline_s)] of procedure section 3."""
    out = []
    for bearing in GOAL_BEARINGS_DEG:
        a = math.radians(bearing)
        goal = (GOAL_DISTANCE * scale * math.cos(a), GOAL_DISTANCE * scale * math.sin(a))
        out.append(("G%d" % int(bearing), goal, None, 2.0 * GOAL_DISTANCE * scale / CRUISE + 10.0))
    goal = (O0_GOAL[0] * scale, O0_GOAL[1] * scale)
    obstacle = tuple(v * scale for v in O0_OBSTACLE)
    out.append(("O0", goal, obstacle, 2.0 * math.hypot(*goal) / CRUISE + 10.0))
    return out


def params_for(vessel, lam, horizon=None):
    """eval_mppi parser defaults (== MPPIParams defaults) + declared scaling, lambda set."""
    from dataclasses import replace

    import mppi_controller as mc
    params = mc.mppi_params_for_vessel(vessel, mc.MPPIParams(lambda_=float(lam)))
    if horizon is not None and int(horizon) != params.H:
        params = replace(params, H=int(horizon))
        params.validate()
    return params


def simulate(vessel, lam, rows, horizon=None, threads=2):
    """Run rows [(name, goal, obstacle, deadline)] in ONE batch; return per-row results."""
    import torch

    import mppi_controller as mc
    torch.set_num_threads(int(threads))
    scale = mc.vessel_length_scale(vessel)
    radius = RADIUS * scale
    params = params_for(vessel, lam, horizon)
    n = len(rows)
    ctrl = mc.MPPIController(n, params=params, dynamics=mc.VesselDynamicsV2(vessel))
    plant = mc.VesselDynamicsV2(vessel, dtype=torch.float64)
    state = torch.zeros((n, 6), dtype=torch.float64)
    targets = torch.tensor([[list(r[1])] for r in rows], dtype=torch.float32)
    centers = torch.zeros((n, 1, 2), dtype=torch.float32)
    radii = torch.zeros((n, 1), dtype=torch.float32)
    active = torch.zeros((n, 1), dtype=torch.bool)
    for i, r in enumerate(rows):
        if r[2] is not None:
            centers[i, 0, 0], centers[i, 0, 1], radii[i, 0] = r[2]
            active[i, 0] = True
    goals64 = targets[:, 0, :].to(torch.float64)
    deadline_steps = [int(math.ceil(r[3] * CONTROL_HZ - 1e-9)) for r in rows]
    result = [{"scenario": r[0], "row": i, "success": None, "time_s": None, "collision": False,
               "min_clearance_m": None, "final_distance_m": None, "ess_sum": 0.0, "ess_steps": 0}
              for i, r in enumerate(rows)]
    open_rows = set(range(n))
    step = 0
    t0 = time.time()
    while open_rows and step < max(deadline_steps):
        action, info = ctrl.rollout(state.to(torch.float32), targets=targets, goal_radius=radius, mode="goal",
                                    obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active)
        ess = info["ess"].tolist()
        state = plant.step(state, action.to(torch.float64))
        step += 1
        dist = torch.linalg.vector_norm(state[:, :2] - goals64, dim=-1).tolist()
        clear = plant.hull_clearance(state[:, None, :2], state[:, None, 2], centers.to(torch.float64),
                                     radii.to(torch.float64))[:, 0, 0].tolist()
        for i in sorted(open_rows):
            rec = result[i]
            rec["ess_sum"] += float(ess[i])
            rec["ess_steps"] += 1
            if rows[i][2] is not None:
                rec["min_clearance_m"] = (clear[i] if rec["min_clearance_m"] is None
                                          else min(rec["min_clearance_m"], clear[i]))
                if clear[i] < 0.0:
                    rec["collision"] = True
            if rec["collision"]:
                rec.update(success=False, final_distance_m=dist[i], time_s=step / CONTROL_HZ)
                open_rows.discard(i)
            elif dist[i] <= radius:
                rec.update(success=True, time_s=step / CONTROL_HZ, final_distance_m=dist[i])
                open_rows.discard(i)
            elif step >= deadline_steps[i]:
                rec.update(success=False, time_s=None, final_distance_m=dist[i])
                open_rows.discard(i)
    for i in open_rows:
        result[i].update(success=False, final_distance_m=float(
            torch.linalg.vector_norm(state[i, :2] - goals64[i])))
    for rec, r in zip(result, rows):
        rec["deadline_s"] = r[3]
        rec["mean_ess"] = rec["ess_sum"] / rec["ess_steps"] if rec["ess_steps"] else None
        del rec["ess_sum"], rec["ess_steps"]
    return {"vessel": vessel, "lambda": float(lam), "H": params.H, "K": params.K,
            "params": params.as_dict(), "rows": result, "wall_s": time.time() - t0,
            "torch_threads": int(threads)}


def objective(point):
    j = 0.0
    for rec in point["rows"]:
        j += rec["time_s"] if rec["success"] else 2.0 * rec["deadline_s"]
    return j


def run_point(vessel, lam, threads=2):
    import mppi_controller as mc
    scale = mc.vessel_length_scale(vessel)
    rows = [sc for sc in scenarios(scale) for _ in range(REPLICAS)]
    point = simulate(vessel, lam, rows, threads=threads)
    point["objective_s"] = objective(point)
    point["successes"] = sum(bool(r["success"]) for r in point["rows"])
    point["accepted"] = point["successes"] == len(point["rows"])
    return point


def choose(vessel, points):
    """Procedure section 4 (R0/R1/R2/R3) on {lambda: point}."""
    def key(lam):
        return (points[lam]["objective_s"], abs(math.log(lam / SHIPPED["lambda"])), lam)

    accepted = [lam for lam in LAMBDA_GRID if points[lam]["accepted"]]
    if not accepted:
        best = sorted(LAMBDA_GRID, key=lambda lam: (-points[lam]["successes"],) + key(lam))[0]
        return best, "R3", False, None
    optimum = min(accepted, key=key)
    # ties |dJ| < 1e-9 already broken by the key order (log distance to 0.3, then smaller)
    if vessel == "blueboat_v2":
        ref = points[SHIPPED["lambda"]]
        if ref["accepted"] and ref["objective_s"] <= points[optimum]["objective_s"] * (1.0 + R1_TOLERANCE):
            return SHIPPED["lambda"], "R1", True, optimum
        return optimum, "R0", True, optimum
    return optimum, "R2", True, optimum


def freeze(procedure, points_dir, out, evidence, supersedes=None, reason=None):
    import mppi_controller as mc
    procedure_sha = lf_sha256(procedure)
    previous = None
    if supersedes:
        raw = Path(supersedes).read_bytes()
        previous = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
    vessels, all_points = {}, {}
    for vessel in VESSELS:
        points = {}
        for lam in LAMBDA_GRID:
            path = Path(points_dir) / point_name(vessel, lam)
            points[lam] = json.loads(path.read_text(encoding="ascii"))
        chosen, rule, accepted, optimum = choose(vessel, points)
        params = params_for(vessel, chosen)
        vessels[vessel] = {
            "H": params.H, "lambda": chosen, "rule": rule, "accepted": accepted,
            "horizon_rule": "declared: mppi_params_for_vessel (H = round(60 s), s = %.6g)"
                            % mc.vessel_length_scale(vessel),
            "length_scale": mc.vessel_length_scale(vessel),
            "r0_optimum_lambda": optimum,
            "shipped_reference": dict(SHIPPED),
            "grid": {("%g" % lam): {"objective_s": points[lam]["objective_s"],
                                    "successes": points[lam]["successes"],
                                    "accepted": points[lam]["accepted"],
                                    "mean_ess": _mean([r["mean_ess"] for r in points[lam]["rows"]])}
                     for lam in LAMBDA_GRID},
        }
        all_points[vessel] = {("%g" % lam): points[lam] for lam in LAMBDA_GRID}
    frozen = {"format": FORMAT, "status": "frozen", "procedure": "MPPI_TUNING_PROCEDURE_V2.md",
              "procedure_sha256": procedure_sha, "lambda_grid": list(LAMBDA_GRID),
              "model": "mppi_controller.VesselDynamicsV2 calm nominal (plant float64, controller float32)",
              "model_damping": {vessel: mc.model_damping_record(vessel) for vessel in VESSELS},
              "vessels": vessels}
    if previous is not None:
        sha, old = previous
        record = {"sha256": sha, "reason": reason, "procedure_sha256": old.get("procedure_sha256"),
                  "model_damping": old.get("model_damping"),
                  "vessels": {v: old["vessels"][v] for v in VESSELS
                              if (old.get("vessels") or {}).get(v) != vessels[v]}}
        frozen["history"] = list(old.get("history") or []) + [record]
    Path(out).write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n", encoding="ascii", newline="\n")
    if evidence:
        Path(evidence).write_text(json.dumps({"format": FORMAT + "-evidence", "procedure_sha256": procedure_sha,
                                              "frozen_sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(),
                                              "points": all_points}, indent=1, sort_keys=True) + "\n",
                                  encoding="ascii", newline="\n")
    return frozen


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def point_name(vessel, lam):
    return "%s_lambda_%g.json" % (vessel, lam)


def abeam_case(lam, horizon=None, threads=2):
    import mppi_controller as mc
    vessel = "wamv_vrx_v2_opt1"
    scale = mc.vessel_length_scale(vessel)
    goal = (0.0, 15.0 * scale)
    rows = [("ABEAM", goal, None, 2.0 * 15.0 * scale / CRUISE + 10.0)]
    return simulate(vessel, lam, rows, horizon=horizon, threads=threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("point")
    p.add_argument("--vessel", required=True, choices=VESSELS)
    p.add_argument("--lambda", dest="lam", type=float, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", required=True)
    a = sub.add_parser("all")
    a.add_argument("--procedure", required=True)
    a.add_argument("--points", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--evidence", default=None)
    a.add_argument("--workers", type=int, default=14)
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--freeze-only", action="store_true")
    a.add_argument("--supersedes", default=None)
    a.add_argument("--supersede-reason", default=None)
    b = sub.add_parser("abeam")
    b.add_argument("--lambda", dest="lam", type=float, required=True)
    b.add_argument("--H", type=int, default=None)
    b.add_argument("--threads", type=int, default=2)
    b.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "point":
        if args.lam not in LAMBDA_GRID:
            parser.error("lambda is not on the declared grid")
        point = run_point(args.vessel, args.lam, args.threads)
        Path(args.out).write_text(json.dumps(point, indent=1, sort_keys=True) + "\n", encoding="ascii",
                                  newline="\n")
        print(args.vessel, args.lam, "J", round(point["objective_s"], 3), "ok", point["successes"],
              "wall", round(point["wall_s"], 1))
        return 0
    if args.cmd == "abeam":
        case = abeam_case(args.lam, args.H, args.threads)
        Path(args.out).write_text(json.dumps(case, indent=1, sort_keys=True) + "\n", encoding="ascii",
                                  newline="\n")
        print(json.dumps(case["rows"]))
        return 0
    points = Path(args.points)
    points.mkdir(parents=True, exist_ok=True)
    if not args.freeze_only:
        jobs = [(v, lam) for v in VESSELS for lam in LAMBDA_GRID]
        running = []
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        while jobs or running:
            while jobs and len(running) < args.workers:
                v, lam = jobs.pop(0)
                cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "point", "--vessel", v, "--lambda",
                       repr(lam), "--threads", str(args.threads), "--out", str(points / point_name(v, lam))]
                running.append((v, lam, subprocess.Popen(cmd, env=env)))
            time.sleep(2.0)
            for item in list(running):
                code = item[2].poll()
                if code is not None:
                    running.remove(item)
                    if code != 0:
                        raise SystemExit("point %s lambda %g failed (%d)" % (item[0], item[1], code))
    if args.supersedes and not args.supersede_reason:
        parser.error("--supersedes needs --supersede-reason")
    frozen = freeze(args.procedure, points, args.out, args.evidence, args.supersedes, args.supersede_reason)
    for vessel, block in frozen["vessels"].items():
        print(vessel, "H", block["H"], "lambda", block["lambda"], block["rule"], "accepted", block["accepted"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
