"""Pre-declared classical gain procedure for the journal vessels (F7; CPU only).

Implements F7 work/TUNING_PROCEDURE.md verbatim (declared before the first
run; its LF-normalised SHA-256 is stamped into the output): the calm nominal
registry model (mppi_controller.VesselDynamicsV2, float64) is the ONLY world;
the journal control laws are the ones the evaluators run
(journal_controllers.HeadingPID / pass_through_thrust / DockingLaw); every
grid point runs every scenario in one batch. Nothing here touches Isaac, an
evaluation seed, a journal layout or an evaluation pack.

  python -B scripts/tune_classical_v2.py --procedure <work>/TUNING_PROCEDURE.md \
      --out scripts/classical_gains_v2.json [--evidence <work>/evidence/tuning_v2.json]
      [--supersedes <old classical_gains_v2.json> --supersede-reason "<why>"]

G4-A5: the output records the tuning model's planar damping per vessel (``model_damping``,
mppi_controller.model_damping_record; journal_controllers.load_gains refuses a file whose
record differs from the current calm nominal model) and, on a re-freeze, the superseded
record under ``history``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402

import journal_controllers as jc  # noqa: E402
from mppi_controller import VesselDynamicsV2, model_damping_record  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
DT = 1.0 / 60.0
LOS_KP = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
LOS_KD = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
DOCK_KP = (0.5, 1.0, 2.0)
DOCK_KRATE = (0.5, 1.0, 2.0, 4.0)
LEGACY = {"los_heading_pid": {"kp": 2.0, "ki": 0.0, "kd": 0.5}, "dock_yaw_pd": {"kp": 1.0, "k_rate": 1.0}}
GOAL_BEARINGS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0)
GOAL_DISTANCE = 20.0
GATES = ((15.0, 0.0), (30.0, 10.0), (45.0, -10.0), (60.0, 0.0))
RADIUS = 2.0
CRUISE = 1.0
DOCK_ERRORS_DEG = (30.0, 90.0, 150.0)
CHATTER_LIMIT = 0.05
DOCK_SETTLE_LIMIT_S = 20.0
R1_TOLERANCE = 0.10


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def los_scenarios(scale):
    """[(name, targets [G,2], deadline_s)] of TUNING_PROCEDURE.md section 3 (s-scaled)."""
    out = []
    for bearing in GOAL_BEARINGS_DEG:
        a = math.radians(bearing)
        goal = (GOAL_DISTANCE * scale * math.cos(a), GOAL_DISTANCE * scale * math.sin(a))
        out.append(("G%d" % int(bearing), [goal], 2.0 * GOAL_DISTANCE * scale / CRUISE + 10.0))
    gates = [(x * scale, y * scale) for x, y in GATES]
    length, cursor = 0.0, (0.0, 0.0)
    for gate in gates:
        length += math.dist(cursor, gate)
        cursor = gate
    out.append(("P0", gates, 2.0 * length / CRUISE + 10.0))
    return out


def run_los(vessel, grid):
    """Every (kp, kd) x LOS scenario at once; returns {(kp, kd): {scenario: result}}."""
    scale = jc.length_scale(vessel)
    scenarios = los_scenarios(scale)
    dyn = VesselDynamicsV2(vessel, dtype=torch.float64)
    rows = [(g, sc) for g in grid for sc in scenarios]
    n = len(rows)
    width = max(len(sc[1]) for sc in scenarios)
    targets = torch.zeros((n, width, 2), dtype=torch.float64)
    count = torch.zeros(n, dtype=torch.long)
    deadline = torch.zeros(n, dtype=torch.float64)
    for i, (_, (_, pts, limit)) in enumerate(rows):
        for j in range(width):
            targets[i, j] = torch.tensor(pts[min(j, len(pts) - 1)], dtype=torch.float64)
        count[i] = len(pts)
        deadline[i] = limit
    kp = torch.tensor([g[0] for g, _ in rows], dtype=torch.float64)
    kd = torch.tensor([g[1] for g, _ in rows], dtype=torch.float64)
    pid = jc.HeadingPID(n, "cpu", torch.float64, DT, kp, torch.zeros(n, dtype=torch.float64), kd)
    state = torch.zeros((n, 6), dtype=torch.float64)
    index = torch.zeros(n, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    finish = torch.full((n,), float("nan"), dtype=torch.float64)
    flips = torch.zeros(n, dtype=torch.long)
    steps = torch.zeros(n, dtype=torch.long)
    previous_yaw = None
    radius = RADIUS * scale
    for k in range(int(math.ceil(float(deadline.max()) / DT))):
        t = (k + 1) * DT
        active = ~done
        target = targets[torch.arange(n), index.clamp(max=width - 1)]
        error = jc.bearing_error(state[:, :2], state[:, 2], target)
        yaw = pid(error)
        action = torch.stack((jc.pass_through_thrust(error), yaw), dim=-1)
        if previous_yaw is not None:
            flips += (((yaw - previous_yaw).abs() > 1.0) & active).long()
        previous_yaw = yaw
        steps += active.long()
        state = dyn.step(state, action)
        reached = (torch.linalg.vector_norm(state[:, :2] - target, dim=-1) <= radius) & active
        index = torch.where(reached, index + 1, index)
        finished = reached & (index >= count)
        finish = torch.where(finished & torch.isnan(finish), torch.full_like(finish, t), finish)
        timed_out = active & (t >= deadline)
        done = done | finished | timed_out
        if bool(done.all()):
            break
    results = {}
    for i, (g, (name, _, limit)) in enumerate(rows):
        ok = bool(not math.isnan(float(finish[i])) and float(finish[i]) <= limit)
        chatter = float(flips[i]) / max(int(steps[i]), 1)
        results.setdefault(g, {})[name] = {"success": ok, "time_s": None if not ok else float(finish[i]),
                                           "deadline_s": limit, "chatter": chatter}
    return results


def run_dock(vessel, grid):
    scale = jc.length_scale(vessel)
    dyn = VesselDynamicsV2(vessel, dtype=torch.float64)
    rows = [(g, e) for g in grid for e in DOCK_ERRORS_DEG]
    n = len(rows)
    gains = {"dock_yaw_pd": {"kp": torch.tensor([g[0] for g, _ in rows], dtype=torch.float64),
                             "k_rate": torch.tensor([g[1] for g, _ in rows], dtype=torch.float64)}}
    law = jc.DockingLaw(n, "cpu", torch.float64, DT, gains, tolerance_m=2.5 * scale, scale=scale)
    state = torch.zeros((n, 6), dtype=torch.float64)
    berth_heading = torch.stack([torch.tensor([math.cos(math.radians(e)), math.sin(math.radians(e))],
                                              dtype=torch.float64) for _, e in rows])
    berth_point = torch.zeros((n, 2), dtype=torch.float64)
    held = torch.zeros(n, dtype=torch.float64)
    settle = torch.full((n,), float("nan"), dtype=torch.float64)
    flips = torch.zeros(n, dtype=torch.long)
    steps = torch.zeros(n, dtype=torch.long)
    previous_yaw = None
    for k in range(int(round(DOCK_SETTLE_LIMIT_S / DT))):
        t = (k + 1) * DT
        active = torch.isnan(settle)
        inputs = jc.JournalInputs(state[:, :2], state[:, :2], state[:, 2], state[:, 3:6],
                                  torch.zeros((n, 36), dtype=torch.float64))
        action = law.act(inputs, berth_point, berth_heading)
        yaw = action[:, 1]
        if previous_yaw is not None:
            flips += (((yaw - previous_yaw).abs() > 1.0) & active).long()
        previous_yaw = yaw
        steps += active.long()
        state = dyn.step(state, action)
        error = jc.wrap(torch.atan2(berth_heading[:, 1], berth_heading[:, 0]) - state[:, 2])
        calm = (error.abs() <= math.radians(10.0)) & (state[:, 5].abs() <= 0.05)
        held = torch.where(calm, held + DT, torch.zeros_like(held))
        newly = active & (held >= 1.0 - 1e-9)
        settle = torch.where(newly, torch.full_like(settle, t), settle)
        if not bool(torch.isnan(settle).any()):
            break
    results = {}
    for i, (g, e) in enumerate(rows):
        ok = not math.isnan(float(settle[i]))
        results.setdefault(g, {})["A%d" % int(e)] = {"success": ok, "time_s": float(settle[i]) if ok else None,
                                                     "limit_s": DOCK_SETTLE_LIMIT_S,
                                                     "chatter": float(flips[i]) / max(int(steps[i]), 1)}
    return results


def summarize(results):
    table = {}
    for g, scen in results.items():
        accepted = all(r["success"] and r["chatter"] < CHATTER_LIMIT for r in scen.values())
        objective = sum(r["time_s"] for r in scen.values()) if accepted else None
        table[g] = {"accepted": accepted, "objective_s": objective, "scenarios": scen}
    return table


def choose(table, legacy_key, vessel):
    accepted = [(v["objective_s"], g) for g, v in table.items() if v["accepted"]]
    if not accepted:
        return None, None, "FAILED"
    best_obj = min(o for o, _ in accepted)
    best = sorted(g for o, g in accepted if abs(o - best_obj) < 1e-9)[0]
    if vessel == "blueboat_v2":
        legacy = table.get(legacy_key)
        if legacy is not None and legacy["accepted"] and legacy["objective_s"] <= (1.0 + R1_TOLERANCE) * best_obj:
            return legacy_key, best, "R1"
        return best, best, "R0"
    return best, best, "R2"


def tune(procedure):
    output = {"format": jc.GAINS_FORMAT, "procedure": "F7 work/TUNING_PROCEDURE.md",
              "procedure_sha256": lf_sha256(procedure), "model": "mppi_controller.VesselDynamicsV2 (calm nominal)",
              "tuner": "scripts/tune_classical_v2.py", "control_dt_s": DT,
              "grids": {"los_heading_pid": {"kp": LOS_KP, "ki": [0.0], "kd": LOS_KD},
                        "dock_yaw_pd": {"kp": DOCK_KP, "k_rate": DOCK_KRATE}},
              "legacy": LEGACY, "vessels": {},
              "model_damping": {vessel: model_damping_record(vessel) for vessel in VESSELS}}
    evidence = {}
    for vessel in VESSELS:
        los = summarize(run_los(vessel, list(itertools.product(LOS_KP, LOS_KD))))
        dock = summarize(run_dock(vessel, list(itertools.product(DOCK_KP, DOCK_KRATE))))
        block = {"length_scale": jc.length_scale(vessel)}
        for name, table, legacy_key, keys in (
                ("los_heading_pid", los, (2.0, 0.5), ("kp", "kd")),
                ("dock_yaw_pd", dock, (1.0, 1.0), ("kp", "k_rate"))):
            chosen, best, rule = choose(table, legacy_key, vessel)
            entry = {"rule": rule, "status": "frozen" if chosen is not None else "FAILED",
                     "legacy": LEGACY[name],
                     "legacy_result": {k: v for k, v in table[legacy_key].items() if k != "scenarios"},
                     "r0_optimum": None if best is None else dict(zip(keys, best)),
                     "r0_objective_s": None if best is None else table[best]["objective_s"],
                     "accepted_points": sum(1 for v in table.values() if v["accepted"])}
            if chosen is not None:
                gains = dict(zip(keys, chosen))
                if name == "los_heading_pid":
                    gains = {"kp": gains["kp"], "ki": 0.0, "kd": gains["kd"]}
                entry["gains"] = gains
                entry["objective_s"] = table[chosen]["objective_s"]
            block[name] = entry
        output["vessels"][vessel] = block
        evidence[vessel] = {"los": {"%g,%g" % g: v for g, v in los.items()},
                            "dock": {"%g,%g" % g: v for g, v in dock.items()}}
    return output, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--supersedes", type=Path, default=None)
    parser.add_argument("--supersede-reason", default=None)
    args = parser.parse_args(argv)
    if args.supersedes is not None and not args.supersede_reason:
        parser.error("--supersedes needs --supersede-reason")
    previous = None
    if args.supersedes is not None:
        raw = args.supersedes.read_bytes()
        previous = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
    torch.set_num_threads(1)
    output, evidence = tune(args.procedure)
    output["evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode("ascii")).hexdigest()
    if previous is not None:
        sha, old = previous
        record = {"sha256": sha, "reason": args.supersede_reason, "procedure_sha256": old.get("procedure_sha256"),
                  "evidence_sha256": old.get("evidence_sha256"), "model_damping": old.get("model_damping"),
                  "vessels": {v: old["vessels"][v] for v in VESSELS
                              if (old.get("vessels") or {}).get(v) != output["vessels"][v]}}
        output["history"] = list(old.get("history") or []) + [record]
    args.out.write_text(json.dumps(output, indent=1, sort_keys=True) + "\n", encoding="ascii", newline="\n")
    if args.evidence is not None:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n", encoding="ascii",
                                 newline="\n")
    for vessel, block in output["vessels"].items():
        for name in ("los_heading_pid", "dock_yaw_pd"):
            e = block[name]
            print(vessel, name, e["status"], e["rule"], e.get("gains"), "objective", e.get("objective_s"),
                  "legacy", e["legacy_result"], "r0", e["r0_optimum"], e["r0_objective_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
