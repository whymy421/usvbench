"""Admission survey: N seeds per task x vessel x tier (CPU only, read-only).

Prints attempts per admitted layout, rejection-reason totals, witness pivot
usage, obstacle counts versus capacity, and nominal-route/deadline ranges,
then per-vessel deadline sums (the compute-budget driver).
Usage: python -B scripts/tests_contracts_v2/admission_survey.py [N]
           [--vessels name,name] [--cruise U] [--json PATH]
  default vessels: the two contract vessels (journal_cfg_v2.VESSELS);
  --vessels: any registered v2 profiles; --cruise: ANALYSIS override of the
  contract cruise speed (m/s) for every surveyed vessel; --json: also write
  one record per layout (written only where the caller points it).
Seeds are D4's: numpy default_rng(10007 * seed + 31 * tier + 1).
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("d4_survey_layouts", ROOT / "tasks/_shared/journal_layouts_v2.py")
L = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = L
spec.loader.exec_module(L)


def pure_geometry_hash(lay):
    """Geometry-only identity (no vessel record, no deadline): equal across
    cruise speeds when the generated geometry is identical."""
    return L.geo.geometry_digest(start=lay.start, goal=lay.goal, start_heading=float(lay.start_heading_rad),
                                 centers=lay.centers, radii=lay.radii, bounds=list(lay.bounds_m))


def main(n, vessels=None, cruise=None, json_path=None):
    t0 = time.time()
    vessels = list(L.cfg.VESSELS) if not vessels else list(vessels)
    records = []
    print("contract version %d; geometry version digest %s" % (L.cfg.CONTRACT_VERSION, L.cfg.GEOMETRY_VERSION_DIGEST))
    for vessel in vessels:
        rec = L.cfg.vessel_record(vessel, cruise)
        print("vessel %s: L=%.4f m B=%.4f m s=%.6f R=%.4f m cruise=%.3f m/s top=%.4f m/s stop=%.4f m mass=%.4f kg"
              " (%s)" % (vessel, rec.length_m, rec.beam_m, rec.length_scale, rec.turning_radius_m,
                         rec.cruise_speed_m_s, rec.top_speed_m_s, rec.stop_distance_m, rec.mass_kg,
                         rec.cruise_speed_status))
    print("task             vessel                 G  ok/N  att(mean,max)  pivots(mean)  n_obst(max/cap)  "
          "L_nom[min,max] m   T[min,max] s   rejections")
    for task in L.cfg.TASKS:
        for vessel in vessels:
            for tier in range(4):
                ok, att, piv, nob, lnom, dl, rej = 0, [], [], [], [], [], Counter()
                for seed in range(n):
                    rng = np.random.default_rng(10007 * seed + 31 * tier + 1)
                    try:
                        lay = L.build_layout(task, tier, vessel, rng, cruise_speed_m_s=cruise)
                    except L.JournalAdmissionError as e:
                        rej.update(e.rejections)
                        records.append(dict(task=task, vessel=vessel, tier=tier, seed=seed, admitted=False,
                                            attempts=e.attempts, rejections=dict(e.rejections)))
                        continue
                    ok += 1
                    att.append(lay.attempts)
                    piv.append(lay.witness.n_pivots)
                    nob.append(lay.obstacle_count)
                    lnom.append(lay.nominal_route_length_m)
                    dl.append(lay.deadline_s)
                    rej.update(lay.rejections)
                    records.append(dict(task=task, vessel=vessel, tier=tier, seed=seed, admitted=True,
                                        attempts=lay.attempts, rejections=dict(lay.rejections),
                                        n_pivots=lay.witness.n_pivots, n_obstacles=lay.obstacle_count,
                                        capacity=L.cfg.STATIC_CAPACITY[task],
                                        L_nom_m=lay.nominal_route_length_m, deadline_s=lay.deadline_s,
                                        deadline_steps=lay.deadline_steps, dwell_s=lay.params["dwell_s"],
                                        cruise_m_s=L.cfg.vessel_record(vessel, cruise).cruise_speed_m_s,
                                        geometry_hash=lay.geometry_hash,
                                        pure_geometry_hash=pure_geometry_hash(lay)))
                print("%-16s %-22s %d  %2d/%-2d  %5.2f,%3d      %5.2f        %4d/%-4d       [%7.1f,%7.1f]  "
                      "[%6.1f,%6.1f]  %s" % (
                          task, vessel, tier, ok, n, np.mean(att) if att else float("nan"),
                          max(att) if att else -1, np.mean(piv) if piv else float("nan"),
                          max(nob) if nob else -1, L.cfg.STATIC_CAPACITY[task],
                          min(lnom) if lnom else float("nan"), max(lnom) if lnom else float("nan"),
                          min(dl) if dl else float("nan"), max(dl) if dl else float("nan"), dict(rej)),
                      flush=True)
    print("deadline sums over the admitted layouts (seconds of simulated time, one episode each):")
    for vessel in vessels:
        mine = [r for r in records if r["vessel"] == vessel and r["admitted"]]
        per_tier = []
        for tier in range(4):
            means = [np.mean([r["deadline_s"] for r in mine if r["task"] == t and r["tier"] == tier])
                     for t in L.cfg.TASKS if any(r["task"] == t and r["tier"] == tier for r in mine)]
            per_tier.append(float(np.sum(means)))
        print("  %-22s admitted %3d/%3d  sum_T %.1f s  sum over tasks of mean T per tier G0..G3: %s" % (
            vessel, len(mine), len([r for r in records if r["vessel"] == vessel]),
            sum(r["deadline_s"] for r in mine), ", ".join("%.1f" % v for v in per_tier)))
    print("elapsed %.1f s" % (time.time() - t0))
    if json_path:
        header = dict(n=n, vessels=vessels, cruise_override=cruise, contract_version=L.cfg.CONTRACT_VERSION,
                      geometry_version_digest=L.cfg.GEOMETRY_VERSION_DIGEST,
                      registry_source_sha256=dict(L.cfg.REGISTRY_SOURCE_SHA256),
                      vessel_records={v: dict(L.cfg.vessel_record(v, cruise).derived_numbers(),
                                              digest=L.cfg.vessel_record(v, cruise).digest,
                                              cruise_speed_status=L.cfg.vessel_record(v, cruise).cruise_speed_status)
                                      for v in vessels})
        Path(json_path).write_text(json.dumps(dict(header=header, layouts=records), indent=1, sort_keys=True),
                                   encoding="ascii")
        print("wrote " + str(json_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=5)
    ap.add_argument("--vessels", default="")
    ap.add_argument("--cruise", type=float, default=None)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    main(a.n, [v for v in a.vessels.split(",") if v], a.cruise, a.json or None)
