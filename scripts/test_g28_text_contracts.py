"""G28 item 4: the documented contract values match the code, and the corrected texts are in place (CPU).

* docking / harbour speed tolerance: 0.3 m/s for BOTH vessels at every tier (speed ratio 1: both cruise speeds
  are 1.0 m/s); the journal_cfg_v2 docstring states x1 and supersedes the D4 'x2 for WAM-V';
* station-keeping settle 20 s unchanged (dwell 80 = settle 20 + hold 60), with its rationale next to the value;
* decided items carry DECIDED labels (PENDING_DECISIONS, PENDING_OWNER_DECISIONS);
* citations: Zu, Garme and Rosen 2024 Table 6 / Table 4, PPO (Schulman et al. 2017), guidance law (Breivik and
  Fossen 2008; Fossen 2011); Stokes drift measured 0-32 %.

Run: python -B scripts/test_g28_text_contracts.py   (or pytest)
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "tests_journal_glue"))

import glue_fixtures as G  # noqa: E402

CFG = G.shared("journal_cfg_v2")
LAYER = G.shared("conditions.hydro_v2_layer")
REPO = os.path.dirname(_HERE)
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def read(relative):
    with open(os.path.join(REPO, relative), encoding="utf-8") as handle:
        return handle.read()


def test_docking_speed_tolerance_is_x1_for_both_vessels():
    for vessel in VESSELS:
        record = CFG.vessel_record(vessel)
        assert record.cruise_speed_m_s == record.reference_cruise_speed_m_s == 1.0
        for task in ("docking", "harbour_mission"):
            for tier in range(4):
                assert CFG.resolve_task_geometry(task, tier, vessel)["speed_tolerance_m_s"] == 0.3
    doc = CFG.__doc__
    assert "ratio" in doc and "is exactly 1" in doc and "0.3 m/s for BOTH" in doc and "superseded" in doc


def test_station_keeping_settle_value_and_rationale():
    for vessel in VESSELS:
        for tier in range(4):
            p = CFG.resolve_task_geometry("station_keeping", tier, vessel)
            assert (p["settle_s"], p["hold_s"], p["dwell_s"]) == (20.0, 60.0, 80.0)
    source = read("tasks/_shared/journal_cfg_v2.py")
    assert "settle_s = 20 s (G28 rationale" in source and "a DECLARED allowance" in source


def test_decided_labels():
    assert CFG.PENDING_DECISIONS["cruise_speed"].startswith("DECIDED")
    assert CFG.PENDING_DECISIONS["wamv_profile"].startswith("DECIDED")
    assert CFG.PENDING_DECISIONS["design_turning_radius"].startswith("PENDING")
    decisions = LAYER.PENDING_OWNER_DECISIONS
    for key in ("wave_ladder", "perception_outage_start", "rollpitch_damping_blueboat", "wamv_profile"):
        assert decisions[key]["status"].startswith("DECIDED"), key
    for key in ("wave_steepness_admission", "wave_spreading", "wave_band", "wind_dose_axis", "wind_gusts",
                "direction_strata", "thrust_line_pitch_coupling"):
        assert decisions[key]["status"].startswith("PENDING"), key


def test_citations_and_corrections():
    formula = read("tasks/_shared/conditions/green_water_formula.py")
    assert "Table 6, the" in formula and "Table 4, section 4.1" in formula
    assert "arXiv:1707.06347" in read("scripts/train_journal_ppo.py")
    baseline = read("scripts/CLASSICAL_BASELINE.md")
    assert "Breivik and Fossen 2008" in baseline and "Fossen 2011" in baseline and "pure-pursuit" in baseline
    hydro = read("tasks/_shared/hydro_v2.py")
    assert "0-32 %" in hydro and 'does NOT appear "automatically"' in hydro


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print("PASS " + name, flush=True)
