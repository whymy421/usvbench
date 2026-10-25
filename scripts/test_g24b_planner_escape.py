"""G24b CPU tests: the memory planner's hold escape (REV-G25b CRITICAL-1) and per-episode hold counts (m2). No Isaac.

* a ghost ring (remembered returns of obstacles that are gone, not carved because no ray of the stationary,
  rotated fan passes within the carve lateral) seals the start: the env holds, and after
  hold_escape_failed_replans (2) consecutive truncated replans while holding it replans on the CURRENT scan only and
  drives (escape plan); the memory itself is untouched; without the escape (rule constant very large) the same
  env holds to the end of the run (the absorbing hold REV-G25b measured);
* a real blockage (goal 0.4 m behind a face, in view): the escape replan fails too, the env keeps holding and the
  failures are counted (REV-G25b tools/hold_absorbing_demo.py scenario);
* per-episode counts: the finished episode's hold / unplannable / escape counts are returned for the record and the
  telemetry rows add up to the run totals; the snap-start radius scales with the vessel (REV-G25b m5).

Run: python -B -m pytest -p no:cacheprovider scripts/test_g24b_planner_escape.py -q
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import planner_memory as pm  # noqa: E402
from test_planner_memory import FakeEnv  # noqa: E402


def _controller(env, vessel="blueboat_v2", rules=None):
    import classical_planner_pid as cp
    import journal_controllers as jc

    s = jc.length_scale(vessel)
    return cp.JournalMemoryPlannerController(env, replan_every=60, waypoint_spacing_m=2.0 * s, accept_radius_m=1.0 * s,
                                             plan_cell_m=0.25 * s, ladder=cp.journal_memory_ladder(vessel),
                                             gains=jc.load_gains(vessel), max_range_m=env.max_range, vessel=vessel,
                                             memory_rules=rules)


def _set_yaw(env, yaw):
    env.robot.data.root_link_pose_w[:, 3] = math.cos(0.5 * yaw)
    env.robot.data.root_link_pose_w[:, 6] = math.sin(0.5 * yaw)


def _ghost_ring_run(steps, rules=None):
    env = FakeEnv(1)
    env.target_pos = torch.tensor([[40.0, 0.0]])
    angles = np.arange(0.0, 2.0 * math.pi, 2.0 * math.pi / 40)
    env.centers = np.stack((6.0 * np.cos(angles), 6.0 * np.sin(angles)), -1)       # a closed (overlapping) ring
    env.radii = np.full(len(angles), 0.55)
    env.refresh()
    ctl = _controller(env, rules=rules)
    first = ctl.act()                                       # remembers the ring; no rung plans out of it -> hold
    assert ctl.unplannable_legs == 1 and torch.equal(first[0], torch.zeros(2))
    size = ctl.memory.size(0)
    env.centers = np.zeros((0, 2))                          # the ring is gone ...
    env.radii = np.zeros(0)
    _set_yaw(env, math.pi / 36)                             # ... and the fan is rotated half a ray spacing: no carve
    env.refresh()
    moved_at = None
    for k in range(1, steps):
        action = ctl.act()
        if moved_at is None and bool((action != 0).any()):
            moved_at = k
    return ctl, size, moved_at


def test_escape_replans_on_the_current_scan_when_remembered_ghosts_seal_the_start():
    ctl, size, moved_at = _ghost_ring_run(150)                 # before the next replan (step 183)
    assert ctl.memory.counts["carved"] == 0 and ctl.memory.size(0) == size        # the memory is untouched
    assert ctl.escape_plans == 1 and ctl.escape_failures == 0
    # holding from step 0; replans at steps 61 (streak 1) and 122 (streak 2 -> escape): moves at the second
    assert moved_at == 2 * (ctl.replan_every + 1), moved_at
    assert ctl.hold_steps == moved_at and ctl.unplannable_legs == 3
    tel = ctl.telemetry()
    assert tel["escape_plans"] == 1 and tel["legs_planned"] == sum(tel["legs_by_rung"]) + tel["unplannable_legs"]
    # without the escape the same env holds to the end of the run (REV-G25b's absorbing hold)
    rules = dict(pm.RULE_CONSTANTS, hold_escape_failed_replans=10 ** 9)
    ctl0, _size, moved0 = _ghost_ring_run(400, rules=rules)
    assert moved0 is None and ctl0.hold_steps == 400 and ctl0.escape_plans == 0


def test_a_real_blockage_keeps_holding_and_counts_escape_failures():
    env = FakeEnv(1)
    env.target_pos = torch.tensor([[20.0, 0.0]])
    env.centers = np.array([[20.8, 0.0]])                   # the goal is 0.4 m behind a face (in view)
    env.radii = np.array([1.2])
    env.refresh()
    ctl = _controller(env)
    nonzero = 0
    for _ in range(1000):
        nonzero += int(bool((ctl.act() != 0).any()))
    assert nonzero == 0 and ctl.hold_steps == 1000
    assert ctl.escape_plans == 0 and ctl.escape_failures >= 10
    assert ctl.escape_failures + ctl.escape_plans <= ctl.unplannable_legs


def test_per_episode_counts_go_to_the_record_and_rows_add_up():
    ctl, _size, moved_at = _ghost_ring_run(150)
    assert ctl.finished_episode(0) == {"hold_steps": 0, "unplannable_legs": 0, "escape_plans": 0, "escape_failures": 0}
    ctl.reset(torch.tensor([True]))
    assert ctl.finished_episode(0) == {"hold_steps": moved_at, "unplannable_legs": 3, "escape_plans": 1,
                                       "escape_failures": 0}
    tel = ctl.telemetry()
    assert tel["episode_rows_fields"] == ["env", "episode", "hold_steps", "unplannable_legs", "escape_plans",
                                          "escape_failures"]
    assert tel["episode_rows"] == [[0, 0, moved_at, 3, 1, 0]] and tel["unfinished_rows"] == []
    assert not bool(ctl.held[0]) and int(ctl.fail_streak[0]) == 0
    import validate_cert_v3 as v
    assert set(tel) == set(v.PLANNER_MEMORY_COUNTS) | set(v.PLANNER_MEMORY_ROW_KEYS)


def test_snap_start_scales_with_the_vessel_and_the_base_planner_keeps_1_5_m():
    import classical_planner_pid as cp
    import journal_controllers as jc

    env = FakeEnv(1, max_range=30.0 * jc.length_scale("wamv_vrx_v2_opt1"))
    ctl = _controller(env, vessel="wamv_vrx_v2_opt1")
    assert ctl.snap_start_m == pm.SNAP_START_REF_M * jc.length_scale("wamv_vrx_v2_opt1")
    base = cp.JournalPlannerLOSPIDController(FakeEnv(1), replan_every=60, waypoint_spacing_m=2.0, accept_radius_m=1.0,
                                             plan_cell_m=0.25, ladder=cp.journal_ladder("blueboat_v2"),
                                             gains=jc.load_gains("blueboat_v2"), max_range_m=30.0)
    assert base.snap_start_m == 1.5
    assert pm.RULE_CONSTANTS["hold_escape_failed_replans"] == 2
