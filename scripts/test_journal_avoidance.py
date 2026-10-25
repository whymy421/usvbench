"""CPU tests of the G17 obstacle-avoidance layer (scripts/journal_avoidance.py). No Isaac.

* frozen constants: classical_avoidance_v2.json equals the derivation, drift is refused, the gate distances
  cover every D1 reveal of the obstacle suites on these tasks;
* gate and hysteresis on synthetic ray inputs; while inactive the action IS the base controller's tensor,
  the base controller's state equals a twin driven alone and no random number is drawn; inactive rows stay
  bitwise equal when another env is active;
* behaviours: path-following rejoin beyond the obstacle, station-keeping evade / hold / return, docking
  berth masking (the env's own walls and ray kernel, many keys and poses: never an unexplained return),
  docking plans and evasion targets clear of the walls, re-plan at the journal planner's cadence without
  motion prediction;
* no forbidden env member is read; the validator's avoidance-block check.
* G24b: the journal build's base laws (station keeping continuous_v2, docking RunupDockingLaw) are the twin's;
  memory rule "passed": no expiry (the remembered disc holds for any number of steps until the current rays see
  through its surface point on both sides), hand-back only then; G17's rule "steps" kept as the comparison arm.

Run: python -B scripts/test_journal_avoidance.py   (or pytest)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPTS)
for _path in (_SCRIPTS, _REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import journal_avoidance as ja  # noqa: E402
import journal_controllers as jc  # noqa: E402
import validate_cert_v3 as validator  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
RAYS = 36
MODE_TASK = {"path_follow": "path_following", "station_keep": "station_keeping", "docking": "docking"}


class ForbiddenAccess(AssertionError):
    pass


def _quat(yaw):
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    return torch.stack((torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)), -1)


class SyntheticEnv:
    """The members a journal classical controller reads, nothing else; every forbidden member raises."""

    def __init__(self, mode, vessel, n=2, origins=None):
        self.mode, self.vessel, self.num_envs, self.device = mode, vessel, int(n), torch.device("cpu")
        task = MODE_TASK[mode]
        self.plan = ja.journal_plan(task, vessel)
        self.params = self.plan.params
        self.max_range = float(self.params["ray_sensor_max_range_m"])
        self.cfg = SimpleNamespace(vehicle=vessel, journal_plan=self.plan, berth_walls=True, journal_task=task,
                                   ray_max_range_m=self.max_range)
        origins = torch.zeros((self.num_envs, 3)) if origins is None else torch.as_tensor(origins, dtype=torch.float32)
        self.scene = SimpleNamespace(env_origins=origins)
        self.robot = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=torch.zeros((self.num_envs, 7))))
        self.robot.data.root_link_pose_w[:, 3] = 1.0
        self.journal_estimate = None
        s = float(self.params["length_scale"])
        seg = float(self.params.get("segment_length_m", 16.0 * s))
        self.waypoints = torch.tensor([[seg, 0.0], [2 * seg, 0.0], [3 * seg, 0.0], [4 * seg, 0.0]]).repeat(self.num_envs, 1, 1)
        self.gates_passed = torch.zeros(self.num_envs, dtype=torch.long)
        self.hold_point = origins[:, :2] + torch.tensor([10.0 * s, 0.0])
        self.dock_point = origins[:, :2].clone()
        self.dock_heading = torch.tensor([1.0, 0.0]).repeat(self.num_envs, 1)

    def __getattr__(self, name):
        if name in jc.FORBIDDEN_ENV_MEMBERS:
            raise ForbiddenAccess("avoidance layer read forbidden env member " + name)
        raise AttributeError(name)

    def _current_waypoint_world(self):
        index = self.gates_passed.clamp(max=self.waypoints.shape[1] - 1)
        return self.waypoints[torch.arange(self.num_envs), index] + self.scene.env_origins[:, :2]

    def set_state(self, position_local, yaw, rays=None, velocity=None):
        position_local = torch.as_tensor(position_local, dtype=torch.float32).reshape(self.num_envs, 2)
        yaw = torch.as_tensor(yaw, dtype=torch.float32).reshape(self.num_envs)
        pose = torch.zeros((self.num_envs, 7))
        pose[:, :2] = position_local + self.scene.env_origins[:, :2]
        pose[:, 3:7] = _quat(yaw)
        self.robot.data.root_link_pose_w = pose
        if rays is None:
            rays = torch.full((self.num_envs, RAYS), self.max_range)
        velocity = torch.zeros((self.num_envs, 3)) if velocity is None else torch.as_tensor(velocity, dtype=torch.float32)
        self.journal_estimate = SimpleNamespace(position=position_local, velocity=velocity,
                                                rays=torch.as_tensor(rays, dtype=torch.float32))


def point_return_rays(com, yaw, centers, radii, max_range):
    """E4 point-return proxy (visibility.py:31-34; journal_path_following.point_return_rays): every object
    returns its nearest surface point on the fan ray nearest its bearing (a tie goes to the higher index)."""
    out = torch.full((com.shape[0], RAYS), float(max_range), dtype=torch.float64)
    step = 2.0 * math.pi / RAYS
    for i in range(com.shape[0]):
        for (cx, cy), r in zip(centers[i], radii[i]):
            dx, dy = cx - float(com[i, 0]), cy - float(com[i, 1])
            edge = max(math.hypot(dx, dy) - r, 0.0)
            bearing = math.atan2(dy, dx) - float(yaw[i])
            k = int(math.floor((bearing % (2.0 * math.pi)) / step + 0.5)) % RAYS
            out[i, k] = min(float(out[i, k]), min(edge, float(max_range)))
    return out


def build(mode, vessel, n=2, rules=None, **kw):
    env = SyntheticEnv(mode, vessel, n, **kw)
    env.set_state(torch.zeros((n, 2)), torch.zeros(n))
    gains = jc.load_gains(vessel)
    s = jc.length_scale(vessel)
    zone = float(env.params.get("hold_radius_m", env.params["goal_radius_m"]))
    args = dict(dt=1.0 / 60.0, zone_radius_m=zone, slow_radius_m=(6.0 if mode == "docking" else 5.0) * s,
                tolerance_m=float(env.params.get("position_tolerance_m", 2.5 * s)), scale=s)
    avoidance, block = ja.prepare(env.cfg, mode, gains, **args)
    if rules:                                        # G24b comparison arms (e.g. the G17 memory rule "steps")
        avoidance = dict(avoidance, rule_constants=dict(avoidance["rule_constants"], **rules))
    controller = ja.build_controller(env, env.cfg, mode, gains, avoidance, **args)
    # G24b: the twin is the journal base law the evaluator builds (hold mode and docking law of the journal rule)
    twin = jc.JournalBaselineController(env, mode, gains, dt=args["dt"], zone_radius_m=args["zone_radius_m"],
                                        slow_radius_m=args["slow_radius_m"], tolerance_m=args["tolerance_m"], scale=s,
                                        hold_mode=ja.journal_hold_mode(mode), runup=ja.runup_spec(mode, avoidance))
    return env, controller, twin, block


def see_through_both_sides(env, controller, position, point):
    """Two control steps at ``position`` with no return, ray 0 passing the remembered surface ``point`` at 0.4 x the
    carve lateral on one side, then on the other (the planner's S1 "passed" condition)."""
    lateral = 0.4 * controller.carver.carve_lateral
    d = float(np.hypot(point[0] - position[0], point[1] - position[1]))
    bearing = math.atan2(point[1] - position[1], point[0] - position[0])
    for sign in (1.0, -1.0):
        env.set_state(torch.tensor([position], dtype=torch.float32), [bearing + sign * math.asin(lateral / d)])
        controller.act(None)


def base_state(controller):
    pid = controller.los.pid
    return [pid.integral.clone(), pid.previous.clone(), pid.initialized.clone(), controller.dock.phase.clone()]


def obstacle_rays_at_gap(env, controller, gap_m, bearing=0.0, radius=None):
    """Rays of one cylinder straight along ``bearing`` whose surface point lies gap_m beyond the footprint."""
    fp = controller.footprint
    edge = (fp[1] if abs(bearing) < 1e-9 else 0.0) + gap_m          # bow face along +x (bearing 0)
    rays = torch.full((env.num_envs, RAYS), env.max_range)
    rays[:, 0] = edge
    return rays


# ------------------------------------------------------------- constants --
def test_frozen_constants_equal_the_derivation_and_drift_is_refused():
    data = json.loads(ja.AVOID_PATH.read_text(encoding="ascii"))
    assert data == json.loads(ja.document_text(ja.derive_document()))
    assert ja.AVOID_PATH.read_bytes().count(b"\r\n") == 0 and ja.AVOID_PATH.read_bytes().isascii()
    for vessel in VESSELS:
        for task in ("path_following", "station_keeping", "docking"):
            loaded = ja.load_avoidance(vessel, task)
            assert loaded["vessel_constants"]["vessel"] == vessel and loaded["task_constants"]["task"] == task
            assert len(loaded["file_sha256"]) == 64
    with tempfile.TemporaryDirectory() as folder:
        for mutate in ("vessel", "rule"):
            tampered = json.loads(json.dumps(data))
            if mutate == "vessel":
                tampered["vessels"]["blueboat_v2"]["trigger_gap_m"] *= 1.01
            else:
                tampered["rule_constants"]["trigger_ttc_ratio"] = 5.0
            path = Path(folder) / ("tampered_%s.json" % mutate)
            path.write_text(ja.document_text(tampered), encoding="ascii")
            try:
                ja.load_avoidance("blueboat_v2", "path_following", path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("tampered constants (%s) must be refused" % mutate)


def test_gate_distances_cover_every_d1_reveal_of_the_obstacle_suites():
    """trigger = 6 t_avoid u_ref covers the appearing D1 trigger gap of the active ladder (M: 5.0 m per vessel; A:
    4 t_avoid u_ref), the largest calibration-sweep gap of path following (BlueBoat 7.5 m, WAM-V 15.0 m; G20c) and
    every D7 D1 detection range the dose table publishes for these tasks (transit crossing; stationary crossing /
    head-on / overtaking); clear = 8 t_avoid u_ref > trigger; both inside the 36-ray range; static tolerance = one
    plan cell."""
    dynamic = jc.shared_module("conditions.dynamic")
    appearing = jc.shared_module("conditions.appearing.config")
    from tasks._shared.conditions.dynamic import dose_table
    for vessel in VESSELS:
        c = ja.load_avoidance(vessel, "path_following")["vessel_constants"]
        envelope = ja.d7_envelope(vessel)
        assert c["t_avoid_s"] == envelope.t_avoid_s and c["reference_speed_mps"] == 1.0
        assert c["trigger_gap_m"] == 6.0 * envelope.t_avoid_s * 1.0
        assert c["clear_gap_m"] == 8.0 * envelope.t_avoid_s * 1.0 > c["trigger_gap_m"]
        assert c["trigger_gap_m"] > appearing.trigger_gap_m(1, envelope.t_avoid_s)
        assert c["trigger_gap_m"] > appearing.start_gap_m("path_following", envelope.t_avoid_s, vessel)
        plan = ja.journal_plan("path_following", vessel)
        assert c["clear_gap_m"] < float(plan.params["ray_sensor_max_range_m"])
        assert c["static_mask_tolerance_m"] == float(plan.params["grid_cell_m"])
        table = dose_table.vessel_dose_table(envelope, sensor_range_m=float(plan.params["ray_sensor_max_range_m"]))
        for motion, kind in (("transit", "crossing"), ("stationary", "crossing"), ("stationary", "head_on"),
                             ("stationary", "overtaking")):
            d1_range = table["detection_m"][motion][kind]["worst"][0]      # COM -> surface range >= footprint gap
            assert d1_range < c["trigger_gap_m"], (vessel, motion, kind, d1_range)
        assert c["obstacle_radius_m"] == appearing.RADIUS_LENGTH_FACTOR * jc.length_scale(vessel)
        assert dynamic.DOSE_TTC_RATIOS[1] == 4.0


# ------------------------------------------------------------------ gate --
def test_inactive_layer_returns_the_base_tensor_and_leaves_the_base_state_alone():
    for mode in ("path_follow", "station_keep", "docking"):
        env, controller, twin, _ = build(mode, "blueboat_v2", n=3, origins=[[0, 0, 0], [500, 0, 0], [0, 900, 0]])
        rng = torch.get_rng_state()
        gen = torch.Generator().manual_seed(7)
        for step in range(120):
            position = torch.rand((3, 2), generator=gen) * 4.0 - torch.tensor([30.0 if mode == "docking" else 0.0, 0.0])
            yaw = torch.rand(3, generator=gen) * 6.0 - 3.0
            velocity = torch.rand((3, 3), generator=gen)
            env.set_state(position, yaw, velocity=velocity)
            reference = twin.act(None)
            out = controller.act(None)
            assert out is not None and torch.equal(out, reference), (mode, step)
            for a, b in zip(base_state(controller.base_controller), base_state(twin)):
                assert torch.equal(a, b), mode
            if step % 40 == 39:
                done = torch.tensor([True, False, step % 80 == 79])
                controller.reset(done)
                twin.reset(done)
        assert torch.equal(torch.get_rng_state(), rng)
        assert controller.counters["activations"] == 0 and not any(controller.active)


def test_identity_of_the_returned_tensor_when_nothing_is_seen():
    env, controller, _, _ = build("path_follow", "blueboat_v2", n=2)
    calls = []
    original = controller.base_controller.act

    def spy(obs=None):
        out = original(obs)
        calls.append(out)
        return out
    controller.base_controller.act = spy
    out = controller.act(None)
    assert out is calls[-1]


def test_gate_hysteresis_with_the_g17_steps_memory_rule_comparison_arm():
    """G17's rule "steps" (the G24b comparison arm): the hand-back waits obstacle_memory_steps after the last return."""
    for vessel in VESSELS:
        env, c, twin, _ = build("path_follow", vessel, n=1, rules={"obstacle_memory_rule": "steps"})
        assert c.memory_rule == "steps" and c.carver is None

        def step(position, rays=None):
            env.set_state(position, [0.0], rays)
            return twin.act(None), c.act(None)
        eps = 1.0e-3 * c.trigger
        # just outside the trigger gap: inactive, the base tensor
        reference, out = step([[0.0, 0.0]], obstacle_rays_at_gap(env, c, c.trigger + eps))
        assert not c.active[0] and torch.equal(out, reference)
        # inside: active
        reference, out = step([[0.0, 0.0]], obstacle_rays_at_gap(env, c, c.trigger - eps))
        assert c.active[0] and c.counters["activations"] == 1
        # between trigger and clear: stays active
        step([[0.0, 0.0]], obstacle_rays_at_gap(env, c, 0.5 * (c.trigger + c.clear)))
        assert c.active[0]
        # just beyond the clear gap but off the path by more than the rejoin tolerance: stays active
        off = float(c.tc["rejoin_tolerance_m"]) * 1.5
        for _ in range(c.memory_steps + 2):
            step([[2.0, off]], obstacle_rays_at_gap(env, c, c.clear + eps))
        assert c.active[0]
        # back on the path, but a return was inside the clear gap less than obstacle_memory_steps ago: stays active
        step([[2.0, 0.1 * off]], obstacle_rays_at_gap(env, c, 0.5 * (c.trigger + c.clear)))
        for _ in range(c.memory_steps):
            step([[2.0, 0.1 * off]], obstacle_rays_at_gap(env, c, c.clear + eps))
            assert c.active[0]
        # clear for longer than the memory and on the path: hands back; this and later steps return the base tensor
        reference, out = step([[2.0, 0.1 * off]], obstacle_rays_at_gap(env, c, c.clear + eps))
        assert not c.active[0] and c.counters["handbacks"] == 1 and torch.equal(out, reference)
        reference, out = step([[2.5, 0.1 * off]])
        assert torch.equal(out, reference) and not c.active[0]


def test_g24b_memory_holds_without_expiry_and_hands_back_once_passed():
    """Rule "passed" (production): after the obstacle vanishes the remembered disc holds, and the hand-back waits,
    for 10 x G17's 61-step period with no ray near its surface point; rays then passing it on both sides drop it
    (memory_dropped) and the layer hands back at once (vessel on the path)."""
    for vessel in VESSELS:
        env, c, twin, _ = build("path_follow", vessel, n=1)
        assert c.memory_rule == "passed" and c.avoidance["rule_constants"]["obstacle_memory_rule"] == "passed"
        rays = obstacle_rays_at_gap(env, c, 0.5 * c.trigger)             # inside the trigger gap: activates
        env.set_state([[0.0, 0.0]], [0.0], rays)
        c.act(None)
        assert c.active[0]
        point = c.memory[0][2][0].copy()                                  # the remembered surface point (ray 0)
        assert c.memory[0][2].shape == (1, 2) and len(c.memory[0][2]) <= RAYS
        half = math.pi / RAYS                                             # half a ray spacing: no ray near the point
        for _ in range(10 * 61):
            env.set_state([[0.0, 0.0]], [half])
            c.act(None)
            assert c.active[0] and c.memory_in_use[0]
        assert c.counters["memory_steps"] >= 610 and c.counters["memory_dropped"] == 0
        see_through_both_sides(env, c, (0.0, 0.0), point)
        assert c.counters["memory_dropped"] == 1 and not c.active[0] and c.counters["handbacks"] == 1
        steps = c.counters["active_steps"]
        env.set_state([[0.5, 0.0]], [0.0])
        c.act(None)
        assert not c.active[0] and c.counters["active_steps"] == steps     # the base law acts again


def test_g24b_memory_point_outside_the_clear_gap_is_kept_but_not_in_force():
    env, c, _, _ = build("path_follow", "blueboat_v2", n=1)
    env.set_state([[0.0, 0.0]], [0.0], obstacle_rays_at_gap(env, c, 0.5 * c.trigger))
    c.act(None)
    point = c.memory[0][2][0].copy()
    far = (float(point[0]) - (c.clear + 2.0 * c.footprint[1] + 1.0), 0.0)      # the point lies beyond the clear gap
    env.set_state(torch.tensor([far], dtype=torch.float32), [math.pi / RAYS])
    c.act(None)
    assert not c.memory_in_use[0] and len(c.memory[0][2]) == 1 and c.counters["memory_dropped"] == 0
    assert not c._memory_holds(0)


def test_inactive_rows_stay_bitwise_equal_while_another_env_is_active():
    env, controller, twin, _ = build("path_follow", "blueboat_v2", n=2, origins=[[0, 0, 0], [300, 0, 0]])
    rays = torch.full((2, RAYS), env.max_range)
    rays[0, 0] = controller.footprint[1] + 0.5 * controller.trigger
    for step in range(30):
        env.set_state(torch.tensor([[0.05 * step, 0.0], [0.05 * step, 0.2]]), torch.tensor([0.0, 0.1]), rays)
        reference = twin.act(None)
        out = controller.act(None)
        assert controller.active[0] and not controller.active[1]
        assert torch.equal(out[1], reference[1])
        assert not torch.equal(out[0], reference[0]) or step == 0


# ------------------------------------------------------------ behaviours --
def test_path_following_rejoins_the_path_beyond_the_obstacle():
    from classical_planner_pid import hazard_geometry
    hg = hazard_geometry()
    for vessel in VESSELS:
        env, c, _, _ = build("path_follow", vessel, n=1)
        s = jc.length_scale(vessel)
        radius = 0.8 * s
        cyl = (0.6 * c.trigger + radius + c.footprint[1], 0.2 * s)
        rays = point_return_rays(torch.zeros((1, 2)), torch.zeros(1), [[cyl]], [[radius]], env.max_range)
        env.set_state([[0.0, 0.0]], [0.0], rays)
        c.act(None)
        assert c.active[0]
        plan, goal = c.plans[0], c.plan_goal[0]
        assert abs(goal[1]) < 1e-9, "the rejoin point lies on the reference path"
        assert goal[0] > cyl[0] + radius, "the rejoin point lies beyond the obstacle"
        # every planned segment clears the TRUE cylinder by the hull-fit rung (minus the grid slack)
        for a, b in zip(plan[:-1], plan[1:]):
            assert not hg.direct_segment_blocked(a, b, np.array([cyl]), np.array([radius]),
                                                 inflation_m=c.keep_inflation - 0.06 * s)
        assert np.linalg.norm(plan[-1] - goal) < 1e-9
        # G24b: the cylinder is gone (no return) and the rays pass its remembered surface point on both sides, the
        # vessel on the path: hand back (no expiry would release it otherwise)
        point = c.memory[0][2][0].copy()
        see_through_both_sides(env, c, (0.0, 0.0), point)
        assert not c.active[0]


def test_station_keeping_evades_holds_and_returns():
    for vessel in VESSELS:
        env, c, _, _ = build("station_keep", vessel, n=1)
        s = jc.length_scale(vessel)
        station = env.hold_point[0].double().numpy()
        radius = 0.8 * s
        intruder = station + np.array([0.0, 2.0 * s])
        pos = torch.tensor([station], dtype=torch.float32)
        env.set_state(pos, [0.0], point_return_rays(pos.double(), torch.zeros(1), [[tuple(intruder)]], [[radius]],
                                                   env.max_range))
        c.act(None)
        assert c.active[0] and c.plan_key[0] == "evade" and c.plan_hold[0]
        target = c.evade_target[0]
        assert np.linalg.norm(target - station) <= float(c.tc["evade_radius_m"]) + 1e-9
        # minimal-deviation rule: the evasion point is SAFE (clearance from the padded disc >= fattest rung + reach +
        # one re-plan period at u_ref) and no safe candidate lies nearer the station
        host = c._host(c.observe(jc.read_journal_inputs(env)))
        keep = host["clear_mask"][0]
        centers = host["points"][0][keep] + c.r_obs * host["direction"][0][keep]
        radii = c.r_obs + (host["ranges"][0][keep] + c.r_obs) * c.bearing_pad
        clearance = lambda p: float(np.min(np.linalg.norm(centers - p, axis=1) - radii))  # noqa: E731
        assert clearance(target) >= c.evade_clearance
        assert np.linalg.norm(target - intruder) - radius > c.ladder[0] + c.reach, "evasion point clear of the disc"
        for p in c._sk_candidates(station):
            if clearance(p) >= c.evade_clearance:
                assert np.linalg.norm(p - station) >= np.linalg.norm(target - station) - 1e-9
        # the intruder left (no return) with the vessel outside the hold radius: return to the station, still active
        away = station + np.array([0.0, -2.0 * float(c.tc["return_tolerance_m"])])
        for _ in range(c.memory_steps + 1):                # the last seen disc is remembered for one re-plan period
            env.set_state(torch.tensor([away], dtype=torch.float32), [0.0])
            c.act(None)
        assert c.active[0] and c.plan_key[0] == "return" and np.allclose(c.plan_goal[0], station)
        inside = station + np.array([0.5 * float(c.tc["return_tolerance_m"]), 0.0])
        for _ in range(c.memory_steps + 1):
            env.set_state(torch.tensor([inside], dtype=torch.float32), [0.0])
            c.act(None)
        assert not c.active[0]


def _docking_scenarios(vessel, count=6):
    from tasks.docking.journal_docking import build_scenario
    return [build_scenario(vessel, 2, 42, env, 0) for env in range(count)]


def test_docking_wall_reconstruction_equals_the_env_walls():
    for vessel in VESSELS:
        for scenario in _docking_scenarios(vessel):
            env, c, _, _ = build("docking", vessel, n=1)
            theta = scenario.berth_heading_rad
            env.dock_heading = torch.tensor([[math.cos(theta), math.sin(theta)]], dtype=torch.float32)
            walls, radii, center, _ = c._walls_local(1)
            assert walls.shape[1] == len(scenario.wall_radii)
            assert float((walls[0] - torch.as_tensor(scenario.wall_centers)).abs().max()) < 1e-5
            assert torch.equal(radii[0], torch.as_tensor(scenario.wall_radii))
            assert float(center.abs().max()) == 0.0


def test_docking_berth_walls_never_trigger_the_layer():
    """The env's own walls (journal_docking.build_scenario) and ray kernel (journal_kernels.ray_circle_ranges,
    float64 cast to float32 as DockingEnv._journal_get_observations does) at many collision-free poses around
    and inside the berth: every wall return is explained, the gate never fires; a cylinder at the mouth does."""
    from tasks.docking.journal_kernels import fan_angles, polygon_circle_clearance, ray_circle_ranges
    plans = jc.shared_module("journal_cfg_v2")
    gen = np.random.default_rng(17)
    for vessel in VESSELS:
        hull = torch.tensor(plans.vessel_record(vessel).footprint.vertices_m, dtype=torch.float64)
        checked = 0
        for scenario in _docking_scenarios(vessel, 4):
            env, c, _, _ = build("docking", vessel, n=1)
            theta = scenario.berth_heading_rad
            env.dock_heading = torch.tensor([[math.cos(theta), math.sin(theta)]], dtype=torch.float32)
            walls = torch.as_tensor(scenario.wall_centers).unsqueeze(0)
            radii = torch.as_tensor(scenario.wall_radii).unsqueeze(0)
            x0, x1, y0, y1 = scenario.bounds_berth_frame
            for _ in range(160):
                bx, by = gen.uniform(x0, x1), gen.uniform(y0, y1)
                yaw = gen.uniform(-math.pi, math.pi)
                point = np.array((math.cos(theta) * bx - math.sin(theta) * by,
                                  math.sin(theta) * bx + math.cos(theta) * by))
                com = torch.tensor([point], dtype=torch.float64)
                yaw_t = torch.tensor([yaw], dtype=torch.float64)
                if float(polygon_circle_clearance(com, yaw_t, hull, walls, radii)[0]) <= 0.0:
                    continue
                rays = ray_circle_ranges(com, fan_angles(yaw_t, RAYS), walls, radii, env.max_range).float()
                env.set_state(com.float(), yaw_t.float(), rays)
                seen = c.observe(jc.read_journal_inputs(env))
                assert not bool(seen["unexplained"].any()), (vessel, bx, by, yaw)
                assert not bool(seen["near_trigger"].any())
                checked += 1
        assert checked > 300, checked
        # a cylinder in front of the mouth on the berth axis is NOT explained and fires the gate
        s = jc.length_scale(vessel)
        x_open = float(c.tc["slip_opening_x_m"])
        mouth = np.array((math.cos(theta) * (x_open - 3.0 * s), math.sin(theta) * (x_open - 3.0 * s)))
        com = torch.tensor([mouth - np.array((math.cos(theta), math.sin(theta))) * 4.0 * s], dtype=torch.float64)
        yaw_t = torch.tensor([theta], dtype=torch.float64)
        both_c = torch.cat((walls, torch.tensor([[mouth]], dtype=torch.float64)), 1)
        both_r = torch.cat((radii, torch.tensor([[0.8 * s]], dtype=torch.float64)), 1)
        rays = ray_circle_ranges(com, fan_angles(yaw_t, RAYS), both_c, both_r, env.max_range).float()
        env.set_state(com.float(), yaw_t.float(), rays)
        seen = c.observe(jc.read_journal_inputs(env))
        assert bool(seen["unexplained"].any()) and bool(seen["near_trigger"].any())


def test_docking_wall_registration_absorbs_own_pose_error_but_not_an_obstacle():
    """Perception displaces the own-pose estimate (noise / bias: a shift; latency / dropout with the current heading: a
    turn). Measured on the env's own walls and ray kernel at collision-free poses around and inside the berth:
    * a shift within the static tolerance never fires the gate (BlueBoat 0.25 m, WAM-V 1.0 m < 1.12 m);
    * at larger errors (1 m, 0.15 rad) the registration removes a large part of the phantom triggers of the plain mask
      (the rest: fewer than wall_registration_min_new_returns displaced wall returns, indistinguishable from an object);
    * with an exact pose an obstacle at the mouth is never registered away, and it still fires under a pose error."""
    from tasks.docking.journal_kernels import fan_angles, polygon_circle_clearance, ray_circle_ranges
    plans = jc.shared_module("journal_cfg_v2")
    for vessel in VESSELS:
        gen = np.random.default_rng(23)
        hull = torch.tensor(plans.vessel_record(vessel).footprint.vertices_m, dtype=torch.float64)
        scenario = _docking_scenarios(vessel, 1)[0]
        env, c, _, _ = build("docking", vessel, n=1)
        theta = scenario.berth_heading_rad
        env.dock_heading = torch.tensor([[math.cos(theta), math.sin(theta)]], dtype=torch.float32)
        walls = torch.as_tensor(scenario.wall_centers).unsqueeze(0)
        radii = torch.as_tensor(scenario.wall_radii).unsqueeze(0)
        x0, x1, y0, y1 = scenario.bounds_berth_frame
        small = 0.25 if vessel == "blueboat_v2" else 1.0
        counts = {"small": [0, 0], "shift": [0, 0], "turn": [0, 0]}
        checked = 0
        for k in range(150):
            bx, by = gen.uniform(x0, x1), gen.uniform(y0, y1)
            truth = torch.tensor([[math.cos(theta) * bx - math.sin(theta) * by, math.sin(theta) * bx + math.cos(theta) * by]],
                                 dtype=torch.float64)
            yaw = torch.tensor([gen.uniform(-math.pi, math.pi)], dtype=torch.float64)
            if float(polygon_circle_clearance(truth, yaw, hull, walls, radii)[0]) <= 0.0:
                continue
            checked += 1
            rays = ray_circle_ranges(truth, fan_angles(yaw, RAYS), walls, radii, env.max_range).float()
            direction = gen.uniform(-math.pi, math.pi)
            unit = torch.tensor([[math.cos(direction), math.sin(direction)]], dtype=torch.float64)
            for case, error, turn in (("small", unit * small, 0.0), ("shift", unit * 1.0, 0.0),
                                      ("turn", unit * 0.0, 0.15 * (1 if k % 2 else -1))):
                env.set_state((truth + error).float(), (yaw + turn).float(), rays)
                inputs = jc.read_journal_inputs(env)
                pos, yw = inputs.position_local.double(), inputs.yaw.double()
                points, _, valid = ja.ray_points(pos, yw, inputs.rays.double(), c.max_range_m, c.no_return)
                w, wr, _, _ = c._walls_local(1)
                plain = valid & ~ja.explained_by_walls(points, w, wr, c.tol)
                gap = ja.footprint_gap(ja.to_body(points, pos, yw), c.footprint)
                counts[case][0] += bool((plain & (gap <= c.trigger)).any())
                counts[case][1] += bool(c.observe(inputs)["near_trigger"].any())
        print("registration %s (%d poses): phantom triggers plain -> registered %s" % (vessel, checked, counts))
        assert checked > 80 and counts["small"] == [0, 0]
        for case in ("shift", "turn"):
            assert counts[case][1] <= counts[case][0]
        assert counts["shift"][1] + counts["turn"][1] <= 0.75 * (counts["shift"][0] + counts["turn"][0]) or             counts["shift"][0] + counts["turn"][0] == 0
        s = jc.length_scale(vessel)
        x_open = float(c.tc["slip_opening_x_m"])
        h = np.array((math.cos(theta), math.sin(theta)))
        mouth = h * (x_open - 1.5 * s)
        truth = torch.tensor([mouth - h * 3.0 * s], dtype=torch.float64)
        yaw = torch.tensor([theta], dtype=torch.float64)
        both_c = torch.cat((walls, torch.tensor([[mouth]], dtype=torch.float64)), 1)
        both_r = torch.cat((radii, torch.tensor([[0.25 * float(plans.vessel_record(vessel).beam_m)]], dtype=torch.float64)), 1)
        rays = ray_circle_ranges(truth, fan_angles(yaw, RAYS), both_c, both_r, env.max_range).float()
        fired = 0
        for error, turn in (((0.0, 0.0), 0.0), ((0.7, -0.7), 0.0), ((0.0, 0.0), 0.15)):
            env.set_state((truth + torch.tensor([error], dtype=torch.float64)).float(), (yaw + turn).float(), rays)
            fired += bool(c.observe(jc.read_journal_inputs(env))["near_trigger"].any())
        assert fired == 3, (vessel, fired)


def test_docking_plans_and_evasion_targets_stay_clear_of_the_walls():
    from classical_planner_pid import hazard_geometry
    from tasks.docking.journal_kernels import fan_angles, ray_circle_ranges
    hg = hazard_geometry()
    for vessel in VESSELS:
        scenario = _docking_scenarios(vessel, 1)[0]
        env, c, _, _ = build("docking", vessel, n=1)
        s = jc.length_scale(vessel)
        theta = scenario.berth_heading_rad
        h = np.array((math.cos(theta), math.sin(theta)))
        env.dock_heading = torch.tensor([h], dtype=torch.float32)
        walls = torch.as_tensor(scenario.wall_centers).unsqueeze(0)
        radii = torch.as_tensor(scenario.wall_radii).unsqueeze(0)
        # approach: spawn-like pose, intruder between the vessel and the mouth
        x_open = float(c.tc["slip_opening_x_m"])
        start = h * (x_open - 10.0 * s)
        intruder = h * (x_open - 5.0 * s) + np.array((-h[1], h[0])) * 0.3 * s
        com = torch.tensor([start], dtype=torch.float64)
        yaw_t = torch.tensor([theta], dtype=torch.float64)
        both_c = torch.cat((walls, torch.tensor([[intruder]], dtype=torch.float64)), 1)
        both_r = torch.cat((radii, torch.tensor([[0.8 * s]], dtype=torch.float64)), 1)
        rays = ray_circle_ranges(com, fan_angles(yaw_t, RAYS), both_c, both_r, env.max_range).float()
        env.set_state(com.float(), yaw_t.float(), rays)
        c.act(None)
        assert c.active[0] and c.plan_key[0] == "approach" and not c.fallback[0]
        plan = c.plans[0]
        wall_c, wall_r = scenario.wall_centers, scenario.wall_radii
        for a, b in zip(plan[:-1], plan[1:]):
            assert not hg.direct_segment_blocked(a, b, wall_c, wall_r, inflation_m=c.keep_inflation - 0.06 * s), \
                "the approach plan crosses a wall"
        # near the berth (inside the mouth) with an intruder at the berth centre: evade inside the berth-bounded set
        inside = h * (0.5 * (x_open + float(c.tc["slip_back_face_x_m"])))
        intruder = h * (-0.5 * s)
        com = torch.tensor([inside], dtype=torch.float64)
        both_c = torch.cat((walls, torch.tensor([[intruder]], dtype=torch.float64)), 1)
        rays = ray_circle_ranges(com, fan_angles(yaw_t, RAYS), both_c, both_r, env.max_range).float()
        env.set_state(com.float(), yaw_t.float(), rays)
        c.reset(torch.tensor([True]))
        c.act(None)
        assert c.active[0] and c.plan_key[0] == "evade"
        target = c.evade_target[0]
        clearance = np.min(np.linalg.norm(wall_c - target, axis=1) - wall_r)
        assert clearance > c.ladder[-1], "evasion target inside the walls"
        # ... and with the obstacle outside the slip (not at the berth) the vessel holds course for the berth centre
        intruder_out = h * (x_open - 0.5 * s)
        both_out = torch.cat((walls, torch.tensor([[intruder_out]], dtype=torch.float64)), 1)
        rays = ray_circle_ranges(com, fan_angles(yaw_t, RAYS), both_out, both_r, env.max_range).float()
        env.set_state(com.float(), yaw_t.float(), rays)
        c.reset(torch.tensor([True]))
        c.act(None)
        assert c.active[0] and c.plan_key[0] == "return", (vessel, c.active[0], c.plan_key[0])
        host = c._host(c.observe(jc.read_journal_inputs(env)))
        for p in c._berth_candidates(0, host):
            assert np.min(np.linalg.norm(wall_c - p, axis=1) - wall_r) > c.ladder[-1]


def test_moving_intruder_replans_at_the_journal_planner_cadence_without_prediction():
    env, c, _, _ = build("path_follow", "blueboat_v2", n=1)
    assert c.replan_every == 60 and c.avoidance["rule_constants"]["replan_every_steps"] == 60
    block = ja.controller_block("path_follow", c.avoidance, hold_zone=c.hold_zone, berth_walls=False)
    assert block["motion_prediction"] is False
    s = 1.0
    cyl = np.array((2.0, 3.0))
    pos = torch.zeros((1, 2))
    env.set_state(pos, [0.0], point_return_rays(pos.double(), torch.zeros(1), [[tuple(cyl)]], [[0.8]], env.max_range))
    c.act(None)
    assert c.active[0]
    plans_after_activation = c.counters["plans"]
    first_plan = c.plans[0].copy()
    # the intruder crosses onto the planned route; the plan is kept until the cadence check, which uses the
    # journal planner's counter rule (classical_planner_pid.PlannerLOSPIDController.act: plan, then
    # replan_every decrements, then the check), i.e. at step replan_every + 1 after the plan ...
    for step in range(1, 62):
        cyl = cyl + np.array((0.0, -0.04))
        env.set_state(pos, [0.0], point_return_rays(pos.double(), torch.zeros(1), [[tuple(cyl)]], [[0.8]],
                                                   env.max_range))
        c.act(None)
        if step <= 60:
            assert c.counters["plans"] == plans_after_activation, step
            assert np.array_equal(c.plans[0], first_plan)
    # ... where the blocked plan is replaced (re-plan from the current map; no motion model)
    assert c.counters["plans"] == plans_after_activation + 1
    assert s == 1.0


def test_closing_return_lifts_the_speed_cap_but_a_static_one_does_not():
    """Closing test: own speed of the nearest return toward the vessel = -(range rate + own velocity . bearing) over
    closing_window_steps; above closing_speed_ratio x u_ref the return moves: no u_ref cap; a static return (the vessel
    itself closes at 1 m/s) keeps the cap."""
    for case in ("closing", "static_own_motion"):
        env, c, _, _ = build("path_follow", "blueboat_v2", n=1)
        window = c.closing_window
        start = c.footprint[1] + 0.8 * c.trigger
        for step in range(window + 5):
            rays = torch.full((1, RAYS), env.max_range)
            velocity = torch.tensor([[2.0, 0.0, 0.0]])                 # the vessel runs at 2 m/s toward the return
            if case == "closing":
                rays[0, 0] = start - (3.2 / 60.0) * step              # ... which itself moves at 1.2 m/s toward it
            else:
                rays[0, 0] = start - (2.0 / 60.0) * step              # ... which is static
            env.set_state([[0.0, 0.0]], [0.0], rays, velocity=velocity)
            out = c.act(None)
        assert c.active[0]
        if case == "closing":
            assert c.closing[0] and c.counters["closing_steps"] > 0
            assert float(out[0, 0]) > 0.0, "a closing return: no u_ref speed cap (pass-through throttle)"
        else:
            assert not c.closing[0] and c.counters["closing_steps"] == 0
            assert float(out[0, 0]) == -1.0, "a static return: the u_ref cap brakes a vessel at 2 m/s"


def test_station_keeping_evades_off_the_line_of_sight_of_a_closing_return():
    for vessel in VESSELS:
        env, c, _, _ = build("station_keep", vessel, n=1)
        s = jc.length_scale(vessel)
        station = env.hold_point[0].double().numpy()
        pos = torch.tensor([station], dtype=torch.float32)
        for step in range(c.closing_window + 3):
            intruder = station + np.array([0.0, 0.7 * c.trigger - 0.02 * s * step])     # straight at the station
            env.set_state(pos, [0.0], point_return_rays(pos.double(), torch.zeros(1), [[tuple(intruder)]], [[0.8 * s]],
                                                       env.max_range))
            c.act(None)
        assert c.active[0] and c.closing[0] and c.plan_key[0] == "evade" and c.evade_side[0] != 0
        target = c.evade_target[0]
        # the line of sight runs along y through the station: the evasion point is off it by disc + evade clearance
        radius = 0.8 * s + (np.linalg.norm(intruder - station) - 0.8 * s + 0.8 * s) * c.bearing_pad
        assert abs(target[0] - station[0]) >= radius + c.evade_clearance - 0.5 * s, (vessel, target, station)


def test_no_forbidden_member_is_read_in_any_mode():
    for mode in ("path_follow", "station_keep", "docking"):
        for vessel in VESSELS:
            env, c, _, _ = build(mode, vessel, n=1)
            s = jc.length_scale(vessel)
            rays = torch.full((1, RAYS), env.max_range)
            rays[0, 3] = 2.0 * s
            start = [[-15.0 * s, 0.0]] if mode == "docking" else [[0.0, 0.0]]
            env.set_state(start, [0.0], rays)
            for _ in range(5):
                c.act(None)
            assert c.active[0]
            try:
                env.obstacle_centers
            except ForbiddenAccess:
                pass
            else:
                raise AssertionError("the synthetic env must trap forbidden members")


# ------------------------------------------------------------- validator --
def _header(mode="path_follow", vessel="blueboat_v2"):
    env, c, _, block = build(mode, vessel, n=1)
    block["telemetry"] = c.telemetry()
    configuration = {"avoidance": block, "hold_mode": ja.journal_hold_mode(mode)}     # classical_baseline's block
    return {"controller": {"name": ja.CONTROLLER_NAMES[mode], "configuration": configuration},
            "vessel": vessel, "journal": {"task": MODE_TASK[mode]}, "complete": True}


def test_validator_checks_the_avoidance_block():
    for mode in ("path_follow", "station_keep", "docking"):
        assert validator.avoidance_layer_errors(_header(mode)) == [], mode
    header = _header()
    header["controller"]["name"] = "los_pid"
    assert validator.avoidance_layer_errors(header)            # a block on a non-layer certificate
    header = _header()
    del header["controller"]["configuration"]["avoidance"]
    assert validator.avoidance_layer_errors(header)            # a layer certificate without the block
    for path, value in ((("vessel_constants", "trigger_gap_m"), 1.0), (("vessel_constants", "vessel"), "wamv_vrx_v2_opt1"),
                        (("task",), "docking"), (("telemetry", "activations"), -1)):
        header = _header()
        target = header["controller"]["configuration"]["avoidance"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert validator.avoidance_layer_errors(header), path
    header = _header()
    header["controller"]["configuration"]["avoidance"]["telemetry"] = None
    assert validator.avoidance_layer_errors(header)            # a complete certificate needs the telemetry
    plain = {"controller": {"name": "los_pid", "configuration": {}}, "vessel": "blueboat_v2"}
    assert validator.avoidance_layer_errors(plain) == []       # pre-G17 journal certificates stay valid
    # G24b pins: station keeping must have run continuous_v2; the docking run-up point and the memory rule
    header = _header("station_keep")
    header["controller"]["configuration"]["hold_mode"] = "deadband"
    assert validator.avoidance_layer_errors(header)
    header = _header("docking")
    header["controller"]["configuration"]["avoidance"]["task_constants"]["runup_x_m"] += 0.5
    assert validator.avoidance_layer_errors(header)
    header = _header("path_follow")
    header["controller"]["configuration"]["avoidance"]["rule_constants"]["obstacle_memory_rule"] = "steps"
    assert validator.avoidance_layer_errors(header)


def test_g24b_names_base_laws_and_hold_mode_resolution():
    assert ja.CONTROLLER_NAMES == {"path_follow": "los_pid_avoid_v2", "station_keep": "los_pid_cont_avoid_v2",
                                   "docking": "docking_runup_pid_avoid_v2"}
    assert ja.BASE_CONTROLLER_NAMES == {"path_follow": "los_pid", "station_keep": "los_pid_cont",
                                        "docking": "docking_runup_pid"}
    assert ja.LAYER_VERSION == "g24b_avoidance_v2" and validator.AVOIDANCE_LAYER_VERSIONS == (ja.LAYER_VERSION,)
    assert {name: set(tasks) for name, tasks in validator.AVOIDANCE_CONTROLLERS.items()} == {
        ja.CONTROLLER_NAMES[m]: {MODE_TASK[m]} for m in MODE_TASK}
    assert ja.journal_hold_mode("station_keep") == ja.journal_hold_mode("station_keep", "continuous_v2") == "continuous_v2"
    for bad in ("deadband", "continuous"):
        try:
            ja.journal_hold_mode("station_keep", bad)
        except ValueError:
            pass
        else:
            raise AssertionError("journal station keeping must refuse --hold-mode " + bad)
    assert ja.journal_hold_mode("path_follow") == ja.journal_hold_mode("docking") == "deadband"
    for vessel in VESSELS:
        for mode in MODE_TASK:
            env, c, twin, block = build(mode, vessel, n=1)
            assert block["base_controller_name"] == ja.BASE_CONTROLLER_NAMES[mode]
            law = type(c.base_controller.dock).__name__
            assert law == ("RunupDockingLaw" if mode == "docking" else "DockingLaw"), (mode, law)
            assert c.base_controller.hold_mode == ja.journal_hold_mode(mode)
        tc = ja.load_avoidance(vessel, "docking")["task_constants"]
        runup = float(jc.shared_module("journal_cfg_v2").vessel_record(vessel).geometry.approach_runup_m)
        assert tc["approach_runup_m"] == runup and tc["runup_x_m"] == tc["slip_opening_x_m"] - runup < 0.0
        spec = ja.runup_spec("docking", ja.load_avoidance(vessel, "docking"))
        assert spec == {"runup_x_m": tc["runup_x_m"],
                        "accept_m": ja.load_avoidance(vessel, "docking")["vessel_constants"]["accept_radius_m"]}


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name, flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            traceback.print_exc()
            print("FAIL", name, type(exc).__name__, exc, flush=True)
    print("%d tests, %d failures" % (len(tests), failures))
    raise SystemExit(1 if failures else 0)
