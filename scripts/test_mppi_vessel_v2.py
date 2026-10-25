"""CPU tests of the MPPI registry-vessel model (F7 item 1). No Isaac.

* the rollout model's terminal speed / spin rate equal the D5 registry's
  top_model_m_s and E2's CPU_SCENARIOS anchors (surge_60s_mps, spin_60s_rad_s)
  within 1 percent, for BlueBoat and the WAM-V;
* the allocation IS the D3 mixer (radial saturation, per-motor limits,
  centreline arm);
* the model reproduces the REAL HydroV2Runtime (D5 profile, calm water)
  driving a toy 6-DOF integrator (F0 glue_fixtures.FakeRobot) on a manoeuvre;
* one model per vessel is selected from a journal cfg's vehicle, legacy cfgs
  keep the certified legacy model, and MPPI steers both vessels (goal and a
  ray-derived obstacle wall);
* round 4b: the per-vessel H / lambda are frozen by the pre-declared
  MPPI_TUNING_PROCEDURE_V2.md in mppi_params_v2.json (procedure digest
  stamped, choice rule reproduced from the recorded grid, fail closed on a
  tampered file or a changed horizon rule); journal ids refuse --H/--lambda.
* G4-A5: the WAM-V journal-default model carries the cross-flow sway and
  strip-theory yaw damping of the env profile (hydro_profiles swayyaw_damping);
  the E2 anchors belong to the VRX-classic comparison profile; the frozen file
  records the tuning model's damping and the superseded freeze, and a changed
  model fails closed.

Run: python -B scripts/test_mppi_vessel_v2.py   (or pytest)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import mppi_controller as mc  # noqa: E402

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
# E2 CPU_SCENARIOS.json "anchors" (20260930_e2_vessels_hydro_core/work/CPU_SCENARIOS.json,
# variants blueboat_v2_zeta0.1 and wamv_vrx_v2_opt1 at dt = 1/120 s): 60 s runtime
# values of the terminal surge speed (action (1, 0)) and zero-surge spin (action (0, 1)).
# E2 ran the WAM-V with the registry VRX-classic sway/yaw damping, i.e. G4-A5's
# swayyaw_damping="vrx_classic" comparison profile (E2_ANCHOR_RULES).
E2_ANCHORS = {"blueboat_v2": (3.0003265027199046, 1.255359387346585),
              "wamv_vrx_v2_opt1": (2.2589523496083315, 0.5135674964012265)}
E2_ANCHOR_RULES = {"blueboat_v2": "journal_default", "wamv_vrx_v2_opt1": "vrx_classic"}
E2_SCENARIOS = Path(r"D:\claude code\codex_tasks\20260930_e2_vessels_hydro_core\work\CPU_SCENARIOS.json")
REL = 0.01


def _drive(dyn, action, seconds):
    state = torch.zeros((1, 6), dtype=torch.float64)
    act = torch.tensor([action], dtype=torch.float64)
    for _ in range(int(round(seconds / dyn.control_dt_s))):
        state = dyn.step(state, act)
    return state[0]


def test_journal_default_spin_is_the_closed_form_of_the_profile_damping():
    # G4-A5: WAM-V journal default N0 = (400 + 1784.61 |r|) r -> 0.2452 rad/s (VRX classic 0.5136).
    dyn = mc.VesselDynamicsV2("wamv_vrx_v2_opt1", dtype=torch.float64)
    p = dyn.params
    torque = p["arm_m"] * 2.0 * min(p["reverse_limit_n"])
    closed = (-p["yaw_lin"] + math.sqrt(p["yaw_lin"] ** 2 + 4.0 * p["yaw_quad"] * torque)) / (2.0 * p["yaw_quad"])
    assert p["swayyaw_damping"] == "crossflow_sarda_strip" and abs(closed - 0.24524) < 1e-4
    assert math.isclose(dyn.terminal_yaw_rate(), closed, rel_tol=1e-12)
    spin = float(_drive(dyn, (0.0, 1.0), 60.0)[5])
    assert abs(spin - closed) <= REL * closed, (spin, closed)
    surge = float(_drive(dyn, (1.0, 0.0), 60.0)[3])
    assert abs(surge - p["top_model_m_s"]) <= REL * p["top_model_m_s"]


def test_terminal_speed_and_spin_match_registry_and_e2_anchors():
    for vessel in VESSELS:
        dyn = mc.VesselDynamicsV2(vessel, dtype=torch.float64, swayyaw_damping=E2_ANCHOR_RULES[vessel])
        surge = float(_drive(dyn, (1.0, 0.0), 60.0)[3])
        spin = float(_drive(dyn, (0.0, 1.0), 60.0)[5])
        top, anchor_spin = E2_ANCHORS[vessel]
        registry_top = dyn.params["top_model_m_s"]
        assert abs(surge - registry_top) <= REL * registry_top, (vessel, surge, registry_top)
        assert abs(surge - top) <= REL * top, (vessel, surge, top)
        assert abs(spin - anchor_spin) <= REL * anchor_spin, (vessel, spin, anchor_spin)
        assert math.isclose(dyn.terminal_speed_forward(), registry_top, rel_tol=1e-9)
        assert abs(dyn.terminal_yaw_rate() - anchor_spin) <= REL * anchor_spin


def test_embedded_anchors_equal_the_e2_file_when_available():
    if not E2_SCENARIOS.is_file():
        print("SKIP E2 CPU_SCENARIOS.json not reachable; embedded anchors used")
        return
    anchors = json.loads(E2_SCENARIOS.read_text(encoding="utf-8"))["anchors"]
    rows = {(a["profile"], a["variant"]): a for a in anchors if abs(a["dt"] - 1.0 / 120.0) < 1e-12}
    for vessel, variant in (("blueboat_v2", "blueboat_v2_zeta0.1"), ("wamv_vrx_v2_opt1", "wamv_vrx_v2_opt1")):
        row = rows[(vessel, variant)]
        assert (row["surge_60s_mps"], row["spin_60s_rad_s"]) == E2_ANCHORS[vessel]


def test_allocation_is_the_d3_mixer_with_radial_saturation():
    mixer = mc._shared_module("thruster_mixer")
    registry = mc._shared_module("vehicles_v2")
    torch.manual_seed(0)
    for vessel in VESSELS:
        dyn = mc.VesselDynamicsV2(vessel, dtype=torch.float64)
        spec = mixer.ThrusterSpec(**registry.to_d3_thruster_kwargs(registry.get_vehicle_v2(vessel)))
        actions = torch.rand((64, 2), dtype=torch.float64) * 2.0 - 1.0
        out = mixer.step_thrusters(actions, mixer.zero_state(actions), spec, 1.0 / 120.0)
        x, n = dyn.wrench(actions)
        assert torch.equal(x, out.body_wrench[:, 0]) and torch.equal(n, out.body_wrench[:, 5])
        # radial: a (1, 1) request is scaled by ONE factor, preserving X/N direction
        full = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        x1, n1 = dyn.wrench(full)
        p = dyn.params
        requested_x = sum(p["forward_limit_n"])
        requested_n = p["yaw_command_limit_nm"]
        assert math.isclose(float(x1) / float(n1), requested_x / requested_n, rel_tol=1e-9)
        # per-motor limits: the starboard motor sits exactly at its forward limit
        starboard = (float(x1) + float(n1) / p["arm_m"]) / 2.0
        assert math.isclose(starboard, p["forward_limit_n"][1], rel_tol=1e-9)
        # centreline arm: pure yaw gives N = arm * 2 * reverse limit
        _, n0 = dyn.wrench(torch.tensor([[0.0, 1.0]], dtype=torch.float64))
        assert math.isclose(float(n0), p["arm_m"] * 2.0 * min(p["reverse_limit_n"]), rel_tol=1e-9)


def test_model_reproduces_the_hydro_v2_runtime_calm_water():
    import glue_fixtures as gf

    profiles = gf.shared("hydro_profiles")
    runtime_module = gf.shared("hydro_v2_runtime")
    for vessel, rule in [(v, "journal_default") for v in VESSELS] + [("wamv_vrx_v2_opt1", "vrx_classic")]:
        profile = profiles.profile_for_vehicle(vessel, swayyaw_damping=rule)
        spec = gf.vehicles_v2.get_vehicle_v2(vessel)
        runtime = runtime_module.HydroV2Runtime(profile, 1, "cpu", 1.0 / 120.0, dtype=torch.float64,
                                                engine_gravity=(0.0, 0.0, -9.81), rigid_body_disable_gravity=False)
        robot = gf.FakeRobot(1, spec.mass_kg, spec.inertia_com_kg_m2, (0.0, 0.0, -9.81))
        robot.data.root_link_pose_w[0, 2] = profile.hydro.equilibrium_com_z_m
        origins = torch.zeros((1, 3), dtype=torch.float64)
        dyn = mc.VesselDynamicsV2(vessel, dtype=torch.float64, swayyaw_damping=rule)
        assert {k: dyn.params[k] for k in mc.PLANAR_DAMPING_KEYS} == mc.profile_planar_damping(profile)
        state = torch.zeros((1, 6), dtype=torch.float64)
        for k in range(480):
            action = torch.tensor([[0.8 if k < 240 else 0.3, 0.4 if k < 160 else -0.6]], dtype=torch.float64)
            for _ in range(2):
                step = runtime.step_from_root_link(action, robot.data.root_link_pose_w.double(),
                                                   robot.data.root_com_vel_w.double(), origins)
                robot.set_external_force_and_torque(step.force_actor[:, None, :], step.torque_actor[:, None, :])
                robot.integrate(1.0 / 120.0)
            state = dyn.step(state, action)
        pose = robot.data.root_link_pose_w[0].double()
        vel = robot.data.root_com_vel_w[0].double()
        q = pose[3:7]
        yaw = math.atan2(2 * float(q[0] * q[3] + q[1] * q[2]), 1 - 2 * float(q[2] ** 2 + q[3] ** 2))
        travelled = float(torch.linalg.vector_norm(state[0, :2]))
        assert float(torch.linalg.vector_norm(pose[:2] - state[0, :2])) <= 0.01 * travelled, vessel
        assert abs(math.remainder(yaw - float(state[0, 2]), 2 * math.pi)) <= 0.01, vessel
        assert abs(float(vel[5]) - float(state[0, 5])) <= 0.01 * max(abs(float(state[0, 5])), 0.1), vessel


def test_one_model_per_vessel_from_the_journal_cfg():
    def cfg(vehicle, journal=True, dt=1.0 / 120.0, decimation=2):
        return SimpleNamespace(journal_contract_version=2 if journal else 0, vehicle=vehicle,
                               underwater_physics_cfg=SimpleNamespace(hydrodynamics_model="hydro_v2"),
                               sim=SimpleNamespace(dt=dt), decimation=decimation)
    for vessel in VESSELS:
        model = mc.dynamics_for_cfg(cfg(vessel))
        assert isinstance(model, mc.VesselDynamicsV2) and model.vessel == vessel
    assert isinstance(mc.dynamics_for_cfg(cfg("blueboat", journal=False)), mc.NominalBlueBoatDynamics)
    for bad in (cfg("blueboat_v2", dt=1.0 / 60.0), cfg("blueboat_v2", decimation=4)):
        try:
            mc.dynamics_for_cfg(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a journal cfg off 1/120 s x 2 must be refused")
    legacy = cfg("blueboat_v2")
    legacy.underwater_physics_cfg.hydrodynamics_model = "legacy"
    try:
        mc.dynamics_for_cfg(legacy)
    except ValueError:
        pass
    else:
        raise AssertionError("a journal cfg without hydro_v2 must be refused")
    a = mc.vessel_params_v2("blueboat_v2")
    b = mc.vessel_params_v2("wamv_vrx_v2_opt1")
    assert a["mass_kg"] != b["mass_kg"] and a["arm_m"] != b["arm_m"]
    assert a["water_velocity_mps"] == [0.0, 0.0] and a["saturation"] == "radial"


def _closed_loop(vessel, goal, seconds, circles=()):
    """MPPI (per-vessel parameters, K=32) on the model; obstacles only from a ray fan."""
    import journal_controllers as jc
    from test_classical_harbour import ray_cast

    s = mc.vessel_length_scale(vessel)
    dyn = mc.VesselDynamicsV2(vessel)
    params = mc.mppi_params_for_vessel(vessel, mc.MPPIParams(K=32, seed=3))
    controller = mc.MPPIController(1, params=params, dynamics=dyn)
    state = torch.zeros((1, 6))
    targets = torch.tensor([[goal]], dtype=torch.float32)
    min_clearance = math.inf
    radius = dyn.params["capsule_radius_m"]
    for _ in range(int(seconds * 60)):
        kwargs = {}
        if circles:
            rays = ray_cast(state[:, :2].double(), state[:, 2].double(), circles, 30.0 * s).float()
            centers, radii, active = jc.ray_obstacles(state[:, :2], state[:, 2], rays, 30.0 * s)
            kwargs = dict(obstacle_centers=centers, obstacle_radii=radii, obstacle_active=active)
        action, _ = controller.rollout(state, targets=targets, goal_radius=2.0 * s, mode="goal", **kwargs)
        state = dyn.step(state, action)
        for cx, cy, r in circles:
            min_clearance = min(min_clearance, math.dist(state[0, :2].tolist(), (cx, cy)) - r - radius)
        if float(torch.linalg.vector_norm(state[0, :2] - targets[0, 0])) <= 2.0 * s:
            return True, min_clearance
    return False, min_clearance


def test_per_vessel_mppi_parameters_follow_the_declared_rule():
    base = mc.MPPIParams()
    assert mc.mppi_params_for_vessel("blueboat_v2", base) == base
    wamv = mc.mppi_params_for_vessel("wamv_vrx_v2_opt1", base)
    s = mc.vessel_length_scale("wamv_vrx_v2_opt1")
    assert wamv.H == round(60 * s) and math.isclose(wamv.clearance_margin_m, 0.6 * s)
    dt = 1.0 / 60.0
    assert math.isclose(wamv.w_dock_heading, 0.3 * dt * (wamv.H - 1) / 2.0)
    assert math.isclose(wamv.w_dock_speed, (wamv.H - 1) * dt / 0.6)
    assert (wamv.K, wamv.sigma, wamv.lambda_, wamv.w_goal) == (base.K, base.sigma, base.lambda_, base.w_goal)


def test_mppi_reaches_an_abeam_goal_with_both_vessel_models():
    for vessel, seconds in (("blueboat_v2", 20.0), ("wamv_vrx_v2_opt1", 60.0)):
        s = mc.vessel_length_scale(vessel)
        reached, _ = _closed_loop(vessel, (0.0, 6.0 * s), seconds)
        assert reached, vessel


def test_mppi_rounds_a_pillar_it_only_sees_through_the_rays():
    for vessel, seconds in (("blueboat_v2", 20.0), ("wamv_vrx_v2_opt1", 60.0)):
        s = mc.vessel_length_scale(vessel)
        reached, clearance = _closed_loop(vessel, (10.0 * s, 0.0), seconds, circles=[(5.0 * s, 0.3 * s, 1.0 * s)])
        assert reached and clearance > 0.0, (vessel, reached, clearance)


def test_frozen_mppi_parameters_follow_the_declared_procedure():
    import copy
    import hashlib
    import tempfile

    import tune_mppi_v2 as tune

    data, sha = mc.load_mppi_frozen()
    procedure = Path(_SCRIPTS) / "MPPI_TUNING_PROCEDURE_V2.md"
    assert data["procedure_sha256"] == hashlib.sha256(procedure.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert tuple(data["lambda_grid"]) == tune.LAMBDA_GRID
    for vessel in VESSELS:
        block = data["vessels"][vessel]
        params, stamp = mc.journal_mppi_params(vessel)
        declared = mc.mppi_params_for_vessel(vessel)
        assert params.H == declared.H == block["H"] and params.lambda_ == block["lambda"]
        assert (params.K, params.sigma, params.w_goal) == (declared.K, declared.sigma, declared.w_goal)
        assert stamp["sha256"] == sha and stamp["procedure_sha256"] == data["procedure_sha256"]
        assert stamp["rule"] == block["rule"] and stamp["accepted"] == block["accepted"]
        # the choice rule of section 4, re-applied to the recorded grid, gives the frozen choice
        points = {lam: dict(block["grid"]["%g" % lam]) for lam in tune.LAMBDA_GRID}
        chosen, rule, accepted, _ = tune.choose(vessel, points)
        assert (chosen, rule, accepted) == (block["lambda"], block["rule"], block["accepted"]), vessel
        # G4-A5: tuned on the current calm nominal model (journal-default sway/yaw rule)
        assert data["model_damping"][vessel] == mc.model_damping_record(vessel) == stamp["model_damping"]
    assert data["model_damping"]["wamv_vrx_v2_opt1"]["swayyaw_damping"] == "crossflow_sarda_strip"
    # the superseded T-base freeze (VRX-classic WAM-V model) is kept; the BlueBoat block did not change
    last = data["history"][-1]
    assert last["sha256"] == "63c8499965f68265ddb3f76863ce36403479268273c637f512d6968ce7250900"
    assert set(last["vessels"]) == {"wamv_vrx_v2_opt1"} and last["vessels"]["wamv_vrx_v2_opt1"]["lambda"] == 3.0
    with tempfile.TemporaryDirectory() as tmp:
        bad = copy.deepcopy(data)
        bad["status"] = "draft"
        path = Path(tmp) / "draft.json"
        path.write_text(json.dumps(bad), encoding="ascii")
        try:
            mc.journal_mppi_params("blueboat_v2", path=path)
            raise AssertionError("a non-frozen file must fail closed")
        except ValueError as exc:
            assert "frozen" in str(exc)
        bad = copy.deepcopy(data)
        bad["vessels"]["wamv_vrx_v2_opt1"]["H"] += 1
        path.write_text(json.dumps(bad), encoding="ascii")
        try:
            mc.journal_mppi_params("wamv_vrx_v2_opt1", path=path)
            raise AssertionError("a horizon that differs from the declared rule must fail closed")
        except ValueError as exc:
            assert "declared rule" in str(exc)
        for tamper in ({"yaw_quad": 0.0, "sway_quad": 0.0, "swayyaw_damping": "vrx_classic"}, None):
            bad = copy.deepcopy(data)
            if tamper is None:
                del bad["model_damping"]
            else:
                bad["model_damping"]["wamv_vrx_v2_opt1"].update(tamper)
            path.write_text(json.dumps(bad), encoding="ascii")
            try:
                mc.journal_mppi_params("wamv_vrx_v2_opt1", path=path)
                raise AssertionError("parameters tuned on another model must fail closed")
            except ValueError as exc:
                assert "model damping" in str(exc)


def test_journal_ids_refuse_mppi_horizon_and_temperature_flags():
    import eval_mppi

    parser = eval_mppi._build_parser()
    task = "Isaac-USV-Reach-BlueBoat-Journal-Direct-v1"
    ok = parser.parse_args(["--task", task])
    ok.journal_task_info = {"journal_task": "reach"}
    eval_mppi._validate_args(parser, ok)
    for flags in (["--lambda", "3.0"], ["--H", "90"]):
        args = parser.parse_args(["--task", task, *flags])
        args.journal_task_info = {"journal_task": "reach"}
        try:
            eval_mppi._validate_args(parser, args)
            raise AssertionError("journal ids must refuse " + flags[0])
        except SystemExit:
            pass
        legacy = parser.parse_args(["--task", "Isaac-USV-HazardCross-Direct-v1", *flags])
        eval_mppi._validate_args(parser, legacy)


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(1 if run_all() else 0)
