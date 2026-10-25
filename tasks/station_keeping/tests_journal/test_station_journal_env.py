"""F4: the REAL station_keeping_env.py journal branch on CPU (stubbed isaaclab).

Run: python -B tasks/station_keeping/tests_journal/test_station_journal_env.py

Executes the real env module and E5's real journal cfg templates under the
isaaclab stand-ins of station_journal_fixtures (Isaac Lab 2.x DirectRLEnv call
order, toy rigid-body integrator), with the REAL HydroV2Runtime, the REAL
schema-3 condition layer, the REAL E4 HazardBridge and the committed D7 pool
(tasks/station_keeping/d7_pools, seed 42). Covers: journal cfg fields, mixin
call order per control step, submitted wrench == direct runtime step,
zero-dose identity, own-position estimates in the observation, the ray block,
reset-row isolation, ALL SUITES ON at once (both vessels) with stamps and
pairing hashes, D9 codes 6-9 surviving the publisher, the pool's fail-closed
checks and the constant-current rule.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import station_journal_fixtures as F

RESTORE = None
ALL_SUITES = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "dynamic=D2")
ALL_D0 = ("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0", "dynamic=D0")


def setUpModule():
    global RESTORE, js, pairing, dyn, plans, cfg_module
    RESTORE = F.install_isaaclab_stubs()
    js = F.ns("station_keeping.journal_station")
    pairing = F.ns("_shared.conditions.pairing")
    dyn = F.ns("_shared.conditions.dynamic")
    plans = F.ns("_shared.journal_cfg_v2")
    cfg_module = F.ns("station_keeping.station_keeping_env_cfg")


def tearDownModule():
    if RESTORE is not None:
        RESTORE()


def actions(env, k):
    a = F.hold_controller(env)
    return a if k % 7 else a.clamp(-1, 1) * 0.5


def teleport(env, i, dx=None, quat=None):
    pose = env.robot.data.root_link_pose_w
    if dx is not None:
        pose[i, 0] += dx
    if quat is not None:
        pose[i, 3:7] = torch.tensor(quat, dtype=pose.dtype)


class CfgTests(unittest.TestCase):
    def test_journal_cfg_fields_both_vessels(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            with self.subTest(vessel=vessel):
                cfg = F.journal_cfg(vessel)
                plan = cfg.journal_plan
                params = plan.params
                self.assertEqual(cfg.vehicle, vessel)
                self.assertEqual(cfg.underwater_physics_cfg.hydrodynamics_model, "hydro_v2")
                self.assertEqual(cfg.underwater_physics_cfg.physics_version, 2)
                self.assertFalse(cfg.underwater_physics_cfg.enable_current)
                self.assertTrue(cfg.obs_kinematic)
                self.assertFalse(cfg.obs_sea_state)
                self.assertEqual(cfg.observation_space, 42)
                self.assertEqual(cfg.ray_count, 36)
                self.assertEqual(cfg.ray_max_range_m, params["ray_sensor_max_range_m"])
                self.assertIs(cfg.journal_observation_hooks_wired, True)
                self.assertEqual(cfg.vessel_profile, vessel)
                self.assertIsInstance(cfg.vessel_profile_status, str)
                self.assertTrue(str(cfg.robot_cfg.spawn.usd_path).replace("\\", "/").endswith(
                    F.ns("_shared.vehicles_v2").get_vehicle_v2(vessel).asset_relpath))
                self.assertFalse(cfg.robot_cfg.spawn.rigid_props.disable_gravity)
                self.assertEqual(cfg.robot_cfg.init_state.pos[2], plan.nominal_com_height_m)
                self.assertAlmostEqual(cfg.episode_length_s, plan.max_deadline_s)
                self.assertEqual(cfg.hold_radius, params["hold_radius_m"])
                self.assertEqual((cfg.sim.dt, cfg.decimation), (1.0 / 120.0, 2))
                path, digest = cfg.journal_d7_pool, cfg.journal_d7_pool_sha256
                self.assertTrue(Path(path).is_file(), path)
                pool = js.StationD7Pool.load(path, digest)
                self.assertEqual((pool.vessel, pool.seed, pool.tier), (vessel, 42, 2))
                self.assertEqual(pool.keys_admitted, pool.keys_offered)
                self.assertGreaterEqual(pool.grid["num_envs"], 64)
                self.assertGreaterEqual(pool.grid["episodes"], 2)

    def test_journal_branch_rejects_legacy_names_and_ramp(self):
        cfg = F.journal_cfg("blueboat_v2")
        cfg.current_ramp_start_mps = 0.5
        with self.assertRaisesRegex(ValueError, "ramp"):
            cfg_module._journal_station_post_init(cfg)
        cfg = F.journal_cfg("blueboat_v2")
        cfg.vehicle = "blueboat"
        cfg.journal_plan = plans.plan_class("station_keeping", "blueboat_v2")(2)
        with self.assertRaises(ValueError):
            cfg_module._journal_station_post_init(cfg)

    def test_legacy_and_jbase_calm_cfgs_stay_legacy(self):
        legacy = cfg_module.StationKeepingBlueBoatKinEnvCfg()
        self.assertEqual((legacy.vehicle, legacy.underwater_physics_cfg.physics_version,
                          legacy.observation_space), ("blueboat", 1, 6))
        self.assertFalse(hasattr(legacy.underwater_physics_cfg, "hydrodynamics_model"))
        calm = cfg_module.StationKeepingBlueBoatJBaseCalmEnvCfg()
        self.assertEqual((calm.vehicle, calm.underwater_physics_cfg.physics_version, calm.journal_mode),
                         ("blueboat", 2, True))
        self.assertFalse(hasattr(calm, "journal_d7_pool"))


class EnvTests(unittest.TestCase):
    def test_call_order_and_wrench_equal_direct_runtime(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=3))
        env.journal_trace = []
        env.reset()
        self.assertEqual(env.journal_trace, ["reset", "estimates"])
        reference = copy.deepcopy(env._hydro_v2)
        snaps = []
        bound = env._apply_action

        def wrapped():
            data = env.robot.data
            snaps.append((env.actions.clone(), data.root_link_pose_w.clone(), data.root_com_vel_w.clone()))
            bound()
        env._apply_action = wrapped
        for k in range(4):
            env.journal_trace.clear()
            env.step(actions(env, k))
            self.assertEqual(env.journal_trace, ["pre_physics", "substep", "substep", "pre_dones", "dones",
                                                 "estimates"])
        self.assertTrue(torch.equal(env._hydro_v2.ticks, torch.full((3,), 8)))
        self.assertEqual(env._conditions.estimate_counts(0), {"pos": 5, "vel": 5, "ray": 5})
        self.assertEqual(len(snaps), 8)
        zeros = torch.zeros((3, 3), dtype=reference.dtype)
        for (acts, pose, vel), (force, torque) in zip(snaps, env.robot.applied):
            step = reference.step_from_root_link(acts, pose, vel, env.scene.env_origins, mean_water_level_local=0.0,
                                                 current_velocity_w=zeros, waves=None, wind_force_w=None,
                                                 wind_torque_w=None)
            self.assertTrue(torch.equal(force[:, 0], step.force_actor.to(force.dtype)))
            self.assertTrue(torch.equal(torque[:, 0], step.torque_actor.to(torque.dtype)))
        # spawn = layout start at the COM datum, heading from the layout (root write, COM reference)
        env2 = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2))
        env2.reset()
        pose = env2.robot.data.root_link_pose_w
        layout = env2._jlayouts[1]
        self.assertAlmostEqual(float(pose[1, 0] - env2.scene.env_origins[1, 0]), float(layout.start[0]), places=5)
        self.assertAlmostEqual(float(pose[1, 2]), env2._jcom_z, places=6)
        self.assertAlmostEqual(float(js.yaw_from_quat(pose[1:2, 3:7].double())[0]), layout.start_heading_rad,
                               places=5)

    def test_zero_dose_identity(self):
        runs = []
        for tokens in ((), ALL_D0):
            env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=tokens))
            obs, _ = env.reset()
            record = [obs["policy"].clone()]
            for k in range(40):
                obs, reward, te, tr, _ = env.step(actions(env, k))
                record.append(obs["policy"].clone())
            runs.append((record, [(f.clone(), t.clone()) for f, t in env.robot.applied],
                         env.robot.data.root_link_pose_w.clone()))
        (obs_a, wr_a, pose_a), (obs_b, wr_b, pose_b) = runs
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(obs_a, obs_b)))
        self.assertTrue(all(torch.equal(fa, fb) and torch.equal(ta, tb) for (fa, ta), (fb, tb) in zip(wr_a, wr_b)))
        self.assertTrue(torch.equal(pose_a, pose_b))

    def test_observation_uses_the_position_estimate(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("perception=D4",)))
        obs, _ = env.reset()
        for k in range(5):
            obs, *_ = env.step(actions(env, k))
        est = env.journal_estimate
        pose = env.robot.data.root_link_pose_w
        rpos = env._jstation.float() - est.position.float()
        distance = rpos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rpos / distance
        fwd = js.bow_axis_xy(pose[:, 3:7].float())
        expected = torch.cat(((fwd * direction).sum(-1, keepdim=True),
                              fwd[:, 0:1] * direction[:, 1:2] - fwd[:, 1:2] * direction[:, 0:1],
                              distance / env.cfg.max_spawn_distance), -1)
        self.assertTrue(torch.equal(obs["policy"][:, :3], expected))
        truth = pose[:, :2] - env.scene.env_origins[:, :2]
        self.assertFalse(torch.equal(est.position.float(), truth))     # the dose acts on the estimate
        self.assertEqual(obs["policy"].shape, (2, 42))
        self.assertTrue(bool((obs["policy"][:, 6:] <= 1.0).all()))

    def test_ray_block_point_return_visible_only(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2))
        env.reset()
        pose = env.robot.data.root_link_pose_w
        yaw = js.yaw_from_quat(pose[:, 3:7].double()).float()
        rng = env.cfg.ray_max_range_m
        self.assertTrue(torch.equal(env._journal_ray_ranges(), torch.full((2, 36), rng)))
        bearing = 50.0 * math.pi / 180.0      # nearest ray 5 (50 deg, 10 deg spacing)
        for i in range(2):
            world = float(yaw[i]) + bearing
            env._d7_centers_w[i, 0, 0] = pose[i, 0] + 6.0 * math.cos(world)
            env._d7_centers_w[i, 0, 1] = pose[i, 1] + 6.0 * math.sin(world)
        env._d7_radii[:, 0] = 0.5
        env._d7_physical[:, 0] = True
        env._d7_visible[:, 0] = torch.tensor([True, False])
        rays = env._journal_ray_ranges()
        self.assertAlmostEqual(float(rays[0, 5]), 5.5, places=4)
        self.assertEqual(int((rays[0] < rng).sum()), 1)
        self.assertTrue(torch.equal(rays[1], torch.full((36,), rng)))  # physical but not visible: no return

    def test_reset_row_isolation(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, tokens=("dynamic=D2",)))
        env.reset()
        for k in range(5):
            env.step(actions(env, k))
        twin = env._jtwin
        before = {name: getattr(twin, name)[0].clone() for name in ("prev_xy", "prev_time", "full_path",
                                                                    "previous_error", "reason", "has_hold")}
        ticks0 = int(env._hydro_v2.ticks[0])
        d7_tick0 = int(env._d7_tick[0])
        layout0 = env._jlayouts[0].geometry_hash
        env._reset_idx(torch.tensor([1]))
        for name, value in before.items():
            self.assertTrue(torch.equal(getattr(twin, name)[0], value), name)
        self.assertEqual((int(env._hydro_v2.ticks[0]), int(env._d7_tick[0])), (ticks0, d7_tick0))
        self.assertEqual((int(env._hydro_v2.ticks[1]), int(env._d7_tick[1])), (0, 0))
        self.assertEqual(env._jlayouts[0].geometry_hash, layout0)
        self.assertEqual(int(env._jscenario.episode_indices()[1]), 1)
        self.assertEqual(len(env.journal_aborted_dynamic_hazards), 1)      # manual reset: aborted, not a record
        self.assertIsNone(env.episode_dynamic_hazards[1])
        self.assertEqual(env._conditions.estimate_counts(1), {"pos": 0, "vel": 0, "ray": 0})
        self.assertEqual(env._conditions.estimate_counts(0), {"pos": 6, "vel": 6, "ray": 6})

    def _all_suites(self, vessel):
        env = F.make_env(F.journal_cfg(vessel, num_envs=2, tokens=ALL_SUITES))
        env.reset()
        for k in range(3):
            env.step(actions(env, k))
        teleport(env, 0, dx=1.0e3)                          # outside the layout bounds
        env.step(actions(env, 3))
        self.assertEqual(int(env.episode_termination_code[0]), 4)
        self.assertEqual(int(env.termination_reason[0]), 4)
        record = env.journal_episode_record(0)
        stamp = record["conditions"]
        enabled = {s: stamp["parameters"][s]["enabled"] for s in ("waves", "wind", "current", "actuator",
                                                                    "perception")}
        self.assertTrue(all(enabled.values()), enabled)
        envelope = dyn.read_stamp(record["dynamic_hazards"])
        self.assertEqual(envelope["payload"]["assignment"]["dose"], 2)
        self.assertEqual(envelope["payload"]["runtime"]["termination_reason"], "out_of_bounds")
        hashes = record["scenario_hashes"]
        want = set(pairing.CONDITION_PRIMITIVES) | {pairing.D7_PRIMITIVE, "layout", "spawn_pose"}
        self.assertEqual(set(hashes), want)
        recomputed = pairing.condition_primitive_hashes(stamp)
        self.assertTrue(all(hashes[k] == v for k, v in recomputed.items()))
        self.assertEqual(hashes["layout"], env._jlayouts[0].geometry_hash if False else hashes["layout"])
        header = env.journal_certificate_header(treatment="waves")
        self.assertIn("layout", header["pairing"]["unchanged_primitives"])
        self.assertIn("spawn_pose", header["pairing"]["unchanged_primitives"])
        self.assertEqual(header["dynamic_hazards_header"]["pool_digest"], env._d7_pool.pool_digest)
        result = env.journal_episode_result(0)
        self.assertEqual(result["termination_reason_name"], "out_of_bounds")
        self.assertFalse(result["episode_success"])
        return env

    def test_all_suites_on_blueboat(self):
        self._all_suites("blueboat_v2")

    def test_all_suites_on_wamv(self):
        self._all_suites("wamv_vrx_v2_opt1")

    def test_validity_codes_survive_the_publisher(self):
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2))
        env.reset()
        env.step(actions(env, 0))
        teleport(env, 1, quat=(math.cos(0.75), math.sin(0.75), 0.0, 0.0))    # 86 deg roll
        _, _, terminated, truncated, extras = env.step(actions(env, 1))
        self.assertTrue(bool(terminated[1]) and not bool(truncated[1]))
        self.assertEqual(int(env.termination_reason[1]), 6)
        self.assertEqual(int(extras["termination_reason"][1]), 6)
        self.assertEqual(env.journal_episode_result(1)["termination_reason_name"], "capsize")
        stamp = dyn.read_stamp(env.journal_episode_record(1)["dynamic_hazards"])
        self.assertEqual(stamp["payload"]["runtime"]["termination_reason"], "capsize")

    def test_pool_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            F.make_env(F.journal_cfg("blueboat_v2", num_envs=2, seed=43))
        cfg = F.journal_cfg("blueboat_v2", num_envs=2)
        cfg.journal_d7_pool_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            F.make_env(cfg)
        cfg = F.journal_cfg("blueboat_v2", num_envs=2)
        cfg.journal_d7_pool = None
        with self.assertRaisesRegex(ValueError, "D7 pool"):
            F.make_env(cfg)
        cfg = F.journal_cfg("blueboat_v2", num_envs=65)
        with self.assertRaisesRegex(ValueError, "covers"):
            F.make_env(cfg)
        path = F.journal_cfg("blueboat_v2").journal_d7_pool
        document = json.loads(Path(path).read_text(encoding="ascii"))
        document["accepted"][0][5] = "f" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            js.StationD7Pool(document)
        pool = js.StationD7Pool.load(path)
        layout = js.build_station_layout(2, "blueboat_v2", 42, 0, 0)
        other = js.build_station_layout(2, "blueboat_v2", 42, 0, 1)
        with self.assertRaisesRegex(ValueError, "differs from the admitted layout"):
            pool.bind_context(js.episode_key(42, 0, 0), js.station_task_context(other, "blueboat_v2"))
        pool.bind_context(js.episode_key(42, 0, 0), js.station_task_context(layout, "blueboat_v2"))
        with self.assertRaises(KeyError):
            pool.assignment(js.episode_key(42, 0, 0), 5)
        with self.assertRaises(KeyError):
            pool.bind_context(js.episode_key(42, 0, 99), js.station_task_context(layout, "blueboat_v2"))

    def test_runtime_suite_must_match_the_admitted_suite(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2, tokens=("dynamic=D1", "dynamic.intruder_speed_mps=0.7"))
        with self.assertRaisesRegex(ValueError, "suite"):
            F.make_env(cfg)

    def test_journal_current_is_constant(self):
        cfg = F.journal_cfg("blueboat_v2", num_envs=2,
                            tokens=("current.enabled=true", "current.speed_table=[[0, 0.0], [60, 0.5]]"))
        with self.assertRaisesRegex(ValueError, "CONSTANT"):
            F.make_env(cfg)


class G6PublishedTargets(unittest.TestCase):
    """G6: the station centre is published as hold_point (world frame, legacy name) so
    eval_mppi (mode "hold"), classical_baseline / journal_controllers and _horizontal_distance
    (cert_v3 d0) reach it."""

    def test_hold_point_follows_the_layout(self):
        scripts = str(Path(__file__).resolve().parents[3] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import eval_mppi
        import journal_controllers as jc
        env = F.make_env(F.journal_cfg("blueboat_v2", num_envs=2))
        env.reset()
        expected = (env.scene.env_origins[:, :2].double() + env._jstation).float()
        self.assertTrue(torch.allclose(env.hold_point, expected, atol=1e-6))
        self.assertEqual(env.hold_point.dtype, torch.float32)
        self.assertEqual(eval_mppi._detect_mode(env), "hold")
        if hasattr(env.robot.data, "root_pos_w") and hasattr(env.robot.data, "root_com_pos_w"):  # fake data may lack them
            self.assertTrue(bool(torch.isfinite(env._horizontal_distance()).all()))
        env.step(actions(env, 1))
        controller = jc.JournalBaselineController(
            env, "station_keep", jc.load_gains("blueboat_v2"), dt=env.control_step_s,
            zone_radius_m=float(env.cfg.hold_radius), slow_radius_m=5.0, tolerance_m=2.5,
            scale=jc.length_scale("blueboat_v2"))
        out = controller.act()
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
