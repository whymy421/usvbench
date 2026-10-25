"""G20b appearing obstacle on journal docking (REAL env methods, stubbed Isaac Lab, CPU).

The REAL DockingEnv journal branch on test_journal_docking_env's stub DirectRLEnv (imported as a module, its
tests are not collected here) with the REAL F0 mixin, HydroV2Runtime, condition layer, E4 bridge and the G3
AppearingBridge, and appearing pools admitted on the REAL scenarios with the active ladder and the G20b docking
rule (start margin 0, berth exclusion) (BlueBoat 4 x 2, WAM-V 2 x 1), PINNED in a temporary family folder
(appearing_pin_fixture; the production DIAG42 pools live in the pool bundle):
  * wiring: own one-slot storage (slot 65 = 64 static + 1 D7), the header names the pinned pool and the
    collision model; the cylinder lies on the approach lane, outside the berth rectangle;
  * trigger mechanics: following the nominal plan, every witnessed dose triggers at the first control step whose
    gap is <= d_k, within one control step of motion; the cylinder is visible at once (the 36-ray block of the
    observation sees it) and physical (its swept-contact latch terminates the contract with a collision);
  * D0 is bit-identical to the suite absent (observations, rewards, dones, poses, condition / D7 stamps, the
    contract results, every other pairing primitive);
  * a forced not-admitted key and a key beyond the grid run D0 with the D20 / D24 stamps.
Run: python -B tasks/docking/test_journal_docking_appearing.py
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import test_journal_docking_env as T  # noqa: E402  (stubs + REAL env helpers; its tests are not collected here)

AC = importlib.import_module("tasks._shared.conditions.appearing.config")
AP = importlib.import_module("tasks._shared.conditions.appearing.placement")
APOOL = importlib.import_module("tasks._shared.conditions.appearing.pool")
AST = importlib.import_module("tasks._shared.conditions.appearing.stamp")
ADM = importlib.import_module("tasks._shared.conditions.appearing.admission")
FAMS = importlib.import_module("tasks._shared.conditions.appearing.families")
TASK = "docking"
CONTROL_DT = 1.0 / 60.0
GRIDS = {"blueboat_v2": (4, 2), "wamv_vrx_v2_opt1": (2, 1)}
_spec = importlib.util.spec_from_file_location(
    "appearing_pin_fixture", T.REPO / "tasks" / "_shared" / "conditions" / "appearing" / "tests" / "appearing_pin_fixture.py")
PIN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PIN)
PINNED = {}
UNPIN = None


def setUpModule():
    global UNPIN
    for vessel, (envs, episodes) in GRIDS.items():
        PINNED[vessel] = PIN.build(ADM, FAMS, TASK, vessel, envs, episodes, seed=T.SEED)
    _, UNPIN = PIN.install(APOOL, FAMS, list(PINNED.values()))


def tearDownModule():
    if UNPIN is not None:
        UNPIN()


def make(vessel="blueboat_v2", tokens=(), num_envs=T.N, appearing_pool=None):
    cfg = T.make_cfg(vessel, tokens, num_envs=num_envs)
    if appearing_pool is not None:
        cfg.journal_appearing_pool = appearing_pool
    return T.docking_env.DockingEnv(cfg)


def yaw_of(env):
    q = env.robot.data.root_link_pose_w[:, 3:7]
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def plan_targets(env, row, spacing_s=0.5):
    a = env._appearing_bridge.rows[row]["assignment"]
    x, y, _ = a.cylinder
    t_site = a.placement["site"]["t_site_s"]
    context = FAMS.admission_inputs(TASK, env.cfg.vehicle, 2, T.SEED, row, int(a.key.episode))[0]
    plan = AP.nominal_plan(context, ADM.admission_suite(env.cfg.vehicle))
    scale = float(env._jd_scale)
    out, t = [], spacing_s
    while t < t_site:
        pose = plan.pose_at(t)
        out.append((pose.x, pose.y, 0.4 * scale))
        t += spacing_s
    return out + [(x, y, 0.0)]


def pursue(env, targets, cursor, thrust=0.45):
    com = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
    yaw = yaw_of(env)
    actions = []
    for i in range(env.num_envs):
        goal = targets[i][min(cursor[i], len(targets[i]) - 1)]
        while cursor[i] < len(targets[i]) - 1 and \
                math.hypot(goal[0] - float(com[i, 0]), goal[1] - float(com[i, 1])) <= max(goal[2], 0.2):
            cursor[i] += 1
            goal = targets[i][cursor[i]]
        bearing = math.atan2(goal[1] - float(com[i, 1]), goal[0] - float(com[i, 0]))
        error = math.atan2(math.sin(bearing - float(yaw[i])), math.cos(bearing - float(yaw[i])))
        actions.append([thrust * max(0.2, math.cos(error)), max(-1.0, min(1.0, 1.5 * error))])
    return torch.tensor(actions, dtype=torch.float32)


def ray_block(env, obs, row):
    """The 36 ranges (m) of an observation (journal_dock6_ray36: six native channels, then rays / max range)."""
    return obs["policy"][row, 6:42].to(torch.float64) * float(env.cfg.ray_max_range_m)


def expected_ray_block(env, row, cylinder):
    """G20c (REV-G20b m8): (base, cylinder, expected) 36 ranges (m) at the CURRENT pose from journal_kernels,
    independent of the env's appearing storage: base = the berth-wall rays (no D7 intruder runs in these cells, so
    the D7 slots add nothing in either the ray-circle or the G23c point-return merge), cylinder = the ray-circle
    ranges of the assigned cylinder, expected = their elementwise minimum."""
    kernels = importlib.import_module(T.docking_env.__name__.rsplit(".", 1)[0] + ".journal_kernels")
    pose = env.robot.data.root_link_pose_w
    origin = (pose[:, :2] - env.scene.env_origins[:, :2]).to(torch.float64)
    angles = kernels.fan_angles(kernels.yaw_from_quat(pose[:, 3:7].to(torch.float64)), env.cfg.ray_count)
    rmax = float(env.cfg.ray_max_range_m)
    base = kernels.ray_circle_ranges(origin, angles, env._jd_wall_centers, env._jd_wall_radii, rmax)
    x, y, r = cylinder
    centers = torch.tensor([[[float(x), float(y)]]], dtype=torch.float64).expand(env.num_envs, 1, 2)
    radii = torch.full((env.num_envs, 1), float(r), dtype=torch.float64)
    cyl = kernels.ray_circle_ranges(origin, angles, centers, radii, rmax)
    return base[row], cyl[row], torch.minimum(base, cyl)[row]


def admitted_row(pool, num_envs, episode=0):
    for env in range(num_envs):
        if pool.is_admitted(env, episode):
            return env
    return None


def end_episode(env, rows):
    for row in rows:
        env.robot.data.root_link_pose_w[row, 0] += 1.0e3


class Wiring(unittest.TestCase):
    def test_slot_storage_header_and_lane(self):
        env = make(tokens=("appearing=D2",))
        self.assertEqual(env._appearing_slot, T.plans.STATIC_CAPACITY[TASK] + T.plans.D7_RESERVED_SLOTS[TASK])
        self.assertEqual(tuple(env._appearing_centers.shape), (T.N, 1, 2))
        header = env.journal_certificate_header()["appearing_hazards_header"]
        self.assertTrue(header["pool_pinned"])
        self.assertTrue(header["pool_file"].endswith("docking__blueboat_v2__s42.json.xz"))
        self.assertEqual(header["grid"], {"num_envs": 4, "episodes": 2})
        self.assertIn("collision_model", header)
        self.assertFalse(hasattr(make(), "_appearing_centers"))
        for vessel, pool in PINNED.items():
            for ident in pool.offered:
                doc = pool.placement(*ident)
                if doc is None:
                    continue
                site, cyl = doc["site"], doc["cylinder"]
                hw = 0.5 * T.jd.d7_envelope(T.dynamic, vessel).footprint.beam_m
                self.assertGreaterEqual(AP.berth_clearance(site["berth_exclusion"], cyl["x_m"], cyl["y_m"],
                                                           cyl["radius_m"]), hw - 1e-9)
                self.assertEqual(site["start_rule"]["margin_lengths"], 0.0)


class TriggerAtDeclaredGap(unittest.TestCase):
    def run_dose(self, vessel, dose):
        pool = PINNED[vessel]
        row = admitted_row(pool, GRIDS[vessel][0])
        self.assertIsNotNone(row, "no admitted key at episode 0 (%s)" % vessel)
        env = make(vessel, ("appearing=D%d" % dose,), num_envs=max(2, row + 1))
        obs, _ = env.reset()
        bridge = env._appearing_bridge
        targets = [plan_targets(env, i) if bridge.rows[i]["active"] else [(0.0, 0.0, 0.0)]
                   for i in range(env.num_envs)]
        cursor = [0] * env.num_envs
        x, y, r = bridge.rows[row]["cylinder"]
        footprint = bridge.footprint
        d_k = AC.trigger_gap_m(dose, bridge.t_avoid_s)
        for _ in range(int(300.0 / CONTROL_DT)):
            pose = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
            self.assertFalse(bool(env._appearing_visible[row, 0] | env._appearing_physical[row, 0]))
            rays_before = ray_block(env, obs, row)
            obs, _, te, tr, _ = env.step(pursue(env, targets, cursor))
            if bool((te | tr)[row]):
                self.fail("episode ended before the trigger: %s"
                          % env.journal_episode_record(row)["appearing_hazards"]["payload"]["runtime"]
                          ["termination_reason"])
            if bridge.triggered(row):
                trig = bridge.rows[row]["trigger"]
                now = env._journal_poses(env.robot.data, env.scene.env_origins)[row]
                gap = AP.gap_at(now, x, y, r, footprint)
                self.assertAlmostEqual(trig["gap_m"], gap, places=9)
                self.assertLessEqual(trig["gap_m"], d_k)
                self.assertGreater(trig["previous_check_gap_m"], d_k)
                motion = math.hypot(now.x - pose.x, now.y - pose.y) + \
                    footprint.reach_m * abs(math.remainder(now.yaw - pose.yaw, 2 * math.pi))
                self.assertGreater(trig["gap_m"], d_k - motion - 1e-9)
                self.assertTrue(bool(env._appearing_visible[row, 0] & env._appearing_physical[row, 0]))
                self.assertAlmostEqual(float(env._appearing_centers[row, 0, 0]), x, places=9)
                rays_after = ray_block(env, obs, row)
                self.assertTrue(bool((rays_after < rays_before - 1e-6).any()) or gap > float(env.cfg.ray_max_range_m))
                # G20c (REV-G20b m8): the observed block IS min(base rays, cylinder ray-circle ranges)
                self.assertFalse(bool(env._d7_visible[row].any()))
                base, cyl, expected = expected_ray_block(env, row, (x, y, r))
                self.assertTrue(torch.allclose(rays_after, expected, rtol=0.0, atol=1e-5),
                                (rays_after - expected).abs().max())
                if gap < float(env.cfg.ray_max_range_m):
                    self.assertTrue(bool((cyl < base).any()))            # the cylinder is what the rays now see
                return env, row
        self.fail("the trigger gap was never reached")

    def test_every_dose_triggers_at_the_declared_gap_blueboat(self):
        # G48/G49 (owner 2026-10-14): BlueBoat docking D1 is not applicable by design (D1 = 7.5 m leaves no route
        # window after the start rule); the env refuses appearing=D1 at reset, every other witnessed dose triggers
        na = AC.not_applicable_doses("docking", "blueboat_v2")
        self.assertEqual(tuple(na), (1,))
        for dose in AC.WITNESSED_DOSES:
            with self.subTest(dose=dose):
                if dose in na:
                    with self.assertRaisesRegex(ValueError, "not applicable by design"):
                        make("blueboat_v2", ("appearing=D%d" % dose,), num_envs=2).reset()
                    continue
                self.run_dose("blueboat_v2", dose)

    def test_wamv_triggers_at_the_declared_gap(self):
        self.run_dose("wamv_vrx_v2_opt1", AC.MAX_DOSE)

    def test_contact_terminates_with_a_collision(self):
        env, row = self.run_dose("blueboat_v2", 2)          # G48/G49: D2 = the largest offered BlueBoat docking gap
        x, y, _ = env._appearing_bridge.rows[row]["cylinder"]
        targets = [[(x, y, 0.0)] for _ in range(env.num_envs)]
        cursor = [0] * env.num_envs
        for _ in range(int(120.0 / CONTROL_DT)):
            _, _, te, tr, _ = env.step(pursue(env, targets, cursor, thrust=0.6))
            if bool((te | tr)[row]):
                break
        stamp = env.journal_episode_record(row)["appearing_hazards"]["payload"]
        self.assertEqual(stamp["runtime"]["termination_reason"], "collision")
        self.assertTrue(stamp["runtime"]["swept_contact_latched"])


class ZeroDoseIdentity(unittest.TestCase):
    def check(self, vessel):
        def run(tokens):
            env = make(vessel, tokens, num_envs=2)
            obs, _ = env.reset()
            out = [obs["policy"].clone()]
            records, results = [], []
            for k in range(40):
                if k == 30:
                    end_episode(env, (1,))
                o, r, te, tr, _ = env.step(T.actions(k, env.num_envs))
                out += [o["policy"].clone(), r.clone(), te.clone(), tr.clone(),
                        env.robot.data.root_link_pose_w.clone()]
                for i in (te | tr).nonzero().flatten().tolist():
                    records.append(copy.deepcopy(env.journal_episode_record(i)))
                    results.append(json.dumps(env.journal_episode_results[i], sort_keys=True, default=str))
            return env, out, records, results
        absent, a, ra, qa = run(())
        present, b, rb, qb = run(("appearing=D0",))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(qa, qb)
        self.assertEqual(len(ra), len(rb))
        self.assertGreater(len(ra), 0)
        for x, y in zip(ra, rb):
            self.assertNotIn("appearing_hazards", x)
            self.assertEqual(y["appearing_hazards"]["payload"]["assignment"]["dose"], 0)
            for field in ("conditions", "dynamic_hazards"):
                self.assertEqual(json.dumps(x[field], sort_keys=True), json.dumps(y[field], sort_keys=True), field)
            hy = dict(y["scenario_hashes"])
            hy.pop(T.pairing.APPEARING_PRIMITIVE)
            self.assertEqual(x["scenario_hashes"], hy)
        self.assertEqual(absent.dynamic_hazard_header, present.dynamic_hazard_header)

    def test_d0_is_bit_identical_to_the_suite_absent_blueboat(self):
        self.check("blueboat_v2")

    def test_d0_is_bit_identical_to_the_suite_absent_wamv(self):
        self.check("wamv_vrx_v2_opt1")


class NotAdmittedAndBeyondGrid(unittest.TestCase):
    def test_forced_not_admitted_key_runs_d0(self):
        pool = PIN.forced(APOOL, PINNED["blueboat_v2"], ((1, 0),))
        env = make(tokens=("appearing=D3",), num_envs=2, appearing_pool=pool)
        env.reset()
        self.assertFalse(env._appearing_bridge.rows[1]["active"])
        env.step(T.actions(0, 2))
        end_episode(env, (1,))
        env.step(T.actions(1, 2))
        record = env.journal_episode_record(1)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("forced_test_rejection",)))
        payload = record["appearing_hazards"]["payload"]
        self.assertEqual((payload["runtime"]["assigned"], payload["runtime"]["itt_count"]), (False, 0))

    def test_beyond_grid_key_runs_d0_with_the_single_reason(self):
        env = make(tokens=("appearing=D2",), num_envs=2)
        env.reset()
        for _ in range(2):                                   # episodes 0 and 1 end early (out of bounds)
            env.step(T.actions(0, 2))
            end_episode(env, (0, 1))
            env.step(T.actions(1, 2))
        self.assertFalse(env._appearing_bridge.rows[0]["active"])
        env.step(T.actions(2, 2))
        end_episode(env, (0,))
        env.step(T.actions(3, 2))
        record = env.journal_episode_record(0)
        kind, reasons = AST.exposure_of(record["appearing_hazards"])
        self.assertEqual((kind, reasons), ("not_admitted", ("beyond_offered_grid",)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
