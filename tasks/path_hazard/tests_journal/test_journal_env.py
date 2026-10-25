"""F2 journal env tests: the REAL PathHazardEnv on stubbed Isaac Lab (CPU only).

Run: python -B tasks/path_hazard/tests_journal/test_journal_env.py

Covers the F-common item 8 checklist for the path-hazard family: legacy byte identity of the
DirectRLEnv hooks vs F-base; legacy cfg behaviour (BlueBoat-only asserts kept for legacy ids);
both journal cfgs (BlueBoat v2, WAM-V) construct through E5's templates (C6/C8); the substep call
order through the REAL HydroV2Runtime / condition layer / HazardBridge; zero-dose identity; ALL
suites on at once (every suite D2 + dynamic D2) runs 3 control steps and stamps; pairing hashes
present and pairing cleanly; termination codes 2/3/4/6 published by the D9 publisher; reset-row
isolation; storage width obstacle_count + 1 with split D7 masks (B04/B07); the legacy sampler is
unreachable from the journal branch; COM reference (never root_com_pos_w after a reset); every
position-derived observation channel comes from the D5 position estimate.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

def _d7_shown(env, row):
    """G23c L2 (rule S2): a row's D7 intruder is physical exactly when its runtime rule has revealed it
    (hidden before the reveal; rows without a runtime rule: physical when they carry an intruder)."""
    bridge = getattr(env._d7_bridge, "inner", env._d7_bridge)     # path hazard wraps it (NotAdmittedBridge)
    episode = bridge.adapter.episodes.get(int(row))
    if episode is None:
        return False
    if episode.release is None:
        return bool(episode.objects)
    return bool(episode.release.released_at(episode.last_time_s))


import ph_fixtures as F  # noqa: E402

M = F.modules()
STUBS = F.install_stubs()
ENV_FILE = F.REPO / "tasks" / "path_hazard" / "path_hazard_env.py"
# SHA-256 of the F-base sources (D:/claude code/codex_tasks/20260930_fbase/work/repo, MANIFEST-verified)
FBASE_METHOD_SHA256 = {
    "_pre_physics_step": "90d7ebb341cdbfe6cad09f8b064cdf422118d7fda3e58b0d814cdb993f7681f8",
    "_apply_action": "a718def543bc56284fb2c7dc61b4fd96e36811aae6d3d275a553463facddd6e5",
    "_get_observations": "88eab5a169b161aec848b3120bb6007e1e538ee1f42462a3af25c6e8814452a8",
    "_get_dones": "c7c9e372719578773e029a5419d5ff1c90bee8068f31be906b1e21e7d36f5d77",
    "_reset_idx": "8f894324f7fd0d5fda767fa9961faf74343cffd7ef04e28b60c269c7967ffe99",
    "_get_rewards": "2a33b79239c7b91596a877b37e1a4e2ddac9b5b6730e30c2b6a39e0e63f6cd5b",
    "_layout_generator": "3bb333b1aabed89d9cd7544757c53d9688aae4bd043ef40c5256e4e6bfc107b6",
    "_compute_buoyancy_forces": "40c0d619e900ff8dd334bbfb23d6b5ce81d84ba9d68482377318347eca70a709",
}
ALL_ON = F.ALL_ON + ("wind.offset_deg=30", "current.offset_deg=-45", "perception.offset_deg=90")
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def run(env, steps, start=0):
    for k in range(start, start + steps):
        env.step(F.actions(k, env.num_envs))


def admitted_rows(env):
    """Rows whose current key E4 admitted (the others run at D0, exposure not_admitted; D16)."""
    episodes = env._scenario.episode_indices().tolist()
    return [i for i in range(env.num_envs) if env._jph_pool.is_admitted(env.cfg.seed, i, episodes[i])]


def rearm_deadline(env, seconds):
    """Re-arm the contract rows with a short deadline so an episode ends inside a unit test."""
    n = env.num_envs
    routes = torch.cat((torch.zeros((n, 1, 2), dtype=torch.float64), env._jph_waypoints), dim=1)
    deadline = torch.full((n,), float(seconds), dtype=torch.float64)
    env._jph_oracle.reset(torch.arange(n), routes, env._jph_gate_radius, deadline, routes[:, 0])


def record_calls(obj, names, log, prefix):
    for name in names:
        original = getattr(obj, name)

        def wrapper(*args, _original=original, _name=name, **kwargs):
            log.append(prefix + "." + _name)
            return _original(*args, **kwargs)
        setattr(obj, name, wrapper)


class LegacyIdentity(unittest.TestCase):
    def test_legacy_hook_sources_are_byte_identical_to_fbase(self):
        source = ENV_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PathHazardEnv")
        found = {n.name: hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest()
                 for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in FBASE_METHOD_SHA256}
        self.assertEqual(found, FBASE_METHOD_SHA256)

    def test_legacy_cfgs_keep_their_asserts_and_defaults(self):
        cfg_module = STUBS["cfg"]
        for cls in (cfg_module.PathHazardEnvCfg, cfg_module.PathHazardV2EnvCfg, cfg_module.PathHazardWaveEnvCfg):
            cfg = cls()
            self.assertEqual(cfg.journal_contract_version, 1)
            self.assertEqual(cfg.vehicle, "blueboat")
            self.assertFalse(hasattr(cfg.underwater_physics_cfg, "hydrodynamics_model"))   # C2 field set
            self.assertFalse(hasattr(cfg.underwater_physics_cfg, "vessel_profile"))
            self.assertEqual(cfg.obstacle_count, 6)
            self.assertEqual(cfg.episode_length_s, 150.0)
        with self.assertRaisesRegex(ValueError, "fixes vehicle='blueboat'"):
            cfg_module.PathHazardEnvCfg(vehicle="wamv")
        with self.assertRaises((KeyError, ValueError)):
            cfg_module.PathHazardEnvCfg(vehicle="blueboat_v2")      # contract 1 never takes a v2 name

    def test_legacy_env_never_enters_the_journal_branch(self):
        cfg = STUBS["cfg"].PathHazardV2EnvCfg()
        cfg.seed = 3
        cfg.scene.num_envs = 2
        for flag in ("enable_water", "enable_obstacles", "enable_waypoint_markers", "enable_path_line"):
            setattr(cfg.visual, flag, False)
        env_cls = STUBS["env"].PathHazardEnv
        original = env_cls.journal_setup

        def refuse(self, *a, **k):
            raise AssertionError("journal_setup on a legacy cfg")
        env_cls.journal_setup = refuse
        try:
            env = F.make_env(cfg)
            env.reset()
            run(env, 3)
        finally:
            env_cls.journal_setup = original
        self.assertNotIn("_apply_action", env.__dict__)            # no per-instance rebinding
        self.assertEqual(tuple(env.obstacle_centers.shape), (2, 6, 2))
        self.assertEqual(tuple(env.termination_reason.shape), (2,))


class JournalCfg(unittest.TestCase):
    def test_both_journal_cfgs_construct_with_the_family_edits(self):
        for vessel in VESSELS:
            cfg = F.journal_cfg(vessel)
            plan = cfg.journal_plan
            record = M.plans.vessel_record(vessel)
            self.assertEqual(cfg.vehicle, vessel)
            self.assertEqual(cfg.journal_contract_version, 2)
            self.assertEqual(cfg.underwater_physics_cfg.physics_version, 2)
            self.assertEqual(cfg.underwater_physics_cfg.hydrodynamics_model, "hydro_v2")
            self.assertEqual(cfg.underwater_physics_cfg.vessel_profile, vessel)
            self.assertTrue(cfg.underwater_physics_cfg.vessel_profile_status)
            self.assertEqual(cfg.ray_count, 36)
            self.assertAlmostEqual(cfg.ray_max_range_m, 30.0 * record.length_scale)
            self.assertEqual(cfg.obstacle_count, M.plans.STATIC_CAPACITY["path_hazard"])
            self.assertEqual(cfg.observation_space, 46)
            self.assertEqual(cfg.observation_contract, "journal_kin46_ray36")
            self.assertTrue(cfg.journal_observation_hooks_wired)
            self.assertEqual(cfg.episode_length_s, plan.max_deadline_s)
            self.assertEqual(cfg.goal_radius, plan.params["gate_radius_m"])
            self.assertFalse(cfg.robot_cfg.spawn.rigid_props.disable_gravity)
            self.assertAlmostEqual(cfg.robot_cfg.init_state.pos[2], record.com_height_above_waterline_m)
            path = Path(cfg.robot_cfg.spawn.usd_path)
            vehicle = STUBS["env"].journal_ph.ASSETS_V2_DIR
            self.assertEqual(path.parent.resolve(), vehicle.resolve())
            M.journal_config.check_journal_cfg(cfg)
            STUBS["templates"].journal_validate(cfg)

    def test_every_suite_admits_on_both_journal_cfgs(self):
        resolved = M.journal_config.resolve_journal_conditions(presets=list(ALL_ON), seed=F.SEED)
        for vessel in VESSELS:
            cfg = F.journal_cfg(vessel)
            report = M.journal_config.admit_journal_conditions(cfg, resolved)
            self.assertEqual(report["contract"], "journal_kin46_ray36")


class JournalEnv(unittest.TestCase):
    def test_isaac_call_order_and_inner_substep_order(self):
        cfg = F.journal_cfg("blueboat_v2", tokens=ALL_ON)
        env = F.make_env(cfg, trace=True)
        env.call_log = []
        inner = []
        record_calls(env._hydro_v2, ("reset", "step_from_root_link"), inner, "runtime")
        record_calls(env._conditions, ("reset_runtime", "configure_runtime", "physics_inputs", "episode_stamp",
                                       "estimate_position", "estimate_velocity", "estimate_rays"), inner, "conditions")
        record_calls(env._d7_bridge, ("reset_rows", "step", "finish"), inner, "bridge")
        original = env._journal_observe_physics

        def observe(*args):
            inner.append("family.observe_physics")
            return original(*args)
        env._journal_observe_physics = observe
        env.reset()
        self.assertEqual(inner, ["runtime.reset", "conditions.reset_runtime", "conditions.configure_runtime",
                                 "bridge.reset_rows", "bridge.step", "conditions.estimate_position",
                                 "conditions.estimate_velocity", "conditions.estimate_rays"])
        inner.clear()
        env.step(F.actions(0, 2))
        substep = ["family.observe_physics", "bridge.step", "conditions.physics_inputs", "runtime.step_from_root_link"]
        self.assertEqual(inner[:8], substep * 2)
        self.assertEqual(inner[8:], ["bridge.step", "conditions.estimate_position", "conditions.estimate_velocity",
                                     "conditions.estimate_rays"])
        self.assertEqual(env.call_log, ["reset_idx", "get_observations", "pre_physics_step", "apply_action",
                                        "apply_action", "get_dones", "get_rewards", "get_observations"])
        self.assertEqual(env.journal_trace, ["setup", "reset", "estimates", "pre_physics", "substep", "substep",
                                             "pre_dones", "dones", "estimates"])
        # a finishing step: latch (stamp + finish) for every done row BEFORE the reset of that row
        rearm_deadline(env, 2.0 / 60.0)
        inner.clear()
        env.step(F.actions(1, 2))
        first_reset = inner.index("runtime.reset")
        self.assertEqual(inner[9:first_reset], ["conditions.episode_stamp", "bridge.finish"] * 2)
        self.assertEqual(env.journal_trace[-6:], ["substep", "substep", "pre_dones", "dones", "reset", "estimates"])

    def test_storage_width_split_masks_and_d7_publication(self):
        # G7: 3 envs; BlueBoat keys (0,0),(1,0) are not admitted, (2,0) is admitted (under the G7 window
        # rule (1,1) is admitted too, so a 2-env grid never mixed the two kinds in one episode)
        cfg = F.journal_cfg("blueboat_v2", tokens=("dynamic=D2",), num_envs=3)
        env = F.make_env(cfg)
        env.reset()
        while len(admitted_rows(env)) in (0, env.num_envs):
            env.reset()
        base = cfg.obstacle_count
        self.assertEqual(tuple(env.obstacle_centers.shape), (env.num_envs, base + 1, 2))
        self.assertEqual(env._jph_d7_slots, (base,))
        run(env, 3)
        adm = admitted_rows(env)
        for i in range(env.num_envs):   # one admitted intruder per admitted row at D2; none in a not-admitted row
            # G23c L2 (rule S2): physical once revealed
            self.assertEqual(bool(env._d7_physical[i, base]), i in adm and _d7_shown(env, i))
            self.assertEqual(len(env._journal_d7_samples[i]), 1 if i in adm else 0)
        self.assertFalse(bool(env.obstacle_active[:, base].any()))      # D7 never writes obstacle_active (B10)
        self.assertFalse(bool(env._d7_physical[:, :base].any()))
        counts = env.obstacle_active[:, :base].sum(dim=-1)
        self.assertEqual(counts.tolist(), [12] * env.num_envs)
        r = adm[0]
        sample = env._journal_d7_samples[r][0]
        self.assertAlmostEqual(float(env._jph_centers[r, base, 0]), sample.x)
        self.assertAlmostEqual(float(env.obstacle_centers[r, base, 0]),
                               sample.x + float(env.scene.env_origins[r, 0]), places=3)
        # rays see a D7 object only when visible; contact uses physical presence
        env._d7_visible[:] = False
        hidden = env._journal_ray_ranges()
        env._jph_centers[0, base] = torch.tensor([3.0, 0.0], dtype=torch.float64) + env._jph_prev_pose[0, :2]
        env._jph_radii[0, base] = 1.0
        self.assertTrue(torch.equal(env._journal_ray_ranges(), hidden))
        env._d7_visible[0, base] = True
        self.assertFalse(torch.equal(env._journal_ray_ranges(), hidden))

    def test_legacy_sampler_is_unreachable_from_the_journal_branch(self):
        module = STUBS["env"]
        original = module.sample_layout

        def refuse(*args, **kwargs):
            raise AssertionError("legacy path_hazard_geometry.sample_layout reached (C1 finding 32)")
        module.sample_layout = refuse
        try:
            env = F.make_env(F.journal_cfg("blueboat_v2"))
            env.reset()
            run(env, 2)
        finally:
            module.sample_layout = original
        self.assertEqual(env._jph_layouts[0].task, "path_hazard")

    def test_zero_dose_identity(self):
        runs = []
        for tokens in ((), F.ALL_D0):
            env = F.make_env(F.journal_cfg("blueboat_v2", tokens=tokens))
            obs = [env.reset()[0]["policy"]]
            for k in range(12):
                obs.append(env.step(F.actions(k, 2))[0]["policy"])
            runs.append((env, obs))
        (a, oa), (b, ob) = runs
        self.assertEqual(len(a.robot.applied), len(b.robot.applied))
        for (fa, ta), (fb, tb) in zip(a.robot.applied, b.robot.applied):
            self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
        for x, y in zip(oa, ob):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(a.robot.data.root_link_pose_w, b.robot.data.root_link_pose_w))
        self.assertTrue(any(bool(torch.count_nonzero(f)) for f, _ in a.robot.applied))

    def test_all_suites_on_three_control_steps_stamp_and_pair(self):
        records = {}
        for vessel in VESSELS:
            for waves in ("waves=D1", "waves=D2"):
                tokens = tuple(t for t in ALL_ON if not t.startswith("waves=")) + (waves,)
                env = F.make_env(F.journal_cfg(vessel, tokens=tokens))
                env.reset()
                adm = admitted_rows(env)                  # keys of episode 0 (the rows auto-reset at the end)
                rearm_deadline(env, 3.0 / 60.0)
                run(env, 3)
                self.assertEqual(env.termination_reason.tolist(), [3, 3])
                self.assertTrue(torch.equal(env.extras["termination_reason"], env.termination_reason))
                header = env.journal_certificate_header(treatment="waves")
                self.assertEqual(header["conditions"]["schema"], "usvbench.conditions/3")
                for i in range(2):
                    record = env.journal_episode_record(i)
                    stamp = record["conditions"]
                    self.assertEqual(stamp["schema"], "usvbench.conditions/3")
                    for suite in ("waves", "current", "wind", "actuator", "perception"):
                        self.assertTrue(stamp["parameters"][suite]["enabled"], suite)
                    self.assertEqual(stamp["runtime"]["runtime_ticks"], 6)
                    self.assertEqual(stamp["runtime"]["estimates"]["counts"], {"pos": 3, "vel": 3, "ray": 3})
                    M.dynamic.read_stamp(record["dynamic_hazards"])
                    payload = record["dynamic_hazards"]["payload"]
                    if i in adm:
                        self.assertEqual(payload["assignment"]["dose"], 2)
                        self.assertNotIn("exposure", payload)
                    else:        # D16: not admitted -> run at D0, stamped not_admitted, suite dose kept
                        self.assertEqual(payload["exposure"], "not_admitted")
                        self.assertEqual((payload["assignment"]["dose"], payload["runtime"]["suite_dose"]), (0, 2))
                    self.assertEqual(set(record["scenario_hashes"]), set(env.journal_pairing_inventory))
                    for name in ("layout", "route", "spawn_pose", "d7_assignment"):
                        self.assertIn(name, record["scenario_hashes"])
                    done = env.journal_completed[i]
                    self.assertEqual(done["contract"]["termination_reason"], 3)
                    self.assertEqual(done["record"]["scenario_hashes"], record["scenario_hashes"])
                    self.assertEqual(env.episode_contract_result[i].termination_reason, 3)
                    records[(vessel, waves, i)] = record
        for vessel in VESSELS:
            for i in range(2):
                a, b = records[(vessel, "waves=D1", i)], records[(vessel, "waves=D2", i)]
                pair = M.pairing.validate_pair(a, b, "waves")
                self.assertEqual(pair["changed_primitives"], ["waves"])
                for name in ("layout", "route", "spawn_pose", "d7_assignment"):
                    self.assertEqual(a["scenario_hashes"][name], b["scenario_hashes"][name])

    def test_termination_codes_are_published_by_the_contract(self):
        cases = {}
        for label in ("capsize", "collision", "out_of_bounds"):
            env = F.make_env(F.journal_cfg("blueboat_v2"))
            env.reset()
            pose = env.robot.data.root_link_pose_w
            origin = env.scene.env_origins[0]
            if label == "capsize":
                pose[0, 3:7] = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
            elif label == "collision":
                center = env._jph_centers[0, 0]
                pose[0, 0] = float(center[0]) + float(origin[0])
                pose[0, 1] = float(center[1]) + float(origin[1])
            else:
                pose[0, 0] = float(env._jph_bounds[0, 1]) + 5.0 + float(origin[0])
            env.step(F.actions(0, 2))
            cases[label] = env
            self.assertEqual(int(env.extras["termination_reason"][1]), 0)
        self.assertEqual(int(cases["capsize"].termination_reason[0]), 6)
        self.assertEqual(int(cases["collision"].termination_reason[0]), 2)
        self.assertEqual(int(cases["out_of_bounds"].termination_reason[0]), 4)
        record = cases["capsize"].journal_completed[0]
        self.assertEqual(record["contract"]["termination_reason_name"], "capsize")
        self.assertEqual(record["record"]["dynamic_hazards"]["payload"]["runtime"]["termination_reason"], "capsize")
        self.assertTrue(bool(cases["collision"].reset_terminated[0]))

    def test_gate_bonus_is_the_threading_passage_payment(self):
        env = F.make_env(F.journal_cfg("blueboat_v2"))
        env.reset()
        n = env.num_envs
        origin = env.scene.env_origins[:, :2].to(torch.float64)
        gate = env._jph_waypoints[:, 0]
        heading = env._jph_portal_heading[:, 0]
        u = torch.stack((torch.cos(heading), torch.sin(heading)), -1)
        radius = env._jph_gate_radius
        state = {"s": -3.2}                   # hull front outside the entry plane x = -r
        z = env.robot.data.root_link_pose_w[:, 2].clone()
        quat = torch.stack((torch.cos(heading / 2), torch.zeros(n, dtype=torch.float64),
                            torch.zeros(n, dtype=torch.float64), torch.sin(heading / 2)), -1).to(torch.float32)

        def place(advance):
            state["s"] += advance
            xy = gate + state["s"] * u
            data = env.robot.data
            data.root_link_pose_w[:, :2] = (origin + xy).to(torch.float32)
            data.root_link_pose_w[:, 2] = z
            data.root_link_pose_w[:, 3:7] = quat
            data.root_com_vel_w[:] = 0.0
            data.root_com_vel_w[:, :2] = (u * (advance / env.physics_dt)).to(torch.float32)
            return xy

        env.obstacle_active[:] = False        # this test scripts a straight transit; statics are not its subject
        xy = place(0.0)
        env._jph_prev_pose[:] = torch.cat((xy, heading.unsqueeze(-1)), -1)
        env.robot.integrate = lambda dt: place(0.02)
        gate_step = bonus_step = None
        rect = env._jph_rect
        for k in range(200):
            env.step(torch.zeros((n, 2)))
            self.assertEqual(env.termination_reason.tolist(), [0] * n, k)
            if gate_step is None and int(env.gates_passed[0]) == 1:
                gate_step = k
            if bool(env._gates_passed_this_step[0]):
                self.assertIsNone(bonus_step)
                bonus_step = k
                hx, _ = F.modules().journal.hull_half_extents(rect, torch.zeros(1, dtype=torch.float64))
                self.assertGreaterEqual(state["s"] - float(hx[0]), radius)          # full hull past the exit
                self.assertLess(state["s"] - 0.04 - float(hx[0]), radius)
            if state["s"] > radius + 1.5:
                break
        self.assertIsNotNone(gate_step)
        self.assertIsNotNone(bonus_step)
        self.assertLess(gate_step, bonus_step)          # COM entered the gate disk long before the payment
        summary = env._jph_threading.summary(0)
        self.assertTrue(summary[0]["paid"])
        self.assertFalse(summary[1]["paid"] or summary[1]["failed"])
        self.assertEqual(int(env._jph_threading.current[0]), 1)

    def test_reset_row_isolation(self):
        envs = [F.make_env(F.journal_cfg("blueboat_v2", tokens=ALL_ON)) for _ in range(2)]
        for env in envs:
            env.reset()
            run(env, 5)
        # env B: row 1 ends at the next step (timeout) and is auto-reset alone inside step (Isaac flow)
        b = envs[1]
        routes = torch.cat((torch.zeros((1, 1, 2), dtype=torch.float64), b._jph_waypoints[1:2]), dim=1)
        next_sample = float(b.episode_length_buf[1] + 1) * b.control_step_s
        b._jph_oracle.reset([1], routes, b._jph_gate_radius, torch.tensor([next_sample], dtype=torch.float64),
                            routes[:, 0])
        outs = []
        for env in envs:
            obs = None
            for k in range(5, 10):
                obs = env.step(F.actions(k, 2))[0]["policy"]
            outs.append(obs)
        a = envs[0]
        self.assertEqual(int(b._scenario.episode_indices()[1]), 1)
        self.assertEqual(int(a._scenario.episode_indices()[1]), 0)
        self.assertEqual(b.journal_completed[1]["contract"]["termination_reason"], 3)
        self.assertEqual(b.journal_aborted_dynamic_hazards, [])
        self.assertTrue(torch.equal(outs[0][0], outs[1][0]))
        self.assertFalse(torch.equal(outs[0][1], outs[1][1]))
        self.assertTrue(torch.equal(a.robot.data.root_link_pose_w[0], b.robot.data.root_link_pose_w[0]))
        self.assertTrue(torch.equal(a.robot.data.root_com_vel_w[0], b.robot.data.root_com_vel_w[0]))
        for name in ("full_len", "xte_int", "min_clear", "thrust_sq", "prev_xy", "gate_times"):
            ta, tb = getattr(a._jph_oracle, name)[0], getattr(b._jph_oracle, name)[0]
            self.assertTrue(torch.equal(ta, tb) or bool(torch.isnan(ta).all() and torch.isnan(tb).all()), name)
        self.assertEqual(int(a._d7_tick[0]), int(b._d7_tick[0]))
        self.assertEqual(int(a._hydro_v2.ticks[0]), int(b._hydro_v2.ticks[0]))
        self.assertEqual(a._conditions.estimate_counts(0), b._conditions.estimate_counts(0))
        self.assertEqual(int(a._hydro_v2.ticks[1]), 20)
        self.assertEqual(int(b._hydro_v2.ticks[1]), 8)
        # a manual subset reset outside the step flow stamps the open D7 row as aborted (never a record)
        b._reset_idx(torch.tensor([0]))
        self.assertEqual(len(b.journal_aborted_dynamic_hazards), 1)

    def test_com_reference_and_estimate_driven_observation(self):
        cfg = F.journal_cfg("blueboat_v2", tokens=("perception=D4",))
        env = F.make_env(cfg)
        data_cls = type(env.robot.data)
        original = data_cls.root_com_pos_w

        def refuse(self):
            raise AssertionError("root_com_pos_w read in the journal branch (E5 C5)")
        data_cls.root_com_pos_w = property(refuse)
        try:
            obs = env.reset()[0]["policy"]
            spawn_z = env.robot.data.root_link_pose_w[:, 2] - env.scene.env_origins[:, 2]
            self.assertTrue(torch.allclose(spawn_z.double(), torch.full((2,), env._jph_equilibrium_z,
                                                                       dtype=torch.float64), atol=1e-6))
            run(env, 4)
            obs = env._get_observations  # bound journal hook
            est = env.journal_estimate
            truth = env._jph_state()[4][:, :2]
            self.assertFalse(torch.equal(torch.as_tensor(est.position, dtype=torch.float64), truth))
            position = torch.as_tensor(est.position, dtype=torch.float64)
            target = env._jph_waypoints[torch.arange(2), env.gates_passed.clamp(max=3)]
            distance = torch.linalg.vector_norm(target - position, dim=-1) / env._jph_distance_norm
            self.assertTrue(torch.allclose(env.obs_buf["policy"][:, 2].double(), distance, atol=1e-6))
            self.assertTrue(callable(obs))
        finally:
            data_cls.root_com_pos_w = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
