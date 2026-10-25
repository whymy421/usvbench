"""F1: HazardNav journal wiring through the REAL env methods on a stubbed Isaac Lab (CPU only).

Every test constructs the REAL HazardNavEnv (tasks/hazard_nav/hazard_nav_env.py)
from a REAL E5 template cfg (journal_runtime_cfg_v2) on journal_test_support's
stub DirectRLEnv, driving the REAL F0 mixin, HydroV2Runtime, schema-3
condition layer and E4 HazardBridge. Covers (F1 brief item 8):
  legacy byte-identity of the legacy methods vs F-base (pinned digests);
  zero-dose identity; substep call order; recorded-trace contract parity
  (scalar ReachEpisode replay); reset-row isolation; ALL SUITES ON at once
  (3 control steps + completion + stamps) on both vessels; pairing hashes;
  D7 publication (reserved slot, split masks) and the moving-basin native
  motion (legacy movers never called, speeds = mu * envelope top speed);
  COM spawn and no root_com_pos_w read after a reset; fail-closed pool checks.

Run: python -B tasks/hazard_nav/test_journal_env.py
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

import journal_test_support as S  # noqa: E402

RESTORE = S.install_isaaclab_stubs()
FAM = S.load_family()
J, C, D = FAM.journal, FAM.contracts, FAM.dynamic
ENV_FILE = S.REPO / "tasks" / "hazard_nav" / "hazard_nav_env.py"
ALL_ON = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "perception.bundle=D2",
          "dynamic=D2")
ALL_D0 = ("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0", "perception.bundle=D0",
          "dynamic=D0")

# SHA-256 (LF-normalised source segment) of every HazardNavEnv method except
# __init__ in F-base (D:/claude code/codex_tasks/20260930_fbase/work/repo,
# MANIFEST sha 1b0...; computed with ast.get_source_segment). D15: legacy
# sources stay byte-identical; the journal hooks are separate methods.
FBASE_METHOD_SHA256 = {
    "current_level": "c1c772469577bb54661fe44c66f60dfc3e3e26e0e2dae6a193bee9de9d7b947c",
    "_setup_scene": "a9401936e37455c9a21babcddde330f72ad4428b47814dc01b2124aa2f2ca78e",
    "_attach_official_blueboat_visual": "39a90a9f671439effdf82528afaae8d9cd07e475690513f46acf9914845fbb16",
    "_create_static_water_mesh": "b39903d0107b1aca29cee9a9a1576be00c57ab918511afa5d2b14f8903f55d7c",
    "_create_render_only_hazard_markers": "29b1a16b7245c741464842a483a71736e0e8624235026b773be8b87f07c0b379",
    "_update_render_only_hazard_markers": "9210825200c3a605b8e7fb8c1175ccfc1ee7646a475e239dae794d2fda598639",
    "_root_quat": "d386a884193b8376bdc65ed834f5a62ce6bac164b866c3dadb71f28487224d8b",
    "_compute_buoyancy_forces": "caa855c8c2eac66ac1b49cc5bc5d43e7a9457193ec028f858e9946faf6ed68d8",
    "_pre_physics_step": "f804a7a4d3dfc7817197b5a1687ee7fc089fce30714b8e3fa17effd5cf0ebfc9",
    "_apply_action": "b865a92823e1e20f73789c295d44edc49c4db7bda4470273221e05ee5a3e87cc",
    "_get_or_add_translate_op": "dcc6a35ccb815c02298da1089f596a864ab65b9a2ba82af64f0b31192f2e28d6",
    "_com_xy": "3f963e1d604f2e719cbf23a7cff330cf843639e1826914fb93607a2d7d6bb724",
    "_horizontal_distance": "4a380aab028b939971bf037c606fe2d3617ee7a131e3d484b0e99ca3588e9119",
    "_forward_2d": "5e57e3ebd6109c47a2a8be423995eb098e2f303bfd473cfdb1cc4cee3f83e722",
    "_planar_velocity_2d": "1912e83536d41c72f7743192c386d57a516473a55fcb8f756af17a383ad8fb1f",
    "_body_motion_measurement": "90b47faf47db281bf650b734df0e03751e5e71023d9da8f41b3d5bf417746af2",
    "_body_motion_observation": "cf0c5971cd5d399874261bcc954483f7994c03410808f54d4de9e7e1434dd43f",
    "_clearance": "79f179e6fdf1c655268bdf9cdb36f6b144e488fa132035a83e3249a74e24a39d",
    "_ray_ranges_m": "4741f3f88b450fb334f0c30d22a2149ee831564863073b7e126d6590955a83c4",
    "_get_observations": "c613e8e050f97c4403c98fed575c56f8631c014780fde4135d74a54e6529e765",
    "_potential": "8eb11e0c8fb11252029bd8259445a4cb392d4af1043071954711b4092e93087e",
    "_geodesic_progress_distance": "4a1eccac3e329ac86521a16196cb86b3a84b72028578f00a1ed2dca053682756",
    "_geodesic_waypoint": "78d8e55078d9c720a7bbb5fb8bdbad3cab4ea23d9e3d2d04740d2757c79da7c1",
    "_get_rewards": "2ba780ac55b1785a77f61e50e5199991f5c36a5f6ad43e4545995a68408f41d1",
    "_reset_appearance": "1c6c915996d02c65be331b23ab79d0473e14367656965a3130cef3f9f3d2d5c6",
    "_update_moving_obstacles": "a759eb2371a638e3348a22e9f2441ac5939e5f1b814166e68a369150502ab14b",
    "_refresh_moving_markers": "7af7fef9fa078e13b7576b9e083537a747f6c91cacc442c495e64560de1d860c",
    "_maybe_appear_obstacles": "7258ed9e6c29d97f235132ff3c44cc9ac9a7cac7bd01918b91e50cd58f4b9320",
    "_get_dones": "3e43bd7f90a611392afb9cfdacb3b97d4688a6fb1521b378fe1e4d84dbbf6147",
    "_reset_idx": "39654c51b7b529b1a7b58e036faa69faaf0e0a599e84f06454f439e3d485c6f0",
    "_suite_s_layout_for": "8320d43ffe2d12be769ea4952a1a10839f79ad3599f6f222299627af2fb3ec6c",
}
FBASE_ENV = Path(r"D:\claude code\codex_tasks\20260930_fbase\work\repo\tasks\hazard_nav\hazard_nav_env.py")


def method_digests(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HazardNavEnv")
    return {n.name: hashlib.sha256(ast.get_source_segment(source, n).replace("\r\n", "\n").encode()).hexdigest()
            for n in cls.body if isinstance(n, ast.FunctionDef)}


def env_for(task, vessel="blueboat_v2", tokens=(), num_envs=2, episodes=(0, 1, 2)):
    pool = S.admitted_pool(FAM, task, vessel, num_envs=num_envs, episodes=episodes)
    cfg = S.journal_cfg(FAM, task, vessel, tokens=tokens, num_envs=num_envs, pool=pool)
    return S.make_env(FAM, cfg)


def record_calls(obj, names, log, prefix):
    for name in names:
        original = getattr(obj, name)

        def wrapper(*args, _original=original, _name=name, **kwargs):
            log.append(prefix + "." + _name)
            return _original(*args, **kwargs)
        setattr(obj, name, wrapper)


def force_timeout(env, steps):
    """Shorten every row's contract deadline so the episode completes after `steps` control steps."""
    env._journal_twin.deadline[:] = steps * FAM.plans.CONTROL_DT_S


class LegacyIdentity(unittest.TestCase):
    def test_legacy_method_sources_unchanged(self):
        now = method_digests(ENV_FILE)
        for name, digest in FBASE_METHOD_SHA256.items():
            self.assertEqual(now.get(name), digest, name)
        if FBASE_ENV.is_file():
            base = method_digests(FBASE_ENV)
            base.pop("__init__")
            self.assertEqual({k: now[k] for k in base}, base)

    def test_journal_hooks_are_bound_per_instance_only(self):
        env = env_for("reach")
        cls = FAM.env.HazardNavEnv
        for name in cls._JOURNAL_BOUND:
            self.assertEqual(getattr(env, name).__func__, getattr(cls, "_journal" + name))
            self.assertIn(name, vars(env))
        self.assertNotIn("_journal", cls._get_dones.__code__.co_names)


class CallOrder(unittest.TestCase):
    def test_isaac_order_and_mixin_inner_order(self):
        env = env_for("scatter", tokens=ALL_ON)
        env.call_log, env.journal_trace = [], []
        inner = []
        record_calls(env._hydro_v2, ("reset", "step_from_root_link"), inner, "runtime")
        record_calls(env._conditions, ("reset_runtime", "configure_runtime", "physics_inputs", "episode_stamp",
                                       "estimate_position", "estimate_velocity", "estimate_rays"), inner,
                     "conditions")
        record_calls(env._d7_bridge, ("reset_rows", "step", "finish"), inner, "bridge")
        record_calls(env._journal_twin, ("reset", "observe_physics", "step"), inner, "twin")
        env.reset()
        self.assertEqual(inner, ["twin.reset", "runtime.reset", "conditions.reset_runtime",
                                 "conditions.configure_runtime", "bridge.reset_rows", "bridge.step",
                                 "conditions.estimate_position", "conditions.estimate_velocity",
                                 "conditions.estimate_rays"])
        inner.clear()
        env.step(S.zero_action(env, (0.5, 0.1)))
        substep = ["twin.observe_physics", "bridge.step", "conditions.physics_inputs",
                   "runtime.step_from_root_link"]
        self.assertEqual(inner[:8], substep * 2)
        self.assertEqual(inner[8:10], ["bridge.step", "twin.step"])
        self.assertEqual(inner[10:], ["conditions.estimate_position", "conditions.estimate_velocity",
                                      "conditions.estimate_rays"])
        self.assertEqual(env.call_log, ["reset_idx", "get_observations", "pre_physics_step", "apply_action",
                                        "apply_action", "get_dones", "get_rewards", "get_observations"])
        self.assertEqual(env.journal_trace, ["reset", "estimates", "pre_physics", "substep", "substep",
                                             "pre_dones", "dones", "estimates"])
        inner.clear()
        force_timeout(env, 2)
        env.step(S.zero_action(env))
        dones = inner[inner.index("twin.step"):]
        first_reset = dones.index("twin.reset")
        self.assertEqual(dones[1:first_reset], ["conditions.episode_stamp", "bridge.finish"] * 2)
        self.assertEqual(dones[first_reset:first_reset + 6],
                         ["twin.reset", "runtime.reset", "conditions.reset_runtime",
                          "conditions.configure_runtime", "bridge.reset_rows", "bridge.step"])

    def test_legacy_movers_never_run_in_the_journal_branch(self):
        env = env_for("reach")

        def forbidden(*args, **kwargs):
            raise AssertionError("legacy mover called in the journal branch")
        env._update_moving_obstacles = forbidden
        env._maybe_appear_obstacles = forbidden
        env.reset()
        for _ in range(3):
            env.step(S.zero_action(env, (0.3, 0.0)))


class ZeroDose(unittest.TestCase):
    def test_all_suites_at_d0_bit_identical_to_no_suites(self):
        def run(tokens):
            env = env_for("scatter", tokens=tokens)
            obs, _ = env.reset()
            out = [obs["policy"].clone()]
            for k in range(6):
                o, r, te, tr, _ = env.step(S.zero_action(env, (0.6, 0.2 * (-1) ** k)))
                out += [o["policy"].clone(), r.clone(), env.robot.data.root_link_pose_w.clone(),
                        env.robot.applied[0].clone()]
            return out
        a, b = run(()), run(ALL_D0)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))


class RecordedParity(unittest.TestCase):
    def test_env_traces_replay_bit_for_bit_through_the_scalar_oracle(self):
        for task, tokens in (("reach", ALL_ON), ("scatter", ("waves=D3", "dynamic=D2"))):
            env = env_for(task, tokens=tokens)
            env.journal_contract_trace = []
            env.reset()
            scenarios = list(env._journal_scenarios)
            limits = C.ValidityLimits(**env.cfg.journal_plan.validity_limits)
            nominal = env._journal_record.com_height_above_waterline_m
            oracles = []
            for sc in scenarios:
                o = C.ReachEpisode(sc.goal_xy, sc.goal_radius_m, 3 * FAM.plans.CONTROL_DT_S, limits)
                o.reset(sc.start_xy)
                oracles.append(o)
            force_timeout(env, 3)
            decisions = []
            for k in range(3):
                env.step(S.zero_action(env, (0.9, 0.3)))
                decisions.append([int(c) for c in env.termination_reason.tolist()])
            completed = [env.journal_completed_records[i]["contract"] for i in range(env.num_envs)]
            step = 0
            for entry in env.journal_contract_trace:
                for i, oracle in enumerate(oracles):
                    state = C.RigidState(entry["com"][i].tolist(), entry["quat"][i].tolist(),
                                         entry["lin"][i].tolist(), entry["ang"][i].tolist(),
                                         water_level_m=float(entry["water"][i]), nominal_com_height_m=nominal)
                    clearance = float(entry["clearance"][i]) if bool(entry["present"][i]) else None
                    if entry["kind"] == "observe":
                        oracle.observe_physics(state, clearance_m=clearance)
                    else:
                        d = oracle.step(float(entry["time"][i]), state, collision=bool(entry["collision"][i]),
                                        out_of_bounds=bool(entry["out_of_bounds"][i]), clearance_m=clearance)
                        self.assertEqual(int(d.reason) if d.done else 0, decisions[step][i], (task, step, i))
                if entry["kind"] == "step":
                    step += 1
            for i, oracle in enumerate(oracles):
                want = oracle.last_completed.evaluator_fields()
                got = completed[i]
                self.assertEqual(set(want), set(got))
                for key in want:
                    if isinstance(want[key], float):
                        self.assertTrue(want[key] == got[key] or (math.isnan(want[key]) and math.isnan(got[key])),
                                        (task, key, want[key], got[key]))
                    else:
                        self.assertEqual(want[key], got[key], (task, key))


class ResetIsolation(unittest.TestCase):
    def test_resetting_one_row_leaves_the_other_untouched(self):
        env = env_for("scatter", tokens=("dynamic=D2", "waves=D2"))
        env.reset()
        for _ in range(3):
            env.step(S.zero_action(env, (0.5, 0.1)))
        keep = 1
        snapshot = {
            "centers": env.obstacle_centers[keep].clone(), "radii": env.obstacle_radii[keep].clone(),
            "active": env.obstacle_active[keep].clone(), "phys": env._d7_physical[keep].clone(),
            "target": env.target_pos[keep].clone(), "tick": int(env._d7_tick[keep]),
            "ticks": int(env._hydro_v2.ticks[keep]), "path": float(env._journal_twin.full_path[keep]),
            "prev": env._journal_twin.prev_xy[keep].clone(), "pose": env.robot.data.root_link_pose_w[keep].clone(),
            "scenario": env._journal_scenarios[keep], "episode_len": int(env.episode_length_buf[keep]),
        }
        old = env._journal_scenarios[0]
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(len(env.journal_aborted_dynamic_hazards), 1)       # manual reset: aborted, no record
        self.assertIsNone(env.journal_completed_records[0])
        self.assertNotEqual(env._journal_scenarios[0].episode, old.episode)
        self.assertTrue(torch.equal(env.obstacle_centers[keep], snapshot["centers"]))
        self.assertTrue(torch.equal(env.obstacle_radii[keep], snapshot["radii"]))
        self.assertTrue(torch.equal(env.obstacle_active[keep], snapshot["active"]))
        self.assertTrue(torch.equal(env._d7_physical[keep], snapshot["phys"]))
        self.assertTrue(torch.equal(env.target_pos[keep], snapshot["target"]))
        self.assertTrue(torch.equal(env.robot.data.root_link_pose_w[keep], snapshot["pose"]))
        self.assertTrue(torch.equal(env._journal_twin.prev_xy[keep], snapshot["prev"]))
        self.assertEqual(int(env._d7_tick[keep]), snapshot["tick"])
        self.assertEqual(int(env._hydro_v2.ticks[keep]), snapshot["ticks"])
        self.assertEqual(float(env._journal_twin.full_path[keep]), snapshot["path"])
        self.assertIs(env._journal_scenarios[keep], snapshot["scenario"])
        self.assertEqual(int(env.episode_length_buf[keep]), snapshot["episode_len"])


class AllSuitesOn(unittest.TestCase):
    def check_vessel(self, vessel, task):
        env = env_for(task, vessel, tokens=ALL_ON)
        env.reset()
        for _ in range(3):
            obs, rew, te, tr, _ = env.step(S.zero_action(env, (0.7, -0.2)))
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            self.assertTrue(torch.isfinite(rew).all())
        self.assertEqual(obs["policy"].shape, (env.num_envs, 42))
        force_timeout(env, 4)
        env.step(S.zero_action(env))
        for i in range(env.num_envs):
            rec = env.journal_completed_records[i]
            self.assertIsNotNone(rec, (vessel, task, i))
            self.assertEqual(rec["termination_name"], "timeout")
            stamp = rec["conditions"]
            self.assertEqual(stamp["schema"], "usvbench.conditions/3")
            for suite in ("waves", "wind", "current", "actuator", "perception"):
                self.assertTrue(stamp["parameters"][suite]["enabled"], suite)
            D.read_stamp(rec["dynamic_hazards"])
            self.assertEqual(rec["dynamic_hazards"]["payload"]["assignment"]["dose"], 2)
            hashes = rec["scenario_hashes"]
            want = set(FAM.pairing.CONDITION_PRIMITIVES) | {"d7_assignment", "layout", "rotation", "spawn_pose"}
            if task == "moving_basin":
                want.add(J.BASIN_PRIMITIVE)
            self.assertEqual(set(hashes), want)
            self.assertTrue(all(FAM.pairing.is_hash(v) for v in hashes.values()))
            self.assertEqual(int(env.episode_termination_reason[i]), 3)
        header = env.journal_certificate_header(treatment="dynamic")
        self.assertIn("d7_assignment", header["pairing"]["changed_primitives"])
        self.assertIn("layout", header["pairing"]["unchanged_primitives"])
        self.assertEqual(header["dynamic_hazards_header"]["pool_digest"], env._d7_pool.pool_digest)

    def test_blueboat_all_suites_reach(self):
        self.check_vessel("blueboat_v2", "reach")

    def test_wamv_all_suites_scatter(self):
        self.check_vessel("wamv_vrx_v2_opt1", "scatter")

    def test_paired_records_differ_only_in_the_treatment(self):
        records = {}
        for tokens in (("dynamic=D0", "waves=D2"), ("dynamic=D2", "waves=D2")):
            env = env_for("reach", tokens=tokens)
            env.reset()
            force_timeout(env, 2)
            env.step(S.zero_action(env, (0.4, 0.0)))
            env.step(S.zero_action(env, (0.4, 0.0)))
            records[tokens[0]] = env.journal_completed_records[0]
        a, b = records["dynamic=D0"], records["dynamic=D2"]
        summary = FAM.pairing.validate_pair(a, b, "dynamic")
        self.assertIn("d7_assignment", summary["changed_primitives"] if isinstance(summary, dict) else ())
        diff = {k for k in a["scenario_hashes"] if a["scenario_hashes"][k] != b["scenario_hashes"][k]}
        self.assertEqual(diff, {"d7_assignment"})


class D7Publication(unittest.TestCase):
    def test_reserved_slot_and_split_masks(self):
        env = env_for("reach", tokens=("dynamic=D2",))
        env.reset()
        slot = J.storage_width("reach") - 1
        self.assertEqual(slot, FAM.plans.STATIC_CAPACITY["reach"])
        self.assertFalse(bool(env.obstacle_active.any()))       # D7 never writes obstacle_active (B04)
        # G23c L2 (rule S2): the intruder is physical exactly when the runtime rule has revealed it (hidden at reset
        # unless the own ship already starts inside the dose)
        revealed = [env._d7_bridge.adapter.episodes[i].release.released_at(0.0) for i in range(env.num_envs)]
        self.assertEqual(env._d7_physical[:, slot].tolist(), revealed)
        a = env._d7_pool.assignment(J.d7_key(env._journal_scenarios[0]), 2)
        intruder = a.candidate.request.encounter.intruders[0]
        local = env.obstacle_centers[0, slot] - env.scene.env_origins[0, :2]
        # the published intruder is the adapter's runtime object (G23b/G23c)
        x, y = env._d7_bridge.adapter.episodes[0].objects[0].position(0.0)
        self.assertAlmostEqual(float(local[0]), x, places=4)
        self.assertAlmostEqual(float(local[1]), y, places=4)
        self.assertAlmostEqual(float(env.obstacle_radii[0, slot]), intruder.radius_m, places=6)
        env.step(S.zero_action(env))
        vis_mask = env.obstacle_active | env._d7_visible
        rays = env._journal_rays(vis_mask)
        if not bool(env._d7_visible[0, slot]):
            self.assertTrue(torch.allclose(rays[0], torch.full_like(rays[0], env.cfg.ray_max_range_m)))

    def test_d0_has_no_intruder(self):
        env = env_for("reach", tokens=("dynamic=D0",))
        env.reset()
        self.assertFalse(bool(env._d7_physical.any()))
        self.assertFalse(bool(env._d7_visible.any()))


class MovingBasin(unittest.TestCase):
    def test_native_motion_through_the_bridge(self):
        env = env_for("moving_basin", tokens=("dynamic=D2",), num_envs=1, episodes=(0, 1))

        def forbidden(*args, **kwargs):
            raise AssertionError("legacy mover called in the journal basin")
        env._update_moving_obstacles = forbidden
        env._maybe_appear_obstacles = forbidden
        env.reset()
        sc = env._journal_scenarios[0]
        native = list(sc.native_slots)
        self.assertTrue(native)
        self.assertFalse(bool(env.obstacle_active[0, native].any()))
        self.assertTrue(bool(env._d7_physical[0, native].all()))
        self.assertTrue(bool(env.obstacle_active[0, list(sc.static_slots)].all()))
        a = env._d7_pool.assignment(J.d7_key(sc), 2)
        meta = dict(a.candidate.request.encounter.metadata)
        top = J.vessel_envelope("blueboat_v2").top_speed_mps
        self.assertEqual(meta["mu"], D.NATIVE_RATIOS[2])
        self.assertEqual(meta["row_speed_mps"], D.NATIVE_RATIOS[2] * top)
        before = env.obstacle_centers[0, native].clone()
        walls = env.obstacle_centers[0, list(sc.static_slots)].clone()
        for _ in range(5):
            env.step(S.zero_action(env, (0.3, 0.0)))
        moved = (env.obstacle_centers[0, native] - before).norm(dim=-1)
        self.assertGreater(float(moved.max()), 1.0e-3)
        self.assertTrue(torch.equal(env.obstacle_centers[0, list(sc.static_slots)], walls))
        self.assertIn(J.BASIN_PRIMITIVE, env.journal_pairing_inventory)


class SpawnAndFailClosed(unittest.TestCase):
    def test_com_spawn_at_equilibrium_and_no_root_com_read_after_reset(self):
        env = env_for("reach")
        data_cls = type(env.robot.data)
        original = data_cls.root_com_pos_w

        def guard(self):
            raise AssertionError("root_com_pos_w read by the journal path")
        data_cls.root_com_pos_w = property(guard)
        try:
            env.reset()
            z = env.robot.data.root_link_pose_w[:, 2]
            want = env.scene.env_origins[:, 2] + env._journal_water_z + env._journal_eq_z
            self.assertTrue(torch.allclose(z, want.to(z.dtype)))
            for i, sc in enumerate(env._journal_scenarios):
                local = env.robot.data.root_link_pose_w[i, :2] - env.scene.env_origins[i, :2]
                self.assertTrue(torch.allclose(local.double(), torch.tensor(sc.start_xy, dtype=torch.float64),
                                               atol=1e-5))
            for _ in range(2):
                env.step(S.zero_action(env, (0.5, 0.0)))
        finally:
            data_cls.root_com_pos_w = original

    def test_pool_for_another_seed_fails_closed(self):
        pool = S.admitted_pool(FAM, "reach", "blueboat_v2", num_envs=2, episodes=(0, 1), seed=7)
        cfg = S.journal_cfg(FAM, "reach", "blueboat_v2", pool=pool)
        env = S.make_env(FAM, cfg)
        with self.assertRaises(ValueError):
            env.reset()

    def test_episode_beyond_the_offered_grid_runs_at_d0(self):
        # CONTEXT_R4 D24 (round T): the next episode past the grid no longer fails closed; it runs as a
        # not-admitted key with the single reason beyond_offered_grid (full coverage: test_journal_d24.py)
        env = env_for("reach", episodes=(0,))
        env.reset()
        force_timeout(env, 1)
        env.step(S.zero_action(env))
        for i in range(env.num_envs):
            self.assertEqual(env._journal_scenarios[i].episode, 1)
            self.assertTrue(env._d7_bridge.is_not_admitted(i))
        force_timeout(env, 1)
        env.step(S.zero_action(env))
        for i in range(env.num_envs):
            payload = env.journal_completed_records[i]["dynamic_hazards"]["payload"]
            self.assertEqual(payload["not_admitted"]["reasons"], [[0, "beyond_offered_grid", 0]])

    def test_env_outside_the_pool_env_range_fails_closed(self):
        pool = S.admitted_pool(FAM, "reach", "blueboat_v2", num_envs=2, episodes=(0,))
        env = S.make_env(FAM, S.journal_cfg(FAM, "reach", "blueboat_v2", num_envs=3, pool=pool))
        with self.assertRaises(ValueError):
            env.reset()

    def test_pinned_pool_file_loads_through_the_cfg_path(self):
        evidence = S.REPO.parent / "evidence" / "d7_pools"
        if not evidence.is_dir():
            self.skipTest("evidence/d7_pools not beside this repository copy")
        cfg = S.journal_cfg(FAM, "reach", "blueboat_v2", tokens=("dynamic=D3",), num_envs=2)
        self.assertIsNone(getattr(cfg, "journal_d7_pool", None))
        cfg.journal_d7_pool_dir = str(evidence)
        env = S.make_env(FAM, cfg)
        pin = J.load_pool_manifest()["pools"][J.pool_file_name("reach", "blueboat_v2")]
        self.assertEqual(env._journal_pool_meta["file_sha256"], pin["sha256"])
        self.assertEqual(env.dynamic_hazard_header["pool_digest"], pin["pool_digest"])
        self.assertEqual(env.dynamic_hazard_header["keys_not_admitted"], pin["keys_not_admitted"])
        self.assertEqual(env.dynamic_hazard_header["pool_file_sha256"], pin["sha256"])
        self.assertEqual(env.dynamic_hazard_header["pool_manifest_sha256"],
                         J.file_sha256(J.POOL_DIR / "MANIFEST.json"))
        env.reset()
        env.step(S.zero_action(env, (0.5, 0.0)))
        self.assertEqual(env._d7_suite.dose, 3)

    def test_storage_width_and_missing_hooks_flag_rejected(self):
        pool = S.admitted_pool(FAM, "reach", "blueboat_v2")
        cfg = S.journal_cfg(FAM, "reach", "blueboat_v2", pool=pool)
        cfg.journal_observation_hooks_wired = False
        with self.assertRaises(ValueError):
            S.make_env(FAM, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
