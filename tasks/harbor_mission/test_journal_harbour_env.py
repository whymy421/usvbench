"""F6: the REAL HarborMissionEnv journal branch on a stubbed Isaac Lab (CPU).

Drives the real env class, the REAL F0 mixin, the REAL HydroV2Runtime, the
REAL schema-3 condition layer, the REAL E4 HazardBridge with an admitted
REAL-layout pool, and the REAL E5 cfg templates through the harbour parent
cfg (task F6 edits). Checks: legacy byte identity (methods vs F-base, legacy
cfg values vs F-base), zero-dose identity, substep call order, all suites on
at once (both vessels) with stamps, termination codes through the D9
publisher with the latch before reset, reset-row isolation, D7 publication
masks (B07), phase targets (frame = estimate frame, no D7 slot), pairing
hashes (D11), dock-phase scope and the skipped-stage record, pool integrity
(fail closed) and the recorded-trace parity of the env's tensor twin with
E5's scalar oracle. Round 4b (D16/D19): a key E4 rejects keeps its E5 layout
and runs a full episode at every dynamic level with exposure not_admitted,
pairs with the same key at D0 of every other suite, one admission pass (no
re-attempt), the explicit not-admitted list and the pinned manifest fail
closed. Round T (D24): an env whose task episode index reaches the offered
grid's episodes per env (and beyond, unbounded) runs full episodes at every
dynamic level as a not-admitted key with the single reason
"beyond_offered_grid" on the key's E5 layout, its D11 d7_assignment hash is
dose independent, keys outside the env range or of another seed/group stay
preflight errors, and the header publishes the offered grid.
Run: python -B test_journal_harbour_env.py
"""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

def _d7_shown(env, row):
    """G23c L2 (rule S2): a row's D7 intruder is physical exactly when its runtime rule has revealed it
    (hidden before the reveal; rows without a runtime rule: physical when they carry an intruder)."""
    episode = env._d7_bridge.adapter.episodes.get(int(row))
    if episode is None:
        return False
    if episode.release is None:
        return bool(episode.objects)
    return bool(episode.release.released_at(episode.last_time_s))


import harbour_journal_fixtures as F  # noqa: E402

hj, hpool, contracts, plans, pairing, dynamic = F.hj, F.hpool, F.contracts, F.plans, F.pairing, F.dynamic
ENV_FILE = Path(__file__).resolve().parent / "harbor_mission_env.py"
ALL_ON = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "perception.bundle=D2",
          "dynamic=D2", "wind.offset_deg=30", "current.offset_deg=-45", "perception.offset_deg=90")
ALL_D0 = ("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0", "perception.bundle=D0",
          "dynamic=D0")
# SHA-256 of every HarborMissionEnv method source in F-base (20260930_fbase MANIFEST
# tasks/harbor_mission/harbor_mission_env.py). __init__ is the ONLY legacy method F6 edits
# (the journal vehicle / _journal_init branches, D15-allowed).
FBASE_METHOD_SHA256 = {
    '_setup_scene': 'bf4f0be39b7aaa250c6b1fe0b70480c76ee7b002502fbc9399b0531421d4daf6',
    '_create_static_water_mesh': '23eeeb2622f2af61116c63c228224a9946c49303b892dbc24f859a810da5c35f',
    '_get_or_add_translate_op': '49591414db054db39e53bdc52a1082d8f7337b5a77fa977f41b0b73f696748d0',
    '_define_marker_mesh': '96a6add42c07351124f7fac1d0b324425b485589f4660eb3a9fc5dc26bc7b6b0',
    '_create_render_only_markers': '7d92f9a3805e953507775e0753b1fe826fa355102279f31f411b2dd5b53d98ab',
    '_update_render_only_markers': 'b6d6829cd431fcab24912225af2806fdb64862c605aa4340ca635381e2d035d1',
    '_root_quat': 'd386a884193b8376bdc65ed834f5a62ce6bac164b866c3dadb71f28487224d8b',
    '_compute_buoyancy_forces': 'ed877f625fc473a4ed5b536980b9996e50aaa5494c1c378783d49be838f153dc',
    '_pre_physics_step': '9839d17f8b0b5beef216844d5e4e49ddfec4de91123eaaa132dd7d12125f5bac',
    '_apply_action': 'd1f289baaef0742462acd8f75b83d309cb8cb8346e303523fdfd38c2e0443ac0',
    '_com_xy': '3f963e1d604f2e719cbf23a7cff330cf843639e1826914fb93607a2d7d6bb724',
    '_linear_velocity_world': '7b2ef362bf58c1c01222f5f18635d36f7a8d830b9b83cff9a28a44dedb2a7fc5',
    '_forward_2d': 'fb4afe224a7d47137013c2c68e3ca669d49aec8bcc369cc3e29e704bcb1ec56e',
    '_clearance': '79f179e6fdf1c655268bdf9cdb36f6b144e488fa132035a83e3249a74e24a39d',
    '_active_gate': 'fadf574aca687747f840ac2482495ad6a83bbaadadfe1adc8c892826dd4dcc06',
    '_phase_target': '010a950fafc5b21b52ace4d68b5f9829f4eece51dcc47d21caf88e2a0c6e0fff',
    '_dock_state': 'f1033d54a8697bb2c72b027e6c3a9443ad7626d4f455cb95484ad657f2e21551',
    '_phase_potential': 'e9219f0ecdb76cb48db449ff6278a8a629ec0a6916b00b026d20d8e90cb86727',
    '_ray_ranges_m': '04c0e604719c6d6d86167182615ba552d08e28ec04d19c3a19304ec48a31e893',
    '_get_observations': 'f4442f0660166746411bb17e571fa5251c82c710bcbcffa98f8e4b473cbe7760',
    '_get_rewards': '0be858da38f15c9a2156707b25407993ef9f94814722ae1a262e7c48341452fa',
    '_get_dones': '391334ef3ac7bdac33991d596c8b58b843473783f63505db6a55ddb2bf41a8fd',
    '_reset_idx': 'f0fd548bd9272a6445f956c1b3e00c1d6ee258afa47ffc87fbc206e160eb3290',
}


def method_sources(path, class_name="HarborMissionEnv"):
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name: ast.get_source_segment(text, item) for item in node.body
                    if isinstance(item, ast.FunctionDef)}
    raise AssertionError("class not found")


def run(env, steps, controller=None):
    for _ in range(steps):
        action = controller(env) if controller is not None else torch.tensor([[0.5, 0.1]] * env.num_envs)
        env.step(action)


def teleport(env, row, *, xy=None, z=None, quat=None, nan=False):
    data = env.robot.data
    origin = env.scene.env_origins[row]
    if xy is not None:
        data.root_link_pose_w[row, 0] = origin[0] + float(xy[0])
        data.root_link_pose_w[row, 1] = origin[1] + float(xy[1])
    if z is not None:
        data.root_link_pose_w[row, 2] = origin[2] + float(z)
    if quat is not None:
        data.root_link_pose_w[row, 3:7] = torch.tensor(quat)
    if nan:
        data.root_link_pose_w[row, 0] = float("nan")


class LegacyIdentity(unittest.TestCase):
    def test_legacy_methods_are_byte_identical_to_fbase(self):
        sources = method_sources(ENV_FILE)
        for name, digest in FBASE_METHOD_SHA256.items():
            self.assertEqual(hashlib.sha256(sources[name].encode("utf-8")).hexdigest(), digest, name)
        fbase = F.FBASE / "tasks" / "harbor_mission" / "harbor_mission_env.py"
        if fbase.is_file():
            base = method_sources(fbase)
            for name in FBASE_METHOD_SHA256:
                self.assertEqual(sources[name], base[name], name)
            self.assertEqual(set(base) - set(sources), set())
        init = sources["__init__"]
        self.assertIn('if getattr(cfg, "journal_contract_version", 1) == 2:', init)
        self.assertIn("self.vehicle_spec = get_vehicle(cfg.vehicle)", init)
        self.assertIn('if getattr(self.cfg, "journal_contract_version", 1) == 2:', init)

    def test_legacy_cfg_values_are_identical_to_fbase(self):
        fbase_cfg = F.FBASE / "tasks" / "harbor_mission" / "harbor_mission_env_cfg.py"
        if not fbase_cfg.is_file():
            self.skipTest("F-base copy not present on this machine")
        F.install_isaaclab_stubs()
        import importlib.util

        def plain(value, depth=0):
            if depth > 6:
                return repr(type(value))
            if isinstance(value, (int, float, str, bool, type(None))):
                return value
            if isinstance(value, (tuple, list)):
                return [plain(v, depth + 1) for v in value]
            if isinstance(value, dict):
                return {k: plain(v, depth + 1) for k, v in value.items()}
            if hasattr(value, "__dict__"):
                return {k: plain(v, depth + 1) for k, v in vars(value).items() if not k.startswith("__")}
            return repr(value)

        def instances(path, name):
            spec = importlib.util.spec_from_file_location(name, path,
                                                          submodule_search_locations=None)
            module = importlib.util.module_from_spec(spec)
            module.__package__ = F.ROOT + ".harbor_mission"
            sys.modules[name] = module
            spec.loader.exec_module(module)
            classes = ("HarborMissionEnvCfg", "HarborStage1EnvCfg", "HarborStage2EnvCfg", "HarborStage3EnvCfg",
                       "HarborDockPhaseEnvCfg", "HarborStage2AllRewEnvCfg", "HarborMissionKinEnvCfg",
                       "HarborMissionWaveEnvCfg")
            return {c: plain(getattr(module, c)()) for c in classes}
        mine = instances(Path(__file__).resolve().parent / "harbor_mission_env_cfg.py",
                         F.ROOT + ".harbor_mission._cfg_mine")
        base = instances(fbase_cfg, F.ROOT + ".harbor_mission._cfg_fbase")
        self.assertEqual(mine, base)


class JournalCfg(unittest.TestCase):
    def test_templates_resolve_v2_vessel_hydro_v2_and_goal_alias(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            cfg = F.journal_cfg(vessel, pool=False)
            plan = cfg.journal_plan
            params = plan.params
            self.assertEqual(cfg.vehicle, vessel)
            self.assertEqual(cfg.underwater_physics_cfg.hydrodynamics_model, "hydro_v2")
            self.assertEqual(cfg.underwater_physics_cfg.physics_version, 2)
            self.assertEqual(cfg.underwater_physics_cfg.vessel_profile, vessel)
            self.assertTrue(cfg.underwater_physics_cfg.vessel_profile_status)
            self.assertEqual(cfg.max_obstacles, plans.STATIC_CAPACITY["harbour_mission"] + 1)
            self.assertEqual(cfg.goal_radius, params["position_tolerance_m"])            # C8 alias
            self.assertEqual(cfg.goal_radius, cfg.success_position_tolerance_m)
            self.assertEqual(cfg.observation_space, 49)
            self.assertTrue(cfg.obs_kinematic)
            self.assertEqual(cfg.ray_count, 36)
            self.assertEqual(cfg.ray_max_range_m, params["ray_sensor_max_range_m"])
            self.assertIs(cfg.journal_observation_hooks_wired, True)
            self.assertEqual(cfg.episode_length_s, plan.max_deadline_s)
            self.assertFalse(cfg.robot_cfg.spawn.rigid_props.disable_gravity)
            self.assertTrue(Path(cfg.robot_cfg.spawn.usd_path).is_file())
            record = plans.vessel_record(vessel)
            self.assertAlmostEqual(cfg.robot_cfg.init_state.pos[2],
                                   F.load("_shared.journal_vehicle").resolve_journal_vehicle(vessel)
                                   .hydro_profile.hydro.equilibrium_com_z_m)
            self.assertGreaterEqual(cfg.scene.env_spacing, 1.25 * (params["reference_length_m"]
                                                                   + 2 * params["region_padding_m"]))
            self.assertEqual(record.key, vessel)

    def test_journal_branch_rejects_legacy_vessel_names_and_jitter(self):
        module = F.templates()
        configclass = sys.modules["isaaclab.utils"].configclass
        bad = configclass(type("Bad", (module.HarbourMissionBlueBoatJournalEnvCfg,), {"spawn_phase_jitter_m": 1.0}))
        with self.assertRaisesRegex(ValueError, "spawn_phase_jitter_m"):
            bad()
        legacy = configclass(type("Legacy", (module.HarbourMissionBlueBoatJournalEnvCfg,), {"vehicle": "blueboat"}))
        with self.assertRaises(ValueError):
            legacy()


class CallOrderAndIdentity(unittest.TestCase):
    def test_substep_call_order_through_the_real_stack(self):
        env = F.make_env(tokens=("waves=D2", "dynamic=D2"))
        trace = []
        env.journal_trace = trace
        inner = []

        def spy(obj, names, prefix):
            for name in names:
                original = getattr(obj, name)

                def wrapper(*args, _o=original, _n=name, **kwargs):
                    inner.append(prefix + "." + _n)
                    return _o(*args, **kwargs)
                setattr(obj, name, wrapper)
        spy(env._hydro_v2, ("reset", "step_from_root_link"), "runtime")
        spy(env._conditions, ("reset_runtime", "configure_runtime", "physics_inputs", "episode_stamp",
                              "estimate_position", "estimate_velocity", "estimate_rays"), "conditions")
        spy(env._d7_bridge, ("reset_rows", "step", "finish"), "bridge")
        spy(env._oracle, ("reset_rows", "observe_physics", "step"), "oracle")
        spy(env, ("_get_rewards",), "env")
        env.reset()
        self.assertEqual(inner, ["oracle.reset_rows", "runtime.reset", "conditions.reset_runtime",
                                 "conditions.configure_runtime", "bridge.reset_rows", "bridge.step",
                                 "conditions.estimate_position", "conditions.estimate_velocity",
                                 "conditions.estimate_rays"])
        inner.clear()
        env.step(torch.zeros(2, 2))
        sub = ["oracle.observe_physics", "bridge.step", "conditions.physics_inputs", "runtime.step_from_root_link"]
        self.assertEqual(inner[:8], sub * 2)
        self.assertEqual(inner[8:10], ["bridge.step", "oracle.step"])
        self.assertIn("oracle.observe_physics", inner[10:11])            # the final substep inside step()
        self.assertEqual(inner[-4:], ["env._get_rewards", "conditions.estimate_position",
                                      "conditions.estimate_velocity", "conditions.estimate_rays"])
        self.assertEqual(trace, ["reset", "estimates", "pre_physics", "substep", "substep", "pre_dones", "dones",
                                 "estimates"])

    def test_zero_dose_is_bit_identical_to_no_suites(self):
        runs = []
        for tokens in ((), ALL_D0):
            env = F.make_env(tokens=tokens)
            env.reset()
            observations = []
            for k in range(12):
                obs, reward, *_ = env.step(torch.tensor([[0.8 * ((k % 7) / 7.0) - 0.2, 0.5],
                                                         [-0.3, 0.9 - 0.1 * (k % 5)]]))
                observations.append((obs["policy"].clone(), reward.clone()))
            runs.append((env, observations))
        (a, oa), (b, ob) = runs
        self.assertEqual(len(a.robot.history), len(b.robot.history))
        for (fa, ta), (fb, tb) in zip(a.robot.history, b.robot.history):
            self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
        for (x, rx), (y, ry) in zip(oa, ob):
            self.assertTrue(torch.equal(x, y) and torch.equal(rx, ry))
        self.assertTrue(torch.equal(a.robot.data.root_link_pose_w, b.robot.data.root_link_pose_w))
        self.assertTrue(any(bool(torch.count_nonzero(f)) for f, _ in a.robot.history))

    def test_wrench_is_the_runtime_step_and_actions_are_clamped_once(self):
        env = F.make_env()
        env.reset()
        seen = []
        original = env._hydro_v2.step_from_root_link

        def spy(actions, *args, **kwargs):
            seen.append(actions.clone())
            step = original(actions, *args, **kwargs)
            seen.append(step)
            return step
        env._hydro_v2.step_from_root_link = spy
        env.step(torch.tensor([[3.0, -7.0], [0.5, -0.25]]))
        self.assertTrue(torch.equal(seen[0], torch.tensor([[1.0, -1.0], [0.5, -0.25]])))
        forces, torques = env.robot.history[-1]
        self.assertTrue(torch.equal(forces[:, 0], seen[-1].force_actor.to(torch.float32)))
        self.assertTrue(torch.equal(torques[:, 0], seen[-1].torque_actor.to(torch.float32)))


class AllSuitesOn(unittest.TestCase):
    def _check(self, vessel):
        env = F.make_env(vessel, tokens=ALL_ON)
        env.reset()
        run(env, 3)
        s = float(env.cfg.journal_plan.params["length_scale"])
        bounds = env._jbounds[1]
        teleport(env, 1, xy=(float(bounds[1]) + 5.0 * s, 0.0))           # out of bounds on the next step
        env.step(torch.zeros(2, 2))
        self.assertEqual(env.extras["termination_reason"].tolist()[1], 4)
        record = env.journal_episode_record(1)
        stamp = record["conditions"]
        self.assertEqual(stamp["schema"], "usvbench.conditions/3")
        for suite in ("waves", "current", "wind", "actuator", "perception"):
            self.assertTrue(stamp["parameters"][suite]["enabled"], suite)
        runtime = stamp["runtime"]
        self.assertEqual(runtime["runtime_ticks"], 8)
        self.assertEqual(runtime["estimates"]["counts"], {"pos": 4, "vel": 4, "ray": 4})
        self.assertAlmostEqual(runtime["direction_offsets_rad"]["wind"], math.radians(30.0))
        dynamic.read_stamp(record["dynamic_hazards"])
        payload = record["dynamic_hazards"]["payload"]
        self.assertEqual(payload["assignment"]["dose"], 2)
        self.assertEqual(payload["assignment"]["task"], "harbour")
        self.assertEqual(payload["runtime"]["termination_reason"], "out_of_bounds")
        self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 8)
        self.assertEqual(sorted(record["scenario_hashes"]), sorted(env.journal_pairing_inventory))
        for name in ("layout", "route", "berth", "spawn_pose"):
            self.assertIn(name, record["scenario_hashes"])
        contract = env.journal_contract_record(1)
        self.assertEqual(contract["termination_reason_name"], "out_of_bounds")
        self.assertEqual(contract["contract_scope"], "harbour_full_mission")
        header = env.journal_certificate_header()
        self.assertEqual(header["dynamic_hazards_header"]["pool_digest"], env._d7_pool.pool_digest)
        self.assertEqual(stamp["context"]["vehicle"], vessel)
        self.assertEqual(env.episode_length_buf.tolist()[1], 0)                # row 1 was reset after the latch
        return env

    def test_all_suites_blueboat(self):
        self._check("blueboat_v2")

    def test_all_suites_wamv(self):
        self._check("wamv_vrx_v2_opt1")


class Termination(unittest.TestCase):
    def test_codes_2_3_6_7_8_publish_and_latch_before_reset(self):
        env = F.make_env(tokens=("dynamic=D1",), pool_episodes=4)
        env.reset()
        run(env, 2)
        cases = []
        field = env._jlayouts[0].field_indices[0]
        center = env._jstatic_centers[0, field].tolist()
        teleport(env, 0, xy=center)                                            # onto a field disk
        cases.append((0, 2))
        env._oracle.deadline_s[1] = float(env.episode_length_buf[1] + 1) * plans.CONTROL_DT_S
        cases.append((1, 3))
        env.step(torch.zeros(2, 2))
        codes = env.extras["termination_reason"].tolist()
        for row, code in cases:
            self.assertEqual(codes[row], code)
            self.assertEqual(int(env.episode_termination_reason[row]), code)
            record = env.journal_episode_record(row)
            name = contracts.TERMINATION_REASON_NAMES[code]
            self.assertEqual(record["dynamic_hazards"]["payload"]["runtime"]["termination_reason"], name)
            self.assertEqual(env.journal_contract_record(row)["termination_reason"], code)
        self.assertTrue(torch.equal(env.termination_reason, env.extras["termination_reason"]))
        run(env, 1)
        teleport(env, 0, quat=(math.cos(0.7), math.sin(0.7), 0.0, 0.0))        # 80 deg roll
        teleport(env, 1, z=-5.0)
        env.step(torch.zeros(2, 2))
        self.assertEqual(env.extras["termination_reason"].tolist(), [6, 7])
        run(env, 1)
        teleport(env, 1, z=5.0)
        env.step(torch.zeros(2, 2))
        self.assertEqual(env.extras["termination_reason"].tolist()[1], 8)
        self.assertEqual(env.journal_contract_record(1)["termination_reason_name"], "airborne")

    def test_non_finite_state_is_contract_code_9_but_the_mixin_d7_pose_fails_first(self):
        # The tensor twin classifies a non-finite state as code 9 (parity test); in the env the
        # F0 mixin builds the E4 Pose of every row BEFORE the contract step and E4 rejects NaN.
        # Recorded as a cross-owner seam (FINAL_REPORT, CROSS_OWNER_PROPOSAL_mixin_nonfinite_pose).
        env = F.make_env()
        env.reset()
        teleport(env, 0, nan=True)
        mixin = (F.REPO / "tasks" / "_shared" / "journal_env_mixin.py").read_text(encoding="utf-8")
        if "if all(math.isfinite(v) for v in r)" in mixin:          # proposal applied: code 9 is published
            env.step(torch.zeros(2, 2))
            self.assertEqual(env.extras["termination_reason"].tolist()[0], 9)
            self.assertEqual(env.journal_contract_record(0)["termination_reason_name"], "non_finite_state")
        else:
            with self.assertRaisesRegex(ValueError, "finite"):
                env.step(torch.zeros(2, 2))


class ResetIsolation(unittest.TestCase):
    def test_reset_rows_are_isolated(self):
        env = F.make_env(tokens=("perception=D2", "wind=D2", "dynamic=D2"))
        env.reset()
        run(env, 5, F.phase_target_controller)
        keep = {"ticks": int(env._hydro_v2.ticks[0]), "counts": env._conditions.estimate_counts(0),
                "d7": int(env._d7_tick[0]), "oracle": (float(env._oracle.full_len[0]), float(env._oracle.prev_time[0])),
                "centers": env.obstacle_centers[0].clone(), "phase": int(env.phase[0])}
        env._reset_idx(torch.tensor([1]))
        self.assertEqual(int(env._hydro_v2.ticks[0]), keep["ticks"])
        self.assertEqual(env._conditions.estimate_counts(0), keep["counts"])
        self.assertEqual(int(env._d7_tick[0]), keep["d7"])
        self.assertEqual((float(env._oracle.full_len[0]), float(env._oracle.prev_time[0])), keep["oracle"])
        self.assertTrue(torch.equal(env.obstacle_centers[0], keep["centers"]))
        self.assertEqual(int(env.phase[0]), keep["phase"])
        self.assertEqual(int(env._hydro_v2.ticks[1]), 0)
        self.assertEqual(float(env._oracle.full_len[1]), 0.0)
        self.assertEqual(len(env.journal_aborted_dynamic_hazards), 1)       # manual reset: stamped "aborted"


class DynamicHazards(unittest.TestCase):
    def test_d7_forced_reveal_physical_equals_revealed(self):
        """G23d (REV-G23c H11): drive the rows until a D7 intruder is REVEALED, so that "physical == revealed" is
        checked on both sides of a reveal (the other env tests may never see one)."""
        env = F.make_env(tokens=("dynamic=D4",))
        env.reset()
        base = plans.STATIC_CAPACITY["harbour_mission"]
        before = after = 0
        for _ in range(6000):
            env.step(F.phase_target_controller(env))
            shown = [_d7_shown(env, i) for i in range(env.num_envs)]
            self.assertEqual(env._d7_physical[:, base].tolist(), shown)
            before += shown.count(False)
            after += shown.count(True)
            if after >= 5:
                break
        self.assertGreater(before, 0)
        self.assertGreaterEqual(after, 5, "no D7 reveal within 6000 steps")
        revealed = [i for i in range(env.num_envs) if _d7_shown(env, i)]
        rel = env._d7_bridge.adapter.episodes[revealed[0]].release
        self.assertIsNotNone(rel.reveal_s)
        self.assertTrue(bool(env._d7_physical[revealed[0], base]))

    def test_d7_slot_masks_and_storage(self):
        env = F.make_env(tokens=("dynamic=D4",))
        env.reset()
        base = plans.STATIC_CAPACITY["harbour_mission"]
        self.assertEqual(env.obstacle_centers.shape[1], base + 1)
        physical_seen = shown_seen = False
        for _ in range(240):
            env.step(F.phase_target_controller(env))
            self.assertFalse(bool(env.obstacle_active[:, base:].any()))       # D7 never writes obstacle_active
            self.assertFalse(bool(env._d7_physical[:, :base].any()))
            physical_seen |= bool(env._d7_physical[:, base].any())
            # G23c L2 (rule S2): physical exactly when revealed
            self.assertEqual(env._d7_physical[:, base].tolist(), [_d7_shown(env, i) for i in range(env.num_envs)])
            shown_seen = shown_seen or any(_d7_shown(env, i) for i in range(env.num_envs))
            vis = env._d7_visible[:, base]
            self.assertFalse(bool((vis & ~env._d7_physical[:, base]).any()))
        self.assertEqual(physical_seen, shown_seen)
        stages = {candidate.stage for candidate in
                  (env._d7_bridge.adapter.episodes[i].assignment.candidate for i in range(2))}
        self.assertTrue(stages <= {"departure", "transit", "docking"})

    def test_zero_dose_has_no_d7_object(self):
        env = F.make_env(tokens=("dynamic=D0",))
        env.reset()
        run(env, 20)
        self.assertFalse(bool(env._d7_physical.any()) or bool(env._d7_visible.any()))

    def test_pool_geometry_mismatch_fails_closed_at_reset(self):
        raw, meta = F.small_pool("blueboat_v2", envs=2, episodes=2)
        other = F.small_pool("blueboat_v2", envs=2, episodes=2, tier=1)[0]
        env = F.make_env(pool=False, journal_d7_pool=hpool.HarbourPoolView(other))
        with self.assertRaisesRegex(ValueError, "geometry"):
            env.reset()
        del raw, meta


class PhaseTargets(unittest.TestCase):
    def test_frame_stage_progress_and_no_d7_slot(self):
        env = F.make_env(tokens=("dynamic=D2",))
        env.reset()
        targets = env.journal_phase_targets()
        self.assertEqual(targets["schema"], hj.PHASE_TARGETS_SCHEMA)
        self.assertEqual(targets["frame"], "env_local_com_xy")
        self.assertEqual(targets["stage"].tolist(), [0, 0])
        self.assertEqual(targets["active_gate_index"].tolist(), [0, 0])
        truth = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
        self.assertTrue(torch.allclose(env.journal_estimate.position.double(), truth.double()))   # perception off
        layout = env._jlayouts[0]
        self.assertTrue(np.allclose(targets["gate_midpoints"][0].numpy(),
                                    hj.gate_table(layout)["midpoints"]))
        self.assertEqual(tuple(targets["static_obstacles"]["centers"].shape[1:]),
                         (plans.STATIC_CAPACITY["harbour_mission"], 2))
        self.assertEqual(int(targets["static_obstacles"]["mask"][0].sum()), layout.obstacle_count)
        self.assertTrue(np.allclose(targets["berth_pose"][0, :2].numpy(), layout.extras["berth"]["center_xy"]))
        seen = set()
        for _ in range(520):
            env.step(F.phase_target_controller(env))
            t = env.journal_phase_targets()
            seen |= set(t["stage"].tolist())
            passed = env._oracle.gate_passed()
            self.assertTrue(torch.equal(t["gate_passed"].cpu(), passed))
            self.assertTrue(torch.equal(t["stage"].cpu(), env._oracle.stage()))
            self.assertTrue(torch.equal(t["stage"].cpu(), env.phase.cpu()))
        self.assertIn(1, seen)                                          # both departure gates crossed

    def test_dock_phase_scope_and_skipped_stage_record(self):
        env = F.make_env(spawn_phase=2, tokens=("dynamic=D1",))
        env.reset()
        targets = env.journal_phase_targets()
        self.assertEqual(targets["contract_scope"], "harbour_dock_phase")
        self.assertEqual(targets["stage"].tolist(), [2, 2])
        self.assertFalse(bool(targets["gate_applicable"][:, :3].any()))
        self.assertEqual(targets["active_gate_index"].tolist(), [3, 3])
        exit_mid = hj.gate_table(env._jlayouts[0])["midpoints"][2]
        truth = env.robot.data.root_link_pose_w[0, :2] - env.scene.env_origins[0, :2]
        self.assertTrue(np.allclose(truth.numpy(), exit_mid, atol=1e-4))
        stages = [env._jd7_stage[i] for i in range(2)]
        bounds = env._jbounds
        for row in range(2):
            teleport(env, row, xy=(float(bounds[row, 1]) + 20.0, 0.0))
        env.step(torch.zeros(2, 2))
        for row in range(2):
            record = env.journal_contract_record(row)
            self.assertEqual(record["contract_scope"], "harbour_dock_phase")
            self.assertIsNone(record["departure_gate_times_s"])
            expected = ("skipped_by_spawn_phase" if stages[row] in ("departure", "transit") else "in_scope")
            self.assertTrue(record["d7_stage_exposure"].startswith(expected))


class Pairing(unittest.TestCase):
    def test_family_hashes_are_exogenous_and_dose_independent(self):
        hashes = []
        for tokens in (("perception=D1",), ("perception=D3",)):
            env = F.make_env(tokens=tokens)
            env.reset()
            run(env, 2)
            hashes.append([env._journal_family_scenario_hashes(i) for i in range(2)])
            for i in range(2):
                pairing.validate_family_hashes(env._journal_family_scenario_hashes(i), "harbor_mission")
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0][0]["layout"], hashes[0][1]["layout"])


class PoolIntegrity(unittest.TestCase):
    def test_write_load_verify_and_tamper(self):
        raw, meta = F.small_pool("blueboat_v2", envs=2, episodes=2)
        tmp = Path(tempfile.mkdtemp(prefix="f6_pool_", dir=str(Path(__file__).resolve().parent)))
        try:
            path = tmp / hpool.pool_file_name("blueboat_v2", 42)
            digest = hpool.write_pool(path, raw, meta)
            loaded, lmeta = hpool.load_pool(path, dynamic, vessel="blueboat_v2", expected_sha256=digest)
            self.assertEqual(loaded.pool_digest, raw.pool_digest)
            self.assertEqual(set(loaded.assignments), set(raw.assignments))
            view = hpool.HarbourPoolView(loaded, lmeta)
            key = dynamic.EpisodeKey(42, 1, 1, hpool.MIXIN_GROUP)
            self.assertIs(view.assignment(key, 3), loaded.assignment(dynamic.EpisodeKey(42, 1, 1, "d7_harbour"), 3))
            with self.assertRaises(hpool.PoolIntegrityError):
                hpool.load_pool(path, dynamic, vessel="wamv_vrx_v2_opt1")
            # env loads BY PATH (cfg.journal_d7_pool_path) and verifies
            env = F.make_env(pool=False, journal_d7_pool_path=str(path))
            env.reset()
            self.assertEqual(env._d7_pool.pool_digest, raw.pool_digest)
            # tamper: one byte of the file -> sidecar mismatch
            blob = bytearray(path.read_bytes())
            blob[len(blob) // 2] ^= 0x01
            bad = tmp / "bad.json.gz"
            bad.write_bytes(bytes(blob))
            shutil.copy(str(path) + ".sha256", str(bad) + ".sha256")
            with self.assertRaisesRegex(hpool.PoolIntegrityError, "SHA-256"):
                hpool.load_pool(bad, dynamic)
            # tamper: a candidate value with a consistent sidecar -> candidate digest mismatch
            document = json.loads(gzip.decompress(path.read_bytes()).decode("ascii"))
            first = document["pool"]["f"]["assignments"]["__d__"][0][1]["f"]
            first["candidate_digest"] = "0" * 64
            forged = tmp / "forged.json.gz"
            raw_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
            forged.write_bytes(gzip.compress(raw_bytes, mtime=0))
            (Path(str(forged) + ".sha256")).write_text(hpool.file_sha256(forged) + "  forged\n", encoding="ascii")
            with self.assertRaisesRegex(hpool.PoolIntegrityError, "digest"):
                hpool.load_pool(forged, dynamic)
            with self.assertRaisesRegex(ValueError, "D7 pool"):
                F.make_env(pool=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_pinned_manifest_and_evidence_pools_verify(self):
        manifest = json.loads(hpool.PINNED_MANIFEST.read_text(encoding="ascii"))
        self.assertEqual(manifest["schema"], hpool.PINNED_SCHEMA)
        self.assertEqual(manifest["pool_schema"], hpool.POOL_SCHEMA)
        entries = {e["vessel"]: e for e in manifest["pools"]
                   if e["seed"] == 42 and e["tier"] == 2 and e["grid"] == {"envs": 64, "episodes": 2}}
        self.assertEqual(set(entries), {"blueboat_v2", "wamv_vrx_v2_opt1"})
        found = 0
        for vessel, entry in sorted(entries.items()):
            self.assertEqual(entry["keys_offered"], 128)
            self.assertEqual(entry["keys_admitted"] + entry["keys_not_admitted"], 128)
            self.assertEqual(entry["per_dose_counts"],
                             [[128, entry["keys_admitted"], entry["keys_not_admitted"]]] * 5)
            folder = hpool.PINNED_DIR if manifest["pools_in_repo"] else F.REPO.parent / "evidence"
            path = folder / entry["file"]
            if not path.is_file():
                continue
            found += 1
            pool, meta = hpool.load_pool(path, dynamic, vessel=vessel)
            self.assertTrue(meta["pinned"])
            self.assertEqual(meta["pool_manifest_sha256"], hpool.file_sha256(hpool.PINNED_MANIFEST))
            self.assertEqual(pool.pool_digest, entry["pool_digest"])
            self.assertEqual(meta["grid"], {"envs": 64, "episodes": 2})
            self.assertEqual(sorted([k.env, k.episode] for k in meta["not_admitted_map"]), entry["not_admitted_keys"])
            view = hpool.HarbourPoolView(pool, meta)
            self.assertEqual(view.grid, {"num_envs": 64, "episodes": 2})                  # D24 offered grid
            self.assertEqual(meta["offered_grid"]["eval_seed"], 42)
            beyond = view.assignment(dynamic.EpisodeKey(42, 63, 2, "d7_harbour"), 4)
            self.assertTrue(isinstance(beyond, hpool.NotAdmitted) and beyond.beyond_grid)
            with self.assertRaises(KeyError):
                view.assignment(dynamic.EpisodeKey(42, 64, 0, "d7_harbour"), 0)
            for env_index, episode in [(5, 1)] + [tuple(k) for k in entry["not_admitted_keys"][:3]]:
                layout = hj.journal_layout(vessel, 2, 42, env_index, episode)
                record = view.assignment(dynamic.EpisodeKey(42, env_index, episode, "d7_harbour"), 0)
                geometry = (record.geometry_id if isinstance(record, hpool.NotAdmitted)
                            else record.candidate.geometry_id)
                self.assertEqual(geometry, layout.geometry_hash)
        if not found:
            self.skipTest("work/evidence pools not present next to this repo copy")


class NotAdmittedD16(unittest.TestCase):
    """Round 4b D16: a key E4 rejects keeps its E5 layout and runs the dynamic suite at D0 (exposure not_admitted)."""

    REJECT = ((0, 1), (1, 0))          # env 1 episode 0 (first episode of row 1), env 0 episode 1 (row 0's second)
    OTHER_SUITES = ("waves", "wind", "current", "actuator", "perception")

    def _record(self, tokens, row=1):
        env = F.make_env(tokens=tokens, reject=self.REJECT)
        env.reset()
        run(env, 2)
        env._oracle.deadline_s[row] = float(env.episode_length_buf[row] + 1) * plans.CONTROL_DT_S
        env.step(torch.zeros(2, 2))
        self.assertEqual(env.extras["termination_reason"].tolist()[row], 3)
        return env.journal_episode_record(row), env

    def test_not_admitted_key_runs_a_full_episode_at_every_dynamic_level(self):
        base = plans.STATIC_CAPACITY["harbour_mission"]
        for dose in range(5):
            env = F.make_env(tokens=("dynamic=D%d" % dose,), reject=self.REJECT)
            env.reset()
            ident = [env._d7_bridge.row_identity(i) for i in range(2)]
            self.assertEqual([x["exposure"] for x in ident], ["admitted", "not_admitted"])
            self.assertEqual(ident[1]["requested_dose"], dose)
            # row 0 (admitted key 0/0) ends after 0.5 s, row 1 (not admitted key 1/0) after 1.0 s: whole
            # episodes from reset to the (shortened) deadline, then both rows switch exposure at the reset
            env._oracle.deadline_s[0] = 0.5
            env._oracle.deadline_s[1] = 1.0
            seen = {0: [], 1: []}
            for _ in range(61):
                env.step(F.phase_target_controller(env))
                codes = env.extras["termination_reason"].tolist()
                for row in (0, 1):
                    if codes[row]:
                        seen[row].append(env.journal_episode_record(row))
                        if row == 0:
                            self.assertEqual(env._d7_bridge.row_identity(0)["exposure"], "not_admitted")  # 0/1
                        else:
                            self.assertEqual(env._d7_bridge.row_identity(1)["exposure"], "admitted")      # 1/1
                if not seen[1]:
                    self.assertFalse(bool(env._d7_physical[1, base:].any()) or bool(env._d7_visible[1, base:].any()))
                    self.assertFalse(env._d7_bridge.contact(1))
                    self.assertEqual(env._journal_d7_samples[1], ())
            self.assertEqual(len(seen[1]), 1, dose)
            self.assertGreaterEqual(len(seen[0]), 1, dose)
            record = seen[1][0]
            stamp = dynamic.read_stamp(record["dynamic_hazards"])
            payload = stamp["payload"]
            self.assertEqual(payload["runtime"]["exposure"], "not_admitted")
            self.assertIs(payload["runtime"]["assigned"], False)                  # D27: never assigned
            self.assertEqual(payload["runtime"]["itt_count"], 0)
            admitted_runtime = dynamic.read_stamp(seen[0][0]["dynamic_hazards"])["payload"]["runtime"]
            self.assertIs(admitted_runtime["assigned"], True)                     # E4 admitted stamp
            self.assertEqual(admitted_runtime["itt_count"], 1)
            self.assertEqual(payload["runtime"]["requested_dose"], dose)
            self.assertEqual(payload["runtime"]["termination_reason"], "timeout")
            self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 120)          # 60 control steps x 2
            self.assertEqual(payload["assignment"]["dose"], 0)
            self.assertIsNone(payload["assignment"]["candidate_digest"])
            self.assertEqual(payload["assignment"]["key"], {"eval_seed": 42, "env": 1, "episode": 0,
                                                            "hazard_group": "d7_harbour"})
            self.assertTrue(payload["assignment"]["not_admitted_failing_reasons"])
            self.assertEqual(hpool.stamp_exposure(record["dynamic_hazards"]), "not_admitted")
            self.assertEqual(hpool.stamp_exposure(seen[0][0]["dynamic_hazards"]), "admitted")
            self.assertEqual(sorted(record["scenario_hashes"]), sorted(env.journal_pairing_inventory))
            self.assertEqual(record["scenario_hashes"]["d7_assignment"],
                             pairing.d7_primitive_hash(record["dynamic_hazards"]))
            self.assertEqual(record["conditions"]["runtime"]["runtime_ticks"], 120)
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["per_dose_counts"], [[4, 2, 2]] * 5)
            self.assertEqual(header["per_dose_counts_fields"], ["offered", "admitted", "not_admitted"])
            self.assertEqual(header["keys_not_admitted"], 2)
            self.assertEqual(header["not_admitted_keys"], [[0, 1], [1, 0]])
            self.assertEqual(header["requested_dose"], dose)
            contract = env.journal_contract_record(1)                        # latched: the not-admitted episode
            self.assertEqual(contract["d7_exposure"], "not_admitted")
            self.assertEqual(contract["d7_requested_dose"], dose)
            self.assertTrue(contract["d7_stage_exposure"].startswith("not_admitted"))
            self.assertTrue(contract["d7_not_admitted_reasons"])
            self.assertEqual(env.journal_contract_record(0)["d7_exposure"], "admitted")

    def test_not_admitted_records_pair_with_d0_of_every_other_suite(self):
        dyn = {dose: self._record(("dynamic=D%d" % dose,))[0] for dose in range(5)}
        hashes = {dyn[d]["scenario_hashes"]["d7_assignment"] for d in dyn}
        self.assertEqual(len(hashes), 1)                                  # unexposed at every requested dose
        for dose in range(1, 5):
            with self.assertRaisesRegex(pairing.PairingError, "no treatment difference"):
                pairing.validate_pair(dyn[dose], dyn[0], treatment="dynamic")
        for suite in self.OTHER_SUITES:
            other = self._record(("%s=D2" % suite, "dynamic=D0"))[0]
            for dose in range(5):
                summary = pairing.validate_pair(dyn[dose], other, treatment=suite)
                self.assertNotIn("d7_assignment", summary["changed_primitives"])
                self.assertTrue(set(summary["changed_primitives"]) <= set(pairing.SUITE_PRIMITIVES[suite]))
        # an ADMITTED key (row 0 = key 0/0) still pairs as an exposed dynamic treatment
        admitted = {}
        for dose in (0, 2):
            admitted[dose] = self._record(("dynamic=D%d" % dose,), row=0)[0]
        self.assertEqual(pairing.validate_pair(admitted[2], admitted[0], treatment="dynamic")["changed_primitives"],
                         ["d7_assignment"])

    def test_not_admitted_layout_is_the_e5_layout_no_redraw(self):
        raw, meta, info = F.forced_pool("blueboat_v2", envs=2, episodes=2, reject=self.REJECT)
        e5 = hj.journal_layout("blueboat_v2", 2, 42, 1, 0)
        key = dynamic.EpisodeKey(42, 1, 0, "d7_harbour")
        self.assertEqual(info["layouts"][key].geometry_hash, e5.geometry_hash)
        entry = [e for e in meta["not_admitted"] if (e[0], e[1]) == (1, 0)][0]
        self.assertEqual(entry[2], e5.geometry_hash)
        forced = F.make_env(tokens=("dynamic=D2",), reject=self.REJECT)
        normal = F.make_env(tokens=("dynamic=D2",))
        forced.reset()
        normal.reset()
        self.assertEqual(forced._d7_bridge.row_identity(1)["exposure"], "not_admitted")
        self.assertEqual(normal._d7_bridge.row_identity(1)["exposure"], "admitted")
        for env in (forced, normal):
            self.assertEqual(env._jlayouts[1].geometry_hash, e5.geometry_hash)
        self.assertEqual(forced._journal_family_scenario_hashes(1), normal._journal_family_scenario_hashes(1))
        self.assertTrue(torch.equal(forced.obstacle_centers[1, :-1], normal.obstacle_centers[1, :-1]))
        self.assertTrue(torch.equal(forced.robot.data.root_link_pose_w[1], normal.robot.data.root_link_pose_w[1]))
        # attempt 0 only: ONE admit_grid call, no stage-site variant or re-attempt machinery left
        import dataclasses

        calls = []
        proxy = type("DynamicProxy", (), {})()
        for name in ("EpisodeKey", "Circle", "StageSite", "TaskContext", "Pose", "SensorSpec"):
            setattr(proxy, name, getattr(dynamic, name))
        proxy.__name__ = dynamic.__name__

        def counting(jobs, **kw):
            jobs = list(jobs)
            calls.append(len(jobs))
            return dynamic.admit_grid(jobs, **kw)
        proxy.admit_grid = counting

        def context(layout, env_envelope, dyn, episode_key):
            ctx = hj.d7_task_context(layout, env_envelope, dyn)
            if (episode_key.env, episode_key.episode) in self.REJECT:
                ctx = dataclasses.replace(ctx, deadline_s=F.FORCED_DEADLINE_S)
            return ctx

        suite = F.journal_config.resolve_journal_conditions(presets=("dynamic=D2",), seed=42).dynamic_config(
            "blueboat_v2")
        pool, info2 = hj.admit_harbour_grid(proxy, "blueboat_v2", 2, 42, [(e, k) for e in range(2) for k in range(2)],
                                            suite, F.envelope("blueboat_v2"), context_fn=context)
        self.assertEqual(calls, [4])
        self.assertEqual(pool.pool_digest, raw.pool_digest)
        self.assertEqual(sorted((k.env, k.episode) for k in info2["not_admitted"]), sorted(self.REJECT))
        tree = ast.parse((Path(__file__).resolve().parent / "harbor_journal.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            names.update(getattr(node, attr) for attr in ("id", "arg", "attr", "name")
                         if isinstance(getattr(node, attr, None), str))
        self.assertFalse({n for n in names if "attempt" in n.lower() or "variant" in n.lower()
                          or "redraw" in n.lower()}, "re-attempt machinery left in harbor_journal")

    def test_pool_file_not_admitted_list_and_pin_fail_closed(self):
        raw, meta, _info = F.forced_pool("blueboat_v2", envs=2, episodes=2, reject=self.REJECT)
        tmp = Path(tempfile.mkdtemp(prefix="f6_pool4b_", dir=str(Path(__file__).resolve().parent)))
        try:
            path = tmp / hpool.pool_file_name("blueboat_v2", 42)
            digest = hpool.write_pool(path, raw, meta)
            loaded, lmeta = hpool.load_pool(path, dynamic, vessel="blueboat_v2")
            self.assertEqual(sorted((k.env, k.episode) for k in lmeta["not_admitted_map"]), sorted(self.REJECT))
            view = hpool.HarbourPoolView(loaded, lmeta)
            record = view.assignment(dynamic.EpisodeKey(42, 1, 0, "d7_harbour"), 3)
            self.assertIsInstance(record, hpool.NotAdmitted)
            self.assertEqual(record.requested_dose, 3)
            with self.assertRaises(KeyError):                                 # outside the admitted grid
                view.assignment(dynamic.EpisodeKey(42, 7, 0, "d7_harbour"), 0)
            # the env loads it BY PATH and stamps the D16 header fields
            env = F.make_env(pool=False, tokens=("dynamic=D1",), journal_d7_pool_path=str(path))
            env.reset()
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["pool_file_sha256"], digest)
            self.assertEqual(header["per_dose_counts"], [[4, 2, 2]] * 5)
            self.assertEqual(header["pool_source"], "cfg.journal_d7_pool_path")
            self.assertEqual(env._d7_bridge.row_identity(1)["exposure"], "not_admitted")
            document = json.loads(gzip.decompress(path.read_bytes()).decode("ascii"))

            def forge(name, mutate):
                doc = copy.deepcopy(document)
                mutate(doc)
                target = tmp / name
                target.write_bytes(gzip.compress(json.dumps(doc, sort_keys=True, separators=(",", ":"))
                                                 .encode("ascii"), mtime=0))
                Path(str(target) + ".sha256").write_text(hpool.file_sha256(target) + "  x\n", encoding="ascii")
                return target

            listed = lambda doc: [kv for kv in doc["meta"]["__d__"] if kv[0] == "not_admitted"][0][1]
            dropped = forge("dropped.json.gz", lambda doc: listed(doc).pop())
            with self.assertRaisesRegex(hpool.PoolIntegrityError, "not_admitted"):
                hpool.load_pool(dropped, dynamic)

            def bad_reason(doc):
                listed(doc)[0][4][0][1] = "invented_reason"
            with self.assertRaisesRegex(hpool.PoolIntegrityError, "reasons"):
                hpool.load_pool(forge("reason.json.gz", bad_reason), dynamic)

            def bad_geometry(doc):
                [e for e in listed(doc) if (e[0], e[1]) == (1, 0)][0][2] = "0" * 64    # row 1's first key
            geo = forge("geometry.json.gz", bad_geometry)
            env = F.make_env(pool=False, tokens=("dynamic=D1",), journal_d7_pool_path=str(geo))
            with self.assertRaisesRegex(ValueError, "geometry"):              # env layout != recorded E5 layout
                env.reset()
            old = forge("old_schema.json.gz", lambda doc: doc.update(schema="usvbench.harbour_d7_pool/1"))
            with self.assertRaisesRegex(hpool.PoolIntegrityError, "schema"):
                hpool.load_pool(old, dynamic)
            # D19 pin: a manifest listing this (vessel, seed, tier, grid) with another digest refuses the file
            manifest = tmp / "POOL_MANIFEST.json"
            entry = {"vessel": "blueboat_v2", "seed": 42, "tier": 2, "grid": meta["grid"], "file_sha256": "1" * 64,
                     "pool_digest": raw.pool_digest}
            manifest.write_text(json.dumps({"schema": hpool.PINNED_SCHEMA, "pools": [entry]}), encoding="ascii")
            with self.assertRaisesRegex(hpool.PoolIntegrityError, "pinned"):
                hpool.load_pool(path, dynamic, manifest_path=manifest)
            entry["file_sha256"] = digest
            manifest.write_text(json.dumps({"schema": hpool.PINNED_SCHEMA, "pools": [entry]}), encoding="ascii")
            _pool, pmeta = hpool.load_pool(path, dynamic, manifest_path=manifest)
            self.assertTrue(pmeta["pinned"])
            self.assertEqual(pmeta["pool_manifest_sha256"], hpool.file_sha256(manifest))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RecordedTraceParity(unittest.TestCase):
    def test_env_twin_matches_the_scalar_oracle_on_recorded_traces(self):
        for vessel, steps in (("blueboat_v2", 700), ("wamv_vrx_v2_opt1", 400)):
            env = F.make_env(vessel, tokens=("perception=D2", "waves=D1", "dynamic=D2"))
            record = plans.vessel_record(vessel)
            log = []
            oracle = env._oracle
            original_reset, original_obs, original_step = oracle.reset_rows, oracle.observe_physics, oracle.step

            def reset_rows(ids, **kw):
                rows = list(torch.as_tensor(ids).reshape(-1).tolist())
                # G23b: the layout of THIS reset (env._jlayouts is set before oracle.reset_rows)
                log.append(("reset", rows, copy.deepcopy(kw), [env._jlayouts[i] for i in rows]))
                return original_reset(ids, **kw)

            def observe(state, **kw):
                log.append(("observe", {k: v.clone() for k, v in state.items()}, copy.deepcopy(kw)))
                return original_obs(state, **kw)

            def step(time_s, state, **kw):
                out = original_step(time_s, state, **kw)
                log.append(("step", time_s.clone(), {k: v.clone() for k, v in state.items()}, copy.deepcopy(kw),
                            tuple(x.clone() for x in out)))
                return out
            oracle.reset_rows, oracle.observe_physics, oracle.step = reset_rows, observe, step
            env.reset()
            run(env, steps, F.phase_target_controller)
            lo, hi = record.heave_limits_m
            limits = contracts.ValidityLimits(record.tilt_limit_rad, lo, hi)
            scalars = {}
            layouts = {}
            compared = 0
            for entry in log:
                if entry[0] == "reset":
                    for j, i in enumerate(entry[1]):
                        layout = entry[3][j]
                        kw = entry[2]
                        scalars[i] = hj.scalar_oracle(layout, record, 0, limits,
                                                      deadline_s=float(kw["deadline_s"][j]))
                        scalars[i].reset(tuple(float(v) for v in kw["com_xy"][j]))
                elif entry[0] == "observe":
                    state, kw = entry[1], entry[2]
                    for i, s in scalars.items():
                        if s.done:
                            continue
                        args = {"clearance_m": float(kw["clearance"][i]) if bool(kw["has_clearance"][i]) else None}
                        if kw.get("motor_thrust") is not None and bool(kw["thrust_mask"][i]):
                            args.update(dt_s=kw["dt_s"], motor_thrust_n=kw["motor_thrust"][i].tolist())
                        s.observe_physics(self._rigid(state, i), **args)
                else:
                    _, t, state, kw, (term, trunc, reason) = entry
                    for i, s in scalars.items():
                        d = s.step(float(t[i]), self._rigid(state, i), collision=bool(kw["collision"][i]),
                                   out_of_bounds=bool(kw["out_of_bounds"][i]),
                                   clearance_m=float(kw["clearance"][i]) if bool(kw["has_clearance"][i]) else None)
                        self.assertEqual((d.terminated, d.truncated), (bool(term[i]), bool(trunc[i])))
                        self.assertEqual(int(d.reason) if d.done else 0, int(reason[i]))
                        compared += 1
                    for i, s in list(scalars.items()):
                        if s.done:
                            self.assertEqual(s.last_completed.evaluator_fields()["termination_reason"],
                                             oracle.last_completed[i].termination_reason)
                            layouts[i] = None
            self.assertGreater(compared, steps)
            if vessel == "blueboat_v2":                    # the WAM-V reaches gate 1 only after ~30 s
                self.assertTrue(any(s.gate_times_s[0] is not None for s in scalars.values()), vessel)

    @staticmethod
    def _rigid(state, i):
        return contracts.RigidState(tuple(state["com"][i].tolist()), tuple(state["quat"][i].tolist()),
                                    tuple(state["lin"][i].tolist()), tuple(state["ang"][i].tolist()),
                                    water_level_m=float(state["water"][i]),
                                    nominal_com_height_m=float(state["nominal"][i]))


class ComReference(unittest.TestCase):
    def test_nonzero_com_offset_is_rejected(self):
        module = F.env_module()
        original = F.FakeData.__init__

        def shifted(self, n, init_state):
            original(self, n, init_state)
            self.com_pos_b[:, 0, 0] = 0.05
        F.FakeData.__init__ = shifted
        try:
            with self.assertRaisesRegex(ValueError, "COM"):
                F.make_env()
        finally:
            F.FakeData.__init__ = original
        del module

    def test_spawn_puts_the_com_at_the_equilibrium_datum(self):
        for vessel in ("blueboat_v2", "wamv_vrx_v2_opt1"):
            env = F.make_env(vessel)
            env.reset()
            z = env.robot.data.root_link_pose_w[:, 2] - env.scene.env_origins[:, 2]
            self.assertTrue(torch.allclose(z.double(), torch.full((2,), env._jdatum, dtype=torch.float64),
                                           atol=1e-6))
            layout = env._jlayouts[0]
            yaw = 2.0 * math.atan2(float(env.robot.data.root_link_pose_w[0, 6]),
                                   float(env.robot.data.root_link_pose_w[0, 3]))
            self.assertAlmostEqual(yaw, layout.start_heading_rad, places=5)


class BeyondOfferedGridD24(unittest.TestCase):
    """Round T D24: task episodes beyond the offered grid run at D0 as not-admitted keys and are never counted."""

    GRID = {"num_envs": 2, "episodes": 2}

    @staticmethod
    def _finish_now(env, row):
        """End ``row``'s current episode at the next control step (timeout), as the evaluators' fast envs do."""
        env._oracle.deadline_s[row] = float(env.episode_length_buf[row] + 1) * plans.CONTROL_DT_S

    def test_beyond_grid_episode_runs_at_every_dynamic_level(self):
        base = plans.STATIC_CAPACITY["harbour_mission"]
        certificate = F.load("_shared.conditions.certificate")
        e5 = hj.journal_layout("blueboat_v2", 2, 42, 0, 2)       # key (seed 42, env 0, episode E = 2)
        d7_hashes = set()
        reason_lists = []
        for dose in range(5):
            env = F.make_env(tokens=("dynamic=D%d" % dose,))   # pool grid 2 envs x 2 episodes
            env.reset()
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["grid"], self.GRID)
            self.assertEqual(header["grid"], env._d7_pool.grid)
            self.assertEqual(header["per_dose_counts"], [[4, 4, 0]] * 5)   # beyond-grid keys are not pool keys
            # row 0 finishes episodes 0 and 1 early; its next task episode index equals E = 2
            for _ in range(2):
                self._finish_now(env, 0)
                env.step(torch.zeros(2, 2))
                self.assertEqual(env.extras["termination_reason"].tolist()[0], 3)
            self.assertEqual(int(env._scenario.episode_indices()[0]), 2)
            self.assertEqual(int(env._scenario.episode_indices()[1]), 0)
            ident = env._d7_bridge.row_identity(0)
            self.assertEqual(ident["exposure"], "not_admitted")
            self.assertTrue(ident["beyond_offered_grid"])
            self.assertIsNone(ident["geometry_id"])
            self.assertIsNone(ident["stage"])
            self.assertEqual(ident["requested_dose"], dose)
            self.assertEqual(ident["reasons"], ((0, "beyond_offered_grid", 0),))
            self.assertEqual(env._d7_bridge.row_identity(1)["exposure"], "admitted")
            self.assertFalse(env._d7_bridge.row_identity(1)["beyond_offered_grid"])
            # the key's E5 layout (same keyed stream as any key), not a pool record
            self.assertEqual(env._jlayouts[0].geometry_hash, e5.geometry_hash)
            count = int(e5.obstacle_count)
            origin = env.scene.env_origins[0, :2].to(env.obstacle_centers.dtype)
            self.assertTrue(torch.allclose(env.obstacle_centers[0, :count] - origin,
                                           torch.as_tensor(np.array(e5.centers), dtype=env.obstacle_centers.dtype),
                                           atol=1e-3))
            self.assertEqual(int(env.obstacle_active[0].sum()), count)
            self.assertEqual(env._journal_family_scenario_hashes(0),
                             hj.family_scenario_hashes(e5, hj.spawn_pose(e5, 0), 0, env._jdatum))
            # a whole episode from reset to its (shortened) deadline: 60 control steps, D7 slot empty throughout
            env._oracle.deadline_s[0] = 1.0
            record = None
            for _ in range(60):
                env.step(torch.zeros(2, 2))
                codes = env.extras["termination_reason"].tolist()
                if codes[0]:
                    record = env.journal_episode_record(0)
                    contract = env.journal_contract_record(0)
                    break
                self.assertFalse(bool(env._d7_physical[0, base:].any()) or bool(env._d7_visible[0, base:].any()))
                self.assertFalse(env._d7_bridge.contact(0))
                self.assertEqual(env._journal_d7_samples[0], ())
            self.assertIsNotNone(record, dose)
            self.assertEqual(codes[0], 3)
            payload = dynamic.read_stamp(record["dynamic_hazards"])["payload"]
            certificate.verify_dynamic_hazard_stamp(record["dynamic_hazards"])
            assignment, runtime = payload["assignment"], payload["runtime"]
            self.assertEqual(assignment["exposure"], "not_admitted")
            self.assertEqual(assignment["key"], {"eval_seed": 42, "env": 0, "episode": 2,
                                                 "hazard_group": "d7_harbour"})
            self.assertEqual(assignment["task"], "harbour")
            self.assertEqual(assignment["dose"], 0)
            self.assertEqual(assignment["requested_dose"], dose)
            self.assertIsNone(assignment["candidate_digest"])
            self.assertEqual(assignment["intruders"], [])
            self.assertIsNone(assignment["geometry_id"])
            self.assertIsNone(assignment["declared_stage"])
            self.assertEqual(assignment["not_admitted_reasons"],
                             [{"dose": 0, "reason": "beyond_offered_grid", "sampled_states": 0}])
            reason_lists.append(assignment["not_admitted_reasons"])
            self.assertEqual(assignment["not_admitted_failing_reasons"], ["beyond_offered_grid"])
            self.assertTrue(assignment["rule"].startswith("D24"))
            self.assertEqual(runtime["exposure"], "not_admitted")
            self.assertIs(runtime["assigned"], False)                             # D27: never assigned
            self.assertEqual(runtime["itt_count"], 0)
            self.assertEqual(runtime["effective_dose"], 0)
            self.assertEqual(runtime["requested_dose"], dose)
            self.assertEqual(runtime["termination_reason"], "timeout")
            self.assertEqual(runtime["bridge"]["last_tick"], 120)                   # 60 control steps x 2
            self.assertEqual(record["conditions"]["runtime"]["runtime_ticks"], 120)
            self.assertEqual(hpool.stamp_exposure(record["dynamic_hazards"]), "not_admitted")
            # F0 pairing accepts the stamp; the d7_assignment hash is the same at every level
            self.assertEqual(record["scenario_hashes"]["d7_assignment"],
                             pairing.d7_primitive_hash(record["dynamic_hazards"]))
            self.assertEqual(sorted(record["scenario_hashes"]), sorted(env.journal_pairing_inventory))
            d7_hashes.add(record["scenario_hashes"]["d7_assignment"])
            self.assertEqual(contract["d7_exposure"], "not_admitted")
            self.assertTrue(contract["d7_stage_exposure"].startswith("not_admitted (D24: beyond_offered_grid"))
            self.assertIsNone(contract["d7_declared_stage"])
            self.assertEqual(contract["geometry_hash"], e5.geometry_hash)
            self.assertEqual(contract["d7_not_admitted_reasons"], [[0, "beyond_offered_grid", 0]])
            self.assertEqual(contract["d7_requested_dose"], dose)
            # unbounded: the row resets into episode 3, again beyond the grid, without an exception
            self.assertEqual(int(env._scenario.episode_indices()[0]), 3)
            self.assertTrue(env._d7_bridge.row_identity(0)["beyond_offered_grid"])
            self.assertEqual(env._jlayouts[0].geometry_hash, hj.journal_layout("blueboat_v2", 2, 42, 0, 3).geometry_hash)
            run(env, 2)
            self.assertEqual(env.journal_certificate_header()["dynamic_hazards_header"]["per_dose_counts"],
                             [[4, 4, 0]] * 5)
        self.assertEqual(len(d7_hashes), 1)                                   # dose independent
        self.assertEqual(reason_lists, [[{"dose": 0, "reason": "beyond_offered_grid", "sampled_states": 0}]] * 5)

    def test_keys_outside_the_env_range_or_of_another_seed_stay_preflight_errors(self):
        raw, meta = F.small_pool("blueboat_v2", envs=2, episodes=2)
        view = hpool.HarbourPoolView(raw, meta)
        self.assertEqual(view.grid, self.GRID)
        for key in (dynamic.EpisodeKey(42, 1, 2, "d7_harbour"), dynamic.EpisodeKey(42, 0, 57, hpool.MIXIN_GROUP)):
            record = view.assignment(key, 3)
            self.assertIsInstance(record, hpool.NotAdmitted)
            self.assertTrue(record.beyond_grid)
            self.assertEqual(record.reasons, ((0, "beyond_offered_grid", 0),))
            self.assertEqual(record.requested_dose, 3)
        self.assertFalse(view.is_beyond_grid(dynamic.EpisodeKey(42, 1, 1, "d7_harbour")))   # an offered key
        for key in (dynamic.EpisodeKey(42, 2, 0, "d7_harbour"),            # env outside the grid
                    dynamic.EpisodeKey(42, 2, 5, "d7_harbour"),
                    dynamic.EpisodeKey(43, 0, 2, "d7_harbour"),            # another eval seed
                    dynamic.EpisodeKey(43, 0, 0, "d7_harbour"),
                    dynamic.EpisodeKey(42, 0, 2, "d7_docking")):           # another hazard group
            self.assertFalse(view.is_beyond_grid(key))
            with self.assertRaises(KeyError):
                view.assignment(key, 0)
        # env level: the mixin turns those KeyErrors into a clear preflight error at reset
        env = F.make_env(num_envs=3, pool=False, journal_d7_pool=hpool.HarbourPoolView(raw, meta))
        with self.assertRaisesRegex(ValueError, "preflight"):
            env.reset()
        env = F.make_env(seed=43, pool=False, journal_d7_pool=hpool.HarbourPoolView(raw, meta))
        with self.assertRaisesRegex(ValueError, "preflight"):
            env.reset()
        # the offered grid fails closed: recorded grid disagrees, or the offered keys are not a full grid
        with self.assertRaisesRegex(hpool.PoolIntegrityError, "grid"):
            hpool.HarbourPoolView(raw, dict(meta, grid={"envs": 3, "episodes": 2}))
        with self.assertRaises(hpool.PoolIntegrityError):
            hpool.HarbourPoolView(raw, dict(meta, offered_keys=list(meta["offered_keys"])[:-1]))

    def test_header_grid_from_the_offered_grid_file_and_d16_pools(self):
        raw, meta, _info = F.forced_pool("blueboat_v2", envs=2, episodes=2, reject=NotAdmittedD16.REJECT)
        tmp = Path(tempfile.mkdtemp(prefix="f6_pool_d24_", dir=str(Path(__file__).resolve().parent)))
        try:
            path = tmp / hpool.pool_file_name("blueboat_v2", 42)
            hpool.write_pool(path, raw, meta)
            loaded, lmeta = hpool.load_pool(path, dynamic, vessel="blueboat_v2")
            self.assertEqual(lmeta["offered_grid"], {"num_envs": 2, "episodes": 2, "eval_seed": 42,
                                                     "hazard_group": "d7_harbour"})
            env = F.make_env(pool=False, tokens=("dynamic=D3",), journal_d7_pool_path=str(path))
            env.reset()
            header = env.journal_certificate_header()["dynamic_hazards_header"]
            self.assertEqual(header["grid"], self.GRID)
            self.assertEqual(header["grid"], {"num_envs": lmeta["grid"]["envs"], "episodes": lmeta["grid"]["episodes"]})
            self.assertEqual(header["per_dose_counts"], [[4, 2, 2]] * 5)         # unchanged by D24
            self.assertEqual(header["keys_not_admitted"], 2)
            # the D16 rejection of key 1/0 and a beyond-grid key coexist; only the former is a pool key
            view = env._d7_pool
            d16 = view.assignment(dynamic.EpisodeKey(42, 1, 0, "d7_harbour"), 3)
            d24 = view.assignment(dynamic.EpisodeKey(42, 1, 2, "d7_harbour"), 3)
            self.assertFalse(d16.beyond_grid)
            self.assertNotIn("beyond_offered_grid", [r for _d, r, _s in d16.reasons])
            self.assertEqual(d16.geometry_id, hj.journal_layout("blueboat_v2", 2, 42, 1, 0).geometry_hash)
            self.assertTrue(d24.beyond_grid)
            self.assertEqual(d24.reasons, ((0, "beyond_offered_grid", 0),))
            self.assertEqual(env._d7_bridge.row_identity(1)["exposure"], "not_admitted")
            self.assertFalse(env._d7_bridge.row_identity(1)["beyond_offered_grid"])
            # a raw E4 pool (no meta) derives the same grid from its admitted and rejected keys
            self.assertEqual(hpool.HarbourPoolView(raw).grid, self.GRID)

            # a pool object that cannot state its offered grid is refused at construction (fail closed)
            class Bare:
                pool_digest = raw.pool_digest

                def assignment(self, key, dose):
                    return raw.assignment(key, dose)
            with self.assertRaisesRegex(ValueError, "offered grid"):
                F.make_env(pool=False, journal_d7_pool=Bare())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
