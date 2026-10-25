"""F0 mixin tests: call order, zero-dose identity, all suites on, estimate-once, D7/D8/D9 glue.

Every test drives the REAL HydroV2Runtime, the REAL schema-3 condition layer
and the REAL E4 HazardBridge through glue_fixtures.FakeJournalEnv (Isaac Lab
call order, toy rigid-body engine). CPU only; run: python -B test_mixin_glue.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

ALL_ON = ("waves=D2", "wind=D2", "current=D2", "actuator=D2", "perception=D2", "perception.bundle=D2",
          "dynamic=D2", "wind.offset_deg=30", "current.offset_deg=-45", "perception.offset_deg=90")
ALL_D0 = ("waves=D0", "wind=D0", "current=D0", "actuator=D0", "perception=D0", "perception.bundle=D0",
          "dynamic=D0")


def record_calls(obj, names, log, prefix):
    """Wrap bound methods of ``obj`` so every call is appended to ``log``."""
    for name in names:
        original = getattr(obj, name)

        def wrapper(*args, _original=original, _name=name, **kwargs):
            log.append(prefix + "." + _name)
            return _original(*args, **kwargs)
        setattr(obj, name, wrapper)


class CallOrder(unittest.TestCase):
    def test_isaac_order_and_inner_order_through_the_real_stack(self):
        trace = []
        env = G.FakeJournalEnv(G.journal_cfg(tokens=ALL_ON), 2, trace=trace, deadline_s=2 / 60)
        inner = []
        record_calls(env._hydro_v2, ("reset", "step_from_root_link"), inner, "runtime")
        record_calls(env._conditions, ("reset_runtime", "configure_runtime", "physics_inputs", "episode_stamp",
                                       "estimate_position", "estimate_velocity", "estimate_rays"), inner,
                     "conditions")
        record_calls(env._d7_bridge, ("reset_rows", "step", "finish"), inner, "bridge")
        original_observe = env._journal_observe_physics

        def observe(*args):
            inner.append("family.observe_physics")
            return original_observe(*args)
        env._journal_observe_physics = observe
        env.reset()
        reset_part = inner[:]
        self.assertEqual(reset_part, ["runtime.reset", "conditions.reset_runtime", "conditions.configure_runtime",
                                      "bridge.reset_rows", "bridge.step", "conditions.estimate_position",
                                      "conditions.estimate_velocity", "conditions.estimate_rays"])
        inner.clear()
        env.step(torch.zeros(2, 2))
        substep = ["family.observe_physics", "bridge.step", "conditions.physics_inputs", "runtime.step_from_root_link"]
        self.assertEqual(inner[:8], substep * 2)
        self.assertEqual(inner[8], "bridge.step")                     # journal_pre_dones: final substep
        self.assertEqual(inner[9:], ["conditions.estimate_position", "conditions.estimate_velocity",
                                     "conditions.estimate_rays"])
        inner.clear()
        env.step(torch.zeros(2, 2))                                   # deadline: both rows finish
        dones = inner[inner.index("bridge.step", 8):]
        # latch (stamp + finish) for every done row BEFORE the reset of that row
        first_reset = dones.index("runtime.reset")
        self.assertEqual(dones[1:first_reset], ["conditions.episode_stamp", "bridge.finish"] * 2)
        self.assertEqual(dones[first_reset:first_reset + 5], ["runtime.reset", "conditions.reset_runtime",
                                                              "conditions.configure_runtime", "bridge.reset_rows",
                                                              "bridge.step"])
        self.assertEqual(trace[:4], ["setup", "reset", "estimates", "pre_physics"])
        self.assertEqual(trace[4:9], ["substep", "substep", "pre_dones", "dones", "estimates"])
        self.assertEqual(trace[-6:], ["substep", "substep", "pre_dones", "dones", "reset", "estimates"])

    def test_substep_count_and_pre_physics_are_enforced(self):
        env = G.FakeJournalEnv(G.journal_cfg(), 2)
        env.reset()
        with self.assertRaises(RuntimeError):
            env.journal_substep(env.robot.data, env.scene.env_origins, G.WATER_Z)   # no pre_physics yet
        env.journal_pre_physics(torch.zeros(2, 2))
        env.journal_substep(env.robot.data, env.scene.env_origins, G.WATER_Z)
        with self.assertRaises(RuntimeError):
            env.journal_pre_dones(env.robot.data, env.scene.env_origins, G.WATER_Z)  # one substep short
        env.journal_substep(env.robot.data, env.scene.env_origins, G.WATER_Z)
        with self.assertRaises(RuntimeError):
            env.journal_substep(env.robot.data, env.scene.env_origins, G.WATER_Z)   # beyond decimation

    def test_actions_are_clamped_once_and_forces_have_body_shape(self):
        env = G.FakeJournalEnv(G.journal_cfg(), 2)
        env.reset()
        seen = []
        original = env._hydro_v2.step_from_root_link

        def spy(actions, *args, **kwargs):
            seen.append(actions.clone())
            return original(actions, *args, **kwargs)
        env._hydro_v2.step_from_root_link = spy
        clamped = env.journal_pre_physics(torch.tensor([[3.0, -7.0], [0.5, -0.25]]))
        self.assertTrue(torch.equal(clamped, torch.tensor([[1.0, -1.0], [0.5, -0.25]])))
        forces, torques = env.journal_substep(env.robot.data, env.scene.env_origins, G.WATER_Z)
        self.assertEqual(tuple(forces.shape), (2, 1, 3))
        self.assertEqual(tuple(torques.shape), (2, 1, 3))
        self.assertTrue(torch.equal(seen[0], clamped))


class ZeroDose(unittest.TestCase):
    def test_every_suite_at_d0_is_bit_identical_to_no_suites(self):
        def actions(k):
            return torch.tensor([[0.8 * ((k % 7) / 7.0) - 0.2, 0.5], [-0.3, 0.9 - 0.1 * (k % 5)]])
        runs = []
        for tokens in ((), ALL_D0):
            env = G.run(G.journal_cfg(tokens=tokens), 12, actions=actions)
            runs.append(env)
        a, b = runs
        self.assertEqual(len(a.robot.applied), len(b.robot.applied))
        for (fa, ta), (fb, tb) in zip(a.robot.applied, b.robot.applied):
            self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
        self.assertTrue(torch.equal(a.robot.data.root_link_pose_w, b.robot.data.root_link_pose_w))
        self.assertTrue(torch.equal(a.robot.data.root_com_vel_w, b.robot.data.root_com_vel_w))
        for oa, ob in zip(a.observations, b.observations):
            self.assertTrue(torch.equal(oa, ob))
        self.assertTrue(any(torch.count_nonzero(f) for f, _ in a.robot.applied))


class AllSuitesOn(unittest.TestCase):
    def test_every_suite_at_once_runs_three_control_steps_and_stamps(self):
        dynamic = G.dynamic
        env = G.run(G.journal_cfg(tokens=ALL_ON), 3, deadline_s=3 / 60)
        self.assertEqual(env.termination_reason.tolist(), [3, 3])       # timeout at the 3-step deadline
        self.assertTrue(torch.equal(env.extras["termination_reason"], env.termination_reason))
        for i in range(2):
            record = env.journal_episode_record(i)
            stamp = record["conditions"]
            self.assertEqual(stamp["schema"], "usvbench.conditions/3")
            self.assertEqual(stamp["parameters_sha256"], G.conditions.parameters_digest(stamp["parameters"]))
            p, rt = stamp["parameters"], stamp["runtime"]
            for suite in ("waves", "current", "wind", "actuator", "perception"):
                self.assertTrue(p[suite]["enabled"], suite)
            for key in ("waves", "wind", "current", "actuator", "perception", "direction_offsets_rad",
                        "estimates"):
                self.assertIn(key, rt)
            self.assertEqual(rt["runtime_ticks"], 6)
            self.assertEqual(rt["estimates"]["counts"], {"pos": 3, "vel": 3, "ray": 3})
            self.assertNotIn("band_hz", rt["waves"])
            self.assertIn("component_frequency_range_hz", rt["waves"])
            self.assertIn("band_edges_hz", rt["waves"])
            self.assertAlmostEqual(rt["direction_offsets_rad"]["wind"], 0.5235987755982988)
            dynamic.read_stamp(record["dynamic_hazards"])
            payload = record["dynamic_hazards"]["payload"]
            self.assertEqual(payload["assignment"]["dose"], 2)
            self.assertEqual(payload["runtime"]["termination_reason"], "timeout")
            self.assertEqual(payload["runtime"]["bridge"]["last_tick"], 6)
            self.assertEqual(sorted(record["scenario_hashes"]), sorted(env.journal_pairing_inventory))
            self.assertEqual(G.pairing.condition_primitive_hashes(stamp),
                             {k: v for k, v in record["scenario_hashes"].items()
                              if k in G.pairing.CONDITION_PRIMITIVES})
        header = env.journal_certificate_header()
        self.assertEqual(header["dynamic_hazards_header"]["pool_digest"], env._d7_pool.pool_digest)
        self.assertEqual(header["pairing"]["changed_primitives"], [])
        self.assertEqual(header["conditions"]["parameters_sha256"], stamp["parameters_sha256"])

    def test_wamv_all_suites_on(self):
        env = G.run(G.journal_cfg("wamv_vrx_v2_opt1", tokens=ALL_ON), 3, deadline_s=3 / 60)
        self.assertEqual(env.termination_reason.tolist(), [3, 3])
        stamp = env.journal_episode_record(1)["conditions"]
        self.assertEqual(stamp["context"]["vehicle"], "wamv_vrx_v2_opt1")
        self.assertEqual(stamp["runtime"]["estimates"]["counts"]["ray"], 3)


class EstimateOnce(unittest.TestCase):
    def fresh(self, tokens=("perception=D2",)):
        env = G.FakeJournalEnv(G.journal_cfg(tokens=tokens), 2, deadline_s=5.0)
        env.reset()                                   # one estimate per group at tick 0
        return env

    def test_second_estimate_in_the_same_control_step_raises(self):
        env = self.fresh()
        with self.assertRaisesRegex(RuntimeError, "exactly once per control step"):
            env._get_observations()

    def test_a_skipped_control_step_raises_at_the_next_estimate(self):
        env = self.fresh()
        env.journal_pre_physics(torch.zeros(2, 2))
        for _ in range(2):
            env.robot.set_external_force_and_torque(*env.journal_substep(env.robot.data, env.scene.env_origins, 0.0))
            env.robot.integrate(G.DT)
        env.journal_pre_physics(torch.zeros(2, 2))    # observations of the step were skipped
        for _ in range(2):
            env.robot.set_external_force_and_torque(*env.journal_substep(env.robot.data, env.scene.env_origins, 0.0))
            env.robot.integrate(G.DT)
        with self.assertRaisesRegex(RuntimeError, "exactly once per control step"):
            env._get_observations()

    def test_episode_stamp_rejects_missing_estimates(self):
        env = self.fresh(())
        for _ in range(2):
            env.journal_pre_physics(torch.zeros(2, 2))
            for _ in range(2):
                env.robot.set_external_force_and_torque(*env.journal_substep(env.robot.data, env.scene.env_origins, 0.0))
                env.robot.integrate(G.DT)
        env.journal_pre_dones(env.robot.data, env.scene.env_origins, 0.0)
        with self.assertRaisesRegex(RuntimeError, "estimates x decimation must equal runtime ticks"):
            env.journal_dones(torch.tensor([3, 0]))

    def test_task_relative_entry_point_is_refused_on_journal_contracts(self):
        env = self.fresh()
        with self.assertRaisesRegex(RuntimeError, "own-state estimate"):
            env._conditions.observation_out("pos", torch.zeros(2, 2))

    def test_estimates_are_the_same_object_for_every_controller(self):
        env = self.fresh(("perception=D3",))
        obs = env.step(torch.zeros(2, 2))[0]
        est = env.journal_estimate
        self.assertTrue(torch.equal(obs[:, :2], torch.tensor([30.0, 0.0]) - est.position))
        true = env.robot.data.root_link_pose_w[:, :2] - env.scene.env_origins[:, :2]
        self.assertFalse(torch.equal(est.position, true))              # degraded own position


class WaterLevelD8(unittest.TestCase):
    def test_observe_physics_uses_the_previous_runtime_step_elevation(self):
        env = G.FakeJournalEnv(G.journal_cfg(tokens=("waves=D3",)), 2, deadline_s=5.0)
        elevations = []
        original = env._hydro_v2.step_from_root_link

        def spy(*args, **kwargs):
            step = original(*args, **kwargs)
            elevations.append(step.waves.elevation.clone())
            return step
        env._hydro_v2.step_from_root_link = spy
        env.reset()
        for _ in range(3):
            env.step(torch.full((2, 2), 0.2))
        levels = env.observed_water_levels
        self.assertTrue(torch.equal(levels[0], torch.zeros(2)))          # first substep: constant surface
        for k in range(1, len(levels)):
            self.assertTrue(torch.equal(levels[k], G.WATER_Z + elevations[k - 1]), k)
        self.assertTrue(any(bool(torch.count_nonzero(e)) for e in elevations))

    def test_calm_water_is_the_constant_surface(self):
        env = G.run(G.journal_cfg(), 3)
        for level in env.observed_water_levels:
            self.assertTrue(torch.equal(level, torch.zeros(2)))


class DynamicHazardGlue(unittest.TestCase):
    def test_finish_before_reset_tick_clock_and_new_row(self):
        env = G.run(G.journal_cfg(tokens=("dynamic=D2",)), 4, deadline_s=3 / 60)
        stamp = env.episode_dynamic_hazards[0]
        G.dynamic.read_stamp(stamp)
        self.assertEqual(stamp["payload"]["runtime"]["bridge"]["last_tick"], 6)
        self.assertEqual(stamp["payload"]["assignment"]["key"]["episode"], 0)
        row = env._d7_bridge.rows[0]
        self.assertFalse(row["finished"])
        self.assertEqual(int(env._d7_tick[0]), 2)                        # second episode, one control step in
        self.assertEqual(int(env._hydro_v2.ticks[0]), 2)
        self.assertEqual(row["last_tick"], 2)

    def test_manual_reset_mid_episode_is_stamped_aborted_not_as_a_record(self):
        env = G.run(G.journal_cfg(tokens=("dynamic=D1",)), 2, deadline_s=5.0)
        env.reset()
        self.assertEqual(len(env.journal_aborted_dynamic_hazards), 2)
        self.assertEqual({s["payload"]["runtime"]["termination_reason"] for s in env.journal_aborted_dynamic_hazards},
                         {"aborted"})
        self.assertEqual(env.episode_dynamic_hazards, [None, None])

    def test_missing_assignment_is_a_preflight_error(self):
        env = G.FakeJournalEnv(G.journal_cfg(), 2, pool=G.d7_pool("blueboat_v2", 2, episodes=1))
        env.reset()
        with self.assertRaisesRegex(ValueError, "no admitted D7 assignment"):
            env.reset()                                                  # episode 1 is not in the pool


class SetupRequirements(unittest.TestCase):
    def test_fail_closed_preconditions(self):
        pool = G.d7_pool("blueboat_v2", 2)
        cases = []
        cfg = G.journal_cfg()
        cfg.seed = None
        cases.append((cfg, "seed"))
        cfg = G.journal_cfg()
        cfg.robot_cfg.spawn.rigid_props.disable_gravity = True
        cases.append((cfg, "disable_gravity"))
        cfg = G.journal_cfg()
        cfg.sim = type(cfg.sim)(dt=G.DT)
        cases.append((cfg, "gravity"))
        cfg = G.journal_cfg()
        cfg.decimation = 4
        cases.append((cfg, "decimation"))
        cfg = G.journal_cfg()
        cfg.underwater_physics_cfg.hydrodynamics_model = "legacy"
        cases.append((cfg, "hydro_v2"))
        cfg = G.journal_cfg()
        cfg.observation_contract = "v3"
        cases.append((cfg, "journal observation contract"))
        for cfg, text in cases:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, text):
                    G.FakeJournalEnv(cfg, 2, pool=pool)
        with self.assertRaisesRegex(ValueError, "assignment pool"):
            env = object.__new__(G.FakeJournalEnv)
            env.cfg, env.num_envs, env.device, env.extras = G.journal_cfg(), 2, torch.device("cpu"), {}
            env.journal_setup()

    def test_reset_rows_are_isolated(self):
        env = G.run(G.journal_cfg(tokens=("perception=D2", "wind=D2")), 2, deadline_s=5.0)
        before = (int(env._hydro_v2.ticks[0]), env._conditions.estimate_counts(0))
        env.robot.data.root_link_pose_w[1, 0] += 0.0
        env._reset_idx(torch.tensor([1]))
        self.assertEqual((int(env._hydro_v2.ticks[0]), env._conditions.estimate_counts(0)), before)
        self.assertEqual(int(env._hydro_v2.ticks[1]), 0)
        self.assertEqual(env._conditions.estimate_counts(1), {"pos": 0, "vel": 0, "ray": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
