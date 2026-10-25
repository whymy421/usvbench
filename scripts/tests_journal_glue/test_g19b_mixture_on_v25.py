"""G19b (IMPLEMENTATION_PLAN P1 on T-base v2.5): the training mixture with the v2.5 physics defaults. CPU only.

The REAL JournalEnvMixin, HydroV2Runtime, schema-3 condition layer, green-water monitor and E4 bridge through
glue_fixtures.FakeJournalEnv (toy engine, reach contract), the mixture's D1 stack and the mixture attached by
the condition-layer factory from cfg.journal_training_mixture:
(a) spawn on waves (T1.35a, default on): a reset row whose waves gate is OFF spawns bitwise exactly as in a calm
    env (pose and COM velocity) and is never rewritten; a row whose waves gate is ON is rewritten onto the wave
    surface (only those rows reach write_root_state_to_sim);
(b) the green-water monitor (G10 / G18, T25 report points on the WAM-V) reports nothing on gated-off rows:
    every latched record whose training component is not 'waves' has no green water, no submerged point
    substep and no settled event or up-crossing;
(c) diagnostic-only overrides never train: a mixture cfg carrying journal_wind_coefficients (G16) or
    journal_rollpitch_zeta (T25 L1) is refused by the factory; a mixture layer refuses the certificate header.
Run: python -B scripts/tests_journal_glue/test_g19b_mixture_on_v25.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

TM = G.shared("conditions.training_mixture")
SEED = G.SEED
VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")


def mixture(vessel, task="reach"):
    return TM.TrainingMixture(seed=SEED, task=task, vessel=vessel, appearing_wired=False)


def mixture_cfg(vessel, m=None, **extra):
    m = mixture(vessel) if m is None else m
    return G.journal_cfg(vessel, tokens=m.required_tokens(), journal_training_mixture=m, **extra), m


def recording_env(cfg, n, episodes, **kwargs):
    env = G.FakeJournalEnv(cfg, n, pool=G.d7_pool(cfg.vehicle, n, episodes), **kwargs)
    writes = []
    original = env.robot.write_root_state_to_sim

    def record(state, ids):
        writes.append(sorted(int(i) for i in torch.as_tensor(ids).flatten().tolist()))
        return original(state, ids)
    env.robot.write_root_state_to_sim = record
    return env, writes


class SpawnOnWaves(unittest.TestCase):
    N, RESETS = 8, 12

    def test_gated_off_rows_spawn_exactly_as_in_a_calm_env(self):
        for vessel in VESSELS:
            cfg, m = mixture_cfg(vessel)
            mix, mix_writes = recording_env(cfg, self.N, self.RESETS)
            calm, calm_writes = recording_env(G.journal_cfg(vessel), self.N, self.RESETS)
            self.assertIs(mix._conditions.training_mixture, m)
            self.assertTrue(mix._journal_spawn_on_waves and calm._journal_spawn_on_waves)
            ids = torch.arange(self.N)
            seen_on = seen_off = 0
            for r in range(self.RESETS):
                del mix_writes[:]
                mix._reset_idx(ids)
                calm._reset_idx(ids)
                on = [i for i in range(self.N) if mix._conditions.training_component(i) == "waves"]
                for i in range(self.N):
                    self.assertEqual(mix._conditions.training_component(i), m.component(i, r))
                    same_pose = torch.equal(mix.robot.data.root_link_pose_w[i], calm.robot.data.root_link_pose_w[i])
                    same_vel = torch.equal(mix.robot.data.root_com_vel_w[i], calm.robot.data.root_com_vel_w[i])
                    if i in on:
                        seen_on += 1
                        self.assertFalse(same_pose and same_vel, (vessel, r, i))
                    else:
                        seen_off += 1
                        self.assertTrue(same_pose and same_vel, (vessel, r, i, m.component(i, r)))
                self.assertEqual(mix_writes, [on] if on else [], (vessel, r))
            self.assertEqual(calm_writes, [])
            self.assertGreater(seen_on, 0)
            self.assertGreater(seen_off, seen_on)
            print(f"G19b SPAWN {vessel}: {seen_on} waves-gated rows on the surface, {seen_off} gated-off rows "
                  f"bitwise calm", flush=True)

    def test_no_mixture_path_keeps_every_reset_row(self):
        cfg = G.journal_cfg("blueboat_v2", tokens=["waves=D3"])
        env, writes = recording_env(cfg, 4, 2)
        layer = env._conditions
        ids = torch.tensor([2, 0])
        self.assertIs(layer.training_wave_rows(ids), ids)
        env._reset_idx(torch.arange(4))
        self.assertEqual(writes, [[0, 1, 2, 3]])


class GreenWaterSilentOffWaves(unittest.TestCase):
    def test_monitor_reports_nothing_on_gated_off_rows(self):
        for vessel in VESSELS:
            cfg, m = mixture_cfg(vessel)
            n, steps = 6, 70
            env = G.FakeJournalEnv(cfg, n, pool=G.d7_pool(vessel, n, 8), deadline_s=0.4)
            self.assertIsNotNone(env._green_water)
            env.reset()
            records = []
            actions = torch.tensor([[0.6, 0.3]] * n)
            for _ in range(steps):
                _, terminated, truncated = env.step(actions)
                for i in (terminated | truncated).nonzero().flatten().tolist():
                    records.append(env.journal_episode_record(i))
            components = [r["conditions"]["runtime"]["training_mixture"]["component"] for r in records]
            off = [r for r, c in zip(records, components) if c != "waves"]
            self.assertGreater(len(off), 10)
            for record in off:
                gw, diag = record["green_water"], record["green_water_diagnostics"]
                self.assertIs(gw["green_water"], False)
                self.assertEqual(gw["submerged_point_substeps"], 0)
                self.assertEqual(gw["longest_run_substeps"], 0)
                self.assertIs(diag["green_water_settled"], False)
                self.assertEqual(diag["events_settled"], 0)
                self.assertEqual(sum(diag["freeboard_upcrossings"]) + sum(diag["mean_level_upcrossings"]), 0)
                for report in (record.get("green_water_report") or {}).values():
                    self.assertIs(report["record"]["green_water"], False)
                    self.assertEqual(report["record"]["submerged_point_substeps"], 0)
            print(f"G19b GREEN WATER {vessel}: {len(off)} gated-off episodes silent "
                  f"({components.count('waves')} waves episodes)", flush=True)


class DiagnosticOverridesNeverTrain(unittest.TestCase):
    def test_wind_coefficient_and_rollpitch_overrides_are_refused(self):
        m = mixture("blueboat_v2")
        for extra in ({"journal_wind_coefficients": {"c_y0": 0.9}}, {"journal_rollpitch_zeta": 0.1}):
            cfg, _ = mixture_cfg("blueboat_v2", m, **extra)
            with self.assertRaises(TM.MixtureError):
                G.conditions.build_conditions(cfg, 2, "cpu", family="hazard_nav")
            with self.assertRaises(TM.MixtureError):
                G.FakeJournalEnv(cfg, 2)
        layer = G.conditions.build_conditions(mixture_cfg("blueboat_v2", m)[0], 2, "cpu", family="hazard_nav")
        with self.assertRaises(TM.MixtureError):          # attach-level guard of the G16 override
            layer.wind_coefficient_override = {"c_y0": 0.9}
            layer._mixture = None
            layer.attach_training_mixture(m)

    def test_mixture_env_never_produces_a_certificate_header(self):
        cfg, _ = mixture_cfg("blueboat_v2")
        env = G.FakeJournalEnv(cfg, 2, pool=G.d7_pool("blueboat_v2", 2, 4))
        env.reset()
        with self.assertRaises(RuntimeError):
            env.journal_certificate_header()

    def test_smoke_purpose_mixture_attaches(self):
        m = TM.TrainingMixture(seed=SEED, task="reach", vessel="blueboat_v2", appearing_wired=False,
                               purpose="training_smoke")
        cfg, _ = mixture_cfg("blueboat_v2", m)
        layer = G.conditions.build_conditions(cfg, 2, "cpu", family="hazard_nav")
        self.assertEqual(layer.context["training_mixture"]["purpose"], "training_smoke")
        names = m.preset_names()
        self.assertIn("journal.waves.D1", names)
        self.assertIn("dynamic=D1", names)
        stamp = m.stamp()
        self.assertEqual(stamp["presets"]["dynamic"]["values"]["ttc_over_t_avoid"], 4.0)
        self.assertEqual(stamp["preset_names"], names)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
