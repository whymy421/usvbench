"""G18: read-only green-water diagnostics and the optional wave-following spawn (CPU).

Diagnostics (conditions/green_water.py GreenWaterMonitor.episode_diagnostics,
conditions/green_water_rule.py diagnostics_errors / diagnostics_summary):
* the rule/1 record, cell and verdict are unchanged by the diagnostics;
* the settling window: an immersion inside the first SETTLE_S seconds after the
  reset counts for rule/1 but not for the settled detector; immersions after it
  count, with their onset, point and number of events; flicker never counts;
* crossing counters: a regular wave over a fixed hull gives one freeboard and
  one mean-level up-crossing per cycle (crest above the deck edge) or none and
  one (crest below); a Gaussian random sea over a fixed hull gives the Rice
  ratio exp(-f^2 / (2 sigma^2));
* reset-row isolation, no host read in the per-substep path;
* validator side: consistent diagnostics pass, tampered ones are rejected, a
  partial set is rejected, the verdict never moves;
* real writer + validator: a waves certificate carries the per-record
  diagnostics and the header summary and validates.
Wave-following spawn (conditions/wave_spawn.py, cfg.journal_spawn_on_waves):
* T25 (owner decision T1.35a): ON by default - absent is bitwise the explicit True run; False is the
  off path (no root write, no stamp);
* on: reset rows sit on the filtered surface (COM height, attitude, heave rate)
  so the first substep's buoyancy balances the weight, the spawn transient and
  the t = 0 green water disappear; calm water and Hs = 0 are no-ops; the header
  stamps the option only when on.

Run: python -B scripts/tests_g18/test_g18_green_water_diagnostics.py   (or pytest)
"""
from __future__ import annotations

import ast
import copy
import inspect
import math
import os
import sys
import textwrap
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

GW = G.shared("conditions.green_water")
RULE = G.shared("conditions.green_water_rule")
SPAWN = G.shared("conditions.wave_spawn")
HV2 = G.shared("hydro_v2")
DT = 1.0 / 120.0
FORMULA = {"version": "usvbench.green_water.formula/1", "probability": 0.0, "limit": 0.05}
SETTLE_TICKS = round(RULE.SETTLE_S / DT)


def monitor(vessel="blueboat_v2", n=2):
    return GW.GreenWaterMonitor(vessel, n, "cpu", torch.float32, DT, dict(FORMULA))


def level_poses(n, z):
    pose = torch.zeros((n, 7), dtype=torch.float32)
    pose[:, 2] = z
    pose[:, 3] = 1.0
    return pose


def run_pattern(m, pattern, origins=None):
    """pattern: list of bools per substep for env 0 (env 1 always dry): True = hull sunk 1 m."""
    n = m.num_envs
    origins = torch.zeros((n, 3), dtype=torch.float32) if origins is None else origins
    for wet in pattern:
        # dry: hull 5 cm above its calm height (every point clearly below the calm mean-level threshold);
        # wet: sunk 1 m (every point submerged)
        z = torch.full((n,), HV2_EQ_Z + 0.05)
        z[0] = -1.0 if wet else HV2_EQ_Z + 0.05
        pose = level_poses(n, 0.0)
        pose[:, 2] = z
        m.observe(pose, origins, 0.0, torch.zeros(n), None)


def single_wave(n, amplitude, period, direction=0.0):
    omega = 2 * math.pi / period
    k = omega ** 2 / 9.81
    full = lambda v: torch.full((n, 1), v, dtype=torch.float32)  # noqa: E731
    return HV2.SpectralWaves(full(amplitude), full(omega), full(k), full(direction), torch.zeros((n, 1)))


class SettlingWindow(unittest.TestCase):
    def test_rule1_record_is_unchanged_and_early_immersion_is_not_settled(self):
        m = monitor()
        run_pattern(m, [True] * 60 + [False] * 600)          # 0.5 s immersion at the reset, then dry
        rec = m.episode_record(0)
        self.assertEqual(tuple(rec), RULE.RECORD_FIELDS)
        self.assertEqual((rec["green_water"], rec["first_time_s"], rec["longest_run_substeps"]), (True, 0.0, 60))
        diag = m.episode_diagnostics(0)
        self.assertEqual(tuple(diag), RULE.DIAGNOSTIC_FIELDS)
        self.assertFalse(diag["green_water_settled"])
        self.assertEqual((diag["events_settled"], diag["first_time_settled_s"], diag["first_point_settled"]),
                         (0, None, None))
        self.assertAlmostEqual(diag["exposure_settled_s"], 660 * DT - RULE.SETTLE_S, places=9)
        self.assertEqual(RULE.diagnostics_errors(diag, rec, m.labels, DT), [])

    def test_immersions_after_the_window_count_with_onset_and_events(self):
        m = monitor()
        start = SETTLE_TICKS + 120                          # 1 s after the window opens
        pattern = [False] * start + [True] * 10 + [False] * 30 + [True] * 6 + [False] * 30
        pattern += ([True] * 5 + [False]) * 20              # 5-on / 1-off flicker: never an event
        run_pattern(m, pattern)
        rec, diag = m.episode_record(0), m.episode_diagnostics(0)
        self.assertTrue(rec["green_water"] and diag["green_water_settled"])
        self.assertAlmostEqual(diag["first_time_settled_s"], start * DT, places=9)
        self.assertEqual(diag["events_settled"], 2)
        self.assertIn(diag["first_point_settled"], m.labels)
        self.assertEqual(RULE.diagnostics_errors(diag, rec, m.labels, DT), [])
        dry = m.episode_diagnostics(1)
        self.assertEqual((dry["green_water_settled"], dry["events_settled"]), (False, 0))
        self.assertEqual(dry["freeboard_upcrossings"], [0] * len(m.labels))

    def test_run_straddling_the_window_start_counts_only_its_settled_part(self):
        m = monitor()
        pattern = [False] * (SETTLE_TICKS - 3) + [True] * 8 + [False] * 10   # 5 settled substeps: no event
        run_pattern(m, pattern)
        self.assertTrue(m.episode_record(0)["green_water"])
        self.assertFalse(m.episode_diagnostics(0)["green_water_settled"])

    def test_reset_row_isolation(self):
        m = monitor()
        run_pattern(m, [False] * SETTLE_TICKS + [True] * 12)
        m.reset([0])
        d0, d1 = m.episode_diagnostics(0), m.episode_diagnostics(1)
        self.assertEqual((d0["events_settled"], d0["exposure_settled_s"], sum(d0["mean_level_upcrossings"])), (0, 0.0, 0))
        self.assertAlmostEqual(d1["exposure_settled_s"], 12 * DT, places=9)


class Crossings(unittest.TestCase):
    def regular(self, amplitude, freeboard_above):
        m = monitor("blueboat_v2", 1)
        waves = single_wave(1, amplitude, period=2.0)
        origins = torch.zeros((1, 3))
        pose = level_poses(1, float(HV2_EQ_Z))
        cycles = 8
        ticks = SETTLE_TICKS + round(cycles * 2.0 / DT)
        for t in range(ticks):
            m.observe(pose, origins, 0.0, torch.tensor([t * DT]), waves)
        d = m.episode_diagnostics(0)
        return d, cycles

    def test_crest_above_the_deck_edge_crosses_both_levels_once_per_cycle(self):
        d, cycles = self.regular(0.30, True)                # amplitude 0.30 m > freeboard 0.153 m
        self.assertTrue(all(abs(v - cycles) <= 1 for v in d["freeboard_upcrossings"]), d)
        self.assertTrue(all(abs(v - cycles) <= 1 for v in d["mean_level_upcrossings"]), d)

    def test_crest_below_the_deck_edge_crosses_only_the_mean_level(self):
        d, cycles = self.regular(0.10, False)
        self.assertEqual(d["freeboard_upcrossings"], [0] * len(d["freeboard_upcrossings"]))
        self.assertTrue(all(abs(v - cycles) <= 1 for v in d["mean_level_upcrossings"]), d)

    def test_gaussian_sea_gives_the_rice_ratio(self):
        torch.manual_seed(7)
        n, comps = 64, 64
        freqs = torch.linspace(0.25, 0.9, comps)
        sigma = 0.09                                      # local rms elevation, m
        amp = torch.full((n, comps), sigma * math.sqrt(2.0 / comps))
        omega = (2 * math.pi * freqs).expand(n, comps).contiguous()
        k = omega ** 2 / 9.81
        direction = (torch.rand(n, comps) - 0.5) * 1.5
        phase = torch.rand(n, comps) * 2 * math.pi
        waves = HV2.SpectralWaves(amp, omega, k, direction, phase)
        m = monitor("blueboat_v2", n)
        origins = torch.zeros((n, 3))
        pose = level_poses(n, float(HV2_EQ_Z))
        for t in range(SETTLE_TICKS + round(240.0 / DT)):
            m.observe(pose, origins, 0.0, torch.full((n,), t * DT), waves)
        up_f = m.up_freeboard.sum(0).double()
        up_0 = m.up_mean.sum(0).double()
        ratio = up_f / up_0
        expected = torch.exp(-m.calm_freeboard.double() ** 2 / (2 * sigma ** 2))
        rel = (ratio / expected - 1).abs()
        self.assertTrue(bool((rel < 0.2).all()), (ratio.tolist(), expected.tolist()))


class NoHostSync(unittest.TestCase):
    def test_diagnostics_substep_path_has_no_host_reads(self):
        forbidden = {"item", "tolist", "cpu", "numpy"}
        source = textwrap.dedent(inspect.getsource(GW.GreenWaterMonitor._observe_diagnostics))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, forbidden, ast.unparse(node))
                if isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"bool", "int", "float"}, ast.unparse(node))
            self.assertNotIsInstance(node, (ast.If, ast.While))


def diag_record(settled=False, events=0, substeps=600, first=None, point=None, up_f=None, up_0=None, labels=4):
    exposure = max(0.0, substeps * DT - RULE.SETTLE_S)
    return {"version": RULE.DIAGNOSTICS_VERSION, "settle_s": RULE.SETTLE_S, "green_water_settled": settled,
            "first_time_settled_s": first, "first_point_settled": point, "events_settled": events,
            "exposure_settled_s": exposure, "freeboard_upcrossings": up_f or [0] * labels,
            "mean_level_upcrossings": up_0 or [3] * labels}


GREEN = {"green_water": True, "first_time_s": 0.0, "first_point": "a", "submerged_point_substeps": 20,
         "longest_run_substeps": 10, "substeps": 600}
DRY = {"green_water": False, "first_time_s": None, "first_point": None, "submerged_point_substeps": 0,
       "longest_run_substeps": 0, "substeps": 600}
LABELS = ["a", "b", "c", "d"]


class RuleSide(unittest.TestCase):
    def test_summary_arithmetic_and_candidates(self):
        diags = [diag_record(True, 2, first=3.5, point="b", up_f=[0, 2, 0, 0], up_0=[5, 5, 5, 5]),
                 diag_record(False, up_f=[0, 1, 0, 0], up_0=[5, 5, 5, 5])]
        s = RULE.diagnostics_summary(0.001, [GREEN, DRY], diags, LABELS)
        self.assertEqual((s["episodes"], s["episodes_with_green_water_settled"], s["share_settled"]), (2, 1, 0.5))
        self.assertEqual(s["events_settled"], 2)
        self.assertAlmostEqual(s["exposure_settled_s"], 2 * (600 * DT - RULE.SETTLE_S))
        self.assertAlmostEqual(s["events_per_minute_settled"], 60 * 2 / (2 * (600 * DT - RULE.SETTLE_S)))
        self.assertEqual(s["simulated_per_cycle_probability"], [0.0, 0.3, 0.0, 0.0])
        self.assertEqual((s["simulated_per_cycle_worst"], s["simulated_per_cycle_worst_point"]), (0.3, "b"))
        cand = s["candidate_verdicts"]
        self.assertTrue(cand["rule_1_formula_or_share"] and cand["formula_or_settled_share"]
                        and cand["per_cycle_formula_or_simulated"])
        self.assertFalse(s["binding"])
        quiet = RULE.diagnostics_summary(0.001, [DRY, DRY], [diag_record(), diag_record()], LABELS)
        self.assertFalse(quiet["candidate_verdicts"]["per_cycle_formula_or_simulated"])
        self.assertEqual(quiet["events_per_minute_settled"], 0.0)

    def test_record_rejections(self):
        ok = diag_record(True, 1, first=3.2, point="a")
        self.assertEqual(RULE.diagnostics_errors(ok, GREEN, LABELS, DT), [])
        cases = {
            "rule/1 record": (ok, DRY),
            "events_settled >= 1": (dict(ok, events_settled=0), GREEN),
            ">= settle_s": (dict(ok, first_time_settled_s=1.0), GREEN),
            "declared deck-edge point": (dict(ok, first_point_settled="zz"), GREEN),
            "events_settled 0 and null": (diag_record(False, 1), GREEN),
            "version": (dict(ok, version="x"), GREEN),
            "settle_s": (dict(ok, settle_s=1.0), GREEN),
            "per declared point": (dict(ok, freeboard_upcrossings=[0, 0]), GREEN),
            "mean-level up-crossings + 1": (dict(ok, freeboard_upcrossings=[9, 0, 0, 0]), GREEN),
            "exposure_settled_s": (dict(ok, exposure_settled_s=1.0), GREEN),
            "lacks": ({k: v for k, v in ok.items() if k != "events_settled"}, GREEN),
        }
        for text, (diag, green) in cases.items():
            errors = RULE.diagnostics_errors(diag, green, LABELS, DT)
            self.assertTrue(any(text in e for e in errors), (text, errors))


class Certificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_journal_evaluators as T
        cls.T = T
        cls.doc = T.journal_document(("waves=D4",))

    def test_waves_certificate_carries_and_validates_the_diagnostics(self):
        doc = copy.deepcopy(self.doc)
        self.assertEqual(self.T._check(doc), [])
        block = doc["header"]["green_water"]
        self.assertEqual(block["detection"]["diagnostics"]["settle_s"], RULE.SETTLE_S)
        summary = block["diagnostics"]
        self.assertEqual(summary["version"], RULE.DIAGNOSTICS_VERSION)
        self.assertEqual(summary["episodes"], block["cell"]["episodes"])
        self.assertTrue(all("green_water_diagnostics" in r for r in doc["records"]))
        # the rule/1 cell is the G10 cell, key for key
        self.assertEqual(set(block["cell"]), {"rule_version", "episodes", "episodes_with_green_water", "share",
                                              "formula_probability", "excluded_green_water", "reasons"})

    def test_validator_rejects_tampered_or_partial_diagnostics(self):
        doc = copy.deepcopy(self.doc)
        doc["header"]["green_water"]["diagnostics"]["events_settled"] += 1
        self.assertTrue(any("diagnostics.events_settled" in e for e in self.T._check(doc)))
        doc = copy.deepcopy(self.doc)
        del doc["records"][0]["green_water_diagnostics"]
        self.assertTrue(any("every record or on none" in e for e in self.T._check(doc)))
        doc = copy.deepcopy(self.doc)
        del doc["header"]["green_water"]["diagnostics"]
        self.assertTrue(any("diagnostics is missing" in e for e in self.T._check(doc)))
        doc = copy.deepcopy(self.doc)
        doc["records"][1]["green_water_diagnostics"]["version"] = "other"
        self.assertTrue(any("diagnostics.version" in e for e in self.T._check(doc)))

    def test_certificate_without_diagnostics_still_validates(self):
        doc = copy.deepcopy(self.doc)
        del doc["header"]["green_water"]["diagnostics"]
        # T25 (T1.35c): the binding rule/2 verdict is computed from the diagnostics; a G10-era certificate carries
        # neither (removing only the diagnostics would leave an inconsistent verdict, which the validator rejects)
        del doc["header"]["green_water"]["exclusion"]
        for r in doc["records"]:
            del r["green_water_diagnostics"]
        self.assertEqual(self.T._check(doc), [])


# ------------------------------------------------------------- wave spawn --
class SpawnEnv(G.FakeJournalEnv):
    """F0 FakeJournalEnv whose fake engine also accepts a root-state write (Isaac Lab API)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_writes = 0

    def _reset_idx(self, env_ids):
        robot = self.robot

        def write_root_state_to_sim(state, ids):
            ids = torch.as_tensor(ids, dtype=torch.long).flatten()
            robot.data.root_link_pose_w[ids] = state[:, :7].to(torch.float32)
            robot.data.root_com_vel_w[ids] = state[:, 7:].to(torch.float32)
            self.root_writes = getattr(self, "root_writes", 0) + 1
        robot.write_root_state_to_sim = write_root_state_to_sim
        super()._reset_idx(env_ids)


def spawn_run(tokens, flag, steps=12, vessel="blueboat_v2", n=4):
    torch.manual_seed(5)
    extra = {} if flag is None else {"journal_spawn_on_waves": flag}
    cfg = G.journal_cfg(vessel, tokens=tokens, **extra)
    env = SpawnEnv(cfg, n, deadline_s=30.0, pool=G.d7_pool(vessel, n))
    env.reset()
    start = env.robot.data.root_link_pose_w.clone(), env.robot.data.root_com_vel_w.clone()
    for _ in range(steps):
        env.step(torch.zeros((n, 2)))
    return env, start


HV2_EQ_Z = G.shared("hydro_profiles").profile_for_vehicle("blueboat_v2").hydro.equilibrium_com_z_m


class WaveSpawn(unittest.TestCase):
    def test_off_is_bitwise_identical_to_absent(self):
        # T25 (owner decision T1.35a): the option is ON by default, so "absent" is now bitwise the explicit True
        # run and the explicit False run is the (former default) off path: no root write, no stamp
        self.assertIs(SPAWN.JOURNAL_SPAWN_ON_WAVES_DEFAULT, True)
        a, _ = spawn_run(["waves=D4"], None)
        b, _ = spawn_run(["waves=D4"], True)
        self.assertEqual(len(a.robot.applied), len(b.robot.applied))
        for (fa, ta), (fb, tb) in zip(a.robot.applied, b.robot.applied):
            self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
        self.assertTrue(torch.equal(a.robot.data.root_link_pose_w, b.robot.data.root_link_pose_w))
        self.assertIn("spawn", a.journal_certificate_header()["green_water_header"])
        off, _ = spawn_run(["waves=D4"], False)
        self.assertNotIn("spawn", off.journal_certificate_header()["green_water_header"])
        self.assertEqual(off.root_writes, 0)

    def test_on_places_the_hull_on_the_filtered_surface(self):
        env, (pose, vel) = spawn_run(["waves=D4"], True, steps=0)
        waves = env._conditions.wave_source()
        spec = env._hydro_v2.profile.hydro
        yaw = torch.zeros(4)
        sample = waves.sample(pose[:, :2], env.scene.env_origins[:, :2], 0.0, yaw, spec, footprint=True)
        self.assertTrue(torch.allclose(pose[:, 2] - HV2_EQ_Z, sample.elevation, atol=1e-5))
        normal = HV2.surface_normal(sample.slope)
        rot = G.rotation(pose[:, 3:7].double()).float()
        self.assertTrue(torch.allclose(rot[:, :, 2], normal, atol=1e-5))
        self.assertTrue(torch.allclose(vel[:, :2], sample.orbital_velocity_world[:, :2], atol=1e-5))
        self.assertEqual(env.journal_certificate_header()["green_water_header"]["spawn"]["version"],
                         SPAWN.SPAWN_VERSION)
        # first substep, world-vertical applied force: buoyancy balances the weight (no slam; the residual is
        # the heave damping of d eta_f/dt against the depth-attenuated orbital velocity); the calm spawn is
        # off by m w_n^2 eta_f
        def world_z(e, start_pose):
            force = e.robot.applied[0][0][:, 0, :].double()
            return (G.rotation(start_pose.double()[:, 3:7]) @ force.unsqueeze(-1)).squeeze(-1)[:, 2]
        env.step(torch.zeros((4, 2)))
        first_on = world_z(env, pose)
        calm, (calm_pose, _) = spawn_run(["waves=D4"], False, steps=1)
        first_off = world_z(calm, calm_pose)
        weight = float(env.robot.mass) * 9.81
        self.assertLess(float((first_on - weight).abs().max()), 0.1 * weight)
        self.assertGreater(float((first_off - weight).abs().max()), 0.5 * weight)

    def test_on_removes_the_t0_green_water_of_a_calm_spawn(self):
        def onsets(flag):
            env, _ = spawn_run(["waves=D4"], flag, steps=15, n=16)
            m = env._green_water
            return int(((m.first_tick == 0) & m.green).sum())
        self.assertGreater(onsets(False), 0)
        self.assertEqual(onsets(True), 0)

    def test_calm_and_zero_hs_are_no_ops(self):
        for tokens in ([], ["waves=D0"]):
            a, _ = spawn_run(tokens, False, steps=6)
            b, _ = spawn_run(tokens, True, steps=6)
            for (fa, ta), (fb, tb) in zip(a.robot.applied, b.robot.applied):
                self.assertTrue(torch.equal(fa, fb) and torch.equal(ta, tb))
            self.assertEqual(b.root_writes, 0)

    def test_flag_must_be_a_bool(self):
        cfg = G.journal_cfg("blueboat_v2", tokens=["waves=D4"], journal_spawn_on_waves=1)
        with self.assertRaises(ValueError):
            SpawnEnv(cfg, 2, pool=G.d7_pool("blueboat_v2", 2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
