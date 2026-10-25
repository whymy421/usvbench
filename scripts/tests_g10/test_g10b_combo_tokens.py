"""G10b: the six combination presets by name, end to end (CPU).

For each of journal.combo.wave_wind.D2/D3, wave_current.D2/D3, wave_wind_local.D2/D3:
* resolution through the path the evaluators use (eval_protocol.apply_journal_suites ->
  journal_config.resolve_from_args, --suite and --suite-file) and run_cell planning (--suite
  pass-through, cell configuration);
* the components expand exactly as presets_v3.expand_presets (same schema-3 parameters as the two
  component tokens, apart from the preset provenance), the combination is stamped by name with its
  role, rule and components, and tokens() re-resolves to the identical stack;
* the components share the scenario keys and the compass-8 stratum (COMBINATION_RULE): the episode
  primitives of a combination run equal those of the single-suite runs, and wind / current flow along
  the mean wave direction;
* a certificate from the REAL writer over the REAL mixin, configured by eval_protocol, carries the name
  in suite_presets (tokens and stamp) and in the condition stamp, round-trips, validates, and its
  green-water formula is the wave component's;
* existing single-suite tokens give byte-identical stamps (digests from the T-base v2.3 + G10 tree).

Run: python -B scripts/tests_g10/test_g10b_combo_tokens.py   (or pytest)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

P = G.shared("conditions.presets_v3")
JC = G.journal_config
PAIRING = G.shared("conditions.pairing")
COMBOS = tuple(P.COMBINATIONS)
LEVEL_TOKEN = {"journal.waves.D2": "waves=D2", "journal.waves.D3": "waves=D3", "journal.waves.B.D2": "waves.B=D2",
               "journal.waves.B.D3": "waves.B=D3", "journal.wind.D2": "wind=D2", "journal.wind.D3": "wind=D3",
               "journal.current.D2": "current=D2", "journal.current.D3": "current=D3"}
# JournalConditions.stamp() digests of single-suite stacks computed on the T-base v2.3 + G10 tree
# (work/apply_check/repo) before G10b; G10b must leave them byte-identical.
SINGLE_SUITE_STAMPS = {
    ("waves=D2",): "3dd1682af527e9ba137c41706c685a0a591bc141e576a70f9b67b2b7a8ae5d3a",
    ("waves.B=D3", "wind=D1"): "18d2b8dd1d6dae6d3dfe2b284f13f58d337aa174541274ca692434d3f941b700",
    ("wind=D3", "current=D2"): "ffdedf7c456a2f1f332cc232e7e4e1bcc79c32b4594523476ea0675c11e5e9a5",
    ("actuator=D2", "perception.bundle=D1"): "1dcbf6700d4b87a0b8a6ce5afaf8fd03c497f3880680fcdbba1be8f0d3563fd2",
    ("journal.waves.D3",): "05bd7b95542353f0bf51056f76476be9bcdd23b2de848a89d4dbc84c120e6c7c",
    ("journal.wind.D2", "current=D1"): "f86e1cdf89556942a2abf81191d8cbf359d36bffec918e454b81554465879877",
    ("journal.waves.supp075",): "9911e67ff164bbb7ea4af6d663402b757121cf49a73a9bfe994992ee81a8d995",
    ("dynamic=D2", "waves=D4"): "dcd7c6414963b3b3185bd19369aba8fa0f4118e97f61b8b622cf9570a024215c",
    ("waves=D1", "wind=D1", "wind.offset_deg=30"): "559dafd78391992554967d89997d83a6becbd0f6ab7c0628ab3803b98c423a1e",
    (): "c2385ae6aa680a3cd33ea1a86cdbb07b45b9b17173bdb54f474bb473ecc7824a",
}


def digest(stamp):
    return hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()


def without_presets(conditions):
    from dataclasses import asdict
    data = asdict(conditions)
    data.pop("presets")
    return data


def _asdict(value):
    from dataclasses import asdict
    return asdict(value)


def evaluator_resolve(tokens, seed=42, suite_file=None, vessel="blueboat_v2"):
    import eval_protocol
    cfg = G.journal_cfg(vessel)
    args = SimpleNamespace(suite=list(tokens), suite_file=suite_file)
    # the evaluator's own function; journal_config comes from the SAME task tree as the env (in a release
    # eval_protocol finds it next to the cfg class; here the glue fixtures' namespace package)
    with patch.object(eval_protocol, "_journal_config_module", lambda _cfg: JC):
        stamp = eval_protocol.apply_journal_suites(cfg, args)
    return cfg, args, stamp


class Resolution(unittest.TestCase):
    def test_each_combination_through_the_evaluator_path(self):
        self.assertEqual(len(COMBOS), 6)
        for name in COMBOS:
            components = P.COMBINATIONS[name]
            cfg, args, stamp = evaluator_resolve([name])
            resolved = cfg.journal_conditions
            # expands exactly as expand_presets: same parameters as the component level tokens
            split = JC.resolve_journal_conditions([LEVEL_TOKEN[c] for c in components], seed=42)
            self.assertEqual(without_presets(resolved.conditions), without_presets(split.conditions), name)
            self.assertEqual(resolved.conditions.presets, (name,))
            self.assertEqual([c for c, _ in P.expand_presets(resolved.conditions.presets)], list(components))
            # stamped by name: tokens, presets, provenance (role, rule, components)
            self.assertEqual(args.journal_suites["tokens"], [name])
            self.assertEqual(stamp["tokens"], [name])
            self.assertEqual(stamp["presets"], [name])
            block = stamp["combinations"]
            self.assertEqual(block, [P.provenance(name)])
            self.assertEqual(block[0]["role"], P.COMBINATION_METADATA[name][0])
            self.assertEqual(block[0]["rule"], P.COMBINATION_RULE)
            self.assertEqual([c["preset"] for c in block[0]["components"]], list(components))
            # tokens() round trip, CLI and file forms
            again = JC.resolve_journal_conditions(resolved.tokens(), seed=42)
            self.assertEqual(again.stamp(), resolved.stamp())
            self.assertEqual(_asdict(again.conditions), _asdict(resolved.conditions))
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "suite.json")
                with open(path, "w", encoding="ascii") as handle:
                    json.dump({"suite": [name], "seed": 42}, handle)
                cfg_file, _, stamp_file = evaluator_resolve([], suite_file=path)
            self.assertEqual(stamp_file, stamp)
            self.assertEqual(_asdict(cfg_file.conditions), _asdict(cfg.conditions))

    def test_combination_with_other_suites_and_conflicts(self):
        resolved = JC.resolve_journal_conditions(["journal.combo.wave_current.D3", "actuator.cap=D1"], seed=5)
        self.assertEqual(resolved.tokens(), ["actuator.cap=D1", "journal.combo.wave_current.D3"])
        self.assertEqual(JC.resolve_journal_conditions(resolved.tokens(), seed=5).stamp(), resolved.stamp())
        same = JC.resolve_journal_conditions(["journal.combo.wave_wind.D2", "waves=D2"], seed=5)
        self.assertEqual(JC.resolve_journal_conditions(same.tokens(), seed=5).stamp(), same.stamp())
        for clash in (["journal.combo.wave_wind.D2", "waves=D3"], ["journal.combo.wave_wind.D2", "wind=D1"],
                      ["journal.combo.wave_wind.D2", "journal.combo.wave_current.D3"]):
            with self.assertRaises(JC.SuiteConflict) as caught:
                JC.resolve_journal_conditions(clash, seed=5)
            self.assertEqual(caught.exception.code, "C1", clash)

    def test_run_cell_planning(self):
        import test_journal_evaluators as T
        for name in COMBOS:
            with patch.dict("os.environ", {}, clear=True):
                plan = T.run_cell.resolve_cell(T._cell("--suite", name, controller="mppi"))
            command = plan["command"]
            self.assertEqual(command[command.index("--suite"):command.index("--suite") + 2], ["--suite", name])
            self.assertEqual(plan["configuration"]["journal"]["suite"], [name])

    def test_single_suite_stamps_are_unchanged(self):
        for tokens, expected in SINGLE_SUITE_STAMPS.items():
            resolved = JC.resolve_journal_conditions(list(tokens), seed=42)
            self.assertNotIn("combinations", resolved.stamp())
            self.assertEqual(digest(resolved.stamp()), expected, tokens)


def episode(tokens, vessel="blueboat_v2"):
    cfg = G.journal_cfg(vessel, tokens=tokens)
    env = G.FakeJournalEnv(cfg, 2, deadline_s=0.5)
    env.reset()
    records = {}
    while len(records) < 2:
        _, terminated, truncated = env.step(torch.tensor([[0.6, 0.4], [0.3, -0.2]]))
        for i in (terminated | truncated).nonzero().flatten().tolist():
            records.setdefault(i, env.journal_episode_record(i))
    return env, records


class SharedScenario(unittest.TestCase):
    def test_components_share_keys_and_stratum(self):
        for name in ("journal.combo.wave_wind.D2", "journal.combo.wave_current.D3",
                     "journal.combo.wave_wind_local.D3"):
            wave, partner = P.COMBINATIONS[name]
            suite = P.JOURNAL_PRESETS[partner]["suite"]
            _, combo = episode([name])
            _, alone_wave = episode([LEVEL_TOKEN[wave]])
            _, alone_partner = episode([LEVEL_TOKEN[partner]])
            for i in (0, 1):
                hashes = combo[i]["scenario_hashes"]
                self.assertEqual(hashes["waves"], alone_wave[i]["scenario_hashes"]["waves"], name)
                self.assertEqual(hashes[suite], alone_partner[i]["scenario_hashes"][suite], name)
                self.assertEqual(hashes["direction_stratum"], alone_wave[i]["scenario_hashes"]["direction_stratum"])
                runtime = combo[i]["conditions"]["runtime"]
                self.assertAlmostEqual(runtime[suite]["world_flow_to_rad"], runtime["waves"]["mean_direction_rad"],
                                       places=5)


class Certificate(unittest.TestCase):
    def test_certificate_from_the_evaluator_path(self):
        import cert_v3
        import test_journal_evaluators as T
        import validate_cert_v3
        for name, vessel in (("journal.combo.wave_wind.D3", "blueboat_v2"),
                             ("journal.combo.wave_wind_local.D3", "wamv_vrx_v2_opt1")):
            cfg, args, stamp = evaluator_resolve([name], vessel=vessel)
            pool = T.PoolView(G.d7_pool(vessel, 2, seed=42))
            env = G.FakeJournalEnv(cfg, 2, deadline_s=0.5, pool=pool)
            env.dynamic_hazard_header["grid"] = dict(T.GRID)
            env.journal_d7_pool_manifest_sha256 = T.MANIFEST_SHA
            provenance = {k: "a" * 64 for k in ("tasks_sha256", "scripts_sha256", "evaluator_sha256")}
            provenance.update(tasks_root="t/tasks", scripts_root="t/scripts", environment_source="t/tasks/x/env.py",
                              evaluator_path="t/scripts/eval.py", hash_algorithm="sha256-path-nul-content-v1")
            provenance["tree_sha256"] = cert_v3.legacy_tree_digest(provenance["tasks_sha256"],
                                                                   provenance["scripts_sha256"])
            runtime = {k: "test" for k in ("hostname", "gpu_name", "gpu_device", "driver", "isaac_sim_version",
                                           "isaac_lab_version", "physx_version")}
            ticks = itertools.count()
            task = T.TASK_ID if vessel == "blueboat_v2" else T.TASK_ID.replace("BlueBoat", "WAMVVRX")
            run = cert_v3.CertificateRun(
                base=env, env_cfg=cfg, evaluator=T.__file__, task=task, controller="test", controller_config={},
                checkpoint=None, eval_seed=42, level=0, episodes_requested=2, overrides={},
                argv=["eval.py", "--suite", name], cli_args={"suite": [name], "journal_suites": args.journal_suites},
                provenance=provenance, runtime=runtime, synchronize=lambda: None,
                clock=lambda: next(ticks) / 1000.0)
            env.reset()
            run.after_reset()
            records, done = [], set()
            while len(done) < 2:
                with run.time_controller():
                    action = torch.tensor([[0.6, 0.1], [0.5, -0.1]])
                with run.time_environment():
                    _, terminated, truncated = env.step(action)
                run.after_step()
                for i in (terminated | truncated).nonzero().flatten().tolist():
                    if i in done:
                        run.discard_episode(i)
                        continue
                    records.append(run.finish_episode(i, dict(env=i, ep=0, success=False,
                                                              terminated=bool(terminated[i]),
                                                              truncated=bool(truncated[i]), tts_s=None,
                                                              path_length_m=1.0)))
                    done.add(i)
            doc = run.build({"records": records})
            self.assertEqual(validate_cert_v3.validate_certificate(doc, num_envs=2, episodes_per_env=1), [])
            header = doc["header"]
            presets = header["suite_presets"]
            self.assertEqual(presets["suite"], [name])
            self.assertEqual(presets["journal_conditions"]["tokens"], [name])
            self.assertEqual(presets["journal_conditions"]["combinations"][0]["preset"], name)
            self.assertEqual(header["conditions"]["parameters"]["presets"], [name])
            # round trip from the certificate alone
            again = JC.resolve_journal_conditions(presets["journal_conditions"]["tokens"], seed=header["eval_seed"])
            self.assertEqual(json.loads(json.dumps(again.stamp())), presets["journal_conditions"])
            self.assertEqual(presets["journal_conditions"]["conditions_parameters_sha256"],
                             header["conditions"]["parameters_sha256"])
            # green-water formula of the wave component
            green = header["green_water"]
            wave = P.COMBINATIONS[name][0]
            self.assertEqual(green["formula"]["preset"], wave)
            self.assertEqual(green["formula"]["preset_role"], "MAIN")
            self.assertEqual(green["presets"], [name])
            self.assertEqual(green["formula"]["inputs"]["hs_m"], P.JOURNAL_PRESETS[wave]["values"]["waves"]["hs_m"])
            # G28: the WAM-V probe is the deck edge (v2.5 pontoon top: 0.0638970551635, formula-excluded)
            expected = {"blueboat_v2": 1.82394684115e-13, "wamv_vrx_v2_opt1": 1.45046054327e-50}[vessel]
            self.assertAlmostEqual(green["formula"]["probability"] / expected, 1.0, places=9)
            self.assertNotIn("formula_probability_exceeds_0.05", green["cell"]["reasons"])
            if vessel == "blueboat_v2":
                self.assertFalse(green["cell"]["excluded_green_water"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
