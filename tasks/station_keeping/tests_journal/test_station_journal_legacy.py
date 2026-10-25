"""F4: legacy byte-identity of the station-keeping family vs F-base (CPU, AST only).

Run: python -B tasks/station_keeping/tests_journal/test_station_journal_legacy.py

CONTEXT_R4 D15: every legacy method source of StationKeepingEnv is unchanged
(SHA-256 of the exact source segment, pinned from F-base
D:/claude code/codex_tasks/20260930_fbase/work/repo, MANIFEST-verified);
__init__ differs ONLY by the leading journal branch; the cfg module differs
only by the journal branch at the top of StationKeepingEnvCfg.__post_init__,
the D10 rename of D1's calm class and appended journal helpers. When the
F-base tree is present on this machine the sources are also compared
directly. No module is imported (pure AST), so no Isaac Lab is needed.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
ENV = REPO / "tasks" / "station_keeping" / "station_keeping_env.py"
CFG = REPO / "tasks" / "station_keeping" / "station_keeping_env_cfg.py"
FBASE = Path("D:/claude code/codex_tasks/20260930_fbase/work/repo")

FBASE_METHOD_SHA256 = {
    "_setup_scene": "476cfe46cfb17585c6af5f82599108be3bcee25ddb205f2212b6e8a53cb0fcaf",
    "_create_static_water_mesh": "f8b059366fbaab61e5c5146a2cd34fd12a8ef5dd5aca451f2e425fc60a5c3923",
    "_create_hold_zone_markers": "9c33fa378323dd46d3171158f957f4cf7412eb0a281076f70aaf8a3bd37ab197",
    "_compute_buoyancy_forces": "32661f08a94d8e845322c0b4e682e58acb4c0c3896eb35881b4e06f0b8bfb02d",
    "_resample_current": "c8261dc7ca62becd98a374df610a2a04e24dc9c1b982fed39251b4d58341ea72",
    "_compute_current_forces": "db9313e92056a227aad5644634ac01178368b1f443c5cea31f7ecb2383af0e08",
    "_pre_physics_step": "d5dceeb1fa55c36fbb21f759b7cc689a64c496f9988aab8859af84bc87e4009e",
    "_apply_action": "428c5ed987f6f6fed75cd2841ccc92ad974064008b61d3db02bf8b8ea7949e92",
    "_horizontal_distance": "e57df9f498f04688217407f0f09ed0cdd3dcf4fb81fb18ba207c0d051272305d",
    "_get_observations": "4bd03a30c1ec4319adde0e4434443ba3db429ecde3f80fb4e966345ccc64c042",
    "_get_rewards": "4833b35edb862ff9b662bcfd9d66189601edd951d809c450ae23fb3ad3c0ecea",
    "_get_dones": "cd0a8b969f70494c9d4c3053c4827e3fa7d1611ce816dbf57e0d1ba995b9fa03",
    "_reset_idx": "0ff1d9f6d923f791f2bd0577dc3e953b58c4bae7ea5f8da7bad7bde0fa165d44",
}
FBASE_INIT_DUMP_SHA256 = "6f03d6ea0695b427267f45cb05c50a7701b1a9717226ce313279e7ce44820ac0"
FBASE_CFG_SOURCE_SHA256 = {
    "_build_robot_cfg": "e8da6159a14147ff0bd9c59c4cf068a4b0d014ce0f851eff5f45903140c8b2ed",
    "UnderwaterPhysicsCfg": "8b6c9fd1a3d14add078adcf9a59a266ead35b521375d2483c424a644cc2a030f",
    "_build_underwater_physics_cfg": "2ddd010c24314bd8713e236b97dcdbce02201c32ba91820406d05aa3009d318e",
    "WavePhysicsCfg": "7fcb5e14ead0f098714261db0a9cb613ddaec02b201e2fefea1b90ed94c8e307",
    "VisualCfg": "beb6463127049f937cad997fae8e5b4ae27483bb41d6fe470038fb598ea45160",
    "StationKeepingBlueBoatEnvCfg": "67811493f00b40b9c1d313d2b4eb093c710f217eacd960494ee1ead651c3e2d0",
    "StationKeepingBlueBoatCurrentEnvCfg": "759e22e172006a03d182f97dc42f91f489c6ad342ed962310df01541e41af255",
    "StationKeepingBlueBoatKinEnvCfg": "d59a8fcddad278bde97a99ba302a740ed2f73105e03238918db5ac11f21ec3fc",
    "StationKeepingBlueBoatCurrentKinEnvCfg": "58747e0dcfd3b97937d74a6ce8b86902fc749bd37877d812bd6c55843f398d94",
    "StationKeepingBlueBoatWaveEnvCfg": "8472e1c93a96913d71c2aeffebd0dce5f9f7e4f40a74325eb8230d0dc0dcdd36",
    "StationKeepingBlueBoatRampCurrentEnvCfg": "db97b91b9fed252039445310d768bd62c2f0fe37622e7e64b51dbd4b7fa92079",
}
FBASE_ENVCFG_CLASS_DUMP_SHA256 = "3901f351f7fdd7cf9806c32149891c5b410d647d1e6084e03d1f73a7889ce70e"
FBASE_D1_CLASS_BODY_SHA256 = "038bb11f46d1033437bff916d9a597104077ad2251ac74f11e54b07d29ecdc75"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse(path):
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text)


def class_node(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def methods(text, node):
    return {i.name: i for i in node.body if isinstance(i, ast.FunctionDef)}


def is_journal_branch(statement):
    return (isinstance(statement, ast.If)
            and ast.unparse(statement.test) == "getattr(cfg, 'journal_contract_version', None) == 2"
            or isinstance(statement, ast.If)
            and ast.unparse(statement.test) == "getattr(self, 'journal_contract_version', None) == 2")


class EnvLegacyIdentityTests(unittest.TestCase):
    def setUp(self):
        self.text, tree = parse(ENV)
        self.cls = class_node(tree, "StationKeepingEnv")
        self.methods = methods(self.text, self.cls)

    def test_every_legacy_method_source_is_byte_identical_to_fbase(self):
        for name, digest in FBASE_METHOD_SHA256.items():
            with self.subTest(method=name):
                self.assertEqual(sha(ast.get_source_segment(self.text, self.methods[name])), digest)

    def test_init_differs_only_by_the_leading_journal_branch(self):
        init = self.methods["__init__"]
        self.assertTrue(is_journal_branch(init.body[0]))
        branch = init.body[0]
        self.assertEqual([ast.unparse(s) for s in branch.body],
                         ["self._journal_pre_init(cfg)", "super().__init__(cfg, render_mode, **kwargs)",
                          "self._journal_post_init()", "return"])
        stripped = ast.FunctionDef(name=init.name, args=init.args, body=init.body[1:],
                                   decorator_list=init.decorator_list, returns=init.returns,
                                   type_comment=init.type_comment, type_params=getattr(init, "type_params", []))
        self.assertEqual(sha(ast.dump(stripped)), FBASE_INIT_DUMP_SHA256)

    def test_new_methods_are_journal_only_and_bound_on_the_instance(self):
        legacy = set(FBASE_METHOD_SHA256) | {"__init__"}
        extra = set(self.methods) - legacy
        self.assertTrue(extra)
        for name in extra:
            self.assertTrue(name.startswith("_journal") or name.startswith("journal_"), name)
        bound = ast.literal_eval(next(s for s in self.cls.body if isinstance(s, ast.Assign)
                                      and ast.unparse(s.targets[0]) == "_JOURNAL_BOUND").value)
        self.assertEqual(set(bound), {"_setup_scene", "_pre_physics_step", "_apply_action", "_get_observations",
                                      "_get_rewards", "_get_dones", "_reset_idx"})
        for name in bound:
            self.assertIn("_journal" + name, self.methods)

    def test_mixin_composed_and_legacy_publisher_replaced_only_in_journal_code(self):
        self.assertEqual([ast.unparse(b) for b in self.cls.bases], ["JournalEnvMixin", "DirectRLEnv"])
        journal_src = "\n".join(ast.get_source_segment(self.text, m) for n, m in self.methods.items()
                                if n.startswith("_journal") or n.startswith("journal_"))
        self.assertNotIn("publish_termination_reason", journal_src)
        self.assertIn("self.journal_dones(codes)", journal_src)
        self.assertNotIn("root_com_pos_w", journal_src)          # COM = root-link pose (E5 C5, D6)
        self.assertNotIn("self._hold_steps", journal_src)           # settled clock is a fixed time (D7 S4)
        self.assertNotIn("self._current_ramp", journal_src)         # legacy-only ramp (D6, E5 C3)

    def test_e2_station_proof_keeps_using_the_d3_proof_copy(self):
        # scripts/tests_hydro_v2/test_station_proof.py switches to the in-tree file when it contains
        # this marker and would then compare J-base methods with C2's (E2-owned test): keep it absent.
        self.assertNotIn("_apply_action" + "_hydro_v2", self.text)

    def test_direct_comparison_with_fbase_when_present(self):
        path = FBASE / "tasks" / "station_keeping" / "station_keeping_env.py"
        if not path.is_file():
            self.skipTest("F-base tree not on this machine (pinned digests above still checked)")
        text, tree = parse(path)
        base = methods(text, class_node(tree, "StationKeepingEnv"))
        for name in FBASE_METHOD_SHA256:
            self.assertEqual(ast.get_source_segment(text, base[name]),
                             ast.get_source_segment(self.text, self.methods[name]), name)


class CfgLegacyIdentityTests(unittest.TestCase):
    def setUp(self):
        self.text, self.tree = parse(CFG)

    def test_legacy_cfg_sources_unchanged(self):
        nodes = {n.name: n for n in self.tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        for name, digest in FBASE_CFG_SOURCE_SHA256.items():
            with self.subTest(node=name):
                self.assertEqual(sha(ast.get_source_segment(self.text, nodes[name])), digest)

    def test_envcfg_differs_only_by_the_post_init_journal_branch(self):
        node = class_node(self.tree, "StationKeepingEnvCfg")
        post = next(i for i in node.body if isinstance(i, ast.FunctionDef) and i.name == "__post_init__")
        index = next(k for k, s in enumerate(post.body) if is_journal_branch(s))
        self.assertEqual(index, 2)            # right after the base post-init call
        self.assertEqual([ast.unparse(s) for s in post.body[index].body],
                         ["_journal_station_post_init(self)", "return"])
        post.body = post.body[:index] + post.body[index + 1:]
        self.assertEqual(sha(ast.dump(node)), FBASE_ENVCFG_CLASS_DUMP_SHA256)

    def test_d10_rename_of_the_d1_calm_class(self):
        names = [n.name for n in self.tree.body if isinstance(n, ast.ClassDef)]
        self.assertNotIn("StationKeepingBlueBoatJournalEnvCfg", names)   # E5's template owns that name
        node = class_node(self.tree, "StationKeepingBlueBoatJBaseCalmEnvCfg")
        self.assertEqual([ast.unparse(b) for b in node.bases], ["StationKeepingBlueBoatKinEnvCfg"])
        body = node.body[1:]
        self.assertEqual(sha(ast.dump(ast.Module(body=body, type_ignores=[]))), FBASE_D1_CLASS_BODY_SHA256)
        self.assertIn("not a journal id", ast.get_docstring(node))

    def test_journal_ids_do_not_point_at_family_cfg_classes(self):
        registration = (REPO / "tasks" / "journal" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("station_keeping.station_keeping_env:StationKeepingEnv", registration)
        self.assertNotIn(":StationKeepingBlueBoatJBaseCalmEnvCfg", registration)
        self.assertIn("{_CFG}:StationKeepingBlueBoatJournalEnvCfg", registration)
        self.assertIn("{_CFG}:StationKeepingWAMVJournalEnvCfg", registration)


if __name__ == "__main__":
    unittest.main(verbosity=2)
