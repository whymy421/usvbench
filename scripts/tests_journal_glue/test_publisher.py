"""F0 / CONTEXT_R4 D9: publish_termination_codes (validation, lifetime, codes 6-9 survive).

CPU only; run: python -B test_publisher.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

T = G.termination


def env():
    return SimpleNamespace(extras={})


class Publisher(unittest.TestCase):
    def test_codes_are_validated_against_the_e5_table(self):
        table = {row.code: row for row in G.contracts.TERMINATION_TABLE}
        self.assertEqual(sorted(T.termination_table()), sorted(table))
        self.assertEqual(sorted(table), list(range(10)))
        for bad in (torch.tensor([0, 10]), torch.tensor([-1, 0]), torch.tensor([0.0, 1.0]),
                    torch.tensor([[0, 1]]), torch.tensor([True, False]), torch.tensor([0, 1], dtype=torch.uint8)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    T.publish_termination_codes(env(), bad)
        with self.assertRaises(TypeError):
            T.publish_termination_codes(env(), [0, 1])

    def test_validity_codes_survive_and_names_follow_the_table(self):
        e = env()
        codes = torch.tensor([6, 7, 8, 9, 0, 1, 2, 3, 4, 5])
        T.publish_termination_codes(e, codes)
        self.assertEqual(e.termination_reason.tolist(), codes.tolist())
        self.assertEqual([T.termination_code_name(int(c)) for c in codes[:4]],
                         ["capsize", "sinking", "airborne", "non_finite_state"])
        terminated, truncated = T.termination_masks_from_codes(codes)
        self.assertEqual(terminated.tolist(), [True] * 4 + [False, True, True, False, True, True])
        self.assertEqual(truncated.tolist(), [False] * 7 + [True, False, False])

    def test_lifetime_matches_publish_termination_reason(self):
        e = env()
        codes = torch.tensor([0, 6, 3])
        out = T.publish_termination_codes(e, codes, terminated=torch.tensor([False, True, False]),
                                          truncated=torch.tensor([False, False, True]))
        self.assertIs(out, e.termination_reason)
        self.assertIsNot(e.termination_reason, codes)                   # detached from the caller
        self.assertIsNot(e.extras["termination_reason"], e.termination_reason)
        codes[1] = 1
        self.assertEqual(e.termination_reason.tolist(), [0, 6, 3])      # input mutation does not alias
        e.termination_reason[0] = 5
        self.assertEqual(e.extras["termination_reason"].tolist(), [0, 6, 3])
        T.publish_termination_codes(e, torch.tensor([0, 0, 0]))        # the next _get_dones replaces it
        self.assertEqual(e.extras["termination_reason"].tolist(), [0, 0, 0])
        # the legacy mask publisher keeps its behaviour (no D4 codes)
        legacy = env()
        T.publish_termination_reason(legacy, torch.tensor([True, False]), torch.tensor([False, True]),
                                     success=torch.tensor([True, False]))
        self.assertEqual(legacy.termination_reason.tolist(), [1, 3])

    def test_done_masks_must_agree_with_the_codes(self):
        cases = ((torch.tensor([0]), [True], [False]), (torch.tensor([6]), [False], [False]),
                 (torch.tensor([3]), [True], [True]), (torch.tensor([3]), [False], [False]))
        for codes, term, trunc in cases:
            with self.subTest(codes=codes.tolist(), term=term, trunc=trunc):
                with self.assertRaises(ValueError):
                    T.publish_termination_codes(env(), codes, terminated=torch.tensor(term),
                                                truncated=torch.tensor(trunc))

    def test_numpy_codes_and_path_load_without_the_package(self):
        e = env()
        T.publish_termination_codes(e, np.array([0, 9], dtype=np.int64))
        self.assertEqual(e.extras["termination_reason"].tolist(), [0, 9])
        spec = importlib.util.spec_from_file_location("_f0_termination_by_path",
                                                      G.SHARED / "termination_reason.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        e = env()
        module.publish_termination_codes(e, torch.tensor([8]))
        self.assertEqual(module.termination_code_name(8), "airborne")

    def test_mixin_publishes_the_contract_codes(self):
        e = G.run(G.journal_cfg(), 3, deadline_s=3 / 60)
        self.assertEqual(e.termination_reason.tolist(), [3, 3])
        self.assertEqual(e.extras["termination_reason"].tolist(), [3, 3])
        e.step(torch.zeros(2, 2))                                        # running rows publish 0
        self.assertEqual(e.extras["termination_reason"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
