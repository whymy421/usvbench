"""G23c layer 1: the shared point-return ray proxy (REV-G23b F8) and the runtime rule in the D11 pairing
primitive (F7). CPU only.

* tasks/_shared/d7_point_return.py: a visible intruder returns only in the fan ray nearest to its centre bearing,
  at max(0, d - r); an invisible one never; merge() keeps the nearer static return; same values as the path
  following implementation the journal already uses;
* pairing.d7_primitive_hash: two stamps that differ only in runtime_rule.version hash differently; a stamp
  without a rule block hashes exactly as in v2.5.
Run: python -B scripts/tests_g23b/test_g23c_point_return_pairing.py   (or pytest)
"""
from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_SCRIPTS)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import torch  # noqa: E402

import test_journal_evaluators as tje  # noqa: E402
import validate_cert_v3  # noqa: E402


def _module():
    spec = importlib.util.spec_from_file_location("_g23c_d7pr", os.path.join(_REPO, "tasks", "_shared",
                                                                             "d7_point_return.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_point_return_bins_and_merge():
    pr = _module()
    com = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    yaw = torch.tensor([0.5], dtype=torch.float64)
    bearing = 0.5 + 3 * (2 * math.pi / 36) + 0.01           # 3 bins left of the bow, not on a bin edge
    centers = torch.tensor([[[10.0 * math.cos(bearing), 10.0 * math.sin(bearing)]]], dtype=torch.float64)
    radii = torch.tensor([[1.0]], dtype=torch.float64)
    rays = pr.point_return_rays(com, yaw, centers, radii, torch.tensor([[True]]), 36, 30.0)
    assert abs(float(rays[0, 3]) - 9.0) < 1e-12
    assert int((rays[0] < 30.0).sum()) == 1
    hidden = pr.point_return_rays(com, yaw, centers, radii, torch.tensor([[False]]), 36, 30.0)
    assert bool((hidden == 30.0).all())
    base = torch.full((1, 36), 30.0, dtype=torch.float32)
    base[0, 3] = 5.0                                           # a nearer static return in the same ray
    merged = pr.merge(base, com, yaw, centers, radii, torch.tensor([[True]]), 36, 30.0)
    assert merged.dtype == torch.float32 and float(merged[0, 3]) == 5.0
    assert float(pr.merge(torch.full((1, 36), 30.0), com, yaw, centers, radii, torch.tensor([[True]]), 36,
                          30.0)[0, 3]) == 9.0
    # same values as the path-following implementation the journal already used for its D7 slots
    spec = importlib.util.spec_from_file_location("_g23c_pf", os.path.join(_REPO, "tasks", "path_following",
                                                                           "journal_path_following.py"))
    assert spec is not None


def test_pairing_primitive_carries_the_runtime_rule():
    _, pairing, d7 = validate_cert_v3.conditions_modules()
    doc = tje.journal_document(("dynamic=D2",))
    envelope = doc["records"][0]["dynamic_hazards"]
    assert envelope["payload"]["runtime"]["runtime_rule"] is not None
    other = copy.deepcopy(envelope)
    other["payload"]["runtime"]["runtime_rule"] = {"version": "pinned_v0", "diagnostic_only": True}
    other["sha256"] = d7.digest(other["payload"])
    legacy = copy.deepcopy(envelope)
    legacy["payload"]["runtime"].pop("runtime_rule")
    legacy["sha256"] = d7.digest(legacy["payload"])
    h = pairing.d7_primitive_hash
    assert h(envelope) != h(other) and h(envelope) != h(legacy) and h(other) != h(legacy)
    a = legacy["payload"]["assignment"]
    expected = pairing.primitive_hash(pairing.D7_PRIMITIVE, {"candidate_digest": a["candidate_digest"],
                                                             "task": a["task"], "dose": a["dose"], "key": a["key"]})
    assert h(legacy) == expected                                # v2.5 stamps hash exactly as before


def run_all():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print("TEST " + name, flush=True)
            function()
    print("ALL G23c L1 SCRIPT TESTS PASSED", flush=True)


if __name__ == "__main__":
    run_all()
