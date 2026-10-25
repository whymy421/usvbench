"""Round T (CONTEXT_R4 D23, D24 (d), D25) and the F5 level-table verdict, F7 side. CPU only.

* D23: the REAL cert_v3 writer refuses to open a journal certificate whose
  dynamic_hazards_header.pool_pinned is False (a missing flag means pinned: F1, F2, F5
  always pin and do not publish it); validate_cert_v3 reports such a certificate as an
  error. Also checked on the REAL family headers in scripts/fixtures_d16 (the family test
  pools given by cfg are unpinned).
* D24 (d) preflight: with the journal header set, the writer reads
  dynamic_hazards_header["grid"] = {"num_envs", "episodes"} and refuses a per-env quota
  above the grid's episodes, a num_envs above the grid's envs, and a header without a
  grid (fail closed). The REAL family headers are checked the same way (today none of the
  five publishes a grid: the family D24 patches add it).
* D24 (d) record time: a counted journal record whose D7 reasons contain
  "beyond_offered_grid" (built in each family's own stamp form from the REAL fixtures) is a
  hard error in the writer and an error in validate_cert_v3.
* D25: every pool-LOCATION cfg attribute the family sources read (getattr or attribute
  access on a cfg) is exactly cfg_override.JOURNAL_RUNTIME_FIELDS (never refused), while
  cfg.journal_d7_pool and the pin field journal_d7_pool_sha256 stay D12-protected.
* F5 CROSS_OWNER_PROPOSAL_f7_docking_level_table: the REAL docking env's level-table stamp
  (fixture extra) goes through cert_v3.docking_level_distance and the validator.

Run: python -B scripts/test_round_t_seams.py   (or pytest)
"""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import torch  # noqa: E402

import cert_v3  # noqa: E402
import cfg_override  # noqa: E402
import d7_exposure  # noqa: E402
import test_d16_real_stamps as real  # noqa: E402
import test_journal_evaluators as tje  # noqa: E402
import validate_cert_v3  # noqa: E402

REPO = Path(_SCRIPTS).parent
FAMILY_DIRS = ("hazard_nav", "path_hazard", "path_following", "station_keeping", "docking", "harbor_mission")


def _raises(exc_type, text, function, *args, **kwargs):
    return tje._raises(exc_type, text, function, *args, **kwargs)


def _writer(header=None, *, episodes_requested=2, not_admitted=(), stamp=None):
    env, cfg = tje.journal_env(("waves=D1",), not_admitted=not_admitted, stamp=stamp, header=header)
    return tje.journal_run(env, cfg, ("waves=D1",), episodes_requested=episodes_requested)


# --------------------------------------------------------------------- D23 --
def test_d23_writer_refuses_an_unpinned_pool_and_the_validator_reports_it():
    _raises(ValueError, "pool_pinned", _writer, {"pool_pinned": False})
    _writer({"pool_pinned": True})                                   # explicitly pinned
    _writer({"pool_pinned": None})                                   # flag absent: pinned (F1, F2, F5)
    doc = tje.journal_document(("waves=D1",))
    assert "pool_pinned" not in doc["header"]["dynamic_hazards_header"] and not tje._check(doc)
    bad = copy.deepcopy(doc)
    bad["header"]["dynamic_hazards_header"]["pool_pinned"] = False
    assert any("pool_pinned" in e and "D23" in e for e in tje._check(bad)), tje._check(bad)
    ok = copy.deepcopy(doc)
    ok["header"]["dynamic_hazards_header"]["pool_pinned"] = True
    assert not tje._check(ok), tje._check(ok)


def test_d23_on_the_real_family_headers():
    for family in tje.D16_FAMILIES:
        header = tje.d16_fixture(family)["dynamic_hazards_header"]
        if header.get("pool_pinned", True) is False:
            _raises(ValueError, "pool_pinned", cert_v3.journal_pool_pinned_check, header)
        else:
            cert_v3.journal_pool_pinned_check(header)
    # the family test pools given by cfg are unpinned (F3: cfg path; F6: in-memory forced pool)
    assert tje.d16_fixture("path_following")["dynamic_hazards_header"]["pool_pinned"] is False


# --------------------------------------------------------------------- D24 --
def test_d24_preflight_reads_the_offered_grid():
    _writer()                                                        # quota 1 <= 4 episodes, 2 <= 2 envs
    _writer(episodes_requested=8)                                    # quota 4 == grid episodes: allowed
    _raises(ValueError, "exceeds", _writer, episodes_requested=10)   # quota 5 > 4
    _raises(ValueError, "exceeds", _writer, {"grid": {"num_envs": 1, "episodes": 4}})
    _raises(ValueError, "grid", _writer, {"grid": None})             # no grid: fail closed
    for bad in ({"num_envs": 2}, {"num_envs": "2", "episodes": 4}, {"num_envs": 2, "episodes": [0, 1, 2, 3]},
                {"envs": 2, "episodes": 4}, {"num_envs": 0, "episodes": 4}, [2, 4]):
        _raises(ValueError, "grid", _writer, {"grid": bad})
    doc = tje.journal_document(("waves=D1",))
    assert doc["header"]["dynamic_hazards_header"]["grid"] == tje.GRID


def test_d24_preflight_on_the_real_family_headers():
    missing = []
    for family in tje.D16_FAMILIES:
        header = tje.d16_fixture(family)["dynamic_hazards_header"]
        if "grid" not in header:
            missing.append(family)
            _raises(ValueError, "grid", cert_v3.journal_grid_check, header, 1, 1)
            continue
        grid = header["grid"]
        assert cert_v3.journal_grid_check(header, grid["num_envs"], grid["episodes"]) == grid
        _raises(ValueError, "exceeds", cert_v3.journal_grid_check, header, grid["num_envs"], grid["episodes"] + 1)
        _raises(ValueError, "exceeds", cert_v3.journal_grid_check, header, grid["num_envs"] + 1, 1)
    print("real family D7 headers without the D24 grid (family D24 patches pending): %s" % missing)


def test_real_family_header_counts_are_readable_except_pending_d22():
    """The writer reads per_dose_counts before D23/D24; D22 (F5 writes triples) is a family patch."""
    unreadable = []
    for family in tje.D16_FAMILIES:
        header = tje.d16_fixture(family)["dynamic_hazards_header"]
        try:
            d7_exposure.per_dose_counts(header)
        except d7_exposure.ExposureError:
            unreadable.append(family)
    assert set(unreadable) <= {"docking"}, unreadable          # F5: E4's two columns until its D22 patch
    print("real family D7 headers whose per_dose_counts the writer cannot read yet (D22 pending): %s"
          % unreadable)


def beyond_grid_stamp(family):
    """The family's REAL not-admitted stamp with the single D24 reason, in the family's own form."""
    def change(payload):
        for path in d7_exposure.REASON_SOURCES:        # every location this family fills
            block = payload
            for name in path[:-1]:
                block = block.get(name) if isinstance(block, dict) else None
            if isinstance(block, dict) and block.get(path[-1]):
                block[path[-1]] = [real.replace_reason(block[path[-1]][0], d7_exposure.BEYOND_OFFERED_GRID)]
    return real.mutated(family, change)


def test_d24_counted_beyond_grid_record_is_a_hard_error():
    _, pairing, _ = real._modules()
    for family in tje.D16_FAMILIES:
        stamp = beyond_grid_stamp(family)
        assert d7_exposure.exposure_of(stamp) == ("not_admitted", (d7_exposure.BEYOND_OFFERED_GRID,)), family
        assert d7_exposure.is_beyond_offered_grid(stamp)
        # writer: the counted record is refused at record time
        _raises(ValueError, "beyond_offered_grid", tje.journal_document, ("waves=D1",), not_admitted=(1,),
                stamp=stamp)
        # validator: a certificate carrying such a record is invalid
        doc = tje.journal_document(("waves=D1",), not_admitted=(1,), stamp=real.stamp_of(family))
        doc["records"][1]["dynamic_hazards"] = stamp
        doc["records"][1]["scenario_hashes"]["d7_assignment"] = pairing.d7_primitive_hash(stamp)
        errors = tje._check(doc)
        assert any("beyond_offered_grid" in e and "D24" in e for e in errors), (family, errors)
        assert sum("beyond_offered_grid" in e for e in errors) == 1, errors


# --------------------------------------------------------------------- D25 --
LOCATION = re.compile(r"pool_(path|dir|file)$")


def _cfg_receiver(node):
    """True for a receiver expression that names a cfg (cfg, self.cfg, env_cfg, ...)."""
    if isinstance(node, ast.Name):
        return "cfg" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return "cfg" in node.attr.lower()
    return False


def family_pool_fields():
    """{field: [file:line, ...]} of every cfg attribute named *pool* in the family sources."""
    found = {}
    for family in FAMILY_DIRS:
        for path in sorted((REPO / "tasks" / family).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                name = None
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in ("getattr", "hasattr", "setattr") and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
                        and _cfg_receiver(node.args[0])):
                    name = node.args[1].value
                elif isinstance(node, ast.Attribute) and _cfg_receiver(node.value):
                    name = node.attr
                if name and "pool" in name:
                    found.setdefault(name, []).append("%s:%d" % (path.relative_to(REPO).as_posix(), node.lineno))
    return found


def test_d25_pool_location_fields_are_runtime_and_pins_stay_protected():
    found = family_pool_fields()
    locations = {name for name in found if LOCATION.search(name)}
    assert locations == set(cfg_override.JOURNAL_RUNTIME_FIELDS), (sorted(locations), found)
    assert set(cfg_override.JOURNAL_RUNTIME_FIELDS) == {"journal_d7_pool_path", "journal_d7_pool_dir",
                                                        "d7_pool_dir", "d7_pool_file"}
    for name in locations:
        assert cfg_override.journal_protected_reason(name) is None, name
        cfg_override.check_journal_assignments([name + "=/somewhere"])
    others = set(found) - locations
    assert {"journal_d7_pool", "journal_d7_pool_sha256"} <= others, sorted(found)
    for name in others:
        assert cfg_override.journal_protected_reason(name) is not None, (name, found[name])
        _raises(cfg_override.OverrideError, "D12", cfg_override.check_journal_assignments, [name + "=x"])
    print("family pool cfg fields: " + ", ".join("%s (%d uses)" % (k, len(v)) for k, v in sorted(found.items())))


# ------------------------------------------------------------- F5 verdict --
def test_f5_level_table_proposal_is_covered_with_the_real_docking_stamp():
    stamp = tje.d16_fixture("docking")["extra"]["docking_level_table"]
    assert stamp["docking_level_table_in_force"] == "journal_plan", stamp
    cfg = SimpleNamespace(eval_level_distances=[4.0], eval_level_distances_v2=[5.0], eval_level=0,
                          docking_level_table_in_force=lambda: "journal_plan")
    got = cert_v3.docking_level_distance(cfg, SimpleNamespace(docking_level_table=stamp), "2")
    assert got["table_field"] == "journal_plan" and got["docking_level_table_in_force"] == "journal_plan"
    header = {"resolved_config": {"eval_level_distances": [4.0], "journal_contract_version": 2},
              "level_distance": got, "journal": {"task": "docking"}, "physics_version": "2", "level": 0}
    assert validate_cert_v3.level_distance_errors(header) == [], validate_cert_v3.level_distance_errors(header)


def run_all():
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                with patch.dict("os.environ", {}, clear=False):
                    function()
                print("PASS " + name, flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                print("FAIL " + failures[-1], flush=True)
    return failures


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(1 if run_all() else 0)
