"""G7 item A (CPU): platform-stable identities.

* Root cause: WAM-V draft_m = sum(point_depths_m) / 4. Python >= 3.12 sums floats with compensated
  (Neumaier) summation, Python 3.11 adds left to right: 0.10916531789468821 vs 0.1091653178946882.
* Registry: every computed value is a declared pin (vehicles_v2_data.PINNED_DERIVED), checked against the
  computed value (rel 1e-9 / abs 1e-12) at import; the profile carries the pin. Under an emulated
  Python 3.11 sum() the registry, every vessel record digest and GEOMETRY_VERSION_DIGEST are identical.
* Identities: geometry_digest quantises every float (1e-6 SI units, versioned), so layout ids, vessel
  record digests and the geometry version digest ignore last-ulp noise and still see real changes.
* Families: F1's reset check, F5's scenario geometry id and F4's rebuilt-candidate check use the same
  quantisation; F4 runs the STORED candidate.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "tasks" / "_shared"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "tasks" not in sys.modules:
    _pkg = ModuleType("tasks")
    _pkg.__path__ = [str(ROOT / "tasks")]
    sys.modules["tasks"] = _pkg
for _folder in (ROOT / "tasks").iterdir():
    _name = "tasks." + _folder.name
    if _folder.is_dir() and _folder.name not in ("_shared", "__pycache__") and _name not in sys.modules:
        _mod = ModuleType(_name)
        _mod.__path__ = [str(_folder)]
        sys.modules[_name] = _mod


def load(name, path, preset=None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(preset or {})
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def left_fold_sum(values, start=0):
    """Python 3.11 builtin sum() on floats: plain left-to-right addition."""
    total = start
    for v in values:
        total = total + v
    return total


class Recorder:
    def __init__(self):
        self.values = {}

    def record_pin(self, name, path, value):
        self.values.setdefault(name, {})[path] = value


def load_registry(tag, *, data_text=None, preset=None, record=False):
    data_path = SHARED / "vehicles_v2_data.py"
    data = ModuleType("g7_reg_data_" + tag)
    data.__file__ = str(data_path)
    exec(compile(data_text if data_text is not None else data_path.read_text(encoding="ascii"),
                 str(data_path), "exec"), data.__dict__)
    recorder = None
    if record:
        recorder = Recorder()
        data.PINNED_DERIVED = recorder
    saved = sys.modules.get("vehicles_v2_data")
    sys.modules["vehicles_v2_data"] = data
    try:
        reg = load("g7_reg_" + tag, SHARED / "vehicles_v2.py", preset)
    finally:
        if saved is None:
            sys.modules.pop("vehicles_v2_data", None)
        else:
            sys.modules["vehicles_v2_data"] = saved
    return reg, data, recorder


def specs(reg):
    """Profiles as plain data (each registry load defines its own dataclass types)."""
    return {name: dataclasses.asdict(spec) for name, spec in reg.VEHICLES_V2.items()}


cfg = load("g7_test_cfg", SHARED / "journal_cfg_v2.py")
L = load("g7_test_layouts", SHARED / "journal_layouts_v2.py")
geo = cfg.geometry


class TestDraftRootCause(unittest.TestCase):
    def test_python311_sum_and_compensated_sum_differ_in_the_last_ulp(self):
        reg, data, _ = load_registry("plain")
        depths = data.WAMV["competition"]["point_depths_m"]
        naive = left_fold_sum(depths) / 4.0
        exact = math.fsum(depths) / 4.0
        self.assertEqual(repr(naive), "0.1091653178946882")          # GPU host value (Python 3.11.15)
        self.assertEqual(repr(exact), "0.10916531789468821")         # correctly rounded mean
        self.assertEqual(math.nextafter(naive, 1.0), exact)          # exactly one ulp apart
        # the pin is the correctly rounded value, and the profile carries it
        self.assertEqual(data.PINNED_DERIVED["wamv_vrx_v2_opt1"]["hydrostatics.draft_m"], exact)
        self.assertEqual(reg.VEHICLES_V2["wamv_vrx_v2_opt1"].hydrostatics.draft_m, exact)
        self.assertEqual(reg.PINS_STATUS, "pinned")

    def test_emulated_python311_registry_gives_identical_records_and_geometry_version(self):
        reg311, _, _ = load_registry("py311", preset={"sum": left_fold_sum})
        reg_now, _, _ = load_registry("now")
        self.assertEqual(specs(reg311), specs(reg_now))
        recs = cfg.build_vessel_records(reg311)
        for key, rec in cfg.REGISTERED_VESSELS.items():
            self.assertEqual(recs[key], rec, key)
            self.assertEqual(recs[key].digest, rec.digest, key)
        contract = {k: recs[k] for k in cfg.VESSELS}
        self.assertEqual(cfg.geometry_version_digest(contract), cfg.GEOMETRY_VERSION_DIGEST)

    def test_unpinned_python311_draft_would_differ_but_the_record_digest_is_quantised(self):
        reg311, _, rec = load_registry("py311_rec", preset={"sum": left_fold_sum}, record=True)
        self.assertEqual(reg311.PINS_STATUS, "recorded")
        self.assertEqual(repr(rec.values["wamv_vrx_v2_opt1"]["hydrostatics.draft_m"]), "0.1091653178946882")
        with self.assertRaises(ValueError):          # journal records never come from an unpinned registry
            cfg.build_vessel_records(reg311)
        shifted = dataclasses.replace(cfg.WAMV, draft_m=0.1091653178946882)
        self.assertEqual(shifted.digest, cfg.WAMV.digest)         # second line of defence: quantisation


class TestRegistryPins(unittest.TestCase):
    @staticmethod
    def literals():
        out = set()
        for path in (SHARED / "vehicles_v2_data.py", SHARED / "vehicles_v2.py"):
            text = path.read_text(encoding="ascii")
            if path.name == "vehicles_v2_data.py":          # the pins themselves are not declarations
                text = text[:text.index("# BEGIN PINNED_DERIVED")]
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                        and not isinstance(node.value, bool):
                    out.update((float(node.value), -float(node.value)))
        return out

    @staticmethod
    def leaves(obj, path=""):
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                yield from TestRegistryPins.leaves(getattr(obj, f.name), path + "." + f.name if path else f.name)
        elif isinstance(obj, tuple):
            for i, v in enumerate(obj):
                yield from TestRegistryPins.leaves(v, "%s[%d]" % (path, i))
        elif isinstance(obj, float):
            yield path, obj

    def test_every_computed_registry_value_is_pinned(self):
        reg, data, _ = load_registry("pins")
        declared = self.literals()
        for name, spec in reg.VEHICLES_V2.items():
            kind = "blueboat" if name.startswith("blueboat") else "wamv"
            pinned = reg.PINNED_PATHS[kind]
            self.assertEqual(set(data.PINNED_DERIVED[name]), set(pinned), name)
            for path, value in self.leaves(spec):
                covered = any(path == p or path.startswith(p + "[") or path.startswith(p + ".") for p in pinned)
                if not covered:
                    self.assertIn(value, declared, "%s.%s = %r is computed but not pinned" % (name, path, value))

    def test_computed_values_must_agree_with_their_pins(self):
        text = (SHARED / "vehicles_v2_data.py").read_bytes().decode("ascii").replace("\r\n", "\n")
        old = "'point_depths_m': (0.11991734380519103,"
        self.assertEqual(text.count(old), 1)
        with self.assertRaisesRegex(ValueError, "disagrees with its pin"):
            load_registry("bad_pin", data_text=text.replace(old, "'point_depths_m': (0.11992734380519103,"))
        # a sub-tolerance change of an input (1 ulp of one point depth) keeps every pinned value
        ulp = repr(math.nextafter(0.11991734380519103, 1.0))
        reg, _, _ = load_registry("ulp_pin", data_text=text.replace(old, "'point_depths_m': (%s," % ulp))
        base, _, _ = load_registry("ulp_base")
        self.assertEqual(specs(reg), specs(base))

    def test_stale_or_missing_pins_fail_the_import(self):
        text = (SHARED / "vehicles_v2_data.py").read_bytes().decode("ascii").replace("\r\n", "\n")
        cut = text[:text.index("# BEGIN PINNED_DERIVED")]
        with self.assertRaisesRegex(ValueError, "no pins"):
            load_registry("no_pins", data_text=cut)


class TestQuantisedDigests(unittest.TestCase):
    def test_layout_geometry_id_ignores_ulp_noise_and_sees_real_changes(self):
        lay = L.build_layout("scatter", 2, cfg.BLUEBOAT.key, np.random.default_rng(11))
        args = dict(task="scatter", vessel=cfg.BLUEBOAT.key, tier=2, start=lay.start, goal=lay.goal,
                    start_heading=float(lay.start_heading_rad), centers=lay.centers, radii=lay.radii,
                    bounds=list(lay.bounds_m), contract_version=cfg.CONTRACT_VERSION,
                    vessel_record=cfg.BLUEBOAT.digest)
        self.assertEqual(geo.geometry_digest(**args), lay.geometry_hash)
        noisy = dict(args, centers=np.nextafter(lay.centers, np.inf), start=np.nextafter(lay.start, -np.inf),
                     start_heading=math.nextafter(float(lay.start_heading_rad), 4.0),
                     bounds=[math.nextafter(b, 1e9) for b in lay.bounds_m])
        self.assertEqual(geo.geometry_digest(**noisy), lay.geometry_hash)
        moved = dict(args, centers=lay.centers + np.array((2.0e-6, 0.0)))
        self.assertNotEqual(geo.geometry_digest(**moved), lay.geometry_hash)

    def test_canonical_form_is_declared_and_versioned(self):
        self.assertEqual(geo.IDENTITY_QUANTUM, 1.0e-6)
        self.assertEqual(geo.canonical_quanta(0.1234565), round(0.1234565 * 1e6))
        self.assertEqual(geo.canonical_quanta(-0.0), 0)
        self.assertEqual(geo.canonical_identity_value({"a": [1.0, 2, "x", None, True]}),
                         {"a": ["f:1000000", 2, "x", None, True]})
        with self.assertRaises(ValueError):
            geo.canonical_quanta(float("nan"))
        d = cfg._jsonable({"x": 0.1})
        self.assertEqual(d, {"x": "f:100000"})

    def test_e4_identity_digest_matches_the_same_rule(self):
        stamp = sys.modules.get("tasks._shared.conditions.dynamic.stamp")
        if stamp is None:
            import tasks._shared.conditions.dynamic.stamp as stamp   # noqa: F811
        self.assertEqual(stamp.IDENTITY_CANON_VERSION, geo.IDENTITY_CANON_VERSION)
        self.assertEqual(stamp.IDENTITY_QUANTUM, geo.IDENTITY_QUANTUM)
        value = {"x": 1.0, "y": [0.30000000000000004, -0.0]}
        self.assertEqual(stamp.identity_digest(value), stamp.identity_digest({"x": 1.0, "y": [0.3, 0.0]}))
        self.assertNotEqual(stamp.identity_digest(value), stamp.identity_digest({"x": 1.000002, "y": [0.3, 0.0]}))
        self.assertNotEqual(stamp.digest(value), stamp.digest({"x": 1.0, "y": [0.3, 0.0]}))


class TestFamilyResetIdentities(unittest.TestCase):
    def test_f1_reset_check_uses_the_quantisation(self):
        import tasks.hazard_nav.journal_hazard_nav as J
        self.assertTrue(J._identity_equal((1.0, -2.5), (math.nextafter(1.0, 2.0), -2.5)))
        self.assertFalse(J._identity_equal((1.0, -2.5), (1.0 + 2.0e-6, -2.5)))
        self.assertFalse(J._identity_equal((1.0,), (1.0, 2.0)))

    def test_f5_scenario_geometry_id_is_ulp_stable(self):
        import tasks.docking.journal_docking as jd
        sc = jd.build_scenario("blueboat_v2", 2, 42, 0, 0, True)
        nudged = dataclasses.replace(sc, berth_heading_rad=math.nextafter(sc.berth_heading_rad, 10.0))
        self.assertEqual(nudged.geometry_id, sc.geometry_id)
        turned = dataclasses.replace(sc, berth_heading_rad=sc.berth_heading_rad + 2.0e-6)
        self.assertNotEqual(turned.geometry_id, sc.geometry_id)

    def test_f4_runs_the_stored_candidate_after_a_quantised_rebuild_check(self):
        import tasks.station_keeping.journal_station as js
        dyn = js._dynamic()
        keys = [(0, 0)]
        pool, layouts, suite = js.admit_station_grid("blueboat_v2", 2, 42, keys)
        sp = js.pool_from_admission(pool, layouts, vessel_key="blueboat_v2", seed=42, tier=2, suite=suite,
                                    grid={"num_envs": 1, "episodes": 1})
        key = js.episode_key(42, 0, 0)
        layout = js.build_station_layout(2, "blueboat_v2", 42, 0, 0)
        sp.bind_context(key, js.station_task_context(layout, "blueboat_v2"))
        self.assertEqual(pool.keys_admitted, 1)
        stored = js.candidate_from_blob(sp._candidates[js._entry_key(key, 3)])
        self.assertEqual(dyn.digest(stored), pool.assignments[(key, 3)].candidate_digest)
        real_build = dyn.build_candidate

        def ulp_noisy_build(context, k, dose, cfg_):
            cand = real_build(context, k, dose, cfg_)
            enc = cand.request.encounter
            if not enc.intruders:
                return cand
            obj = enc.intruders[0]
            obj = dataclasses.replace(obj, x0=math.nextafter(obj.x0, math.inf))
            enc = dataclasses.replace(enc, intruders=(obj,))
            return dataclasses.replace(cand, request=dataclasses.replace(cand.request, encounter=enc))

        def far_build(context, k, dose, cfg_):
            cand = real_build(context, k, dose, cfg_)
            enc = cand.request.encounter
            obj = dataclasses.replace(enc.intruders[0], x0=enc.intruders[0].x0 + 1.0e-3)
            enc = dataclasses.replace(enc, intruders=(obj,))
            return dataclasses.replace(cand, request=dataclasses.replace(cand.request, encounter=enc))

        try:
            dyn.build_candidate = ulp_noisy_build          # run machine differs in the last ulp
            a = sp.assignment(key, 3)
            self.assertEqual(dyn.digest(a.candidate), a.candidate_digest)   # the stored candidate runs
            sp._cache.clear()
            dyn.build_candidate = far_build                 # a real difference still fails closed
            with self.assertRaisesRegex(ValueError, "identity"):
                sp.assignment(key, 3)
        finally:
            dyn.build_candidate = real_build


if __name__ == "__main__":
    unittest.main(verbosity=2)
