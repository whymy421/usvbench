"""E5 regressions (CPU only): vessel records derived from D5's registry, the
geometry version digest, the task-contract vessel parameters, the exported
termination table and the journal id scheme.

The registry (tasks/_shared/vehicles_v2.py + vehicles_v2_data.py, D5's files
borrowed unmodified) is loaded here INDEPENDENTLY of journal_cfg_v2 (own
module names) and every expected number is recomputed with elementary
formulas, so a test failure means the cfg does not follow the registry.
"""

import ast
import dataclasses
import importlib.util
import math
from pathlib import Path
import re
import sys
from types import ModuleType
import unittest

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "tasks/_shared"


def load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _PinRecorder:
    """G7 item A: captures the registry's computed values (vehicles_v2._pin_profile record mode)."""

    def __init__(self):
        self.values = {}

    def record_pin(self, name, path, value):
        self.values.setdefault(name, {})[path] = value


def _data_module(name, data_source, data_path):
    data = ModuleType(name)
    data.__file__ = str(data_path)
    exec(compile(data_source, str(data_path), "exec"), data.__dict__)
    return data


def load_registry(tag, data_source=None):
    """Registry module; data_source (text) replaces vehicles_v2_data.py IN MEMORY.

    G7 item A: every computed registry value is a declared pin (vehicles_v2_data.PINNED_DERIVED), so an
    edited registry is re-pinned first, exactly as a real registry change must be: one load in record
    mode captures the edited registry's computed values, the second load runs with them as the pins.
    """
    data_path = SHARED / "vehicles_v2_data.py"
    if data_source is None:
        data = load("e5_reg_data_" + tag, data_path)
    else:
        recorder = _PinRecorder()
        data = _data_module("e5_reg_data_rec_" + tag, data_source, data_path)
        data.PINNED_DERIVED = recorder
        saved = sys.modules.get("vehicles_v2_data")
        sys.modules["vehicles_v2_data"] = data
        try:
            assert load("e5_reg_rec_" + tag, SHARED / "vehicles_v2.py").PINS_STATUS == "recorded"
        finally:
            if saved is None:
                sys.modules.pop("vehicles_v2_data", None)
            else:
                sys.modules["vehicles_v2_data"] = saved
        data = _data_module("e5_reg_data_" + tag, data_source, data_path)
        data.PINNED_DERIVED = recorder.values
    saved = sys.modules.get("vehicles_v2_data")
    sys.modules["vehicles_v2_data"] = data
    try:
        return load("e5_reg_" + tag, SHARED / "vehicles_v2.py")
    finally:
        if saved is None:
            sys.modules.pop("vehicles_v2_data", None)
        else:
            sys.modules["vehicles_v2_data"] = saved


cfg = load("e5_test_cfg", SHARED / "journal_cfg_v2.py")
L = load("e5_test_layouts", SHARED / "journal_layouts_v2.py")
J = load("e5_test_contracts", SHARED / "journal_contracts_v2.py")
geo = cfg.geometry
REG = load_registry("ref")


def close(a, b, tol=1.0e-12):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def expected_numbers(spec, bb_spec, cruise=None):
    """Independent recomputation of every contract number from one profile."""
    poly = spec.collision.polygon_com_xy_m
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    bxs = [p[0] for p in bb_spec.collision.polygon_com_xy_m]
    length, beam = max(xs) - min(xs), max(ys) - min(ys)
    inscribed = math.inf
    for a, b in zip(poly, poly[1:] + poly[:1]):          # origin (COM) is inside
        ex, ey = b[0] - a[0], b[1] - a[1]
        inscribed = min(inscribed, abs(ex * (-a[1]) - ey * (-a[0])) / math.hypot(ex, ey))
    u = spec.speeds.cruise_m_s if cruise is None else cruise
    a = sum(m.reverse_limit_n for m in spec.propulsion.motors) / spec.mass_kg
    tau = max(m.time_constant_s for m in spec.propulsion.motors)
    h = spec.hydrostatics
    depth = h.draft_m + h.freeboard_m
    return dict(length_m=length, beam_m=beam, inscribed_radius_m=inscribed,
                circumscribed_radius_m=max(math.hypot(*p) for p in poly),
                length_scale=length / (max(bxs) - min(bxs)), mass_kg=spec.mass_kg,
                stop_distance_m=u * tau + u * u / (2.0 * a), turning_radius_m=length,
                tilt_limit_rad=math.radians(60.0), min_heave_m=-(h.freeboard_m + depth),
                max_heave_m=h.draft_m + depth, cruise_speed_m_s=u, top_speed_m_s=spec.speeds.top_model_m_s,
                com_height_above_waterline_m=h.com_height_above_water_m)


class RegistryDerivationTests(unittest.TestCase):
    def test_every_derived_number_equals_registry_recomputation(self):
        bb = REG.get_vehicle_v2("blueboat_v2")
        self.assertEqual(set(cfg.REGISTERED_VESSELS), set(REG.VEHICLES_V2))
        for name in sorted(REG.VEHICLES_V2):
            spec, rec = REG.get_vehicle_v2(name), cfg.REGISTERED_VESSELS[name]
            got, want = rec.derived_numbers(), expected_numbers(spec, bb)
            self.assertEqual(set(got), set(want))
            for key in want:
                self.assertTrue(close(got[key], want[key]), (name, key, got[key], want[key]))
            self.assertEqual(rec.polygon_com_xy_m, tuple(map(tuple, spec.collision.polygon_com_xy_m)))
            self.assertEqual(rec.capsule_center_com_xy_m, tuple(spec.collision.capsule_center_com_xy_m))
            self.assertEqual(rec.capsule_segment_half_length_m, spec.collision.capsule_segment_half_length_m)
            self.assertEqual(rec.capsule_radius_m, spec.collision.capsule_radius_m)
            v = rec.geometry
            self.assertEqual((v.capsule.center_xy_m, v.capsule.half_length_m, v.capsule.rounding_m),
                             (tuple(spec.collision.capsule_center_com_xy_m),
                              spec.collision.capsule_segment_half_length_m, spec.collision.capsule_radius_m))
            self.assertTrue(v.capsule.contains_polygon(v.footprint))
            s = want["length_scale"]
            for k, c in enumerate((4.0, 3.0, 2.0, 1.2)):              # shared tier rule g = B + c_k s
                t = geo.physical_tier(k, v)
                self.assertTrue(close(t.physical_surface_gap_m, want["beam_m"] + c * s, 1e-12))
                self.assertTrue(close(t.planning_margin_m, 0.2 * s, 1e-12))
            p = cfg.resolve_task_geometry("docking", 2, name)
            self.assertTrue(close(p["speed_tolerance_m_s"], 0.3 * want["cruise_speed_m_s"] / bb.speeds.cruise_m_s))
            self.assertTrue(close(p["position_tolerance_m"], 2.5 * s))
            self.assertTrue(close(p["slip_back_face_x_m"], max(q[0] for q in spec.collision.polygon_com_xy_m)
                                  + 0.25 * s))
            self.assertTrue(close(p["route_budget_m"], 1.40 * 19.0 * s + math.pi * want["length_m"]))
            self.assertTrue(close(p["ray_sensor_max_range_m"], 30.0 * s))
            mb = cfg.resolve_task_geometry("moving_basin", 1, name)
            self.assertEqual(mb["top_speed_m_s"], spec.speeds.top_model_m_s)
            plan = cfg.plan_class("station_keeping", name)(1)
            self.assertEqual(plan.validity_limits, dict(capsize_angle_rad=want["tilt_limit_rad"],
                                                        min_heave_m=rec.heave_limits_m[0],
                                                        max_heave_m=rec.heave_limits_m[1]))
            self.assertTrue(close(plan.validity_limits["min_heave_m"], want["min_heave_m"]))
            self.assertEqual(plan.nominal_com_height_m, spec.hydrostatics.com_height_above_water_m)

    def test_reference_length_is_the_registry_blueboat(self):
        bb = REG.get_vehicle_v2("blueboat_v2")
        xs = [p[0] for p in bb.collision.polygon_com_xy_m]
        for rec in cfg.REGISTERED_VESSELS.values():
            self.assertEqual(rec.reference_length_m, max(xs) - min(xs))
            self.assertEqual(rec.geometry.reference_length_m, rec.reference_length_m)
        self.assertEqual(cfg.BLUEBOAT.geometry.length_scale, 1.0)
        # the stand-alone default of the geometry module agrees with the registry
        self.assertEqual(geo.L_REF_M, cfg.BLUEBOAT.length_m)

    def test_recorded_capsule_must_contain_polygon(self):
        fp = cfg.WAMV.footprint
        too_small = geo.CapsuleFootprint((0.0, 0.0), 1.0, 0.5)
        with self.assertRaises(ValueError):
            geo.VesselGeometry("bad", fp, 5.0, 1.0, capsule_footprint=too_small)
        # without a recorded capsule the D4 enclosing construction is used
        plain = geo.VesselGeometry("plain", fp, 5.0, 1.0)
        self.assertTrue(plain.capsule.contains_polygon(fp))

    def test_contract_vessels_and_pending_defaults(self):
        self.assertEqual(list(cfg.VESSELS), ["blueboat_v2", "wamv_vrx_v2_opt1"])      # decision 5
        self.assertEqual(cfg.DEFAULT_WAMV_PROFILE, REG.DEFAULT_WAMV_PROFILE)
        self.assertEqual(cfg.WAMV.key, "wamv_vrx_v2_opt1")
        self.assertEqual(cfg.WAMV.mass_kg, REG.get_vehicle_v2("wamv_vrx_v2_opt1").mass_kg)
        self.assertTrue(close(cfg.WAMV.mass_kg, 283.3847, 1e-9))                      # competition assembly
        for rec in cfg.VESSELS.values():
            self.assertEqual(rec.cruise_speed_m_s, 1.0)                                 # decision 14
            self.assertEqual(rec.turning_radius_m, rec.length_m)                        # R = L
            self.assertEqual(rec.tilt_limit_rad, math.pi / 3.0)
            self.assertIn("PENDING", rec.cruise_speed_status)
        self.assertEqual(dict(cfg.CRUISE_SPEED_OVERRIDES_M_S), {})
        for key in ("wamv_profile", "cruise_speed", "design_turning_radius", "tilt_limit", "ray_sensor"):
            self.assertIn(key, cfg.PENDING_DECISIONS)
        self.assertEqual(set(cfg.REGISTRY_SOURCE_SHA256), {"vehicles_v2.py", "vehicles_v2_data.py"})

    def test_no_pinned_vessel_numbers_in_cfg_source(self):
        numbers = set()
        for spec in REG.VEHICLES_V2.values():
            h = spec.hydrostatics
            vals = [spec.mass_kg, h.draft_m, h.freeboard_m, h.com_height_above_water_m,
                    spec.speeds.top_model_m_s, spec.collision.capsule_radius_m,
                    spec.collision.capsule_segment_half_length_m]
            vals += [c for p in spec.collision.polygon_com_xy_m for c in p]
            vals += [m.reverse_limit_n for m in spec.motors] + [sum(m.reverse_limit_n for m in spec.motors)]
            numbers.update(abs(float(x)) for x in vals if abs(float(x)) > 1e-6)
        tree = ast.parse((SHARED / "journal_cfg_v2.py").read_text(encoding="ascii"))
        literals = [abs(float(n.value)) for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, float)]
        pinned = sorted({x for x in literals for y in numbers if math.isclose(x, y, rel_tol=1e-6)})
        self.assertEqual(pinned, [], "journal_cfg_v2.py must not pin registry numbers")

    def test_registry_change_changes_geometry_version_digest(self):
        # a separately loaded, unmodified registry reproduces the digest
        same = cfg.build_vessel_records(load_registry("same"))
        contract = {k: same[k] for k in cfg.VESSELS}
        self.assertEqual(cfg.geometry_version_digest(contract), cfg.GEOMETRY_VERSION_DIGEST)
        self.assertEqual(cfg.geometry_version_digest(), cfg.GEOMETRY_VERSION_DIGEST)
        text = (SHARED / "vehicles_v2_data.py").read_text(encoding="ascii")

        def replace_once(old, new):
            def edit(t):
                self.assertEqual(t.count(old), 1, old)
                return t.replace(old, new)
            return edit

        def shorten_blueboat_bow(t):
            # BlueBoat collision rectangle bow x 0.606825846 -> 0.606325846 m
            # (still encloses the 1.20 m LOA and stays inside the capsule).
            lines = t.splitlines(True)
            hit = [i for i, ln in enumerate(lines) if "'footprint_rectangle_com_xy_m'" in ln]
            self.assertEqual(len(hit), 1)
            lines[hit[0]] = lines[hit[0]].replace("0.606825846247375", "0.606325846247375")
            return "".join(lines)

        edits = {"wamv_mass": replace_once("'mass_kg': 283.3847,", "'mass_kg': 290.0,"),
                 "blueboat_draft": replace_once("'draft_m': 0.22299312651731804,", "'draft_m': 0.23,"),
                 "blueboat_bow": shorten_blueboat_bow}
        digests = {cfg.GEOMETRY_VERSION_DIGEST}
        changed_records = {}
        for tag, edit in edits.items():
            changed_reg = load_registry("mod_" + tag, edit(text))
            recs = cfg.build_vessel_records(changed_reg)
            changed_records[tag] = recs
            contract = {k: recs[k] for k in cfg.VESSELS}
            d = cfg.geometry_version_digest(contract)
            self.assertNotIn(d, digests, tag)
            digests.add(d)
            changed = [k for k in cfg.VESSELS if recs[k].digest != cfg.VESSELS[k].digest]
            self.assertTrue(changed, tag)
        # the WAM-V mass edit changes the WAM-V record only (stop distance)
        recs = changed_records["wamv_mass"]
        self.assertEqual(recs["blueboat_v2"].digest, cfg.BLUEBOAT.digest)
        self.assertNotEqual(recs[cfg.WAMV.key].stop_distance_m, cfg.WAMV.stop_distance_m)
        # a shorter BlueBoat changes the reference length, hence every WAM-V length
        recs = changed_records["blueboat_bow"]
        self.assertAlmostEqual(recs["blueboat_v2"].length_m, cfg.BLUEBOAT.length_m - 0.0005, places=12)
        self.assertEqual(recs["blueboat_v2"].length_scale, 1.0)
        self.assertGreater(recs[cfg.WAMV.key].length_scale, cfg.WAMV.length_scale)
        # the heave band follows the registry draft
        self.assertNotEqual(changed_records["blueboat_draft"]["blueboat_v2"].heave_limits_m,
                            cfg.BLUEBOAT.heave_limits_m)

    def test_layout_hash_carries_the_vessel_record_digest(self):
        lay = L.build_layout("reach", 1, cfg.WAMV.key, np.random.default_rng(3))
        want = geo.geometry_digest(task="reach", vessel=cfg.WAMV.key, tier=1, start=lay.start, goal=lay.goal,
                                   start_heading=lay.start_heading_rad, centers=lay.centers, radii=lay.radii,
                                   bounds=list(lay.bounds_m), contract_version=cfg.CONTRACT_VERSION,
                                   vessel_record=cfg.WAMV.digest)
        self.assertEqual(lay.geometry_hash, want)
        self.assertEqual(lay.params["vessel_digest"], cfg.WAMV.digest)
        sensor = lay.d7_hook["sensor"]
        self.assertEqual((sensor["kind"], sensor["ray_count"]), ("ray_fan", 36))
        self.assertTrue(close(sensor["max_range_m"], 30.0 * cfg.WAMV.length_scale))

    def test_cruise_override_changes_deadlines_not_geometry(self):
        for task in ("reach", "docking", "scatter"):
            a = L.build_layout(task, 2, cfg.WAMV.key, np.random.default_rng(21))
            b = L.build_layout(task, 2, cfg.WAMV.key, np.random.default_rng(21), cruise_speed_m_s=1.5)
            for name in ("start", "goal", "centers", "radii"):
                self.assertTrue(np.array_equal(getattr(a, name), getattr(b, name)), (task, name))
            self.assertEqual(a.start_heading_rad, b.start_heading_rad)
            self.assertEqual(a.nominal_route_length_m, b.nominal_route_length_m)
            self.assertEqual(b.deadline_steps, cfg.deadline_steps(b.nominal_route_length_m, 1.5,
                                                                  b.params["dwell_s"]))
            self.assertLess(b.deadline_s, a.deadline_s)
            self.assertNotEqual(a.geometry_hash, b.geometry_hash)          # cruise is contract identity
            self.assertEqual(b.d7_hook["envelope"]["reference_speed_mps"], 1.5)
        p = cfg.resolve_task_geometry("docking", 2, cfg.WAMV.key, cruise_speed_m_s=1.5)
        self.assertTrue(close(p["speed_tolerance_m_s"], 0.45))
        self.assertIs(cfg.vessel_record(cfg.WAMV.key), cfg.WAMV)                  # contract record untouched
        for bad in (0.0, -1.0, float("nan"), True, cfg.WAMV.top_speed_m_s):
            with self.assertRaises(ValueError):
                cfg.vessel_record(cfg.WAMV.key, bad)
        with self.assertRaises(ValueError):
            cfg.build_vessel_records(REG, {"wamv_vrx_v3": 1.0})

    def test_every_registered_profile_is_supported(self):
        for name in sorted(cfg.REGISTERED_VESSELS):
            for task in cfg.TASKS:
                for tier in range(4):
                    p = cfg.resolve_task_geometry(task, tier, name)
                    self.assertEqual(p["vessel"], name)
                    self.assertGreater(cfg.plan_class(task, name)(tier).max_deadline_s, 0.0)
            for task in ("reach", "docking"):
                lay = L.build_layout(task, 3, name, np.random.default_rng(8))
                self.assertEqual(lay.vessel, name)
                self.assertTrue(lay.witness.feasible)
        self.assertIs(cfg.plan_class("reach", cfg.WAMV.key), cfg.JOURNAL_PLAN_CLASSES[("reach", cfg.WAMV.key)])
        with self.assertRaises(ValueError):
            cfg.plan_class("reach", "wamv_vrx_v2")                              # D4's retired draft key
        with self.assertRaises(ValueError):
            cfg.resolve_task_geometry("reach", 0, "wamv")

    def test_validity_limits_default_is_the_blueboat_band(self):
        lim = J.ValidityLimits()
        bb = cfg.JOURNAL_PLAN_CLASSES[("reach", "blueboat_v2")](0).validity_limits
        self.assertEqual(lim.capsize_angle_rad, bb["capsize_angle_rad"])
        self.assertAlmostEqual(lim.min_heave_m, bb["min_heave_m"], delta=5e-4)
        self.assertAlmostEqual(lim.max_heave_m, bb["max_heave_m"], delta=5e-4)


class TerminationTableTests(unittest.TestCase):
    def test_table_covers_the_enum_with_d4_names(self):
        codes = sorted(row.code for row in J.TERMINATION_TABLE)
        self.assertEqual(codes, [int(r) for r in J.TerminationReason])
        for row in J.TERMINATION_TABLE:
            self.assertEqual(row.name, J.TERMINATION_REASON_NAMES[row.code])
            self.assertEqual(row.origin, "C5" if row.code <= 5 else "D4")
        self.assertEqual(J.TERMINATION_PRECEDENCE, (9, 6, 7, 8, 4, 2, 1, 5, 3))
        by = {row.code: row for row in J.TERMINATION_TABLE}
        self.assertEqual([by[c].outcome for c in (6, 7, 8, 9)],
                         ["capsized", "sinking", "airborne", "non_finite_state"])
        self.assertEqual((by[3].ends_as, by[0].ends_as, by[1].ends_as), ("truncated", "none", "terminated"))

    def test_c5_rows_match_the_jbase_module(self):
        c5 = load("e5_test_c5_termination", SHARED / "termination_reason.py")
        for code, name in c5.TERMINATION_REASON_NAMES.items():
            self.assertEqual(J.TERMINATION_REASON_NAMES[code], name)
        self.assertEqual(sorted(c5.TERMINATION_REASON_NAMES), list(range(6)))

    def test_precedence_is_what_termination_decision_does(self):
        rank = {row.code: row.precedence for row in J.TERMINATION_TABLE}
        flags = ("out_of_bounds", "collision", "success", "failure", "timeout")
        codes = {"out_of_bounds": 4, "collision": 2, "success": 1, "failure": 5, "timeout": 3}
        for validity in (J.TerminationReason.RUNNING,) + J.VALIDITY_REASONS:
            for mask in range(32):
                kw = {f: bool(mask >> i & 1) for i, f in enumerate(flags)}
                d = J.termination_decision(validity=validity, **kw)
                active = [codes[f] for f in flags if kw[f]]
                if validity != J.TerminationReason.RUNNING:
                    active.append(int(validity))
                want = min(active, key=lambda c: rank[c]) if active else 0
                self.assertEqual(int(d.reason), want, (validity, kw))
                self.assertEqual(d.truncated, want == 3)

    def test_validity_order_inside_one_sample(self):
        def st(z, roll, x=0.0):
            return J.RigidState((x, 0.0, z), (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0),
                                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        lim = J.ValidityLimits()
        rank = {row.code: row.precedence for row in J.TERMINATION_TABLE}
        self.assertEqual(J.validity_reason(st(lim.min_heave_m - 1, math.pi / 2), lim), J.TerminationReason.CAPSIZE)
        self.assertEqual(J.validity_reason(st(lim.max_heave_m + 1, math.pi / 2), lim), J.TerminationReason.CAPSIZE)
        self.assertEqual(J.validity_reason(st(0.0, math.pi / 2, x=math.nan), lim), J.TerminationReason.NONFINITE)
        self.assertLess(rank[9], rank[6])
        self.assertLess(rank[6], rank[7])
        self.assertLess(rank[7], rank[8])


class JournalIdTests(unittest.TestCase):
    def test_one_scheme_24_unique_ids_no_legacy_collision(self):
        ids = dict(cfg.JOURNAL_ENV_IDS)
        self.assertEqual(len(ids), 24)
        self.assertEqual(len(set(ids.values())), 24)
        pat = re.compile(r"^Isaac-USV-[A-Za-z]+-(BlueBoat|WAMVVRX)-Journal-Direct-v1$")
        for (task, vessel), gid in ids.items():
            self.assertRegex(gid, pat)
            self.assertIn("-%s-" % cfg.JOURNAL_ID_TASK_TOKENS[task], gid)
        # D1's proposals (MERGE_NOTES section 3) agree where the task is the same
        self.assertEqual(ids[("station_keeping", "blueboat_v2")], "Isaac-USV-StationKeep-BlueBoat-Journal-Direct-v1")
        self.assertEqual(ids[("forced_crossing", "blueboat_v2")], "Isaac-USV-HazardCross-BlueBoat-Journal-Direct-v1")
        legacy = set()
        for init in (ROOT / "tasks").glob("*/__init__.py"):
            if init.parent.name == "journal":   # F0 journal registration package (CONTEXT_R4 D10), not legacy
                continue
            legacy.update(re.findall(r'id="([^"]+)"', init.read_text(encoding="utf-8")))
        self.assertTrue(legacy)
        self.assertFalse(legacy & set(ids.values()))
        with self.assertRaises(ValueError):
            cfg.journal_env_id("reach", "wamv_vrx_v2_opt2")                     # profiles are data, not ids

    def test_runtime_template_classes_match_the_ids(self):
        tree = ast.parse((SHARED / "journal_runtime_cfg_v2.py").read_text(encoding="ascii"))
        keys = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                values = {s.target.id: ast.literal_eval(s.value) for s in node.body
                          if isinstance(s, ast.AnnAssign) and s.value is not None}
                self.assertIs(values.get("journal_mode"), True, node.name)
                keys.add((values["journal_task"], values["vehicle"]))
        self.assertEqual(keys, set(cfg.JOURNAL_ENV_IDS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
