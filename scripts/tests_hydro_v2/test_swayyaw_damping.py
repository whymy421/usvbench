"""G4-A5 sway/yaw damping rule (hydro_profiles ``swayyaw_damping``), CPU only, no Isaac.

Controller decision G4-A5 (2026-10-01, from the G1 damping sweep variant B2q_xf1_quadyaw):
the journal-default VRX-classic WAM-V (wamv_vrx_v2_opt1 and its bare variant) keeps the
VRX-classic linear sway 40 N s/m and yaw 400 N m s/rad and adds
  sway Y_v|v| = Sarda et al. 2016 Table 3 cross-flow (485.4 N s2/m2 with L 4.9 m,
                T 0.10917 m, B_hull 0.426 m), and
  yaw  N_r|r| = Y_v|v| L^3 / 32 (strip theory, 1784.6 N m s2/rad2);
VRX classic (40 / 0, 400 / 0) stays selectable as the comparison profile; the BlueBoat
and VRX option 2 keep their registry values. Checked here:
  * formula values against an independent evaluation of the formulas, the brief's rounded
    numbers, the registry inputs and their file:line citations;
  * comparison profile, registry rule and refusals; everything but sway/yaw unchanged;
  * the stamp (status, sources, options, journal vehicle record, certificate vessel profile);
  * every consumer resolves the SAME coefficients: the env profile (resolve_journal_vehicle,
    as journal_env_mixin.journal_setup and the family cfg builders call it), the
    HydroV2Runtime constants, the MPPI model (VesselDynamicsV2 / dynamics_for_cfg with
    cfg.journal_vehicle_options), the frozen-tuning model record and the D7 envelope spec;
  * the D7 envelope: identical envelope fields, steady spin rate above the declared yaw-rate
    bound with the new yaw damping, t_avoid unchanged.

Run: python -B scripts/tests_hydro_v2/test_swayyaw_damping.py (or through run_cpu_tests.py)
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
for _path in (str(REPO / "scripts"), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import mppi_controller as mc  # noqa: E402

torch.set_num_threads(1)
profiles = mc._shared_module("hydro_profiles")
registry = mc._shared_module("vehicles_v2")
journal = mc._shared_module("journal_vehicle")
crossflow = mc._shared_module("crossflow_damping")
runtime_module = mc._shared_module("hydro_v2_runtime")
HydroSpec = mc._shared_module("hydro_v2").HydroSpec

WAMV = "wamv_vrx_v2_opt1"
CLASSIC = (WAMV, "wamv_vrx_v2_opt1_bare")
OTHERS = ("blueboat_v2", "wamv_vrx_v2_opt2", "wamv_vrx_v2_opt2_bare")
REGISTRY_FILE = REPO / "tasks" / "_shared" / "vehicles_v2.py"


def independent_sarda(length, draft, beam, rho):
    """Sarda 2016 Table 3 row Y_v, written out again (sarda2016_text.txt:371-380)."""
    return 0.5 * rho * (1.1 + 0.0045 * length / draft - 0.1 * beam / draft + 0.016 * (beam / draft) ** 2) \
        * (math.pi * draft * length / 2.0)


def tbase_hydro(name, rollpitch="journal_default"):
    """The T-base v1 (pre-G4) construction of profile_for_vehicle's HydroSpec, repeated here."""
    spec = registry.get_vehicle_v2(name)
    kwargs = registry.to_d3_hydro_kwargs(spec)
    # T25 switch (b), owner decision T1.35b: the journal default is zeta 0.1 for every vessel (was BlueBoat only)
    if rollpitch == "journal_default":
        h, inertia = spec.hydrostatics, spec.inertia_com_kg_m2
        pair = (2.0 * 0.1 * math.sqrt(h.stiffness_roll_nm_rad * inertia[0][0]),
                2.0 * 0.1 * math.sqrt(h.stiffness_pitch_nm_rad * inertia[1][1]))
        kwargs["angular_linear_damping"] = (pair[0], pair[1], kwargs["angular_linear_damping"][2])
    return HydroSpec(**kwargs, rotational_relative_motion=True, froude_krylov_horizontal=True)


def swayyaw(hydro):
    return (hydro.linear_damping[1], hydro.quadratic_damping[1], hydro.angular_linear_damping[2],
            hydro.angular_quadratic_damping[2])


def valid_rules(name):
    if name in CLASSIC:
        return profiles.SWAYYAW_RULES
    return ("journal_default", "registry")


class FormulaValues(unittest.TestCase):
    def test_wamv_journal_default_coefficients(self):
        profile = profiles.profile_for_vehicle(WAMV)
        spec = registry.get_vehicle_v2(WAMV)
        length, beam = registry.WAMV_HULL_LENGTH_M, 2.0 * registry.WAMV_HULL_RADIUS_M
        draft, rho = spec.hydrostatics.draft_m, spec.hydrostatics.water_density_kg_m3
        # the brief's inputs: L 4.9 m, T = registry draft 0.10917 m, B_hull 0.426 m, fresh water
        self.assertEqual((length, beam, rho), (4.9, 0.426, 1000.0))
        self.assertLess(abs(draft - 0.10917), 5e-6)
        y_vv = independent_sarda(length, draft, beam, rho)
        n_rr = y_vv * length ** 3 / 32.0
        self.assertLess(abs(y_vv - 485.4), 0.05)        # brief: 485.4 N s2/m2
        self.assertLess(abs(n_rr - 1784.6), 0.05)       # brief: 1784.6 N m s2/rad2
        lin_v, quad_v, lin_r, quad_r = swayyaw(profile.hydro)
        self.assertEqual((lin_v, lin_r), (40.0, 400.0))   # VRX-classic linear terms kept
        self.assertTrue(math.isclose(quad_v, y_vv, rel_tol=1e-12), (quad_v, y_vv))
        self.assertTrue(math.isclose(quad_r, n_rr, rel_tol=1e-12), (quad_r, n_rr))
        self.assertEqual(crossflow.strip_yaw_quadratic(quad_v, length), quad_r)
        self.assertTrue(math.isclose(crossflow.sarda_sway_quadratic(length, draft, beam, rho), y_vv, rel_tol=1e-12))

    def test_bare_variant_uses_its_own_registry_draft(self):
        spec = registry.get_vehicle_v2("wamv_vrx_v2_opt1_bare")
        draft = spec.hydrostatics.draft_m
        self.assertNotEqual(draft, registry.get_vehicle_v2(WAMV).hydrostatics.draft_m)
        y_vv = independent_sarda(4.9, draft, 0.426, 1000.0)
        _, quad_v, lin_r, quad_r = swayyaw(profiles.profile_for_vehicle("wamv_vrx_v2_opt1_bare").hydro)
        self.assertTrue(math.isclose(quad_v, y_vv, rel_tol=1e-12))
        self.assertTrue(math.isclose(quad_r, y_vv * 4.9 ** 3 / 32.0, rel_tol=1e-12))
        self.assertEqual(lin_r, 400.0)

    def test_inputs_are_read_from_the_registry_with_valid_citations(self):
        spec = registry.get_vehicle_v2(WAMV)
        inputs = profiles.crossflow_inputs(spec)
        lines = REGISTRY_FILE.read_text(encoding="utf-8").splitlines()
        anchors = {"length_m": ["WAMV_HULL_LENGTH_M = 4.9"], "beam_m": ["WAMV_HULL_RADIUS_M = 0.213"],
                   "water_density_kg_m3": ["RHO_WATER = 1000.0"],
                   "draft_m": ["mean_depth = sum(", "draft_m=mean_depth"]}
        for key, expected in anchors.items():
            cited = [int(part.split(":")[1].split()[0].rstrip(",)")) for part in inputs[key]["source"].split()
                     if part.startswith("tasks/_shared/vehicles_v2.py:")]
            self.assertEqual(len(cited), len(expected), (key, inputs[key]["source"]))
            for number, text in zip(cited, expected):
                self.assertIn(text, lines[number - 1], (key, number))
        self.assertEqual(inputs["draft_m"]["value"], spec.hydrostatics.draft_m)
        self.assertEqual(inputs["draft_m"]["value"], sum(registry._data.WAMV["competition"]["point_depths_m"]) / 4.0)
        for other in OTHERS:
            with self.assertRaises(ValueError):
                profiles.crossflow_inputs(registry.get_vehicle_v2(other))

    def test_formula_module_refuses_bad_inputs(self):
        for bad in ((0.0, 0.1, 0.4, 1000.0), (4.9, -0.1, 0.4, 1000.0), (4.9, 0.1, float("nan"), 1000.0),
                    (4.9, 0.1, 0.4, True)):
            with self.assertRaises(ValueError):
                crossflow.sarda_sway_quadratic(*bad)
        with self.assertRaises(ValueError):
            crossflow.strip_yaw_quadratic(0.0, 4.9)


class RulesAndComparisonProfile(unittest.TestCase):
    def test_vrx_classic_and_registry_reproduce_the_tbase_profiles(self):
        for name in CLASSIC:
            for rule in ("vrx_classic", "registry"):
                profile = profiles.profile_for_vehicle(name, swayyaw_damping=rule)
                self.assertEqual(profile.hydro, tbase_hydro(name), (name, rule))
                self.assertEqual(swayyaw(profile.hydro), (40.0, 0.0, 400.0, 0.0))
                self.assertEqual(profile.sources["options"]["swayyaw_damping"], rule)
        for name in OTHERS:      # journal default = registry, HydroSpec identical to T-base
            for rule in ("journal_default", "registry"):
                profile = profiles.profile_for_vehicle(name, swayyaw_damping=rule)
                self.assertEqual(profile.hydro, tbase_hydro(name), (name, rule))
                self.assertEqual(profile.sources["options"]["swayyaw_damping"], "registry")
        legacy = profiles.profile_for_vehicle("blueboat_v2", rollpitch_damping="legacy_470")
        self.assertEqual(legacy.hydro, tbase_hydro("blueboat_v2", rollpitch="legacy_470"))

    def test_only_sway_and_yaw_change(self):
        for name in CLASSIC:
            new = profiles.profile_for_vehicle(name)
            old = profiles.profile_for_vehicle(name, swayyaw_damping="vrx_classic")
            for field in dataclasses.fields(new.hydro):
                a, b = getattr(new.hydro, field.name), getattr(old.hydro, field.name)
                if field.name in ("quadratic_damping", "angular_quadratic_damping"):
                    index = 1 if field.name == "quadratic_damping" else 2
                    self.assertEqual(tuple(a[:index]) + tuple(a[index + 1:]), tuple(b[:index]) + tuple(b[index + 1:]))
                    self.assertGreater(a[index], 0.0)
                    self.assertEqual(b[index], 0.0)
                else:
                    self.assertEqual(a, b, (name, field.name))
            self.assertEqual(new.thrusters, old.thrusters)
            self.assertEqual((new.com_offset_actor_m, new.actor_to_hull), (old.com_offset_actor_m, old.actor_to_hull))

    def test_refusals(self):
        for name in OTHERS:
            for rule in ("crossflow_sarda_strip", "vrx_classic"):
                with self.assertRaisesRegex(ValueError, "VRX-classic WAM-V"):
                    profiles.profile_for_vehicle(name, swayyaw_damping=rule)
        for bad in ("crossflow", "classic", "", None, 1.0, True):
            with self.assertRaises(ValueError):
                profiles.profile_for_vehicle(WAMV, swayyaw_damping=bad)
        # the registry provenance values are untouched by the rule
        self.assertEqual(registry.get_vehicle_v2(WAMV).damping, registry.WAMV_DAMPING_C)
        self.assertEqual(registry.WAMV_DAMPING_C.linear[1::4], (40.0, 400.0))
        self.assertEqual(registry.WAMV_DAMPING_C.quadratic[1::4], (0.0, 0.0))


class Stamp(unittest.TestCase):
    def test_status_sources_options_and_certificate_profile(self):
        profile = profiles.profile_for_vehicle(WAMV)
        self.assertIn("sway/yaw damping crossflow_sarda_strip -> sway 40 N s/m + 485.406 N s2/m2, "
                      "yaw 400 N m s/rad + 1784.61 N m s2/rad2", profile.status)
        record = profile.sources["swayyaw_damping"]
        self.assertEqual(record["rule"], "crossflow_sarda_strip")
        self.assertEqual(record["requested"], "journal_default")
        self.assertEqual(record["sway_formula"], crossflow.SARDA_FORMULA)
        self.assertEqual(record["yaw_formula"], crossflow.STRIP_FORMULA)
        self.assertIn("Sarda", record["sway_source"])
        self.assertIn("G4-A5", record["decision"])
        self.assertEqual(record["sway_linear_quadratic"], list(swayyaw(profile.hydro)[:2]))
        self.assertEqual(record["yaw_linear_quadratic"], list(swayyaw(profile.hydro)[2:]))
        self.assertEqual(set(record["inputs"]), {"length_m", "draft_m", "beam_m", "water_density_kg_m3"})
        self.assertEqual(record["registry_sway_linear_quadratic"], [40.0, 0.0])
        classic = profiles.profile_for_vehicle(WAMV, swayyaw_damping="vrx_classic")
        self.assertIn("sway/yaw damping vrx_classic -> sway 40 N s/m + 0 N s2/m2, yaw 400 N m s/rad + 0", classic.status)
        self.assertIn("comparison profile", classic.sources["swayyaw_damping"]["note"])
        blueboat = profiles.profile_for_vehicle("blueboat_v2")
        self.assertIn("sway/yaw damping registry -> sway 27 N s/m + 18 N s2/m2, yaw 4 N m s/rad + 6", blueboat.status)
        # journal vehicle record and the certificate's vessel_profile stamp carry it
        cert_v3 = __import__("cert_v3")
        for rule in ("journal_default", "vrx_classic"):
            vehicle = journal.resolve_journal_vehicle(WAMV, swayyaw_damping=rule)
            self.assertEqual(vehicle.profile_status, profiles.profile_for_vehicle(WAMV, swayyaw_damping=rule).status)
            self.assertEqual(vehicle.profile_options["swayyaw_damping"],
                             "crossflow_sarda_strip" if rule == "journal_default" else rule)
            cfg = SimpleNamespace(vessel_profile=vehicle.name, vessel_profile_status=vehicle.profile_status)
            stamp = cert_v3.declared_vessel_profile(cfg)
            self.assertIn("sway/yaw damping " + vehicle.profile_options["swayyaw_damping"], stamp["status"])
        self.assertTrue(any("G4-A5" in item for item in journal.resolve_journal_vehicle(WAMV).pending))


class EveryConsumerAgrees(unittest.TestCase):
    def test_env_profile_runtime_mppi_model_and_envelope_spec_carry_identical_coefficients(self):
        for name in CLASSIC + OTHERS:
            for rule in valid_rules(name):
                vehicle = journal.resolve_journal_vehicle(name, swayyaw_damping=rule)
                env = mc.profile_planar_damping(vehicle.hydro_profile)
                model = mc.VesselDynamicsV2(name, swayyaw_damping=rule).params
                self.assertEqual({k: model[k] for k in mc.PLANAR_DAMPING_KEYS}, env, (name, rule))
                self.assertEqual(model["swayyaw_damping"], vehicle.profile_options["swayyaw_damping"])
                cfg = SimpleNamespace(journal_contract_version=2, vehicle=name, decimation=2,
                                      underwater_physics_cfg=SimpleNamespace(hydrodynamics_model="hydro_v2"),
                                      sim=SimpleNamespace(dt=1.0 / 120.0), journal_vehicle_options={"swayyaw_damping": rule})
                from_cfg = mc.dynamics_for_cfg(cfg).params
                self.assertEqual({k: from_cfg[k] for k in mc.PLANAR_DAMPING_KEYS}, env, (name, rule))
                runtime = runtime_module.HydroV2Runtime(vehicle.hydro_profile, 1, "cpu", 1.0 / 120.0,
                                                        dtype=torch.float64, engine_gravity=(0.0, 0.0, -9.81),
                                                        rigid_body_disable_gravity=False)
                c = runtime.constants
                self.assertEqual((float(c.linear_damping[1]), float(c.quadratic_damping[1]),
                                  float(c.angular_linear_damping[2]), float(c.angular_quadratic_damping[2])),
                                 (env["sway_lin"], env["sway_quad"], env["yaw_lin"], env["yaw_quad"]))
                spec = journal.envelope_spec(vehicle)
                self.assertEqual((spec.damping.linear[1], spec.damping.quadratic[1], spec.damping.linear[5],
                                  spec.damping.quadratic[5]),
                                 (env["sway_lin"], env["sway_quad"], env["yaw_lin"], env["yaw_quad"]), (name, rule))
                if vehicle.profile_options["swayyaw_damping"] != "crossflow_sarda_strip":
                    self.assertIs(spec, registry.get_vehicle_v2(name))
        # without options a journal cfg resolves the journal default everywhere
        cfg = SimpleNamespace(journal_contract_version=2, vehicle=WAMV, decimation=2,
                              underwater_physics_cfg=SimpleNamespace(hydrodynamics_model="hydro_v2"),
                              sim=SimpleNamespace(dt=1.0 / 120.0))
        self.assertEqual(mc.dynamics_for_cfg(cfg).params["swayyaw_damping"], "crossflow_sarda_strip")
        record = mc.model_damping_record(WAMV)
        self.assertEqual(record["swayyaw_damping"], "crossflow_sarda_strip")
        self.assertEqual({k: record[k] for k in mc.PLANAR_DAMPING_KEYS},
                         mc.profile_planar_damping(journal.resolve_journal_vehicle(WAMV).hydro_profile))


class D7Envelope(unittest.TestCase):
    def test_envelope_builds_with_the_new_yaw_damping_and_t_avoid_is_unchanged(self):
        dynamic = mc._shared_module("conditions.dynamic")
        cfg_v2 = mc._shared_module("journal_cfg_v2")
        for name in ("blueboat_v2", WAMV):
            record = cfg_v2.vessel_record(name)
            kwargs = dict(reference_speed_mps=record.cruise_speed_m_s, turning_radius_m=record.turning_radius_design_m)
            old_env, old = dynamic.envelope_from_spec_v2(registry.get_vehicle_v2(name), **kwargs)
            vehicle = journal.resolve_journal_vehicle(name)
            new_env, new = dynamic.envelope_from_spec_v2(journal.envelope_spec(vehicle), **kwargs)
            old, new = dict(old), dict(new)
            self.assertEqual(dataclasses.asdict(new_env), dataclasses.asdict(old_env), name)
            self.assertEqual(new["t_avoid_s"], old["t_avoid_s"])
            self.assertLessEqual(new["yaw_rate_bound_rps"], new["steady_spin_rate_rps"])
            h = vehicle.hydro_profile.hydro
            lin, quad = h.angular_linear_damping[2], h.angular_quadratic_damping[2]
            torque = new["zero_surge_yaw_torque_nm"]
            spin = torque / lin if quad == 0.0 else (-lin + math.sqrt(lin * lin + 4.0 * quad * torque)) / (2.0 * quad)
            self.assertTrue(math.isclose(new["steady_spin_rate_rps"], spin, rel_tol=1e-12))
        # WAM-V numbers (G4 FINAL_REPORT item 3): spin 0.2452 rad/s > bound 0.1860 rad/s; t_avoid 3.7427 s
        self.assertLess(abs(new["steady_spin_rate_rps"] - 0.24524), 1e-4)
        self.assertLess(abs(new["yaw_rate_bound_rps"] - 0.18597), 1e-4)
        self.assertLess(abs(new["t_avoid_s"] - 3.74273), 1e-4)
        self.assertLess(abs(old["steady_spin_rate_rps"] - 0.51357), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
