"""T25 switch patch (b), owner decision T1.35b: WAM-V roll/pitch damping journal default "zeta:0.1" (CPU).

* the switch is one named constant (hydro_profiles.WAMV_ROLLPITCH_JOURNAL_DEFAULT) citing the decision; the
  BlueBoat default stays "zeta:0.1";
* every WAM-V registry profile built with the journal default now carries c = 2 x 0.1 x sqrt(K I) for roll
  and pitch; yaw, surge, sway, heave, quadratic terms, mass, inertia and thrust are those of the "registry"
  profile, which stays selectable (the VRX values);
* the journal env runs it (runtime profile, condition stamp rotational_damping, rollpitch_damping stamp
  pinned rule) and the certificate stamp validates; the diagnostic hydro.rollpitch_zeta override still works
  on top of it; the WAM-V vehicle record names the decision.

Run: python -B scripts/tests_t25/test_t25_switch_b_wamv_rollpitch.py   (or pytest)
"""
from __future__ import annotations

import json
import math
import os
import sys
import unittest

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
for _path in (_SCRIPTS, os.path.join(_SCRIPTS, "tests_journal_glue")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

import glue_fixtures as G  # noqa: E402

HP = G.shared("hydro_profiles")
JV = G.shared("journal_vehicle")
JC = G.journal_config


def zeta_pair(spec, zeta):
    h, inertia = spec.hydrostatics, spec.inertia_com_kg_m2
    return (2.0 * zeta * math.sqrt(h.stiffness_roll_nm_rad * inertia[0][0]),
            2.0 * zeta * math.sqrt(h.stiffness_pitch_nm_rad * inertia[1][1]))


class Switch(unittest.TestCase):
    def test_named_constant_and_blueboat_unchanged(self):
        self.assertEqual(HP.WAMV_ROLLPITCH_JOURNAL_DEFAULT, "zeta:0.1")
        self.assertIn("T1.35b", HP.WAMV_ROLLPITCH_DECISION)
        self.assertEqual(HP._WAMV_DEFAULT_RULE, HP.WAMV_ROLLPITCH_JOURNAL_DEFAULT)
        self.assertEqual(HP._BLUEBOAT_DEFAULT_RULE, "zeta:0.1")
        blue = HP.profile_for_vehicle("blueboat_v2")
        self.assertEqual(blue.sources["options"]["rollpitch_damping"], "zeta:0.1")
        self.assertEqual(tuple(blue.hydro.angular_linear_damping[:2]),
                         zeta_pair(G.vehicles_v2.get_vehicle_v2("blueboat_v2"), 0.1))

    def test_every_wamv_profile_takes_zeta_0p1_and_registry_stays_selectable(self):
        names = sorted(n for n in G.vehicles_v2.VEHICLES_V2 if n.startswith("wamv_"))
        self.assertIn("wamv_vrx_v2_opt1", names)
        for name in names:
            spec = G.vehicles_v2.get_vehicle_v2(name)
            new = HP.profile_for_vehicle(name)
            vrx = HP.profile_for_vehicle(name, rollpitch_damping="registry")
            self.assertEqual(new.sources["options"]["rollpitch_damping"], "zeta:0.1")
            self.assertEqual(tuple(new.hydro.angular_linear_damping[:2]), zeta_pair(spec, 0.1))
            self.assertEqual(tuple(vrx.hydro.angular_linear_damping[:2]),
                             (float(spec.damping.linear[3]), float(spec.damping.linear[4])))
            self.assertEqual(new.hydro.angular_linear_damping[2], vrx.hydro.angular_linear_damping[2])
            for field in ("mass_kg", "inertia_kgm2", "linear_damping", "quadratic_damping",
                          "angular_quadratic_damping", "roll_stiffness_nm", "pitch_stiffness_nm"):
                self.assertEqual(getattr(new.hydro, field), getattr(vrx.hydro, field), (name, field))
            self.assertEqual(new.thrusters, vrx.thrusters)
            self.assertNotEqual(tuple(new.hydro.angular_linear_damping[:2]), tuple(vrx.hydro.angular_linear_damping[:2]))
            if spec.damping == G.vehicles_v2.WAMV_DAMPING_C:       # VRX-classic set (roll/pitch 50 / 50)
                self.assertGreater(new.hydro.angular_linear_damping[1], 10.0 * vrx.hydro.angular_linear_damping[1])

    def test_env_runs_it_and_the_stamp_validates(self):
        import validate_cert_v3 as V
        cfg = G.journal_cfg("wamv_vrx_v2_opt1", tokens=["waves=D2"])
        env = G.FakeJournalEnv(cfg, 2)
        env.reset()
        spec = G.vehicles_v2.get_vehicle_v2("wamv_vrx_v2_opt1")
        self.assertEqual(tuple(env._hydro_v2.profile.hydro.angular_linear_damping[:2]), zeta_pair(spec, 0.1))
        stamp = env.journal_rollpitch_damping_stamp()
        self.assertEqual(stamp["pinned"]["rule"], "zeta:0.1")
        self.assertTrue(stamp["pinned_values"])
        header = env.journal_certificate_header()
        self.assertEqual(header["conditions"]["context"]["rotational_damping"]["angular_linear_damping"][:2],
                         list(zeta_pair(spec, 0.1)))
        doc = json.loads(json.dumps({"header": {"journal": {"task": "reach"}, "conditions": header["conditions"],
                                                "seed_namespace": {"namespace": "dose_pilots", "purpose": "pilot"},
                                                "rollpitch_damping": stamp, "runtime": {}}, "records": []}))
        self.assertEqual(V.rollpitch_damping_errors(doc), [])
        # the diagnostic override still applies on top of the new default
        cfg2 = G.journal_cfg("wamv_vrx_v2_opt1", tokens=["waves=D2"])
        JC.apply_rollpitch_override(cfg2, {"rollpitch_zeta": 0.2})
        over = G.FakeJournalEnv(cfg2, 2)
        self.assertEqual(tuple(over._hydro_v2.profile.hydro.angular_linear_damping[:2]), zeta_pair(spec, 0.2))
        self.assertEqual(over.journal_rollpitch_damping_stamp()["pinned"]["rule"], "zeta:0.1")

    def test_vehicle_record_names_the_decision(self):
        record = JV.resolve_journal_vehicle("wamv_vrx_v2_opt1")
        text = " ".join(getattr(record, "pending_decisions", ()) or ())
        self.assertTrue("T1.35b" in text or "T1.35b" in " ".join(JV.PENDING_WAMV))
        self.assertNotIn("keeps the VRX values", " ".join(JV.PENDING_WAMV))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
