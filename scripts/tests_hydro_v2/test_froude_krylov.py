"""E2 CPU tests for the horizontal Froude-Krylov wave force (controller decision 7).

Brief items (tested here; results also written to <work>/FK_TESTS.json):
  (i)   regular long wave (Tp 4 s, H 0.5 m), zero thrust: the free hull's
        horizontal velocity follows the footprint-filtered orbital velocity at
        its position (amplitude ratio >= 0.9, phase lag <= 15 deg) for
        blueboat_v2 and wamv_vrx_v2_opt1, head (along +x) and beam seas, using
        the D3 free-body 6-DOF integrator driving HydroV2Runtime;
  (ii)  the same with the term OFF shows the deficit (numbers reported; the
        ratio must be below the FK-on ratio);
  (iii) mean drift after spin-up versus the Stokes drift of the filtered
        field, u_s = omega k (H_f a)^2 exp(-2 k d) (DIAGNOSTIC, no gate);
  (iv)  calm (waves None), zero amplitude and FK OFF are bit-identical to
        D3's code (byte copies in d3_reference/, hashes asserted);
  (v)   time-step convergence 1/120, 1/240, 1/480 s with the term on: every
        metric's two-finest and 1/120-vs-1/480 differences are reported; the
        two-finest amplitude-ratio change is gated at 5 %;
  (vi)  stationary-hull FK RMS at D9 ladder A4 (Tp 4 s, Hs 1 m, gamma 3.3,
        uniform +/-45 deg, 1024 components, band [0.2, 4.0] fp) with D9's
        definition (m * sqrt(max over head/beam of var(a_x) + var(a_y))):
        D9's own numbers are first reproduced from D9's formula
        (vessel_envelope.py:267-283, 344-375), then the E2 model value is
        computed analytically and by Monte Carlo through hydro_wrench, and the
        ratio to D9's 19.9 N / 239.5 N is decomposed factor by factor.
Plus unit checks of the kernel: acceleration = time derivative of the same
orbital velocity, FK enters the predictor and the applied force exactly once,
C_a != 0 rejected. CPU only, float64 unless stated, no Isaac import.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from dataclasses import fields, replace

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHARED = ROOT / "tasks" / "_shared"
REFERENCE = HERE / "d3_reference"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(HERE))


def load(name, path=None, alias=None):
    spec = importlib.util.spec_from_file_location(alias or name, path or SHARED / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias or name] = module
    spec.loader.exec_module(module)
    return module


physics = load("physics_v2")
load("restoring_v2")
hydro = load("hydro_v2")
mixer = load("thruster_mixer")
load("reference_point")
registry = load("vehicles_v2")
load("journal_vehicle")
profiles = load("hydro_profiles")
runtime = load("hydro_v2_runtime")
scenarios = load("cpu_scenarios", HERE / "cpu_scenarios.py", alias="e2_cpu_scenarios")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1")
DIRECTIONS = (("head", 0.0), ("beam", math.pi / 2))
TP, HEIGHT = 4.0, 0.5                        # brief (i): Tp 4 s, H 0.5 m
DURATION, WINDOW = 96.0, 48.0                # 12 analysed periods after 48 s spin-up
D9_A4_FK_RMS = {"blueboat_v2": 19.938559212960364, "wamv_vrx_v2_opt1": 239.49742970554522}
# D9 results/wave.A.D4.json vessel_screens/*/horizontal/froude_krylov_force_rms_n_no_added_mass
D3_REFERENCE_SHA256 = {
    "hydro_v2.py": "0d3571a9ca3c29c9880b2dfc335c34b928f52e67c7e172a5d0b3112ff7a298cd",
    "restoring_v2.py": "e1904d05de5e0209c6a6a0acddaa9fb480280ab660d4f04cbbde109e728dd5dc",
    "physics_v2.py": "f822b7b01b65a409a3f5ccf72e1fc5a7a03fbaef57b0d547bdf6fc1639af6a78",
    "thruster_mixer.py": "881fadb9687a1cbb66807313923be3afed76d2a21c993cdd68efb5a26db9dfb2",
    "hydro_v2_runtime.py": "dc60c6eb372c0589cdc63502dc114b0f83bd0d7f96a7b73d7e8ab19142e7c91a",
}
EVIDENCE = {}


def load_d3_reference():
    """Load the D3 byte copies under their canonical names in isolation."""
    names = ("physics_v2", "restoring_v2", "hydro_v2", "thruster_mixer", "hydro_v2_runtime")
    for name in names:
        digest = hashlib.sha256((REFERENCE / (name + ".py")).read_bytes()).hexdigest()
        if digest != D3_REFERENCE_SHA256[name + ".py"]:
            raise AssertionError("d3_reference/%s.py is not the D3 deliverable" % name)
    saved = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            sys.modules.pop(name, None)
        for name in names:
            loaded[name] = load(name, REFERENCE / (name + ".py"))
    finally:
        for name in names:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]
    return loaded


D3 = load_d3_reference()


def d3_profile(profile):
    """The same vessel numbers as a D3 HydroSpec/ThrusterSpec (no E2 fields)."""
    names = {f.name for f in fields(D3["hydro_v2"].HydroSpec)}
    spec = D3["hydro_v2"].HydroSpec(**{name: getattr(profile.hydro, name) for name in names})
    t = profile.thrusters
    thrusters = D3["thruster_mixer"].ThrusterSpec(t.forward_limit_n, t.reverse_limit_n, t.arm_m,
                                                  t.yaw_command_limit_nm, t.nominal_lag_s, t.mount_z_m)
    return types.SimpleNamespace(name=profile.name, hydro=spec, thrusters=thrusters,
                                 actor_to_hull=profile.actor_to_hull, com_offset_actor_m=(0.0, 0.0, 0.0))


def bits_equal(a, b):
    """Bitwise identity (stricter than torch.equal: also the sign of zero)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = torch.int64 if a.dtype == torch.float64 else torch.int32
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


def regular_waves(n_dirs, spec, height=HEIGHT, period=TP, dtype=torch.float64):
    omega = torch.full((len(n_dirs), 1), 2 * math.pi / period, dtype=dtype)
    return hydro.SpectralWaves(torch.full_like(omega, height / 2), omega, omega.square() / spec.gravity,
                               torch.tensor([[d] for d in n_dirs], dtype=dtype), torch.zeros_like(omega))


def harmonic_fit(t, y, omega):
    """Least squares y ~ A cos(wt) + B sin(wt) + C; returns (amplitude, phase, mean)."""
    basis = torch.stack((torch.cos(omega * t), torch.sin(omega * t), torch.ones_like(t)), -1)
    coef = torch.linalg.lstsq(basis, y.unsqueeze(-1)).solution.squeeze(-1)
    return float(torch.hypot(coef[0], coef[1])), math.atan2(float(coef[1]), float(coef[0])), float(coef[2])


def wrap_deg(angle):
    return (math.degrees(angle) + 180.0) % 360.0 - 180.0


def following_run(vessel, froude_krylov, dt):
    """Regular-wave free-drift run; one env per direction; metrics per direction."""
    profile = profiles.profile_for_vehicle(vessel, froude_krylov_horizontal=froude_krylov)
    spec = profile.hydro
    dirs = [angle for _, angle in DIRECTIONS]
    waves = regular_waves(dirs, spec)
    out = scenarios.simulate_free_body(profile, torch.zeros(len(dirs), 2), DURATION, dt, waves=waves,
                                       record=True)
    t = out["t"]
    keep = t >= DURATION - WINDOW - 1e-9
    omega = 2 * math.pi / TP
    k = omega ** 2 / spec.gravity
    rows = {}
    for index, (label, angle) in enumerate(DIRECTIONS):
        unit = torch.tensor([math.cos(angle), math.sin(angle), 0.0])
        v = (out["vel"][keep, index] * unit).sum(-1)
        u = (out["orbital"][keep, index] * unit).sum(-1)
        v_amp, v_phase, v_mean = harmonic_fit(t[keep], v, omega)
        u_amp, u_phase, u_mean = harmonic_fit(t[keep], u, omega)
        transfer = float(hydro.footprint_transfer(torch.tensor([[k * math.cos(angle)]]),
                                                  torch.tensor([[k * math.sin(angle)]]), torch.zeros(1),
                                                  spec.length_m, spec.beam_m))
        amplitude = transfer * HEIGHT / 2
        stokes = omega * k * amplitude ** 2 * math.exp(-2 * k * spec.orbital_depth_m)
        fk = out["fk"][keep, index]
        rows[label] = {
            "amplitude_ratio": v_amp / u_amp, "phase_lag_deg": wrap_deg(v_phase - u_phase),
            "hull_velocity_amplitude_mps": v_amp, "orbital_velocity_amplitude_mps": u_amp,
            "mean_drift_mps": v_mean, "stokes_drift_filtered_mps": stokes,
            "drift_over_stokes": v_mean / stokes, "footprint_transfer": transfer,
            "fk_force_rms_n": float(fk.square().sum(-1).mean().sqrt()),
            "cross_track_velocity_rms_mps": float((out["vel"][keep, index, :2]
                                                   - v[:, None] * unit[:2]).square().sum(-1).mean().sqrt()),
            "final_yaw_rate_rad_s": float(out["omega"][-1, index, 2]),
        }
    return rows


# ------------------------------------------------------------------ D9 (vi)
def midpoint_spectrum(hs, tp, gamma=3.3, band=(0.2, 4.0), n=1024):
    """D9 vessel_envelope.py:464-480 (C2 v2 midpoint grid, Hs renormalised)."""
    lo, hi = band
    n = max(int(n), 256, math.ceil((hi - lo) / 0.0125))
    step = (hi - lo) / n
    x = lo + step * (np.arange(n) + 0.5)
    fp = 1.0 / tp
    f = x * fp
    alpha = 5.0 / 16.0 * hs ** 2 * fp ** 4
    s = np.where(f <= fp, 0.07, 0.09)
    spec = alpha * f ** -5 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** np.exp(-0.5 * ((f - fp) / (s * fp)) ** 2)
    e = spec * step * fp
    if hs > 0:
        e = e * ((hs / 4.0) ** 2 / e.sum())
    return f, e


def horizontal_acceleration_variance(f, e, heading_deg, spread, transfer2, order=48):
    """Direction-ensemble var(a_x)+var(a_y) (D9 vessel_envelope.py:267-283 layout)."""
    omega = 2 * math.pi * f
    x, w = np.polynomial.legendre.leggauss(order)
    total = 0.0
    for t, wt in zip(math.radians(heading_deg) + spread * x, w / 2.0):
        total += wt * float(np.sum(e * omega ** 4 * transfer2(t)))
    return total


def fk_rms(mass, f, e, transfer2, spread=math.pi / 4):
    return mass * math.sqrt(max(horizontal_acceleration_variance(f, e, h, spread, transfer2) for h in (0.0, 90.0)))


def sinc_rectangle(k, length, beam):
    return lambda t: (np.sinc(k * math.cos(t) * length / (2 * math.pi))
                      * np.sinc(k * math.sin(t) * beam / (2 * math.pi))) ** 2


def gaussian_footprint(k, length, beam, depth=0.0):
    kk = torch.tensor(k)

    def transfer2(t):
        h = hydro.footprint_transfer((kk * math.cos(t))[None], (kk * math.sin(t))[None], torch.zeros(1),
                                     length, beam)[0].numpy()
        return h ** 2 * np.exp(-2 * k * depth)
    return transfer2


D9_SCREENS = {
    # D9 vessel_envelope.py:104-162: mass, waterline length, beam = 2*max(cy+w/2).
    "blueboat_v2": dict(mass=17.26, length=1.2, beam=0.9),
    "wamv_vrx_v2_opt1": dict(mass=283.385, length=4.9,
                             beam=2 * (1.03 + math.sqrt(2 * 0.213 * 0.113 - 0.113 ** 2))),
}


class FroudeKrylovKernel(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(2718)
        self.n = 64

    def random_waves(self, spec, m=40, scale=.08):
        amp = torch.rand(self.n, m, generator=self.gen) * scale
        omega = 2 * math.pi * (0.2 + torch.rand(self.n, m, generator=self.gen))
        return hydro.SpectralWaves(amp, omega, omega.square() / spec.gravity,
                                   torch.rand(self.n, m, generator=self.gen) * 2 * math.pi,
                                   torch.rand(self.n, m, generator=self.gen) * 2 * math.pi)

    def test_acceleration_is_time_derivative_of_the_same_orbital_velocity(self):
        spec = profiles.profile_for_vehicle("blueboat_v2").hydro
        waves = self.random_waves(spec)
        xy = torch.randn(self.n, 2, generator=self.gen)
        yaw = torch.rand(self.n, generator=self.gen) * 6
        eps = 1e-6
        a = waves.sample(xy, torch.zeros_like(xy), 1.3, yaw, spec)
        plus = waves.sample(xy, torch.zeros_like(xy), 1.3 + eps, yaw, spec)
        minus = waves.sample(xy, torch.zeros_like(xy), 1.3 - eps, yaw, spec)
        derivative = (plus.orbital_velocity_world - minus.orbital_velocity_world) / (2 * eps)
        torch.testing.assert_close(a.orbital_acceleration_world, derivative, atol=1e-7, rtol=1e-6)
        # Same filter and depth: zero depth and no footprint give the bare field.
        bare = waves.sample(xy, torch.zeros_like(xy), 1.3, yaw, replace(spec, orbital_depth_m=0.0), footprint=False)
        theta = (waves.k * torch.cos(waves.direction) * xy[:, :1] + waves.k * torch.sin(waves.direction) * xy[:, 1:]
                 - waves.omega * 1.3 + waves.phase)
        expected_x = (waves.amplitude * waves.omega ** 2 * torch.sin(theta) * torch.cos(waves.direction)).sum(-1)
        torch.testing.assert_close(bare.orbital_acceleration_world[:, 0], expected_x)
        calm = hydro.WaveSample.calm(torch.zeros(self.n, 3))
        self.assertTrue(torch.equal(calm.orbital_acceleration_world, torch.zeros(self.n, 3)))

    def test_force_formula_predictor_and_single_application(self):
        for vessel in VESSELS:
            on = profiles.profile_for_vehicle(vessel)
            off = profiles.profile_for_vehicle(vessel, froude_krylov_horizontal=False)
            spec = on.hydro
            waves = self.random_waves(spec)
            q = torch.randn(self.n, 4, generator=self.gen) * torch.tensor([1.0, .05, .05, .5])
            rotation = physics.rotation_matrix(q / q.norm(dim=-1, keepdim=True))
            pos = torch.randn(self.n, 3, generator=self.gen) * .05
            pos[:, 2] += spec.equilibrium_com_z_m
            vel, omega = torch.randn(self.n, 3, generator=self.gen), torch.randn(self.n, 3, generator=self.gen)
            yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
            sample = waves.sample(pos[:, :2], torch.zeros(self.n, 2), .7, yaw, spec)
            thrust = torch.randn(self.n, 3, generator=self.gen) * 5
            a = hydro.hydro_wrench(spec, rotation, pos, vel, omega, torch.zeros(self.n, 3), sample, 1/120, thrust)
            b = hydro.hydro_wrench(off.hydro, rotation, pos, vel, omega, torch.zeros(self.n, 3), sample, 1/120,
                                   thrust)
            self.assertIsNone(b.froude_krylov_force_body)
            world = torch.zeros(self.n, 3)
            world[:, :2] = spec.mass_kg * sample.orbital_acceleration_world[:, :2]
            expected = (rotation.transpose(-1, -2) @ world.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(a.froude_krylov_force_body, expected, rtol=1e-12, atol=1e-12)
            # Applied once: force_body = buoyancy + damping + FK (buoyancy identical).
            torch.testing.assert_close(a.force_body - a.damping_force_body - a.froude_krylov_force_body,
                                       b.force_body - b.damping_force_body, rtol=1e-12, atol=1e-12)
            # Predictor sees FK: damping = -D (v_next - u) with v_next from ALL loads.
            drag, _ = hydro.damping_matrices(spec, a.relative_velocity_body, a.relative_angular_velocity_body)
            gravity = (rotation.transpose(-1, -2) @ torch.tensor([0.0, 0.0, -spec.gravity]).expand(self.n, 3)
                       .unsqueeze(-1)).squeeze(-1)
            body_v = (rotation.transpose(-1, -2) @ vel.unsqueeze(-1)).squeeze(-1)
            body_u = (rotation.transpose(-1, -2) @ a.water_velocity_world.unsqueeze(-1)).squeeze(-1)
            next_v = body_v + (1/120) * (a.force_body + thrust + spec.mass_kg * gravity) / spec.mass_kg
            torch.testing.assert_close(a.damping_force_body, -(drag @ (next_v - body_u).unsqueeze(-1)).squeeze(-1),
                                       rtol=1e-9, atol=1e-9)
            # Runtime: force_hull carries FK exactly once.
            n = self.n
            ra = runtime.HydroV2Runtime(on, n, "cpu", 1/120, dtype=torch.float64)
            rb = runtime.HydroV2Runtime(off, n, "cpu", 1/120, dtype=torch.float64)
            quat = q / q.norm(dim=-1, keepdim=True)
            actions = torch.rand(n, 2, generator=self.gen) * 2 - 1
            sa = ra.step(actions, pos, torch.cat((vel, omega), -1), quat, torch.zeros(n, 3), waves=waves)
            sb = rb.step(actions, pos, torch.cat((vel, omega), -1), quat, torch.zeros(n, 3), waves=waves)
            torch.testing.assert_close(sa.force_hull - sa.hydro.damping_force_body - sa.hydro.froude_krylov_force_body,
                                       sb.force_hull - sb.hydro.damping_force_body, rtol=1e-12, atol=1e-12)

    def test_added_mass_coefficient_and_flag_validation(self):
        spec = profiles.profile_for_vehicle("blueboat_v2").hydro
        for bad in (0.5, -1.0, float("nan"), True, "0"):
            with self.assertRaises(ValueError):
                replace(spec, added_mass_coefficient_horizontal=bad)
        with self.assertRaises(ValueError):
            replace(spec, froude_krylov_horizontal=1)
        self.assertEqual(replace(spec, added_mass_coefficient_horizontal=0).added_mass_coefficient_horizontal, 0)


class BitIdentityWithD3(unittest.TestCase):
    """(iv) calm, Hs = 0 and FK OFF reproduce D3's code bit for bit."""

    def check(self, dtype):
        gen = torch.Generator().manual_seed(4096)
        n, m = 48, 30
        for vessel, fk in (("blueboat_v2", True), ("wamv_vrx_v2_opt1", True), ("blueboat_v2", False),
                           ("wamv_vrx_v2_opt1", False)):
            e2 = profiles.profile_for_vehicle(vessel, froude_krylov_horizontal=fk)
            d3 = d3_profile(e2)
            amp = torch.rand(n, m, generator=gen, dtype=dtype) * .1
            omega = 2 * math.pi * (0.2 + torch.rand(n, m, generator=gen, dtype=dtype))
            direction = torch.rand(n, m, generator=gen, dtype=dtype) * 6.28
            phase = torch.rand(n, m, generator=gen, dtype=dtype) * 6.28
            real = (amp, omega, omega.square() / 9.81, direction, phase)
            zero = (torch.zeros_like(amp),) + real[1:]
            cases = [("calm", None, None), ("zero_amplitude", zero, zero)]
            if not fk:
                cases.append(("waves_fk_off", real, real))
            for label, e2_waves, d3_waves in cases:
                e2w = None if e2_waves is None else hydro.SpectralWaves(*e2_waves)
                d3w = None if d3_waves is None else D3["hydro_v2"].SpectralWaves(*d3_waves)
                e2rt = runtime.HydroV2Runtime(e2, n, "cpu", 1/120, dtype=dtype, gyroscopic_predictor=True)
                d3rt = D3["hydro_v2_runtime"].HydroV2Runtime(d3, n, "cpu", 1/120, dtype=dtype,
                                                             gyroscopic_predictor=True)
                degradation = (torch.rand(n, 1, generator=gen, dtype=dtype), torch.rand(n, 2, generator=gen,
                               dtype=dtype), .05)
                scale = 1 + torch.rand(n, generator=gen, dtype=dtype)
                e2rt.configure(mixer.ThrusterDegradation(*degradation), resistance_scale=scale)
                d3rt.configure(D3["thruster_mixer"].ThrusterDegradation(*degradation), resistance_scale=scale)
                pos = torch.randn(n, 3, generator=gen, dtype=dtype) * .05
                pos[:, 2] += e2.hydro.equilibrium_com_z_m
                q = torch.cat((torch.ones(n, 1, dtype=dtype), torch.randn(n, 3, generator=gen, dtype=dtype) * .1), -1)
                q = q / q.norm(dim=-1, keepdim=True)
                current = torch.randn(n, 3, generator=gen, dtype=dtype) * .3
                wind = torch.randn(n, 3, generator=gen, dtype=dtype)
                origins = torch.randn(n, 3, generator=gen, dtype=dtype) * 30
                for tick in range(4):
                    vel = torch.randn(n, 6, generator=gen, dtype=dtype) * .4
                    actions = torch.rand(n, 2, generator=gen, dtype=dtype) * 2 - 1
                    args = (actions, pos + origins, vel, q, origins)
                    kwargs = dict(current_velocity_w=current, wind_force_w=wind, wind_torque_w=wind * .1)
                    a = e2rt.step(*args, waves=e2w, **kwargs)
                    b = d3rt.step(*args, waves=d3w, **kwargs)
                    for x, y, what in ((a.force_actor, b.force_actor, "force"), (a.torque_actor, b.torque_actor,
                                                                                   "torque"),
                                       (a.motors.body_wrench, b.motors.body_wrench, "motors")):
                        self.assertTrue(bits_equal(x, y), "%s %s FK=%s %s tick %d %s" % (vessel, label, fk, what,
                                                                                         tick, dtype))
                    if e2w is not None:
                        for x, y in ((a.waves.elevation, b.waves.elevation), (a.waves.slope, b.waves.slope),
                                     (a.waves.orbital_velocity_world, b.waves.orbital_velocity_world),
                                     (a.waves.surface_angular_velocity_world, b.waves.surface_angular_velocity_world)):
                            self.assertTrue(bits_equal(x, y))
        EVIDENCE.setdefault("bit_identity_vs_d3", []).append(str(dtype))

    def test_float64(self):
        self.check(torch.float64)

    def test_float32_runtime_default(self):
        self.check(torch.float32)

    def test_kernel_validating_path_matches_d3(self):
        gen = torch.Generator().manual_seed(77)
        n = 32
        e2 = profiles.profile_for_vehicle("wamv_vrx_v2_opt1", froude_krylov_horizontal=False)
        d3 = d3_profile(e2)
        q = torch.randn(n, 4, generator=gen)
        rotation = physics.rotation_matrix(q / q.norm(dim=-1, keepdim=True))
        pos, vel, omega = (torch.randn(n, 3, generator=gen) for _ in range(3))
        amp = torch.rand(n, 9, generator=gen) * .2
        w = 2 * math.pi * (.2 + torch.rand(n, 9, generator=gen))
        parts = (amp, w, w.square() / 9.81, torch.rand(n, 9, generator=gen) * 6, torch.rand(n, 9, generator=gen) * 6)
        e2s = hydro.SpectralWaves(*parts).sample(pos[:, :2], torch.zeros(n, 2), .4, torch.zeros(n), e2.hydro)
        d3s = D3["hydro_v2"].SpectralWaves(*parts).sample(pos[:, :2], torch.zeros(n, 2), .4, torch.zeros(n),
                                                          d3.hydro)
        a = hydro.hydro_wrench(e2.hydro, rotation, pos, vel, omega, vel * .2, e2s, 1/240, vel, omega, 1.3)
        b = D3["hydro_v2"].hydro_wrench(d3.hydro, rotation, pos, vel, omega, vel * .2, d3s, 1/240, vel, omega, 1.3)
        self.assertTrue(bits_equal(a.force_body, b.force_body))
        self.assertTrue(bits_equal(a.torque_body, b.torque_body))


class RegularWaveFollowing(unittest.TestCase):
    """(i), (ii), (iii), (v): free drift in a regular Tp 4 s, H 0.5 m wave."""

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for vessel in VESSELS:
            cls.results[vessel] = {
                "fk_on": {str(round(1 / dt)): following_run(vessel, True, dt) for dt in (1/120, 1/240, 1/480)},
                "fk_off": {"240": following_run(vessel, False, 1/240)},
            }
        EVIDENCE["regular_wave"] = {"period_s": TP, "height_m": HEIGHT, "duration_s": DURATION,
                                    "analysis_window_s": WINDOW, "results": cls.results}
        print("\nRegular wave Tp %.1f s, H %.2f m, zero thrust, last %.0f s of %.0f s:" % (TP, HEIGHT, WINDOW,
                                                                                         DURATION))
        print("vessel | FK | dt | dir | |v|/|u_f| | lag deg | drift m/s | Stokes u_s m/s | drift/u_s | FK RMS N")
        for vessel, runs in cls.results.items():
            for key, by_dt in runs.items():
                for dt_label, rows in by_dt.items():
                    for direction, r in rows.items():
                        print("%s | %s | 1/%s | %s | %.4f | %.3f | %.5f | %.5f | %.3f | %.3f" % (
                            vessel, key, dt_label, direction, r["amplitude_ratio"], r["phase_lag_deg"],
                            r["mean_drift_mps"], r["stokes_drift_filtered_mps"], r["drift_over_stokes"],
                            r["fk_force_rms_n"]))

    def test_i_hull_follows_filtered_orbital_velocity_with_fk(self):
        for vessel in VESSELS:
            for direction, r in self.results[vessel]["fk_on"]["240"].items():
                self.assertGreaterEqual(r["amplitude_ratio"], 0.9, (vessel, direction, r))
                self.assertLessEqual(abs(r["phase_lag_deg"]), 15.0, (vessel, direction, r))

    def test_ii_without_fk_the_hull_lags_and_undershoots(self):
        for vessel in VESSELS:
            for direction, off in self.results[vessel]["fk_off"]["240"].items():
                on = self.results[vessel]["fk_on"]["240"][direction]
                self.assertLess(off["amplitude_ratio"], on["amplitude_ratio"], (vessel, direction))
                self.assertGreater(off["phase_lag_deg"], on["phase_lag_deg"], (vessel, direction))

    def test_iii_mean_drift_diagnostic_is_finite(self):
        # DIAGNOSTIC only (no gate on the ratio): see PHYSICS_NOTES_E2.md for why
        # about one half of the classical Stokes drift is expected.
        for vessel in VESSELS:
            for runs in self.results[vessel].values():
                for rows in runs.values():
                    for r in rows.values():
                        self.assertTrue(math.isfinite(r["drift_over_stokes"]))

    def test_v_timestep_convergence(self):
        table = {}
        for vessel in VESSELS:
            runs = self.results[vessel]["fk_on"]
            for direction in runs["480"]:
                for metric in ("amplitude_ratio", "phase_lag_deg", "mean_drift_mps", "hull_velocity_amplitude_mps",
                               "fk_force_rms_n", "drift_over_stokes"):
                    fine, mid, coarse = (runs[k][direction][metric] for k in ("480", "240", "120"))
                    scale = max(abs(fine), 1e-12)
                    table["%s %s %s" % (vessel, direction, metric)] = {
                        "1/120": coarse, "1/240": mid, "1/480": fine,
                        "two_finest_abs": abs(mid - fine), "two_finest_rel": abs(mid - fine) / scale,
                        "coarse_vs_fine_abs": abs(coarse - fine), "coarse_vs_fine_rel": abs(coarse - fine) / scale}
                self.assertLess(table["%s %s amplitude_ratio" % (vessel, direction)]["two_finest_rel"], 0.05)
        EVIDENCE["timestep_convergence_fk_on"] = table
        print("\nTime-step convergence with the term on (relative to 1/480 s):")
        for key, row in table.items():
            print("  %-48s 1/240-1/480 %.3e (rel %.3e); 1/120-1/480 %.3e (rel %.3e)" % (
                key, row["two_finest_abs"], row["two_finest_rel"], row["coarse_vs_fine_abs"],
                row["coarse_vs_fine_rel"]))


class LadderA4StationaryRms(unittest.TestCase):
    """(vi) stationary-hull FK RMS at D9 A4 versus D9's 19.9 N / 239.5 N."""

    def test_vi_against_d9(self):
        f, e = midpoint_spectrum(1.0, 4.0)
        k = (2 * math.pi * f) ** 2 / 9.81
        report = {}
        for vessel in VESSELS:
            spec = profiles.profile_for_vehicle(vessel).hydro
            d9 = D9_SCREENS[vessel]
            reproduced = fk_rms(d9["mass"], f, e, sinc_rectangle(k, d9["length"], d9["beam"]))
            self.assertAlmostEqual(reproduced / D9_A4_FK_RMS[vessel], 1.0, delta=1e-6)
            steps = {
                "a_d9_definition_d9_inputs": reproduced,
                "b_sinc_surface_registry_mass_and_footprint": fk_rms(spec.mass_kg, f, e,
                                                                     sinc_rectangle(k, spec.length_m, spec.beam_m)),
                "c_gaussian_surface_registry": fk_rms(spec.mass_kg, f, e,
                                                      gaussian_footprint(k, spec.length_m, spec.beam_m)),
                "d_e2_model_gaussian_depth_registry": fk_rms(spec.mass_kg, f, e,
                                                             gaussian_footprint(k, spec.length_m, spec.beam_m,
                                                                                spec.orbital_depth_m)),
            }
            required = hydro.required_high_frequency_hz(spec, 0.25) / 0.25
            f_ext, e_ext = midpoint_spectrum(1.0, 4.0, band=(0.2, max(4.0, required)),
                                             n=max(1024, math.ceil((max(4.0, required) - .2) / .0125 * 1.0)))
            k_ext = (2 * math.pi * f_ext) ** 2 / 9.81
            steps["e_e2_model_band_to_required_edge_%.3gfp" % required] = fk_rms(
                spec.mass_kg, f_ext, e_ext, gaussian_footprint(k_ext, spec.length_m, spec.beam_m,
                                                               spec.orbital_depth_m))
            monte_carlo = self.monte_carlo(vessel, f, e)
            analytic = steps["d_e2_model_gaussian_depth_registry"]
            self.assertLess(abs(monte_carlo / analytic - 1), 0.05, (vessel, monte_carlo, analytic))
            report[vessel] = {"d9_published_n": D9_A4_FK_RMS[vessel], "decomposition_n": steps,
                              "monte_carlo_through_hydro_wrench_n": monte_carlo,
                              "ratio_e2_to_d9": analytic / D9_A4_FK_RMS[vessel],
                              "ratio_monte_carlo_to_d9": monte_carlo / D9_A4_FK_RMS[vessel],
                              "required_band_fp": required}
        EVIDENCE["a4_stationary_fk_rms"] = report
        print("\nA4 stationary-hull FK RMS (N), definition m*sqrt(max_head/beam var(a_x)+var(a_y)):")
        for vessel, row in report.items():
            print("  %s: D9 %.3f; E2 analytic %.3f (ratio %.3f); Monte Carlo via hydro_wrench %.3f (ratio %.3f)" % (
                vessel, row["d9_published_n"], row["decomposition_n"]["d_e2_model_gaussian_depth_registry"],
                row["ratio_e2_to_d9"], row["monte_carlo_through_hydro_wrench_n"], row["ratio_monte_carlo_to_d9"]))
            for key, value in row["decomposition_n"].items():
                print("      %-52s %.3f N" % (key, value))

    def monte_carlo(self, vessel, f, e, realizations=96, samples=160):
        """Random-phase, uniform +/-45 deg realisations sampled through hydro_wrench (stationary hull)."""
        profile = profiles.profile_for_vehicle(vessel)
        spec = profile.hydro
        gen = torch.Generator().manual_seed(9001)
        amplitude = torch.tensor(np.sqrt(2 * e)).expand(realizations, -1)
        omega = torch.tensor(2 * math.pi * f).expand(realizations, -1)
        worst = 0.0
        for heading in (0.0, math.pi / 2):
            direction = heading + (torch.rand(realizations, len(f), generator=gen) - .5) * (math.pi / 2)
            phase = torch.rand(realizations, len(f), generator=gen) * 2 * math.pi
            waves = hydro.SpectralWaves(amplitude.clone(), omega.clone(), omega.square() / spec.gravity,
                                        direction, phase)
            pos = torch.zeros(realizations, 3)
            pos[:, 2] = spec.equilibrium_com_z_m
            identity = torch.eye(3).expand(realizations, 3, 3)
            zero = torch.zeros(realizations, 3)
            total = 0.0
            times = torch.rand(samples, generator=gen) * 2000.0
            for t in times.tolist():
                sample = waves.sample(pos[:, :2], torch.zeros(realizations, 2), t, torch.zeros(realizations), spec)
                wrench = hydro.hydro_wrench(spec, identity, pos, zero, zero, zero, sample, 1/120)
                total += float(wrench.froude_krylov_force_body[:, :2].square().sum())
            worst = max(worst, math.sqrt(total / (realizations * samples)))
        return worst


def tearDownModule():
    path = ROOT.parent / "FK_TESTS.json"
    path.write_text(json.dumps(EVIDENCE, indent=1, sort_keys=True) + "\n", encoding="ascii")
    print("Evidence written to %s" % path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
