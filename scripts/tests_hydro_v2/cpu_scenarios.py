"""Small free-body 6-DOF CPU probe; NOT a PhysX replay or controller evaluation.

World linear velocity, body angular velocity, unit quaternion attitude. Semi-
implicit translation and exponential-map orientation, explicit gyroscopic term.
Legacy force laws reproduce the C2 v1 shared-family BlueBoat formulas at a
COMMON COM datum, excluding contacts/engine caps. Stop a row if |omega|>20
rad/s or speed>30 m/s; report divergence rather than hiding it behind clipping.

E2 (2026-09-30) re-run with the D5 registry profiles (decision 1) through
hydro_profiles.profile_for_vehicle, for the variants
  blueboat_v2 with zeta 0.1 (decision 2 default) and with legacy 470,
  wamv_vrx_v2_opt1 (competition, VRX-classic damping and thrust),
each with the horizontal Froude-Krylov force ON (decision 7 default) and OFF.
G4-A5 (2026-10-01): profile_for_vehicle("wamv_vrx_v2_opt1") now resolves the
journal-default sway/yaw rule (Sarda cross-flow sway + strip-theory yaw added
to the VRX-classic linear terms), so variant "wamv_vrx_v2_opt1" is that
default; the E2 run above (and its CPU_SCENARIOS.json anchors) used the
VRX-classic values, reproduced by variant "wamv_vrx_v2_opt1_vrx_classic".
Part 1 (legacy vs v2, per variant): straight run, turn, head/beam current,
legacy W1-W4 head/beam waves (30 % thrust), combined wave+current, tilted
passive release, and journal ladder A (D9: Tp 4 s, Hs 0.125/0.25/0.5/1.0 m,
gamma 3.3, +/-45 deg) head/beam waves at ZERO thrust (horizontal excursion
and mean drift). Part 2 (anchors through HydroV2Runtime, calm water, per
profile): terminal surge speed, zero-surge spin rate, heave/roll/pitch free
decay periods and damping ratios against closed forms and report values.
simulate_free_body is the reusable batched integrator used by
test_froude_krylov.py. Results: <work>/CPU_SCENARIOS.json. See DESIGN_NOTES.md
and PHYSICS_NOTES_E2.md.
"""
import importlib.util
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "tasks" / "_shared"
sys.path.insert(0, str(SHARED))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SHARED / (name+".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


physics = load("physics_v2")
restore = load("restoring_v2")
hydro = load("hydro_v2")
mixer = load("thruster_mixer")
load("reference_point")
registry = load("vehicles_v2")
load("journal_vehicle")
profiles = load("hydro_profiles")
runtime = load("hydro_v2_runtime")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

PROFILE_VARIANTS = {
    "blueboat_v2_zeta0.1": dict(name="blueboat_v2", rollpitch_damping="zeta:0.1"),
    "blueboat_v2_legacy470": dict(name="blueboat_v2", rollpitch_damping="legacy_470"),
    "wamv_vrx_v2_opt1": dict(name="wamv_vrx_v2_opt1"),
    "wamv_vrx_v2_opt1_vrx_classic": dict(name="wamv_vrx_v2_opt1", swayyaw_damping="vrx_classic"),
}


def variant_profile(variant, froude_krylov=True):
    options = dict(PROFILE_VARIANTS[variant])
    name = options.pop("name")
    return profiles.profile_for_vehicle(name, froude_krylov_horizontal=froude_krylov, **options)


BLUEBOAT = variant_profile("blueboat_v2_zeta0.1")
WAMV = variant_profile("wamv_vrx_v2_opt1")
# The legacy (C2 v1) laws act on the same USD mass properties.
LEGACY_MASS = BLUEBOAT.hydro.mass_kg
LEGACY_INERTIA = BLUEBOAT.hydro.inertia_kgm2
# Legacy W1-W4 midpoint probes. Source: C1 audit inventory 02, lines 431-438
# (Hs/Tp boxes (0.05-0.10,1.0-1.4), (0.15-0.25,1.6-2.0), (0.30-0.45,2.2-2.7),
# (0.55-0.75,2.8-3.5)); midpoints below.
LADDER = ((.075, 1.2), (.20, 1.8), (.375, 2.45), (.65, 3.15))
# D9 ladder A (DOSE_DESIGN.md section 5; CONTEXT_R3 decision 8): Tp 4 s.
LADDER_A = ((.125, 4.0), (.25, 4.0), (.50, 4.0), (1.00, 4.0))
SPREAD_W = math.pi/6          # D3 probe: uniform +/-30 deg
SPREAD_A = math.pi/4          # D9 ladder A: uniform +/-45 deg


def cases():
    rows = [dict(name="straight", action=(.65, 0), current=(0, 0, 0)),
            dict(name="turn", action=(1, 1), current=(0, 0, 0)),
            dict(name="head_current", action=(.45, 0), current=(-1, 0, 0)),
            dict(name="beam_current", action=(.45, 0), current=(0, 1, 0))]
    # Wave direction = propagation direction; the hull moves along +x, so
    # direction pi is head seas and pi/2 is beam seas.
    for index, (hs, tp) in enumerate(LADDER, 1):
        for heading in (math.pi, math.pi/2):
            rows.append(dict(name="W%d_%s" % (index, "head" if heading == math.pi else "beam"),
                             action=(.3, 0), current=(0, 0, 0), hs=hs, tp=tp, heading=heading))
    rows.append(dict(name="W2_head_plus_head_current_0.5", action=(.3, 0), current=(-.5, 0, 0),
                     hs=.20, tp=1.8, heading=math.pi))
    rows.append(dict(name="tilted_passive", action=(0, 0), current=(0, 0, 0), tilt=True))
    return rows


def ladder_a_cases():
    """Zero-thrust journal ladder A rows (E2): 0 = waves along the hull x axis.

    Integrated as a SEPARATE batch: ``spectrum`` sizes every row of a batch to
    the widest band, and the random directions are drawn after the phases, so
    adding rows with a wider band to D3's batch would change D3's W-row
    realisations. With two batches D3's rows get exactly the realisations D3's
    own program draws for the same profile. (Seeds restart at 342019 in each
    batch; the A rows are analysed on their own, so the shared seeds do not
    matter.)
    """
    rows = []
    for index, (hs, tp) in enumerate(LADDER_A, 1):
        for heading in (0.0, math.pi/2):
            rows.append(dict(name="A%d_%s_drift" % (index, "head" if heading == 0.0 else "beam"),
                             action=(0, 0), current=(0, 0, 0), hs=hs, tp=tp, heading=heading,
                             spread=SPREAD_A))
    return rows


def spectrum(rows, legacy=False, cutoff=None, nested=False, spec=None):
    """Fixed seed 342019+row, gamma3.3, uniform directional spreading per row.

    Legacy uses its historical 104 bins .04..1.60 Hz (C2 sea_state.py v1).
    v2 uses df/fp=.01 over .2fp..f_hi, with f_hi from the production helper
    hydro_v2.required_high_frequency_hz unless ``cutoff`` (in fp) is given.
    Nested convergence uses identical .2..16fp bins/phases and fixed 16fp
    normalization, zeroing bins above ``cutoff``. Spread half-width per row
    (D3 rows +/-30 deg as before; E2 ladder-A rows +/-45 deg).
    """
    spec = BLUEBOAT.hydro if spec is None else spec
    amplitude, frequency, direction, phase = [], [], [], []
    width = 0
    for row in rows:
        fp = 1/row.get("tp", 2.5)
        top = 16. if nested else (cutoff if cutoff is not None else hydro.required_high_frequency_hz(spec, fp)/fp)
        width = max(width, 104 if legacy else round((top-.2)/.01))
    for row_index, row in enumerate(rows):
        hs, tp = row.get("hs", 0.), row.get("tp", 2.5)
        fp = 1/tp
        n = width
        if legacy:
            df = (1.6-.04)/n
            f = .04+(torch.arange(n)+.5)*df
            keep = torch.ones(n)
        else:
            top = 16. if nested else (cutoff if cutoff is not None else hydro.required_high_frequency_hz(spec, fp)/fp)
            df = .01*fp
            f = (.2+(torch.arange(n)+.5)*.01)*fp
            keep = (f <= top*fp).to(f.dtype)
        sigma = torch.where(f <= fp, .07, .09)
        peak = torch.exp(-.5*((f-fp)/(sigma*fp))**2)
        density = f**-5*torch.exp(-1.25*(fp/f)**4)*3.3**peak*keep
        a = torch.sqrt(2*density*df)
        a = a*((hs/4)/torch.sqrt((a*a/2).sum()))
        if nested:
            a = a*(f <= cutoff*fp)
        generator = torch.Generator().manual_seed(342019+row_index)
        phases = torch.rand(n, generator=generator)*2*math.pi
        angles = row.get("heading", 0.)+(torch.rand(n, generator=generator)-.5)*2*row.get("spread", SPREAD_W)
        amplitude.append(a); frequency.append(f); direction.append(angles); phase.append(phases)
    freq = torch.stack(frequency)
    omega = 2*math.pi*freq
    return hydro.SpectralWaves(torch.stack(amplitude), omega, omega**2/spec.gravity,
                               torch.stack(direction), torch.stack(phase))


def quaternion_step(quaternion, omega_body, dt):
    angle = omega_body.norm(dim=-1, keepdim=True)*dt
    vector = omega_body*dt*.5*torch.sinc(angle/(2*math.pi))
    increment = torch.cat((torch.cos(angle/2), vector), -1)
    a, b = quaternion[:, :1], quaternion[:, 1:]
    c, d = increment[:, :1], increment[:, 1:]
    q = torch.cat((a*c-(b*d).sum(-1, keepdim=True), a*d+c*b+torch.cross(b, d, dim=-1)), -1)
    return q/q.norm(dim=-1, keepdim=True)


def legacy_wrench(rotation, pos, velocity, omega_world, action, current, wave, enabled_wave, enabled_current):
    """C2 v1 BlueBoat laws: ground-relative hull drag, separate current/wave
    8|v|v, constant 100*eta, constant 200*slope, world angular damping 470,
    anisotropic helper restoring and the -0.05 m CB couple. Coefficients are
    the legacy registry values (C2 vehicles.py:123-138, calm cfg:139-145);
    this function is a reproduction of the OLD model, not a kernel.
    """
    inverse = rotation.transpose(-1, -2)
    body_v = (inverse@velocity.unsqueeze(-1)).squeeze(-1)
    force_body = torch.zeros_like(pos)
    force_body[:, 0] = torch.where(action[:, 0] >= 0, 80*action[:, 0], 48*action[:, 0])
    force_body[:, :2] -= (torch.tensor((9.1, 27.))+torch.tensor((5.9, 18.))*body_v[:, :2].abs())*body_v[:, :2]
    force = (rotation@force_body.unsqueeze(-1)).squeeze(-1)
    force[:, 2] -= 200*velocity[:, 2]
    buoyancy = 1000*9.8*.0346*torch.clamp(.5-pos[:, 2]/.376, 0, 1)
    force[:, 2] += buoyancy
    b = rotation[:, 2, :]
    torque_body = torch.stack((-280*b[:, 1], 141*b[:, 0], 23*action[:, 1]), -1)
    torque = (rotation@torque_body.unsqueeze(-1)).squeeze(-1)
    offset = -.05*rotation[:, :, 2]
    bf = torch.zeros_like(pos); bf[:, 2] = buoyancy
    torque += torch.cross(offset, bf, dim=-1)
    torque[:, :2] -= 470*omega_world[:, :2]
    torque[:, 2] -= (4+6*omega_world[:, 2].abs())*omega_world[:, 2]
    rel = current[:, :2]-velocity[:, :2]
    force[:, :2] += enabled_current[:, None]*8*rel*rel.norm(dim=-1, keepdim=True)
    rel = wave.orbital_velocity_world[:, :2]-velocity[:, :2]
    force[:, :2] += enabled_wave[:, None]*8*rel*rel.norm(dim=-1, keepdim=True)
    force[:, 2] += enabled_wave*100*wave.elevation
    torque[:, 0] += enabled_wave*200*wave.slope[:, 1]
    torque[:, 1] -= enabled_wave*200*wave.slope[:, 0]
    return force, (inverse@torque.unsqueeze(-1)).squeeze(-1)


def integrate(rows, model="v2", dt=1/120, duration=20., cutoff=None, nested=False, profile=None):
    profile = BLUEBOAT if profile is None else profile
    spec, thrust = profile.hydro, profile.thrusters
    n = len(rows)
    pos, vel, omega = (torch.zeros(n, 3) for _ in range(3))
    quat = torch.zeros(n, 4); quat[:, 0] = 1
    for index, row in enumerate(rows):
        if row.get("tilt"):
            quat[index] = quaternion_step(quat[index:index+1], torch.tensor([[.01, -.015, 0]]), 1)[0]
    if model == "legacy":
        # Legacy equilibrium of its own clipped box law (buoyancy uses 9.8).
        pos[:, 2] = .376*(.5-LEGACY_MASS*9.81/(1000*9.8*.0346))
        mass, inertia, gravity = LEGACY_MASS, torch.tensor(LEGACY_INERTIA), 9.81
    else:
        pos[:, 2] = spec.equilibrium_com_z_m
        mass, inertia, gravity = spec.mass_kg, torch.tensor(spec.inertia_kgm2), spec.gravity
    action = torch.tensor([row["action"] for row in rows], dtype=torch.float64)
    current = torch.tensor([row["current"] for row in rows], dtype=torch.float64)
    enabled_wave = torch.tensor([row.get("hs", 0) > 0 for row in rows])
    enabled_current = current.norm(dim=-1) > 0
    spectral = spectrum(rows, legacy=model == "legacy", cutoff=cutoff, nested=nested, spec=spec)
    sample_spec = replace(spec, orbital_depth_m=0) if model == "legacy" else spec
    state = mixer.zero_state(action)
    origin = torch.zeros(n, 2)
    alive = torch.ones(n, dtype=torch.bool)
    failure = torch.full((n,), float("nan"))
    sums = torch.zeros(n, 6); counts = torch.zeros(n)
    maximum_omega = torch.zeros(n)
    immersion_clipped = torch.zeros(n, dtype=torch.int64)
    max_tilt = torch.zeros(n)
    steps = round(duration/dt)
    second_half = []
    for tick in range(steps):
        rotation = physics.rotation_matrix(quat)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        wave = spectral.sample(pos[:, :2], origin, tick*dt, yaw, sample_spec, footprint=model != "legacy")
        if model == "legacy":
            volume = .0346*(.5-pos[:, 2]/.376)
        else:
            volume = spec.mass_kg/spec.water_density + spec.waterplane_m2*(
                wave.elevation+spec.equilibrium_com_z_m-pos[:, 2])
        immersion_clipped += alive & ((volume <= 0) | (volume >= spec.maximum_volume_m3))
        max_tilt = torch.maximum(max_tilt, torch.acos(rotation[:, 2, 2].clamp(-1, 1)))
        omega_world = (rotation@omega.unsqueeze(-1)).squeeze(-1)
        gyro = -torch.cross(omega, (inertia@omega.unsqueeze(-1)).squeeze(-1), dim=-1)
        if model == "legacy":
            force, torque = legacy_wrench(rotation, pos, vel, omega_world, action, current, wave,
                                          enabled_wave, enabled_current)
        else:
            command = mixer.step_thrusters(action, state, thrust, dt)
            state = command.state
            known_force, known_torque = command.body_wrench[:, :3], command.body_wrench[:, 3:]
            result = hydro.hydro_wrench(spec, rotation, pos, vel, omega_world, current, wave, dt,
                                        known_force, known_torque+gyro)
            force = (rotation@(result.force_body+known_force).unsqueeze(-1)).squeeze(-1)
            torque = result.torque_body+known_torque
        acceleration = force/mass
        acceleration[:, 2] -= gravity
        next_vel = vel+dt*acceleration
        next_omega = omega+dt*torch.linalg.solve(inertia, (torque+gyro).unsqueeze(-1)).squeeze(-1)
        bad = (~torch.isfinite(next_vel).all(-1) | ~torch.isfinite(next_omega).all(-1)
               | (next_vel.norm(dim=-1) > 30) | (next_omega.norm(dim=-1) > 20))
        failure[alive & bad] = (tick+1)*dt
        alive &= ~bad
        vel = torch.where(alive[:, None], next_vel, vel)
        omega = torch.where(alive[:, None], next_omega, omega)
        pos = pos+dt*vel*alive[:, None]
        quat = torch.where(alive[:, None], quaternion_step(quat, omega, dt), quat)
        maximum_omega = torch.maximum(maximum_omega, omega.norm(dim=-1))
        if tick >= steps//2:
            rotation = physics.rotation_matrix(quat)
            roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
            pitch = torch.asin(torch.clamp(-rotation[:, 2, 0], -1, 1))
            values = torch.stack((pos[:, 2]-(spec.equilibrium_com_z_m if model != "legacy" else 0.), roll, pitch,
                                  vel[:, 0], vel[:, 1], omega[:, 2]), -1)
            sums += values.square()*alive[:, None]
            counts += alive
            second_half.append(pos[:, :2].clone())
    rms = torch.sqrt(sums/counts.clamp_min(1)[:, None])
    # Horizontal excursion (E2): RMS of the second-half XY track about its own
    # least-squares straight line (oscillatory wave-following motion), and the
    # slope of that line (mean drift or transit velocity), world axes.
    track = torch.stack(second_half)                       # [T, n, 2]
    times = (torch.arange(track.shape[0]) - (track.shape[0]-1)/2)*dt
    slope = (times[:, None, None]*(track-track.mean(0))).sum(0)/times.square().sum()
    residual = track-track.mean(0)-times[:, None, None]*slope
    excursion = residual.square().sum(-1).mean(0).sqrt()
    options = profile.sources.get("options", {})
    result = []
    for i, row in enumerate(rows):
        result.append(dict(scenario=row["name"], model=model,
                           profile=profile.name if model == "v2" else "legacy_c2_v1",
                           froude_krylov=bool(spec.froude_krylov_horizontal) if model == "v2" else None,
                           rollpitch_damping=options.get("rollpitch_damping") if model == "v2"
                           else "legacy_470_explicit", dt=dt,
                           cutoff_fp=(("adaptive" if cutoff is None else cutoff) if model == "v2" else None),
                           completed=bool(alive[i]), diverged_at_s=None if alive[i] else float(failure[i]),
                           final_xyz_m=pos[i].tolist(), final_velocity_mps=vel[i].tolist(),
                           final_yaw_rate_rad_s=float(omega[i, 2]),
                           max_accepted_angular_speed_rad_s=float(maximum_omega[i]),
                           second_half_samples=int(counts[i]), immersion_clipped_steps=int(immersion_clipped[i]),
                           max_tilt_deg=float(max_tilt[i])*180/math.pi,
                           second_half_rms_dz_roll_pitch_vx_vy_yawrate=rms[i].tolist() if counts[i] > 0 else None,
                           horizontal_excursion_rms_m=float(excursion[i]) if alive[i] else None,
                           second_half_mean_velocity_xy_mps=slope[i].tolist() if alive[i] else None))
    return result


# --------------------------------------------------------------- free body
def simulate_free_body(profile, actions, duration, dt, waves=None, z_offset=0.0, roll_rad=0.0,
                       pitch_rad=0.0, current=None, record=False):
    """Batched free-body integration driving HydroV2Runtime (the E2 runtime path).

    actions [N,2]; waves a SpectralWaves with N rows or None. Integrator as in
    ``integrate`` (semi-implicit translation, exponential-map attitude,
    explicit gyroscopic term, gyroscopic_predictor=True). With record=True the
    returned dict holds per-step tensors sampled BEFORE each step (state and
    the wave sample the runtime used): t [T], pos/vel/omega [T,N,3], orbital
    [T,N,3] (filtered orbital velocity at the hull COM), fk [T,N,3] (hull-axis
    Froude-Krylov force), roll/pitch [T,N]; always the final state as
    final_pos/final_vel/final_omega/final_quat.
    """
    h = profile.hydro
    n = actions.shape[0]
    rt = runtime.HydroV2Runtime(profile, n, "cpu", dt, dtype=torch.float64, gyroscopic_predictor=True,
                                engine_gravity=(0.0, 0.0, -h.gravity), rigid_body_disable_gravity=False)
    pos = torch.zeros(n, 3)
    pos[:, 2] = h.equilibrium_com_z_m + z_offset
    vel = torch.zeros(n, 3)
    omega = torch.zeros(n, 3)
    quat = torch.zeros(n, 4); quat[:, 0] = 1.0
    if roll_rad or pitch_rad:
        quat = quaternion_step(quat, torch.tensor([[roll_rad, pitch_rad, 0.0]]).expand(n, 3), 1.0)
    inertia = torch.tensor(h.inertia_kgm2)
    origins = torch.zeros(n, 3)
    trace = {key: [] for key in ("pos", "vel", "omega", "orbital", "fk", "roll", "pitch")}
    times = []
    for tick in range(round(duration/dt)):
        rotation = physics.rotation_matrix(quat)
        omega_world = (rotation@omega.unsqueeze(-1)).squeeze(-1)
        step = rt.step(actions, pos, torch.cat((vel, omega_world), -1), quat, origins,
                       current_velocity_w=current, waves=waves)
        if record:
            times.append(tick*dt)
            trace["pos"].append(pos.clone())
            trace["vel"].append(vel.clone())
            trace["omega"].append(omega.clone())
            trace["orbital"].append(step.waves.orbital_velocity_world.clone())
            fk = step.hydro.froude_krylov_force_body
            trace["fk"].append(torch.zeros(n, 3) if fk is None else fk.clone())
            trace["roll"].append(torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2]))
            trace["pitch"].append(torch.asin(torch.clamp(-rotation[:, 2, 0], -1, 1)))
        gyro = -torch.cross(omega, (inertia@omega.unsqueeze(-1)).squeeze(-1), dim=-1)
        force_world = (rotation@step.force_actor.unsqueeze(-1)).squeeze(-1)
        acceleration = force_world/h.mass_kg
        acceleration[:, 2] -= h.gravity
        vel = vel+dt*acceleration
        omega = omega+dt*torch.linalg.solve(inertia, (step.torque_actor+gyro).unsqueeze(-1)).squeeze(-1)
        pos = pos+dt*vel
        quat = quaternion_step(quat, omega, dt)
    out = {"final_pos": pos, "final_vel": vel, "final_omega": omega, "final_quat": quat}
    if record:
        out["t"] = torch.tensor(times)
        out.update({key: torch.stack(value) for key, value in trace.items()})
    return out


def simulate_runtime(profile, action, duration, dt, z_offset=0.0, roll_rad=0.0, pitch_rad=0.0):
    """Single-hull calm-water trace AFTER each step: (z - datum, roll, pitch, vx, yaw rate)."""
    out = simulate_free_body(profile, torch.tensor([action], dtype=torch.float64), duration, dt,
                             z_offset=z_offset, roll_rad=roll_rad, pitch_rad=pitch_rad, record=True)
    h = profile.hydro
    rotation = physics.rotation_matrix(out["final_quat"])
    final = (float(out["final_pos"][0, 2])-h.equilibrium_com_z_m,
             math.atan2(float(rotation[0, 2, 1]), float(rotation[0, 2, 2])),
             math.asin(max(-1.0, min(1.0, -float(rotation[0, 2, 0])))),
             float(out["final_vel"][0, 0]), float(out["final_omega"][0, 2]))
    trace = [(float(out["pos"][i, 0, 2])-h.equilibrium_com_z_m, float(out["roll"][i, 0]),
              float(out["pitch"][i, 0]), float(out["vel"][i, 0, 0]), float(out["omega"][i, 0, 2]))
             for i in range(1, out["pos"].shape[0])]
    trace.append(final)
    return trace


def decay(signal, dt):
    """Period from successive maxima and per-cycle amplitude ratio."""
    peaks = [i for i in range(1, len(signal)-1) if signal[i] > signal[i-1] and signal[i] >= signal[i+1]
             and signal[i] > 0]
    peaks = peaks[:4]
    if len(peaks) < 3:
        return None, None
    period = (peaks[-1]-peaks[0])*dt/(len(peaks)-1)
    ratio = (signal[peaks[-1]]/signal[peaks[0]])**(1/(len(peaks)-1))
    return period, ratio


REPORT_SPEEDS = {
    # (terminal surge report, zero-surge spin report). BlueBoat: published
    # maximum 3 m/s (store page, D5 SpeedReference.top_published_m_s; the legacy
    # surge drag was fitted to it, so agreement is circular); WAM-V classic:
    # VRX report 6.5 (2.26 m/s, 0.51 rad/s).
    "blueboat_v2": (3.0, None),
    "wamv_vrx_v2_opt1": (2.26, 0.51),
}


def anchors(profile_list=None):
    rows = []
    profile_list = profile_list or [(variant, variant_profile(variant)) for variant in PROFILE_VARIANTS]
    for variant, profile in profile_list:
        report_speed, report_spin = REPORT_SPEEDS[profile.name]
        h, t = profile.hydro, profile.thrusters
        for dt in (1/120, 1/480):
            surge = simulate_runtime(profile, (1., 0.), 60., dt)[-1][3]
            spin = simulate_runtime(profile, (0., 1.), 60., dt)[-1][4]
            heave = [row[0] for row in simulate_runtime(profile, (0., 0.), 6., dt, z_offset=.05)]
            roll = [row[1] for row in simulate_runtime(profile, (0., 0.), 6., dt, roll_rad=math.radians(2))]
            pitch = [row[2] for row in simulate_runtime(profile, (0., 0.), 6., dt, pitch_rad=math.radians(2))]
            kz = h.heave_stiffness_n_per_m
            wn_z = math.sqrt(kz/h.mass_kg)
            zeta_z = h.linear_damping[2]/(2*math.sqrt(kz*h.mass_kg))
            ixx, iyy = h.inertia_kgm2[0][0], h.inertia_kgm2[1][1]
            wn_r = math.sqrt(h.roll_stiffness_nm/ixx)
            zeta_r = h.angular_linear_damping[0]/(2*math.sqrt(h.roll_stiffness_nm*ixx))
            wn_p = math.sqrt(h.pitch_stiffness_nm/iyy)
            zeta_p = h.angular_linear_damping[1]/(2*math.sqrt(h.pitch_stiffness_nm*iyy))
            spin_torque = 2*t.reverse_limit_n[0]*t.arm_m
            lin, quad = h.angular_linear_damping[2], h.angular_quadratic_damping[2]
            spin_closed = spin_torque/lin if quad == 0 else (-lin+math.sqrt(lin*lin+4*quad*spin_torque))/(2*quad)
            thrust = sum(t.forward_limit_n)
            lin, quad = h.linear_damping[0], h.quadratic_damping[0]
            surge_closed = (-lin+math.sqrt(lin*lin+4*quad*thrust))/(2*quad)
            heave_period, heave_ratio = decay(heave, dt)
            roll_period, roll_ratio = decay(roll, dt)
            pitch_period, pitch_ratio = decay(pitch, dt)

            def expected_ratio(zeta):
                return math.exp(-2*math.pi*zeta/math.sqrt(1-zeta*zeta)) if zeta < 1 else None

            def damped_period(wn, zeta):
                return 2*math.pi/(wn*math.sqrt(1-zeta**2)) if zeta < 1 else None

            def measured_zeta(ratio):
                if ratio is None or ratio <= 0:
                    return None
                delta = -math.log(ratio)
                return delta/math.sqrt(4*math.pi**2+delta**2)
            rows.append(dict(
                variant=variant, profile=profile.name, dt=dt,
                rollpitch_damping_nms=list(h.angular_linear_damping[:2]),
                surge_60s_mps=surge, surge_closed_form_mps=surge_closed, surge_report_mps=report_speed,
                spin_60s_rad_s=spin, spin_closed_form_rad_s=spin_closed, spin_report_rad_s=report_spin,
                heave_period_s=heave_period, heave_period_closed_s=damped_period(wn_z, zeta_z),
                heave_cycle_ratio=heave_ratio, heave_cycle_ratio_closed=expected_ratio(zeta_z),
                heave_zeta=zeta_z, heave_zeta_measured=measured_zeta(heave_ratio),
                roll_period_s=roll_period, roll_period_closed_s=damped_period(wn_r, zeta_r),
                roll_cycle_ratio=roll_ratio, roll_cycle_ratio_closed=expected_ratio(zeta_r),
                roll_zeta=zeta_r, roll_zeta_measured=measured_zeta(roll_ratio),
                pitch_period_s=pitch_period, pitch_period_closed_s=damped_period(wn_p, zeta_p),
                pitch_cycle_ratio=pitch_ratio, pitch_cycle_ratio_closed=expected_ratio(zeta_p),
                pitch_zeta=zeta_p, pitch_zeta_measured=measured_zeta(pitch_ratio),
                roll_final_abs_rad=abs(roll[-1]), pitch_final_abs_rad=abs(pitch[-1])))
    return rows


def max_change(group_a, group_b):
    changes = []
    for a, b in zip(group_a, group_b):
        va = torch.tensor(a["second_half_rms_dz_roll_pitch_vx_vy_yawrate"])
        vb = torch.tensor(b["second_half_rms_dz_roll_pitch_vx_vy_yawrate"])
        significant = vb.abs() > 1e-4
        changes.append(float(((va-vb).abs()/vb.abs().clamp_min(1e-4))[significant].max())
                       if significant.any() else 0.)
    return max(changes)


def integrate_batches(batches, model, dt, **kwargs):
    """Integrate each row batch separately (see ladder_a_cases) and concatenate."""
    out = []
    for rows in batches:
        out.extend(integrate(rows, model, dt, **kwargs))
    return out


def main():
    batches = (cases(), ladder_a_cases())
    rows = [row for batch in batches for row in batch]
    results = []
    gates = {}
    print("Part 1: 6-DOF CPU probes, 20 s, seed342019+case, gamma3.3; no engine caps or contacts.", flush=True)
    print("Profiles: %s x Froude-Krylov on/off" % ", ".join(PROFILE_VARIANTS), flush=True)
    for dt in (1/120, 1/480):
        batch = integrate_batches(batches, "legacy", dt)
        results.extend(batch)
        print("legacy dt=1/%d: %d/%d completed" % (round(1/dt), sum(r["completed"] for r in batch), len(batch)),
              flush=True)
    for variant in PROFILE_VARIANTS:
        for fk in (True, False):
            profile = variant_profile(variant, fk)
            label = "%s FK %s" % (variant, "on" if fk else "off")
            groups = {}
            for dt in (1/120, 1/240, 1/480):
                batch = integrate_batches(batches, "v2", dt, profile=profile)
                for row in batch:
                    row["variant"] = variant
                results.extend(batch)
                groups[dt] = batch
                print("%s v2 dt=1/%d: %d/%d completed" % (label, round(1/dt), sum(r["completed"] for r in batch),
                                                          len(batch)), flush=True)
            complete = all(r["completed"] for group in groups.values() for r in group)
            change_fine = max_change(groups[1/240], groups[1/480]) if complete else float("inf")
            change_coarse = max_change(groups[1/120], groups[1/480]) if complete else float("inf")
            gates[label] = {"all_completed": complete, "max_rms_change_240_480": change_fine,
                            "max_rms_change_120_480": change_coarse}
            print("%s: 1/240->1/480 max relative RMS change %.6f (gate 0.05); 1/120->1/480 %.6f (reported)"
                  % (label, change_fine, change_coarse), flush=True)
    print("scenario | model | variant | FK | dt | final x,y m | dz RMS m | roll RMS deg | pitch RMS deg | "
          "final vx m/s | excursion RMS m | mean vxy m/s | status")
    for row in results:
        status = "complete" if row["completed"] else "DIVERGED at %.4fs" % row["diverged_at_s"]
        if row["second_half_rms_dz_roll_pitch_vx_vy_yawrate"] is None:
            print("%s | %s | %s | %s | 1/%d | %.4f,%.4f | N/A | N/A | N/A | N/A | N/A | N/A | %s" % (
                row["scenario"], row["model"], row.get("variant", "-"), row["froude_krylov"], round(1/row["dt"]),
                *row["final_xyz_m"][:2], status))
            continue
        z, r, p, *_ = row["second_half_rms_dz_roll_pitch_vx_vy_yawrate"]
        mean = row["second_half_mean_velocity_xy_mps"]
        print("%s | %s | %s | %s | 1/%d | %.4f,%.4f | %.5f | %.4f | %.4f | %.4f | %s | %s | %s" % (
            row["scenario"], row["model"], row.get("variant", "-"), row["froude_krylov"], round(1/row["dt"]),
            *row["final_xyz_m"][:2], z, r*180/math.pi, p*180/math.pi, row["final_velocity_mps"][0],
            "N/A" if row["horizontal_excursion_rms_m"] is None else "%.4f" % row["horizontal_excursion_rms_m"],
            "N/A" if mean is None else "%.4f,%.4f" % tuple(mean), status))
    failures = [label for label, gate in gates.items()
                if not gate["all_completed"] or gate["max_rms_change_240_480"] > .05]
    if failures:
        raise AssertionError("v2 CPU timestep response gate failed: %r" % failures)
    band_changes = {}
    bands_out = {}
    wave_batches = tuple([r for r in batch if r.get("hs", 0) > 0 and "current" not in r["name"]]
                         for batch in batches)
    for variant in PROFILE_VARIANTS:
        profile = variant_profile(variant, True)
        bands = {}
        for cutoff in (4., 8., 16.):
            bands[cutoff] = integrate_batches(wave_batches, "v2", 1/240, duration=20., cutoff=cutoff, nested=True,
                                              profile=profile)
        for lower in (4., 8.):
            worst = 0.
            for a, b in zip(bands[lower], bands[16.]):
                va = torch.tensor(a["second_half_rms_dz_roll_pitch_vx_vy_yawrate"][:3])
                vb = torch.tensor(b["second_half_rms_dz_roll_pitch_vx_vy_yawrate"][:3])
                worst = max(worst, float(((va-vb).abs()/vb.abs().clamp_min(1e-6)).max()))
            band_changes["%s %g->16fp" % (variant, lower)] = worst
            print("Nested nonlinear %s %g->16fp maximum heave/roll/pitch RMS relative change: %.9g"
                  % (variant, lower, worst), flush=True)
        if band_changes["%s 8->16fp" % variant] > .01 or not all(
                r["completed"] for group in bands.values() for r in group):
            raise AssertionError("nonlinear spectral band response gate failed for " + variant)
        bands_out[variant] = {str(k): v for k, v in bands.items()}
    print("Part 2: v2 anchors through HydroV2Runtime (calm water, closed forms and report values)", flush=True)
    anchor_rows = anchors()
    failures = []
    for a in anchor_rows:
        print(json.dumps(a), flush=True)
        key = (a["variant"], a["dt"])
        if abs(a["surge_60s_mps"]-a["surge_closed_form_mps"]) > .002*a["surge_closed_form_mps"]:
            failures.append(key+("surge",))
        if abs(a["surge_closed_form_mps"]-a["surge_report_mps"]) > .006:
            failures.append(key+("surge vs report",))
        if abs(a["spin_60s_rad_s"]-a["spin_closed_form_rad_s"]) > .005*a["spin_closed_form_rad_s"]:
            failures.append(key+("spin",))
        if a["spin_report_rad_s"] is not None and abs(a["spin_closed_form_rad_s"]-a["spin_report_rad_s"]) > .006:
            failures.append(key+("spin vs report",))
        for name in ("heave", "roll", "pitch"):
            closed = a[name+"_period_closed_s"]
            if closed is not None and a[name+"_period_s"] is not None and a[name+"_zeta"] < .5:
                if abs(a[name+"_period_s"]-closed) > .03*closed:
                    failures.append(key+(name+" period",))
                if abs(a[name+"_cycle_ratio"]-a[name+"_cycle_ratio_closed"]) > .05:
                    failures.append(key+(name+" decay ratio",))
    if failures:
        raise AssertionError("anchor checks failed: %r" % failures)
    output = {"assumptions": "COM-aligned free-body model, no contacts/engine caps; wave grids differ across"
                             " legacy/v2 (legacy 104 bins 0.04-1.60 Hz; v2 adaptive band); v2 starts at the"
                             " hydrostatic COM datum, legacy at its own box-law equilibrium; wave direction is"
                             " propagation direction (pi = head seas for a +x-moving hull; the zero-thrust A rows"
                             " use 0 = along +x, pi/2 = beam). E2: D5 registry profiles; variants blueboat_v2"
                             " zeta 0.1 / legacy 470 and wamv_vrx_v2_opt1, Froude-Krylov on/off.",
              "rms_definition": "second-half (10-20 s) RMS of [z-z_datum, roll, pitch, vx, vy, yaw rate]; for"
                                " diverged rows fields describe the last accepted state",
              "excursion_definition": "second-half RMS of the XY track about its least-squares line; mean"
                                      " velocity = slope of that line (world axes)",
              "duration_s": 20, "results": results, "nested_spectral": bands_out,
              "timestep_gates": gates, "max_nested_band_rms_change": band_changes,
              "anchors": anchor_rows}
    path = ROOT.parent/"CPU_SCENARIOS.json"
    path.write_text(json.dumps(output, indent=2)+"\n", encoding="ascii")
    print("PASS: free-body timestep gates, nonlinear band gates and runtime anchors for every profile variant. "
          "Results: CPU_SCENARIOS.json")


if __name__ == "__main__":
    main()
