"""Deterministic CPU spectral/load/linear-response convergence, no simulator.

Analytic independent-random-phase second moments avoid finite-run sampling
noise. This verifies the stationary small-motion force model, not finite-wave
nonlinear six-DOF behavior. Run with python -B and inspect SPECTRAL_REPORT.txt.

E2: profiles from the D5 registry via hydro_profiles.profile_for_vehicle
(BlueBoat v2 zeta 0.1 and legacy 470, WAM-V opt1 and opt2, Froude-Krylov on
and off), and two horizontal metrics for decision 7: the Froude-Krylov force
m*omega^2*exp(-k d) per unit filtered elevation (zero when the term is off)
and the linear hull horizontal velocity response (transfer 1 with the term,
D/|D + i omega m| per hull axis without it; linear damping only).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import torch

SHARED = Path(__file__).resolve().parents[2] / "tasks" / "_shared"
sys.path.insert(0, str(SHARED))


def load_file(name):
    spec = importlib.util.spec_from_file_location(name, SHARED / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_file("physics_v2")
load_file("restoring_v2")
hydro = load_file("hydro_v2")
load_file("thruster_mixer")
load_file("vehicles_v2")
load_file("journal_vehicle")
profiles = load_file("hydro_profiles")

REFERENCE_CUTOFF = 64
CUTOFFS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
CASES = (("W1", .075, 1.2), ("W2", .2, 1.8), ("W3", .375, 2.45),
         ("W4", .65, 3.15), ("Tp1_probe", .075, 1.0),
         ("Tp20_probe", .075, 20.0))
GAMMAS = (1.0, 3.3, 5.0)
HEADINGS = (0.0, 45.0, 90.0)
METRICS = ("elevation_m", "slope_norm", "orbital_horizontal_mps",
           "orbital_vertical_mps", "heave_load_n", "roll_load_nm",
           "pitch_load_nm", "heave_response_m", "roll_response_rad",
           "pitch_response_rad", "fk_force_n", "horizontal_velocity_response_mps")
# E2: response metrics are no longer contiguous (the horizontal velocity
# response is appended after the Froude-Krylov load); an index tuple replaces
# D3's slice(7, 10) with the same meaning plus the new response metric.
RESPONSE_SLICE = (7, 8, 9, 11)
DTYPE = torch.float64


def spectrum_shape(x, gamma):
    """Dimensionless JONSWAP shape; normalization sets m0=(Hs/4)^2.

    x=f/fp; sigma=.07 below peak and .09 above; gamma enhancement is the
    standard exp(-(x-1)^2/(2 sigma^2)) exponent. No random realizations drawn.
    """
    sigma = torch.where(x <= 1, .07, .09)
    peak = torch.exp(-.5 * ((x - 1) / sigma).square())
    return x.pow(-5) * torch.exp(-1.25 * x.pow(-4)) * gamma**peak


def response_squared(spec, omega, k, heading_deg):
    """Squared transfer from filtered elevation to each requested quantity.

    eta=Re(a exp(-i omega t)), u_z=-i omega exp(-k depth)*eta.
    Heave: (K-m omega^2-i C omega) z = (K-i C omega exp(-k depth)) eta.
    Roll/pitch, rotational_relative_motion=True (hydro_v2 default): damping
    acts on (hull rate - surface-normal rate), so the wave also drives the
    damper: (K-I omega^2-i C omega) angle = (K-i C omega)*(+/-slope).
    With rotational_relative_motion=False only K*slope drives the hull.
    Linear coefficients only; quadratic angular damping is not linearized.
    Loads are incident excitation (including known water-relative damping),
    not net force after the moving-hull restoring/damping response.
    """
    cosine, sine = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    cosine = 0.0 if abs(cosine) < 1e-12 else cosine
    sine = 0.0 if abs(sine) < 1e-12 else sine
    attenuation = torch.exp(-k * spec.orbital_depth_m)
    orbital2 = (omega * attenuation).square()
    kh = spec.water_density * spec.gravity * spec.waterplane_m2
    ch = spec.linear_damping[2]
    heave_load2 = kh**2 + (ch * omega * attenuation).square()
    relative = 1.0 if spec.rotational_relative_motion else 0.0
    roll_load2 = (spec.roll_stiffness_nm**2 + relative * (spec.angular_linear_damping[0] * omega).square()) \
        * (k * sine).square()
    pitch_load2 = (spec.pitch_stiffness_nm**2 + relative * (spec.angular_linear_damping[1] * omega).square()) \
        * (k * cosine).square()
    heave_den = (kh - spec.mass_kg * omega.square()).square() + (ch * omega).square()
    # E2 horizontal: FK force amplitude m*omega^2*exp(-k d) per unit filtered
    # elevation (both horizontal components together), and the hull velocity
    # response per hull axis |(D + i omega m fk)/(D + i omega m)| times the
    # orbital speed omega*exp(-k d), fk = 1 with the term, 0 without.
    fk = 1.0 if spec.froude_krylov_horizontal else 0.0
    inertia_w2 = (spec.mass_kg * omega).square()
    fk_load2 = fk * inertia_w2 * orbital2
    surge_d, sway_d = spec.linear_damping[0], spec.linear_damping[1]
    surge_t2 = (surge_d**2 + fk * inertia_w2) / (surge_d**2 + inertia_w2)
    sway_t2 = (sway_d**2 + fk * inertia_w2) / (sway_d**2 + inertia_w2)
    horizontal2 = orbital2 * (cosine**2 * surge_t2 + sine**2 * sway_t2)
    roll_den = (spec.roll_stiffness_nm - spec.inertia_kgm2[0][0] * omega.square()).square()
    roll_den += (spec.angular_linear_damping[0] * omega).square()
    pitch_den = (spec.pitch_stiffness_nm - spec.inertia_kgm2[1][1] * omega.square()).square()
    pitch_den += (spec.angular_linear_damping[1] * omega).square()
    return torch.stack((torch.ones_like(omega), k.square(), orbital2, orbital2,
                        heave_load2, roll_load2, pitch_load2,
                        heave_load2 / heave_den, roll_load2 / roll_den,
                        pitch_load2 / pitch_den, fk_load2, horizontal2), dim=-1)


def calculate(spec, dx=.0025, reference_cutoff=REFERENCE_CUTOFF):
    """Midpoint quadrature with nested fixed cutoffs and fixed normalization."""
    x = torch.arange(.2 + dx / 2, reference_cutoff, dx, dtype=DTYPE,
                     device="cpu")
    results = {}
    for label, hs, tp in CASES:
        omega = 2 * math.pi * x / tp
        k = omega.square() / spec.gravity
        for gamma in GAMMAS:
            raw = spectrum_shape(x, gamma) * dx
            variance = raw / raw.sum() * (hs / 4)**2
            for heading in HEADINGS:
                beta = math.radians(heading)
                transfer = hydro.footprint_transfer(
                    (k * math.cos(beta))[None], (k * math.sin(beta))[None],
                    torch.zeros(1, dtype=DTYPE), spec.length_m, spec.beam_m)[0]
                gains2 = response_squared(spec, omega, k, heading)
                contributions = variance[:, None] * transfer[:, None].square() * gains2
                row = {}
                for cutoff in CUTOFFS:
                    mask = x < cutoff
                    capture = float(raw[mask].sum() / raw.sum())
                    fixed = contributions[mask].sum(0).sqrt()
                    normalized = fixed / math.sqrt(capture)
                    row[cutoff] = {"capture": capture, "fixed": fixed,
                                   "normalized": normalized}
                if reference_cutoff not in row:
                    row[reference_cutoff] = {"capture": 1.0,
                        "fixed": contributions.sum(0).sqrt(),
                        "normalized": contributions.sum(0).sqrt()}
                adaptive_ratio = hydro.required_high_frequency_hz(spec, 1/tp) * tp
                assert adaptive_ratio < reference_cutoff, "Increase the reference band"
                mask = x < adaptive_ratio
                capture = float(raw[mask].sum() / raw.sum())
                fixed = contributions[mask].sum(0).sqrt()
                row["adaptive"] = {"capture":capture, "fixed":fixed,
                    "normalized":fixed/math.sqrt(capture), "cutoff_fp":adaptive_ratio}
                results[(label, gamma, heading)] = row
    return results


def relative_error(value, reference):
    return torch.where(reference > 1e-18, (value-reference).abs() /
                       reference.clamp_min(1e-18), torch.zeros_like(reference))


def worst_for_cutoff(results, cutoff, variant="fixed", group=None):
    worst, worst_key, worst_metric = -1.0, None, None
    indices = list(range(len(METRICS))) if group is None else list(group)
    for key, row in results.items():
        errors = relative_error(row[cutoff][variant], row[REFERENCE_CUTOFF]["fixed"])[indices]
        error, index = errors.max(0)
        if error.item() > worst:
            worst, worst_key = error.item(), key
            worst_metric = METRICS[indices[int(index)]]
    return worst, worst_key, worst_metric


def compare_quadrature(coarse, fine):
    maximum = 0.0
    for key, row in coarse.items():
        for cutoff in CUTOFFS + ("adaptive",):
            for variant in ("fixed", "normalized"):
                maximum = max(maximum, relative_error(
                    row[cutoff][variant], fine[key][cutoff][variant]).max().item())
    return maximum


def audit_properties(results, spec):
    """Fail on nonfinite moments, nonmonotone tails or filter implementation drift."""
    for row in results.values():
        previous = torch.zeros(len(METRICS), dtype=DTYPE)
        for cutoff in CUTOFFS:
            value = row[cutoff]["fixed"]
            assert torch.isfinite(value).all() and (value >= 0).all()
            assert (value >= previous - 1e-14).all(), "Tail variance must be monotone"
            previous = value
            assert 0 < row[cutoff]["capture"] <= 1.0 + 1e-12
    k = torch.logspace(-3, 2, 500, dtype=DTYPE)
    beta = torch.linspace(0, math.pi / 2, 500, dtype=DTYPE)
    h = hydro.footprint_transfer((k*torch.cos(beta))[None], (k*torch.sin(beta))[None],
                                torch.zeros(1, dtype=DTYPE), spec.length_m, spec.beam_m)[0]
    bound = torch.exp(-min(spec.length_m, spec.beam_m)**2 * k.square() / 12)
    assert (h.square() <= bound + 1e-15).all(), "Gaussian tail bound violated"


def report(spec, results, refinement_error, extended, production_spacing_error, name="", source=""):
    lines = ["D3 SPECTRAL, LOAD AND LINEAR RESPONSE CONVERGENCE (CPU): VESSEL PROFILE %s" % name,
             "Method: deterministic analytic random-phase variances; float64; no Isaac/GPU.",
             "All 54 cells: W1-W4 midpoints + Tp=1,20 s probes; gamma=1,3.3,5;",
             "incident headings 0,45,90 deg relative to hull x. Hs/Tp values:"]
    lines.extend("  %s Hs=%.6g m Tp=%.6g s" % case for case in CASES)
    lines += ["", "Spectrum and normalization:",
        "  JONSWAP S(x) proportional x^-5 exp(-1.25*x^-4) gamma^r, x=f/fp.",
        "  Sigma=0.07 below peak, 0.09 above. Lower band edge=0.2 fp.",
        "  Midpoint spacing df/fp=0.0025; checked against 0.00125.",
        "  FIXED normalization uses the [0.2,64]fp integral for every cutoff.",
        "  RENORM reproduces per-cutoff target Hs normalization; both compare",
        "  with the same filtered 64fp reference, so hidden rescaling is disclosed.",
        "  A 16fp reference is not sufficient for long-period probes (Tp=20 s),",
        "  so cutoffs up to 64fp are reported and 64fp is checked against 128fp.",
        "  Maximum quadrature-refinement relative RMS change = %.9g%%" % (100*refinement_error),
        "  Production spacings df/fp=0.01 and 0.0125 (C2 SeaState v2 bound) vs 0.00125:",
        "  worse of the two, max RMS change = %.9g%%" %
        (100*production_spacing_error),
        "", "Model and coefficients:",
        "  Source: tasks/_shared/hydro_profiles.py (%s) and hydro_v2.footprint_transfer;" % source,
        "  rotational_relative_motion=%s (roll/pitch damping on hull-minus-surface rate)." %
        spec.rotational_relative_motion,
        "  m=%.9g kg; Iroll=%.9g, Ipitch=%.9g kg m2." % (spec.mass_kg,
            spec.inertia_kgm2[0][0], spec.inertia_kgm2[1][1]),
        "  L=%.9g, B=%.9g m; orbital sampling depth=%.9g m." % (
            spec.length_m, spec.beam_m, spec.orbital_depth_m),
        "  Heave K=rho*g*A=%.9g N/m; C=%.9g N s/m." % (
            spec.water_density*spec.gravity*spec.waterplane_m2,spec.linear_damping[2]),
        "  Roll/pitch K=%.9g/%.9g N m/rad, C=%.9g/%.9g N m s/rad." % (
            spec.roll_stiffness_nm,spec.pitch_stiffness_nm,
            spec.angular_linear_damping[0],spec.angular_linear_damping[1]),
        "  Loads: incident hydrostatic + vertical-water damping force, incident",
        "  roll/pitch wave-slope moment. Response: stationary linear heave/roll/pitch.",
        "  Vertical orbital motion enters heave in quadrature with elevation:",
        "  |z/eta_filtered|^2=(Kh^2+(Ch*w*exp(-k*d))^2)/((Kh-m*w^2)^2+(Ch*w)^2).",
        "  Roll/pitch |angle/eta_filtered|^2=(K^2+r*(C*w)^2)*(k*direction_component)^2/",
        "  ((K-I*w^2)^2+(C*w)^2), r=1 relative-motion damping, r=0 inertial damping.",
        "  E2 horizontal (froude_krylov_horizontal=%s): FK force |F/eta_f|=m*w^2*exp(-k*d);" %
        spec.froude_krylov_horizontal,
        "  hull velocity |v_axis/u_axis|^2=(D^2+fk*(w*m)^2)/(D^2+(w*m)^2), fk=1 on, 0 off,",
        "  D=linear surge %.9g / sway %.9g N s/m (quadratic drag not linearized)." % (
            spec.linear_damping[0], spec.linear_damping[1]),
        "  No extra 100 N/m or 200 N m slope terms.",
        "", "Gaussian footprint and proof:",
        "  H=exp(-[(k_long*L)^2+(k_cross*B)^2]/24); rectangular second moments.",
        "  H^2<=exp(-min(L,B)^2*k^2/12), and deep-water k=(2*pi*f)^2/g.",
        "  Unfiltered JONSWAP slope variance has S(f)*k^2 ~ const/f, hence",
        "  diverges logarithmically. Filtered slope/load tail is bounded by",
        "  const*integral_F^infinity f^-1*exp(-a*f^4) df, a>0, which converges.",
        "  Every included velocity/load/linear-response transfer has at most",
        "  polynomial frequency growth; the same exponential bound proves all",
        "  reported filtered second moments converge as upper band tends infinity.",
        "  The filter is a declared smooth hull-average approximation, not a measured RAO."]
    for heading in HEADINGS:
        beta = math.radians(heading)
        effective = math.sqrt((spec.length_m*math.cos(beta))**2 +
                              (spec.beam_m*math.sin(beta))**2)
        k3 = math.sqrt(12*math.log(2))/effective
        f3 = math.sqrt(spec.gravity*k3)/(2*math.pi)
        lines.append("  Heading %2g deg: amplitude -3 dB frequency %.6g Hz." % (heading,f3))
    lines += ["", "Worst errors against 64fp across ALL cells (percent RMS):",
        "cut/fp  min_elev_capture  fixed_all  fixed_response  renorm_all  renorm_response"]
    pass_response, pass_all = [], []
    for cutoff in CUTOFFS:
        capture = min(row[cutoff]["capture"] for row in results.values())
        all_fixed = worst_for_cutoff(results,cutoff)[0]
        response_fixed = worst_for_cutoff(results,cutoff,group=RESPONSE_SLICE)[0]
        all_norm = worst_for_cutoff(results,cutoff,"normalized")[0]
        response_norm = worst_for_cutoff(results,cutoff,"normalized",RESPONSE_SLICE)[0]
        lines.append("%6g %17.9f %10.6f %15.6f %11.6f %16.6f" %
                     (cutoff,100*capture,100*all_fixed,100*response_fixed,
                      100*all_norm,100*response_norm))
        if max(response_fixed,response_norm)<.01:
            pass_response.append(cutoff)
        if max(all_fixed,all_norm)<.01 and capture>=.99:
            pass_all.append(cutoff)
    lines += ["Smallest tested cutoff with <1%% all response RMS errors in BOTH normalizations: %s fp" %
              (pass_response[0] if pass_response else "NONE"),
              "Smallest tested cutoff with <1%% ALL metrics and >=99%% raw elevation capture: %s fp" %
              (pass_all[0] if pass_all else "NONE")]
    for cutoff in (3,8):
        value,key,metric=worst_for_cutoff(results,cutoff)
        lines.append("  %gfp worst fixed metric: %.9g%%, %s, %s" %
                     (cutoff,100*value,key,metric))
    adaptive_capture=min(row["adaptive"]["capture"] for row in results.values())
    adaptive_fixed=worst_for_cutoff(results,"adaptive")[0]
    adaptive_normalized=worst_for_cutoff(results,"adaptive","normalized")[0]
    adaptive_fixed_response=worst_for_cutoff(results,"adaptive",group=RESPONSE_SLICE)[0]
    adaptive_normalized_response=worst_for_cutoff(results,"adaptive","normalized",RESPONSE_SLICE)[0]
    lines += ["", "PRIMARY ADAPTIVE BAND POLICY (production helper exercised directly):",
        "  f_hi=max(4*fp, sqrt(g*sqrt(12*ln(1e8))/min(L,B))/(2*pi)).",
        "  Worst-heading H^2 at or above the high edge <=1e-8.",
        "  At least 256 components and df<=0.01fp; period-dependent component",
        "  counts must be allowed instead of silently retaining a short-wave cutoff.",
        "  Minimum raw elevation-variance capture: %.9g%%" % (100*adaptive_capture),
        "  Worst ALL RMS error vs64fp: fixed %.9g%%, renormalized %.9g%%." %
        (100*adaptive_fixed,100*adaptive_normalized),
        "  Worst RESPONSE RMS error vs64fp: fixed %.9g%%, renormalized %.9g%%." %
        (100*adaptive_fixed_response,100*adaptive_normalized_response)]
    for label,hs,tp in CASES:
        ratio=results[(label,3.3,45.0)]["adaptive"]["cutoff_fp"]
        lines.append("  %s: f_hi=%.9g Hz = %.9g fp" % (label,ratio/tp,ratio))
    lines.append("  PASS: >=99% capture and <1% RMS errors across all 54 cells in BOTH normalizations.")
    lines += ["", "Additional tail checks (normalization fixed at 128fp):"]
    tail_max=0.0
    for lower,upper in ((8,16),(16,32),(32,64),(64,128)):
        pair_max=0.0
        pair_response=0.0
        for key,row in extended.items():
            errors=relative_error(row[lower]["fixed"],row[upper]["fixed"])
            pair_max=max(pair_max,errors.max().item())
            pair_response=max(pair_response,errors[list(RESPONSE_SLICE)].max().item())
        lines.append("  %g->%gfp: worst all RMS %.9g%%, response RMS %.9g%%; %s for <1%% all metrics" %
                     (lower,upper,100*pair_max,100*pair_response,
                      "PASS" if pair_max < .01 else "FAIL"))
        if upper==128:
            tail_max=pair_max
    primary={key:row for key,row in results.items() if key[0] in ("W1","W2","W3","W4")}
    lines += ["", "W1-W4 ONLY: worst relative RMS errors against 64fp (percent):",
              "cut/fp fixed_all fixed_response renorm_all renorm_response"]
    for cutoff in (2,3,4,6,8,12,16):
        values=(worst_for_cutoff(primary,cutoff)[0],
                worst_for_cutoff(primary,cutoff,group=RESPONSE_SLICE)[0],
                worst_for_cutoff(primary,cutoff,"normalized")[0],
                worst_for_cutoff(primary,cutoff,"normalized",RESPONSE_SLICE)[0])
        lines.append("%6g %s" % (cutoff," ".join("%.9g"%(100*v) for v in values)))
    lines += ["", "Representative physical statistics at 64fp (gamma=3.3, heading=45 deg):",
        "case " + " ".join(METRICS)]
    for label,_,_ in CASES:
        values=results[(label,3.3,45.0)][REFERENCE_CUTOFF]["fixed"]
        lines.append(label+" "+" ".join("%.9g" % x for x in values.tolist()))
    # Quantify logarithmic unfiltered divergence without a rescaling confound.
    x=torch.arange(.2+.0025/2,128,.0025,dtype=DTYPE)
    raw=spectrum_shape(x,3.3)*.0025
    omega=2*math.pi*x/1.2
    k=omega.square()/spec.gravity
    variance=raw/raw[x<16].sum()*(.075/4)**2
    unfiltered=[float((variance[x<cut]*k[x<cut].square()).sum()) for cut in (16,32,64,128)]
    increments=[b-a for a,b in zip(unfiltered[:-1],unfiltered[1:])]
    lines += ["", "Unfiltered W1 slope VARIANCE, normalization fixed at 16fp:",
        "  cutoffs 16/32/64/128fp: "+" / ".join("%.10g"%v for v in unfiltered),
        "  successive doubling increments: "+" / ".join("%.10g"%v for v in increments),
        "  Nonzero constant doubling increments demonstrate the logarithmic tail.",
        "", "Acceptance and limits:",
        "  PASS: finite nonnegative statistics, monotone fixed-normalization tail",
        "  accumulation, Gaussian bound, and midpoint-grid refinement assertions.",
        "  Convergence is not physical calibration or a proof of validity at W4.",
        "  This report does not certify production coarse component spacing, random",
        "  finite episodes, moving-heading filter behavior, volume clipping,",
        "  numerical time stepping, slamming/emergence/capsize, radiation or diffraction.",
        "  Those remain UNVERIFIED beyond the separate CPU nonlinear integrator and",
        "  require D9/GPU/physical response tests. Vessel values: D5 v2 registry",
        "  (tasks/_shared/vehicles_v2.py) through hydro_profiles.profile_for_vehicle."]
    assert refinement_error < .001, "Quadrature is not sufficiently resolved"
    assert production_spacing_error < .001, "Suggested production frequency spacing is unresolved"
    assert tail_max < .01, "64fp reference does not converge to 128fp within 1%"
    assert adaptive_capture >= .99 and max(adaptive_fixed,adaptive_normalized)<.01
    assert pass_all, "No tested cutoff meets declared metric acceptance"
    assert max(increments)/min(increments) < 1.001
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",type=Path,default=Path(__file__).resolve().parents[3]/"SPECTRAL_REPORT.txt")
    args=parser.parse_args()
    torch.set_num_threads(1)
    pv=profiles.profile_for_vehicle
    cases=((pv("blueboat_v2"),"profile_for_vehicle('blueboat_v2') [zeta 0.1, FK on]"),
           (pv("blueboat_v2",rollpitch_damping="legacy_470"),
            "profile_for_vehicle('blueboat_v2', rollpitch_damping='legacy_470') [FK on]"),
           (pv("blueboat_v2",froude_krylov_horizontal=False),
            "profile_for_vehicle('blueboat_v2', froude_krylov_horizontal=False) [zeta 0.1, FK OFF]"),
           (pv("blueboat_v2",rollpitch_damping="legacy_470",froude_krylov_horizontal=False),
            "profile_for_vehicle('blueboat_v2', rollpitch_damping='legacy_470', froude_krylov_horizontal=False)"),
           (pv("wamv_vrx_v2_opt1"),"profile_for_vehicle('wamv_vrx_v2_opt1') [FK on]"),
           (pv("wamv_vrx_v2_opt1",froude_krylov_horizontal=False),
            "profile_for_vehicle('wamv_vrx_v2_opt1', froude_krylov_horizontal=False) [FK OFF]"),
           (pv("wamv_vrx_v2_opt2"),"profile_for_vehicle('wamv_vrx_v2_opt2') [FK on]"),
           (pv("blueboat_v2",rotational_relative_motion=False),
            "profile_for_vehicle('blueboat_v2', rotational_relative_motion=False) [alternative]"))
    outputs=[]
    with torch.no_grad():
        for profile,source in cases:
            spec=profile.hydro
            results=calculate(spec)
            audit_properties(results,spec)
            fine=calculate(spec,dx=.00125)
            production_spacing=calculate(spec,dx=.01)
            c2_spacing=calculate(spec,dx=.0125)
            extended=calculate(spec,reference_cutoff=128.0)
            c2_error=compare_quadrature(c2_spacing,fine)
            assert c2_error < .001, "C2 SeaState v2 spacing df/fp=0.0125 is unresolved"
            outputs.append(report(spec,results,compare_quadrature(results,fine),extended,
                                  max(compare_quadrature(production_spacing,fine),c2_error),
                                  profile.name+(" (inertial rotational damping)" if not spec.rotational_relative_motion else "")
                                  +" [roll/pitch damping %s, FK %s]" % (profile.sources["options"]["rollpitch_damping"],
                                  "on" if spec.froude_krylov_horizontal else "OFF"),
                                  source))
    output=("\n"+"="*78+"\n\n").join(outputs)
    args.report.write_text(output,encoding="ascii")
    print(output)


if __name__=="__main__":
    main()
