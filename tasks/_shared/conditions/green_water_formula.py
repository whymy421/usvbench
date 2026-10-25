"""Wave-following per-cycle deck-wetness probability (G10 item 1; D9 port).

Port of D9's wave-following wetness screen (D9 work/scripts/vessel_envelope.py:
``relative_motion`` :196-243, ``exceedance`` :246-260, ``midpoint_spectrum``
:464-480, ``direction_nodes`` :178-183; method text DOSE_DESIGN.md section 3,
lines 51-61) as a deterministic function of (wave dose, vessel record):

1. Quasi-static wave following. The hull follows the least-squares plane of
   the free surface over its waterplane (heave = waterplane mean elevation,
   trim / heel = first moment / second moment). Long-wave hydrostatic limit
   (natural periods well below the wave periods; D9 DOSE_DESIGN.md:53).
2. Relative vertical motion at each declared deck-edge point p:
   eta(p) - (h + theta_x (p_x - x_c) + theta_y (p_y - y_c)), averaged over the
   uniform direction law of the preset (Gauss-Legendre, 48 nodes, D9 order).
3. Per-cycle Rayleigh exceedance of the point's calm freeboard f_p:
   P = exp(-f_p^2 / (2 m0_rel)). The cell value is the WORST point and the
   WORST mean heading. Criterion: NORDFORSK (1987) deck wetness 0.05 per
   cycle (Zu, Garme and Rosen 2024, Ocean Eng. 297:116785: Table 6, the
   summary of the NORDFORSK 1987 standards; the same 0.05 is the Karppinen
   1987 standard adopted by NORDFORSK in Table 4, section 4.1, where the
   source cautions against applying the slamming and deck-wetness standards
   to fast small craft; checked against the open-access text, G28; D9
   vessel_envelope.py:22-32).

Differences from D9 (all deliberate, all stamped in ``inputs``):
  * vessel inputs come from the FINAL registry (vehicles_v2 / vehicles_v2_data)
    through green_water.wetness_geometry: the waterplane strips and the
    criterion points are the declared deck-edge points of the simulated
    detection (one point set for formula and simulation), each with its own
    calm freeboard (point height above the calm water in the simulator);
    D9 used provisional inputs (BlueBoat 1.2 x 0.9 m rectangle, freeboard
    0.15 m; WAM-V strips 4.9 x 0.376 m at y = +/-1.03 m, freeboard 0.29 m);
  * the spectrum is the REALISED band of the simulator: [0.2, hi] fp with hi
    raised to the vessel's hydro_v2 requirement and the component count
    raised for the group-repeat rule, exactly as HydroV2ConditionLayer.
    bind_runtime and SeaState(physics_version=2) do (``resolved_band``);
    D9 used [0.2, 4.0] fp and 1024 components for every vessel;
  * headings: the compass-8 set 0..315 deg (D9: 0..180 deg; identical for a
    port/starboard-symmetric point set, which both registry sets are; tested);
  * the waterplane integrals are evaluated in closed form for rectangular
    strips (exact); D9 used 32 x 8 Gauss-Legendre nodes per strip (exact to
    rounding for these wavenumbers). ``relative_motion_quadrature`` keeps
    D9's quadrature as the reference; the tests check both agree.

Pure numpy; no torch, no simulator. Deterministic: the same inputs give the
same float64 result on every platform up to libm rounding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

FORMULA_VERSION = "usvbench.green_water.formula/1"
CRITERION = "NORDFORSK per-cycle wetness, limit 0.05"
NORDFORSK_WETNESS_PER_CYCLE = 0.05
GRAVITY = 9.81                      # D9 vessel_envelope.py:48; HydroSpec / registry GRAVITY
HEADINGS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
D9_HEADINGS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0)   # vessel_envelope.py:54
DIRECTION_ORDER = 48                                 # vessel_envelope.py:178 default
METHOD = ("quasi-static wave following (least-squares plane over the waterplane), relative vertical "
          "motion at the declared deck-edge points, uniform direction law (Gauss-Legendre 48), "
          "per-cycle Rayleigh exceedance of each point's calm freeboard; worst point and worst heading "
          "of the compass-8 set (D9 vessel_envelope.py relative_motion/exceedance)")


@dataclass(frozen=True)
class Strip:
    """Rectangular waterplane strip centred at (cx, cy) (body frame, COM origin), length along x."""
    cx: float
    cy: float
    length: float
    width: float


@dataclass(frozen=True)
class WetnessGeometry:
    """Formula inputs of one vessel: waterplane strips and criterion points with freeboards."""
    vessel: str
    waterplane: tuple            # (Strip, ...)
    points: tuple                # ((label, x_m, y_m, freeboard_m), ...) body frame, COM origin
    sources: tuple = ()          # ((field, citation), ...)

    def as_dict(self):
        return {"vessel": self.vessel,
                "waterplane_strips_m": [asdict(s) for s in self.waterplane],
                "points": [{"label": p[0], "x_m": p[1], "y_m": p[2], "calm_freeboard_m": p[3]}
                           for p in self.points],
                "sources": {k: v for k, v in self.sources}}


# ------------------------------------------------------------- spectrum --
def midpoint_spectrum(hs, tp, gamma, band_fp=(0.2, 4.0), n=1024):
    """Float64 mirror of C2 SeaState v2's grid and Hs renormalisation.

    Copied from D9 vessel_envelope.py:464-480 (itself a mirror of
    sea_state.py SeaState.__init__ / resample / _jonswap, physics_version 2):
    n = max(n, 256, ceil((hi - lo) / 0.0125)); midpoint grid over [lo, hi] fp;
    JONSWAP sigma 0.07 / 0.09; variance renormalised so 4 sqrt(m0) = Hs.
    Returns (freqs_hz, component variance a_i^2 / 2).
    """
    lo, hi = band_fp
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


def direction_nodes(heading_rad, spread_rad, order=DIRECTION_ORDER):
    """Nodes/weights of the uniform law on [heading - spread, heading + spread] (D9 :178-183)."""
    if spread_rad == 0:
        return np.array([heading_rad]), np.ones(1)
    x, w = np.polynomial.legendre.leggauss(order)
    return heading_rad + spread_rad * x, w / 2.0


# ------------------------------------------------------- waterplane fit --
def _sinc(a):
    """sin(a)/a, exactly 1 at 0."""
    out = np.ones_like(a)
    nz = a != 0.0
    out[nz] = np.sin(a[nz]) / a[nz]
    return out


def _first_moment_factor(k, half):
    """(1/(2 half)) * int_{-half}^{half} s exp(i k s) ds / i  (real), stable for small k*half.

    int s e^{iks} ds = 2i (sin(k a) - k a cos(k a)) / k^2 = i 2 a^2 (sin x - x cos x) / x^2.
    """
    x = k * half
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = half * (xs / 3.0 - xs ** 3 / 30.0 + xs ** 5 / 840.0)
    xl = x[~small]
    out[~small] = half * (np.sin(xl) - xl * np.cos(xl)) / xl ** 2
    return out


def _waterplane(geometry):
    strips = geometry.waterplane
    area = sum(s.length * s.width for s in strips)
    xc = sum(s.length * s.width * s.cx for s in strips) / area
    yc = sum(s.length * s.width * s.cy for s in strips) / area
    iyy = sum(s.length * s.width * ((s.cx - xc) ** 2 + s.length ** 2 / 12.0) for s in strips)
    ixx = sum(s.length * s.width * ((s.cy - yc) ** 2 + s.width ** 2 / 12.0) for s in strips)
    return area, xc, yc, iyy, ixx


def _plane_fit(geometry, kx, ky):
    """Closed-form (h, theta_x, theta_y) of the waterplane LS plane per component (complex)."""
    area, xc, yc, iyy, ixx = _waterplane(geometry)
    h = np.zeros(kx.shape, complex)
    tx = np.zeros(kx.shape, complex)
    ty = np.zeros(kx.shape, complex)
    for s in geometry.waterplane:
        a, b = 0.5 * s.length, 0.5 * s.width
        sx, sy = _sinc(kx * a), _sinc(ky * b)
        centre = np.exp(1j * (kx * s.cx + ky * s.cy))
        full = s.length * s.width * centre
        h += full * sx * sy
        # int (x - xc) e^{ik.x} dA = full * sy * [(cx - xc) sx + i mx]: the s-integral over the
        # strip, int_{-a}^{a} s e^{iks} ds = i 2a mx with mx = _first_moment_factor, and 2a = length
        mx = _first_moment_factor(kx, a)
        my = _first_moment_factor(ky, b)
        tx += full * sy * ((s.cx - xc) * sx + 1j * mx)
        ty += full * sx * ((s.cy - yc) * sy + 1j * my)
    return h / area, tx / iyy, ty / ixx, xc, yc


def relative_motion_m0(geometry, freqs_hz, energy_m2, heading_deg, spread_rad,
                       order=DIRECTION_ORDER, gravity=GRAVITY):
    """Following relative-motion variance m0 (m^2) at every point, one mean heading (closed form)."""
    f = np.asarray(freqs_hz, float)
    e = np.asarray(energy_m2, float)
    omega = 2.0 * math.pi * f
    k = omega ** 2 / gravity
    pts = np.array([[p[1], p[2]] for p in geometry.points], float)
    th, tw = direction_nodes(math.radians(heading_deg), spread_rad, order)
    m0 = np.zeros(len(pts))
    for t, wt in zip(th, tw):
        kx, ky = k * math.cos(t), k * math.sin(t)
        h, tx, ty, xc, yc = _plane_fit(geometry, kx, ky)
        local = np.exp(1j * (np.outer(kx, pts[:, 0]) + np.outer(ky, pts[:, 1])))
        plane = h[:, None] + tx[:, None] * (pts[:, 0] - xc) + ty[:, None] * (pts[:, 1] - yc)
        m0 += wt * (e[:, None] * np.abs(local - plane) ** 2).sum(axis=0)
    return m0


def relative_motion_quadrature(geometry, freqs_hz, energy_m2, heading_deg, spread_rad,
                               order=DIRECTION_ORDER, gravity=GRAVITY, nx=32, ny=8):
    """D9's Gauss-Legendre path (vessel_envelope.py:165-175, 196-232), reference for the tests."""
    f = np.asarray(freqs_hz, float)
    e = np.asarray(energy_m2, float)
    omega = 2.0 * math.pi * f
    k = omega ** 2 / gravity
    xs, wx = np.polynomial.legendre.leggauss(nx)
    ys, wy = np.polynomial.legendre.leggauss(ny)
    nodes, wts = [], []
    for s in geometry.waterplane:
        X, Y = np.meshgrid(s.cx + 0.5 * xs * s.length, s.cy + 0.5 * ys * s.width, indexing="ij")
        nodes.append(np.c_[X.ravel(), Y.ravel()])
        wts.append(np.outer(0.5 * wx * s.length, 0.5 * wy * s.width).ravel())
    nodes, w = np.vstack(nodes), np.concatenate(wts)
    area = w.sum()
    xc, yc = (w @ nodes[:, 0]) / area, (w @ nodes[:, 1]) / area
    dx, dy = nodes[:, 0] - xc, nodes[:, 1] - yc
    iyy, ixx = w @ dx ** 2, w @ dy ** 2
    pts = np.array([[p[1], p[2]] for p in geometry.points], float)
    th, tw = direction_nodes(math.radians(heading_deg), spread_rad, order)
    m0 = np.zeros(len(pts))
    for t, wt in zip(th, tw):
        kx, ky = k * math.cos(t), k * math.sin(t)
        phase = np.exp(1j * (np.outer(kx, nodes[:, 0]) + np.outer(ky, nodes[:, 1])))
        h = phase @ w / area
        theta_x = phase @ (w * dx) / iyy
        theta_y = phase @ (w * dy) / ixx
        local = np.exp(1j * (np.outer(kx, pts[:, 0]) + np.outer(ky, pts[:, 1])))
        plane = h[:, None] + theta_x[:, None] * (pts[:, 0] - xc) + theta_y[:, None] * (pts[:, 1] - yc)
        m0 += wt * (e[:, None] * np.abs(local - plane) ** 2).sum(axis=0)
    return m0


def exceedance(m0, threshold_m):
    """Per-cycle Rayleigh probability exp(-f^2 / (2 m0)) (D9 vessel_envelope.py:246-260, no velocity term)."""
    if m0 <= 0:
        return 0.0
    return math.exp(-threshold_m ** 2 / (2.0 * m0))


def wetness(geometry, freqs_hz, energy_m2, spread_rad, headings_deg=HEADINGS_DEG,
            order=DIRECTION_ORDER, gravity=GRAVITY, quadrature=False):
    """Worst-heading, worst-point per-cycle wetness probability of one spectrum.

    Ties (symmetric headings or points equal to rounding, relative 1e-12) resolve to the FIRST
    heading / point in declaration order, so the stamped labels do not depend on libm rounding.
    """
    per_heading = []
    motion = relative_motion_quadrature if quadrature else relative_motion_m0
    for hd in headings_deg:
        m0 = motion(geometry, freqs_hz, energy_m2, hd, spread_rad, order=order, gravity=gravity)
        probs = [exceedance(float(m), p[3]) for m, p in zip(m0, geometry.points)]
        j = _first_max(probs)
        per_heading.append({"heading_deg": float(hd), "probability": probs[j],
                            "point": geometry.points[j][0], "rel_rms_m": math.sqrt(float(m0[j])),
                            "_m0": float(m0[j])})
    best = per_heading[_first_max([h["probability"] for h in per_heading])]
    worst = {"probability": best["probability"], "heading_deg": best["heading_deg"], "point": best["point"],
             "m0_m2": best["_m0"]}
    for h in per_heading:
        del h["_m0"]
    worst["per_heading"] = per_heading
    return worst


def _first_max(values):
    top = max(values)
    for j, value in enumerate(values):
        if value >= top * (1.0 - 1e-12):
            return j
    return 0


# ----------------------------------------------------- realised wave band --
def resolved_band(waves, hydro):
    """(lo, hi, n_components) the simulator realises for a schema-3 wave dose and hull.

    Mirrors HydroV2ConditionLayer.bind_runtime (hydro_v2_layer.py:240-253:
    required = hydro_v2.required_high_frequency_hz(hydro, fp) * Tp; hi raised
    to ceil(required * 100) / 100 under band_policy "raise_to_vessel"; n =
    max(n_components, ceil(recurrence_min_s (hi - lo) / Tp)) and SeaState v2
    (sea_state.py SeaState.__init__: n = max(n, 256, ceil((hi - lo)/0.0125))).
    Tested equal to the layer's resolved block for every wave preset.
    """
    from importlib import import_module
    hydro_v2 = import_module(__package__.rsplit(".", 1)[0] + ".hydro_v2")
    fp = 1.0 / waves.tp_s
    required = hydro_v2.required_high_frequency_hz(hydro, fp) * waves.tp_s
    hi = float(waves.band_max_over_fp)
    if waves.band_policy == "raise_to_vessel" and hi < required:
        hi = math.ceil(required * 100.0 - 1e-9) / 100.0
    lo = float(waves.band_min_over_fp)
    n = max(int(waves.n_components), int(math.ceil(waves.recurrence_min_s * (hi - lo) / waves.tp_s - 1e-9)))
    n = max(n, 256, math.ceil((hi - lo) / 0.0125))
    return lo, hi, int(n)


def formula_record(waves, geometry, *, band=None, hydro=None, preset=None, preset_role=None):
    """JSON-native formula stamp of one wave dose and vessel geometry.

    ``band`` = (lo, hi, n) realised by the simulator (the condition layer's
    resolved band); else it is resolved from ``hydro`` (resolved_band).
    Waves disabled or Hs = 0: probability exactly 0 (calm surface).
    """
    calm = (not waves.enabled) or waves.hs_m == 0.0
    out = {"version": FORMULA_VERSION, "criterion": CRITERION, "limit": NORDFORSK_WETNESS_PER_CYCLE,
           "method": METHOD, "vessel": geometry.vessel, "preset": preset, "preset_role": preset_role}
    if calm:
        out.update(probability=0.0, worst_heading_deg=None, worst_point=None, worst_rel_rms_m=0.0,
                   inputs={"calm": True, "hs_m": float(waves.hs_m), "enabled": bool(waves.enabled),
                           "geometry": geometry.as_dict()})
        return out
    if band is None:
        if hydro is None:
            raise ValueError("formula_record needs the realised band or the hydro_v2 hull spec")
        band = resolved_band(waves, hydro)
    lo, hi, n = float(band[0]), float(band[1]), int(band[2])
    f, e = midpoint_spectrum(waves.hs_m, waves.tp_s, waves.gamma, (lo, hi), n)
    result = wetness(geometry, f, e, waves.spreading_rad)
    out.update(probability=result["probability"], worst_heading_deg=result["heading_deg"],
               worst_point=result["point"], worst_rel_rms_m=math.sqrt(result["m0_m2"]),
               per_heading=result["per_heading"],
               inputs={"calm": False, "hs_m": float(waves.hs_m), "tp_s": float(waves.tp_s),
                       "gamma": float(waves.gamma), "spreading_rad": float(waves.spreading_rad),
                       "spreading_law": "uniform +/- spreading_rad (C2 SeaState v2 direction draw)",
                       "band_over_fp": [lo, hi], "n_components": n,
                       "spectrum": "JONSWAP midpoint grid, Hs renormalised (float64 mirror of C2 SeaState v2)",
                       "headings_deg": list(HEADINGS_DEG), "direction_order": DIRECTION_ORDER,
                       "gravity_m_s2": GRAVITY, "geometry": geometry.as_dict()})
    return out


__all__ = ["FORMULA_VERSION", "CRITERION", "NORDFORSK_WETNESS_PER_CYCLE", "Strip", "WetnessGeometry",
           "midpoint_spectrum", "direction_nodes", "relative_motion_m0", "relative_motion_quadrature",
           "exceedance", "wetness", "resolved_band", "formula_record", "HEADINGS_DEG", "D9_HEADINGS_DEG"]
