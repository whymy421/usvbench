"""Read-only simulated green-water detection and the per-cell formula stamp (G10 item 2).

Definition (versioned: green_water_rule.DETECTION_VERSION). Every physics
substep, for every env and every declared deck-edge point p (body frame, COM
origin, FLU):
  * world position  x_p = c + R(q) p  from the SAME root-link pose (COM
    position c, attitude q: heave, roll, pitch and yaw included) that
    HydroV2Runtime.step_from_root_link uses for the hull wrench of that
    substep (decision 12: root link = COM for every v2 asset);
  * local free-surface elevation eta(x_p, t) from the SAME SpectralWaves
    realisation and the SAME runtime clock t (RuntimeStep.time_s) the
    buoyancy uses, with the SAME arithmetic as SpectralWaves.sample
    (footprint=False: the free surface at the point, not the hull-averaged
    value the buoyancy applies at the COM); bit-identical to
    ``waves.sample(x_p_xy, origin_xy, t, yaw, spec, footprint=False).elevation``
    (tested). Calm water (waves None): eta = 0, the constant surface (D8);
  * submerged  <=>  z_p < z_origin + mean_water_level_local + eta(x_p, t).
A per-env, per-point counter counts consecutive submerged substeps (reset to
0 at the first dry substep). An episode has GREEN WATER when any point stays
submerged for at least N_CONSECUTIVE_SUBSTEPS consecutive substeps.

G18 diagnostics (read-only, DIAGNOSTICS_VERSION; never part of the rule/1
record or verdict): from the SAME submerged mask, (i) the N-substep detector
re-run counting only substeps at or after SETTLE_S from the reset (the reset
transient: every family spawns the hull at the calm draft, level and at rest
while the wave clock starts at 0), with the number of qualifying events (a
hull-level qualifying state that begins) after that window; (ii) per point,
after the window, the up-crossings of the calm freeboard (submersion onsets)
and of the calm mean level (margin > -calm_freeboard) by the relative surface
height margin = z_origin + level + eta - z_p. Their ratio is the simulated
per-cycle exceedance (Rice), the unit of the formula. episode_diagnostics()
returns them; the mixin latches them beside the rule/1 record.

Nothing here writes back: no force, state, observation, reward, termination,
random draw or existing certificate field changes (tests compare every
existing field of calm, wave-free and wave runs with the detector on and
off). All state is device tensors updated with tensor ops only (no host
synchronisation per substep); the host reads one env's counters only when
its finished episode is latched (journal_dones).

Point sets (declared here, from the registry geometry; COM origin, FLU):
  * BlueBoat (blueboat_v2): the deck edge (gunwale) of both hulls at their
    freeboard: for each hull (centreline +/-BB_HALF_SPACING_M = 0.3607 m in
    the hull frame, vehicles_v2.py) the outer and inner deck edge (hull
    half-width from the asset bounds vehicles_v2_data.BLUEBOAT bounds_com_m:
    outer edge 0.44936 m from the hull-frame centre line, i.e. half-width
    0.08866 m; the hull is taken symmetric about its own centreline,
    UNVERIFIED) at the bow end, midship and stern end of the asset bounds
    (x = -0.59195 / 0.00744 / 0.60683 m from the COM); height z = registry
    freeboard_m - com_height_above_water_m (0.153022 - 0.007007 = 0.146015 m),
    so each point sits exactly at the registry freeboard above calm water in
    the simulator. 12 points.
  * WAM-V (wamv_vrx_v2_*), G28 (owner decision 2026-10-03 17:0x, decision
    pack 02 item 3: "the WAM-V green-water check location is the deck edge;
    the pontoon top stays report-only"): the DECK EDGE, the perimeter of the
    top face of the VRX deck plate (collision box "top_base", 1.85 x 1.0 x
    0.1 m centred at base_link z = 1.25 m, top face z = 1.30 m), aft / mid /
    fore stations (x = -0.925 / 0 / +0.925 m) of both long edges
    (y = +/-0.5 m), converted to the COM frame with the configuration's
    com_base_link_m (wamv_deck_edge_points; the six points T25 added as the
    report-only set). Calm freeboard 1.189 m (competition), 1.197 m (bare).
    6 points.
  * Report-only (report_point_sets): WAM-V the pontoon top line
    (pontoon_top_points, the binding set of v2.4 / v2.5): y = +/-1.03 m
    (WAMV_PONTOON_Y_M), top z = 0.4 m in base_link (WAMV_PONTOON_TOP_BASE_Z_M,
    main float r = 0.2 m at z = 0.2 m), five stations over the 4.9 m hull
    (WAMV_HULL_LENGTH_M) centred on the buoyancy centre x = -0.4 m in base_link
    (VRX wamv_gazebo_dynamics_plugin.xacro:15-16,32-33 points 0.6 / -1.4;
    wamv_base.urdf.xacro:10 main-float origin x = -0.4), at -2.45 / -1.225 /
    0 / +1.225 / +2.45 m; converted to the COM frame with the configuration's
    com_base_link_m (vehicles_v2_data.WAMV). Calm freeboard in the simulator
    = z_com + com_height_above_water_m = 0.28937 m (competition), 1.5 mm
    below the registry freeboard_m 0.29083 m, which is defined at the VRX
    trimmed equilibrium (static pitch 0.0107 rad) that the upright hydro_v2
    hull does not reproduce. 10 points. BlueBoat has no report-only set.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math

import torch

from . import green_water_formula as formula
from . import green_water_rule as rule

DETECTION_VERSION = rule.DETECTION_VERSION
DIAGNOSTICS_VERSION = rule.DIAGNOSTICS_VERSION
SETTLE_S = rule.SETTLE_S
SETTLE_RATIONALE = ("G18: every journal family spawns the hull at the calm draft, level and at rest while the wave "
                    "clock starts at 0, so the first substeps see the local crest above a deck edge in a share of "
                    "episodes that grows with Hs (an immediate detector hit) and a heave / attitude transient; "
                    "CPU replay of the G15 cells: the hull deviation from its wave-following state reaches the "
                    "value of a hull started ON the filtered surface after about 1-2 s (BlueBoat) and the WAM-V "
                    "heave transient has decayed after about 3 s; the diagnostics count only substeps at or "
                    "after settle_s")
MEAN_LEVEL_RULE = ("per point, after settle_s: freeboard up-crossings = submersion onsets (margin > 0 begins), "
                   "mean-level up-crossings = margin > -calm_freeboard begins, margin = z_origin + level + "
                   "eta(x_p, t) - z_p; their ratio is the simulated per-cycle exceedance of the calm freeboard")
# N rationale: 6 substeps = 0.05 s at the journal physics step 1/120 s (CONTEXT_R4 D2) = 3 control
# steps. A physical deck immersion by a wave of period T whose crest overtops the freeboard by a
# fraction d of its amplitude lasts (T/pi) arccos(1 - d); at the shortest journal peak period
# (ladder B1, Tp 1.95 s) 0.05 s corresponds to d = 0.6 %, at Tp 4 s to d = 0.15 %, so N removes
# only single- or few-substep toggles (solver jitter of the root pose, sub-millimetre
# high-frequency ripple near the threshold), never a wave-scale immersion.
N_CONSECUTIVE_SUBSTEPS = rule.N_CONSECUTIVE_SUBSTEPS
N_RATIONALE = ("6 consecutive physics substeps = 0.05 s at dt = 1/120 s (3 control steps); a wave-scale "
               "immersion lasts (T/pi) arccos(1 - d) for a crest overtopping the freeboard by a fraction d "
               "of its amplitude: 0.05 s is d = 0.6 % at Tp 1.95 s (ladder B1) and d = 0.15 % at Tp 4 s, so "
               "N filters only single/few-substep flicker (pose jitter, near-threshold ripple)")
SURFACE_RULE = ("submerged <=> z_p < z_origin + mean_water_level_local + eta(x_p, t); eta from the SAME "
                "SpectralWaves realisation and runtime clock t as the buoyancy of that substep, SAME "
                "arithmetic as SpectralWaves.sample(footprint=False) (local free surface at the point); "
                "calm water: eta = 0")
POSE_RULE = ("x_p = c + R(q) p from the root-link pose passed to HydroV2Runtime.step_from_root_link at the "
             "same substep (root link = COM, decision 12); heave, roll, pitch and yaw included")
WAMV_PONTOON_CENTRE_BASE_X_M = -0.4   # VRX dynamics xacro:15-16,32-33 (0.6 / -1.4); wamv_base.urdf.xacro:10
# T25 (owner decision T1.35d, 2026-10-02, "green-water check location on the WAM-V: data first"): a SECOND,
# REPORT-ONLY point set at the WAM-V DECK edge, observed with the same detector arithmetic beside the pontoon-top
# set (deck_edge_points) and recorded next to it (records[].green_water_report, header.green_water.report); it is
# never used by the rule/1 record, the cell summary, any exclusion verdict or the formula. Geometry (the same VRX
# source as the registry's WAMV_PONTOON_TOP_BASE_Z_M, wamv_base.urdf.xacro at vehicles_v2_data.VRX_COMMIT; D5
# evidence copy 20260930_d5_vessels_v2/work/evidence/web/C_wamv_base.urdf.xacro:120-125): the deck plate is the
# "top_base" collision box, centre base_link (0, 0, 1.25) m, size 1.85 x 1.0 x 0.1 m; its deck edge is the
# perimeter of the TOP face (base_link z = 1.25 + 0.05 = 1.30 m); points at the aft / mid / fore stations
# (x = -0.925 / 0 / +0.925 m) of both long edges (y = +/-0.5 m), converted to the COM frame with the registry
# COM (vehicles_v2_data.WAMV[config]["com_base_link_m"]); calm freeboard = z_COM-frame + the registry calm COM
# height (hydrostatics.com_height_above_water_m): 1.189 m competition, 1.197 m bare.
# BlueBoat: no second set; its declared points (deck_edge_points: outer and inner hull deck edges at the
# registry hydrostatics.freeboard_m, bow / mid / stern) already ARE its deck edge.
# G28 (owner decision 2026-10-03 17:0x item 3): the roles of the two WAM-V sets are swapped. The deck edge above is
# now the BINDING set (deck_edge_points, header.green_water.detection.points, rule/1 cell, G18 diagnostics, rule/2
# exclusion, formula points); the pontoon top line is the report-only set "wamv_pontoon_top" (report_point_sets).
REPORT_VERSION = rule.REPORT_VERSION
REPORT_DECISION = rule.REPORT_DECISION
WAMV_DECK_CENTRE_BASE_M = (0.0, 0.0, 1.25)     # wamv_base.urdf.xacro top_base <origin xyz="0 -0 1.25"/>
WAMV_DECK_SIZE_M = (1.85, 1.0, 0.1)            # wamv_base.urdf.xacro top_base <box size="1.85 1 0.1"/>
WAMV_DECK_SOURCE = ("VRX wamv_base.urdf.xacro collision 'top_base' (box 1.85 x 1.0 x 0.1 m at base_link z 1.25 m; "
                    "D5 evidence copy C_wamv_base.urdf.xacro:120-125, VRX commit vehicles_v2_data.VRX_COMMIT): "
                    "deck edge = perimeter of the top face (base_link z 1.30 m), aft / mid / fore stations of both "
                    "long edges; COM frame from vehicles_v2_data.WAMV[config].com_base_link_m; calm freeboard = "
                    "z + hydrostatics.com_height_above_water_m")
BLUEBOAT_REPORT_NOTE = ("BlueBoat: no report-only set; deck_edge_points already lie on its deck edge (outer and inner "
                        "hull deck edges at the registry hydrostatics.freeboard_m)")
WAMV_PONTOON_SOURCE = ("VRX pontoon top line (the binding set of v2.4 / v2.5, report-only since G28): y = +/-1.03 m "
                       "(WAMV_PONTOON_Y_M), base_link z 0.4 m (WAMV_PONTOON_TOP_BASE_Z_M), five stations over the 4.9 m "
                       "hull centred at base_link x = -0.4 m; COM frame from vehicles_v2_data.WAMV[config].com_base_link_m")
REPORT_SET_SOURCES = {"wamv_deck_edge": WAMV_DECK_SOURCE, "wamv_pontoon_top": WAMV_PONTOON_SOURCE}


def _shared(name):
    return import_module(__package__.rsplit(".", 1)[0] + "." + name)


@dataclass(frozen=True)
class DeckPoint:
    label: str
    x_m: float
    y_m: float
    z_m: float                  # body frame, COM origin, FLU
    calm_freeboard_m: float     # z_m + com_height_above_water_m (height above calm water, upright)


def deck_edge_points(vessel):
    """Declared deck-edge points of a registered v2 vessel (see the module docstring)."""
    registry = _shared("vehicles_v2")
    data = _shared("vehicles_v2_data")
    spec = registry.get_vehicle_v2(vessel)
    hydro = spec.hydrostatics
    if spec.name == "blueboat_v2":
        d = data.BLUEBOAT
        com_h = d["com_hull_frame_m"]
        lo, hi = d["bounds_com_m"]
        outer = hi[1] + com_h[1]                      # hull-frame outer deck edge (port)
        if not math.isclose(outer, -(lo[1] + com_h[1]), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("BlueBoat asset bounds are not symmetric about the hull centreline")
        half_width = outer - registry.BB_HALF_SPACING_M
        inner = registry.BB_HALF_SPACING_M - half_width
        z = hydro.freeboard_m - hydro.com_height_above_water_m
        stations = (("bow", hi[0]), ("mid", 0.5 * (lo[0] + hi[0])), ("stern", lo[0]))
        points = []
        for side, sign in (("port", 1.0), ("starboard", -1.0)):
            for edge, y_hull in (("outer", outer), ("inner", inner)):
                for station, x in stations:
                    points.append(DeckPoint(f"{side}_{edge}_{station}", float(x), float(sign * y_hull - com_h[1]),
                                            float(z), float(z + hydro.com_height_above_water_m)))
        return tuple(points)
    if spec.name.startswith("wamv_vrx_v2_"):
        return wamv_deck_edge_points(vessel)          # G28: the deck edge is the binding WAM-V set
    raise ValueError(f"no declared deck-edge point set for vessel {vessel!r}")


def pontoon_top_points(vessel):
    """WAM-V pontoon top line (10 points; binding in v2.4 / v2.5, report-only since G28; module docstring)."""
    registry = _shared("vehicles_v2")
    data = _shared("vehicles_v2_data")
    spec = registry.get_vehicle_v2(vessel)
    if not spec.name.startswith("wamv_vrx_v2_"):
        raise ValueError(f"pontoon-top points are defined for the WAM-V only, not {vessel!r}")
    hydro = spec.hydrostatics
    config = "bare" if spec.name.endswith("_bare") else "competition"
    com_b = data.WAMV[config]["com_base_link_m"]
    half = 0.5 * registry.WAMV_HULL_LENGTH_M
    stations = (("stern", -half), ("aft", -0.5 * half), ("mid", 0.0), ("fore", 0.5 * half), ("bow", half))
    z = registry.WAMV_PONTOON_TOP_BASE_Z_M - com_b[2]
    points = []
    for side, sign in (("port", 1.0), ("starboard", -1.0)):
        for station, dx in stations:
            points.append(DeckPoint(f"{side}_pontoon_top_{station}",
                                    float(WAMV_PONTOON_CENTRE_BASE_X_M + dx - com_b[0]),
                                    float(sign * registry.WAMV_PONTOON_Y_M - com_b[1]), float(z),
                                    float(z + hydro.com_height_above_water_m)))
    return tuple(points)


def report_point_sets(vessel):
    """((set name, points), ...) of the REPORT-ONLY point sets of a v2 vessel. G28: WAM-V -> the pontoon top line
    ("wamv_pontoon_top"; T25 had the deck edge here); BlueBoat -> none."""
    registry = _shared("vehicles_v2")
    spec = registry.get_vehicle_v2(vessel)
    if not spec.name.startswith("wamv_vrx_v2_"):
        return ()
    return (("wamv_pontoon_top", pontoon_top_points(vessel)),)


def wamv_deck_edge_points(vessel):
    """WAM-V deck-plate edge (6 points; T25 report-only set, BINDING since G28; module docstring)."""
    registry = _shared("vehicles_v2")
    data = _shared("vehicles_v2_data")
    spec = registry.get_vehicle_v2(vessel)
    if not spec.name.startswith("wamv_vrx_v2_"):
        raise ValueError(f"WAM-V deck-edge points are defined for the WAM-V only, not {vessel!r}")
    config = "bare" if spec.name.endswith("_bare") else "competition"
    com_b = data.WAMV[config]["com_base_link_m"]
    hydro = spec.hydrostatics
    cx, cy, cz = WAMV_DECK_CENTRE_BASE_M
    hx, hy, hz = (0.5 * s for s in WAMV_DECK_SIZE_M)
    z = cz + hz - com_b[2]
    points = []
    for side, sign in (("port", 1.0), ("starboard", -1.0)):
        for station, dx in (("aft", -hx), ("mid", 0.0), ("fore", hx)):
            points.append(DeckPoint(f"{side}_deck_edge_{station}", float(cx + dx - com_b[0]),
                                    float(cy + sign * hy - com_b[1]), float(z),
                                    float(z + hydro.com_height_above_water_m)))
    return tuple(points)


def wetness_geometry(vessel, points=None):
    """Formula inputs (green_water_formula.WetnessGeometry) from the registry and the point set.

    ``points`` (G28): None = the binding set (deck_edge_points, the exclusion probe); a DeckPoint tuple evaluates
    the same formula at another set (diagnostics and regression tests only, e.g. pontoon_top_points)."""
    registry = _shared("vehicles_v2")
    data = _shared("vehicles_v2_data")
    spec = registry.get_vehicle_v2(vessel)
    hydro = spec.hydrostatics
    chosen = deck_edge_points(vessel) if points is None else tuple(points)
    pts = tuple((p.label, p.x_m, p.y_m, p.calm_freeboard_m) for p in chosen)
    labels = [p.label for p in chosen]
    if spec.name == "blueboat_v2":
        d = data.BLUEBOAT
        com_h = d["com_hull_frame_m"]
        length = d["waterline_length_m"]
        width = d["waterplane_m2"] / (2.0 * length)
        cx = d["buoyancy_centre_x_hull_m"] - com_h[0]
        strips = tuple(formula.Strip(float(cx), float(sign * registry.BB_HALF_SPACING_M - com_h[1]),
                                     float(length), float(width)) for sign in (1.0, -1.0))
        sources = (("waterplane", "two rectangular strips at the hull centrelines +/-0.3607 m (BB_HALF_SPACING_M), "
                                  "length = waterline_length_m, width = waterplane_m2 / (2 length), centred at "
                                  "buoyancy_centre_x_hull_m (vehicles_v2_data.BLUEBOAT, CAD hydrostatics)"),
                   ("points", "deck-edge points of green_water.deck_edge_points (12, registry freeboard)"
                    if points is None else "caller-supplied point set (%d points, not the binding set)" % len(labels)))
    elif spec.name.startswith("wamv_vrx_v2_"):
        config = "bare" if spec.name.endswith("_bare") else "competition"
        com_b = data.WAMV[config]["com_base_link_m"]
        length = registry.WAMV_HULL_LENGTH_M
        width = hydro.waterplane_area_m2 / (2.0 * length)
        strips = tuple(formula.Strip(float(WAMV_PONTOON_CENTRE_BASE_X_M - com_b[0]),
                                     float(sign * registry.WAMV_PONTOON_Y_M - com_b[1]),
                                     float(length), float(width)) for sign in (1.0, -1.0))
        sources = (("waterplane", "two rectangular strips at the pontoon centrelines +/-1.03 m (WAMV_PONTOON_Y_M), "
                                  "length 4.9 m (WAMV_HULL_LENGTH_M) centred at base_link x = -0.4 m (VRX "
                                  "buoyancy points), width = registry waterplane_area_m2 / (2 x 4.9)"),
                   ("points", "deck-plate edge of green_water.deck_edge_points (6; G28 binding WAM-V probe, owner "
                              "decision 2026-10-03)" if points is None else
                    "caller-supplied point set (%d points, not the binding set)" % len(labels)))
    else:
        raise ValueError(f"no wetness geometry for vessel {vessel!r}")
    return formula.WetnessGeometry(spec.name, strips, pts, sources)


def wave_preset_role(presets):
    """(wave preset name or None, role) of a resolved condition stack's preset tuple."""
    presets_v3 = import_module(__package__ + ".presets_v3")
    for name in presets:
        for component, values in presets_v3.expand_presets((name,)):
            if "waves" in values:
                entry = presets_v3.JOURNAL_PRESETS[component]
                return component, entry.get("role", "MAIN")
    return None, "OVERRIDE"


def formula_for(cfg_waves, vessel, *, band=None, hydro=None, presets=(), points=None):
    """Formula stamp of one resolved wave dose (WavesV3) and vessel (green_water_formula.formula_record);
    ``points`` None = the binding set (G28: WAM-V deck edge)."""
    preset, role = wave_preset_role(presets)
    return formula.formula_record(cfg_waves, wetness_geometry(vessel, points), band=band, hydro=hydro,
                                  preset=preset, preset_role=role)


class GreenWaterMonitor:
    """Per-env, per-point consecutive-submersion counters (vectorised; read-only)."""

    # all points in ONE [N,P,M] evaluation (few kernel launches) while N x P x M stays below this many
    # elements (2^24 = 64 MiB per float32 temporary); above it, one [N,M] evaluation per point (the
    # memory of a single wave sample). Both paths are bit-identical to SpectralWaves.sample (tested).
    vector_limit = 1 << 24

    def __init__(self, vessel, num_envs, device, dtype, physics_dt, formula_stamp, presets=(), points=None):
        rotation_matrix = _shared("physics_v2").rotation_matrix
        self._rotation_matrix = rotation_matrix
        self.vessel = str(vessel)
        # T25 (T1.35d): ``points`` = a report-only set (report_monitors_for); None = the declared deck-edge set
        self.points = deck_edge_points(vessel) if points is None else tuple(points)
        self.labels = tuple(p.label for p in self.points)
        self.num_envs, self.device, self.dtype = int(num_envs), torch.device(device), dtype
        self.physics_dt = float(physics_dt)
        self.n_required = int(N_CONSECUTIVE_SUBSTEPS)
        self.formula = formula_stamp
        self.presets = tuple(presets)
        self.body_points = torch.tensor([[p.x_m, p.y_m, p.z_m] for p in self.points],
                                        dtype=torch.float64).to(device=self.device, dtype=dtype)
        n, m = self.num_envs, len(self.points)
        self.run = torch.zeros((n, m), dtype=torch.int64, device=self.device)
        self.longest = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.total = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.green = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.first_tick = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.first_point = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.ticks = torch.zeros(n, dtype=torch.int64, device=self.device)
        # G18 read-only diagnostics (settling window, events, crossings)
        self.settle_ticks = int(round(SETTLE_S / self.physics_dt))
        self.calm_freeboard = torch.tensor([p.calm_freeboard_m for p in self.points],
                                           dtype=torch.float64).to(device=self.device, dtype=dtype)
        self.run_settled = torch.zeros((n, m), dtype=torch.int64, device=self.device)
        self.green_settled = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.first_tick_settled = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.first_point_settled = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.events_settled = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.qualified_settled = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.up_freeboard = torch.zeros((n, m), dtype=torch.int64, device=self.device)
        self.up_mean = torch.zeros((n, m), dtype=torch.int64, device=self.device)
        self.prev_submerged = torch.zeros((n, m), dtype=torch.bool, device=self.device)
        self.prev_above_mean = torch.zeros((n, m), dtype=torch.bool, device=self.device)

    # -------------------------------------------------------------- geometry --
    def point_positions(self, root_link_pose_w):
        """[N,P,3] world positions of the deck-edge points (runtime dtype)."""
        pose = root_link_pose_w.to(self.dtype)
        rotation = self._rotation_matrix(pose[:, 3:7])
        return pose[:, None, :3] + torch.einsum("nij,pj->npi", rotation, self.body_points)

    @staticmethod
    def wave_terms(waves, time_s, like):
        """Point-independent factors of SpectralWaves.sample: (kx, ky, omega t, amplitude), computed once."""
        time = torch.as_tensor(time_s, device=like.device, dtype=like.dtype).reshape(-1, 1)
        kx, ky = waves.k * torch.cos(waves.direction), waves.k * torch.sin(waves.direction)
        return kx, ky, waves.omega * time, waves.amplitude * 1.0

    def point_elevation(self, waves, world_xy, env_origin_xy, time_s, terms=None):
        """[N] local elevation at one point per env with the arithmetic (and evaluation order) of
        SpectralWaves.sample(footprint=False): theta = kx x + ky y - omega t + phase, sum(a cos theta)."""
        kx, ky, omega_t, amp = self.wave_terms(waves, time_s, world_xy) if terms is None else terms
        local = world_xy - env_origin_xy
        theta = kx * local[:, 0:1] + ky * local[:, 1:2] - omega_t + waves.phase
        return (amp * torch.cos(theta)).sum(-1)

    def surface_elevations(self, waves, positions, env_origins_w, time_s):
        """[N,P] free-surface elevation at every point (zeros in calm water)."""
        if waves is None:
            return torch.zeros(positions.shape[:2], dtype=self.dtype, device=positions.device)
        origin_xy = env_origins_w[:, :2].to(self.dtype)
        terms = self.wave_terms(waves, time_s, positions)
        n, p = positions.shape[:2]
        if n * p * waves.amplitude.shape[-1] <= self.vector_limit:
            kx, ky, omega_t, amp = (t.unsqueeze(-2) for t in terms)
            local = positions[..., :2] - origin_xy.unsqueeze(1)
            theta = kx * local[..., 0:1] + ky * local[..., 1:2] - omega_t + waves.phase.unsqueeze(-2)
            return (amp * torch.cos(theta)).sum(-1)
        return torch.stack([self.point_elevation(waves, positions[:, j, :2], origin_xy, time_s, terms)
                            for j in range(p)], -1)

    # --------------------------------------------------------------- substep --
    def observe(self, root_link_pose_w, env_origins_w, mean_water_level_local, time_s, waves):
        """One physics substep (call right after runtime.step_from_root_link with ITS inputs)."""
        positions = self.point_positions(root_link_pose_w)
        eta = self.surface_elevations(waves, positions, env_origins_w, time_s)
        # the runtime's water level: origins z + mean_water_level_local (scalar or [N]), runtime dtype
        level = (env_origins_w[:, 2].to(self.dtype) + mean_water_level_local).to(self.dtype)
        submerged = positions[..., 2] < level.reshape(-1, 1) + eta
        self.run = torch.where(submerged, self.run + 1, torch.zeros_like(self.run))
        self.longest = torch.maximum(self.longest, self.run.max(dim=1).values)
        self.total = self.total + submerged.sum(dim=1)
        qualified = self.run >= self.n_required
        any_qualified = qualified.any(dim=1)
        newly = any_qualified & ~self.green
        self.first_tick = torch.where(newly, self.ticks - (self.n_required - 1), self.first_tick)
        self.first_point = torch.where(newly, qualified.to(torch.int32).argmax(dim=1).to(torch.int64),
                                       self.first_point)
        self.green = self.green | any_qualified
        self._observe_diagnostics(submerged, level.reshape(-1, 1) + eta - positions[..., 2])
        self.ticks = self.ticks + 1
        return submerged

    def _observe_diagnostics(self, submerged, margin):
        """G18 read-only diagnostics of this substep (tensor ops only; self.ticks = this substep's index)."""
        window = (self.ticks >= self.settle_ticks).unsqueeze(1)            # this substep is inside the window
        crossing = (self.ticks > self.settle_ticks).unsqueeze(1)           # ... and so is the previous one
        sub_w = submerged & window
        self.run_settled = torch.where(sub_w, self.run_settled + 1, torch.zeros_like(self.run_settled))
        qualified = self.run_settled >= self.n_required
        any_qualified = qualified.any(dim=1)
        newly = any_qualified & ~self.green_settled
        self.first_tick_settled = torch.where(newly, self.ticks - (self.n_required - 1), self.first_tick_settled)
        self.first_point_settled = torch.where(newly, qualified.to(torch.int32).argmax(dim=1).to(torch.int64),
                                               self.first_point_settled)
        self.events_settled = self.events_settled + (any_qualified & ~self.qualified_settled).to(torch.int64)
        self.qualified_settled = any_qualified
        self.green_settled = self.green_settled | any_qualified
        above_mean = margin > -self.calm_freeboard
        self.up_freeboard = self.up_freeboard + (submerged & ~self.prev_submerged & crossing).to(torch.int64)
        self.up_mean = self.up_mean + (above_mean & ~self.prev_above_mean & crossing).to(torch.int64)
        self.prev_submerged = submerged
        self.prev_above_mean = above_mean

    def reset(self, env_ids):
        """Zero exactly the reset rows (reset-row isolation)."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        self.run[ids] = 0
        self.longest[ids] = 0
        self.total[ids] = 0
        self.green[ids] = False
        self.first_tick[ids] = -1
        self.first_point[ids] = -1
        self.ticks[ids] = 0
        self.run_settled[ids] = 0
        self.green_settled[ids] = False
        self.first_tick_settled[ids] = -1
        self.first_point_settled[ids] = -1
        self.events_settled[ids] = 0
        self.qualified_settled[ids] = False
        self.up_freeboard[ids] = 0
        self.up_mean[ids] = 0
        self.prev_submerged[ids] = False
        self.prev_above_mean[ids] = False

    # ----------------------------------------------------------- certificate --
    def episode_record(self, env_id, runtime_ticks=None):
        """Per-episode record of the FINISHED episode of env_id (host read; call before its reset)."""
        i = int(env_id)
        ticks = int(self.ticks[i])
        if runtime_ticks is not None and int(runtime_ticks) != ticks:
            raise RuntimeError(f"env {i}: green-water detector observed {ticks} substeps but the runtime ran "
                               f"{int(runtime_ticks)}; observe() must run once per runtime.step")
        green = bool(self.green[i])
        first = int(self.first_tick[i])
        point = int(self.first_point[i])
        return {"green_water": green,
                "first_time_s": float(first * self.physics_dt) if green else None,
                "first_point": self.labels[point] if green else None,
                "submerged_point_substeps": int(self.total[i]),
                "longest_run_substeps": int(self.longest[i]),
                "substeps": ticks}

    def episode_diagnostics(self, env_id):
        """G18 read-only diagnostics of the FINISHED episode of env_id (host read; call before its reset)."""
        i = int(env_id)
        ticks = int(self.ticks[i])
        settled = bool(self.green_settled[i])
        first = int(self.first_tick_settled[i])
        point = int(self.first_point_settled[i])
        return {"version": DIAGNOSTICS_VERSION, "settle_s": SETTLE_S,
                "green_water_settled": settled,
                "first_time_settled_s": float(first * self.physics_dt) if settled else None,
                "first_point_settled": self.labels[point] if settled else None,
                "events_settled": int(self.events_settled[i]),
                "exposure_settled_s": max(0.0, ticks * self.physics_dt - SETTLE_S),
                "freeboard_upcrossings": [int(v) for v in self.up_freeboard[i].tolist()],
                "mean_level_upcrossings": [int(v) for v in self.up_mean[i].tolist()]}

    def header(self):
        """Static header block (cert_v3 adds the cell summary over the counted episodes)."""
        return {"schema": rule.HEADER_SCHEMA, "vessel": self.vessel, "rule": rule.rule_block(),
                "formula": self.formula,
                "detection": {"version": DETECTION_VERSION, "n_consecutive_substeps": self.n_required,
                              "physics_dt_s": self.physics_dt,
                              "min_duration_s": self.n_required * self.physics_dt,
                              "n_rationale": N_RATIONALE, "surface": SURFACE_RULE, "pose": POSE_RULE,
                              "points_frame": "body FLU, COM origin (actor frame of the v2 asset)",
                              "points": [{"label": p.label, "x_m": p.x_m, "y_m": p.y_m, "z_m": p.z_m,
                                          "calm_freeboard_m": p.calm_freeboard_m} for p in self.points],
                              "record_fields": list(rule.RECORD_FIELDS),
                              "first_time": "onset of the first qualifying run: (tick of detection - N + 1) "
                                            "x physics dt, episode time from the reset",
                              "diagnostics": {"version": DIAGNOSTICS_VERSION, "settle_s": SETTLE_S,
                                              "settle_rationale": SETTLE_RATIONALE,
                                              "crossings": MEAN_LEVEL_RULE, "binding": False,
                                              "record_key": "records[].green_water_diagnostics"}},
                "presets": list(self.presets)}


def monitor_for(layer, runtime):
    """GreenWaterMonitor when the condition stack includes waves (enabled, any Hs), else None.

    Built once at journal_setup; the formula uses the band the layer
    realises (resolved block) or, at Hs = 0, is exactly 0.
    """
    waves = layer.cfg.waves
    if not waves.enabled:
        return None
    resolved = layer.context.get("resolved", {}).get("waves", {})
    band = None
    if waves.hs_m > 0.0:
        band = (resolved["band_min_over_fp"], resolved["band_max_over_fp"], resolved["n_components"])
    stamp = formula_for(waves, layer.spec.name, band=band, presets=layer.cfg.presets)
    return GreenWaterMonitor(layer.spec.name, layer.num_envs, layer.device, runtime.dtype, runtime.dt, stamp,
                             presets=layer.cfg.presets)


def report_monitors_for(main):
    """T25 (T1.35d): {set name: GreenWaterMonitor of a report-only point set} beside the main monitor ({} when
    the vessel has none or the stack has no waves). Same vessel, envs, device, dtype, physics step and presets;
    no formula (never part of a verdict)."""
    if main is None:
        return {}
    return {name: GreenWaterMonitor(main.vessel, main.num_envs, main.device, main.dtype, main.physics_dt, None,
                                    presets=main.presets, points=points)
            for name, points in report_point_sets(main.vessel)}


def report_header(monitors):
    """T25 (T1.35d): static header block header.green_water.report_points of the report-only sets."""
    return {"version": REPORT_VERSION, "binding": False, "decision": REPORT_DECISION,
            "record_key": "records[].green_water_report", "summary_key": "header.green_water.report",
            "detection": ("the detector of header.green_water.detection (same surface, pose, N consecutive substeps) "
                          "and the G18 diagnostics (settle_s, events, freeboard and mean-level up-crossings) on "
                          "these points; never used by the rule/1 record, the cell summary or any verdict"),
            "sets": {name: {"source": REPORT_SET_SOURCES.get(name),
                            "n_consecutive_substeps": m.n_required, "physics_dt_s": m.physics_dt,
                            "settle_s": SETTLE_S,
                            "points": [{"label": p.label, "x_m": p.x_m, "y_m": p.y_m, "z_m": p.z_m,
                                        "calm_freeboard_m": p.calm_freeboard_m} for p in m.points]}
                     for name, m in monitors.items()}}


def report_episode(monitors, env_id, runtime_ticks):
    """T25 (T1.35d): records[].green_water_report of the FINISHED episode (call before its reset)."""
    return {name: {"record": m.episode_record(env_id, runtime_ticks), "diagnostics": m.episode_diagnostics(env_id)}
            for name, m in monitors.items()}


__all__ = ["DETECTION_VERSION", "DIAGNOSTICS_VERSION", "SETTLE_S", "N_CONSECUTIVE_SUBSTEPS", "N_RATIONALE",
           "DeckPoint", "deck_edge_points", "pontoon_top_points", "wamv_deck_edge_points", "REPORT_SET_SOURCES",
           "wetness_geometry", "wave_preset_role", "formula_for", "GreenWaterMonitor", "monitor_for",
           "REPORT_VERSION", "REPORT_DECISION", "WAMV_DECK_CENTRE_BASE_M", "WAMV_DECK_SIZE_M", "report_point_sets",
           "report_monitors_for", "report_header", "report_episode"]
