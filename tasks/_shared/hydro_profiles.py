"""hydro_v2 vessel profiles: a thin adapter over the D5 v2 registry (E2).

Controller decision 1: vessel numbers have ONE source, the D5 registry
(tasks/_shared/vehicles_v2.py + vehicles_v2_data.py). This module builds the
D3 contract ``VesselProfileV2(hydro=HydroSpec, thrusters=ThrusterSpec, ...)``
ONLY through the registry adapters ``to_d3_hydro_kwargs`` and
``to_d3_thruster_kwargs``; the one physics override it applies itself is
the roll/pitch damping rule of decision 2 (``rollpitch_damping``), computed
from the registry's own stiffness and inertia. It contains no vessel number
(test_profiles_runtime.test_no_vessel_numbers_in_hydro_profiles). D3's
provisional builders (blueboat_profile / wamv_vrx_profile with legacy box
hydrostatics and report numbers) are superseded and removed.

All v2 assets are FLU with the actor origin at the COM (vehicles_v2.py
validate_vehicle), so every profile has com_offset_actor_m = (0, 0, 0) and
the identity actor_to_hull basis.

Roll/pitch damping (decision 2): BlueBoat zeta 0.1 DECIDED by the owner
(2026-10-01 00:0x, item A1; kept after the G1 sweep); WAM-V entry switched by
owner decision T1.35b (switch b, zeta 0.1; adopted for v2.6, controller
decision J1). The runtime ``status`` string below still says "PENDING owner
decisions": it is frozen because it enters the D7 envelope provenance
(dynamic/vessels.py -> trajectory identity) and so the admitted pools; G28
leaves it unchanged.
  "journal_default"  BlueBoat -> "zeta:0.1"; WAM-V -> "zeta:0.1"
                     (WAMV_ROLLPITCH_JOURNAL_DEFAULT; before T1.35b "registry", the VRX values)
  "zeta:<ratio>"     c_axis = 2 * ratio * sqrt(K_axis * I_axis) per axis,
                     roll: (K_roll, I_xx), pitch: (K_pitch, I_yy), with the
                     registry's TOTAL stiffness and the diagonal of its
                     inertia tensor about the COM
  "legacy_470"       the registry's roll/pitch linear damping of the BlueBoat,
                     i.e. its legacy calibration (registry provenance
                     "damping": blueboat_calm_nav_env_cfg.py:73-97); rejected
                     for other vessels
  "registry"         the registry's own roll/pitch linear damping
  <positive number>  that value for roll AND pitch (via to_d3_hydro_kwargs)
The brief's default "zeta:0.1" applies to BlueBoat; decision 2 kept the VRX
values for the WAM-V, hence the per-vessel "journal_default" sentinel; owner
decision T1.35b switches the WAM-V entry to "zeta:0.1" as well (G18 REPORT
recommendation 4: removes the 10x pitch resonance of the VRX zeta 0.0054;
"registry" stays selectable). The resolved mode and values are recorded in
``sources`` and ``status``.

Sway/yaw damping (controller decision G4-A5, 2026-10-01, from the G1 sweep
variant B2q_xf1_quadyaw; rule ``swayyaw_damping``):
  "journal_default"        VRX-classic WAM-V (the registry profiles whose damping
                           set is vehicles_v2.WAMV_DAMPING_C: wamv_vrx_v2_opt1 and
                           wamv_vrx_v2_opt1_bare) -> "crossflow_sarda_strip";
                           every other vessel (BlueBoat, VRX option 2) ->
                           "registry"
  "crossflow_sarda_strip"  VRX-classic WAM-V only: the registry LINEAR sway and
                           yaw terms (40 N s/m, 400 N m s/rad) are kept and the
                           quadratic terms become
                             sway Y_v|v| = Sarda et al. 2016 Table 3 cross-flow
                             yaw  N_r|r| = Y_v|v| L^3 / 32 (strip theory)
                           (crossflow_damping.py), evaluated with the registry's
                           own geometry: L = vehicles_v2.WAMV_HULL_LENGTH_M,
                           B = 2 * vehicles_v2.WAMV_HULL_RADIUS_M, T = the
                           vessel's hydrostatics.draft_m, rho = its
                           hydrostatics.water_density_kg_m3
  "vrx_classic"            VRX-classic WAM-V only: the registry values (sway
                           40 / 0, yaw 400 / 0), the supplement comparison profile
  "registry"               the registry's own sway/yaw damping (any vessel)
Surge, heave, roll, pitch, mass, inertia and thrust are never touched by this
rule. The registry values stay the provenance values; the rule, its formula,
inputs (with file:line) and sources are stamped in ``sources["swayyaw_damping"]``,
``sources["options"]["swayyaw_damping"]`` and ``status``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import functools
import inspect
import math

try:
    from .hydro_v2 import HydroSpec
    from .thruster_mixer import ThrusterSpec
    from . import vehicles_v2 as _registry
    from . import crossflow_damping as _crossflow
    from .journal_vehicle import v2_spec_or_raise
except ImportError:  # Path-loaded CPU tests; never import the tasks package.
    from hydro_v2 import HydroSpec
    from thruster_mixer import ThrusterSpec
    import vehicles_v2 as _registry
    import crossflow_damping as _crossflow
    from journal_vehicle import v2_spec_or_raise


IDENTITY_BASIS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO_OFFSET = (0.0, 0.0, 0.0)
JOURNAL_DEFAULT = "journal_default"
# Per-vessel journal default of the roll/pitch damping rule (decision 2),
# keyed by the registry's ``option`` field ("blueboat" or the WAM-V labels).
_BLUEBOAT_OPTION = "blueboat"
_BLUEBOAT_DEFAULT_RULE = "zeta:0.1"
# Owner decision T1.35b (switch patch b, T25): WAM-V roll/pitch damping journal default "zeta:0.1" (was
# "registry" = the VRX values, roll/pitch zeta 0.008 / 0.0054, G18 REPORT Q1/Q3); BlueBoat unchanged.
WAMV_ROLLPITCH_JOURNAL_DEFAULT = "zeta:0.1"
WAMV_ROLLPITCH_DECISION = ("owner decision T1.35b: WAM-V roll/pitch damping journal default zeta:0.1 (was the VRX "
                           "registry values)")
_WAMV_DEFAULT_RULE = WAMV_ROLLPITCH_JOURNAL_DEFAULT
# Sway/yaw damping rule (G4-A5). Axis indices of the registry six-vectors
# (surge, sway, heave, roll, pitch, yaw), vehicles_v2.AXES.
SWAYYAW_RULES = ("journal_default", "crossflow_sarda_strip", "vrx_classic", "registry")
_CROSSFLOW_RULE = "crossflow_sarda_strip"
_VRX_CLASSIC_RULE = "vrx_classic"
_REGISTRY_RULE = "registry"
_SWAY = _registry.AXES.index("sway")
_YAW = _registry.AXES.index("yaw")
_REGISTRY_FILE = "tasks/_shared/vehicles_v2.py"


@dataclass(frozen=True)
class VesselProfileV2:
    name: str
    hydro: HydroSpec
    thrusters: ThrusterSpec
    com_offset_actor_m: tuple
    actor_to_hull: tuple = IDENTITY_BASIS
    status: str = "provisional"
    sources: dict = field(default_factory=dict)


def _positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be a positive finite number")
    return float(value)


def _rollpitch_rule(spec, rollpitch_damping):
    """Resolve decision 2 into (rule label, override for to_d3_hydro_kwargs, per-axis pair)."""
    rule = rollpitch_damping
    if isinstance(rule, str) and rule == JOURNAL_DEFAULT:
        rule = _BLUEBOAT_DEFAULT_RULE if spec.option == _BLUEBOAT_OPTION else _WAMV_DEFAULT_RULE
    if isinstance(rule, str):
        if rule == "registry":
            return rule, None, None
        if rule == "legacy_470":
            if spec.option != _BLUEBOAT_OPTION:
                raise ValueError("rollpitch_damping='legacy_470' is the legacy BlueBoat calibration; "
                                 "%s keeps its registry values ('registry')" % spec.name)
            return rule, None, None
        if rule.startswith("zeta:"):
            try:
                ratio = float(rule[len("zeta:"):])
            except ValueError:
                raise ValueError("rollpitch_damping %r: expected 'zeta:<ratio>'" % rule) from None
            ratio = _positive_finite(ratio, "zeta ratio")
            h, inertia = spec.hydrostatics, spec.inertia_com_kg_m2
            pair = (2.0 * ratio * math.sqrt(h.stiffness_roll_nm_rad * inertia[0][0]),
                    2.0 * ratio * math.sqrt(h.stiffness_pitch_nm_rad * inertia[1][1]))
            return rule, None, pair
        raise ValueError("rollpitch_damping must be 'journal_default', 'zeta:<ratio>', 'legacy_470', "
                         "'registry' or a positive number, got %r" % (rollpitch_damping,))
    value = _positive_finite(rule, "rollpitch_damping")
    return "value:%r" % value, value, None


@functools.lru_cache(maxsize=None)
def _registry_line(anchor):
    """file:line of the single registry source line containing ``anchor`` (stamped citations)."""
    lines = inspect.getsource(_registry).splitlines()
    hits = [index for index, line in enumerate(lines, start=1) if anchor in line]
    if len(hits) != 1:
        raise RuntimeError("registry citation anchor %r found %d times" % (anchor, len(hits)))
    return "%s:%d" % (_REGISTRY_FILE, hits[0])


def _is_vrx_classic(spec):
    """True for the registry profiles carrying the VRX-classic damping set (option 1, both configurations)."""
    return spec.damping == _registry.WAMV_DAMPING_C


def crossflow_inputs(spec):
    """Inputs of the cross-flow rule, read from the registry, each with its citation."""
    if not _is_vrx_classic(spec):
        raise ValueError("cross-flow inputs are defined for the VRX-classic WAM-V profiles only, not %s" % spec.name)
    h = spec.hydrostatics
    return {
        "length_m": {"value": float(_registry.WAMV_HULL_LENGTH_M),
                     "source": "vehicles_v2.WAMV_HULL_LENGTH_M, %s (VRX wamv_gazebo_dynamics_plugin.xacro:11 "
                               "hull_length)" % _registry_line("WAMV_HULL_LENGTH_M = ")},
        "draft_m": {"value": float(h.draft_m),
                    "source": "registry hydrostatics.draft_m of %s = mean immersion of the four VRX buoyancy "
                              "points, %s and %s (vehicles_v2_data.py point_depths_m)"
                              % (spec.name, _registry_line("mean_depth = sum("), _registry_line("draft_m=mean_depth"))},
        "beam_m": {"value": float(2.0 * _registry.WAMV_HULL_RADIUS_M),
                   "source": "2 * vehicles_v2.WAMV_HULL_RADIUS_M, %s (VRX wamv_gazebo_dynamics_plugin.xacro:12 "
                             "hull_radius; pontoon diameter)" % _registry_line("WAMV_HULL_RADIUS_M = ")},
        "water_density_kg_m3": {"value": float(h.water_density_kg_m3),
                                "source": "registry hydrostatics.water_density_kg_m3 = vehicles_v2.RHO_WATER, %s"
                                          % _registry_line("RHO_WATER = ")},
    }


def _swayyaw_rule(spec, swayyaw_damping):
    """Resolve the G4-A5 rule into (label, (sway lin, quad), (yaw lin, quad), stamp record)."""
    if not isinstance(swayyaw_damping, str) or swayyaw_damping not in SWAYYAW_RULES:
        raise ValueError("swayyaw_damping must be one of %s, got %r" % (", ".join(SWAYYAW_RULES), swayyaw_damping))
    classic = _is_vrx_classic(spec)
    rule = swayyaw_damping
    if rule == JOURNAL_DEFAULT:
        rule = _CROSSFLOW_RULE if classic else _REGISTRY_RULE
    if rule in (_CROSSFLOW_RULE, _VRX_CLASSIC_RULE) and not classic:
        raise ValueError("swayyaw_damping=%r is defined for the VRX-classic WAM-V profiles (damping set "
                         "vehicles_v2.WAMV_DAMPING_C) only; %s keeps its registry values ('registry')"
                         % (rule, spec.name))
    lin, quad = spec.damping.linear, spec.damping.quadratic
    sway = (float(lin[_SWAY]), float(quad[_SWAY]))
    yaw = (float(lin[_YAW]), float(quad[_YAW]))
    record = {"rule": rule, "requested": swayyaw_damping,
              "registry_sway_linear_quadratic": list(sway), "registry_yaw_linear_quadratic": list(yaw),
              "registry_source": spec.damping.source_set}
    if rule == _CROSSFLOW_RULE:
        inputs = crossflow_inputs(spec)
        geometry = (inputs["length_m"]["value"], inputs["draft_m"]["value"], inputs["beam_m"]["value"])
        sway_quadratic = _crossflow.sarda_sway_quadratic(*geometry, inputs["water_density_kg_m3"]["value"])
        yaw_quadratic = _crossflow.strip_yaw_quadratic(sway_quadratic, inputs["length_m"]["value"])
        sway, yaw = (sway[0], sway_quadratic), (yaw[0], yaw_quadratic)
        record.update({
            "decision": _crossflow.DECISION, "code": "tasks/_shared/crossflow_damping.py",
            "sway_formula": _crossflow.SARDA_FORMULA, "sway_source": _crossflow.SARDA_SOURCE,
            "sarda_bracket": _crossflow.sarda_coefficient(*geometry),
            "yaw_formula": _crossflow.STRIP_FORMULA, "yaw_source": _crossflow.STRIP_SOURCE,
            "linear_terms": "registry VRX-classic linear sway and yaw kept (%s)" % spec.damping.source_set,
            "inputs": inputs})
    elif rule == _VRX_CLASSIC_RULE:
        record["note"] = "VRX-classic comparison profile (registry values; supplement sensitivity row)"
    record["sway_linear_quadratic"] = list(sway)
    record["yaw_linear_quadratic"] = list(yaw)
    return rule, sway, yaw, record


def swayyaw_profile_spec(profile):
    """Registry VesselSpecV2 of ``profile`` with the profile's sway and yaw damping.

    For consumers that take a VesselSpecV2 rather than a profile: the D7
    vessel envelope (conditions/dynamic/vessels.envelope_from_spec_v2 reads the
    yaw damping for its steady-spin check). Only the sway and yaw entries of
    the registry damping six-vectors are replaced; when they already equal the
    registry values the registry object itself is returned.
    """
    spec = v2_spec_or_raise(profile.name)
    h = profile.hydro
    yaw_index = _YAW - len(h.linear_damping)
    sway = (float(h.linear_damping[_SWAY]), float(h.quadratic_damping[_SWAY]))
    yaw = (float(h.angular_linear_damping[yaw_index]), float(h.angular_quadratic_damping[yaw_index]))
    lin, quad = list(spec.damping.linear), list(spec.damping.quadratic)
    if (lin[_SWAY], quad[_SWAY]) == sway and (lin[_YAW], quad[_YAW]) == yaw:
        return spec
    lin[_SWAY], quad[_SWAY] = sway
    lin[_YAW], quad[_YAW] = yaw
    rule = profile.sources.get("options", {}).get("swayyaw_damping")
    damping = replace(spec.damping, linear=tuple(lin), quadratic=tuple(quad),
                      source_set="%s; sway/yaw from hydro_profiles swayyaw_damping=%s" % (spec.damping.source_set, rule))
    return replace(spec, damping=damping)


def profile_for_vehicle(name, *, rollpitch_damping=JOURNAL_DEFAULT, rotational_relative_motion=True,
                        pitch_coupling=False, froude_krylov_horizontal=True,
                        swayyaw_damping=JOURNAL_DEFAULT) -> VesselProfileV2:
    """hydro_v2 profile of the v2 registry vessel ``name`` (v2 names only).

    Built only through vehicles_v2.to_d3_hydro_kwargs / to_d3_thruster_kwargs;
    see the module docstring for ``rollpitch_damping`` and ``swayyaw_damping``
    (G4-A5). pitch_coupling=True passes the geometric thrust-line height to the
    mixer (decision 6 default OFF). froude_krylov_horizontal is decision 7
    (default ON, C_a = 0).
    Legacy names ('blueboat', 'wamv', 'rov') are rejected with the reason.
    """
    spec = v2_spec_or_raise(name)
    if not isinstance(rotational_relative_motion, bool) or not isinstance(pitch_coupling, bool) \
            or not isinstance(froude_krylov_horizontal, bool):
        raise ValueError("rotational_relative_motion, pitch_coupling and froude_krylov_horizontal "
                         "must be bools")
    rule, override, pair = _rollpitch_rule(spec, rollpitch_damping)
    swayyaw, sway, yaw, swayyaw_record = _swayyaw_rule(spec, swayyaw_damping)
    hydro_kwargs = _registry.to_d3_hydro_kwargs(spec, rollpitch_damping=override)
    if pair is not None:
        yaw_linear = hydro_kwargs["angular_linear_damping"][2]
        hydro_kwargs["angular_linear_damping"] = (pair[0], pair[1], yaw_linear)
    if swayyaw == _CROSSFLOW_RULE:
        lin, quad = list(hydro_kwargs["linear_damping"]), list(hydro_kwargs["quadratic_damping"])
        ang_lin = list(hydro_kwargs["angular_linear_damping"])
        ang_quad = list(hydro_kwargs["angular_quadratic_damping"])
        yaw_index = _YAW - len(lin)
        lin[_SWAY], quad[_SWAY] = sway
        ang_lin[yaw_index], ang_quad[yaw_index] = yaw
        hydro_kwargs.update(linear_damping=tuple(lin), quadratic_damping=tuple(quad),
                            angular_linear_damping=tuple(ang_lin), angular_quadratic_damping=tuple(ang_quad))
    hydro = HydroSpec(**hydro_kwargs, rotational_relative_motion=rotational_relative_motion,
                      froude_krylov_horizontal=froude_krylov_horizontal)
    thrusters = ThrusterSpec(**_registry.to_d3_thruster_kwargs(spec, pitch_coupling=pitch_coupling))
    options = {"rollpitch_damping": rule, "rotational_relative_motion": rotational_relative_motion,
               "pitch_coupling": pitch_coupling, "froude_krylov_horizontal": froude_krylov_horizontal,
               "added_mass_coefficient_horizontal": hydro.added_mass_coefficient_horizontal,
               "swayyaw_damping": swayyaw}
    roll_c, pitch_c = hydro.angular_linear_damping[:2]
    status = ("v2 registry %s (schema %d); roll/pitch damping %s -> %.6g / %.6g N m s/rad; "
              "sway/yaw damping %s -> sway %.6g N s/m + %.6g N s2/m2, yaw %.6g N m s/rad + %.6g N m s2/rad2; "
              "PENDING owner decisions: roll/pitch damping (R3 2), WAM-V option (R3 5), pitch coupling "
              "and motor lag (R3 6); Froude-Krylov %s (R3 7)"
              % (spec.name, spec.schema_version, rule, roll_c, pitch_c, swayyaw, sway[0], sway[1], yaw[0], yaw[1],
                 "on" if froude_krylov_horizontal else "OFF"))
    sources = {"registry": "tasks/_shared/vehicles_v2.py get_vehicle_v2(%r)" % spec.name,
               "hydro": "vehicles_v2.to_d3_hydro_kwargs",
               "thrusters": "vehicles_v2.to_d3_thruster_kwargs(pitch_coupling=%r)" % pitch_coupling,
               "rollpitch_damping": ("decision 2 zeta rule: 2*zeta*sqrt(K*I) with registry K_roll/I_xx, "
                                     "K_pitch/I_yy" if pair is not None else rule),
               "swayyaw_damping": swayyaw_record,
               "asset": "%s sha256 %s" % (spec.asset_relpath, spec.asset_sha256),
               "options": options}
    return VesselProfileV2(spec.name, hydro, thrusters, ZERO_OFFSET, IDENTITY_BASIS, status, sources)


__all__ = ["VesselProfileV2", "profile_for_vehicle", "JOURNAL_DEFAULT", "IDENTITY_BASIS", "SWAYYAW_RULES",
           "WAMV_ROLLPITCH_JOURNAL_DEFAULT", "WAMV_ROLLPITCH_DECISION",
           "crossflow_inputs", "swayyaw_profile_spec"]
