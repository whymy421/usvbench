"""Vessel-parameterised manoeuvring envelopes for D7 (pure Python, SI units).

Every task runs on BOTH vessels (owner decision 2026-09-30), so every D7
quantity that depends on the ship -- t_avoid, the swept collision footprint,
intruder size (in beams), witness bounds, moving-basin reference speed -- is
derived from one VesselPhysics record. D7 does not own vessel numbers: D5
does (CONTEXT_R3 decision 1: ONE source, D5's final registry
tasks/_shared/vehicles_v2.py + vehicles_v2_data.py).

E4 integration (2026-09-30): the PROVISIONAL presets that D7 copied from D5's
in-progress draft were REMOVED. Envelopes are built only from a final
VesselSpecV2 record with envelope_from_spec_v2 (duck-typed; this package never
imports the registry). In the simulator pass `get_vehicle_v2(name)` from
tasks/_shared/vehicles_v2.py; CPU tools and tests may use
load_registry_by_path().

Planar thrust convention (CONTEXT_R3 decision 1; D5 PARAMS.md section 9
item 2; D5 vehicles_v2.py:40-46 and Propulsion at :134-142): the planar arm is
measured from the hull CENTRELINE and there is NO COM-offset lever:
  straight capacity ahead   T_fwd = sum of the per-motor forward limits
  straight capacity astern  T_rev = sum of the per-motor reverse limits
  zero-surge yaw torque     N_0   = 2 * arm * min(fwd, rev)   (per motor)
two_motor_capacities (D7's COM-lever version) is kept only as an audit /
comparison function; it is not used to build journal envelopes.

Journal parameters (implemented as parameters with the controller defaults
and stamped in the envelope provenance; G28: the reference speed 1.0 m/s is
owner-decided (CONTEXT_R6 section 3), R = L stays open; the provenance
strings still read "PENDING" because they are part of the D7 pool identity
and are left unchanged until a planned pool re-admission):
  reference (cruise) speed 1.0 m/s for BOTH vessels (CONTEXT_R3 decision 14);
  design turning radius R = L, the COM-relative footprint length (D4
    TASK_CONTRACTS_v2.md:73 and journal_cfg_v2.py VesselRecord.turning_radius_m;
    in round F both values are read from D4's journal_cfg_v2.py);
  witness speed-cap fraction 0.8 for both vessels (D7's 0.95 special case for
    the WAM-V existed only because of its former 2.0 m/s reference).

Derivation (declared, conservative constant bounds for a bounded-control
unicycle; not a hydrodynamic manoeuvring model):
  terminal speed  u_t solves  T_fwd = a*u + b*u^2      (surge damping a, b)
  witness cap     u_cap = min(top, speed_cap_fraction * u_t)
  accel bound     (T_fwd - D(u_cap)) / m   (D increases with u, so it holds
                                            everywhere on [0, u_cap])
  decel bound     T_rev / m                (drag only helps braking at u>0)
  astern cap      v_rev = reverse_cap_fraction * astern terminal speed
  astern accel    (T_rev - D(v_rev)) / m   (bound while moving astern; the
                                            witness uses min(decel, astern))
  yaw-rate bound  u_ref / R_turn           (declared design turning radius,
                                            the same number D4 freezes), and it
                                            must not exceed the steady spin
                                            rate solving N_0 = c*r + d*r^2.
"""

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
import types

from .model import Footprint, VesselEnvelope, check_number

# Journal defaults (see module docstring; reference speed owner-decided, G28 label note there).
JOURNAL_REFERENCE_SPEED_MPS = 1.0
JOURNAL_SPEED_CAP_FRACTION = 0.8
JOURNAL_REACTION_S = 0.5
# D5 registry names of the journal vessels; DEFAULT_WAMV_PROFILE mirrors
# D5 vehicles_v2.py:888 (CONTEXT_R3 decision 5; decided by the owner 2026-10-01, A4).
JOURNAL_VESSELS = ("blueboat_v2", "wamv_vrx_v2_opt1", "wamv_vrx_v2_opt2")
DEFAULT_WAMV_PROFILE = "wamv_vrx_v2_opt1"
PENDING_PARAMETERS = (
    "reference_speed_mps default 1.0 for both vessels (CONTEXT_R3 decision 14, PENDING)",
    "turning_radius_m default R = L footprint length (D4 TASK_CONTRACTS_v2.md:73, PENDING)",
    "speed_cap_fraction 0.8 for both vessels (E4 brief step 3)",
    "reaction_s 0.5 declared engineering latency (D7 DYNAMIC_HAZARDS.md section 1.1, owner may change)",
)


def _quadratic_speed(force, linear, quadratic):
    """Positive root of quadratic*u^2 + linear*u = force."""
    check_number("force", force, positive=True)
    check_number("linear", linear, minimum=0.0)
    check_number("quadratic", quadratic, minimum=0.0)
    if quadratic == 0.0:
        if linear == 0.0:
            raise ValueError("zero damping gives no terminal speed")
        return force / linear
    return (-linear + math.sqrt(linear * linear + 4.0 * quadratic * force)) / (2.0 * quadratic)


def two_motor_capacities(port_fwd, port_rev, stbd_fwd, stbd_rev, port_arm, stbd_arm):
    """AUDIT ONLY (superseded by CONTEXT_R3 decision 1): COM-lever capacities.

    port_arm / stbd_arm are the unsigned lateral distances of each motor line
    from the COM. Straight thrust must satisfy T_p*port_arm = T_s*stbd_arm (no
    yaw), so the weaker side limits the total. Zero net surge needs equal and
    opposite motor forces f <= the smaller of one side's forward and the other
    side's reverse limit. Journal envelopes use centreline_capacities instead.
    """
    for name, value in (("port_fwd", port_fwd), ("port_rev", port_rev), ("stbd_fwd", stbd_fwd),
                        ("stbd_rev", stbd_rev), ("port_arm", port_arm), ("stbd_arm", stbd_arm)):
        check_number(name, value, positive=True)

    def straight(p_lim, s_lim):
        # T_p = k*stbd_arm, T_s = k*port_arm, largest k within both limits
        k = min(p_lim / stbd_arm, s_lim / port_arm)
        return k * (stbd_arm + port_arm)

    fwd = straight(port_fwd, stbd_fwd)
    rev = straight(port_rev, stbd_rev)
    spin = max(min(port_fwd, stbd_rev), min(stbd_fwd, port_rev))
    torque = spin * (port_arm + stbd_arm)
    return fwd, rev, torque


def centreline_capacities(port_fwd, port_rev, stbd_fwd, stbd_rev, arm):
    """Straight ahead/astern capacity and zero-surge yaw torque, centreline arm.

    CONTEXT_R3 decision 1 (D5 PARAMS.md section 9 item 2): each motor line is
    `arm` from the hull centreline and hull resistance acts on the centreline,
    so full thrust on both motors is yaw-free: T_fwd = port_fwd + stbd_fwd,
    T_rev = port_rev + stbd_rev. Zero net surge needs one motor ahead and the
    other astern with equal magnitude f <= min(forward of one, reverse of the
    other): N_0 = 2 * arm * f, i.e. 2 * arm * min(fwd, rev) for identical
    motors (D5 Propulsion.pure_yaw_limit_nm, vehicles_v2.py:139,371,495).
    """
    for name, value in (("port_fwd", port_fwd), ("port_rev", port_rev), ("stbd_fwd", stbd_fwd),
                        ("stbd_rev", stbd_rev), ("arm", arm)):
        check_number(name, value, positive=True)
    spin = max(min(port_fwd, stbd_rev), min(stbd_fwd, port_rev))
    return port_fwd + stbd_fwd, port_rev + stbd_rev, 2.0 * arm * spin


@dataclass(frozen=True)
class VesselPhysics:
    """Minimal planar physics inputs; every field must carry a source in `sources`.

    thrust_fwd_total_n / thrust_rev_total_n are yaw-free straight capacities
    (centreline_capacities for journal vessels), not raw per-motor limits.
    """

    vessel_id: str
    mass_kg: float
    thrust_fwd_total_n: float
    thrust_rev_total_n: float
    surge_lin: float
    surge_quad: float
    yaw_torque_zero_surge_nm: float
    yaw_lin: float
    yaw_quad: float
    footprint: Footprint
    reference_speed_mps: float
    turning_radius_m: float
    top_speed_mps: float = 0.0          # 0 -> use the computed terminal speed
    reaction_s: float = 0.5
    speed_cap_fraction: float = 0.8
    sources: tuple = ()
    status: str = "UNSPECIFIED"
    reverse_cap_fraction: float = 0.5   # astern witness speed cap / astern terminal speed

    def __post_init__(self):
        if not isinstance(self.vessel_id, str) or not self.vessel_id:
            raise ValueError("vessel_id must be a nonempty string")
        for name in ("mass_kg", "thrust_fwd_total_n", "thrust_rev_total_n",
                     "yaw_torque_zero_surge_nm", "reference_speed_mps", "turning_radius_m"):
            check_number(name, getattr(self, name), positive=True)
        for name in ("surge_lin", "surge_quad", "yaw_lin", "yaw_quad", "top_speed_mps", "reaction_s"):
            check_number(name, getattr(self, name), minimum=0.0)
        check_number("speed_cap_fraction", self.speed_cap_fraction, positive=True)
        if self.speed_cap_fraction > 1.0:
            raise ValueError("speed_cap_fraction must be in (0, 1]")
        check_number("reverse_cap_fraction", self.reverse_cap_fraction, minimum=0.0)
        if self.reverse_cap_fraction >= 1.0:
            raise ValueError("reverse_cap_fraction must be in [0, 1)")
        if not isinstance(self.footprint, Footprint):
            raise ValueError("footprint must be a Footprint")
        if not isinstance(self.sources, tuple):
            raise ValueError("sources must be a tuple of citation strings")

    @property
    def terminal_speed_mps(self):
        return _quadratic_speed(self.thrust_fwd_total_n, self.surge_lin, self.surge_quad)

    @property
    def astern_terminal_speed_mps(self):
        """Declared: astern drag uses the ahead surge coefficients (UNVERIFIED)."""
        return _quadratic_speed(self.thrust_rev_total_n, self.surge_lin, self.surge_quad)

    @property
    def steady_spin_rate_rps(self):
        return _quadratic_speed(self.yaw_torque_zero_surge_nm, self.yaw_lin, self.yaw_quad)

    def surge_drag(self, speed):
        return self.surge_lin * speed + self.surge_quad * speed * speed


def envelope_from_physics(physics, t_avoid_method=None, clearance_beams=1.0):
    """Build the frozen VesselEnvelope and a derivation record (for stamps/docs)."""
    if not isinstance(physics, VesselPhysics):
        raise ValueError("physics must be a VesselPhysics record")
    terminal = physics.terminal_speed_mps
    top = terminal if physics.top_speed_mps == 0.0 else min(physics.top_speed_mps, terminal)
    cap = min(top, physics.speed_cap_fraction * terminal)
    accel = (physics.thrust_fwd_total_n - physics.surge_drag(cap)) / physics.mass_kg
    decel = physics.thrust_rev_total_n / physics.mass_kg
    reverse_cap = physics.reverse_cap_fraction * physics.astern_terminal_speed_mps
    astern_accel = (physics.thrust_rev_total_n - physics.surge_drag(reverse_cap)) / physics.mass_kg
    yaw_rate = physics.reference_speed_mps / physics.turning_radius_m
    spin = physics.steady_spin_rate_rps
    if accel <= 0.0:
        raise ValueError("no acceleration reserve at the witness speed cap")
    if physics.reference_speed_mps > cap:
        raise ValueError("reference speed %.4f exceeds witness cap %.4f; lower the reference "
                         "or raise speed_cap_fraction" % (physics.reference_speed_mps, cap))
    if yaw_rate > spin:
        raise ValueError("declared turning radius needs a yaw rate above the steady spin rate")
    kwargs = {}
    if t_avoid_method is not None:
        kwargs["t_avoid_method"] = t_avoid_method
    provenance = "; ".join((physics.status,) + physics.sources)
    envelope = VesselEnvelope(physics.vessel_id, physics.footprint, physics.reference_speed_mps,
                              cap, top, accel, decel, yaw_rate, physics.reaction_s,
                              provenance, clearance_beams=clearance_beams,
                              max_reverse_speed_mps=reverse_cap,
                              max_astern_accel_mps2=astern_accel if reverse_cap > 0.0 else 0.0,
                              **kwargs)
    derivation = (("terminal_speed_mps", terminal), ("top_speed_mps", top),
                  ("witness_speed_cap_mps", cap), ("accel_bound_mps2", accel),
                  ("decel_bound_mps2", decel), ("reverse_speed_cap_mps", reverse_cap),
                  ("astern_accel_bound_mps2", astern_accel), ("yaw_rate_bound_rps", yaw_rate),
                  ("steady_spin_rate_rps", spin), ("t_avoid_s", envelope.t_avoid_s),
                  ("t_avoid_method", envelope.t_avoid_method)) + tuple(envelope.t_avoid_components)
    return envelope, derivation


def footprint_from_polygon(polygon_xy, com_xy):
    """Axis-aligned COM-relative rectangle enclosing a body-frame polygon."""
    xs = [float(p[0]) - float(com_xy[0]) for p in polygon_xy]
    ys = [float(p[1]) - float(com_xy[1]) for p in polygon_xy]
    return Footprint(min(xs), max(xs), min(ys), max(ys))


def _close(a, b, rel=1e-9, absolute=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=absolute)


def physics_from_spec_v2(spec, *, reference_speed_mps=JOURNAL_REFERENCE_SPEED_MPS, turning_radius_m=None,
                         reaction_s=JOURNAL_REACTION_S, speed_cap_fraction=JOURNAL_SPEED_CAP_FRACTION):
    """VesselPhysics from a FINAL D5 VesselSpecV2 (duck-typed; D5 vehicles_v2.py:181-209).

    Reads (final schema): schema_version, bow_body_axis, com_actor_m (must be
    exactly zero: v2 actor origin = COM, CONTEXT_R3 decision 12), mass_kg,
    damping.linear/quadratic (index 0 surge, 5 yaw), propulsion.motors
    (port, starboard) forward_limit_n / reverse_limit_n / position_com_m /
    direction_body, propulsion.planar_arm_m and pure_yaw_limit_nm (cross-check),
    collision.polygon_com_xy_m (about the COM), speeds.top_model_m_s
    (cross-checked against the terminal speed of the same laws) and
    speeds.reference_m_s (recorded only; the journal reference is a parameter).
    turning_radius_m None -> R = L, the footprint length (D4 rule).
    Returns (VesselPhysics, inputs) where inputs is an immutable record of
    what was read, for stamps and the dose table.
    """
    name = spec.name
    if getattr(spec, "schema_version", None) != 2:
        raise ValueError("envelope_from_spec_v2 needs a schema-2 VesselSpecV2")
    if getattr(spec, "bow_body_axis", "+x") != "+x":
        raise ValueError("v2 assets are FLU with the bow on +x")
    com = tuple(float(v) for v in spec.com_actor_m)
    if com != (0.0, 0.0, 0.0):
        raise ValueError("v2 actor origin must be the COM (com_actor_m == 0); got %r" % (com,))
    propulsion = spec.propulsion
    motors = tuple(propulsion.motors)
    if len(motors) != 2 or [m.name for m in motors] != ["port", "starboard"]:
        raise ValueError("exactly (port, starboard) motors are supported")
    port, stbd = motors
    for motor in motors:
        if tuple(float(v) for v in motor.direction_body) != (1.0, 0.0, 0.0):
            raise ValueError("motors must push along +x")
    arm = float(propulsion.planar_arm_m)
    check_number("planar_arm_m", arm, positive=True)
    centre = 0.5 * (float(port.position_com_m[1]) + float(stbd.position_com_m[1]))
    if not (_close(float(port.position_com_m[1]) - centre, arm, 0.0, 1e-9)
            and _close(centre - float(stbd.position_com_m[1]), arm, 0.0, 1e-9)):
        raise ValueError("motors must be mirror images about the hull centreline at +/- planar_arm_m")
    fwd, rev, torque = centreline_capacities(float(port.forward_limit_n), float(port.reverse_limit_n),
                                             float(stbd.forward_limit_n), float(stbd.reverse_limit_n), arm)
    registry_torque = float(propulsion.pure_yaw_limit_nm)
    if not _close(torque, registry_torque):
        raise ValueError("zero-surge yaw torque %.9g disagrees with the registry pure_yaw_limit_nm %.9g"
                         % (torque, registry_torque))
    lin, quad = spec.damping.linear, spec.damping.quadratic
    footprint = footprint_from_polygon(spec.collision.polygon_com_xy_m, (0.0, 0.0))
    top = float(spec.speeds.top_model_m_s)
    terminal = _quadratic_speed(fwd, float(lin[0]), float(quad[0]))
    if not _close(top, terminal):
        raise ValueError("registry top_model_m_s %.9g is not the terminal speed %.9g of its own laws"
                         % (top, terminal))
    radius = footprint.length_m if turning_radius_m is None else float(turning_radius_m)
    radius_source = ("R = L = %.6f m footprint length (D4 TASK_CONTRACTS_v2.md:73, PENDING)" % radius
                     if turning_radius_m is None else "R = %.6f m supplied by caller" % radius)
    sources = (
        "D5 final registry VesselSpecV2 %s (vehicles_v2.py + vehicles_v2_data.py)" % name,
        "centreline arm %.6f m: T_fwd = sum fwd, T_rev = sum rev, N_0 = 2*arm*min(fwd, rev) "
        "(CONTEXT_R3 decision 1; D5 PARAMS.md section 9 item 2)" % arm,
        "reference speed %.6f m/s (CONTEXT_R3 decision 14, PENDING; registry reference_m_s %.6f)"
        % (float(reference_speed_mps), float(spec.speeds.reference_m_s)),
        radius_source,
        "witness speed cap fraction %.3f; reaction %.3f s" % (float(speed_cap_fraction), float(reaction_s)),
    )
    physics = VesselPhysics(
        vessel_id=name, mass_kg=float(spec.mass_kg), thrust_fwd_total_n=fwd, thrust_rev_total_n=rev,
        surge_lin=float(lin[0]), surge_quad=float(quad[0]), yaw_torque_zero_surge_nm=torque,
        yaw_lin=float(lin[5]), yaw_quad=float(quad[5]), footprint=footprint,
        reference_speed_mps=float(reference_speed_mps), turning_radius_m=radius, top_speed_mps=top,
        reaction_s=float(reaction_s), speed_cap_fraction=float(speed_cap_fraction), sources=sources,
        status="D5_FINAL_REGISTRY_E4")
    inputs = (("vessel_id", name), ("mass_kg", float(spec.mass_kg)),
              ("motor_forward_limit_n", float(port.forward_limit_n)),
              ("motor_reverse_limit_n", float(port.reverse_limit_n)),
              ("planar_arm_m", arm), ("straight_thrust_fwd_n", fwd), ("straight_thrust_rev_n", rev),
              ("zero_surge_yaw_torque_nm", torque), ("surge_lin", float(lin[0])),
              ("surge_quad", float(quad[0])), ("yaw_lin", float(lin[5])), ("yaw_quad", float(quad[5])),
              ("registry_top_model_mps", top), ("registry_reference_mps", float(spec.speeds.reference_m_s)),
              ("journal_reference_mps", float(reference_speed_mps)), ("turning_radius_m", radius),
              ("speed_cap_fraction", float(speed_cap_fraction)), ("reaction_s", float(reaction_s)))
    return physics, inputs


def envelope_from_spec_v2(spec, *, reference_speed_mps=JOURNAL_REFERENCE_SPEED_MPS, turning_radius_m=None,
                          reaction_s=JOURNAL_REACTION_S, speed_cap_fraction=JOURNAL_SPEED_CAP_FRACTION,
                          t_avoid_method=None, clearance_beams=1.0):
    """Frozen VesselEnvelope from a final D5 VesselSpecV2 (see physics_from_spec_v2).

    Returns (envelope, derivation); derivation = registry inputs + envelope
    derivation (terminal/top/cap/bounds/t_avoid and its components).
    """
    physics, inputs = physics_from_spec_v2(spec, reference_speed_mps=reference_speed_mps,
                                           turning_radius_m=turning_radius_m, reaction_s=reaction_s,
                                           speed_cap_fraction=speed_cap_fraction)
    envelope, derivation = envelope_from_physics(physics, t_avoid_method=t_avoid_method,
                                                 clearance_beams=clearance_beams)
    return envelope, inputs + derivation


def load_registry_by_path(shared_dir, package_name="_e4_vessel_registry"):
    """CPU tools/tests only: load D5's vehicles_v2.py from `shared_dir` by path.

    The folder is bound to a bare namespace module (its __init__.py is NOT
    executed, so no task package or simulator is imported) and vehicles_v2
    is imported as a submodule so its relative import of vehicles_v2_data
    works. Returns the vehicles_v2 module. In the simulator, import
    tasks._shared.vehicles_v2 normally instead.
    """
    shared = Path(shared_dir).resolve()
    full = package_name + ".vehicles_v2"
    if full in sys.modules:
        module = sys.modules[full]
        if Path(module.__file__).resolve().parent != shared:
            raise ValueError("a different registry is already loaded under " + package_name)
        return module
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(shared)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(full, shared / "vehicles_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full, None)
        raise
    return module
