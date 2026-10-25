"""Moving basin: the task's own motion-dose axis, with no second hidden multiplier.

Motion laws mirror the C4 copy of the environment exactly:
  translate (tasks/hazard_nav/hazard_nav_env.py:1714-1728):
      raw = (speed*t + phase) mod (2*span); offset = fold(raw) - span/2
  orbit (tasks/hazard_nav/hazard_nav_env.py:1730-1741):
      theta = omega*t + phase0, omega = tangential speed / orbit radius
Legacy speeds are mu * MOVING_TOP_SPEED_MS (hazard_geometry.py:2819,3171-3172),
a fixed BlueBoat constant; here the reference is the resolved vessel's
top_speed_mps. The legacy evaluation convention pins moving_mu and moving_nu
to the SAME rung (hazard_nav_env_cfg.py:1129-1136), so the default policy is
"tied": nu = mu. "fixed" keeps nu at a declared constant for every dose.
D0 (mu = 0, tied nu = 0) freezes every object at its time-zero position.
"""

from dataclasses import dataclass, replace
import math

from .model import Encounter, check_number

NATIVE_RATIOS = (0.0, 0.04, 0.08, 0.16, 0.30)
NATIVE_BASELINE_RATIO = 0.08
NU_POLICIES = ("tied", "fixed")


@dataclass(frozen=True)
class NativeMotion:
    kind: str
    cx: float
    cy: float
    radius_m: float
    speed_mps: float
    dx: float = 1.0
    dy: float = 0.0
    span_m: float = 1.0
    phase_m: float = 0.0
    orbit_radius_m: float = 1.0
    phase_rad: float = 0.0
    reveal_s: float = 0.0
    reveal_kind: str = "native_visible"

    def __post_init__(self):
        if self.kind not in ("static", "translate", "orbit"):
            raise ValueError("unknown native motion")
        for name in ("cx", "cy", "radius_m", "speed_mps", "dx", "dy", "span_m", "phase_m",
                     "orbit_radius_m", "phase_rad", "reveal_s"):
            check_number(name, getattr(self, name))
        if self.radius_m <= 0 or self.speed_mps < 0 or self.span_m <= 0 or self.orbit_radius_m <= 0:
            raise ValueError("invalid native dimensions or speed")
        if self.reveal_s != 0 or self.reveal_kind != "native_visible":
            raise ValueError("native geometry is present and visible from reset")
        if self.kind == "translate" and abs(math.hypot(self.dx, self.dy) - 1) > 1e-9:
            raise ValueError("translation direction must be normalized")

    @property
    def speed_bound_mps(self):
        return 0.0 if self.kind == "static" else self.speed_mps

    def position(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("invalid episode time")
        if self.kind == "static":
            return self.cx, self.cy
        if self.kind == "translate":
            raw = (self.speed_mps * time_s + self.phase_m) % (2 * self.span_m)
            offset = (raw if raw <= self.span_m else 2 * self.span_m - raw) - self.span_m / 2
            return self.cx + self.dx * offset, self.cy + self.dy * offset
        angle = self.phase_rad + self.speed_mps * time_s / self.orbit_radius_m
        return (self.cx + self.orbit_radius_m * math.cos(angle),
                self.cy + self.orbit_radius_m * math.sin(angle))


def native_speeds(dose, envelope, nu_policy="tied", fixed_nu=0.0):
    if type(dose) is not int or not 0 <= dose <= 4:
        raise ValueError("dose must be D0..D4")
    if nu_policy not in NU_POLICIES:
        raise ValueError("unknown nu policy")
    mu = NATIVE_RATIOS[dose]
    nu = mu if nu_policy == "tied" else float(fixed_nu)
    check_number("nu", nu, minimum=0.0)
    if mu >= 1.0 or nu >= 1.0:
        raise ValueError("native ratios must stay below the vessel top speed")
    return mu, nu, mu * envelope.top_speed_mps, nu * envelope.top_speed_mps


def native_encounter(key, dose, envelope, geometry, duration_s, nu_policy="tied", fixed_nu=0.0):
    """geometry: frozen tuple of NativeMotion templates; template speeds are replaced.

    Object count, positions, spans, phases and orbit radius are identical at
    every dose; only the translation (mu) and orbital (nu) speeds change.
    """
    if not geometry or not all(isinstance(obj, NativeMotion) for obj in geometry):
        raise ValueError("moving basin requires its fixed native population")
    mu, nu, row_speed, orbit_speed = native_speeds(dose, envelope, nu_policy, fixed_nu)
    objects = tuple(replace(obj, speed_mps=(row_speed if obj.kind == "translate"
                                            else orbit_speed if obj.kind == "orbit" else 0.0))
                    for obj in geometry)
    return Encounter(key, "moving_basin", dose, envelope.t_avoid_s, None, objects, float(duration_s),
                     (("axis", "native_motion_ratio"), ("mu", mu), ("nu", nu),
                      ("nu_policy", nu_policy), ("reference_top_speed_mps", envelope.top_speed_mps),
                      ("row_speed_mps", row_speed), ("orbit_tangential_speed_mps", orbit_speed),
                      ("other_suite_baseline_mu", NATIVE_BASELINE_RATIO),
                      ("vessel_id", envelope.vessel_id)))
