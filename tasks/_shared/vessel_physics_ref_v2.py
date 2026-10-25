"""Pure-Python reference calculations for the v2 vessel profiles (D5).

No numpy, torch, Isaac or simulator imports: the functions are used by the
asset builder (to derive profile constants), by the analysis script and by the
CPU tests (to re-derive the registry constants independently of stored text).

Frames: body x bow, y port, z up (FLU). Angles in radians, SI units.

Sources for the modelled VRX laws (local checkout pinned at osrf/vrx
7609d1bd90ce7edb29d040a082f949e8b089c864, and classic branch
c9b9388308f8976c724da4af6685f69c9c378983 downloaded to work/evidence/web/):
- Surface buoyancy: vrx_gz/src/Surface.cc:229-261 (per-point vertical force
  rho*g*A(R, clip(depth,0,R))*L/n_points), CircleSegment :302-306.
- Classic thrust GLF: evidence/web/C_usv_gazebo_thrust_plugin.cc:318-341.
"""

from __future__ import annotations

import math
from typing import Sequence

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------- linear algebra
def rpy_matrix(roll: float, pitch: float, yaw: float) -> tuple[Vec3, Vec3, Vec3]:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def mat_vec(m, v) -> Vec3:
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def add(a, b) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))


def scale(a, s) -> Vec3:
    return tuple(x * s for x in a)


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def solve3(a, b) -> Vec3:
    """Solve a 3x3 linear system by Cramer's rule (well-conditioned here)."""
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det(a)
    if abs(d) < 1e-300:
        raise ZeroDivisionError("singular 3x3 system")
    out = []
    for col in range(3):
        m = [list(row) for row in a]
        for row in range(3):
            m[row][col] = b[row]
        out.append(det(m) / d)
    return tuple(out)


# ---------------------------------------------------------------- mass aggregation
def aggregate_inertia(parts: Sequence[tuple[float, Vec3, tuple]]) -> tuple[float, Vec3, tuple]:
    """Sum (mass, com, inertia-about-own-com in common axes) with parallel axes.

    Returns (total mass, total COM, inertia about the total COM).
    I = sum(I_i + m_i * ((d.d) E - d d^T)), d = com_i - COM.
    """
    total = sum(m for m, _, _ in parts)
    if not total > 0:
        raise ValueError("total mass must be positive")
    com = tuple(sum(m * c[k] for m, c, _ in parts) / total for k in range(3))
    tensor = [[0.0] * 3 for _ in range(3)]
    for m, c, inertia in parts:
        d = sub(c, com)
        dd = dot(d, d)
        for i in range(3):
            for j in range(3):
                tensor[i][j] += inertia[i][j] + m * ((dd if i == j else 0.0) - d[i] * d[j])
    return total, com, tuple(tuple(row) for row in tensor)


def rotate_inertia(rot, inertia):
    """R I R^T: inertia expressed in the parent axes."""
    return mat_mul(mat_mul(rot, inertia), transpose(rot))


# ---------------------------------------------------------------- VRX Surface buoyancy
def circle_segment_area(radius: float, depth: float) -> float:
    """VRX Surface.cc:302-306, valid for depth in [0, radius] (clipped by caller)."""
    return radius * radius * math.acos((radius - depth) / radius) - (
        radius - depth) * math.sqrt(max(0.0, 2.0 * radius * depth - depth * depth))


def circle_segment_chord(radius: float, depth: float) -> float:
    """d(area)/d(depth) = chord length 2*sqrt(2 R h - h^2) for 0 < h < R."""
    return 2.0 * math.sqrt(max(0.0, 2.0 * radius * depth - depth * depth))


def vrx_surface_generalized_force(com_world_z: float, roll: float, pitch: float, *,
                                  mass: float, com_body: Vec3, points_body: Sequence[Vec3],
                                  hull_length: float, hull_radius: float,
                                  rho: float, g: float, points_per_plugin: int = 2):
    """Net vertical force (buoyancy - weight) and buoyancy moments about the COM.

    The body frame is the VRX base_link frame; points are the Surface plugin
    points (dynamics xacro). Each point carries hull_length / points_per_plugin
    of hull (Surface.cc:252-255; one plugin instance per hull, two points each).
    Moments are about world x and y axes through the COM. Water level is 0.
    """
    rot = mat_mul(rpy_matrix(0.0, pitch, 0.0), rpy_matrix(roll, 0.0, 0.0))
    com_rot = mat_vec(rot, com_body)
    origin_z = com_world_z - com_rot[2]
    seg_len = hull_length / points_per_plugin
    fz, mx, my = -mass * g, 0.0, 0.0
    for p in points_body:
        lever = mat_vec(rot, sub(p, com_body))
        world_z = origin_z + mat_vec(rot, p)[2]
        depth = min(max(-world_z, 0.0), hull_radius)
        force = circle_segment_area(hull_radius, depth) * seg_len * g * rho
        fz += force
        mx += lever[1] * force
        my += -lever[0] * force
    return fz, mx, my


def vrx_surface_hydrostatics(*, mass: float, com_body: Vec3, points_body: Sequence[Vec3],
                             hull_length: float, hull_radius: float, rho: float, g: float,
                             points_per_plugin: int = 2) -> dict:
    """Free-floating equilibrium and COM-referenced 3x3 stiffness of the VRX model.

    Unknowns: COM world height zc, roll, pitch. Newton with central differences.
    Stiffness K = -d(Fz, Mx, My)/d(zc, roll, pitch) at the equilibrium, i.e. the
    total small-motion hydrostatic stiffness about the COM including the
    -W*(z_G - z_points) term (buoyancy points below the COM).
    """
    kw = dict(mass=mass, com_body=com_body, points_body=points_body,
              hull_length=hull_length, hull_radius=hull_radius, rho=rho, g=g,
              points_per_plugin=points_per_plugin)

    def q(state):
        return vrx_surface_generalized_force(state[0], state[1], state[2], **kw)

    def jac(state, h=1e-7):
        cols = []
        for j in range(3):
            up = list(state)
            dn = list(state)
            up[j] += h
            dn[j] -= h
            fu, fd = q(up), q(dn)
            cols.append(tuple((a - b) / (2 * h) for a, b in zip(fu, fd)))
        return tuple(tuple(cols[j][i] for j in range(3)) for i in range(3))

    # Initial guess: upright, depth such that uniform immersion supports weight.
    n_pts = len(points_body)
    seg_len = hull_length / points_per_plugin
    lo, hi = 0.0, hull_radius
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if n_pts * circle_segment_area(hull_radius, mid) * seg_len * rho < mass:
            lo = mid
        else:
            hi = mid
    uniform_depth = 0.5 * (lo + hi)
    state = [com_body[2] - uniform_depth, 0.0, 0.0]
    for _ in range(60):
        res = q(state)
        step = solve3(jac(state), tuple(-r for r in res))
        state = [s + d for s, d in zip(state, step)]
        if max(abs(d) for d in step) < 1e-13:
            break
    residual = q(state)
    k_neg = jac(state)
    stiffness = tuple(tuple(-k_neg[i][j] for j in range(3)) for i in range(3))
    # Per-point immersion at equilibrium.
    rot = mat_mul(rpy_matrix(0.0, state[2], 0.0), rpy_matrix(state[1], 0.0, 0.0))
    origin_z = state[0] - mat_vec(rot, com_body)[2]
    depths = [min(max(-(origin_z + mat_vec(rot, p)[2]), 0.0), hull_radius) for p in points_body]
    max_volume = n_pts * seg_len * math.pi * hull_radius ** 2 / 2.0
    # Upright heave-only equilibrium for comparison.
    return {
        "com_world_z_m": state[0],
        "roll_rad": state[1],
        "pitch_rad": state[2],
        "origin_world_z_m": origin_z,
        "point_depths_m": depths,
        "uniform_upright_depth_m": uniform_depth,
        "residual_force_n": residual[0],
        "residual_moment_nm": (residual[1], residual[2]),
        "stiffness_heave_n_m": stiffness[0][0],
        "stiffness_roll_nm_rad": stiffness[1][1],
        "stiffness_pitch_nm_rad": stiffness[2][2],
        "stiffness_matrix_z_roll_pitch": stiffness,
        "equivalent_waterplane_m2": stiffness[0][0] / (rho * g),
        "displaced_volume_m3": mass / rho,
        "maximum_volume_m3": max_volume,
    }


# ---------------------------------------------------------------- VRX classic GLF thrust
def _f32(x: float) -> float:
    """Round to IEEE single precision (the classic plugin passes float constants)."""
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def classic_glf_thrust(cmd: float, max_pos: float = 250.0, max_neg: float = -100.0) -> float:
    """Per-motor thrust [N] of VRX classic usv_gazebo_thrust_plugin, mappingType 1.

    Reproduces C_usv_gazebo_thrust_plugin.cc:318-341 exactly, including its
    branch quirk: cmd > 0.01 uses the forward curve, cmd < 0.01 (including 0)
    the reverse curve, and cmd == 0.01 returns 0. maxCmd is 1.0 in the classic
    thruster config (C_wamv_gazebo_thruster_config.xacro:15), so cmd is the
    normalized command. Clipping values from :16-17.
    """
    def glf(x, a, k, b, v, c, m):
        a, k, b, v, c, m = (_f32(t) for t in (a, k, b, v, c, m))
        return a + (k - a) / (math.pow(c + math.exp(-b * (x - m)), 1.0 / v))
    if cmd > 0.01:
        return min(glf(cmd, 0.01, 59.82, 5.0, 0.38, 0.56, 0.28), max_pos)
    if cmd < 0.01:
        return max(glf(cmd, -199.13, -0.09, 8.84, 5.34, 0.99, -0.57), max_neg)
    return 0.0


# ---------------------------------------------------------------- planar performance
def terminal_speed(linear: float, quadratic: float, force: float) -> float:
    """Positive root of quadratic*u^2 + linear*u = force (force >= 0)."""
    if force <= 0:
        return 0.0
    if quadratic == 0:
        return force / linear
    return (-linear + math.sqrt(linear * linear + 4.0 * quadratic * force)) / (2.0 * quadratic)


def drag(linear: float, quadratic: float, speed: float) -> float:
    return linear * speed + quadratic * speed * abs(speed)


def max_differential_yaw_moment(arm: float, fwd: float, rev: float) -> float:
    """One motor full ahead, the other full astern (BlueBoat legacy convention,
    blueboat_calm_nav_env_cfg.py:139-143)."""
    return arm * (fwd + rev)


def pure_yaw_moment(arm: float, fwd: float, rev: float) -> float:
    """Zero net surge: both motors at min(fwd, rev) in opposite directions."""
    return 2.0 * arm * min(fwd, rev)


def steady_turn(arm: float, fwd: float, rev: float, surge_lin: float, surge_quad: float,
                yaw_lin: float, yaw_quad: float, speed: float) -> dict:
    """Decoupled planar steady-turn estimate at forward speed `speed`.

    Surge thrust must equal surge drag X=D_u(U); the outer motor is at its
    forward limit and the inner motor supplies X - F_fwd (bounded by reverse).
    Yaw moment N = arm*(S-P) balances yaw drag. Sway drift, Munk moment and
    sway-yaw coupling are ignored: this is an order-of-magnitude radius, not a
    manoeuvring prediction.
    """
    x_req = drag(surge_lin, surge_quad, speed)
    if x_req > 2 * fwd:
        return {"feasible": False, "speed_m_s": speed}
    outer = min(fwd, x_req + rev)
    inner = x_req - outer
    moment = arm * (outer - inner)
    r = terminal_speed(yaw_lin, yaw_quad, moment)
    return {"feasible": True, "speed_m_s": speed, "surge_force_n": x_req,
            "yaw_moment_nm": moment, "yaw_rate_rad_s": r,
            "radius_m": (speed / r) if r > 0 else math.inf}


def explicit_damping_numbers(c_lin: float, c_quad: float, speed: float, inertia: float,
                             dt: float, stiffness: float = 0.0) -> dict:
    """C1 AUDIT_physics.md:26 local explicit-Euler diagnostics.

    A=c*dt/M, Q=b*|v|*dt/M; local multiplier 1-A-2Q; linear stability needs
    0<A+2Q<2 (quadratic derivative doubles Q); spring Jury 2A+B<4, B=k dt^2/M.
    """
    a = c_lin * dt / inertia
    qd = c_quad * abs(speed) * dt / inertia
    b = stiffness * dt * dt / inertia
    eff = a + 2.0 * qd
    stable = (0.0 <= eff < 2.0) and (2.0 * eff + b < 4.0)
    return {"A": a, "Q": qd, "A_plus_2Q": eff, "B": b, "jury_2A_plus_B": 2 * eff + b,
            "stable_explicit": stable, "monotone_explicit": eff < 1.0}


# ---------------------------------------------------------------- wind profile
def log_profile_speed(u10: float, height: float, roughness: float,
                      reference_height: float = 10.0) -> float:
    """Neutral log law U(z)=U_ref*ln(z/z0)/ln(z_ref/z0); z must exceed z0."""
    if height <= roughness or reference_height <= roughness:
        raise ValueError("heights must exceed the roughness length")
    return u10 * math.log(height / roughness) / math.log(reference_height / roughness)


def wind_loads(speed: float, frontal_area: float, lateral_area: float, length: float,
               c_x0: float, c_y0: float, c_n0: float, rho_air: float = 1.225) -> dict:
    """C3 Fossen first-harmonic model peak magnitudes (wind.py FossenCoefficients):
    head wind |X|, beam wind |Y|, peak yaw at 45 degrees |N|."""
    q = 0.5 * rho_air * speed * speed
    return {"head_force_n": q * frontal_area * c_x0,
            "beam_force_n": q * lateral_area * c_y0,
            "peak_yaw_moment_nm": q * lateral_area * length * c_n0}


# ---------------------------------------------------------------- footprint
def enclosing_capsule_for_rectangle(x_min: float, x_max: float, half_beam: float,
                                    y_center: float = 0.0) -> dict:
    """Minimum-area x-aligned capsule containing an axis-aligned rectangle.

    Capsule = segment [cx-a, cx+a] swept by radius r >= half_beam. Containment of
    the corners requires (L/2-a)^2 + b^2 <= r^2; area = pi r^2 + 4 a r.
    """
    half_len = 0.5 * (x_max - x_min)
    cx = 0.5 * (x_max + x_min)
    b = half_beam

    def area_for(r):
        a = max(0.0, half_len - math.sqrt(max(0.0, r * r - b * b)))
        return math.pi * r * r + 4.0 * a * r, a

    lo, hi = b, math.hypot(half_len, b)
    for _ in range(200):  # golden-section search on a unimodal 1-D function
        m1 = lo + (hi - lo) * 0.381966
        m2 = lo + (hi - lo) * 0.618034
        if area_for(m1)[0] <= area_for(m2)[0]:
            hi = m2
        else:
            lo = m1
    r = 0.5 * (lo + hi)
    area, a = area_for(r)
    return {"center_xy_m": (cx, y_center), "segment_half_length_m": a,
            "radius_m": r, "area_m2": area}
