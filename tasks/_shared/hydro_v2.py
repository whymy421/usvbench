"""Single water-relative hull force model. Pure Python/torch; SI units, COM origin.

The hull frame is right handed: x forward, y port, z up. Rotation matrices
map hull vectors to world. A vessel adapter maps an asset's actor frame into
this frame (see INTEGRATION_SPEC.md section 1). No vessel numbers live in this
module: every coefficient comes from a HydroSpec built by hydro_profiles.py
from the D5 v2 registry (E2). No added mass, radiation memory, slamming, diffraction or
capsize prediction is claimed. Hydrostatic coefficients are effective TOTAL
small-angle values about the COM. See DESIGN_NOTES.md for sources, validity
and the spectral-convergence derivation.

One force law, one water velocity:
  u_w = current(t) + filtered wave orbital velocity at the env-local COM
  v_rel = v_hull - u_w (hull axes), damped implicitly (backward Euler) with
  D(v_rel) frozen at this state and D u_w entering as known forcing.
There is no separate wave or current drag term. Zero current and zero waves
give exactly the calm model; a hull drifting with a uniform current feels
zero damping force.

E2 (2026-09-30): horizontal Froude-Krylov inertial wave force (controller
decision 7; D9 DOSE_DESIGN.md section 6). Vessel numbers now come only from
the D5 registry through hydro_profiles.py.

Derivation. For a hull that is small against the wavelength, the
Froude-Krylov force is the integral of the undisturbed incident-wave
pressure over the wetted surface, F = -oint p n dS = -int_V grad p dV
(Gauss). Linearised Euler for the incident field gives
-grad p = rho du/dt + rho g e_z, so
    F_FK = rho V <du/dt>_V + rho g V e_z,
with <.>_V the average over the displaced volume and du/dt the LOCAL time
derivative of the orbital velocity (the convective term u.grad u is second
order in wave steepness and is dropped, as in linear theory). The vertical
part is carried by the existing buoyancy law, which evaluates the local
(footprint-filtered) elevation; nothing vertical is added here. Horizontal:
at calm equilibrium rho V = m (the hull mass), and the volume average is
approximated exactly as for the orbital velocity itself: the Gaussian
footprint transfer H_f per component (waterplane average) and exp(-k d) at
the orbital sampling depth d = draft/2 (draft average). For deep-water
linear waves eta = sum a cos(theta), theta = k.x - omega t + phase, the
sampled horizontal velocity is u_h = sum H_f a omega exp(-k d) cos(theta) e_dir
and its local time derivative is
    a_h = du_h/dt |_x = sum H_f a omega^2 exp(-k d) sin(theta) e_dir,
returned by SpectralWaves.sample as WaveSample.orbital_acceleration_world
(the vertical row, -sum H_f a omega^2 exp(-k d) cos(theta), is returned for
completeness and is unused). The applied force, in world axes, is
    F_fk = m (1 + C_a) (a_h,x, a_h,y, 0),     C_a = 0 declared,
i.e. the Morison inertia term rho V (1 + C_a) a_fluid with the added-mass
reaction -C_a rho V a_body omitted; v2 has no added mass (D5 AddedMass
active_in_v2=False), so any C_a != 0 is rejected rather than half-modelled.
F_fk is known forcing: it enters the translational backward-Euler predictor
next to thrust, wind, buoyancy and gravity, and it is added exactly once to
the returned force_body (hydro_wrench), which the runtime applies once.

Consequences (tested in scripts/tests_hydro_v2/test_froude_krylov.py): the
horizontal equation m dv/dt = F_fk + D (u_h - v) gives, for e = v - u_h,
m de/dt = -D e (+ the second-order convective residue), so a free hull
follows the filtered orbital velocity for ANY damping; without F_fk the hull
velocity is only a first-order low-pass of u_h with time constant m/D
(amplitude 1/sqrt(1 + (omega m/D)^2)). Zero amplitude, calm water (waves
None) or froude_krylov_horizontal=False reproduce the pre-E2 results
exactly. No explicit Stokes-drift term exists (decision 7): drift arises at
second order from sampling at the moving hull position. Because the orbital
velocity is sampled at a FIXED depth below the mean surface (not at the
hull's instantaneous height), only the horizontal-gradient half of the
Lagrangian (Stokes) drift can emerge: a perfectly following hull drifts at
about omega k A^2 / 2 with A = H_f a exp(-k d), half of the classical
omega k A^2; the hull-axis heave damping of a wave-tilted hull reduces it
further. Measured drift / classical u_s in the E2 CPU probes (FK on, dt
1/240 s; PHYSICS_NOTES_E2.md section 2.3): BlueBoat head -0.025, beam 0.321;
WAM-V head 0.324, beam 0.279, i.e. 0-32 % of the classical value. Stokes
drift therefore does NOT appear "automatically" at its classical value
(G28 correction of the decision-record wording, NC 39). Reported as a
diagnostic, not corrected.

Host synchronisation (D2 REVIEW F13): hydro_wrench(..., constants=...) takes
tensors cached at construction (prepare_hydro_constants) and skips the
per-call validation and the error check of the linear solve; results are
bit-identical to the validating path (tested).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

try:
    from .physics_v2 import implicit_damping_wrench
    from .restoring_v2 import restoring_moment
except ImportError:  # Path-loaded CPU tests; never import the tasks package.
    from physics_v2 import implicit_damping_wrench
    from restoring_v2 import restoring_moment


@dataclass(frozen=True)
class HydroSpec:
    """Explicit vessel contract. Values come from a vessel profile, never here.

    maximum_volume_m3 is full model immersion, NOT equilibrium displacement.
    Equilibrium displacement is mass_kg / water_density. waterplane_m2 is the
    local derivative dV/d(draft). equilibrium_com_z_m is the COM height above
    the mean water level at calm equilibrium. The constant-area clipped volume
    law is a small-motion approximation.

    Damping triples are (surge, sway, heave) and (roll, pitch, yaw) in hull
    axes; linear in N s/m or N m s/rad, quadratic in N s^2/m^2 or
    N m s^2/rad^2. roll/pitch stiffness are TOTAL metacentric stiffnesses
    (N m/rad) about the COM; no separate buoyancy-centre couple is added.

    rotational_relative_motion selects the reference of roll/pitch/yaw
    damping: True (default) damps the hull angular velocity RELATIVE to the
    rotation rate of the filtered local wave-surface normal, the rotational
    analogue of the relative vertical velocity used for heave (strip-theory
    relative-motion hypothesis); False damps the inertial angular velocity.
    Both are identical in calm water. See DESIGN_NOTES.md section 3.5.

    froude_krylov_horizontal (E2, controller decision 7, default True) adds
    the horizontal inertial wave force m (1 + C_a) a_h (module docstring);
    False reproduces the pre-E2 model exactly. added_mass_coefficient_horizontal
    is C_a: declared 0 because v2 has no added mass; any other value is
    rejected until an added-mass model (with its body-acceleration reaction)
    exists.
    """
    mass_kg: float
    inertia_kgm2: tuple
    linear_damping: tuple
    quadratic_damping: tuple
    angular_linear_damping: tuple
    angular_quadratic_damping: tuple
    maximum_volume_m3: float
    waterplane_m2: float
    roll_stiffness_nm: float
    pitch_stiffness_nm: float
    length_m: float
    beam_m: float
    orbital_depth_m: float = 0.0
    equilibrium_com_z_m: float = 0.0
    water_density: float = 1000.0
    gravity: float = 9.81
    rotational_relative_motion: bool = True
    froude_krylov_horizontal: bool = True
    added_mass_coefficient_horizontal: float = 0.0

    def __post_init__(self):
        if not isinstance(self.froude_krylov_horizontal, bool):
            raise ValueError("froude_krylov_horizontal must be a bool")
        ca = self.added_mass_coefficient_horizontal
        if isinstance(ca, bool) or not isinstance(ca, (int, float)) or not math.isfinite(ca):
            raise ValueError("added_mass_coefficient_horizontal must be a finite number")
        if ca != 0:
            raise ValueError("added_mass_coefficient_horizontal must be 0: hydro_v2 has no added-mass "
                             "model (decision 7 declares C_a = 0); nonzero values are rejected")
        positive = ("mass_kg", "maximum_volume_m3", "waterplane_m2",
                    "length_m", "beam_m", "water_density", "gravity")
        nonnegative = ("roll_stiffness_nm", "pitch_stiffness_nm", "orbital_depth_m")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(name + " has invalid physical value")
        if not math.isfinite(self.equilibrium_com_z_m):
            raise ValueError("equilibrium COM datum must be finite")
        if not isinstance(self.rotational_relative_motion, bool):
            raise ValueError("rotational_relative_motion must be a bool")
        for name in ("linear_damping", "quadratic_damping", "angular_linear_damping",
                     "angular_quadratic_damping"):
            values = getattr(self, name)
            if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(name + " requires three nonnegative finite values")
        inertia = torch.as_tensor(self.inertia_kgm2, dtype=torch.float64)
        if (inertia.shape != (3, 3) or not torch.isfinite(inertia).all()
                or not torch.allclose(inertia, inertia.T)
                or not (torch.linalg.eigvalsh(inertia) > 0).all()):
            raise ValueError("inertia must be a symmetric positive definite 3x3 tensor")
        if self.mass_kg / self.water_density >= self.maximum_volume_m3:
            raise ValueError("equilibrium displacement must be below full immersion")

    @property
    def heave_stiffness_n_per_m(self) -> float:
        """rho*g*A_wp: the ONE heave stiffness used for waves and hull motion."""
        return self.water_density * self.gravity * self.waterplane_m2


@dataclass(frozen=True)
class WaveSample:
    """Filtered wave kinematics at each env's COM, world frame.

    elevation [N] (m), slope [N,2] (d eta/dx, d eta/dy), orbital velocity
    [N,3] (m/s, including vertical) and the angular velocity of the local
    surface normal [N,3] (rad/s). None for the last field means zero.

    E2: orbital_acceleration_world [N,3] (m/s^2) is the LOCAL time
    derivative of orbital_velocity_world (same components, amplitudes,
    phases, footprint transfer, depth attenuation and sampling depth);
    hydro_wrench uses its horizontal rows for the Froude-Krylov force. None
    means zero, like surface_angular_velocity_world: SpectralWaves.sample
    and calm() always fill it; only a hand-built sample can omit it.
    """
    elevation: torch.Tensor
    slope: torch.Tensor
    orbital_velocity_world: torch.Tensor
    surface_angular_velocity_world: torch.Tensor | None = None
    orbital_acceleration_world: torch.Tensor | None = None

    @classmethod
    def calm(cls, position_world):
        return cls(torch.zeros_like(position_world[..., 0]),
                   torch.zeros_like(position_world[..., :2]),
                   torch.zeros_like(position_world),
                   torch.zeros_like(position_world),
                   torch.zeros_like(position_world))


def footprint_transfer(kx, ky, yaw, length_m, beam_m):
    """Gaussian hull average, RMS extents L/sqrt(12), B/sqrt(12).

    H=exp(-[(k_long L)^2+(k_cross B)^2]/24). This normalized positive
    spatial kernel matches a rectangular footprint's second moments, while
    suppressing short waves smoothly. It is NOT an exact hull pressure RAO.
    For direction beta to bow, L_eff^2=L^2 cos^2(beta)+B^2 sin^2(beta).
    Amplitude -3 dB at k=sqrt(12 ln(2))/L_eff; f=sqrt(g k)/(2 pi).
    """
    yaw = torch.as_tensor(yaw, dtype=kx.dtype, device=kx.device).reshape(-1, 1)
    long = kx * torch.cos(yaw) + ky * torch.sin(yaw)
    cross = -kx * torch.sin(yaw) + ky * torch.cos(yaw)
    return torch.exp(-((long * length_m).square() + (cross * beam_m).square()) / 24.0)


def surface_normal(slope):
    """Unit upward normal of eta(x,y): normalize(-eta_x, -eta_y, 1)."""
    normal = torch.cat((-slope, torch.ones_like(slope[..., :1])), -1)
    return normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)


def surface_angular_velocity(slope, slope_rate):
    """Angular velocity of the unit normal n(t) of a surface with slope s(t).

    N=(-s_x,-s_y,1), n=N/|N|. For a unit vector, Omega=n x dn/dt is the
    angular velocity perpendicular to n; dn/dt=(I-n n^T) dN/dt/|N| and
    n x n = 0 give Omega = n x dN/dt / |N|. Small slope: (ds_y/dt, -ds_x/dt, 0).
    """
    raw = torch.cat((-slope, torch.ones_like(slope[..., :1])), -1)
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    rate = torch.cat((-slope_rate, torch.zeros_like(slope_rate[..., :1])), -1)
    return torch.cross(raw / norm, rate, dim=-1) / norm


@dataclass(frozen=True)
class SpectralWaves:
    """Kinematic input, one row per env and one column per spectral component.

    Deep-water linear waves eta=sum a*cos(k.x-omega*t+phase). A source such
    as C2 SeaState owns draws/clocks; this adapter owns no RNG or reset state.
    Use the source's CURRENT tensors after subset reset. No .forces() call.
    """
    amplitude: torch.Tensor
    omega: torch.Tensor
    k: torch.Tensor
    direction: torch.Tensor
    phase: torch.Tensor

    def __post_init__(self):
        if self.amplitude.ndim != 2 or min(self.amplitude.shape) < 1:
            raise ValueError("amplitude must be a nonempty [N,M] tensor")
        for name in ("amplitude", "omega", "k", "direction", "phase"):
            value = getattr(self, name)
            if value.shape not in (self.amplitude.shape, self.amplitude.shape[-1:]):
                raise ValueError(name + " must have shape [N,M] or [M]")
            if (value.device != self.amplitude.device or value.dtype != self.amplitude.dtype
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError("spectral tensors must be finite with the same floating dtype/device")
            if name in ("omega", "k") and (value < 0).any():
                raise ValueError("wave frequency and wavenumber must be nonnegative")

    @classmethod
    def from_sea_state(cls, sea):
        return cls(sea.amplitude, sea.omega, sea.k, sea.direction, sea.phase)

    def sample(self, world_xy, env_origin_xy, time_s, yaw, spec, footprint=True):
        """Return averaged elevation/slope and depth-attenuated 3-D water motion.

        Always subtract env origin BEFORE phase evaluation (C1 finding 9).
        Vertical orbital velocity is +a*omega*sin(theta), consistent with
        d(eta)/dt. Slope rates are the LOCAL time derivatives at the sampled
        point (d/dt of -a k sin(theta) = a k omega cos(theta)), consistent with
        the orbital velocity being evaluated at a fixed point. The filter is
        evaluated at the current heading and held fixed for this sample.
        E2: the orbital acceleration is the local time derivative of the
        same orbital velocity: d/dt of s*cos(theta) = s*omega*sin(theta)
        (horizontal, times the direction cosines) and d/dt of s*sin(theta) =
        -s*omega*cos(theta) (vertical), s = H_f a omega exp(-k d). The
        elevation, slope, velocity and surface-rate sums are unchanged.
        """
        if world_xy.shape != env_origin_xy.shape or world_xy.shape[-1] != 2:
            raise ValueError("world and env origins must have identical [N,2] shapes")
        n = world_xy.shape[0]
        if self.amplitude.ndim != 2 or self.amplitude.shape[0] != n:
            raise ValueError("wave amplitudes require one row per environment")
        if torch.as_tensor(yaw).numel() not in (1, n):
            raise ValueError("yaw requires a scalar or one heading per environment")
        time = torch.as_tensor(time_s, device=world_xy.device, dtype=world_xy.dtype)
        if time.numel() not in (1, n):
            raise ValueError("time requires a scalar or one clock per environment")
        time = time.reshape(-1, 1)
        local = world_xy - env_origin_xy
        kx, ky = self.k * torch.cos(self.direction), self.k * torch.sin(self.direction)
        theta = kx * local[:, 0:1] + ky * local[:, 1:2] - self.omega * time + self.phase
        transfer = footprint_transfer(kx, ky, yaw, spec.length_m, spec.beam_m) if footprint else 1.0
        amp = self.amplitude * transfer
        cosine, sine = torch.cos(theta), torch.sin(theta)
        elevation = (amp * cosine).sum(-1)
        slope = torch.stack(((-amp * kx * sine).sum(-1), (-amp * ky * sine).sum(-1)), -1)
        slope_rate = torch.stack(((amp * kx * self.omega * cosine).sum(-1),
                                  (amp * ky * self.omega * cosine).sum(-1)), -1)
        speed = amp * self.omega * torch.exp(-self.k * spec.orbital_depth_m)
        orbital = torch.stack(((speed * cosine * torch.cos(self.direction)).sum(-1),
                               (speed * cosine * torch.sin(self.direction)).sum(-1),
                               (speed * sine).sum(-1)), -1)
        rate = speed * self.omega
        acceleration = torch.stack(((rate * sine * torch.cos(self.direction)).sum(-1),
                                    (rate * sine * torch.sin(self.direction)).sum(-1),
                                    (-rate * cosine).sum(-1)), -1)
        return WaveSample(elevation, slope, orbital,
                          surface_angular_velocity(slope, slope_rate), acceleration)


@dataclass(frozen=True)
class HydroConstants:
    """Device tensors of one HydroSpec, built and validated ONCE (E2).

    Passing ``constants`` to hydro_wrench / damping_matrices /
    hydrostatic_wrench avoids re-creating these tensors from Python data on
    every physics step (a host-to-device copy on CUDA) and skips per-call
    value validation (host synchronisation, D2 REVIEW F13). The tensors are
    created with exactly the calls the validating path uses, so both paths
    give bit-identical results. ``spec`` is the identity key.
    """
    spec: object
    dtype: torch.dtype
    device: object
    mass_matrix: torch.Tensor
    inertia: torch.Tensor
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    angular_linear_damping: torch.Tensor
    angular_quadratic_damping: torch.Tensor
    roll_stiffness: torch.Tensor
    pitch_stiffness: torch.Tensor


def prepare_hydro_constants(spec, device, dtype) -> HydroConstants:
    """Cache the spec's tensors on ``device``; HydroSpec already validated them."""
    # Duck-typed (path-loaded tests may hold a second copy of this module).
    if not all(hasattr(spec, name) for name in ("mass_kg", "inertia_kgm2", "linear_damping",
                                                  "roll_stiffness_nm", "froude_krylov_horizontal")):
        raise ValueError("prepare_hydro_constants needs a HydroSpec")

    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device)
    mass_matrix = torch.eye(3, device=device, dtype=dtype) * spec.mass_kg
    return HydroConstants(
        spec=spec, dtype=dtype, device=mass_matrix.device,
        mass_matrix=mass_matrix,
        inertia=tensor(spec.inertia_kgm2),
        linear_damping=tensor(spec.linear_damping), quadratic_damping=tensor(spec.quadratic_damping),
        angular_linear_damping=tensor(spec.angular_linear_damping),
        angular_quadratic_damping=tensor(spec.angular_quadratic_damping),
        roll_stiffness=tensor(spec.roll_stiffness_nm), pitch_stiffness=tensor(spec.pitch_stiffness_nm))


def _check_constants(spec, constants, like):
    """Python-level identity/dtype checks only (no tensor reads)."""
    if constants.spec is not spec:
        raise ValueError("HydroConstants were prepared for a different HydroSpec")
    if constants.dtype != like.dtype or constants.device != like.device:
        raise ValueError("HydroConstants dtype/device differ from the state tensors")


def damping_matrices(spec, velocity_relative_body, angular_velocity_body, resistance_scale=1.0,
                     constants=None):
    """Freeze nonnegative quadratic coefficients at this water-relative state.

    C7 resistance multiplier covers surge/sway/yaw (tow resistance), preserving
    the C2 drag_scale scope; heave/roll/pitch retain their own damping.
    ``angular_velocity_body`` must be the RELATIVE angular velocity used by
    the selected rotational reference (see HydroSpec). With ``constants``
    (E2) the coefficients are the cached tensors and ``resistance_scale``
    must be a tensor validated by the caller (HydroV2Runtime.configure).
    """
    def tensor(value):
        return torch.as_tensor(value, dtype=velocity_relative_body.dtype, device=velocity_relative_body.device)
    scale = tensor(resistance_scale)
    if constants is None:
        if not torch.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("resistance scale must be finite and nonnegative")
        lin, quad = tensor(spec.linear_damping), tensor(spec.quadratic_damping)
        ang_lin, ang_quad = tensor(spec.angular_linear_damping), tensor(spec.angular_quadratic_damping)
    else:
        _check_constants(spec, constants, velocity_relative_body)
        lin, quad = constants.linear_damping, constants.quadratic_damping
        ang_lin, ang_quad = constants.angular_linear_damping, constants.angular_quadratic_damping
    scale = torch.broadcast_to(scale, velocity_relative_body.shape[:-1])
    one = torch.ones_like(scale)
    linear = (lin + quad * velocity_relative_body.abs())
    angular = (ang_lin + ang_quad * angular_velocity_body.abs())
    return (torch.diag_embed(linear * torch.stack((scale, scale, one), -1)),
            torch.diag_embed(angular * torch.stack((one, one, scale), -1)))


def hydrostatic_wrench(spec, rotation, com_position_world, waves, water_level_world=0.0,
                       constants=None):
    """Buoyancy at COM plus TOTAL metacentric moment: no extra r_CB cross F.

    Fz=rho*g*clip(m/rho+A*(water+eta+z_eq-z_COM),0,Vmax), so the local
    heave stiffness is rho*g*A for BOTH hull motion and wave elevation (no
    separate 100 N/m wave term). Gravity is engine-owned; it is NOT part of
    this returned applied wrench. Restoring aligns the hull to the filtered
    local normal n=normalize(-eta_x,-eta_y,1) (no separate 200 N m/slope
    term). On flat water it has identically zero world-vertical torque at
    every attitude; in waves the invariant axis is n (see restoring_v2.py).
    Returns (force_body, torque_body). With ``constants`` (E2) the restoring
    stiffnesses are the cached tensors and no value validation runs.
    """
    height = water_level_world + waves.elevation + spec.equilibrium_com_z_m - com_position_world[..., 2]
    volume = torch.clamp(spec.mass_kg / spec.water_density + spec.waterplane_m2 * height,
                         min=0.0, max=spec.maximum_volume_m3)
    force_world = torch.zeros_like(com_position_world)
    force_world[..., 2] = spec.water_density * spec.gravity * volume
    if constants is None:
        torque = restoring_moment(rotation, spec.roll_stiffness_nm, spec.pitch_stiffness_nm,
                                  surface_normal(waves.slope))
    else:
        _check_constants(spec, constants, com_position_world)
        torque = restoring_moment(rotation, constants.roll_stiffness, constants.pitch_stiffness,
                                  surface_normal(waves.slope), validate=False)
    return (rotation.transpose(-1, -2) @ force_world.unsqueeze(-1)).squeeze(-1), torque


@dataclass(frozen=True)
class HydroWrench:
    force_body: torch.Tensor
    torque_body: torch.Tensor
    damping_force_body: torch.Tensor
    damping_torque_body: torch.Tensor
    water_velocity_world: torch.Tensor
    relative_velocity_body: torch.Tensor
    relative_angular_velocity_body: torch.Tensor
    # E2 diagnostic: the Froude-Krylov part of force_body (hull axes), None
    # when the term is off or the sample carries no acceleration.
    froude_krylov_force_body: torch.Tensor | None = None


def froude_krylov_force_world(spec, waves):
    """World horizontal Froude-Krylov force m (1 + C_a) (a_x, a_y, 0), or None.

    None when spec.froude_krylov_horizontal is False or the sample carries no
    orbital acceleration (calm(), sample() always carry one).
    """
    acceleration = waves.orbital_acceleration_world
    if not spec.froude_krylov_horizontal or acceleration is None:
        return None
    force = torch.zeros_like(acceleration)
    force[..., :2] = (spec.mass_kg * (1.0 + spec.added_mass_coefficient_horizontal)) * acceleration[..., :2]
    return force


def hydro_wrench(spec, rotation, com_position_world, velocity_world, angular_velocity_world,
                 current_velocity_world, waves, dt, other_force_body=None,
                 other_torque_body=None, resistance_scale=1.0, water_level_world=0.0,
                 constants=None):
    """Applied hull wrench, with known thrust/wind in the implicit predictor.

    Inputs/outputs [N,3], rotation [N,3,3]. Current is world [N,3], including
    any declared vertical component (normally zero). Return excludes both
    gravity and the supplied other wrench: the caller adds each ONCE.
    Buoyancy/restoring are included. If the engine integrates body angular
    velocity with the Euler gyroscopic term, pass -omega x (I omega) inside
    other_torque_body for the PREDICTOR only (never apply it twice).

    E2: when spec.froude_krylov_horizontal is True the horizontal
    Froude-Krylov force (module docstring) is added to the translational
    predictor's known forcing AND to the returned force_body, exactly once;
    it is also reported as froude_krylov_force_body. ``constants`` (from
    prepare_hydro_constants) selects the cached, non-validating path used
    by HydroV2Runtime.step (bit-identical results).
    """
    if velocity_world.ndim != 2 or velocity_world.shape[-1] != 3:
        raise ValueError("velocity must have shape [N,3]")
    n = velocity_world.shape[0]
    if rotation.shape != (n, 3, 3) or any(value.shape != (n, 3) for value in
            (com_position_world, angular_velocity_world, waves.orbital_velocity_world)):
        raise ValueError("rotation and COM state must have matching environment rows")
    if waves.elevation.shape != (n,) or waves.slope.shape != (n, 2):
        raise ValueError("wave elevation/slope must have shapes [N] and [N,2]")
    surface_rate = waves.surface_angular_velocity_world
    if surface_rate is not None and surface_rate.shape != (n, 3):
        raise ValueError("surface angular velocity must have shape [N,3]")
    acceleration = waves.orbital_acceleration_world
    if acceleration is not None and acceleration.shape != (n, 3):
        raise ValueError("orbital acceleration must have shape [N,3]")
    if any(value is not None and value.shape != (n, 3) for value in (other_force_body, other_torque_body)):
        raise ValueError("known body force/torque must have shape [N,3]")
    if current_velocity_world.shape != velocity_world.shape:
        raise ValueError("current must be world [N,3], not an independent drag wrench")
    if constants is not None:
        _check_constants(spec, constants, velocity_world)
    inverse = rotation.transpose(-1, -2)

    def rotate(value):
        return (inverse @ value.unsqueeze(-1)).squeeze(-1)

    water = current_velocity_world + waves.orbital_velocity_world
    body_velocity, body_water = rotate(velocity_world), rotate(water)
    angular = rotate(angular_velocity_world)
    if spec.rotational_relative_motion and surface_rate is not None:
        body_water_rate = rotate(surface_rate)
    else:
        body_water_rate = None
    relative = body_velocity - body_water
    relative_angular = angular if body_water_rate is None else angular - body_water_rate
    drag, angular_drag = damping_matrices(spec, relative, relative_angular, resistance_scale, constants)
    buoyancy, restoring = hydrostatic_wrench(spec, rotation, com_position_world, waves, water_level_world,
                                             constants)
    other_force = torch.zeros_like(velocity_world) if other_force_body is None else other_force_body
    other_torque = torch.zeros_like(angular) if other_torque_body is None else other_torque_body
    gravity_world = torch.zeros_like(velocity_world)
    gravity_world[..., 2] = -spec.gravity
    if constants is None:
        mass = torch.eye(3, device=velocity_world.device, dtype=velocity_world.dtype) * spec.mass_kg
        inertia = torch.as_tensor(spec.inertia_kgm2, device=velocity_world.device, dtype=velocity_world.dtype)
    else:
        mass, inertia = constants.mass_matrix, constants.inertia
    known_force = other_force + buoyancy + spec.mass_kg * rotate(gravity_world)
    fk_world = froude_krylov_force_world(spec, waves)
    fk_body = None if fk_world is None else rotate(fk_world)
    if fk_body is not None:
        # An exactly zero FK component (calm water, Hs = 0) must leave every bit
        # of the pre-E2 arithmetic unchanged, signed zeros included (x + (+0)
        # would turn -0 into +0), so zero components are skipped elementwise
        # (tensor-valued select, no host synchronisation).
        fk_zero = fk_body == 0
        known_force = torch.where(fk_zero, known_force, known_force + fk_body)
    check = constants is None
    damping_force = implicit_damping_wrench(body_velocity, mass, drag, dt, known_force, body_water,
                                            check_errors=check)
    damping_torque = implicit_damping_wrench(angular, inertia, angular_drag, dt,
                                             other_torque + restoring, body_water_rate,
                                             check_errors=check)
    force = buoyancy + damping_force
    if fk_body is not None:
        force = torch.where(fk_zero, force, force + fk_body)
    return HydroWrench(force, restoring + damping_torque,
                       damping_force, damping_torque, water, relative, relative_angular, fk_body)


def required_high_frequency_hz(spec, peak_frequency_hz, squared_transfer_tolerance=1e-8):
    """Conservative Gaussian-filter band edge, not an arbitrary fp multiple.

    f_hi=max(4 fp, sqrt(g*sqrt(12 log(1/tol))/min(L,B))/(2 pi)).
    This captures >=99.51% unenhanced deep-water JONSWAP elevation variance
    above .2fp, and makes worst-heading Gaussian squared transfer <=tol at
    the high edge. Long-period waves can require far more than 3fp for slope
    convergence. Use df<=.01fp and >=256 bins, then verify response refinement.
    Disabled footprint filtering has no finite cutoff-independent point-slope
    limit; this helper applies only to the declared Gaussian footprint policy.
    """
    if not math.isfinite(peak_frequency_hz) or peak_frequency_hz <= 0:
        raise ValueError("peak frequency must be finite and positive")
    if not 0 < squared_transfer_tolerance < 1:
        raise ValueError("squared transfer tolerance must lie strictly in (0,1)")
    k_edge = math.sqrt(12*math.log(1/squared_transfer_tolerance))/min(spec.length_m, spec.beam_m)
    return max(4*peak_frequency_hz, math.sqrt(spec.gravity*k_edge)/(2*math.pi))
