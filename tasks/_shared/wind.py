"""Task-independent wind loads and keyed, physics-step OU gusts.

SI units. Directions describe where air FLOWS TO, counterclockwise from world
+X. The horizontal vessel frame is FLU: X forward, Y port/left, Z up. For the
requested coefficient convention, gamma = atan2(v_rel_port, -v_rel_forward):
zero is wind FROM ahead, +pi/2 is wind FROM starboard. Thus CX=-CX0*cos(gamma)
and CY=CY0*sin(gamma) both push downwind. Do not use atan2(vy, vx) here.

forces() returns (world_force_xyz, world_torque_xyz), each (num_envs, 3), like
SeaState.forces. It uses a yaw-plane approximation, ignoring attitude changes
to exposed area and vertical airflow. Body surge/sway are yaw-rotated to the
world; the roll axis is the horizontal bow axis. The env must perform its
existing full-quaternion world-to-body conversion only once. In this z-up
frame the right-hand roll torque is K=-Y*z_bar (r cross F); the signed
heel-load convention H=Y*z_bar has the opposite sign. Pitch loading is omitted.
z_bar is an effective arm above the waterline; the model assumes a waterline
roll reference, not an independently measured COM-to-pressure-centre arm.

An explicit reset assigns each stream (eval_seed, logical_env_key, episode_id).
Gusts start in their stationary distribution. step(dt) advances exact OU once
per PHYSICS step; forces() and velocity_world() never advance it. Each Gaussian
innovation is a versioned BLAKE2b counter draw, so reset order, batch order,
other episodes finishing, and unrelated RNG use cannot change a stream.
No Python/NumPy/Torch global RNG is accessed. The normal backend runs on CPU
and transfers one vector to the requested device; GPU throughput is UNVERIFIED.
No Isaac dependency. Disabling the module returns exact zeros and draws no RNG.

See WIND_DESIGN.md for hook locations, reference-frame and dose assumptions.

RNG PROTOCOL v2 (E3, opt-in; ``rng_protocol="usvbench.wind.ou.v2"``)
--------------------------------------------------------------------
The default protocol ``usvbench.wind.ou.v1`` above is unchanged (one BLAKE2b
digest per env per physics step). Protocol v2 drives the SAME exact OU
recursion from a counter-based keyed stream (conditions/keyed_rng.py,
``usvbench.keyed_counter.v1``): the stream key is derived once per (eval
seed, env key, episode) at reset with the shared scenario seed-folding
protocol (group ``cond3.wind.gust``), and innovations are generated in
vectorised chunks of ``chunk_steps`` physics steps per env (D2 review item
4 / F4: no per-step, per-env Python hashing; counters are host NumPy, the
tape is a device tensor). Innovation for physics step s is normal index s;
the stationary initial draw uses s = 0, as in v1. Values are independent of
reset order, batching and chunk size. v1 and v2 streams differ numerically
(different keyed generator); the protocol name is part of every stamp.
``check_finite=False`` skips the two per-call finiteness reductions in
``forces`` (host synchronisations); shape/device/dtype checks remain.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import operator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np
import torch

RNG_PROTOCOL_V1 = "usvbench.wind.ou.v1"
RNG_PROTOCOL_V2 = "usvbench.wind.ou.v2"
RNG_PROTOCOLS = (RNG_PROTOCOL_V1, RNG_PROTOCOL_V2)
GUST_STREAM = "wind.gust"


def _keyed_rng():
    """conditions/keyed_rng.py, imported lazily (that package imports this module)."""
    try:
        from .conditions import keyed_rng as module
        return module
    except ImportError:
        name = "_usvbench_wind_keyed_rng"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, Path(__file__).resolve().parent / "conditions" / "keyed_rng.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        return sys.modules[name]


@dataclass
class VesselWindage:
    """Geometry in metres and coefficients (engineering assumptions unless fit).

    longitudinal_cp_m is a geometric side-silhouette centroid relative to the
    vessel reference. It is metadata in the default model: CN already supplies
    the yaw moment. Adding cp*Y as well would count two separate yaw models.
    A custom coefficient callable can replace CN with a calibrated CP model.
    """

    frontal_area_m2: float
    lateral_area_m2: float
    length_m: float
    centroid_height_m: float
    longitudinal_cp_m: float = 0.0
    c_x0: float = 0.9
    c_y0: float = 0.9
    c_n0: float = 0.1


@dataclass
class WindCfg:
    """Uniform mean wind at vessel height; no implicit 10 m extrapolation.

    gust_intensity = stationary along-wind sigma / mean_speed. The signed
    along-wind OU is NOT clipped (clipping would change its mean/variance).
    Extremely rare flow reversal is therefore possible. Direction drift is
    deterministic radians per second of this environment's episode time.
    """

    enable: bool = False
    mean_speed_mps: float = 0.0
    reference_height_m: float = 0.25
    direction_rad: float = 0.0
    direction_drift_rad_s: float = 0.0
    gust_intensity: float = 0.0
    gust_correlation_time_s: float = 2.0
    air_density_kg_m3: float = 1.225
    windage: VesselWindage | None = None


@dataclass(frozen=True)
class FossenCoefficients:
    """Replaceable first-harmonic skeleton, not calibrated vessel aerodynamics."""

    c_x0: float = 0.9
    c_y0: float = 0.9
    c_n0: float = 0.1

    def __call__(self, gamma: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            -self.c_x0 * torch.cos(gamma),
            self.c_y0 * torch.sin(gamma),
            self.c_n0 * torch.sin(2.0 * gamma),
        )


CoefficientModel = Callable[[torch.Tensor], tuple[torch.Tensor, ...]]


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _normal_for_key(seed: int, env: int, episode: int, step: int) -> float:
    """One stateless standard normal; stable serialization, never Python hash()."""
    key = f"usvbench.wind.ou.v1|{seed}|{env}|{episode}|{step}".encode("ascii")
    digest = hashlib.blake2b(key, digest_size=16).digest()
    # 52 random bits + half a bin keeps both uniforms strictly inside (0, 1).
    u1 = ((int.from_bytes(digest[:8], "little") >> 12) + 0.5) / (1 << 52)
    u2 = ((int.from_bytes(digest[8:], "little") >> 12) + 0.5) / (1 << 52)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _validate_cfg(cfg: WindCfg) -> None:
    for name in (
        "mean_speed_mps", "reference_height_m", "direction_rad",
        "direction_drift_rad_s", "gust_intensity", "gust_correlation_time_s",
        "air_density_kg_m3",
    ):
        if not math.isfinite(float(getattr(cfg, name))):
            raise ValueError(f"{name} must be finite")
    if cfg.mean_speed_mps < 0 or cfg.gust_intensity < 0:
        raise ValueError("mean speed and gust intensity must be nonnegative")
    if min(cfg.reference_height_m, cfg.gust_correlation_time_s,
           cfg.air_density_kg_m3) <= 0:
        raise ValueError("reference height, correlation time and density must be positive")
    if cfg.enable and cfg.windage is None:
        raise ValueError("enabled wind requires explicit vessel windage")
    if cfg.windage is not None:
        for name, value in vars(cfg.windage).items():
            if not math.isfinite(float(value)):
                raise ValueError(f"windage.{name} must be finite")
        w = cfg.windage
        if min(w.frontal_area_m2, w.lateral_area_m2, w.centroid_height_m,
               w.c_x0, w.c_y0) < 0 or w.length_m <= 0:
            raise ValueError("areas, height, CX0 and CY0 must be nonnegative; length positive")


class Wind:
    """Shared wind layer with explicit per-episode stream identities.

    env_keys defaults to local indices. Supply globally stable logical IDs when
    sharding evaluation across processes. Episode IDs belong to each logical
    env, not a first-finished pooled episode counter. Reusing a key replays it.
    Restart replay also requires identical dt, configuration, dtype and inputs.
    Bitwise equality across different Torch devices/versions is not promised.
    """

    def __init__(
        self, cfg: WindCfg, num_envs: int, device: torch.device | str = "cpu",
        *, eval_seed: int = 0, env_keys: Sequence[int] | None = None,
        dtype: torch.dtype = torch.float32,
        coefficient_model: CoefficientModel | None = None,
        rng_protocol: str = RNG_PROTOCOL_V1, chunk_steps: int = 1024,
        check_finite: bool = True,
    ):
        # Mutable input dataclasses support the repo's dotted CLI overrides.
        # Snapshot after resolution so later edits to task cfg cannot silently
        # change an existing episode or its stationary OU variance.
        cfg = deepcopy(cfg)
        _validate_cfg(cfg)
        self.cfg = cfg
        self.num_envs = _integer(num_envs, "num_envs")
        if self.num_envs == 0:
            raise ValueError("num_envs must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.device, self.dtype = torch.device(device), dtype
        self.eval_seed = _integer(eval_seed, "eval_seed")
        raw_keys = range(num_envs) if env_keys is None else env_keys
        self.env_keys = tuple(_integer(k, "env key") for k in raw_keys)
        if len(self.env_keys) != num_envs or len(set(self.env_keys)) != num_envs:
            raise ValueError("env_keys must contain one unique key per environment")
        w = cfg.windage
        self.coefficient_model = coefficient_model or FossenCoefficients(
            w.c_x0 if w else 0.9, w.c_y0 if w else 0.9, w.c_n0 if w else 0.1
        )
        self.gust_mps = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self.time_s = torch.zeros(num_envs, device=self.device, dtype=torch.float64)
        # D1 (J-base) extension: per-environment mean flow-to direction, set at
        # reset. Without an explicit per-episode direction every env uses
        # cfg.direction_rad, which reproduces the C3 behaviour exactly.
        self.direction_rad = torch.full((num_envs,), float(cfg.direction_rad),
                                        device=self.device, dtype=torch.float64)
        self._episode_ids: list[int | None] = [None] * num_envs
        self._physics_steps = [0] * num_envs
        self.normal_draw_count = 0
        # E3: optional counter-based batched protocol (module docstring).
        if rng_protocol not in RNG_PROTOCOLS:
            raise ValueError(f"rng_protocol must be one of {RNG_PROTOCOLS}")
        if not isinstance(check_finite, bool):
            raise ValueError("check_finite must be a bool")
        self.rng_protocol = rng_protocol
        self.check_finite = check_finite
        self._v2 = rng_protocol == RNG_PROTOCOL_V2
        self._tape = None
        self.key_derivations = 0
        if self._v2:
            self._keyed = _keyed_rng()
            self._all_ids = np.arange(self.num_envs, dtype=np.int64)
            self._all_ids_t = torch.arange(self.num_envs, device=self.device)
            self._steps_v2 = np.zeros(self.num_envs, dtype=np.int64)
            self._episodes_v2 = np.full(self.num_envs, -1, dtype=np.int64)
            self._reset_v2 = np.zeros(self.num_envs, dtype=bool)
            if cfg.enable and cfg.mean_speed_mps * cfg.gust_intensity > 0:
                self._tape = self._keyed.KeyedNormalTape(
                    self.num_envs, 1, self.device, dtype, chunk_steps=chunk_steps)

    def physics_steps(self, env_id: int) -> int:
        """Physics steps of env ``env_id`` since its last reset (either protocol)."""
        i = int(env_id)
        return int(self._steps_v2[i]) if self._v2 else int(self._physics_steps[i])

    def episode_id(self, env_id: int) -> int | None:
        i = int(env_id)
        if self._v2:
            value = int(self._episodes_v2[i])
            return None if value < 0 else value
        return self._episode_ids[i]

    def _v2_ids(self, env_ids) -> np.ndarray:
        if env_ids is None:
            return self._all_ids
        return np.asarray(self._ids(env_ids), dtype=np.int64)

    def _reset_v2_streams(self, ids, episodes, directions) -> None:
        ids_np = np.asarray(ids, dtype=np.int64)
        self._episodes_v2[ids_np] = np.asarray(episodes, dtype=np.int64)
        self._steps_v2[ids_np] = 0
        self._reset_v2[ids_np] = True
        self.time_s[ids] = 0.0
        if ids:
            self.direction_rad[ids] = (
                torch.tensor(directions, device=self.device, dtype=torch.float64)
                if directions is not None else float(self.cfg.direction_rad)
            )
        sigma = self.cfg.mean_speed_mps * self.cfg.gust_intensity
        if sigma and ids:
            keys = [self._keyed.stream_key(self.eval_seed, self.env_keys[i], episode, GUST_STREAM)
                    for i, episode in zip(ids, episodes)]
            self.key_derivations += len(keys)
            self._tape.reset(ids_np, keys)
            draw = self._tape.gather(ids_np, np.zeros(len(ids), dtype=np.int64))[:, 0]
            self.normal_draw_count += len(ids)
            self.gust_mps[ids] = sigma * draw
        else:
            self.gust_mps[ids] = 0.0

    def _step_v2(self, dt: float, env_ids) -> None:
        ids_np = self._v2_ids(env_ids)
        if not self.cfg.enable or ids_np.size == 0:
            return
        if not bool(self._reset_v2[ids_np].all()):
            raise RuntimeError("reset(env_ids, episode_ids) is required before enabled use")
        self._steps_v2[ids_np] += 1
        ids_t = self._all_ids_t if env_ids is None else torch.as_tensor(
            ids_np, dtype=torch.long).to(self.device)
        self.time_s[ids_t] += dt
        sigma = self.cfg.mean_speed_mps * self.cfg.gust_intensity
        if sigma:
            ratio = float(dt) / self.cfg.gust_correlation_time_s
            a = math.exp(-ratio)
            b = sigma * math.sqrt(-math.expm1(-2.0 * ratio))
            innovation = self._tape.gather(ids_np, self._steps_v2[ids_np])[:, 0]
            self.normal_draw_count += int(ids_np.size)
            self.gust_mps[ids_t] = a * self.gust_mps[ids_t] + b * innovation

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> list[int]:
        raw = range(self.num_envs) if env_ids is None else env_ids
        if isinstance(raw, torch.Tensor):
            if raw.ndim != 1:
                raise ValueError("env_ids must be one-dimensional")
            raw = raw.detach().cpu().tolist()
        ids = [_integer(i, "env id") for i in raw]
        if len(set(ids)) != len(ids) or any(i >= self.num_envs for i in ids):
            raise ValueError("env_ids must be unique valid local indices")
        return ids

    def _require_reset(self, ids: Sequence[int]) -> None:
        if any(self._episode_ids[i] is None for i in ids):
            raise RuntimeError("reset(env_ids, episode_ids) is required before enabled use")

    def _normals(self, ids: Sequence[int]) -> torch.Tensor:
        values = [
            _normal_for_key(self.eval_seed, self.env_keys[i],
                            self._episode_ids[i], self._physics_steps[i])
            for i in ids
        ]
        self.normal_draw_count += len(values)
        return torch.tensor(values, device=self.device, dtype=self.dtype)

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | None,
        episode_ids: Sequence[int] | torch.Tensor,
        direction_rad: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """Reset selected streams at t=0, drawing stationary OU only if active.

        ``direction_rad`` (D1 extension) optionally sets each reset env's
        mean flow-to azimuth for the new episode (world frame, CCW from +X);
        None restores cfg.direction_rad. It consumes no random draws.
        """
        ids = self._ids(env_ids)
        raw_episodes = episode_ids
        if isinstance(raw_episodes, torch.Tensor):
            if raw_episodes.ndim != 1:
                raise ValueError("episode_ids must be one-dimensional")
            raw_episodes = raw_episodes.detach().cpu().tolist()
        episodes = [_integer(e, "episode id") for e in raw_episodes]
        if len(episodes) != len(ids):
            raise ValueError("episode_ids must align with env_ids")
        directions = None
        if direction_rad is not None:
            raw_directions = direction_rad
            if isinstance(raw_directions, torch.Tensor):
                if raw_directions.ndim != 1:
                    raise ValueError("direction_rad must be one-dimensional")
                raw_directions = raw_directions.detach().cpu().tolist()
            directions = [float(value) for value in raw_directions]
            if len(directions) != len(ids) or not all(math.isfinite(d) for d in directions):
                raise ValueError("direction_rad must be finite and align with env_ids")
        if not self.cfg.enable:
            return
        if self._v2:
            self._reset_v2_streams(ids, episodes, directions)
            return
        for i, episode in zip(ids, episodes):
            self._episode_ids[i], self._physics_steps[i] = episode, 0
        self.time_s[ids] = 0.0
        if ids:
            self.direction_rad[ids] = (
                torch.tensor(directions, device=self.device, dtype=torch.float64)
                if directions is not None else float(self.cfg.direction_rad)
            )
        sigma = self.cfg.mean_speed_mps * self.cfg.gust_intensity
        self.gust_mps[ids] = sigma * self._normals(ids) if sigma and ids else 0.0

    def step(self, dt: float, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Exact OU transition; call ONCE each physics step, not control step.

        a=exp(-dt/tau); g_next=a*g + sigma*sqrt(1-a*a)*N(0,1).
        Supports subsets for paused streams; unrelated streams remain untouched.
        """
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("physics dt must be finite and positive")
        if self._v2:
            self._step_v2(dt, env_ids)
            return
        ids = self._ids(env_ids)
        if not self.cfg.enable or not ids:
            return
        self._require_reset(ids)
        for i in ids:
            self._physics_steps[i] += 1
        self.time_s[ids] += dt
        sigma = self.cfg.mean_speed_mps * self.cfg.gust_intensity
        if sigma:
            ratio = float(dt) / self.cfg.gust_correlation_time_s
            a = math.exp(-ratio)
            b = sigma * math.sqrt(-math.expm1(-2.0 * ratio))
            self.gust_mps[ids] = a * self.gust_mps[ids] + b * self._normals(ids)

    def velocity_world(self) -> torch.Tensor:
        """Current air flow-to velocity (N, 2); read-only, zero when disabled."""
        if not self.cfg.enable:
            return torch.zeros((self.num_envs, 2), device=self.device, dtype=self.dtype)
        if self._v2:
            if not bool(self._reset_v2.all()):
                raise RuntimeError("reset(env_ids, episode_ids) is required before enabled use")
        else:
            self._require_reset(range(self.num_envs))
        angle = (self.direction_rad +
                 self.cfg.direction_drift_rad_s * self.time_s).to(self.dtype)
        speed = self.cfg.mean_speed_mps + self.gust_mps
        return speed[:, None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)

    def forces(
        self, hull_velocity_xy: torch.Tensor, heading_world_rad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame wrench, using true air velocity minus vessel velocity.

        heading_world_rad is the PHYSICAL bow heading, not necessarily the USD
        asset's +X axis. Inputs are (N,2) and (N,), on this module's device and
        dtype. No gust or clock updates happen here. Forces act at the hook's
        reference point with explicit roll/yaw torque; do not add another CP
        application-point lever arm. At enable=True, U=0, a moving boat still
        experiences aerodynamic drag. Only enable=False is a force-free control.
        """
        force = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.dtype)
        torque = torch.zeros_like(force)
        if not self.cfg.enable:
            return force, torque
        for value, shape, name in (
            (hull_velocity_xy, (self.num_envs, 2), "hull_velocity_xy"),
            (heading_world_rad, (self.num_envs,), "heading_world_rad"),
        ):
            if (value.shape != shape or value.device != self.gust_mps.device or
                    value.dtype != self.dtype):
                raise ValueError(f"{name} must have shape {shape} and module device/dtype")
            if self.check_finite and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        relative = self.velocity_world() - hull_velocity_xy
        c, s = torch.cos(heading_world_rad), torch.sin(heading_world_rad)
        forward = c * relative[:, 0] + s * relative[:, 1]
        port = -s * relative[:, 0] + c * relative[:, 1]
        gamma = torch.atan2(port, -forward)
        cx, cy, cn = self.coefficient_model(gamma)
        dynamic_pressure = 0.5 * self.cfg.air_density_kg_m3 * relative.square().sum(-1)
        w = self.cfg.windage
        surge = dynamic_pressure * w.frontal_area_m2 * cx
        sway = dynamic_pressure * w.lateral_area_m2 * cy
        yaw = dynamic_pressure * w.lateral_area_m2 * w.length_m * cn
        roll = -sway * w.centroid_height_m
        force[:, 0], force[:, 1] = c * surge - s * sway, s * surge + c * sway
        torque[:, 0], torque[:, 1], torque[:, 2] = c * roll, s * roll, yaw
        return force, torque


__all__ = ["WindCfg", "VesselWindage", "FossenCoefficients", "CoefficientModel", "Wind",
           "RNG_PROTOCOL_V1", "RNG_PROTOCOL_V2", "RNG_PROTOCOLS", "GUST_STREAM"]
