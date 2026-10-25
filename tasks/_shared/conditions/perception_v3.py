"""Journal perception degradation (schema 3, protocol usvbench.perception.keyed.v1).

C7 STRATEGY.md section 5 primary axes, applied to the physical measurement
groups the observation layout consumes (observation_groups.py), exactly once
per control step per group, BEFORE normalisation/feature construction:

  position-estimate noise  e_n: stationary Gauss-Markov (first-order) per
      world axis, sigma = pos_noise_sigma_m, correlation time tau:
      e_0 = sigma n_0, e_k = a e_{k-1} + sigma sqrt(1 - a^2) n_k,
      a = exp(-dt_control / tau).
  position bias  e_b: fixed magnitude pos_bias_m, direction = task reference
      axis + the scenario's compass-8 stratum angle (balanced; no RNG).
  The estimated own position is truth + e_n + e_b, so the task-relative
  position vector ("pos" group) is measured as x - e_n - e_b.
  sensor bundle (labelled separately): white noise on body surge/sway
      (vel_noise_sigma_mps), yaw rate (yaw_rate_noise_sigma_rad_s, never the
      linear value) and every range return (ray_noise_sigma_m; output
      clamped to [0, max range]).
  latency  L = latency_s / dt_control frames (must be an integer): the
      output at frame k is the degraded frame k - L; before frame L it is the
      episode's first degraded frame (age reported).
  outage  one contiguous interval of D = outage_s / dt_control frames
      starting at frame s: the output holds the last pre-outage output.
      s comes from the task contract (``outage_start_s`` at reset, the
      "critical phase"; station keeping only), otherwise (G28, owner decision
      2026-10-04 "replace the 5-20 s window by a rule") from the REFERENCE
      ROUTE: s = f x T_ref with T_ref = nominal (witness) route length /
      reference (cruise) speed of the scenario (``reference_duration_s`` at
      reset; both are frozen with the layout, before any dose or controller)
      and f drawn per scenario key, uniform in OUTAGE_ROUTE_FRACTION =
      [0.40, 0.60] from the keyed ``perception.outage_start`` stream (draw 0,
      the same uniform the old window used): the middle of the reference
      transit, the same phase for both vessels and every route length, never
      dependent on the own ship's actual progress. Only a caller that supplies
      neither (legacy / direct use; journal tasks never do: the mixin refuses
      it) keeps the keyed uniform draw in ``outage_window_s`` (stamped
      "fallback_window_PENDING").

F0 own-state estimates (CONTEXT_R4 D5). ``estimate(name, x)`` degrades the
OWN state instead of a task-relative vector: for ``pos`` the input is the
true env-local COM xy and the output is the estimate truth + e_n + e_b
(the task-relative ``apply("pos", goal - com)`` of E3 returns x - e_n - e_b,
the same error with the task-relative sign); ``vel`` and ``ray`` are
identical to ``apply``. Both entry points share every internal (keyed
tapes, AR(1) state, ring buffer, outage hold), so a group must be driven by
ONE of them per episode. For a journal contract (``estimate_groups``) the
degrader builds all three estimate groups whenever a temporal axis is active
(one consistent delayed state for every controller); the noise axes stay
restricted to the consumed groups by admission (observation_groups.py).

Randomness: one independent keyed stream per primitive (keyed_rng.py),
names ``perception.pos_noise``, ``perception.vel_noise``,
``perception.ray_noise``, ``perception.outage_start``; keys are derived
once per (seed, env, episode) at reset and draws are materialised in chunks.
Switching bias (no RNG), noise or latency on or off never changes the outage
draw or any other primitive's draws (C1 finding 31, D2 items 11/F11).
Reward, termination and metrics keep reading ground truth.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import keyed_rng
from .strata import relative_angles

GROUP_DIMS = {"pos": 2, "vel": 3}
STREAMS = {"pos": "perception.pos_noise", "vel": "perception.vel_noise",
           "ray": "perception.ray_noise", "outage": "perception.outage_start"}
# G28 (owner decision 2026-10-04, 39_defaults list section 2 item 4): outage start rule for every task
# without a declared critical phase (station keeping keeps its key phase): s = f x T_ref, f keyed in
# OUTAGE_ROUTE_FRACTION (the middle fifth of the reference transit, the same [0.40, 0.60] band the owner
# fixed for the scatter appearing obstacle on 2026-10-01), T_ref = reference route length / reference speed.
OUTAGE_ROUTE_FRACTION = (0.40, 0.60)
OUTAGE_ROUTE_SOURCE = "reference_route_fraction"
OUTAGE_ROUTE_RULE = ("start = f x T_ref; T_ref = nominal reference-route length / reference (cruise) speed of the "
                     "scenario (frozen with the layout); f = 0.40 + 0.20 u, u = draw 0 of the keyed stream "
                     "perception.outage_start of (seed, env, episode); controller- and progress-independent")


def _frames(seconds: float, dt: float, name: str) -> int:
    value = float(seconds) / float(dt)
    frames = int(round(value))
    if abs(value - frames) > 1e-6:
        raise ValueError(f"perception {name}={seconds} s is not an integer multiple of the "
                         f"control period {dt} s")
    return frames


class _Group:
    def __init__(self, name, dim, num_envs, device, latency):
        self.name, self.dim = name, int(dim)
        self.frames = np.zeros(num_envs, dtype=np.int64)
        self.noise_state = torch.zeros((num_envs, self.dim), device=device)
        self.first = torch.zeros((num_envs, self.dim), device=device)
        self.hold = torch.zeros((num_envs, self.dim), device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ring = (torch.zeros((latency + 1, num_envs, self.dim), device=device)
                     if latency > 0 else None)
        self.tape = None


class PerceptionV3Degrader:
    def __init__(self, cfg, num_envs, device, seed, contract, control_dt, *,
                 ray_count=None, ray_max_range_m=None, chunk_steps=512):
        self.cfg, self.contract = cfg, contract
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.seed, self.dt = int(seed), float(control_dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("control period must be positive and finite")
        self.latency = _frames(cfg.latency_s, self.dt, "latency_s") if cfg.enabled else 0
        self.outage = _frames(cfg.outage_s, self.dt, "outage_s") if cfg.enabled else 0
        self.window = tuple(float(v) for v in cfg.outage_window_s)
        self.a = math.exp(-self.dt / cfg.pos_noise_correlation_s)
        self.b = math.sqrt(-math.expm1(-2.0 * self.dt / cfg.pos_noise_correlation_s))
        self.ray_max = None if ray_max_range_m is None else float(ray_max_range_m)
        self.ids = np.arange(self.num_envs, dtype=np.int64)
        on = bool(cfg.enabled)
        self.axes = {
            "pos_noise": on and cfg.pos_noise_sigma_m > 0,
            "pos_bias": on and cfg.pos_bias_m > 0,
            "vel_noise": on and (cfg.vel_noise_sigma_mps > 0 or cfg.yaw_rate_noise_sigma_rad_s > 0),
            "ray_noise": on and cfg.ray_noise_sigma_m > 0,
            "latency": on and self.latency > 0,
            "outage": on and self.outage > 0,
        }
        temporal = self.axes["latency"] or self.axes["outage"]
        self.groups = {}
        for name in (getattr(contract, "estimate_groups", ()) or contract.groups):
            specific = {"pos": self.axes["pos_noise"] or self.axes["pos_bias"],
                        "vel": self.axes["vel_noise"], "ray": self.axes["ray_noise"]}[name]
            if not (specific or temporal):
                continue
            if name == "ray":
                if ray_count is None or ray_max_range_m is None:
                    raise ValueError("ray perception needs ray_count and ray_max_range_m")
                dim = int(ray_count)
            else:
                dim = GROUP_DIMS[name]
            group = _Group(name, dim, self.num_envs, self.device, self.latency)
            noisy = {"pos": self.axes["pos_noise"], "vel": self.axes["vel_noise"],
                     "ray": self.axes["ray_noise"]}[name]
            if noisy:
                # Cap each tape at 4 Mi values (16 MB float32): wide groups
                # (36 rays) at 4096 envs get short chunks instead of 300 MB.
                chunk = max(16, min(int(chunk_steps), (1 << 22) // (self.num_envs * dim)))
                group.tape = keyed_rng.KeyedNormalTape(self.num_envs, dim, self.device,
                                                       chunk_steps=chunk)
            self.groups[name] = group
        self.sigma = {
            "vel": torch.tensor([cfg.vel_noise_sigma_mps, cfg.vel_noise_sigma_mps,
                                 cfg.yaw_rate_noise_sigma_rad_s], device=self.device),
        }
        self.bias = torch.zeros((self.num_envs, 2), device=self.device)
        self.bias_direction = torch.zeros(self.num_envs, dtype=torch.float64)
        self.outage_start = np.full(self.num_envs, -1, dtype=np.int64)
        self.outage_source = ["none"] * self.num_envs
        self.outage_fraction = np.full(self.num_envs, np.nan)          # G28: f of the route rule
        self.outage_reference_s = np.full(self.num_envs, np.nan)       # G28: T_ref of the route rule
        self.episode = np.full(self.num_envs, -1, dtype=np.int64)
        self.keys = {name: np.zeros(self.num_envs, dtype=np.uint64) for name in STREAMS}
        self.key_derivations = 0
        # G19 (training only; to be rebased on v2.5): per-env axis gates of a training mixture.
        # None = no mixture = unchanged behaviour.
        self._train = None

    # ------------------------------------------------ training mixture (G19) --
    TRAINING_AXES = ("noise", "bias", "latency", "dropout")

    def attach_training_gates(self) -> None:
        """Every axis starts OFF on every env; set_training_gates switches them per reset row."""
        if self.axes["vel_noise"] or self.axes["ray_noise"]:
            raise ValueError("the sensor bundle is not a training mixture component")
        self._train = {axis: np.zeros(self.num_envs, dtype=bool) for axis in self.TRAINING_AXES}

    def set_training_gates(self, env_ids, gates) -> None:
        if self._train is None:
            raise RuntimeError("attach_training_gates() first")
        ids = np.asarray(torch.as_tensor(env_ids).detach().cpu().reshape(-1), dtype=np.int64)
        for axis in self.TRAINING_AXES:
            values = np.asarray(gates[axis], dtype=bool).reshape(-1)
            if values.size != ids.size:
                raise ValueError("one gate value per reset env is required")
            self._train[axis][ids] = values

    def training_gates_on(self, env_id) -> list:
        if self._train is None:
            return []
        return [axis for axis in self.TRAINING_AXES if bool(self._train[axis][int(env_id)])]

    # ------------------------------------------------------------ queries --
    @property
    def active(self) -> bool:
        return bool(self.groups)

    def wants(self, name: str) -> bool:
        if name not in ("pos", "vel", "ray"):
            raise KeyError(name)
        return name in self.groups

    # -------------------------------------------------------------- reset --
    def reset(self, env_ids, episodes, strata, reference_axis_rad, outage_start_s=None,
              reference_duration_s=None) -> None:
        ids = np.asarray(torch.as_tensor(env_ids).detach().cpu().reshape(-1), dtype=np.int64)
        episodes = np.asarray(torch.as_tensor(episodes).detach().cpu().reshape(-1), dtype=np.int64)
        if ids.size != episodes.size:
            raise ValueError("one episode index per reset env is required")
        if ids.size == 0 or not self.active:
            self.episode[ids] = episodes
            return
        self.episode[ids] = episodes
        for group in self.groups.values():
            group.frames[ids] = 0
            if group.tape is not None:
                stream = STREAMS[group.name]
                keys = [keyed_rng.stream_key(self.seed, int(i), int(e), stream)
                        for i, e in zip(ids, episodes)]
                self.key_derivations += len(keys)
                self.keys[group.name][ids] = np.asarray(keys, dtype=np.uint64)
                group.tape.reset(ids, keys)
        ids_t = torch.as_tensor(ids, dtype=torch.long).to(self.device)
        if self.axes["pos_bias"]:
            axis = torch.as_tensor(reference_axis_rad, dtype=torch.float64).detach().cpu().reshape(-1)
            if axis.numel() != ids.size or not bool(torch.isfinite(axis).all()):
                raise ValueError("position bias needs one finite task reference axis per env")
            direction = axis + relative_angles(strata)
            self.bias_direction[torch.as_tensor(ids)] = direction
            vector = torch.stack((torch.cos(direction), torch.sin(direction)), -1) * self.cfg.pos_bias_m
            self.bias[ids_t] = vector.to(self.bias.dtype).to(self.device)
        if self.axes["outage"]:
            if outage_start_s is not None and reference_duration_s is not None:
                raise ValueError("pass either outage_start_s (task critical phase) or reference_duration_s "
                                 "(reference-route rule), not both")
            fraction = reference = None
            if outage_start_s is not None:
                start = torch.as_tensor(outage_start_s, dtype=torch.float64).detach().cpu().reshape(-1)
                if start.numel() != ids.size or not bool(torch.isfinite(start).all()) or bool((start < 0).any()):
                    raise ValueError("outage_start_s needs one finite nonnegative value per env")
                frames = np.floor(start.numpy() / self.dt + 1e-9).astype(np.int64)
                source = "task_critical_phase"
            elif reference_duration_s is not None:
                # G28: reference-route rule (module docstring); same key and draw index as the fallback
                reference = torch.as_tensor(reference_duration_s, dtype=torch.float64).detach().cpu().reshape(-1)
                if reference.numel() != ids.size or not bool(torch.isfinite(reference).all()) \
                        or bool((reference <= 0).any()):
                    raise ValueError("reference_duration_s needs one finite positive value per env")
                reference = reference.numpy()
                keys = [keyed_rng.stream_key(self.seed, int(i), int(e), STREAMS["outage"])
                        for i, e in zip(ids, episodes)]
                self.key_derivations += len(keys)
                self.keys["outage"][ids] = np.asarray(keys, dtype=np.uint64)
                u = keyed_rng.uniforms(np.asarray(keys, dtype=np.uint64), np.zeros(ids.size))
                lo, hi = OUTAGE_ROUTE_FRACTION
                fraction = lo + u * (hi - lo)
                frames = np.floor(fraction * reference / self.dt).astype(np.int64)
                source = OUTAGE_ROUTE_SOURCE
            else:
                keys = [keyed_rng.stream_key(self.seed, int(i), int(e), STREAMS["outage"])
                        for i, e in zip(ids, episodes)]
                self.key_derivations += len(keys)
                self.keys["outage"][ids] = np.asarray(keys, dtype=np.uint64)
                u = keyed_rng.uniforms(np.asarray(keys, dtype=np.uint64), np.zeros(ids.size))
                lo, hi = self.window
                frames = np.floor((lo + u * (hi - lo)) / self.dt).astype(np.int64)
                source = "fallback_window_PENDING"
            self.outage_start[ids] = frames
            self.outage_fraction[ids] = np.nan if fraction is None else fraction
            self.outage_reference_s[ids] = np.nan if reference is None else reference
            for i in ids:
                self.outage_source[int(i)] = source

    # -------------------------------------------------------------- apply --
    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Degrade one [N, dim] measurement; call exactly once per control step.

        ``pos`` is the task-relative vector (E3 semantics: x - e_n - e_b).
        """
        return self._degrade(name, x, own_state=False)

    def estimate(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """F0 / D5: degraded OWN-state estimate of one [N, dim] group.

        ``pos``: true env-local COM xy -> truth + e_n + e_b. ``vel`` / ``ray``:
        as ``apply``. Same internals as ``apply``; call once per control step.
        """
        return self._degrade(name, x, own_state=True)

    def _degrade(self, name: str, x: torch.Tensor, *, own_state: bool) -> torch.Tensor:
        if self._train is not None:
            return self._degrade_gated(name, x, own_state=own_state)
        if name not in ("pos", "vel", "ray"):
            raise KeyError(name)
        group = self.groups.get(name)
        if group is None:
            return x
        if x.shape != (self.num_envs, group.dim):
            raise ValueError(f"group {name!r} expects shape {(self.num_envs, group.dim)}, got {tuple(x.shape)}")
        frames = group.frames.copy()
        first = torch.as_tensor(frames == 0).to(self.device)
        corrupted = x
        if group.tape is not None:
            normals = group.tape.gather(self.ids, frames).to(x.dtype)
            if name == "pos":
                sigma = float(self.cfg.pos_noise_sigma_m)
                state = torch.where(first.unsqueeze(-1), sigma * normals,
                                    self.a * group.noise_state + (sigma * self.b) * normals)
                group.noise_state = state
                corrupted = corrupted + state if own_state else corrupted - state
            elif name == "vel":
                corrupted = corrupted + normals * self.sigma["vel"].to(x.dtype)
            else:
                corrupted = corrupted + normals * float(self.cfg.ray_noise_sigma_m)
        if name == "pos" and self.axes["pos_bias"]:
            bias = self.bias.to(x.dtype)
            corrupted = corrupted + bias if own_state else corrupted - bias
        group.first = torch.where(first.unsqueeze(-1), corrupted, group.first.to(x.dtype))
        age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.latency > 0:
            capacity = self.latency + 1
            write = torch.as_tensor(frames % capacity).to(self.device)
            read = torch.as_tensor((frames - self.latency) % capacity).to(self.device)
            rows = torch.as_tensor(self.ids).to(self.device)
            group.ring = group.ring.to(x.dtype)
            group.ring[write, rows] = corrupted
            warmed = torch.as_tensor(frames >= self.latency).to(self.device)
            delayed = torch.where(warmed.unsqueeze(-1), group.ring[read, rows], group.first)
            age = torch.where(warmed, torch.full_like(age, self.latency),
                              torch.as_tensor(frames).to(self.device))
        else:
            delayed = corrupted
        if self.outage > 0:
            inside = (frames >= self.outage_start) & (frames < self.outage_start + self.outage)
            inside_t = torch.as_tensor(inside).to(self.device)
            # First-frame rule: an outage that starts at frame 0 holds the
            # episode's first (degraded, delayed) estimate, never stale data.
            hold = torch.where(first.unsqueeze(-1), delayed, group.hold.to(x.dtype))
            output = torch.where(inside_t.unsqueeze(-1), hold, delayed)
            held = torch.as_tensor(np.where(
                inside, frames - np.maximum(self.outage_start - 1, 0), 0)).to(self.device)
            age = age + held
            group.hold = output
        else:
            output = delayed
        if name == "ray":
            output = output.clamp(min=0.0, max=self.ray_max)
        group.age = age
        group.frames = frames + 1
        return output

    def _degrade_gated(self, name: str, x: torch.Tensor, *, own_state: bool) -> torch.Tensor:
        """G19: ``_degrade`` with each axis applied only on the rows whose training gate is on.

        Same keyed tapes, AR(1) state, ring buffer and outage hold as ``_degrade``; a row with no
        perception gate on returns its input row unchanged (exactly the disabled suite).
        """
        if name not in ("pos", "vel", "ray"):
            raise KeyError(name)
        group = self.groups.get(name)
        if group is None:
            return x
        if x.shape != (self.num_envs, group.dim):
            raise ValueError(f"group {name!r} expects shape {(self.num_envs, group.dim)}, got {tuple(x.shape)}")
        gates = self._train
        g_noise = torch.as_tensor(gates["noise"]).to(self.device)
        g_bias = torch.as_tensor(gates["bias"]).to(self.device)
        any_on = gates["noise"] | gates["bias"] | gates["latency"] | gates["dropout"]
        frames = group.frames.copy()
        first = torch.as_tensor(frames == 0).to(self.device)
        corrupted = x
        if group.tape is not None:
            normals = group.tape.gather(self.ids, frames).to(x.dtype)
            if name == "pos":
                sigma = float(self.cfg.pos_noise_sigma_m)
                state = torch.where(first.unsqueeze(-1), sigma * normals,
                                    self.a * group.noise_state + (sigma * self.b) * normals)
                group.noise_state = state
                state = torch.where(g_noise.unsqueeze(-1), state, torch.zeros_like(state))
                corrupted = corrupted + state if own_state else corrupted - state
        if name == "pos" and self.axes["pos_bias"]:
            bias = torch.where(g_bias.unsqueeze(-1), self.bias.to(x.dtype), torch.zeros_like(self.bias, dtype=x.dtype))
            corrupted = corrupted + bias if own_state else corrupted - bias
        group.first = torch.where(first.unsqueeze(-1), corrupted, group.first.to(x.dtype))
        age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.latency > 0:
            capacity = self.latency + 1
            latency = np.where(gates["latency"], self.latency, 0).astype(np.int64)
            write = torch.as_tensor(frames % capacity).to(self.device)
            read = torch.as_tensor((frames - latency) % capacity).to(self.device)
            rows = torch.as_tensor(self.ids).to(self.device)
            group.ring = group.ring.to(x.dtype)
            group.ring[write, rows] = corrupted
            warmed_np = frames >= latency
            warmed = torch.as_tensor(warmed_np).to(self.device)
            delayed = torch.where(warmed.unsqueeze(-1), group.ring[read, rows], group.first)
            age = torch.as_tensor(np.where(warmed_np, latency, frames)).to(self.device)
        else:
            delayed = corrupted
        if self.outage > 0:
            inside = (frames >= self.outage_start) & (frames < self.outage_start + self.outage) & gates["dropout"]
            inside_t = torch.as_tensor(inside).to(self.device)
            hold = torch.where(first.unsqueeze(-1), delayed, group.hold.to(x.dtype))
            output = torch.where(inside_t.unsqueeze(-1), hold, delayed)
            held = torch.as_tensor(np.where(
                inside, frames - np.maximum(self.outage_start - 1, 0), 0)).to(self.device)
            age = age + held
            group.hold = output
        else:
            output = delayed
        if name == "ray":
            output = output.clamp(min=0.0, max=self.ray_max)
        on = torch.as_tensor(any_on).to(self.device)
        output = torch.where(on.unsqueeze(-1), output, x)
        group.age = torch.where(on, age, torch.zeros_like(age))
        group.frames = frames + 1
        return output

    def measurement_age_s(self, name: str) -> torch.Tensor:
        """Age (s) of the last output of ``name``: latency plus outage hold."""
        group = self.groups.get(name)
        if group is None:
            return torch.zeros(self.num_envs, device=self.device)
        return group.age.to(torch.float32) * self.dt

    def episode_parameters(self, env_id: int) -> dict:
        i = int(env_id)
        result = {"protocol": self.cfg.protocol, "active": self.active,
                  "episode_index": int(self.episode[i]), "control_dt_s": self.dt,
                  "latency_frames": self.latency, "outage_frames": self.outage}
        if not self.active:
            return result
        result["groups"] = {}
        for name, group in self.groups.items():
            entry = {"frames": int(group.frames[i]), "dim": group.dim}
            if group.tape is not None:
                entry["stream"] = STREAMS[name]
                entry["stream_key"] = format(int(self.keys[name][i]), "016x")
            result["groups"][name] = entry
        if self.axes["pos_bias"]:
            result["bias_vector_world_m"] = [float(v) for v in self.bias[i].detach().cpu()]
            result["bias_direction_world_rad"] = float(self.bias_direction[i])
        if self.axes["outage"]:
            result["outage_start_frame"] = int(self.outage_start[i])
            result["outage_end_frame"] = int(self.outage_start[i] + self.outage)
            result["outage_start_s"] = float(self.outage_start[i] * self.dt)
            result["outage_source"] = self.outage_source[i]
            if self.outage_source[i].startswith("fallback"):
                result["outage_stream_key"] = format(int(self.keys["outage"][i]), "016x")
            elif self.outage_source[i] == OUTAGE_ROUTE_SOURCE:          # G28 reference-route rule
                result["outage_stream_key"] = format(int(self.keys["outage"][i]), "016x")
                result["outage_start_fraction"] = float(self.outage_fraction[i])
                result["outage_reference_duration_s"] = float(self.outage_reference_s[i])
                result["outage_fraction_window"] = list(OUTAGE_ROUTE_FRACTION)
        return result


__all__ = ["PerceptionV3Degrader", "STREAMS", "OUTAGE_ROUTE_FRACTION", "OUTAGE_ROUTE_SOURCE", "OUTAGE_ROUTE_RULE"]
