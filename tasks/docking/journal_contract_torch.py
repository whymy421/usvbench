"""Tensor twin of E5's scalar DockingEpisode oracle (journal_contracts_v2.py), one row per env.

The env runs THIS batch oracle; E5's scalar ``DockingEpisode`` stays the
reference (tasks/docking/test_journal_docking_contract.py replays recorded and
randomized traces through both and requires bit-identical codes, latches,
times, path lengths and terminal extras).

Semantics mirrored line by line from journal_contracts_v2.py (E5):
  EpisodeBase.reset / observe_physics / _observe / _finish (:381-440),
  ValidityLatch.update + validity_reason (:165-181, :224-240),
  termination_decision (:247-270; precedence validity > out_of_bounds >
  collision > success > timeout), EpisodeMetrics.observe / latch_success
  (:327-346), DockingEpisode.reset / _conjunction / observe_physics /
  _entry_update / _docking_candidate / step (:520-603), Berth.local /
  contains_polygon (:472-517).

Bit parity: every float operation is issued in the scalar's order as a
separate IEEE double operation (no fused ops); minima use where(b < a, b, a)
like Python's min(a, b). Transcendentals and hypot are evaluated with the
SAME functions the scalar calls (Python ``math.cos/sin/atan2/acos/sqrt/
hypot``, applied elementwise on the CPU), because torch's vectorised kernels
differ from libm by up to 1 ulp (measured: FINAL_REPORT.md). Everything else
(latches, masks, arithmetic, selection) is vectorised torch float64.
"""
from __future__ import annotations

import math

import numpy as np
import torch

RUNNING, SUCCESS, COLLISION, TIMEOUT, OUT_OF_BOUNDS = 0, 1, 2, 3, 4
CAPSIZE, SINKING, AIRBORNE, NONFINITE = 6, 7, 8, 9
F64 = torch.float64
_UFUNCS = {}


def _libm(fn, *tensors):
    """Elementwise Python ``math`` function on float64 tensors (exactly the scalar's calls)."""
    if not tensors[0].numel():
        return torch.zeros_like(tensors[0], dtype=F64)
    key = (fn, len(tensors))
    if key not in _UFUNCS:
        _UFUNCS[key] = np.frompyfunc(fn, len(tensors), 1)
    arrays = [t.detach().to("cpu", F64).numpy() for t in tensors]
    out = np.asarray(_UFUNCS[key](*arrays), dtype=np.float64)
    return torch.from_numpy(out).to(tensors[0].device)


WRAPPED = "wrapped"          # _libm_batch request: atan2(sin(a), cos(a)), i.e. wrapped(a)


def _host_eval(fn, arrays):
    """The host half of _libm: the cached np.frompyfunc of the Python math function, float64 out."""
    key = (fn, len(arrays))
    if key not in _UFUNCS:
        _UFUNCS[key] = np.frompyfunc(fn, len(arrays), 1)
    return np.asarray(_UFUNCS[key](*arrays), dtype=np.float64)


def _host_copies(tensors):
    """[t.detach().to("cpu") for t in tensors] in ONE device-to-host copy (bytes verbatim; the same
    helper as tasks/_shared/journal_env_mixin.host_copies, kept local so this twin stays standalone)."""
    tensors = [t.detach() for t in tensors]
    if not tensors or len({t.device for t in tensors}) != 1:
        return [t.to("cpu") for t in tensors]
    order = sorted(range(len(tensors)), key=lambda i: -tensors[i].element_size())
    blob = torch.cat([_bytes(tensors[i]) for i in order]).to("cpu")
    return _views(blob, tensors, order)


def _device_copies(tensors, device):
    """[t.to(device) for t in tensors] (host tensors) in ONE host-to-device copy (bytes verbatim)."""
    order = sorted(range(len(tensors)), key=lambda i: -tensors[i].element_size())
    blob = torch.cat([_bytes(tensors[i]) for i in order]).to(device)
    return _views(blob, tensors, order)


def _bytes(t):
    """The tensor's elements as contiguous uint8 bytes on its device (a 1-element tensor may be
    'contiguous' with a 0 stride; it is then copied so the byte view is legal)."""
    flat = t.reshape(-1)
    if flat.numel() and flat.stride(0) != 1:
        flat = flat.clone(memory_format=torch.contiguous_format)
    return flat.view(torch.uint8)


def _views(blob, tensors, order):
    out, pos = [None] * len(tensors), 0
    for i in order:
        t = tensors[i]
        size = t.numel() * t.element_size()
        out[i] = blob[pos:pos + size].view(t.dtype).reshape(t.shape)
        pos += size
    return out


def _libm_batch(requests):
    """G11: several elementwise Python ``math`` evaluations with ONE device-to-host and ONE
    host-to-device copy. ``requests`` = [(fn, (tensors...)), ...] with fn a math function (exactly
    _libm(fn, *tensors)) or WRAPPED (exactly wrapped(t) = atan2(sin(t), cos(t)), whose two inner
    results only made a lossless device round trip). Every result has the dtype, shape and bits of
    the per-call result; only the copies are shared."""
    out = [None] * len(requests)
    live = []
    for i, (fn, tensors) in enumerate(requests):
        if not tensors[0].numel():
            out[i] = torch.zeros_like(tensors[0], dtype=F64)
        else:
            live.append(i)
    if not live:
        return out
    devices = {requests[i][1][0].device for i in live}
    if len(devices) != 1:
        for i in live:
            fn, tensors = requests[i]
            out[i] = wrapped(*tensors) if fn is WRAPPED else _libm(fn, *tensors)
        return out
    hosts = _host_copies([t for i in live for t in requests[i][1]])
    results, pos = [], 0
    for i in live:
        fn, tensors = requests[i]
        arrays = [h.to("cpu", F64).numpy() for h in hosts[pos:pos + len(tensors)]]
        pos += len(tensors)
        if fn is WRAPPED:
            value = _host_eval(math.atan2, (_host_eval(math.sin, arrays), _host_eval(math.cos, arrays)))
        else:
            value = _host_eval(fn, arrays)
        results.append(torch.from_numpy(value))
    for i, moved in zip(live, _device_copies(results, devices.pop())):
        out[i] = moved
    return out


def cos(x):
    return _libm(math.cos, x)


def sin(x):
    return _libm(math.sin, x)


def atan2(y, x):
    return _libm(math.atan2, y, x)


def hypot(x, y):
    return _libm(math.hypot, x, y)


def wrapped(angle):
    return atan2(sin(angle), cos(angle))


def _min(a, b):
    """Python min(a, b): b if b < a else a."""
    return torch.where(b < a, b, a)


class DockingEpisodeBatch:
    """N parallel DockingEpisode oracles (E5 contract), vectorised.

    reset(env_ids, com_xy, berth=..., deadline_s=..., ...): per-row reset.
    observe_physics(state, collision, clearance_m): every physics substep.
    step(time_s, state, collision, out_of_bounds, clearance_m) -> codes [N]
    (0 = running, else E5 termination code) once per control step; the
    completed rows' results are latched in ``last_completed[i]`` (a dict with
    EpisodeResult.evaluator_fields() keys) before anything can reset them.
    ``state`` is a dict of float64 tensors: com [N,3], quat [N,4] (wxyz),
    lin [N,3], ang [N,3], water [N], nominal [N].
    """

    def __init__(self, num_envs, device, hull_vertices_m):
        self.n = int(num_envs)
        self.device = torch.device(device)
        hull = torch.as_tensor(np.asarray(hull_vertices_m, dtype=np.float64), dtype=F64, device=self.device)
        if hull.ndim != 2 or hull.shape[1] != 2 or hull.shape[0] < 3:
            raise ValueError("hull polygon requires at least three XY vertices")
        self.hull = hull
        z = lambda: torch.zeros(self.n, dtype=F64, device=self.device)
        b = lambda value=False: torch.full((self.n,), value, dtype=torch.bool, device=self.device)
        nan = lambda: torch.full((self.n,), math.nan, dtype=F64, device=self.device)
        # configuration (per row, set at reset)
        self.deadline_s, self.hold_s = z(), z()
        self.pos_tol, self.head_tol, self.speed_tol = z(), z(), z()
        self.capsize, self.min_heave, self.max_heave = z(), z(), z()
        self.cx, self.cy, self.heading = z(), z(), z()
        self.opening, self.back, self.half_width = z(), z(), z()
        # EpisodeBase state
        self.done, self.success, self.clean, self.contact_since_step = b(True), b(), b(True), b()
        self.reason = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.nonfinite_seen = b()
        self.prev_x, self.prev_y, self.prev_t = z(), z(), z()
        self.full_len, self.prefix_len, self.tts = z(), nan(), nan()
        self.valid_prefix_only = b()
        self.min_clearance = torch.full((self.n,), math.inf, dtype=F64, device=self.device)
        # DockingEpisode state
        self.entered, self.has_hold, self.hold_started = b(), b(), z()
        self.prev_lx, self.prev_ly, self.prev_lt = z(), z(), z()
        self.has_entry, self.entry_time = b(), z()
        # per step outputs (control-step view for rewards/metrics)
        self.conjunction = b()
        self.dwell_s = z()
        self.last_completed = [None] * self.n
        self.configured = b()
        self._g11_heading_cs = None                 # G11: (heading tensor, version, cos, sin)
        self.last_code_list = None                  # G11: host codes of the last step (None = all running)

    # ------------------------------------------------------------------ reset --
    def reset(self, env_ids, com_xy, *, berth, deadline_s, hold_s, position_tolerance_m, heading_tolerance_rad,
              speed_tolerance_mps, capsize_angle_rad, min_heave_m, max_heave_m):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if not ids.numel():
            return
        k = ids.numel()

        def rows(value):
            t = torch.as_tensor(value, dtype=F64, device=self.device)
            return t.expand(k).clone() if t.ndim == 0 else t.reshape(k)

        com = torch.as_tensor(com_xy, dtype=F64, device=self.device).reshape(k, 2)
        values = [rows(v) for v in (deadline_s, hold_s, position_tolerance_m, heading_tolerance_rad,
                                    speed_tolerance_mps)]
        if not all(bool(torch.isfinite(v).all()) and bool((v > 0).all()) for v in values):
            raise ValueError("docking deadline and tolerances must be positive and finite")
        for name, value in zip(("deadline_s", "hold_s", "pos_tol", "head_tol", "speed_tol"), values):
            getattr(self, name)[ids] = value
        for name, value in (("capsize", capsize_angle_rad), ("min_heave", min_heave_m), ("max_heave", max_heave_m)):
            getattr(self, name)[ids] = rows(value)
        cx, cy = rows(berth["center_x"]), rows(berth["center_y"])
        heading = rows(berth["heading_rad"])
        opening, back, half = rows(berth["opening_x_m"]), rows(berth["back_x_m"]), rows(berth["half_width_m"])
        if not bool((opening < back).all()) or not bool((half > 0).all()):
            raise ValueError("invalid berth interior")
        self.cx[ids], self.cy[ids], self.heading[ids] = cx, cy, heading
        self.opening[ids], self.back[ids], self.half_width[ids] = opening, back, half
        # EpisodeBase.reset
        self.prev_x[ids], self.prev_y[ids] = com[:, 0], com[:, 1]
        self.prev_t[ids] = 0.0
        self.full_len[ids] = 0.0
        self.prefix_len[ids] = math.nan
        self.tts[ids] = math.nan
        self.valid_prefix_only[ids] = False
        self.min_clearance[ids] = math.inf
        self.reason[ids] = RUNNING
        self.nonfinite_seen[ids] = False
        self.done[ids] = False
        self.success[ids] = False
        self.clean[ids] = True
        self.contact_since_step[ids] = False
        # DockingEpisode.reset
        self.entered[ids] = False
        self.has_hold[ids] = False
        self.hold_started[ids] = 0.0
        lx, ly = self._local(com[:, 0], com[:, 1], cx, cy, heading)
        self.prev_lx[ids], self.prev_ly[ids] = lx, ly
        self.prev_lt[ids] = 0.0
        self.has_entry[ids] = False
        self.entry_time[ids] = 0.0
        self.conjunction[ids] = False
        self.dwell_s[ids] = 0.0
        self.configured[ids] = True

    # -------------------------------------------------------------- geometry --
    @staticmethod
    def _local(px, py, cx, cy, heading):
        dx, dy = px - cx, py - cy
        c, s = cos(heading), sin(heading)
        return c * dx + s * dy, (-s) * dx + c * dy

    def _contains(self, px, py, yaw, rows=None):
        cx, cy, heading = self.cx, self.cy, self.heading
        opening, back, half = self.opening, self.back, self.half_width
        if rows is not None:
            cx, cy, heading, opening, back, half = (t[rows] for t in (cx, cy, heading, opening, back, half))
        x, y = self._local(px, py, cx, cy, heading)
        rel = yaw - heading
        c, s = cos(rel), sin(rel)
        vx, vy = self.hull[:, 0].unsqueeze(0), self.hull[:, 1].unsqueeze(0)
        c, s, x, y = c.unsqueeze(1), s.unsqueeze(1), x.unsqueeze(1), y.unsqueeze(1)
        along = x + c * vx - s * vy
        across = y + s * vx + c * vy
        inside = (opening.unsqueeze(1) <= along) & (along <= back.unsqueeze(1)) & (across.abs() <= half.unsqueeze(1))
        return inside.all(dim=1)

    @staticmethod
    def _yaw(q):
        w, x, y, z = q.unbind(-1)
        return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _state(self, state):
        """_safe_state: (finite mask, com, normalized quat, lin) with non-finite rows zeroed."""
        com = state["com"].to(F64)
        quat = state["quat"].to(F64)
        lin = state["lin"].to(F64)
        ang = state["ang"].to(F64)
        water = state["water"].to(F64)
        nominal = state["nominal"].to(F64)
        finite = (torch.isfinite(com).all(1) & torch.isfinite(quat).all(1) & torch.isfinite(lin).all(1)
                  & torch.isfinite(ang).all(1) & torch.isfinite(water) & torch.isfinite(nominal))
        q = torch.where(finite.unsqueeze(1), quat, torch.zeros_like(quat))
        sq = q[:, 0] * q[:, 0]
        sq = sq + q[:, 1] * q[:, 1]
        sq = sq + q[:, 2] * q[:, 2]
        sq = sq + q[:, 3] * q[:, 3]
        norm = _libm(math.sqrt, sq)
        finite = finite & torch.isfinite(norm) & (norm > 1.0e-12)
        safe_norm = torch.where(finite, norm, torch.ones_like(norm))
        qn = torch.where(finite.unsqueeze(1), q / safe_norm.unsqueeze(1), torch.zeros_like(q))
        qn[:, 0] = torch.where(finite, qn[:, 0], torch.ones_like(norm))
        zero3 = torch.zeros_like(com)
        com = torch.where(finite.unsqueeze(1), com, zero3)
        lin = torch.where(finite.unsqueeze(1), lin, zero3)
        water = torch.where(finite, water, torch.zeros_like(water))
        nominal = torch.where(finite, nominal, torch.zeros_like(nominal))
        return finite, com, qn, lin, water, nominal

    def _validity(self, finite, com, qn, water, nominal):
        x, y = qn[:, 1], qn[:, 2]
        arg = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
        return self._validity_from_tilt(finite, com, _libm(math.acos, arg), water, nominal)

    def _validity_from_tilt(self, finite, com, tilt, water, nominal):
        heave = com[:, 2] - water - nominal
        out = torch.zeros(self.n, dtype=torch.long, device=self.device)
        out = torch.where(heave > self.max_heave, torch.full_like(out, AIRBORNE), out)
        out = torch.where(heave < self.min_heave, torch.full_like(out, SINKING), out)
        out = torch.where(tilt > self.capsize, torch.full_like(out, CAPSIZE), out)
        return torch.where(finite, out, torch.full_like(out, NONFINITE))

    def _conj(self, finite, com, qn, lin):
        yaw = self._yaw(qn)
        contained = self._contains(com[:, 0], com[:, 1], yaw)
        distance = hypot(com[:, 0] - self.cx, com[:, 1] - self.cy)
        heading_error = wrapped(yaw - self.heading).abs()
        speed = hypot(lin[:, 0], lin[:, 1])
        conj = (self.entered & contained & (distance <= self.pos_tol) & (heading_error <= self.head_tol)
                & (speed <= self.speed_tol))
        return conj & finite, dict(yaw=yaw, contained=contained, distance=distance, heading_error=heading_error,
                                   speed=speed)

    # ------------------------------------------------------- G11 batched pieces --
    def _heading_cs(self):
        """(cos, sin) of self.heading, the values cos(self.heading) / sin(self.heading) return (same math
        calls on the same doubles), evaluated again only after self.heading changes (reset writes it)."""
        h = self.heading
        cache = self._g11_heading_cs
        if cache is None or cache[0] is not h or cache[1] != h._version:
            c, s = _libm_batch([(math.cos, (h,)), (math.sin, (h,))])
            cache = (h, h._version, c, s)
            self._g11_heading_cs = cache
        return cache[2], cache[3]

    @staticmethod
    def _local_cs(px, py, cx, cy, c, s):
        """_local with c, s = cos(heading), sin(heading) given."""
        dx, dy = px - cx, py - cy
        return c * dx + s * dy, (-s) * dx + c * dy

    def _contains_cs(self, px, py, c, s):
        """_contains(px, py, yaw) (rows=None) with c, s = cos(yaw - heading), sin(yaw - heading) given."""
        cx, cy = self.cx, self.cy
        opening, back, half = self.opening, self.back, self.half_width
        x, y = self._local_cs(px, py, cx, cy, *self._heading_cs())
        vx, vy = self.hull[:, 0].unsqueeze(0), self.hull[:, 1].unsqueeze(0)
        c, s, x, y = c.unsqueeze(1), s.unsqueeze(1), x.unsqueeze(1), y.unsqueeze(1)
        along = x + c * vx - s * vy
        across = y + s * vx + c * vy
        inside = (opening.unsqueeze(1) <= along) & (along <= back.unsqueeze(1)) & (across.abs() <= half.unsqueeze(1))
        return inside.all(dim=1)

    def _conj_from_parts(self, finite, parts):
        conj = (self.entered & parts["contained"] & (parts["distance"] <= self.pos_tol)
                & (parts["heading_error"] <= self.head_tol) & (parts["speed"] <= self.speed_tol))
        return conj & finite

    def _observe(self, state, collision, clearance_m, rows):
        """observe_physics; returns (finite, com, qn, lin, parts, wrapped yaw error) for step().

        G11: _state's sqrt, then ONE batched host evaluation of {acos (validity), atan2 (yaw), hypot
        (berth distance), hypot (speed)} and ONE of {cos, sin, wrapped} of yaw - heading (the _conj
        parts); every value is the one the per-call _libm evaluations gave (same math calls, same
        doubles, same device arithmetic), only the host copies are shared."""
        active = ~self.done if rows is None else (~self.done & rows)
        finite, com, qn, lin, water, nominal = self._state(state)
        x, y = qn[:, 1], qn[:, 2]
        arg = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
        qw, qx, qy, qz = qn.unbind(-1)
        tilt, yaw, distance, speed = _libm_batch([
            (math.acos, (arg,)),
            (math.atan2, (2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))),
            (math.hypot, (com[:, 0] - self.cx, com[:, 1] - self.cy)),
            (math.hypot, (lin[:, 0], lin[:, 1]))])
        observed = self._validity_from_tilt(finite, com, tilt, water, nominal)
        self.nonfinite_seen |= active & (observed == NONFINITE)
        self.reason = torch.where(active & (self.reason == RUNNING), observed, self.reason)
        hit = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if collision is None
               else torch.as_tensor(collision, dtype=torch.bool, device=self.device).clone())
        if clearance_m is not None:
            c = torch.as_tensor(clearance_m, dtype=F64, device=self.device)
            c_finite = torch.isfinite(c)
            safe_c = torch.where(c_finite, c, torch.zeros_like(c))
            self.min_clearance = torch.where(active & c_finite, _min(self.min_clearance, safe_c), self.min_clearance)
            hit = hit | ~c_finite | (c_finite & (safe_c <= 0.0))
        self.contact_since_step |= active & hit
        # DockingEpisode: break the hold on any failed sample
        hit2 = (torch.zeros_like(hit) if collision is None
                else torch.as_tensor(collision, dtype=torch.bool, device=self.device))
        if clearance_m is not None:
            hit2 = hit2 | ~(torch.as_tensor(clearance_m, dtype=F64, device=self.device) > 0.0)
        rel = yaw - self.heading
        c_rel, s_rel, wrapped_rel = _libm_batch([(math.cos, (rel,)), (math.sin, (rel,)), (WRAPPED, (rel,))])
        parts = dict(yaw=yaw, contained=self._contains_cs(com[:, 0], com[:, 1], c_rel, s_rel), distance=distance,
                     heading_error=wrapped_rel.abs(), speed=speed)
        conj = self._conj_from_parts(finite, parts)
        broken = (self.reason != RUNNING) | hit2 | ~finite | ~conj
        self.has_hold = torch.where(active & broken, torch.zeros_like(self.has_hold), self.has_hold)
        return finite, com, qn, lin, parts, wrapped_rel

    # --------------------------------------------------------- physics substep --
    def observe_physics(self, state, collision=None, clearance_m=None, rows=None):
        """EpisodeBase.observe_physics + DockingEpisode.observe_physics for the active rows.

        G11: returns _observe's (finite, com, qn, lin, parts, wrapped yaw error) for step(); the env
        and every other caller ignore it (it returned None)."""
        return self._observe(state, collision, clearance_m, rows)

    # ---------------------------------------------------------- control step --
    def step(self, time_s, state, collision=None, out_of_bounds=None, clearance_m=None):
        """One control-step decision per row; returns int64 codes (0 = running)."""
        t = torch.as_tensor(time_s, dtype=F64, device=self.device).expand(self.n).clone()
        # G11: the five entry checks travel to the host in ONE transfer (same checks, same order of the errors)
        checks = torch.stack((self.configured.all(), self.done.any(), torch.isfinite(t).all(),
                              (t <= self.prev_t).any(), (t > self.deadline_s + 1.0e-9).any())).tolist()
        if not checks[0]:
            raise RuntimeError("every row must be reset before the first step")
        if checks[1]:
            raise RuntimeError("reset required after completion (rows %s)" % self.done.nonzero().flatten().tolist())
        if not checks[2] or checks[3]:
            raise ValueError("sample clock must increase from the reset time zero")
        if checks[4]:
            raise ValueError("the adapter must include a sample exactly at the deadline")
        oob = (torch.zeros(self.n, dtype=torch.bool, device=self.device) if out_of_bounds is None
               else torch.as_tensor(out_of_bounds, dtype=torch.bool, device=self.device))
        # ---- _observe (its _state and _conj parts are the values step's own _state / _yaw / _conj gave:
        # the same state, the same qn; the parts do not read self.entered, so they are reused below)
        finite, com, qn, lin, parts, rel = self.observe_physics(state, collision=collision, clearance_m=clearance_m)
        contact = self.contact_since_step.clone()
        self.contact_since_step = torch.zeros_like(contact)
        self.clean = self.clean & ~contact
        self.valid_prefix_only |= ~finite
        # G11: hypot (step length), sin and cos of rel = wrapped(yaw - heading) in ONE batched evaluation
        step_len, sr, cr = _libm_batch([(math.hypot, (com[:, 0] - self.prev_x, com[:, 1] - self.prev_y)),
                                        (math.sin, (rel,)), (math.cos, (rel,))])
        self.full_len = torch.where(finite, self.full_len + step_len, self.full_len)
        self.prev_x = torch.where(finite, com[:, 0], self.prev_x)
        self.prev_y = torch.where(finite, com[:, 1], self.prev_y)
        self.prev_t = torch.where(finite, t, self.prev_t)
        # ---- _docking_candidate / _entry_update
        cos_h, sin_h = self._heading_cs()
        lx, ly = self._local_cs(com[:, 0], com[:, 1], self.cx, self.cy, cos_h, sin_h)
        dx = lx - self.prev_lx
        crossing = finite & ~self.entered & (self.prev_lx < self.opening) & (self.opening <= lx) & (dx > 0.0)
        safe_dx = torch.where(crossing, dx, torch.ones_like(dx))
        fraction = (self.opening - self.prev_lx) / safe_dx
        y_cross = self.prev_ly + fraction * (ly - self.prev_ly)
        half = (sr.unsqueeze(1) * self.hull[:, 0].unsqueeze(0) + cr.unsqueeze(1) * self.hull[:, 1].unsqueeze(0))
        half = half.abs().amax(dim=1)
        inward = lin[:, 0] * cos_h + lin[:, 1] * sin_h
        enter = crossing & self.clean & (inward > 0.0) & (cr > 0.0) & (y_cross.abs() + half <= self.half_width)
        entry_time = self.prev_lt + fraction * (t - self.prev_lt)
        self.entry_time = torch.where(enter, entry_time, self.entry_time)
        self.has_entry = self.has_entry | enter
        self.entered = self.entered | enter
        self.prev_lx = torch.where(finite, lx, self.prev_lx)
        self.prev_ly = torch.where(finite, ly, self.prev_ly)
        self.prev_lt = torch.where(finite, t, self.prev_lt)
        conj = self._conj_from_parts(finite, parts)
        conj = conj & self.clean
        start = conj & ~self.has_hold
        self.hold_started = torch.where(start, t, self.hold_started)
        self.has_hold = conj.clone()
        dwell = torch.where(conj, t - self.hold_started, torch.zeros_like(t))
        candidate = conj & (dwell >= self.hold_s - 1.0e-9)
        self.conjunction, self.dwell_s = conj, dwell
        # ---- _finish
        safe = (self.reason == RUNNING) & self.clean & ~oob
        latch = candidate & safe & ~self.success
        self.success = self.success | latch
        first = latch & torch.isnan(self.tts)
        self.tts = torch.where(first, t, self.tts)
        self.prefix_len = torch.where(first, self.full_len, self.prefix_len)
        timeout = t >= self.deadline_s - 1.0e-9
        codes = torch.zeros(self.n, dtype=torch.long, device=self.device)
        codes = torch.where(timeout, torch.full_like(codes, TIMEOUT), codes)
        codes = torch.where(self.success, torch.full_like(codes, SUCCESS), codes)
        codes = torch.where(contact, torch.full_like(codes, COLLISION), codes)
        codes = torch.where(oob, torch.full_like(codes, OUT_OF_BOUNDS), codes)
        codes = torch.where(self.reason != RUNNING, self.reason, codes)
        ended = codes != RUNNING
        self.last_code_list = None                  # G11: host codes of the last step (None = all running)
        if bool(ended.any()):
            keep = (codes == SUCCESS) | (codes == TIMEOUT)
            self.success = torch.where(ended & ~keep, torch.zeros_like(self.success), self.success)
            self.done = self.done | ended
            self._latch(ended, codes, finite, parts, lx, ly, dwell)
        return codes

    def _latch(self, ended, codes, finite, parts, lx, ly, dwell):
        rows = ended.nonzero().flatten().tolist()
        host = {name: getattr(self, name).detach().cpu().tolist() for name in (
            "success", "tts", "prefix_len", "full_len", "valid_prefix_only", "min_clearance", "nonfinite_seen",
            "deadline_s", "entered", "has_entry", "entry_time")}
        fin = finite.detach().cpu().tolist()
        p = {name: parts[name].detach().cpu().tolist() for name in ("distance", "heading_error", "speed", "contained")}
        loc = (lx.detach().cpu().tolist(), ly.detach().cpu().tolist())
        dw = dwell.detach().cpu().tolist()
        code_list = codes.detach().cpu().tolist()
        self.last_code_list = code_list             # G11: the env reuses it (rows with a nonzero code)
        for i in rows:
            entry = host["entry_time"][i] if host["has_entry"][i] else None
            if fin[i]:
                extras = dict(docking_position_error_m=p["distance"][i], docking_heading_error_rad=p["heading_error"][i],
                              docking_planar_speed_mps=p["speed"][i], docking_longitudinal_error_m=loc[0][i],
                              docking_lateral_error_m=loc[1][i], berth_contained=bool(p["contained"][i]),
                              berth_entry_latched=bool(host["entered"][i]), berth_entry_time_s=entry,
                              docking_valid_dwell_s=dw[i])
            else:
                extras = dict(docking_position_error_m=None, docking_heading_error_rad=None,
                              docking_planar_speed_mps=None, docking_longitudinal_error_m=None,
                              docking_lateral_error_m=None, berth_contained=False,
                              berth_entry_latched=bool(host["entered"][i]), berth_entry_time_s=entry,
                              docking_valid_dwell_s=0.0)
            extras.update(path_cross_track_error_integral_m_s=0.0,
                          path_length_valid_prefix_only=bool(host["valid_prefix_only"][i]),
                          min_clearance_m=host["min_clearance"][i], nonfinite_seen=bool(host["nonfinite_seen"][i]),
                          deadline_s=host["deadline_s"][i], integrated_squared_thrust_n2s=None)
            success = bool(host["success"][i])
            reason = int(code_list[i])
            names = {0: "running", 1: "success", 2: "collision", 3: "timeout", 4: "out_of_bounds", 5: "failure",
                     6: "capsize", 7: "sinking", 8: "airborne", 9: "non_finite_state"}
            self.last_completed[i] = dict(
                episode_success=success, time_to_success=host["tts"][i] if success else math.nan,
                episode_path_length=host["full_len"][i],
                success_prefix_path_length_m=host["prefix_len"][i] if success else math.nan,
                full_episode_path_length_m=host["full_len"][i], termination_reason=reason,
                termination_reason_name=names[reason], **extras)


def scalar_fields(result):
    """EpisodeResult.evaluator_fields() of E5's scalar oracle, for parity checks."""
    return dict(result.evaluator_fields())


__all__ = ["DockingEpisodeBatch", "scalar_fields", "cos", "sin", "atan2", "hypot", "wrapped"]
