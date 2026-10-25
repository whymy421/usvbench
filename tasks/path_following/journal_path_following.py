"""Journal (task contract v2) wiring of the path-following family (round F, task F3).

Everything here is used ONLY by journal cfgs (``journal_contract_version == 2``,
E5's ``PathFollowing{BlueBoat,WAMV}JournalEnvCfg``). The legacy ids keep the
byte-identical legacy methods of ``path_following_env.PathFollowingEnv``: the
env composes ``PathFollowingJournalMixin`` and, only for a journal cfg,
constructs itself through ``_journal_construct`` which binds the ``_journal_*``
methods below over the DirectRLEnv hooks on the INSTANCE (the class methods
stay untouched, CONTEXT_R4 D15; same technique as D3's station proof).

No Isaac Lab import happens in this module (CPU-testable); the env module
supplies the Isaac base class.

Contents
  * ``py_hypot``: a float64 torch emulation of THIS interpreter's
    ``math.hypot`` (CPython vector_norm), because the scalar contract oracle
    (E5 ``journal_contracts_v2``) measures every distance with ``math.hypot``
    and ``torch.hypot`` differs from it in the last bit on ~1e-5 of inputs.
    ``assert_hypot_parity`` re-checks it at env construction (fail closed).
  * ``OrderedPathTensorOracle``: the vectorised torch twin of
    ``journal_contracts_v2.OrderedPathEpisode`` (gates in order, swept between
    control samples, XTE integrated on the ACTIVE segment, validity latched at
    substeps, E5 termination precedence). Bit-for-bit parity with the scalar
    oracle is tested (``tasks/path_following/tests_journal``).
  * journal layout (``build_layout``) keyed by the scenario protocol (D16:
    the E5 layout of the key, never redrawn), the D7 ``TaskContext`` built
    from the REAL layout, the D7 admission-pool file codec (JSON + binary
    witness block, xz, SHA-256 sidecar, pinned in ``journal_d7/POOL_MANIFEST.json``;
    loaded by path, fail closed), the D16 pool/bridge wrappers (admitted
    ladder or explicit ``not_admitted`` record; a not-admitted key runs D0 with
    exposure ``not_admitted``; a key beyond the pool's offered grid is a
    not-admitted key with the single reason ``beyond_offered_grid``, CONTEXT_R4
    D24) and the 36-ray point-return proxy (E4
    ``visibility.py``: centre ray to the nearest surface point of each VISIBLE
    D7 object; the family has no static obstacles, so the reserved D7 slot is
    the only return).
  * ``PathFollowingJournalMixin``: the family hooks of ``JournalEnvMixin``
    and the journal versions of the DirectRLEnv hooks, following the mixin's
    EXACT CALL ORDER.
"""
from __future__ import annotations

from dataclasses import asdict, fields as dataclass_fields, is_dataclass
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np
import torch

from .._shared import journal_cfg_v2 as plans
from .._shared import journal_contracts_v2 as contracts
from .._shared import journal_layouts_v2 as layouts
from .._shared import scenario_rng
from .._shared import vehicles_v2
from .._shared.conditions import pairing
from .._shared.journal_env_mixin import JournalEnvMixin, device_copies
from .._shared.scenario_draws import GROUP_ROUTE, make_scenario_rng

TASK = "path_following"
FAMILY = "path_following"
HAZARD_GROUP = "d7_" + TASK
POOL_SCHEMA = "usvbench.f3.d7_admission_pool/2"
POOL_DIR = Path(__file__).resolve().parent / "journal_d7"
F3_WIRING_VERSION = "usvbench.f3.path_following_journal/3"
DBL_MIN = 2.2250738585072014e-308
F64 = torch.float64


def _dynamic():
    from .._shared.conditions import dynamic
    return dynamic


# =========================================================== numerics ======
def _split(x):
    """Dekker/Veltkamp split: x == hi + lo exactly, hi has <= 26 significant bits."""
    t = x * 134217729.0
    hi = t - (t - x)
    return hi, x - hi


def _dl_mul(x, y):
    """Exact product as (z, zz) with z + zz == x*y (Dekker mul12; = CPython fma form)."""
    xh, xl = _split(x)
    yh, yl = _split(y)
    p = xh * yh
    q = xh * yl + xl * yh
    z = p + q
    zz = p - z + q + xl * yl
    return z, zz


def _fast_sum(a, b):
    x = a + b
    return x, (a - x) + b


def _vector_norm_312(values, maximum):
    """CPython >= 3.12 Modules/mathmodule.c vector_norm (dl_mul / dl_fast_sum form)."""
    _, exponent = torch.frexp(maximum)
    scale = torch.ldexp(torch.ones_like(maximum), -exponent)
    csum = torch.ones_like(maximum)
    frac1 = torch.zeros_like(maximum)
    frac2 = torch.zeros_like(maximum)
    for value in values:
        x = value * scale
        pr_hi, pr_lo = _dl_mul(x, x)
        sm_hi, sm_lo = _fast_sum(csum, pr_hi)
        csum = sm_hi
        frac1 = frac1 + pr_lo
        frac2 = frac2 + sm_lo
    h = torch.sqrt(csum - 1.0 + (frac1 + frac2))
    pr_hi, pr_lo = _dl_mul(-h, h)
    sm_hi, sm_lo = _fast_sum(csum, pr_hi)
    csum = sm_hi
    frac1 = frac1 + pr_lo
    frac2 = frac2 + sm_lo
    x = csum - 1.0 + (frac1 + frac2)
    h = h + x / (2.0 * h)
    return h / scale


def _vector_norm_310(values, maximum):
    """CPython 3.10/3.11 vector_norm (T27 split, three compensation terms).

    UNVERIFIED on a 3.10/3.11 interpreter here (only 3.13 was available);
    ``assert_hypot_parity`` checks it at runtime on the interpreter in use.
    """
    t27 = 134217729.0
    _, exponent = torch.frexp(maximum)
    scale = torch.ldexp(torch.ones_like(maximum), -exponent)
    csum = torch.ones_like(maximum)
    frac1 = torch.zeros_like(maximum)
    frac2 = torch.zeros_like(maximum)
    frac3 = torch.zeros_like(maximum)
    for value in values:
        x = value * scale
        t = x * t27
        hi = t - (t - x)
        lo = x - hi
        x = hi * hi
        old = csum
        csum = csum + x
        frac1 = frac1 + ((old - csum) + x)
        x = 2.0 * hi * lo
        old = csum
        csum = csum + x
        frac2 = frac2 + ((old - csum) + x)
        frac3 = frac3 + lo * lo
    h = torch.sqrt(csum - 1.0 + (frac1 + frac2 + frac3))
    x = h
    t = x * t27
    hi = t - (t - x)
    lo = x - hi
    x = -hi * hi
    old = csum
    csum = csum + x
    frac1 = frac1 + ((old - csum) + x)
    x = -2.0 * hi * lo
    old = csum
    csum = csum + x
    frac2 = frac2 + ((old - csum) + x)
    x = -lo * lo
    old = csum
    csum = csum + x
    frac3 = frac3 + ((old - csum) + x)
    x = csum - 1.0 + (frac1 + frac2 + frac3)
    h = h + x / (2.0 * h)
    return h / scale


_VECTOR_NORM = _vector_norm_312 if sys.version_info >= (3, 12) else _vector_norm_310


def py_hypot(a, b):
    """``math.hypot(a, b)`` element-wise on float64 tensors, bit-identical to CPython >= 3.10."""
    a = torch.as_tensor(a, dtype=F64).abs()
    b = torch.as_tensor(b, dtype=F64, device=a.device).abs()
    a, b = torch.broadcast_tensors(a, b)
    maximum = torch.maximum(a, b)
    nan = torch.isnan(a) | torch.isnan(b)
    inf = torch.isinf(a) | torch.isinf(b)
    zero = maximum == 0.0
    sub = (maximum < DBL_MIN) & ~zero
    special = zero | nan | inf
    one = torch.ones_like(maximum)
    va = torch.where(special, one, torch.where(sub, a / DBL_MIN, a))
    vb = torch.where(special, one, torch.where(sub, b / DBL_MIN, b))
    vm = torch.where(special, one, torch.where(sub, maximum / DBL_MIN, maximum))
    out = _VECTOR_NORM((va, vb), vm)
    out = torch.where(sub, out * DBL_MIN, out)
    out = torch.where(zero, torch.zeros_like(out), out)
    out = torch.where(nan, torch.full_like(out, math.nan), out)
    return torch.where(inf, torch.full_like(out, math.inf), out)


_HYPOT_CHECKED = {}


def assert_hypot_parity(count=20000, seed=20260930):
    """Fail closed unless py_hypot reproduces this interpreter's math.hypot bit for bit."""
    key = (sys.version_info[:3], count, seed)
    if key in _HYPOT_CHECKED:
        return _HYPOT_CHECKED[key]
    if sys.version_info < (3, 10):
        raise RuntimeError("the path-following tensor twin emulates math.hypot of CPython >= 3.10")
    rng = np.random.default_rng(seed)
    a = (rng.random(count) - 0.5) * np.exp(rng.normal(0.0, 6.0, count))
    b = (rng.random(count) - 0.5) * np.exp(rng.normal(0.0, 6.0, count))
    b[::5] = a[::5] * 1.0e-9
    b[::7] = 0.0
    a[::11] = 3.0
    b[::11] = 4.0
    got = py_hypot(torch.as_tensor(a), torch.as_tensor(b)).tolist()
    bad = [(x, y) for x, y, g in zip(a.tolist(), b.tolist(), got) if math.hypot(x, y) != g]
    if bad:
        raise RuntimeError("py_hypot differs from this interpreter's math.hypot on %d of %d samples "
                           "(first %r); the tensor twin cannot claim parity with the scalar oracle"
                           % (len(bad), count, bad[0]))
    _HYPOT_CHECKED[key] = count
    return count


def _bits(value):
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _float(bits):
    return struct.unpack("<d", struct.pack("<q", int(bits)))[0]


def _order_key(value):
    """Monotone integer key over all doubles (-0.0 and +0.0 share key 0)."""
    bits = _bits(value)
    return bits if bits >= 0 else -(bits & 0x7FFFFFFFFFFFFFFF)


def _order_value(key):
    return _float(key) if key >= 0 else _float((-key) - (1 << 63))


def capsize_cos_threshold(limit_rad):
    """Smallest double c in [-1, 1] with math.acos(c) <= limit_rad.

    The scalar oracle tests ``math.acos(clamp(1 - 2(x^2+y^2))) > limit``; for
    a monotone libm acos that is exactly ``clamped < c`` with this c, so the
    tensor twin reproduces the capsize decision bit for bit without calling
    acos. The monotonicity is checked on +-4096 ulps around c (fail closed).
    """
    limit = float(limit_rad)
    if not 0.0 < limit < math.pi:
        raise ValueError("capsize limit must lie in (0, pi)")
    low, high = _order_key(-1.0), _order_key(1.0)      # acos(-1) = pi > limit, acos(1) = 0 <= limit
    while high - low > 1:
        mid = (low + high) // 2
        if math.acos(_order_value(mid)) <= limit:
            high = mid
        else:
            low = mid
    threshold = _order_value(high)
    for step in range(-4096, 4097):
        c = _order_value(high + step)
        if -1.0 <= c <= 1.0 and (math.acos(c) > limit) != (c < threshold):
            raise RuntimeError("math.acos is not monotone near the capsize threshold; "
                               "the tensor twin cannot reproduce the scalar decision")
    return threshold


# ===================================================== tensor twin =========
class OrderedPathTensorOracle:
    """Vectorised twin of ``journal_contracts_v2.OrderedPathEpisode`` (float64).

    Per env: ``reset`` (route, gate radius, deadline, validity limits, COM
    xy), ``observe_physics`` after every physics substep (validity latch on
    the PREVIOUS substep's state; contact accepted too), ``step`` once per
    control sample (returns the E5 reason code, 0 while running). Completed
    rows keep an immutable ``journal_contracts_v2.EpisodeResult`` in
    ``last_completed[i]`` (the scalar oracle's type) until the next reset.
    Rows are independent: every operation is masked per row.
    """

    RUNNING = int(contracts.TerminationReason.RUNNING)

    def __init__(self, num_envs, num_gates, device):
        n, g = int(num_envs), int(num_gates)
        if n <= 0 or g <= 0:
            raise ValueError("num_envs and num_gates must be positive")
        self.num_envs, self.num_gates, self.device = n, g, torch.device(device)
        z = lambda dtype=F64: torch.zeros(n, dtype=dtype, device=self.device)
        self.route = torch.zeros((n, g + 1, 2), dtype=F64, device=self.device)
        self.gate_radius = torch.ones(n, dtype=F64, device=self.device)
        self.deadline = torch.ones(n, dtype=F64, device=self.device)
        self.cos_threshold = torch.full((n,), 0.5, dtype=F64, device=self.device)
        self.min_heave = torch.full((n,), -1.0, dtype=F64, device=self.device)
        self.max_heave = torch.ones(n, dtype=F64, device=self.device)
        self.done = torch.ones(n, dtype=torch.bool, device=self.device)
        self.armed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = z(torch.bool)
        self.clean = z(torch.bool)
        self.contact_since_step = z(torch.bool)
        self.validity = z(torch.long)
        self.nonfinite_seen = z(torch.bool)
        self.next_gate = torch.ones(n, dtype=torch.long, device=self.device)
        self.gate_times = torch.full((n, g), math.nan, dtype=F64, device=self.device)
        self.previous_xy = torch.zeros((n, 2), dtype=F64, device=self.device)
        self.previous_time = z()
        self.full_path = z()
        self.prefix_path = torch.full((n,), math.nan, dtype=F64, device=self.device)
        self.time_to_success = torch.full((n,), math.nan, dtype=F64, device=self.device)
        self.xte_integral = z()
        self.previous_xte = z()
        self.previous_xte_set = z(torch.bool)
        self.valid_prefix_only = z(torch.bool)
        self.min_clearance = torch.full((n,), math.inf, dtype=F64, device=self.device)
        self.xte_square_sum = z()
        self.xte_samples = z(torch.long)
        self.last_completed = [None] * n
        self.last_code_list = None              # G11: host codes of the last step (None = all zero)
        self._thresholds = {}

    # ---------------------------------------------------------- geometry --
    @staticmethod
    def _clamp01(v):
        """Python ``max(0.0, min(1.0, v))`` element-wise (NaN -> 1.0 like Python)."""
        m = torch.where(v < 1.0, v, torch.ones_like(v))
        return torch.where(m > 0.0, m, torch.zeros_like(m))

    @classmethod
    def cross_track(cls, point, a, b):
        """``journal_contracts_v2.cross_track_error_m(point, (a, b))`` for one segment per row."""
        dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        n = dx * dx + dy * dy
        nonzero = n != 0.0
        ratio = ((point[:, 0] - a[:, 0]) * dx + (point[:, 1] - a[:, 1]) * dy) / torch.where(
            nonzero, n, torch.ones_like(n))
        t = torch.where(nonzero, cls._clamp01(ratio), torch.zeros_like(n))
        return py_hypot(point[:, 0] - (a[:, 0] + t * dx), point[:, 1] - (a[:, 1] + t * dy))

    @classmethod
    def segment_hits_disk(cls, p0, p1, center, radius):
        """``journal_contracts_v2.segment_hits_disk`` element-wise."""
        ax, ay = p0[:, 0] - center[:, 0], p0[:, 1] - center[:, 1]
        dx, dy = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        n = dx * dx + dy * dy
        zero = n == 0.0
        ratio = -(ax * dx + ay * dy) / torch.where(zero, torch.ones_like(n), n)
        t = torch.where(zero, torch.zeros_like(n), cls._clamp01(ratio))
        return py_hypot(ax + t * dx, ay + t * dy) <= radius

    # ---------------------------------------------------------- validity --
    def _rows_f64(self, value):
        """torch.as_tensor(value, F64, self.device).expand(N); a Python number (the nominal COM height)
        is written by torch.full instead: the same float64 value, a fill kernel, no host-to-device copy."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.full((self.num_envs,), float(value), dtype=F64, device=self.device)
        return torch.as_tensor(value, dtype=F64, device=self.device).expand(self.num_envs)

    def _state(self, com, quat, lin, ang, water, nominal):
        """(valid mask, observed validity code) with the scalar priority order."""
        com, quat = com.to(F64), quat.to(F64)
        lin, ang = lin.to(F64), ang.to(F64)
        water = self._rows_f64(water)
        nominal = self._rows_f64(nominal)
        w, x, y, z = quat.unbind(-1)
        norm = torch.sqrt(((w * w + x * x) + y * y) + z * z)
        valid = (torch.isfinite(com).all(-1) & torch.isfinite(quat).all(-1) & torch.isfinite(lin).all(-1)
                 & torch.isfinite(ang).all(-1) & torch.isfinite(water) & torch.isfinite(nominal)
                 & torch.isfinite(norm) & (norm > 1.0e-12))
        safe = torch.where(valid, norm, torch.ones_like(norm))
        xn, yn = x / safe, y / safe
        cosine = 1.0 - 2.0 * (xn * xn + yn * yn)
        cosine = torch.where(cosine < 1.0, cosine, torch.ones_like(cosine))
        cosine = torch.where(cosine > -1.0, cosine, -torch.ones_like(cosine))
        heave = com[:, 2] - water - nominal
        code = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        code = torch.where(heave > self.max_heave, int(contracts.TerminationReason.AIRBORNE), code)
        code = torch.where(heave < self.min_heave, int(contracts.TerminationReason.SINKING), code)
        code = torch.where(cosine < self.cos_threshold, int(contracts.TerminationReason.CAPSIZE), code)
        code = torch.where(valid, code, int(contracts.TerminationReason.NONFINITE))
        return valid, code

    def _latch(self, rows, observed):
        nonfinite = rows & (observed == int(contracts.TerminationReason.NONFINITE))
        self.nonfinite_seen |= nonfinite
        first = rows & (self.validity == self.RUNNING)
        self.validity = torch.where(first, observed, self.validity)

    # ------------------------------------------------------------- reset --
    def reset(self, env_ids, route_xy, gate_radius_m, deadline_s, validity_limits, com_xy):
        """Install new episodes for ``env_ids`` (other rows untouched)."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        route = torch.as_tensor(route_xy, dtype=F64, device=self.device).reshape(ids.numel(), -1, 2)
        if route.shape[1] != self.num_gates + 1:
            raise ValueError("route needs a start plus %d gates" % self.num_gates)
        radius = torch.as_tensor(gate_radius_m, dtype=F64, device=self.device).expand(ids.numel()).clone()
        deadline = torch.as_tensor(deadline_s, dtype=F64, device=self.device).expand(ids.numel()).clone()
        com = torch.as_tensor(com_xy, dtype=F64, device=self.device).reshape(ids.numel(), 2)
        if not bool(torch.isfinite(route).all()) or not bool(torch.isfinite(com).all()):
            raise ValueError("malformed or non-finite vector")
        if not bool(((radius > 0) & torch.isfinite(radius)).all()):
            raise ValueError("path needs a start, gates, and a positive radius")
        if not bool(((deadline > 0) & torch.isfinite(deadline)).all()):
            raise ValueError("deadline must be positive and finite")
        limits = []
        for item in (validity_limits if isinstance(validity_limits, (list, tuple))
                     else [validity_limits] * ids.numel()):
            limits.append(item if isinstance(item, contracts.ValidityLimits)
                          else contracts.ValidityLimits(**dict(item)))
        cos_t = []
        for lim in limits:
            if lim.capsize_angle_rad not in self._thresholds:
                self._thresholds[lim.capsize_angle_rad] = capsize_cos_threshold(lim.capsize_angle_rad)
            cos_t.append(self._thresholds[lim.capsize_angle_rad])
        self.route[ids] = route
        self.gate_radius[ids] = radius
        self.deadline[ids] = deadline
        self.cos_threshold[ids] = torch.as_tensor(cos_t, dtype=F64, device=self.device)
        self.min_heave[ids] = torch.as_tensor([l.min_heave_m for l in limits], dtype=F64, device=self.device)
        self.max_heave[ids] = torch.as_tensor([l.max_heave_m for l in limits], dtype=F64, device=self.device)
        self.done[ids] = False
        self.armed[ids] = True
        self.success[ids] = False
        self.clean[ids] = True
        self.contact_since_step[ids] = False
        self.validity[ids] = self.RUNNING
        self.nonfinite_seen[ids] = False
        self.next_gate[ids] = 1
        self.gate_times[ids] = math.nan
        self.previous_xy[ids] = com
        self.previous_time[ids] = 0.0
        self.full_path[ids] = 0.0
        self.prefix_path[ids] = math.nan
        self.time_to_success[ids] = math.nan
        self.xte_integral[ids] = 0.0
        self.previous_xte[ids] = self.cross_track(com, route[:, 0], route[:, 1])
        self.previous_xte_set[ids] = True
        self.valid_prefix_only[ids] = False
        self.min_clearance[ids] = math.inf
        self.xte_square_sum[ids] = 0.0
        self.xte_samples[ids] = 0
        for i in ids.tolist():
            self.last_completed[i] = None

    # --------------------------------------------------------- substeps --
    def observe_physics(self, com, quat, lin, ang, water, nominal, collision=None):
        """EpisodeBase.observe_physics on every active (not done) row."""
        rows = self.armed & ~self.done
        _, observed = self._state(com, quat, lin, ang, water, nominal)
        self._latch(rows, observed)
        if collision is not None:
            hit = torch.as_tensor(collision, dtype=torch.bool, device=self.device).expand(self.num_envs)
            self.contact_since_step |= rows & hit

    # ------------------------------------------------------------- step --
    def step(self, time_s, com, quat, lin, ang, water, nominal, *, collision=None, out_of_bounds=None):
        """One control sample for every active row; returns (codes, gate_event, xte).

        ``time_s`` [N] (float64 seconds since the row's reset); rows that are
        done or were never reset are left unchanged and report code 0.
        """
        n = self.num_envs
        rows = self.armed & ~self.done
        t = torch.as_tensor(time_s, dtype=F64, device=self.device).expand(n)
        collision = (torch.zeros(n, dtype=torch.bool, device=self.device) if collision is None else
                     torch.as_tensor(collision, dtype=torch.bool, device=self.device).expand(n))
        oob = (torch.zeros(n, dtype=torch.bool, device=self.device) if out_of_bounds is None else
               torch.as_tensor(out_of_bounds, dtype=torch.bool, device=self.device).expand(n))
        bad_clock = rows & (~torch.isfinite(t) | (t <= self.previous_time))
        # G11: both entry checks travel to the host in ONE transfer (same checks, same order of the errors)
        checks = torch.stack((bad_clock.any(), (rows & (t > self.deadline + 1.0e-9)).any())).tolist()
        if checks[0]:
            raise ValueError("sample clock must increase from the reset time zero")
        if checks[1]:
            raise ValueError("the adapter must include a sample exactly at the deadline")
        com64 = com.to(F64)
        valid, observed = self._state(com, quat, lin, ang, water, nominal)
        index = torch.arange(n, device=self.device)
        start = self.route[index, (self.next_gate - 1).clamp(0, self.num_gates)]
        end = self.route[index, self.next_gate.clamp(0, self.num_gates)]
        point = torch.where(valid.unsqueeze(-1), com64[:, :2], torch.zeros_like(com64[:, :2]))
        xte = self.cross_track(point, start, end)
        # _observe -> observe_physics(state, collision) then the contact bookkeeping
        self._latch(rows, observed)
        self.contact_since_step |= rows & collision
        contact = rows & self.contact_since_step
        self.contact_since_step = torch.where(rows, torch.zeros_like(contact), self.contact_since_step)
        self.clean = torch.where(rows, self.clean & ~contact, self.clean)
        live = rows & valid
        self.valid_prefix_only |= rows & ~valid
        previous = self.previous_xy.clone()
        step_len = py_hypot(point[:, 0] - previous[:, 0], point[:, 1] - previous[:, 1])
        self.full_path = torch.where(live, self.full_path + step_len, self.full_path)
        prior = torch.where(self.previous_xte_set, self.previous_xte, xte)
        increment = 0.5 * (prior + xte) * (t - self.previous_time)
        self.xte_integral = torch.where(live, self.xte_integral + increment, self.xte_integral)
        self.previous_xte = torch.where(live, xte, self.previous_xte)
        self.previous_xte_set |= live
        self.xte_square_sum = torch.where(live, self.xte_square_sum + xte * xte, self.xte_square_sum)
        self.xte_samples = torch.where(live, self.xte_samples + 1, self.xte_samples)
        self.previous_xy = torch.where(live.unsqueeze(-1), point, self.previous_xy)
        self.previous_time = torch.where(live, t, self.previous_time)
        # ordered gate: swept COM motion between control samples
        can = live & (self.next_gate < self.num_gates + 1)
        gate = self.route[index, self.next_gate.clamp(0, self.num_gates)]
        hit = can & self.segment_hits_disk(previous, point, gate, self.gate_radius)
        slot = (self.next_gate - 1).clamp(0, self.num_gates - 1)
        times = self.gate_times.clone()
        times[index, slot] = torch.where(hit, t, times[index, slot])
        self.gate_times = times
        self.next_gate = torch.where(hit, self.next_gate + 1, self.next_gate)
        more = hit & (self.next_gate < self.num_gates + 1)
        a = self.route[index, (self.next_gate - 1).clamp(0, self.num_gates)]
        b = self.route[index, self.next_gate.clamp(0, self.num_gates)]
        self.previous_xte = torch.where(more, self.cross_track(point, a, b), self.previous_xte)
        # _finish
        candidate = self.next_gate == self.num_gates + 1
        running = self.validity == self.RUNNING
        safe = running & self.clean & ~oob
        newly = rows & candidate & safe & ~self.success
        self.success |= newly
        first = newly & torch.isnan(self.time_to_success)
        self.time_to_success = torch.where(first, t, self.time_to_success)
        self.prefix_path = torch.where(first, self.full_path, self.prefix_path)
        timeout = t >= self.deadline - 1.0e-9
        code = torch.zeros(n, dtype=torch.long, device=self.device)
        code = torch.where(timeout, int(contracts.TerminationReason.TIMEOUT), code)
        code = torch.where(self.success, int(contracts.TerminationReason.SUCCESS), code)
        code = torch.where(contact, int(contracts.TerminationReason.COLLISION), code)
        code = torch.where(oob, int(contracts.TerminationReason.OUT_OF_BOUNDS), code)
        code = torch.where(~running, self.validity, code)
        code = torch.where(rows, code, torch.zeros_like(code))
        finished = code != 0
        keep = (code == int(contracts.TerminationReason.SUCCESS)) | (code == int(contracts.TerminationReason.TIMEOUT))
        self.success = torch.where(finished & ~keep, torch.zeros_like(self.success), self.success)
        self.done |= finished
        self.last_code_list = None              # G11: host codes of the last step (None = all zero)
        if bool(finished.any()):
            codes_host = code.tolist()
            self.last_code_list = codes_host
            # finished == code != 0: the rows from the host list (was a second nonzero transfer)
            for i in [j for j, c in enumerate(codes_host) if c != 0]:
                self.last_completed[i] = self._result(i, codes_host[i])
        return code, hit, xte

    def _result(self, i, code):
        success = bool(self.success[i])
        gates = int(self.next_gate[i]) - 1
        times = tuple(float(v) for v in self.gate_times[i, :gates].tolist())
        details = dict(ordered_gates_completed=gates, ordered_gates_total=self.num_gates,
                       gate_times_s=times)
        details.update(path_cross_track_error_integral_m_s=float(self.xte_integral[i]),
                       path_length_valid_prefix_only=bool(self.valid_prefix_only[i]),
                       min_clearance_m=float(self.min_clearance[i]),
                       nonfinite_seen=bool(self.nonfinite_seen[i]),
                       deadline_s=float(self.deadline[i]),
                       integrated_squared_thrust_n2s=None)
        return contracts.EpisodeResult(
            success, float(self.time_to_success[i]) if success else math.nan,
            float(self.prefix_path[i]) if success else math.nan,
            float(self.full_path[i]), int(code), details)

    def xte_rms(self, i=None):
        """RMS of the active-segment XTE over the control samples (reported beside the integral)."""
        rms = torch.sqrt(self.xte_square_sum / self.xte_samples.clamp(min=1).to(F64))
        return rms if i is None else float(rms[i])


# ================================================= layouts and D7 context ==
EXPOSURE_NOT_ADMITTED = "not_admitted"
EXPOSURE_ASSIGNED = "assigned"
PER_DOSE_FIELDS = ("offered", "admitted", "not_admitted")
D16_RULE = (
    "CONTEXT_R4 D16: the scenario of key (seed, env, episode) is ALWAYS the E5 layout of that key (scenario "
    "protocol 2, route stream 'route'; never redrawn, no attempt streams). The pool stores, per offered key, "
    "either the jointly admitted D0..D4 assignment ladder or an explicit not_admitted record with E4's "
    "rejection reasons. A not-admitted key runs the dynamic suite at D0 at every requested level (no D7 "
    "object; the bridge row carries an empty assignment) and its episode stamp carries exposure "
    "'not_admitted' with the reasons (unexposed for the dynamic treatment, paired for every other suite)")
BEYOND_GRID_REASON = "beyond_offered_grid"
D24_RULE = (
    "CONTEXT_R4 D24: a key with the run's evaluation seed and hazard group, env inside the pool's offered grid "
    "and episode index >= the grid's episodes per env is a NOT-ADMITTED key with the single reason "
    "'beyond_offered_grid' (rejections [[0, 'beyond_offered_grid']]): the env builds the key's E5 layout (the "
    "pool has no record of it, so the pool geometry check is skipped), the dynamic suite runs D0 (no D7 "
    "object) and the stamp is the D16 not_admitted stamp. Such trailing episodes (per-env quota evaluators) "
    "are never counted. Keys outside the env range or of another seed or hazard group stay preflight errors "
    "(D7 B02)")
ONLINE_BEYOND_GRID_RULE = (
    "opt-in cfg.journal_d7_online_admission: keys outside the committed grid (seed and hazard group of the "
    "pool) are admitted online on their E5 layout under their own pool digest (online_admitted_keys); off for "
    "journal runs (CONTEXT_R4 D24 (e))")


def journal_layout(vehicle, tier, seed, env_index, episode_index):
    """The E5 journal layout of one scenario key (seed, env, episode) (scenario protocol 2, group 'route').

    D16: this is the ONLY layout of the key, whatever E4 decides about its
    dynamic-hazard ladder.
    """
    rng = scenario_rng.numpy_generator(scenario_rng.SCENARIO_PROTOCOL_VERSION, int(seed), int(env_index),
                                       int(episode_index), GROUP_ROUTE)
    layout = layouts.build_layout(TASK, int(tier), vehicle, rng)
    if len(layout.waypoints) != 4:
        raise ValueError("journal path following expects four waypoint gates")
    return layout


def _vessel_of_job(job):
    return job[0].envelope.vessel_id


def admit_path_following_grid(vessels, seed, num_envs, episodes, tier=2, workers=1, log=None):
    """Offline D7 admission of the grid on the REAL layouts (D7 B05; E4 admit_grid, joint; D16).

    Every key (vessel, env < num_envs, episode < episodes) is offered ONCE on
    its E5 layout. Returns {vessel: {"pool", "offered", "geometry",
    "not_admitted"}}: the E4 AdmissionPool, the offered EpisodeKeys (admission
    order), {(env, episode): layout geometry hash} for every offered key and
    {(env, episode): ((dose, reason), ...)} for every key E4 did not admit.
    """
    keys = [(v, e, k) for v in vessels for e in range(int(num_envs)) for k in range(int(episodes))]
    return admit_path_following_keys(keys, seed, tier=tier, workers=workers, log=log)


def admit_path_following_keys(keys, seed, tier=2, workers=1, log=None):
    """admit_path_following_grid for an explicit list of (vessel, env, episode) keys."""
    dyn = _dynamic()
    keys = [(str(v), int(e), int(k)) for v, e, k in keys]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate scenario keys offered for admission")
    jobs = []
    for vessel, env, episode in keys:
        layout = journal_layout(vessel, tier, seed, env, episode)
        jobs.append((d7_task_context(layout, vessel), dyn.EpisodeKey(int(seed), env, episode, HAZARD_GROUP),
                     admission_suite(vessel)))
    pools = dyn.admit_grid(jobs, joint=True, workers=workers, group_by=_vessel_of_job)
    out = {}
    for vessel in sorted({v for v, _, _ in keys}):
        pool = pools[vessel]
        vessel_jobs = [job for job in jobs if _vessel_of_job(job) == vessel]
        offered = [job[1] for job in vessel_jobs]
        geometry = {(job[1].env, job[1].episode): job[0].geometry_id for job in vessel_jobs}
        not_admitted = derive_not_admitted(pool, offered)
        out[vessel] = {"pool": pool, "offered": offered, "geometry": geometry, "not_admitted": not_admitted}
        if log is not None:
            log("%s: %d keys offered, %d admitted, %d not_admitted (D16: no redraw)"
                % (vessel, len(offered), pool.keys_admitted, len(not_admitted)))
    return out


def derive_not_admitted(pool, offered):
    """{(env, episode): ((dose, reason), ...)} of the offered keys E4 did not admit (joint ladders).

    A joint pool admits a key at all five doses or at none; a partial ladder,
    or a not-admitted key without an E4 rejection record, is an error.
    """
    reasons = {}
    for rejection in pool.rejections:
        reasons.setdefault(rejection.key, set()).add((int(rejection.dose), str(rejection.reason)))
    out = {}
    for key in offered:
        admitted = [(key, dose) in pool.assignments for dose in range(5)]
        if all(admitted):
            continue
        if any(admitted):
            raise ValueError("D7 pool admits %s at some doses only (joint admission expected)" % (key,))
        found = tuple(sorted(reasons.get(key, ())))
        if not found:
            raise ValueError("not-admitted key %s carries no E4 rejection reason" % (key,))
        out[(int(key.env), int(key.episode))] = found
    return out


def primary_reasons(rejections):
    """The E4 reasons that caused a joint rejection (without the 'joint_ladder_member_rejected' echoes)."""
    return tuple(sorted({reason for _, reason in rejections if reason != "joint_ladder_member_rejected"}))


def per_dose_counts_with_not_admitted(pool):
    """[[offered, admitted, not_admitted] per dose] (E4's [offered, admitted] plus the D16 count)."""
    return [[int(o), int(a), int(o) - int(a)] for o, a in pool.per_dose_counts]


def pool_meta(vessel, seed, num_envs, episodes, tier, admitted, **extra):
    """Provenance block of a pool file (the env checks task, vessel, seed, tier, geometry version, suite)."""
    pool = admitted["pool"]
    not_admitted = admitted["not_admitted"]
    meta = {"task": TASK, "vessel": vessel, "seed": int(seed), "geometry_tier": int(tier),
            "num_envs": int(num_envs), "episodes": int(episodes), "hazard_group": HAZARD_GROUP,
            "contract_version": plans.CONTRACT_VERSION, "geometry_version_digest": plans.GEOMETRY_VERSION_DIGEST,
            "suite": suite_identity(admission_suite(vessel)), "joint": True,
            "keys_offered": pool.keys_offered, "keys_admitted": pool.keys_admitted,
            "keys_not_admitted": len(not_admitted),
            "per_dose_counts": per_dose_counts_with_not_admitted(pool),
            "per_dose_counts_fields": list(PER_DOSE_FIELDS),
            "exposure_rule": D16_RULE,
            "layout_geometry": [[e, k, g] for (e, k), g in sorted(admitted["geometry"].items())],
            "not_admitted": [[e, k, [[d, r] for d, r in reasons]] for (e, k), reasons in sorted(not_admitted.items())],
            "wiring": F3_WIRING_VERSION}
    meta.update(extra)
    return meta


def route_of(layout):
    """((start,) + waypoints) as a float64 (5, 2) array (OrderedPathEpisode route)."""
    return np.asarray((tuple(map(float, layout.start)),) + tuple(layout.waypoints), dtype=np.float64)


_ENVELOPES = {}


def d7_envelope(vehicle):
    """VesselEnvelope exactly as JournalEnvMixin.journal_setup builds it (bridge speed bound)."""
    if vehicle not in _ENVELOPES:
        record = plans.vessel_record(vehicle)
        envelope, _ = _dynamic().envelope_from_spec_v2(vehicles_v2.get_vehicle_v2(vehicle),
                                                       reference_speed_mps=record.cruise_speed_m_s,
                                                       turning_radius_m=record.turning_radius_design_m)
        _ENVELOPES[vehicle] = envelope
    return _ENVELOPES[vehicle]


def d7_task_context(layout, vehicle):
    """D7 TaskContext of one REAL journal layout (``layout.d7_hook`` mapped onto E4 dataclasses).

    The spawn is at rest (initial pose speed 0), the sensor is the plan's
    36-ray fan (30 m x length scale, the hook's ``sensor``), the envelope is
    the one the runtime bridge uses (``d7_envelope``).
    """
    dyn = _dynamic()
    hook = layout.d7_hook
    if hook["task"] != TASK or hook["envelope"]["vessel_id"] != vehicle:
        raise ValueError("d7 hook belongs to another task or vessel")
    x, y, heading = hook["initial_pose"]
    sensor = hook["sensor"]
    return dyn.TaskContext(
        task=TASK, envelope=d7_envelope(vehicle), initial_pose=dyn.Pose(float(x), float(y), float(heading), 0.0),
        bounds=tuple(float(v) for v in hook["bounds"]), goal=dyn.Circle(*map(float, hook["goal"])),
        deadline_s=float(hook["deadline_s"]),
        route=tuple(dyn.Circle(*map(float, c)) for c in hook["route"]),
        route_mandatory=tuple(bool(f) for f in hook["route_mandatory"]),
        static_obstacles=tuple(dyn.Circle(*map(float, c)) for c in hook["static_obstacles"]),
        sensor=dyn.SensorSpec(kind=sensor["kind"], ray_count=int(sensor["ray_count"]),
                              max_range_m=float(sensor["max_range_m"]),
                              source="journal plan 36-ray fan, 30 m x length scale (E5 d7_hook)"),
        site_index=int(hook["site_index"]), base_max=int(hook["base_max"]), base_count=int(hook["base_count"]),
        geometry_id=str(hook["geometry_id"]))


def appearing_admission_inputs(vehicle, tier, seed, env_index, episode_index):
    """G3: (D7 TaskContext, length scale, None) of one key's REAL journal layout (no static radius rule)."""
    layout = journal_layout(vehicle, tier, seed, env_index, episode_index)
    return d7_task_context(layout, vehicle), float(layout.params["length_scale"]), None


def admission_suite(vehicle):
    """The DynamicSuiteConfig the offline admission builds candidates with (dose set per ladder)."""
    return _dynamic().DynamicSuiteConfig(enabled=True, vessel_id=vehicle)


def suite_identity(suite):
    """Suite parameters that shape candidates (everything except the dose switch)."""
    values = asdict(suite)
    for name in ("enabled", "dose", "verify_witness_on_reset"):
        values.pop(name, None)
    return values




# ============================================= D16 not-admitted records ===
def _d7_stamp_module():
    _dynamic()
    from .._shared.conditions.dynamic import stamp
    return stamp


def not_admitted_digest(key, geometry_id, vessel_id, rejections):
    """Dose-independent identity of a not-admitted key (its D11 d7_assignment 'candidate digest').

    Equal at every requested dynamic level, so a not-admitted episode pairs
    with the same key under every other suite and is unexposed for the
    dynamic treatment; never equal to an admitted candidate digest.
    """
    return _dynamic().digest({"exposure": EXPOSURE_NOT_ADMITTED, "task": TASK,
                              "key": [int(key.eval_seed), int(key.env), int(key.episode), str(key.hazard_group)],
                              "geometry_id": str(geometry_id), "vessel_id": str(vessel_id),
                              "rejections": [[int(d), str(r)] for d, r in rejections],
                              "model_version": _d7_stamp_module().MODEL_VERSION})


BEYOND_GRID_REJECTIONS = ((0, BEYOND_GRID_REASON),)


class NotAdmittedAssignment:
    """D16 empty assignment: E4 did not admit the key (or, D24, the key lies beyond the pool's offered grid);
    the row runs D0 (no D7 object)."""

    exposure = EXPOSURE_NOT_ADMITTED

    def __init__(self, key, requested_dose, geometry_id, vessel_id, rejections, pool_digest):
        self.key = key
        self.requested_dose = int(requested_dose)
        self.geometry_id = str(geometry_id)
        self.vessel_id = str(vessel_id)
        self.rejections = tuple((int(d), str(r)) for d, r in rejections)
        self.pool_digest = str(pool_digest)
        self.candidate_digest = not_admitted_digest(key, geometry_id, vessel_id, self.rejections)
        # D24: a key beyond the pool's offered grid carries exactly ((0, "beyond_offered_grid"),)
        self.beyond_offered_grid = self.rejections == BEYOND_GRID_REJECTIONS

    def __repr__(self):
        return "NotAdmittedAssignment(%r, requested_dose=%d, reasons=%r)" % (
            self.key, self.requested_dose, primary_reasons(self.rejections))


def assignment_geometry(assignment):
    """Layout geometry hash an assignment (admitted or not) was built on."""
    if isinstance(assignment, NotAdmittedAssignment):
        return assignment.geometry_id
    return assignment.candidate.geometry_id


def not_admitted_stamp(assignment, env, termination_reason, row, physics_dt_s):
    """Episode stamp of a not-admitted row: E4 stamp envelope layout {payload, sha256} (read_stamp-valid).

    ``payload.exposure`` = ``payload.assignment.exposure`` = ``payload.runtime.exposure`` =
    "not_admitted"; ``assignment.dose`` is the dose that RAN (0, no intruder),
    ``assignment.requested_dose`` the suite level, ``assignment.rejections`` /
    ``rejection_reasons`` the E4 reasons. The envelope keys stay exactly
    {payload, sha256} so E3 ``verify_dynamic_hazard_stamp`` and F0
    ``pairing.d7_primitive_hash`` / ``validate_pair`` accept it unchanged.
    """
    from .._shared.conditions.dynamic import adapters
    stamp = _d7_stamp_module()
    if termination_reason not in adapters.TERMINATION_REASONS:
        raise ValueError("explicit terminal reason required: " + repr(termination_reason))
    key = assignment.key
    dt = float(physics_dt_s)
    summary = {
        "key": {"eval_seed": key.eval_seed, "env": key.env, "episode": key.episode,
                "hazard_group": key.hazard_group},
        "task": TASK, "dose": 0, "requested_dose": assignment.requested_dose,
        "exposure": EXPOSURE_NOT_ADMITTED, "site": adapters.PROFILES[TASK].carrier, "stage": "",
        "geometry_id": assignment.geometry_id, "vessel_id": assignment.vessel_id,
        "intruders": [], "perceivable": False, "sensor": None, "sensor_source": "none",
        "candidate_digest": assignment.candidate_digest, "witness_digest": None,
        "pool_digest": assignment.pool_digest,
        "rejections": [[d, r] for d, r in assignment.rejections],
        "rejection_reasons": list(primary_reasons(assignment.rejections)),
    }
    last_tick = int(row["last_tick"])
    runtime = {
        "assigned": False, "itt_count": 0, "exposure": EXPOSURE_NOT_ADMITTED, "env": int(env),
        "termination_reason": termination_reason, "last_time_s": max(0.0, last_tick * dt),
        "first_detection_s": None, "realized_ttc_at_first_detection_s": None,
        "detected_samples": 0, "observer_updates": 0, "min_sampled_surface_clearance_m": None,
        "sampled_contact": False, "non_exposure_reason": EXPOSURE_NOT_ADMITTED,
        "bridge": {"swept_contact_latched": False, "first_swept_contact_interval_start_s": None,
                   "swept_intervals": int(row["swept_intervals"]), "last_tick": last_tick, "physics_dt_s": dt,
                   "swept_contact_semantics": "no D7 object on a not-admitted row (D16)"},
    }
    semantics = dict(stamp.SEMANTICS)
    semantics["not_admitted"] = D16_RULE
    if getattr(assignment, "beyond_offered_grid", False):
        semantics[BEYOND_GRID_REASON] = D24_RULE
    payload = {"schema_version": stamp.SCHEMA_VERSION, "model_version": stamp.MODEL_VERSION,
               "exposure": EXPOSURE_NOT_ADMITTED, "assignment": summary, "runtime": runtime,
               "semantics": semantics}
    return json.loads(stamp.canonical({"payload": payload, "sha256": stamp.digest(payload)}))


def stamp_exposure(envelope):
    """'not_admitted' for a D16 record, 'assigned' for an E4 assignment stamp (digest checked)."""
    value = _d7_stamp_module().read_stamp(envelope)
    exposure = value["payload"].get("exposure", EXPOSURE_ASSIGNED)
    if exposure not in (EXPOSURE_ASSIGNED, EXPOSURE_NOT_ADMITTED):
        raise ValueError("unknown dynamic-hazard exposure %r" % (exposure,))
    if exposure == EXPOSURE_NOT_ADMITTED and (value["payload"]["assignment"].get("exposure") != exposure
                                              or value["payload"]["runtime"].get("exposure") != exposure
                                              or value["payload"]["assignment"]["intruders"]):
        raise ValueError("inconsistent not_admitted dynamic-hazard record")
    return exposure


def offered_grid(offered):
    """(num_envs, episodes) when the offered keys are exactly env < num_envs x episode < episodes, else None."""
    items = {(int(k.env), int(k.episode)) for k in offered}
    if not items or min(min(e, k) for e, k in items) < 0:
        return None
    num_envs = max(e for e, _ in items) + 1
    episodes = max(k for _, k in items) + 1
    if len(items) != num_envs * episodes:
        return None
    return num_envs, episodes


class D16Pool:
    """Admitted ladders plus the explicit not-admitted keys of one E4 AdmissionPool (duck-typed for the mixin).

    ``assignment(key, dose)`` returns the E4 Assignment of an admitted key, a
    ``NotAdmittedAssignment`` for a not-admitted key (never an exception) and,
    CONTEXT_R4 D24, a ``NotAdmittedAssignment`` with the single reason
    ``beyond_offered_grid`` for a key beyond the offered grid (``beyond_grid``:
    the pool's seed and hazard group, env < grid num_envs, episode >= grid
    episodes; built on the key's E5 layout, which needs ``tier``). Any other
    key outside the offered grid (another seed or hazard group, an env outside
    the grid's range, a dose outside 0..4) raises KeyError (D7 B02 preflight).
    Without ``tier`` or without a rectangular offered grid the D24 path is
    off and every key outside the grid raises KeyError.
    """

    def __init__(self, pool, offered, geometry, not_admitted, vessel, tier=None):
        self.pool, self.offered, self.vessel = pool, tuple(offered), str(vessel)
        self.geometry_tier = None if tier is None else int(tier)
        self.grid = offered_grid(self.offered)
        self.pool_digest = pool.pool_digest
        self.keys_offered = pool.keys_offered
        self.keys_admitted = pool.keys_admitted
        self.joint = pool.joint
        self.assignments = pool.assignments
        self.per_dose_counts = tuple(tuple(x) for x in per_dose_counts_with_not_admitted(pool))
        self._geometry = {(int(e), int(k)): str(g) for (e, k), g in dict(geometry).items()}
        self._not_admitted = {(int(e), int(k)): tuple((int(d), str(r)) for d, r in v)
                              for (e, k), v in dict(not_admitted).items()}
        self.keys_not_admitted = len(self._not_admitted)
        seeds = {k.eval_seed for k in self.offered}
        groups = {k.hazard_group for k in self.offered}
        if len(seeds) != 1 or groups != {HAZARD_GROUP}:
            raise ValueError("a D7 pool holds one evaluation seed of the path-following hazard group")
        self.seed = seeds.pop()
        items = [(int(k.env), int(k.episode)) for k in self.offered]
        if len(set(items)) != len(items) or set(items) != set(self._geometry):
            raise ValueError("D7 pool layout geometry must list every offered key exactly once")
        if derive_not_admitted(pool, self.offered) != self._not_admitted:
            raise ValueError("D7 pool not-admitted list differs from the E4 rejections")
        if self.keys_admitted + self.keys_not_admitted != self.keys_offered:
            raise ValueError("D7 pool key counts do not add up")

    def covers(self, env, episode):
        return (int(env), int(episode)) in self._geometry

    def grid_record(self):
        """{"num_envs": N, "episodes": E} of the offered grid (D24 header field), or None."""
        if self.grid is None:
            return None
        return {"num_envs": int(self.grid[0]), "episodes": int(self.grid[1])}

    def beyond_grid(self, key):
        """D24: the pool's seed and hazard group, env inside the grid, episode >= the grid's episodes per env."""
        if self.grid is None or self.geometry_tier is None:
            return False
        num_envs, episodes = self.grid
        return (key.eval_seed == self.seed and key.hazard_group == HAZARD_GROUP
                and 0 <= int(key.env) < num_envs and int(key.episode) >= episodes)

    def is_admitted(self, env, episode):
        return self.covers(env, episode) and (int(env), int(episode)) not in self._not_admitted

    def not_admitted_keys(self):
        """[[env, episode, [primary E4 reasons]]] (sorted)."""
        return [[e, k, list(primary_reasons(v))] for (e, k), v in sorted(self._not_admitted.items())]

    def geometry(self, env, episode):
        """Layout geometry hash the pool recorded for a key; ValueError outside the grid."""
        try:
            return self._geometry[(int(env), int(episode))]
        except KeyError:
            raise ValueError("no admitted D7 pool record (admitted or not_admitted) for env %d episode %d in "
                             "the %s pool of seed %d (the grid is its offered keys; D7 B02 preflight error)"
                             % (int(env), int(episode), self.vessel, self.seed)) from None

    def assignment(self, key, dose):
        dose = int(dose)
        if not 0 <= dose <= 4:
            raise KeyError((key, dose))
        item = (int(key.env), int(key.episode))
        if item in self._not_admitted and key.eval_seed == self.seed and key.hazard_group == HAZARD_GROUP:
            return NotAdmittedAssignment(key, dose, self._geometry[item], self.vessel, self._not_admitted[item],
                                         self.pool_digest)
        if self.beyond_grid(key):
            # D24: the pool has no record of the key; its identity is the key's E5 layout (a pure
            # function of the key), so the candidate digest is dose-independent.
            layout = journal_layout(self.vessel, self.geometry_tier, self.seed, item[0], item[1])
            return NotAdmittedAssignment(key, dose, layout.geometry_hash, self.vessel, BEYOND_GRID_REJECTIONS,
                                         self.pool_digest)
        return self.pool.assignment(key, dose)


# ==================================================== D7 pool file codec ===
# Container (xz): MAGIC, 16 hex digits = JSON length, LF, canonical JSON (ASCII),
# then the binary block of the witness trajectories: for every tuple of E4
# TimedStates, 7 float64 columns (time, x, y, yaw, speed, accel, yaw rate),
# little endian, each column byte-shuffled (byte k of every value together);
# the JSON holds ["S", offset, count] in its place. Exact round trip (IEEE
# bits); the loader consumes the block strictly in order and re-derives every
# witness/candidate digest and the pool digest (fail closed).
POOL_MAGIC = b"USVB-F3-D7POOL/2\n"
_STATE_COLUMNS = 7


def _state_rows(value):
    """7-tuples of exact floats for a tuple of TimedState, or None if it must stay generic JSON."""
    if not isinstance(value, tuple) or not value:
        return None
    rows = []
    for item in value:
        if type(item).__name__ != "TimedState" or not is_dataclass(item):
            return None
        if [f.name for f in dataclass_fields(item) if f.init] != ["time_s", "pose", "accel_mps2", "yaw_rate_rps"]:
            return None
        pose = item.pose
        if type(pose).__name__ != "Pose" or [f.name for f in dataclass_fields(pose) if f.init] != \
                ["x", "y", "yaw", "speed"]:
            return None
        row = (item.time_s, pose.x, pose.y, pose.yaw, pose.speed, item.accel_mps2, item.yaw_rate_rps)
        if any(type(v) is not float for v in row):
            return None
        rows.append(row)
    return rows


def _pack_columns(rows):
    n = len(rows)
    out = []
    for column in zip(*rows):
        raw = struct.pack("<%dd" % n, *column)
        out.append(b"".join(raw[k::8] for k in range(8)))
    return b"".join(out)


def _unpack_columns(block, n):
    columns = []
    width = 8 * n
    for c in range(_STATE_COLUMNS):
        segment = block[c * width:(c + 1) * width]
        raw = bytearray(width)
        for k in range(8):
            raw[k::8] = segment[k * n:(k + 1) * n]
        columns.append(struct.unpack("<%dd" % n, bytes(raw)))
    return list(zip(*columns))


def encode_value(value, blob=None):
    """Tagged, exact JSON form: dataclasses, tuples, lists, dicts and non-finite floats survive.

    With ``blob`` (a bytearray), TimedState trajectories go to the binary block.
    """
    if blob is not None:
        rows = _state_rows(value)
        if rows is not None:
            offset = len(blob)
            blob += _pack_columns(rows)
            return ["S", offset, len(rows)]
    if is_dataclass(value) and not isinstance(value, type):
        # init fields only: derived (init=False) fields are recomputed by __post_init__ on decode, and the
        # loader re-derives every candidate/witness digest (fail closed if any value changed)
        return ["D", type(value).__name__,
                [encode_value(getattr(value, f.name), blob) for f in dataclass_fields(value) if f.init]]
    if isinstance(value, tuple):
        return ["T", [encode_value(v, blob) for v in value]]
    if isinstance(value, list):
        return ["L", [encode_value(v, blob) for v in value]]
    if isinstance(value, dict):
        return ["M", [[encode_value(k, blob), encode_value(v, blob)] for k, v in value.items()]]
    if isinstance(value, float) and not math.isfinite(value):
        return ["F", repr(value)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("cannot encode %r in a D7 pool file" % (type(value).__name__,))


class _BlobReader:
    def __init__(self, blob):
        self.blob = memoryview(blob)
        self.cursor = 0

    def states(self, offset, count, classes):
        if type(offset) is not int or type(count) is not int or count <= 0 or offset != self.cursor:
            raise ValueError("D7 pool trajectory block out of order")
        end = offset + _STATE_COLUMNS * 8 * count
        if end > len(self.blob):
            raise ValueError("D7 pool trajectory block truncated")
        rows = _unpack_columns(self.blob[offset:end], count)
        self.cursor = end
        state, pose = classes["TimedState"], classes["Pose"]
        return tuple(state(r[0], pose(r[1], r[2], r[3], r[4]), r[5], r[6]) for r in rows)


def decode_value(item, classes, reader=None):
    if not isinstance(item, list):
        return item
    tag = item[0]
    if tag == "D":
        cls = classes.get(item[1])
        if cls is None:
            raise ValueError("unknown class %r in a D7 pool file (only E4 dataclasses are accepted)" % (item[1],))
        return cls(*[decode_value(v, classes, reader) for v in item[2]])
    if tag == "T":
        return tuple(decode_value(v, classes, reader) for v in item[1])
    if tag == "L":
        return [decode_value(v, classes, reader) for v in item[1]]
    if tag == "M":
        return {decode_value(k, classes, reader): decode_value(v, classes, reader) for k, v in item[1]}
    if tag == "F":
        return float(item[1])
    if tag == "S" and reader is not None and len(item) == 3:
        return reader.states(item[1], item[2], classes)
    raise ValueError("malformed D7 pool file item")


def join_pool_container(text, blob):
    data = text.encode("ascii")
    return POOL_MAGIC + (b"%016x\n" % len(data)) + data + bytes(blob)


def split_pool_container(data):
    """(JSON text, binary block) of a decompressed pool container."""
    if not data.startswith(POOL_MAGIC):
        raise ValueError("not a %s pool container" % POOL_SCHEMA)
    head = len(POOL_MAGIC)
    try:
        length = int(data[head:head + 16].decode("ascii"), 16)
    except ValueError:
        raise ValueError("malformed D7 pool container header") from None
    if data[head + 16:head + 17] != b"\n" or head + 17 + length > len(data):
        raise ValueError("malformed D7 pool container header")
    start = head + 17
    return data[start:start + length].decode("ascii"), data[start + length:]


def _pool_classes():
    _dynamic()
    from .._shared.conditions.dynamic import adapters, capacity, feasibility, model, native, visibility
    classes = (model.EpisodeKey, model.Footprint, model.VesselEnvelope, model.Pose, model.TimedState,
               model.Circle, model.Intruder, model.Encounter, feasibility.WitnessRequest,
               feasibility.WitnessResult, capacity.CapacityPlan, visibility.SensorSpec,
               visibility.RevealEvidence, native.NativeMotion, adapters.Candidate, adapters.Assignment,
               adapters.Rejection, adapters.AdmissionPool)
    return {cls.__name__: cls for cls in classes}


def _pool_manifest_digest(offered, pool):
    """Recompute E4 adapters._assemble's pool digest from the stored pool (fail closed)."""
    dyn = _dynamic()
    from .._shared.conditions.dynamic.adapters import _witness_digest
    accepted = sorted([[k.eval_seed, k.env, k.episode, k.hazard_group, d, dyn.digest(a.candidate),
                        _witness_digest(a.witness)] for (k, d), a in pool.assignments.items()])
    return dyn.digest({"offered": list(offered), "accepted": accepted, "rejections": list(pool.rejections),
                       "joint": pool.joint})


def pool_file_name(vehicle, seed):
    return "d7_pool_%s_%s_seed%d.d7pool.xz" % (TASK, vehicle, int(seed))


def write_pool_file(path, pool, offered, meta):
    """Write the pool container (xz) plus ``<path>.sha256`` (SHA-256 of the decompressed container)."""
    if type(pool).__name__ != "AdmissionPool":
        raise TypeError("write_pool_file needs an E4 AdmissionPool")
    if _pool_manifest_digest(offered, pool) != pool.pool_digest:
        raise ValueError("pool digest does not match its offered keys (pass the admission key order)")
    blob = bytearray()
    offered_item = encode_value(tuple(offered), blob)
    pool_item = encode_value(pool, blob)
    document = {"schema": POOL_SCHEMA, "meta": meta, "offered": offered_item, "pool": pool_item}
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = join_pool_container(text, blob)
    digest = hashlib.sha256(data).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64,
                                   preset=9 | lzma.PRESET_EXTREME))
    Path(str(path) + ".sha256").write_bytes((digest + "  " + path.name + "\n").encode("ascii"))
    return digest


class LoadedD7Pool(D16Pool):
    """A D16 pool read back from a pool file, plus its provenance."""

    def __init__(self, pool, offered, meta, sha256, path):
        meta = dict(meta)
        geometry = {(int(e), int(k)): g for e, k, g in meta.get("layout_geometry", ())}
        not_admitted = {(int(e), int(k)): tuple((int(d), str(r)) for d, r in reasons)
                        for e, k, reasons in meta.get("not_admitted", ())}
        super().__init__(pool, offered, geometry, not_admitted, meta.get("vessel"), tier=meta.get("geometry_tier"))
        if self.grid is None or self.grid != (meta.get("num_envs"), meta.get("episodes")):
            raise ValueError("D7 pool offered keys are not the %s x %s grid its meta declares (D24 needs the "
                             "offered grid)" % (meta.get("num_envs"), meta.get("episodes")))
        self.meta = meta
        self.sha256, self.path = sha256, str(path)
        if [list(x) for x in self.per_dose_counts] != meta.get("per_dose_counts") \
                or meta.get("keys_not_admitted") != self.keys_not_admitted:
            raise ValueError("D7 pool meta counts differ from the stored pool")


class OnlineExtendedPool:
    """Opt-in extension of a committed pool for keys OUTSIDE its grid (``cfg.journal_d7_online_admission``).

    The per-env-quota evaluators keep stepping every env until the slowest one
    has its quota (scripts/eval_v6_frozen.py:240-244), so rows past their quota
    autoreset into episodes the committed grid does not contain. With the
    opt-in, such a key is admitted at its reset by the SAME joint procedure on
    its E5 layout (``admit_path_following_keys``; D16: admitted ladder or
    not_admitted, never a redraw) under its OWN pool digest, so its stamp can
    never be mistaken for a committed-grid record; every such key is listed in
    the env's dynamic_hazard_header["online_admitted_keys"]. Keys inside the
    committed grid always come from the committed file. Default off: a missing
    key then stays a preflight error (D7 B02).
    """

    def __init__(self, committed, vessel, seed, tier, record):
        self.committed = committed
        self.vessel, self.seed, self.tier = vessel, int(seed), int(tier)
        self.record = record
        self.online = {}
        for name in ("pool_digest", "per_dose_counts", "keys_offered", "keys_admitted", "keys_not_admitted",
                     "joint", "meta", "path", "sha256"):
            setattr(self, name, getattr(committed, name))

    def not_admitted_keys(self):
        return self.committed.not_admitted_keys()

    def grid_record(self):
        return self.committed.grid_record()

    def beyond_grid(self, key):
        """Never: with the opt-in, keys beyond the committed grid are admitted online (CONTEXT_R4 D24 (e))."""
        return False

    def _online(self, env, episode):
        item = (int(env), int(episode))
        if item not in self.online:
            admitted = admit_path_following_keys([(self.vessel, item[0], item[1])], self.seed, tier=self.tier)
            admitted = admitted[self.vessel]
            pool = D16Pool(admitted["pool"], admitted["offered"], admitted["geometry"], admitted["not_admitted"],
                           self.vessel)
            self.online[item] = pool
            self.record.append({"env": item[0], "episode": item[1], "pool_digest": pool.pool_digest,
                                "exposure": EXPOSURE_ASSIGNED if pool.is_admitted(*item) else EXPOSURE_NOT_ADMITTED})
        return self.online[item]

    def assignment(self, key, dose):
        if self.committed.covers(key.env, key.episode) or key.eval_seed != self.seed \
                or key.hazard_group != HAZARD_GROUP:
            return self.committed.assignment(key, dose)
        return self._online(key.env, key.episode).assignment(key, dose)

    def geometry(self, env, episode):
        if self.committed.covers(env, episode):
            return self.committed.geometry(env, episode)
        return self._online(env, episode).geometry(env, episode)


_POOL_CACHE = {}


def read_pool_file(path, expected_sha256=None):
    """Load and verify a pool file (fail closed on any mismatch)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("D7 admission pool %s is missing (run tools_f3/admit_d7_pool.py; D7 B02/B05)"
                                % path)
    data = lzma.decompress(path.read_bytes(), format=lzma.FORMAT_XZ)
    sha = hashlib.sha256(data).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    declared = expected_sha256
    if declared is None:
        if not sidecar.is_file():
            raise FileNotFoundError("D7 pool digest file %s is missing" % sidecar)
        declared = sidecar.read_text(encoding="ascii").split()[0]
    if sha != declared:
        raise ValueError("D7 pool %s SHA-256 %s differs from the declared %s" % (path, sha, declared))
    cache_key = (str(path.resolve()), sha)
    if cache_key in _POOL_CACHE:
        return _POOL_CACHE[cache_key]
    text, blob = split_pool_container(data)
    document = json.loads(text)
    if document.get("schema") != POOL_SCHEMA:
        raise ValueError("unsupported D7 pool schema %r" % (document.get("schema"),))
    classes = _pool_classes()
    reader = _BlobReader(blob)
    offered = decode_value(document["offered"], classes, reader)
    pool = decode_value(document["pool"], classes, reader)
    if reader.cursor != len(blob):
        raise ValueError("D7 pool trajectory block has trailing bytes")
    if type(pool).__name__ != "AdmissionPool":
        raise ValueError("pool file does not hold an AdmissionPool")
    if _pool_manifest_digest(offered, pool) != pool.pool_digest:
        raise ValueError("D7 pool digest does not match its assignments (tampered or truncated file)")
    for assignment in pool.assignments.values():
        if assignment.pool_digest != pool.pool_digest:
            raise ValueError("assignment from another pool")
    loaded = LoadedD7Pool(pool, offered, document["meta"], sha, path)
    _POOL_CACHE[cache_key] = loaded
    return loaded


# --------------------------------------------------- pinned pool manifest --
POOL_MANIFEST_NAME = "POOL_MANIFEST.json"
POOL_MANIFEST_SCHEMA = "usvbench.f3.d7_pool_manifest/1"
POOL_DIR_ENV = "USVBENCH_D7_POOL_DIR"


def pool_manifest_path():
    return POOL_DIR / POOL_MANIFEST_NAME


def read_pool_manifest():
    """(manifest dict, SHA-256 of its bytes) of the pinned pool manifest in the family folder."""
    path = pool_manifest_path()
    if not path.is_file():
        raise FileNotFoundError("pinned D7 pool manifest %s is missing (D19)" % path)
    data = path.read_bytes()
    manifest = json.loads(data.decode("ascii"))
    if manifest.get("schema") != POOL_MANIFEST_SCHEMA or manifest.get("task") != TASK:
        raise ValueError("unsupported pinned D7 pool manifest %s" % path)
    return manifest, hashlib.sha256(data).hexdigest()


def resolve_pool_path(cfg):
    """(path, source): cfg.journal_d7_pool_path (absolute, or relative to the family folder; 'cfg'),
    else $USVBENCH_D7_POOL_DIR/<default name> ('env'), else the family folder ('pinned')."""
    declared = getattr(cfg, "journal_d7_pool_path", None)
    if declared:
        path = Path(declared)
        return (path if path.is_absolute() else POOL_DIR / path), "cfg"
    name = pool_file_name(cfg.vehicle, cfg.seed)
    folder = os.environ.get(POOL_DIR_ENV)
    if folder:
        return Path(folder) / name, "env"
    return POOL_DIR / name, "pinned"


def load_journal_d7_pool(cfg):
    """Preflight (D7 B02): the pool for this cfg's vessel and seed, verified; else raise.

    A pool found by its default name (family folder or $USVBENCH_D7_POOL_DIR)
    must be pinned in POOL_MANIFEST.json (container SHA-256 and pool digest);
    a cfg path is verified against ``cfg.journal_d7_pool_sha256`` or its sidecar.
    """
    path, source = resolve_pool_path(cfg)
    expected = getattr(cfg, "journal_d7_pool_sha256", None)
    pinned = None
    if source != "cfg":
        manifest, _ = read_pool_manifest()
        pinned = manifest["pools"].get(path.name)
        if pinned is None:
            raise ValueError("D7 pool %s is not pinned in %s (re-admit with tools_f3/admit_d7_pool.py; D19)"
                             % (path.name, pool_manifest_path()))
        if expected is not None and expected != pinned["container_sha256"]:
            raise ValueError("cfg.journal_d7_pool_sha256 differs from the pinned manifest")
        expected = pinned["container_sha256"]
    pool = read_pool_file(path, expected)
    if pinned is not None and pool.pool_digest != pinned["pool_digest"]:
        raise ValueError("D7 pool digest %s differs from the pinned %s" % (pool.pool_digest, pinned["pool_digest"]))
    meta = pool.meta
    if meta.get("task") != TASK or meta.get("vessel") != cfg.vehicle:
        raise ValueError("D7 pool was admitted for %s/%s, not %s/%s"
                         % (meta.get("task"), meta.get("vessel"), TASK, cfg.vehicle))
    if int(meta.get("seed", -1)) != int(cfg.seed):
        raise ValueError("D7 pool was admitted for evaluation seed %s, not %s (D7 B02: a missing key is a "
                         "preflight error)" % (meta.get("seed"), cfg.seed))
    if int(meta.get("geometry_tier", -1)) != int(cfg.journal_geometry_tier):
        raise ValueError("D7 pool geometry tier differs from the cfg")
    if meta.get("geometry_version_digest") != plans.GEOMETRY_VERSION_DIGEST:
        raise ValueError("D7 pool was admitted on another journal geometry version")
    return pool, source


# ================================================= D16 bridge (rows) =======
class D16HazardBridge:
    """E4 ``HazardBridge`` plus D16 not-admitted rows (duck-typed for ``JournalEnvMixin``).

    Rows whose assignment is a ``NotAdmittedAssignment`` never reach the E4
    adapter: they carry the empty assignment, return no D7 sample, never
    latch contact, keep the same tick discipline as E4 rows, and ``finish``
    returns ``not_admitted_stamp``. Admitted rows are passed through unchanged.
    """

    def __init__(self, inner, on_finish=None):
        self.inner = inner
        self.adapter, self.cfg, self.dt = inner.adapter, inner.cfg, inner.dt
        self._empty = {}
        self._kind = {}
        self._on_finish = on_finish

    @property
    def rows(self):
        return {env: (self._empty[env] if kind == EXPOSURE_NOT_ADMITTED else self.inner.rows[env])
                for env, kind in self._kind.items()}

    def exposure(self, env):
        return self._kind[int(env)]

    def reset_rows(self, env_ids, assignments):
        ids = tuple(int(i) for i in env_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("reset ids must be unique rows")
        rows = self.rows
        for env in ids:
            if env in rows and not rows[env]["finished"]:
                raise ValueError("finish and stamp every assigned episode before reset")
        empty = [env for env in ids if isinstance(assignments[env], NotAdmittedAssignment)]
        full = [env for env in ids if env not in empty]
        for env in empty:
            item = assignments[env]
            if item.key.env != env or item.key.hazard_group != HAZARD_GROUP:
                raise ValueError("wrong task or environment not-admitted assignment")
        if full:
            self.inner.reset_rows(full, {env: assignments[env] for env in full})
        for env in full:
            self._kind[env] = EXPOSURE_ASSIGNED
        for env in empty:
            self._empty[env] = {"assignment": assignments[env], "last_tick": -1, "cache": (), "contact": False,
                                "first_contact_s": None, "swept_intervals": 0, "finished": False, "stamp": None}
            self._kind[env] = EXPOSURE_NOT_ADMITTED

    def step(self, ticks, poses):
        Pose = _dynamic().Pose
        full = {env: tick for env, tick in ticks.items() if self._kind.get(env) != EXPOSURE_NOT_ADMITTED}
        out = self.inner.step(full, poses) if full else {}
        for env, tick in ticks.items():
            if self._kind.get(env) != EXPOSURE_NOT_ADMITTED:
                continue
            row = self._empty[env]
            if row["finished"]:
                raise ValueError("row finished; reset it before stepping")
            if type(tick) is not int or tick < 0:
                raise ValueError("tick must be a nonnegative integer")
            if tick < row["last_tick"]:
                raise ValueError("physics tick went backwards without a reset")
            if tick != row["last_tick"]:
                if not isinstance(poses[env], Pose):
                    raise TypeError("pose must be a Pose")
                if row["last_tick"] >= 0:
                    row["swept_intervals"] += 1
                row["last_tick"] = tick
            out[env] = ()
        return out

    def contact(self, env):
        if self._kind.get(env) == EXPOSURE_NOT_ADMITTED:
            return False
        return self.inner.contact(env)

    def finish(self, env, termination_reason):
        if self._kind.get(env) != EXPOSURE_NOT_ADMITTED:
            if self.inner.rows[env]["finished"]:
                return self.inner.finish(env, termination_reason)
            stamp = self.inner.finish(env, termination_reason)
            exposure = EXPOSURE_ASSIGNED
        else:
            row = self._empty[env]
            if row["finished"]:
                return row["stamp"]
            stamp = not_admitted_stamp(row["assignment"], env, termination_reason, row, self.dt)
            row["stamp"], row["finished"] = stamp, True
            exposure = EXPOSURE_NOT_ADMITTED
        if self._on_finish is not None:
            self._on_finish(exposure, termination_reason)
        return stamp


# ======================================================= ray block (D14) ===
def ray_angles(ray_count, device=None):
    """Body-relative fan angles: ray k at k * 2pi / count from the bow, counter-clockwise."""
    return torch.arange(int(ray_count), dtype=F64, device=device) * (2.0 * math.pi / int(ray_count))


def point_return_rays(com_xy, yaw, centers, radii, visible, ray_count, max_range_m):
    """36-ray block under the E4 point-return proxy (visibility.py:31-34).

    For every VISIBLE object the centre ray (observer COM -> object centre)
    returns the nearest surface point at range max(0, d - r); the return is
    assigned to the fan ray nearest that bearing (a tie goes to the higher index);
    every other ray reads ``max_range_m``. Inputs: com_xy [N,2], yaw [N],
    centers [N,S,2], radii [N,S], visible [N,S] bool; all env-local. Returns
    float64 [N, ray_count] physical ranges (clamped to [0, max_range_m]).
    """
    com = com_xy.to(F64)
    n = com.shape[0]
    count = int(ray_count)
    out = torch.full((n, count), float(max_range_m), dtype=F64, device=com.device)
    if centers.shape[1] == 0:
        return out
    delta = centers.to(F64) - com.unsqueeze(1)
    distance = torch.sqrt((delta * delta).sum(-1))
    edge = (distance - radii.to(F64)).clamp(min=0.0)
    bearing = torch.atan2(delta[..., 1], delta[..., 0]) - yaw.to(F64).unsqueeze(1)
    step = 2.0 * math.pi / count
    index = torch.remainder(torch.floor(torch.remainder(bearing, 2.0 * math.pi) / step + 0.5), count).long()
    ranges = torch.where(visible, edge.clamp(max=float(max_range_m)), torch.full_like(edge, float(max_range_m)))
    for slot in range(centers.shape[1]):
        out.scatter_reduce_(1, index[:, slot:slot + 1], ranges[:, slot:slot + 1], reduce="amin")
    return out


# ===================================================== small quaternion ====
def yaw_from_quat(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_apply_inverse(quat, vec):
    """Rotate world vectors into the body frame (unit wxyz quaternions)."""
    w = quat[..., :1]
    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec - w * t + torch.cross(xyz, t, dim=-1)


def quat_apply(quat, vec):
    w = quat[..., :1]
    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def quat_from_yaw(yaw):
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), -1)


# ================================================== family scenario hashes ==
def family_hash_content(layout, vehicle, tier, spawn_com_height_m):
    """Exogenous content of the two path-following primitives (pairing.FAMILY_PRIMITIVES)."""
    route = route_of(layout)
    return {
        "route": {"task": TASK, "vessel": vehicle, "geometry_tier": int(tier),
                  "geometry_hash": layout.geometry_hash,
                  "route_xy_m": [[float(x), float(y)] for x, y in route.tolist()],
                  "gate_radius_m": float(layout.params["gate_radius_m"]),
                  "deadline_s": float(layout.deadline_s), "deadline_steps": int(layout.deadline_steps),
                  "bounds_m": [float(v) for v in layout.bounds_m],
                  "d7_site_index": int(layout.d7_hook["site_index"])},
        "spawn_pose": {"com_xy_m": [float(v) for v in layout.start],
                       "heading_rad": float(layout.start_heading_rad),
                       "com_height_above_water_m": float(spawn_com_height_m), "speed_mps": 0.0},
    }


def family_hashes(content):
    return {name: pairing.primitive_hash(name, value) for name, value in sorted(content.items())}


# ============================================================ the mixin ====
class PathFollowingJournalMixin(JournalEnvMixin):
    """Journal wiring of ``PathFollowingEnv`` (see the module docstring).

    Call order = ``journal_env_mixin`` EXACT CALL ORDER. Rewards, termination,
    metrics and certificates use truth; the observation uses the D5
    own-position estimate, the true heading and the degraded ray block; the
    observation has NO velocity channel (F0 contract journal_nav7_ray36).
    """

    journal_family = FAMILY
    # G3 appearing obstacle: wired on the path-following journal task (phase 1)
    journal_appearing_tasks = (TASK,)

    # ---------------------------------------------------------- construction --
    def _journal_construct(self, cfg, render_mode=None, **kwargs):
        if getattr(cfg, "journal_task", None) != TASK:
            raise ValueError("PathFollowingEnv carries only the 'path_following' journal task, got %r"
                             % (getattr(cfg, "journal_task", None),))
        seed = getattr(cfg, "seed", None)
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("journal cfgs must carry an integer evaluation seed (cfg.seed; CONTEXT_R4 D7)")
        if getattr(getattr(cfg, "sea_state", None), "enable", False):
            raise ValueError("journal path following never runs the legacy sea state")
        assert_hypot_parity()
        self.physics_cfg = cfg.underwater_physics_cfg
        self.vehicle_spec = _JournalAssetSpec(cfg.vehicle)
        training_pool = getattr(cfg, "journal_d7_pool", None)    # G19b: training-only online-admission pool
        if getattr(training_pool, "TRAINING_ADAPTER", False) is True:  # (tasks/path_following/training_pool.py)
            d7_pool, pool_source = training_pool, "training_online"
        else:
            d7_pool, pool_source = load_journal_d7_pool(cfg)          # D7 B02 preflight, before the scene
        self._journal_super_init(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device
        self.control_step_s = self.cfg.sim.dt * self.cfg.decimation
        plan = cfg.journal_plan
        self._journal_plan = plan
        self._journal_tier = int(cfg.journal_geometry_tier)
        self._journal_limits = contracts.ValidityLimits(**plan.validity_limits)
        self._journal_nominal_height = float(plan.nominal_com_height_m)
        params = plan.params
        self._journal_ray_count = int(params["ray_sensor_ray_count"])
        self._journal_ray_range = float(params["ray_sensor_max_range_m"])
        self._journal_length_scale = float(params["length_scale"])
        self._journal_distance_norm = 20.0 * self._journal_length_scale
        self._journal_water_z = float(self.physics_cfg.water_surface_z)
        gates = 4
        self.waypoints = torch.zeros((n, gates, 2), device=device)
        self.segment_lengths = torch.zeros((n, gates), device=device)
        self.route_length = torch.zeros(n, device=device)
        self.gates_passed = torch.zeros(n, dtype=torch.long, device=device)
        self.path_length = torch.zeros(n, device=device)
        self.xte_rms = torch.zeros(n, device=device)
        self.episode_success = torch.zeros(n, dtype=torch.bool, device=device)
        self.time_to_success = torch.full((n,), torch.nan, device=device)
        self.episode_path_length = torch.zeros(n, device=device)
        self.episode_gates_passed = torch.zeros(n, dtype=torch.long, device=device)
        self.episode_xte_rms = torch.zeros(n, device=device)
        self.episode_route_length = torch.zeros(n, device=device)
        self.episode_success_prefix_path_length_m = torch.full((n,), torch.nan, device=device)
        self.episode_full_path_length_m = torch.zeros(n, device=device)
        self.episode_xte_integral_m_s = torch.zeros(n, device=device)
        self.episode_deadline_s = torch.zeros(n, device=device)
        self.episode_termination_reason = torch.zeros(n, dtype=torch.long, device=device)
        self.journal_episode_results = [None] * n
        self.actions = torch.zeros((n, 2), device=device)
        self._journal_oracle = OrderedPathTensorOracle(n, gates, device)
        self._journal_bounds = torch.zeros((n, 4), dtype=F64, device=device)
        self._journal_collision = torch.zeros(n, dtype=torch.bool, device=device)
        self._journal_layouts = [None] * n
        self._journal_d7_exposure = [None] * n
        self._journal_family_content = [None] * n
        self._journal_family_hashes = [None] * n
        self._journal_completed_family = [None] * n
        self._journal_prev_distance = torch.zeros(n, device=device)
        self._journal_pre_distance = torch.zeros(n, device=device)
        self._journal_gate_event = torch.zeros(n, dtype=torch.bool, device=device)
        self._journal_success_event = torch.zeros(n, dtype=torch.bool, device=device)
        slots = int(plans.STATIC_CAPACITY[TASK]) + int(plans.D7_RESERVED_SLOTS[TASK])    # B04: 0 + 1
        self._d7_slots = slots
        self._d7_centers_local = torch.zeros((n, slots, 2), dtype=F64, device=device)
        self._d7_radii = torch.zeros((n, slots), dtype=F64, device=device)
        self._d7_physical = torch.zeros((n, slots), dtype=torch.bool, device=device)
        self._d7_visible = torch.zeros((n, slots), dtype=torch.bool, device=device)
        self._scenario = make_scenario_rng(self.cfg, n, device)
        self._scenario_params = [{} for _ in range(n)]
        self._scenario_hashes = [{} for _ in range(n)]
        self.episode_scenario = [{} for _ in range(n)]
        self._sea = None
        self.journal_setup(d7_pool=d7_pool)
        if self._appearing_bridge is not None:
            # G3: the appearing cylinder's own storage (slot STATIC + D7 = 1, CapacityPlan.appearing_slots);
            # the D7 slot storage above is untouched, so the D7 publication never overwrites it.
            self._appearing_centers_local = torch.zeros((n, 1, 2), dtype=F64, device=device)
            self._appearing_radii = torch.zeros((n, 1), dtype=F64, device=device)
            self._appearing_physical = torch.zeros((n, 1), dtype=torch.bool, device=device)
            self._appearing_visible = torch.zeros((n, 1), dtype=torch.bool, device=device)
            self.appearing_hazard_header["collision_model"] = (
                "path following has no static obstacles: the appearing swept-contact latch (D7 swept guard) is "
                "OR'ed into the contract collision ledger like the D7 contact; rays use the point-return proxy "
                "over the visible D7 and appearing slots")
        suite = suite_identity(self._d7_suite)
        if suite != d7_pool.meta.get("suite"):
            raise ValueError("the runtime D7 suite parameters differ from the ones the pool was admitted "
                             "with (only the dose may change): %r vs %r" % (suite, d7_pool.meta.get("suite")))
        profile = self.journal_vehicle.hydro_profile
        self._journal_spawn_com_z = float(profile.hydro.equilibrium_com_z_m)
        self._journal_d7_pool_record = {"path": d7_pool.path, "sha256": d7_pool.sha256,
                                        "pool_digest": d7_pool.pool_digest, "source": pool_source}
        header = self.dynamic_hazard_header
        header["pool_file"] = dict(self._journal_d7_pool_record)
        _, manifest_sha = read_pool_manifest()
        header["pool_manifest"] = "tasks/path_following/journal_d7/" + POOL_MANIFEST_NAME
        header["pool_manifest_sha256"] = manifest_sha
        header["pool_pinned"] = pool_source not in ("cfg", "training_online")
        # D16: header per-dose counts carry the not-admitted keys ([offered, admitted, not_admitted])
        header["per_dose_counts_fields"] = list(PER_DOSE_FIELDS)
        header["keys_not_admitted"] = d7_pool.keys_not_admitted
        header["not_admitted_keys"] = d7_pool.not_admitted_keys()
        header["exposure_rule"] = D16_RULE
        self._journal_finished_by_exposure = {EXPOSURE_ASSIGNED: 0, EXPOSURE_NOT_ADMITTED: 0, "aborted": 0}
        header["finished_episodes_by_exposure"] = dict(self._journal_finished_by_exposure)
        self._d7_bridge = D16HazardBridge(self._d7_bridge, self._journal_count_finished)
        online = getattr(cfg, "journal_d7_online_admission", False)
        if online not in (True, False):
            raise ValueError("cfg.journal_d7_online_admission must be a bool")
        grid = d7_pool.grid_record()
        if grid is None:
            raise ValueError("the D7 pool has no rectangular offered grid (CONTEXT_R4 D24)")
        header["grid"] = grid                      # D24 (c): the pool's offered grid (per-dose counts unchanged)
        header["beyond_grid_rule"] = ONLINE_BEYOND_GRID_RULE if online else D24_RULE
        self.dynamic_hazard_header["online_admission"] = bool(online)
        self.dynamic_hazard_header["online_admitted_keys"] = []
        if online:
            self._d7_pool = OnlineExtendedPool(d7_pool, cfg.vehicle, cfg.seed, self._journal_tier,
                                               self.dynamic_hazard_header["online_admitted_keys"])
        self._pre_physics_step = self._journal_pre_physics_step
        self._apply_action = self._journal_apply_action
        self._get_observations = self._journal_get_observations
        self._get_rewards = self._journal_get_rewards
        self._get_dones = self._journal_get_dones
        self._reset_idx = self._journal_reset_idx

    def _journal_super_init(self, cfg, render_mode=None, **kwargs):
        # The DirectRLEnv constructor: the class after the two mixins in the MRO.
        super(JournalEnvMixin, self).__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------- truth --
    def _journal_truth(self, robot_data=None, env_origins=None):
        """Env-local COM xyz, quaternion, linear and angular velocity (float64, root-link = COM)."""
        data = self.robot.data if robot_data is None else robot_data
        origins = self.scene.env_origins if env_origins is None else env_origins
        pose = data.root_link_pose_w
        vel = data.root_com_vel_w
        com = (pose[:, :3] - origins[:, :3]).to(F64)
        return com, pose[:, 3:7].to(F64), vel[:, :3].to(F64), vel[:, 3:6].to(F64)

    def _journal_count_finished(self, exposure, termination_reason):
        key = "aborted" if termination_reason == "aborted" else exposure
        self._journal_finished_by_exposure[key] += 1
        self.dynamic_hazard_header["finished_episodes_by_exposure"] = dict(self._journal_finished_by_exposure)

    # ------------------------------------------------------------ hooks --
    def _journal_observe_physics(self, robot_data, env_origins, water_level_local):
        com, quat, lin, ang = self._journal_truth(robot_data, env_origins)
        self._journal_oracle.observe_physics(com, quat, lin, ang, water_level_local,
                                             self._journal_nominal_height)

    def _journal_family_scenario_hashes(self, env_id):
        hashes = self._journal_family_hashes[int(env_id)]
        if hashes is None:
            raise RuntimeError("env %d has no journal layout" % int(env_id))
        return dict(hashes)

    def _journal_publish_d7(self, samples):
        """B07: write D7 samples into the reserved slot storage (env-local), split physical/visible.

        One batched write per field (host lists -> index tensors), so a GPU
        env pays a handful of transfers per bridge step, not one per env.
        """
        if not samples:
            return
        rows = sorted(int(env) for env in samples)
        if rows == list(range(int(self.num_envs))):
            self._journal_publish_d7_all_rows(samples, rows)
            return
        physical = [[False] * self._d7_slots for _ in rows]
        visible = [[False] * self._d7_slots for _ in rows]
        centers = self._d7_centers_local[rows].tolist()
        radii = self._d7_radii[rows].tolist()
        for k, env in enumerate(rows):
            for item in samples[env]:
                slot = int(item.slot)
                if not 0 <= slot < self._d7_slots:
                    raise ValueError("D7 sample slot %d outside the reserved storage" % slot)
                centers[k][slot] = [float(item.x), float(item.y)]
                radii[k][slot] = float(item.radius_m)
                physical[k][slot] = bool(item.physical)
                visible[k][slot] = bool(item.visible)
        index = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        self._d7_centers_local[index] = torch.as_tensor(centers, dtype=F64, device=self.device)
        self._d7_radii[index] = torch.as_tensor(radii, dtype=F64, device=self.device)
        self._d7_physical[index] = torch.as_tensor(physical, dtype=torch.bool, device=self.device)
        self._d7_visible[index] = torch.as_tensor(visible, dtype=torch.bool, device=self.device)

    def _journal_publish_d7_all_rows(self, samples, rows):
        """G11: _journal_publish_d7 when every row is present, without the device read-back.

        The read-back path rewrote every (row, slot) centre / radius with the value it had just read
        (a lossless float64 -> Python float -> float64 round trip) and every mask row with False except
        the sampled slots; writing only the sampled entries (last item per (row, slot) wins, as in the
        host lists) after clearing both masks leaves exactly the same bits. One host-to-device copy for
        the sampled entries, none at all when no row carries an object."""
        entries = {}
        for k, env in enumerate(rows):
            for item in samples[env]:
                slot = int(item.slot)
                if not 0 <= slot < self._d7_slots:
                    raise ValueError("D7 sample slot %d outside the reserved storage" % slot)
                entries[(env, slot)] = (float(item.x), float(item.y), float(item.radius_m), bool(item.physical),
                                        bool(item.visible))
        self._d7_physical[:] = False
        self._d7_visible[:] = False
        if not entries:
            return
        keys = list(entries)
        env_t, slot_t, xy_t, radius_t, physical_t, visible_t = device_copies(
            [torch.as_tensor([e for e, _ in keys], dtype=torch.long),
             torch.as_tensor([s for _, s in keys], dtype=torch.long),
             torch.as_tensor([[entries[k][0], entries[k][1]] for k in keys], dtype=F64),
             torch.as_tensor([entries[k][2] for k in keys], dtype=F64),
             torch.as_tensor([entries[k][3] for k in keys], dtype=torch.bool),
             torch.as_tensor([entries[k][4] for k in keys], dtype=torch.bool)], self.device)
        self._d7_centers_local[env_t, slot_t] = xy_t
        self._d7_radii[env_t, slot_t] = radius_t
        self._d7_physical[env_t, slot_t] = physical_t
        self._d7_visible[env_t, slot_t] = visible_t

    def _journal_publish_appearing(self, samples):
        """G3: write the appearing sample (one slot) into its own storage; physical = visible."""
        if not samples:
            return
        rows = sorted(int(env) for env in samples if samples[env])
        if not rows:
            return
        centers, radii, physical, visible = [], [], [], []
        for env in rows:
            (item,) = samples[env]
            if int(item.slot) != self._appearing_slot or bool(item.physical) != bool(item.visible):
                raise ValueError("appearing sample slot %d / masks do not match the appearing slot" % int(item.slot))
            centers.append([[float(item.x), float(item.y)]])
            radii.append([float(item.radius_m)])
            physical.append([bool(item.physical)])
            visible.append([bool(item.visible)])
        index = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        self._appearing_centers_local[index] = torch.as_tensor(centers, dtype=F64, device=self.device)
        self._appearing_radii[index] = torch.as_tensor(radii, dtype=F64, device=self.device)
        self._appearing_physical[index] = torch.as_tensor(physical, dtype=torch.bool, device=self.device)
        self._appearing_visible[index] = torch.as_tensor(visible, dtype=torch.bool, device=self.device)

    def _journal_appearing_geometry(self, env_id):
        """Geometry hash of the env's current E5 layout (the appearing pool records it per key)."""
        layout = self._journal_layouts[int(env_id)]
        if layout is None:
            raise RuntimeError("env %d has no journal layout" % int(env_id))
        return layout.geometry_hash

    # ------------------------------------------------------ DirectRLEnv --
    def _journal_pre_physics_step(self, actions):
        self.actions = self.journal_pre_physics(actions)

    def _journal_apply_action(self):
        from .._shared.body_wrench import apply_body_wrench   # G6: no per-call deprecation warning
        forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins, self._journal_water_z)
        apply_body_wrench(self.robot, forces, torques)

    def _journal_target_indices(self):
        active = (self._journal_oracle.next_gate - 1).clamp(0, 3)
        return active, (active + 1).clamp(max=3)

    def _journal_get_dones(self):
        view = self.journal_pre_dones(self.robot.data, self.scene.env_origins, self._journal_water_z)
        com, quat, lin, ang = self._journal_truth()
        index = torch.arange(self.num_envs, device=self.device)
        active, _ = self._journal_target_indices()
        target = self.waypoints[index, active].to(F64)
        self._journal_pre_distance.copy_(torch.linalg.vector_norm(target - com[:, :2], dim=-1).float())
        bounds = self._journal_bounds
        oob = ((com[:, 0] < bounds[:, 0]) | (com[:, 0] > bounds[:, 1])
               | (com[:, 1] < bounds[:, 2]) | (com[:, 1] > bounds[:, 3]))
        self._journal_collision |= view.d7_contact.to(self.device)
        if self._appearing_bridge is not None:
            self._journal_collision |= self.journal_appearing_contact().to(self.device)   # G3 swept latch
        time_s = self.episode_length_buf.to(F64) * plans.CONTROL_DT_S
        codes, gate_event, _ = self._journal_oracle.step(
            time_s, com, quat, lin, ang, view.water_level_local, self._journal_nominal_height,
            collision=self._journal_collision, out_of_bounds=oob)
        oracle = self._journal_oracle
        self._journal_gate_event.copy_(gate_event)
        self._journal_success_event.copy_(codes == int(contracts.TerminationReason.SUCCESS))
        self.gates_passed.copy_(oracle.next_gate - 1)
        self.path_length.copy_(oracle.full_path.float())
        self.xte_rms.copy_(oracle.xte_rms().float())
        # G11: rows with a nonzero code from the host code list the oracle step already transferred
        # (None: no row finished, the step's own finished.any() said so)
        code_list = oracle.last_code_list
        finished = [] if code_list is None else [i for i, c in enumerate(code_list) if c != 0]
        for i in finished:
            result = oracle.last_completed[i]
            self.journal_episode_results[i] = result
            self.episode_success[i] = bool(result.episode_success)
            self.time_to_success[i] = result.time_to_success
            self.episode_path_length[i] = result.full_episode_path_length_m
            self.episode_full_path_length_m[i] = result.full_episode_path_length_m
            self.episode_success_prefix_path_length_m[i] = result.success_prefix_path_length_m
            self.episode_gates_passed[i] = int(result.metrics["ordered_gates_completed"])
            self.episode_xte_rms[i] = oracle.xte_rms(i)
            self.episode_xte_integral_m_s[i] = result.metrics["path_cross_track_error_integral_m_s"]
            self.episode_route_length[i] = self.route_length[i]
            self.episode_deadline_s[i] = result.metrics["deadline_s"]
            self.episode_termination_reason[i] = int(result.termination_reason)
            self.episode_scenario[i] = self._scenario_params[i]
            self._journal_completed_family[i] = self._journal_family_content[i]
        return self.journal_dones(codes)

    def _journal_get_rewards(self):
        """Reference-only potential progress on TRUTH (success never reads it); lengths / s."""
        progress = (self._journal_prev_distance - self._journal_pre_distance) / self._journal_length_scale
        reward = (self.cfg.reference_reward_progress_scale * progress
                  + self._journal_gate_event.float() * self.cfg.reference_reward_gate_bonus
                  + self.cfg.reference_reward_terminal_success * self._journal_success_event.float())
        com, _, _, _ = self._journal_truth()
        index = torch.arange(self.num_envs, device=self.device)
        active, _ = self._journal_target_indices()
        target = self.waypoints[index, active].to(F64)
        self._journal_prev_distance.copy_(torch.linalg.vector_norm(target - com[:, :2], dim=-1).float())
        return reward

    def _journal_reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        completed = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(completed) > 0:
            self.extras.setdefault("log", {})
            self.extras["log"]["Episode/success"] = self.episode_success[completed].float().mean()
            self.extras["log"]["Episode/gates_passed"] = self.episode_gates_passed[completed].float().mean()
            self.extras["log"]["Episode/path_length_m"] = self.episode_path_length[completed].mean()
        super(JournalEnvMixin, self)._reset_idx(env_ids)          # DirectRLEnv._reset_idx
        self._scenario.reset_idx(env_ids)
        ids = env_ids.tolist()
        episodes = self._scenario.episode_indices()[env_ids.cpu()]
        cfg = self.cfg
        routes, radii, deadlines, starts, headings, bounds = [], [], [], [], [], []
        route_lengths = []                                   # G28: nominal route length (outage rule)
        for i, episode in zip(ids, episodes.tolist()):
            key = _dynamic().EpisodeKey(int(cfg.seed), int(i), int(episode), HAZARD_GROUP)
            beyond = self._d7_pool.beyond_grid(key)            # D24: the pool has no record of such a key
            if not beyond:
                geometry = self._d7_pool.geometry(i, episode)  # ValueError outside the grid (B02 preflight)
            layout = journal_layout(cfg.vehicle, self._journal_tier, cfg.seed, i, episode)   # D16: E5 layout only
            if not beyond and layout.geometry_hash != geometry:
                raise ValueError("env %d episode %d: the E5 layout has geometry %s, the D7 pool recorded %s "
                                 "(pool admitted on another layout; D16 never redraws)"
                                 % (i, episode, layout.geometry_hash, geometry))
            self._journal_layouts[i] = layout
            content = family_hash_content(layout, cfg.vehicle, self._journal_tier, self._journal_spawn_com_z)
            self._journal_family_content[i] = content
            self._journal_family_hashes[i] = family_hashes(content)
            self._scenario_params[i] = {"route": content["route"], "spawn_pose": content["spawn_pose"]}
            self._scenario_hashes[i] = dict(self._journal_family_hashes[i])
            routes.append(route_of(layout))
            radii.append(float(layout.params["gate_radius_m"]))
            deadlines.append(float(layout.deadline_s))
            route_lengths.append(float(layout.nominal_route_length_m))
            starts.append([float(v) for v in layout.start])
            headings.append(float(layout.start_heading_rad))
            bounds.append([float(v) for v in layout.bounds_m])
        self._journal_check_admitted_geometry(ids, episodes.tolist())     # fail closed before any write
        route = torch.as_tensor(np.stack(routes), dtype=F64, device=self.device)
        self.waypoints[env_ids] = route[:, 1:].float()
        segments = route[:, 1:] - route[:, :-1]
        self.segment_lengths[env_ids] = torch.linalg.vector_norm(segments, dim=-1).float()
        self.route_length[env_ids] = self.segment_lengths[env_ids].sum(dim=1)
        self._journal_bounds[env_ids] = torch.as_tensor(bounds, dtype=F64, device=self.device)
        start = torch.as_tensor(starts, dtype=F64, device=self.device)
        heading = torch.as_tensor(headings, dtype=F64, device=self.device)
        # spawn: COM at the layout start, water level + calm equilibrium COM height (D3 section 4);
        # v2 assets have their origin at the COM (verified by the runtime), so the root pose IS the COM.
        origins = self.scene.env_origins[env_ids]
        root_state = torch.zeros((len(ids), 13), device=self.device, dtype=origins.dtype)
        root_state[:, 0:2] = origins[:, 0:2] + start.to(origins.dtype)
        root_state[:, 2] = origins[:, 2] + self._journal_water_z + self._journal_spawn_com_z
        root_state[:, 3:7] = quat_from_yaw(heading).to(origins.dtype)
        self.robot.write_root_state_to_sim(root_state, env_ids)
        self._journal_oracle.reset(env_ids, route, torch.as_tensor(radii, dtype=F64),
                                   torch.as_tensor(deadlines, dtype=F64), self._journal_limits, start)
        self.gates_passed[env_ids] = 0
        self.path_length[env_ids] = 0.0
        self.xte_rms[env_ids] = 0.0
        self._journal_collision[env_ids] = False
        self._journal_gate_event[env_ids] = False
        self._journal_success_event[env_ids] = False
        first = torch.linalg.vector_norm(route[:, 1] - start, dim=-1).float()
        self._journal_prev_distance[env_ids] = first
        self._journal_pre_distance[env_ids] = first
        self._d7_physical[env_ids] = False
        self._d7_visible[env_ids] = False
        if self._appearing_bridge is not None:
            self._appearing_physical[env_ids] = False
            self._appearing_visible[env_ids] = False
        # reference axis = nominal direction of travel on the first leg (start -> gate 1), world rad
        axis = torch.atan2(route[:, 1, 1] - route[:, 0, 1], route[:, 1, 0] - route[:, 0, 0])
        self.journal_reset(env_ids, axis, episodes, reference_route_length_m=torch.as_tensor(route_lengths, dtype=F64))
        if getattr(self.cfg.visual, "enable_waypoint_markers", False) and 0 in ids:
            try:
                self._create_env0_waypoint_markers()
            except Exception as exc:
                print(f"[WARN] Waypoint visualization could not be created: {exc}")
        if getattr(self.cfg.visual, "enable_path_line", False) and 0 in ids:
            try:
                self._create_env0_path_line()
            except Exception as exc:
                print(f"[WARN] Path-line visualization could not be created: {exc}")

    def _journal_check_admitted_geometry(self, ids, episodes):
        """The D7 pool record (admitted candidate or not-admitted key) must belong to THIS layout (fail closed).

        A D24 beyond-grid key has no pool record: its not-admitted assignment
        carries the geometry of the key's E5 layout (built by the pool from the
        key alone), so the comparison below is a consistency check of that
        rebuild, not a pool-record check.
        """
        for i, episode in zip(ids, episodes):
            key = _dynamic().EpisodeKey(int(self.cfg.seed), int(i), int(episode), HAZARD_GROUP)
            try:
                assignment = self._d7_pool.assignment(key, int(self._d7_suite.dose))
            except KeyError:
                raise ValueError("no admitted D7 assignment for %s at dose D%d (preflight error, never an "
                                 "in-episode redraw; D7 B02): the pool covers %s envs x %s episodes"
                                 % (key, int(self._d7_suite.dose), self._d7_pool.meta.get("num_envs"),
                                    self._d7_pool.meta.get("episodes"))) from None
            geometry = assignment_geometry(assignment)
            self._journal_d7_exposure[i] = getattr(assignment, "exposure", EXPOSURE_ASSIGNED)
            if geometry != self._journal_layouts[i].geometry_hash:
                raise ValueError("D7 assignment %s was admitted on layout %s, the env built %s"
                                 % (key, geometry, self._journal_layouts[i].geometry_hash))

    def _journal_ray_ranges(self, com_xy, yaw):
        if self._appearing_bridge is None:
            return point_return_rays(com_xy, yaw, self._d7_centers_local, self._d7_radii, self._d7_visible,
                                     self._journal_ray_count, self._journal_ray_range)
        # G3: the appearing slot joins the point-return proxy (invisible until triggered)
        return point_return_rays(com_xy, yaw, torch.cat((self._d7_centers_local, self._appearing_centers_local), 1),
                                 torch.cat((self._d7_radii, self._appearing_radii), 1),
                                 torch.cat((self._d7_visible, self._appearing_visible), 1),
                                 self._journal_ray_count, self._journal_ray_range)

    def _journal_get_observations(self):
        com, quat, lin, ang = self._journal_truth()
        yaw = yaw_from_quat(quat)
        body_lin = quat_apply_inverse(quat, lin)
        body_ang = quat_apply_inverse(quat, ang)
        body_rates = torch.stack((body_lin[:, 0], body_lin[:, 1], body_ang[:, 2]), -1).float()
        truth_xy = (self.robot.data.root_link_pose_w[:, :2] - self.scene.env_origins[:, :2])
        rays = self._journal_ray_ranges(com[:, :2], yaw).float()
        est = self.journal_estimates(truth_xy, body_rates, rays)
        position = est.position.to(self.waypoints.dtype)
        forward = quat_apply(self.robot.data.root_link_quat_w, self.forward_vec)[:, :2]
        forward = forward / torch.norm(forward, dim=-1, keepdim=True).clamp(min=1.0e-6)
        index = torch.arange(self.num_envs, device=self.device)
        active, following = self._journal_target_indices()
        features = []
        for target in (self.waypoints[index, active], self.waypoints[index, following]):
            rpos = target - position
            distance = torch.norm(rpos, dim=-1, keepdim=True)
            direction = rpos / distance.clamp(min=1.0e-6)
            dot = torch.sum(forward * direction, dim=-1, keepdim=True)
            cross = forward[:, 0:1] * direction[:, 1:2] - forward[:, 1:2] * direction[:, 0:1]
            features.append((dot, cross, distance / self._journal_distance_norm))
        stage = (self._journal_oracle.next_gate - 1).float().unsqueeze(-1) / 4.0
        (dot, cross, dist), (ndot, ncross, ndist) = features
        rays_norm = est.rays.to(self.waypoints.dtype) / self._journal_ray_range
        return {"policy": torch.hstack([dot, cross, dist, stage, ndot, ncross, ndist, rays_norm])}

    # ------------------------------------------------------ certificates --
    def journal_path_record(self, env_id):
        """Completed-episode export: mixin record + contract result + family content (before reset)."""
        i = int(env_id)
        record = self.journal_episode_record(i)
        result = self.journal_episode_results[i]
        if result is None:
            raise RuntimeError("env %d has no completed path-following contract result" % i)
        record["contract"] = result.evaluator_fields()
        record["dynamic_hazards_exposure"] = stamp_exposure(record["dynamic_hazards"])
        if "appearing_hazards" in record:
            from .._shared.conditions.appearing.stamp import exposure_of
            record["appearing_hazards_exposure"] = exposure_of(record["appearing_hazards"])[0]
        record["family"] = {"primitives": self._journal_completed_family[i], "wiring": F3_WIRING_VERSION}
        return record


class _JournalAssetSpec:
    """What the legacy _setup_scene reads from ``vehicle_spec`` for a v2 journal vessel."""

    asset_kind = "rigid_object"
    bow_body_axis = "+x"             # every v2 asset: bow +x, origin = COM (vehicles_v2.py:8-10)

    def __init__(self, name):
        self.name = name


__all__ = ["PathFollowingJournalMixin", "OrderedPathTensorOracle", "py_hypot", "assert_hypot_parity",
           "capsize_cos_threshold", "journal_layout", "route_of", "d7_task_context", "d7_envelope",
           "admission_suite", "suite_identity", "admit_path_following_grid", "admit_path_following_keys",
           "derive_not_admitted", "primary_reasons", "per_dose_counts_with_not_admitted", "pool_meta",
           "NotAdmittedAssignment", "not_admitted_digest", "not_admitted_stamp", "stamp_exposure",
           "assignment_geometry", "D16Pool", "LoadedD7Pool", "OnlineExtendedPool", "D16HazardBridge",
           "write_pool_file", "read_pool_file", "load_journal_d7_pool", "resolve_pool_path",
           "read_pool_manifest", "pool_manifest_path", "split_pool_container", "join_pool_container",
           "pool_file_name", "point_return_rays", "family_hash_content", "family_hashes", "TASK", "FAMILY",
           "HAZARD_GROUP", "POOL_DIR", "POOL_DIR_ENV", "POOL_MANIFEST_NAME", "POOL_SCHEMA", "F3_WIRING_VERSION",
           "D16_RULE", "EXPOSURE_ASSIGNED", "EXPOSURE_NOT_ADMITTED", "PER_DOSE_FIELDS", "offered_grid",
           "BEYOND_GRID_REASON", "BEYOND_GRID_REJECTIONS", "D24_RULE", "ONLINE_BEYOND_GRID_RULE"]
