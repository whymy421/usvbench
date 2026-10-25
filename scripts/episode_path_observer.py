"""Read-only pre-autoreset path telemetry, installed on one env instance.

Finding 60: publish success-prefix and full-episode travel as separate fields.

No task source edits and no dynamics/reset writes. The wrapper returns the
original _get_dones result unchanged (same object). Isaac Lab's DirectRLEnv.step
calls self._get_dones() BEFORE it resets finished envs, so the positions read
here still belong to the finishing episode. Distances are horizontal root
travel sampled at control steps, matching the legacy path sampling, not
physics substeps.

G11 (per-step cost): when the env hands torch tensors (every Isaac Lab env),
a control step costs no host synchronisation and no Python loop over envs:
observe() and after_step() only snapshot the horizontal root positions, the
success latch and the done mask on the env's device (copy kernels) and append
them to a log. The log is moved to the host (one transfer per kind) and
replayed when the telemetry is read: take() (the evaluator records a finished
episode), any read of the state attributes, a step whose inputs are not
tensors, and every LOG_CAPACITY steps (bounds device memory). The replay
performs the eager observer's per-env arithmetic in the same order: Python
math.hypot of the float64 differences of the exactly widened positions,
float64 accumulation per env, the first-success prefix latch, the done latch
and the autoreset exclusion (vectorised over envs with numpy float64, whose
elementwise add is the same IEEE double add as Python's). Inputs that are not
tensors (unit tests, legacy fakes) take the eager path, which is the original
code.
"""
import math

import numpy as np

LOG_CAPACITY = 1024     # deferred control steps before a forced replay (bounds device memory)


def _flags(values, n):
    """Flatten a done/success mask ([n] or [n, 1]) to n Python booleans."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "reshape") and not isinstance(values, (list, tuple)):
        values = values.reshape(-1)
    flat = values.tolist() if hasattr(values, "tolist") else list(values)
    flat = [v[0] if isinstance(v, (list, tuple)) else v for v in flat]
    if len(flat) != n:
        raise ValueError("mask does not have one entry per environment")
    return [bool(v) for v in flat]


def _is_tensor(value):
    return (type(value).__module__.split(".")[0] == "torch" and hasattr(value, "detach")
            and hasattr(value, "dtype") and hasattr(value, "device"))


def _device_flags(values, n, snapshot=True):
    """On-device mask with _flags' truth value per entry (bool(v) is v != 0; NaN -> True).

    A bool mask is cloned (snapshot) unless the caller combines it into a new tensor at once."""
    flat = values.detach().reshape(-1)
    if flat.numel() != n:
        raise ValueError("mask does not have one entry per environment")
    if str(flat.dtype) == "torch.bool":
        return flat.clone() if snapshot else flat
    return flat != 0


class EpisodePathObserver:
    def __init__(self, base):
        self.base = base
        self.n = int(base.num_envs)
        self._previous = None
        self._full = [0.0] * self.n
        self._prefix = [None] * self.n
        self._completed = {}
        self._pending = set()
        self._log = []              # deferred device snapshots of control steps, replayed in order
        # Always wrap the env's ORIGINAL method: a later observer on the same
        # env (convergence_curve builds one certificate per rung) replaces the
        # earlier one instead of stacking wrappers.
        original = getattr(base, "_usvbench_original_get_dones", None)
        if original is None:
            original = getattr(base, "_get_dones", None)
            if callable(original):
                base._usvbench_original_get_dones = original
        self.enabled = callable(original) and self._positions() is not None
        if self.enabled:
            def observed_dones(*args, **kwargs):
                result = original(*args, **kwargs)
                self.observe(result)
                return result
            base._get_dones = observed_dones

    # ------------------------------------------------- state (replayed view) --
    @property
    def previous(self):
        self._flush()
        return self._previous

    @previous.setter
    def previous(self, value):
        self._flush()
        self._previous = value

    @property
    def full(self):
        self._flush()
        return self._full

    @full.setter
    def full(self, value):
        self._flush()
        self._full = value

    @property
    def prefix(self):
        self._flush()
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        self._flush()
        self._prefix = value

    @property
    def completed(self):
        self._flush()
        return self._completed

    @property
    def pending(self):
        self._flush()
        return self._pending

    # ------------------------------------------------------------- inputs --
    def _raw_positions(self):
        robot = getattr(self.base, "robot", None)
        data = getattr(robot, "data", None)
        return getattr(data, "root_pos_w", None)

    def _positions(self):
        positions = self._raw_positions()
        if positions is None:
            return None
        if hasattr(positions, "detach"):
            positions = positions.detach()
        values = positions.tolist() if hasattr(positions, "tolist") else positions
        return [(float(row[0]), float(row[1])) for row in values]

    def _deferrable_positions(self, raw):
        """A [n, >=2] tensor (the eager _positions() could neither raise nor truncate)."""
        return _is_tensor(raw) and raw.dim() == 2 and int(raw.shape[0]) == self.n and int(raw.shape[1]) >= 2

    # ---------------------------------------------------------- eager steps --
    def _observe_eager(self, dones):
        positions = self._positions()
        success = getattr(self.base, "_success", None)
        success = _flags(success, self.n) if success is not None else None
        term, trunc = dones
        term, trunc = _flags(term, self.n), _flags(trunc, self.n)
        for i, (before, after) in enumerate(zip(self._previous, positions)):
            distance = math.hypot(after[0] - before[0], after[1] - before[1])
            self._full[i] += distance
            if self._prefix[i] is None and success is not None and success[i]:
                self._prefix[i] = self._full[i]
            if term[i] or trunc[i]:
                self._completed[i] = (self._full[i], self._prefix[i])
                self._pending.add(i)
        self._previous = positions

    def _after_step_eager(self, positions):
        if positions is not None and self._previous is not None:
            for i in self._pending:
                self._previous[i] = positions[i]
                self._full[i] = 0.0
                self._prefix[i] = None
        self._pending.clear()

    # ------------------------------------------------------------- replay --
    def _flush(self):
        """Move the deferred snapshots to the host (one transfer per kind) and replay them in order."""
        if not self._log:
            return
        log, self._log = self._log, []
        import torch
        positions = [entry[1] for entry in log]
        if len({p.dtype for p in positions}) != 1:
            positions = [p.to(torch.float64) for p in positions]
        host_xy = torch.stack(positions).cpu().numpy().astype(np.float64)       # exact widening
        flags = [f for entry in log if entry[0] == "observe" for f in entry[2:] if f is not None]
        host_flags = torch.stack(flags).cpu().numpy().astype(bool) if flags else None
        events, k = [], 0
        for index, entry in enumerate(log):
            if entry[0] == "observe":
                success = None
                if entry[2] is not None:
                    success, k = host_flags[k], k + 1
                events.append(("observe", host_xy[index], success, host_flags[k]))
                k += 1
            else:
                events.append(("after_step", host_xy[index]))
        if self._previous is not None and len(self._previous) == self.n:
            self._replay_vectorised(events)
        else:
            self._replay_scalar(events)

    def _replay_scalar(self, events):
        """The eager code on the replayed inputs (used only when the state is not n rows)."""
        for event in events:
            xy = [(float(x), float(y)) for x, y in event[1].tolist()]
            if event[0] == "after_step":
                self._after_step_eager(xy)
                continue
            success = None if event[2] is None else [bool(v) for v in event[2].tolist()]
            done = [bool(v) for v in event[3].tolist()]
            for i, (before, after) in enumerate(zip(self._previous, xy)):
                distance = math.hypot(after[0] - before[0], after[1] - before[1])
                self._full[i] += distance
                if self._prefix[i] is None and success is not None and success[i]:
                    self._prefix[i] = self._full[i]
                if done[i]:
                    self._completed[i] = (self._full[i], self._prefix[i])
                    self._pending.add(i)
            self._previous = xy

    def _replay_vectorised(self, events):
        """Same per-env operations as _observe_eager / _after_step_eager, vectorised over envs."""
        with np.errstate(all="ignore"):     # inf - inf -> nan exactly as Python floats, without a warning
            self._replay_events(events)

    def _replay_events(self, events):
        n = self.n
        previous = np.array(self._previous, dtype=np.float64).reshape(n, 2)
        full = np.array(self._full, dtype=np.float64)
        has_prefix = np.array([p is not None for p in self._prefix], dtype=bool)
        prefix = np.array([p if p is not None else 0.0 for p in self._prefix], dtype=np.float64)
        for event in events:
            after = event[1]
            if event[0] == "after_step":
                if self._pending:
                    rows = sorted(self._pending)
                    previous[rows] = after[rows]
                    full[rows] = 0.0
                    has_prefix[rows] = False
                    prefix[rows] = 0.0
                self._pending.clear()
                continue
            success, done = event[2], event[3]
            dx = after[:, 0] - previous[:, 0]
            dy = after[:, 1] - previous[:, 1]
            full = full + np.fromiter(map(math.hypot, dx.tolist(), dy.tolist()), dtype=np.float64, count=n)
            if success is not None:
                latch = success & ~has_prefix
                prefix = np.where(latch, full, prefix)
                has_prefix = has_prefix | latch
            for i in np.flatnonzero(done).tolist():
                self._completed[i] = (float(full[i]), float(prefix[i]) if has_prefix[i] else None)
                self._pending.add(i)
            previous = after.copy()
        self._previous = [(float(x), float(y)) for x, y in previous.tolist()]
        self._full = [float(v) for v in full.tolist()]
        self._prefix = [float(v) if h else None for v, h in zip(prefix.tolist(), has_prefix.tolist())]

    # ------------------------------------------------------------- public --
    def after_reset(self):
        self._log = []          # after_reset replaces every piece of state the log could change
        self._previous = self._positions()
        self._full = [0.0] * self.n
        self._prefix = [None] * self.n
        self._completed.clear()
        self._pending.clear()

    def observe(self, dones):
        if self._previous is None and not self._log:
            return
        term, trunc = dones
        raw = self._raw_positions()
        success = getattr(self.base, "_success", None)
        if (self._deferrable_positions(raw) and _is_tensor(term) and _is_tensor(trunc)
                and (success is None or _is_tensor(success))):
            # _flags order: success first, then terminated, truncated (same ValueError, raised now)
            snap_success = None if success is None else _device_flags(success, self.n)
            done = _device_flags(term, self.n, snapshot=False) | _device_flags(trunc, self.n, snapshot=False)
            self._log.append(("observe", raw.detach()[:, :2].clone(), snap_success, done))
            if len(self._log) >= LOG_CAPACITY:
                self._flush()
            return
        self._flush()
        self._observe_eager(dones)

    def after_step(self):
        # step() has now performed automatic resets. Exclude teleport travel.
        raw = self._raw_positions()
        if self._deferrable_positions(raw) and (self._log or self._previous is not None):
            if self._log or self._pending:
                self._log.append(("after_step", raw.detach()[:, :2].clone()))
                if len(self._log) >= LOG_CAPACITY:
                    self._flush()
            # else: nothing pending -> the eager loop is empty and pending.clear() a no-op
            return
        self._flush()
        self._after_step_eager(self._positions())

    def take(self, env, success):
        self._flush()
        full, prefix = self._completed.pop(env, (None, None))
        if not success:
            prefix = None
        # Invalid physics is unavailable, never a made-up finite measurement.
        return {"path_length_full_episode_m": full if full is None or math.isfinite(full) else None,
                "path_length_success_prefix_m": prefix if prefix is None or math.isfinite(prefix) else None}
