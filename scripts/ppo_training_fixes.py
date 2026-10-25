"""G44 (v2.6.3, owner decision 2026-10-13): PPO training fixes R1 and R2. Training only; the evaluator, the reward
code of every task, the network, the budget, the seeds and the physics are unchanged.

R2  FAILURE TERMINAL PENALTY. Every row that terminates with a failure code (journal_contracts_v2 TERMINATION_TABLE:
    every 'terminated' code except success 1, i.e. collision 2, out_of_bounds 4, failure 5, capsize 6, sinking 7,
    airborne 8, non_finite_state 9) receives, in the training wrapper and before the agent sees it, the reward
    P = -(A + B * D) on top of the family's own step reward. D = diagonal of that episode's layout bounds box (the
    largest distance between two in-bounds points; the task target lies inside the box). A, B come from the family's
    per-step reward terms (read from the run's env cfg) and the horizon N = ceil(episode_length_s / control_dt) + 1
    (episode_length_s = the plan's maximum deadline, journal_runtime_cfg_v2.py:47).
    Derivation (per family, penalty_terms): write the step reward as r = (telescoping progress) + (dense per-step
    term d, lo <= d <= hi) + (events e, e <= e_hi on a failure step, e >= 0 in a continuation that does not fail;
    contact ends the episode, so contact entry / dwell never occur in such a continuation). A failure at step t is
    worth r_t + P; any continuation from the state before step t to the time limit that does not fail is worth at
    least (lowest progress it can lose) + N * min(lo, 0). P <= (progress range) + N * min(lo, 0) - max(hi, 0) - e_hi
    therefore guarantees: failing is never worth more than continuing to the time limit, from any state.
    -A - B * D is exactly that bound (no extra margin).

R1  (a) the state normaliser (skrl RunningStandardScaler, agent._state_preprocessor) is FROZEN once the trainer
    reaches WARMUP_FRACTION of the budget: its forward() is called with train=False from then on (the running
    statistics stop moving; the value preprocessor is unchanged);
    (b) learning-rate floor: KLAdaptiveLR.min_lr = max(min_lr, LR_FLOOR_FRACTION x the yaml learning_rate);
    (c) periodic evaluation: every saved checkpoint (10 % of the budget, unchanged) is evaluated by the protocol's
    own evaluator (calm, 64 x 1) and the best one by evaluation is kept beside the final checkpoint
    (select_best: highest success rate, tie -> the LATER checkpoint, i.e. closest to the last-checkpoint rule).
    Which of final / best is the headline is an owner decision; both are evaluated.
Pure Python (math, hashlib); torch only inside the agent hooks. No simulator import.
"""
from __future__ import annotations

import hashlib
import json
import math

G44_VERSION = "g44-ppo-fixes-v1"
FAILURE_CODES = (2, 4, 5, 6, 7, 8, 9)
SUCCESS_CODE, TIMEOUT_CODE = 1, 3
# one control step covers at most STEP_SPEED_BOUND_MPS x control_dt metres (both vessels top out near 3 m/s:
# tasks/_shared/vehicles_v2_data.py; 10 m/s leaves a wide margin for current / wave drift)
STEP_SPEED_BOUND_MPS = 10.0
BOUNDS_ATTRIBUTE = {"hazard_nav": "_journal_bounds", "path_following": "_journal_bounds", "path_hazard": "_jph_bounds",
                    "station_keeping": "_jbounds", "docking": "_jd_bounds", "harbor_mission": "_jbounds"}
WARMUP_FRACTION = 0.05
LR_FLOOR_FRACTION = 0.1
FAILURE_PENALTY_RULE = ("G44 R2: a row terminating with a failure code %s gets P = -(A + B x D) added to its step "
                        "reward before the agent sees it; D = diagonal of the episode's layout bounds box; A, B from "
                        "the family's per-step reward range (env cfg) and the horizon N = ceil(episode_length_s / "
                        "control_dt) + 1; failing is then never worth more than any continuation to the time limit "
                        "that does not fail" % (FAILURE_CODES,))
FREEZE_RULE = ("G44 R1a: the state normaliser (RunningStandardScaler) updates only until the trainer reaches %.0f %% "
               "of the budget; from then on it is called with train=False (statistics frozen)" % (100 * WARMUP_FRACTION))
LR_FLOOR_RULE = ("G44 R1b: KLAdaptiveLR min_lr = max(min_lr, %.2g x the yaml learning_rate)" % LR_FLOOR_FRACTION)
BEST_RULE = ("G44 R1c: every periodic checkpoint is evaluated (protocol evaluator, calm, 64 x 1); best = highest "
             "success rate, tie -> the later checkpoint; final and best are both kept and evaluated")


class FixError(ValueError):
    """A fix cannot be derived or installed (refuse rather than train with an unknown reward bound)."""


# ------------------------------------------------------------------ R2 failure penalty
def _num(cfg, name, minimum=None):
    value = getattr(cfg, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FixError("env cfg field %s must be a finite number (got %r)" % (name, value))
    value = float(value)
    if minimum is not None and value < minimum:
        raise FixError("env cfg field %s must be >= %s (got %r)" % (name, minimum, value))
    return value


def control_dt(cfg):
    sim = getattr(cfg, "sim", None)
    dt = _num(sim, "dt", 0.0) if sim is not None else None
    if not dt:
        raise FixError("env cfg sim.dt must be positive")
    decimation = getattr(cfg, "decimation", None)
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
        raise FixError("env cfg decimation must be a positive integer (got %r)" % (decimation,))
    return dt * decimation


def horizon_steps(cfg):
    length = _num(cfg, "episode_length_s", 0.0)
    return int(math.ceil(length / control_dt(cfg) - 1.0e-9)) + 1


def _prox_log(cfg):
    """max of -mean(log(clamp(ray, floor, cap) / cap)) = log(cap / floor), cap = safe_clearance_m + half_beam_m."""
    cap = _num(cfg, "safe_clearance_m", 0.0) + _num(cfg, "half_beam_m", 0.0)
    floor = _num(cfg, "prox_ray_floor_m", 0.0)
    if floor <= 0.0:
        raise FixError("prox_ray_floor_m must be positive (the log barrier is unbounded otherwise)")
    return max(0.0, math.log(cap / floor))


def penalty_terms(carrier, cfg, plan_params):
    """{A, B, N, control_dt_s, parts}: the failure penalty of an episode with bounds diagonal D is -(A + B * D).
    ``cfg`` = the run's env cfg (attribute access), ``plan_params`` = journal_cfg_v2 plan params of the task/vessel."""
    dt = control_dt(cfg)
    n = horizon_steps(cfg)
    step_m = STEP_SPEED_BOUND_MPS * dt
    parts = {}
    if carrier == "hazard_nav":
        # hazard_nav_env.py _get_rewards: progress_scale x (Phi - Phi_prev), Phi = -clip(d / d0, 0, 1) (or 1 - clip with
        # pbrs_shift_potential; range 1 either way); costs clearance, log-barrier, swift, open water, reverse (each
        # scale x dt x [0, 1] or [0, log(cap/floor)]); goal bonus entry + time (an out-of-bounds step may carry it).
        if _num(cfg, "reward_threading_amplitude", 0.0) > 0.0:
            raise FixError("threading bonus enabled: its per-step range is not derived here (G44)")
        prog = _num(cfg, "reward_progress_scale", 0.0)
        extra = 0.0
        if bool(getattr(cfg, "pbrs_correct", False)) and bool(getattr(cfg, "pbrs_shift_potential", False)):
            extra = n * (1.0 - _num(cfg, "pbrs_gamma", 0.0))   # (gamma - 1) Phi' with Phi' in [0, 1] per step
        dense = dt * (_num(cfg, "reward_clearance_scale", 0.0) + _num(cfg, "reward_prox_scale", 0.0) * _prox_log(cfg)
                      + _num(cfg, "reward_swift_scale", 0.0) + _num(cfg, "reward_open_water_scale", 0.0)
                      + _num(cfg, "reward_reverse_action_scale", 0.0))
        events = _num(cfg, "reward_goal_entry_bonus", 0.0) + _num(cfg, "reward_goal_time_bonus", 0.0)
        parts = {"progress_range": prog * (1.0 + extra), "dense_cost_per_step": dense, "failure_step_events": events}
        a, b = prog * (1.0 + extra) + n * dense + events, 0.0
    elif carrier == "path_following":
        # journal_path_following.py _journal_get_rewards: progress_scale x (d_prev - d) / length_scale to the active gate
        # (telescoping per gate segment; a segment ends with d <= gate radius + one step), gate bonus, success bonus.
        prog = _num(cfg, "reference_reward_progress_scale", 0.0)
        gate = _num(cfg, "reference_reward_gate_bonus", 0.0)
        succ = _num(cfg, "reference_reward_terminal_success", 0.0)
        ls = float(plan_params["length_scale"])
        r_g = float(plan_params["gate_radius_m"])
        gates = int(plan_params["segment_count"]) + 1
        per_gate = max(0.0, prog * (r_g + step_m) / ls - gate)
        parts = {"progress_per_metre": prog / ls, "failure_step_events": gate + succ, "later_gate_segments": gates,
                 "later_gate_segment_loss": per_gate}
        a, b = gate + succ + gates * per_gate, prog / ls
    elif carrier == "path_hazard":
        # path_hazard_env.py _get_rewards: progress_scale x (d_prev - d) in metres (per gate segment), gate bonus,
        # clearance / log-barrier / reverse / swift costs, last-gate entry + time bonus, contact terms.
        prog = _num(cfg, "reference_reward_progress_scale", 0.0)
        gate = _num(cfg, "reference_reward_gate_bonus", 0.0)
        r_g = float(plan_params["gate_radius_m"])
        gates = int(plan_params["segment_count"]) + 1
        dense = dt * (_num(cfg, "reward_clearance_scale", 0.0) + _num(cfg, "reward_prox_scale", 0.0) * _prox_log(cfg)
                      + _num(cfg, "reward_reverse_action_scale", 0.0) + _num(cfg, "reward_swift_scale", 0.0))
        events = gate + _num(cfg, "reward_goal_entry_bonus", 0.0) + _num(cfg, "reward_goal_time_bonus", 0.0)
        per_gate = max(0.0, prog * (r_g + step_m) - gate)
        parts = {"progress_per_metre": prog, "dense_cost_per_step": dense, "failure_step_events": events,
                 "later_gate_segments": gates, "later_gate_segment_loss": per_gate}
        a, b = n * dense + events + gates * per_gate, prog
    elif carrier == "station_keeping":
        # station_keeping_env.py _journal_get_rewards: -d / distance_scale + 1[inside the hold circle], d <= D.
        scale = _num(cfg, "reference_reward_distance_scale", 0.0)
        if scale <= 0.0:
            raise FixError("reference_reward_distance_scale must be positive")
        parts = {"distance_cost_per_metre_per_step": 1.0 / scale, "failure_step_events": 1.0}
        a, b = 1.0, n / scale
    elif carrier == "docking":
        # docking_env.py _journal_get_rewards: -d / distance_scale + alignment x cos x exp(-d/decay) + braking x exp x
        # clip(1 - v/vs, 0, 1) + success_bonus x conjunction; lo = -D/scale - alignment, hi = alignment + braking + bonus.
        scale = _num(cfg, "reference_reward_distance_scale_m", 0.0)
        if scale <= 0.0:
            raise FixError("reference_reward_distance_scale_m must be positive")
        align = abs(_num(cfg, "reference_reward_alignment_scale"))
        brake = abs(_num(cfg, "reference_reward_braking_scale"))
        bonus = abs(_num(cfg, "reference_reward_success_bonus"))
        parts = {"distance_cost_per_metre_per_step": 1.0 / scale, "dense_cost_per_step": align,
                 "failure_step_events": align + brake + bonus}
        a, b = n * align + align + brake + bonus, n / scale
    elif carrier == "harbor_mission":
        # harbor_mission_env.py _get_rewards: progress_scale x phase potential (clip(x)/3 per phase, 3 phases: range 1),
        # clearance / log-barrier costs, milestones (3), stage entry + time bonus, dock conjunction.
        prog = _num(cfg, "reward_progress_scale", 0.0)
        dense = dt * (_num(cfg, "reward_clearance_scale", 0.0) + _num(cfg, "reward_prox_scale", 0.0) * _prox_log(cfg))
        events = (3.0 * _num(cfg, "reward_milestone_bonus", 0.0) + _num(cfg, "reward_stage_goal_bonus", 0.0)
                  + _num(cfg, "reward_stage_time_bonus", 0.0) + abs(_num(cfg, "reward_dock_conjunction")))
        parts = {"progress_range": prog, "dense_cost_per_step": dense, "failure_step_events": events}
        a, b = prog + n * dense + events, 0.0
    else:
        raise FixError("no failure-penalty derivation for carrier %r" % (carrier,))
    return {"A": float(a), "B": float(b), "N": int(n), "control_dt_s": float(dt), "carrier": carrier,
            "parts": {k: (float(v) if isinstance(v, float) else v) for k, v in parts.items()}}


def bounds_diagonal(bounds):
    x0, x1, y0, y1 = (float(v) for v in bounds)
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1)) or x1 < x0 or y1 < y0:
        raise FixError("invalid layout bounds %r" % (bounds,))
    return math.hypot(x1 - x0, y1 - y0)


def failure_penalty(terms, diagonal):
    return -(terms["A"] + terms["B"] * float(diagonal))


class FailurePenalty:
    """Per-env penalty captured at each episode start (the env auto-resets inside step, so the value of the episode
    that just ended must be read before the step)."""

    def __init__(self, terms, base, carrier, num_envs):
        self.terms, self.base, self.num_envs = terms, base, int(num_envs)
        self.attribute = BOUNDS_ATTRIBUTE[carrier]
        self.values = None
        self.rows = 0
        self.per_code = {}
        self.min_applied = self.max_applied = None
        self.diag_min = self.diag_max = None
        self.last = {}

    def _bounds_rows(self):
        bounds = getattr(self.base, self.attribute, None)
        if bounds is None:
            raise FixError("the env has no per-env bounds %s" % self.attribute)
        rows = bounds.detach().to("cpu").tolist() if hasattr(bounds, "detach") else [list(r) for r in bounds]
        if len(rows) != self.num_envs:
            raise FixError("bounds %s has %d rows for %d envs" % (self.attribute, len(rows), self.num_envs))
        return rows

    def refresh(self, rows_to_update=None):
        rows = self._bounds_rows()
        if self.values is None:
            self.values = [0.0] * self.num_envs
            rows_to_update = range(self.num_envs)
        for i in rows_to_update:
            d = bounds_diagonal(rows[i])
            self.diag_min = d if self.diag_min is None else min(self.diag_min, d)
            self.diag_max = d if self.diag_max is None else max(self.diag_max, d)
            self.values[i] = failure_penalty(self.terms, d)

    def penalties(self, dones, codes):
        """{row: penalty} for the rows that FAIL in this step (call before refresh of those rows)."""
        if self.values is None:
            raise FixError("failure penalties not initialised (refresh before the first step)")
        if codes is None:
            raise FixError("the env publishes no termination codes: the failure penalty cannot be applied")
        out = {}
        self.last = out
        for i, done in enumerate(dones):
            if done and int(codes[i]) in FAILURE_CODES:
                value = self.values[i]
                out[i] = value
                self.rows += 1
                key = str(int(codes[i]))
                self.per_code[key] = self.per_code.get(key, 0) + 1
                self.min_applied = value if self.min_applied is None else min(self.min_applied, value)
                self.max_applied = value if self.max_applied is None else max(self.max_applied, value)
        return out

    def summary(self):
        return {"rule": FAILURE_PENALTY_RULE, "failure_codes": list(FAILURE_CODES), "terms": self.terms,
                "bounds_attribute": self.attribute, "rows_penalised": self.rows,
                "per_code": dict(sorted(self.per_code.items())), "penalty_min": self.min_applied,
                "penalty_max": self.max_applied, "bounds_diagonal_min_m": self.diag_min,
                "bounds_diagonal_max_m": self.diag_max}


# ------------------------------------------------------------------ R1a normaliser freeze
def module_digest(module):
    """SHA-256 over the module's state_dict (names, dtypes, shapes, bytes)."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        t = tensor.detach().to("cpu").contiguous()
        h.update(name.encode("ascii"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class NormaliserFreeze:
    def __init__(self, warmup_timesteps):
        self.warmup = int(warmup_timesteps)
        if self.warmup < 0:
            raise FixError("warm-up timesteps must be >= 0")
        self.frozen = False
        self.frozen_at = None
        self.scaler = None
        self.digest_at_freeze = None
        self.train_calls_blocked = 0
        self.train_calls_before = 0

    def attach(self, agent):
        scaler = getattr(agent, "_state_preprocessor", None)
        if scaler is None or not hasattr(scaler, "running_mean") or not callable(getattr(scaler, "forward", None)):
            raise FixError("the agent has no RunningStandardScaler state preprocessor to freeze")
        original = scaler.forward
        freeze = self

        def gated(x, *args, **kwargs):
            train = kwargs.get("train", args[0] if args else False)
            if train:
                if freeze.frozen:
                    freeze.train_calls_blocked += 1
                    if args:
                        args = (False,) + tuple(args[1:])
                    else:
                        kwargs["train"] = False
                else:
                    freeze.train_calls_before += 1
            return original(x, *args, **kwargs)

        scaler.forward = gated          # instance attribute: nn.Module.__call__ resolves it; module identity kept
        self.scaler = scaler

    def maybe_freeze(self, timestep):
        if not self.frozen and int(timestep) >= self.warmup:
            self.frozen = True
            self.frozen_at = int(timestep)
            self.digest_at_freeze = module_digest(self.scaler) if self.scaler is not None else None
        return self.frozen

    def summary(self):
        end = module_digest(self.scaler) if self.scaler is not None else None
        return {"rule": FREEZE_RULE, "warmup_timesteps": self.warmup, "frozen": self.frozen,
                "frozen_at_timestep": self.frozen_at, "digest_at_freeze": self.digest_at_freeze,
                "digest_at_end": end, "unchanged_after_freeze": bool(self.frozen and end == self.digest_at_freeze),
                "train_calls_before_freeze": self.train_calls_before,
                "train_calls_blocked_after_freeze": self.train_calls_blocked}


# ------------------------------------------------------------------ R1b learning-rate floor
class LearningRateFloor:
    def __init__(self, base_lr, fraction=LR_FLOOR_FRACTION):
        self.base_lr = float(base_lr)
        if not (self.base_lr > 0.0 and math.isfinite(self.base_lr)):
            raise FixError("the yaml learning_rate must be positive")
        self.floor = float(fraction) * self.base_lr
        self.previous_min_lr = None
        self.lowest_seen = None
        self.updates_at_floor = 0
        self.updates_seen = 0
        self.agent = None

    def attach(self, agent):
        scheduler = getattr(agent, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "min_lr"):
            raise FixError("the agent has no KLAdaptiveLR scheduler with min_lr")
        self.previous_min_lr = float(scheduler.min_lr)
        scheduler.min_lr = max(self.previous_min_lr, self.floor)
        self.agent = agent

    def current_lr(self):
        return float(self.agent.optimizer.param_groups[0]["lr"])

    def observe(self):
        lr = self.current_lr()
        self.updates_seen += 1
        self.lowest_seen = lr if self.lowest_seen is None else min(self.lowest_seen, lr)
        if lr <= self.floor * (1.0 + 1.0e-9):
            self.updates_at_floor += 1
        return lr

    def summary(self):
        return {"rule": LR_FLOOR_RULE, "base_lr": self.base_lr, "floor": self.floor,
                "previous_min_lr": self.previous_min_lr, "lowest_lr_seen": self.lowest_seen,
                "updates_seen": self.updates_seen, "updates_at_floor": self.updates_at_floor,
                "floor_respected": self.lowest_seen is None or self.lowest_seen >= self.floor * (1.0 - 1.0e-9)}


# ------------------------------------------------------------------ R1c best checkpoint by evaluation
def certificate_outcome(document):
    """(successes, episodes, checkpoint sha256) of an eval_v6_frozen certificate (as the G33 judge reads it)."""
    recs = document.get("records") if isinstance(document, dict) else None
    header = document.get("header") if isinstance(document, dict) else None
    if not isinstance(recs, list) or not recs or not isinstance(header, dict) or header.get("complete") is not True:
        raise FixError("not a complete evaluation certificate")
    k = sum(1 for r in recs if isinstance(r, dict) and r.get("success") is True)
    sha = ((header.get("checkpoint") or {}).get("sha256") or "").lower()
    return k, len(recs), sha


def select_best(evaluations):
    """evaluations: [{timestep, successes, episodes, sha256}] -> the best entry (highest success rate; tie -> later)."""
    if not evaluations:
        raise FixError("no evaluated checkpoint")
    n = {int(e["episodes"]) for e in evaluations}
    if len(n) != 1 or min(n) <= 0:
        raise FixError("every checkpoint must be evaluated on the same positive number of episodes (got %s)" % sorted(n))
    steps = [int(e["timestep"]) for e in evaluations]
    if len(set(steps)) != len(steps):
        raise FixError("a checkpoint is evaluated twice")
    return max(evaluations, key=lambda e: (int(e["successes"]), int(e["timestep"])))


def sidecar(failure, freeze, floor):
    return {"schema": "usvbench.ppo_fix_g44/1", "version": G44_VERSION,
            "failure_terminal_penalty": failure.summary(), "normaliser_freeze": freeze.summary(),
            "learning_rate_floor": floor.summary(), "best_checkpoint_rule": BEST_RULE}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
