"""v2.6.4 PPO final configuration (owner decision 2026-10-14; plan 46 config P1, tested as the G46 dev candidate,
KL arm adopted, fixed learning rates rejected): PPO training changes, applied by scripts/train_journal_ppo.py.
Training only; the evaluator, the task reward code, the network, the registration yaml (KLAdaptiveLR, lr 1.5e-4,
shared networks, gamma 0.99), budgets, seeds and physics are unchanged. (File name kept from the G46 dev candidate.)

P1-a  NO learning-rate floor: KLAdaptiveLR exactly as in the registration yaml (G44 R1b removed). The G46 smoke
      switch USVB_G46_LR is REMOVED: a set value is refused (removed_switch_error). The learning rate of every
      update is logged (LearningRateLog: lowest / highest / last) into ppo_fix.json.
P1-b  DISCOUNTED failure terminal penalty (replaces the G44 R2 value; same failure codes, same per-episode wrapper
      FailurePenalty, same term inventory as ppo_training_fixes.penalty_terms):
          P = -(H x L + E)     (per episode -(A + B x D) with D = the layout bounds diagonal; B = 0 except the
                                two families whose per-step cost grows with the distance)
      H = min(ceil(1 / (1 - gamma)), N) control steps: gamma = the run's agent discount_factor (100 for 0.99), capped
      by the episode horizon N = ceil(episode_length_s / control_dt) + 1 (the continuation cannot be longer; the
      yaml time_limit_bootstrap is False). sum_k gamma^k <= H, so a continuation that loses at most L per step is
      worth at least -H x L in the return PPO optimises.
      L = the largest loss per control step, from the run's env cfg; progress terms use the vessel's top speed
      (tasks/_shared/vehicles_v2_data.py PINNED_DERIVED[<profile>]['speeds.top_model_m_s'] x control_dt metres per
      step), not 10 m/s.
      E = everything the failure step itself can earn: the family's failure-step events (G44 inventory) plus, for a
      per-step progress term, one step of the largest progress gain.
      Per family (derivation in discounted_terms): path_following (H + 1) x prog x s / length_scale + gate + success;
      path_hazard (H + 1) x prog x s + H x dense + events; hazard_nav prog x (1 + H x pbrs drift) + H x dense + events
      (the progress potential is normalised to a range of 1 and its discounted telescoping sum loses at most that
      range, so the speed bound is not needed); station_keeping 1 + H x D / scale; docking H x align + align + brake
      + bonus + H x D / scale; harbor_mission prog + H x dense + events.
P1-c  normaliser freeze at 5 % (G44 R1a, unchanged); checkpoints every 2 % of the budget = the trainer DEFAULT
      (CHECKPOINT_PARTS = 50: BlueBoat 96000 / 50 = 1920, WAM-V 192000 / 50 = 3840 timesteps).
SELECT every periodic checkpoint is evaluated on the VALIDATION seed (VALIDATION_RULE; enforced by
      scripts/ppo_checkpoint_eval.py select): campaign runs (purpose 'training') on the selection namespace block
      selection-02 (seed 11002) only; training-smoke runs on selection-02 or the diagnostic seed 42; a seed of the
      final_test / pilot / calibration namespaces is never used for selection. Best = highest successes, tie ->
      the later checkpoint; the final checkpoint is evaluated and reported too.
INIT  the untrained agent is saved to out/g46_init/agent_0.pt (skrl whole-agent format) for its evaluation.
Pure Python except save_init / LearningRateLog (torch objects passed in).
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path

import ppo_training_fixes as FX

G46_VERSION = "v264-ppo-final-v1"          # v2.6.4 (G47); the G46 dev candidate was "g46-ppo-final-v1"
REMOVED_ENV_VAR = "USVB_G46_LR"             # the G46 smoke switch, removed in v2.6.4 (a set value is refused)
INIT_CHECKPOINT = "g46_init/agent_0.pt"
CHECKPOINT_PARTS = 50
VEHICLES_DATA = Path(__file__).resolve().parents[1] / "tasks" / "_shared" / "vehicles_v2_data.py"
PENALTY_RULE = ("G46 P1-b: a row terminating with a failure code %s gets P = -(H x L + E) (per episode -(A + B x D), "
                "D = layout bounds diagonal) added to its step reward before the agent sees it; H = min(ceil(1/(1 - "
                "gamma)), N) control steps, L = largest loss per control step (progress at the vessel's top speed), E = "
                "what the failure step itself can earn; failing is then never worth more, in the discounted return, "
                "than any continuation that does not fail" % (FX.FAILURE_CODES,))
LR_RULE = ("v2.6.4 P1-a: no learning-rate floor; the registration yaml's KLAdaptiveLR unchanged (no learning-rate "
           "mode switch)")
# v2.6.4 (owner decisions 2026-10-13 23:xx / 10-14 04:xx): the seed every periodic checkpoint is evaluated on
VALIDATION_BLOCKS = {"training": ({"namespace": "selection", "block_id": "selection-02", "seed": 11002},),
                     "training_smoke": ({"namespace": "selection", "block_id": "selection-02", "seed": 11002},
                                        {"namespace": "diagnostic", "block_id": None, "seed": 42})}
FORBIDDEN_SELECTION_PURPOSES = ("final_test", "pilot", "calibration")
VALIDATION_RULE = ("v2.6.4: every periodic checkpoint (every 2 percent of the budget) is evaluated by the protocol "
                   "evaluator (calm, 64 x 1) on the validation seed: campaign runs (purpose training) on the "
                   "selection namespace block selection-02 (seed 11002) only; training-smoke runs on selection-02 "
                   "or the diagnostic seed 42; seeds of the final_test / pilot / calibration namespaces are never "
                   "used for selection; best = highest successes, tie -> the later checkpoint; the final "
                   "checkpoint is evaluated and reported too")


class G46Error(FX.FixError):
    """A G46 change cannot be derived or applied (refuse)."""


# ------------------------------------------------------------------ removed smoke switch
def removed_switch_error(environ=None):
    """None, or the refusal text when the removed G46 smoke switch USVB_G46_LR is set (v2.6.4: KLAdaptiveLR only)."""
    value = (os.environ if environ is None else environ).get(REMOVED_ENV_VAR, "").strip()
    if not value:
        return None
    return ("%s=%r: the G46 learning-rate smoke switch is removed in v2.6.4 (the registration yaml's KLAdaptiveLR is "
            "the only mode); unset it" % (REMOVED_ENV_VAR, value))


def checkpoint_every_error(purpose, budget, every):
    """None, or the refusal text: a campaign run ('training') keeps the default interval budget / CHECKPOINT_PARTS
    (every 2 %; the validation selection is defined over exactly those checkpoints)."""
    want = int(budget) // CHECKPOINT_PARTS
    if purpose == "training" and (int(every) != want or int(budget) % CHECKPOINT_PARTS):
        return ("a campaign run checkpoints every 2 %% of the budget (budget %d / %d = %d timesteps); got %d"
                % (int(budget), CHECKPOINT_PARTS, want, int(every)))
    return None


ASSETS_ENV_VAR = "USVBENCH_ASSETS_V2"


def assets_env_error(environ=None):
    """None, or the refusal text for an EMPTY USVBENCH_ASSETS_V2 (G45/G46 S1 failure: the env cfg then resolves the
    vessel asset relative to the cwd; unset = the cfg default <release>/assets_v2, a non-empty value = a deployed
    copy)."""
    env = os.environ if environ is None else environ
    if ASSETS_ENV_VAR in env and not str(env[ASSETS_ENV_VAR]).strip():
        return ("%s is set but EMPTY: unset it (the env cfg default <release>/assets_v2 applies) or point it at the "
                "assets folder" % ASSETS_ENV_VAR)
    return None


# ------------------------------------------------------------------ discounted penalty
def top_speed_mps(vehicle_profile, data_path=VEHICLES_DATA):
    spec = importlib.util.spec_from_file_location("_g46_vehicles_v2_data", str(data_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, "PINNED_DERIVED", {}).get(vehicle_profile)
    if entry is None or "speeds.top_model_m_s" not in entry:
        raise G46Error("no pinned top speed for vehicle profile %r (vehicles_v2_data.PINNED_DERIVED)" % (vehicle_profile,))
    value = float(entry["speeds.top_model_m_s"])
    if not (value > 0.0 and math.isfinite(value)):
        raise G46Error("top speed of %r must be positive" % (vehicle_profile,))
    return value


def discount_horizon(gamma, n_steps):
    gamma = float(gamma)
    if not (0.0 <= gamma < 1.0):
        raise G46Error("discount_factor must be in [0, 1) (got %r)" % gamma)
    h = int(math.ceil(1.0 / (1.0 - gamma) - 1.0e-9))
    return min(h, int(n_steps)), h


def discounted_terms(carrier, cfg, plan_params, gamma, top_speed):
    """{A, B, N, H, H_uncapped, gamma, control_dt_s, top_speed_mps, step_m, L, E, carrier, parts}: the failure penalty
    of an episode with bounds diagonal D is -(A + B x D) = -(H x L + E) with L, E possibly linear in D."""
    dt = FX.control_dt(cfg)
    n = FX.horizon_steps(cfg)
    h, h_unc = discount_horizon(gamma, n)
    v = float(top_speed)
    if not (v > 0.0 and math.isfinite(v)):
        raise G46Error("top speed must be positive")
    step_m = v * dt
    l_d = 0.0                     # the part of L (per step) proportional to D
    if carrier == "hazard_nav":
        if FX._num(cfg, "reward_threading_amplitude", 0.0) > 0.0:
            raise G46Error("threading bonus enabled: its per-step range is not derived here")
        prog = FX._num(cfg, "reward_progress_scale", 0.0)
        drift = 0.0
        if bool(getattr(cfg, "pbrs_correct", False)) and bool(getattr(cfg, "pbrs_shift_potential", False)):
            drift = 1.0 - FX._num(cfg, "pbrs_gamma", 0.0)       # (gamma_pbrs - 1) Phi', Phi' in [0, 1], per step
        dense = dt * (FX._num(cfg, "reward_clearance_scale", 0.0)
                      + FX._num(cfg, "reward_prox_scale", 0.0) * FX._prox_log(cfg)
                      + FX._num(cfg, "reward_swift_scale", 0.0) + FX._num(cfg, "reward_open_water_scale", 0.0)
                      + FX._num(cfg, "reward_reverse_action_scale", 0.0))
        events = FX._num(cfg, "reward_goal_entry_bonus", 0.0) + FX._num(cfg, "reward_goal_time_bonus", 0.0)
        # the normalised potential: discounted telescoping progress of any continuation >= -prog x range (range 1)
        progress_total = prog * (1.0 + h * drift)
        l_const = dense
        e = events + progress_total      # events + the discounted progress loss bound (range, not per step)
        a = progress_total + h * dense + events
        parts = {"progress_range_total": progress_total, "dense_cost_per_step": dense, "pbrs_drift_per_step": drift,
                 "failure_step_events": events, "speed_used": False}
    elif carrier == "path_following":
        prog = FX._num(cfg, "reference_reward_progress_scale", 0.0)
        gate = FX._num(cfg, "reference_reward_gate_bonus", 0.0)
        succ = FX._num(cfg, "reference_reward_terminal_success", 0.0)
        ls = float(plan_params["length_scale"])
        if not ls > 0.0:
            raise G46Error("length_scale must be positive")
        lp = prog * step_m / ls          # distance to the (fixed within a step) active gate moves <= step_m
        l_const = lp
        e = gate + succ + lp             # events + one step of progress gain
        a = h * lp + e
        parts = {"progress_loss_per_step": lp, "failure_step_events": gate + succ, "failure_step_progress_gain": lp,
                 "length_scale": ls, "speed_used": True}
    elif carrier == "path_hazard":
        prog = FX._num(cfg, "reference_reward_progress_scale", 0.0)
        gate = FX._num(cfg, "reference_reward_gate_bonus", 0.0)
        dense = dt * (FX._num(cfg, "reward_clearance_scale", 0.0)
                      + FX._num(cfg, "reward_prox_scale", 0.0) * FX._prox_log(cfg)
                      + FX._num(cfg, "reward_reverse_action_scale", 0.0) + FX._num(cfg, "reward_swift_scale", 0.0))
        events = gate + FX._num(cfg, "reward_goal_entry_bonus", 0.0) + FX._num(cfg, "reward_goal_time_bonus", 0.0)
        lp = prog * step_m               # progress in metres
        l_const = lp + dense
        e = events + lp
        a = h * l_const + e
        parts = {"progress_loss_per_step": lp, "dense_cost_per_step": dense, "failure_step_events": events,
                 "failure_step_progress_gain": lp, "speed_used": True}
    elif carrier == "station_keeping":
        scale = FX._num(cfg, "reference_reward_distance_scale", 0.0)
        if scale <= 0.0:
            raise G46Error("reference_reward_distance_scale must be positive")
        # r = -d / scale + 1[inside]; d <= D: per step >= -D / scale; the failure step earns at most 1
        l_const, l_d, e = 0.0, 1.0 / scale, 1.0
        a = e
        parts = {"distance_cost_per_metre_per_step": l_d, "failure_step_events": 1.0, "speed_used": False}
    elif carrier == "docking":
        scale = FX._num(cfg, "reference_reward_distance_scale_m", 0.0)
        if scale <= 0.0:
            raise G46Error("reference_reward_distance_scale_m must be positive")
        align = abs(FX._num(cfg, "reference_reward_alignment_scale"))
        brake = abs(FX._num(cfg, "reference_reward_braking_scale"))
        bonus = abs(FX._num(cfg, "reference_reward_success_bonus"))
        l_const, l_d, e = align, 1.0 / scale, align + brake + bonus
        a = h * align + e
        parts = {"distance_cost_per_metre_per_step": l_d, "dense_cost_per_step": align, "failure_step_events": e,
                 "speed_used": False}
    elif carrier == "harbor_mission":
        prog = FX._num(cfg, "reward_progress_scale", 0.0)
        dense = dt * (FX._num(cfg, "reward_clearance_scale", 0.0)
                      + FX._num(cfg, "reward_prox_scale", 0.0) * FX._prox_log(cfg))
        events = (3.0 * FX._num(cfg, "reward_milestone_bonus", 0.0) + FX._num(cfg, "reward_stage_goal_bonus", 0.0)
                  + FX._num(cfg, "reward_stage_time_bonus", 0.0) + abs(FX._num(cfg, "reward_dock_conjunction")))
        l_const = dense
        e = prog + events                # phase potential range 1 (telescoping, as hazard_nav)
        a = prog + h * dense + events
        parts = {"progress_range_total": prog, "dense_cost_per_step": dense, "failure_step_events": events,
                 "speed_used": False}
    else:
        raise G46Error("no discounted failure-penalty derivation for carrier %r" % (carrier,))
    b = h * l_d
    return {"A": float(a), "B": float(b), "N": int(n), "H": int(h), "H_uncapped": int(h_unc), "gamma": float(gamma),
            "control_dt_s": float(dt), "top_speed_mps": v, "step_m": float(step_m), "L_per_step": float(l_const),
            "L_per_step_per_metre_D": float(l_d), "E": float(e), "carrier": carrier, "rule": PENALTY_RULE,
            "parts": {k: (float(x) if isinstance(x, float) else x) for k, x in parts.items()}}


# ------------------------------------------------------------------ plan block / install / evidence
def block(purpose, agent_cfg, vehicle_profile, budget, environ=None):
    """The preflight 'g46' block (v2.6.4 PPO final); refuses the removed USVB_G46_LR switch and an agent yaml whose
    learning-rate scheduler is not KLAdaptiveLR."""
    problem = removed_switch_error(environ)
    if problem:
        raise G46Error(problem)
    agent = agent_cfg["agent"]
    if agent.get("learning_rate_scheduler") != "KLAdaptiveLR":
        raise G46Error("the registration yaml must keep learning_rate_scheduler KLAdaptiveLR (got %r)"
                       % (agent.get("learning_rate_scheduler"),))
    if purpose not in VALIDATION_BLOCKS:
        raise G46Error("no validation rule for purpose %r" % (purpose,))
    gamma = float(agent["discount_factor"])
    return {"version": G46_VERSION, "lr_mode": {"mode": "kl", "lr": None}, "lr_floor": None, "lr_rule": LR_RULE,
            "gamma": gamma, "vehicle_profile": vehicle_profile, "top_speed_mps": top_speed_mps(vehicle_profile),
            "warmup_fraction": FX.WARMUP_FRACTION, "checkpoint_parts": CHECKPOINT_PARTS,
            "recommended_checkpoint_every": int(budget) // CHECKPOINT_PARTS, "init_checkpoint": INIT_CHECKPOINT,
            "penalty_rule": PENALTY_RULE, "validation_rule": VALIDATION_RULE,
            "validation_blocks": [dict(b) for b in VALIDATION_BLOCKS[purpose]]}


class LearningRateLog:
    """Logs the optimizer learning rate after every update (no floor; KLAdaptiveLR only in v2.6.4)."""

    def __init__(self, lr_mode):
        self.lr_mode = dict(lr_mode)
        self.agent = None
        self.lowest = self.highest = self.last = self.first = None
        self.updates = 0
        self.scheduler = None

    def attach(self, agent):
        if getattr(agent, "optimizer", None) is None:
            raise G46Error("the agent has no optimizer")
        sched = getattr(agent, "scheduler", None)
        self.scheduler = None if sched is None else type(sched).__name__
        if self.lr_mode["mode"] != "kl":
            raise G46Error("v2.6.4 trains with the yaml KLAdaptiveLR only (lr mode %r)" % (self.lr_mode,))
        if sched is None or not hasattr(sched, "min_lr"):
            raise G46Error("kl mode requested but the agent has no KLAdaptiveLR scheduler")
        self.agent = agent

    def observe(self):
        lr = float(self.agent.optimizer.param_groups[0]["lr"])
        self.updates += 1
        self.first = lr if self.first is None else self.first
        self.lowest = lr if self.lowest is None else min(self.lowest, lr)
        self.highest = lr if self.highest is None else max(self.highest, lr)
        self.last = lr
        return lr

    def summary(self):
        return {"rule": LR_RULE, "lr_mode": self.lr_mode, "scheduler": self.scheduler, "updates_seen": self.updates,
                "first_lr": self.first, "lowest_lr_seen": self.lowest, "highest_lr_seen": self.highest,
                "last_lr": self.last}


def install(plan, env_cfg, base, agent):
    """The P1 fixes on the skrl agent: discounted failure penalty, normaliser freeze, LR log (no floor)."""
    g44, g46 = plan["g44"], plan["g46"]
    terms = discounted_terms(plan["carrier"], env_cfg, g44["plan_params"], g46["gamma"], g46["top_speed_mps"])
    failure = FX.FailurePenalty(terms, base, plan["carrier"], plan["num_envs"])
    freeze = FX.NormaliserFreeze(g44["warmup_timesteps"])
    freeze.attach(agent)
    lrlog = LearningRateLog(g46["lr_mode"])
    lrlog.attach(agent)
    return {"failure": failure, "freeze": freeze, "floor": lrlog}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_init(agent, out, torch_module):
    """Save every checkpoint module of the untrained agent (skrl write_checkpoint whole-agent format)."""
    path = Path(out) / INIT_CHECKPOINT
    path.parent.mkdir(parents=True, exist_ok=False)
    modules = {}
    for name, module in agent.checkpoint_modules.items():
        modules[name] = module.state_dict() if hasattr(module, "state_dict") else module
    torch_module.save(modules, str(path))
    return {"path": INIT_CHECKPOINT, "sha256": file_sha256(path), "modules": sorted(modules)}


def sidecar(g46, fixes, init):
    """ppo_fix.json of a v2.6.4 run (G44 schema keys kept; the LR floor is recorded as removed)."""
    failure = fixes["failure"].summary()
    failure["rule"] = PENALTY_RULE
    return {"schema": "usvbench.ppo_fix_g44/1", "version": FX.G44_VERSION, "g46_version": G46_VERSION,
            "failure_terminal_penalty": failure, "normaliser_freeze": fixes["freeze"].summary(),
            "learning_rate_floor": {"disabled": True, "removed_by": G46_VERSION},
            "learning_rate_log": fixes["floor"].summary(), "best_checkpoint_rule": FX.BEST_RULE,
            "validation_rule": VALIDATION_RULE, "g46": dict(g46, init_checkpoint=init)}


def effective(fix):
    """Completion check: the freeze held and the KLAdaptiveLR scheduler was attached (LR log)."""
    return (bool(fix["normaliser_freeze"]["unchanged_after_freeze"])
            and fix["learning_rate_log"]["scheduler"] is not None)
