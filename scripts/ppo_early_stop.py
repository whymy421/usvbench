"""Diagnostic branch yutong/ppo-fair-fix: stop a training run that has not finished a single successful episode for
a long stretch (STOP_RULE_BACKTEST.md in usvbench_ppo_fix_runs, 2026-11-16).

Rule: from min_fraction x budget on, checked at every rollout boundary, the run stops when no training episode has
ended with the success termination code within the last window_fraction x budget timesteps. The latest saved
checkpoint then becomes final.pt (status "early_stopped") and is evaluated like a completed run.

Backtest on the 35 finished runs of rounds 1-9: this exact rule (1/3, 1/3, zero successes) stopped 27/35 and saved
43% of the training steps without stopping any run whose final evaluation was > 0; looser rules (window mean below
1%) stopped the only self-learned non-zero run at 34%. Counting every success episode directly is at least as
sensitive as the backtest, which read skrl's success records (a record is non-zero only if some episode in its
interval succeeded).

A resumed run restarts the window at the resume timestep (conservative: it can only stop later). When the
environment publishes no termination codes, the run never stops early (recorded as codes_missing_steps).
"""
import math

DEFAULTS = {"min_fraction": 1.0 / 3.0, "window_fraction": 1.0 / 3.0, "success_code": 1}
RULE = ("early stop: from ceil(min_fraction x budget) on, at every rollout boundary, stop when no training episode "
        "ended with the success code within the last ceil(window_fraction x budget) timesteps; final.pt = the latest "
        "saved checkpoint, status early_stopped, evaluated like a completed run")
SIDECAR = "early_stop.json"


class EarlyStop(Exception):
    """Raised from the step wrapper; run_training ends the run as 'early_stopped'."""

    def __init__(self, block):
        super().__init__(block["reason"])
        self.block = block


class NoSuccessStop:
    def __init__(self, cfg, budget):
        self.cfg = dict(cfg)
        self.budget = int(budget)
        self.success_code = int(cfg["success_code"])
        self.min_step = int(math.ceil(self.budget * float(cfg["min_fraction"])))
        self.window = int(math.ceil(self.budget * float(cfg["window_fraction"])))
        if not (0 < self.min_step <= self.budget and 0 < self.window <= self.budget):
            raise ValueError("early stop: min_fraction and window_fraction must lie in (0, 1]")
        self.window_start = 0
        self.last_success = None
        self.successes = 0
        self.episodes = 0
        self.codes_missing_steps = 0
        self.restarts = []
        self.stopped = None

    def restart(self, timestep):
        """A resumed run: the window starts again at the resume timestep."""
        self.window_start = int(timestep)
        self.restarts.append(int(timestep))

    def observe(self, timestep, dones, codes):
        """timestep = the count AFTER this env step; dones / codes = this step's per-row lists."""
        if not any(dones):
            return
        if codes is None:
            self.codes_missing_steps += 1
            return
        done_codes = [int(c) for c, d in zip(codes, dones) if d]
        self.episodes += len(done_codes)
        n = sum(1 for c in done_codes if c == self.success_code)
        if n:
            self.successes += n
            self.last_success = int(timestep)

    def check(self, timestep):
        """None to continue, else the stop block. Call at rollout boundaries."""
        if self.codes_missing_steps or timestep < self.min_step:
            return None
        anchor = max(self.window_start, self.last_success or 0)
        if timestep - anchor < self.window:
            return None
        self.stopped = {"timestep": int(timestep), "last_success_timestep": self.last_success,
                        "window_start": anchor,
                        "reason": "no successful training episode in timesteps (%d, %d] (window %d = %.4g of the "
                                  "budget %d)" % (anchor, timestep, self.window, self.cfg["window_fraction"],
                                                  self.budget)}
        return dict(self.stopped)

    def summary(self):
        return {"rule": RULE, "config": self.cfg, "budget": self.budget, "min_step": self.min_step,
                "window": self.window, "episodes_seen": self.episodes, "successes_seen": self.successes,
                "last_success_timestep": self.last_success, "codes_missing_steps": self.codes_missing_steps,
                "restarts": list(self.restarts), "stopped": self.stopped}
