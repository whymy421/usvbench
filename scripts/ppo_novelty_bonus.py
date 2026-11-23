"""Diagnostic branch yutong/ppo-fair-fix, round 9 (PROTOCOL_R9.md): an episodic novelty bonus for PPO training.

The first visit of a cell (cell_m x cell_m, on a grid of +-half_extent_m around the env origin) within an episode
pays `bonus`, at most `cap_cells` paid cells per episode. Training reward only: the position never enters the
observation, and the evaluator does not use this module.

The env auto-resets inside step(): on a done row the position read after the step is the NEW episode's start, so
that row's grid is cleared, its start cell is marked and nothing is paid for it.
"""

DEFAULTS = {"cell_m": 2.0, "bonus": 0.25, "cap_cells": 80, "half_extent_m": 64.0}
RULE = ("round 9 novelty: +bonus for the first visit of a cell_m x cell_m cell (grid +-half_extent_m around the "
        "env origin, positions outside are clamped to the edge cells) within an episode, at most cap_cells paid "
        "cells per episode; training reward only, never observed")


class NoveltyBonus:
    def __init__(self, cfg, base, num_envs, torch):
        self.cfg = dict(cfg)
        self.base, self.n, self.torch = base, int(num_envs), torch
        self.cell = float(cfg["cell_m"])
        self.bonus = float(cfg["bonus"])
        self.cap = int(cfg["cap_cells"])
        self.half = float(cfg["half_extent_m"])
        self.g = int(round(2.0 * self.half / self.cell))
        if not (self.cell > 0 and self.bonus > 0 and self.cap > 0 and self.g > 0):
            raise ValueError("novelty bonus: cell_m, bonus, cap_cells and half_extent_m must be positive")
        self.device = None
        self.visited = None
        self.paid = None
        self.started = False
        self.rows = None
        # evidence
        self.steps = 0
        self.outside_steps = 0
        self.total_paid = 0.0
        self.episodes = 0
        self.cells_sum = 0
        self.cells_max = 0
        self.capped_episodes = 0
        self.xy_min = None
        self.xy_max = None

    def _local_xy(self):
        return self.base._com_xy() - self.base.scene.env_origins[:, :2]

    def _cells(self):
        torch = self.torch
        xy = self._local_xy().detach()
        if self.device is None:
            self.device = xy.device
            self.visited = torch.zeros(self.n, self.g * self.g, dtype=torch.bool, device=self.device)
            self.paid = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self.rows = torch.arange(self.n, device=self.device)
        lo = xy.min(dim=0).values.tolist()
        hi = xy.max(dim=0).values.tolist()
        self.xy_min = lo if self.xy_min is None else [min(a, b) for a, b in zip(self.xy_min, lo)]
        self.xy_max = hi if self.xy_max is None else [max(a, b) for a, b in zip(self.xy_max, hi)]
        ij = torch.floor((xy + self.half) / self.cell).long()
        outside = ((ij < 0) | (ij >= self.g)).any(dim=-1)
        self.outside_steps += int(outside.sum())
        ij = ij.clamp(0, self.g - 1)
        return ij[:, 0] * self.g + ij[:, 1]

    def step(self, dones):
        """dones: list of bools (this step's finished rows). Returns the per-row bonus as a float tensor."""
        torch = self.torch
        idx = self._cells()
        done = torch.tensor([bool(d) for d in dones], dtype=torch.bool, device=self.device)
        if not self.started:                                # mark the first episodes' start cells, pay nothing
            self.visited[self.rows, idx] = True
            self.started = True
            done = torch.ones_like(done)
        elif bool(done.any()):
            finished = self.paid[done]
            self.episodes += int(finished.numel())
            self.cells_sum += int(finished.sum())
            self.cells_max = max(self.cells_max, int(finished.max()))
            self.capped_episodes += int((finished >= self.cap).sum())
            self.visited[done] = False
            self.paid[done] = 0
        new = ~self.visited[self.rows, idx]
        self.visited[self.rows, idx] = True
        pay = new & ~done & (self.paid < self.cap)
        self.paid += pay.long()
        bonus = pay.float() * self.bonus
        self.steps += 1
        self.total_paid += float(bonus.sum())
        return bonus

    def summary(self):
        return {"rule": RULE, "config": self.cfg, "grid_cells_per_side": self.g, "steps": self.steps,
                "env_steps_outside_grid": self.outside_steps, "total_bonus_paid": round(self.total_paid, 3),
                "episodes_finished": self.episodes,
                "mean_paid_cells_per_episode": (self.cells_sum / self.episodes) if self.episodes else None,
                "max_paid_cells_per_episode": self.cells_max, "episodes_at_cap": self.capped_episodes,
                "local_xy_min_m": self.xy_min, "local_xy_max_m": self.xy_max,
                "note": "counts since the last (re)start of the training process"}
